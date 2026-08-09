"""wasi-testsuite adapter for the in-tree Zig-based wamr CLI.

Mirrors `wasm-micro-runtime.py` from upstream wasi-testsuite but reads the
runtime path from the `WAMR` environment variable (set by
`zig build wasi-testsuite`) instead of `IWASM`. Lives outside the vendored
submodule (`tests/wasi-testsuite/`) so this repo can edit it without a
fork. Keep the contract aligned with upstream — any drift will silently
break the runner.

The wamr CLI uses a Wasmtime-shaped subcommand layout
(`wamr run <file.wasm>`, `wamr version`), so the adapter inserts `run` /
`version` after the binary path.

Since #680, the `wamr` runtime is AOT-only and no longer embeds the
compiler. This adapter therefore pre-compiles every `.wasm` fixture with
`wamrc compile -o <sibling>.cwasm` (cached by mtime) and then hands the
resulting `.cwasm` to `wamr run`. Set `WAMRC` to point at the freshly-built
`wamrc` binary; the adapter falls back to a `wamrc` sibling of `WAMR` and
then to `PATH`.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WAMR = shlex.split(os.getenv("WAMR", "wamr"))
_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_COMPILE_TIMEOUT_SECONDS = 600.0


def _resolve_wamrc() -> List[str]:
    """Locate the `wamrc` binary. Preference order: $WAMRC, sibling of
    $WAMR, then `wamrc` on $PATH. Mirrors `findWamrBinary` in
    `src/compiler/main.zig` (just in reverse — there wamrc finds wamr).
    """
    env = os.getenv("WAMRC")
    if env:
        return shlex.split(env)
    if WAMR:
        sibling = Path(WAMR[0]).resolve().with_name(
            "wamrc.exe" if os.name == "nt" else "wamrc"
        )
        if sibling.exists():
            return [str(sibling)]
    return ["wamrc"]


WAMRC = _resolve_wamrc()
_SAMPLE_LAUNCHER = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "wasi_p3_sample_launch.py"
)


def _emit_timing(
    fixture: Path,
    phase: str,
    artifact_kind: str,
    duration_ns: int,
    cache: str,
) -> None:
    """Append one opt-in timing event without touching guest streams."""
    output = os.getenv("WAMR_PROFILE_TIMINGS")
    if not output:
        return
    event = {
        "schema_version": 1,
        "event": "phase_timing",
        "run_id": os.getenv("WAMR_PROFILE_RUN_ID", ""),
        "mode": os.getenv(
            "WAMR_PROFILE_MODE",
            "jit" if os.getenv("WAMR_JIT_TESTSUITE") else "aot",
        ),
        "fixture": fixture.stem,
        "phase": phase,
        "artifact_kind": artifact_kind,
        "cache": cache,
        "duration_ns": duration_ns,
        "pid": os.getpid(),
    }
    encoded = (json.dumps(event, sort_keys=True) + "\n").encode("UTF-8")
    fd = os.open(output, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)


def _profile_requested(fixture: Path, phase: str) -> bool:
    selection_path = os.getenv("WAMR_PROFILE_SELECTION")
    if not selection_path:
        return False
    try:
        selection = json.loads(Path(selection_path).read_text(encoding="UTF-8"))
    except (OSError, json.JSONDecodeError):
        return False
    mode = os.getenv("WAMR_PROFILE_MODE", "aot")
    return any(
        item.get("mode") == mode
        and item.get("fixture") == fixture.stem
        and item.get("phase") == phase
        for item in selection.get("profiles", [])
    )


def _profile_command(cmd: List[str], fixture: Path, phase: str) -> List[str]:
    profile_dir = os.getenv("WAMR_PROFILE_OUTPUT_DIR")
    if not profile_dir or not _profile_requested(fixture, phase):
        return cmd

    output_dir = Path(profile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = os.getenv("WAMR_PROFILE_MODE", "aot")
    profile = output_dir / f"{fixture.stem}-{phase}-{mode}.sample.txt"
    log = output_dir / f"{fixture.stem}-{phase}-{mode}.sample.log"
    return [
        sys.executable,
        str(_SAMPLE_LAUNCHER),
        "--output",
        str(profile),
        "--log",
        str(log),
        "--",
        *cmd,
    ]


def get_compile_timeout_seconds() -> float:
    """Wall-clock bound for a single `wamrc` invocation.

    Precompilation runs before the guest process exists, so it is not
    covered by `WAMR_TESTSUITE_TIMEOUT` (which only bounds the guest
    wait). An unbounded compile turns a codegen hang into a silent
    multi-hour stall that outlives the CI job timeout and destroys the
    diagnostics with it. Override with `WAMR_COMPILE_TIMEOUT`. (#616 D3.)
    """
    raw = os.getenv("WAMR_COMPILE_TIMEOUT")
    if not raw:
        return _DEFAULT_COMPILE_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 0
    if timeout <= 0:
        print(
            f"warning: ignoring invalid WAMR_COMPILE_TIMEOUT={raw!r}; "
            f"falling back to {_DEFAULT_COMPILE_TIMEOUT_SECONDS}s",
            file=sys.stderr,
        )
        return _DEFAULT_COMPILE_TIMEOUT_SECONDS
    return timeout


def _run_compile(cmd: List[str], fixture: Path, phase: str) -> None:
    """Run wamrc, optionally attaching macOS's sampling profiler."""
    timeout = get_compile_timeout_seconds()
    try:
        subprocess.run(
            _profile_command(cmd, fixture, phase),
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"wamrc {phase} for {fixture.name} exceeded "
            f"{timeout:g}s (WAMR_COMPILE_TIMEOUT)"
        ) from exc


def get_name() -> str:
    return "wamr-zig"


def get_version() -> str:
    result = subprocess.run(
        WAMR + ["version"],
        encoding="UTF-8",
        capture_output=True,
        check=True,
    )
    output = result.stdout.splitlines()[0].split(" ")
    return output[1]


def get_wasi_versions() -> List[str]:
    # Declares both Preview 1 (covered by `zig build wasi-testsuite`) and
    # Preview 3 (covered by `zig build wasi-p3-testsuite`, issue #489) so
    # the upstream `UnsupportedWasiTestExcludeFilter` doesn't auto-skip
    # the wasm32-wasip3 fixtures.
    return ["wasm32-wasip1", "wasm32-wasip3"]


def get_wasi_worlds() -> List[str]:
    # Declares both `wasi:cli/command` (every cli/filesystem/sockets
    # fixture) and `wasi:http/service` (the `http-service` fixture,
    # which exports `wasi:http/incoming-handler@0.3.0.handle`). Without
    # the http world declaration the upstream
    # `UnsupportedWasiTestExcludeFilter` auto-skips http-service
    # before any of our dispatch glue gets a chance to run. (#570)
    return ["wasi:cli/command", "wasi:http/service"]


def get_timeout_seconds() -> float:
    raw = os.getenv("WAMR_TESTSUITE_TIMEOUT")
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 0
    if timeout <= 0:
        print(
            f"warning: ignoring invalid WAMR_TESTSUITE_TIMEOUT={raw!r}; "
            f"falling back to {_DEFAULT_TIMEOUT_SECONDS}s",
            file=sys.stderr,
        )
        return _DEFAULT_TIMEOUT_SECONDS
    return timeout


def _isolate_root(root: Optional[Path]) -> Optional[Path]:
    """Snapshot the preopened root into a fresh tempdir so a
    filesystem test that mutates the mapped directory doesn't pollute
    state for subsequent tests in the run (the upstream runner reuses
    the same host path across invocations). The tempdirs are leaked
    intentionally; they are tiny and the per-suite TMPDIR is cleared
    between CI invocations. (#564.)
    """
    if root is None:
        return None
    root_path = Path(root)
    if not root_path.is_dir():
        return root_path
    snapshot = Path(tempfile.mkdtemp(prefix="wamr-zig-fs-"))
    shutil.copytree(root_path, snapshot, dirs_exist_ok=True, symlinks=True)
    return snapshot


def _is_component(path: Path) -> bool:
    """Read the 8-byte wasm prefix and distinguish a component from a
    core module. Both start with `\\0asm`; core's version word is
    `0x0000_0001`, component's is `0x0001_000d`.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return False
    if len(head) < 8 or head[:4] != b"\x00asm":
        return False
    return head[4:8] == b"\x0d\x00\x01\x00"


def _precompile(test_path: str) -> str:
    """Compile `<test_path>` to a sibling AOT artifact, skipping
    recompile if the artifact exists and is newer than the source.
    Returns the path `wamr run` should be invoked with — the sibling
    `.cwasm` for core wasm, or the source `.wasm` itself for a
    component (since `wamr run` auto-probes the sibling
    `<stem>.cwasm.json` manifest). Inputs that don't end in `.wasm`
    pass through unchanged.

    Passes `--no-verify-ir` to match the verifier setting the
    in-process compiler used pre-#680 (`compileCoreWasm` defaults to
    `.verify_mode = .off`). Verifier failures on production wasm
    binaries are tracked separately (#662) and already gated through
    `tests/wasi-testsuite-expectations.toml`; running the verifier here
    surfaces those same bugs as adapter failures and breaks suites
    that previously hit the silent codegen path.

    #856: when `WAMR_JIT_TESTSUITE` is set, skip precompilation
    entirely and hand the raw `.wasm` straight to `wamr run` /
    `wamr serve`, which JIT-compiles it in memory on a `-Djit=true`
    build (see `zig build wasi-testsuite-jit` /
    `wasi-p3-testsuite-jit` in build.zig). This proves the in-process
    JIT path is behavior-identical to the AOT-precompiled path this
    function normally produces, since both flow through the exact
    same compiler and AOT loader/runtime — only the "when" differs.
    """
    if os.getenv("WAMR_JIT_TESTSUITE"):
        if os.getenv("WAMR_PROFILE_TIMINGS"):
            p = Path(test_path)
            artifact_kind = "component" if _is_component(p) else "core"
            _emit_timing(
                p,
                f"{artifact_kind}_precompile",
                artifact_kind,
                0,
                "bypass",
            )
        return test_path
    p = Path(test_path)
    if p.suffix != ".wasm":
        return test_path
    started_ns = time.perf_counter_ns()
    is_component = _is_component(p)
    artifact_kind = "component" if is_component else "core"
    phase = f"{artifact_kind}_precompile"
    if is_component:
        manifest = p.with_suffix(".cwasm.json")
        if not manifest.exists() or manifest.stat().st_mtime < p.stat().st_mtime:
            # `wamrc compile-component` already disables the IR
            # verifier internally (compileCoreWasm hard-codes
            # verify_mode=.off), so no extra flag is needed here.
            _run_compile(
                WAMRC + ["compile-component", "-o", str(manifest), str(p)],
                p,
                phase,
            )
            _emit_timing(
                p, phase, artifact_kind, time.perf_counter_ns() - started_ns, "miss"
            )
        else:
            _emit_timing(
                p, phase, artifact_kind, time.perf_counter_ns() - started_ns, "hit"
            )
        return test_path
    cwasm = p.with_suffix(".cwasm")
    if cwasm.exists() and cwasm.stat().st_mtime >= p.stat().st_mtime:
        _emit_timing(
            p, phase, artifact_kind, time.perf_counter_ns() - started_ns, "hit"
        )
        return str(cwasm)
    _run_compile(
        WAMRC + ["compile", "--no-verify-ir", "-o", str(cwasm), str(p)],
        p,
        phase,
    )
    _emit_timing(
        p,
        phase,
        artifact_kind,
        time.perf_counter_ns() - started_ns,
        "miss",
    )
    return str(cwasm)


def compute_argv(
    test_path: str,
    args_env_root: Tuple[List[str], Dict[str, str], Optional[Path]],
    proposals: List[str],
    wasi_world: str,
    wasi_version: str,
) -> List[str]:
    argv: List[str] = []
    argv += WAMR
    args, env, root = args_env_root

    # `wasi:http/service` fixtures (`http-service.wasm`) export
    # `wasi:http/incoming-handler@0.3.0.handle` and are served via the
    # `serve` subcommand (#845); every other fixture runs as a
    # `wasi:cli/run` command component (or core module) under `run`.
    is_http_service = wasi_world == "wasi:http/service"
    argv += ["serve"] if is_http_service else ["run"]

    for k, v in env.items():
        argv += ["--env", f"{k}={v}"]

    # `--map-dir` / `--allow-net` are `run`-only host-config flags (the
    # HTTP serve path does not wire filesystem / sockets preopens), so
    # only emit them for the `run` verb.
    if not is_http_service:
        isolated_root = _isolate_root(root)
        if isolated_root:
            argv += ["--map-dir", f"{isolated_root}::/"]

        # wasi:sockets fixtures need an explicit allow-list to escape the
        # adapter's default deny-all posture. Localhost is sufficient for
        # every wasi-testsuite sockets fixture (they all bind/connect to
        # 127.0.0.1 / ::1). (#520 wave 2)
        if "sockets" in proposals:
            argv += ["--allow-net", "127.0.0.0/8"]
            argv += ["--allow-net", "::1/128"]

    # The host binds a TCP listener, accepts on it, and routes incoming
    # HTTP over the guest export. `--addr 127.0.0.1:0` selects an
    # ephemeral bind and triggers `announce_listening` — the wamr CLI
    # prints `http://<host>:<port>` so the wasi-testsuite
    # `TestCaseRunner.get_http_server` URL-scrape succeeds. (#570, #845)
    if is_http_service:
        argv += ["--addr", "127.0.0.1:0"]

    # Pre-compile the wasm fixture to its sibling AOT artifact: for
    # core wasm we get a `.cwasm`; for components we write a
    # `<stem>.cwasm.json` manifest + per-core `.cwasm` files that
    # `wamr run` / `wamr serve` auto-discover via sibling probing. (#680)
    argv += [_precompile(test_path)]
    argv += args
    return _profile_command(argv, Path(test_path), "fixture_execution")
