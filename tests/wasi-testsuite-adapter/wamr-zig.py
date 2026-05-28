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

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


WAMR = shlex.split(os.getenv("WAMR", "wamr"))


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
    # the wasm32-wasip3 fixtures. Per-test gating for incomplete P3
    # adapter coverage lives in `tests/wasi-p3-testsuite-skip.json`.
    return ["wasm32-wasip1", "wasm32-wasip3"]


def get_wasi_worlds() -> List[str]:
    # Declares both `wasi:cli/command` (every cli/filesystem/sockets
    # fixture) and `wasi:http/service` (the `http-service` fixture,
    # which exports `wasi:http/incoming-handler@0.3.0.handle`). Without
    # the http world declaration the upstream
    # `UnsupportedWasiTestExcludeFilter` auto-skips http-service
    # before any of our dispatch glue gets a chance to run. (#570)
    return ["wasi:cli/command", "wasi:http/service"]


def _isolate_preopens(dirs: List[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
    """Snapshot each preopen host directory into a fresh tempdir so a
    filesystem test that mutates the mapped directory doesn't pollute
    state for subsequent tests in the run (the upstream runner reuses
    the same host paths across invocations). The tempdirs are leaked
    intentionally — they're tiny and the per-suite TMPDIR is cleared
    between CI invocations. (#564.)
    """
    isolated: List[Tuple[Path, str]] = []
    for host, guest in dirs:
        host_path = Path(host)
        if not host_path.is_dir():
            isolated.append((host, guest))
            continue
        snapshot = Path(tempfile.mkdtemp(prefix="wamr-zig-fs-"))
        # `copytree(..., dirs_exist_ok=True)` lets us land into the
        # just-created mkdtemp root rather than under a child dir.
        shutil.copytree(host_path, snapshot, dirs_exist_ok=True, symlinks=True)
        isolated.append((snapshot, guest))
    return isolated


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
    `tests/wasi-testsuite-skip.json`; running the verifier here
    surfaces those same bugs as adapter failures and breaks suites
    that previously hit the silent codegen path.
    """
    p = Path(test_path)
    if p.suffix != ".wasm":
        return test_path
    if _is_component(p):
        manifest = p.with_suffix(".cwasm.json")
        if not manifest.exists() or manifest.stat().st_mtime < p.stat().st_mtime:
            # `wamrc compile-component` already disables the IR
            # verifier internally (compileCoreWasm hard-codes
            # verify_mode=.off), so no extra flag is needed here.
            subprocess.run(
                WAMRC + ["compile-component", "-o", str(manifest), str(p)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        return test_path
    cwasm = p.with_suffix(".cwasm")
    if cwasm.exists() and cwasm.stat().st_mtime >= p.stat().st_mtime:
        return str(cwasm)
    subprocess.run(
        WAMRC + ["compile", "--no-verify-ir", "-o", str(cwasm), str(p)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return str(cwasm)


def compute_argv(
    test_path: str,
    args_env_dirs: Tuple[List[str], Dict[str, str], List[Tuple[Path, str]]],
    proposals: List[str],
    wasi_world: str,
    wasi_version: str,
) -> List[str]:
    argv: List[str] = []
    argv += WAMR
    argv += ["run"]
    args, env, dirs = args_env_dirs

    for k, v in env.items():
        argv += ["--env", f"{k}={v}"]

    for host, guest in _isolate_preopens(dirs):
        argv += ["--map-dir", f"{host}::{guest}"]

    # wasi:sockets fixtures need an explicit allow-list to escape the
    # adapter's default deny-all posture. Localhost is sufficient for
    # every wasi-testsuite sockets fixture (they all bind/connect to
    # 127.0.0.1 / ::1). (#520 wave 2)
    if "sockets" in proposals:
        argv += ["--allow-net", "127.0.0.0/8"]
        argv += ["--allow-net", "::1/128"]

    # `wasi:http/service` fixtures (`http-service.wasm`) export
    # `wasi:http/incoming-handler@0.3.0.handle` and expect the host
    # to bind a TCP listener, accept on it, and route incoming HTTP
    # over the guest export. Bare `--listen` selects an ephemeral
    # 127.0.0.1:0 bind and triggers `announce_listening` — the wamr
    # CLI prints `http://<host>:<port>` to stderr so the
    # wasi-testsuite `TestCaseRunner.get_http_server` URL-scrape
    # succeeds. (#570)
    if wasi_world == "wasi:http/service":
        argv += ["--listen"]

    # Pre-compile the wasm fixture to its sibling AOT artifact: for
    # core wasm we get a `.cwasm`; for components we write a
    # `<stem>.cwasm.json` manifest + per-core `.cwasm` files that
    # `wamr run` auto-discovers via sibling probing. (#680)
    argv += [_precompile(test_path)]
    argv += args
    return argv
