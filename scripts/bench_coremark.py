#!/usr/bin/env python3
"""Run a reproducible CoreMark comparison across WAMR refs and Wasmtime.

The default ``authoritative`` profile discards two warmups and records ten
measured runs.  The tracked CoreMark wasm fixture is used directly, so results
do not depend on whichever EEMBC/CoreMark commit happens to be upstream.

Usage
-----
    scripts/bench_coremark.py --baseline origin/main --target HEAD
    scripts/bench_coremark.py --wasmtime-baseline auto --wasmtime /path/to/wasmtime
    scripts/bench_coremark.py --profile ci --min-delta-pct=-5
    scripts/bench_coremark.py --optimize both
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bench_optimize import OPTIMIZE_CHOICES, fmt_ratio, optimize_slug, parse_optimize_modes

ITER_PATTERN = re.compile(r"Iterations/Sec\s*:\s*([0-9]+(?:\.[0-9]+)?)")
VALIDATION_TEXT = "Correct operation validated."
PERFORMANCE_MARKER = "2K performance run parameters for coremark."
ITERATIONS_PATTERN = re.compile(r"^\s*Iterations\s*:\s*(\d+)\s*$")
THROUGHPUT_PATTERN = re.compile(
    r"^\s*Iterations/Sec\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$"
)
COREMARK_GUEST_ARGS = ("0", "0", "0", "400000", "0")
EXPECTED_ITERATIONS = 400000
PROFILE_ITERATIONS = {
    "authoritative": EXPECTED_ITERATIONS,
    "ci": 200000,
}
DEFAULT_FIXTURE = Path("tests/benchmarks/coremark/coremark_wasi.wasm")
DEFAULT_FIXTURE_SHA256 = "f4b7591296ead10264e0f101f355bdf848865c31329325594e66fbabefec235b"
PINNED_WASMTIME_VERSION = "44.0.1"
PROFILE_COUNTS = {
    "authoritative": (2, 10),
    "ci": (0, 3),
}
PINNED_WASMTIME_ASSETS = {
    ("linux", "x86_64"): (
        "wasmtime-v44.0.1-x86_64-linux.tar.xz",
        "afd58715f105e3a7f454169daed22168c5736ec5f225fb04c4ac62c54c9508a3",
        "47e59a672f6a35ad071e3fbe7c0c541dcaea77b085d7aada4fd1a9ac77a0d601",
    ),
    ("linux", "aarch64"): (
        "wasmtime-v44.0.1-aarch64-linux.tar.xz",
        "9ec1e606c541099a7e8266865494e190a6f928abb3a2fb7bb4195fd2e8499d2b",
        "ad27d96d24d193ff32665a59552699e3f5355baeaf7220fb4cd88f023694e93d",
    ),
}

# Signature of the known x86_64 native AOT-run flake from issue #406.
_TRAP_FLAKE_PATTERN = re.compile(
    r'wasm trap: out of bounds memory access.*local_func\[-?\d+\](?:\s+"[^"]*")?\+0x0',
    re.IGNORECASE,
)
_TRAP_RETRY_MAX = 5


@dataclass(frozen=True)
class EngineResult:
    engine: str
    version: str
    optimize: str
    values: list[float]
    samples: list["SampleRecord"] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedCoreMark:
    throughput: float
    iterations: int


@dataclass(frozen=True)
class SampleRecord:
    engine_key: str
    engine: str
    phase: str
    engine_ordinal: int
    schedule_position: int
    started_at: str
    completed_at: str
    elapsed_seconds: float
    iterations_per_second: float
    iterations: int


@dataclass(frozen=True)
class PreparedEngine:
    key: str
    engine: str
    version: str
    optimize: str
    cmd: list[str]
    cwd: Path
    env: dict
    expected_iterations: int
    retry_wamr_flake: bool = False


@dataclass(frozen=True)
class AffinityInfo:
    allowed_cpus: tuple[int, ...]
    selected_cpu: int
    taskset: str


@dataclass(frozen=True)
class HostIdentity:
    arch: str
    cpu_count: int
    cpu_model: str
    runner_name: str
    boot_id: str

    def fingerprint(self) -> str:
        encoded = "\0".join(
            (
                self.arch,
                str(self.cpu_count),
                self.cpu_model,
                self.runner_name,
                self.boot_id,
            )
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(
            f"\n[harness] command failed (exit {exc.returncode}): "
            f"{' '.join(str(c) for c in cmd)}\n"
        )
        if cwd is not None:
            sys.stderr.write(f"[harness]   cwd: {cwd}\n")
        if exc.stdout:
            sys.stderr.write(f"[harness] --- stdout ---\n{exc.stdout}")
            if not exc.stdout.endswith("\n"):
                sys.stderr.write("\n")
        if exc.stderr:
            sys.stderr.write(f"[harness] --- stderr ---\n{exc.stderr}")
            if not exc.stderr.endswith("\n"):
                sys.stderr.write("\n")
        sys.stderr.flush()
        raise
    return proc.stdout + proc.stderr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_fixture(repo: Path, fixture_arg: Path) -> tuple[Path, str]:
    fixture = fixture_arg if fixture_arg.is_absolute() else repo / fixture_arg
    fixture = fixture.resolve()
    if not fixture.is_file():
        raise FileNotFoundError(f"CoreMark fixture not found: {fixture}")
    digest = sha256_file(fixture)
    default_fixture = (repo / DEFAULT_FIXTURE).resolve()
    if fixture == default_fixture and digest != DEFAULT_FIXTURE_SHA256:
        raise RuntimeError(
            "tracked CoreMark fixture checksum mismatch: "
            f"expected {DEFAULT_FIXTURE_SHA256}, got {digest}"
        )
    return fixture, digest


def parse_coremark_output(
    output: str,
    engine: str,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> ParsedCoreMark:
    if (
        output.count(VALIDATION_TEXT) != 1
        or re.search(r"(?m)^.*ERROR!", output)
    ):
        raise RuntimeError(
            f"{engine} did not produce a CRC-validated CoreMark result:\n{output}"
        )
    if output.count(PERFORMANCE_MARKER) != 1:
        raise RuntimeError(
            f"{engine} did not produce exactly one {PERFORMANCE_MARKER!r} marker"
        )

    throughput_lines = [
        line for line in output.splitlines() if "Iterations/Sec" in line
    ]
    if len(throughput_lines) != 1:
        raise RuntimeError(
            f"{engine} produced {len(throughput_lines)} Iterations/Sec fields; "
            f"expected exactly one"
        )
    throughput_match = THROUGHPUT_PATTERN.fullmatch(throughput_lines[0])
    if throughput_match is None:
        raise RuntimeError(
            f"{engine} produced malformed Iterations/Sec output: "
            f"{throughput_lines[0]!r}"
        )

    iteration_lines = [
        line
        for line in output.splitlines()
        if re.match(r"^\s*Iterations\s*:", line)
    ]
    if len(iteration_lines) != 1:
        raise RuntimeError(
            f"{engine} produced {len(iteration_lines)} Iterations fields; "
            f"expected exactly one"
        )
    iteration_match = ITERATIONS_PATTERN.fullmatch(iteration_lines[0])
    if iteration_match is None:
        raise RuntimeError(
            f"{engine} produced malformed Iterations output: "
            f"{iteration_lines[0]!r}"
        )
    iterations = int(iteration_match.group(1))
    if iterations != expected_iterations:
        raise RuntimeError(
            f"{engine} ran {iterations} iterations; expected fixed "
            f"{expected_iterations} (self-calibration is forbidden)"
        )
    return ParsedCoreMark(float(throughput_match.group(1)), iterations)


def select_cpu_affinity() -> AffinityInfo:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("authoritative CPU affinity requires sched_getaffinity")
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    if not allowed:
        raise RuntimeError("authoritative CPU affinity set is empty")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("authoritative CPU affinity requires taskset")
    selected = allowed[0]
    verify_script = (
        "import os,sys; "
        f"expected={selected}; actual=sorted(os.sched_getaffinity(0)); "
        "print(','.join(str(cpu) for cpu in actual)); "
        "sys.exit(0 if actual == [expected] else 1)"
    )
    output = run(
        [
            taskset,
            "--cpu-list",
            str(selected),
            sys.executable,
            "-c",
            verify_script,
        ]
    ).strip()
    if output != str(selected):
        raise RuntimeError(
            f"taskset affinity verification returned {output!r}; "
            f"expected {selected}"
        )
    return AffinityInfo(allowed, selected, taskset)


def apply_affinity(cmd: list[str], affinity: AffinityInfo | None) -> list[str]:
    if affinity is None:
        return cmd
    return [
        affinity.taskset,
        "--cpu-list",
        str(affinity.selected_cpu),
        *cmd,
    ]


def counterbalanced_order(keys: list[str], count_per_engine: int) -> list[str]:
    if count_per_engine < 0:
        raise ValueError("count_per_engine must be non-negative")
    if not keys and count_per_engine:
        raise ValueError("cannot schedule samples without engines")
    order: list[str] = []
    for repetition in range(count_per_engine):
        sequence = keys if repetition % 2 == 0 else list(reversed(keys))
        order.extend(sequence)
    return order


def coremark_guest_args(iterations: int) -> tuple[str, ...]:
    if iterations <= 0:
        raise ValueError("fixed CoreMark iterations must be positive")
    return ("0", "0", "0", str(iterations), "0")


def resolve_ref_sha(repo: Path, ref: str) -> str:
    return run(["git", "rev-parse", ref], cwd=repo).strip()


def make_worktree(repo: Path, ref: str, root: Path, label: str) -> tuple[Path, str]:
    sha = resolve_ref_sha(repo, ref)
    wt = root / f"{label}-{sha[:12]}"
    run(["git", "worktree", "add", "--detach", str(wt), sha], cwd=repo)
    return wt, sha


def worktree_env(wt: Path) -> dict:
    env = os.environ.copy()
    env.pop("ZIG_LOCAL_CACHE_DIR", None)
    global_cache = wt / ".zig-global-cache"
    global_cache.mkdir(exist_ok=True)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    return env


def _run_wamr_with_retry(
    cmd: list[str],
    cwd: Path,
    env: dict,
    sample_idx: int,
    samples_total: int,
) -> str:
    last_exc = None
    for attempt in range(1 + _TRAP_RETRY_MAX):
        try:
            return run(cmd, cwd=cwd, env=env)
        except subprocess.CalledProcessError as exc:
            failure_output = (exc.stdout or "") + (exc.stderr or "")
            if not _TRAP_FLAKE_PATTERN.search(failure_output):
                raise
            if attempt == _TRAP_RETRY_MAX:
                last_exc = exc
                break
            print(
                f"[harness]   sample {sample_idx + 1}/{samples_total}: x86_64 "
                f"AOT trap flake (issue #406) — retry "
                f"{attempt + 1}/{_TRAP_RETRY_MAX}",
                file=sys.stderr,
            )
    assert last_exc is not None
    raise last_exc


def measure_command(
    *,
    engine: str,
    cmd: list[str],
    cwd: Path,
    env: dict,
    warmups: int,
    runs: int,
    retry_wamr_flake: bool = False,
    affinity: AffinityInfo | None = None,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> list[float]:
    values: list[float] = []
    total = warmups + runs
    cmd = apply_affinity(cmd, affinity)
    for i in range(total):
        if retry_wamr_flake:
            output = _run_wamr_with_retry(cmd, cwd, env, i, total)
        else:
            output = run(cmd, cwd=cwd, env=env)
        parsed = parse_coremark_output(output, engine, expected_iterations)
        value = parsed.throughput
        if i < warmups:
            print(
                f"[harness]   {engine} warmup {i + 1}/{warmups}: "
                f"{value:.1f} iter/s (discarded)",
                file=sys.stderr,
            )
        else:
            measured_idx = i - warmups + 1
            print(
                f"[harness]   {engine} run {measured_idx}/{runs}: "
                f"{value:.1f} iter/s",
                file=sys.stderr,
            )
            values.append(value)
    return values


def measure_prepared_engines(
    engines: list[PreparedEngine],
    *,
    warmups: int,
    runs: int,
    affinity: AffinityInfo | None,
) -> tuple[dict[str, EngineResult], list[SampleRecord]]:
    if len({engine.key for engine in engines}) != len(engines):
        raise RuntimeError("prepared engine keys must be unique")
    by_key = {engine.key: engine for engine in engines}
    records: list[SampleRecord] = []
    engine_records: dict[str, list[SampleRecord]] = {
        engine.key: [] for engine in engines
    }
    ordinals = {
        phase: {engine.key: 0 for engine in engines}
        for phase in ("warmup", "measured")
    }
    schedule_position = 0
    keys = [engine.key for engine in engines]

    for phase, count in (("warmup", warmups), ("measured", runs)):
        for key in counterbalanced_order(keys, count):
            engine = by_key[key]
            schedule_position += 1
            ordinals[phase][key] += 1
            ordinal = ordinals[phase][key]
            cmd = apply_affinity(engine.cmd, affinity)
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            if engine.retry_wamr_flake:
                output = _run_wamr_with_retry(
                    cmd,
                    engine.cwd,
                    engine.env,
                    ordinal - 1,
                    count,
                )
            else:
                output = run(cmd, cwd=engine.cwd, env=engine.env)
            elapsed = time.monotonic() - started
            completed_at = datetime.now(timezone.utc).isoformat()
            parsed = parse_coremark_output(
                output, engine.engine, engine.expected_iterations
            )
            record = SampleRecord(
                engine_key=engine.key,
                engine=engine.engine,
                phase=phase,
                engine_ordinal=ordinal,
                schedule_position=schedule_position,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=elapsed,
                iterations_per_second=parsed.throughput,
                iterations=parsed.iterations,
            )
            records.append(record)
            engine_records[key].append(record)
            suffix = " (discarded)" if phase == "warmup" else ""
            print(
                f"[harness]   schedule {schedule_position}: {engine.engine} "
                f"{phase} {ordinal}/{count}: {parsed.throughput:.1f} iter/s"
                f"{suffix}",
                file=sys.stderr,
            )

    results = {}
    for engine in engines:
        measured = [
            record
            for record in engine_records[engine.key]
            if record.phase == "measured"
        ]
        if len(measured) != runs:
            raise RuntimeError(
                f"{engine.engine} produced {len(measured)} measured records; "
                f"expected {runs}"
            )
        results[engine.key] = EngineResult(
            engine.engine,
            engine.version,
            engine.optimize,
            [record.iterations_per_second for record in measured],
            engine_records[engine.key],
        )
    return results, records


def prepare_wamr(
    wt: Path,
    ref: str,
    sha: str,
    fixture: Path,
    optimize: str,
    guest_args: tuple[str, ...] = COREMARK_GUEST_ARGS,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> PreparedEngine:
    env = worktree_env(wt)
    print(f"[harness] building WAMR {ref} ({sha[:12]}, {optimize})", file=sys.stderr)
    run(["zig", "build", f"-Doptimize={optimize}"], cwd=wt, env=env)

    wamrc = wt / "zig-out/bin/wamrc"
    wamr = wt / "zig-out/bin/wamr"
    cwasm = wt / ".bench-coremark.cwasm"
    print(f"[harness] AOT-compiling tracked fixture with WAMR {ref}", file=sys.stderr)
    run(
        [str(wamrc), "compile", str(fixture), "-o", str(cwasm)],
        cwd=wt,
        env=env,
    )
    return PreparedEngine(
        key=f"wamr:{sha}:{optimize}",
        engine=f"WAMR {ref}",
        version=f"{ref} ({sha})",
        optimize=optimize,
        cmd=[
            str(wamr),
            "run",
            str(cwasm),
            *guest_args,
        ],
        cwd=wt,
        env=env,
        expected_iterations=expected_iterations,
        retry_wamr_flake=True,
    )


def build_and_run_wamr(
    wt: Path,
    ref: str,
    sha: str,
    fixture: Path,
    optimize: str,
    warmups: int,
    runs: int,
    affinity: AffinityInfo | None = None,
    guest_args: tuple[str, ...] = COREMARK_GUEST_ARGS,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> EngineResult:
    prepared = prepare_wamr(
        wt,
        ref,
        sha,
        fixture,
        optimize,
        guest_args,
        expected_iterations,
    )
    values = measure_command(
        engine=f"WAMR {ref}",
        cmd=prepared.cmd,
        cwd=wt,
        env=prepared.env,
        warmups=warmups,
        runs=runs,
        retry_wamr_flake=True,
        affinity=affinity,
        expected_iterations=expected_iterations,
    )
    return EngineResult("WAMR", f"{ref} ({sha})", optimize, values)


def normalize_arch_name(arch: str) -> str:
    arch = arch.lower()
    return {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "aarch64",
    }.get(arch, arch)


def normalize_arch() -> str:
    return normalize_arch_name(platform.machine())


def wasmtime_version(path: Path) -> str:
    output = run([str(path), "--version"]).strip()
    match = re.search(r"\bwasmtime\s+v?([0-9]+\.[0-9]+\.[0-9]+)\b", output)
    if not match:
        raise RuntimeError(f"could not parse Wasmtime version from: {output!r}")
    return match.group(1)


def validate_pinned_wasmtime(path: Path) -> str:
    version = wasmtime_version(path)
    if version != PINNED_WASMTIME_VERSION:
        raise RuntimeError(
            f"historical Wasmtime baseline must be {PINNED_WASMTIME_VERSION}; "
            f"{path} reports {version}"
        )
    return version


def install_pinned_wasmtime(repo: Path, cache_arg: Path | None) -> Path:
    key = (platform.system().lower(), normalize_arch())
    asset = PINNED_WASMTIME_ASSETS.get(key)
    if asset is None:
        supported = ", ".join(f"{os_name}/{arch}" for os_name, arch in PINNED_WASMTIME_ASSETS)
        raise RuntimeError(
            f"automatic Wasmtime installation is unsupported on {key[0]}/{key[1]}; "
            f"supported: {supported}. Pass --wasmtime-baseline PATH instead."
        )
    asset_name, expected_archive_sha, expected_binary_sha = asset
    cache = (
        cache_arg.expanduser().resolve()
        if cache_arg is not None
        else repo / ".bench-coremark/tools"
    )
    install_dir = cache / f"wasmtime-v{PINNED_WASMTIME_VERSION}-{key[1]}-{key[0]}"
    binary = install_dir / "wasmtime"
    if binary.is_file():
        actual_binary_sha = sha256_file(binary)
        if actual_binary_sha != expected_binary_sha:
            raise RuntimeError(
                f"cached Wasmtime binary checksum mismatch: expected "
                f"{expected_binary_sha}, got {actual_binary_sha}"
            )
        validate_pinned_wasmtime(binary)
        return binary

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / asset_name
    url = (
        "https://github.com/bytecodealliance/wasmtime/releases/download/"
        f"v{PINNED_WASMTIME_VERSION}/{asset_name}"
    )
    print(f"[harness] downloading pinned Wasmtime {PINNED_WASMTIME_VERSION}", file=sys.stderr)
    urllib.request.urlretrieve(url, archive)
    actual_sha = sha256_file(archive)
    if actual_sha != expected_archive_sha:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            "Wasmtime archive checksum mismatch: expected "
            f"{expected_archive_sha}, got {actual_sha}"
        )

    install_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.isfile() and Path(member.name).name == "wasmtime"
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"expected one wasmtime binary in {archive}, found {len(members)}"
            )
        source = tar.extractfile(members[0])
        if source is None:
            raise RuntimeError(f"could not extract wasmtime from {archive}")
        with binary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    binary.chmod(0o755)
    archive.unlink()
    actual_binary_sha = sha256_file(binary)
    if actual_binary_sha != expected_binary_sha:
        binary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Wasmtime binary checksum mismatch: expected {expected_binary_sha}, "
            f"got {actual_binary_sha}"
        )
    validate_pinned_wasmtime(binary)
    return binary


def measure_wasmtime(
    path: Path,
    label: str,
    version: str,
    fixture: Path,
    warmups: int,
    runs: int,
    affinity: AffinityInfo | None = None,
    guest_args: tuple[str, ...] = COREMARK_GUEST_ARGS,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> EngineResult:
    binary_sha = sha256_file(path)
    values = measure_command(
        engine=label,
        cmd=[
            str(path),
            "run",
            str(fixture),
            *guest_args,
        ],
        cwd=fixture.parent,
        env=os.environ.copy(),
        warmups=warmups,
        runs=runs,
        affinity=affinity,
        expected_iterations=expected_iterations,
    )
    return EngineResult(
        label,
        f"{version} (sha256:{binary_sha}; {path})",
        "default JIT",
        values,
    )


def prepare_wasmtime(
    path: Path,
    label: str,
    version: str,
    fixture: Path,
    guest_args: tuple[str, ...] = COREMARK_GUEST_ARGS,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> PreparedEngine:
    binary_sha = sha256_file(path)
    return PreparedEngine(
        key=f"wasmtime:{path.resolve()}",
        engine=label,
        version=f"{version} (sha256:{binary_sha}; {path})",
        optimize="default JIT",
        cmd=[
            str(path),
            "run",
            str(fixture),
            *guest_args,
        ],
        cwd=fixture.parent,
        env=os.environ.copy(),
        expected_iterations=expected_iterations,
    )


def fmt_stats(values: list[float]) -> tuple[float, float, float, float]:
    return (
        statistics.fmean(values),
        statistics.median(values),
        min(values),
        max(values),
    )


def compute_delta_pct(baseline_vals: list[float], target_vals: list[float]) -> float:
    return (statistics.fmean(target_vals) / statistics.fmean(baseline_vals) - 1.0) * 100.0


def host_cpu_model() -> str:
    try:
        out = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if line.lower().startswith("model name:"):
                val = line.split(":", 1)[1].strip()
                if val:
                    return val
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                key = line.split(":", 1)[0].strip().lower()
                if key in ("model name", "model") and ":" in line:
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return val
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def read_optional_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def capture_host_identity(*, require_boot_id: bool = False) -> HostIdentity:
    arch = normalize_arch()
    cpu_count = os.cpu_count() or 0
    cpu_model = host_cpu_model()
    runner_name = os.environ.get("RUNNER_NAME", "")
    boot_id = read_optional_text(Path("/proc/sys/kernel/random/boot_id"))
    if require_boot_id and not boot_id:
        raise RuntimeError("cannot establish host identity: Linux boot ID is unavailable")
    return HostIdentity(arch, cpu_count, cpu_model, runner_name, boot_id)


def host_emulation_evidence() -> str:
    evidence = [
        platform.platform(),
        os.environ.get("QEMU_CPU", ""),
        os.environ.get("QEMU_LD_PREFIX", ""),
        read_optional_text(Path("/proc/cpuinfo")),
        read_optional_text(Path("/sys/class/dmi/id/product_name")),
        read_optional_text(Path("/sys/class/dmi/id/sys_vendor")),
    ]
    try:
        evidence.append(
            subprocess.run(
                ["lscpu"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return "\n".join(evidence)


def validate_native_host(expected_arch: str) -> HostIdentity:
    expected = normalize_arch_name(expected_arch)
    identity = capture_host_identity(require_boot_id=True)
    if identity.arch != expected:
        raise RuntimeError(
            f"authoritative host architecture must be {expected}; got {identity.arch}"
        )
    runner_arch = os.environ.get("RUNNER_ARCH")
    if runner_arch and normalize_arch_name(runner_arch) != expected:
        raise RuntimeError(
            f"RUNNER_ARCH must be {expected}; got {runner_arch}"
        )
    if identity.cpu_count <= 0:
        raise RuntimeError("authoritative host CPU count is unavailable")
    if not identity.cpu_model or identity.cpu_model == "unknown CPU":
        raise RuntimeError("authoritative host CPU model is unavailable")

    evidence = host_emulation_evidence()
    if re.search(r"\b(qemu|tcg|emulat(?:e|ed|ion|or))\b", evidence, re.IGNORECASE):
        raise RuntimeError("authoritative measurements cannot run under emulation")
    if expected == "aarch64" and re.search(
        r"\b(Intel|AMD|Xeon|EPYC)\b", identity.cpu_model, re.IGNORECASE
    ):
        raise RuntimeError(
            "AArch64 process reports an x86 CPU model, indicating user-mode emulation"
        )
    return identity


def validate_same_host(expected: HostIdentity) -> None:
    actual = capture_host_identity(require_boot_id=bool(expected.boot_id))
    if actual != expected:
        raise RuntimeError(
            "host identity changed during measurement: "
            f"started {expected.fingerprint()}, ended {actual.fingerprint()}"
        )


def host_info(identity: HostIdentity) -> str:
    parts = [
        f"arch `{identity.arch}`",
        f"{identity.cpu_count or '?'} vCPU",
        f"`{identity.cpu_model}`",
    ]
    if identity.runner_name:
        parts.append(f"runner `{identity.runner_name}`")
    parts.append(f"host fingerprint `{identity.fingerprint()}`")
    return "_Host: " + " · ".join(parts) + "_"


def format_samples(values: list[float]) -> str:
    return ", ".join(f"{value:.1f}" for value in values)


def schedule_label(records: list[SampleRecord]) -> str:
    names: dict[str, str] = {}
    labels = []
    for record in records:
        if record.engine_key not in names:
            names[record.engine_key] = chr(ord("A") + len(names))
        labels.append(names[record.engine_key])
    legend = ", ".join(
        f"{label}={next(r.engine for r in records if r.engine_key == key)}"
        for key, label in names.items()
    )
    return "".join(labels) + (f" ({legend})" if legend else "")


def render_table(
    results: list[EngineResult],
    *,
    profile: str,
    warmups: int,
    runs: int,
    fixture: Path,
    fixture_sha: str,
    host: HostIdentity,
    schedule_records: list[SampleRecord] | None = None,
    affinity: AffinityInfo | None = None,
    guest_args: tuple[str, ...] = COREMARK_GUEST_ARGS,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> str:
    for result in results:
        if len(result.values) != runs:
            raise RuntimeError(
                f"{result.engine} produced {len(result.values)} measured samples; "
                f"expected {runs}"
            )
    baseline, target = results[0], results[1]
    delta_pct = compute_delta_pct(baseline.values, target.values)
    sign = "+" if delta_pct >= 0 else ""
    lines = [
        "### CoreMark cross-engine comparison",
        "",
        f"- Measurement profile: `{profile}` ({warmups} warmups discarded, "
        f"{runs} measured runs per engine)",
        f"- Fixture: `{fixture}` (`sha256:{fixture_sha}`)",
        f"- Fixed guest args: `{' '.join(guest_args)}`; required "
        f"`Iterations: {expected_iterations}` and `{PERFORMANCE_MARKER}`",
        f"- WAMR optimize mode: `{target.optimize}`; Wasmtime mode: `default JIT`",
        f"- CRC validation: all {warmups + runs} invocations per measured engine "
        f"produced exactly one `{VALIDATION_TEXT}`, the fixed iteration count, "
        f"and no CoreMark CRC error; warmups were discarded",
        "- Methodology correction: supersedes non-authoritative run "
        "`33576430466`, which allowed independent self-calibration and "
        "separate unpinned engine blocks",
        "",
        "| Engine | Version / ref | Optimize mode | Mean iter/s | Median | Range | Samples (iter/s) |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        mean, median, minimum, maximum = fmt_stats(result.values)
        lines.append(
            f"| {result.engine} | `{result.version}` | `{result.optimize}` | "
            f"{mean:.1f} | {median:.1f} | {minimum:.1f}–{maximum:.1f} | "
            f"{format_samples(result.values)} |"
        )
    lines.extend(
        [
            "",
            f"**WAMR target vs baseline:** {sign}{delta_pct:.2f}%",
        ]
    )
    wasmtime_results = results[2:]
    if wasmtime_results:
        lines.extend(
            [
                "",
                "| Same-host ratio | Median iter/s ratio | Mean iter/s ratio |",
                "|---|---:|---:|",
            ]
        )
        target_mean = statistics.fmean(target.values)
        target_median = statistics.median(target.values)
        for result in wasmtime_results:
            median_ratio = target_median / statistics.median(result.values)
            mean_ratio = target_mean / statistics.fmean(result.values)
            lines.append(
                f"| WAMR target / {result.engine} `{result.version}` | "
                f"{median_ratio:.3f}× | {mean_ratio:.3f}× |"
            )
    if affinity is not None:
        lines.extend(
            [
                "",
                f"- CPU affinity: allowed `{','.join(map(str, affinity.allowed_cpus))}`; "
                f"selected and verified CPU `{affinity.selected_cpu}` via "
                f"`{affinity.taskset}`",
            ]
        )
    if schedule_records:
        lines.extend(
            [
                f"- Counterbalanced order: `{schedule_label(schedule_records)}`",
                "",
                "| Position | Phase | Engine | Engine sample | Started (UTC) | "
                "Completed (UTC) | Iterations | Iter/s |",
                "|---:|---|---|---:|---|---|---:|---:|",
            ]
        )
        for record in schedule_records:
            lines.append(
                f"| {record.schedule_position} | {record.phase} | "
                f"{record.engine} | {record.engine_ordinal} | "
                f"`{record.started_at}` | `{record.completed_at}` | "
                f"{record.iterations} | {record.iterations_per_second:.1f} |"
            )
    lines.extend(["", host_info(host)])
    return "\n".join(lines)


def build_json_report(
    results: list[EngineResult],
    *,
    profile: str,
    warmups: int,
    runs: int,
    fixture: Path,
    fixture_sha: str,
    host: HostIdentity,
    schedule_records: list[SampleRecord],
    affinity: AffinityInfo | None,
    guest_args: tuple[str, ...] = COREMARK_GUEST_ARGS,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> dict:
    engines = []
    for result in results:
        mean, median, minimum, maximum = fmt_stats(result.values)
        engines.append(
            {
                "engine": result.engine,
                "version": result.version,
                "optimize": result.optimize,
                "mean_iterations_per_second": mean,
                "median_iterations_per_second": median,
                "minimum_iterations_per_second": minimum,
                "maximum_iterations_per_second": maximum,
                "values": result.values,
            }
        )
    ratios = []
    target = results[1]
    for result in results[2:]:
        ratios.append(
            {
                "numerator": target.engine,
                "denominator": result.engine,
                "median_ratio": (
                    statistics.median(target.values)
                    / statistics.median(result.values)
                ),
                "mean_ratio": (
                    statistics.fmean(target.values)
                    / statistics.fmean(result.values)
                ),
            }
        )
    return {
        "schema_version": 1,
        "kind": "coremark-authoritative-comparison",
        "profile": profile,
        "warmups_per_engine": warmups,
        "measured_samples_per_engine": runs,
        "fixture": {"path": str(fixture), "sha256": fixture_sha},
        "guest_args": list(guest_args),
        "expected_iterations": expected_iterations,
        "required_performance_marker": PERFORMANCE_MARKER,
        "required_crc_marker": VALIDATION_TEXT,
        "invalidated_run": 33576430466,
        "host": {
            "arch": host.arch,
            "cpu_count": host.cpu_count,
            "cpu_model": host.cpu_model,
            "runner_name": host.runner_name,
            "fingerprint": host.fingerprint(),
        },
        "affinity": (
            {
                "allowed_cpus": list(affinity.allowed_cpus),
                "selected_cpu": affinity.selected_cpu,
                "taskset": affinity.taskset,
                "verified": True,
            }
            if affinity is not None
            else None
        ),
        "schedule": [asdict(record) for record in schedule_records],
        "engines": engines,
        "ratios": ratios,
    }


def render_optimize_table(
    target_ref: str,
    results: dict[str, list[float] | None],
    *,
    profile: str,
    warmups: int,
    runs: int,
    fixture: Path,
    fixture_sha: str,
    host: HostIdentity,
) -> str:
    lines = [
        "### CoreMark AOT optimize-mode comparison",
        "",
        f"- Measurement profile: `{profile}` ({warmups} warmups discarded, "
        f"{runs} measured runs per mode)",
        f"- Fixture: `{fixture}` (`sha256:{fixture_sha}`)",
        "",
        "| Ref | Mode | Mean iter/s | Median | Range | Samples (iter/s) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for mode in ("ReleaseFast", "ReleaseSafe"):
        values = results[mode]
        if values is None:
            lines.append(f"| `{target_ref}` | `{mode}` | failed | failed | failed | failed |")
            continue
        mean, median, minimum, maximum = fmt_stats(values)
        lines.append(
            f"| `{target_ref}` | `{mode}` | {mean:.1f} | {median:.1f} | "
            f"{minimum:.1f}–{maximum:.1f} | {format_samples(values)} |"
        )
    fast_vals = results["ReleaseFast"]
    safe_vals = results["ReleaseSafe"]
    if fast_vals is not None and safe_vals is not None:
        lines.extend(
            [
                "",
                "**ReleaseSafe / ReleaseFast:** "
                + fmt_ratio(statistics.fmean(safe_vals), statistics.fmean(fast_vals)),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "At least one optimize mode failed before producing complete timings.",
            ]
        )
    lines.extend(["", host_info(host)])
    return "\n".join(lines)


def resolve_counts(
    profile: str, warmups_arg: int | None, runs_arg: int | None
) -> tuple[int, int]:
    default_warmups, default_runs = PROFILE_COUNTS[profile]
    warmups = default_warmups if warmups_arg is None else warmups_arg
    runs = default_runs if runs_arg is None else runs_arg
    if warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if runs <= 0:
        raise ValueError("--runs must be greater than zero")
    return warmups, runs


def profile_label(
    profile: str, warmups_arg: int | None, runs_arg: int | None
) -> str:
    defaults = PROFILE_COUNTS[profile]
    resolved = resolve_counts(profile, warmups_arg, runs_arg)
    return profile if resolved == defaults else f"{profile} (overridden)"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", default="origin/main", help="WAMR baseline git ref")
    p.add_argument("--target", default="HEAD", help="WAMR target git ref")
    p.add_argument(
        "--profile",
        choices=PROFILE_COUNTS,
        default="authoritative",
        help="measurement profile (default: authoritative = 2 warmups + 10 runs; ci = 0 + 3)",
    )
    p.add_argument("--warmups", type=int, default=None, help="override discarded warmups")
    p.add_argument("--runs", type=int, default=None, help="override measured runs")
    p.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help=f"CoreMark wasm fixture (default: {DEFAULT_FIXTURE})",
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="path to wamr repo",
    )
    p.add_argument("--out", type=Path, default=None, help="write markdown report here")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write machine-readable comparison report here",
    )
    p.add_argument(
        "--emit",
        choices=["markdown", "github"],
        default="markdown",
        help="`github` also appends to $GITHUB_STEP_SUMMARY",
    )
    p.add_argument(
        "--optimize",
        choices=OPTIMIZE_CHOICES,
        default="ReleaseFast",
        help="WAMR Zig optimize mode; `both` compares target ReleaseFast vs ReleaseSafe",
    )
    p.add_argument(
        "--wasmtime-baseline",
        default=None,
        metavar="PATH|auto",
        help=f"historical pinned Wasmtime {PINNED_WASMTIME_VERSION}; `auto` downloads a checksum-verified binary",
    )
    p.add_argument(
        "--wasmtime",
        type=Path,
        default=None,
        help="caller-selected/current Wasmtime binary; its exact version is reported",
    )
    p.add_argument(
        "--wasmtime-cache",
        type=Path,
        default=None,
        help="cache for --wasmtime-baseline auto (default: .bench-coremark/tools)",
    )
    p.add_argument(
        "--require-native-arch",
        choices=("aarch64", "x86_64"),
        default=None,
        help="fail unless the run is on a native, non-emulated host of this architecture",
    )
    p.add_argument(
        "--min-delta-pct",
        type=float,
        default=None,
        help="fail if WAMR target mean is below WAMR baseline by more than this delta",
    )
    args = p.parse_args()

    try:
        warmups, runs = resolve_counts(args.profile, args.warmups, args.runs)
    except ValueError as exc:
        p.error(str(exc))
    effective_profile = profile_label(args.profile, args.warmups, args.runs)
    expected_iterations = PROFILE_ITERATIONS[args.profile]
    guest_args = coremark_guest_args(expected_iterations)
    if args.optimize == "both" and (args.wasmtime_baseline or args.wasmtime):
        p.error("Wasmtime comparisons require a single --optimize mode")
    if args.optimize == "both" and args.json_out:
        p.error("--json-out is not supported with --optimize both")

    repo = args.repo.resolve()
    fixture, fixture_sha = resolve_fixture(repo, args.fixture)
    host = (
        validate_native_host(args.require_native_arch)
        if args.require_native_arch
        else capture_host_identity()
    )
    work_root = repo / ".bench-coremark" / f"run-{uuid.uuid4().hex}"
    work_root.mkdir(parents=True)
    optimize_modes = parse_optimize_modes(args.optimize)
    baseline_vals: list[float] | None = None
    target_vals: list[float] = []
    optimize_failed = False
    schedule_records: list[SampleRecord] = []
    affinity = (
        select_cpu_affinity() if args.profile == "authoritative" else None
    )
    report_results: list[EngineResult] | None = None

    try:
        if args.optimize == "both":
            mode_results: dict[str, list[float] | None] = {}
            for optimize in optimize_modes:
                wt, sha = make_worktree(
                    repo, args.target, work_root, f"target-{optimize_slug(optimize)}"
                )
                try:
                    result = build_and_run_wamr(
                        wt,
                        args.target,
                        sha,
                        fixture,
                        optimize,
                        warmups,
                        runs,
                        affinity,
                        guest_args,
                        expected_iterations,
                    )
                    mode_results[optimize] = result.values
                except Exception as exc:
                    print(
                        f"[harness] {optimize} failed before producing complete "
                        f"CoreMark timings: {exc}",
                        file=sys.stderr,
                    )
                    mode_results[optimize] = None
                    optimize_failed = True
            table = render_optimize_table(
                args.target,
                mode_results,
                profile=effective_profile,
                warmups=warmups,
                runs=runs,
                fixture=fixture,
                fixture_sha=fixture_sha,
                host=host,
            )
            target_vals = mode_results["ReleaseFast"] or []
        else:
            optimize = optimize_modes[0]
            baseline_sha = resolve_ref_sha(repo, args.baseline)
            target_sha = resolve_ref_sha(repo, args.target)
            if args.wasmtime_baseline or args.wasmtime:
                prepared: list[PreparedEngine] = []
                if baseline_sha == target_sha:
                    target_wt, target_sha = make_worktree(
                        repo, args.target, work_root, "target"
                    )
                    target_prepared = prepare_wamr(
                        target_wt,
                        args.target,
                        target_sha,
                        fixture,
                        optimize,
                        guest_args,
                        expected_iterations,
                    )
                    prepared.append(target_prepared)
                    baseline_prepared = target_prepared
                else:
                    baseline_wt, baseline_sha = make_worktree(
                        repo, args.baseline, work_root, "baseline"
                    )
                    target_wt, target_sha = make_worktree(
                        repo, args.target, work_root, "target"
                    )
                    baseline_prepared = prepare_wamr(
                        baseline_wt,
                        args.baseline,
                        baseline_sha,
                        fixture,
                        optimize,
                        guest_args,
                        expected_iterations,
                    )
                    target_prepared = prepare_wamr(
                        target_wt,
                        args.target,
                        target_sha,
                        fixture,
                        optimize,
                        guest_args,
                        expected_iterations,
                    )
                    prepared.extend([baseline_prepared, target_prepared])

                wasmtime_specs: list[tuple[Path, str, str]] = []
                if args.wasmtime_baseline:
                    pinned_path = (
                        install_pinned_wasmtime(repo, args.wasmtime_cache)
                        if args.wasmtime_baseline == "auto"
                        else Path(args.wasmtime_baseline).expanduser().resolve()
                    )
                    if not pinned_path.is_file():
                        raise FileNotFoundError(
                            f"historical Wasmtime binary not found: {pinned_path}"
                        )
                    wasmtime_specs.append(
                        (
                            pinned_path,
                            "Wasmtime historical pin",
                            validate_pinned_wasmtime(pinned_path),
                        )
                    )
                if args.wasmtime:
                    current_path = args.wasmtime.expanduser().resolve()
                    if not current_path.is_file():
                        raise FileNotFoundError(
                            f"caller-selected Wasmtime binary not found: {current_path}"
                        )
                    wasmtime_specs.append(
                        (
                            current_path,
                            "Wasmtime caller-selected",
                            wasmtime_version(current_path),
                        )
                    )

                prepared_wasmtime: dict[Path, PreparedEngine] = {}
                for path, label, version in wasmtime_specs:
                    resolved = path.resolve()
                    if resolved not in prepared_wasmtime:
                        engine = prepare_wasmtime(
                            path,
                            label,
                            version,
                            fixture,
                            guest_args,
                            expected_iterations,
                        )
                        prepared_wasmtime[resolved] = engine
                        prepared.append(engine)

                measured, schedule_records = measure_prepared_engines(
                    prepared,
                    warmups=warmups,
                    runs=runs,
                    affinity=affinity,
                )
                target_measured = measured[target_prepared.key]
                target_result = EngineResult(
                    "WAMR",
                    target_prepared.version,
                    optimize,
                    target_measured.values,
                    target_measured.samples,
                )
                if baseline_sha == target_sha:
                    baseline_result = EngineResult(
                        "WAMR",
                        f"{args.baseline} ({baseline_sha})",
                        optimize,
                        target_result.values.copy(),
                        target_result.samples,
                    )
                    print(
                        "[harness] WAMR baseline and target resolve to the same "
                        "commit; reusing the target samples",
                        file=sys.stderr,
                    )
                else:
                    baseline_measured = measured[baseline_prepared.key]
                    baseline_result = EngineResult(
                        "WAMR",
                        baseline_prepared.version,
                        optimize,
                        baseline_measured.values,
                        baseline_measured.samples,
                    )
                results = [baseline_result, target_result]
                for path, label, version in wasmtime_specs:
                    measured_result = measured[
                        prepared_wasmtime[path.resolve()].key
                    ]
                    results.append(
                        EngineResult(
                            label,
                            f"{version} (sha256:{sha256_file(path)}; {path})",
                            "default JIT",
                            measured_result.values,
                            measured_result.samples,
                        )
                    )
            else:
                if baseline_sha == target_sha:
                    target_wt, target_sha = make_worktree(
                        repo, args.target, work_root, "target"
                    )
                    target_result = build_and_run_wamr(
                        target_wt,
                        args.target,
                        target_sha,
                        fixture,
                        optimize,
                        warmups,
                        runs,
                        affinity,
                        guest_args,
                        expected_iterations,
                    )
                    baseline_result = EngineResult(
                        "WAMR",
                        f"{args.baseline} ({baseline_sha})",
                        optimize,
                        target_result.values.copy(),
                    )
                else:
                    baseline_wt, baseline_sha = make_worktree(
                        repo, args.baseline, work_root, "baseline"
                    )
                    target_wt, target_sha = make_worktree(
                        repo, args.target, work_root, "target"
                    )
                    baseline_result = build_and_run_wamr(
                        baseline_wt,
                        args.baseline,
                        baseline_sha,
                        fixture,
                        optimize,
                        warmups,
                        runs,
                        affinity,
                        guest_args,
                        expected_iterations,
                    )
                    target_result = build_and_run_wamr(
                        target_wt,
                        args.target,
                        target_sha,
                        fixture,
                        optimize,
                        warmups,
                        runs,
                        affinity,
                        guest_args,
                        expected_iterations,
                    )
                results = [baseline_result, target_result]

            report_results = results
            baseline_vals = baseline_result.values
            target_vals = target_result.values
            table = render_table(
                report_results,
                profile=effective_profile,
                warmups=warmups,
                runs=runs,
                fixture=fixture,
                fixture_sha=fixture_sha,
                host=host,
                schedule_records=schedule_records,
                affinity=affinity,
                guest_args=guest_args,
                expected_iterations=expected_iterations,
            )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        run(["git", "worktree", "prune"], cwd=repo)

    validate_same_host(host)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
    if args.json_out:
        assert report_results is not None
        report = build_json_report(
            report_results,
            profile=effective_profile,
            warmups=warmups,
            runs=runs,
            fixture=fixture,
            fixture_sha=fixture_sha,
            host=host,
            schedule_records=schedule_records,
            affinity=affinity,
            guest_args=guest_args,
            expected_iterations=expected_iterations,
        )
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    if args.emit == "github":
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as fh:
                fh.write(table + "\n")

    if optimize_failed:
        return 1
    if args.min_delta_pct is not None and baseline_vals is not None:
        delta_pct = compute_delta_pct(baseline_vals, target_vals)
        if delta_pct < args.min_delta_pct:
            print(
                f"CoreMark AOT regression: {delta_pct:.2f}% is below "
                f"allowed minimum {args.min_delta_pct:.2f}%",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
