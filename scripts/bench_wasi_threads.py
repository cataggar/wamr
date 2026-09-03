#!/usr/bin/env python3
"""Run reproducible paired WASI pthread, atomic, and cancel-poll benchmarks."""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import re
import shlex
import signal
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmark_schema import (
    BenchmarkDataError,
    SCHEMA_VERSION,
    alternating_pair_order,
    atomic_write_json,
    cache_key,
    collected_at,
    command_identity,
    host_metadata,
    require,
    sample_stats,
    sha256_bytes,
    sha256_file,
    validate_common_report,
)

CANONICAL_PLATFORMS = {
    "ubuntu-22.04-x86_64": ("Linux", "x86_64"),
    "ubuntu-24.04-aarch64": ("Linux", "aarch64"),
}


KIND = "wasi-thread-benchmark"
PROFILE_COUNTS = {
    "authoritative": (2, 10),
    "smoke": (1, 3),
}
MIN_TIMED_INTERVAL_MS = 100.0
AOT_VERSION = 11
# Stable fast-path signatures emitted by emitCancelPoint in each backend.
# Counting these signatures avoids treating instruction-sequence byte sizes as
# an ABI; text growth per site is derived from each on/off artifact pair.
CANCEL_POLL_SIGNATURES = {
    "x86_64": bytes.fromhex("83bbb001000000740c"),
    "aarch64": struct.pack("<II", 0xB941B270, 0x34000090),
}
WASI_SDK = {
    "version": "25.0",
    "clang": "clang version 19.1.5-wasi-sdk",
    "archive": "wasi-sdk-25.0-x86_64-linux.tar.gz",
    "archive_sha256": "52640dde13599bf127a95499e61d6d640256119456d1af8897ab6725bcf3d89c",
    "url": (
        "https://github.com/WebAssembly/wasi-sdk/releases/download/"
        "wasi-sdk-25/wasi-sdk-25.0-x86_64-linux.tar.gz"
    ),
    "wasi_libc_revision": "574b88da481569b65a237cb80daf9a2d5aeaf82d",
}
FIXTURES = {
    "single": {
        "path": Path("tests/benchmarks/wasi-threads/single.wasm"),
        "sha256": "c307570e7086b929b4740beb08b6859353e57421ce5ffa2944d921d8eadf2402",
    },
    "threaded": {
        "path": Path("tests/benchmarks/wasi-threads/threaded.wasm"),
        "sha256": "3905e4d35da45fd7b60abde10388dc23b276439eb6f4c67e844fff05d76c5474",
    },
}
MASK64 = (1 << 64) - 1


class HarnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Build:
    name: str
    mode: str
    threads_enabled: bool
    prefix: Path
    wamr: Path
    wamrc: Path | None
    key: str
    command: list[str]
    reused: bool


@dataclass(frozen=True)
class Scenario:
    workload: str
    threads: int
    iterations: int

    @property
    def key(self) -> str:
        return f"{self.workload}/{self.threads}"


def planned_scenarios(args: argparse.Namespace) -> list[Scenario]:
    return [
        Scenario(
            workload,
            threads,
            iterations // threads
            if workload == "wait-notify"
            else iterations,
        )
        for workload, iterations in (
            ("hot", args.hot_iterations),
            ("atomic", args.atomic_iterations),
            ("wait-notify", args.wait_iterations),
            ("spawn-join", args.spawn_iterations),
        )
        for threads in args.thread_counts
    ]


def planned_pair_specs(
    args: argparse.Namespace,
    modes: tuple[str, ...],
) -> list[dict[str, str]]:
    pairs = [
        {
            "pair_kind": "single-infrastructure",
            "pair_key": f"single-infrastructure/{mode}",
            "left": "threads-disabled",
            "right": "threads-enabled",
        }
        for mode in modes
    ]
    for scenario in planned_scenarios(args):
        if len(modes) == 2:
            pairs.append(
                {
                    "pair_kind": "runtime-parity",
                    "pair_key": f"runtime/{scenario.key}",
                    "left": "interpreter",
                    "right": "aot",
                }
            )
        else:
            mode = modes[0]
            pairs.append(
                {
                    "pair_kind": "repeatability",
                    "pair_key": f"runtime/{scenario.key}/{mode}",
                    "left": f"{mode}-a",
                    "right": f"{mode}-b",
                }
            )
    if "aot" in modes:
        for threads in args.thread_counts:
            pairs.append(
                {
                    "pair_kind": "cancel-point-cost",
                    "pair_key": f"cancel-points/hot/{threads}",
                    "left": "cancel-points-off",
                    "right": "cancel-points-on",
                }
            )
    return pairs


def expected_pair_specs_for_plan(plan: dict[str, Any]) -> list[dict[str, str]]:
    modes = tuple(plan["modes"])
    thread_counts = tuple(plan["thread_counts"])
    iterations = plan["iterations"]
    pairs = [
        {
            "pair_kind": "single-infrastructure",
            "pair_key": f"single-infrastructure/{mode}",
            "left": "threads-disabled",
            "right": "threads-enabled",
        }
        for mode in modes
    ]
    for workload in ("hot", "atomic", "wait-notify", "spawn-join"):
        require(workload in iterations, f"plan iterations missing {workload}")
        for threads in thread_counts:
            if len(modes) == 2:
                pairs.append(
                    {
                        "pair_kind": "runtime-parity",
                        "pair_key": f"runtime/{workload}/{threads}",
                        "left": "interpreter",
                        "right": "aot",
                    }
                )
            else:
                mode = modes[0]
                pairs.append(
                    {
                        "pair_kind": "repeatability",
                        "pair_key": f"runtime/{workload}/{threads}/{mode}",
                        "left": f"{mode}-a",
                        "right": f"{mode}-b",
                    }
                )
    if "aot" in modes:
        for threads in thread_counts:
            pairs.append(
                {
                    "pair_kind": "cancel-point-cost",
                    "pair_key": f"cancel-points/hot/{threads}",
                    "left": "cancel-points-off",
                    "right": "cancel-points-on",
                }
            )
    return pairs


def parse_thread_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thread counts must be comma-separated integers") from exc
    if not counts or any(count not in (1, 2, 4, 8) for count in counts):
        raise argparse.ArgumentTypeError("thread counts must be selected from 1,2,4,8")
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("thread counts must not contain duplicates")
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("zig-out/wasi-thread-bench")
    )
    parser.add_argument("--profile", choices=PROFILE_COUNTS, default="authoritative")
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--modes", choices=("both", "interpreter", "aot"), default="both")
    parser.add_argument("--thread-counts", type=parse_thread_counts, default=(1, 2, 4, 8))
    parser.add_argument("--single-iterations", type=int, default=128_000_000)
    parser.add_argument("--hot-iterations", type=int, default=128_000_000)
    parser.add_argument("--atomic-iterations", type=int, default=64_000_000)
    parser.add_argument("--wait-iterations", type=int, default=512_000)
    parser.add_argument("--spawn-iterations", type=int, default=3_000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--min-interval-ms", type=float, default=MIN_TIMED_INTERVAL_MS
    )
    parser.add_argument(
        "--platform-id",
        default=f"local-{platform.system().lower()}-{platform.machine().lower()}",
    )
    parser.add_argument("--optimize", default="ReleaseFast")
    parser.add_argument(
        "--target",
        default=None,
        help="optional Zig runtime target, e.g. aarch64-linux-musl",
    )
    parser.add_argument(
        "--aot-target",
        choices=("x86_64", "aarch64"),
        default=None,
        help="wamrc output architecture (default: execution architecture)",
    )
    parser.add_argument(
        "--runner",
        default="",
        help="command prefix for cross execution, e.g. 'qemu-aarch64'",
    )
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--no-budget", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    if args.warmups is None or args.samples is None:
        default_warmups, default_samples = PROFILE_COUNTS[args.profile]
        args.warmups = default_warmups if args.warmups is None else args.warmups
        args.samples = default_samples if args.samples is None else args.samples
    if args.warmups < 0 or args.samples <= 0:
        parser.error("--warmups must be >= 0 and --samples must be > 0")
    for name in (
        "single_iterations",
        "hot_iterations",
        "atomic_iterations",
        "wait_iterations",
        "spawn_iterations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if any(
        args.wait_iterations % threads != 0
        for threads in args.thread_counts
    ):
        parser.error(
            "--wait-iterations must be divisible by every selected thread count"
        )
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.min_interval_ms <= 0:
        parser.error("--min-interval-ms must be > 0")
    if args.budget and args.no_budget:
        parser.error("--budget and --no-budget are mutually exclusive")
    return args


def execution_arch(args: argparse.Namespace) -> str:
    if args.aot_target:
        return args.aot_target
    if args.target and args.target.startswith("aarch64"):
        return "aarch64"
    machine = platform.machine().lower()
    return "aarch64" if machine in ("aarch64", "arm64") else "x86_64"


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def source_identity(repo: Path) -> dict[str, str]:
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", "build.zig", "src"],
        cwd=repo,
    )
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "build.zig", "build.zig.zon", "src"],
        cwd=repo,
    ).split(b"\0")
    content = bytearray()
    for raw_path in tracked:
        if not raw_path:
            continue
        path = raw_path.decode("UTF-8")
        content.extend(raw_path)
        content.append(0)
        content.extend((repo / path).read_bytes())
        content.append(0)
    return {
        "commit": git_output(repo, "rev-parse", "HEAD"),
        "tracked_diff_sha256": sha256_bytes(diff),
        "build_source_sha256": sha256_bytes(bytes(content)),
    }


def controlled_env(cache_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    cache_root.mkdir(parents=True, exist_ok=True)
    global_cache = cache_root / "global"
    local_cache = cache_root / "local"
    temp = cache_root / "tmp"
    for directory in (global_cache, local_cache, temp):
        directory.mkdir(parents=True, exist_ok=True)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    env["ZIG_LOCAL_CACHE_DIR"] = str(local_cache)
    env["TMPDIR"] = str(temp)
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    env["TZ"] = "UTC"
    return env


def build_variant(
    *,
    repo: Path,
    root: Path,
    mode: str,
    threads_enabled: bool,
    optimize: str,
    target: str | None,
    source: dict[str, str],
    rebuild: bool,
    compiler_toggle: bool,
) -> Build:
    name = f"{'enabled' if threads_enabled else 'disabled'}-{mode}"
    parts: dict[str, Any] = {
        "build_source_sha256": source["build_source_sha256"],
        "mode": mode,
        "threads_enabled": threads_enabled,
        "optimize": optimize,
        "target": target or "native",
        "compiler_toggle": compiler_toggle,
        "zig": command_identity(["zig", "version"]),
    }
    key = cache_key(parts)
    prefix = root / "builds" / f"{name}-{key[:16]}"
    marker = prefix / "benchmark-build.json"
    wamr = prefix / "bin" / "wamr"
    wamrc = prefix / "bin" / "wamrc" if mode == "aot" else None
    expected = [wamr] + ([wamrc] if wamrc is not None else [])
    if not rebuild and marker.is_file() and all(path.is_file() for path in expected):
        try:
            stored = json.loads(marker.read_text(encoding="UTF-8"))
        except json.JSONDecodeError:
            stored = {}
        if stored.get("schema_version") == SCHEMA_VERSION and stored.get("key") == key:
            return Build(name, mode, threads_enabled, prefix, wamr, wamrc, key, stored["command"], True)

    cache = root / "cache" / name
    env = controlled_env(cache)
    command = [
        "zig",
        "build",
        f"-Doptimize={optimize}",
        f"-Dlib_wasi_threads={'true' if threads_enabled else 'false'}",
        f"-Dinterp={'true' if mode == 'interpreter' else 'false'}",
        f"-Daot={'true' if mode == 'aot' else 'false'}",
        (
            "-Dbenchmark-interp-fuel=4000000000"
            if mode == "interpreter"
            else "-Dbenchmark-interp-fuel=100000000"
        ),
        (
            "-Dbenchmark-cancel-point-toggle=true"
            if compiler_toggle
            else "-Dbenchmark-cancel-point-toggle=false"
        ),
        "--prefix",
        str(prefix),
    ]
    if target:
        command.insert(2, f"-Dtarget={target}")
    prefix.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, cwd=repo, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise HarnessError(f"build failed for {name}: exit {exc.returncode}") from exc
    if not all(path.is_file() for path in expected):
        raise HarnessError(f"build {name} did not produce expected binaries")
    atomic_write_json(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "key": key,
            "parts": parts,
            "command": command,
        },
    )
    return Build(name, mode, threads_enabled, prefix, wamr, wamrc, key, command, False)


def resolve_fixtures(repo: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, spec in FIXTURES.items():
        path = (repo / spec["path"]).resolve()
        if not path.is_file():
            raise HarnessError(f"missing fixture: {path}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise HarnessError(
                f"{name} fixture checksum mismatch: expected {spec['sha256']}, got {digest}"
            )
        result[name] = {
            "path": str(spec["path"]),
            "sha256": digest,
            "size": path.stat().st_size,
        }
    for source_name in (
        "kernel.h",
        "output.h",
        "timing.h",
        "single.c",
        "threaded.c",
        "build-fixtures.sh",
    ):
        path = repo / "tests/benchmarks/wasi-threads" / source_name
        result[f"source:{source_name}"] = {
            "path": str(path.relative_to(repo)),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    return result


def compile_aot_fixtures(
    repo: Path,
    output: Path,
    compiler: Build,
    arch: str,
) -> dict[str, Path]:
    if compiler.wamrc is None:
        raise HarnessError("AOT compiler build is missing wamrc")
    artifacts = output / "aot"
    artifacts.mkdir(parents=True, exist_ok=True)
    commands = {
        "single": [
            str(compiler.wamrc),
            "compile",
            "--target",
            arch,
            str(repo / FIXTURES["single"]["path"]),
            "-o",
            str(artifacts / "single.cwasm"),
        ],
        "threaded-polls-on": [
            str(compiler.wamrc),
            "compile",
            "--target",
            arch,
            str(repo / FIXTURES["threaded"]["path"]),
            "-o",
            str(artifacts / "threaded-polls-on.cwasm"),
        ],
        "threaded-polls-off": [
            str(compiler.wamrc),
            "compile",
            "--target",
            arch,
            "--benchmark-disable-cancel-points",
            str(repo / FIXTURES["threaded"]["path"]),
            "-o",
            str(artifacts / "threaded-polls-off.cwasm"),
        ],
    }
    env = controlled_env(output / "cache" / "compile-aot")
    for name, command in commands.items():
        try:
            subprocess.run(command, cwd=repo, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            raise HarnessError(f"AOT fixture compilation failed for {name}") from exc
    return {name: Path(command[-1]) for name, command in commands.items()}


def aot_text_section(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 8 or data[:4] != b"\x00aot":
        raise HarnessError(f"not a WAMR AOT file: {path}")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != AOT_VERSION:
        raise HarnessError(
            f"unsupported WAMR AOT version {version} in {path}; "
            f"expected {AOT_VERSION}"
        )
    position = 8
    while position + 8 <= len(data):
        section_type, section_size = struct.unpack_from("<II", data, position)
        position += 8
        if position + section_size > len(data):
            raise HarnessError(f"truncated AOT section: {path}")
        if section_type == 2:
            return data[position : position + section_size]
        position += section_size
    raise HarnessError(f"AOT text section missing: {path}")


def aot_artifact_report(
    artifacts: dict[str, Path],
    arch: str,
) -> dict[str, Any]:
    text_sections = {
        name: aot_text_section(path) for name, path in artifacts.items()
    }
    report = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "file_bytes": path.stat().st_size,
            "text_bytes": len(text_sections[name]),
        }
        for name, path in sorted(artifacts.items())
    }
    try:
        signature = CANCEL_POLL_SIGNATURES[arch]
    except KeyError as exc:
        raise HarnessError(f"unsupported cancel-poll architecture: {arch}") from exc
    on_text = text_sections["threaded-polls-on"]
    off_text = text_sections["threaded-polls-off"]
    sites_enabled = on_text.count(signature)
    sites_disabled = off_text.count(signature)
    on_bytes = report["threaded-polls-on"]["text_bytes"]
    off_bytes = report["threaded-polls-off"]["text_bytes"]
    delta = on_bytes - off_bytes
    if sites_enabled <= 0 or sites_disabled != 0:
        raise HarnessError(
            f"cancel-poll signature count is invalid for {arch}: "
            f"enabled={sites_enabled}, disabled={sites_disabled}"
        )
    if delta <= 0 or delta % sites_enabled != 0:
        raise HarnessError(
            f"cancel-poll text delta {delta} cannot be attributed to "
            f"{sites_enabled} detected sites for {arch}"
        )
    poll_bytes = delta // sites_enabled
    report["cancel_poll_static"] = {
        "architecture": arch,
        "detection": "machine-code-signature",
        "signature_hex": signature.hex(),
        "bytes_per_site": poll_bytes,
        "text_delta_bytes": delta,
        "sites_enabled": sites_enabled,
        "sites_disabled": sites_disabled,
    }
    return report


@functools.lru_cache(maxsize=None)
def hot_kernel(seed: int, iterations: int) -> int:
    value = seed & MASK64
    for index in range(iterations):
        value = (((value << 7) & MASK64) | (value >> 57))
        value ^= (index + 0xD1B54A32D192ED03) & MASK64
    return value & MASK64


def worker_seed(index: int) -> int:
    return (
        0x243F6A8885A308D3 ^ (0x9E3779B97F4A7C15 * (index + 1))
    ) & MASK64


@functools.lru_cache(maxsize=None)
def expected_result(workload: str, threads: int, iterations: int) -> dict[str, int | str]:
    operations = threads * iterations
    if workload == "single-hot":
        checksum = hot_kernel(worker_seed(0), iterations)
    elif workload == "hot":
        checksum = sum(
            hot_kernel(worker_seed(index), iterations) for index in range(threads)
        ) & MASK64
    elif workload in ("atomic", "wait-notify"):
        checksum = operations
    elif workload == "spawn-join":
        checksum = iterations * threads * (threads + 1) // 2
    else:
        raise HarnessError(f"unknown workload {workload}")
    return {
        "kind": "wasi-thread-benchmark-result",
        "workload": workload,
        "threads": threads,
        "iterations": iterations,
        "operations": operations,
        "checksum": checksum,
        "clock_id": "wasi-monotonic",
        "metric_kind": (
            "spawn-join-lifecycle"
            if workload == "spawn-join"
            else "steady-state-kernel"
        ),
        "timed_loop_backedges": (
            operations if workload in ("single-hot", "hot") else 0
        ),
        "clock_calls_in_timed_loop": 0,
    }


def parse_guest_result(
    stdout: str,
    expected: dict[str, int | str],
    min_interval_ns: int,
) -> dict[str, int | str]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise HarnessError(f"expected one guest JSON line, got {len(lines)}")
    try:
        result = json.loads(lines[0], object_pairs_hook=_reject_guest_keys)
    except json.JSONDecodeError as exc:
        raise HarnessError(f"guest output is not JSON: {lines[0]!r}") from exc
    timing_keys = {
        "raw_elapsed_ns",
        "timing_overhead_ns",
        "elapsed_ns",
        "timing_overhead_ppm",
    }
    if set(result) != set(expected) | timing_keys:
        raise HarnessError(
            "guest timing fields mismatch: "
            f"expected {sorted(set(expected) | timing_keys)}, "
            f"got {sorted(result)}"
        )
    for key, value in expected.items():
        if result.get(key) != value:
            raise HarnessError(
                f"guest result mismatch for {key}: "
                f"expected {value!r}, got {result.get(key)!r}"
            )
    for key in timing_keys:
        if not isinstance(result.get(key), int) or isinstance(result[key], bool):
            raise HarnessError(f"guest timing {key} must be an integer")
    raw = result["raw_elapsed_ns"]
    overhead = result["timing_overhead_ns"]
    elapsed = result["elapsed_ns"]
    overhead_ppm = result["timing_overhead_ppm"]
    if raw <= 0 or overhead < 0 or elapsed <= 0:
        raise HarnessError("guest timing values must be positive")
    if raw - overhead != elapsed:
        raise HarnessError("guest elapsed_ns must equal raw_elapsed_ns - overhead")
    expected_ppm = overhead * 1_000_000 // raw
    if overhead_ppm != expected_ppm:
        raise HarnessError("guest timing_overhead_ppm is inconsistent")
    if overhead_ppm >= 10_000:
        raise HarnessError(
            f"guest timing overhead {overhead_ppm / 10_000:.3f}% is not below 1%"
        )
    if elapsed < min_interval_ns:
        raise HarnessError(
            f"guest timed interval {elapsed}ns is below required {min_interval_ns}ns"
        )
    return result


def _reject_guest_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"guest output contains duplicate key {key!r}")
        result[key] = value
    return result


def run_process(command: list[str], cwd: Path, timeout: float) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        stdout, stderr = proc.communicate()
        raise HarnessError(
            f"command timed out after {timeout:g}s: {' '.join(command)}"
        ) from exc
    try:
        return (
            proc.returncode,
            stdout.decode("UTF-8"),
            stderr.decode("UTF-8"),
        )
    except UnicodeDecodeError as exc:
        raise HarnessError(
            f"command produced non-UTF-8 output: {' '.join(command)}"
        ) from exc


def classify_guest_failure(stderr: str, workload: str) -> str:
    if "outcome=backend-error" in stderr:
        return "atomic-wait-backend-error"
    if "outcome=unexpected-timeout" in stderr or "failed: 11" in stderr:
        return "atomic-wait-unexpected-timeout"
    if "outcome=cancelled" in stderr:
        return "atomic-wait-cancelled"
    if "outcome=closed" in stderr:
        return "atomic-wait-closed"
    if "failed: 13" in stderr:
        return "barrier-value-mismatch"
    if "failed: 12" in stderr:
        return "atomic-wait-invalid-result"
    if "failed: 10" in stderr:
        return "barrier-peer-abort"
    if "worker[" in stderr and "failed:" in stderr:
        return "barrier-unclassified-worker-failure"
    return f"{workload}-guest-failure"


def measure_once(
    *,
    repo: Path,
    runner: list[str],
    build: Build,
    module: Path,
    workload: str,
    threads: int,
    iterations: int,
    timeout: float,
    min_interval_ns: int,
    record_fields: dict[str, Any],
) -> dict[str, Any]:
    guest_args = (
        [str(iterations)]
        if workload == "single-hot"
        else [workload, str(threads), str(iterations)]
    )
    command = [*runner, str(build.wamr), "run", str(module), *guest_args]
    started = time.perf_counter_ns()
    try:
        returncode, stdout, stderr = run_process(command, repo, timeout)
    except HarnessError as exc:
        if "command timed out" not in str(exc):
            raise
        classification = (
            "notification-loss"
            if workload == "wait-notify"
            else "guest-watchdog-timeout"
        )
        raise HarnessError(
            f"guest failure classification={classification}: {exc}"
        ) from exc
    host_wall_elapsed_ns = time.perf_counter_ns() - started
    if returncode != 0:
        classification = classify_guest_failure(stderr, workload)
        raise HarnessError(
            f"guest failure classification={classification}; "
            f"exit {returncode}: {' '.join(command)}\n{stderr}"
        )
    expected = expected_result(workload, threads, iterations)
    guest = parse_guest_result(stdout, expected, min_interval_ns)
    operations = int(guest["operations"])
    guest_elapsed_ns = int(guest["elapsed_ns"])
    throughput = operations / (guest_elapsed_ns / 1e9)
    cancel_points = record_fields.get("cancel_points")
    cancel_polls_per_operation: float | None = None
    if workload in ("single-hot", "hot"):
        if cancel_points == "on":
            cancel_polls_per_operation = 1.0
        elif cancel_points in ("off", "not-applicable"):
            cancel_polls_per_operation = 0.0
    return {
        **record_fields,
        "command": command,
        "elapsed_ns": guest_elapsed_ns,
        "guest_elapsed_ns": guest_elapsed_ns,
        "raw_guest_elapsed_ns": int(guest["raw_elapsed_ns"]),
        "timing_overhead_ns": int(guest["timing_overhead_ns"]),
        "timing_overhead_ppm": int(guest["timing_overhead_ppm"]),
        "host_wall_elapsed_ns": host_wall_elapsed_ns,
        "host_wall_over_guest": host_wall_elapsed_ns / guest_elapsed_ns,
        "metric_kind": guest["metric_kind"],
        "cancel_polls_per_operation": cancel_polls_per_operation,
        "operations": operations,
        "throughput_ops_per_second": throughput,
        "per_thread_ops_per_second": throughput / threads,
        "guest": guest,
        "correct": True,
        "correctness": {
            "passed": True,
            "expected": expected,
            "actual": guest,
        },
        "stdout": stdout,
        "stderr": stderr,
    }


def collect_pair(
    *,
    records: list[dict[str, Any]],
    pair_kind: str,
    pair_key: str,
    left: str,
    right: str,
    warmups: int,
    samples: int,
    measure: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    total = warmups + samples
    for index in range(total):
        phase = "warmup" if index < warmups else "measure"
        phase_index = index if phase == "warmup" else index - warmups
        order = alternating_pair_order(index, left, right)
        for order_index, condition in enumerate(order):
            record = measure(
                condition,
                {
                    "pair_kind": pair_kind,
                    "pair_key": pair_key,
                    "pair_index": phase_index,
                    "phase": phase,
                    "order": order_index,
                    "condition": condition,
                    "pair_left": left,
                    "pair_right": right,
                },
            )
            records.append(record)
            print(
                f"[thread-bench] {pair_key} {phase} {phase_index + 1}/"
                f"{warmups if phase == 'warmup' else samples} {condition}: "
                f"guest={record['guest_elapsed_ns'] / 1e6:.3f} ms "
                f"host={record['host_wall_elapsed_ns'] / 1e6:.3f} ms",
                file=sys.stderr,
            )


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record["phase"] != "measure":
            continue
        key = (record["pair_kind"], record["pair_key"], record["condition"])
        grouped.setdefault(key, []).append(record)
    summaries = []
    for (pair_kind, pair_key, condition), selected in sorted(grouped.items()):
        elapsed = [record["elapsed_ns"] for record in selected]
        host_wall = [record["host_wall_elapsed_ns"] for record in selected]
        throughput = [record["throughput_ops_per_second"] for record in selected]
        per_thread = [record["per_thread_ops_per_second"] for record in selected]
        summaries.append(
            {
                "pair_kind": pair_kind,
                "pair_key": pair_key,
                "condition": condition,
                "metric_kind": selected[0]["metric_kind"],
                "elapsed": sample_stats(elapsed, "samples_ns"),
                "host_wall": sample_stats(host_wall, "samples_ns"),
                "throughput": sample_stats(throughput, "samples_ops_per_second"),
                "per_thread_throughput": sample_stats(
                    per_thread, "samples_ops_per_second"
                ),
            }
        )
    return summaries


def paired_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        if record["phase"] != "measure":
            continue
        key = (record["pair_kind"], record["pair_key"], record["pair_index"])
        cells.setdefault(key, {})[record["condition"]] = record
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for (pair_kind, pair_key, _), conditions in cells.items():
        if len(conditions) != 2:
            raise HarnessError(f"incomplete measured pair: {pair_key}")
        pair_records = list(conditions.values())
        left = pair_records[0]["pair_left"]
        right = pair_records[0]["pair_right"]
        if any(
            record["pair_left"] != left or record["pair_right"] != right
            for record in pair_records
        ):
            raise HarnessError(f"inconsistent pair direction: {pair_key}")
        if set(conditions) != {left, right}:
            raise HarnessError(f"pair conditions do not match direction: {pair_key}")
        left_record = conditions[left]
        right_record = conditions[right]
        ratio = right_record["elapsed_ns"] / left_record["elapsed_ns"]
        grouped.setdefault((pair_kind, pair_key, left, right), []).append(ratio)
    result = []
    for (pair_kind, pair_key, left, right), ratios in sorted(grouped.items()):
        result.append(
            {
                "pair_kind": pair_kind,
                "pair_key": pair_key,
                "left": left,
                "right": right,
                "elapsed_right_over_left": sample_stats(ratios, "samples"),
                "median_elapsed_delta_pct": (statistics.median(ratios) - 1) * 100,
            }
        )
    return result


def validate_report(document: dict[str, Any]) -> None:
    validate_common_report(document, KIND)
    metadata = document["metadata"]
    require(
        isinstance(metadata.get("platform_id"), str) and metadata["platform_id"],
        "metadata.platform_id",
    )
    for key in ("fixture_set_sha256", "plan_sha256"):
        require(
            isinstance(metadata.get(key), str) and len(metadata[key]) == 64,
            f"metadata.{key}",
        )
    plan = document["plan"]
    require(plan.get("warmups", -1) >= 0, "plan.warmups")
    require(plan.get("samples", 0) > 0, "plan.samples")
    require(
        plan.get("minimum_timed_interval_ns", 0) > 0,
        "plan.minimum_timed_interval_ns",
    )
    require(metadata["plan_sha256"] == cache_key(plan), "metadata.plan_sha256")
    pair_plan = plan.get("pairs")
    require(isinstance(pair_plan, list) and pair_plan, "plan.pairs")
    expected_pair_plan = expected_pair_specs_for_plan(plan)
    require(pair_plan == expected_pair_plan, "plan.pairs is incomplete or reordered")
    pair_by_key: dict[str, dict[str, str]] = {}
    for pair in pair_plan:
        require(isinstance(pair, dict), "plan pair object")
        pair_key = pair.get("pair_key")
        require(isinstance(pair_key, str) and pair_key, "plan pair key")
        require(pair_key not in pair_by_key, f"duplicate plan pair {pair_key}")
        require(pair.get("left") != pair.get("right"), f"pair direction {pair_key}")
        pair_by_key[pair_key] = pair

    seen: set[tuple[str, str, int, str]] = set()
    per_cell: dict[tuple[str, str, int], set[str]] = {}
    for record in document["records"]:
        require(record.get("phase") in ("warmup", "measure"), "record phase")
        require(record.get("correct") is True, "record correctness")
        require(record.get("guest_elapsed_ns", 0) > 0, "record guest elapsed")
        require(
            record.get("host_wall_elapsed_ns", 0) >= record["guest_elapsed_ns"],
            "record host wall diagnostic",
        )
        require(record.get("timing_overhead_ppm", 10_000) < 10_000, "timing overhead")
        require(
            record["guest_elapsed_ns"] >= plan["minimum_timed_interval_ns"],
            "record minimum timed interval",
        )
        pair_key = record.get("pair_key")
        require(pair_key in pair_by_key, f"unknown record pair {pair_key}")
        pair = pair_by_key[pair_key]
        require(record.get("pair_kind") == pair["pair_kind"], "record pair kind")
        require(record.get("pair_left") == pair["left"], "record pair left")
        require(record.get("pair_right") == pair["right"], "record pair right")
        require(
            record.get("condition") in (pair["left"], pair["right"]),
            "record pair condition",
        )
        key = (
            pair_key,
            record["condition"],
            record["pair_index"],
            record["phase"],
        )
        require(key not in seen, f"duplicate record {key}")
        seen.add(key)
        cell = (pair_key, record["phase"], record["pair_index"])
        per_cell.setdefault(cell, set()).add(record["condition"])
    expected_records_per_pair = 2 * (plan["warmups"] + plan["samples"])
    pair_counts: dict[str, int] = {}
    for record in document["records"]:
        pair_counts[record["pair_key"]] = pair_counts.get(record["pair_key"], 0) + 1
    require(
        set(pair_counts) == set(pair_by_key),
        "report is missing planned pairs",
    )
    require(
        all(
            pair_counts[pair_key] == expected_records_per_pair
            for pair_key in pair_by_key
        ),
        "incomplete sample pairing",
    )
    for pair_key, pair in pair_by_key.items():
        for phase, count in (
            ("warmup", plan["warmups"]),
            ("measure", plan["samples"]),
        ):
            for index in range(count):
                require(
                    per_cell.get((pair_key, phase, index))
                    == {pair["left"], pair["right"]},
                    f"incomplete pair cell {pair_key}/{phase}/{index}",
                )

    expected_summary_keys = {
        (pair_key, condition)
        for pair_key, pair in pair_by_key.items()
        for condition in (pair["left"], pair["right"])
    }
    actual_summary_keys = {
        (summary.get("pair_key"), summary.get("condition"))
        for summary in document["summaries"]
    }
    require(
        actual_summary_keys == expected_summary_keys,
        "summaries do not cover every planned condition",
    )
    expected_paired_keys = set(pair_by_key)
    actual_paired_keys = {
        summary.get("pair_key") for summary in document["paired_summaries"]
    }
    require(
        actual_paired_keys == expected_paired_keys,
        "paired summaries do not cover every planned pair",
    )
    for summary in document["paired_summaries"]:
        pair = pair_by_key[summary["pair_key"]]
        require(summary.get("left") == pair["left"], "paired summary left")
        require(summary.get("right") == pair["right"], "paired summary right")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"budget contains duplicate key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise HarnessError(
            f"{label} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )


def load_budget(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    try:
        budget = json.loads(
            path.read_text(encoding="UTF-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"invalid budget {path}: {exc}") from exc
    if not isinstance(budget, dict):
        raise HarnessError("budget must be an object")
    _require_exact_keys(
        budget,
        {
            "schema_version",
            "kind",
            "calibrated",
            "calibration_requirements",
            "cohort",
            "platforms",
        },
        "budget",
    )
    if budget["schema_version"] != SCHEMA_VERSION or budget["kind"] != "wasi-thread-benchmark-budget":
        raise HarnessError("budget schema/kind mismatch")
    if budget["calibrated"] is not True:
        raise HarnessError(
            "budget is not calibrated; collect retained hosted reports and "
            "run with --no-budget"
        )
    requirements = budget["calibration_requirements"]
    cohort = budget["cohort"]
    platforms = budget["platforms"]
    if not isinstance(requirements, dict) or not isinstance(cohort, dict) or not isinstance(platforms, dict):
        raise HarnessError("budget requirements/cohort/platforms must be objects")
    _require_exact_keys(
        requirements,
        {
            "minimum_reports_per_platform",
            "required_profile",
            "required_platforms",
        },
        "calibration_requirements",
    )
    minimum_reports = requirements["minimum_reports_per_platform"]
    required_platforms = requirements["required_platforms"]
    if not isinstance(minimum_reports, int) or minimum_reports < 20:
        raise HarnessError("budget minimum_reports_per_platform must be >= 20")
    if requirements["required_profile"] != "authoritative":
        raise HarnessError("budget required_profile must be authoritative")
    if (
        not isinstance(required_platforms, list)
        or len(required_platforms) != len(CANONICAL_PLATFORMS)
        or len(set(required_platforms)) != len(required_platforms)
        or set(required_platforms) != set(CANONICAL_PLATFORMS)
    ):
        raise HarnessError(
            "budget required_platforms must be exactly the canonical hosted platforms"
        )
    _require_exact_keys(
        cohort,
        {
            "baseline_commit",
            "baseline_build_source_sha256",
            "fixture_set_sha256",
            "plan_sha256",
            "profile",
            "report_count_by_platform",
        },
        "cohort",
    )
    for key, length in (
        ("baseline_commit", 40),
        ("baseline_build_source_sha256", 64),
        ("fixture_set_sha256", 64),
        ("plan_sha256", 64),
    ):
        value = cohort[key]
        if (
            not isinstance(value, str)
            or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
        ):
            raise HarnessError(f"budget cohort.{key} has invalid identity")
    if cohort["profile"] != requirements["required_profile"]:
        raise HarnessError("budget cohort profile mismatch")
    counts = cohort["report_count_by_platform"]
    if not isinstance(counts, dict) or set(counts) != set(required_platforms):
        raise HarnessError("budget cohort report counts do not cover platforms")
    if any(not isinstance(counts[item], int) or counts[item] < minimum_reports for item in required_platforms):
        raise HarnessError("budget cohort has insufficient retained reports")
    if set(platforms) != set(required_platforms):
        raise HarnessError("budget platforms are incomplete or unknown")

    metadata = report["metadata"]
    identity_checks = {
        "baseline_commit": metadata["commit"],
        "baseline_build_source_sha256": metadata["build_source_sha256"],
        "fixture_set_sha256": metadata["fixture_set_sha256"],
        "plan_sha256": metadata["plan_sha256"],
        "profile": report["plan"]["profile"],
    }
    for key, actual in identity_checks.items():
        if cohort[key] != actual:
            raise HarnessError(
                f"budget/report identity mismatch for {key}: "
                f"{cohort[key]!r} != {actual!r}"
            )
    platform_id = metadata["platform_id"]
    if platform_id not in platforms:
        raise HarnessError(f"budget has no platform {platform_id!r}")
    platform_budget = platforms[platform_id]
    if not isinstance(platform_budget, dict):
        raise HarnessError("platform budget must be an object")
    _require_exact_keys(
        platform_budget,
        {"host_system", "host_machine", "pairs", "scenarios"},
        f"platforms.{platform_id}",
    )
    host = metadata["host"]
    expected_host = CANONICAL_PLATFORMS[platform_id]
    if (host["system"], host["machine"]) != expected_host:
        raise HarnessError("report platform ID does not match its canonical host identity")
    if platform_budget["host_system"] != host["system"] or platform_budget["host_machine"] != host["machine"]:
        raise HarnessError("budget/report host platform mismatch")

    pair_plan = {item["pair_key"]: item for item in report["plan"]["pairs"]}
    pair_thresholds = platform_budget["pairs"]
    if not isinstance(pair_thresholds, list):
        raise HarnessError("platform pairs must be an array")
    seen_pairs: set[str] = set()
    for item in pair_thresholds:
        if not isinstance(item, dict):
            raise HarnessError("pair threshold must be an object")
        _require_exact_keys(
            item,
            {"pair_key", "left", "right", "max_median_elapsed_delta_pct"},
            "pair threshold",
        )
        pair_key = item["pair_key"]
        if pair_key in seen_pairs:
            raise HarnessError(f"duplicate pair threshold {pair_key}")
        seen_pairs.add(pair_key)
        if pair_key not in pair_plan:
            raise HarnessError(f"unknown pair threshold {pair_key}")
        planned = pair_plan[pair_key]
        if item["left"] != planned["left"] or item["right"] != planned["right"]:
            raise HarnessError(f"pair threshold direction mismatch for {pair_key}")
        if not isinstance(item["max_median_elapsed_delta_pct"], (int, float)):
            raise HarnessError(f"pair threshold limit must be numeric for {pair_key}")
    if seen_pairs != set(pair_plan):
        raise HarnessError("pair thresholds do not cover the complete report plan")

    expected_scenarios = {
        (item["pair_key"], item["condition"]): item
        for item in report["summaries"]
    }
    scenario_thresholds = platform_budget["scenarios"]
    if not isinstance(scenario_thresholds, list):
        raise HarnessError("platform scenarios must be an array")
    seen_scenarios: set[tuple[str, str]] = set()
    for item in scenario_thresholds:
        if not isinstance(item, dict):
            raise HarnessError("scenario threshold must be an object")
        _require_exact_keys(
            item,
            {
                "pair_key",
                "condition",
                "metric_kind",
                "min_median_ops_per_second",
            },
            "scenario threshold",
        )
        key = (item["pair_key"], item["condition"])
        if key in seen_scenarios:
            raise HarnessError(f"duplicate scenario threshold {key}")
        seen_scenarios.add(key)
        if key not in expected_scenarios:
            raise HarnessError(f"unknown scenario threshold {key}")
        if item["metric_kind"] != expected_scenarios[key]["metric_kind"]:
            raise HarnessError(f"scenario metric mismatch for {key}")
        minimum = item["min_median_ops_per_second"]
        if not isinstance(minimum, (int, float)) or minimum <= 0:
            raise HarnessError(f"scenario minimum must be positive for {key}")
    if seen_scenarios != set(expected_scenarios):
        raise HarnessError("scenario thresholds do not cover every report summary")
    return platform_budget


def evaluate_budget(
    platform_budget: dict[str, Any],
    summaries: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    pair_lookup = {item["pair_key"]: item for item in pairs}
    for threshold in platform_budget["pairs"]:
        pair_key = threshold["pair_key"]
        actual = pair_lookup[pair_key]
        limit = float(threshold["max_median_elapsed_delta_pct"])
        if actual["median_elapsed_delta_pct"] > limit:
            failures.append(
                f"{pair_key}: {actual['median_elapsed_delta_pct']:+.2f}% "
                f"> {limit:+.2f}%"
            )
    summary_lookup = {
        f"{item['pair_key']}::{item['condition']}": item for item in summaries
    }
    for threshold in platform_budget["scenarios"]:
        key = f"{threshold['pair_key']}::{threshold['condition']}"
        minimum = float(threshold["min_median_ops_per_second"])
        actual = summary_lookup[key]
        if actual["throughput"]["median"] < minimum:
            failures.append(
                f"{key}: {actual['throughput']['median']:.3f} "
                f"< {minimum:.3f}"
            )
    return failures


def build_tool_report(builds: dict[str, Build], runner: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, build in sorted(builds.items()):
        binary_runner = [] if name == "host-compiler" else runner
        entry = {
            "mode": build.mode,
            "threads_enabled": build.threads_enabled,
            "cache_key": build.key,
            "cache_reused": build.reused,
            "build_command": build.command,
            "wamr": {
                "path": str(build.wamr),
                "sha256": sha256_file(build.wamr),
                "version": command_identity(
                    [*binary_runner, str(build.wamr), "version"]
                ),
            },
        }
        if build.wamrc is not None:
            entry["wamrc"] = {
                "path": str(build.wamrc),
                "sha256": sha256_file(build.wamrc),
                "version": command_identity(
                    [*binary_runner, str(build.wamrc), "version"]
                ),
            }
        report[name] = entry
    return report


def render_markdown(document: dict[str, Any]) -> str:
    host = document["metadata"]["host"]
    lines = [
        "# WASI threaded benchmark",
        "",
        f"- Commit: `{document['metadata']['commit']}`",
        f"- Platform identity: `{document['metadata']['platform_id']}`",
        f"- Host: `{host['system']} {host['release']}` · `{host['machine']}` · "
        f"{host['logical_cpus']} CPUs · `{host['cpu']}`",
        f"- Profile: `{document['plan']['profile']}` "
        f"({document['plan']['warmups']} warmups, {document['plan']['samples']} samples)",
        f"- Budget: `{document['budget']['status']}`",
    ]
    poll_static = document["metadata"].get("aot_artifacts", {}).get(
        "cancel_poll_static"
    )
    if poll_static:
        lines.append(
            f"- AOT cancel polls: {poll_static['sites_enabled']} static sites, "
            f"{poll_static['bytes_per_site']} bytes/site; timed hot kernel "
            "executes 1 poll opportunity per operation when enabled"
        )
    lines += [
        "",
        "| Pair | Condition | Metric | Guest median ms | Host-wall median ms | Guest range ms | Median aggregate ops/s | Median per-thread ops/s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in document["summaries"]:
        elapsed = item["elapsed"]
        host_wall = item["host_wall"]
        throughput = item["throughput"]
        per_thread = item["per_thread_throughput"]
        lines.append(
            f"| `{item['pair_key']}` | `{item['condition']}` | "
            f"`{item['metric_kind']}` | "
            f"{elapsed['median'] / 1e6:.3f} | "
            f"{host_wall['median'] / 1e6:.3f} | "
            f"{elapsed['min'] / 1e6:.3f}–{elapsed['max'] / 1e6:.3f} | "
            f"{throughput['median']:.3f} | {per_thread['median']:.3f} |"
        )
    lines += [
        "",
        "| Paired comparison | Right / left guest-time median delta | Raw paired ratios |",
        "|---|---:|---|",
    ]
    for item in document["paired_summaries"]:
        samples = ", ".join(
            f"{value:.4f}" for value in item["elapsed_right_over_left"]["samples"]
        )
        lines.append(
            f"| `{item['pair_key']}`: `{item['right']}` / `{item['left']}` | "
            f"{item['median_elapsed_delta_pct']:+.2f}% | {samples} |"
        )
    lines += [
        "",
        "Kernel throughput uses guest monotonic time corrected by a same-process "
        "barrier/timer calibration below 1%; host wall time is diagnostic only. "
        "Spawn/join is reported separately as a guest-timed lifecycle metric.",
        "",
        "Every warmup and measured invocation passed its deterministic timing, "
        "checksum, operation count, workload, thread-count, and iteration assertions.",
        "",
    ]
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (repo / args.output_dir).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    source = source_identity(repo)
    fixture_report = resolve_fixtures(repo)
    fixture_set_sha256 = cache_key(
        {
            name: {
                "path": item["path"],
                "sha256": item["sha256"],
            }
            for name, item in sorted(fixture_report.items())
        }
    )
    minimum_interval_ns = int(args.min_interval_ms * 1_000_000)
    modes = ("interpreter", "aot") if args.modes == "both" else (args.modes,)
    pair_plan = planned_pair_specs(args, modes)
    runner = shlex.split(args.runner)
    builds: dict[str, Build] = {}
    for mode in modes:
        for enabled in (False, True):
            build = build_variant(
                repo=repo,
                root=output,
                mode=mode,
                threads_enabled=enabled,
                optimize=args.optimize,
                target=args.target,
                source=source,
                rebuild=args.rebuild,
                compiler_toggle=enabled and mode == "aot",
            )
            builds[build.name] = build

    compiler = None
    aot_artifacts: dict[str, Path] = {}
    aot_artifacts_metadata: dict[str, Any] = {}
    if "aot" in modes:
        if args.target:
            compiler = build_variant(
                repo=repo,
                root=output,
                mode="aot",
                threads_enabled=True,
                optimize=args.optimize,
                target=None,
                source=source,
                rebuild=args.rebuild,
                compiler_toggle=True,
            )
            builds["host-compiler"] = compiler
        else:
            compiler = builds["enabled-aot"]
        aot_artifacts = compile_aot_fixtures(
            repo, output, compiler, execution_arch(args)
        )
        aot_artifacts_metadata = aot_artifact_report(
            aot_artifacts, execution_arch(args)
        )

    records: list[dict[str, Any]] = []
    single_wasm = repo / FIXTURES["single"]["path"]
    threaded_wasm = repo / FIXTURES["threaded"]["path"]

    for mode in modes:
        disabled = builds[f"disabled-{mode}"]
        enabled = builds[f"enabled-{mode}"]
        module = single_wasm if mode == "interpreter" else aot_artifacts["single"]

        def single_measure(
            condition: str, fields: dict[str, Any]
        ) -> dict[str, Any]:
            selected = disabled if condition == "threads-disabled" else enabled
            return measure_once(
                repo=repo,
                runner=runner,
                build=selected,
                module=module,
                workload="single-hot",
                threads=1,
                iterations=args.single_iterations,
                timeout=args.timeout,
                min_interval_ns=minimum_interval_ns,
                record_fields={
                    **fields,
                    "mode": mode,
                    "threads_enabled": selected.threads_enabled,
                    "cancel_points": "not-applicable",
                    "static_cancel_poll_sites": 0,
                    "workload": "single-hot",
                    "threads": 1,
                    "iterations": args.single_iterations,
                },
            )

        collect_pair(
            records=records,
            pair_kind="single-infrastructure",
            pair_key=f"single-infrastructure/{mode}",
            left="threads-disabled",
            right="threads-enabled",
            warmups=args.warmups,
            samples=args.samples,
            measure=single_measure,
        )

    scenarios = planned_scenarios(args)
    for scenario in scenarios:
        if len(modes) == 2:
            def runtime_measure(
                condition: str, fields: dict[str, Any]
            ) -> dict[str, Any]:
                mode = condition
                build = builds[f"enabled-{mode}"]
                module = (
                    threaded_wasm
                    if mode == "interpreter"
                    else aot_artifacts["threaded-polls-on"]
                )
                return measure_once(
                    repo=repo,
                    runner=runner,
                    build=build,
                    module=module,
                    workload=scenario.workload,
                    threads=scenario.threads,
                    iterations=scenario.iterations,
                    timeout=args.timeout,
                    min_interval_ns=minimum_interval_ns,
                    record_fields={
                        **fields,
                        "mode": mode,
                        "threads_enabled": True,
                        "cancel_points": (
                            "on" if mode == "aot" else "interpreter-dispatch"
                        ),
                        "static_cancel_poll_sites": (
                            aot_artifacts_metadata["cancel_poll_static"]["sites_enabled"]
                            if mode == "aot"
                            else None
                        ),
                        "workload": scenario.workload,
                        "threads": scenario.threads,
                        "iterations": scenario.iterations,
                    },
                )

            collect_pair(
                records=records,
                pair_kind="runtime-parity",
                pair_key=f"runtime/{scenario.key}",
                left="interpreter",
                right="aot",
                warmups=args.warmups,
                samples=args.samples,
                measure=runtime_measure,
            )
        else:
            mode = modes[0]
            build = builds[f"enabled-{mode}"]
            module = (
                threaded_wasm
                if mode == "interpreter"
                else aot_artifacts["threaded-polls-on"]
            )

            def single_mode_measure(
                condition: str, fields: dict[str, Any]
            ) -> dict[str, Any]:
                return measure_once(
                    repo=repo,
                    runner=runner,
                    build=build,
                    module=module,
                    workload=scenario.workload,
                    threads=scenario.threads,
                    iterations=scenario.iterations,
                    timeout=args.timeout,
                    min_interval_ns=minimum_interval_ns,
                    record_fields={
                        **fields,
                        "mode": mode,
                        "threads_enabled": True,
                        "cancel_points": (
                            "on" if mode == "aot" else "interpreter-dispatch"
                        ),
                        "static_cancel_poll_sites": (
                            aot_artifacts_metadata["cancel_poll_static"]["sites_enabled"]
                            if mode == "aot"
                            else None
                        ),
                        "workload": scenario.workload,
                        "threads": scenario.threads,
                        "iterations": scenario.iterations,
                    },
                )

            collect_pair(
                records=records,
                pair_kind="repeatability",
                pair_key=f"runtime/{scenario.key}/{mode}",
                left=f"{mode}-a",
                right=f"{mode}-b",
                warmups=args.warmups,
                samples=args.samples,
                measure=single_mode_measure,
            )

    if "aot" in modes:
        aot_build = builds["enabled-aot"]
        for threads in args.thread_counts:
            scenario = Scenario("hot", threads, args.hot_iterations)

            def poll_measure(
                condition: str, fields: dict[str, Any]
            ) -> dict[str, Any]:
                polls = "off" if condition == "cancel-points-off" else "on"
                module = aot_artifacts[f"threaded-polls-{polls}"]
                return measure_once(
                    repo=repo,
                    runner=runner,
                    build=aot_build,
                    module=module,
                    workload="hot",
                    threads=threads,
                    iterations=args.hot_iterations,
                    timeout=args.timeout,
                    min_interval_ns=minimum_interval_ns,
                    record_fields={
                        **fields,
                        "mode": "aot",
                        "threads_enabled": True,
                        "cancel_points": polls,
                        "static_cancel_poll_sites": (
                            aot_artifacts_metadata["cancel_poll_static"][
                                "sites_enabled"
                            ]
                            if polls == "on"
                            else 0
                        ),
                        "workload": "hot",
                        "threads": threads,
                        "iterations": args.hot_iterations,
                    },
                )

            collect_pair(
                records=records,
                pair_kind="cancel-point-cost",
                pair_key=f"cancel-points/hot/{threads}",
                left="cancel-points-off",
                right="cancel-points-on",
                warmups=args.warmups,
                samples=args.samples,
                measure=poll_measure,
            )

    summaries = summarize(records)
    pairs = paired_summaries(records)
    plan = {
        "profile": args.profile,
        "warmups": args.warmups,
        "samples": args.samples,
        "modes": list(modes),
        "thread_counts": list(args.thread_counts),
        "iterations": {
            "single-hot": args.single_iterations,
            "hot": args.hot_iterations,
            "atomic": args.atomic_iterations,
            "wait-notify": args.wait_iterations,
            "spawn-join": args.spawn_iterations,
        },
        "timeout_seconds": args.timeout,
        "minimum_timed_interval_ns": minimum_interval_ns,
        "optimize": args.optimize,
        "pairs": pair_plan,
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "metadata": {
            **source,
            "collected_at": collected_at(),
            "platform_id": args.platform_id,
            "fixture_set_sha256": fixture_set_sha256,
            "plan_sha256": cache_key(plan),
            "host": host_metadata(),
            "execution": {
                "target": args.target or "native",
                "aot_target": execution_arch(args),
                "runner": runner,
            },
            "tools": build_tool_report(builds, runner),
            "fixture_toolchain": WASI_SDK,
            "fixtures": fixture_report,
            "aot_artifacts": aot_artifacts_metadata,
        },
        "plan": plan,
        "records": records,
        "summaries": summaries,
        "paired_summaries": pairs,
        "budget": {
            "status": "disabled" if args.no_budget else "not-selected",
            "path": str(args.budget.resolve()) if args.budget else None,
            "failures": [],
        },
    }
    validate_report(document)
    budget_failures: list[str] = []
    if args.budget:
        platform_budget = load_budget(args.budget.resolve(), document)
        budget_failures = evaluate_budget(platform_budget, summaries, pairs)
        document["budget"]["status"] = (
            "passed" if not budget_failures else "failed"
        )
        document["budget"]["failures"] = budget_failures
    atomic_write_json(output / "report.json", document)
    (output / "report.md").write_text(
        render_markdown(document) + "\n", encoding="UTF-8"
    )
    if budget_failures:
        raise HarnessError("budget failures: " + "; ".join(budget_failures))
    return document


def main(argv: list[str] | None = None) -> int:
    try:
        document = execute(parse_args(argv))
    except (OSError, subprocess.CalledProcessError, BenchmarkDataError, HarnessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(render_markdown(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
