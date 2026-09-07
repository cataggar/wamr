#!/usr/bin/env python3
"""Run reproducible paired WASI pthread, atomic, and cancel-poll benchmarks."""

from __future__ import annotations

import argparse
import copy
import functools
import json
import math
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
    HOST_FINGERPRINT_FIELDS,
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
REVISION_ROLES = ("baseline", "candidate")
SINGLE_REVISION_ROLES = ("candidate",)
COMPARISON_PURPOSES = ("candidate-evaluation", "noise-calibration")
MEASUREMENT_PLAN_IDENTITY_VERSION = 1
MEASUREMENT_PLAN_IDENTITY_KIND = "wasi-thread-measurement-plan"
PROFILE_COUNTS = {
    "authoritative": (2, 10),
    "smoke": (1, 4),
}
ATOMIC_WAIT_PREFLIGHT_RUNS = {
    "authoritative": 64,
    "smoke": 8,
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
        "sha256": "27e0ec911816a8dd62519f317e749af547dc7af65c9b939fe7ba0452de74cdb0",
    },
}
MASK64 = (1 << 64) - 1


def measurement_plan_sha256(plan: dict[str, Any]) -> str:
    """Hash every measurement-plan field except comparison purpose."""

    require(
        isinstance(plan, dict) and "comparison_purpose" in plan,
        "measurement plan comparison_purpose",
    )
    normalized = copy.deepcopy(plan)
    del normalized["comparison_purpose"]
    return cache_key(
        {
            "schema_version": MEASUREMENT_PLAN_IDENTITY_VERSION,
            "kind": MEASUREMENT_PLAN_IDENTITY_KIND,
            "plan_without_comparison_purpose": normalized,
        }
    )


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
            (
                iterations // threads
                if workload == "wait-notify"
                else atomic_iterations(args, threads)
                if workload == "atomic"
                else iterations
            ),
        )
        for workload, iterations in (
            ("hot", args.hot_iterations),
            ("atomic", args.atomic_iterations),
            ("wait-notify", args.wait_iterations),
            ("spawn-join", args.spawn_iterations),
        )
        for threads in args.thread_counts
    ]


def cancel_iterations(args: argparse.Namespace, threads: int) -> int:
    return max(args.hot_iterations, args.cancel_iterations // threads)


def atomic_iterations(args: argparse.Namespace, threads: int) -> int:
    return max(args.atomic_iterations, args.atomic_total_iterations // threads)


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
        "--baseline-repo",
        type=Path,
        help="immutable baseline checkout (requires --candidate-repo)",
    )
    parser.add_argument(
        "--candidate-repo",
        type=Path,
        help="immutable candidate checkout (requires --baseline-repo)",
    )
    parser.add_argument(
        "--comparison-purpose",
        choices=COMPARISON_PURPOSES,
        help="required purpose for paired baseline/candidate checkouts",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("zig-out/wasi-thread-bench")
    )
    parser.add_argument("--profile", choices=PROFILE_COUNTS, default="authoritative")
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--modes", choices=("both", "interpreter", "aot"), default="both")
    parser.add_argument("--thread-counts", type=parse_thread_counts, default=(1, 2, 4, 8))
    parser.add_argument("--single-iterations", type=int, default=224_000_000)
    parser.add_argument("--cancel-iterations", type=int, default=224_000_000)
    parser.add_argument("--hot-iterations", type=int, default=128_000_000)
    parser.add_argument("--atomic-iterations", type=int, default=64_000_000)
    parser.add_argument("--atomic-total-iterations", type=int, default=256_000_000)
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
    parser.add_argument(
        "--runner-environment",
        default=None,
        help="stable runner class, for example github-hosted or self-hosted",
    )
    parser.add_argument(
        "--host-pair-id",
        default=None,
        help="identity shared by the baseline/candidate measurements on one host",
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
        "cancel_iterations",
        "hot_iterations",
        "atomic_iterations",
        "atomic_total_iterations",
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
    if (args.baseline_repo is None) != (args.candidate_repo is None):
        parser.error("--baseline-repo and --candidate-repo must be supplied together")
    paired = args.baseline_repo is not None
    if paired and args.comparison_purpose is None:
        parser.error("--comparison-purpose is required for paired revisions")
    if not paired and args.comparison_purpose is not None:
        parser.error("--comparison-purpose requires paired revisions")
    if paired and args.samples % 2 != 0:
        parser.error("paired revision measurements require an even --samples count")
    if args.comparison_purpose == "noise-calibration" and not args.no_budget:
        parser.error("noise calibration requires --no-budget")
    if args.budget and args.comparison_purpose != "candidate-evaluation":
        parser.error("budget enforcement requires paired candidate evaluation")
    if args.host_pair_id is not None and not args.host_pair_id.strip():
        parser.error("--host-pair-id must not be empty")
    if args.runner_environment is not None and not args.runner_environment.strip():
        parser.error("--runner-environment must not be empty")
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


def fixture_set_identity(fixtures: dict[str, dict[str, Any]]) -> str:
    return cache_key(
        {
            name: {
                "path": item["path"],
                "sha256": item["sha256"],
            }
            for name, item in sorted(fixtures.items())
        }
    )


def host_pair_identity(
    platform_id: str,
    host: dict[str, Any],
    explicit: str | None,
) -> dict[str, str]:
    fingerprint = host.get("host_fingerprint")
    require(isinstance(fingerprint, dict), "host.host_fingerprint")
    fingerprint_sha256 = fingerprint.get("sha256")
    require(
        isinstance(fingerprint_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", fingerprint_sha256) is not None,
        "host.host_fingerprint.sha256",
    )
    run_id = host.get("github_run_id", "")
    run_attempt = host.get("github_run_attempt", "")
    pair_id = explicit
    if pair_id is None and run_id:
        pair_id = f"github:{run_id}:{run_attempt or '1'}:{platform_id}"
    if pair_id is None:
        pair_id = f"local:{platform_id}:{fingerprint_sha256[:16]}"
    return {
        "id": pair_id,
        "runner_environment": host["runner_environment"],
        "host_fingerprint_sha256": fingerprint_sha256,
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
    if "controller barrier failed:" in stderr:
        return "controller-barrier-failure"
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


def collect_revision_pair(
    *,
    records: list[dict[str, Any]],
    pair_kind: str,
    pair_key: str,
    left: str,
    right: str,
    warmups: int,
    samples: int,
    revision_roles: tuple[str, ...],
    revision_fields: dict[str, dict[str, Any]],
    measure: Callable[[str, str, dict[str, Any]], dict[str, Any]],
) -> None:
    require(
        revision_roles in (REVISION_ROLES, SINGLE_REVISION_ROLES),
        "revision roles",
    )
    require(set(revision_fields) == set(revision_roles), "revision fields")
    total = warmups + samples
    for index in range(total):
        phase = "warmup" if index < warmups else "measure"
        phase_index = index if phase == "warmup" else index - warmups
        condition_order = alternating_pair_order(index, left, right)
        revision_order = (
            alternating_pair_order(index, *REVISION_ROLES)
            if revision_roles == REVISION_ROLES
            else SINGLE_REVISION_ROLES
        )
        for revision_index, revision in enumerate(revision_order):
            for condition_index, condition in enumerate(condition_order):
                record = measure(
                    revision,
                    condition,
                    {
                        **revision_fields[revision],
                        "revision": revision,
                        "revision_order": revision_index,
                        "pair_kind": pair_kind,
                        "pair_key": pair_key,
                        "pair_index": phase_index,
                        "phase": phase,
                        "order": condition_index,
                        "condition": condition,
                        "pair_left": left,
                        "pair_right": right,
                    },
                )
                records.append(record)
                print(
                    f"[thread-bench] {pair_key} {phase} {phase_index + 1}/"
                    f"{warmups if phase == 'warmup' else samples} "
                    f"{revision}/{condition}: "
                    f"guest={record['guest_elapsed_ns'] / 1e6:.3f} ms "
                    f"host={record['host_wall_elapsed_ns'] / 1e6:.3f} ms",
                    file=sys.stderr,
                )


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record["phase"] != "measure":
            continue
        key = (
            record["revision"],
            record["pair_kind"],
            record["pair_key"],
            record["condition"],
        )
        grouped.setdefault(key, []).append(record)
    summaries = []
    for (revision, pair_kind, pair_key, condition), selected in sorted(
        grouped.items()
    ):
        elapsed = [record["elapsed_ns"] for record in selected]
        host_wall = [record["host_wall_elapsed_ns"] for record in selected]
        throughput = [record["throughput_ops_per_second"] for record in selected]
        per_thread = [record["per_thread_ops_per_second"] for record in selected]
        summaries.append(
            {
                "revision": revision,
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
    cells: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        if record["phase"] != "measure":
            continue
        key = (
            record["revision"],
            record["pair_kind"],
            record["pair_key"],
            record["pair_index"],
        )
        cells.setdefault(key, {})[record["condition"]] = record
    grouped: dict[
        tuple[str, str, str, str, str],
        dict[str, list[float]],
    ] = {}
    for (revision, pair_kind, pair_key, _), conditions in cells.items():
        if len(conditions) != 2:
            raise HarnessError(
                f"incomplete measured pair: {revision}/{pair_key}"
            )
        pair_records = list(conditions.values())
        left = pair_records[0]["pair_left"]
        right = pair_records[0]["pair_right"]
        if any(
            record["pair_left"] != left or record["pair_right"] != right
            for record in pair_records
        ):
            raise HarnessError(
                f"inconsistent pair direction: {revision}/{pair_key}"
            )
        if set(conditions) != {left, right}:
            raise HarnessError(
                f"pair conditions do not match direction: {revision}/{pair_key}"
            )
        left_record = conditions[left]
        right_record = conditions[right]
        selected = grouped.setdefault(
            (revision, pair_kind, pair_key, left, right),
            {"elapsed": [], "throughput": []},
        )
        selected["elapsed"].append(
            right_record["elapsed_ns"] / left_record["elapsed_ns"]
        )
        selected["throughput"].append(
            right_record["throughput_ops_per_second"]
            / left_record["throughput_ops_per_second"]
        )
    result = []
    for (
        revision,
        pair_kind,
        pair_key,
        left,
        right,
    ), ratios in sorted(grouped.items()):
        result.append(
            {
                "revision": revision,
                "pair_kind": pair_kind,
                "pair_key": pair_key,
                "left": left,
                "right": right,
                "elapsed_right_over_left": sample_stats(
                    ratios["elapsed"], "samples"
                ),
                "throughput_right_over_left": sample_stats(
                    ratios["throughput"], "samples"
                ),
                "median_elapsed_delta_pct": (
                    statistics.median(ratios["elapsed"]) - 1
                )
                * 100,
                "median_throughput_delta_pct": (
                    statistics.median(ratios["throughput"]) - 1
                )
                * 100,
            }
        )
    return result


def comparison_summaries(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    record_roles = {record["revision"] for record in records}
    if record_roles == set(SINGLE_REVISION_ROLES):
        return []
    if record_roles != set(REVISION_ROLES):
        raise HarnessError("comparison records have incomplete revision roles")
    cells: dict[
        tuple[str, str, str, int],
        dict[str, dict[str, Any]],
    ] = {}
    for record in records:
        if record["phase"] != "measure":
            continue
        key = (
            record["pair_kind"],
            record["pair_key"],
            record["condition"],
            record["pair_index"],
        )
        revision = record["revision"]
        if revision in cells.setdefault(key, {}):
            raise HarnessError(
                f"duplicate revision sample: {revision}/{record['pair_key']}/"
                f"{record['condition']}/{record['pair_index']}"
            )
        cells[key][revision] = record
    grouped: dict[
        tuple[str, str, str, str],
        dict[str, list[float]],
    ] = {}
    for (pair_kind, pair_key, condition, _), revisions in cells.items():
        if set(revisions) != set(REVISION_ROLES):
            raise HarnessError(
                f"incomplete revision pair: {pair_key}/{condition}"
            )
        baseline = revisions["baseline"]
        candidate = revisions["candidate"]
        if baseline["metric_kind"] != candidate["metric_kind"]:
            raise HarnessError(
                f"mixed metric kind: {pair_key}/{condition}"
            )
        selected = grouped.setdefault(
            (pair_kind, pair_key, condition, baseline["metric_kind"]),
            {"elapsed": [], "throughput": []},
        )
        selected["elapsed"].append(
            candidate["elapsed_ns"] / baseline["elapsed_ns"]
        )
        selected["throughput"].append(
            candidate["throughput_ops_per_second"]
            / baseline["throughput_ops_per_second"]
        )
    result = []
    for (
        pair_kind,
        pair_key,
        condition,
        metric_kind,
    ), ratios in sorted(grouped.items()):
        result.append(
            {
                "pair_kind": pair_kind,
                "pair_key": pair_key,
                "condition": condition,
                "metric_kind": metric_kind,
                "baseline": "baseline",
                "candidate": "candidate",
                "elapsed_candidate_over_baseline": sample_stats(
                    ratios["elapsed"], "samples"
                ),
                "throughput_candidate_over_baseline": sample_stats(
                    ratios["throughput"], "samples"
                ),
                "median_elapsed_delta_pct": (
                    statistics.median(ratios["elapsed"]) - 1
                )
                * 100,
                "median_throughput_delta_pct": (
                    statistics.median(ratios["throughput"]) - 1
                )
                * 100,
            }
        )
    return result


def ratio_of_ratios_summaries(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    record_roles = {record["revision"] for record in records}
    if record_roles == set(SINGLE_REVISION_ROLES):
        return []
    if record_roles != set(REVISION_ROLES):
        raise HarnessError("ratio-of-ratios records have incomplete revision roles")
    cells: dict[
        tuple[str, str, int],
        dict[str, dict[str, dict[str, Any]]],
    ] = {}
    for record in records:
        if record["phase"] != "measure":
            continue
        key = (record["pair_kind"], record["pair_key"], record["pair_index"])
        revision = record["revision"]
        condition = record["condition"]
        revisions = cells.setdefault(key, {})
        conditions = revisions.setdefault(revision, {})
        if condition in conditions:
            raise HarnessError(
                f"duplicate ratio-of-ratios sample: {revision}/"
                f"{record['pair_key']}/{condition}/{record['pair_index']}"
            )
        conditions[condition] = record
    grouped: dict[
        tuple[str, str, str, str],
        dict[str, list[float]],
    ] = {}
    for (pair_kind, pair_key, _), revisions in cells.items():
        if set(revisions) != set(REVISION_ROLES):
            raise HarnessError(f"incomplete revisions for pair {pair_key}")
        first = next(iter(revisions["baseline"].values()))
        left = first["pair_left"]
        right = first["pair_right"]
        if any(
            set(revisions[revision]) != {left, right}
            for revision in REVISION_ROLES
        ):
            raise HarnessError(f"incomplete internal pair for {pair_key}")
        baseline = revisions["baseline"]
        candidate = revisions["candidate"]
        baseline_elapsed = (
            baseline[right]["elapsed_ns"] / baseline[left]["elapsed_ns"]
        )
        candidate_elapsed = (
            candidate[right]["elapsed_ns"] / candidate[left]["elapsed_ns"]
        )
        baseline_throughput = (
            baseline[right]["throughput_ops_per_second"]
            / baseline[left]["throughput_ops_per_second"]
        )
        candidate_throughput = (
            candidate[right]["throughput_ops_per_second"]
            / candidate[left]["throughput_ops_per_second"]
        )
        selected = grouped.setdefault(
            (pair_kind, pair_key, left, right),
            {"elapsed": [], "throughput": []},
        )
        selected["elapsed"].append(candidate_elapsed / baseline_elapsed)
        selected["throughput"].append(
            candidate_throughput / baseline_throughput
        )
    result = []
    for (pair_kind, pair_key, left, right), ratios in sorted(grouped.items()):
        result.append(
            {
                "pair_kind": pair_kind,
                "pair_key": pair_key,
                "left": left,
                "right": right,
                "baseline": "baseline",
                "candidate": "candidate",
                "elapsed_ratio_of_ratios": sample_stats(
                    ratios["elapsed"], "samples"
                ),
                "throughput_ratio_of_ratios": sample_stats(
                    ratios["throughput"], "samples"
                ),
                "median_elapsed_delta_pct": (
                    statistics.median(ratios["elapsed"]) - 1
                )
                * 100,
                "median_throughput_delta_pct": (
                    statistics.median(ratios["throughput"]) - 1
                )
                * 100,
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
    for key in (
        "fixture_set_sha256",
        "plan_sha256",
        "measurement_plan_sha256",
    ):
        require(
            isinstance(metadata.get(key), str)
            and re.fullmatch(r"[0-9a-f]{64}", metadata[key]) is not None,
            f"metadata.{key}",
        )
    require(
        metadata.get("measurement_plan_version")
        == MEASUREMENT_PLAN_IDENTITY_VERSION,
        "metadata.measurement_plan_version",
    )
    plan = document["plan"]
    require(plan.get("warmups", -1) >= 0, "plan.warmups")
    require(plan.get("samples", 0) > 0, "plan.samples")
    require(
        plan.get("minimum_timed_interval_ns", 0) > 0,
        "plan.minimum_timed_interval_ns",
    )
    revision_mode = plan.get("revision_mode")
    comparison_purpose = plan.get("comparison_purpose")
    if revision_mode == "paired-revisions":
        revision_roles = REVISION_ROLES
        require(
            comparison_purpose in COMPARISON_PURPOSES,
            "plan.comparison_purpose",
        )
        require(
            plan["samples"] % 2 == 0,
            "paired revision samples must be even",
        )
    else:
        require(
            revision_mode == "single-revision-compatibility",
            "plan.revision_mode",
        )
        revision_roles = SINGLE_REVISION_ROLES
        require(
            comparison_purpose == "single-revision-compatibility",
            "plan.comparison_purpose",
        )
    require(
        plan.get("revision_roles") == list(revision_roles),
        "plan.revision_roles",
    )
    require(metadata["plan_sha256"] == cache_key(plan), "metadata.plan_sha256")
    require(
        metadata["measurement_plan_sha256"]
        == measurement_plan_sha256(plan),
        "metadata.measurement_plan_sha256",
    )

    revisions = metadata.get("revisions")
    require(
        isinstance(revisions, dict) and set(revisions) == set(revision_roles),
        "metadata.revisions",
    )
    revision_checkouts = metadata.get("revision_checkouts")
    require(
        isinstance(revision_checkouts, dict)
        and set(revision_checkouts) == set(revision_roles)
        and all(
            isinstance(path, str) and bool(path)
            for path in revision_checkouts.values()
        ),
        "metadata.revision_checkouts",
    )
    if revision_mode == "paired-revisions":
        require(
            len(set(revision_checkouts.values())) == len(REVISION_ROLES),
            "paired revisions use the same checkout path",
        )
    revision_keys = {
        "commit",
        "tracked_diff_sha256",
        "build_source_sha256",
        "fixture_set_sha256",
        "plan_sha256",
        "host_pair_id",
        "host_fingerprint_sha256",
    }
    for role in revision_roles:
        revision = revisions[role]
        require(isinstance(revision, dict), f"metadata.revisions.{role}")
        require(
            set(revision) == revision_keys,
            f"metadata.revisions.{role} fields",
        )
        require(
            re.fullmatch(r"[0-9a-f]{40}", revision.get("commit", ""))
            is not None,
            f"metadata.revisions.{role}.commit",
        )
        for key in (
            "tracked_diff_sha256",
            "build_source_sha256",
            "fixture_set_sha256",
            "plan_sha256",
            "host_fingerprint_sha256",
        ):
            require(
                re.fullmatch(r"[0-9a-f]{64}", revision.get(key, ""))
                is not None,
                f"metadata.revisions.{role}.{key}",
            )
        require(
            revision["fixture_set_sha256"] == metadata["fixture_set_sha256"],
            f"mixed fixture identity for {role}",
        )
        require(
            revision["plan_sha256"] == metadata["plan_sha256"],
            f"mixed plan identity for {role}",
        )
    candidate = revisions["candidate"]
    for key in ("commit", "tracked_diff_sha256", "build_source_sha256"):
        require(
            metadata.get(key) == candidate[key],
            f"metadata.{key} candidate compatibility alias",
        )
    host = metadata.get("host")
    require(isinstance(host, dict), "metadata.host")
    fingerprint = host.get("host_fingerprint")
    require(isinstance(fingerprint, dict), "metadata.host.host_fingerprint")
    fingerprint_fields = fingerprint.get("fields")
    require(
        isinstance(fingerprint_fields, dict),
        "metadata.host.host_fingerprint.fields",
    )
    require(
        set(fingerprint_fields) == set(HOST_FINGERPRINT_FIELDS),
        "host fingerprint fields",
    )
    require(
        fingerprint.get("sha256")
        == cache_key(fingerprint_fields),
        "metadata.host.host_fingerprint",
    )
    host_pair = metadata.get("host_pair")
    require(
        isinstance(host_pair, dict)
        and set(host_pair)
        == {"id", "runner_environment", "host_fingerprint_sha256"},
        "metadata.host_pair",
    )
    require(
        isinstance(host_pair["id"], str) and bool(host_pair["id"]),
        "metadata.host_pair.id",
    )
    require(
        host_pair["runner_environment"] == host.get("runner_environment"),
        "metadata.host_pair runner environment",
    )
    require(
        isinstance(host_pair["runner_environment"], str)
        and bool(host_pair["runner_environment"]),
        "metadata.host_pair runner environment",
    )
    require(
        host_pair["host_fingerprint_sha256"] == fingerprint.get("sha256"),
        "metadata.host_pair fingerprint",
    )
    for role in revision_roles:
        revision = revisions[role]
        require(
            revision["host_pair_id"] == host_pair["id"],
            f"mixed host pair for {role}",
        )
        require(
            revision["host_fingerprint_sha256"]
            == host_pair["host_fingerprint_sha256"],
            f"mixed host fingerprint for {role}",
        )
    if comparison_purpose == "noise-calibration":
        require(
            all(
                revisions["baseline"][key] == revisions["candidate"][key]
                for key in (
                    "commit",
                    "tracked_diff_sha256",
                    "build_source_sha256",
                )
            ),
            "noise calibration revision identity",
        )
    budget = document.get("budget")
    require(isinstance(budget, dict), "budget")
    budget_status = budget.get("status")
    require(
        budget_status in ("disabled", "not-selected", "passed", "failed"),
        "budget.status",
    )
    if comparison_purpose == "noise-calibration":
        require(
            budget_status == "disabled"
            and budget.get("path") is None
            and budget.get("failures") == [],
            "noise calibration must be non-enforcing",
        )
    if budget_status in ("passed", "failed"):
        require(
            comparison_purpose == "candidate-evaluation",
            "budget enforcement requires candidate evaluation",
        )
        require(
            revisions["baseline"]["commit"]
            != revisions["candidate"]["commit"],
            "budget enforcement requires distinct revision commits",
        )
        require(
            revisions["baseline"]["build_source_sha256"]
            != revisions["candidate"]["build_source_sha256"],
            "budget enforcement requires distinct build identities",
        )
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

    seen: set[tuple[str, str, str, int, str]] = set()
    per_cell: dict[
        tuple[str, str, int],
        set[tuple[str, str]],
    ] = {}
    cell_order: dict[
        tuple[str, str, int],
        list[tuple[str, str]],
    ] = {}
    for record in document["records"]:
        require(isinstance(record, dict), "record object")
        require(record.get("phase") in ("warmup", "measure"), "record phase")
        revision_role = record.get("revision")
        require(revision_role in revision_roles, "record revision")
        revision = revisions[revision_role]
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
        for key in (
            "commit",
            "build_source_sha256",
            "fixture_set_sha256",
            "plan_sha256",
            "host_pair_id",
            "host_fingerprint_sha256",
        ):
            expected_key = (
                "revision_commit"
                if key == "commit"
                else "revision_build_source_sha256"
                if key == "build_source_sha256"
                else key
            )
            require(
                record.get(expected_key) == revision[key],
                f"record mixed {key}",
            )
        global_index = (
            record["pair_index"]
            if record["phase"] == "warmup"
            else plan["warmups"] + record["pair_index"]
        )
        expected_revisions = (
            alternating_pair_order(global_index, *REVISION_ROLES)
            if revision_roles == REVISION_ROLES
            else SINGLE_REVISION_ROLES
        )
        expected_conditions = alternating_pair_order(
            global_index, pair["left"], pair["right"]
        )
        require(
            record.get("revision_order")
            == expected_revisions.index(revision_role),
            "record revision order",
        )
        require(
            record.get("order")
            == expected_conditions.index(record["condition"]),
            "record condition order",
        )
        key = (
            pair_key,
            revision_role,
            record["condition"],
            record["pair_index"],
            record["phase"],
        )
        require(key not in seen, f"duplicate record {key}")
        seen.add(key)
        cell = (pair_key, record["phase"], record["pair_index"])
        per_cell.setdefault(cell, set()).add(
            (revision_role, record["condition"])
        )
        cell_order.setdefault(cell, []).append(
            (revision_role, record["condition"])
        )
    expected_records_per_pair = (
        len(revision_roles)
        * 2
        * (plan["warmups"] + plan["samples"])
    )
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
                    == {
                        (revision, condition)
                        for revision in revision_roles
                        for condition in (pair["left"], pair["right"])
                    },
                    f"incomplete pair cell {pair_key}/{phase}/{index}",
                )
                global_index = (
                    index if phase == "warmup" else plan["warmups"] + index
                )
                require(
                    cell_order.get((pair_key, phase, index))
                    == [
                        (revision, condition)
                        for revision in (
                            alternating_pair_order(
                                global_index, *REVISION_ROLES
                            )
                            if revision_roles == REVISION_ROLES
                            else SINGLE_REVISION_ROLES
                        )
                        for condition in alternating_pair_order(
                            global_index, pair["left"], pair["right"]
                        )
                    ],
                    f"inverted pair order {pair_key}/{phase}/{index}",
                )

    expected_summary_keys = {
        (revision, pair_key, condition)
        for revision in revision_roles
        for pair_key, pair in pair_by_key.items()
        for condition in (pair["left"], pair["right"])
    }
    actual_summary_keys = {
        (
            summary.get("revision"),
            summary.get("pair_key"),
            summary.get("condition"),
        )
        for summary in document["summaries"]
    }
    require(
        actual_summary_keys == expected_summary_keys,
        "summaries do not cover every planned condition",
    )
    expected_paired_keys = {
        (revision, pair_key)
        for revision in revision_roles
        for pair_key in pair_by_key
    }
    actual_paired_keys = {
        (summary.get("revision"), summary.get("pair_key"))
        for summary in document["paired_summaries"]
    }
    require(
        actual_paired_keys == expected_paired_keys,
        "paired summaries do not cover every planned pair",
    )
    for summary in document["paired_summaries"]:
        pair = pair_by_key[summary["pair_key"]]
        require(summary.get("left") == pair["left"], "paired summary left")
        require(summary.get("right") == pair["right"], "paired summary right")
    expected_comparison_keys = (
        {
            (pair_key, condition)
            for pair_key, pair in pair_by_key.items()
            for condition in (pair["left"], pair["right"])
        }
        if revision_roles == REVISION_ROLES
        else set()
    )
    comparisons = document.get("comparison_summaries")
    require(isinstance(comparisons, list), "comparison_summaries")
    require(
        {
            (summary.get("pair_key"), summary.get("condition"))
            for summary in comparisons
        }
        == expected_comparison_keys,
        "comparison summaries do not cover every planned condition",
    )
    ratios = document.get("ratio_of_ratios_summaries")
    require(isinstance(ratios, list), "ratio_of_ratios_summaries")
    require(
        {summary.get("pair_key") for summary in ratios}
        == (set(pair_by_key) if revision_roles == REVISION_ROLES else set()),
        "ratio-of-ratios summaries do not cover every planned pair",
    )
    require(
        document["summaries"] == summarize(document["records"]),
        "revision summaries do not match records",
    )
    require(
        document["paired_summaries"]
        == paired_summaries(document["records"]),
        "paired summaries do not match records",
    )
    require(
        comparisons == comparison_summaries(document["records"]),
        "comparison summaries do not match records",
    )
    require(
        ratios == ratio_of_ratios_summaries(document["records"]),
        "ratio-of-ratios summaries do not match records",
    )


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


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def require_budget_eligible_report(report: dict[str, Any]) -> None:
    try:
        validate_report(report)
    except BenchmarkDataError as exc:
        raise HarnessError(f"invalid report for budget enforcement: {exc}") from exc
    plan = report["plan"]
    if (
        plan["revision_mode"] != "paired-revisions"
        or plan["comparison_purpose"] != "candidate-evaluation"
    ):
        raise HarnessError(
            "budget enforcement requires a paired candidate-evaluation report"
        )
    report_revisions = report["metadata"]["revisions"]
    if (
        report_revisions["baseline"]["commit"]
        == report_revisions["candidate"]["commit"]
    ):
        raise HarnessError(
            "budget enforcement requires distinct baseline/candidate commits"
        )
    if (
        report_revisions["baseline"]["build_source_sha256"]
        == report_revisions["candidate"]["build_source_sha256"]
    ):
        raise HarnessError(
            "budget enforcement requires distinct baseline/candidate build identities"
        )


def load_budget(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    require_budget_eligible_report(report)
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
            "enforcement",
            "calibration_requirements",
            "calibration_provenance",
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
    if budget["enforcement"] is not True:
        raise HarnessError("budget enforcement is disabled; run with --no-budget")
    requirements = budget["calibration_requirements"]
    calibration = budget["calibration_provenance"]
    platforms = budget["platforms"]
    if (
        not isinstance(requirements, dict)
        or not isinstance(calibration, dict)
        or not isinstance(platforms, dict)
    ):
        raise HarnessError(
            "budget requirements/calibration/platforms must be objects"
        )
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
    if (
        not isinstance(minimum_reports, int)
        or isinstance(minimum_reports, bool)
        or minimum_reports < 20
    ):
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
        calibration,
        {
            "baseline_revision",
            "candidate_revision",
            "comparison_purpose",
            "fixture_set_sha256",
            "plan_sha256",
            "measurement_plan_version",
            "measurement_plan_sha256",
            "profile",
            "report_count_by_platform",
        },
        "calibration_provenance",
    )
    if calibration["comparison_purpose"] != "noise-calibration":
        raise HarnessError(
            "budget calibration provenance must be noise calibration"
        )
    for role in REVISION_ROLES:
        identity = calibration[f"{role}_revision"]
        if not isinstance(identity, dict):
            raise HarnessError(
                f"budget calibration {role} revision must be an object"
            )
        _require_exact_keys(
            identity,
            {"commit", "build_source_sha256"},
            f"calibration_provenance.{role}_revision",
        )
        for key, length in (("commit", 40), ("build_source_sha256", 64)):
            value = identity[key]
            if (
                not isinstance(value, str)
                or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
            ):
                raise HarnessError(
                    f"budget calibration {role}.{key} has invalid identity"
                )
    if (
        calibration["baseline_revision"]
        != calibration["candidate_revision"]
    ):
        raise HarnessError(
            "budget noise-calibration revisions must have identical identities"
        )
    for key in (
        "fixture_set_sha256",
        "plan_sha256",
        "measurement_plan_sha256",
    ):
        value = calibration[key]
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise HarnessError(
                f"budget calibration_provenance.{key} has invalid identity"
            )
    if (
        calibration["measurement_plan_version"]
        != MEASUREMENT_PLAN_IDENTITY_VERSION
    ):
        raise HarnessError("budget measurement plan identity version mismatch")
    if calibration["profile"] != requirements["required_profile"]:
        raise HarnessError("budget calibration profile mismatch")
    counts = calibration["report_count_by_platform"]
    if not isinstance(counts, dict) or set(counts) != set(required_platforms):
        raise HarnessError(
            "budget calibration report counts do not cover platforms"
        )
    if any(
        not isinstance(counts[item], int)
        or isinstance(counts[item], bool)
        or counts[item] < minimum_reports
        for item in required_platforms
    ):
        raise HarnessError(
            "budget calibration has insufficient retained reports"
        )
    if set(platforms) != set(required_platforms):
        raise HarnessError("budget platforms are incomplete or unknown")

    metadata = report["metadata"]
    baseline = metadata["revisions"]["baseline"]
    identity_checks = {
        "baseline_revision.commit": baseline["commit"],
        "baseline_revision.build_source_sha256": baseline[
            "build_source_sha256"
        ],
        "fixture_set_sha256": metadata["fixture_set_sha256"],
        "measurement_plan_version": metadata["measurement_plan_version"],
        "measurement_plan_sha256": metadata["measurement_plan_sha256"],
        "profile": report["plan"]["profile"],
    }
    for key, actual in identity_checks.items():
        if "." in key:
            identity_key = key.split(".", 1)[1]
            expected = calibration["baseline_revision"][identity_key]
        else:
            expected = calibration[key]
        if expected != actual:
            raise HarnessError(
                f"budget/report calibration mismatch for {key}: "
                f"{expected!r} != {actual!r}"
            )
    platform_id = metadata["platform_id"]
    if platform_id not in platforms:
        raise HarnessError(f"budget has no platform {platform_id!r}")
    platform_budget = platforms[platform_id]
    if not isinstance(platform_budget, dict):
        raise HarnessError("platform budget must be an object")
    _require_exact_keys(
        platform_budget,
        {
            "host_system",
            "host_machine",
            "runner_environment",
            "comparisons",
            "ratio_of_ratios",
        },
        f"platforms.{platform_id}",
    )
    host = metadata["host"]
    expected_host = CANONICAL_PLATFORMS[platform_id]
    if (host["system"], host["machine"]) != expected_host:
        raise HarnessError("report platform ID does not match its canonical host identity")
    if platform_budget["host_system"] != host["system"] or platform_budget["host_machine"] != host["machine"]:
        raise HarnessError("budget/report host platform mismatch")
    if platform_budget["runner_environment"] != host["runner_environment"]:
        raise HarnessError("budget/report runner environment mismatch")

    expected_comparisons = {
        (item["pair_key"], item["condition"]): item
        for item in report["comparison_summaries"]
    }
    comparison_thresholds = platform_budget["comparisons"]
    if not isinstance(comparison_thresholds, list):
        raise HarnessError("platform comparisons must be an array")
    seen_comparisons: set[tuple[str, str]] = set()
    for item in comparison_thresholds:
        if not isinstance(item, dict):
            raise HarnessError("comparison threshold must be an object")
        _require_exact_keys(
            item,
            {
                "pair_key",
                "condition",
                "metric_kind",
                "min_candidate_over_baseline_throughput_ratio",
                "max_candidate_over_baseline_elapsed_ratio",
            },
            "comparison threshold",
        )
        key = (item["pair_key"], item["condition"])
        if key in seen_comparisons:
            raise HarnessError(f"duplicate comparison threshold {key}")
        seen_comparisons.add(key)
        if key not in expected_comparisons:
            raise HarnessError(f"unknown comparison threshold {key}")
        if item["metric_kind"] != expected_comparisons[key]["metric_kind"]:
            raise HarnessError(f"comparison metric mismatch for {key}")
        for threshold_key in (
            "min_candidate_over_baseline_throughput_ratio",
            "max_candidate_over_baseline_elapsed_ratio",
        ):
            value = item[threshold_key]
            if not _positive_number(value):
                raise HarnessError(
                    f"comparison ratio must be positive for {key}"
                )
    if seen_comparisons != set(expected_comparisons):
        raise HarnessError(
            "comparison thresholds do not cover every report comparison"
        )

    pair_plan = {item["pair_key"]: item for item in report["plan"]["pairs"]}
    ratio_thresholds = platform_budget["ratio_of_ratios"]
    if not isinstance(ratio_thresholds, list):
        raise HarnessError("platform ratio_of_ratios must be an array")
    seen_pairs: set[str] = set()
    for item in ratio_thresholds:
        if not isinstance(item, dict):
            raise HarnessError("ratio-of-ratios threshold must be an object")
        _require_exact_keys(
            item,
            {
                "pair_key",
                "left",
                "right",
                "min_candidate_over_baseline_throughput_ratio_of_ratios",
                "max_candidate_over_baseline_elapsed_ratio_of_ratios",
            },
            "ratio-of-ratios threshold",
        )
        pair_key = item["pair_key"]
        if pair_key in seen_pairs:
            raise HarnessError(f"duplicate ratio-of-ratios threshold {pair_key}")
        seen_pairs.add(pair_key)
        if pair_key not in pair_plan:
            raise HarnessError(f"unknown ratio-of-ratios threshold {pair_key}")
        planned = pair_plan[pair_key]
        if item["left"] != planned["left"] or item["right"] != planned["right"]:
            raise HarnessError(
                f"ratio-of-ratios direction mismatch for {pair_key}"
            )
        for threshold_key in (
            "min_candidate_over_baseline_throughput_ratio_of_ratios",
            "max_candidate_over_baseline_elapsed_ratio_of_ratios",
        ):
            value = item[threshold_key]
            if not _positive_number(value):
                raise HarnessError(
                    f"ratio-of-ratios limit must be positive for {pair_key}"
                )
    if seen_pairs != set(pair_plan):
        raise HarnessError(
            "ratio-of-ratios thresholds do not cover the complete report plan"
        )
    return platform_budget


def evaluate_budget(
    platform_budget: dict[str, Any],
    report: dict[str, Any],
) -> list[str]:
    require_budget_eligible_report(report)
    comparisons = report["comparison_summaries"]
    ratio_of_ratios = report["ratio_of_ratios_summaries"]
    failures: list[str] = []
    comparison_lookup = {
        (item["pair_key"], item["condition"]): item
        for item in comparisons
    }
    for threshold in platform_budget["comparisons"]:
        key = (threshold["pair_key"], threshold["condition"])
        actual = comparison_lookup[key]
        minimum = float(
            threshold[
                "min_candidate_over_baseline_throughput_ratio"
            ]
        )
        maximum = float(
            threshold["max_candidate_over_baseline_elapsed_ratio"]
        )
        throughput = actual["throughput_candidate_over_baseline"]["median"]
        elapsed = actual["elapsed_candidate_over_baseline"]["median"]
        if throughput < minimum:
            failures.append(
                f"{key[0]}::{key[1]} candidate/baseline throughput: "
                f"{throughput:.4f} < {minimum:.4f}"
            )
        if elapsed > maximum:
            failures.append(
                f"{key[0]}::{key[1]} candidate/baseline elapsed: "
                f"{elapsed:.4f} > {maximum:.4f}"
            )
    ratio_lookup = {item["pair_key"]: item for item in ratio_of_ratios}
    for threshold in platform_budget["ratio_of_ratios"]:
        pair_key = threshold["pair_key"]
        actual = ratio_lookup[pair_key]
        minimum = float(
            threshold[
                "min_candidate_over_baseline_throughput_ratio_of_ratios"
            ]
        )
        maximum = float(
            threshold[
                "max_candidate_over_baseline_elapsed_ratio_of_ratios"
            ]
        )
        throughput = actual["throughput_ratio_of_ratios"]["median"]
        elapsed = actual["elapsed_ratio_of_ratios"]["median"]
        if throughput < minimum:
            failures.append(
                f"{pair_key} candidate/baseline throughput ratio-of-ratios: "
                f"{throughput:.4f} < {minimum:.4f}"
            )
        if elapsed > maximum:
            failures.append(
                f"{pair_key} candidate/baseline elapsed ratio-of-ratios: "
                f"{elapsed:.4f} > {maximum:.4f}"
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
    revisions = document["metadata"]["revisions"]
    lines = [
        "# WASI threaded benchmark",
        "",
        f"- Candidate: `{revisions['candidate']['commit']}`",
    ]
    if document["plan"]["revision_mode"] == "paired-revisions":
        lines.insert(2, f"- Baseline: `{revisions['baseline']['commit']}`")
    lines += [
        f"- Comparison purpose: `{document['plan']['comparison_purpose']}`",
        f"- Platform identity: `{document['metadata']['platform_id']}`",
        f"- Host pair: `{document['metadata']['host_pair']['id']}` · "
        f"fingerprint `{document['metadata']['host_pair']['host_fingerprint_sha256']}`",
        f"- Runner environment: `{host['runner_environment']}`",
        f"- Host: `{host['system']} {host['release']}` · `{host['machine']}` · "
        f"{host['logical_cpus']} CPUs · `{host['cpu']}`",
        f"- Profile: `{document['plan']['profile']}` "
        f"({document['plan']['warmups']} warmups, {document['plan']['samples']} samples)",
        f"- Budget: `{document['budget']['status']}`",
    ]
    poll_static = (
        document["metadata"]
        .get("aot_artifacts", {})
        .get("candidate", {})
        .get("cancel_poll_static")
    )
    if poll_static:
        lines.append(
            f"- AOT cancel polls: {poll_static['sites_enabled']} static sites, "
            f"{poll_static['bytes_per_site']} bytes/site; timed hot kernel "
            "executes 1 poll opportunity per operation when enabled"
        )
    lines += [
        "",
        "| Revision | Pair | Condition | Metric | Guest median ms | Host-wall median ms | Guest range ms | Median aggregate ops/s | Median per-thread ops/s |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in document["summaries"]:
        elapsed = item["elapsed"]
        host_wall = item["host_wall"]
        throughput = item["throughput"]
        per_thread = item["per_thread_throughput"]
        lines.append(
            f"| `{item['revision']}` | `{item['pair_key']}` | "
            f"`{item['condition']}` | "
            f"`{item['metric_kind']}` | "
            f"{elapsed['median'] / 1e6:.3f} | "
            f"{host_wall['median'] / 1e6:.3f} | "
            f"{elapsed['min'] / 1e6:.3f}–{elapsed['max'] / 1e6:.3f} | "
            f"{throughput['median']:.3f} | {per_thread['median']:.3f} |"
        )
    lines += [
        "",
        "| Revision | Internal pair | Right / left guest-time median delta | Right / left throughput median delta |",
        "|---|---|---:|---:|",
    ]
    for item in document["paired_summaries"]:
        lines.append(
            f"| `{item['revision']}` | `{item['pair_key']}`: "
            f"`{item['right']}` / `{item['left']}` | "
            f"{item['median_elapsed_delta_pct']:+.2f}% | "
            f"{item['median_throughput_delta_pct']:+.2f}% |"
        )
    if document["comparison_summaries"]:
        lines += [
            "",
            "| Matched revision comparison | Candidate / baseline elapsed | Candidate / baseline throughput |",
            "|---|---:|---:|",
        ]
        for item in document["comparison_summaries"]:
            lines.append(
                f"| `{item['pair_key']}` / `{item['condition']}` | "
                f"{item['elapsed_candidate_over_baseline']['median']:.4f} | "
                f"{item['throughput_candidate_over_baseline']['median']:.4f} |"
            )
        lines += [
            "",
            "| Internal-pair ratio-of-ratios | Candidate / baseline elapsed ratio | Candidate / baseline throughput ratio |",
            "|---|---:|---:|",
        ]
        for item in document["ratio_of_ratios_summaries"]:
            lines.append(
                f"| `{item['pair_key']}`: `{item['right']}` / "
                f"`{item['left']}` | "
                f"{item['elapsed_ratio_of_ratios']['median']:.4f} | "
                f"{item['throughput_ratio_of_ratios']['median']:.4f} |"
            )
    lines += [
        "",
        "Kernel throughput uses guest monotonic time corrected by a same-process "
        "barrier/timer calibration below 1%; host wall time is diagnostic only. "
        "Spawn/join is reported separately as a guest-timed lifecycle metric. "
        "Absolute values remain diagnostic; budget enforcement uses only "
        "matched candidate/baseline ratios.",
        "",
        "Every warmup and measured invocation passed its deterministic timing, "
        "checksum, operation count, workload, thread-count, and iteration assertions.",
        "",
    ]
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.baseline_repo is None:
        candidate_repo = args.repo.resolve()
        revision_mode = "single-revision-compatibility"
        comparison_purpose = "single-revision-compatibility"
        revision_roles = SINGLE_REVISION_ROLES
        revision_repos = {"candidate": candidate_repo}
    else:
        baseline_repo = args.baseline_repo.resolve()
        candidate_repo = args.candidate_repo.resolve()
        if args.comparison_purpose not in COMPARISON_PURPOSES:
            raise HarnessError("paired revisions require an explicit comparison purpose")
        if args.samples % 2 != 0:
            raise HarnessError("paired revision measurements require even samples")
        if args.comparison_purpose == "noise-calibration" and not args.no_budget:
            raise HarnessError("noise calibration must be non-enforcing")
        if baseline_repo == candidate_repo:
            raise HarnessError(
                "paired revisions must use distinct independently built checkout paths"
            )
        revision_mode = "paired-revisions"
        comparison_purpose = args.comparison_purpose
        revision_roles = REVISION_ROLES
        revision_repos = {
            "baseline": baseline_repo,
            "candidate": candidate_repo,
        }
    repo = candidate_repo
    output = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (repo / args.output_dir).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        role: source_identity(revision_repo)
        for role, revision_repo in revision_repos.items()
    }
    if comparison_purpose == "noise-calibration":
        if any(
            sources["baseline"][key] != sources["candidate"][key]
            for key in (
                "commit",
                "tracked_diff_sha256",
                "build_source_sha256",
            )
        ):
            raise HarnessError(
                "noise calibration requires identical revision identities "
                "from distinct checkout paths"
            )
    fixture_reports = {
        role: resolve_fixtures(revision_repo)
        for role, revision_repo in revision_repos.items()
    }
    fixture_set_identities = {
        role: fixture_set_identity(fixtures)
        for role, fixtures in fixture_reports.items()
    }
    if len(set(fixture_set_identities.values())) != 1:
        raise HarnessError(
            "baseline/candidate reports have mixed fixture identity"
        )
    fixture_set_sha256 = fixture_set_identities["candidate"]
    minimum_interval_ns = int(args.min_interval_ms * 1_000_000)
    modes = ("interpreter", "aot") if args.modes == "both" else (args.modes,)
    pair_plan = planned_pair_specs(args, modes)
    runner = shlex.split(args.runner)
    plan = {
        "profile": args.profile,
        "warmups": args.warmups,
        "samples": args.samples,
        "revision_mode": revision_mode,
        "comparison_purpose": comparison_purpose,
        "revision_roles": list(revision_roles),
        "modes": list(modes),
        "thread_counts": list(args.thread_counts),
        "iterations": {
            "single-hot": args.single_iterations,
            "cancel-hot": args.cancel_iterations,
            "hot": args.hot_iterations,
            "atomic": args.atomic_iterations,
            "atomic-total": args.atomic_total_iterations,
            "wait-notify": args.wait_iterations,
            "spawn-join": args.spawn_iterations,
        },
        "timeout_seconds": args.timeout,
        "minimum_timed_interval_ns": minimum_interval_ns,
        "atomic_wait_preflight_runs": ATOMIC_WAIT_PREFLIGHT_RUNS[args.profile],
        "optimize": args.optimize,
        "pairs": pair_plan,
    }
    plan_sha256 = cache_key(plan)
    measurement_plan_identity = measurement_plan_sha256(plan)
    host = host_metadata(args.runner_environment)
    host_pair = host_pair_identity(args.platform_id, host, args.host_pair_id)
    revisions = {
        role: {
            **sources[role],
            "fixture_set_sha256": fixture_set_identities[role],
            "plan_sha256": plan_sha256,
            "host_pair_id": host_pair["id"],
            "host_fingerprint_sha256": host_pair[
                "host_fingerprint_sha256"
            ],
        }
        for role in revision_roles
    }
    revision_fields = {
        role: {
            "revision_commit": revisions[role]["commit"],
            "revision_build_source_sha256": revisions[role][
                "build_source_sha256"
            ],
            "fixture_set_sha256": revisions[role]["fixture_set_sha256"],
            "plan_sha256": plan_sha256,
            "host_pair_id": host_pair["id"],
            "host_fingerprint_sha256": host_pair[
                "host_fingerprint_sha256"
            ],
        }
        for role in revision_roles
    }

    contexts: dict[str, dict[str, Any]] = {}
    for role in revision_roles:
        revision_repo = revision_repos[role]
        revision_output = output / "revisions" / role
        builds: dict[str, Build] = {}
        for mode in modes:
            for enabled in (False, True):
                build = build_variant(
                    repo=revision_repo,
                    root=revision_output,
                    mode=mode,
                    threads_enabled=enabled,
                    optimize=args.optimize,
                    target=args.target,
                    source=sources[role],
                    rebuild=args.rebuild,
                    compiler_toggle=enabled and mode == "aot",
                )
                builds[build.name] = build
        aot_artifacts: dict[str, Path] = {}
        aot_artifacts_metadata: dict[str, Any] = {}
        if "aot" in modes:
            if args.target:
                compiler = build_variant(
                    repo=revision_repo,
                    root=revision_output,
                    mode="aot",
                    threads_enabled=True,
                    optimize=args.optimize,
                    target=None,
                    source=sources[role],
                    rebuild=args.rebuild,
                    compiler_toggle=True,
                )
                builds["host-compiler"] = compiler
            else:
                compiler = builds["enabled-aot"]
            aot_artifacts = compile_aot_fixtures(
                revision_repo,
                revision_output,
                compiler,
                execution_arch(args),
            )
            aot_artifacts_metadata = aot_artifact_report(
                aot_artifacts, execution_arch(args)
            )
        contexts[role] = {
            "repo": revision_repo,
            "builds": builds,
            "aot_artifacts": aot_artifacts,
            "aot_artifacts_metadata": aot_artifacts_metadata,
            "single_wasm": revision_repo / FIXTURES["single"]["path"],
            "threaded_wasm": revision_repo / FIXTURES["threaded"]["path"],
        }

    if "aot" in modes:
        for role in revision_roles:
            context = contexts[role]
            for index in range(ATOMIC_WAIT_PREFLIGHT_RUNS[args.profile]):
                measure_once(
                    repo=context["repo"],
                    runner=runner,
                    build=context["builds"]["enabled-aot"],
                    module=context["aot_artifacts"]["threaded-polls-on"],
                    workload="atomic",
                    threads=1,
                    iterations=1_000_000,
                    timeout=args.timeout,
                    min_interval_ns=1,
                    record_fields={
                        "revision": role,
                        "pair_kind": "atomic-wait-preflight",
                        "pair_key": "atomic-wait-preflight",
                        "pair_index": index,
                        "phase": "preflight",
                        "order": 0,
                        "condition": "aot",
                        "pair_left": "aot",
                        "pair_right": "aot",
                        "mode": "aot",
                        "threads_enabled": True,
                        "cancel_points": "on",
                        "static_cancel_poll_sites": (
                            context["aot_artifacts_metadata"][
                                "cancel_poll_static"
                            ]["sites_enabled"]
                        ),
                        "workload": "atomic",
                        "threads": 1,
                        "iterations": 1_000_000,
                    },
                )

    records: list[dict[str, Any]] = []
    for mode in modes:
        def single_measure(
            revision: str,
            condition: str,
            fields: dict[str, Any],
        ) -> dict[str, Any]:
            context = contexts[revision]
            builds = context["builds"]
            disabled = builds[f"disabled-{mode}"]
            enabled = builds[f"enabled-{mode}"]
            selected = disabled if condition == "threads-disabled" else enabled
            module = (
                context["single_wasm"]
                if mode == "interpreter"
                else context["aot_artifacts"]["single"]
            )
            return measure_once(
                repo=context["repo"],
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

        collect_revision_pair(
            records=records,
            pair_kind="single-infrastructure",
            pair_key=f"single-infrastructure/{mode}",
            left="threads-disabled",
            right="threads-enabled",
            warmups=args.warmups,
            samples=args.samples,
            revision_roles=revision_roles,
            revision_fields=revision_fields,
            measure=single_measure,
        )

    scenarios = planned_scenarios(args)
    for scenario in scenarios:
        if len(modes) == 2:
            def runtime_measure(
                revision: str,
                condition: str,
                fields: dict[str, Any],
            ) -> dict[str, Any]:
                context = contexts[revision]
                mode = condition
                build = context["builds"][f"enabled-{mode}"]
                module = (
                    context["threaded_wasm"]
                    if mode == "interpreter"
                    else context["aot_artifacts"]["threaded-polls-on"]
                )
                return measure_once(
                    repo=context["repo"],
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
                            context["aot_artifacts_metadata"][
                                "cancel_poll_static"
                            ]["sites_enabled"]
                            if mode == "aot"
                            else None
                        ),
                        "workload": scenario.workload,
                        "threads": scenario.threads,
                        "iterations": scenario.iterations,
                    },
                )

            collect_revision_pair(
                records=records,
                pair_kind="runtime-parity",
                pair_key=f"runtime/{scenario.key}",
                left="interpreter",
                right="aot",
                warmups=args.warmups,
                samples=args.samples,
                revision_roles=revision_roles,
                revision_fields=revision_fields,
                measure=runtime_measure,
            )
        else:
            mode = modes[0]

            def single_mode_measure(
                revision: str,
                condition: str,
                fields: dict[str, Any],
            ) -> dict[str, Any]:
                context = contexts[revision]
                build = context["builds"][f"enabled-{mode}"]
                module = (
                    context["threaded_wasm"]
                    if mode == "interpreter"
                    else context["aot_artifacts"]["threaded-polls-on"]
                )
                return measure_once(
                    repo=context["repo"],
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
                            context["aot_artifacts_metadata"][
                                "cancel_poll_static"
                            ]["sites_enabled"]
                            if mode == "aot"
                            else None
                        ),
                        "workload": scenario.workload,
                        "threads": scenario.threads,
                        "iterations": scenario.iterations,
                    },
                )

            collect_revision_pair(
                records=records,
                pair_kind="repeatability",
                pair_key=f"runtime/{scenario.key}/{mode}",
                left=f"{mode}-a",
                right=f"{mode}-b",
                warmups=args.warmups,
                samples=args.samples,
                revision_roles=revision_roles,
                revision_fields=revision_fields,
                measure=single_mode_measure,
            )

    if "aot" in modes:
        for threads in args.thread_counts:
            scenario = Scenario("hot", threads, cancel_iterations(args, threads))

            def poll_measure(
                revision: str,
                condition: str,
                fields: dict[str, Any],
            ) -> dict[str, Any]:
                context = contexts[revision]
                aot_build = context["builds"]["enabled-aot"]
                polls = "off" if condition == "cancel-points-off" else "on"
                module = context["aot_artifacts"][f"threaded-polls-{polls}"]
                return measure_once(
                    repo=context["repo"],
                    runner=runner,
                    build=aot_build,
                    module=module,
                    workload="hot",
                    threads=threads,
                    iterations=scenario.iterations,
                    timeout=args.timeout,
                    min_interval_ns=minimum_interval_ns,
                    record_fields={
                        **fields,
                        "mode": "aot",
                        "threads_enabled": True,
                        "cancel_points": polls,
                        "static_cancel_poll_sites": (
                            context["aot_artifacts_metadata"][
                                "cancel_poll_static"
                            ][
                                "sites_enabled"
                            ]
                            if polls == "on"
                            else 0
                        ),
                        "workload": "hot",
                        "threads": threads,
                        "iterations": scenario.iterations,
                    },
                )

            collect_revision_pair(
                records=records,
                pair_kind="cancel-point-cost",
                pair_key=f"cancel-points/hot/{threads}",
                left="cancel-points-off",
                right="cancel-points-on",
                warmups=args.warmups,
                samples=args.samples,
                revision_roles=revision_roles,
                revision_fields=revision_fields,
                measure=poll_measure,
            )

    summaries = summarize(records)
    pairs = paired_summaries(records)
    comparisons = comparison_summaries(records)
    ratios = ratio_of_ratios_summaries(records)
    candidate = revisions["candidate"]
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "metadata": {
            "commit": candidate["commit"],
            "tracked_diff_sha256": candidate["tracked_diff_sha256"],
            "build_source_sha256": candidate["build_source_sha256"],
            "revisions": revisions,
            "revision_checkouts": {
                role: str(revision_repos[role]) for role in revision_roles
            },
            "collected_at": collected_at(),
            "platform_id": args.platform_id,
            "fixture_set_sha256": fixture_set_sha256,
            "plan_sha256": plan_sha256,
            "measurement_plan_version": MEASUREMENT_PLAN_IDENTITY_VERSION,
            "measurement_plan_sha256": measurement_plan_identity,
            "host": host,
            "host_pair": host_pair,
            "execution": {
                "target": args.target or "native",
                "aot_target": execution_arch(args),
                "runner": runner,
            },
            "tools": {
                role: build_tool_report(contexts[role]["builds"], runner)
                for role in revision_roles
            },
            "fixture_toolchain": WASI_SDK,
            "fixtures": fixture_reports["candidate"],
            "aot_artifacts": {
                role: contexts[role]["aot_artifacts_metadata"]
                for role in revision_roles
            },
        },
        "plan": plan,
        "records": records,
        "summaries": summaries,
        "paired_summaries": pairs,
        "comparison_summaries": comparisons,
        "ratio_of_ratios_summaries": ratios,
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
        budget_failures = evaluate_budget(platform_budget, document)
        document["budget"]["status"] = (
            "passed" if not budget_failures else "failed"
        )
        document["budget"]["failures"] = budget_failures
        validate_report(document)
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
