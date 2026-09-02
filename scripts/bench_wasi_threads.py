#!/usr/bin/env python3
"""Run reproducible paired WASI pthread, atomic, and cancel-poll benchmarks."""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import shlex
import signal
import statistics
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


KIND = "wasi-thread-benchmark"
PROFILE_COUNTS = {
    "authoritative": (2, 10),
    "smoke": (1, 3),
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
        "sha256": "46eea2831e46a4d91f6f2cbdbc481116fd91c7402fbe1d1b355b9812a01ad44f",
    },
    "threaded": {
        "path": Path("tests/benchmarks/wasi-threads/threaded.wasm"),
        "sha256": "dc546f00c797a89c692416e255f9ec0f3a7ec9f01d5b3f2dc7d0d3dbeffc850e",
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
    parser.add_argument("--single-iterations", type=int, default=1_000_000)
    parser.add_argument("--hot-iterations", type=int, default=1_000_000)
    parser.add_argument("--atomic-iterations", type=int, default=50_000)
    parser.add_argument("--wait-iterations", type=int, default=500)
    parser.add_argument("--spawn-iterations", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
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
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
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
        if stored.get("key") == key:
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
    atomic_write_json(marker, {"schema_version": 1, "key": key, "parts": parts, "command": command})
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
    }


def parse_guest_result(
    stdout: str, expected: dict[str, int | str]
) -> dict[str, int | str]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise HarnessError(f"expected one guest JSON line, got {len(lines)}")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise HarnessError(f"guest output is not JSON: {lines[0]!r}") from exc
    if result != expected:
        raise HarnessError(f"guest result mismatch: expected {expected}, got {result}")
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
    record_fields: dict[str, Any],
) -> dict[str, Any]:
    guest_args = (
        [str(iterations)]
        if workload == "single-hot"
        else [workload, str(threads), str(iterations)]
    )
    command = [*runner, str(build.wamr), "run", str(module), *guest_args]
    started = time.perf_counter_ns()
    returncode, stdout, stderr = run_process(command, repo, timeout)
    elapsed_ns = time.perf_counter_ns() - started
    if returncode != 0:
        raise HarnessError(
            f"guest failed with exit {returncode}: {' '.join(command)}\n{stderr}"
        )
    expected = expected_result(workload, threads, iterations)
    guest = parse_guest_result(stdout, expected)
    operations = int(guest["operations"])
    throughput = operations / (elapsed_ns / 1e9)
    return {
        **record_fields,
        "command": command,
        "elapsed_ns": elapsed_ns,
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
                },
            )
            records.append(record)
            print(
                f"[thread-bench] {pair_key} {phase} {phase_index + 1}/"
                f"{warmups if phase == 'warmup' else samples} {condition}: "
                f"{record['elapsed_ns'] / 1e6:.3f} ms",
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
        throughput = [record["throughput_ops_per_second"] for record in selected]
        per_thread = [record["per_thread_ops_per_second"] for record in selected]
        summaries.append(
            {
                "pair_kind": pair_kind,
                "pair_key": pair_key,
                "condition": condition,
                "elapsed": sample_stats(elapsed, "samples_ns"),
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
        left, right = sorted(conditions)
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
    plan = document["plan"]
    require(plan.get("warmups", -1) >= 0, "plan.warmups")
    require(plan.get("samples", 0) > 0, "plan.samples")
    seen: set[tuple[str, str, int, str]] = set()
    for record in document["records"]:
        require(record.get("phase") in ("warmup", "measure"), "record phase")
        require(record.get("correct") is True, "record correctness")
        require(record.get("elapsed_ns", 0) > 0, "record elapsed")
        key = (
            record["pair_key"],
            record["condition"],
            record["pair_index"],
            record["phase"],
        )
        require(key not in seen, f"duplicate record {key}")
        seen.add(key)
    expected_records_per_pair = 2 * (plan["warmups"] + plan["samples"])
    pair_counts: dict[str, int] = {}
    for record in document["records"]:
        pair_counts[record["pair_key"]] = pair_counts.get(record["pair_key"], 0) + 1
    require(
        all(count == expected_records_per_pair for count in pair_counts.values()),
        "incomplete sample pairing",
    )


def load_budget(path: Path) -> dict[str, Any]:
    try:
        budget = json.loads(path.read_text(encoding="UTF-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"invalid budget {path}: {exc}") from exc
    if (
        budget.get("schema_version") != SCHEMA_VERSION
        or budget.get("kind") != "wasi-thread-benchmark-budget"
    ):
        raise HarnessError("budget schema/kind mismatch")
    if budget.get("calibrated") is not True:
        raise HarnessError(
            "budget is not calibrated; collect retained hosted reports and "
            "run with --no-budget"
        )
    return budget


def evaluate_budget(
    budget: dict[str, Any],
    summaries: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    pair_lookup = {item["pair_key"]: item for item in pairs}
    for pair_key, limit in budget.get("paired_elapsed_delta_pct", {}).items():
        actual = pair_lookup.get(pair_key)
        if actual is None:
            failures.append(f"missing paired result {pair_key}")
        elif actual["median_elapsed_delta_pct"] > float(limit):
            failures.append(
                f"{pair_key}: {actual['median_elapsed_delta_pct']:+.2f}% "
                f"> {float(limit):+.2f}%"
            )
    summary_lookup = {
        f"{item['pair_key']}::{item['condition']}": item for item in summaries
    }
    for key, minimum in budget.get("min_median_ops_per_second", {}).items():
        actual = summary_lookup.get(key)
        if actual is None:
            failures.append(f"missing throughput result {key}")
        elif actual["throughput"]["median"] < float(minimum):
            failures.append(
                f"{key}: {actual['throughput']['median']:.3f} "
                f"< {float(minimum):.3f}"
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
        f"- Host: `{host['system']} {host['release']}` · `{host['machine']}` · "
        f"{host['logical_cpus']} CPUs · `{host['cpu']}`",
        f"- Profile: `{document['plan']['profile']}` "
        f"({document['plan']['warmups']} warmups, {document['plan']['samples']} samples)",
        f"- Budget: `{document['budget']['status']}`",
        "",
        "| Pair | Condition | Median ms | Range ms | Median aggregate ops/s | Median per-thread ops/s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in document["summaries"]:
        elapsed = item["elapsed"]
        throughput = item["throughput"]
        per_thread = item["per_thread_throughput"]
        lines.append(
            f"| `{item['pair_key']}` | `{item['condition']}` | "
            f"{elapsed['median'] / 1e6:.3f} | "
            f"{elapsed['min'] / 1e6:.3f}–{elapsed['max'] / 1e6:.3f} | "
            f"{throughput['median']:.3f} | {per_thread['median']:.3f} |"
        )
    lines += [
        "",
        "| Paired comparison | Right / left elapsed median delta | Raw paired ratios |",
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
        "Every warmup and measured invocation passed its deterministic checksum, "
        "operation count, workload, thread-count, and iteration assertions.",
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
    modes = ("interpreter", "aot") if args.modes == "both" else (args.modes,)
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
                record_fields={
                    **fields,
                    "mode": mode,
                    "threads_enabled": selected.threads_enabled,
                    "cancel_points": "not-applicable",
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

    scenarios = [
        Scenario(workload, threads, iterations)
        for workload, iterations in (
            ("hot", args.hot_iterations),
            ("atomic", args.atomic_iterations),
            ("wait-notify", args.wait_iterations),
            ("spawn-join", args.spawn_iterations),
        )
        for threads in args.thread_counts
    ]
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
                    record_fields={
                        **fields,
                        "mode": mode,
                        "threads_enabled": True,
                        "cancel_points": (
                            "on" if mode == "aot" else "interpreter-dispatch"
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
                    record_fields={
                        **fields,
                        "mode": mode,
                        "threads_enabled": True,
                        "cancel_points": (
                            "on" if mode == "aot" else "interpreter-dispatch"
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
                    record_fields={
                        **fields,
                        "mode": "aot",
                        "threads_enabled": True,
                        "cancel_points": polls,
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
    budget_status = "disabled"
    budget_failures: list[str] = []
    if args.budget:
        budget = load_budget(args.budget.resolve())
        budget_failures = evaluate_budget(budget, summaries, pairs)
        budget_status = "passed" if not budget_failures else "failed"
    elif not args.no_budget:
        budget_status = "not-selected"

    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "metadata": {
            **source,
            "collected_at": collected_at(),
            "host": host_metadata(),
            "execution": {
                "target": args.target or "native",
                "aot_target": execution_arch(args),
                "runner": runner,
            },
            "tools": build_tool_report(builds, runner),
            "fixture_toolchain": WASI_SDK,
            "fixtures": fixture_report,
        },
        "plan": {
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
            "optimize": args.optimize,
        },
        "records": records,
        "summaries": summaries,
        "paired_summaries": pairs,
        "budget": {
            "status": budget_status,
            "path": str(args.budget.resolve()) if args.budget else None,
            "failures": budget_failures,
        },
    }
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
