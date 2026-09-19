#!/usr/bin/env python3
"""Measure AOT function-entry cancel-poll cost in a leaf-call-heavy thread."""

from __future__ import annotations

import argparse
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
from typing import Any

from benchmark_schema import (
    atomic_write_json,
    cache_key,
    collected_at,
    command_identity,
    host_metadata,
    sample_stats,
    sha256_bytes,
    sha256_file,
)

KIND = "leaf-cancel-cost"
SCHEMA_VERSION = 1
AOT_VERSION = 11
LEAF_UNROLL = 64
LEAF_SEED = 0x243F6A8885A308D3
LEAF_STEP = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
MAX_LEAF_CALLS = 10_000_000_000
FIXTURE = Path("tests/benchmarks/leaf-cancel-cost/leaf_calls.wasm")
SOURCE_PATHS = (
    ".github/workflows/leaf-cancel-cost.yml",
    "tests/benchmarks/leaf-cancel-cost/README.md",
    "tests/benchmarks/leaf-cancel-cost/leaf_calls.c",
    "tests/benchmarks/leaf-cancel-cost/build-fixture.sh",
    "tests/benchmarks/leaf-cancel-cost/fixtures.sha256",
    "tests/benchmarks/leaf-cancel-cost/report.schema.json",
    "scripts/bench_leaf_cancel_cost.py",
    "scripts/test_bench_leaf_cancel_cost.py",
    "scripts/benchmark_schema.py",
)
CANCEL_POLL_SIGNATURES = {
    "x86_64": bytes.fromhex("83bbb001000000740c"),
    "aarch64": struct.pack("<II", 0xB941B270, 0x34000090),
}
FIXTURE_TOOLCHAIN = {
    "wasi_sdk_version": "25.0",
    "clang": "clang version 19.1.5-wasi-sdk",
    "archive_sha256": (
        "52640dde13599bf127a95499e61d6d640256119456d1af8897ab6725bcf3d89c"
    ),
}
GUEST_KEYS = {
    "kind",
    "leaf_calls",
    "leaf_unroll",
    "batches",
    "checksum",
    "clock_id",
    "raw_elapsed_ns",
    "timing_overhead_ns",
    "elapsed_ns",
    "timing_overhead_ppm",
}


class HarnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Build:
    name: str
    prefix: Path
    wamr: Path
    wamrc: Path | None
    command: list[str]
    cache_key: str


def expected_result(calls: int) -> dict[str, int | str]:
    if calls <= 0 or calls % LEAF_UNROLL != 0:
        raise HarnessError(f"leaf calls must be a positive multiple of {LEAF_UNROLL}")
    return {
        "kind": "leaf-cancel-cost-result",
        "leaf_calls": calls,
        "leaf_unroll": LEAF_UNROLL,
        "batches": calls // LEAF_UNROLL,
        "checksum": (LEAF_SEED + LEAF_STEP * calls) & MASK64,
        "clock_id": "wasi-process-cputime",
    }


def parse_guest_result(
    stdout: str,
    calls: int,
    minimum_interval_ns: int,
    *,
    enforce_quality: bool,
) -> dict[str, int | str]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise HarnessError(f"expected one guest JSON line, got {len(lines)}")
    try:
        result = json.loads(lines[0], object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise HarnessError("guest output is not valid JSON") from exc
    if set(result) != GUEST_KEYS:
        raise HarnessError(
            f"guest fields mismatch: expected {sorted(GUEST_KEYS)}, got {sorted(result)}"
        )
    for key, value in expected_result(calls).items():
        if result.get(key) != value:
            raise HarnessError(
                f"guest result mismatch for {key}: expected {value!r}, got {result.get(key)!r}"
            )
    for key in (
        "raw_elapsed_ns",
        "timing_overhead_ns",
        "elapsed_ns",
        "timing_overhead_ppm",
    ):
        if not isinstance(result[key], int) or isinstance(result[key], bool):
            raise HarnessError(f"{key} must be an integer")
    raw = int(result["raw_elapsed_ns"])
    overhead = int(result["timing_overhead_ns"])
    elapsed = int(result["elapsed_ns"])
    if raw <= 0 or overhead < 0 or elapsed <= 0 or raw - overhead != elapsed:
        raise HarnessError("guest timing values are inconsistent")
    if result["timing_overhead_ppm"] != overhead * 1_000_000 // raw:
        raise HarnessError("guest timing_overhead_ppm is inconsistent")
    if enforce_quality and 99 * overhead >= elapsed:
        raise HarnessError("guest timing overhead is not below one percent")
    if enforce_quality and elapsed < minimum_interval_ns:
        raise HarnessError(
            f"guest interval {elapsed}ns is below required {minimum_interval_ns}ns"
        )
    return result


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"duplicate guest key {key!r}")
        result[key] = value
    return result


def round_calls(value: int) -> int:
    if value <= 0:
        raise HarnessError("leaf call count must be positive")
    return ((value + LEAF_UNROLL - 1) // LEAF_UNROLL) * LEAF_UNROLL


def select_calls(
    pilot_calls: int,
    pilot_elapsed_ns: int,
    target_interval_ns: int,
) -> int:
    if pilot_elapsed_ns <= 0 or target_interval_ns <= 0:
        raise HarnessError("pilot and target intervals must be positive")
    selected = math.ceil(pilot_calls * target_interval_ns / pilot_elapsed_ns)
    if selected > MAX_LEAF_CALLS:
        raise HarnessError(
            f"selected leaf call count exceeds safety limit {MAX_LEAF_CALLS}"
        )
    return round_calls(max(selected, LEAF_UNROLL))


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def source_identity(repo: Path) -> dict[str, Any]:
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    files = {}
    for relative in SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise HarnessError(f"missing harness source: {relative}")
        files[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    build_paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "build.zig", "build.zig.zon", "src"],
        cwd=repo,
    ).split(b"\0")
    build_source = bytearray()
    for raw in build_paths:
        if not raw:
            continue
        build_source.extend(raw)
        build_source.append(0)
        build_source.extend((repo / raw.decode()).read_bytes())
        build_source.append(0)
    return {
        "commit": git_output(repo, "rev-parse", "HEAD"),
        "tracked_diff_sha256": sha256_bytes(diff),
        "build_source_sha256": sha256_bytes(bytes(build_source)),
        "harness_files": files,
    }


def controlled_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    directories = {
        "ZIG_GLOBAL_CACHE_DIR": root / "zig-global-cache",
        "ZIG_LOCAL_CACHE_DIR": root / "zig-local-cache",
        "XDG_CACHE_HOME": root / "xdg-cache",
        "TMPDIR": root / "tmp",
    }
    for key, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update({"LANG": "C", "LC_ALL": "C", "TZ": "UTC"})
    return env


def build_tools(
    repo: Path,
    output: Path,
    source: dict[str, Any],
    optimize: str,
    target: str | None,
    rebuild: bool,
) -> tuple[Build, Build]:
    compiler = build_one(
        repo=repo,
        output=output,
        source=source,
        name="host-compiler",
        optimize=optimize,
        target=None,
        compiler=True,
        rebuild=rebuild,
    )
    if target is None:
        return compiler, compiler
    runtime = build_one(
        repo=repo,
        output=output,
        source=source,
        name="target-runtime",
        optimize=optimize,
        target=target,
        compiler=False,
        rebuild=rebuild,
    )
    return compiler, runtime


def build_one(
    *,
    repo: Path,
    output: Path,
    source: dict[str, Any],
    name: str,
    optimize: str,
    target: str | None,
    compiler: bool,
    rebuild: bool,
) -> Build:
    parts = {
        "build_source_sha256": source["build_source_sha256"],
        "optimize": optimize,
        "target": target or "native",
        "compiler": compiler,
        "benchmark_cancel_point_toggle": True,
        "zig": command_identity(["zig", "version"]),
    }
    key = cache_key(parts)
    prefix = output / "builds" / f"{name}-{key[:16]}"
    wamr = prefix / "bin/wamr"
    wamrc = prefix / "bin/wamrc" if compiler else None
    marker = prefix / "leaf-cancel-build.json"
    expected = [wamr] + ([wamrc] if wamrc else [])
    if not rebuild and marker.is_file() and all(path.is_file() for path in expected):
        try:
            stored = json.loads(marker.read_text(encoding="UTF-8"))
        except json.JSONDecodeError:
            stored = {}
        if stored.get("cache_key") == key and isinstance(
            stored.get("command"), list
        ):
            return Build(name, prefix, wamr, wamrc, stored["command"], key)
    command = [
        "zig",
        "build",
        f"-Doptimize={optimize}",
        "-Dlib_wasi_threads=true",
        "-Dinterp=false",
        "-Daot=true",
        "-Dbenchmark-cancel-point-toggle=true",
        "--prefix",
        str(prefix),
    ]
    if target:
        command.insert(2, f"-Dtarget={target}")
    prefix.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            command,
            cwd=repo,
            env=controlled_env(output / "cache" / name),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HarnessError(f"{name} build failed with exit {exc.returncode}") from exc
    if not all(path.is_file() for path in expected):
        raise HarnessError(f"{name} build did not produce expected binaries")
    atomic_write_json(marker, {"cache_key": key, "parts": parts, "command": command})
    return Build(name, prefix, wamr, wamrc, command, key)


def fixture_identity(repo: Path) -> dict[str, Any]:
    fixture = repo / FIXTURE
    if not fixture.is_file():
        raise HarnessError(f"missing fixture: {fixture}")
    manifest = repo / FIXTURE.parent / "fixtures.sha256"
    fields = manifest.read_text(encoding="UTF-8").strip().split()
    if len(fields) != 2 or fields[1] != FIXTURE.name:
        raise HarnessError("invalid fixture checksum manifest")
    actual = sha256_file(fixture)
    if actual != fields[0]:
        raise HarnessError(
            f"fixture checksum mismatch: expected {fields[0]}, got {actual}"
        )
    return {
        "path": str(FIXTURE),
        "sha256": actual,
        "bytes": fixture.stat().st_size,
        "toolchain": FIXTURE_TOOLCHAIN,
    }


def compile_artifacts(
    repo: Path,
    output: Path,
    compiler: Build,
    arch: str,
) -> dict[str, Path]:
    if compiler.wamrc is None:
        raise HarnessError("host compiler is missing wamrc")
    directory = output / "aot"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "cancel-points-off": directory / "leaf-calls-polls-off.cwasm",
        "cancel-points-on": directory / "leaf-calls-polls-on.cwasm",
    }
    for condition, path in artifacts.items():
        command = [str(compiler.wamrc), "compile", "--target", arch]
        if condition == "cancel-points-off":
            command.append("--benchmark-disable-cancel-points")
        command += [str(repo / FIXTURE), "-o", str(path)]
        try:
            subprocess.run(
                command,
                cwd=repo,
                env=controlled_env(output / "cache/compile-aot"),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise HarnessError(f"AOT compilation failed for {condition}") from exc
    return artifacts


def aot_text(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 8 or data[:4] != b"\x00aot":
        raise HarnessError(f"not a WAMR AOT artifact: {path}")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != AOT_VERSION:
        raise HarnessError(f"unsupported AOT version {version}")
    offset = 8
    while offset + 8 <= len(data):
        section_type, size = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + size > len(data):
            raise HarnessError(f"truncated AOT artifact: {path}")
        if section_type == 2:
            return data[offset : offset + size]
        offset += size
    raise HarnessError(f"AOT text section missing: {path}")


def artifact_identity(artifacts: dict[str, Path], arch: str) -> dict[str, Any]:
    if arch not in CANCEL_POLL_SIGNATURES:
        raise HarnessError(f"unsupported AOT architecture: {arch}")
    signature = CANCEL_POLL_SIGNATURES[arch]
    texts = {name: aot_text(path) for name, path in artifacts.items()}
    enabled = texts["cancel-points-on"].count(signature)
    disabled = texts["cancel-points-off"].count(signature)
    delta = len(texts["cancel-points-on"]) - len(texts["cancel-points-off"])
    if enabled <= 0 or disabled != 0:
        raise HarnessError(
            f"cancel-poll signatures invalid: enabled={enabled}, disabled={disabled}"
        )
    if delta <= 0 or delta % enabled != 0:
        raise HarnessError(
            f"AOT text delta {delta} is not attributable to {enabled} poll sites"
        )
    return {
        "architecture": arch,
        "cancel_poll_signature_hex": signature.hex(),
        "cancel_poll_sites_enabled": enabled,
        "cancel_poll_sites_disabled": disabled,
        "text_delta_bytes": delta,
        "bytes_per_poll_site": delta // enabled,
        "conditions": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "file_bytes": path.stat().st_size,
                "text_bytes": len(texts[name]),
            }
            for name, path in sorted(artifacts.items())
        },
    }


def run_process(
    command: list[str], cwd: Path, timeout: float
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise HarnessError(f"command timed out after {timeout:g} seconds") from exc
    try:
        return process.returncode, stdout.decode(), stderr.decode()
    except UnicodeDecodeError as exc:
        raise HarnessError("guest output is not UTF-8") from exc


def selected_cpu_affinity() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        return []
    allowed = sorted(os.sched_getaffinity(0))
    return allowed[:1]


def measure(
    *,
    repo: Path,
    runner: list[str],
    runtime: Build,
    artifact: Path,
    condition: str,
    calls: int,
    timeout: float,
    minimum_interval_ns: int,
    enforce_quality: bool,
    phase: str,
    pair_index: int,
    position: int,
) -> dict[str, Any]:
    affinity = selected_cpu_affinity()
    command = []
    if affinity:
        command += ["taskset", "--cpu-list", str(affinity[0])]
    command += [*runner, str(runtime.wamr), "run", str(artifact), str(calls)]
    started = time.perf_counter_ns()
    returncode, stdout, stderr = run_process(command, repo, timeout)
    host_elapsed = time.perf_counter_ns() - started
    if returncode != 0:
        raise HarnessError(
            f"{condition} guest exited {returncode}: {' '.join(command)}\n{stderr}"
        )
    guest = parse_guest_result(
        stdout, calls, minimum_interval_ns, enforce_quality=enforce_quality
    )
    elapsed = int(guest["elapsed_ns"])
    return {
        "phase": phase,
        "pair_index": pair_index,
        "position": position,
        "condition": condition,
        "command": command,
        "leaf_calls": calls,
        "leaf_entry_poll_opportunities": calls if condition == "cancel-points-on" else 0,
        "batches": calls // LEAF_UNROLL,
        "guest_elapsed_ns": elapsed,
        "raw_guest_elapsed_ns": int(guest["raw_elapsed_ns"]),
        "timing_overhead_ns": int(guest["timing_overhead_ns"]),
        "timing_overhead_ppm": int(guest["timing_overhead_ppm"]),
        "host_wall_elapsed_ns": host_elapsed,
        "leaf_calls_per_second": calls / (elapsed / 1e9),
        "guest": guest,
        "correct": True,
    }


def condition_order(pair_index: int) -> tuple[str, str]:
    return (
        ("cancel-points-off", "cancel-points-on")
        if pair_index % 2 == 0
        else ("cancel-points-on", "cancel-points-off")
    )


def summarize(records: list[dict[str, Any]], calls: int) -> tuple[list[dict], dict]:
    summaries = []
    for condition in ("cancel-points-off", "cancel-points-on"):
        selected = [item for item in records if item["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "guest_elapsed_ns": sample_stats(
                    (item["guest_elapsed_ns"] for item in selected),
                    "samples",
                ),
                "leaf_calls_per_second": sample_stats(
                    (item["leaf_calls_per_second"] for item in selected),
                    "samples",
                ),
                "host_wall_elapsed_ns": sample_stats(
                    (item["host_wall_elapsed_ns"] for item in selected),
                    "samples",
                ),
            }
        )
    pair_rows = []
    for pair_index in sorted({item["pair_index"] for item in records}):
        pair = {
            item["condition"]: item
            for item in records
            if item["pair_index"] == pair_index
        }
        off = pair["cancel-points-off"]["guest_elapsed_ns"]
        on = pair["cancel-points-on"]["guest_elapsed_ns"]
        pair_rows.append(
            {
                "pair_index": pair_index,
                "on_over_off_elapsed_ratio": on / off,
                "on_minus_off_ns": on - off,
                "on_minus_off_ns_per_leaf_call": (on - off) / calls,
                "on_over_off_throughput_ratio": off / on,
            }
        )
    return summaries, {
        "pairs": pair_rows,
        "median_on_over_off_elapsed_ratio": statistics.median(
            item["on_over_off_elapsed_ratio"] for item in pair_rows
        ),
        "median_elapsed_delta_percent": (
            statistics.median(item["on_over_off_elapsed_ratio"] for item in pair_rows)
            - 1.0
        )
        * 100.0,
        "median_on_minus_off_ns_per_leaf_call": statistics.median(
            item["on_minus_off_ns_per_leaf_call"] for item in pair_rows
        ),
        "median_on_over_off_throughput_ratio": statistics.median(
            item["on_over_off_throughput_ratio"] for item in pair_rows
        ),
    }


def validate_invocations(
    invocations: list[dict[str, Any]],
    *,
    phase: str,
    pairs: int,
    calls: int,
) -> None:
    if len(invocations) != pairs * 2:
        raise HarnessError(f"{phase} invocation count mismatch")
    expected_guest = expected_result(calls)
    for pair_index in range(pairs):
        pair = invocations[pair_index * 2 : pair_index * 2 + 2]
        for position, (item, condition) in enumerate(
            zip(pair, condition_order(pair_index), strict=True)
        ):
            if (
                item.get("phase") != phase
                or item.get("pair_index") != pair_index
                or item.get("position") != position
                or item.get("condition") != condition
            ):
                raise HarnessError(f"{phase} invocation ordering mismatch")
            if item.get("leaf_calls") != calls or item.get("batches") != (
                calls // LEAF_UNROLL
            ):
                raise HarnessError(f"{phase} invocation work mismatch")
            expected_opportunities = calls if condition == "cancel-points-on" else 0
            if (
                item.get("leaf_entry_poll_opportunities")
                != expected_opportunities
            ):
                raise HarnessError(f"{phase} poll opportunity mismatch")
            guest = item.get("guest")
            if not isinstance(guest, dict) or any(
                guest.get(key) != value for key, value in expected_guest.items()
            ):
                raise HarnessError(f"{phase} guest work mismatch")
            if item.get("correct") is not True:
                raise HarnessError(f"{phase} correctness marker mismatch")


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != KIND:
        raise HarnessError("report identity mismatch")
    plan = report.get("plan")
    records = report.get("records")
    if not isinstance(plan, dict) or not isinstance(records, list) or not records:
        raise HarnessError("report plan or records missing")
    calls = plan.get("leaf_calls")
    if not isinstance(calls, int) or isinstance(calls, bool):
        raise HarnessError("report leaf call count is invalid")
    expected_result(calls)
    samples = plan.get("samples")
    warmup_pairs = plan.get("warmups")
    if (
        not isinstance(samples, int)
        or isinstance(samples, bool)
        or not isinstance(warmup_pairs, int)
        or isinstance(warmup_pairs, bool)
    ):
        raise HarnessError("report sample counts are invalid")
    validate_invocations(records, phase="sample", pairs=samples, calls=calls)
    warmups = report.get("warmups")
    if not isinstance(warmups, list):
        raise HarnessError("report warmups missing")
    validate_invocations(
        warmups, phase="warmup", pairs=warmup_pairs, calls=calls
    )
    pilot = report.get("pilot")
    sizing = plan.get("sizing")
    if not isinstance(sizing, dict):
        raise HarnessError("report sizing missing")
    if pilot is None:
        if sizing.get("kind") != "explicit-call-count":
            raise HarnessError("explicit sizing identity mismatch")
    else:
        pilot_calls = sizing.get("pilot_calls")
        if (
            sizing.get("kind") != "single-enabled-pilot-linear-scale"
            or not isinstance(pilot_calls, int)
            or isinstance(pilot_calls, bool)
        ):
            raise HarnessError("pilot sizing identity mismatch")
        if (
            pilot.get("phase") != "pilot"
            or pilot.get("pair_index") != 0
            or pilot.get("position") != 0
            or pilot.get("condition") != "cancel-points-on"
            or pilot.get("leaf_calls") != pilot_calls
            or pilot.get("correct") is not True
        ):
            raise HarnessError("pilot invocation mismatch")
        pilot_guest = pilot.get("guest")
        if not isinstance(pilot_guest, dict):
            raise HarnessError("pilot guest result missing")
        for key, value in expected_result(pilot_calls).items():
            if pilot_guest.get(key) != value:
                raise HarnessError("pilot guest work mismatch")
    expected_non_leaf_bound = calls // LEAF_UNROLL + 1
    expected_leaf_share = calls / (calls + expected_non_leaf_bound)
    if (
        plan.get("non_leaf_poll_opportunities_upper_bound_per_enabled_sample")
        != expected_non_leaf_bound
        or not math.isclose(
            plan.get("minimum_leaf_entry_poll_share", -1),
            expected_leaf_share,
        )
        or sizing.get("selected_calls") != calls
    ):
        raise HarnessError("report poll-opportunity or sizing bound mismatch")
    artifacts = report["metadata"]["artifacts"]
    if artifacts["cancel_poll_sites_enabled"] <= 0:
        raise HarnessError("enabled artifact has no cancel polls")
    if artifacts["cancel_poll_sites_disabled"] != 0:
        raise HarnessError("disabled artifact retains cancel polls")
    quality_invocations = warmups + records
    checksums = {item["guest"]["checksum"] for item in quality_invocations}
    minimum_interval_ns = plan["minimum_interval_ns"]
    expected_quality = {
        "equivalent_guest_work": True,
        "equivalent_checksums": len(checksums) == 1,
        "all_intervals_passed": all(
            item["guest_elapsed_ns"] >= minimum_interval_ns
            for item in quality_invocations
        ),
        "all_timing_overhead_below_one_percent": all(
            99 * item["timing_overhead_ns"] < item["guest_elapsed_ns"]
            for item in quality_invocations
        ),
    }
    if report.get("quality") != expected_quality:
        raise HarnessError("report quality summary mismatch")


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    comparison = report["comparison"]
    source = metadata["source"]
    host = metadata["host"]
    tools = metadata["tools"]
    artifacts = metadata["artifacts"]
    plan = report["plan"]
    lines = [
        "# Leaf-call AOT cancel-poll cost",
        "",
        "## Immutable identities",
        "",
        f"- Source commit: `{source['commit']}`",
        f"- Tracked diff SHA-256: `{source['tracked_diff_sha256']}`",
        f"- Build-source SHA-256: `{source['build_source_sha256']}`",
        f"- Platform: `{metadata['platform_id']}`",
        f"- Host fingerprint: `{host['host_fingerprint']['sha256']}`",
        f"- Host: `{host['system']} {host['release']}` · `{host['machine']}` · "
        f"`{host['cpu']}` · {host['logical_cpus']} logical CPUs · "
        f"`{host['runner_environment']}`",
        f"- AOT architecture: `{artifacts['architecture']}`",
        f"- Fixture: `{metadata['fixture']['sha256']}`",
        f"- Zig: `{tools['zig']}`; Python: `{tools['python']}`",
        f"- wamrc: `{tools['compiler']['wamrc_sha256']}` · "
        f"`{tools['compiler']['wamrc_version']}`",
        f"- wamr: `{tools['runtime']['wamr_sha256']}` · "
        f"`{tools['runtime']['wamr_version']}`",
        "",
        "| AOT condition | SHA-256 | File bytes | Text bytes |",
        "|---|---|---:|---:|",
    ]
    for condition in ("cancel-points-off", "cancel-points-on"):
        artifact = artifacts["conditions"][condition]
        lines.append(
            f"| `{condition}` | `{artifact['sha256']}` | "
            f"{artifact['file_bytes']} | {artifact['text_bytes']} |"
        )
    lines += [
        "",
        "## Measurement plan and validation",
        "",
        f"- Work: `{plan['leaf_calls']}` noinline leaf calls "
        f"(`{plan['batches']}` batches of `{plan['leaf_unroll']}`).",
        f"- Invocations: `{plan['warmups']}` discarded balanced warmup pairs; "
        f"`{plan['samples']}` retained balanced sample pairs.",
        f"- Order: `{plan['pair_order']}`; raw rows below retain execution order.",
        f"- Sizing: `{plan['sizing']['kind']}`; target "
        f"`{plan['target_interval_ns']}` ns; minimum "
        f"`{plan['minimum_interval_ns']}` ns.",
        "- Validation covers every warmup and retained sample; the optional "
        "sizing pilot intentionally does not enforce the minimum interval.",
        f"- CPU affinity: `{metadata['execution']['host_cpu_affinity']}`; "
        f"target `{metadata['execution']['target']}`; "
        f"runner `{metadata['execution']['runner']}`.",
        f"- Timed enabled-path leaf-entry poll opportunities: "
        f"`{plan['leaf_entry_poll_opportunities_per_enabled_sample']}`; "
        "non-leaf timed poll opportunities are bounded by "
        f"`{plan['non_leaf_poll_opportunities_upper_bound_per_enabled_sample']}` "
        f"({plan['minimum_leaf_entry_poll_share']:.6%} minimum leaf-entry share).",
        f"- Static poll sites on/off: "
        f"`{artifacts['cancel_poll_sites_enabled']}` / "
        f"`{artifacts['cancel_poll_sites_disabled']}`; "
        f"`{artifacts['bytes_per_poll_site']}` bytes/site.",
        "",
        "| Validation | Result |",
        "|---|---|",
    ]
    for key, value in report["quality"].items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    lines += [
        "",
        "## Summary",
        "",
        "| Condition | Guest median ms | Median leaf calls/s | Host-wall median ms |",
        "|---|---:|---:|---:|",
    ]
    for summary in report["summaries"]:
        lines.append(
            f"| `{summary['condition']}` | "
            f"{summary['guest_elapsed_ns']['median'] / 1e6:.3f} | "
            f"{summary['leaf_calls_per_second']['median']:.3f} | "
            f"{summary['host_wall_elapsed_ns']['median'] / 1e6:.3f} |"
        )
    lines += [
        "",
        f"- Median polls-on / polls-off guest time: "
        f"`{comparison['median_on_over_off_elapsed_ratio']:.9f}` "
        f"(`{comparison['median_elapsed_delta_percent']:+.4f}%`).",
        f"- Median polls-on minus polls-off cost per timed leaf call: "
        f"`{comparison['median_on_minus_off_ns_per_leaf_call']:.6f} ns`.",
        f"- Median polls-on / polls-off throughput: "
        f"`{comparison['median_on_over_off_throughput_ratio']:.9f}`.",
        "",
        "The per-leaf-call delta is an amortized enabled-path measurement. "
        "The timed driver has at most one loop-header poll per 64 leaf calls "
        "plus one driver entry poll; the bound and leaf-entry share are retained "
        "above rather than presenting the delta as an uncontaminated single-poll "
        "latency.",
        "",
        "Both conditions use the same wasm fixture, runtime, call count, worker-thread "
        "path, and expected checksum. Only wamrc's benchmark-only cancel-point "
        "suppression flag differs.",
        "",
        "## Raw invocation order",
        "",
        "| Phase | Pair | Position | Condition | Calls | Leaf polls | Checksum | "
        "Raw guest ns | Timer overhead ns | Guest ns | Host-wall ns | Correct | Command |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    invocations = []
    if report["pilot"] is not None:
        invocations.append(report["pilot"])
    invocations.extend(report["warmups"])
    invocations.extend(report["records"])
    for item in invocations:
        command = shlex.join(item["command"]).replace("|", "\\|")
        lines.append(
            f"| `{item['phase']}` | {item['pair_index']} | {item['position']} | "
            f"`{item['condition']}` | {item['leaf_calls']} | "
            f"{item['leaf_entry_poll_opportunities']} | "
            f"{item['guest']['checksum']} | {item['raw_guest_elapsed_ns']} | "
            f"{item['timing_overhead_ns']} | {item['guest_elapsed_ns']} | "
            f"{item['host_wall_elapsed_ns']} | "
            f"`{str(item['correct']).lower()}` | `{command}` |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("zig-out/leaf-cancel-cost")
    )
    parser.add_argument("--calls", type=int)
    parser.add_argument("--pilot-calls", type=int, default=4_194_304)
    parser.add_argument("--target-interval-ms", type=float, default=3000.0)
    parser.add_argument("--min-interval-ms", type=float, default=1250.0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--optimize", default="ReleaseFast")
    parser.add_argument("--target")
    parser.add_argument("--aot-target", choices=("x86_64", "aarch64"))
    parser.add_argument("--runner", default="")
    parser.add_argument(
        "--platform-id",
        default=f"local-{platform.system().lower()}-{platform.machine().lower()}",
    )
    parser.add_argument("--runner-environment")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    for name in ("calls", "pilot_calls"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or value % LEAF_UNROLL != 0):
            parser.error(f"--{name.replace('_', '-')} must be a positive multiple of 64")
        if value is not None and value > MAX_LEAF_CALLS:
            parser.error(
                f"--{name.replace('_', '-')} must not exceed {MAX_LEAF_CALLS}"
            )
    if args.target_interval_ms < args.min_interval_ms or args.min_interval_ms <= 0:
        parser.error("--target-interval-ms must be >= positive --min-interval-ms")
    if (
        args.warmups < 0
        or args.warmups > 2
        or args.samples <= 0
        or args.samples > 12
        or args.samples % 2
    ):
        parser.error(
            "--warmups must be between 0 and 2 and --samples positive, "
            "even, and at most 12"
        )
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def execution_arch(args: argparse.Namespace) -> str:
    if args.aot_target:
        return args.aot_target
    if args.target and args.target.startswith("aarch64"):
        return "aarch64"
    return "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale_report in (output / "report.json", output / "report.md"):
        stale_report.unlink(missing_ok=True)
    source = source_identity(repo)
    fixture = fixture_identity(repo)
    compiler, runtime = build_tools(
        repo, output, source, args.optimize, args.target, args.rebuild
    )
    arch = execution_arch(args)
    artifacts = compile_artifacts(repo, output, compiler, arch)
    artifact_report = artifact_identity(artifacts, arch)
    runner = shlex.split(args.runner)
    minimum_interval_ns = int(args.min_interval_ms * 1_000_000)
    target_interval_ns = int(args.target_interval_ms * 1_000_000)

    pilot = None
    calls = args.calls
    if calls is None:
        pilot = measure(
            repo=repo,
            runner=runner,
            runtime=runtime,
            artifact=artifacts["cancel-points-on"],
            condition="cancel-points-on",
            calls=args.pilot_calls,
            timeout=args.timeout,
            minimum_interval_ns=minimum_interval_ns,
            enforce_quality=False,
            phase="pilot",
            pair_index=0,
            position=0,
        )
        calls = select_calls(
            args.pilot_calls,
            pilot["guest_elapsed_ns"],
            target_interval_ns,
        )

    warmups = []
    for pair_index in range(args.warmups):
        for position, condition in enumerate(condition_order(pair_index)):
            warmups.append(
                measure(
                    repo=repo,
                    runner=runner,
                    runtime=runtime,
                    artifact=artifacts[condition],
                    condition=condition,
                    calls=calls,
                    timeout=args.timeout,
                    minimum_interval_ns=minimum_interval_ns,
                    enforce_quality=True,
                    phase="warmup",
                    pair_index=pair_index,
                    position=position,
                )
            )

    records = []
    for pair_index in range(args.samples):
        for position, condition in enumerate(condition_order(pair_index)):
            records.append(
                measure(
                    repo=repo,
                    runner=runner,
                    runtime=runtime,
                    artifact=artifacts[condition],
                    condition=condition,
                    calls=calls,
                    timeout=args.timeout,
                    minimum_interval_ns=minimum_interval_ns,
                    enforce_quality=True,
                    phase="sample",
                    pair_index=pair_index,
                    position=position,
                )
            )
    summaries, comparison = summarize(records, calls)
    host = host_metadata(args.runner_environment)
    tool_report = {
        "zig": command_identity(["zig", "version"]),
        "python": platform.python_version(),
        "compiler": {
            "build_command": compiler.command,
            "cache_key": compiler.cache_key,
            "wamrc_path": str(compiler.wamrc),
            "wamrc_sha256": sha256_file(compiler.wamrc),
            "wamrc_version": command_identity([str(compiler.wamrc), "version"]),
        },
        "runtime": {
            "build_command": runtime.command,
            "cache_key": runtime.cache_key,
            "wamr_path": str(runtime.wamr),
            "wamr_sha256": sha256_file(runtime.wamr),
            "wamr_version": command_identity([*runner, str(runtime.wamr), "version"]),
        },
    }
    quality_invocations = warmups + records
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "metadata": {
            "collected_at": collected_at(),
            "platform_id": args.platform_id,
            "source": source,
            "fixture": fixture,
            "host": host,
            "tools": tool_report,
            "artifacts": artifact_report,
            "execution": {
                "target": args.target or "native",
                "aot_target": arch,
                "runner": runner,
                "host_cpu_affinity": selected_cpu_affinity(),
                "optimize": args.optimize,
            },
        },
        "plan": {
            "identity": "issue-963-leaf-cancel-cost-v1",
            "scope": "isolated-from-wasi-thread-benchmark-966",
            "leaf_calls": calls,
            "leaf_unroll": LEAF_UNROLL,
            "batches": calls // LEAF_UNROLL,
            "leaf_entry_poll_opportunities_per_enabled_sample": calls,
            "non_leaf_poll_opportunities_upper_bound_per_enabled_sample": (
                calls // LEAF_UNROLL + 1
            ),
            "minimum_leaf_entry_poll_share": calls
            / (calls + calls // LEAF_UNROLL + 1),
            "warmups": args.warmups,
            "samples": args.samples,
            "minimum_interval_ns": minimum_interval_ns,
            "target_interval_ns": target_interval_ns,
            "pair_order": "alternating-off-on/on-off",
            "clock_id": "wasi-process-cputime",
            "sizing": {
                "kind": "single-enabled-pilot-linear-scale"
                if pilot
                else "explicit-call-count",
                "pilot_calls": args.pilot_calls if pilot else None,
                "selected_calls": calls,
            },
        },
        "pilot": pilot,
        "warmups": warmups,
        "records": records,
        "summaries": summaries,
        "comparison": comparison,
        "quality": {
            "equivalent_guest_work": True,
            "equivalent_checksums": len(
                {item["guest"]["checksum"] for item in quality_invocations}
            )
            == 1,
            "all_intervals_passed": all(
                item["guest_elapsed_ns"] >= minimum_interval_ns
                for item in quality_invocations
            ),
            "all_timing_overhead_below_one_percent": all(
                99 * item["timing_overhead_ns"] < item["guest_elapsed_ns"]
                for item in quality_invocations
            ),
        },
    }
    validate_report(report)
    atomic_write_json(output / "report.json", report)
    (output / "report.md").write_text(render_markdown(report), encoding="UTF-8")
    print(output / "report.json")
    print(output / "report.md")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HarnessError, OSError, subprocess.SubprocessError) as exc:
        print(f"leaf cancel-cost benchmark failed: {exc}", file=sys.stderr)
        sys.exit(2)
