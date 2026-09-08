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
REPORT_SCHEMA_VERSION = 4
REVISION_ROLES = ("baseline", "candidate")
SINGLE_REVISION_ROLES = ("candidate",)
COMPARISON_PURPOSES = ("candidate-evaluation", "noise-calibration")
MEASUREMENT_PLAN_IDENTITY_VERSION = 4
MEASUREMENT_PLAN_IDENTITY_KIND = "wasi-thread-measurement-plan"
SIZING_ALGORITHM_VERSION = 3
SIZING_ALGORITHM_KIND = "fastest-valid-one-shot-pilot"
SIZING_FORMULA = (
    "round_up_3_significant_digits(ceil(P*target_duration_ns*"
    "safety_numerator/(E*safety_denominator)))"
)
SIZING_CELL_ENVELOPE_FORMULA = (
    "round_up_3_significant_digits(ceil(P*quality_floor_ns*"
    "acceleration_numerator/(E*acceleration_denominator)))"
)
SIZING_CELL_SELECTION = (
    "max(general_rounded_iterations, applicable_envelope_rounded_iterations)"
)
PROFILE_COUNTS = {
    "authoritative": (2, 10),
    "smoke": (1, 4),
}
ATOMIC_WAIT_PREFLIGHT_RUNS = {
    "authoritative": 64,
    "smoke": 8,
}
MIN_TIMED_INTERVAL_MS = 1_250.0
TIMING_OVERHEAD_RATIO_LIMIT = 0.01
TARGET_BARRIER_NS = 12_456_000
TARGET_BARRIER_REQUIRED_INTERVAL_NS = 99 * TARGET_BARRIER_NS
MINIMUM_INTERVAL_HEADROOM_NS = (
    int(MIN_TIMED_INTERVAL_MS * 1_000_000)
    - TARGET_BARRIER_REQUIRED_INTERVAL_NS
)
SIZING_TARGET_NS = 1_750_000_000
SIZING_SAFETY_NUMERATOR = 11
SIZING_SAFETY_DENOMINATOR = 10
SIZING_SIGNIFICANT_DIGITS = 3
SIZING_CELL_ENVELOPES = (
    {
        "name": "aot-wait-notify-1-rate-acceleration",
        "selector": {
            "mode": "aot",
            "workload": "wait-notify",
            "threads": 1,
        },
        "quality_floor_ns": 1_250_000_000,
        "measurement_to_pilot_rate_envelope": {
            "numerator": 12,
            "denominator": 5,
        },
        "projected_pilot_duration_ns": 3_000_000_000,
        "formula": SIZING_CELL_ENVELOPE_FORMULA,
    },
)
PROJECTED_EVIDENCE_MINIMUM_NS = (
    SIZING_TARGET_NS * SIZING_SAFETY_NUMERATOR
    + SIZING_SAFETY_DENOMINATOR
    - 1
) // SIZING_SAFETY_DENOMINATOR
PILOT_CLOCK_RESOLUTION_MINIMUM_NS = 1_000_000
MAXIMUM_PILOT_CORRECTED_NS = 30_000_000_000
MAXIMUM_PILOT_HOST_WALL_NS = 35_000_000_000
WORKFLOW_JOB_TIMEOUT_NS = 180 * 60 * 1_000_000_000
JOB_NON_BENCHMARK_RESERVE_NS = 83 * 60 * 1_000_000_000
PROJECTED_BENCHMARK_LIMIT_NS = (
    WORKFLOW_JOB_TIMEOUT_NS - JOB_NON_BENCHMARK_RESERVE_NS
)
AUXILIARY_INVOCATION_BUDGET_NS = 10 * 60 * 1_000_000_000
INT32_MAX = (1 << 31) - 1
SIZING_WORKLOAD_CAPS = {
    "single-hot": 16_000_000_000,
    "hot": 16_000_000_000,
    "cancel-hot": 16_000_000_000,
    "atomic": 8_000_000_000,
    "wait-notify": 100_000_000,
    "spawn-join": 1_000_000,
}
TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD = 4
DEFAULT_PILOT_ITERATION_PLAN = {
    "interpreter": {
        "single-hot": 30_000_000,
        "hot": {
            "1": 28_000_000,
            "2": 28_000_000,
            "4": 20_000_000,
            "8": 10_000_000,
        },
        "atomic": {
            "1": 72_000_000,
            "2": 40_000_000,
            "4": 28_000_000,
            "8": 14_000_000,
        },
        "wait-notify": {
            "1": 128_000,
            "2": 64_000,
            "4": 32_000,
            "8": 16_000,
        },
        "spawn-join": {"1": 9_000, "2": 4_500, "4": 2_250, "8": 1_250},
    },
    "aot": {
        "single-hot": 1_900_000_000,
        "hot": {
            "1": 1_800_000_000,
            "2": 1_800_000_000,
            "4": 900_000_000,
            "8": 450_000_000,
        },
        "atomic": {
            "1": 850_000_000,
            "2": 180_000_000,
            "4": 64_000_000,
            "8": 64_000_000,
        },
        "wait-notify": {
            "1": 1_500_000,
            "2": 64_000,
            "4": 32_000,
            "8": 16_000,
        },
        "spawn-join": {"1": 10_000, "2": 5_000, "4": 2_500, "8": 1_250},
        "cancel-hot": {
            "1": 1_900_000_000,
            "2": 1_900_000_000,
            "4": 950_000_000,
            "8": 475_000_000,
        },
    },
}
# Compatibility alias for callers that used the old fixed-count name. These
# counts are pilots only; evidence counts are resolved once per report.
DEFAULT_ITERATION_PLAN = DEFAULT_PILOT_ITERATION_PLAN
LEGACY_ITERATION_DEFAULTS = {
    "single": 224_000_000,
    "cancel": 224_000_000,
    "hot": 128_000_000,
    "atomic": 64_000_000,
    "atomic_total": 256_000_000,
    "wait": 512_000,
    "spawn": 3_000,
}
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
HOT_KERNEL_COUNTER_BASE = 0xD1B54A32D192ED03


def canonical_measurement_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return the portable plan, excluding only purpose and host resolutions."""

    require(
        isinstance(plan, dict)
        and "comparison_purpose" in plan
        and "iterations" in plan
        and isinstance(plan.get("sizing"), dict)
        and "resolved" in plan["sizing"],
        "measurement plan comparison_purpose",
    )
    normalized = copy.deepcopy(plan)
    del normalized["comparison_purpose"]
    del normalized["iterations"]
    del normalized["sizing"]["resolved"]
    return normalized


def measurement_plan_sha256(plan: dict[str, Any]) -> str:
    """Hash the canonical portable sizing and measurement contract."""

    return cache_key(
        {
            "schema_version": MEASUREMENT_PLAN_IDENTITY_VERSION,
            "kind": MEASUREMENT_PLAN_IDENTITY_KIND,
            "canonical_plan": canonical_measurement_plan(plan),
        }
    )


class HarnessError(RuntimeError):
    pass


class TimingQualityError(HarnessError):
    def __init__(
        self,
        message: str,
        *,
        raw_elapsed_ns: int,
        timing_overhead_ns: int,
        elapsed_ns: int,
        timing_overhead_ppm: int,
        reason: str,
    ) -> None:
        super().__init__(message)
        self.raw_elapsed_ns = raw_elapsed_ns
        self.timing_overhead_ns = timing_overhead_ns
        self.elapsed_ns = elapsed_ns
        self.timing_overhead_ppm = timing_overhead_ppm
        self.reason = reason


class PreflightProbeError(HarnessError):
    def __init__(
        self,
        message: str,
        *,
        samples: list[dict[str, Any]],
        scenario: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.samples = samples
        self.scenario = scenario


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

    @property
    def key(self) -> str:
        return f"{self.workload}/{self.threads}"


def resolved_iteration_plan(
    args: argparse.Namespace,
    modes: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    plan = {
        mode: copy.deepcopy(DEFAULT_ITERATION_PLAN[mode])
        for mode in modes
    }
    selected_threads = {str(threads) for threads in args.thread_counts}
    for values in plan.values():
        for workload, count in list(values.items()):
            if isinstance(count, dict):
                values[workload] = {
                    thread: iterations
                    for thread, iterations in count.items()
                    if thread in selected_threads
                }

    if args.single_iterations is not None:
        for values in plan.values():
            values["single-hot"] = args.single_iterations
    if args.hot_iterations is not None:
        for values in plan.values():
            values["hot"] = {
                str(threads): args.hot_iterations
                for threads in args.thread_counts
            }
    if (
        args.atomic_iterations is not None
        or args.atomic_total_iterations is not None
    ):
        per_worker = (
            args.atomic_iterations
            if args.atomic_iterations is not None
            else LEGACY_ITERATION_DEFAULTS["atomic"]
        )
        aggregate = (
            args.atomic_total_iterations
            if args.atomic_total_iterations is not None
            else LEGACY_ITERATION_DEFAULTS["atomic_total"]
        )
        for values in plan.values():
            values["atomic"] = {
                str(threads): max(per_worker, aggregate // threads)
                for threads in args.thread_counts
            }
    if args.wait_iterations is not None:
        for values in plan.values():
            values["wait-notify"] = {
                str(threads): args.wait_iterations // threads
                for threads in args.thread_counts
            }
    if args.spawn_iterations is not None:
        for values in plan.values():
            values["spawn-join"] = {
                str(threads): args.spawn_iterations
                for threads in args.thread_counts
            }
    if "aot" in plan and (
        args.cancel_iterations is not None or args.hot_iterations is not None
    ):
        cancel_total = (
            args.cancel_iterations
            if args.cancel_iterations is not None
            else LEGACY_ITERATION_DEFAULTS["cancel"]
        )
        hot_floor = (
            args.hot_iterations
            if args.hot_iterations is not None
            else LEGACY_ITERATION_DEFAULTS["hot"]
        )
        plan["aot"]["cancel-hot"] = {
            str(threads): max(hot_floor, cancel_total // threads)
            for threads in args.thread_counts
        }
    return plan


def validate_iteration_plan_ranges(
    plan: dict[str, dict[str, Any]],
    thread_counts: tuple[int, ...],
) -> None:
    for mode, workloads in plan.items():
        single = workloads["single-hot"]
        if not 0 < single <= MASK64:
            raise HarnessError(f"{mode} single-hot iterations must fit uint64")
        for workload in ("hot", "atomic", "wait-notify", "spawn-join"):
            for threads in thread_counts:
                iterations = workloads[workload][str(threads)]
                if not 0 < iterations <= MASK64 // threads:
                    raise HarnessError(
                        f"{mode} {workload}/{threads} operations overflow uint64"
                    )
                if workload == "wait-notify" and iterations > INT32_MAX:
                    raise HarnessError(
                        f"{mode} wait-notify/{threads} iterations exceed int32"
                    )
                if (
                    workload == "spawn-join"
                    and iterations
                    > MASK64 // (threads * (threads + 1) // 2)
                ):
                    raise HarnessError(
                        f"{mode} spawn-join/{threads} checksum overflows uint64"
                    )
        if mode == "aot":
            for threads in thread_counts:
                iterations = workloads["cancel-hot"][str(threads)]
                if not 0 < iterations <= MASK64 // threads:
                    raise HarnessError(
                        f"aot cancel-hot/{threads} operations overflow uint64"
                    )


def ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise HarnessError("ceiling division requires non-negative/positive inputs")
    return (numerator + denominator - 1) // denominator


def round_up_significant(value: int, significant_digits: int) -> int:
    if value <= 0 or significant_digits <= 0:
        raise HarnessError("sizing rounding inputs must be positive")
    digits = len(str(value))
    quantum = 10 ** max(0, digits - significant_digits)
    return ceil_div(value, quantum) * quantum


def sizing_cell_key(mode: str, workload: str, threads: int) -> str:
    return f"{mode}/{workload}/{threads}"


def sizing_cell_for_condition(
    pair: dict[str, str],
    condition: str,
) -> tuple[str, str, int]:
    if condition not in (pair["left"], pair["right"]):
        raise HarnessError(f"condition {condition!r} is outside {pair['pair_key']}")
    if pair["pair_kind"] == "single-infrastructure":
        return pair["pair_key"].rsplit("/", 1)[1], "single-hot", 1
    if pair["pair_kind"] == "cancel-point-cost":
        return "aot", "cancel-hot", int(pair["pair_key"].rsplit("/", 1)[1])
    _, workload, raw_threads, *_ = pair["pair_key"].split("/")
    if pair["pair_kind"] == "runtime-parity":
        mode = condition
    elif pair["pair_kind"] == "repeatability":
        mode = pair["left"].removesuffix("-a")
    else:
        raise HarnessError(f"unsupported pair kind {pair['pair_kind']!r}")
    return mode, workload, int(raw_threads)


def sizing_algorithm_spec(timeout_seconds: float) -> dict[str, Any]:
    timeout_ns = int(timeout_seconds * 1_000_000_000)
    return {
        "version": SIZING_ALGORITHM_VERSION,
        "kind": SIZING_ALGORITHM_KIND,
        "selection_rate": "fastest-valid-pilot-across-all-revisions-and-conditions",
        "formula": SIZING_FORMULA,
        "cell_selection": SIZING_CELL_SELECTION,
        "cell_envelopes": copy.deepcopy(list(SIZING_CELL_ENVELOPES)),
        "target_duration_ns": SIZING_TARGET_NS,
        "safety_factor": {
            "numerator": SIZING_SAFETY_NUMERATOR,
            "denominator": SIZING_SAFETY_DENOMINATOR,
        },
        "rounding": {
            "kind": "decimal-significant-digits-ceiling",
            "significant_digits": SIZING_SIGNIFICANT_DIGITS,
        },
        "pilot_clock_resolution_minimum_ns": (
            PILOT_CLOCK_RESOLUTION_MINIMUM_NS
        ),
        "maximum_pilot_corrected_ns": MAXIMUM_PILOT_CORRECTED_NS,
        "maximum_pilot_host_wall_ns": MAXIMUM_PILOT_HOST_WALL_NS,
        "projected_evidence_minimum_ns": PROJECTED_EVIDENCE_MINIMUM_NS,
        "timing_overhead_ratio_limit": TIMING_OVERHEAD_RATIO_LIMIT,
        "limits": {
            "uint64_max": MASK64,
            "wait_notify_int32_max": INT32_MAX,
            "workload_iteration_caps": copy.deepcopy(SIZING_WORKLOAD_CAPS),
            "per_invocation_timeout_ns": timeout_ns,
            "workflow_job_timeout_ns": WORKFLOW_JOB_TIMEOUT_NS,
            "job_non_benchmark_reserve_ns": JOB_NON_BENCHMARK_RESERVE_NS,
            "projected_benchmark_limit_ns": PROJECTED_BENCHMARK_LIMIT_NS,
            "auxiliary_invocation_budget_ns": AUXILIARY_INVOCATION_BUDGET_NS,
        },
        "failure_policy": "one-shot-no-retry-no-discard",
    }


def pilot_order_for_plan(
    pairs: list[dict[str, str]],
    revision_roles: tuple[str, ...],
    pilot_plan: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    order = []
    for pair in pairs:
        for revision in revision_roles:
            for condition in (pair["left"], pair["right"]):
                mode, workload, threads = sizing_cell_for_condition(
                    pair, condition
                )
                order.append(
                    {
                        "pilot_index": len(order),
                        "revision": revision,
                        "pair_kind": pair["pair_kind"],
                        "pair_key": pair["pair_key"],
                        "condition": condition,
                        "mode": mode,
                        "workload": workload,
                        "threads": threads,
                        "iterations": iteration_count(
                            pilot_plan, mode, workload, threads
                        ),
                    }
                )
    return order


def effective_sizing_cap(workload: str, threads: int) -> int:
    cap = SIZING_WORKLOAD_CAPS[workload]
    cap = min(cap, MASK64 // threads)
    if workload == "wait-notify":
        cap = min(cap, INT32_MAX)
    if workload == "spawn-join":
        cap = min(cap, MASK64 // (threads * (threads + 1) // 2))
    return cap


def selected_iterations_from_elapsed(
    pilot_iterations: int,
    pilot_elapsed_ns: int,
) -> tuple[int, int]:
    if pilot_iterations <= 0 or pilot_elapsed_ns <= 0:
        raise HarnessError("pilot sizing inputs must be positive")
    required = ceil_div(
        pilot_iterations * SIZING_TARGET_NS * SIZING_SAFETY_NUMERATOR,
        pilot_elapsed_ns * SIZING_SAFETY_DENOMINATOR,
    )
    return required, round_up_significant(
        required, SIZING_SIGNIFICANT_DIGITS
    )


def sizing_candidates_for_cell(
    mode: str,
    workload: str,
    threads: int,
    pilot_iterations: int,
    pilot_elapsed_ns: int,
) -> dict[str, Any]:
    general_required, general_rounded = selected_iterations_from_elapsed(
        pilot_iterations, pilot_elapsed_ns
    )
    envelope_candidates = []
    selector = {
        "mode": mode,
        "workload": workload,
        "threads": threads,
    }
    for envelope in SIZING_CELL_ENVELOPES:
        if envelope["selector"] != selector:
            continue
        acceleration = envelope["measurement_to_pilot_rate_envelope"]
        required = ceil_div(
            pilot_iterations
            * envelope["quality_floor_ns"]
            * acceleration["numerator"],
            pilot_elapsed_ns * acceleration["denominator"],
        )
        envelope_candidates.append(
            {
                "name": envelope["name"],
                "required_iterations": required,
                "rounded_iterations": round_up_significant(
                    required, SIZING_SIGNIFICANT_DIGITS
                ),
            }
        )
    selected_required = max(
        [general_required]
        + [item["required_iterations"] for item in envelope_candidates]
    )
    selected = max(
        [general_rounded]
        + [item["rounded_iterations"] for item in envelope_candidates]
    )
    selected_sources = (
        ["general"] if general_rounded == selected else []
    ) + [
        item["name"]
        for item in envelope_candidates
        if item["rounded_iterations"] == selected
    ]
    return {
        "general_required_iterations": general_required,
        "general_rounded_iterations": general_rounded,
        "applicable_envelopes": envelope_candidates,
        "selected_required_iterations": selected_required,
        "selected_iterations": selected,
        "selected_sources": selected_sources,
    }


def validate_sizing_pilot(
    record: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for key in (
        "pilot_index",
        "revision",
        "pair_kind",
        "pair_key",
        "condition",
        "mode",
        "workload",
        "threads",
        "iterations",
    ):
        if record.get(key) != expected[key]:
            raise HarnessError(f"sizing pilot order mismatch for {key}")
    if record.get("correct") is not True:
        raise HarnessError("sizing pilot correctness failed")
    elapsed = record.get("guest_elapsed_ns")
    overhead = record.get("timing_overhead_ns")
    raw = record.get("raw_guest_elapsed_ns")
    host_wall = record.get("host_wall_elapsed_ns")
    if (
        not isinstance(elapsed, int)
        or isinstance(elapsed, bool)
        or elapsed < PILOT_CLOCK_RESOLUTION_MINIMUM_NS
    ):
        raise HarnessError("sizing pilot clock resolution is insufficient")
    if elapsed > MAXIMUM_PILOT_CORRECTED_NS:
        raise HarnessError("sizing pilot corrected duration exceeds 30 seconds")
    if (
        not isinstance(host_wall, int)
        or isinstance(host_wall, bool)
        or host_wall < elapsed
        or host_wall > MAXIMUM_PILOT_HOST_WALL_NS
    ):
        raise HarnessError("sizing pilot host-wall duration exceeds 35 seconds")
    if (
        not isinstance(overhead, int)
        or isinstance(overhead, bool)
        or overhead < 0
        or not isinstance(raw, int)
        or isinstance(raw, bool)
        or raw != elapsed + overhead
    ):
        raise HarnessError("sizing pilot barrier diagnostics are invalid")
    expected_operations = expected_result(
        "hot" if expected["workload"] == "cancel-hot" else expected["workload"],
        expected["threads"],
        expected["iterations"],
    )["operations"]
    if record.get("operations") != expected_operations:
        raise HarnessError("sizing pilot operation count mismatch")


def pilot_progress_bound(
    *,
    pilot_records: list[dict[str, Any]],
    total_pilots: int,
    warmups: int,
    samples: int,
) -> dict[str, int]:
    completed_wall_ns = sum(
        record["host_wall_elapsed_ns"] for record in pilot_records
    )
    completed_elapsed_ns = sum(
        record["guest_elapsed_ns"] for record in pilot_records
    )
    remaining_pilots = total_pilots - len(pilot_records)
    if remaining_pilots < 0:
        raise HarnessError("sizing pilot progress exceeds declared order")
    remaining_pilot_bound_ns = (
        remaining_pilots * MAXIMUM_PILOT_HOST_WALL_NS
    )
    minimum_evidence_bound_ns = (
        total_pilots
        * (warmups + samples)
        * PROJECTED_EVIDENCE_MINIMUM_NS
    )
    earliest_complete_bound_ns = (
        completed_wall_ns
        + remaining_pilot_bound_ns
        + minimum_evidence_bound_ns
        + AUXILIARY_INVOCATION_BUDGET_NS
    )
    if earliest_complete_bound_ns >= PROJECTED_BENCHMARK_LIMIT_NS:
        raise HarnessError(
            "sizing pilot progress cannot fit the 97-minute benchmark bound"
        )
    return {
        "completed_pilot_elapsed_ns": completed_elapsed_ns,
        "completed_pilot_host_wall_ns": completed_wall_ns,
        "remaining_pilot_bound_ns": remaining_pilot_bound_ns,
        "minimum_evidence_bound_ns": minimum_evidence_bound_ns,
        "earliest_complete_bound_ns": earliest_complete_bound_ns,
    }


def empty_iteration_plan(
    modes: tuple[str, ...],
    thread_counts: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for mode in modes:
        result[mode] = {"single-hot": 0}
        for workload in ("hot", "atomic", "wait-notify", "spawn-join"):
            result[mode][workload] = {
                str(threads): 0 for threads in thread_counts
            }
        if mode == "aot":
            result[mode]["cancel-hot"] = {
                str(threads): 0 for threads in thread_counts
            }
    return result


def set_iteration_count(
    plan: dict[str, dict[str, Any]],
    mode: str,
    workload: str,
    threads: int,
    value: int,
) -> None:
    if workload == "single-hot":
        plan[mode][workload] = value
    else:
        plan[mode][workload][str(threads)] = value


def resolve_one_shot_sizing(
    *,
    pilot_records: list[dict[str, Any]],
    pilot_order: list[dict[str, Any]],
    modes: tuple[str, ...],
    thread_counts: tuple[int, ...],
    warmups: int,
    samples: int,
    timeout_seconds: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if len(pilot_records) != len(pilot_order):
        raise HarnessError("sizing pilot set is incomplete")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for expected, record in zip(pilot_order, pilot_records, strict=True):
        validate_sizing_pilot(record, expected)
        key = (
            expected["mode"],
            expected["workload"],
            expected["threads"],
        )
        grouped.setdefault(key, []).append(record)

    expected_cells = {
        (
            mode,
            workload,
            threads,
        )
        for mode in modes
        for workload in (
            "single-hot",
            "hot",
            "atomic",
            "wait-notify",
            "spawn-join",
            *(("cancel-hot",) if mode == "aot" else ()),
        )
        for threads in ((1,) if workload == "single-hot" else thread_counts)
    }
    if set(grouped) != expected_cells:
        raise HarnessError("sizing pilots do not cover every selected cell")

    selected_plan = empty_iteration_plan(modes, thread_counts)
    cells = []
    selected_by_key: dict[tuple[str, str, int], int] = {}
    for key in sorted(grouped):
        mode, workload, threads = key
        records = grouped[key]
        fastest = min(records, key=lambda item: item["guest_elapsed_ns"])
        baseline_records = [
            item for item in records if item["revision"] == "baseline"
        ]
        baseline_fastest = min(
            baseline_records or records,
            key=lambda item: item["guest_elapsed_ns"],
        )
        baseline_candidates = sizing_candidates_for_cell(
            mode,
            workload,
            threads,
            baseline_fastest["iterations"],
            baseline_fastest["guest_elapsed_ns"],
        )
        fastest_candidates = sizing_candidates_for_cell(
            mode,
            workload,
            threads,
            fastest["iterations"],
            fastest["guest_elapsed_ns"],
        )
        selected = fastest_candidates["selected_iterations"]
        cap = effective_sizing_cap(workload, threads)
        if selected > cap:
            raise HarnessError(
                f"sizing cell {sizing_cell_key(*key)} requires {selected} "
                f"iterations above cap {cap}"
            )
        set_iteration_count(
            selected_plan, mode, workload, threads, selected
        )
        selected_by_key[key] = selected
        cells.append(
            {
                "key": sizing_cell_key(*key),
                "mode": mode,
                "workload": workload,
                "threads": threads,
                "pilot_count": fastest["iterations"],
                "pilot_observations": len(records),
                "baseline_fastest_pilot_index": baseline_fastest["pilot_index"],
                "baseline_fastest_elapsed_ns": baseline_fastest[
                    "guest_elapsed_ns"
                ],
                "baseline_required_iterations": baseline_candidates[
                    "selected_required_iterations"
                ],
                "baseline_rounded_iterations": baseline_candidates[
                    "selected_iterations"
                ],
                "fastest_pilot_index": fastest["pilot_index"],
                "fastest_elapsed_ns": fastest["guest_elapsed_ns"],
                "fastest_required_iterations": fastest_candidates[
                    "selected_required_iterations"
                ],
                "general_required_iterations": fastest_candidates[
                    "general_required_iterations"
                ],
                "general_rounded_iterations": fastest_candidates[
                    "general_rounded_iterations"
                ],
                "applicable_envelopes": fastest_candidates[
                    "applicable_envelopes"
                ],
                "selected_iterations": selected,
                "selected_sources": fastest_candidates["selected_sources"],
                "effective_iteration_cap": cap,
            }
        )

    timeout_ns = int(timeout_seconds * 1_000_000_000)
    projections = []
    projected_evidence_ns = 0
    for record in pilot_records:
        key = (record["mode"], record["workload"], record["threads"])
        selected = selected_by_key[key]
        projected_guest = ceil_div(
            record["guest_elapsed_ns"] * selected, record["iterations"]
        )
        projected_host = ceil_div(
            record["host_wall_elapsed_ns"] * selected, record["iterations"]
        )
        if projected_guest < max(
            int(MIN_TIMED_INTERVAL_MS * 1_000_000),
            PROJECTED_EVIDENCE_MINIMUM_NS,
        ):
            raise HarnessError(
                f"sizing projection is below the evidence minimum for pilot "
                f"{record['pilot_index']}"
            )
        if 99 * record["timing_overhead_ns"] >= projected_guest:
            raise HarnessError(
                f"sizing projected barrier ratio is not below 1% for pilot "
                f"{record['pilot_index']}"
            )
        if projected_guest >= timeout_ns or projected_host >= timeout_ns:
            raise HarnessError(
                f"sizing projection exceeds {timeout_seconds:g}s invocation "
                f"timeout for pilot {record['pilot_index']}"
            )
        projected_evidence_ns += projected_host * (warmups + samples)
        projections.append(
            {
                "pilot_index": record["pilot_index"],
                "selected_iterations": selected,
                "projected_guest_elapsed_ns": projected_guest,
                "projected_host_wall_elapsed_ns": projected_host,
                "projected_timing_overhead_ratio": (
                    record["timing_overhead_ns"]
                    / (projected_guest + record["timing_overhead_ns"])
                ),
                "projected_evidence_minimum_ns": max(
                    int(MIN_TIMED_INTERVAL_MS * 1_000_000),
                    PROJECTED_EVIDENCE_MINIMUM_NS,
                ),
            }
        )
    pilot_host_ns = sum(record["host_wall_elapsed_ns"] for record in pilot_records)
    pilot_elapsed_ns = sum(record["guest_elapsed_ns"] for record in pilot_records)
    maximum_pre_admission_pilot_bound_ns = (
        len(pilot_order) * MAXIMUM_PILOT_HOST_WALL_NS
    )
    projected_evidence_limit_ns = (
        PROJECTED_BENCHMARK_LIMIT_NS
        - pilot_host_ns
        - AUXILIARY_INVOCATION_BUDGET_NS
    )
    if projected_evidence_limit_ns <= 0:
        raise HarnessError("declared pilot set leaves no evidence runtime budget")
    if projected_evidence_ns >= projected_evidence_limit_ns:
        raise HarnessError(
            "sizing evidence projection cannot fit the 97-minute benchmark bound"
        )
    projected_benchmark_ns = (
        pilot_host_ns
        + projected_evidence_ns
        + AUXILIARY_INVOCATION_BUDGET_NS
    )
    if projected_benchmark_ns >= PROJECTED_BENCHMARK_LIMIT_NS:
        raise HarnessError(
            "sizing projection exceeds the benchmark share of the workflow timeout"
        )
    validate_iteration_plan_ranges(selected_plan, thread_counts)
    return selected_plan, {
        "pilots": copy.deepcopy(pilot_records),
        "cells": cells,
        "projections": projections,
        "pilot_corrected_elapsed_ns": pilot_elapsed_ns,
        "pilot_host_wall_elapsed_ns": pilot_host_ns,
        "maximum_pre_admission_pilot_bound_ns": (
            maximum_pre_admission_pilot_bound_ns
        ),
        "projected_evidence_host_wall_ns": projected_evidence_ns,
        "projected_evidence_limit_ns": projected_evidence_limit_ns,
        "auxiliary_invocation_budget_ns": AUXILIARY_INVOCATION_BUDGET_NS,
        "projected_benchmark_ns": projected_benchmark_ns,
        "projected_benchmark_limit_ns": PROJECTED_BENCHMARK_LIMIT_NS,
    }


def validate_sizing_plan(plan: dict[str, Any]) -> None:
    sizing = plan.get("sizing")
    require(
        isinstance(sizing, dict)
        and set(sizing)
        == {"algorithm", "pilot_iterations", "pilot_order", "resolved"},
        "plan.sizing",
    )
    require(
        sizing["algorithm"] == sizing_algorithm_spec(plan["timeout_seconds"]),
        "plan.sizing.algorithm",
    )
    pilot_iterations = sizing["pilot_iterations"]
    require(
        isinstance(pilot_iterations, dict)
        and set(pilot_iterations) == set(plan["modes"]),
        "plan.sizing.pilot_iterations",
    )
    validate_iteration_plan_ranges(
        pilot_iterations, tuple(plan["thread_counts"])
    )
    revision_roles = tuple(plan["revision_roles"])
    expected_pilot_order = pilot_order_for_plan(
        plan["pairs"], revision_roles, pilot_iterations
    )
    require(
        sizing["pilot_order"] == expected_pilot_order,
        "plan.sizing.pilot_order",
    )
    recomputed_iterations, recomputed_resolution = resolve_one_shot_sizing(
        pilot_records=sizing["resolved"].get("pilots", []),
        pilot_order=expected_pilot_order,
        modes=tuple(plan["modes"]),
        thread_counts=tuple(plan["thread_counts"]),
        warmups=plan["warmups"],
        samples=plan["samples"],
        timeout_seconds=plan["timeout_seconds"],
    )
    require(
        plan["iterations"] == recomputed_iterations,
        "plan resolved iterations",
    )
    require(
        sizing["resolved"] == recomputed_resolution,
        "plan sizing resolution",
    )


def iteration_count(
    iteration_plan: dict[str, dict[str, Any]],
    mode: str,
    workload: str,
    threads: int,
) -> int:
    value = iteration_plan[mode][workload]
    return value if isinstance(value, int) else value[str(threads)]


def planned_scenarios(args: argparse.Namespace) -> list[Scenario]:
    return [
        Scenario(workload, threads)
        for workload in ("hot", "atomic", "wait-notify", "spawn-join")
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
        require(
            all(workload in iterations[mode] for mode in modes),
            f"plan iterations missing {workload}",
        )
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
    parser.add_argument("--single-iterations", type=int)
    parser.add_argument("--cancel-iterations", type=int)
    parser.add_argument("--hot-iterations", type=int)
    parser.add_argument("--atomic-iterations", type=int)
    parser.add_argument("--atomic-total-iterations", type=int)
    parser.add_argument("--wait-iterations", type=int)
    parser.add_argument("--spawn-iterations", type=int)
    parser.add_argument("--timeout", type=float, default=90.0)
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
    parser.add_argument(
        "--trusted-calibration-preflight",
        action="store_true",
        help=(
            "run the fixed scheduler/barrier quality preflight; valid only for "
            "paired noise calibration"
        ),
    )
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
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.wait_iterations is not None and any(
        args.wait_iterations % threads != 0 for threads in args.thread_counts
    ):
        parser.error(
            "--wait-iterations must be divisible by every selected thread count"
        )
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.min_interval_ms < MIN_TIMED_INTERVAL_MS:
        parser.error(
            f"--min-interval-ms must be >= {MIN_TIMED_INTERVAL_MS:g}"
        )
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
    if args.trusted_calibration_preflight and (
        not paired or args.comparison_purpose != "noise-calibration"
    ):
        parser.error(
            "--trusted-calibration-preflight requires paired noise calibration"
        )
    modes = ("interpreter", "aot") if args.modes == "both" else (args.modes,)
    args.pilot_iteration_plan = resolved_iteration_plan(args, modes)
    args.iteration_plan = args.pilot_iteration_plan
    try:
        validate_iteration_plan_ranges(
            args.pilot_iteration_plan, args.thread_counts
        )
    except HarnessError as exc:
        parser.error(str(exc))
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


def host_quiescence_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "collected_at": collected_at(),
        "logical_cpus": os.cpu_count(),
        "available_cpu_count": None,
        "cpu_affinity": [],
        "load_average": None,
        "proc_loadavg": None,
        "cpu_pressure": None,
        "runner_worker_process_count": None,
        "runner_job": {
            "runner_name": os.getenv("RUNNER_NAME", ""),
            "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
            "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
            "github_job": os.getenv("GITHUB_JOB", ""),
            "github_workflow": os.getenv("GITHUB_WORKFLOW", ""),
        },
    }
    try:
        affinity = sorted(os.sched_getaffinity(0))
        diagnostics["cpu_affinity"] = affinity
        diagnostics["available_cpu_count"] = len(affinity)
    except (AttributeError, OSError):
        pass
    try:
        diagnostics["load_average"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    try:
        diagnostics["proc_loadavg"] = Path("/proc/loadavg").read_text(
            encoding="UTF-8"
        ).strip()
    except OSError:
        pass
    try:
        diagnostics["cpu_pressure"] = Path("/proc/pressure/cpu").read_text(
            encoding="UTF-8"
        ).strip()
    except OSError:
        pass
    try:
        diagnostics["runner_worker_process_count"] = sum(
            1
            for comm in Path("/proc").glob("[0-9]*/comm")
            if comm.read_text(encoding="UTF-8").strip() == "Runner.Worker"
        )
    except OSError:
        pass
    return diagnostics


def maximum_preflight_barrier_ns(minimum_interval_ns: int) -> int:
    if minimum_interval_ns <= 0:
        raise HarnessError("minimum timed interval must be positive")
    return (minimum_interval_ns - 1) // 99


def preflight_sample_accepted(
    timing_overhead_ns: int,
    timed_interval_ns: int,
    minimum_interval_ns: int,
) -> bool:
    return (
        timed_interval_ns >= minimum_interval_ns
        and timing_overhead_ns >= 0
        and 99 * timing_overhead_ns < minimum_interval_ns
    )


def failure_diagnostic_markdown(document: dict[str, Any]) -> str:
    scenario = document["scenario"]
    lines = [
        "# WASI threaded benchmark quality failure",
        "",
        f"- Stage: `{document['stage']}`",
        f"- Reason: `{document['reason']}`",
        f"- Scenario: `{scenario.get('workload', '')}` / "
        f"`{scenario.get('mode', '')}` / {scenario.get('threads', '')} threads",
        f"- Barrier: `{document['timing_overhead_ns']}` ns",
        f"- Timed interval: `{document['timed_interval_ns']}` ns",
        f"- Ratio: `{document['timing_overhead_ratio']}`",
        (
            "- Ratio at minimum interval: "
            f"`{document['ratio_at_minimum_timed_interval']}`"
        ),
        f"- Fixed limit: `< {document['timing_overhead_ratio_limit']}`",
        "- Pilot clock-resolution minimum / corrected cap / host-wall cap: "
        f"`{document['pilot_clock_resolution_minimum_ns']}` / "
        f"`{document['maximum_pilot_corrected_ns']}` / "
        f"`{document['maximum_pilot_host_wall_ns']}` ns",
        "- Projected evidence minimum / benchmark limit / job reserve: "
        f"`{document['projected_evidence_minimum_ns']}` / "
        f"`{document['projected_benchmark_limit_ns']}` / "
        f"`{document['job_non_benchmark_reserve_ns']}` ns",
        f"- Runner: `{document['host'].get('runner_name', '')}`",
        f"- Host fingerprint: "
        f"`{document['host_pair']['host_fingerprint_sha256']}`",
        "",
        f"All {len(document['preflight_samples'])} preflight samples are retained "
        "in `failure-diagnostic.json`.",
        "",
    ]
    return "\n".join(lines)


def write_failure_diagnostic(
    *,
    output: Path,
    stage: str,
    reason: str,
    scenario: dict[str, Any],
    timing_overhead_ns: int | None,
    timed_interval_ns: int | None,
    raw_elapsed_ns: int | None,
    timing_overhead_ppm: int | None,
    minimum_interval_ns: int,
    host: dict[str, Any],
    host_pair: dict[str, str],
    host_quiescence_at_start: dict[str, Any],
    host_quiescence_at_failure: dict[str, Any],
    preflight_samples: list[dict[str, Any]],
    message: str,
    ratio_at_minimum_timed_interval: float | None = None,
) -> dict[str, Any]:
    ratio = (
        timing_overhead_ns / raw_elapsed_ns
        if timing_overhead_ns is not None
        and raw_elapsed_ns is not None
        and raw_elapsed_ns > 0
        else None
    )
    if (
        ratio_at_minimum_timed_interval is None
        and timing_overhead_ns is not None
    ):
        ratio_at_minimum_timed_interval = timing_overhead_ns / (
            minimum_interval_ns + timing_overhead_ns
        )
    document = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "wasi-thread-benchmark-quality-failure",
        "collected_at": collected_at(),
        "stage": stage,
        "reason": reason,
        "message": message,
        "scenario": scenario,
        "timing_overhead_ns": timing_overhead_ns,
        "timed_interval_ns": timed_interval_ns,
        "raw_elapsed_ns": raw_elapsed_ns,
        "timing_overhead_ppm": timing_overhead_ppm,
        "timing_overhead_ratio": ratio,
        "ratio_at_minimum_timed_interval": (
            ratio_at_minimum_timed_interval
        ),
        "timing_overhead_ratio_limit": TIMING_OVERHEAD_RATIO_LIMIT,
        "pilot_clock_resolution_minimum_ns": (
            PILOT_CLOCK_RESOLUTION_MINIMUM_NS
        ),
        "maximum_pilot_corrected_ns": MAXIMUM_PILOT_CORRECTED_NS,
        "maximum_pilot_host_wall_ns": MAXIMUM_PILOT_HOST_WALL_NS,
        "projected_evidence_minimum_ns": PROJECTED_EVIDENCE_MINIMUM_NS,
        "projected_benchmark_limit_ns": PROJECTED_BENCHMARK_LIMIT_NS,
        "job_non_benchmark_reserve_ns": JOB_NON_BENCHMARK_RESERVE_NS,
        "minimum_timed_interval_ns": minimum_interval_ns,
        "maximum_preflight_barrier_ns": maximum_preflight_barrier_ns(
            minimum_interval_ns
        ),
        "host": host,
        "host_pair": host_pair,
        "host_quiescence_at_start": host_quiescence_at_start,
        "host_quiescence_at_failure": host_quiescence_at_failure,
        "preflight_samples": preflight_samples,
    }
    atomic_write_json(output / "failure-diagnostic.json", document)
    (output / "failure-diagnostic.md").write_text(
        failure_diagnostic_markdown(document) + "\n",
        encoding="UTF-8",
    )
    return document


def raise_with_failure_diagnostic(
    error: HarnessError,
    **diagnostic: Any,
) -> None:
    write_failure_diagnostic(**diagnostic, message=str(error))
    raise error


def measure_with_quality_diagnostic(
    *,
    output: Path,
    stage: str,
    minimum_interval_ns: int,
    host: dict[str, Any],
    host_pair: dict[str, str],
    host_quiescence_at_start: dict[str, Any],
    preflight_samples: list[dict[str, Any]],
    **measure_args: Any,
) -> dict[str, Any]:
    try:
        return measure_once(**measure_args)
    except TimingQualityError as exc:
        fields = measure_args["record_fields"]
        raise_with_failure_diagnostic(
            exc,
            output=output,
            stage=stage,
            reason=exc.reason,
            scenario={
                "revision": fields.get("revision"),
                "mode": fields.get("mode"),
                "workload": measure_args["workload"],
                "threads": measure_args["threads"],
                "iterations": measure_args["iterations"],
                "condition": fields.get("condition"),
                "pair_key": fields.get("pair_key"),
                "pair_index": fields.get("pair_index"),
                "phase": fields.get("phase"),
            },
            timing_overhead_ns=exc.timing_overhead_ns,
            timed_interval_ns=exc.elapsed_ns,
            raw_elapsed_ns=exc.raw_elapsed_ns,
            timing_overhead_ppm=exc.timing_overhead_ppm,
            minimum_interval_ns=minimum_interval_ns,
            host=host,
            host_pair=host_pair,
            host_quiescence_at_start=host_quiescence_at_start,
            host_quiescence_at_failure=host_quiescence_diagnostics(),
            preflight_samples=preflight_samples,
        )
    except HarnessError as exc:
        fields = measure_args["record_fields"]
        raise_with_failure_diagnostic(
            exc,
            output=output,
            stage=stage,
            reason="execution-or-correctness-failure",
            scenario={
                "revision": fields.get("revision"),
                "mode": fields.get("mode"),
                "workload": fields.get("workload"),
                "threads": measure_args["threads"],
                "iterations": measure_args["iterations"],
                "condition": fields.get("condition"),
                "pair_key": fields.get("pair_key"),
                "pair_index": fields.get("pair_index"),
                "phase": fields.get("phase"),
            },
            timing_overhead_ns=None,
            timed_interval_ns=None,
            raw_elapsed_ns=None,
            timing_overhead_ppm=None,
            minimum_interval_ns=minimum_interval_ns,
            host=host,
            host_pair=host_pair,
            host_quiescence_at_start=host_quiescence_at_start,
            host_quiescence_at_failure=host_quiescence_diagnostics(),
            preflight_samples=preflight_samples,
        )


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


def rotate_left_u64(value: int, bits: int) -> int:
    shift = bits & 63
    value &= MASK64
    if shift == 0:
        return value
    return ((value << shift) & MASK64) | (value >> (64 - shift))


def xor_upto(value: int) -> int:
    if value < 0:
        return 0
    return (value, 1, value + 1, 0)[value & 3]


def xor_wrapped_u58_range(start: int, count: int) -> int:
    modulus = 1 << 58
    if not 0 <= count <= modulus:
        raise HarnessError("hot-kernel grouped range exceeds uint64 iteration space")
    if count == 0 or count == modulus:
        return 0
    start %= modulus
    end = start + count
    if end <= modulus:
        return xor_upto(end - 1) ^ xor_upto(start - 1)
    return (
        xor_upto(modulus - 1)
        ^ xor_upto(start - 1)
        ^ xor_upto(end - modulus - 1)
    )


@functools.lru_cache(maxsize=None)
def hot_kernel_jump_ahead(seed: int, iterations: int) -> int:
    if not 0 <= iterations <= MASK64:
        raise HarnessError("hot-kernel iterations must fit uint64")
    value = rotate_left_u64(seed, 7 * iterations)
    for residue in range(min(64, iterations)):
        terms = (iterations - 1 - residue) // 64 + 1
        first = (HOT_KERNEL_COUNTER_BASE + residue) & MASK64
        grouped_xor = xor_wrapped_u58_range(first >> 6, terms) << 6
        if terms & 1:
            grouped_xor ^= first & 63
        value ^= rotate_left_u64(
            grouped_xor,
            7 * (iterations - 1 - residue),
        )
    return value & MASK64


def worker_seed(index: int) -> int:
    return (
        0x243F6A8885A308D3 ^ (0x9E3779B97F4A7C15 * (index + 1))
    ) & MASK64


@functools.lru_cache(maxsize=None)
def expected_result(
    workload: str, threads: int, iterations: int
) -> dict[str, int | str]:
    if not 0 < threads <= 8 or not 0 <= iterations <= MASK64 // threads:
        raise HarnessError("expected-result operations must fit uint64")
    operations = threads * iterations
    if workload == "single-hot":
        checksum = hot_kernel_jump_ahead(worker_seed(0), iterations)
    elif workload == "hot":
        checksum = sum(
            hot_kernel_jump_ahead(worker_seed(index), iterations)
            for index in range(threads)
        ) & MASK64
    elif workload in ("atomic", "wait-notify"):
        checksum = operations
    elif workload == "spawn-join":
        checksum = iterations * threads * (threads + 1) // 2
        if checksum > MASK64:
            raise HarnessError("spawn-join checksum must fit uint64")
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


def prepare_expected_results(
    iteration_plan: dict[str, dict[str, Any]],
    modes: tuple[str, ...],
    thread_counts: tuple[int, ...],
) -> dict[str, Any]:
    requests: list[tuple[str, int, int]] = []
    for mode in modes:
        requests.append(
            ("single-hot", 1, int(iteration_plan[mode]["single-hot"]))
        )
        for workload in ("hot", "atomic", "wait-notify", "spawn-join"):
            for threads in thread_counts:
                requests.append(
                    (
                        workload,
                        threads,
                        iteration_count(
                            iteration_plan, mode, workload, threads
                        ),
                    )
                )
    if "aot" in modes:
        for threads in thread_counts:
            requests.append(
                (
                    "hot",
                    threads,
                    iteration_count(
                        iteration_plan, "aot", "cancel-hot", threads
                    ),
                )
            )
    unique_requests = list(dict.fromkeys(requests))
    samples = []
    total_started = time.perf_counter_ns()
    for workload, threads, iterations in unique_requests:
        started = time.perf_counter_ns()
        expected_result(workload, threads, iterations)
        elapsed_ns = time.perf_counter_ns() - started
        samples.append(
            {
                "workload": workload,
                "threads": threads,
                "iterations": iterations,
                "elapsed_ns": elapsed_ns,
            }
        )
    total_ns = time.perf_counter_ns() - total_started
    worst = max(samples, key=lambda sample: sample["elapsed_ns"])
    return {
        "algorithm": "64-residue-xor-jump-ahead",
        "complexity": "O(64 * threads), independent of iteration count",
        "unique_keys": len(samples),
        "total_ns": total_ns,
        "worst_ns": worst["elapsed_ns"],
        "worst_key": {
            key: worst[key] for key in ("workload", "threads", "iterations")
        },
    }


def parse_guest_result(
    stdout: str,
    expected: dict[str, int | str],
    min_interval_ns: int,
    *,
    enforce_timing_quality: bool = True,
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
    if enforce_timing_quality and 99 * overhead >= elapsed:
        raise TimingQualityError(
            f"guest timing overhead {overhead / raw * 100:.3f}% is not below 1%",
            raw_elapsed_ns=raw,
            timing_overhead_ns=overhead,
            elapsed_ns=elapsed,
            timing_overhead_ppm=overhead_ppm,
            reason="timing-overhead",
        )
    if enforce_timing_quality and elapsed < min_interval_ns:
        raise TimingQualityError(
            f"guest timed interval {elapsed}ns is below required {min_interval_ns}ns",
            raw_elapsed_ns=raw,
            timing_overhead_ns=overhead,
            elapsed_ns=elapsed,
            timing_overhead_ppm=overhead_ppm,
            reason="minimum-timed-interval",
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
    enforce_timing_quality: bool = True,
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
    guest = parse_guest_result(
        stdout,
        expected,
        min_interval_ns,
        enforce_timing_quality=enforce_timing_quality,
    )
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


def run_trusted_barrier_preflight(
    *,
    repo: Path,
    runner: list[str],
    build: Build,
    module: Path,
    thread_counts: tuple[int, ...],
    iterations_by_thread: dict[str, int],
    timeout: float,
    minimum_interval_ns: int,
    static_cancel_poll_sites: int,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for threads in thread_counts:
        iterations = iterations_by_thread[str(threads)]
        for probe_index in range(TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD):
            try:
                measured = measure_once(
                    repo=repo,
                    runner=runner,
                    build=build,
                    module=module,
                    workload="hot",
                    threads=threads,
                    iterations=iterations,
                    timeout=timeout,
                    min_interval_ns=minimum_interval_ns,
                    enforce_timing_quality=False,
                    record_fields={
                        "mode": "aot",
                        "threads_enabled": True,
                        "cancel_points": "on",
                        "static_cancel_poll_sites": static_cancel_poll_sites,
                        "workload": "hot",
                        "threads": threads,
                        "iterations": iterations,
                    },
                )
            except HarnessError as exc:
                raise PreflightProbeError(
                    str(exc),
                    samples=copy.deepcopy(samples),
                    scenario={
                        "mode": "aot",
                        "workload": "hot",
                        "threads": threads,
                        "probe_index": probe_index,
                    },
                ) from exc
            overhead = measured["timing_overhead_ns"]
            timed = measured["guest_elapsed_ns"]
            raw = measured["raw_guest_elapsed_ns"]
            samples.append(
                {
                    "probe_index": probe_index,
                    "mode": "aot",
                    "workload": "hot",
                    "threads": threads,
                    "iterations": iterations,
                    "timing_overhead_ns": overhead,
                    "timed_interval_ns": timed,
                    "raw_elapsed_ns": raw,
                    "timing_overhead_ppm": measured["timing_overhead_ppm"],
                    "timing_overhead_ratio": overhead / raw,
                    "ratio_at_minimum_timed_interval": (
                        overhead / (minimum_interval_ns + overhead)
                    ),
                    "accepted": preflight_sample_accepted(
                        overhead,
                        timed,
                        minimum_interval_ns,
                    ),
                }
            )
    overhead_values = [sample["timing_overhead_ns"] for sample in samples]
    return {
        "enabled": True,
        "status": (
            "passed" if all(sample["accepted"] for sample in samples) else "failed"
        ),
        "probe_count": len(samples),
        "probes_per_thread": TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD,
        "mode": "aot",
        "workload": "hot",
        "thread_counts": list(thread_counts),
        "minimum_timed_interval_ns": minimum_interval_ns,
        "timing_overhead_ratio_limit": TIMING_OVERHEAD_RATIO_LIMIT,
        "maximum_accepted_barrier_ns": maximum_preflight_barrier_ns(
            minimum_interval_ns
        ),
        "acceptance_rule": (
            "every probe must have timed_interval_ns >= minimum_timed_interval_ns "
            "and 99 * timing_overhead_ns < minimum_timed_interval_ns"
        ),
        "summary": {
            "minimum_barrier_ns": min(overhead_values),
            "median_barrier_ns": statistics.median(overhead_values),
            "maximum_barrier_ns": max(overhead_values),
        },
        "samples": samples,
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
    validate_common_report(document, KIND, REPORT_SCHEMA_VERSION)
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
        plan.get("minimum_timed_interval_ns", 0)
        >= int(MIN_TIMED_INTERVAL_MS * 1_000_000),
        "plan.minimum_timed_interval_ns",
    )
    preflight_plan = plan.get("scheduler_barrier_preflight")
    require(
        isinstance(preflight_plan, dict),
        "plan.scheduler_barrier_preflight",
    )
    require(
        isinstance(preflight_plan.get("enabled"), bool)
        and isinstance(preflight_plan.get("acceptance_rule"), str)
        and bool(preflight_plan["acceptance_rule"])
        and preflight_plan.get("mode") == "aot"
        and preflight_plan.get("workload") == "hot"
        and preflight_plan.get("probes_per_thread")
        == TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD
        and preflight_plan.get("timing_overhead_ratio_limit")
        == TIMING_OVERHEAD_RATIO_LIMIT
        and preflight_plan.get("target_barrier_ns") == TARGET_BARRIER_NS
        and preflight_plan.get("target_required_interval_ns")
        == TARGET_BARRIER_REQUIRED_INTERVAL_NS
        and preflight_plan.get("minimum_interval_headroom_ns")
        == plan["minimum_timed_interval_ns"]
        - TARGET_BARRIER_REQUIRED_INTERVAL_NS
        and preflight_plan.get("maximum_accepted_barrier_ns")
        == maximum_preflight_barrier_ns(plan["minimum_timed_interval_ns"]),
        "plan.scheduler_barrier_preflight policy",
    )
    iterations = plan.get("iterations")
    require(
        isinstance(iterations, dict) and set(iterations) == set(plan["modes"]),
        "plan.iterations modes",
    )
    thread_keys = {str(threads) for threads in plan["thread_counts"]}
    for mode, values in iterations.items():
        expected_workloads = {
            "single-hot",
            "hot",
            "atomic",
            "wait-notify",
            "spawn-join",
        } | ({"cancel-hot"} if mode == "aot" else set())
        require(
            isinstance(values, dict) and set(values) == expected_workloads,
            f"plan.iterations.{mode} workloads",
        )
        require(
            isinstance(values["single-hot"], int)
            and not isinstance(values["single-hot"], bool)
            and values["single-hot"] > 0,
            f"plan.iterations.{mode}.single-hot",
        )
        for workload in expected_workloads - {"single-hot"}:
            counts = values[workload]
            require(
                isinstance(counts, dict) and set(counts) == thread_keys,
                f"plan.iterations.{mode}.{workload} threads",
            )
            require(
                all(
                    isinstance(count, int)
                    and not isinstance(count, bool)
                    and count > 0
                    for count in counts.values()
                ),
                f"plan.iterations.{mode}.{workload} counts",
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
    expected_preflight_count = (
        TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD
        * len(plan.get("thread_counts", []))
        if preflight_plan.get("enabled") is True
        else 0
    )
    require(
        preflight_plan.get("probe_count") == expected_preflight_count,
        "plan.scheduler_barrier_preflight probe count",
    )
    if preflight_plan["enabled"]:
        require(
            revision_mode == "paired-revisions"
            and comparison_purpose == "noise-calibration"
            and "aot" in plan.get("modes", []),
            "trusted quality preflight activation",
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
    quality_preflight = document.get("quality_preflight")
    require(isinstance(quality_preflight, dict), "quality_preflight")
    require(
        quality_preflight.get("enabled") is preflight_plan.get("enabled"),
        "quality_preflight enabled",
    )
    require(
        quality_preflight.get("probe_count") == expected_preflight_count
        and quality_preflight.get("probes_per_thread")
        == TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD
        and quality_preflight.get("minimum_timed_interval_ns")
        == plan["minimum_timed_interval_ns"]
        and quality_preflight.get("timing_overhead_ratio_limit")
        == TIMING_OVERHEAD_RATIO_LIMIT
        and quality_preflight.get("maximum_accepted_barrier_ns")
        == maximum_preflight_barrier_ns(plan["minimum_timed_interval_ns"])
        and quality_preflight.get("mode") == preflight_plan["mode"]
        and quality_preflight.get("workload") == preflight_plan["workload"]
        and quality_preflight.get("thread_counts") == plan["thread_counts"]
        and quality_preflight.get("acceptance_rule")
        == preflight_plan["acceptance_rule"]
        and isinstance(
            quality_preflight.get("host_quiescence_at_start"), dict
        ),
        "quality_preflight policy",
    )
    preflight_samples = quality_preflight.get("samples")
    require(isinstance(preflight_samples, list), "quality_preflight samples")
    require(
        len(preflight_samples) == expected_preflight_count,
        "quality_preflight sample count",
    )
    if preflight_plan["enabled"]:
        require(
            quality_preflight.get("status") == "passed",
            "quality_preflight status",
        )
        require(
            all(
                sample.get("accepted") is True
                and preflight_sample_accepted(
                    sample.get("timing_overhead_ns", -1),
                    sample.get("timed_interval_ns", -1),
                    plan["minimum_timed_interval_ns"],
                )
                for sample in preflight_samples
            ),
            "quality_preflight acceptance",
        )
        expected_probe_order = [
            (threads, probe_index)
            for threads in plan["thread_counts"]
            for probe_index in range(
                TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD
            )
        ]
        require(
            [
                (sample.get("threads"), sample.get("probe_index"))
                for sample in preflight_samples
            ]
            == expected_probe_order,
            "quality_preflight probe order",
        )
        for sample in preflight_samples:
            overhead = sample["timing_overhead_ns"]
            timed = sample["timed_interval_ns"]
            raw = sample.get("raw_elapsed_ns")
            require(
                sample.get("mode") == "aot"
                and sample.get("workload") == "hot"
                and sample.get("iterations")
                == plan["iterations"]["aot"]["hot"][str(sample["threads"])]
                and raw == timed + overhead
                and sample.get("timing_overhead_ppm")
                == overhead * 1_000_000 // raw
                and math.isclose(
                    sample.get("timing_overhead_ratio", -1),
                    overhead / raw,
                )
                and math.isclose(
                    sample.get("ratio_at_minimum_timed_interval", -1),
                    overhead
                    / (plan["minimum_timed_interval_ns"] + overhead),
                ),
                "quality_preflight sample",
            )
        overhead_values = [
            sample["timing_overhead_ns"] for sample in preflight_samples
        ]
        require(
            quality_preflight.get("summary")
            == {
                "minimum_barrier_ns": min(overhead_values),
                "median_barrier_ns": statistics.median(overhead_values),
                "maximum_barrier_ns": max(overhead_values),
            },
            "quality_preflight summary",
        )
    else:
        require(
            quality_preflight.get("status") == "not-requested"
            and preflight_samples == [],
            "disabled quality_preflight",
        )
    pair_plan = plan.get("pairs")
    require(isinstance(pair_plan, list) and pair_plan, "plan.pairs")
    expected_pair_plan = expected_pair_specs_for_plan(plan)
    require(pair_plan == expected_pair_plan, "plan.pairs is incomplete or reordered")
    validate_sizing_plan(plan)
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
        require(
            99 * record.get("timing_overhead_ns", -1)
            < record["guest_elapsed_ns"],
            "timing overhead",
        )
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
        if record["pair_kind"] == "single-infrastructure":
            expected_iterations = plan["iterations"][record["mode"]][
                "single-hot"
            ]
        elif record["pair_kind"] == "cancel-point-cost":
            expected_iterations = plan["iterations"]["aot"]["cancel-hot"][
                str(record["threads"])
            ]
        else:
            expected_iterations = plan["iterations"][record["mode"]][
                record["workload"]
            ][str(record["threads"])]
        require(
            record.get("iterations") == expected_iterations,
            "record plan iterations",
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
    if (
        budget["schema_version"] != REPORT_SCHEMA_VERSION
        or budget["kind"] != "wasi-thread-benchmark-budget"
    ):
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
        "- One-shot sizing: "
        f"`{document['plan']['sizing']['algorithm']['kind']}` v"
        f"{document['plan']['sizing']['algorithm']['version']}; "
        f"{len(document['plan']['sizing']['resolved']['pilots'])} retained pilots, "
        f"{len(document['plan']['sizing']['resolved']['cells'])} frozen cells, "
        f"{len(document['plan']['sizing']['algorithm']['cell_envelopes'])} "
        "declarative cell envelope; "
        f"projected benchmark "
        f"{document['plan']['sizing']['resolved']['projected_benchmark_ns'] / 1e9:.1f}s",
        f"- Budget: `{document['budget']['status']}`",
        "- Checksum preparation: "
        f"`{document['metadata']['checksum_preparation']['algorithm']}`; "
        f"worst {document['metadata']['checksum_preparation']['worst_ns']} ns, "
        f"total {document['metadata']['checksum_preparation']['total_ns']} ns",
    ]
    quality_preflight = document["quality_preflight"]
    if quality_preflight["enabled"]:
        lines.append(
            "- Scheduler/barrier preflight: "
            f"`{quality_preflight['status']}`; "
            f"{quality_preflight['probe_count']} fixed probes; maximum "
            f"{quality_preflight['summary']['maximum_barrier_ns']} ns "
            f"(limit {quality_preflight['maximum_accepted_barrier_ns']} ns)"
        )
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
    for stale_output in (
        output / "report.json",
        output / "report.md",
        output / "failure-diagnostic.json",
        output / "failure-diagnostic.md",
    ):
        stale_output.unlink(missing_ok=True)
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
    pilot_iteration_plan = args.pilot_iteration_plan
    if args.trusted_calibration_preflight and "aot" not in modes:
        raise HarnessError(
            "trusted calibration preflight requires the AOT runtime path"
        )
    pair_plan = planned_pair_specs(args, modes)
    pilot_order = pilot_order_for_plan(
        pair_plan, revision_roles, pilot_iteration_plan
    )
    runner = shlex.split(args.runner)
    preflight_acceptance_rule = (
        "every probe must have timed_interval_ns >= minimum_timed_interval_ns "
        "and 99 * timing_overhead_ns < minimum_timed_interval_ns"
    )
    host = host_metadata(args.runner_environment)
    host_pair = host_pair_identity(args.platform_id, host, args.host_pair_id)
    host_quiescence_at_start = host_quiescence_diagnostics()

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

    quality_preflight: dict[str, Any] = {
        "enabled": False,
        "status": "not-requested",
        "probe_count": 0,
        "probes_per_thread": TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD,
        "mode": "aot",
        "workload": "hot",
        "thread_counts": list(args.thread_counts),
        "minimum_timed_interval_ns": minimum_interval_ns,
        "timing_overhead_ratio_limit": TIMING_OVERHEAD_RATIO_LIMIT,
        "maximum_accepted_barrier_ns": maximum_preflight_barrier_ns(
            minimum_interval_ns
        ),
        "acceptance_rule": preflight_acceptance_rule,
        "summary": None,
        "samples": [],
        "host_quiescence_at_start": host_quiescence_at_start,
    }
    pilot_records: list[dict[str, Any]] = []
    pilot_progress_bound(
        pilot_records=pilot_records,
        total_pilots=len(pilot_order),
        warmups=args.warmups,
        samples=args.samples,
    )
    pilot_measured = functools.partial(
        measure_with_quality_diagnostic,
        output=output,
        stage="sizing-pilot",
        minimum_interval_ns=PILOT_CLOCK_RESOLUTION_MINIMUM_NS,
        host=host,
        host_pair=host_pair,
        host_quiescence_at_start=host_quiescence_at_start,
        preflight_samples=pilot_records,
    )
    for spec in pilot_order:
        context = contexts[spec["revision"]]
        mode = spec["mode"]
        sizing_workload = spec["workload"]
        guest_workload = (
            "hot" if sizing_workload == "cancel-hot" else sizing_workload
        )
        if sizing_workload == "single-hot":
            selected = context["builds"][
                (
                    "disabled-"
                    if spec["condition"] == "threads-disabled"
                    else "enabled-"
                )
                + mode
            ]
            module = (
                context["single_wasm"]
                if mode == "interpreter"
                else context["aot_artifacts"]["single"]
            )
            cancel_points = "not-applicable"
            static_cancel_poll_sites = 0
        elif sizing_workload == "cancel-hot":
            selected = context["builds"]["enabled-aot"]
            cancel_points = (
                "off"
                if spec["condition"] == "cancel-points-off"
                else "on"
            )
            module = context["aot_artifacts"][
                f"threaded-polls-{cancel_points}"
            ]
            static_cancel_poll_sites = (
                context["aot_artifacts_metadata"]["cancel_poll_static"][
                    "sites_enabled"
                ]
                if cancel_points == "on"
                else 0
            )
        else:
            selected = context["builds"][f"enabled-{mode}"]
            module = (
                context["threaded_wasm"]
                if mode == "interpreter"
                else context["aot_artifacts"]["threaded-polls-on"]
            )
            cancel_points = (
                "on" if mode == "aot" else "interpreter-dispatch"
            )
            static_cancel_poll_sites = (
                context["aot_artifacts_metadata"]["cancel_poll_static"][
                    "sites_enabled"
                ]
                if mode == "aot"
                else None
            )
        pilot = pilot_measured(
            repo=context["repo"],
            runner=runner,
            build=selected,
            module=module,
            workload=guest_workload,
            threads=spec["threads"],
            iterations=spec["iterations"],
            timeout=args.timeout,
            min_interval_ns=PILOT_CLOCK_RESOLUTION_MINIMUM_NS,
            enforce_timing_quality=False,
            record_fields={
                **spec,
                "phase": "pilot",
                "threads_enabled": selected.threads_enabled,
                "cancel_points": cancel_points,
                "static_cancel_poll_sites": static_cancel_poll_sites,
            },
        )
        pilot_records.append(pilot)
        try:
            validate_sizing_pilot(pilot, spec)
            pilot_progress_bound(
                pilot_records=pilot_records,
                total_pilots=len(pilot_order),
                warmups=args.warmups,
                samples=args.samples,
            )
        except HarnessError as exc:
            raise_with_failure_diagnostic(
                exc,
                output=output,
                stage="sizing-pilot",
                reason="pilot-quality-or-runtime-bound",
                scenario={
                    "revision": spec["revision"],
                    "mode": spec["mode"],
                    "workload": spec["workload"],
                    "threads": spec["threads"],
                    "iterations": spec["iterations"],
                    "condition": spec["condition"],
                    "pair_key": spec["pair_key"],
                    "pilot_index": spec["pilot_index"],
                },
                timing_overhead_ns=pilot["timing_overhead_ns"],
                timed_interval_ns=pilot["guest_elapsed_ns"],
                raw_elapsed_ns=pilot["raw_guest_elapsed_ns"],
                timing_overhead_ppm=pilot["timing_overhead_ppm"],
                minimum_interval_ns=PILOT_CLOCK_RESOLUTION_MINIMUM_NS,
                host=host,
                host_pair=host_pair,
                host_quiescence_at_start=host_quiescence_at_start,
                host_quiescence_at_failure=host_quiescence_diagnostics(),
                preflight_samples=pilot_records,
            )
        print(
            f"[thread-bench] sizing pilot {spec['pilot_index'] + 1}/"
            f"{len(pilot_order)} {spec['revision']}/{spec['pair_key']}/"
            f"{spec['condition']}: "
            f"guest={pilot['guest_elapsed_ns'] / 1e6:.3f} ms",
            file=sys.stderr,
        )

    try:
        iteration_plan, sizing_resolution = resolve_one_shot_sizing(
            pilot_records=pilot_records,
            pilot_order=pilot_order,
            modes=modes,
            thread_counts=args.thread_counts,
            warmups=args.warmups,
            samples=args.samples,
            timeout_seconds=args.timeout,
        )
    except HarnessError as exc:
        raise_with_failure_diagnostic(
            exc,
            output=output,
            stage="sizing-resolution",
            reason="projected-quality-or-runtime-bound",
            scenario={
                "pilot_count": len(pilot_records),
                "warmups": args.warmups,
                "samples": args.samples,
            },
            timing_overhead_ns=None,
            timed_interval_ns=None,
            raw_elapsed_ns=None,
            timing_overhead_ppm=None,
            minimum_interval_ns=minimum_interval_ns,
            host=host,
            host_pair=host_pair,
            host_quiescence_at_start=host_quiescence_at_start,
            host_quiescence_at_failure=host_quiescence_diagnostics(),
            preflight_samples=pilot_records,
        )
    plan = {
        "profile": args.profile,
        "warmups": args.warmups,
        "samples": args.samples,
        "revision_mode": revision_mode,
        "comparison_purpose": comparison_purpose,
        "revision_roles": list(revision_roles),
        "modes": list(modes),
        "thread_counts": list(args.thread_counts),
        "iterations": copy.deepcopy(iteration_plan),
        "sizing": {
            "algorithm": sizing_algorithm_spec(args.timeout),
            "pilot_iterations": copy.deepcopy(pilot_iteration_plan),
            "pilot_order": copy.deepcopy(pilot_order),
            "resolved": sizing_resolution,
        },
        "timeout_seconds": args.timeout,
        "minimum_timed_interval_ns": minimum_interval_ns,
        "atomic_wait_preflight_runs": ATOMIC_WAIT_PREFLIGHT_RUNS[args.profile],
        "scheduler_barrier_preflight": {
            "enabled": args.trusted_calibration_preflight,
            "mode": "aot",
            "workload": "hot",
            "probes_per_thread": TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD,
            "probe_count": (
                TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD
                * len(args.thread_counts)
                if args.trusted_calibration_preflight
                else 0
            ),
            "timing_overhead_ratio_limit": TIMING_OVERHEAD_RATIO_LIMIT,
            "target_barrier_ns": TARGET_BARRIER_NS,
            "target_required_interval_ns": TARGET_BARRIER_REQUIRED_INTERVAL_NS,
            "minimum_interval_headroom_ns": (
                minimum_interval_ns - TARGET_BARRIER_REQUIRED_INTERVAL_NS
            ),
            "maximum_accepted_barrier_ns": maximum_preflight_barrier_ns(
                minimum_interval_ns
            ),
            "acceptance_rule": preflight_acceptance_rule,
        },
        "optimize": args.optimize,
        "pairs": pair_plan,
    }
    plan_sha256 = cache_key(plan)
    measurement_plan_identity = measurement_plan_sha256(plan)
    checksum_preparation = prepare_expected_results(
        iteration_plan, modes, args.thread_counts
    )
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

    if args.trusted_calibration_preflight:
        context = contexts["candidate"]
        try:
            quality_preflight = run_trusted_barrier_preflight(
                repo=context["repo"],
                runner=runner,
                build=context["builds"]["enabled-aot"],
                module=context["aot_artifacts"]["threaded-polls-on"],
                thread_counts=args.thread_counts,
                iterations_by_thread=iteration_plan["aot"]["hot"],
                timeout=args.timeout,
                minimum_interval_ns=minimum_interval_ns,
                static_cancel_poll_sites=context["aot_artifacts_metadata"][
                    "cancel_poll_static"
                ]["sites_enabled"],
            )
        except PreflightProbeError as exc:
            raise_with_failure_diagnostic(
                exc,
                output=output,
                stage="scheduler-barrier-preflight",
                reason="probe-execution-failure",
                scenario={
                    "revision": "candidate",
                    **exc.scenario,
                    "condition": "cancel-points-on",
                },
                timing_overhead_ns=None,
                timed_interval_ns=None,
                raw_elapsed_ns=None,
                timing_overhead_ppm=None,
                minimum_interval_ns=minimum_interval_ns,
                host=host,
                host_pair=host_pair,
                host_quiescence_at_start=host_quiescence_at_start,
                host_quiescence_at_failure=host_quiescence_diagnostics(),
                preflight_samples=[*pilot_records, *exc.samples],
            )
        quality_preflight["host_quiescence_at_start"] = (
            host_quiescence_at_start
        )
        if quality_preflight["status"] != "passed":
            failed = next(
                sample
                for sample in quality_preflight["samples"]
                if not sample["accepted"]
            )
            message = (
                "trusted scheduler/barrier preflight failed: "
                f"{failed['timing_overhead_ns']}ns barrier cannot remain below "
                f"1% at the fixed {minimum_interval_ns}ns minimum interval"
            )
            raise_with_failure_diagnostic(
                HarnessError(message),
                output=output,
                stage="scheduler-barrier-preflight",
                reason="timing-quality",
                scenario={
                    "revision": "candidate",
                    "mode": failed["mode"],
                    "workload": failed["workload"],
                    "threads": failed["threads"],
                    "condition": "cancel-points-on",
                    "probe_index": failed["probe_index"],
                },
                timing_overhead_ns=failed["timing_overhead_ns"],
                timed_interval_ns=failed["timed_interval_ns"],
                raw_elapsed_ns=failed["raw_elapsed_ns"],
                timing_overhead_ppm=failed["timing_overhead_ppm"],
                minimum_interval_ns=minimum_interval_ns,
                host=host,
                host_pair=host_pair,
                host_quiescence_at_start=host_quiescence_at_start,
                host_quiescence_at_failure=host_quiescence_diagnostics(),
                preflight_samples=[
                    *pilot_records,
                    *quality_preflight["samples"],
                ],
                ratio_at_minimum_timed_interval=failed[
                    "ratio_at_minimum_timed_interval"
                ],
            )

    measured = functools.partial(
        measure_with_quality_diagnostic,
        output=output,
        stage="measurement",
        minimum_interval_ns=minimum_interval_ns,
        host=host,
        host_pair=host_pair,
        host_quiescence_at_start=host_quiescence_at_start,
        preflight_samples=[
            *pilot_records,
            *quality_preflight["samples"],
        ],
    )

    if "aot" in modes:
        for role in revision_roles:
            context = contexts[role]
            for index in range(ATOMIC_WAIT_PREFLIGHT_RUNS[args.profile]):
                measured(
                    stage="atomic-wait-preflight",
                    minimum_interval_ns=1,
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
            return measured(
                repo=context["repo"],
                runner=runner,
                build=selected,
                module=module,
                workload="single-hot",
                threads=1,
                iterations=iteration_plan[mode]["single-hot"],
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
                    "iterations": iteration_plan[mode]["single-hot"],
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
                return measured(
                    repo=context["repo"],
                    runner=runner,
                    build=build,
                    module=module,
                    workload=scenario.workload,
                    threads=scenario.threads,
                    iterations=iteration_count(
                        iteration_plan,
                        mode,
                        scenario.workload,
                        scenario.threads,
                    ),
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
                        "iterations": iteration_count(
                            iteration_plan,
                            mode,
                            scenario.workload,
                            scenario.threads,
                        ),
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
                return measured(
                    repo=context["repo"],
                    runner=runner,
                    build=build,
                    module=module,
                    workload=scenario.workload,
                    threads=scenario.threads,
                    iterations=iteration_count(
                        iteration_plan,
                        mode,
                        scenario.workload,
                        scenario.threads,
                    ),
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
                        "iterations": iteration_count(
                            iteration_plan,
                            mode,
                            scenario.workload,
                            scenario.threads,
                        ),
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
            iterations = iteration_count(
                iteration_plan, "aot", "cancel-hot", threads
            )

            def poll_measure(
                revision: str,
                condition: str,
                fields: dict[str, Any],
            ) -> dict[str, Any]:
                context = contexts[revision]
                aot_build = context["builds"]["enabled-aot"]
                polls = "off" if condition == "cancel-points-off" else "on"
                module = context["aot_artifacts"][f"threaded-polls-{polls}"]
                return measured(
                    repo=context["repo"],
                    runner=runner,
                    build=aot_build,
                    module=module,
                    workload="hot",
                    threads=threads,
                    iterations=iterations,
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
                        "iterations": iterations,
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
        "schema_version": REPORT_SCHEMA_VERSION,
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
            "checksum_preparation": checksum_preparation,
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
        "quality_preflight": quality_preflight,
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
