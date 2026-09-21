#!/usr/bin/env python3
"""Run and validate the non-authoritative WASI thread duration-cross diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import (
    BenchmarkDataError,
    atomic_write_json,
    cache_key,
    collected_at,
    host_metadata,
    require,
    sha256_file,
)
from bench_wasi_threads import (
    ATOMIC_WAIT_PREFLIGHT_ITERATIONS,
    ATOMIC_WAIT_PREFLIGHT_RUNS,
    DEFAULT_PILOT_ITERATION_PLAN,
    FIXTURE_SOURCE_POLICY,
    HarnessError,
    MASK64,
    PAIR_EXECUTION_POLICY,
    PILOT_CLOCK_RESOLUTION_MINIMUM_NS,
    REVISION_ARTIFACT_POLICY,
    REVISION_ROLES,
    SIZING_SIGNIFICANT_DIGITS,
    aot_artifact_report,
    build_tool_report,
    build_variant,
    compile_aot_fixtures,
    cpu_affinity_for,
    discover_cpu_placement,
    effective_sizing_cap,
    execution_arch,
    host_pair_identity,
    host_quiescence_diagnostics,
    measure_once,
    paired_invocation_order,
    position_balanced_ratio_stats,
    projected_duration_floor_for_cell,
    resolve_measurement_fixtures,
    run_trusted_barrier_preflight,
    sample_stats,
    sizing_candidates_for_cell,
    source_identity,
    validate_cpu_placement,
    validate_sizing_pilot,
)


KIND = "wasi-thread-duration-cross-diagnostic"
REPORT_SCHEMA_VERSION = 1
PLAN_KIND = "wasi-thread-duration-cross-plan"
PLAN_VERSION = 21
PROFILE = {"warmups": 2, "samples": 12, "blocks": 3, "block_size": 4}
ARMS = ("current", "doubled")
PARTITIONS = {sequence: ("training" if sequence <= 16 else "holdout") for sequence in range(1, 21)}
TRUSTED_X86_RUNNER_NAME = "vm31e-wamr-temp-20260906"
X86_CPU_CLASS = "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz"
ARM_CPU_CLASS = "Neoverse-N2"
INVOCATION_TIMEOUT_SECONDS = 180.0
MINIMUM_INTERVAL_NS = 1_250_000_000
DIAGNOSTIC_JOB_LIMIT_NS = 660 * 60 * 1_000_000_000
SIDECAR_MAX_SAMPLES = 50_000
TELEMETRY_POLICY = {
    "pre_post": (
        "bounded immediately-before/immediately-after snapshots outside guest "
        "timed intervals"
    ),
    "required": [
        "proc_stat_including_steal_and_context_switches",
        "proc_pressure_cpu_io_memory",
        "load_average",
        "cpu_frequency",
    ],
    "sidecar": (
        "optional 1Hz benchmark-CPU frequency plus temperature/package-power "
        "samples, with the collector pinned to a logical CPU outside every "
        "benchmark assignment"
    ),
    "sensor_absence": "record-unavailable-never-retry-or-exclude",
}
DESIGN_PROVENANCE = {
    "issue": 966,
    "comment_id": 5761582333,
    "original_analysis_json_sha256": (
        "8189a60fbe5d5bfa678c4c62261ea4e56a1480e99bc7d89f0857822120149ad1"
    ),
    "original_analysis_markdown_sha256": (
        "f9fc56a1625a6c6455a4c28c41b7505c43aac9e54057c9442070a37f4a60256c"
    ),
    "independent_review_json_sha256": (
        "46e6459e2b45441759aaa25a9cbb1a01d5bd37984f78b5d16dd068df930f7340"
    ),
    "independent_review_markdown_sha256": (
        "2401118a44b6a9aa64aa7508f647051cdd9b3ec8512a331483583da3d5bbadcf"
    ),
}


def _cell(
    pair_key: str,
    pair_kind: str,
    left: str,
    right: str,
    workload: str,
    threads: int,
    targets: dict[str, float],
) -> dict[str, Any]:
    return {
        "pair_key": pair_key,
        "pair_kind": pair_kind,
        "left": left,
        "right": right,
        "workload": workload,
        "threads": threads,
        "current_target_seconds": targets,
    }


X86_CELLS = (
    _cell(
        "cancel-points/hot/8",
        "cancel-point-cost",
        "cancel-points-off",
        "cancel-points-on",
        "hot",
        8,
        {"cancel-points-off": 20.0, "cancel-points-on": 20.0},
    ),
    _cell(
        "runtime/atomic/1",
        "runtime-parity",
        "interpreter",
        "aot",
        "atomic",
        1,
        {"interpreter": 5.0, "aot": 5.0},
    ),
    *(
        _cell(
            f"runtime/atomic/{threads}",
            "runtime-parity",
            "interpreter",
            "aot",
            "atomic",
            threads,
            {"interpreter": seconds, "aot": seconds},
        )
        for threads, seconds in ((2, 40.0), (4, 40.0), (8, 20.0))
    ),
    _cell(
        "runtime/hot/1",
        "runtime-parity",
        "interpreter",
        "aot",
        "hot",
        1,
        {"interpreter": 5.0, "aot": 5.0},
    ),
    _cell(
        "runtime/hot/8",
        "runtime-parity",
        "interpreter",
        "aot",
        "hot",
        8,
        {"interpreter": 20.0, "aot": 20.0},
    ),
    *(
        _cell(
            f"runtime/spawn-join/{threads}",
            "runtime-parity",
            "interpreter",
            "aot",
            "spawn-join",
            threads,
            {"interpreter": 2.5, "aot": 2.5},
        )
        for threads in (1, 2, 8)
    ),
    _cell(
        "runtime/wait-notify/1",
        "runtime-parity",
        "interpreter",
        "aot",
        "wait-notify",
        1,
        {"interpreter": 5.0, "aot": 20.0},
    ),
)
ARM_CELLS = tuple(
    _cell(
        f"runtime/atomic/{threads}",
        "runtime-parity",
        "interpreter",
        "aot",
        "atomic",
        threads,
        {"interpreter": seconds, "aot": seconds},
    )
    for threads, seconds in ((2, 40.0), (4, 40.0), (8, 20.0))
)
PLATFORM_CELLS = {
    "ubuntu-22.04-x86_64": X86_CELLS,
    "ubuntu-24.04-aarch64": ARM_CELLS,
}
ACCEPTANCE_SURFACE = {
    "ubuntu-22.04-x86_64": {
        "comparisons": (
            ("cancel-points/hot/8", "cancel-points-on", "steady-state-kernel"),
            ("runtime/atomic/2", "interpreter", "steady-state-kernel"),
            ("runtime/atomic/4", "aot", "steady-state-kernel"),
            ("runtime/atomic/4", "interpreter", "steady-state-kernel"),
            ("runtime/atomic/8", "aot", "steady-state-kernel"),
            ("runtime/atomic/8", "interpreter", "steady-state-kernel"),
            ("runtime/hot/1", "interpreter", "steady-state-kernel"),
            ("runtime/spawn-join/2", "interpreter", "spawn-join-lifecycle"),
            ("runtime/wait-notify/1", "interpreter", "steady-state-kernel"),
        ),
        "ratio_of_ratios": (
            ("cancel-points/hot/8", "cancel-points-off", "cancel-points-on"),
            ("runtime/atomic/1", "interpreter", "aot"),
            ("runtime/atomic/2", "interpreter", "aot"),
            ("runtime/atomic/4", "interpreter", "aot"),
            ("runtime/atomic/8", "interpreter", "aot"),
            ("runtime/hot/1", "interpreter", "aot"),
            ("runtime/hot/8", "interpreter", "aot"),
            ("runtime/spawn-join/1", "interpreter", "aot"),
            ("runtime/spawn-join/2", "interpreter", "aot"),
            ("runtime/spawn-join/8", "interpreter", "aot"),
        ),
    },
    "ubuntu-24.04-aarch64": {
        "comparisons": (
            ("runtime/atomic/2", "aot", "steady-state-kernel"),
            ("runtime/atomic/2", "interpreter", "steady-state-kernel"),
            ("runtime/atomic/4", "aot", "steady-state-kernel"),
            ("runtime/atomic/4", "interpreter", "steady-state-kernel"),
            ("runtime/atomic/8", "aot", "steady-state-kernel"),
            ("runtime/atomic/8", "interpreter", "steady-state-kernel"),
        ),
        "ratio_of_ratios": (
            ("runtime/atomic/2", "interpreter", "aot"),
            ("runtime/atomic/4", "interpreter", "aot"),
            ("runtime/atomic/8", "interpreter", "aot"),
        ),
    },
}


def cells_for_platform(platform_id: str) -> tuple[dict[str, Any], ...]:
    try:
        return PLATFORM_CELLS[platform_id]
    except KeyError as exc:
        raise HarnessError(
            f"duration-cross requires one of {sorted(PLATFORM_CELLS)}"
        ) from exc


def acceptance_checks_for_platform(platform_id: str) -> list[dict[str, Any]]:
    surface = ACCEPTANCE_SURFACE[platform_id]
    return [
        {"collection": collection, "key": list(key)}
        for collection in ("comparisons", "ratio_of_ratios")
        for key in surface[collection]
    ]


def arm_order(sequence: int, block_index: int) -> tuple[str, str]:
    require(sequence in PARTITIONS, "report sequence must be in 1..20")
    require(block_index in range(PROFILE["blocks"]), "block index must be in 0..2")
    return ARMS if (sequence + block_index) % 2 else tuple(reversed(ARMS))


def warmup_arm_order(sequence: int, warmup_index: int) -> tuple[str, str]:
    require(sequence in PARTITIONS, "report sequence must be in 1..20")
    require(warmup_index in range(PROFILE["warmups"]), "warmup index must be in 0..1")
    return ARMS if (sequence + warmup_index) % 2 else tuple(reversed(ARMS))


def plan_identity(plan: dict[str, Any]) -> str:
    portable = copy.deepcopy(plan)
    portable.pop("resolved_counts", None)
    portable.pop("pilots", None)
    portable.pop("runtime_admission", None)
    return cache_key(
        {
            "schema_version": PLAN_VERSION,
            "kind": PLAN_KIND,
            "portable_plan": portable,
        }
    )


def leg_spec(cell: dict[str, Any], condition: str) -> dict[str, Any]:
    if condition not in (cell["left"], cell["right"]):
        raise HarnessError(
            f"{cell['pair_key']}: unexpected condition {condition!r}"
        )
    if cell["pair_kind"] == "cancel-point-cost":
        return {
            "mode": "aot",
            "sizing_workload": "cancel-hot",
            "sizing_policy_workload": "hot",
            "guest_workload": "hot",
            "cancel_points": (
                "off" if condition == "cancel-points-off" else "on"
            ),
        }
    return {
        "mode": condition,
        "sizing_workload": cell["workload"],
        "sizing_policy_workload": cell["workload"],
        "guest_workload": cell["workload"],
        "cancel_points": "on" if condition == "aot" else "interpreter-dispatch",
    }


def pilot_iterations(spec: dict[str, Any], threads: int) -> int:
    value = DEFAULT_PILOT_ITERATION_PLAN[spec["mode"]][spec["sizing_workload"]]
    return value if isinstance(value, int) else value[str(threads)]


def build_plan(platform_id: str, sequence: int, source: dict[str, str]) -> dict[str, Any]:
    cells = copy.deepcopy(list(cells_for_platform(platform_id)))
    return {
        "kind": PLAN_KIND,
        "version": PLAN_VERSION,
        "diagnostic_only": True,
        "production_budget_eligible": False,
        "production_report_kind": "wasi-thread-benchmark",
        "production_measurement_plan_version": 20,
        "reviewed_design": copy.deepcopy(DESIGN_PROVENANCE),
        "platform_id": platform_id,
        "report_sequence": sequence,
        "partition": PARTITIONS[sequence],
        "source_revision": source,
        "artifact_policy": {
            "kind": "artifact-identical-a-a",
            "logical_revisions": list(REVISION_ROLES),
            "builds": 1,
            "required_identity": (
                "commit, tracked diff, build source, runtime binary, wasm and "
                "AOT artifact hashes must be byte-identical"
            ),
        },
        "profile": copy.deepcopy(PROFILE),
        "invocation_timeout_seconds": INVOCATION_TIMEOUT_SECONDS,
        "minimum_interval_ns": MINIMUM_INTERVAL_NS,
        "duration_arms": {
            "current": {"iteration_multiplier": 1, "selects_thresholds": False},
            "doubled": {"iteration_multiplier": 2, "selects_thresholds": True},
        },
        "arm_order_by_block": [
            {
                "block_index": block,
                "sample_indices": list(range(block * 4, block * 4 + 4)),
                "arm_order": list(arm_order(sequence, block)),
            }
            for block in range(PROFILE["blocks"])
        ],
        "warmup_arm_order": [
            {
                "warmup_index": index,
                "arm_order": list(warmup_arm_order(sequence, index)),
            }
            for index in range(PROFILE["warmups"])
        ],
        "invocations_per_duration_arm_sample": 4,
        "pair_execution": copy.deepcopy(PAIR_EXECUTION_POLICY),
        "sizing": {
            "kind": "same-production-v20-one-shot-pilots-frozen-before-evidence",
            "significant_digits": SIZING_SIGNIFICANT_DIGITS,
            "adaptation": "forbidden-after-first-measured-invocation",
            "doubled_admission": "fail-if-exactly-two-times-current-is-not-admitted",
        },
        "telemetry": copy.deepcopy(TELEMETRY_POLICY),
        "cells": cells,
        "acceptance_checks": acceptance_checks_for_platform(platform_id),
        "prohibitions": [
            "retry",
            "replacement",
            "exclusion",
            "adaptive-sleep",
            "adaptive-count",
            "adaptive-duration",
            "cell-extension",
            "report-extension",
        ],
        "resolved_counts": {},
        "pilots": [],
        "runtime_admission": {},
    }


def read_text(path: Path) -> dict[str, Any]:
    try:
        return {"available": True, "value": path.read_text(encoding="UTF-8").strip()}
    except OSError as exc:
        return {"available": False, "reason": str(exc)}


def read_proc_stat() -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "cpu": None,
        "ctxt": None,
        "processes": None,
        "procs_running": None,
        "procs_blocked": None,
    }
    try:
        lines = Path("/proc/stat").read_text(encoding="UTF-8").splitlines()
    except OSError as exc:
        result["reason"] = str(exc)
        return result
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "cpu" and len(fields) >= 9:
            names = (
                "user",
                "nice",
                "system",
                "idle",
                "iowait",
                "irq",
                "softirq",
                "steal",
                "guest",
                "guest_nice",
            )
            values = [int(value) for value in fields[1:]]
            result["cpu"] = dict(zip(names, values, strict=False))
        elif fields[0] in ("ctxt", "processes", "procs_running", "procs_blocked"):
            result[fields[0]] = int(fields[1])
    result["available"] = result["cpu"] is not None
    return result


def frequency_snapshot(cpus: set[int] | None = None) -> dict[str, Any]:
    paths = sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")
    )
    values = []
    errors = []
    for path in paths:
        try:
            cpu = int(path.parts[-3].removeprefix("cpu"))
            if cpus is not None and cpu not in cpus:
                continue
            values.append(
                {
                    "cpu": cpu,
                    "khz": int(path.read_text(encoding="UTF-8").strip()),
                }
            )
        except (OSError, ValueError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})
    return {
        "available": bool(values),
        "samples": values,
        "unavailable": errors,
        "reason": None if values else "no readable scaling_cur_freq sensors",
    }


def sensor_snapshot(pattern: str, label: str) -> dict[str, Any]:
    paths = sorted(Path("/sys").glob(pattern))
    values = []
    errors = []
    for path in paths:
        try:
            values.append(
                {"path": str(path), "value": int(path.read_text(encoding="UTF-8").strip())}
            )
        except (OSError, ValueError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})
    return {
        "available": bool(values),
        "sensor": label,
        "samples": values,
        "unavailable": errors,
        "reason": None if values else f"no readable {label} sensors",
    }


def telemetry_snapshot(frequency_cpus: set[int]) -> dict[str, Any]:
    return {
        "collected_at": collected_at(),
        "monotonic_ns": time.monotonic_ns(),
        "proc_stat": read_proc_stat(),
        "pressure": {
            name: read_text(Path(f"/proc/pressure/{name}"))
            for name in ("cpu", "io", "memory")
        },
        "loadavg": read_text(Path("/proc/loadavg")),
        "frequency": frequency_snapshot(frequency_cpus),
    }


class TelemetrySidecar:
    def __init__(self, cpu: int | None, observed_cpus: set[int]) -> None:
        self.cpu = cpu
        self.observed_cpus = set(observed_cpus)
        self.samples: list[dict[str, Any]] = []
        self.status: dict[str, Any] = {
            "requested": True,
            "available": cpu is not None,
            "cpu": cpu,
            "observed_cpus": sorted(observed_cpus),
            "interval_seconds": 1,
            "reason": None if cpu is not None else "no CPU outside benchmark assignments",
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.cpu is None:
            return
        self._thread = threading.Thread(target=self._run, name="duration-cross-telemetry")
        self._thread.start()

    def _run(self) -> None:
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {self.cpu})
        except OSError as exc:
            self.status.update({"available": False, "reason": f"affinity failed: {exc}"})
            return
        try:
            self._collect()
        except Exception as exc:
            self.status.update(
                {"available": False, "reason": f"collector failed: {exc}"}
            )

    def _collect(self) -> None:
        previous_energy: dict[str, int] | None = None
        previous_ns: int | None = None
        while not self._stop.is_set():
            if len(self.samples) >= SIDECAR_MAX_SAMPLES:
                self.status.update(
                    {
                        "available": False,
                        "reason": (
                            f"sidecar exceeded its fixed {SIDECAR_MAX_SAMPLES}-sample "
                            "bound"
                        ),
                    }
                )
                return
            sampled_ns = time.monotonic_ns()
            energy = sensor_snapshot(
                "class/powercap/**/energy_uj", "package-power-energy"
            )
            current_energy = {
                item["path"]: item["value"] for item in energy["samples"]
            }
            watts = []
            if previous_energy is not None and previous_ns is not None:
                interval_seconds = (sampled_ns - previous_ns) / 1_000_000_000
                for path, value in current_energy.items():
                    prior = previous_energy.get(path)
                    if prior is not None and value >= prior and interval_seconds > 0:
                        watts.append(
                            {
                                "path": path,
                                "watts": (value - prior)
                                / 1_000_000
                                / interval_seconds,
                            }
                        )
            self.samples.append(
                {
                    "collected_at": collected_at(),
                    "monotonic_ns": sampled_ns,
                    "frequency": frequency_snapshot(self.observed_cpus),
                    "temperature": sensor_snapshot(
                        "class/thermal/thermal_zone*/temp", "temperature"
                    ),
                    "package_power": {
                        "available": bool(watts),
                        "watts": watts,
                        "energy_counters": energy,
                        "reason": (
                            None
                            if watts
                            else "package energy unavailable or no prior 1Hz sample"
                        ),
                    },
                }
            )
            previous_energy = current_energy
            previous_ns = sampled_ns
            self._stop.wait(1.0)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise HarnessError("telemetry sidecar did not stop within 5 seconds")
        if (
            self.cpu is not None
            and not self.status["available"]
            and str(self.status.get("reason", "")).startswith("sidecar exceeded")
        ):
            raise HarnessError(self.status["reason"])
        return {**self.status, "samples": self.samples}


def benchmark_cpus(
    cpu_placement: dict[str, Any], cells: tuple[dict[str, Any], ...]
) -> set[int]:
    result: set[int] = set()
    for cell in cells:
        result.update(
            cpu_affinity_for(cpu_placement, cell["workload"], cell["threads"])
        )
    return result


def telemetry_cpu(cpu_placement: dict[str, Any], cells: tuple[dict[str, Any], ...]) -> int | None:
    assigned = benchmark_cpus(cpu_placement, cells)
    return next(
        (
            cpu
            for cpu in reversed(cpu_placement["ordered_logical_cpus"])
            if cpu not in assigned
        ),
        None,
    )


def module_for(
    context: dict[str, Any], cell: dict[str, Any], condition: str
) -> tuple[Any, Path, tuple[str, ...], dict[str, Any]]:
    spec = leg_spec(cell, condition)
    mode = spec["mode"]
    build = context["builds"][f"enabled-{mode}"]
    if cell["pair_kind"] == "cancel-point-cost":
        polls = spec["cancel_points"]
        module = context["aot_artifacts"][f"threaded-polls-{polls}"]
        static_sites = (
            context["aot_artifacts_metadata"]["cancel_poll_static"]["sites_enabled"]
            if polls == "on"
            else 0
        )
    else:
        module = (
            context["threaded_wasm"]
            if mode == "interpreter"
            else context["aot_artifacts"]["threaded-polls-on"]
        )
        static_sites = (
            context["aot_artifacts_metadata"]["cancel_poll_static"]["sites_enabled"]
            if mode == "aot"
            else None
        )
    return (
        build,
        module,
        (),
        {
            "mode": mode,
            "threads_enabled": True,
            "thread_manager_enabled": True,
            "cancel_points": spec["cancel_points"],
            "static_cancel_poll_sites": static_sites,
            "workload": spec["guest_workload"],
            "threads": cell["threads"],
        },
    )


def measured_with_telemetry(
    *,
    context: dict[str, Any],
    cpu_placement: dict[str, Any],
    runner: list[str],
    cell: dict[str, Any],
    condition: str,
    iterations: int,
    timeout: float,
    minimum_interval_ns: int,
    fields: dict[str, Any],
    enforce_timing_quality: bool = True,
) -> dict[str, Any]:
    build, module, runtime_args, leg_fields = module_for(context, cell, condition)
    frequency_cpus = set(
        cpu_affinity_for(cpu_placement, cell["workload"], cell["threads"])
    )
    before = telemetry_snapshot(frequency_cpus)
    record = measure_once(
        repo=context["repo"],
        runner=runner,
        cpu_placement=cpu_placement,
        build=build,
        module=module,
        workload=leg_fields["workload"],
        threads=cell["threads"],
        iterations=iterations,
        timeout=timeout,
        min_interval_ns=minimum_interval_ns,
        enforce_timing_quality=enforce_timing_quality,
        runtime_args=runtime_args,
        record_fields={**leg_fields, **fields, "iterations": iterations},
    )
    after = telemetry_snapshot(frequency_cpus)
    record["telemetry"] = {
        "collection": TELEMETRY_POLICY["pre_post"],
        "before": before,
        "after": after,
    }
    return record


def resolve_counts(
    plan: dict[str, Any],
    pilots: list[dict[str, Any]],
    timeout_seconds: float,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    expected_specs = []
    for cell in plan["cells"]:
        for condition in (cell["left"], cell["right"]):
            spec = leg_spec(cell, condition)
            expected_specs.append(
                {
                    "pilot_index": len(expected_specs),
                    "revision": "baseline",
                    "pair_kind": cell["pair_kind"],
                    "pair_key": cell["pair_key"],
                    "condition": condition,
                    "mode": spec["mode"],
                    "workload": spec["sizing_workload"],
                    "sizing_policy_workload": spec["sizing_policy_workload"],
                    "threads": cell["threads"],
                    "iterations": pilot_iterations(spec, cell["threads"]),
                }
            )
    if len(pilots) != len(expected_specs):
        raise HarnessError("duration-cross sizing pilot set is incomplete")
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for expected, pilot in zip(expected_specs, pilots, strict=True):
        validate_sizing_pilot(pilot, expected)
        if (
            pilot.get("sizing_policy_workload")
            != expected["sizing_policy_workload"]
        ):
            raise HarnessError("sizing pilot policy workload mismatch")
        grouped[
            (
                expected["mode"],
                expected["workload"],
                expected["sizing_policy_workload"],
                expected["threads"],
            )
        ].append(pilot)

    cell_counts: dict[str, dict[str, int]] = {}
    sizing_cells: list[dict[str, Any]] = []
    selected_by_sizing_key: dict[tuple[str, str, str, int], int] = {}
    for key, values in sorted(grouped.items()):
        fastest = min(values, key=lambda item: item["guest_elapsed_ns"])
        candidates = sizing_candidates_for_cell(
            key[0],
            key[2],
            key[3],
            fastest["iterations"],
            fastest["guest_elapsed_ns"],
        )
        current = candidates["selected_iterations"]
        doubled = current * 2
        cap = effective_sizing_cap(key[1], key[3])
        if doubled > cap or doubled > MASK64 // key[3]:
            raise HarnessError(
                f"duration-cross {key} doubled count {doubled} cannot be admitted "
                f"under frozen cap {cap}; counts must not be shortened"
            )
        selected_by_sizing_key[key] = current
        sizing_cells.append(
            {
                "key": f"{key[0]}/{key[1]}/{key[3]}",
                "mode": key[0],
                "workload": key[1],
                "sizing_policy_workload": key[2],
                "threads": key[3],
                "pilot_observations": len(values),
                "fastest_pilot_index": fastest["pilot_index"],
                "fastest_elapsed_ns": fastest["guest_elapsed_ns"],
                "current_iterations": current,
                "doubled_iterations": doubled,
                "effective_iteration_cap": cap,
                "selected_sources": candidates["selected_sources"],
            }
        )

    total_projected_host_ns = sum(pilot["host_wall_elapsed_ns"] for pilot in pilots)
    projections = []
    timeout_ns = int(timeout_seconds * 1_000_000_000)
    for pilot in pilots:
        key = (
            pilot["mode"],
            pilot["workload"],
            pilot["sizing_policy_workload"],
            pilot["threads"],
        )
        current = selected_by_sizing_key[key]
        for arm, multiplier in (("current", 1), ("doubled", 2)):
            selected = current * multiplier
            guest = math.ceil(
                pilot["guest_elapsed_ns"] * selected / pilot["iterations"]
            )
            host = math.ceil(
                pilot["host_wall_elapsed_ns"] * selected / pilot["iterations"]
            )
            if (
                guest
                < projected_duration_floor_for_cell(key[0], key[2], key[3])
                * multiplier
                or 99 * pilot["timing_overhead_ns"] >= guest
                or guest >= timeout_ns
                or host >= timeout_ns
            ):
                raise HarnessError(
                    f"duration-cross {arm} arm cannot be admitted for pilot "
                    f"{pilot['pilot_index']}; duration/count adaptation is forbidden"
                )
            total_projected_host_ns += (
                host
                * (PROFILE["warmups"] + PROFILE["samples"])
                * len(REVISION_ROLES)
            )
            projections.append(
                {
                    "pilot_index": pilot["pilot_index"],
                    "arm": arm,
                    "iterations": selected,
                    "projected_guest_elapsed_ns": guest,
                    "projected_host_wall_elapsed_ns": host,
                }
            )
    if total_projected_host_ns >= DIAGNOSTIC_JOB_LIMIT_NS:
        raise HarnessError(
            "duration-cross frozen projection exceeds the 660-minute benchmark "
            "bound; duration/count/cell adaptation is forbidden"
        )

    for cell in plan["cells"]:
        counts = {}
        for condition in (cell["left"], cell["right"]):
            spec = leg_spec(cell, condition)
            expected_target = (
                projected_duration_floor_for_cell(
                    spec["mode"],
                    spec["sizing_policy_workload"],
                    cell["threads"],
                )
                / 1_000_000_000
            )
            if cell["current_target_seconds"][condition] != expected_target:
                raise HarnessError(
                    f"{cell['pair_key']}/{condition} current target "
                    f"{cell['current_target_seconds'][condition]} does not match "
                    f"the frozen sizing target {expected_target}"
                )
            key = (
                spec["mode"],
                spec["sizing_workload"],
                spec["sizing_policy_workload"],
                cell["threads"],
            )
            counts[condition] = selected_by_sizing_key[key]
        cell_counts[cell["pair_key"]] = counts
    return cell_counts, {
        "sizing_cells": sizing_cells,
        "projections": projections,
        "projected_host_wall_ns": total_projected_host_ns,
        "limit_ns": DIAGNOSTIC_JOB_LIMIT_NS,
    }


def summarize_report(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    ratios: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []
    measured = [record for record in records if record["phase"] == "measure"]
    pair_keys = sorted({record["pair_key"] for record in measured})
    for arm in ARMS:
        for pair_key in pair_keys:
            selected = [
                record
                for record in measured
                if record["arm"] == arm and record["pair_key"] == pair_key
            ]
            first = selected[0]
            left, right = first["pair_left"], first["pair_right"]
            by_sample: dict[int, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            for record in selected:
                by_sample[record["sample_index"]][record["condition"]][
                    record["revision"]
                ] = record
            comparison_values: dict[str, dict[str, list[float]]] = {
                left: {"elapsed": [], "throughput": []},
                right: {"elapsed": [], "throughput": []},
            }
            ratio_values = {"elapsed": [], "throughput": []}
            internal_values: dict[str, dict[str, list[float]]] = {
                revision: {"elapsed": [], "throughput": []}
                for revision in REVISION_ROLES
            }
            for sample_index in range(PROFILE["samples"]):
                cell = by_sample.get(sample_index)
                if cell is None or set(cell) != {left, right} or any(
                    set(cell[condition]) != set(REVISION_ROLES)
                    for condition in (left, right)
                ):
                    raise HarnessError(
                        f"incomplete measured duration-cross sample "
                        f"{arm}/{pair_key}/{sample_index}"
                    )
                for condition in (left, right):
                    baseline = cell[condition]["baseline"]
                    candidate = cell[condition]["candidate"]
                    comparison_values[condition]["elapsed"].append(
                        candidate["elapsed_ns"] / baseline["elapsed_ns"]
                    )
                    comparison_values[condition]["throughput"].append(
                        candidate["throughput_ops_per_second"]
                        / baseline["throughput_ops_per_second"]
                    )
                baseline_elapsed = (
                    cell[right]["baseline"]["elapsed_ns"]
                    / cell[left]["baseline"]["elapsed_ns"]
                )
                candidate_elapsed = (
                    cell[right]["candidate"]["elapsed_ns"]
                    / cell[left]["candidate"]["elapsed_ns"]
                )
                baseline_throughput = (
                    cell[right]["baseline"]["throughput_ops_per_second"]
                    / cell[left]["baseline"]["throughput_ops_per_second"]
                )
                candidate_throughput = (
                    cell[right]["candidate"]["throughput_ops_per_second"]
                    / cell[left]["candidate"]["throughput_ops_per_second"]
                )
                ratio_values["elapsed"].append(candidate_elapsed / baseline_elapsed)
                ratio_values["throughput"].append(
                    candidate_throughput / baseline_throughput
                )
                for revision in REVISION_ROLES:
                    internal_values[revision]["elapsed"].append(
                        cell[right][revision]["elapsed_ns"]
                        / cell[left][revision]["elapsed_ns"]
                    )
                    internal_values[revision]["throughput"].append(
                        cell[right][revision]["throughput_ops_per_second"]
                        / cell[left][revision]["throughput_ops_per_second"]
                    )
            for condition in (left, right):
                comparisons.append(
                    {
                        "arm": arm,
                        "pair_kind": first["pair_kind"],
                        "pair_key": pair_key,
                        "condition": condition,
                        "metric_kind": cell[condition]["baseline"]["metric_kind"],
                        "elapsed_candidate_over_baseline": (
                            position_balanced_ratio_stats(
                                comparison_values[condition]["elapsed"]
                            )
                        ),
                        "throughput_candidate_over_baseline": (
                            position_balanced_ratio_stats(
                                comparison_values[condition]["throughput"]
                            )
                        ),
                    }
                )
            ratios.append(
                {
                    "arm": arm,
                    "pair_kind": first["pair_kind"],
                    "pair_key": pair_key,
                    "left": left,
                    "right": right,
                    "elapsed_ratio_of_ratios": position_balanced_ratio_stats(
                        ratio_values["elapsed"]
                    ),
                    "throughput_ratio_of_ratios": position_balanced_ratio_stats(
                        ratio_values["throughput"]
                    ),
                }
            )
            for revision in REVISION_ROLES:
                internal.append(
                    {
                        "arm": arm,
                        "pair_kind": first["pair_kind"],
                        "pair_key": pair_key,
                        "revision": revision,
                        "left": left,
                        "right": right,
                        "elapsed_right_over_left": sample_stats(
                            internal_values[revision]["elapsed"], "samples"
                        ),
                        "throughput_right_over_left": sample_stats(
                            internal_values[revision]["throughput"], "samples"
                        ),
                    }
                )
    return {
        "comparisons": comparisons,
        "ratio_of_ratios": ratios,
        "internal_pairs": internal,
    }


def _required_hash(value: Any, label: str) -> None:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        label,
    )


def _validate_availability(
    value: Any, label: str, *, require_samples: bool = False
) -> dict[str, Any]:
    require(isinstance(value, dict), label)
    available = value.get("available")
    require(isinstance(available, bool), f"{label}.available")
    if available and require_samples:
        require(
            isinstance(value.get("samples"), list) and bool(value["samples"]),
            f"{label}.samples",
        )
    if not available:
        require(
            isinstance(value.get("reason"), str) and bool(value["reason"]),
            f"{label}.reason",
        )
    return value


def _validate_telemetry_snapshot(
    value: Any, label: str, expected_frequency_cpus: set[int]
) -> None:
    require(isinstance(value, dict), label)
    require(
        isinstance(value.get("collected_at"), str)
        and isinstance(value.get("monotonic_ns"), int)
        and not isinstance(value["monotonic_ns"], bool)
        and value["monotonic_ns"] > 0,
        f"{label} timestamps",
    )
    proc = _validate_availability(value.get("proc_stat"), f"{label}.proc_stat")
    if proc["available"]:
        cpu = proc.get("cpu")
        require(
            isinstance(cpu, dict)
            and all(
                isinstance(cpu.get(name), int)
                and not isinstance(cpu[name], bool)
                and cpu[name] >= 0
                for name in (
                    "user",
                    "nice",
                    "system",
                    "idle",
                    "iowait",
                    "irq",
                    "softirq",
                    "steal",
                )
            )
            and isinstance(proc.get("ctxt"), int)
            and not isinstance(proc["ctxt"], bool)
            and proc["ctxt"] >= 0,
            f"{label}.proc_stat counters",
        )
    pressure = value.get("pressure")
    require(
        isinstance(pressure, dict)
        and set(pressure) == {"cpu", "io", "memory"},
        f"{label}.pressure",
    )
    for name in ("cpu", "io", "memory"):
        status = _validate_availability(
            pressure[name], f"{label}.pressure.{name}"
        )
        if status["available"]:
            require(
                isinstance(status.get("value"), str) and bool(status["value"]),
                f"{label}.pressure.{name}.value",
            )
    loadavg = _validate_availability(value.get("loadavg"), f"{label}.loadavg")
    if loadavg["available"]:
        require(
            isinstance(loadavg.get("value"), str) and bool(loadavg["value"]),
            f"{label}.loadavg.value",
        )
    frequency = _validate_availability(
        value.get("frequency"), f"{label}.frequency", require_samples=True
    )
    require(
        isinstance(frequency.get("samples"), list)
        and isinstance(frequency.get("unavailable"), list),
        f"{label}.frequency samples",
    )
    for sample in frequency["samples"]:
        require(
            isinstance(sample, dict)
            and isinstance(sample.get("cpu"), int)
            and not isinstance(sample["cpu"], bool)
            and sample["cpu"] >= 0
            and isinstance(sample.get("khz"), int)
            and not isinstance(sample["khz"], bool)
            and sample["khz"] > 0,
            f"{label}.frequency sample",
        )
    require(
        {
            sample["cpu"] for sample in frequency["samples"]
        }
        <= expected_frequency_cpus,
        f"{label}.frequency CPU scope",
    )


def _validate_record_telemetry(
    value: Any, label: str, expected_frequency_cpus: set[int]
) -> None:
    require(
        isinstance(value, dict)
        and value.get("collection") == TELEMETRY_POLICY["pre_post"],
        f"{label}.collection",
    )
    _validate_telemetry_snapshot(
        value.get("before"), f"{label}.before", expected_frequency_cpus
    )
    _validate_telemetry_snapshot(
        value.get("after"), f"{label}.after", expected_frequency_cpus
    )
    require(
        value["after"]["monotonic_ns"] >= value["before"]["monotonic_ns"],
        f"{label} monotonic ordering",
    )


def _validate_sidecar(value: Any, expected_observed_cpus: set[int]) -> None:
    require(
        isinstance(value, dict)
        and value.get("requested") is True
        and value.get("interval_seconds") == 1
        and isinstance(value.get("available"), bool)
        and value.get("observed_cpus") == sorted(expected_observed_cpus)
        and isinstance(value.get("samples"), list)
        and len(value["samples"]) <= SIDECAR_MAX_SAMPLES,
        "telemetry sidecar identity",
    )
    if value["available"]:
        require(
            isinstance(value.get("cpu"), int)
            and not isinstance(value["cpu"], bool)
            and value["cpu"] >= 0
            and bool(value["samples"]),
            "telemetry sidecar available state",
        )
    else:
        require(
            isinstance(value.get("reason"), str) and bool(value["reason"]),
            "telemetry sidecar unavailable reason",
        )
    for index, sample in enumerate(value["samples"]):
        require(
            isinstance(sample, dict)
            and isinstance(sample.get("collected_at"), str)
            and isinstance(sample.get("monotonic_ns"), int)
            and not isinstance(sample["monotonic_ns"], bool)
            and sample["monotonic_ns"] > 0,
            f"telemetry sidecar sample {index} timestamps",
        )
        frequency = _validate_availability(
            sample.get("frequency"),
            f"telemetry sidecar sample {index}.frequency",
            require_samples=True,
        )
        require(
            isinstance(frequency.get("samples"), list),
            f"telemetry sidecar sample {index}.frequency samples",
        )
        require(
            {
                item["cpu"]
                for item in frequency["samples"]
                if isinstance(item, dict) and isinstance(item.get("cpu"), int)
            }
            <= expected_observed_cpus,
            f"telemetry sidecar sample {index}.frequency CPU scope",
        )
        temperature = _validate_availability(
            sample.get("temperature"),
            f"telemetry sidecar sample {index}.temperature",
            require_samples=True,
        )
        require(
            isinstance(temperature.get("samples"), list),
            f"telemetry sidecar sample {index}.temperature samples",
        )
        power = _validate_availability(
            sample.get("package_power"),
            f"telemetry sidecar sample {index}.package_power",
        )
        require(
            isinstance(power.get("watts"), list)
            and isinstance(power.get("energy_counters"), dict),
            f"telemetry sidecar sample {index}.package_power samples",
        )
        if power["available"]:
            require(
                bool(power["watts"]),
                f"telemetry sidecar sample {index}.package_power watts",
            )


def validate_report(document: dict[str, Any]) -> None:
    require(isinstance(document, dict), "report must be an object")
    require(document.get("schema_version") == REPORT_SCHEMA_VERSION, "schema_version")
    require(document.get("kind") == KIND, "report kind")
    require(document.get("authoritative") is False, "authoritative must be false")
    require(document.get("production_budget") is None, "production budget must be null")
    metadata = document.get("metadata")
    plan = document.get("plan")
    records = document.get("records")
    summaries = document.get("summaries")
    require(isinstance(metadata, dict), "metadata")
    require(isinstance(plan, dict), "plan")
    require(isinstance(records, list), "records")
    require(isinstance(summaries, dict), "summaries")
    platform_id = metadata.get("platform_id")
    sequence = metadata.get("report_sequence")
    require(platform_id in PLATFORM_CELLS, "metadata.platform_id")
    require(sequence in PARTITIONS, "metadata.report_sequence")
    require(metadata.get("partition") == PARTITIONS[sequence], "metadata.partition")
    require(
        isinstance(metadata.get("cohort_id"), str) and metadata["cohort_id"],
        "metadata.cohort_id",
    )
    require(
        plan.get("kind") == PLAN_KIND
        and plan.get("version") == PLAN_VERSION
        and plan.get("diagnostic_only") is True
        and plan.get("production_budget_eligible") is False,
        "diagnostic plan identity",
    )
    require(plan.get("platform_id") == platform_id, "plan.platform_id")
    require(plan.get("report_sequence") == sequence, "plan.report_sequence")
    require(plan.get("partition") == PARTITIONS[sequence], "plan.partition")
    require(plan.get("profile") == PROFILE, "plan.profile")
    require(
        plan.get("invocation_timeout_seconds") == INVOCATION_TIMEOUT_SECONDS
        and plan.get("minimum_interval_ns") == MINIMUM_INTERVAL_NS,
        "plan timing bounds",
    )
    require(plan.get("cells") == list(cells_for_platform(platform_id)), "plan.cells")
    require(
        plan.get("acceptance_checks")
        == acceptance_checks_for_platform(platform_id),
        "plan acceptance checks",
    )
    require(
        plan.get("duration_arms", {}).get("current", {}).get("selects_thresholds")
        is False
        and plan.get("duration_arms", {}).get("doubled", {}).get(
            "selects_thresholds"
        )
        is True,
        "plan.duration_arms selection",
    )
    expected_blocks = [
        {
            "block_index": block,
            "sample_indices": list(range(block * 4, block * 4 + 4)),
            "arm_order": list(arm_order(sequence, block)),
        }
        for block in range(3)
    ]
    require(plan.get("arm_order_by_block") == expected_blocks, "plan arm ordering")
    require(
        plan.get("warmup_arm_order")
        == [
            {
                "warmup_index": index,
                "arm_order": list(warmup_arm_order(sequence, index)),
            }
            for index in range(PROFILE["warmups"])
        ],
        "plan warmup arm ordering",
    )
    _required_hash(metadata.get("plan_sha256"), "metadata.plan_sha256")
    _required_hash(metadata.get("plan_identity_sha256"), "metadata.plan_identity_sha256")
    require(metadata["plan_sha256"] == cache_key(plan), "metadata plan hash")
    require(metadata["plan_identity_sha256"] == plan_identity(plan), "plan identity hash")
    source = metadata.get("source_revision")
    revisions = metadata.get("logical_revisions")
    require(
        isinstance(source, dict)
        and set(source) == {"commit", "tracked_diff_sha256", "build_source_sha256"}
        and isinstance(revisions, dict)
        and set(revisions) == set(REVISION_ROLES),
        "artifact A/A identities",
    )
    require(
        isinstance(source["commit"], str)
        and re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is not None,
        "source commit",
    )
    _required_hash(source["tracked_diff_sha256"], "source tracked diff hash")
    _required_hash(source["build_source_sha256"], "source build hash")
    _required_hash(metadata.get("fixture_set_sha256"), "fixture set hash")
    require(
        metadata.get("fixture_source_policy") == FIXTURE_SOURCE_POLICY
        and metadata.get("revision_artifact_policy")
        == REVISION_ARTIFACT_POLICY,
        "fixture and revision artifact policies",
    )
    expected_plan = build_plan(platform_id, sequence, source)
    static_plan = copy.deepcopy(plan)
    for field in ("resolved_counts", "pilots", "runtime_admission"):
        static_plan.pop(field, None)
        expected_plan.pop(field, None)
    require(static_plan == expected_plan, "predeclared diagnostic plan")
    for revision in REVISION_ROLES:
        require(revisions[revision] == source, f"{revision} source identity")
    artifacts = metadata.get("artifact_identity")
    require(
        isinstance(artifacts, dict)
        and artifacts.get("baseline") == artifacts.get("candidate")
        and isinstance(artifacts.get("baseline"), dict)
        and bool(artifacts["baseline"]),
        "artifact-identical A/A hashes",
    )
    expected_artifact_names = {
        "runtime/enabled-aot",
        "compiler/enabled-aot",
        "runtime/enabled-interpreter",
        "aot/single",
        "aot/threaded-polls-off",
        "aot/threaded-polls-on",
        "fixture/single",
        "fixture/threaded",
    }
    require(
        set(artifacts["baseline"]) == expected_artifact_names,
        "complete artifact identity",
    )
    for label, digest in artifacts["baseline"].items():
        _required_hash(digest, f"artifact hash {label}")

    resolved = plan.get("resolved_counts")
    pilots = plan.get("pilots")
    admission = plan.get("runtime_admission")
    require(isinstance(resolved, dict), "plan.resolved_counts")
    require(isinstance(pilots, list) and pilots, "plan.pilots")
    require(
        isinstance(admission, dict)
        and admission.get("projected_host_wall_ns", DIAGNOSTIC_JOB_LIMIT_NS)
        < admission.get("limit_ns", 0)
        == DIAGNOSTIC_JOB_LIMIT_NS,
        "plan.runtime_admission",
    )
    expected_pilot_count = len(cells_for_platform(platform_id)) * 2
    require(len(pilots) == expected_pilot_count, "plan pilot count")
    recomputed_counts, recomputed_admission = resolve_counts(
        plan, pilots, INVOCATION_TIMEOUT_SECONDS
    )
    require(resolved == recomputed_counts, "pilot-derived resolved counts")
    require(admission == recomputed_admission, "pilot-derived runtime admission")
    thread_counts = tuple(
        sorted({cell["threads"] for cell in cells_for_platform(platform_id)})
    )
    cpu_placement = metadata.get("cpu_placement")
    validate_cpu_placement(cpu_placement, thread_counts)
    for index, pilot in enumerate(pilots):
        _validate_record_telemetry(
            pilot.get("telemetry"),
            f"plan.pilots[{index}].telemetry",
            set(
                cpu_affinity_for(
                    cpu_placement,
                    (
                        "hot"
                        if pilot["workload"] == "cancel-hot"
                        else pilot["workload"]
                    ),
                    pilot["threads"],
                )
            ),
        )
    host = metadata.get("host")
    require(isinstance(host, dict), "metadata.host")
    if platform_id == "ubuntu-22.04-x86_64":
        require(
            host.get("runner_name") == TRUSTED_X86_RUNNER_NAME
            and host.get("cpu") == X86_CPU_CLASS,
            "trusted x86 runner identity",
        )
    else:
        require(
            isinstance(host.get("cpu"), str) and ARM_CPU_CLASS in host["cpu"],
            "trusted Arm CPU class",
        )
    require(
        metadata.get("host_quiescence_at_start", {}).get(
            "runner_worker_process_count"
        )
        == 1,
        "single runner worker",
    )
    require(
        metadata.get("workflow_run_attempt") in ("", "1"),
        "workflow reruns are forbidden",
    )
    quality_preflight = document.get("quality_preflight")
    require(
        isinstance(quality_preflight, dict)
        and quality_preflight.get("status") == "passed",
        "quality preflight",
    )
    expected_coordinates = set()
    expected_record_order = []
    for cell in cells_for_platform(platform_id):
        counts = resolved.get(cell["pair_key"])
        require(
            isinstance(counts, dict)
            and set(counts) == {cell["left"], cell["right"]},
            f"resolved counts {cell['pair_key']}",
        )
        for condition, count in counts.items():
            spec = leg_spec(cell, condition)
            cap = effective_sizing_cap(
                spec["sizing_workload"], cell["threads"]
            )
            require(
                isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
                and count * 2 <= cap,
                f"resolved count admission {cell['pair_key']}/{condition}",
            )
        for phase, amount in (
            ("warmup", PROFILE["warmups"]),
            ("measure", PROFILE["samples"]),
        ):
            for phase_index in range(amount):
                for arm in ARMS:
                    for revision in REVISION_ROLES:
                        for condition in (cell["left"], cell["right"]):
                            expected_coordinates.add(
                                (
                                    cell["pair_key"],
                                    phase,
                                    phase_index,
                                    arm,
                                    revision,
                                    condition,
                                )
                            )
        for warmup_index in range(PROFILE["warmups"]):
            for arm in warmup_arm_order(sequence, warmup_index):
                for revision, condition in paired_invocation_order(
                    warmup_index,
                    cell["pair_kind"],
                    cell["left"],
                    cell["right"],
                    REVISION_ROLES,
                ):
                    expected_record_order.append(
                        (
                            cell["pair_key"],
                            "warmup",
                            warmup_index,
                            arm,
                            revision,
                            condition,
                        )
                    )
        for block_index in range(PROFILE["blocks"]):
            for sample_in_block in range(PROFILE["block_size"]):
                sample_index = block_index * PROFILE["block_size"] + sample_in_block
                for arm in arm_order(sequence, block_index):
                    for revision, condition in paired_invocation_order(
                        sample_in_block,
                        cell["pair_kind"],
                        cell["left"],
                        cell["right"],
                        REVISION_ROLES,
                    ):
                        expected_record_order.append(
                            (
                                cell["pair_key"],
                                "measure",
                                sample_index,
                                arm,
                                revision,
                                condition,
                            )
                        )
    observed_coordinates = set()
    for index, record in enumerate(records):
        coordinate = (
            record.get("pair_key"),
            record.get("phase"),
            record.get("phase_index"),
            record.get("arm"),
            record.get("revision"),
            record.get("condition"),
        )
        require(coordinate not in observed_coordinates, f"duplicate record {coordinate}")
        observed_coordinates.add(coordinate)
        pair_key, phase, phase_index, arm, revision, condition = coordinate
        cell = next(
            (item for item in cells_for_platform(platform_id) if item["pair_key"] == pair_key),
            None,
        )
        require(cell is not None, f"record {index} unexpected cell")
        require(phase in ("warmup", "measure"), f"record {index} phase")
        limit = PROFILE["warmups"] if phase == "warmup" else PROFILE["samples"]
        require(
            isinstance(phase_index, int) and 0 <= phase_index < limit,
            f"record {index} phase_index",
        )
        require(arm in ARMS and revision in REVISION_ROLES, f"record {index} identity")
        require(condition in (cell["left"], cell["right"]), f"record {index} condition")
        spec = leg_spec(cell, condition)
        require(
            record.get("pair_kind") == cell["pair_kind"]
            and record.get("pair_left") == cell["left"]
            and record.get("pair_right") == cell["right"]
            and record.get("mode") == spec["mode"]
            and record.get("workload") == spec["guest_workload"]
            and record.get("threads") == cell["threads"]
            and record.get("cancel_points") == spec["cancel_points"]
            and record.get("pair_execution") == PAIR_EXECUTION_POLICY["default"],
            f"record {index} cell identity",
        )
        require(
            record.get("sample_index")
            == (phase_index if phase == "measure" else None)
            and record.get("block_index")
            == (phase_index // PROFILE["block_size"] if phase == "measure" else None)
            and record.get("sample_in_block")
            == (phase_index % PROFILE["block_size"] if phase == "measure" else None),
            f"record {index} block coordinates",
        )
        expected_iterations = resolved[pair_key][condition] * (
            2 if arm == "doubled" else 1
        )
        require(record.get("iterations") == expected_iterations, f"record {index} count")
        require(record.get("correct") is True, f"record {index} correctness")
        elapsed_ns = record.get("elapsed_ns")
        overhead_ns = record.get("timing_overhead_ns")
        raw_elapsed_ns = record.get("raw_guest_elapsed_ns")
        operations = record.get("operations")
        throughput = record.get("throughput_ops_per_second")
        require(
            isinstance(elapsed_ns, int)
            and not isinstance(elapsed_ns, bool)
            and elapsed_ns >= MINIMUM_INTERVAL_NS
            and record.get("guest_elapsed_ns") == elapsed_ns
            and isinstance(overhead_ns, int)
            and not isinstance(overhead_ns, bool)
            and overhead_ns >= 0
            and raw_elapsed_ns == elapsed_ns + overhead_ns
            and 99 * overhead_ns < elapsed_ns
            and isinstance(operations, int)
            and not isinstance(operations, bool)
            and operations > 0
            and isinstance(throughput, (int, float))
            and not isinstance(throughput, bool)
            and math.isfinite(throughput)
            and math.isclose(
                throughput,
                operations / (elapsed_ns / 1_000_000_000),
                rel_tol=1e-12,
            ),
            f"record {index} timing quality",
        )
        require(
            isinstance(record.get("host_wall_elapsed_ns"), int)
            and record["host_wall_elapsed_ns"] > 0
            and isinstance(record.get("host_started_ns"), int)
            and isinstance(record.get("host_finished_ns"), int)
            and record["host_finished_ns"] >= record["host_started_ns"]
            and record["host_finished_ns"] - record["host_started_ns"]
            == record["host_wall_elapsed_ns"],
            f"record {index} host timing",
        )
        correctness = record.get("correctness")
        require(
            isinstance(correctness, dict)
            and correctness.get("passed") is True
            and correctness.get("actual") == record.get("guest")
            and isinstance(correctness.get("expected"), dict),
            f"record {index} correctness evidence",
        )
        require(
            record.get("cpu_affinity")
            == cpu_affinity_for(
                cpu_placement, spec["guest_workload"], cell["threads"]
            ),
            f"record {index} CPU affinity",
        )
        _validate_record_telemetry(
            record.get("telemetry"),
            f"record {index}.telemetry",
            set(record["cpu_affinity"]),
        )
    require(
        observed_coordinates == expected_coordinates,
        "report records are missing, partial, or unexpected",
    )
    require(
        [
            (
                record["pair_key"],
                record["phase"],
                record["phase_index"],
                record["arm"],
                record["revision"],
                record["condition"],
            )
            for record in records
        ]
        == expected_record_order,
        "report arm/invocation ordering differs from the predeclared sequence and block plan",
    )
    observed_sidecar_cpus = benchmark_cpus(
        cpu_placement, cells_for_platform(platform_id)
    )
    _validate_sidecar(
        document.get("telemetry_sidecar"), observed_sidecar_cpus
    )
    require(
        document["telemetry_sidecar"].get("cpu")
        == telemetry_cpu(cpu_placement, cells_for_platform(platform_id)),
        "telemetry sidecar CPU placement",
    )
    recomputed = summarize_report(records)
    require(summaries == recomputed, "summary recomputation")
    for collection in ("comparisons", "ratio_of_ratios"):
        expected = (
            len(cells_for_platform(platform_id)) * 4
            if collection == "comparisons"
            else len(cells_for_platform(platform_id)) * 2
        )
        require(len(summaries[collection]) == expected, f"summaries.{collection}")


def artifact_identity(context: dict[str, Any]) -> dict[str, str]:
    identity = {}
    for name, build in sorted(context["builds"].items()):
        identity[f"runtime/{name}"] = sha256_file(build.wamr)
        if build.wamrc is not None:
            identity[f"compiler/{name}"] = sha256_file(build.wamrc)
    for name, path in sorted(context["aot_artifacts"].items()):
        identity[f"aot/{name}"] = sha256_file(path)
    identity["fixture/single"] = sha256_file(context["single_wasm"])
    identity["fixture/threaded"] = sha256_file(context["threaded_wasm"])
    return identity


def render_markdown(document: dict[str, Any]) -> str:
    metadata = document["metadata"]
    lines = [
        "# WASI thread duration-cross diagnostic",
        "",
        "> Non-authoritative diagnostic only. This report cannot emit or select a production budget.",
        "",
        f"- Platform: `{metadata['platform_id']}`",
        f"- Sequence/partition: `{metadata['report_sequence']}` / `{metadata['partition']}`",
        f"- Source: `{metadata['source_revision']['commit']}`",
        f"- Plan: `{metadata['plan_identity_sha256']}`",
        f"- Cells: {len(document['plan']['cells'])}",
        f"- Profile: 2 discarded warmups, 12 measured samples in 3 complete blocks",
        f"- Sidecar available: `{document['telemetry_sidecar']['available']}`",
        "",
        "| Arm | Pair | Metric | Throughput candidate/baseline | Elapsed candidate/baseline |",
        "|---|---|---|---:|---:|",
    ]
    for item in document["summaries"]["comparisons"]:
        lines.append(
            f"| `{item['arm']}` | `{item['pair_key']}` | `{item['condition']}` | "
            f"{item['throughput_candidate_over_baseline']['median']:.6f} | "
            f"{item['elapsed_candidate_over_baseline']['median']:.6f} |"
        )
    lines += [
        "",
        "Current-duration observations are retained but select no threshold. "
        "Only a separately validated 20-report cohort may evaluate the doubled arm.",
        "",
    ]
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    missing = [
        name
        for name in (
            "output_dir",
            "platform_id",
            "report_sequence",
            "runner_environment",
            "host_pair_id",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise HarnessError(
            "duration-cross execution requires " + ", ".join(missing)
        )
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    if args.timeout != INVOCATION_TIMEOUT_SECONDS:
        raise HarnessError(
            f"duration-cross timeout is frozen at {INVOCATION_TIMEOUT_SECONDS:g} seconds"
        )
    if int(args.min_interval_ms * 1_000_000) != MINIMUM_INTERVAL_NS:
        raise HarnessError(
            "duration-cross minimum timed interval is frozen at 1250 milliseconds"
        )
    if not str(output).startswith("/d/") and os.getenv("GITHUB_ACTIONS") == "true":
        raise HarnessError("GitHub duration-cross output must remain under /d")
    output.mkdir(parents=True, exist_ok=True)
    for stale in (output / "report.json", output / "report.md"):
        stale.unlink(missing_ok=True)
    source = source_identity(repo)
    if args.source_sha and source["commit"] != args.source_sha:
        raise HarnessError(
            f"source checkout {source['commit']} does not match immutable "
            f"--source-sha {args.source_sha}"
        )
    plan = build_plan(args.platform_id, args.report_sequence, source)
    fixture_repo, fixture_report, fixture_set_sha256 = resolve_measurement_fixtures(
        {"candidate": repo}
    )
    runner = shlex.split(args.runner)
    thread_counts = tuple(
        sorted({cell["threads"] for cell in cells_for_platform(args.platform_id)})
    )
    cpu_placement = discover_cpu_placement(thread_counts)
    host = host_metadata(args.runner_environment)
    if args.platform_id == "ubuntu-22.04-x86_64":
        if os.getenv("GITHUB_ACTIONS") == "true" and host["runner_name"] != TRUSTED_X86_RUNNER_NAME:
            raise HarnessError(
                f"trusted x86 runner must be {TRUSTED_X86_RUNNER_NAME!r}"
            )
        if os.getenv("GITHUB_ACTIONS") == "true" and host["cpu"] != X86_CPU_CLASS:
            raise HarnessError(
                f"trusted x86 CPU must be {X86_CPU_CLASS!r}, got {host['cpu']!r}"
            )
    elif ARM_CPU_CLASS not in host["cpu"]:
        raise HarnessError(
            f"Arm duration-cross requires {ARM_CPU_CLASS} class, got {host['cpu']!r}"
        )
    quiescence = host_quiescence_diagnostics()
    if os.getenv("GITHUB_ACTIONS") == "true" and quiescence["runner_worker_process_count"] != 1:
        raise HarnessError("duration-cross requires exactly one Runner.Worker at start")
    host_pair = host_pair_identity(args.platform_id, host, args.host_pair_id)

    root = output / "build"
    builds = {}
    for mode in ("interpreter", "aot"):
        build = build_variant(
            repo=repo,
            root=root,
            mode=mode,
            threads_enabled=True,
            optimize=args.optimize,
            target=args.target,
            source=source,
            rebuild=args.rebuild,
            compiler_toggle=mode == "aot",
        )
        builds[build.name] = build
    compiler = builds["enabled-aot"]
    if args.target:
        compiler = build_variant(
            repo=repo,
            root=root,
            mode="aot",
            threads_enabled=True,
            optimize=args.optimize,
            target=None,
            source=source,
            rebuild=args.rebuild,
            compiler_toggle=True,
        )
        builds["host-compiler"] = compiler
    aot_artifacts = compile_aot_fixtures(
        fixture_repo, root, compiler, execution_arch(args)
    )
    context = {
        "repo": repo,
        "builds": builds,
        "aot_artifacts": aot_artifacts,
        "aot_artifacts_metadata": aot_artifact_report(
            aot_artifacts, execution_arch(args)
        ),
        "single_wasm": fixture_repo / "tests/benchmarks/wasi-threads/single.wasm",
        "threaded_wasm": fixture_repo / "tests/benchmarks/wasi-threads/threaded.wasm",
    }
    hashes = artifact_identity(context)

    pilot_records = []
    for cell in plan["cells"]:
        for condition in (cell["left"], cell["right"]):
            spec = leg_spec(cell, condition)
            iterations = pilot_iterations(spec, cell["threads"])
            fields = {
                "pilot_index": len(pilot_records),
                "revision": "baseline",
                "pair_kind": cell["pair_kind"],
                "pair_key": cell["pair_key"],
                "condition": condition,
                "mode": spec["mode"],
                "workload": spec["sizing_workload"],
                "sizing_policy_workload": spec["sizing_policy_workload"],
                "threads": cell["threads"],
                "iterations": iterations,
                "phase": "pilot",
            }
            pilot = measured_with_telemetry(
                context=context,
                cpu_placement=cpu_placement,
                runner=runner,
                cell=cell,
                condition=condition,
                iterations=iterations,
                timeout=args.timeout,
                minimum_interval_ns=PILOT_CLOCK_RESOLUTION_MINIMUM_NS,
                enforce_timing_quality=False,
                fields=fields,
            )
            validate_sizing_pilot(pilot, fields)
            pilot_records.append(pilot)
    counts, admission = resolve_counts(plan, pilot_records, args.timeout)
    plan["resolved_counts"] = counts
    plan["pilots"] = pilot_records
    plan["runtime_admission"] = admission

    quality_preflight = run_trusted_barrier_preflight(
        repo=repo,
        runner=runner,
        cpu_placement=cpu_placement,
        build=builds["enabled-aot"],
        module=aot_artifacts["threaded-polls-on"],
        thread_counts=thread_counts,
        iterations_by_thread={
            str(threads): max(
                counts[cell["pair_key"]][condition]
                for cell in plan["cells"]
                for condition in (cell["left"], cell["right"])
                if cell["threads"] == threads
            )
            for threads in thread_counts
        },
        timeout=args.timeout,
        minimum_interval_ns=int(args.min_interval_ms * 1_000_000),
        static_cancel_poll_sites=context["aot_artifacts_metadata"][
            "cancel_poll_static"
        ]["sites_enabled"],
    )
    if quality_preflight["status"] != "passed":
        raise HarnessError("trusted scheduler/barrier preflight failed")
    for index in range(ATOMIC_WAIT_PREFLIGHT_RUNS["authoritative"]):
        measured_with_telemetry(
            context=context,
            cpu_placement=cpu_placement,
            runner=runner,
            cell=_cell(
                "atomic-wait-preflight",
                "runtime-parity",
                "aot",
                "aot",
                "atomic",
                1,
                {"aot": 0.0},
            ),
            condition="aot",
            iterations=ATOMIC_WAIT_PREFLIGHT_ITERATIONS,
            timeout=args.timeout,
            minimum_interval_ns=1,
            fields={
                "revision": "baseline",
                "pair_kind": "atomic-wait-preflight",
                "pair_key": "atomic-wait-preflight",
                "phase": "preflight",
                "phase_index": index,
                "arm": "current",
                "condition": "aot",
                "pair_left": "aot",
                "pair_right": "aot",
            },
        )

    platform_cells = cells_for_platform(args.platform_id)
    sidecar = TelemetrySidecar(
        telemetry_cpu(cpu_placement, platform_cells),
        benchmark_cpus(cpu_placement, platform_cells),
    )
    sidecar.start()
    records: list[dict[str, Any]] = []
    try:
        for cell in plan["cells"]:
            for warmup_index in range(PROFILE["warmups"]):
                for arm in warmup_arm_order(args.report_sequence, warmup_index):
                    for revision, condition in paired_invocation_order(
                        warmup_index,
                        cell["pair_kind"],
                        cell["left"],
                        cell["right"],
                        REVISION_ROLES,
                    ):
                        record = measured_with_telemetry(
                            context=context,
                            cpu_placement=cpu_placement,
                            runner=runner,
                            cell=cell,
                            condition=condition,
                            iterations=counts[cell["pair_key"]][condition]
                            * (2 if arm == "doubled" else 1),
                            timeout=args.timeout,
                            minimum_interval_ns=int(args.min_interval_ms * 1_000_000),
                            fields={
                                "revision": revision,
                                "pair_kind": cell["pair_kind"],
                                "pair_key": cell["pair_key"],
                                "phase": "warmup",
                                "phase_index": warmup_index,
                                "sample_index": None,
                                "block_index": None,
                                "sample_in_block": None,
                                "arm": arm,
                                "condition": condition,
                                "pair_left": cell["left"],
                                "pair_right": cell["right"],
                                "pair_execution": PAIR_EXECUTION_POLICY["default"],
                            },
                        )
                        records.append(record)
            for block_index in range(PROFILE["blocks"]):
                for sample_in_block in range(PROFILE["block_size"]):
                    sample_index = block_index * PROFILE["block_size"] + sample_in_block
                    for arm in arm_order(args.report_sequence, block_index):
                        for revision, condition in paired_invocation_order(
                            sample_in_block,
                            cell["pair_kind"],
                            cell["left"],
                            cell["right"],
                            REVISION_ROLES,
                        ):
                            record = measured_with_telemetry(
                                context=context,
                                cpu_placement=cpu_placement,
                                runner=runner,
                                cell=cell,
                                condition=condition,
                                iterations=counts[cell["pair_key"]][condition]
                                * (2 if arm == "doubled" else 1),
                                timeout=args.timeout,
                                minimum_interval_ns=int(args.min_interval_ms * 1_000_000),
                                fields={
                                    "revision": revision,
                                    "pair_kind": cell["pair_kind"],
                                    "pair_key": cell["pair_key"],
                                    "phase": "measure",
                                    "phase_index": sample_index,
                                    "sample_index": sample_index,
                                    "block_index": block_index,
                                    "sample_in_block": sample_in_block,
                                    "arm": arm,
                                    "condition": condition,
                                    "pair_left": cell["left"],
                                    "pair_right": cell["right"],
                                    "pair_execution": PAIR_EXECUTION_POLICY["default"],
                                },
                            )
                            records.append(record)
    finally:
        sidecar_report = sidecar.stop()

    summaries = summarize_report(records)
    plan_sha256 = cache_key(plan)
    document = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": KIND,
        "authoritative": False,
        "production_budget": None,
        "metadata": {
            "collected_at": collected_at(),
            "platform_id": args.platform_id,
            "report_sequence": args.report_sequence,
            "partition": PARTITIONS[args.report_sequence],
            "cohort_id": args.cohort_id,
            "workflow_run_id": os.getenv("GITHUB_RUN_ID", ""),
            "workflow_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
            "source_revision": source,
            "logical_revisions": {
                revision: copy.deepcopy(source) for revision in REVISION_ROLES
            },
            "artifact_identity": {
                revision: copy.deepcopy(hashes) for revision in REVISION_ROLES
            },
            "fixture_set_sha256": fixture_set_sha256,
            "fixture_source_policy": FIXTURE_SOURCE_POLICY,
            "revision_artifact_policy": REVISION_ARTIFACT_POLICY,
            "plan_sha256": plan_sha256,
            "plan_identity_sha256": plan_identity(plan),
            "host": host,
            "host_pair": host_pair,
            "host_quiescence_at_start": quiescence,
            "cpu_placement": cpu_placement,
            "tools": build_tool_report(builds, runner),
            "fixtures": fixture_report,
            "aot_artifacts": context["aot_artifacts_metadata"],
        },
        "plan": plan,
        "quality_preflight": quality_preflight,
        "records": records,
        "summaries": summaries,
        "telemetry_sidecar": sidecar_report,
    }
    validate_report(document)
    atomic_write_json(output / "report.json", document)
    (output / "report.md").write_text(
        render_markdown(document) + "\n", encoding="UTF-8", newline="\n"
    )
    return document


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--platform-id", choices=tuple(PLATFORM_CELLS))
    parser.add_argument("--report-sequence", type=int, choices=range(1, 21))
    parser.add_argument("--cohort-id", default="manual")
    parser.add_argument("--runner-environment")
    parser.add_argument("--host-pair-id")
    parser.add_argument("--timeout", type=float, default=INVOCATION_TIMEOUT_SECONDS)
    parser.add_argument("--min-interval-ms", type=float, default=1250.0)
    parser.add_argument("--optimize", default="ReleaseFast")
    parser.add_argument("--target")
    parser.add_argument("--aot-target", choices=("x86_64", "aarch64"))
    parser.add_argument("--runner", default="")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--validate-report",
        type=Path,
        help="validate an existing report and exit without executing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.validate_report is not None:
            validate_report(json.loads(args.validate_report.read_text(encoding="UTF-8")))
            print(f"validated {args.validate_report}")
            return 0
        document = execute(args)
        print(render_markdown(document))
        return 0
    except (
        BenchmarkDataError,
        HarnessError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
