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
import statistics
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
    CANONICAL_PLATFORMS,
    DEFAULT_PILOT_ITERATION_PLAN,
    FIXTURE_SOURCE_POLICY,
    HarnessError,
    MASK64,
    PAIR_EXECUTION_POLICY,
    PILOT_CLOCK_RESOLUTION_MINIMUM_NS,
    REVISION_ARTIFACT_POLICY,
    REVISION_ROLES,
    SIZING_SIGNIFICANT_DIGITS,
    WASI_MONOTONIC_CLOCK_ID,
    aot_artifact_report,
    build_tool_report,
    build_variant,
    compile_aot_fixtures,
    cpu_affinity_for,
    discover_cpu_placement,
    effective_sizing_cap,
    execution_arch,
    expected_guest_clock_id,
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
        "optional 1Hz frequency/temperature/package-power samples pinned to a "
        "logical CPU outside every benchmark assignment"
    ),
    "sensor_absence": "record-unavailable-never-retry-or-exclude",
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


def cells_for_platform(platform_id: str) -> tuple[dict[str, Any], ...]:
    try:
        return PLATFORM_CELLS[platform_id]
    except KeyError as exc:
        raise HarnessError(
            f"duration-cross requires one of {sorted(PLATFORM_CELLS)}"
        ) from exc


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
            "guest_workload": "hot",
            "cancel_points": (
                "off" if condition == "cancel-points-off" else "on"
            ),
        }
    return {
        "mode": condition,
        "sizing_workload": cell["workload"],
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


def frequency_snapshot() -> dict[str, Any]:
    paths = sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")
    )
    values = []
    errors = []
    for path in paths:
        try:
            values.append(
                {
                    "cpu": int(path.parts[-3].removeprefix("cpu")),
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


def telemetry_snapshot() -> dict[str, Any]:
    return {
        "collected_at": collected_at(),
        "monotonic_ns": time.monotonic_ns(),
        "proc_stat": read_proc_stat(),
        "pressure": {
            name: read_text(Path(f"/proc/pressure/{name}"))
            for name in ("cpu", "io", "memory")
        },
        "loadavg": read_text(Path("/proc/loadavg")),
        "frequency": frequency_snapshot(),
    }


class TelemetrySidecar:
    def __init__(self, cpu: int | None) -> None:
        self.cpu = cpu
        self.samples: list[dict[str, Any]] = []
        self.status: dict[str, Any] = {
            "requested": True,
            "available": cpu is not None,
            "cpu": cpu,
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
                    "frequency": frequency_snapshot(),
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


def telemetry_cpu(cpu_placement: dict[str, Any], cells: tuple[dict[str, Any], ...]) -> int | None:
    benchmark_cpus: set[int] = set()
    for cell in cells:
        benchmark_cpus.update(
            cpu_affinity_for(cpu_placement, cell["workload"], cell["threads"])
        )
    return next(
        (
            cpu
            for cpu in reversed(cpu_placement["ordered_logical_cpus"])
            if cpu not in benchmark_cpus
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
) -> dict[str, Any]:
    build, module, runtime_args, leg_fields = module_for(context, cell, condition)
    before = telemetry_snapshot()
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
        runtime_args=runtime_args,
        record_fields={**fields, **leg_fields, "iterations": iterations},
    )
    after = telemetry_snapshot()
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
                    "threads": cell["threads"],
                    "iterations": pilot_iterations(spec, cell["threads"]),
                }
            )
    if len(pilots) != len(expected_specs):
        raise HarnessError("duration-cross sizing pilot set is incomplete")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for expected, pilot in zip(expected_specs, pilots, strict=True):
        validate_sizing_pilot(pilot, expected)
        grouped[(expected["mode"], expected["workload"], expected["threads"])].append(
            pilot
        )

    cell_counts: dict[str, dict[str, int]] = {}
    sizing_cells: list[dict[str, Any]] = []
    selected_by_sizing_key: dict[tuple[str, str, int], int] = {}
    for key, values in sorted(grouped.items()):
        fastest = min(values, key=lambda item: item["guest_elapsed_ns"])
        candidates = sizing_candidates_for_cell(
            *key, fastest["iterations"], fastest["guest_elapsed_ns"]
        )
        current = candidates["selected_iterations"]
        doubled = current * 2
        cap = effective_sizing_cap(key[1], key[2])
        if doubled > cap or doubled > MASK64 // key[2]:
            raise HarnessError(
                f"duration-cross {key} doubled count {doubled} cannot be admitted "
                f"under frozen cap {cap}; counts must not be shortened"
            )
        selected_by_sizing_key[key] = current
        sizing_cells.append(
            {
                "key": f"{key[0]}/{key[1]}/{key[2]}",
                "mode": key[0],
                "workload": key[1],
                "threads": key[2],
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
        key = (pilot["mode"], pilot["workload"], pilot["threads"])
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
                guest < projected_duration_floor_for_cell(*key) * multiplier
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
            key = (spec["mode"], spec["sizing_workload"], cell["threads"])
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
    require(plan.get("cells") == list(cells_for_platform(platform_id)), "plan.cells")
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
    _required_hash(metadata.get("plan_sha256"), "metadata.plan_sha256")
    _required_hash(metadata.get("plan_identity_sha256"), "metadata.plan_identity_sha256")
    require(metadata["plan_sha256"] == cache_key(plan), "metadata plan hash")
    require(metadata["plan_identity_sha256"] == plan_identity(plan), "plan identity hash")
    source = metadata.get("source_revision")
    revisions = metadata.get("logical_revisions")
    require(
        isinstance(source, dict)
        and isinstance(revisions, dict)
        and set(revisions) == set(REVISION_ROLES),
        "artifact A/A identities",
    )
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
        expected_iterations = resolved[pair_key][condition] * (
            2 if arm == "doubled" else 1
        )
        require(record.get("iterations") == expected_iterations, f"record {index} count")
        require(record.get("correct") is True, f"record {index} correctness")
        telemetry = record.get("telemetry")
        require(
            isinstance(telemetry, dict)
            and telemetry.get("collection") == TELEMETRY_POLICY["pre_post"]
            and all(
                isinstance(telemetry.get(side), dict)
                and "proc_stat" in telemetry[side]
                and "pressure" in telemetry[side]
                and "loadavg" in telemetry[side]
                and "frequency" in telemetry[side]
                for side in ("before", "after")
            ),
            f"record {index} telemetry",
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
    require(
        isinstance(document.get("telemetry_sidecar"), dict)
        and isinstance(document["telemetry_sidecar"].get("available"), bool)
        and isinstance(document["telemetry_sidecar"].get("samples"), list)
        and len(document["telemetry_sidecar"].get("samples", []))
        <= SIDECAR_MAX_SAMPLES
        and (
            document["telemetry_sidecar"]["available"]
            or bool(document["telemetry_sidecar"].get("reason"))
        ),
        "telemetry sidecar availability",
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
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
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
                "threads": cell["threads"],
                "iterations": iterations,
                "phase": "pilot",
            }
            build, module, runtime_args, leg_fields = module_for(context, cell, condition)
            pilot = measure_once(
                repo=repo,
                runner=runner,
                cpu_placement=cpu_placement,
                build=build,
                module=module,
                workload=leg_fields["workload"],
                threads=cell["threads"],
                iterations=iterations,
                timeout=args.timeout,
                min_interval_ns=PILOT_CLOCK_RESOLUTION_MINIMUM_NS,
                enforce_timing_quality=False,
                runtime_args=runtime_args,
                record_fields={
                    **leg_fields,
                    **fields,
                    "guest_workload": leg_fields["workload"],
                },
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

    sidecar = TelemetrySidecar(
        telemetry_cpu(cpu_placement, cells_for_platform(args.platform_id))
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--platform-id", choices=tuple(PLATFORM_CELLS), required=True)
    parser.add_argument("--report-sequence", type=int, choices=range(1, 21), required=True)
    parser.add_argument("--cohort-id", default="manual")
    parser.add_argument("--runner-environment", required=True)
    parser.add_argument("--host-pair-id", required=True)
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
