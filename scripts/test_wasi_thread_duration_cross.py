#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_schema import BenchmarkDataError, cache_key, sha256_file  # noqa: E402
from bench_wasi_threads import (  # noqa: E402
    HarnessError,
    cpu_affinity_for,
    cpu_placement_from_topology,
    expected_guest_clock_id,
    expected_result,
    paired_invocation_order,
    projected_duration_floor_for_cell,
    WASI_MONOTONIC_CLOCK_ID,
)
import wasi_thread_duration_cross as duration  # noqa: E402
import wasi_thread_duration_cross_cohort as cohort  # noqa: E402


SHA = "1" * 40
DIGEST = "2" * 64
COHORT_ID = "3" * 32


def telemetry() -> dict:
    snapshot = {
        "collected_at": "2026-09-21T00:00:00+00:00",
        "monotonic_ns": 1,
        "proc_stat": {
            "available": True,
            "cpu": {
                "user": 1,
                "nice": 0,
                "system": 1,
                "idle": 1,
                "iowait": 0,
                "irq": 0,
                "softirq": 0,
                "steal": 0,
            },
            "ctxt": 1,
            "processes": 1,
            "procs_running": 1,
            "procs_blocked": 0,
        },
        "pressure": {
            name: {"available": True, "value": "some avg10=0.00"}
            for name in ("cpu", "io", "memory")
        },
        "loadavg": {"available": True, "value": "0 0 0 1/1 1"},
        "frequency": {
            "available": False,
            "samples": [],
            "unavailable": [],
            "reason": "no readable scaling_cur_freq sensors",
        },
    }
    return {
        "collection": duration.TELEMETRY_POLICY["pre_post"],
        "before": copy.deepcopy(snapshot),
        "after": copy.deepcopy(snapshot),
    }


def synthetic_report(
    *,
    sequence: int = 1,
    platform: str = "ubuntu-24.04-aarch64",
    ratio: float = 1.0,
) -> dict:
    source = {
        "commit": SHA,
        "tracked_diff_sha256": DIGEST,
        "build_source_sha256": DIGEST,
    }
    plan = duration.build_plan(platform, sequence, source)
    pilots = []
    for cell in plan["cells"]:
        for condition in (cell["left"], cell["right"]):
            spec = duration.leg_spec(cell, condition)
            iterations = duration.pilot_iterations(spec, cell["threads"])
            elapsed = projected_duration_floor_for_cell(
                spec["mode"],
                spec["sizing_policy_workload"],
                cell["threads"],
            )
            guest_workload = spec["guest_workload"]
            host_wall = (
                elapsed
                if expected_guest_clock_id(guest_workload)
                == WASI_MONOTONIC_CLOCK_ID
                else max(1_000_000_000, elapsed // cell["threads"])
            )
            pilots.append(
                {
                    "pilot_index": len(pilots),
                    "revision": "baseline",
                    "pair_kind": cell["pair_kind"],
                    "pair_key": cell["pair_key"],
                    "condition": condition,
                    "mode": spec["mode"],
                    "workload": spec["sizing_workload"],
                    "sizing_policy_workload": spec["sizing_policy_workload"],
                    "threads": cell["threads"],
                    "iterations": iterations,
                    "guest_elapsed_ns": elapsed,
                    "raw_guest_elapsed_ns": elapsed,
                    "timing_overhead_ns": 0,
                    "host_wall_elapsed_ns": host_wall,
                    "operations": expected_result(
                        guest_workload, cell["threads"], iterations
                    )["operations"],
                    "correct": True,
                    "telemetry": telemetry(),
                }
            )
    counts, admission = duration.resolve_counts(
        plan, pilots, duration.INVOCATION_TIMEOUT_SECONDS
    )
    plan["pilots"] = pilots
    plan["resolved_counts"] = counts
    plan["runtime_admission"] = admission
    cpu_placement = cpu_placement_from_topology(
        list(range(16)),
        {cpu: (0, cpu) for cpu in range(16)},
        tuple(sorted({cell["threads"] for cell in plan["cells"]})),
        "taskset synthetic",
    )
    plan["resolved_counts"] = counts
    records = []
    for cell in plan["cells"]:
        for warmup_index in range(duration.PROFILE["warmups"]):
            for arm in duration.warmup_arm_order(sequence, warmup_index):
                for revision, condition in paired_invocation_order(
                    warmup_index,
                    cell["pair_kind"],
                    cell["left"],
                    cell["right"],
                    duration.REVISION_ROLES,
                ):
                    records.append(
                        synthetic_record(
                            cell,
                            "warmup",
                            warmup_index,
                            arm,
                            revision,
                            condition,
                            counts,
                            ratio,
                            cpu_placement,
                        )
                    )
        for block in range(duration.PROFILE["blocks"]):
            for position in range(duration.PROFILE["block_size"]):
                sample = block * 4 + position
                for arm in duration.arm_order(sequence, block):
                    for revision, condition in paired_invocation_order(
                        position,
                        cell["pair_kind"],
                        cell["left"],
                        cell["right"],
                        duration.REVISION_ROLES,
                    ):
                        records.append(
                            synthetic_record(
                                cell,
                                "measure",
                                sample,
                                arm,
                                revision,
                                condition,
                                counts,
                                ratio,
                                cpu_placement,
                            )
                        )
    artifact = {
        name: DIGEST
        for name in (
            "runtime/enabled-aot",
            "compiler/enabled-aot",
            "runtime/enabled-interpreter",
            "aot/single",
            "aot/threaded-polls-off",
            "aot/threaded-polls-on",
            "fixture/single",
            "fixture/threaded",
        )
    }
    report = {
        "schema_version": duration.REPORT_SCHEMA_VERSION,
        "kind": duration.KIND,
        "authoritative": False,
        "production_budget": None,
        "metadata": {
            "collected_at": "2026-09-21T00:00:00+00:00",
            "platform_id": platform,
            "report_sequence": sequence,
            "partition": duration.PARTITIONS[sequence],
            "cohort_id": COHORT_ID,
            "workflow_run_id": str(10_000 + sequence),
            "workflow_run_attempt": "1",
            "source_revision": source,
            "logical_revisions": {
                role: copy.deepcopy(source) for role in duration.REVISION_ROLES
            },
            "artifact_identity": {
                role: copy.deepcopy(artifact) for role in duration.REVISION_ROLES
            },
            "fixture_set_sha256": DIGEST,
            "fixture_source_policy": "candidate-measurement-fixtures-for-all-revisions",
            "revision_artifact_policy": (
                "reuse-exact-artifacts-for-identical-noise-calibration-revisions"
            ),
            "plan_sha256": cache_key(plan),
            "plan_identity_sha256": duration.plan_identity(plan),
            "host": {
                "cpu": (
                    cohort.X86_CPU_CLASS
                    if platform == "ubuntu-22.04-x86_64"
                    else "Arm Neoverse-N2"
                ),
                "runner_name": (
                    duration.TRUSTED_X86_RUNNER_NAME
                    if platform == "ubuntu-22.04-x86_64"
                    else "GitHub Actions 1"
                ),
            },
            "host_pair": {"host_fingerprint_sha256": DIGEST},
            "host_quiescence_at_start": {"runner_worker_process_count": 1},
            "cpu_placement": cpu_placement,
            "tools": {},
            "fixtures": {},
            "aot_artifacts": {},
        },
        "plan": plan,
        "quality_preflight": {"enabled": True, "status": "passed"},
        "records": records,
        "summaries": duration.summarize_report(records),
        "telemetry_sidecar": {
            "requested": True,
            "available": False,
            "cpu": duration.telemetry_cpu(
                cpu_placement, duration.cells_for_platform(platform)
            ),
            "observed_cpus": sorted(
                duration.benchmark_cpus(
                    cpu_placement, duration.cells_for_platform(platform)
                )
            ),
            "interval_seconds": 1,
            "reason": "affinity failed: synthetic",
            "samples": [],
        },
    }
    return report


def synthetic_record(
    cell: dict,
    phase: str,
    index: int,
    arm: str,
    revision: str,
    condition: str,
    counts: dict,
    ratio: float,
    cpu_placement: dict,
) -> dict:
    elapsed = 2_000_000_000
    operations = 2_000
    if revision == "candidate":
        elapsed = round(elapsed * ratio)
    throughput = operations / (elapsed / 1_000_000_000)
    guest = {
        "metric_kind": (
            "spawn-join-lifecycle"
            if cell["workload"] == "spawn-join"
            else "steady-state-kernel"
        )
    }
    return {
        "revision": revision,
        "pair_kind": cell["pair_kind"],
        "pair_key": cell["pair_key"],
        "phase": phase,
        "phase_index": index,
        "sample_index": index if phase == "measure" else None,
        "block_index": index // 4 if phase == "measure" else None,
        "sample_in_block": index % 4 if phase == "measure" else None,
        "arm": arm,
        "condition": condition,
        "pair_left": cell["left"],
        "pair_right": cell["right"],
        "pair_execution": "sequential-position-balanced",
        "mode": duration.leg_spec(cell, condition)["mode"],
        "workload": cell["workload"],
        "threads": cell["threads"],
        "cancel_points": duration.leg_spec(cell, condition)["cancel_points"],
        "cpu_affinity": cpu_affinity_for(
            cpu_placement, cell["workload"], cell["threads"]
        ),
        "iterations": counts[cell["pair_key"]][condition]
        * (2 if arm == "doubled" else 1),
        "elapsed_ns": elapsed,
        "guest_elapsed_ns": elapsed,
        "raw_guest_elapsed_ns": elapsed,
        "timing_overhead_ns": 0,
        "host_started_ns": 1,
        "host_finished_ns": elapsed + 1,
        "host_wall_elapsed_ns": elapsed,
        "operations": operations,
        "throughput_ops_per_second": throughput,
        "metric_kind": guest["metric_kind"],
        "guest": guest,
        "correct": True,
        "correctness": {"passed": True, "expected": {}, "actual": guest},
        "telemetry": telemetry(),
    }


def metric_stats(value: float) -> dict:
    return {
        "samples": [value] * 12,
        "runs": 12,
        "mean": value,
        "median": value,
        "min": value,
        "max": value,
        "range": 0.0,
        "estimator": "median-of-four-position-geometric-means",
        "position_block_size": 4,
        "position_block_estimates": [value] * 3,
    }


def synthetic_summaries(platform: str, value: float = 1.0) -> dict:
    comparisons = []
    ratios = []
    internal = []
    for arm in duration.ARMS:
        for cell in duration.cells_for_platform(platform):
            for condition in (cell["left"], cell["right"]):
                comparisons.append(
                    {
                        "arm": arm,
                        "pair_kind": cell["pair_kind"],
                        "pair_key": cell["pair_key"],
                        "condition": condition,
                        "metric_kind": (
                            "spawn-join-lifecycle"
                            if cell["workload"] == "spawn-join"
                            else "steady-state-kernel"
                        ),
                        "elapsed_candidate_over_baseline": metric_stats(value),
                        "throughput_candidate_over_baseline": metric_stats(value),
                    }
                )
            ratios.append(
                {
                    "arm": arm,
                    "pair_kind": cell["pair_kind"],
                    "pair_key": cell["pair_key"],
                    "left": cell["left"],
                    "right": cell["right"],
                    "elapsed_ratio_of_ratios": metric_stats(value),
                    "throughput_ratio_of_ratios": metric_stats(value),
                }
            )
            for revision in duration.REVISION_ROLES:
                internal.append(
                    {
                        "arm": arm,
                        "pair_kind": cell["pair_kind"],
                        "pair_key": cell["pair_key"],
                        "revision": revision,
                    }
                )
    return {
        "comparisons": comparisons,
        "ratio_of_ratios": ratios,
        "internal_pairs": internal,
    }


def completed_dispatch() -> dict:
    result = cohort.make_dispatch_plan(
        source_sha=SHA,
        workflow_ref="wasi-thread-duration-cross-v21",
        repository="cataggar/wamr",
        cohort_id=COHORT_ID,
    )
    result["runs"] = [
        {
            "sequence": sequence,
            "partition": duration.PARTITIONS[sequence],
            "run_id": 10_000 + sequence,
            "url": f"https://example.invalid/{sequence}",
            "status": "completed",
            "conclusion": "success",
            "attempt": 1,
            "artifacts": [
                {
                    "id": sequence * 10 + index,
                    "name": (
                        f"wasi-thread-duration-cross-{platform}-"
                        f"{10_000 + sequence}-1"
                    ),
                    "size_in_bytes": 1000,
                    "digest_sha256": cache_key(
                        {
                            "sequence": sequence,
                            "platform": platform,
                            "kind": "artifact-zip",
                        }
                    ),
                    "expired": False,
                }
                for index, platform in enumerate(duration.PLATFORM_CELLS, 1)
            ],
        }
        for sequence in range(1, 21)
    ]
    return result


def synthetic_cohort(value: float = 1.0) -> dict:
    observations = []
    for sequence in range(1, 21):
        for platform in duration.PLATFORM_CELLS:
            report_digest = cache_key(
                {"sequence": sequence, "platform": platform, "kind": "canonical"}
            )
            observations.append(
                {
                    "report_sha256": report_digest,
                    "report_file_sha256": cache_key(
                        {"sequence": sequence, "platform": platform, "kind": "json"}
                    ),
                    "report_markdown_sha256": cache_key(
                        {"sequence": sequence, "platform": platform, "kind": "markdown"}
                    ),
                    "artifact_zip_sha256": cache_key(
                        {"sequence": sequence, "platform": platform, "kind": "zip"}
                    ),
                    "artifact_identity_sha256": cache_key(
                        {
                            "sequence": sequence,
                            "platform": platform,
                            "kind": "artifacts",
                        }
                    ),
                    "run_id": str(10_000 + sequence),
                    "workflow_run_attempt": "1",
                    "artifact_id": sequence * 10 + (
                        1 if platform == "ubuntu-22.04-x86_64" else 2
                    ),
                    "sequence": sequence,
                    "partition": duration.PARTITIONS[sequence],
                    "platform": platform,
                    "artifact_name": (
                        f"wasi-thread-duration-cross-{platform}-"
                        f"{10_000 + sequence}-1"
                    ),
                    "cpu_class": (
                        cohort.X86_CPU_CLASS
                        if platform == "ubuntu-22.04-x86_64"
                        else "Arm Neoverse-N2"
                    ),
                    "host_fingerprint_sha256": DIGEST,
                    "plan_sha256": DIGEST,
                    "plan_identity_sha256": DIGEST,
                    "telemetry_sidecar": {
                        "requested": True,
                        "available": False,
                        "cpu": None,
                        "observed_cpus": [],
                        "interval_seconds": 1,
                        "reason": "unavailable",
                        "sample_count": 0,
                    },
                    "summaries": synthetic_summaries(platform, value),
                }
            )
    return {
        "schema_version": 1,
        "kind": cohort.COHORT_KIND,
        "authoritative": False,
        "production_budget": None,
        "validated_at": "2026-09-21T00:00:00+00:00",
        "dispatch": completed_dispatch(),
        "download_manifest_sha256": DIGEST,
        "identity": {
            "source_revision": {
                "commit": SHA,
                "tracked_diff_sha256": DIGEST,
                "build_source_sha256": DIGEST,
            },
            "platforms": {
                platform: {"plan_kind": duration.PLAN_KIND}
                for platform in duration.PLATFORM_CELLS
            },
        },
        "platform_counts": {platform: 20 for platform in duration.PLATFORM_CELLS},
        "partition_counts": {
            platform: {"training": 16, "holdout": 4}
            for platform in duration.PLATFORM_CELLS
        },
        "excluded_observations": [],
        "observations": observations,
    }


def synthetic_download_manifest(root: Path) -> tuple[dict, dict]:
    dispatch = completed_dispatch()
    entries = []
    for run in dispatch["runs"]:
        for artifact in run["artifacts"]:
            artifact_dir = root / artifact["name"]
            artifact_dir.mkdir(parents=True)
            json_path = artifact_dir / "report.json"
            markdown_path = artifact_dir / "report.md"
            zip_path = artifact_dir / "artifact.zip"
            json_path.write_text(
                json.dumps({"artifact": artifact["name"]}) + "\n",
                encoding="UTF-8",
            )
            markdown_path.write_text(
                f"# {artifact['name']}\n",
                encoding="UTF-8",
            )
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(json_path, "report.json")
                archive.write(markdown_path, "report.md")
            artifact["size_in_bytes"] = zip_path.stat().st_size
            artifact["digest_sha256"] = sha256_file(zip_path)
            platform = artifact["name"].removeprefix(
                "wasi-thread-duration-cross-"
            ).removesuffix(f"-{run['run_id']}-1")
            entries.append(
                {
                    "sequence": run["sequence"],
                    "partition": run["partition"],
                    "run_id": run["run_id"],
                    "workflow_run_attempt": 1,
                    "platform": platform,
                    "artifact_id": artifact["id"],
                    "artifact_name": artifact["name"],
                    "artifact_size_in_bytes": artifact["size_in_bytes"],
                    "artifact_zip": {
                        "path": f"{artifact['name']}/artifact.zip",
                        "sha256": sha256_file(zip_path),
                        "size_in_bytes": zip_path.stat().st_size,
                    },
                    "report_json": {
                        "path": f"{artifact['name']}/report.json",
                        "sha256": sha256_file(json_path),
                    },
                    "report_markdown": {
                        "path": f"{artifact['name']}/report.md",
                        "sha256": sha256_file(markdown_path),
                    },
                }
            )
    return dispatch, {
        "schema_version": 1,
        "kind": cohort.DOWNLOAD_MANIFEST_KIND,
        "downloaded_at": "2026-09-21T00:00:00+00:00",
        "dispatch_sha256": cache_key(dispatch),
        "cohort_id": COHORT_ID,
        "source_sha": SHA,
        "entries": sorted(
            entries, key=lambda item: (item["sequence"], item["platform"])
        ),
    }


def retained_fixture(
    root: Path,
    *,
    ratios: dict[tuple[int, str], float] | None = None,
) -> dict[str, Path]:
    ratios = ratios or {}
    input_dir = root / "reports"
    input_dir.mkdir(parents=True)
    dispatch = completed_dispatch()
    entries = []
    for run in dispatch["runs"]:
        for artifact in run["artifacts"]:
            platform = artifact["name"].removeprefix(
                "wasi-thread-duration-cross-"
            ).removesuffix(f"-{run['run_id']}-1")
            artifact_dir = input_dir / artifact["name"]
            artifact_dir.mkdir()
            report = synthetic_report(
                sequence=run["sequence"],
                platform=platform,
                ratio=ratios.get((run["sequence"], platform), 1.0),
            )
            report_path = artifact_dir / "report.json"
            markdown_path = artifact_dir / "report.md"
            zip_path = artifact_dir / "artifact.zip"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="UTF-8",
            )
            markdown_path.write_text(
                duration.render_markdown(report) + "\n",
                encoding="UTF-8",
            )
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(report_path, "report.json")
                archive.write(markdown_path, "report.md")
            artifact["size_in_bytes"] = zip_path.stat().st_size
            artifact["digest_sha256"] = sha256_file(zip_path)
            entries.append(
                {
                    "sequence": run["sequence"],
                    "partition": run["partition"],
                    "run_id": run["run_id"],
                    "workflow_run_attempt": 1,
                    "platform": platform,
                    "artifact_id": artifact["id"],
                    "artifact_name": artifact["name"],
                    "artifact_size_in_bytes": artifact["size_in_bytes"],
                    "artifact_zip": {
                        "path": f"{artifact['name']}/artifact.zip",
                        "sha256": sha256_file(zip_path),
                        "size_in_bytes": zip_path.stat().st_size,
                    },
                    "report_json": {
                        "path": f"{artifact['name']}/report.json",
                        "sha256": sha256_file(report_path),
                    },
                    "report_markdown": {
                        "path": f"{artifact['name']}/report.md",
                        "sha256": sha256_file(markdown_path),
                    },
                }
            )
    dispatch_path = root / "dispatch.json"
    dispatch_path.write_text(
        json.dumps(dispatch, indent=2, sort_keys=True) + "\n",
        encoding="UTF-8",
    )
    manifest = {
        "schema_version": 1,
        "kind": cohort.DOWNLOAD_MANIFEST_KIND,
        "downloaded_at": "2026-09-21T00:00:00+00:00",
        "dispatch_sha256": cache_key(dispatch),
        "cohort_id": COHORT_ID,
        "source_sha": SHA,
        "entries": sorted(
            entries, key=lambda item: (item["sequence"], item["platform"])
        ),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="UTF-8",
    )
    cohort_path = root / "cohort.json"
    cohort.validate_cohort(
        input_dir,
        dispatch_path,
        manifest_path,
        cohort_path,
    )
    return {
        "input_dir": input_dir,
        "dispatch": dispatch_path,
        "manifest": manifest_path,
        "cohort": cohort_path,
    }


class DurationCrossReportTests(unittest.TestCase):
    def test_exact_cells_targets_and_identity_are_isolated(self) -> None:
        self.assertEqual(len(duration.X86_CELLS), 11)
        self.assertEqual(len(duration.ARM_CELLS), 3)
        self.assertEqual(
            {
                (platform, cell["pair_key"]): cell["current_target_seconds"]
                for platform, cells in duration.PLATFORM_CELLS.items()
                for cell in cells
            },
            {
                ("ubuntu-22.04-x86_64", "cancel-points/hot/8"): {
                    "cancel-points-off": 20.0,
                    "cancel-points-on": 20.0,
                },
                ("ubuntu-22.04-x86_64", "runtime/atomic/1"): {
                    "interpreter": 5.0,
                    "aot": 5.0,
                },
                ("ubuntu-22.04-x86_64", "runtime/atomic/2"): {
                    "interpreter": 40.0,
                    "aot": 40.0,
                },
                ("ubuntu-22.04-x86_64", "runtime/atomic/4"): {
                    "interpreter": 40.0,
                    "aot": 40.0,
                },
                ("ubuntu-22.04-x86_64", "runtime/atomic/8"): {
                    "interpreter": 20.0,
                    "aot": 20.0,
                },
                ("ubuntu-22.04-x86_64", "runtime/hot/1"): {
                    "interpreter": 5.0,
                    "aot": 5.0,
                },
                ("ubuntu-22.04-x86_64", "runtime/hot/8"): {
                    "interpreter": 20.0,
                    "aot": 20.0,
                },
                ("ubuntu-22.04-x86_64", "runtime/spawn-join/1"): {
                    "interpreter": 2.5,
                    "aot": 2.5,
                },
                ("ubuntu-22.04-x86_64", "runtime/spawn-join/2"): {
                    "interpreter": 2.5,
                    "aot": 2.5,
                },
                ("ubuntu-22.04-x86_64", "runtime/spawn-join/8"): {
                    "interpreter": 2.5,
                    "aot": 2.5,
                },
                ("ubuntu-22.04-x86_64", "runtime/wait-notify/1"): {
                    "interpreter": 5.0,
                    "aot": 20.0,
                },
                ("ubuntu-24.04-aarch64", "runtime/atomic/2"): {
                    "interpreter": 40.0,
                    "aot": 40.0,
                },
                ("ubuntu-24.04-aarch64", "runtime/atomic/4"): {
                    "interpreter": 40.0,
                    "aot": 40.0,
                },
                ("ubuntu-24.04-aarch64", "runtime/atomic/8"): {
                    "interpreter": 20.0,
                    "aot": 20.0,
                },
            },
        )
        hot = next(
            cell
            for cell in duration.X86_CELLS
            if cell["pair_key"] == "cancel-points/hot/8"
        )
        self.assertEqual(
            hot["current_target_seconds"],
            {"cancel-points-off": 20.0, "cancel-points-on": 20.0},
        )
        self.assertEqual(
            duration.leg_spec(hot, "cancel-points-on")[
                "sizing_policy_workload"
            ],
            "hot",
        )
        wait = next(
            cell
            for cell in duration.X86_CELLS
            if cell["pair_key"] == "runtime/wait-notify/1"
        )
        self.assertEqual(
            wait["current_target_seconds"],
            {"interpreter": 5.0, "aot": 20.0},
        )
        report = synthetic_report()
        duration.validate_report(report)
        self.assertEqual(report["kind"], duration.KIND)
        self.assertNotEqual(report["kind"], "wasi-thread-benchmark")
        self.assertEqual(report["plan"]["version"], 21)
        self.assertEqual(
            len(duration.acceptance_checks_for_platform("ubuntu-22.04-x86_64")),
            19,
        )
        self.assertEqual(
            len(duration.acceptance_checks_for_platform("ubuntu-24.04-aarch64")),
            9,
        )
        self.assertIsNone(report["production_budget"])
        independently_built = copy.deepcopy(report)
        for role in duration.REVISION_ROLES:
            independently_built["metadata"]["artifact_identity"][role][
                "runtime/enabled-aot"
            ] = "4" * 64
        self.assertEqual(
            cohort.report_identity(report),
            cohort.report_identity(independently_built),
        )

    def test_arm_order_alternates_by_sequence_and_block(self) -> None:
        self.assertEqual(duration.arm_order(1, 0), ("current", "doubled"))
        self.assertEqual(duration.arm_order(1, 1), ("doubled", "current"))
        self.assertEqual(duration.arm_order(2, 0), ("doubled", "current"))
        report = synthetic_report(sequence=2)
        duration.validate_report(report)
        changed = copy.deepcopy(report)
        changed["records"][0], changed["records"][1] = (
            changed["records"][1],
            changed["records"][0],
        )
        changed["summaries"] = duration.summarize_report(changed["records"])
        with self.assertRaisesRegex(BenchmarkDataError, "ordering"):
            duration.validate_report(changed)

    def test_rejects_missing_duplicate_partial_and_unexpected_records(self) -> None:
        report = synthetic_report()
        missing = copy.deepcopy(report)
        missing["records"].pop()
        with self.assertRaisesRegex(BenchmarkDataError, "missing|incomplete"):
            duration.validate_report(missing)
        duplicate = copy.deepcopy(report)
        duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
        with self.assertRaisesRegex(BenchmarkDataError, "duplicate"):
            duration.validate_report(duplicate)
        unexpected = copy.deepcopy(report)
        unexpected["records"][0]["pair_key"] = "runtime/unexpected/1"
        with self.assertRaisesRegex(BenchmarkDataError, "unexpected"):
            duration.validate_report(unexpected)

    def test_rejects_mixed_artifacts_and_dishonest_telemetry(self) -> None:
        report = synthetic_report()
        mixed = copy.deepcopy(report)
        mixed["metadata"]["artifact_identity"]["candidate"][
            "runtime/enabled-aot"
        ] = "4" * 64
        with self.assertRaisesRegex(BenchmarkDataError, "artifact-identical"):
            duration.validate_report(mixed)
        absent = copy.deepcopy(report)
        absent["telemetry_sidecar"]["available"] = False
        absent["telemetry_sidecar"]["reason"] = ""
        with self.assertRaisesRegex(BenchmarkDataError, "unavailable reason"):
            duration.validate_report(absent)
        missing = copy.deepcopy(report)
        del missing["records"][0]["telemetry"]["before"]["proc_stat"]
        with self.assertRaisesRegex(BenchmarkDataError, "telemetry"):
            duration.validate_report(missing)

    def test_rejects_unadmitted_doubled_count_without_adaptation(self) -> None:
        report = synthetic_report()
        cell = report["plan"]["cells"][0]
        condition = cell["left"]
        spec = duration.leg_spec(cell, condition)
        cap = duration.effective_sizing_cap(
            spec["sizing_workload"], cell["threads"]
        )
        report["plan"]["resolved_counts"][cell["pair_key"]][condition] = (
            cap // 2 + 1
        )
        report["metadata"]["plan_sha256"] = cache_key(report["plan"])
        report["metadata"]["plan_identity_sha256"] = duration.plan_identity(
            report["plan"]
        )
        with self.assertRaisesRegex(BenchmarkDataError, "pilot-derived|count admission"):
            duration.validate_report(report)

    def test_schema_files_pin_diagnostic_identity(self) -> None:
        report_schema = json.loads(
            (
                REPO_ROOT
                / "tests/benchmarks/wasi-threads/duration-cross-report.schema.json"
            ).read_text(encoding="UTF-8")
        )
        cohort_schema = json.loads(
            (
                REPO_ROOT
                / "tests/benchmarks/wasi-threads/duration-cross-cohort.schema.json"
            ).read_text(encoding="UTF-8")
        )
        self.assertEqual(
            report_schema["properties"]["kind"]["const"], duration.KIND
        )
        self.assertEqual(
            report_schema["properties"]["plan"]["properties"]["version"]["const"],
            21,
        )
        self.assertEqual(
            cohort_schema["properties"]["kind"]["const"], cohort.COHORT_KIND
        )
        self.assertEqual(
            report_schema["properties"]["production_budget"]["type"], "null"
        )

    def test_validate_report_mode_needs_no_execution_arguments(self) -> None:
        scratch = REPO_ROOT / "zig-out" / f"duration-report-{uuid.uuid4().hex}"
        scratch.mkdir(parents=True)
        try:
            report_path = scratch / "report.json"
            report_path.write_text(
                json.dumps(synthetic_report()), encoding="UTF-8"
            )
            self.assertEqual(
                duration.main(["--validate-report", str(report_path)]), 0
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


class DurationCrossCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_root = (
            REPO_ROOT / "zig-out" / f"duration-cross-bound-{uuid.uuid4().hex}"
        )
        cls.fixture_root.mkdir(parents=True)
        cls.passing_fixture = retained_fixture(cls.fixture_root / "passing")
        cls.failing_fixture = retained_fixture(
            cls.fixture_root / "failing",
            ratios={
                (1, "ubuntu-22.04-x86_64"): math.exp(0.20),
                (17, "ubuntu-24.04-aarch64"): 1.25,
            },
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.fixture_root, ignore_errors=True)

    def setUp(self) -> None:
        self.scratch = (
            REPO_ROOT / "zig-out" / f"duration-cross-unit-{uuid.uuid4().hex}"
        )
        self.scratch.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def analyze_fixture(
        self,
        fixture: dict[str, Path],
        output: Path,
        *,
        policy: str = "derivation-policy.production.json",
    ) -> int:
        with mock.patch.object(
            cohort, "require_live_dispatch_membership", return_value=None
        ):
            return cohort.analyze(
                fixture["cohort"],
                fixture["input_dir"],
                fixture["dispatch"],
                fixture["manifest"],
                REPO_ROOT / "tests/benchmarks/wasi-threads" / policy,
                output,
            )

    def test_dispatch_membership_is_exact_and_no_retry(self) -> None:
        plan = completed_dispatch()
        cohort.validate_dispatch_plan(plan, completed=True)
        self.assertEqual(
            [item["partition"] for item in plan["split"]["assignments"][:16]],
            ["training"] * 16,
        )
        self.assertEqual(
            [item["partition"] for item in plan["split"]["assignments"][16:]],
            ["holdout"] * 4,
        )
        self.assertEqual(plan["retry_policy"]["retries"], 0)
        partial = copy.deepcopy(plan)
        partial["runs"].pop()
        with self.assertRaisesRegex(HarnessError, "exactly 20"):
            cohort.validate_dispatch_plan(partial, completed=True)
        duplicate_run = copy.deepcopy(plan)
        duplicate_run["runs"][1]["run_id"] = duplicate_run["runs"][0]["run_id"]
        duplicate_run["runs"][1]["artifacts"] = copy.deepcopy(
            duplicate_run["runs"][0]["artifacts"]
        )
        with self.assertRaisesRegex(HarnessError, "duplicate run ID"):
            cohort.validate_dispatch_plan(duplicate_run, completed=True)
        duplicate_artifact = copy.deepcopy(plan)
        duplicate_artifact["runs"][1]["artifacts"][0]["id"] = (
            duplicate_artifact["runs"][0]["artifacts"][0]["id"]
        )
        with self.assertRaisesRegex(HarnessError, "duplicate artifact ID"):
            cohort.validate_dispatch_plan(duplicate_artifact, completed=True)
        duplicate_zip = copy.deepcopy(plan)
        duplicate_zip["runs"][1]["artifacts"][0]["digest_sha256"] = (
            duplicate_zip["runs"][0]["artifacts"][0]["digest_sha256"]
        )
        with self.assertRaisesRegex(HarnessError, "duplicate artifact ZIP"):
            cohort.validate_dispatch_plan(duplicate_zip, completed=True)

    def test_download_manifest_rejects_partial_or_unexpected_files(self) -> None:
        download_dir = self.scratch / "download"
        dispatch, manifest = synthetic_download_manifest(download_dir)
        self.assertEqual(
            len(
                cohort.validate_download_manifest(
                    manifest, dispatch, download_dir
                )
            ),
            40,
        )
        (download_dir / "unexpected.txt").write_text("unexpected", encoding="UTF-8")
        with self.assertRaisesRegex(HarnessError, "unexpected files"):
            cohort.validate_download_manifest(manifest, dispatch, download_dir)

    def test_download_manifest_rejects_zip_not_bound_to_dispatch(self) -> None:
        download_dir = self.scratch / "download"
        dispatch, manifest = synthetic_download_manifest(download_dir)
        manifest["entries"][1]["artifact_zip"]["sha256"] = manifest["entries"][0][
            "artifact_zip"
        ]["sha256"]
        with self.assertRaisesRegex(HarnessError, "differs from dispatch identity"):
            cohort.validate_download_manifest(manifest, dispatch, download_dir)

    def test_late_duplicate_is_rejected_after_initial_discovery(self) -> None:
        dispatch = completed_dispatch()
        selected = {"databaseId": dispatch["runs"][0]["run_id"]}
        duplicate = {"databaseId": 99_999}
        with mock.patch.object(
            cohort,
            "matching_runs",
            side_effect=[[selected], [selected, duplicate]],
        ):
            self.assertEqual(
                cohort.find_run(dispatch, 1, 1)["databaseId"],
                selected["databaseId"],
            )
            with self.assertRaisesRegex(HarnessError, "exactly one"):
                cohort.require_recorded_run_unique(
                    dispatch,
                    dispatch["runs"][0],
                    1,
                )

    def test_matching_runs_requires_exact_tag_ref(self) -> None:
        dispatch = completed_dispatch()
        title = (
            f"WASI thread duration-cross {dispatch['cohort_id']}-1-training"
        )
        correct = {
            "databaseId": dispatch["runs"][0]["run_id"],
            "displayTitle": title,
            "headSha": SHA,
            "headBranch": "wasi-thread-duration-cross-v21",
        }
        wrong_ref = {**correct, "databaseId": 99_999, "headBranch": "main"}
        with mock.patch.object(
            cohort,
            "gh_json",
            return_value=[wrong_ref, correct],
        ) as gh:
            self.assertEqual(cohort.matching_runs(dispatch, 1, 1), [correct])
        arguments = gh.call_args.args[0]
        self.assertEqual(
            arguments[arguments.index("--branch") + 1],
            "wasi-thread-duration-cross-v21",
        )

    def test_duplicate_is_rejected_during_dispatch_finalization(self) -> None:
        dispatch = completed_dispatch()
        with mock.patch.object(
            cohort,
            "matching_runs",
            return_value=[
                {"databaseId": dispatch["runs"][0]["run_id"]},
                {"databaseId": 99_999},
            ],
        ):
            with self.assertRaisesRegex(HarnessError, "exactly one"):
                cohort.finalize_dispatch_state(dispatch, 1)

    def test_duplicate_is_rejected_before_download(self) -> None:
        dispatch = completed_dispatch()
        dispatch_path = self.scratch / "dispatch.json"
        dispatch_path.write_text(json.dumps(dispatch), encoding="UTF-8")
        with mock.patch.object(
            cohort,
            "matching_runs",
            return_value=[
                {"databaseId": dispatch["runs"][0]["run_id"]},
                {"databaseId": 99_999},
            ],
        ):
            with self.assertRaisesRegex(HarnessError, "exactly one"):
                cohort.download_artifacts(
                    dispatch_path,
                    self.scratch / "download",
                    self.scratch / "manifest.json",
                    1,
                )

    def test_duplicate_is_rejected_before_analysis(self) -> None:
        dispatch = json.loads(
            self.passing_fixture["dispatch"].read_text(encoding="UTF-8")
        )
        with mock.patch.object(
            cohort,
            "matching_runs",
            return_value=[
                {"databaseId": dispatch["runs"][0]["run_id"]},
                {"databaseId": 99_999},
            ],
        ):
            with self.assertRaisesRegex(HarnessError, "exactly one"):
                cohort.analyze(
                    self.passing_fixture["cohort"],
                    self.passing_fixture["input_dir"],
                    self.passing_fixture["dispatch"],
                    self.passing_fixture["manifest"],
                    REPO_ROOT
                    / "tests/benchmarks/wasi-threads/"
                    "derivation-policy.production.json",
                    self.scratch / "conclusion.json",
                    1,
                )

    def test_synthetic_pass_accepts_all_doubled_gates(self) -> None:
        output = self.scratch / "conclusion.json"
        self.assertEqual(
            self.analyze_fixture(self.passing_fixture, output),
            0,
        )
        conclusion = json.loads(output.read_text(encoding="UTF-8"))
        self.assertTrue(conclusion["passed"])
        self.assertTrue(output.with_suffix(".md").is_file())
        self.assertIsNone(conclusion["production_budget"])
        for platform in duration.PLATFORM_CELLS:
            doubled = conclusion["evidence"][platform]["doubled"]
            current = conclusion["evidence"][platform]["current"]
            self.assertTrue(
                all(item["selected"] for item in doubled["comparisons"])
            )
            self.assertEqual(
                len(doubled["comparisons"])
                + len(doubled["ratio_of_ratios"]),
                len(duration.acceptance_checks_for_platform(platform)),
            )
            self.assertTrue(
                all(not item["selected"] for item in current["comparisons"])
            )
            self.assertTrue(
                all(
                    len(item["holdout"]) == 4
                    and all(result["passed"] for result in item["holdout"])
                    for item in doubled["comparisons"]
                )
            )

    def test_bound_training_and_holdout_failures_are_rejected(self) -> None:
        output = self.scratch / "conclusion.json"
        with self.assertRaises(HarnessError) as captured:
            self.analyze_fixture(self.failing_fixture, output)
        self.assertIn("training final bound", str(captured.exception))
        self.assertIn("holdout sequence 17 failed", str(captured.exception))
        conclusion = json.loads(output.read_text(encoding="UTF-8"))
        self.assertFalse(conclusion["passed"])
        self.assertTrue(output.with_suffix(".md").is_file())

    def test_nonproduction_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(HarnessError, "unchanged production"):
            self.analyze_fixture(
                self.passing_fixture,
                self.scratch / "conclusion.json",
                policy="derivation-policy.synthetic.json",
            )

    def test_fabricated_cohort_summary_is_rejected(self) -> None:
        path = self.passing_fixture["cohort"]
        original = path.read_bytes()
        markdown = path.with_suffix(".md")
        original_markdown = markdown.read_bytes()
        try:
            document = json.loads(original)
            document["observations"][0]["summaries"]["comparisons"][0][
                "throughput_candidate_over_baseline"
            ] = metric_stats(2.0)
            path.write_text(json.dumps(document), encoding="UTF-8")
            markdown.write_text(
                cohort.render_cohort_markdown(document) + "\n",
                encoding="UTF-8",
            )
            with self.assertRaisesRegex(HarnessError, "recomputed summaries"):
                self.analyze_fixture(
                    self.passing_fixture,
                    self.scratch / "conclusion.json",
                )
        finally:
            path.write_bytes(original)
            markdown.write_bytes(original_markdown)

    def test_changed_report_bytes_are_rejected(self) -> None:
        report = next(self.passing_fixture["input_dir"].glob("*/report.json"))
        original = report.read_bytes()
        try:
            report.write_bytes(original + b" ")
            with self.assertRaisesRegex(HarnessError, "hash mismatch"):
                self.analyze_fixture(
                    self.passing_fixture,
                    self.scratch / "conclusion.json",
                )
        finally:
            report.write_bytes(original)

    def test_changed_report_markdown_is_rejected(self) -> None:
        markdown = next(self.passing_fixture["input_dir"].glob("*/report.md"))
        original = markdown.read_bytes()
        try:
            markdown.write_bytes(original + b"changed\n")
            with self.assertRaisesRegex(HarnessError, "hash mismatch"):
                self.analyze_fixture(
                    self.passing_fixture,
                    self.scratch / "conclusion.json",
                )
        finally:
            markdown.write_bytes(original)

    def test_missing_artifact_zip_is_rejected(self) -> None:
        artifact_zip = next(
            self.passing_fixture["input_dir"].glob("*/artifact.zip")
        )
        original = artifact_zip.read_bytes()
        try:
            artifact_zip.unlink()
            with self.assertRaisesRegex(HarnessError, "ZIP.*missing"):
                self.analyze_fixture(
                    self.passing_fixture,
                    self.scratch / "conclusion.json",
                )
        finally:
            artifact_zip.write_bytes(original)

    def test_cohort_rejects_duplicate_and_partial_membership(self) -> None:
        document = synthetic_cohort()
        duplicate = copy.deepcopy(document)
        duplicate["observations"][-1] = copy.deepcopy(duplicate["observations"][0])
        with self.assertRaisesRegex(HarnessError, "duplicate"):
            cohort.validate_cohort_document(duplicate)
        partial = copy.deepcopy(document)
        partial["observations"].pop()
        with self.assertRaisesRegex(HarnessError, "observations"):
            cohort.validate_cohort_document(partial)
        unexpected = synthetic_cohort()
        unexpected["observations"][0]["platform"] = "unexpected-platform"
        with self.assertRaisesRegex(HarnessError, "membership"):
            cohort.validate_cohort_document(unexpected)

    def test_workflow_is_manual_minimal_and_path_bounded(self) -> None:
        workflow = (
            REPO_ROOT / ".github/workflows/wasi-thread-duration-cross.yml"
        ).read_text(encoding="UTF-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn("group: wasi-thread-duration-cross-diagnostic", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("needs: x86", workflow)
        self.assertIn("runs-on: wamr-temp-20260906", workflow)
        self.assertIn("Neoverse-N2", workflow)
        self.assertNotIn("mlugg/setup-zig", workflow)
        self.assertIn("zig-x86_64-linux-0.16.0.tar.xz", workflow)
        self.assertIn("zig-aarch64-linux-0.16.0.tar.xz", workflow)
        self.assertIn("${{ github.run_attempt }}", workflow)
        self.assertIn("/d/wamr-duration-cross/", workflow)
        self.assertNotIn("~/.cache", workflow)
        self.assertNotIn("TMPDIR: /tmp", workflow)
        self.assertNotIn("/var/tmp", workflow)


if __name__ == "__main__":
    unittest.main()
