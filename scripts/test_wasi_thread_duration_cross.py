#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
import unittest
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_schema import BenchmarkDataError, cache_key  # noqa: E402
from bench_wasi_threads import HarnessError, paired_invocation_order  # noqa: E402
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
            "cpu": {"user": 1, "steal": 0},
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
    plan["pilots"] = [{"pilot_index": index} for index in range(len(plan["cells"]) * 2)]
    plan["runtime_admission"] = {
        "projected_host_wall_ns": 1,
        "limit_ns": duration.DIAGNOSTIC_JOB_LIMIT_NS,
    }
    counts = {}
    for cell in plan["cells"]:
        counts[cell["pair_key"]] = {
            cell["left"]: 100,
            cell["right"]: 100,
        }
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
                            )
                        )
    artifact = {"runtime/enabled-aot": DIGEST, "fixture/threaded": DIGEST}
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
            "cpu_placement": {},
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
            "cpu": None,
            "interval_seconds": 1,
            "reason": "no CPU outside benchmark assignments",
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
) -> dict:
    elapsed = 1_000_000_000
    throughput = 1000.0
    if revision == "candidate":
        elapsed = round(elapsed * ratio)
        throughput *= ratio
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
        "iterations": counts[cell["pair_key"]][condition]
        * (2 if arm == "doubled" else 1),
        "elapsed_ns": elapsed,
        "throughput_ops_per_second": throughput,
        "metric_kind": (
            "guest-lifecycle"
            if cell["workload"] == "spawn-join"
            else "guest-throughput"
        ),
        "correct": True,
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
                        "metric_kind": "guest-throughput",
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
        }
        for sequence in range(1, 21)
    ]
    return result


def synthetic_cohort(value: float = 1.0) -> dict:
    observations = []
    for sequence in range(1, 21):
        for platform in duration.PLATFORM_CELLS:
            observations.append(
                {
                    "report_sha256": DIGEST,
                    "run_id": str(10_000 + sequence),
                    "sequence": sequence,
                    "partition": duration.PARTITIONS[sequence],
                    "platform": platform,
                    "cpu_class": (
                        cohort.X86_CPU_CLASS
                        if platform == "ubuntu-22.04-x86_64"
                        else "Arm Neoverse-N2"
                    ),
                    "host_fingerprint_sha256": DIGEST,
                    "plan_sha256": DIGEST,
                    "plan_identity_sha256": DIGEST,
                    "telemetry_sidecar": {"available": False, "reason": "unavailable"},
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
        "identity": {
            "source_revision": {
                "commit": SHA,
                "tracked_diff_sha256": DIGEST,
                "build_source_sha256": DIGEST,
            },
            "platforms": {},
        },
        "platform_counts": {platform: 20 for platform in duration.PLATFORM_CELLS},
        "partition_counts": {
            platform: {"training": 16, "holdout": 4}
            for platform in duration.PLATFORM_CELLS
        },
        "excluded_observations": [],
        "observations": observations,
    }


class DurationCrossReportTests(unittest.TestCase):
    def test_exact_cells_targets_and_identity_are_isolated(self) -> None:
        self.assertEqual(len(duration.X86_CELLS), 11)
        self.assertEqual(len(duration.ARM_CELLS), 3)
        hot = next(
            cell
            for cell in duration.X86_CELLS
            if cell["pair_key"] == "cancel-points/hot/8"
        )
        self.assertEqual(
            hot["current_target_seconds"],
            {"cancel-points-off": 20.0, "cancel-points-on": 20.0},
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
        self.assertIsNone(report["production_budget"])

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
        with self.assertRaisesRegex(BenchmarkDataError, "availability"):
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
        with self.assertRaisesRegex(BenchmarkDataError, "count admission"):
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


class DurationCrossCohortTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = (
            REPO_ROOT / "zig-out" / f"duration-cross-unit-{uuid.uuid4().hex}"
        )
        self.scratch.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

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

    def test_synthetic_pass_accepts_all_doubled_gates(self) -> None:
        cohort_document = synthetic_cohort()
        cohort_path = self.scratch / "cohort.json"
        output = self.scratch / "conclusion.json"
        cohort_path.write_text(json.dumps(cohort_document), encoding="UTF-8")
        policy = (
            REPO_ROOT
            / "tests/benchmarks/wasi-threads/derivation-policy.synthetic.json"
        )
        self.assertEqual(cohort.analyze(cohort_path, policy, output), 0)
        conclusion = json.loads(output.read_text(encoding="UTF-8"))
        self.assertTrue(conclusion["passed"])
        self.assertIsNone(conclusion["production_budget"])
        for platform in duration.PLATFORM_CELLS:
            doubled = conclusion["evidence"][platform]["doubled"]
            current = conclusion["evidence"][platform]["current"]
            self.assertTrue(
                all(item["selected"] for item in doubled["comparisons"])
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

    def test_training_ceiling_failure_is_rejected(self) -> None:
        document = synthetic_cohort()
        for observation in document["observations"]:
            if (
                observation["platform"] == "ubuntu-22.04-x86_64"
                and observation["sequence"] == 1
            ):
                item = next(
                    metric
                    for metric in observation["summaries"]["comparisons"]
                    if metric["arm"] == "doubled"
                )
                item["throughput_candidate_over_baseline"] = metric_stats(
                    math.exp(-0.20)
                )
                break
        cohort_path = self.scratch / "cohort.json"
        cohort_path.write_text(json.dumps(document), encoding="UTF-8")
        output = self.scratch / "conclusion.json"
        with self.assertRaisesRegex(HarnessError, "training final bound"):
            cohort.analyze(
                cohort_path,
                REPO_ROOT
                / "tests/benchmarks/wasi-threads/derivation-policy.synthetic.json",
                output,
            )
        conclusion = json.loads(output.read_text(encoding="UTF-8"))
        self.assertFalse(conclusion["passed"])

    def test_holdout_failure_is_rejected(self) -> None:
        document = synthetic_cohort()
        for observation in document["observations"]:
            if (
                observation["platform"] == "ubuntu-24.04-aarch64"
                and observation["sequence"] == 17
            ):
                item = next(
                    metric
                    for metric in observation["summaries"]["ratio_of_ratios"]
                    if metric["arm"] == "doubled"
                )
                item["throughput_ratio_of_ratios"] = metric_stats(0.8)
                break
        cohort_path = self.scratch / "cohort.json"
        cohort_path.write_text(json.dumps(document), encoding="UTF-8")
        with self.assertRaisesRegex(HarnessError, "holdout sequence 17 failed"):
            cohort.analyze(
                cohort_path,
                REPO_ROOT
                / "tests/benchmarks/wasi-threads/derivation-policy.synthetic.json",
                self.scratch / "conclusion.json",
            )

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
        self.assertIn("runs-on: wamr-temp-20260906", workflow)
        self.assertIn("Neoverse-N2", workflow)
        self.assertIn("/d/wamr-duration-cross/", workflow)
        self.assertNotIn("~/.cache", workflow)
        self.assertNotIn("/tmp/", workflow)


if __name__ == "__main__":
    unittest.main()
