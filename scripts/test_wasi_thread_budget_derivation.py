#!/usr/bin/env python3

from __future__ import annotations

import copy
import io
import json
import math
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_wasi_threads as bench  # noqa: E402
import wasi_thread_cohort as cohort  # noqa: E402
from bench_wasi_threads import HarnessError  # noqa: E402
from benchmark_schema import cache_key  # noqa: E402


POLICY_PATH = (
    ROOT
    / "tests"
    / "benchmarks"
    / "wasi-threads"
    / "derivation-policy.synthetic.json"
)


def synthetic_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="UTF-8"))


def attach_synthetic_sizing(plan: dict) -> None:
    pilot_iterations = copy.deepcopy(plan["iterations"])
    order = bench.pilot_order_for_plan(
        plan["pairs"], tuple(plan["revision_roles"]), pilot_iterations
    )
    pilots = []
    for spec in order:
        elapsed = 1_500_000_000 - spec["pilot_index"] * 1_000
        overhead = 100_000
        workload = "hot" if spec["workload"] == "cancel-hot" else spec["workload"]
        pilots.append(
            {
                **spec,
                "phase": "pilot",
                "correct": True,
                "operations": bench.expected_result(
                    workload, spec["threads"], spec["iterations"]
                )["operations"],
                "guest_elapsed_ns": elapsed,
                "elapsed_ns": elapsed,
                "raw_guest_elapsed_ns": elapsed + overhead,
                "timing_overhead_ns": overhead,
                "timing_overhead_ppm": (
                    overhead * 1_000_000 // (elapsed + overhead)
                ),
                "host_wall_elapsed_ns": elapsed + 1_000_000,
            }
        )
    selected, resolved = bench.resolve_one_shot_sizing(
        pilot_records=pilots,
        pilot_order=order,
        modes=tuple(plan["modes"]),
        thread_counts=tuple(plan["thread_counts"]),
        warmups=plan["warmups"],
        samples=plan["samples"],
        timeout_seconds=plan["timeout_seconds"],
    )
    plan["iterations"] = selected
    plan["sizing"] = {
        "algorithm": bench.sizing_algorithm_spec(plan["timeout_seconds"]),
        "pilot_iterations": pilot_iterations,
        "pilot_order": order,
        "resolved": resolved,
    }


def synthetic_cohort() -> dict:
    """Build an explicitly synthetic 20-run calibration cohort."""

    plan = {
        "profile": "authoritative",
        "warmups": 2,
        "samples": 10,
        "revision_mode": "paired-revisions",
        "comparison_purpose": "noise-calibration",
        "revision_roles": ["baseline", "candidate"],
        "modes": ["aot"],
        "thread_counts": [1],
        "iterations": {
            "aot": {
                "single-hot": 100,
                "hot": {"1": 100},
                "atomic": {"1": 100},
                "wait-notify": {"1": 100},
                "spawn-join": {"1": 10},
                "cancel-hot": {"1": 100},
            }
        },
        "timeout_seconds": 60,
        "minimum_timed_interval_ns": 1,
        "atomic_wait_preflight_runs": 64,
        "atomic_wait_preflight_iterations": 1_000_000,
        "atomic_wait_preflight_timing_quality": "correctness-only",
        "optimize": "ReleaseFast",
        "pairs": [],
    }
    plan["pairs"] = bench.expected_pair_specs_for_plan(plan)
    attach_synthetic_sizing(plan)
    plan_sha256 = cache_key(plan)
    measurement_plan_sha256 = bench.measurement_plan_sha256(plan)
    run_ids = [str(10_000 + index) for index in range(20)]
    assignments = [
        {
            "sequence": index + 1,
            "partition": "training" if index < 10 else "holdout",
        }
        for index in range(20)
    ]
    revision = {
        "commit": "a" * 40,
        "tracked_diff_sha256": "b" * 64,
        "build_source_sha256": "c" * 64,
    }
    identity = {
        "baseline": dict(revision),
        "candidate": dict(revision),
        "fixture_set_sha256": "d" * 64,
        "plan_sha256_distribution": {plan_sha256: 40},
        "measurement_plan_version": bench.MEASUREMENT_PLAN_IDENTITY_VERSION,
        "measurement_plan_sha256": measurement_plan_sha256,
        "profile": "authoritative",
        "comparison_purpose": "noise-calibration",
        "warmups": 2,
        "samples": 10,
    }
    observations = []
    for index, run_id in enumerate(run_ids):
        partition = assignments[index]["partition"]
        # Small deterministic synthetic noise; holdout remains untouched.
        offset = ((index % 5) - 2) * 0.0005
        for platform_index, platform in enumerate(cohort.DEFAULT_PLATFORMS):
            platform_offset = platform_index * 0.0001
            throughput = 1.0 + offset - platform_offset
            elapsed = 1.0 - offset + platform_offset
            ratio_throughput = 1.0 + offset / 2
            ratio_elapsed = 1.0 - offset / 2
            report_identity = {
                "baseline": {
                    **revision,
                    "fixture_set_sha256": "d" * 64,
                    "plan_sha256": plan_sha256,
                },
                "candidate": {
                    **revision,
                    "fixture_set_sha256": "d" * 64,
                    "plan_sha256": plan_sha256,
                },
                "fixture_set_sha256": "d" * 64,
                "plan_sha256": plan_sha256,
                "measurement_plan_version": (
                    bench.MEASUREMENT_PLAN_IDENTITY_VERSION
                ),
                "measurement_plan_sha256": measurement_plan_sha256,
                "profile": "authoritative",
                "comparison_purpose": "noise-calibration",
            }
            observations.append(
                {
                    "sequence": index + 1,
                    "partition": partition,
                    "run_id": run_id,
                    "platform": platform,
                    "path": f"synthetic/{platform}/{run_id}/report.json",
                    "host_pair_id": f"synthetic-{platform}-{run_id}",
                    "host_fingerprint_sha256": (
                        str(platform_index + 1) * 64
                    ),
                    "records": 1,
                    "report_sha256": cache_key(
                        {"synthetic": True, "platform": platform, "run_id": run_id}
                    ),
                    "identity": report_identity,
                    "plan": copy.deepcopy(plan),
                    "metrics": {
                        "comparisons": [
                            {
                                "pair_kind": "single-infrastructure",
                                "pair_key": "single-infrastructure/aot",
                                "condition": "threads-disabled",
                                "metric_kind": "steady-state-kernel",
                                "elapsed_candidate_over_baseline": [
                                    elapsed,
                                    elapsed,
                                ],
                                "throughput_candidate_over_baseline": [
                                    throughput,
                                    throughput,
                                ],
                            },
                            {
                                "pair_kind": "single-infrastructure",
                                "pair_key": "single-infrastructure/aot",
                                "condition": "threads-enabled",
                                "metric_kind": "steady-state-kernel",
                                "elapsed_candidate_over_baseline": [
                                    elapsed,
                                    elapsed,
                                ],
                                "throughput_candidate_over_baseline": [
                                    throughput,
                                    throughput,
                                ],
                            },
                        ],
                        "ratio_of_ratios": [
                            {
                                "pair_kind": "single-infrastructure",
                                "pair_key": "single-infrastructure/aot",
                                "left": "threads-disabled",
                                "right": "threads-enabled",
                                "elapsed_ratio_of_ratios": [
                                    ratio_elapsed,
                                    ratio_elapsed,
                                ],
                                "throughput_ratio_of_ratios": [
                                    ratio_throughput,
                                    ratio_throughput,
                                ],
                            }
                        ],
                        "direct_candidate_single_infrastructure": [
                            {
                                "pair_kind": "single-infrastructure",
                                "pair_key": "single-infrastructure/aot",
                                "left": "threads-disabled",
                                "right": "threads-enabled",
                                "elapsed_right_over_left": [0.995, 0.995],
                                "throughput_right_over_left": [1.005, 1.005],
                            }
                        ],
                    },
                }
            )
    platforms = {
        platform: {
            "reports": 20,
            "run_ids": list(run_ids),
            "host_fingerprint_distribution": {
                str(index + 1) * 64: 20
            },
            "cpu_distribution": {
                f"synthetic-{platform}-cpu": 20
            },
            "runner_image_distribution": {
                f"synthetic-{platform}-image": 20
            },
        }
        for index, platform in enumerate(cohort.DEFAULT_PLATFORMS)
    }
    return {
        "schema_version": bench.REPORT_SCHEMA_VERSION,
        "kind": "wasi-thread-paired-cohort",
        "authoritative": True,
        "validated_at": "synthetic-time",
        "dispatch": {
            "repository": "synthetic/example",
            "workflow": "synthetic.yml",
            "workflow_ref": "synthetic-ref",
            "workflow_head_sha": "f" * 40,
            "cohort_id": "1" * 32,
            "runner_target": "github-hosted",
            "requested_runs": 20,
            "requested_reports": 40,
        },
        "identity": identity,
        "split": {
            "method": "predeclared-sequence",
            "training_runs": 10,
            "holdout_runs": 10,
            "assignments": assignments,
            "run_ids": {
                "training": run_ids[:10],
                "holdout": run_ids[10:],
            },
        },
        "platforms": platforms,
        "observations": observations,
        "excluded_observations": [],
    }


class BudgetDerivationTests(unittest.TestCase):
    scratch = ROOT / "zig-out" / "test-wasi-thread-budget-derivation"

    def setUp(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def test_directionality_and_zero_mad(self) -> None:
        lower = cohort.derive_one_sided_threshold(
            [0.99, 0.99],
            "lower",
            0.001,
            0.5,
            "synthetic lower",
        )
        upper = cohort.derive_one_sided_threshold(
            [1.01, 1.01],
            "upper",
            0.001,
            0.5,
            "synthetic upper",
        )
        self.assertLess(lower["threshold_ratio"], 0.99)
        self.assertGreater(upper["threshold_ratio"], 1.01)
        self.assertEqual(lower["mad_log_ratio"], 0.0)
        self.assertEqual(upper["mad_log_ratio"], 0.0)

    def test_selects_worst_or_mad_whichever_is_more_permissive(self) -> None:
        worst = cohort.derive_one_sided_threshold(
            [1.0] * 10 + [0.8],
            "lower",
            0.0,
            1.0,
            "synthetic worst",
        )
        robust = cohort.derive_one_sided_threshold(
            [math.exp(-0.1), 1.0, math.exp(0.1)],
            "lower",
            0.0,
            1.0,
            "synthetic robust",
        )
        self.assertEqual(worst["selected_source"], "worst-observed")
        self.assertEqual(
            robust["selected_source"], "median-plus-six-scaled-mad"
        )
        self.assertGreater(
            robust["robust_deviation_log"],
            robust["worst_observed_deviation_log"],
        )
        self.assertEqual(worst["dropped_observations"], [])

    def test_ceiling_breach_fails(self) -> None:
        with self.assertRaisesRegex(HarnessError, "exceeds engineering policy"):
            cohort.derive_one_sided_threshold(
                [0.5],
                "lower",
                0.01,
                0.1,
                "synthetic breach",
            )

    def test_non_finite_and_non_positive_ratios_fail(self) -> None:
        for invalid in (0.0, -1.0, math.nan, math.inf):
            value = synthetic_cohort()
            value["observations"][0]["metrics"]["comparisons"][0][
                "throughput_candidate_over_baseline"
            ] = [invalid]
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                HarnessError, "positive number"
            ):
                cohort.derive_budget_documents(value, synthetic_policy())

    def test_holdout_failure_refuses_derivation(self) -> None:
        value = synthetic_cohort()
        holdout = next(
            item
            for item in value["observations"]
            if item["partition"] == "holdout"
        )
        holdout["metrics"]["comparisons"][0][
            "throughput_candidate_over_baseline"
        ] = [0.5, 0.5]
        with self.assertRaisesRegex(HarnessError, "untouched holdout"):
            cohort.derive_budget_documents(value, synthetic_policy())

    def test_split_overlap_and_membership_changes_fail(self) -> None:
        overlap = synthetic_cohort()
        overlap["split"]["run_ids"]["holdout"][0] = overlap["split"]["run_ids"][
            "training"
        ][0]
        with self.assertRaisesRegex(HarnessError, "overlap"):
            cohort.derive_budget_documents(overlap, synthetic_policy())

        changed = synthetic_cohort()
        changed["observations"][0]["partition"] = "holdout"
        with self.assertRaisesRegex(HarnessError, "changed predeclared membership"):
            cohort.derive_budget_documents(changed, synthetic_policy())

    def test_rejects_wrong_shape_identity_purpose_and_metrics(self) -> None:
        cases = []
        smoke = synthetic_cohort()
        smoke["identity"]["profile"] = "smoke"
        cases.append((smoke, "smoke"))
        purpose = synthetic_cohort()
        purpose["identity"]["comparison_purpose"] = "candidate-evaluation"
        cases.append((purpose, "noise-calibration purpose"))
        mixed = synthetic_cohort()
        mixed["identity"]["candidate"]["commit"] = "9" * 40
        cases.append((mixed, "mixed revision identity"))
        partial = synthetic_cohort()
        partial["observations"].pop()
        cases.append((partial, "observation count"))
        legacy = synthetic_cohort()
        legacy["kind"] = "wasi-thread-cohort-legacy-compatibility"
        cases.append((legacy, "authoritative"))
        excluded = synthetic_cohort()
        excluded["excluded_observations"] = ["synthetic"]
        cases.append((excluded, "forbids excluded"))
        missing = synthetic_cohort()
        del missing["observations"][0]["metrics"]
        cases.append((missing, "missing raw per-report metrics"))
        for value, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                HarnessError, message
            ):
                cohort.derive_budget_documents(value, synthetic_policy())

    def test_direct_candidate_delta_is_required_and_strict(self) -> None:
        failed = synthetic_cohort()
        failed["observations"][0]["metrics"][
            "direct_candidate_single_infrastructure"
        ][0]["throughput_right_over_left"] = [1.02, 1.02]
        with self.assertRaisesRegex(
            HarnessError, "direct candidate single-infrastructure policy failed"
        ):
            cohort.derive_budget_documents(failed, synthetic_policy())

        missing = synthetic_cohort()
        missing["observations"][0]["metrics"][
            "direct_candidate_single_infrastructure"
        ] = []
        with self.assertRaisesRegex(HarnessError, "missing raw per-report metrics"):
            cohort.derive_budget_documents(missing, synthetic_policy())

    def test_deterministic_serialization_and_threshold_completeness(self) -> None:
        first = cohort.derive_budget_documents(
            synthetic_cohort(), synthetic_policy()
        )
        second = cohort.derive_budget_documents(
            synthetic_cohort(), synthetic_policy()
        )
        for left, right in zip(first[:2], second[:2]):
            self.assertEqual(
                json.dumps(left, sort_keys=True, separators=(",", ":")),
                json.dumps(right, sort_keys=True, separators=(",", ":")),
            )
        self.assertEqual(first[2], second[2])
        budget, evidence, _ = first
        self.assertTrue(budget["calibrated"])
        self.assertFalse(budget["enforcement"])
        self.assertEqual(evidence["holdout_status"], "passed")
        for platform in cohort.DEFAULT_PLATFORMS:
            thresholds = budget["platforms"][platform]
            self.assertEqual(len(thresholds["comparisons"]), 2)
            self.assertEqual(len(thresholds["ratio_of_ratios"]), 1)
            self.assertEqual(
                {
                    (item["pair_key"], item["condition"])
                    for item in thresholds["comparisons"]
                },
                {
                    ("single-infrastructure/aot", "threads-disabled"),
                    ("single-infrastructure/aot", "threads-enabled"),
                },
            )

    def test_budget_and_policy_schema_compatibility(self) -> None:
        budget, _, _ = cohort.derive_budget_documents(
            synthetic_cohort(), synthetic_policy()
        )
        directory = POLICY_PATH.parent
        budget_schema = json.loads(
            (directory / "budget.schema.json").read_text(encoding="UTF-8")
        )
        policy_schema = json.loads(
            (directory / "derivation-policy.schema.json").read_text(
                encoding="UTF-8"
            )
        )
        self.assertEqual(set(budget), set(budget_schema["required"]))
        self.assertEqual(
            set(synthetic_policy()), set(policy_schema["required"])
        )
        calibrated_then = budget_schema["allOf"][1]["then"]["properties"]
        self.assertNotIn("enforcement", calibrated_then)
        self.assertFalse(budget["enforcement"])
        for platform in cohort.DEFAULT_PLATFORMS:
            platform_budget = budget["platforms"][platform]
            self.assertEqual(
                set(platform_budget),
                set(budget_schema["$defs"]["platform"]["required"]),
            )

    def test_derive_cli_is_deterministic_and_fails_closed(self) -> None:
        cohort_path = self.scratch / "cohort.json"
        policy_path = self.scratch / "policy.json"
        cohort_path.write_text(
            json.dumps(synthetic_cohort(), sort_keys=True),
            encoding="UTF-8",
        )
        policy_path.write_text(
            json.dumps(synthetic_policy(), sort_keys=True),
            encoding="UTF-8",
        )

        outputs = []
        for suffix in ("one", "two"):
            budget = self.scratch / f"budget-{suffix}.json"
            evidence = self.scratch / f"evidence-{suffix}.json"
            markdown = self.scratch / f"evidence-{suffix}.md"
            self.assertEqual(
                cohort.main(
                    [
                        "derive",
                        "--cohort",
                        str(cohort_path),
                        "--policy",
                        str(policy_path),
                        "--budget-output",
                        str(budget),
                        "--evidence-json-output",
                        str(evidence),
                        "--evidence-markdown-output",
                        str(markdown),
                    ]
                ),
                0,
            )
            outputs.append(
                (
                    budget.read_bytes(),
                    evidence.read_bytes(),
                    markdown.read_bytes(),
                )
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertTrue(outputs[0][0].endswith(b"\n"))
        self.assertTrue(outputs[0][1].endswith(b"\n"))
        self.assertIn(b"Enforcement: **disabled**", outputs[0][2])

        invalid_policy = synthetic_policy()
        invalid_policy["engineering_policy_ceiling_log"][
            "comparison_throughput_lower"
        ] = 0.0001
        policy_path.write_text(
            json.dumps(invalid_policy, sort_keys=True),
            encoding="UTF-8",
        )
        failed_budget = self.scratch / "failed-budget.json"
        with mock.patch.object(cohort.sys, "stderr", io.StringIO()) as stderr:
            self.assertEqual(
                cohort.main(
                    [
                        "derive",
                        "--cohort",
                        str(cohort_path),
                        "--policy",
                        str(policy_path),
                        "--budget-output",
                        str(failed_budget),
                        "--evidence-json-output",
                        str(self.scratch / "failed-evidence.json"),
                        "--evidence-markdown-output",
                        str(self.scratch / "failed-evidence.md"),
                    ]
                ),
                2,
            )
        self.assertIn("rounding cushion exceeds", stderr.getvalue())
        self.assertFalse(failed_budget.exists())


if __name__ == "__main__":
    unittest.main()
