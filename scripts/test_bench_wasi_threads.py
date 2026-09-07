#!/usr/bin/env python3

from __future__ import annotations

import copy
import io
import json
import shutil
import struct
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_wasi_threads as bench  # noqa: E402
import benchmark_schema as schema  # noqa: E402
import wasi_thread_cohort as cohort  # noqa: E402
from benchmark_schema import (  # noqa: E402
    BenchmarkDataError,
    SCHEMA_VERSION,
    alternating_pair_order,
    cache_key,
)


def guest_result(
    workload: str = "atomic",
    threads: int = 1,
    iterations: int = 10,
    elapsed_ns: int = 200_000_000,
    overhead_ns: int = 100_000,
) -> dict:
    result = dict(bench.expected_result(workload, threads, iterations))
    raw = elapsed_ns + overhead_ns
    result.update(
        {
            "raw_elapsed_ns": raw,
            "timing_overhead_ns": overhead_ns,
            "elapsed_ns": elapsed_ns,
            "timing_overhead_ppm": overhead_ns * 1_000_000 // raw,
        }
    )
    return result


def stats(value: float, key: str) -> dict:
    return {
        key: [value],
        "runs": 1,
        "mean": value,
        "median": value,
        "min": value,
        "max": value,
        "range": 0,
    }


def make_report(
    platform_id: str = "ubuntu-22.04-x86_64",
    machine: str = "x86_64",
    commit: str = "a" * 40,
    baseline_commit: str | None = None,
    baseline_source: str = "e" * 64,
    candidate_source: str = "c" * 64,
    run_id: str = "1",
    revision_mode: str = "paired-revisions",
    comparison_purpose: str | None = None,
    samples: int | None = None,
) -> dict:
    baseline_commit = baseline_commit or "b" * 40
    if revision_mode == "paired-revisions":
        revision_roles = bench.REVISION_ROLES
        comparison_purpose = comparison_purpose or "candidate-evaluation"
        samples = 2 if samples is None else samples
    else:
        revision_roles = bench.SINGLE_REVISION_ROLES
        comparison_purpose = "single-revision-compatibility"
        samples = 1 if samples is None else samples
    plan = {
        "profile": "authoritative",
        "warmups": 0,
        "samples": samples,
        "revision_mode": revision_mode,
        "comparison_purpose": comparison_purpose,
        "revision_roles": list(revision_roles),
        "modes": ["aot"],
        "thread_counts": [1],
        "iterations": {
            "single-hot": 10,
            "hot": 10,
            "atomic": 10,
            "wait-notify": 10,
            "spawn-join": 10,
        },
        "timeout_seconds": 60,
        "minimum_timed_interval_ns": 1,
        "atomic_wait_preflight_runs": 64,
        "optimize": "ReleaseFast",
        "pairs": [],
    }
    plan["pairs"] = bench.expected_pair_specs_for_plan(plan)
    plan_sha256 = cache_key(plan)
    host_fields = {
        "system": "Linux",
        "machine": machine,
        "cpu": "test cpu",
        "logical_cpus": 4,
        "runner_environment": "github-hosted",
        "runner_image": "ubuntu",
        "runner_os": "Linux",
        "runner_arch": machine,
    }
    host_fingerprint = cache_key(host_fields)
    host_pair_id = f"github:{run_id}:1:{platform_id}"
    all_revisions = {
        "baseline": {
            "commit": baseline_commit,
            "tracked_diff_sha256": "b" * 64,
            "build_source_sha256": baseline_source,
            "fixture_set_sha256": "d" * 64,
            "plan_sha256": plan_sha256,
            "host_pair_id": host_pair_id,
            "host_fingerprint_sha256": host_fingerprint,
        },
        "candidate": {
            "commit": commit,
            "tracked_diff_sha256": "b" * 64,
            "build_source_sha256": candidate_source,
            "fixture_set_sha256": "d" * 64,
            "plan_sha256": plan_sha256,
            "host_pair_id": host_pair_id,
            "host_fingerprint_sha256": host_fingerprint,
        },
    }
    revisions = {role: all_revisions[role] for role in revision_roles}
    revision_fields = {
        role: {
            "revision_commit": revision["commit"],
            "revision_build_source_sha256": revision[
                "build_source_sha256"
            ],
            "fixture_set_sha256": revision["fixture_set_sha256"],
            "plan_sha256": revision["plan_sha256"],
            "host_pair_id": revision["host_pair_id"],
            "host_fingerprint_sha256": revision[
                "host_fingerprint_sha256"
            ],
        }
        for role, revision in revisions.items()
    }
    records = []
    for pair in plan["pairs"]:
        def measure(revision, condition, fields):
            condition_index = (
                0 if condition == pair["left"] else 1
            )
            elapsed = (
                100 + condition_index * 20
                if revision == "baseline"
                else 90 + condition_index * 30
            )
            operations = 1_000
            throughput = operations / (elapsed / 1e9)
            metric_kind = (
                "spawn-join-lifecycle"
                if "spawn-join" in pair["pair_key"]
                else "steady-state-kernel"
            )
            return {
                **fields,
                "mode": "aot",
                "threads_enabled": True,
                "cancel_points": "on",
                "workload": "hot",
                "threads": 1,
                "iterations": 10,
                "command": ["wamr"],
                "elapsed_ns": elapsed,
                "guest_elapsed_ns": elapsed,
                "raw_guest_elapsed_ns": elapsed + 1,
                "timing_overhead_ns": 1,
                "timing_overhead_ppm": 1,
                "host_wall_elapsed_ns": elapsed + 100,
                "host_wall_over_guest": (elapsed + 100) / elapsed,
                "metric_kind": metric_kind,
                "cancel_polls_per_operation": 1.0,
                "static_cancel_poll_sites": 1,
                "operations": operations,
                "throughput_ops_per_second": throughput,
                "per_thread_ops_per_second": throughput,
                "guest": {"metric_kind": metric_kind},
                "correct": True,
                "correctness": {"passed": True},
                "stdout": "{}",
                "stderr": "",
            }

        with mock.patch.object(bench.sys, "stderr", io.StringIO()):
            bench.collect_revision_pair(
                records=records,
                pair_kind=pair["pair_kind"],
                pair_key=pair["pair_key"],
                left=pair["left"],
                right=pair["right"],
                warmups=0,
                samples=samples,
                revision_roles=revision_roles,
                revision_fields=revision_fields,
                measure=measure,
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": bench.KIND,
        "metadata": {
            "commit": commit,
            "tracked_diff_sha256": "b" * 64,
            "build_source_sha256": candidate_source,
            "revisions": revisions,
            "revision_checkouts": (
                {
                    "baseline": "/checkouts/baseline",
                    "candidate": "/checkouts/candidate",
                }
                if revision_mode == "paired-revisions"
                else {"candidate": "/checkouts/candidate"}
            ),
            "collected_at": "2026-09-02T00:00:00+00:00",
            "platform_id": platform_id,
            "fixture_set_sha256": "d" * 64,
            "plan_sha256": plan_sha256,
            "host": {
                "system": "Linux",
                "machine": machine,
                "runner_environment": "github-hosted",
                "github_run_id": run_id,
                "host_fingerprint": {
                    "sha256": host_fingerprint,
                    "fields": host_fields,
                },
            },
            "host_pair": {
                "id": host_pair_id,
                "runner_environment": "github-hosted",
                "host_fingerprint_sha256": host_fingerprint,
            },
            "execution": {},
            "tools": {},
            "fixture_toolchain": {},
            "fixtures": {},
        },
        "plan": plan,
        "records": records,
        "summaries": bench.summarize(records),
        "paired_summaries": bench.paired_summaries(records),
        "comparison_summaries": bench.comparison_summaries(records),
        "ratio_of_ratios_summaries": bench.ratio_of_ratios_summaries(
            records
        ),
        "budget": {"status": "disabled", "path": None, "failures": []},
    }
    bench.validate_report(report)
    return report


def complete_budget(report: dict) -> dict:
    platform_id = report["metadata"]["platform_id"]
    other_platform_id = (
        "ubuntu-24.04-aarch64"
        if platform_id == "ubuntu-22.04-x86_64"
        else "ubuntu-22.04-x86_64"
    )
    other_system, other_machine = bench.CANONICAL_PLATFORMS[other_platform_id]
    platform_budget = {
        "host_system": report["metadata"]["host"]["system"],
        "host_machine": report["metadata"]["host"]["machine"],
        "runner_environment": report["metadata"]["host"][
            "runner_environment"
        ],
        "comparisons": [
            {
                "pair_key": item["pair_key"],
                "condition": item["condition"],
                "metric_kind": item["metric_kind"],
                "min_candidate_over_baseline_throughput_ratio": 0.1,
                "max_candidate_over_baseline_elapsed_ratio": 10.0,
            }
            for item in report["comparison_summaries"]
        ],
        "ratio_of_ratios": [
            {
                "pair_key": item["pair_key"],
                "left": item["left"],
                "right": item["right"],
                "min_candidate_over_baseline_throughput_ratio_of_ratios": 0.1,
                "max_candidate_over_baseline_elapsed_ratio_of_ratios": 10.0,
            }
            for item in report["ratio_of_ratios_summaries"]
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-thread-benchmark-budget",
        "calibrated": True,
        "enforcement": True,
        "calibration_requirements": {
            "minimum_reports_per_platform": 20,
            "required_profile": "authoritative",
            "required_platforms": list(bench.CANONICAL_PLATFORMS),
        },
        "calibration_provenance": {
            "comparison_purpose": "noise-calibration",
            "baseline_revision": {
                "commit": report["metadata"]["revisions"]["baseline"][
                    "commit"
                ],
                "build_source_sha256": report["metadata"]["revisions"][
                    "baseline"
                ]["build_source_sha256"],
            },
            "candidate_revision": {
                "commit": report["metadata"]["revisions"]["baseline"][
                    "commit"
                ],
                "build_source_sha256": report["metadata"]["revisions"][
                    "baseline"
                ]["build_source_sha256"],
            },
            "fixture_set_sha256": report["metadata"]["fixture_set_sha256"],
            "plan_sha256": report["metadata"]["plan_sha256"],
            "profile": report["plan"]["profile"],
            "report_count_by_platform": {
                item: 20 for item in bench.CANONICAL_PLATFORMS
            },
        },
        "platforms": {
            platform_id: platform_budget,
            other_platform_id: {
                **copy.deepcopy(platform_budget),
                "host_system": other_system,
                "host_machine": other_machine,
            },
        },
    }


def make_single_report(**kwargs) -> dict:
    return make_report(
        revision_mode="single-revision-compatibility",
        **kwargs,
    )


class ThreadBenchmarkTests(unittest.TestCase):
    scratch = ROOT / "zig-out" / "test-bench-wasi-threads"

    def setUp(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def write_budget(self, value: dict | str) -> Path:
        path = self.scratch / "budget.json"
        path.write_text(
            value if isinstance(value, str) else json.dumps(value),
            encoding="UTF-8",
        )
        return path

    def test_atomic_json_write_skips_windows_directory_fsync(self) -> None:
        output = self.scratch / "report.json"
        with (
            mock.patch.object(schema.os, "name", "nt"),
            mock.patch.object(schema.os, "open") as open_mock,
        ):
            schema.atomic_write_json(output, {"ok": True})
        open_mock.assert_not_called()
        self.assertEqual(json.loads(output.read_text(encoding="UTF-8")), {"ok": True})

    def test_cli_parsing_and_profiles(self) -> None:
        args = bench.parse_args(
            [
                "--profile",
                "smoke",
                "--thread-counts",
                "1,4,8",
                "--modes",
                "aot",
                "--platform-id",
                "test-x86",
                "--no-budget",
            ]
        )
        self.assertEqual((args.warmups, args.samples), (1, 4))
        self.assertEqual(args.single_iterations, 224_000_000)
        self.assertEqual(args.cancel_iterations, 224_000_000)
        self.assertEqual(args.atomic_total_iterations, 256_000_000)
        self.assertEqual(
            [bench.cancel_iterations(args, threads) for threads in (1, 2, 4, 8)],
            [224_000_000, 128_000_000, 128_000_000, 128_000_000],
        )
        self.assertEqual(
            [bench.atomic_iterations(args, threads) for threads in (1, 2, 4, 8)],
            [256_000_000, 128_000_000, 64_000_000, 64_000_000],
        )
        atomic_scenarios = {
            scenario.threads: scenario.iterations
            for scenario in bench.planned_scenarios(args)
            if scenario.workload == "atomic"
        }
        self.assertEqual(
            atomic_scenarios,
            {1: 256_000_000, 4: 64_000_000, 8: 64_000_000},
        )
        self.assertEqual(bench.ATOMIC_WAIT_PREFLIGHT_RUNS["smoke"], 8)
        self.assertEqual(bench.ATOMIC_WAIT_PREFLIGHT_RUNS["authoritative"], 64)
        self.assertEqual(args.thread_counts, (1, 4, 8))
        self.assertEqual(args.modes, "aot")
        self.assertEqual(args.min_interval_ms, 100)
        self.assertIsNone(args.baseline_repo)
        self.assertIsNone(args.candidate_repo)
        wait_iterations = {
            scenario.threads: scenario.iterations
            for scenario in bench.planned_scenarios(args)
            if scenario.workload == "wait-notify"
        }
        self.assertEqual(wait_iterations, {1: 512_000, 4: 128_000, 8: 64_000})
        with self.assertRaises(SystemExit):
            bench.parse_args(["--thread-counts", "1,3"])
        with self.assertRaises(SystemExit):
            bench.parse_args(
                [
                    "--thread-counts",
                    "1,8",
                    "--wait-iterations",
                    "10",
                ]
            )
        with self.assertRaises(SystemExit):
            bench.parse_args(["--baseline-repo", "baseline"])
        paired = bench.parse_args(
            [
                "--baseline-repo",
                "baseline",
                "--candidate-repo",
                "candidate",
                "--comparison-purpose",
                "candidate-evaluation",
                "--runner-environment",
                "github-hosted",
                "--host-pair-id",
                "run-1/x86",
                "--no-budget",
            ]
        )
        self.assertEqual(paired.baseline_repo, Path("baseline"))
        self.assertEqual(paired.candidate_repo, Path("candidate"))
        with self.assertRaises(SystemExit):
            bench.parse_args(
                [
                    "--baseline-repo",
                    "baseline",
                    "--candidate-repo",
                    "candidate",
                    "--comparison-purpose",
                    "candidate-evaluation",
                    "--samples",
                    "3",
                    "--no-budget",
                ]
            )
        single_odd = bench.parse_args(
            ["--samples", "3", "--no-budget"]
        )
        self.assertEqual(single_odd.samples, 3)
        with self.assertRaisesRegex(
            BenchmarkDataError, "samples must be even"
        ):
            make_report(samples=3)

    def test_pair_direction_never_depends_on_condition_sorting(self) -> None:
        records = []
        observed = []

        def measure(revision, condition, fields):
            observed.append(
                (
                    fields["pair_index"],
                    revision,
                    fields["revision_order"],
                    condition,
                    fields["order"],
                )
            )
            return {
                **fields,
                "elapsed_ns": 10 if condition == "z-baseline" else 20,
                "guest_elapsed_ns": 10 if condition == "z-baseline" else 20,
                "host_wall_elapsed_ns": 30,
                "throughput_ops_per_second": (
                    20 if condition == "z-baseline" else 10
                ),
                "correct": True,
            }

        bench.collect_revision_pair(
            records=records,
            pair_kind="test",
            pair_key="pair",
            left="z-baseline",
            right="a-target",
            warmups=0,
            samples=2,
            revision_roles=bench.REVISION_ROLES,
            revision_fields={"baseline": {}, "candidate": {}},
            measure=measure,
        )
        result = next(
            item
            for item in bench.paired_summaries(records)
            if item["revision"] == "candidate"
        )
        self.assertEqual(result["left"], "z-baseline")
        self.assertEqual(result["right"], "a-target")
        self.assertEqual(result["elapsed_right_over_left"]["median"], 2.0)
        self.assertEqual(
            observed[:4],
            [
                (0, "baseline", 0, "z-baseline", 0),
                (0, "baseline", 0, "a-target", 1),
                (0, "candidate", 1, "z-baseline", 0),
                (0, "candidate", 1, "a-target", 1),
            ],
        )
        self.assertEqual(
            observed[4:],
            [
                (1, "candidate", 0, "a-target", 0),
                (1, "candidate", 0, "z-baseline", 1),
                (1, "baseline", 1, "a-target", 0),
                (1, "baseline", 1, "z-baseline", 1),
            ],
        )
        self.assertEqual(
            alternating_pair_order(1, "left", "right"), ("right", "left")
        )

    def test_measured_revision_positions_balance_after_odd_warmup(self) -> None:
        records = []

        def measure(revision, condition, fields):
            return {
                **fields,
                "elapsed_ns": 10,
                "guest_elapsed_ns": 10,
                "host_wall_elapsed_ns": 20,
                "throughput_ops_per_second": 1.0,
                "correct": True,
            }

        with mock.patch.object(bench.sys, "stderr", io.StringIO()):
            bench.collect_revision_pair(
                records=records,
                pair_kind="test",
                pair_key="balanced",
                left="left",
                right="right",
                warmups=1,
                samples=2,
                revision_roles=bench.REVISION_ROLES,
                revision_fields={"baseline": {}, "candidate": {}},
                measure=measure,
            )
        first_positions = [
            record["revision"]
            for record in records
            if record["phase"] == "measure"
            and record["condition"]
            == alternating_pair_order(
                1 + record["pair_index"], "left", "right"
            )[0]
            and record["revision_order"] == 0
        ]
        self.assertEqual(first_positions, ["candidate", "baseline"])

    def test_single_revision_cli_emits_candidate_only_report(self) -> None:
        output = self.scratch / "compat-report"
        args = bench.parse_args(
            [
                "--repo",
                str(ROOT),
                "--output-dir",
                str(output),
                "--profile",
                "smoke",
                "--warmups",
                "0",
                "--samples",
                "3",
                "--modes",
                "interpreter",
                "--thread-counts",
                "1",
                "--single-iterations",
                "1",
                "--cancel-iterations",
                "1",
                "--hot-iterations",
                "1",
                "--atomic-iterations",
                "1",
                "--atomic-total-iterations",
                "1",
                "--wait-iterations",
                "1",
                "--spawn-iterations",
                "1",
                "--min-interval-ms",
                "0.000001",
                "--no-budget",
            ]
        )

        def fake_build(**kwargs):
            name = (
                f"{'enabled' if kwargs['threads_enabled'] else 'disabled'}-"
                f"{kwargs['mode']}"
            )
            return bench.Build(
                name,
                kwargs["mode"],
                kwargs["threads_enabled"],
                output,
                output / "wamr",
                None,
                name,
                ["zig", "build"],
                False,
            )

        def fake_measure(**kwargs):
            fields = kwargs["record_fields"]
            return {
                **fields,
                "command": ["wamr"],
                "elapsed_ns": 100,
                "guest_elapsed_ns": 100,
                "raw_guest_elapsed_ns": 101,
                "timing_overhead_ns": 1,
                "timing_overhead_ppm": 1,
                "host_wall_elapsed_ns": 200,
                "host_wall_over_guest": 2.0,
                "metric_kind": (
                    "spawn-join-lifecycle"
                    if kwargs["workload"] == "spawn-join"
                    else "steady-state-kernel"
                ),
                "cancel_polls_per_operation": 0.0,
                "operations": 1,
                "throughput_ops_per_second": 10_000_000.0,
                "per_thread_ops_per_second": 10_000_000.0,
                "guest": {},
                "correct": True,
                "correctness": {"passed": True},
                "stdout": "{}",
                "stderr": "",
            }

        with (
            mock.patch.object(bench, "build_variant", side_effect=fake_build),
            mock.patch.object(bench, "measure_once", side_effect=fake_measure),
            mock.patch.object(bench, "build_tool_report", return_value={}),
            mock.patch.object(bench.sys, "stderr", io.StringIO()),
        ):
            report = bench.execute(args)
        self.assertEqual(
            report["plan"]["revision_mode"],
            "single-revision-compatibility",
        )
        self.assertEqual(
            set(report["metadata"]["revisions"]),
            set(bench.SINGLE_REVISION_ROLES),
        )
        self.assertEqual(
            {record["revision"] for record in report["records"]},
            set(bench.SINGLE_REVISION_ROLES),
        )
        self.assertEqual(
            len(report["records"]),
            2 * 3 * len(report["plan"]["pairs"]),
        )
        self.assertEqual(report["comparison_summaries"], [])
        self.assertEqual(report["ratio_of_ratios_summaries"], [])
        self.assertTrue((output / "report.json").is_file())

    def test_guest_timing_parser_rejects_missing_duplicate_and_malformed(self) -> None:
        expected = bench.expected_result("atomic", 1, 10)
        result = guest_result()
        self.assertEqual(
            bench.parse_guest_result(json.dumps(result), expected, 100_000_000),
            result,
        )
        for key in (
            "elapsed_ns",
            "timing_overhead_ns",
            "timing_overhead_ppm",
        ):
            corrupt = dict(result)
            corrupt.pop(key)
            with self.subTest(missing=key), self.assertRaises(bench.HarnessError):
                bench.parse_guest_result(json.dumps(corrupt), expected, 1)
        corrupt = dict(result, elapsed_ns=result["elapsed_ns"] + 1)
        with self.assertRaisesRegex(bench.HarnessError, "must equal"):
            bench.parse_guest_result(json.dumps(corrupt), expected, 1)
        corrupt = dict(
            result,
            raw_elapsed_ns=200_000_000,
            timing_overhead_ns=3_000_000,
            elapsed_ns=197_000_000,
            timing_overhead_ppm=15_000,
        )
        with self.assertRaisesRegex(bench.HarnessError, "below 1%"):
            bench.parse_guest_result(json.dumps(corrupt), expected, 1)
        with self.assertRaisesRegex(bench.HarnessError, "below required"):
            bench.parse_guest_result(json.dumps(result), expected, 300_000_000)
        duplicate_key = json.dumps(result)[:-1] + ',"elapsed_ns":200000000}'
        with self.assertRaisesRegex(bench.HarnessError, "duplicate key"):
            bench.parse_guest_result(duplicate_key, expected, 1)
        with self.assertRaisesRegex(bench.HarnessError, "one guest JSON"):
            bench.parse_guest_result("{}\n{}\n", expected, 1)

    def test_host_startup_delay_does_not_change_guest_throughput(self) -> None:
        build = bench.Build(
            "test", "aot", True, self.scratch, Path("wamr"), None, "key", [], False
        )
        output = json.dumps(guest_result("atomic", 1, 10))
        with mock.patch.object(
            bench, "run_process", return_value=(0, output, "")
        ), mock.patch.object(
            bench.time, "perf_counter_ns", side_effect=[0, 1_000_000_000]
        ):
            first = bench.measure_once(
                repo=ROOT,
                runner=[],
                build=build,
                module=Path("fixture"),
                workload="atomic",
                threads=1,
                iterations=10,
                timeout=1,
                min_interval_ns=1,
                record_fields={"cancel_points": "on"},
            )
        with mock.patch.object(
            bench, "run_process", return_value=(0, output, "")
        ), mock.patch.object(
            bench.time, "perf_counter_ns", side_effect=[0, 9_000_000_000]
        ):
            delayed = bench.measure_once(
                repo=ROOT,
                runner=[],
                build=build,
                module=Path("fixture"),
                workload="atomic",
                threads=1,
                iterations=10,
                timeout=1,
                min_interval_ns=1,
                record_fields={"cancel_points": "on"},
            )
        self.assertEqual(
            first["throughput_ops_per_second"],
            delayed["throughput_ops_per_second"],
        )
        self.assertNotEqual(
            first["host_wall_elapsed_ns"], delayed["host_wall_elapsed_ns"]
        )

        slower_output = json.dumps(
            guest_result("atomic", 1, 10, elapsed_ns=400_000_000)
        )
        with mock.patch.object(
            bench, "run_process", return_value=(0, slower_output, "")
        ), mock.patch.object(
            bench.time, "perf_counter_ns", side_effect=[0, 1_000_000_000]
        ):
            slower = bench.measure_once(
                repo=ROOT,
                runner=[],
                build=build,
                module=Path("fixture"),
                workload="atomic",
                threads=1,
                iterations=10,
                timeout=1,
                min_interval_ns=1,
                record_fields={"cancel_points": "on"},
            )
        self.assertEqual(
            first["throughput_ops_per_second"] / 2,
            slower["throughput_ops_per_second"],
        )

    def test_timeout_and_failure_propagate(self) -> None:
        sleeper = self.scratch / "sleep.py"
        sleeper.write_text("import time\ntime.sleep(2)\n", encoding="UTF-8")
        with self.assertRaisesRegex(bench.HarnessError, "timed out"):
            bench.run_process([sys.executable, str(sleeper)], ROOT, 0.01)

        failure = self.scratch / "failure.py"
        failure.write_text("raise SystemExit(7)\n", encoding="UTF-8")
        returncode, _, _ = bench.run_process(
            [sys.executable, str(failure)], ROOT, 5
        )
        self.assertEqual(returncode, 7)

    def test_guest_failure_classification_is_specific(self) -> None:
        cases = {
            "wamr-aot-atomic-wait32 outcome=backend-error detail=SystemFailure": (
                "atomic-wait-backend-error"
            ),
            "wamr-aot-atomic-wait32 outcome=unexpected-timeout": (
                "atomic-wait-unexpected-timeout"
            ),
            "wamr-aot-atomic-wait32 outcome=cancelled": "atomic-wait-cancelled",
            "wamr-aot-atomic-wait32 outcome=closed": "atomic-wait-closed",
            "worker[0] failed: 13": "barrier-value-mismatch",
            "worker[0] failed: 12": "atomic-wait-invalid-result",
            "worker[0] failed: 11": "atomic-wait-unexpected-timeout",
            "worker[0] failed: 10": "barrier-peer-abort",
            "controller barrier failed: 141": "controller-barrier-failure",
        }
        for stderr, expected in cases.items():
            with self.subTest(stderr=stderr):
                self.assertEqual(
                    bench.classify_guest_failure(stderr, "atomic"),
                    expected,
                )

    def test_barrier_calibration_accepts_same_clock_tick(self) -> None:
        source = (
            ROOT / "tests/benchmarks/wasi-threads/threaded.c"
        ).read_text(encoding="UTF-8")
        self.assertIn("if (end < start)", source)
        self.assertNotIn("if (end <= start)", source)

    def test_cache_key_is_canonical_and_configuration_sensitive(self) -> None:
        left = {"target": "native", "threads": True, "mode": "aot"}
        right = {"mode": "aot", "threads": True, "target": "native"}
        self.assertEqual(cache_key(left), cache_key(right))
        self.assertNotEqual(cache_key(left), cache_key(dict(left, threads=False)))

    def test_host_fingerprint_excludes_high_cardinality_runner_identity(self) -> None:
        with mock.patch.dict(
            schema.os.environ,
            {
                "RUNNER_NAME": "hosted-runner-123",
                "GITHUB_RUN_ID": "100",
                "GITHUB_RUN_ATTEMPT": "1",
            },
            clear=False,
        ):
            first = schema.host_metadata("github-hosted")
        with mock.patch.dict(
            schema.os.environ,
            {
                "RUNNER_NAME": "hosted-runner-987",
                "GITHUB_RUN_ID": "200",
                "GITHUB_RUN_ATTEMPT": "2",
            },
            clear=False,
        ):
            second = schema.host_metadata("github-hosted")
        self.assertNotEqual(first["runner_name"], second["runner_name"])
        self.assertEqual(
            first["host_fingerprint"]["sha256"],
            second["host_fingerprint"]["sha256"],
        )
        fields = first["host_fingerprint"]["fields"]
        self.assertNotIn("runner_name", fields)
        self.assertNotIn("github_run_id", fields)

    def test_report_fails_closed_on_revision_and_provenance_corruption(self) -> None:
        report = make_report()
        self.assertNotEqual(
            report["metadata"]["revisions"]["baseline"][
                "build_source_sha256"
            ],
            report["metadata"]["revisions"]["candidate"][
                "build_source_sha256"
            ],
        )

        missing_revision = copy.deepcopy(report)
        del missing_revision["metadata"]["revisions"]["candidate"]
        with self.assertRaisesRegex(BenchmarkDataError, "metadata.revisions"):
            bench.validate_report(missing_revision)

        duplicate = copy.deepcopy(report)
        duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
        with self.assertRaisesRegex(BenchmarkDataError, "duplicate record"):
            bench.validate_report(duplicate)

        inverted = copy.deepcopy(report)
        inverted["records"][0]["revision"] = "candidate"
        with self.assertRaisesRegex(BenchmarkDataError, "mixed|duplicate"):
            bench.validate_report(inverted)

        inverted_order = copy.deepcopy(report)
        inverted_order["records"][0], inverted_order["records"][1] = (
            inverted_order["records"][1],
            inverted_order["records"][0],
        )
        with self.assertRaisesRegex(BenchmarkDataError, "inverted pair order"):
            bench.validate_report(inverted_order)

        mutations = (
            (
                "host",
                lambda value: value["metadata"]["revisions"]["candidate"].__setitem__(
                    "host_pair_id", "other-host-pair"
                ),
            ),
            (
                "plan",
                lambda value: value["metadata"]["revisions"]["candidate"].__setitem__(
                    "plan_sha256", "f" * 64
                ),
            ),
            (
                "fixture",
                lambda value: value["metadata"]["revisions"]["candidate"].__setitem__(
                    "fixture_set_sha256", "f" * 64
                ),
            ),
        )
        for label, mutate in mutations:
            corrupt = copy.deepcopy(report)
            mutate(corrupt)
            with self.subTest(label=label), self.assertRaisesRegex(
                BenchmarkDataError, f"mixed {label}"
            ):
                bench.validate_report(corrupt)

    def test_candidate_identity_is_not_calibration_provenance(self) -> None:
        calibrated_report = make_report(
            baseline_commit="a" * 40,
            commit="a" * 40,
            baseline_source="c" * 64,
            candidate_source="c" * 64,
        )
        budget = complete_budget(calibrated_report)
        future_candidate = make_report(
            baseline_commit="a" * 40,
            commit="e" * 40,
            baseline_source="c" * 64,
            candidate_source="f" * 64,
        )
        loaded = bench.load_budget(
            self.write_budget(budget), future_candidate
        )
        self.assertEqual(
            loaded["host_machine"],
            future_candidate["metadata"]["host"]["machine"],
        )

        changed_baseline = make_report(
            baseline_commit="e" * 40,
            commit="f" * 40,
            baseline_source="f" * 64,
            candidate_source="e" * 64,
        )
        with self.assertRaisesRegex(
            bench.HarnessError, "baseline_revision"
        ):
            bench.load_budget(self.write_budget(budget), changed_baseline)

    def test_paired_purpose_prevents_checkout_and_a_a_gate_bugs(self) -> None:
        same_checkout = bench.parse_args(
            [
                "--baseline-repo",
                str(ROOT),
                "--candidate-repo",
                str(ROOT),
                "--comparison-purpose",
                "candidate-evaluation",
                "--samples",
                "2",
                "--no-budget",
            ]
        )
        with self.assertRaisesRegex(
            bench.HarnessError, "distinct independently built checkout paths"
        ):
            bench.execute(same_checkout)

        noise = make_report(
            commit="a" * 40,
            baseline_commit="a" * 40,
            baseline_source="c" * 64,
            candidate_source="c" * 64,
            comparison_purpose="noise-calibration",
        )
        self.assertEqual(
            noise["plan"]["comparison_purpose"], "noise-calibration"
        )
        with self.assertRaisesRegex(
            BenchmarkDataError, "noise calibration revision identity"
        ):
            make_report(comparison_purpose="noise-calibration")
        same_path = copy.deepcopy(noise)
        same_path["metadata"]["revision_checkouts"]["baseline"] = (
            same_path["metadata"]["revision_checkouts"]["candidate"]
        )
        with self.assertRaisesRegex(
            BenchmarkDataError, "same checkout path"
        ):
            bench.validate_report(same_path)
        with self.assertRaisesRegex(
            bench.HarnessError, "paired candidate-evaluation"
        ):
            bench.load_budget(self.write_budget(complete_budget(noise)), noise)
        with self.assertRaisesRegex(
            bench.HarnessError, "paired candidate-evaluation"
        ):
            bench.evaluate_budget({}, noise)

        identical_evaluation = make_report(
            commit="a" * 40,
            baseline_commit="a" * 40,
            baseline_source="c" * 64,
            candidate_source="c" * 64,
        )
        with self.assertRaisesRegex(
            bench.HarnessError, "distinct baseline/candidate commits"
        ):
            bench.load_budget(
                self.write_budget(complete_budget(identical_evaluation)),
                identical_evaluation,
            )
        with self.assertRaisesRegex(
            bench.HarnessError, "distinct baseline/candidate commits"
        ):
            bench.evaluate_budget({}, identical_evaluation)

        identical_build = make_report(
            commit="a" * 40,
            baseline_commit="b" * 40,
            baseline_source="c" * 64,
            candidate_source="c" * 64,
        )
        with self.assertRaisesRegex(
            bench.HarnessError, "distinct baseline/candidate build identities"
        ):
            bench.load_budget(
                self.write_budget(complete_budget(identical_build)),
                identical_build,
            )

        with self.assertRaises(SystemExit):
            bench.parse_args(
                [
                    "--baseline-repo",
                    "baseline",
                    "--candidate-repo",
                    "candidate",
                    "--comparison-purpose",
                    "noise-calibration",
                    "--samples",
                    "2",
                ]
            )

    def test_ratio_of_ratios_direction_and_budget_limits(self) -> None:
        report = make_report()
        ratio = report["ratio_of_ratios_summaries"][0]
        self.assertAlmostEqual(
            ratio["elapsed_ratio_of_ratios"]["median"],
            (120 / 90) / (120 / 100),
        )
        self.assertAlmostEqual(
            ratio["throughput_ratio_of_ratios"]["median"],
            (90 / 120) / (100 / 120),
        )
        budget = complete_budget(report)
        platform = budget["platforms"][report["metadata"]["platform_id"]]
        platform["ratio_of_ratios"][0][
            "min_candidate_over_baseline_throughput_ratio_of_ratios"
        ] = 0.95
        platform["ratio_of_ratios"][0][
            "max_candidate_over_baseline_elapsed_ratio_of_ratios"
        ] = 1.05
        loaded = bench.load_budget(self.write_budget(budget), report)
        failures = bench.evaluate_budget(loaded, report)
        self.assertTrue(
            any("throughput ratio-of-ratios" in item for item in failures)
        )
        self.assertTrue(
            any("elapsed ratio-of-ratios" in item for item in failures)
        )

    def test_report_rejects_missing_scenario_and_direction_corruption(self) -> None:
        report = make_report()
        missing = copy.deepcopy(report)
        missing["records"] = missing["records"][2:]
        with self.assertRaisesRegex(BenchmarkDataError, "incomplete sample pairing"):
            bench.validate_report(missing)

        corrupt = copy.deepcopy(report)
        corrupt["plan"]["pairs"][0]["left"] = "renamed"
        corrupt["metadata"]["plan_sha256"] = cache_key(corrupt["plan"])
        with self.assertRaisesRegex(BenchmarkDataError, "plan identity|incomplete"):
            bench.validate_report(corrupt)

    def test_budget_rejects_empty_partial_unknown_duplicate_and_direction(self) -> None:
        report = make_report()
        budget = complete_budget(report)
        loaded = bench.load_budget(self.write_budget(budget), report)
        self.assertFalse(bench.evaluate_budget(loaded, report))
        uncalibrated = copy.deepcopy(budget)
        uncalibrated["calibrated"] = False
        with self.assertRaisesRegex(bench.HarnessError, "not calibrated"):
            bench.load_budget(self.write_budget(uncalibrated), report)

        cases = {}
        value = copy.deepcopy(budget)
        value["platforms"] = {}
        cases["empty"] = value
        value = copy.deepcopy(budget)
        value["platforms"][report["metadata"]["platform_id"]][
            "comparisons"
        ].pop()
        cases["partial"] = value
        value = copy.deepcopy(budget)
        value["platforms"][report["metadata"]["platform_id"]][
            "comparisons"
        ][0]["pair_key"] = "unknown"
        cases["unknown"] = value
        value = copy.deepcopy(budget)
        value["platforms"][report["metadata"]["platform_id"]][
            "comparisons"
        ].append(
            copy.deepcopy(
                value["platforms"][report["metadata"]["platform_id"]][
                    "comparisons"
                ][0]
            )
        )
        cases["duplicate"] = value
        value = copy.deepcopy(budget)
        pair = value["platforms"][report["metadata"]["platform_id"]][
            "ratio_of_ratios"
        ][0]
        pair["left"], pair["right"] = pair["right"], pair["left"]
        cases["direction"] = value
        value = copy.deepcopy(budget)
        value["platforms"][report["metadata"]["platform_id"]][
            "comparisons"
        ][0]["min_candidate_over_baseline_throughput_ratio"] = float("nan")
        cases["nonfinite"] = value
        for label, corrupt in cases.items():
            with self.subTest(label=label), self.assertRaises(bench.HarnessError):
                bench.load_budget(self.write_budget(corrupt), report)

    def test_budget_rejects_identity_count_profile_and_host_mismatch(self) -> None:
        report = make_report()
        budget = complete_budget(report)
        mutations = (
            ("commit", lambda value: value["calibration_provenance"]["baseline_revision"].__setitem__("commit", "e" * 40)),
            ("source", lambda value: value["calibration_provenance"]["baseline_revision"].__setitem__("build_source_sha256", "f" * 64)),
            ("fixture", lambda value: value["calibration_provenance"].__setitem__("fixture_set_sha256", "f" * 64)),
            ("plan", lambda value: value["calibration_provenance"].__setitem__("plan_sha256", "f" * 64)),
            ("profile", lambda value: value["calibration_provenance"].__setitem__("profile", "smoke")),
            ("count", lambda value: value["calibration_provenance"]["report_count_by_platform"].__setitem__(report["metadata"]["platform_id"], 19)),
            ("host", lambda value: value["platforms"][report["metadata"]["platform_id"]].__setitem__("host_machine", "aarch64")),
        )
        for label, mutate in mutations:
            corrupt = copy.deepcopy(budget)
            mutate(corrupt)
            with self.subTest(label=label), self.assertRaises(bench.HarnessError):
                bench.load_budget(self.write_budget(corrupt), report)

        duplicate_json = (
            '{"schema_version":2,"schema_version":2,'
            '"kind":"wasi-thread-benchmark-budget","calibrated":true,'
            '"enforcement":true,"calibration_requirements":{},'
            '"calibration_provenance":{},"platforms":{}}'
        )
        with self.assertRaisesRegex(bench.HarnessError, "duplicate"):
            bench.load_budget(self.write_budget(duplicate_json), report)

    def test_budget_requires_exact_canonical_platform_set_in_either_order(self) -> None:
        report = make_report()
        budget = complete_budget(report)
        reversed_budget = copy.deepcopy(budget)
        reversed_budget["calibration_requirements"]["required_platforms"].reverse()
        bench.load_budget(self.write_budget(reversed_budget), report)

        invalid_sets = (
            ["ubuntu-22.04-x86_64"],
            ["ubuntu-24.04-aarch64"],
            ["arbitrary-platform"],
            [
                "ubuntu-22.04-x86_64",
                "ubuntu-24.04-aarch64",
                "third-platform",
            ],
            ["ubuntu-22.04-x86_64", "ubuntu-22.04-x86_64"],
        )
        for required_platforms in invalid_sets:
            corrupt = copy.deepcopy(budget)
            corrupt["calibration_requirements"][
                "required_platforms"
            ] = required_platforms
            with self.subTest(required_platforms=required_platforms):
                with self.assertRaisesRegex(
                    bench.HarnessError, "canonical hosted platforms"
                ):
                    bench.load_budget(self.write_budget(corrupt), report)

    def test_cohort_rejects_schema_v3_paired_reports_until_updated(self) -> None:
        paired = make_report(run_id="100")
        with self.assertRaisesRegex(
            bench.HarnessError, "baseline-aware cohort aggregation"
        ):
            cohort.validate_documents(
                [(Path("paired"), paired)],
                cohort.DEFAULT_PLATFORMS,
                1,
            )

    def test_cohort_rejects_mixed_identity_and_duplicate_run_ids(self) -> None:
        x86 = make_single_report(run_id="100")
        arm = make_single_report(
            platform_id="ubuntu-24.04-aarch64",
            machine="aarch64",
            run_id="100",
        )
        result = cohort.validate_documents(
            [(Path("x86"), x86), (Path("arm"), arm)],
            cohort.DEFAULT_PLATFORMS,
            1,
        )
        self.assertEqual(result["identity"]["commit"], "a" * 40)

        mixed = make_single_report(
            platform_id="ubuntu-24.04-aarch64",
            machine="aarch64",
            commit="e" * 40,
            run_id="100",
        )
        with self.assertRaisesRegex(bench.HarnessError, "mixed"):
            cohort.validate_documents(
                [(Path("x86"), x86), (Path("arm"), mixed)],
                cohort.DEFAULT_PLATFORMS,
                1,
            )
        duplicate = copy.deepcopy(x86)
        with self.assertRaisesRegex(bench.HarnessError, "duplicate"):
            cohort.validate_documents(
                [(Path("x1"), x86), (Path("x2"), duplicate), (Path("arm"), arm)],
                cohort.DEFAULT_PLATFORMS,
                1,
            )

    def test_cohort_requires_canonical_single_revision_platform_reports(self) -> None:
        x86 = make_single_report(run_id="100")
        arm = make_single_report(
            platform_id="ubuntu-24.04-aarch64",
            machine="aarch64",
            run_id="100",
        )
        for platforms in (
            ("ubuntu-22.04-x86_64",),
            ("ubuntu-24.04-aarch64",),
            ("arbitrary-platform",),
            (
                "ubuntu-22.04-x86_64",
                "ubuntu-24.04-aarch64",
                "third-platform",
            ),
            ("ubuntu-22.04-x86_64", "ubuntu-22.04-x86_64"),
        ):
            with self.subTest(platforms=platforms), self.assertRaisesRegex(
                bench.HarnessError, "canonical hosted set"
            ):
                cohort.validate_documents(
                    [(Path("x86"), x86), (Path("arm"), arm)],
                    platforms,
                    1,
                )

        masquerading_arm = make_single_report(
            platform_id="ubuntu-24.04-aarch64",
            machine="x86_64",
            run_id="100",
        )
        with self.assertRaisesRegex(bench.HarnessError, "canonical host identity"):
            cohort.validate_documents(
                [(Path("x86"), x86), (Path("arm"), masquerading_arm)],
                cohort.DEFAULT_PLATFORMS,
                1,
            )

        other_run_arm = make_single_report(
            platform_id="ubuntu-24.04-aarch64",
            machine="aarch64",
            run_id="101",
        )
        with self.assertRaisesRegex(bench.HarnessError, "same workflow runs"):
            cohort.validate_documents(
                [(Path("x86"), x86), (Path("arm"), other_run_arm)],
                cohort.DEFAULT_PLATFORMS,
                1,
            )

        second_x86 = make_single_report(run_id="101")
        with self.assertRaisesRegex(bench.HarnessError, "different report counts"):
            cohort.validate_documents(
                [
                    (Path("x86-100"), x86),
                    (Path("x86-101"), second_x86),
                    (Path("arm-100"), arm),
                ],
                cohort.DEFAULT_PLATFORMS,
                1,
            )

    def test_cohort_dispatch_uses_ref_pinned_to_immutable_sha(self) -> None:
        target = "a" * 40
        workflow_ref = "calibration/966-immutable"
        output = self.scratch / "dispatch.json"
        responses = [
            "https://github.com/cataggar/wamr/actions/runs/123\n",
            json.dumps(
                {
                    "status": "completed",
                    "conclusion": "success",
                    "headSha": target,
                    "url": "https://github.com/cataggar/wamr/actions/runs/123",
                }
            ),
            json.dumps({"artifacts": []}),
        ]
        with mock.patch.object(
            cohort.subprocess,
            "check_output",
            side_effect=responses,
        ) as run, mock.patch.object(cohort.time, "sleep"):
            cohort.dispatch(
                Namespace(
                    target_sha=target,
                    runs=1,
                    max_in_flight=1,
                    output=output,
                    repository="cataggar/wamr",
                    workflow="wasi-thread-bench.yml",
                    workflow_ref=workflow_ref,
                    poll_seconds=0,
                )
            )
        dispatch_command = run.call_args_list[0].args[0]
        self.assertEqual(
            dispatch_command[dispatch_command.index("--ref") + 1],
            workflow_ref,
        )

        mismatch_output = self.scratch / "dispatch-mismatch.json"
        mismatch_responses = [
            "https://github.com/cataggar/wamr/actions/runs/124\n",
            json.dumps(
                {
                    "status": "completed",
                    "conclusion": "success",
                    "headSha": "b" * 40,
                    "url": "https://github.com/cataggar/wamr/actions/runs/124",
                }
            ),
        ]
        with (
            mock.patch.object(
                cohort.subprocess,
                "check_output",
                side_effect=mismatch_responses,
            ),
            mock.patch.object(cohort.time, "sleep"),
            self.assertRaisesRegex(bench.HarnessError, "does not match target"),
        ):
            cohort.dispatch(
                Namespace(
                    target_sha=target,
                    runs=1,
                    max_in_flight=1,
                    output=mismatch_output,
                    repository="cataggar/wamr",
                    workflow="wasi-thread-bench.yml",
                    workflow_ref=workflow_ref,
                    poll_seconds=0,
                )
            )

    def test_cohort_main_handles_report_schema_errors(self) -> None:
        with mock.patch.object(
            cohort,
            "validate_cohort",
            side_effect=BenchmarkDataError("bad report"),
        ):
            self.assertEqual(
                cohort.main(
                    [
                        "validate",
                        "--input-dir",
                        str(self.scratch),
                    ]
                ),
                2,
            )

    def test_cancel_poll_sites_use_machine_code_signatures(self) -> None:
        for arch, bytes_per_site in (("x86_64", 21), ("aarch64", 20)):
            signature = bench.CANCEL_POLL_SIGNATURES[arch]
            off_text = bytes(32)
            on_text = (
                signature
                + bytes(7)
                + signature
                + bytes(len(off_text) + 2 * bytes_per_site - 2 * len(signature) - 7)
            )
            artifacts = {}
            for name, text in (
                ("single", bytes(8)),
                ("threaded-polls-on", on_text),
                ("threaded-polls-off", off_text),
            ):
                path = self.scratch / f"{arch}-{name}.cwasm"
                path.write_bytes(
                    b"\x00aot"
                    + struct.pack("<I", 11)
                    + struct.pack("<II", 2, len(text))
                    + text
                )
                artifacts[name] = path
            report = bench.aot_artifact_report(artifacts, arch)
            detected = report["cancel_poll_static"]
            self.assertEqual(detected["detection"], "machine-code-signature")
            self.assertEqual(detected["sites_enabled"], 2)
            self.assertEqual(detected["sites_disabled"], 0)
            self.assertEqual(detected["bytes_per_site"], bytes_per_site)

        unsupported = self.scratch / "unsupported-version.cwasm"
        unsupported.write_bytes(
            b"\x00aot"
            + struct.pack("<I", bench.AOT_VERSION - 1)
            + struct.pack("<II", 2, 0)
        )
        with self.assertRaisesRegex(bench.HarnessError, "unsupported WAMR AOT version"):
            bench.aot_text_section(unsupported)

    def test_fixture_hashes_and_schema_are_pinned(self) -> None:
        fixtures = bench.resolve_fixtures(ROOT)
        self.assertEqual(fixtures["single"]["sha256"], bench.FIXTURES["single"]["sha256"])
        self.assertEqual(fixtures["threaded"]["sha256"], bench.FIXTURES["threaded"]["sha256"])
        schema = json.loads(
            (
                ROOT
                / "tests"
                / "benchmarks"
                / "wasi-threads"
                / "report.schema.json"
            ).read_text(encoding="UTF-8")
        )
        self.assertEqual(
            schema["properties"]["schema_version"]["const"], SCHEMA_VERSION
        )
        self.assertEqual(schema["properties"]["kind"]["const"], bench.KIND)
        self.assertEqual(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema",
        )
        self.assertIn("revision_checkouts", schema["properties"]["metadata"]["required"])
        self.assertIn("comparison_purpose", schema["properties"]["plan"]["required"])
        paired_contract = schema["allOf"][0]["then"]["properties"]
        self.assertEqual(
            paired_contract["plan"]["properties"]["samples"]["multipleOf"],
            2,
        )
        self.assertEqual(
            paired_contract["plan"]["properties"]["revision_roles"]["const"],
            list(bench.REVISION_ROLES),
        )
        single_contract = schema["allOf"][0]["else"]["properties"]
        self.assertEqual(
            single_contract["plan"]["properties"]["revision_roles"]["const"],
            list(bench.SINGLE_REVISION_ROLES),
        )
        self.assertEqual(
            single_contract["comparison_summaries"]["maxItems"], 0
        )
        for report in (make_report(), make_single_report()):
            self.assertEqual(set(report), set(schema["required"]))
            bench.validate_report(report)
        budget_schema = json.loads(
            (
                ROOT
                / "tests"
                / "benchmarks"
                / "wasi-threads"
                / "budget.schema.json"
            ).read_text(encoding="UTF-8")
        )
        self.assertEqual(
            budget_schema["properties"]["schema_version"]["const"],
            SCHEMA_VERSION,
        )
        self.assertEqual(
            budget_schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema",
        )
        self.assertEqual(
            budget_schema["$defs"]["calibration_provenance"]["properties"][
                "comparison_purpose"
            ]["const"],
            "noise-calibration",
        )
        calibrated = complete_budget(make_report())
        self.assertEqual(set(calibrated), set(budget_schema["required"]))
        uncalibrated = json.loads(
            (
                ROOT
                / "tests"
                / "benchmarks"
                / "wasi-threads"
                / "budget.json"
            ).read_text(encoding="UTF-8")
        )
        self.assertEqual(set(uncalibrated), set(budget_schema["required"]))
        self.assertFalse(uncalibrated["calibrated"])
        self.assertFalse(uncalibrated["enforcement"])
        self.assertIsNone(uncalibrated["calibration_provenance"])
        self.assertEqual(uncalibrated["platforms"], {})


if __name__ == "__main__":
    unittest.main()
