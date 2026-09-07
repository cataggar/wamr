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
    cpu: str = "test cpu",
    runner_environment: str = "github-hosted",
    runner_image: str = "ubuntu",
    runner_name: str = "test-runner",
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
        "cpu": cpu,
        "logical_cpus": 4,
        "runner_environment": runner_environment,
        "runner_image": runner_image,
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
                "runner_environment": runner_environment,
                "runner_name": runner_name,
                "github_run_id": run_id,
                "host_fingerprint": {
                    "sha256": host_fingerprint,
                    "fields": host_fields,
                },
            },
            "host_pair": {
                "id": host_pair_id,
                "runner_environment": runner_environment,
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


def make_dispatch_state(
    run_ids: tuple[str, ...] = ("100", "101"),
    baseline_sha: str = "b" * 40,
    candidate_sha: str = "a" * 40,
    purpose: str = "candidate-evaluation",
    runner_target: str = "github-hosted",
) -> dict:
    cohort_id = "d" * 32
    training_runs = len(run_ids) // 2
    assignments = [
        {
            "sequence": sequence,
            "partition": "training" if sequence <= training_runs else "holdout",
        }
        for sequence in range(1, len(run_ids) + 1)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-thread-cohort-dispatch",
        "created_at": "2026-09-07T00:00:00+00:00",
        "repository": "cataggar/wamr",
        "workflow": "wasi-thread-bench.yml",
        "workflow_ref": "main",
        "workflow_head_sha": "f" * 40,
        "cohort_id": cohort_id,
        "baseline_sha": baseline_sha,
        "candidate_sha": candidate_sha,
        "comparison_purpose": purpose,
        "profile": "authoritative",
        "warmups": 0,
        "samples": 2,
        "runner_target": runner_target,
        "required_platforms": list(cohort.DEFAULT_PLATFORMS),
        "requested_runs": len(run_ids),
        "requested_reports": len(run_ids) * len(cohort.DEFAULT_PLATFORMS),
        "max_in_flight": 2,
        "timeout_seconds": cohort.DEFAULT_DISPATCH_TIMEOUT_SECONDS,
        "split": {
            "method": "predeclared-sequence",
            "training_runs": training_runs,
            "holdout_runs": len(run_ids) - training_runs,
            "assignments": assignments,
        },
        "runs": [
            {
                "sequence": sequence,
                "partition": assignments[sequence - 1]["partition"],
                "run_id": int(run_id),
                "run_name": (
                    f"WASI thread cohort-{cohort_id}-{sequence}-"
                    f"{assignments[sequence - 1]['partition']}"
                ),
                "url": f"https://github.com/cataggar/wamr/actions/runs/{run_id}",
                "status": "completed",
                "conclusion": "success",
                "workflow_head_sha": "f" * 40,
                "artifacts": [],
            }
            for sequence, run_id in enumerate(run_ids, 1)
        ],
    }


def make_paired_cohort_reports(
    run_ids: tuple[str, ...] = ("100", "101"),
    **kwargs,
) -> list[tuple[Path, dict]]:
    reports = []
    for run_id in run_ids:
        reports.extend(
            [
                (
                    Path(f"x86-{run_id}"),
                    make_report(run_id=run_id, **kwargs),
                ),
                (
                    Path(f"arm-{run_id}"),
                    make_report(
                        platform_id="ubuntu-24.04-aarch64",
                        machine="aarch64",
                        run_id=run_id,
                        **kwargs,
                    ),
                ),
            ]
        )
    return reports


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

    def test_paired_cohort_preserves_exact_runs_and_predeclared_split(self) -> None:
        dispatch_state = make_dispatch_state()
        reports = make_paired_cohort_reports()
        result = cohort.validate_documents(
            reports,
            cohort.DEFAULT_PLATFORMS,
            1,
            dispatch_state,
        )
        self.assertTrue(result["authoritative"])
        self.assertEqual(result["identity"]["baseline"]["commit"], "b" * 40)
        self.assertEqual(result["identity"]["candidate"]["commit"], "a" * 40)
        self.assertEqual(
            result["split"]["run_ids"],
            {"training": ["100"], "holdout": ["101"]},
        )
        self.assertEqual(len(result["observations"]), len(reports))
        self.assertEqual(result["excluded_observations"], [])
        self.assertEqual(
            {
                (item["run_id"], item["platform"])
                for item in result["observations"]
            },
            {
                (run_id, platform)
                for run_id in ("100", "101")
                for platform in cohort.DEFAULT_PLATFORMS
            },
        )

    def test_trusted_noise_cohort_pairs_self_hosted_x86_with_hosted_arm(self) -> None:
        dispatch_state = make_dispatch_state(
            baseline_sha="a" * 40,
            candidate_sha="a" * 40,
            purpose="noise-calibration",
            runner_target="trusted-calibration",
        )
        reports = []
        for run_id in ("100", "101"):
            common = {
                "run_id": run_id,
                "baseline_commit": "a" * 40,
                "commit": "a" * 40,
                "baseline_source": "c" * 64,
                "candidate_source": "c" * 64,
                "comparison_purpose": "noise-calibration",
            }
            reports.extend(
                [
                    (
                        Path(f"x86-{run_id}"),
                        make_report(
                            runner_environment="self-hosted",
                            runner_name="vm31e",
                            **common,
                        ),
                    ),
                    (
                        Path(f"arm-{run_id}"),
                        make_report(
                            platform_id="ubuntu-24.04-aarch64",
                            machine="aarch64",
                            cpu=f"hosted arm {run_id}",
                            runner_image=f"ubuntu-arm-{run_id}",
                            **common,
                        ),
                    ),
                ]
            )
        result = cohort.validate_documents(
            reports,
            cohort.DEFAULT_PLATFORMS,
            1,
            dispatch_state,
        )
        self.assertEqual(
            result["dispatch"]["runner_target"], "trusted-calibration"
        )
        self.assertEqual(
            result["identity"]["comparison_purpose"], "noise-calibration"
        )
        self.assertEqual(
            result["platforms"]["ubuntu-22.04-x86_64"][
                "trusted_runner_name"
            ],
            "vm31e",
        )
        self.assertEqual(
            len(
                result["platforms"]["ubuntu-24.04-aarch64"][
                    "host_fingerprint_distribution"
                ]
            ),
            2,
        )

    def test_paired_cohort_requires_manifest_exact_count_and_no_legacy(self) -> None:
        reports = make_paired_cohort_reports()
        dispatch_state = make_dispatch_state()
        with self.assertRaisesRegex(bench.HarnessError, "dispatch manifest"):
            cohort.validate_documents(
                reports,
                cohort.DEFAULT_PLATFORMS,
                1,
            )
        with self.assertRaisesRegex(bench.HarnessError, "expected exactly"):
            cohort.validate_documents(
                reports[:-1],
                cohort.DEFAULT_PLATFORMS,
                1,
                dispatch_state,
            )
        duplicate = reports[:-1] + [
            (Path("duplicate"), copy.deepcopy(reports[2][1]))
        ]
        with self.assertRaisesRegex(bench.HarnessError, "duplicate report"):
            cohort.validate_documents(
                duplicate,
                cohort.DEFAULT_PLATFORMS,
                1,
                dispatch_state,
            )
        mixed = reports[:-1] + [
            (
                Path("legacy"),
                make_single_report(
                    platform_id="ubuntu-24.04-aarch64",
                    machine="aarch64",
                    run_id="101",
                ),
            )
        ]
        with self.assertRaisesRegex(bench.HarnessError, "legacy or unpaired"):
            cohort.validate_documents(
                mixed,
                cohort.DEFAULT_PLATFORMS,
                1,
                dispatch_state,
            )

    def test_paired_cohort_accepts_hosted_host_distributions(self) -> None:
        reports = make_paired_cohort_reports()
        reports[2] = (
            Path("x86-101-other-host"),
            make_report(
                run_id="101",
                cpu="different cpu",
                runner_image="ubuntu-24.04",
            ),
        )
        result = cohort.validate_documents(
            reports,
            cohort.DEFAULT_PLATFORMS,
            1,
            make_dispatch_state(),
        )
        x86 = result["platforms"]["ubuntu-22.04-x86_64"]
        self.assertEqual(len(x86["host_fingerprint_distribution"]), 2)
        self.assertEqual(
            x86["cpu_distribution"], {"different cpu": 1, "test cpu": 1}
        )
        self.assertEqual(
            x86["runner_image_distribution"],
            {"ubuntu": 1, "ubuntu-24.04": 1},
        )
        self.assertEqual(len(result["observations"]), len(reports))
        self.assertEqual(result["excluded_observations"], [])

    def test_trusted_cohort_rejects_x86_drift_and_wrong_runner(self) -> None:
        state = make_dispatch_state(
            baseline_sha="a" * 40,
            candidate_sha="a" * 40,
            purpose="noise-calibration",
            runner_target="trusted-calibration",
        )

        def reports(second_cpu: str = "test cpu", runner_name: str = "vm31e"):
            result = []
            for run_id, cpu in (("100", "test cpu"), ("101", second_cpu)):
                common = {
                    "run_id": run_id,
                    "baseline_commit": "a" * 40,
                    "commit": "a" * 40,
                    "baseline_source": "c" * 64,
                    "candidate_source": "c" * 64,
                    "comparison_purpose": "noise-calibration",
                }
                result.extend(
                    [
                        (
                            Path(f"x86-{run_id}"),
                            make_report(
                                cpu=cpu,
                                runner_environment="self-hosted",
                                runner_name=runner_name,
                                **common,
                            ),
                        ),
                        (
                            Path(f"arm-{run_id}"),
                            make_report(
                                platform_id="ubuntu-24.04-aarch64",
                                machine="aarch64",
                                **common,
                            ),
                        ),
                    ]
                )
            return result

        with self.assertRaisesRegex(
            bench.HarnessError, "mixed vm31e host fingerprints"
        ):
            cohort.validate_documents(
                reports(second_cpu="different cpu"),
                cohort.DEFAULT_PLATFORMS,
                1,
                state,
            )
        with self.assertRaisesRegex(bench.HarnessError, "runner 'vm31e'"):
            cohort.validate_documents(
                reports(runner_name="other-runner"),
                cohort.DEFAULT_PLATFORMS,
                1,
                state,
            )

    def test_paired_cohort_rejects_mixed_baseline_report_host_plan_and_pair_order(self) -> None:
        dispatch_state = make_dispatch_state()
        reports = make_paired_cohort_reports()

        mixed_baseline = copy.deepcopy(reports)
        changed = mixed_baseline[-1][1]
        changed["metadata"]["revisions"]["baseline"]["commit"] = "c" * 40
        for record in changed["records"]:
            if record["revision"] == "baseline":
                record["revision_commit"] = "c" * 40
        bench.validate_report(changed)
        with self.assertRaisesRegex(bench.HarnessError, "baseline SHA"):
            cohort.validate_documents(
                mixed_baseline,
                cohort.DEFAULT_PLATFORMS,
                1,
                dispatch_state,
            )

        mixed_report_host = copy.deepcopy(reports)
        changed = mixed_report_host[0][1]
        changed["metadata"]["revisions"]["candidate"][
            "host_fingerprint_sha256"
        ] = "f" * 64
        for record in changed["records"]:
            if record["revision"] == "candidate":
                record["host_fingerprint_sha256"] = "f" * 64
        with self.assertRaisesRegex(BenchmarkDataError, "mixed host fingerprint"):
            cohort.validate_documents(
                mixed_report_host,
                cohort.DEFAULT_PLATFORMS,
                1,
                dispatch_state,
            )

        mixed_plan = copy.deepcopy(reports)
        mixed_plan[-1][1]["plan"]["profile"] = "smoke"
        mixed_plan[-1][1]["metadata"]["plan_sha256"] = cache_key(
            mixed_plan[-1][1]["plan"]
        )
        for revision in mixed_plan[-1][1]["metadata"]["revisions"].values():
            revision["plan_sha256"] = mixed_plan[-1][1]["metadata"]["plan_sha256"]
        for record in mixed_plan[-1][1]["records"]:
            record["plan_sha256"] = mixed_plan[-1][1]["metadata"]["plan_sha256"]
        bench.validate_report(mixed_plan[-1][1])
        with self.assertRaisesRegex(bench.HarnessError, "plan/profile"):
            cohort.validate_documents(
                mixed_plan,
                cohort.DEFAULT_PLATFORMS,
                1,
                dispatch_state,
            )

        inverted = copy.deepcopy(reports)
        inverted[0][1]["records"][0], inverted[0][1]["records"][1] = (
            inverted[0][1]["records"][1],
            inverted[0][1]["records"][0],
        )
        with self.assertRaisesRegex(BenchmarkDataError, "inverted pair order"):
            cohort.validate_documents(
                inverted,
                cohort.DEFAULT_PLATFORMS,
                1,
                dispatch_state,
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

    def test_cohort_dispatch_sends_full_paired_identity_and_pins_workflow(self) -> None:
        baseline = "b" * 40
        candidate = "a" * 40
        workflow_head = "f" * 40
        workflow_ref = "calibration/966-immutable"
        output = self.scratch / "dispatch.json"
        responses = [
            json.dumps({"sha": workflow_head}),
            "https://github.com/cataggar/wamr/actions/runs/123\n",
            json.dumps(
                {
                    "status": "completed",
                    "conclusion": "success",
                    "headSha": workflow_head,
                    "url": "https://github.com/cataggar/wamr/actions/runs/123",
                }
            ),
            json.dumps({"artifacts": []}),
            "https://github.com/cataggar/wamr/actions/runs/125\n",
            json.dumps(
                {
                    "status": "completed",
                    "conclusion": "success",
                    "headSha": workflow_head,
                    "url": "https://github.com/cataggar/wamr/actions/runs/125",
                }
            ),
            json.dumps({"artifacts": []}),
        ]
        with (
            mock.patch.object(
                cohort.subprocess,
                "check_output",
                side_effect=responses,
            ) as run,
            mock.patch.object(cohort.time, "sleep"),
            mock.patch.object(
                cohort.uuid,
                "uuid4",
                return_value=Namespace(hex="d" * 32),
            ),
        ):
            cohort.dispatch(
                Namespace(
                    baseline_sha=baseline,
                    candidate_sha=candidate,
                    purpose="candidate-evaluation",
                    profile="authoritative",
                    warmups=2,
                    samples=10,
                    runner_target="github-hosted",
                    runs=2,
                    training_runs=1,
                    max_in_flight=1,
                    timeout_seconds=3600,
                    output=output,
                    repository="cataggar/wamr",
                    workflow="wasi-thread-bench.yml",
                    workflow_ref=workflow_ref,
                    poll_seconds=0,
                    lookup_attempts=1,
                    lookup_seconds=0,
                )
            )
        dispatch_command = run.call_args_list[0].args[0]
        self.assertEqual(
            dispatch_command,
            [
                "gh",
                "api",
                "repos/cataggar/wamr/commits/calibration%2F966-immutable",
            ],
        )
        dispatch_command = run.call_args_list[1].args[0]
        self.assertEqual(
            dispatch_command[dispatch_command.index("--ref") + 1],
            workflow_ref,
        )
        self.assertEqual(
            {
                dispatch_command[index + 1]
                for index, value in enumerate(dispatch_command)
                if value == "-f"
            },
            {
                f"baseline_sha={baseline}",
                f"candidate_sha={candidate}",
                "purpose=candidate-evaluation",
                "profile=authoritative",
                "warmups=2",
                "samples=10",
                "runner_target=github-hosted",
                f"cohort_id={'d' * 32}",
                "cohort_sequence=1",
                "cohort_partition=training",
            },
        )
        state = json.loads(output.read_text(encoding="UTF-8"))
        self.assertEqual(state["workflow_head_sha"], workflow_head)
        self.assertEqual(
            [item["partition"] for item in state["split"]["assignments"]],
            ["training", "holdout"],
        )

        mismatch_output = self.scratch / "dispatch-mismatch.json"
        mismatch_responses = [
            json.dumps({"sha": workflow_head}),
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
            self.assertRaisesRegex(
                bench.HarnessError, "does not match immutable workflow head"
            ),
        ):
            cohort.dispatch(
                Namespace(
                    baseline_sha=baseline,
                    candidate_sha=candidate,
                    purpose="candidate-evaluation",
                    profile="authoritative",
                    warmups=2,
                    samples=10,
                    runner_target="github-hosted",
                    runs=2,
                    training_runs=1,
                    max_in_flight=1,
                    timeout_seconds=3600,
                    output=mismatch_output,
                    repository="cataggar/wamr",
                    workflow="wasi-thread-bench.yml",
                    workflow_ref=workflow_ref,
                    poll_seconds=0,
                    lookup_attempts=1,
                    lookup_seconds=0,
                )
            )

    def test_cohort_dispatch_rejects_mutable_or_incompatible_targets(self) -> None:
        common = {
            "baseline_sha": "b" * 40,
            "candidate_sha": "a" * 40,
            "purpose": "candidate-evaluation",
            "profile": "authoritative",
            "warmups": 2,
            "samples": 10,
            "runner_target": "github-hosted",
            "repository": "cataggar/wamr",
            "workflow": "wasi-thread-bench.yml",
            "workflow_ref": "main",
            "runs": 2,
            "training_runs": 1,
            "max_in_flight": 1,
            "timeout_seconds": 3600,
            "poll_seconds": 0,
            "lookup_attempts": 1,
            "lookup_seconds": 0,
        }
        for field, value, message in (
            ("baseline_sha", "B" * 40, "lowercase"),
            ("samples", 3, "even"),
            ("candidate_sha", "b" * 40, "distinct"),
            ("runner_target", "trusted-calibration", "noise calibration only"),
            ("timeout_seconds", 0, "timeout"),
        ):
            args = Namespace(**dict(common, **{field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(
                bench.HarnessError, message
            ):
                cohort.validate_dispatch_options(args)

        trusted = Namespace(
            **dict(
                common,
                baseline_sha="a" * 40,
                candidate_sha="a" * 40,
                purpose="noise-calibration",
                runner_target="trusted-calibration",
                max_in_flight=3,
            )
        )
        with self.assertRaisesRegex(bench.HarnessError, "cannot exceed 2"):
            cohort.validate_dispatch_options(trusted)

    def test_cohort_dispatch_times_out_queued_run_and_retains_state(self) -> None:
        output = self.scratch / "dispatch-timeout.json"
        with (
            mock.patch.object(
                cohort.subprocess,
                "check_output",
                side_effect=[
                    json.dumps({"sha": "f" * 40}),
                    "https://github.com/cataggar/wamr/actions/runs/123\n",
                ],
            ),
            mock.patch.object(
                cohort.time,
                "monotonic",
                side_effect=[0, 0, 0, 0, 2],
            ),
            mock.patch.object(
                cohort.uuid,
                "uuid4",
                return_value=Namespace(hex="d" * 32),
            ),
            self.assertRaisesRegex(bench.HarnessError, "timed out"),
        ):
            cohort.dispatch(
                Namespace(
                    baseline_sha="b" * 40,
                    candidate_sha="a" * 40,
                    purpose="candidate-evaluation",
                    profile="authoritative",
                    warmups=2,
                    samples=10,
                    runner_target="github-hosted",
                    runs=2,
                    training_runs=1,
                    max_in_flight=1,
                    timeout_seconds=1,
                    output=output,
                    repository="cataggar/wamr",
                    workflow="wasi-thread-bench.yml",
                    workflow_ref="main",
                    poll_seconds=60,
                    lookup_attempts=1,
                    lookup_seconds=0,
                )
            )
        state = json.loads(output.read_text(encoding="UTF-8"))
        self.assertEqual(state["runs"][0]["status"], "queued")
        self.assertEqual(state["timeout_seconds"], 1)

    def test_cohort_api_field_errors_are_contextual(self) -> None:
        with self.assertRaisesRegex(
            bench.HarnessError, "workflow run 123.*status"
        ):
            cohort.parse_workflow_run({"headSha": "f" * 40}, 123)
        with self.assertRaisesRegex(
            bench.HarnessError, "artifact response for run 123.*artifacts"
        ):
            cohort.parse_artifacts({}, 123)
        args = Namespace(
            repository="cataggar/wamr",
            workflow="wasi-thread-bench.yml",
            workflow_ref="main",
            lookup_attempts=1,
            lookup_seconds=0,
        )
        with (
            mock.patch.object(
                cohort,
                "gh_json",
                return_value=[{"displayTitle": "incomplete"}],
            ),
            self.assertRaisesRegex(
                bench.HarnessError, "run list item 0.*databaseId"
            ),
        ):
            cohort.find_dispatched_run(args, "expected", "f" * 40)

    def test_cohort_dispatch_finds_exact_named_run_when_cli_has_no_url(self) -> None:
        args = Namespace(
            repository="cataggar/wamr",
            workflow="wasi-thread-bench.yml",
            workflow_ref="main",
            lookup_attempts=2,
            lookup_seconds=0,
        )
        run_name = f"WASI thread cohort-{'d' * 32}-1-training"
        with (
            mock.patch.object(
                cohort,
                "gh_json",
                side_effect=[
                    [],
                    [
                        {
                            "databaseId": 123,
                            "displayTitle": run_name,
                            "headSha": "f" * 40,
                            "url": (
                                "https://github.com/cataggar/wamr/"
                                "actions/runs/123"
                            ),
                        }
                    ],
                ],
            ),
            mock.patch.object(cohort.time, "sleep"),
        ):
            self.assertEqual(
                cohort.find_dispatched_run(args, run_name, "f" * 40),
                (
                    123,
                    "https://github.com/cataggar/wamr/actions/runs/123",
                ),
            )

    def test_workflow_self_hosted_label_is_manual_trusted_only(self) -> None:
        workflow = (
            ROOT / ".github/workflows/wasi-thread-bench.yml"
        ).read_text(encoding="UTF-8")
        self.assertEqual(workflow.count("wamr-temp-20260906"), 1)
        start = workflow.index("\n  trusted-calibration-x86:")
        end = workflow.index("\n  trusted-calibration-arm:", start)
        trusted_x86 = workflow[start:end]
        self.assertIn("github.event_name == 'workflow_dispatch'", trusted_x86)
        self.assertIn("inputs.runner_target == 'trusted-calibration'", trusted_x86)
        self.assertIn("inputs.purpose == 'noise-calibration'", trusted_x86)
        self.assertIn("github.ref == 'refs/heads/main'", trusted_x86)
        self.assertIn("runs-on: wamr-temp-20260906", trusted_x86)
        self.assertNotIn("runs-on: [", trusted_x86)
        self.assertNotIn("runs-on: self-hosted", trusted_x86)
        self.assertNotIn("pull_request", trusted_x86)
        self.assertNotIn("\n  push:", trusted_x86)

        hosted_start = workflow.index("\n  benchmark:")
        hosted_end = workflow.index("\n  trusted-calibration-x86:", hosted_start)
        hosted = workflow[hosted_start:hosted_end]
        self.assertIn("github.event_name != 'workflow_dispatch'", hosted)
        self.assertIn("inputs.runner_target == 'github-hosted'", hosted)
        self.assertNotIn("self-hosted", hosted)
        self.assertNotIn("wamr-temp-20260906", hosted)
        self.assertNotIn("github.workspace }}/wasi-thread-bench-out", workflow)
        self.assertEqual(
            workflow.count("Clean run-scoped benchmark output and caches"), 3
        )
        self.assertEqual(workflow.count('rm -rf -- "$EXPECTED_ROOT"'), 3)
        self.assertEqual(
            workflow.count('echo "WASI_THREAD_OUTPUT=$EXPECTED_ROOT/output"'),
            3,
        )
        for start_marker, end_marker in (
            ("\n  benchmark:", "\n  trusted-calibration-x86:"),
            ("\n  trusted-calibration-x86:", "\n  trusted-calibration-arm:"),
            ("\n  trusted-calibration-arm:", "\n  comment:"),
        ):
            block = workflow[
                workflow.index(start_marker) : workflow.index(
                    end_marker, workflow.index(start_marker) + 1
                )
            ]
            self.assertLess(
                block.index("Upload retained paired report"),
                block.index("Clean run-scoped benchmark output and caches"),
            )
        self.assertIn("push:\n    branches: [main]\n    paths:", workflow)
        self.assertIn(
            "Download retained x86 report\n        continue-on-error: true",
            workflow,
        )
        self.assertIn("prepare:\n    runs-on: ubuntu-22.04\n    timeout-minutes: 15", workflow)
        self.assertIn(
            "comment:\n    if: always() && github.event_name == 'pull_request'\n"
            "    needs: benchmark\n    runs-on: ubuntu-22.04\n"
            "    timeout-minutes: 10",
            workflow,
        )
        for input_name in (
            "baseline_sha:",
            "candidate_sha:",
            "purpose:",
            "profile:",
            "warmups:",
            "samples:",
            "runner_target:",
        ):
            self.assertIn(input_name, workflow)

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
