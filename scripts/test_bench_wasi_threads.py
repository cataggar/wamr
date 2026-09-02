#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import shutil
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
    run_id: str = "1",
) -> dict:
    plan = {
        "profile": "authoritative",
        "warmups": 0,
        "samples": 1,
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
        "optimize": "ReleaseFast",
        "pairs": [],
    }
    plan["pairs"] = bench.expected_pair_specs_for_plan(plan)
    records = []
    summaries = []
    paired = []
    for pair in plan["pairs"]:
        pair_records = {}
        for order, condition in enumerate((pair["left"], pair["right"])):
            elapsed = 100 + order * 10
            record = {
                "pair_kind": pair["pair_kind"],
                "pair_key": pair["pair_key"],
                "pair_index": 0,
                "phase": "measure",
                "order": order,
                "condition": condition,
                "pair_left": pair["left"],
                "pair_right": pair["right"],
                "correct": True,
                "elapsed_ns": elapsed,
                "guest_elapsed_ns": elapsed,
                "raw_guest_elapsed_ns": elapsed + 1,
                "timing_overhead_ns": 1,
                "timing_overhead_ppm": 1,
                "host_wall_elapsed_ns": elapsed + 100,
                "host_wall_over_guest": (elapsed + 100) / elapsed,
                "metric_kind": (
                    "spawn-join-lifecycle"
                    if "spawn-join" in pair["pair_key"]
                    else "steady-state-kernel"
                ),
                "throughput_ops_per_second": 1.0,
                "per_thread_ops_per_second": 1.0,
            }
            records.append(record)
            pair_records[condition] = record
            summaries.append(
                {
                    "pair_kind": pair["pair_kind"],
                    "pair_key": pair["pair_key"],
                    "condition": condition,
                    "metric_kind": record["metric_kind"],
                    "elapsed": stats(elapsed, "samples_ns"),
                    "host_wall": stats(elapsed + 100, "samples_ns"),
                    "throughput": stats(1.0, "samples_ops_per_second"),
                    "per_thread_throughput": stats(
                        1.0, "samples_ops_per_second"
                    ),
                }
            )
        ratio = pair_records[pair["right"]]["elapsed_ns"] / pair_records[pair["left"]]["elapsed_ns"]
        paired.append(
            {
                "pair_kind": pair["pair_kind"],
                "pair_key": pair["pair_key"],
                "left": pair["left"],
                "right": pair["right"],
                "elapsed_right_over_left": stats(ratio, "samples"),
                "median_elapsed_delta_pct": (ratio - 1) * 100,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": bench.KIND,
        "metadata": {
            "commit": commit,
            "tracked_diff_sha256": "b" * 64,
            "build_source_sha256": "c" * 64,
            "collected_at": "2026-09-02T00:00:00+00:00",
            "platform_id": platform_id,
            "fixture_set_sha256": "d" * 64,
            "plan_sha256": cache_key(plan),
            "host": {
                "system": "Linux",
                "machine": machine,
                "github_run_id": run_id,
            },
            "tools": {},
            "fixtures": {},
        },
        "plan": plan,
        "records": records,
        "summaries": summaries,
        "paired_summaries": paired,
        "budget": {"status": "disabled", "path": None, "failures": []},
    }
    bench.validate_report(report)
    return report


def complete_budget(report: dict) -> dict:
    platform_id = report["metadata"]["platform_id"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-thread-benchmark-budget",
        "calibrated": True,
        "calibration_requirements": {
            "minimum_reports_per_platform": 20,
            "required_profile": "authoritative",
            "required_platforms": [platform_id],
        },
        "cohort": {
            "baseline_commit": report["metadata"]["commit"],
            "baseline_build_source_sha256": report["metadata"]["build_source_sha256"],
            "fixture_set_sha256": report["metadata"]["fixture_set_sha256"],
            "plan_sha256": report["metadata"]["plan_sha256"],
            "profile": report["plan"]["profile"],
            "report_count_by_platform": {platform_id: 20},
        },
        "platforms": {
            platform_id: {
                "host_system": report["metadata"]["host"]["system"],
                "host_machine": report["metadata"]["host"]["machine"],
                "pairs": [
                    {
                        "pair_key": item["pair_key"],
                        "left": item["left"],
                        "right": item["right"],
                        "max_median_elapsed_delta_pct": 100.0,
                    }
                    for item in report["plan"]["pairs"]
                ],
                "scenarios": [
                    {
                        "pair_key": item["pair_key"],
                        "condition": item["condition"],
                        "metric_kind": item["metric_kind"],
                        "min_median_ops_per_second": 0.1,
                    }
                    for item in report["summaries"]
                ],
            }
        },
    }


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
        self.assertEqual((args.warmups, args.samples), (1, 3))
        self.assertEqual(args.thread_counts, (1, 4, 8))
        self.assertEqual(args.modes, "aot")
        self.assertEqual(args.min_interval_ms, 100)
        with self.assertRaises(SystemExit):
            bench.parse_args(["--thread-counts", "1,3"])

    def test_pair_direction_never_depends_on_condition_sorting(self) -> None:
        records = []

        def measure(condition, fields):
            return {
                **fields,
                "elapsed_ns": 10 if condition == "z-baseline" else 20,
                "guest_elapsed_ns": 10 if condition == "z-baseline" else 20,
                "host_wall_elapsed_ns": 30,
                "correct": True,
            }

        bench.collect_pair(
            records=records,
            pair_kind="test",
            pair_key="pair",
            left="z-baseline",
            right="a-target",
            warmups=0,
            samples=1,
            measure=measure,
        )
        result = bench.paired_summaries(records)[0]
        self.assertEqual(result["left"], "z-baseline")
        self.assertEqual(result["right"], "a-target")
        self.assertEqual(result["elapsed_right_over_left"]["median"], 2.0)
        self.assertEqual(
            alternating_pair_order(1, "left", "right"), ("right", "left")
        )

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

    def test_cache_key_is_canonical_and_configuration_sensitive(self) -> None:
        left = {"target": "native", "threads": True, "mode": "aot"}
        right = {"mode": "aot", "threads": True, "target": "native"}
        self.assertEqual(cache_key(left), cache_key(right))
        self.assertNotEqual(cache_key(left), cache_key(dict(left, threads=False)))

    def test_report_rejects_missing_scenario_and_direction_corruption(self) -> None:
        report = make_report()
        missing = copy.deepcopy(report)
        missing["records"] = missing["records"][2:]
        with self.assertRaisesRegex(BenchmarkDataError, "missing planned pairs"):
            bench.validate_report(missing)

        corrupt = copy.deepcopy(report)
        corrupt["plan"]["pairs"][0]["left"] = "renamed"
        corrupt["metadata"]["plan_sha256"] = cache_key(corrupt["plan"])
        with self.assertRaisesRegex(BenchmarkDataError, "incomplete"):
            bench.validate_report(corrupt)

    def test_budget_rejects_empty_partial_unknown_duplicate_and_direction(self) -> None:
        report = make_report()
        budget = complete_budget(report)
        loaded = bench.load_budget(self.write_budget(budget), report)
        self.assertFalse(bench.evaluate_budget(
            loaded, report["summaries"], report["paired_summaries"]
        ))
        uncalibrated = copy.deepcopy(budget)
        uncalibrated["calibrated"] = False
        with self.assertRaisesRegex(bench.HarnessError, "not calibrated"):
            bench.load_budget(self.write_budget(uncalibrated), report)

        cases = {}
        value = copy.deepcopy(budget)
        value["platforms"] = {}
        cases["empty"] = value
        value = copy.deepcopy(budget)
        value["platforms"][report["metadata"]["platform_id"]]["pairs"].pop()
        cases["partial"] = value
        value = copy.deepcopy(budget)
        value["platforms"][report["metadata"]["platform_id"]]["pairs"][0]["pair_key"] = "unknown"
        cases["unknown"] = value
        value = copy.deepcopy(budget)
        value["platforms"][report["metadata"]["platform_id"]]["pairs"].append(
            copy.deepcopy(value["platforms"][report["metadata"]["platform_id"]]["pairs"][0])
        )
        cases["duplicate"] = value
        value = copy.deepcopy(budget)
        pair = value["platforms"][report["metadata"]["platform_id"]]["pairs"][0]
        pair["left"], pair["right"] = pair["right"], pair["left"]
        cases["direction"] = value
        for label, corrupt in cases.items():
            with self.subTest(label=label), self.assertRaises(bench.HarnessError):
                bench.load_budget(self.write_budget(corrupt), report)

    def test_budget_rejects_identity_count_profile_and_host_mismatch(self) -> None:
        report = make_report()
        budget = complete_budget(report)
        mutations = (
            ("commit", lambda value: value["cohort"].__setitem__("baseline_commit", "bad")),
            ("source", lambda value: value["cohort"].__setitem__("baseline_build_source_sha256", "bad")),
            ("fixture", lambda value: value["cohort"].__setitem__("fixture_set_sha256", "f" * 64)),
            ("plan", lambda value: value["cohort"].__setitem__("plan_sha256", "f" * 64)),
            ("profile", lambda value: value["cohort"].__setitem__("profile", "smoke")),
            ("count", lambda value: value["cohort"]["report_count_by_platform"].__setitem__(report["metadata"]["platform_id"], 19)),
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
            '"calibration_requirements":{},"cohort":{},"platforms":{}}'
        )
        with self.assertRaisesRegex(bench.HarnessError, "duplicate"):
            bench.load_budget(self.write_budget(duplicate_json), report)

    def test_cohort_rejects_mixed_identity_and_duplicate_run_ids(self) -> None:
        x86 = make_report(run_id="100")
        arm = make_report(
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

        mixed = copy.deepcopy(arm)
        mixed["metadata"]["commit"] = "e" * 40
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

    def test_cohort_dispatch_uses_immutable_sha_as_workflow_ref(self) -> None:
        target = "a" * 40
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
                    poll_seconds=0,
                )
            )
        dispatch_command = run.call_args_list[0].args[0]
        self.assertEqual(
            dispatch_command[dispatch_command.index("--ref") + 1],
            target,
        )

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


if __name__ == "__main__":
    unittest.main()
