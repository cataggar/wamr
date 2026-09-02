#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_wasi_threads as bench  # noqa: E402
from benchmark_schema import (  # noqa: E402
    BenchmarkDataError,
    alternating_pair_order,
    cache_key,
)


class ThreadBenchmarkTests(unittest.TestCase):
    scratch = ROOT / "zig-out" / "test-bench-wasi-threads"

    def setUp(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def test_cli_parsing_and_profiles(self) -> None:
        args = bench.parse_args(
            [
                "--profile",
                "smoke",
                "--thread-counts",
                "1,4,8",
                "--modes",
                "aot",
                "--no-budget",
            ]
        )
        self.assertEqual((args.warmups, args.samples), (1, 3))
        self.assertEqual(args.thread_counts, (1, 4, 8))
        self.assertEqual(args.modes, "aot")
        with self.assertRaises(SystemExit):
            bench.parse_args(["--thread-counts", "1,3"])

    def test_sample_pairing_alternates_and_is_complete(self) -> None:
        self.assertEqual(
            alternating_pair_order(0, "left", "right"), ("left", "right")
        )
        self.assertEqual(
            alternating_pair_order(1, "left", "right"), ("right", "left")
        )
        records = []

        def measure(condition, fields):
            return {
                **fields,
                "elapsed_ns": 10 if condition == "left" else 11,
                "throughput_ops_per_second": 1.0,
                "per_thread_ops_per_second": 1.0,
                "correct": True,
            }

        bench.collect_pair(
            records=records,
            pair_kind="test",
            pair_key="pair",
            left="left",
            right="right",
            warmups=1,
            samples=2,
            measure=measure,
        )
        self.assertEqual(len(records), 6)
        self.assertEqual(
            [record["condition"] for record in records[:4]],
            ["left", "right", "right", "left"],
        )

    def test_guest_result_rejects_corruption_and_mismatch(self) -> None:
        expected = bench.expected_result("atomic", 4, 10)
        self.assertEqual(
            bench.parse_guest_result(json.dumps(expected) + "\n", expected),
            expected,
        )
        corrupted = dict(expected, checksum=39)
        with self.assertRaisesRegex(bench.HarnessError, "mismatch"):
            bench.parse_guest_result(json.dumps(corrupted), expected)
        with self.assertRaisesRegex(bench.HarnessError, "one guest JSON"):
            bench.parse_guest_result("{}\n{}\n", expected)

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
        changed = dict(left, threads=False)
        self.assertNotEqual(cache_key(left), cache_key(changed))

    def test_schema_rejects_corrupt_and_incomplete_pairs(self) -> None:
        record = {
            "pair_kind": "test",
            "pair_key": "pair",
            "pair_index": 0,
            "phase": "measure",
            "condition": "left",
            "correct": True,
            "elapsed_ns": 1,
        }
        document = {
            "schema_version": 1,
            "kind": bench.KIND,
            "metadata": {
                "commit": "a" * 40,
                "tracked_diff_sha256": "b" * 64,
                "build_source_sha256": "c" * 64,
                "collected_at": "now",
                "host": {},
                "tools": {},
                "fixtures": {},
            },
            "plan": {"warmups": 0, "samples": 1},
            "records": [record],
            "summaries": [{}],
        }
        with self.assertRaisesRegex(BenchmarkDataError, "incomplete"):
            bench.validate_report(document)
        document["schema_version"] = 99
        with self.assertRaisesRegex(BenchmarkDataError, "schema_version"):
            bench.validate_report(document)

    def test_fixture_hashes_are_pinned(self) -> None:
        fixtures = bench.resolve_fixtures(ROOT)
        self.assertEqual(fixtures["single"]["sha256"], bench.FIXTURES["single"]["sha256"])
        self.assertEqual(
            fixtures["threaded"]["sha256"], bench.FIXTURES["threaded"]["sha256"]
        )
        schema = json.loads(
            (
                ROOT
                / "tests"
                / "benchmarks"
                / "wasi-threads"
                / "report.schema.json"
            ).read_text(encoding="UTF-8")
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(schema["properties"]["kind"]["const"], bench.KIND)


if __name__ == "__main__":
    unittest.main()
