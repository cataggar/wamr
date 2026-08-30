#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_coremark


VALID_OUTPUT = """\
Iterations/Sec   : 12345.5
[0]crcfinal      : 0x33ff
Correct operation validated. See README.md for run and reporting rules.
"""


class BenchCoremarkTests(unittest.TestCase):
    def test_tracked_fixture_checksum_is_pinned(self):
        repo = Path(__file__).resolve().parents[1]
        fixture, digest = bench_coremark.resolve_fixture(
            repo, bench_coremark.DEFAULT_FIXTURE
        )
        self.assertEqual(
            fixture, (repo / bench_coremark.DEFAULT_FIXTURE).resolve()
        )
        self.assertEqual(digest, bench_coremark.DEFAULT_FIXTURE_SHA256)

    def test_parse_requires_crc_validation(self):
        self.assertEqual(
            bench_coremark.parse_coremark_output(VALID_OUTPUT, "test"), 12345.5
        )
        with self.assertRaisesRegex(RuntimeError, "CRC-validated"):
            bench_coremark.parse_coremark_output(
                "Iterations/Sec : 12345.5\nERROR! bad crc\n", "test"
            )
        with self.assertRaisesRegex(RuntimeError, "CRC-validated"):
            bench_coremark.parse_coremark_output(
                "Iterations/Sec : 12345.5\n", "test"
            )

    def test_parse_rejects_ambiguous_throughput(self):
        with self.assertRaisesRegex(RuntimeError, "2 Iterations/Sec"):
            bench_coremark.parse_coremark_output(
                VALID_OUTPUT + "Iterations/Sec : 1\n", "test"
            )

    def test_profile_defaults_and_overrides(self):
        self.assertEqual(
            bench_coremark.resolve_counts("authoritative", None, None), (2, 10)
        )
        self.assertEqual(bench_coremark.resolve_counts("ci", None, None), (0, 3))
        self.assertEqual(
            bench_coremark.resolve_counts("ci", 1, 4), (1, 4)
        )
        with self.assertRaises(ValueError):
            bench_coremark.resolve_counts("authoritative", -1, 10)
        with self.assertRaises(ValueError):
            bench_coremark.resolve_counts("authoritative", 2, 0)
        self.assertEqual(
            bench_coremark.profile_label("authoritative", None, None),
            "authoritative",
        )
        self.assertEqual(
            bench_coremark.profile_label("authoritative", None, 3),
            "authoritative (overridden)",
        )

    @mock.patch.object(bench_coremark, "host_info", return_value="_Host: test_")
    def test_report_distinguishes_wasmtime_versions_and_lists_samples(self, _):
        results = [
            bench_coremark.EngineResult(
                "WAMR", "origin/main (aaaa)", "ReleaseFast", [50.0, 52.0]
            ),
            bench_coremark.EngineResult(
                "WAMR", "HEAD (bbbb)", "ReleaseFast", [60.0, 62.0]
            ),
            bench_coremark.EngineResult(
                "Wasmtime historical pin",
                "44.0.1 (/pinned/wasmtime)",
                "default JIT",
                [100.0, 102.0],
            ),
            bench_coremark.EngineResult(
                "Wasmtime caller-selected",
                "48.0.1 (/current/wasmtime)",
                "default JIT",
                [120.0, 122.0],
            ),
        ]
        report = bench_coremark.render_table(
            results,
            profile="authoritative",
            warmups=2,
            runs=2,
            fixture=Path("coremark.wasm"),
            fixture_sha="abc",
        )
        self.assertIn("Median", report)
        self.assertIn("50.0, 52.0", report)
        self.assertIn("44.0.1 (/pinned/wasmtime)", report)
        self.assertIn("48.0.1 (/current/wasmtime)", report)
        self.assertIn("WAMR target / Wasmtime historical pin", report)
        self.assertIn("WAMR target / Wasmtime caller-selected", report)

    @mock.patch.object(
        bench_coremark, "run", return_value="wasmtime 44.0.1 (abcdef)"
    )
    def test_pinned_wasmtime_version_is_enforced(self, _):
        self.assertEqual(
            bench_coremark.validate_pinned_wasmtime(Path("/bin/wasmtime")),
            "44.0.1",
        )
        with mock.patch.object(
            bench_coremark, "run", return_value="wasmtime 48.0.1 (abcdef)"
        ):
            with self.assertRaisesRegex(RuntimeError, "must be 44.0.1"):
                bench_coremark.validate_pinned_wasmtime(Path("/bin/wasmtime"))


if __name__ == "__main__":
    unittest.main()
