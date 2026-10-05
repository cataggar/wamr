import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_loop_passes as bench


class LoopPassBenchTests(unittest.TestCase):
    def test_ref_build_uses_its_declared_compiler_generation(self):
        with patch.object(Path, "read_text", return_value='.minimum_zig_version = "0.16.0"'):
            self.assertEqual(
                ["zig016", "build", "-Doptimize=ReleaseFast"],
                bench.zig_build_command(Path("baseline"), "ReleaseFast"),
            )
        with patch.object(Path, "read_text", return_value='.minimum_zig_version = "0.17.0"'):
            self.assertEqual(
                ["zig", "build", "-Doptimize=fast"],
                bench.zig_build_command(Path("target"), "ReleaseFast"),
            )
            self.assertEqual(
                ["zig", "build", "-Doptimize=safe"],
                bench.zig_build_command(Path("target"), "ReleaseSafe"),
            )

    def test_ref_build_refuses_unqualified_compiler_generation(self):
        with patch.object(Path, "read_text", return_value='.minimum_zig_version = "0.18.0"'):
            with self.assertRaisesRegex(ValueError, "Zig 0.16 or 0.17"):
                bench.zig_build_command(Path("target"), "ReleaseFast")

    def test_tracked_fixture_checksums_are_pinned(self):
        repo = Path(__file__).resolve().parents[1]
        fixtures = bench.resolve_fixtures(repo)
        self.assertEqual(
            [fixture.sha256 for fixture in bench.FIXTURES],
            [fixture.sha256 for fixture in fixtures],
        )

    def test_authoritative_profile_counts(self):
        self.assertEqual((2, 10), bench.resolve_counts("authoritative", None, None))
        self.assertEqual((0, 3), bench.resolve_counts("ci", None, None))

    def test_invalid_run_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            bench.resolve_counts("ci", None, 0)

    def test_min_speedup_parser(self):
        self.assertEqual(
            {"unroll4": 20.0, "iv_store": -2.0},
            bench.parse_min_speedups(["unroll4=20", "iv_store=-2"]),
        )
        with self.assertRaisesRegex(ValueError, "expected CASE=PCT"):
            bench.parse_min_speedups(["unknown=1"])

    def test_speedup_uses_elapsed_time_ratio(self):
        self.assertAlmostEqual(100.0, bench.speedup_pct([2.0, 2.0], [1.0, 1.0]))
        self.assertAlmostEqual(-50.0, bench.speedup_pct([1.0], [2.0]))


if __name__ == "__main__":
    unittest.main()
