#!/usr/bin/env python3

import copy
import io
import json
import os
import shutil
import sys
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_coremark
import native_benchmark


REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github/workflows/coremark-aarch64.yml"
PROFILE_SCRIPT = REPO / "scripts/profile_coremark_aarch64.py"
VALID_OUTPUT = """\
2K performance run parameters for coremark.
Iterations/Sec   : 12345.5
Iterations       : 400000
[0]crcfinal      : 0x33ff
Correct operation validated. See README.md for run and reporting rules.
"""


def wamr_identity(ref: str, source_sha: str, digest: str) -> dict:
    return {
        "type": "wamr",
        "source": {"ref": ref, "sha": source_sha},
        "optimize": "ReleaseFast",
        "runtime": {"name": "wamr", "path": "/wamr", "sha256": digest},
        "compiler": {"name": "wamrc", "path": "/wamrc", "sha256": digest},
        "module": {"format": "cwasm", "path": "/coremark.cwasm", "sha256": digest},
    }


def wasmtime_identity(digest: str) -> dict:
    return {
        "type": "wasmtime",
        "channel": "historical-pin",
        "version": bench_coremark.PINNED_WASMTIME_VERSION,
        "runtime": {"name": "wasmtime", "path": "/wasmtime", "sha256": digest},
    }


def report_provenance() -> dict:
    return {
        "report_id": "12345678-1234-5678-1234-567812345678",
        "generated_at": "2026-09-08T00:00:00+00:00",
        "producer": {
            "source_sha": "c" * 40,
            "script": {
                "path": "scripts/bench_coremark.py",
                "sha256": "d" * 64,
            },
        },
        "execution": {"provider": "local", "run_id": "test-run"},
    }


class BenchCoremarkTests(unittest.TestCase):
    def authoritative_report(self):
        host = bench_coremark.HostIdentity(
            "aarch64", 4, "Neoverse-N2", "runner", "boot-id"
        )
        affinity = bench_coremark.AffinityInfo((0, 1, 2, 3), 0, "/usr/bin/taskset")
        target_identity = wamr_identity("target", "b" * 40, "2" * 64)
        records = []
        samples = {"baseline": [], "target": [], "wasmtime": []}
        position = 0
        for phase, count in (("warmup", 2), ("measured", 10)):
            ordinals = {"baseline": 0, "target": 0, "wasmtime": 0}
            for key in bench_coremark.counterbalanced_order(
                ["baseline", "target", "wasmtime"], count
            ):
                position += 1
                ordinals[key] += 1
                record = bench_coremark.SampleRecord(
                    key,
                    (
                        "Wasmtime"
                        if key == "wasmtime"
                        else f"WAMR {key}"
                    ),
                    phase,
                    ordinals[key],
                    position,
                    "2026-09-08T00:00:00+00:00",
                    "2026-09-08T00:00:01+00:00",
                    1.0,
                    {"baseline": 100.0, "target": 110.0, "wasmtime": 220.0}[key],
                    bench_coremark.EXPECTED_ITERATIONS,
                )
                records.append(record)
                samples[key].append(record)
        results = [
            bench_coremark.EngineResult(
                "WAMR",
                "baseline",
                "ReleaseFast",
                [100.0] * 10,
                samples=samples["baseline"],
                identity=wamr_identity("baseline", "a" * 40, "1" * 64),
            ),
            bench_coremark.EngineResult(
                "WAMR",
                "target",
                "ReleaseFast",
                [110.0] * 10,
                samples=samples["target"],
                identity=target_identity,
            ),
            bench_coremark.EngineResult(
                "Wasmtime historical pin",
                "44.0.1",
                "default JIT",
                [220.0] * 10,
                samples=samples["wasmtime"],
                identity=wasmtime_identity("3" * 64),
            ),
        ]
        report = bench_coremark.build_json_report(
            results,
            profile="authoritative",
            warmups=2,
            runs=10,
            fixture=Path("coremark.wasm"),
            fixture_sha="f" * 64,
            host=host,
            schedule_records=records,
            affinity=affinity,
            provenance=report_provenance(),
        )
        return report, host, affinity, target_identity

    def test_tracked_fixture_checksum_is_pinned(self):
        fixture, digest = bench_coremark.resolve_fixture(
            REPO, bench_coremark.DEFAULT_FIXTURE
        )
        self.assertEqual(
            fixture, (REPO / bench_coremark.DEFAULT_FIXTURE).resolve()
        )
        self.assertEqual(digest, bench_coremark.DEFAULT_FIXTURE_SHA256)

    def test_parse_requires_crc_validation(self):
        parsed = bench_coremark.parse_coremark_output(VALID_OUTPUT, "test")
        self.assertEqual(parsed.throughput, 12345.5)
        self.assertEqual(parsed.iterations, 400000)
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

    def test_parse_rejects_invalid_fixed_workload(self):
        with self.assertRaisesRegex(RuntimeError, "0 Iterations fields"):
            bench_coremark.parse_coremark_output(
                VALID_OUTPUT.replace("Iterations       : 400000\n", ""),
                "test",
            )
        with self.assertRaisesRegex(RuntimeError, "2 Iterations fields"):
            bench_coremark.parse_coremark_output(
                VALID_OUTPUT + "Iterations : 400000\n", "test"
            )
        with self.assertRaisesRegex(RuntimeError, "malformed Iterations"):
            bench_coremark.parse_coremark_output(
                VALID_OUTPUT.replace("Iterations       : 400000", "Iterations : auto"),
                "test",
            )
        with self.assertRaisesRegex(RuntimeError, "self-calibration is forbidden"):
            bench_coremark.parse_coremark_output(
                VALID_OUTPUT.replace("400000", "300000"), "test"
            )
        with self.assertRaisesRegex(RuntimeError, "2K performance"):
            bench_coremark.parse_coremark_output(
                VALID_OUTPUT.replace(
                    "2K performance run parameters for coremark.\n", ""
                ),
                "test",
            )
        with self.assertRaisesRegex(RuntimeError, "CRC-validated"):
            bench_coremark.parse_coremark_output(
                VALID_OUTPUT + "Correct operation validated.\n", "test"
            )

    def test_fixed_guest_args_are_applied_to_both_engines(self):
        self.assertEqual(
            ("0", "0", "0", "200000", "0"),
            bench_coremark.coremark_guest_args(
                bench_coremark.PROFILE_ITERATIONS["ci"]
            ),
        )
        with (
            mock.patch.object(bench_coremark, "run", return_value=""),
            mock.patch.object(bench_coremark, "worktree_env", return_value={}),
            mock.patch.object(bench_coremark, "sha256_file", return_value="abc"),
        ):
            wamr = bench_coremark.prepare_wamr(
                Path("/worktree"),
                "HEAD",
                "a" * 40,
                Path("/fixture.wasm"),
                "ReleaseFast",
            )
            wasmtime = bench_coremark.prepare_wasmtime(
                Path("/wasmtime"),
                "Wasmtime",
                "44.0.1",
                Path("/fixture.wasm"),
            )
        self.assertEqual(
            list(bench_coremark.COREMARK_GUEST_ARGS),
            wamr.cmd[-5:],
        )
        self.assertEqual(
            list(bench_coremark.COREMARK_GUEST_ARGS),
            wasmtime.cmd[-5:],
        )

    def test_authoritative_affinity_is_selected_and_verified(self):
        with (
            mock.patch.object(
                bench_coremark.os,
                "sched_getaffinity",
                return_value={4, 7},
            ),
            mock.patch.object(
                bench_coremark.shutil,
                "which",
                return_value="/usr/bin/taskset",
            ),
            mock.patch.object(bench_coremark, "run", return_value="4\n") as run,
        ):
            affinity = bench_coremark.select_cpu_affinity()
        self.assertEqual((4, 7), affinity.allowed_cpus)
        self.assertEqual(4, affinity.selected_cpu)
        self.assertEqual("/usr/bin/taskset", run.call_args.args[0][0])
        self.assertEqual(
            ["/usr/bin/taskset", "--cpu-list", "4", "engine"],
            bench_coremark.apply_affinity(["engine"], affinity),
        )

    def test_counterbalanced_order_is_abba(self):
        self.assertEqual(
            ["A", "B", "B", "A"],
            bench_coremark.counterbalanced_order(["A", "B"], 2),
        )
        order = bench_coremark.counterbalanced_order(["A", "B"], 10)
        self.assertEqual(10, order.count("A"))
        self.assertEqual(10, order.count("B"))
        self.assertEqual(["A", "B", "B", "A"], order[:4])

    def test_schedule_groups_samples_and_recomputes_ratio(self):
        def output(value):
            return VALID_OUTPUT.replace("12345.5", str(value))

        engines = [
            bench_coremark.PreparedEngine(
                "A",
                "WAMR",
                "commit",
                "ReleaseFast",
                ["wamr"],
                Path("."),
                {},
                400000,
            ),
            bench_coremark.PreparedEngine(
                "B",
                "Wasmtime",
                "44.0.1",
                "default JIT",
                ["wasmtime"],
                Path("."),
                {},
                400000,
            ),
        ]
        with mock.patch.object(
            bench_coremark,
            "run",
            side_effect=[output(10), output(20), output(22), output(12)],
        ):
            results, records = bench_coremark.measure_prepared_engines(
                engines,
                warmups=0,
                runs=2,
                affinity=None,
            )
        self.assertEqual([10.0, 12.0], results["A"].values)
        self.assertEqual([20.0, 22.0], results["B"].values)
        self.assertEqual(["A", "B", "B", "A"], [r.engine_key for r in records])
        report = bench_coremark.build_json_report(
            [
                results["A"],
                results["A"],
                results["B"],
            ],
            profile="authoritative",
            warmups=0,
            runs=2,
            fixture=Path("fixture.wasm"),
            fixture_sha="abc",
            host=bench_coremark.HostIdentity(
                "aarch64", 4, "Neoverse-N2", "runner", "boot"
            ),
            schedule_records=records,
            affinity=None,
            provenance=report_provenance(),
        )
        self.assertAlmostEqual(11 / 21, report["ratios"][0]["median_ratio"])

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

    def test_report_distinguishes_wasmtime_versions_and_lists_samples(self):
        results = [
            bench_coremark.EngineResult(
                "WAMR", "origin/main (aaaa)", "ReleaseFast", [50.0, 52.0]
            ),
            bench_coremark.EngineResult(
                "WAMR", "HEAD (bbbb)", "ReleaseFast", [60.0, 62.0]
            ),
            bench_coremark.EngineResult(
                "Wasmtime historical pin",
                "44.0.1 (sha256:abc; /pinned/wasmtime)",
                "default JIT",
                [100.0, 102.0],
            ),
            bench_coremark.EngineResult(
                "Wasmtime caller-selected",
                "48.0.1 (sha256:def; /current/wasmtime)",
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
            host=bench_coremark.HostIdentity(
                "aarch64", 4, "Neoverse-N2", "runner", "boot-id"
            ),
        )
        self.assertIn("Median", report)
        self.assertIn("50.0, 52.0", report)
        self.assertIn("44.0.1 (sha256:abc; /pinned/wasmtime)", report)
        self.assertIn("48.0.1 (sha256:def; /current/wasmtime)", report)
        self.assertIn("WAMR target / Wasmtime historical pin", report)
        self.assertIn("WAMR target / Wasmtime caller-selected", report)
        self.assertIn("Median iter/s ratio", report)
        self.assertIn("0.603960×", report)
        self.assertIn("CRC validation", report)
        self.assertIn("Neoverse-N2", report)
        self.assertIn("host fingerprint", report)

    def test_published_ratio_uses_full_precision_raw_statistics(self):
        wamr = [11702.9216445]
        wasmtime = [24350.89644]
        raw_ratio = 0.48059510553690316
        rounded_display_ratio = 11702.9 / 24350.9
        median_ratio, _ = bench_coremark.compute_ratio_stats(wamr, wasmtime)
        self.assertAlmostEqual(raw_ratio, median_ratio, places=15)
        self.assertNotEqual(round(raw_ratio, 9), round(rounded_display_ratio, 9))

        results = [
            bench_coremark.EngineResult(
                "WAMR", "main", "ReleaseFast", wamr
            ),
            bench_coremark.EngineResult(
                "WAMR", "main", "ReleaseFast", wamr
            ),
            bench_coremark.EngineResult(
                "Wasmtime historical pin",
                "44.0.1",
                "default JIT",
                wasmtime,
            ),
        ]
        report = bench_coremark.render_table(
            results,
            profile="authoritative",
            warmups=0,
            runs=1,
            fixture=Path("coremark.wasm"),
            fixture_sha="abc",
            host=bench_coremark.HostIdentity(
                "aarch64", 4, "Neoverse-N2", "runner", "boot"
            ),
        )
        self.assertIn("0.480595×", report)
        payload = bench_coremark.build_json_report(
            results,
            profile="authoritative",
            warmups=0,
            runs=1,
            fixture=Path("coremark.wasm"),
            fixture_sha="abc",
            host=bench_coremark.HostIdentity(
                "aarch64", 4, "Neoverse-N2", "runner", "boot"
            ),
            schedule_records=[],
            affinity=None,
            provenance=report_provenance(),
        )
        self.assertEqual(raw_ratio, payload["ratios"][0]["median_ratio"])

    def test_report_rejects_missing_samples(self):
        results = [
            bench_coremark.EngineResult(
                "WAMR", "origin/main (aaaa)", "ReleaseFast", [50.0]
            ),
            bench_coremark.EngineResult(
                "WAMR", "HEAD (bbbb)", "ReleaseFast", [60.0, 62.0]
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "produced 1 measured samples"):
            bench_coremark.render_table(
                results,
                profile="authoritative",
                warmups=2,
                runs=2,
                fixture=Path("coremark.wasm"),
                fixture_sha="abc",
                host=bench_coremark.HostIdentity(
                    "aarch64", 4, "Neoverse-N2", "runner", "boot-id"
                ),
            )

    def test_current_benchmark_profile_identity_is_validated(self):
        report, host, affinity, target_identity = self.authoritative_report()
        linkage = bench_coremark.validate_benchmark_profile_match(
            report,
            expected_arch="aarch64",
            fixture_sha="f" * 64,
            host=host,
            affinity=affinity,
            wamr_source_sha="b" * 40,
            wamr_optimize="ReleaseFast",
            wamr_runtime_sha=target_identity["runtime"]["sha256"],
            wamr_compiler_sha=target_identity["compiler"]["sha256"],
            wamr_module_sha=target_identity["module"]["sha256"],
            wasmtime_version_value=bench_coremark.PINNED_WASMTIME_VERSION,
            wasmtime_runtime_sha="3" * 64,
            producer_source_sha="c" * 40,
            producer_script_sha="d" * 64,
            current_execution={"provider": "local", "run_id": "test-run"},
        )
        self.assertEqual(
            report["provenance"]["report_id"], linkage["report_id"]
        )
        self.assertEqual("b" * 40, linkage["target"]["identity"]["source"]["sha"])
        baseline = next(
            engine
            for engine in report["engines"]
            if engine["role"] == "wamr-baseline"
        )
        baseline_linkage = bench_coremark.validate_benchmark_profile_match(
            report,
            expected_arch="aarch64",
            fixture_sha="f" * 64,
            host=host,
            affinity=affinity,
            wamr_source_sha="a" * 40,
            wamr_optimize="ReleaseFast",
            wamr_runtime_sha=baseline["identity"]["runtime"]["sha256"],
            wamr_compiler_sha=baseline["identity"]["compiler"]["sha256"],
            wamr_module_sha=baseline["identity"]["module"]["sha256"],
            wasmtime_version_value=bench_coremark.PINNED_WASMTIME_VERSION,
            wasmtime_runtime_sha="3" * 64,
            producer_source_sha="c" * 40,
            producer_script_sha="d" * 64,
            current_execution={"provider": "local", "run_id": "test-run"},
            benchmark_role="wamr-baseline",
        )
        self.assertEqual("wamr-baseline", baseline_linkage["selected_role"])
        self.assertNotIn("target", baseline_linkage)

    def test_benchmark_profile_mismatches_fail_closed(self):
        report, host, affinity, target_identity = self.authoritative_report()
        kwargs = {
            "expected_arch": "aarch64",
            "fixture_sha": "f" * 64,
            "host": host,
            "affinity": affinity,
            "wamr_source_sha": "b" * 40,
            "wamr_optimize": "ReleaseFast",
            "wamr_runtime_sha": target_identity["runtime"]["sha256"],
            "wamr_compiler_sha": target_identity["compiler"]["sha256"],
            "wamr_module_sha": target_identity["module"]["sha256"],
            "wasmtime_version_value": bench_coremark.PINNED_WASMTIME_VERSION,
            "wasmtime_runtime_sha": "3" * 64,
            "producer_source_sha": "c" * 40,
            "producer_script_sha": "d" * 64,
            "current_execution": {"provider": "local", "run_id": "test-run"},
        }
        with self.assertRaisesRegex(RuntimeError, "WAMR source sha mismatch"):
            bench_coremark.validate_benchmark_profile_match(
                report, **{**kwargs, "wamr_source_sha": "9" * 40}
            )
        with self.assertRaisesRegex(RuntimeError, "runtime sha256 mismatch"):
            bench_coremark.validate_benchmark_profile_match(
                report, **{**kwargs, "wamr_runtime_sha": "9" * 64}
            )
        with self.assertRaisesRegex(RuntimeError, "compiler sha256 mismatch"):
            bench_coremark.validate_benchmark_profile_match(
                report, **{**kwargs, "wamr_compiler_sha": "9" * 64}
            )
        with self.assertRaisesRegex(RuntimeError, "Wasmtime runtime sha256"):
            bench_coremark.validate_benchmark_profile_match(
                report, **{**kwargs, "wasmtime_runtime_sha": "9" * 64}
            )
        with self.assertRaisesRegex(RuntimeError, "fixture/input hash"):
            bench_coremark.validate_benchmark_profile_match(
                report, **{**kwargs, "fixture_sha": "9" * 64}
            )
        with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
            bench_coremark.validate_benchmark_profile_match(
                report, **{**kwargs, "expected_arch": "x86_64"}
            )
        with self.assertRaisesRegex(RuntimeError, "host cpu_model"):
            bench_coremark.validate_benchmark_profile_match(
                report,
                **{
                    **kwargs,
                    "host": bench_coremark.HostIdentity(
                        "aarch64", 4, "different", "runner", "boot-id"
                    ),
                },
            )
        with self.assertRaisesRegex(RuntimeError, "nonempty local execution IDs"):
            bench_coremark.validate_benchmark_profile_match(
                report,
                **{
                    **kwargs,
                    "current_execution": {"provider": "local", "run_id": ""},
                },
            )
        with self.assertRaisesRegex(RuntimeError, "local execution ID"):
            bench_coremark.validate_benchmark_profile_match(
                report,
                **{
                    **kwargs,
                    "current_execution": {
                        "provider": "local",
                        "run_id": "stale-run",
                    },
                },
            )
        github_report = copy.deepcopy(report)
        github_report["provenance"]["execution"] = {
            "provider": "github-actions",
            "repository": "cataggar/wamr",
            "run_id": "100",
            "run_attempt": "1",
        }
        with self.assertRaisesRegex(RuntimeError, "run_id"):
            bench_coremark.validate_benchmark_profile_match(
                github_report,
                **{
                    **kwargs,
                    "current_execution": {
                        "provider": "github-actions",
                        "repository": "cataggar/wamr",
                        "run_id": "101",
                        "run_attempt": "1",
                    },
                },
            )

        shared = copy.deepcopy(report)
        shared.pop("wamr_comparison")
        shared["schedule"] = [
            record for record in shared["schedule"]
            if record["engine_key"] != "baseline"
        ]
        for position, record in enumerate(shared["schedule"], 1):
            record["schedule_position"] = position
        by_role = {engine["role"]: engine for engine in shared["engines"]}
        for role, key in (
            ("wamr-baseline", "target"),
            ("wamr-target", "target"),
            ("wasmtime-baseline", "wasmtime"),
        ):
            by_role[role]["sample_schedule_positions"] = [
                record["schedule_position"] for record in shared["schedule"]
                if record["engine_key"] == key
            ]
        with self.assertRaisesRegex(RuntimeError, "cannot share samples"):
            bench_coremark.validate_benchmark_profile_match(shared, **kwargs)
        baseline = by_role["wamr-baseline"]
        target = by_role["wamr-target"]
        baseline["identity"] = copy.deepcopy(target["identity"])
        baseline["identity"]["source"]["ref"] = "baseline-alias"
        baseline["values"] = target["values"].copy()
        bench_coremark.validate_benchmark_profile_match(shared, **kwargs)
        baseline["values"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "cannot share samples"):
            bench_coremark.validate_benchmark_profile_match(shared, **kwargs)

        wrong_values = copy.deepcopy(report)
        wrong_values["schedule"][-1]["iterations_per_second"] += 1
        with self.assertRaisesRegex(RuntimeError, "measured schedule"):
            bench_coremark.validate_benchmark_profile_match(wrong_values, **kwargs)
        wrong_keys = copy.deepcopy(report)
        wrong_keys["schedule"][0]["engine_key"] = "other"
        with self.assertRaisesRegex(RuntimeError, "engine keys"):
            bench_coremark.validate_benchmark_profile_match(wrong_keys, **kwargs)

    def test_missing_or_legacy_benchmark_provenance_is_not_authoritative(self):
        report, _, _, _ = self.authoritative_report()
        missing = copy.deepcopy(report)
        del missing["provenance"]
        with self.assertRaisesRegex(RuntimeError, "missing required provenance"):
            bench_coremark.validate_authoritative_benchmark_report(missing)

        wrong_args = copy.deepcopy(report)
        wrong_args["guest_args"][-2] = "300000"
        with self.assertRaisesRegex(RuntimeError, "guest args"):
            bench_coremark.validate_authoritative_benchmark_report(wrong_args)

        missing_local_id = copy.deepcopy(report)
        missing_local_id["provenance"]["execution"]["run_id"] = ""
        with self.assertRaisesRegex(RuntimeError, "execution provenance"):
            bench_coremark.validate_authoritative_benchmark_report(
                missing_local_id
            )

        legacy = {
            "schema_version": 1,
            "kind": bench_coremark.REPORT_KIND,
        }
        status = bench_coremark.benchmark_report_status(legacy)
        self.assertFalse(status["authoritative"])
        self.assertEqual("legacy-unverified", status["status"])
        with self.assertRaisesRegex(RuntimeError, "legacy-unverified"):
            bench_coremark.validate_authoritative_benchmark_report(legacy)

    @mock.patch.dict(bench_coremark.os.environ, {}, clear=True)
    def test_local_execution_identity_must_be_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "local execution identity"):
            bench_coremark.capture_execution_identity()
        with self.assertRaisesRegex(RuntimeError, "local execution identity"):
            bench_coremark.capture_execution_identity("   ")
        self.assertEqual(
            {"provider": "local", "run_id": "shared-run"},
            bench_coremark.capture_execution_identity("shared-run"),
        )

    @mock.patch.dict(bench_coremark.os.environ, {}, clear=True)
    def test_standalone_json_provenance_generates_local_execution_id(self):
        with mock.patch.object(
            bench_coremark, "resolve_ref_sha", return_value="b" * 40
        ):
            provenance = bench_coremark.capture_report_provenance(REPO)
        execution = provenance["execution"]
        self.assertEqual("local", execution["provider"])
        self.assertTrue(execution["run_id"])
        with self.assertRaisesRegex(RuntimeError, "nonempty local execution IDs"):
            bench_coremark.validate_execution_match(
                execution, {"provider": "local", "run_id": ""}
            )

    def test_target_artifact_handoff_survives_distinct_build_path(self):
        root = REPO / ".cache/test-coremark-artifact-handoff"
        source = root / "temporary-benchmark-worktree/zig-out/bin"
        artifact_dir = root / "profile-artifacts"
        baseline_artifact_dir = root / "baseline-profile-artifacts"
        shutil.rmtree(root, ignore_errors=True)
        source.mkdir(parents=True)
        wamr = source / "wamr"
        wamrc = source / "wamrc"
        simd_runner = source / "simd-bench-runner"
        cwasm = source.parent.parent / ".bench-coremark.cwasm"
        wamr.write_bytes(b"wamr-with-build-path-a")
        wamrc.write_bytes(b"wamrc-with-build-path-a")
        simd_runner.write_bytes(b"simd-runner-with-build-path-a")
        cwasm.write_bytes(b"exact-cwasm")
        prepared = bench_coremark.PreparedEngine(
            "target",
            "WAMR target",
            "target",
            "ReleaseFast",
            [str(wamr), "run", str(cwasm)],
            source,
            {},
            bench_coremark.EXPECTED_ITERATIONS,
            identity=bench_coremark.make_wamr_identity(
                ref="target",
                source_sha="b" * 40,
                optimize="ReleaseFast",
                runtime_path=wamr,
                compiler_path=wamrc,
                module_path=cwasm,
                simd_runner_path=simd_runner,
            ),
        )
        try:
            retained = bench_coremark.retain_wamr_artifact_handoff(
                prepared, artifact_dir, role="wamr-target"
            )
            self.assertEqual(
                "wamr-target",
                json.loads((artifact_dir / "manifest.json").read_text())["role"],
            )
            manifest = (artifact_dir / "manifest.json").read_bytes()
            for existing_dir in (artifact_dir, root):
                with self.subTest(existing_dir=existing_dir):
                    with self.assertRaises(FileExistsError):
                        bench_coremark.retain_wamr_artifact_handoff(
                            prepared, existing_dir, role="wamr-target"
                        )
                    self.assertEqual(
                        manifest, (artifact_dir / "manifest.json").read_bytes()
                    )
                    self.assertEqual(b"wamr-with-build-path-a", wamr.read_bytes())
            shutil.rmtree(root / "temporary-benchmark-worktree")
            loaded = bench_coremark.load_wamr_artifact_handoff(
                artifact_dir, retained, role="wamr-target"
            )
            self.assertEqual(b"wamr-with-build-path-a", loaded["runtime"].read_bytes())
            self.assertEqual(
                b"simd-runner-with-build-path-a",
                loaded["simd_runner"].read_bytes(),
            )
            self.assertNotEqual(wamr.parent, loaded["runtime"].parent)
            baseline_prepared = bench_coremark.PreparedEngine(
                **{
                    **prepared.__dict__,
                    "identity": {
                        **retained,
                        "source": {
                            **retained["source"],
                            "ref": "baseline",
                        },
                    },
                }
            )
            baseline_retained = bench_coremark.retain_wamr_artifact_handoff(
                baseline_prepared,
                baseline_artifact_dir,
                role="wamr-baseline",
            )
            baseline_loaded = bench_coremark.load_wamr_artifact_handoff(
                baseline_artifact_dir,
                baseline_retained,
                role="wamr-baseline",
            )
            self.assertEqual(
                b"simd-runner-with-build-path-a",
                baseline_loaded["simd_runner"].read_bytes(),
            )
            baseline_loaded["simd_runner"].write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                bench_coremark.load_wamr_artifact_handoff(
                    baseline_artifact_dir,
                    baseline_retained,
                    role="wamr-baseline",
                )

            legacy_dir = root / "legacy-target-artifacts"
            legacy_dir.mkdir()
            legacy_identity = {
                key: value
                for key, value in retained.items()
                if key != "simd_runner"
            }
            for key, name in (
                ("runtime", "wamr"),
                ("compiler", "wamrc"),
                ("module", "coremark.cwasm"),
            ):
                source_path = Path(legacy_identity[key]["path"])
                destination = legacy_dir / name
                shutil.copy2(source_path, destination)
                legacy_identity[key] = {
                    **legacy_identity[key],
                    "path": str(destination),
                }
            (legacy_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "coremark-wamr-artifact-handoff",
                        "identity": legacy_identity,
                    }
                )
            )
            legacy_loaded = bench_coremark.load_wamr_artifact_handoff(
                legacy_dir, legacy_identity
            )
            self.assertNotIn("simd_runner", legacy_loaded)
            no_simd_dir = root / "new-target-without-simd"
            no_simd = bench_coremark.retain_wamr_artifact_handoff(
                bench_coremark.replace(prepared, identity=legacy_identity),
                no_simd_dir,
            )
            self.assertNotIn(
                "simd_runner",
                bench_coremark.load_wamr_artifact_handoff(no_simd_dir, no_simd),
            )
            with self.assertRaisesRegex(RuntimeError, "legacy"):
                bench_coremark.load_wamr_artifact_handoff(
                    legacy_dir,
                    legacy_identity,
                    role="wamr-baseline",
                )

            loaded["runtime"].write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                bench_coremark.load_wamr_artifact_handoff(
                    artifact_dir, retained
                )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            manifest["role"] = "wamr-baseline"
            (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "role"):
                bench_coremark.load_wamr_artifact_handoff(
                    artifact_dir, retained, role="wamr-target"
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_malformed_handoff_shapes_fail_explicitly(self):
        root = REPO / ".cache/test-coremark-malformed-handoff"
        root.mkdir(parents=True, exist_ok=True)
        try:
            for payload in (None, [], "invalid", 1, {
                "schema_version": True,
                "kind": "coremark-wamr-artifact-handoff",
            }):
                with self.subTest(payload=payload):
                    (root / "manifest.json").write_text(json.dumps(payload))
                    with self.assertRaisesRegex(RuntimeError, "unsupported"):
                        bench_coremark.load_wamr_artifact_handoff(root, {})
        finally:
            shutil.rmtree(root)

    def test_invalid_raw_benchmark_scores_fail_explicitly(self):
        for value in (None, True, "100", 0, -1, float("nan"), float("inf"), 10**400):
            with self.subTest(value=value):
                report, _, _, _ = self.authoritative_report()
                report["engines"][0]["values"][0] = value
                with self.assertRaisesRegex(RuntimeError, "invalid measured samples"):
                    bench_coremark.validate_authoritative_benchmark_report(report)

    def test_main_ordinary_pr_cli_path_renders_without_json_provenance(self):
        host = bench_coremark.HostIdentity(
            "x86_64", 4, "test CPU", "", "boot-id"
        )
        result = bench_coremark.EngineResult(
            "WAMR",
            "HEAD",
            "ReleaseFast",
            [100.0, 101.0, 102.0],
            identity=wamr_identity("HEAD", "b" * 40, "2" * 64),
        )
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "bench_coremark.py",
                    "--baseline",
                    "HEAD",
                    "--target",
                    "HEAD",
                    "--profile",
                    "ci",
                ],
            ),
            mock.patch.object(
                bench_coremark,
                "resolve_fixture",
                return_value=(Path("/fixture.wasm"), "f" * 64),
            ),
            mock.patch.object(
                bench_coremark, "capture_host_identity", return_value=host
            ),
            mock.patch.object(
                bench_coremark, "resolve_ref_sha", return_value="b" * 40
            ),
            mock.patch.object(
                bench_coremark,
                "make_worktree",
                return_value=(Path("/benchmark-worktree"), "b" * 40),
            ),
            mock.patch.object(
                bench_coremark, "build_and_run_wamr", return_value=result
            ),
            mock.patch.object(bench_coremark, "validate_same_host"),
            mock.patch.object(bench_coremark, "run", return_value=""),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            self.assertEqual(0, bench_coremark.main())

    @mock.patch.dict(bench_coremark.os.environ, {}, clear=True)
    def test_main_authoritative_json_cli_path_wires_provenance(self):
        root = REPO / ".cache/test-coremark-main-json"
        report_path = root / "report.json"
        wasmtime = root / "wasmtime"
        artifact_dir = root / "profile-artifacts"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        wasmtime.write_bytes(b"wasmtime")
        build_dir = root / "benchmark-build"
        build_dir.mkdir()
        wamr = build_dir / "wamr"
        wamrc = build_dir / "wamrc"
        simd_runner = build_dir / "simd-bench-runner"
        cwasm = build_dir / "coremark.cwasm"
        wamr.write_bytes(b"wamr-benchmark-build")
        wamrc.write_bytes(b"wamrc-benchmark-build")
        simd_runner.write_bytes(b"simd-runner-benchmark-build")
        cwasm.write_bytes(b"coremark-benchmark-module")
        host = bench_coremark.HostIdentity(
            "aarch64", 4, "Neoverse-N2", "runner", "boot-id"
        )
        affinity = bench_coremark.AffinityInfo((0, 1), 0, "/usr/bin/taskset")
        target_identity = bench_coremark.make_wamr_identity(
            ref="HEAD",
            source_sha="b" * 40,
            optimize="ReleaseFast",
            runtime_path=wamr,
            compiler_path=wamrc,
            module_path=cwasm,
            simd_runner_path=simd_runner,
        )
        target = bench_coremark.PreparedEngine(
            "wamr-target",
            "WAMR HEAD",
            "HEAD",
            "ReleaseFast",
            ["wamr"],
            root,
            {},
            bench_coremark.EXPECTED_ITERATIONS,
            identity=target_identity,
        )

        def measured(engines, *, warmups, runs, affinity):
            records = []
            by_key = {engine.key: [] for engine in engines}
            position = 0
            for phase, count in (("warmup", warmups), ("measured", runs)):
                ordinals = {engine.key: 0 for engine in engines}
                for key in bench_coremark.counterbalanced_order(
                    [engine.key for engine in engines], count
                ):
                    engine = next(item for item in engines if item.key == key)
                    position += 1
                    ordinals[key] += 1
                    record = bench_coremark.SampleRecord(
                        key,
                        engine.engine,
                        phase,
                        ordinals[key],
                        position,
                        "2026-09-08T00:00:00+00:00",
                        "2026-09-08T00:00:01+00:00",
                        1.0,
                        100.0,
                        bench_coremark.EXPECTED_ITERATIONS,
                    )
                    records.append(record)
                    by_key[key].append(record)
            results = {
                engine.key: bench_coremark.EngineResult(
                    engine.engine,
                    engine.version,
                    engine.optimize,
                    [100.0] * runs,
                    by_key[engine.key],
                    engine.identity,
                )
                for engine in engines
            }
            return results, records

        try:
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "bench_coremark.py",
                        "--baseline",
                        "HEAD",
                        "--target",
                        "HEAD",
                        "--profile",
                        "authoritative",
                        "--wasmtime-baseline",
                        "auto",
                        "--require-native-arch",
                        "aarch64",
                        "--execution-id",
                        "cli-shared-run",
                        "--json-out",
                        str(report_path),
                        "--retain-target-artifacts",
                        str(artifact_dir),
                    ],
                ),
                mock.patch.object(
                    bench_coremark,
                    "resolve_fixture",
                    return_value=(Path("/fixture.wasm"), "f" * 64),
                ),
                mock.patch.object(
                    bench_coremark, "validate_native_host", return_value=host
                ),
                mock.patch.object(
                    bench_coremark, "select_cpu_affinity", return_value=affinity
                ),
                mock.patch.object(
                    bench_coremark, "resolve_ref_sha", return_value="b" * 40
                ),
                mock.patch.object(
                    bench_coremark,
                    "make_worktree",
                    return_value=(Path("/benchmark-worktree"), "b" * 40),
                ),
                mock.patch.object(
                    bench_coremark, "prepare_wamr", return_value=target
                ),
                mock.patch.object(
                    bench_coremark,
                    "install_pinned_wasmtime",
                    return_value=wasmtime,
                ),
                mock.patch.object(
                    bench_coremark,
                    "validate_pinned_wasmtime",
                    return_value=bench_coremark.PINNED_WASMTIME_VERSION,
                ),
                mock.patch.object(
                    bench_coremark,
                    "measure_prepared_engines",
                    side_effect=measured,
                ),
                mock.patch.object(bench_coremark, "validate_same_host"),
                mock.patch.object(bench_coremark, "run", return_value=""),
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(0, bench_coremark.main())
            report = json.loads(report_path.read_text())
            self.assertEqual("cli-shared-run", report["provenance"]["execution"]["run_id"])
            self.assertEqual(bench_coremark.REPORT_SCHEMA_VERSION, report["schema_version"])
            target_report = next(
                engine
                for engine in report["engines"]
                if engine["role"] == "wamr-target"
            )
            loaded = bench_coremark.load_wamr_artifact_handoff(
                artifact_dir, target_report["identity"]
            )
            self.assertEqual(wamr.read_bytes(), loaded["runtime"].read_bytes())
        finally:
            shutil.rmtree(root, ignore_errors=True)

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

    @mock.patch.object(bench_coremark, "host_emulation_evidence", return_value="")
    @mock.patch.dict(
        bench_coremark.os.environ,
        {"RUNNER_ARCH": "ARM64"},
        clear=True,
    )
    @mock.patch.object(
        bench_coremark,
        "capture_host_identity",
        return_value=bench_coremark.HostIdentity(
            "aarch64", 4, "Neoverse-N2", "runner", "boot-id"
        ),
    )
    def test_native_aarch64_host_validation(self, _, __):
        identity = bench_coremark.validate_native_host("aarch64")
        self.assertEqual(identity.cpu_model, "Neoverse-N2")

    @mock.patch.object(
        bench_coremark,
        "capture_host_identity",
        return_value=bench_coremark.HostIdentity(
            "aarch64", 4, "Neoverse-N2", "runner", "boot-id"
        ),
    )
    @mock.patch.object(
        bench_coremark,
        "host_emulation_evidence",
        return_value="Hypervisor vendor: QEMU",
    )
    @mock.patch.dict(
        bench_coremark.os.environ,
        {"RUNNER_ARCH": "ARM64"},
        clear=True,
    )
    def test_native_host_rejects_emulation(self, _, __):
        with self.assertRaisesRegex(RuntimeError, "under emulation"):
            bench_coremark.validate_native_host("aarch64")

    @mock.patch.object(
        bench_coremark,
        "capture_host_identity",
        return_value=bench_coremark.HostIdentity(
            "aarch64", 8, "Neoverse-N2", "runner", "other-boot"
        ),
    )
    def test_host_consistency_rejects_mixed_host(self, _):
        with self.assertRaisesRegex(RuntimeError, "host identity changed"):
            bench_coremark.validate_same_host(
                bench_coremark.HostIdentity(
                    "aarch64", 4, "Neoverse-N2", "runner", "boot-id"
                )
            )

    def test_aarch64_workflow_contract(self):
        workflow = WORKFLOW.read_text()
        self.assertIn("runs-on: ubuntu-24.04-arm", workflow)
        self.assertIn(
            "group: coremark-aarch64-${{ github.event_name }}-${{ github.ref }}",
            workflow,
        )
        self.assertIn("- profile", workflow)
        self.assertIn("- paired-profile", workflow)
        self.assertNotIn("19d046a5b23b9c39acf5f7062976f04c5ca8ca75", workflow)
        self.assertIn(
            'profile_sha="$(git rev-parse --verify --end-of-options "${profile_ref}^{commit}")"',
            workflow,
        )
        self.assertIn('echo "baseline=$baseline_sha"', workflow)
        self.assertIn('echo "target=$target_sha"', workflow)
        self.assertIn("Unsupported mode:", workflow)
        dispatch = workflow.split(
            "- name: Run authoritative same-host CoreMark comparison", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertIn("github.event_name == 'workflow_dispatch'", dispatch)
        self.assertIn("--profile  authoritative", dispatch)
        self.assertIn("--wasmtime-baseline auto", dispatch)
        self.assertIn("--require-native-arch aarch64", dispatch)
        self.assertIn("--json-out coremark-report.json", dispatch)
        self.assertNotIn("--runs", dispatch)

        pr = workflow.split("- name: Run CoreMark PR comparison", 1)[1].split(
            "\n      - name:", 1
        )[0]
        self.assertIn("github.event_name == 'pull_request'", pr)
        self.assertIn("--profile  ci", pr)
        self.assertNotIn("--wasmtime-baseline", pr)

        identity_benchmark = workflow.split(
            "- name: Run authoritative profile identity benchmark", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertIn('--baseline "$PROFILE_SHA"', identity_benchmark)
        self.assertIn('--target   "$PROFILE_SHA"', identity_benchmark)
        self.assertIn("--profile  authoritative", identity_benchmark)
        self.assertIn(
            "--json-out coremark-profile/benchmark-report.json",
            identity_benchmark,
        )
        self.assertIn(
            '--retain-target-artifacts "$RUNNER_TEMP/coremark-profile-artifacts"',
            identity_benchmark,
        )

        perf_setup = workflow.split(
            "- name: Install matching perf for profiling", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertIn("github.event.inputs.mode == 'profile'", perf_setup)
        self.assertIn('package="linux-tools-${kernel}"', perf_setup)
        self.assertIn('"$perf_binary" record -e cycles:u', perf_setup)
        self.assertIn("native cycles:u sampling permitted", perf_setup)
        self.assertNotIn("qemu", perf_setup.lower())

        profile_step = workflow.split(
            "- name: Capture matched-host CoreMark profiles", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertIn("github.event_name == 'workflow_dispatch'", profile_step)
        self.assertIn("github.event.inputs.mode == 'profile'", profile_step)
        self.assertIn("profile_coremark_aarch64.py", profile_step)
        self.assertIn(
            "--benchmark-report coremark-profile/benchmark-report.json",
            profile_step,
        )
        self.assertIn(
            '--benchmark-artifacts "$RUNNER_TEMP/coremark-profile-artifacts"',
            profile_step,
        )
        self.assertIn('--wamr-ref "$PROFILE_SHA"', profile_step)
        self.assertNotIn("--work-root", profile_step)
        self.assertIn(
            '--wasmtime-cache "$RUNNER_TEMP/coremark-wasmtime"', profile_step
        )
        self.assertIn("--min-samples 1000", profile_step)

        paired_benchmark = workflow.split(
            "- name: Run paired native profile benchmark", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertIn('--baseline "$BASELINE"', paired_benchmark)
        self.assertIn('--target   "$TARGET"', paired_benchmark)
        self.assertIn("--retain-baseline-artifacts", paired_benchmark)
        self.assertIn("--retain-target-artifacts", paired_benchmark)
        self.assertNotIn("--min-delta-pct", paired_benchmark)
        self.assertNotIn("--min-median-delta-pct", paired_benchmark)
        self.assertNotIn("--paired-simd-acceptance", paired_benchmark)

        baseline_profile = workflow.split(
            "- name: Capture baseline frame profile", 1
        )[1].split("\n      - name:", 1)[0]
        target_profile = workflow.split(
            "- name: Capture target frame profile", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertIn("--benchmark-role wamr-baseline", baseline_profile)
        self.assertIn("--benchmark-role wamr-target", target_profile)
        self.assertIn('--frame-func "${FRAME_FUNC:-10}"', baseline_profile)
        self.assertIn('--frame-func "${FRAME_FUNC:-10}"', target_profile)
        self.assertNotIn("Enforce paired native acceptance", workflow)
        paired_upload = workflow.split(
            "- name: Upload paired profile evidence", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertNotIn("coremark-paired-profile/\n", paired_upload)
        self.assertIn("baseline-profile/*.data.gz", paired_upload)
        self.assertIn("target-profile/*.data.gz", paired_upload)
        self.assertIn("- name: Retain paired baseline tools", workflow)
        self.assertIn("- name: Retain paired target tools", workflow)

        profile_script = PROFILE_SCRIPT.read_text()
        self.assertIn("--profile=jitdump", profile_script)
        self.assertIn("WAMR_AOT_SPILL_METRIC", profile_script)
        self.assertIn("WAMR_AOT_CODEGEN_TIMING", profile_script)
        self.assertIn('"cycles:u"', profile_script)
        self.assertNotIn("AUTHORITATIVE_BASELINE_RUN", profile_script)
        self.assertIn("--benchmark-report", profile_script)
        self.assertIn("select_cpu_affinity()", profile_script)
        self.assertIn("coremark_guest_args(", profile_script)
        self.assertIn("PROFILE_CAPTURES_PER_ENGINE = 2", profile_script)
        self.assertIn("MIN_ATTRIBUTION_COVERAGE_PCT = 99.0", profile_script)
        self.assertIn("--benchmark-role", profile_script)
        self.assertIn('"--authoritative"', profile_script)
        self.assertIn('"--min-attribution-pct"', profile_script)
        self.assertIn("all_alu", profile_script)
        self.assertIn(
            "the all-ALU differential is not address/check headroom",
            profile_script,
        )

        for line in workflow.splitlines():
            stripped = line.strip()
            if not stripped.startswith("uses:") or "uses: ./" in stripped:
                continue
            action = stripped.split("#", 1)[0].strip()
            self.assertRegex(action, r"@[0-9a-f]{40}$")


class NativeBenchmarkTests(unittest.TestCase):
    """All native records/artifacts in this class are explicitly synthetic."""

    def setUp(self):
        self.root = REPO / ".bench-coremark" / f"native-tests-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.source = {"commit": "a" * 40, "tree_sha256": "b" * 64,
                       "tracked_diff_sha256": "c" * 64}
        self.platform = {"arch": "x86_64", "cpu_model": "synthetic-test-cpu",
                         "active_cpu_count": 1, "azure_sku": "synthetic-test-sku",
                         "azure_region": "synthetic-test-region"}
        self.options = {"optimize": "ReleaseFast", "bounds_checks": True,
                        "stack_checks": True, "simd": False, "threads": False,
                        "memory64": False}
        self.config = {"profile": "ci", "warmups": 1, "runs": 1,
                       "steady_invocations": 2, "valid_hours": 24,
                       "workloads": ["coremark", "coremark-nofp", "compute", "memory"],
                       "abi_proofs": dict.fromkeys(native_benchmark.FIXTURES),
                       "producer_source": self.source, "targets": {}}
        compiler_path = self.write("compiler", b"synthetic compiler")
        for target in ("linux", "unikraft"):
            runtime_path = self.write(f"{target}-runtime", f"synthetic {target} runtime".encode())
            image_path = self.write(f"{target}-image", f"synthetic complete {target} image".encode())
            aot_paths = {name: str(self.write(f"{target}-{name}.cwasm",
                                             f"synthetic {target} {name} AOT".encode()))
                         for name in self.config["workloads"]}
            spec = {"runtime_path": str(runtime_path), "compiler_path": str(compiler_path),
                    "compiler_version": "synthetic-v1", "source": self.source,
                    "compile_profile": None if target == "linux" else "unikraft-x86_64",
                    "target_abi": f"synthetic-{target}-abi",
                    "platform": copy.deepcopy(self.platform), "options": copy.deepcopy(self.options),
                    "image_path": str(image_path), "aot_paths": aot_paths}
            receipt = {"schema_version": 1, "kind": "wamr-native-image-receipt",
                       "evidence_kind": "synthetic", "os": target,
                       "runtime": native_benchmark.artifact(runtime_path),
                       "image": native_benchmark.artifact(image_path),
                       "compiler": {"binary": native_benchmark.artifact(compiler_path),
                                    "source": self.source, "version": "synthetic-v1"},
                       "options": spec["options"],
                       "compile_profile": spec["compile_profile"],
                       "source": self.source, "target_abi": spec["target_abi"],
                       "platform": spec["platform"], "configured_vm_ram_bytes": 1024**3,
                       "compiler_embedded": False,
                       "aot_modules": {name: native_benchmark.artifact(path)
                                       for name, path in aot_paths.items()}}
            receipt_path = self.write(f"{target}-receipt.json", json.dumps(receipt).encode())
            spec["image_receipt_path"] = str(receipt_path)
            self.config["targets"][target] = spec
        self.manifest = native_benchmark.create_plan(
            self.config, REPO, now=datetime.now(timezone.utc) - timedelta(minutes=1),
            allow_synthetic=True)

    def write(self, name, contents):
        path = self.root / name
        path.write_bytes(contents)
        return path

    def result(self, run):
        config = native_benchmark.run_configuration(self.manifest, run["run_id"])
        target = config["target"]
        output = """\
2K performance run parameters for coremark.
Iterations/Sec   : 20000
Iterations       : 400000
Total time (secs): 20
Total ticks      : 20000000
seedcrc          : 0xe9f5
[0]crclist       : 0xe714
[0]crcmatrix     : 0x1fd7
[0]crcstate      : 0x8e3a
[0]crcfinal      : 0x33ff
Correct operation validated. See README.md for run and reporting rules.
"""
        if not run["workload"].startswith("coremark"):
            output = ""
        return {
            "schema_version": 1, "kind": "wamr-native-benchmark-result",
            "evidence_kind": "synthetic", "campaign_id": self.manifest["campaign_id"],
            "run_id": run["run_id"], "config_sha256": native_benchmark.cache_key(config),
            "image_receipt_sha256": target["image_receipt_sha256"],
            "observed": {"image_sha256": target["image"]["sha256"],
                         "runtime_sha256": target["runtime"]["sha256"],
                         "aot_sha256": target["aot_modules"][run["workload"]]["sha256"],
                         "wasm_sha256": config["workload"]["sha256"], "platform": target["platform"],
                         "options": target["options"], "mode": "aot", "jit_preset": None,
                         "compile_profile": target["compile_profile"]},
            "outcome": "success", "exit_code": 0, "phase_contract": "wamr-embedding-v1",
            "clock": {"source": "synthetic-clock", "unit": "us", "ticks_per_second": 1000000,
                      "resolution_ticks": 1},
            "phases": {"compile_ticks": None, "load_ticks": 10, "instantiate_ticks": 20,
                       "first_invocation_ticks": 21000000,
                       "steady_state_ticks": [21000000, 21000000]},
            "invocations": [{"phase": phase, "outcome": "returned", "exit_code": 0,
                             "stdout": output} for phase in ("first", "steady", "steady")],
            "memory": {"image_sha256": target["image"]["sha256"],
                       "configured_vm_ram_bytes": 1024**3,
                       "coverage": "partial-guest", "method": "synthetic-page-accounting",
                       "covered_regions": ["runtime-pages"], "omitted_regions": ["kernel-pages"],
                       "samples": [{"stage": stage, "reserved_address_bytes": 2**20,
                                    "committed_bytes": 2**16}
                                   for stage in ("after_instantiation", "after_first", "after_steady")]},
        }

    def capture_directories(self, mutate=None):
        directories = []
        for run in self.manifest["schedule"]:
            result = self.result(run)
            if mutate:
                mutate(result, run)
            directory = self.root / run["run_id"]
            raw = ("private boot /subscriptions/PRIVATE-ID\n" + native_benchmark.PREFIX +
                   json.dumps(result) + "\n").encode()
            native_benchmark.persist_observation(
                self.manifest, run["run_id"], directory, raw, b"private stderr PRIVATE-ID",
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(), observation_seconds=100,
                control_plane_seconds=45)
            directories.append(directory)
        return directories

    def test_native_fixtures_pin_current_bytes(self):
        for name, (path, digest) in native_benchmark.FIXTURES.items():
            with self.subTest(name=name):
                self.assertEqual(native_benchmark.sha256_file(REPO / path), digest)

    def test_native_synthetic_report_preserves_phases_and_redacts_raw(self):
        report = native_benchmark.build_report(
            self.manifest, self.capture_directories(), allow_synthetic=True)
        self.assertEqual(report["evidence_kind"], "synthetic")
        self.assertFalse(report["status"]["paired_measurement_complete"])
        self.assertTrue(report["status"]["all_attempts_successful"])
        self.assertFalse(report["status"]["profile_counts_match"])
        self.assertEqual(report["status"]["coremark_compliance"], "not-certified")
        self.assertEqual(len(report["records"]), 16)
        self.assertEqual(report["summary"]["linux"]["coremark"]["load_seconds"]["mean"], .00001)
        self.assertEqual(report["summary"]["linux"]["coremark"]["steady_state_seconds"]["mean"], 21)
        self.assertEqual(report["records"][0]["observation"]["observation_seconds"], 100)
        self.assertEqual(report["records"][0]["observation"]["control_plane_seconds"], 45)
        text = json.dumps(report)
        self.assertNotIn("PRIVATE-ID", text)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("Iterations/Sec", text)
        self.assertIn("partial-guest", text)
        self.assertNotIn("rss", text.lower())
        if os.name == "posix":
            self.assertEqual((self.root / "run-0001/stdout.bin").stat().st_mode & 0o777, 0o600)

    def test_native_cli_and_default_validators_reject_synthetic(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            native_benchmark.validate_manifest(self.manifest)
        with self.assertRaisesRegex(ValueError, "synthetic"):
            native_benchmark.build_report(self.manifest, [])
        path = self.write("manifest.json", json.dumps(self.manifest).encode())
        with mock.patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(native_benchmark.main(
                ["native-request", "--manifest", str(path), "--run-id", "run-0001",
                 "--out", str(self.root / "request.json")]), 2)
        self.assertFalse((self.root / "request.json").exists())

    def test_native_manifest_matches_sources_options_platform(self):
        for field, replacement in (
                ("source", {**self.source, "commit": "d" * 40}),
                ("options", {**self.options, "bounds_checks": False}),
                ("platform", {**self.platform, "active_cpu_count": 2})):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest["targets"]["unikraft"][field] = replacement
                manifest["targets"]["unikraft"]["image_receipt"][field] = replacement
                if field == "source":
                    manifest["targets"]["unikraft"]["compiler"]["source"] = replacement
                    manifest["targets"]["unikraft"]["image_receipt"]["compiler"]["source"] = replacement
                with self.assertRaisesRegex(ValueError, "unmatched"):
                    native_benchmark.validate_manifest(manifest, allow_synthetic=True)

    def test_native_plan_checks_exact_image_bytes_and_receipt(self):
        image = Path(self.config["targets"]["unikraft"]["image_path"])
        image.write_bytes(b"changed synthetic image")
        with self.assertRaisesRegex(ValueError, "image mismatch"):
            native_benchmark.create_plan(self.config, REPO, allow_synthetic=True)

    def test_native_compiler_embedded_and_jit_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["targets"]["unikraft"]["image_receipt"]["compiler_embedded"] = True
        with self.assertRaisesRegex(ValueError, "compiler-free"):
            native_benchmark.validate_manifest(manifest, allow_synthetic=True)
        manifest = copy.deepcopy(self.manifest)
        manifest["targets"]["unikraft"]["mode"] = "jit"
        manifest["targets"]["unikraft"]["jit_preset"] = "fast"
        with self.assertRaisesRegex(ValueError, "JIT"):
            native_benchmark.validate_manifest(manifest, allow_synthetic=True)

    def test_native_identical_aot_requires_bound_abi_proof(self):
        manifest = copy.deepcopy(self.manifest)
        for target in manifest["targets"].values():
            target["aot_modules"]["compute"] = manifest["targets"]["linux"]["aot_modules"]["compute"]
            target["image_receipt"]["aot_modules"]["compute"] = target["aot_modules"]["compute"]
        manifest["abi_compatibility"]["compute"] = {"strategy": "identical", "proof_sha256": None}
        with self.assertRaisesRegex(ValueError, "ABI proof"):
            native_benchmark.validate_manifest(manifest, allow_synthetic=True)
        manifest["abi_compatibility"]["compute"]["proof_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "compiler profiles"):
            native_benchmark.validate_manifest(manifest, allow_synthetic=True)

    def test_native_unikraft_requires_explicit_compiler_profile(self):
        manifest = copy.deepcopy(self.manifest)
        target = manifest["targets"]["unikraft"]
        target["compile_profile"] = None
        target["image_receipt"]["compile_profile"] = None
        with self.assertRaisesRegex(ValueError, "explicit compiler profile"):
            native_benchmark.validate_manifest(manifest, allow_synthetic=True)

    def test_native_plan_verifies_identical_aot_proof_file(self):
        linux = self.config["targets"]["linux"]
        unikraft = self.config["targets"]["unikraft"]
        linux["compile_profile"] = unikraft["compile_profile"]
        linux_receipt_path = Path(linux["image_receipt_path"])
        linux_receipt = json.loads(linux_receipt_path.read_text())
        linux_receipt["compile_profile"] = linux["compile_profile"]
        linux_receipt_path.write_text(json.dumps(linux_receipt))
        unikraft["aot_paths"]["compute"] = linux["aot_paths"]["compute"]
        receipt_path = Path(unikraft["image_receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        aot = native_benchmark.artifact(linux["aot_paths"]["compute"])
        receipt["aot_modules"]["compute"] = aot
        receipt_path.write_text(json.dumps(receipt))
        proof = {"schema_version": 1, "kind": "wamr-aot-abi-proof",
                 "evidence_kind": "synthetic", "compatible": True, "source": self.source,
                 "target_abis": [linux["target_abi"], unikraft["target_abi"]],
                 "aot_sha256": aot["sha256"]}
        path = self.write("abi-proof.json", json.dumps(proof).encode())
        self.config["abi_proofs"]["compute"] = str(path)
        manifest = native_benchmark.create_plan(self.config, REPO, allow_synthetic=True)
        self.assertEqual(manifest["abi_compatibility"]["compute"],
                         {"strategy": "identical", "proof_sha256": native_benchmark.sha256_file(path)})
        proof["aot_sha256"] = "d" * 64
        path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError, "ABI proof"):
            native_benchmark.create_plan(self.config, REPO, allow_synthetic=True)

    def test_native_duplicates_partial_stale_and_unknown_fields_rejected(self):
        directories = self.capture_directories()
        for captures, message in ((directories[:-1], "partial"), (directories + directories[:1], "duplicate")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                native_benchmark.build_report(self.manifest, captures, allow_synthetic=True)
        run = self.manifest["schedule"][0]
        for field, value in (("campaign_id", str(uuid.uuid4())),
                             ("config_sha256", "e" * 64),
                             ("image_receipt_sha256", "e" * 64),
                             ("private_cloud_id", "private")):
            result = self.result(run)
            result[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                native_benchmark.validate_result(result, self.manifest, run["run_id"])

    def test_native_strict_stream_framing(self):
        result = self.result(self.manifest["schedule"][0])
        raw = (native_benchmark.PREFIX + json.dumps(result) + "\n").encode()
        self.assertEqual(native_benchmark.parse_result_stream(b"boot message\n" + raw), result)
        for bad in (raw + raw, raw[:-8], b"prefix:" + raw, b"boot only",
                    b'WAMR_BENCH_RESULT={"a":1,"a":2}\n',
                    b'WAMR_BENCH_RESULT={"a":NaN}\n'):
            with self.subTest(bad=bad[:40]), self.assertRaises(ValueError):
                native_benchmark.parse_result_stream(bad)

    def test_native_bad_clock_missing_phase_and_memory_rejected(self):
        run = self.manifest["schedule"][0]
        mutations = [
            lambda r: r["clock"].update(unit="cycles"),
            lambda r: r["clock"].update(ticks_per_second=1),
            lambda r: r["clock"].update(resolution_ticks=0),
            lambda r: r["clock"].update(resolution_ticks=2**64 - 1),
            lambda r: r["phases"].update(compile_ticks=12),
            lambda r: r["phases"].update(load_ticks=None),
            lambda r: r["phases"].update(steady_state_ticks=[1]),
            lambda r: r["phases"].update(load_ticks=True),
            lambda r: r["phases"].update(load_ticks=2**64 - 1),
            lambda r: r["memory"].update(image_sha256="d" * 64),
            lambda r: r["memory"].update(coverage="complete-guest"),
            lambda r: r["memory"].update(configured_vm_ram_bytes=1),
            lambda r: r.update(schema_version=True),
            lambda r: r.update(observed={**r["observed"], "platform": {
                **r["observed"]["platform"], "active_cpu_count": True}}),
            lambda r: r.update(observed={**r["observed"], "compile_profile": "unexpected"}),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                result = self.result(run)
                mutate(result)
                native_benchmark.validate_result(result, self.manifest, run["run_id"])

    def test_native_every_invocation_requires_correctness(self):
        run = next(run for run in self.manifest["schedule"] if run["workload"] == "coremark")
        for change in ("duplicate", "crc", "iterations", "trap", "throughput", "extra-context"):
            result = self.result(run)
            last = result["invocations"][-1]
            if change == "duplicate":
                last["stdout"] += "seedcrc : 0xe9f5\n"
            elif change == "crc":
                last["stdout"] = last["stdout"].replace("0xe714", "0x0000")
            elif change == "iterations":
                last["stdout"] = last["stdout"].replace("400000", "200000")
            elif change == "throughput":
                last["stdout"] = last["stdout"].replace("Iterations/Sec   : 20000",
                                                       "Iterations/Sec   : 200000")
            elif change == "extra-context":
                last["stdout"] += "[1]crclist : 0xe714\n"
            else:
                last["outcome"] = "trap"
            with self.subTest(change=change), self.assertRaises(ValueError):
                native_benchmark.validate_result(result, self.manifest, run["run_id"])

    def test_native_coremark_minimum_timing_independent_of_profile(self):
        run = next(run for run in self.manifest["schedule"] if run["workload"] == "coremark-nofp")
        result = self.result(run)
        for invocation in result["invocations"]:
            invocation["stdout"] = invocation["stdout"].replace("Total time (secs): 20",
                                                               "Total time (secs): 9").replace(
                                                                   "Iterations/Sec   : 20000",
                                                                   "Iterations/Sec   : 44444")
        checks = native_benchmark.validate_result(result, self.manifest, run["run_id"])
        self.assertTrue(all(not check["minimum_timing_met"] for check in checks))
        self.assertTrue(all(check["crc"]["seedcrc"] == "e9f5" for check in checks))

    def test_native_failed_attempts_and_warmups_are_retained(self):
        def fail_first(result, run):
            if run["position"] == 1:
                result["outcome"] = "trap"
                result["exit_code"] = None
                result["invocations"][-1]["outcome"] = "trap"
                result["invocations"][-1]["exit_code"] = None
        report = native_benchmark.build_report(
            self.manifest, self.capture_directories(fail_first), allow_synthetic=True)
        self.assertFalse(report["status"]["all_attempts_successful"])
        self.assertEqual(report["records"][0]["result"]["outcome"], "trap")
        self.assertEqual(report["records"][0]["run"]["phase"], "warmup")
        self.assertEqual(len(report["records"]), len(self.manifest["schedule"]))

    def test_native_failed_crc_output_retained_without_public_leak(self):
        def fail_crc(result, run):
            if run["workload"] == "coremark":
                result["outcome"] = "error"
                for invocation in result["invocations"]:
                    invocation["stdout"] += "\nERROR! private failure /secret/path\n"
        report = native_benchmark.build_report(
            self.manifest, self.capture_directories(fail_crc), allow_synthetic=True)
        records = [record for record in report["records"] if record["run"]["workload"] == "coremark"]
        self.assertTrue(all(record["correctness"][0]["self_check"] == "failed" for record in records))
        self.assertEqual(report["summary"]["linux"]["coremark"]["successful_runs"], 0)
        self.assertIsNone(report["summary"]["linux"]["coremark"]["load_seconds"])
        self.assertNotIn("/secret/path", json.dumps(report))

    def test_native_failed_attempt_retains_partial_phase_memory(self):
        run = self.manifest["schedule"][0]
        result = self.result(run)
        result["outcome"] = "error"
        result["exit_code"] = None
        result["phases"]["first_invocation_ticks"] = None
        result["phases"]["steady_state_ticks"] = []
        result["invocations"] = []
        result["memory"]["samples"] = result["memory"]["samples"][:1]
        self.assertEqual(native_benchmark.validate_result(result, self.manifest, run["run_id"]), [])

    def test_native_unavailable_clock_is_an_honest_early_failure(self):
        run = self.manifest["schedule"][0]
        result = self.result(run)
        result.update(outcome="error", exit_code=None, clock=None, invocations=[], memory=None)
        result["phases"] = {"compile_ticks": None, "load_ticks": None,
                            "instantiate_ticks": None, "first_invocation_ticks": None,
                            "steady_state_ticks": []}
        self.assertEqual(native_benchmark.validate_result(result, self.manifest, run["run_id"]), [])
        result["phases"]["load_ticks"] = 1
        with self.assertRaisesRegex(ValueError, "unavailable clock"):
            native_benchmark.validate_result(result, self.manifest, run["run_id"])
        result["phases"]["load_ticks"] = None
        result.update(outcome="success", exit_code=0)
        with self.assertRaisesRegex(ValueError, "unavailable clock"):
            native_benchmark.validate_result(result, self.manifest, run["run_id"])
        result.update(outcome="error", exit_code=None)

        def unavailable_clock(record, scheduled):
            if scheduled["run_id"] == run["run_id"]:
                record.update(copy.deepcopy(result))

        report = native_benchmark.build_report(
            self.manifest, self.capture_directories(unavailable_clock), allow_synthetic=True)
        self.assertIsNone(report["records"][0]["result"]["clock"])
        self.assertFalse(report["status"]["all_attempts_successful"])

    def test_native_proc_exit_preserves_zero_and_full_u32_status(self):
        run = next(run for run in self.manifest["schedule"] if run["workload"] == "coremark")
        result = self.result(run)
        for invocation in result["invocations"]:
            invocation["outcome"] = "proc_exit"
        checks = native_benchmark.validate_result(result, self.manifest, run["run_id"])
        self.assertEqual(len(checks), 3)
        result.update(outcome="error", exit_code=0xffffffff)
        result["invocations"][-1]["exit_code"] = 0xffffffff
        checks = native_benchmark.validate_result(result, self.manifest, run["run_id"])
        self.assertEqual(checks[-1]["self_check"], "failed")
        for truncated in (-1, 255, None, 2**32):
            result["exit_code"] = truncated
            with self.subTest(status=truncated), self.assertRaises(ValueError):
                native_benchmark.validate_result(result, self.manifest, run["run_id"])
        result["exit_code"] = 0xffffffff
        result["invocations"][0]["exit_code"] = 0xffffffff
        with self.assertRaisesRegex(ValueError, "continued"):
            native_benchmark.validate_result(result, self.manifest, run["run_id"])

    def test_native_matched_final_crc_disagreement_is_rejected(self):
        def change_crc(result, run):
            if run["target"] == "unikraft" and run["workload"] == "coremark":
                for invocation in result["invocations"]:
                    invocation["stdout"] = invocation["stdout"].replace("0x33ff", "0x1234")
        with self.assertRaisesRegex(ValueError, "final CRC differs"):
            native_benchmark.build_report(self.manifest, self.capture_directories(change_crc),
                                          allow_synthetic=True)

    def test_native_measurement_flag_does_not_authorize_synthetic_platform(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["evidence_kind"] = "measurement"
        for target in manifest["targets"].values():
            target["image_receipt"]["evidence_kind"] = "measurement"
        with self.assertRaisesRegex(ValueError, "synthetic platform"):
            native_benchmark.validate_manifest(manifest)

    def test_native_tampered_raw_evidence_and_stale_capture_rejected(self):
        directories = self.capture_directories()
        path = directories[0] / "observation.json"
        observation = json.loads(path.read_text())
        observation["started_at"] = "2000-01-01T00:00:00+00:00"
        path.write_text(json.dumps(observation))
        with self.assertRaisesRegex(ValueError, "outside campaign"):
            native_benchmark.build_report(self.manifest, directories, allow_synthetic=True)
        observation["started_at"] = observation["completed_at"]
        path.write_text(json.dumps(observation))
        (directories[0] / "stdout.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "raw stdout"):
            native_benchmark.build_report(self.manifest, directories, allow_synthetic=True)

    def test_native_linux_capture_executes_producer_and_retains_timeout(self):
        run = next(run for run in self.manifest["schedule"] if run["target"] == "linux")
        result = self.result(run)
        producer = self.write("synthetic_producer.py", (
            "import json,os\n"
            "request=json.load(open(os.environ['WAMR_BENCH_REQUEST']))\n"
            "assert request['config_sha256']==os.environ['WAMR_BENCH_CONFIG_SHA256']\n"
            f"print({native_benchmark.PREFIX!r}+{json.dumps(result)!r})\n").encode())
        output = self.root / "process-capture"
        self.assertTrue(native_benchmark.capture(
            self.manifest, run["run_id"], [sys.executable, str(producer)], output, 5,
            allow_synthetic=True))
        record = native_benchmark.consume_capture(self.manifest, output, allow_synthetic=True)
        self.assertEqual(record["result"]["phases"]["load_ticks"], 10)
        self.assertLess(record["observation"]["observation_seconds"], 5)
        timed_out = self.root / "timeout-capture"
        self.assertFalse(native_benchmark.capture(
            self.manifest, run["run_id"], [sys.executable, "-c", "import time; time.sleep(5)"],
            timed_out, .01, allow_synthetic=True))
        record = native_benchmark.consume_capture(self.manifest, timed_out, allow_synthetic=True)
        self.assertIsNone(record["result"])
        self.assertEqual(record["observation"]["transport_outcome"], "timeout")
        self.assertIsNone(record["observation"]["control_plane_seconds"])

    def test_native_capture_rejects_cloud_target_and_keeps_invalid_output(self):
        run = next(run for run in self.manifest["schedule"] if run["target"] == "unikraft")
        with self.assertRaisesRegex(ValueError, "only"):
            native_benchmark.capture(self.manifest, run["run_id"], [sys.executable],
                                     self.root / "cloud", 1, allow_synthetic=True)
        run = next(run for run in self.manifest["schedule"] if run["target"] == "linux")
        output = self.root / "invalid-capture"
        with self.assertRaisesRegex(ValueError, "terminal result"):
            native_benchmark.capture(self.manifest, run["run_id"],
                                     [sys.executable, "-c", "print('partial native output')"],
                                     output, 5, allow_synthetic=True)
        self.assertTrue((output / "observation.json").exists())
        self.assertEqual((output / "stdout.bin").read_bytes().splitlines(), [b"partial native output"])

    def test_native_capture_retains_launch_error(self):
        run = next(run for run in self.manifest["schedule"] if run["target"] == "linux")
        output = self.root / "launch-error"
        self.assertFalse(native_benchmark.capture(
            self.manifest, run["run_id"], [str(self.root / "missing-producer")], output, 1,
            allow_synthetic=True))
        record = native_benchmark.consume_capture(self.manifest, output, allow_synthetic=True)
        self.assertEqual(record["observation"]["transport_outcome"], "launch_error")
        self.assertIsNone(record["result"])
        self.assertTrue((output / "stderr.bin").read_bytes())


if __name__ == "__main__":
    unittest.main()
