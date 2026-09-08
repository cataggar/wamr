#!/usr/bin/env python3

import copy
import io
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_coremark


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
        results = [
            bench_coremark.EngineResult(
                "WAMR",
                "baseline",
                "ReleaseFast",
                [100.0] * 10,
                identity=wamr_identity("baseline", "a" * 40, "1" * 64),
            ),
            bench_coremark.EngineResult(
                "WAMR",
                "target",
                "ReleaseFast",
                [110.0] * 10,
                identity=target_identity,
            ),
            bench_coremark.EngineResult(
                "Wasmtime historical pin",
                "44.0.1",
                "default JIT",
                [220.0] * 10,
                identity=wasmtime_identity("3" * 64),
            ),
        ]
        records = []
        position = 0
        for phase, count in (("warmup", 2), ("measured", 10)):
            ordinals = {"target": 0, "wasmtime": 0}
            for key in bench_coremark.counterbalanced_order(
                ["target", "wasmtime"], count
            ):
                position += 1
                ordinals[key] += 1
                records.append(
                    bench_coremark.SampleRecord(
                        key,
                        "WAMR target" if key == "target" else "Wasmtime",
                        phase,
                        ordinals[key],
                        position,
                        "2026-09-08T00:00:00+00:00",
                        "2026-09-08T00:00:01+00:00",
                        1.0,
                        100.0,
                        bench_coremark.EXPECTED_ITERATIONS,
                    )
                )
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
        for engine in report["engines"]:
            if engine["role"] == "wamr-target":
                engine["sample_schedule_positions"] = [
                    record.schedule_position
                    for record in records
                    if record.engine_key == "target"
                ]
            elif engine["role"] == "wasmtime-baseline":
                engine["sample_schedule_positions"] = [
                    record.schedule_position
                    for record in records
                    if record.engine_key == "wasmtime"
                ]
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
        shutil.rmtree(root, ignore_errors=True)
        source.mkdir(parents=True)
        wamr = source / "wamr"
        wamrc = source / "wamrc"
        cwasm = source.parent.parent / ".bench-coremark.cwasm"
        wamr.write_bytes(b"wamr-with-build-path-a")
        wamrc.write_bytes(b"wamrc-with-build-path-a")
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
            ),
        )
        try:
            retained = bench_coremark.retain_wamr_artifact_handoff(
                prepared, artifact_dir
            )
            manifest = (artifact_dir / "manifest.json").read_bytes()
            for existing_dir in (artifact_dir, root):
                with self.subTest(existing_dir=existing_dir):
                    with self.assertRaises(FileExistsError):
                        bench_coremark.retain_wamr_artifact_handoff(
                            prepared, existing_dir
                        )
                    self.assertEqual(
                        manifest, (artifact_dir / "manifest.json").read_bytes()
                    )
                    self.assertEqual(b"wamr-with-build-path-a", wamr.read_bytes())
            shutil.rmtree(root / "temporary-benchmark-worktree")
            loaded = bench_coremark.load_wamr_artifact_handoff(
                artifact_dir, retained
            )
            self.assertEqual(b"wamr-with-build-path-a", loaded["runtime"].read_bytes())
            self.assertNotEqual(wamr.parent, loaded["runtime"].parent)

            loaded["runtime"].write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                bench_coremark.load_wamr_artifact_handoff(
                    artifact_dir, retained
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)

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
        cwasm = build_dir / "coremark.cwasm"
        wamr.write_bytes(b"wamr-benchmark-build")
        wamrc.write_bytes(b"wamrc-benchmark-build")
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
        self.assertNotIn("19d046a5b23b9c39acf5f7062976f04c5ca8ca75", workflow)
        self.assertIn('profile_sha="$(git rev-parse "$profile_ref")"', workflow)
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


if __name__ == "__main__":
    unittest.main()
