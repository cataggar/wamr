#!/usr/bin/env python3

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


class BenchCoremarkTests(unittest.TestCase):
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
        self.assertIn("0.604×", report)
        self.assertIn("CRC validation", report)
        self.assertIn("Neoverse-N2", report)
        self.assertIn("host fingerprint", report)

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
        self.assertIn(
            'default: "19d046a5b23b9c39acf5f7062976f04c5ca8ca75"',
            workflow,
        )
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
        self.assertIn('--wamr-ref "$PROFILE_REF"', profile_step)
        self.assertIn("--min-samples 1000", profile_step)

        profile_script = PROFILE_SCRIPT.read_text()
        self.assertIn("--profile=jitdump", profile_script)
        self.assertIn("WAMR_AOT_SPILL_METRIC", profile_script)
        self.assertIn("WAMR_AOT_CODEGEN_TIMING", profile_script)
        self.assertIn('"cycles:u"', profile_script)
        self.assertIn("AUTHORITATIVE_BASELINE_RUN = 33631050708", profile_script)
        self.assertIn("select_cpu_affinity()", profile_script)
        self.assertIn("coremark_guest_args(", profile_script)
        self.assertIn("PROFILE_CAPTURES_PER_ENGINE = 2", profile_script)

        for line in workflow.splitlines():
            stripped = line.strip()
            if not stripped.startswith("uses:") or "uses: ./" in stripped:
                continue
            action = stripped.split("#", 1)[0].strip()
            self.assertRegex(action, r"@[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
