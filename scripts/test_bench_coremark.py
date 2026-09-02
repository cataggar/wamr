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
Iterations/Sec   : 12345.5
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
        self.assertIn("- profile", workflow)
        self.assertIn(
            'default: "e32b7b7d2d12007eb66679a66f943b1e4ea6a393"',
            workflow,
        )
        dispatch = workflow.split(
            "- name: Run authoritative same-host CoreMark comparison", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertIn("github.event_name == 'workflow_dispatch'", dispatch)
        self.assertIn("--profile  authoritative", dispatch)
        self.assertIn("--wasmtime-baseline auto", dispatch)
        self.assertIn("--require-native-arch aarch64", dispatch)
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

        for line in workflow.splitlines():
            stripped = line.strip()
            if not stripped.startswith("uses:") or "uses: ./" in stripped:
                continue
            action = stripped.split("#", 1)[0].strip()
            self.assertRegex(action, r"@[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
