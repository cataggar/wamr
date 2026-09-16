import copy
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import native_jit_benchmark as bench


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.executables = {"aot": {"sha256": "a" * 64, "bytes": 100},
                            "jit": {"sha256": "b" * 64, "bytes": 200}}
        self.images = {"aot": {"sha256": "c" * 64, "bytes": 1000},
                       "jit": {"sha256": "d" * 64, "bytes": 2000}}
        self.receipt = {
            "schema_version": 1, "kind": "wamr-jit-independent-image-deployment-receipt",
            "issuer": "unit-test-not-evidence", "source_commit": "a" * 40,
            "deployment_receipt_sha256": "f" * 64, "os": "linux", "arch": "x86_64",
            "hardware_execution": True, "safety": dict(bench.SAFETY),
            "platform": {"arch": "x86_64", "cpu_model": "fixture-cpu", "active_cpu_count": 1,
                         "azure_sku": "fixture-sku", "azure_region": "fixture-region"},
            "images": self.images, "executables": self.executables,
            "compiler_embedded": {"aot": False, "jit": True}, "runtime_linkage": "static",
            "lifecycle": bench.LIFECYCLE, "allocator": "caller-init-gpa",
            "page_policy": "mmap-mprotect-munmap", "wasm": self.executables["aot"],
            "aot_module": self.executables["jit"],
        }

    def test_independent_receipt_checks_all_bindings(self):
        bench.validate_receipt(self.receipt, self.executables, self.images)
        for key, value in (
            ("compiler_embedded", {"aot": True, "jit": True}),
            ("runtime_linkage", "dynamic"), ("hardware_execution", False),
            ("arch", "aarch64"), ("os", "unikraft"), ("lifecycle", "snapshot-replay"),
            ("executables", self.images), ("images", self.executables),
            ("safety", {**bench.SAFETY, "wx": False}), ("allocator", "unknown"),
        ):
            with self.subTest(key=key):
                altered = {**self.receipt, key: value}
                with self.assertRaises(ValueError):
                    bench.validate_receipt(altered, self.executables, self.images)

    def test_emulated_measurement_fails_before_process_launch(self):
        with mock.patch.object(bench.base, "artifact", side_effect=self.executables.values()), \
             mock.patch.object(bench.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "native x86_64"):
                bench.capture("aot", "jit", "unused", runner=["qemu-x86_64"],
                              receipt=self.receipt, images=self.images,
                              trusted_receipt_sha256=bench.sha(bench.encoded(self.receipt)))
            run.assert_not_called()

    def test_executable_files_cannot_be_relabelled_as_complete_images(self):
        receipt = {**self.receipt, "images": self.executables}
        with self.assertRaisesRegex(ValueError, "complete image"):
            bench.validate_receipt(receipt, self.executables, self.executables)

    def test_untrusted_receipt_fails_before_process_launch(self):
        with mock.patch.object(bench.base, "artifact", side_effect=self.executables.values()), \
             mock.patch.object(bench.platform, "system", return_value="Linux"), \
             mock.patch.object(bench.platform, "machine", return_value="x86_64"), \
             mock.patch.object(bench.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "untrusted receipt"):
                bench.capture("aot", "jit", "unused", receipt=self.receipt,
                              images=self.images, trusted_receipt_sha256="0" * 64)
            run.assert_not_called()

    def test_timeout_preserves_failed_raw_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(bench.base, "artifact", side_effect=[
                *self.executables.values(), *self.executables.values()
            ]), mock.patch.object(bench.subprocess, "run",
                                  side_effect=subprocess.TimeoutExpired("fixture", 1)):
                with self.assertRaisesRegex(ValueError, "raw output"):
                    bench.capture("aot", "jit", Path(temp) / "capture")
            for mode in bench.MODES:
                status = bench.read_json(Path(temp) / "capture" / f"{mode}.status.json")
                self.assertIn("TimeoutExpired", status["error"])
                self.assertIsNone(status["returncode"])
                self.assertEqual(status["stdout"]["bytes"], 0)
            self.assertFalse((Path(temp) / "capture" / "comparison.json").exists())


@unittest.skipUnless(os.environ.get("WAMR_JIT_BENCH_AOT") and os.environ.get("WAMR_JIT_BENCH_JIT"),
                     "build test-native-jit-bench for actual executable coverage")
class ActualMatchedSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name) / "capture"
        cls.aot = os.environ["WAMR_JIT_BENCH_AOT"]
        cls.jit = os.environ["WAMR_JIT_BENCH_JIT"]
        cls.runner = shlex.split(os.environ.get("WAMR_JIT_BENCH_RUNNER", ""))
        cls.result = bench.capture(cls.aot, cls.jit, cls.directory, runner=cls.runner)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_modes_phases_exact_results_growth_and_teardown(self):
        self.assertEqual(self.result["evidence_kind"], "correctness-only")
        self.assertIsNone(self.result["complete_image_bytes_growth"])
        self.assertEqual(set(self.result["samples"]), set(bench.MODES))
        for mode, sample in self.result["samples"].items():
            self.assertEqual(len(sample["invocations"]), 4)
            self.assertTrue(all(call["value"] == bench.expected() for call in sample["invocations"]))
            self.assertEqual(sample["memory_after"]["linear_committed_bytes"], 3 * 65536)
            self.assertEqual(sample["caller_live_after_teardown"], 0)
            self.assertEqual(sample["reserved_after_teardown"], 0)
            self.assertEqual(sample["compile_ns"] is None, mode == "aot")
        fixtures = Path(self.aot).parent.parent / "native-jit-bench"
        source = bench.base.artifact(fixtures / "matched.wasm")
        module = bench.base.artifact(fixtures / "matched.cwasm")
        sample = self.result["samples"]["aot"]
        self.assertEqual(source, {"sha256": sample["wasm_sha256"], "bytes": sample["wasm_bytes"]})
        self.assertEqual(module, {"sha256": sample["cwasm_sha256"], "bytes": sample["cwasm_bytes"]})

    def test_comparator_has_no_compiler_symbols(self):
        aot = subprocess.check_output(["nm", "-a", self.aot], text=True)
        jit = subprocess.check_output(["nm", "-a", self.jit], text=True)
        markers = ("aot_compile.compileCoreWasm", "api.jit.compile",
                   "compiler.frontend.", "compiler.codegen.x86_64.compile.")
        self.assertFalse(any(marker in aot for marker in markers),
                         "AOT comparator contains compiler symbols")
        self.assertTrue(any(marker in jit for marker in markers),
                        "JIT sampler lacks compiler symbols")

    def test_strict_sample_rejects_false_results_modes_and_metrics(self):
        original = self.result["samples"]["fast"]
        for key, value in (
            ("compiler_embedded", False), ("compile_ns", None),
            ("compiler_peak_bytes", 65 * 1024 * 1024), ("fuel_per_invocation", None),
            ("caller_live_after_teardown", 1), ("growth_previous_pages", 3),
            ("request_sha256", "0" * 64), ("expected", 0), ("extra", True),
        ):
            with self.subTest(key=key):
                altered = {**original, key: value}
                with self.assertRaises(ValueError):
                    bench.validate_sample(altered, "fast", original["request_sha256"])
        changed = copy.deepcopy(original)
        changed["invocations"][1]["value"] = 0
        with self.assertRaises(ValueError):
            bench.validate_sample(changed, "fast", original["request_sha256"])

    def test_raw_capture_revalidation_detects_changes(self):
        raw = self.directory / "full.stdout"
        saved = raw.read_bytes()
        try:
            raw.write_bytes(saved + b"\n")
            with self.assertRaisesRegex(ValueError, "altered raw stdout"):
                bench.report(self.directory)
        finally:
            raw.write_bytes(saved)
        self.assertEqual(bench.report(self.directory), self.result)

    def test_compiler_free_binary_rejects_jit_mode_without_fallback(self):
        result = subprocess.run([*self.runner, self.aot, "fast", "0" * 64],
                                capture_output=True, check=False, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(bench.PREFIX, result.stdout)


if __name__ == "__main__":
    unittest.main()
