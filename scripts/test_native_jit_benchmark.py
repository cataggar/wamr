import copy
import os
import shlex
import subprocess
import sys
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


class NativeExternalTransportTests(unittest.TestCase):
    """Synthetic UNIT TEST fixtures only: none are image/deployment evidence."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {}
        for name in ("aot", "jit", "aot-image", "jit-image"):
            self.paths[name] = self.root / name
            self.paths[name].write_bytes(f"UNIT TEST ONLY, NOT AN IMAGE OR EXECUTABLE: {name}".encode())
        self.executables = {mode: bench.base.artifact(self.paths[mode]) for mode in ("aot", "jit")}
        self.images = {mode: bench.base.artifact(self.paths[f"{mode}-image"]) for mode in ("aot", "jit")}
        legacy = QualificationTests()
        legacy.setUp()
        self.receipt = {
            **legacy.receipt, "schema_version": 2, "evidence_kind": "measurement", "os": "unikraft",
            "images": self.images, "executables": self.executables,
            "allocator": "unit-test-allocator", "page_policy": "unit-test-page-policy",
            "target": copy.deepcopy(bench.NATIVE_TARGET), "options": copy.deepcopy(bench.NATIVE_OPTIONS),
            "build_options": {**bench.NATIVE_BUILD_OPTIONS, "optimize": "ReleaseSafe"},
            "adapter": {"name": "unit-test-not-qualified", "source_commit": "b" * 40,
                        "qualification_receipt_sha256": "9" * 64},
            "clock": {"method": "unit-test-clock", "resolution_ns": 1, "scope": "guest-monotonic-execution"},
            "native_stack": {"generated_frames_bytes": 256 * 1024,
                             "compiler_embedder_callbacks_bytes": 1024 * 1024,
                             "provisioned_bytes": 2 * 1024 * 1024},
            "memory_observer": {"method": "unit-test-not-an-observation",
                                "quantity": "physical-backing-bytes",
                                "coverage": "caller-allocator-and-native-pages",
                                "excludes": ["image", "native-stack", "other-kernel-allocations", "page-tables"],
                                "sampling": "continuous-high-water", "interval_ns": None},
        }
        self.receipt_sha = bench.sha(bench.encoded(self.receipt))
        self.directory = self.root / "private"
        self.request = bench.prepare_native(
            self.paths["aot"], self.paths["jit"], self.directory, images=self.images,
            receipt=self.receipt, trusted_receipt_sha256=self.receipt_sha)
        self.request_sha = bench.sha((self.directory / "request.json").read_bytes())
        self.external = self.root / "unit-test-external"
        self.external.mkdir()
        self.capture = {
            "schema_version": 1, "kind": "wamr-jit-native-capture-receipt",
            "evidence_kind": "measurement", "request_sha256": self.request_sha,
            "image_receipt_sha256": self.receipt_sha, "hardware_execution": True,
            "adapter_qualification_sha256": self.receipt["adapter"]["qualification_receipt_sha256"],
            "records": {},
        }
        for mode in bench.MODES:
            sample = self.synthetic_sample(mode)
            raw = b"\xff\xfeUNIT TEST BOOT NOISE\r\n" + bench.PREFIX + bench.encoded(sample) + b"\r\n\x80noise\n"
            (self.external / f"{mode}.serial").write_bytes(raw)
            image = "aot" if mode == "aot" else "jit"
            self.capture["records"][mode] = {
                "image": self.images[image], "executable": self.executables[image],
                "serial": bench.stream_artifact(self.external / f"{mode}.serial"),
                "outcome": "success", "capture_complete": True,
                "started_at": self.request["created_at"], "completed_at": self.request["created_at"],
                "memory": {"before_bytes": 4096, "observed_max_bytes": 2 * 1024 * 1024,
                           "after_teardown_bytes": 4096, "observation_count": 3},
            }
        (self.external / "capture.json").write_bytes(bench.encoded(self.capture))

    def synthetic_sample(self, mode):
        compiler = mode != "aot"
        memory = {"heap_live_bytes": 4096, "heap_peak_bytes": 8192, "code_bytes": 1024,
                  "code_reserved_bytes": 4096, "linear_reserved_bytes": 8 * 65536,
                  "linear_committed_bytes": 2 * 65536}
        sample = {
            "schema_version": 1, "kind": "wamr-native-jit-sample",
            "qualification": "requires-independent-image-and-deployment-evidence",
            "request_sha256": self.request_sha, "mode": mode, "compiler_embedded": compiler,
            "wasm_sha256": self.receipt["wasm"]["sha256"], "wasm_bytes": self.receipt["wasm"]["bytes"],
            "cwasm_sha256": self.receipt["aot_module"]["sha256"],
            "cwasm_bytes": 4096 if compiler else self.receipt["aot_module"]["bytes"],
            "workload": "volatile-compute-memory-2000", "expected": bench.expected(),
            "lifecycle": bench.LIFECYCLE, "clock_resolution_ns": 1,
            "compile_ns": 100 if compiler else None,
            "compiler_phases_ns": dict.fromkeys(("parse", "lower", "optimize", "codegen", "emit"), 10) if compiler else None,
            "compiler_peak_bytes": 65536 if compiler else 0, "compiler_retained_bytes": 4096 if compiler else 0,
            "compiler_polls": 100 if compiler else 0, "load_ns": 10, "instantiate_ns": 10,
            "start_ns": 10, "growth_ns": 10, "growth_previous_pages": 2,
            "fuel_per_invocation": 100000 if compiler else None,
            "invocations": [{"ns": 10, "outcome": "returned", "value": bench.expected(), "diagnostic": None}
                            for _ in range(4)],
            "memory_before": memory, "memory_after": {**memory, "linear_committed_bytes": 3 * 65536},
            "caller_peak_bytes": 100000, "caller_live_after_teardown": 0, "reserved_after_teardown": 0,
            "failure_stage": None, "failure": None,
        }
        self.assertEqual(set(sample), bench.SAMPLE_KEYS)
        return sample

    def import_capture(self, trusted=None):
        return bench.import_native(
            self.directory, self.external, aot=self.paths["aot"], jit=self.paths["jit"], images=self.images,
            trusted_receipt_sha256=self.receipt_sha,
            trusted_capture_sha256=trusted or bench.sha(bench.encoded(self.capture)))

    def archive(self):
        for name in ("capture.json", *(f"{mode}.serial" for mode in bench.MODES)):
            bench.base.private_write(self.directory / name, (self.external / name).read_bytes())

    def revalidate(self, capture=None):
        capture = capture or self.capture
        (self.directory / "capture.json").write_bytes(bench.encoded(capture))
        return bench.native_report(self.directory, self.receipt_sha, bench.sha(bench.encoded(capture)))

    def test_qualified_external_import_is_host_independent_private_and_clock_truthful(self):
        with mock.patch.object(bench.platform, "system", return_value="Darwin"), \
             mock.patch.object(bench.platform, "machine", return_value="aarch64"), \
             mock.patch.object(bench.subprocess, "run") as run:
            result = self.import_capture()
        run.assert_not_called()
        self.assertEqual(result["transport"], "external-native")
        self.assertEqual(result["images"], self.images)
        self.assertEqual(result["native_memory_observer"], self.receipt["memory_observer"])
        self.assertEqual(result["native_memory_observations"]["fast"], self.capture["records"]["fast"]["memory"])
        self.assertEqual(result["samples"]["fast"]["compile_ns"], 100)
        self.assertIn("not-serial-arrival", result["timing_scope"])
        self.assertIn("not-hardware-proof", result["qualification"])
        self.assertEqual(self.directory.stat().st_mode & 0o777, 0o700)
        for path in self.directory.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(bench.native_report(self.directory, self.receipt_sha,
                                            bench.sha(bench.encoded(self.capture))), result)
        with self.assertRaisesRegex(ValueError, "already attempted"):
            self.import_capture()

    def test_native_image_qualification_cannot_relabel_linux_or_software_metrics(self):
        mutations = [
            ("schema_version", 1), ("os", "linux"), ("evidence_kind", "correctness-only"),
            ("hardware_execution", False), ("target", {**bench.NATIVE_TARGET, "abi": "linux"}),
            ("options", {**bench.NATIVE_OPTIONS, "rounds": 0}),
            ("build_options", {**self.receipt["build_options"], "red_zone": True}),
            ("build_options", {**self.receipt["build_options"], "pic": False}),
            ("adapter", {}), ("clock", {**self.receipt["clock"], "scope": "collector-latency"}),
            ("native_stack", {**self.receipt["native_stack"], "generated_frames_bytes": 1024}),
            ("memory_observer", {**self.receipt["memory_observer"], "quantity": "logical-page-commitment"}),
            ("memory_observer", {**self.receipt["memory_observer"], "excludes": []}),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    bench.validate_native_receipt({**self.receipt, key: value}, self.executables, self.images)
        with self.assertRaises(ValueError):
            bench.validate_receipt(self.receipt, self.executables, self.images)

    def test_native_capture_receipt_bindings_and_real_memory_are_required(self):
        self.archive()
        for key, value in (("request_sha256", "0" * 64), ("image_receipt_sha256", "0" * 64),
                           ("adapter_qualification_sha256", "0" * 64), ("hardware_execution", False),
                           ("evidence_kind", "correctness-only"), ("schema_version", True)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.revalidate({**self.capture, key: value})
        for key, value in (("image", self.images["aot"]), ("executable", self.executables["aot"]),
                           ("outcome", "timeout"), ("capture_complete", False),
                           ("started_at", "2000-01-01T00:00:00+00:00"),
                           ("memory", None), ("memory", {"requested_bytes": 100})):
            with self.subTest(key=key):
                altered = copy.deepcopy(self.capture)
                altered["records"]["fast"][key] = value
                with self.assertRaises(ValueError):
                    self.revalidate(altered)
        self.revalidate()

    def test_duplicate_truncated_non_utf8_and_oversized_records_fail_despite_attestation(self):
        self.archive()
        raw = self.directory / "fast.serial"
        saved = raw.read_bytes()
        encoded_sample = bench.PREFIX + bench.encoded(self.synthetic_sample("fast"))
        for malformed in (
            saved + encoded_sample + b"\n", encoded_sample, b"noise\n" + bench.PREFIX + b"\xff\n",
            b"not-a-line:" + encoded_sample + b"\n", bench.PREFIX + b" " * bench.MAX_RECORD_BYTES + b"\n",
            bench.PREFIX + b'{"mode":"fast","mode":"fast"}\n',
            b"WAMR_BENCH_RESULT={}\n", b"boot noise only\n",
        ):
            with self.subTest(raw=malformed[:40]):
                raw.write_bytes(malformed)
                altered = copy.deepcopy(self.capture)
                altered["records"]["fast"]["serial"] = bench.stream_artifact(raw)
                with self.assertRaises(ValueError):
                    self.revalidate(altered)
        raw.write_bytes(saved)
        self.revalidate()

    def test_sample_failures_stale_input_and_clock_mismatch_reject(self):
        self.archive()
        raw = self.directory / "fast.serial"
        for key, value in (("failure", "ClockFailed"), ("request_sha256", "0" * 64),
                           ("wasm_sha256", "0" * 64), ("clock_resolution_ns", 2),
                           ("caller_live_after_teardown", 1), ("compiler_phases_ns", None),
                           ("growth_previous_pages", 1)):
            with self.subTest(key=key):
                sample = {**self.synthetic_sample("fast"), key: value}
                raw.write_bytes(bench.PREFIX + bench.encoded(sample) + b"\n")
                altered = copy.deepcopy(self.capture)
                altered["records"]["fast"]["serial"] = bench.stream_artifact(raw)
                with self.assertRaises(ValueError):
                    self.revalidate(altered)

    def test_untrusted_or_failed_capture_retains_private_raw_without_comparison(self):
        with self.assertRaisesRegex(ValueError, "untrusted native capture"):
            self.import_capture(trusted="0" * 64)
        for mode in bench.MODES:
            self.assertEqual((self.directory / f"{mode}.serial").read_bytes(),
                             (self.external / f"{mode}.serial").read_bytes())
            self.assertEqual((self.directory / f"{mode}.serial").stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.directory / "comparison.json").exists())
        self.assertFalse(bench.read_json(self.directory / "import.status.json")["success"])

    def test_native_raw_and_prepared_artifact_tampering_reject(self):
        self.paths["jit"].write_bytes(b"changed unit-test executable")
        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            self.import_capture()
        self.archive()
        with (self.directory / "full.serial").open("ab") as raw:
            raw.write(b"\n")
        with self.assertRaisesRegex(ValueError, "altered raw native serial"):
            self.revalidate()

    def test_native_import_never_accepts_linux_correctness_request(self):
        request = {**self.request, "schema_version": 1, "evidence_kind": "correctness-only"}
        with self.assertRaises(ValueError):
            bench.validate_native_request(request, self.receipt_sha)
        with self.assertRaisesRegex(ValueError, "untrusted native image"):
            bench.validate_native_request(self.request, "0" * 64)
        with self.assertRaisesRegex(ValueError, "request: expected"):
            bench.report(self.directory)

    def test_native_cli_requires_real_artifacts_and_both_independent_trust_inputs(self):
        command = [sys.executable, "-m", "scripts.native_jit_benchmark",
                   "--aot", str(self.paths["aot"]), "--jit", str(self.paths["jit"]),
                   "--output", str(self.root / "cli")]
        result = subprocess.run([*command, "--native-prepare"], capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"native transport needs complete images", result.stderr)
        result = subprocess.run([*command, "--trusted-capture-sha256", "0" * 64],
                                capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"cannot qualify a Linux capture", result.stderr)
        self.assertFalse((self.root / "cli").exists())


if __name__ == "__main__":
    unittest.main()
