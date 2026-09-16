"""Explicit matched JIT experiment. This does not extend the native v2 AOT protocol."""
import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

# Existing benchmark tools also support direct script imports from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_benchmark as base

MODES = ("aot", "fast", "full")
PREFIX = b"WAMR_JIT_SAMPLE="
LIFECYCLE = "same-instance-workload-initializes-memory-in-timed-call"
SAMPLE_KEYS = set("""
schema_version kind qualification request_sha256 mode compiler_embedded
wasm_sha256 wasm_bytes cwasm_sha256 cwasm_bytes workload expected lifecycle
clock_resolution_ns compile_ns compiler_phases_ns compiler_peak_bytes
compiler_retained_bytes compiler_polls load_ns instantiate_ns start_ns growth_ns
growth_previous_pages fuel_per_invocation invocations memory_before memory_after
caller_peak_bytes caller_live_after_teardown reserved_after_teardown failure_stage failure
""".split())
MEMORY_KEYS = set("""
heap_live_bytes heap_peak_bytes code_bytes code_reserved_bytes
linear_reserved_bytes linear_committed_bytes
""".split())
SAFETY = {"bounds_checks": True, "checked_imports": True, "checked_traps": True, "wx": True}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_json(path):
    return base.strict_json(Path(path).read_text())


def stream_artifact(path):
    return {"sha256": base.sha256_file(Path(path)), "bytes": Path(path).stat().st_size}


def expected():
    cells = [i * 17 + 3 for i in range(256)]
    result = 0
    for i in range(2000):
        index = i & 255
        cells[index] = ((cells[index] * 3 + i) if i & 1 == 0
                        else cells[index] ^ (i * 7)) & 0xffffffff
        result = (result + cells[index]) & 0xffffffff
    return result


def validate_receipt(receipt, executables, images):
    """An out-of-band trusted hash pins this attestation; it is not hardware proof."""
    base.keys(receipt, {"schema_version", "kind", "issuer", "source_commit",
                       "deployment_receipt_sha256", "os", "arch", "hardware_execution",
                       "platform", "safety", "images", "executables", "compiler_embedded",
                       "runtime_linkage", "lifecycle", "allocator", "page_policy",
                       "wasm", "aot_module"}, "receipt")
    base.require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1 and receipt["kind"] ==
                 "wamr-jit-independent-image-deployment-receipt", "receipt schema")
    base.public_token(receipt["issuer"], "independent receipt issuer")
    base.require(isinstance(receipt["source_commit"], str) and
                 len(receipt["source_commit"]) == 40 and
                 all(c in "0123456789abcdef" for c in receipt["source_commit"]), "source commit")
    base.digest(receipt["deployment_receipt_sha256"], "deployment receipt")
    base.require(receipt["os"] == "linux" and receipt["arch"] == "x86_64" and
                 receipt["hardware_execution"] is True, "only independently qualified native Linux capture")
    base.validate_platform(receipt["platform"])
    base.require(receipt["platform"]["arch"] == "x86_64", "platform architecture mismatch")
    base.require(base.identical(receipt["safety"], SAFETY), "required safety protections")
    base.require(base.identical(receipt["compiler_embedded"], {"aot": False, "jit": True}),
                 "separate compiler-free AOT comparator required")
    base.require(receipt["runtime_linkage"] == "static" and receipt["lifecycle"] == LIFECYCLE,
                 "linkage/lifecycle mismatch")
    for field in ("wasm", "aot_module"):
        base.validate_artifact(receipt[field], field)
    for field in ("allocator", "page_policy"):
        base.public_token(receipt[field], field)
        base.require(receipt[field].lower() not in ("unknown", "unspecified", "placeholder"),
                     f"unavailable {field}")
    for field, actual in (("executables", executables), ("images", images)):
        base.keys(receipt[field], {"aot", "jit"}, field)
        for name in ("aot", "jit"):
            base.validate_artifact(receipt[field][name], f"{field}.{name}")
        base.require(base.identical(receipt[field], actual), f"{field} receipt mismatch")
        base.require(actual["aot"] != actual["jit"], f"separate {field} required")
    for name in ("aot", "jit"):
        base.require(images[name]["sha256"] != executables[name]["sha256"],
                     "executable file size cannot substitute for a complete image")


def validate_sample(sample, mode, request_sha):
    base.keys(sample, SAMPLE_KEYS, "JIT sample")
    base.require(type(sample["schema_version"]) is int and sample["schema_version"] == 1 and
                 sample["kind"] == "wamr-native-jit-sample",
                 "sample schema")
    base.require(sample["qualification"] == "requires-independent-image-and-deployment-evidence",
                 "producer cannot qualify itself")
    base.require(sample["request_sha256"] == request_sha and sample["mode"] == mode,
                 "sample/request mismatch")
    base.require(sample["compiler_embedded"] is (mode != "aot"), "compiler identity mismatch")
    base.require(sample["failure"] is None and sample["failure_stage"] is None, "sample failed")
    base.require(sample["workload"] == "volatile-compute-memory-2000" and
                 sample["expected"] == expected() and sample["lifecycle"] == LIFECYCLE,
                 "workload/lifecycle mismatch")
    for field in ("wasm_sha256", "cwasm_sha256"):
        base.digest(sample[field], field)
    for field in ("wasm_bytes", "cwasm_bytes", "clock_resolution_ns", "caller_peak_bytes"):
        base.number(sample[field], field, 1, integer=True)
    for field in ("load_ns", "instantiate_ns", "start_ns", "growth_ns"):
        base.number(sample[field], field, integer=True)
    base.require(sample["growth_previous_pages"] == 2, "growth result")
    for field in ("caller_live_after_teardown", "reserved_after_teardown"):
        base.number(sample[field], field, integer=True)
    base.require(sample["caller_live_after_teardown"] == 0 and
                 sample["reserved_after_teardown"] == 0, "incomplete teardown")
    base.require(isinstance(sample["invocations"], list) and len(sample["invocations"]) == 4,
                 "one first and three repeated calls required")
    for call in sample["invocations"]:
        base.keys(call, {"ns", "outcome", "value", "diagnostic"}, "invocation")
        base.number(call["ns"], "invocation time", integer=True)
        base.require(call["outcome"] == "returned" and call["value"] == expected() and
                     call["diagnostic"] is None, "incorrect workload result")
    for stage in ("memory_before", "memory_after"):
        base.keys(sample[stage], MEMORY_KEYS, stage)
        for field in MEMORY_KEYS:
            base.number(sample[stage][field], field, 1, integer=True)
        base.require(sample[stage]["heap_live_bytes"] <= sample[stage]["heap_peak_bytes"],
                     "invalid heap accounting")
        base.require(sample[stage]["heap_peak_bytes"] <= 16 * 1024 * 1024 and
                     sample[stage]["code_bytes"] <= 4 * 1024 * 1024 and
                     sample[stage]["code_reserved_bytes"] + sample[stage]["linear_reserved_bytes"] <=
                     8 * 1024 * 1024, "runtime caps exceeded")
    before, after = sample["memory_before"], sample["memory_after"]
    base.require(before["linear_committed_bytes"] == 2 * 65536 and
                 after["linear_committed_bytes"] == 3 * 65536 and
                 before["linear_reserved_bytes"] == after["linear_reserved_bytes"] == 8 * 65536,
                 "unobserved or incorrect memory growth")
    if mode == "aot":
        base.require(sample["compile_ns"] is None and sample["compiler_phases_ns"] is None and
                     sample["fuel_per_invocation"] is None, "AOT must not compile or claim fuel")
        base.require(all(sample[key] == 0 for key in
                         ("compiler_peak_bytes", "compiler_retained_bytes", "compiler_polls")),
                     "compiler-free AOT reported compiler work")
    else:
        base.number(sample["compile_ns"], "compile time", integer=True)
        base.keys(sample["compiler_phases_ns"], {"parse", "lower", "optimize", "codegen", "emit"},
                  "compiler phases")
        for value in sample["compiler_phases_ns"].values():
            base.number(value, "compiler phase time", integer=True)
        base.require(sum(sample["compiler_phases_ns"].values()) <= sample["compile_ns"],
                     "compiler phases exceed encompassing compile interval")
        for key in ("compiler_peak_bytes", "compiler_retained_bytes", "compiler_polls"):
            base.number(sample[key], key, 1, integer=True)
        base.require(sample["compiler_retained_bytes"] <= sample["compiler_peak_bytes"] <=
                     64 * 1024 * 1024 and sample["compiler_polls"] <= 100000,
                     "compiler limits exceeded")
        base.require(sample["fuel_per_invocation"] == 100000, "incorrect runtime fuel")
    return sample


def report(directory, trusted_receipt_sha256=None):
    directory = Path(directory)
    request_bytes = (directory / "request.json").read_bytes()
    request = base.strict_json(request_bytes.decode())
    base.keys(request, {"schema_version", "kind", "evidence_kind", "nonce", "executables",
                       "receipt", "receipt_sha256", "images", "runner"}, "request")
    base.require(type(request["schema_version"]) is int and request["schema_version"] == 1 and
                 request["kind"] == "wamr-jit-comparison-request",
                 "request schema")
    base.require(str(uuid.UUID(request["nonce"])) == request["nonce"], "nonce")
    base.require(request["evidence_kind"] in ("correctness-only", "measurement"), "evidence kind")
    base.keys(request["executables"], {"aot", "jit"}, "executables")
    for value in request["executables"].values():
        base.validate_artifact(value, "executable")
    base.require(request["executables"]["aot"] != request["executables"]["jit"],
                 "separate executables required")
    if request["evidence_kind"] == "measurement":
        base.require(not request["runner"], "emulated measurement forbidden")
        base.digest(trusted_receipt_sha256, "out-of-band trusted receipt hash")
        base.require(request["receipt_sha256"] == trusted_receipt_sha256 and
                     sha(encoded(request["receipt"])) == trusted_receipt_sha256, "untrusted receipt")
        validate_receipt(request["receipt"], request["executables"], request["images"])
    else:
        base.require(request["receipt"] is None and request["receipt_sha256"] is None and
                     request["images"] is None, "correctness captures cannot be upgraded to measurements")
    samples = {}
    for mode in MODES:
        status = read_json(directory / f"{mode}.status.json")
        base.keys(status, {"returncode", "error", "stdout", "stderr"}, "capture status")
        for stream in ("stdout", "stderr"):
            base.require(status[stream] == stream_artifact(directory / f"{mode}.{stream}"),
                         f"altered raw {stream}")
        base.require(type(status["returncode"]) is int and status["returncode"] == 0 and status["error"] is None,
                     f"{mode} failed; raw output and status retained")
        lines = (directory / f"{mode}.stdout").read_bytes().splitlines()
        base.require(len(lines) == 1 and lines[0].startswith(PREFIX), "exactly one sampler record required")
        samples[mode] = validate_sample(base.strict_json(lines[0][len(PREFIX):].decode()),
                                        mode, sha(request_bytes))
    base.require(len({(s["wasm_sha256"], s["wasm_bytes"]) for s in samples.values()}) == 1,
                 "unmatched input modules")
    if request["receipt"] is not None:
        for field, actual in (
            ("wasm", {"sha256": samples["aot"]["wasm_sha256"], "bytes": samples["aot"]["wasm_bytes"]}),
            ("aot_module", {"sha256": samples["aot"]["cwasm_sha256"], "bytes": samples["aot"]["cwasm_bytes"]}),
        ):
            base.require(request["receipt"][field] == actual, f"receipt {field} mismatch")
    return {
        "schema_version": 1, "kind": "wamr-jit-comparison", "request_sha256": sha(request_bytes),
        "evidence_kind": request["evidence_kind"], "samples": samples,
        "executable_bytes_growth": request["executables"]["jit"]["bytes"] -
                                   request["executables"]["aot"]["bytes"],
        "complete_image_bytes_growth": None if request["images"] is None else
            request["images"]["jit"]["bytes"] - request["images"]["aot"]["bytes"],
        "caller_requested_peak_growth": {mode: samples[mode]["caller_peak_bytes"] -
                                        samples["aot"]["caller_peak_bytes"] for mode in MODES[1:]},
        "memory_scope": "requested-allocator-bytes-and-logical-page-commitment-not-RSS",
        "qualification": "correctness-only-no-performance-claim" if request["evidence_kind"] ==
                         "correctness-only" else "bound-to-independently-supplied-attestation-not-hardware-proof",
    }


def capture(aot, jit, output, *, runner=(), timeout=120, receipt=None,
            trusted_receipt_sha256=None, images=None):
    base.number(timeout, "process timeout", 1)
    base.require(timeout <= 600, "process timeout exceeds 600 seconds")
    executables = {"aot": base.artifact(aot), "jit": base.artifact(jit)}
    base.require(executables["aot"] != executables["jit"], "separate AOT/JIT binaries required")
    receipt_hash = None
    if receipt is not None:
        base.require(not runner and platform.system() == "Linux" and platform.machine() == "x86_64",
                     "measurement requires native x86_64 Linux without an emulator")
        base.digest(trusted_receipt_sha256, "out-of-band trusted receipt hash")
        receipt_hash = sha(encoded(receipt))
        base.require(receipt_hash == trusted_receipt_sha256, "untrusted receipt")
        validate_receipt(receipt, executables, images)
    else:
        base.require(images is None and trusted_receipt_sha256 is None,
                     "image evidence needs an independent deployment receipt")
    request = {"schema_version": 1, "kind": "wamr-jit-comparison-request",
               "evidence_kind": "measurement" if receipt is not None else "correctness-only",
               "nonce": str(uuid.uuid4()), "executables": executables, "receipt": receipt,
               "receipt_sha256": receipt_hash, "images": images, "runner": list(runner)}
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    request_bytes = encoded(request)
    (directory / "request.json").write_bytes(request_bytes)
    for mode in MODES:
        binary = aot if mode == "aot" else jit
        status = {"returncode": None, "error": None}
        with (directory / f"{mode}.stdout").open("wb") as stdout, \
             (directory / f"{mode}.stderr").open("wb") as stderr:
            try:
                result = subprocess.run([*runner, str(Path(binary).resolve()), mode, sha(request_bytes)],
                                        stdout=stdout, stderr=stderr, timeout=timeout, check=False)
                status["returncode"] = result.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                status["error"] = f"{type(error).__name__}: {error}"
        for stream in ("stdout", "stderr"):
            status[stream] = stream_artifact(directory / f"{mode}.{stream}")
        (directory / f"{mode}.status.json").write_bytes(encoded(status))
    base.require(executables == {"aot": base.artifact(aot), "jit": base.artifact(jit)},
                 "executables changed during capture")
    result = report(directory, trusted_receipt_sha256)
    (directory / "comparison.json").write_bytes(encoded(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aot", required=True)
    parser.add_argument("--jit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner", default="", help="correctness-only, e.g. qemu-x86_64 -cpu max")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--receipt", help="independently supplied image/deployment attestation")
    parser.add_argument("--trusted-receipt-sha256", help="out-of-band SHA256 of canonical receipt JSON")
    parser.add_argument("--aot-image")
    parser.add_argument("--jit-image")
    args = parser.parse_args()
    images, receipt = None, None
    evidence = (args.aot_image, args.jit_image, args.receipt, args.trusted_receipt_sha256)
    if any(evidence):
        if not all(evidence):
            parser.error("measurement needs both complete images, independent receipt, and its trusted hash")
        images = {"aot": base.artifact(args.aot_image), "jit": base.artifact(args.jit_image)}
        receipt = read_json(args.receipt)
    capture(args.aot, args.jit, args.output, runner=shlex.split(args.runner),
            timeout=args.timeout, images=images, receipt=receipt,
            trusted_receipt_sha256=args.trusted_receipt_sha256)


if __name__ == "__main__":
    main()
