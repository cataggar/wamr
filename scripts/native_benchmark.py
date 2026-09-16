#!/usr/bin/env python3
"""Native guest adapter for bench_coremark.py; never an in-guest Python runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from benchmark_schema import atomic_write_json, cache_key, sample_stats, sha256_file
from bench_coremark import (
    COREMARK_GUEST_ARGS,
    DEFAULT_FIXTURE_SHA256,
    EXPECTED_ITERATIONS,
    PROFILE_COUNTS,
    counterbalanced_order,
    parse_coremark_output,
    resolve_ref_sha,
)

VERSION = 1
PREFIX = "WAMR_BENCH_RESULT="
FIXTURES = {
    "coremark": (
        "tests/benchmarks/coremark/coremark_wasi.wasm",
        DEFAULT_FIXTURE_SHA256,
    ),
    "coremark-nofp": (
        "tests/benchmarks/coremark/coremark_wasi_nofp.wasm",
        "24c0cc1bd52b641cf9e8ae74d1be188cba38d74cdb7ac18378de47382aab9541",
    ),
    "compute": (
        "tests/benchmarks/loop-passes/unroll4.wasm",
        "6870b3373e4098117c82b6736d0ca7cbcc7d8d747fe87ca5b7d1ebf0e4d12890",
    ),
    "memory": (
        "tests/benchmarks/loop-passes/iv_store.wasm",
        "b1979dd330c14d5f898b8a7c8c313e58f6521ad6eb7db9a6e7d7e78788ac6e72",
    ),
}
OPTIONS = {"optimize", "bounds_checks", "stack_checks", "simd", "threads", "memory64"}
PLATFORM = {"arch", "cpu_model", "active_cpu_count", "azure_sku", "azure_region"}
SOURCE = {"commit", "tree_sha256", "tracked_diff_sha256"}
PHASES = {"compile_ticks", "load_ticks", "instantiate_ticks",
          "first_invocation_ticks", "steady_state_ticks"}
OUTCOMES = {"success", "trap", "timeout", "abort", "error"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected),
            f"{label}: expected exactly {sorted(expected)}")


def header(value, kind):
    require(type(value["schema_version"]) is int and value["schema_version"] == VERSION
            and value["kind"] == kind, f"{kind}: version/kind mismatch")


def identical(left, right):
    return cache_key(left) == cache_key(right)


def digest(value, label, length=64):
    require(isinstance(value, str) and
            re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None,
            f"{label}: invalid digest")
    return value


def public_token(value, label):
    require(isinstance(value, str) and
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}", value) is not None,
            f"{label}: expected a public token, not a path or cloud resource ID")
    return value


def number(value, label, minimum=0, integer=False):
    require(type(value) in ((int,) if integer else (int, float))
            and minimum <= value <= 2**64 - 1 and math.isfinite(value),
            f"{label}: invalid number")


def timestamp(value):
    require(isinstance(value, str), "timestamp must be an ISO UTC string")
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0),
            "timestamp must be UTC")
    return parsed


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def private_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(data)


def artifact(path):
    path = Path(path)
    require(path.is_file(), f"artifact missing: {path}")
    require(path.stat().st_size > 0, f"artifact empty: {path}")
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def validate_artifact(value, label):
    keys(value, {"sha256", "bytes"}, label)
    digest(value["sha256"], label)
    number(value["bytes"], label, 1, integer=True)


def validate_source(source):
    keys(source, SOURCE, "source")
    for field in SOURCE:
        digest(source[field], field, 40 if field == "commit" else 64)


def validate_options(options):
    keys(options, OPTIONS, "options")
    require(options["optimize"] in ("ReleaseFast", "ReleaseSafe", "ReleaseSmall", "Debug"),
            "invalid optimize option")
    for name in OPTIONS - {"optimize"}:
        require(type(options[name]) is bool, f"{name} must be boolean")


def validate_platform(platform):
    keys(platform, PLATFORM, "platform")
    number(platform["active_cpu_count"], "active_cpu_count", 1, integer=True)
    for name in PLATFORM - {"active_cpu_count"}:
        public_token(platform[name], name)
        require(platform[name].lower() not in ("unknown", "placeholder", "unspecified"),
                f"{name}: unavailable platform identity")
    require(platform["arch"] in ("x86_64", "aarch64"), "unsupported architecture")


def validate_image_receipt(receipt, target, evidence_kind):
    keys(receipt, {"schema_version", "kind", "evidence_kind", "os", "image",
                   "runtime", "source", "target_abi", "platform",
                   "options", "compiler", "compile_profile", "configured_vm_ram_bytes",
                   "compiler_embedded", "aot_modules"},
         "image receipt")
    header(receipt, "wamr-native-image-receipt")
    require(receipt["evidence_kind"] == evidence_kind, "image receipt evidence kind")
    for key in ("os", "image", "runtime", "source", "target_abi", "platform",
                "options", "compiler", "compile_profile"):
        require(identical(receipt[key], target[key]), f"image receipt {key} mismatch")
    require(receipt["compiler_embedded"] is False, "AOT comparator must be compiler-free")
    number(receipt["configured_vm_ram_bytes"], "configured VM RAM", 1, integer=True)
    require(identical(receipt["aot_modules"], target["aot_modules"]), "image receipt AOT mismatch")


def validate_manifest(manifest, *, allow_synthetic=False):
    keys(manifest, {"schema_version", "kind", "evidence_kind", "campaign_id",
                    "created_at", "expires_at", "profile", "warmups", "runs",
                    "steady_invocations", "targets", "workloads", "schedule",
                    "abi_compatibility", "producer"}, "manifest")
    header(manifest, "wamr-native-benchmark-plan")
    require(manifest["evidence_kind"] == "measurement" or
            (allow_synthetic and manifest["evidence_kind"] == "synthetic"),
            "synthetic data is not measurement evidence")
    require(str(uuid.UUID(manifest["campaign_id"])) == manifest["campaign_id"],
            "campaign_id must be a canonical public random UUID")
    require(timestamp(manifest["created_at"]) < timestamp(manifest["expires_at"]),
            "invalid campaign validity interval")
    require(timestamp(manifest["expires_at"]) - timestamp(manifest["created_at"])
            <= timedelta(days=7), "campaign validity exceeds seven days")
    require(manifest["profile"] in PROFILE_COUNTS, "invalid profile")
    number(manifest["warmups"], "warmups", integer=True)
    number(manifest["runs"], "runs", 1, integer=True)
    number(manifest["steady_invocations"], "steady_invocations", 1, integer=True)
    keys(manifest["producer"], {"source", "script_sha256", "helper_sha256", "schema_sha256"},
         "producer")
    validate_source(manifest["producer"]["source"])
    for name in ("script_sha256", "helper_sha256", "schema_sha256"):
        digest(manifest["producer"][name], name)
    workloads = manifest["workloads"]
    require(isinstance(workloads, dict) and bool(workloads), "no workloads")
    require(2 * len(workloads) * (manifest["warmups"] + manifest["runs"]) <= 10000
            and manifest["steady_invocations"] <= 10000, "campaign sample count is excessive")
    for name, workload in workloads.items():
        require(name in FIXTURES, f"unsupported workload: {name}")
        keys(workload, {"path", "sha256", "export", "args"}, "workload")
        path, expected = FIXTURES[name]
        require(workload == {"path": path, "sha256": expected, "export": "_start",
                            "args": list(COREMARK_GUEST_ARGS) if name.startswith("coremark")
                            else []}, f"unpinned workload {name}")
    keys(manifest["targets"], {"linux", "unikraft"}, "targets")
    for os_name, target in manifest["targets"].items():
        keys(target, {"os", "runtime", "compiler", "source", "target_abi", "platform",
                      "options", "image", "image_receipt", "image_receipt_sha256",
                      "aot_modules", "mode", "jit_preset", "compile_profile"}, "target")
        require(target["os"] == os_name, "target OS mismatch")
        require(target["mode"] == "aot" and target["jit_preset"] is None,
                "only compiler-free AOT is qualified; JIT fast/full is reserved")
        validate_source(target["source"])
        validate_platform(target["platform"])
        if manifest["evidence_kind"] == "measurement":
            require(all("synthetic" not in str(value).lower()
                        for value in target["platform"].values()),
                    "synthetic platform is not measurement evidence")
        validate_options(target["options"])
        public_token(target["target_abi"], "target_abi")
        require(target["compile_profile"] in (None, "unikraft-x86_64"),
                "unsupported explicit compiler profile")
        require(os_name != "unikraft" or target["compile_profile"] == "unikraft-x86_64",
                "native Unikraft requires its explicit compiler profile")
        require(target["compile_profile"] != "unikraft-x86_64" or
                target["platform"]["arch"] == "x86_64", "compiler profile architecture mismatch")
        for name in ("image", "runtime"):
            validate_artifact(target[name], name)
        keys(target["compiler"], {"binary", "source", "version"}, "compiler")
        validate_artifact(target["compiler"]["binary"], "compiler")
        validate_source(target["compiler"]["source"])
        public_token(target["compiler"]["version"], "compiler version")
        require(target["compiler"]["source"] == target["source"],
                "compiler/runtime sources must match")
        keys(target["aot_modules"], workloads, "aot_modules")
        for module in target["aot_modules"].values():
            validate_artifact(module, "AOT module")
        digest(target["image_receipt_sha256"], "image receipt")
        validate_image_receipt(target["image_receipt"], target, manifest["evidence_kind"])
    linux, unikraft = (manifest["targets"][name] for name in ("linux", "unikraft"))
    for name in ("source", "compiler", "options", "platform"):
        require(linux[name] == unikraft[name], f"unmatched {name}")
    compatibility = manifest["abi_compatibility"]
    keys(compatibility, workloads, "abi compatibility")
    for workload, declaration in compatibility.items():
        same = linux["aot_modules"][workload] == unikraft["aot_modules"][workload]
        keys(declaration, {"strategy", "proof_sha256"}, "ABI declaration")
        require(declaration["strategy"] == ("identical" if same else "target-specific"),
                "ABI strategy contradicts actual artifact hashes")
        if same:
            digest(declaration["proof_sha256"], "identical AOT requires ABI proof receipt")
            require(linux["compile_profile"] == unikraft["compile_profile"],
                    "identical AOT cannot claim different compiler profiles")
        else:
            require(declaration["proof_sha256"] is None and
                    linux["target_abi"] != unikraft["target_abi"],
                    "different AOT bytes require explicit target ABI differences")
    expected = make_schedule(manifest)
    require(identical(manifest["schedule"], expected), "incomplete or altered run schedule")


def make_schedule(manifest):
    number(manifest["warmups"], "warmups", integer=True)
    number(manifest["runs"], "runs", 1, integer=True)
    require(2 * len(manifest["workloads"]) * (manifest["warmups"] + manifest["runs"]) <= 10000,
            "campaign sample count is excessive")
    schedule = []
    for phase, count in (("warmup", manifest["warmups"]), ("measured", manifest["runs"])):
        for workload in sorted(manifest["workloads"]):
            for target in counterbalanced_order(["linux", "unikraft"], count):
                schedule.append({"run_id": f"run-{len(schedule) + 1:04d}",
                                 "target": target, "workload": workload, "phase": phase,
                                 "attempt": 1, "position": len(schedule) + 1})
    return schedule


def run_configuration(manifest, run_id):
    found = [run for run in manifest["schedule"] if run["run_id"] == run_id]
    require(len(found) == 1, "unexpected run_id")
    run = found[0]
    return {"campaign_id": manifest["campaign_id"], "run": run,
            "target": manifest["targets"][run["target"]],
            "workload": manifest["workloads"][run["workload"]],
            "steady_invocations": manifest["steady_invocations"],
            "phase_contract": "wamr-embedding-v1"}


def create_plan(config, repo, *, now=None, allow_synthetic=False):
    keys(config, {"profile", "warmups", "runs", "steady_invocations", "valid_hours",
                  "targets", "workloads", "abi_proofs", "producer_source"}, "config")
    require(config["profile"] in PROFILE_COUNTS, "invalid profile")
    number(config["valid_hours"], "valid_hours", 0.01)
    require(config["valid_hours"] <= 168, "campaign validity exceeds seven days")
    script_directory = Path(__file__).resolve().parent
    validate_source(config["producer_source"])
    if not allow_synthetic:
        require(config["producer_source"]["commit"] ==
                resolve_ref_sha(script_directory.parent, "HEAD"),
                "producer source commit differs from the executing tooling checkout")
    now = now or datetime.now(timezone.utc)
    manifest = {"schema_version": VERSION, "kind": "wamr-native-benchmark-plan",
                "evidence_kind": "synthetic" if allow_synthetic else "measurement",
                "campaign_id": str(uuid.uuid4()),
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(hours=config["valid_hours"])).isoformat(),
                "profile": config["profile"], "warmups": config["warmups"],
                "runs": config["runs"], "steady_invocations": config["steady_invocations"],
                "targets": {}, "workloads": {}, "abi_compatibility": {},
                "producer": {"source": config["producer_source"],
                             "script_sha256": sha256_file(script_directory / "bench_coremark.py"),
                             "helper_sha256": sha256_file(Path(__file__)),
                             "schema_sha256": sha256_file(script_directory / "benchmark_schema.py")}}
    require(isinstance(config["workloads"], list) and
            len(set(config["workloads"])) == len(config["workloads"]), "duplicate workloads")
    for name in sorted(config["workloads"]):
        require(name in FIXTURES, f"unknown fixture {name}")
        path, expected = FIXTURES[name]
        require(sha256_file(repo / path) == expected, f"fixture drift: {path}")
        manifest["workloads"][name] = {"path": path, "sha256": expected, "export": "_start",
                                      "args": list(COREMARK_GUEST_ARGS)
                                      if name.startswith("coremark") else []}
    keys(config["targets"], {"linux", "unikraft"}, "config targets")
    for os_name, spec in config["targets"].items():
        keys(spec, {"runtime_path", "compiler_path", "compiler_version", "source",
                    "target_abi", "platform", "options", "image_path",
                    "image_receipt_path", "aot_paths", "compile_profile"}, "target config")
        keys(spec["aot_paths"], manifest["workloads"], "aot_paths")
        receipt_path = Path(spec["image_receipt_path"])
        manifest["targets"][os_name] = {
            "os": os_name, "mode": "aot", "jit_preset": None,
            "runtime": artifact(spec["runtime_path"]), "source": spec["source"],
            "compiler": {"binary": artifact(spec["compiler_path"]),
                         "source": spec["source"], "version": spec["compiler_version"]},
            "compile_profile": spec["compile_profile"],
            "target_abi": spec["target_abi"], "platform": spec["platform"],
            "options": spec["options"], "image": artifact(spec["image_path"]),
            "image_receipt": read_json(receipt_path),
            "image_receipt_sha256": sha256_file(receipt_path),
            "aot_modules": {name: artifact(path) for name, path in spec["aot_paths"].items()},
        }
    keys(config["abi_proofs"], manifest["workloads"], "abi_proofs")
    for name, proof_path in config["abi_proofs"].items():
        linux, unikraft = (manifest["targets"][os_name] for os_name in ("linux", "unikraft"))
        same = linux["aot_modules"][name] == unikraft["aot_modules"][name]
        if same:
            proof = read_json(proof_path)
            require(identical(proof, {
                "schema_version": VERSION, "kind": "wamr-aot-abi-proof",
                "evidence_kind": manifest["evidence_kind"], "compatible": True,
                "source": linux["source"],
                "target_abis": [linux["target_abi"], unikraft["target_abi"]],
                "aot_sha256": linux["aot_modules"][name]["sha256"],
            }), "ABI proof does not attest these exact artifacts/targets")
        else:
            require(proof_path is None, "target-specific artifacts must use null ABI proof")
        manifest["abi_compatibility"][name] = {
            "strategy": "identical" if same else "target-specific",
            "proof_sha256": sha256_file(Path(proof_path)) if same else None,
        }
    manifest["schedule"] = make_schedule(manifest)
    validate_manifest(manifest, allow_synthetic=allow_synthetic)
    return manifest


def parse_result_stream(raw):
    results = []
    for line in raw.decode("utf-8", errors="strict").splitlines():
        if PREFIX in line:
            require(line.startswith(PREFIX), "result prefix must start its own line")
            results.append(strict_json(line[len(PREFIX):]))
    require(len(results) == 1, f"expected exactly one terminal result, got {len(results)}")
    return results[0]


def validate_memory(memory, target, *, complete=True):
    keys(memory, {"image_sha256", "configured_vm_ram_bytes", "coverage",
                  "method", "covered_regions", "omitted_regions", "samples"}, "memory")
    require(memory["image_sha256"] == target["image"]["sha256"], "memory image mismatch")
    number(memory["configured_vm_ram_bytes"], "memory configured VM RAM", 1, integer=True)
    require(memory["configured_vm_ram_bytes"] ==
            target["image_receipt"]["configured_vm_ram_bytes"], "memory VM RAM mismatch")
    require(memory["coverage"] in ("complete-guest", "partial-guest"), "memory coverage")
    public_token(memory["method"], "memory method")
    for name in ("covered_regions", "omitted_regions"):
        require(isinstance(memory[name], list), f"memory {name}")
        for region in memory[name]:
            public_token(region, "memory region")
        require(len(set(memory[name])) == len(memory[name]), "duplicate memory region")
    require(bool(memory["covered_regions"]), "memory requires measured coverage")
    require(not set(memory["covered_regions"]) & set(memory["omitted_regions"]),
            "memory coverage overlaps omissions")
    require((memory["coverage"] == "complete-guest") ==
            (not memory["omitted_regions"]), "memory omissions/coverage mismatch")
    require(isinstance(memory["samples"], list) and
            (len(memory["samples"]) == 3 if complete else 1 <= len(memory["samples"]) <= 3),
            "memory requires three phase snapshots on success, a nonempty prefix on failure")
    for sample, stage in zip(memory["samples"],
                             ("after_instantiation", "after_first", "after_steady")):
        keys(sample, {"stage", "reserved_address_bytes", "committed_bytes"}, "memory sample")
        require(sample["stage"] == stage, "memory stage mismatch")
        for field in ("reserved_address_bytes", "committed_bytes"):
            number(sample[field], field, 1, integer=True)
        require(sample["committed_bytes"] <= sample["reserved_address_bytes"],
                "committed memory exceeds measured reservation")


def field_once(stdout, label, pattern):
    lines = [line for line in stdout.splitlines() if label in line]
    require(len(lines) == 1, f"expected exactly one {label} marker")
    match = re.fullmatch(pattern, lines[0].strip())
    require(match is not None, f"malformed {label} marker")
    return match.group(1)


def coremark_correctness(stdout, invocation_seconds):
    parsed = parse_coremark_output(stdout, "native", EXPECTED_ITERATIONS)
    crc = {}
    for name, expected in (("seedcrc", "e9f5"), ("crclist", "e714"),
                           ("crcmatrix", "1fd7"), ("crcstate", "8e3a"),
                           ("crcfinal", None)):
        marker = name if name == "seedcrc" else f"[0]{name}"
        value = field_once(stdout, name, re.escape(marker) + r"\s*:\s*0x([0-9a-fA-F]{4})")
        crc[name] = value.lower()
        require(expected is None or value.lower() == expected, f"unexpected {name} CRC")
    seconds = float(field_once(stdout, "Total time (secs)",
                               r"Total time \(secs\)\s*:\s*(\d+(?:\.\d+)?)"))
    ticks = int(field_once(stdout, "Total ticks", r"Total ticks\s*:\s*(\d+)"))
    require(seconds > 0 and ticks > 0, "CoreMark reported non-positive time")
    require(math.isfinite(parsed.throughput) and parsed.throughput > 0,
            "CoreMark reported invalid throughput")
    expected_rate = EXPECTED_ITERATIONS / seconds
    require(abs(parsed.throughput - expected_rate) <= max(1, expected_rate * 0.001),
            "CoreMark throughput/time/iterations disagree")
    require(seconds <= invocation_seconds + 0.01, "CoreMark time exceeds invocation duration")
    return {"crc": crc, "iterations": parsed.iterations,
            "iterations_per_second": parsed.throughput, "reported_seconds": seconds,
            "reported_ticks": ticks,
            "minimum_timing_met": seconds >= 10 and invocation_seconds >= 10,
            "compliance": "not-certified"}


def validate_result(result, manifest, run_id):
    config = run_configuration(manifest, run_id)
    target = config["target"]
    keys(result, {"schema_version", "kind", "evidence_kind", "campaign_id", "run_id",
                  "config_sha256", "image_receipt_sha256", "observed", "outcome",
                  "exit_code", "clock", "phase_contract", "phases", "invocations",
                  "memory"}, "result")
    header(result, "wamr-native-benchmark-result")
    require(result["evidence_kind"] == manifest["evidence_kind"], "result evidence kind")
    require(result["campaign_id"] == manifest["campaign_id"] and result["run_id"] == run_id,
            "stale or unexpected campaign/run")
    require(result["config_sha256"] == cache_key(config), "run configuration mismatch")
    require(result["image_receipt_sha256"] == target["image_receipt_sha256"],
            "image receipt mismatch")
    require(identical(result["observed"], {
        "image_sha256": target["image"]["sha256"],
        "runtime_sha256": target["runtime"]["sha256"],
        "aot_sha256": target["aot_modules"][config["run"]["workload"]]["sha256"],
        "wasm_sha256": config["workload"]["sha256"],
        "platform": target["platform"], "options": target["options"],
        "mode": "aot", "jit_preset": None, "compile_profile": target["compile_profile"],
    }), "observed artifact/platform/mode mismatch")
    require(result["outcome"] in OUTCOMES, "unknown terminal outcome")
    require(result["exit_code"] is None or
            (type(result["exit_code"]) is int and 0 <= result["exit_code"] <= 0xffffffff),
            "guest exit_code must be a complete u32 status, not a host process returncode")
    require(result["outcome"] != "trap" or result["exit_code"] is None,
            "trap must not report a guest exit code")
    require(result["phase_contract"] == "wamr-embedding-v1", "unknown phase contract")
    phases = result["phases"]
    keys(phases, PHASES, "phases")
    require(phases["compile_ticks"] is None, "AOT must not conflate runtime compilation")
    clock = result["clock"]
    if clock is None:
        require(result["outcome"] != "success" and
                all(phases[name] is None for name in
                    ("load_ticks", "instantiate_ticks", "first_invocation_ticks")) and
                phases["steady_state_ticks"] == [] and result["invocations"] == [] and
                result["memory"] is None,
                "unavailable clock requires a failed, unstarted attempt without measurements")
        return []
    keys(clock, {"source", "unit", "ticks_per_second", "resolution_ticks"}, "clock")
    public_token(clock["source"], "clock source")
    require(clock["unit"] in ("ns", "us", "ms"), "unsupported clock unit")
    require(type(clock["ticks_per_second"]) is int and clock["ticks_per_second"] ==
            {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1000}[clock["unit"]],
            "clock frequency/unit mismatch")
    number(clock["resolution_ticks"], "clock resolution", 1, integer=True)
    require(clock["resolution_ticks"] <= clock["ticks_per_second"],
            "clock resolution is too coarse for this benchmark")
    success = result["outcome"] == "success"
    for phase in ("load_ticks", "instantiate_ticks", "first_invocation_ticks"):
        require(not success or phases[phase] is not None, f"missing {phase}")
        if phases[phase] is not None:
            number(phases[phase], phase, integer=True)
    require(isinstance(phases["steady_state_ticks"], list), "steady state ticks")
    require(len(phases["steady_state_ticks"]) <= config["steady_invocations"],
            "extra steady-state invocations")
    if success:
        require(len(phases["steady_state_ticks"]) == config["steady_invocations"],
                "partial steady state result")
        require(type(result["exit_code"]) is int and result["exit_code"] == 0,
                "success requires exit code zero")
    for value in phases["steady_state_ticks"]:
        number(value, "steady ticks", integer=True)
    require(phases["instantiate_ticks"] is None or phases["load_ticks"] is not None,
            "instantiation without load")
    require(phases["first_invocation_ticks"] is None or phases["instantiate_ticks"] is not None,
            "invocation without instantiation")
    require(not phases["steady_state_ticks"] or phases["first_invocation_ticks"] is not None,
            "steady state without first invocation")
    durations = ([] if phases["first_invocation_ticks"] is None
                 else [phases["first_invocation_ticks"]]) + phases["steady_state_ticks"]
    timed_ticks = sum(durations) + sum(phases[name] or 0 for name in
                                      ("load_ticks", "instantiate_ticks"))
    campaign_seconds = (timestamp(manifest["expires_at"]) -
                        timestamp(manifest["created_at"])).total_seconds()
    require(timed_ticks <= campaign_seconds * clock["ticks_per_second"],
            "guest phase duration exceeds campaign validity (overflow or stale clock evidence)")
    require(isinstance(result["invocations"], list) and
            len(result["invocations"]) == len(durations), "invocation evidence count mismatch")
    checks = []
    for index, (invocation, ticks) in enumerate(zip(result["invocations"], durations)):
        keys(invocation, {"phase", "outcome", "exit_code", "stdout"}, "invocation")
        require(invocation["phase"] == ("first" if index == 0 else "steady"),
                "invocation phase mismatch")
        require(invocation["outcome"] in ("returned", "proc_exit", "trap", "error"),
                "invocation outcome")
        require(invocation["exit_code"] is None or
                (type(invocation["exit_code"]) is int and 0 <= invocation["exit_code"] <= 0xffffffff),
                "invocation exit_code must be a complete u32 status")
        require(invocation["outcome"] != "proc_exit" or invocation["exit_code"] is not None,
                "proc_exit requires its u32 status")
        require(invocation["outcome"] != "trap" or invocation["exit_code"] is None,
                "trapped invocation must not report an exit code")
        require(isinstance(invocation["stdout"], str), "invocation stdout")
        good = invocation["outcome"] in ("returned", "proc_exit") and invocation["exit_code"] == 0
        require(index == len(durations) - 1 or good,
                "invocations continued after a failed terminal outcome")
        if invocation["outcome"] == "proc_exit" and invocation["exit_code"] != 0:
            require(result["exit_code"] == invocation["exit_code"],
                    "guest terminal status lost or truncated proc_exit")
        require(not success or good, "success contradicts invocation terminal outcome")
        if good and config["run"]["workload"].startswith("coremark"):
            try:
                checks.append(coremark_correctness(
                    invocation["stdout"], ticks / clock["ticks_per_second"]))
            except (ValueError, RuntimeError) as error:
                if success:
                    # The legacy parser includes raw output in its diagnostic.
                    raise ValueError("guest success contradicts CoreMark validation; "
                                     "inspect private stdout evidence") from error
                checks.append({"self_check": "failed", "failure": "coremark-validation"})
        elif good:
            require(invocation["outcome"] == "returned" and invocation["stdout"] == "",
                    "no-import fixture must return without output, exit or trap")
            checks.append({"self_check": "returned-without-trap"})
        else:
            checks.append({"self_check": "failed"})
    if success:
        validate_memory(result["memory"], target)
    elif result["memory"] is not None:
        validate_memory(result["memory"], target, complete=False)
    if result["memory"] is not None:
        reached_snapshots = (int(phases["instantiate_ticks"] is not None) +
                             int(phases["first_invocation_ticks"] is not None) +
                             int(bool(phases["steady_state_ticks"])))
        require(len(result["memory"]["samples"]) <= reached_snapshots,
                "memory snapshot refers to an unstarted phase")
    return checks


def persist_observation(manifest, run_id, output, stdout, stderr, *,
                        started_at, completed_at, observation_seconds,
                        control_plane_seconds=None, transport_outcome="exited",
                        process_returncode=0):
    config = run_configuration(manifest, run_id)
    output = Path(output)
    private_write(output / "stdout.bin", stdout)
    private_write(output / "stderr.bin", stderr)
    record = {"schema_version": VERSION, "kind": "wamr-native-host-observation",
              "evidence_kind": manifest["evidence_kind"], "campaign_id": manifest["campaign_id"],
              "run_id": run_id, "config_sha256": cache_key(config),
              "started_at": started_at, "completed_at": completed_at,
              "observation_seconds": observation_seconds,
              "control_plane_seconds": control_plane_seconds,
              "transport_outcome": transport_outcome, "process_returncode": process_returncode,
              "stdout_sha256": sha256_file(output / "stdout.bin"),
              "stderr_sha256": sha256_file(output / "stderr.bin")}
    private_write(output / "observation.json", (json.dumps(record, sort_keys=True) + "\n").encode())


def capture(manifest, run_id, command, output, timeout, *, allow_synthetic=False):
    validate_manifest(manifest, allow_synthetic=allow_synthetic)
    now = datetime.now(timezone.utc)
    require(timestamp(manifest["created_at"]) <= now < timestamp(manifest["expires_at"]),
            "campaign is not current")
    config = run_configuration(manifest, run_id)
    require(config["run"]["target"] == "linux",
            "native-capture launches only an external Linux embedding producer, not cloud guests")
    require(command, "embedding producer command required")
    number(timeout, "timeout", 0.001)
    require(timeout <= (timestamp(manifest["expires_at"]) - now).total_seconds(),
            "producer timeout extends beyond campaign validity")
    output = Path(output)
    require(not output.exists(), "capture output already exists")
    output.mkdir(parents=True, mode=0o700)
    request = {"config": config, "config_sha256": cache_key(config)}
    private_write(output / "request.json", (json.dumps(request, sort_keys=True) + "\n").encode())
    environment = os.environ.copy()
    environment["WAMR_BENCH_REQUEST"] = str((output / "request.json").resolve())
    environment["WAMR_BENCH_CONFIG_SHA256"] = cache_key(config)
    start = time.monotonic()
    status, returncode = "exited", None
    try:
        process = subprocess.run(command, capture_output=True, timeout=timeout, env=environment)
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        status, stdout, stderr = "timeout", error.stdout or b"", error.stderr or b""
    except OSError as error:
        status, stdout, stderr = "launch_error", b"", str(error).encode()
    elapsed = time.monotonic() - start
    persist_observation(manifest, run_id, output, stdout, stderr,
                        started_at=now.isoformat(),
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        observation_seconds=elapsed, transport_outcome=status,
                        process_returncode=returncode)
    # Evidence is persisted before validation, including failed or partial attempts.
    if status != "exited":
        return False
    result = parse_result_stream(stdout)
    validate_result(result, manifest, run_id)
    require(returncode == 0 or result["outcome"] != "success",
            "producer process failed after reporting guest success")
    return result["outcome"] == "success"


def consume_capture(manifest, directory, *, allow_synthetic=False):
    directory = Path(directory)
    observation = read_json(directory / "observation.json")
    keys(observation, {"schema_version", "kind", "evidence_kind", "campaign_id", "run_id",
                       "config_sha256", "started_at", "completed_at", "observation_seconds",
                       "control_plane_seconds", "transport_outcome", "process_returncode",
                       "stdout_sha256", "stderr_sha256"}, "observation")
    header(observation, "wamr-native-host-observation")
    require(observation["evidence_kind"] == manifest["evidence_kind"] and
            (observation["evidence_kind"] == "measurement" or allow_synthetic),
            "synthetic observation rejected")
    config = run_configuration(manifest, observation["run_id"])
    require(observation["campaign_id"] == manifest["campaign_id"] and
            observation["config_sha256"] == cache_key(config), "stale observation/config")
    start, end = timestamp(observation["started_at"]), timestamp(observation["completed_at"])
    require(timestamp(manifest["created_at"]) <= start <= end <= timestamp(manifest["expires_at"]),
            "observation outside campaign interval")
    for field in ("observation_seconds", "control_plane_seconds"):
        if field == "observation_seconds" or observation[field] is not None:
            number(observation[field], field)
    require(observation["transport_outcome"] in ("exited", "timeout", "launch_error"),
            "unknown transport outcome")
    require(observation["process_returncode"] is None or
            type(observation["process_returncode"]) is int, "process returncode")
    require((observation["transport_outcome"] == "exited") ==
            (observation["process_returncode"] is not None), "transport returncode mismatch")
    for stream in ("stdout", "stderr"):
        require(sha256_file(directory / f"{stream}.bin") == observation[f"{stream}_sha256"],
                f"raw {stream} evidence mismatch")
    raw = (directory / "stdout.bin").read_bytes()
    if observation["transport_outcome"] != "exited":
        require(PREFIX.encode() not in raw, "terminal result contradicts transport failure")
        result, checks = None, []
    else:
        result = parse_result_stream(raw)
        checks = validate_result(result, manifest, observation["run_id"])
        require(observation["process_returncode"] == 0 or result["outcome"] != "success",
                "process failure contradicts guest success")
    return {"run": config["run"], "observation": observation, "result": result,
            "correctness": checks}


def build_report(manifest, directories, *, allow_synthetic=False):
    validate_manifest(manifest, allow_synthetic=allow_synthetic)
    records = [consume_capture(manifest, directory, allow_synthetic=allow_synthetic)
               for directory in directories]
    ids = [record["run"]["run_id"] for record in records]
    require(len(set(ids)) == len(ids), "unexpected duplicate run result")
    require(set(ids) == {run["run_id"] for run in manifest["schedule"]},
            "partial campaign: every planned attempt (including failures/warmups) is required")
    records.sort(key=lambda record: record["run"]["position"])
    for workload in manifest["workloads"]:
        final_crcs = {check["crc"]["crcfinal"] for record in records
                      if record["run"]["workload"] == workload
                      for check in record["correctness"] if "crc" in check}
        require(len(final_crcs) <= 1, "CoreMark final CRC differs across matched invocations")
    summary = {}
    all_success = all(record["result"] is not None and
                      record["result"]["outcome"] == "success" for record in records)
    for target in ("linux", "unikraft"):
        summary[target] = {}
        for workload in manifest["workloads"]:
            selected = [record for record in records if record["run"]["target"] == target
                        and record["run"]["workload"] == workload
                        and record["run"]["phase"] == "measured"
                        and record["result"] is not None
                        and record["result"]["outcome"] == "success"]
            row = {"successful_runs": len(selected), "planned_runs": manifest["runs"]}
            for phase in ("load_ticks", "instantiate_ticks", "first_invocation_ticks"):
                values = [record["result"]["phases"][phase] /
                          record["result"]["clock"]["ticks_per_second"] for record in selected]
                row[phase.replace("_ticks", "_seconds")] = (
                    sample_stats(values, "values") if values else None)
            values = [tick / record["result"]["clock"]["ticks_per_second"]
                      for record in selected
                      for tick in record["result"]["phases"]["steady_state_ticks"]]
            row["steady_state_seconds"] = sample_stats(values, "values") if values else None
            summary[target][workload] = row
    timing_met = all(check.get("minimum_timing_met", True)
                     for record in records for check in record["correctness"])
    has_coremark = any(name.startswith("coremark") for name in manifest["workloads"])
    public_records = []
    for record in records:
        result = record["result"]
        # Raw guest output/diagnostics and local paths remain in mode-0600 captures.
        if result is not None:
            result = {key: value for key, value in result.items() if key != "invocations"}
            result["invocations"] = [
                {key: value for key, value in invocation.items() if key != "stdout"}
                for invocation in record["result"]["invocations"]]
        public_records.append({**record, "result": result})
    return {"schema_version": VERSION, "kind": "wamr-native-matched-comparison",
            "evidence_kind": manifest["evidence_kind"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": cache_key(manifest), "plan": manifest,
            "records": public_records, "summary": summary,
            "status": {"all_attempts_successful": all_success,
                       "profile_counts_match": (manifest["warmups"], manifest["runs"]) ==
                       PROFILE_COUNTS[manifest["profile"]],
                       "coremark_minimum_timing_met": (timing_met and all_success)
                       if has_coremark else None,
                       "coremark_compliance": "not-certified" if has_coremark else "not-applicable",
                       "paired_measurement_complete": all_success and
                       manifest["evidence_kind"] == "measurement",
                       "os_attribution": "requires-experimental-review",
                       "performance_threshold": None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("native-plan", help="verify local artifacts and create a matched plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    plan.add_argument("--out", type=Path, required=True)
    request = commands.add_parser("native-request", help="export one exact guest producer request")
    request.add_argument("--manifest", type=Path, required=True)
    request.add_argument("--run-id", required=True)
    request.add_argument("--out", type=Path, required=True)
    capture_parser = commands.add_parser("native-capture", help="capture a Linux embedding producer")
    capture_parser.add_argument("--manifest", type=Path, required=True)
    capture_parser.add_argument("--run-id", required=True)
    capture_parser.add_argument("--out", type=Path, required=True)
    capture_parser.add_argument("--timeout", type=float, required=True)
    capture_parser.add_argument("producer", nargs=argparse.REMAINDER)
    ingest = commands.add_parser("native-import", help="import externally collected native serial evidence")
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--run-id", required=True)
    ingest.add_argument("--stdout", type=Path, required=True)
    ingest.add_argument("--stderr", type=Path, required=True)
    ingest.add_argument("--started-at", required=True)
    ingest.add_argument("--completed-at", required=True)
    ingest.add_argument("--observation-seconds", type=float, required=True)
    ingest.add_argument("--control-plane-seconds", type=float)
    ingest.add_argument("--transport-outcome", choices=("exited", "timeout", "launch_error"),
                        default="exited")
    ingest.add_argument("--process-returncode", type=int)
    ingest.add_argument("--out", type=Path, required=True)
    report = commands.add_parser("native-report", help="strictly consume paired native evidence")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--capture", type=Path, action="append", required=True)
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "native-plan":
            document = create_plan(read_json(args.config), args.repo.resolve())
        else:
            manifest = read_json(args.manifest)
            validate_manifest(manifest)
            if args.command == "native-request":
                document = run_configuration(manifest, args.run_id)
                document = {"config": document, "config_sha256": cache_key(document)}
            elif args.command == "native-capture":
                producer = args.producer[1:] if args.producer[:1] == ["--"] else args.producer
                return 0 if capture(manifest, args.run_id, producer, args.out, args.timeout) else 1
            elif args.command == "native-import":
                require(not args.out.exists(), "capture output already exists")
                persist_observation(
                    manifest, args.run_id, args.out, args.stdout.read_bytes(), args.stderr.read_bytes(),
                    started_at=args.started_at, completed_at=args.completed_at,
                    observation_seconds=args.observation_seconds,
                    control_plane_seconds=args.control_plane_seconds,
                    transport_outcome=args.transport_outcome,
                    process_returncode=args.process_returncode)
                record = consume_capture(manifest, args.out)
                return 0 if record["result"] and record["result"]["outcome"] == "success" else 1
            else:
                document = build_report(manifest, args.capture)
        require(not args.out.exists(), "output already exists; refusing to replace evidence")
        atomic_write_json(args.out, document)
        if args.command == "native-report":
            return 0 if document["status"]["all_attempts_successful"] else 1
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(f"native benchmark: {error}", file=sys.stderr)
        return 2
