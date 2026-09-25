#!/usr/bin/env python3
"""Audit retained #986 sample-level provenance without reclassifying unknowns.

The profile's sampled instruction rows and per-engine totals are the input,
not a new perf capture. This does not independently rerun native disassembly
or prove semantics absent from the original classifier.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import aarch64_instruction_provenance as provenance
import profile_coremark_aarch64 as profile


def audit_rows(analysis, *, engine):
    provenance.validate_analysis_samples(analysis, name=engine)
    universe = analysis["common_gating_universe"]
    counts = {category: 0 for category in provenance.CATEGORIES}
    seen = set()
    for row in universe["sampled_instructions"]:
        offset = row["offset"]
        samples = row["samples"]
        category = row["category"]
        if (
            type(offset) is not int or offset < 0 or offset in seen
            or type(samples) is not int or samples <= 0
            or category not in counts
            or not isinstance(row.get("reason"), str)
            or not isinstance(row.get("path_evidence"), list)
        ):
            raise ValueError(f"{engine}: malformed or duplicate sampled instruction")
        seen.add(offset)
        counts[category] += samples
    for category in counts:
        if counts[category] != universe["categories"][category]["samples"]:
            raise ValueError(f"{engine}: {category} sampled rows do not reconcile")
    return sorted(
        (row for row in universe["sampled_instructions"]
         if row["category"] in {"unknown", "mixed"}),
        key=lambda row: (-row["samples"], row["offset"]),
    )


def reanalyze(report, cwasm_bytes, benchmark_bytes):
    profile.validate_report(report)
    artifact = report["retained_analysis_artifacts"]["wamr_cwasm"]
    if artifact["path"] != "wamr-profiled.cwasm.gz" or len(cwasm_bytes) != artifact["source_size_bytes"]:
        raise ValueError("retained WAMR binary identity or size differs from artifact manifest")
    if hashlib.sha256(benchmark_bytes).hexdigest() != report["benchmark"]["report_sha256"]:
        raise ValueError("benchmark report hash differs from retained profile")
    benchmark = json.loads(benchmark_bytes)
    if benchmark.get("provenance", {}).get("report_id") != report["benchmark"]["report_id"]:
        raise ValueError("benchmark report identity differs from retained profile")
    if hashlib.sha256(cwasm_bytes).hexdigest() != report["wamr"]["cwasm_sha256"]:
        raise ValueError("retained WAMR binary hash differs from profile")
    if hashlib.sha256(cwasm_bytes).hexdigest() != artifact["source_sha256"]:
        raise ValueError("retained WAMR binary hash differs from artifact manifest")
    targets = [item for item in report["matched_functions"] if item["local_func"] == 3]
    if len(targets) != 1 or targets[0]["wasm_function_index"] != 15:
        raise ValueError("missing or ambiguous core_bench_list function identity")
    target = targets[0]
    analyses = target["alu_provenance"]
    ranked = {engine: audit_rows(analyses[engine], engine=engine)
              for engine in ("wamr", "wasmtime")}
    gate = provenance.compare_engine_analyses(
        analyses["wamr"], analyses["wasmtime"], threshold_pct=5.0
    )
    if gate != analyses["gate"] or gate != report["alu_provenance"]["gate"]:
        raise ValueError("retained conservative gate differs from recomputed gate")
    return {
        "source_commit": report["provenance"]["analysis_sources"]["commit"],
        "cwasm_sha256": report["wamr"]["cwasm_sha256"],
        "function": "core_bench_list",
        "local_func": 3,
        "wasm_index": 15,
        "total_run_samples": {
            engine: analyses[engine]["total_run_samples"]
            for engine in ("wamr", "wasmtime")
        },
        "categories": {
            engine: analyses[engine]["common_gating_universe"]["categories"]
            for engine in ("wamr", "wasmtime")
        },
        "top_uncertain_paths": {
            engine: ranked[engine][:20] for engine in ("wamr", "wasmtime")
        },
        "gate": gate,
        "scope": (
            "Audits retained sampled instruction partitions and recomputes "
            "the conservative gate; does not independently rerun the native "
            "classifier or turn unknown control/escape paths into addresses."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    root = args.artifact_dir
    report = json.loads((root / "profile.json").read_text())
    result = reanalyze(
        report,
        gzip.decompress((root / "wamr-profiled.cwasm.gz").read_bytes()),
        (root / "benchmark-report.json").read_bytes(),
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.json_out:
        args.json_out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
