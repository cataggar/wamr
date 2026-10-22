#!/usr/bin/env python3
"""Audit retained #986 sample-level provenance and terminal-sink exhaustion.

Validates the retained profile's sampled instruction partitions, groups the
uncertain samples into deterministic unknown-reason clusters, and when the
retained Wasmtime disassembly (`wasmtime-objdump.txt`) is provided, reruns
the native AArch64 classifier and join/cycle-unblocked terminal-sink
exhaustion analysis to prove whether conservative headroom can clear +5.00 pp.
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


def _reanalyze_wasmtime_native_exhaustion(analyses, wasmtime_objdump_text):
    wasmtime_saved = analyses["wasmtime"]
    aot = profile.load_aot_helper(Path(__file__).resolve().parents[1])
    insts = profile.compare_hot_function.parse_disassembly(
        wasmtime_objdump_text, wasmtime_wasm_index=15
    )
    address_base = profile._wasmtime_function_address_base(
        wasmtime_objdump_text, 15
    )
    adapter = [
        aot.Instruction(
            address=item.offset,
            offset=item.offset,
            size=item.size,
            text=item.text,
        )
        for item in insts
    ]
    broad_classes_list = aot.classify_instruction_stream(
        adapter, architecture="aarch64"
    )
    inst_by_offset = {int(inst.offset): inst for inst in insts}
    samples_by_offset = {}
    for row in wasmtime_saved["common_gating_universe"]["sampled_instructions"]:
        offset = int(row["offset"])
        if offset not in inst_by_offset:
            raise ValueError(
                f"wasmtime sampled offset {offset} is absent from native disassembly"
            )
        inst = inst_by_offset[offset]
        expected_text = str(inst.text).strip()
        if row["instruction"] != expected_text:
            raise ValueError(
                f"wasmtime sampled instruction at offset {offset} "
                f"({row['instruction']!r}) differs from native disassembly "
                f"({expected_text!r})"
            )
        samples_by_offset[offset] = int(row["samples"])

    total_run_samples = int(wasmtime_saved["total_run_samples"])
    global_attributed = int(
        wasmtime_saved["global_sample_mapping"]["attributed_samples"]
    )
    rerun = provenance.analyze_instruction_stream(
        insts,
        broad_classes=broad_classes_list,
        samples_by_offset=samples_by_offset,
        total_run_samples=total_run_samples,
        address_base=address_base,
        global_attributed_samples=global_attributed,
    )
    for key in (
        "categories",
        "coverage",
        "common_gating_universe",
        "cfg",
        "sampled_instructions",
        "reason_counts",
    ):
        if rerun[key] != wasmtime_saved[key]:
            raise ValueError(
                f"retained Wasmtime {key} differs from native disassembly rerun"
            )

    exhaustion = provenance.analyze_terminal_sink_exhaustion(
        insts,
        broad_classes=broad_classes_list,
        samples_by_offset=samples_by_offset,
        total_run_samples=total_run_samples,
        address_base=address_base,
        global_attributed_samples=global_attributed,
    )
    exhaustion_gate = provenance.compare_terminal_sink_exhaustion(
        analyses["wamr"],
        wasmtime_saved,
        wasmtime_exhaustion=exhaustion,
        threshold_pct=5.0,
    )
    sink_summary = {
        sink: {
            "samples": data["samples"],
            "percent_of_run": data["percent_of_run"],
            "percent_of_common_universe": data["percent_of_common_universe"],
            "static_instructions": data["static_instructions"],
            "top_sampled_instructions": data["sampled_instructions"][:10],
        }
        for sink, data in exhaustion["sink_classes"].items()
    }
    return {
        "verified_retained_provenance": True,
        "address_base": address_base,
        "static_instructions": len(insts),
        "common_static_instructions": exhaustion["common_static_instructions"],
        "sink_classes": sink_summary,
        "reclassified_from_unknown": exhaustion["reclassified_from_unknown"],
        "exhaustion_gate": exhaustion_gate,
    }


def reanalyze(
    report, cwasm_bytes, benchmark_bytes, *, wasmtime_objdump_text=None
):
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
    clusters = {
        engine: provenance.classify_unknown_reason_clusters(
            analyses[engine], name=engine
        )
        for engine in ("wamr", "wasmtime")
    }
    gate = provenance.compare_engine_analyses(
        analyses["wamr"], analyses["wasmtime"], threshold_pct=5.0
    )
    if gate != analyses["gate"] or gate != report["alu_provenance"]["gate"]:
        raise ValueError("retained conservative gate differs from recomputed gate")
    result = {
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
        "unknown_reason_clusters": {
            engine: {
                "uncertain_samples": clusters[engine]["uncertain_samples"],
                "uncertain_percent_of_run": clusters[engine][
                    "uncertain_percent_of_run"
                ],
                "clusters": {
                    name: {
                        "samples": item["samples"],
                        "percent_of_run": item["percent_of_run"],
                        "percent_of_uncertain": item["percent_of_uncertain"],
                        "sampled_instruction_count": item[
                            "sampled_instruction_count"
                        ],
                        "top_sampled_instructions": item[
                            "sampled_instructions"
                        ][:5],
                    }
                    for name, item in clusters[engine]["clusters"].items()
                },
            }
            for engine in ("wamr", "wasmtime")
        },
        "top_uncertain_paths": {
            engine: ranked[engine][:20] for engine in ("wamr", "wasmtime")
        },
        "gate": gate,
        "scope": (
            "Audits retained sampled instruction partitions, groups uncertain "
            "samples by reason cluster, recomputes the conservative gate, and "
            "when native Wasmtime disassembly is supplied, verifies terminal-"
            "sink exhaustion across CFG joins and cycles."
        ),
    }
    if wasmtime_objdump_text is not None:
        result["wasmtime_native_exhaustion"] = (
            _reanalyze_wasmtime_native_exhaustion(
                analyses, wasmtime_objdump_text
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    root = args.artifact_dir
    report = json.loads((root / "profile.json").read_text())
    objdump_path = root / "wasmtime-objdump.txt"
    wasmtime_objdump_text = (
        objdump_path.read_text() if objdump_path.is_file() else None
    )
    result = reanalyze(
        report,
        gzip.decompress((root / "wamr-profiled.cwasm.gz").read_bytes()),
        (root / "benchmark-report.json").read_bytes(),
        wasmtime_objdump_text=wasmtime_objdump_text,
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.json_out:
        args.json_out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
