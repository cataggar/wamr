#!/usr/bin/env python3
"""Capture matched-host native AArch64 CoreMark profiles for WAMR and Wasmtime."""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bench_coremark
import compare_hot_function
import aarch64_instruction_provenance


REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "coremark-aarch64-matched-profile"
HISTORICAL_REPORT_SCHEMA_VERSION = 1
HISTORICAL_BASELINE_RUN = 33631050708
DEFAULT_MIN_SAMPLES = 1000
DEFAULT_TOP_FUNCTIONS = 10
DEFAULT_CLASSIFY_FUNCTIONS = 3
DEFAULT_MAX_PERF_BYTES = 25 * 1024 * 1024
PROFILE_CAPTURES_PER_ENGINE = 2
MIN_ATTRIBUTION_COVERAGE_PCT = 99.0
ALL_ALU_WORDING = (
    "all ALU-class instructions: add/sub, logical operations, mul/div, "
    "shifts, compares, csel, and address-generation instructions"
)
NARROW_ALU_WORDING = (
    "architecture-only CFG and reaching-definition paths preserve the legacy "
    "all_alu partition and separately classify a complete common gating "
    "universe as address_generation, structural_address_guard, "
    "algorithmic_alu, mixed, or unknown without engine-register heuristics"
)
REQUIRED_ANALYSIS_SOURCES = {
    "scripts/profile_coremark_aarch64.py",
    "scripts/aarch64_instruction_provenance.py",
    "scripts/compare_hot_function.py",
    "scripts/bench_coremark.py",
    "scripts/bench_optimize.py",
    ".github/skills/aot-perf-profile/aot_jit_attr.py",
}
WASMTIME_SYMBOL_RE = re.compile(
    r"wasm\[(?P<module>\d+)\]::function\[(?P<function>\d+)\]"
    r"(?:::(?P<name>[^+\s(]+))?"
)
WASMTIME_JITTED_SYMBOL_RE = re.compile(
    r"^\s*(?:0x)?[0-9a-fA-F]+\s+"
    r"(?P<name>[^+\s(]+)"
    r"(?:\+0x(?P<offset>[0-9a-fA-F]+))?\s+"
    r"\((?P<dso>[^)]+/jitted-\d+-(?P<local>\d+)\.so)\)"
)
CLASS_GROUPS = {
    "frame_traffic": {
        "frame_load_unattributed",
        "frame_store_unattributed",
        "unknown_frame_load",
        "unknown_frame_store",
    },
    "reg_moves": {"regmov"},
    "all_alu": {"alu"},
    "bounds_checks": {"bounds_cmp", "bounds_branch"},
    "linear_memory": {"linear_memory", "mem_access"},
    "calls": {"call"},
    "conditional_branches": {"cond_branch"},
    "direct_branches": {"direct_branch", "jmp"},
    "indirect_dispatch": {"indirect_dispatch", "dispatch_jmp"},
}


class ProfileError(RuntimeError):
    pass


def load_aot_helper(repo: Path):
    path = repo / ".github/skills/aot-perf-profile/aot_jit_attr.py"
    spec = importlib.util.spec_from_file_location("coremark_aot_jit_attr", path)
    if spec is None or spec.loader is None:
        raise ProfileError(f"cannot import attribution helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_source_paths(repo: Path, aot) -> list[Path]:
    repo = repo.resolve()
    paths = {
        Path(module.__file__).resolve()
        for module in (
            aot,
            bench_coremark,
            compare_hot_function,
            aarch64_instruction_provenance,
        )
    } | {Path(__file__).resolve()}
    if any(not path.is_relative_to(repo) for path in paths):
        raise ProfileError("analysis dependencies must belong to the profiling checkout")
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if isinstance(filename, str) and filename.endswith((".py", ".pyc", ".so", ".pyd")):
            path = Path(filename)
            if not path.is_absolute():
                path = repo / path
            if not path.is_relative_to(repo):
                continue
            path = path.resolve()
            if not path.is_relative_to(repo):
                raise ProfileError("analysis dependency resolves outside the checkout")
            paths.add(path)
    return sorted(paths)


def capture_analysis_sources(repo: Path, aot, commit: str) -> dict[str, Any]:
    repo = repo.resolve()
    files = {}
    for path in analysis_source_paths(repo, aot):
        relative = path.relative_to(repo).as_posix()
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo,
            check=False,
            capture_output=True,
        )
        if committed.returncode != 0:
            raise ProfileError(
                f"analysis dependency {relative} is not tracked at {commit}"
            )
        current = path.read_bytes()
        if current != committed.stdout:
            raise ProfileError(
                f"analysis dependency {relative} differs from {commit}; "
                "commit tooling changes before an authoritative capture"
            )
        files[relative] = hashlib.sha256(current).hexdigest()
    return {
        "source_mode": "commit-verified",
        "commit": commit,
        "python_version": platform.python_version(),
        "files": files,
    }


def validate_analysis_sources_unchanged(repo: Path, sources: dict[str, Any]) -> None:
    for relative, expected in sources["files"].items():
        if sha256_file(repo / relative) != expected:
            raise ProfileError(f"analysis dependency {relative} changed during capture")


def perf_binary() -> str:
    return os.environ.get("PERF", "perf")


class CommandRecorder:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.commands: list[dict[str, Any]] = []

    def run(
        self,
        command: list[str],
        log_name: str,
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        display: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        rendered = display or shlex.join(str(part) for part in command)
        started = time.monotonic()
        proc = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
        )
        elapsed = time.monotonic() - started
        log = self.out_dir / log_name
        log.write_text(
            f"$ {rendered}\n"
            f"[exit={proc.returncode} elapsed_seconds={elapsed:.3f}]\n"
            f"--- stdout ---\n{proc.stdout}"
            f"{'' if proc.stdout.endswith(chr(10)) or not proc.stdout else chr(10)}"
            f"--- stderr ---\n{proc.stderr}"
            f"{'' if proc.stderr.endswith(chr(10)) or not proc.stderr else chr(10)}",
            encoding="utf-8",
        )
        self.commands.append(
            {
                "command": rendered,
                "cwd": str(cwd),
                "exit_code": proc.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "log": log.name,
            }
        )
        if proc.returncode:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise ProfileError(
                f"command failed ({rendered}): {detail or f'exit {proc.returncode}'}"
            )
        return proc


def parse_validated_coremark(output: str, engine: str, expected_runs: int) -> list[float]:
    if expected_runs != 1:
        raise ProfileError(
            "profile invocations are validated individually; expected_runs "
            "must be one"
        )
    try:
        parsed = bench_coremark.parse_coremark_output(output, engine)
    except RuntimeError as exc:
        raise ProfileError(str(exc)) from exc
    return [parsed.throughput]


def aggregate_wamr_rankings(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ProfileError("no WAMR attribution reports to aggregate")
    text_size = reports[0]["text_size"]
    function_count = reports[0]["function_count"]
    samples = Counter()
    code_bytes: dict[int, int] = {}
    for report in reports:
        if (
            report["text_size"] != text_size
            or report["function_count"] != function_count
        ):
            raise ProfileError("WAMR attribution captures disagree on cwasm layout")
        for item in report["top_functions"]:
            local_func = item["local_func"]
            if local_func in code_bytes and code_bytes[local_func] != item["code_bytes"]:
                raise ProfileError(
                    f"WAMR local_func={local_func} code size changed across captures"
                )
            code_bytes[local_func] = item["code_bytes"]
            samples[local_func] += item["samples"]
    total = sum(report["total_samples"] for report in reports)
    attributed = sum(report["attributed_samples"] for report in reports)
    top = [
        {
            "local_func": local_func,
            "samples": count,
            "percent_of_run": 100.0 * count / total,
            "code_bytes": code_bytes[local_func],
        }
        for local_func, count in samples.most_common()
    ]
    return {
        "total_samples": total,
        "attributed_samples": attributed,
        "attribution_coverage_pct": 100.0 * attributed / total,
        "text_size": text_size,
        "function_count": function_count,
        "top_functions": top,
    }


def validate_wamr_capture(
    report: dict[str, Any],
    *,
    minimum_samples: int,
    minimum_coverage_pct: float = MIN_ATTRIBUTION_COVERAGE_PCT,
) -> dict[str, Any]:
    total = report.get("total_samples")
    attributed = report.get("attributed_samples")
    mapping = report.get("mapping")
    if not isinstance(total, int) or total < minimum_samples:
        raise ProfileError(
            f"WAMR capture has {total!r} total samples; requires "
            f"at least {minimum_samples}"
        )
    if not isinstance(attributed, int) or not 0 < attributed <= total:
        raise ProfileError("WAMR capture has invalid attributed sample count")
    coverage = 100.0 * attributed / total
    if coverage < minimum_coverage_pct:
        raise ProfileError(
            f"WAMR capture attribution coverage {coverage:.4f}% is below "
            f"{minimum_coverage_pct:.4f}%"
        )
    if (
        not isinstance(mapping, dict)
        or mapping.get("authoritative") is not True
        or mapping.get("override") is not None
        or mapping.get("size") != mapping.get("expected_size")
    ):
        raise ProfileError(
            "WAMR capture did not use automatic exact-size mmap attribution"
        )
    return {
        "total_samples": total,
        "attributed_samples": attributed,
        "coverage_pct": coverage,
        "mapping": mapping,
    }


def validate_wasmtime_capture(
    capture: dict[str, Any],
    *,
    minimum_samples: int,
    minimum_coverage_pct: float = MIN_ATTRIBUTION_COVERAGE_PCT,
) -> dict[str, Any]:
    total = capture["total_samples"]
    attributed = sum(
        entry["samples"] for entry in capture["functions"].values()
    )
    if total < minimum_samples:
        raise ProfileError(
            f"Wasmtime capture has {total} samples; requires "
            f"at least {minimum_samples}"
        )
    coverage = 100.0 * attributed / total
    if coverage < minimum_coverage_pct:
        raise ProfileError(
            f"Wasmtime capture attribution coverage {coverage:.4f}% is below "
            f"{minimum_coverage_pct:.4f}%"
        )
    return {
        "total_samples": total,
        "attributed_samples": attributed,
        "coverage_pct": coverage,
    }


def aggregate_wamr_function(
    reports: list[dict[str, Any]],
    *,
    total_samples: int,
    function_start: int,
) -> dict[str, Any]:
    if not reports:
        raise ProfileError("no WAMR function reports to aggregate")
    local_func = reports[0]["classified_function"]["local_func"]
    class_samples = Counter()
    static_counts: dict[str, int] = {}
    hot = Counter()
    function_samples = 0
    for report in reports:
        function = report["classified_function"]
        if function["local_func"] != local_func:
            raise ProfileError("WAMR function captures disagree on local_func")
        function_samples += function["samples"]
        for name, values in function["classes"].items():
            static = values.get("static_instructions", 0)
            if name in static_counts and static_counts[name] != static:
                raise ProfileError(
                    f"WAMR {name} static count changed across captures"
                )
            static_counts[name] = static
            class_samples[name] += values["samples"]
        function_base = report["text_base"] + function_start
        for item in function["hottest_instructions"]:
            offset = item["address"] - function_base
            hot[(offset, item["instruction"])] += item["samples"]
    return {
        "local_func": local_func,
        "samples": function_samples,
        "percent_of_run": 100.0 * function_samples / total_samples,
        "instruction_count": reports[0]["classified_function"][
            "instruction_count"
        ],
        "classes": {
            name: {
                "samples": class_samples.get(name, 0),
                "percent_of_run": (
                    100.0 * class_samples.get(name, 0) / total_samples
                ),
                "static_instructions": static_counts.get(name, 0),
            }
            for name in sorted(set(static_counts) | set(class_samples))
        },
        "hottest_instructions": [
            {
                "offset": offset,
                "instruction": instruction,
                "samples": count,
                "percent_of_run": 100.0 * count / total_samples,
            }
            for (offset, instruction), count in hot.most_common(20)
        ],
    }


def aggregate_wasmtime_samples(
    captures: list[dict[str, Any]],
) -> dict[str, Any]:
    if not captures:
        raise ProfileError("no Wasmtime samples to aggregate")
    total = sum(capture["total_samples"] for capture in captures)
    functions: dict[int, dict[str, Any]] = {}
    for capture in captures:
        for wasm_index, source in capture["functions"].items():
            target = functions.setdefault(
                wasm_index,
                {
                    "samples": 0,
                    "names": set(),
                    "offsets": Counter(),
                    "mapping_methods": set(),
                },
            )
            target["samples"] += source["samples"]
            target["names"].update(source["names"])
            target["offsets"].update(source["offsets"])
            target["mapping_methods"].update(source["mapping_methods"])
    return {"total_samples": total, "functions": functions}


def parse_spill_metrics(text: str) -> dict[int, dict[str, Any]]:
    metrics: dict[int, dict[str, Any]] = {}
    integer_fields = {
        "local_func",
        "mod",
        "insts",
        "clobbers",
        "slots",
        "spilled_vregs",
        "scalar",
        "v128",
        "slots_scalar",
        "slots_v128",
        "spill_ld",
        "spill_st",
        "remat",
        "callee_saved",
    }
    for line in text.splitlines():
        if "[aot-spill-metric]" not in line:
            continue
        fields: dict[str, Any] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = int(value) if key in integer_fields else value
        local_func = fields.get("local_func")
        if not isinstance(local_func, int):
            raise ProfileError(f"malformed spill metric line: {line}")
        if local_func in metrics:
            raise ProfileError(f"duplicate spill metric for local_func={local_func}")
        metrics[local_func] = fields
    return metrics


def parse_wasmtime_samples(
    text: str,
    identity: compare_hot_function.WasmModuleIdentity | None = None,
) -> dict[str, Any]:
    total = 0
    functions: dict[int, dict[str, Any]] = {}
    names_to_indices: dict[str, list[int]] = {}
    if identity is not None:
        for index, name in identity.function_names.items():
            names_to_indices.setdefault(name, []).append(index)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        total += 1
        matches = list(WASMTIME_SYMBOL_RE.finditer(stripped))
        if len(matches) > 1:
            raise ProfileError(f"ambiguous Wasmtime sample mapping: {line}")
        if matches:
            match = matches[0]
            if int(match.group("module")) != 0:
                continue
            wasm_index = int(match.group("function"))
            name = match.group("name")
            offset_match = re.search(
                r"(?:^|\s|\+)(?:0x)([0-9a-fA-F]+)(?=\s|$|\()",
                stripped[match.end() :],
            )
            offset = int(offset_match.group(1), 16) if offset_match else None
            mapping = "full_wasm_symbol"
        else:
            plain = WASMTIME_JITTED_SYMBOL_RE.match(stripped)
            if plain is None or identity is None:
                continue
            name = plain.group("name")
            name_matches = names_to_indices.get(name, [])
            if len(name_matches) != 1:
                if name_matches:
                    raise ProfileError(
                        f"Wasmtime JIT symbol {name!r} is ambiguous at wasm "
                        f"indices {name_matches}"
                    )
                continue
            wasm_index = name_matches[0]
            expected_local = wasm_index - identity.imported_function_count
            dso_local = int(plain.group("local"))
            if expected_local != dso_local:
                raise ProfileError(
                    f"Wasmtime JIT DSO {plain.group('dso')} implies local "
                    f"function {dso_local}, but name-section mapping for "
                    f"{name!r} implies {expected_local}"
                )
            offset = (
                int(plain.group("offset"), 16)
                if plain.group("offset") is not None
                else None
            )
            mapping = "name_section_and_jitted_dso_local_index"
        entry = functions.setdefault(
            wasm_index,
            {
                "samples": 0,
                "names": set(),
                "offsets": Counter(),
                "mapping_methods": set(),
            },
        )
        entry["samples"] += 1
        entry["mapping_methods"].add(mapping)
        if name:
            entry["names"].add(name)
        if offset is not None:
            entry["offsets"][offset] += 1
    if total == 0:
        raise ProfileError("perf script produced no Wasmtime samples")
    return {"total_samples": total, "functions": functions}


def validate_wasmtime_mapping(
    parsed: dict[str, Any],
    identity: compare_hot_function.WasmModuleIdentity,
) -> None:
    for wasm_index, entry in parsed["functions"].items():
        if wasm_index < identity.imported_function_count:
            raise ProfileError(
                f"Wasmtime mapped generated code to imported function {wasm_index}"
            )
        if wasm_index >= identity.function_count:
            raise ProfileError(
                f"Wasmtime function index {wasm_index} exceeds module function count"
            )
        expected_name = identity.function_names.get(wasm_index)
        names = entry["names"]
        if len(names) > 1:
            raise ProfileError(
                f"Wasmtime function {wasm_index} has ambiguous names {sorted(names)}"
            )
        if names and expected_name and names != {expected_name}:
            raise ProfileError(
                f"Wasmtime function {wasm_index} name mismatch: "
                f"{sorted(names)} != {expected_name!r}"
            )


def class_samples(classes: dict[str, dict[str, Any]], names: set[str]) -> int:
    return sum(
        int(values.get("samples", 0))
        for name, values in classes.items()
        if name in names
    )


def classify_wasmtime_function(
    *,
    aot,
    objdump_text: str,
    wasm_index: int,
    offsets: Counter,
    total_samples: int,
) -> dict[str, Any]:
    parsed = compare_hot_function.parse_disassembly(
        objdump_text, wasmtime_wasm_index=wasm_index
    )
    instructions = [
        aot.Instruction(
            address=item.offset,
            offset=item.offset,
            size=item.size,
            text=item.text,
        )
        for item in parsed
    ]
    classes = aot.classify_instruction_stream(
        instructions, architecture="aarch64"
    )
    starts = [item.offset for item in parsed]
    by_class = Counter()
    by_instruction = Counter()
    unresolved = 0
    for offset, samples in offsets.items():
        index = bisect.bisect_right(starts, offset) - 1
        if index < 0 or offset >= parsed[index].offset + parsed[index].size:
            unresolved += samples
            continue
        by_class[classes[index]] += samples
        by_instruction[index] += samples
    function_samples = sum(offsets.values())
    mapped = function_samples - unresolved
    if function_samples and mapped * 100 < function_samples * 90:
        raise ProfileError(
            f"Wasmtime function {wasm_index} instruction mapping covered only "
            f"{mapped}/{function_samples} samples"
        )
    hot = []
    for index, samples in by_instruction.most_common(20):
        item = parsed[index]
        hot.append(
            {
                "offset": item.offset,
                "samples": samples,
                "percent_of_run": 100.0 * samples / total_samples,
                "instruction": item.text,
            }
        )
    return {
        "instruction_count": len(parsed),
        "mapped_instruction_samples": mapped,
        "unresolved_instruction_samples": unresolved,
        "mapping_coverage_pct": (
            100.0 * mapped / function_samples if function_samples else 0.0
        ),
        "classes": {
            name: {
                "samples": by_class.get(name, 0),
                "percent_of_run": 100.0 * by_class.get(name, 0) / total_samples,
                "static_instructions": classes.count(name),
            }
            for name in sorted(set(classes) | set(by_class))
        },
        "hottest_instructions": hot,
    }


def _wasmtime_function_address_base(
    objdump_text: str, wasm_index: int
) -> int:
    for line in objdump_text.splitlines():
        match = compare_hot_function.WASMTIME_HEADER_RE.match(line)
        if (
            match is not None
            and int(match.group(1)) == 0
            and int(match.group(2)) == wasm_index
        ):
            address = re.match(r"\s*([0-9a-fA-F]+)\s+", line)
            if address is None:
                raise ProfileError(
                    f"Wasmtime function {wasm_index} header lacks an address"
                )
            return int(address.group(1), 16)
    raise ProfileError(
        f"Wasmtime objdump lacks function header for wasm index {wasm_index}"
    )


def _map_samples_to_instruction_starts(
    instructions: list[compare_hot_function.Instruction],
    offsets: Counter,
) -> tuple[Counter, int]:
    starts = [item.offset for item in instructions]
    mapped = Counter()
    unresolved = 0
    for offset, samples in offsets.items():
        index = bisect.bisect_right(starts, offset) - 1
        if index < 0 or offset >= instructions[index].offset + instructions[index].size:
            unresolved += samples
            continue
        mapped[instructions[index].offset] += samples
    return mapped, unresolved


def analyze_wasmtime_alu_provenance(
    *,
    aot,
    objdump_text: str,
    wasm_index: int,
    offsets: Counter,
    total_samples: int,
    global_attributed_samples: int | None = None,
) -> dict[str, Any]:
    instructions = compare_hot_function.parse_disassembly(
        objdump_text, wasmtime_wasm_index=wasm_index
    )
    adapter = [
        aot.Instruction(
            address=item.offset,
            offset=item.offset,
            size=item.size,
            text=item.text,
        )
        for item in instructions
    ]
    broad_classes = aot.classify_instruction_stream(
        adapter, architecture="aarch64"
    )
    mapped, unresolved = _map_samples_to_instruction_starts(
        instructions, offsets
    )
    result = aarch64_instruction_provenance.analyze_instruction_stream(
        instructions,
        broad_classes=broad_classes,
        samples_by_offset=mapped,
        total_run_samples=total_samples,
        address_base=_wasmtime_function_address_base(
            objdump_text, wasm_index
        ),
        global_attributed_samples=global_attributed_samples,
    )
    result["sample_mapping"] = {
        "mapped_function_samples": sum(mapped.values()),
        "unresolved_function_samples": unresolved,
    }
    return result


def analyze_wamr_alu_provenance(
    *,
    aot,
    cwasm_info,
    local_func: int,
    capture_counts: list[tuple[dict[int, int], int]],
    total_samples: int,
    scratch_dir: Path,
    global_attributed_samples: int | None = None,
) -> dict[str, Any]:
    start, end = aot.function_bounds(cwasm_info, local_func)
    function_code = cwasm_info.data[
        cwasm_info.text_file_offset + start :
        cwasm_info.text_file_offset + end
    ]
    instructions = aot.disassemble_blob(
        function_code,
        0,
        scratch_dir,
        f"alu-provenance-{local_func}",
        architecture="aarch64",
    )
    broad_classes = aot.classify_instruction_stream(
        instructions, architecture="aarch64"
    )
    samples = Counter()
    instruction_offsets = {instruction.offset for instruction in instructions}
    unresolved = 0
    for counts, text_base in capture_counts:
        function_base = text_base + start
        for address, count in counts.items():
            if not function_base <= address < function_base + len(function_code):
                continue
            offset = address - function_base
            if offset in instruction_offsets:
                samples[offset] += count
            else:
                unresolved += count
    result = aarch64_instruction_provenance.analyze_instruction_stream(
        instructions,
        broad_classes=broad_classes,
        samples_by_offset=samples,
        total_run_samples=total_samples,
        global_attributed_samples=global_attributed_samples,
    )
    result["sample_mapping"] = {
        "mapped_function_samples": sum(samples.values()),
        "unresolved_function_samples": unresolved,
    }
    return result


def analyze_wamr_frame_provenance(
    *,
    aot,
    cwasm_info,
    metadata_path: Path,
    local_func: int,
    capture_counts: list[tuple[dict[int, int], int]],
    ranking_reports: list[dict[str, Any]],
    total_samples: int,
    spill_metric: dict[str, Any],
    scratch_dir: Path,
) -> dict[str, Any]:
    start, end = aot.function_bounds(cwasm_info, local_func)
    text = cwasm_info.data[
        cwasm_info.text_file_offset : cwasm_info.text_file_offset + cwasm_info.text_size
    ]
    metadata = aot.load_frame_metadata(
        metadata_path, local_func, text[start:end], cwasm_info.version,
        text, start, cwasm_info,
    )
    if metadata.raw["module"] != 0:
        raise ProfileError("CoreMark frame metadata must describe module 0")
    instructions = aot.disassemble_function(
        text[start:end], start, scratch_dir, f"frame-provenance-{local_func}",
        metadata.inline_data_ranges, metadata.raw["architecture"],
    )
    aot.validate_metadata_disassembly(metadata, instructions)
    if len(capture_counts) != len(ranking_reports):
        raise ProfileError("frame counts and ranking captures disagree")
    combined = Counter()
    captures = []
    for ordinal, ((counts, text_base), ranking) in enumerate(
        zip(capture_counts, ranking_reports), 1
    ):
        normalized = Counter(
            {
                address - text_base: count
                for address, count in counts.items()
                if text_base + start <= address < text_base + end
            }
        )
        expected = sum(
            item["samples"]
            for item in ranking["top_functions"]
            if item["local_func"] == local_func
        )
        if sum(normalized.values()) != expected or expected <= 0:
            raise ProfileError(
                "frame function samples do not match the captured function ranking"
            )
        combined.update(normalized)
        captures.append(
            {
                "ordinal": ordinal,
                "total_samples": ranking["total_samples"],
                "function_samples": expected,
                "samples_by_offset": [
                    [address - start, count]
                    for address, count in sorted(normalized.items())
                ],
            }
        )
    # Reuse the consumer once on merged IP counts: pair samples are indivisible,
    # while static component counts and per-vreg snapshots must not be added.
    summary = aot.build_frame_summary(instructions, combined, metadata)
    reconciliation = summary["reconciliation"]
    if (
        reconciliation["matches"] is not True
        or reconciliation["emitted_allocator_loads"] != spill_metric.get("spill_ld")
        or reconciliation["emitted_allocator_stores"] != spill_metric.get("spill_st")
    ):
        raise ProfileError("frame origins disagree with the emitted spill metric log")
    for values in summary["origins"].values():
        values["percent_of_run"] = 100.0 * values["samples"] / total_samples
    for contributor in summary["allocator_contributors"]:
        contributor["percent_of_run"] = (
            100.0 * contributor["samples"] / total_samples
        )
    return {
        "schema_version": 1,
        "module": 0,
        "local_func": local_func,
        "code_size": end - start,
        "function_samples": sum(combined.values()),
        "total_samples": total_samples,
        "sample_coordinates": "function-relative native byte offsets",
        "metadata": {
            "path": metadata_path.name,
            "sha256": sha256_file(metadata_path),
            **{
                key: metadata.raw[key]
                for key in (
                    "schema", "schema_version", "architecture", "abi",
                    "compiler_build_id", "module_text_sha256", "normalized_code_sha256",
                )
            },
        },
        "captures": captures,
        "summary": summary,
    }


def assemble_alu_provenance(
    *,
    wamr: dict[str, Any],
    wasmtime: dict[str, Any],
    expected_wamr_all_alu_samples: int,
    expected_wasmtime_all_alu_samples: int,
) -> dict[str, Any]:
    if wamr.get("status") == "measured" and (
        wamr.get("broad_alu_samples") != expected_wamr_all_alu_samples
    ):
        raise ProfileError(
            "WAMR narrow ALU partitions do not match existing all_alu samples"
        )
    if wasmtime.get("status") == "measured" and (
        wasmtime.get("broad_alu_samples")
        != expected_wasmtime_all_alu_samples
    ):
        raise ProfileError(
            "Wasmtime narrow ALU partitions do not match existing all_alu samples"
        )
    return {
        "schema_version": aarch64_instruction_provenance.SCHEMA_VERSION,
        "kind": aarch64_instruction_provenance.ANALYSIS_KIND,
        "wording": NARROW_ALU_WORDING,
        "wamr": wamr,
        "wasmtime": wasmtime,
        "gate": aarch64_instruction_provenance.compare_engine_analyses(
            wamr, wasmtime
        ),
    }


def load_benchmark_report(path: Path) -> tuple[dict[str, Any], str]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read benchmark report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ProfileError("benchmark report root must be a JSON object")
    try:
        bench_coremark.validate_authoritative_benchmark_report(report)
    except RuntimeError as exc:
        raise ProfileError(str(exc)) from exc
    return report, sha256_file(path)


def profile_report_status(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("kind") != REPORT_KIND:
        raise ProfileError("profile report has an unsupported kind")
    if report.get("schema_version") == HISTORICAL_REPORT_SCHEMA_VERSION:
        return {
            "status": "historical-unverified",
            "authoritative": False,
            "baseline_run": report.get("authoritative_baseline_run"),
            "known_historical_baseline": (
                report.get("authoritative_baseline_run")
                == HISTORICAL_BASELINE_RUN
            ),
            "reason": (
                "schema version 1 named a historical run without embedding "
                "validated benchmark/report provenance"
            ),
        }
    validate_report(report)
    return {"status": "current", "authoritative": True}


def _require_sha(value: object, label: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or re.fullmatch(r"[0-9a-f]+", value) is None
    ):
        raise ProfileError(f"{label} must be a {length}-character lowercase hex digest")
    return value


def _validate_narrow_alu_analysis(
    analysis: dict[str, Any],
    *,
    engine: str,
    expected_samples: int,
    function_samples: int,
    total_samples: int,
    global_attributed_samples: int,
) -> None:
    try:
        aarch64_instruction_provenance.validate_analysis_samples(
            analysis, name=engine
        )
    except aarch64_instruction_provenance.ProvenanceError as exc:
        raise ProfileError(str(exc)) from exc
    if (
        analysis.get("total_run_samples") != total_samples
        or analysis.get("broad_alu_samples") != expected_samples
    ):
        raise ProfileError(
            f"{engine} narrow ALU samples do not reconcile to all_alu"
        )
    if analysis["global_sample_mapping"]["attributed_samples"] != global_attributed_samples:
        raise ProfileError(f"{engine} global sample mapping is invalid")
    mapped = analysis["instruction_sample_accounting"]["mapped_instruction_samples"]
    unresolved = analysis.get("sample_mapping", {}).get("unresolved_function_samples", 0)
    if mapped + unresolved != function_samples:
        raise ProfileError(
            f"{engine} instruction samples do not reconcile to its matched function"
        )


def _frame_count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProfileError(f"invalid frame {name} count")
    return value


def validate_frame_provenance(frame: dict[str, Any], report: dict[str, Any]) -> None:
    local_func = _frame_count(frame.get("local_func"), "local function")
    total = report["engines"]["wamr"]["total_samples"]
    if (
        frame.get("schema_version") != 1
        or frame.get("module") != 0
        or local_func >= report["wasm"]["local_function_count"]
        or frame.get("wasm_function_index")
        != local_func + report["wasm"]["imported_function_count"]
        or frame.get("total_samples") != total
        or frame.get("compiler_sha256") != report["wamr"]["compiler_sha256"]
        or frame.get("cwasm_sha256") != report["wamr"]["cwasm_sha256"]
        or frame.get("sample_coordinates") != "function-relative native byte offsets"
    ):
        raise ProfileError("frame provenance disagrees with the measured artifact")
    metadata = frame["metadata"]
    if (
        metadata.get("schema") != "wamr-aot-frame-attribution"
        or metadata.get("schema_version") != 2
        or metadata.get("architecture") != "aarch64"
        or metadata.get("abi") != "aapcs64"
        or metadata.get("path") != f"wamr-frame.mod0.func{local_func}.json"
    ):
        raise ProfileError("frame metadata identity is incompatible")
    for key in ("sha256", "module_text_sha256", "normalized_code_sha256"):
        _require_sha(metadata.get(key), f"frame metadata {key}", 64)
    code_size = _frame_count(frame.get("code_size"), "code size")
    if not code_size or code_size % 4:
        raise ProfileError("frame code size is not AArch64 instruction-aligned")
    captures = frame["captures"]
    if len(captures) != PROFILE_CAPTURES_PER_ENGINE:
        raise ProfileError("frame provenance must contain both WAMR captures")
    function_samples = 0
    for ordinal, (capture, measured) in enumerate(
        zip(captures, report["wamr_captures"]), 1
    ):
        if (
            capture.get("ordinal") != ordinal
            or capture.get("total_samples") != measured["total_samples"]
        ):
            raise ProfileError("frame captures do not match the measured schedule")
        samples = 0
        previous = -1
        for offset, count in capture["samples_by_offset"]:
            _frame_count(offset, "instruction offset")
            _frame_count(count, "instruction samples")
            if not previous < offset < code_size or offset % 4 or count == 0:
                raise ProfileError("frame instruction samples have invalid offsets/counts")
            previous = offset
            samples += count
        if (
            samples != capture.get("function_samples")
            or not 0 < samples <= measured["attributed_samples"]
        ):
            raise ProfileError("frame function samples exceed their capture budget")
        function_samples += samples
    expected = sum(
        item["samples"]
        for item in report["engines"]["wamr"]["top_functions"]
        if item["local_func"] == local_func
    )
    if frame.get("function_samples") != function_samples or expected != function_samples:
        raise ProfileError("frame function samples do not reconcile to the ranking")
    summary = frame["summary"]
    coverage = summary["coverage"]
    for suffix in ("instructions", "samples"):
        whole = _frame_count(coverage.get(f"frame_{suffix}"), suffix)
        mapped = _frame_count(coverage.get(f"attributed_frame_{suffix}"), f"mapped {suffix}")
        proven = _frame_count(coverage.get(f"proven_origin_frame_{suffix}"), f"proven {suffix}")
        unknown = _frame_count(coverage.get(f"unknown_frame_{suffix}"), f"unknown {suffix}")
        if not proven <= mapped <= whole or unknown != whole - proven:
            raise ProfileError("frame coverage does not conserve counts")
    if (
        coverage["frame_samples"] > function_samples
        or coverage["frame_instructions"] > code_size // 4
    ):
        raise ProfileError("frame coverage exceeds the selected function")
    origins = summary["origins"]
    if not set(origins) <= {
        "allocator_spill", "wasm_local_or_phi", "explicit_frame_storage",
        "fixed_runtime_frame_state", "unknown",
    }:
        raise ProfileError("frame report has an unsupported origin")
    for key, suffix in (("samples", "samples"), ("static_instructions", "instructions")):
        values = {
            origin: _frame_count(value.get(key), f"{origin} {key}")
            for origin, value in origins.items()
        }
        if (
            sum(values.values()) != coverage[f"frame_{suffix}"]
            or sum(count for origin, count in values.items() if origin != "unknown")
            != coverage[f"proven_origin_frame_{suffix}"]
        ):
            raise ProfileError("frame origins do not conserve coverage counts")
    for value in origins.values():
        if value.get("percent_of_run") != 100.0 * value["samples"] / total:
            raise ProfileError("frame origin percentage disagrees with its samples")
    contributors = summary["allocator_contributors"]
    contributor_samples = sum(
        _frame_count(item.get("samples"), "allocator contributor") for item in contributors
    )
    if contributor_samples != origins.get("allocator_spill", {}).get("samples", 0):
        raise ProfileError("allocator contributor samples are duplicated or missing")
    for item in contributors:
        if item.get("percent_of_run") != 100.0 * item["samples"] / total:
            raise ProfileError("allocator percentage disagrees with its samples")
    reconciliation = summary["reconciliation"]
    if reconciliation.get("matches") is not True:
        raise ProfileError("frame emitted counts did not reconcile")
    for direction in ("loads", "stores"):
        emitted = _frame_count(
            reconciliation.get(f"emitted_allocator_{direction}"), f"emitted {direction}"
        )
        metric = _frame_count(
            reconciliation.get(f"spill_metric_{direction}"), f"spill metric {direction}"
        )
        ranked_components = sum(
            _frame_count(item.get(f"static_{direction}"), f"component {direction}")
            for item in contributors
        )
        component_counts = summary["allocator_component_counts"]
        components = _frame_count(
            component_counts.get(f"total_{direction}"), f"total component {direction}"
        )
        unranked = _frame_count(
            component_counts.get(f"unranked_{direction}"), f"unranked component {direction}"
        )
        if not emitted == metric == components == ranked_components + unranked:
            raise ProfileError("frame static component counts were duplicated or lost")


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ProfileError("profile report has an unsupported schema_version")
    if report.get("kind") != REPORT_KIND:
        raise ProfileError("profile report has an unsupported kind")
    if report.get("architecture") != "aarch64":
        raise ProfileError("profile report architecture must be aarch64")
    benchmark = report.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ProfileError("profile report lacks benchmark provenance")
    if not benchmark.get("report_id") or not benchmark.get("generated_at"):
        raise ProfileError("profile report has incomplete benchmark linkage")
    _require_sha(benchmark.get("report_sha256"), "benchmark report sha256", 64)
    handoff = benchmark.get("artifact_handoff")
    if not isinstance(handoff, dict) or not handoff.get("directory"):
        raise ProfileError("profile report lacks benchmark artifact handoff")
    _require_sha(
        handoff.get("manifest_sha256"), "benchmark handoff manifest sha256", 64
    )
    selected_role = benchmark.get("selected_role", "wamr-target")
    selected = benchmark.get("selected_wamr")
    legacy_target = selected is None and selected_role == "wamr-target"
    if selected is None and selected_role == "wamr-target":
        selected = benchmark.get("target")
    if (
        selected_role not in ("wamr-baseline", "wamr-target")
        or not isinstance(selected, dict)
        or selected.get("role", selected_role if legacy_target else None)
        != selected_role
    ):
        raise ProfileError("profile report lacks its selected benchmark WAMR identity")
    if "target" in benchmark and (
        selected_role != "wamr-target" or benchmark["target"] != selected
    ):
        raise ProfileError("profile benchmark target alias contradicts its selected role")
    wamr = report.get("wamr", {})
    if wamr.get("benchmark_role", "wamr-target") != selected_role:
        raise ProfileError("profile and benchmark WAMR roles differ")
    selected_source = selected.get("identity", {}).get("source", {})
    if selected_source.get("sha") != wamr.get("commit"):
        raise ProfileError("profile and benchmark WAMR source SHAs differ")
    selected_identity = selected.get("identity", {})
    if (
        selected_identity.get("runtime", {}).get("sha256")
        != wamr.get("runtime_sha256")
        or selected_identity.get("compiler", {}).get("sha256")
        != wamr.get("compiler_sha256")
        or selected_identity.get("module", {}).get("sha256")
        != wamr.get("cwasm_sha256")
    ):
        raise ProfileError("profile and benchmark WAMR tool identities differ")
    baseline = benchmark.get("wasmtime_baseline")
    if not isinstance(baseline, dict):
        raise ProfileError("profile report lacks benchmark Wasmtime identity")
    baseline_identity = baseline.get("identity", {})
    if (
        baseline_identity.get("version") != report.get("wasmtime", {}).get("version")
        or baseline_identity.get("runtime", {}).get("sha256")
        != report.get("wasmtime", {}).get("sha256")
    ):
        raise ProfileError("profile and benchmark Wasmtime identities differ")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ProfileError("profile report lacks producer provenance")
    _require_sha(
        provenance.get("producer_source_sha"), "profile producer source SHA", 40
    )
    _require_sha(
        provenance.get("script_sha256"), "profile producer script sha256", 64
    )
    if (
        provenance.get("producer_source_sha")
        != benchmark.get("producer", {}).get("source_sha")
    ):
        raise ProfileError("profile and benchmark tooling source SHAs differ")
    if provenance.get("execution") != benchmark.get("execution"):
        raise ProfileError("profile and benchmark execution provenance differ")
    if report.get("guest_args") != list(bench_coremark.COREMARK_GUEST_ARGS):
        raise ProfileError("profile report guest args are not authoritative")
    if report.get("expected_iterations") != bench_coremark.EXPECTED_ITERATIONS:
        raise ProfileError("profile report iteration count is not authoritative")
    if report.get("classifier_wording", {}).get("all_alu") != ALL_ALU_WORDING:
        raise ProfileError("profile report has ambiguous ALU classifier wording")
    narrow_summary = report.get("alu_provenance")
    if narrow_summary is not None:
        if (
            report.get("classifier_wording", {}).get(
                "narrow_alu_provenance"
            )
            != NARROW_ALU_WORDING
        ):
            raise ProfileError(
                "profile report has ambiguous narrow ALU classifier wording"
            )
        analysis_module = provenance.get("analysis_module")
        if not isinstance(analysis_module, dict):
            raise ProfileError("profile report lacks ALU analysis module identity")
        _require_sha(
            analysis_module.get("sha256"),
            "ALU analysis module sha256",
            64,
        )
        sources = provenance.get("analysis_sources")
        if (
            not isinstance(sources, dict)
            or sources.get("source_mode") != "commit-verified"
            or sources.get("commit") != provenance["producer_source_sha"]
            or not isinstance(sources.get("files"), dict)
            or not REQUIRED_ANALYSIS_SOURCES.issubset(sources["files"])
        ):
            raise ProfileError("profile report lacks complete committed analysis sources")
        for path, digest in sources["files"].items():
            if (
                not isinstance(path, str)
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ProfileError("profile analysis source path is invalid")
            _require_sha(digest, f"analysis source {path} sha256", 64)
        if (
            sources["files"]["scripts/profile_coremark_aarch64.py"]
            != provenance["script_sha256"]
            or provenance.get("script_path") != "scripts/profile_coremark_aarch64.py"
            or sources["files"]["scripts/aarch64_instruction_provenance.py"]
            != analysis_module["sha256"]
            or analysis_module.get("path") != "scripts/aarch64_instruction_provenance.py"
            or sources["files"]["scripts/bench_coremark.py"]
            != benchmark.get("producer", {}).get("script", {}).get("sha256")
        ):
            raise ProfileError("profile analysis source identities disagree")
        if (
            narrow_summary.get("schema_version")
            != aarch64_instruction_provenance.SCHEMA_VERSION
            or narrow_summary.get("kind")
            != aarch64_instruction_provenance.ANALYSIS_KIND
        ):
            raise ProfileError("profile report has invalid narrow ALU summary")
    affinity = report.get("affinity")
    if not isinstance(affinity, dict) or affinity.get("verified") is not True:
        raise ProfileError("profile report lacks verified CPU affinity")
    schedule = report.get("profile_schedule")
    if not isinstance(schedule, list) or len(schedule) != 8:
        raise ProfileError("profile report must contain four warmups and four captures")
    order = [item.get("engine") for item in schedule]
    phases = [item.get("phase") for item in schedule]
    if order != ["wamr", "wasmtime", "wasmtime", "wamr"] * 2:
        raise ProfileError("profile report execution order is not ABBA/ABBA")
    if phases != ["warmup"] * 4 + ["profile"] * 4:
        raise ProfileError("profile report phases are not balanced")
    if report.get("minimum_attribution_coverage_pct") != MIN_ATTRIBUTION_COVERAGE_PCT:
        raise ProfileError("profile report has the wrong coverage threshold")
    wamr_captures = report.get("wamr_captures")
    wasmtime_captures = report.get("wasmtime_captures")
    if (
        not isinstance(wamr_captures, list)
        or len(wamr_captures) != PROFILE_CAPTURES_PER_ENGINE
        or not isinstance(wasmtime_captures, list)
        or len(wasmtime_captures) != PROFILE_CAPTURES_PER_ENGINE
    ):
        raise ProfileError("profile report must validate two captures per engine")
    for capture in wamr_captures:
        if (
            capture.get("coverage_pct", 0) < MIN_ATTRIBUTION_COVERAGE_PCT
            or capture.get("mapping", {}).get("authoritative") is not True
        ):
            raise ProfileError("WAMR capture failed authoritative attribution")
    for capture in wasmtime_captures:
        if capture.get("coverage_pct", 0) < MIN_ATTRIBUTION_COVERAGE_PCT:
            raise ProfileError("Wasmtime capture failed attribution coverage")
    engines = report.get("engines")
    if not isinstance(engines, dict) or set(engines) != {"wamr", "wasmtime"}:
        raise ProfileError("profile report must contain WAMR and Wasmtime engines")
    for name, engine in engines.items():
        total = engine.get("total_samples")
        attributed = engine.get("attributed_samples")
        if (
            not isinstance(total, int)
            or total <= 0
            or not isinstance(attributed, int)
            or attributed <= 0
            or attributed > total
        ):
            raise ProfileError(f"invalid {name} sample totals")
    matched = report.get("matched_functions")
    if not isinstance(matched, list) or not matched:
        raise ProfileError("profile report has no matched functions")
    for item in matched:
        expected = item["local_func"] + report["wasm"]["imported_function_count"]
        if item["wasm_function_index"] != expected:
            raise ProfileError("local_func/wasm function mapping is inconsistent")
        if item["wamr"]["samples"] <= 0 or item["wasmtime"]["samples"] <= 0:
            raise ProfileError("matched functions must have samples in both engines")
        if narrow_summary is not None:
            narrow = item.get("alu_provenance")
            if (
                not isinstance(narrow, dict)
                or narrow.get("wording") != NARROW_ALU_WORDING
            ):
                raise ProfileError(
                    "matched function lacks narrow ALU provenance"
                )
            _validate_narrow_alu_analysis(
                narrow.get("wamr", {}),
                engine="WAMR",
                expected_samples=item["class_groups"]["all_alu"][
                    "wamr_samples"
                ],
                function_samples=item["wamr"]["samples"],
                total_samples=engines["wamr"]["total_samples"],
                global_attributed_samples=engines["wamr"]["attributed_samples"],
            )
            _validate_narrow_alu_analysis(
                narrow.get("wasmtime", {}),
                engine="Wasmtime",
                expected_samples=item["class_groups"]["all_alu"][
                    "wasmtime_samples"
                ],
                function_samples=item["wasmtime"]["samples"],
                total_samples=engines["wasmtime"]["total_samples"],
                global_attributed_samples=engines["wasmtime"]["attributed_samples"],
            )
            gate = narrow.get("gate")
            if (
                not isinstance(gate, dict)
                or "all_alu" in gate.get("categories", {})
                or gate != aarch64_instruction_provenance.compare_engine_analyses(
                    narrow["wamr"], narrow["wasmtime"]
                )
            ):
                raise ProfileError(
                    "narrow ALU gate does not match its validated sample evidence"
                )
    if narrow_summary is not None:
        retained = report.get("retained_analysis_artifacts", {}).get(
            "wamr_cwasm"
        )
        if (
            not isinstance(retained, dict)
            or retained.get("retained") is not True
            or retained.get("source_sha256") != wamr.get("cwasm_sha256")
        ):
            raise ProfileError(
                "profile report did not retain the exact analyzed WAMR cwasm"
            )
        target = next(
            (
                item
                for item in matched
                if item["local_func"]
                == narrow_summary.get("target_local_func")
                and item["wasm_function_index"]
                == narrow_summary.get("target_wasm_function_index")
            ),
            None,
        )
        if target is not None and (
            narrow_summary.get("gate")
            != target["alu_provenance"].get("gate")
        ):
            raise ProfileError(
                "top-level narrow ALU gate does not match its target function"
            )
        if target is None and narrow_summary.get("gate", {}).get("optimization_authorized"):
            raise ProfileError("unmatched target function cannot authorize optimization")
    if report.get("frame_attribution") is not None:
        if narrow_summary is None:
            raise ProfileError("frame profiles require complete committed analysis sources")
        validate_frame_provenance(report["frame_attribution"], report)


def render_markdown(report: dict[str, Any]) -> str:
    host = report["host"]
    wamr = report["engines"]["wamr"]
    wasmtime = report["engines"]["wasmtime"]
    benchmark = report["benchmark"]
    execution = benchmark["execution"]
    if execution.get("provider") == "github-actions":
        repository = execution["repository"]
        run_id = execution["run_id"]
        benchmark_link = (
            f"[{run_id}](https://github.com/{repository}/actions/runs/{run_id})"
        )
    else:
        benchmark_link = f"`{execution['run_id']}`"
    order = "".join(
        "A" if item["engine"] == "wamr" else "B"
        for item in report["profile_schedule"]
    )
    wamr_capture_coverage = ", ".join(
        f"{item['coverage_pct']:.4f}%" for item in report["wamr_captures"]
    )
    wasmtime_capture_coverage = ", ".join(
        f"{item['coverage_pct']:.4f}%"
        for item in report["wasmtime_captures"]
    )
    lines = [
        "### Matched-host AArch64 CoreMark profiles",
        "",
        f"- Commit: `{report['wamr']['commit']}` (`ReleaseFast`)",
        f"- Authoritative benchmark: report `{benchmark['report_id']}` "
        f"(`sha256:{benchmark['report_sha256']}`), run {benchmark_link}",
        f"- Exact WAMR artifact handoff: manifest "
        f"`sha256:{benchmark['artifact_handoff']['manifest_sha256']}`",
        f"- Benchmarked WAMR role: `{benchmark.get('selected_role', 'wamr-target')}` "
        f"at `{(benchmark.get('selected_wamr') or benchmark.get('target'))['identity']['source']['sha']}`; "
        f"producer tooling `{benchmark['producer']['source_sha']}`",
        f"- Fixture: `{report['fixture']['path']}` "
        f"(`sha256:{report['fixture']['sha256']}`)",
        f"- Fixed guest args: `{' '.join(report['guest_args'])}`; every run "
        f"required `Iterations: {report['expected_iterations']}`",
        f"- CPU affinity: allowed "
        f"`{','.join(map(str, report['affinity']['allowed_cpus']))}`; "
        f"selected/verified CPU `{report['affinity']['selected_cpu']}` via "
        f"`{report['affinity']['taskset']}`",
        f"- Counterbalanced execution order: `{order}` "
        f"(A=WAMR, B=Wasmtime; warmups then profile captures)",
        f"- Per-capture attribution gate: "
        f"`≥{report['minimum_attribution_coverage_pct']:.2f}%`; WAMR "
        f"`{wamr_capture_coverage}`, Wasmtime "
        f"`{wasmtime_capture_coverage}`",
        f"- Host: `{host['architecture']}` · {host['cpu_count']} vCPU · "
        f"`{host['cpu_model']}` · kernel `{host['kernel']}` · "
        f"fingerprint `{host['fingerprint']}`",
        f"- perf: `{report['perf']['version']}` · package "
        f"`{report['perf']['package']}` · paranoid "
        f"`{report['perf']['paranoid_initial']} → "
        f"{report['perf']['paranoid_effective']}` · native sampling verified",
        f"- Wasmtime: `{report['wasmtime']['version']}` "
        f"(`sha256:{report['wasmtime']['sha256']}`), `--profile=jitdump`",
        f"- Validated runs: WAMR {wamr['wall_seconds']:.2f}s at "
        f"{wamr['iterations_per_second']:.1f} iter/s; Wasmtime "
        f"{wasmtime['wall_seconds']:.2f}s at "
        f"{wasmtime['iterations_per_second']:.1f} iter/s",
        "",
        "| Engine | Total self samples | Generated-wasm samples | Engine share |",
        "|---|---:|---:|---:|",
        f"| WAMR | {wamr['total_samples']} | {wamr['attributed_samples']} | "
        f"{wamr['attribution_coverage_pct']:.2f}% |",
        f"| Wasmtime | {wasmtime['total_samples']} | "
        f"{wasmtime['attributed_samples']} | "
        f"{wasmtime['attribution_coverage_pct']:.2f}% |",
        "",
        "#### Top functions",
        "",
        "| WAMR local | Wasm index | Name | WAMR run share | Wasmtime run share |",
        "|---:|---:|---|---:|---:|",
    ]
    for item in report["matched_functions"]:
        lines.append(
            f"| {item['local_func']} | {item['wasm_function_index']} | "
            f"`{item['name']}` | {item['wamr']['percent_of_run']:.2f}% | "
            f"{item['wasmtime']['percent_of_run']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "#### Hottest sampled instructions",
            "",
            "| Function | Engine | Run share | Instruction |",
            "|---|---|---:|---|",
        ]
    )
    for item in report["matched_functions"]:
        for engine in ("wamr", "wasmtime"):
            for hot in item[engine]["hottest_instructions"][:3]:
                lines.append(
                    f"| `{item['name']}` | {engine.upper()} | "
                    f"{hot['percent_of_run']:.2f}% | "
                    f"`{hot['instruction']}` |"
                )
    lines.extend(
        [
            "",
            "#### Same-function instruction-class differences",
            "",
            "| Function | Class | WAMR run share | Wasmtime run share | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in report["matched_functions"]:
        for name, values in sorted(
            item["class_groups"].items(),
            key=lambda pair: -pair[1]["delta_percentage_points"],
        ):
            lines.append(
                f"| `{item['name']}` | `{name}` | "
                f"{values['wamr_percent_of_run']:.2f}% | "
                f"{values['wasmtime_percent_of_run']:.2f}% | "
                f"{values['delta_percentage_points']:+.2f} pp |"
            )
    if report.get("alu_provenance") is not None:
        lines.extend(
            [
                "",
                "#### Conservative common-universe ALU provenance",
                "",
                "| Function | Category | WAMR samples / run share | "
                "Wasmtime samples / run share | Conservative headroom |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for item in report["matched_functions"]:
            narrow = item["alu_provenance"]
            gate_categories = narrow["gate"].get("categories", {})
            for category in aarch64_instruction_provenance.CATEGORIES:
                wamr_values = narrow["wamr"]["common_gating_universe"][
                    "categories"
                ][category]
                wasmtime_values = narrow["wasmtime"][
                    "common_gating_universe"
                ]["categories"][category]
                gate_value = gate_categories.get(category)
                headroom = (
                    f"{gate_value['conservative_headroom_percentage_points']:+.2f} pp"
                    if gate_value is not None
                    else "not gate-eligible"
                )
                lines.append(
                    f"| `{item['name']}` | `{category}` | "
                    f"{wamr_values['samples']} / "
                    f"{wamr_values['percent_of_run']:.2f}% | "
                    f"{wasmtime_values['samples']} / "
                    f"{wasmtime_values['percent_of_run']:.2f}% | "
                    f"{headroom} |"
                )
            lines.append(
                f"| `{item['name']}` | `proven coverage` | "
                f"{narrow['wamr']['common_gating_universe']['coverage']['proven_percent']:.2f}% "
                "of common universe | "
                f"{narrow['wasmtime']['common_gating_universe']['coverage']['proven_percent']:.2f}% "
                "of common universe | "
                f"gate `{narrow['gate']['status']}` |"
            )
    if report.get("frame_attribution") is not None:
        frame = report["frame_attribution"]
        summary = frame["summary"]
        coverage = summary["coverage"]
        lines.extend(
            [
                "",
                "#### Exact WAMR frame origins",
                "",
                f"`{frame['name']}`: module 0, local_func {frame['local_func']}, "
                f"wasm index {frame['wasm_function_index']}. "
                f"Sidecar `{frame['metadata']['path']}` "
                f"(`sha256:{frame['metadata']['sha256']}`).",
                "",
                f"Proven origins: {coverage['proven_origin_frame_instructions']}/"
                f"{coverage['frame_instructions']} static native frame instructions; "
                f"{coverage['proven_origin_frame_samples']}/"
                f"{coverage['frame_samples']} frame samples. "
                f"Unknown: {coverage['unknown_frame_instructions']} instructions / "
                f"{coverage['unknown_frame_samples']} samples.",
                "",
                "| Origin | Static native instructions | Self samples | Run share |",
                "|---|---:|---:|---:|",
            ]
        )
        for origin, values in sorted(
            summary["origins"].items(), key=lambda pair: -pair[1]["samples"]
        ):
            lines.append(
                f"| `{origin}` | {values['static_instructions']} | "
                f"{values['samples']} | {values['percent_of_run']:.2f}% |"
            )
        lines.extend(
            [
                "",
                "| Allocator slot / frame offset | Vreg(s) | Source | Self samples | Run share |",
                "|---|---|---|---:|---:|",
            ]
        )
        for item in summary["allocator_contributors"][:10]:
            if item.get("paired_components"):
                slot = "paired " + ", ".join(
                    f"{component['slot']} / {component['frame_offset']}"
                    for component in item["paired_components"]
                )
            else:
                slot = f"{item['slot']} / {item['frame_offset']}"
            vreg = (
                f"{item['candidate_vregs']} (ambiguous)"
                if item["vreg_ambiguous"] else str(item["vreg"])
            )
            source = f"{item['defining_opcode'] or '?'}/{item['source_class'] or '?'}"
            lines.append(
                f"| {slot} | {vreg} | `{source}` | {item['samples']} | "
                f"{item['percent_of_run']:.2f}% |"
            )
        reconciliation = summary["reconciliation"]
        lines.extend(
            [
                "",
                "Emitter reconciliation: "
                f"{reconciliation['emitted_allocator_loads']} allocator load "
                f"components / {reconciliation['emitted_allocator_stores']} store "
                "components, matching the selected-function spill metric. "
                "Paired samples count once; static components and IR snapshots "
                "are not multiplied by the capture count. Broad instruction "
                "classes retain their original metadata-free definitions.",
            ]
        )
        components = summary["allocator_component_counts"]
        if components["unranked_loads"] or components["unranked_stores"]:
            lines.append(
                "Mixed-origin allocator components outside the sample ranking: "
                f"{components['unranked_loads']} loads / "
                f"{components['unranked_stores']} stores. Their indivisible "
                "instruction samples remain unknown."
            )
    lines.extend(
        [
            "",
            "#### Spill-metric cross-check",
            "",
            "| Function | spill_ld/st (basis) | Static frame ld/st | "
            "Frame-traffic run share |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in report["matched_functions"]:
        spill = item["spill_metric"]
        frame = item["wamr"]["frame_cross_check"]
        lines.append(
            f"| `{item['name']}` | {spill['spill_ld']}/{spill['spill_st']} "
            f"({item.get('spill_metric_basis', 'pre-emission IR estimate')}) | "
            f"{frame['static_frame_loads']}/{frame['static_frame_stores']} | "
            f"{frame['percent_of_run']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "WAMR `local_func` excludes imports; Wasmtime symbols use the full "
            f"wasm function index. This module has "
            f"{report['wasm']['imported_function_count']} function imports, so "
            "`wasm_index = local_func + import_count` was verified against the "
            "name section for every matched row.",
            "",
            "Caveats: perf self samples only; no DWARF unwinding through WAMR "
            "generated code. Unselected AArch64 spill metrics are pre-emission "
            "estimates; a selected sidecar uses emitter-traced component counts. "
            "Broad frame traffic is a conservative cross-check, not a claim "
            "that every frame access is an allocator spill. `all_alu` means "
            f"{report['classifier_wording']['all_alu']}; its cross-engine "
            "difference is not address/check headroom. Narrow provenance uses "
            "the same complete architecture-only gating universe and CFG/"
            "def-use rules for both engines. Common-universe unknown/mixed, "
            "instruction-unresolved, and globally unattributed reference "
            "samples remain possible upper bounds. Structural address guards "
            "are diagnostic only and do not imply removable engine checks.",
        ]
    )
    return "\n".join(lines) + "\n"


def gzip_if_small(path: Path, max_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size
    if size > max_bytes:
        return {"path": path.name, "size_bytes": size, "retained": False}
    target = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(target, "wb", compresslevel=9) as dest:
        shutil.copyfileobj(source, dest)
    return {
        "path": target.name,
        "source_size_bytes": size,
        "size_bytes": target.stat().st_size,
        "retained": True,
    }


def retain_exact_binary(
    source: Path, target: Path, max_bytes: int
) -> dict[str, Any]:
    size = source.stat().st_size
    source_sha = sha256_file(source)
    if size > max_bytes:
        return {
            "path": target.name,
            "source_size_bytes": size,
            "source_sha256": source_sha,
            "retained": False,
            "reason": f"source exceeds retention limit {max_bytes}",
        }
    with source.open("rb") as input_stream, gzip.open(
        target, "wb", compresslevel=9
    ) as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
    return {
        "path": target.name,
        "source_size_bytes": size,
        "source_sha256": source_sha,
        "size_bytes": target.stat().st_size,
        "retained": True,
    }


def compile_wamr_diagnostics(
    *,
    recorder: CommandRecorder,
    wamrc: Path,
    fixture: Path,
    cwasm: Path,
    out_dir: Path,
    build_repo: Path,
    helper: Path,
    frame_func: int | None,
) -> tuple[dict[int, dict[str, Any]], Path | None]:
    diagnostic_cwasm = out_dir / "coremark.diagnostic.cwasm"
    compile_env = os.environ.copy()
    for key in (
        "WAMR_AOT_FRAME_ATTRIBUTION",
        "WAMR_AOT_FRAME_ATTRIBUTION_MODULE",
        "WAMR_AOT_FRAME_ATTRIBUTION_FUNC",
    ):
        compile_env.pop(key, None)
    options = {
        "WAMR_AOT_SPILL_METRIC": "1",
        "WAMR_AOT_SPILL_METRIC_MIN_SPILLS": "1",
        "WAMR_AOT_CODEGEN_TIMING": "1",
        "WAMR_AOT_CODEGEN_TIMING_THRESHOLD_MS": "0",
    }
    metadata = None
    if frame_func is not None:
        if type(frame_func) is not int or frame_func < 0:
            raise ProfileError("frame function must be a nonnegative local index")
        prefix = out_dir / "wamr-frame"
        metadata = out_dir / f"wamr-frame.mod0.func{frame_func}.json"
        if metadata.exists():
            raise ProfileError(f"refusing to reuse existing frame metadata: {metadata}")
        options.update(
            {
                "WAMR_AOT_FRAME_ATTRIBUTION": str(prefix),
                "WAMR_AOT_FRAME_ATTRIBUTION_MODULE": "0",
                "WAMR_AOT_FRAME_ATTRIBUTION_FUNC": str(frame_func),
            }
        )
    compile_env.update(options)
    command = [
        str(wamrc), "compile", str(fixture), "-o", str(diagnostic_cwasm)
    ]
    result = recorder.run(
        command,
        "wamr-compile.log",
        cwd=build_repo,
        env=compile_env,
        display=shlex.join(
            ["env", *(f"{key}={value}" for key, value in options.items()), *command]
        ),
    )
    if sha256_file(diagnostic_cwasm) != sha256_file(cwasm):
        raise ProfileError(
            "diagnostic cwasm does not match the exact benchmark-built module"
        )
    diagnostic_cwasm.unlink()
    if metadata is not None:
        if not metadata.is_file():
            raise ProfileError(
                "the retained benchmark compiler did not emit requested frame "
                "metadata; benchmark a candidate with AArch64 frame attribution"
            )
        recorder.run(
            [
                sys.executable, str(helper),
                "--cwasm", str(cwasm),
                "--arch", "aarch64",
                "--func", str(frame_func),
                "--frame-metadata", str(metadata),
                "--validate-frame-metadata",
            ],
            "wamr-frame-validation.log",
            cwd=build_repo,
        )
    return parse_spill_metrics(result.stdout + result.stderr), metadata


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = CommandRecorder(out_dir)
    aot = load_aot_helper(repo)
    benchmark_report, benchmark_report_sha = load_benchmark_report(
        args.benchmark_report.resolve()
    )
    benchmark_wamr = next(
        engine
        for engine in benchmark_report["engines"]
        if engine["role"] == args.benchmark_role
    )
    benchmark_wamr_sha = benchmark_wamr["identity"]["source"]["sha"]
    current_execution = bench_coremark.capture_execution_identity(
        args.execution_id
    )
    try:
        bench_coremark.validate_execution_match(
            benchmark_report["provenance"]["execution"], current_execution
        )
        handoff = bench_coremark.load_wamr_artifact_handoff(
            args.benchmark_artifacts,
            benchmark_wamr["identity"],
            role=args.benchmark_role,
        )
    except RuntimeError as exc:
        raise ProfileError(str(exc)) from exc
    handoff_manifest_sha = sha256_file(
        args.benchmark_artifacts.resolve() / "manifest.json"
    )

    host_identity = bench_coremark.validate_native_host("aarch64")
    fixture, fixture_sha = bench_coremark.resolve_fixture(
        repo, bench_coremark.DEFAULT_FIXTURE
    )
    wasm_identity = compare_hot_function.parse_core_wasm(fixture)
    if args.frame_func is not None and not (
        0 <= args.frame_func < wasm_identity.local_function_count
    ):
        raise ProfileError("frame function is outside the benchmark module")
    wamr_ref = args.wamr_ref or benchmark_wamr_sha
    commit = recorder.run(
        ["git", "rev-parse", wamr_ref], "git-identity.log", cwd=repo
    ).stdout.strip()
    if commit != benchmark_wamr_sha:
        raise ProfileError(
            f"--wamr-ref resolves to {commit}, but benchmark role "
            f"{args.benchmark_role} is {benchmark_wamr_sha}"
        )
    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    analysis_sources = capture_analysis_sources(repo, aot, checkout_commit)
    build_repo = args.benchmark_artifacts.resolve()
    kernel = recorder.run(
        ["uname", "-r"], "kernel.log", cwd=repo
    ).stdout.strip()
    uname = recorder.run(
        ["uname", "-a"], "uname.log", cwd=repo
    ).stdout.strip()
    perf_version = recorder.run(
        [perf_binary(), "--version"], "perf-version.log", cwd=repo
    ).stdout.strip()

    wamr = handoff["runtime"]
    wamrc = handoff["compiler"]
    cwasm = handoff["module"]
    wamr_version = recorder.run(
        [str(wamr), "version"], "wamr-version.log", cwd=build_repo
    ).stdout.strip()
    wamrc_version = recorder.run(
        [str(wamrc), "version"], "wamrc-version.log", cwd=build_repo
    ).stdout.strip()

    helper = repo / ".github/skills/aot-perf-profile/aot_jit_attr.py"
    spill_metrics, frame_metadata = compile_wamr_diagnostics(
        recorder=recorder,
        wamrc=wamrc,
        fixture=fixture,
        cwasm=cwasm,
        out_dir=out_dir,
        build_repo=build_repo,
        helper=helper,
        frame_func=args.frame_func,
    )
    frame_metadata_sha = (
        sha256_file(frame_metadata) if frame_metadata is not None else None
    )
    cwasm_info = aot.parse_cwasm(cwasm)
    if cwasm_info.version not in aot.SUPPORTED_AOT_VERSIONS:
        raise ProfileError("WAMR cwasm version is unsupported by attribution")
    profile_wamr_identity = bench_coremark.make_wamr_identity(
        ref=wamr_ref,
        source_sha=commit,
        optimize="ReleaseFast",
        runtime_path=wamr,
        compiler_path=wamrc,
        module_path=cwasm,
    )
    retained_wamr_binary = retain_exact_binary(
        cwasm,
        out_dir / "wamr-profiled.cwasm.gz",
        args.max_perf_bytes,
    )
    if (
        retained_wamr_binary["source_sha256"]
        != profile_wamr_identity["module"]["sha256"]
    ):
        raise ProfileError("retained WAMR cwasm identity changed during handoff")

    wasmtime = bench_coremark.install_pinned_wasmtime(
        repo, args.wasmtime_cache
    )
    wasmtime_version = bench_coremark.validate_pinned_wasmtime(wasmtime)
    profile_wasmtime_identity = bench_coremark.make_wasmtime_identity(
        version=wasmtime_version,
        runtime_path=wasmtime,
        channel="historical-pin",
    )
    wasmtime_cwasm = out_dir / "coremark.wasmtime.cwasm"
    recorder.run(
        [
            str(wasmtime),
            "compile",
            "-O",
            "opt-level=2",
            str(fixture),
            "-o",
            str(wasmtime_cwasm),
        ],
        "wasmtime-compile.log",
        cwd=out_dir,
    )
    objdump = recorder.run(
        [
            str(wasmtime),
            "objdump",
            str(wasmtime_cwasm),
            "--addresses",
            "--bytes",
        ],
        "wasmtime-objdump-command.log",
        cwd=out_dir,
    ).stdout
    (out_dir / "wasmtime-objdump.txt").write_text(objdump, encoding="utf-8")

    affinity = bench_coremark.select_cpu_affinity()
    try:
        benchmark_identity = bench_coremark.validate_benchmark_profile_match(
            benchmark_report,
            expected_arch="aarch64",
            fixture_sha=fixture_sha,
            host=host_identity,
            affinity=affinity,
            wamr_source_sha=commit,
            wamr_optimize="ReleaseFast",
            wamr_runtime_sha=profile_wamr_identity["runtime"]["sha256"],
            wamr_compiler_sha=profile_wamr_identity["compiler"]["sha256"],
            wamr_module_sha=profile_wamr_identity["module"]["sha256"],
            wasmtime_version_value=wasmtime_version,
            wasmtime_runtime_sha=profile_wasmtime_identity["runtime"]["sha256"],
            producer_source_sha=checkout_commit,
            producer_script_sha=sha256_file(
                Path(bench_coremark.__file__).resolve()
            ),
            current_execution=current_execution,
            benchmark_role=args.benchmark_role,
        )
    except RuntimeError as exc:
        raise ProfileError(str(exc)) from exc
    guest_args = bench_coremark.coremark_guest_args(
        bench_coremark.EXPECTED_ITERATIONS
    )
    wamr_command = bench_coremark.apply_affinity(
        [str(wamr), "run", str(cwasm), *guest_args], affinity
    )
    wasmtime_command = bench_coremark.apply_affinity(
        [
            str(wasmtime),
            "run",
            "--allow-precompiled",
            str(wasmtime_cwasm),
            *guest_args,
        ],
        affinity,
    )
    wasmtime_profile_command = bench_coremark.apply_affinity(
        [
            str(wasmtime),
            "run",
            "--allow-precompiled",
            "--profile=jitdump",
            str(wasmtime_cwasm),
            *guest_args,
        ],
        affinity,
    )

    profile_schedule = []
    warmup_ordinals = Counter()
    schedule_position = 0
    for engine in bench_coremark.counterbalanced_order(
        ["wamr", "wasmtime"], 2
    ):
        schedule_position += 1
        warmup_ordinals[engine] += 1
        command = wamr_command if engine == "wamr" else wasmtime_command
        cwd = build_repo if engine == "wamr" else out_dir
        started_at = datetime.now(timezone.utc).isoformat()
        result = recorder.run(
            command,
            f"{engine}-warmup-{warmup_ordinals[engine]}.log",
            cwd=cwd,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        value = parse_validated_coremark(
            result.stdout + result.stderr, engine, 1
        )[0]
        profile_schedule.append(
            {
                "position": schedule_position,
                "phase": "warmup",
                "engine": engine,
                "engine_ordinal": warmup_ordinals[engine],
                "started_at": started_at,
                "completed_at": completed_at,
                "iterations": bench_coremark.EXPECTED_ITERATIONS,
                "iterations_per_second": value,
            }
        )

    wamr_perfs = []
    wasmtime_perfs = []
    jitdumps = []
    wamr_values = []
    wasmtime_values = []
    wasmtime_captures = []
    profile_ordinals = Counter()
    for engine in bench_coremark.counterbalanced_order(
        ["wamr", "wasmtime"], PROFILE_CAPTURES_PER_ENGINE
    ):
        schedule_position += 1
        profile_ordinals[engine] += 1
        ordinal = profile_ordinals[engine]
        started_at = datetime.now(timezone.utc).isoformat()
        if engine == "wamr":
            perf_path = out_dir / f"wamr-{ordinal}.perf.data"
            result = recorder.run(
                [
                    perf_binary(),
                    "record",
                    "-k",
                    "mono",
                    "-F",
                    str(args.frequency),
                    "-e",
                    "cycles:u",
                    "-o",
                    str(perf_path),
                    "--",
                    *wamr_command,
                ],
                f"wamr-run-{ordinal}.log",
                cwd=build_repo,
            )
            value = parse_validated_coremark(
                result.stdout + result.stderr, "WAMR", 1
            )[0]
            wamr_values.append(value)
            wamr_perfs.append(perf_path)
        else:
            perf_path = out_dir / f"wasmtime-{ordinal}.perf.data"
            before = set(out_dir.glob("jit-*.dump"))
            result = recorder.run(
                [
                    perf_binary(),
                    "record",
                    "-k",
                    "mono",
                    "-F",
                    str(args.frequency),
                    "-e",
                    "cycles:u",
                    "-o",
                    str(perf_path),
                    "--",
                    *wasmtime_profile_command,
                ],
                f"wasmtime-run-{ordinal}.log",
                cwd=out_dir,
            )
            value = parse_validated_coremark(
                result.stdout + result.stderr, "Wasmtime", 1
            )[0]
            wasmtime_values.append(value)
            wasmtime_perfs.append(perf_path)
            new_dumps = sorted(set(out_dir.glob("jit-*.dump")) - before)
            if len(new_dumps) != 1:
                raise ProfileError(
                    f"Wasmtime capture {ordinal} produced {len(new_dumps)} "
                    "new jitdump files; expected exactly one"
                )
            jitdumps.extend(new_dumps)
            injected = out_dir / f"wasmtime-{ordinal}.perf.jit.data"
            recorder.run(
                [
                    perf_binary(),
                    "inject",
                    "--jit",
                    "-i",
                    str(perf_path),
                    "-o",
                    str(injected),
                ],
                f"wasmtime-perf-inject-{ordinal}.log",
                cwd=out_dir,
            )
            perf_script = recorder.run(
                [
                    perf_binary(),
                    "script",
                    "-i",
                    str(injected),
                    "-F",
                    "ip,sym,symoff,dso",
                ],
                f"wasmtime-perf-script-{ordinal}.log",
                cwd=out_dir,
            ).stdout
            (out_dir / f"wasmtime-samples-{ordinal}.txt").write_text(
                perf_script, encoding="utf-8"
            )
            wasmtime_captures.append(
                parse_wasmtime_samples(perf_script, wasm_identity)
            )
            validate_wasmtime_mapping(wasmtime_captures[-1], wasm_identity)
            validate_wasmtime_capture(
                wasmtime_captures[-1],
                minimum_samples=args.min_samples,
            )
        completed_at = datetime.now(timezone.utc).isoformat()
        profile_schedule.append(
            {
                "position": schedule_position,
                "phase": "profile",
                "engine": engine,
                "engine_ordinal": ordinal,
                "started_at": started_at,
                "completed_at": completed_at,
                "iterations": bench_coremark.EXPECTED_ITERATIONS,
                "iterations_per_second": value,
                "perf_file": perf_path.name,
            }
        )

    ranking_reports = []
    wamr_capture_validations = []
    for ordinal, wamr_perf in enumerate(wamr_perfs, 1):
        ranking_json = out_dir / f"wamr-attribution-{ordinal}.json"
        recorder.run(
            [
                sys.executable,
                str(helper),
                "--perf",
                str(wamr_perf),
                "--cwasm",
                str(cwasm),
                "--arch",
                "aarch64",
                "--top",
                str(len(cwasm_info.func_offsets)),
                "--min-samples",
                str(args.min_samples),
                "--min-attribution-pct",
                str(MIN_ATTRIBUTION_COVERAGE_PCT),
                "--authoritative",
                "--json-out",
                str(ranking_json),
            ],
            f"wamr-attribution-{ordinal}.log",
            cwd=repo,
        )
        ranking_report = json.loads(ranking_json.read_text(encoding="utf-8"))
        wamr_capture_validations.append(
            validate_wamr_capture(
                ranking_report,
                minimum_samples=args.min_samples,
            )
        )
        ranking_reports.append(ranking_report)
    wamr_attribution = aggregate_wamr_rankings(ranking_reports)
    top_functions = wamr_attribution["top_functions"][: args.classify]
    if len(top_functions) != args.classify:
        raise ProfileError(
            f"WAMR attribution produced only {len(top_functions)} hot functions"
        )
    classified_wamr = {}
    for item in top_functions:
        local_func = item["local_func"]
        function_reports = []
        for ordinal, wamr_perf in enumerate(wamr_perfs, 1):
            path = out_dir / f"wamr-func-{local_func}-{ordinal}.json"
            recorder.run(
                [
                    sys.executable,
                    str(helper),
                    "--perf",
                    str(wamr_perf),
                    "--cwasm",
                    str(cwasm),
                    "--arch",
                    "aarch64",
                    "--func",
                    str(local_func),
                    "--top",
                    str(args.top),
                    "--min-samples",
                    str(args.min_samples),
                    "--min-attribution-pct",
                    str(MIN_ATTRIBUTION_COVERAGE_PCT),
                    "--authoritative",
                    "--json-out",
                    str(path),
                ],
                f"wamr-func-{local_func}-{ordinal}.log",
                cwd=repo,
            )
            function_reports.append(
                json.loads(path.read_text(encoding="utf-8"))
            )
        function_start, _ = aot.function_bounds(cwasm_info, local_func)
        classified_wamr[local_func] = aggregate_wamr_function(
            function_reports,
            total_samples=wamr_attribution["total_samples"],
            function_start=function_start,
        )

    wamr_capture_counts = []
    for wamr_perf, ranking_report in zip(wamr_perfs, ranking_reports):
        counts, total = aot.addr_counts(str(wamr_perf))
        if total != ranking_report["total_samples"]:
            raise ProfileError(
                "WAMR perf sample totals changed while building ALU provenance"
            )
        wamr_capture_counts.append((counts, ranking_report["text_base"]))

    frame_attribution = None
    if frame_metadata is not None:
        frame_attribution = analyze_wamr_frame_provenance(
            aot=aot,
            cwasm_info=cwasm_info,
            metadata_path=frame_metadata,
            local_func=args.frame_func,
            capture_counts=wamr_capture_counts,
            ranking_reports=ranking_reports,
            total_samples=wamr_attribution["total_samples"],
            spill_metric=spill_metrics.get(args.frame_func, {}),
            scratch_dir=out_dir,
        )
        frame_attribution.update(
            {
                "wasm_function_index": (
                    args.frame_func + wasm_identity.imported_function_count
                ),
                "name": wasm_identity.function_names.get(
                    args.frame_func + wasm_identity.imported_function_count
                ),
                "compiler_sha256": profile_wamr_identity["compiler"]["sha256"],
                "cwasm_sha256": profile_wamr_identity["module"]["sha256"],
            }
        )
        if frame_attribution["metadata"]["sha256"] != frame_metadata_sha:
            raise ProfileError("frame metadata changed during capture")

    parsed_wasmtime = aggregate_wasmtime_samples(wasmtime_captures)
    validate_wasmtime_mapping(parsed_wasmtime, wasm_identity)
    if parsed_wasmtime["total_samples"] < args.min_samples:
        raise ProfileError(
            f"Wasmtime perf data has only {parsed_wasmtime['total_samples']} samples"
        )
    wasmtime_attributed = sum(
        item["samples"] for item in parsed_wasmtime["functions"].values()
    )

    matched = []
    for top in top_functions:
        local_func = top["local_func"]
        wasm_index = local_func + wasm_identity.imported_function_count
        name = wasm_identity.function_names.get(wasm_index) or f"func_{wasm_index}"
        wasmtime_entry = parsed_wasmtime["functions"].get(wasm_index)
        if not wasmtime_entry or wasmtime_entry["samples"] <= 0:
            raise ProfileError(
                f"Wasmtime has no samples for WAMR local_func={local_func}, "
                f"wasm function {wasm_index} ({name})"
            )
        if sum(wasmtime_entry["offsets"].values()) != wasmtime_entry["samples"]:
            raise ProfileError(
                f"Wasmtime function {wasm_index} has "
                f"{wasmtime_entry['samples']} symbol samples but only "
                f"{sum(wasmtime_entry['offsets'].values())} symbol offsets"
            )
        wamr_function = classified_wamr[local_func]
        wasmtime_instruction = classify_wasmtime_function(
            aot=aot,
            objdump_text=objdump,
            wasm_index=wasm_index,
            offsets=wasmtime_entry["offsets"],
            total_samples=parsed_wasmtime["total_samples"],
        )
        spill = spill_metrics.get(
            local_func,
            {"spill_ld": 0, "spill_st": 0, "spilled_vregs": 0, "slots": 0},
        )
        wamr_classes = wamr_function["classes"]
        wasmtime_classes = wasmtime_instruction["classes"]
        class_groups = {}
        for group, names in CLASS_GROUPS.items():
            wamr_samples = class_samples(wamr_classes, names)
            wasmtime_samples = class_samples(wasmtime_classes, names)
            wamr_pct = 100.0 * wamr_samples / wamr_attribution["total_samples"]
            wasmtime_pct = (
                100.0 * wasmtime_samples / parsed_wasmtime["total_samples"]
            )
            class_groups[group] = {
                "wamr_samples": wamr_samples,
                "wasmtime_samples": wasmtime_samples,
                "wamr_percent_of_run": wamr_pct,
                "wasmtime_percent_of_run": wasmtime_pct,
                "delta_percentage_points": wamr_pct - wasmtime_pct,
            }
        static_frame_loads = sum(
            values.get("static_instructions", 0)
            for key, values in wamr_classes.items()
            if key in {"frame_load_unattributed", "unknown_frame_load"}
        )
        static_frame_stores = sum(
            values.get("static_instructions", 0)
            for key, values in wamr_classes.items()
            if key in {"frame_store_unattributed", "unknown_frame_store"}
        )
        frame_samples = class_groups["frame_traffic"]["wamr_samples"]
        wamr_alu_provenance = analyze_wamr_alu_provenance(
            aot=aot,
            cwasm_info=cwasm_info,
            local_func=local_func,
            capture_counts=wamr_capture_counts,
            total_samples=wamr_attribution["total_samples"],
            scratch_dir=out_dir,
            global_attributed_samples=wamr_attribution[
                "attributed_samples"
            ],
        )
        wasmtime_alu_provenance = analyze_wasmtime_alu_provenance(
            aot=aot,
            objdump_text=objdump,
            wasm_index=wasm_index,
            offsets=wasmtime_entry["offsets"],
            total_samples=parsed_wasmtime["total_samples"],
            global_attributed_samples=wasmtime_attributed,
        )
        narrow_alu = assemble_alu_provenance(
            wamr=wamr_alu_provenance,
            wasmtime=wasmtime_alu_provenance,
            expected_wamr_all_alu_samples=class_groups["all_alu"][
                "wamr_samples"
            ],
            expected_wasmtime_all_alu_samples=class_groups["all_alu"][
                "wasmtime_samples"
            ],
        )
        matched.append(
            {
                "local_func": local_func,
                "wasm_function_index": wasm_index,
                "name": name,
                "spill_metric": {
                    key: spill.get(key, 0)
                    for key in (
                        "spill_ld",
                        "spill_st",
                        "spilled_vregs",
                        "slots",
                    )
                },
                "spill_metric_basis": (
                    "emitter-traced"
                    if local_func == args.frame_func
                    else "pre-emission IR estimate"
                ),
                "wamr": {
                    "samples": wamr_function["samples"],
                    "percent_of_run": wamr_function["percent_of_run"],
                    "classes": wamr_classes,
                    "hottest_instructions": wamr_function[
                        "hottest_instructions"
                    ],
                    "frame_cross_check": {
                        "static_frame_loads": static_frame_loads,
                        "static_frame_stores": static_frame_stores,
                        "samples": frame_samples,
                        "percent_of_run": (
                            100.0
                            * frame_samples
                            / wamr_attribution["total_samples"]
                        ),
                    },
                },
                "wasmtime": {
                    "samples": wasmtime_entry["samples"],
                    "percent_of_run": (
                        100.0
                        * wasmtime_entry["samples"]
                        / parsed_wasmtime["total_samples"]
                    ),
                    **wasmtime_instruction,
                },
                "class_groups": class_groups,
                "alu_provenance": narrow_alu,
            }
        )

    perf_artifacts = [
        gzip_if_small(path, args.max_perf_bytes)
        for path in [*wamr_perfs, *wasmtime_perfs, *jitdumps]
    ]
    target_alu = next(
        (
            item["alu_provenance"]
            for item in matched
            if item["local_func"] == 3
            and item["wasm_function_index"]
            == 3 + wasm_identity.imported_function_count
        ),
        None,
    )
    validate_analysis_sources_unchanged(repo, analysis_sources)
    analysis_sources = capture_analysis_sources(repo, aot, checkout_commit)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "architecture": "aarch64",
        "benchmark": {
            **benchmark_identity,
            "report_path": str(args.benchmark_report.resolve()),
            "report_sha256": benchmark_report_sha,
            "schema_version": benchmark_report["schema_version"],
            "kind": benchmark_report["kind"],
            "artifact_handoff": {
                "directory": str(args.benchmark_artifacts.resolve()),
                "manifest_sha256": handoff_manifest_sha,
            },
        },
        "provenance": {
            "producer_source_sha": checkout_commit,
            "script_path": str(Path(__file__).resolve().relative_to(repo)),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "analysis_module": {
                "path": str(
                    Path(aarch64_instruction_provenance.__file__)
                    .resolve()
                    .relative_to(repo)
                ),
                "sha256": sha256_file(
                    Path(aarch64_instruction_provenance.__file__).resolve()
                ),
            },
            "analysis_sources": analysis_sources,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "execution": current_execution,
        },
        "guest_args": list(guest_args),
        "expected_iterations": bench_coremark.EXPECTED_ITERATIONS,
        "affinity": {
            "allowed_cpus": list(affinity.allowed_cpus),
            "selected_cpu": affinity.selected_cpu,
            "taskset": affinity.taskset,
            "verified": True,
        },
        "profile_schedule": profile_schedule,
        "classifier_wording": {
            "all_alu": ALL_ALU_WORDING,
            "narrow_alu_provenance": NARROW_ALU_WORDING,
        },
        "minimum_attribution_coverage_pct": MIN_ATTRIBUTION_COVERAGE_PCT,
        "wamr_captures": wamr_capture_validations,
        "wasmtime_captures": [
            validate_wasmtime_capture(
                capture,
                minimum_samples=args.min_samples,
            )
            for capture in wasmtime_captures
        ],
        "fixture": {"path": str(fixture), "sha256": fixture_sha},
        "wasm": {
            "imported_function_count": wasm_identity.imported_function_count,
            "local_function_count": wasm_identity.local_function_count,
        },
        "host": {
            "architecture": host_identity.arch,
            "cpu_count": host_identity.cpu_count,
            "cpu_model": host_identity.cpu_model,
            "runner_name": host_identity.runner_name,
            "fingerprint": host_identity.fingerprint(),
            "kernel": kernel,
            "uname": uname,
        },
        "perf": {
            "version": perf_version,
            "package": os.environ.get("COREMARK_PERF_PACKAGE", "unknown"),
            "package_version": os.environ.get(
                "COREMARK_PERF_PACKAGE_VERSION", "unknown"
            ),
            "paranoid_initial": os.environ.get(
                "COREMARK_PERF_PARANOID_INITIAL", "unknown"
            ),
            "paranoid_effective": Path(
                "/proc/sys/kernel/perf_event_paranoid"
            ).read_text().strip(),
            "event": "cycles:u",
            "frequency": args.frequency,
            "sampling_permitted": True,
        },
        "wamr": {
            "benchmark_role": args.benchmark_role,
            "commit": commit,
            "version": wamr_version,
            "wamrc_version": wamrc_version,
            "cwasm_aot_version": cwasm_info.version,
            "runtime_sha256": profile_wamr_identity["runtime"]["sha256"],
            "compiler_sha256": profile_wamr_identity["compiler"]["sha256"],
            "cwasm_sha256": profile_wamr_identity["module"]["sha256"],
        },
        "wasmtime": {
            "version": wasmtime_version,
            "sha256": profile_wasmtime_identity["runtime"]["sha256"],
            "profile_strategy": "jitdump",
            "function_mapping": (
                "Wasmtime v44 perf inject emitted name-only symbols in "
                "jitted-<pid>-<defined-func-index>.so. Each symbol name was "
                "resolved uniquely through the wasm name section and the DSO "
                "suffix was required to equal wasm_index - import_count."
            ),
            "cwasm_sha256": sha256_file(wasmtime_cwasm),
        },
        "engines": {
            "wamr": {
                "total_samples": wamr_attribution["total_samples"],
                "attributed_samples": wamr_attribution["attributed_samples"],
                "attribution_coverage_pct": wamr_attribution[
                    "attribution_coverage_pct"
                ],
                "iterations_per_second_samples": wamr_values,
                "iterations_per_second": statistics.fmean(wamr_values),
                "wall_seconds": sum(
                    entry["elapsed_seconds"]
                    for entry in recorder.commands
                    if entry["log"].startswith("wamr-run-")
                ),
                "top_functions": wamr_attribution["top_functions"],
            },
            "wasmtime": {
                "total_samples": parsed_wasmtime["total_samples"],
                "attributed_samples": wasmtime_attributed,
                "attribution_coverage_pct": (
                    100.0
                    * wasmtime_attributed
                    / parsed_wasmtime["total_samples"]
                ),
                "iterations_per_second_samples": wasmtime_values,
                "iterations_per_second": statistics.fmean(wasmtime_values),
                "wall_seconds": sum(
                    entry["elapsed_seconds"]
                    for entry in recorder.commands
                    if entry["log"].startswith("wasmtime-run-")
                ),
                "top_functions": [
                    {
                        "wasm_function_index": index,
                        "name": (
                            next(iter(entry["names"]))
                            if entry["names"]
                            else wasm_identity.function_names.get(index)
                        ),
                        "samples": entry["samples"],
                        "percent_of_run": (
                            100.0
                            * entry["samples"]
                            / parsed_wasmtime["total_samples"]
                        ),
                    }
                    for index, entry in sorted(
                        parsed_wasmtime["functions"].items(),
                        key=lambda pair: -pair[1]["samples"],
                    )[: args.top]
                ],
            },
        },
        "matched_functions": matched,
        "alu_provenance": {
            "schema_version": aarch64_instruction_provenance.SCHEMA_VERSION,
            "kind": aarch64_instruction_provenance.ANALYSIS_KIND,
            "target_local_func": 3,
            "target_wasm_function_index": (
                3 + wasm_identity.imported_function_count
            ),
            "gate": (
                target_alu["gate"]
                if target_alu is not None
                else {
                    "status": "blocked",
                    "optimization_authorized": False,
                    "reason": (
                        "core_bench_list was not among the classified "
                        "functions"
                    ),
                }
            ),
        },
        "commands": recorder.commands,
        "retained_perf_artifacts": perf_artifacts,
        "retained_analysis_artifacts": {
            "wamr_cwasm": retained_wamr_binary,
        },
        "caveats": [
            "Self samples only; WAMR generated code has no unwind CFI.",
            "Unselected AArch64 spill metrics are pre-emission estimates; "
            "selected frame sidecars reconcile emitted component counts.",
            "Wasmtime wasm symbols use full function indices including imports.",
            "Two captures per engine were collected in ABBA order after ABBA warmups.",
            ALL_ALU_WORDING
            + "; the all-ALU differential is not address/check headroom.",
            "Narrow ALU provenance uses an identical complete architecture-only "
            "gating universe in both engines; common unknown/mixed, instruction-"
            "unresolved, and globally unattributed reference samples cannot "
            "clear the optimization gate. Structural guards are diagnostic only.",
        ],
    }
    bench_coremark.validate_same_host(host_identity)
    validate_analysis_sources_unchanged(repo, analysis_sources)
    if frame_attribution is not None:
        if sha256_file(frame_metadata) != frame_metadata_sha:
            raise ProfileError("frame metadata changed during capture")
        report["frame_attribution"] = frame_attribution
    validate_report(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        required=True,
        help=(
            "schema-version-2 authoritative bench_coremark.py JSON report "
            "from this host/run"
        ),
    )
    parser.add_argument(
        "--wamr-ref",
        default=None,
        help="exact ref to profile (default: selected benchmark role SHA)",
    )
    parser.add_argument(
        "--benchmark-role",
        choices=("wamr-baseline", "wamr-target"),
        default="wamr-target",
        help="WAMR role from the paired benchmark to profile (default: target)",
    )
    parser.add_argument(
        "--benchmark-artifacts",
        type=Path,
        required=True,
        help="exact role-bound wamr/wamrc/cwasm handoff from bench_coremark.py",
    )
    parser.add_argument(
        "--execution-id",
        default=None,
        help=(
            "required shared local execution identity; GitHub Actions uses "
            "the workflow run identity"
        ),
    )
    parser.add_argument(
        "--wasmtime-cache",
        type=Path,
        default=None,
        help="cache containing the checksum-verified Wasmtime 44.0.1 binary",
    )
    parser.add_argument("--frequency", type=int, default=999)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_FUNCTIONS)
    parser.add_argument(
        "--classify", type=int, default=DEFAULT_CLASSIFY_FUNCTIONS
    )
    parser.add_argument(
        "--frame-func",
        type=int,
        default=None,
        help=(
            "opt-in exact frame origins for one module-0 local function "
            "(core_state_transition is 10); requires a capable benchmark compiler"
        ),
    )
    parser.add_argument(
        "--max-perf-bytes", type=int, default=DEFAULT_MAX_PERF_BYTES
    )
    args = parser.parse_args()
    if args.frequency <= 0 or args.min_samples <= 0:
        parser.error("--frequency and --min-samples must be positive")
    if args.top <= 0 or args.classify <= 0 or args.classify > args.top:
        parser.error("--classify must be positive and no greater than --top")
    if args.frame_func is not None and args.frame_func < 0:
        parser.error("--frame-func must be nonnegative")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = run_profile(args)
    except Exception as exc:
        retained = []
        for pattern in ("*.perf.data", "jit-*.dump"):
            for path in args.out_dir.glob(pattern):
                try:
                    retained.append(gzip_if_small(path, args.max_perf_bytes))
                except OSError:
                    pass
        failure = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "status": "failed",
            "error": str(exc),
            "retained_perf_artifacts": retained,
        }
        (args.out_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.out_dir / "profile.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (args.out_dir / "profile.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
