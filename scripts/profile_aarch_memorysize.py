#!/usr/bin/env python3
"""Attribute AArch64 CoreMark samples to VmCtx.memory_size and frame traffic."""

from __future__ import annotations

import argparse
import bisect
import json
import re
import struct
import subprocess
from collections import Counter
from pathlib import Path


AOT_MAGIC = 0x746F6100
AOT_VERSION = 8
SEC_TEXT = 2
SEC_FUNCTION = 3


def run(argv: list[str]) -> str:
    proc = subprocess.run(argv, check=False, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(
            f"{' '.join(argv)} failed ({proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def parse_cwasm(path: Path) -> tuple[bytes, int, list[int]]:
    data = path.read_bytes()
    if len(data) < 8:
        raise RuntimeError("truncated cwasm header")
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != AOT_MAGIC or version != AOT_VERSION:
        raise RuntimeError(f"unsupported cwasm header: magic={magic:#x} version={version}")

    text = None
    offsets = None
    pos = 8
    while pos < len(data):
        section_type, size = struct.unpack_from("<II", data, pos)
        pos += 8
        end = pos + size
        if end > len(data):
            raise RuntimeError(f"section {section_type} overruns cwasm")
        if section_type == SEC_TEXT:
            text = data[pos:end]
        elif section_type == SEC_FUNCTION:
            count = struct.unpack_from("<I", data, pos)[0]
            fields = struct.unpack_from(f"<{count * 2}I", data, pos + 4)
            offsets = list(fields[0::2])
        pos = end
    if text is None or offsets is None:
        raise RuntimeError("cwasm lacks text/function section")
    return text, version, offsets


def text_base(perf: Path, text_size: int) -> int:
    output = run(["perf", "script", "-i", str(perf), "--show-mmap-events"])
    pattern = re.compile(
        r"\[(0x[0-9a-fA-F]+)\((0x[0-9a-fA-F]+)\).*?\]: r[w-]xp //anon"
    )
    maps = {
        (int(match.group(1), 16), int(match.group(2), 16))
        for line in output.splitlines()
        if (match := pattern.search(line))
    }
    matches = [(base, size) for base, size in maps if abs(size - text_size) <= 0x2000]
    if len(matches) != 1:
        rendered = ", ".join(f"{base:#x}/{size:#x}" for base, size in sorted(matches))
        raise RuntimeError(
            f"expected one executable mmap near text size {text_size:#x}; "
            f"found {len(matches)}: {rendered}"
        )
    return matches[0][0]


def sample_counts(perf: Path) -> tuple[dict[int, int], int]:
    output = run(
        [
            "perf",
            "report",
            "-i",
            str(perf),
            "--stdio",
            "-g",
            "none",
            "-n",
            "--sort=dso,symbol",
            "--percent-limit",
            "0",
        ]
    )
    row = re.compile(r"^\s*[\d.]+%\s+[\d.]+%\s+(\d+)\s+(.*)$")
    counts: dict[int, int] = {}
    total = 0
    for line in output.splitlines():
        match = row.match(line)
        if not match:
            continue
        count = int(match.group(1))
        total += count
        if "[JIT]" not in match.group(2):
            continue
        symbol = re.search(r"\[[.]\]\s+(0x[0-9a-fA-F]+)", match.group(2))
        if symbol:
            ip = int(symbol.group(1), 16)
            counts[ip] = counts.get(ip, 0) + count
    if total == 0:
        raise RuntimeError("perf report contained no self samples")
    return counts, total


def disassemble(text: bytes, base: int, out_dir: Path) -> list[dict]:
    blob = out_dir / "coremark-text.bin"
    blob.write_bytes(text)
    commands = [
        [
            "objdump",
            "-D",
            "-b",
            "binary",
            "-m",
            "aarch64",
            f"--adjust-vma=0x{base:x}",
            str(blob),
        ],
        [
            "aarch64-linux-gnu-objdump",
            "-D",
            "-b",
            "binary",
            "-m",
            "aarch64",
            f"--adjust-vma=0x{base:x}",
            str(blob),
        ],
    ]
    output = None
    failures = []
    for command in commands:
        try:
            output = run(command)
            break
        except (FileNotFoundError, RuntimeError) as exc:
            failures.append(str(exc))
    if output is None:
        raise RuntimeError("AArch64 objdump unavailable: " + " | ".join(failures))

    pattern = re.compile(
        r"^\s*([0-9a-fA-F]+):\s+"
        r"(?:(?:[0-9a-fA-F]{2}\s+){4}|[0-9a-fA-F]{8}\s+)"
        r"(.*)$"
    )
    instructions = []
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        instructions.append(
            {
                "address": int(match.group(1), 16),
                "instruction": match.group(2).strip(),
            }
        )
    if not instructions:
        raise RuntimeError("objdump produced no AArch64 instructions")
    (out_dir / "coremark-text.disasm").write_text(output, encoding="utf-8")
    return instructions


def parse_spills(path: Path) -> dict[int, dict]:
    reports = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "[aot-spill-metric]" not in line:
            continue
        fields = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = int(value) if value.isdigit() else value
        if "local_func" in fields:
            reports[int(fields["local_func"])] = fields
    return reports


MEMSIZE_LOAD = re.compile(
    r"^ldr\s+x(?:[0-9]|[12][0-9]|30),\s*\[x19,\s*#(?:0x)?8\]$",
    re.IGNORECASE,
)
FRAME_REF = re.compile(r"\[(?:x29|sp)(?:,|\])", re.IGNORECASE)
FRAME_LOAD = re.compile(r"^(?:ldr|ldp|ldur)\b", re.IGNORECASE)
FRAME_STORE = re.compile(r"^(?:str|stp|stur)\b", re.IGNORECASE)
GPR = re.compile(r"\bx(?:[0-9]|[12][0-9]|30)\b", re.IGNORECASE)


def pct(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", type=Path, required=True)
    parser.add_argument("--cwasm", type=Path, required=True)
    parser.add_argument("--spill-log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hot-funcs", default="3,7,10")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    text, version, offsets = parse_cwasm(args.cwasm)
    base = text_base(args.perf, len(text))
    counts, total_samples = sample_counts(args.perf)
    instructions = disassemble(text, base, args.out_dir)
    spills = parse_spills(args.spill_log)
    hot_funcs = [int(value) for value in args.hot_funcs.split(",")]

    function_samples: Counter[int] = Counter()
    in_text_samples = 0
    for address, samples in counts.items():
        offset = address - base
        if not 0 <= offset < len(text):
            continue
        func = bisect.bisect_right(offsets, offset) - 1
        if func >= 0:
            function_samples[func] += samples
            in_text_samples += samples

    memsize_samples = 0
    frame_load_samples = 0
    frame_store_samples = 0
    memsize_sites = []
    by_func: dict[int, dict] = {}
    all_func_ids = set(hot_funcs)

    for inst in instructions:
        address = inst["address"]
        offset = address - base
        if not 0 <= offset < len(text):
            continue
        func = bisect.bisect_right(offsets, offset) - 1
        if func < 0:
            continue
        all_func_ids.add(func)
        text_inst = inst["instruction"]
        samples = counts.get(address, 0)
        record = by_func.setdefault(
            func,
            {
                "static_instructions": 0,
                "memory_size_loads": 0,
                "memory_size_samples": 0,
                "frame_loads": 0,
                "frame_load_samples": 0,
                "frame_stores": 0,
                "frame_store_samples": 0,
                "registers": set(),
                "instructions": [],
            },
        )
        record["static_instructions"] += 1
        record["registers"].update(reg.lower() for reg in GPR.findall(text_inst))
        record["instructions"].append((samples, address, text_inst))

        if MEMSIZE_LOAD.match(text_inst):
            record["memory_size_loads"] += 1
            record["memory_size_samples"] += samples
            memsize_samples += samples
            memsize_sites.append(
                {
                    "local_func": func,
                    "address": address,
                    "native_offset": offset - offsets[func],
                    "samples": samples,
                    "percent_of_run": pct(samples, total_samples),
                    "instruction": text_inst,
                }
            )
        if FRAME_REF.search(text_inst) and FRAME_LOAD.match(text_inst):
            record["frame_loads"] += 1
            record["frame_load_samples"] += samples
            frame_load_samples += samples
        if FRAME_REF.search(text_inst) and FRAME_STORE.match(text_inst):
            record["frame_stores"] += 1
            record["frame_store_samples"] += samples
            frame_store_samples += samples

    functions = {}
    for func in sorted(all_func_ids):
        start = offsets[func]
        end = offsets[func + 1] if func + 1 < len(offsets) else len(text)
        raw = by_func.get(
            func,
            {
                "static_instructions": 0,
                "memory_size_loads": 0,
                "memory_size_samples": 0,
                "frame_loads": 0,
                "frame_load_samples": 0,
                "frame_stores": 0,
                "frame_store_samples": 0,
                "registers": set(),
                "instructions": [],
            },
        )
        used_callee = sorted(
            reg for reg in raw["registers"] if reg in {f"x{i}" for i in range(21, 29)}
        )
        functions[str(func)] = {
            "code_bytes": end - start,
            "samples": function_samples.get(func, 0),
            "percent_of_run": pct(function_samples.get(func, 0), total_samples),
            "static_instructions": raw["static_instructions"],
            "memory_size_loads": raw["memory_size_loads"],
            "memory_size_samples": raw["memory_size_samples"],
            "memory_size_percent_of_run": pct(raw["memory_size_samples"], total_samples),
            "frame_loads": raw["frame_loads"],
            "frame_load_samples": raw["frame_load_samples"],
            "frame_stores": raw["frame_stores"],
            "frame_store_samples": raw["frame_store_samples"],
            "distinct_gprs": len(raw["registers"]),
            "used_allocatable_callee_saved": used_callee,
            "spill_metric": spills.get(func),
            "hottest_instructions": [
                {
                    "samples": samples,
                    "percent_of_run": pct(samples, total_samples),
                    "address": address,
                    "instruction": instruction,
                }
                for samples, address, instruction in sorted(
                    raw["instructions"], reverse=True
                )[:30]
                if samples
            ],
        }

        disasm_lines = [
            f"{address:016x}: {instruction}"
            for _, address, instruction in raw["instructions"]
        ]
        (args.out_dir / f"local_func_{func}.disasm").write_text(
            "\n".join(disasm_lines) + "\n", encoding="utf-8"
        )

    report = {
        "schema": "wamr-aarch64-memory-size-profile",
        "schema_version": 1,
        "aot_version": version,
        "text_base": base,
        "text_size": len(text),
        "function_count": len(offsets),
        "total_samples": total_samples,
        "attributed_samples": in_text_samples,
        "attribution_coverage_pct": pct(in_text_samples, total_samples),
        "memory_size_loads_static": len(memsize_sites),
        "memory_size_samples": memsize_samples,
        "memory_size_percent_of_run": pct(memsize_samples, total_samples),
        "frame_load_samples": frame_load_samples,
        "frame_load_percent_of_run": pct(frame_load_samples, total_samples),
        "frame_store_samples": frame_store_samples,
        "frame_store_percent_of_run": pct(frame_store_samples, total_samples),
        "top_functions": [
            {
                "local_func": func,
                "samples": samples,
                "percent_of_run": pct(samples, total_samples),
                "code_bytes": (
                    offsets[func + 1] if func + 1 < len(offsets) else len(text)
                )
                - offsets[func],
            }
            for func, samples in function_samples.most_common(20)
        ],
        "memory_size_sites": sorted(
            memsize_sites, key=lambda site: (-site["samples"], site["address"])
        ),
        "functions": {str(func): functions[str(func)] for func in hot_funcs},
    }
    (args.out_dir / "memory-size-profile.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# AArch64 CoreMark memory-size profile",
        "",
        f"- Total self samples: {total_samples}",
        (
            f"- AOT text attribution: {in_text_samples} "
            f"({pct(in_text_samples, total_samples):.2f}%)"
        ),
        (
            f"- `VmCtx.memory_size` load sites: {len(memsize_sites)} static; "
            f"{memsize_samples} samples ({pct(memsize_samples, total_samples):.2f}% of run)"
        ),
        (
            f"- FP/SP frame traffic samples: loads {frame_load_samples} "
            f"({pct(frame_load_samples, total_samples):.2f}%), stores {frame_store_samples} "
            f"({pct(frame_store_samples, total_samples):.2f}%)"
        ),
        "",
        "## Runtime-hot functions",
        "",
        "| local_func | run % | code B | mem-size loads | mem-size samples/run % | "
        "frame ld/st samples | spill ld/st | spilled vregs | callee-saved used |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for func in hot_funcs:
        item = functions[str(func)]
        spill = item["spill_metric"] or {}
        lines.append(
            f"| {func} | {item['percent_of_run']:.2f}% | {item['code_bytes']} | "
            f"{item['memory_size_loads']} | {item['memory_size_samples']} / "
            f"{item['memory_size_percent_of_run']:.2f}% | "
            f"{item['frame_load_samples']}/{item['frame_store_samples']} | "
            f"{spill.get('spill_ld', '?')}/{spill.get('spill_st', '?')} | "
            f"{spill.get('spilled_vregs', '?')} | "
            f"{', '.join(item['used_allocatable_callee_saved']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Hottest memory-size load sites",
            "",
            "| local_func | function offset | samples | run % | instruction |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for site in report["memory_size_sites"][:20]:
        lines.append(
            f"| {site['local_func']} | `+0x{site['native_offset']:x}` | "
            f"{site['samples']} | {site['percent_of_run']:.2f}% | "
            f"`{site['instruction']}` |"
        )
    (args.out_dir / "memory-size-profile.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
