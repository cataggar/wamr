#!/usr/bin/env python3
"""De-anonymize WAMR AOT perf data and attribute x86 frame traffic.

The optional compiler sidecar is deliberately authoritative: it identifies
each emitted frame-access instruction by a function-relative native byte
range and binds allocator traffic to a physical slot and, only when sound, a
vreg/source. Without a sidecar, frame moves remain "unattributed" rather than
being mislabeled as spills.
"""

import argparse
import bisect
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


AOT_MAGIC = 0x746F6100  # "\0aot"
AOT_VERSION = 8
SEC_TEXT = 2
SEC_FUNCTION = 3
FRAME_SCHEMA = "wamr-aot-frame-attribution"
FRAME_SCHEMA_VERSION = 1
FRAME_ORIGINS = {
    "allocator_spill",
    "wasm_local_or_phi",
    "explicit_frame_storage",
    "fixed_runtime_frame_state",
    "unknown",
}


class AttributionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CwasmInfo:
    func_offsets: list[int]
    text_size: int
    text_file_offset: int
    data: bytes
    version: int


@dataclass(frozen=True)
class Instruction:
    address: int
    offset: int
    size: int
    text: str


@dataclass(frozen=True)
class FrameOperand:
    kind: str
    base: str
    offset: int | None
    complex_address: bool


@dataclass
class FrameMetadata:
    raw: dict
    access_by_start: dict[int, dict]
    value_by_vreg: dict[int, dict]
    values_by_slot: dict[int, list[dict]]
    inline_data_ranges: list[dict]
    reconciliation: dict


def _run_checked(argv, what):
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise AttributionError(f"{what} failed: {detail}")
    return proc.stdout


def parse_cwasm(path):
    """Parse the text/function sections and reject incompatible layouts."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise AttributionError(f"{path}: cannot read: {exc}") from exc
    if len(data) < 8:
        raise AttributionError(f"{path}: truncated .cwasm header")
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != AOT_MAGIC:
        raise AttributionError(f"{path}: bad magic {magic:#x} (not a .cwasm)")
    if version != AOT_VERSION:
        raise AttributionError(
            f"{path}: incompatible aot_version={version}; tool requires {AOT_VERSION}"
        )

    pos = 8
    offsets = None
    text_size = None
    text_file_offset = None
    while pos < len(data):
        if pos + 8 > len(data):
            raise AttributionError(f"{path}: truncated section header at {pos:#x}")
        section_type, size = struct.unpack_from("<II", data, pos)
        pos += 8
        end = pos + size
        if end > len(data):
            raise AttributionError(
                f"{path}: section {section_type} overruns file "
                f"({pos:#x}+{size:#x}>{len(data):#x})"
            )
        if section_type == SEC_TEXT:
            if text_size is not None:
                raise AttributionError(f"{path}: duplicate text section")
            text_file_offset, text_size = pos, size
        elif section_type == SEC_FUNCTION:
            if offsets is not None:
                raise AttributionError(f"{path}: duplicate function section")
            if size < 4:
                raise AttributionError(f"{path}: truncated function section")
            count = struct.unpack_from("<I", data, pos)[0]
            needed = 4 + count * 8
            if needed > size:
                raise AttributionError(
                    f"{path}: function section declares {count} entries "
                    f"but has only {size} bytes"
                )
            interleaved = struct.unpack_from(f"<{count * 2}I", data, pos + 4)
            offsets = list(interleaved[0::2])
        pos = end

    if offsets is None or text_size is None or text_file_offset is None:
        raise AttributionError(f"{path}: missing function/text section")
    for index, offset in enumerate(offsets):
        if offset >= text_size:
            raise AttributionError(
                f"{path}: function {index} offset {offset:#x} is outside "
                f"text_size={text_size:#x}"
            )
        if index and offset <= offsets[index - 1]:
            raise AttributionError(
                f"{path}: ambiguous/non-increasing function offsets at "
                f"{index - 1}/{index}: {offsets[index - 1]:#x}, {offset:#x}"
            )
    return CwasmInfo(offsets, text_size, text_file_offset, data, version)


def function_bounds(info, func_index):
    if func_index < 0 or func_index >= len(info.func_offsets):
        raise AttributionError(
            f"--func {func_index} out of range (0..{len(info.func_offsets) - 1})"
        )
    start = info.func_offsets[func_index]
    end = (
        info.func_offsets[func_index + 1]
        if func_index + 1 < len(info.func_offsets)
        else info.text_size
    )
    if end <= start:
        raise AttributionError(
            f"local_func={func_index} has an empty or ambiguous native range"
        )
    return start, end


def jit_exec_mmaps(perf):
    """Return unique anonymous executable mappings, largest first."""
    output = _run_checked(
        ["perf", "script", "-i", perf, "--show-mmap-events"],
        "perf script --show-mmap-events",
    )
    maps = []
    pattern = re.compile(
        r"\[(0x[0-9a-fA-F]+)\((0x[0-9a-fA-F]+)\).*?\]: r[w-]xp //anon"
    )
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            maps.append((int(match.group(1), 16), int(match.group(2), 16)))
    return sorted(set(maps), key=lambda item: -item[1])


def addr_counts(perf):
    """Return ({ip: self_samples}, total_self_samples)."""
    output = _run_checked(
        [
            "perf",
            "report",
            "-i",
            perf,
            "--stdio",
            "-g",
            "none",
            "-n",
            "--sort=dso,symbol",
            "--percent-limit",
            "0",
        ],
        "perf report",
    )
    row = re.compile(r"^\s*[\d.]+%\s+[\d.]+%\s+(\d+)\s+(.*)$")
    counts, total = {}, 0
    for line in output.splitlines():
        match = row.match(line)
        if not match:
            continue
        count = int(match.group(1))
        total += count
        rest = match.group(2)
        if "[JIT]" not in rest:
            continue
        symbol = re.search(r"\[[.]\]\s+(0x[0-9a-fA-F]+)", rest)
        if symbol:
            ip = int(symbol.group(1), 16)
            counts[ip] = counts.get(ip, 0) + count
    return counts, total


def select_text_base(perf, text_size, explicit_base=None):
    if explicit_base is not None:
        return int(explicit_base, 16)
    maps = jit_exec_mmaps(perf)
    matches = [(base, size) for base, size in maps if abs(size - text_size) <= 0x2000]
    if len(matches) != 1:
        rendered = ", ".join(f"{base:#x}/{size:#x}" for base, size in matches)
        raise AttributionError(
            f"expected exactly one anonymous executable mmap matching "
            f"text_size={text_size}, found {len(matches)}"
            f"{': ' + rendered if rendered else ''}; pass --base explicitly"
        )
    return matches[0][0]


def disassemble_blob(
    blob, virtual_address, scratch_dir, label, function_offset_base=0
):
    scratch_dir = Path(scratch_dir)
    scratch = scratch_dir / f".aot-jit-attr-{os.getpid()}-{label}.bin"
    try:
        scratch.write_bytes(blob)
        output = _run_checked(
            [
                "objdump",
                "-D",
                "-b",
                "binary",
                "-m",
                "i386:x86-64",
                "-M",
                "intel",
                f"--adjust-vma=0x{virtual_address:x}",
                str(scratch),
            ],
            "objdump",
        )
    finally:
        scratch.unlink(missing_ok=True)

    pattern = re.compile(
        r"^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+)+)\s*(.*)$"
    )
    instructions = []
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        address = int(match.group(1), 16)
        encoded = match.group(2).split()
        instructions.append(
            Instruction(
                address=address,
                offset=function_offset_base + address - virtual_address,
                size=len(encoded),
                text=match.group(3).strip(),
            )
        )
    if not instructions and blob:
        raise AttributionError("objdump returned no instructions for selected function")
    return instructions


def disassemble_function(
    function_code,
    function_base,
    scratch_dir,
    label,
    inline_data_ranges,
):
    instructions = []
    cursor = 0
    for index, data_range in enumerate(inline_data_ranges):
        start = data_range["native_start"]
        end = data_range["native_end"]
        if cursor < start:
            instructions.extend(
                disassemble_blob(
                    function_code[cursor:start],
                    function_base + cursor,
                    scratch_dir,
                    f"{label}-segment-{index}",
                    cursor,
                )
            )
        cursor = end
    if cursor < len(function_code):
        instructions.extend(
            disassemble_blob(
                function_code[cursor:],
                function_base + cursor,
                scratch_dir,
                f"{label}-segment-tail",
                cursor,
            )
        )
    return instructions


def _split_operands(text):
    parts = text.split(None, 1)
    mnemonic = parts[0].lower() if parts else ""
    if len(parts) <= 1:
        return mnemonic, []
    operands, current, depth = [], [], 0
    for char in parts[1]:
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            operands.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        operands.append("".join(current).strip())
    return mnemonic, operands


def parse_frame_operand(text):
    """Identify an explicit rbp/rsp memory operand and its access direction."""
    mnemonic, operands = _split_operands(text)
    if mnemonic == "push" and operands and "[" not in operands[0]:
        return FrameOperand("store", "rsp", -8, False)
    if mnemonic == "pop" and operands and "[" not in operands[0]:
        return FrameOperand("load", "rsp", 0, False)
    if mnemonic.startswith("lea"):
        return None
    for operand_index, operand in enumerate(operands):
        bracket = re.search(r"\[([^\]]+)\]", operand)
        if not bracket:
            continue
        expression = re.sub(r"\s+", "", bracket.group(1).lower())
        base_match = re.search(r"(?<![a-z0-9_])(rbp|rsp)(?![a-z0-9_])", expression)
        if not base_match:
            continue
        base = base_match.group(1)
        simple = re.fullmatch(r"(rbp|rsp)(?:([+-])(0x[0-9a-f]+|\d+))?", expression)
        offset = None
        complex_address = True
        if simple:
            magnitude = int(simple.group(3), 0) if simple.group(3) else 0
            offset = -magnitude if simple.group(2) == "-" else magnitude
            complex_address = False
        # For MOV-family instructions the first memory operand is the
        # destination. Other explicit frame operands are conservatively reads.
        kind = (
            "store"
            if mnemonic.startswith("mov") and operand_index == 0
            else "load"
        )
        return FrameOperand(kind, base, offset, complex_address)
    return None


def normalized_code_sha256(code, rel32_offsets):
    normalized = bytearray(code)
    prior_end = 0
    for offset in rel32_offsets:
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise AttributionError("direct_call_rel32_offsets must contain integers")
        if offset < prior_end or offset + 4 > len(normalized):
            raise AttributionError(
                f"invalid/overlapping direct-call rel32 offset {offset}"
            )
        normalized[offset : offset + 4] = b"\0\0\0\0"
        prior_end = offset + 4
    return hashlib.sha256(normalized).hexdigest()


def _require_int(mapping, key):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AttributionError(f"frame metadata field {key!r} must be an integer")
    return value


def _require_list(mapping, key):
    value = mapping.get(key)
    if not isinstance(value, list):
        raise AttributionError(f"frame metadata field {key!r} must be an array")
    return value


def load_frame_metadata(
    path,
    func_index,
    function_code,
    cwasm_version,
    module_text,
    function_offset,
):
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttributionError(f"{path}: malformed frame metadata: {exc}") from exc
    if not isinstance(raw, dict):
        raise AttributionError(f"{path}: frame metadata root must be an object")
    if raw.get("schema") != FRAME_SCHEMA:
        raise AttributionError(
            f"{path}: incompatible frame schema {raw.get('schema')!r}"
        )
    if raw.get("schema_version") != FRAME_SCHEMA_VERSION:
        raise AttributionError(
            f"{path}: incompatible frame schema_version="
            f"{raw.get('schema_version')!r}; expected {FRAME_SCHEMA_VERSION}"
        )
    if raw.get("architecture") != "x86_64":
        raise AttributionError(
            f"{path}: frame metadata architecture must be x86_64"
        )
    _require_int(raw, "module")
    if raw.get("abi") not in {"sysv", "win64"}:
        raise AttributionError(f"{path}: frame metadata abi must be sysv or win64")
    if not isinstance(raw.get("compiler_build_id"), str):
        raise AttributionError(f"{path}: compiler_build_id must be a string")
    if _require_int(raw, "cwasm_aot_version") != cwasm_version:
        raise AttributionError(
            f"{path}: metadata/cwasm AOT version mismatch "
            f"({_require_int(raw, 'cwasm_aot_version')} != {cwasm_version})"
        )
    if _require_int(raw, "local_func") != func_index:
        raise AttributionError(
            f"{path}: metadata local_func={raw.get('local_func')} does not match "
            f"--func {func_index}"
        )
    module_text_size = _require_int(raw, "module_text_size")
    if module_text_size != len(module_text):
        raise AttributionError(
            f"{path}: metadata module_text_size={module_text_size} does not "
            f"match cwasm text_size={len(module_text)}"
        )
    if _require_int(raw, "function_offset") != function_offset:
        raise AttributionError(
            f"{path}: metadata function_offset={raw.get('function_offset')} "
            f"does not match cwasm offset={function_offset}"
        )
    module_hash = raw.get("module_text_sha256")
    actual_module_hash = hashlib.sha256(module_text).hexdigest()
    if not isinstance(module_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", module_hash
    ):
        raise AttributionError(
            f"{path}: module_text_sha256 must be 64 lowercase hex digits"
        )
    if module_hash != actual_module_hash:
        raise AttributionError(
            f"{path}: module text hash mismatch; metadata belongs to a "
            f"different core ({module_hash} != {actual_module_hash})"
        )
    code_size = _require_int(raw, "code_size")
    if code_size != len(function_code):
        raise AttributionError(
            f"{path}: metadata code_size={code_size} does not match cwasm "
            f"function span={len(function_code)}"
        )
    rel32_offsets = _require_list(raw, "direct_call_rel32_offsets")
    actual_hash = normalized_code_sha256(function_code, rel32_offsets)
    expected_hash = raw.get("normalized_code_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ):
        raise AttributionError(
            f"{path}: normalized_code_sha256 must be 64 lowercase hex digits"
        )
    if actual_hash != expected_hash:
        raise AttributionError(
            f"{path}: normalized native-code hash mismatch; metadata does not "
            f"belong to this local_func ({expected_hash} != {actual_hash})"
        )

    layout = raw.get("frame_layout")
    metric = raw.get("spill_metric")
    if not isinstance(layout, dict) or not isinstance(metric, dict):
        raise AttributionError(f"{path}: missing frame_layout/spill_metric object")
    for key in (
        "frame_size",
        "local_count",
        "param_count",
        "reserved_vmctx_offset",
        "locals_first_offset",
        "explicit_storage_first_offset",
        "explicit_storage_slots",
        "spill_base",
        "spill_stride",
        "spill_slots",
    ):
        _require_int(layout, key)
    for key in (
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
    ):
        _require_int(metric, key)

    values = _require_list(raw, "allocator_values")
    value_by_vreg = {}
    values_by_slot = defaultdict(list)
    for value in values:
        if not isinstance(value, dict):
            raise AttributionError(f"{path}: allocator_values entries must be objects")
        vreg = _require_int(value, "vreg")
        slot = _require_int(value, "slot")
        slot_count = _require_int(value, "slot_count")
        for key in (
            "frame_offset",
            "ir_use_count",
            "ir_def_count",
            "reload_count",
            "store_count",
        ):
            _require_int(value, key)
        if not isinstance(value.get("rematerialization_eligible"), bool):
            raise AttributionError(
                f"{path}: rematerialization_eligible must be boolean"
            )
        if not isinstance(value.get("reused"), bool):
            raise AttributionError(f"{path}: reused must be boolean")
        if vreg in value_by_vreg or slot < 0 or slot_count <= 0:
            raise AttributionError(
                f"{path}: invalid/duplicate allocator value vreg={vreg}"
            )
        value_by_vreg[vreg] = value
        for occupied in range(slot, slot + slot_count):
            values_by_slot[occupied].append(value)
    for value in values:
        actually_reused = any(
            len(values_by_slot[occupied]) > 1
            for occupied in range(
                value["slot"], value["slot"] + value["slot_count"]
            )
        )
        if value["reused"] != actually_reused:
            raise AttributionError(
                f"{path}: vreg {value['vreg']} reused flag disagrees with "
                "overlapping allocator slots"
            )

    inline_data_ranges = _require_list(raw, "inline_data_ranges")
    prior_data_end = 0
    for data_range in inline_data_ranges:
        if not isinstance(data_range, dict):
            raise AttributionError(
                f"{path}: inline_data_ranges entries must be objects"
            )
        start = _require_int(data_range, "native_start")
        end = _require_int(data_range, "native_end")
        if (
            start < prior_data_end
            or end <= start
            or end > code_size
            or data_range.get("kind") != "br_table"
        ):
            raise AttributionError(
                f"{path}: invalid/overlapping inline data range [{start},{end})"
            )
        prior_data_end = end

    accesses = _require_list(raw, "accesses")
    access_by_start = {}
    prior_end = 0
    emitted_loads = emitted_stores = 0
    resolved_by_vreg = defaultdict(lambda: {"load": 0, "store": 0})
    for access in accesses:
        if not isinstance(access, dict):
            raise AttributionError(f"{path}: accesses entries must be objects")
        start = _require_int(access, "native_start")
        end = _require_int(access, "native_end")
        kind = access.get("kind")
        base = access.get("base")
        origin = access.get("origin")
        _require_int(access, "frame_offset")
        _require_int(access, "width")
        if start < prior_end or end <= start or end > code_size:
            raise AttributionError(
                f"{path}: overlapping/out-of-range native access [{start},{end})"
            )
        if start in access_by_start:
            raise AttributionError(f"{path}: duplicate access at native offset {start}")
        if kind not in {"load", "store"} or base not in {"rbp", "rsp"}:
            raise AttributionError(
                f"{path}: invalid frame access kind/base at offset {start}"
            )
        if origin not in FRAME_ORIGINS:
            raise AttributionError(
                f"{path}: invalid frame origin {origin!r} at offset {start}"
            )
        if any(
            data_range["native_start"] <= start < data_range["native_end"]
            for data_range in inline_data_ranges
        ):
            raise AttributionError(
                f"{path}: frame access at {start} overlaps inline data"
            )
        vreg = access.get("vreg")
        ambiguous = access.get("vreg_ambiguous")
        if vreg is not None and (isinstance(vreg, bool) or not isinstance(vreg, int)):
            raise AttributionError(f"{path}: vreg must be integer or null")
        if not isinstance(ambiguous, bool) or (vreg is not None and ambiguous):
            raise AttributionError(
                f"{path}: invalid vreg/vreg_ambiguous combination at {start}"
            )
        if origin == "allocator_spill":
            slot = access.get("slot")
            if isinstance(slot, bool) or not isinstance(slot, int):
                raise AttributionError(
                    f"{path}: allocator access at {start} lacks integer slot"
                )
            candidates = values_by_slot.get(slot, [])
            if not candidates:
                raise AttributionError(
                    f"{path}: allocator access at {start} names unassigned slot {slot}"
                )
            if vreg is not None and all(item["vreg"] != vreg for item in candidates):
                raise AttributionError(
                    f"{path}: access vreg={vreg} does not occupy slot {slot}"
                )
            if len(candidates) > 1 and vreg is None and not ambiguous:
                raise AttributionError(
                    f"{path}: reused slot {slot} lacks a resolved vreg or "
                    f"vreg_ambiguous=true"
                )
            if len(candidates) == 1 and vreg is None:
                raise AttributionError(
                    f"{path}: unique allocator slot {slot} omitted provable vreg"
                )
            if vreg is not None:
                resolved_by_vreg[vreg][kind] += 1
            if kind == "load":
                emitted_loads += 1
            else:
                emitted_stores += 1
        access_by_start[start] = access
        prior_end = end

    for vreg, value in value_by_vreg.items():
        resolved = resolved_by_vreg[vreg]
        if (
            value["reload_count"] != resolved["load"]
            or value["store_count"] != resolved["store"]
        ):
            raise AttributionError(
                f"{path}: vreg {vreg} emitted count mismatch "
                f"({value['reload_count']}/{value['store_count']} != "
                f"{resolved['load']}/{resolved['store']})"
            )

    declared_loads = _require_int(raw, "emitted_allocator_loads")
    declared_stores = _require_int(raw, "emitted_allocator_stores")
    if (declared_loads, declared_stores) != (emitted_loads, emitted_stores):
        raise AttributionError(
            f"{path}: emitted allocator counts disagree with access records "
            f"({declared_loads}/{declared_stores} != "
            f"{emitted_loads}/{emitted_stores})"
        )
    metric_loads = _require_int(metric, "spill_ld")
    metric_stores = _require_int(metric, "spill_st")
    if (emitted_loads, emitted_stores) != (metric_loads, metric_stores):
        raise AttributionError(
            f"{path}: allocator access reconciliation failed: emitted "
            f"ld/st={emitted_loads}/{emitted_stores}, "
            f"WAMR_AOT_SPILL_METRIC={metric_loads}/{metric_stores}"
        )

    return FrameMetadata(
        raw=raw,
        access_by_start=access_by_start,
        value_by_vreg=value_by_vreg,
        values_by_slot=dict(values_by_slot),
        inline_data_ranges=inline_data_ranges,
        reconciliation={
            "emitted_allocator_loads": emitted_loads,
            "emitted_allocator_stores": emitted_stores,
            "spill_metric_loads": metric_loads,
            "spill_metric_stores": metric_stores,
            "matches": True,
        },
    )


def validate_metadata_disassembly(metadata, instructions):
    instruction_by_offset = {instruction.offset: instruction for instruction in instructions}
    for start, access in metadata.access_by_start.items():
        instruction = instruction_by_offset.get(start)
        if instruction is None:
            raise AttributionError(
                f"metadata access at +0x{start:x} is not an instruction boundary"
            )
        if start + instruction.size != access["native_end"]:
            raise AttributionError(
                f"metadata range [{start},{access['native_end']}) disagrees with "
                f"objdump instruction size {instruction.size}"
            )
        operand = parse_frame_operand(instruction.text)
        if operand is None:
            raise AttributionError(
                f"metadata access at +0x{start:x} is not a frame load/store: "
                f"{instruction.text}"
            )
        if operand.kind != access["kind"] or operand.base != access["base"]:
            raise AttributionError(
                f"metadata access kind/base mismatch at +0x{start:x}: "
                f"{access['kind']}/{access['base']} vs "
                f"{operand.kind}/{operand.base}"
            )
        if operand.offset is not None and operand.offset != access["frame_offset"]:
            raise AttributionError(
                f"metadata frame offset mismatch at +0x{start:x}: "
                f"{access['frame_offset']} vs {operand.offset}"
            )


def classify_basic(text):
    mnemonic = text.split()[0].lower() if text else ""
    frame = parse_frame_operand(text)
    if frame:
        return f"frame_{frame.kind}_unattributed"
    is_mov = mnemonic in {
        "mov",
        "movzx",
        "movsx",
        "movsxd",
        "movsd",
        "movss",
        "movdqu",
        "movdqa",
        "movaps",
        "movups",
        "movq",
        "movd",
    }
    if mnemonic == "cmp" and re.search(r"\[(rbx|r10|r11)\+0x8\]", text):
        return "bounds_cmp"
    if mnemonic in {"ja", "jae", "jb", "jbe"}:
        return "bounds_branch"
    if mnemonic == "call":
        return "call"
    if mnemonic == "jmp":
        return "dispatch_jmp" if re.search(r"jmp\s+r(ax|10|11)", text) else "jmp"
    if mnemonic in {
        "je",
        "jne",
        "jl",
        "jle",
        "jg",
        "jge",
        "js",
        "jns",
        "jp",
        "jnp",
        "jo",
        "jno",
    }:
        return "cond_branch"
    if is_mov and "[" in text and "rip" not in text:
        return "mem_access"
    if is_mov:
        return "regmov"
    if mnemonic in {
        "add",
        "sub",
        "and",
        "or",
        "xor",
        "shl",
        "shr",
        "sar",
        "imul",
        "mul",
        "inc",
        "dec",
        "neg",
        "not",
        "test",
        "lea",
        "sete",
        "setne",
        "seta",
        "cdqe",
        "cqo",
    }:
        return "alu"
    return "other"


def classify_instruction(instruction, metadata=None):
    if metadata:
        access = metadata.access_by_start.get(instruction.offset)
        if access:
            return f"{access['origin']}_{access['kind']}"
        frame = parse_frame_operand(instruction.text)
        if frame:
            return f"unknown_frame_{frame.kind}"
    return classify_basic(instruction.text)


def _percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else 0.0


def build_frame_summary(instructions, counts, metadata):
    origin_static = Counter()
    origin_samples = Counter()
    frame_instruction_count = 0
    frame_samples = 0
    attributed_instructions = 0
    attributed_samples = 0
    proven_origin_instructions = 0
    proven_origin_samples = 0
    unknown = []
    contributors = {}

    for instruction in instructions:
        operand = parse_frame_operand(instruction.text)
        if operand is None:
            continue
        samples = counts.get(instruction.address, 0)
        frame_instruction_count += 1
        frame_samples += samples
        access = metadata.access_by_start.get(instruction.offset)
        if access is None:
            unknown.append(
                {
                    "native_offset": instruction.offset,
                    "address": instruction.address,
                    "samples": samples,
                    "instruction": instruction.text,
                    "base": operand.base,
                    "frame_offset": operand.offset,
                    "complex_address": operand.complex_address,
                }
            )
            origin_static["unknown"] += 1
            origin_samples["unknown"] += samples
            continue

        origin = access["origin"]
        attributed_instructions += 1
        attributed_samples += samples
        origin_static[origin] += 1
        origin_samples[origin] += samples
        if origin != "unknown":
            proven_origin_instructions += 1
            proven_origin_samples += samples
        else:
            unknown.append(
                {
                    "native_offset": instruction.offset,
                    "address": instruction.address,
                    "samples": samples,
                    "instruction": instruction.text,
                    "base": access["base"],
                    "frame_offset": access["frame_offset"],
                    "complex_address": False,
                    "detail": access.get("detail"),
                }
            )
        if origin != "allocator_spill":
            continue

        slot = access["slot"]
        vreg = access.get("vreg")
        key = (slot, vreg)
        if key not in contributors:
            value = metadata.value_by_vreg.get(vreg) if vreg is not None else None
            contributors[key] = {
                "slot": slot,
                "frame_offset": access["frame_offset"],
                "vreg": vreg,
                "vreg_ambiguous": access["vreg_ambiguous"],
                "candidate_vregs": [
                    candidate["vreg"]
                    for candidate in metadata.values_by_slot.get(slot, [])
                ],
                "defining_opcode": access.get("defining_opcode"),
                "source_class": access.get("source_class"),
                "rematerialization_eligible": access.get(
                    "rematerialization_eligible"
                ),
                "source_reload_count": value.get("reload_count") if value else None,
                "source_store_count": value.get("store_count") if value else None,
                "source_ir_use_count": value.get("ir_use_count") if value else None,
                "source_ir_def_count": value.get("ir_def_count") if value else None,
                "static_loads": 0,
                "static_stores": 0,
                "samples": 0,
            }
        record = contributors[key]
        record[f"static_{access['kind']}s"] += 1
        record["samples"] += samples

    ranked = sorted(
        contributors.values(),
        key=lambda item: (
            -item["samples"],
            -(item["static_loads"] + item["static_stores"]),
            item["slot"],
            -1 if item["vreg"] is None else item["vreg"],
        ),
    )
    unknown.sort(key=lambda item: (-item["samples"], item["native_offset"]))
    origins = {
        origin: {
            "static_instructions": origin_static.get(origin, 0),
            "samples": origin_samples.get(origin, 0),
            "percent_of_frame_samples": _percent(
                origin_samples.get(origin, 0), frame_samples
            ),
        }
        for origin in sorted(origin_static)
    }
    return {
        "coverage": {
            "frame_instructions": frame_instruction_count,
            "attributed_frame_instructions": attributed_instructions,
            "static_coverage_pct": _percent(
                attributed_instructions, frame_instruction_count
            ),
            "frame_samples": frame_samples,
            "attributed_frame_samples": attributed_samples,
            "metadata_mapping_coverage_pct": _percent(
                attributed_samples, frame_samples
            ),
            "proven_origin_frame_instructions": proven_origin_instructions,
            "unknown_frame_instructions": frame_instruction_count
            - proven_origin_instructions,
            "origin_coverage_pct": _percent(
                proven_origin_instructions, frame_instruction_count
            ),
            "proven_origin_frame_samples": proven_origin_samples,
            "unknown_frame_samples": frame_samples - proven_origin_samples,
            "sample_coverage_pct": _percent(attributed_samples, frame_samples),
            "origin_sample_coverage_pct": _percent(
                proven_origin_samples, frame_samples
            ),
        },
        "origins": origins,
        "allocator_contributors": ranked,
        "unknown_instructions": unknown,
        "reconciliation": metadata.reconciliation,
    }


def print_frame_summary(summary, total_samples, top):
    coverage = summary["coverage"]
    reconciliation = summary["reconciliation"]
    print("\n=== frame-origin attribution ===")
    print(
        "  metadata mapping: "
        f"{coverage['attributed_frame_instructions']}/"
        f"{coverage['frame_instructions']} static frame instructions "
        f"({coverage['static_coverage_pct']:.1f}%); "
        f"{coverage['attributed_frame_samples']}/"
        f"{coverage['frame_samples']} frame samples "
        f"({coverage['sample_coverage_pct']:.1f}%)"
    )
    print(
        "  proven origins: "
        f"{coverage['proven_origin_frame_instructions']}/"
        f"{coverage['frame_instructions']} static frame instructions "
        f"({coverage['origin_coverage_pct']:.1f}%); "
        f"{coverage['proven_origin_frame_samples']}/"
        f"{coverage['frame_samples']} frame samples "
        f"({coverage['origin_sample_coverage_pct']:.1f}%)"
    )
    print(
        "  reconciliation: emitted allocator ld/st="
        f"{reconciliation['emitted_allocator_loads']}/"
        f"{reconciliation['emitted_allocator_stores']} == "
        "WAMR_AOT_SPILL_METRIC "
        f"{reconciliation['spill_metric_loads']}/"
        f"{reconciliation['spill_metric_stores']} (match)"
    )
    for origin, values in sorted(
        summary["origins"].items(),
        key=lambda item: (-item[1]["samples"], item[0]),
    ):
        print(
            f"  {origin:<27} static={values['static_instructions']:<6} "
            f"samples={values['samples']:<7} "
            f"({_percent(values['samples'], total_samples):.2f}% of run)"
        )

    contributors = summary["allocator_contributors"][:top]
    if contributors:
        print(f"\n=== top {len(contributors)} allocator slot/source contributors ===")
        for item in contributors:
            identity = (
                f"v{item['vreg']}"
                if item["vreg"] is not None
                else f"ambiguous{item['candidate_vregs']}"
            )
            print(
                f"  samples={item['samples']:<6} "
                f"slot={item['slot']:<5} off={item['frame_offset']:<7} "
                f"{identity:<20} "
                f"src={item['defining_opcode'] or '?'}"
                f"/{item['source_class'] or '?'} "
                f"static_ld/st={item['static_loads']}/{item['static_stores']} "
                f"ir_use/def={item['source_ir_use_count']}/"
                f"{item['source_ir_def_count']}"
            )
    if coverage["unknown_frame_instructions"]:
        print(
            f"\n  unknown frame instructions: "
            f"{coverage['unknown_frame_instructions']} "
            f"({coverage['unknown_frame_samples']} samples)"
        )


def _write_json(path, report):
    Path(path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", help="perf.data file (omit only for static validation)")
    parser.add_argument("--cwasm", required=True, help="core .cwasm owning the code")
    parser.add_argument("--func", type=int, help="selected local_func")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--base", help="explicit text mmap base (hex)")
    parser.add_argument("--json-out", help="write machine-readable attribution summary")
    parser.add_argument("--min-samples", type=int, default=0)
    parser.add_argument(
        "--require-size-match",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--frame-metadata",
        help="exact compiler sidecar for --func "
        "(<prefix>.mod<M>.func<F>.json)",
    )
    parser.add_argument(
        "--validate-frame-metadata",
        action="store_true",
        help="validate cwasm + sidecar + disassembly without perf data",
    )
    args = parser.parse_args()

    if not args.perf and not args.validate_frame_metadata:
        parser.error("--perf is required unless --validate-frame-metadata is set")
    if args.validate_frame_metadata and (args.func is None or not args.frame_metadata):
        parser.error("--validate-frame-metadata requires --func and --frame-metadata")
    if args.frame_metadata and args.func is None:
        parser.error("--frame-metadata requires --func")

    info = parse_cwasm(args.cwasm)
    counts, total = ({}, 0)
    base = None
    if args.perf:
        counts, total = addr_counts(args.perf)
        if total == 0:
            raise AttributionError("no samples in perf data")
        if total < args.min_samples:
            raise AttributionError(
                f"perf data has only {total} self samples; "
                f"--min-samples requires at least {args.min_samples}"
            )
        base = select_text_base(args.perf, info.text_size, args.base)

    per_function, in_core = {}, 0
    top_functions = []
    if args.perf:
        end = base + info.text_size
        print(
            f"core text base={base:#x} size={info.text_size} "
            f"({info.text_size / 1048576:.1f} MB), "
            f"func_count={len(info.func_offsets)}, total self samples={total}"
        )
        for ip, count in counts.items():
            if not (base <= ip < end):
                continue
            local_func = bisect.bisect_right(info.func_offsets, ip - base) - 1
            if local_func < 0:
                continue
            in_core += count
            per_function[local_func] = per_function.get(local_func, 0) + count
        print(
            f"samples in this core: {in_core} "
            f"({_percent(in_core, total):.1f}% of run)\n"
        )
        print(f"=== top {args.top} functions by self samples ===")
        for local_func, count in sorted(
            per_function.items(), key=lambda item: -item[1]
        )[: args.top]:
            start, end = function_bounds(info, local_func)
            top_functions.append(
                {
                    "local_func": local_func,
                    "samples": count,
                    "percent_of_run": _percent(count, total),
                    "code_bytes": end - start,
                }
            )
            print(
                f"  local_func={local_func:<6} samples={count:<6} "
                f"({_percent(count, total):.2f}% of run)  "
                f"code_bytes={end - start}"
            )

    report = {
        "schema_version": 2,
        "perf": str(Path(args.perf).resolve()) if args.perf else None,
        "cwasm": str(Path(args.cwasm).resolve()),
        "text_base": base,
        "text_size": info.text_size,
        "function_count": len(info.func_offsets),
        "total_samples": total,
        "attributed_samples": in_core,
        "attribution_coverage_pct": _percent(in_core, total),
        "top_functions": top_functions,
    }
    if args.func is None:
        if args.json_out:
            _write_json(args.json_out, report)
        return

    start, end = function_bounds(info, args.func)
    function_code = info.data[
        info.text_file_offset + start : info.text_file_offset + end
    ]
    function_base = (base + start) if base is not None else 0
    scratch_dir = (
        Path(args.perf).resolve().parent
        if args.perf
        else Path(args.frame_metadata).resolve().parent
    )

    metadata = None
    frame_summary = None
    if args.frame_metadata:
        metadata = load_frame_metadata(
            args.frame_metadata,
            args.func,
            function_code,
            info.version,
            info.data[
                info.text_file_offset : info.text_file_offset + info.text_size
            ],
            start,
        )
        instructions = disassemble_function(
            function_code,
            function_base,
            scratch_dir,
            f"func-{args.func}",
            metadata.inline_data_ranges,
        )
        validate_metadata_disassembly(metadata, instructions)
        frame_summary = build_frame_summary(instructions, counts, metadata)
    else:
        instructions = disassemble_blob(
            function_code, function_base, scratch_dir, f"func-{args.func}"
        )

    function_counts = {
        ip: count
        for ip, count in counts.items()
        if function_base <= ip < function_base + len(function_code)
    }
    function_samples = sum(function_counts.values())
    by_class = Counter()
    hot = []
    instruction_addresses = set()
    for instruction in instructions:
        instruction_addresses.add(instruction.address)
        samples = counts.get(instruction.address, 0)
        by_class[classify_instruction(instruction, metadata)] += samples
        if samples:
            hot.append((samples, instruction.address, instruction.text))
    for address, samples in function_counts.items():
        if address in instruction_addresses:
            continue
        offset = address - function_base
        in_inline_data = metadata is not None and any(
            data_range["native_start"] <= offset < data_range["native_end"]
            for data_range in metadata.inline_data_ranges
        )
        class_name = (
            "inline_data_or_sample_skid"
            if in_inline_data
            else "unknown_instruction_boundary"
        )
        by_class[class_name] += samples
        hot.append((samples, address, f"<{class_name}>"))

    mode = "static validation" if not args.perf else (
        f"self={function_samples}, {_percent(function_samples, total):.1f}% of run"
    )
    print(f"\n=== local_func={args.func} instruction-class mix ({mode}) ===")
    for class_name, samples in sorted(
        by_class.items(), key=lambda item: (-item[1], item[0])
    ):
        print(
            f"  {class_name:<36} {samples:>7} "
            f"({_percent(samples, total):.2f}% of run)"
        )
    if metadata is None:
        frame_count = sum(
            1 for instruction in instructions if parse_frame_operand(instruction.text)
        )
        if frame_count:
            print(
                f"  note: {frame_count} frame instructions remain unattributed; "
                "pass --frame-metadata to distinguish spills from locals/fixed state"
            )
    else:
        print_frame_summary(frame_summary, total, args.top)

    if args.perf:
        print(f"\n=== top 20 hottest instructions in local_func={args.func} ===")
        for samples, address, text in sorted(hot, reverse=True)[:20]:
            print(
                f"  {samples:>5} ({_percent(samples, total):.2f}%)  "
                f"{address:x}: {text}"
            )

    report["classified_function"] = {
        "local_func": args.func,
        "samples": function_samples,
        "percent_of_run": _percent(function_samples, total),
        "classes": {
            class_name: {
                "samples": samples,
                "percent_of_run": _percent(samples, total),
            }
            for class_name, samples in sorted(by_class.items())
        },
        "hottest_instructions": [
            {
                "samples": samples,
                "percent_of_run": _percent(samples, total),
                "address": address,
                "instruction": text,
            }
            for samples, address, text in sorted(hot, reverse=True)[:20]
        ],
    }
    if metadata is not None:
        report["classified_function"]["frame_attribution"] = frame_summary
        report["classified_function"]["frame_metadata"] = str(
            Path(args.frame_metadata).resolve()
        )
        report["classified_function"]["frame_metadata_module"] = metadata.raw[
            "module"
        ]
        report["classified_function"]["frame_metadata_identity"] = {
            "schema": metadata.raw["schema"],
            "schema_version": metadata.raw["schema_version"],
            "compiler_build_id": metadata.raw["compiler_build_id"],
            "module_text_sha256": metadata.raw["module_text_sha256"],
            "normalized_code_sha256": metadata.raw[
                "normalized_code_sha256"
            ],
        }
    if args.json_out:
        _write_json(args.json_out, report)


if __name__ == "__main__":
    try:
        main()
    except AttributionError as exc:
        sys.exit(str(exc))
