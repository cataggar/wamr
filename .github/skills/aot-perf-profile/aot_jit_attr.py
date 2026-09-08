#!/usr/bin/env python3
"""De-anonymize WAMR AOT perf data and classify generated instructions.

The optional compiler sidecar is deliberately authoritative: it identifies
each emitted frame-access instruction by a function-relative native byte
range and binds allocator traffic to a physical slot and, only when sound, a
vreg/source where the emitter can prove it. Without a sidecar, x86_64 and
AArch64 frame moves remain
"unattributed" rather than being mislabeled as spills.
"""

import argparse
import bisect
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


AOT_MAGIC = 0x746F6100  # "\0aot"
AOT_VERSION = 11
SUPPORTED_AOT_VERSIONS = {9, 10, AOT_VERSION}
SEC_TEXT = 2
SEC_FUNCTION = 3
SEC_TARGET_INFO = 0
FRAME_SCHEMA = "wamr-aot-frame-attribution"
FRAME_SCHEMA_VERSION = 2
SUPPORTED_FRAME_SCHEMA_VERSIONS = {1, FRAME_SCHEMA_VERSION}
FRAME_ORIGINS = {
    "allocator_spill",
    "wasm_local_or_phi",
    "explicit_frame_storage",
    "fixed_runtime_frame_state",
    "unknown",
}
SUPPORTED_ARCHITECTURES = {"x86_64", "aarch64"}


class AttributionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CwasmInfo:
    func_offsets: list[int]
    text_size: int
    text_file_offset: int
    data: bytes
    version: int
    architecture: str
    abi: str | None
    target_format: str
    target_verified: bool


@dataclass(frozen=True)
class Instruction:
    address: int
    offset: int
    size: int
    text: str


@dataclass(frozen=True)
class TextMappingSelection:
    base: int
    size: int
    text_size: int
    page_size: int
    expected_size: int
    candidates: list[tuple[int, int]]
    authoritative: bool
    override: str | None


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


def perf_binary():
    return os.environ.get("PERF", "perf")


def normalize_architecture(value=None):
    arch = (platform.machine() if value is None else value).lower()
    arch = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86-64": "x86_64",
        "arm64": "aarch64",
    }.get(arch, arch)
    if arch not in SUPPORTED_ARCHITECTURES:
        raise AttributionError(
            f"unsupported disassembly architecture {arch!r}; expected "
            "x86_64 or aarch64"
        )
    return arch


def _parse_target_info(path, payload):
    if len(payload) != 40:
        raise AttributionError(
            f"{path}: target-info section must be exactly 40 bytes"
        )
    (
        bin_type,
        abi_type,
        e_type,
        e_machine,
        e_flags,
        reserved,
    ) = struct.unpack_from("<HHHHII", payload)
    arch_field = payload[16:32]
    nul = arch_field.find(b"\0")
    arch_bytes = arch_field if nul < 0 else arch_field[:nul]
    if nul >= 0 and any(arch_field[nul:]):
        raise AttributionError(f"{path}: target-info architecture has nonzero padding")
    try:
        architecture = normalize_architecture(arch_bytes.decode("ascii"))
    except (UnicodeDecodeError, AttributionError) as exc:
        raise AttributionError(
            f"{path}: invalid target-info architecture {arch_bytes!r}"
        ) from exc
    if reserved != 0:
        raise AttributionError(f"{path}: target-info reserved field is nonzero")

    legacy = (
        bin_type == 1
        and abi_type == 0
        and e_type == 0
        and e_machine == 0
        and e_flags == 0
    )
    if legacy:
        return architecture, None, "legacy-unspecified", False

    expected = {
        ("x86_64", 2): (0x3E, "sysv", "elf64-little"),
        ("x86_64", 6): (0x8664, "win64", "coff64"),
        ("aarch64", 2): (0xB7, "aapcs64", "elf64-little"),
    }.get((architecture, bin_type))
    if expected is None:
        raise AttributionError(
            f"{path}: unsupported target-info architecture/format "
            f"{architecture}/{bin_type}"
        )
    machine, abi, target_format = expected
    if abi_type != 0 or e_type != 1 or e_machine != machine or e_flags != 0:
        raise AttributionError(
            f"{path}: inconsistent target-info for {architecture}/{target_format} "
            f"(abi_type={abi_type}, e_type={e_type}, "
            f"e_machine={e_machine:#x}, e_flags={e_flags:#x})"
        )
    return architecture, abi, target_format, True


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
    if version not in SUPPORTED_AOT_VERSIONS:
        raise AttributionError(
            f"{path}: incompatible aot_version={version}; tool supports "
            f"{sorted(SUPPORTED_AOT_VERSIONS)}"
        )

    pos = 8
    offsets = None
    text_size = None
    text_file_offset = None
    target = None
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
        if section_type == SEC_TARGET_INFO:
            if target is not None:
                raise AttributionError(f"{path}: duplicate target-info section")
            target = _parse_target_info(path, data[pos:end])
        elif section_type == SEC_TEXT:
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

    if target is None:
        raise AttributionError(f"{path}: missing target-info section")
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
    return CwasmInfo(
        offsets,
        text_size,
        text_file_offset,
        data,
        version,
        target[0],
        target[1],
        target[2],
        target[3],
    )


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
        [perf_binary(), "script", "-i", perf, "--show-mmap-events"],
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
            perf_binary(),
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
    if counts and total:
        return counts, total

    # Newer perf releases can omit unresolved per-address rows from
    # `perf report` even with `--percent-limit 0`. `perf script` still emits
    # one self IP per sample, and the caller filters those IPs against the
    # exact anonymous text mapping selected from mmap events.
    script = _run_checked(
        [
            perf_binary(),
            "script",
            "-i",
            perf,
            "-F",
            "ip,dso",
        ],
        "perf script self samples",
    )
    counts, total = {}, 0
    row = re.compile(r"^\s*(?:0x)?([0-9a-fA-F]+)\s+")
    for line in script.splitlines():
        match = row.match(line)
        if not match:
            continue
        total += 1
        ip = int(match.group(1), 16)
        counts[ip] = counts.get(ip, 0) + 1
    return counts, total


def system_page_size():
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        page_size = 0
    if not isinstance(page_size, int) or page_size <= 0:
        raise AttributionError("cannot determine the host page size")
    return page_size


def expected_text_mapping_size(text_size, page_size):
    if text_size <= 0:
        raise AttributionError("cwasm text size must be positive")
    if page_size <= 0 or page_size & (page_size - 1):
        raise AttributionError(f"invalid host page size {page_size}")
    return (text_size + page_size - 1) & -page_size


def _render_mappings(maps):
    return ", ".join(f"{base:#x}/{size:#x}" for base, size in maps) or "none"


def select_text_mapping(perf, text_size, explicit_base=None, page_size=None):
    page_size = page_size or system_page_size()
    expected_size = expected_text_mapping_size(text_size, page_size)
    maps = jit_exec_mmaps(perf)
    if explicit_base is not None:
        try:
            base = int(explicit_base, 16)
        except ValueError as exc:
            raise AttributionError(
                f"manual --base must be hexadecimal, got {explicit_base!r}"
            ) from exc
        matches = [(candidate, size) for candidate, size in maps if candidate == base]
        if len(matches) != 1:
            raise AttributionError(
                f"manual --base {base:#x} must identify exactly one anonymous "
                f"executable mmap; candidates: {_render_mappings(maps)}"
            )
        return TextMappingSelection(
            base=matches[0][0],
            size=matches[0][1],
            text_size=text_size,
            page_size=page_size,
            expected_size=expected_size,
            candidates=maps,
            authoritative=False,
            override=explicit_base,
        )

    matches = [(base, size) for base, size in maps if size == expected_size]
    if len(matches) != 1:
        raise AttributionError(
            f"expected exactly one anonymous executable mmap of page-rounded "
            f"size {expected_size:#x} for text_size={text_size:#x} and "
            f"page_size={page_size:#x}; found {len(matches)} exact matches "
            f"among candidates: {_render_mappings(maps)}"
        )
    return TextMappingSelection(
        base=matches[0][0],
        size=matches[0][1],
        text_size=text_size,
        page_size=page_size,
        expected_size=expected_size,
        candidates=maps,
        authoritative=True,
        override=None,
    )


def select_text_base(perf, text_size, explicit_base=None, page_size=None):
    return select_text_mapping(
        perf, text_size, explicit_base, page_size
    ).base


def disassemble_blob(
    blob,
    virtual_address,
    scratch_dir,
    label,
    function_offset_base=0,
    architecture=None,
):
    architecture = normalize_architecture(architecture)
    scratch_dir = Path(scratch_dir)
    scratch = scratch_dir / f".aot-jit-attr-{os.getpid()}-{label}.bin"
    machine_args = (
        ["-m", "i386:x86-64", "-M", "intel"]
        if architecture == "x86_64"
        else ["-m", "aarch64"]
    )
    try:
        scratch.write_bytes(blob)
        output = _run_checked(
            [
                "objdump",
                "-D",
                "-b",
                "binary",
                *machine_args,
                f"--adjust-vma=0x{virtual_address:x}",
                str(scratch),
            ],
            "objdump",
        )
    finally:
        scratch.unlink(missing_ok=True)

    pattern = re.compile(
        r"^\s*([0-9a-fA-F]+):\s+"
        r"((?:(?:[0-9a-fA-F]{2}|[0-9a-fA-F]{8})\s+)+)\s*(.*)$"
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
                size=sum(len(token) // 2 for token in encoded),
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
    architecture=None,
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
                    architecture,
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
                architecture,
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
        base_match = re.search(
            r"(?<![a-z0-9_])(rbp|rsp)(?![a-z0-9_])", expression
        )
        if not base_match:
            continue
        base = base_match.group(1)
        simple = re.fullmatch(
            r"(rbp|rsp)(?:([+-])(0x[0-9a-f]+|\d+))?", expression
        )
        offset = None
        complex_address = True
        if simple:
            magnitude = int(simple.group(3), 0) if simple.group(3) else 0
            offset = -magnitude if simple.group(2) == "-" else magnitude
            complex_address = False
        kind = (
            "store"
            if mnemonic.startswith("mov") and operand_index == 0
            else "load"
        )
        return FrameOperand(kind, base, offset, complex_address)
    return None


def parse_aarch64_memory_operand(text, frame_only=False):
    """Identify an AArch64 load/store memory operand."""
    mnemonic, operands = _split_operands(text)
    load_mnemonics = {
        "ldr",
        "ldrb",
        "ldrh",
        "ldrsb",
        "ldrsh",
        "ldrsw",
        "ldur",
        "ldurb",
        "ldurh",
        "ldursb",
        "ldursh",
        "ldursw",
        "ldp",
        "ldpsw",
    }
    store_mnemonics = {
        "str",
        "strb",
        "strh",
        "stur",
        "sturb",
        "sturh",
        "stp",
    }
    if mnemonic not in load_mnemonics | store_mnemonics:
        return None
    for operand in operands:
        base_pattern = (
            r"(x29|fp|sp)"
            if frame_only
            else r"(x(?:30|[12][0-9]|[0-9])|fp|sp)"
        )
        bracket = re.search(
            rf"\[\s*{base_pattern}\b\s*"
            r"(?:,\s*#?(-?(?:0x[0-9a-f]+|\d+)))?",
            operand.lower(),
        )
        if not bracket:
            continue
        base = "x29" if bracket.group(1) in {"x29", "fp"} else "sp"
        if bracket.group(1) not in {"x29", "fp", "sp"}:
            base = bracket.group(1)
        offset = int(bracket.group(2), 0) if bracket.group(2) else 0
        complex_address = bool(
            re.search(
                r"\[\s*(?:x(?:30|[12][0-9]|[0-9])|fp|sp)\b\s*,"
                r"(?!\s*#?-?(?:0x[0-9a-f]+|\d+))",
                operand.lower(),
            )
        )
        return FrameOperand(
            "load" if mnemonic in load_mnemonics else "store",
            base,
            None if complex_address else offset,
            complex_address,
        )
    return None


def parse_aarch64_frame_operand(text):
    """Identify AArch64 x29/sp frame loads and stores."""
    return parse_aarch64_memory_operand(text, frame_only=True)


def instruction_frame_operand(text, architecture=None):
    architecture = normalize_architecture(architecture)
    return (
        parse_frame_operand(text)
        if architecture == "x86_64"
        else parse_aarch64_frame_operand(text)
    )


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


def normalized_code_sha256_v2(code, relocations, architecture):
    normalized = bytearray(code)
    prior_end = 0
    for relocation in relocations:
        if not isinstance(relocation, dict):
            raise AttributionError(
                "normalized_relocations entries must be objects"
            )
        start = _require_int(relocation, "native_start")
        end = _require_int(relocation, "native_end")
        kind = relocation.get("kind")
        if start < prior_end or end != start + 4 or end > len(normalized):
            raise AttributionError(
                f"invalid/overlapping normalized relocation [{start},{end})"
            )
        if architecture == "aarch64":
            expected_opcode = {
                "aarch64_call_imm26": 0x94000000,
                "aarch64_tail_call_imm26": 0x14000000,
            }.get(kind)
            if expected_opcode is None:
                raise AttributionError(
                    f"invalid AArch64 relocation kind {kind!r}"
                )
            word = struct.unpack_from("<I", normalized, start)[0]
            if word & 0xFC000000 != expected_opcode:
                raise AttributionError(
                    f"AArch64 relocation at {start} does not match {kind}"
                )
            struct.pack_into("<I", normalized, start, word & 0xFC000000)
        elif architecture == "x86_64":
            if kind != "x86_64_call_rel32" or start == 0:
                raise AttributionError(
                    f"invalid x86_64 relocation kind/range at {start}"
                )
            if normalized[start - 1] != 0xE8:
                raise AttributionError(
                    f"x86_64 relocation at {start} is not CALL rel32"
                )
            normalized[start:end] = b"\0\0\0\0"
        else:
            raise AttributionError(
                f"unsupported metadata architecture {architecture!r}"
            )
        prior_end = end
    return hashlib.sha256(normalized).hexdigest()


def _signed_bits(value, bits):
    sign = 1 << (bits - 1)
    return (value ^ sign) - sign


def _decode_aarch64_memory(word):
    rn = (word >> 5) & 31
    rt = word & 31
    top = word & 0xFFC00000
    unsigned = {
        0xF9400000: ("load", 8, 8),
        0xF9000000: ("store", 8, 8),
        0xB9400000: ("load", 4, 4),
        0xB9800000: ("load", 4, 4),
        0xB9000000: ("store", 4, 4),
        0x39400000: ("load", 1, 1),
        0x39800000: ("load", 1, 1),
        0x39C00000: ("load", 1, 1),
        0x39000000: ("store", 1, 1),
        0x79400000: ("load", 2, 2),
        0x79800000: ("load", 2, 2),
        0x79C00000: ("load", 2, 2),
        0x79000000: ("store", 2, 2),
    }.get(top)
    if unsigned:
        kind, width, scale = unsigned
        return {
            "kind": kind,
            "rn": rn,
            "offset": ((word >> 10) & 0xFFF) * scale,
            "mode": "unsigned_scaled",
            "width": width,
            "pair": False,
            "registers": [(rt, "gpr")],
            "load_gprs": [rt] if kind == "load" else [],
            "writeback": None,
        }

    unscaled_top = word & 0xFFE00C00
    unscaled = {
        0xF8000000: ("store", 8, "gpr"),
        0xF8400000: ("load", 8, "gpr"),
        0xB8000000: ("store", 4, "gpr"),
        0xB8400000: ("load", 4, "gpr"),
        0xB8800000: ("load", 4, "gpr"),
        0x38000000: ("store", 1, "gpr"),
        0x38400000: ("load", 1, "gpr"),
        0x38800000: ("load", 1, "gpr"),
        0x38C00000: ("load", 1, "gpr"),
        0x78000000: ("store", 2, "gpr"),
        0x78400000: ("load", 2, "gpr"),
        0x78800000: ("load", 2, "gpr"),
        0x78C00000: ("load", 2, "gpr"),
        0x3C800000: ("store", 16, "simd"),
        0x3CC00000: ("load", 16, "simd"),
        0xBC000000: ("store", 4, "simd"),
        0xBC400000: ("load", 4, "simd"),
        0xFC000000: ("store", 8, "simd"),
        0xFC400000: ("load", 8, "simd"),
    }.get(unscaled_top)
    if unscaled:
        kind, width, register_class = unscaled
        return {
            "kind": kind,
            "rn": rn,
            "offset": _signed_bits((word >> 12) & 0x1FF, 9),
            "mode": "signed_unscaled",
            "width": width,
            "pair": False,
            "registers": [(rt, register_class)],
            "load_gprs": [rt] if kind == "load" and register_class == "gpr" else [],
            "writeback": None,
        }

    pair = {
        0xA9000000: ("store", 8, "signed_pair", False),
        0xA9400000: ("load", 8, "signed_pair", False),
        0xA9800000: ("store", 8, "pre_index", True),
        0xA8C00000: ("load", 8, "post_index", True),
        0x29000000: ("store", 4, "signed_pair", False),
        0x29400000: ("load", 4, "signed_pair", False),
    }.get(top)
    if pair:
        kind, width, mode, writeback = pair
        displacement = _signed_bits((word >> 15) & 0x7F, 7) * width
        rt2 = (word >> 10) & 31
        return {
            "kind": kind,
            "rn": rn,
            "offset": 0 if mode == "post_index" else displacement,
            "mode": mode,
            "width": width,
            "pair": True,
            "registers": [(rt, "gpr"), (rt2, "gpr")],
            "load_gprs": [rt, rt2] if kind == "load" else [],
            "writeback": displacement if writeback else None,
        }

    simd_top = word & 0xFFFFFC00
    simd = {
        0x3DC00000: ("load", 16),
        0x3D800000: ("store", 16),
        0xBD400000: ("load", 4),
        0xFD400000: ("load", 8),
    }.get(simd_top)
    if simd:
        kind, width = simd
        return {
            "kind": kind,
            "rn": rn,
            "offset": 0,
            "mode": "simd_zero_offset",
            "width": width,
            "pair": False,
            "registers": [(rt, "simd")],
            "load_gprs": [],
            "writeback": None,
        }
    return None


def _decode_aarch64_frame_accesses(code, inline_data_ranges):
    relations = [None] * 32
    constants = [None] * 32
    accesses = {}
    inline_offsets = {
        offset
        for item in inline_data_ranges
        for offset in range(item["native_start"], item["native_end"], 4)
    }

    def relation_for(reg):
        if reg == 29:
            return ("x29", 0)
        if reg == 31:
            return ("sp", 0)
        return relations[reg]

    for native_start in range(0, len(code), 4):
        if native_start + 4 > len(code):
            raise AttributionError("AArch64 function code is not word-aligned")
        if native_start in inline_offsets:
            relations = [None] * 32
            constants = [None] * 32
            continue
        word = struct.unpack_from("<I", code, native_start)[0]
        memory = _decode_aarch64_memory(word)
        if memory is not None:
            relation = relation_for(memory["rn"])
            if relation is not None:
                base, relation_offset = relation
                mode = (
                    memory["mode"]
                    if memory["rn"] in {29, 31}
                    else "materialized"
                )
                components = []
                for index, (register, register_class) in enumerate(
                    memory["registers"]
                ):
                    components.append(
                        {
                            "base": base,
                            "frame_offset": (
                                relation_offset
                                + memory["offset"]
                                + index * memory["width"]
                            ),
                            "width": memory["width"],
                            "data_register": register,
                            "data_register_class": register_class,
                        }
                    )
                accesses[native_start] = {
                    "kind": memory["kind"],
                    "encoded_base": (
                        "sp" if memory["rn"] == 31 else f"x{memory['rn']}"
                    ),
                    "encoded_offset": memory["offset"],
                    "addressing_mode": mode,
                    "components": components,
                }
            for register in memory["load_gprs"]:
                relations[register] = None
                constants[register] = None
            if memory["writeback"] is not None:
                relations[memory["rn"]] = None
                constants[memory["rn"]] = None
            continue

        move_top = word & 0xFF800000
        if move_top in {0xD2800000, 0xF2800000}:
            register = word & 31
            value = (word >> 5) & 0xFFFF
            shift = ((word >> 21) & 3) * 16
            if move_top == 0xF2800000 and constants[register] is not None:
                mask = ~(0xFFFF << shift) & ((1 << 64) - 1)
                constants[register] = (
                    constants[register] & mask
                ) | (value << shift)
            elif move_top == 0xD2800000:
                constants[register] = value << shift
            else:
                constants[register] = None
            relations[register] = None
            continue

        if word & 0x1F000000 == 0x11000000 and (word >> 31) & 1:
            rd, rn = word & 31, (word >> 5) & 31
            immediate = ((word >> 10) & 0xFFF) << (
                12 if (word >> 22) & 1 else 0
            )
            displacement = -immediate if (word >> 30) & 1 else immediate
            relation = relation_for(rn)
            relations[rd] = (
                (relation[0], relation[1] + displacement)
                if relation is not None
                else None
            )
            constants[rd] = None
            continue

        if word & 0xFFE0FC00 == 0x8B000000:
            rd, rn, rm = word & 31, (word >> 5) & 31, (word >> 16) & 31
            rn_relation, rm_relation = relation_for(rn), relation_for(rm)
            if rn_relation is not None and constants[rm] is not None:
                relations[rd] = (
                    rn_relation[0],
                    rn_relation[1] + constants[rm],
                )
            elif rm_relation is not None and constants[rn] is not None:
                relations[rd] = (
                    rm_relation[0],
                    rm_relation[1] + constants[rn],
                )
            else:
                relations[rd] = None
            constants[rd] = None
            continue

        if (
            word & 0x7C000000 == 0x14000000
            or word & 0xFF000010 == 0x54000000
            or word & 0x7E000000 == 0x34000000
            or word & 0x7E000000 == 0x36000000
            or word & 0xFFFFFC1F in {0xD61F0000, 0xD63F0000, 0xD65F0000}
        ):
            for index in range(29):
                relations[index] = None
                constants[index] = None
            continue

        rd = word & 31
        if rd < 29:
            relations[rd] = None
            constants[rd] = None
    return accesses


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
    artifact_info,
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
    schema_version = raw.get("schema_version")
    if schema_version not in SUPPORTED_FRAME_SCHEMA_VERSIONS:
        raise AttributionError(
            f"{path}: incompatible frame schema_version="
            f"{schema_version!r}; expected one of "
            f"{sorted(SUPPORTED_FRAME_SCHEMA_VERSIONS)}"
        )
    architecture = raw.get("architecture")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise AttributionError(
            f"{path}: unsupported frame metadata architecture {architecture!r}"
        )
    if schema_version == 1 and architecture != "x86_64":
        raise AttributionError(
            f"{path}: schema-v1 frame metadata is x86_64-only"
        )
    if architecture != artifact_info.architecture:
        raise AttributionError(
            f"{path}: metadata architecture {architecture} does not match "
            f"artifact target {artifact_info.architecture}"
        )
    if not artifact_info.target_verified or artifact_info.abi is None:
        raise AttributionError(
            f"{path}: artifact target-info is legacy/unspecified; "
            "authoritative frame ABI validation is unavailable"
        )
    if raw.get("abi") != artifact_info.abi:
        raise AttributionError(
            f"{path}: metadata ABI {raw.get('abi')!r} does not match "
            f"artifact ABI {artifact_info.abi!r}"
        )
    _require_int(raw, "module")
    valid_abis = (
        {"sysv", "win64"} if architecture == "x86_64" else {"aapcs64"}
    )
    if raw.get("abi") not in valid_abis:
        raise AttributionError(
            f"{path}: frame metadata abi must be one of {sorted(valid_abis)}"
        )
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
    if schema_version == 1:
        rel32_offsets = _require_list(raw, "direct_call_rel32_offsets")
        actual_hash = normalized_code_sha256(function_code, rel32_offsets)
    else:
        relocations = _require_list(raw, "normalized_relocations")
        actual_hash = normalized_code_sha256_v2(
            function_code, relocations, architecture
        )
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
    if layout.get("frame_pointer") not in {"rbp", "x29"}:
        raise AttributionError(f"{path}: invalid frame pointer")
    if (
        layout["frame_size"] <= 0
        or layout["spill_stride"] == 0
        or layout["spill_slots"] < 0
        or layout["local_count"] < 0
        or layout["explicit_storage_slots"] < 0
    ):
        raise AttributionError(f"{path}: invalid frame layout dimensions")
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
    if schema_version == 2:
        regions = _require_list(raw, "frame_regions")
        prior_region_end = None
        for region in regions:
            if not isinstance(region, dict):
                raise AttributionError(
                    f"{path}: frame_regions entries must be objects"
                )
            region_start = _require_int(region, "start")
            region_end = _require_int(region, "end")
            if (
                region_end <= region_start
                or region_start < 0
                or region_end > layout["frame_size"]
                or (
                    prior_region_end is not None
                    and region_start < prior_region_end
                )
                or region.get("origin") not in FRAME_ORIGINS
                or not isinstance(region.get("detail"), str)
            ):
                raise AttributionError(
                    f"{path}: invalid/overlapping frame region "
                    f"[{region_start},{region_end})"
                )
            prior_region_end = region_end
    else:
        regions = []

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
        expected_offset = layout["spill_base"] + slot * layout["spill_stride"]
        type_slots = 2 if value.get("value_type") == "v128" else 1
        if (
            vreg in value_by_vreg
            or slot < 0
            or slot_count <= 0
            or slot + slot_count > layout["spill_slots"]
            or value["frame_offset"] != expected_offset
            or slot_count != type_slots
        ):
            raise AttributionError(
                f"{path}: invalid allocator slot/offset/type for vreg={vreg}"
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

    def validate_component(component, kind, start):
        nonlocal emitted_loads, emitted_stores
        if not isinstance(component, dict):
            raise AttributionError(
                f"{path}: access components must be objects"
            )
        base = component.get("base")
        valid_bases = (
            {"rbp", "rsp"} if architecture == "x86_64" else {"x29", "sp"}
        )
        if base not in valid_bases:
            raise AttributionError(
                f"{path}: invalid effective frame base {base!r} at {start}"
            )
        frame_offset = _require_int(component, "frame_offset")
        width = _require_int(component, "width")
        if width <= 0:
            raise AttributionError(
                f"{path}: invalid frame access width at {start}"
            )
        origin = component.get("origin")
        if origin not in FRAME_ORIGINS:
            raise AttributionError(
                f"{path}: invalid frame origin {origin!r} at offset {start}"
            )
        if architecture == "aarch64" and schema_version == 2:
            allowed_details = {
                "allocator_spill": {"allocator_slot"},
                "wasm_local_or_phi": {"wasm_local_or_lowered_phi"},
                "explicit_frame_storage": {
                    "hidden_return_pointer",
                    "call_result_scratch",
                },
                "fixed_runtime_frame_state": {
                    "saved_frame_pointer",
                    "return_address",
                    "reserved_vmctx",
                    "caller_saved_register",
                    "callee_saved_register",
                    "incoming_abi_argument",
                    "prologue_saved_fp_lr",
                    "epilogue_restored_fp_lr",
                    "outgoing_abi_frame",
                },
                "unknown": {"unclassified_frame_access"},
            }
            if component.get("detail") not in allowed_details[origin]:
                raise AttributionError(
                    f"{path}: invalid frame detail "
                    f"{component.get('detail')!r} for {origin} at {start}"
                )
            data_register = component.get("data_register")
            register_class = component.get("data_register_class")
            if (
                isinstance(data_register, bool)
                or not isinstance(data_register, int)
                or not 0 <= data_register <= 31
                or register_class not in {"gpr", "simd"}
            ):
                raise AttributionError(
                    f"{path}: invalid data-register identity at {start}"
                )
            if base == "x29" and 0 <= frame_offset < layout["frame_size"]:
                containing = [
                    region
                    for region in regions
                    if region["start"] <= frame_offset
                    and frame_offset + width <= region["end"]
                ]
                if not containing:
                    if origin != "unknown":
                        raise AttributionError(
                            f"{path}: frame access [{frame_offset},"
                            f"{frame_offset + width}) is outside declared regions"
                        )
                elif containing[0]["origin"] != origin:
                    raise AttributionError(
                        f"{path}: frame access origin {origin} disagrees with "
                        f"region {containing[0]['origin']} at {start}"
                    )
        local_index = component.get("local_index")
        if origin == "wasm_local_or_phi":
            if (
                isinstance(local_index, bool)
                or not isinstance(local_index, int)
                or not 0 <= local_index < layout["local_count"]
            ):
                raise AttributionError(
                    f"{path}: local frame access at {start} lacks valid local_index"
                )
        explicit_slot = component.get("explicit_slot")
        if origin == "explicit_frame_storage":
            if (
                isinstance(explicit_slot, bool)
                or not isinstance(explicit_slot, int)
                or not 0 <= explicit_slot < layout["explicit_storage_slots"]
            ):
                raise AttributionError(
                    f"{path}: explicit frame access at {start} lacks valid slot"
                )
            if architecture == "aarch64" and (
                frame_offset
                != layout["explicit_storage_first_offset"] + explicit_slot * 8
            ):
                raise AttributionError(
                    f"{path}: explicit slot/offset mismatch at {start}"
                )
        vreg = component.get("vreg")
        ambiguous = component.get("vreg_ambiguous")
        if vreg is not None and (
            isinstance(vreg, bool) or not isinstance(vreg, int)
        ):
            raise AttributionError(f"{path}: vreg must be integer or null")
        if not isinstance(ambiguous, bool) or (
            vreg is not None and ambiguous
        ):
            raise AttributionError(
                f"{path}: invalid vreg/vreg_ambiguous combination at {start}"
            )
        if origin != "allocator_spill":
            return
        slot = component.get("slot")
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise AttributionError(
                f"{path}: allocator access at {start} lacks integer slot"
            )
        candidates = values_by_slot.get(slot, [])
        if not candidates:
            raise AttributionError(
                f"{path}: allocator access at {start} names unassigned slot {slot}"
            )
        stride = abs(layout["spill_stride"])
        occupied_slots = (width + stride - 1) // stride
        if slot + occupied_slots > layout["spill_slots"]:
            raise AttributionError(
                f"{path}: allocator component at {start} exceeds spill layout"
            )
        candidates = [
            item
            for item in candidates
            if slot + occupied_slots <= item["slot"] + item["slot_count"]
        ]
        if not candidates:
            raise AttributionError(
                f"{path}: allocator component at {start} exceeds value coverage"
            )
        expected_offset = layout["spill_base"] + slot * layout["spill_stride"]
        if frame_offset != expected_offset:
            raise AttributionError(
                f"{path}: allocator slot/offset mismatch at {start}: "
                f"slot {slot} maps to {expected_offset}, not {frame_offset}"
            )
        if vreg is not None and all(
            item["vreg"] != vreg for item in candidates
        ):
            raise AttributionError(
                f"{path}: access vreg={vreg} does not occupy slot {slot}"
            )
        if vreg is not None:
            value = value_by_vreg[vreg]
            remaining_slots = value["slot"] + value["slot_count"] - slot
            if remaining_slots <= 0 or width > remaining_slots * stride:
                raise AttributionError(
                    f"{path}: allocator component at {start} exceeds vreg "
                    f"{vreg} slot coverage"
                )
        if len(candidates) > 1 and vreg is None and not ambiguous:
            raise AttributionError(
                f"{path}: reused slot {slot} lacks a resolved vreg or "
                "vreg_ambiguous=true"
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

    for access in accesses:
        if not isinstance(access, dict):
            raise AttributionError(f"{path}: accesses entries must be objects")
        start = _require_int(access, "native_start")
        end = _require_int(access, "native_end")
        kind = access.get("kind")
        base = access.get("base")
        origin = access.get("origin")
        _require_int(access, "frame_offset")
        width = _require_int(access, "width")
        if start < prior_end or end <= start or end > code_size:
            raise AttributionError(
                f"{path}: overlapping/out-of-range native access [{start},{end})"
            )
        if start in access_by_start:
            raise AttributionError(f"{path}: duplicate access at native offset {start}")
        valid_bases = (
            {"rbp", "rsp"} if architecture == "x86_64" else {"x29", "sp"}
        )
        if kind not in {"load", "store"} or base not in valid_bases:
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
        if schema_version == 1:
            validate_component(access, kind, start)
        else:
            encoded_base = access.get("encoded_base")
            encoded_offset = access.get("encoded_offset")
            addressing_mode = access.get("addressing_mode")
            valid_encoded_bases = (
                {"rbp", "rsp"}
                if architecture == "x86_64"
                else {"sp", *(f"x{index}" for index in range(31))}
            )
            valid_addressing_modes = {
                "unsigned_scaled",
                "signed_unscaled",
                "signed_pair",
                "pre_index",
                "post_index",
                "simd_zero_offset",
                "materialized",
            }
            if encoded_base not in valid_encoded_bases or (
                isinstance(encoded_offset, bool)
                or not isinstance(encoded_offset, int)
            ) or addressing_mode not in valid_addressing_modes:
                raise AttributionError(
                    f"{path}: schema-v2 access at {start} lacks encoded address"
                )
            components = _require_list(access, "components")
            if not 1 <= len(components) <= 2:
                raise AttributionError(
                    f"{path}: access at {start} must have one or two components"
                )
            for component in components:
                validate_component(component, kind, start)
            if sum(_require_int(item, "width") for item in components) != width:
                raise AttributionError(
                    f"{path}: component widths disagree at {start}"
                )
            if len(components) == 2 and (
                components[1].get("base") != components[0].get("base")
                or components[1].get("frame_offset")
                != components[0].get("frame_offset")
                + components[0].get("width")
            ):
                raise AttributionError(
                    f"{path}: pair components are not contiguous at {start}"
                )
            if (len(components) == 2) != (
                addressing_mode in {"signed_pair", "pre_index", "post_index"}
            ):
                raise AttributionError(
                    f"{path}: component count/addressing mode mismatch at {start}"
                )
            if len(components) == 2 and (
                components[0]["width"] != components[1]["width"]
            ):
                raise AttributionError(
                    f"{path}: pair component widths disagree at {start}"
                )
            component_origins = {item.get("origin") for item in components}
            expected_origin = (
                next(iter(component_origins))
                if len(component_origins) == 1
                else "unknown"
            )
            component_details = {item.get("detail") for item in components}
            expected_detail = (
                next(iter(component_details))
                if len(component_details) == 1
                else "mixed_pair_frame_access"
            )
            if origin != expected_origin:
                raise AttributionError(
                    f"{path}: top-level/component origin mismatch at {start}"
                )
            if access.get("detail") != expected_detail:
                raise AttributionError(
                    f"{path}: top-level/component detail mismatch at {start}"
                )
            if access.get("base") != components[0].get("base") or (
                access.get("frame_offset")
                != components[0].get("frame_offset")
            ):
                raise AttributionError(
                    f"{path}: top-level/component address mismatch at {start}"
                )
            if len(components) == 1:
                for key in (
                    "slot",
                    "local_index",
                    "explicit_slot",
                    "vreg",
                    "vreg_ambiguous",
                    "defining_opcode",
                    "source_class",
                    "rematerialization_eligible",
                ):
                    if access.get(key) != components[0].get(key):
                        raise AttributionError(
                            f"{path}: top-level/component {key} mismatch at "
                            f"{start}"
                        )
        access_by_start[start] = access
        prior_end = end

    if schema_version == 2 and architecture == "aarch64":
        decoded_accesses = _decode_aarch64_frame_accesses(
            function_code, inline_data_ranges
        )
        if set(decoded_accesses) != set(access_by_start):
            raise AttributionError(
                f"{path}: emitted frame-access offsets disagree with native "
                f"code ({sorted(access_by_start)} != {sorted(decoded_accesses)})"
            )
        for start, decoded in decoded_accesses.items():
            access = access_by_start[start]
            for key in (
                "kind",
                "encoded_base",
                "encoded_offset",
                "addressing_mode",
            ):
                if access.get(key) != decoded[key]:
                    raise AttributionError(
                        f"{path}: native {key} mismatch at {start}: "
                        f"{access.get(key)!r} != {decoded[key]!r}"
                    )
            components = access["components"]
            if len(components) != len(decoded["components"]):
                raise AttributionError(
                    f"{path}: native component count mismatch at {start}"
                )
            for component, expected in zip(
                components, decoded["components"]
            ):
                for key in (
                    "base",
                    "frame_offset",
                    "width",
                    "data_register",
                    "data_register_class",
                ):
                    if component.get(key) != expected[key]:
                        raise AttributionError(
                            f"{path}: native component {key} mismatch at "
                            f"{start}: {component.get(key)!r} != "
                            f"{expected[key]!r}"
                        )
            if decoded["components"][0]["base"] == "sp":
                registers = [
                    (item["data_register"], item["data_register_class"])
                    for item in decoded["components"]
                ]
                if registers == [(29, "gpr"), (30, "gpr")]:
                    expected_detail = (
                        "prologue_saved_fp_lr"
                        if decoded["kind"] == "store"
                        else "epilogue_restored_fp_lr"
                    )
                else:
                    expected_detail = "outgoing_abi_frame"
                if any(
                    item.get("detail") != expected_detail
                    for item in components
                ):
                    raise AttributionError(
                        f"{path}: SP frame detail mismatch at {start}; "
                        f"expected {expected_detail}"
                    )

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
    architecture = metadata.raw["architecture"]
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
        operand = (
            parse_frame_operand(instruction.text)
            if architecture == "x86_64"
            else parse_aarch64_memory_operand(instruction.text)
        )
        if operand is None:
            raise AttributionError(
                f"metadata access at +0x{start:x} is not a frame load/store: "
                f"{instruction.text}"
            )
        expected_base = (
            access["base"]
            if metadata.raw["schema_version"] == 1
            else access["encoded_base"]
        )
        expected_offset = (
            access["frame_offset"]
            if metadata.raw["schema_version"] == 1
            else access["encoded_offset"]
        )
        if operand.kind != access["kind"] or operand.base != expected_base:
            raise AttributionError(
                f"metadata access kind/base mismatch at +0x{start:x}: "
                f"{access['kind']}/{expected_base} vs "
                f"{operand.kind}/{operand.base}"
            )
        if operand.offset is not None and operand.offset != expected_offset:
            raise AttributionError(
                f"metadata frame offset mismatch at +0x{start:x}: "
                f"{expected_offset} vs {operand.offset}"
            )


def classify_x86_64_basic(text):
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


def _is_aarch64_register(value):
    return bool(
        re.fullmatch(
            r"(?:[xw](?:[0-9]|[12][0-9]|30)|sp|xzr|wzr|"
            r"[bhsdqv](?:[0-9]|[12][0-9]|3[01]))",
            value.strip().lower(),
        )
    )


def classify_aarch64_basic(text):
    mnemonic, operands = _split_operands(text)
    frame = parse_aarch64_frame_operand(text)
    if frame:
        return f"frame_{frame.kind}_unattributed"
    if mnemonic in {"bl", "blr"}:
        return "call"
    if mnemonic == "br":
        return "indirect_dispatch"
    if mnemonic == "b":
        return "direct_branch"
    if (
        mnemonic.startswith("b.")
        or mnemonic in {"cbz", "cbnz", "tbz", "tbnz"}
    ):
        return "cond_branch"
    load_store = mnemonic.startswith(("ldr", "ldur", "str", "stur", "ldp", "stp"))
    if load_store and any("[" in operand for operand in operands):
        if re.search(r"\[\s*x20(?:\s*,|\s*\])", text, re.IGNORECASE):
            return "linear_memory"
        return "mem_access"
    if (
        mnemonic == "mov"
        and len(operands) == 2
        and _is_aarch64_register(operands[0])
        and _is_aarch64_register(operands[1])
    ):
        return "regmov"
    if mnemonic in {
        "add",
        "adds",
        "sub",
        "subs",
        "adc",
        "adcs",
        "sbc",
        "sbcs",
        "and",
        "ands",
        "orr",
        "eor",
        "bic",
        "bics",
        "madd",
        "msub",
        "mul",
        "mneg",
        "udiv",
        "sdiv",
        "lsl",
        "lsr",
        "asr",
        "ror",
        "extr",
        "ubfm",
        "sbfm",
        "bfm",
        "uxtb",
        "uxth",
        "sxtb",
        "sxth",
        "sxtw",
        "cmp",
        "cmn",
        "tst",
        "csel",
        "csinc",
        "csinv",
        "csneg",
        "adr",
        "adrp",
    }:
        return "alu"
    return "other"


def classify_basic(text, architecture=None):
    architecture = normalize_architecture(architecture)
    return (
        classify_x86_64_basic(text)
        if architecture == "x86_64"
        else classify_aarch64_basic(text)
    )


def classify_instruction(instruction, metadata=None, architecture=None):
    architecture = normalize_architecture(architecture)
    if metadata:
        access = metadata.access_by_start.get(instruction.offset)
        if access:
            return f"{access['origin']}_{access['kind']}"
        frame = instruction_frame_operand(instruction.text, architecture)
        if frame:
            return f"unknown_frame_{frame.kind}"
    return classify_basic(instruction.text, architecture)


def classify_instruction_stream(instructions, metadata=None, architecture=None):
    architecture = normalize_architecture(architecture)
    classes = [
        classify_instruction(instruction, metadata, architecture)
        for instruction in instructions
    ]
    if architecture != "aarch64":
        return classes

    for index in range(len(instructions) - 1):
        mnemonic, _ = _split_operands(instructions[index].text)
        next_mnemonic, _ = _split_operands(instructions[index + 1].text)
        if mnemonic not in {"cmp", "cmn"} or next_mnemonic not in {
            "b.hi",
            "b.hs",
            "b.lo",
            "b.ls",
        }:
            continue
        window = " ".join(
            item.text for item in instructions[max(0, index - 3) : index]
        )
        if re.search(
            r"\[\s*x19\s*,\s*#?(?:0x)?0*8\s*\]",
            window,
            re.IGNORECASE,
        ):
            classes[index] = "bounds_cmp"
            classes[index + 1] = "bounds_branch"
    return classes


def _percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else 0.0


def require_attribution_coverage(attributed, total, minimum_pct):
    coverage = _percent(attributed, total)
    if coverage < minimum_pct:
        raise AttributionError(
            f"cwasm attribution coverage {coverage:.4f}% "
            f"({attributed}/{total}) is below required {minimum_pct:.4f}%"
        )
    return coverage


def build_frame_summary(instructions, counts, metadata):
    architecture = metadata.raw.get("architecture", "x86_64")
    schema_version = metadata.raw.get("schema_version", 1)
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
        access = metadata.access_by_start.get(instruction.offset)
        operand = instruction_frame_operand(instruction.text, architecture)
        if access is None and operand is None:
            continue
        samples = counts.get(instruction.address, 0)
        frame_instruction_count += 1
        frame_samples += samples
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

        components = (
            access["components"] if schema_version == 2 else [access]
        )
        allocator_components = [
            component
            for component in components
            if component["origin"] == "allocator_spill"
        ]
        component_keys = []
        for component in allocator_components:
            slot = component["slot"]
            vreg = component.get("vreg")
            key = (slot, vreg)
            component_keys.append(key)
            if key not in contributors:
                value = (
                    metadata.value_by_vreg.get(vreg)
                    if vreg is not None
                    else None
                )
                contributors[key] = {
                    "slot": slot,
                    "frame_offset": component["frame_offset"],
                    "vreg": vreg,
                    "vreg_ambiguous": component["vreg_ambiguous"],
                    "candidate_vregs": [
                        candidate["vreg"]
                        for candidate in metadata.values_by_slot.get(slot, [])
                    ],
                    "defining_opcode": component.get("defining_opcode"),
                    "source_class": component.get("source_class"),
                    "rematerialization_eligible": component.get(
                        "rematerialization_eligible"
                    ),
                    "source_reload_count": (
                        value.get("reload_count") if value else None
                    ),
                    "source_store_count": (
                        value.get("store_count") if value else None
                    ),
                    "source_ir_use_count": (
                        value.get("ir_use_count") if value else None
                    ),
                    "source_ir_def_count": (
                        value.get("ir_def_count") if value else None
                    ),
                    "static_loads": 0,
                    "static_stores": 0,
                    "samples": 0,
                }
            contributors[key][f"static_{access['kind']}s"] += 1

        unique_keys = list(dict.fromkeys(component_keys))
        if len(unique_keys) == 1:
            contributors[unique_keys[0]]["samples"] += samples
        elif unique_keys:
            paired_key = ("paired", instruction.offset)
            contributors[paired_key] = {
                "slot": min(component["slot"] for component in allocator_components),
                "frame_offset": min(
                    component["frame_offset"]
                    for component in allocator_components
                ),
                "vreg": None,
                "vreg_ambiguous": True,
                "candidate_vregs": sorted(
                    {
                        component["vreg"]
                        for component in allocator_components
                        if component.get("vreg") is not None
                    }
                ),
                "paired_components": [
                    {
                        "slot": component["slot"],
                        "frame_offset": component["frame_offset"],
                        "vreg": component.get("vreg"),
                    }
                    for component in allocator_components
                ],
                "defining_opcode": None,
                "source_class": "paired_allocator_access",
                "rematerialization_eligible": None,
                "source_reload_count": None,
                "source_store_count": None,
                "source_ir_use_count": None,
                "source_ir_def_count": None,
                "static_loads": 0,
                "static_stores": 0,
                "samples": samples,
            }

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
    parser.add_argument(
        "--arch",
        choices=sorted(SUPPORTED_ARCHITECTURES),
        default=normalize_architecture(),
        help="generated-code architecture (default: current host)",
    )
    parser.add_argument("--json-out", help="write machine-readable attribution summary")
    parser.add_argument("--min-samples", type=int, default=0)
    parser.add_argument(
        "--min-attribution-pct",
        type=float,
        default=0.0,
        help="fail if fewer than this percentage of self samples map to cwasm text",
    )
    parser.add_argument(
        "--authoritative",
        action="store_true",
        help="require automatic exact-size mmap selection; disallow --base",
    )
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
    if not 0.0 <= args.min_attribution_pct <= 100.0:
        parser.error("--min-attribution-pct must be between 0 and 100")
    if args.authoritative and args.base:
        parser.error("--base is a non-authoritative diagnostic override")

    info = parse_cwasm(args.cwasm)
    if info.architecture != args.arch:
        raise AttributionError(
            f"artifact target architecture {info.architecture} does not match "
            f"--arch {args.arch}"
        )
    counts, total = ({}, 0)
    base = None
    mapping = None
    if args.perf:
        counts, total = addr_counts(args.perf)
        if total == 0:
            raise AttributionError("no samples in perf data")
        if total < args.min_samples:
            raise AttributionError(
                f"perf data has only {total} self samples; "
                f"--min-samples requires at least {args.min_samples}"
            )
        mapping = select_text_mapping(args.perf, info.text_size, args.base)
        base = mapping.base

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
        require_attribution_coverage(
            in_core, total, args.min_attribution_pct
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
        "architecture": args.arch,
        "artifact_target": {
            "architecture": info.architecture,
            "abi": info.abi,
            "format": info.target_format,
            "verified": info.target_verified,
        },
        "authoritative": bool(
            args.authoritative and mapping is not None and mapping.authoritative
        ),
        "perf": str(Path(args.perf).resolve()) if args.perf else None,
        "cwasm": str(Path(args.cwasm).resolve()),
        "text_base": base,
        "text_size": info.text_size,
        "function_count": len(info.func_offsets),
        "total_samples": total,
        "attributed_samples": in_core,
        "attribution_coverage_pct": _percent(in_core, total),
        "minimum_attribution_coverage_pct": args.min_attribution_pct,
        "mapping": (
            {
                "base": mapping.base,
                "size": mapping.size,
                "text_size": mapping.text_size,
                "page_size": mapping.page_size,
                "expected_size": mapping.expected_size,
                "candidates": [
                    {"base": candidate, "size": size}
                    for candidate, size in mapping.candidates
                ],
                "authoritative": mapping.authoritative,
                "override": mapping.override,
            }
            if mapping is not None
            else None
        ),
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
            info,
        )
        instructions = disassemble_function(
            function_code,
            function_base,
            scratch_dir,
            f"func-{args.func}",
            metadata.inline_data_ranges,
            args.arch,
        )
        validate_metadata_disassembly(metadata, instructions)
        frame_summary = build_frame_summary(instructions, counts, metadata)
    else:
        instructions = disassemble_blob(
            function_code,
            function_base,
            scratch_dir,
            f"func-{args.func}",
            architecture=args.arch,
        )

    function_counts = {
        ip: count
        for ip, count in counts.items()
        if function_base <= ip < function_base + len(function_code)
    }
    function_samples = sum(function_counts.values())
    by_class = Counter()
    static_by_class = Counter()
    hot = []
    instruction_addresses = set()
    instruction_classes = classify_instruction_stream(
        instructions, metadata, args.arch
    )
    for instruction, class_name in zip(instructions, instruction_classes):
        instruction_addresses.add(instruction.address)
        samples = counts.get(instruction.address, 0)
        static_by_class[class_name] += 1
        by_class[class_name] += samples
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
            1
            for instruction in instructions
            if instruction_frame_operand(instruction.text, args.arch)
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
        "instruction_count": sum(static_by_class.values()),
        "classes": {
            class_name: {
                "samples": by_class.get(class_name, 0),
                "percent_of_run": _percent(
                    by_class.get(class_name, 0), total
                ),
                "static_instructions": static_by_class.get(class_name, 0),
            }
            for class_name in sorted(set(static_by_class) | set(by_class))
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
