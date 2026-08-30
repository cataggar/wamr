#!/usr/bin/env python3
"""Compare one runtime-hot wasm function in precompiled WAMR and Wasmtime.

The ``capture`` command verifies a shared raw core-module identity, maps WAMR's
local function index into the wasm function-index space, recompiles that exact
core with the Wasmtime options recorded by ``bench_keyvault.py``, and records
compact machine-readable static/dynamic metrics.

The ``report`` command consumes only that capture JSON.  It ranks deltas and
applies a deliberately conservative recommendation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = 1
CAPTURE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
AOT_MAGIC = 0x746F6100
AOT_VERSION = 8
AOT_TEXT_SECTION = 2
AOT_FUNCTION_SECTION = 3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

METRIC_ORDER = (
    "native_instructions",
    "code_size_bytes",
    "frame_size_bytes",
    "frame_loads",
    "frame_stores",
    "reg_reg_moves",
    "address_generation",
    "branches",
    "indirect_dispatch",
    "calls",
)
METRIC_LABELS = {
    "native_instructions": "Native instructions",
    "code_size_bytes": "Code size",
    "frame_size_bytes": "Fixed frame size",
    "frame_loads": "Frame-load instructions",
    "frame_stores": "Frame-store instructions",
    "reg_reg_moves": "Register-to-register moves",
    "address_generation": "Address-generation instructions",
    "branches": "Branch instructions",
    "indirect_dispatch": "Indirect dispatch instructions",
    "calls": "Call instructions",
}
METRIC_UNITS = {
    "native_instructions": "instructions",
    "code_size_bytes": "bytes",
    "frame_size_bytes": "bytes",
    "frame_loads": "instructions",
    "frame_stores": "instructions",
    "reg_reg_moves": "instructions",
    "address_generation": "instructions",
    "branches": "instructions",
    "indirect_dispatch": "instructions",
    "calls": "instructions",
}
GATE_METRICS = {
    "frame_loads",
    "frame_stores",
    "reg_reg_moves",
    "address_generation",
    "branches",
    "indirect_dispatch",
    "calls",
}
LEVER_FOR_METRIC = {
    "frame_loads": "reduce frame traffic",
    "frame_stores": "reduce frame traffic",
    "reg_reg_moves": "coalesce or eliminate register moves",
    "address_generation": "simplify address generation",
    "branches": "reduce branch overhead",
    "indirect_dispatch": "optimize indirect dispatch",
    "calls": "reduce call overhead",
}


class ComparisonError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="UTF-8"))
    except FileNotFoundError as exc:
        raise ComparisonError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must contain a JSON object")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must be a JSON object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ComparisonError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) < minimum
    ):
        raise ComparisonError(f"{label} must be a number >= {minimum:g}")
    return float(value)


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label).lower()
    if not SHA256_RE.fullmatch(text) or text == "0" * 64:
        raise ComparisonError(f"{label} must be a non-zero lowercase SHA-256")
    return text


def _resolve(base: Path, value: Any, label: str) -> Path:
    path = Path(_string(value, label)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


class ByteReader:
    def __init__(self, data: bytes, label: str):
        self.data = data
        self.label = label
        self.pos = 0

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def read_byte(self) -> int:
        if self.pos >= len(self.data):
            raise ComparisonError(f"{self.label}: unexpected end of input")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def read_bytes(self, size: int) -> bytes:
        if size < 0 or self.pos + size > len(self.data):
            raise ComparisonError(f"{self.label}: unexpected end of input")
        value = self.data[self.pos : self.pos + size]
        self.pos += size
        return value

    def read_u32(self) -> int:
        value = 0
        shift = 0
        for _ in range(5):
            byte = self.read_byte()
            value |= (byte & 0x7F) << shift
            if byte & 0x80 == 0:
                if value > 0xFFFFFFFF:
                    break
                return value
            shift += 7
        raise ComparisonError(f"{self.label}: invalid u32 LEB128")

    def read_u64(self) -> int:
        value = 0
        shift = 0
        for _ in range(10):
            byte = self.read_byte()
            value |= (byte & 0x7F) << shift
            if byte & 0x80 == 0:
                return value
            shift += 7
        raise ComparisonError(f"{self.label}: invalid u64 LEB128")

    def read_s33(self) -> int:
        value = 0
        shift = 0
        byte = 0
        for _ in range(5):
            byte = self.read_byte()
            value |= (byte & 0x7F) << shift
            shift += 7
            if byte & 0x80 == 0:
                if shift < 33 and byte & 0x40:
                    value |= -(1 << shift)
                return value
        raise ComparisonError(f"{self.label}: invalid s33 LEB128")

    def read_name(self) -> str:
        raw = self.read_bytes(self.read_u32())
        try:
            return raw.decode("UTF-8")
        except UnicodeDecodeError as exc:
            raise ComparisonError(f"{self.label}: invalid UTF-8 name") from exc


@dataclass(frozen=True)
class WasmModuleIdentity:
    sha256: str
    imported_function_count: int
    local_function_count: int
    function_names: dict[int, str]

    @property
    def function_count(self) -> int:
        return self.imported_function_count + self.local_function_count


def _read_value_type(reader: ByteReader) -> None:
    first = reader.read_byte()
    if first in (0x63, 0x64):
        reader.read_s33()


def _read_limits(reader: ByteReader) -> None:
    flags = reader.read_u32()
    read_bound = reader.read_u64 if flags & 0x4 else reader.read_u32
    read_bound()
    if flags & 0x1:
        read_bound()


def _parse_imports(payload: bytes, label: str) -> int:
    reader = ByteReader(payload, label)
    function_count = 0
    for _ in range(reader.read_u32()):
        reader.read_name()
        reader.read_name()
        kind = reader.read_byte()
        if kind == 0:
            reader.read_u32()
            function_count += 1
        elif kind == 1:
            _read_value_type(reader)
            _read_limits(reader)
        elif kind == 2:
            _read_limits(reader)
        elif kind == 3:
            _read_value_type(reader)
            reader.read_byte()
        elif kind == 4:
            reader.read_byte()
            reader.read_u32()
        else:
            raise ComparisonError(f"{label}: unsupported import kind {kind}")
    if reader.remaining() != 0:
        raise ComparisonError(f"{label}: trailing bytes in import section")
    return function_count


def _parse_function_names(payload: bytes, label: str) -> dict[int, str]:
    reader = ByteReader(payload, label)
    if reader.read_name() != "name":
        return {}
    names: dict[int, str] = {}
    while reader.remaining() > 0:
        subsection_id = reader.read_byte()
        subsection = reader.read_bytes(reader.read_u32())
        if subsection_id != 1:
            continue
        sub = ByteReader(subsection, f"{label} function names")
        for _ in range(sub.read_u32()):
            index = sub.read_u32()
            name = sub.read_name()
            if index in names:
                raise ComparisonError(
                    f"{label}: duplicate function-name entry for wasm index {index}"
                )
            names[index] = name
        if sub.remaining() != 0:
            raise ComparisonError(f"{label}: trailing bytes in function-name map")
    return names


def parse_core_wasm(path: Path) -> WasmModuleIdentity:
    data = path.read_bytes()
    if len(data) < 8 or data[:4] != b"\0asm" or data[4:8] != b"\x01\0\0\0":
        raise ComparisonError(f"{path}: expected a core wasm v1 module")
    reader = ByteReader(data[8:], str(path))
    imports = 0
    saw_import_section = False
    local_functions: int | None = None
    code_function_count: int | None = None
    names: dict[int, str] = {}
    while reader.remaining() > 0:
        section_id = reader.read_byte()
        payload = reader.read_bytes(reader.read_u32())
        if section_id == 0:
            parsed = _parse_function_names(payload, str(path))
            for index, name in parsed.items():
                if index in names:
                    raise ComparisonError(
                        f"{path}: duplicate function-name entry for wasm index {index}"
                    )
                names[index] = name
        elif section_id == 2:
            if saw_import_section:
                raise ComparisonError(f"{path}: duplicate import section")
            saw_import_section = True
            imports += _parse_imports(payload, f"{path} import section")
        elif section_id == 3:
            section = ByteReader(payload, f"{path} function section")
            count = section.read_u32()
            for _ in range(count):
                section.read_u32()
            if section.remaining() != 0:
                raise ComparisonError(f"{path}: trailing bytes in function section")
            if local_functions is not None:
                raise ComparisonError(f"{path}: duplicate function section")
            local_functions = count
        elif section_id == 10:
            section = ByteReader(payload, f"{path} code section")
            count = section.read_u32()
            for _ in range(count):
                section.read_bytes(section.read_u32())
            if section.remaining() != 0:
                raise ComparisonError(f"{path}: trailing bytes in code section")
            if code_function_count is not None:
                raise ComparisonError(f"{path}: duplicate code section")
            code_function_count = count
    if local_functions is None:
        local_functions = 0
    if code_function_count is None:
        code_function_count = 0
    if code_function_count != local_functions:
        raise ComparisonError(
            f"{path}: function/code count mismatch "
            f"({local_functions} != {code_function_count})"
        )
    total = imports + local_functions
    for index in names:
        if index >= total:
            raise ComparisonError(
                f"{path}: function-name index {index} is outside 0..{max(total - 1, 0)}"
            )
    return WasmModuleIdentity(
        sha256=hashlib.sha256(data).hexdigest(),
        imported_function_count=imports,
        local_function_count=local_functions,
        function_names=names,
    )


def resolve_wasm_function(
    identity: WasmModuleIdentity, wasm_index: int | None, name: str | None
) -> tuple[int, str | None]:
    if wasm_index is None:
        if name is None:
            raise ComparisonError("function must provide wasm_index, name, or both")
        matches = sorted(
            index
            for index, candidate in identity.function_names.items()
            if candidate == name
        )
        if not matches:
            raise ComparisonError(f"function name {name!r} is absent from the name section")
        if len(matches) != 1:
            raise ComparisonError(
                f"function name {name!r} is ambiguous at wasm indices {matches}; "
                "provide function.wasm_index"
            )
        wasm_index = matches[0]
    if wasm_index < identity.imported_function_count:
        raise ComparisonError(
            f"wasm function {wasm_index} is imported; a WAMR local_func must be defined"
        )
    if wasm_index >= identity.function_count:
        raise ComparisonError(
            f"wasm function {wasm_index} is outside 0..{identity.function_count - 1}"
        )
    actual_name = identity.function_names.get(wasm_index)
    if name is not None and actual_name != name:
        raise ComparisonError(
            f"wasm function {wasm_index} name mismatch: expected {name!r}, "
            f"found {actual_name!r}"
        )
    return wasm_index, actual_name


@dataclass(frozen=True)
class CwasmFunction:
    code: bytes
    code_size: int
    function_count: int


def read_wamr_function(path: Path, local_func: int) -> CwasmFunction:
    data = path.read_bytes()
    if len(data) < 8:
        raise ComparisonError(f"{path}: truncated WAMR cwasm")
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != AOT_MAGIC:
        raise ComparisonError(f"{path}: bad WAMR cwasm magic {magic:#x}")
    if version != AOT_VERSION:
        raise ComparisonError(
            f"{path}: WAMR cwasm version {version} is unsupported; expected {AOT_VERSION}"
        )
    pos = 8
    text: bytes | None = None
    offsets: list[int] | None = None
    while pos + 8 <= len(data):
        section_type, section_size = struct.unpack_from("<II", data, pos)
        pos += 8
        end = pos + section_size
        if end > len(data):
            raise ComparisonError(f"{path}: truncated cwasm section {section_type}")
        payload = data[pos:end]
        if section_type == AOT_TEXT_SECTION:
            text = payload
        elif section_type == AOT_FUNCTION_SECTION:
            if len(payload) < 4:
                raise ComparisonError(f"{path}: truncated function section")
            count = struct.unpack_from("<I", payload, 0)[0]
            expected = 4 + count * 8
            if len(payload) != expected:
                raise ComparisonError(
                    f"{path}: function section size {len(payload)} != {expected}"
                )
            values = struct.unpack_from(f"<{count * 2}I", payload, 4)
            offsets = list(values[0::2])
        pos = end
    if pos != len(data):
        raise ComparisonError(f"{path}: trailing truncated cwasm header")
    if text is None or offsets is None:
        raise ComparisonError(f"{path}: missing text or function section")
    if not offsets or offsets[0] != 0 or offsets != sorted(set(offsets)):
        raise ComparisonError(f"{path}: invalid function offsets")
    if offsets[-1] >= len(text):
        raise ComparisonError(f"{path}: final function offset is outside text")
    if local_func >= len(offsets):
        raise ComparisonError(
            f"WAMR local_func {local_func} is outside 0..{len(offsets) - 1}"
        )
    start = offsets[local_func]
    end = offsets[local_func + 1] if local_func + 1 < len(offsets) else len(text)
    return CwasmFunction(
        code=text[start:end], code_size=end - start, function_count=len(offsets)
    )


@dataclass
class Instruction:
    offset: int
    size: int
    mnemonic: str
    operands: str
    text: str
    raw_bytes: bytes


INSTRUCTION_RE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+((?:(?:[0-9a-fA-F]{2})\s+)+)(.*?)\s*$"
)
CONTINUATION_RE = re.compile(r"^\s*((?:(?:[0-9a-fA-F]{2})\s*)+)\s*$")
WASMTIME_HEADER_RE = re.compile(
    r"^\s*(?:[0-9a-fA-F]+\s+)?wasm\[(\d+)\]::function\[(\d+)\](?:::(.*))?:\s*$"
)


def _normalize_mnemonic(text: str) -> tuple[str, str]:
    parts = text.strip().split(None, 1)
    if not parts:
        return "", ""
    if parts[0].lower() in {"lock", "rep", "repe", "repz", "repne", "repnz"}:
        parts = parts[1].split(None, 1) if len(parts) == 2 else [""]
    mnemonic = parts[0].lower()
    aliases = {
        "callq": "call",
        "jmpq": "jmp",
        "pushq": "push",
        "popq": "pop",
        "retq": "ret",
        "leaveq": "leave",
        "leaq": "lea",
        "movq": "mov",
        "movl": "mov",
        "movw": "mov",
        "movb": "mov",
    }
    mnemonic = aliases.get(mnemonic, mnemonic)
    if len(mnemonic) > 2 and mnemonic[-1] in "bwlq" and mnemonic[:-1] in {
        "add",
        "sub",
        "and",
        "or",
        "xor",
        "cmp",
        "test",
        "inc",
        "dec",
        "neg",
        "not",
        "shl",
        "shr",
        "sar",
        "imul",
    }:
        mnemonic = mnemonic[:-1]
    operands = parts[1].strip() if len(parts) == 2 else ""
    return mnemonic, operands


def parse_disassembly(text: str, *, wasmtime_wasm_index: int | None = None) -> list[Instruction]:
    instructions: list[Instruction] = []
    selected = wasmtime_wasm_index is None
    saw_header = False
    saw_selected_header = False
    base_offset: int | None = None
    for line in text.splitlines():
        header = WASMTIME_HEADER_RE.match(line)
        if header:
            saw_header = True
            module_index = int(header.group(1))
            function_index = int(header.group(2))
            selected = (
                wasmtime_wasm_index is not None
                and module_index == 0
                and function_index == wasmtime_wasm_index
            )
            if selected:
                if saw_selected_header:
                    raise ComparisonError(
                        "Wasmtime objdump contained duplicate matching function headers"
                    )
                saw_selected_header = True
            continue
        if saw_header and not selected:
            continue
        match = INSTRUCTION_RE.match(line)
        if match:
            raw_offset = int(match.group(1), 16)
            byte_count = len(match.group(2).split())
            assembly = match.group(3).strip()
            if not assembly:
                if instructions:
                    instructions[-1].size += byte_count
                    instructions[-1].raw_bytes += bytes.fromhex(match.group(2))
                continue
            mnemonic, operands = _normalize_mnemonic(assembly)
            if not mnemonic:
                continue
            if base_offset is None:
                base_offset = raw_offset
            instructions.append(
                Instruction(
                    offset=raw_offset - base_offset,
                    size=byte_count,
                    mnemonic=mnemonic,
                    operands=operands,
                    text=assembly,
                    raw_bytes=bytes.fromhex(match.group(2)),
                )
            )
            continue
        continuation = CONTINUATION_RE.match(line)
        if continuation and instructions:
            extra = bytes.fromhex(continuation.group(1))
            instructions[-1].size += len(extra)
            instructions[-1].raw_bytes += extra
    if wasmtime_wasm_index is not None and not saw_selected_header:
        raise ComparisonError(
            "Wasmtime objdump did not contain exactly "
            f"wasm[0]::function[{wasmtime_wasm_index}]"
        )
    if not instructions:
        raise ComparisonError("disassembly contained no instructions")
    previous = -1
    for instruction in instructions:
        if instruction.offset <= previous:
            raise ComparisonError("disassembly instruction offsets are not increasing")
        previous = instruction.offset
    return instructions


def parse_wasmtime_explore(
    path: Path, wasm_index: int
) -> tuple[list[Instruction], list[Instruction]]:
    text = path.read_text(encoding="UTF-8")
    marker = "window.ASM ="
    start = text.find(marker)
    if start < 0:
        raise ComparisonError("Wasmtime explore output has no window.ASM payload")
    start += len(marker)
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:].lstrip())
    except json.JSONDecodeError as exc:
        raise ComparisonError(
            f"Wasmtime explore window.ASM payload is malformed: {exc}"
        ) from exc
    root = _object(payload, "Wasmtime explore ASM")
    functions = root.get("functions")
    if not isinstance(functions, list):
        raise ComparisonError("Wasmtime explore ASM functions must be an array")
    matches = [
        item
        for item in functions
        if isinstance(item, dict) and item.get("func_index") == wasm_index
    ]
    if len(matches) != 1:
        raise ComparisonError(
            "Wasmtime explore must contain exactly one function with wasm index "
            f"{wasm_index}, found {len(matches)}"
        )
    raw_instructions = matches[0].get("instructions")
    if not isinstance(raw_instructions, list) or not raw_instructions:
        raise ComparisonError("Wasmtime explore function has no instructions")
    first_address: int | None = None
    all_instructions: list[Instruction] = []
    mapped_indices: list[int] = []
    for index, raw in enumerate(raw_instructions):
        item = _object(raw, f"Wasmtime explore instruction {index}")
        address = _integer(item.get("address"), f"explore instruction {index}.address")
        raw_bytes = item.get("bytes")
        if (
            not isinstance(raw_bytes, list)
            or not raw_bytes
            or any(
                not isinstance(byte, int)
                or isinstance(byte, bool)
                or not 0 <= byte <= 255
                for byte in raw_bytes
            )
        ):
            raise ComparisonError(
                f"explore instruction {index}.bytes must be a non-empty byte array"
            )
        mnemonic = _string(
            item.get("mnemonic"), f"explore instruction {index}.mnemonic"
        ).lower()
        operands = item.get("operands")
        if not isinstance(operands, str):
            raise ComparisonError(
                f"explore instruction {index}.operands must be a string"
            )
        normalized, _ = _normalize_mnemonic(
            f"{mnemonic} {operands}".rstrip()
        )
        if first_address is None:
            first_address = address
        instruction = Instruction(
            offset=address - first_address,
            size=len(raw_bytes),
            mnemonic=normalized,
            operands=operands,
            text=f"{mnemonic} {operands}".rstrip(),
            raw_bytes=bytes(raw_bytes),
        )
        if all_instructions:
            previous = all_instructions[-1]
            expected = previous.offset + previous.size
            if instruction.offset != expected:
                raise ComparisonError(
                    "Wasmtime explore instruction bytes are not contiguous"
                )
        all_instructions.append(instruction)
        wasm_offset = item.get("wasm_offset")
        if wasm_offset is not None:
            _integer(wasm_offset, f"explore instruction {index}.wasm_offset")
            mapped_indices.append(index)
    if not mapped_indices:
        raise ComparisonError(
            "Wasmtime explore emitted no wasm address mappings for the function"
        )
    final = mapped_indices[-1]
    terminators = {"ret", "ud2", "jmp", "hlt", "int3"}
    while (
        final + 1 < len(all_instructions)
        and all_instructions[final].mnemonic not in terminators
    ):
        final += 1
    if all_instructions[final].mnemonic not in terminators:
        raise ComparisonError(
            "cannot separate Wasmtime trailing data: no final mapped terminator"
        )
    return all_instructions, all_instructions[: final + 1]


def _split_operands(operands: str) -> list[str]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(operands):
        if char in "[(":
            depth += 1
        elif char in "])":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            result.append(operands[start:index].strip())
            start = index + 1
    if operands[start:].strip():
        result.append(operands[start:].strip())
    return result


def _is_frame_memory(operand: str) -> bool:
    compact = operand.lower().replace(" ", "")
    return bool(
        re.search(
            r"\[(?:[^\]]*[+\-*])?(?:rbp|rsp)(?:[+\-*][^\]]*)?\]",
            compact,
        )
        or re.search(r"\([^)]*%(?:rbp|rsp)(?:,[^)]*)?\)", compact)
    )


def _is_register(operand: str) -> bool:
    value = re.sub(
        r"^(?:byte|word|dword|qword|xmmword|ymmword|zmmword)\s+ptr\s+",
        "",
        operand.strip().lower(),
    )
    value = value.lstrip("*%")
    return bool(
        re.fullmatch(
            r"(?:"
            r"r(?:[0-9]|1[0-5])(?:b|w|d)?|"
            r"r(?:ax|bx|cx|dx|si|di|bp|sp)|"
            r"e(?:ax|bx|cx|dx|si|di|bp|sp)|"
            r"(?:ax|bx|cx|dx|si|di|bp|sp)|"
            r"(?:si|di|bp|sp)l|"
            r"[abcd][lh]|"
            r"(?:xmm|ymm|zmm|mm|k)\d+"
            r")",
            value,
        )
    )


def _is_move(mnemonic: str) -> bool:
    return mnemonic.startswith(("mov", "vmov", "kmov"))


def _frame_accesses(instruction: Instruction) -> tuple[bool, bool]:
    if not _is_move(instruction.mnemonic):
        return False, False
    operands = _split_operands(instruction.operands)
    frame_positions = [
        index for index, operand in enumerate(operands) if _is_frame_memory(operand)
    ]
    if not frame_positions:
        return False, False
    destination = len(operands) - 1 if _is_att_syntax(instruction) else 0
    store = destination in frame_positions
    load = any(position != destination for position in frame_positions)
    return load, store


def _is_att_syntax(instruction: Instruction) -> bool:
    return "%" in instruction.operands or "$" in instruction.operands


def _parse_immediate(value: str) -> int | None:
    text = value.strip().lower().replace("$", "")
    try:
        return int(text, 0)
    except ValueError:
        return None


def infer_frame_size(instructions: list[Instruction]) -> dict[str, Any]:
    fixed = 0
    saw_adjustment = False
    saw_sub_adjustment = False
    prologue_open = True
    for instruction in instructions:
        operands = _split_operands(instruction.operands)
        att = _is_att_syntax(instruction)
        destination_operand = (
            operands[-1] if att else operands[0] if operands else ""
        )
        destination = destination_operand.lstrip("%").lower()
        if prologue_open:
            if instruction.mnemonic in {"endbr64", "nop"}:
                continue
            if instruction.mnemonic == "push":
                fixed += 8
                saw_adjustment = True
                continue
            if (
                instruction.mnemonic == "sub"
                and len(operands) == 2
                and destination == "rsp"
            ):
                if saw_sub_adjustment:
                    return {
                        "status": "unavailable",
                        "reason": "multiple prologue stack allocations are ambiguous",
                    }
                amount = _parse_immediate(operands[0] if att else operands[1])
                if amount is None or amount < 0:
                    return {
                        "status": "unavailable",
                        "reason": "non-constant stack adjustment in function prologue",
                    }
                fixed += amount
                saw_adjustment = True
                saw_sub_adjustment = True
                continue
            if (
                instruction.mnemonic == "and"
                and operands
                and destination == "rsp"
            ):
                return {
                    "status": "unavailable",
                    "reason": "dynamic stack alignment prevents fixed-frame recovery",
                }
            if instruction.mnemonic in {"mov", "lea"}:
                continue
            prologue_open = False
    return {
        "status": "measured",
        "value": fixed if saw_adjustment else 0,
        "unit": "bytes",
        "method": "fixed x86_64 prologue stack adjustment",
    }


def aggregate_static_metrics(
    instructions: list[Instruction], code_size: int
) -> dict[str, dict[str, Any]]:
    counts = {
        "frame_loads": 0,
        "frame_stores": 0,
        "reg_reg_moves": 0,
        "address_generation": 0,
        "branches": 0,
        "indirect_dispatch": 0,
        "calls": 0,
    }
    for instruction in instructions:
        operands = _split_operands(instruction.operands)
        load, store = _frame_accesses(instruction)
        counts["frame_loads"] += int(load)
        counts["frame_stores"] += int(store)
        if (
            _is_move(instruction.mnemonic)
            and len(operands) >= 2
            and _is_register(operands[0])
            and _is_register(operands[1])
        ):
            counts["reg_reg_moves"] += 1
        if instruction.mnemonic == "lea":
            counts["address_generation"] += 1
        is_branch = instruction.mnemonic == "jmp" or (
            instruction.mnemonic.startswith("j") and len(instruction.mnemonic) > 1
        )
        counts["branches"] += int(is_branch)
        if instruction.mnemonic == "call":
            counts["calls"] += 1
        if instruction.mnemonic in {"call", "jmp"} and operands:
            target = operands[0].lstrip("*")
            if _is_register(target) or "[" in target or "(" in target:
                counts["indirect_dispatch"] += 1
    metrics: dict[str, dict[str, Any]] = {
        "native_instructions": {
            "status": "measured",
            "value": len(instructions),
            "unit": "instructions",
            "method": "parsed native disassembly",
        },
        "code_size_bytes": {
            "status": "measured",
            "value": code_size,
            "unit": "bytes",
            "method": "function byte extent",
        },
        "frame_size_bytes": infer_frame_size(instructions),
    }
    for name, value in counts.items():
        metrics[name] = {
            "status": "measured",
            "value": value,
            "unit": "instructions",
            "method": "classified native disassembly",
        }
    return metrics


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _sample_metric(samples: int, function_samples: int, total_samples: int) -> dict[str, Any]:
    return {
        "status": "measured",
        "samples": samples,
        "percent_of_function_samples": (
            100.0 * samples / function_samples if function_samples else 0.0
        ),
        "percent_of_run_samples": 100.0 * samples / total_samples if total_samples else 0.0,
        "method": "perf self samples classified by sampled native instruction",
    }


def wamr_dynamic_metrics(attribution: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    total = _integer(attribution.get("total_samples"), "attribution.total_samples", 1)
    classified = _object(
        attribution.get("classified_function"), "attribution.classified_function"
    )
    function_samples = _integer(
        classified.get("samples"), "attribution.classified_function.samples", 1
    )
    share = _number(
        classified.get("percent_of_run"),
        "attribution.classified_function.percent_of_run",
    )
    classes = _object(
        classified.get("classes"), "attribution.classified_function.classes"
    )
    if function_samples > total:
        raise ComparisonError(
            "attribution classified function samples exceed total samples"
        )
    expected_share = 100.0 * function_samples / total
    if abs(share - expected_share) > 0.05:
        raise ComparisonError(
            "attribution classified function percent_of_run is inconsistent "
            "with its sample counts"
        )
    class_total = 0
    for class_name, raw in classes.items():
        value = _object(raw, f"attribution class {class_name!r}")
        class_total += _integer(
            value.get("samples"), f"attribution class {class_name!r}.samples"
        )
    if class_total != function_samples:
        raise ComparisonError(
            "attribution instruction-class samples do not sum to function samples"
        )

    def samples_for(name: str) -> int:
        item = _object(classes.get(name), f"attribution class {name!r}")
        return _integer(item.get("samples"), f"attribution class {name!r}.samples")

    dynamic: dict[str, Any] = {
        name: _unavailable("this metric is static metadata, not an instruction class")
        for name in ("code_size_bytes", "frame_size_bytes")
    }
    dynamic["native_instructions"] = _sample_metric(
        function_samples, function_samples, total
    )
    class_map = {
        "frame_loads": "spill_load (reloads)",
        "frame_stores": "frame stores",
        "reg_reg_moves": "reg-reg mov",
        "indirect_dispatch": "dispatch (computed-goto)",
        "calls": "call",
    }
    for metric, class_name in class_map.items():
        dynamic[metric] = _sample_metric(
            samples_for(class_name), function_samples, total
        )
    dynamic["address_generation"] = _unavailable(
        "aot_jit_attr.py groups LEA with ALU, so address-generation samples "
        "cannot be separated without changing the concurrently-owned attribution tool"
    )
    dynamic["branches"] = _unavailable(
        "aot_jit_attr.py combines bounds compares with bounds branches, so exact "
        "branch-only samples are unavailable"
    )
    return dynamic, {
        "status": "measured",
        "source": "WAMR perf self samples",
        "total_samples": total,
        "function_samples": function_samples,
        "percent_of_run": share,
    }


def load_explicit_profile(
    path: Path,
    expected_sha256: str,
    engine: str,
    module_sha256: str,
    wasm_index: int,
    function_name: str | None,
) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ComparisonError(
            f"{engine} profile SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    profile = _load_json(path, f"{engine} profile")
    if profile.get("schema_version") != 1 or profile.get("kind") != "wasm-hot-function-profile":
        raise ComparisonError(f"{engine} profile has an unsupported schema or kind")
    if profile.get("engine") != engine:
        raise ComparisonError(
            f"profile engine mismatch: expected {engine!r}, got {profile.get('engine')!r}"
        )
    if profile.get("mapping_verified") is not True:
        raise ComparisonError(f"{engine} profile must set mapping_verified=true")
    _string(profile.get("mapping_evidence"), f"{engine} profile.mapping_evidence")
    if _sha256(profile.get("module_sha256"), f"{engine} profile.module_sha256") != module_sha256:
        raise ComparisonError(f"{engine} profile module identity does not match the core wasm")
    if _integer(profile.get("wasm_function_index"), "profile.wasm_function_index") != wasm_index:
        raise ComparisonError(f"{engine} profile function index does not match")
    if profile.get("function_name") != function_name:
        raise ComparisonError(f"{engine} profile function name does not match")
    total = _integer(profile.get("total_samples"), "profile.total_samples", 1)
    function_samples = _integer(
        profile.get("function_samples"), "profile.function_samples", 1
    )
    if function_samples > total:
        raise ComparisonError("profile.function_samples cannot exceed total_samples")
    metrics = _object(profile.get("metrics"), "profile.metrics")
    result: dict[str, Any] = {
        "native_instructions": _sample_metric(
            function_samples, function_samples, total
        ),
        "code_size_bytes": _unavailable(
            "this metric is static metadata, not an instruction class"
        ),
        "frame_size_bytes": _unavailable(
            "this metric is static metadata, not an instruction class"
        ),
    }
    for name in GATE_METRICS:
        item = metrics.get(name)
        if item is None:
            result[name] = _unavailable("metric omitted by explicit profile")
            continue
        value = _object(item, f"profile.metrics.{name}")
        status = value.get("status")
        if status == "measured":
            result[name] = _sample_metric(
                _integer(value.get("samples"), f"profile.metrics.{name}.samples"),
                function_samples,
                total,
            )
        elif status in {"unavailable", "partial"}:
            result[name] = {
                "status": status,
                "reason": _string(
                    value.get("reason"), f"profile.metrics.{name}.reason"
                ),
            }
        else:
            raise ComparisonError(
                f"profile.metrics.{name}.status must be measured, partial, or unavailable"
            )
    return result


def _run_checked(command: list[str], label: str, timeout: float = 300.0) -> str:
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env={
                key: value
                for key, value in os.environ.items()
                if key in {"PATH", "HOME"}
            }
            | {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
        )
    except FileNotFoundError as exc:
        raise ComparisonError(f"{label} tool not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ComparisonError(f"{label} exceeded {timeout:g}s") from exc
    if proc.returncode != 0:
        raise ComparisonError(
            f"{label} failed with exit {proc.returncode}\n"
            f"command: {' '.join(command)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def find_wamr_jump_tables(code: bytes) -> list[dict[str, int]]:
    dispatch_tail = bytes.fromhex("4f 63 1c 9a 4d 01 da 41 ff e2")
    ranges: list[dict[str, int]] = []
    search_from = 0
    while True:
        tail = code.find(dispatch_tail, search_from)
        if tail < 0:
            break
        lea = tail - 7
        search_from = tail + 1
        if any(table["start"] <= tail < table["end"] for table in ranges):
            continue
        if lea < 6 or code[lea : lea + 3] != bytes.fromhex("4c 8d 15"):
            raise ComparisonError(
                "WAMR br_table dispatch tail lacks its expected RIP-relative LEA"
            )
        jae = lea - 6
        if code[jae : jae + 2] != bytes.fromhex("0f 83"):
            raise ComparisonError(
                "WAMR br_table dispatch lacks its expected unsigned bounds branch"
            )
        count: int | None = None
        if jae >= 4 and code[jae - 4 : jae - 1] == bytes.fromhex("49 83 fb"):
            count = code[jae - 1]
        elif jae >= 7 and code[jae - 7 : jae - 4] == bytes.fromhex("49 81 fb"):
            count = struct.unpack_from("<I", code, jae - 4)[0]
        if count is None:
            raise ComparisonError(
                "WAMR br_table dispatch target count could not be recovered"
            )
        displacement = struct.unpack_from("<i", code, lea + 3)[0]
        table_start = lea + 7 + displacement
        dispatch_end = tail + len(dispatch_tail)
        if table_start != dispatch_end:
            raise ComparisonError(
                "WAMR br_table LEA does not point immediately after the dispatch"
            )
        table_end = table_start + count * 4
        if count == 0:
            continue
        if table_end > len(code):
            raise ComparisonError("WAMR br_table extends beyond the function")
        valid = True
        for offset in range(table_start, table_end, 4):
            relative = struct.unpack_from("<i", code, offset)[0]
            target = table_start + relative
            if not 0 <= target < len(code):
                valid = False
                break
        if valid:
            ranges.append(
                {"start": table_start, "end": table_end, "entries": count}
            )
        else:
            raise ComparisonError(
                "WAMR br_table contains an out-of-function target"
            )
    ranges.sort(key=lambda item: item["start"])
    for previous, current in zip(ranges, ranges[1:]):
        if previous["end"] > current["start"]:
            raise ComparisonError("overlapping WAMR inline jump tables")
    return ranges


def disassemble_wamr(
    function: CwasmFunction, work_dir: Path
) -> tuple[list[Instruction], list[str], list[dict[str, int]]]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ComparisonError(
            "native hot-function capture currently requires Linux x86_64"
        )
    objdump = "objdump"
    scratch = work_dir / f".wamr-hot-function-{os.getpid()}.bin"
    jump_tables = find_wamr_jump_tables(function.code)
    disassembly_bytes = bytearray(function.code)
    for table in jump_tables:
        disassembly_bytes[table["start"] : table["end"]] = b"\x90" * (
            table["end"] - table["start"]
        )
    scratch.write_bytes(disassembly_bytes)
    command = [
        objdump,
        "-D",
        "-b",
        "binary",
        "-m",
        "i386:x86-64",
        "-M",
        "intel",
        "--adjust-vma=0",
        str(scratch),
    ]
    try:
        output = _run_checked(command, "WAMR function objdump")
    finally:
        scratch.unlink(missing_ok=True)
    instructions = parse_disassembly(output)
    covered = max(i.offset + i.size for i in instructions)
    if covered != function.code_size:
        raise ComparisonError(
            f"WAMR disassembly covered {covered} of {function.code_size} function bytes"
        )
    instructions = [
        instruction
        for instruction in instructions
        if not any(
            table["start"] <= instruction.offset < table["end"]
            for table in jump_tables
        )
    ]
    return instructions, command, jump_tables


def _artifact_hash(report: dict[str, Any], path: Path) -> str:
    precompile = _object(report.get("precompile"), "benchmark.precompile")
    artifacts = precompile.get("artifacts")
    if not isinstance(artifacts, list):
        raise ComparisonError("benchmark.precompile.artifacts must be an array")
    matches = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and Path(str(item.get("path", ""))).resolve() == path.resolve()
    ]
    if len(matches) != 1:
        raise ComparisonError(
            f"benchmark report must contain exactly one artifact entry for {path}"
        )
    return _sha256(matches[0].get("sha256"), f"artifact {path}.sha256")


def _derive_wasmtime_compile_command(
    report: dict[str, Any], core_wasm: Path, output: Path
) -> tuple[list[str], dict[str, Any]]:
    validation = _object(report.get("validation"), "benchmark.validation")
    tools = _object(validation.get("tools"), "benchmark.validation.tools")
    tool = _object(tools.get("wasmtime"), "benchmark.validation.tools.wasmtime")
    tool_path = Path(_string(tool.get("path"), "wasmtime tool path")).resolve()
    expected_tool_sha = _sha256(tool.get("sha256"), "wasmtime tool sha256")
    if not tool_path.is_file() or not os.access(tool_path, os.X_OK):
        raise ComparisonError(f"recorded Wasmtime binary is absent or not executable: {tool_path}")
    actual_tool_sha = sha256_file(tool_path)
    if actual_tool_sha != expected_tool_sha:
        raise ComparisonError(
            f"Wasmtime binary SHA-256 mismatch: expected {expected_tool_sha}, "
            f"got {actual_tool_sha}"
        )
    expected_version = _string(tool.get("version"), "wasmtime tool version")
    actual_version = _run_checked([str(tool_path), "--version"], "Wasmtime version").strip()
    if actual_version != expected_version:
        raise ComparisonError(
            f"Wasmtime version mismatch: expected {expected_version!r}, "
            f"got {actual_version!r}"
        )

    precompile = _object(report.get("precompile"), "benchmark.precompile")
    commands = _object(precompile.get("commands"), "benchmark.precompile.commands")
    original = commands.get("wasmtime")
    if not isinstance(original, list) or not all(isinstance(arg, str) for arg in original):
        raise ComparisonError("benchmark Wasmtime precompile command must be a string array")
    if len(original) < 4 or Path(original[0]).resolve() != tool_path or original[1] != "compile":
        raise ComparisonError("benchmark Wasmtime command is not the recorded tool plus 'compile'")
    component = _object(
        _object(validation.get("inputs"), "benchmark.validation.inputs").get("component"),
        "benchmark component",
    )
    component_path = Path(_string(component.get("path"), "benchmark component path")).resolve()
    rebuilt: list[str] = []
    input_replacements = 0
    output_replacements = 0
    index = 0
    while index < len(original):
        arg = original[index]
        if index == 0:
            rebuilt.append(str(tool_path))
        elif arg in {"-o", "--output"}:
            if index + 1 >= len(original):
                raise ComparisonError("benchmark Wasmtime command has a dangling output option")
            rebuilt.extend([arg, str(output)])
            output_replacements += 1
            index += 1
        elif arg.startswith("--output="):
            rebuilt.append(f"--output={output}")
            output_replacements += 1
        elif Path(arg).is_absolute() and Path(arg).resolve() == component_path:
            rebuilt.append(str(core_wasm))
            input_replacements += 1
        else:
            rebuilt.append(arg)
        index += 1
    if input_replacements != 1 or output_replacements != 1:
        raise ComparisonError(
            "could not replace exactly one component input and output in the "
            "recorded Wasmtime compile command"
        )
    return rebuilt, {
        "path": str(tool_path),
        "sha256": actual_tool_sha,
        "version": actual_version,
        "benchmark_compile_command": original,
    }


def capture_wasmtime(
    report: dict[str, Any],
    core_wasm: Path,
    wasm_index: int,
    work_dir: Path,
) -> tuple[list[Instruction], int, dict[str, Any]]:
    cwasm = work_dir / "wasmtime-hot-core.cwasm"
    cwasm.unlink(missing_ok=True)
    compile_command, tool = _derive_wasmtime_compile_command(
        report, core_wasm, cwasm
    )
    _run_checked(compile_command, "Wasmtime exact-core precompile")
    if not cwasm.is_file():
        raise ComparisonError("Wasmtime compile succeeded without producing its output")
    objdump_command = [
        tool["path"],
        "objdump",
        str(cwasm),
        "--addresses",
        "--bytes",
        "--color",
        "never",
        "--addrmap=false",
        "--traps=false",
        "--stack-maps=false",
        "--exception-tables=false",
        "--frame-tables=false",
        "--funcs",
        "wasm",
        "--filter",
        f"wasm[0]::function[{wasm_index}]",
    ]
    output = _run_checked(objdump_command, "Wasmtime exact-core objdump")
    objdump_instructions = parse_disassembly(
        output, wasmtime_wasm_index=wasm_index
    )

    explore_html = work_dir / "wasmtime-hot-core.explore.html"
    explore_html.unlink(missing_ok=True)
    explore_command = list(compile_command)
    explore_command[1] = "explore"
    for index, arg in enumerate(explore_command):
        if arg in {"-o", "--output"} and index + 1 < len(explore_command):
            explore_command[index + 1] = str(explore_html)
        elif arg.startswith("--output="):
            explore_command[index] = f"--output={explore_html}"
    _run_checked(explore_command, "Wasmtime exact-core explore")
    if not explore_html.is_file():
        raise ComparisonError("Wasmtime explore succeeded without producing its output")
    explore_all, instructions = parse_wasmtime_explore(
        explore_html, wasm_index
    )
    objdump_bytes = b"".join(i.raw_bytes for i in objdump_instructions)
    explore_bytes = b"".join(i.raw_bytes for i in explore_all)
    if not objdump_bytes or not explore_bytes.startswith(objdump_bytes):
        raise ComparisonError(
            "Wasmtime precompiled objdump bytes are not a prefix of the "
            "explore disassembly bytes"
        )
    code_size = max(
        instruction.offset + instruction.size for instruction in explore_all
    )
    trailing_bytes = code_size - max(
        instruction.offset + instruction.size for instruction in instructions
    )
    return instructions, code_size, {
        "tool": tool,
        "compile_command": compile_command,
        "objdump_command": objdump_command,
        "explore_command": explore_command,
        "cwasm": {
            "path": str(cwasm),
            "sha256": sha256_file(cwasm),
        },
        "explore": {
            "path": str(explore_html),
            "sha256": sha256_file(explore_html),
            "precompiled_objdump_prefix_bytes_verified": len(objdump_bytes),
            "trailing_unmapped_bytes_excluded_from_instruction_counts": trailing_bytes,
        },
        "scope": (
            "standalone precompile of the exact raw core module with the "
            "benchmark's Wasmtime compile options"
        ),
    }


@dataclass(frozen=True)
class CaptureConfig:
    path: Path
    benchmark_report: Path
    benchmark_report_sha256: str
    core_wasm: Path
    core_wasm_sha256: str
    function_wasm_index: int | None
    function_name: str | None
    wamr_manifest: Path
    wamr_cwasm: Path
    wamr_local_func: int
    wasmtime_profile: Path | None
    wasmtime_profile_sha256: str | None


def load_capture_config(path: Path) -> CaptureConfig:
    root = _load_json(path, "comparison config")
    if root.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ComparisonError(
            f"comparison config schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    base = path.resolve().parent
    benchmark = _object(root.get("benchmark_report"), "benchmark_report")
    core = _object(root.get("core_wasm"), "core_wasm")
    function = _object(root.get("function"), "function")
    wamr = _object(root.get("wamr"), "wamr")
    wasmtime = _object(root.get("wasmtime"), "wasmtime")
    wasm_index_raw = function.get("wasm_index")
    wasm_index = (
        None
        if wasm_index_raw is None
        else _integer(wasm_index_raw, "function.wasm_index")
    )
    name_raw = function.get("name")
    if name_raw is not None and (not isinstance(name_raw, str) or not name_raw):
        raise ComparisonError("function.name must be null or a non-empty string")
    profile_raw = wasmtime.get("profile")
    profile_path: Path | None = None
    profile_sha: str | None = None
    if profile_raw is not None:
        profile = _object(profile_raw, "wasmtime.profile")
        profile_path = _resolve(base, profile.get("path"), "wasmtime.profile.path")
        profile_sha = _sha256(
            profile.get("sha256"), "wasmtime.profile.sha256"
        )
    return CaptureConfig(
        path=path.resolve(),
        benchmark_report=_resolve(
            base, benchmark.get("path"), "benchmark_report.path"
        ),
        benchmark_report_sha256=_sha256(
            benchmark.get("sha256"), "benchmark_report.sha256"
        ),
        core_wasm=_resolve(base, core.get("path"), "core_wasm.path"),
        core_wasm_sha256=_sha256(core.get("sha256"), "core_wasm.sha256"),
        function_wasm_index=wasm_index,
        function_name=name_raw,
        wamr_manifest=_resolve(base, wamr.get("manifest"), "wamr.manifest"),
        wamr_cwasm=_resolve(base, wamr.get("cwasm"), "wamr.cwasm"),
        wamr_local_func=_integer(wamr.get("local_func"), "wamr.local_func"),
        wasmtime_profile=profile_path,
        wasmtime_profile_sha256=profile_sha,
    )


def _validate_benchmark(
    config: CaptureConfig,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    actual_report_sha = sha256_file(config.benchmark_report)
    if actual_report_sha != config.benchmark_report_sha256:
        raise ComparisonError(
            "benchmark report SHA-256 mismatch: "
            f"expected {config.benchmark_report_sha256}, got {actual_report_sha}"
        )
    report = _load_json(config.benchmark_report, "benchmark report")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "keyvault-tcgc-aot-benchmark"
    ):
        raise ComparisonError("benchmark report is not a supported keyvault AOT report")
    validation = _object(report.get("validation"), "benchmark.validation")
    component = _object(
        _object(validation.get("inputs"), "benchmark.validation.inputs").get(
            "component"
        ),
        "benchmark component",
    )
    component_sha = _sha256(component.get("sha256"), "benchmark component sha256")
    perf = _object(report.get("perf"), "benchmark.perf")
    if perf.get("selected") is not True:
        raise ComparisonError(
            "benchmark report has no WAMR perf attribution; rerun bench_keyvault.py "
            "with --profile"
        )
    attribution = _object(perf.get("attribution"), "benchmark.perf.attribution")
    return report, attribution, component_sha


def _validate_wamr_identity(
    config: CaptureConfig,
    report: dict[str, Any],
    attribution: dict[str, Any],
    component_sha: str,
    core_identity: WasmModuleIdentity,
    wasm_index: int,
) -> tuple[dict[str, Any], CwasmFunction]:
    manifest = _load_json(config.wamr_manifest, "WAMR component manifest")
    if manifest.get("version") != 2:
        raise ComparisonError("WAMR component manifest must use identity-safe version 2")
    if _sha256(manifest.get("component_sha256"), "WAMR manifest component_sha256") != component_sha:
        raise ComparisonError("WAMR manifest and benchmark component identities differ")
    modules = manifest.get("modules")
    if not isinstance(modules, list):
        raise ComparisonError("WAMR manifest.modules must be an array")
    matches: list[dict[str, Any]] = []
    for raw in modules:
        if not isinstance(raw, dict):
            raise ComparisonError("WAMR manifest module entries must be objects")
        candidate = (config.wamr_manifest.parent / str(raw.get("path", ""))).resolve()
        if candidate == config.wamr_cwasm:
            matches.append(raw)
    if len(matches) != 1:
        raise ComparisonError(
            "WAMR manifest must contain exactly one entry for the configured cwasm"
        )
    entry = matches[0]
    entry_core_sha = _sha256(entry.get("core_sha256"), "WAMR manifest core_sha256")
    if entry_core_sha != core_identity.sha256:
        raise ComparisonError(
            "raw core module SHA-256 does not match the selected WAMR manifest entry"
        )
    actual_cwasm_sha = sha256_file(config.wamr_cwasm)
    if _sha256(entry.get("sha256"), "WAMR manifest cwasm sha256") != actual_cwasm_sha:
        raise ComparisonError("selected WAMR cwasm does not match its manifest SHA-256")
    if _artifact_hash(report, config.wamr_cwasm) != actual_cwasm_sha:
        raise ComparisonError("selected WAMR cwasm does not match the benchmark artifact")
    attribution_cwasm = Path(
        _string(attribution.get("cwasm"), "attribution.cwasm")
    ).resolve()
    if attribution_cwasm != config.wamr_cwasm:
        raise ComparisonError(
            "WAMR attribution was captured from a different cwasm artifact"
        )
    classified = _object(
        attribution.get("classified_function"), "attribution.classified_function"
    )
    attributed_local = _integer(
        classified.get("local_func"), "attribution.classified_function.local_func"
    )
    if attributed_local != config.wamr_local_func:
        raise ComparisonError(
            "WAMR attribution function and configured local_func differ"
        )
    mapped_wasm_index = (
        core_identity.imported_function_count + config.wamr_local_func
    )
    if mapped_wasm_index != wasm_index:
        raise ComparisonError(
            f"WAMR local_func {config.wamr_local_func} maps to wasm function "
            f"{mapped_wasm_index}, not configured function {wasm_index}"
        )
    function = read_wamr_function(config.wamr_cwasm, config.wamr_local_func)
    return {
        "manifest": str(config.wamr_manifest),
        "manifest_sha256": sha256_file(config.wamr_manifest),
        "cwasm": str(config.wamr_cwasm),
        "cwasm_sha256": actual_cwasm_sha,
        "manifest_entry": {
            "idx": entry.get("idx"),
            "path": entry.get("path"),
            "core_sha256": entry_core_sha,
        },
        "local_func": config.wamr_local_func,
        "mapped_wasm_function_index": mapped_wasm_index,
    }, function


def capture(config_path: Path, work_dir: Path) -> dict[str, Any]:
    config = load_capture_config(config_path)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if not config.core_wasm.is_file():
        raise ComparisonError(f"raw core wasm not found: {config.core_wasm}")
    core_identity = parse_core_wasm(config.core_wasm)
    if core_identity.sha256 != config.core_wasm_sha256:
        raise ComparisonError(
            f"core wasm SHA-256 mismatch: expected {config.core_wasm_sha256}, "
            f"got {core_identity.sha256}"
        )
    wasm_index, function_name = resolve_wasm_function(
        core_identity, config.function_wasm_index, config.function_name
    )
    report, attribution, component_sha = _validate_benchmark(config)
    wamr_identity, wamr_function = _validate_wamr_identity(
        config,
        report,
        attribution,
        component_sha,
        core_identity,
        wasm_index,
    )
    wamr_instructions, wamr_objdump_command, wamr_jump_tables = disassemble_wamr(
        wamr_function, work_dir
    )
    wamr_static = aggregate_static_metrics(
        wamr_instructions, wamr_function.code_size
    )
    wamr_dynamic, hotness = wamr_dynamic_metrics(attribution)

    wasmtime_instructions, wasmtime_code_size, wasmtime_provenance = (
        capture_wasmtime(report, config.core_wasm, wasm_index, work_dir)
    )
    wasmtime_static = aggregate_static_metrics(
        wasmtime_instructions, wasmtime_code_size
    )
    if config.wasmtime_profile is None:
        reason = (
            "bench_keyvault.py profiles WAMR only, and Wasmtime "
            "component symbols expose a compilation-local wasm[N] ordinal but no "
            "raw core SHA-256; provide an explicitly verified profile JSON rather "
            "than guessing that ordinal"
        )
        wasmtime_dynamic = {name: _unavailable(reason) for name in METRIC_ORDER}
    else:
        assert config.wasmtime_profile_sha256 is not None
        wasmtime_dynamic = load_explicit_profile(
            config.wasmtime_profile,
            config.wasmtime_profile_sha256,
            "wasmtime",
            core_identity.sha256,
            wasm_index,
            function_name,
        )

    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "kind": "wamr-wasmtime-hot-function-capture",
        "identity": {
            "component_sha256": component_sha,
            "core_wasm": str(config.core_wasm),
            "core_wasm_sha256": core_identity.sha256,
            "imported_function_count": core_identity.imported_function_count,
            "local_function_count": core_identity.local_function_count,
            "wasm_function_index": wasm_index,
            "function_name": function_name,
        },
        "hotness": hotness,
        "engines": {
            "wamr": {
                "function_identity": {
                    "local_func": config.wamr_local_func,
                    "wasm_function_index": wasm_index,
                },
                "static": wamr_static,
                "dynamic": wamr_dynamic,
                "provenance": {
                    **wamr_identity,
                    "objdump_command": wamr_objdump_command,
                    "inline_jump_tables_excluded_from_instruction_counts": wamr_jump_tables,
                    "attribution": {
                        "perf": attribution.get("perf"),
                        "total_samples": attribution.get("total_samples"),
                        "attributed_samples": attribution.get("attributed_samples"),
                    },
                },
            },
            "wasmtime": {
                "function_identity": {
                    "standalone_module_index": 0,
                    "wasm_function_index": wasm_index,
                },
                "static": wasmtime_static,
                "dynamic": wasmtime_dynamic,
                "provenance": wasmtime_provenance,
            },
        },
        "provenance": {
            "config": str(config.path),
            "config_sha256": sha256_file(config.path),
            "benchmark_report": str(config.benchmark_report),
            "benchmark_report_sha256": config.benchmark_report_sha256,
        },
        "limitations": [
            (
                "Wasmtime static code is compiled from the exact raw core module "
                "with the benchmark's recorded options. This avoids equating WAMR "
                "local_func and Wasmtime component-local module ordinals."
            ),
            (
                "Wasmtime dynamic samples remain unavailable unless an external "
                "profile supplies the same core SHA-256 and wasm function index "
                "with explicit mapping evidence."
            ),
            (
                "Static instruction categories can overlap (for example an "
                "indirect call counts as both a call and indirect dispatch)."
            ),
        ],
    }


def _static_value(engine: dict[str, Any], metric: str) -> int | None:
    static = _object(engine.get("static"), "engine.static")
    item = _object(static.get(metric), f"engine.static.{metric}")
    if item.get("status") != "measured":
        return None
    return _integer(item.get("value"), f"engine.static.{metric}.value")


def _round(value: float) -> float:
    return round(value, 6)


def _validate_capture(capture_doc: dict[str, Any]) -> None:
    if (
        capture_doc.get("schema_version") != CAPTURE_SCHEMA_VERSION
        or capture_doc.get("kind") != "wamr-wasmtime-hot-function-capture"
    ):
        raise ComparisonError("capture has an unsupported schema or kind")
    identity = _object(capture_doc.get("identity"), "capture.identity")
    _sha256(identity.get("component_sha256"), "capture component_sha256")
    _sha256(identity.get("core_wasm_sha256"), "capture core_wasm_sha256")
    wasm_index = _integer(
        identity.get("wasm_function_index"), "capture wasm_function_index"
    )
    hotness = _object(capture_doc.get("hotness"), "capture.hotness")
    if hotness.get("status") != "measured":
        raise ComparisonError("capture hotness must be measured")
    total_samples = _integer(
        hotness.get("total_samples"), "capture hotness total_samples", 1
    )
    function_samples = _integer(
        hotness.get("function_samples"), "capture hotness function_samples", 1
    )
    if function_samples > total_samples:
        raise ComparisonError("capture hotness function samples exceed total samples")
    share = _number(hotness.get("percent_of_run"), "capture hotness percent_of_run")
    if share > 100:
        raise ComparisonError("capture hotness percent_of_run cannot exceed 100")
    if abs(share - 100.0 * function_samples / total_samples) > 0.05:
        raise ComparisonError("capture hotness percentage is inconsistent with samples")
    engines = _object(capture_doc.get("engines"), "capture.engines")
    for engine_name in ("wamr", "wasmtime"):
        engine = _object(engines.get(engine_name), f"capture.engines.{engine_name}")
        function_identity = _object(
            engine.get("function_identity"), f"{engine_name}.function_identity"
        )
        if (
            _integer(
                function_identity.get("wasm_function_index"),
                f"{engine_name}.function_identity.wasm_function_index",
            )
            != wasm_index
        ):
            raise ComparisonError(
                f"{engine_name} capture function identity does not match"
            )
        static = _object(engine.get("static"), f"{engine_name}.static")
        dynamic = _object(engine.get("dynamic"), f"{engine_name}.dynamic")
        for metric in METRIC_ORDER:
            item = _object(static.get(metric), f"{engine_name}.static.{metric}")
            if item.get("status") == "measured":
                _integer(item.get("value"), f"{engine_name}.static.{metric}.value")
            elif item.get("status") == "unavailable":
                _string(item.get("reason"), f"{engine_name}.static.{metric}.reason")
            else:
                raise ComparisonError(
                    f"{engine_name}.static.{metric}.status must be measured or unavailable"
                )
            dyn = _object(dynamic.get(metric), f"{engine_name}.dynamic.{metric}")
            if dyn.get("status") == "measured":
                _integer(dyn.get("samples"), f"{engine_name}.dynamic.{metric}.samples")
                pct = _number(
                    dyn.get("percent_of_function_samples"),
                    f"{engine_name}.dynamic.{metric}.percent_of_function_samples",
                )
                if pct > 100:
                    raise ComparisonError(
                        f"{engine_name}.dynamic.{metric} percentage exceeds 100"
                    )
            elif dyn.get("status") in {"unavailable", "partial"}:
                _string(dyn.get("reason"), f"{engine_name}.dynamic.{metric}.reason")
            else:
                raise ComparisonError(
                    f"{engine_name}.dynamic.{metric}.status is invalid"
                )


def compare_capture(
    capture_doc: dict[str, Any],
    gate_headroom_pct: float = 5.0,
    gate_min_differences: int = 2,
) -> dict[str, Any]:
    _validate_capture(capture_doc)
    engines = _object(capture_doc["engines"], "capture.engines")
    wamr = _object(engines["wamr"], "capture.engines.wamr")
    wasmtime = _object(engines["wasmtime"], "capture.engines.wasmtime")
    hotness = _object(capture_doc["hotness"], "capture.hotness")
    hot_share = float(hotness["percent_of_run"])
    wamr_total_instructions = _static_value(wamr, "native_instructions")
    if not wamr_total_instructions:
        raise ComparisonError("WAMR native instruction count must be positive")

    metric_reports: dict[str, Any] = {}
    ranked: list[dict[str, Any]] = []
    for metric in METRIC_ORDER:
        wamr_static = _object(wamr["static"][metric], f"wamr.static.{metric}")
        wasmtime_static = _object(
            wasmtime["static"][metric], f"wasmtime.static.{metric}"
        )
        wamr_dynamic = _object(wamr["dynamic"][metric], f"wamr.dynamic.{metric}")
        wasmtime_dynamic = _object(
            wasmtime["dynamic"][metric], f"wasmtime.dynamic.{metric}"
        )
        w_value = _static_value(wamr, metric)
        t_value = _static_value(wasmtime, metric)
        delta: dict[str, Any]
        if w_value is None or t_value is None:
            delta = {
                "status": "unavailable",
                "reason": "both static values are required for a delta",
            }
        else:
            absolute = w_value - t_value
            ratio = None if t_value == 0 else _round(w_value / t_value)
            reduction = 0.0 if w_value == 0 else max(0.0, absolute / w_value)
            headroom: dict[str, Any]
            if metric == "frame_size_bytes":
                headroom = {
                    "status": "unavailable",
                    "reason": "fixed frame bytes are space, not a direct runtime cost",
                }
            elif metric == "code_size_bytes":
                headroom = {
                    "status": "unavailable",
                    "reason": "code-byte reduction alone is not a measured execution share",
                }
            elif metric == "native_instructions":
                headroom = {
                    "status": "estimated",
                    "percent_of_run": _round(hot_share * reduction),
                    "method": (
                        "measured WAMR hot-function share × static total-instruction "
                        "reduction; aggregate metric is not recommendation-gate evidence"
                    ),
                }
            elif wamr_dynamic.get("status") != "measured":
                static_share = w_value / wamr_total_instructions
                headroom = {
                    "status": "static-only",
                    "upper_bound_percent_of_run": _round(
                        hot_share * static_share * reduction
                    ),
                    "reason": (
                        "exact WAMR dynamic samples for this class are unavailable; "
                        "static-only estimates cannot clear the recommendation gate"
                    ),
                }
            else:
                dynamic_share = (
                    float(wamr_dynamic["percent_of_function_samples"]) / 100.0
                )
                static_share = w_value / wamr_total_instructions
                conservative_share = min(dynamic_share, static_share)
                headroom = {
                    "status": "estimated",
                    "percent_of_run": _round(
                        hot_share * conservative_share * reduction
                    ),
                    "method": (
                        "measured WAMR hot-function share × min(WAMR dynamic "
                        "sample share, WAMR static instruction share) × static "
                        "Wasmtime reduction fraction"
                    ),
                    "dynamic_share_of_function": _round(dynamic_share * 100.0),
                    "static_share_of_function": _round(static_share * 100.0),
                    "static_reduction_fraction_pct": _round(reduction * 100.0),
                }
            delta = {
                "status": "measured",
                "wamr_minus_wasmtime": absolute,
                "wamr_over_wasmtime": ratio,
                "wasmtime_static_reduction_pct": _round(reduction * 100.0),
                "conservative_theoretical_headroom": headroom,
            }
            if absolute > 0:
                ranked.append(
                    {
                        "metric": metric,
                        "label": METRIC_LABELS[metric],
                        "wamr_minus_wasmtime": absolute,
                        "wasmtime_static_reduction_pct": _round(reduction * 100.0),
                        "theoretical_headroom_pct": (
                            headroom.get("percent_of_run")
                            if headroom.get("status") == "estimated"
                            else None
                        ),
                        "headroom_status": headroom.get("status"),
                    }
                )
        metric_reports[metric] = {
            "label": METRIC_LABELS[metric],
            "unit": METRIC_UNITS[metric],
            "wamr": {"static": wamr_static, "dynamic": wamr_dynamic},
            "wasmtime": {
                "static": wasmtime_static,
                "dynamic": wasmtime_dynamic,
            },
            "delta": delta,
        }

    ranked.sort(
        key=lambda item: (
            item["theoretical_headroom_pct"] is None,
            -(item["theoretical_headroom_pct"] or 0.0),
            -item["wasmtime_static_reduction_pct"],
            item["metric"],
        )
    )
    for index, item in enumerate(ranked, 1):
        item["rank"] = index

    qualified: list[dict[str, Any]] = []
    for item in ranked:
        if item["metric"] not in GATE_METRICS:
            continue
        headroom = item["theoretical_headroom_pct"]
        if headroom is not None and headroom >= gate_headroom_pct:
            qualified.append(item)
    if len(qualified) >= gate_min_differences:
        levers: dict[str, list[str]] = {}
        for item in qualified:
            lever = LEVER_FOR_METRIC[item["metric"]]
            levers.setdefault(lever, []).append(item["metric"])
        recommendation = {
            "status": "recommend",
            "gate": {
                "minimum_differences": gate_min_differences,
                "minimum_headroom_pct_each": gate_headroom_pct,
                "qualified_differences": len(qualified),
            },
            "optimizations": [
                {"lever": lever, "evidence_metrics": metrics}
                for lever, metrics in sorted(levers.items())
            ],
            "evidence": qualified,
            "warning": "Headroom values are conservative theoretical estimates and are not additive.",
        }
    else:
        recommendation = {
            "status": "no-lever-clears-gate",
            "gate": {
                "minimum_differences": gate_min_differences,
                "minimum_headroom_pct_each": gate_headroom_pct,
                "qualified_differences": len(qualified),
            },
            "evidence": qualified,
            "reason": (
                "Fewer than two explicitly evidenced instruction-class "
                f"differences reach {gate_headroom_pct:g}% conservative "
                "theoretical run-wide headroom."
            ),
        }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "wamr-wasmtime-hot-function-comparison",
        "identity": capture_doc["identity"],
        "hotness": hotness,
        "metrics": metric_reports,
        "ranked_deltas": ranked,
        "recommendation": recommendation,
        "methodology": {
            "static": (
                "Exact native instruction counts and function byte extents from "
                "disassembly; move-form frame accesses and other classes are "
                "conservative x86_64 syntax classification."
            ),
            "dynamic": (
                "Perf self-sample counts, not executed-instruction counts. "
                "Unavailable and partial metrics remain explicit."
            ),
            "headroom": (
                "Measured WAMR hot-function run share multiplied by the smaller "
                "of its static and dynamic class shares, then by the cross-engine "
                "static reduction fraction."
            ),
        },
        "limitations": capture_doc.get("limitations", []),
        "capture_provenance": capture_doc.get("provenance", {}),
    }


def _format_static(item: dict[str, Any]) -> str:
    if item.get("status") != "measured":
        return f"unavailable ({item.get('reason', 'no reason')})"
    return str(item["value"])


def _format_dynamic(item: dict[str, Any]) -> str:
    if item.get("status") != "measured":
        return f"{item.get('status', 'unavailable')} ({item.get('reason', 'no reason')})"
    return (
        f"{item['samples']} samples "
        f"({item['percent_of_function_samples']:.2f}% of function; "
        f"{item['percent_of_run_samples']:.2f}% of run)"
    )


def render_markdown(report: dict[str, Any]) -> str:
    identity = report["identity"]
    hotness = report["hotness"]
    lines = [
        "# WAMR / Wasmtime hot-function comparison",
        "",
        f"- Component SHA-256: `{identity['component_sha256']}`",
        f"- Core wasm SHA-256: `{identity['core_wasm_sha256']}`",
        f"- Wasm function: `{identity['wasm_function_index']}`"
        + (
            f" (`{identity['function_name']}`)"
            if identity.get("function_name") is not None
            else " (no name-section entry)"
        ),
        f"- WAMR measured hot share: **{hotness['percent_of_run']:.2f}% of run** "
        f"({hotness['function_samples']} / {hotness['total_samples']} self samples)",
        "",
        "Static counts and dynamic samples are different quantities. Dynamic values below are",
        "perf self samples, not executed-instruction counts; unavailable values are never treated",
        "as zero.",
        "",
        "| Metric | WAMR static | Wasmtime static | WAMR dynamic | Wasmtime dynamic | WAMR/Wasmtime | Headroom |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for metric in METRIC_ORDER:
        item = report["metrics"][metric]
        delta = item["delta"]
        ratio = (
            "unavailable"
            if delta.get("status") != "measured"
            or delta.get("wamr_over_wasmtime") is None
            else f"{delta['wamr_over_wasmtime']:.3f}×"
        )
        headroom = (
            delta.get("conservative_theoretical_headroom", {})
            if delta.get("status") == "measured"
            else {}
        )
        if headroom.get("status") == "estimated":
            headroom_text = f"{headroom['percent_of_run']:.2f}%"
        elif headroom.get("status") == "static-only":
            headroom_text = (
                f"static-only ≤{headroom['upper_bound_percent_of_run']:.2f}%"
            )
        else:
            headroom_text = "unavailable"
        lines.append(
            f"| {item['label']} | {_format_static(item['wamr']['static'])} | "
            f"{_format_static(item['wasmtime']['static'])} | "
            f"{_format_dynamic(item['wamr']['dynamic'])} | "
            f"{_format_dynamic(item['wasmtime']['dynamic'])} | {ratio} | "
            f"{headroom_text} |"
        )
    lines += ["", "## Ranked WAMR excesses", ""]
    if report["ranked_deltas"]:
        lines += [
            "| Rank | Metric | Static excess | Static reduction | Conservative run-wide headroom |",
            "|---:|---|---:|---:|---:|",
        ]
        for item in report["ranked_deltas"]:
            headroom = (
                "unavailable"
                if item["theoretical_headroom_pct"] is None
                else f"{item['theoretical_headroom_pct']:.2f}%"
            )
            lines.append(
                f"| {item['rank']} | {item['label']} | "
                f"{item['wamr_minus_wasmtime']} | "
                f"{item['wasmtime_static_reduction_pct']:.2f}% | {headroom} |"
            )
    else:
        lines.append("- No positive WAMR static deltas.")
    recommendation = report["recommendation"]
    lines += ["", "## Recommendation gate", ""]
    if recommendation["status"] == "recommend":
        lines.append(
            f"**Gate passed:** {recommendation['gate']['qualified_differences']} "
            "explicitly evidenced differences clear the threshold."
        )
        for optimization in recommendation["optimizations"]:
            lines.append(
                f"- Recommend **{optimization['lever']}**; evidence: "
                + ", ".join(f"`{name}`" for name in optimization["evidence_metrics"])
                + "."
            )
        lines.append(f"- {recommendation['warning']}")
    else:
        lines.append("**No optimization lever clears the evidence gate.**")
        lines.append(f"- {recommendation['reason']}")
    lines += ["", "## Limitations", ""]
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="UTF-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser(
        "capture", help="verify identities and capture compact machine-readable metrics"
    )
    capture_parser.add_argument("--config", type=Path, required=True)
    capture_parser.add_argument("--work-dir", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)

    report_parser = subparsers.add_parser(
        "report", help="compare a capture and render deterministic reports"
    )
    report_parser.add_argument("--capture", type=Path, required=True)
    report_parser.add_argument("--report-json", type=Path, required=True)
    report_parser.add_argument("--report-markdown", type=Path, required=True)
    report_parser.add_argument("--gate-headroom-pct", type=float, default=5.0)
    report_parser.add_argument("--gate-min-differences", type=int, default=2)

    args = parser.parse_args()
    try:
        if args.command == "capture":
            result = capture(args.config, args.work_dir)
            _write_json(args.output, result)
        else:
            if args.gate_headroom_pct < 0:
                raise ComparisonError("--gate-headroom-pct must be >= 0")
            if args.gate_min_differences < 1:
                raise ComparisonError("--gate-min-differences must be >= 1")
            captured = _load_json(args.capture, "capture")
            result = compare_capture(
                captured,
                gate_headroom_pct=args.gate_headroom_pct,
                gate_min_differences=args.gate_min_differences,
            )
            _write_json(args.report_json, result)
            args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
            args.report_markdown.write_text(
                render_markdown(result), encoding="UTF-8"
            )
    except ComparisonError as exc:
        print(f"hot-function comparison error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
