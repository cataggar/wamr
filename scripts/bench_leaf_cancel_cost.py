#!/usr/bin/env python3
"""Measure AOT function-entry cancel-poll cost in a leaf-call-heavy thread."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import signal
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_schema import (
    atomic_write_json,
    cache_key,
    collected_at,
    command_identity,
    host_metadata,
    sample_stats,
    sha256_bytes,
    sha256_file,
)
from compare_hot_function import (
    ComparisonError,
    find_wamr_jump_tables,
    parse_disassembly,
)

KIND = "leaf-cancel-cost"
SCHEMA_VERSION = 1
AOT_VERSION = 11
AOT_TEXT_SECTION = 2
AOT_FUNCTION_SECTION = 3
AOT_EXPORT_SECTION = 4
AOT_IMPORT_SECTION = 8
LEAF_UNROLL = 64
LEAF_SEED = 0x243F6A8885A308D3
LEAF_STEP = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
MAX_LEAF_CALLS = 10_000_000_000
FIXTURE = Path("tests/benchmarks/leaf-cancel-cost/leaf_calls.wasm")
SOURCE_PATHS = (
    ".github/workflows/leaf-cancel-cost.yml",
    "tests/benchmarks/leaf-cancel-cost/README.md",
    "tests/benchmarks/leaf-cancel-cost/leaf_calls.c",
    "tests/benchmarks/leaf-cancel-cost/build-fixture.sh",
    "tests/benchmarks/leaf-cancel-cost/fixtures.sha256",
    "tests/benchmarks/leaf-cancel-cost/report.schema.json",
    "scripts/bench_leaf_cancel_cost.py",
    "scripts/test_bench_leaf_cancel_cost.py",
    "scripts/benchmark_schema.py",
    "scripts/compare_hot_function.py",
)
CANCEL_POLL_SEQUENCES = {
    "x86_64": bytes.fromhex(
        "83bbb001000000740c4889df488b87b8010000ffd0"
    ),
    "aarch64": struct.pack(
        "<IIIII",
        0xB941B270,
        0x34000090,
        0xAA1303E0,
        0xF940DE70,
        0xD63F0200,
    ),
}
CANCEL_POLL_ENTRY_PREFIX_SUFFIXES = {
    "x86_64": bytes.fromhex("4889fb4c8bbb00000000"),
    "aarch64": struct.pack("<III", 0xAA0003F3, 0xF9400274, 0xF9000FA1),
}
FIXTURE_TOOLCHAIN = {
    "wasi_sdk_version": "25.0",
    "clang": "clang version 19.1.5-wasi-sdk",
    "archive_sha256": (
        "52640dde13599bf127a95499e61d6d640256119456d1af8897ab6725bcf3d89c"
    ),
}
GUEST_KEYS = {
    "kind",
    "leaf_calls",
    "leaf_unroll",
    "batches",
    "checksum",
    "clock_id",
    "raw_elapsed_ns",
    "timing_overhead_ns",
    "elapsed_ns",
    "timing_overhead_ppm",
}


class HarnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Build:
    name: str
    prefix: Path
    wamr: Path
    wamrc: Path | None
    command: list[str]
    cache_key: str


@dataclass(frozen=True)
class AotExport:
    name: str
    kind: int
    index: int


@dataclass(frozen=True)
class AotImage:
    path: Path
    data: bytes
    sections: tuple[tuple[int, bytes], ...]
    text: bytes
    function_offsets: tuple[int, ...]
    function_type_indices: tuple[int, ...]
    exports: tuple[AotExport, ...]
    imported_function_count: int

    def function_code(self, local_index: int) -> bytes:
        if not 0 <= local_index < len(self.function_offsets):
            raise HarnessError(
                f"{self.path}: local function {local_index} is out of range"
            )
        start = self.function_offsets[local_index]
        end = (
            self.function_offsets[local_index + 1]
            if local_index + 1 < len(self.function_offsets)
            else len(self.text)
        )
        return self.text[start:end]


def expected_result(calls: int) -> dict[str, int | str]:
    if calls <= 0 or calls % LEAF_UNROLL != 0:
        raise HarnessError(f"leaf calls must be a positive multiple of {LEAF_UNROLL}")
    return {
        "kind": "leaf-cancel-cost-result",
        "leaf_calls": calls,
        "leaf_unroll": LEAF_UNROLL,
        "batches": calls // LEAF_UNROLL,
        "checksum": (LEAF_SEED + LEAF_STEP * calls) & MASK64,
        "clock_id": "wasi-process-cputime",
    }


def parse_guest_result(
    stdout: str,
    calls: int,
    minimum_interval_ns: int,
    *,
    enforce_quality: bool,
) -> dict[str, int | str]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise HarnessError(f"expected one guest JSON line, got {len(lines)}")
    try:
        result = json.loads(lines[0], object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise HarnessError("guest output is not valid JSON") from exc
    if set(result) != GUEST_KEYS:
        raise HarnessError(
            f"guest fields mismatch: expected {sorted(GUEST_KEYS)}, got {sorted(result)}"
        )
    for key, value in expected_result(calls).items():
        if result.get(key) != value:
            raise HarnessError(
                f"guest result mismatch for {key}: expected {value!r}, got {result.get(key)!r}"
            )
    for key in (
        "raw_elapsed_ns",
        "timing_overhead_ns",
        "elapsed_ns",
        "timing_overhead_ppm",
    ):
        if not isinstance(result[key], int) or isinstance(result[key], bool):
            raise HarnessError(f"{key} must be an integer")
    raw = int(result["raw_elapsed_ns"])
    overhead = int(result["timing_overhead_ns"])
    elapsed = int(result["elapsed_ns"])
    if raw <= 0 or overhead < 0 or elapsed <= 0 or raw - overhead != elapsed:
        raise HarnessError("guest timing values are inconsistent")
    if result["timing_overhead_ppm"] != overhead * 1_000_000 // raw:
        raise HarnessError("guest timing_overhead_ppm is inconsistent")
    if enforce_quality and 99 * overhead >= elapsed:
        raise HarnessError("guest timing overhead is not below one percent")
    if enforce_quality and elapsed < minimum_interval_ns:
        raise HarnessError(
            f"guest interval {elapsed}ns is below required {minimum_interval_ns}ns"
        )
    return result


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"duplicate guest key {key!r}")
        result[key] = value
    return result


def round_calls(value: int) -> int:
    if value <= 0:
        raise HarnessError("leaf call count must be positive")
    return ((value + LEAF_UNROLL - 1) // LEAF_UNROLL) * LEAF_UNROLL


def select_calls(
    pilot_calls: int,
    pilot_elapsed_ns: int,
    target_interval_ns: int,
) -> int:
    if pilot_elapsed_ns <= 0 or target_interval_ns <= 0:
        raise HarnessError("pilot and target intervals must be positive")
    selected = math.ceil(pilot_calls * target_interval_ns / pilot_elapsed_ns)
    if selected > MAX_LEAF_CALLS:
        raise HarnessError(
            f"selected leaf call count exceeds safety limit {MAX_LEAF_CALLS}"
        )
    return round_calls(max(selected, LEAF_UNROLL))


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def source_identity(repo: Path) -> dict[str, Any]:
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    files = {}
    for relative in SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise HarnessError(f"missing harness source: {relative}")
        files[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    build_paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "build.zig", "build.zig.zon", "src"],
        cwd=repo,
    ).split(b"\0")
    build_source = bytearray()
    for raw in build_paths:
        if not raw:
            continue
        build_source.extend(raw)
        build_source.append(0)
        build_source.extend((repo / raw.decode()).read_bytes())
        build_source.append(0)
    return {
        "commit": git_output(repo, "rev-parse", "HEAD"),
        "tracked_diff_sha256": sha256_bytes(diff),
        "build_source_sha256": sha256_bytes(bytes(build_source)),
        "harness_files": files,
    }


def controlled_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    directories = {
        "ZIG_GLOBAL_CACHE_DIR": root / "zig-global-cache",
        "ZIG_LOCAL_CACHE_DIR": root / "zig-local-cache",
        "XDG_CACHE_HOME": root / "xdg-cache",
        "TMPDIR": root / "tmp",
    }
    for key, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update({"LANG": "C", "LC_ALL": "C", "TZ": "UTC"})
    return env


def build_tools(
    repo: Path,
    output: Path,
    source: dict[str, Any],
    optimize: str,
    target: str | None,
    rebuild: bool,
) -> tuple[Build, Build]:
    compiler = build_one(
        repo=repo,
        output=output,
        source=source,
        name="host-compiler",
        optimize=optimize,
        target=None,
        compiler=True,
        rebuild=rebuild,
    )
    if target is None:
        return compiler, compiler
    runtime = build_one(
        repo=repo,
        output=output,
        source=source,
        name="target-runtime",
        optimize=optimize,
        target=target,
        compiler=False,
        rebuild=rebuild,
    )
    return compiler, runtime


def build_one(
    *,
    repo: Path,
    output: Path,
    source: dict[str, Any],
    name: str,
    optimize: str,
    target: str | None,
    compiler: bool,
    rebuild: bool,
) -> Build:
    parts = {
        "build_source_sha256": source["build_source_sha256"],
        "optimize": optimize,
        "target": target or "native",
        "compiler": compiler,
        "benchmark_cancel_point_toggle": True,
        "zig": command_identity(["zig", "version"]),
    }
    key = cache_key(parts)
    prefix = output / "builds" / f"{name}-{key[:16]}"
    wamr = prefix / "bin/wamr"
    wamrc = prefix / "bin/wamrc" if compiler else None
    marker = prefix / "leaf-cancel-build.json"
    expected = [wamr] + ([wamrc] if wamrc else [])
    if not rebuild and marker.is_file() and all(path.is_file() for path in expected):
        try:
            stored = json.loads(marker.read_text(encoding="UTF-8"))
        except json.JSONDecodeError:
            stored = {}
        if stored.get("cache_key") == key and isinstance(
            stored.get("command"), list
        ):
            return Build(name, prefix, wamr, wamrc, stored["command"], key)
    command = [
        "zig",
        "build",
        f"-Doptimize={optimize}",
        "-Dlib_wasi_threads=true",
        "-Dinterp=false",
        "-Daot=true",
        "-Dbenchmark-cancel-point-toggle=true",
        "--prefix",
        str(prefix),
    ]
    if target:
        command.insert(2, f"-Dtarget={target}")
    prefix.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            command,
            cwd=repo,
            env=controlled_env(output / "cache" / name),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HarnessError(f"{name} build failed with exit {exc.returncode}") from exc
    if not all(path.is_file() for path in expected):
        raise HarnessError(f"{name} build did not produce expected binaries")
    atomic_write_json(marker, {"cache_key": key, "parts": parts, "command": command})
    return Build(name, prefix, wamr, wamrc, command, key)


def fixture_identity(repo: Path) -> dict[str, Any]:
    fixture = repo / FIXTURE
    if not fixture.is_file():
        raise HarnessError(f"missing fixture: {fixture}")
    manifest = repo / FIXTURE.parent / "fixtures.sha256"
    fields = manifest.read_text(encoding="UTF-8").strip().split()
    if len(fields) != 2 or fields[1] != FIXTURE.name:
        raise HarnessError("invalid fixture checksum manifest")
    actual = sha256_file(fixture)
    if actual != fields[0]:
        raise HarnessError(
            f"fixture checksum mismatch: expected {fields[0]}, got {actual}"
        )
    return {
        "path": str(FIXTURE),
        "sha256": actual,
        "bytes": fixture.stat().st_size,
        "toolchain": FIXTURE_TOOLCHAIN,
    }


def compile_artifacts(
    repo: Path,
    output: Path,
    compiler: Build,
    arch: str,
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    if compiler.wamrc is None:
        raise HarnessError("host compiler is missing wamrc")
    directory = output / "aot"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "cancel-points-off": directory / "leaf-calls-polls-off.cwasm",
        "cancel-points-on": directory / "leaf-calls-polls-on.cwasm",
    }
    commands = {}
    for condition, path in artifacts.items():
        command = [str(compiler.wamrc), "compile", "--target", arch]
        if condition == "cancel-points-off":
            command.append("--benchmark-disable-cancel-points")
        command += [str(repo / FIXTURE), "-o", str(path)]
        commands[condition] = command
        try:
            subprocess.run(
                command,
                cwd=repo,
                env=controlled_env(output / "cache/compile-aot"),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise HarnessError(f"AOT compilation failed for {condition}") from exc
    return artifacts, commands


def _read_u32(payload: bytes, offset: int, label: str) -> tuple[int, int]:
    if offset + 4 > len(payload):
        raise HarnessError(f"{label}: truncated u32")
    return struct.unpack_from("<I", payload, offset)[0], offset + 4


def _read_bytes(
    payload: bytes, offset: int, size: int, label: str
) -> tuple[bytes, int]:
    if size < 0 or offset + size > len(payload):
        raise HarnessError(f"{label}: truncated byte range")
    return payload[offset : offset + size], offset + size


def _read_name(payload: bytes, offset: int, label: str) -> tuple[str, int]:
    size, offset = _read_u32(payload, offset, label)
    raw, offset = _read_bytes(payload, offset, size, label)
    try:
        return raw.decode("UTF-8"), offset
    except UnicodeDecodeError as exc:
        raise HarnessError(f"{label}: name is not UTF-8") from exc


def _parse_function_section(
    payload: bytes, label: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count, offset = _read_u32(payload, 0, label)
    expected_size = 4 + count * 8
    if len(payload) != expected_size:
        raise HarnessError(
            f"{label}: function section size {len(payload)} != {expected_size}"
        )
    offsets = []
    type_indices = []
    for _ in range(count):
        function_offset, offset = _read_u32(payload, offset, label)
        type_index, offset = _read_u32(payload, offset, label)
        offsets.append(function_offset)
        type_indices.append(type_index)
    return tuple(offsets), tuple(type_indices)


def _parse_exports(payload: bytes, label: str) -> tuple[AotExport, ...]:
    count, offset = _read_u32(payload, 0, label)
    exports = []
    for _ in range(count):
        name, offset = _read_name(payload, offset, label)
        raw_kind, offset = _read_bytes(payload, offset, 1, label)
        index, offset = _read_u32(payload, offset, label)
        exports.append(AotExport(name, raw_kind[0], index))
    if offset != len(payload):
        raise HarnessError(f"{label}: trailing export bytes")
    return tuple(exports)


def _parse_imported_function_count(payload: bytes, label: str) -> int:
    count, offset = _read_u32(payload, 0, label)
    functions = 0
    for _ in range(count):
        _, offset = _read_name(payload, offset, label)
        _, offset = _read_name(payload, offset, label)
        raw_kind, offset = _read_bytes(payload, offset, 1, label)
        kind = raw_kind[0]
        if kind == 0:
            functions += 1
            _, offset = _read_u32(payload, offset, label)
        elif kind == 1:
            _, offset = _read_bytes(payload, offset, 1, label)
            _, offset = _read_u32(payload, offset, label)
            has_max, offset = _read_bytes(payload, offset, 1, label)
            if has_max[0] not in (0, 1):
                raise HarnessError(f"{label}: invalid table maximum marker")
            if has_max[0]:
                _, offset = _read_u32(payload, offset, label)
        elif kind == 2:
            _, offset = _read_u32(payload, offset, label)
            has_max, offset = _read_bytes(payload, offset, 1, label)
            if has_max[0] not in (0, 1):
                raise HarnessError(f"{label}: invalid memory maximum marker")
            if has_max[0]:
                _, offset = _read_u32(payload, offset, label)
            _, offset = _read_bytes(payload, offset, 2, label)
        elif kind == 3:
            _, offset = _read_bytes(payload, offset, 2, label)
        elif kind == 4:
            _, offset = _read_u32(payload, offset, label)
        else:
            raise HarnessError(f"{label}: unsupported import kind {kind}")
    if offset != len(payload):
        raise HarnessError(f"{label}: trailing import bytes")
    return functions


def parse_aot(path: Path) -> AotImage:
    data = path.read_bytes()
    if len(data) < 8 or data[:4] != b"\x00aot":
        raise HarnessError(f"not a WAMR AOT artifact: {path}")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != AOT_VERSION:
        raise HarnessError(f"unsupported AOT version {version}")
    offset = 8
    sections = []
    by_type = {}
    while offset + 8 <= len(data):
        section_type, size = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + size > len(data):
            raise HarnessError(f"truncated AOT artifact: {path}")
        if section_type in by_type:
            raise HarnessError(f"{path}: duplicate AOT section {section_type}")
        payload = data[offset : offset + size]
        sections.append((section_type, payload))
        by_type[section_type] = payload
        offset += size
    if offset != len(data):
        raise HarnessError(f"{path}: trailing truncated AOT header")
    for section_type in (
        AOT_TEXT_SECTION,
        AOT_FUNCTION_SECTION,
        AOT_EXPORT_SECTION,
        AOT_IMPORT_SECTION,
    ):
        if section_type not in by_type:
            raise HarnessError(f"{path}: missing AOT section {section_type}")
    function_offsets, type_indices = _parse_function_section(
        by_type[AOT_FUNCTION_SECTION], f"{path} function section"
    )
    text = by_type[AOT_TEXT_SECTION]
    if (
        not function_offsets
        or function_offsets[0] != 0
        or tuple(sorted(set(function_offsets))) != function_offsets
        or function_offsets[-1] >= len(text)
    ):
        raise HarnessError(f"{path}: invalid AOT function offsets")
    return AotImage(
        path=path,
        data=data,
        sections=tuple(sections),
        text=text,
        function_offsets=function_offsets,
        function_type_indices=type_indices,
        exports=_parse_exports(
            by_type[AOT_EXPORT_SECTION], f"{path} export section"
        ),
        imported_function_count=_parse_imported_function_count(
            by_type[AOT_IMPORT_SECTION], f"{path} import section"
        ),
    )


def _poll_ranges(code: bytes, sequence: bytes) -> list[tuple[int, int]]:
    ranges = []
    offset = 0
    while True:
        start = code.find(sequence, offset)
        if start < 0:
            return ranges
        ranges.append((start, start + len(sequence)))
        offset = start + len(sequence)


def _normalize_local_offset(
    offset: int, poll_ranges: list[tuple[int, int]], label: str
) -> int:
    removed = 0
    for start, end in poll_ranges:
        if start < offset < end:
            raise HarnessError(f"{label}: control-flow target enters a cancel poll")
        if end <= offset:
            removed += end - start
    return offset - removed


def _function_index_for_offset(image: AotImage, offset: int) -> int:
    if offset == len(image.text):
        return len(image.function_offsets)
    for index, start in enumerate(image.function_offsets):
        end = (
            image.function_offsets[index + 1]
            if index + 1 < len(image.function_offsets)
            else len(image.text)
        )
        if start <= offset < end:
            return index
    raise HarnessError(f"{image.path}: text target {offset} is out of range")


def _normalize_on_text_offset(
    enabled: AotImage,
    disabled: AotImage,
    poll_ranges: list[list[tuple[int, int]]],
    offset: int,
) -> int:
    function_index = _function_index_for_offset(enabled, offset)
    if function_index == len(enabled.function_offsets):
        return len(disabled.text)
    local = offset - enabled.function_offsets[function_index]
    return disabled.function_offsets[function_index] + _normalize_local_offset(
        local,
        poll_ranges[function_index],
        f"{enabled.path} function {function_index}",
    )


def _x86_relative_instruction(raw: bytes) -> tuple[bytes, int] | None:
    if len(raw) == 5 and raw[0] in (0xE8, 0xE9):
        return raw[:1], struct.unpack_from("<i", raw, 1)[0]
    if len(raw) == 2 and (
        raw[0] == 0xEB or 0x70 <= raw[0] <= 0x7F
    ):
        return raw[:1], struct.unpack_from("<b", raw, 1)[0]
    if len(raw) == 6 and raw[0] == 0x0F and 0x80 <= raw[1] <= 0x8F:
        return raw[:2], struct.unpack_from("<i", raw, 2)[0]
    return None


def _disassemble_x86_function(
    code: bytes, work_dir: Path, label: str
) -> tuple[list[Any], list[dict[str, int]]]:
    try:
        tables = find_wamr_jump_tables(code)
    except ComparisonError as exc:
        raise HarnessError(f"{label}: {exc}") from exc
    disassembly = bytearray(code)
    for table in tables:
        disassembly[table["start"] : table["end"]] = b"\x90" * (
            table["end"] - table["start"]
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch = work_dir / f"{label}.bin"
    scratch.write_bytes(disassembly)
    command = [
        "objdump",
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
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    finally:
        scratch.unlink(missing_ok=True)
    if result.returncode != 0:
        raise HarnessError(
            f"{label}: objdump failed with exit {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    try:
        instructions = parse_disassembly(result.stdout)
    except ComparisonError as exc:
        raise HarnessError(f"{label}: {exc}") from exc
    if max(item.offset + item.size for item in instructions) != len(code):
        raise HarnessError(f"{label}: disassembly did not cover the function")
    if any(item.mnemonic in ("(bad)", ".byte") for item in instructions):
        raise HarnessError(f"{label}: undecodable x86 instruction")
    return (
        [
            item
            for item in instructions
            if not any(
                table["start"] <= item.offset < table["end"]
                for table in tables
            )
        ],
        tables,
    )


def _compare_x86_function(
    *,
    function_index: int,
    enabled: AotImage,
    disabled: AotImage,
    enabled_code: bytes,
    disabled_code: bytes,
    all_poll_ranges: list[list[tuple[int, int]]],
    work_dir: Path,
) -> tuple[int, int]:
    on_instructions, on_tables = _disassemble_x86_function(
        enabled_code, work_dir, f"on-{function_index}"
    )
    off_instructions, off_tables = _disassemble_x86_function(
        disabled_code, work_dir, f"off-{function_index}"
    )
    function_polls = all_poll_ranges[function_index]
    poll_instruction_offsets = set()
    for start, end in function_polls:
        selected = [
            item
            for item in on_instructions
            if start <= item.offset < end
        ]
        if (
            not selected
            or selected[0].offset != start
            or selected[-1].offset + selected[-1].size != end
            or b"".join(item.raw_bytes for item in selected)
            != CANCEL_POLL_SEQUENCES["x86_64"]
        ):
            raise HarnessError(
                f"enabled function {function_index}: malformed cancel poll"
            )
        poll_instruction_offsets.update(item.offset for item in selected)
    on_retained = [
        item
        for item in on_instructions
        if item.offset not in poll_instruction_offsets
    ]
    if len(on_retained) != len(off_instructions):
        raise HarnessError(
            f"function {function_index}: normalized instruction count differs"
        )
    layout_fields = 0
    for on_item, off_item in zip(on_retained, off_instructions, strict=True):
        normalized_offset = _normalize_local_offset(
            on_item.offset,
            function_polls,
            f"enabled function {function_index}",
        )
        if (
            normalized_offset != off_item.offset
            or on_item.size != off_item.size
            or on_item.mnemonic != off_item.mnemonic
        ):
            raise HarnessError(
                f"function {function_index}: normalized instruction layout differs"
            )
        if on_item.raw_bytes == off_item.raw_bytes:
            continue
        on_relative = _x86_relative_instruction(on_item.raw_bytes)
        off_relative = _x86_relative_instruction(off_item.raw_bytes)
        if (
            on_relative is None
            or off_relative is None
            or on_relative[0] != off_relative[0]
        ):
            raise HarnessError(
                f"function {function_index}: unrelated x86 instruction difference "
                f"at normalized offset {off_item.offset}"
            )
        on_target = (
            enabled.function_offsets[function_index]
            + on_item.offset
            + on_item.size
            + on_relative[1]
        )
        off_target = (
            disabled.function_offsets[function_index]
            + off_item.offset
            + off_item.size
            + off_relative[1]
        )
        if (
            _normalize_on_text_offset(
                enabled, disabled, all_poll_ranges, on_target
            )
            != off_target
        ):
            raise HarnessError(
                f"function {function_index}: x86 control-flow target differs"
            )
        layout_fields += 1
    if len(on_tables) != len(off_tables):
        raise HarnessError(f"function {function_index}: jump-table count differs")
    table_entries = 0
    for on_table, off_table in zip(on_tables, off_tables, strict=True):
        if (
            on_table["entries"] != off_table["entries"]
            or _normalize_local_offset(
                on_table["start"],
                function_polls,
                f"enabled function {function_index}",
            )
            != off_table["start"]
        ):
            raise HarnessError(
                f"function {function_index}: jump-table layout differs"
            )
        for entry_index in range(on_table["entries"]):
            on_entry = on_table["start"] + entry_index * 4
            off_entry = off_table["start"] + entry_index * 4
            on_relative = struct.unpack_from("<i", enabled_code, on_entry)[0]
            off_relative = struct.unpack_from("<i", disabled_code, off_entry)[0]
            on_target = on_table["start"] + on_relative
            off_target = off_table["start"] + off_relative
            if (
                _normalize_local_offset(
                    on_target,
                    function_polls,
                    f"enabled function {function_index}",
                )
                != off_target
            ):
                raise HarnessError(
                    f"function {function_index}: jump-table target differs"
                )
            table_entries += 1
    return layout_fields, table_entries


def _sign_extend(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value ^ sign) - sign


def _aarch64_relative_instruction(word: int) -> tuple[int, int] | None:
    if word & 0xFC000000 in (0x14000000, 0x94000000):
        return word & 0xFC000000, _sign_extend(word & 0x03FFFFFF, 26) << 2
    if word & 0xFF000010 == 0x54000000:
        return word & ~0x00FFFFE0, _sign_extend(
            (word >> 5) & 0x7FFFF, 19
        ) << 2
    if word & 0x7E000000 == 0x34000000:
        return word & ~0x00FFFFE0, _sign_extend(
            (word >> 5) & 0x7FFFF, 19
        ) << 2
    if word & 0x7E000000 == 0x36000000:
        return word & ~0x0007FFE0, _sign_extend(
            (word >> 5) & 0x3FFF, 14
        ) << 2
    return None


def _compare_aarch64_function(
    *,
    function_index: int,
    enabled: AotImage,
    disabled: AotImage,
    enabled_code: bytes,
    disabled_code: bytes,
    all_poll_ranges: list[list[tuple[int, int]]],
) -> int:
    if len(enabled_code) % 4 or len(disabled_code) % 4:
        raise HarnessError(f"function {function_index}: unaligned AArch64 code")
    sequence = CANCEL_POLL_SEQUENCES["aarch64"]
    function_polls = all_poll_ranges[function_index]
    for start, end in function_polls:
        if start % 4 or enabled_code[start:end] != sequence:
            raise HarnessError(
                f"enabled function {function_index}: malformed cancel poll"
            )
    stripped = bytearray()
    cursor = 0
    for start, end in function_polls:
        stripped.extend(enabled_code[cursor:start])
        cursor = end
    stripped.extend(enabled_code[cursor:])
    if len(stripped) != len(disabled_code):
        raise HarnessError(
            f"function {function_index}: normalized AArch64 size differs"
        )
    layout_fields = 0
    for offset in range(0, len(disabled_code), 4):
        on_word = struct.unpack_from("<I", stripped, offset)[0]
        off_word = struct.unpack_from("<I", disabled_code, offset)[0]
        if on_word == off_word:
            continue
        on_relative = _aarch64_relative_instruction(on_word)
        off_relative = _aarch64_relative_instruction(off_word)
        if (
            on_relative is None
            or off_relative is None
            or on_relative[0] != off_relative[0]
        ):
            raise HarnessError(
                f"function {function_index}: unrelated AArch64 word difference "
                f"at normalized offset {offset}"
            )
        on_original_offset = offset
        for start, end in function_polls:
            if start <= on_original_offset:
                on_original_offset += end - start
        on_target = (
            enabled.function_offsets[function_index]
            + on_original_offset
            + on_relative[1]
        )
        off_target = (
            disabled.function_offsets[function_index]
            + offset
            + off_relative[1]
        )
        if (
            _normalize_on_text_offset(
                enabled, disabled, all_poll_ranges, on_target
            )
            != off_target
        ):
            raise HarnessError(
                f"function {function_index}: AArch64 control-flow target differs"
            )
        layout_fields += 1
    return layout_fields


def _compile_command_identity(
    artifacts: dict[str, Path],
    commands: dict[str, list[str]],
) -> dict[str, Any]:
    normalized = {}
    flag = "--benchmark-disable-cancel-points"
    for condition in ("cancel-points-off", "cancel-points-on"):
        command = commands.get(condition)
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise HarnessError(f"missing compile command for {condition}")
        expected_flags = 1 if condition == "cancel-points-off" else 0
        if command.count(flag) != expected_flags or command.count("-o") != 1:
            raise HarnessError(f"invalid compile command for {condition}")
        output_index = command.index("-o") + 1
        if output_index >= len(command) or command[output_index] != str(
            artifacts[condition]
        ):
            raise HarnessError(f"compile output path mismatch for {condition}")
        normalized_command = []
        index = 0
        while index < len(command):
            item = command[index]
            if item == flag:
                index += 1
                continue
            if item == "-o":
                normalized_command += ["-o", "<OUTPUT>"]
                index += 2
                continue
            normalized_command.append(item)
            index += 1
        normalized[condition] = normalized_command
    if normalized["cancel-points-off"] != normalized["cancel-points-on"]:
        raise HarnessError(
            "AOT compile commands differ beyond the cancel-point flag and output"
        )
    return {
        "differing_flag": flag,
        "output_paths_condition_bound": True,
        "normalized_command": normalized["cancel-points-on"],
        "normalized_command_sha256": sha256_bytes(
            json.dumps(
                normalized["cancel-points-on"],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("UTF-8")
        ),
    }


def artifact_identity(
    artifacts: dict[str, Path],
    commands: dict[str, list[str]],
    arch: str,
    work_dir: Path,
) -> dict[str, Any]:
    if arch not in CANCEL_POLL_SEQUENCES:
        raise HarnessError(f"unsupported AOT architecture: {arch}")
    command_comparison = _compile_command_identity(artifacts, commands)
    images = {name: parse_aot(path) for name, path in artifacts.items()}
    enabled_image = images["cancel-points-on"]
    disabled_image = images["cancel-points-off"]
    if [item[0] for item in enabled_image.sections] != [
        item[0] for item in disabled_image.sections
    ]:
        raise HarnessError("AOT section ordering differs")
    for (on_type, on_payload), (off_type, off_payload) in zip(
        enabled_image.sections, disabled_image.sections, strict=True
    ):
        if (
            on_type not in (AOT_TEXT_SECTION, AOT_FUNCTION_SECTION)
            and on_payload != off_payload
        ):
            raise HarnessError(f"unrelated AOT section {on_type} differs")
    if (
        len(enabled_image.function_offsets)
        != len(disabled_image.function_offsets)
        or enabled_image.function_type_indices
        != disabled_image.function_type_indices
    ):
        raise HarnessError("AOT function identity metadata differs")

    sequence = CANCEL_POLL_SEQUENCES[arch]
    enabled_prefix = sequence[:9] if arch == "x86_64" else sequence[:8]
    enabled = enabled_image.text.count(sequence)
    disabled = disabled_image.text.count(sequence)
    if (
        enabled <= 0
        or disabled != 0
        or enabled_image.text.count(enabled_prefix) != enabled
        or disabled_image.text.count(enabled_prefix) != 0
    ):
        raise HarnessError(
            f"complete cancel-poll sequences invalid: "
            f"enabled={enabled}, disabled={disabled}"
        )
    all_poll_ranges = [
        _poll_ranges(enabled_image.function_code(index), sequence)
        for index in range(len(enabled_image.function_offsets))
    ]
    if sum(len(ranges) for ranges in all_poll_ranges) != enabled:
        raise HarnessError("cancel-poll sequence crosses a function boundary")
    cumulative_removed = 0
    layout_fields = 0
    jump_table_entries = 0
    for function_index, function_polls in enumerate(all_poll_ranges):
        if (
            enabled_image.function_offsets[function_index]
            != disabled_image.function_offsets[function_index]
            + cumulative_removed
        ):
            raise HarnessError("AOT function offsets have unrelated differences")
        enabled_code = enabled_image.function_code(function_index)
        disabled_code = disabled_image.function_code(function_index)
        removed = len(function_polls) * len(sequence)
        if len(enabled_code) != len(disabled_code) + removed:
            raise HarnessError(
                f"function {function_index}: size delta is not cancel polls"
            )
        if arch == "x86_64":
            fields, entries = _compare_x86_function(
                function_index=function_index,
                enabled=enabled_image,
                disabled=disabled_image,
                enabled_code=enabled_code,
                disabled_code=disabled_code,
                all_poll_ranges=all_poll_ranges,
                work_dir=work_dir,
            )
            layout_fields += fields
            jump_table_entries += entries
        else:
            layout_fields += _compare_aarch64_function(
                function_index=function_index,
                enabled=enabled_image,
                disabled=disabled_image,
                enabled_code=enabled_code,
                disabled_code=disabled_code,
                all_poll_ranges=all_poll_ranges,
            )
        cumulative_removed += removed
    delta = len(enabled_image.text) - len(disabled_image.text)
    if delta != enabled * len(sequence):
        raise HarnessError(
            f"AOT text delta {delta} is not exactly {enabled} complete polls"
        )

    leaf_exports = [
        item
        for item in enabled_image.exports
        if item.name == "leaf_step" and item.kind == 0
    ]
    if len(leaf_exports) != 1:
        raise HarnessError("enabled AOT does not have one leaf_step function export")
    leaf_export = leaf_exports[0]
    leaf_local = leaf_export.index - enabled_image.imported_function_count
    if (
        leaf_export.index < enabled_image.imported_function_count
        or not 0 <= leaf_local < len(enabled_image.function_offsets)
        or enabled_image.exports != disabled_image.exports
        or enabled_image.imported_function_count
        != disabled_image.imported_function_count
    ):
        raise HarnessError("leaf_step AOT function mapping is invalid")
    leaf_ranges = all_poll_ranges[leaf_local]
    leaf_disabled_code = disabled_image.function_code(leaf_local)
    leaf_enabled_code = enabled_image.function_code(leaf_local)
    entry_prefix_suffix = CANCEL_POLL_ENTRY_PREFIX_SUFFIXES[arch]
    if (
        len(leaf_ranges) != 1
        or sequence in leaf_disabled_code
        or not leaf_enabled_code[: leaf_ranges[0][0]].endswith(
            entry_prefix_suffix
        )
    ):
        raise HarnessError(
            "leaf_step does not have exactly one enabled entry poll after "
            "the architecture ABI prefix and none disabled"
        )
    return {
        "architecture": arch,
        "cancel_poll_sequence_hex": sequence.hex(),
        "cancel_poll_sites_enabled": enabled,
        "cancel_poll_sites_disabled": disabled,
        "text_delta_bytes": delta,
        "bytes_per_poll_site": len(sequence),
        "compile_commands": commands,
        "compile_command_comparison": command_comparison,
        "normalized_comparison": {
            "method": "complete-poll-and-layout-normalization-v1",
            "other_sections_byte_identical": True,
            "function_count": len(enabled_image.function_offsets),
            "function_type_indices_identical": True,
            "normalized_functions_identical": len(
                enabled_image.function_offsets
            ),
            "poll_sequences_removed": enabled,
            "relative_control_flow_fields_normalized": layout_fields,
            "jump_table_entries_normalized": jump_table_entries,
        },
        "leaf_step": {
            "export_function_index": leaf_export.index,
            "imported_function_count": enabled_image.imported_function_count,
            "local_function_index": leaf_local,
            "enabled_function_offset": enabled_image.function_offsets[leaf_local],
            "disabled_function_offset": disabled_image.function_offsets[leaf_local],
            "enabled_code_sha256": sha256_bytes(leaf_enabled_code),
            "disabled_code_sha256": sha256_bytes(leaf_disabled_code),
            "enabled_entry_poll_offset": leaf_ranges[0][0],
            "entry_prefix_suffix_hex": entry_prefix_suffix.hex(),
            "enabled_poll_sequences": 1,
            "disabled_poll_sequences": 0,
            "normalized_body_identical": True,
        },
        "conditions": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "file_bytes": path.stat().st_size,
                "text_bytes": len(images[name].text),
            }
            for name, path in sorted(artifacts.items())
        },
    }


def run_process(
    command: list[str], cwd: Path, timeout: float
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise HarnessError(f"command timed out after {timeout:g} seconds") from exc
    try:
        return process.returncode, stdout.decode(), stderr.decode()
    except UnicodeDecodeError as exc:
        raise HarnessError("guest output is not UTF-8") from exc


def selected_cpu_affinity() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        return []
    allowed = sorted(os.sched_getaffinity(0))
    return allowed[:1]


def measure(
    *,
    repo: Path,
    runner: list[str],
    runtime: Build,
    artifact: Path,
    condition: str,
    calls: int,
    timeout: float,
    minimum_interval_ns: int,
    enforce_quality: bool,
    phase: str,
    pair_index: int,
    position: int,
    leaf_entry_poll_proven: bool,
) -> dict[str, Any]:
    if not leaf_entry_poll_proven:
        raise HarnessError("leaf-entry poll opportunities lack artifact proof")
    affinity = selected_cpu_affinity()
    command = []
    if affinity:
        command += ["taskset", "--cpu-list", str(affinity[0])]
    command += [*runner, str(runtime.wamr), "run", str(artifact), str(calls)]
    started = time.perf_counter_ns()
    returncode, stdout, stderr = run_process(command, repo, timeout)
    host_elapsed = time.perf_counter_ns() - started
    if returncode != 0:
        raise HarnessError(
            f"{condition} guest exited {returncode}: {' '.join(command)}\n{stderr}"
        )
    guest = parse_guest_result(
        stdout, calls, minimum_interval_ns, enforce_quality=enforce_quality
    )
    elapsed = int(guest["elapsed_ns"])
    return {
        "phase": phase,
        "pair_index": pair_index,
        "position": position,
        "condition": condition,
        "command": command,
        "leaf_calls": calls,
        "leaf_entry_poll_opportunities": (
            calls if condition == "cancel-points-on" else 0
        ),
        "batches": calls // LEAF_UNROLL,
        "guest_elapsed_ns": elapsed,
        "raw_guest_elapsed_ns": int(guest["raw_elapsed_ns"]),
        "timing_overhead_ns": int(guest["timing_overhead_ns"]),
        "timing_overhead_ppm": int(guest["timing_overhead_ppm"]),
        "host_wall_elapsed_ns": host_elapsed,
        "leaf_calls_per_second": calls / (elapsed / 1e9),
        "guest": guest,
        "correct": True,
    }


def condition_order(pair_index: int) -> tuple[str, str]:
    return (
        ("cancel-points-off", "cancel-points-on")
        if pair_index % 2 == 0
        else ("cancel-points-on", "cancel-points-off")
    )


def summarize(records: list[dict[str, Any]], calls: int) -> tuple[list[dict], dict]:
    summaries = []
    for condition in ("cancel-points-off", "cancel-points-on"):
        selected = [item for item in records if item["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "guest_elapsed_ns": sample_stats(
                    (item["guest_elapsed_ns"] for item in selected),
                    "samples",
                ),
                "leaf_calls_per_second": sample_stats(
                    (item["leaf_calls_per_second"] for item in selected),
                    "samples",
                ),
                "host_wall_elapsed_ns": sample_stats(
                    (item["host_wall_elapsed_ns"] for item in selected),
                    "samples",
                ),
            }
        )
    pair_rows = []
    for pair_index in sorted({item["pair_index"] for item in records}):
        pair = {
            item["condition"]: item
            for item in records
            if item["pair_index"] == pair_index
        }
        off = pair["cancel-points-off"]["guest_elapsed_ns"]
        on = pair["cancel-points-on"]["guest_elapsed_ns"]
        pair_rows.append(
            {
                "pair_index": pair_index,
                "on_over_off_elapsed_ratio": on / off,
                "on_minus_off_ns": on - off,
                "on_minus_off_ns_per_leaf_call": (on - off) / calls,
                "on_over_off_throughput_ratio": off / on,
            }
        )
    return summaries, {
        "pairs": pair_rows,
        "median_on_over_off_elapsed_ratio": statistics.median(
            item["on_over_off_elapsed_ratio"] for item in pair_rows
        ),
        "median_elapsed_delta_percent": (
            statistics.median(item["on_over_off_elapsed_ratio"] for item in pair_rows)
            - 1.0
        )
        * 100.0,
        "median_on_minus_off_ns_per_leaf_call": statistics.median(
            item["on_minus_off_ns_per_leaf_call"] for item in pair_rows
        ),
        "median_on_over_off_throughput_ratio": statistics.median(
            item["on_over_off_throughput_ratio"] for item in pair_rows
        ),
    }


def validate_invocations(
    invocations: list[dict[str, Any]],
    *,
    phase: str,
    pairs: int,
    calls: int,
) -> None:
    if len(invocations) != pairs * 2:
        raise HarnessError(f"{phase} invocation count mismatch")
    expected_guest = expected_result(calls)
    for pair_index in range(pairs):
        pair = invocations[pair_index * 2 : pair_index * 2 + 2]
        for position, (item, condition) in enumerate(
            zip(pair, condition_order(pair_index), strict=True)
        ):
            if (
                item.get("phase") != phase
                or item.get("pair_index") != pair_index
                or item.get("position") != position
                or item.get("condition") != condition
            ):
                raise HarnessError(f"{phase} invocation ordering mismatch")
            if item.get("leaf_calls") != calls or item.get("batches") != (
                calls // LEAF_UNROLL
            ):
                raise HarnessError(f"{phase} invocation work mismatch")
            expected_opportunities = calls if condition == "cancel-points-on" else 0
            if (
                item.get("leaf_entry_poll_opportunities")
                != expected_opportunities
            ):
                raise HarnessError(f"{phase} poll opportunity mismatch")
            guest = item.get("guest")
            if not isinstance(guest, dict) or any(
                guest.get(key) != value for key, value in expected_guest.items()
            ):
                raise HarnessError(f"{phase} guest work mismatch")
            if item.get("correct") is not True:
                raise HarnessError(f"{phase} correctness marker mismatch")


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != KIND:
        raise HarnessError("report identity mismatch")
    plan = report.get("plan")
    records = report.get("records")
    if not isinstance(plan, dict) or not isinstance(records, list) or not records:
        raise HarnessError("report plan or records missing")
    calls = plan.get("leaf_calls")
    if not isinstance(calls, int) or isinstance(calls, bool):
        raise HarnessError("report leaf call count is invalid")
    expected_result(calls)
    artifacts = report.get("metadata", {}).get("artifacts")
    if not isinstance(artifacts, dict):
        raise HarnessError("report artifact proof missing")
    leaf_proof = artifacts.get("leaf_step")
    normalized = artifacts.get("normalized_comparison")
    if (
        artifacts.get("cancel_poll_sites_enabled", 0) <= 0
        or artifacts.get("cancel_poll_sites_disabled") != 0
        or not isinstance(leaf_proof, dict)
        or leaf_proof.get("enabled_poll_sequences") != 1
        or leaf_proof.get("disabled_poll_sequences") != 0
        or leaf_proof.get("normalized_body_identical") is not True
        or not isinstance(normalized, dict)
        or normalized.get("method")
        != "complete-poll-and-layout-normalization-v1"
        or normalized.get("other_sections_byte_identical") is not True
        or normalized.get("function_type_indices_identical") is not True
        or normalized.get("normalized_functions_identical")
        != normalized.get("function_count")
        or normalized.get("poll_sequences_removed")
        != artifacts.get("cancel_poll_sites_enabled")
        or artifacts.get("cancel_poll_sequence_hex")
        != CANCEL_POLL_SEQUENCES.get(artifacts.get("architecture"), b"").hex()
        or leaf_proof.get("entry_prefix_suffix_hex")
        != CANCEL_POLL_ENTRY_PREFIX_SUFFIXES.get(
            artifacts.get("architecture"), b""
        ).hex()
    ):
        raise HarnessError("report artifact or leaf-entry proof mismatch")
    command_paths = {
        condition: Path(details["path"])
        for condition, details in artifacts.get("conditions", {}).items()
    }
    command_comparison = _compile_command_identity(
        command_paths, artifacts.get("compile_commands", {})
    )
    if artifacts.get("compile_command_comparison") != command_comparison:
        raise HarnessError("report compile command identity mismatch")
    samples = plan.get("samples")
    warmup_pairs = plan.get("warmups")
    if (
        not isinstance(samples, int)
        or isinstance(samples, bool)
        or not isinstance(warmup_pairs, int)
        or isinstance(warmup_pairs, bool)
    ):
        raise HarnessError("report sample counts are invalid")
    validate_invocations(records, phase="sample", pairs=samples, calls=calls)
    warmups = report.get("warmups")
    if not isinstance(warmups, list):
        raise HarnessError("report warmups missing")
    validate_invocations(
        warmups, phase="warmup", pairs=warmup_pairs, calls=calls
    )
    pilot = report.get("pilot")
    sizing = plan.get("sizing")
    if not isinstance(sizing, dict):
        raise HarnessError("report sizing missing")
    if pilot is None:
        if sizing.get("kind") != "explicit-call-count":
            raise HarnessError("explicit sizing identity mismatch")
    else:
        pilot_calls = sizing.get("pilot_calls")
        if (
            sizing.get("kind") != "single-enabled-pilot-linear-scale"
            or not isinstance(pilot_calls, int)
            or isinstance(pilot_calls, bool)
        ):
            raise HarnessError("pilot sizing identity mismatch")
        if (
            pilot.get("phase") != "pilot"
            or pilot.get("pair_index") != 0
            or pilot.get("position") != 0
            or pilot.get("condition") != "cancel-points-on"
            or pilot.get("leaf_calls") != pilot_calls
            or pilot.get("leaf_entry_poll_opportunities") != pilot_calls
            or pilot.get("correct") is not True
        ):
            raise HarnessError("pilot invocation mismatch")
        pilot_guest = pilot.get("guest")
        if not isinstance(pilot_guest, dict):
            raise HarnessError("pilot guest result missing")
        for key, value in expected_result(pilot_calls).items():
            if pilot_guest.get(key) != value:
                raise HarnessError("pilot guest work mismatch")
    expected_non_leaf_bound = calls // LEAF_UNROLL + 1
    expected_leaf_share = calls / (calls + expected_non_leaf_bound)
    if (
        plan.get("non_leaf_poll_opportunities_upper_bound_per_enabled_sample")
        != expected_non_leaf_bound
        or not math.isclose(
            plan.get("minimum_leaf_entry_poll_share", -1),
            expected_leaf_share,
        )
        or sizing.get("selected_calls") != calls
    ):
        raise HarnessError("report poll-opportunity or sizing bound mismatch")
    quality_invocations = warmups + records
    checksums = {item["guest"]["checksum"] for item in quality_invocations}
    minimum_interval_ns = plan["minimum_interval_ns"]
    expected_quality = {
        "equivalent_guest_work": True,
        "equivalent_checksums": len(checksums) == 1,
        "all_intervals_passed": all(
            item["guest_elapsed_ns"] >= minimum_interval_ns
            for item in quality_invocations
        ),
        "all_timing_overhead_below_one_percent": all(
            99 * item["timing_overhead_ns"] < item["guest_elapsed_ns"]
            for item in quality_invocations
        ),
    }
    if report.get("quality") != expected_quality:
        raise HarnessError("report quality summary mismatch")


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    comparison = report["comparison"]
    source = metadata["source"]
    host = metadata["host"]
    tools = metadata["tools"]
    artifacts = metadata["artifacts"]
    plan = report["plan"]
    lines = [
        "# Leaf-call AOT cancel-poll cost",
        "",
        "## Immutable identities",
        "",
        f"- Source commit: `{source['commit']}`",
        f"- Tracked diff SHA-256: `{source['tracked_diff_sha256']}`",
        f"- Build-source SHA-256: `{source['build_source_sha256']}`",
        f"- Platform: `{metadata['platform_id']}`",
        f"- Host fingerprint: `{host['host_fingerprint']['sha256']}`",
        f"- Host: `{host['system']} {host['release']}` · `{host['machine']}` · "
        f"`{host['cpu']}` · {host['logical_cpus']} logical CPUs · "
        f"`{host['runner_environment']}`",
        f"- AOT architecture: `{artifacts['architecture']}`",
        f"- Fixture: `{metadata['fixture']['sha256']}`",
        f"- Zig: `{tools['zig']}`; Python: `{tools['python']}`",
        f"- Objdump: `{tools['objdump']}`",
        f"- wamrc: `{tools['compiler']['wamrc_sha256']}` · "
        f"`{tools['compiler']['wamrc_version']}`",
        f"- wamr: `{tools['runtime']['wamr_sha256']}` · "
        f"`{tools['runtime']['wamr_version']}`",
        "",
        "| AOT condition | SHA-256 | File bytes | Text bytes |",
        "|---|---|---:|---:|",
    ]
    for condition in ("cancel-points-off", "cancel-points-on"):
        artifact = artifacts["conditions"][condition]
        lines.append(
            f"| `{condition}` | `{artifact['sha256']}` | "
            f"{artifact['file_bytes']} | {artifact['text_bytes']} |"
        )
    lines += [
        "",
        "## AOT single-variable proof",
        "",
        f"- Normalization: "
        f"`{artifacts['normalized_comparison']['method']}`; "
        f"{artifacts['normalized_comparison']['normalized_functions_identical']} "
        "functions identical after removing "
        f"{artifacts['normalized_comparison']['poll_sequences_removed']} complete "
        "poll sequences and normalizing "
        f"{artifacts['normalized_comparison']['relative_control_flow_fields_normalized']} "
        "control-flow layout fields and "
        f"{artifacts['normalized_comparison']['jump_table_entries_normalized']} "
        "jump-table entries.",
        f"- Normalized compile command SHA-256: "
        f"`{artifacts['compile_command_comparison']['normalized_command_sha256']}`; "
        "the only semantic option difference is "
        f"`{artifacts['compile_command_comparison']['differing_flag']}`.",
        f"- `leaf_step`: wasm function "
        f"`{artifacts['leaf_step']['export_function_index']}`, local function "
        f"`{artifacts['leaf_step']['local_function_index']}`, enabled entry-poll "
        f"offset `{artifacts['leaf_step']['enabled_entry_poll_offset']}`; complete "
        "poll sequences on/off `1 / 0`; compiler ABI entry-prefix tail "
        f"`{artifacts['leaf_step']['entry_prefix_suffix_hex']}`; "
        "normalized body identical.",
        "",
        "| AOT condition | Exact compile command |",
        "|---|---|",
    ]
    for condition in ("cancel-points-off", "cancel-points-on"):
        command = shlex.join(artifacts["compile_commands"][condition]).replace(
            "|", "\\|"
        )
        lines.append(f"| `{condition}` | `{command}` |")
    lines += [
        "",
        "## Measurement plan and validation",
        "",
        f"- Work: `{plan['leaf_calls']}` noinline leaf calls "
        f"(`{plan['batches']}` batches of `{plan['leaf_unroll']}`).",
        f"- Invocations: `{plan['warmups']}` discarded balanced warmup pairs; "
        f"`{plan['samples']}` retained balanced sample pairs.",
        f"- Order: `{plan['pair_order']}`; raw rows below retain execution order.",
        f"- Sizing: `{plan['sizing']['kind']}`; target "
        f"`{plan['target_interval_ns']}` ns; minimum "
        f"`{plan['minimum_interval_ns']}` ns.",
        "- Validation covers every warmup and retained sample; the optional "
        "sizing pilot intentionally does not enforce the minimum interval.",
        f"- CPU affinity: `{metadata['execution']['host_cpu_affinity']}`; "
        f"target `{metadata['execution']['target']}`; "
        f"runner `{metadata['execution']['runner']}`.",
        f"- Timed enabled-path leaf-entry poll opportunities: "
        f"`{plan['leaf_entry_poll_opportunities_per_enabled_sample']}`; "
        "non-leaf timed poll opportunities are bounded by "
        f"`{plan['non_leaf_poll_opportunities_upper_bound_per_enabled_sample']}` "
        f"({plan['minimum_leaf_entry_poll_share']:.6%} minimum leaf-entry share).",
        f"- Static poll sites on/off: "
        f"`{artifacts['cancel_poll_sites_enabled']}` / "
        f"`{artifacts['cancel_poll_sites_disabled']}`; "
        f"`{artifacts['bytes_per_poll_site']}` bytes/site.",
        "",
        "| Validation | Result |",
        "|---|---|",
    ]
    for key, value in report["quality"].items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")
    lines += [
        "",
        "## Summary",
        "",
        "| Condition | Guest median ms | Median leaf calls/s | Host-wall median ms |",
        "|---|---:|---:|---:|",
    ]
    for summary in report["summaries"]:
        lines.append(
            f"| `{summary['condition']}` | "
            f"{summary['guest_elapsed_ns']['median'] / 1e6:.3f} | "
            f"{summary['leaf_calls_per_second']['median']:.3f} | "
            f"{summary['host_wall_elapsed_ns']['median'] / 1e6:.3f} |"
        )
    lines += [
        "",
        f"- Median polls-on / polls-off guest time: "
        f"`{comparison['median_on_over_off_elapsed_ratio']:.9f}` "
        f"(`{comparison['median_elapsed_delta_percent']:+.4f}%`).",
        f"- Median polls-on minus polls-off cost per timed leaf call: "
        f"`{comparison['median_on_minus_off_ns_per_leaf_call']:.6f} ns`.",
        f"- Median polls-on / polls-off throughput: "
        f"`{comparison['median_on_over_off_throughput_ratio']:.9f}`.",
        "",
        "The per-leaf-call delta is an amortized enabled-path measurement. "
        "The timed driver has at most one loop-header poll per 64 leaf calls "
        "plus one driver entry poll; the bound and leaf-entry share are retained "
        "above rather than presenting the delta as an uncontaminated single-poll "
        "latency.",
        "",
        "Both conditions use the same wasm fixture, runtime, call count, worker-thread "
        "path, and expected checksum. The retained compile commands are identical "
        "apart from condition-bound output paths and wamrc's benchmark-only "
        "cancel-point suppression flag; complete AOT normalization rejects every "
        "other section, function, instruction, or layout difference.",
        "",
        "## Raw invocation order",
        "",
        "| Phase | Pair | Position | Condition | Calls | Leaf polls | Checksum | "
        "Raw guest ns | Timer overhead ns | Guest ns | Host-wall ns | Correct | Command |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    invocations = []
    if report["pilot"] is not None:
        invocations.append(report["pilot"])
    invocations.extend(report["warmups"])
    invocations.extend(report["records"])
    for item in invocations:
        command = shlex.join(item["command"]).replace("|", "\\|")
        lines.append(
            f"| `{item['phase']}` | {item['pair_index']} | {item['position']} | "
            f"`{item['condition']}` | {item['leaf_calls']} | "
            f"{item['leaf_entry_poll_opportunities']} | "
            f"{item['guest']['checksum']} | {item['raw_guest_elapsed_ns']} | "
            f"{item['timing_overhead_ns']} | {item['guest_elapsed_ns']} | "
            f"{item['host_wall_elapsed_ns']} | "
            f"`{str(item['correct']).lower()}` | `{command}` |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("zig-out/leaf-cancel-cost")
    )
    parser.add_argument("--calls", type=int)
    parser.add_argument("--pilot-calls", type=int, default=4_194_304)
    parser.add_argument("--target-interval-ms", type=float, default=3000.0)
    parser.add_argument("--min-interval-ms", type=float, default=1250.0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--optimize", default="ReleaseFast")
    parser.add_argument("--target")
    parser.add_argument("--aot-target", choices=("x86_64", "aarch64"))
    parser.add_argument("--runner", default="")
    parser.add_argument(
        "--platform-id",
        default=f"local-{platform.system().lower()}-{platform.machine().lower()}",
    )
    parser.add_argument("--runner-environment")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    for name in ("calls", "pilot_calls"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or value % LEAF_UNROLL != 0):
            parser.error(f"--{name.replace('_', '-')} must be a positive multiple of 64")
        if value is not None and value > MAX_LEAF_CALLS:
            parser.error(
                f"--{name.replace('_', '-')} must not exceed {MAX_LEAF_CALLS}"
            )
    if args.target_interval_ms < args.min_interval_ms or args.min_interval_ms <= 0:
        parser.error("--target-interval-ms must be >= positive --min-interval-ms")
    if (
        args.warmups < 0
        or args.warmups > 2
        or args.samples <= 0
        or args.samples > 12
        or args.samples % 2
    ):
        parser.error(
            "--warmups must be between 0 and 2 and --samples positive, "
            "even, and at most 12"
        )
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def execution_arch(args: argparse.Namespace) -> str:
    if args.aot_target:
        return args.aot_target
    if args.target and args.target.startswith("aarch64"):
        return "aarch64"
    return "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale_report in (output / "report.json", output / "report.md"):
        stale_report.unlink(missing_ok=True)
    source = source_identity(repo)
    fixture = fixture_identity(repo)
    compiler, runtime = build_tools(
        repo, output, source, args.optimize, args.target, args.rebuild
    )
    arch = execution_arch(args)
    artifacts, compile_commands = compile_artifacts(repo, output, compiler, arch)
    artifact_report = artifact_identity(
        artifacts,
        compile_commands,
        arch,
        output / "cache/artifact-normalization",
    )
    leaf_entry_poll_proven = (
        artifact_report["leaf_step"]["enabled_poll_sequences"] == 1
        and artifact_report["leaf_step"]["disabled_poll_sequences"] == 0
        and artifact_report["leaf_step"]["normalized_body_identical"] is True
    )
    runner = shlex.split(args.runner)
    minimum_interval_ns = int(args.min_interval_ms * 1_000_000)
    target_interval_ns = int(args.target_interval_ms * 1_000_000)

    pilot = None
    calls = args.calls
    if calls is None:
        pilot = measure(
            repo=repo,
            runner=runner,
            runtime=runtime,
            artifact=artifacts["cancel-points-on"],
            condition="cancel-points-on",
            calls=args.pilot_calls,
            timeout=args.timeout,
            minimum_interval_ns=minimum_interval_ns,
            enforce_quality=False,
            phase="pilot",
            pair_index=0,
            position=0,
            leaf_entry_poll_proven=leaf_entry_poll_proven,
        )
        calls = select_calls(
            args.pilot_calls,
            pilot["guest_elapsed_ns"],
            target_interval_ns,
        )

    warmups = []
    for pair_index in range(args.warmups):
        for position, condition in enumerate(condition_order(pair_index)):
            warmups.append(
                measure(
                    repo=repo,
                    runner=runner,
                    runtime=runtime,
                    artifact=artifacts[condition],
                    condition=condition,
                    calls=calls,
                    timeout=args.timeout,
                    minimum_interval_ns=minimum_interval_ns,
                    enforce_quality=True,
                    phase="warmup",
                    pair_index=pair_index,
                    position=position,
                    leaf_entry_poll_proven=leaf_entry_poll_proven,
                )
            )

    records = []
    for pair_index in range(args.samples):
        for position, condition in enumerate(condition_order(pair_index)):
            records.append(
                measure(
                    repo=repo,
                    runner=runner,
                    runtime=runtime,
                    artifact=artifacts[condition],
                    condition=condition,
                    calls=calls,
                    timeout=args.timeout,
                    minimum_interval_ns=minimum_interval_ns,
                    enforce_quality=True,
                    phase="sample",
                    pair_index=pair_index,
                    position=position,
                    leaf_entry_poll_proven=leaf_entry_poll_proven,
                )
            )
    summaries, comparison = summarize(records, calls)
    host = host_metadata(args.runner_environment)
    tool_report = {
        "zig": command_identity(["zig", "version"]),
        "python": platform.python_version(),
        "objdump": command_identity(["objdump", "--version"]),
        "compiler": {
            "build_command": compiler.command,
            "cache_key": compiler.cache_key,
            "wamrc_path": str(compiler.wamrc),
            "wamrc_sha256": sha256_file(compiler.wamrc),
            "wamrc_version": command_identity([str(compiler.wamrc), "version"]),
        },
        "runtime": {
            "build_command": runtime.command,
            "cache_key": runtime.cache_key,
            "wamr_path": str(runtime.wamr),
            "wamr_sha256": sha256_file(runtime.wamr),
            "wamr_version": command_identity([*runner, str(runtime.wamr), "version"]),
        },
    }
    quality_invocations = warmups + records
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "metadata": {
            "collected_at": collected_at(),
            "platform_id": args.platform_id,
            "source": source,
            "fixture": fixture,
            "host": host,
            "tools": tool_report,
            "artifacts": artifact_report,
            "execution": {
                "target": args.target or "native",
                "aot_target": arch,
                "runner": runner,
                "host_cpu_affinity": selected_cpu_affinity(),
                "optimize": args.optimize,
            },
        },
        "plan": {
            "identity": "issue-963-leaf-cancel-cost-v1",
            "scope": "isolated-from-wasi-thread-benchmark-966",
            "leaf_calls": calls,
            "leaf_unroll": LEAF_UNROLL,
            "batches": calls // LEAF_UNROLL,
            "leaf_entry_poll_opportunities_per_enabled_sample": calls,
            "non_leaf_poll_opportunities_upper_bound_per_enabled_sample": (
                calls // LEAF_UNROLL + 1
            ),
            "minimum_leaf_entry_poll_share": calls
            / (calls + calls // LEAF_UNROLL + 1),
            "warmups": args.warmups,
            "samples": args.samples,
            "minimum_interval_ns": minimum_interval_ns,
            "target_interval_ns": target_interval_ns,
            "pair_order": "alternating-off-on/on-off",
            "clock_id": "wasi-process-cputime",
            "sizing": {
                "kind": "single-enabled-pilot-linear-scale"
                if pilot
                else "explicit-call-count",
                "pilot_calls": args.pilot_calls if pilot else None,
                "selected_calls": calls,
            },
        },
        "pilot": pilot,
        "warmups": warmups,
        "records": records,
        "summaries": summaries,
        "comparison": comparison,
        "quality": {
            "equivalent_guest_work": True,
            "equivalent_checksums": len(
                {item["guest"]["checksum"] for item in quality_invocations}
            )
            == 1,
            "all_intervals_passed": all(
                item["guest_elapsed_ns"] >= minimum_interval_ns
                for item in quality_invocations
            ),
            "all_timing_overhead_below_one_percent": all(
                99 * item["timing_overhead_ns"] < item["guest_elapsed_ns"]
                for item in quality_invocations
            ),
        },
    }
    validate_report(report)
    atomic_write_json(output / "report.json", report)
    (output / "report.md").write_text(render_markdown(report), encoding="UTF-8")
    print(output / "report.json")
    print(output / "report.md")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HarnessError, OSError, subprocess.SubprocessError) as exc:
        print(f"leaf cancel-cost benchmark failed: {exc}", file=sys.stderr)
        sys.exit(2)
