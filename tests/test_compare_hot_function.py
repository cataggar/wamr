#!/usr/bin/env python3
"""Controlled tests for WAMR/Wasmtime hot-function comparison tooling."""

from __future__ import annotations

import json
import platform
import shutil
import stat
import struct
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import compare_hot_function as compare  # noqa: E402


CAPTURE_SUPPORTED = (
    sys.platform.startswith("linux")
    and platform.machine() == "x86_64"
    and shutil.which("objdump") is not None
)


def leb(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            result.append(byte | 0x80)
        else:
            result.append(byte)
            return bytes(result)


def wasm_name(value: str) -> bytes:
    raw = value.encode("UTF-8")
    return leb(len(raw)) + raw


def wasm_section(section_id: int, payload: bytes) -> bytes:
    return bytes([section_id]) + leb(len(payload)) + payload


def make_core_wasm(function_names: list[str]) -> bytes:
    type_section = leb(1) + b"\x60\x00\x00"
    import_section = (
        leb(1) + wasm_name("env") + wasm_name("host") + b"\x00" + leb(0)
    )
    function_section = leb(len(function_names)) + b"".join(
        leb(0) for _ in function_names
    )
    code_section = leb(len(function_names)) + b"".join(
        leb(2) + b"\x00\x0b" for _ in function_names
    )
    name_map = leb(len(function_names))
    for index, name in enumerate(function_names, 1):
        name_map += leb(index) + wasm_name(name)
    custom_name = wasm_name("name") + b"\x01" + leb(len(name_map)) + name_map
    return (
        b"\0asm\x01\0\0\0"
        + wasm_section(1, type_section)
        + wasm_section(2, import_section)
        + wasm_section(3, function_section)
        + wasm_section(10, code_section)
        + wasm_section(0, custom_name)
    )


def make_wamr_cwasm(functions: list[bytes]) -> bytes:
    offsets: list[int] = []
    text = bytearray()
    for code in functions:
        offsets.append(len(text))
        text.extend(code)
    function_section = struct.pack("<I", len(functions))
    for offset in offsets:
        function_section += struct.pack("<II", offset, 0)
    return (
        struct.pack("<II", compare.AOT_MAGIC, compare.AOT_VERSION)
        + struct.pack("<II", compare.AOT_TEXT_SECTION, len(text))
        + text
        + struct.pack(
            "<II", compare.AOT_FUNCTION_SECTION, len(function_section)
        )
        + function_section
    )


WAMR_CODE = bytes.fromhex(
    "55 "
    "48 89 e5 "
    "48 83 ec 20 "
    "48 8b 45 f8 "
    "48 8b 55 f0 "
    "48 89 45 e8 "
    "48 89 55 e0 "
    "48 89 c1 "
    "48 89 d6 "
    "4c 8d 45 e0 "
    "ff d2 "
    "75 02 "
    "ff e0 "
    "c9 "
    "c3"
)

WASMTIME_OBJDUMP = """\
00000000 wasm[0]::function[1]::hot:
         0: 55                         push    rbp
         1: 48 89 e5                   mov     rbp,rsp
         4: 48 83 ec 10                sub     rsp,0x10
         8: 48 8b 45 f8                mov     rax,QWORD PTR [rbp-0x8]
         c: 48 89 c1                   mov     rcx,rax
         f: e8 00 00 00 00             call    0x14
        14: 75 01                      jne     0x17
        16: c9                         leave
        17: c3                         ret
        18: 00 00                      add     BYTE PTR [rax],al
"""

WASMTIME_EXPLORE_ASM = {
    "functions": [
        {
            "func_index": 1,
            "name": "hot",
            "demangled_name": "hot",
            "instructions": [
                {
                    "wasm_offset": None,
                    "address": 4096,
                    "bytes": [0x55],
                    "mnemonic": "push",
                    "operands": "rbp",
                },
                {
                    "wasm_offset": None,
                    "address": 4097,
                    "bytes": [0x48, 0x89, 0xE5],
                    "mnemonic": "mov",
                    "operands": "rbp, rsp",
                },
                {
                    "wasm_offset": None,
                    "address": 4100,
                    "bytes": [0x48, 0x83, 0xEC, 0x10],
                    "mnemonic": "sub",
                    "operands": "rsp, 0x10",
                },
                {
                    "wasm_offset": 10,
                    "address": 4104,
                    "bytes": [0x48, 0x8B, 0x45, 0xF8],
                    "mnemonic": "mov",
                    "operands": "rax, qword ptr [rbp - 8]",
                },
                {
                    "wasm_offset": 11,
                    "address": 4108,
                    "bytes": [0x48, 0x89, 0xC1],
                    "mnemonic": "mov",
                    "operands": "rcx, rax",
                },
                {
                    "wasm_offset": 12,
                    "address": 4111,
                    "bytes": [0xE8, 0, 0, 0, 0],
                    "mnemonic": "call",
                    "operands": "0x1014",
                },
                {
                    "wasm_offset": 13,
                    "address": 4116,
                    "bytes": [0x75, 0x01],
                    "mnemonic": "jne",
                    "operands": "0x1017",
                },
                {
                    "wasm_offset": 14,
                    "address": 4118,
                    "bytes": [0xC9],
                    "mnemonic": "leave",
                    "operands": "",
                },
                {
                    "wasm_offset": 14,
                    "address": 4119,
                    "bytes": [0xC3],
                    "mnemonic": "ret",
                    "operands": "",
                },
                {
                    "wasm_offset": None,
                    "address": 4120,
                    "bytes": [0, 0],
                    "mnemonic": "add",
                    "operands": "byte ptr [rax], al",
                },
            ],
        }
    ]
}


def measured_static(value: int, unit: str = "instructions") -> dict:
    return {"status": "measured", "value": value, "unit": unit, "method": "fixture"}


def measured_dynamic(samples: int, function_samples: int = 100) -> dict:
    return {
        "status": "measured",
        "samples": samples,
        "percent_of_function_samples": samples / function_samples * 100,
        "percent_of_run_samples": samples / 1.25,
        "method": "fixture samples",
    }


def comparison_capture(*, dynamic: bool = True) -> dict:
    wamr_values = {
        "native_instructions": 100,
        "code_size_bytes": 400,
        "frame_size_bytes": 64,
        "frame_loads": 30,
        "frame_stores": 20,
        "reg_reg_moves": 10,
        "address_generation": 4,
        "branches": 8,
        "indirect_dispatch": 2,
        "calls": 4,
    }
    wasmtime_values = {
        "native_instructions": 80,
        "code_size_bytes": 320,
        "frame_size_bytes": 32,
        "frame_loads": 10,
        "frame_stores": 10,
        "reg_reg_moves": 8,
        "address_generation": 4,
        "branches": 7,
        "indirect_dispatch": 1,
        "calls": 4,
    }
    dynamic_samples = {
        "native_instructions": 100,
        "frame_loads": 40,
        "frame_stores": 30,
        "reg_reg_moves": 10,
        "address_generation": 4,
        "branches": 8,
        "indirect_dispatch": 2,
        "calls": 4,
    }

    def dynamic_metrics(engine: str) -> dict:
        values: dict[str, dict] = {}
        for metric in compare.METRIC_ORDER:
            if metric in {"code_size_bytes", "frame_size_bytes"}:
                values[metric] = {
                    "status": "unavailable",
                    "reason": "not a sampled class",
                }
            elif engine == "wamr" and dynamic:
                values[metric] = measured_dynamic(dynamic_samples[metric])
            else:
                values[metric] = {
                    "status": "unavailable",
                    "reason": "fixture profile unavailable",
                }
        return values

    return {
        "schema_version": 1,
        "kind": "wamr-wasmtime-hot-function-capture",
        "identity": {
            "component_sha256": "1" * 64,
            "core_wasm_sha256": "2" * 64,
            "wasm_function_index": 7,
            "function_name": "hot",
        },
        "hotness": {
            "status": "measured",
            "source": "fixture",
            "total_samples": 125,
            "function_samples": 100,
            "percent_of_run": 80.0,
        },
        "engines": {
            "wamr": {
                "function_identity": {
                    "local_func": 3,
                    "wasm_function_index": 7,
                },
                "static": {
                    metric: measured_static(
                        value, "bytes" if metric.endswith("_bytes") else "instructions"
                    )
                    for metric, value in wamr_values.items()
                },
                "dynamic": dynamic_metrics("wamr"),
            },
            "wasmtime": {
                "function_identity": {
                    "standalone_module_index": 0,
                    "wasm_function_index": 7,
                },
                "static": {
                    metric: measured_static(
                        value, "bytes" if metric.endswith("_bytes") else "instructions"
                    )
                    for metric, value in wasmtime_values.items()
                },
                "dynamic": dynamic_metrics("wasmtime"),
            },
        },
        "limitations": ["controlled fixture; not a real workload result"],
        "provenance": {"fixture": True},
    }


class HotFunctionComparisonTest(unittest.TestCase):
    scratch_root = ROOT / "zig-out"

    def setUp(self) -> None:
        self.scratch = self.scratch_root / f"hot-compare-{self._testMethodName}"
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.component = self.scratch / "component.wasm"
        self.component.write_bytes(b"controlled component")
        self.core_wasm = self.scratch / "hot-core.wasm"
        self.core_wasm.write_bytes(make_core_wasm(["hot"]))
        self.wamr_cwasm = self.scratch / "keyvault.4.cwasm"
        self.wamr_cwasm.write_bytes(make_wamr_cwasm([WAMR_CODE]))
        self.wamr_manifest = self.scratch / "keyvault.cwasm.json"
        self.wamr_manifest.write_text(
            json.dumps(
                {
                    "version": 2,
                    "wamr_build_id": "fixture",
                    "component_sha256": compare.sha256_file(self.component),
                    "modules": [
                        {
                            "idx": 4,
                            "path": self.wamr_cwasm.name,
                            "sha256": compare.sha256_file(self.wamr_cwasm),
                            "core_sha256": compare.sha256_file(self.core_wasm),
                        }
                    ],
                }
            ),
            encoding="UTF-8",
        )
        self.wasmtime = self.scratch / "wasmtime"
        self.wasmtime.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            "if [ \"${1:-}\" = --version ]; then\n"
            "  echo 'wasmtime 46.0.1'\n"
            "  exit 0\n"
            "fi\n"
            "case \"${1:-}\" in\n"
            "  compile)\n"
            "    out=''\n"
            "    while [ \"$#\" -gt 0 ]; do\n"
            "      if [ \"$1\" = -o ] || [ \"$1\" = --output ]; then\n"
            "        shift\n"
            "        out=$1\n"
            "      fi\n"
            "      shift\n"
            "    done\n"
            "    test -n \"$out\"\n"
            "    printf 'controlled wasmtime cwasm' > \"$out\"\n"
            "    ;;\n"
            "  explore)\n"
            "    out=''\n"
            "    while [ \"$#\" -gt 0 ]; do\n"
            "      if [ \"$1\" = -o ] || [ \"$1\" = --output ]; then\n"
            "        shift\n"
            "        out=$1\n"
            "      fi\n"
            "      shift\n"
            "    done\n"
            "    test -n \"$out\"\n"
            "    cat > \"$out\" <<'EOF'\n"
            "<html><script>window.ASM = "
            + json.dumps(WASMTIME_EXPLORE_ASM, separators=(",", ":"))
            + ";</script></html>\n"
            "EOF\n"
            "    ;;\n"
            "  objdump)\n"
            "    cat <<'EOF'\n"
            + WASMTIME_OBJDUMP
            + "EOF\n"
            "    ;;\n"
            "  *) exit 9 ;;\n"
            "esac\n",
            encoding="UTF-8",
        )
        self.wasmtime.chmod(self.wasmtime.stat().st_mode | stat.S_IXUSR)
        self.benchmark = self.scratch / "benchmark.json"
        self._write_benchmark()
        self.config = self.scratch / "comparison.json"
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _attribution(self) -> dict:
        classes = {
            "spill_load (reloads)": 30,
            "frame stores": 20,
            "reg-reg mov": 10,
            "ALU": 3,
            "bounds-check": 1,
            "linear-mem access": 1,
            "dispatch (computed-goto)": 5,
            "call": 4,
            "other branches": 1,
            "other": 0,
        }
        return {
            "schema_version": 1,
            "perf": str(self.scratch / "wamr.perf.data"),
            "cwasm": str(self.wamr_cwasm),
            "text_base": 0x100000,
            "text_size": len(WAMR_CODE),
            "function_count": 1,
            "total_samples": 100,
            "attributed_samples": 90,
            "attribution_coverage_pct": 90.0,
            "top_functions": [
                {
                    "local_func": 0,
                    "samples": 75,
                    "percent_of_run": 75.0,
                    "code_bytes": len(WAMR_CODE),
                }
            ],
            "classified_function": {
                "local_func": 0,
                "samples": 75,
                "percent_of_run": 75.0,
                "classes": {
                    name: {
                        "samples": samples,
                        "percent_of_run": float(samples),
                    }
                    for name, samples in classes.items()
                },
                "hottest_instructions": [],
            },
        }

    def _write_benchmark(self) -> None:
        document = {
            "schema_version": 1,
            "kind": "keyvault-tcgc-aot-benchmark",
            "validation": {
                "tools": {
                    "wasmtime": {
                        "path": str(self.wasmtime),
                        "sha256": compare.sha256_file(self.wasmtime),
                        "version": "wasmtime 46.0.1",
                    }
                },
                "inputs": {
                    "component": {
                        "path": str(self.component),
                        "sha256": compare.sha256_file(self.component),
                        "size": self.component.stat().st_size,
                    }
                },
            },
            "precompile": {
                "commands": {
                    "wasmtime": [
                        str(self.wasmtime),
                        "compile",
                        "-W",
                        "max-memory-size=4294967296",
                        "-O",
                        "opt-level=2",
                        str(self.component),
                        "-o",
                        str(self.scratch / "keyvault.wasmtime.cwasm"),
                    ]
                },
                "artifacts": [
                    {
                        "path": str(self.wamr_manifest),
                        "size": self.wamr_manifest.stat().st_size,
                        "sha256": compare.sha256_file(self.wamr_manifest),
                    },
                    {
                        "path": str(self.wamr_cwasm),
                        "size": self.wamr_cwasm.stat().st_size,
                        "sha256": compare.sha256_file(self.wamr_cwasm),
                    },
                ],
            },
            "perf": {"selected": True, "attribution": self._attribution()},
        }
        self.benchmark.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="UTF-8"
        )

    def _write_config(
        self, *, wasm_index: int | None = 1, function_name: str | None = "hot"
    ) -> None:
        document = {
            "schema_version": 1,
            "benchmark_report": {
                "path": str(self.benchmark),
                "sha256": compare.sha256_file(self.benchmark),
            },
            "core_wasm": {
                "path": str(self.core_wasm),
                "sha256": compare.sha256_file(self.core_wasm),
            },
            "function": {"wasm_index": wasm_index, "name": function_name},
            "wamr": {
                "manifest": str(self.wamr_manifest),
                "cwasm": str(self.wamr_cwasm),
                "local_func": 0,
            },
            "wasmtime": {"profile": None},
        }
        self.config.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="UTF-8"
        )

    @unittest.skipUnless(CAPTURE_SUPPORTED, "capture requires Linux x86_64 objdump")
    def test_controlled_capture_matches_exact_module_and_function(self) -> None:
        capture = compare.capture(self.config, self.scratch / "work")
        self.assertEqual(capture["identity"]["wasm_function_index"], 1)
        self.assertEqual(capture["identity"]["function_name"], "hot")
        self.assertEqual(
            capture["engines"]["wamr"]["function_identity"]["local_func"], 0
        )
        self.assertEqual(
            capture["engines"]["wamr"]["static"]["frame_loads"]["value"], 2
        )
        self.assertEqual(
            capture["engines"]["wamr"]["static"]["frame_stores"]["value"], 2
        )
        self.assertEqual(
            capture["engines"]["wasmtime"]["static"]["frame_loads"]["value"], 1
        )
        self.assertEqual(
            capture["engines"]["wasmtime"]["static"]["native_instructions"][
                "value"
            ],
            9,
        )
        self.assertEqual(
            capture["engines"]["wasmtime"]["static"]["code_size_bytes"]["value"],
            26,
        )
        self.assertEqual(
            capture["engines"]["wasmtime"]["provenance"]["explore"][
                "trailing_unmapped_bytes_excluded_from_instruction_counts"
            ],
            2,
        )
        compile_command = capture["engines"]["wasmtime"]["provenance"][
            "compile_command"
        ]
        self.assertIn(str(self.core_wasm), compile_command)
        self.assertNotIn(str(self.component), compile_command)

    def test_mismatched_module_identity_fails(self) -> None:
        manifest = json.loads(self.wamr_manifest.read_text(encoding="UTF-8"))
        manifest["modules"][0]["core_sha256"] = "a" * 64
        self.wamr_manifest.write_text(json.dumps(manifest), encoding="UTF-8")
        self._write_benchmark()
        self._write_config()
        with self.assertRaisesRegex(compare.ComparisonError, "raw core module SHA-256"):
            compare.capture(self.config, self.scratch / "work")

    def test_mismatched_function_mapping_fails(self) -> None:
        self.core_wasm.write_bytes(make_core_wasm(["hot", "other"]))
        manifest = json.loads(self.wamr_manifest.read_text(encoding="UTF-8"))
        manifest["modules"][0]["core_sha256"] = compare.sha256_file(self.core_wasm)
        self.wamr_manifest.write_text(json.dumps(manifest), encoding="UTF-8")
        self._write_benchmark()
        self._write_config(wasm_index=2, function_name="other")
        with self.assertRaisesRegex(compare.ComparisonError, "maps to wasm function 1"):
            compare.capture(self.config, self.scratch / "work")

    def test_duplicate_names_fail_name_only_selection(self) -> None:
        path = self.scratch / "duplicates.wasm"
        path.write_bytes(make_core_wasm(["same", "same"]))
        identity = compare.parse_core_wasm(path)
        with self.assertRaisesRegex(compare.ComparisonError, "ambiguous"):
            compare.resolve_wasm_function(identity, None, "same")
        index, name = compare.resolve_wasm_function(identity, 2, "same")
        self.assertEqual((index, name), (2, "same"))

    def test_wasmtime_att_syntax_classification(self) -> None:
        text = """\
00000000 wasm[0]::function[1]::hot:
         0: 55                         pushq   %rbp
         1: 48 89 e5                   movq    %rsp, %rbp
         4: 48 83 ec 20                subq    $0x20, %rsp
         8: 48 8b 45 f8                movq    -0x8(%rbp), %rax
         c: 48 89 45 f0                movq    %rax, -0x10(%rbp)
        10: 48 89 c1                   movq    %rax, %rcx
        13: 48 8d 55 e0                leaq    -0x20(%rbp), %rdx
        17: ff d0                      callq   *%rax
        19: ff e1                      jmpq    *%rcx
"""
        instructions = compare.parse_disassembly(text, wasmtime_wasm_index=1)
        metrics = compare.aggregate_static_metrics(instructions, 27)
        self.assertEqual(metrics["frame_size_bytes"]["value"], 40)
        self.assertEqual(metrics["frame_loads"]["value"], 1)
        self.assertEqual(metrics["frame_stores"]["value"], 1)
        self.assertEqual(metrics["reg_reg_moves"]["value"], 2)
        self.assertEqual(metrics["address_generation"]["value"], 1)
        self.assertEqual(metrics["branches"]["value"], 1)
        self.assertEqual(metrics["indirect_dispatch"]["value"], 2)
        self.assertEqual(metrics["calls"]["value"], 1)

    @unittest.skipUnless(CAPTURE_SUPPORTED, "capture requires Linux x86_64 objdump")
    def test_wamr_inline_jump_table_bytes_are_not_instructions(self) -> None:
        code = bytes.fromhex(
            "31 c0 "
            "49 83 fb 02 "
            "0f 83 19 00 00 00 "
            "4c 8d 15 0a 00 00 00 "
            "4f 63 1c 9a "
            "4d 01 da "
            "41 ff e2 "
            "08 00 00 00 "
            "08 00 00 00 "
            "c3"
        )
        function = compare.CwasmFunction(
            code=code, code_size=len(code), function_count=1
        )
        instructions, _, tables = compare.disassemble_wamr(
            function, self.scratch
        )
        self.assertEqual(
            tables, [{"start": 29, "end": 37, "entries": 2}]
        )
        self.assertEqual(len(instructions), 8)
        self.assertEqual(instructions[-1].mnemonic, "ret")
        self.assertEqual(instructions[-1].offset, 37)

    def test_ambiguous_multiple_frame_allocations_are_unavailable(self) -> None:
        instructions = compare.parse_disassembly(
            """\
   0: 55             push rbp
   1: 48 89 e5       mov rbp,rsp
   4: 48 83 ec 40    sub rsp,0x40
   8: 48 89 5d f8    mov QWORD PTR [rbp-0x8],rbx
   c: 48 83 ec 10    sub rsp,0x10
"""
        )
        frame = compare.infer_frame_size(instructions)
        self.assertEqual(frame["status"], "unavailable")
        self.assertIn("multiple prologue", frame["reason"])

    def test_missing_dynamic_metrics_cannot_clear_gate(self) -> None:
        report = compare.compare_capture(comparison_capture(dynamic=False))
        self.assertEqual(
            report["recommendation"]["status"], "no-lever-clears-gate"
        )
        loads = report["metrics"]["frame_loads"]["delta"]
        self.assertEqual(
            loads["conservative_theoretical_headroom"]["status"], "static-only"
        )

    def test_ratio_and_conservative_headroom_math(self) -> None:
        report = compare.compare_capture(comparison_capture())
        loads = report["metrics"]["frame_loads"]["delta"]
        stores = report["metrics"]["frame_stores"]["delta"]
        self.assertEqual(loads["wamr_over_wasmtime"], 3.0)
        self.assertAlmostEqual(
            loads["conservative_theoretical_headroom"]["percent_of_run"],
            16.0,
        )
        self.assertAlmostEqual(
            stores["conservative_theoretical_headroom"]["percent_of_run"],
            8.0,
        )
        self.assertEqual(report["recommendation"]["status"], "recommend")
        self.assertEqual(
            report["recommendation"]["gate"]["qualified_differences"], 2
        )

    def test_malformed_json_and_capture_fail_closed(self) -> None:
        malformed = self.scratch / "malformed.json"
        malformed.write_text("{", encoding="UTF-8")
        with self.assertRaisesRegex(compare.ComparisonError, "invalid JSON"):
            compare._load_json(malformed, "fixture")
        capture = comparison_capture()
        del capture["engines"]["wamr"]["static"]["calls"]
        with self.assertRaisesRegex(compare.ComparisonError, "static.calls"):
            compare.compare_capture(capture)

    def test_explicit_profile_requires_matching_verified_mapping(self) -> None:
        profile = self.scratch / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "wasm-hot-function-profile",
                    "engine": "wasmtime",
                    "mapping_verified": True,
                    "mapping_evidence": "controlled fixture",
                    "module_sha256": compare.sha256_file(self.core_wasm),
                    "wasm_function_index": 2,
                    "function_name": "hot",
                    "total_samples": 100,
                    "function_samples": 50,
                    "metrics": {},
                }
            ),
            encoding="UTF-8",
        )
        with self.assertRaisesRegex(compare.ComparisonError, "function index"):
            compare.load_explicit_profile(
                profile,
                compare.sha256_file(profile),
                "wasmtime",
                compare.sha256_file(self.core_wasm),
                1,
                "hot",
            )

    def test_reports_are_deterministic(self) -> None:
        capture = comparison_capture()
        first = compare.compare_capture(capture)
        second = compare.compare_capture(json.loads(json.dumps(capture)))
        self.assertEqual(first, second)
        self.assertEqual(
            compare.render_markdown(first), compare.render_markdown(second)
        )
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )

    def test_checked_in_config_schema_and_example_track_parser(self) -> None:
        directory = ROOT / "tests" / "benchmarks" / "keyvault"
        schema = json.loads(
            (directory / "hot-comparison.schema.json").read_text(encoding="UTF-8")
        )
        example = json.loads(
            (directory / "hot-comparison.example.json").read_text(encoding="UTF-8")
        )
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            compare.CONFIG_SCHEMA_VERSION,
        )
        self.assertEqual(example["schema_version"], compare.CONFIG_SCHEMA_VERSION)
        self.assertEqual(example["wamr"]["local_func"], 6145)
        self.assertIsNone(example["wasmtime"]["profile"])

    @unittest.skipUnless(CAPTURE_SUPPORTED, "capture requires Linux x86_64 objdump")
    def test_cli_capture_and_report_smoke(self) -> None:
        capture_path = self.scratch / "capture.json"
        report_path = self.scratch / "report.json"
        markdown_path = self.scratch / "report.md"
        script = ROOT / "scripts" / "compare_hot_function.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "capture",
                "--config",
                str(self.config),
                "--work-dir",
                str(self.scratch / "work"),
                "--output",
                str(capture_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(script),
                "report",
                "--capture",
                str(capture_path),
                "--report-json",
                str(report_path),
                "--report-markdown",
                str(markdown_path),
            ],
            check=True,
        )
        report = json.loads(report_path.read_text(encoding="UTF-8"))
        self.assertEqual(
            report["identity"]["core_wasm_sha256"],
            compare.sha256_file(self.core_wasm),
        )
        self.assertIn(
            "Gate passed",
            markdown_path.read_text(encoding="UTF-8"),
        )


if __name__ == "__main__":
    unittest.main()
