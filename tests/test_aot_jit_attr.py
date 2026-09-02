#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/skills/aot-perf-profile/aot_jit_attr.py"
SPEC = importlib.util.spec_from_file_location("aot_jit_attr", SCRIPT)
aot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aot
SPEC.loader.exec_module(aot)

WAMRC = None
if "--wamrc" in sys.argv:
    index = sys.argv.index("--wamrc")
    if index + 1 >= len(sys.argv):
        raise SystemExit("--wamrc requires a path")
    WAMRC = Path(sys.argv[index + 1]).resolve()
    del sys.argv[index : index + 2]


LOAD = bytes.fromhex("488b85d0fdffff")  # mov rax,QWORD PTR [rbp-0x230]
STORE = bytes.fromhex("488985d0fdffff")  # mov QWORD PTR [rbp-0x230],rax


def metadata_for(code=LOAD + STORE):
    return {
        "schema": aot.FRAME_SCHEMA,
        "schema_version": aot.FRAME_SCHEMA_VERSION,
        "cwasm_aot_version": aot.AOT_VERSION,
        "compiler_build_id": "test",
        "architecture": "x86_64",
        "abi": "sysv",
        "module": 4,
        "local_func": 7,
        "function_name": "fixture",
        "module_text_size": len(code),
        "module_text_sha256": hashlib.sha256(code).hexdigest(),
        "function_offset": 0,
        "code_size": len(code),
        "normalized_code_sha256": aot.normalized_code_sha256(code, []),
        "direct_call_rel32_offsets": [],
        "inline_data_ranges": [],
        "frame_layout": {
            "frame_pointer": "rbp",
            "frame_size": 576,
            "local_count": 2,
            "param_count": 0,
            "reserved_vmctx_offset": -8,
            "locals_first_offset": -16,
            "explicit_storage_first_offset": -32,
            "explicit_storage_slots": 64,
            "spill_base": -560,
            "spill_stride": -8,
            "spill_slots": 1,
        },
        "spill_metric": {
            "slots": 1,
            "spilled_vregs": 2,
            "scalar": 2,
            "v128": 0,
            "slots_scalar": 2,
            "slots_v128": 0,
            "spill_ld": 1,
            "spill_st": 1,
            "remat": 0,
            "callee_saved": 0,
        },
        "emitted_allocator_loads": 1,
        "emitted_allocator_stores": 1,
        "allocator_values": [
            {
                "vreg": 10,
                "frame_offset": -560,
                "slot": 0,
                "slot_count": 1,
                "value_type": "i64",
                "live_start": 0,
                "live_end": 3,
                "defining_opcode": "load",
                "source_class": "memory_or_runtime",
                "ir_use_count": 1,
                "ir_def_count": 1,
                "reload_count": 1,
                "store_count": 0,
                "rematerialization_eligible": False,
                "reused": True,
            },
            {
                "vreg": 11,
                "frame_offset": -560,
                "slot": 0,
                "slot_count": 1,
                "value_type": "i64",
                "live_start": 4,
                "live_end": 8,
                "defining_opcode": "add",
                "source_class": "computed",
                "ir_use_count": 0,
                "ir_def_count": 1,
                "reload_count": 0,
                "store_count": 0,
                "rematerialization_eligible": False,
                "reused": True,
            },
        ],
        "accesses": [
            {
                "native_start": 0,
                "native_end": len(LOAD),
                "kind": "load",
                "base": "rbp",
                "frame_offset": -560,
                "width": 8,
                "origin": "allocator_spill",
                "detail": "allocator_slot",
                "slot": 0,
                "local_index": None,
                "explicit_slot": None,
                "vreg": 10,
                "vreg_ambiguous": False,
                "defining_opcode": "load",
                "source_class": "memory_or_runtime",
                "rematerialization_eligible": False,
                "ir_position": 2,
                "ir_opcode": "add",
            },
            {
                "native_start": len(LOAD),
                "native_end": len(code),
                "kind": "store",
                "base": "rbp",
                "frame_offset": -560,
                "width": 8,
                "origin": "allocator_spill",
                "detail": "allocator_slot",
                "slot": 0,
                "local_index": None,
                "explicit_slot": None,
                "vreg": None,
                "vreg_ambiguous": True,
                "defining_opcode": None,
                "source_class": None,
                "rematerialization_eligible": None,
                "ir_position": None,
                "ir_opcode": None,
            },
        ],
    }


def load_from_dict(raw, code=LOAD + STORE):
    with mock.patch.object(
        aot.Path, "read_text", return_value=json.dumps(raw)
    ):
        return aot.load_frame_metadata(
            "metadata.json", 7, code, aot.AOT_VERSION, code, 0
        )


class FrameAttributionUnitTests(unittest.TestCase):
    def test_frame_operand_supports_signed_rsp_and_complex_forms(self):
        self.assertEqual(
            aot.FrameOperand("load", "rbp", -560, False),
            aot.parse_frame_operand("mov rax,QWORD PTR [rbp-0x230]"),
        )
        self.assertEqual(
            aot.FrameOperand("store", "rbp", 48, False),
            aot.parse_frame_operand("mov QWORD PTR [rbp+0x30],rax"),
        )
        self.assertEqual(
            aot.FrameOperand("load", "rsp", 0, False),
            aot.parse_frame_operand("mov rdx,QWORD PTR [rsp]"),
        )
        self.assertEqual(
            aot.FrameOperand("store", "rsp", -8, False),
            aot.parse_frame_operand("push r12"),
        )
        self.assertEqual(
            aot.FrameOperand("load", "rsp", 0, False),
            aot.parse_frame_operand("pop r12"),
        )
        complex_operand = aot.parse_frame_operand(
            "mov rax,QWORD PTR [rbp+rcx*8-0x20]"
        )
        self.assertEqual("rbp", complex_operand.base)
        self.assertIsNone(complex_operand.offset)
        self.assertTrue(complex_operand.complex_address)
        self.assertIsNone(
            aot.parse_frame_operand("lea rax,[rbp+rcx*8-0x20]")
        )

    def test_reused_slot_accepts_emitted_range_identity_and_ambiguity(self):
        metadata = load_from_dict(metadata_for())
        self.assertEqual(10, metadata.access_by_start[0]["vreg"])
        self.assertTrue(metadata.access_by_start[len(LOAD)]["vreg_ambiguous"])
        self.assertEqual([10, 11], [
            value["vreg"] for value in metadata.values_by_slot[0]
        ])

    def test_reused_slot_without_identity_or_ambiguity_fails(self):
        raw = metadata_for()
        raw["accesses"][1]["vreg_ambiguous"] = False
        with self.assertRaisesRegex(
            aot.AttributionError, "reused slot 0"
        ):
            load_from_dict(raw)

    def test_reconciliation_mismatch_fails(self):
        raw = metadata_for()
        raw["spill_metric"]["spill_ld"] = 2
        with self.assertRaisesRegex(
            aot.AttributionError, "reconciliation failed"
        ):
            load_from_dict(raw)

    def test_schema_hash_and_function_mismatches_fail_closed(self):
        raw = metadata_for()
        raw["schema_version"] = 99
        with self.assertRaisesRegex(aot.AttributionError, "schema_version"):
            load_from_dict(raw)

        raw = metadata_for()
        raw["normalized_code_sha256"] = "0" * 64
        with self.assertRaisesRegex(aot.AttributionError, "hash mismatch"):
            load_from_dict(raw)

        raw = metadata_for()
        raw["module_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(aot.AttributionError, "module text hash"):
            load_from_dict(raw)

        raw = metadata_for()
        raw["local_func"] = 8
        with self.assertRaisesRegex(aot.AttributionError, "does not match"):
            load_from_dict(raw)

    def test_malformed_and_overlapping_metadata_fail_closed(self):
        with mock.patch.object(aot.Path, "read_text", return_value="{"):
            with self.assertRaisesRegex(aot.AttributionError, "malformed"):
                aot.load_frame_metadata(
                    "broken.json",
                    7,
                    LOAD + STORE,
                    aot.AOT_VERSION,
                    LOAD + STORE,
                    0,
                )

        raw = metadata_for()
        raw["accesses"][1]["native_start"] = 1
        with self.assertRaisesRegex(
            aot.AttributionError, "overlapping/out-of-range"
        ):
            load_from_dict(raw)

    def test_disassembly_mapping_mismatch_fails(self):
        metadata = load_from_dict(metadata_for())
        instructions = [
            aot.Instruction(0x1000, 0, len(LOAD), "mov rax,QWORD PTR [rbp-0x230]"),
            aot.Instruction(
                0x1000 + len(LOAD),
                len(LOAD),
                len(STORE),
                "mov QWORD PTR [rbp-0x238],rax",
            ),
        ]
        with self.assertRaisesRegex(aot.AttributionError, "offset mismatch"):
            aot.validate_metadata_disassembly(metadata, instructions)

    def test_inline_br_table_ranges_split_disassembly(self):
        calls = []

        def fake_disassemble(
            blob, address, scratch, label, offset_base=0, architecture=None
        ):
            self.assertEqual("x86_64", architecture)
            calls.append((bytes(blob), address, offset_base))
            return [
                aot.Instruction(
                    address, offset_base, len(blob), "nop"
                )
            ] if blob else []

        with mock.patch.object(aot, "disassemble_blob", fake_disassemble):
            instructions = aot.disassemble_function(
                b"abcdefghijkl",
                0x1000,
                ROOT,
                "table",
                [{"native_start": 3, "native_end": 7, "kind": "br_table"}],
                "x86_64",
            )
        self.assertEqual(
            [(b"abc", 0x1000, 0), (b"hijkl", 0x1007, 7)], calls
        )
        self.assertEqual([0, 7], [item.offset for item in instructions])

    def test_mixed_origin_summary_reports_coverage_and_unknowns(self):
        accesses = {
            0: {
                "origin": "allocator_spill",
                "kind": "load",
                "slot": 0,
                "frame_offset": -560,
                "vreg": 10,
                "vreg_ambiguous": False,
                "defining_opcode": "load",
                "source_class": "memory_or_runtime",
                "rematerialization_eligible": False,
            },
            7: {
                "origin": "wasm_local_or_phi",
                "kind": "store",
                "frame_offset": -16,
                "vreg": None,
                "vreg_ambiguous": False,
            },
            14: {
                "origin": "explicit_frame_storage",
                "kind": "load",
                "frame_offset": -32,
                "vreg": None,
                "vreg_ambiguous": False,
            },
            21: {
                "origin": "fixed_runtime_frame_state",
                "kind": "store",
                "frame_offset": 32,
                "vreg": None,
                "vreg_ambiguous": False,
            },
        }
        value = metadata_for()["allocator_values"][0]
        metadata = aot.FrameMetadata(
            raw={},
            access_by_start=accesses,
            value_by_vreg={10: value},
            values_by_slot={0: [value]},
            inline_data_ranges=[],
            reconciliation={
                "emitted_allocator_loads": 1,
                "emitted_allocator_stores": 0,
                "spill_metric_loads": 1,
                "spill_metric_stores": 0,
                "matches": True,
            },
        )
        instructions = [
            aot.Instruction(0x1000, 0, 7, "mov rax,QWORD PTR [rbp-0x230]"),
            aot.Instruction(0x1007, 7, 7, "mov QWORD PTR [rbp-0x10],rax"),
            aot.Instruction(0x100E, 14, 7, "mov rax,QWORD PTR [rbp-0x20]"),
            aot.Instruction(0x1015, 21, 8, "mov QWORD PTR [rsp+0x20],rax"),
            aot.Instruction(0x101D, 29, 7, "mov rax,QWORD PTR [rbp-0x88]"),
        ]
        counts = {
            0x1000: 30,
            0x1007: 20,
            0x100E: 10,
            0x1015: 5,
            0x101D: 2,
        }
        summary = aot.build_frame_summary(instructions, counts, metadata)
        self.assertEqual(5, summary["coverage"]["frame_instructions"])
        self.assertEqual(4, summary["coverage"]["attributed_frame_instructions"])
        self.assertEqual(2, summary["coverage"]["unknown_frame_samples"])
        self.assertEqual(30, summary["origins"]["allocator_spill"]["samples"])
        self.assertEqual(
            20, summary["origins"]["wasm_local_or_phi"]["samples"]
        )

    def test_cwasm_version_and_duplicate_offsets_fail_closed(self):
        text = b"\x90\xc3"
        function = struct.pack("<I", 1) + struct.pack("<II", 0, 0)
        good = (
            struct.pack("<II", aot.AOT_MAGIC, aot.AOT_VERSION)
            + struct.pack("<II", aot.SEC_TEXT, len(text))
            + text
            + struct.pack("<II", aot.SEC_FUNCTION, len(function))
            + function
        )
        with mock.patch.object(aot.Path, "read_bytes", return_value=good):
            info = aot.parse_cwasm("ok.cwasm")
        self.assertEqual([0], info.func_offsets)

        bad_version = bytearray(good)
        struct.pack_into("<I", bad_version, 4, aot.AOT_VERSION - 1)
        with mock.patch.object(
            aot.Path, "read_bytes", return_value=bytes(bad_version)
        ):
            with self.assertRaisesRegex(aot.AttributionError, "incompatible"):
                aot.parse_cwasm("old.cwasm")

        duplicate_function = (
            struct.pack("<I", 2)
            + struct.pack("<II", 0, 0)
            + struct.pack("<II", 0, 0)
        )
        duplicate = (
            struct.pack("<II", aot.AOT_MAGIC, aot.AOT_VERSION)
            + struct.pack("<II", aot.SEC_TEXT, len(text))
            + text
            + struct.pack("<II", aot.SEC_FUNCTION, len(duplicate_function))
            + duplicate_function
        )
        with mock.patch.object(aot.Path, "read_bytes", return_value=duplicate):
            with self.assertRaisesRegex(
                aot.AttributionError, "ambiguous/non-increasing"
            ):
                aot.parse_cwasm("ambiguous.cwasm")

    def test_equal_size_jit_mappings_require_explicit_base(self):
        with mock.patch.object(
            aot, "jit_exec_mmaps", return_value=[(0x1000, 4096), (0x4000, 4096)]
        ):
            with self.assertRaisesRegex(
                aot.AttributionError, "exactly one"
            ):
                aot.select_text_base("perf.data", 4096)
        self.assertEqual(
            0x4000, aot.select_text_base("perf.data", 4096, "0x4000")
        )

    @mock.patch.object(aot, "_run_checked", return_value="")
    @mock.patch.dict(aot.os.environ, {"PERF": "/opt/perf"}, clear=False)
    def test_perf_binary_override_is_honored(self, run):
        aot.jit_exec_mmaps("perf.data")
        self.assertEqual("/opt/perf", run.call_args.args[0][0])

    def test_address_counts_fall_back_to_perf_script_self_ips(self):
        report = "no parseable per-address rows\n"
        script = """\
0xffff00001000 ([JIT])
0xffff00001000 ([JIT])
0xaaaa00002000 /usr/bin/wamr
"""
        with mock.patch.object(
            aot, "_run_checked", side_effect=[report, script]
        ):
            counts, total = aot.addr_counts("perf.data")
        self.assertEqual(3, total)
        self.assertEqual(2, counts[0xFFFF00001000])
        self.assertEqual(1, counts[0xAAAA00002000])

    def test_current_aot_version_matches_emitter(self):
        source = (ROOT / "src/compiler/emit_aot.zig").read_text()
        match = re.search(r"pub const aot_version: u32 = (\d+);", source)
        self.assertIsNotNone(match)
        self.assertEqual(aot.AOT_VERSION, int(match.group(1)))

    def test_aarch64_instruction_classes(self):
        text = [
            "add x17, x15, #1",
            "ldr x16, [x19, #8]",
            "cmp x17, x16",
            "b.ls 0x1020",
            "ldr w8, [x20, x15]",
            "ldr x9, [x29, #-16]",
            "str x10, [sp, #32]",
            "mov w11, w8",
            "add w12, w11, w10",
            "bl 0x2000",
            "br x13",
            "b 0x3000",
            "cbnz w8, 0x4000",
        ]
        instructions = [
            aot.Instruction(0x1000 + index * 4, index * 4, 4, item)
            for index, item in enumerate(text)
        ]
        self.assertEqual(
            [
                "alu",
                "mem_access",
                "bounds_cmp",
                "bounds_branch",
                "linear_memory",
                "frame_load_unattributed",
                "frame_store_unattributed",
                "regmov",
                "alu",
                "call",
                "indirect_dispatch",
                "direct_branch",
                "cond_branch",
            ],
            aot.classify_instruction_stream(
                instructions, architecture="aarch64"
            ),
        )

    def test_x86_classification_remains_compatible(self):
        self.assertEqual(
            "regmov",
            aot.classify_basic("mov eax,edx", architecture="x86_64"),
        )
        self.assertEqual(
            "dispatch_jmp",
            aot.classify_basic("jmp r10", architecture="x86_64"),
        )
        self.assertEqual(
            "frame_load_unattributed",
            aot.classify_basic(
                "mov rax,QWORD PTR [rbp-0x20]", architecture="x86_64"
            ),
        )

    def test_aarch64_objdump_selection_and_parsing(self):
        output = "   1000:  8b010000  add x0, x0, x1\n"
        with mock.patch.object(aot, "_run_checked", return_value=output) as run:
            instructions = aot.disassemble_blob(
                b"\x00\x00\x01\x8b",
                0x1000,
                ROOT,
                "aarch64-test",
                architecture="aarch64",
            )
        argv = run.call_args.args[0]
        self.assertIn("aarch64", argv)
        self.assertNotIn("intel", argv)
        self.assertEqual("add x0, x0, x1", instructions[0].text)

    def test_direct_call_rel32_is_the_only_normalized_function_region(self):
        first = b"\x90\xe8\x01\x02\x03\x04\xc3"
        second = b"\x90\xe8\xaa\xbb\xcc\xdd\xc3"
        self.assertEqual(
            aot.normalized_code_sha256(first, [2]),
            aot.normalized_code_sha256(second, [2]),
        )
        with self.assertRaisesRegex(aot.AttributionError, "overlapping"):
            aot.normalized_code_sha256(first, [2, 3])


@unittest.skipUnless(WAMRC, "pass --wamrc for compiler/sidecar smoke coverage")
class FrameAttributionCompilerSmoke(unittest.TestCase):
    def test_tracked_core_wasm_sidecar_reconciles(self):
        work = ROOT / ".zig-cache" / f"frame-attribution-smoke-{os.getpid()}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(work, ignore_errors=True))

        fixture = ROOT / "tests/benchmarks/frame_attribution/frame_origins.wasm"
        cwasm = work / "frame_origins.cwasm"
        baseline_cwasm = work / "frame_origins.baseline.cwasm"
        prefix = work / "frame"
        baseline_result = subprocess.run(
            [str(WAMRC), "compile", str(fixture), "-o", str(baseline_cwasm)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            0,
            baseline_result.returncode,
            baseline_result.stdout + baseline_result.stderr,
        )
        env = os.environ.copy()
        env.update(
            {
                "WAMR_AOT_FRAME_ATTRIBUTION": str(prefix),
                "WAMR_AOT_FRAME_ATTRIBUTION_FUNC": "0",
                "WAMR_AOT_SPILL_METRIC": "1",
                "WAMR_AOT_SPILL_METRIC_FUNC": "0",
            }
        )
        compile_result = subprocess.run(
            [str(WAMRC), "compile", str(fixture), "-o", str(cwasm)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            0,
            compile_result.returncode,
            compile_result.stdout + compile_result.stderr,
        )
        self.assertEqual(
            baseline_cwasm.read_bytes(),
            cwasm.read_bytes(),
            "frame diagnostics must not change production AOT bytes",
        )

        sidecar = Path(f"{prefix}.mod0.func0.json")
        info = aot.parse_cwasm(cwasm)
        start, end = aot.function_bounds(info, 0)
        code = info.data[
            info.text_file_offset + start : info.text_file_offset + end
        ]
        metadata = aot.load_frame_metadata(
            sidecar,
            0,
            code,
            info.version,
            info.data[
                info.text_file_offset : info.text_file_offset + info.text_size
            ],
            start,
        )
        spill_line = re.search(
            r"\[aot-spill-metric\].*spill_ld=(\d+)\s+spill_st=(\d+)",
            compile_result.stderr,
        )
        self.assertIsNotNone(spill_line, compile_result.stderr)
        self.assertEqual(
            int(spill_line.group(1)),
            metadata.reconciliation["emitted_allocator_loads"],
        )
        self.assertEqual(
            int(spill_line.group(2)),
            metadata.reconciliation["emitted_allocator_stores"],
        )
        instructions = aot.disassemble_blob(code, 0, work, "smoke")
        aot.validate_metadata_disassembly(metadata, instructions)
        summary = aot.build_frame_summary(instructions, {}, metadata)
        self.assertTrue(metadata.reconciliation["matches"])
        self.assertGreater(
            metadata.reconciliation["emitted_allocator_loads"], 0
        )
        self.assertIn("allocator_spill", summary["origins"])
        self.assertIn("wasm_local_or_phi", summary["origins"])
        self.assertEqual(0, summary["coverage"]["unknown_frame_instructions"])

        wamr = WAMRC.with_name("wamr")
        if wamr.exists():
            run_result = subprocess.run(
                [str(wamr), "run", str(cwasm)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                0, run_result.returncode, run_result.stdout + run_result.stderr
            )


if __name__ == "__main__":
    unittest.main()
