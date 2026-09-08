#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aarch64_instruction_provenance as provenance


def instruction(offset: int, text: str) -> provenance.Instruction:
    mnemonic, operands = provenance.split_instruction_text(text)
    return provenance.Instruction(
        offset=offset,
        size=4,
        mnemonic=mnemonic,
        operands=operands,
        text=text,
        address=offset,
    )


def analyze(lines, alu_offsets, samples=None, total=100):
    instructions = [
        instruction(index * 4, text) for index, text in enumerate(lines)
    ]
    classes = [
        "alu" if item.offset in set(alu_offsets) else "other"
        for item in instructions
    ]
    return provenance.analyze_instruction_stream(
        instructions,
        broad_classes=classes,
        samples_by_offset=samples
        or {offset: 1 for offset in alu_offsets},
        total_run_samples=total,
    )


def category(result, name):
    return result["categories"][name]["samples"]


def common_category(result, name):
    return result["common_gating_universe"]["categories"][name]["samples"]


class AArch64InstructionProvenanceTests(unittest.TestCase):
    def test_w_and_x_aliases_and_extended_index_are_address_generation(self):
        result = analyze(
            [
                "add w9, w0, #1",
                "ldr w10, [x2, w9, uxtw #2]",
                "ret",
            ],
            [0],
            {0: 17},
        )
        self.assertEqual(17, category(result, "address_generation"))
        self.assertEqual(0, category(result, "unknown"))

    def test_dead_flag_output_does_not_poison_address_chain(self):
        result = analyze(
            [
                "add x9, x0, #1",
                "adds x10, x9, #4",
                "ldr w11, [x2, x10]",
                "ret",
            ],
            [0, 4],
            {0: 6, 4: 7},
        )
        self.assertEqual(13, category(result, "address_generation"))

    def test_address_value_stored_through_untyped_memory_is_unknown(self):
        result = analyze(
            [
                "add x9, x1, x2",
                "ldr w10, [x3, x9]",
                "str x9, [x4]",
                "ret",
            ],
            [0],
            {0: 11},
        )
        self.assertEqual(11, category(result, "unknown"))

    def test_address_value_also_used_for_control_is_mixed(self):
        result = analyze(
            [
                "add x9, x1, x2",
                "ldr w10, [x3, x9]",
                "cbnz x9, #0x10",
                "ret",
                "ret",
            ],
            [0],
            {0: 11},
        )
        self.assertEqual(11, category(result, "mixed"))

    def test_untyped_store_is_not_algorithmic_proof(self):
        result = analyze(
            [
                "add w9, w1, w2",
                "str w9, [x3]",
                "ret",
            ],
            [0],
            {0: 13},
        )
        self.assertEqual(13, category(result, "unknown"))

    def test_signatureless_return_is_an_unknown_escape(self):
        result = analyze(
            [
                "add x0, x1, #8",
                "ldr w9, [x2, x0]",
                "ret",
            ],
            [0],
            {0: 7},
        )
        self.assertEqual(7, category(result, "unknown"))

    def test_spilled_eventual_pointer_is_not_algorithmic_proof(self):
        result = analyze(
            [
                "add x9, x1, #8",
                "str x9, [sp, #16]",
                "ldr x10, [sp, #16]",
                "ldr w0, [x10]",
                "ret",
            ],
            [0],
            {0: 17},
        )
        self.assertEqual(17, category(result, "unknown"))

    def test_untyped_direct_return_is_not_algorithmic_proof(self):
        result = analyze(
            [
                "add x0, x1, #8",
                "ret",
            ],
            [0],
            {0: 13},
        )
        self.assertEqual(13, category(result, "unknown"))

    def test_complete_trap_cfg_proves_only_structural_address_guard(self):
        result = analyze(
            [
                "cmp x9, x10",
                "b.hs #0x10",
                "ldr w0, [x1, x9, lsl #2]",
                "ret",
                "brk #0",
            ],
            [0],
            {0: 19},
        )
        self.assertEqual(19, category(result, "structural_address_guard"))
        self.assertEqual(
            1, result["cfg"]["structural_address_guard_branches"]
        )
        gate = provenance.compare_engine_analyses(result, result)
        self.assertNotIn("structural_address_guard", gate["categories"])

    def test_call_to_trap_like_target_is_not_bounds_proof(self):
        result = analyze(
            [
                "cmp x9, x10",
                "b.hs #0x10",
                "ldr w0, [x1, x9]",
                "ret",
                "bl #0x1000",
                "ret",
            ],
            [0],
            {0: 5},
        )
        self.assertEqual(5, category(result, "unknown"))
        self.assertEqual(
            0, result["cfg"]["structural_address_guard_branches"]
        )

    def test_flag_clobber_disconnects_earlier_compare(self):
        result = analyze(
            [
                "cmp x9, x10",
                "adds x11, x11, #1",
                "b.eq #0x10",
                "ret",
                "brk #0",
            ],
            [0, 4],
            {0: 3, 4: 4},
        )
        self.assertEqual(7, category(result, "unknown"))

    def test_float_compare_clobbers_nzcv_before_branch(self):
        result = analyze(
            [
                "cmp x9, x10",
                "fcmp d0, d1",
                "b.hs #0x14",
                "ldr w0, [x1, x9]",
                "ret",
                "brk #0",
            ],
            [0],
            {0: 23},
        )
        self.assertEqual(23, category(result, "unknown"))
        self.assertEqual(
            0, result["cfg"]["structural_address_guard_branches"]
        )

    def test_conditional_compare_cannot_reuse_prior_flag_proof(self):
        result = analyze(
            [
                "cmp x9, x10",
                "ccmp x11, x12, #0, eq",
                "b.hs #0x14",
                "ldr w0, [x1, x9]",
                "ret",
                "brk #0",
            ],
            [0],
            {0: 21},
        )
        self.assertEqual(21, category(result, "unknown"))
        self.assertEqual(
            0, result["cfg"]["structural_address_guard_branches"]
        )

    def test_literal_load_replaces_compared_address_definition(self):
        result = analyze(
            [
                "cmp x9, x10",
                "b.hs #0x14",
                "ldr x9, 0x100",
                "ldr w0, [x1, x9]",
                "ret",
                "brk #0",
            ],
            [0],
            {0: 31},
        )
        self.assertEqual(31, category(result, "unknown"))
        self.assertEqual(
            0, result["cfg"]["structural_address_guard_branches"]
        )

    def test_opaque_effect_invalidates_registers_flags_and_cfg(self):
        result = analyze(
            [
                "cmp x9, x10",
                "mystery d0, d1",
                "b.hs #0x14",
                "ldr w0, [x1, x9]",
                "ret",
                "brk #0",
            ],
            [0],
            {0: 29},
        )
        self.assertEqual(29, category(result, "unknown"))
        self.assertEqual(
            0, result["cfg"]["structural_address_guard_branches"]
        )

    def test_control_flow_join_remains_unknown(self):
        result = analyze(
            [
                "cbz x0, #0xc",
                "add x9, x1, #4",
                "b #0x10",
                "sub x9, x1, #4",
                "ldr w2, [x3, x9]",
                "ret",
            ],
            [4, 12],
            {4: 6, 12: 8},
        )
        self.assertEqual(14, category(result, "unknown"))
        reasons = " ".join(
            item["reason"] for item in result["reason_counts"]
        )
        self.assertIn("multiple reaching definitions", reasons)

    def test_loop_phi_does_not_hang_or_become_address_proof(self):
        result = analyze(
            [
                "mov x9, xzr",
                "add x9, x9, #1",
                "ldr w10, [x2, x9]",
                "cmp x9, x3",
                "b.lo #0x4",
                "ret",
            ],
            [4, 12],
            {4: 9, 12: 2},
        )
        self.assertEqual(11, category(result, "unknown"))

    def test_call_clobber_breaks_caller_saved_path(self):
        result = analyze(
            [
                "add x9, x1, #4",
                "bl #0x1000",
                "ldr w0, [x2, x9]",
                "ret",
            ],
            [0],
            {0: 4},
        )
        self.assertEqual(4, category(result, "unknown"))

    def test_callee_saved_path_survives_call_without_argument_guess(self):
        result = analyze(
            [
                "add x19, x1, #4",
                "bl #0x1000",
                "ldr w0, [x2, x19]",
                "ret",
            ],
            [0],
            {0: 4},
        )
        self.assertEqual(4, category(result, "address_generation"))

    def test_post_index_writeback_store_escape_is_unknown(self):
        result = analyze(
            [
                "add x9, x1, #4",
                "ldr x0, [x9], #8",
                "str x9, [x2]",
                "ret",
            ],
            [0],
            {0: 10},
        )
        self.assertEqual(10, category(result, "unknown"))

    def test_unrecognized_alu_semantics_fail_closed(self):
        result = analyze(
            [
                "mystery x9, x1",
                "ldr w0, [x2, x9]",
                "ret",
            ],
            [0],
            {0: 12},
        )
        self.assertEqual(12, category(result, "unknown"))

    def test_partitions_reconcile_exactly_to_broad_alu(self):
        result = analyze(
            [
                "add x9, x1, #4",
                "ldr w10, [x2, x9]",
                "add w11, w10, #1",
                "str w11, [x3]",
                "ret",
            ],
            [0, 8],
            {0: 20, 8: 30},
            total=200,
        )
        self.assertEqual(50, result["broad_alu_samples"])
        self.assertEqual(50, result["partition_samples"])
        self.assertEqual(
            50,
            sum(
                result["categories"][name]["samples"]
                for name in provenance.CATEGORIES
            ),
        )
        common = result["common_gating_universe"]
        self.assertEqual(common["samples"], common["partition_samples"])

    def test_common_universe_covers_legacy_classifier_opcode_asymmetry(self):
        wamr_instructions = [
            instruction(0, "madd x9, x1, x2, x3"),
            instruction(4, "ldr w0, [x4, x9]"),
            instruction(8, "ret"),
        ]
        wasmtime_instructions = [
            instruction(0, "umaddl x9, w1, w2, x3"),
            instruction(4, "ldr w0, [x4, x9]"),
            instruction(8, "ret"),
        ]
        wamr = provenance.analyze_instruction_stream(
            wamr_instructions,
            broad_classes=["alu", "other", "other"],
            samples_by_offset={0: 6},
            total_run_samples=100,
        )
        wasmtime = provenance.analyze_instruction_stream(
            wasmtime_instructions,
            broad_classes=["other", "other", "other"],
            samples_by_offset={0: 6},
            total_run_samples=100,
        )
        self.assertEqual(6, wamr["broad_alu_samples"])
        self.assertEqual(0, wasmtime["broad_alu_samples"])
        self.assertEqual(6, common_category(wamr, "address_generation"))
        self.assertEqual(6, common_category(wasmtime, "address_generation"))
        gate = provenance.compare_engine_analyses(wamr, wasmtime)
        self.assertFalse(gate["optimization_authorized"])
        self.assertAlmostEqual(
            0.0,
            gate["categories"]["address_generation"][
                "conservative_headroom_percentage_points"
            ],
        )

    def test_opaque_other_opcode_is_common_universe_uncertainty(self):
        result = provenance.analyze_instruction_stream(
            [
                instruction(0, "mystery x9, x1"),
                instruction(4, "ret"),
            ],
            broad_classes=["other", "other"],
            samples_by_offset={0: 6},
            total_run_samples=100,
        )
        self.assertEqual(0, result["broad_alu_samples"])
        self.assertEqual(6, common_category(result, "unknown"))

    def test_gate_subtracts_wasmtime_unknown_and_mixed_upper_bound(self):
        wamr = analyze(
            ["add x9, x1, #4", "ldr w0, [x2, x9]", "ret"],
            [0],
            {0: 8},
        )
        wasmtime = analyze(
            [
                "add x9, x1, #4",
                "ldr w0, [x2, x9]",
                "mystery x10, x1",
                "ret",
            ],
            [0, 8],
            {0: 1, 8: 4},
        )
        gate = provenance.compare_engine_analyses(wamr, wasmtime)
        address = gate["categories"]["address_generation"]
        self.assertAlmostEqual(3.0, address["conservative_headroom_percentage_points"])
        self.assertFalse(address["clears_threshold"])
        self.assertFalse(gate["optimization_authorized"])

    def test_gate_can_clear_only_from_proven_dynamic_samples(self):
        wamr = analyze(
            ["add x9, x1, #4", "ldr w0, [x2, x9]", "ret"],
            [0],
            {0: 8},
        )
        wasmtime = analyze(
            ["add x9, x1, #4", "ldr w0, [x2, x9]", "ret"],
            [0],
            {0: 1},
        )
        gate = provenance.compare_engine_analyses(wamr, wasmtime)
        self.assertTrue(
            gate["categories"]["address_generation"]["clears_threshold"]
        )

    def test_unresolved_reference_samples_are_part_of_gate_upper_bound(self):
        wamr = analyze(
            ["add x9, x1, #4", "ldr w0, [x2, x9]", "ret"],
            [0],
            {0: 8},
        )
        wasmtime = analyze(
            ["add x9, x1, #4", "ldr w0, [x2, x9]", "ret"],
            [0],
            {0: 1},
        )
        wasmtime["sample_mapping"] = {
            "mapped_function_samples": 1,
            "unresolved_function_samples": 3,
        }
        gate = provenance.compare_engine_analyses(wamr, wasmtime)
        self.assertFalse(
            gate["categories"]["address_generation"]["clears_threshold"]
        )

    def test_global_unattributed_reference_samples_prevent_false_gate(self):
        instructions = [
            instruction(0, "add x9, x1, #4"),
            instruction(4, "ldr w0, [x2, x9]"),
            instruction(8, "ret"),
        ]
        wamr = provenance.analyze_instruction_stream(
            instructions,
            broad_classes=["alu", "other", "other"],
            samples_by_offset={0: 11},
            total_run_samples=200,
            global_attributed_samples=200,
        )
        wasmtime = provenance.analyze_instruction_stream(
            instructions,
            broad_classes=["alu", "other", "other"],
            samples_by_offset={},
            total_run_samples=200,
            global_attributed_samples=198,
        )
        gate = provenance.compare_engine_analyses(wamr, wasmtime)
        address = gate["categories"]["address_generation"]
        self.assertEqual(2, address["wasmtime_global_unattributed_samples"])
        self.assertAlmostEqual(
            4.5, address["conservative_headroom_percentage_points"]
        )
        self.assertFalse(address["clears_threshold"])

    def test_global_and_instruction_unresolved_samples_are_disjoint_upper_bounds(self):
        instructions = [
            instruction(0, "add x9, x1, #4"),
            instruction(4, "ldr w0, [x2, x9]"),
            instruction(8, "ret"),
        ]
        wamr = provenance.analyze_instruction_stream(
            instructions,
            broad_classes=["alu", "other", "other"],
            samples_by_offset={0: 11},
            total_run_samples=200,
        )
        wasmtime = provenance.analyze_instruction_stream(
            instructions,
            broad_classes=["alu", "other", "other"],
            samples_by_offset={},
            total_run_samples=200,
            global_attributed_samples=198,
        )
        wasmtime["sample_mapping"] = {
            "mapped_function_samples": 10,
            "unresolved_function_samples": 1,
        }
        gate = provenance.compare_engine_analyses(wamr, wasmtime)
        address = gate["categories"]["address_generation"]
        self.assertEqual(2, address["wasmtime_global_unattributed_samples"])
        self.assertEqual(1, address["wasmtime_instruction_unresolved_samples"])
        self.assertAlmostEqual(
            4.0, address["conservative_headroom_percentage_points"]
        )

    def test_unavailable_engine_blocks_gate(self):
        measured = analyze(
            ["add x9, x1, #4", "ldr w0, [x2, x9]", "ret"],
            [0],
        )
        gate = provenance.compare_engine_analyses(
            provenance.unavailable_analysis("exact cwasm missing"),
            measured,
        )
        self.assertEqual("blocked", gate["status"])
        self.assertFalse(gate["optimization_authorized"])

    def test_missing_or_malformed_disassembly_fails_closed(self):
        with self.assertRaisesRegex(provenance.ProvenanceError, "no instructions"):
            provenance.analyze_instruction_stream(
                [],
                broad_classes=[],
                samples_by_offset={},
                total_run_samples=1,
            )
        malformed = [instruction(0, "add x0, x1, #1")] * 2
        with self.assertRaisesRegex(provenance.ProvenanceError, "increasing"):
            provenance.analyze_instruction_stream(
                malformed,
                broad_classes=["alu", "alu"],
                samples_by_offset={},
                total_run_samples=1,
            )
        with self.assertRaisesRegex(
            provenance.ProvenanceError, "exact nonnegative instruction offsets"
        ):
            provenance.analyze_instruction_stream(
                [instruction(0, "add x0, x1, #1")],
                broad_classes=["alu"],
                samples_by_offset={2: 1},
                total_run_samples=1,
            )


if __name__ == "__main__":
    unittest.main()
