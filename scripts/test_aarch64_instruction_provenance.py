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

    def test_address_value_also_stored_is_mixed(self):
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
        self.assertEqual(11, category(result, "mixed"))

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

    def test_stored_arithmetic_is_algorithmic(self):
        result = analyze(
            [
                "add w9, w1, w2",
                "str w9, [x3]",
                "ret",
            ],
            [0],
            {0: 13},
        )
        self.assertEqual(13, category(result, "algorithmic_alu"))

    def test_return_value_prevents_address_only_claim(self):
        result = analyze(
            [
                "add x0, x1, #8",
                "ldr w9, [x2, x0]",
                "ret",
            ],
            [0],
            {0: 7},
        )
        self.assertEqual(7, category(result, "mixed"))

    def test_complete_trap_cfg_and_shared_address_definition_proves_bounds(self):
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
        self.assertEqual(19, category(result, "proven_bounds_check"))
        self.assertEqual(1, result["cfg"]["proven_bounds_branches"])

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
        self.assertEqual(0, result["cfg"]["proven_bounds_branches"])

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

    def test_post_index_writeback_data_use_is_mixed(self):
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
        self.assertEqual(10, category(result, "mixed"))

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
