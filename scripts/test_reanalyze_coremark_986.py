import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aarch64_instruction_provenance as provenance
import reanalyze_coremark_986 as reanalysis


def synthetic_analysis(samples, total):
    lines = ["add x9, x0, #4", "ldr w1, [x2, x9]", "cmp w3, w4",
             "b.eq #0x18", "ret", "nop", "ret"]
    instructions = [
        provenance.Instruction(i * 4, 4, *provenance.split_instruction_text(text),
                               text, i * 4)
        for i, text in enumerate(lines)
    ]
    return provenance.analyze_instruction_stream(
        instructions, broad_classes=["alu", "linear_memory", "alu", "branch",
                                     "other", "other", "other"],
        samples_by_offset={0: samples[0], 8: samples[1]},
        total_run_samples=total,
    )


class RetainedEvidenceTests(unittest.TestCase):
    def test_rank_unknown_and_reject_mutated_sample_partitions(self):
        result = synthetic_analysis((7423, 5129), 68829)
        ranked = reanalysis.audit_rows(result, engine="wamr")
        self.assertEqual((8, 5129), (ranked[0]["offset"], ranked[0]["samples"]))
        for mutate in ("duplicate", "wrong_samples", "wrong_category"):
            with self.subTest(mutate=mutate):
                altered = copy.deepcopy(result)
                rows = altered["common_gating_universe"]["sampled_instructions"]
                if mutate == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                elif mutate == "wrong_samples":
                    rows[0]["samples"] += 1
                else:
                    rows[0]["category"] = "unknown"
                with self.assertRaises(ValueError):
                    reanalysis.audit_rows(altered, engine="wamr")

    def test_unknown_reference_budget_blocks_observed_address_gap(self):
        wamr = synthetic_analysis((7423, 5129), 68829)
        wasmtime = synthetic_analysis((254, 5740), 33113)
        wasmtime["global_sample_mapping"] = {
            "total_samples": 33113, "attributed_samples": 33075,
            "unattributed_samples": 38,
        }
        result = provenance.compare_engine_analyses(wamr, wasmtime)
        address = result["categories"]["address_generation"]
        self.assertGreater(address["observed_delta_percentage_points"], 5)
        self.assertLess(address["conservative_headroom_percentage_points"], 5)
        self.assertFalse(result["optimization_authorized"])


if __name__ == "__main__":
    unittest.main()
