import copy
import gzip
import json
import os
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

    @unittest.skipUnless(
        os.environ.get("COREMARK_986_ARTIFACT_DIR"),
        "requires downloaded run 34197146825 artifact",
    )
    def test_actual_retained_artifact_and_fail_closed_mutations(self):
        root = Path(os.environ["COREMARK_986_ARTIFACT_DIR"])
        report = json.loads((root / "profile.json").read_text())
        cwasm = gzip.decompress((root / "wamr-profiled.cwasm.gz").read_bytes())
        benchmark = (root / "benchmark-report.json").read_bytes()
        result = reanalysis.reanalyze(report, cwasm, benchmark)
        self.assertEqual(
            "c825824ac05eb7601e1e90fef5522d458ef26fec26871841f07d726190e7bb4c",
            result["cwasm_sha256"],
        )
        self.assertEqual(
            {"wamr": 68848, "wasmtime": 33119}, result["total_run_samples"]
        )
        gate = result["gate"]
        self.assertEqual("not-cleared", gate["status"])
        self.assertFalse(gate["optimization_authorized"])
        address = gate["categories"]["address_generation"]
        self.assertEqual((7423, 254), (address["wamr_samples"], address["wasmtime_samples"]))
        self.assertAlmostEqual(
            -7.431388244843346, address["conservative_headroom_percentage_points"]
        )
        self.assertEqual(
            (708, 5129), tuple(result["top_uncertain_paths"]["wamr"][0][k] for k in ("offset", "samples"))
        )
        self.assertEqual(
            (284, 1495), tuple(result["top_uncertain_paths"]["wasmtime"][0][k] for k in ("offset", "samples"))
        )

        with self.subTest("benchmark hash"), self.assertRaisesRegex(
            ValueError, "benchmark report hash"
        ):
            reanalysis.reanalyze(report, cwasm, benchmark + b" ")
        with self.subTest("binary hash"), self.assertRaisesRegex(
            ValueError, "binary hash"
        ):
            reanalysis.reanalyze(report, cwasm[:-1] + bytes([cwasm[-1] ^ 1]), benchmark)
        changed = copy.deepcopy(report)
        function = next(x for x in changed["matched_functions"] if x["local_func"] == 3)
        function["alu_provenance"]["wasmtime"]["common_gating_universe"]["sampled_instructions"][0]["samples"] += 1
        with self.subTest("sample partition"), self.assertRaisesRegex(
            ValueError, "sampled rows do not reconcile"
        ):
            reanalysis.reanalyze(changed, cwasm, benchmark)
        changed = copy.deepcopy(report)
        changed["alu_provenance"]["gate"]["optimization_authorized"] = True
        with self.subTest("gate tampering"), self.assertRaisesRegex(
            (ValueError, reanalysis.profile.ProfileError), "gate"
        ):
            reanalysis.reanalyze(changed, cwasm, benchmark)


if __name__ == "__main__":
    unittest.main()
