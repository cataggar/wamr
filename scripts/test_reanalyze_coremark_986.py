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
        wamr_clusters = result["unknown_reason_clusters"]["wamr"]
        self.assertEqual(13027, wamr_clusters["uncertain_samples"])
        self.assertEqual(
            8769, wamr_clusters["clusters"]["non_trap_control"]["samples"]
        )
        self.assertEqual(
            2879, wamr_clusters["clusters"]["ambiguous_cfg_join"]["samples"]
        )
        self.assertEqual(
            1303, wamr_clusters["clusters"]["mixed_categories"]["samples"]
        )
        wt_clusters = result["unknown_reason_clusters"]["wasmtime"]
        self.assertEqual(5740, wt_clusters["uncertain_samples"])
        self.assertEqual(
            3411, wt_clusters["clusters"]["non_trap_control"]["samples"]
        )
        self.assertEqual(
            2025, wt_clusters["clusters"]["ambiguous_cfg_join"]["samples"]
        )
        self.assertEqual(
            216, wt_clusters["clusters"]["untyped_escape"]["samples"]
        )
        self.assertEqual(
            76, wt_clusters["clusters"]["opaque_or_call_boundary"]["samples"]
        )
        self.assertEqual(
            12, wt_clusters["clusters"]["combined_memory_writeback"]["samples"]
        )

        objdump_text = (root / "wasmtime-objdump.txt").read_text()
        with_exhaustion = reanalysis.reanalyze(
            report, cwasm, benchmark, wasmtime_objdump_text=objdump_text
        )
        native_ex = with_exhaustion["wasmtime_native_exhaustion"]
        self.assertTrue(native_ex["verified_retained_provenance"])
        self.assertEqual(0x600, native_ex["address_base"])
        self.assertEqual(410, native_ex["static_instructions"])
        self.assertEqual(208, native_ex["common_static_instructions"])
        sinks = native_ex["sink_classes"]
        self.assertEqual(256, sinks["pure_address"]["samples"])
        self.assertEqual(0, sinks["pure_structural_guard"]["samples"])
        self.assertEqual(0, sinks["pure_algorithmic"]["samples"])
        self.assertEqual(3555, sinks["non_trap_control"]["samples"])
        self.assertEqual(1735, sinks["untyped_escape"]["samples"])
        self.assertEqual(
            336, sinks["mixed_address_and_control_or_escape"]["samples"]
        )
        self.assertEqual(100, sinks["opaque_or_call_boundary"]["samples"])
        self.assertEqual(12, sinks["combined_memory_writeback"]["samples"])
        self.assertEqual(0, sinks["dead_or_unconsumed"]["samples"])
        reclass_addr = native_ex["reclassified_from_unknown"]["pure_address"]
        self.assertEqual(2, reclass_addr["samples"])
        self.assertEqual(
            [540, 1224],
            [row["offset"] for row in reclass_addr["sampled_instructions"]],
        )
        ex_gate = native_ex["exhaustion_gate"]
        self.assertEqual("exhausted", ex_gate["status"])
        self.assertTrue(ex_gate["proven_exhausted"])
        self.assertFalse(ex_gate["optimization_authorized"])
        ex_addr = ex_gate["categories"]["address_generation"]
        self.assertEqual((254, 256), (ex_addr["strict_wasmtime_samples"], ex_addr["unblocked_wasmtime_samples"]))
        self.assertEqual(5738, ex_addr["wasmtime_remaining_uncertain_common_samples"])
        self.assertEqual(5776, ex_addr["wasmtime_remaining_upper_bound_samples"])
        self.assertAlmostEqual(
            -7.431388244843346,
            ex_addr["unblocked_conservative_headroom_percentage_points"],
        )
        self.assertAlmostEqual(
            -5.538813297132446,
            ex_addr["wamr_mixed_credit_upper_bound_headroom_percentage_points"],
        )
        with self.subTest("objdump tampering"), self.assertRaisesRegex(
            ValueError, "differs from native disassembly"
        ):
            reanalysis.reanalyze(
                report,
                cwasm,
                benchmark,
                wasmtime_objdump_text=objdump_text.replace(
                    "x3, x19", "x3, x20"
                ),
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
