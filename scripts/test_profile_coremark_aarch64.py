#!/usr/bin/env python3

import importlib.util
import copy
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/profile_coremark_aarch64.py"
SPEC = importlib.util.spec_from_file_location("profile_coremark_aarch64", SCRIPT)
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


class CoreMarkProfileTests(unittest.TestCase):
    def test_validated_coremark_run_counts_fail_closed(self):
        output = (
            "2K performance run parameters for coremark.\n"
            "Iterations/Sec : 12000.0\n"
            "Iterations : 400000\n"
            "Correct operation validated.\n"
        )
        self.assertEqual(
            [12000.0],
            profile.parse_validated_coremark(output, "engine", 1),
        )
        with self.assertRaisesRegex(profile.ProfileError, "must be one"):
            profile.parse_validated_coremark(output, "engine", 2)
        with self.assertRaisesRegex(profile.ProfileError, "CRC-validated"):
            profile.parse_validated_coremark(output + "ERROR! bad crc\n", "engine", 1)

    def test_spill_metrics_are_keyed_by_local_function(self):
        text = (
            "[aot-spill-metric] local_func=3 mod=0 name=core_bench_list "
            "insts=10 clobbers=2 slots=4 spilled_vregs=3 scalar=3 v128=0 "
            "slots_scalar=4 slots_v128=0 spill_ld=7 spill_st=2 remat=1 "
            "callee_saved=4\n"
        )
        metrics = profile.parse_spill_metrics(text)
        self.assertEqual(7, metrics[3]["spill_ld"])
        self.assertEqual("core_bench_list", metrics[3]["name"])
        with self.assertRaisesRegex(profile.ProfileError, "duplicate"):
            profile.parse_spill_metrics(text + text)

    def test_wasmtime_function_index_and_offsets_are_parsed(self):
        text = """\
ffff0000 wasm[0]::function[15]::core_bench_list+0x4 (jitted-1.so)
ffff0004 wasm[0]::function[15]::core_bench_list +0x8 (jitted-1.so)
aaaa0000 wasmtime::runtime+0x10 (/bin/wasmtime)
"""
        parsed = profile.parse_wasmtime_samples(text)
        self.assertEqual(3, parsed["total_samples"])
        self.assertEqual(2, parsed["functions"][15]["samples"])
        self.assertEqual(Counter({4: 1, 8: 1}), parsed["functions"][15]["offsets"])

        identity = profile.compare_hot_function.WasmModuleIdentity(
            sha256="a" * 64,
            imported_function_count=12,
            local_function_count=73,
            function_names={15: "core_bench_list"},
        )
        profile.validate_wasmtime_mapping(parsed, identity)
        bad = profile.parse_wasmtime_samples(
            "ffff wasm[0]::function[15]::wrong+0x4 (jitted.so)\n"
        )
        with self.assertRaisesRegex(profile.ProfileError, "name mismatch"):
            profile.validate_wasmtime_mapping(bad, identity)

        plain = profile.parse_wasmtime_samples(
            "ffff core_bench_list+0x14 (/work/jitted-99-3.so)\n",
            identity,
        )
        self.assertEqual(1, plain["functions"][15]["samples"])
        self.assertEqual(Counter({0x14: 1}), plain["functions"][15]["offsets"])
        with self.assertRaisesRegex(profile.ProfileError, "implies local function"):
            profile.parse_wasmtime_samples(
                "ffff core_bench_list+0x14 (/work/jitted-99-4.so)\n",
                identity,
            )

    def test_ambiguous_wasmtime_sample_mapping_fails(self):
        with self.assertRaisesRegex(profile.ProfileError, "ambiguous"):
            profile.parse_wasmtime_samples(
                "ffff wasm[0]::function[15]+0x4 "
                "wasm[0]::function[16]+0x8\n"
            )

    def test_wasmtime_hot_instruction_mapping_and_classes(self):
        aot = profile.load_aot_helper(ROOT)
        instructions = [
            profile.compare_hot_function.Instruction(
                offset=0,
                size=4,
                mnemonic="mov",
                operands="w0, w1",
                text="mov w0, w1",
                raw_bytes=b"\0" * 4,
            ),
            profile.compare_hot_function.Instruction(
                offset=4,
                size=4,
                mnemonic="add",
                operands="w0, w0, #1",
                text="add w0, w0, #1",
                raw_bytes=b"\0" * 4,
            ),
        ]
        with mock.patch.object(
            profile.compare_hot_function,
            "parse_disassembly",
            return_value=instructions,
        ):
            result = profile.classify_wasmtime_function(
                aot=aot,
                objdump_text="ignored",
                wasm_index=15,
                offsets=Counter({0: 6, 4: 4}),
                total_samples=20,
            )
        self.assertEqual(10, result["mapped_instruction_samples"])
        self.assertEqual(6, result["classes"]["regmov"]["samples"])
        self.assertEqual(4, result["classes"]["alu"]["samples"])

    def test_wasmtime_narrow_alu_uses_real_parser_and_instruction_mapping(self):
        objdump = """\
00000100 wasm[0]::function[15]::core_bench_list:
       100: 29 04 00 91                  add     x9, x1, #1
       104: 40 68 69 b8                  ldr     w0, [x2, x9]
       108: c0 03 5f d6                  ret
"""
        result = profile.analyze_wasmtime_alu_provenance(
            aot=profile.load_aot_helper(ROOT),
            objdump_text=objdump,
            wasm_index=15,
            offsets=Counter({0: 9, 4: 3, 8: 1}),
            total_samples=20,
        )
        self.assertEqual(9, result["broad_alu_samples"])
        self.assertEqual(
            9, result["categories"]["address_generation"]["samples"]
        )
        self.assertEqual(13, result["sample_mapping"]["mapped_function_samples"])
        self.assertEqual(0, result["sample_mapping"]["unresolved_function_samples"])

    def test_narrow_report_assembly_reconciles_existing_all_alu(self):
        instructions = [
            profile.aarch64_instruction_provenance.Instruction(
                offset=0,
                size=4,
                mnemonic="add",
                operands="x9, x1, #4",
                text="add x9, x1, #4",
                address=0,
            ),
            profile.aarch64_instruction_provenance.Instruction(
                offset=4,
                size=4,
                mnemonic="ldr",
                operands="w0, [x2, x9]",
                text="ldr w0, [x2, x9]",
                address=4,
            ),
            profile.aarch64_instruction_provenance.Instruction(
                offset=8,
                size=4,
                mnemonic="ret",
                operands="",
                text="ret",
                address=8,
            ),
        ]
        wamr = profile.aarch64_instruction_provenance.analyze_instruction_stream(
            instructions,
            broad_classes=["alu", "other", "other"],
            samples_by_offset={0: 8},
            total_run_samples=100,
        )
        wasmtime = (
            profile.aarch64_instruction_provenance.analyze_instruction_stream(
                instructions,
                broad_classes=["alu", "other", "other"],
                samples_by_offset={0: 1},
                total_run_samples=100,
            )
        )
        assembled = profile.assemble_alu_provenance(
            wamr=wamr,
            wasmtime=wasmtime,
            expected_wamr_all_alu_samples=8,
            expected_wasmtime_all_alu_samples=1,
        )
        self.assertTrue(assembled["gate"]["optimization_authorized"])
        with self.assertRaisesRegex(profile.ProfileError, "existing all_alu"):
            profile.assemble_alu_provenance(
                wamr=wamr,
                wasmtime=wasmtime,
                expected_wamr_all_alu_samples=7,
                expected_wasmtime_all_alu_samples=1,
            )

    def test_report_schema_requires_consistent_index_mapping(self):
        schedule = [
            {
                "engine": engine,
                "phase": phase,
            }
            for phase in ("warmup", "profile")
            for engine in ("wamr", "wasmtime", "wasmtime", "wamr")
        ]
        report = {
            "schema_version": profile.REPORT_SCHEMA_VERSION,
            "kind": profile.REPORT_KIND,
            "architecture": "aarch64",
            "benchmark": {
                "report_id": "12345678-1234-5678-1234-567812345678",
                "generated_at": "2026-09-08T00:00:00+00:00",
                "report_sha256": "f" * 64,
                "artifact_handoff": {
                    "directory": "/artifacts",
                    "manifest_sha256": "e" * 64,
                },
                "execution": {"provider": "local", "run_id": "test-run"},
                "producer": {"source_sha": "c" * 40},
                "target": {
                    "identity": {
                        "source": {"sha": "b" * 40},
                        "runtime": {"sha256": "1" * 64},
                        "compiler": {"sha256": "1" * 64},
                        "module": {"sha256": "1" * 64},
                    }
                },
                "wasmtime_baseline": {
                    "identity": {
                        "version": profile.bench_coremark.PINNED_WASMTIME_VERSION,
                        "runtime": {"sha256": "3" * 64},
                    }
                },
            },
            "provenance": {
                "producer_source_sha": "c" * 40,
                "script_sha256": "d" * 64,
                "execution": {"provider": "local", "run_id": "test-run"},
            },
            "guest_args": list(profile.bench_coremark.COREMARK_GUEST_ARGS),
            "expected_iterations": profile.bench_coremark.EXPECTED_ITERATIONS,
            "classifier_wording": {"all_alu": profile.ALL_ALU_WORDING},
            "affinity": {"verified": True},
            "profile_schedule": schedule,
            "minimum_attribution_coverage_pct": (
                profile.MIN_ATTRIBUTION_COVERAGE_PCT
            ),
            "wamr_captures": [
                {
                    "coverage_pct": 99.96,
                    "mapping": {"authoritative": True},
                },
                {
                    "coverage_pct": 99.95,
                    "mapping": {"authoritative": True},
                },
            ],
            "wasmtime_captures": [
                {"coverage_pct": 99.9},
                {"coverage_pct": 99.9},
            ],
            "wasm": {"imported_function_count": 12},
            "wamr": {
                "commit": "b" * 40,
                "runtime_sha256": "1" * 64,
                "compiler_sha256": "1" * 64,
                "cwasm_sha256": "1" * 64,
            },
            "wasmtime": {
                "version": profile.bench_coremark.PINNED_WASMTIME_VERSION,
                "sha256": "3" * 64,
            },
            "engines": {
                "wamr": {"total_samples": 100, "attributed_samples": 99},
                "wasmtime": {"total_samples": 100, "attributed_samples": 98},
            },
            "matched_functions": [
                {
                    "local_func": 3,
                    "wasm_function_index": 15,
                    "wamr": {"samples": 30},
                    "wasmtime": {"samples": 28},
                }
            ],
        }
        profile.validate_report(report)
        narrow_instructions = [
            profile.aarch64_instruction_provenance.Instruction(
                offset=0,
                size=4,
                mnemonic="add",
                operands="x9, x1, #4",
                text="add x9, x1, #4",
                address=0,
            ),
            profile.aarch64_instruction_provenance.Instruction(
                offset=4,
                size=4,
                mnemonic="ldr",
                operands="w0, [x2, x9]",
                text="ldr w0, [x2, x9]",
                address=4,
            ),
            profile.aarch64_instruction_provenance.Instruction(
                offset=8,
                size=4,
                mnemonic="ret",
                operands="",
                text="ret",
                address=8,
            ),
        ]
        narrow_wamr = (
            profile.aarch64_instruction_provenance.analyze_instruction_stream(
                narrow_instructions,
                broad_classes=["alu", "other", "other"],
                samples_by_offset={0: 1},
                total_run_samples=100,
            )
        )
        narrow_wasmtime = copy.deepcopy(narrow_wamr)
        report["classifier_wording"]["narrow_alu_provenance"] = (
            profile.NARROW_ALU_WORDING
        )
        report["provenance"]["analysis_module"] = {
            "path": "scripts/aarch64_instruction_provenance.py",
            "sha256": "a" * 64,
        }
        report["matched_functions"][0]["class_groups"] = {
            "all_alu": {"wamr_samples": 1, "wasmtime_samples": 1}
        }
        report["matched_functions"][0]["alu_provenance"] = (
            profile.assemble_alu_provenance(
                wamr=narrow_wamr,
                wasmtime=narrow_wasmtime,
                expected_wamr_all_alu_samples=1,
                expected_wasmtime_all_alu_samples=1,
            )
        )
        report["alu_provenance"] = {
            "schema_version": (
                profile.aarch64_instruction_provenance.SCHEMA_VERSION
            ),
            "kind": profile.aarch64_instruction_provenance.ANALYSIS_KIND,
            "gate": report["matched_functions"][0]["alu_provenance"]["gate"],
        }
        report["retained_analysis_artifacts"] = {
            "wamr_cwasm": {
                "retained": True,
                "source_sha256": "1" * 64,
            }
        }
        profile.validate_report(report)
        stale_execution = copy.deepcopy(report)
        stale_execution["provenance"]["execution"]["run_id"] = "stale-run"
        with self.assertRaisesRegex(profile.ProfileError, "execution provenance"):
            profile.validate_report(stale_execution)
        report["matched_functions"][0]["wasm_function_index"] = 14
        with self.assertRaisesRegex(profile.ProfileError, "inconsistent"):
            profile.validate_report(report)

    def test_historical_profile_is_readable_but_not_current_authority(self):
        status = profile.profile_report_status(
            {
                "schema_version": profile.HISTORICAL_REPORT_SCHEMA_VERSION,
                "kind": profile.REPORT_KIND,
                "authoritative_baseline_run": profile.HISTORICAL_BASELINE_RUN,
            }
        )
        self.assertEqual("historical-unverified", status["status"])
        self.assertFalse(status["authoritative"])
        self.assertTrue(status["known_historical_baseline"])

    def test_each_wamr_capture_must_pass_exact_mapping_and_coverage(self):
        base = {
            "total_samples": 34410,
            "attributed_samples": 34398,
            "mapping": {
                "authoritative": True,
                "override": None,
                "size": 122880,
                "expected_size": 122880,
            },
        }
        first = profile.validate_wamr_capture(
            base, minimum_samples=1000
        )
        self.assertGreater(first["coverage_pct"], 99.9)
        second = profile.validate_wamr_capture(
            {
                **base,
                "total_samples": 34419,
                "attributed_samples": 34404,
            },
            minimum_samples=1000,
        )
        self.assertGreater(second["coverage_pct"], 99.9)

        with self.assertRaisesRegex(profile.ProfileError, "below"):
            profile.validate_wamr_capture(
                {
                    **base,
                    "total_samples": 1000,
                    "attributed_samples": 980,
                },
                minimum_samples=1000,
            )
        with self.assertRaisesRegex(profile.ProfileError, "requires at least"):
            profile.validate_wamr_capture(
                {
                    **base,
                    "total_samples": 999,
                    "attributed_samples": 998,
                },
                minimum_samples=1000,
            )
        with self.assertRaisesRegex(profile.ProfileError, "exact-size"):
            profile.validate_wamr_capture(
                {
                    **base,
                    "mapping": {
                        **base["mapping"],
                        "authoritative": False,
                        "override": "0x1000",
                    },
                },
                minimum_samples=1000,
            )

        with self.assertRaisesRegex(profile.ProfileError, "below"):
            profile.validate_wasmtime_capture(
                {
                    "total_samples": 1000,
                    "functions": {15: {"samples": 980}},
                },
                minimum_samples=1000,
            )

    def test_classifier_names_all_alu_without_address_claim(self):
        self.assertIn("all_alu", profile.CLASS_GROUPS)
        self.assertNotIn("alu", profile.CLASS_GROUPS)
        self.assertIn("address-generation", profile.ALL_ALU_WORDING)
        delta = 13627 / 68829 * 100 - 3880 / 33113 * 100
        self.assertAlmostEqual(8.0808884555, delta, places=10)

    def test_profile_aggregates_preserve_balanced_samples(self):
        rankings = [
            {
                "text_size": 100,
                "function_count": 2,
                "total_samples": 100,
                "attributed_samples": 99,
                "top_functions": [
                    {
                        "local_func": 0,
                        "samples": 60,
                        "percent_of_run": 60.0,
                        "code_bytes": 40,
                    }
                ],
            },
            {
                "text_size": 100,
                "function_count": 2,
                "total_samples": 120,
                "attributed_samples": 118,
                "top_functions": [
                    {
                        "local_func": 0,
                        "samples": 70,
                        "percent_of_run": 58.3,
                        "code_bytes": 40,
                    }
                ],
            },
        ]
        merged = profile.aggregate_wamr_rankings(rankings)
        self.assertEqual(220, merged["total_samples"])
        self.assertEqual(130, merged["top_functions"][0]["samples"])

        captures = [
            {
                "total_samples": 10,
                "functions": {
                    15: {
                        "samples": 8,
                        "names": {"core_bench_list"},
                        "offsets": Counter({4: 8}),
                        "mapping_methods": {"jitdump"},
                    }
                },
            },
            {
                "total_samples": 12,
                "functions": {
                    15: {
                        "samples": 9,
                        "names": {"core_bench_list"},
                        "offsets": Counter({4: 9}),
                        "mapping_methods": {"jitdump"},
                    }
                },
            },
        ]
        merged_wasmtime = profile.aggregate_wasmtime_samples(captures)
        self.assertEqual(22, merged_wasmtime["total_samples"])
        self.assertEqual(17, merged_wasmtime["functions"][15]["samples"])

    def test_current_aot_version_is_shared(self):
        aot = profile.load_aot_helper(ROOT)
        self.assertEqual(aot.AOT_VERSION, profile.compare_hot_function.AOT_VERSION)


if __name__ == "__main__":
    unittest.main()
