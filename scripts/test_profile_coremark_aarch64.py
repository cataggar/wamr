#!/usr/bin/env python3

import importlib.util
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
            "authoritative_baseline_run": profile.AUTHORITATIVE_BASELINE_RUN,
            "guest_args": list(profile.bench_coremark.COREMARK_GUEST_ARGS),
            "expected_iterations": profile.bench_coremark.EXPECTED_ITERATIONS,
            "affinity": {"verified": True},
            "profile_schedule": schedule,
            "wasm": {"imported_function_count": 12},
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
        report["matched_functions"][0]["wasm_function_index"] = 14
        with self.assertRaisesRegex(profile.ProfileError, "inconsistent"):
            profile.validate_report(report)

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
