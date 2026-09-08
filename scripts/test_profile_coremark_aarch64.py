#!/usr/bin/env python3

import importlib.util
import copy
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/profile_coremark_aarch64.py"
SPEC = importlib.util.spec_from_file_location("profile_coremark_aarch64", SCRIPT)
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


class CoreMarkProfileTests(unittest.TestCase):
    def test_paired_acceptance_binds_roles_and_reduces_frame_traffic(self):
        execution = {"provider": "local", "run_id": "paired"}
        engines = {
            role: {
                "role": role,
                "identity": {"source": {"sha": digest * 40}},
            }
            for role, digest in (
                ("wamr-baseline", "a"),
                ("wamr-target", "b"),
            )
        }
        benchmark = {
            "provenance": {
                "report_id": "12345678-1234-5678-1234-567812345678",
                "execution": execution,
            },
            "engines": list(engines.values()),
            "wamr_comparison": {"median_delta_pct": 2.5},
            "simd_acceptance": {
                "correctness_passed": True,
                "performance_passed": True,
                "max_aot_regression_pct": 2.0,
            },
        }

        def profile_report(role, frame_samples):
            return {
                "benchmark": {
                    "selected_role": role,
                    "selected_wamr": engines[role],
                    "report_sha256": "f" * 64,
                    "report_id": benchmark["provenance"]["report_id"],
                    "execution": execution,
                },
                "provenance": {"execution": execution},
                "host": {
                    "architecture": "aarch64",
                    "cpu_count": 4,
                    "cpu_model": "Neoverse",
                    "runner_name": "runner",
                    "fingerprint": "host",
                },
                "wamr": {"commit": engines[role]["identity"]["source"]["sha"]},
                "frame_attribution": {
                    "local_func": 10,
                    "total_samples": 1000,
                    "summary": {
                        "coverage": {"frame_samples": frame_samples},
                        "origins": {
                            "allocator_spill": {
                                "samples": frame_samples,
                                "percent_of_run": frame_samples / 10,
                            }
                        },
                        "reconciliation": {
                            "emitted_allocator_loads": (
                                42 if role == "wamr-baseline" else 38
                            ),
                            "emitted_allocator_stores": 24,
                        },
                    },
                },
            }

        baseline = profile_report("wamr-baseline", 130)
        target = profile_report("wamr-target", 110)
        correctness = {
            "schema_version": 1,
            "kind": "coremark-aarch64-candidate-correctness",
            "status": "passed",
            "target_sha": "b" * 40,
            "command": [
                "zig",
                "build",
                "-j4",
                "test",
                "-Doptimize=ReleaseFast",
                "--summary",
                "all",
            ],
        }
        with (
            mock.patch.object(
                profile.bench_coremark,
                "validate_authoritative_benchmark_report",
            ),
            mock.patch.object(profile, "validate_report"),
            mock.patch.object(profile.bench_coremark, "validate_simd_acceptance"),
        ):
            acceptance = profile.build_paired_acceptance(
                benchmark_report=benchmark,
                benchmark_report_sha="f" * 64,
                baseline_profile=baseline,
                baseline_profile_sha="1" * 64,
                target_profile=target,
                target_profile_sha="2" * 64,
                correctness_report=correctness,
                correctness_report_sha="3" * 64,
            )
            self.assertEqual("passed", acceptance["status"])
            frame_gate = acceptance["gates"][
                "core_state_transition_frame_traffic"
            ]
            self.assertEqual(13.0, frame_gate["baseline"]["percent_of_run"])
            self.assertEqual(11.0, frame_gate["target"]["percent_of_run"])
            self.assertEqual(38, frame_gate["target"]["emitted_allocator_loads"])

            wrong_role = copy.deepcopy(target)
            wrong_role["benchmark"]["selected_role"] = "wamr-baseline"
            with self.assertRaisesRegex(profile.ProfileError, "does not match"):
                profile.build_paired_acceptance(
                    benchmark_report=benchmark,
                    benchmark_report_sha="f" * 64,
                    baseline_profile=baseline,
                    baseline_profile_sha="1" * 64,
                    target_profile=wrong_role,
                    target_profile_sha="2" * 64,
                    correctness_report=correctness,
                    correctness_report_sha="3" * 64,
                )

            regressed = profile_report("wamr-target", 140)
            failed = profile.build_paired_acceptance(
                benchmark_report=benchmark,
                benchmark_report_sha="f" * 64,
                baseline_profile=baseline,
                baseline_profile_sha="1" * 64,
                target_profile=regressed,
                target_profile_sha="2" * 64,
                correctness_report=correctness,
                correctness_report_sha="3" * 64,
            )
            self.assertEqual("failed", failed["status"])
            self.assertFalse(
                failed["gates"]["core_state_transition_frame_traffic"]["passed"]
            )
            stale_correctness = {**correctness, "target_sha": "c" * 40}
            with self.assertRaisesRegex(profile.ProfileError, "correctness"):
                profile.build_paired_acceptance(
                    benchmark_report=benchmark,
                    benchmark_report_sha="f" * 64,
                    baseline_profile=baseline,
                    baseline_profile_sha="1" * 64,
                    target_profile=target,
                    target_profile_sha="2" * 64,
                    correctness_report=stale_correctness,
                    correctness_report_sha="3" * 64,
                )

    def test_frame_diagnostics_use_exact_compiler_and_artifact(self):
        cache = ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        for mode in ("enabled", "disabled", "unsupported", "changed-code", "stale"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory(dir=cache) as temp,
            ):
                root = Path(temp)
                cwasm = root / "benchmark.cwasm"
                cwasm.write_bytes(b"exact benchmark code")
                compiler = root / "retained-wamrc"
                fixture = root / "fixture.wasm"
                helper = root / "aot_jit_attr.py"
                metadata = root / "wamr-frame.mod0.func10.json"
                if mode == "stale":
                    metadata.write_text("old metadata")
                calls = []

                def run(command, log_name, **kwargs):
                    calls.append((command, kwargs))
                    if command[0] == str(compiler):
                        self.assertEqual(root, kwargs["cwd"])
                        self.assertEqual(str(fixture), command[2])
                        env = kwargs["env"]
                        if mode == "disabled":
                            self.assertNotIn("WAMR_AOT_FRAME_ATTRIBUTION", env)
                            self.assertNotIn("WAMR_AOT_FRAME_ATTRIBUTION_FUNC", env)
                        else:
                            self.assertEqual(
                                str(root / "wamr-frame"),
                                env["WAMR_AOT_FRAME_ATTRIBUTION"],
                            )
                            self.assertEqual("0", env["WAMR_AOT_FRAME_ATTRIBUTION_MODULE"])
                            self.assertEqual("10", env["WAMR_AOT_FRAME_ATTRIBUTION_FUNC"])
                        Path(command[-1]).write_bytes(
                            b"changed code" if mode == "changed-code" else cwasm.read_bytes()
                        )
                        if mode not in ("disabled", "unsupported"):
                            metadata.write_text("{}")
                    else:
                        self.assertEqual(str(helper), command[1])
                        self.assertIn("--validate-frame-metadata", command)
                        self.assertEqual(
                            str(cwasm), command[command.index("--cwasm") + 1]
                        )
                        self.assertEqual(
                            str(metadata), command[command.index("--frame-metadata") + 1]
                        )
                    return SimpleNamespace(stdout="", stderr="")

                def compile_diagnostics():
                    return profile.compile_wamr_diagnostics(
                        recorder=SimpleNamespace(run=run),
                        wamrc=compiler,
                        fixture=fixture,
                        cwasm=cwasm,
                        out_dir=root,
                        build_repo=root,
                        helper=helper,
                        frame_func=None if mode == "disabled" else 10,
                    )

                with mock.patch.dict(
                    profile.os.environ,
                    {
                        "WAMR_AOT_FRAME_ATTRIBUTION": "/unrecorded",
                        "WAMR_AOT_FRAME_ATTRIBUTION_MODULE": "9",
                        "WAMR_AOT_FRAME_ATTRIBUTION_FUNC": "99",
                    },
                ):
                    if mode == "unsupported":
                        with self.assertRaisesRegex(profile.ProfileError, "benchmark a candidate"):
                            compile_diagnostics()
                        self.assertEqual(1, len(calls))
                    elif mode == "changed-code":
                        with self.assertRaisesRegex(profile.ProfileError, "exact benchmark"):
                            compile_diagnostics()
                        self.assertEqual(1, len(calls))
                    elif mode == "stale":
                        with self.assertRaisesRegex(profile.ProfileError, "existing frame metadata"):
                            compile_diagnostics()
                        self.assertEqual([], calls)
                        self.assertEqual("old metadata", metadata.read_text())
                    else:
                        metrics, path = compile_diagnostics()
                        self.assertEqual({}, metrics)
                        self.assertEqual(metadata if mode == "enabled" else None, path)
                        self.assertEqual(2 if mode == "enabled" else 1, len(calls))
                        self.assertFalse((root / "coremark.diagnostic.cwasm").exists())

    def test_frame_capture_merge_preserves_pairs_and_static_counts(self):
        summary = {
            "coverage": {
                "frame_instructions": 2, "attributed_frame_instructions": 2,
                "proven_origin_frame_instructions": 2, "unknown_frame_instructions": 0,
                "frame_samples": 36, "attributed_frame_samples": 36,
                "proven_origin_frame_samples": 36, "unknown_frame_samples": 0,
            },
            "origins": {
                "allocator_spill": {"samples": 36, "static_instructions": 2},
            },
            "allocator_contributors": [
                {"slot": 0, "vreg": 1, "samples": 16, "static_loads": 1, "static_stores": 0},
                {"slot": 1, "vreg": 2, "samples": 0, "static_loads": 1, "static_stores": 0},
                {"slot": 2, "vreg": 3, "samples": 0, "static_loads": 1, "static_stores": 0},
                {
                    "slot": 1, "vreg": None, "samples": 20,
                    "static_loads": 0, "static_stores": 0,
                    "paired_components": [{"slot": 1}, {"slot": 2}],
                },
            ],
            "reconciliation": {
                "emitted_allocator_loads": 3, "emitted_allocator_stores": 0,
                "spill_metric_loads": 3, "spill_metric_stores": 0, "matches": True,
            },
            "allocator_component_counts": {
                "total_loads": 3, "total_stores": 0,
                "unranked_loads": 0, "unranked_stores": 0,
            },
            "unknown_instructions": [],
        }
        metadata = SimpleNamespace(
            inline_data_ranges=[],
            raw={
                "module": 0, "schema": "wamr-aot-frame-attribution",
                "schema_version": 2, "architecture": "aarch64", "abi": "aapcs64",
                "compiler_build_id": "dev", "module_text_sha256": "a" * 64,
                "normalized_code_sha256": "b" * 64,
            },
        )
        instructions = [object(), object()]
        aot = SimpleNamespace(
            function_bounds=mock.Mock(return_value=(8, 24)),
            load_frame_metadata=mock.Mock(return_value=metadata),
            disassemble_function=mock.Mock(return_value=instructions),
            validate_metadata_disassembly=mock.Mock(),
            build_frame_summary=mock.Mock(return_value=summary),
        )
        counts = [
            ({1012: 5, 1016: 7, 9999: 50}, 1000),
            ({7012: 11, 7016: 13, 9999: 50}, 7000),
        ]
        rankings = [
            {"total_samples": 100, "top_functions": [{"local_func": 10, "samples": 12}]},
            {"total_samples": 100, "top_functions": [{"local_func": 10, "samples": 24}]},
        ]
        cache = ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            root = Path(temp)
            path = root / "wamr-frame.mod0.func10.json"
            path.write_text("{}")
            frame = profile.analyze_wamr_frame_provenance(
                aot=aot,
                cwasm_info=SimpleNamespace(
                    data=b"\0" * 36, text_file_offset=4, text_size=32, version=11
                ),
                metadata_path=path,
                local_func=10,
                capture_counts=counts,
                ranking_reports=rankings,
                total_samples=200,
                spill_metric={"spill_ld": 3, "spill_st": 0},
                scratch_dir=root,
            )
        aot.build_frame_summary.assert_called_once_with(
            instructions, Counter({12: 16, 16: 20}), metadata
        )
        self.assertEqual([[4, 5], [8, 7]], frame["captures"][0]["samples_by_offset"])
        self.assertEqual(36, frame["function_samples"])
        self.assertEqual(3, frame["summary"]["reconciliation"]["emitted_allocator_loads"])
        self.assertEqual(18.0, frame["summary"]["origins"]["allocator_spill"]["percent_of_run"])
        frame.update(
            wasm_function_index=22, compiler_sha256="c" * 64, cwasm_sha256="d" * 64
        )
        report = {
            "wasm": {"local_function_count": 73, "imported_function_count": 12},
            "wamr": {"compiler_sha256": "c" * 64, "cwasm_sha256": "d" * 64},
            "wamr_captures": [
                {"total_samples": 100, "attributed_samples": 99},
                {"total_samples": 100, "attributed_samples": 99},
            ],
            "engines": {
                "wamr": {
                    "total_samples": 200,
                    "top_functions": [{"local_func": 10, "samples": 36}],
                }
            },
        }
        profile.validate_frame_provenance(frame, report)
        for case in (
            "duplicate-ip", "negative-samples", "wrong-capture", "wrong-artifact",
            "double-pair", "double-static", "false-coverage",
        ):
            with self.subTest(case=case):
                bad = copy.deepcopy(frame)
                if case == "duplicate-ip":
                    bad["captures"][0]["samples_by_offset"].append([8, 7])
                elif case == "negative-samples":
                    bad["captures"][0]["samples_by_offset"][0][1] = -5
                elif case == "wrong-capture":
                    bad["captures"][0]["total_samples"] = 101
                elif case == "wrong-artifact":
                    bad["cwasm_sha256"] = "e" * 64
                elif case == "double-pair":
                    bad["summary"]["allocator_contributors"][-1]["samples"] = 40
                elif case == "double-static":
                    bad["summary"]["allocator_contributors"][0]["static_loads"] = 2
                else:
                    bad["summary"]["coverage"]["unknown_frame_samples"] = 1
                with self.assertRaises(profile.ProfileError):
                    profile.validate_frame_provenance(bad, report)

    def test_real_mixed_pair_passes_frame_report_conservation(self):
        from tests.test_aot_jit_attr import (
            aot, aarch64_pair_metadata_for, load_aarch64_from_dict,
        )

        raw, code = aarch64_pair_metadata_for(mixed=True)
        metadata = load_aarch64_from_dict(raw, code)
        summary = aot.build_frame_summary(
            [aot.Instruction(0, 0, 4, "ldp x0, x1, [x29, #48]")],
            {0: 12},
            metadata,
        )
        for values in summary["origins"].values():
            values["percent_of_run"] = 100.0 * values["samples"] / 200
        frame = {
            "schema_version": 1, "module": 0, "local_func": 10,
            "wasm_function_index": 22, "total_samples": 200,
            "compiler_sha256": "c" * 64, "cwasm_sha256": "d" * 64,
            "sample_coordinates": "function-relative native byte offsets",
            "metadata": {
                **{key: raw[key] for key in (
                    "schema", "schema_version", "architecture", "abi",
                    "module_text_sha256", "normalized_code_sha256",
                )},
                "path": "wamr-frame.mod0.func10.json", "sha256": "e" * 64,
            },
            "code_size": 4, "function_samples": 12, "summary": summary,
            "captures": [
                {"ordinal": 1, "total_samples": 100, "function_samples": 5, "samples_by_offset": [[0, 5]]},
                {"ordinal": 2, "total_samples": 100, "function_samples": 7, "samples_by_offset": [[0, 7]]},
            ],
        }
        report = {
            "wasm": {"local_function_count": 73, "imported_function_count": 12},
            "wamr": {"compiler_sha256": "c" * 64, "cwasm_sha256": "d" * 64},
            "wamr_captures": [
                {"total_samples": 100, "attributed_samples": 99},
                {"total_samples": 100, "attributed_samples": 99},
            ],
            "engines": {
                "wamr": {
                    "total_samples": 200,
                    "top_functions": [{"local_func": 10, "samples": 12}],
                }
            },
        }
        profile.validate_frame_provenance(frame, report)
        self.assertEqual([], frame["summary"]["allocator_contributors"])
        self.assertEqual(12, frame["summary"]["coverage"]["unknown_frame_samples"])
        frame["summary"]["allocator_component_counts"]["unranked_loads"] = 0
        with self.assertRaisesRegex(profile.ProfileError, "static component"):
            profile.validate_frame_provenance(frame, report)

    def test_analysis_sources_include_loaded_comparison_dependencies(self):
        aot = profile.load_aot_helper(ROOT)
        sources = {
            path.relative_to(ROOT).as_posix()
            for path in profile.analysis_source_paths(ROOT, aot)
        }
        self.assertTrue(profile.REQUIRED_ANALYSIS_SOURCES.issubset(sources))

    def test_analysis_dependencies_must_match_committed_source(self):
        cache = ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="profile-sources-", dir=cache) as temp:
            repo = Path(temp)
            files = [
                repo / "scripts/compare_hot_function.py",
                repo / ".github/skills/aot-perf-profile/aot_jit_attr.py",
            ]
            original = b"VALUE = 1\n"
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original)
            committed = SimpleNamespace(returncode=0, stdout=original)
            with (
                mock.patch.object(profile, "analysis_source_paths", return_value=files),
                mock.patch.object(profile.subprocess, "run", return_value=committed),
            ):
                sources = profile.capture_analysis_sources(repo, None, "a" * 40)
                self.assertEqual("commit-verified", sources["source_mode"])
                profile.validate_analysis_sources_unchanged(repo, sources)
                for path in files:
                    with self.subTest(path=path):
                        path.write_bytes(b"VALUE = 2\n")
                        with self.assertRaisesRegex(profile.ProfileError, "differs from"):
                            profile.capture_analysis_sources(repo, None, "a" * 40)
                        with self.assertRaisesRegex(profile.ProfileError, "changed during"):
                            profile.validate_analysis_sources_unchanged(repo, sources)
                        path.write_bytes(original)
            with (
                mock.patch.object(profile, "analysis_source_paths", return_value=files),
                mock.patch.object(
                    profile.subprocess, "run",
                    return_value=SimpleNamespace(returncode=128, stdout=b""),
                ),
                self.assertRaisesRegex(profile.ProfileError, "not tracked"),
            ):
                profile.capture_analysis_sources(repo, None, "a" * 40)

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
            global_attributed_samples=18,
        )
        self.assertEqual(9, result["broad_alu_samples"])
        self.assertEqual(
            9, result["categories"]["address_generation"]["samples"]
        )
        self.assertEqual(13, result["sample_mapping"]["mapped_function_samples"])
        self.assertEqual(0, result["sample_mapping"]["unresolved_function_samples"])
        self.assertEqual(
            2, result["global_sample_mapping"]["unattributed_samples"]
        )

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
                samples_by_offset={0: 1, 4: 29},
                total_run_samples=100,
                global_attributed_samples=99,
            )
        )
        narrow_wasmtime = profile.aarch64_instruction_provenance.analyze_instruction_stream(
            narrow_instructions,
            broad_classes=["alu", "other", "other"],
            samples_by_offset={0: 1, 4: 27},
            total_run_samples=100,
            global_attributed_samples=98,
        )
        report["classifier_wording"]["narrow_alu_provenance"] = (
            profile.NARROW_ALU_WORDING
        )
        report["provenance"]["analysis_module"] = {
            "path": "scripts/aarch64_instruction_provenance.py",
            "sha256": "a" * 64,
        }
        report["provenance"]["script_path"] = "scripts/profile_coremark_aarch64.py"
        report["benchmark"]["producer"]["script"] = {
            "path": "scripts/bench_coremark.py",
            "sha256": "e" * 64,
        }
        source_hashes = {path: "f" * 64 for path in profile.REQUIRED_ANALYSIS_SOURCES}
        source_hashes.update({
            "scripts/profile_coremark_aarch64.py": "d" * 64,
            "scripts/aarch64_instruction_provenance.py": "a" * 64,
            "scripts/bench_coremark.py": "e" * 64,
        })
        report["provenance"]["analysis_sources"] = {
            "source_mode": "commit-verified",
            "commit": "c" * 40,
            "files": source_hashes,
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
            "target_local_func": 3,
            "target_wasm_function_index": 15,
            "gate": report["matched_functions"][0]["alu_provenance"]["gate"],
        }
        report["retained_analysis_artifacts"] = {
            "wamr_cwasm": {
                "retained": True,
                "source_sha256": "1" * 64,
            }
        }
        profile.validate_report(report)
        missing_dependency = copy.deepcopy(report)
        del missing_dependency["provenance"]["analysis_sources"]["files"][
            "scripts/compare_hot_function.py"
        ]
        with self.assertRaisesRegex(profile.ProfileError, "complete committed"):
            profile.validate_report(missing_dependency)
        wrong_global_mapping = copy.deepcopy(report)
        wrong_global_mapping["engines"]["wasmtime"]["attributed_samples"] = 99
        with self.assertRaisesRegex(profile.ProfileError, "global sample mapping"):
            profile.validate_report(wrong_global_mapping)
        missing_samples = copy.deepcopy(report)
        missing_samples["matched_functions"][0]["wasmtime"]["samples"] = 29
        with self.assertRaisesRegex(profile.ProfileError, "matched function"):
            profile.validate_report(missing_samples)
        forged_gate = copy.deepcopy(report)
        forged_gate["matched_functions"][0]["alu_provenance"]["gate"][
            "optimization_authorized"
        ] = True
        with self.assertRaisesRegex(profile.ProfileError, "validated sample evidence"):
            profile.validate_report(forged_gate)
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
