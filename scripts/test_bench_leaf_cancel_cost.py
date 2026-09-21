#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_leaf_cancel_cost as bench  # noqa: E402


class LeafCancelCostTests(unittest.TestCase):
    def test_expected_result_is_closed_form_and_requires_unroll(self) -> None:
        calls = 128
        result = bench.expected_result(calls)
        self.assertEqual(result["batches"], 2)
        self.assertEqual(
            result["checksum"],
            (bench.LEAF_SEED + bench.LEAF_STEP * calls) & bench.MASK64,
        )
        with self.assertRaises(bench.HarnessError):
            bench.expected_result(65)

    def test_guest_parser_rejects_work_mismatch(self) -> None:
        expected = bench.expected_result(64)
        result = {
            **expected,
            "raw_elapsed_ns": 2_000_100,
            "timing_overhead_ns": 100,
            "elapsed_ns": 2_000_000,
            "timing_overhead_ppm": 49,
        }
        parsed = bench.parse_guest_result(
            json.dumps(result), 64, 1_000_000, enforce_quality=True
        )
        self.assertEqual(parsed["checksum"], expected["checksum"])
        result["leaf_calls"] = 128
        with self.assertRaisesRegex(bench.HarnessError, "leaf_calls"):
            bench.parse_guest_result(
                json.dumps(result), 64, 1_000_000, enforce_quality=True
            )

    def test_sizing_rounds_to_complete_unrolled_batches(self) -> None:
        selected = bench.select_calls(
            pilot_calls=64,
            pilot_elapsed_ns=100,
            target_interval_ns=201,
        )
        self.assertEqual(selected, 192)
        self.assertEqual(selected % bench.LEAF_UNROLL, 0)
        with self.assertRaisesRegex(bench.HarnessError, "safety limit"):
            bench.select_calls(
                pilot_calls=bench.MAX_LEAF_CALLS,
                pilot_elapsed_ns=1,
                target_interval_ns=2,
            )

    def test_balanced_pair_summary_reports_on_minus_off(self) -> None:
        records = [
            {
                "pair_index": 0,
                "condition": "cancel-points-off",
                "guest_elapsed_ns": 100,
                "leaf_calls_per_second": 640_000_000,
                "host_wall_elapsed_ns": 120,
            },
            {
                "pair_index": 0,
                "condition": "cancel-points-on",
                "guest_elapsed_ns": 120,
                "leaf_calls_per_second": 533_333_333,
                "host_wall_elapsed_ns": 140,
            },
            {
                "pair_index": 1,
                "condition": "cancel-points-on",
                "guest_elapsed_ns": 132,
                "leaf_calls_per_second": 484_848_484,
                "host_wall_elapsed_ns": 150,
            },
            {
                "pair_index": 1,
                "condition": "cancel-points-off",
                "guest_elapsed_ns": 110,
                "leaf_calls_per_second": 581_818_181,
                "host_wall_elapsed_ns": 130,
            },
        ]
        _, comparison = bench.summarize(records, 64)
        self.assertAlmostEqual(
            comparison["median_on_over_off_elapsed_ratio"], 1.2
        )
        self.assertAlmostEqual(
            comparison["median_on_minus_off_ns_per_leaf_call"], 21 / 64
        )

    def test_invocation_validation_rejects_reordered_pair(self) -> None:
        expected = bench.expected_result(64)
        pair = []
        for position, condition in enumerate(bench.condition_order(0)):
            pair.append(
                {
                    "phase": "sample",
                    "pair_index": 0,
                    "position": position,
                    "condition": condition,
                    "leaf_calls": 64,
                    "leaf_entry_poll_opportunities": (
                        64 if condition == "cancel-points-on" else 0
                    ),
                    "batches": 1,
                    "guest": expected,
                    "correct": True,
                }
            )
        bench.validate_invocations(pair, phase="sample", pairs=1, calls=64)
        with self.assertRaisesRegex(bench.HarnessError, "ordering"):
            bench.validate_invocations(
                list(reversed(pair)), phase="sample", pairs=1, calls=64
            )

    def test_measurement_requires_leaf_entry_artifact_proof(self) -> None:
        dummy = ROOT / "zig-out/leaf-cancel-proof-test"
        build = bench.Build(
            "dummy", dummy, dummy / "wamr", None, [], "0" * 64
        )
        with self.assertRaisesRegex(bench.HarnessError, "artifact proof"):
            bench.measure(
                repo=ROOT,
                runner=[],
                runtime=build,
                artifact=dummy / "leaf.cwasm",
                condition="cancel-points-on",
                calls=64,
                timeout=1,
                minimum_interval_ns=1,
                enforce_quality=True,
                phase="sample",
                pair_index=0,
                position=0,
                leaf_entry_poll_proven=False,
            )

    def test_artifact_identity_proves_commands_layout_and_leaf_entry(self) -> None:
        directory = ROOT / "zig-out/leaf-cancel-cost-test"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for arch, sequence in self.artifact_test_architectures():
                with self.subTest(arch=arch):
                    off = directory / f"{arch}-off.cwasm"
                    on = directory / f"{arch}-on.cwasm"
                    body = self.entry_body(arch)
                    off.write_bytes(self.aot([body], leaf_local=0))
                    on.write_bytes(
                        self.aot(
                            [
                                bench.CANCEL_POLL_ENTRY_PREFIX_SUFFIXES[arch]
                                + sequence
                                + self.return_instruction(arch)
                            ],
                            leaf_local=0,
                        )
                    )
                    commands = self.compile_commands(arch, off, on)
                    identity = bench.artifact_identity(
                        {"cancel-points-off": off, "cancel-points-on": on},
                        commands,
                        arch,
                        directory / f"{arch}-normalization",
                    )
                    self.assertEqual(identity["cancel_poll_sites_enabled"], 1)
                    self.assertEqual(identity["cancel_poll_sites_disabled"], 0)
                    self.assertEqual(
                        identity["text_delta_bytes"], len(sequence)
                    )
                    self.assertEqual(
                        identity["leaf_step"]["local_function_index"], 0
                    )
                    self.assertTrue(
                        identity["leaf_step"]["normalized_body_identical"]
                    )
                    off.unlink()
                    on.unlink()
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_global_poll_sequence_without_leaf_entry_is_rejected(self) -> None:
        directory = ROOT / "zig-out/leaf-cancel-cost-global-test"
        directory.mkdir(parents=True, exist_ok=True)
        arch = "aarch64"
        sequence = bench.CANCEL_POLL_SEQUENCES[arch]
        body = self.entry_body(arch)
        other = self.return_instruction(arch)
        off = directory / "off.cwasm"
        on = directory / "on.cwasm"
        off.write_bytes(self.aot([body, other], leaf_local=0))
        on.write_bytes(self.aot([body, sequence + other], leaf_local=0))
        try:
            with self.assertRaisesRegex(bench.HarnessError, "leaf_step"):
                bench.artifact_identity(
                    {"cancel-points-off": off, "cancel-points-on": on},
                    self.compile_commands(arch, off, on),
                    arch,
                    directory / "normalization",
                )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_unrelated_function_difference_is_rejected(self) -> None:
        directory = ROOT / "zig-out/leaf-cancel-cost-difference-test"
        directory.mkdir(parents=True, exist_ok=True)
        for arch, sequence in self.artifact_test_architectures():
            with self.subTest(arch=arch):
                body = self.entry_body(arch)
                changed = (
                    b"\x90"
                    if arch == "x86_64"
                    else struct.pack("<I", 0xD503201F)
                )
                off = directory / f"{arch}-off.cwasm"
                on = directory / f"{arch}-on.cwasm"
                off.write_bytes(self.aot([body], leaf_local=0))
                on.write_bytes(
                    self.aot(
                        [
                            bench.CANCEL_POLL_ENTRY_PREFIX_SUFFIXES[arch]
                            + sequence
                            + changed
                        ],
                        leaf_local=0,
                    )
                )
                with self.assertRaisesRegex(
                    bench.HarnessError, "unrelated|layout"
                ):
                    bench.artifact_identity(
                        {"cancel-points-off": off, "cancel-points-on": on},
                        self.compile_commands(arch, off, on),
                        arch,
                        directory / f"{arch}-normalization",
                    )
        shutil.rmtree(directory, ignore_errors=True)

    def test_compile_command_extra_option_is_rejected(self) -> None:
        directory = ROOT / "zig-out/leaf-cancel-cost-command-test"
        directory.mkdir(parents=True, exist_ok=True)
        arch = "aarch64"
        sequence = bench.CANCEL_POLL_SEQUENCES[arch]
        body = self.entry_body(arch)
        off = directory / "off.cwasm"
        on = directory / "on.cwasm"
        off.write_bytes(self.aot([body], leaf_local=0))
        on.write_bytes(
            self.aot(
                [
                    bench.CANCEL_POLL_ENTRY_PREFIX_SUFFIXES[arch]
                    + sequence
                    + self.return_instruction(arch)
                ],
                leaf_local=0,
            )
        )
        commands = self.compile_commands(arch, off, on)
        commands["cancel-points-on"].insert(4, "-O0")
        try:
            with self.assertRaisesRegex(bench.HarnessError, "commands differ"):
                bench.artifact_identity(
                    {"cancel-points-off": off, "cancel-points-on": on},
                    commands,
                    arch,
                    directory / "normalization",
                )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_unrelated_aot_metadata_difference_is_rejected(self) -> None:
        directory = ROOT / "zig-out/leaf-cancel-cost-metadata-test"
        directory.mkdir(parents=True, exist_ok=True)
        arch = "aarch64"
        sequence = bench.CANCEL_POLL_SEQUENCES[arch]
        body = self.entry_body(arch)
        off = directory / "off.cwasm"
        on = directory / "on.cwasm"
        off.write_bytes(self.aot([body], leaf_local=0))
        enabled = bytearray(
            self.aot(
                [
                    bench.CANCEL_POLL_ENTRY_PREFIX_SUFFIXES[arch]
                    + sequence
                    + self.return_instruction(arch)
                ],
                leaf_local=0,
            )
        )
        enabled[16] = 1
        on.write_bytes(enabled)
        try:
            with self.assertRaisesRegex(bench.HarnessError, "section 0"):
                bench.artifact_identity(
                    {"cancel-points-off": off, "cancel-points-on": on},
                    self.compile_commands(arch, off, on),
                    arch,
                    directory / "normalization",
                )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_schema_requires_retained_evidence(self) -> None:
        schema = json.loads(
            (
                ROOT
                / "tests/benchmarks/leaf-cancel-cost/report.schema.json"
            ).read_text(encoding="UTF-8")
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        plan_required = schema["$defs"]["plan"]["required"]
        self.assertIn(
            "non_leaf_poll_opportunities_upper_bound_per_enabled_sample",
            plan_required,
        )
        metadata_required = schema["$defs"]["metadata"]["required"]
        for field in ("source", "fixture", "host", "tools", "artifacts", "execution"):
            self.assertIn(field, metadata_required)
        invocation_required = schema["$defs"]["invocation"]["required"]
        for field in ("command", "guest", "host_wall_elapsed_ns", "correct"):
            self.assertIn(field, invocation_required)
        artifact_required = schema["$defs"]["artifacts"]["required"]
        for field in (
            "compile_commands",
            "compile_command_comparison",
            "normalized_comparison",
            "leaf_step",
        ):
            self.assertIn(field, artifact_required)

    @staticmethod
    def return_instruction(arch: str) -> bytes:
        return b"\xc3" if arch == "x86_64" else struct.pack("<I", 0xD65F03C0)

    @staticmethod
    def artifact_test_architectures() -> list[tuple[str, bytes]]:
        result = subprocess.run(
            ["objdump", "-i"],
            text=True,
            capture_output=True,
            check=False,
        )
        supports_x86 = (
            result.returncode == 0 and "i386:x86-64" in result.stdout
        )
        return [
            (arch, sequence)
            for arch, sequence in bench.CANCEL_POLL_SEQUENCES.items()
            if arch != "x86_64" or supports_x86
        ]

    @classmethod
    def entry_body(cls, arch: str) -> bytes:
        return (
            bench.CANCEL_POLL_ENTRY_PREFIX_SUFFIXES[arch]
            + cls.return_instruction(arch)
        )

    @staticmethod
    def section(section_type: int, payload: bytes) -> bytes:
        return struct.pack("<II", section_type, len(payload)) + payload

    @classmethod
    def aot(cls, functions: list[bytes], *, leaf_local: int) -> bytes:
        offsets = []
        text = bytearray()
        for function in functions:
            offsets.append(len(text))
            text.extend(function)
        function_payload = bytearray(struct.pack("<I", len(functions)))
        for offset in offsets:
            function_payload.extend(struct.pack("<II", offset, 0))
        name = b"leaf_step"
        export_payload = (
            struct.pack("<II", 1, len(name))
            + name
            + b"\x00"
            + struct.pack("<I", leaf_local)
        )
        return (
            b"\x00aot"
            + struct.pack("<I", bench.AOT_VERSION)
            + cls.section(0, b"\x00" * 40)
            + cls.section(2, bytes(text))
            + cls.section(3, bytes(function_payload))
            + cls.section(4, export_payload)
            + cls.section(8, struct.pack("<I", 0))
        )

    @staticmethod
    def compile_commands(
        arch: str, off: Path, on: Path
    ) -> dict[str, list[str]]:
        base = ["wamrc", "compile", "--target", arch]
        fixture = str(ROOT / bench.FIXTURE)
        return {
            "cancel-points-off": [
                *base,
                "--benchmark-disable-cancel-points",
                fixture,
                "-o",
                str(off),
            ],
            "cancel-points-on": [*base, fixture, "-o", str(on)],
        }


if __name__ == "__main__":
    unittest.main()
