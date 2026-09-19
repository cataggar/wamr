#!/usr/bin/env python3

from __future__ import annotations

import json
import struct
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

    def test_artifact_identity_requires_signature_only_when_enabled(self) -> None:
        directory = ROOT / "zig-out/leaf-cancel-cost-test"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for arch, signature in bench.CANCEL_POLL_SIGNATURES.items():
                with self.subTest(arch=arch):
                    off = directory / f"{arch}-off.cwasm"
                    on = directory / f"{arch}-on.cwasm"
                    off.write_bytes(self.aot(b"\x90" * 16))
                    on.write_bytes(
                        self.aot(b"\x90" * 16 + signature + b"\x90" * 3)
                    )
                    identity = bench.artifact_identity(
                        {"cancel-points-off": off, "cancel-points-on": on}, arch
                    )
                    self.assertEqual(identity["cancel_poll_sites_enabled"], 1)
                    self.assertEqual(identity["cancel_poll_sites_disabled"], 0)
                    self.assertEqual(
                        identity["text_delta_bytes"], len(signature) + 3
                    )
                    off.unlink()
                    on.unlink()
        finally:
            for path in directory.glob("*.cwasm"):
                path.unlink()
            directory.rmdir()

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

    @staticmethod
    def aot(text: bytes) -> bytes:
        return (
            b"\x00aot"
            + struct.pack("<I", bench.AOT_VERSION)
            + struct.pack("<II", 2, len(text))
            + text
        )


if __name__ == "__main__":
    unittest.main()
