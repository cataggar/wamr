#!/usr/bin/env python3
"""Unit tests for P3 timing instrumentation and artifact aggregation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import wasi_p3_profile as profile  # noqa: E402
import wasi_p3_profile_aggregate as aggregate  # noqa: E402


def _load_adapter():
    path = ROOT / "tests" / "wasi-testsuite-adapter" / "wamr-zig.py"
    spec = importlib.util.spec_from_file_location("wamr_zig_profile_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProfileTestCase(unittest.TestCase):
    scratch = ROOT / "zig-out" / "wasi-p3-profile-unit"

    def setUp(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def test_jsonl_parser_rejects_unknown_phase(self) -> None:
        path = self.scratch / "bad.jsonl"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "event": "phase_timing",
                    "run_id": "aot-cold-01",
                    "mode": "aot",
                    "fixture": "noop",
                    "phase": "combined",
                    "artifact_kind": "core",
                    "cache": "miss",
                    "duration_ns": 1,
                    "pid": 1,
                }
            )
            + "\n",
            encoding="UTF-8",
        )
        with self.assertRaises(profile.ProfileDataError):
            profile.parse_jsonl(path)

    def test_adapter_records_core_miss_and_hit_only_when_enabled(self) -> None:
        adapter = _load_adapter()
        wasm = self.scratch / "fixture.wasm"
        wasm.write_bytes(b"\x00asm\x01\x00\x00\x00")
        timing = self.scratch / "timings.jsonl"

        def fake_compile(command, fixture, phase):
            del fixture, phase
            Path(command[command.index("-o") + 1]).write_bytes(b"cwasm")

        with mock.patch.dict(
            os.environ,
            {
                "WAMR_PROFILE_TIMINGS": str(timing),
                "WAMR_PROFILE_RUN_ID": "aot-cold-01",
                "WAMR_PROFILE_MODE": "aot",
            },
            clear=False,
        ), mock.patch.object(adapter, "_run_compile", side_effect=fake_compile):
            compiled = adapter._precompile(str(wasm))
            self.assertEqual(Path(compiled), wasm.with_suffix(".cwasm"))
            adapter._precompile(str(wasm))

        events = profile.parse_jsonl(timing)
        self.assertEqual([event["cache"] for event in events], ["miss", "hit"])
        self.assertTrue(all(event["phase"] == "core_precompile" for event in events))

        timing.unlink()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            adapter, "_run_compile", side_effect=fake_compile
        ):
            wasm.with_suffix(".cwasm").unlink()
            adapter._precompile(str(wasm))
        self.assertFalse(timing.exists())

    def test_default_compute_argv_has_no_instrumentation_wrapper(self) -> None:
        adapter = _load_adapter()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            adapter, "_precompile", return_value="fixture.cwasm"
        ):
            argv = adapter.compute_argv(
                "fixture.wasm", ([], {}, None), [], "wasi:cli/command", "wasm32-wasip3"
            )
        self.assertEqual(argv, adapter.WAMR + ["run", "fixture.cwasm"])

    def test_checked_in_schemas_match_parser_phases(self) -> None:
        run_schema = json.loads(
            (
                ROOT
                / "tests"
                / "benchmarks"
                / "wasi-p3-profile"
                / "run.schema.json"
            ).read_text(encoding="UTF-8")
        )
        aggregate_schema = json.loads(
            (
                ROOT
                / "tests"
                / "benchmarks"
                / "wasi-p3-profile"
                / "aggregate.schema.json"
            ).read_text(encoding="UTF-8")
        )
        self.assertEqual(run_schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(
            set(run_schema["$defs"]["event"]["properties"]["phase"]["enum"]),
            profile.PHASES,
        )
        self.assertEqual(
            aggregate_schema["properties"]["kind"]["const"],
            "wasi-p3-profile-aggregate",
        )

    def _write_run(self, platform_id: str, mode: str) -> None:
        base = self.scratch / f"wasi-p3-measure-{platform_id}"
        mode_dir = base / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for temperature, count in (("cold", 5), ("warm", 10)):
            for index in range(1, count + 1):
                events = []
                for fixture_index in range(profile.EXPECTED_FIXTURES):
                    fixture = (
                        "http-fields" if fixture_index == 0 else f"fixture-{fixture_index}"
                    )
                    scale = 3 if platform_id == "macos-arm64" else 1
                    events += [
                        {
                            "schema_version": 1,
                            "event": "phase_timing",
                            "run_id": f"{mode}-{temperature}-{index:02d}",
                            "mode": mode,
                            "fixture": fixture,
                            "phase": "component_precompile",
                            "artifact_kind": "component",
                            "cache": (
                                "bypass"
                                if mode == "jit"
                                else "miss"
                                if temperature == "cold"
                                else "hit"
                            ),
                            "duration_ns": 0 if mode == "jit" else 100 * scale,
                            "pid": 1,
                        },
                        {
                            "schema_version": 1,
                            "event": "phase_timing",
                            "run_id": f"{mode}-{temperature}-{index:02d}",
                            "mode": mode,
                            "fixture": fixture,
                            "phase": "fixture_execution",
                            "artifact_kind": "component",
                            "cache": "n/a",
                            "duration_ns": (
                                300_000_000 * scale
                                if fixture == "http-fields"
                                else 1_000_000 * scale
                            ),
                            "pid": 1,
                        },
                    ]
                samples.append(
                    {
                        "id": f"{mode}-{temperature}-{index:02d}",
                        "temperature": temperature,
                        "index": index,
                        "valid": True,
                        "errors": [],
                        "returncode": 0,
                        "suite_duration_ns": 1_000_000_000 * scale,
                        "counts": {
                            "fixtures": 41,
                            "executed": 41,
                            "passed": 41,
                            "failed": 0,
                        },
                        "events": events,
                    }
                )
        document = {
            "schema_version": 1,
            "kind": "wasi-p3-profile-runs",
            "metadata": {
                "platform_id": platform_id,
                "mode": mode,
                "optimize": "ReleaseSafe",
                "commit": "a" * 40,
                "host": {
                    "system": "Darwin" if platform_id == "macos-arm64" else "Linux",
                    "machine": "arm64" if platform_id == "macos-arm64" else "aarch64",
                },
                "tools": {"zig": "0.16.0", "wamr": "wamr dev", "wamrc": "wamrc dev"},
                "cache": {"zig_global_cache_dir": "isolated"},
            },
            "plan": {
                "cold_samples": 5,
                "warm_samples": 10,
                "optimize": "ReleaseSafe",
            },
            "samples": samples,
        }
        (mode_dir / "samples.json").write_text(
            json.dumps(document), encoding="UTF-8"
        )
        microbench = base / "microbench"
        microbench.mkdir(exist_ok=True)
        (microbench / "report.json").write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "name": f"scenario-{i}",
                            "median_ns": 10,
                            "p95_ns": 12,
                            "verdict": "no-budget",
                        }
                        for i in range(4)
                    ]
                }
            ),
            encoding="UTF-8",
        )

    def test_aggregate_requires_matched_valid_41_of_41_samples(self) -> None:
        for platform_id in ("macos-arm64", "linux-arm64"):
            for mode in ("aot", "jit"):
                self._write_run(platform_id, mode)
        output = self.scratch / "aggregate"
        result = aggregate.aggregate(
            argparse.Namespace(input_dir=self.scratch, output_dir=output)
        )
        self.assertEqual(result, 0)
        document = json.loads(
            (output / "aggregate.json").read_text(encoding="UTF-8")
        )
        aggregate.validate_aggregate_document(document)
        selection = json.loads(
            (output / "profile-selection.json").read_text(encoding="UTF-8")
        )
        keys = {
            (item["mode"], item["fixture"], item["phase"])
            for item in selection["profiles"]
        }
        self.assertIn(("aot", "http-fields", "fixture_execution"), keys)
        self.assertIn(("jit", "http-fields", "fixture_execution"), keys)


    def test_sample_timeout_terminates_runner_and_its_children(self) -> None:
        """A wedged sample must be bounded and leave nothing behind.

        Before #616 D3 the runner was invoked without a timeout, so a
        stuck `wamrc` ran until the CI job timeout cancelled the job,
        which skipped the `if: always()` upload and destroyed the
        evidence.
        """
        marker = self.scratch / "grandchild-started"
        stub = self.scratch / "hang.py"
        stub.write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c',\n"
            f"    \"import time; open({str(marker)!r}, 'w').close();\"\n"
            '    "time.sleep(600)"])\n'
            "time.sleep(600)\n",
            encoding="UTF-8",
        )
        with mock.patch.object(profile, "UNFILTERED", stub):
            sample = profile._run_sample(
                "jit", "warm", 1, self.scratch / "out", timeout_s=5
            )

        self.assertTrue(sample["timed_out"])
        self.assertFalse(sample["valid"])
        self.assertLess(sample["suite_duration_ns"], 60_000_000_000)
        self.assertIn("exceeded", sample["errors"][0])
        # The document must still satisfy the schema so partial evidence
        # can be uploaded and inspected.
        profile.validate_run_document(
            {
                "schema_version": profile.SCHEMA_VERSION,
                "kind": "wasi-p3-profile-runs",
                "metadata": {
                    "mode": "jit",
                    "platform_id": "test",
                    "commit": "0" * 40,
                },
                "samples": [sample],
            }
        )
        self.assertTrue(marker.exists(), "stub never spawned a grandchild")
        if os.name != "nt":
            import subprocess as sp

            survivors = sp.run(
                ["pgrep", "-f", "time.sleep(600)"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.split()
            self.assertEqual(survivors, [], "descendant processes leaked")

    def test_collection_deadline_stops_early_and_persists_evidence(self) -> None:
        output_dir = self.scratch / "deadline"
        args = argparse.Namespace(
            mode="jit",
            platform_id="test",
            output_dir=output_dir,
            cold_samples=5,
            warm_samples=10,
            sample_timeout=900.0,
            deadline_minutes=0.001,
        )
        self.assertEqual(profile.run_collection(args), 1)
        document = json.loads(
            (output_dir / "samples.json").read_text(encoding="UTF-8")
        )
        self.assertTrue(document["stopped_early"])
        self.assertIn("deadline", document["stop_reason"])
        self.assertEqual(document["planned_samples"], 15)
        self.assertEqual(document["samples"], [])

    def test_compile_timeout_is_bounded_and_configurable(self) -> None:
        adapter = _load_adapter()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                adapter.get_compile_timeout_seconds(),
                adapter._DEFAULT_COMPILE_TIMEOUT_SECONDS,
            )
        with mock.patch.dict(os.environ, {"WAMR_COMPILE_TIMEOUT": "12"}, clear=True):
            self.assertEqual(adapter.get_compile_timeout_seconds(), 12.0)
        with mock.patch.dict(os.environ, {"WAMR_COMPILE_TIMEOUT": "nope"}, clear=True):
            self.assertEqual(
                adapter.get_compile_timeout_seconds(),
                adapter._DEFAULT_COMPILE_TIMEOUT_SECONDS,
            )
        with mock.patch.dict(os.environ, {"WAMR_COMPILE_TIMEOUT": "2"}, clear=True):
            with self.assertRaises(RuntimeError) as caught:
                adapter._run_compile(
                    [sys.executable, "-c", "import time; time.sleep(120)"],
                    Path("fixture.wasm"),
                    "component_precompile",
                )
        self.assertIn("WAMR_COMPILE_TIMEOUT", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
