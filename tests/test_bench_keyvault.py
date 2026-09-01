#!/usr/bin/env python3
"""Controlled tests for the pinned keyvault/TCGC benchmark harness."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import platform
import shutil
import stat
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import bench_keyvault as bench  # noqa: E402


def _load_attr_module():
    path = ROOT / ".github" / "skills" / "aot-perf-profile" / "aot_jit_attr.py"
    spec = importlib.util.spec_from_file_location("aot_jit_attr_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KeyvaultHarnessTest(unittest.TestCase):
    scratch_root = ROOT / "zig-out"

    def setUp(self) -> None:
        self.scratch = self.scratch_root / f"keyvault-harness-{self._testMethodName}"
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.sdk = self.scratch / "sdk"
        self.spec_repo = self.scratch / "spec-repo"
        self.component = self.sdk / "component.wasm"
        self.preopens = self.sdk / "stdlib-preopens.txt"
        self.stdlib = self.sdk / "stdlib"
        self.spec = self.spec_repo / "Secrets"
        self.stdlib.mkdir(parents=True)
        self.spec.mkdir(parents=True)
        self.component.write_bytes(b"\0asm controlled component")
        (self.stdlib / "lib.tsp").write_text("model Library {}\n", encoding="UTF-8")
        (self.spec / "main.tsp").write_text("model Secret {}\n", encoding="UTF-8")
        self.preopens.write_text(
            f"{self.stdlib.resolve()}=/stdlib\n", encoding="UTF-8"
        )
        self.tool = self.scratch / "fake-tool"
        self.tool.write_text("#!/bin/sh\necho pinned-tool-1.0\n", encoding="UTF-8")
        self.tool.chmod(self.tool.stat().st_mode | stat.S_IXUSR)
        self.sdk_revision = self._init_repo(self.sdk)
        self.spec_revision = self._init_repo(self.spec_repo)
        self.manifest = self.scratch / "manifest.json"
        self.manifest.write_text(
            json.dumps(self._manifest_document(), indent=2), encoding="UTF-8"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _init_repo(self, path: Path) -> str:
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "config", "user.name", "Harness Test"],
            check=True,
        )
        subprocess.run(["git", "-C", str(path), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(path), "commit", "-qm", "fixture"], check=True
        )
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    def _manifest_document(self) -> dict:
        tool_sha = bench.sha256_file(self.tool)
        return {
            "schema_version": 1,
            "tools": {
                name: {
                    "path": str(self.tool),
                    "sha256": tool_sha,
                    "version": "pinned-tool-1.0",
                }
                for name in ("wamr", "wamrc", "wasmtime")
            },
            "sources": [
                {
                    "name": "azure-sdk-for-zig",
                    "path": str(self.sdk),
                    "revision": self.sdk_revision,
                },
                {
                    "name": "azure-rest-api-specs",
                    "path": str(self.spec_repo),
                    "revision": self.spec_revision,
                },
            ],
            "workload": {
                "component": {
                    "path": str(self.component),
                    "sha256": bench.sha256_file(self.component),
                },
                "preopens_file": {
                    "path": str(self.preopens),
                    "sha256": bench.sha256_file(self.preopens),
                },
                "mounts": [
                    {
                        "path": str(self.spec),
                        "guest": "/spec",
                        "sha256_tree": bench.tree_snapshot(self.spec)["sha256"],
                    },
                    {
                        "path": str(self.stdlib),
                        "guest": "/stdlib",
                        "sha256_tree": bench.tree_snapshot(self.stdlib)["sha256"],
                    },
                ],
                "package_name": "keyvault-secrets",
                "env": {},
                "expected_tcgc_response_bytes": 58187,
            },
            "perf": {
                "core_index": 4,
                "hot_func": 6145,
                "min_samples": 5000,
                "min_attribution_coverage_pct": 50,
                "base": None,
            },
        }

    def test_manifest_validation_checks_hashes_revisions_and_preopens(self) -> None:
        config = bench.load_manifest(self.manifest)
        with mock.patch.object(
            bench, "command_version", return_value="pinned-tool-1.0"
        ), mock.patch.object(bench.os, "access", return_value=True):
            report = bench.validate_config(config)
        self.assertEqual(report["inputs"]["component"]["size"], len(self.component.read_bytes()))
        self.assertEqual(len(report["sources"]), 2)
        self.assertEqual(report["inputs"]["mounts"][1]["file_count"], 1)

        (self.stdlib / "lib.tsp").write_text("changed\n", encoding="UTF-8")
        with self.assertRaisesRegex(
            bench.HarnessError, "tracked modifications|tree SHA-256 mismatch"
        ), mock.patch.object(
            bench, "command_version", return_value="pinned-tool-1.0"
        ), mock.patch.object(bench.os, "access", return_value=True):
            bench.validate_config(config)

    def test_manifest_rejects_placeholders_and_short_revisions(self) -> None:
        document = self._manifest_document()
        document["tools"]["wamr"]["sha256"] = "0" * 64
        document["sources"][0]["revision"] = "19e06fc47"
        self.manifest.write_text(json.dumps(document), encoding="UTF-8")
        with self.assertRaisesRegex(bench.HarnessError, "non-zero"):
            bench.load_manifest(self.manifest)

        document["tools"]["wamr"]["sha256"] = bench.sha256_file(self.tool)
        self.manifest.write_text(json.dumps(document), encoding="UTF-8")
        with self.assertRaisesRegex(bench.HarnessError, "40-character"):
            bench.load_manifest(self.manifest)

    def test_relative_paths_resolve_against_manifest_directory(self) -> None:
        document = self._manifest_document()
        manifest_dir = self.manifest.parent.resolve()
        document["workload"]["component"]["path"] = str(
            self.component.resolve().relative_to(manifest_dir)
        )
        document["workload"]["preopens_file"]["path"] = str(
            self.preopens.resolve().relative_to(manifest_dir)
        )
        document["workload"]["mounts"][0]["path"] = str(
            self.spec.resolve().relative_to(manifest_dir)
        )
        document["perf"]["base"] = "0x400000"
        self.manifest.write_text(json.dumps(document), encoding="UTF-8")

        config = bench.load_manifest(self.manifest)
        self.assertEqual(config.component, self.component.resolve())
        self.assertEqual(config.preopens_file, self.preopens.resolve())
        self.assertEqual(config.mounts[0].host, self.spec.resolve())
        self.assertEqual(config.perf["base"], "0x400000")

    def test_checked_in_schema_and_example_track_parser_version(self) -> None:
        directory = ROOT / "tests" / "benchmarks" / "keyvault"
        schema = json.loads(
            (directory / "manifest.schema.json").read_text(encoding="UTF-8")
        )
        example = json.loads(
            (directory / "manifest.example.json").read_text(encoding="UTF-8")
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(example["schema_version"], bench.SCHEMA_VERSION)
        self.assertEqual(
            example["workload"]["component"]["sha256"],
            "fe08b02fa85209855850ba1ea72bc3ab554cec961fdee005cacb6348d9da8c92",
        )

    def test_command_construction_precompiles_then_runs_artifacts(self) -> None:
        config = bench.load_manifest(self.manifest)
        artifacts = self.scratch / "artifacts"
        wamr_compile, wasmtime_compile, wamr_manifest, wasmtime_cwasm = (
            bench.precompile_commands(config, artifacts)
        )
        self.assertEqual(
            wamr_compile[1:4],
            ["compile-component", "--target=x86_64", str(self.component)],
        )
        self.assertIn("max-memory-size=4294967296", wasmtime_compile)
        self.assertEqual(wasmtime_compile[-2:], ["-o", str(wasmtime_cwasm)])

        out = self.scratch / "out"
        wamr_run = bench.runtime_command(
            config, "wamr", out, wamr_manifest, wasmtime_cwasm
        )
        wasmtime_run = bench.runtime_command(
            config, "wasmtime", out, wamr_manifest, wasmtime_cwasm
        )
        self.assertIn("--precompiled-manifest", wamr_run)
        self.assertIn(str(wamr_manifest), wamr_run)
        self.assertIn("WAMR_KEYVAULT_HARNESS=1", wamr_run)
        self.assertIn("--allow-precompiled", wasmtime_run)
        self.assertIn(str(wasmtime_cwasm), wasmtime_run)
        self.assertIn("WAMR_KEYVAULT_HARNESS=1", wasmtime_run)
        self.assertNotIn("compile-component", wamr_run)
        self.assertNotIn("compile", wasmtime_run)

    def test_timing_and_tcgc_output_parsing(self) -> None:
        self.assertEqual(
            bench.parse_tcgc_response_bytes(
                "calling tcgc.compile...\ngot 58187 bytes back from tcgc.compile\n"
            ),
            58187,
        )
        with self.assertRaisesRegex(bench.HarnessError, "exactly one"):
            bench.parse_tcgc_response_bytes("no response marker")
        records = [
            {
                "runtime": "wamr",
                "phase": "measure",
                "elapsed_seconds": value,
            }
            for value in (5.1, 5.2, 5.3, 5.4, 5.5)
        ]
        self.assertEqual(bench.parse_timing_samples(records, "wamr")[2], 5.3)
        with self.assertRaisesRegex(bench.HarnessError, "at least 5"):
            bench.parse_timing_samples(records[:4], "wamr")

    def test_equivalence_detects_58187_58188_and_file_hash_drift(self) -> None:
        output = {
            "sha256_tree": "a" * 64,
            "file_count": 1,
            "total_bytes": 4,
            "files": [{"path": "a.zig", "size": 4, "sha256": "b" * 64}],
        }
        base = {
            "runtime": "wamr",
            "phase": "measure",
            "index": 1,
            "tcgc_response_bytes": 58187,
            "output": output,
        }
        drift = dict(base, runtime="wasmtime", tcgc_response_bytes=58188)
        with self.assertRaisesRegex(bench.HarnessError, "response-size drift"):
            bench.verify_equivalence([base, drift])

        changed_output = dict(output, total_bytes=5)
        drift = dict(base, runtime="wasmtime", output=changed_output)
        with self.assertRaisesRegex(bench.HarnessError, "generated output mismatch"):
            bench.verify_equivalence([base, drift])

    def test_missing_prerequisite_is_actionable(self) -> None:
        config = bench.load_manifest(self.manifest)
        self.tool.unlink()
        with self.assertRaisesRegex(
            bench.HarnessError, "required wamr binary is absent"
        ):
            bench.validate_config(config)

    def test_profile_mode_never_falls_back_when_unsupported(self) -> None:
        config = bench.load_manifest(self.manifest)
        with mock.patch.object(bench.platform, "system", return_value="Darwin"):
            with self.assertRaisesRegex(bench.HarnessError, "requires Linux x86_64"):
                bench.run_perf(
                    config,
                    self.scratch,
                    self.scratch / "manifest.cwasm.json",
                    self.scratch / "wasmtime.cwasm",
                    10,
                    {},
                )

    def test_attribution_json_reports_sample_coverage(self) -> None:
        attr = _load_attr_module()
        cwasm = self.scratch / "core.cwasm"
        text = b"\x90" * 8
        functions = struct.pack("<III", 1, 0, 0)
        cwasm.write_bytes(
            struct.pack("<II", attr.AOT_MAGIC, attr.AOT_VERSION)
            + struct.pack("<II", attr.SEC_TEXT, len(text))
            + text
            + struct.pack("<II", attr.SEC_FUNCTION, len(functions))
            + functions
        )
        perf = self.scratch / "perf.data"
        perf.write_bytes(b"controlled")
        output = self.scratch / "attribution.json"
        base = 0x100000
        argv = [
            "aot_jit_attr.py",
            "--perf",
            str(perf),
            "--cwasm",
            str(cwasm),
            "--json-out",
            str(output),
            "--min-samples",
            "5",
            "--require-size-match",
        ]
        with mock.patch.object(attr, "addr_counts", return_value=({base: 10}, 10)), (
            mock.patch.object(attr, "jit_exec_mmaps", return_value=[(base, 8)])
        ), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
            io.StringIO()
        ):
            attr.main()
        report = json.loads(output.read_text(encoding="UTF-8"))
        self.assertEqual(report["total_samples"], 10)
        self.assertEqual(report["attributed_samples"], 10)
        self.assertEqual(report["attribution_coverage_pct"], 100)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and platform.machine() == "x86_64",
        "controlled timing cohort requires Linux x86_64",
    )
    def test_controlled_end_to_end_smoke(self) -> None:
        self.tool.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            "case \"${1:-}\" in\n"
            "  version|--version) echo pinned-tool-1.0; exit 0 ;;\n"
            "  compile-component)\n"
            "    while [ \"$#\" -gt 0 ]; do\n"
            "      if [ \"$1\" = -o ]; then shift; out=\"$1\"; break; fi\n"
            "      shift\n"
            "    done\n"
            "    printf '{\"modules\":[]}' > \"$out\"\n"
            "    stem=${out%.cwasm.json}\n"
            "    printf core > \"${stem}.4.cwasm\"\n"
            "    exit 0 ;;\n"
            "  compile)\n"
            "    while [ \"$#\" -gt 0 ]; do\n"
            "      if [ \"$1\" = -o ]; then shift; out=\"$1\"; break; fi\n"
            "      shift\n"
            "    done\n"
            "    printf wasmtime > \"$out\"\n"
            "    exit 0 ;;\n"
            "  run)\n"
            "    out=''\n"
            "    while [ \"$#\" -gt 0 ]; do\n"
            "      case \"$1\" in\n"
            "        *::/out) out=${1%::/out} ;;\n"
            "      esac\n"
            "      shift\n"
            "    done\n"
            "    mkdir -p \"$out\"\n"
            "    printf 'pub const generated = true;\\n' > \"$out/generated.zig\"\n"
            "    echo 'got 58187 bytes back from tcgc.compile'\n"
            "    exit 0 ;;\n"
            "esac\n"
            "exit 3\n",
            encoding="UTF-8",
        )
        self.tool.chmod(self.tool.stat().st_mode | stat.S_IXUSR)
        document = self._manifest_document()
        current_sha = bench.sha256_file(self.tool)
        for tool in document["tools"].values():
            tool["sha256"] = current_sha
        self.manifest.write_text(json.dumps(document), encoding="UTF-8")
        report_json = self.scratch / "report.json"
        report_markdown = self.scratch / "report.md"
        with contextlib.redirect_stderr(io.StringIO()):
            report = bench.execute(
                argparse.Namespace(
                    manifest=self.manifest,
                    work_dir=self.scratch / "work",
                    warmups=1,
                    runs=5,
                    compile_timeout=10.0,
                    run_timeout=10.0,
                    profile=False,
                    report_json=report_json,
                    report_markdown=report_markdown,
                )
            )
        self.assertEqual(report["timing"]["wamr"]["runs"], 5)
        self.assertEqual(report["timing"]["wasmtime"]["runs"], 5)
        self.assertTrue(report["equivalence"]["cross_runtime_exact_match"])
        self.assertTrue(report_json.is_file())
        self.assertIn(
            "Wasmtime-time/WAMR-time ratio", report_markdown.read_text()
        )


if __name__ == "__main__":
    unittest.main()
