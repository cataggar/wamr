"""Existing v2 host validation over real native guest-boundary executions.

The test executable supplies explicitly synthetic metadata and CHECK transport.
Nothing here qualifies or fabricates a deployable image receipt.
"""
import json
import os
import shlex
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_benchmark


@unittest.skipUnless(os.environ.get("WAMR_NATIVE_GUEST_TEST"), "requires native guest fixture")
class NativeGuestProtocolTests(unittest.TestCase):
    def test_actual_native_payloads_match_existing_v2_validator(self):
        command = shlex.split(os.environ.get("WAMR_NATIVE_GUEST_TEST_RUNNER", ""))
        command.append(os.environ["WAMR_NATIVE_GUEST_TEST"])
        completed = subprocess.run(command, capture_output=True, check=True, timeout=120)
        lines = completed.stdout.decode("utf-8").splitlines()
        self.assertEqual(len(lines), 12)
        results = []
        for index in range(0, len(lines), 2):
            self.assertTrue(lines[index].startswith("WAMR_NATIVE_TEST_MANIFEST="))
            self.assertTrue(lines[index + 1].startswith("WAMR_NATIVE_CHECK_RESULT="))
            manifest = json.loads(lines[index].split("=", 1)[1])
            result = json.loads(lines[index + 1].split("=", 1)[1])
            self.assertIn("synthetic", manifest["campaign_id"])
            for field in ("created_at", "expires_at"):
                stamp = manifest[field]
                self.assertEqual(datetime.fromisoformat(stamp).isoformat(), stamp,
                                 "fixture timestamps must use the host's canonical UTC offset, not Python-3.11-only Z parsing")
            native_benchmark.validate_result(result, manifest, "run-0001")
            with self.assertRaises(ValueError):
                native_benchmark.parse_result_stream(lines[index + 1].encode())
            results.append(result)
        self.assertEqual([r["outcome"] for r in results],
                         ["success", "error", "error", "error", "error", "error"])
        self.assertEqual([len(r["invocations"]) for r in results], [3, 1, 1, 1, 2, 1])
        self.assertEqual(results[1]["phases"]["first_invocation_ticks"], None)
        self.assertEqual(results[2]["reset_events"][0]["outcome"], "error")
        self.assertEqual(results[3]["reset_events"][0]["outcome"], "completed")
        self.assertEqual(results[4]["phases"]["steady_state_ticks"], [None])
        self.assertEqual(results[5]["exit_code"], 0xffffffff)
        self.assertEqual(results[5]["invocations"][0]["outcome"], "proc_exit")
        self.assertEqual(results[5]["invocations"][0]["stdout_base64"], "AP/DKAoA")
        self.assertIn(b"WAMR_NATIVE_INVOCATION=", completed.stderr)
        self.assertIn(b"method=caller-requested-bytes live=0", completed.stderr)
