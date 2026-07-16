#!/usr/bin/env python3
"""Collect reproducible WASI P3 phase timings and macOS sample profiles."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_FIXTURES = 41
ROOT = Path(__file__).resolve().parent.parent
SUITE = (
    ROOT
    / "tests"
    / "wasi-testsuite"
    / "tests"
    / "rust"
    / "testsuite"
    / "wasm32-wasip3"
)
UNFILTERED = ROOT / "tests" / "wasi-p3-unfiltered.py"
PHASES = {
    "component_precompile",
    "core_precompile",
    "fixture_execution",
}


class ProfileDataError(ValueError):
    """Raised when a timing artifact violates the versioned contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileDataError(message)


def validate_event(event: dict[str, Any]) -> None:
    _require(event.get("schema_version") == SCHEMA_VERSION, "event schema_version")
    _require(event.get("event") == "phase_timing", "event kind")
    _require(event.get("mode") in {"aot", "jit"}, "event mode")
    _require(isinstance(event.get("fixture"), str) and event["fixture"], "event fixture")
    _require(event.get("phase") in PHASES, "event phase")
    _require(event.get("artifact_kind") in {"component", "core"}, "artifact kind")
    _require(event.get("cache") in {"miss", "hit", "bypass", "n/a"}, "event cache")
    _require(
        isinstance(event.get("duration_ns"), int) and event["duration_ns"] >= 0,
        "event duration_ns",
    )


def validate_run_document(document: dict[str, Any]) -> None:
    _require(document.get("schema_version") == SCHEMA_VERSION, "run schema_version")
    _require(document.get("kind") == "wasi-p3-profile-runs", "run kind")
    metadata = document.get("metadata")
    _require(isinstance(metadata, dict), "run metadata")
    _require(metadata.get("mode") in {"aot", "jit"}, "metadata mode")
    _require(isinstance(metadata.get("platform_id"), str), "metadata platform_id")
    _require(isinstance(metadata.get("commit"), str), "metadata commit")
    samples = document.get("samples")
    _require(isinstance(samples, list), "samples")
    for sample in samples:
        _require(isinstance(sample, dict), "sample object")
        _require(sample.get("temperature") in {"cold", "warm"}, "sample temperature")
        _require(isinstance(sample.get("index"), int), "sample index")
        _require(isinstance(sample.get("valid"), bool), "sample valid")
        _require(isinstance(sample.get("suite_duration_ns"), int), "suite duration")
        counts = sample.get("counts")
        _require(isinstance(counts, dict), "sample counts")
        for name in ("fixtures", "executed", "passed", "failed"):
            _require(isinstance(counts.get(name), int), f"counts.{name}")
        events = sample.get("events")
        _require(isinstance(events, list), "sample events")
        for event in events:
            _require(isinstance(event, dict), "event object")
            validate_event(event)


def _command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else f"exit {result.returncode}"


def collect_metadata(mode: str, platform_id: str) -> dict[str, Any]:
    wamr = shlex.split(os.getenv("WAMR", str(ROOT / "zig-out" / "bin" / "wamr")))
    wamrc = shlex.split(os.getenv("WAMRC", str(ROOT / "zig-out" / "bin" / "wamrc")))
    return {
        "platform_id": platform_id,
        "mode": mode,
        "optimize": "ReleaseSafe",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "runner_name": os.getenv("RUNNER_NAME", ""),
            "runner_image": os.getenv("ImageOS", ""),
            "runner_os": os.getenv("RUNNER_OS", ""),
            "runner_arch": os.getenv("RUNNER_ARCH", ""),
        },
        "tools": {
            "zig": _command_version(["zig", "version"]),
            "wamr": _command_version(wamr + ["version"]),
            "wamrc": _command_version(wamrc + ["version"]),
        },
        "cache": {
            "zig_global_cache_dir": os.getenv("ZIG_GLOBAL_CACHE_DIR", ""),
            "zig_local_cache_dir": os.getenv("ZIG_LOCAL_CACHE_DIR", ""),
            "aot_sidecars": "mtime sibling cache",
            "jit": "in-process; no persistent compiled-artifact cache",
        },
    }


def _remove_sidecars() -> None:
    for pattern in ("*.cwasm", "*.cwasm.json"):
        for sidecar in SUITE.glob(pattern):
            sidecar.unlink()


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="UTF-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProfileDataError(f"{path}:{line_number}: {exc}") from exc
            _require(isinstance(event, dict), f"{path}:{line_number}: event object")
            validate_event(event)
            events.append(event)
    return events


def _report_counts(report: Path) -> dict[str, int]:
    data = json.loads(report.read_text(encoding="UTF-8"))
    suites = data.get("results", [])
    _require(len(suites) == 1, f"{report}: expected one suite")
    tests = suites[0].get("tests", [])
    executed = [test for test in tests if test.get("executed")]
    passed = [test for test in executed if not test.get("failures")]
    return {
        "fixtures": len(tests),
        "executed": len(executed),
        "passed": len(passed),
        "failed": len(executed) - len(passed),
    }


def _sample_is_valid(
    mode: str,
    temperature: str,
    returncode: int,
    counts: dict[str, int],
    events: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if returncode:
        errors.append(f"runner exit {returncode}")
    if counts != {
        "fixtures": EXPECTED_FIXTURES,
        "executed": EXPECTED_FIXTURES,
        "passed": EXPECTED_FIXTURES,
        "failed": 0,
    }:
        errors.append(f"conformance counts {counts}, expected 41/41")

    executions = [e for e in events if e["phase"] == "fixture_execution"]
    precompiles = [e for e in events if e["phase"].endswith("_precompile")]
    if len(executions) != EXPECTED_FIXTURES:
        errors.append(f"execution events={len(executions)}, expected 41")
    if len(precompiles) != EXPECTED_FIXTURES:
        errors.append(f"precompile events={len(precompiles)}, expected 41")
    if len({e["fixture"] for e in executions}) != EXPECTED_FIXTURES:
        errors.append("execution fixture names are not unique")
    expected_cache = (
        "bypass" if mode == "jit" else "miss" if temperature == "cold" else "hit"
    )
    wrong_cache = [e["fixture"] for e in precompiles if e["cache"] != expected_cache]
    if wrong_cache:
        errors.append(
            f"{len(wrong_cache)} precompile events did not use cache={expected_cache}"
        )
    return not errors, errors


def _run_sample(
    mode: str,
    temperature: str,
    index: int,
    output_dir: Path,
) -> dict[str, Any]:
    run_id = f"{mode}-{temperature}-{index:02d}"
    raw_dir = output_dir / "raw"
    report_dir = output_dir / "reports"
    log_dir = output_dir / "logs"
    for directory in (raw_dir, report_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    timing = (raw_dir / f"{run_id}.jsonl").resolve()
    report = (report_dir / f"{run_id}.json").resolve()
    log = log_dir / f"{run_id}.log"
    timing.unlink(missing_ok=True)
    report.unlink(missing_ok=True)

    if mode == "aot" and temperature == "cold":
        _remove_sidecars()

    env = os.environ.copy()
    env.update(
        {
            "WAMR_PROFILE_TIMINGS": str(timing),
            "WAMR_PROFILE_RUN_ID": run_id,
            "WAMR_PROFILE_MODE": mode,
            "WAMR_P3_REPORT": str(report),
        }
    )
    if mode == "jit":
        env["WAMR_JIT_TESTSUITE"] = "1"
    else:
        env.pop("WAMR_JIT_TESTSUITE", None)

    started_ns = time.perf_counter_ns()
    with log.open("w", encoding="UTF-8") as output:
        proc = subprocess.run(
            [sys.executable, str(UNFILTERED)],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    suite_duration_ns = time.perf_counter_ns() - started_ns

    counts = (
        _report_counts(report)
        if report.is_file()
        else {"fixtures": 0, "executed": 0, "passed": 0, "failed": 0}
    )
    events = parse_jsonl(timing) if timing.is_file() else []
    valid, errors = _sample_is_valid(
        mode, temperature, proc.returncode, counts, events
    )
    print(
        f"{run_id}: valid={valid} passed={counts['passed']}/"
        f"{EXPECTED_FIXTURES} elapsed={suite_duration_ns / 1e9:.3f}s",
        flush=True,
    )
    return {
        "id": run_id,
        "temperature": temperature,
        "index": index,
        "valid": valid,
        "errors": errors,
        "returncode": proc.returncode,
        "suite_duration_ns": suite_duration_ns,
        "counts": counts,
        "events": events,
        "raw_timing": str(timing.relative_to(output_dir.resolve())),
        "report": str(report.relative_to(output_dir.resolve())),
        "log": str(log.resolve().relative_to(output_dir.resolve())),
    }


def run_collection(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-p3-profile-runs",
        "metadata": collect_metadata(args.mode, args.platform_id),
        "plan": {
            "cold_samples": args.cold_samples,
            "warm_samples": args.warm_samples,
            "optimize": "ReleaseSafe",
        },
        "samples": [],
    }
    destination = output_dir / "samples.json"
    for temperature, count in (
        ("cold", args.cold_samples),
        ("warm", args.warm_samples),
    ):
        for index in range(1, count + 1):
            document["samples"].append(
                _run_sample(args.mode, temperature, index, output_dir)
            )
            destination.write_text(
                json.dumps(document, indent=2) + "\n", encoding="UTF-8"
            )
    validate_run_document(document)
    return 0 if all(sample["valid"] for sample in document["samples"]) else 1


def run_profile(args: argparse.Namespace) -> int:
    selection = json.loads(args.selection.read_text(encoding="UTF-8"))
    requested = [
        item for item in selection.get("profiles", []) if item.get("mode") == args.mode
    ]
    if not requested:
        print(f"No {args.mode} profiles selected.")
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_sidecars()
    timing = output_dir / f"profile-{args.mode}.jsonl"
    report = output_dir / f"profile-{args.mode}-report.json"
    log = output_dir / f"profile-{args.mode}.log"
    env = os.environ.copy()
    env.update(
        {
            "WAMR_PROFILE_TIMINGS": str(timing),
            "WAMR_PROFILE_RUN_ID": f"profile-{args.mode}",
            "WAMR_PROFILE_MODE": args.mode,
            "WAMR_PROFILE_SELECTION": str(args.selection.resolve()),
            "WAMR_PROFILE_OUTPUT_DIR": str(output_dir),
            "WAMR_P3_REPORT": str(report),
            "WAMR_TESTSUITE_TIMEOUT": "300",
        }
    )
    if args.mode == "jit":
        env["WAMR_JIT_TESTSUITE"] = "1"
    else:
        env.pop("WAMR_JIT_TESTSUITE", None)
    with log.open("w", encoding="UTF-8") as output:
        proc = subprocess.run(
            [sys.executable, str(UNFILTERED)],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )

    missing = []
    for item in requested:
        profile = (
            output_dir
            / f"{item['fixture']}-{item['phase']}-{args.mode}.sample.txt"
        )
        if not profile.is_file() or profile.stat().st_size == 0:
            missing.append(profile.name)
    if proc.returncode or missing:
        print(
            f"profile {args.mode} failed: runner={proc.returncode}, missing={missing}",
            file=sys.stderr,
        )
        return 1
    print(f"Captured {len(requested)} {args.mode} sampling profile(s).")
    return 0


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("run", help="collect cold/warm suite samples")
    collect.add_argument("--mode", choices=("aot", "jit"), required=True)
    collect.add_argument("--platform-id", required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--cold-samples", type=_positive_int, default=5)
    collect.add_argument("--warm-samples", type=_positive_int, default=10)
    collect.set_defaults(func=run_collection)

    profile = subparsers.add_parser("profile", help="capture selected macOS samples")
    profile.add_argument("--mode", choices=("aot", "jit"), required=True)
    profile.add_argument("--selection", type=Path, required=True)
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.set_defaults(func=run_profile)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (OSError, ProfileDataError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
