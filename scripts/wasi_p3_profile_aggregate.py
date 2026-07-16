#!/usr/bin/env python3
"""Validate and aggregate matched macOS/Linux ARM64 WASI P3 profiles."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from wasi_p3_profile import (
    EXPECTED_FIXTURES,
    PHASES,
    ProfileDataError,
    SCHEMA_VERSION,
    validate_run_document,
)


EXPECTED_PLATFORMS = {"macos-arm64", "linux-arm64"}
EXPECTED_MODES = {"aot", "jit"}
EXPECTED_SAMPLE_COUNTS = {"cold": 5, "warm": 10}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileDataError(message)


def _percentile(values: Iterable[int], percentile: float) -> int:
    ordered = sorted(values)
    _require(bool(ordered), "cannot summarize an empty sample")
    index = min(len(ordered) - 1, max(0, int(len(ordered) * percentile + 0.999) - 1))
    return ordered[index]


def _stats(values: list[int]) -> dict[str, int]:
    return {
        "samples": len(values),
        "min_ns": min(values),
        "median_ns": int(statistics.median(values)),
        "p95_ns": _percentile(values, 0.95),
        "max_ns": max(values),
    }


def _load_documents(input_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    documents: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(input_dir.rglob("samples.json")):
        document = json.loads(path.read_text(encoding="UTF-8"))
        validate_run_document(document)
        documents.append((path, document))
    _require(bool(documents), f"no samples.json documents under {input_dir}")
    return documents


def _validate_plan(documents: list[tuple[Path, dict[str, Any]]]) -> str:
    keys: set[tuple[str, str]] = set()
    commits: set[str] = set()
    for path, document in documents:
        metadata = document["metadata"]
        platform_id = metadata["platform_id"]
        mode = metadata["mode"]
        _require(platform_id in EXPECTED_PLATFORMS, f"{path}: platform {platform_id}")
        _require(mode in EXPECTED_MODES, f"{path}: mode {mode}")
        _require(metadata.get("optimize") == "ReleaseSafe", f"{path}: optimize mode")
        host = metadata.get("host")
        tools = metadata.get("tools")
        cache = metadata.get("cache")
        _require(isinstance(host, dict), f"{path}: host metadata")
        _require(isinstance(tools, dict), f"{path}: tool metadata")
        _require(isinstance(cache, dict), f"{path}: cache metadata")
        machine = str(host.get("machine", "")).lower()
        _require(machine in {"arm64", "aarch64"}, f"{path}: non-ARM64 host {machine}")
        expected_system = "Darwin" if platform_id == "macos-arm64" else "Linux"
        _require(host.get("system") == expected_system, f"{path}: host system")
        _require(
            all(tools.get(name) for name in ("zig", "wamr", "wamrc")),
            f"{path}: tools",
        )
        _require((platform_id, mode) not in keys, f"duplicate {platform_id}/{mode}")
        keys.add((platform_id, mode))
        commits.add(metadata["commit"])
        samples = document["samples"]
        for temperature, expected in EXPECTED_SAMPLE_COUNTS.items():
            selected = [s for s in samples if s["temperature"] == temperature]
            _require(
                len(selected) == expected,
                f"{path}: {temperature} samples={len(selected)}, expected={expected}",
            )
        _require(all(s["valid"] for s in samples), f"{path}: invalid suite sample")
        for sample in samples:
            _require(
                sample["counts"]
                == {
                    "fixtures": EXPECTED_FIXTURES,
                    "executed": EXPECTED_FIXTURES,
                    "passed": EXPECTED_FIXTURES,
                    "failed": 0,
                },
                f"{path}: {sample['id']} did not pass 41/41",
            )
    expected_keys = {
        (platform_id, mode)
        for platform_id in EXPECTED_PLATFORMS
        for mode in EXPECTED_MODES
    }
    _require(keys == expected_keys, f"platform/mode set {keys}, expected {expected_keys}")
    _require(len(commits) == 1, f"measurements use different revisions: {commits}")
    return next(iter(commits))


def _load_microbench(
    documents: list[tuple[Path, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path, document in documents:
        platform_id = document["metadata"]["platform_id"]
        if platform_id in reports:
            continue
        candidate = path.parent.parent / "microbench" / "report.json"
        _require(candidate.is_file(), f"missing {platform_id} microbench {candidate}")
        report = json.loads(candidate.read_text(encoding="UTF-8"))
        scenarios = report.get("scenarios")
        _require(isinstance(scenarios, list) and len(scenarios) == 4, candidate.as_posix())
        _require(
            all(item.get("verdict") == "no-budget" for item in scenarios),
            f"{candidate}: budget enforcement was not disabled",
        )
        reports[platform_id] = report
    _require(set(reports) == EXPECTED_PLATFORMS, "microbench platform set")
    return reports


def _suite_rows(
    documents: list[tuple[Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for _, document in documents:
        metadata = document["metadata"]
        for sample in document["samples"]:
            grouped[
                (metadata["platform_id"], metadata["mode"], sample["temperature"])
            ].append(sample["suite_duration_ns"])
    return [
        {
            "platform_id": key[0],
            "mode": key[1],
            "temperature": key[2],
            **_stats(values),
        }
        for key, values in sorted(grouped.items())
    ]


def _phase_values(
    documents: list[tuple[Path, dict[str, Any]]]
) -> tuple[
    dict[tuple[str, str, str, str, str], list[int]],
    dict[tuple[str, str, str, int, str, str], int],
]:
    grouped: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    paired: dict[tuple[str, str, str, int, str, str], int] = {}
    for _, document in documents:
        metadata = document["metadata"]
        for sample in document["samples"]:
            for event in sample["events"]:
                grouped[
                    (
                        metadata["platform_id"],
                        metadata["mode"],
                        sample["temperature"],
                        event["phase"],
                        event["fixture"],
                    )
                ].append(event["duration_ns"])
                paired[
                    (
                        metadata["platform_id"],
                        metadata["mode"],
                        sample["temperature"],
                        sample["index"],
                        event["phase"],
                        event["fixture"],
                    )
                ] = event["duration_ns"]
    return grouped, paired


def _phase_rows(
    grouped: dict[tuple[str, str, str, str, str], list[int]]
) -> list[dict[str, Any]]:
    return [
        {
            "platform_id": key[0],
            "mode": key[1],
            "temperature": key[2],
            "phase": key[3],
            "fixture": key[4],
            **_stats(values),
        }
        for key, values in sorted(grouped.items())
    ]


def _profile_selection(
    grouped: dict[tuple[str, str, str, str, str], list[int]],
    paired: dict[tuple[str, str, str, int, str, str], int],
    suites: list[dict[str, Any]],
) -> dict[str, Any]:
    selected: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for mode in EXPECTED_MODES:
        selected[(mode, "http-fields", "fixture_execution")].add(
            "required http-fields baseline"
        )

    suite_medians = {
        (row["platform_id"], row["mode"], row["temperature"]): row["median_ns"]
        for row in suites
    }
    for key, values in grouped.items():
        platform_id, mode, temperature, phase, fixture = key
        if platform_id != "macos-arm64":
            continue
        phase_median = statistics.median(values)
        suite_median = suite_medians[(platform_id, mode, temperature)]
        share = phase_median / suite_median if suite_median else 0.0
        if share >= 0.20:
            selected[(mode, fixture, phase)].add(
                f"{temperature} median is {share:.1%} of suite wall time"
            )

    candidates = {
        (mode, phase, fixture)
        for platform_id, mode, _, phase, fixture in grouped
        if platform_id == "macos-arm64"
    }
    for mode, phase, fixture in candidates:
        ratios: list[float] = []
        temperatures: set[str] = set()
        for temperature, count in EXPECTED_SAMPLE_COUNTS.items():
            cell: list[float] = []
            for index in range(1, count + 1):
                mac = paired.get(
                    ("macos-arm64", mode, temperature, index, phase, fixture)
                )
                linux = paired.get(
                    ("linux-arm64", mode, temperature, index, phase, fixture)
                )
                if mac is not None and linux is not None and linux > 0:
                    cell.append(mac / linux)
            if cell:
                temperatures.add(temperature)
                ratios.extend(cell)
        if not ratios or temperatures != {"cold", "warm"}:
            continue
        fraction = sum(ratio >= 2.0 for ratio in ratios) / len(ratios)
        cell_medians = []
        for temperature in ("cold", "warm"):
            mac_values = grouped[
                ("macos-arm64", mode, temperature, phase, fixture)
            ]
            linux_values = grouped[
                ("linux-arm64", mode, temperature, phase, fixture)
            ]
            linux_median = statistics.median(linux_values)
            if linux_median <= 0:
                break
            cell_medians.append(statistics.median(mac_values) / linux_median)
        if (
            len(cell_medians) == 2
            and min(cell_medians) >= 2.0
            and fraction >= 0.80
        ):
            selected[(mode, fixture, phase)].add(
                "macOS/Linux median >=2x in cold and warm cells "
                f"({fraction:.0%} paired samples >=2x)"
            )

    profiles = [
        {
            "mode": mode,
            "fixture": fixture,
            "phase": phase,
            "reasons": sorted(reasons),
        }
        for (mode, fixture, phase), reasons in sorted(selected.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-p3-profile-selection",
        "profiles": profiles,
    }


def validate_aggregate_document(document: dict[str, Any]) -> None:
    _require(document.get("schema_version") == SCHEMA_VERSION, "aggregate schema")
    _require(document.get("kind") == "wasi-p3-profile-aggregate", "aggregate kind")
    _require(isinstance(document.get("commit"), str), "aggregate commit")
    suite = document.get("suite")
    _require(isinstance(suite, list) and len(suite) == 8, "aggregate suite rows")
    phases = document.get("phases")
    _require(isinstance(phases, list) and phases, "aggregate phase rows")
    for row in phases:
        _require(row.get("phase") in PHASES, "aggregate phase")
        _require(isinstance(row.get("median_ns"), int), "aggregate phase median")


def _markdown(
    document: dict[str, Any],
    selection: dict[str, Any],
    microbench: dict[str, dict[str, Any]],
) -> str:
    date = datetime.now(timezone.utc).date().isoformat()
    lines = [
        f"# WASI P3 macOS ARM64 performance investigation — {date}",
        "",
        f"Revision: `{document['commit']}`.",
        "",
        "All 60 measured suite runs were valid unfiltered ReleaseSafe runs: "
        "5 cold + 10 warm for AOT and JIT on each native ARM64 platform, "
        "with every sample passing 41/41.",
        "",
        "## Suite wall time",
        "",
        "| platform | mode | cache state | samples | median ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in document["suite"]:
        lines.append(
            f"| {row['platform_id']} | {row['mode']} | {row['temperature']} | "
            f"{row['samples']} | {row['median_ns'] / 1e6:.3f} | "
            f"{row['p95_ns'] / 1e6:.3f} |"
        )
    lines += [
        "",
        "## Sampling-profile selection",
        "",
        "| mode | fixture | phase | reason |",
        "|---|---|---|---|",
    ]
    for item in selection["profiles"]:
        lines.append(
            f"| {item['mode']} | {item['fixture']} | {item['phase']} | "
            f"{'; '.join(item['reasons'])} |"
        )
    lines += [
        "",
        "## WASI microbenchmark (budget disabled)",
        "",
        "| platform | scenario | median ns | p95 ns |",
        "|---|---|---:|---:|",
    ]
    for platform_id in sorted(microbench):
        for scenario in microbench[platform_id]["scenarios"]:
            lines.append(
                f"| {platform_id} | {scenario['name']} | "
                f"{scenario['median_ns']} | {scenario['p95_ns']} |"
            )
    lines += [
        "",
        "The existing Linux x86_64 budgets were not applied or changed. This "
        "manual workflow is non-blocking and does not alter normal conformance gates.",
        "",
    ]
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> int:
    documents = _load_documents(args.input_dir)
    commit = _validate_plan(documents)
    microbench = _load_microbench(documents)
    suites = _suite_rows(documents)
    grouped, paired = _phase_values(documents)
    phases = _phase_rows(grouped)
    selection = _profile_selection(grouped, paired, suites)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-p3-profile-aggregate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "valid": True,
        "contract": {
            "platforms": sorted(EXPECTED_PLATFORMS),
            "modes": sorted(EXPECTED_MODES),
            "cold_samples_per_platform_mode": 5,
            "warm_samples_per_platform_mode": 10,
            "fixtures_per_sample": EXPECTED_FIXTURES,
        },
        "suite": suites,
        "phases": phases,
        "microbench": microbench,
    }
    validate_aggregate_document(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="UTF-8"
    )
    (args.output_dir / "profile-selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="UTF-8"
    )
    (args.output_dir / "report.md").write_text(
        _markdown(result, selection, microbench), encoding="UTF-8"
    )
    print(
        f"Validated 60 41/41 suite samples at {commit}; "
        f"selected {len(selection['profiles'])} profiles."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return aggregate(args)
    except (OSError, json.JSONDecodeError, ProfileDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
