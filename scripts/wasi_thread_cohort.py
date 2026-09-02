#!/usr/bin/env python3
"""Dispatch or validate an immutable WASI thread benchmark cohort."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import (
    BenchmarkDataError,
    SCHEMA_VERSION,
    atomic_write_json,
    collected_at,
)
from bench_wasi_threads import CANONICAL_PLATFORMS, HarnessError, validate_report


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_PLATFORMS = tuple(CANONICAL_PLATFORMS)


def require_sha(value: str) -> str:
    normalized = value.lower()
    if SHA_RE.fullmatch(normalized) is None:
        raise HarnessError("target SHA must be an immutable 40-character hex commit")
    return normalized


def gh_json(command: list[str]) -> Any:
    output = subprocess.check_output(
        ["gh", *command], text=True, stderr=subprocess.STDOUT
    )
    return json.loads(output)


def dispatch(args: argparse.Namespace) -> int:
    target_sha = require_sha(args.target_sha)
    if args.runs <= 0 or args.max_in_flight <= 0:
        raise HarnessError("--runs and --max-in-flight must be positive")
    state = {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-thread-cohort-dispatch",
        "created_at": collected_at(),
        "repository": args.repository,
        "workflow": args.workflow,
        "workflow_ref": target_sha,
        "target_sha": target_sha,
        "profile": "authoritative",
        "warmups": 2,
        "samples": 10,
        "requested_runs": args.runs,
        "max_in_flight": args.max_in_flight,
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    active: dict[int, dict[str, Any]] = {}
    launched = 0
    while launched < args.runs or active:
        while launched < args.runs and len(active) < args.max_in_flight:
            output = subprocess.check_output(
                [
                    "gh",
                    "workflow",
                    "run",
                    args.workflow,
                    "--repo",
                    args.repository,
                    "--ref",
                    target_sha,
                    "-f",
                    f"target_sha={target_sha}",
                    "-f",
                    "profile=authoritative",
                    "-f",
                    "warmups=2",
                    "-f",
                    "samples=10",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            match = re.search(r"/actions/runs/(\d+)", output)
            if match is None:
                raise HarnessError(f"could not parse workflow run URL: {output!r}")
            run_id = int(match.group(1))
            record = {
                "sequence": launched + 1,
                "run_id": run_id,
                "url": output,
                "status": "queued",
                "conclusion": "",
                "artifacts": [],
            }
            state["runs"].append(record)
            active[run_id] = record
            launched += 1
            atomic_write_json(args.output, state)

        time.sleep(args.poll_seconds)
        for run_id in list(active):
            record = active[run_id]
            run = gh_json(
                [
                    "run",
                    "view",
                    str(run_id),
                    "--repo",
                    args.repository,
                    "--json",
                    "status,conclusion,headSha,url",
                ]
            )
            record.update(
                {
                    "status": run["status"],
                    "conclusion": run["conclusion"],
                    "workflow_head_sha": run["headSha"],
                    "url": run["url"],
                }
            )
            if str(run["headSha"]).lower() != target_sha:
                atomic_write_json(args.output, state)
                raise HarnessError(
                    f"workflow run {run_id} head SHA {run['headSha']!r} "
                    f"does not match target {target_sha}"
                )
            if run["status"] == "completed":
                artifacts = gh_json(
                    [
                        "api",
                        f"repos/{args.repository}/actions/runs/{run_id}/artifacts",
                    ]
                )
                record["artifacts"] = [
                    {
                        "id": item["id"],
                        "name": item["name"],
                        "size_in_bytes": item["size_in_bytes"],
                        "expired": item["expired"],
                    }
                    for item in artifacts["artifacts"]
                ]
                if run["conclusion"] != "success":
                    atomic_write_json(args.output, state)
                    raise HarnessError(
                        f"workflow run {run_id} concluded {run['conclusion']}"
                    )
                del active[run_id]
        atomic_write_json(args.output, state)
    print(f"Recorded {args.runs} successful immutable dispatches in {args.output}")
    return 0


def validate_documents(
    documents: list[tuple[Path, dict[str, Any]]],
    required_platforms: tuple[str, ...],
    minimum_reports: int,
) -> dict[str, Any]:
    if minimum_reports < 1:
        raise HarnessError("minimum report count must be positive")
    if (
        len(required_platforms) != len(DEFAULT_PLATFORMS)
        or len(set(required_platforms)) != len(required_platforms)
        or set(required_platforms) != set(DEFAULT_PLATFORMS)
    ):
        raise HarnessError("cohort platforms must be exactly the canonical hosted set")
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    identities: set[tuple[str, str, str, str, str]] = set()
    for path, document in documents:
        validate_report(document)
        metadata = document["metadata"]
        platform_id = metadata["platform_id"]
        if platform_id not in required_platforms:
            raise HarnessError(f"{path}: unknown platform {platform_id!r}")
        expected_host = CANONICAL_PLATFORMS[platform_id]
        host = metadata["host"]
        if (host["system"], host["machine"]) != expected_host:
            raise HarnessError(
                f"{path}: platform ID does not match canonical host identity"
            )
        if document["plan"]["profile"] != "authoritative":
            raise HarnessError(f"{path}: cohort report is not authoritative")
        identities.add(
            (
                metadata["commit"],
                metadata["build_source_sha256"],
                metadata["fixture_set_sha256"],
                metadata["plan_sha256"],
                document["plan"]["profile"],
            )
        )
        grouped[platform_id].append((path, document))
    if len(identities) != 1:
        raise HarnessError("cohort reports have mixed commit/source/fixture/plan identity")
    if set(grouped) != set(required_platforms):
        raise HarnessError("cohort platform set is incomplete")
    run_ids_by_platform: dict[str, set[str]] = {}
    for platform_id in required_platforms:
        selected = grouped[platform_id]
        if len(selected) < minimum_reports:
            raise HarnessError(
                f"{platform_id}: {len(selected)} reports, need {minimum_reports}"
            )
        run_ids = [
            item["metadata"]["host"].get("github_run_id", "")
            for _, item in selected
        ]
        if any(not run_id for run_id in run_ids) or len(set(run_ids)) != len(run_ids):
            raise HarnessError(f"{platform_id}: workflow run IDs are missing or duplicate")
        run_ids_by_platform[platform_id] = set(run_ids)
    if len({len(items) for items in grouped.values()}) != 1:
        raise HarnessError("cohort platforms have different report counts")
    if len({frozenset(items) for items in run_ids_by_platform.values()}) != 1:
        raise HarnessError("cohort platforms do not contain the same workflow runs")
    identity = next(iter(identities))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-thread-cohort",
        "validated_at": collected_at(),
        "identity": {
            "commit": identity[0],
            "build_source_sha256": identity[1],
            "fixture_set_sha256": identity[2],
            "plan_sha256": identity[3],
            "profile": identity[4],
        },
        "platforms": {
            platform_id: {
                "reports": len(grouped[platform_id]),
                "run_ids": sorted(
                    document["metadata"]["host"]["github_run_id"]
                    for _, document in grouped[platform_id]
                ),
                "paths": sorted(str(path) for path, _ in grouped[platform_id]),
            }
            for platform_id in required_platforms
        },
    }


def validate_cohort(args: argparse.Namespace) -> int:
    documents = []
    for path in sorted(args.input_dir.rglob("report.json")):
        documents.append((path, json.loads(path.read_text(encoding="UTF-8"))))
    if not documents:
        raise HarnessError(f"no report.json files under {args.input_dir}")
    result = validate_documents(
        documents,
        DEFAULT_PLATFORMS,
        args.minimum_reports,
    )
    atomic_write_json(args.output, result)
    print(
        f"Validated commit {result['identity']['commit']} across "
        f"{sum(item['reports'] for item in result['platforms'].values())} reports"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dispatch_parser = sub.add_parser("dispatch")
    dispatch_parser.add_argument("--repository", default="cataggar/wamr")
    dispatch_parser.add_argument("--workflow", default="wasi-thread-bench.yml")
    dispatch_parser.add_argument("--target-sha", required=True)
    dispatch_parser.add_argument("--runs", type=int, default=20)
    dispatch_parser.add_argument("--max-in-flight", type=int, default=2)
    dispatch_parser.add_argument("--poll-seconds", type=float, default=60)
    dispatch_parser.add_argument(
        "--output", type=Path, default=Path("wasi-thread-cohort-dispatch.json")
    )

    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--input-dir", type=Path, required=True)
    validate_parser.add_argument("--minimum-reports", type=int, default=20)
    validate_parser.add_argument(
        "--output", type=Path, default=Path("wasi-thread-cohort.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return dispatch(args) if args.command == "dispatch" else validate_cohort(args)
    except (
        BenchmarkDataError,
        HarnessError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
