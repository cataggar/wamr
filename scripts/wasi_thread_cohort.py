#!/usr/bin/env python3
"""Dispatch or validate an immutable paired WASI thread benchmark cohort."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from benchmark_schema import (
    BenchmarkDataError,
    SCHEMA_VERSION,
    atomic_write_json,
    collected_at,
)
from bench_wasi_threads import (
    CANONICAL_PLATFORMS,
    COMPARISON_PURPOSES,
    HarnessError,
    PROFILE_COUNTS,
    validate_report,
)


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_PLATFORMS = tuple(CANONICAL_PLATFORMS)
RUNNER_TARGETS = ("github-hosted", "trusted-calibration")
RUNNER_ENVIRONMENTS = {
    "github-hosted": {
        "ubuntu-22.04-x86_64": "github-hosted",
        "ubuntu-24.04-aarch64": "github-hosted",
    },
    "trusted-calibration": {
        "ubuntu-22.04-x86_64": "self-hosted",
        "ubuntu-24.04-aarch64": "github-hosted",
    },
}


def require_sha(value: str, name: str = "SHA") -> str:
    if SHA_RE.fullmatch(value) is None:
        raise HarnessError(
            f"{name} must be an immutable 40-character lowercase hex commit"
        )
    return value


def gh_json(command: list[str]) -> Any:
    output = subprocess.check_output(
        ["gh", *command], text=True, stderr=subprocess.STDOUT
    )
    return json.loads(output)


def resolve_workflow_head(repository: str, workflow_ref: str) -> str:
    result = gh_json(
        ["api", f"repos/{repository}/commits/{quote(workflow_ref, safe='')}"]
    )
    if not isinstance(result, dict):
        raise HarnessError("GitHub returned an invalid workflow ref response")
    return require_sha(str(result.get("sha", "")), "workflow head SHA")


def validate_dispatch_options(args: argparse.Namespace) -> tuple[str, str, int]:
    for name in ("repository", "workflow", "workflow_ref"):
        if not isinstance(getattr(args, name, None), str) or not getattr(args, name):
            raise HarnessError(f"--{name.replace('_', '-')} must not be empty")
    baseline_sha = require_sha(args.baseline_sha, "baseline SHA")
    candidate_sha = require_sha(args.candidate_sha, "candidate SHA")
    if args.purpose not in COMPARISON_PURPOSES:
        raise HarnessError("unsupported comparison purpose")
    if args.profile not in PROFILE_COUNTS:
        raise HarnessError("unsupported measurement profile")
    if args.runner_target not in RUNNER_TARGETS:
        raise HarnessError("unsupported runner target")
    if args.warmups < 0 or args.samples <= 0:
        raise HarnessError("warmups must be non-negative and samples must be positive")
    if args.samples % 2:
        raise HarnessError("paired revision samples must be even")
    if args.purpose == "noise-calibration" and baseline_sha != candidate_sha:
        raise HarnessError("noise calibration requires identical target SHAs")
    if args.purpose == "candidate-evaluation" and baseline_sha == candidate_sha:
        raise HarnessError("candidate evaluation requires distinct target SHAs")
    if (
        args.runner_target == "trusted-calibration"
        and args.purpose != "noise-calibration"
    ):
        raise HarnessError(
            "the trusted calibration runner accepts noise calibration only"
        )
    if (
        args.runner_target == "trusted-calibration"
        and args.workflow_ref != "main"
    ):
        raise HarnessError(
            "the trusted calibration runner is reachable only from workflow ref main"
        )
    if (
        args.runs <= 1
        or args.max_in_flight <= 0
        or args.poll_seconds < 0
        or args.lookup_attempts <= 0
        or args.lookup_seconds < 0
    ):
        raise HarnessError("--runs must exceed one and --max-in-flight must be positive")
    training_runs = (
        args.runs // 2 if args.training_runs is None else args.training_runs
    )
    if training_runs <= 0 or training_runs >= args.runs:
        raise HarnessError("--training-runs must be between one and runs minus one")
    return baseline_sha, candidate_sha, training_runs


def find_dispatched_run(
    args: argparse.Namespace,
    run_name: str,
    workflow_head_sha: str,
) -> tuple[int, str]:
    for attempt in range(args.lookup_attempts):
        runs = gh_json(
            [
                "run",
                "list",
                "--repo",
                args.repository,
                "--workflow",
                args.workflow,
                "--event",
                "workflow_dispatch",
                "--limit",
                "100",
                "--json",
                "databaseId,displayTitle,headSha,url",
            ]
        )
        if not isinstance(runs, list):
            raise HarnessError("GitHub returned an invalid workflow run list")
        matches = [
            run
            for run in runs
            if isinstance(run, dict)
            and run.get("displayTitle") == run_name
            and str(run.get("headSha", "")).lower() == workflow_head_sha
        ]
        if len(matches) > 1:
            raise HarnessError(f"duplicate workflow runs found for {run_name!r}")
        if matches:
            run_id = matches[0].get("databaseId")
            url = matches[0].get("url")
            if isinstance(run_id, int) and isinstance(url, str) and url:
                return run_id, url
            raise HarnessError("GitHub returned an invalid workflow run identity")
        if attempt + 1 < args.lookup_attempts:
            time.sleep(args.lookup_seconds)
    raise HarnessError(f"could not locate dispatched workflow run {run_name!r}")


def dispatch(args: argparse.Namespace) -> int:
    baseline_sha, candidate_sha, training_runs = validate_dispatch_options(args)
    workflow_head_sha = resolve_workflow_head(args.repository, args.workflow_ref)
    cohort_id = uuid.uuid4().hex
    state = {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-thread-cohort-dispatch",
        "created_at": collected_at(),
        "repository": args.repository,
        "workflow": args.workflow,
        "workflow_ref": args.workflow_ref,
        "workflow_head_sha": workflow_head_sha,
        "cohort_id": cohort_id,
        "baseline_sha": baseline_sha,
        "candidate_sha": candidate_sha,
        "comparison_purpose": args.purpose,
        "profile": args.profile,
        "warmups": args.warmups,
        "samples": args.samples,
        "runner_target": args.runner_target,
        "required_platforms": list(DEFAULT_PLATFORMS),
        "requested_runs": args.runs,
        "requested_reports": args.runs * len(DEFAULT_PLATFORMS),
        "max_in_flight": args.max_in_flight,
        "split": {
            "method": "predeclared-sequence",
            "training_runs": training_runs,
            "holdout_runs": args.runs - training_runs,
            "assignments": [
                {
                    "sequence": sequence,
                    "partition": (
                        "training" if sequence <= training_runs else "holdout"
                    ),
                }
                for sequence in range(1, args.runs + 1)
            ],
        },
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    active: dict[int, dict[str, Any]] = {}
    launched = 0
    while launched < args.runs or active:
        while launched < args.runs and len(active) < args.max_in_flight:
            sequence = launched + 1
            partition = "training" if sequence <= training_runs else "holdout"
            run_name = (
                f"WASI thread cohort-{cohort_id}-{sequence}-{partition}"
            )
            output = subprocess.check_output(
                [
                    "gh",
                    "workflow",
                    "run",
                    args.workflow,
                    "--repo",
                    args.repository,
                    "--ref",
                    args.workflow_ref,
                    "-f",
                    f"baseline_sha={baseline_sha}",
                    "-f",
                    f"candidate_sha={candidate_sha}",
                    "-f",
                    f"purpose={args.purpose}",
                    "-f",
                    f"profile={args.profile}",
                    "-f",
                    f"warmups={args.warmups}",
                    "-f",
                    f"samples={args.samples}",
                    "-f",
                    f"runner_target={args.runner_target}",
                    "-f",
                    f"cohort_id={cohort_id}",
                    "-f",
                    f"cohort_sequence={sequence}",
                    "-f",
                    f"cohort_partition={partition}",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            match = re.search(r"(https?://\S+/actions/runs/(\d+))", output)
            if match is None:
                run_id, run_url = find_dispatched_run(
                    args, run_name, workflow_head_sha
                )
            else:
                run_id = int(match.group(2))
                run_url = match.group(1)
            record = {
                "sequence": sequence,
                "partition": partition,
                "run_id": run_id,
                "run_name": run_name,
                "url": run_url,
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
            if str(run["headSha"]).lower() != workflow_head_sha:
                atomic_write_json(args.output, state)
                raise HarnessError(
                    f"workflow run {run_id} head SHA {run['headSha']!r} "
                    f"does not match immutable workflow head {workflow_head_sha}"
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
                        f"workflow run {run_id} concluded {run['conclusion']}; "
                        "the failed observation is retained and is not retried"
                    )
                del active[run_id]
        atomic_write_json(args.output, state)
    print(f"Recorded {args.runs} successful immutable dispatches in {args.output}")
    return 0


def validate_platforms(required_platforms: tuple[str, ...]) -> None:
    if (
        len(required_platforms) != len(DEFAULT_PLATFORMS)
        or len(set(required_platforms)) != len(required_platforms)
        or set(required_platforms) != set(DEFAULT_PLATFORMS)
    ):
        raise HarnessError("cohort platforms must be exactly the canonical hosted set")


def validate_dispatch_state(
    state: dict[str, Any], required_platforms: tuple[str, ...]
) -> dict[str, Any]:
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("kind") != "wasi-thread-cohort-dispatch"
    ):
        raise HarnessError("paired cohort requires a schema-v3 dispatch manifest")
    validate_platforms(required_platforms)
    if state.get("required_platforms") != list(required_platforms):
        raise HarnessError("dispatch manifest platform order or identity changed")
    for key in ("repository", "workflow", "workflow_ref"):
        if not isinstance(state.get(key), str) or not state[key]:
            raise HarnessError(f"dispatch manifest {key}")
    cohort_id = state.get("cohort_id")
    if (
        not isinstance(cohort_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", cohort_id) is None
    ):
        raise HarnessError("dispatch manifest cohort_id")
    baseline_sha = require_sha(str(state.get("baseline_sha", "")), "baseline SHA")
    candidate_sha = require_sha(str(state.get("candidate_sha", "")), "candidate SHA")
    workflow_head_sha = require_sha(
        str(state.get("workflow_head_sha", "")), "workflow head SHA"
    )
    purpose = state.get("comparison_purpose")
    profile = state.get("profile")
    runner_target = state.get("runner_target")
    if purpose not in COMPARISON_PURPOSES:
        raise HarnessError("dispatch manifest comparison purpose")
    if profile not in PROFILE_COUNTS:
        raise HarnessError("dispatch manifest profile")
    if runner_target not in RUNNER_TARGETS:
        raise HarnessError("dispatch manifest runner target")
    warmups = state.get("warmups")
    samples = state.get("samples")
    if (
        not isinstance(warmups, int)
        or warmups < 0
        or not isinstance(samples, int)
        or samples <= 0
        or samples % 2
    ):
        raise HarnessError("dispatch manifest warmup/sample plan")
    if purpose == "noise-calibration" and baseline_sha != candidate_sha:
        raise HarnessError("noise calibration dispatch has mixed target SHAs")
    if purpose == "candidate-evaluation" and baseline_sha == candidate_sha:
        raise HarnessError("candidate evaluation dispatch has identical target SHAs")
    requested_runs = state.get("requested_runs")
    requested_reports = state.get("requested_reports")
    if (
        not isinstance(requested_runs, int)
        or requested_runs <= 1
        or requested_reports != requested_runs * len(required_platforms)
    ):
        raise HarnessError("dispatch manifest requested report count")
    split = state.get("split")
    assignments = split.get("assignments") if isinstance(split, dict) else None
    if (
        not isinstance(assignments, list)
        or len(assignments) != requested_runs
        or split.get("method") != "predeclared-sequence"
    ):
        raise HarnessError("dispatch manifest training/holdout split")
    assignment_by_sequence: dict[int, str] = {}
    for item in assignments:
        if not isinstance(item, dict):
            raise HarnessError("dispatch manifest split assignment")
        sequence = item.get("sequence")
        partition = item.get("partition")
        if (
            not isinstance(sequence, int)
            or sequence in assignment_by_sequence
            or partition not in ("training", "holdout")
        ):
            raise HarnessError("dispatch manifest split assignment")
        assignment_by_sequence[sequence] = partition
    if set(assignment_by_sequence) != set(range(1, requested_runs + 1)):
        raise HarnessError("dispatch manifest split sequence coverage")
    counts = {
        partition: sum(value == partition for value in assignment_by_sequence.values())
        for partition in ("training", "holdout")
    }
    if (
        not all(counts.values())
        or split.get("training_runs") != counts["training"]
        or split.get("holdout_runs") != counts["holdout"]
    ):
        raise HarnessError("dispatch manifest split counts")
    runs = state.get("runs")
    if not isinstance(runs, list) or len(runs) != requested_runs:
        raise HarnessError("dispatch manifest run count")
    run_by_id: dict[str, dict[str, Any]] = {}
    seen_sequences: set[int] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise HarnessError("dispatch manifest run record")
        sequence = run.get("sequence")
        run_id = str(run.get("run_id", ""))
        if (
            not isinstance(sequence, int)
            or sequence in seen_sequences
            or sequence not in assignment_by_sequence
            or not run_id.isdigit()
            or run_id in run_by_id
        ):
            raise HarnessError("dispatch manifest duplicate or invalid run identity")
        if run.get("partition") != assignment_by_sequence[sequence]:
            raise HarnessError("dispatch manifest run split changed after dispatch")
        expected_run_name = (
            f"WASI thread cohort-{cohort_id}-{sequence}-"
            f"{assignment_by_sequence[sequence]}"
        )
        if run.get("run_name") != expected_run_name:
            raise HarnessError("dispatch manifest run name changed after dispatch")
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            raise HarnessError(f"workflow run {run_id} is partial or unsuccessful")
        if str(run.get("workflow_head_sha", "")).lower() != workflow_head_sha:
            raise HarnessError(f"workflow run {run_id} used a mixed workflow head")
        seen_sequences.add(sequence)
        run_by_id[run_id] = run
    if seen_sequences != set(range(1, requested_runs + 1)):
        raise HarnessError("dispatch manifest run sequence coverage")
    return {
        "baseline_sha": baseline_sha,
        "candidate_sha": candidate_sha,
        "workflow_head_sha": workflow_head_sha,
        "cohort_id": cohort_id,
        "purpose": purpose,
        "profile": profile,
        "warmups": warmups,
        "samples": samples,
        "runner_target": runner_target,
        "requested_runs": requested_runs,
        "requested_reports": requested_reports,
        "run_by_id": run_by_id,
        "split": split,
    }


def validate_legacy_documents(
    documents: list[tuple[Path, dict[str, Any]]],
    required_platforms: tuple[str, ...],
    minimum_reports: int,
) -> dict[str, Any]:
    """Retain the pre-paired compatibility API without making it authoritative."""
    if minimum_reports < 1:
        raise HarnessError("minimum report count must be positive")
    validate_platforms(required_platforms)
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    identities: set[tuple[str, str, str, str, str]] = set()
    for path, document in documents:
        validate_report(document)
        if document["plan"]["revision_mode"] != "single-revision-compatibility":
            raise HarnessError(
                f"{path}: paired authoritative reports require their exact "
                "schema-v3 dispatch manifest"
            )
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
        "kind": "wasi-thread-cohort-legacy-compatibility",
        "authoritative": False,
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


def validate_paired_documents(
    documents: list[tuple[Path, dict[str, Any]]],
    required_platforms: tuple[str, ...],
    dispatch_state: dict[str, Any],
) -> dict[str, Any]:
    expected = validate_dispatch_state(dispatch_state, required_platforms)
    if len(documents) != expected["requested_reports"]:
        raise HarnessError(
            f"cohort has {len(documents)} reports, expected exactly "
            f"{expected['requested_reports']}"
        )
    grouped: dict[str, dict[str, tuple[Path, dict[str, Any]]]] = {
        platform: {} for platform in required_platforms
    }
    identity_by_role: dict[str, set[tuple[str, str, str]]] = {
        "baseline": set(),
        "candidate": set(),
    }
    fixture_identities: set[str] = set()
    plan_identities: set[str] = set()
    host_fingerprints: dict[str, set[str]] = defaultdict(set)
    host_pair_ids: set[str] = set()
    observations: list[dict[str, Any]] = []
    expected_run_ids = set(expected["run_by_id"])
    for path, document in documents:
        validate_report(document)
        plan = document["plan"]
        metadata = document["metadata"]
        if plan["revision_mode"] != "paired-revisions":
            raise HarnessError(f"{path}: legacy or unpaired report is not authoritative")
        platform_id = metadata["platform_id"]
        if platform_id not in required_platforms:
            raise HarnessError(f"{path}: unsupported platform {platform_id!r}")
        host = metadata["host"]
        if (host.get("system"), host.get("machine")) != CANONICAL_PLATFORMS[platform_id]:
            raise HarnessError(
                f"{path}: platform ID does not match canonical host identity"
            )
        run_id = str(host.get("github_run_id", ""))
        if run_id not in expected_run_ids:
            raise HarnessError(f"{path}: unexpected or missing workflow run ID")
        if run_id in grouped[platform_id]:
            raise HarnessError(
                f"{path}: duplicate report for {platform_id} workflow run {run_id}"
            )
        grouped[platform_id][run_id] = (path, document)
        if (
            plan["profile"] != expected["profile"]
            or plan["warmups"] != expected["warmups"]
            or plan["samples"] != expected["samples"]
            or plan["comparison_purpose"] != expected["purpose"]
        ):
            raise HarnessError(f"{path}: mixed fixture plan/profile/purpose identity")
        if document["budget"] != {
            "status": "disabled",
            "path": None,
            "failures": [],
        }:
            raise HarnessError(f"{path}: cohort report must be non-enforcing")
        revisions = metadata["revisions"]
        if revisions["baseline"]["commit"] != expected["baseline_sha"]:
            raise HarnessError(f"{path}: baseline SHA does not match dispatch")
        if revisions["candidate"]["commit"] != expected["candidate_sha"]:
            raise HarnessError(f"{path}: candidate SHA does not match dispatch")
        expected_environment = RUNNER_ENVIRONMENTS[expected["runner_target"]][
            platform_id
        ]
        if host.get("runner_environment") != expected_environment:
            raise HarnessError(f"{path}: mixed or unexpected runner environment")
        fingerprint = host["host_fingerprint"]["sha256"]
        if metadata["host_pair"]["host_fingerprint_sha256"] != fingerprint:
            raise HarnessError(f"{path}: mixed host fingerprint within report")
        host_fingerprints[platform_id].add(fingerprint)
        host_pair_id = metadata["host_pair"]["id"]
        if host_pair_id in host_pair_ids:
            raise HarnessError(f"{path}: duplicate host-pair identity")
        host_pair_ids.add(host_pair_id)
        for role in ("baseline", "candidate"):
            identity_by_role[role].add(
                (
                    revisions[role]["commit"],
                    revisions[role]["tracked_diff_sha256"],
                    revisions[role]["build_source_sha256"],
                )
            )
        fixture_identities.add(metadata["fixture_set_sha256"])
        plan_identities.add(metadata["plan_sha256"])
        run = expected["run_by_id"][run_id]
        observations.append(
            {
                "sequence": run["sequence"],
                "partition": run["partition"],
                "run_id": run_id,
                "platform": platform_id,
                "path": str(path),
                "host_pair_id": host_pair_id,
                "host_fingerprint_sha256": fingerprint,
                "records": len(document["records"]),
            }
        )
    for platform_id, reports in grouped.items():
        if set(reports) != expected_run_ids:
            missing = sorted(expected_run_ids - set(reports))
            extra = sorted(set(reports) - expected_run_ids)
            raise HarnessError(
                f"{platform_id}: incomplete exact run pairing; "
                f"missing={missing}, extra={extra}"
            )
    if any(len(items) != 1 for items in identity_by_role.values()):
        raise HarnessError("cohort has mixed baseline/candidate/build identities")
    if len(fixture_identities) != 1 or len(plan_identities) != 1:
        raise HarnessError("cohort has mixed fixture or plan identity")
    if any(len(items) != 1 for items in host_fingerprints.values()):
        raise HarnessError("cohort has mixed host fingerprints within a platform")
    observations.sort(key=lambda item: (item["sequence"], item["platform"]))
    if len(observations) != len(documents):
        raise HarnessError("cohort observation exclusion is forbidden")
    baseline_identity = next(iter(identity_by_role["baseline"]))
    candidate_identity = next(iter(identity_by_role["candidate"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-thread-paired-cohort",
        "authoritative": True,
        "validated_at": collected_at(),
        "dispatch": {
            "repository": dispatch_state["repository"],
            "workflow": dispatch_state["workflow"],
            "workflow_ref": dispatch_state["workflow_ref"],
            "workflow_head_sha": expected["workflow_head_sha"],
            "cohort_id": expected["cohort_id"],
            "runner_target": expected["runner_target"],
            "requested_runs": expected["requested_runs"],
            "requested_reports": expected["requested_reports"],
        },
        "identity": {
            "baseline": {
                "commit": baseline_identity[0],
                "tracked_diff_sha256": baseline_identity[1],
                "build_source_sha256": baseline_identity[2],
            },
            "candidate": {
                "commit": candidate_identity[0],
                "tracked_diff_sha256": candidate_identity[1],
                "build_source_sha256": candidate_identity[2],
            },
            "fixture_set_sha256": next(iter(fixture_identities)),
            "plan_sha256": next(iter(plan_identities)),
            "profile": expected["profile"],
            "comparison_purpose": expected["purpose"],
            "warmups": expected["warmups"],
            "samples": expected["samples"],
        },
        "split": {
            **expected["split"],
            "run_ids": {
                partition: sorted(
                    run_id
                    for run_id, run in expected["run_by_id"].items()
                    if run["partition"] == partition
                )
                for partition in ("training", "holdout")
            },
        },
        "platforms": {
            platform_id: {
                "reports": len(grouped[platform_id]),
                "run_ids": sorted(grouped[platform_id]),
                "host_fingerprint_sha256": next(
                    iter(host_fingerprints[platform_id])
                ),
            }
            for platform_id in required_platforms
        },
        "observations": observations,
        "excluded_observations": [],
    }


def validate_documents(
    documents: list[tuple[Path, dict[str, Any]]],
    required_platforms: tuple[str, ...],
    minimum_reports: int,
    dispatch_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not documents:
        raise HarnessError("cohort has no reports")
    paired = [
        document.get("plan", {}).get("revision_mode") == "paired-revisions"
        for _, document in documents
    ]
    if any(paired):
        if not all(paired):
            raise HarnessError("paired cohort contains legacy or unpaired reports")
        if dispatch_state is None:
            raise HarnessError(
                "paired authoritative reports require their exact schema-v3 "
                "dispatch manifest"
            )
        return validate_paired_documents(
            documents, required_platforms, dispatch_state
        )
    if dispatch_state is not None:
        raise HarnessError("dispatch-backed authoritative cohort rejects legacy reports")
    return validate_legacy_documents(
        documents, required_platforms, minimum_reports
    )


def validate_cohort(args: argparse.Namespace) -> int:
    documents = [
        (path, json.loads(path.read_text(encoding="UTF-8")))
        for path in sorted(args.input_dir.rglob("report.json"))
    ]
    if not documents:
        raise HarnessError(f"no report.json files under {args.input_dir}")
    dispatch_state = (
        json.loads(args.dispatch_state.read_text(encoding="UTF-8"))
        if args.dispatch_state is not None
        else None
    )
    result = validate_documents(
        documents,
        DEFAULT_PLATFORMS,
        args.minimum_reports,
        dispatch_state,
    )
    atomic_write_json(args.output, result)
    print(
        f"Validated {len(documents)} retained reports with "
        f"authoritative={result['authoritative']} in {args.output}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dispatch_parser = sub.add_parser("dispatch")
    dispatch_parser.add_argument("--repository", default="cataggar/wamr")
    dispatch_parser.add_argument("--workflow", default="wasi-thread-bench.yml")
    dispatch_parser.add_argument("--workflow-ref", default="main")
    dispatch_parser.add_argument("--baseline-sha", required=True)
    dispatch_parser.add_argument("--candidate-sha", required=True)
    dispatch_parser.add_argument(
        "--purpose", choices=COMPARISON_PURPOSES, required=True
    )
    dispatch_parser.add_argument(
        "--profile", choices=PROFILE_COUNTS, default="authoritative"
    )
    dispatch_parser.add_argument("--warmups", type=int, default=2)
    dispatch_parser.add_argument("--samples", type=int, default=10)
    dispatch_parser.add_argument(
        "--runner-target", choices=RUNNER_TARGETS, default="trusted-calibration"
    )
    dispatch_parser.add_argument("--runs", type=int, default=20)
    dispatch_parser.add_argument("--training-runs", type=int)
    dispatch_parser.add_argument("--max-in-flight", type=int, default=2)
    dispatch_parser.add_argument("--poll-seconds", type=float, default=60)
    dispatch_parser.add_argument("--lookup-attempts", type=int, default=30)
    dispatch_parser.add_argument("--lookup-seconds", type=float, default=2)
    dispatch_parser.add_argument(
        "--output", type=Path, default=Path("wasi-thread-cohort-dispatch.json")
    )

    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--input-dir", type=Path, required=True)
    validate_parser.add_argument(
        "--dispatch-state",
        type=Path,
        help="exact schema-v3 dispatch manifest (required for paired cohorts)",
    )
    validate_parser.add_argument(
        "--minimum-reports",
        type=int,
        default=20,
        help="legacy single-revision compatibility only",
    )
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
