#!/usr/bin/env python3
"""Dispatch or validate an immutable paired WASI thread benchmark cohort."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from benchmark_schema import (
    BenchmarkDataError,
    SCHEMA_VERSION,
    atomic_write_json,
    cache_key,
    collected_at,
)
from bench_wasi_threads import (
    CANONICAL_PLATFORMS,
    COMPARISON_PURPOSES,
    HarnessError,
    MEASUREMENT_PLAN_IDENTITY_VERSION,
    PROFILE_COUNTS,
    measurement_plan_sha256,
    validate_report,
)


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_PLATFORMS = tuple(CANONICAL_PLATFORMS)
RUNNER_TARGETS = ("github-hosted", "trusted-calibration")
TRUSTED_X86_RUNNER_NAME = "vm31e-wamr-temp-20260906"
TRUSTED_X86_RUNNER_LABEL = "wamr-temp-20260906"
DEFAULT_DISPATCH_TIMEOUT_SECONDS = 72 * 60 * 60
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
DERIVATION_POLICY_KIND = "wasi-thread-budget-derivation-policy"
DERIVATION_EVIDENCE_KIND = "wasi-thread-budget-derivation-evidence"
DIRECT_THREAD_DELTA_POLICY_LIMIT = 0.02
MAD_SCALE = 1.4826
ROBUST_MAD_MULTIPLIER = 6.0


def require_sha(value: str, name: str = "SHA") -> str:
    if SHA_RE.fullmatch(value) is None:
        raise HarnessError(
            f"{name} must be an immutable 40-character lowercase hex commit"
        )
    return value


def gh_json(command: list[str], timeout_seconds: float | None = None) -> Any:
    try:
        output = subprocess.check_output(
            ["gh", *command],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(
            f"GitHub command timed out: gh {' '.join(command)}"
        ) from exc
    return json.loads(output)


def remaining_timeout(deadline: float, context: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise HarnessError(f"cohort dispatch timed out while {context}")
    return remaining


def require_api_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HarnessError(f"GitHub returned an invalid {context} object")
    return value


def require_api_string(
    value: dict[str, Any], key: str, context: str, *, allow_empty: bool = False
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or (not allow_empty and not result):
        raise HarnessError(f"GitHub {context} is missing valid {key}")
    return result


def parse_workflow_run(value: Any, run_id: int) -> dict[str, str]:
    run = require_api_object(value, f"workflow run {run_id}")
    return {
        "status": require_api_string(run, "status", f"workflow run {run_id}"),
        "conclusion": require_api_string(
            run,
            "conclusion",
            f"workflow run {run_id}",
            allow_empty=True,
        ),
        "headSha": require_sha(
            require_api_string(run, "headSha", f"workflow run {run_id}").lower(),
            f"workflow run {run_id} head SHA",
        ),
        "url": require_api_string(run, "url", f"workflow run {run_id}"),
    }


def parse_artifacts(value: Any, run_id: int) -> list[dict[str, Any]]:
    response = require_api_object(value, f"artifact response for run {run_id}")
    artifacts = response.get("artifacts")
    if not isinstance(artifacts, list):
        raise HarnessError(
            f"GitHub artifact response for run {run_id} is missing artifacts"
        )
    parsed = []
    for index, value in enumerate(artifacts):
        context = f"artifact {index} for run {run_id}"
        artifact = require_api_object(value, context)
        artifact_id = artifact.get("id")
        size = artifact.get("size_in_bytes")
        expired = artifact.get("expired")
        if not isinstance(artifact_id, int) or artifact_id <= 0:
            raise HarnessError(f"GitHub {context} is missing valid id")
        if not isinstance(size, int) or size < 0:
            raise HarnessError(f"GitHub {context} is missing valid size_in_bytes")
        if not isinstance(expired, bool):
            raise HarnessError(f"GitHub {context} is missing valid expired")
        parsed.append(
            {
                "id": artifact_id,
                "name": require_api_string(artifact, "name", context),
                "size_in_bytes": size,
                "expired": expired,
            }
        )
    return parsed


def resolve_workflow_head(
    repository: str,
    workflow_ref: str,
    timeout_seconds: float | None = None,
) -> str:
    result = gh_json(
        ["api", f"repos/{repository}/commits/{quote(workflow_ref, safe='')}"],
        timeout_seconds,
    )
    if not isinstance(result, dict):
        raise HarnessError("GitHub returned an invalid workflow ref response")
    return require_sha(str(result.get("sha", "")), "workflow head SHA")


def preflight_runner_inventory(
    repository: str,
    runner_target: str,
    timeout_seconds: float | None = None,
) -> None:
    if runner_target != "trusted-calibration":
        return
    response = require_api_object(
        gh_json(
            [
                "api",
                f"repos/{repository}/actions/runners?per_page=100",
            ],
            timeout_seconds,
        ),
        f"Actions runner inventory for {repository}",
    )
    total_count = response.get("total_count")
    runners = response.get("runners")
    if (
        not isinstance(total_count, int)
        or isinstance(total_count, bool)
        or total_count < 0
    ):
        raise HarnessError(
            f"GitHub Actions runner inventory for {repository} "
            "is missing valid total_count"
        )
    if not isinstance(runners, list):
        raise HarnessError(
            f"GitHub Actions runner inventory for {repository} "
            "is missing runners"
        )
    if total_count != len(runners):
        raise HarnessError(
            f"GitHub Actions runner inventory for {repository} is incomplete: "
            f"received {len(runners)} of {total_count} runners in the bounded query"
        )

    parsed = []
    for index, value in enumerate(runners):
        context = f"Actions runner inventory item {index} for {repository}"
        runner = require_api_object(value, context)
        name = require_api_string(runner, "name", context)
        status = require_api_string(runner, "status", context)
        busy = runner.get("busy")
        if not isinstance(busy, bool):
            raise HarnessError(f"GitHub {context} is missing valid busy")
        labels = runner.get("labels")
        if not isinstance(labels, list):
            raise HarnessError(f"GitHub {context} is missing labels")
        parsed_labels = []
        for label_index, value in enumerate(labels):
            label_context = f"{context} label {label_index}"
            label = require_api_object(value, label_context)
            parsed_labels.append(
                {
                    "name": require_api_string(label, "name", label_context),
                    "type": require_api_string(label, "type", label_context),
                }
            )
        parsed.append(
            {
                "name": name,
                "status": status,
                "busy": busy,
                "labels": parsed_labels,
            }
        )

    matching_label = [
        runner
        for runner in parsed
        if any(
            label["name"] == TRUSTED_X86_RUNNER_LABEL
            for label in runner["labels"]
        )
    ]
    if not matching_label:
        if any(
            runner["name"] == TRUSTED_X86_RUNNER_NAME for runner in parsed
        ):
            raise HarnessError(
                f"trusted calibration runner label drift in {repository}: "
                f"{TRUSTED_X86_RUNNER_NAME!r} must have sole label "
                f"{TRUSTED_X86_RUNNER_LABEL!r}"
            )
        raise HarnessError(
            f"trusted calibration runner is missing in {repository}: "
            f"expected one runner with label {TRUSTED_X86_RUNNER_LABEL!r}"
        )
    if len(matching_label) != 1:
        raise HarnessError(
            f"trusted calibration runner label is duplicated in {repository}: "
            f"found {len(matching_label)} runners with label "
            f"{TRUSTED_X86_RUNNER_LABEL!r}"
        )

    runner = matching_label[0]
    if runner["name"] != TRUSTED_X86_RUNNER_NAME:
        raise HarnessError(
            f"trusted calibration runner name drift in {repository}: label "
            f"{TRUSTED_X86_RUNNER_LABEL!r} belongs to {runner['name']!r}, "
            f"expected {TRUSTED_X86_RUNNER_NAME!r}"
        )
    expected_labels = [
        {"name": TRUSTED_X86_RUNNER_LABEL, "type": "custom"}
    ]
    if runner["labels"] != expected_labels:
        raise HarnessError(
            f"trusted calibration runner label drift in {repository}: "
            f"{TRUSTED_X86_RUNNER_NAME!r} has labels {runner['labels']!r}, "
            f"expected sole custom label {TRUSTED_X86_RUNNER_LABEL!r}"
        )
    if runner["status"] != "online":
        raise HarnessError(
            f"trusted calibration runner is offline in {repository}: "
            f"{TRUSTED_X86_RUNNER_NAME!r} reports status {runner['status']!r}"
        )
    if runner["busy"]:
        raise HarnessError(
            f"trusted calibration runner is busy in {repository}: "
            f"{TRUSTED_X86_RUNNER_NAME!r} must be idle before cohort dispatch"
        )


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
    timeout_seconds = getattr(args, "timeout_seconds", None)
    if (
        args.runs <= 1
        or args.max_in_flight <= 0
        or args.poll_seconds < 0
        or args.lookup_attempts <= 0
        or args.lookup_seconds < 0
        or not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise HarnessError(
            "--runs must exceed one; dispatch counts and timeout must be positive"
        )
    if args.runner_target == "trusted-calibration" and args.max_in_flight > 2:
        raise HarnessError(
            "trusted calibration --max-in-flight cannot exceed 2"
        )
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
    deadline: float | None = None,
) -> tuple[int, str]:
    for attempt in range(args.lookup_attempts):
        timeout = (
            remaining_timeout(deadline, f"locating {run_name!r}")
            if deadline is not None
            else None
        )
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
            ],
            timeout,
        )
        if not isinstance(runs, list):
            raise HarnessError("GitHub returned an invalid workflow run list")
        matches = []
        for index, value in enumerate(runs):
            context = f"workflow run list item {index}"
            run = require_api_object(value, context)
            run_id = run.get("databaseId")
            if not isinstance(run_id, int) or run_id <= 0:
                raise HarnessError(f"GitHub {context} is missing valid databaseId")
            title = require_api_string(run, "displayTitle", context)
            head_sha = require_sha(
                require_api_string(run, "headSha", context).lower(),
                f"{context} head SHA",
            )
            url = require_api_string(run, "url", context)
            if title == run_name and head_sha == workflow_head_sha:
                matches.append({"databaseId": run_id, "url": url})
        if len(matches) > 1:
            raise HarnessError(f"duplicate workflow runs found for {run_name!r}")
        if matches:
            return matches[0]["databaseId"], matches[0]["url"]
        if attempt + 1 < args.lookup_attempts:
            sleep_seconds = args.lookup_seconds
            if deadline is not None:
                sleep_seconds = min(
                    sleep_seconds,
                    remaining_timeout(deadline, f"locating {run_name!r}"),
                )
            time.sleep(sleep_seconds)
    raise HarnessError(f"could not locate dispatched workflow run {run_name!r}")


def dispatch(args: argparse.Namespace) -> int:
    baseline_sha, candidate_sha, training_runs = validate_dispatch_options(args)
    deadline = time.monotonic() + args.timeout_seconds
    if args.runner_target == "trusted-calibration":
        preflight_runner_inventory(
            args.repository,
            args.runner_target,
            remaining_timeout(deadline, "checking the runner inventory"),
        )
    workflow_head_sha = resolve_workflow_head(
        args.repository,
        args.workflow_ref,
        remaining_timeout(deadline, "resolving the workflow head"),
    )
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
        "timeout_seconds": args.timeout_seconds,
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
            remaining_timeout(deadline, "launching workflow runs")
            sequence = launched + 1
            partition = "training" if sequence <= training_runs else "holdout"
            run_name = (
                f"WASI thread cohort-{cohort_id}-{sequence}-{partition}"
            )
            try:
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
                    timeout=remaining_timeout(
                        deadline, f"dispatching cohort sequence {sequence}"
                    ),
                ).strip()
            except subprocess.TimeoutExpired as exc:
                atomic_write_json(args.output, state)
                raise HarnessError(
                    f"cohort dispatch timed out while launching sequence {sequence}"
                ) from exc
            match = re.search(r"(https?://\S+/actions/runs/(\d+))", output)
            if match is None:
                run_id, run_url = find_dispatched_run(
                    args, run_name, workflow_head_sha, deadline
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

        try:
            sleep_seconds = min(
                args.poll_seconds,
                remaining_timeout(deadline, "waiting for workflow runs"),
            )
        except HarnessError:
            atomic_write_json(args.output, state)
            raise
        time.sleep(sleep_seconds)
        for run_id in list(active):
            record = active[run_id]
            run = parse_workflow_run(
                gh_json(
                    [
                        "run",
                        "view",
                        str(run_id),
                        "--repo",
                        args.repository,
                        "--json",
                        "status,conclusion,headSha,url",
                    ],
                    remaining_timeout(deadline, f"polling workflow run {run_id}"),
                ),
                run_id,
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
                record["artifacts"] = parse_artifacts(
                    gh_json(
                        [
                            "api",
                            f"repos/{args.repository}/actions/runs/"
                            f"{run_id}/artifacts",
                        ],
                        remaining_timeout(
                            deadline, f"listing artifacts for workflow run {run_id}"
                        ),
                    ),
                    run_id,
                )
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
    timeout_seconds = state.get("timeout_seconds")
    if (
        not isinstance(requested_runs, int)
        or requested_runs <= 1
        or requested_reports != requested_runs * len(required_platforms)
        or not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise HarnessError("dispatch manifest requested report count or timeout")
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
        "timeout_seconds": timeout_seconds,
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
    identities: set[tuple[str, str, str, str, int, str, str]] = set()
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
                metadata["measurement_plan_version"],
                metadata["measurement_plan_sha256"],
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
            "measurement_plan_version": identity[4],
            "measurement_plan_sha256": identity[5],
            "profile": identity[6],
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


def report_calibration_metrics(document: dict[str, Any]) -> dict[str, Any]:
    """Retain the per-sample ratios needed for reproducible calibration."""

    comparisons = [
        {
            "pair_kind": item["pair_kind"],
            "pair_key": item["pair_key"],
            "condition": item["condition"],
            "metric_kind": item["metric_kind"],
            "elapsed_candidate_over_baseline": list(
                item["elapsed_candidate_over_baseline"]["samples"]
            ),
            "throughput_candidate_over_baseline": list(
                item["throughput_candidate_over_baseline"]["samples"]
            ),
        }
        for item in document["comparison_summaries"]
    ]
    ratio_of_ratios = [
        {
            "pair_kind": item["pair_kind"],
            "pair_key": item["pair_key"],
            "left": item["left"],
            "right": item["right"],
            "elapsed_ratio_of_ratios": list(
                item["elapsed_ratio_of_ratios"]["samples"]
            ),
            "throughput_ratio_of_ratios": list(
                item["throughput_ratio_of_ratios"]["samples"]
            ),
        }
        for item in document["ratio_of_ratios_summaries"]
    ]
    direct_candidate = [
        {
            "pair_kind": item["pair_kind"],
            "pair_key": item["pair_key"],
            "left": item["left"],
            "right": item["right"],
            "elapsed_right_over_left": list(
                item["elapsed_right_over_left"]["samples"]
            ),
            "throughput_right_over_left": list(
                item["throughput_right_over_left"]["samples"]
            ),
        }
        for item in document["paired_summaries"]
        if item["revision"] == "candidate"
        and item["pair_kind"] == "single-infrastructure"
    ]
    if not direct_candidate:
        raise HarnessError(
            "report cannot prove the direct candidate threads-enabled/disabled "
            "delta: candidate single-infrastructure raw ratios are missing"
        )
    return {
        "comparisons": comparisons,
        "ratio_of_ratios": ratio_of_ratios,
        "direct_candidate_single_infrastructure": direct_candidate,
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
    measurement_plan_identities: set[tuple[int, str]] = set()
    host_fingerprints: dict[str, Counter[str]] = defaultdict(Counter)
    host_cpus: dict[str, Counter[str]] = defaultdict(Counter)
    runner_images: dict[str, Counter[str]] = defaultdict(Counter)
    runner_names: dict[str, set[str]] = defaultdict(set)
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
        fingerprint_fields = host["host_fingerprint"]["fields"]
        cpu = fingerprint_fields["cpu"]
        runner_image = fingerprint_fields["runner_image"]
        runner_name = host.get("runner_name", "")
        host_fingerprints[platform_id][fingerprint] += 1
        host_cpus[platform_id][cpu] += 1
        runner_images[platform_id][runner_image] += 1
        if runner_name:
            runner_names[platform_id].add(runner_name)
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
        measurement_plan_identities.add(
            (
                metadata["measurement_plan_version"],
                measurement_plan_sha256(plan),
            )
        )
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
                "report_sha256": cache_key(document),
                "identity": {
                    "baseline": revisions["baseline"],
                    "candidate": revisions["candidate"],
                    "fixture_set_sha256": metadata["fixture_set_sha256"],
                    "plan_sha256": metadata["plan_sha256"],
                    "measurement_plan_version": metadata[
                        "measurement_plan_version"
                    ],
                    "measurement_plan_sha256": metadata[
                        "measurement_plan_sha256"
                    ],
                    "profile": plan["profile"],
                    "comparison_purpose": plan["comparison_purpose"],
                },
                "plan": copy.deepcopy(plan),
                "metrics": report_calibration_metrics(document),
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
    if len(measurement_plan_identities) != 1:
        raise HarnessError("cohort has mixed measurement plan identity")
    trusted_x86 = "ubuntu-22.04-x86_64"
    if expected["runner_target"] == "trusted-calibration":
        if runner_names[trusted_x86] != {TRUSTED_X86_RUNNER_NAME}:
            raise HarnessError(
                f"trusted x86 reports must all come from runner "
                f"{TRUSTED_X86_RUNNER_NAME!r}"
            )
        if len(host_fingerprints[trusted_x86]) != 1:
            raise HarnessError(
                f"trusted x86 reports have mixed "
                f"{TRUSTED_X86_RUNNER_NAME} host fingerprints"
            )
    observations.sort(key=lambda item: (item["sequence"], item["platform"]))
    if len(observations) != len(documents):
        raise HarnessError("cohort observation exclusion is forbidden")
    baseline_identity = next(iter(identity_by_role["baseline"]))
    candidate_identity = next(iter(identity_by_role["candidate"]))
    measurement_plan_identity = next(iter(measurement_plan_identities))
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
            "measurement_plan_version": measurement_plan_identity[0],
            "measurement_plan_sha256": measurement_plan_identity[1],
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
                "host_fingerprint_distribution": dict(
                    sorted(host_fingerprints[platform_id].items())
                ),
                "cpu_distribution": dict(
                    sorted(host_cpus[platform_id].items())
                ),
                "runner_image_distribution": dict(
                    sorted(runner_images[platform_id].items())
                ),
                **(
                    {
                        "trusted_runner_name": TRUSTED_X86_RUNNER_NAME,
                        "host_fingerprint_sha256": next(
                            iter(host_fingerprints[platform_id])
                        ),
                    }
                    if expected["runner_target"] == "trusted-calibration"
                    and platform_id == trusted_x86
                    else {}
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


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HarnessError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="UTF-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise HarnessError(f"{label} must be a JSON object")
    return value


def _finite_number(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (positive and value <= 0)
        or (nonnegative and value < 0)
    ):
        qualifier = "positive " if positive else "non-negative " if nonnegative else ""
        raise HarnessError(f"{label} must be a finite {qualifier}number")
    return float(value)


def validate_derivation_policy(policy: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "minimum_reports_per_platform",
        "rounding_cushion_log",
        "engineering_policy_ceiling_log",
        "direct_candidate_single_infrastructure_max_delta_fraction",
    }
    if set(policy) != expected_keys:
        raise HarnessError("derivation policy fields are incomplete or unknown")
    if policy["schema_version"] != 1 or policy["kind"] != DERIVATION_POLICY_KIND:
        raise HarnessError("derivation policy schema/kind mismatch")
    minimum = policy["minimum_reports_per_platform"]
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or minimum < 20
    ):
        raise HarnessError(
            "derivation policy minimum_reports_per_platform must be >= 20"
        )
    cushion = _finite_number(
        policy["rounding_cushion_log"],
        "derivation policy rounding_cushion_log",
        nonnegative=True,
    )
    ceiling_names = {
        "comparison_throughput_lower",
        "comparison_elapsed_upper",
        "ratio_of_ratios_throughput_lower",
        "ratio_of_ratios_elapsed_upper",
    }
    ceilings = policy["engineering_policy_ceiling_log"]
    if not isinstance(ceilings, dict) or set(ceilings) != ceiling_names:
        raise HarnessError(
            "derivation policy engineering ceilings are incomplete or unknown"
        )
    normalized_ceilings = {
        name: _finite_number(
            ceilings[name],
            f"derivation policy ceiling {name}",
            positive=True,
        )
        for name in sorted(ceiling_names)
    }
    if any(cushion > value for value in normalized_ceilings.values()):
        raise HarnessError(
            "derivation policy rounding cushion exceeds an engineering ceiling"
        )
    direct_limit = _finite_number(
        policy["direct_candidate_single_infrastructure_max_delta_fraction"],
        "direct candidate single-infrastructure delta limit",
        positive=True,
    )
    if direct_limit != DIRECT_THREAD_DELTA_POLICY_LIMIT:
        raise HarnessError(
            "direct candidate single-infrastructure delta limit must be exactly "
            "the issue policy value 0.02"
        )
    return {
        "schema_version": 1,
        "kind": DERIVATION_POLICY_KIND,
        "minimum_reports_per_platform": minimum,
        "rounding_cushion_log": cushion,
        "engineering_policy_ceiling_log": normalized_ceilings,
        "direct_candidate_single_infrastructure_max_delta_fraction": direct_limit,
    }


def _sample_median(values: Any, label: str) -> float:
    if not isinstance(values, list) or not values:
        raise HarnessError(f"{label} is missing raw per-report metric samples")
    normalized = [
        _finite_number(value, f"{label}[{index}]", positive=True)
        for index, value in enumerate(values)
    ]
    return float(statistics.median(normalized))


def _metric_key(item: dict[str, Any], kind: str) -> tuple[str, ...]:
    if kind == "comparison":
        keys = ("pair_key", "condition", "metric_kind")
    else:
        keys = ("pair_key", "left", "right")
    if any(not isinstance(item.get(key), str) or not item[key] for key in keys):
        raise HarnessError(f"{kind} metric identity is incomplete")
    return tuple(item[key] for key in keys)


def _indexed_metrics(
    value: Any,
    kind: str,
    sample_fields: tuple[str, str],
    context: str,
) -> dict[tuple[str, ...], dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise HarnessError(f"{context} is missing raw per-report metrics")
    indexed: dict[tuple[str, ...], dict[str, Any]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise HarnessError(f"{context}[{index}] must be an object")
        key = _metric_key(item, kind)
        if key in indexed:
            raise HarnessError(f"{context} contains duplicate metric {key}")
        normalized = dict(item)
        for field in sample_fields:
            normalized[field] = list(item.get(field, []))
            _sample_median(
                normalized[field],
                f"{context}[{index}].{field}",
            )
        indexed[key] = normalized
    return indexed


def validate_calibration_cohort(
    cohort: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    if (
        cohort.get("schema_version") != SCHEMA_VERSION
        or cohort.get("kind") != "wasi-thread-paired-cohort"
        or cohort.get("authoritative") is not True
    ):
        raise HarnessError(
            "derivation requires a validated authoritative schema-v3 paired cohort"
        )
    if cohort.get("excluded_observations") != []:
        raise HarnessError("derivation forbids excluded observations or outlier drops")
    identity = cohort.get("identity")
    dispatch = cohort.get("dispatch")
    split = cohort.get("split")
    platforms = cohort.get("platforms")
    observations = cohort.get("observations")
    if not all(
        isinstance(value, dict)
        for value in (identity, dispatch, split, platforms)
    ) or not isinstance(observations, list):
        raise HarnessError("validated cohort is missing required provenance")
    if identity.get("profile") != "authoritative":
        raise HarnessError("derivation rejects smoke or non-authoritative cohorts")
    if identity.get("comparison_purpose") != "noise-calibration":
        raise HarnessError("derivation requires noise-calibration purpose")
    baseline = identity.get("baseline")
    candidate = identity.get("candidate")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise HarnessError("validated cohort revision identity is incomplete")
    revision_keys = ("commit", "tracked_diff_sha256", "build_source_sha256")
    if any(baseline.get(key) != candidate.get(key) for key in revision_keys):
        raise HarnessError("noise-calibration cohort has mixed revision identity")
    require_sha(str(baseline.get("commit", "")), "calibration commit")
    for role, revision in (("baseline", baseline), ("candidate", candidate)):
        for key in ("tracked_diff_sha256", "build_source_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(revision.get(key, ""))) is None:
                raise HarnessError(f"calibration {role} {key} is invalid")
    for key in (
        "fixture_set_sha256",
        "plan_sha256",
        "measurement_plan_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(identity.get(key, ""))) is None:
            raise HarnessError(f"validated cohort {key} is invalid")
    if (
        identity.get("measurement_plan_version")
        != MEASUREMENT_PLAN_IDENTITY_VERSION
    ):
        raise HarnessError("validated cohort measurement plan version is invalid")
    if set(platforms) != set(DEFAULT_PLATFORMS):
        raise HarnessError("validated cohort platform set is partial or mixed")
    if dispatch.get("requested_reports") != len(observations):
        raise HarnessError(
            "validated cohort observation count changed after validation"
        )
    requested_runs = dispatch.get("requested_runs")
    if not isinstance(requested_runs, int) or requested_runs < 2:
        raise HarnessError("validated cohort requested run count is invalid")
    for key in ("repository", "workflow", "workflow_ref", "cohort_id", "runner_target"):
        if not isinstance(dispatch.get(key), str) or not dispatch[key]:
            raise HarnessError(f"validated cohort dispatch {key} is missing")
    require_sha(
        str(dispatch.get("workflow_head_sha", "")),
        "validated cohort workflow head SHA",
    )
    if re.fullmatch(r"[0-9a-f]{32}", dispatch["cohort_id"]) is None:
        raise HarnessError("validated cohort dispatch cohort_id is invalid")

    assignments = split.get("assignments")
    run_ids = split.get("run_ids")
    if (
        split.get("method") != "predeclared-sequence"
        or not isinstance(assignments, list)
        or not isinstance(run_ids, dict)
        or set(run_ids) != {"training", "holdout"}
    ):
        raise HarnessError(
            "validated cohort lacks a predeclared training/holdout split"
        )
    assignment_by_sequence: dict[int, str] = {}
    for item in assignments:
        if not isinstance(item, dict):
            raise HarnessError("validated cohort split assignment is invalid")
        sequence = item.get("sequence")
        partition = item.get("partition")
        if (
            not isinstance(sequence, int)
            or sequence in assignment_by_sequence
            or partition not in ("training", "holdout")
        ):
            raise HarnessError("validated cohort split assignment is invalid")
        assignment_by_sequence[sequence] = partition
    if set(assignment_by_sequence) != set(range(1, requested_runs + 1)):
        raise HarnessError("validated cohort split sequence coverage changed")
    assignment_counts = Counter(assignment_by_sequence.values())
    if (
        split.get("training_runs") != assignment_counts["training"]
        or split.get("holdout_runs") != assignment_counts["holdout"]
    ):
        raise HarnessError("validated cohort split counts changed")
    declared_run_ids: dict[str, set[str]] = {}
    for partition in ("training", "holdout"):
        values = run_ids[partition]
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str) or not value.isdigit()
                for value in values
            )
            or len(values) != len(set(values))
        ):
            raise HarnessError(f"validated cohort {partition} run IDs are invalid")
        declared_run_ids[partition] = set(values)
    overlap = declared_run_ids["training"] & declared_run_ids["holdout"]
    if overlap:
        raise HarnessError(
            f"validated cohort training/holdout overlap: {sorted(overlap)}"
        )

    expected_identity = {
        "baseline": baseline,
        "candidate": candidate,
        "fixture_set_sha256": identity["fixture_set_sha256"],
        "plan_sha256": identity["plan_sha256"],
        "measurement_plan_version": identity["measurement_plan_version"],
        "measurement_plan_sha256": identity["measurement_plan_sha256"],
        "profile": "authoritative",
        "comparison_purpose": "noise-calibration",
    }
    seen: set[tuple[str, str]] = set()
    observed_by_partition: dict[str, set[str]] = {
        "training": set(),
        "holdout": set(),
    }
    normalized_observations: list[dict[str, Any]] = []
    expected_metric_keys: dict[str, set[tuple[str, ...]]] = {}
    platform_counts: Counter[str] = Counter()
    run_platforms: dict[str, set[str]] = defaultdict(set)
    run_sequences: dict[str, int] = {}
    sequence_runs: dict[int, str] = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise HarnessError(f"cohort observation {index} must be an object")
        run_id = observation.get("run_id")
        platform = observation.get("platform")
        sequence = observation.get("sequence")
        partition = observation.get("partition")
        if (
            not isinstance(run_id, str)
            or not run_id.isdigit()
            or platform not in DEFAULT_PLATFORMS
            or not isinstance(sequence, int)
            or partition not in ("training", "holdout")
        ):
            raise HarnessError(f"cohort observation {index} identity is invalid")
        if (run_id, platform) in seen:
            raise HarnessError(f"duplicate cohort observation {run_id}/{platform}")
        if assignment_by_sequence.get(sequence) != partition:
            raise HarnessError(
                f"cohort observation {run_id}/{platform} changed predeclared membership"
            )
        if run_id not in declared_run_ids[partition]:
            raise HarnessError(
                f"cohort observation {run_id}/{platform} is outside its declared split"
            )
        if run_id in run_sequences and run_sequences[run_id] != sequence:
            raise HarnessError(f"cohort run {run_id} has mixed sequence identity")
        if sequence in sequence_runs and sequence_runs[sequence] != run_id:
            raise HarnessError(
                f"cohort sequence {sequence} has mixed workflow run identity"
            )
        run_sequences[run_id] = sequence
        sequence_runs[sequence] = run_id
        report_sha256 = observation.get("report_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", str(report_sha256 or "")) is None:
            raise HarnessError(
                f"cohort observation {run_id}/{platform} lacks validated "
                "report identity"
            )
        report_identity = observation.get("identity")
        if not isinstance(report_identity, dict):
            raise HarnessError(
                f"cohort observation {run_id}/{platform} lacks report identity"
            )
        for key in (
            "fixture_set_sha256",
            "plan_sha256",
            "measurement_plan_version",
            "measurement_plan_sha256",
            "profile",
            "comparison_purpose",
        ):
            if report_identity.get(key) != expected_identity[key]:
                raise HarnessError(
                    f"cohort observation {run_id}/{platform} has mixed {key}"
                )
        report_plan = observation.get("plan")
        if not isinstance(report_plan, dict):
            raise HarnessError(
                f"cohort observation {run_id}/{platform} lacks its full report plan"
            )
        if cache_key(report_plan) != report_identity["plan_sha256"]:
            raise HarnessError(
                f"cohort observation {run_id}/{platform} full plan hash changed"
            )
        if (
            measurement_plan_sha256(report_plan)
            != report_identity["measurement_plan_sha256"]
        ):
            raise HarnessError(
                f"cohort observation {run_id}/{platform} measurement plan "
                "identity changed"
            )
        for role in ("baseline", "candidate"):
            revision = report_identity.get(role)
            if not isinstance(revision, dict) or any(
                revision.get(key) != expected_identity[role].get(key)
                for key in revision_keys
            ):
                raise HarnessError(
                    f"cohort observation {run_id}/{platform} has mixed {role} identity"
                )
        metrics = observation.get("metrics")
        if not isinstance(metrics, dict):
            raise HarnessError(
                f"cohort observation {run_id}/{platform} is missing raw "
                "per-report metrics"
            )
        comparisons = _indexed_metrics(
            metrics.get("comparisons"),
            "comparison",
            (
                "elapsed_candidate_over_baseline",
                "throughput_candidate_over_baseline",
            ),
            f"observation {run_id}/{platform} comparisons",
        )
        ratios = _indexed_metrics(
            metrics.get("ratio_of_ratios"),
            "ratio-of-ratios",
            ("elapsed_ratio_of_ratios", "throughput_ratio_of_ratios"),
            f"observation {run_id}/{platform} ratio_of_ratios",
        )
        direct = _indexed_metrics(
            metrics.get("direct_candidate_single_infrastructure"),
            "direct candidate single-infrastructure",
            ("elapsed_right_over_left", "throughput_right_over_left"),
            f"observation {run_id}/{platform} direct candidate metrics",
        )
        if any(
            item.get("pair_kind") != "single-infrastructure"
            or item.get("left") != "threads-disabled"
            or item.get("right") != "threads-enabled"
            for item in direct.values()
        ):
            raise HarnessError(
                f"observation {run_id}/{platform} cannot prove the direct "
                "threads-enabled/disabled direction"
            )
        metric_sets = {
            "comparisons": set(comparisons),
            "ratio_of_ratios": set(ratios),
            "direct": set(direct),
        }
        if not expected_metric_keys:
            expected_metric_keys = metric_sets
        elif metric_sets != expected_metric_keys:
            raise HarnessError(
                f"observation {run_id}/{platform} has partial or mixed metric coverage"
            )
        seen.add((run_id, platform))
        observed_by_partition[partition].add(run_id)
        platform_counts[platform] += 1
        run_platforms[run_id].add(platform)
        normalized_observations.append(
            {
                **observation,
                "comparisons": comparisons,
                "ratio_of_ratios": ratios,
                "direct": direct,
            }
        )
    if observed_by_partition != declared_run_ids:
        raise HarnessError("validated cohort split membership changed after validation")
    all_run_ids = declared_run_ids["training"] | declared_run_ids["holdout"]
    if len(all_run_ids) != requested_runs:
        raise HarnessError("validated cohort run count does not match its split")
    if any(run_platforms[run_id] != set(DEFAULT_PLATFORMS) for run_id in all_run_ids):
        raise HarnessError("validated cohort has partial cross-platform run pairing")
    minimum = policy["minimum_reports_per_platform"]
    if any(platform_counts[platform] < minimum for platform in DEFAULT_PLATFORMS):
        raise HarnessError(
            f"validated cohort needs at least {minimum} reports per platform"
        )
    if any(
        platforms[platform].get("reports") != platform_counts[platform]
        for platform in DEFAULT_PLATFORMS
    ):
        raise HarnessError("validated cohort platform report counts changed")
    for platform in DEFAULT_PLATFORMS:
        for key in (
            "host_fingerprint_distribution",
            "cpu_distribution",
            "runner_image_distribution",
        ):
            distribution = platforms[platform].get(key)
            if (
                not isinstance(distribution, dict)
                or not distribution
                or any(
                    not isinstance(name, str)
                    or not name
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or count <= 0
                    for name, count in distribution.items()
                )
                or sum(distribution.values()) != platform_counts[platform]
            ):
                raise HarnessError(
                    f"validated cohort {platform} {key} is incomplete"
                )
    return {
        "identity": identity,
        "dispatch": dispatch,
        "split": split,
        "platforms": platforms,
        "observations": normalized_observations,
        "platform_counts": dict(platform_counts),
        "metric_keys": expected_metric_keys,
    }


def derive_one_sided_threshold(
    values: list[float],
    direction: str,
    rounding_cushion_log: float,
    engineering_ceiling_log: float,
    label: str,
) -> dict[str, Any]:
    if direction not in ("lower", "upper"):
        raise HarnessError(f"{label}: invalid threshold direction")
    if not values:
        raise HarnessError(f"{label}: no training observations")
    logs = [
        math.log(_finite_number(value, f"{label}[{index}]", positive=True))
        for index, value in enumerate(values)
    ]
    median_log = float(statistics.median(logs))
    mad_log = float(
        statistics.median(abs(value - median_log) for value in logs)
    )
    robust_radius = ROBUST_MAD_MULTIPLIER * MAD_SCALE * mad_log
    if direction == "lower":
        worst_observed_deviation_log = max(0.0, -min(logs))
        robust_deviation_log = max(0.0, -(median_log - robust_radius))
    else:
        worst_observed_deviation_log = max(0.0, max(logs))
        robust_deviation_log = max(0.0, median_log + robust_radius)
    selected_source = (
        "worst-observed"
        if worst_observed_deviation_log >= robust_deviation_log
        else "median-plus-six-scaled-mad"
    )
    selected_noise_bound_log = max(
        worst_observed_deviation_log, robust_deviation_log
    )
    final_bound_log = selected_noise_bound_log + rounding_cushion_log
    if final_bound_log > engineering_ceiling_log:
        raise HarnessError(
            f"{label}: derived log bound {final_bound_log:.17g} exceeds "
            f"engineering policy ceiling {engineering_ceiling_log:.17g}"
        )
    threshold = math.exp(
        -final_bound_log if direction == "lower" else final_bound_log
    )
    lower_diagnostic = median_log - robust_radius
    upper_diagnostic = median_log + robust_radius
    outlier_indices = [
        index
        for index, value in enumerate(logs)
        if value < lower_diagnostic or value > upper_diagnostic
    ]
    return {
        "direction": direction,
        "observation_count": len(values),
        "input_ratios": list(values),
        "input_log_ratios": logs,
        "median_log_ratio": median_log,
        "mad_log_ratio": mad_log,
        "mad_scale": MAD_SCALE,
        "mad_multiplier": ROBUST_MAD_MULTIPLIER,
        "worst_observed_deviation_log": worst_observed_deviation_log,
        "robust_deviation_log": robust_deviation_log,
        "selected_source": selected_source,
        "selected_noise_bound_log": selected_noise_bound_log,
        "rounding_cushion_log": rounding_cushion_log,
        "final_bound_log": final_bound_log,
        "engineering_policy_ceiling_log": engineering_ceiling_log,
        "threshold_ratio": threshold,
        "diagnostic_outlier_indices": outlier_indices,
        "dropped_observations": [],
    }


def _observation_metric_median(
    observation: dict[str, Any],
    collection: str,
    key: tuple[str, ...],
    field: str,
) -> float:
    return _sample_median(
        observation[collection][key][field],
        f"{observation['run_id']}/{observation['platform']}/{key}/{field}",
    )


def derive_budget_documents(
    cohort: dict[str, Any],
    policy_document: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    policy = validate_derivation_policy(policy_document)
    validated = validate_calibration_cohort(cohort, policy)
    observations = validated["observations"]
    cushion = policy["rounding_cushion_log"]
    ceilings = policy["engineering_policy_ceiling_log"]
    direct_limit = policy[
        "direct_candidate_single_infrastructure_max_delta_fraction"
    ]
    platform_budgets: dict[str, Any] = {}
    platform_evidence: dict[str, Any] = {}
    holdout_failures: list[str] = []
    direct_failures: list[str] = []

    for platform in DEFAULT_PLATFORMS:
        selected = [item for item in observations if item["platform"] == platform]
        training = [item for item in selected if item["partition"] == "training"]
        holdout = [item for item in selected if item["partition"] == "holdout"]
        comparison_budget = []
        comparison_evidence = []
        for key in sorted(validated["metric_keys"]["comparisons"]):
            throughput_values = [
                _observation_metric_median(
                    item,
                    "comparisons",
                    key,
                    "throughput_candidate_over_baseline",
                )
                for item in training
            ]
            elapsed_values = [
                _observation_metric_median(
                    item,
                    "comparisons",
                    key,
                    "elapsed_candidate_over_baseline",
                )
                for item in training
            ]
            throughput_rule = derive_one_sided_threshold(
                throughput_values,
                "lower",
                cushion,
                ceilings["comparison_throughput_lower"],
                f"{platform} comparison {key} throughput",
            )
            elapsed_rule = derive_one_sided_threshold(
                elapsed_values,
                "upper",
                cushion,
                ceilings["comparison_elapsed_upper"],
                f"{platform} comparison {key} elapsed",
            )
            threshold = {
                "pair_key": key[0],
                "condition": key[1],
                "metric_kind": key[2],
                "min_candidate_over_baseline_throughput_ratio": throughput_rule[
                    "threshold_ratio"
                ],
                "max_candidate_over_baseline_elapsed_ratio": elapsed_rule[
                    "threshold_ratio"
                ],
            }
            holdout_results = []
            for item in holdout:
                throughput = _observation_metric_median(
                    item,
                    "comparisons",
                    key,
                    "throughput_candidate_over_baseline",
                )
                elapsed = _observation_metric_median(
                    item,
                    "comparisons",
                    key,
                    "elapsed_candidate_over_baseline",
                )
                passed = (
                    throughput
                    >= threshold[
                        "min_candidate_over_baseline_throughput_ratio"
                    ]
                    and elapsed
                    <= threshold["max_candidate_over_baseline_elapsed_ratio"]
                )
                holdout_results.append(
                    {
                        "run_id": item["run_id"],
                        "throughput_ratio": throughput,
                        "elapsed_ratio": elapsed,
                        "passed": passed,
                    }
                )
                if not passed:
                    holdout_failures.append(
                        f"{platform} run {item['run_id']} comparison "
                        f"{key[0]}/{key[1]} failed: throughput={throughput:.17g} "
                        "(min="
                        f"{threshold['min_candidate_over_baseline_throughput_ratio']:.17g}), "
                        f"elapsed={elapsed:.17g} "
                        "(max="
                        f"{threshold['max_candidate_over_baseline_elapsed_ratio']:.17g})"
                    )
            comparison_budget.append(threshold)
            comparison_evidence.append(
                {
                    "pair_key": key[0],
                    "condition": key[1],
                    "metric_kind": key[2],
                    "training_run_ids": [item["run_id"] for item in training],
                    "throughput_rule": throughput_rule,
                    "elapsed_rule": elapsed_rule,
                    "holdout": holdout_results,
                }
            )

        ratio_budget = []
        ratio_evidence = []
        for key in sorted(validated["metric_keys"]["ratio_of_ratios"]):
            throughput_values = [
                _observation_metric_median(
                    item,
                    "ratio_of_ratios",
                    key,
                    "throughput_ratio_of_ratios",
                )
                for item in training
            ]
            elapsed_values = [
                _observation_metric_median(
                    item,
                    "ratio_of_ratios",
                    key,
                    "elapsed_ratio_of_ratios",
                )
                for item in training
            ]
            throughput_rule = derive_one_sided_threshold(
                throughput_values,
                "lower",
                cushion,
                ceilings["ratio_of_ratios_throughput_lower"],
                f"{platform} ratio-of-ratios {key} throughput",
            )
            elapsed_rule = derive_one_sided_threshold(
                elapsed_values,
                "upper",
                cushion,
                ceilings["ratio_of_ratios_elapsed_upper"],
                f"{platform} ratio-of-ratios {key} elapsed",
            )
            threshold = {
                "pair_key": key[0],
                "left": key[1],
                "right": key[2],
                "min_candidate_over_baseline_throughput_ratio_of_ratios": (
                    throughput_rule["threshold_ratio"]
                ),
                "max_candidate_over_baseline_elapsed_ratio_of_ratios": (
                    elapsed_rule["threshold_ratio"]
                ),
            }
            holdout_results = []
            for item in holdout:
                throughput = _observation_metric_median(
                    item,
                    "ratio_of_ratios",
                    key,
                    "throughput_ratio_of_ratios",
                )
                elapsed = _observation_metric_median(
                    item,
                    "ratio_of_ratios",
                    key,
                    "elapsed_ratio_of_ratios",
                )
                passed = (
                    throughput
                    >= threshold[
                        "min_candidate_over_baseline_throughput_ratio_of_ratios"
                    ]
                    and elapsed
                    <= threshold[
                        "max_candidate_over_baseline_elapsed_ratio_of_ratios"
                    ]
                )
                holdout_results.append(
                    {
                        "run_id": item["run_id"],
                        "throughput_ratio_of_ratios": throughput,
                        "elapsed_ratio_of_ratios": elapsed,
                        "passed": passed,
                    }
                )
                if not passed:
                    holdout_failures.append(
                        f"{platform} run {item['run_id']} ratio-of-ratios "
                        f"{key[0]} failed: throughput={throughput:.17g} "
                        "(min="
                        f"{threshold['min_candidate_over_baseline_throughput_ratio_of_ratios']:.17g}), "
                        f"elapsed={elapsed:.17g} "
                        "(max="
                        f"{threshold['max_candidate_over_baseline_elapsed_ratio_of_ratios']:.17g})"
                    )
            ratio_budget.append(threshold)
            ratio_evidence.append(
                {
                    "pair_key": key[0],
                    "left": key[1],
                    "right": key[2],
                    "training_run_ids": [item["run_id"] for item in training],
                    "throughput_rule": throughput_rule,
                    "elapsed_rule": elapsed_rule,
                    "holdout": holdout_results,
                }
            )

        direct_evidence = []
        for key in sorted(validated["metric_keys"]["direct"]):
            results = []
            for item in selected:
                throughput = _observation_metric_median(
                    item, "direct", key, "throughput_right_over_left"
                )
                elapsed = _observation_metric_median(
                    item, "direct", key, "elapsed_right_over_left"
                )
                throughput_delta = abs(throughput - 1.0)
                elapsed_delta = abs(elapsed - 1.0)
                passed = (
                    throughput_delta < direct_limit
                    and elapsed_delta < direct_limit
                )
                results.append(
                    {
                        "run_id": item["run_id"],
                        "partition": item["partition"],
                        "throughput_ratio": throughput,
                        "elapsed_ratio": elapsed,
                        "absolute_throughput_delta_fraction": throughput_delta,
                        "absolute_elapsed_delta_fraction": elapsed_delta,
                        "passed": passed,
                    }
                )
                if not passed:
                    direct_failures.append(
                        f"{platform} run {item['run_id']} direct candidate "
                        f"{key[0]} threads-enabled/disabled delta failed: "
                        f"throughput_delta={throughput_delta:.17g}, "
                        f"elapsed_delta={elapsed_delta:.17g}, "
                        f"required both < {direct_limit:.17g}"
                    )
            direct_evidence.append(
                {
                    "pair_key": key[0],
                    "left": key[1],
                    "right": key[2],
                    "policy": "absolute median ratio delta must be strictly below 2%",
                    "limit_fraction": direct_limit,
                    "observations": results,
                }
            )

        system, machine = CANONICAL_PLATFORMS[platform]
        runner_target = validated["dispatch"].get("runner_target")
        if runner_target not in RUNNER_ENVIRONMENTS:
            raise HarnessError("validated cohort runner target is unknown")
        platform_budgets[platform] = {
            "host_system": system,
            "host_machine": machine,
            "runner_environment": RUNNER_ENVIRONMENTS[runner_target][platform],
            "comparisons": comparison_budget,
            "ratio_of_ratios": ratio_budget,
        }
        platform_evidence[platform] = {
            "report_count": len(selected),
            "training_report_count": len(training),
            "holdout_report_count": len(holdout),
            "host_fingerprint_distribution": validated["platforms"][platform].get(
                "host_fingerprint_distribution", {}
            ),
            "cpu_distribution": validated["platforms"][platform].get(
                "cpu_distribution", {}
            ),
            "runner_image_distribution": validated["platforms"][platform].get(
                "runner_image_distribution", {}
            ),
            "comparisons": comparison_evidence,
            "ratio_of_ratios": ratio_evidence,
            "direct_candidate_single_infrastructure": direct_evidence,
        }

    if direct_failures:
        raise HarnessError(
            "direct candidate single-infrastructure policy failed:\n"
            + "\n".join(direct_failures)
        )
    if holdout_failures:
        raise HarnessError(
            "untouched holdout validation failed:\n" + "\n".join(holdout_failures)
        )

    identity = validated["identity"]
    calibration_revision = {
        "commit": identity["baseline"]["commit"],
        "build_source_sha256": identity["baseline"]["build_source_sha256"],
    }
    budget = {
        "schema_version": SCHEMA_VERSION,
        "kind": "wasi-thread-benchmark-budget",
        "calibrated": True,
        "enforcement": False,
        "calibration_requirements": {
            "minimum_reports_per_platform": policy[
                "minimum_reports_per_platform"
            ],
            "required_profile": "authoritative",
            "required_platforms": list(DEFAULT_PLATFORMS),
        },
        "calibration_provenance": {
            "baseline_revision": calibration_revision,
            "candidate_revision": dict(calibration_revision),
            "comparison_purpose": "noise-calibration",
            "fixture_set_sha256": identity["fixture_set_sha256"],
            "plan_sha256": identity["plan_sha256"],
            "measurement_plan_version": identity[
                "measurement_plan_version"
            ],
            "measurement_plan_sha256": identity[
                "measurement_plan_sha256"
            ],
            "profile": "authoritative",
            "report_count_by_platform": validated["platform_counts"],
        },
        "platforms": platform_budgets,
    }
    evidence = {
        "schema_version": 1,
        "kind": DERIVATION_EVIDENCE_KIND,
        "source_cohort_sha256": cache_key(cohort),
        "candidate_budget_sha256": cache_key(budget),
        "enforcement_enabled": False,
        "method": {
            "domain": "natural log of each per-report median raw ratio",
            "robust_rule": (
                "For each one-sided metric, select the larger of the worst "
                "observed training deviation and median +/- "
                "6*1.4826*MAD, then add the predeclared log rounding cushion."
            ),
            "ceiling_rule": (
                "Fail when the selected bound plus cushion exceeds its "
                "separately declared engineering policy ceiling."
            ),
            "holdout_rule": (
                "Every untouched holdout report must satisfy every derived "
                "one-sided threshold."
            ),
            "outlier_rule": "Outliers are diagnostic only; no observation is dropped.",
        },
        "policy": policy,
        "calibration_provenance": {
            "repository": validated["dispatch"].get("repository"),
            "workflow": validated["dispatch"].get("workflow"),
            "workflow_ref": validated["dispatch"].get("workflow_ref"),
            "workflow_head_sha": validated["dispatch"].get("workflow_head_sha"),
            "cohort_id": validated["dispatch"].get("cohort_id"),
            "runner_target": validated["dispatch"].get("runner_target"),
            "workflow_run_ids": sorted(
                {
                    item["run_id"]
                    for item in validated["observations"]
                },
                key=int,
            ),
            "training_run_ids": list(validated["split"]["run_ids"]["training"]),
            "holdout_run_ids": list(validated["split"]["run_ids"]["holdout"]),
            "baseline_revision": identity["baseline"],
            "candidate_revision": identity["candidate"],
            "fixture_set_sha256": identity["fixture_set_sha256"],
            "plan_sha256": identity["plan_sha256"],
            "measurement_plan_version": identity[
                "measurement_plan_version"
            ],
            "measurement_plan_sha256": identity[
                "measurement_plan_sha256"
            ],
            "profile": identity["profile"],
            "comparison_purpose": identity["comparison_purpose"],
            "warmups": identity.get("warmups"),
            "samples": identity.get("samples"),
        },
        "platforms": platform_evidence,
        "excluded_observations": [],
        "holdout_status": "passed",
        "direct_candidate_single_infrastructure_status": "passed",
    }
    lines = [
        "# WASI thread budget derivation evidence",
        "",
        "- Enforcement: **disabled** (a later proof/final PR must enable it).",
        f"- Source cohort SHA-256: `{evidence['source_cohort_sha256']}`",
        f"- Candidate budget SHA-256: `{evidence['candidate_budget_sha256']}`",
        "- Formula: natural-log ratios; max(worst observed training deviation, "
        "median ± 6×1.4826×MAD), plus the predeclared log rounding cushion.",
        "- Outliers: diagnostic only; zero observations dropped.",
        "- Holdout: all untouched holdout reports passed.",
        "- Direct candidate threads-enabled/disabled: every report on both "
        "architectures has absolute median throughput and elapsed ratio delta <2%.",
        "",
        "## Platforms",
        "",
        "| Platform | Reports | Training | Holdout | Comparisons | Ratio-of-ratios |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for platform in DEFAULT_PLATFORMS:
        item = platform_evidence[platform]
        lines.append(
            f"| {platform} | {item['report_count']} | "
            f"{item['training_report_count']} | {item['holdout_report_count']} | "
            f"{len(item['comparisons'])} | {len(item['ratio_of_ratios'])} |"
        )
    lines.extend(
        [
            "",
            "The machine-readable evidence JSON contains every formula input, "
            "intermediate bound, diagnostic outlier index, holdout result, "
            "workflow run ID, revision identity, and host/CPU/image distribution.",
            "",
        ]
    )
    return budget, evidence, "\n".join(lines)


def derive_budget(args: argparse.Namespace) -> int:
    cohort = _load_json_object(args.cohort, "validated cohort")
    policy = _load_json_object(args.policy, "derivation policy")
    budget, evidence, markdown = derive_budget_documents(cohort, policy)
    outputs = {
        args.budget_output.resolve(),
        args.evidence_json_output.resolve(),
        args.evidence_markdown_output.resolve(),
    }
    if len(outputs) != 3:
        raise HarnessError("derive output paths must be distinct")
    atomic_write_json(args.budget_output, budget)
    atomic_write_json(args.evidence_json_output, evidence)
    args.evidence_markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_markdown_output.write_text(
        markdown,
        encoding="UTF-8",
        newline="\n",
    )
    print(
        f"Derived disabled candidate budget in {args.budget_output}; "
        f"holdout evidence is in {args.evidence_json_output}"
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
    dispatch_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_DISPATCH_TIMEOUT_SECONDS,
        help="wall-clock limit for dispatch and polling (default: 72 hours)",
    )
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
    derive_parser = sub.add_parser(
        "derive",
        help="derive a disabled candidate budget from a validated paired cohort",
    )
    derive_parser.add_argument("--cohort", type=Path, required=True)
    derive_parser.add_argument("--policy", type=Path, required=True)
    derive_parser.add_argument(
        "--budget-output",
        type=Path,
        default=Path("wasi-thread-budget.candidate.json"),
    )
    derive_parser.add_argument(
        "--evidence-json-output",
        type=Path,
        default=Path("wasi-thread-budget-evidence.json"),
    )
    derive_parser.add_argument(
        "--evidence-markdown-output",
        type=Path,
        default=Path("wasi-thread-budget-evidence.md"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.command == "dispatch":
            return dispatch(args)
        if args.command == "validate":
            return validate_cohort(args)
        return derive_budget(args)
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
