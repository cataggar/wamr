#!/usr/bin/env python3
"""Plan, dispatch, validate, and analyze the WASI thread duration-cross cohort."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import atomic_write_json, cache_key, collected_at, sha256_file
from bench_wasi_threads import HarnessError
from wasi_thread_cohort import (
    derive_one_sided_threshold,
    validate_derivation_policy,
)
from wasi_thread_duration_cross import (
    ARM_CPU_CLASS,
    ARMS,
    ACCEPTANCE_SURFACE,
    DESIGN_PROVENANCE,
    PARTITIONS,
    PLATFORM_CELLS,
    PLAN_KIND,
    PLAN_VERSION,
    SIDECAR_MAX_SAMPLES,
    TRUSTED_X86_RUNNER_NAME,
    X86_CPU_CLASS,
    arm_order,
    render_markdown,
    validate_report,
)


DISPATCH_KIND = "wasi-thread-duration-cross-dispatch"
COHORT_KIND = "wasi-thread-duration-cross-cohort"
CONCLUSION_KIND = "wasi-thread-duration-cross-conclusion"
DOWNLOAD_MANIFEST_KIND = "wasi-thread-duration-cross-download-manifest"
COHORT_SCHEMA_VERSION = 1
WORKFLOW = "wasi-thread-duration-cross.yml"
TAG_RE = re.compile(
    r"^wasi-thread-duration-cross-[A-Za-z0-9][A-Za-z0-9._-]*$"
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_TIMEOUT_SECONDS = 14 * 24 * 60 * 60
ACCEPTANCE_CEILING_LOG = 0.10
PRODUCTION_DERIVATION_POLICY = {
    "schema_version": 1,
    "kind": "wasi-thread-budget-derivation-policy",
    "minimum_reports_per_platform": 20,
    "rounding_cushion_log": 0.001,
    "engineering_policy_ceiling_log": {
        "comparison_elapsed_upper": 0.1,
        "comparison_throughput_lower": 0.1,
        "ratio_of_ratios_elapsed_upper": 0.1,
        "ratio_of_ratios_throughput_lower": 0.1,
    },
    "direct_candidate_single_infrastructure_max_delta_fraction": 0.02,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def require_sha(value: str, label: str) -> str:
    if SHA_RE.fullmatch(value) is None:
        raise HarnessError(
            f"{label} must be an immutable 40-character lowercase commit SHA"
        )
    return value


def expected_artifact_names(run_id: int) -> set[str]:
    return {
        f"wasi-thread-duration-cross-{platform}-{run_id}-1"
        for platform in PLATFORM_CELLS
    }


def parse_artifacts(value: Any, run_id: int) -> list[dict[str, Any]]:
    require(isinstance(value, dict), f"artifact response for run {run_id}")
    artifacts = value.get("artifacts")
    require(isinstance(artifacts, list), f"artifact list for run {run_id}")
    result = []
    for index, artifact in enumerate(artifacts):
        require(
            isinstance(artifact, dict)
            and isinstance(artifact.get("id"), int)
            and not isinstance(artifact["id"], bool)
            and artifact["id"] > 0
            and isinstance(artifact.get("name"), str)
            and bool(artifact["name"])
            and isinstance(artifact.get("size_in_bytes"), int)
            and not isinstance(artifact["size_in_bytes"], bool)
            and artifact["size_in_bytes"] > 0
            and artifact.get("expired") is False,
            f"artifact {index} for run {run_id}",
        )
        result.append(
            {
                "id": artifact["id"],
                "name": artifact["name"],
                "size_in_bytes": artifact["size_in_bytes"],
                "expired": False,
            }
        )
    require(
        {artifact["name"] for artifact in result}
        == expected_artifact_names(run_id)
        and len(result) == len(PLATFORM_CELLS),
        f"run {run_id} artifact membership",
    )
    return sorted(result, key=lambda artifact: artifact["name"])


def make_dispatch_plan(
    *,
    source_sha: str,
    workflow_ref: str,
    repository: str,
    cohort_id: str | None = None,
) -> dict[str, Any]:
    require_sha(source_sha, "source SHA")
    require(
        TAG_RE.fullmatch(workflow_ref) is not None,
        "duration-cross workflow ref must be an immutable "
        "wasi-thread-duration-cross-* tag",
    )
    resolved_cohort_id = cohort_id or uuid.uuid4().hex
    require(
        re.fullmatch(r"[0-9a-f]{32}", resolved_cohort_id) is not None,
        "cohort ID must be 32 lowercase hexadecimal characters",
    )
    assignments = []
    for sequence in range(1, 21):
        assignments.append(
            {
                "sequence": sequence,
                "partition": PARTITIONS[sequence],
                "arm_order_by_block": [
                    {
                        "block_index": block,
                        "arm_order": list(arm_order(sequence, block)),
                    }
                    for block in range(3)
                ],
            }
        )
    return {
        "schema_version": COHORT_SCHEMA_VERSION,
        "kind": DISPATCH_KIND,
        "created_at": collected_at(),
        "repository": repository,
        "workflow": WORKFLOW,
        "workflow_ref": workflow_ref,
        "workflow_head_sha": source_sha,
        "source_sha": source_sha,
        "cohort_id": resolved_cohort_id,
        "diagnostic_only": True,
        "production_budget_eligible": False,
        "runner_contract": {
            "maximum_workflows_in_flight": 1,
            "trusted_x86_runner_name": TRUSTED_X86_RUNNER_NAME,
            "x86_cpu_class": X86_CPU_CLASS,
            "arm_cpu_class": ARM_CPU_CLASS,
            "runner_worker_count_at_start": 1,
        },
        "requested_workflows": 20,
        "requested_reports": 40,
        "required_platforms": list(PLATFORM_CELLS),
        "split": {
            "method": "predeclared-sequence",
            "training_sequences": list(range(1, 17)),
            "holdout_sequences": list(range(17, 21)),
            "assignments": assignments,
        },
        "retry_policy": {
            "retries": 0,
            "replacements": 0,
            "exclusions": 0,
            "partial_reuse": False,
        },
        "runs": [],
    }


def validate_dispatch_plan(document: dict[str, Any], *, completed: bool) -> None:
    require(document.get("schema_version") == COHORT_SCHEMA_VERSION, "dispatch schema")
    require(document.get("kind") == DISPATCH_KIND, "dispatch kind")
    require(document.get("diagnostic_only") is True, "dispatch diagnostic identity")
    require(
        document.get("production_budget_eligible") is False,
        "dispatch production budget eligibility",
    )
    require(document.get("workflow") == WORKFLOW, "dispatch workflow identity")
    require_sha(str(document.get("source_sha", "")), "dispatch source SHA")
    require(
        document.get("workflow_head_sha") == document["source_sha"],
        "dispatch workflow/source identity",
    )
    require(
        TAG_RE.fullmatch(str(document.get("workflow_ref", ""))) is not None,
        "dispatch immutable tag",
    )
    require(
        re.fullmatch(r"[0-9a-f]{32}", str(document.get("cohort_id", ""))) is not None,
        "dispatch cohort ID",
    )
    require(
        document.get("requested_workflows") == 20
        and document.get("requested_reports") == 40,
        "dispatch exact cohort size",
    )
    require(
        document.get("required_platforms") == list(PLATFORM_CELLS),
        "dispatch platform set",
    )
    contract = document.get("runner_contract")
    require(
        isinstance(contract, dict)
        and contract.get("maximum_workflows_in_flight") == 1
        and contract.get("trusted_x86_runner_name") == TRUSTED_X86_RUNNER_NAME
        and contract.get("x86_cpu_class") == X86_CPU_CLASS
        and contract.get("arm_cpu_class") == ARM_CPU_CLASS
        and contract.get("runner_worker_count_at_start") == 1,
        "dispatch runner contract",
    )
    split = document.get("split")
    require(
        isinstance(split, dict)
        and split.get("training_sequences") == list(range(1, 17))
        and split.get("holdout_sequences") == list(range(17, 21)),
        "dispatch split",
    )
    assignments = split.get("assignments")
    require(isinstance(assignments, list) and len(assignments) == 20, "dispatch assignments")
    for sequence, item in enumerate(assignments, 1):
        require(
            item
            == {
                "sequence": sequence,
                "partition": PARTITIONS[sequence],
                "arm_order_by_block": [
                    {
                        "block_index": block,
                        "arm_order": list(arm_order(sequence, block)),
                    }
                    for block in range(3)
                ],
            },
            f"dispatch assignment {sequence}",
        )
    require(
        document.get("retry_policy")
        == {
            "retries": 0,
            "replacements": 0,
            "exclusions": 0,
            "partial_reuse": False,
        },
        "dispatch retry policy",
    )
    runs = document.get("runs")
    require(isinstance(runs, list), "dispatch runs")
    if completed:
        require(len(runs) == 20, "completed dispatch requires exactly 20 runs")
        for sequence, run in enumerate(runs, 1):
            require(
                isinstance(run, dict)
                and run.get("sequence") == sequence
                and run.get("partition") == PARTITIONS[sequence]
                and isinstance(run.get("run_id"), int)
                and run["run_id"] > 0
                and run.get("status") == "completed"
                and run.get("conclusion") == "success",
                f"dispatch run {sequence} is missing, partial, or unsuccessful",
            )
            require(run.get("attempt") == 1, f"dispatch run {sequence} was rerun")
            require(
                isinstance(run.get("artifacts"), list)
                and run["artifacts"]
                == sorted(run["artifacts"], key=lambda artifact: artifact["name"])
                and {artifact.get("name") for artifact in run["artifacts"]}
                == expected_artifact_names(run["run_id"])
                and len(run["artifacts"]) == len(PLATFORM_CELLS)
                and all(
                    isinstance(artifact.get("id"), int)
                    and artifact["id"] > 0
                    and isinstance(artifact.get("size_in_bytes"), int)
                    and artifact["size_in_bytes"] > 0
                    and artifact.get("expired") is False
                    for artifact in run["artifacts"]
                ),
                f"dispatch run {sequence} artifacts",
            )


def gh_json(arguments: list[str], timeout: float) -> Any:
    output = subprocess.check_output(
        ["gh", *arguments],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return json.loads(output)


def resolve_workflow_head(plan: dict[str, Any], timeout: float) -> str:
    response = gh_json(
        [
            "api",
            f"repos/{plan['repository']}/commits/{plan['workflow_ref']}",
        ],
        timeout,
    )
    require(isinstance(response, dict), "GitHub returned invalid commit identity")
    return require_sha(str(response.get("sha", "")), "workflow head SHA")


def matching_runs(
    plan: dict[str, Any], sequence: int, timeout: float
) -> list[dict[str, Any]]:
    title = (
        f"WASI thread duration-cross {plan['cohort_id']}-"
        f"{sequence}-{PARTITIONS[sequence]}"
    )
    runs = gh_json(
        [
            "run",
            "list",
            "--repo",
            plan["repository"],
            "--workflow",
            plan["workflow"],
            "--event",
            "workflow_dispatch",
            "--limit",
            "100",
            "--json",
            "databaseId,displayTitle,headSha,url,status,conclusion",
        ],
        timeout,
    )
    require(isinstance(runs, list), "GitHub returned invalid workflow run list")
    return [
        run
        for run in runs
        if run.get("displayTitle") == title
        and run.get("headSha") == plan["workflow_head_sha"]
    ]


def find_run(plan: dict[str, Any], sequence: int, timeout: float) -> dict[str, Any]:
    for _ in range(30):
        matches = matching_runs(plan, sequence, timeout)
        if len(matches) > 1:
            raise HarnessError(f"duplicate workflow runs found for sequence {sequence}")
        if matches:
            return matches[0]
        time.sleep(2)
    raise HarnessError(f"could not locate workflow run for sequence {sequence}")


def dispatch(plan_path: Path, output: Path, timeout_seconds: float) -> int:
    plan = json.loads(plan_path.read_text(encoding="UTF-8"))
    validate_dispatch_plan(plan, completed=False)
    require(plan["runs"] == [], "dispatch plan already contains run observations")
    deadline = time.monotonic() + timeout_seconds

    def remaining(context: str) -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise HarnessError(f"duration-cross dispatch timed out while {context}")
        return value

    resolved = resolve_workflow_head(plan, remaining("resolving immutable tag"))
    require(
        resolved == plan["source_sha"],
        f"immutable tag resolves to {resolved}, expected {plan['source_sha']}",
    )
    state = copy.deepcopy(plan)
    for sequence in range(1, 21):
        require(
            matching_runs(
                state, sequence, remaining(f"checking sequence {sequence}")
            )
            == [],
            f"sequence {sequence} already has a workflow run; refusing a retry",
        )
        subprocess.check_output(
            [
                "gh",
                "workflow",
                "run",
                state["workflow"],
                "--repo",
                state["repository"],
                "--ref",
                state["workflow_ref"],
                "-f",
                f"source_sha={state['source_sha']}",
                "-f",
                f"cohort_id={state['cohort_id']}",
                "-f",
                f"report_sequence={sequence}",
                "-f",
                f"partition={PARTITIONS[sequence]}",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=remaining(f"dispatching sequence {sequence}"),
        )
        run = find_run(state, sequence, remaining(f"finding sequence {sequence}"))
        run_id = run.get("databaseId")
        require(
            isinstance(run_id, int) and run_id > 0,
            f"sequence {sequence} returned invalid run ID",
        )
        record = {
            "sequence": sequence,
            "partition": PARTITIONS[sequence],
            "run_id": run_id,
            "url": run.get("url"),
            "status": run.get("status"),
            "conclusion": run.get("conclusion") or "",
            "attempt": None,
            "artifacts": [],
        }
        state["runs"].append(record)
        atomic_write_json(output, state)
        while record["status"] != "completed":
            time.sleep(min(60, remaining(f"waiting for sequence {sequence}")))
            viewed = gh_json(
                [
                    "run",
                    "view",
                    str(run_id),
                    "--repo",
                    state["repository"],
                    "--json",
                    "attempt,status,conclusion,headSha,url",
                ],
                remaining(f"polling sequence {sequence}"),
            )
            require(
                viewed.get("headSha") == state["workflow_head_sha"],
                f"sequence {sequence} workflow head changed",
            )
            record.update(
                {
                    "status": viewed.get("status"),
                    "conclusion": viewed.get("conclusion") or "",
                    "url": viewed.get("url"),
                    "attempt": viewed.get("attempt"),
                }
            )
            atomic_write_json(output, state)
        if record["attempt"] is None:
            viewed = gh_json(
                [
                    "run",
                    "view",
                    str(run_id),
                    "--repo",
                    state["repository"],
                    "--json",
                    "attempt,status,conclusion,headSha,url",
                ],
                remaining(f"validating sequence {sequence}"),
            )
            require(
                viewed.get("headSha") == state["workflow_head_sha"],
                f"sequence {sequence} workflow head changed",
            )
            record.update(
                {
                    "status": viewed.get("status"),
                    "conclusion": viewed.get("conclusion") or "",
                    "url": viewed.get("url"),
                    "attempt": viewed.get("attempt"),
                }
            )
            atomic_write_json(output, state)
        if record["conclusion"] != "success":
            raise HarnessError(
                f"sequence {sequence} concluded {record['conclusion']}; "
                "the observation is retained and will not be retried or replaced"
            )
        require(record["attempt"] == 1, f"sequence {sequence} was rerun")
        record["artifacts"] = parse_artifacts(
            gh_json(
                [
                    "api",
                    f"repos/{state['repository']}/actions/runs/{run_id}/artifacts",
                ],
                remaining(f"listing artifacts for sequence {sequence}"),
            ),
            run_id,
        )
        atomic_write_json(output, state)
    validate_dispatch_plan(state, completed=True)
    print(f"recorded 20 successful no-retry workflows in {output}")
    return 0


def download_artifacts(
    dispatch_path: Path,
    output_dir: Path,
    manifest_path: Path,
    timeout_seconds: float,
) -> int:
    dispatch_document = json.loads(dispatch_path.read_text(encoding="UTF-8"))
    validate_dispatch_plan(dispatch_document, completed=True)
    require(timeout_seconds > 0, "download timeout must be positive")
    output_dir = output_dir.resolve()
    require(
        manifest_path.resolve() != output_dir
        and output_dir not in manifest_path.resolve().parents,
        "download manifest must remain outside the exact artifact directory",
    )
    if os.getenv("GITHUB_ACTIONS") == "true":
        require(str(output_dir).startswith("/d/"), "downloads must remain under /d")
    require(
        not output_dir.exists() or not any(output_dir.iterdir()),
        "download directory must be absent or empty; partial downloads are not reusable",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    entries = []
    for run in dispatch_document["runs"]:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "artifact download timed out")
        viewed = gh_json(
            [
                "run",
                "view",
                str(run["run_id"]),
                "--repo",
                dispatch_document["repository"],
                "--json",
                "attempt,status,conclusion,headSha",
            ],
            remaining,
        )
        require(
            viewed.get("attempt") == 1
            and viewed.get("status") == "completed"
            and viewed.get("conclusion") == "success"
            and viewed.get("headSha") == dispatch_document["workflow_head_sha"],
            f"run {run['run_id']} changed or was rerun before download",
        )
        remaining = deadline - time.monotonic()
        require(remaining > 0, "artifact download timed out")
        remote_artifacts = parse_artifacts(
            gh_json(
                [
                    "api",
                    (
                        f"repos/{dispatch_document['repository']}/actions/runs/"
                        f"{run['run_id']}/artifacts"
                    ),
                ],
                remaining,
            ),
            run["run_id"],
        )
        require(
            remote_artifacts == run["artifacts"],
            f"run {run['run_id']} artifacts changed before download",
        )
        for artifact in run["artifacts"]:
            artifact_dir = output_dir / artifact["name"]
            remaining = deadline - time.monotonic()
            require(remaining > 0, "artifact download timed out")
            subprocess.check_output(
                [
                    "gh",
                    "run",
                    "download",
                    str(run["run_id"]),
                    "--repo",
                    dispatch_document["repository"],
                    "--name",
                    artifact["name"],
                    "--dir",
                    str(artifact_dir),
                ],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=remaining,
            )
            files = sorted(
                path.relative_to(artifact_dir).as_posix()
                for path in artifact_dir.rglob("*")
                if path.is_file()
            )
            require(
                files == ["report.json", "report.md"],
                f"artifact {artifact['name']} is partial or contains unexpected files",
            )
            platform = artifact["name"].removeprefix(
                "wasi-thread-duration-cross-"
            ).removesuffix(f"-{run['run_id']}-1")
            require(platform in PLATFORM_CELLS, f"artifact {artifact['name']} platform")
            entries.append(
                {
                    "sequence": run["sequence"],
                    "partition": run["partition"],
                    "run_id": run["run_id"],
                    "workflow_run_attempt": 1,
                    "platform": platform,
                    "artifact_id": artifact["id"],
                    "artifact_name": artifact["name"],
                    "artifact_size_in_bytes": artifact["size_in_bytes"],
                    "report_json": {
                        "path": (
                            Path(artifact["name"]) / "report.json"
                        ).as_posix(),
                        "sha256": sha256_file(artifact_dir / "report.json"),
                    },
                    "report_markdown": {
                        "path": (
                            Path(artifact["name"]) / "report.md"
                        ).as_posix(),
                        "sha256": sha256_file(artifact_dir / "report.md"),
                    },
                }
            )
    manifest = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "kind": DOWNLOAD_MANIFEST_KIND,
        "downloaded_at": collected_at(),
        "dispatch_sha256": cache_key(dispatch_document),
        "cohort_id": dispatch_document["cohort_id"],
        "source_sha": dispatch_document["source_sha"],
        "entries": sorted(
            entries, key=lambda item: (item["sequence"], item["platform"])
        ),
    }
    atomic_write_json(manifest_path, manifest)
    print(f"downloaded 40 exact diagnostic artifacts into {output_dir}")
    return 0


def report_identity(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report["metadata"]
    return {
        "source_revision": metadata["source_revision"],
        "fixture_set_sha256": metadata["fixture_set_sha256"],
        "plan_kind": report["plan"]["kind"],
        "plan_version": report["plan"]["version"],
        "artifact_policy": report["plan"]["artifact_policy"],
    }


def markdown_output_path(output: Path) -> Path:
    return output.with_suffix(".md")


def render_cohort_markdown(cohort: dict[str, Any]) -> str:
    lines = [
        "# WASI thread duration-cross validated cohort",
        "",
        "> Non-authoritative diagnostic evidence only. No production budget is emitted.",
        "",
        f"- Cohort: `{cohort['dispatch']['cohort_id']}`",
        f"- Source: `{cohort['dispatch']['source_sha']}`",
        f"- Download manifest: `{cohort['download_manifest_sha256']}`",
        f"- Workflows: `{cohort['dispatch']['requested_workflows']}`",
        f"- Reports: `{len(cohort['observations'])}`",
        "- Exclusions/retries/replacements: `0 / 0 / 0`",
        "",
        "| Sequence | Partition | Platform | Run | Artifact identity | JSON SHA-256 | Markdown SHA-256 |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for item in cohort["observations"]:
        lines.append(
            f"| {item['sequence']} | `{item['partition']}` | "
            f"`{item['platform']}` | {item['run_id']} | "
            f"`{item['artifact_identity_sha256']}` | "
            f"`{item['report_file_sha256']}` | "
            f"`{item['report_markdown_sha256']}` |"
        )
    return "\n".join(lines)


def validate_download_manifest(
    manifest: dict[str, Any],
    dispatch_document: dict[str, Any],
    input_dir: Path,
) -> list[Path]:
    require(
        manifest.get("schema_version") == COHORT_SCHEMA_VERSION
        and manifest.get("kind") == DOWNLOAD_MANIFEST_KIND
        and manifest.get("dispatch_sha256") == cache_key(dispatch_document)
        and manifest.get("cohort_id") == dispatch_document["cohort_id"]
        and manifest.get("source_sha") == dispatch_document["source_sha"],
        "download manifest identity",
    )
    entries = manifest.get("entries")
    require(
        isinstance(entries, list) and len(entries) == 40,
        "download manifest membership",
    )
    expected = {
        (sequence, platform)
        for sequence in range(1, 21)
        for platform in PLATFORM_CELLS
    }
    seen = set()
    report_paths = []
    for entry in entries:
        require(isinstance(entry, dict), "download manifest entry")
        key = (entry.get("sequence"), entry.get("platform"))
        require(key in expected and key not in seen, f"download manifest entry {key}")
        seen.add(key)
        run = dispatch_document["runs"][entry["sequence"] - 1]
        require(
            entry.get("partition") == PARTITIONS[entry["sequence"]]
            and entry.get("run_id") == run["run_id"]
            and entry.get("workflow_run_attempt") == 1,
            f"download manifest entry {key} run identity",
        )
        artifact_name = (
            f"wasi-thread-duration-cross-{entry['platform']}-{run['run_id']}-1"
        )
        artifact = next(
            (
                item
                for item in run["artifacts"]
                if item["name"] == artifact_name
            ),
            None,
        )
        require(
            artifact is not None
            and entry.get("artifact_id") == artifact["id"]
            and entry.get("artifact_name") == artifact_name
            and entry.get("artifact_size_in_bytes") == artifact["size_in_bytes"],
            f"download manifest entry {key} artifact identity",
        )
        for field, filename in (
            ("report_json", "report.json"),
            ("report_markdown", "report.md"),
        ):
            item = entry.get(field)
            require(
                isinstance(item, dict)
                and item.get("path")
                == (Path(artifact_name) / filename).as_posix()
                and isinstance(item.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
                f"download manifest entry {key} {field}",
            )
            path = input_dir / item["path"]
            require(path.is_file(), f"downloaded file {path} is missing")
            require(
                sha256_file(path) == item["sha256"],
                f"downloaded file {path} hash mismatch",
            )
        report_paths.append(input_dir / entry["report_json"]["path"])
    require(seen == expected, "download manifest has partial or unexpected membership")
    require(
        sorted(
            path.relative_to(input_dir).as_posix()
            for path in input_dir.rglob("*")
            if path.is_file()
        )
        == sorted(
            item[field]["path"]
            for item in entries
            for field in ("report_json", "report_markdown")
        ),
        "download directory contains partial or unexpected files",
    )
    return sorted(report_paths)


def validate_cohort(
    input_dir: Path,
    dispatch_path: Path,
    manifest_path: Path,
    output: Path,
) -> int:
    dispatch_document = json.loads(dispatch_path.read_text(encoding="UTF-8"))
    validate_dispatch_plan(dispatch_document, completed=True)
    input_dir = input_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="UTF-8"))
    paths = validate_download_manifest(manifest, dispatch_document, input_dir)
    reports = []
    seen = set()
    source_identity = None
    platform_identity: dict[str, dict[str, Any]] = {}
    platform_counts: Counter[str] = Counter()
    partition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    run_platforms: dict[str, set[str]] = defaultdict(set)
    expected_run_ids = {
        item["sequence"]: str(item["run_id"]) for item in dispatch_document["runs"]
    }
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="UTF-8"))
            validate_report(report)
        except (json.JSONDecodeError, HarnessError, ValueError) as exc:
            raise HarnessError(f"{path}: invalid duration-cross report: {exc}") from exc
        metadata = report["metadata"]
        sequence = metadata["report_sequence"]
        platform = metadata["platform_id"]
        run_id = metadata["workflow_run_id"]
        run_attempt = metadata["workflow_run_attempt"]
        key = (sequence, platform)
        require(key not in seen, f"duplicate report for sequence/platform {key}")
        require(
            metadata["cohort_id"] == dispatch_document["cohort_id"],
            f"{path}: report belongs to unexpected cohort",
        )
        require(
            run_id == expected_run_ids[sequence],
            f"{path}: run ID {run_id!r} does not match predeclared sequence "
            f"{sequence} run {expected_run_ids[sequence]}",
        )
        require(run_attempt == "1", f"{path}: workflow reruns are forbidden")
        artifact_name = (
            f"wasi-thread-duration-cross-{platform}-{run_id}-{run_attempt}"
        )
        require(
            artifact_name in path.parts,
            f"{path}: report is not under its predeclared artifact {artifact_name}",
        )
        require(
            artifact_name
            in {
                artifact["name"]
                for artifact in dispatch_document["runs"][sequence - 1]["artifacts"]
            },
            f"{path}: report artifact is absent from the dispatch record",
        )
        markdown_path = path.with_name("report.md")
        require(markdown_path.is_file(), f"{path}: paired report.md is missing")
        require(
            markdown_path.read_text(encoding="UTF-8")
            == render_markdown(report) + "\n",
            f"{markdown_path}: Markdown does not match report.json",
        )
        require(
            metadata["partition"] == PARTITIONS[sequence],
            f"{path}: report changed predeclared partition",
        )
        require(
            metadata["source_revision"]["commit"] == dispatch_document["source_sha"],
            f"{path}: report source differs from immutable dispatch source",
        )
        cpu = metadata["host"]["cpu"]
        if platform == "ubuntu-22.04-x86_64":
            require(cpu == X86_CPU_CLASS, f"{path}: unexpected x86 CPU class {cpu!r}")
            require(
                metadata["host"]["runner_name"] == TRUSTED_X86_RUNNER_NAME,
                f"{path}: trusted x86 runner identity changed",
            )
        else:
            require(
                ARM_CPU_CLASS in cpu,
                f"{path}: Arm CPU is not {ARM_CPU_CLASS} class: {cpu!r}",
            )
        current_identity = report_identity(report)
        current_source = current_identity["source_revision"]
        if source_identity is None:
            source_identity = current_source
        else:
            require(
                current_source == source_identity,
                f"{path}: source identity differs within cohort",
            )
        if platform not in platform_identity:
            platform_identity[platform] = current_identity
        else:
            require(
                current_identity == platform_identity[platform],
                f"{path}: source/fixture/plan identity differs within {platform}",
            )
        seen.add(key)
        platform_counts[platform] += 1
        partition_counts[platform][metadata["partition"]] += 1
        run_platforms[run_id].add(platform)
        reports.append(
            {
                "path": str(path),
                "report_sha256": cache_key(report),
                "report_file_sha256": sha256_file(path),
                "report_markdown_sha256": sha256_file(markdown_path),
                "artifact_identity_sha256": cache_key(
                    metadata["artifact_identity"]["baseline"]
                ),
                "run_id": run_id,
                "workflow_run_attempt": run_attempt,
                "artifact_name": artifact_name,
                "sequence": sequence,
                "partition": metadata["partition"],
                "platform": platform,
                "cpu_class": cpu,
                "host_fingerprint_sha256": metadata["host_pair"][
                    "host_fingerprint_sha256"
                ],
                "plan_sha256": metadata["plan_sha256"],
                "plan_identity_sha256": metadata["plan_identity_sha256"],
                "telemetry_sidecar": {
                    "requested": report["telemetry_sidecar"]["requested"],
                    "available": report["telemetry_sidecar"]["available"],
                    "cpu": report["telemetry_sidecar"]["cpu"],
                    "observed_cpus": report["telemetry_sidecar"][
                        "observed_cpus"
                    ],
                    "interval_seconds": report["telemetry_sidecar"][
                        "interval_seconds"
                    ],
                    "reason": report["telemetry_sidecar"]["reason"],
                    "sample_count": len(report["telemetry_sidecar"]["samples"]),
                },
                "summaries": report["summaries"],
            }
        )
    expected_keys = {
        (sequence, platform)
        for sequence in range(1, 21)
        for platform in PLATFORM_CELLS
    }
    require(seen == expected_keys, "cohort has missing, partial, or unexpected membership")
    require(
        all(platform_counts[platform] == 20 for platform in PLATFORM_CELLS),
        "cohort must contain exactly 20 reports per platform",
    )
    require(
        all(
            partition_counts[platform] == Counter({"training": 16, "holdout": 4})
            for platform in PLATFORM_CELLS
        ),
        "cohort partition counts must be 16 training and 4 holdout per platform",
    )
    require(
        all(platforms == set(PLATFORM_CELLS) for platforms in run_platforms.values())
        and len(run_platforms) == 20,
        "cohort contains a partial cross-platform workflow",
    )
    cohort = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "kind": COHORT_KIND,
        "authoritative": False,
        "production_budget": None,
        "validated_at": collected_at(),
        "dispatch": dispatch_document,
        "download_manifest_sha256": cache_key(manifest),
        "identity": {
            "source_revision": source_identity,
            "platforms": platform_identity,
        },
        "platform_counts": dict(platform_counts),
        "partition_counts": {
            platform: dict(partition_counts[platform])
            for platform in PLATFORM_CELLS
        },
        "excluded_observations": [],
        "observations": sorted(
            reports, key=lambda item: (item["sequence"], item["platform"])
        ),
    }
    validate_cohort_document(cohort)
    atomic_write_json(output, cohort)
    markdown_output_path(output).write_text(
        render_cohort_markdown(cohort) + "\n",
        encoding="UTF-8",
        newline="\n",
    )
    print(f"validated 40 duration-cross reports in {output}")
    return 0


def validate_cohort_document(cohort: dict[str, Any]) -> None:
    require(cohort.get("schema_version") == COHORT_SCHEMA_VERSION, "cohort schema")
    require(cohort.get("kind") == COHORT_KIND, "cohort kind")
    require(cohort.get("authoritative") is False, "cohort authoritative identity")
    require(cohort.get("production_budget") is None, "cohort production budget")
    require(cohort.get("excluded_observations") == [], "cohort exclusions")
    require(
        isinstance(cohort.get("download_manifest_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", cohort["download_manifest_sha256"]
        )
        is not None,
        "cohort download manifest hash",
    )
    dispatch_document = cohort.get("dispatch")
    require(isinstance(dispatch_document, dict), "cohort dispatch")
    validate_dispatch_plan(dispatch_document, completed=True)
    observations = cohort.get("observations")
    require(isinstance(observations, list) and len(observations) == 40, "cohort observations")
    require(
        cohort.get("platform_counts")
        == {platform: 20 for platform in PLATFORM_CELLS},
        "cohort platform counts",
    )
    require(
        cohort.get("partition_counts")
        == {
            platform: {"training": 16, "holdout": 4}
            for platform in PLATFORM_CELLS
        },
        "cohort partition counts",
    )
    identity = cohort.get("identity")
    require(
        isinstance(identity, dict)
        and isinstance(identity.get("source_revision"), dict)
        and identity["source_revision"].get("commit")
        == dispatch_document["source_sha"]
        and isinstance(identity.get("platforms"), dict)
        and set(identity["platforms"]) == set(PLATFORM_CELLS),
        "cohort identity",
    )
    expected_run_ids = {
        run["sequence"]: str(run["run_id"]) for run in dispatch_document["runs"]
    }
    seen = set()
    report_hashes = set()
    for observation in observations:
        require(isinstance(observation, dict), "cohort observation object")
        key = (observation.get("sequence"), observation.get("platform"))
        require(key not in seen, f"duplicate cohort observation {key}")
        seen.add(key)
        require(
            observation.get("partition") == PARTITIONS.get(observation.get("sequence")),
            f"cohort observation {key} partition",
        )
        sequence, platform = key
        require(
            sequence in PARTITIONS and platform in PLATFORM_CELLS,
            f"cohort observation {key} membership",
        )
        require(
            observation.get("run_id") == expected_run_ids.get(sequence)
            and observation.get("workflow_run_attempt") == "1",
            f"cohort observation {key} run identity",
        )
        require(
            observation.get("artifact_name")
            == f"wasi-thread-duration-cross-{platform}-{observation.get('run_id')}-1",
            f"cohort observation {key} artifact identity",
        )
        for field in (
            "report_sha256",
            "report_file_sha256",
            "report_markdown_sha256",
            "artifact_identity_sha256",
            "host_fingerprint_sha256",
            "plan_sha256",
            "plan_identity_sha256",
        ):
            require(
                isinstance(observation.get(field), str)
                and re.fullmatch(r"[0-9a-f]{64}", observation[field]) is not None,
                f"cohort observation {key} {field}",
            )
        require(
            observation["report_sha256"] not in report_hashes,
            f"duplicate cohort report hash {observation['report_sha256']}",
        )
        report_hashes.add(observation["report_sha256"])
        sidecar = observation.get("telemetry_sidecar")
        require(
            isinstance(sidecar, dict)
            and sidecar.get("requested") is True
            and isinstance(sidecar.get("available"), bool)
            and isinstance(sidecar.get("observed_cpus"), list)
            and all(
                isinstance(cpu, int)
                and not isinstance(cpu, bool)
                and cpu >= 0
                for cpu in sidecar["observed_cpus"]
            )
            and len(set(sidecar["observed_cpus"]))
            == len(sidecar["observed_cpus"])
            and sidecar.get("interval_seconds") == 1
            and isinstance(sidecar.get("sample_count"), int)
            and not isinstance(sidecar["sample_count"], bool)
            and 0 <= sidecar["sample_count"] <= SIDECAR_MAX_SAMPLES
            and (
                sidecar["available"]
                or isinstance(sidecar.get("reason"), str)
                and bool(sidecar["reason"])
            ),
            f"cohort observation {key} telemetry sidecar",
        )
        require(
            isinstance(observation.get("summaries"), dict),
            f"cohort observation {key} summaries",
        )
        expected_comparisons = {
            (
                cell["pair_key"],
                condition,
                (
                    "spawn-join-lifecycle"
                    if cell["workload"] == "spawn-join"
                    else "steady-state-kernel"
                ),
            )
            for cell in PLATFORM_CELLS[platform]
            for condition in (cell["left"], cell["right"])
        }
        expected_ratios = {
            (cell["pair_key"], cell["left"], cell["right"])
            for cell in PLATFORM_CELLS[platform]
        }
        for arm in ARMS:
            require(
                set(indexed_metrics(observation, "comparisons", arm))
                == expected_comparisons
                and set(indexed_metrics(observation, "ratio_of_ratios", arm))
                == expected_ratios,
                f"cohort observation {key}/{arm} summary coverage",
            )
    require(
        seen
        == {
            (sequence, platform)
            for sequence in range(1, 21)
            for platform in PLATFORM_CELLS
        },
        "cohort observation membership",
    )


def metric_estimate(item: dict[str, Any], field: str) -> float:
    value = item.get(field)
    require(isinstance(value, dict), f"metric {field}")
    median = value.get("median")
    require(
        isinstance(median, (int, float))
        and not isinstance(median, bool)
        and math.isfinite(median)
        and median > 0,
        f"metric {field} median",
    )
    return float(median)


def indexed_metrics(
    observation: dict[str, Any], collection: str, arm: str
) -> dict[tuple[str, ...], dict[str, Any]]:
    items = observation["summaries"].get(collection)
    require(isinstance(items, list), f"{collection} summaries")
    result = {}
    for item in items:
        require(
            isinstance(item, dict) and item.get("arm") in ARMS,
            f"{collection} item",
        )
        if item.get("arm") != arm:
            continue
        if collection == "comparisons":
            key = (item.get("pair_key"), item.get("condition"), item.get("metric_kind"))
        else:
            key = (item.get("pair_key"), item.get("left"), item.get("right"))
        require(all(isinstance(value, str) and value for value in key), f"{collection} key")
        require(key not in result, f"duplicate {arm} {collection} key {key}")
        result[key] = item
    return result


def analyze_metric(
    *,
    training: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    throughput_field: str,
    elapsed_field: str,
    cushion: float,
    throughput_ceiling: float,
    elapsed_ceiling: float,
    label: str,
    selected: bool,
) -> tuple[dict[str, Any], list[str]]:
    throughput_rule = derive_one_sided_threshold(
        [metric_estimate(item, throughput_field) for item in training],
        "lower",
        cushion,
        1e9,
        f"{label} throughput",
    )
    elapsed_rule = derive_one_sided_threshold(
        [metric_estimate(item, elapsed_field) for item in training],
        "upper",
        cushion,
        1e9,
        f"{label} elapsed",
    )
    throughput_rule["engineering_policy_ceiling_log"] = throughput_ceiling
    elapsed_rule["engineering_policy_ceiling_log"] = elapsed_ceiling
    throughput_rule["engineering_ceiling_passed"] = (
        throughput_rule["final_bound_log"] <= throughput_ceiling
    )
    elapsed_rule["engineering_ceiling_passed"] = (
        elapsed_rule["final_bound_log"] <= elapsed_ceiling
    )
    holdout_results = []
    failures = []
    if selected and not throughput_rule["engineering_ceiling_passed"]:
        failures.append(
            f"{label} throughput training final bound "
            f"{throughput_rule['final_bound_log']:.17g} exceeds diagnostic "
            f"ceiling {throughput_ceiling:.17g}"
        )
    if selected and not elapsed_rule["engineering_ceiling_passed"]:
        failures.append(
            f"{label} elapsed training final bound "
            f"{elapsed_rule['final_bound_log']:.17g} exceeds diagnostic "
            f"ceiling {elapsed_ceiling:.17g}"
        )
    for item in holdout:
        throughput = metric_estimate(item, throughput_field)
        elapsed = metric_estimate(item, elapsed_field)
        passed = (
            throughput >= throughput_rule["threshold_ratio"]
            and elapsed <= elapsed_rule["threshold_ratio"]
        )
        holdout_results.append(
            {
                "sequence": item["_sequence"],
                "run_id": item["_run_id"],
                "throughput_ratio": throughput,
                "elapsed_ratio": elapsed,
                "passed": passed,
            }
        )
        if selected and not passed:
            failures.append(
                f"{label} holdout sequence {item['_sequence']} failed: "
                f"throughput={throughput:.17g} "
                f"(min={throughput_rule['threshold_ratio']:.17g}), "
                f"elapsed={elapsed:.17g} "
                f"(max={elapsed_rule['threshold_ratio']:.17g})"
            )
    return (
        {
            "selected": selected,
            "training_count": len(training),
            "holdout_count": len(holdout),
            "throughput_rule": throughput_rule,
            "elapsed_rule": elapsed_rule,
            "holdout": holdout_results,
        },
        failures,
    )


def render_conclusion_markdown(conclusion: dict[str, Any]) -> str:
    lines = [
        "# WASI thread duration-cross diagnostic conclusion",
        "",
        "> Non-authoritative diagnostic only. This conclusion cannot authorize a production budget.",
        "",
        f"- Result: `{'PASS' if conclusion['passed'] else 'FAIL'}`",
        f"- Cohort SHA-256: `{conclusion['source_cohort_sha256']}`",
        f"- Production policy SHA-256: `{conclusion['policy_sha256']}`",
        "- Selected arm: `doubled`",
        "- Retained non-selecting arm: `current`",
        "- Training/holdout sequences: `1-16 / 17-20`",
        "- Frozen engineering ceiling: `0.10`",
        "",
    ]
    if conclusion["failures"]:
        lines += ["## Failures", ""]
        lines.extend(f"- {failure}" for failure in conclusion["failures"])
        lines.append("")
    lines += [
        "## Evaluated implicated checks",
        "",
        "| Platform | Arm | Collection | Check | Training bound pass | Holdouts pass |",
        "|---|---|---|---|---|---|",
    ]
    for platform, platform_evidence in conclusion["evidence"].items():
        for arm, arm_evidence in platform_evidence.items():
            for collection in ("comparisons", "ratio_of_ratios"):
                for item in arm_evidence[collection]:
                    training_passed = (
                        item["throughput_rule"]["engineering_ceiling_passed"]
                        and item["elapsed_rule"]["engineering_ceiling_passed"]
                    )
                    holdouts_passed = all(
                        result["passed"] for result in item["holdout"]
                    )
                    lines.append(
                        f"| `{platform}` | `{arm}` | `{collection}` | "
                        f"`{' / '.join(item['key'])}` | "
                        f"`{training_passed}` | `{holdouts_passed}` |"
                    )
    return "\n".join(lines)


def analyze(cohort_path: Path, policy_path: Path, output: Path) -> int:
    cohort = json.loads(cohort_path.read_text(encoding="UTF-8"))
    validate_cohort_document(cohort)
    policy = validate_derivation_policy(
        json.loads(policy_path.read_text(encoding="UTF-8"))
    )
    require(
        policy == PRODUCTION_DERIVATION_POLICY,
        "duration-cross analysis requires the unchanged production derivation policy",
    )
    cushion = policy["rounding_cushion_log"]
    ceilings = policy["engineering_policy_ceiling_log"]
    evidence: dict[str, Any] = {}
    failures: list[str] = []
    for platform in PLATFORM_CELLS:
        observations = [
            item for item in cohort["observations"] if item["platform"] == platform
        ]
        platform_evidence: dict[str, Any] = {}
        for arm in ARMS:
            selected = arm == "doubled"
            arm_evidence = {"comparisons": [], "ratio_of_ratios": []}
            indexed_by_observation = []
            for observation in observations:
                indexed_by_observation.append(
                    {
                        "observation": observation,
                        "comparisons": indexed_metrics(observation, "comparisons", arm),
                        "ratio_of_ratios": indexed_metrics(
                            observation, "ratio_of_ratios", arm
                        ),
                    }
                )
            expected_comparison_keys = set(
                ACCEPTANCE_SURFACE[platform]["comparisons"]
            )
            expected_ratio_keys = set(
                ACCEPTANCE_SURFACE[platform]["ratio_of_ratios"]
            )
            require(
                all(
                    expected_comparison_keys <= set(item["comparisons"])
                    and expected_ratio_keys <= set(item["ratio_of_ratios"])
                    for item in indexed_by_observation
                ),
                f"{platform}/{arm} omits an implicated v20 metric",
            )
            for collection, keys, throughput_field, elapsed_field, throughput_ceiling, elapsed_ceiling in (
                (
                    "comparisons",
                    expected_comparison_keys,
                    "throughput_candidate_over_baseline",
                    "elapsed_candidate_over_baseline",
                    ceilings["comparison_throughput_lower"],
                    ceilings["comparison_elapsed_upper"],
                ),
                (
                    "ratio_of_ratios",
                    expected_ratio_keys,
                    "throughput_ratio_of_ratios",
                    "elapsed_ratio_of_ratios",
                    ceilings["ratio_of_ratios_throughput_lower"],
                    ceilings["ratio_of_ratios_elapsed_upper"],
                ),
            ):
                for key in sorted(keys):
                    values = []
                    for indexed in indexed_by_observation:
                        observation = indexed["observation"]
                        values.append(
                            {
                                **indexed[collection][key],
                                "_sequence": observation["sequence"],
                                "_run_id": observation["run_id"],
                            }
                        )
                    training = [
                        item for item in values if PARTITIONS[item["_sequence"]] == "training"
                    ]
                    holdout = [
                        item for item in values if PARTITIONS[item["_sequence"]] == "holdout"
                    ]
                    require(
                        len(training) == 16 and len(holdout) == 4,
                        f"{platform}/{arm}/{collection}/{key} membership",
                    )
                    metric_evidence, metric_failures = analyze_metric(
                        training=training,
                        holdout=holdout,
                        throughput_field=throughput_field,
                        elapsed_field=elapsed_field,
                        cushion=cushion,
                        throughput_ceiling=throughput_ceiling,
                        elapsed_ceiling=elapsed_ceiling,
                        label=f"{platform} {arm} {collection} {key}",
                        selected=selected,
                    )
                    arm_evidence[collection].append(
                        {"key": list(key), **metric_evidence}
                    )
                    failures.extend(metric_failures)
            platform_evidence[arm] = arm_evidence
        evidence[platform] = platform_evidence
    conclusion = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "kind": CONCLUSION_KIND,
        "authoritative": False,
        "production_budget": None,
        "source_cohort_sha256": cache_key(cohort),
        "policy_sha256": cache_key(policy),
        "diagnostic_plan": {
            "kind": PLAN_KIND,
            "version": PLAN_VERSION,
            "reviewed_design": copy.deepcopy(DESIGN_PROVENANCE),
            "selected_arm": "doubled",
            "retained_nonselecting_arm": "current",
            "training_sequences": list(range(1, 17)),
            "holdout_sequences": list(range(17, 21)),
            "engineering_ceiling_log": ACCEPTANCE_CEILING_LOG,
            "estimator": "median-of-four-position-geometric-means",
            "no_retry_replacement_exclusion_adaptation": True,
        },
        "passed": not failures,
        "failures": failures,
        "evidence": evidence,
    }
    atomic_write_json(output, conclusion)
    markdown_output_path(output).write_text(
        render_conclusion_markdown(conclusion) + "\n",
        encoding="UTF-8",
        newline="\n",
    )
    if failures:
        raise HarnessError(
            "duration-cross diagnostic acceptance failed:\n" + "\n".join(failures)
        )
    print(
        f"duration-cross diagnostic passed; diagnostic-only conclusion is in {output}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--source-sha", required=True)
    plan_parser.add_argument("--workflow-ref", required=True)
    plan_parser.add_argument("--repository", default="cataggar/wamr")
    plan_parser.add_argument("--cohort-id")
    plan_parser.add_argument(
        "--output", type=Path, default=Path("wasi-thread-duration-cross-dispatch.json")
    )
    dispatch_parser = sub.add_parser("dispatch")
    dispatch_parser.add_argument("--plan", type=Path, required=True)
    dispatch_parser.add_argument(
        "--output", type=Path, default=Path("wasi-thread-duration-cross-dispatch.completed.json")
    )
    dispatch_parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    download_parser = sub.add_parser("download")
    download_parser.add_argument("--dispatch", type=Path, required=True)
    download_parser.add_argument("--output-dir", type=Path, required=True)
    download_parser.add_argument("--manifest", type=Path, required=True)
    download_parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--input-dir", type=Path, required=True)
    validate_parser.add_argument("--dispatch", type=Path, required=True)
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument(
        "--output", type=Path, default=Path("wasi-thread-duration-cross-cohort.json")
    )
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--cohort", type=Path, required=True)
    analyze_parser.add_argument("--policy", type=Path, required=True)
    analyze_parser.add_argument(
        "--output", type=Path, default=Path("wasi-thread-duration-cross-conclusion.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.command == "plan":
            document = make_dispatch_plan(
                source_sha=args.source_sha,
                workflow_ref=args.workflow_ref,
                repository=args.repository,
                cohort_id=args.cohort_id,
            )
            validate_dispatch_plan(document, completed=False)
            atomic_write_json(args.output, document)
            print(f"wrote predeclared 20-workflow plan to {args.output}")
            return 0
        if args.command == "dispatch":
            return dispatch(args.plan, args.output, args.timeout_seconds)
        if args.command == "download":
            return download_artifacts(
                args.dispatch,
                args.output_dir,
                args.manifest,
                args.timeout_seconds,
            )
        if args.command == "validate":
            return validate_cohort(
                args.input_dir,
                args.dispatch,
                args.manifest,
                args.output,
            )
        return analyze(args.cohort, args.policy, args.output)
    except (
        HarnessError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
