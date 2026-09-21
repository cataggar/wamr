#!/usr/bin/env python3
"""Plan, dispatch, validate, and analyze the WASI thread duration-cross cohort."""

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

from benchmark_schema import atomic_write_json, cache_key, collected_at
from bench_wasi_threads import HarnessError
from wasi_thread_cohort import (
    derive_one_sided_threshold,
    validate_derivation_policy,
)
from wasi_thread_duration_cross import (
    ARM_CPU_CLASS,
    ARMS,
    KIND as REPORT_KIND,
    PARTITIONS,
    PLATFORM_CELLS,
    PLAN_KIND,
    PLAN_VERSION,
    REPORT_SCHEMA_VERSION,
    TRUSTED_X86_RUNNER_NAME,
    X86_CPU_CLASS,
    arm_order,
    validate_report,
)


DISPATCH_KIND = "wasi-thread-duration-cross-dispatch"
COHORT_KIND = "wasi-thread-duration-cross-cohort"
CONCLUSION_KIND = "wasi-thread-duration-cross-conclusion"
COHORT_SCHEMA_VERSION = 1
WORKFLOW = "wasi-thread-duration-cross.yml"
TAG_RE = re.compile(
    r"^wasi-thread-duration-cross-[A-Za-z0-9][A-Za-z0-9._-]*$"
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_TIMEOUT_SECONDS = 14 * 24 * 60 * 60
ACCEPTANCE_CEILING_LOG = 0.10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def require_sha(value: str, label: str) -> str:
    if SHA_RE.fullmatch(value) is None:
        raise HarnessError(
            f"{label} must be an immutable 40-character lowercase commit SHA"
        )
    return value


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


def find_run(plan: dict[str, Any], sequence: int, timeout: float) -> dict[str, Any]:
    title = (
        f"WASI thread duration-cross {plan['cohort_id']}-"
        f"{sequence}-{PARTITIONS[sequence]}"
    )
    for _ in range(30):
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
        matches = [
            run
            for run in runs
            if run.get("displayTitle") == title
            and run.get("headSha") == plan["workflow_head_sha"]
        ]
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
                    "status,conclusion,headSha,url",
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
                }
            )
            atomic_write_json(output, state)
        if record["conclusion"] != "success":
            raise HarnessError(
                f"sequence {sequence} concluded {record['conclusion']}; "
                "the observation is retained and will not be retried or replaced"
            )
    validate_dispatch_plan(state, completed=True)
    print(f"recorded 20 successful no-retry workflows in {output}")
    return 0


def report_identity(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report["metadata"]
    return {
        "source_revision": metadata["source_revision"],
        "artifact_identity": metadata["artifact_identity"],
        "fixture_set_sha256": metadata["fixture_set_sha256"],
        "plan_kind": report["plan"]["kind"],
        "plan_version": report["plan"]["version"],
    }


def validate_cohort(input_dir: Path, dispatch_path: Path, output: Path) -> int:
    dispatch_document = json.loads(dispatch_path.read_text(encoding="UTF-8"))
    validate_dispatch_plan(dispatch_document, completed=True)
    paths = sorted(input_dir.rglob("report.json"))
    require(len(paths) == 40, f"expected exactly 40 report.json files, found {len(paths)}")
    reports = []
    seen = set()
    source_identity = None
    platform_identity: dict[str, dict[str, Any]] = {}
    platform_counts: Counter[str] = Counter()
    partition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    run_platforms: dict[str, set[str]] = defaultdict(set)
    sequence_runs: dict[int, str] = {}
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
                f"{path}: artifact/plan identity differs within {platform}",
            )
        seen.add(key)
        platform_counts[platform] += 1
        partition_counts[platform][metadata["partition"]] += 1
        run_platforms[run_id].add(platform)
        sequence_runs[sequence] = run_id
        reports.append(
            {
                "path": str(path),
                "report_sha256": cache_key(report),
                "run_id": run_id,
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
    print(f"validated 40 duration-cross reports in {output}")
    return 0


def validate_cohort_document(cohort: dict[str, Any]) -> None:
    require(cohort.get("schema_version") == COHORT_SCHEMA_VERSION, "cohort schema")
    require(cohort.get("kind") == COHORT_KIND, "cohort kind")
    require(cohort.get("authoritative") is False, "cohort authoritative identity")
    require(cohort.get("production_budget") is None, "cohort production budget")
    require(cohort.get("excluded_observations") == [], "cohort exclusions")
    dispatch_document = cohort.get("dispatch")
    require(isinstance(dispatch_document, dict), "cohort dispatch")
    validate_dispatch_plan(dispatch_document, completed=True)
    observations = cohort.get("observations")
    require(isinstance(observations, list) and len(observations) == 40, "cohort observations")
    seen = set()
    for observation in observations:
        require(isinstance(observation, dict), "cohort observation object")
        key = (observation.get("sequence"), observation.get("platform"))
        require(key not in seen, f"duplicate cohort observation {key}")
        seen.add(key)
        require(
            observation.get("partition") == PARTITIONS.get(observation.get("sequence")),
            f"cohort observation {key} partition",
        )
        require(
            isinstance(observation.get("summaries"), dict),
            f"cohort observation {key} summaries",
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


def analyze(cohort_path: Path, policy_path: Path, output: Path) -> int:
    cohort = json.loads(cohort_path.read_text(encoding="UTF-8"))
    validate_cohort_document(cohort)
    policy = validate_derivation_policy(
        json.loads(policy_path.read_text(encoding="UTF-8"))
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
            expected_comparison_keys = set(indexed_by_observation[0]["comparisons"])
            expected_ratio_keys = set(indexed_by_observation[0]["ratio_of_ratios"])
            require(
                all(
                    set(item["comparisons"]) == expected_comparison_keys
                    and set(item["ratio_of_ratios"]) == expected_ratio_keys
                    for item in indexed_by_observation
                ),
                f"{platform}/{arm} has partial or unexpected metric coverage",
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
                        throughput_ceiling=min(
                            throughput_ceiling, ACCEPTANCE_CEILING_LOG
                        ),
                        elapsed_ceiling=min(
                            elapsed_ceiling, ACCEPTANCE_CEILING_LOG
                        ),
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
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--input-dir", type=Path, required=True)
    validate_parser.add_argument("--dispatch", type=Path, required=True)
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
        if args.command == "validate":
            return validate_cohort(args.input_dir, args.dispatch, args.output)
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
