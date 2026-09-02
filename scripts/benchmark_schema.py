#!/usr/bin/env python3
"""Shared helpers for versioned, reproducible benchmark reports."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2


class BenchmarkDataError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkDataError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def command_identity(command: list[str], timeout: float = 15) -> str:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else f"exit {result.returncode}"


def cpu_model() -> str:
    try:
        output = subprocess.run(
            ["lscpu"], text=True, capture_output=True, check=False, timeout=5
        ).stdout
        for line in output.splitlines():
            if line.lower().startswith("model name:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
    except (OSError, subprocess.TimeoutExpired):
        pass
    return platform.processor() or "unknown"


def host_metadata() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "runner_name": os.getenv("RUNNER_NAME", ""),
        "runner_image": os.getenv("ImageOS", ""),
        "runner_os": os.getenv("RUNNER_OS", ""),
        "runner_arch": os.getenv("RUNNER_ARCH", ""),
        "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "github_workflow": os.getenv("GITHUB_WORKFLOW", ""),
    }


def collected_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_stats(values: Iterable[float], sample_key: str) -> dict[str, Any]:
    samples = list(values)
    require(bool(samples), f"{sample_key}: no measured samples")
    return {
        sample_key: samples,
        "runs": len(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "range": max(samples) - min(samples),
    }


def alternating_pair_order(
    pair_index: int, left: str, right: str
) -> tuple[str, str]:
    require(pair_index >= 0, "pair index must be non-negative")
    return (left, right) if pair_index % 2 == 0 else (right, left)


def cache_key(parts: dict[str, Any]) -> str:
    canonical = json.dumps(
        parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("UTF-8")
    return sha256_bytes(canonical)


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="UTF-8", newline="\n") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, existing_mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def validate_common_report(document: dict[str, Any], kind: str) -> None:
    require(isinstance(document, dict), "report must be an object")
    require(document.get("schema_version") == SCHEMA_VERSION, "schema_version")
    require(document.get("kind") == kind, "report kind")
    metadata = document.get("metadata")
    require(isinstance(metadata, dict), "metadata")
    for key in (
        "commit",
        "tracked_diff_sha256",
        "build_source_sha256",
        "collected_at",
        "host",
        "tools",
        "fixtures",
    ):
        require(key in metadata, f"metadata.{key}")
    require(isinstance(document.get("plan"), dict), "plan")
    records = document.get("records")
    require(isinstance(records, list) and records, "records")
    summaries = document.get("summaries")
    require(isinstance(summaries, list) and summaries, "summaries")
