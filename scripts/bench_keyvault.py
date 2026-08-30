#!/usr/bin/env python3
"""Pinned dual-runtime keyvault/SpiderMonkey-TCGC benchmark harness.

The harness never downloads or builds third-party inputs.  A manifest must
identify every binary, checkout, file, and mounted tree by exact revision or
SHA-256.  Both runtimes are precompiled before warmups and measured execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MIN_MEASURED_RUNS = 5
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
TCGC_BYTES_RE = re.compile(r"got\s+(\d+)\s+bytes\s+back\s+from\s+tcgc\.compile")


class HarnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    path: Path
    sha256: str
    version: str


@dataclass(frozen=True)
class Mount:
    host: Path
    guest: str
    sha256_tree: str


@dataclass(frozen=True)
class Config:
    manifest_path: Path
    tools: dict[str, Tool]
    sources: list[dict[str, Any]]
    component: Path
    component_sha256: str
    preopens_file: Path
    preopens_sha256: str
    mounts: list[Mount]
    package_name: str
    guest_env: dict[str, str]
    expected_response_bytes: int
    perf: dict[str, Any]


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HarnessError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise HarnessError(f"{label} must be a non-empty JSON array")
    return value


def _require_string(obj: dict[str, Any], key: str, label: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise HarnessError(f"{label}.{key} must be a non-empty string")
    return value


def _require_sha(value: str, label: str) -> str:
    value = value.lower()
    if not SHA256_RE.fullmatch(value) or value == "0" * 64:
        raise HarnessError(f"{label} must be a non-zero, lowercase SHA-256")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> Config:
    try:
        raw = json.loads(path.read_text(encoding="UTF-8"))
    except FileNotFoundError as exc:
        raise HarnessError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HarnessError(f"invalid JSON in {path}: {exc}") from exc
    root = _require_object(raw, "manifest")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise HarnessError(
            f"schema_version must be {SCHEMA_VERSION}, got {root.get('schema_version')!r}"
        )
    base = path.resolve().parent

    tools_raw = _require_object(root.get("tools"), "tools")
    tools: dict[str, Tool] = {}
    for name in ("wamr", "wamrc", "wasmtime"):
        item = _require_object(tools_raw.get(name), f"tools.{name}")
        tools[name] = Tool(
            name=name,
            path=_resolve(base, _require_string(item, "path", f"tools.{name}")),
            sha256=_require_sha(
                _require_string(item, "sha256", f"tools.{name}"),
                f"tools.{name}.sha256",
            ),
            version=_require_string(item, "version", f"tools.{name}"),
        )

    sources = _require_list(root.get("sources"), "sources")
    normalized_sources: list[dict[str, Any]] = []
    for index, value in enumerate(sources):
        item = _require_object(value, f"sources[{index}]")
        revision = _require_string(item, "revision", f"sources[{index}]").lower()
        if not REVISION_RE.fullmatch(revision):
            raise HarnessError(
                f"sources[{index}].revision must be a full 40-character commit SHA"
            )
        normalized_sources.append(
            {
                "name": _require_string(item, "name", f"sources[{index}]"),
                "path": _resolve(
                    base, _require_string(item, "path", f"sources[{index}]")
                ),
                "revision": revision,
            }
        )

    workload = _require_object(root.get("workload"), "workload")
    component = _require_object(workload.get("component"), "workload.component")
    preopens = _require_object(
        workload.get("preopens_file"), "workload.preopens_file"
    )
    mounts_raw = _require_list(workload.get("mounts"), "workload.mounts")
    mounts: list[Mount] = []
    guests: set[str] = set()
    for index, value in enumerate(mounts_raw):
        item = _require_object(value, f"workload.mounts[{index}]")
        guest = _require_string(item, "guest", f"workload.mounts[{index}]")
        if not guest.startswith("/") or guest in guests:
            raise HarnessError(
                f"workload.mounts[{index}].guest must be a unique absolute guest path"
            )
        guests.add(guest)
        mounts.append(
            Mount(
                host=_resolve(
                    base,
                    _require_string(item, "path", f"workload.mounts[{index}]"),
                ),
                guest=guest,
                sha256_tree=_require_sha(
                    _require_string(
                        item, "sha256_tree", f"workload.mounts[{index}]"
                    ),
                    f"workload.mounts[{index}].sha256_tree",
                ),
            )
        )
    if "/spec" not in guests:
        raise HarnessError("workload.mounts must include the input spec tree at /spec")

    expected_response_bytes = workload.get("expected_tcgc_response_bytes")
    if not isinstance(expected_response_bytes, int) or expected_response_bytes <= 0:
        raise HarnessError(
            "workload.expected_tcgc_response_bytes must be a positive integer"
        )
    guest_env_raw = _require_object(workload.get("env"), "workload.env")
    guest_env: dict[str, str] = {}
    for key, value in guest_env_raw.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or not isinstance(value, str)
        ):
            raise HarnessError(
                "workload.env must map non-empty names without '=' to string values"
            )
        guest_env[key] = value
    if "WAMR_KEYVAULT_HARNESS" in guest_env:
        raise HarnessError("workload.env reserves WAMR_KEYVAULT_HARNESS")

    perf = _require_object(root.get("perf"), "perf")
    core_index = perf.get("core_index")
    hot_func = perf.get("hot_func")
    min_samples = perf.get("min_samples")
    min_coverage_pct = perf.get("min_attribution_coverage_pct")
    base = perf.get("base")
    if not isinstance(core_index, int) or core_index < 0:
        raise HarnessError("perf.core_index must be a non-negative integer")
    if hot_func is not None and (not isinstance(hot_func, int) or hot_func < 0):
        raise HarnessError("perf.hot_func must be null or a non-negative integer")
    if not isinstance(min_samples, int) or min_samples < 1000:
        raise HarnessError("perf.min_samples must be an integer >= 1000")
    if not isinstance(min_coverage_pct, (int, float)) or not (
        0 < float(min_coverage_pct) <= 100
    ):
        raise HarnessError(
            "perf.min_attribution_coverage_pct must be in the range (0, 100]"
        )
    if base is not None and (
        not isinstance(base, str) or re.fullmatch(r"0x[0-9a-fA-F]+", base) is None
    ):
        raise HarnessError("perf.base must be null or a hexadecimal address")

    return Config(
        manifest_path=path.resolve(),
        tools=tools,
        sources=normalized_sources,
        component=_resolve(
            base, _require_string(component, "path", "workload.component")
        ),
        component_sha256=_require_sha(
            _require_string(component, "sha256", "workload.component"),
            "workload.component.sha256",
        ),
        preopens_file=_resolve(
            base, _require_string(preopens, "path", "workload.preopens_file")
        ),
        preopens_sha256=_require_sha(
            _require_string(preopens, "sha256", "workload.preopens_file"),
            "workload.preopens_file.sha256",
        ),
        mounts=mounts,
        package_name=_require_string(workload, "package_name", "workload"),
        guest_env=guest_env,
        expected_response_bytes=expected_response_bytes,
        perf={
            "core_index": core_index,
            "hot_func": hot_func,
            "min_samples": min_samples,
            "min_attribution_coverage_pct": float(min_coverage_pct),
            "base": base,
        },
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise HarnessError(f"tree input is not a directory: {path}")
    entries: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        rel = item.relative_to(path).as_posix()
        if item.is_symlink():
            raise HarnessError(f"symlinks are not allowed in hashed trees: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise HarnessError(f"unsupported non-file tree entry: {item}")
        entries.append(
            {"path": rel, "size": item.stat().st_size, "sha256": sha256_file(item)}
        )
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(item["size"] for item in entries),
        "files": entries,
    }


def parse_preopens(path: Path) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="UTF-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise HarnessError(
                f"{path}:{line_number}: expected HOST=GUEST preopen mapping"
            )
        host, guest = line.split("=", 1)
        if not host or not guest.startswith("/"):
            raise HarnessError(
                f"{path}:{line_number}: expected non-empty HOST and absolute GUEST"
            )
        host_path = Path(host).expanduser()
        if not host_path.is_absolute():
            host_path = path.parent / host_path
        result.append((host_path.resolve(), guest))
    if not result:
        raise HarnessError(f"preopens file has no mappings: {path}")
    return result


def command_version(tool: Tool) -> str:
    version_arg = "--version" if tool.name == "wasmtime" else "version"
    proc = subprocess.run(
        [str(tool.path), version_arg],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
        env=controlled_host_env(),
    )
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        raise HarnessError(
            f"{tool.name} --version failed with exit {proc.returncode}: {output}"
        )
    if tool.version not in output:
        raise HarnessError(
            f"{tool.name} version mismatch: expected output containing "
            f"{tool.version!r}, got {output!r}"
        )
    return output


def validate_config(config: Config) -> dict[str, Any]:
    tool_report: dict[str, Any] = {}
    for tool in config.tools.values():
        if not tool.path.is_file():
            raise HarnessError(
                f"required {tool.name} binary is absent: {tool.path}; "
                "install/build the pinned version and update the manifest path"
            )
        if not os.access(tool.path, os.X_OK):
            raise HarnessError(f"required {tool.name} binary is not executable: {tool.path}")
        actual = sha256_file(tool.path)
        if actual != tool.sha256:
            raise HarnessError(
                f"{tool.name} SHA-256 mismatch for {tool.path}: "
                f"expected {tool.sha256}, got {actual}"
            )
        tool_report[tool.name] = {
            "path": str(tool.path),
            "sha256": actual,
            "version": command_version(tool),
        }

    source_report = []
    for source in config.sources:
        path = source["path"]
        if not (path / ".git").exists() and not (
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--git-dir"],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        ):
            raise HarnessError(f"source checkout is absent or not a git tree: {path}")
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        actual = proc.stdout.strip().lower()
        if proc.returncode != 0 or actual != source["revision"]:
            raise HarnessError(
                f"source revision mismatch for {source['name']} at {path}: "
                f"expected {source['revision']}, got {actual or proc.stderr.strip()}"
            )
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if dirty.returncode != 0 or dirty.stdout.strip():
            raise HarnessError(
                f"source checkout has tracked modifications: {source['name']} at {path}"
            )
        source_report.append(
            {"name": source["name"], "path": str(path), "revision": actual}
        )

    for path, expected, label in (
        (config.component, config.component_sha256, "component"),
        (config.preopens_file, config.preopens_sha256, "preopens file"),
    ):
        if not path.is_file():
            raise HarnessError(f"required {label} is absent: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise HarnessError(
                f"{label} SHA-256 mismatch for {path}: expected {expected}, got {actual}"
            )

    mount_report = []
    configured = {(mount.host, mount.guest): mount for mount in config.mounts}
    file_mappings = parse_preopens(config.preopens_file)
    configured_stdlib = {
        (mount.host, mount.guest)
        for mount in config.mounts
        if mount.guest not in ("/spec", "/out")
    }
    if set(file_mappings) != configured_stdlib:
        missing = set(file_mappings) - configured_stdlib
        extra = configured_stdlib - set(file_mappings)
        raise HarnessError(
            "preopens file and workload.mounts differ; "
            f"unconfigured={sorted(map(str, missing))}, extra={sorted(map(str, extra))}"
        )
    for key, mount in configured.items():
        snapshot = tree_snapshot(mount.host)
        if snapshot["sha256"] != mount.sha256_tree:
            raise HarnessError(
                f"tree SHA-256 mismatch for mount {mount.host}::{mount.guest}: "
                f"expected {mount.sha256_tree}, got {snapshot['sha256']}"
            )
        mount_report.append(
            {
                "path": str(mount.host),
                "guest": mount.guest,
                "sha256_tree": snapshot["sha256"],
                "file_count": snapshot["file_count"],
                "total_bytes": snapshot["total_bytes"],
            }
        )
    return {
        "tools": tool_report,
        "sources": source_report,
        "inputs": {
            "component": {
                "path": str(config.component),
                "sha256": config.component_sha256,
                "size": config.component.stat().st_size,
            },
            "preopens_file": {
                "path": str(config.preopens_file),
                "sha256": config.preopens_sha256,
            },
            "mounts": mount_report,
            "package_name": config.package_name,
            "guest_env": config.guest_env,
            "expected_tcgc_response_bytes": config.expected_response_bytes,
        },
    }


def precompile_commands(
    config: Config, artifact_dir: Path
) -> tuple[list[str], list[str], Path, Path]:
    wamr_manifest = artifact_dir / "keyvault.cwasm.json"
    wasmtime_cwasm = artifact_dir / "keyvault.wasmtime.cwasm"
    wamr = [
        str(config.tools["wamrc"].path),
        "compile-component",
        "--target=x86_64",
        str(config.component),
        "-o",
        str(wamr_manifest),
    ]
    wasmtime = [
        str(config.tools["wasmtime"].path),
        "compile",
        "-W",
        "max-memory-size=4294967296",
        "-O",
        "opt-level=2",
        str(config.component),
        "-o",
        str(wasmtime_cwasm),
    ]
    return wamr, wasmtime, wamr_manifest, wasmtime_cwasm


def runtime_command(
    config: Config,
    runtime: str,
    output_dir: Path,
    wamr_manifest: Path,
    wasmtime_cwasm: Path,
) -> list[str]:
    if runtime == "wamr":
        command = [
            str(config.tools["wamr"].path),
            "run",
            "--precompiled-manifest",
            str(wamr_manifest),
        ]
        for key, value in sorted(
            {"WAMR_KEYVAULT_HARNESS": "1", **config.guest_env}.items()
        ):
            command += ["--env", f"{key}={value}"]
        for mount in config.mounts:
            command += ["--map-dir", f"{mount.host}::{mount.guest}"]
        command += ["--map-dir", f"{output_dir.resolve()}::/out"]
        command += [
            str(config.component),
            "/spec",
            "/out",
            "--package-name",
            config.package_name,
        ]
        return command
    if runtime == "wasmtime":
        command = [
            str(config.tools["wasmtime"].path),
            "run",
            "--allow-precompiled",
            "-S",
            "http",
            "-W",
            "max-memory-size=4294967296",
        ]
        for key, value in sorted(
            {"WAMR_KEYVAULT_HARNESS": "1", **config.guest_env}.items()
        ):
            command += ["--env", f"{key}={value}"]
        for mount in config.mounts:
            command += ["--dir", f"{mount.host}::{mount.guest}"]
        command += ["--dir", f"{output_dir.resolve()}::/out"]
        command += [
            str(wasmtime_cwasm),
            "/spec",
            "/out",
            "--package-name",
            config.package_name,
        ]
        return command
    raise HarnessError(f"unknown runtime: {runtime}")


def parse_tcgc_response_bytes(output: str) -> int:
    matches = TCGC_BYTES_RE.findall(output)
    if len(matches) != 1:
        raise HarnessError(
            "expected exactly one 'got N bytes back from tcgc.compile' marker, "
            f"found {len(matches)}"
        )
    return int(matches[0])


def parse_timing_samples(records: list[dict[str, Any]], runtime: str) -> list[float]:
    samples = [
        record.get("elapsed_seconds")
        for record in records
        if record.get("runtime") == runtime and record.get("phase") == "measure"
    ]
    if any(not isinstance(value, (int, float)) or value <= 0 for value in samples):
        raise HarnessError(f"{runtime} has an invalid measured timing sample")
    if len(samples) < MIN_MEASURED_RUNS:
        raise HarnessError(
            f"{runtime} has {len(samples)} measured runs; "
            f"at least {MIN_MEASURED_RUNS} are required"
        )
    return [float(value) for value in samples]


def controlled_host_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
    }
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    return env


def _run_checked(command: list[str], *, timeout: float, label: str) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=controlled_host_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"{label} exceeded timeout of {timeout:g}s") from exc
    if proc.returncode != 0:
        raise HarnessError(
            f"{label} failed with exit {proc.returncode}\n"
            f"command: {' '.join(command)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def precompile(config: Config, artifact_dir: Path, timeout: float) -> tuple[Path, Path, dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    wamr_cmd, wasmtime_cmd, wamr_manifest, wasmtime_cwasm = precompile_commands(
        config, artifact_dir
    )
    for old in artifact_dir.iterdir():
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()
    start = time.perf_counter_ns()
    _run_checked(wamr_cmd, timeout=timeout, label="WAMR precompile")
    wamr_seconds = (time.perf_counter_ns() - start) / 1e9
    start = time.perf_counter_ns()
    _run_checked(wasmtime_cmd, timeout=timeout, label="Wasmtime precompile")
    wasmtime_seconds = (time.perf_counter_ns() - start) / 1e9
    if not wamr_manifest.is_file() or not wasmtime_cwasm.is_file():
        raise HarnessError("precompile completed without producing both artifacts")
    artifacts = []
    for item in sorted(artifact_dir.iterdir()):
        if item.is_file():
            artifacts.append(
                {
                    "path": str(item),
                    "size": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    return wamr_manifest, wasmtime_cwasm, {
        "excluded_from_measurement": True,
        "wamr_seconds": wamr_seconds,
        "wasmtime_seconds": wasmtime_seconds,
        "commands": {"wamr": wamr_cmd, "wasmtime": wasmtime_cmd},
        "artifacts": artifacts,
    }


def run_one(
    config: Config,
    runtime: str,
    phase: str,
    index: int,
    output_root: Path,
    wamr_manifest: Path,
    wasmtime_cwasm: Path,
    timeout: float,
) -> dict[str, Any]:
    output_dir = output_root / runtime / f"{phase}-{index:02d}"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    command = runtime_command(
        config, runtime, output_dir, wamr_manifest, wasmtime_cwasm
    )
    start = time.perf_counter_ns()
    proc = _run_checked(
        command, timeout=timeout, label=f"{runtime} {phase} run {index}"
    )
    elapsed = (time.perf_counter_ns() - start) / 1e9
    combined = proc.stdout + "\n" + proc.stderr
    response_bytes = parse_tcgc_response_bytes(combined)
    if response_bytes != config.expected_response_bytes:
        raise HarnessError(
            f"{runtime} {phase} run {index} returned {response_bytes} bytes from "
            f"tcgc.compile; manifest requires {config.expected_response_bytes}"
        )
    snapshot = tree_snapshot(output_dir)
    return {
        "runtime": runtime,
        "phase": phase,
        "index": index,
        "elapsed_seconds": elapsed,
        "tcgc_response_bytes": response_bytes,
        "output": {
            "sha256_tree": snapshot["sha256"],
            "file_count": snapshot["file_count"],
            "total_bytes": snapshot["total_bytes"],
            "files": snapshot["files"],
        },
        "command": command,
    }


def verify_equivalence(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise HarnessError("no runtime records were produced")
    expected_bytes = records[0]["tcgc_response_bytes"]
    expected_output = records[0]["output"]
    for record in records[1:]:
        if record["tcgc_response_bytes"] != expected_bytes:
            raise HarnessError(
                "tcgc response-size drift: "
                f"expected {expected_bytes}, got {record['tcgc_response_bytes']} "
                f"for {record['runtime']} {record['phase']} {record['index']}"
            )
        if record["output"] != expected_output:
            raise HarnessError(
                "generated output mismatch: exact paths, sizes, or SHA-256 values "
                f"differ for {record['runtime']} {record['phase']} {record['index']}"
            )
    return {
        "tcgc_response_bytes": expected_bytes,
        "output_sha256_tree": expected_output["sha256_tree"],
        "file_count": expected_output["file_count"],
        "total_bytes": expected_output["total_bytes"],
        "deterministic_across_all_runs": True,
        "cross_runtime_exact_match": True,
    }


def sample_stats(values: list[float]) -> dict[str, Any]:
    return {
        "samples_seconds": values,
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "range_seconds": max(values) - min(values),
        "runs": len(values),
    }


def host_report() -> dict[str, Any]:
    cpu = "unknown"
    try:
        out = subprocess.run(
            ["lscpu"], text=True, capture_output=True, check=False, timeout=5
        ).stdout
        for line in out.splitlines():
            if line.lower().startswith("model name:"):
                cpu = line.split(":", 1)[1].strip()
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }


def run_perf(
    config: Config,
    work_dir: Path,
    wamr_manifest: Path,
    wasmtime_cwasm: Path,
    timeout: float,
    reference_output: dict[str, Any],
) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise HarnessError("--profile requires Linux x86_64")
    for name in ("perf", "objdump"):
        if shutil.which(name) is None:
            raise HarnessError(
                f"--profile explicitly selected, but required tool {name!r} is absent"
            )
    paranoid = Path("/proc/sys/kernel/perf_event_paranoid")
    if paranoid.is_file():
        value = int(paranoid.read_text().strip())
        if value > 2:
            raise HarnessError(
                "--profile explicitly selected, but kernel.perf_event_paranoid "
                f"is {value}; set it to <= 2"
            )

    perf_dir = work_dir / "perf"
    output_dir = perf_dir / "output"
    shutil.rmtree(perf_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    runtime = runtime_command(
        config, "wamr", output_dir, wamr_manifest, wasmtime_cwasm
    )
    perf_data = perf_dir / "wamr.perf.data"
    command = [
        "perf",
        "record",
        "-g",
        "--call-graph=dwarf",
        "-F",
        "999",
        "-e",
        "cycles:u",
        "-o",
        str(perf_data),
        "--",
        *runtime,
    ]
    proc = _run_checked(command, timeout=timeout, label="WAMR perf capture")
    response_bytes = parse_tcgc_response_bytes(proc.stdout + "\n" + proc.stderr)
    snapshot = tree_snapshot(output_dir)
    current = {
        "sha256_tree": snapshot["sha256"],
        "file_count": snapshot["file_count"],
        "total_bytes": snapshot["total_bytes"],
        "files": snapshot["files"],
    }
    if response_bytes != config.expected_response_bytes or current != reference_output:
        raise HarnessError("profiled WAMR run was not semantically equivalent")

    core = perf_dir.parent / "artifacts" / f"keyvault.{config.perf['core_index']}.cwasm"
    if not core.is_file():
        raise HarnessError(
            f"configured perf core artifact is absent: {core}; "
            "check perf.core_index and the WAMR manifest"
        )
    attr_json = perf_dir / "attribution.json"
    helper = Path(__file__).resolve().parents[1] / ".github/skills/aot-perf-profile/aot_jit_attr.py"
    attr_command = [
        sys.executable,
        str(helper),
        "--perf",
        str(perf_data),
        "--cwasm",
        str(core),
        "--json-out",
        str(attr_json),
        "--min-samples",
        str(config.perf["min_samples"]),
        "--require-size-match",
    ]
    if config.perf["base"] is not None:
        attr_command.remove("--require-size-match")
        attr_command += ["--base", config.perf["base"]]
    if config.perf["hot_func"] is not None:
        attr_command += ["--func", str(config.perf["hot_func"])]
    _run_checked(attr_command, timeout=timeout, label="perf attribution")
    attribution = json.loads(attr_json.read_text(encoding="UTF-8"))
    coverage = attribution["attribution_coverage_pct"]
    if coverage < config.perf["min_attribution_coverage_pct"]:
        raise HarnessError(
            f"perf attribution coverage {coverage:.2f}% is below manifest minimum "
            f"{config.perf['min_attribution_coverage_pct']:.2f}%"
        )
    return {
        "selected": True,
        "command": command,
        "perf_data": str(perf_data),
        "attribution": attribution,
        "attribution_command": attr_command,
    }


def render_markdown(report: dict[str, Any]) -> str:
    timing = report["timing"]
    wamr = timing["wamr"]
    wasmtime = timing["wasmtime"]
    host = report["host"]
    lines = [
        "# Keyvault SpiderMonkey/TCGC AOT benchmark",
        "",
        f"- Host: `{host['system']} {host['release']}` · `{host['machine']}` · "
        f"{host['logical_cpus']} CPUs · `{host['cpu']}`",
        f"- Manifest: `{report['manifest']}`",
        f"- Warmups: {report['plan']['warmups']} per runtime",
        f"- Authoritative measured runs: {report['plan']['runs']} per runtime",
        "- Compilation: precompiled once per runtime and excluded from measured execution",
        "",
        "| Runtime | Samples (s) | Mean | Median | Min | Max | Range |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| WAMR | {', '.join(f'{v:.3f}' for v in wamr['samples_seconds'])} | "
        f"{wamr['mean_seconds']:.3f} | {wamr['median_seconds']:.3f} | "
        f"{wamr['min_seconds']:.3f} | {wamr['max_seconds']:.3f} | "
        f"{wamr['range_seconds']:.3f} |",
        f"| Wasmtime | {', '.join(f'{v:.3f}' for v in wasmtime['samples_seconds'])} | "
        f"{wasmtime['mean_seconds']:.3f} | {wasmtime['median_seconds']:.3f} | "
        f"{wasmtime['min_seconds']:.3f} | {wasmtime['max_seconds']:.3f} | "
        f"{wasmtime['range_seconds']:.3f} |",
        "",
        f"**Wasmtime-time/WAMR-time ratio: "
        f"{timing['wasmtime_time_over_wamr_time']:.4f}×**",
        "",
        "## Semantic equivalence",
        "",
        f"- `tcgc.compile` bytes: {report['equivalence']['tcgc_response_bytes']}",
        f"- Generated files: {report['equivalence']['file_count']} "
        f"({report['equivalence']['total_bytes']} bytes)",
        f"- Output tree SHA-256: `{report['equivalence']['output_sha256_tree']}`",
        "- Every warmup/measured output and both runtimes matched exactly by path, size, and SHA-256.",
        "",
        "## Exact tools and inputs",
        "",
    ]
    for name, tool in report["validation"]["tools"].items():
        lines.append(
            f"- {name}: `{tool['version']}` · SHA-256 `{tool['sha256']}` · `{tool['path']}`"
        )
    lines.append(
        f"- component: SHA-256 `{report['validation']['inputs']['component']['sha256']}` · "
        f"{report['validation']['inputs']['component']['size']} bytes"
    )
    lines.append(
        "- preopens file: SHA-256 "
        f"`{report['validation']['inputs']['preopens_file']['sha256']}` · "
        f"`{report['validation']['inputs']['preopens_file']['path']}`"
    )
    for mount in report["validation"]["inputs"]["mounts"]:
        lines.append(
            f"- mount `{mount['guest']}`: tree SHA-256 `{mount['sha256_tree']}` · "
            f"{mount['file_count']} files / {mount['total_bytes']} bytes · "
            f"`{mount['path']}`"
        )
    for source in report["validation"]["sources"]:
        lines.append(
            f"- source {source['name']}: `{source['revision']}` · `{source['path']}`"
        )
    if report["perf"]["selected"]:
        attr = report["perf"]["attribution"]
        lines += [
            "",
            "## Perf attribution",
            "",
            f"- Total self samples: {attr['total_samples']}",
            f"- Samples attributed to configured core: {attr['attributed_samples']}",
            f"- Attribution coverage: {attr['attribution_coverage_pct']:.2f}%",
            f"- `perf.data`: `{report['perf']['perf_data']}`",
        ]
    else:
        lines += [
            "",
            "## Perf attribution",
            "",
            "- Not selected. Re-run with `--profile`; missing/unsupported perf is a hard error in that mode.",
        ]
    return "\n".join(lines) + "\n"


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.runs < MIN_MEASURED_RUNS:
        raise HarnessError(
            f"--runs must be >= {MIN_MEASURED_RUNS} for an authoritative result"
        )
    if args.warmups < 0:
        raise HarnessError("--warmups must be >= 0")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise HarnessError(
            "the pinned #798 timing cohort requires Linux x86_64"
        )
    config = load_manifest(args.manifest)
    validation = validate_config(config)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    wamr_manifest, wasmtime_cwasm, compile_report = precompile(
        config, work_dir / "artifacts", args.compile_timeout
    )
    records: list[dict[str, Any]] = []
    for index in range(1, args.warmups + 1):
        for runtime in ("wamr", "wasmtime"):
            records.append(
                run_one(
                    config,
                    runtime,
                    "warmup",
                    index,
                    work_dir / "outputs",
                    wamr_manifest,
                    wasmtime_cwasm,
                    args.run_timeout,
                )
            )
    for index in range(1, args.runs + 1):
        order = ("wamr", "wasmtime") if index % 2 else ("wasmtime", "wamr")
        for runtime in order:
            record = run_one(
                config,
                runtime,
                "measure",
                index,
                work_dir / "outputs",
                wamr_manifest,
                wasmtime_cwasm,
                args.run_timeout,
            )
            records.append(record)
            print(
                f"[keyvault] {runtime} measured {index}/{args.runs}: "
                f"{record['elapsed_seconds']:.3f}s",
                file=sys.stderr,
            )
    equivalence = verify_equivalence(records)
    wamr_samples = parse_timing_samples(records, "wamr")
    wasmtime_samples = parse_timing_samples(records, "wasmtime")
    timing = {
        "wamr": sample_stats(wamr_samples),
        "wasmtime": sample_stats(wasmtime_samples),
        "wasmtime_time_over_wamr_time": (
            statistics.fmean(wasmtime_samples) / statistics.fmean(wamr_samples)
        ),
    }
    measured_reference = next(
        record["output"] for record in records if record["phase"] == "measure"
    )
    perf_report = (
        run_perf(
            config,
            work_dir,
            wamr_manifest,
            wasmtime_cwasm,
            args.run_timeout,
            measured_reference,
        )
        if args.profile
        else {"selected": False}
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "keyvault-tcgc-aot-benchmark",
        "manifest": str(config.manifest_path),
        "host": host_report(),
        "plan": {
            "warmups": args.warmups,
            "runs": args.runs,
            "compile_timeout_seconds": args.compile_timeout,
            "run_timeout_seconds": args.run_timeout,
        },
        "validation": validation,
        "precompile": compile_report,
        "records": records,
        "equivalence": equivalence,
        "timing": timing,
        "perf": perf_report,
    }
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="UTF-8"
    )
    args.report_markdown.write_text(render_markdown(report), encoding="UTF-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--hash",
        type=Path,
        help="print the harness SHA-256 for a file or directory, then exit",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--compile-timeout", type=float, default=1800)
    parser.add_argument("--run-timeout", type=float, default=600)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report-json", type=Path, default=Path("keyvault-report.json"))
    parser.add_argument(
        "--report-markdown", type=Path, default=Path("keyvault-report.md")
    )
    args = parser.parse_args()
    try:
        if args.hash:
            path = args.hash.resolve()
            if path.is_file():
                print(json.dumps({"path": str(path), "sha256": sha256_file(path)}))
            elif path.is_dir():
                print(json.dumps(tree_snapshot(path), indent=2, sort_keys=True))
            else:
                raise HarnessError(f"path to hash is absent: {path}")
            return 0
        if args.manifest is None:
            raise HarnessError("--manifest is required unless --hash is used")
        if not args.validate_only and args.work_dir is None:
            raise HarnessError("--work-dir is required for benchmark execution")
        config = load_manifest(args.manifest)
        if args.validate_only:
            print(json.dumps(validate_config(config), indent=2, sort_keys=True))
            return 0
        args.report_json = args.report_json.resolve()
        args.report_markdown = args.report_markdown.resolve()
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
        report = execute(args)
        print(render_markdown(report), end="")
        return 0
    except (HarnessError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
