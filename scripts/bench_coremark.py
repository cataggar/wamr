#!/usr/bin/env python3
"""Run a reproducible CoreMark comparison across WAMR refs and Wasmtime.

The default ``authoritative`` profile discards two warmups and records ten
measured runs.  The tracked CoreMark wasm fixture is used directly, so results
do not depend on whichever EEMBC/CoreMark commit happens to be upstream.

Usage
-----
    scripts/bench_coremark.py --baseline origin/main --target HEAD
    scripts/bench_coremark.py --wasmtime-baseline auto --wasmtime /path/to/wasmtime
    scripts/bench_coremark.py --profile ci --min-delta-pct=-5
    scripts/bench_coremark.py --optimize both
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from bench_optimize import OPTIMIZE_CHOICES, fmt_ratio, optimize_slug, parse_optimize_modes

ITER_PATTERN = re.compile(r"Iterations/Sec\s*:\s*([0-9]+(?:\.[0-9]+)?)")
VALIDATION_TEXT = "Correct operation validated."
DEFAULT_FIXTURE = Path("tests/benchmarks/coremark/coremark_wasi.wasm")
DEFAULT_FIXTURE_SHA256 = "f4b7591296ead10264e0f101f355bdf848865c31329325594e66fbabefec235b"
PINNED_WASMTIME_VERSION = "44.0.1"
PROFILE_COUNTS = {
    "authoritative": (2, 10),
    "ci": (0, 3),
}
PINNED_WASMTIME_ASSETS = {
    ("linux", "x86_64"): (
        "wasmtime-v44.0.1-x86_64-linux.tar.xz",
        "afd58715f105e3a7f454169daed22168c5736ec5f225fb04c4ac62c54c9508a3",
        "47e59a672f6a35ad071e3fbe7c0c541dcaea77b085d7aada4fd1a9ac77a0d601",
    ),
    ("linux", "aarch64"): (
        "wasmtime-v44.0.1-aarch64-linux.tar.xz",
        "9ec1e606c541099a7e8266865494e190a6f928abb3a2fb7bb4195fd2e8499d2b",
        "ad27d96d24d193ff32665a59552699e3f5355baeaf7220fb4cd88f023694e93d",
    ),
}

# Signature of the known x86_64 native AOT-run flake from issue #406.
_TRAP_FLAKE_PATTERN = re.compile(
    r'wasm trap: out of bounds memory access.*local_func\[-?\d+\](?:\s+"[^"]*")?\+0x0',
    re.IGNORECASE,
)
_TRAP_RETRY_MAX = 5


@dataclass(frozen=True)
class EngineResult:
    engine: str
    version: str
    optimize: str
    values: list[float]


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(
            f"\n[harness] command failed (exit {exc.returncode}): "
            f"{' '.join(str(c) for c in cmd)}\n"
        )
        if cwd is not None:
            sys.stderr.write(f"[harness]   cwd: {cwd}\n")
        if exc.stdout:
            sys.stderr.write(f"[harness] --- stdout ---\n{exc.stdout}")
            if not exc.stdout.endswith("\n"):
                sys.stderr.write("\n")
        if exc.stderr:
            sys.stderr.write(f"[harness] --- stderr ---\n{exc.stderr}")
            if not exc.stderr.endswith("\n"):
                sys.stderr.write("\n")
        sys.stderr.flush()
        raise
    return proc.stdout + proc.stderr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_fixture(repo: Path, fixture_arg: Path) -> tuple[Path, str]:
    fixture = fixture_arg if fixture_arg.is_absolute() else repo / fixture_arg
    fixture = fixture.resolve()
    if not fixture.is_file():
        raise FileNotFoundError(f"CoreMark fixture not found: {fixture}")
    digest = sha256_file(fixture)
    default_fixture = (repo / DEFAULT_FIXTURE).resolve()
    if fixture == default_fixture and digest != DEFAULT_FIXTURE_SHA256:
        raise RuntimeError(
            "tracked CoreMark fixture checksum mismatch: "
            f"expected {DEFAULT_FIXTURE_SHA256}, got {digest}"
        )
    return fixture, digest


def parse_coremark_output(output: str, engine: str) -> float:
    if VALIDATION_TEXT not in output or re.search(r"(?m)^.*ERROR!", output):
        raise RuntimeError(
            f"{engine} did not produce a CRC-validated CoreMark result:\n{output}"
        )
    matches = ITER_PATTERN.findall(output)
    if len(matches) != 1:
        raise RuntimeError(
            f"{engine} produced {len(matches)} Iterations/Sec values; expected one:\n"
            f"{output}"
        )
    return float(matches[0])


def make_worktree(repo: Path, ref: str, root: Path, label: str) -> tuple[Path, str]:
    sha = run(["git", "rev-parse", ref], cwd=repo).strip()
    wt = root / f"{label}-{sha[:12]}"
    run(["git", "worktree", "add", "--detach", str(wt), sha], cwd=repo)
    return wt, sha


def worktree_env(wt: Path) -> dict:
    env = os.environ.copy()
    env.pop("ZIG_LOCAL_CACHE_DIR", None)
    global_cache = wt / ".zig-global-cache"
    global_cache.mkdir(exist_ok=True)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    return env


def _run_wamr_with_retry(
    cmd: list[str],
    cwd: Path,
    env: dict,
    sample_idx: int,
    samples_total: int,
) -> str:
    last_exc = None
    for attempt in range(1 + _TRAP_RETRY_MAX):
        try:
            return run(cmd, cwd=cwd, env=env)
        except subprocess.CalledProcessError as exc:
            failure_output = (exc.stdout or "") + (exc.stderr or "")
            if not _TRAP_FLAKE_PATTERN.search(failure_output):
                raise
            if attempt == _TRAP_RETRY_MAX:
                last_exc = exc
                break
            print(
                f"[harness]   sample {sample_idx + 1}/{samples_total}: x86_64 "
                f"AOT trap flake (issue #406) — retry "
                f"{attempt + 1}/{_TRAP_RETRY_MAX}",
                file=sys.stderr,
            )
    assert last_exc is not None
    raise last_exc


def measure_command(
    *,
    engine: str,
    cmd: list[str],
    cwd: Path,
    env: dict,
    warmups: int,
    runs: int,
    retry_wamr_flake: bool = False,
) -> list[float]:
    values: list[float] = []
    total = warmups + runs
    for i in range(total):
        if retry_wamr_flake:
            output = _run_wamr_with_retry(cmd, cwd, env, i, total)
        else:
            output = run(cmd, cwd=cwd, env=env)
        value = parse_coremark_output(output, engine)
        if i < warmups:
            print(
                f"[harness]   {engine} warmup {i + 1}/{warmups}: "
                f"{value:.1f} iter/s (discarded)",
                file=sys.stderr,
            )
        else:
            measured_idx = i - warmups + 1
            print(
                f"[harness]   {engine} run {measured_idx}/{runs}: "
                f"{value:.1f} iter/s",
                file=sys.stderr,
            )
            values.append(value)
    return values


def build_and_run_wamr(
    wt: Path,
    ref: str,
    sha: str,
    fixture: Path,
    optimize: str,
    warmups: int,
    runs: int,
) -> EngineResult:
    env = worktree_env(wt)
    print(f"[harness] building WAMR {ref} ({sha[:12]}, {optimize})", file=sys.stderr)
    run(["zig", "build", f"-Doptimize={optimize}"], cwd=wt, env=env)

    wamrc = wt / "zig-out/bin/wamrc"
    wamr = wt / "zig-out/bin/wamr"
    cwasm = wt / ".bench-coremark.cwasm"
    print(f"[harness] AOT-compiling tracked fixture with WAMR {ref}", file=sys.stderr)
    run(
        [str(wamrc), "compile", str(fixture), "-o", str(cwasm)],
        cwd=wt,
        env=env,
    )
    values = measure_command(
        engine=f"WAMR {ref}",
        cmd=[str(wamr), "run", str(cwasm)],
        cwd=wt,
        env=env,
        warmups=warmups,
        runs=runs,
        retry_wamr_flake=True,
    )
    return EngineResult("WAMR", f"{ref} ({sha[:12]})", optimize, values)


def normalize_arch() -> str:
    arch = platform.machine().lower()
    return {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "aarch64",
    }.get(arch, arch)


def wasmtime_version(path: Path) -> str:
    output = run([str(path), "--version"]).strip()
    match = re.search(r"\bwasmtime\s+v?([0-9]+\.[0-9]+\.[0-9]+)\b", output)
    if not match:
        raise RuntimeError(f"could not parse Wasmtime version from: {output!r}")
    return match.group(1)


def validate_pinned_wasmtime(path: Path) -> str:
    version = wasmtime_version(path)
    if version != PINNED_WASMTIME_VERSION:
        raise RuntimeError(
            f"historical Wasmtime baseline must be {PINNED_WASMTIME_VERSION}; "
            f"{path} reports {version}"
        )
    return version


def install_pinned_wasmtime(repo: Path, cache_arg: Path | None) -> Path:
    key = (platform.system().lower(), normalize_arch())
    asset = PINNED_WASMTIME_ASSETS.get(key)
    if asset is None:
        supported = ", ".join(f"{os_name}/{arch}" for os_name, arch in PINNED_WASMTIME_ASSETS)
        raise RuntimeError(
            f"automatic Wasmtime installation is unsupported on {key[0]}/{key[1]}; "
            f"supported: {supported}. Pass --wasmtime-baseline PATH instead."
        )
    asset_name, expected_archive_sha, expected_binary_sha = asset
    cache = (
        cache_arg.expanduser().resolve()
        if cache_arg is not None
        else repo / ".bench-coremark/tools"
    )
    install_dir = cache / f"wasmtime-v{PINNED_WASMTIME_VERSION}-{key[1]}-{key[0]}"
    binary = install_dir / "wasmtime"
    if binary.is_file():
        actual_binary_sha = sha256_file(binary)
        if actual_binary_sha != expected_binary_sha:
            raise RuntimeError(
                f"cached Wasmtime binary checksum mismatch: expected "
                f"{expected_binary_sha}, got {actual_binary_sha}"
            )
        validate_pinned_wasmtime(binary)
        return binary

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / asset_name
    url = (
        "https://github.com/bytecodealliance/wasmtime/releases/download/"
        f"v{PINNED_WASMTIME_VERSION}/{asset_name}"
    )
    print(f"[harness] downloading pinned Wasmtime {PINNED_WASMTIME_VERSION}", file=sys.stderr)
    urllib.request.urlretrieve(url, archive)
    actual_sha = sha256_file(archive)
    if actual_sha != expected_archive_sha:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            "Wasmtime archive checksum mismatch: expected "
            f"{expected_archive_sha}, got {actual_sha}"
        )

    install_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.isfile() and Path(member.name).name == "wasmtime"
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"expected one wasmtime binary in {archive}, found {len(members)}"
            )
        source = tar.extractfile(members[0])
        if source is None:
            raise RuntimeError(f"could not extract wasmtime from {archive}")
        with binary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    binary.chmod(0o755)
    archive.unlink()
    actual_binary_sha = sha256_file(binary)
    if actual_binary_sha != expected_binary_sha:
        binary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Wasmtime binary checksum mismatch: expected {expected_binary_sha}, "
            f"got {actual_binary_sha}"
        )
    validate_pinned_wasmtime(binary)
    return binary


def measure_wasmtime(
    path: Path,
    label: str,
    version: str,
    fixture: Path,
    warmups: int,
    runs: int,
) -> EngineResult:
    values = measure_command(
        engine=label,
        cmd=[str(path), "run", str(fixture)],
        cwd=fixture.parent,
        env=os.environ.copy(),
        warmups=warmups,
        runs=runs,
    )
    return EngineResult(label, f"{version} ({path})", "default JIT", values)


def fmt_stats(values: list[float]) -> tuple[float, float, float, float]:
    return (
        statistics.fmean(values),
        statistics.median(values),
        min(values),
        max(values),
    )


def compute_delta_pct(baseline_vals: list[float], target_vals: list[float]) -> float:
    return (statistics.fmean(target_vals) / statistics.fmean(baseline_vals) - 1.0) * 100.0


def host_cpu_model() -> str:
    try:
        out = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if line.lower().startswith("model name:"):
                val = line.split(":", 1)[1].strip()
                if val:
                    return val
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                key = line.split(":", 1)[0].strip().lower()
                if key in ("model name", "model") and ":" in line:
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return val
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def host_info() -> str:
    arch = platform.machine() or "unknown-arch"
    ncpu = os.cpu_count()
    parts = [f"arch `{arch}`", f"{ncpu or '?'} vCPU", f"`{host_cpu_model()}`"]
    runner = os.environ.get("RUNNER_NAME")
    if runner:
        parts.append(f"runner `{runner}`")
    return "_Host: " + " · ".join(parts) + "_"


def format_samples(values: list[float]) -> str:
    return ", ".join(f"{value:.1f}" for value in values)


def render_table(
    results: list[EngineResult],
    *,
    profile: str,
    warmups: int,
    runs: int,
    fixture: Path,
    fixture_sha: str,
) -> str:
    baseline, target = results[0], results[1]
    delta_pct = compute_delta_pct(baseline.values, target.values)
    sign = "+" if delta_pct >= 0 else ""
    lines = [
        "### CoreMark cross-engine comparison",
        "",
        f"- Measurement profile: `{profile}` ({warmups} warmups discarded, "
        f"{runs} measured runs per engine)",
        f"- Fixture: `{fixture}` (`sha256:{fixture_sha}`)",
        f"- WAMR optimize mode: `{target.optimize}`; Wasmtime mode: `default JIT`",
        "",
        "| Engine | Version / ref | Optimize mode | Mean iter/s | Median | Range | Samples (iter/s) |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        mean, median, minimum, maximum = fmt_stats(result.values)
        lines.append(
            f"| {result.engine} | `{result.version}` | `{result.optimize}` | "
            f"{mean:.1f} | {median:.1f} | {minimum:.1f}–{maximum:.1f} | "
            f"{format_samples(result.values)} |"
        )
    lines.extend(
        [
            "",
            f"**WAMR target vs baseline:** {sign}{delta_pct:.2f}%",
        ]
    )
    wasmtime_results = results[2:]
    if wasmtime_results:
        lines.extend(
            [
                "",
                "| Same-host ratio | Mean iter/s ratio |",
                "|---|---:|",
            ]
        )
        target_mean = statistics.fmean(target.values)
        for result in wasmtime_results:
            ratio = target_mean / statistics.fmean(result.values)
            lines.append(f"| WAMR target / {result.engine} `{result.version}` | {ratio:.3f}× |")
    lines.extend(["", host_info()])
    return "\n".join(lines)


def render_optimize_table(
    target_ref: str,
    results: dict[str, list[float] | None],
    *,
    profile: str,
    warmups: int,
    runs: int,
    fixture: Path,
    fixture_sha: str,
) -> str:
    lines = [
        "### CoreMark AOT optimize-mode comparison",
        "",
        f"- Measurement profile: `{profile}` ({warmups} warmups discarded, "
        f"{runs} measured runs per mode)",
        f"- Fixture: `{fixture}` (`sha256:{fixture_sha}`)",
        "",
        "| Ref | Mode | Mean iter/s | Median | Range | Samples (iter/s) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for mode in ("ReleaseFast", "ReleaseSafe"):
        values = results[mode]
        if values is None:
            lines.append(f"| `{target_ref}` | `{mode}` | failed | failed | failed | failed |")
            continue
        mean, median, minimum, maximum = fmt_stats(values)
        lines.append(
            f"| `{target_ref}` | `{mode}` | {mean:.1f} | {median:.1f} | "
            f"{minimum:.1f}–{maximum:.1f} | {format_samples(values)} |"
        )
    fast_vals = results["ReleaseFast"]
    safe_vals = results["ReleaseSafe"]
    if fast_vals is not None and safe_vals is not None:
        lines.extend(
            [
                "",
                "**ReleaseSafe / ReleaseFast:** "
                + fmt_ratio(statistics.fmean(safe_vals), statistics.fmean(fast_vals)),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "At least one optimize mode failed before producing complete timings.",
            ]
        )
    lines.extend(["", host_info()])
    return "\n".join(lines)


def resolve_counts(
    profile: str, warmups_arg: int | None, runs_arg: int | None
) -> tuple[int, int]:
    default_warmups, default_runs = PROFILE_COUNTS[profile]
    warmups = default_warmups if warmups_arg is None else warmups_arg
    runs = default_runs if runs_arg is None else runs_arg
    if warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if runs <= 0:
        raise ValueError("--runs must be greater than zero")
    return warmups, runs


def profile_label(
    profile: str, warmups_arg: int | None, runs_arg: int | None
) -> str:
    defaults = PROFILE_COUNTS[profile]
    resolved = resolve_counts(profile, warmups_arg, runs_arg)
    return profile if resolved == defaults else f"{profile} (overridden)"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", default="origin/main", help="WAMR baseline git ref")
    p.add_argument("--target", default="HEAD", help="WAMR target git ref")
    p.add_argument(
        "--profile",
        choices=PROFILE_COUNTS,
        default="authoritative",
        help="measurement profile (default: authoritative = 2 warmups + 10 runs; ci = 0 + 3)",
    )
    p.add_argument("--warmups", type=int, default=None, help="override discarded warmups")
    p.add_argument("--runs", type=int, default=None, help="override measured runs")
    p.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help=f"CoreMark wasm fixture (default: {DEFAULT_FIXTURE})",
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="path to wamr repo",
    )
    p.add_argument("--out", type=Path, default=None, help="write markdown report here")
    p.add_argument(
        "--emit",
        choices=["markdown", "github"],
        default="markdown",
        help="`github` also appends to $GITHUB_STEP_SUMMARY",
    )
    p.add_argument(
        "--optimize",
        choices=OPTIMIZE_CHOICES,
        default="ReleaseFast",
        help="WAMR Zig optimize mode; `both` compares target ReleaseFast vs ReleaseSafe",
    )
    p.add_argument(
        "--wasmtime-baseline",
        default=None,
        metavar="PATH|auto",
        help=f"historical pinned Wasmtime {PINNED_WASMTIME_VERSION}; `auto` downloads a checksum-verified binary",
    )
    p.add_argument(
        "--wasmtime",
        type=Path,
        default=None,
        help="caller-selected/current Wasmtime binary; its exact version is reported",
    )
    p.add_argument(
        "--wasmtime-cache",
        type=Path,
        default=None,
        help="cache for --wasmtime-baseline auto (default: .bench-coremark/tools)",
    )
    p.add_argument(
        "--min-delta-pct",
        type=float,
        default=None,
        help="fail if WAMR target mean is below WAMR baseline by more than this delta",
    )
    args = p.parse_args()

    try:
        warmups, runs = resolve_counts(args.profile, args.warmups, args.runs)
    except ValueError as exc:
        p.error(str(exc))
    effective_profile = profile_label(args.profile, args.warmups, args.runs)
    if args.optimize == "both" and (args.wasmtime_baseline or args.wasmtime):
        p.error("Wasmtime comparisons require a single --optimize mode")

    repo = args.repo.resolve()
    fixture, fixture_sha = resolve_fixture(repo, args.fixture)
    work_root = repo / ".bench-coremark" / f"run-{uuid.uuid4().hex}"
    work_root.mkdir(parents=True)
    optimize_modes = parse_optimize_modes(args.optimize)
    baseline_vals: list[float] | None = None
    target_vals: list[float] = []
    optimize_failed = False

    try:
        if args.optimize == "both":
            mode_results: dict[str, list[float] | None] = {}
            for optimize in optimize_modes:
                wt, sha = make_worktree(
                    repo, args.target, work_root, f"target-{optimize_slug(optimize)}"
                )
                try:
                    result = build_and_run_wamr(
                        wt, args.target, sha, fixture, optimize, warmups, runs
                    )
                    mode_results[optimize] = result.values
                except Exception as exc:
                    print(
                        f"[harness] {optimize} failed before producing complete "
                        f"CoreMark timings: {exc}",
                        file=sys.stderr,
                    )
                    mode_results[optimize] = None
                    optimize_failed = True
            table = render_optimize_table(
                args.target,
                mode_results,
                profile=effective_profile,
                warmups=warmups,
                runs=runs,
                fixture=fixture,
                fixture_sha=fixture_sha,
            )
            target_vals = mode_results["ReleaseFast"] or []
        else:
            optimize = optimize_modes[0]
            baseline_wt, baseline_sha = make_worktree(
                repo, args.baseline, work_root, "baseline"
            )
            target_wt, target_sha = make_worktree(repo, args.target, work_root, "target")
            baseline_result = build_and_run_wamr(
                baseline_wt,
                args.baseline,
                baseline_sha,
                fixture,
                optimize,
                warmups,
                runs,
            )
            target_result = build_and_run_wamr(
                target_wt,
                args.target,
                target_sha,
                fixture,
                optimize,
                warmups,
                runs,
            )
            results = [baseline_result, target_result]
            baseline_vals = baseline_result.values
            target_vals = target_result.values

            measured_paths: dict[Path, EngineResult] = {}
            if args.wasmtime_baseline:
                baseline_path = (
                    install_pinned_wasmtime(repo, args.wasmtime_cache)
                    if args.wasmtime_baseline == "auto"
                    else Path(args.wasmtime_baseline).expanduser().resolve()
                )
                if not baseline_path.is_file():
                    raise FileNotFoundError(
                        f"historical Wasmtime binary not found: {baseline_path}"
                    )
                version = validate_pinned_wasmtime(baseline_path)
                result = measure_wasmtime(
                    baseline_path,
                    "Wasmtime historical pin",
                    version,
                    fixture,
                    warmups,
                    runs,
                )
                results.append(result)
                measured_paths[baseline_path.resolve()] = result

            if args.wasmtime:
                current_path = args.wasmtime.expanduser().resolve()
                if not current_path.is_file():
                    raise FileNotFoundError(
                        f"caller-selected Wasmtime binary not found: {current_path}"
                    )
                version = wasmtime_version(current_path)
                cached = measured_paths.get(current_path)
                if cached is None:
                    current_result = measure_wasmtime(
                        current_path,
                        "Wasmtime caller-selected",
                        version,
                        fixture,
                        warmups,
                        runs,
                    )
                else:
                    current_result = EngineResult(
                        "Wasmtime caller-selected",
                        cached.version,
                        cached.optimize,
                        cached.values,
                    )
                results.append(current_result)

            table = render_table(
                results,
                profile=effective_profile,
                warmups=warmups,
                runs=runs,
                fixture=fixture,
                fixture_sha=fixture_sha,
            )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        run(["git", "worktree", "prune"], cwd=repo)

    print(table)
    if args.out:
        args.out.write_text(table + "\n")
    if args.emit == "github":
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as fh:
                fh.write(table + "\n")

    if optimize_failed:
        return 1
    if args.min_delta_pct is not None and baseline_vals is not None:
        delta_pct = compute_delta_pct(baseline_vals, target_vals)
        if delta_pct < args.min_delta_pct:
            print(
                f"CoreMark AOT regression: {delta_pct:.2f}% is below "
                f"allowed minimum {args.min_delta_pct:.2f}%",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
