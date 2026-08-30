#!/usr/bin/env python3
"""Compare focused IV-simplification and loop-unroll AOT microbenchmarks."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

PROFILE_COUNTS = {
    "authoritative": (2, 10),
    "ci": (0, 3),
}


@dataclass(frozen=True)
class Fixture:
    name: str
    path: Path
    sha256: str


FIXTURES = (
    Fixture(
        "iv_store",
        Path("tests/benchmarks/loop-passes/iv_store.wasm"),
        "b1979dd330c14d5f898b8a7c8c313e58f6521ad6eb7db9a6e7d7e78788ac6e72",
    ),
    Fixture(
        "unroll4",
        Path("tests/benchmarks/loop-passes/unroll4.wasm"),
        "6870b3373e4098117c82b6736d0ca7cbcc7d8d747fe87ca5b7d1ebf0e4d12890",
    ),
)


@dataclass(frozen=True)
class AotSize:
    file_bytes: int
    text_bytes: int


@dataclass(frozen=True)
class CompiledFixture:
    fixture: Fixture
    cwasm: Path
    size: AotSize


@dataclass(frozen=True)
class RefBuild:
    label: str
    ref: str
    sha: str
    worktree: Path
    wamr: Path
    fixtures: dict[str, CompiledFixture]


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
            f"{' '.join(str(part) for part in cmd)}\n"
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_fixtures(repo: Path) -> tuple[Fixture, ...]:
    resolved: list[Fixture] = []
    for fixture in FIXTURES:
        path = (repo / fixture.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"loop benchmark fixture not found: {path}")
        actual = sha256_file(path)
        if actual != fixture.sha256:
            raise RuntimeError(
                f"{fixture.name} checksum mismatch: expected "
                f"{fixture.sha256}, got {actual}"
            )
        resolved.append(Fixture(fixture.name, path, actual))
    return tuple(resolved)


def parse_aot_size(path: Path) -> AotSize:
    data = path.read_bytes()
    if len(data) < 8 or data[:4] != b"\x00aot":
        raise RuntimeError(f"not a WAMR AOT file: {path}")
    pos = 8
    text_bytes = None
    while pos + 8 <= len(data):
        section_type, size = struct.unpack_from("<II", data, pos)
        pos += 8
        if pos + size > len(data):
            raise RuntimeError(f"truncated AOT section in {path}")
        if section_type == 2:
            text_bytes = size
        pos += size
    if text_bytes is None:
        raise RuntimeError(f"AOT text section missing from {path}")
    return AotSize(file_bytes=len(data), text_bytes=text_bytes)


def worktree_env(worktree: Path) -> dict:
    env = os.environ.copy()
    env.pop("ZIG_LOCAL_CACHE_DIR", None)
    global_cache = worktree / ".zig-global-cache"
    global_cache.mkdir(exist_ok=True)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    return env


def make_worktree(repo: Path, ref: str, root: Path, label: str) -> tuple[Path, str]:
    sha = run(["git", "rev-parse", ref], cwd=repo).strip()
    worktree = root / f"{label}-{sha[:12]}"
    run(["git", "worktree", "add", "--detach", str(worktree), sha], cwd=repo)
    return worktree, sha


def build_ref(
    repo: Path,
    root: Path,
    label: str,
    ref: str,
    fixtures: tuple[Fixture, ...],
    optimize: str,
) -> RefBuild:
    worktree, sha = make_worktree(repo, ref, root, label)
    try:
        env = worktree_env(worktree)
        print(
            f"[harness] building {label} {ref} ({sha[:12]}, {optimize})",
            file=sys.stderr,
        )
        run(["zig", "build", f"-Doptimize={optimize}"], cwd=worktree, env=env)
        wamrc = worktree / "zig-out/bin/wamrc"
        wamr = worktree / "zig-out/bin/wamr"
        compiled: dict[str, CompiledFixture] = {}
        for fixture in fixtures:
            cwasm = worktree / f".bench-loop-{fixture.name}.cwasm"
            run(
                [str(wamrc), "compile", str(fixture.path), "-o", str(cwasm)],
                cwd=worktree,
                env=env,
            )
            compiled[fixture.name] = CompiledFixture(
                fixture=fixture,
                cwasm=cwasm,
                size=parse_aot_size(cwasm),
            )
        return RefBuild(label, ref, sha, worktree, wamr, compiled)
    except Exception:
        try:
            run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo,
            )
        except Exception:
            pass
        raise


def measure_once(build: RefBuild, fixture_name: str) -> float:
    compiled = build.fixtures[fixture_name]
    started = time.perf_counter()
    run([str(build.wamr), "run", str(compiled.cwasm)], cwd=build.worktree)
    return time.perf_counter() - started


def measure_fixture(
    baseline: RefBuild,
    target: RefBuild,
    fixture_name: str,
    warmups: int,
    runs: int,
) -> dict[str, list[float]]:
    values = {"baseline": [], "target": []}
    builds = {"baseline": baseline, "target": target}
    total = warmups + runs
    for sample_index in range(total):
        phase = "warmup" if sample_index < warmups else "run"
        phase_index = (
            sample_index + 1
            if phase == "warmup"
            else sample_index - warmups + 1
        )
        order = ("baseline", "target")
        if sample_index % 2:
            order = tuple(reversed(order))
        for role in order:
            elapsed = measure_once(builds[role], fixture_name)
            discarded = " (discarded)" if phase == "warmup" else ""
            print(
                f"[harness]   {fixture_name} {role} {phase} "
                f"{phase_index}/{warmups if phase == 'warmup' else runs}: "
                f"{elapsed:.6f} s{discarded}",
                file=sys.stderr,
            )
            if phase == "run":
                values[role].append(elapsed)
    return values


def fmt_stats(values: list[float]) -> tuple[float, float, float, float]:
    return (
        statistics.fmean(values),
        statistics.median(values),
        min(values),
        max(values),
    )


def speedup_pct(baseline: list[float], target: list[float]) -> float:
    return (statistics.fmean(baseline) / statistics.fmean(target) - 1.0) * 100.0


def format_samples(values: list[float]) -> str:
    return ", ".join(f"{value:.6f}" for value in values)


def host_cpu_model() -> str:
    try:
        output = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in output.splitlines():
            if line.lower().startswith("model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def render_report(
    baseline: RefBuild,
    target: RefBuild,
    measurements: dict[str, dict[str, list[float]]],
    *,
    profile: str,
    warmups: int,
    runs: int,
    optimize: str,
) -> str:
    lines = [
        "### Focused AOT loop-pass comparison",
        "",
        f"- Measurement profile: `{profile}` ({warmups} warmups discarded, "
        f"{runs} measured runs per fixture/ref)",
        f"- WAMR optimize mode: `{optimize}`",
        "",
        "| Fixture | Ref | Mean s | Median s | Range s | Text bytes | AOT bytes | Samples s |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for fixture in FIXTURES:
        values = measurements[fixture.name]
        for role, build in (("baseline", baseline), ("target", target)):
            mean, median, minimum, maximum = fmt_stats(values[role])
            size = build.fixtures[fixture.name].size
            lines.append(
                f"| `{fixture.name}` | `{build.ref} ({role}, {build.sha[:12]})` | "
                f"{mean:.6f} | {median:.6f} | {minimum:.6f}–{maximum:.6f} | "
                f"{size.text_bytes} | {size.file_bytes} | "
                f"{format_samples(values[role])} |"
            )
        delta = speedup_pct(values["baseline"], values["target"])
        lines.append(
            f"| `{fixture.name}` | **target speedup** | **{delta:+.2f}%** |  |  |  |  |  |"
        )
    lines.extend(
        [
            "",
            "A sample is valid only when the benchmark `_start` returns normally; "
            "each fixture traps on an incorrect result.",
            "",
            f"_Host: arch `{platform.machine()}` · {os.cpu_count() or '?'} vCPU · "
            f"`{host_cpu_model()}`_",
        ]
    )
    return "\n".join(lines)


def parse_min_speedups(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    valid_names = {fixture.name for fixture in FIXTURES}
    for value in values:
        name, sep, raw_threshold = value.partition("=")
        if not sep or name not in valid_names:
            expected = ", ".join(sorted(valid_names))
            raise ValueError(
                f"invalid --min-speedup {value!r}; expected CASE=PCT "
                f"with CASE one of: {expected}"
            )
        thresholds[name] = float(raw_threshold)
    return thresholds


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="origin/main")
    parser.add_argument("--target", default="HEAD")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", choices=PROFILE_COUNTS, default="authoritative")
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--optimize", default="ReleaseFast")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emit", choices=["markdown", "github"], default="markdown")
    parser.add_argument(
        "--min-speedup",
        action="append",
        default=[],
        metavar="CASE=PCT",
        help="fail if target speedup for CASE is below PCT; repeatable",
    )
    args = parser.parse_args()

    try:
        warmups, runs = resolve_counts(args.profile, args.warmups, args.runs)
        thresholds = parse_min_speedups(args.min_speedup)
    except ValueError as exc:
        parser.error(str(exc))

    repo = args.repo.resolve()
    fixtures = resolve_fixtures(repo)
    work_root = repo / ".bench-loop-passes" / f"run-{uuid.uuid4().hex}"
    work_root.mkdir(parents=True)
    baseline = target = None
    try:
        baseline = build_ref(
            repo, work_root, "baseline", args.baseline, fixtures, args.optimize
        )
        target = build_ref(
            repo, work_root, "target", args.target, fixtures, args.optimize
        )
        measurements = {
            fixture.name: measure_fixture(
                baseline, target, fixture.name, warmups, runs
            )
            for fixture in fixtures
        }
        report = render_report(
            baseline,
            target,
            measurements,
            profile=args.profile,
            warmups=warmups,
            runs=runs,
            optimize=args.optimize,
        )
        print(report)
        if args.out is not None:
            args.out.write_text(report + "\n")
        if args.emit == "github":
            summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary_path:
                with Path(summary_path).open("a") as summary:
                    summary.write(report + "\n")

        failures = []
        for fixture_name, minimum in thresholds.items():
            actual = speedup_pct(
                measurements[fixture_name]["baseline"],
                measurements[fixture_name]["target"],
            )
            if actual < minimum:
                failures.append(
                    f"{fixture_name}: {actual:+.2f}% < required {minimum:+.2f}%"
                )
        if failures:
            print(
                "[harness] speedup threshold failed: " + "; ".join(failures),
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        for build in (target, baseline):
            if build is not None and build.worktree.exists():
                try:
                    run(
                        [
                            "git",
                            "worktree",
                            "remove",
                            "--force",
                            str(build.worktree),
                        ],
                        cwd=repo,
                    )
                except Exception as exc:
                    print(
                        f"[harness] warning: could not remove {build.worktree}: {exc}",
                        file=sys.stderr,
                    )
        shutil.rmtree(work_root, ignore_errors=True)
        try:
            work_root.parent.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
