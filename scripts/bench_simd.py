#!/usr/bin/env python3
"""Compare SIMD benchmark status/timing between two git refs.

The script builds a temporary worktree for each ref, overlays the current SIMD
benchmark harness into that worktree, and runs `simd-bench-runner`.  Overlaying
the harness lets older refs such as `origin/main` report "unsupported" for SIMD
AOT rather than failing just because the harness file did not exist yet.

Usage
-----
    scripts/bench_simd.py --baseline origin/main --target HEAD --runs 3
    scripts/bench_simd.py --baseline origin/main --target HEAD --emit github
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HARNESS_OVERLAY = (
    "build.zig",
    "src/tests/aot_harness.zig",
    "src/tests/simd_bench_runner.zig",
)


@dataclass(frozen=True)
class Measurement:
    case: str
    engine: str
    status: str
    result: int | None
    compile_ns: int | None
    run_ns: int | None
    iterations: int
    code_size: int | None
    run_index: int


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout + proc.stderr


def make_worktree(repo: Path, ref: str, root: Path, label: str) -> Path:
    sha = run(["git", "rev-parse", ref], cwd=repo).strip()
    wt = root / f"{label}-{sha[:12]}"
    if wt.exists():
        try:
            run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo)
        except subprocess.CalledProcessError:
            shutil.rmtree(wt)
    run(["git", "worktree", "add", "--detach", str(wt), sha], cwd=repo)
    return wt


def overlay_harness(source_repo: Path, wt: Path) -> None:
    for rel in HARNESS_OVERLAY:
        src = source_repo / rel
        dst = wt / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def parse_optional_int(value: str) -> int | None:
    if value == "-":
        return None
    return int(value)


def parse_runner_output(output: str, run_index: int) -> list[Measurement]:
    rows: list[Measurement] = []
    for line in output.splitlines():
        if not line.startswith("bench\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 9:
            raise RuntimeError(f"malformed simd bench row: {line}")
        _, case, engine, status, result, compile_ns, run_ns, iterations, code_size = parts
        rows.append(
            Measurement(
                case=case,
                engine=engine,
                status=status,
                result=parse_optional_int(result),
                compile_ns=parse_optional_int(compile_ns),
                run_ns=parse_optional_int(run_ns),
                iterations=int(iterations),
                code_size=parse_optional_int(code_size),
                run_index=run_index,
            )
        )
    if not rows:
        raise RuntimeError(f"simd-bench-runner produced no parseable rows:\n{output}")
    return rows


def build_and_run(
    wt: Path,
    source_repo: Path,
    runs: int,
    iterations: int,
) -> list[Measurement]:
    env = os.environ.copy()
    overlay_harness(source_repo, wt)

    print(f"[harness] building {wt.name} (ReleaseFast)", file=sys.stderr)
    run(["zig", "build", "-Doptimize=ReleaseFast"], cwd=wt, env=env)

    runner = wt / "zig-out/bin/simd-bench-runner"
    if not runner.exists():
        raise RuntimeError(f"expected runner was not built: {runner}")

    measurements: list[Measurement] = []
    for i in range(runs):
        print(
            f"[harness] running {wt.name} ({i + 1}/{runs}, iterations={iterations})",
            file=sys.stderr,
        )
        out = run([str(runner), "--iterations", str(iterations)], cwd=wt, env=env)
        measurements.extend(parse_runner_output(out, i + 1))
    return measurements


def fmt_ns(value: float | int | None) -> str:
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} ms"
    if value >= 1_000:
        return f"{value / 1_000:.3f} us"
    return f"{value:.0f} ns"


def fmt_value(value: int | None) -> str:
    return "-" if value is None else str(value)


def summarize(rows: list[Measurement]) -> dict[tuple[str, str], dict[str, object]]:
    grouped: dict[tuple[str, str], list[Measurement]] = {}
    for row in rows:
        grouped.setdefault((row.case, row.engine), []).append(row)

    summary: dict[tuple[str, str], dict[str, object]] = {}
    for key, values in grouped.items():
        ok_values = [v for v in values if v.status == "ok"]
        selected = ok_values if ok_values else values
        run_times = [v.run_ns for v in selected if v.run_ns is not None]
        compile_times = [v.compile_ns for v in selected if v.compile_ns is not None]
        code_sizes = [v.code_size for v in selected if v.code_size is not None]
        results = [v.result for v in selected if v.result is not None]
        summary[key] = {
            "status": "ok" if ok_values else selected[0].status,
            "result": results[0] if results else None,
            "run_ns": statistics.median(run_times) if run_times else None,
            "compile_ns": statistics.median(compile_times) if compile_times else None,
            "iterations": selected[0].iterations,
            "code_size": code_sizes[0] if code_sizes else None,
        }
    return summary


def render_table(
    baseline_ref: str,
    baseline_rows: list[Measurement],
    target_ref: str,
    target_rows: list[Measurement],
) -> str:
    baseline = summarize(baseline_rows)
    target = summarize(target_rows)
    keys = sorted(set(baseline) | set(target))

    lines = [
        "### SIMD AOT benchmark comparison",
        "",
        "| Case | Engine | Ref | Status | Result | Median run | Median compile | Code size | Iterations |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for case, engine in keys:
        for role, ref, table in (
            ("baseline", baseline_ref, baseline),
            ("target", target_ref, target),
        ):
            row = table.get((case, engine))
            ref_label = f"{ref} ({role})"
            if row is None:
                lines.append(f"| `{case}` | `{engine}` | `{ref_label}` | missing | - | - | - | - | - |")
                continue
            lines.append(
                "| `{case}` | `{engine}` | `{ref}` | {status} | {result} | {run_ns} | {compile_ns} | {code_size} | {iterations} |".format(
                    case=case,
                    engine=engine,
                    ref=ref_label,
                    status=row["status"],
                    result=fmt_value(row["result"]),  # type: ignore[arg-type]
                    run_ns=fmt_ns(row["run_ns"]),  # type: ignore[arg-type]
                    compile_ns=fmt_ns(row["compile_ns"]),  # type: ignore[arg-type]
                    code_size=fmt_value(row["code_size"]),  # type: ignore[arg-type]
                    iterations=row["iterations"],
                )
            )

    lines.extend(
        [
            "",
            "AOT rows with `unsupported` are expected for SIMD cases until native v128 lowering lands.",
            "CoreMark is scalar, so this harness is the SIMD-specific signal for issue #220.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", default="origin/main", help="git ref for the baseline")
    p.add_argument("--target", default="HEAD", help="git ref for the target")
    p.add_argument("--runs", type=int, default=3, help="runner invocations per ref")
    p.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="function calls per runner invocation",
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="path to wamr repo (default: parent of scripts/)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="if given, write the markdown table here as well",
    )
    p.add_argument(
        "--emit",
        choices=["markdown", "github"],
        default="markdown",
        help="`github` also appends to $GITHUB_STEP_SUMMARY when present",
    )
    args = p.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    repo = args.repo.resolve()
    with tempfile.TemporaryDirectory(
        prefix="bench-simd-",
        dir="/work" if Path("/work").is_dir() else None,
    ) as tmp:
        root = Path(tmp)
        try:
            wt_b = make_worktree(repo, args.baseline, root, "baseline")
            wt_t = make_worktree(repo, args.target, root, "target")

            baseline_rows = build_and_run(wt_b, repo, args.runs, args.iterations)
            target_rows = build_and_run(wt_t, repo, args.runs, args.iterations)
        finally:
            run(["git", "worktree", "prune"], cwd=repo)

    table = render_table(args.baseline, baseline_rows, args.target, target_rows)
    print(table)

    if args.out:
        args.out.write_text(table + "\n")

    if args.emit == "github":
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(table + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
