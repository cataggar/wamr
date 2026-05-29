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
    scripts/bench_simd.py --optimize both
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

from bench_optimize import OPTIMIZE_CHOICES, fmt_ratio, optimize_slug, parse_optimize_modes

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
        # The default str() on CalledProcessError omits the captured output,
        # which makes CI failures of `zig build ...` (and similar) opaque.
        # Surface stdout + stderr verbatim before re-raising so the actual
        # error (e.g. a wasm trap printed by the underlying tool) is visible
        # in the workflow log.
        sys.stderr.write(
            f"\n[harness] command failed (exit {exc.returncode}): "
            f"{' '.join(str(c) for c in cmd)}\n"
        )
        if cwd is not None:
            sys.stderr.write(f"[harness]   cwd: {cwd}\n")
        if exc.stdout:
            sys.stderr.write("[harness] --- stdout ---\n")
            sys.stderr.write(exc.stdout)
            if not exc.stdout.endswith("\n"):
                sys.stderr.write("\n")
        if exc.stderr:
            sys.stderr.write("[harness] --- stderr ---\n")
            sys.stderr.write(exc.stderr)
            if not exc.stderr.endswith("\n"):
                sys.stderr.write("\n")
        sys.stderr.flush()
        raise
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


def worktree_env(wt: Path) -> dict:
    """Return an environment that keeps Zig caches isolated per worktree."""
    env = os.environ.copy()
    # setup-zig exports both cache dirs to the checkout's .zig-cache. This
    # script builds two temporary worktrees; sharing cache state can let the
    # target ref reuse baseline artifacts. Keep each ref's local and global Zig
    # caches independent.
    env.pop("ZIG_LOCAL_CACHE_DIR", None)
    global_cache = wt / ".zig-global-cache"
    global_cache.mkdir(exist_ok=True)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    return env


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
    wasmtime: bool,
    wasmtime_path: str,
    wasmtime_iterations: int | None,
    optimize: str,
) -> list[Measurement]:
    env = worktree_env(wt)
    overlay_harness(source_repo, wt)

    print(f"[harness] building {wt.name} ({optimize})", file=sys.stderr)
    run(["zig", "build", f"-Doptimize={optimize}"], cwd=wt, env=env)

    runner = wt / "zig-out/bin/simd-bench-runner"
    if not runner.exists():
        raise RuntimeError(f"expected runner was not built: {runner}")

    measurements: list[Measurement] = []
    for i in range(runs):
        print(
            f"[harness] running {wt.name} ({i + 1}/{runs}, iterations={iterations})",
            file=sys.stderr,
        )
        runner_args = [str(runner), "--iterations", str(iterations)]
        if wasmtime:
            runner_args.append("--wasmtime")
            runner_args.extend(["--wasmtime-path", wasmtime_path])
            if wasmtime_iterations is not None:
                runner_args.extend(["--wasmtime-iterations", str(wasmtime_iterations)])
        out = run(runner_args, cwd=wt, env=env)
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
    optimize: str = "ReleaseFast",
) -> str:
    baseline = summarize(baseline_rows)
    target = summarize(target_rows)
    keys = sorted(set(baseline) | set(target))

    lines = [
        "### SIMD AOT benchmark comparison" if optimize == "ReleaseFast" else f"### SIMD AOT benchmark comparison ({optimize})",
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
            "AOT rows with `unsupported` are expected for refs or SIMD opcode families without native v128 lowering.",
            "CoreMark is scalar, so this harness is the SIMD-specific signal for issue #220.",
        ]
    )
    return "\n".join(lines)




def render_optimize_table(
    target_ref: str,
    results: dict[str, list[Measurement]],
) -> str:
    summaries = {optimize: summarize(rows) for optimize, rows in results.items()}
    keys = sorted(set(summaries["ReleaseFast"]) | set(summaries["ReleaseSafe"]))
    lines = [
        "### SIMD AOT optimize-mode comparison",
        "",
        "| Case | Engine | Ref | Fast status | Safe status | Fast median run | Safe median run | Safe/Fast run | Fast median compile | Safe median compile | Safe/Fast compile |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, engine in keys:
        fast = summaries["ReleaseFast"].get((case, engine))
        safe = summaries["ReleaseSafe"].get((case, engine))
        fast_run = fast["run_ns"] if fast is not None else None
        safe_run = safe["run_ns"] if safe is not None else None
        fast_compile = fast["compile_ns"] if fast is not None else None
        safe_compile = safe["compile_ns"] if safe is not None else None
        lines.append(
            "| `{case}` | `{engine}` | `{ref}` (target) | {fast_status} | {safe_status} | {fast_run} | {safe_run} | {run_ratio} | {fast_compile} | {safe_compile} | {compile_ratio} |".format(
                case=case,
                engine=engine,
                ref=target_ref,
                fast_status=fast["status"] if fast is not None else "missing",
                safe_status=safe["status"] if safe is not None else "missing",
                fast_run=fmt_ns(fast_run),
                safe_run=fmt_ns(safe_run),
                run_ratio=fmt_ratio(safe_run, fast_run),
                fast_compile=fmt_ns(fast_compile),
                safe_compile=fmt_ns(safe_compile),
                compile_ratio=fmt_ratio(safe_compile, fast_compile),
            )
        )
    lines.extend(
        [
            "",
            "Safe/Fast ratios compare ReleaseSafe timing divided by ReleaseFast timing for the same target/case/engine.",
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
        "--wasmtime",
        action="store_true",
        help="include Wasmtime CLI rows as an external baseline",
    )
    p.add_argument(
        "--wasmtime-path",
        default="wasmtime",
        help="Wasmtime executable to use with --wasmtime",
    )
    p.add_argument(
        "--wasmtime-iterations",
        type=int,
        default=None,
        help="Wasmtime CLI invocations per run; defaults to min(iterations, 10)",
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
    p.add_argument(
        "--optimize",
        choices=OPTIMIZE_CHOICES,
        default="ReleaseFast",
        help="Zig optimize mode for wamr builds (default: ReleaseFast); `both` compares ReleaseFast vs ReleaseSafe for --target only",
    )
    args = p.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.wasmtime_iterations is not None and args.wasmtime_iterations <= 0:
        raise ValueError("--wasmtime-iterations must be positive")

    repo = args.repo.resolve()
    with tempfile.TemporaryDirectory(
        prefix="bench-simd-",
        dir="/work" if Path("/work").is_dir() else None,
    ) as tmp:
        root = Path(tmp)
        try:
            optimize_modes = parse_optimize_modes(args.optimize)
            if args.optimize == "both":
                optimize_results: dict[str, list[Measurement]] = {}
                for optimize in optimize_modes:
                    wt_t = make_worktree(repo, args.target, root, f"target-{optimize_slug(optimize)}")
                    optimize_results[optimize] = build_and_run(
                        wt_t,
                        repo,
                        args.runs,
                        args.iterations,
                        args.wasmtime,
                        args.wasmtime_path,
                        args.wasmtime_iterations,
                        optimize,
                    )
            else:
                only = optimize_modes[0]
                wt_b = make_worktree(repo, args.baseline, root, "baseline")
                wt_t = make_worktree(repo, args.target, root, "target")
                baseline_rows = build_and_run(
                    wt_b,
                    repo,
                    args.runs,
                    args.iterations,
                    args.wasmtime,
                    args.wasmtime_path,
                    args.wasmtime_iterations,
                    only,
                )
                target_rows = build_and_run(
                    wt_t,
                    repo,
                    args.runs,
                    args.iterations,
                    args.wasmtime,
                    args.wasmtime_path,
                    args.wasmtime_iterations,
                    only,
                )
        finally:
            run(["git", "worktree", "prune"], cwd=repo)

    if args.optimize == "both":
        table = render_optimize_table(args.target, optimize_results)
    else:
        only = parse_optimize_modes(args.optimize)[0]
        table = render_table(args.baseline, baseline_rows, args.target, target_rows, only)
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
