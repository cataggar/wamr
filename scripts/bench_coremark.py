#!/usr/bin/env python3
"""Compare CoreMark AOT iter/s between two git refs.

Builds `wamr` + `wamrc` at each ref in a throwaway worktree, AOT-compiles the
CoreMark .wasm via `tests/benchmarks/coremark`, runs it N times, and prints a
markdown table with mean / min / max iter/s and delta %.

Intended for use both locally and from `.github/workflows/coremark-aarch64.yml`.
Requires the CoreMark sources at `tests/benchmarks/coremark/coremark/` (cloned
on first invocation).

Usage
-----
    scripts/bench_coremark.py --baseline origin/main --target HEAD --runs 3
    scripts/bench_coremark.py --baseline origin/main --target HEAD --emit github
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ITER_PATTERN = re.compile(r"Iterations/Sec\s*:\s*([0-9]+(?:\.[0-9]+)?)")


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


def ensure_coremark_src(repo: Path) -> None:
    src = repo / "tests/benchmarks/coremark/coremark"
    if (src / "core_main.c").exists():
        return
    print(f"[harness] cloning CoreMark sources into {src}", file=sys.stderr)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/eembc/coremark.git",
            str(src),
        ]
    )


def make_worktree(repo: Path, ref: str, root: Path) -> Path:
    """Create a fresh worktree at `ref` under `root` so concurrent builds don't fight."""
    sha = run(["git", "rev-parse", ref], cwd=repo).strip()
    wt = root / f"wt-{sha[:12]}"
    if wt.exists():
        run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo)
    run(["git", "worktree", "add", "--detach", str(wt), sha], cwd=repo)
    return wt


def build_and_run(wt: Path, runs: int, coremark_src: Path) -> list[float]:
    """Build wamr + AOT-compile + run CoreMark `runs` times, return iter/s list."""
    env = os.environ.copy()
    # Each worktree owns its own .zig-cache (already on /work for our runner).
    print(f"[harness] building {wt.name} (ReleaseFast)", file=sys.stderr)
    run(["zig", "build", "-Doptimize=ReleaseFast"], cwd=wt, env=env)

    cm = wt / "tests/benchmarks/coremark"
    # Symlink CoreMark sources from the canonical location to avoid re-cloning
    # per worktree.  Worktrees only contain tracked files; CoreMark sources
    # are external and not tracked.
    src_link = cm / "coremark"
    if not src_link.exists():
        src_link.symlink_to(coremark_src, target_is_directory=True)

    print(f"[harness] AOT-compiling CoreMark in {wt.name}", file=sys.stderr)
    run(["zig", "build", "aot"], cwd=cm, env=env)

    results: list[float] = []
    for i in range(runs):
        out = run(["zig", "build", "run-aot"], cwd=cm, env=env)
        m = ITER_PATTERN.search(out)
        if not m:
            raise RuntimeError(
                f"could not parse Iterations/Sec from CoreMark output:\n{out}"
            )
        val = float(m.group(1))
        print(f"[harness]   run {i + 1}/{runs}: {val:.1f} iter/s", file=sys.stderr)
        results.append(val)
    return results


def fmt_stats(values: list[float]) -> tuple[float, float, float]:
    return statistics.fmean(values), min(values), max(values)


def render_table(
    baseline_ref: str,
    baseline_vals: list[float],
    target_ref: str,
    target_vals: list[float],
) -> str:
    bm, bmin, bmax = fmt_stats(baseline_vals)
    tm, tmin, tmax = fmt_stats(target_vals)
    delta_pct = (tm / bm - 1.0) * 100.0
    sign = "+" if delta_pct >= 0 else ""
    lines = [
        "### CoreMark AOT comparison",
        "",
        f"| Ref | Mean iter/s | Min | Max | Runs |",
        f"|---|---:|---:|---:|---:|",
        f"| `{baseline_ref}` (baseline) | {bm:.1f} | {bmin:.1f} | {bmax:.1f} | {len(baseline_vals)} |",
        f"| `{target_ref}` (target) | {tm:.1f} | {tmin:.1f} | {tmax:.1f} | {len(target_vals)} |",
        f"| **Δ** | **{sign}{delta_pct:.2f}%** | | | |",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", default="origin/main", help="git ref for the baseline")
    p.add_argument("--target", default="HEAD", help="git ref for the target")
    p.add_argument("--runs", type=int, default=3, help="runs per ref (default 3)")
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

    repo = args.repo
    ensure_coremark_src(repo)
    coremark_src = repo / "tests/benchmarks/coremark/coremark"

    with tempfile.TemporaryDirectory(prefix="bench-coremark-", dir="/work" if Path("/work").is_dir() else None) as tmp:
        root = Path(tmp)
        try:
            wt_b = make_worktree(repo, args.baseline, root)
            wt_t = make_worktree(repo, args.target, root)

            baseline_vals = build_and_run(wt_b, args.runs, coremark_src)
            target_vals = build_and_run(wt_t, args.runs, coremark_src)
        finally:
            # Clean up worktrees so the parent repo isn't left with stale refs.
            run(["git", "worktree", "prune"], cwd=repo)

    table = render_table(args.baseline, baseline_vals, args.target, target_vals)
    print(table)

    if args.out:
        args.out.write_text(table + "\n")

    if args.emit == "github":
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as fh:
                fh.write(table + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
