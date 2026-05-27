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

# Signature of the known x86_64 native AOT-run flake from issue #406:
# `wamr run` traps with `out of bounds memory access (..., local_func[N]+0x0, ...)`
# at the very first byte of a local function (or with a synthetic `[-1]`
# function index). The trap is non-deterministic on shared GitHub-hosted
# x86_64 runners — it has been observed firing on baseline `main` after a
# clean run of the same binary, proving it's not introduced by any single
# change. Real coremark regressions would trap deeper inside a function
# (non-zero offset), so a `+0x0` offset is the discriminator. Retry this
# failure mode up to `_TRAP_RETRY_MAX` times before treating it as a real
# regression.
_TRAP_FLAKE_PATTERN = re.compile(
    r"wasm trap: out of bounds memory access.*local_func\[-?\d+\]\+0x0",
    re.IGNORECASE,
)
_TRAP_RETRY_MAX = 3


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
        # which makes CI failures of `zig build run-aot` (and similar) opaque.
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


def worktree_env(wt: Path) -> dict:
    """Return an environment that keeps Zig caches isolated per worktree."""
    env = os.environ.copy()
    # setup-zig exports both cache dirs to the checkout's .zig-cache. That is
    # unsafe for this script because it builds multiple git worktrees in
    # sequence; the target ref can otherwise reuse baseline build artifacts.
    # Keep each ref's local and global Zig caches independent.
    env.pop("ZIG_LOCAL_CACHE_DIR", None)
    global_cache = wt / ".zig-global-cache"
    global_cache.mkdir(exist_ok=True)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    return env


def build_and_run(wt: Path, runs: int, coremark_src: Path) -> list[float]:
    """Build wamr + AOT-compile + run CoreMark `runs` times, return iter/s list."""
    env = worktree_env(wt)
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
        out = _run_aot_with_retry(cm, env, run_idx=i, runs_total=runs)
        m = ITER_PATTERN.search(out)
        if not m:
            raise RuntimeError(
                f"could not parse Iterations/Sec from CoreMark output:\n{out}"
            )
        val = float(m.group(1))
        print(f"[harness]   run {i + 1}/{runs}: {val:.1f} iter/s", file=sys.stderr)
        results.append(val)
    return results


def _run_aot_with_retry(cm: Path, env: dict, run_idx: int, runs_total: int) -> str:
    """Wrap `zig build run-aot` so the issue-#406 x86_64 trap flake doesn't
    fail the gate on the first incidence. Real failures still propagate
    after `_TRAP_RETRY_MAX` retries."""
    last_exc = None
    for attempt in range(1 + _TRAP_RETRY_MAX):
        try:
            return run(["zig", "build", "run-aot"], cwd=cm, env=env)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            if not _TRAP_FLAKE_PATTERN.search(stderr):
                # Not the known x86_64 trap flake — let it propagate.
                raise
            if attempt == _TRAP_RETRY_MAX:
                last_exc = exc
                break
            print(
                f"[harness]   run {run_idx + 1}/{runs_total}: x86_64 AOT trap flake "
                f"(issue #406) — retry {attempt + 1}/{_TRAP_RETRY_MAX}",
                file=sys.stderr,
            )
    # Re-raise the last exception so the original stderr + diagnostic
    # formatting from `run()` are visible.
    assert last_exc is not None
    raise last_exc


def fmt_stats(values: list[float]) -> tuple[float, float, float]:
    return statistics.fmean(values), min(values), max(values)


def compute_delta_pct(baseline_vals: list[float], target_vals: list[float]) -> float:
    return (statistics.fmean(target_vals) / statistics.fmean(baseline_vals) - 1.0) * 100.0


def render_table(
    baseline_ref: str,
    baseline_vals: list[float],
    target_ref: str,
    target_vals: list[float],
) -> str:
    bm, bmin, bmax = fmt_stats(baseline_vals)
    tm, tmin, tmax = fmt_stats(target_vals)
    delta_pct = compute_delta_pct(baseline_vals, target_vals)
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
    p.add_argument(
        "--min-delta-pct",
        type=float,
        default=None,
        help="fail if target mean is below baseline by more than this percent delta",
    )
    args = p.parse_args()

    repo = args.repo
    ensure_coremark_src(repo)
    coremark_src = repo / "tests/benchmarks/coremark/coremark"

    # Prefer the runner's own scratch dir (writable, already on the NVMe mount
    # under self-hosted runners). Fall back to /work only if writable — the
    # azure-nvme convention places /work on local disk but the runner user
    # may not own its root. Otherwise let tempfile pick the system default.
    runner_temp = os.environ.get("RUNNER_TEMP")
    if runner_temp and os.access(runner_temp, os.W_OK):
        tmp_root: str | None = runner_temp
    elif os.access("/work", os.W_OK):
        tmp_root = "/work"
    else:
        tmp_root = None
    with tempfile.TemporaryDirectory(prefix="bench-coremark-", dir=tmp_root) as tmp:
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

    if args.min_delta_pct is not None:
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
    sys.exit(main())
