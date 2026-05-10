#!/usr/bin/env python3
"""Compare CLI cold-start latency between two git refs.

Builds ``wamr`` + ``wamrc`` at each ref in a throwaway worktree, then spawns
the CLI ``--samples`` times against a tiny noop module and reports
median/min/max/p95/σ. Optionally adds Wasmtime columns and a synthesized
``add100`` module that surfaces JIT-startup cost more clearly.

Companion to the in-process budget tests in ``src/tests/coldstart_test.zig``
(issue #395). Subprocess timing is the user-visible total (process spawn +
runtime startup + execute + exit), and is what regressions in eager
initialization actually move.

Intended for use both locally and from
``.github/workflows/coldstart-aarch64.yml``.

Usage
-----
::

    scripts/bench_coldstart.py --baseline origin/main --target HEAD
    scripts/bench_coldstart.py --baseline origin/main --target HEAD \\
        --samples 50 --warmup 5 --regression-ratio 1.5 \\
        --out coldstart-table.md --json coldstart.json --emit github
    scripts/bench_coldstart.py --target HEAD --wasmtime --include-jit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


# Modules in the order they should appear in tables / JSON output.
NOOP_MODULE = "noop"
ADD100_MODULE = "add100"

# Engines in the order they should appear in tables.
ENGINE_TARGET = "wamr-target"
ENGINE_BASELINE = "wamr-baseline"
ENGINE_WASMTIME_AOT = "wasmtime-aot"
ENGINE_WASMTIME_JIT = "wasmtime-jit"

WAMR_ENGINES = (ENGINE_TARGET, ENGINE_BASELINE)


# ---------------------------------------------------------------------------
# Subprocess + worktree helpers (mirroring bench_coremark.py / bench_simd.py).
# ---------------------------------------------------------------------------


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd``, surfacing stdout+stderr verbatim on failure."""
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=check,
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


def make_worktree(repo: Path, ref: str, root: Path, label: str) -> Path:
    """Create a fresh detached worktree at ``ref`` under ``root``."""
    sha = run(["git", "rev-parse", ref], cwd=repo).stdout.strip()
    wt = root / f"{label}-{sha[:12]}"
    if wt.exists():
        try:
            run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo)
        except subprocess.CalledProcessError:
            shutil.rmtree(wt)
    run(["git", "worktree", "add", "--detach", str(wt), sha], cwd=repo)
    return wt


def worktree_env(wt: Path) -> dict:
    """Return an environment that keeps Zig caches isolated per worktree.

    See repo memory: multi-ref perf worktrees must unset
    ``ZIG_LOCAL_CACHE_DIR`` and set per-worktree ``ZIG_GLOBAL_CACHE_DIR``.
    Sharing caches between baseline and target lets the second build reuse
    the first's artifacts and hides or fabricates timing deltas.
    """
    env = os.environ.copy()
    env.pop("ZIG_LOCAL_CACHE_DIR", None)
    global_cache = wt / ".zig-global-cache"
    global_cache.mkdir(exist_ok=True)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    return env


def build_worktree(wt: Path) -> dict:
    """``zig build -Doptimize=ReleaseFast`` and return the env to reuse."""
    env = worktree_env(wt)
    print(f"[harness] building {wt.name} (ReleaseFast)", file=sys.stderr)
    run(["zig", "build", "-Doptimize=ReleaseFast"], cwd=wt, env=env)
    wamr = wt / "zig-out/bin/wamr"
    wamrc = wt / "zig-out/bin/wamrc"
    if not wamr.exists():
        raise RuntimeError(f"missing {wamr} after build")
    if not wamrc.exists():
        raise RuntimeError(f"missing {wamrc} after build")
    return env


# ---------------------------------------------------------------------------
# Module preparation: noop.wasm/cwasm + optional add100.wasm.
# ---------------------------------------------------------------------------


def find_wasm_tools() -> str | None:
    return shutil.which("wasm-tools")


def find_wasmtime(override: str | None) -> str | None:
    if override is None:
        return shutil.which("wasmtime")
    candidate = shutil.which(override) or override
    return candidate if Path(candidate).exists() else None


def synthesize_add100_wat() -> str:
    """A ~1 KB module: 99 helper i32.add functions + an exported ``_start``.

    Surfaces JIT-startup cost more clearly than the 36-byte noop, while
    staying small enough that AOT load time still dominates execution.
    """
    parts: list[str] = ["(module"]
    for i in range(99):
        parts.append(
            f"  (func $f{i} (param i32 i32) (result i32) "
            "local.get 0 local.get 1 i32.add)"
        )
    parts.append('  (func (export "_start") (result i32) i32.const 0)')
    parts.append(")")
    return "\n".join(parts) + "\n"


def prepare_modules(
    repo: Path,
    target_wt: Path,
    target_env: dict,
    workdir: Path,
    *,
    include_jit: bool,
    use_wasmtime: bool,
    wasmtime_path: str | None,
) -> dict[str, dict[str, Path]]:
    """Materialize all module variants used by the timing loop.

    Returns ``{module_name: {kind: path}}``. ``kind`` is one of
    ``wasm``, ``wamr_cwasm``, ``wasmtime_cwasm``.
    """
    modules: dict[str, dict[str, Path]] = {}
    workdir.mkdir(parents=True, exist_ok=True)

    # --- noop.wasm: reuse the committed 36-byte fixture ----------------
    src_noop = repo / "tests/coldstart/noop.wasm"
    if not src_noop.exists():
        raise RuntimeError(f"missing committed noop fixture at {src_noop}")
    noop_wasm = workdir / "noop.wasm"
    shutil.copy2(src_noop, noop_wasm)

    # --- noop.cwasm via the *target* wamrc ----------------------------
    target_wamrc = target_wt / "zig-out/bin/wamrc"
    noop_cwasm = workdir / "wamr" / "noop.cwasm"
    noop_cwasm.parent.mkdir(parents=True, exist_ok=True)
    print("[harness] AOT-compiling noop.cwasm via target wamrc", file=sys.stderr)
    run(
        [str(target_wamrc), "compile", str(noop_wasm), "-o", str(noop_cwasm)],
        cwd=workdir,
        env=target_env,
    )

    entry: dict[str, Path] = {"wasm": noop_wasm, "wamr_cwasm": noop_cwasm}

    # --- noop.cwasm via wasmtime (if requested) -----------------------
    if use_wasmtime and wasmtime_path is not None:
        wt_cwasm = workdir / "wasmtime" / "noop.cwasm"
        wt_cwasm.parent.mkdir(parents=True, exist_ok=True)
        print("[harness] AOT-compiling noop.cwasm via wasmtime", file=sys.stderr)
        run(
            [wasmtime_path, "compile", "-o", str(wt_cwasm), str(noop_wasm)],
            cwd=workdir,
        )
        entry["wasmtime_cwasm"] = wt_cwasm

    modules[NOOP_MODULE] = entry

    # --- add100 (optional, behind --include-jit) ----------------------
    if include_jit:
        ws = find_wasm_tools()
        if ws is None:
            print(
                "[harness] --include-jit requested but `wasm-tools` is not on "
                "PATH; skipping add100 module",
                file=sys.stderr,
            )
        else:
            wat = workdir / "add100.wat"
            wat.write_text(synthesize_add100_wat())
            wasm = workdir / "add100.wasm"
            print("[harness] synthesizing add100.wasm", file=sys.stderr)
            run([ws, "parse", str(wat), "-o", str(wasm)], cwd=workdir)

            cwasm = workdir / "wamr" / "add100.cwasm"
            run(
                [str(target_wamrc), "compile", str(wasm), "-o", str(cwasm)],
                cwd=workdir,
                env=target_env,
            )

            jit_entry: dict[str, Path] = {"wasm": wasm, "wamr_cwasm": cwasm}
            if use_wasmtime and wasmtime_path is not None:
                wt_cwasm = workdir / "wasmtime" / "add100.cwasm"
                run(
                    [wasmtime_path, "compile", "-o", str(wt_cwasm), str(wasm)],
                    cwd=workdir,
                )
                jit_entry["wasmtime_cwasm"] = wt_cwasm
            modules[ADD100_MODULE] = jit_entry

    return modules


# ---------------------------------------------------------------------------
# Timing loop + statistics.
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    engine: str
    module: str
    variant: str  # "wasm" or "cwasm"
    elapsed_ns: int


@dataclass
class Summary:
    engine: str
    module: str
    variant: str
    samples: int
    median_ns: int
    min_ns: int
    max_ns: int
    p95_ns: int
    stdev_ns: int
    raw_ns: list[int] = field(default_factory=list)


def time_invocation(cmd: list[str]) -> int:
    """Spawn ``cmd`` once and return wall-clock ns. Raises on non-zero exit."""
    start = time.perf_counter_ns()
    proc = subprocess.run(cmd, capture_output=True, check=False)
    elapsed = time.perf_counter_ns() - start
    if proc.returncode != 0:
        sys.stderr.write(
            f"[harness] cold-start invocation failed (exit "
            f"{proc.returncode}): {' '.join(cmd)}\n"
        )
        if proc.stdout:
            sys.stderr.write(proc.stdout.decode("utf-8", "replace"))
        if proc.stderr:
            sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        sys.stderr.flush()
        raise RuntimeError(f"cold-start invocation returned {proc.returncode}")
    return elapsed


def time_engine_module(
    label: str,
    cmd: list[str],
    samples: int,
    warmup: int,
    engine: str,
    module: str,
    variant: str,
) -> Summary:
    """Run ``cmd`` ``samples + warmup`` times, summarize discarding warmups."""
    print(
        f"[harness] timing {label} ({engine}/{module}/{variant}) "
        f"warmup={warmup} samples={samples}",
        file=sys.stderr,
    )
    raw: list[int] = []
    total = samples + warmup
    for i in range(total):
        elapsed = time_invocation(cmd)
        if i >= warmup:
            raw.append(elapsed)
    raw_sorted = sorted(raw)
    median_ns = int(statistics.median(raw_sorted))
    p95_idx = max(0, int(round(0.95 * (len(raw_sorted) - 1))))
    p95_ns = int(raw_sorted[p95_idx])
    stdev_ns = int(statistics.stdev(raw_sorted)) if len(raw_sorted) > 1 else 0
    return Summary(
        engine=engine,
        module=module,
        variant=variant,
        samples=len(raw),
        median_ns=median_ns,
        min_ns=int(raw_sorted[0]),
        max_ns=int(raw_sorted[-1]),
        p95_ns=p95_ns,
        stdev_ns=stdev_ns,
        raw_ns=raw,
    )


def collect_summaries(
    *,
    target_wamr: Path,
    baseline_wamr: Path,
    wasmtime_path: str | None,
    modules: dict[str, dict[str, Path]],
    samples: int,
    warmup: int,
) -> list[Summary]:
    out: list[Summary] = []
    for module_name, paths in modules.items():
        wasm = paths["wasm"]
        wamr_cwasm = paths["wamr_cwasm"]

        # wamr-target: cwasm + wasm
        out.append(
            time_engine_module(
                f"{target_wamr} run {wamr_cwasm.name}",
                [str(target_wamr), "run", str(wamr_cwasm)],
                samples, warmup, ENGINE_TARGET, module_name, "cwasm",
            )
        )
        out.append(
            time_engine_module(
                f"{target_wamr} run {wasm.name}",
                [str(target_wamr), "run", str(wasm)],
                samples, warmup, ENGINE_TARGET, module_name, "wasm",
            )
        )

        # wamr-baseline: cwasm + wasm. The baseline build's wamr binary
        # is paired with the *baseline* wamrc when AOT-compiling, since
        # cwasm format is engine-version specific. Compile a dedicated
        # baseline cwasm next to the baseline wamr.
        baseline_cwasm = baseline_wamr.parent.parent / "_coldstart" / f"{module_name}.cwasm"
        baseline_cwasm.parent.mkdir(parents=True, exist_ok=True)
        baseline_wamrc = baseline_wamr.parent / "wamrc"
        if not baseline_cwasm.exists():
            print(
                f"[harness] AOT-compiling {module_name}.cwasm via baseline wamrc",
                file=sys.stderr,
            )
            run(
                [str(baseline_wamrc), "compile", str(wasm), "-o", str(baseline_cwasm)],
                cwd=baseline_wamr.parent,
            )
        out.append(
            time_engine_module(
                f"{baseline_wamr} run {baseline_cwasm.name}",
                [str(baseline_wamr), "run", str(baseline_cwasm)],
                samples, warmup, ENGINE_BASELINE, module_name, "cwasm",
            )
        )
        out.append(
            time_engine_module(
                f"{baseline_wamr} run {wasm.name}",
                [str(baseline_wamr), "run", str(wasm)],
                samples, warmup, ENGINE_BASELINE, module_name, "wasm",
            )
        )

        if wasmtime_path is not None and "wasmtime_cwasm" in paths:
            wt_cwasm = paths["wasmtime_cwasm"]
            out.append(
                time_engine_module(
                    f"{wasmtime_path} {wt_cwasm.name}",
                    [
                        wasmtime_path,
                        "run",
                        "--allow-precompiled",
                        str(wt_cwasm),
                    ],
                    samples, warmup, ENGINE_WASMTIME_AOT, module_name, "cwasm",
                )
            )
            out.append(
                time_engine_module(
                    f"{wasmtime_path} {wasm.name}",
                    [wasmtime_path, "run", str(wasm)],
                    samples, warmup, ENGINE_WASMTIME_JIT, module_name, "wasm",
                )
            )

    return out


# ---------------------------------------------------------------------------
# Rendering + regression detection.
# ---------------------------------------------------------------------------


def fmt_ms(ns: int | float | None) -> str:
    if ns is None:
        return "—"
    return f"{ns / 1_000_000:.3f}"


def lookup(
    summaries: list[Summary],
    engine: str,
    module: str,
    variant: str,
) -> Summary | None:
    for s in summaries:
        if s.engine == engine and s.module == module and s.variant == variant:
            return s
    return None


def render_table(
    baseline_ref: str,
    target_ref: str,
    summaries: list[Summary],
    wasmtime_path: str | None,
) -> str:
    """Markdown table grouped by (module, variant)."""
    modules_seen: list[str] = []
    for s in summaries:
        if s.module not in modules_seen:
            modules_seen.append(s.module)

    engines: list[tuple[str, str]] = [
        (ENGINE_TARGET, f"target (`{target_ref}`)"),
        (ENGINE_BASELINE, f"baseline (`{baseline_ref}`)"),
    ]
    if wasmtime_path is not None:
        engines.append((ENGINE_WASMTIME_AOT, "wasmtime (precompiled)"))
        engines.append((ENGINE_WASMTIME_JIT, "wasmtime (JIT)"))

    lines: list[str] = ["### Cold-start CLI comparison (median ms)", ""]
    lines.append(
        "| Module | Variant | Engine | Median | Min | Max | p95 | σ | Δ vs baseline |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")

    for module in modules_seen:
        for variant in ("cwasm", "wasm"):
            base = lookup(summaries, ENGINE_BASELINE, module, variant)
            for engine_id, engine_label in engines:
                s = lookup(summaries, engine_id, module, variant)
                if s is None:
                    continue
                if engine_id == ENGINE_BASELINE or base is None:
                    delta = "—"
                else:
                    ratio = s.median_ns / base.median_ns
                    pct = (ratio - 1.0) * 100.0
                    sign = "+" if pct >= 0 else ""
                    delta = f"{sign}{pct:.1f}% (×{ratio:.2f})"
                lines.append(
                    f"| `{module}` | `{variant}` | {engine_label} | "
                    f"{fmt_ms(s.median_ns)} | {fmt_ms(s.min_ns)} | "
                    f"{fmt_ms(s.max_ns)} | {fmt_ms(s.p95_ns)} | "
                    f"{fmt_ms(s.stdev_ns)} | {delta} |"
                )
    return "\n".join(lines)


@dataclass
class Regression:
    module: str
    variant: str
    target_median_ns: int
    baseline_median_ns: int
    ratio: float
    threshold: float
    kind: str  # "ratio" or "budget"


def find_regressions(
    summaries: list[Summary],
    *,
    regression_ratio: float,
    budget_ms: float | None,
) -> list[Regression]:
    regressions: list[Regression] = []
    modules: list[str] = []
    for s in summaries:
        if s.module not in modules:
            modules.append(s.module)

    for module in modules:
        for variant in ("cwasm", "wasm"):
            t = lookup(summaries, ENGINE_TARGET, module, variant)
            b = lookup(summaries, ENGINE_BASELINE, module, variant)
            if t is None or b is None:
                continue
            ratio = t.median_ns / b.median_ns
            if ratio > regression_ratio:
                regressions.append(
                    Regression(
                        module=module,
                        variant=variant,
                        target_median_ns=t.median_ns,
                        baseline_median_ns=b.median_ns,
                        ratio=ratio,
                        threshold=regression_ratio,
                        kind="ratio",
                    )
                )
            if budget_ms is not None and t.median_ns > budget_ms * 1_000_000:
                regressions.append(
                    Regression(
                        module=module,
                        variant=variant,
                        target_median_ns=t.median_ns,
                        baseline_median_ns=b.median_ns,
                        ratio=t.median_ns / (budget_ms * 1_000_000),
                        threshold=budget_ms,
                        kind="budget",
                    )
                )
    return regressions


def write_json(
    path: Path,
    *,
    samples: int,
    warmup: int,
    baseline_ref: str,
    target_ref: str,
    summaries: list[Summary],
    regressions: list[Regression],
    regression_ratio: float,
    budget_ms: float | None,
) -> None:
    payload = {
        "samples": samples,
        "warmup": warmup,
        "baseline_ref": baseline_ref,
        "target_ref": target_ref,
        "regression_ratio": regression_ratio,
        "budget_ms": budget_ms,
        "results": [
            {
                "engine": s.engine,
                "module": s.module,
                "variant": s.variant,
                "samples": s.samples,
                "median_ns": s.median_ns,
                "min_ns": s.min_ns,
                "max_ns": s.max_ns,
                "p95_ns": s.p95_ns,
                "stdev_ns": s.stdev_ns,
                "raw_ns": s.raw_ns,
            }
            for s in summaries
        ],
        "regressions": [
            {
                "module": r.module,
                "variant": r.variant,
                "target_median_ns": r.target_median_ns,
                "baseline_median_ns": r.baseline_median_ns,
                "ratio": r.ratio,
                "threshold": r.threshold,
                "kind": r.kind,
            }
            for r in regressions
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", default="origin/main", help="git ref for the baseline")
    p.add_argument("--target", default="HEAD", help="git ref for the target")
    p.add_argument("--samples", type=int, default=30, help="timed invocations per (engine, module, variant)")
    p.add_argument("--warmup", type=int, default=3, help="warmup invocations per (engine, module, variant) discarded before timing")
    p.add_argument("--wasmtime", action="store_true", help="include wasmtime columns")
    p.add_argument("--wasmtime-path", default=None, help="wasmtime executable to use with --wasmtime (default: PATH lookup)")
    p.add_argument("--include-jit", action="store_true", help="also run a synthesized add100 module to surface JIT-startup cost")
    p.add_argument("--budget-ms", type=float, default=None, help="optional absolute backstop: fail if any wamr-target median exceeds this")
    p.add_argument("--regression-ratio", type=float, default=1.5, help="fail if any wamr target_median / baseline_median exceeds this ratio (default 1.5)")
    p.add_argument("--json", dest="json_path", type=Path, default=None, help="write machine-readable results to this file")
    p.add_argument("--out", type=Path, default=None, help="if given, write the markdown table here as well")
    p.add_argument("--emit", choices=["markdown", "github"], default="markdown", help="`github` also appends to $GITHUB_STEP_SUMMARY when present")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1], help="path to wamr repo (default: parent of scripts/)")
    args = p.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.regression_ratio <= 0:
        raise ValueError("--regression-ratio must be positive")
    if args.budget_ms is not None and args.budget_ms <= 0:
        raise ValueError("--budget-ms must be positive")

    repo = args.repo.resolve()

    wasmtime_path: str | None = None
    if args.wasmtime:
        wasmtime_path = find_wasmtime(args.wasmtime_path)
        if wasmtime_path is None:
            print(
                "[harness] --wasmtime requested but no wasmtime binary "
                "found; continuing without wasmtime rows",
                file=sys.stderr,
            )

    tmp_root = Path("/work") if Path("/work").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="bench-coldstart-", dir=tmp_root) as tmp:
        root = Path(tmp)
        try:
            wt_t = make_worktree(repo, args.target, root, "target")
            wt_b = make_worktree(repo, args.baseline, root, "baseline")

            target_env = build_worktree(wt_t)
            build_worktree(wt_b)

            modules = prepare_modules(
                repo,
                wt_t,
                target_env,
                root / "modules",
                include_jit=args.include_jit,
                use_wasmtime=args.wasmtime,
                wasmtime_path=wasmtime_path,
            )

            summaries = collect_summaries(
                target_wamr=wt_t / "zig-out/bin/wamr",
                baseline_wamr=wt_b / "zig-out/bin/wamr",
                wasmtime_path=wasmtime_path,
                modules=modules,
                samples=args.samples,
                warmup=args.warmup,
            )
        finally:
            run(["git", "worktree", "prune"], cwd=repo, check=False)

    table = render_table(args.baseline, args.target, summaries, wasmtime_path)
    print(table)

    regressions = find_regressions(
        summaries,
        regression_ratio=args.regression_ratio,
        budget_ms=args.budget_ms,
    )

    if regressions:
        print("", file=sys.stderr)
        print("[harness] regressions detected:", file=sys.stderr)
        for r in regressions:
            if r.kind == "ratio":
                print(
                    f"  - {r.module}/{r.variant}: "
                    f"{fmt_ms(r.target_median_ns)} ms vs baseline "
                    f"{fmt_ms(r.baseline_median_ns)} ms "
                    f"(×{r.ratio:.2f} > {r.threshold:.2f})",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  - {r.module}/{r.variant}: "
                    f"{fmt_ms(r.target_median_ns)} ms exceeds budget "
                    f"{r.threshold:.2f} ms (×{r.ratio:.2f})",
                    file=sys.stderr,
                )

    if args.out:
        args.out.write_text(table + "\n")

    if args.emit == "github":
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(table + "\n")

    if args.json_path is not None:
        write_json(
            args.json_path,
            samples=args.samples,
            warmup=args.warmup,
            baseline_ref=args.baseline,
            target_ref=args.target,
            summaries=summaries,
            regressions=regressions,
            regression_ratio=args.regression_ratio,
            budget_ms=args.budget_ms,
        )

    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
