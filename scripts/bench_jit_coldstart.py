#!/usr/bin/env python3
"""Compare cold-start latency and steady-state throughput across every
execution mode this runtime exposes via its CLI: two-step AOT (cold and
warm), `wamrc run` (one-step compile+execute), the in-process JIT (#852-861,
both the `.fast` default preset and the `.full` preset), and wasmtime's
default JIT — for issue #861.

Unlike `bench_coldstart.py` / `bench_coremark.py` (which compare two git
refs of *this* runtime, always via a precompiled `.cwasm`), this script
compares *execution modes* of a single build, since #861 is about
characterizing the JIT feature itself rather than catching a regression
between two commits. A single `-Djit=true` build is used for every wamr-side
measurement: `-Djit=true` only *adds* the in-process compile+run path: the
`wamrc compile` / `wamr run <cwasm>` two-step flow is unaffected by the
build flag (verified in #860's PR: byte-identical `.cwasm` output).

Note: this runtime's CLI is AOT/JIT-only (#644) — a plain `.wasm` module
run without `-Djit=true` errors out asking the caller to precompile or
rebuild with `-Djit=true`; there is no CLI-accessible interpreter mode
(the interpreter is only reachable via the library API). That axis is
therefore intentionally omitted below rather than faked.

Usage
-----
::

    scripts/bench_jit_coldstart.py
    scripts/bench_jit_coldstart.py --samples 30 --warmup 5 --wasmtime-path ~/.wasmtime/bin/wasmtime
    scripts/bench_jit_coldstart.py --skip-coremark
"""

from __future__ import annotations

import argparse
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

REPO = Path(__file__).resolve().parent.parent

ITER_PATTERN = re.compile(r"Iterations/Sec\s*:\s*([0-9]+(?:\.[0-9]+)?)")


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, cwd=cwd, env=env, check=check, text=True, capture_output=True
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(
            f"\n[harness] command failed (exit {exc.returncode}): {' '.join(cmd)}\n"
        )
        if exc.stdout:
            sys.stderr.write(f"[harness] --- stdout ---\n{exc.stdout}\n")
        if exc.stderr:
            sys.stderr.write(f"[harness] --- stderr ---\n{exc.stderr}\n")
        raise


def worktree_env(wt: Path) -> dict:
    env = os.environ.copy()
    env.pop("ZIG_LOCAL_CACHE_DIR", None)
    global_cache = wt / ".zig-global-cache"
    global_cache.mkdir(exist_ok=True)
    env["ZIG_GLOBAL_CACHE_DIR"] = str(global_cache)
    return env


def host_cpu_model() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if line.lower().startswith("model name:"):
                val = line.split(":", 1)[1].strip()
                if val:
                    return val
    except Exception:
        pass
    proc = platform.processor()
    return proc if proc else "unknown CPU"


def host_info() -> str:
    arch = platform.machine() or "unknown-arch"
    ncpu = os.cpu_count()
    model = host_cpu_model()
    return f"_Host: arch `{arch}` · {ncpu or '?'} vCPU · `{model}`_"


@dataclass
class Sample:
    label: str
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    n: int


def time_cmds(label: str, cmd_batches: list[list[list[str]]], *, cwd: Path, env: dict, samples: int, warmup: int) -> Sample:
    """Time `samples` invocations of a (possibly multi-command) batch,
    discarding `warmup` invocations first. Each element of `cmd_batches`
    is itself a list of one-or-more commands run back to back (e.g. AOT
    cold = [wamrc compile, wamr run] timed together as one sample)."""
    durations_ms: list[float] = []
    total = warmup + samples
    for i in range(total):
        start = time.perf_counter()
        for cmd in cmd_batches[i % len(cmd_batches)]:
            run(cmd, cwd=cwd, env=env)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if i >= warmup:
            durations_ms.append(elapsed_ms)
    durations_ms.sort()
    n = len(durations_ms)
    p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
    return Sample(
        label=label,
        median_ms=statistics.median(durations_ms),
        p95_ms=durations_ms[p95_idx],
        min_ms=durations_ms[0],
        max_ms=durations_ms[-1],
        n=n,
    )


def coremark_iters_per_sec(cmd: list[str], cwd: Path, env: dict) -> float:
    out = run(cmd, cwd=cwd, env=env).stdout
    m = ITER_PATTERN.search(out)
    if not m:
        raise RuntimeError(f"could not parse Iterations/Sec from output:\n{out}")
    return float(m.group(1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=20, help="timed invocations per mode (default 20)")
    p.add_argument("--warmup", type=int, default=3, help="warmup invocations discarded before timing (default 3)")
    p.add_argument("--coremark-runs", type=int, default=3, help="CoreMark runs per mode (default 3)")
    p.add_argument("--skip-coremark", action="store_true", help="skip the CoreMark throughput comparison")
    p.add_argument("--wasmtime-path", default=None, help="wasmtime executable (default: PATH lookup, then ~/.wasmtime/bin/wasmtime)")
    p.add_argument("--out", default=None, help="write the markdown report here as well as stdout")
    args = p.parse_args()

    wasmtime_path = args.wasmtime_path or shutil.which("wasmtime")
    if wasmtime_path is None:
        candidate = Path.home() / ".wasmtime/bin/wasmtime"
        if candidate.exists():
            wasmtime_path = str(candidate)
    if wasmtime_path is None:
        print("[harness] warning: wasmtime not found; wasmtime columns will be omitted", file=sys.stderr)

    env = worktree_env(REPO)
    print("[harness] building wamr/wamrc (-Djit=true -Doptimize=ReleaseFast)", file=sys.stderr)
    run(["zig", "build", "-Djit=true", "-Doptimize=ReleaseFast"], cwd=REPO, env=env)
    wamr = REPO / "zig-out/bin/wamr"
    wamrc = REPO / "zig-out/bin/wamrc"
    assert wamr.exists() and wamrc.exists()

    workdir = REPO / "bench-jit-coldstart-tmp"
    workdir.mkdir(exist_ok=True)
    noop_wasm = workdir / "noop.wasm"
    shutil.copy2(REPO / "tests/coldstart/noop.wasm", noop_wasm)
    noop_cwasm = workdir / "noop.cwasm"
    run([str(wamrc), "compile", str(noop_wasm), "-o", str(noop_cwasm)], cwd=workdir, env=env)

    full_opt_env = dict(env)
    full_opt_env["WAMR_JIT_FULL_OPT"] = "1"

    print(f"[harness] timing cold-start ({args.samples} samples, {args.warmup} warmup)...", file=sys.stderr)
    samples: list[Sample] = []

    samples.append(time_cmds(
        "AOT cold (wamrc compile + wamr run, no cache)",
        [[[str(wamrc), "compile", str(noop_wasm), "-o", str(workdir / "noop_cold.cwasm")],
          [str(wamr), "run", str(workdir / "noop_cold.cwasm")]]],
        cwd=workdir, env=env, samples=args.samples, warmup=args.warmup,
    ))
    samples.append(time_cmds(
        "AOT warm (wamr run, precompiled .cwasm)",
        [[[str(wamr), "run", str(noop_cwasm)]]],
        cwd=workdir, env=env, samples=args.samples, warmup=args.warmup,
    ))
    samples.append(time_cmds(
        "wamrc run (one-step compile+execute, no persistent cache)",
        [[[str(wamrc), "run", str(noop_wasm)]]],
        cwd=workdir, env=env, samples=args.samples, warmup=args.warmup,
    ))
    samples.append(time_cmds(
        "in-process JIT, .fast preset (default)",
        [[[str(wamr), "run", str(noop_wasm)]]],
        cwd=workdir, env=env, samples=args.samples, warmup=args.warmup,
    ))
    samples.append(time_cmds(
        "in-process JIT, .full preset (WAMR_JIT_FULL_OPT=1)",
        [[[str(wamr), "run", str(noop_wasm)]]],
        cwd=workdir, env=full_opt_env, samples=args.samples, warmup=args.warmup,
    ))
    if wasmtime_path:
        samples.append(time_cmds(
            "wasmtime run (default JIT)",
            [[[wasmtime_path, "run", str(noop_wasm)]]],
            cwd=workdir, env=env, samples=args.samples, warmup=args.warmup,
        ))

    lines: list[str] = []
    lines.append(f"# JIT cold-start comparison — {time.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(host_info())
    lines.append("")
    lines.append("## Cold-start (noop module, subprocess wall time)")
    lines.append("")
    lines.append("| Mode | Median | p95 | Min | Max | Samples |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in samples:
        lines.append(
            f"| {s.label} | {s.median_ms:.3f} ms | {s.p95_ms:.3f} ms | "
            f"{s.min_ms:.3f} ms | {s.max_ms:.3f} ms | {s.n} |"
        )
    lines.append("")

    if not args.skip_coremark:
        print(f"[harness] timing CoreMark throughput ({args.coremark_runs} runs per mode)...", file=sys.stderr)
        cm_wasm = REPO / "tests/benchmarks/coremark/coremark_wasi.wasm"
        cm_cwasm = workdir / "coremark.cwasm"
        run([str(wamrc), "compile", str(cm_wasm), "-o", str(cm_cwasm)], cwd=workdir, env=env)

        def cm_runs(cmd: list[str], run_env: dict) -> list[float]:
            vals = []
            for i in range(args.coremark_runs):
                v = coremark_iters_per_sec(cmd, workdir, run_env)
                print(f"[harness]   run {i + 1}/{args.coremark_runs}: {v:.1f} iter/s", file=sys.stderr)
                vals.append(v)
            return vals

        cm_results: list[tuple[str, list[float]]] = []
        cm_results.append(("AOT (precompiled .cwasm)", cm_runs([str(wamr), "run", str(cm_cwasm)], env)))
        cm_results.append((".fast preset (in-process JIT default)", cm_runs([str(wamr), "run", str(cm_wasm)], env)))
        cm_results.append((".full preset (in-process JIT, WAMR_JIT_FULL_OPT=1)", cm_runs([str(wamr), "run", str(cm_wasm)], full_opt_env)))
        if wasmtime_path:
            cm_results.append(("wasmtime (default JIT)", cm_runs([wasmtime_path, "run", str(cm_wasm)], env)))

        lines.append("## Steady-state throughput (CoreMark)")
        lines.append("")
        lines.append("| Mode | Mean iter/s | Min | Max | Runs |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, vals in cm_results:
            mean = statistics.fmean(vals)
            lines.append(f"| {label} | {mean:.1f} | {min(vals):.1f} | {max(vals):.1f} | {len(vals)} |")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")

    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
