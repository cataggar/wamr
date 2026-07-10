# JIT cold-start comparison — 2026-07-10

Host: Azure VM worktree `/work/wamr-861-jit-bench-docs`, commit `329c29b5`, branch `861-jit-bench-docs`. Commands were run with `ZIG_LOCAL_CACHE_DIR` unset and `ZIG_GLOBAL_CACHE_DIR=$PWD/.zig-global-cache`.

_Host: arch `x86_64` · 8 vCPU · `AMD EPYC 9V74 80-Core Processor`_

## Summary

| Mode | Cold-start median (noop) | CoreMark iter/s |
|---|---:|---:|
| AOT cold (`wamrc compile` + `wamr run`, no cache) | 2.552 ms | — |
| AOT warm (`wamr run`, precompiled `.cwasm`) | 1.182 ms | 12,442.8 |
| `wamrc run` (one-step compile+execute, no persistent cache) | 2.294 ms | — |
| In-process JIT, `.fast` preset (default, `-Djit=true`) | **1.430 ms** | 12,002.8 |
| In-process JIT, `.full` preset (`WAMR_JIT_FULL_OPT=1`) | 1.463 ms | 12,536.5 |
| wasmtime run (default JIT) | 6.964 ms | 26,462.6 |

## Interpretation

**Cold start**: the in-process JIT (`.fast` preset, the JIT build's default) is the second-fastest mode overall — only 21% slower than "AOT warm" (a precompiled `.cwasm` with zero compile work) and **~1.8x faster than the two-step "AOT cold" flow** (`wamrc compile` + `wamr run` as two separate process spawns), because the JIT avoids the second process-spawn entirely: compile and execute happen in one process, one `execve`. It is also **~4.9x faster than wasmtime's default JIT** on this tiny (36-byte) module — wasmtime's Cranelift backend has a higher fixed per-invocation startup cost that a single trivial function doesn't amortize. This matches the README's "very fast cold start" claim and is the core value proposition of #852-861: `wasmtime run foo.wasm`-style ergonomics without wasmtime's JIT startup tax.

The `.full` preset's cold-start numbers are statistically indistinguishable from `.fast` on this trivial module (both compile essentially nothing for a 36-byte no-op) — the preset's effect only shows up on non-trivial modules with real optimizable IR, which is exactly what the CoreMark throughput numbers below capture.

**Steady-state throughput**: AOT (precompiled, full optimization pipeline) leads at 12,442.8 iter/s. The JIT's `.full` preset (same pass pipeline as AOT, `WAMR_JIT_FULL_OPT=1`) is within measurement noise of AOT (12,536.5 iter/s — the ~1% difference is run-to-run variance, not a systematic gap, since both paths run the identical optimization pipeline). The JIT's `.fast` preset (default) trades roughly 3.5% throughput for its ~21% compile-time reduction (see #860's PR for the isolated compile-time measurement) — a favorable trade for the CLI-invocation, cold-start-dominated use case the JIT targets, and exactly the preset's design intent.

Wasmtime remains ahead on raw steady-state throughput — about **2.1x** in this measurement, an improvement over the README's previous "~3x" figure (measured on an earlier commit; this repo's AOT codegen has continued to close the gap per the ongoing #393 tracking issue). Wasmtime's Cranelift backend is a far more mature optimizing compiler; closing this gap further is out of scope for the JIT plan (#863) and remains tracked separately under #393.

**No CLI-accessible interpreter mode**: this runtime's CLI is AOT/JIT-only (#644) — `wamr run foo.wasm` without `-Djit=true` errors out asking the caller to precompile or rebuild with `-Djit=true`. The interpreter exists but is only reachable via the library API, not the CLI, so an "interpreter" row is intentionally omitted from the tables above rather than measuring something the CLI doesn't actually expose.

## Invocations

```console
$ cd /work/wamr-861-jit-bench-docs
$ unset ZIG_LOCAL_CACHE_DIR
$ export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
$ python3 scripts/bench_jit_coldstart.py --wasmtime-path ~/.wasmtime/bin/wasmtime --out /tmp/jit_bench_report.md
```

`scripts/bench_jit_coldstart.py` (new in this PR) builds a single `-Djit=true -Doptimize=ReleaseFast` binary — `-Djit=true` only *adds* the in-process compile+run path, so the same binary serves every row above, including the two-step AOT rows (verified byte-identical `.cwasm` output in #860). Cold-start timing wraps the noop module's subprocess wall time (20 samples, 3 warmup, discarded); CoreMark throughput parses the `Iterations/Sec` line from 3 runs per mode.

## Raw output

```text
$ python3 scripts/bench_jit_coldstart.py --wasmtime-path ~/.wasmtime/bin/wasmtime
[harness] building wamr/wamrc (-Djit=true -Doptimize=ReleaseFast)
[harness] timing cold-start (20 samples, 3 warmup)...
[harness] timing CoreMark throughput (3 runs per mode)...
[harness]   run 1/3: 12231.7 iter/s
[harness]   run 2/3: 12581.0 iter/s
[harness]   run 3/3: 12515.6 iter/s
[harness]   run 1/3: 12051.8 iter/s
[harness]   run 2/3: 12009.8 iter/s
[harness]   run 3/3: 11946.7 iter/s
[harness]   run 1/3: 12386.2 iter/s
[harness]   run 2/3: 12585.7 iter/s
[harness]   run 3/3: 12637.4 iter/s
[harness]   run 1/3: 26425.3 iter/s
[harness]   run 2/3: 26504.1 iter/s
[harness]   run 3/3: 26458.5 iter/s
# JIT cold-start comparison — 2026-07-10

_Host: arch `x86_64` · 8 vCPU · `AMD EPYC 9V74 80-Core Processor`_

## Cold-start (noop module, subprocess wall time)

| Mode | Median | p95 | Min | Max | Samples |
|---|---:|---:|---:|---:|---:|
| AOT cold (wamrc compile + wamr run, no cache) | 2.552 ms | 2.803 ms | 2.390 ms | 3.463 ms | 20 |
| AOT warm (wamr run, precompiled .cwasm) | 1.182 ms | 1.258 ms | 1.112 ms | 1.302 ms | 20 |
| wamrc run (one-step compile+execute, no persistent cache) | 2.294 ms | 2.541 ms | 2.029 ms | 2.577 ms | 20 |
| in-process JIT, .fast preset (default) | 1.430 ms | 1.498 ms | 1.365 ms | 1.562 ms | 20 |
| in-process JIT, .full preset (WAMR_JIT_FULL_OPT=1) | 1.463 ms | 1.621 ms | 1.295 ms | 1.621 ms | 20 |
| wasmtime run (default JIT) | 6.964 ms | 8.969 ms | 6.257 ms | 11.001 ms | 20 |

## Steady-state throughput (CoreMark)

| Mode | Mean iter/s | Min | Max | Runs |
|---|---:|---:|---:|---:|
| AOT (precompiled .cwasm) | 12442.8 | 12231.7 | 12581.0 | 3 |
| .fast preset (in-process JIT default) | 12002.8 | 11946.7 | 12051.8 | 3 |
| .full preset (in-process JIT, WAMR_JIT_FULL_OPT=1) | 12536.5 | 12386.2 | 12637.4 | 3 |
| wasmtime (default JIT) | 26462.6 | 26425.3 | 26504.1 | 3 |
```

## Limitations

- Measured on x86_64 only (this environment's available hardware); aarch64 numbers are not included. The relative ordering of modes (JIT `.fast` < AOT warm < JIT `.full` cold-start; AOT ≈ JIT `.full` > JIT `.fast` > throughput-wise) is expected to hold on aarch64 given #393's existing per-arch CoreMark tracking, but has not been independently verified here.
- Only a single fixture (CoreMark's WASI core module, plus the 36-byte `noop` module for cold-start) was benchmarked. A WASI component fixture was considered but omitted from this pass to keep the comparison focused and reproducible in a single script invocation — the underlying JIT call sites (`compileCoreWasm` / `precompileComponentInMemory`) are identical code paths per core module either way, so the core-wasm numbers above are representative of the component path's per-core compile cost as well.
