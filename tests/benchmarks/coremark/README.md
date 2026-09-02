# CoreMark Benchmark

[CoreMark](https://www.eembc.org/coremark) is a simple processor benchmark. This directory
builds and runs CoreMark using Zig's cross-compiler for both native and wasm32-wasi targets.

## Prerequisites

1. Build `wamr` and `wamrc` from the repo root:
   ```
   zig build
   ```
2. Clone the CoreMark source at the pinned revision used for reproducible
   source rebuilds:
   ```
   cd tests/benchmarks/coremark
   git clone https://github.com/eembc/coremark.git
   git -C coremark checkout cfa9ab377835911f23d9b0831c7be302ed1f58de
   ```

The comparison harness does not rebuild from this checkout. It always uses the
tracked canonical fixture `coremark_wasi.wasm`
(`sha256:f4b7591296ead10264e0f101f355bdf848865c31329325594e66fbabefec235b`)
and verifies that checksum before running, eliminating upstream-source drift.

## Building

```
zig build                          # compile native + wasm
zig build aot                      # compile .wasm → .aot via wamrc
```

## Running

```
zig build run-native               # run native benchmark
zig build run-aot                  # run AOT benchmark via wamr
zig build run-interp               # run interpreter benchmark via wamr
zig build bench                    # run all three
```

## Reproducible cross-engine comparison

From the repository root:

```
python3 scripts/bench_coremark.py \
  --baseline origin/main \
  --target HEAD \
  --wasmtime-baseline auto \
  --wasmtime /path/to/current/wasmtime
```

The authoritative defaults are two discarded warmups and ten measured runs
per engine. Reports include architecture, CPU model, optimize mode, exact git
refs and Wasmtime versions, fixture checksum, every measured sample,
mean/median/range, and same-host WAMR/Wasmtime ratios. Every sample must contain
CoreMark's `Correct operation validated.` CRC result; command failures, CRC
errors, missing results, and ambiguous output fail the run.

`--wasmtime-baseline auto` downloads Wasmtime 44.0.1 (the latest Wasmtime
release line available before #393's 2026-05-07 reference measurement), the
pinned historical baseline for future comparisons, and verifies the official
release archive checksum. Pass
`--wasmtime /path/to/wasmtime` separately for a caller-selected or current
binary; the report keeps the two versions in distinct rows. Automatic download
supports Linux x86_64 and aarch64; other hosts can pass an explicit 44.0.1
binary.

PR CI deliberately uses `--profile ci` (zero warmups, three measured runs) to
keep the regression gate affordable. Use the default `--profile authoritative`
for publishable cross-engine numbers.

The AArch64 workflow also exposes a manual `profile` mode. It installs the
Ubuntu `linux-tools-$(uname -r)` package, verifies native `cycles:u` sampling,
precompiles the canonical fixture once with WAMR spill/codegen diagnostics,
and records load+execute-only self samples for WAMR and Wasmtime 44.0.1 on the
same native host. Wasmtime uses its documented v44
`--profile=jitdump` integration followed by `perf inject --jit`. Reports map
WAMR `local_func` indices and Wasmtime's full wasm function indices through
the fixture name section so the same functions are compared explicitly.

Run it from GitHub Actions with **CoreMark (aarch64) → Run workflow → mode:
profile**. `profile_ref` defaults to the exact merged-main commit measured by
run 33576430466 (`e32b7b7d2d12007eb66679a66f943b1e4ea6a393`); the workflow
builds that ref in an isolated worktree while using the merged profiling
tooling. It uploads compact JSON/Markdown reports, diagnostics, and compressed
raw perf/jitdump data when each input is at most 25 MiB.

## Options

```
zig build -Diterations=1000        # reduce iterations for quick testing
zig build -Dwamrc=/path/to/wamrc   # custom wamrc path
zig build -Dwamr=/path/to/wamr     # custom wamr path
```

## Notes

- The AOT and interpreter modes require `wamrc`/`wamr` to support all wasm opcodes
  used by CoreMark. As the Zig-based toolchain matures, these modes will become functional.
- Native mode works immediately and can be used as the baseline comparison.

## Profiling

To dig into where the AOT spends its time, run the in-process sampling
profiler (linux + aarch64/x86_64; SIGPROF + setitimer at 1 ms):

```
zig build coremark-profile
```

It prints a header line, a deterministic top-10 hot-function table, and the
disassembly of the top-3 hot functions. Function names are resolved from the
wasm `name` custom section; samples that land in import trampolines or runtime
helpers are aggregated under `<helper>` / `<host>` synthetic buckets so the
counts add up.

Sample (aarch64, `coremark_wasi_nofp.wasm`):

```
[coremark-profile] total_samples=3725 dropped=0 wall_ms=14904 interval_us=1000
[coremark-profile] top-10 hot functions:
    idx    samples     self_ms   self_pct  name
  -----  ---------  ----------  ---------  ----
     15       1512      1512.0     40.32%  core_bench_list
     22       1070      1070.0     28.53%  core_state_transition
     19        799       799.0     21.31%  matrix_test
     ...
```

The top-3 set is stable across runs; the tail of the top-10 has minor jitter
between runs (a couple of single-digit-sample functions exchange positions),
which is expected of any sampling profiler.

To bisect against the wasmtime gap, the equivalent invocation on the wasmtime
side is provided as a documented helper (not built into `zig build`):

```
bash scripts/profile_wasmtime_coremark.sh
```

This script runs `wasmtime compile` followed by `perf record -F 1000 -e
cycles:u` and then `perf report --stdio --no-children`, finishing with the
first 80 lines of `llvm-objdump -d` on the precompiled `.cwasm`. The `perf`
path requires `kernel.perf_event_paranoid <= 2`; bump it once with
`sudo sysctl -w kernel.perf_event_paranoid=1`. The wamr `coremark-profile`
step does **not** depend on `perf_event_paranoid` (it only uses POSIX SIGPROF).
