# CoreMark Benchmark

[CoreMark](https://www.eembc.org/coremark) is a simple processor benchmark. This directory
builds and runs CoreMark using Zig's cross-compiler for both native and wasm32-wasi targets.

## Prerequisites

1. Build `wamr` and `wamrc` from the repo root:
   ```
   zig build
   ```
2. Clone the CoreMark source:
   ```
   cd tests/benchmarks/coremark
   git clone https://github.com/eembc/coremark.git
   ```

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

