# SIMD benchmark harness

Issue #220 tracks native AArch64 AOT lowering for Wasm SIMD/v128 operations. Standard CoreMark is scalar, so it is not expected to move materially for this work. This harness provides SIMD-specific status and timing probes.

Run the embedded benchmark runner directly:

```sh
zig build simd-bench
./zig-out/bin/simd-bench-runner --iterations 10000
```

Compare two git refs:

```sh
scripts/bench_simd.py --baseline origin/main --target HEAD --runs 3 --iterations 10000
```

`scripts/bench_simd.py` creates temporary worktrees under `/work`, overlays the current harness into each worktree, and builds each ref in `ReleaseFast`. Overlaying the harness lets older refs report SIMD AOT as `unsupported` instead of failing because the harness did not exist.

The runner emits tab-separated rows for interpreter and AOT engines. It records status, scalar checksum result, compile time, run time, iteration count, and code size. SIMD AOT rows are expected to be `unsupported` until the relevant v128 IR/frontend/backend lowering slice lands.

The `scalar_i32_mem_add_4k_loop` and `simd_i32x4_mem_add_4k_loop` rows are the first throughput-oriented probes. Each exported call walks two 4 KiB input arrays in linear memory, writes an output array, and returns a scalar checksum from the last element. The SIMD row processes the same data with `v128.load`, `i32x4.add`, and `v128.store`, making it a better signal for vector memory-loop quality than one-instruction microbenchmarks.

The small `simd_i32x4_*_lane0` rows are coverage/status probes for individual opcode families. They intentionally return one scalar lane so interpreter, AOT, and optional Wasmtime rows can be compared before the runtime supports direct exported v128 values.

Wasmtime can be included as an external baseline:

```sh
zig build simd-bench -- --wasmtime --wasmtime-iterations 3 --iterations 100
```

Wasmtime rows currently use the CLI, so their timings include process startup and compilation. Treat them as correctness/status context rather than an apples-to-apples in-process throughput comparison.
