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

The runner emits tab-separated rows for interpreter and AOT engines. It records status, scalar checksum result, compile time, run time, iteration count, and code size. SIMD AOT rows are expected to be `unsupported` until the first v128 IR/frontend/backend lowering slice lands.
