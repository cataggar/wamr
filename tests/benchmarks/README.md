# Benchmarks

Benchmark inputs and harnesses for wamr.

## Layout

- [`coremark/`](coremark/) — EEMBC CoreMark via `zig build` (native, wasm,
  AOT). See [`coremark/README.md`](coremark/README.md) and the
  `coremark-aot` step exposed by the repo's root `build.zig`.
- [`leaf-cancel-cost/`](leaf-cancel-cost/) — isolated #963 AOT function-entry
  cancel-poll cost evidence using a leaf-call-heavy worker pthread.
- `coremark/coremark_wasi*.wasm` — prebuilt CoreMark wasm modules used
  by the `coremark-aot` step in the root build to gate the Zig AOT
  backend on real CoreMark workloads.
- [`loop-passes/`](loop-passes/) — checksum-pinned focused AOT fixtures for
  induction-variable simplification and bounded loop unrolling, compared
  across refs by `scripts/bench_loop_passes.py`.

The SIMD comparison also has an experimental native implementation:
`zig build build-bench-simd` builds `zig-out/bin/bench-simd`. The existing
`scripts/bench_simd.py` remains authoritative while real-ref and platform
behavior are validated. SIGINT and SIGTERM cleanup has been exercised with
mock builds, and interrupted Git checkout cleanup with a small real repository,
on Linux. Execution on macOS remains unverified; check for leftover worktrees
after interruption.
