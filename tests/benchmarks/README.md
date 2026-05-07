# Benchmarks

Benchmark inputs and harnesses for wamr.

## Layout

- [`coremark/`](coremark/) — EEMBC CoreMark via `zig build` (native, wasm,
  AOT). See [`coremark/README.md`](coremark/README.md) and the
  `coremark-aot` step exposed by the repo's root `build.zig`.
- `coremark/coremark_wasi*.wasm` — prebuilt CoreMark wasm modules used
  by the `coremark-aot` step in the root build to gate the Zig AOT
  backend on real CoreMark workloads.

