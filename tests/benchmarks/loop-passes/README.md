# Focused loop-pass benchmarks

These core-wasm fixtures isolate the two loop transforms discussed in #385:

- `iv_store.wasm` repeatedly stores through `base + i`, exercising
  induction-variable address strength reduction.
- `unroll4.wasm` repeatedly calls a four-trip scalar loop whose final value is
  live after the loop, exercising bounded full unrolling and live-out repair.

Each `_start` traps if its result is wrong, so a timed sample is accepted only
when the AOT process exits successfully. The tracked `.wasm` files are generated
from the adjacent `.wat` sources with the repository-pinned `wabt` dependency:

```sh
wabt text parse tests/benchmarks/loop-passes/iv_store.wat \
  -o tests/benchmarks/loop-passes/iv_store.wasm
wabt text parse tests/benchmarks/loop-passes/unroll4.wat \
  -o tests/benchmarks/loop-passes/unroll4.wasm
```

Run the reproducible ref comparison from the repository root:

```sh
python3 scripts/bench_loop_passes.py \
  --baseline origin/main \
  --target HEAD
```

The authoritative profile discards two warmups and records ten measured runs
per fixture/ref. CI can use `--profile ci` for zero warmups and three runs.
