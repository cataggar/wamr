# Optimize mode comparison — 2026-05-29

Host: Azure VM worktree `/work/wamr-710bench`, commit `0c60142e`, branch `710-bench-releasesafe`. Commands were run with `ZIG_LOCAL_CACHE_DIR` unset and `ZIG_GLOBAL_CACHE_DIR=$PWD/.zig-global-cache`.

## Summary

| Workload | ReleaseFast | ReleaseSafe | Safe/Fast | Notes |
|---|---:|---:|---:|---|
| CoreMark AOT | 6728.3 iter/s | failed | — | `ReleaseSafe` `wamrc` trips `IR verifier: MultipleTerminators` before timing. |
| Cold-start `noop.cwasm` | 2.041 ms | 1.095 ms | ×0.54 | Median of 30 samples after 3 warmups. |
| SIMD interpreter rows | — | — | ×0.99 median | 201 ok interpreter rows; mean ×0.98, range ×0.79–×1.10. AOT SIMD rows are currently `unsupported`. |
| Keyvault codegen-cli repro | 0.481 s to trap | 3.879 s to panic | ×8.06 to failure | Both modes fail before completing; this is a time-to-failure comparison only. |

## Interpretation

The hot SIMD interpreter workload is effectively flat (median Safe/Fast ×0.99) and the tiny cold-start path is faster in this run under `ReleaseSafe`. CoreMark cannot currently produce a `ReleaseSafe` timing because safety checks in `wamrc` catch an IR verifier invariant before AOT output is written; that is useful safety signal rather than a runtime slowdown number. README therefore keeps `ReleaseSafe` as the primary source-build recommendation and points here for the measured gap and caveats.

## Invocations

```console
$ cd /work/wamr-710bench
$ unset ZIG_LOCAL_CACHE_DIR
$ export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
$ python3 scripts/bench_coremark.py --optimize both
$ python3 scripts/bench_coldstart.py --optimize both
$ python3 scripts/bench_simd.py --optimize both
```

For the keyvault repro, the requested `/tmp/kv-*` output directories were replaced with `/work/wamr-710bench/bench-keyvault/kv-*` so all file operations stayed on the NVMe worktree.

```console
$ cd /home/g/azure-sdk-for-zig
$ time WAMR=1 WAMR_BIN=/work/wamr-710bench/zig-out-releasefast/bin/wamr \
    codegen/cli/scripts/run.sh /home/g/azure-rest-api-specs/specification/keyvault/data-plane/Secrets \
    /work/wamr-710bench/bench-keyvault/kv-fast
$ time WAMR=1 WAMR_BIN=/work/wamr-710bench/zig-out-releasesafe/bin/wamr \
    codegen/cli/scripts/run.sh /home/g/azure-rest-api-specs/specification/keyvault/data-plane/Secrets \
    /work/wamr-710bench/bench-keyvault/kv-safe
```

## Raw output: CoreMark

```text
$ python3 scripts/bench_coremark.py --optimize both
[harness] building target-fast-0c60142e7252 (ReleaseFast)
[harness] AOT-compiling CoreMark in target-fast-0c60142e7252
[harness]   run 1/3: 7886.5 iter/s
[harness]   run 2/3: 4156.0 iter/s
[harness]   run 3/3: 8142.4 iter/s
[harness] building target-safe-0c60142e7252 (ReleaseSafe)
[harness] AOT-compiling CoreMark in target-safe-0c60142e7252

[harness] command failed (exit 1): zig build aot
[harness]   cwd: /work/bench-coremark-ie7ie366/target-safe-0c60142e7252/tests/benchmarks/coremark
[harness] --- stderr ---
aot
+- install generated to bin/coremark.cwasm
   +- run ../../../zig-out/bin/wamrc (coremark.cwasm) failure
Loaded ./.zig-cache/o/1966d6ac30aa02978422ece02db1a55f/coremark.wasm (1004529 bytes)
Parsed: 22 types, 84 functions, 2 exports
Lowered 84 functions to IR
IR verifier: MultipleTerminators after pass 'inlineSmallFunctions' func #1 block #1 inst #2 — terminator before end of block
Error optimizing IR: error.MultipleTerminators
error: process exited with error code 1
failed command: ../../../zig-out/bin/wamrc compile -o /work/bench-coremark-ie7ie366/target-safe-0c60142e7252/tests/benchmarks/coremark/.zig-cache/o/a872c877855e6d1a76ad43406994b45d/coremark.cwasm ./.zig-cache/o/1966d6ac30aa02978422ece02db1a55f/coremark.wasm

Build Summary: 1/4 steps succeeded (1 failed)
aot transitive failure
+- install generated to bin/coremark.cwasm transitive failure
   +- run ../../../zig-out/bin/wamrc (coremark.cwasm) failure

error: the following build command failed with exit code 1:
.zig-cache/o/f30ef4f7352ea887eaf3720bcddaa70e/build /home/g/.local/share/ghr/tools/ctaggart/zig/zig-x86_64-linux-0.16.0/zig /home/g/.local/share/ghr/tools/ctaggart/zig/zig-x86_64-linux-0.16.0/lib /work/bench-coremark-ie7ie366/target-safe-0c60142e7252/tests/benchmarks/coremark .zig-cache /work/bench-coremark-ie7ie366/target-safe-0c60142e7252/.zig-global-cache --seed 0x65de8d4f -Zfb0021a593286222 aot
[harness] ReleaseSafe failed before producing complete CoreMark timings: Command '['zig', 'build', 'aot']' returned non-zero exit status 1.
### CoreMark AOT optimize-mode comparison

| Ref | ReleaseFast mean iter/s | ReleaseSafe mean iter/s | Safe/Fast iter/s | ReleaseFast min..max | ReleaseSafe min..max | Runs |
|---|---:|---:|---:|---:|---:|---:|
| `HEAD` (target) | 6728.3 | failed | — | 4156.0..8142.4 | failed | 3/0 |

At least one optimize mode failed before producing a CoreMark timing; see raw harness output above.
```

## Raw output: cold-start

```text
$ python3 scripts/bench_coldstart.py --optimize both
[harness] building target-fast-0c60142e7252 (ReleaseFast)
[harness] AOT-compiling noop.cwasm via target wamrc
[harness] timing /work/bench-coldstart-vuqbfffp/target-fast-0c60142e7252/zig-out/bin/wamr run noop.cwasm (wamr-target/noop/cwasm) warmup=3 samples=30
[harness] building target-safe-0c60142e7252 (ReleaseSafe)
[harness] AOT-compiling noop.cwasm via target wamrc
[harness] timing /work/bench-coldstart-vuqbfffp/target-safe-0c60142e7252/zig-out/bin/wamr run noop.cwasm (wamr-target/noop/cwasm) warmup=3 samples=30
### Cold-start CLI optimize-mode comparison (median ms)

| Module | Variant | Engine | ReleaseFast median | ReleaseSafe median | Safe/Fast | ReleaseFast p95 | ReleaseSafe p95 |
|---|---|---|---:|---:|---:|---:|---:|
| `noop` | `cwasm` | target (`HEAD`) | 2.041 | 1.095 | ×0.54 | 7.081 | 1.281 |

Safe/Fast ratios compare ReleaseSafe median divided by ReleaseFast median for the same target/module/engine.
```

## Raw output: SIMD

```text
$ python3 scripts/bench_simd.py --optimize both
[harness] building target-fast-0c60142e7252 (ReleaseFast)
[harness] running target-fast-0c60142e7252 (1/3, iterations=10000)
[harness] running target-fast-0c60142e7252 (2/3, iterations=10000)
[harness] running target-fast-0c60142e7252 (3/3, iterations=10000)
[harness] building target-safe-0c60142e7252 (ReleaseSafe)
[harness] running target-safe-0c60142e7252 (1/3, iterations=10000)
[harness] running target-safe-0c60142e7252 (2/3, iterations=10000)
[harness] running target-safe-0c60142e7252 (3/3, iterations=10000)
### SIMD AOT optimize-mode comparison

| Case | Engine | Ref | Fast status | Safe status | Fast median run | Safe median run | Safe/Fast run | Fast median compile | Safe median compile | Safe/Fast compile |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `scalar_i32_add` | `aot` | `HEAD` (target) | ok | ok | 375.000 us | 389.000 us | ×1.04 | 222.000 us | 180.000 us | ×0.81 |
| `scalar_i32_add` | `interp` | `HEAD` (target) | ok | ok | 523.000 us | 463.000 us | ×0.89 | 0 ns | 0 ns | — |
| `scalar_i32_mem_add_4k_loop` | `aot` | `HEAD` (target) | ok | ok | 25.402 ms | 21.270 ms | ×0.84 | 10.631 ms | 10.482 ms | ×0.99 |
| `scalar_i32_mem_add_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 1677.514 ms | 1620.953 ms | ×0.97 | 0 ns | 0 ns | — |
| `simd_extadd_pairwise_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 213.000 us | 206.000 us | ×0.97 |
| `simd_extadd_pairwise_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 791.738 ms | 789.578 ms | ×1.00 | 0 ns | 0 ns | — |
| `simd_extend_lowhigh_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 233.000 us | 229.000 us | ×0.98 |
| `simd_extend_lowhigh_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 1170.619 ms | 1070.969 ms | ×0.91 | 0 ns | 0 ns | — |
| `simd_extmul_lowhigh_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 282.000 us | 265.000 us | ×0.94 |
| `simd_extmul_lowhigh_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 1570.482 ms | 1564.244 ms | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_abs_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_f32x4_abs_lane0` | `interp` | `HEAD` (target) | ok | ok | 452.000 us | 469.000 us | ×1.04 | 0 ns | 0 ns | — |
| `simd_f32x4_add_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_f32x4_add_lane0` | `interp` | `HEAD` (target) | ok | ok | 510.000 us | 509.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_ceil_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_f32x4_ceil_lane0` | `interp` | `HEAD` (target) | ok | ok | 440.000 us | 441.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_convert_i32x4_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_f32x4_convert_i32x4_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 449.000 us | 449.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_convert_i32x4_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 12.000 us | ×0.63 |
| `simd_f32x4_convert_i32x4_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 447.000 us | 451.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_f32x4_demote_f64x2_zero_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_f32x4_demote_f64x2_zero_lane0` | `interp` | `HEAD` (target) | ok | ok | 444.000 us | 441.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_f32x4_div_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 16.000 us | 12.000 us | ×0.75 |
| `simd_f32x4_div_lane0` | `interp` | `HEAD` (target) | ok | ok | 519.000 us | 518.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_eq_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 13.000 us | ×0.62 |
| `simd_f32x4_eq_lane0` | `interp` | `HEAD` (target) | ok | ok | 494.000 us | 486.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_f32x4_extract_lane3` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_f32x4_extract_lane3` | `interp` | `HEAD` (target) | ok | ok | 455.000 us | 453.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_floor_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_f32x4_floor_lane0` | `interp` | `HEAD` (target) | ok | ok | 443.000 us | 442.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_ge_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 13.000 us | ×0.62 |
| `simd_f32x4_ge_lane0` | `interp` | `HEAD` (target) | ok | ok | 498.000 us | 492.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_f32x4_gt_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 23.000 us | 13.000 us | ×0.57 |
| `simd_f32x4_gt_lane0` | `interp` | `HEAD` (target) | ok | ok | 524.000 us | 495.000 us | ×0.94 | 0 ns | 0 ns | — |
| `simd_f32x4_le_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 13.000 us | ×0.59 |
| `simd_f32x4_le_lane0` | `interp` | `HEAD` (target) | ok | ok | 493.000 us | 495.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_lt_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 14.000 us | ×0.70 |
| `simd_f32x4_lt_lane0` | `interp` | `HEAD` (target) | ok | ok | 512.000 us | 492.000 us | ×0.96 | 0 ns | 0 ns | — |
| `simd_f32x4_max_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_f32x4_max_lane0` | `interp` | `HEAD` (target) | ok | ok | 547.000 us | 526.000 us | ×0.96 | 0 ns | 0 ns | — |
| `simd_f32x4_min_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 12.000 us | ×0.57 |
| `simd_f32x4_min_lane0` | `interp` | `HEAD` (target) | ok | ok | 554.000 us | 527.000 us | ×0.95 | 0 ns | 0 ns | — |
| `simd_f32x4_mul_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_f32x4_mul_lane0` | `interp` | `HEAD` (target) | ok | ok | 514.000 us | 521.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_f32x4_ne_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 14.000 us | ×0.70 |
| `simd_f32x4_ne_lane0` | `interp` | `HEAD` (target) | ok | ok | 496.000 us | 512.000 us | ×1.03 | 0 ns | 0 ns | — |
| `simd_f32x4_nearest_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 12.000 us | ×0.57 |
| `simd_f32x4_nearest_lane0` | `interp` | `HEAD` (target) | ok | ok | 469.000 us | 468.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_neg_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_f32x4_neg_lane0` | `interp` | `HEAD` (target) | ok | ok | 566.000 us | 448.000 us | ×0.79 | 0 ns | 0 ns | — |
| `simd_f32x4_pmax_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 12.000 us | ×0.60 |
| `simd_f32x4_pmax_lane0` | `interp` | `HEAD` (target) | ok | ok | 511.000 us | 501.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_f32x4_pmin_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 12.000 us | ×0.63 |
| `simd_f32x4_pmin_lane0` | `interp` | `HEAD` (target) | ok | ok | 595.000 us | 503.000 us | ×0.85 | 0 ns | 0 ns | — |
| `simd_f32x4_replace_lane2` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 15.000 us | ×0.71 |
| `simd_f32x4_replace_lane2` | `interp` | `HEAD` (target) | ok | ok | 594.000 us | 582.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_f32x4_splat_lane3` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 17.000 us | ×0.77 |
| `simd_f32x4_splat_lane3` | `interp` | `HEAD` (target) | ok | ok | 537.000 us | 550.000 us | ×1.02 | 0 ns | 0 ns | — |
| `simd_f32x4_sqrt_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 18.000 us | 13.000 us | ×0.72 |
| `simd_f32x4_sqrt_lane0` | `interp` | `HEAD` (target) | ok | ok | 481.000 us | 460.000 us | ×0.96 | 0 ns | 0 ns | — |
| `simd_f32x4_sub_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_f32x4_sub_lane0` | `interp` | `HEAD` (target) | ok | ok | 508.000 us | 507.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f32x4_trunc_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 13.000 us | ×0.62 |
| `simd_f32x4_trunc_lane0` | `interp` | `HEAD` (target) | ok | ok | 441.000 us | 441.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f64x2_abs_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 24.000 us | 15.000 us | ×0.62 |
| `simd_f64x2_abs_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 563.000 us | 545.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_f64x2_ceil_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 14.000 us | ×0.56 |
| `simd_f64x2_ceil_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 527.000 us | 525.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f64x2_convert_low_i32x4_s_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 28.000 us | 15.000 us | ×0.54 |
| `simd_f64x2_convert_low_i32x4_s_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 618.000 us | 542.000 us | ×0.88 | 0 ns | 0 ns | — |
| `simd_f64x2_convert_low_i32x4_u_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 15.000 us | ×0.71 |
| `simd_f64x2_convert_low_i32x4_u_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 565.000 us | 548.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_f64x2_extract_lane1_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 17.000 us | ×0.77 |
| `simd_f64x2_extract_lane1_high` | `interp` | `HEAD` (target) | ok | ok | 497.000 us | 497.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f64x2_floor_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 15.000 us | ×0.71 |
| `simd_f64x2_floor_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 534.000 us | 540.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_f64x2_nearest_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 14.000 us | ×0.64 |
| `simd_f64x2_nearest_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 605.000 us | 555.000 us | ×0.92 | 0 ns | 0 ns | — |
| `simd_f64x2_neg_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 14.000 us | ×0.64 |
| `simd_f64x2_neg_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 545.000 us | 541.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_f64x2_promote_low_f32x4_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 28.000 us | 13.000 us | ×0.46 |
| `simd_f64x2_promote_low_f32x4_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 561.000 us | 617.000 us | ×1.10 | 0 ns | 0 ns | — |
| `simd_f64x2_replace_lane1_low` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 15.000 us | ×0.75 |
| `simd_f64x2_replace_lane1_low` | `interp` | `HEAD` (target) | ok | ok | 509.000 us | 509.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f64x2_splat_lane1_low` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 15.000 us | ×0.68 |
| `simd_f64x2_splat_lane1_low` | `interp` | `HEAD` (target) | ok | ok | 490.000 us | 498.000 us | ×1.02 | 0 ns | 0 ns | — |
| `simd_f64x2_sqrt_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 13.000 us | ×0.62 |
| `simd_f64x2_sqrt_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 545.000 us | 545.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_f64x2_trunc_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 12.000 us | ×0.60 |
| `simd_f64x2_trunc_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 526.000 us | 529.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i16x8_abs_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 24.000 us | 19.000 us | ×0.79 |
| `simd_i16x8_abs_lane0` | `interp` | `HEAD` (target) | ok | ok | 458.000 us | 451.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i16x8_add_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 14.000 us | ×0.70 |
| `simd_i16x8_add_lane0` | `interp` | `HEAD` (target) | ok | ok | 509.000 us | 504.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_add_sat_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 15.000 us | 14.000 us | ×0.93 |
| `simd_i16x8_add_sat_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 571.000 us | 515.000 us | ×0.90 | 0 ns | 0 ns | — |
| `simd_i16x8_add_sat_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 12.000 us | 14.000 us | ×1.17 |
| `simd_i16x8_add_sat_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 513.000 us | 507.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_all_true_mixed` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 35.000 us | 28.000 us | ×0.80 |
| `simd_i16x8_all_true_mixed` | `interp` | `HEAD` (target) | ok | ok | 770.000 us | 772.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i16x8_arith_extra_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 269.000 us | 259.000 us | ×0.96 |
| `simd_i16x8_arith_extra_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 1366.879 ms | 1292.969 ms | ×0.95 | 0 ns | 0 ns | — |
| `simd_i16x8_avgr_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 12.000 us | ×0.55 |
| `simd_i16x8_avgr_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 520.000 us | 536.000 us | ×1.03 | 0 ns | 0 ns | — |
| `simd_i16x8_bitmask_mixed` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 17.000 us | ×0.77 |
| `simd_i16x8_bitmask_mixed` | `interp` | `HEAD` (target) | ok | ok | 414.000 us | 407.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i16x8_extadd_pairwise_i8x16_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_i16x8_extadd_pairwise_i8x16_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 450.000 us | 443.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i16x8_extadd_pairwise_i8x16_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 18.000 us | 13.000 us | ×0.72 |
| `simd_i16x8_extadd_pairwise_i8x16_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 439.000 us | 440.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i16x8_extend_high_i8x16_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 17.000 us | 13.000 us | ×0.76 |
| `simd_i16x8_extend_high_i8x16_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 449.000 us | 448.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i16x8_extend_high_i8x16_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_i16x8_extend_high_i8x16_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 446.000 us | 443.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_extend_low_i8x16_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_i16x8_extend_low_i8x16_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 450.000 us | 453.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i16x8_extend_low_i8x16_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_i16x8_extend_low_i8x16_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 450.000 us | 453.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i16x8_extmul_high_i8x16_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 14.000 us | ×0.70 |
| `simd_i16x8_extmul_high_i8x16_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 506.000 us | 501.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_extmul_high_i8x16_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 12.000 us | ×0.63 |
| `simd_i16x8_extmul_high_i8x16_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 600.000 us | 499.000 us | ×0.83 | 0 ns | 0 ns | — |
| `simd_i16x8_extmul_low_i8x16_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 15.000 us | ×0.68 |
| `simd_i16x8_extmul_low_i8x16_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 511.000 us | 506.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_extmul_low_i8x16_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 14.000 us | ×0.56 |
| `simd_i16x8_extmul_low_i8x16_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 534.000 us | 503.000 us | ×0.94 | 0 ns | 0 ns | — |
| `simd_i16x8_gt_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_i16x8_gt_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 509.000 us | 494.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_i16x8_gt_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 12.000 us | ×0.60 |
| `simd_i16x8_gt_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 597.000 us | 493.000 us | ×0.83 | 0 ns | 0 ns | — |
| `simd_i16x8_max_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 13.000 us | 13.000 us | ×1.00 |
| `simd_i16x8_max_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 517.000 us | 512.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_max_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 23.000 us | 13.000 us | ×0.57 |
| `simd_i16x8_max_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 587.000 us | 506.000 us | ×0.86 | 0 ns | 0 ns | — |
| `simd_i16x8_mem_add_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 185.000 us | 172.000 us | ×0.93 |
| `simd_i16x8_mem_add_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 407.844 ms | 405.355 ms | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_min_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 17.000 us | 13.000 us | ×0.76 |
| `simd_i16x8_min_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 522.000 us | 506.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_i16x8_min_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 14.000 us | 13.000 us | ×0.93 |
| `simd_i16x8_min_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 509.000 us | 503.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_mul_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 21.000 us | ×0.84 |
| `simd_i16x8_mul_lane0` | `interp` | `HEAD` (target) | ok | ok | 509.000 us | 524.000 us | ×1.03 | 0 ns | 0 ns | — |
| `simd_i16x8_narrow_i32x4_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_i16x8_narrow_i32x4_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 523.000 us | 516.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_narrow_i32x4_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_i16x8_narrow_i32x4_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 521.000 us | 515.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_neg_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 13.000 us | ×0.62 |
| `simd_i16x8_neg_lane0` | `interp` | `HEAD` (target) | ok | ok | 472.000 us | 454.000 us | ×0.96 | 0 ns | 0 ns | — |
| `simd_i16x8_q15mulr_sat_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 14.000 us | ×0.70 |
| `simd_i16x8_q15mulr_sat_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 538.000 us | 521.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_i16x8_replace_lane5` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 14.000 us | ×0.70 |
| `simd_i16x8_replace_lane5` | `interp` | `HEAD` (target) | ok | ok | 591.000 us | 596.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i16x8_shift_mix_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 215.000 us | 201.000 us | ×0.93 |
| `simd_i16x8_shift_mix_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 622.427 ms | 624.666 ms | ×1.00 | 0 ns | 0 ns | — |
| `simd_i16x8_shl_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 82.000 us | 70.000 us | ×0.85 |
| `simd_i16x8_shl_lane0` | `interp` | `HEAD` (target) | ok | ok | 2.023 ms | 1.950 ms | ×0.96 | 0 ns | 0 ns | — |
| `simd_i16x8_shr_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 53.000 us | 44.000 us | ×0.83 |
| `simd_i16x8_shr_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 1.959 ms | 1.936 ms | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_shr_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 51.000 us | 40.000 us | ×0.78 |
| `simd_i16x8_shr_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 1.958 ms | 1.955 ms | ×1.00 | 0 ns | 0 ns | — |
| `simd_i16x8_splat_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 14.000 us | ×0.67 |
| `simd_i16x8_splat_lane0` | `interp` | `HEAD` (target) | ok | ok | 478.000 us | 475.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i16x8_sub_sat_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 15.000 us | 14.000 us | ×0.93 |
| `simd_i16x8_sub_sat_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 580.000 us | 506.000 us | ×0.87 | 0 ns | 0 ns | — |
| `simd_i16x8_sub_sat_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 13.000 us | 13.000 us | ×1.00 |
| `simd_i16x8_sub_sat_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 515.000 us | 507.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i32x4_abs_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 19.000 us | ×0.95 |
| `simd_i32x4_abs_lane0` | `interp` | `HEAD` (target) | ok | ok | 453.000 us | 450.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_add_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 34.000 us | 30.000 us | ×0.88 |
| `simd_i32x4_add_lane0` | `interp` | `HEAD` (target) | ok | ok | 513.000 us | 511.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_all_true_mixed` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 31.000 us | 23.000 us | ×0.74 |
| `simd_i32x4_all_true_mixed` | `interp` | `HEAD` (target) | ok | ok | 763.000 us | 761.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_bitmask_mixed` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 16.000 us | 14.000 us | ×0.88 |
| `simd_i32x4_bitmask_mixed` | `interp` | `HEAD` (target) | ok | ok | 376.000 us | 377.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_dot_i16x8_s_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 183.000 us | 174.000 us | ×0.95 |
| `simd_i32x4_dot_i16x8_s_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 402.779 ms | 405.068 ms | ×1.01 | 0 ns | 0 ns | — |
| `simd_i32x4_dot_i16x8_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 18.000 us | 19.000 us | ×1.06 |
| `simd_i32x4_dot_i16x8_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 509.000 us | 506.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_eq_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 15.000 us | ×0.79 |
| `simd_i32x4_eq_lane0` | `interp` | `HEAD` (target) | ok | ok | 513.000 us | 494.000 us | ×0.96 | 0 ns | 0 ns | — |
| `simd_i32x4_extadd_pairwise_i16x8_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 18.000 us | 15.000 us | ×0.83 |
| `simd_i32x4_extadd_pairwise_i16x8_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 442.000 us | 439.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_extadd_pairwise_i16x8_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 17.000 us | 17.000 us | ×1.00 |
| `simd_i32x4_extadd_pairwise_i16x8_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 434.000 us | 436.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_extend_high_i16x8_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 15.000 us | 15.000 us | ×1.00 |
| `simd_i32x4_extend_high_i16x8_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 446.000 us | 444.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_extend_high_i16x8_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 18.000 us | ×0.82 |
| `simd_i32x4_extend_high_i16x8_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 443.000 us | 445.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_extend_low_i16x8_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 18.000 us | ×0.95 |
| `simd_i32x4_extend_low_i16x8_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 448.000 us | 454.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i32x4_extend_low_i16x8_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 24.000 us | 17.000 us | ×0.71 |
| `simd_i32x4_extend_low_i16x8_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 498.000 us | 447.000 us | ×0.90 | 0 ns | 0 ns | — |
| `simd_i32x4_extmul_high_i16x8_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 27.000 us | 13.000 us | ×0.48 |
| `simd_i32x4_extmul_high_i16x8_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 499.000 us | 516.000 us | ×1.03 | 0 ns | 0 ns | — |
| `simd_i32x4_extmul_high_i16x8_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 24.000 us | 13.000 us | ×0.54 |
| `simd_i32x4_extmul_high_i16x8_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 511.000 us | 491.000 us | ×0.96 | 0 ns | 0 ns | — |
| `simd_i32x4_extmul_low_i16x8_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 13.000 us | ×0.62 |
| `simd_i32x4_extmul_low_i16x8_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 507.000 us | 505.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_extmul_low_i16x8_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 14.000 us | ×0.64 |
| `simd_i32x4_extmul_low_i16x8_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 516.000 us | 504.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i32x4_gt_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 15.000 us | ×0.68 |
| `simd_i32x4_gt_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 494.000 us | 491.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_gt_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 31.000 us | 24.000 us | ×0.77 |
| `simd_i32x4_gt_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 497.000 us | 491.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_max_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 16.000 us | ×0.76 |
| `simd_i32x4_max_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 513.000 us | 503.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i32x4_max_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 17.000 us | ×0.85 |
| `simd_i32x4_max_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 510.000 us | 504.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_mem_add_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 189.000 us | 169.000 us | ×0.89 |
| `simd_i32x4_mem_add_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 414.669 ms | 406.118 ms | ×0.98 | 0 ns | 0 ns | — |
| `simd_i32x4_mem_sum8_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 223.000 us | 209.000 us | ×0.94 |
| `simd_i32x4_mem_sum8_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 247.862 ms | 231.709 ms | ×0.93 | 0 ns | 0 ns | — |
| `simd_i32x4_min_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 17.000 us | 18.000 us | ×1.06 |
| `simd_i32x4_min_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 506.000 us | 501.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_min_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 18.000 us | 19.000 us | ×1.06 |
| `simd_i32x4_min_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 507.000 us | 534.000 us | ×1.05 | 0 ns | 0 ns | — |
| `simd_i32x4_minmax_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 217.000 us | 203.000 us | ×0.94 |
| `simd_i32x4_minmax_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 801.287 ms | 789.332 ms | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_mul_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 28.000 us | 25.000 us | ×0.89 |
| `simd_i32x4_mul_lane0` | `interp` | `HEAD` (target) | ok | ok | 504.000 us | 504.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_ne_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 33.000 us | 25.000 us | ×0.76 |
| `simd_i32x4_ne_lane0` | `interp` | `HEAD` (target) | ok | ok | 496.000 us | 493.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_neg_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 17.000 us | 18.000 us | ×1.06 |
| `simd_i32x4_neg_lane0` | `interp` | `HEAD` (target) | ok | ok | 464.000 us | 451.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_i32x4_replace_lane2` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 49.000 us | 40.000 us | ×0.82 |
| `simd_i32x4_replace_lane2` | `interp` | `HEAD` (target) | ok | ok | 582.000 us | 586.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i32x4_shift_mix_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 200.000 us | 174.000 us | ×0.87 |
| `simd_i32x4_shift_mix_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 511.960 ms | 499.262 ms | ×0.98 | 0 ns | 0 ns | — |
| `simd_i32x4_shl_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 80.000 us | 71.000 us | ×0.89 |
| `simd_i32x4_shl_lane0` | `interp` | `HEAD` (target) | ok | ok | 1.956 ms | 1.942 ms | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_shr_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 55.000 us | 44.000 us | ×0.80 |
| `simd_i32x4_shr_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 1.951 ms | 1.922 ms | ×0.99 | 0 ns | 0 ns | — |
| `simd_i32x4_shr_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 56.000 us | 42.000 us | ×0.75 |
| `simd_i32x4_shr_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 2.009 ms | 1.955 ms | ×0.97 | 0 ns | 0 ns | — |
| `simd_i32x4_splat_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 42.000 us | 33.000 us | ×0.79 |
| `simd_i32x4_splat_lane0` | `interp` | `HEAD` (target) | ok | ok | 476.000 us | 480.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i32x4_trunc_sat_f32x4_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 14.000 us | ×0.74 |
| `simd_i32x4_trunc_sat_f32x4_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 479.000 us | 477.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_trunc_sat_f32x4_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_i32x4_trunc_sat_f32x4_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 479.000 us | 480.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_trunc_sat_f64x2_s_zero_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_i32x4_trunc_sat_f64x2_s_zero_lane0` | `interp` | `HEAD` (target) | ok | ok | 461.000 us | 463.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i32x4_trunc_sat_f64x2_u_zero_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 18.000 us | 13.000 us | ×0.72 |
| `simd_i32x4_trunc_sat_f64x2_u_zero_lane0` | `interp` | `HEAD` (target) | ok | ok | 458.000 us | 461.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i64x2_abs_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 12.000 us | ×0.60 |
| `simd_i64x2_abs_lane0` | `interp` | `HEAD` (target) | ok | ok | 462.000 us | 461.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i64x2_add_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 23.000 us | 17.000 us | ×0.74 |
| `simd_i64x2_add_lane0` | `interp` | `HEAD` (target) | ok | ok | 522.000 us | 518.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i64x2_all_true_mixed` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 30.000 us | 26.000 us | ×0.87 |
| `simd_i64x2_all_true_mixed` | `interp` | `HEAD` (target) | ok | ok | 758.000 us | 762.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i64x2_bitmask_mixed` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 17.000 us | 17.000 us | ×1.00 |
| `simd_i64x2_bitmask_mixed` | `interp` | `HEAD` (target) | ok | ok | 368.000 us | 398.000 us | ×1.08 | 0 ns | 0 ns | — |
| `simd_i64x2_eq_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 15.000 us | ×0.71 |
| `simd_i64x2_eq_lane0` | `interp` | `HEAD` (target) | ok | ok | 521.000 us | 518.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i64x2_extend_high_i32x4_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 20.000 us | ×0.80 |
| `simd_i64x2_extend_high_i32x4_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 761.000 us | 762.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i64x2_extend_high_i32x4_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 24.000 us | 20.000 us | ×0.83 |
| `simd_i64x2_extend_high_i32x4_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 765.000 us | 789.000 us | ×1.03 | 0 ns | 0 ns | — |
| `simd_i64x2_extend_low_i32x4_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 30.000 us | 23.000 us | ×0.77 |
| `simd_i64x2_extend_low_i32x4_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 774.000 us | 778.000 us | ×1.01 | 0 ns | 0 ns | — |
| `simd_i64x2_extend_low_i32x4_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 19.000 us | ×0.76 |
| `simd_i64x2_extend_low_i32x4_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 780.000 us | 776.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i64x2_extmul_high_i32x4_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 28.000 us | 21.000 us | ×0.75 |
| `simd_i64x2_extmul_high_i32x4_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 871.000 us | 870.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i64x2_extmul_high_i32x4_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 29.000 us | 20.000 us | ×0.69 |
| `simd_i64x2_extmul_high_i32x4_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 878.000 us | 861.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i64x2_extmul_low_i32x4_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 32.000 us | 25.000 us | ×0.78 |
| `simd_i64x2_extmul_low_i32x4_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 887.000 us | 890.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i64x2_extmul_low_i32x4_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 30.000 us | 20.000 us | ×0.67 |
| `simd_i64x2_extmul_low_i32x4_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 907.000 us | 890.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i64x2_gt_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 12.000 us | ×0.63 |
| `simd_i64x2_gt_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 555.000 us | 517.000 us | ×0.93 | 0 ns | 0 ns | — |
| `simd_i64x2_mem_add_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 193.000 us | 178.000 us | ×0.92 |
| `simd_i64x2_mem_add_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 413.972 ms | 405.203 ms | ×0.98 | 0 ns | 0 ns | — |
| `simd_i64x2_neg_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 13.000 us | ×0.59 |
| `simd_i64x2_neg_lane0` | `interp` | `HEAD` (target) | ok | ok | 575.000 us | 566.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i64x2_replace_lane1` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 30.000 us | 12.000 us | ×0.40 |
| `simd_i64x2_replace_lane1` | `interp` | `HEAD` (target) | ok | ok | 525.000 us | 518.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i64x2_shift_mix_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 236.000 us | 218.000 us | ×0.92 |
| `simd_i64x2_shift_mix_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 734.058 ms | 736.001 ms | ×1.00 | 0 ns | 0 ns | — |
| `simd_i64x2_shl_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 84.000 us | 70.000 us | ×0.83 |
| `simd_i64x2_shl_lane0` | `interp` | `HEAD` (target) | ok | ok | 2.040 ms | 2.032 ms | ×1.00 | 0 ns | 0 ns | — |
| `simd_i64x2_shr_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 72.000 us | 65.000 us | ×0.90 |
| `simd_i64x2_shr_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 2.057 ms | 2.007 ms | ×0.98 | 0 ns | 0 ns | — |
| `simd_i64x2_shr_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 69.000 us | 49.000 us | ×0.71 |
| `simd_i64x2_shr_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 2.159 ms | 2.012 ms | ×0.93 | 0 ns | 0 ns | — |
| `simd_i64x2_splat_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 15.000 us | ×0.68 |
| `simd_i64x2_splat_lane0` | `interp` | `HEAD` (target) | ok | ok | 518.000 us | 503.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_i64x2_sub_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 14.000 us | ×0.67 |
| `simd_i64x2_sub_lane0` | `interp` | `HEAD` (target) | ok | ok | 525.000 us | 523.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i8x16_abs_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 13.000 us | ×0.62 |
| `simd_i8x16_abs_lane0` | `interp` | `HEAD` (target) | ok | ok | 443.000 us | 442.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i8x16_add_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 14.000 us | ×0.67 |
| `simd_i8x16_add_lane0` | `interp` | `HEAD` (target) | ok | ok | 500.000 us | 496.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i8x16_add_sat_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 24.000 us | 12.000 us | ×0.50 |
| `simd_i8x16_add_sat_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 547.000 us | 490.000 us | ×0.90 | 0 ns | 0 ns | — |
| `simd_i8x16_add_sat_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 11.000 us | ×0.55 |
| `simd_i8x16_add_sat_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 496.000 us | 513.000 us | ×1.03 | 0 ns | 0 ns | — |
| `simd_i8x16_all_true_mixed` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 39.000 us | 31.000 us | ×0.79 |
| `simd_i8x16_all_true_mixed` | `interp` | `HEAD` (target) | ok | ok | 743.000 us | 741.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i8x16_arith_extra_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 234.000 us | 224.000 us | ×0.96 |
| `simd_i8x16_arith_extra_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 1019.950 ms | 1002.344 ms | ×0.98 | 0 ns | 0 ns | — |
| `simd_i8x16_avgr_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 12.000 us | ×0.63 |
| `simd_i8x16_avgr_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 506.000 us | 522.000 us | ×1.03 | 0 ns | 0 ns | — |
| `simd_i8x16_bitmask_mixed` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 27.000 us | 18.000 us | ×0.67 |
| `simd_i8x16_bitmask_mixed` | `interp` | `HEAD` (target) | ok | ok | 501.000 us | 441.000 us | ×0.88 | 0 ns | 0 ns | — |
| `simd_i8x16_eq_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 11.000 us | ×0.55 |
| `simd_i8x16_eq_lane0` | `interp` | `HEAD` (target) | ok | ok | 507.000 us | 494.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_i8x16_gt_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_i8x16_gt_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 499.000 us | 496.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i8x16_gt_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 13.000 us | ×0.68 |
| `simd_i8x16_gt_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 496.000 us | 496.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i8x16_max_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 10.000 us | ×0.53 |
| `simd_i8x16_max_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 498.000 us | 497.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i8x16_max_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 11.000 us | ×0.52 |
| `simd_i8x16_max_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 519.000 us | 491.000 us | ×0.95 | 0 ns | 0 ns | — |
| `simd_i8x16_mem_add_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 187.000 us | 172.000 us | ×0.92 |
| `simd_i8x16_mem_add_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 461.429 ms | 405.723 ms | ×0.88 | 0 ns | 0 ns | — |
| `simd_i8x16_min_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 11.000 us | ×0.52 |
| `simd_i8x16_min_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 501.000 us | 492.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i8x16_min_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 19.000 us | 11.000 us | ×0.58 |
| `simd_i8x16_min_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 501.000 us | 490.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i8x16_narrow_i16x8_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 26.000 us | 14.000 us | ×0.54 |
| `simd_i8x16_narrow_i16x8_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 546.000 us | 501.000 us | ×0.92 | 0 ns | 0 ns | — |
| `simd_i8x16_narrow_i16x8_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 13.000 us | ×0.65 |
| `simd_i8x16_narrow_i16x8_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 503.000 us | 502.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_i8x16_neg_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 11.000 us | ×0.50 |
| `simd_i8x16_neg_lane0` | `interp` | `HEAD` (target) | ok | ok | 462.000 us | 440.000 us | ×0.95 | 0 ns | 0 ns | — |
| `simd_i8x16_replace_lane13` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 14.000 us | ×0.64 |
| `simd_i8x16_replace_lane13` | `interp` | `HEAD` (target) | ok | ok | 592.000 us | 586.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_i8x16_shift_mix_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 229.000 us | 218.000 us | ×0.95 |
| `simd_i8x16_shift_mix_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 759.213 ms | 727.016 ms | ×0.96 | 0 ns | 0 ns | — |
| `simd_i8x16_shl_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 57.000 us | 49.000 us | ×0.86 |
| `simd_i8x16_shl_lane0` | `interp` | `HEAD` (target) | ok | ok | 1.995 ms | 1.955 ms | ×0.98 | 0 ns | 0 ns | — |
| `simd_i8x16_shr_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 52.000 us | 47.000 us | ×0.90 |
| `simd_i8x16_shr_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 1.958 ms | 1.959 ms | ×1.00 | 0 ns | 0 ns | — |
| `simd_i8x16_shr_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 56.000 us | 46.000 us | ×0.82 |
| `simd_i8x16_shr_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 2.124 ms | 1.970 ms | ×0.93 | 0 ns | 0 ns | — |
| `simd_i8x16_shuffle_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 19.000 us | ×0.76 |
| `simd_i8x16_shuffle_lane0` | `interp` | `HEAD` (target) | ok | ok | 851.000 us | 813.000 us | ×0.96 | 0 ns | 0 ns | — |
| `simd_i8x16_splat_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 15.000 us | ×0.75 |
| `simd_i8x16_splat_lane0` | `interp` | `HEAD` (target) | ok | ok | 474.000 us | 488.000 us | ×1.03 | 0 ns | 0 ns | — |
| `simd_i8x16_sub_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 14.000 us | ×0.67 |
| `simd_i8x16_sub_lane0` | `interp` | `HEAD` (target) | ok | ok | 499.000 us | 511.000 us | ×1.02 | 0 ns | 0 ns | — |
| `simd_i8x16_sub_sat_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 11.000 us | ×0.55 |
| `simd_i8x16_sub_sat_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 535.000 us | 497.000 us | ×0.93 | 0 ns | 0 ns | — |
| `simd_i8x16_sub_sat_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 20.000 us | 11.000 us | ×0.55 |
| `simd_i8x16_sub_sat_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 501.000 us | 489.000 us | ×0.98 | 0 ns | 0 ns | — |
| `simd_i8x16_swizzle_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 17.000 us | ×0.68 |
| `simd_i8x16_swizzle_lane0` | `interp` | `HEAD` (target) | ok | ok | 814.000 us | 777.000 us | ×0.95 | 0 ns | 0 ns | — |
| `simd_int_absneg_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 214.000 us | 198.000 us | ×0.93 |
| `simd_int_absneg_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 629.485 ms | 584.635 ms | ×0.93 | 0 ns | 0 ns | — |
| `simd_narrow_4k_loop` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 241.000 us | 222.000 us | ×0.92 |
| `simd_narrow_4k_loop` | `interp` | `HEAD` (target) | ok | ok | 1116.749 ms | 1097.460 ms | ×0.98 | 0 ns | 0 ns | — |
| `simd_v128_any_true_mixed` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 29.000 us | 18.000 us | ×0.62 |
| `simd_v128_any_true_mixed` | `interp` | `HEAD` (target) | ok | ok | 464.000 us | 446.000 us | ×0.96 | 0 ns | 0 ns | — |
| `simd_v128_bitselect_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 18.000 us | 17.000 us | ×0.94 |
| `simd_v128_bitselect_lane0` | `interp` | `HEAD` (target) | ok | ok | 803.000 us | 803.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_v128_load16_lane2` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 12.000 us | ×0.55 |
| `simd_v128_load16_lane2` | `interp` | `HEAD` (target) | ok | ok | 617.000 us | 632.000 us | ×1.02 | 0 ns | 0 ns | — |
| `simd_v128_load16_splat_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 14.000 us | ×0.67 |
| `simd_v128_load16_splat_lane0` | `interp` | `HEAD` (target) | ok | ok | 531.000 us | 544.000 us | ×1.02 | 0 ns | 0 ns | — |
| `simd_v128_load16x4_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 14.000 us | ×0.64 |
| `simd_v128_load16x4_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 557.000 us | 527.000 us | ×0.95 | 0 ns | 0 ns | — |
| `simd_v128_load16x4_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 13.000 us | ×0.62 |
| `simd_v128_load16x4_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 543.000 us | 525.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_v128_load32_lane1` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 23.000 us | 13.000 us | ×0.57 |
| `simd_v128_load32_lane1` | `interp` | `HEAD` (target) | ok | ok | 622.000 us | 616.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_v128_load32_splat_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 24.000 us | 12.000 us | ×0.50 |
| `simd_v128_load32_splat_lane0` | `interp` | `HEAD` (target) | ok | ok | 541.000 us | 523.000 us | ×0.97 | 0 ns | 0 ns | — |
| `simd_v128_load32_zero_lanes` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 35.000 us | 30.000 us | ×0.86 |
| `simd_v128_load32_zero_lanes` | `interp` | `HEAD` (target) | ok | ok | 1.246 ms | 1.247 ms | ×1.00 | 0 ns | 0 ns | — |
| `simd_v128_load32x2_s_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 16.000 us | ×0.64 |
| `simd_v128_load32x2_s_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 600.000 us | 578.000 us | ×0.96 | 0 ns | 0 ns | — |
| `simd_v128_load32x2_u_lane0_high` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 23.000 us | 15.000 us | ×0.65 |
| `simd_v128_load32x2_u_lane0_high` | `interp` | `HEAD` (target) | ok | ok | 578.000 us | 574.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_v128_load64_lane1` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 23.000 us | 14.000 us | ×0.61 |
| `simd_v128_load64_lane1` | `interp` | `HEAD` (target) | ok | ok | 603.000 us | 614.000 us | ×1.02 | 0 ns | 0 ns | — |
| `simd_v128_load64_splat_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 12.000 us | ×0.55 |
| `simd_v128_load64_splat_lane0` | `interp` | `HEAD` (target) | ok | ok | 509.000 us | 522.000 us | ×1.03 | 0 ns | 0 ns | — |
| `simd_v128_load64_zero_lanes` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 49.000 us | 38.000 us | ×0.78 |
| `simd_v128_load64_zero_lanes` | `interp` | `HEAD` (target) | ok | ok | 1.340 ms | 1.348 ms | ×1.01 | 0 ns | 0 ns | — |
| `simd_v128_load8_lane5` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 13.000 us | ×0.52 |
| `simd_v128_load8_lane5` | `interp` | `HEAD` (target) | ok | ok | 677.000 us | 613.000 us | ×0.91 | 0 ns | 0 ns | — |
| `simd_v128_load8_splat_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 21.000 us | 15.000 us | ×0.71 |
| `simd_v128_load8_splat_lane0` | `interp` | `HEAD` (target) | ok | ok | 526.000 us | 526.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_v128_load8x8_s_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 25.000 us | 13.000 us | ×0.52 |
| `simd_v128_load8x8_s_lane0` | `interp` | `HEAD` (target) | ok | ok | 550.000 us | 542.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_v128_load8x8_u_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 13.000 us | ×0.59 |
| `simd_v128_load8x8_u_lane0` | `interp` | `HEAD` (target) | ok | ok | 525.000 us | 522.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_v128_load_store_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 31.000 us | 24.000 us | ×0.77 |
| `simd_v128_load_store_lane0` | `interp` | `HEAD` (target) | ok | ok | 1.039 ms | 1.002 ms | ×0.96 | 0 ns | 0 ns | — |
| `simd_v128_store16_lane2` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 27.000 us | 13.000 us | ×0.48 |
| `simd_v128_store16_lane2` | `interp` | `HEAD` (target) | ok | ok | 666.000 us | 658.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_v128_store32_lane1` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 24.000 us | 14.000 us | ×0.58 |
| `simd_v128_store32_lane1` | `interp` | `HEAD` (target) | ok | ok | 663.000 us | 661.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_v128_store64_lane1` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 23.000 us | 16.000 us | ×0.70 |
| `simd_v128_store64_lane1` | `interp` | `HEAD` (target) | ok | ok | 660.000 us | 661.000 us | ×1.00 | 0 ns | 0 ns | — |
| `simd_v128_store8_lane5` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 27.000 us | 18.000 us | ×0.67 |
| `simd_v128_store8_lane5` | `interp` | `HEAD` (target) | ok | ok | 665.000 us | 656.000 us | ×0.99 | 0 ns | 0 ns | — |
| `simd_v128_xor_lane0` | `aot` | `HEAD` (target) | unsupported | unsupported | - | - | — | 22.000 us | 18.000 us | ×0.82 |
| `simd_v128_xor_lane0` | `interp` | `HEAD` (target) | ok | ok | 660.000 us | 639.000 us | ×0.97 | 0 ns | 0 ns | — |

Safe/Fast ratios compare ReleaseSafe timing divided by ReleaseFast timing for the same target/case/engine.
```

## Raw output: keyvault codegen-cli

```text
$ cd /home/g/azure-sdk-for-zig
$ time WAMR=1 WAMR_BIN=<ReleaseFast wamr> codegen/cli/scripts/run.sh ... bench-keyvault/kv-fast
wamr: loaded AOT manifest from /home/g/azure-sdk-for-zig/codegen/cli/zig-out/bin/codegen-cli.composed.cwasm.json (8 cores precompiled)
collecting spec files from /spec
wasm trap: out of bounds memory access (code+0x90a235, local_func[1890] "array_hash_map.Custom([]const u8,void,array_hash_map.StringContext,true).getIndexAdapted__anon_52286"+0xaa1, mem_size=0x10f0000)
real 0.481
user 0.191
sys 0.288
ReleaseFast exit=2

$ time WAMR=1 WAMR_BIN=<ReleaseSafe wamr> codegen/cli/scripts/run.sh ... bench-keyvault/kv-safe
wamr: loaded AOT manifest from /home/g/azure-sdk-for-zig/codegen/cli/zig-out/bin/codegen-cli.composed.cwasm.json (8 cores precompiled)
thread 1060978 panic: access of union field 'enum_val' while field 'variant_val' is active
/work/wamr-710bench/src/component/canonical_abi.zig:1044:20: 0x10e744a in storeValFromDef (wamr)
            if (val.option_val.is_some) {
                   ^
/work/wamr-710bench/src/component/canonical_abi.zig:921:32: 0x10e68b8 in storeValReg (wamr)
                try storeValReg(memory, alignUp(ptr + disc_sz, payload_align), ct, payload.*, reg);
                               ^
/work/wamr-710bench/src/component/executor.zig:1402:28: 0x128c877 in storeInterfaceValue (wamr)
        try abi.storeValReg(mem, ptr, t, val, registry);
                           ^
/work/wamr-710bench/src/component/executor.zig:6033:50: 0x128d178 in wamrAotDispatchComponentTrampolineAot (wamr)
    const result = dispatchAotComponentTrampoline(ctx, lowered_sig.*, .{ a1, a2, a3, a4, a5, a6, a7, a8, a9, 0 }) catch |err| {
                                                 ^
/work/wamr-710bench/src/runtime/aot/host_trampolines.zig:187:66: 0x123bb28 in genericDispatcher (wamr)
        .canon_lower_aot => wamrAotDispatchComponentTrampolineAot(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
                                                                 ^
???:?:?: 0x7088c1895523 in ??? (???)
Unwind error at address `???:0x7088c1895523` (unwind info unavailable), remaining frames may be incorrect
/work/wamr-710bench/src/runtime/aot/runtime.zig:2202:25: 0x10e8573 in callFuncScalar (wamr)
            break :blk f(vmctx, raw[0], raw[1]);
                        ^
/work/wamr-710bench/src/component/executor.zig:6129:47: 0x128bd30 in dispatchAotCrossInstance (wamr)
    const results = aot_runtime.callFuncScalar(
                                              ^
/work/wamr-710bench/src/runtime/aot/host_trampolines.zig:188:56: 0x123b9a2 in genericDispatcher (wamr)
        .cross_instance => wamrAotDispatchCrossInstance(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
                                                       ^
???:?:?: 0x7088c189471c in ??? (???)
???:?:?: 0x7088c17c8704 in ??? (???)
???:?:?: 0x7088c17c7f3e in ??? (???)
???:?:?: 0x7088c17d1efa in ??? (???)
/work/wamr-710bench/src/runtime/aot/runtime.zig:2202:25: 0x10e8573 in callFuncScalar (wamr)
            break :blk f(vmctx, raw[0], raw[1]);
                        ^
/work/wamr-710bench/src/component/executor.zig:6129:47: 0x128bd30 in dispatchAotCrossInstance (wamr)
    const results = aot_runtime.callFuncScalar(
                                              ^
/work/wamr-710bench/src/runtime/aot/host_trampolines.zig:188:56: 0x123b9a2 in genericDispatcher (wamr)
        .cross_instance => wamrAotDispatchCrossInstance(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
                                                       ^
???:?:?: 0x7088c1894ebd in ??? (???)
/work/wamr-710bench/src/runtime/aot/runtime.zig:2202:25: 0x10e8573 in callFuncScalar (wamr)
            break :blk f(vmctx, raw[0], raw[1]);
                        ^
/work/wamr-710bench/src/component/executor.zig:6129:47: 0x128bd30 in dispatchAotCrossInstance (wamr)
    const results = aot_runtime.callFuncScalar(
                                              ^
/work/wamr-710bench/src/runtime/aot/host_trampolines.zig:188:56: 0x123b9a2 in genericDispatcher (wamr)
        .cross_instance => wamrAotDispatchCrossInstance(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
                                                       ^
???:?:?: 0x7088c18937da in ??? (???)
???:?:?: 0x7088c0f5a4af in ??? (???)
/work/wamr-710bench/src/runtime/aot/runtime.zig:2194:25: 0x10e83ff in callFuncScalar (wamr)
            break :blk f(vmctx);
                        ^
/work/wamr-710bench/src/component/executor.zig:6129:47: 0x128bd30 in dispatchAotCrossInstance (wamr)
    const results = aot_runtime.callFuncScalar(
                                              ^
/work/wamr-710bench/src/runtime/aot/host_trampolines.zig:188:56: 0x123b9a2 in genericDispatcher (wamr)
        .cross_instance => wamrAotDispatchCrossInstance(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9),
                                                       ^
???:?:?: 0x7088c18942ae in ??? (???)
/work/wamr-710bench/src/runtime/aot/runtime.zig:2194:25: 0x10e83ff in callFuncScalar (wamr)
            break :blk f(vmctx);
                        ^
/work/wamr-710bench/src/component/call_frame.zig:282:47: 0x10fae22 in executeCore (wamr)
        const got = aot_runtime.callFuncScalar(
                                              ^
/work/wamr-710bench/src/component/executor.zig:388:22: 0x10f7153 in callComponentFuncByLocal (wamr)
    frame.executeCore(exported.core_func_idx, &.{}, core_result_types) catch {
                     ^
/work/wamr-710bench/src/component/executor.zig:209:36: 0x10d6472 in callComponentFunc (wamr)
    return callComponentFuncByLocal(flat.owner, flat.local, args, out_results, allocator);
                                   ^
/work/wamr-710bench/src/main.zig:561:32: 0x107fa69 in runRun (wamr)
            return runComponent(wasm_data, allocator, path, wasm_args.items, env_list.items, map_dirs.items, allow_net.items, effective_log_level, cfg_entries, keyvalue_store_path, precompiled_cores);
                               ^
/work/wamr-710bench/src/main.zig:65:27: 0x1082253 in main (wamr)
        .run => try runRun(init, allocator, args[2..]),
                          ^
/home/g/.local/share/ghr/tools/ctaggart/zig/zig-x86_64-linux-0.16.0/lib/std/start.zig:190:5: 0x107b73d in _start (wamr)
    asm volatile (switch (native_arch) {
    ^
bash: line 1: 1060978 Aborted                 (core dumped) env WAMR=1 WAMR_BIN="$safe_bin" codegen/cli/scripts/run.sh "$spec" "$out_safe"
real 3.879
user 0.519
sys 1.334
ReleaseSafe exit=134
```
