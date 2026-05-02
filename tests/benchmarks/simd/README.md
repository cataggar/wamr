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

The `simd_i32x4_mem_sum8_4k_loop` row is a v128 register-pressure probe. Each loop iteration loads eight vectors from the two 4 KiB input arrays, reduces them with `i32x4.add`, and returns a scalar checksum from the final reduction. It intentionally keeps more than two vector values live in one block so backend changes to v128 register placement show up as reduced frame `ldr q` / `str q` traffic and better AOT timing.

The `simd_i32x4_shift_mix_4k_loop` row is a dynamic-shift throughput probe. Each loop iteration derives the scalar shift count from the loop index, exercises `i32x4.shl`, `i32x4.shr_u`, and `i32x4.shr_s`, stores a vector result, and returns one scalar checksum lane. This keeps shift counts data-dependent and above the lane width so modulo masking is covered in the AOT path.

The `simd_i16x8_mem_add_4k_loop` row is the 16-bit lane counterpart to the i32x4 memory-add loop. It walks two 4 KiB arrays as packed 16-bit lanes with `v128.load`, `i16x8.add`, and `v128.store`, then returns the final unsigned 16-bit lane as a scalar checksum.

The `simd_i16x8_shift_mix_4k_loop` row is the 16-bit dynamic-shift counterpart to the i32x4 shift probe. Each loop iteration derives scalar counts from the loop index, exercises `i16x8.shl`, `i16x8.shr_u`, and `i16x8.shr_s`, stores a vector result, and returns one unsigned 16-bit checksum lane.

The `simd_i16x8_arith_extra_4k_loop` row extends the halfword-lane memory probe with Q15 saturating rounded multiply, saturated add/subtract, signed/unsigned min/max, and unsigned rounding average operations. It uses high-bit halfword data so signed and unsigned arithmetic paths produce distinct checksums.

The `simd_i8x16_mem_add_4k_loop` row is the byte-lane memory-add probe. It walks the same 4 KiB input shape as the wider-lane rows using packed unsigned bytes, `v128.load`, wrapping `i8x16.add`, and `v128.store`, then returns the final byte lane as an unsigned scalar checksum.

The `simd_i8x16_shift_mix_4k_loop` row is the byte-lane dynamic-shift counterpart to the i16/i32 shift probes. Each loop iteration derives scalar counts from the vector index, exercises `i8x16.shl`, `i8x16.shr_u`, and `i8x16.shr_s`, stores a vector result, and returns one unsigned byte checksum lane. The derived counts intentionally exceed 8 so AOT modulo-8 count masking is covered.

The `simd_i8x16_arith_extra_4k_loop` row extends the byte-lane memory probe with saturated add/subtract, signed/unsigned min/max, and unsigned rounding average operations. It uses high-bit byte data so signed and unsigned arithmetic paths produce distinct checksums.

The `simd_i64x2_mem_add_4k_loop` row is the 64-bit lane counterpart to the integer memory-add probes. It walks the same 4 KiB input shape as packed 64-bit lanes with `v128.load`, wrapping `i64x2.add`, and `v128.store`, then extracts an `i64` checksum lane and returns it via `i32.wrap_i64`.

The `simd_i64x2_shift_mix_4k_loop` row is the 64-bit dynamic-shift counterpart to the narrower shift probes. Each loop iteration derives scalar counts from the vector index, exercises `i64x2.shl`, `i64x2.shr_u`, and `i64x2.shr_s`, stores a vector result, then extracts and wraps an `i64` checksum lane. The derived counts intentionally exceed 64 so AOT modulo-64 count masking is covered.

The small `simd_i32x4_*_lane0`, `simd_i16x8_*_lane0`, `simd_i8x16_*`, and `simd_i64x2_*` rows are coverage/status probes for individual opcode families. They intentionally return one scalar lane so interpreter, AOT, and optional Wasmtime rows can be compared before the runtime supports direct exported v128 values. The i16x8 and i8x16 comparison and replace-lane rows cover signed vs unsigned extraction of all-ones masks and high-bit lane values; the i64x2 rows wrap extracted i64 lanes to i32 for the exported checksum.

Wasmtime can be included as an external baseline:

```sh
zig build simd-bench -- --wasmtime --wasmtime-iterations 3 --iterations 100
```

Wasmtime rows currently use the CLI, so their timings include process startup and compilation. Treat them as correctness/status context rather than an apples-to-apples in-process throughput comparison.
