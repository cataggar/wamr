//! OSS-Fuzz shim for fuzz-canon. Exposes `LLVMFuzzerTestOneInput`
//! so libFuzzer-driven binaries can drive the same per-input
//! function the CLI harness in `canon.zig` uses.
//!
//! Built into a static library by the `fuzz-oss` step in build.zig
//! and linked with `$LIB_FUZZING_ENGINE` (clang `-fsanitize=fuzzer`)
//! by `oss-fuzz/build.sh`. The shim is local-only; no upstream
//! OSS-Fuzz submission is implied. See `tests/fuzz/OSS_FUZZ.md`.

const std = @import("std");
const canon = @import("canon.zig");

var arena_state: std.heap.ArenaAllocator = undefined;
var initialized: bool = false;

export fn LLVMFuzzerTestOneInput(data: [*]const u8, size: usize) c_int {
    if (!initialized) {
        arena_state = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        initialized = true;
    }
    _ = arena_state.reset(.retain_capacity);

    const bytes: []const u8 = if (size == 0) &[_]u8{} else data[0..size];
    // Typed canonical-ABI errors are expected. OOM on a small input
    // is a bug; let the panic path through libFuzzer.
    canon.runOnce(arena_state.allocator(), bytes) catch {};
    return 0;
}
