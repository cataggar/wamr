//! Compiler-free freestanding matched-sampler module, not a boot entry point.
const std = @import("std");
pub const sample = @import("bench/native_jit_aot_guest.zig");
pub const aot = sample.aot;

export fn wamr_jit_aot_sample_link_check(allocator: *const std.mem.Allocator, native: *const aot.Platform, wasm: [*]const u8, wasm_len: usize, cwasm: [*]const u8, cwasm_len: usize, request: *const [64]u8, resolution: u64, out: *sample.Capture, writer: *std.Io.Writer) bool {
    sample.run(allocator.*, native.*, wasm[0..wasm_len], cwasm[0..cwasm_len], request, resolution, out) catch return false;
    out.writeRecord(writer) catch return false;
    return true;
}
