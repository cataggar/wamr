//! Downstream-style public-module link audit, not an executable boot entry.
const std = @import("std");
const guest = @import("guest");
const workload = @import("wamr-jit-workload");
const with_compiler = @import("guest_audit_options").with_compiler;
pub const std_options: std.Options = if (with_compiler) guest.std_options else .{};

export fn wamr_jit_guest_sample_link_check(allocator: *const std.mem.Allocator, platform: *const guest.aot.Platform, full: bool, request: *const [64]u8, resolution: u64, cwasm: [*]const u8, cwasm_len: usize, out: *guest.sample.Capture, writer: *std.Io.Writer) bool {
    if (with_compiler) {
        guest.sample.run(allocator.*, platform.*, workload.wasm, if (full) .full else .fast, request, resolution, out) catch return false;
    } else {
        if (full) return false;
        guest.sample.run(allocator.*, platform.*, workload.wasm, cwasm[0..cwasm_len], request, resolution, out) catch return false;
    }
    out.writeRecord(writer) catch return false;
    return true;
}
