//! Separately linkable compiler-free half of the matched sampler.
const std = @import("std");
const common = @import("native_jit_capture.zig");
pub const Capture = common.Capture;
pub const aot = common.aot;

/// The integrator embeds matching wasm and build-time-compiled cwasm. Identity
/// comparison is an independent image receipt obligation, not a runtime compile.
pub fn run(allocator: std.mem.Allocator, native: aot.Platform, wasm: []const u8, cwasm: []const u8, request_sha256: []const u8, clock_resolution_ns: u64, out: *Capture) !void {
    try out.initialize(request_sha256, .aot, wasm, clock_resolution_ns);
    var accounting = common.Accounting.init(allocator, native);
    defer accounting.finish(out);
    out.identifyCode(cwasm);
    common.sample.run(accounting.counter.allocator(), accounting.pages.platform(), cwasm, &out.report) catch |err| {
        out.report.failure = @errorName(err);
        return err;
    };
    if (accounting.counter.live != 0 or accounting.pages.reserved != 0) return error.IncompleteTeardown;
}
