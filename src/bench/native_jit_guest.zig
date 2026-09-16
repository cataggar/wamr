//! Explicit freestanding fast/full sampling, never a loader compile fallback.
const std = @import("std");
const jit = @import("../api/jit.zig");
const common = @import("native_jit_capture.zig");
pub const Capture = common.Capture;
pub const Preset = enum { fast, full };
pub const aot = jit.aot;

fn compilerClock(ctx: ?*anyopaque) error{ClockFailed}!u64 {
    const native: *const aot.Platform = @ptrCast(@alignCast(ctx.?));
    return native.monotonicNs() catch error.ClockFailed;
}

/// wasm must be the matching embedded workload, not an external fetch. All
/// compiler/runtime limits are fixed; cancellation remains cooperative.
/// out is initialized before compiler/platform work and retains failure details.
/// Every compiler allocation and instance reservation is released before return.
pub fn run(allocator: std.mem.Allocator, native: aot.Platform, wasm: []const u8, preset: Preset, request_sha256: []const u8, clock_resolution_ns: u64, out: *Capture) !void {
    try out.initialize(request_sha256, if (preset == .fast) .fast else .full, wasm, clock_resolution_ns);
    var accounting = common.Accounting.init(allocator, native);
    defer accounting.finish(out);
    execute(accounting.counter.allocator(), accounting.pages.platform(), wasm, preset, out) catch |err| {
        out.report.failure = @errorName(err);
        return err;
    };
    if (accounting.counter.live != 0 or accounting.pages.reserved != 0) return error.IncompleteTeardown;
}

fn execute(allocator: std.mem.Allocator, native: aot.Platform, wasm: []const u8, preset: Preset, out: *Capture) !void {
    out.report.failure_stage = "compile";
    const begin = try native.monotonicNs();
    var artifact = try jit.compile(allocator, wasm, .{
        .preset = if (preset == .fast) .fast else .full,
        .context = @constCast(&native),
        .monotonic_ns = compilerClock,
    });
    defer artifact.deinit();
    out.report.compile_ns = std.math.sub(u64, try native.monotonicNs(), begin) catch return error.ClockFailed;
    out.report.compiler_phases_ns = .{
        .parse = artifact.metrics.parse_ns orelse return error.ClockFailed,
        .lower = artifact.metrics.lower_ns orelse return error.ClockFailed,
        .optimize = artifact.metrics.optimize_ns orelse return error.ClockFailed,
        .codegen = artifact.metrics.codegen_ns orelse return error.ClockFailed,
        .emit = artifact.metrics.emit_ns orelse return error.ClockFailed,
    };
    out.report.compiler_peak_bytes = artifact.compiler_peak_bytes;
    out.report.compiler_retained_bytes = artifact.compiler_retained_bytes;
    out.report.compiler_polls = artifact.polls;
    out.identifyCode(artifact.bytes);
    try common.sample.run(allocator, native, artifact.bytes, &out.report);
}
