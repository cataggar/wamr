//! Explicit Linux sampler; qualification and process capture belong to the host.
const std = @import("std");
const sample = @import("native_jit_sample.zig");
const linux = @import("native_linux.zig");
const allocations = @import("native_allocations.zig");
const fixture = @import("jit_bench_fixture");

fn compilerClock(_: ?*anyopaque) error{ClockFailed}!u64 {
    return linux.now() catch error.ClockFailed;
}

pub fn main(comptime with_compiler: bool, init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 or args[2].len != 64) return error.InvalidArguments;
    for (args[2]) |c| if (!std.ascii.isHex(c)) return error.InvalidArguments;
    const mode = args[1];
    if (with_compiler) {
        if (!std.mem.eql(u8, mode, "fast") and !std.mem.eql(u8, mode, "full")) return error.InvalidArguments;
    } else if (!std.mem.eql(u8, mode, "aot")) return error.InvalidArguments;
    var report: sample.Report = .{
        .request_sha256 = args[2],
        .mode = mode,
        .compiler_embedded = with_compiler,
        .wasm_sha256 = &fixture.wasm_sha256,
        .wasm_bytes = fixture.wasm_size,
        .clock_resolution_ns = try linux.resolution(.MONOTONIC),
        .fuel_per_invocation = if (with_compiler) 100_000 else null,
    };
    var counter: allocations.Counter = .{ .child = init.gpa };
    var pages: linux.Pages = .{};
    var code_hash: [64]u8 = undefined;
    execute(with_compiler, counter.allocator(), &pages, &report, &code_hash) catch |err| {
        report.failure = @errorName(err);
    };
    report.caller_peak_bytes = counter.peak;
    report.caller_live_after_teardown = counter.live;
    report.reserved_after_teardown = pages.reserved();
    if (counter.live != 0 or pages.reserved() != 0) return error.IncompleteTeardown;
    var buffer: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &buffer);
    try out.interface.writeAll("WAMR_JIT_SAMPLE=");
    try std.json.Stringify.value(report, .{}, &out.interface);
    try out.interface.writeByte('\n');
    try out.interface.flush();
    if (report.failure != null) return error.SampleFailed;
}

fn execute(comptime with_compiler: bool, allocator: std.mem.Allocator, pages: *linux.Pages, report: *sample.Report, code_hash: *[64]u8) !void {
    if (with_compiler) {
        const jit = @import("../api/jit.zig");
        report.failure_stage = "compile";
        const begin = try linux.now();
        var artifact = try jit.compile(allocator, fixture.wasm, .{
            .preset = if (std.mem.eql(u8, report.mode, "fast")) .fast else .full,
            .monotonic_ns = compilerClock,
        });
        defer artifact.deinit();
        report.compile_ns = try std.math.sub(u64, try linux.now(), begin);
        report.compiler_phases_ns = .{
            .parse = artifact.metrics.parse_ns.?,
            .lower = artifact.metrics.lower_ns.?,
            .optimize = artifact.metrics.optimize_ns.?,
            .codegen = artifact.metrics.codegen_ns.?,
            .emit = artifact.metrics.emit_ns.?,
        };
        report.compiler_peak_bytes = artifact.compiler_peak_bytes;
        report.compiler_retained_bytes = artifact.compiler_retained_bytes;
        report.compiler_polls = artifact.polls;
        identifyCode(artifact.bytes, report, code_hash);
        try sample.run(allocator, pages.platform(), artifact.bytes, report);
    } else {
        identifyCode(fixture.aot, report, code_hash);
        try sample.run(allocator, pages.platform(), fixture.aot, report);
    }
}

fn identifyCode(bytes: []const u8, report: *sample.Report, output: *[64]u8) void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    output.* = std.fmt.bytesToHex(digest, .lower);
    report.cwasm_sha256 = output;
    report.cwasm_bytes = bytes.len;
}
