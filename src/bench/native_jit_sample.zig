//! No-import matched workload sampling through the checked staged native API.
const std = @import("std");
const aot = @import("../api/aot.zig");

pub const rounds: u32 = 2000;
pub const invocation_count = 4;
pub const runtime_limits: aot.Options = .{
    .max_memory_pages = 8,
    .max_table_elements = 16,
    .max_heap_bytes = 16 * 1024 * 1024,
    .max_reserved_bytes = 8 * 1024 * 1024,
    .max_code_bytes = 4 * 1024 * 1024,
};

pub const Invocation = struct {
    ns: ?u64 = null,
    outcome: []const u8 = "not-called",
    value: ?u32 = null,
    diagnostic: ?[]const u8 = null,
};

pub const Report = struct {
    schema_version: u32 = 1,
    kind: []const u8 = "wamr-native-jit-sample",
    qualification: []const u8 = "requires-independent-image-and-deployment-evidence",
    request_sha256: []const u8,
    mode: []const u8,
    compiler_embedded: bool,
    wasm_sha256: []const u8,
    wasm_bytes: usize,
    cwasm_sha256: ?[]const u8 = null,
    cwasm_bytes: usize = 0,
    workload: []const u8 = "volatile-compute-memory-2000",
    expected: u32 = expected(),
    lifecycle: []const u8 = "same-instance-workload-initializes-memory-in-timed-call",
    clock_resolution_ns: u64,
    compile_ns: ?u64 = null,
    compiler_phases_ns: ?struct { parse: u64, lower: u64, optimize: u64, codegen: u64, emit: u64 } = null,
    compiler_peak_bytes: usize = 0,
    compiler_retained_bytes: usize = 0,
    compiler_polls: u64 = 0,
    load_ns: ?u64 = null,
    instantiate_ns: ?u64 = null,
    start_ns: ?u64 = null,
    growth_ns: ?u64 = null,
    growth_previous_pages: ?u32 = null,
    fuel_per_invocation: ?u32 = null,
    invocations: [invocation_count]Invocation = @splat(.{}),
    memory_before: ?aot.MemoryStats = null,
    memory_after: ?aot.MemoryStats = null,
    caller_peak_bytes: usize = 0,
    caller_live_after_teardown: usize = 0,
    reserved_after_teardown: usize = 0,
    failure_stage: ?[]const u8 = null,
    failure: ?[]const u8 = null,
};

pub fn expected() u32 {
    @setEvalBranchQuota(10000);
    var cells: [256]u32 = undefined;
    for (&cells, 0..) |*cell, index| cell.* = @as(u32, @intCast(index)) *% 17 +% 3;
    var result: u32 = 0;
    for (0..rounds) |index| {
        const i: u32 = @intCast(index);
        const p = &cells[i & 255];
        p.* = if ((i & 1) == 0) p.* *% 3 +% i else p.* ^ (i *% 7);
        result +%= p.*;
    }
    return result;
}

fn elapsed(native: aot.Platform, since: u64) !u64 {
    return std.math.sub(u64, try native.monotonicNs(), since) catch error.ClockFailed;
}

pub fn run(allocator: std.mem.Allocator, native: aot.Platform, bytes: []const u8, report: *Report) !void {
    var options = runtime_limits;
    options.max_run_fuel = report.fuel_per_invocation;
    report.failure_stage = "load";
    var begin = try native.monotonicNs();
    const instance = try aot.Instance.loadModule(allocator, native, bytes, options);
    defer instance.deinit();
    report.load_ns = try elapsed(native, begin);
    report.failure_stage = "instantiate";
    begin = try native.monotonicNs();
    try instance.instantiate(&.{}, options);
    report.instantiate_ns = try elapsed(native, begin);
    report.failure_stage = "start";
    begin = try native.monotonicNs();
    const startup = try instance.start();
    report.start_ns = try elapsed(native, begin);
    if (startup != .returned or startup.returned != 0) return error.UnexpectedStartOutcome;
    report.memory_before = instance.memoryStats();
    for (&report.invocations, 0..) |*invocation, index| {
        report.failure_stage = "invocation";
        begin = try native.monotonicNs();
        var values: [1]aot.Value = undefined;
        const outcome = try instance.call("workload", &.{.{ .i32 = rounds }}, &values);
        invocation.outcome = @tagName(outcome);
        switch (outcome) {
            .returned => |count| {
                if (count != 1 or values[0] != .i32) return error.UnexpectedResult;
                invocation.value = @bitCast(values[0].i32);
            },
            .trap => |trap| invocation.diagnostic = @tagName(trap),
            .host_error => |err| invocation.diagnostic = @errorName(err),
            .exit => invocation.diagnostic = "unexpected-exit",
        }
        invocation.ns = try elapsed(native, begin);
        if (invocation.value != report.expected) return error.UnexpectedResult;
        if (index == 0) {
            report.failure_stage = "growth";
            begin = try native.monotonicNs();
            const growth = try instance.call("grow", &.{.{ .i32 = 1 }}, &values);
            report.growth_ns = try elapsed(native, begin);
            if (growth != .returned or growth.returned != 1 or values[0] != .i32 or values[0].i32 != 2)
                return error.UnexpectedGrowth;
            report.growth_previous_pages = 2;
        }
    }
    report.memory_after = instance.memoryStats();
    report.failure_stage = null;
}
