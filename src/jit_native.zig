//! Opt-in native JIT root. The compiler-free root remains aot_native.zig.
const std = @import("std");
pub const jit = @import("api/jit.zig");
pub const aot = jit.aot;
pub const std_options: std.Options = .{ .logFn = log };

fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    _ = level;
    _ = scope;
    _ = format;
    _ = args;
}

// Retains the real compiler API in the freestanding link audit, even without a
// downstream application. This is a Zig ABI entry point, not the public C API.
export fn wamr_jit_link_check(allocator: *const std.mem.Allocator, bytes: [*]const u8, length: usize, options: *const jit.Options, out: *jit.Artifact) bool {
    out.* = jit.compile(allocator.*, bytes[0..length], options.*) catch return false;
    return true;
}

export fn wamr_jit_runtime_link_check(allocator: *const std.mem.Allocator, native: *const aot.Platform, bytes: [*]const u8, length: usize, options: *const aot.Options) bool {
    const instance = aot.Instance.load(allocator.*, native.*, bytes[0..length], &.{}, options.*) catch return false;
    defer instance.deinit();
    switch (instance.start() catch return false) {
        .returned => {},
        else => return false,
    }
    var results: [1]aot.Value = undefined;
    return switch (instance.call("workload", &.{.{ .i32 = 2000 }}, &results) catch return false) {
        .returned => |count| count == 1,
        else => false,
    };
}
