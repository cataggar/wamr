//! CLI options for the experimental native SIMD benchmark.
const std = @import("std");
const optimize = @import("benchmark_optimize.zig");

pub const Options = struct {
    baseline: []const u8 = "origin/main",
    target: []const u8 = "HEAD",
    runs: usize = 3,
    iterations: usize = 10_000,
    wasmtime: bool = false,
    wasmtime_path: []const u8 = "wasmtime",
    wasmtime_iterations: ?usize = null,
    repo: ?[]const u8 = null,
    output: ?[]const u8 = null,
    emit: []const u8 = "markdown",
    mode: []const u8 = "ReleaseFast",
};

fn positive(raw: []const u8) !usize {
    const text = std.mem.trim(u8, raw, " \t\r\n\x0b\x0c");
    for (text, 0..) |byte, index| {
        if (byte >= 0x80) return error.UnsupportedSimdInteger;
        if (byte == '_' and (index == 0 or index + 1 == text.len or
            !std.ascii.isDigit(text[index - 1]) or !std.ascii.isDigit(text[index + 1])))
            return error.InvalidSimdInteger;
    }
    const value = std.fmt.parseInt(i128, text, 10) catch |err| switch (err) {
        error.InvalidCharacter => return error.InvalidSimdInteger,
        error.Overflow => return error.SimdIntegerTooLarge,
    };
    if (value <= 0) return error.NonpositiveSimdCount;
    return std.math.cast(usize, value) orelse error.SimdIntegerTooLarge;
}

pub fn parse(args: []const []const u8) !Options {
    var result: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return error.HelpRequested;
        if (std.mem.eql(u8, arg, "--wasmtime")) {
            result.wasmtime = true;
            continue;
        }
        const equal = std.mem.indexOfScalar(u8, arg, '=');
        const name = if (equal) |at| arg[0..at] else arg;
        const value = if (equal) |at| arg[at + 1 ..] else blk: {
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            break :blk args[i];
        };
        if (std.mem.eql(u8, name, "--baseline")) {
            result.baseline = value;
        } else if (std.mem.eql(u8, name, "--target")) {
            result.target = value;
        } else if (std.mem.eql(u8, name, "--runs")) {
            result.runs = try positive(value);
        } else if (std.mem.eql(u8, name, "--iterations")) {
            result.iterations = try positive(value);
        } else if (std.mem.eql(u8, name, "--wasmtime-path")) {
            result.wasmtime_path = value;
        } else if (std.mem.eql(u8, name, "--wasmtime-iterations")) {
            result.wasmtime_iterations = try positive(value);
        } else if (std.mem.eql(u8, name, "--repo")) {
            result.repo = value;
        } else if (std.mem.eql(u8, name, "--out")) {
            result.output = value;
        } else if (std.mem.eql(u8, name, "--emit")) {
            result.emit = value;
        } else if (std.mem.eql(u8, name, "--optimize")) {
            result.mode = value;
        } else return error.UnknownOption;
    }
    _ = try optimize.parseModes(result.mode);
    if (!std.mem.eql(u8, result.emit, "markdown") and !std.mem.eql(u8, result.emit, "github")) return error.InvalidOutputMode;
    return result;
}

test "SIMD CLI retains defaults, modes, and iteration overrides" {
    const defaults = try parse(&.{});
    try std.testing.expectEqualStrings("origin/main", defaults.baseline);
    try std.testing.expectEqual(@as(usize, 3), defaults.runs);
    try std.testing.expectEqual(@as(usize, 10_000), defaults.iterations);
    const options = try parse(&.{ "--target=new", "--runs", "2", "--iterations", "1_000", "--wasmtime", "--wasmtime-path", "wt", "--wasmtime-iterations", "5", "--optimize", "both", "--emit", "github" });
    try std.testing.expectEqualStrings("new", options.target);
    try std.testing.expectEqual(@as(usize, 2), options.runs);
    try std.testing.expectEqual(@as(usize, 1000), options.iterations);
    try std.testing.expect(options.wasmtime);
    try std.testing.expectEqual(@as(?usize, 5), options.wasmtime_iterations);
    try std.testing.expectEqualStrings("both", options.mode);
    try std.testing.expectError(error.NonpositiveSimdCount, parse(&.{ "--runs", "0" }));
    try std.testing.expectError(error.InvalidSimdInteger, parse(&.{ "--runs", "1__0" }));
    try std.testing.expectError(error.InvalidOptimizeMode, parse(&.{ "--optimize", "Debug" }));
    try std.testing.expectError(error.InvalidOutputMode, parse(&.{ "--emit", "json" }));
}
