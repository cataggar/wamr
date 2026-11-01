//! Command-line options for the focused loop-pass benchmark.
const std = @import("std");
const core = @import("bench_loop_passes_core.zig");

pub const Options = struct {
    baseline: []const u8 = "origin/main",
    target: []const u8 = "HEAD",
    repo: ?[]const u8 = null,
    profile: []const u8 = "authoritative",
    warmups: ?i64 = null,
    runs: ?i64 = null,
    optimize: []const u8 = "ReleaseFast",
    output: ?[]const u8 = null,
    emit: []const u8 = "markdown",
    minimum: [core.fixtures.len]?f64 = @splat(null),
    minimum_order: [core.fixtures.len]usize = @splat(0),
    minimum_count: usize = 0,
};

fn countValue(raw: []const u8) !i64 {
    const value = std.mem.trim(u8, raw, " \t\r\n\x0b\x0c");
    for (value, 0..) |byte, index| {
        if (byte == '_' and (index == 0 or index + 1 == value.len or
            !std.ascii.isDigit(value[index - 1]) or !std.ascii.isDigit(value[index + 1])))
            return error.InvalidCharacter;
    }
    return std.fmt.parseInt(i64, value, 10);
}

pub fn parse(args: []const []const u8) !Options {
    var result: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return error.HelpRequested;
        const split = std.mem.indexOfScalar(u8, arg, '=');
        const name = if (split) |at| arg[0..at] else arg;
        const value = if (split) |at| arg[at + 1 ..] else blk: {
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            break :blk args[i];
        };
        if (std.mem.eql(u8, name, "--baseline")) {
            result.baseline = value;
        } else if (std.mem.eql(u8, name, "--target")) {
            result.target = value;
        } else if (std.mem.eql(u8, name, "--repo")) {
            result.repo = value;
        } else if (std.mem.eql(u8, name, "--profile")) {
            result.profile = value;
        } else if (std.mem.eql(u8, name, "--warmups")) {
            result.warmups = try countValue(value);
        } else if (std.mem.eql(u8, name, "--runs")) {
            result.runs = try countValue(value);
        } else if (std.mem.eql(u8, name, "--optimize")) {
            result.optimize = value;
        } else if (std.mem.eql(u8, name, "--out")) {
            result.output = value;
        } else if (std.mem.eql(u8, name, "--emit")) {
            result.emit = value;
        } else if (std.mem.eql(u8, name, "--min-speedup")) {
            const separator = std.mem.indexOfScalar(u8, value, '=') orelse return error.InvalidSpeedupThreshold;
            const fixture_name = value[0..separator];
            const index = for (core.fixtures, 0..) |fixture, index| {
                if (std.mem.eql(u8, fixture.name, fixture_name)) break index;
            } else return error.InvalidSpeedupThreshold;
            const number = std.mem.trim(u8, value[separator + 1 ..], " \t\r\n\x0b\x0c");
            const unsigned = if (std.mem.startsWith(u8, number, "+") or std.mem.startsWith(u8, number, "-")) number[1..] else number;
            if (unsigned.len >= 2 and unsigned[0] == '0' and std.ascii.toLower(unsigned[1]) == 'x')
                return error.InvalidSpeedupThreshold;
            const threshold: f64 = if (std.ascii.eqlIgnoreCase(number, "infinity") or std.ascii.eqlIgnoreCase(number, "+infinity"))
                std.math.inf(f64)
            else if (std.ascii.eqlIgnoreCase(number, "-infinity"))
                -std.math.inf(f64)
            else
                std.fmt.parseFloat(f64, number) catch return error.InvalidSpeedupThreshold;
            if (result.minimum[index] == null) {
                result.minimum_order[result.minimum_count] = index;
                result.minimum_count += 1;
            }
            result.minimum[index] = threshold;
        } else return error.UnknownOption;
    }
    _ = try core.counts(result.profile, result.warmups, result.runs);
    if (!std.mem.eql(u8, result.emit, "markdown") and !std.mem.eql(u8, result.emit, "github")) return error.InvalidOutputMode;
    return result;
}

test "benchmark options retain defaults, overrides, and speedup thresholds" {
    const defaults = try parse(&.{});
    try std.testing.expectEqualStrings("origin/main", defaults.baseline);
    try std.testing.expectEqualStrings("authoritative", defaults.profile);
    const options = try parse(&.{ "--baseline=old", "--profile", "ci", "--runs", "2", "--emit", "github", "--min-speedup", "unroll4=20", "--min-speedup=iv_store=-2", "--min-speedup", "unroll4=30" });
    try std.testing.expectEqualStrings("old", options.baseline);
    try std.testing.expectEqualStrings("ci", options.profile);
    try std.testing.expectEqual(@as(?i64, 2), options.runs);
    try std.testing.expectEqual(@as(?f64, -2), options.minimum[0]);
    try std.testing.expectEqual(@as(?f64, 30), options.minimum[1]);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, options.minimum_order[0..options.minimum_count]);
    try std.testing.expectError(error.InvalidSpeedupThreshold, parse(&.{ "--min-speedup", "unknown=1" }));
    try std.testing.expectError(error.InvalidSpeedupThreshold, parse(&.{ "--min-speedup", "unroll4=0x1p0" }));
    try std.testing.expectError(error.InvalidRuns, parse(&.{ "--runs", "0" }));
    try std.testing.expectEqual(@as(?i64, 10), (try parse(&.{ "--runs", " 1_0 " })).runs);
    try std.testing.expectError(error.InvalidCharacter, parse(&.{ "--runs", "1__0" }));
    try std.testing.expectError(error.InvalidWarmups, parse(&.{ "--warmups", "-1" }));
    try std.testing.expectError(error.InvalidOutputMode, parse(&.{ "--emit", "json" }));
}
