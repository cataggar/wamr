//! Summary statistics for the existing WASI P3 profile reports.
const std = @import("std");
const integer_median = @import("integer_median.zig");

pub const Stats = struct {
    samples: usize,
    min_ns: i128,
    median_ns: i128,
    p95_ns: i128,
    max_ns: i128,
};

pub fn summarize(allocator: std.mem.Allocator, values: []const i128) !Stats {
    if (values.len == 0) return error.EmptySamples;
    const ordered = try allocator.dupe(i128, values);
    defer allocator.free(ordered);
    std.mem.sort(i128, ordered, {}, std.sort.asc(i128));
    const middle = ordered.len / 2;
    const median: i128 = if (ordered.len % 2 == 1)
        ordered[middle]
    else blk: {
        // Python statistics.median divides an exact integer sum as a float.
        const result = integer_median.average(ordered[middle - 1], ordered[middle]);
        if (result >= 0x1p127 or result < -0x1p127) return error.ProfileIntegerOverflow;
        break :blk @intFromFloat(result);
    };
    const rank: usize = @intFromFloat(@as(f64, @floatFromInt(ordered.len)) * 0.95 + 0.999);
    const percentile = @min(ordered.len - 1, if (rank == 0) 0 else rank - 1);
    return .{
        .samples = ordered.len,
        .min_ns = ordered[0],
        .median_ns = median,
        .p95_ns = ordered[percentile],
        .max_ns = ordered[ordered.len - 1],
    };
}

test "profile statistics preserve median and nearest-rank p95" {
    const allocator = std.testing.allocator;
    const cold = [_]i128{ 50, 10, 30, 20, 40 };
    const cold_stats = try summarize(allocator, &cold);
    try std.testing.expectEqualDeep(Stats{ .samples = 5, .min_ns = 10, .median_ns = 30, .p95_ns = 50, .max_ns = 50 }, cold_stats);
    const warm = [_]i128{ 10, 2, 8, 4, 6, 9, 1, 7, 3, 5 };
    const warm_stats = try summarize(allocator, &warm);
    try std.testing.expectEqualDeep(Stats{ .samples = 10, .min_ns = 1, .median_ns = 5, .p95_ns = 10, .max_ns = 10 }, warm_stats);
    try std.testing.expectError(error.EmptySamples, summarize(allocator, &.{}));
}

test "even medians use floating-point conversion and truncate toward zero" {
    const allocator = std.testing.allocator;
    const negative = try summarize(allocator, &.{ -4, -1 });
    try std.testing.expectEqual(@as(i128, -2), negative.median_ns);
    const large = try summarize(allocator, &.{ std.math.maxInt(i64) - 1, std.math.maxInt(i64) });
    try std.testing.expectEqual(@as(i128, 9223372036854775808), large.median_ns);
    const unsigned = try summarize(allocator, &.{ std.math.maxInt(u64), std.math.maxInt(u64) });
    try std.testing.expectEqual(@as(i128, 18446744073709551616), unsigned.median_ns);
    const high = std.math.maxInt(i128) / 2 + 1;
    const high_stats = try summarize(allocator, &.{ high, high });
    try std.testing.expectEqual(@as(i128, 0x40000000000000000000000000000000), high_stats.median_ns);
    const low_stats = try summarize(allocator, &.{ std.math.minInt(i128), std.math.minInt(i128) });
    try std.testing.expectEqual(std.math.minInt(i128), low_stats.median_ns);
    try std.testing.expectError(error.ProfileIntegerOverflow, summarize(allocator, &.{ std.math.maxInt(i128), std.math.maxInt(i128) }));
}
