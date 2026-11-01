//! Summary statistics for finite floating-point benchmark samples.
const std = @import("std");
const ExactSum = @import("float_sum.zig").ExactSum;

pub const Stats = struct {
    runs: usize,
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
    range: f64,
};

pub fn summarize(allocator: std.mem.Allocator, samples: []const f64) !Stats {
    if (samples.len == 0) return error.EmptyMeasurements;
    var sum: ExactSum = .{};
    defer sum.deinit(allocator);
    const ordered = try allocator.dupe(f64, samples);
    defer allocator.free(ordered);
    for (samples) |sample| try sum.add(allocator, sample);
    std.mem.sort(f64, ordered, {}, std.sort.asc(f64));
    const midpoint = ordered.len / 2;
    return .{
        .runs = samples.len,
        .mean = sum.result() / @as(f64, @floatFromInt(samples.len)),
        .median = if (ordered.len % 2 == 0) (ordered[midpoint - 1] + ordered[midpoint]) / 2 else ordered[midpoint],
        .min = ordered[0],
        .max = ordered[ordered.len - 1],
        .range = ordered[ordered.len - 1] - ordered[0],
    };
}

test "finite sample summaries preserve accurate means and original order" {
    const samples = [_]f64{ 1e16, 1, 1, 1 };
    const actual = try summarize(std.testing.allocator, &samples);
    try std.testing.expectEqualDeep(Stats{
        .runs = 4,
        .mean = 2500000000000001,
        .median = 1,
        .min = 1,
        .max = 1e16,
        .range = 1e16 - 1,
    }, actual);
    try std.testing.expectEqualSlices(f64, &.{ 1e16, 1, 1, 1 }, &samples);
    try std.testing.expectError(error.EmptyMeasurements, summarize(std.testing.allocator, &.{}));
    try std.testing.expectError(error.InvalidMeasurement, summarize(std.testing.allocator, &.{std.math.nan(f64)}));
}
