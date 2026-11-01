//! JSON report schema shared with the upstream wasi-testsuite reporter.
const std = @import("std");
const suite = @import("wasi_fixture_suite.zig");
const Allocator = std.mem.Allocator;

pub const FixtureResult = struct {
    name: []const u8,
    executed: bool,
    outcome: suite.Outcome,
    duration_s: f64,
    failures: []const []const u8,

    pub fn init(name: []const u8, executed: bool, expected_fail: bool, duration_s: f64, failures: []const []const u8) FixtureResult {
        return .{
            .name = name,
            .executed = executed,
            .outcome = suite.Outcome.evaluate(executed, failures.len != 0, expected_fail),
            .duration_s = duration_s,
            .failures = failures,
        };
    }
};

pub const Runtime = struct { name: []const u8, version: []const u8 };

pub const SuiteResult = struct {
    name: []const u8,
    runtime: Runtime,
    duration_s: f64,
    failed: usize,
    skipped: usize,
    passed: usize,
    xfailed: usize,
    xpassed: usize,
    tests: []const FixtureResult,

    pub fn init(name: []const u8, runtime: Runtime, duration_s: f64, tests: []const FixtureResult) SuiteResult {
        var counts: suite.Counts = .{};
        for (tests) |test_result| counts.add(test_result.outcome);
        return .{
            .name = name,
            .runtime = runtime,
            .duration_s = duration_s,
            .failed = counts.failed,
            .skipped = counts.skipped,
            .passed = counts.passed,
            .xfailed = counts.xfail,
            .xpassed = counts.xpass,
            .tests = tests,
        };
    }
};

pub const Report = struct {
    execution: struct { start_timestamp: []const u8, finish_timestamp: []const u8 },
    results: []const SuiteResult,

    pub fn json(self: Report, allocator: Allocator) ![]u8 {
        return std.json.Stringify.valueAlloc(allocator, self, .{});
    }
};

/// Format a UTC Unix timestamp using the upstream reporter's millisecond precision.
pub fn timestamp(allocator: Allocator, nanoseconds: i96) ![]u8 {
    if (nanoseconds < 0 or nanoseconds >= 253_402_300_800 * std.time.ns_per_s)
        return error.UnsupportedReportTimestamp;
    const seconds: u64 = @intCast(@divTrunc(nanoseconds, std.time.ns_per_s));
    const milliseconds: u16 = @intCast(@divTrunc(@rem(nanoseconds, std.time.ns_per_s), std.time.ns_per_ms));
    const epoch: std.time.epoch.EpochSeconds = .{ .secs = seconds };
    const date = epoch.getEpochDay().calculateYearDay();
    const month_day = date.calculateMonthDay();
    const day = epoch.getDaySeconds();
    return std.fmt.allocPrint(allocator, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}Z", .{
        date.year,
        month_day.month.numeric(),
        @as(u8, month_day.day_index) + 1,
        day.getHoursIntoDay(),
        day.getMinutesIntoHour(),
        day.getSecondsIntoMinute(),
        milliseconds,
    });
}

test "UTC report timestamps use millisecond precision" {
    const allocator = std.testing.allocator;
    const epoch = try timestamp(allocator, 0);
    defer allocator.free(epoch);
    try std.testing.expectEqualStrings("1970-01-01T00:00:00.000Z", epoch);
    const leap = try timestamp(allocator, 1_709_210_096_123_456_789);
    defer allocator.free(leap);
    try std.testing.expectEqualStrings("2024-02-29T12:34:56.123Z", leap);
    try std.testing.expectError(error.UnsupportedReportTimestamp, timestamp(allocator, -1));
}

test "upstream report shape preserves xfail, xpass, skips, and failure messages" {
    const fixtures = [_]FixtureResult{
        .init("ok", true, false, 1.0, &.{}),
        .init("skip", false, false, 0.0, &.{}),
        .init("xfail", true, true, 2.0, &.{"expected failure"}),
        .init("xpass", true, true, 3.0, &.{}),
        .init("failure", true, false, 4.0, &.{"unexpected failure"}),
    };
    const suites = [_]SuiteResult{.init("suite", .{ .name = "wamr", .version = "1" }, 10.0, &fixtures)};
    const report: Report = .{
        .execution = .{ .start_timestamp = "2026-01-01T00:00:00.000Z", .finish_timestamp = "2026-01-01T00:00:10.000Z" },
        .results = &suites,
    };
    const json = try report.json(std.testing.allocator);
    defer std.testing.allocator.free(json);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    const result = parsed.value.object.get("results").?.array.items[0].object;
    try std.testing.expectEqual(@as(i64, 2), result.get("passed").?.integer);
    try std.testing.expectEqual(@as(i64, 2), result.get("failed").?.integer);
    try std.testing.expectEqual(@as(i64, 1), result.get("skipped").?.integer);
    try std.testing.expectEqual(@as(i64, 1), result.get("xfailed").?.integer);
    try std.testing.expectEqual(@as(i64, 1), result.get("xpassed").?.integer);
    const items = result.get("tests").?.array.items;
    try std.testing.expectEqualStrings("xfail", items[2].object.get("outcome").?.string);
    try std.testing.expectEqualStrings("xpass", items[3].object.get("outcome").?.string);
    try std.testing.expectEqualStrings("unexpected failure", items[4].object.get("failures").?.array.items[0].string);
    try std.testing.expect(!items[1].object.get("executed").?.bool);
}
