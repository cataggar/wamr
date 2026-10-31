//! Sample validation for the experimental native WASI P3 profile collector.
const std = @import("std");
const schema = @import("wasi_p3_profile_schema.zig");
const text_lines = @import("text_lines.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Counts = struct {
    fixtures: usize,
    executed: usize,
    passed: usize,
    failed: usize,
};

pub const expected_fixtures: usize = 41;
pub const Mode = enum { aot, jit };
pub const Temperature = enum { cold, warm };

/// Use an arena; parsed values may borrow bytes. Diagnostics identify invalid lines.
pub fn parseEvents(allocator: Allocator, path: []const u8, bytes: []const u8) ![]const Value {
    if (!std.unicode.utf8ValidateSlice(bytes)) return error.InvalidEventEncoding;
    var events: std.ArrayList(Value) = .empty;
    var offset: usize = 0;
    var line_number: usize = 0;
    while (offset < bytes.len) {
        line_number += 1;
        const end = std.mem.indexOfAnyPos(u8, bytes, offset, "\r\n") orelse bytes.len;
        const line = bytes[offset..end];
        offset = end;
        if (offset < bytes.len) {
            offset += 1;
            if (bytes[end] == '\r' and offset < bytes.len and bytes[offset] == '\n') offset += 1;
        }
        if ((try text_lines.strip(line)).len == 0) continue;
        const event = std.json.parseFromSliceLeaky(Value, allocator, line, .{ .duplicate_field_behavior = .use_last }) catch |err| {
            std.debug.print("{s}:{d}: {s}\n", .{ path, line_number, @errorName(err) });
            return err;
        };
        schema.validateEvent(event) catch |err| {
            std.debug.print("{s}:{d}: {s}\n", .{ path, line_number, @errorName(err) });
            return err;
        };
        try events.append(allocator, event);
    }
    return events.toOwnedSlice(allocator);
}

pub fn reportCounts(document: Value) !Counts {
    if (document != .object) return error.InvalidSuiteReport;
    const suites = document.object.get("results") orelse return error.InvalidSuiteReport;
    if (suites != .array or suites.array.items.len != 1) return error.InvalidSuiteReport;
    const suite = suites.array.items[0];
    if (suite != .object) return error.InvalidSuiteReport;
    const tests = suite.object.get("tests") orelse return .{ .fixtures = 0, .executed = 0, .passed = 0, .failed = 0 };
    if (tests != .array) return error.InvalidSuiteReport;
    var counts: Counts = .{ .fixtures = tests.array.items.len, .executed = 0, .passed = 0, .failed = 0 };
    for (tests.array.items) |test_result| {
        if (test_result != .object) return error.InvalidSuiteReport;
        if (try schema.truthy(test_result.object.get("executed"))) {
            counts.executed += 1;
            if (!try schema.truthy(test_result.object.get("failures"))) counts.passed += 1;
        }
    }
    counts.failed = counts.executed - counts.passed;
    return counts;
}

/// Use an arena for the returned diagnostics and their strings.
pub fn sampleErrors(
    allocator: Allocator,
    mode: Mode,
    temperature: Temperature,
    exit_code: i64,
    counts: Counts,
    events: []const Value,
) ![]const []const u8 {
    var errors: std.ArrayList([]const u8) = .empty;
    if (exit_code != 0)
        try errors.append(allocator, try std.fmt.allocPrint(allocator, "runner exit {d}", .{exit_code}));
    if (counts.fixtures != 41 or counts.executed != 41 or counts.passed != 41 or counts.failed != 0) {
        try errors.append(allocator, try std.fmt.allocPrint(
            allocator,
            "conformance counts {{'fixtures': {d}, 'executed': {d}, 'passed': {d}, 'failed': {d}}}, expected 41/41",
            .{ counts.fixtures, counts.executed, counts.passed, counts.failed },
        ));
    }

    var fixtures: std.StringHashMap(void) = .init(allocator);
    defer fixtures.deinit();
    var executions: usize = 0;
    var precompiles: usize = 0;
    var wrong_cache: usize = 0;
    const expected_cache: []const u8 = if (mode == .jit) "bypass" else if (temperature == .cold) "miss" else "hit";
    for (events) |event| {
        try schema.validateEvent(event);
        const phase = event.object.get("phase").?.string;
        if (std.mem.eql(u8, phase, "fixture_execution")) {
            executions += 1;
            try fixtures.put(event.object.get("fixture").?.string, {});
        } else if (std.mem.endsWith(u8, phase, "_precompile")) {
            precompiles += 1;
            if (!std.mem.eql(u8, event.object.get("cache").?.string, expected_cache))
                wrong_cache += 1;
        }
    }
    if (executions != 41)
        try errors.append(allocator, try std.fmt.allocPrint(allocator, "execution events={d}, expected 41", .{executions}));
    if (precompiles != 41)
        try errors.append(allocator, try std.fmt.allocPrint(allocator, "precompile events={d}, expected 41", .{precompiles}));
    if (fixtures.count() != 41)
        try errors.append(allocator, "execution fixture names are not unique");
    if (wrong_cache != 0)
        try errors.append(allocator, try std.fmt.allocPrint(allocator, "{d} precompile events did not use cache={s}", .{ wrong_cache, expected_cache }));
    return errors.toOwnedSlice(allocator);
}

test "timing JSONL validates events, universal newlines, and blank lines" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const input =
        " \t\r\n\xc2\xa0\r" ++
        "{\"schema_version\":0,\"schema_version\":1,\"event\":\"phase_timing\",\"mode\":\"aot\"," ++
        "\"fixture\":\"noop\",\"phase\":\"fixture_execution\",\"artifact_kind\":\"core\",\"cache\":\"n/a\",\"duration_ns\":1}\n";
    const events = try parseEvents(allocator, "timings.jsonl", input);
    try std.testing.expectEqual(@as(usize, 1), events.len);
    try std.testing.expectEqualStrings("noop", events[0].object.get("fixture").?.string);
    try std.testing.expectError(error.InvalidEventObject, parseEvents(allocator, "timings.jsonl", "\n[]\n"));
    try std.testing.expectError(error.InvalidEventEncoding, parseEvents(allocator, "timings.jsonl", "\xff"));
}

test "suite report counts use executed and failure truthiness" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const document = try std.json.parseFromSliceLeaky(Value, allocator,
        \\{"results":[{"tests":[{"executed":true},{"executed":1,"failures":["failure"]},{"executed":false},{"executed":"yes","failures":[]}]}]}
    , .{});
    try std.testing.expectEqualDeep(Counts{ .fixtures = 4, .executed = 3, .passed = 2, .failed = 1 }, try reportCounts(document));
    const empty = try std.json.parseFromSliceLeaky(Value, allocator, "{\"results\":[{}]}", .{});
    try std.testing.expectEqualDeep(Counts{ .fixtures = 0, .executed = 0, .passed = 0, .failed = 0 }, try reportCounts(empty));
    const multiple = try std.json.parseFromSliceLeaky(Value, allocator, "{\"results\":[{},{}]}", .{});
    try std.testing.expectError(error.InvalidSuiteReport, reportCounts(multiple));
}

test "sample validity checks every fixture, event, and cache state" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const counts: Counts = .{ .fixtures = 41, .executed = 41, .passed = 41, .failed = 0 };
    var events: [82]Value = undefined;
    for (0..41) |index| for (0..2) |kind| {
        const json = try std.fmt.allocPrint(allocator,
            \\{{"schema_version":1,"event":"phase_timing","mode":"aot","fixture":"fixture-{d}","phase":"{s}","artifact_kind":"core","cache":"{s}","duration_ns":1}}
        , .{ index, if (kind == 0) "fixture_execution" else "core_precompile", if (kind == 0) "n/a" else "miss" });
        events[index * 2 + kind] = try std.json.parseFromSliceLeaky(Value, allocator, json, .{});
    };
    try std.testing.expectEqual(@as(usize, 0), (try sampleErrors(allocator, .aot, .cold, 0, counts, &events)).len);
    const warm = try sampleErrors(allocator, .aot, .warm, 0, counts, &events);
    try std.testing.expectEqualStrings("41 precompile events did not use cache=hit", warm[0]);
    const jit = try sampleErrors(allocator, .jit, .warm, 0, counts, &events);
    try std.testing.expectEqualStrings("41 precompile events did not use cache=bypass", jit[0]);
    events[2].object.getPtr("fixture").?.* = events[0].object.get("fixture").?;
    const failures = try sampleErrors(allocator, .aot, .cold, 7, .{ .fixtures = 41, .executed = 40, .passed = 39, .failed = 1 }, events[0..80]);
    try std.testing.expectEqual(@as(usize, 5), failures.len);
    try std.testing.expectEqualStrings("runner exit 7", failures[0]);
    try std.testing.expectEqualStrings("conformance counts {'fixtures': 41, 'executed': 40, 'passed': 39, 'failed': 1}, expected 41/41", failures[1]);
    try std.testing.expectEqualStrings("execution events=40, expected 41", failures[2]);
    try std.testing.expectEqualStrings("precompile events=40, expected 41", failures[3]);
    try std.testing.expectEqualStrings("execution fixture names are not unique", failures[4]);
    events[1].object.getPtr("cache").?.* = .{ .string = "invalid" };
    try std.testing.expectError(error.InvalidEventCache, sampleErrors(allocator, .aot, .cold, 0, counts, &events));
}
