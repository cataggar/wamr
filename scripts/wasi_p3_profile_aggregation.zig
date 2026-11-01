//! Aggregate matched WASI P3 profile runs without changing their report schema.
const std = @import("std");
const schema = @import("wasi_p3_profile_schema.zig");
const stats = @import("wasi_p3_profile_stats.zig");
const integer_median = @import("integer_median.zig");
const fixed = @import("decimal_format.zig").fixed;
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

const Key = struct {
    platform: []const u8,
    mode: []const u8,
    temperature: []const u8,
    phase: []const u8 = "",
    fixture: []const u8 = "",

    fn equal(a: Key, b: Key) bool {
        return std.mem.eql(u8, a.platform, b.platform) and std.mem.eql(u8, a.mode, b.mode) and
            std.mem.eql(u8, a.temperature, b.temperature) and std.mem.eql(u8, a.phase, b.phase) and
            std.mem.eql(u8, a.fixture, b.fixture);
    }

    fn less(_: void, a: Group, b: Group) bool {
        for ([_][2][]const u8{
            .{ a.key.platform, b.key.platform },       .{ a.key.mode, b.key.mode },
            .{ a.key.temperature, b.key.temperature }, .{ a.key.phase, b.key.phase },
            .{ a.key.fixture, b.key.fixture },
        }) |pair| switch (std.mem.order(u8, pair[0], pair[1])) {
            .lt => return true,
            .gt => return false,
            .eq => {},
        };
        return false;
    }
};

const Group = struct {
    key: Key,
    values: std.ArrayList(i128) = .empty,
    indices: std.AutoHashMap(i64, i128),
};

const SuiteRow = struct {
    platform_id: []const u8,
    mode: []const u8,
    temperature: []const u8,
    samples: usize,
    min_ns: i128,
    median_ns: i128,
    p95_ns: i128,
    max_ns: i128,
};

const PhaseRow = struct {
    platform_id: []const u8,
    mode: []const u8,
    temperature: []const u8,
    phase: []const u8,
    fixture: []const u8,
    samples: usize,
    min_ns: i128,
    median_ns: i128,
    p95_ns: i128,
    max_ns: i128,
};

pub const Profile = struct {
    mode: []const u8,
    fixture: []const u8,
    phase: []const u8,
    reasons: []const []const u8,
};

pub const Analysis = struct {
    commit: []const u8,
    suite: []const SuiteRow,
    phases: []const PhaseRow,
    profiles: []const Profile,
};

fn integer(value: Value) !i128 {
    return switch (value) {
        .integer => value.integer,
        .bool => if (value.bool) 1 else 0,
        .number_string => try std.fmt.parseInt(i128, value.number_string, 10),
        else => error.InvalidProfileInteger,
    };
}

fn find(groups: []Group, key: Key) ?*Group {
    for (groups) |*group| if (group.key.equal(key)) return group;
    return null;
}

fn pairedIndex(value: Value) !?i64 {
    const index = integer(value) catch |err| switch (err) {
        error.Overflow => return null,
        else => return err,
    };
    if (index < 1 or index > 10) return null;
    return @intCast(index);
}

fn add(allocator: Allocator, groups: *std.ArrayList(Group), key: Key, index: ?i64, value: i128) !void {
    const group = find(groups.items, key) orelse blk: {
        try groups.append(allocator, .{ .key = key, .indices = std.AutoHashMap(i64, i128).init(allocator) });
        break :blk &groups.items[groups.items.len - 1];
    };
    try group.values.append(allocator, value);
    if (index) |number| try group.indices.put(number, value);
}

const Median = union(enum) {
    integer: i128,
    float: f64,

    fn asFloat(value: Median) f64 {
        return switch (value) {
            .integer => |n| @floatFromInt(n),
            .float => |n| n,
        };
    }

    fn divide(numerator: Median, denominator: Median) f64 {
        if (numerator == .integer and denominator == .integer) {
            const result = @as(f128, @floatFromInt(numerator.integer)) /
                @as(f128, @floatFromInt(denominator.integer));
            return @floatCast(result);
        }
        return numerator.asFloat() / denominator.asFloat();
    }
};

fn rawMedian(allocator: Allocator, values: []const i128) !Median {
    if (values.len == 0) return error.EmptySamples;
    const ordered = try allocator.dupe(i128, values);
    defer allocator.free(ordered);
    std.mem.sort(i128, ordered, {}, std.sort.asc(i128));
    const middle = ordered.len / 2;
    if (ordered.len % 2 == 1) return .{ .integer = ordered[middle] };
    return .{ .float = integer_median.average(ordered[middle - 1], ordered[middle]) };
}

const Selection = struct {
    mode: []const u8,
    fixture: []const u8,
    phase: []const u8,
    reasons: std.ArrayList([]const u8) = .empty,

    fn less(_: void, a: Selection, b: Selection) bool {
        for ([_][2][]const u8{ .{ a.mode, b.mode }, .{ a.fixture, b.fixture }, .{ a.phase, b.phase } }) |pair|
            switch (std.mem.order(u8, pair[0], pair[1])) {
                .lt => return true,
                .gt => return false,
                .eq => {},
            };
        return false;
    }
};

fn addReason(allocator: Allocator, selected: *std.ArrayList(Selection), mode: []const u8, fixture: []const u8, phase: []const u8, reason: []const u8) !void {
    const entry = for (selected.items) |*item| {
        if (std.mem.eql(u8, item.mode, mode) and std.mem.eql(u8, item.fixture, fixture) and
            std.mem.eql(u8, item.phase, phase)) break item;
    } else blk: {
        try selected.append(allocator, .{ .mode = mode, .fixture = fixture, .phase = phase });
        break :blk &selected.items[selected.items.len - 1];
    };
    for (entry.reasons.items) |existing| if (std.mem.eql(u8, existing, reason)) return;
    try entry.reasons.append(allocator, reason);
}

fn reasonLess(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn pick(allocator: Allocator, suites: []Group, phases: []Group) ![]const Profile {
    var selected: std.ArrayList(Selection) = .empty;
    for ([_][]const u8{ "aot", "jit" }) |mode|
        try addReason(allocator, &selected, mode, "http-fields", "fixture_execution", "required http-fields baseline");
    for (phases) |group| {
        if (!std.mem.eql(u8, group.key.platform, "macos-arm64")) continue;
        const suite = find(suites, .{ .platform = group.key.platform, .mode = group.key.mode, .temperature = group.key.temperature }) orelse return error.MissingSuiteGroup;
        const suite_median = (try stats.summarize(allocator, suite.values.items)).median_ns;
        const median = try rawMedian(allocator, group.values.items);
        const share = if (suite_median == 0) 0 else Median.divide(median, .{ .integer = suite_median });
        if (share >= 0.20) {
            const percent = try fixed(allocator, share * 100, 1);
            try addReason(allocator, &selected, group.key.mode, group.key.fixture, group.key.phase, try std.fmt.allocPrint(allocator, "{s} median is {s}% of suite wall time", .{ group.key.temperature, percent }));
        }
        // Process each mode/phase/fixture once; candidates can appear in either temperature.
        if (!std.mem.eql(u8, group.key.temperature, "cold")) continue;
        var matches: usize = 0;
        var total: usize = 0;
        var both_temperatures = true;
        var medians_at_least_two = true;
        for ([_][]const u8{ "cold", "warm" }, [_]i64{ 5, 10 }) |temperature, count| {
            var mac_key = group.key;
            mac_key.temperature = temperature;
            var linux_key = mac_key;
            linux_key.platform = "linux-arm64";
            const mac = find(phases, mac_key);
            const linux = find(phases, linux_key);
            var in_cell: usize = 0;
            if (mac != null and linux != null) {
                var index: i64 = 1;
                while (index <= count) : (index += 1) {
                    const mac_value = mac.?.indices.get(index) orelse continue;
                    const linux_value = linux.?.indices.get(index) orelse continue;
                    if (linux_value <= 0) continue;
                    const ratio = Median.divide(.{ .integer = mac_value }, .{ .integer = linux_value });
                    if (ratio >= 2.0) matches += 1;
                    total += 1;
                    in_cell += 1;
                }
            }
            if (in_cell == 0) {
                both_temperatures = false;
                continue;
            }
            const linux_median = try rawMedian(allocator, linux.?.values.items);
            if (linux_median.asFloat() <= 0 or Median.divide(try rawMedian(allocator, mac.?.values.items), linux_median) < 2.0)
                medians_at_least_two = false;
        }
        if (both_temperatures and medians_at_least_two and total != 0) {
            const fraction = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(total));
            if (fraction >= 0.80) {
                const percent = try fixed(allocator, fraction * 100, 0);
                try addReason(allocator, &selected, group.key.mode, group.key.fixture, group.key.phase, try std.fmt.allocPrint(allocator, "macOS/Linux median >=2x in cold and warm cells ({s}% paired samples >=2x)", .{percent}));
            }
        }
    }
    std.mem.sort(Selection, selected.items, {}, Selection.less);
    const profiles = try allocator.alloc(Profile, selected.items.len);
    for (selected.items, profiles) |*entry, *profile| {
        std.mem.sort([]const u8, entry.reasons.items, {}, reasonLess);
        profile.* = .{ .mode = entry.mode, .fixture = entry.fixture, .phase = entry.phase, .reasons = entry.reasons.items };
    }
    return profiles;
}

pub fn analyze(allocator: Allocator, documents: []const Value) !Analysis {
    const commit = try schema.validatePlan(documents);
    var suites: std.ArrayList(Group) = .empty;
    var phases: std.ArrayList(Group) = .empty;
    for (documents) |document| {
        const metadata = document.object.get("metadata").?.object;
        const platform = metadata.get("platform_id").?.string;
        const mode = metadata.get("mode").?.string;
        for (document.object.get("samples").?.array.items) |sample| {
            const object = sample.object;
            const temperature = object.get("temperature").?.string;
            const index = try pairedIndex(object.get("index").?);
            const key: Key = .{ .platform = platform, .mode = mode, .temperature = temperature };
            try add(allocator, &suites, key, index, try integer(object.get("suite_duration_ns").?));
            for (object.get("events").?.array.items) |event| {
                var phase_key = key;
                phase_key.phase = event.object.get("phase").?.string;
                phase_key.fixture = event.object.get("fixture").?.string;
                try add(allocator, &phases, phase_key, index, try integer(event.object.get("duration_ns").?));
            }
        }
    }
    if (phases.items.len == 0) return error.EmptyProfilePhases;
    std.mem.sort(Group, suites.items, {}, Key.less);
    std.mem.sort(Group, phases.items, {}, Key.less);
    const suite_rows = try allocator.alloc(SuiteRow, suites.items.len);
    for (suites.items, suite_rows) |group, *row| {
        const summary = try stats.summarize(allocator, group.values.items);
        row.* = .{ .platform_id = group.key.platform, .mode = group.key.mode, .temperature = group.key.temperature, .samples = summary.samples, .min_ns = summary.min_ns, .median_ns = summary.median_ns, .p95_ns = summary.p95_ns, .max_ns = summary.max_ns };
    }
    const phase_rows = try allocator.alloc(PhaseRow, phases.items.len);
    for (phases.items, phase_rows) |group, *row| {
        const summary = try stats.summarize(allocator, group.values.items);
        row.* = .{ .platform_id = group.key.platform, .mode = group.key.mode, .temperature = group.key.temperature, .phase = group.key.phase, .fixture = group.key.fixture, .samples = summary.samples, .min_ns = summary.min_ns, .median_ns = summary.median_ns, .p95_ns = summary.p95_ns, .max_ns = summary.max_ns };
    }
    return .{ .commit = commit, .suite = suite_rows, .phases = phase_rows, .profiles = try pick(allocator, suites.items, phases.items) };
}

test "raw medians retain representable extreme inputs" {
    const allocator = std.testing.allocator;
    const maximum = std.math.maxInt(i128);
    const minimum = std.math.minInt(i128);
    try std.testing.expectEqual(@as(f64, 0x1p127), (try rawMedian(allocator, &.{ maximum, maximum })).float);
    try std.testing.expectEqual(@as(f64, -0x1p127), (try rawMedian(allocator, &.{ minimum, minimum })).float);
    try std.testing.expectEqual(@as(f64, -0.5), (try rawMedian(allocator, &.{ minimum, maximum })).float);
}

test "integer ratios retain precision at selection thresholds" {
    const numerator = Median{ .integer = 18_014_398_509_481_984 };
    const denominator = Median{ .integer = 9_007_199_254_740_993 };
    try std.testing.expect(Median.divide(numerator, denominator) < 2.0);
    try std.testing.expectEqual(@as(f64, 2.0), numerator.asFloat() / denominator.asFloat());
}

test "unpaired indices outside the sample range are retained only in statistics" {
    const index = Value{ .number_string = "10000000000000000000000000000000000000000" };
    try std.testing.expectEqual(@as(?i64, null), try pairedIndex(index));
    try std.testing.expectEqual(@as(?i64, 1), try pairedIndex(.{ .bool = true }));
}

test "aggregation groups timings and selects paired platform regressions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var documents: [4]Value = undefined;
    for (0..4) |cell| {
        const platform = if (cell < 2) "linux-arm64" else "macos-arm64";
        const system = if (cell < 2) "Linux" else "Darwin";
        const mode = if (cell % 2 == 0) "aot" else "jit";
        const scale: u32 = if (cell < 2) 1 else 3;
        const json = try std.fmt.allocPrint(allocator,
            \\{{"schema_version":1,"kind":"wasi-p3-profile-runs",
            \\ "metadata":{{"mode":"{s}","platform_id":"{s}","commit":"abc",
            \\ "optimize":"ReleaseSafe","host":{{"machine":"arm64","system":"{s}"}},
            \\ "tools":{{"zig":"z","wamr":"w","wamrc":"c"}},"cache":{{}}}},"samples":[]}}
        , .{ mode, platform, system });
        documents[cell] = try std.json.parseFromSliceLeaky(Value, allocator, json, .{});
        const samples = &documents[cell].object.getPtr("samples").?.array;
        for (0..15) |sample_index| {
            const sample_json = try std.fmt.allocPrint(allocator,
                \\{{"id":"sample","temperature":"{s}","index":{d},"valid":true,"suite_duration_ns":{d},
                \\ "counts":{{"fixtures":41,"executed":41,"passed":41,"failed":0}},
                \\ "events":[{{"schema_version":1,"event":"phase_timing","mode":"{s}",
                \\ "fixture":"http-fields","phase":"fixture_execution","artifact_kind":"component",
                \\ "cache":"n/a","duration_ns":{d}}}]}}
            , .{ if (sample_index < 5) "cold" else "warm", if (sample_index < 5) sample_index + 1 else sample_index - 4, 1000 * scale, mode, 300 * scale });
            try samples.append(try std.json.parseFromSliceLeaky(Value, allocator, sample_json, .{}));
        }
    }
    const result = try analyze(allocator, &documents);
    try std.testing.expectEqualStrings("abc", result.commit);
    try std.testing.expectEqual(@as(usize, 8), result.suite.len);
    try std.testing.expectEqual(@as(usize, 8), result.phases.len);
    try std.testing.expectEqual(@as(usize, 2), result.profiles.len);
    try std.testing.expectEqualStrings("aot", result.profiles[0].mode);
    try std.testing.expectEqualStrings("http-fields", result.profiles[0].fixture);
    const reasons = result.profiles[0].reasons;
    try std.testing.expectEqual(@as(usize, 4), reasons.len);
    try std.testing.expectEqualStrings("cold median is 30.0% of suite wall time", reasons[0]);
    try std.testing.expectEqualStrings("macOS/Linux median >=2x in cold and warm cells (100% paired samples >=2x)", reasons[1]);
    try std.testing.expectEqualStrings("required http-fields baseline", reasons[2]);
    try std.testing.expectEqualStrings("warm median is 30.0% of suite wall time", reasons[3]);
}
