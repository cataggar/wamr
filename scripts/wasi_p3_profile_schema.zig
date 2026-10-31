//! Validation for the existing WASI P3 timing and run-document contracts.
const std = @import("std");
const Value = std.json.Value;

fn field(value: Value, name: []const u8) ?Value {
    return if (value == .object) value.object.get(name) else null;
}

fn string(value: ?Value) ?[]const u8 {
    const item = value orelse return null;
    return if (item == .string) item.string else null;
}

fn isInteger(value: ?Value) bool {
    const item = value orelse return false;
    // Python treats booleans as integers when validating the existing documents.
    return switch (item) {
        .integer, .bool => true,
        .number_string => |text| blk: {
            const digits = if (std.mem.startsWith(u8, text, "-")) text[1..] else text;
            if (digits.len == 0) break :blk false;
            for (digits) |byte| if (!std.ascii.isDigit(byte)) break :blk false;
            break :blk true;
        },
        else => false,
    };
}

fn schemaVersion(value: ?Value) bool {
    const item = value orelse return false;
    return switch (item) {
        .integer => item.integer == 1,
        .float => item.float == 1,
        .bool => item.bool,
        .number_string => (std.fmt.parseFloat(f64, item.number_string) catch return false) == 1,
        else => false,
    };
}

fn nonnegativeInteger(value: ?Value) bool {
    if (!isInteger(value)) return false;
    const item = value.?;
    return switch (item) {
        .integer => item.integer >= 0,
        .bool => true,
        .number_string => !std.mem.startsWith(u8, item.number_string, "-") or std.mem.eql(u8, item.number_string, "-0"),
        else => unreachable,
    };
}

fn equals(value: ?Value, expected: []const u8) bool {
    return std.mem.eql(u8, string(value) orelse return false, expected);
}

fn oneOf(value: ?Value, choices: []const []const u8) bool {
    for (choices) |choice| if (equals(value, choice)) return true;
    return false;
}

pub fn validateEvent(event: Value) !void {
    if (event != .object) return error.InvalidEventObject;
    if (!schemaVersion(field(event, "schema_version"))) return error.InvalidEventSchema;
    if (!equals(field(event, "event"), "phase_timing")) return error.InvalidEventKind;
    if (!oneOf(field(event, "mode"), &.{ "aot", "jit" })) return error.InvalidEventMode;
    const fixture = string(field(event, "fixture")) orelse return error.InvalidEventFixture;
    if (fixture.len == 0) return error.InvalidEventFixture;
    if (!oneOf(field(event, "phase"), &.{ "component_precompile", "core_precompile", "fixture_execution" }))
        return error.InvalidEventPhase;
    if (!oneOf(field(event, "artifact_kind"), &.{ "component", "core" })) return error.InvalidArtifactKind;
    if (!oneOf(field(event, "cache"), &.{ "miss", "hit", "bypass", "n/a" })) return error.InvalidEventCache;
    if (!nonnegativeInteger(field(event, "duration_ns"))) return error.InvalidEventDuration;
}

pub fn validateRun(document: Value) !void {
    if (document != .object) return error.InvalidRunObject;
    if (!schemaVersion(field(document, "schema_version"))) return error.InvalidRunSchema;
    if (!equals(field(document, "kind"), "wasi-p3-profile-runs")) return error.InvalidRunKind;
    const metadata = field(document, "metadata") orelse return error.InvalidRunMetadata;
    if (metadata != .object) return error.InvalidRunMetadata;
    if (!oneOf(field(metadata, "mode"), &.{ "aot", "jit" })) return error.InvalidRunMode;
    if (string(field(metadata, "platform_id")) == null or string(field(metadata, "commit")) == null)
        return error.InvalidRunMetadata;
    const samples = field(document, "samples") orelse return error.InvalidRunSamples;
    if (samples != .array) return error.InvalidRunSamples;
    for (samples.array.items) |sample| {
        if (sample != .object) return error.InvalidSampleObject;
        if (!oneOf(field(sample, "temperature"), &.{ "cold", "warm" })) return error.InvalidSampleTemperature;
        if (!isInteger(field(sample, "index")) or !isInteger(field(sample, "suite_duration_ns")))
            return error.InvalidSampleInteger;
        const valid = field(sample, "valid") orelse return error.InvalidSampleValidity;
        if (valid != .bool) return error.InvalidSampleValidity;
        const counts = field(sample, "counts") orelse return error.InvalidSampleCounts;
        if (counts != .object) return error.InvalidSampleCounts;
        for ([_][]const u8{ "fixtures", "executed", "passed", "failed" }) |name|
            if (!isInteger(field(counts, name))) return error.InvalidSampleCounts;
        const events = field(sample, "events") orelse return error.InvalidSampleEvents;
        if (events != .array) return error.InvalidSampleEvents;
        for (events.array.items) |event| try validateEvent(event);
    }
}

pub fn truthy(value: ?Value) !bool {
    const item = value orelse return false;
    return switch (item) {
        .null => false,
        .bool => item.bool,
        .integer => item.integer != 0,
        .float => item.float != 0,
        .number_string => (try std.fmt.parseFloat(f64, item.number_string)) != 0,
        .string => item.string.len != 0,
        .array => item.array.items.len != 0,
        .object => item.object.count() != 0,
    };
}

fn integerEqual(value: ?Value, expected: i64) bool {
    const item = value orelse return false;
    return switch (item) {
        .integer => item.integer == expected,
        .bool => @as(i64, if (item.bool) 1 else 0) == expected,
        .number_string => (std.fmt.parseInt(i64, item.number_string, 10) catch return false) == expected,
        else => false,
    };
}

/// Validate the four matched platform/mode runs required by aggregation.
pub fn validatePlan(documents: []const Value) ![]const u8 {
    var seen: [4]bool = @splat(false);
    var commit: ?[]const u8 = null;
    for (documents) |document| {
        try validateRun(document);
        const metadata = field(document, "metadata").?;
        const platform = string(field(metadata, "platform_id")).?;
        const mode = string(field(metadata, "mode")).?;
        const platform_index: usize = if (std.mem.eql(u8, platform, "linux-arm64"))
            0
        else if (std.mem.eql(u8, platform, "macos-arm64"))
            1
        else
            return error.InvalidPlanPlatform;
        const mode_index: usize = if (std.mem.eql(u8, mode, "aot")) 0 else 1;
        if (!equals(field(metadata, "optimize"), "ReleaseSafe")) return error.InvalidPlanOptimize;
        const host = field(metadata, "host") orelse return error.InvalidPlanHost;
        const tools = field(metadata, "tools") orelse return error.InvalidPlanTools;
        const cache = field(metadata, "cache") orelse return error.InvalidPlanCache;
        if (host != .object) return error.InvalidPlanHost;
        if (tools != .object) return error.InvalidPlanTools;
        if (cache != .object) return error.InvalidPlanCache;
        const machine = string(field(host, "machine")) orelse return error.InvalidPlanHost;
        if (!std.ascii.eqlIgnoreCase(machine, "arm64") and !std.ascii.eqlIgnoreCase(machine, "aarch64"))
            return error.InvalidPlanHost;
        if (!equals(field(host, "system"), if (platform_index == 0) "Linux" else "Darwin"))
            return error.InvalidPlanHost;
        for ([_][]const u8{ "zig", "wamr", "wamrc" }) |name|
            if (!try truthy(field(tools, name))) return error.InvalidPlanTools;
        const cell = platform_index * 2 + mode_index;
        if (seen[cell]) return error.DuplicatePlanCell;
        seen[cell] = true;
        const revision = string(field(metadata, "commit")).?;
        if (commit) |previous| {
            if (!std.mem.eql(u8, previous, revision)) return error.PlanRevisionMismatch;
        } else commit = revision;
        var counts: [2]usize = @splat(0);
        for (field(document, "samples").?.array.items) |sample| {
            const temperature = string(field(sample, "temperature")).?;
            counts[if (std.mem.eql(u8, temperature, "cold")) 0 else 1] += 1;
            if (!field(sample, "valid").?.bool) return error.InvalidPlanSample;
            if (field(sample, "id") == null) return error.MissingSampleId;
            const sample_counts = field(sample, "counts").?;
            if (sample_counts.object.count() != 4 or
                !integerEqual(field(sample_counts, "fixtures"), 41) or
                !integerEqual(field(sample_counts, "executed"), 41) or
                !integerEqual(field(sample_counts, "passed"), 41) or
                !integerEqual(field(sample_counts, "failed"), 0))
                return error.InvalidPlanSample;
        }
        if (counts[0] != 5 or counts[1] != 10) return error.InvalidPlanSampleCount;
    }
    for (seen) |present| if (!present) return error.IncompletePlan;
    return commit orelse return error.IncompletePlan;
}

test "run documents validate every timing event" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const json =
        \\{"schema_version":1,"kind":"wasi-p3-profile-runs",
        \\ "metadata":{"mode":"aot","platform_id":"linux-arm64","commit":"abc"},
        \\ "samples":[{"temperature":"cold","index":1,"valid":true,"suite_duration_ns":3,
        \\ "counts":{"fixtures":41,"executed":41,"passed":41,"failed":0},
        \\ "events":[{"schema_version":1,"event":"phase_timing","mode":"aot",
        \\ "fixture":"noop","phase":"fixture_execution","artifact_kind":"core",
        \\ "cache":"n/a","duration_ns":3}]}]}
    ;
    const document = try std.json.parseFromSliceLeaky(Value, allocator, json, .{});
    try validateRun(document);
    const samples = document.object.get("samples").?.array.items;
    const event = &samples[0].object.getPtr("events").?.array.items[0];
    event.object.getPtr("schema_version").?.* = .{ .float = 1.0 };
    event.object.getPtr("duration_ns").?.* = .{ .number_string = "18446744073709551616" };
    try validateRun(document);
    event.object.getPtr("phase").?.* = .{ .string = "combined" };
    try std.testing.expectError(error.InvalidEventPhase, validateRun(document));
    event.object.getPtr("phase").?.* = .{ .string = "fixture_execution" };
    event.object.getPtr("duration_ns").?.* = .{ .integer = -1 };
    try std.testing.expectError(error.InvalidEventDuration, validateRun(document));
}

test "run sample validity and counts cannot be omitted or mistyped" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const json =
        \\{"schema_version":1,"kind":"wasi-p3-profile-runs",
        \\ "metadata":{"mode":"jit","platform_id":"macos-arm64","commit":"abc"},
        \\ "samples":[{"temperature":"warm","index":1,"valid":true,"suite_duration_ns":3,
        \\ "counts":{"fixtures":41,"executed":41,"passed":41,"failed":0},"events":[]}]}
    ;
    const document = try std.json.parseFromSliceLeaky(Value, allocator, json, .{});
    try validateRun(document);
    const sample = &document.object.get("samples").?.array.items[0];
    sample.object.getPtr("valid").?.* = .{ .integer = 1 };
    try std.testing.expectError(error.InvalidSampleValidity, validateRun(document));
    sample.object.getPtr("valid").?.* = .{ .bool = true };
    const counts = sample.object.getPtr("counts").?;
    counts.object.getPtr("failed").?.* = .{ .string = "zero" };
    try std.testing.expectError(error.InvalidSampleCounts, validateRun(document));
}

test "aggregation requires all four matched complete and valid cells" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var documents: [4]Value = undefined;
    for (0..4) |index| {
        const platform = if (index < 2) "linux-arm64" else "macos-arm64";
        const system = if (index < 2) "Linux" else "Darwin";
        const mode = if (index % 2 == 0) "aot" else "jit";
        const json = try std.fmt.allocPrint(allocator,
            \\{{"schema_version":1,"kind":"wasi-p3-profile-runs",
            \\ "metadata":{{"mode":"{s}","platform_id":"{s}","commit":"abc",
            \\ "optimize":"ReleaseSafe","host":{{"machine":"arm64","system":"{s}"}},
            \\ "tools":{{"zig":"z","wamr":"w","wamrc":"c"}},"cache":{{}}}},"samples":[]}}
        , .{ mode, platform, system });
        documents[index] = try std.json.parseFromSliceLeaky(Value, allocator, json, .{});
        const samples = &documents[index].object.getPtr("samples").?.array;
        for (0..15) |sample_index| {
            const sample_json = try std.fmt.allocPrint(allocator,
                \\{{"id":"sample","temperature":"{s}","index":1,"valid":true,"suite_duration_ns":3,
                \\ "counts":{{"fixtures":41,"executed":41,"passed":41,"failed":0}},"events":[]}}
            , .{if (sample_index < 5) "cold" else "warm"});
            try samples.append(try std.json.parseFromSliceLeaky(Value, allocator, sample_json, .{}));
        }
    }
    try std.testing.expectEqualStrings("abc", try validatePlan(&documents));
    try std.testing.expectError(error.IncompletePlan, validatePlan(documents[0..3]));
    const duplicate = [_]Value{ documents[0], documents[0], documents[2], documents[3] };
    try std.testing.expectError(error.DuplicatePlanCell, validatePlan(&duplicate));
    const metadata = documents[3].object.getPtr("metadata").?;
    metadata.object.getPtr("commit").?.* = .{ .string = "different" };
    try std.testing.expectError(error.PlanRevisionMismatch, validatePlan(&documents));
    metadata.object.getPtr("commit").?.* = .{ .string = "abc" };
    const samples = documents[0].object.getPtr("samples").?;
    _ = samples.array.items[0].object.swapRemove("id");
    try std.testing.expectError(error.MissingSampleId, validatePlan(&documents));
    try samples.array.items[0].object.put(allocator, "id", .null);
    try std.testing.expectEqualStrings("abc", try validatePlan(&documents));
    samples.array.items[0].object.getPtr("valid").?.* = .{ .bool = false };
    try std.testing.expectError(error.InvalidPlanSample, validatePlan(&documents));
}
