//! Classify differences between two wasi-testsuite JSON reports.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const Object = std.json.ObjectMap;

const Key = struct { suite: []const u8, test_name: []const u8 };
const KeyContext = struct {
    pub fn hash(_: @This(), key: Key) u64 {
        var hash_state = std.hash.Wyhash.init(0);
        hash_state.update(key.suite);
        hash_state.update(&.{0});
        hash_state.update(key.test_name);
        return hash_state.final();
    }
    pub fn eql(_: @This(), a: Key, b: Key) bool {
        return std.mem.eql(u8, a.suite, b.suite) and std.mem.eql(u8, a.test_name, b.test_name);
    }
};
const Fixture = struct {
    executed: bool,
    failures: []const Value,
    fn passed(self: Fixture) bool {
        return self.executed and self.failures.len == 0;
    }
};
const Fixtures = std.HashMap(Key, Fixture, KeyContext, std.hash_map.default_max_load_percentage);
const Report = struct { fixtures: Fixtures, label: []const u8 };
const Pair = [2][]const u8;
const Documented = struct { suite: []const u8, @"test": []const u8, tracking: []const u8 };
const Stale = struct { @"test": []const u8, why: []const u8, tracking: []const u8 };
const Summary = struct {
    labels: struct { a: []const u8, b: []const u8 },
    only_in_a: []const Pair,
    only_in_b: []const Pair,
    regressions: []const Pair,
    fixture_bugs: []const Pair,
    documented: []const Documented,
    stale_skip: []const Stale,
    shared_failures: []const Pair,
    shared_skips: []const Pair,
    execution_mismatches: []const Pair,

    fn failed(self: Summary, strict: bool) bool {
        return self.regressions.len != 0 or self.stale_skip.len != 0 or
            self.execution_mismatches.len != 0 or (strict and self.fixture_bugs.len != 0);
    }
};

fn object(value: Value) !Object {
    return if (value == .object) value.object else error.ExpectedObject;
}
fn string(value: Value) ![]const u8 {
    return if (value == .string) value.string else error.ExpectedString;
}
fn array(value: Value) ![]const Value {
    return if (value == .array) value.array.items else error.ExpectedArray;
}

fn parseReport(allocator: Allocator, source: []const u8, fallback_label: []const u8) !Report {
    const root = try object(try std.json.parseFromSliceLeaky(Value, allocator, source, .{ .allocate = .alloc_always }));
    const suites = if (root.get("results")) |value| try array(value) else &.{};
    var fixtures: Fixtures = .init(allocator);
    var label = fallback_label;
    if (suites.len > 0) {
        const first = try object(suites[0]);
        if (first.get("runtime")) |value| {
            const rt = try object(value);
            const name = if (rt.get("name")) |n| if (n == .null) fallback_label else try string(n) else fallback_label;
            const version = if (rt.get("version")) |v| if (v == .null) "" else try string(v) else "";
            label = std.mem.trim(u8, try std.fmt.allocPrint(allocator, "{s} {s}", .{ if (name.len == 0) fallback_label else name, version }), " \t\r\n");
        }
    }
    for (suites) |suite_value| {
        const suite = try object(suite_value);
        const name = if (suite.get("name")) |n| try string(n) else "<unnamed-suite>";
        const tests = if (suite.get("tests")) |t| try array(t) else &.{};
        for (tests) |test_value| {
            const test_item = try object(test_value);
            const test_name = try string(test_item.get("name") orelse return error.MissingTestName);
            const executed = if (test_item.get("executed")) |v| switch (v) {
                .bool => |b| b,
                else => return error.ExpectedBoolean,
            } else true;
            const failures = if (test_item.get("failures")) |v|
                if (v == .null) &.{} else try array(v)
            else
                &.{};
            for (failures) |failure| _ = try string(failure);
            try fixtures.put(.{ .suite = name, .test_name = test_name }, .{ .executed = executed, .failures = failures });
        }
    }
    return .{ .fixtures = fixtures, .label = label };
}

fn parseSkip(allocator: Allocator, source: []const u8) !std.StringHashMap([]const u8) {
    const root = try object(try std.json.parseFromSliceLeaky(Value, allocator, source, .{ .allocate = .alloc_always }));
    var skips: std.StringHashMap([]const u8) = .init(allocator);
    var it = root.iterator();
    while (it.next()) |entry| {
        if (std.mem.startsWith(u8, entry.key_ptr.*, "_")) continue;
        try skips.put(entry.key_ptr.*, try string(entry.value_ptr.*));
    }
    return skips;
}

fn keyLess(_: void, a: Key, b: Key) bool {
    const order = std.mem.order(u8, a.suite, b.suite);
    return if (order == .eq) std.mem.lessThan(u8, a.test_name, b.test_name) else order == .lt;
}
fn sortedKeys(allocator: Allocator, fixtures: *const Fixtures) ![]Key {
    const result = try allocator.alloc(Key, fixtures.count());
    var it = fixtures.keyIterator();
    var index: usize = 0;
    while (it.next()) |key| : (index += 1) result[index] = key.*;
    std.mem.sort(Key, result, {}, keyLess);
    return result;
}
fn pair(key: Key) Pair {
    return .{ key.suite, key.test_name };
}

fn classify(allocator: Allocator, a: *const Report, b: *const Report, skips: *const std.StringHashMap([]const u8), label_a: []const u8, label_b: []const u8) !Summary {
    var only_a: std.ArrayList(Pair) = .empty;
    var only_b: std.ArrayList(Pair) = .empty;
    var regressions: std.ArrayList(Pair) = .empty;
    var fixture_bugs: std.ArrayList(Pair) = .empty;
    var documented: std.ArrayList(Documented) = .empty;
    var stale: std.ArrayList(Stale) = .empty;
    var shared_failures: std.ArrayList(Pair) = .empty;
    var shared_skips: std.ArrayList(Pair) = .empty;
    var execution_mismatches: std.ArrayList(Pair) = .empty;
    var documented_names: std.StringHashMap(void) = .init(allocator);
    const keys_a = try sortedKeys(allocator, &a.fixtures);
    const keys_b = try sortedKeys(allocator, &b.fixtures);
    for (keys_a) |key| {
        const left = a.fixtures.get(key).?;
        const right = b.fixtures.get(key) orelse {
            try only_a.append(allocator, pair(key));
            continue;
        };
        if (!left.executed or !right.executed) {
            if (!left.executed and !right.executed) {
                try shared_skips.append(allocator, pair(key));
            } else {
                try execution_mismatches.append(allocator, pair(key));
            }
        } else if (!left.passed() and !right.passed()) {
            try shared_failures.append(allocator, pair(key));
        } else if (!left.passed()) {
            try regressions.append(allocator, pair(key));
        } else if (!right.passed()) {
            if (skips.get(key.test_name)) |tracking| {
                try documented.append(allocator, .{ .suite = key.suite, .@"test" = key.test_name, .tracking = tracking });
                try documented_names.put(key.test_name, {});
            } else {
                try fixture_bugs.append(allocator, pair(key));
            }
        }
    }
    for (keys_b) |key| {
        if (!a.fixtures.contains(key)) try only_b.append(allocator, pair(key));
    }
    var names: std.ArrayList([]const u8) = .empty;
    var skip_it = skips.keyIterator();
    while (skip_it.next()) |key| try names.append(allocator, key.*);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, left: []const u8, right: []const u8) bool {
            return std.mem.lessThan(u8, left, right);
        }
    }.less);
    for (names.items) |name| {
        if (documented_names.contains(name)) continue;
        var found: ?Key = null;
        for (keys_a) |key| {
            if (std.mem.eql(u8, key.test_name, name) and b.fixtures.contains(key)) {
                found = key;
                break;
            }
        }
        const why = if (found) |key| blk: {
            if (a.fixtures.get(key).?.passed() and b.fixtures.get(key).?.passed()) {
                break :blk "both runtimes now pass — drop entry";
            }
            continue;
        } else "fixture not present in either report";
        try stale.append(allocator, .{ .@"test" = name, .why = why, .tracking = skips.get(name).? });
    }
    return .{
        .labels = .{ .a = label_a, .b = label_b },
        .only_in_a = try only_a.toOwnedSlice(allocator),
        .only_in_b = try only_b.toOwnedSlice(allocator),
        .regressions = try regressions.toOwnedSlice(allocator),
        .fixture_bugs = try fixture_bugs.toOwnedSlice(allocator),
        .documented = try documented.toOwnedSlice(allocator),
        .stale_skip = try stale.toOwnedSlice(allocator),
        .shared_failures = try shared_failures.toOwnedSlice(allocator),
        .shared_skips = try shared_skips.toOwnedSlice(allocator),
        .execution_mismatches = try execution_mismatches.toOwnedSlice(allocator),
    };
}

pub const Options = struct {
    report_a: []const u8,
    report_b: []const u8,
    label_a: ?[]const u8 = null,
    label_b: ?[]const u8 = null,
    skip: ?[]const u8 = null,
    output: ?[]const u8 = null,
    strict: bool = false,
};

fn parseOptions(args: []const []const u8) !Options {
    var options: Options = .{ .report_a = "", .report_b = "" };
    var positional: usize = 0;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const raw = args[i];
        const equals = std.mem.indexOfScalar(u8, raw, '=');
        const arg = if (equals) |index| raw[0..index] else raw;
        if (std.mem.eql(u8, arg, "--strict")) {
            if (equals != null) return error.UnknownOption;
            options.strict = true;
        } else if (std.mem.eql(u8, arg, "--label-a") or std.mem.eql(u8, arg, "--label-b") or
            std.mem.eql(u8, arg, "--parity-skip") or std.mem.eql(u8, arg, "--json"))
        {
            const value = if (equals) |index| raw[index + 1 ..] else blk: {
                i += 1;
                if (i == args.len) return error.MissingOptionValue;
                break :blk args[i];
            };
            if (std.mem.eql(u8, arg, "--label-a")) options.label_a = value else if (std.mem.eql(u8, arg, "--label-b")) options.label_b = value else if (std.mem.eql(u8, arg, "--parity-skip")) options.skip = value else options.output = value;
        } else if (std.mem.startsWith(u8, arg, "-")) {
            return error.UnknownOption;
        } else {
            if (positional == 0) options.report_a = raw else if (positional == 1) options.report_b = raw else return error.TooManyArguments;
            positional += 1;
        }
    }
    if (positional != 2) return error.MissingReport;
    return options;
}

fn printPairs(label: []const u8, entries: []const Pair, report: ?*const Report) void {
    if (entries.len == 0) return;
    std.debug.print("\n{s} ({d}):\n", .{ label, entries.len });
    for (entries) |entry| {
        std.debug.print("  {s} :: {s}", .{ entry[0], entry[1] });
        if (report) |source| {
            const failures = source.fixtures.get(.{ .suite = entry[0], .test_name = entry[1] }).?.failures;
            if (failures.len > 0) {
                const message = failures[0].string;
                const first = message[0 .. std.mem.indexOfScalar(u8, message, '\n') orelse message.len];
                std.debug.print(" — {s}{s}", .{ first, if (failures.len > 1 or first.len != message.len) " …" else "" });
            }
        }
        std.debug.print("\n", .{});
    }
}

pub fn run(init: std.process.Init, options: Options) !bool {
    const allocator = init.arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const a_source = try cwd.readFileAlloc(init.io, options.report_a, allocator, .unlimited);
    const b_source = try cwd.readFileAlloc(init.io, options.report_b, allocator, .unlimited);
    const a = try parseReport(allocator, a_source, std.fs.path.stem(options.report_a));
    const b = try parseReport(allocator, b_source, std.fs.path.stem(options.report_b));
    var skips: std.StringHashMap([]const u8) = .init(allocator);
    if (options.skip) |path| skips = try parseSkip(allocator, try cwd.readFileAlloc(init.io, path, allocator, .unlimited));
    const label_a = if (options.label_a) |label| if (label.len == 0) a.label else label else a.label;
    const label_b = if (options.label_b) |label| if (label.len == 0) b.label else label else b.label;
    const result = try classify(allocator, &a, &b, &skips, label_a, label_b);
    std.debug.print("Comparing {s} (A) vs {s} (B)\n", .{ result.labels.a, result.labels.b });
    if (result.only_in_a.len != 0 or result.only_in_b.len != 0) std.debug.print("warning: fixture sets are not identical; comparing the intersection\n", .{});
    printPairs("A only", result.only_in_a, null);
    printPairs("B only", result.only_in_b, null);
    printPairs("regressions", result.regressions, &a);
    printPairs("fixture/runtime bugs", result.fixture_bugs, &b);
    printPairs("shared failures", result.shared_failures, null);
    printPairs("shared skips", result.shared_skips, null);
    printPairs("execution mismatches", result.execution_mismatches, null);
    for (result.documented) |item| std.debug.print("documented: {s} :: {s} [{s}]\n", .{ item.suite, item.@"test", item.tracking });
    for (result.stale_skip) |item| std.debug.print("stale parity-skip: {s} [{s}] — {s}\n", .{ item.@"test", item.tracking, item.why });
    if (options.output) |path| {
        const json = try std.json.Stringify.valueAlloc(allocator, result, .{ .whitespace = .indent_2 });
        const text = try std.mem.concat(allocator, u8, &.{ json, "\n" });
        try cwd.writeFile(init.io, .{ .sub_path = path, .data = text });
    }
    if (result.failed(options.strict)) {
        std.debug.print("\n::error::Wasmtime parity diff failed: {d} regressions, {d} stale parity-skip entries, {d} execution mismatches, {d} undocumented fixture/runtime bugs{s}.\n", .{
            result.regressions.len,  result.stale_skip.len,                   result.execution_mismatches.len,
            result.fixture_bugs.len, if (options.strict) " (strict)" else "",
        });
    } else {
        std.debug.print("\nWasmtime parity diff: 0 regressions, {d} undocumented fixture/runtime-bug warning(s), {d} documented fixture/runtime-bug(s), {d} shared failure(s), {d} shared skip(s).\n", .{
            result.fixture_bugs.len, result.documented.len, result.shared_failures.len, result.shared_skips.len,
        });
    }
    return result.failed(options.strict);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try std.Io.File.stdout().writeStreamingAll(init.io, "Usage: diff-testsuite-reports REPORT_A REPORT_B [--label-a LABEL] [--label-b LABEL] [--parity-skip PATH] [--strict] [--json PATH]\n");
            return;
        }
    }
    const options = parseOptions(args[1..]) catch |err| {
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    const failed = run(init, options) catch |err| {
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (failed) std.process.exit(1);
}

test "classify parity outcomes, execution mismatches and stale entries" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const a = try parseReport(allocator,
        \\{"results":[{"name":"suite","runtime":{"name":"wamr","version":"1"},"tests":[
        \\{"name":"pass","executed":true,"failures":[]},
        \\{"name":"regression","executed":true,"failures":["broken"]},
        \\{"name":"shared","executed":true,"failures":["broken"]},
        \\{"name":"skip","executed":false,"failures":[]},
        \\{"name":"mismatch","executed":false,"failures":[]},
        \\{"name":"documented","executed":true,"failures":[]},
        \\{"name":"bug","executed":true,"failures":[]},
        \\{"name":"only_a","executed":true,"failures":[]}
        \\]}]}
    , "a");
    const b = try parseReport(allocator,
        \\{"results":[{"name":"suite","runtime":{"name":"wasmtime","version":"2"},"tests":[
        \\{"name":"pass","executed":true,"failures":[]},
        \\{"name":"regression","executed":true,"failures":[]},
        \\{"name":"shared","executed":true,"failures":["broken"]},
        \\{"name":"skip","executed":false,"failures":[]},
        \\{"name":"mismatch","executed":true,"failures":[]},
        \\{"name":"documented","executed":true,"failures":["broken"]},
        \\{"name":"bug","executed":true,"failures":["broken"]},
        \\{"name":"only_b","executed":true,"failures":[]}
        \\]}]}
    , "b");
    const skips = try parseSkip(allocator, "{\"documented\":\"issue 1\",\"pass\":\"issue 2\",\"missing\":\"issue 3\",\"_comment\":\"ignored\"}");
    const summary = try classify(allocator, &a, &b, &skips, a.label, b.label);
    try std.testing.expectEqualStrings("wamr 1", summary.labels.a);
    try std.testing.expectEqualStrings("wasmtime 2", summary.labels.b);
    try std.testing.expectEqualDeep(&[_]Pair{.{ "suite", "only_a" }}, summary.only_in_a);
    try std.testing.expectEqualDeep(&[_]Pair{.{ "suite", "only_b" }}, summary.only_in_b);
    try std.testing.expectEqualDeep(&[_]Pair{.{ "suite", "regression" }}, summary.regressions);
    try std.testing.expectEqualDeep(&[_]Pair{.{ "suite", "bug" }}, summary.fixture_bugs);
    try std.testing.expectEqualDeep(&[_]Pair{.{ "suite", "shared" }}, summary.shared_failures);
    try std.testing.expectEqualDeep(&[_]Pair{.{ "suite", "skip" }}, summary.shared_skips);
    try std.testing.expectEqualDeep(&[_]Pair{.{ "suite", "mismatch" }}, summary.execution_mismatches);
    try std.testing.expectEqualStrings("documented", summary.documented[0].@"test");
    try std.testing.expectEqualStrings("issue 1", summary.documented[0].tracking);
    try std.testing.expectEqual(@as(usize, 2), summary.stale_skip.len);
    try std.testing.expectEqualStrings("missing", summary.stale_skip[0].@"test");
    try std.testing.expectEqualStrings("pass", summary.stale_skip[1].@"test");
    try std.testing.expect(summary.failed(false));
}

test "strict classification, malformed inputs and report defaults" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const a = try parseReport(allocator, "{\"results\":[{\"name\":\"s\",\"tests\":[{\"name\":\"x\"}]}]}", "a");
    const b = try parseReport(allocator, "{\"results\":[{\"name\":\"s\",\"tests\":[{\"name\":\"x\",\"failures\":[\"failed\"]}]}]}", "b");
    const skips = try parseSkip(allocator, "{}");
    const summary = try classify(allocator, &a, &b, &skips, a.label, b.label);
    try std.testing.expect(!summary.failed(false));
    try std.testing.expect(summary.failed(true));
    try std.testing.expectError(error.ExpectedString, parseSkip(allocator, "{\"x\":1}"));
    try std.testing.expectError(error.MissingTestName, parseReport(allocator, "{\"results\":[{\"tests\":[{}]}]}", "a"));
    try std.testing.expectError(error.MissingReport, parseOptions(&.{"only-one"}));
    const options = try parseOptions(&.{ "a", "b", "--label-a=custom", "--json=result.json", "--strict" });
    try std.testing.expectEqualStrings("custom", options.label_a.?);
    try std.testing.expectEqualStrings("result.json", options.output.?);
    try std.testing.expect(options.strict);
}
