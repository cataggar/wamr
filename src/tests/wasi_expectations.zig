//! WASI expectation-file semantics over a TOML 1.0 parser.
const std = @import("std");
const toml = @import("toml");
const Allocator = std.mem.Allocator;

pub const Expectation = struct {
    skip: bool = false,
    expected_fail: bool = false,
};

const Tests = std.StringHashMapUnmanaged(Expectation);
const Suites = std.StringHashMapUnmanaged(Tests);

pub const Expectations = struct {
    arena: std.heap.ArenaAllocator,
    suites: Suites,

    pub fn parse(gpa: Allocator, source: []const u8) !Expectations {
        if (std.mem.startsWith(u8, source, "\xef\xbb\xbf")) return error.InvalidTomlBom;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const allocator = arena.allocator();
        const parser = try toml.Parser.init(allocator);
        defer parser.deinit();
        const document = try parser.parseData(source);
        const root = document.getTable();
        if (root.get("version")) |version| {
            const supported = switch (version) {
                .int => |n| n == 1,
                .float => |n| n == 1,
                .bool => |n| n,
                else => false,
            };
            if (!supported) return error.UnsupportedExpectationVersion;
        }

        var suites: Suites = .empty;
        const suite_values = try tableArray(root.get("suite"));
        for (suite_values) |suite_value| {
            if (suite_value != .table) return error.ExpectedSuiteTable;
            const suite_table = suite_value.table;
            const suite_name = try requiredString(&suite_table, "name");
            var tests: Tests = .empty;
            const test_values = try tableArray(suite_table.get("test"));
            for (test_values) |test_value| {
                if (test_value != .table) return error.ExpectedTestTable;
                const test_table = test_value.table;
                const test_name = try requiredString(&test_table, "name");
                const action = try optionalString(&test_table, "action");
                const expected = try optionalString(&test_table, "expected");
                if (action == null and expected == null) return error.MissingExpectation;
                if (action) |value| {
                    if (!std.mem.eql(u8, value, "skip")) return error.InvalidAction;
                }
                if (expected) |value| {
                    if (!std.mem.eql(u8, value, "pass") and !std.mem.eql(u8, value, "fail")) return error.InvalidExpectedOutcome;
                }
                try tests.put(allocator, test_name, .{
                    .skip = action != null,
                    .expected_fail = if (expected) |value| std.mem.eql(u8, value, "fail") else false,
                });
            }
            try suites.put(allocator, suite_name, tests);
        }
        return .{ .arena = arena, .suites = suites };
    }

    pub fn deinit(self: *Expectations) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn lookup(self: *const Expectations, suite: []const u8, test_name: []const u8) Expectation {
        const tests = self.suites.get(suite) orelse return .{};
        return tests.get(test_name) orelse .{};
    }
};

pub fn combine(filters: []const Expectations, suite: []const u8, test_name: []const u8) Expectation {
    var result: Expectation = .{};
    for (filters) |*filter| {
        const entry = filter.lookup(suite, test_name);
        result.skip = result.skip or entry.skip;
        result.expected_fail = result.expected_fail or entry.expected_fail;
    }
    return result;
}

fn tableArray(value: ?toml.TomlValue) ![]const toml.TomlValue {
    const item = value orelse return &.{};
    if (item != .array) return error.ExpectedTableArray;
    return item.array.items;
}

fn optionalString(table: *const toml.TomlTable, key: []const u8) !?[]const u8 {
    const value = table.get(key) orelse return null;
    if (value != .string) return error.ExpectedString;
    return value.string;
}

fn requiredString(table: *const toml.TomlTable, key: []const u8) ![]const u8 {
    return try optionalString(table, key) orelse error.ExpectedString;
}

test "expectations honor skip, xfail, pass and duplicate suite and test names" {
    const source =
        \\version = 1
        \\[[suite]]
        \\name = "old"
        \\[[suite.test]]
        \\name = "gone"
        \\action = "skip"
        \\[[suite]]
        \\name = "old"
        \\[[suite.test]]
        \\name = "duplicate"
        \\action = "skip"
        \\[[suite.test]]
        \\name = "duplicate"
        \\expected = "fail"
        \\[[suite.test]]
        \\name = "both"
        \\action = "skip"
        \\expected = "pass"
    ;
    var expectations = try Expectations.parse(std.testing.allocator, source);
    defer expectations.deinit();
    try std.testing.expectEqualDeep(Expectation{}, expectations.lookup("old", "gone"));
    try std.testing.expectEqualDeep(Expectation{ .expected_fail = true }, expectations.lookup("old", "duplicate"));
    try std.testing.expectEqualDeep(Expectation{ .skip = true }, expectations.lookup("old", "both"));
    try std.testing.expectEqualDeep(Expectation{}, expectations.lookup("missing", "both"));
}

test "multiple expectation files combine skips and expected failures" {
    var first = try Expectations.parse(std.testing.allocator,
        \\[[suite]]
        \\name = 's'
        \\[[suite.test]]
        \\name = 't'
        \\action = 'skip'
    );
    defer first.deinit();
    var second = try Expectations.parse(std.testing.allocator,
        \\[[suite]]
        \\name = 's'
        \\[[suite.test]]
        \\name = 't'
        \\expected = 'fail'
    );
    defer second.deinit();
    try std.testing.expectEqualDeep(Expectation{ .skip = true, .expected_fail = true }, combine(&.{ first, second }, "s", "t"));
    try std.testing.expectEqualDeep(Expectation{}, combine(&.{ first, second }, "s", "other"));
}

test "reject invalid syntax and expectation fields" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.UnsupportedExpectationVersion, Expectations.parse(a, "version = 2"));
    try std.testing.expectError(error.ExpectedTableArray, Expectations.parse(a, "suite = 1"));
    try std.testing.expectError(error.ExpectedSuiteTable, Expectations.parse(a, "suite = [1]"));
    try std.testing.expectError(error.ExpectedString, Expectations.parse(a, "[[suite]]\nname = 1"));
    try std.testing.expectError(error.ExpectedTableArray, Expectations.parse(a, "[[suite]]\nname = 's'\ntest = 1"));
    try std.testing.expectError(error.ExpectedTestTable, Expectations.parse(a, "[[suite]]\nname = 's'\ntest = [1]"));
    try std.testing.expectError(error.MissingExpectation, Expectations.parse(a, "[[suite]]\nname = 's'\n[[suite.test]]\nname = 't'"));
    try std.testing.expectError(error.InvalidAction, Expectations.parse(a, "[[suite]]\nname = 's'\n[[suite.test]]\nname = 't'\naction = 'ignore'"));
    try std.testing.expectError(error.InvalidExpectedOutcome, Expectations.parse(a, "[[suite]]\nname = 's'\n[[suite.test]]\nname = 't'\nexpected = 'skip'"));
    try std.testing.expectError(error.ExpectedString, Expectations.parse(a, "[[suite]]\nname = 's'\n[[suite.test]]\nname = 't'\naction = 1"));
    try std.testing.expectError(error.DuplicateKeyValuePair, Expectations.parse(a, "version = 1\nversion = 1"));
    try std.testing.expectError(error.TrailingComma, Expectations.parse(a, "test = { a = 1, }"));
    try std.testing.expectError(error.InvalidTomlBom, Expectations.parse(a, "\xef\xbb\xbfversion = 1"));
}

test "version and additional valid TOML forms match the upstream filter" {
    const sources = [_][]const u8{
        "",
        "version = true",
        "version = 1.0",
        "suite = [{ name = 's', test = [{ name = 't', expected = 'fail' }] }]",
        "ignored = 1979-05-27T07:32:00Z\n[[suite]]\nname = '''s'''\n",
    };
    for (sources) |source| {
        var expectations = try Expectations.parse(std.testing.allocator, source);
        defer expectations.deinit();
        if (std.mem.startsWith(u8, source, "suite")) {
            try std.testing.expect(expectations.lookup("s", "t").expected_fail);
        }
    }
}

test "repository expectation files parse" {
    const paths = [_][]const u8{
        "tests/wasi-testsuite-expectations.toml",
        "tests/wasi-p3-testsuite-expectations.toml",
    };
    for (paths) |path| {
        const source = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .unlimited);
        defer std.testing.allocator.free(source);
        var expectations = try Expectations.parse(std.testing.allocator, source);
        defer expectations.deinit();
        if (std.mem.indexOf(u8, path, "p3") == null) {
            try std.testing.expect(expectations.lookup("WASI C tests [wasm32-wasip1]", "clock_gettime-monotonic").skip);
            try std.testing.expect(expectations.lookup("WASI AssemblyScript tests [wasm32-wasip1]", "environ_sizes_get-no-variables").skip);
        }
    }
}
