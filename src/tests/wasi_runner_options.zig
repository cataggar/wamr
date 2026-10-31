//! Command-line options for the upstream-compatible WASI suite runner.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Options = struct {
    suites: []const []const u8,
    expectations: []const []const u8,
    adapter: []const u8,
    json_output: ?[]const u8 = null,
    disable_colors: bool = false,

    pub fn deinit(self: *Options, allocator: Allocator) void {
        allocator.free(self.suites);
        allocator.free(self.expectations);
        self.* = undefined;
    }
};

fn isValue(arg: []const u8) bool {
    if (arg.len <= 1 or arg[0] != '-') return true;
    var dots: usize = 0;
    var digits_after_dot: usize = 0;
    for (arg[1..]) |char| {
        if (char == '.') {
            dots += 1;
            if (dots > 1) return false;
        } else if (std.ascii.isDigit(char)) {
            if (dots != 0) digits_after_dot += 1;
        } else return false;
    }
    return dots == 0 or digits_after_dot != 0;
}

/// Values borrow from `args`; slices are owned by the caller's allocator.
pub fn parse(allocator: Allocator, args: []const []const u8) !Options {
    var suites: std.ArrayList([]const u8) = .empty;
    errdefer suites.deinit(allocator);
    var filters: std.ArrayList([]const u8) = .empty;
    errdefer filters.deinit(allocator);
    var adapter: ?[]const u8 = null;
    var json_output: ?[]const u8 = null;
    var disable_colors = false;
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const raw = args[index];
        if (std.mem.eql(u8, raw, "-h") or std.mem.eql(u8, raw, "--help")) return error.HelpRequested;
        if (std.mem.eql(u8, raw, "--disable-colors")) {
            disable_colors = true;
            continue;
        }
        var option = raw;
        var attached: ?[]const u8 = null;
        if (std.mem.startsWith(u8, raw, "--")) {
            if (std.mem.indexOfScalar(u8, raw, '=')) |position| {
                option = raw[0..position];
                attached = raw[position + 1 ..];
            }
        } else if (raw.len > 2 and raw[0] == '-' and (raw[1] == 't' or raw[1] == 'e' or raw[1] == 'r')) {
            option = raw[0..2];
            attached = if (raw[2] == '=') raw[3..] else raw[2..];
        }
        const is_suite = std.mem.eql(u8, option, "-t") or std.mem.eql(u8, option, "--test-suite");
        const is_filter = std.mem.eql(u8, option, "-e") or std.mem.eql(u8, option, "--expectations") or
            std.mem.eql(u8, option, "--exclude-filter");
        const is_adapter = std.mem.eql(u8, option, "-r") or std.mem.eql(u8, option, "--runtime-adapter");
        const is_output = std.mem.eql(u8, option, "--json-output-location");
        if (!is_suite and !is_filter and !is_adapter and !is_output) return error.UnknownOption;
        if (is_suite or is_filter) {
            const list = if (is_suite) &suites else &filters;
            list.clearRetainingCapacity();
            if (attached) |value| {
                try list.append(allocator, value);
                continue;
            }
            const start = index + 1;
            var end = start;
            while (end < args.len and isValue(args[end])) : (end += 1) {}
            if (end == start) return error.MissingOptionValue;
            try list.appendSlice(allocator, args[start..end]);
            index = end - 1;
        } else {
            const value = if (attached) |value| value else value: {
                index += 1;
                if (index == args.len or !isValue(args[index])) return error.MissingOptionValue;
                break :value args[index];
            };
            if (is_adapter) adapter = value else json_output = value;
        }
    }
    if (suites.items.len == 0) return error.MissingTestSuite;
    const adapter_value = adapter orelse return error.MissingRuntimeAdapter;
    const suite_values = try suites.toOwnedSlice(allocator);
    errdefer allocator.free(suite_values);
    const filter_values = try filters.toOwnedSlice(allocator);
    return .{
        .suites = suite_values,
        .expectations = filter_values,
        .adapter = adapter_value,
        .json_output = json_output,
        .disable_colors = disable_colors,
    };
}

test "suite runner options retain argparse aliases and last-value behavior" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const options = try parse(allocator, &.{
        "-t",               "old",        "suite",       "--test-suite", "new",     "-e",               "old.toml",
        "--exclude-filter", "first.toml", "second.toml", "-r",           "adapter", "--disable-colors", "--json-output-location=result.json",
    });
    try std.testing.expectEqualDeep(&[_][]const u8{"new"}, options.suites);
    try std.testing.expectEqualDeep(&[_][]const u8{ "first.toml", "second.toml" }, options.expectations);
    try std.testing.expectEqualStrings("adapter", options.adapter);
    try std.testing.expectEqualStrings("result.json", options.json_output.?);
    try std.testing.expect(options.disable_colors);
    const attached = try parse(allocator, &.{ "-tsuite", "-radapter" });
    try std.testing.expectEqualStrings("suite", attached.suites[0]);
    try std.testing.expectEqualStrings("adapter", attached.adapter);
    const numeric = try parse(allocator, &.{ "-t", "-1", "-0.5", "-", "-r", "-1" });
    try std.testing.expectEqualDeep(&[_][]const u8{ "-1", "-0.5", "-" }, numeric.suites);
    try std.testing.expectEqualStrings("-1", numeric.adapter);
    try std.testing.expectError(error.HelpRequested, parse(allocator, &.{"--help"}));
    try std.testing.expectError(error.MissingTestSuite, parse(allocator, &.{ "-r", "adapter" }));
    try std.testing.expectError(error.MissingRuntimeAdapter, parse(allocator, &.{ "-t", "suite" }));
    try std.testing.expectError(error.MissingOptionValue, parse(allocator, &.{ "-t", "-r", "adapter" }));
    try std.testing.expectError(error.UnknownOption, parse(allocator, &.{ "-t", "suite", "-r", "adapter", "--other" }));
    try std.testing.expectError(error.UnknownOption, parse(allocator, &.{ "-t=suite", "extra", "-r", "adapter" }));
}

test "option parsing releases replaced lists and errors" {
    const allocator = std.testing.allocator;
    var options = try parse(allocator, &.{ "-t", "old", "-e", "old.toml", "-t", "new", "-e", "new.toml", "-r", "adapter" });
    defer options.deinit(allocator);
    try std.testing.expectError(error.UnknownOption, parse(allocator, &.{ "-t", "suite", "-e", "filter", "--unknown" }));
}
