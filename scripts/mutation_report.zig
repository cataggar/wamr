//! Markdown reporting for mutation test results.
const std = @import("std");
const Mutant = @import("mutation_candidates.zig").Mutant;
const Range = @import("mutation_candidates.zig").Range;
const fixed = @import("decimal_format.zig").fixed;
const text_lines = @import("text_lines.zig");
const Allocator = std.mem.Allocator;

pub const Result = struct {
    mutant: Mutant,
    survived: bool,
    exit_code: i64,
    seconds: f64,
    output_tail: []const u8,
};

pub const Report = struct {
    target: []const u8,
    generated: []const u8,
    command: []const u8,
    seed: ?[]const u8,
    limit: ?[]const u8,
    ranges: []const Range,
    total_available: usize,
    results: []const Result,
};

pub const strip = text_lines.strip;

fn escape(writer: *std.Io.Writer, text: []const u8) !void {
    for (text) |byte| switch (byte) {
        '\n' => try writer.writeAll("\\n"),
        '|' => try writer.writeAll("\\|"),
        '&' => try writer.writeAll("&amp;"),
        '<' => try writer.writeAll("&lt;"),
        '>' => try writer.writeAll("&gt;"),
        '"' => try writer.writeAll("&quot;"),
        '\'' => try writer.writeAll("&#x27;"),
        else => try writer.writeByte(byte),
    };
}

fn less(_: void, a: Result, b: Result) bool {
    if (a.survived != b.survived) return a.survived;
    if (a.mutant.line_no != b.mutant.line_no) return a.mutant.line_no < b.mutant.line_no;
    return std.mem.lessThan(u8, a.mutant.mutator, b.mutant.mutator);
}

pub fn render(allocator: Allocator, report: Report) ![]u8 {
    const sorted = try allocator.dupe(Result, report.results);
    defer allocator.free(sorted);
    std.mem.sort(Result, sorted, {}, less);
    var survived: usize = 0;
    for (report.results) |result| survived += @intFromBool(result.survived);
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.print(
        \\# Mutation test report: `{s}`
        \\
        \\Generated: {s}
        \\
        \\## Summary
        \\
        \\- Result: {d} survived / {d} killed / {d} executed
        \\- Mutants available before sampling: {d}
        \\- Seed: {s}
        \\- Limit: {s}
        \\- Line ranges: 
    , .{ report.target, report.generated, survived, report.results.len - survived, report.results.len, report.total_available, report.seed orelse "none", report.limit orelse "none" });
    if (report.ranges.len == 0) {
        try w.writeAll("all lines");
    } else {
        for (report.ranges, 0..) |range, index| {
            if (index != 0) try w.writeAll(", ");
            try w.print("{d}:{d}", .{ range.start, range.end });
        }
    }
    try w.print(
        \\
        \\- Test command: `{s}`
        \\
        \\## Mutants
        \\
        \\SURVIVED rows are sorted first.
        \\
        \\| Result | Line | Mutator | Original line | Exit | Seconds | Output tail |
        \\| --- | ---: | --- | --- | ---: | ---: | --- |
        \\
    , .{report.command});
    for (sorted) |result| {
        try w.print("| {s} | {d} | ", .{ if (result.survived) "SURVIVED" else "KILLED", result.mutant.line_no });
        try escape(w, result.mutant.mutator);
        try w.writeAll(" | `");
        try escape(w, try strip(result.mutant.original_line));
        const seconds = try fixed(allocator, result.seconds, 1);
        defer allocator.free(seconds);
        try w.print("` | {d} | {s} | `", .{ result.exit_code, seconds });
        try escape(w, result.output_tail);
        try w.writeAll("` |\n");
    }
    return out.toOwnedSlice();
}

test "report sorts survivors first and escapes table content" {
    const allocator = std.testing.allocator;
    const results = [_]Result{
        .{ .mutant = .{ .line_no = 3, .original_line = "  x < 1;", .mutated_line = "", .mutator = "comparison < -> >", .detail = "" }, .survived = false, .exit_code = 7, .seconds = 1.25, .output_tail = "bad | 'err'\n" },
        .{ .mutant = .{ .line_no = 5, .original_line = "\xc2\xa0value & 1;\xc2\xa0", .mutated_line = "", .mutator = "integer 1 -> 0", .detail = "" }, .survived = true, .exit_code = 0, .seconds = 0.25, .output_tail = "\"ok\"" },
    };
    const actual = try render(allocator, .{
        .target = "target.zig",
        .generated = "2024-01-02 03:04:05Z",
        .command = "test",
        .seed = "737",
        .limit = "2",
        .ranges = &.{.{ .start = 2, .end = 5 }},
        .total_available = 4,
        .results = &results,
    });
    defer allocator.free(actual);
    try std.testing.expectEqualStrings(
        \\# Mutation test report: `target.zig`
        \\
        \\Generated: 2024-01-02 03:04:05Z
        \\
        \\## Summary
        \\
        \\- Result: 1 survived / 1 killed / 2 executed
        \\- Mutants available before sampling: 4
        \\- Seed: 737
        \\- Limit: 2
        \\- Line ranges: 2:5
        \\- Test command: `test`
        \\
        \\## Mutants
        \\
        \\SURVIVED rows are sorted first.
        \\
        \\| Result | Line | Mutator | Original line | Exit | Seconds | Output tail |
        \\| --- | ---: | --- | --- | ---: | ---: | --- |
        \\| SURVIVED | 5 | integer 1 -&gt; 0 | `value &amp; 1;` | 0 | 0.2 | `&quot;ok&quot;` |
        \\| KILLED | 3 | comparison &lt; -&gt; &gt; | `x &lt; 1;` | 7 | 1.2 | `bad \| &#x27;err&#x27;\n` |
        \\
    , actual);
}

test "empty reports preserve default summary" {
    const allocator = std.testing.allocator;
    const actual = try render(allocator, .{
        .target = "target.zig",
        .generated = "2024-01-02 03:04:05Z",
        .command = "test",
        .seed = null,
        .limit = null,
        .ranges = &.{},
        .total_available = 0,
        .results = &.{},
    });
    defer allocator.free(actual);
    try std.testing.expect(std.mem.indexOf(u8, actual, "- Result: 0 survived / 0 killed / 0 executed\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, actual, "- Seed: none\n- Limit: none\n- Line ranges: all lines\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, actual, "| --- | ---: | --- | --- | ---: | ---: | --- |\n"));
}
