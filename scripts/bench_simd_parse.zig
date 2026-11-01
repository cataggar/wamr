//! Parser for the SIMD benchmark runner's tab-separated measurements.
const std = @import("std");
const text_lines = @import("text_lines.zig");
const Allocator = std.mem.Allocator;

pub const Measurement = struct {
    case: []const u8,
    engine: []const u8,
    status: []const u8,
    result: ?i128,
    compile_ns: ?i128,
    run_ns: ?i128,
    iterations: i128,
    code_size: ?i128,
    run_index: usize,
};

fn integer(raw: []const u8) !i128 {
    const text = std.mem.trim(u8, raw, " \t\r\n\x0b\x0c");
    for (text, 0..) |byte, index| {
        if (byte >= 0x80) return error.UnsupportedSimdInteger;
        if (byte == '_' and (index == 0 or index + 1 == text.len or
            !std.ascii.isDigit(text[index - 1]) or !std.ascii.isDigit(text[index + 1])))
            return error.InvalidSimdInteger;
    }
    return std.fmt.parseInt(i128, text, 10) catch |err| switch (err) {
        error.Overflow => error.SimdIntegerTooLarge,
        error.InvalidCharacter => error.InvalidSimdInteger,
    };
}

fn optionalInteger(text: []const u8) !?i128 {
    return if (std.mem.eql(u8, text, "-")) null else try integer(text);
}

fn stripLineEnd(line: []const u8) []const u8 {
    for ([_][]const u8{ "\r\n", "\xc2\x85", "\xe2\x80\xa8", "\xe2\x80\xa9", "\r", "\n", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e" }) |suffix| {
        if (std.mem.endsWith(u8, line, suffix)) return line[0 .. line.len - suffix.len];
    }
    return line;
}

/// Measurement strings borrow `output`; free the returned slice with `allocator`.
pub fn parse(allocator: Allocator, output: []const u8, run_index: usize) ![]Measurement {
    if (!std.unicode.utf8ValidateSlice(output)) return error.InvalidSimdOutputEncoding;
    var rows: std.ArrayList(Measurement) = .empty;
    errdefer rows.deinit(allocator);
    var start: usize = 0;
    while (start < output.len) {
        const end = try text_lines.nextLineEnd(output, start);
        const line = stripLineEnd(output[start..end]);
        start = end;
        if (!std.mem.startsWith(u8, line, "bench\t")) continue;
        var fields: [9][]const u8 = undefined;
        var parts = std.mem.splitScalar(u8, line, '\t');
        for (&fields) |*field| field.* = parts.next() orelse return error.MalformedSimdRow;
        if (parts.next() != null) return error.MalformedSimdRow;
        try rows.append(allocator, .{
            .case = fields[1],
            .engine = fields[2],
            .status = fields[3],
            .result = try optionalInteger(fields[4]),
            .compile_ns = try optionalInteger(fields[5]),
            .run_ns = try optionalInteger(fields[6]),
            .iterations = try integer(fields[7]),
            .code_size = try optionalInteger(fields[8]),
            .run_index = run_index,
        });
    }
    if (rows.items.len == 0) return error.NoSimdMeasurements;
    return rows.toOwnedSlice(allocator);
}

test "runner output preserves optional fields and Unicode line boundaries" {
    const allocator = std.testing.allocator;
    const rows = try parse(
        allocator,
        "diagnostic\r\nbench\tadd\twamr\tok\t42\t1000\t2_000\t100\t256\xc2\x85" ++
            "bench\tmul\taot\tunsupported\t-\t-\t-\t100\t-\xe2\x80\xa8",
        3,
    );
    defer allocator.free(rows);
    try std.testing.expectEqual(@as(usize, 2), rows.len);
    try std.testing.expectEqualStrings("add", rows[0].case);
    try std.testing.expectEqual(@as(?i128, 2000), rows[0].run_ns);
    try std.testing.expectEqual(@as(usize, 3), rows[0].run_index);
    try std.testing.expectEqualStrings("unsupported", rows[1].status);
    try std.testing.expectEqual(@as(?i128, null), rows[1].result);
}

test "runner output rejects malformed and unrepresentable measurements" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.NoSimdMeasurements, parse(allocator, "not a row\n", 1));
    try std.testing.expectError(error.MalformedSimdRow, parse(allocator, "bench\tx\ty\n", 1));
    try std.testing.expectError(error.MalformedSimdRow, parse(allocator, "bench\ta\tb\tok\t1\t1\t1\t1\t1\textra\n", 1));
    try std.testing.expectError(error.InvalidSimdInteger, parse(allocator, "bench\ta\tb\tok\t1__0\t-\t-\t1\t-\n", 1));
    try std.testing.expectError(error.SimdIntegerTooLarge, parse(allocator, "bench\ta\tb\tok\t999999999999999999999999999999999999999999999\t-\t-\t1\t-\n", 1));
}
