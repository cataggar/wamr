//! Experimental candidate enumeration for the mutation-testing helper.
const std = @import("std");
const text_lines = @import("text_lines.zig");
const Allocator = std.mem.Allocator;

pub const Mutant = struct {
    line_no: usize,
    original_line: []const u8,
    mutated_line: []const u8,
    mutator: []const u8,
    detail: []const u8,
};

pub const Range = struct { start: usize, end: usize };

pub fn normalizeNewlines(allocator: Allocator, source: []const u8) ![]u8 {
    const normalized = try allocator.alloc(u8, source.len);
    errdefer allocator.free(normalized);
    var read: usize = 0;
    var written: usize = 0;
    while (read < source.len) {
        if (source[read] == '\r') {
            normalized[written] = '\n';
            read += 1;
            if (read < source.len and source[read] == '\n') read += 1;
        } else {
            normalized[written] = source[read];
            read += 1;
        }
        written += 1;
    }
    return allocator.realloc(normalized, written);
}

fn whitespace(point: u21) bool {
    return switch (point) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn word(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_';
}

fn wordAfterOne(text: []const u8, index: usize) !bool {
    if (index == text.len) return false;
    if (text[index] < 0x80) return word(text[index]);
    const width = try std.unicode.utf8ByteSequenceLength(text[index]);
    const point = try std.unicode.utf8Decode(text[index..][0..width]);
    if (whitespace(point)) return false;
    if (point <= 0xff) return switch (point) {
        0xaa,
        0xb2...0xb3,
        0xb5,
        0xb9...0xba,
        0xbc...0xbe,
        0xc0...0xd6,
        0xd8...0xf6,
        0xf8...0xff,
        => true,
        else => false,
    };
    return error.UnsupportedMutationSource;
}

fn skipWhitespace(text: []const u8, start: usize) !usize {
    var index = start;
    while (index < text.len) {
        const width = try std.unicode.utf8ByteSequenceLength(text[index]);
        if (width > text.len - index) return error.InvalidMutationSourceEncoding;
        const point = try std.unicode.utf8Decode(text[index..][0..width]);
        if (!whitespace(point)) break;
        index += width;
    }
    return index;
}

fn trimStart(text: []const u8) ![]const u8 {
    return text[try skipWhitespace(text, 0)..];
}

fn trimEnd(text: []const u8) ![]const u8 {
    var index: usize = 0;
    var end: usize = 0;
    while (index < text.len) {
        const width = try std.unicode.utf8ByteSequenceLength(text[index]);
        if (width > text.len - index) return error.InvalidMutationSourceEncoding;
        const point = try std.unicode.utf8Decode(text[index..][0..width]);
        index += width;
        if (!whitespace(point)) end = index;
    }
    return text[0..end];
}

pub fn nextLineEnd(source: []const u8, start: usize) !usize {
    return text_lines.nextLineEnd(source, start) catch error.InvalidMutationSourceEncoding;
}

fn column(code: []const u8, index: usize) !usize {
    return (try std.unicode.utf8CountCodepoints(code[0..index])) + 1;
}

fn splitCode(line: []const u8) usize {
    var in_string = false;
    var escaped = false;
    for (line, 0..) |char, i| {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (char == '\\') {
                escaped = true;
            } else if (char == '"') {
                in_string = false;
            }
        } else if (char == '"') {
            in_string = true;
        } else if (char == '/' and i + 1 < line.len and line[i + 1] == '/') {
            return i;
        }
    }
    return line.len;
}

fn addReplacement(
    allocator: Allocator,
    out: *std.ArrayList(Mutant),
    line_no: usize,
    original: []const u8,
    line: []const u8,
    start: usize,
    end: usize,
    replacement: []const u8,
    mutator: []const u8,
    detail: []const u8,
) !void {
    try out.append(allocator, .{
        .line_no = line_no,
        .original_line = original,
        .mutated_line = try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ line[0..start], replacement, line[end..] }),
        .mutator = mutator,
        .detail = detail,
    });
}

fn enumerateLine(allocator: Allocator, out: *std.ArrayList(Mutant), line: []const u8, line_no: usize) !void {
    const code = line[0..splitCode(line)];
    const newline: []const u8 = if (std.mem.endsWith(u8, line, "\n")) "\n" else "";
    const body = line[0 .. line.len - newline.len];
    const original = std.mem.trimEnd(u8, line, "\n");
    const stripped = try trimEnd(try trimStart(body));
    const left = try trimStart(body);
    const indent = body[0 .. body.len - left.len];

    if (std.mem.indexOf(u8, code, "clearAll(") != null and std.mem.endsWith(u8, stripped, ";")) {
        try out.append(allocator, .{
            .line_no = line_no,
            .original_line = original,
            .mutated_line = try std.fmt.allocPrint(allocator, "{s}// mutation_test: no-op {s}{s}", .{ indent, left, newline }),
            .mutator = "clearAll -> no-op",
            .detail = "comment out clearAll line",
        });
    }
    var index: usize = 0;
    while (index < code.len) {
        const ch = code[index];
        if ((ch == '<' or ch == '>') and
            (index == 0 or std.mem.indexOfScalar(u8, "<>=!", code[index - 1]) == null))
        {
            const end = index + 1 + @as(usize, @intFromBool(index + 1 < code.len and code[index + 1] == '='));
            if (end == code.len or std.mem.indexOfScalar(u8, "<>=", code[end]) == null) {
                const op = code[index..end];
                const replacement = if (ch == '<') (if (op.len == 2) ">=" else ">") else (if (op.len == 2) "<=" else "<");
                try addReplacement(allocator, out, line_no, original, line, index, end, replacement, try std.fmt.allocPrint(allocator, "comparison {s} -> {s} at column {d}", .{ op, replacement, try column(code, index) }), "swap relational operator");
                index = end;
                continue;
            }
        }
        index += 1;
    }

    for ([_][]const u8{ "continue", "break", "return" }) |keyword| {
        if (!std.mem.startsWith(u8, left, keyword) or left.len <= keyword.len or left[keyword.len] != ';') continue;
        const rest = try trimStart(left[keyword.len + 1 ..]);
        if (rest.len != 0 and !std.mem.startsWith(u8, rest, "//")) continue;
        try out.append(allocator, .{
            .line_no = line_no,
            .original_line = original,
            .mutated_line = try std.fmt.allocPrint(allocator, "{s}// mutation_test: deleted {s};{s}", .{ indent, keyword, newline }),
            .mutator = try std.fmt.allocPrint(allocator, "delete {s};", .{keyword}),
            .detail = "delete standalone control statement",
        });
        break;
    }

    for (code, 0..) |ch, i| {
        if ((ch != '0' and ch != '1') or (i > 0 and word(code[i - 1])) or
            (i + 1 < code.len and word(code[i + 1]))) continue;
        const after = try skipWhitespace(code, i + 1);
        if (after == code.len or std.mem.indexOfScalar(u8, ";,)}+-*/%<>=!&|?:", code[after]) == null) continue;
        const replacement: []const u8 = if (ch == '0') "1" else "0";
        try addReplacement(allocator, out, line_no, original, line, i, i + 1, replacement, try std.fmt.allocPrint(allocator, "integer {c} -> {s} at column {d}", .{ ch, replacement, try column(code, i) }), "flip zero/one integer literal");
    }
    index = 0;
    while (index < code.len) {
        const op = code[index];
        if (op == '+' or op == '-') {
            const after = try skipWhitespace(code, index + 1);
            if (after > index + 1 and after < code.len and code[after] == '1') {
                if (try wordAfterOne(code, after + 1)) {
                    index += 1;
                    continue;
                }
                const replacement: []const u8 = if (op == '+') "- 1" else "+ 1";
                try addReplacement(allocator, out, line_no, original, line, index, after + 1, replacement, try std.fmt.allocPrint(allocator, "off-by-one {c} 1 -> {s} at column {d}", .{ op, replacement, try column(code, index) }), "flip +/- 1");
                index = after + 1;
                continue;
            }
        }
        index += 1;
    }
}

/// Use an arena: returned strings borrow `source` or are allocated in `allocator`.
/// Unrecognized non-ASCII word boundaries fail until Unicode regex parity is complete.
pub fn collect(allocator: Allocator, source: []const u8, ranges: []const Range) ![]Mutant {
    if (!std.unicode.utf8ValidateSlice(source)) return error.InvalidMutationSourceEncoding;
    var mutants: std.ArrayList(Mutant) = .empty;
    var start: usize = 0;
    var line_no: usize = 1;
    while (start < source.len) : (line_no += 1) {
        const end = try nextLineEnd(source, start);
        var included = ranges.len == 0;
        for (ranges) |range| if (line_no >= range.start and line_no <= range.end) {
            included = true;
            break;
        };
        if (included) try enumerateLine(allocator, &mutants, source[start..end], line_no);
        start = end;
    }
    return mutants.toOwnedSlice(allocator);
}

test "enumerate candidate kinds in reference order without mutating comments" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const candidates = try collect(arena.allocator(),
        \\  value.clearAll();
        \\  if (x <= 1) return; // ignored < 1
        \\  continue; // note
        \\  x = x + 1;
        \\
    , &.{});
    const names = [_][]const u8{
        "clearAll -> no-op",
        "comparison <= -> >= at column 9",
        "integer 1 -> 0 at column 12",
        "delete continue;",
        "integer 1 -> 0 at column 11",
        "off-by-one + 1 -> - 1 at column 9",
    };
    try std.testing.expectEqual(names.len, candidates.len);
    for (names, candidates) |name, candidate| try std.testing.expectEqualStrings(name, candidate.mutator);
    try std.testing.expectEqualStrings("  if (x >= 1) return; // ignored < 1\n", candidates[1].mutated_line);
    try std.testing.expectEqualStrings("  // mutation_test: deleted continue;\n", candidates[3].mutated_line);
}

test "ranges, quoted comment markers, operator boundaries and unsupported input" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const source = "ignored < 1;\r\nconst s = \"// < 0;\"; // > 1\nx <<= 1; y != 0;\n";
    const found = try collect(allocator, source, &.{.{ .start = 2, .end = 3 }});
    try std.testing.expectEqual(@as(usize, 4), found.len);
    try std.testing.expectEqualStrings("comparison < -> > at column 15", found[0].mutator);
    try std.testing.expectEqualStrings("integer 0 -> 1 at column 17", found[1].mutator);
    try std.testing.expectEqualStrings("integer 1 -> 0 at column 7", found[2].mutator);
    try std.testing.expectEqualStrings("integer 0 -> 1 at column 15", found[3].mutator);
    const adjacent = try collect(allocator, "x = x-1;\nx = x - 1;\n", &.{});
    try std.testing.expectEqual(@as(usize, 3), adjacent.len);
    try std.testing.expectEqualStrings("off-by-one - 1 -> + 1 at column 7", adjacent[2].mutator);
    const unicode_comment = try collect(allocator, "x = 0; // caf\xc3\xa9\n", &.{});
    try std.testing.expectEqualStrings("x = 1; // caf\xc3\xa9\n", unicode_comment[0].mutated_line);
    const unicode = try collect(allocator, "\xce\xbb <\xc2\xa01;\xe2\x80\xa8x +\xc2\xa01;", &.{});
    try std.testing.expectEqualStrings("comparison < -> > at column 3", unicode[0].mutator);
    try std.testing.expectEqualStrings("integer 1 -> 0 at column 5", unicode[1].mutator);
    try std.testing.expectEqualStrings("off-by-one + 1 -> - 1 at column 3", unicode[3].mutator);
    try std.testing.expectEqual(@as(usize, 0), (try collect(allocator, "x + 1\xc3\xa9;", &.{})).len);
    const punctuation = try collect(allocator, "x + 1\xc2\xa9;", &.{});
    try std.testing.expectEqualStrings("off-by-one + 1 -> - 1 at column 3", punctuation[0].mutator);
    try std.testing.expectError(error.UnsupportedMutationSource, collect(allocator, "x + 1\xe2\x98\x83;", &.{}));
    try std.testing.expectError(error.InvalidMutationSourceEncoding, collect(allocator, "x = 0; // \xff", &.{}));
}

test "Latin-1 word boundaries match the reference character classes" {
    const word_classes: u128 = 0xff7fffffff7fffff762c040000000000;
    for (0x80..0x100) |point| {
        var encoded: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(@intCast(point), &encoded);
        const expected = (word_classes & (@as(u128, 1) << @as(u7, @intCast(point - 0x80)))) != 0;
        try std.testing.expectEqual(expected, try wordAfterOne(encoded[0..len], 0));
    }
}
