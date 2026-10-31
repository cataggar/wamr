//! UTF-8 line boundaries for text read by automation tools.
const std = @import("std");

pub fn strip(text: []const u8) ![]const u8 {
    var index: usize = 0;
    var start: ?usize = null;
    var end: usize = 0;
    while (index < text.len) {
        const width = try std.unicode.utf8ByteSequenceLength(text[index]);
        if (width > text.len - index) return error.InvalidUtf8;
        const codepoint = try std.unicode.utf8Decode(text[index..][0..width]);
        const whitespace = switch (codepoint) {
            0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
            else => false,
        };
        if (!whitespace) {
            if (start == null) start = index;
            end = index + width;
        }
        index += width;
    }
    return if (start) |first| text[first..end] else "";
}

pub fn nextLineEnd(source: []const u8, start: usize) !usize {
    var end = start;
    while (end < source.len) {
        const width = try std.unicode.utf8ByteSequenceLength(source[end]);
        if (width > source.len - end) return error.InvalidTextEncoding;
        const point = try std.unicode.utf8Decode(source[end..][0..width]);
        end += width;
        if (switch (point) {
            '\n', '\r', 0x0b, 0x0c, 0x1c...0x1e, 0x85, 0x2028, 0x2029 => true,
            else => false,
        }) {
            if (point == '\r' and end < source.len and source[end] == '\n') end += 1;
            break;
        }
    }
    return end;
}

test "line boundaries include CRLF and Unicode separators" {
    const text = "a\r\nb\xc2\x85c\xe2\x80\xa8d";
    var start: usize = 0;
    for ([_][]const u8{ "a\r\n", "b\xc2\x85", "c\xe2\x80\xa8", "d" }) |line| {
        const end = try nextLineEnd(text, start);
        try std.testing.expectEqualStrings(line, text[start..end]);
        start = end;
    }
    try std.testing.expectEqual(text.len, start);
}
