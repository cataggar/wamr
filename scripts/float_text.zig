//! Shortest float text with the reference decimal notation thresholds.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn format(allocator: Allocator, value: f64) ![]const u8 {
    if (std.math.isNan(value)) return allocator.dupe(u8, "nan");
    if (std.math.isInf(value)) return allocator.dupe(u8, if (value < 0) "-inf" else "inf");
    const text = try std.fmt.allocPrint(allocator, "{d}", .{value});
    defer allocator.free(text);
    const negative = std.mem.startsWith(u8, text, "-");
    const body = text[@intFromBool(negative)..];
    const dot = std.mem.indexOfScalar(u8, body, '.') orelse body.len;
    const first = for (body, 0..) |ch, index| {
        if (ch >= '1' and ch <= '9') break index;
    } else return allocator.dupe(u8, if (negative) "-0.0" else "0.0");
    const exponent: isize = @as(isize, @intCast(dot)) - @as(isize, @intCast(first)) - @as(isize, if (first < dot) 1 else 0);
    if (exponent >= -4 and exponent < 16) {
        return if (dot == body.len) std.fmt.allocPrint(allocator, "{s}.0", .{text}) else allocator.dupe(u8, text);
    }
    const last = for (0..body.len) |offset| {
        const index = body.len - 1 - offset;
        if (body[index] >= '1' and body[index] <= '9') break index;
    } else unreachable;
    var out: std.Io.Writer.Allocating = .init(allocator);
    const w = &out.writer;
    if (negative) try w.writeByte('-');
    try w.writeByte(body[first]);
    if (last > first) {
        try w.writeByte('.');
        for (body[first + 1 .. last + 1]) |ch| if (ch != '.') try w.writeByte(ch);
    }
    try w.print("e{s}{d:0>2}", .{ if (exponent < 0) "-" else "+", @abs(exponent) });
    return out.toOwnedSlice();
}
