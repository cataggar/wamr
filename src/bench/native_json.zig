//! Small JSON helpers for consuming the existing native_benchmark.py protocol.
const std = @import("std");
pub const Value = std.json.Value;
pub const Error = error{InvalidConfiguration};

pub fn field(value: Value, name: []const u8) Error!Value {
    if (value != .object) return error.InvalidConfiguration;
    return value.object.get(name) orelse error.InvalidConfiguration;
}

pub fn text(value: Value) Error![]const u8 {
    if (value != .string) return error.InvalidConfiguration;
    return value.string;
}

pub fn string(value: Value, name: []const u8) Error![]const u8 {
    return text(try field(value, name));
}

pub fn integer(value: Value) Error!u64 {
    if (value != .integer or value.integer < 0) return error.InvalidConfiguration;
    return @intCast(value.integer);
}

pub fn number(value: Value, name: []const u8) Error!u64 {
    return integer(try field(value, name));
}

pub fn matches(value: Value, expected: []const u8) bool {
    return value == .string and std.mem.eql(u8, value.string, expected);
}

pub fn require(ok: bool) Error!void {
    if (!ok) return error.InvalidConfiguration;
}

pub fn object(allocator: std.mem.Allocator) Value {
    _ = allocator;
    return .{ .object = .{} };
}

pub fn put(allocator: std.mem.Allocator, value: *Value, name: []const u8, child: Value) !void {
    try value.object.put(allocator, name, child);
}

pub fn array(allocator: std.mem.Allocator) Value {
    return .{ .array = std.array_list.Managed(Value).init(allocator) };
}

pub fn str(value: []const u8) Value {
    return .{ .string = value };
}

pub fn num(value: u64) Value {
    return .{ .integer = @intCast(value) };
}

pub fn digest(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var output: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &output, .{});
    return allocator.dupe(u8, &std.fmt.bytesToHex(output, .lower));
}

/// Requests contain ASCII public tokens and JSON integers. Sort keys and use
/// Python's ensure_ascii=True encoding, rejecting floats rather than silently
/// producing a different canonical representation.
pub fn canonical(allocator: std.mem.Allocator, value: Value) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    try canonicalWrite(allocator, value, &out.writer);
    return out.toOwnedSlice();
}

fn canonicalWrite(allocator: std.mem.Allocator, value: Value, out: *std.Io.Writer) anyerror!void {
    switch (value) {
        .object => |map| {
            const names = try allocator.dupe([]const u8, map.keys());
            defer allocator.free(names);
            std.mem.sort([]const u8, names, {}, struct {
                fn less(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.less);
            try out.writeByte('{');
            for (names, 0..) |name, index| {
                if (index != 0) try out.writeByte(',');
                try std.json.Stringify.value(name, .{ .escape_unicode = true }, out);
                try out.writeByte(':');
                try canonicalWrite(allocator, map.get(name).?, out);
            }
            try out.writeByte('}');
        },
        .array => |list| {
            try out.writeByte('[');
            for (list.items, 0..) |child, index| {
                if (index != 0) try out.writeByte(',');
                try canonicalWrite(allocator, child, out);
            }
            try out.writeByte(']');
        },
        .float, .number_string => return error.InvalidConfiguration,
        else => try std.json.Stringify.value(value, .{ .escape_unicode = true }, out),
    }
}

pub fn equal(allocator: std.mem.Allocator, left: Value, right: Value) !bool {
    const a = try canonical(allocator, left);
    defer allocator.free(a);
    const b = try canonical(allocator, right);
    defer allocator.free(b);
    return std.mem.eql(u8, a, b);
}

test "native request canonical JSON matches Python" {
    const allocator = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Value, allocator, "{\"z\":true,\"a\":[1,null,\"abc\"]}", .{});
    defer parsed.deinit();
    const bytes = try canonical(allocator, parsed.value);
    defer allocator.free(bytes);
    try std.testing.expectEqualStrings("{\"a\":[1,null,\"abc\"],\"z\":true}", bytes);
    try std.testing.expectError(error.InvalidConfiguration, canonical(allocator, .{ .float = 1.5 }));
    try std.testing.expectError(error.DuplicateField, std.json.parseFromSlice(Value, allocator, "{\"a\":1,\"a\":2}", .{}));
}
