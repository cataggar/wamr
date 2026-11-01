//! Strict TAR iteration for release inspection; unsupported PAX features fail closed.
const std = @import("std");
const block_size = 512;

fn field(bytes: []const u8) []const u8 {
    return bytes[0 .. std.mem.indexOfScalar(u8, bytes, 0) orelse bytes.len];
}

fn number(bytes: []const u8) !usize {
    if (bytes[0] == 0x80) {
        var value: usize = 0;
        for (bytes[1..]) |byte| {
            value = try std.math.mul(usize, value, 256);
            value = try std.math.add(usize, value, byte);
        }
        return value;
    }
    if (bytes[0] & 0x80 != 0) return error.InvalidTarNumber;
    const text = std.mem.trim(u8, bytes, " \x00");
    if (text.len == 0) return 0;
    return std.fmt.parseInt(usize, text, 8) catch error.InvalidTarNumber;
}

fn checksum(header: *const [block_size]u8) !void {
    const expected = try number(header[148..156]);
    var unsigned: usize = 0;
    var signed: i64 = 0;
    for (header, 0..) |byte, index| {
        const value: u8 = if (index >= 148 and index < 156) ' ' else byte;
        unsigned += value;
        signed += @as(i8, @bitCast(value));
    }
    if (expected != unsigned and (signed < 0 or expected != signed))
        return error.InvalidTarChecksum;
}

const Attributes = struct {
    comment: ?[]const u8 = null,
    path: ?[]const u8 = null,
    size: ?usize = null,
};

fn parsePax(data: []const u8, attrs: *Attributes, global: bool) !void {
    var pos: usize = 0;
    while (pos < data.len) {
        const space = std.mem.indexOfScalarPos(u8, data, pos, ' ') orelse return error.InvalidPaxRecord;
        const len = std.fmt.parseInt(usize, data[pos..space], 10) catch return error.InvalidPaxRecord;
        if (len <= space - pos + 2 or len > data.len - pos) return error.InvalidPaxRecord;
        const record = data[space + 1 .. pos + len];
        if (record[record.len - 1] != '\n') return error.InvalidPaxRecord;
        const equals = std.mem.indexOfScalar(u8, record, '=') orelse return error.InvalidPaxRecord;
        if (equals == 0) return error.InvalidPaxRecord;
        const key = record[0..equals];
        const value = record[equals + 1 .. record.len - 1];
        if (std.mem.indexOfScalar(u8, value, 0) != null) return error.InvalidPaxRecord;
        if (std.mem.eql(u8, key, "comment")) {
            if (global) attrs.comment = value;
        } else if (std.mem.eql(u8, key, "path")) {
            if (global) return error.UnsupportedGlobalPaxAttribute;
            attrs.path = value;
        } else if (std.mem.eql(u8, key, "size")) {
            if (global) return error.UnsupportedGlobalPaxAttribute;
            attrs.size = std.fmt.parseInt(usize, value, 10) catch return error.InvalidPaxRecord;
        } else if (std.mem.startsWith(u8, key, "GNU.sparse.") or std.mem.eql(u8, key, "linkpath")) {
            return error.UnsupportedPaxAttribute;
        }
        pos += len;
    }
}

pub const Entry = struct {
    name: []const u8,
    kind: u8,
    mode: u32,
    data: []const u8,
};

/// Entry names are valid until the next call; contents borrow the input.
pub const Iterator = struct {
    tar: []const u8,
    pos: usize = 0,
    global: Attributes = .{},
    local: Attributes = .{},
    local_pending: bool = false,
    name_buffer: [256]u8 = undefined,

    pub fn init(tar: []const u8) Iterator {
        return .{ .tar = tar };
    }

    pub fn comment(self: *const Iterator) ?[]const u8 {
        return self.global.comment;
    }

    pub fn next(self: *Iterator) !?Entry {
        while (self.pos < self.tar.len) {
            if (self.tar.len - self.pos < block_size) return error.TruncatedTar;
            const header = self.tar[self.pos..][0..block_size];
            if (std.mem.allEqual(u8, header, 0)) {
                if (self.local_pending) return error.TruncatedTar;
                self.pos = self.tar.len;
                return null;
            }
            try checksum(header);
            const kind = header[156];
            const header_size = try number(header[124..136]);
            const meta = kind == 'g' or kind == 'x' or kind == 'L';
            const size = if (meta) header_size else self.local.size orelse header_size;
            self.pos += block_size;
            if (size > self.tar.len - self.pos) return error.TruncatedTar;
            const data = self.tar[self.pos..][0..size];
            const padding = (block_size - size % block_size) % block_size;
            if (padding > self.tar.len - self.pos - size) return error.TruncatedTar;
            self.pos += size + padding;
            if (kind == 'g') {
                try parsePax(data, &self.global, true);
                continue;
            }
            if (kind == 'x') {
                if (self.local_pending) return error.UnsupportedPaxSequence;
                try parsePax(data, &self.local, false);
                self.local_pending = true;
                continue;
            }
            if (kind == 'L') {
                if (self.local_pending) return error.UnsupportedPaxSequence;
                self.local.path = field(data);
                self.local_pending = true;
                continue;
            }
            const name = field(header[0..100]);
            const prefix = if (std.mem.startsWith(u8, header[257..263], "ustar")) field(header[345..500]) else "";
            const header_name = if (prefix.len == 0) name else std.fmt.bufPrint(&self.name_buffer, "{s}/{s}", .{ prefix, name }) catch return error.TarPathTooLong;
            const path = if (kind == '5')
                std.mem.trimEnd(u8, self.local.path orelse header_name, "/")
            else
                self.local.path orelse header_name;
            if ((kind == '5' or kind == '1' or kind == '2') and size != 0)
                return error.UnsupportedTarEntry;
            const mode = std.math.cast(u32, try number(header[100..108])) orelse return error.InvalidTarNumber;
            self.local = .{};
            self.local_pending = false;
            return .{ .name = path, .kind = kind, .mode = mode, .data = data };
        }
        if (self.local_pending) return error.TruncatedTar;
        return null;
    }
};

test "PAX attributes reject global path overrides and malformed lengths" {
    var attrs: Attributes = .{};
    try std.testing.expectError(error.UnsupportedGlobalPaxAttribute, parsePax("14 path=other\n", &attrs, true));
    try std.testing.expectError(error.InvalidPaxRecord, parsePax("99 comment=x\n", &attrs, true));
    try parsePax("16 path=foo/bar\n", &attrs, false);
    try std.testing.expectEqualStrings("foo/bar", attrs.path.?);
}
