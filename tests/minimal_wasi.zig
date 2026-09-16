const std = @import("std");
const wasi = @import("minimal-wasi");

const Reader = struct {
    bytes: []const u8,
    offset: usize = 0,

    fn take(self: *Reader, count: usize) ![]const u8 {
        if (count > self.bytes.len - self.offset) return error.TruncatedFixture;
        const result = self.bytes[self.offset..][0..count];
        self.offset += count;
        return result;
    }

    fn byte(self: *Reader) !u8 {
        return (try self.take(1))[0];
    }

    fn leb(self: *Reader) !u32 {
        var result: u32 = 0;
        var shift: u5 = 0;
        for (0..5) |_| {
            const value = try self.byte();
            if (shift == 28 and value > 15) return error.InvalidFixture;
            result |= @as(u32, value & 0x7f) << shift;
            if (value < 128) return result;
            shift += 7;
        }
        return error.InvalidFixture;
    }

    fn string(self: *Reader) ![]const u8 {
        return self.take(try self.leb());
    }
};

fn checkFixture(bytes: []const u8, expected_hash: []const u8) !void {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    try std.testing.expectEqualStrings(expected_hash, &std.fmt.bytesToHex(hash, .lower));
    var module: Reader = .{ .bytes = bytes };
    try std.testing.expectEqualSlices(u8, "\x00asm\x01\x00\x00\x00", try module.take(8));
    const Signature = struct { params: []const u8, results: []const u8 };
    var types: std.ArrayList(Signature) = .empty;
    defer types.deinit(std.testing.allocator);
    var seen = [_]bool{false} ** wasi.imports.len;
    var import_count: u32 = 0;
    while (module.offset < bytes.len) {
        const section_id = try module.byte();
        const section_len = try module.leb();
        var section: Reader = .{ .bytes = try module.take(section_len) };
        switch (section_id) {
            1 => {
                const count = try section.leb();
                for (0..count) |_| {
                    try std.testing.expectEqual(@as(u8, 0x60), try section.byte());
                    const params = try section.string();
                    const results = try section.string();
                    try types.append(std.testing.allocator, .{ .params = params, .results = results });
                }
                try std.testing.expectEqual(section.bytes.len, section.offset);
            },
            2 => {
                import_count = try section.leb();
                try std.testing.expectEqual(@as(u32, wasi.imports.len), import_count);
                for (0..import_count) |_| {
                    const namespace = try section.string();
                    const name = try section.string();
                    try std.testing.expectEqual(@as(u8, 0), try section.byte());
                    const index = try section.leb();
                    try std.testing.expect(index < types.items.len);
                    const signature = types.items[index];
                    const function = wasi.resolve(namespace, name, signature.params, signature.results) orelse return error.UnexpectedImport;
                    try std.testing.expect(!seen[@intFromEnum(function)]);
                    seen[@intFromEnum(function)] = true;
                }
                try std.testing.expectEqual(section.bytes.len, section.offset);
            },
            else => {},
        }
    }
    try std.testing.expectEqual(@as(u32, wasi.imports.len), import_count);
    for (seen) |present| try std.testing.expect(present);
}

test "minimal WASI tracked CoreMark floating point workload identity and actual ABI" {
    try checkFixture(
        @embedFile("benchmarks/coremark/coremark_wasi.wasm"),
        "f4b7591296ead10264e0f101f355bdf848865c31329325594e66fbabefec235b",
    );
}

test "minimal WASI tracked CoreMark nofp workload identity and actual ABI" {
    try checkFixture(
        @embedFile("benchmarks/coremark/coremark_wasi_nofp.wasm"),
        "24c0cc1bd52b641cf9e8ae74d1be188cba38d74cdb7ac18378de47382aab9541",
    );
}
