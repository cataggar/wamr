//! Integer-seeded reference sampling for mutation candidates.
const std = @import("std");

pub const Generator = struct {
    state: [624]u32,
    index: usize,

    /// The absolute seed is expressed as little-endian 32-bit limbs.
    pub fn init(seed: []const u32) Generator {
        var length = seed.len;
        while (length > 0 and seed[length - 1] == 0) : (length -= 1) {}
        const key: []const u32 = if (length == 0) &.{0} else seed[0..length];
        var result: Generator = .{ .state = undefined, .index = 624 };
        result.state[0] = 19_650_218;
        for (1..624) |i| {
            const previous = result.state[i - 1];
            result.state[i] = 1_812_433_253 *% (previous ^ (previous >> 30)) +% @as(u32, @intCast(i));
        }
        var i: usize = 1;
        var j: usize = 0;
        for (0..@max(624, key.len)) |_| {
            const previous = result.state[i - 1];
            result.state[i] = (result.state[i] ^ ((previous ^ (previous >> 30)) *% 1_664_525)) +% key[j] +% @as(u32, @truncate(j));
            i += 1;
            j += 1;
            if (i == 624) {
                result.state[0] = result.state[623];
                i = 1;
            }
            if (j == key.len) j = 0;
        }
        for (0..623) |_| {
            const previous = result.state[i - 1];
            result.state[i] = (result.state[i] ^ ((previous ^ (previous >> 30)) *% 1_566_083_941)) -% @as(u32, @intCast(i));
            i += 1;
            if (i == 624) {
                result.state[0] = result.state[623];
                i = 1;
            }
        }
        result.state[0] = 0x80000000;
        return result;
    }

    fn next(self: *Generator) u32 {
        if (self.index == 624) {
            for (0..624) |i| {
                const combined = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624] ^ (combined >> 1) ^ (if (combined & 1 != 0) @as(u32, 0x9908b0df) else 0);
            }
            self.index = 0;
        }
        var value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680;
        value ^= (value << 15) & 0xefc60000;
        value ^= value >> 18;
        return value;
    }

    fn below(self: *Generator, bound: usize) usize {
        std.debug.assert(bound > 0);
        const bits: u7 = @intCast(@bitSizeOf(usize) - @clz(bound));
        while (true) {
            var value: u64 = self.next();
            if (bits <= 32) {
                value >>= @intCast(32 - bits);
            } else {
                const high = self.next() >> @as(u5, @intCast(64 - bits));
                value |= @as(u64, high) << 32;
            }
            if (value < bound) return @intCast(value);
        }
    }

    pub fn shuffle(self: *Generator, comptime T: type, items: []T) void {
        var count = items.len;
        while (count > 1) {
            const other = self.below(count);
            count -= 1;
            std.mem.swap(T, &items[count], &items[other]);
        }
    }
};

test "integer-seeded shuffles preserve reference sampling" {
    const cases = [_]struct { seed: []const u32, expected: [16]u8 }{
        .{ .seed = &.{}, .expected = .{ 10, 14, 5, 1, 9, 2, 3, 11, 13, 7, 8, 4, 0, 6, 15, 12 } },
        .{ .seed = &.{1}, .expected = .{ 2, 10, 0, 14, 6, 5, 3, 8, 7, 11, 15, 1, 12, 13, 9, 4 } },
        .{ .seed = &.{737}, .expected = .{ 10, 2, 1, 14, 8, 12, 6, 7, 4, 5, 11, 13, 15, 3, 9, 0 } },
        .{ .seed = &.{ 12345, 0, 0, 16 }, .expected = .{ 15, 13, 5, 3, 8, 12, 6, 10, 7, 2, 0, 11, 9, 4, 1, 14 } },
        .{ .seed = &.{ 0, 0, 0, 0x80000000 }, .expected = .{ 11, 5, 9, 8, 12, 7, 15, 1, 0, 13, 4, 3, 10, 6, 2, 14 } },
    };
    for (cases) |case| {
        var rng = Generator.init(case.seed);
        var values: [16]u8 = undefined;
        for (&values, 0..) |*value, index| value.* = @intCast(index);
        rng.shuffle(u8, &values);
        try std.testing.expectEqualSlices(u8, &case.expected, &values);
    }
}

test "long shuffles and wide bounds preserve reference sampling" {
    var values: [4097]u8 = undefined;
    for (&values, 0..) |*value, index| value.* = @truncate(index);
    var rng = Generator.init(&.{737});
    rng.shuffle(u8, &values);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&values, &digest, .{});
    try std.testing.expectEqualStrings(
        "629e553204410e997f0e9db75753bb980cab6e9fea98e326f7cf91ea3390577f",
        &std.fmt.bytesToHex(digest, .lower),
    );
    if (@bitSizeOf(usize) < 64) return;
    rng = Generator.init(&.{737});
    for ([_]struct { bound: u64, expected: u64 }{
        .{ .bound = 2, .expected = 0 },
        .{ .bound = 3, .expected = 2 },
        .{ .bound = 1 << 32, .expected = 812851847 },
        .{ .bound = (1 << 32) + 1, .expected = 3552311903 },
        .{ .bound = 1 << 40, .expected = 148916052850 },
        .{ .bound = 1 << 63, .expected = 8027236290496157357 },
        .{ .bound = std.math.maxInt(u64), .expected = 4427955292976619570 },
    }) |case| try std.testing.expectEqual(case.expected, rng.below(@intCast(case.bound)));
}
