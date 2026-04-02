const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const sha256_digest_length = Sha256.digest_length; // 32

/// Compute SHA-256 hash of data in one shot.
pub fn sha256(data: []const u8) [sha256_digest_length]u8 {
    var digest: [sha256_digest_length]u8 = undefined;
    Sha256.hash(data, &digest, .{});
    return digest;
}

/// Streaming SHA-256 hasher (for incremental hashing).
pub const Sha256Hasher = struct {
    state: Sha256,

    pub fn init() Sha256Hasher {
        return .{ .state = Sha256.init(.{}) };
    }

    pub fn update(self: *Sha256Hasher, data: []const u8) void {
        self.state.update(data);
    }

    pub fn final(self: *Sha256Hasher) [sha256_digest_length]u8 {
        return self.state.finalResult();
    }
};

/// Compare two hashes in constant time (prevents timing attacks).
pub fn hashEqual(a: [sha256_digest_length]u8, b: [sha256_digest_length]u8) bool {
    return std.crypto.timing_safe.eql([sha256_digest_length]u8, a, b);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn hexToBytes(comptime hex: *const [64]u8) [32]u8 {
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, hex) catch unreachable;
    return out;
}

test "sha256 - empty input matches known digest" {
    const expected = hexToBytes("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    const actual = sha256("");
    try std.testing.expectEqual(expected, actual);
}

test "sha256 - known test vector 'abc'" {
    const expected = hexToBytes("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const actual = sha256("abc");
    try std.testing.expectEqual(expected, actual);
}

test "streaming hash matches one-shot hash" {
    const data = "The quick brown fox jumps over the lazy dog";
    const one_shot = sha256(data);

    var hasher = Sha256Hasher.init();
    // Feed data in chunks to exercise streaming.
    hasher.update(data[0..10]);
    hasher.update(data[10..20]);
    hasher.update(data[20..]);
    const streamed = hasher.final();

    try std.testing.expectEqual(one_shot, streamed);
}

test "hashEqual - identical hashes" {
    const h = sha256("hello");
    try std.testing.expect(hashEqual(h, h));
}

test "hashEqual - different hashes" {
    const a = sha256("hello");
    const b = sha256("world");
    try std.testing.expect(!hashEqual(a, b));
}

test "sha256 - large input (1 MB of zeros)" {
    const one_mb = 1024 * 1024;
    const zeros = [_]u8{0} ** one_mb;
    const digest = sha256(&zeros);
    // Just verify it produces a 32-byte result without error.
    try std.testing.expectEqual(@as(usize, sha256_digest_length), digest.len);
}
