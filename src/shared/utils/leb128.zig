//! LEB128 (Little-Endian Base 128) decoding for WebAssembly.
//!
//! Zig port of `bh_leb128.h` / `bh_leb128.c`.  The C implementation exposes
//! a single `bh_leb_read` function parameterised by `maxbits` and `sign`.
//! This module provides type-safe generic `readUnsigned` and `readSigned`
//! functions that accept `u32`, `u64`, `i32`, or `i64`.

const std = @import("std");
const testing = std.testing;

pub const Error = error{
    /// The encoded value does not fit in the target type.
    Overflow,
    /// The byte stream ended before the value was fully decoded.
    UnexpectedEnd,
};

/// Result of a successful LEB128 decode.
pub fn Result(comptime T: type) type {
    return struct {
        value: T,
        bytes_read: usize,
    };
}

/// Decode an unsigned LEB128 value from `bytes` into a `T`-sized unsigned integer.
///
/// `T` must be `u32` or `u64`.
pub inline fn readUnsigned(comptime T: type, bytes: []const u8) Error!Result(T) {
    comptime {
        if (T != u32 and T != u64) @compileError("readUnsigned only supports u32 and u64");
    }
    if (bytes.len > 0 and bytes[0] < 0x80) {
        return .{ .value = bytes[0], .bytes_read = 1 };
    }
    return readUnsignedMulti(T, bytes);
}

fn readUnsignedMulti(comptime T: type, bytes: []const u8) Error!Result(T) {
    const maxbits: u7 = @bitSizeOf(T);
    const max_bytes: usize = (maxbits + 6) / 7;

    var result: u64 = 0;
    var shift: u32 = 0;
    var byte_count: usize = 0;

    while (true) {
        if (byte_count >= max_bytes) return Error.Overflow;
        if (byte_count >= bytes.len) return Error.UnexpectedEnd;

        const byte: u64 = bytes[byte_count];
        byte_count += 1;
        result |= (byte & 0x7f) << @intCast(shift);
        shift += 7;

        if ((byte & 0x80) == 0) break;
    }

    // Overflow check for u32: if we consumed enough bytes that shift >= 32,
    // the last byte must not have bits set above the valid range.
    if (T == u32 and shift >= maxbits) {
        const last_byte: u8 = bytes[byte_count - 1];
        if ((last_byte & 0xf0) != 0) return Error.Overflow;
    }

    return .{ .value = @truncate(result), .bytes_read = byte_count };
}

/// Decode a signed LEB128 value from `bytes` into a `T`-sized signed integer.
///
/// `T` must be `i32` or `i64`.
pub inline fn readSigned(comptime T: type, bytes: []const u8) Error!Result(T) {
    comptime {
        if (T != i32 and T != i64) @compileError("readSigned only supports i32 and i64");
    }
    if (bytes.len > 0 and bytes[0] < 0x80) {
        const sign_extended: i8 = @as(i8, @bitCast(bytes[0] << 1)) >> 1;
        return .{ .value = sign_extended, .bytes_read = 1 };
    }
    return readSignedMulti(T, bytes);
}

fn readSignedMulti(comptime T: type, bytes: []const u8) Error!Result(T) {
    const maxbits: u7 = @bitSizeOf(T);
    const max_bytes: usize = (maxbits + 6) / 7;

    var result: u64 = 0;
    var shift: u32 = 0;
    var byte_count: usize = 0;

    while (true) {
        if (byte_count >= max_bytes) return Error.Overflow;
        if (byte_count >= bytes.len) return Error.UnexpectedEnd;

        const byte: u64 = bytes[byte_count];
        byte_count += 1;
        result |= (byte & 0x7f) << @intCast(shift);
        shift += 7;

        if ((byte & 0x80) == 0) break;
    }

    const last_byte: u8 = bytes[byte_count - 1];

    if (T == i32) {
        if (shift < maxbits) {
            // Sign-extend if the sign bit (bit 6 of last byte) is set.
            if ((last_byte & 0x40) != 0) {
                result |= ~@as(u64, 0) << @intCast(shift);
            }
        } else {
            // shift >= maxbits: validate that the top bits are a sign extension.
            const sign_bit_set = (last_byte & 0x8) != 0;
            const top_bits: u8 = last_byte & 0xf0;
            if ((sign_bit_set and top_bits != 0x70) or
                (!sign_bit_set and top_bits != 0))
                return Error.Overflow;
        }
    } else {
        // i64
        if (shift < maxbits) {
            if ((last_byte & 0x40) != 0) {
                result |= ~@as(u64, 0) << @as(u6, @intCast(shift));
            }
        } else {
            const sign_bit_set = (last_byte & 0x1) != 0;
            const top_bits: u8 = last_byte & 0xfe;
            if ((sign_bit_set and top_bits != 0x7e) or
                (!sign_bit_set and top_bits != 0))
                return Error.Overflow;
        }
    }

    return .{ .value = @bitCast(@as(@Int(.unsigned, maxbits), @truncate(result))), .bytes_read = byte_count };
}

// ─────────────────────────────────────────────────────────────────────────
// Lossy convenience wrappers
//
// These take `bytes` and a `pos: *usize` cursor and advance it in-place.
// On malformed/truncated input they return a fallback value and skip any
// remaining continuation bytes so the caller does not spin. They exist
// because the interpreter and AOT frontend decode bytecode that has
// already been validated at load time; a decode error at runtime is
// effectively "can't happen" and the bytecode dispatch loops cannot
// propagate errors through their tail-call structure. New code should
// prefer `readUnsigned` / `readSigned` and handle the error result.
// ─────────────────────────────────────────────────────────────────────────

fn skipToEndPos(bytes: []const u8, start_pos: usize) usize {
    // Skip any continuation bytes so callers don't spin on malformed input.
    var p = start_pos;
    while (p < bytes.len and (bytes[p] & 0x80) != 0) p += 1;
    if (p < bytes.len) p += 1;
    return p;
}

fn LossyResult(comptime T: type) type {
    return struct {
        value: T,
        next_pos: usize,
    };
}

noinline fn readUnsignedLossySlow(comptime T: type, bytes: []const u8, p: usize) LossyResult(T) {
    if (comptime @bitSizeOf(T) >= 32) {
        if (p + 2 <= bytes.len) {
            const b0 = bytes[p];
            const b1 = bytes[p + 1];
            if (b0 >= 0x80 and b1 < 0x80) {
                return .{
                    .value = @as(T, b0 & 0x7f) | (@as(T, b1) << 7),
                    .next_pos = p + 2,
                };
            }
            if (p + 5 <= bytes.len) {
                const b2 = bytes[p + 2];
                const b3 = bytes[p + 3];
                const b4 = bytes[p + 4];
                if ((b0 & b1 & b2 & b3) >= 0x80 and b4 < 0x10) {
                    const v: u64 = @as(u64, b0 & 0x7f) |
                        (@as(u64, b1 & 0x7f) << 7) |
                        (@as(u64, b2 & 0x7f) << 14) |
                        (@as(u64, b3 & 0x7f) << 21) |
                        (@as(u64, b4) << 28);
                    return .{
                        .value = @intCast(v),
                        .next_pos = p + 5,
                    };
                }
            }
        }
    }
    if (p >= bytes.len) return .{ .value = 0, .next_pos = p };
    const r = readUnsignedMulti(T, bytes[p..]) catch {
        return .{ .value = 0, .next_pos = skipToEndPos(bytes, p) };
    };
    return .{ .value = r.value, .next_pos = p + r.bytes_read };
}

noinline fn readSignedLossySlow(comptime T: type, bytes: []const u8, p: usize) LossyResult(T) {
    if (p >= bytes.len) return .{ .value = 0, .next_pos = p };
    const r = readSignedMulti(T, bytes[p..]) catch {
        return .{ .value = 0, .next_pos = skipToEndPos(bytes, p) };
    };
    return .{ .value = r.value, .next_pos = p + r.bytes_read };
}

/// Decode an unsigned LEB128 from `bytes[pos.*..]`, advance `pos` past it,
/// and return the value. Returns 0 on malformed or truncated input.
pub inline fn readUnsignedLossy(comptime T: type, bytes: []const u8, pos: *usize) T {
    const p = pos.*;
    if (p < bytes.len) {
        const b0 = bytes[p];
        if (b0 < 0x80) {
            pos.* = p + 1;
            return b0;
        }
    }
    const r = readUnsignedLossySlow(T, bytes, p);
    pos.* = r.next_pos;
    return r.value;
}

/// Decode a signed LEB128 from `bytes[pos.*..]`, advance `pos` past it,
/// and return the value. Returns 0 on malformed or truncated input.
pub inline fn readSignedLossy(comptime T: type, bytes: []const u8, pos: *usize) T {
    const p = pos.*;
    if (p < bytes.len) {
        const b0 = bytes[p];
        if (b0 < 0x80) {
            pos.* = p + 1;
            const sign_extended: i8 = @as(i8, @bitCast(b0 << 1)) >> 1;
            return sign_extended;
        }
    }
    const r = readSignedLossySlow(T, bytes, p);
    pos.* = r.next_pos;
    return r.value;
}

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

// -- Unsigned u32 ---------------------------------------------------------

test "u32: zero" {
    const r = try readUnsigned(u32, &.{0x00});
    try testing.expectEqual(@as(u32, 0), r.value);
    try testing.expectEqual(@as(usize, 1), r.bytes_read);
}

test "u32: one" {
    const r = try readUnsigned(u32, &.{0x01});
    try testing.expectEqual(@as(u32, 1), r.value);
    try testing.expectEqual(@as(usize, 1), r.bytes_read);
}

test "u32: 127 (single byte max payload)" {
    const r = try readUnsigned(u32, &.{0x7f});
    try testing.expectEqual(@as(u32, 127), r.value);
    try testing.expectEqual(@as(usize, 1), r.bytes_read);
}

test "u32: 128 (two bytes)" {
    const r = try readUnsigned(u32, &.{ 0x80, 0x01 });
    try testing.expectEqual(@as(u32, 128), r.value);
    try testing.expectEqual(@as(usize, 2), r.bytes_read);
}

test "u32: 624485 (multi-byte example from Wikipedia)" {
    const r = try readUnsigned(u32, &.{ 0xe5, 0x8e, 0x26 });
    try testing.expectEqual(@as(u32, 624485), r.value);
    try testing.expectEqual(@as(usize, 3), r.bytes_read);
}

test "u32: max value (0xFFFFFFFF)" {
    const r = try readUnsigned(u32, &.{ 0xff, 0xff, 0xff, 0xff, 0x0f });
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), r.value);
    try testing.expectEqual(@as(usize, 5), r.bytes_read);
}

test "u32: overflow (6 bytes)" {
    const result = readUnsigned(u32, &.{ 0x80, 0x80, 0x80, 0x80, 0x80, 0x01 });
    try testing.expectError(Error.Overflow, result);
}

test "u32: overflow (high bits in 5th byte)" {
    const result = readUnsigned(u32, &.{ 0xff, 0xff, 0xff, 0xff, 0x1f });
    try testing.expectError(Error.Overflow, result);
}

test "u32: unexpected end" {
    const result = readUnsigned(u32, &.{0x80});
    try testing.expectError(Error.UnexpectedEnd, result);
}

test "u32: unexpected end (empty)" {
    const result = readUnsigned(u32, &.{});
    try testing.expectError(Error.UnexpectedEnd, result);
}

// -- Unsigned u64 ---------------------------------------------------------

test "u64: zero" {
    const r = try readUnsigned(u64, &.{0x00});
    try testing.expectEqual(@as(u64, 0), r.value);
}

test "u64: max value" {
    const r = try readUnsigned(u64, &.{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01 });
    try testing.expectEqual(@as(u64, 0xFFFFFFFFFFFFFFFF), r.value);
    try testing.expectEqual(@as(usize, 10), r.bytes_read);
}

test "u64: overflow (11 bytes)" {
    const result = readUnsigned(u64, &.{ 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01 });
    try testing.expectError(Error.Overflow, result);
}

test "u64: unexpected end" {
    const result = readUnsigned(u64, &.{ 0x80, 0x80 });
    try testing.expectError(Error.UnexpectedEnd, result);
}

// -- Signed i32 -----------------------------------------------------------

test "i32: zero" {
    const r = try readSigned(i32, &.{0x00});
    try testing.expectEqual(@as(i32, 0), r.value);
    try testing.expectEqual(@as(usize, 1), r.bytes_read);
}

test "i32: one" {
    const r = try readSigned(i32, &.{0x01});
    try testing.expectEqual(@as(i32, 1), r.value);
}

test "i32: minus one" {
    const r = try readSigned(i32, &.{0x7f});
    try testing.expectEqual(@as(i32, -1), r.value);
    try testing.expectEqual(@as(usize, 1), r.bytes_read);
}

test "i32: minus two" {
    const r = try readSigned(i32, &.{0x7e});
    try testing.expectEqual(@as(i32, -2), r.value);
}

test "i32: 63" {
    const r = try readSigned(i32, &.{0x3f});
    try testing.expectEqual(@as(i32, 63), r.value);
}

test "i32: -64" {
    const r = try readSigned(i32, &.{0x40});
    try testing.expectEqual(@as(i32, -64), r.value);
}

test "i32: 128" {
    const r = try readSigned(i32, &.{ 0x80, 0x01 });
    try testing.expectEqual(@as(i32, 128), r.value);
}

test "i32: -128" {
    const r = try readSigned(i32, &.{ 0x80, 0x7f });
    try testing.expectEqual(@as(i32, -128), r.value);
}

test "i32: max (2147483647)" {
    const r = try readSigned(i32, &.{ 0xff, 0xff, 0xff, 0xff, 0x07 });
    try testing.expectEqual(@as(i32, 2147483647), r.value);
    try testing.expectEqual(@as(usize, 5), r.bytes_read);
}

test "i32: min (-2147483648)" {
    const r = try readSigned(i32, &.{ 0x80, 0x80, 0x80, 0x80, 0x78 });
    try testing.expectEqual(@as(i32, -2147483648), r.value);
    try testing.expectEqual(@as(usize, 5), r.bytes_read);
}

test "i32: overflow" {
    const result = readSigned(i32, &.{ 0x80, 0x80, 0x80, 0x80, 0x80, 0x01 });
    try testing.expectError(Error.Overflow, result);
}

test "i32: unexpected end" {
    const result = readSigned(i32, &.{0x80});
    try testing.expectError(Error.UnexpectedEnd, result);
}

// -- Signed i64 -----------------------------------------------------------

test "i64: zero" {
    const r = try readSigned(i64, &.{0x00});
    try testing.expectEqual(@as(i64, 0), r.value);
}

test "i64: minus one" {
    const r = try readSigned(i64, &.{0x7f});
    try testing.expectEqual(@as(i64, -1), r.value);
}

test "i64: max (9223372036854775807)" {
    const r = try readSigned(i64, &.{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00 });
    try testing.expectEqual(@as(i64, std.math.maxInt(i64)), r.value);
    try testing.expectEqual(@as(usize, 10), r.bytes_read);
}

test "i64: min (-9223372036854775808)" {
    const r = try readSigned(i64, &.{ 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f });
    try testing.expectEqual(@as(i64, std.math.minInt(i64)), r.value);
    try testing.expectEqual(@as(usize, 10), r.bytes_read);
}

test "i64: overflow (11 bytes)" {
    const result = readSigned(i64, &.{ 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01 });
    try testing.expectError(Error.Overflow, result);
}

test "i64: unexpected end" {
    const result = readSigned(i64, &.{ 0x80, 0x80, 0x80 });
    try testing.expectError(Error.UnexpectedEnd, result);
}

// -- Extra bytes after value are ignored ----------------------------------

test "extra trailing bytes are ignored" {
    const r = try readUnsigned(u32, &.{ 0x01, 0xFF, 0xFF });
    try testing.expectEqual(@as(u32, 1), r.value);
    try testing.expectEqual(@as(usize, 1), r.bytes_read);
}

// -- Lossy wrapper behavior ------------------------------------------------

test "lossy: valid u32 single byte" {
    var pos: usize = 0;
    const v = readUnsignedLossy(u32, &.{0x2A}, &pos);
    try testing.expectEqual(@as(u32, 42), v);
    try testing.expectEqual(@as(usize, 1), pos);
}

test "lossy: valid i32 single-byte -4 (the coremark regression case)" {
    var pos: usize = 0;
    const v = readSignedLossy(i32, &.{0x7C}, &pos);
    try testing.expectEqual(@as(i32, -4), v);
    try testing.expectEqual(@as(usize, 1), pos);
}

test "lossy: truncated input advances pos to end and returns 0" {
    var pos: usize = 0;
    const v = readUnsignedLossy(u32, &.{0x80}, &pos);
    try testing.expectEqual(@as(u32, 0), v);
    try testing.expectEqual(@as(usize, 1), pos);
}

test "lossy: overflow skips continuation bytes and returns 0" {
    var pos: usize = 0;
    const v = readUnsignedLossy(u32, &.{ 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0xAA }, &pos);
    try testing.expectEqual(@as(u32, 0), v);
    // cursor should be past all 6 continuation/terminator bytes
    try testing.expectEqual(@as(usize, 6), pos);
}

test "lossy: empty input returns 0, pos unchanged" {
    var pos: usize = 0;
    const v = readUnsignedLossy(u32, &.{}, &pos);
    try testing.expectEqual(@as(u32, 0), v);
    try testing.expectEqual(@as(usize, 0), pos);
}

test "lossy: advances correctly on back-to-back reads" {
    // Encode: u32 1, i32 -4, u32 624485.
    const buf = &[_]u8{ 0x01, 0x7C, 0xe5, 0x8e, 0x26 };
    var pos: usize = 0;
    try testing.expectEqual(@as(u32, 1), readUnsignedLossy(u32, buf, &pos));
    try testing.expectEqual(@as(i32, -4), readSignedLossy(i32, buf, &pos));
    try testing.expectEqual(@as(u32, 624485), readUnsignedLossy(u32, buf, &pos));
    try testing.expectEqual(buf.len, pos);
}

test "lossy: 2-byte and 5-byte relocation-padded u32 fast paths" {
    const buf = &[_]u8{
        0x80, 0x01, // 2-byte 128
        0x88, 0xa6, 0x80, 0x80, 0x00, // 5-byte wasm-ld padded 4872
    };
    var pos: usize = 0;
    try testing.expectEqual(@as(u32, 128), readUnsignedLossy(u32, buf, &pos));
    try testing.expectEqual(@as(usize, 2), pos);
    try testing.expectEqual(@as(u32, 4872), readUnsignedLossy(u32, buf, &pos));
    try testing.expectEqual(buf.len, pos);
}

// -- Round-trip encode/decode fuzz ----------------------------------------

/// Encode a u64 as unsigned LEB128 into `out`, returning bytes written.
fn encodeUnsigned(value: u64, out: []u8) usize {
    var v = value;
    var i: usize = 0;
    while (true) {
        var byte: u8 = @intCast(v & 0x7f);
        v >>= 7;
        if (v != 0) {
            byte |= 0x80;
            out[i] = byte;
            i += 1;
        } else {
            out[i] = byte;
            i += 1;
            return i;
        }
    }
}

/// Encode an i64 as signed LEB128 into `out`, returning bytes written.
fn encodeSigned(value: i64, out: []u8) usize {
    var v = value;
    var i: usize = 0;
    while (true) {
        const byte_val: u8 = @as(u8, @truncate(@as(u64, @bitCast(v)))) & 0x7f;
        // Arithmetic shift right to preserve sign.
        v >>= 7;
        const done = (v == 0 and (byte_val & 0x40) == 0) or
            (v == -1 and (byte_val & 0x40) != 0);
        if (done) {
            out[i] = byte_val;
            i += 1;
            return i;
        } else {
            out[i] = byte_val | 0x80;
            i += 1;
        }
    }
}

test "round-trip fuzz: u32" {
    var prng = std.Random.DefaultPrng.init(0xC0DE_0001);
    const rnd = prng.random();
    var buf: [10]u8 = undefined;
    var i: usize = 0;
    while (i < 2000) : (i += 1) {
        const v: u32 = rnd.int(u32);
        const n = encodeUnsigned(v, &buf);
        const r = try readUnsigned(u32, buf[0..n]);
        try testing.expectEqual(v, r.value);
        try testing.expectEqual(n, r.bytes_read);
    }
}

test "round-trip fuzz: u64" {
    var prng = std.Random.DefaultPrng.init(0xC0DE_0002);
    const rnd = prng.random();
    var buf: [10]u8 = undefined;
    var i: usize = 0;
    while (i < 2000) : (i += 1) {
        const v: u64 = rnd.int(u64);
        const n = encodeUnsigned(v, &buf);
        const r = try readUnsigned(u64, buf[0..n]);
        try testing.expectEqual(v, r.value);
        try testing.expectEqual(n, r.bytes_read);
    }
}

test "round-trip fuzz: i32" {
    var prng = std.Random.DefaultPrng.init(0xC0DE_0003);
    const rnd = prng.random();
    var buf: [10]u8 = undefined;
    var i: usize = 0;
    while (i < 2000) : (i += 1) {
        const v: i32 = rnd.int(i32);
        const n = encodeSigned(v, &buf);
        const r = try readSigned(i32, buf[0..n]);
        try testing.expectEqual(v, r.value);
        try testing.expectEqual(n, r.bytes_read);
    }
}

test "round-trip fuzz: i64" {
    var prng = std.Random.DefaultPrng.init(0xC0DE_0004);
    const rnd = prng.random();
    var buf: [10]u8 = undefined;
    var i: usize = 0;
    while (i < 2000) : (i += 1) {
        const v: i64 = rnd.int(i64);
        const n = encodeSigned(v, &buf);
        const r = try readSigned(i64, buf[0..n]);
        try testing.expectEqual(v, r.value);
        try testing.expectEqual(n, r.bytes_read);
    }
}

test "round-trip fuzz: i32 small values (all 1-byte and 2-byte)" {
    // Exhaustively cover every 1-byte and 2-byte signed LEB128 value, which
    // is where the frontend's sign-extension bug lived.
    var buf: [10]u8 = undefined;
    var v: i32 = -16384;
    while (v < 16384) : (v += 1) {
        const n = encodeSigned(v, &buf);
        const r = try readSigned(i32, buf[0..n]);
        try testing.expectEqual(v, r.value);
        try testing.expectEqual(n, r.bytes_read);
    }
}
