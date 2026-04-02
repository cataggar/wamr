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
pub fn readUnsigned(comptime T: type, bytes: []const u8) Error!Result(T) {
    comptime {
        if (T != u32 and T != u64) @compileError("readUnsigned only supports u32 and u64");
    }
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
pub fn readSigned(comptime T: type, bytes: []const u8) Error!Result(T) {
    comptime {
        if (T != i32 and T != i64) @compileError("readSigned only supports i32 and i64");
    }
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

    return .{ .value = @bitCast(@as(std.meta.Int(.unsigned, maxbits), @truncate(result))), .bytes_read = byte_count };
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
