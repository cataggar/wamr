//! Canonical ABI — lifting and lowering between component and core values.
//!
//! Implements the Canonical ABI specification for converting between
//! component-level interface types and core WebAssembly values.
//! See: https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md

const std = @import("std");
const ctypes = @import("types.zig");

// ── Despecialization ────────────────────────────────────────────────────────
// Converts syntactic-sugar types to their underlying representations.

/// Despecialize a type for canonical ABI processing.
/// tuple → record, enum → variant, option → variant, result → variant.
pub fn despecialize(t: ctypes.TypeDef) ctypes.TypeDef {
    return switch (t) {
        .tuple => |tup| .{ .record = .{
            .fields = tupleToFields(tup.fields),
        } },
        .enum_ => |en| .{ .variant = .{
            .cases = enumToCases(en.names),
        } },
        .option => |opt| .{ .variant = .{
            .cases = &[_]ctypes.Case{
                .{ .name = "none", .type = null },
                .{ .name = "some", .type = opt.inner },
            },
        } },
        .result => |res| .{ .variant = .{
            .cases = &[_]ctypes.Case{
                .{ .name = "ok", .type = res.ok },
                .{ .name = "error", .type = res.err },
            },
        } },
        else => t,
    };
}

fn tupleToFields(_: []const ctypes.ValType) []const ctypes.Field {
    // Tuple fields are positional; full field generation requires allocation.
    // For despecialization purposes, return empty — callers handle positionally.
    return &.{};
}

fn enumToCases(_: []const []const u8) []const ctypes.Case {
    return &.{};
}

// ── Alignment ───────────────────────────────────────────────────────────────

/// Return the byte alignment requirement for an interface value type.
pub fn alignment(t: ctypes.ValType) u32 {
    return switch (t) {
        .bool, .s8, .u8 => 1,
        .s16, .u16 => 2,
        .s32, .u32, .f32, .char => 4,
        .s64, .u64, .f64 => 8,
        .string => 4, // ptr + length (2x i32)
        .own, .borrow => 4, // handle is i32
        .list => 4, // ptr + length (2x i32)
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => 4,
    };
}

/// Return the byte size of an interface value type in linear memory.
pub fn elemSize(t: ctypes.ValType) u32 {
    return switch (t) {
        .bool, .s8, .u8 => 1,
        .s16, .u16 => 2,
        .s32, .u32, .f32, .char => 4,
        .s64, .u64, .f64 => 8,
        .string => 8, // ptr(i32) + length(i32)
        .own, .borrow => 4,
        .list => 8, // ptr(i32) + length(i32)
        .record, .variant, .tuple, .flags, .enum_, .option, .result, .type_idx => 4,
    };
}

// ── Flattening ──────────────────────────────────────────────────────────────
// Map interface types to sequences of core value types for stack passing.

/// Maximum number of flattened parameter core values before spilling to memory.
pub const MAX_FLAT_PARAMS: u32 = 16;
/// Maximum number of flattened result core values before spilling to memory.
pub const MAX_FLAT_RESULTS: u32 = 1;

/// Flatten an interface value type to a sequence of core value types.
/// Returns the core types that represent this value on the stack.
pub fn flatten(t: ctypes.ValType) []const ctypes.CoreValType {
    return switch (t) {
        .bool, .s8, .u8, .s16, .u16, .s32, .u32, .char, .own, .borrow => &.{.i32},
        .s64, .u64 => &.{.i64},
        .f32 => &.{.f32},
        .f64 => &.{.f64},
        .string => &.{ .i32, .i32 }, // ptr, len
        .list => &.{ .i32, .i32 }, // ptr, len
        .enum_, .flags => &.{.i32},
        .option, .result => &.{ .i32, .i32 }, // discriminant + payload
        .record, .variant, .tuple, .type_idx => &.{.i32}, // spill to memory
    };
}

// ── Loading from linear memory ──────────────────────────────────────────────

/// Load an interface value from linear memory at the given byte offset.
pub fn loadVal(memory: []const u8, ptr: u32, t: ctypes.ValType) !InterfaceValue {
    return switch (t) {
        .bool => .{ .bool = loadU8(memory, ptr) != 0 },
        .s8 => .{ .s8 = @bitCast(loadU8(memory, ptr)) },
        .u8 => .{ .u8 = loadU8(memory, ptr) },
        .s16 => .{ .s16 = @bitCast(loadU16(memory, ptr)) },
        .u16 => .{ .u16 = loadU16(memory, ptr) },
        .s32 => .{ .s32 = @bitCast(loadU32(memory, ptr)) },
        .u32 => .{ .u32 = loadU32(memory, ptr) },
        .s64 => .{ .s64 = @bitCast(loadU64(memory, ptr)) },
        .u64 => .{ .u64 = loadU64(memory, ptr) },
        .f32 => .{ .f32 = @bitCast(loadU32(memory, ptr)) },
        .f64 => .{ .f64 = @bitCast(loadU64(memory, ptr)) },
        .char => .{ .char = loadU32(memory, ptr) },
        .own, .borrow => .{ .handle = loadU32(memory, ptr) },
        .string => .{ .string = .{
            .ptr = loadU32(memory, ptr),
            .len = loadU32(memory, ptr + 4),
        } },
        .list => .{ .list = .{
            .ptr = loadU32(memory, ptr),
            .len = loadU32(memory, ptr + 4),
        } },
        else => .{ .handle = 0 }, // compound types need type registry for full loading
    };
}

/// Store an interface value to linear memory at the given byte offset.
pub fn storeVal(memory: []u8, ptr: u32, t: ctypes.ValType, val: InterfaceValue) void {
    switch (t) {
        .bool => storeU8(memory, ptr, if (val.bool) 1 else 0),
        .s8 => storeU8(memory, ptr, @bitCast(val.s8)),
        .u8 => storeU8(memory, ptr, val.u8),
        .s16 => storeU16(memory, ptr, @bitCast(val.s16)),
        .u16 => storeU16(memory, ptr, val.u16),
        .s32 => storeU32(memory, ptr, @bitCast(val.s32)),
        .u32 => storeU32(memory, ptr, val.u32),
        .s64 => storeU64(memory, ptr, @bitCast(val.s64)),
        .u64 => storeU64(memory, ptr, val.u64),
        .f32 => storeU32(memory, ptr, @bitCast(val.f32)),
        .f64 => storeU64(memory, ptr, @bitCast(val.f64)),
        .char => storeU32(memory, ptr, val.char),
        .own, .borrow => storeU32(memory, ptr, val.handle),
        .string => {
            storeU32(memory, ptr, val.string.ptr);
            storeU32(memory, ptr + 4, val.string.len);
        },
        .list => {
            storeU32(memory, ptr, val.list.ptr);
            storeU32(memory, ptr + 4, val.list.len);
        },
        else => {},
    }
}

// ── Flat lifting ────────────────────────────────────────────────────────────

/// Lift a flat sequence of core values to an interface value.
pub fn liftFlat(core_vals: []const u32, t: ctypes.ValType) InterfaceValue {
    if (core_vals.len == 0) return .{ .handle = 0 };
    return switch (t) {
        .bool => .{ .bool = core_vals[0] != 0 },
        .s8 => .{ .s8 = @bitCast(@as(u8, @truncate(core_vals[0]))) },
        .u8 => .{ .u8 = @truncate(core_vals[0]) },
        .s16 => .{ .s16 = @bitCast(@as(u16, @truncate(core_vals[0]))) },
        .u16 => .{ .u16 = @truncate(core_vals[0]) },
        .s32 => .{ .s32 = @bitCast(core_vals[0]) },
        .u32, .char => .{ .u32 = core_vals[0] },
        .own, .borrow => .{ .handle = core_vals[0] },
        .string => .{ .string = .{
            .ptr = core_vals[0],
            .len = if (core_vals.len > 1) core_vals[1] else 0,
        } },
        .list => .{ .list = .{
            .ptr = core_vals[0],
            .len = if (core_vals.len > 1) core_vals[1] else 0,
        } },
        else => .{ .handle = core_vals[0] },
    };
}

/// Lower an interface value to flat core values. Writes into `out`.
/// Returns the number of core values written.
pub fn lowerFlat(val: InterfaceValue, t: ctypes.ValType, out: []u32) u32 {
    switch (t) {
        .bool => {
            out[0] = if (val.bool) 1 else 0;
            return 1;
        },
        .s8 => {
            out[0] = @as(u32, @intCast(@as(u8, @bitCast(val.s8))));
            return 1;
        },
        .u8 => {
            out[0] = val.u8;
            return 1;
        },
        .s16 => {
            out[0] = @as(u32, @intCast(@as(u16, @bitCast(val.s16))));
            return 1;
        },
        .u16 => {
            out[0] = val.u16;
            return 1;
        },
        .s32 => {
            out[0] = @bitCast(val.s32);
            return 1;
        },
        .u32, .char => {
            out[0] = val.u32;
            return 1;
        },
        .own, .borrow => {
            out[0] = val.handle;
            return 1;
        },
        .string => {
            out[0] = val.string.ptr;
            if (out.len > 1) out[1] = val.string.len;
            return 2;
        },
        .list => {
            out[0] = val.list.ptr;
            if (out.len > 1) out[1] = val.list.len;
            return 2;
        },
        else => {
            out[0] = val.handle;
            return 1;
        },
    }
}

// ── Interface value representation ──────────────────────────────────────────

/// Runtime representation of a component interface value.
pub const InterfaceValue = union(enum) {
    bool: bool,
    s8: i8,
    u8: u8,
    s16: i16,
    u16: u16,
    s32: i32,
    u32: u32,
    s64: i64,
    u64: u64,
    f32: u32, // bit pattern
    f64: u64, // bit pattern
    char: u32, // Unicode code point
    handle: u32, // resource handle
    string: PtrLen,
    list: PtrLen,

    pub const PtrLen = struct {
        ptr: u32,
        len: u32,
    };
};

// ── String encoding ─────────────────────────────────────────────────────────

/// Validate and measure a UTF-8 encoded string in linear memory.
pub fn validateUtf8(memory: []const u8, ptr: u32, len: u32) bool {
    if (ptr + len > memory.len) return false;
    return std.unicode.utf8ValidateSlice(memory[ptr .. ptr + len]);
}

/// Transcode UTF-8 bytes to UTF-16LE, writing into `out`.
/// Returns the number of u16 code units written.
pub fn utf8ToUtf16(src: []const u8, out: []u16) !u32 {
    var written: u32 = 0;
    var view = std.unicode.Utf8View.initUnchecked(src);
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| {
        if (written >= out.len) return error.BufferTooSmall;
        if (cp <= 0xFFFF) {
            out[written] = @intCast(cp);
            written += 1;
        } else {
            // Surrogate pair
            if (written + 1 >= out.len) return error.BufferTooSmall;
            const adj = cp - 0x10000;
            out[written] = @intCast(0xD800 + (adj >> 10));
            out[written + 1] = @intCast(0xDC00 + (adj & 0x3FF));
            written += 2;
        }
    }
    return written;
}

// ── Memory helpers ──────────────────────────────────────────────────────────

fn loadU8(mem: []const u8, ptr: u32) u8 {
    if (ptr >= mem.len) return 0;
    return mem[ptr];
}

fn loadU16(mem: []const u8, ptr: u32) u16 {
    if (ptr + 2 > mem.len) return 0;
    return std.mem.readInt(u16, mem[ptr..][0..2], .little);
}

fn loadU32(mem: []const u8, ptr: u32) u32 {
    if (ptr + 4 > mem.len) return 0;
    return std.mem.readInt(u32, mem[ptr..][0..4], .little);
}

fn loadU64(mem: []const u8, ptr: u32) u64 {
    if (ptr + 8 > mem.len) return 0;
    return std.mem.readInt(u64, mem[ptr..][0..8], .little);
}

fn storeU8(mem: []u8, ptr: u32, val: u8) void {
    if (ptr >= mem.len) return;
    mem[ptr] = val;
}

fn storeU16(mem: []u8, ptr: u32, val: u16) void {
    if (ptr + 2 > mem.len) return;
    std.mem.writeInt(u16, mem[ptr..][0..2], val, .little);
}

fn storeU32(mem: []u8, ptr: u32, val: u32) void {
    if (ptr + 4 > mem.len) return;
    std.mem.writeInt(u32, mem[ptr..][0..4], val, .little);
}

fn storeU64(mem: []u8, ptr: u32, val: u64) void {
    if (ptr + 8 > mem.len) return;
    std.mem.writeInt(u64, mem[ptr..][0..8], val, .little);
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "alignment: primitive types" {
    try std.testing.expectEqual(@as(u32, 1), alignment(.bool));
    try std.testing.expectEqual(@as(u32, 4), alignment(.s32));
    try std.testing.expectEqual(@as(u32, 8), alignment(.f64));
    try std.testing.expectEqual(@as(u32, 4), alignment(.string));
    try std.testing.expectEqual(@as(u32, 4), alignment(.{ .own = 0 }));
}

test "elemSize: primitive types" {
    try std.testing.expectEqual(@as(u32, 1), elemSize(.bool));
    try std.testing.expectEqual(@as(u32, 4), elemSize(.u32));
    try std.testing.expectEqual(@as(u32, 8), elemSize(.string));
    try std.testing.expectEqual(@as(u32, 8), elemSize(.f64));
}

test "flatten: basic types" {
    const flat_i32 = flatten(.s32);
    try std.testing.expectEqual(@as(usize, 1), flat_i32.len);
    try std.testing.expectEqual(ctypes.CoreValType.i32, flat_i32[0]);

    const flat_str = flatten(.string);
    try std.testing.expectEqual(@as(usize, 2), flat_str.len);
}

test "loadVal/storeVal: i32 roundtrip" {
    var mem = [_]u8{0} ** 16;
    storeVal(&mem, 0, .s32, .{ .s32 = -42 });
    const val = try loadVal(&mem, 0, .s32);
    try std.testing.expectEqual(@as(i32, -42), val.s32);
}

test "loadVal/storeVal: string roundtrip" {
    var mem = [_]u8{0} ** 16;
    storeVal(&mem, 0, .string, .{ .string = .{ .ptr = 100, .len = 5 } });
    const val = try loadVal(&mem, 0, .string);
    try std.testing.expectEqual(@as(u32, 100), val.string.ptr);
    try std.testing.expectEqual(@as(u32, 5), val.string.len);
}

test "liftFlat/lowerFlat: bool roundtrip" {
    const vals = [_]u32{1};
    const lifted = liftFlat(&vals, .bool);
    try std.testing.expect(lifted.bool);

    var out: [2]u32 = undefined;
    const count = lowerFlat(lifted, .bool, &out);
    try std.testing.expectEqual(@as(u32, 1), count);
    try std.testing.expectEqual(@as(u32, 1), out[0]);
}

test "liftFlat/lowerFlat: string roundtrip" {
    const vals = [_]u32{ 200, 10 };
    const lifted = liftFlat(&vals, .string);
    try std.testing.expectEqual(@as(u32, 200), lifted.string.ptr);
    try std.testing.expectEqual(@as(u32, 10), lifted.string.len);

    var out: [2]u32 = undefined;
    const count = lowerFlat(lifted, .string, &out);
    try std.testing.expectEqual(@as(u32, 2), count);
    try std.testing.expectEqual(@as(u32, 200), out[0]);
    try std.testing.expectEqual(@as(u32, 10), out[1]);
}

test "validateUtf8: valid and invalid" {
    const mem = "Hello, world!";
    try std.testing.expect(validateUtf8(mem, 0, 13));
    try std.testing.expect(!validateUtf8(mem, 0, 100)); // out of bounds
}

test "utf8ToUtf16: basic ASCII" {
    const src = "Hi";
    var out: [4]u16 = undefined;
    const written = try utf8ToUtf16(src, &out);
    try std.testing.expectEqual(@as(u32, 2), written);
    try std.testing.expectEqual(@as(u16, 'H'), out[0]);
    try std.testing.expectEqual(@as(u16, 'i'), out[1]);
}
