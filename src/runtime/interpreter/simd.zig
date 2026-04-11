//! SIMD v128 execution support for the WebAssembly interpreter.
//!
//! Implements all ~230 SIMD opcodes (0xFD prefix) using Zig's @Vector builtins.
//! The v128 type is stored as u128 on the operand stack and reinterpreted as
//! lane vectors via @bitCast for operations.

const std = @import("std");
const types = @import("../common/types.zig");
const ExecEnv = @import("../common/exec_env.zig").ExecEnv;

// ── Lane type aliases ───────────────────────────────────────────────────

const I8x16 = @Vector(16, i8);
const U8x16 = @Vector(16, u8);
const I16x8 = @Vector(8, i16);
const U16x8 = @Vector(8, u16);
const I32x4 = @Vector(4, i32);
const U32x4 = @Vector(4, u32);
const I64x2 = @Vector(2, i64);
const U64x2 = @Vector(2, u64);
const F32x4 = @Vector(4, f32);
const F64x2 = @Vector(2, f64);

// ── Error types ─────────────────────────────────────────────────────────

pub const SimdError = error{
    Unreachable,
    StackOverflow,
    StackUnderflow,
    OutOfBoundsMemoryAccess,
    UnknownOpcode,
};

// ── Stack helpers ───────────────────────────────────────────────────────

fn pushV128(env: *ExecEnv, val: u128) SimdError!void {
    env.push(.{ .v128 = val }) catch return error.StackOverflow;
}

fn popV128(env: *ExecEnv) SimdError!u128 {
    const v = env.pop() catch return error.StackUnderflow;
    return switch (v) {
        .v128 => |val| val,
        .i32 => |val| @intCast(@as(u32, @bitCast(val))),
        .i64 => |val| @intCast(@as(u64, @bitCast(val))),
        else => 0,
    };
}

fn popI32(env: *ExecEnv) SimdError!i32 {
    return env.popI32() catch return error.StackUnderflow;
}

fn pushI32(env: *ExecEnv, val: i32) SimdError!void {
    env.pushI32(val) catch return error.StackOverflow;
}

fn pushI64(env: *ExecEnv, val: i64) SimdError!void {
    env.pushI64(val) catch return error.StackOverflow;
}

fn pushF32(env: *ExecEnv, val: f32) SimdError!void {
    env.pushF32(val) catch return error.StackOverflow;
}

fn pushF64(env: *ExecEnv, val: f64) SimdError!void {
    env.pushF64(val) catch return error.StackOverflow;
}

// ── LEB128 + bytecode helpers ───────────────────────────────────────────

fn readU32(code: []const u8, ip: *usize) u32 {
    var result: u32 = 0;
    var shift: u5 = 0;
    while (true) {
        if (ip.* >= code.len) return result;
        const byte = code[ip.*];
        ip.* += 1;
        result |= @as(u32, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        if (shift >= 28) break;
        shift +|= 7;
    }
    return result;
}

const Memarg = struct { mem_idx: u32, offset: u32 };

fn readMemarg(code: []const u8, ip: *usize) Memarg {
    const align_flags = readU32(code, ip);
    const mem_idx: u32 = if (align_flags & 0x40 != 0) readU32(code, ip) else 0;
    const offset = readU32(code, ip);
    return .{ .mem_idx = mem_idx, .offset = offset };
}

fn getMemSlice(env: *ExecEnv, ma: Memarg, size: u64) SimdError![]u8 {
    const base: u32 = @bitCast(popI32(env) catch return error.StackUnderflow);
    const addr = @as(u64, base) + ma.offset;
    const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
    if (addr + size > mem.data.len) return error.OutOfBoundsMemoryAccess;
    const a: usize = @intCast(addr);
    return mem.data[a..][0..@intCast(size)];
}

// ── NaN canonicalization (shared with interp.zig) ───────────────────────

inline fn canonF32(val: f32) f32 {
    return if (std.math.isNan(val)) @as(f32, @bitCast(@as(u32, 0x7FC00000))) else val;
}
inline fn canonF64(val: f64) f64 {
    return if (std.math.isNan(val)) @as(f64, @bitCast(@as(u64, 0x7FF8000000000000))) else val;
}

inline fn wasmMinF32(a: f32, b: f32) f32 {
    if (std.math.isNan(a) or std.math.isNan(b)) return canonF32(std.math.nan(f32));
    if (a == b) return @bitCast(@as(u32, @bitCast(a)) | @as(u32, @bitCast(b)));
    return @min(a, b);
}
inline fn wasmMaxF32(a: f32, b: f32) f32 {
    if (std.math.isNan(a) or std.math.isNan(b)) return canonF32(std.math.nan(f32));
    if (a == b) return @bitCast(@as(u32, @bitCast(a)) & @as(u32, @bitCast(b)));
    return @max(a, b);
}
inline fn wasmMinF64(a: f64, b: f64) f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return canonF64(std.math.nan(f64));
    if (a == b) return @bitCast(@as(u64, @bitCast(a)) | @as(u64, @bitCast(b)));
    return @min(a, b);
}
inline fn wasmMaxF64(a: f64, b: f64) f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return canonF64(std.math.nan(f64));
    if (a == b) return @bitCast(@as(u64, @bitCast(a)) & @as(u64, @bitCast(b)));
    return @max(a, b);
}

inline fn wasmNearestF32(x: f32) f32 {
    if (std.math.isNan(x)) return canonF32(x);
    const mag = @abs(x);
    if (mag == 0.0 or mag >= 0x1.0p23) return x;
    const magic: f32 = 0x1.0p23;
    const result = (mag + magic) - magic;
    return std.math.copysign(result, x);
}
inline fn wasmNearestF64(x: f64) f64 {
    if (std.math.isNan(x)) return canonF64(x);
    const mag = @abs(x);
    if (mag == 0.0 or mag >= 0x1.0p52) return x;
    const magic: f64 = 0x1.0p52;
    const result = (mag + magic) - magic;
    return std.math.copysign(result, x);
}

// ── Main SIMD dispatch ──────────────────────────────────────────────────

pub fn executeSIMD(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const sub_op = readU32(code, ip);
    switch (sub_op) {
        // ── Memory loads ────────────────────────────────────────────
        0x00 => { // v128.load
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 16);
            try pushV128(env, std.mem.readInt(u128, slice[0..16], .little));
        },
        0x01 => { // v128.load8x8_s
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 8);
            var result: I16x8 = undefined;
            for (0..8) |i| result[i] = @as(i16, @as(i8, @bitCast(slice[i])));
            try pushV128(env, @bitCast(result));
        },
        0x02 => { // v128.load8x8_u
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 8);
            var result: U16x8 = undefined;
            for (0..8) |i| result[i] = @as(u16, slice[i]);
            try pushV128(env, @bitCast(result));
        },
        0x03 => { // v128.load16x4_s
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 8);
            var result: I32x4 = undefined;
            for (0..4) |i| result[i] = @as(i32, std.mem.readInt(i16, slice[i * 2 ..][0..2], .little));
            try pushV128(env, @bitCast(result));
        },
        0x04 => { // v128.load16x4_u
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 8);
            var result: U32x4 = undefined;
            for (0..4) |i| result[i] = @as(u32, std.mem.readInt(u16, slice[i * 2 ..][0..2], .little));
            try pushV128(env, @bitCast(result));
        },
        0x05 => { // v128.load32x2_s
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 8);
            var result: I64x2 = undefined;
            for (0..2) |i| result[i] = @as(i64, std.mem.readInt(i32, slice[i * 4 ..][0..4], .little));
            try pushV128(env, @bitCast(result));
        },
        0x06 => { // v128.load32x2_u
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 8);
            var result: U64x2 = undefined;
            for (0..2) |i| result[i] = @as(u64, std.mem.readInt(u32, slice[i * 4 ..][0..4], .little));
            try pushV128(env, @bitCast(result));
        },
        0x07 => { // v128.load8_splat
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 1);
            try pushV128(env, @bitCast(@as(U8x16, @splat(slice[0]))));
        },
        0x08 => { // v128.load16_splat
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 2);
            const val = std.mem.readInt(u16, slice[0..2], .little);
            try pushV128(env, @bitCast(@as(U16x8, @splat(val))));
        },
        0x09 => { // v128.load32_splat
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 4);
            const val = std.mem.readInt(u32, slice[0..4], .little);
            try pushV128(env, @bitCast(@as(U32x4, @splat(val))));
        },
        0x0A => { // v128.load64_splat
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 8);
            const val = std.mem.readInt(u64, slice[0..8], .little);
            try pushV128(env, @bitCast(@as(U64x2, @splat(val))));
        },

        // ── Memory store ────────────────────────────────────────────
        0x0B => { // v128.store
            const ma = readMemarg(code, ip);
            const val = try popV128(env);
            const base: u32 = @bitCast(try popI32(env));
            const addr = @as(u64, base) + ma.offset;
            const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
            if (addr + 16 > mem.data.len) return error.OutOfBoundsMemoryAccess;
            const a: usize = @intCast(addr);
            std.mem.writeInt(u128, mem.data[a..][0..16], val, .little);
        },

        // ── v128.const ──────────────────────────────────────────────
        0x0C => { // v128.const
            if (ip.* + 16 > code.len) return error.Unreachable;
            const val = std.mem.readInt(u128, code[ip.*..][0..16], .little);
            ip.* += 16;
            try pushV128(env, val);
        },

        // ── Shuffle / Swizzle ───────────────────────────────────────
        0x0D => { // i8x16.shuffle
            if (ip.* + 16 > code.len) return error.Unreachable;
            const lanes = code[ip.*..][0..16];
            ip.* += 16;
            const b: U8x16 = @bitCast(try popV128(env));
            const a: U8x16 = @bitCast(try popV128(env));
            var result: U8x16 = undefined;
            for (0..16) |i| {
                const idx = lanes[i];
                result[i] = if (idx < 16) a[idx] else if (idx < 32) b[idx - 16] else 0;
            }
            try pushV128(env, @bitCast(result));
        },
        0x0E => { // i8x16.swizzle
            const indices: U8x16 = @bitCast(try popV128(env));
            const a: U8x16 = @bitCast(try popV128(env));
            var result: U8x16 = undefined;
            for (0..16) |i| {
                result[i] = if (indices[i] < 16) a[indices[i]] else 0;
            }
            try pushV128(env, @bitCast(result));
        },

        // ── Splat ───────────────────────────────────────────────────
        0x0F => { // i8x16.splat
            const val: u8 = @truncate(@as(u32, @bitCast(try popI32(env))));
            try pushV128(env, @bitCast(@as(U8x16, @splat(val))));
        },
        0x10 => { // i16x8.splat
            const val: u16 = @truncate(@as(u32, @bitCast(try popI32(env))));
            try pushV128(env, @bitCast(@as(U16x8, @splat(val))));
        },
        0x11 => { // i32x4.splat
            const val = try popI32(env);
            try pushV128(env, @bitCast(@as(I32x4, @splat(val))));
        },
        0x12 => { // i64x2.splat
            const val = env.popI64() catch return error.StackUnderflow;
            try pushV128(env, @bitCast(@as(I64x2, @splat(val))));
        },
        0x13 => { // f32x4.splat
            const val = env.popF32() catch return error.StackUnderflow;
            try pushV128(env, @bitCast(@as(F32x4, @splat(val))));
        },
        0x14 => { // f64x2.splat
            const val = env.popF64() catch return error.StackUnderflow;
            try pushV128(env, @bitCast(@as(F64x2, @splat(val))));
        },

        // ── Lane extract/replace ────────────────────────────────────
        0x15 => try extractLaneI8s(env, code, ip),
        0x16 => try extractLaneI8u(env, code, ip),
        0x17 => try replaceLaneI8(env, code, ip),
        0x18 => try extractLaneI16s(env, code, ip),
        0x19 => try extractLaneI16u(env, code, ip),
        0x1A => try replaceLaneI16(env, code, ip),
        0x1B => try extractLaneI32(env, code, ip),
        0x1C => try replaceLaneI32(env, code, ip),
        0x1D => try extractLaneI64(env, code, ip),
        0x1E => try replaceLaneI64(env, code, ip),
        0x1F => try extractLaneF32(env, code, ip),
        0x20 => try replaceLaneF32(env, code, ip),
        0x21 => try extractLaneF64(env, code, ip),
        0x22 => try replaceLaneF64(env, code, ip),

        // ── Comparisons ─────────────────────────────────────────────
        // i8x16 cmp
        0x23 => try cmpOp(I8x16, env, .eq),
        0x24 => try cmpOp(I8x16, env, .neq),
        0x25 => try cmpOp(I8x16, env, .lt),
        0x26 => try cmpOp(U8x16, env, .lt),
        0x27 => try cmpOp(I8x16, env, .gt),
        0x28 => try cmpOp(U8x16, env, .gt),
        0x29 => try cmpOp(I8x16, env, .lte),
        0x2A => try cmpOp(U8x16, env, .lte),
        0x2B => try cmpOp(I8x16, env, .gte),
        0x2C => try cmpOp(U8x16, env, .gte),
        // i16x8 cmp
        0x2D => try cmpOp(I16x8, env, .eq),
        0x2E => try cmpOp(I16x8, env, .neq),
        0x2F => try cmpOp(I16x8, env, .lt),
        0x30 => try cmpOp(U16x8, env, .lt),
        0x31 => try cmpOp(I16x8, env, .gt),
        0x32 => try cmpOp(U16x8, env, .gt),
        0x33 => try cmpOp(I16x8, env, .lte),
        0x34 => try cmpOp(U16x8, env, .lte),
        0x35 => try cmpOp(I16x8, env, .gte),
        0x36 => try cmpOp(U16x8, env, .gte),
        // i32x4 cmp
        0x37 => try cmpOp(I32x4, env, .eq),
        0x38 => try cmpOp(I32x4, env, .neq),
        0x39 => try cmpOp(I32x4, env, .lt),
        0x3A => try cmpOp(U32x4, env, .lt),
        0x3B => try cmpOp(I32x4, env, .gt),
        0x3C => try cmpOp(U32x4, env, .gt),
        0x3D => try cmpOp(I32x4, env, .lte),
        0x3E => try cmpOp(U32x4, env, .lte),
        0x3F => try cmpOp(I32x4, env, .gte),
        0x40 => try cmpOp(U32x4, env, .gte),
        // f32x4 cmp
        0x41 => try cmpOp(F32x4, env, .eq),
        0x42 => try cmpOp(F32x4, env, .neq),
        0x43 => try cmpOp(F32x4, env, .lt),
        0x44 => try cmpOp(F32x4, env, .gt),
        0x45 => try cmpOp(F32x4, env, .lte),
        0x46 => try cmpOp(F32x4, env, .gte),
        // f64x2 cmp
        0x47 => try cmpOp(F64x2, env, .eq),
        0x48 => try cmpOp(F64x2, env, .neq),
        0x49 => try cmpOp(F64x2, env, .lt),
        0x4A => try cmpOp(F64x2, env, .gt),
        0x4B => try cmpOp(F64x2, env, .lte),
        0x4C => try cmpOp(F64x2, env, .gte),

        // ── v128 bitwise ────────────────────────────────────────────
        0x4D => { // v128.not
            const a = try popV128(env);
            try pushV128(env, ~a);
        },
        0x4E => { // v128.and
            const b = try popV128(env);
            const a = try popV128(env);
            try pushV128(env, a & b);
        },
        0x4F => { // v128.andnot
            const b = try popV128(env);
            const a = try popV128(env);
            try pushV128(env, a & ~b);
        },
        0x50 => { // v128.or
            const b = try popV128(env);
            const a = try popV128(env);
            try pushV128(env, a | b);
        },
        0x51 => { // v128.xor
            const b = try popV128(env);
            const a = try popV128(env);
            try pushV128(env, a ^ b);
        },
        0x52 => { // v128.bitselect
            const c = try popV128(env);
            const b = try popV128(env);
            const a = try popV128(env);
            try pushV128(env, (a & c) | (b & ~c));
        },
        0x53 => { // v128.any_true
            const a = try popV128(env);
            try pushI32(env, @intFromBool(a != 0));
        },

        // ── Load/store lane ─────────────────────────────────────────
        0x54 => try loadLane(env, code, ip, 1),   // v128.load8_lane
        0x55 => try loadLane(env, code, ip, 2),   // v128.load16_lane
        0x56 => try loadLane(env, code, ip, 4),   // v128.load32_lane
        0x57 => try loadLane(env, code, ip, 8),   // v128.load64_lane
        0x58 => try storeLane(env, code, ip, 1),  // v128.store8_lane
        0x59 => try storeLane(env, code, ip, 2),  // v128.store16_lane
        0x5A => try storeLane(env, code, ip, 4),  // v128.store32_lane
        0x5B => try storeLane(env, code, ip, 8),  // v128.store64_lane
        0x5C => { // v128.load32_zero
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 4);
            var bytes: [16]u8 = .{0} ** 16;
            @memcpy(bytes[0..4], slice[0..4]);
            try pushV128(env, std.mem.readInt(u128, &bytes, .little));
        },
        0x5D => { // v128.load64_zero
            const ma = readMemarg(code, ip);
            const slice = try getMemSlice(env, ma, 8);
            var bytes: [16]u8 = .{0} ** 16;
            @memcpy(bytes[0..8], slice[0..8]);
            try pushV128(env, std.mem.readInt(u128, &bytes, .little));
        },

        // ── Float conversion ────────────────────────────────────────
        0x5E => try f32x4DemoteF64x2Zero(env),
        0x5F => try f64x2PromoteLowF32x4(env),

        // ── i8x16 operations ────────────────────────────────────────
        0x60 => try unaryOp(I8x16, env, .abs),
        0x61 => try unaryOp(I8x16, env, .neg),
        0x62 => try i8x16Popcnt(env),
        0x63 => try allTrue(I8x16, env),
        0x64 => try bitmask(I8x16, env),
        0x65 => try narrowOp(I16x8, I8x16, env, true),   // i8x16.narrow_i16x8_s
        0x66 => try narrowOp(I16x8, I8x16, env, false),  // i8x16.narrow_i16x8_u
        0x67 => try f32x4Unary(env, .ceil),
        0x68 => try f32x4Unary(env, .floor),
        0x69 => try f32x4Unary(env, .trunc),
        0x6A => try f32x4Unary(env, .nearest),
        0x6B => try shiftOp(I8x16, U8x16, env, .shl),
        0x6C => try shiftOp(I8x16, I8x16, env, .shr),
        0x6D => try shiftOp(I8x16, U8x16, env, .shr),
        0x6E => try binOp(I8x16, env, .add),
        0x6F => try satBinOp(I8x16, env, .add_sat),
        0x70 => try satBinOp(U8x16, env, .add_sat),
        0x71 => try binOp(I8x16, env, .sub),
        0x72 => try satBinOp(I8x16, env, .sub_sat),
        0x73 => try satBinOp(U8x16, env, .sub_sat),
        0x74 => try f64x2Unary(env, .ceil),
        0x75 => try f64x2Unary(env, .floor),
        0x76 => try binOp(I8x16, env, .min),
        0x77 => try binOp(U8x16, env, .min),
        0x78 => try binOp(I8x16, env, .max),
        0x79 => try binOp(U8x16, env, .max),
        0x7A => try f64x2Unary(env, .trunc),
        0x7B => try avgr(U8x16, env),
        0x7C => try extaddPairwise(I8x16, I16x8, env, true),
        0x7D => try extaddPairwise(U8x16, U16x8, env, false),
        0x7E => try extaddPairwise(I16x8, I32x4, env, true),
        0x7F => try extaddPairwise(U16x8, U32x4, env, false),

        // ── i16x8 operations ────────────────────────────────────────
        0x80 => try unaryOp(I16x8, env, .abs),
        0x81 => try unaryOp(I16x8, env, .neg),
        0x82 => try q15mulrSatS(env),
        0x83 => try allTrue(I16x8, env),
        0x84 => try bitmask(I16x8, env),
        0x85 => try narrowOp(I32x4, I16x8, env, true),
        0x86 => try narrowOp(I32x4, I16x8, env, false),
        0x87 => try extendOp(I8x16, I16x8, env, .low, true),
        0x88 => try extendOp(I8x16, I16x8, env, .high, true),
        0x89 => try extendOp(U8x16, U16x8, env, .low, false),
        0x8A => try extendOp(U8x16, U16x8, env, .high, false),
        0x8B => try shiftOp(I16x8, U16x8, env, .shl),
        0x8C => try shiftOp(I16x8, I16x8, env, .shr),
        0x8D => try shiftOp(I16x8, U16x8, env, .shr),
        0x8E => try binOp(I16x8, env, .add),
        0x8F => try satBinOp(I16x8, env, .add_sat),
        0x90 => try satBinOp(U16x8, env, .add_sat),
        0x91 => try binOp(I16x8, env, .sub),
        0x92 => try satBinOp(I16x8, env, .sub_sat),
        0x93 => try satBinOp(U16x8, env, .sub_sat),
        0x94 => try f64x2Unary(env, .nearest),
        0x95 => try binOp(I16x8, env, .mul),
        0x96 => try binOp(I16x8, env, .min),
        0x97 => try binOp(U16x8, env, .min),
        0x98 => try binOp(I16x8, env, .max),
        0x99 => try binOp(U16x8, env, .max),
        // 0x9A placeholder
        0x9B => try avgr(U16x8, env),
        0x9C => try extmulOp(I8x16, I16x8, env, .low, true),
        0x9D => try extmulOp(I8x16, I16x8, env, .high, true),
        0x9E => try extmulOp(U8x16, U16x8, env, .low, false),
        0x9F => try extmulOp(U8x16, U16x8, env, .high, false),

        // ── i32x4 operations ────────────────────────────────────────
        0xA0 => try unaryOp(I32x4, env, .abs),
        0xA1 => try unaryOp(I32x4, env, .neg),
        // 0xA2 placeholder
        0xA3 => try allTrue(I32x4, env),
        0xA4 => try bitmask(I32x4, env),
        // 0xA5-0xA6 placeholder
        0xA7 => try extendOp(I16x8, I32x4, env, .low, true),
        0xA8 => try extendOp(I16x8, I32x4, env, .high, true),
        0xA9 => try extendOp(U16x8, U32x4, env, .low, false),
        0xAA => try extendOp(U16x8, U32x4, env, .high, false),
        0xAB => try shiftOp(I32x4, U32x4, env, .shl),
        0xAC => try shiftOp(I32x4, I32x4, env, .shr),
        0xAD => try shiftOp(I32x4, U32x4, env, .shr),
        0xAE => try binOp(I32x4, env, .add),
        // 0xAF-0xB0 placeholder
        0xB1 => try binOp(I32x4, env, .sub),
        // 0xB2-0xB4 placeholder
        0xB5 => try binOp(I32x4, env, .mul),
        0xB6 => try binOp(I32x4, env, .min),
        0xB7 => try binOp(U32x4, env, .min),
        0xB8 => try binOp(I32x4, env, .max),
        0xB9 => try binOp(U32x4, env, .max),
        0xBA => try i32x4DotI16x8S(env),
        // 0xBB placeholder
        0xBC => try extmulOp(I16x8, I32x4, env, .low, true),
        0xBD => try extmulOp(I16x8, I32x4, env, .high, true),
        0xBE => try extmulOp(U16x8, U32x4, env, .low, false),
        0xBF => try extmulOp(U16x8, U32x4, env, .high, false),

        // ── i64x2 operations ────────────────────────────────────────
        0xC0 => try unaryOp(I64x2, env, .abs),
        0xC1 => try unaryOp(I64x2, env, .neg),
        // 0xC2 placeholder
        0xC3 => try allTrue(I64x2, env),
        0xC4 => try bitmask(I64x2, env),
        // 0xC5-0xC6 placeholder
        0xC7 => try extendOp(I32x4, I64x2, env, .low, true),
        0xC8 => try extendOp(I32x4, I64x2, env, .high, true),
        0xC9 => try extendOp(U32x4, U64x2, env, .low, false),
        0xCA => try extendOp(U32x4, U64x2, env, .high, false),
        0xCB => try shiftOp(I64x2, U64x2, env, .shl),
        0xCC => try shiftOp(I64x2, I64x2, env, .shr),
        0xCD => try shiftOp(I64x2, U64x2, env, .shr),
        0xCE => try binOp(I64x2, env, .add),
        // 0xCF-0xD0 placeholder
        0xD1 => try binOp(I64x2, env, .sub),
        // 0xD2-0xD4 placeholder
        0xD5 => try binOp(I64x2, env, .mul),
        0xD6 => try cmpOp(I64x2, env, .eq),
        0xD7 => try cmpOp(I64x2, env, .neq),
        0xD8 => try cmpOp(I64x2, env, .lt),
        0xD9 => try cmpOp(I64x2, env, .gt),
        0xDA => try cmpOp(I64x2, env, .lte),
        0xDB => try cmpOp(I64x2, env, .gte),
        0xDC => try extmulOp(I32x4, I64x2, env, .low, true),
        0xDD => try extmulOp(I32x4, I64x2, env, .high, true),
        0xDE => try extmulOp(U32x4, U64x2, env, .low, false),
        0xDF => try extmulOp(U32x4, U64x2, env, .high, false),

        // ── f32x4 operations ────────────────────────────────────────
        0xE0 => try f32x4Unary(env, .abs),
        0xE1 => try f32x4Unary(env, .neg),
        // 0xE2 placeholder
        0xE3 => try f32x4Unary(env, .sqrt),
        0xE4 => try f32x4Binary(env, .add),
        0xE5 => try f32x4Binary(env, .sub),
        0xE6 => try f32x4Binary(env, .mul),
        0xE7 => try f32x4Binary(env, .div),
        0xE8 => try f32x4Binary(env, .min),
        0xE9 => try f32x4Binary(env, .max),
        0xEA => try f32x4Binary(env, .pmin),
        0xEB => try f32x4Binary(env, .pmax),

        // ── f64x2 operations ────────────────────────────────────────
        0xEC => try f64x2Unary(env, .abs),
        0xED => try f64x2Unary(env, .neg),
        // 0xEE placeholder
        0xEF => try f64x2Unary(env, .sqrt),
        0xF0 => try f64x2Binary(env, .add),
        0xF1 => try f64x2Binary(env, .sub),
        0xF2 => try f64x2Binary(env, .mul),
        0xF3 => try f64x2Binary(env, .div),
        0xF4 => try f64x2Binary(env, .min),
        0xF5 => try f64x2Binary(env, .max),
        0xF6 => try f64x2Binary(env, .pmin),
        0xF7 => try f64x2Binary(env, .pmax),

        // ── Conversions ─────────────────────────────────────────────
        0xF8 => try i32x4TruncSatF32x4(env, true),   // i32x4.trunc_sat_f32x4_s
        0xF9 => try i32x4TruncSatF32x4(env, false),  // i32x4.trunc_sat_f32x4_u
        0xFA => try f32x4ConvertI32x4(env, true),     // f32x4.convert_i32x4_s
        0xFB => try f32x4ConvertI32x4(env, false),    // f32x4.convert_i32x4_u
        0xFC => try i32x4TruncSatF64x2Zero(env, true),  // i32x4.trunc_sat_f64x2_s_zero
        0xFD => try i32x4TruncSatF64x2Zero(env, false), // i32x4.trunc_sat_f64x2_u_zero
        0xFE => try f64x2ConvertLowI32x4(env, true),    // f64x2.convert_low_i32x4_s
        0xFF => try f64x2ConvertLowI32x4(env, false),   // f64x2.convert_low_i32x4_u

        else => return error.UnknownOpcode,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Helper implementations
// ═══════════════════════════════════════════════════════════════════════

// ── Lane extract/replace ────────────────────────────────────────────────

fn extractLaneI8s(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u4 = @intCast(code[ip.*] & 0xF);
    ip.* += 1;
    const v: I8x16 = @bitCast(try popV128(env));
    try pushI32(env, @as(i32, v[lane]));
}

fn extractLaneI8u(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u4 = @intCast(code[ip.*] & 0xF);
    ip.* += 1;
    const v: U8x16 = @bitCast(try popV128(env));
    try pushI32(env, @as(i32, @intCast(v[lane])));
}

fn replaceLaneI8(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u4 = @intCast(code[ip.*] & 0xF);
    ip.* += 1;
    const val: u8 = @truncate(@as(u32, @bitCast(try popI32(env))));
    var v: U8x16 = @bitCast(try popV128(env));
    v[lane] = val;
    try pushV128(env, @bitCast(v));
}

fn extractLaneI16s(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u3 = @intCast(code[ip.*] & 0x7);
    ip.* += 1;
    const v: I16x8 = @bitCast(try popV128(env));
    try pushI32(env, @as(i32, v[lane]));
}

fn extractLaneI16u(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u3 = @intCast(code[ip.*] & 0x7);
    ip.* += 1;
    const v: U16x8 = @bitCast(try popV128(env));
    try pushI32(env, @as(i32, @intCast(v[lane])));
}

fn replaceLaneI16(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u3 = @intCast(code[ip.*] & 0x7);
    ip.* += 1;
    const val: u16 = @truncate(@as(u32, @bitCast(try popI32(env))));
    var v: U16x8 = @bitCast(try popV128(env));
    v[lane] = val;
    try pushV128(env, @bitCast(v));
}

fn extractLaneI32(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u2 = @intCast(code[ip.*] & 0x3);
    ip.* += 1;
    const v: I32x4 = @bitCast(try popV128(env));
    try pushI32(env, v[lane]);
}

fn replaceLaneI32(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u2 = @intCast(code[ip.*] & 0x3);
    ip.* += 1;
    const val = try popI32(env);
    var v: I32x4 = @bitCast(try popV128(env));
    v[lane] = val;
    try pushV128(env, @bitCast(v));
}

fn extractLaneI64(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u1 = @intCast(code[ip.*] & 0x1);
    ip.* += 1;
    const v: I64x2 = @bitCast(try popV128(env));
    try pushI64(env, v[lane]);
}

fn replaceLaneI64(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u1 = @intCast(code[ip.*] & 0x1);
    ip.* += 1;
    const val = env.popI64() catch return error.StackUnderflow;
    var v: I64x2 = @bitCast(try popV128(env));
    v[lane] = val;
    try pushV128(env, @bitCast(v));
}

fn extractLaneF32(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u2 = @intCast(code[ip.*] & 0x3);
    ip.* += 1;
    const v: F32x4 = @bitCast(try popV128(env));
    try pushF32(env, v[lane]);
}

fn replaceLaneF32(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u2 = @intCast(code[ip.*] & 0x3);
    ip.* += 1;
    const val = env.popF32() catch return error.StackUnderflow;
    var v: F32x4 = @bitCast(try popV128(env));
    v[lane] = val;
    try pushV128(env, @bitCast(v));
}

fn extractLaneF64(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u1 = @intCast(code[ip.*] & 0x1);
    ip.* += 1;
    const v: F64x2 = @bitCast(try popV128(env));
    try pushF64(env, v[lane]);
}

fn replaceLaneF64(env: *ExecEnv, code: []const u8, ip: *usize) SimdError!void {
    const lane: u1 = @intCast(code[ip.*] & 0x1);
    ip.* += 1;
    const val = env.popF64() catch return error.StackUnderflow;
    var v: F64x2 = @bitCast(try popV128(env));
    v[lane] = val;
    try pushV128(env, @bitCast(v));
}

// ── Load/store lane ─────────────────────────────────────────────────────

fn loadLane(env: *ExecEnv, code: []const u8, ip: *usize, comptime byte_width: comptime_int) SimdError!void {
    const ma = readMemarg(code, ip);
    const lane_idx = code[ip.*];
    ip.* += 1;
    var v: [16]u8 = @bitCast(try popV128(env));
    const slice = try getMemSlice(env, ma, byte_width);
    @memcpy(v[lane_idx * byte_width ..][0..byte_width], slice[0..byte_width]);
    try pushV128(env, @bitCast(v));
}

fn storeLane(env: *ExecEnv, code: []const u8, ip: *usize, comptime byte_width: comptime_int) SimdError!void {
    const ma = readMemarg(code, ip);
    const lane_idx = code[ip.*];
    ip.* += 1;
    const v: [16]u8 = @bitCast(try popV128(env));
    const base: u32 = @bitCast(popI32(env) catch return error.StackUnderflow);
    const addr = @as(u64, base) + ma.offset;
    const mem = env.module_inst.getMemory(ma.mem_idx) orelse return error.OutOfBoundsMemoryAccess;
    if (addr + byte_width > mem.data.len) return error.OutOfBoundsMemoryAccess;
    const a: usize = @intCast(addr);
    @memcpy(mem.data[a..][0..byte_width], v[lane_idx * byte_width ..][0..byte_width]);
}

// ── Generic comparison ──────────────────────────────────────────────────

const CmpKind = enum { eq, neq, lt, gt, lte, gte };

fn cmpOp(comptime T: type, env: *ExecEnv, comptime kind: CmpKind) SimdError!void {
    const b: T = @bitCast(try popV128(env));
    const a: T = @bitCast(try popV128(env));
    const lanes = comptime @typeInfo(T).vector.len;
    const Child = @typeInfo(T).vector.child;
    const Signed = std.meta.Int(.signed, @bitSizeOf(Child));
    const SV = @Vector(lanes, Signed);
    const mask: @Vector(lanes, bool) = switch (kind) {
        .eq => a == b,
        .neq => a != b,
        .lt => a < b,
        .gt => a > b,
        .lte => a <= b,
        .gte => a >= b,
    };
    const result: SV = @select(Signed, mask, @as(SV, @splat(-1)), @as(SV, @splat(0)));
    try pushV128(env, @bitCast(result));
}

// ── Generic unary ───────────────────────────────────────────────────────

const UnaryKind = enum { abs, neg };

fn unaryOp(comptime T: type, env: *ExecEnv, comptime kind: UnaryKind) SimdError!void {
    const a: T = @bitCast(try popV128(env));
    const result: T = switch (kind) {
        .abs => blk: {
            const Child = @typeInfo(T).vector.child;
            if (@typeInfo(Child).int.signedness == .unsigned) break :blk a;
            const lanes = @typeInfo(T).vector.len;
            var r: T = undefined;
            for (0..lanes) |i| {
                r[i] = if (a[i] == std.math.minInt(Child))
                    std.math.minInt(Child)
                else if (a[i] < 0)
                    -a[i]
                else
                    a[i];
            }
            break :blk r;
        },
        .neg => -%a,
    };
    try pushV128(env, @bitCast(result));
}

// ── Generic binary ──────────────────────────────────────────────────────

const BinKind = enum { add, sub, mul, min, max };

fn binOp(comptime T: type, env: *ExecEnv, comptime kind: BinKind) SimdError!void {
    const b: T = @bitCast(try popV128(env));
    const a: T = @bitCast(try popV128(env));
    const result: T = switch (kind) {
        .add => a +% b,
        .sub => a -% b,
        .mul => a *% b,
        .min => @min(a, b),
        .max => @max(a, b),
    };
    try pushV128(env, @bitCast(result));
}

// ── Saturating binary ───────────────────────────────────────────────────

const SatBinKind = enum { add_sat, sub_sat };

fn satBinOp(comptime T: type, env: *ExecEnv, comptime kind: SatBinKind) SimdError!void {
    const b: T = @bitCast(try popV128(env));
    const a: T = @bitCast(try popV128(env));
    const result: T = switch (kind) {
        .add_sat => a +| b,
        .sub_sat => a -| b,
    };
    try pushV128(env, @bitCast(result));
}

// ── Shift operations ────────────────────────────────────────────────────

const ShiftKind = enum { shl, shr };

fn shiftOp(comptime T: type, comptime ShiftT: type, env: *ExecEnv, comptime kind: ShiftKind) SimdError!void {
    const lanes = @typeInfo(T).vector.len;
    const bits = @bitSizeOf(@typeInfo(T).vector.child);
    const raw_shift: u32 = @bitCast(try popI32(env));
    const shift_amount = raw_shift % bits;
    const ShiftAmt = std.math.Log2Int(@typeInfo(T).vector.child);
    const s: @Vector(lanes, ShiftAmt) = @splat(@intCast(shift_amount));
    const v: ShiftT = @bitCast(try popV128(env));
    const result = switch (kind) {
        .shl => @as(T, @bitCast(v << s)),
        .shr => @as(T, @bitCast(v >> s)),
    };
    try pushV128(env, @bitCast(result));
}

// ── All true / bitmask ──────────────────────────────────────────────────

fn allTrue(comptime T: type, env: *ExecEnv) SimdError!void {
    const v: T = @bitCast(try popV128(env));
    const zero: T = @splat(0);
    const all = @reduce(.And, v != zero);
    try pushI32(env, @intFromBool(all));
}

fn bitmask(comptime T: type, env: *ExecEnv) SimdError!void {
    const v: T = @bitCast(try popV128(env));
    const lanes = @typeInfo(T).vector.len;
    var result: u32 = 0;
    for (0..lanes) |i| {
        if (v[i] < 0) result |= @as(u32, 1) << @intCast(i);
    }
    try pushI32(env, @bitCast(result));
}

// ── Popcnt ──────────────────────────────────────────────────────────────

fn i8x16Popcnt(env: *ExecEnv) SimdError!void {
    const v: U8x16 = @bitCast(try popV128(env));
    var result: U8x16 = undefined;
    for (0..16) |i| result[i] = @popCount(v[i]);
    try pushV128(env, @bitCast(result));
}

// ── Average (round up) ─────────────────────────────────────────────────

fn avgr(comptime T: type, env: *ExecEnv) SimdError!void {
    const b: T = @bitCast(try popV128(env));
    const a: T = @bitCast(try popV128(env));
    const lanes = @typeInfo(T).vector.len;
    const Child = @typeInfo(T).vector.child;
    const Wide = std.meta.Int(.unsigned, @bitSizeOf(Child) * 2);
    var result: T = undefined;
    for (0..lanes) |i| {
        result[i] = @intCast((@as(Wide, a[i]) + @as(Wide, b[i]) + 1) / 2);
    }
    try pushV128(env, @bitCast(result));
}

// ── Narrow ──────────────────────────────────────────────────────────────

fn narrowOp(comptime SrcT: type, comptime DstT: type, env: *ExecEnv, comptime signed: bool) SimdError!void {
    const b: SrcT = @bitCast(try popV128(env));
    const a: SrcT = @bitCast(try popV128(env));
    const src_lanes = @typeInfo(SrcT).vector.len;
    const DstChild = @typeInfo(DstT).vector.child;
    const dst_lanes = src_lanes * 2;
    var result: @Vector(dst_lanes, DstChild) = undefined;
    for (0..src_lanes) |i| {
        result[i] = saturateTo(DstChild, a[i], signed);
        result[src_lanes + i] = saturateTo(DstChild, b[i], signed);
    }
    try pushV128(env, @bitCast(result));
}

fn saturateTo(comptime DstChild: type, val: anytype, comptime signed: bool) DstChild {
    if (signed) {
        const lo = std.math.minInt(DstChild);
        const hi = std.math.maxInt(DstChild);
        if (val < lo) return @intCast(lo);
        if (val > hi) return @intCast(hi);
        return @intCast(val);
    } else {
        const UDst = std.meta.Int(.unsigned, @bitSizeOf(DstChild));
        const hi = std.math.maxInt(UDst);
        if (val < 0) return @bitCast(@as(UDst, 0));
        if (val > hi) return @bitCast(@as(UDst, hi));
        return @bitCast(@as(UDst, @intCast(val)));
    }
}

// ── Extend ──────────────────────────────────────────────────────────────

const Half = enum { low, high };

fn extendOp(comptime SrcT: type, comptime DstT: type, env: *ExecEnv, comptime half: Half, comptime signed: bool) SimdError!void {
    _ = signed;
    const v: SrcT = @bitCast(try popV128(env));
    const dst_lanes = @typeInfo(DstT).vector.len;
    const offset = if (half == .high) dst_lanes else 0;
    const DstChild = @typeInfo(DstT).vector.child;
    var result: @Vector(dst_lanes, DstChild) = undefined;
    for (0..dst_lanes) |i| {
        result[i] = @intCast(v[offset + i]);
    }
    try pushV128(env, @bitCast(result));
}

// ── Extmul ──────────────────────────────────────────────────────────────

fn extmulOp(comptime SrcT: type, comptime DstT: type, env: *ExecEnv, comptime half: Half, comptime signed: bool) SimdError!void {
    _ = signed;
    const bb: SrcT = @bitCast(try popV128(env));
    const aa: SrcT = @bitCast(try popV128(env));
    const dst_lanes = @typeInfo(DstT).vector.len;
    const DstChild = @typeInfo(DstT).vector.child;
    const offset = if (half == .high) dst_lanes else 0;
    var result: @Vector(dst_lanes, DstChild) = undefined;
    for (0..dst_lanes) |i| {
        const a_wide: DstChild = @intCast(aa[offset + i]);
        const b_wide: DstChild = @intCast(bb[offset + i]);
        result[i] = a_wide *% b_wide;
    }
    try pushV128(env, @bitCast(result));
}

// ── Extadd pairwise ─────────────────────────────────────────────────────

fn extaddPairwise(comptime SrcT: type, comptime DstT: type, env: *ExecEnv, comptime signed: bool) SimdError!void {
    _ = signed;
    const v: SrcT = @bitCast(try popV128(env));
    const dst_lanes = @typeInfo(DstT).vector.len;
    const DstChild = @typeInfo(DstT).vector.child;
    var result: @Vector(dst_lanes, DstChild) = undefined;
    for (0..dst_lanes) |i| {
        const a: DstChild = @intCast(v[i * 2]);
        const b: DstChild = @intCast(v[i * 2 + 1]);
        result[i] = a +% b;
    }
    try pushV128(env, @bitCast(result));
}

// ── Q15mulr sat ─────────────────────────────────────────────────────────

fn q15mulrSatS(env: *ExecEnv) SimdError!void {
    const b: I16x8 = @bitCast(try popV128(env));
    const a: I16x8 = @bitCast(try popV128(env));
    var result: I16x8 = undefined;
    for (0..8) |i| {
        const prod: i32 = @as(i32, a[i]) * @as(i32, b[i]);
        const rounded = (prod + 0x4000) >> 15;
        result[i] = @intCast(std.math.clamp(rounded, -32768, 32767));
    }
    try pushV128(env, @bitCast(result));
}

// ── Dot product ─────────────────────────────────────────────────────────

fn i32x4DotI16x8S(env: *ExecEnv) SimdError!void {
    const b: I16x8 = @bitCast(try popV128(env));
    const a: I16x8 = @bitCast(try popV128(env));
    var result: I32x4 = undefined;
    for (0..4) |i| {
        const lo: i32 = @as(i32, a[i * 2]) * @as(i32, b[i * 2]);
        const hi: i32 = @as(i32, a[i * 2 + 1]) * @as(i32, b[i * 2 + 1]);
        result[i] = lo +% hi;
    }
    try pushV128(env, @bitCast(result));
}

// ── f32x4 operations ────────────────────────────────────────────────────

const F32x4UnaryKind = enum { abs, neg, sqrt, ceil, floor, trunc, nearest };

fn f32x4Unary(env: *ExecEnv, comptime kind: F32x4UnaryKind) SimdError!void {
    const v: F32x4 = @bitCast(try popV128(env));
    var result: F32x4 = undefined;
    for (0..4) |i| {
        result[i] = switch (kind) {
            .abs => @abs(v[i]),
            .neg => -v[i],
            .sqrt => canonF32(@sqrt(v[i])),
            .ceil => canonF32(@ceil(v[i])),
            .floor => canonF32(@floor(v[i])),
            .trunc => canonF32(@trunc(v[i])),
            .nearest => wasmNearestF32(v[i]),
        };
    }
    try pushV128(env, @bitCast(result));
}

const F32x4BinaryKind = enum { add, sub, mul, div, min, max, pmin, pmax };

fn f32x4Binary(env: *ExecEnv, comptime kind: F32x4BinaryKind) SimdError!void {
    const b: F32x4 = @bitCast(try popV128(env));
    const a: F32x4 = @bitCast(try popV128(env));
    var result: F32x4 = undefined;
    for (0..4) |i| {
        result[i] = switch (kind) {
            .add => canonF32(a[i] + b[i]),
            .sub => canonF32(a[i] - b[i]),
            .mul => canonF32(a[i] * b[i]),
            .div => canonF32(a[i] / b[i]),
            .min => wasmMinF32(a[i], b[i]),
            .max => wasmMaxF32(a[i], b[i]),
            .pmin => if (b[i] < a[i]) b[i] else a[i],
            .pmax => if (a[i] < b[i]) b[i] else a[i],
        };
    }
    try pushV128(env, @bitCast(result));
}

// ── f64x2 operations ────────────────────────────────────────────────────

const F64x2UnaryKind = enum { abs, neg, sqrt, ceil, floor, trunc, nearest };

fn f64x2Unary(env: *ExecEnv, comptime kind: F64x2UnaryKind) SimdError!void {
    const v: F64x2 = @bitCast(try popV128(env));
    var result: F64x2 = undefined;
    for (0..2) |i| {
        result[i] = switch (kind) {
            .abs => @abs(v[i]),
            .neg => -v[i],
            .sqrt => canonF64(@sqrt(v[i])),
            .ceil => canonF64(@ceil(v[i])),
            .floor => canonF64(@floor(v[i])),
            .trunc => canonF64(@trunc(v[i])),
            .nearest => wasmNearestF64(v[i]),
        };
    }
    try pushV128(env, @bitCast(result));
}

const F64x2BinaryKind = enum { add, sub, mul, div, min, max, pmin, pmax };

fn f64x2Binary(env: *ExecEnv, comptime kind: F64x2BinaryKind) SimdError!void {
    const b: F64x2 = @bitCast(try popV128(env));
    const a: F64x2 = @bitCast(try popV128(env));
    var result: F64x2 = undefined;
    for (0..2) |i| {
        result[i] = switch (kind) {
            .add => canonF64(a[i] + b[i]),
            .sub => canonF64(a[i] - b[i]),
            .mul => canonF64(a[i] * b[i]),
            .div => canonF64(a[i] / b[i]),
            .min => wasmMinF64(a[i], b[i]),
            .max => wasmMaxF64(a[i], b[i]),
            .pmin => if (b[i] < a[i]) b[i] else a[i],
            .pmax => if (a[i] < b[i]) b[i] else a[i],
        };
    }
    try pushV128(env, @bitCast(result));
}

// ── Conversions ─────────────────────────────────────────────────────────

fn i32x4TruncSatF32x4(env: *ExecEnv, comptime signed: bool) SimdError!void {
    const v: F32x4 = @bitCast(try popV128(env));
    if (signed) {
        var result: I32x4 = undefined;
        for (0..4) |i| {
            if (std.math.isNan(v[i])) {
                result[i] = 0;
            } else if (v[i] >= 2147483648.0) {
                result[i] = 2147483647;
            } else if (v[i] <= -2147483649.0) {
                result[i] = -2147483648;
            } else {
                result[i] = @intFromFloat(@trunc(v[i]));
            }
        }
        try pushV128(env, @bitCast(result));
    } else {
        var result: U32x4 = undefined;
        for (0..4) |i| {
            if (std.math.isNan(v[i]) or v[i] <= -1.0) {
                result[i] = 0;
            } else if (v[i] >= 4294967296.0) {
                result[i] = 4294967295;
            } else {
                result[i] = @intFromFloat(@trunc(v[i]));
            }
        }
        try pushV128(env, @bitCast(result));
    }
}

fn f32x4ConvertI32x4(env: *ExecEnv, comptime signed: bool) SimdError!void {
    if (signed) {
        const v: I32x4 = @bitCast(try popV128(env));
        var result: F32x4 = undefined;
        for (0..4) |i| result[i] = @floatFromInt(v[i]);
        try pushV128(env, @bitCast(result));
    } else {
        const v: U32x4 = @bitCast(try popV128(env));
        var result: F32x4 = undefined;
        for (0..4) |i| result[i] = @floatFromInt(v[i]);
        try pushV128(env, @bitCast(result));
    }
}

fn i32x4TruncSatF64x2Zero(env: *ExecEnv, comptime signed: bool) SimdError!void {
    const v: F64x2 = @bitCast(try popV128(env));
    var result: I32x4 = .{ 0, 0, 0, 0 };
    if (signed) {
        for (0..2) |i| {
            if (std.math.isNan(v[i])) {
                result[i] = 0;
            } else {
                const clamped = std.math.clamp(v[i], -2147483648.0, 2147483647.0);
                result[i] = @intFromFloat(clamped);
            }
        }
    } else {
        const ru: U32x4 = @bitCast(result);
        var r = ru;
        for (0..2) |i| {
            if (std.math.isNan(v[i]) or v[i] < 0.0) {
                r[i] = 0;
            } else {
                const clamped = @min(v[i], 4294967295.0);
                r[i] = @intFromFloat(clamped);
            }
        }
        try pushV128(env, @bitCast(r));
        return;
    }
    try pushV128(env, @bitCast(result));
}

fn f64x2ConvertLowI32x4(env: *ExecEnv, comptime signed: bool) SimdError!void {
    if (signed) {
        const v: I32x4 = @bitCast(try popV128(env));
        const result: F64x2 = .{ @floatFromInt(v[0]), @floatFromInt(v[1]) };
        try pushV128(env, @bitCast(result));
    } else {
        const v: U32x4 = @bitCast(try popV128(env));
        const result: F64x2 = .{ @floatFromInt(v[0]), @floatFromInt(v[1]) };
        try pushV128(env, @bitCast(result));
    }
}

fn f32x4DemoteF64x2Zero(env: *ExecEnv) SimdError!void {
    const v: F64x2 = @bitCast(try popV128(env));
    const result: F32x4 = .{
        canonF32(@floatCast(v[0])),
        canonF32(@floatCast(v[1])),
        0.0,
        0.0,
    };
    try pushV128(env, @bitCast(result));
}

fn f64x2PromoteLowF32x4(env: *ExecEnv) SimdError!void {
    const v: F32x4 = @bitCast(try popV128(env));
    const result: F64x2 = .{
        @as(f64, v[0]),
        @as(f64, v[1]),
    };
    try pushV128(env, @bitCast(result));
}
