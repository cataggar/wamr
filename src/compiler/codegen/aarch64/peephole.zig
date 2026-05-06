//! Conservative peephole optimizer for emitted AArch64 instruction words.
//!
//! The pass is intentionally local: it only rewrites patterns that are proven
//! equivalent from adjacent instruction encodings, so branch fixups can keep
//! using normal CodeBuffer offsets while code is emitted.

const std = @import("std");

pub const Options = struct {
    enabled: bool = true,
    enable_mem_pair: bool = true,
    enable_mul_add: bool = true,
    enable_conditional_select: bool = true,
};

pub const AppendResult = struct {
    replace_last: ?u32 = null,
    append_word: ?u32 = null,
};

const Width = enum { w32, x64 };
const MemKind = enum { ldr, str };

const MemUnsigned = struct {
    kind: MemKind,
    width: Width,
    rt: u5,
    rn: u5,
    offset: u12,
};

const Mul = struct {
    width: Width,
    rd: u5,
    rn: u5,
    rm: u5,
};

const AddSubReg = struct {
    is_sub: bool,
    width: Width,
    rd: u5,
    rn: u5,
    rm: u5,
};

const AddImm = struct {
    width: Width,
    rd: u5,
    rn: u5,
    imm: u12,
};

const CmpImmZero = struct {
    width: Width,
    rn: u5,
};

const Csel = struct {
    width: Width,
    rd: u5,
    rn: u5,
    rm: u5,
    cond: u4,
};

const UnaryTransform = union(enum) {
    inc: struct { tmp: u5, src: u5, width: Width },
    inv: struct { tmp: u5, src: u5, width: Width },
    neg: struct { tmp: u5, src: u5, width: Width },
};

pub fn optimizeWindow(prev2: ?u32, prev1: ?u32, new_word: u32, options: Options) AppendResult {
    if (!options.enabled) return .{ .append_word = new_word };

    if (prev1) |last| {
        if (options.enable_mem_pair) {
            if (tryFuseMemPair(last, new_word)) |fused| return .{ .replace_last = fused };
        }
        if (options.enable_mul_add) {
            if (tryFuseMulAdd(last, new_word)) |fused| return .{ .replace_last = fused };
        }
    }

    if (options.enable_conditional_select) {
        if (prev2) |transform_word| {
            if (prev1) |cmp_word| {
                if (tryConditionalTransform(transform_word, cmp_word, new_word)) |rewritten| {
                    return .{ .append_word = rewritten };
                }
            }
        }
    }

    return .{ .append_word = new_word };
}

pub fn optimizeWords(insts: []const u32, allocator: std.mem.Allocator, options: Options) ![]u32 {
    var out: std.ArrayListUnmanaged(u32) = .empty;
    errdefer out.deinit(allocator);

    for (insts) |word| {
        const prev1 = if (out.items.len >= 1) out.items[out.items.len - 1] else null;
        const prev2 = if (out.items.len >= 2) out.items[out.items.len - 2] else null;
        const r = optimizeWindow(prev2, prev1, word, options);
        if (r.replace_last) |replacement| out.items[out.items.len - 1] = replacement;
        if (r.append_word) |append| try out.append(allocator, append);
    }

    return out.toOwnedSlice(allocator);
}

fn tryFuseMemPair(a_word: u32, b_word: u32) ?u32 {
    const a = decodeMemUnsigned(a_word) orelse return null;
    const b = decodeMemUnsigned(b_word) orelse return null;
    if (a.kind != b.kind or a.width != b.width or a.rn != b.rn) return null;
    if (@as(u32, a.offset) + 1 != b.offset) return null;
    if (a.offset > 63) return null;

    if (a.kind == .ldr) {
        if (a.rt == a.rn or b.rt == a.rn or a.rt == b.rt) return null;
    }

    const imm7: u32 = a.offset;
    return switch (a.width) {
        .x64 => switch (a.kind) {
            .ldr => 0xA9400000 | (imm7 << 15) | (@as(u32, b.rt) << 10) | (@as(u32, a.rn) << 5) | a.rt,
            .str => 0xA9000000 | (imm7 << 15) | (@as(u32, b.rt) << 10) | (@as(u32, a.rn) << 5) | a.rt,
        },
        .w32 => switch (a.kind) {
            .ldr => 0x29400000 | (imm7 << 15) | (@as(u32, b.rt) << 10) | (@as(u32, a.rn) << 5) | a.rt,
            .str => 0x29000000 | (imm7 << 15) | (@as(u32, b.rt) << 10) | (@as(u32, a.rn) << 5) | a.rt,
        },
    };
}

fn tryFuseMulAdd(mul_word: u32, add_word: u32) ?u32 {
    const mul = decodeMul(mul_word) orelse return null;
    const op = decodeAddSubReg(add_word) orelse return null;
    if (mul.width != op.width) return null;

    if (op.is_sub) {
        if (op.rm != mul.rd or op.rd != mul.rd) return null;
        return encodeMAddSub(mul.width, true, op.rd, mul.rn, mul.rm, op.rn);
    }

    if (op.rd != mul.rd) return null;
    if (op.rm == mul.rd) return encodeMAddSub(mul.width, false, op.rd, mul.rn, mul.rm, op.rn);
    if (op.rn == mul.rd) return encodeMAddSub(mul.width, false, op.rd, mul.rn, mul.rm, op.rm);
    return null;
}

fn tryConditionalTransform(transform_word: u32, cmp_word: u32, csel_word: u32) ?u32 {
    _ = decodeCmpImmZero(cmp_word) orelse return null;
    const csel = decodeCsel(csel_word) orelse return null;
    const transform = decodeUnaryTransform(transform_word) orelse return null;

    return switch (transform) {
        .inc => |t| blk: {
            if (t.width != csel.width or t.tmp != csel.rm or t.src != csel.rn or t.tmp == t.src) break :blk null;
            break :blk encodeCs(.inc, csel.width, csel.rd, t.src, t.src, csel.cond);
        },
        .inv => |t| blk: {
            if (t.width != csel.width or t.tmp != csel.rm or t.src != csel.rn or t.tmp == t.src) break :blk null;
            break :blk encodeCs(.inv, csel.width, csel.rd, t.src, t.src, csel.cond);
        },
        .neg => |t| blk: {
            if (t.width != csel.width or t.tmp != csel.rm or t.src != csel.rn or t.tmp == t.src) break :blk null;
            break :blk encodeCs(.neg, csel.width, csel.rd, t.src, t.src, csel.cond);
        },
    };
}

fn decodeMemUnsigned(word: u32) ?MemUnsigned {
    const top = word & 0xFFC00000;
    const kind: MemKind, const width: Width = switch (top) {
        0xF9400000 => .{ .ldr, .x64 },
        0xF9000000 => .{ .str, .x64 },
        0xB9400000 => .{ .ldr, .w32 },
        0xB9000000 => .{ .str, .w32 },
        else => return null,
    };
    return .{
        .kind = kind,
        .width = width,
        .rt = @intCast(word & 0x1F),
        .rn = @intCast((word >> 5) & 0x1F),
        .offset = @intCast((word >> 10) & 0xFFF),
    };
}

fn decodeMul(word: u32) ?Mul {
    const width: Width = if ((word & 0xFFE0FC00) == 0x9B007C00)
        .x64
    else if ((word & 0xFFE0FC00) == 0x1B007C00)
        .w32
    else
        return null;
    return .{
        .width = width,
        .rd = @intCast(word & 0x1F),
        .rn = @intCast((word >> 5) & 0x1F),
        .rm = @intCast((word >> 16) & 0x1F),
    };
}

fn decodeAddSubReg(word: u32) ?AddSubReg {
    if ((word & 0x3FE0FC00) != 0x0B000000) return null;
    const sf = (word >> 31) & 1;
    const op = (word >> 30) & 1;
    return .{
        .is_sub = op == 1,
        .width = if (sf == 1) .x64 else .w32,
        .rd = @intCast(word & 0x1F),
        .rn = @intCast((word >> 5) & 0x1F),
        .rm = @intCast((word >> 16) & 0x1F),
    };
}

fn decodeAddImm(word: u32) ?AddImm {
    if ((word & 0x7F800000) != 0x11000000) return null;
    if (((word >> 22) & 1) != 0) return null;
    return .{
        .width = if (((word >> 31) & 1) == 1) .x64 else .w32,
        .rd = @intCast(word & 0x1F),
        .rn = @intCast((word >> 5) & 0x1F),
        .imm = @intCast((word >> 10) & 0xFFF),
    };
}

fn decodeCmpImmZero(word: u32) ?CmpImmZero {
    const width: Width = if ((word & 0xFFC003FF) == 0xF100001F)
        .x64
    else if ((word & 0xFFC003FF) == 0x7100001F)
        .w32
    else
        return null;
    if (((word >> 10) & 0xFFF) != 0) return null;
    return .{ .width = width, .rn = @intCast((word >> 5) & 0x1F) };
}

fn decodeCsel(word: u32) ?Csel {
    const width: Width = if ((word & 0xFFE00C00) == 0x9A800000)
        .x64
    else if ((word & 0xFFE00C00) == 0x1A800000)
        .w32
    else
        return null;
    return .{
        .width = width,
        .rd = @intCast(word & 0x1F),
        .rn = @intCast((word >> 5) & 0x1F),
        .rm = @intCast((word >> 16) & 0x1F),
        .cond = @intCast((word >> 12) & 0xF),
    };
}

fn decodeUnaryTransform(word: u32) ?UnaryTransform {
    if (decodeAddImm(word)) |a| {
        if (a.imm == 1) return .{ .inc = .{ .tmp = a.rd, .src = a.rn, .width = a.width } };
    }

    if ((word & 0x7FE0FC00) == 0x2A200000) {
        const width: Width = if (((word >> 31) & 1) == 1) .x64 else .w32;
        const rn: u5 = @intCast((word >> 5) & 0x1F);
        if (rn == 31) return .{ .inv = .{ .tmp = @intCast(word & 0x1F), .src = @intCast((word >> 16) & 0x1F), .width = width } };
    }

    if (decodeAddSubReg(word)) |s| {
        if (s.is_sub and s.rn == 31) return .{ .neg = .{ .tmp = s.rd, .src = s.rm, .width = s.width } };
    }

    return null;
}

fn encodeMAddSub(width: Width, is_sub: bool, rd: u5, rn: u5, rm: u5, ra: u5) u32 {
    const base: u32 = switch (width) {
        .x64 => if (is_sub) 0x9B008000 else 0x9B000000,
        .w32 => if (is_sub) 0x1B008000 else 0x1B000000,
    };
    return base | (@as(u32, rm) << 16) | (@as(u32, ra) << 10) | (@as(u32, rn) << 5) | rd;
}

const CsKind = enum { inc, inv, neg };

fn encodeCs(kind: CsKind, width: Width, rd: u5, rn: u5, rm: u5, cond: u4) u32 {
    const base: u32 = switch (width) {
        .x64 => switch (kind) {
            .inc => 0x9A800400,
            .inv => 0xDA800000,
            .neg => 0xDA800400,
        },
        .w32 => switch (kind) {
            .inc => 0x1A800400,
            .inv => 0x5A800000,
            .neg => 0x5A800400,
        },
    };
    return base | (@as(u32, rm) << 16) | (@as(u32, cond) << 12) | (@as(u32, rn) << 5) | rd;
}

fn expectOptimized(input: []const u32, expected: []const u32) !void {
    const got = try optimizeWords(input, std.testing.allocator, .{});
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualSlices(u32, expected, got);
}

test "peephole: fuses adjacent 64-bit ldr into ldp" {
    const input = [_]u32{
        0xF9401020, // ldr x0, [x1, #16]
        0xF9401422, // ldr x2, [x1, #24]
    };
    const expected = [_]u32{0xA9420820}; // ldp x0, x2, [x1, #16]
    try expectOptimized(&input, &expected);
}

test "peephole: does not fuse load pair when first load rewrites base" {
    const input = [_]u32{
        0xF9401021, // ldr x1, [x1, #16]
        0xF9401422, // ldr x2, [x1, #24]
    };
    try expectOptimized(&input, &input);
}

test "peephole: fuses adjacent 64-bit str into stp" {
    const input = [_]u32{
        0xF9001060, // str x0, [x3, #32]
        0xF9001461, // str x1, [x3, #40]
    };
    const expected = [_]u32{0xA9020460}; // stp x0, x1, [x3, #32]
    try expectOptimized(&input, &expected);
}

test "peephole: leaves non-adjacent memory offsets untouched" {
    const input = [_]u32{
        0xF9401020, // ldr x0, [x1, #16]
        0xF9401C22, // ldr x2, [x1, #56]
    };
    try expectOptimized(&input, &input);
}

test "peephole: fuses mul plus add into madd when add overwrites the mul result" {
    const input = [_]u32{
        0x9B027C20, // mul x0, x1, x2
        0x8B000060, // add x0, x3, x0
    };
    const expected = [_]u32{0x9B020C20}; // madd x0, x1, x2, x3
    try expectOptimized(&input, &expected);
}

test "peephole: does not fuse mul plus add when mul result remains live in another reg" {
    const input = [_]u32{
        0x9B027C20, // mul x0, x1, x2
        0x8B000063, // add x3, x3, x0
    };
    try expectOptimized(&input, &input);
}

test "peephole: fuses mul plus sub into msub when sub overwrites the mul result" {
    const input = [_]u32{
        0x9B027C20, // mul x0, x1, x2
        0xCB000060, // sub x0, x3, x0
    };
    const expected = [_]u32{0x9B028C20}; // msub x0, x1, x2, x3
    try expectOptimized(&input, &expected);
}

test "peephole: rewrites cmp+csel of an increment transform to csinc" {
    const input = [_]u32{
        0x91000443, // add x3, x2, #1
        0xF100001F, // cmp x0, #0
        0x9A830041, // csel x1, x2, x3, eq
    };
    const expected = [_]u32{
        0x91000443,
        0xF100001F,
        0x9A820441, // csinc x1, x2, x2, eq
    };
    try expectOptimized(&input, &expected);
}

test "peephole: rewrites cmp+csel of invert and neg transforms" {
    const input = [_]u32{
        0xAA2503E4, // mvn x4, x5
        0xF100001F, // cmp x0, #0
        0x9A8400A3, // csel x3, x5, x4, eq
        0xCB0703E6, // neg x6, x7
        0xF100001F, // cmp x0, #0
        0x9A8600E8, // csel x8, x7, x6, eq
    };
    const expected = [_]u32{
        0xAA2503E4,
        0xF100001F,
        0xDA8500A3, // csinv x3, x5, x5, eq
        0xCB0703E6,
        0xF100001F,
        0xDA8704E8, // csneg x8, x7, x7, eq
    };
    try expectOptimized(&input, &expected);
}

test "peephole: leaves cmp+csel untouched without adjacent transform" {
    const input = [_]u32{
        0xD503201F, // nop
        0xF100001F, // cmp x0, #0
        0x9A820041, // csel x1, x2, x2, eq
    };
    try expectOptimized(&input, &input);
}
