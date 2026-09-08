//! Reuse unchanged scalar frame spills along uniquely reached fallthrough paths.
//! Removed loads become NOPs, preserving branch, relocation and attribution offsets.

const std = @import("std");

const nop: u32 = 0xD503201F;
const fp = 29;
const Fact = struct { offset: u32, width: u32 };
const Facts = [31]?Fact;
const empty_facts: Facts = @splat(null);

fn branchDelta(word: u32) ?i32 {
    if (word & 0x7C000000 == 0x14000000) {
        const imm: i26 = @bitCast(@as(u26, @truncate(word)));
        return @as(i32, imm) * 4;
    }
    if (word & 0xFF000010 == 0x54000000 or word & 0x7E000000 == 0x34000000) {
        const imm: i19 = @bitCast(@as(u19, @truncate(word >> 5)));
        return @as(i32, imm) * 4;
    }
    if (word & 0x7E000000 == 0x36000000) {
        const imm: i14 = @bitCast(@as(u14, @truncate(word >> 5)));
        return @as(i32, imm) * 4;
    }
    return null;
}

fn isKnownStop(word: u32) bool {
    return word & 0xFFFFFC1F == 0xD63F0000 or // BLR
        word == 0xD65F03C0 or // RET LR, the emitter's normal ABI return
        word & 0xFFE0001F == 0xD4200000; // BRK
}

fn killRegister(facts: *Facts, reg: u5) void {
    if (reg == fp) {
        facts.* = empty_facts;
    } else if (reg != 31) {
        facts[reg] = null;
    }
}

/// Run after local branches have been resolved and before tracing frame accesses.
/// The emitter supplies instruction-only code and the scalar allocator's private
/// FP-relative spill range. Its BL/BLR instructions are ABI calls to function
/// entries or VmCtx helpers, never intra-function dispatch; calls kill all facts.
/// Refuse other indirect control flow or potential literal pools rather than
/// guessing their targets or decoding data as code.
pub fn eliminate(
    allocator: std.mem.Allocator,
    code: []u8,
    spill_start: u32,
    spill_end: u32,
) !usize {
    if (code.len % 4 != 0) return error.InvalidInstructionBuffer;
    if (spill_end < spill_start) return error.InvalidSpillRange;
    if (spill_start == spill_end or code.len == 0) return 0;

    var targets = try std.DynamicBitSet.initEmpty(allocator, code.len / 4);
    defer targets.deinit();
    var offset: usize = 0;
    while (offset < code.len) : (offset += 4) {
        const word = std.mem.readInt(u32, code[offset..][0..4], .little);
        if (branchDelta(word)) |delta| {
            const target = @as(i64, @intCast(offset)) + delta;
            if (target >= 0 and target < @as(i64, @intCast(code.len))) {
                targets.set(@intCast(@divExact(target, 4)));
            }
        } else if (word & 0x1F000000 == 0x10000000 or // ADR/ADRP
            word & 0x3B000000 == 0x18000000 or // literal load
            (word & 0x1C000000 == 0x14000000 and word != nop and !isKnownStop(word)))
        {
            // Includes BR, authenticated branches and unknown branch/system
            // encodings. HINT aliases are not generally side-effect free.
            return 0;
        }
    }

    var facts = empty_facts;
    var removed: usize = 0;
    offset = 0;
    while (offset < code.len) : (offset += 4) {
        if (targets.isSet(offset / 4)) facts = empty_facts;
        const word = std.mem.readInt(u32, code[offset..][0..4], .little);
        if (word == nop) continue;
        if (branchDelta(word) != null) {
            // Only the untaken edge of a conditional branch preserves facts.
            if (word & 0x7C000000 == 0x14000000) facts = empty_facts;
            continue;
        }

        // CMP (unshifted register or immediate) only changes NZCV.
        if (word & 0x7FE0FC1F == 0x6B00001F or word & 0x7F80001F == 0x7100001F) continue;
        const rt: u5 = @truncate(word);
        // CSEL/CSINC/CSINV/CSNEG, including both W and X CSET aliases.
        if (word & 0x3FE00800 == 0x1A800000) {
            killRegister(&facts, rt);
            continue;
        }

        const top = word & 0xFFC00000;
        const load_width: u32 = switch (top) {
            0xB9400000 => 4,
            0xF9400000 => 8,
            else => 0,
        };
        const rn: u5 = @truncate(word >> 5);
        const imm = (word >> 10) & 0xFFF;
        if (load_width != 0) {
            const displacement = imm * load_width;
            if (rn == fp and rt != fp and rt != 31 and
                displacement >= spill_start and displacement + load_width <= spill_end)
            {
                if (facts[rt]) |fact| {
                    if (fact.offset == displacement and fact.width == load_width) {
                        std.mem.writeInt(u32, code[offset..][0..4], nop, .little);
                        removed += 1;
                        continue;
                    }
                }
                facts[rt] = .{ .offset = displacement, .width = load_width };
            } else {
                killRegister(&facts, rt);
            }
            continue;
        }

        const store_width: u32 = switch (top) {
            0x39000000 => 1,
            0x79000000 => 2,
            0xB9000000 => 4,
            0xF9000000 => 8,
            else => 0,
        };
        if (store_width != 0 and rn == fp) {
            const displacement = imm * store_width;
            for (&facts) |*maybe_fact| {
                if (maybe_fact.*) |fact| {
                    if (displacement < fact.offset + fact.width and fact.offset < displacement + store_width) {
                        maybe_fact.* = null;
                    }
                }
            }
            continue;
        }
        // Calls, traps, other stores (possibly aliasing the frame), writeback,
        // SP/FP changes and every unrecognized instruction invalidate all facts.
        facts = empty_facts;
    }
    return removed;
}

fn mem(top: u32, rt: u5, rn: u5, displacement: u32, width: u32) u32 {
    return top | (@divExact(displacement, width) << 10) | (@as(u32, rn) << 5) | rt;
}

const load = mem(0xF9400000, 16, fp, 248, 8);
const ret: u32 = 0xD65F03C0;

fn expectRemoved(words: []const u32, removed_indices: []const usize) !void {
    const bytes = try std.testing.allocator.alloc(u8, words.len * 4);
    defer std.testing.allocator.free(bytes);
    for (words, 0..) |word, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], word, .little);
    try std.testing.expectEqual(removed_indices.len, try eliminate(std.testing.allocator, bytes, 240, 312));
    for (words, 0..) |word, i| {
        const expected = if (std.mem.indexOfScalar(usize, removed_indices, i) != null) nop else word;
        try std.testing.expectEqual(expected, std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    }
}

test "state spill reloads survive compares, copies and conditional fallthrough" {
    try expectRemoved(&.{
        load,
        mem(0xF9000000, 16, fp, 280, 8),
        mem(0xF9000000, 0, fp, 272, 8),
        0x7100003F, // cmp w1, #0
        0x540001A1, // b.ne instruction 17
        load,
        0x6B0C021F, // cmp w16, w12
        0x1A9F97E1, // cset w1, hi
        load,
        mem(0xF9000000, 16, fp, 288, 8),
        0x7100003F,
        0x540000C1, // b.ne instruction 17
        load,
        mem(0xF9000000, 16, fp, 288, 8),
        load,
        0x7100021F, // cmp w16, #0
        nop,
        ret,
    }, &.{ 5, 8, 12, 14 });
}

test "every conditional branch kind preserves only its fallthrough facts" {
    for ([_]u32{ 0x54000040, 0xB4000040, 0x35000040, 0x36000040, 0x37000040 }) |branch| {
        try expectRemoved(&.{ load, branch, load, ret }, &.{2});
        try expectRemoved(&.{ branch, load, load, ret }, &.{});
    }
}

test "forward joins and backward loop entries discard incoming facts" {
    try expectRemoved(&.{ 0x14000003, load, nop, load, ret }, &.{});
    try expectRemoved(&.{ load, nop, 0x54FFFFE1, load, ret }, &.{});
    try expectRemoved(&.{ load, nop, 0x35FFFFE0, load, ret }, &.{});
    try expectRemoved(&.{ load, nop, 0x37FFFFE0, load, ret }, &.{});
    try expectRemoved(&.{ load, nop, 0x17FFFFFF, load, ret }, &.{});
}

test "calls returns traps and unconditional branches are barriers" {
    for ([_]u32{ 0x94000000, 0xD63F0200, ret, 0xD4200020, 0x14000001 }) |barrier| {
        try expectRemoved(&.{ load, barrier, load, ret }, &.{});
    }
    try expectRemoved(&.{ load, load, 0xD63F0200, load, load, ret }, &.{ 1, 4 });
}

test "indirect internal branches, literal pools and unknown system effects refuse the function" {
    for ([_]u32{
        0xD61F0200, // br x16
        0xD65F0200, // ret x16, not a normal ABI return
        0xD71F0A00, // authenticated branch
        0xD503233F, // paciasp, a HINT alias
        0xD5032010, // unrecognized system encoding
        0x54000010, // BC.cond, not B.cond
        0x10000010, // adr x16, .
        0x90000010, // adrp x16, .
        0x58000010, // ldr x16, literal
    }) |barrier| {
        // A late unsupported encoding must prevent earlier rewrites too.
        try expectRemoved(&.{ load, load, barrier, ret }, &.{});
    }
}

test "W and X writes alias and FP writes invalidate every cached slot" {
    for ([_]u32{
        0x1A9F07F0, // cset w16
        0x9A9F07F0, // cset x16
        0x5A800010, // csinv w16
        0xDA800410, // csneg x16
        0x1A9F07FD, // cset w29
        mem(0xF9400000, fp, fp, 0, 8),
        0x9100001F, // mov sp, x0
        0x52800010, // mov w16, #0 (unknown ALU instruction)
        0x9E660010, // fmov x16, d0
        0xA9404410, // ldp x16, x17, [x0]
    }) |write| {
        try expectRemoved(&.{ load, write, load, ret }, &.{});
    }
    try expectRemoved(&.{ load, 0x1A9F07E1, load, ret }, &.{2});
}

test "overlapping byte halfword word and doubleword stores invalidate cached loads" {
    for ([_]u32{
        mem(0x39000000, 0, fp, 255, 1),
        mem(0x79000000, 0, fp, 254, 2),
        mem(0xB9000000, 0, fp, 252, 4),
        mem(0xF9000000, 0, fp, 248, 8),
    }) |store| {
        try expectRemoved(&.{ load, store, load, ret }, &.{});
    }
    try expectRemoved(&.{ load, mem(0xF9000000, 0, fp, 240, 8), load, ret }, &.{2});
    try expectRemoved(&.{ load, mem(0x39000000, 0, fp, 256, 1), load, ret }, &.{2});
}

test "unknown and non-FP stores may alias, and writeback invalidates the base" {
    for ([_]u32{
        mem(0xF9000000, 0, 1, 248, 8),
        0xA90007A0, // stp x0, x1, [fp]
        0x3D8003A0, // str q0, [fp]
        0xF80003A0, // stur x0, [fp]
        0xF84087A0, // ldr x0, [fp], #8
        0xC89FFFA0, // stlr x0, [fp]
    }) |barrier| {
        try expectRemoved(&.{ load, barrier, load, ret }, &.{});
    }
}

test "only identical loads into the same non-special register within scalar spills are cached" {
    for ([_]u32{
        mem(0xF9400000, 16, fp, 232, 8),
        mem(0xF9400000, 16, fp, 312, 8),
        mem(0xF9400000, 16, 1, 248, 8),
        mem(0xF9400000, 31, fp, 248, 8),
        mem(0xF9400000, fp, fp, 248, 8),
    }) |uncached| {
        try expectRemoved(&.{ uncached, uncached, ret }, &.{});
    }
    const load_w = mem(0xB9400000, 16, fp, 248, 4);
    try expectRemoved(&.{ load, load_w, load, ret }, &.{});
    try expectRemoved(&.{ load_w, load_w, ret }, &.{1});
    try expectRemoved(&.{ load, mem(0xF9400000, 17, fp, 248, 8), load, ret }, &.{2});
    try expectRemoved(&.{ load, mem(0xF9400000, 16, fp, 256, 8), load, ret }, &.{});
}

test "malformed buffers and spill ranges fail explicitly" {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, load, .little);
    try std.testing.expectError(error.InvalidInstructionBuffer, eliminate(std.testing.allocator, bytes[0..3], 240, 312));
    try std.testing.expectError(error.InvalidSpillRange, eliminate(std.testing.allocator, &bytes, 312, 240));
    try std.testing.expectEqual(@as(usize, 0), try eliminate(std.testing.allocator, &bytes, 240, 240));
    try std.testing.expectEqual(@as(usize, 0), try eliminate(std.testing.allocator, bytes[0..0], 240, 312));
}
