//! AArch64 Machine Code Emitter
//!
//! Translates compiler IR into AArch64 (ARM64) machine code.
//! AArch64 uses fixed 4-byte instruction encoding, which is simpler
//! than x86-64's variable-length encoding.

const std = @import("std");

/// AArch64 general-purpose registers (64-bit X registers).
pub const Reg = enum(u5) {
    x0 = 0,
    x1 = 1,
    x2 = 2,
    x3 = 3,
    x4 = 4,
    x5 = 5,
    x6 = 6,
    x7 = 7,
    x8 = 8,
    x9 = 9,
    x10 = 10,
    x11 = 11,
    x12 = 12,
    x13 = 13,
    x14 = 14,
    x15 = 15,
    x16 = 16,
    x17 = 17,
    x18 = 18,
    x19 = 19,
    x20 = 20,
    x21 = 21,
    x22 = 22,
    x23 = 23,
    x24 = 24,
    x25 = 25,
    x26 = 26,
    x27 = 27,
    x28 = 28,
    fp = 29, // frame pointer (x29)
    lr = 30, // link register (x30)
    sp = 31, // stack pointer (also encodes as XZR in some contexts)

    pub fn encoding(self: Reg) u5 {
        return @intFromEnum(self);
    }
};

/// Condition codes for B.cond instructions.
pub const Cond = enum(u4) {
    eq = 0b0000, // equal (Z=1)
    ne = 0b0001, // not equal (Z=0)
    hs = 0b0010, // unsigned higher or same (C=1)
    lo = 0b0011, // unsigned lower (C=0)
    mi = 0b0100, // negative (N=1)
    pl = 0b0101, // positive or zero (N=0)
    vs = 0b0110, // overflow (V=1)
    vc = 0b0111, // no overflow (V=0)
    hi = 0b1000, // unsigned higher (C=1 & Z=0)
    ls = 0b1001, // unsigned lower or same (C=0 | Z=1)
    ge = 0b1010, // signed greater or equal (N=V)
    lt = 0b1011, // signed less than (N≠V)
    gt = 0b1100, // signed greater than (Z=0 & N=V)
    le = 0b1101, // signed less or equal (Z=1 | N≠V)
    al = 0b1110, // always
};

/// Machine code buffer with AArch64 instruction encoding helpers.
pub const CodeBuffer = struct {
    bytes: std.ArrayListUnmanaged(u8) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CodeBuffer {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *CodeBuffer) void {
        self.bytes.deinit(self.allocator);
    }

    pub fn len(self: *const CodeBuffer) usize {
        return self.bytes.items.len;
    }

    pub fn getCode(self: *const CodeBuffer) []const u8 {
        return self.bytes.items;
    }

    /// Emit a 4-byte AArch64 instruction word (little-endian).
    fn emit32(self: *CodeBuffer, word: u32) !void {
        const bytes = std.mem.toBytes(std.mem.nativeToLittle(u32, word));
        try self.bytes.appendSlice(self.allocator, &bytes);
    }

    // ── Data Processing (Register) ──────────────────────────────────

    /// ADD Xd, Xn, Xm (64-bit register add)
    pub fn addRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        // 1|00|01011|00|0|Rm|000000|Rn|Rd
        try self.emit32(0x8B000000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SUB Xd, Xn, Xm (64-bit register sub)
    pub fn subRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        // 1|10|01011|00|0|Rm|000000|Rn|Rd
        try self.emit32(0xCB000000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// AND Xd, Xn, Xm
    pub fn andRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        // 1|00|01010|00|0|Rm|000000|Rn|Rd
        try self.emit32(0x8A000000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// AND Wd, Wn, #0x1f — mask a scalar i32x4 shift count modulo 32.
    pub fn andImm32Mask31(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        try self.emit32(0x12001000 | (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// AND Wd, Wn, #0xf — mask a scalar i16x8 shift count modulo 16.
    pub fn andImm32Mask15(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        try self.emit32(0x12000C00 | (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// ORR Xd, Xn, Xm
    pub fn orrRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        // 1|01|01010|00|0|Rm|000000|Rn|Rd
        try self.emit32(0xAA000000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// EOR Xd, Xn, Xm (exclusive or)
    pub fn eorRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        // 1|10|01010|00|0|Rm|000000|Rn|Rd
        try self.emit32(0xCA000000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// MUL Xd, Xn, Xm (alias for MADD Xd, Xn, Xm, XZR)
    pub fn mulRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        // 1|00|11011|000|Rm|0|11111|Rn|Rd  (Ra=XZR=31)
        try self.emit32(0x9B007C00 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SUBS XZR, Xn, Xm (CMP alias — sets condition flags)
    pub fn cmpRegReg(self: *CodeBuffer, rn: Reg, rm: Reg) !void {
        // 1|11|01011|00|0|Rm|000000|Rn|11111
        try self.emit32(0xEB00001F | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5));
    }

    // ── Data Processing (Immediate) ─────────────────────────────────

    /// ADD Xd, Xn, #imm12 (64-bit immediate add). Rn may be SP.
    pub fn addImm(self: *CodeBuffer, rd: Reg, rn: Reg, imm12: u12) !void {
        // 1|00|100010|0|imm12|Rn|Rd
        try self.emit32(0x91000000 | (@as(u32, imm12) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// ADD Xd, Xn, #imm12, LSL #12 (imm scaled by 4096).
    pub fn addImmShift12(self: *CodeBuffer, rd: Reg, rn: Reg, imm12: u12) !void {
        // sh=1 (bit 22) shifts the imm12 left by 12.
        try self.emit32(0x91400000 | (@as(u32, imm12) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SUB Xd, Xn, #imm12 (64-bit immediate sub). Rn may be SP.
    pub fn subImm(self: *CodeBuffer, rd: Reg, rn: Reg, imm12: u12) !void {
        // 1|10|100010|0|imm12|Rn|Rd
        try self.emit32(0xD1000000 | (@as(u32, imm12) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SUB Xd, Xn, #imm12, LSL #12 (imm scaled by 4096).
    pub fn subImmShift12(self: *CodeBuffer, rd: Reg, rn: Reg, imm12: u12) !void {
        try self.emit32(0xD1400000 | (@as(u32, imm12) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SUBS Xd, Xn, #imm12 — sets flags. Alias CMP when rd=XZR.
    pub fn subsImm(self: *CodeBuffer, rd: Reg, rn: Reg, imm12: u12) !void {
        try self.emit32(0xF1000000 | (@as(u32, imm12) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// CMP Xn, #imm12 (alias SUBS XZR, Xn, #imm12)
    pub fn cmpImm(self: *CodeBuffer, rn: Reg, imm12: u12) !void {
        try self.subsImm(.sp, rn, imm12); // reg 31 in SUBS = XZR
    }

    /// CMP Wn, #imm12 — 32-bit variant (alias SUBS WZR, Wn, #imm12)
    pub fn cmpImm32(self: *CodeBuffer, rn: Reg, imm12: u12) !void {
        // 0|11|100010|0|imm12|Rn|11111
        try self.emit32(0x7100001F | (@as(u32, imm12) << 10) |
            (@as(u32, rn.encoding()) << 5));
    }

    /// SUBS WZR, Wn, Wm (32-bit CMP — sets flags)
    pub fn cmpRegReg32(self: *CodeBuffer, rn: Reg, rm: Reg) !void {
        // 0|11|01011|00|0|Rm|000000|Rn|11111
        try self.emit32(0x6B00001F | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5));
    }

    // ── Move ────────────────────────────────────────────────────────

    /// MOV Xd, Xn (alias for ORR Xd, XZR, Xn)
    /// NOTE: reg 31 in logical ops encodes XZR, *not* SP. Use `movFromSp`
    /// for `mov Xd, sp` (that's ADD Xd, SP, #0).
    pub fn movRegReg(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        try self.orrRegReg(rd, .sp, rn); // sp encodes as XZR in logic ops
    }

    /// MOV Xd, SP  (alias for ADD Xd, SP, #0). Needed because ORR treats
    /// reg 31 as XZR, not SP.
    pub fn movFromSp(self: *CodeBuffer, rd: Reg) !void {
        try self.addImm(rd, .sp, 0);
    }

    /// MOV SP, Xn  (alias for ADD SP, Xn, #0).
    pub fn movToSp(self: *CodeBuffer, rn: Reg) !void {
        try self.addImm(.sp, rn, 0);
    }

    /// MOV Wd, Wn (32-bit — zero-extends into Xd)
    pub fn movRegReg32(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        // ORR Wd, WZR, Wn: 0|01|01010|00|0|Rm|000000|11111|Rd
        try self.emit32(0x2A0003E0 | (@as(u32, rn.encoding()) << 16) | rd.encoding());
    }

    /// MOVZ Xd, #imm16 (move wide with zero, optionally shifted)
    pub fn movz(self: *CodeBuffer, rd: Reg, imm16: u16, shift: u2) !void {
        // 1|10|100101|hw|imm16|Rd
        try self.emit32(0xD2800000 | (@as(u32, shift) << 21) |
            (@as(u32, imm16) << 5) | rd.encoding());
    }

    /// MOVK Xd, #imm16, LSL #shift (move wide with keep)
    pub fn movk(self: *CodeBuffer, rd: Reg, imm16: u16, shift: u2) !void {
        // 1|11|100101|hw|imm16|Rd
        try self.emit32(0xF2800000 | (@as(u32, shift) << 21) |
            (@as(u32, imm16) << 5) | rd.encoding());
    }

    /// Load a 32-bit immediate into Xd (MOVZ, possibly + MOVK).
    pub fn movImm32(self: *CodeBuffer, rd: Reg, val: i32) !void {
        const uval: u32 = @bitCast(val);
        const lo: u16 = @truncate(uval);
        const hi: u16 = @truncate(uval >> 16);
        try self.movz(rd, lo, 0);
        if (hi != 0) try self.movk(rd, hi, 1);
    }

    /// Load a 64-bit immediate into Xd (MOVZ + up to 3 MOVK).
    pub fn movImm64(self: *CodeBuffer, rd: Reg, val: u64) !void {
        try self.movz(rd, @truncate(val), 0);
        if (val >> 16 != 0) try self.movk(rd, @truncate(val >> 16), 1);
        if (val >> 32 != 0) try self.movk(rd, @truncate(val >> 32), 2);
        if (val >> 48 != 0) try self.movk(rd, @truncate(val >> 48), 3);
    }

    // ── Control Flow ────────────────────────────────────────────────

    /// B imm26 (unconditional branch, PC-relative ±128MB)
    pub fn b(self: *CodeBuffer, offset_words: i26) !void {
        const imm: u32 = @as(u32, @as(u26, @bitCast(offset_words)));
        try self.emit32(0x14000000 | imm);
    }

    /// BL imm26 (branch with link / call)
    pub fn bl(self: *CodeBuffer, offset_words: i26) !void {
        const imm: u32 = @as(u32, @as(u26, @bitCast(offset_words)));
        try self.emit32(0x94000000 | imm);
    }

    /// B.cond imm19 (conditional branch)
    pub fn bCond(self: *CodeBuffer, cond: Cond, offset_words: i19) !void {
        const imm: u32 = @as(u32, @as(u19, @bitCast(offset_words)));
        try self.emit32(0x54000000 | (imm << 5) | @intFromEnum(cond));
    }

    /// BLR Xn (branch to register with link / indirect call)
    pub fn blr(self: *CodeBuffer, rn: Reg) !void {
        try self.emit32(0xD63F0000 | (@as(u32, rn.encoding()) << 5));
    }

    /// BR Xn (branch to register, no link — used for tail calls)
    pub fn br(self: *CodeBuffer, rn: Reg) !void {
        try self.emit32(0xD61F0000 | (@as(u32, rn.encoding()) << 5));
    }

    /// RET (return via LR, alias for BR X30)
    pub fn ret(self: *CodeBuffer) !void {
        try self.emit32(0xD65F03C0);
    }

    /// BRK #imm16 (breakpoint — trap)
    pub fn brk(self: *CodeBuffer, imm16: u16) !void {
        try self.emit32(0xD4200000 | (@as(u32, imm16) << 5));
    }

    /// NOP
    pub fn nop(self: *CodeBuffer) !void {
        try self.emit32(0xD503201F);
    }

    // ── Memory Access ───────────────────────────────────────────────

    /// LDR Xt, [Xn, #imm12*8] (64-bit load, unsigned offset scaled by 8)
    pub fn ldrImm(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        // 11|111|00|11|0|imm12|Rn|Rt
        try self.emit32(0xF9400000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// STR Xt, [Xn, #imm12*8] (64-bit store, unsigned offset scaled by 8)
    pub fn strImm(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        // 11|111|00|10|0|imm12|Rn|Rt
        try self.emit32(0xF9000000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// LDR Wt, [Xn, #imm12*4] (32-bit load, unsigned offset scaled by 4)
    pub fn ldrImm32(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0xB9400000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// STR Wt, [Xn, #imm12*4] (32-bit store, unsigned offset scaled by 4)
    pub fn strImm32(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0xB9000000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// LDRB Wt, [Xn, #imm12] (zero-extended byte load, unscaled offset).
    pub fn ldrbImm(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0x39400000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// STRB Wt, [Xn, #imm12] (byte store, unscaled offset).
    pub fn strbImm(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0x39000000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// LDRH Wt, [Xn, #imm12*2] (zero-extended halfword load, scaled by 2).
    pub fn ldrhImm(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0x79400000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// STRH Wt, [Xn, #imm12*2] (halfword store, scaled by 2).
    pub fn strhImm(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0x79000000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// LDRSB Xt, [Xn, #imm12] (sign-extend byte → 64-bit).
    pub fn ldrsbImm64(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0x39800000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// LDRSB Wt, [Xn, #imm12] (sign-extend byte → 32-bit, zero-ext upper).
    pub fn ldrsbImm32(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0x39C00000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// LDRSH Xt, [Xn, #imm12*2] (sign-extend halfword → 64-bit).
    pub fn ldrshImm64(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0x79800000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// LDRSH Wt, [Xn, #imm12*2] (sign-extend halfword → 32-bit, zero-ext upper).
    pub fn ldrshImm32(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0x79C00000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// LDRSW Xt, [Xn, #imm12*4] (sign-extend word → 64-bit).
    pub fn ldrswImm(self: *CodeBuffer, rt: Reg, rn: Reg, offset: u12) !void {
        try self.emit32(0xB9800000 | (@as(u32, offset) << 10) |
            (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// STP Xt1, Xt2, [Xn, #imm7*8]! (store pair, pre-index)
    pub fn stpPre(self: *CodeBuffer, rt1: Reg, rt2: Reg, rn: Reg, imm7: i7) !void {
        const imm: u32 = @as(u32, @as(u7, @bitCast(imm7)));
        try self.emit32(0xA9800000 | (imm << 15) |
            (@as(u32, rt2.encoding()) << 10) | (@as(u32, rn.encoding()) << 5) | rt1.encoding());
    }

    /// LDP Xt1, Xt2, [Xn], #imm7*8 (load pair, post-index)
    pub fn ldpPost(self: *CodeBuffer, rt1: Reg, rt2: Reg, rn: Reg, imm7: i7) !void {
        const imm: u32 = @as(u32, @as(u7, @bitCast(imm7)));
        try self.emit32(0xA8C00000 | (imm << 15) |
            (@as(u32, rt2.encoding()) << 10) | (@as(u32, rn.encoding()) << 5) | rt1.encoding());
    }

    /// STP Xt1, Xt2, [Xn, #imm7*8] (signed-offset, no writeback)
    pub fn stpImm(self: *CodeBuffer, rt1: Reg, rt2: Reg, rn: Reg, imm7: i7) !void {
        const imm: u32 = @as(u32, @as(u7, @bitCast(imm7)));
        try self.emit32(0xA9000000 | (imm << 15) |
            (@as(u32, rt2.encoding()) << 10) | (@as(u32, rn.encoding()) << 5) | rt1.encoding());
    }

    /// LDP Xt1, Xt2, [Xn, #imm7*8] (signed-offset, no writeback)
    pub fn ldpImm(self: *CodeBuffer, rt1: Reg, rt2: Reg, rn: Reg, imm7: i7) !void {
        const imm: u32 = @as(u32, @as(u7, @bitCast(imm7)));
        try self.emit32(0xA9400000 | (imm << 15) |
            (@as(u32, rt2.encoding()) << 10) | (@as(u32, rn.encoding()) << 5) | rt1.encoding());
    }

    // ── Conditional / Shift / Bit Ops ───────────────────────────────

    /// CSET Xd, cond — set Xd = 1 if cond else 0 (alias CSINC Xd, XZR, XZR, !cond).
    pub fn cset(self: *CodeBuffer, rd: Reg, cond: Cond) !void {
        const inv: u4 = @intFromEnum(cond) ^ 1; // invert condition
        // CSINC Xd, XZR, XZR, !cond: 1|0|0|11010100|Rm(11111)|cond|01|Rn(11111)|Rd
        try self.emit32(0x9A9F07E0 | (@as(u32, inv) << 12) | rd.encoding());
    }

    /// CSET Wd, cond — 32-bit variant (zero-extends into Xd).
    pub fn cset32(self: *CodeBuffer, rd: Reg, cond: Cond) !void {
        const inv: u4 = @intFromEnum(cond) ^ 1;
        try self.emit32(0x1A9F07E0 | (@as(u32, inv) << 12) | rd.encoding());
    }

    /// CSEL Xd, Xn, Xm, cond — Xd = (cond) ? Xn : Xm
    pub fn csel(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg, cond: Cond) !void {
        // 1|0|0|11010100|Rm|cond|00|Rn|Rd
        try self.emit32(0x9A800000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, @intFromEnum(cond)) << 12) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// CSEL Wd, Wn, Wm, cond — 32-bit variant.
    pub fn csel32(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg, cond: Cond) !void {
        try self.emit32(0x1A800000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, @intFromEnum(cond)) << 12) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// LSLV/LSRV/ASRV/RORV Xd, Xn, Xm (variable shift, 64-bit)
    pub fn shiftRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg, op: ShiftOp) !void {
        // 1|0|0|11010110|Rm|0010|op|Rn|Rd
        const opc: u2 = switch (op) {
            .lsl => 0b00,
            .lsr => 0b01,
            .asr => 0b10,
            .ror => 0b11,
        };
        try self.emit32(0x9AC02000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, opc) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// 32-bit variable shift (mask count by 5 per AArch64 semantics — matches wasm i32).
    pub fn shiftRegReg32(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg, op: ShiftOp) !void {
        const opc: u2 = switch (op) {
            .lsl => 0b00,
            .lsr => 0b01,
            .asr => 0b10,
            .ror => 0b11,
        };
        try self.emit32(0x1AC02000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, opc) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    pub const ShiftOp = enum { lsl, lsr, asr, ror };

    /// SDIV Xd, Xn, Xm (signed divide, 64-bit). Rounds toward zero. Divide
    /// by zero yields 0 (no trap); INT_MIN/-1 yields INT_MIN (no trap) —
    /// caller must emit wasm traps.
    pub fn sdivRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        // 1|0|0|11010110|Rm|000011|Rn|Rd
        try self.emit32(0x9AC00C00 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SDIV Wd, Wn, Wm (signed divide, 32-bit).
    pub fn sdivRegReg32(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        try self.emit32(0x1AC00C00 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// UDIV Xd, Xn, Xm (unsigned divide, 64-bit).
    pub fn udivRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        // 1|0|0|11010110|Rm|000010|Rn|Rd
        try self.emit32(0x9AC00800 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// UDIV Wd, Wn, Wm (unsigned divide, 32-bit).
    pub fn udivRegReg32(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        try self.emit32(0x1AC00800 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// MADD Xd, Xn, Xm, Xa — Xd = Xa + Xn*Xm (64-bit).
    pub fn maddRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg, ra: Reg) !void {
        // 1|00|11011|000|Rm|0|Ra|Rn|Rd
        try self.emit32(0x9B000000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, ra.encoding()) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// MADD Wd, Wn, Wm, Wa (32-bit).
    pub fn maddRegReg32(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg, ra: Reg) !void {
        try self.emit32(0x1B000000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, ra.encoding()) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// MSUB Xd, Xn, Xm, Xa — Xd = Xa - Xn*Xm (64-bit).
    pub fn msubRegReg(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg, ra: Reg) !void {
        // 1|00|11011|000|Rm|1|Ra|Rn|Rd
        try self.emit32(0x9B008000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, ra.encoding()) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// MSUB Wd, Wn, Wm, Wa (32-bit).
    pub fn msubRegReg32(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg, ra: Reg) !void {
        try self.emit32(0x1B008000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, ra.encoding()) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    // ── Scalar FPU ops ──────────────────────────────────────────────
    // The AArch64 backend currently has no V-register allocator; float
    // values flow through the integer register file (matching the
    // don't-care upper-bits convention used by sub-word integer ops).
    // For the actual FPU computation we shuttle values through scratch
    // V-regs V0/V1 (non-allocatable) via FMOV. This mirrors the way
    // emitFSignBit uses integer EOR/AND on the raw bit pattern, but
    // gives us access to the hardware's rounding / denormal / NaN
    // semantics for add/sub/mul/div/sqrt/min/max and the various
    // float→int conversions.

    /// FMOV Sd, Wn — copy bits from a W register into the low 32 bits
    /// of a scalar V register (other lanes zeroed).
    pub fn fmovSFromGp32(self: *CodeBuffer, vd: u5, rn: Reg) !void {
        // sf=0 type=00 rmode=00 opcode=111 : 0001 1110 0010 0111 0000 00 Rn Rd
        try self.emit32(0x1E270000 | (@as(u32, rn.encoding()) << 5) | vd);
    }

    /// FMOV Wd, Sn — copy low 32 bits of a scalar V register into a W
    /// register. Upper 32 bits of the destination X register are zeroed.
    pub fn fmovGpFromS32(self: *CodeBuffer, rd: Reg, vn: u5) !void {
        // sf=0 type=00 rmode=00 opcode=110
        try self.emit32(0x1E260000 | (@as(u32, vn) << 5) | rd.encoding());
    }

    /// FMOV Dd, Xn — copy bits from an X register into the low 64 bits
    /// of a scalar V register.
    pub fn fmovDFromGp64(self: *CodeBuffer, vd: u5, rn: Reg) !void {
        // sf=1 type=01 rmode=00 opcode=111
        try self.emit32(0x9E670000 | (@as(u32, rn.encoding()) << 5) | vd);
    }

    /// FMOV Xd, Dn — copy low 64 bits of a scalar V register into an X
    /// register.
    pub fn fmovGpFromD64(self: *CodeBuffer, rd: Reg, vn: u5) !void {
        // sf=1 type=01 rmode=00 opcode=110
        try self.emit32(0x9E660000 | (@as(u32, vn) << 5) | rd.encoding());
    }

    /// Kind of scalar FPU binary op. Values are the opcode bits [15:12]
    /// in the `Floating-point data-processing (2 source)` encoding.
    pub const FBinOp = enum(u4) {
        mul = 0b0000,
        div = 0b0001,
        add = 0b0010,
        sub = 0b0011,
        max = 0b0100,
        min = 0b0101,
    };

    fn emitFBinOp(self: *CodeBuffer, is_f64: bool, op: FBinOp, vd: u5, vn: u5, vm: u5) !void {
        // 0001 1110 0 ty 1 Rm opcode 10 Rn Rd
        //   ty: 00 = single (f32), 01 = double (f64), bit 22
        const ty: u32 = if (is_f64) 1 else 0;
        const base: u32 = 0x1E200800 | (ty << 22);
        const opcode_bits: u32 = @as(u32, @intFromEnum(op)) << 12;
        try self.emit32(base | opcode_bits |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5) |
            vd);
    }

    pub fn faddScalar(self: *CodeBuffer, is_f64: bool, vd: u5, vn: u5, vm: u5) !void {
        return self.emitFBinOp(is_f64, .add, vd, vn, vm);
    }

    pub fn fsubScalar(self: *CodeBuffer, is_f64: bool, vd: u5, vn: u5, vm: u5) !void {
        return self.emitFBinOp(is_f64, .sub, vd, vn, vm);
    }

    pub fn fmulScalar(self: *CodeBuffer, is_f64: bool, vd: u5, vn: u5, vm: u5) !void {
        return self.emitFBinOp(is_f64, .mul, vd, vn, vm);
    }

    pub fn fdivScalar(self: *CodeBuffer, is_f64: bool, vd: u5, vn: u5, vm: u5) !void {
        return self.emitFBinOp(is_f64, .div, vd, vn, vm);
    }

    /// FSQRT Sd, Sn / FSQRT Dd, Dn (floating-point data-processing 1 source).
    pub fn fsqrtScalar(self: *CodeBuffer, is_f64: bool, vd: u5, vn: u5) !void {
        // 0001 1110 0 ty 1 00001 10000 Rn Rd
        const ty: u32 = if (is_f64) 1 else 0;
        try self.emit32(0x1E21C000 | (ty << 22) | (@as(u32, vn) << 5) | vd);
    }

    /// FMIN Sd, Sn, Sm / FMIN Dd, Dn, Dm — IEEE-754 minNum.
    /// NaN-propagating (matches wasm's f.min for non-NaN inputs).
    pub fn fminScalar(self: *CodeBuffer, is_f64: bool, vd: u5, vn: u5, vm: u5) !void {
        return self.emitFBinOp(is_f64, .min, vd, vn, vm);
    }

    /// FMAX Sd, Sn, Sm / FMAX Dd, Dn, Dm — IEEE-754 maxNum.
    pub fn fmaxScalar(self: *CodeBuffer, is_f64: bool, vd: u5, vn: u5, vm: u5) !void {
        return self.emitFBinOp(is_f64, .max, vd, vn, vm);
    }

    /// Rounding mode for FRINT* (floating-point round to integral).
    /// Values are opcode[2:0] at bits [17:15] in the 1-source FP encoding;
    /// opcode[5:3] is the fixed prefix `001` already present in the base.
    pub const FRoundMode = enum(u3) {
        /// FRINTN — round to nearest, ties to even (wasm f.nearest).
        nearest = 0b000,
        /// FRINTP — round toward +inf (wasm f.ceil).
        ceil = 0b001,
        /// FRINTM — round toward -inf (wasm f.floor).
        floor = 0b010,
        /// FRINTZ — round toward zero, a.k.a. truncate (wasm f.trunc).
        trunc = 0b011,
    };

    /// FRINTN/P/M/Z Sd, Sn / FRINTN/P/M/Z Dd, Dn — round to integral.
    pub fn frintScalar(self: *CodeBuffer, is_f64: bool, mode: FRoundMode, vd: u5, vn: u5) !void {
        // 0001 1110 0 ty 1 001 <mode:3> 10000 Rn Rd
        const ty: u32 = if (is_f64) 1 else 0;
        const m: u32 = @intFromEnum(mode);
        try self.emit32(0x1E244000 | (ty << 22) | (m << 15) |
            (@as(u32, vn) << 5) | vd);
    }

    /// CNT Vd.8B, Vn.8B — population count per byte in the low 64 bits of
    /// a V register. Each destination byte holds popcount(0..=8) of the
    /// corresponding source byte. Used to implement wasm i32/i64.popcnt
    /// by following with ADDV B, Bd, Vn.8B to sum all bytes.
    pub fn cnt8b(self: *CodeBuffer, vd: u5, vn: u5) !void {
        // 0 Q 1 01110 size 10000 00101 10 Rn Rd, Q=0 (8B), size=00
        try self.emit32(0x0E205800 | (@as(u32, vn) << 5) | vd);
    }

    /// ADDV Bd, Vn.8B — sum all 8 bytes of Vn into the low byte of Vd.
    pub fn addvB8b(self: *CodeBuffer, vd: u5, vn: u5) !void {
        // 0 Q 0 01110 size 11000 11011 10 Rn Rd, Q=0, size=00
        try self.emit32(0x0E31B800 | (@as(u32, vn) << 5) | vd);
    }

    /// LDR Qt, [Xn] — full-width 128-bit vector load.
    pub fn ldrQ(self: *CodeBuffer, vt: u5, rn: Reg) !void {
        try self.emit32(0x3DC00000 | (@as(u32, rn.encoding()) << 5) | vt);
    }

    /// STR Qt, [Xn] — full-width 128-bit vector store.
    pub fn strQ(self: *CodeBuffer, vt: u5, rn: Reg) !void {
        try self.emit32(0x3D800000 | (@as(u32, rn.encoding()) << 5) | vt);
    }

    /// INS Vd.D[lane], Xn (alias: MOV Vd.D[lane], Xn).
    pub fn insDFromGp64(self: *CodeBuffer, vd: u5, lane: u1, rn: Reg) !void {
        try self.emit32(0x4E081C00 |
            (@as(u32, lane) << 20) |
            (@as(u32, rn.encoding()) << 5) |
            vd);
    }

    /// DUP Vd.4S, Wn.
    pub fn dup4sFromGp32(self: *CodeBuffer, vd: u5, rn: Reg) !void {
        try self.emit32(0x4E040C00 | (@as(u32, rn.encoding()) << 5) | vd);
    }

    /// DUP Vd.8H, Wn.
    pub fn dup8hFromGp32(self: *CodeBuffer, vd: u5, rn: Reg) !void {
        try self.emit32(0x4E020C00 | (@as(u32, rn.encoding()) << 5) | vd);
    }

    /// DUP Vd.16B, Wn.
    pub fn dup16bFromGp32(self: *CodeBuffer, vd: u5, rn: Reg) !void {
        try self.emit32(0x4E010C00 | (@as(u32, rn.encoding()) << 5) | vd);
    }

    /// INS Vd.S[lane], Wn (alias: MOV Vd.S[lane], Wn).
    pub fn insSFromGp32(self: *CodeBuffer, vd: u5, lane: u2, rn: Reg) !void {
        try self.emit32(0x4E041C00 |
            (@as(u32, lane) << 19) |
            (@as(u32, rn.encoding()) << 5) |
            vd);
    }

    /// INS Vd.B[lane], Wn (alias: MOV Vd.B[lane], Wn).
    pub fn insBFromGp32(self: *CodeBuffer, vd: u5, lane: u4, rn: Reg) !void {
        const imm5 = (@as(u32, lane) << 1) | 1;
        try self.emit32(0x4E001C00 |
            (imm5 << 16) |
            (@as(u32, rn.encoding()) << 5) |
            vd);
    }

    /// INS Vd.H[lane], Wn (alias: MOV Vd.H[lane], Wn).
    pub fn insHFromGp32(self: *CodeBuffer, vd: u5, lane: u3, rn: Reg) !void {
        try self.emit32(0x4E021C00 |
            (@as(u32, lane) << 18) |
            (@as(u32, rn.encoding()) << 5) |
            vd);
    }

    /// MVN Vd.16B, Vn.16B (alias for NOT).
    pub fn mvn16b(self: *CodeBuffer, vd: u5, vn: u5) !void {
        try self.emit32(0x6E205800 | (@as(u32, vn) << 5) | vd);
    }

    pub const V128BitwiseOp = enum(u32) {
        @"and" = 0x4E201C00,
        bic = 0x4E601C00,
        orr = 0x4EA01C00,
        eor = 0x6E201C00,
    };

    /// AND/BIC/ORR/EOR Vd.16B, Vn.16B, Vm.16B.
    pub fn bitwise16b(self: *CodeBuffer, op: V128BitwiseOp, vd: u5, vn: u5, vm: u5) !void {
        try self.emit32(@intFromEnum(op) |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5) |
            vd);
    }

    pub const I32x4Op = enum(u32) {
        add = 0x4EA08400,
        sub = 0x6EA08400,
        mul = 0x4EA09C00,
        cmeq = 0x6EA08C00,
        cmgt = 0x4EA03400,
        cmge = 0x4EA03C00,
        cmhi = 0x6EA03400,
        cmhs = 0x6EA03C00,
    };

    /// Integer 4S binary vector op: ADD/SUB/MUL/CMEQ/CMGT/CMGE/CMHI/CMHS.
    pub fn i32x4Op(self: *CodeBuffer, op: I32x4Op, vd: u5, vn: u5, vm: u5) !void {
        try self.emit32(@intFromEnum(op) |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5) |
            vd);
    }

    pub const I8x16Op = enum(u32) {
        add = 0x4E208400,
        sub = 0x6E208400,
        cmeq = 0x6E208C00,
        cmgt = 0x4E203400,
        cmge = 0x4E203C00,
        cmhi = 0x6E203400,
        cmhs = 0x6E203C00,
    };

    /// Integer 16B binary vector op: ADD/SUB/CMEQ/CMGT/CMGE/CMHI/CMHS.
    pub fn i8x16Op(self: *CodeBuffer, op: I8x16Op, vd: u5, vn: u5, vm: u5) !void {
        try self.emit32(@intFromEnum(op) |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5) |
            vd);
    }

    pub const I16x8Op = enum(u32) {
        add = 0x4E608400,
        sub = 0x6E608400,
        mul = 0x4E609C00,
        cmeq = 0x6E608C00,
        cmgt = 0x4E603400,
        cmge = 0x4E603C00,
        cmhi = 0x6E603400,
        cmhs = 0x6E603C00,
    };

    /// Integer 8H binary vector op: ADD/SUB/MUL/CMEQ/CMGT/CMGE/CMHI/CMHS.
    pub fn i16x8Op(self: *CodeBuffer, op: I16x8Op, vd: u5, vn: u5, vm: u5) !void {
        try self.emit32(@intFromEnum(op) |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5) |
            vd);
    }

    /// SSHL Vd.4S, Vn.4S, Vm.4S — signed variable shift.
    pub fn sshl4s(self: *CodeBuffer, vd: u5, vn: u5, vm: u5) !void {
        try self.emit32(0x4EA04400 |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5) |
            vd);
    }

    /// USHL Vd.4S, Vn.4S, Vm.4S — unsigned variable shift.
    pub fn ushl4s(self: *CodeBuffer, vd: u5, vn: u5, vm: u5) !void {
        try self.emit32(0x6EA04400 |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5) |
            vd);
    }

    /// NEG Vd.4S, Vn.4S.
    pub fn neg4s(self: *CodeBuffer, vd: u5, vn: u5) !void {
        try self.emit32(0x6EA0B800 |
            (@as(u32, vn) << 5) |
            vd);
    }

    /// SSHL Vd.8H, Vn.8H, Vm.8H — signed variable shift.
    pub fn sshl8h(self: *CodeBuffer, vd: u5, vn: u5, vm: u5) !void {
        try self.emit32(0x4E604400 |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5) |
            vd);
    }

    /// USHL Vd.8H, Vn.8H, Vm.8H — unsigned variable shift.
    pub fn ushl8h(self: *CodeBuffer, vd: u5, vn: u5, vm: u5) !void {
        try self.emit32(0x6E604400 |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5) |
            vd);
    }

    /// NEG Vd.8H, Vn.8H.
    pub fn neg8h(self: *CodeBuffer, vd: u5, vn: u5) !void {
        try self.emit32(0x6E60B800 |
            (@as(u32, vn) << 5) |
            vd);
    }

    /// UMOV Wd, Vn.S[lane] (alias: MOV Wd, Vn.S[lane]).
    pub fn umovWFromS(self: *CodeBuffer, rd: Reg, vn: u5, lane: u2) !void {
        try self.emit32(0x0E043C00 |
            (@as(u32, lane) << 19) |
            (@as(u32, vn) << 5) |
            rd.encoding());
    }

    /// UMOV Wd, Vn.H[lane] (alias: MOV Wd, Vn.H[lane]).
    pub fn umovWFromH(self: *CodeBuffer, rd: Reg, vn: u5, lane: u3) !void {
        try self.emit32(0x0E023C00 |
            (@as(u32, lane) << 18) |
            (@as(u32, vn) << 5) |
            rd.encoding());
    }

    /// UMOV Wd, Vn.B[lane] (alias: MOV Wd, Vn.B[lane]).
    pub fn umovWFromB(self: *CodeBuffer, rd: Reg, vn: u5, lane: u4) !void {
        const imm5 = (@as(u32, lane) << 1) | 1;
        try self.emit32(0x0E003C00 |
            (imm5 << 16) |
            (@as(u32, vn) << 5) |
            rd.encoding());
    }

    /// SMOV Wd, Vn.H[lane].
    pub fn smovWFromH(self: *CodeBuffer, rd: Reg, vn: u5, lane: u3) !void {
        try self.emit32(0x0E022C00 |
            (@as(u32, lane) << 18) |
            (@as(u32, vn) << 5) |
            rd.encoding());
    }

    /// SMOV Wd, Vn.B[lane].
    pub fn smovWFromB(self: *CodeBuffer, rd: Reg, vn: u5, lane: u4) !void {
        const imm5 = (@as(u32, lane) << 1) | 1;
        try self.emit32(0x0E002C00 |
            (imm5 << 16) |
            (@as(u32, vn) << 5) |
            rd.encoding());
    }

    /// DMB ISH — full data memory barrier (inner shareable domain).
    /// Provides seq-cst ordering when paired around plain loads/stores.
    pub fn dmbIsh(self: *CodeBuffer) !void {
        try self.emit32(0xD5033BBF);
    }

    /// LDAR / LDARB / LDARH — load-acquire (seq-cst), no offset.
    /// size: 1 → LDARB Wt, 2 → LDARH Wt, 4 → LDAR Wt, 8 → LDAR Xt.
    pub fn ldarSized(self: *CodeBuffer, rt: Reg, rn: Reg, size: u8) !void {
        const base: u32 = switch (size) {
            1 => 0x08DFFC00,
            2 => 0x48DFFC00,
            4 => 0x88DFFC00,
            8 => 0xC8DFFC00,
            else => return error.BadLdarSize,
        };
        try self.emit32(base | (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// STLR / STLRB / STLRH — store-release (seq-cst), no offset.
    /// size: 1 → STLRB Wt, 2 → STLRH Wt, 4 → STLR Wt, 8 → STLR Xt.
    pub fn stlrSized(self: *CodeBuffer, rt: Reg, rn: Reg, size: u8) !void {
        const base: u32 = switch (size) {
            1 => 0x089FFC00,
            2 => 0x489FFC00,
            4 => 0x889FFC00,
            8 => 0xC89FFC00,
            else => return error.BadStlrSize,
        };
        try self.emit32(base | (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// LDAXR / LDAXRB / LDAXRH — load-acquire exclusive (CAS loop entry).
    pub fn ldaxrSized(self: *CodeBuffer, rt: Reg, rn: Reg, size: u8) !void {
        const base: u32 = switch (size) {
            1 => 0x085FFC00,
            2 => 0x485FFC00,
            4 => 0x885FFC00,
            8 => 0xC85FFC00,
            else => return error.BadLdaxrSize,
        };
        try self.emit32(base | (@as(u32, rn.encoding()) << 5) | rt.encoding());
    }

    /// STLXR / STLXRB / STLXRH — store-release exclusive. `rs` receives
    /// the status (0 = success). rs must differ from rt and rn.
    pub fn stlxrSized(self: *CodeBuffer, rs: Reg, rt: Reg, rn: Reg, size: u8) !void {
        const base: u32 = switch (size) {
            1 => 0x0800FC00,
            2 => 0x4800FC00,
            4 => 0x8800FC00,
            8 => 0xC800FC00,
            else => return error.BadStlxrSize,
        };
        try self.emit32(base |
            (@as(u32, rs.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) |
            rt.encoding());
    }

    /// CBNZ Wt, #imm19 — branch if 32-bit reg is non-zero (word offset).
    pub fn cbnz32(self: *CodeBuffer, rt: Reg, offset_words: i19) !void {
        const imm19: u19 = @bitCast(offset_words);
        try self.emit32(0x35000000 | (@as(u32, imm19) << 5) | rt.encoding());
    }

    /// CBZ Xt, #imm19 — branch if 64-bit reg is zero (word offset).
    pub fn cbz64(self: *CodeBuffer, rt: Reg, offset_words: i19) !void {
        const imm19: u19 = @bitCast(offset_words);
        try self.emit32(0xB4000000 | (@as(u32, imm19) << 5) | rt.encoding());
    }

    /// LSE atomic read-modify-write ops (ARMv8.1-A). All variants emit a
    /// single seq-cst instruction (acquire + release) that atomically
    /// loads the old value at [Rn] into Rt and writes a derived value.
    /// Size selects the base opcode (byte/half/word/dword); opcode12 is
    /// the operation selector in bits [15:12].
    pub const LseOp = enum(u32) {
        add = 0x0000, // LDADD  — new = old + Rs
        clr = 0x1000, // LDCLR  — new = old & ~Rs
        eor = 0x2000, // LDEOR  — new = old ^ Rs
        set = 0x3000, // LDSET  — new = old | Rs
        swp = 0x8000, // SWP    — new = Rs
    };

    pub fn lseAtomic(
        self: *CodeBuffer,
        op: LseOp,
        rs: Reg,
        rt: Reg,
        rn: Reg,
        size: u8,
    ) !void {
        // A=1, R=1 (acquire+release) — "AL" suffix.
        const size_base: u32 = switch (size) {
            1 => 0x38E00000,
            2 => 0x78E00000,
            4 => 0xB8E00000,
            8 => 0xF8E00000,
            else => return error.BadLseSize,
        };
        try self.emit32(size_base | @intFromEnum(op) |
            (@as(u32, rs.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) |
            rt.encoding());
    }

    /// LSE CAS (AL variant): atomically, if [Rn] == Rs then [Rn] := Rt;
    /// Rs is updated to the old value of [Rn] unconditionally. Size picks
    /// the variant (CASALB/H/W/X).
    pub fn casAl(self: *CodeBuffer, rs: Reg, rt: Reg, rn: Reg, size: u8) !void {
        const size_base: u32 = switch (size) {
            1 => 0x08E0FC00,
            2 => 0x48E0FC00,
            4 => 0x88E0FC00,
            8 => 0xC8E0FC00,
            else => return error.BadCasSize,
        };
        try self.emit32(size_base |
            (@as(u32, rs.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) |
            rt.encoding());
    }

    /// NEG Xd, Xm — 64-bit negation via SUB Xd, XZR, Xm.
    pub fn negReg64(self: *CodeBuffer, rd: Reg, rm: Reg) !void {
        try self.emit32(0xCB0003E0 | (@as(u32, rm.encoding()) << 16) | rd.encoding());
    }

    /// MVN Xd, Xm — 64-bit bitwise NOT via ORN Xd, XZR, Xm.
    pub fn mvnReg64(self: *CodeBuffer, rd: Reg, rm: Reg) !void {
        try self.emit32(0xAA2003E0 | (@as(u32, rm.encoding()) << 16) | rd.encoding());
    }

    /// FCMP Sn, Sm / FCMP Dn, Dm — set NZCV from float compare.
    /// Unordered (NaN involved) sets NZCV = 0011 (N=0, Z=0, C=1, V=1).
    pub fn fcmpScalar(self: *CodeBuffer, is_f64: bool, vn: u5, vm: u5) !void {
        // 0001 1110 0 ty 1 Rm 00 1000 Rn 0 0000
        const ty: u32 = if (is_f64) 1 else 0;
        try self.emit32(0x1E202000 | (ty << 22) |
            (@as(u32, vm) << 16) |
            (@as(u32, vn) << 5));
    }

    // ── Float <-> int conversions (scalar, non-trapping) ────────────
    // Integer→float conversions always succeed. `sf` selects whether
    // the source integer reg is W (32-bit) or X (64-bit); `is_f64_dst`
    // selects whether the destination V reg is interpreted as D or S.

    /// SCVTF Sd/Dd, Wn/Xn — signed integer to float.
    pub fn scvtfFromGp(
        self: *CodeBuffer,
        is_f64_dst: bool,
        src_is_x: bool,
        vd: u5,
        rn: Reg,
    ) !void {
        // sf 0 0 11110 type 1 00 010 00000 Rn Rd
        //   sf bit 31, type bit 22, opcode 010 at bits 18:16
        const sf: u32 = if (src_is_x) 1 else 0;
        const ty: u32 = if (is_f64_dst) 1 else 0;
        try self.emit32(0x1E220000 | (sf << 31) | (ty << 22) |
            (@as(u32, rn.encoding()) << 5) | vd);
    }

    /// UCVTF Sd/Dd, Wn/Xn — unsigned integer to float.
    pub fn ucvtfFromGp(
        self: *CodeBuffer,
        is_f64_dst: bool,
        src_is_x: bool,
        vd: u5,
        rn: Reg,
    ) !void {
        // opcode 011 at bits 18:16 (bit 16 set vs SCVTF)
        const sf: u32 = if (src_is_x) 1 else 0;
        const ty: u32 = if (is_f64_dst) 1 else 0;
        try self.emit32(0x1E230000 | (sf << 31) | (ty << 22) |
            (@as(u32, rn.encoding()) << 5) | vd);
    }

    /// FCVT Dd, Sn — promote single-precision to double-precision.
    pub fn fcvtPromoteSToD(self: *CodeBuffer, vd: u5, vn: u5) !void {
        try self.emit32(0x1E22C000 | (@as(u32, vn) << 5) | vd);
    }

    /// FCVT Sd, Dn — demote double-precision to single-precision.
    pub fn fcvtDemoteDToS(self: *CodeBuffer, vd: u5, vn: u5) !void {
        try self.emit32(0x1E624000 | (@as(u32, vn) << 5) | vd);
    }

    /// FCVTZS Wd/Xd, Sn/Dn — float to signed integer, round toward zero.
    /// Saturates on overflow and returns 0 for NaN (useful for trunc_sat;
    /// the trapping wasm form requires callers to bounds-check first).
    pub fn fcvtzsToGp(
        self: *CodeBuffer,
        dst_is_x: bool,
        src_is_d: bool,
        rd: Reg,
        vn: u5,
    ) !void {
        // sf 0 0 11110 type 1 rmode=11 opcode=000 00000 Rn Rd
        // Base (W<-S): 0x1E380000
        const sf: u32 = if (dst_is_x) 1 else 0;
        const ty: u32 = if (src_is_d) 1 else 0;
        try self.emit32(0x1E380000 | (sf << 31) | (ty << 22) |
            (@as(u32, vn) << 5) | rd.encoding());
    }

    /// FCVTZU Wd/Xd, Sn/Dn — float to unsigned integer, round toward zero.
    pub fn fcvtzuToGp(
        self: *CodeBuffer,
        dst_is_x: bool,
        src_is_d: bool,
        rd: Reg,
        vn: u5,
    ) !void {
        // Same as FCVTZS but opcode=001 (bit 16 set).
        const sf: u32 = if (dst_is_x) 1 else 0;
        const ty: u32 = if (src_is_d) 1 else 0;
        try self.emit32(0x1E390000 | (sf << 31) | (ty << 22) |
            (@as(u32, vn) << 5) | rd.encoding());
    }

    /// NEG Xd, Xn (alias SUB Xd, XZR, Xn)
    pub fn negReg(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        // SUB Xd, XZR, Xn: 1|10|01011|00|0|Rm(Rn)|000000|Rn(11111)|Rd
        try self.emit32(0xCB0003E0 | (@as(u32, rn.encoding()) << 16) | rd.encoding());
    }

    /// NEG Wd, Wn (32-bit)
    pub fn negReg32(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        try self.emit32(0x4B0003E0 | (@as(u32, rn.encoding()) << 16) | rd.encoding());
    }

    /// CLZ Xd, Xn (count leading zeros, 64-bit — returns 64 for 0)
    pub fn clzReg(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        // 1|1|0|11010110|00000|00010|0|Rn|Rd
        try self.emit32(0xDAC01000 | (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// CLZ Wd, Wn (32-bit — returns 32 for 0)
    pub fn clzReg32(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        try self.emit32(0x5AC01000 | (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// RBIT Xd, Xn (reverse bits, 64-bit) — combine with CLZ for CTZ
    pub fn rbitReg(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        // 1|1|0|11010110|00000|00000|0|Rn|Rd
        try self.emit32(0xDAC00000 | (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// RBIT Wd, Wn (32-bit)
    pub fn rbitReg32(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        try self.emit32(0x5AC00000 | (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SXTB Xd, Wn (sign-extend byte — alias SBFM Xd, Xn, #0, #7)
    pub fn sxtb(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        // SBFM Xd, Xn, #0, #7: 1|00|100110|1|000000|000111|Rn|Rd
        try self.emit32(0x93401C00 | (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SXTH Xd, Wn (sign-extend halfword — alias SBFM Xd, Xn, #0, #15)
    pub fn sxth(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        // SBFM Xd, Xn, #0, #15
        try self.emit32(0x93403C00 | (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SXTW Xd, Wn (sign-extend word — alias SBFM Xd, Xn, #0, #31)
    pub fn sxtw(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        // SBFM Xd, Xn, #0, #31
        try self.emit32(0x93407C00 | (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// UXTW Xd, Wn — zero-extend low 32 bits. Implemented as AND Wd, Wn, Wn
    /// (equivalent to MOV Wd, Wn, which zero-extends to Xd).
    pub fn uxtw(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        try self.movRegReg32(rd, rn);
    }

    /// LSL Xd, Xn, #shift (64-bit). Alias of UBFM Xd, Xn, #-shift mod 64,
    /// #63-shift. Use for scaling an index by a power of 2 at the emit
    /// level without materializing the shift amount into a register.
    pub fn lslImm(self: *CodeBuffer, rd: Reg, rn: Reg, shift: u6) !void {
        const immr: u32 = @as(u32, 64 - @as(u32, shift)) & 0x3F;
        const imms: u32 = 63 - @as(u32, shift);
        // sf=1, N=1, UBFM: 1|10|100110|1|immr(6)|imms(6)|Rn|Rd
        try self.emit32(0xD3400000 | (immr << 16) | (imms << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// LSR Xd, Xn, #shift (64-bit). Alias of UBFM Xd, Xn, #shift, #63.
    pub fn lsrImm(self: *CodeBuffer, rd: Reg, rn: Reg, shift: u6) !void {
        const immr: u32 = @as(u32, shift);
        const imms: u32 = 63;
        try self.emit32(0xD3400000 | (immr << 16) | (imms << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// ADD Xd, Xn, Wm, UXTW #3 — add Xn + (zero_extend_32(Wm) << 3).
    /// Convenient for computing `base + idx*8` in one instruction when the
    /// index is a 32-bit wasm table or memory index.
    pub fn addExtUxtw3(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg) !void {
        // ADD (extended register) 64-bit: 1|0|0|01011|00|1|Rm|option(010)|imm3(3)|Rn|Rd
        try self.emit32(0x8B204C00 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    // ── Prologue / Epilogue ─────────────────────────────────────────

    /// Emit function prologue: STP FP, LR, [SP, #-frame_size]!; MOV FP, SP
    /// For frames that exceed the STP pre-index immediate range (±512), a
    /// two-step sequence is used: SUB SP, SP, #frame_size; STP FP, LR, [SP].
    pub fn emitPrologue(self: *CodeBuffer, frame_size: u32) !void {
        if (frame_size / 8 <= 63) {
            // STP x29, x30, [sp, #-frame_size]!
            const scaled: i7 = @intCast(-@as(i8, @intCast(frame_size / 8)));
            try self.stpPre(.fp, .lr, .sp, scaled);
        } else {
            try self.emitSpAdjust(frame_size, .sub);
            // STP x29, x30, [sp, #0]
            try self.stpImm(.fp, .lr, .sp, 0);
        }
        // MOV x29, sp  (ADD x29, sp, #0 — NOT ORR, which would use XZR)
        try self.movFromSp(.fp);
    }

    /// Emit function epilogue: LDP FP, LR, [SP], #frame_size; RET
    /// Matching large-frame fallback for emitPrologue.
    pub fn emitEpilogue(self: *CodeBuffer, frame_size: u32) !void {
        try self.emitEpilogueNoRet(frame_size);
        try self.ret();
    }

    /// Same as emitEpilogue but without the trailing RET — used by tail
    /// calls, which tear down the frame and then branch (B / BR) directly
    /// to the target function.
    pub fn emitEpilogueNoRet(self: *CodeBuffer, frame_size: u32) !void {
        if (frame_size / 8 <= 63) {
            const scaled: i7 = @intCast(@as(i8, @intCast(frame_size / 8)));
            try self.ldpPost(.fp, .lr, .sp, scaled);
        } else {
            try self.ldpImm(.fp, .lr, .sp, 0);
            try self.emitSpAdjust(frame_size, .add);
        }
    }

    /// Shared SP adjustment used by both prologue and epilogue for
    /// frames that exceed the STP pre-/post-index scaled-immediate range.
    /// Covers the full 24-bit range by splitting into high 12 bits
    /// (LSL #12) and low 12 bits where needed — i.e. frames up to
    /// ~16 MB, well beyond any plausible wasm function. Larger frames
    /// return `error.FrameTooLarge`.
    fn emitSpAdjust(self: *CodeBuffer, size: u32, comptime op: enum { add, sub }) !void {
        if (size > 0xFFFFFF) return error.FrameTooLarge;
        const high: u12 = @intCast(size >> 12);
        const low: u12 = @intCast(size & 0xFFF);
        if (high != 0) {
            switch (op) {
                .add => try self.addImmShift12(.sp, .sp, high),
                .sub => try self.subImmShift12(.sp, .sp, high),
            }
        }
        if (low != 0) {
            switch (op) {
                .add => try self.addImm(.sp, .sp, low),
                .sub => try self.subImm(.sp, .sp, low),
            }
        }
    }

    /// Patch a 32-bit value at a given offset (for branch fixups).
    pub fn patch32(self: *CodeBuffer, offset: usize, val: u32) void {
        const bytes = std.mem.toBytes(std.mem.nativeToLittle(u32, val));
        @memcpy(self.bytes.items[offset..][0..4], &bytes);
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "emit: ADD x0, x1, x2" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.addRegReg(.x0, .x1, .x2);
    try std.testing.expectEqual(@as(usize, 4), code.len());
    const word = std.mem.readInt(u32, code.getCode()[0..4], .little);
    try std.testing.expectEqual(@as(u32, 0x8B020020), word);
}

test "emit: SUB x3, x4, x5" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.subRegReg(.x3, .x4, .x5);
    const word = std.mem.readInt(u32, code.getCode()[0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xCB050083), word);
}

test "emit: MOVZ x0, #42" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.movz(.x0, 42, 0);
    const word = std.mem.readInt(u32, code.getCode()[0..4], .little);
    // MOVZ x0, #42 = 0xD2800000 | (42 << 5) | 0 = 0xD2800540
    try std.testing.expectEqual(@as(u32, 0xD2800540), word);
}

test "emit: RET" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.ret();
    const word = std.mem.readInt(u32, code.getCode()[0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xD65F03C0), word);
}

test "emit: BRK #0" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.brk(0);
    const word = std.mem.readInt(u32, code.getCode()[0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xD4200000), word);
}

test "emit: NOP" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.nop();
    const word = std.mem.readInt(u32, code.getCode()[0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xD503201F), word);
}

test "emit: movImm32 small value" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.movImm32(.x0, 100);
    // Small value: just MOVZ (4 bytes)
    try std.testing.expectEqual(@as(usize, 4), code.len());
}

test "emit: movImm32 large value needs MOVZ+MOVK" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.movImm32(.x0, @bitCast(@as(u32, 0x00010064))); // hi != 0
    // MOVZ + MOVK = 8 bytes
    try std.testing.expectEqual(@as(usize, 8), code.len());
}

test "emit: prologue + epilogue roundtrip" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.emitPrologue(16);
    try code.emitEpilogue(16);
    // prologue: STP(4) + MOV(4) = 8; epilogue: LDP(4) + RET(4) = 8
    try std.testing.expectEqual(@as(usize, 16), code.len());
}

test "emit: MUL x0, x1, x2" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.mulRegReg(.x0, .x1, .x2);
    const word = std.mem.readInt(u32, code.getCode()[0..4], .little);
    // MADD x0, x1, x2, xzr = 0x9B027C20
    try std.testing.expectEqual(@as(u32, 0x9B027C20), word);
}

test "emit: SDIV x0, x1, x2" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.sdivRegReg(.x0, .x1, .x2);
    try expectWord(0x9AC20C20, &code);
}

test "emit: SDIV w0, w1, w2" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.sdivRegReg32(.x0, .x1, .x2);
    try expectWord(0x1AC20C20, &code);
}

test "emit: UDIV x3, x4, x5" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.udivRegReg(.x3, .x4, .x5);
    try expectWord(0x9AC50883, &code);
}

test "emit: UDIV w3, w4, w5" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.udivRegReg32(.x3, .x4, .x5);
    try expectWord(0x1AC50883, &code);
}

test "emit: MSUB x0, x1, x2, x3" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.msubRegReg(.x0, .x1, .x2, .x3);
    try expectWord(0x9B028C20, &code);
}

test "emit: MSUB w0, w1, w2, w3" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.msubRegReg32(.x0, .x1, .x2, .x3);
    try expectWord(0x1B028C20, &code);
}

test "emit: MADD x0, x1, x2, x3" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.maddRegReg(.x0, .x1, .x2, .x3);
    try expectWord(0x9B020C20, &code);
}

test "emit: MADD w0, w1, w2, w3" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.maddRegReg32(.x0, .x1, .x2, .x3);
    try expectWord(0x1B020C20, &code);
}

test "emit: LSL x3, x4, #3 (UBFM alias)" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.lslImm(.x3, .x4, 3);
    // LSL Xd, Xn, #3 = UBFM Xd, Xn, #61, #60.
    // sf=1 N=1 UBFM base=0xD3400000. immr=61=0x3D -> <<16 = 0x3D0000.
    // imms=60=0x3C -> <<10 = 0xF000. Rn=4<<5=0x80. Rd=3.
    try expectWord(0xD37DF083, &code);
}

test "emit: ADD x3, x4, w5, UXTW #3" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.addExtUxtw3(.x3, .x4, .x5);
    // Base 0x8B204C00. Rm=5<<16=0x50000. Rn=4<<5=0x80. Rd=3.
    try expectWord(0x8B254C83, &code);
}

test "emit: AND w3, w4, #31" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.andImm32Mask31(.x3, .x4);
    try expectWord(0x12001083, &code);
}

test "emit: AND w3, w4, #15" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.andImm32Mask15(.x3, .x4);
    try expectWord(0x12000C83, &code);
}

fn expectWord(expected: u32, code: *const CodeBuffer) !void {
    try std.testing.expectEqual(expected, std.mem.readInt(u32, code.getCode()[0..4], .little));
}

test "emit: prologue sets FP via ADD (not ORR-XZR)" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.emitPrologue(16);
    // Second word should be ADD x29, sp, #0 = 0x910003FD
    const w2 = std.mem.readInt(u32, code.getCode()[4..8], .little);
    try std.testing.expectEqual(@as(u32, 0x910003FD), w2);
}

test "emit: CSET Xd, EQ" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.cset(.x0, .eq);
    // CSINC Xd, XZR, XZR, NE (inverted): 0x9A9F17E0
    try expectWord(0x9A9F17E0, &code);
}

test "emit: CSET Wd, NE" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.cset32(.x0, .ne);
    // 32-bit CSET, inverted cond = EQ(0000): 0x1A9F07E0
    try expectWord(0x1A9F07E0, &code);
}

test "emit: CSEL x0, x1, x2, eq" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.csel(.x0, .x1, .x2, .eq);
    // 64-bit CSEL: 0x9A820020
    try expectWord(0x9A820020, &code);
}

test "emit: LSLV x0, x1, x2" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.shiftRegReg(.x0, .x1, .x2, .lsl);
    // LSLV X0, X1, X2 = 0x9AC22020
    try expectWord(0x9AC22020, &code);
}

test "emit: RORV w0, w1, w2" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.shiftRegReg32(.x0, .x1, .x2, .ror);
    // RORV W0, W1, W2 = 0x1AC22C20
    try expectWord(0x1AC22C20, &code);
}

test "emit: NEG X0, X1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.negReg(.x0, .x1);
    // NEG X0, X1 (alias SUB X0, XZR, X1) = 0xCB0103E0
    try expectWord(0xCB0103E0, &code);
}

test "emit: CLZ X0, X1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.clzReg(.x0, .x1);
    // CLZ X0, X1 = 0xDAC01020
    try expectWord(0xDAC01020, &code);
}

test "emit: CLZ W0, W1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.clzReg32(.x0, .x1);
    try expectWord(0x5AC01020, &code);
}

test "emit: RBIT X0, X1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.rbitReg(.x0, .x1);
    try expectWord(0xDAC00020, &code);
}

test "emit: SXTB X0, W1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.sxtb(.x0, .x1);
    // SBFM X0, X1, #0, #7 = 0x93401C20
    try expectWord(0x93401C20, &code);
}

test "emit: SXTW X0, W1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.sxtw(.x0, .x1);
    // SBFM X0, X1, #0, #31 = 0x93407C20
    try expectWord(0x93407C20, &code);
}

test "emit: CMP Xn, #imm" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.cmpImm(.x0, 42);
    // SUBS XZR, X0, #42 = 0xF100A81F
    try expectWord(0xF100A81F, &code);
}

test "emit: CMP Wn, Wm" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.cmpRegReg32(.x0, .x1);
    // SUBS WZR, W0, W1 = 0x6B01001F
    try expectWord(0x6B01001F, &code);
}

test "emit: MOV W0, W1 (zero-extends)" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.movRegReg32(.x0, .x1);
    // ORR W0, WZR, W1 = 0x2A0103E0
    try expectWord(0x2A0103E0, &code);
}

test "emit: MOV Xd, SP (ADD xd, sp, #0)" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.movFromSp(.fp);
    // ADD FP (X29), SP, #0 = 0x910003FD
    try expectWord(0x910003FD, &code);
}

test "emit: LDRB W0, [X1, #0]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.ldrbImm(.x0, .x1, 0);
    try expectWord(0x39400020, &code);
}

test "emit: STRB W0, [X1, #0]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.strbImm(.x0, .x1, 0);
    try expectWord(0x39000020, &code);
}

test "emit: LDRH W0, [X1, #0]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.ldrhImm(.x0, .x1, 0);
    try expectWord(0x79400020, &code);
}

test "emit: STRH W0, [X1, #0]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.strhImm(.x0, .x1, 0);
    try expectWord(0x79000020, &code);
}

test "emit: LDRSB X0, [X1, #0]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.ldrsbImm64(.x0, .x1, 0);
    try expectWord(0x39800020, &code);
}

test "emit: LDRSB W0, [X1, #0]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.ldrsbImm32(.x0, .x1, 0);
    try expectWord(0x39C00020, &code);
}

test "emit: LDRSH X0, [X1, #0]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.ldrshImm64(.x0, .x1, 0);
    try expectWord(0x79800020, &code);
}

test "emit: LDRSH W0, [X1, #0]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.ldrshImm32(.x0, .x1, 0);
    try expectWord(0x79C00020, &code);
}

test "emit: LDRSW X0, [X1, #0]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.ldrswImm(.x0, .x1, 0);
    try expectWord(0xB9800020, &code);
}

test "emit: FMIN s0, s1, s2" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.fminScalar(false, 0, 1, 2);
    try expectWord(0x1E225820, &code);
}

test "emit: FMAX d0, d1, d2" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.fmaxScalar(true, 0, 1, 2);
    try expectWord(0x1E624820, &code);
}

test "emit: FRINTN s0, s1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.frintScalar(false, .nearest, 0, 1);
    try expectWord(0x1E244020, &code);
}

test "emit: FRINTP d0, d1 (ceil)" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.frintScalar(true, .ceil, 0, 1);
    try expectWord(0x1E64C020, &code);
}

test "emit: FRINTM s0, s1 (floor)" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.frintScalar(false, .floor, 0, 1);
    try expectWord(0x1E254020, &code);
}

test "emit: FRINTZ d0, d1 (trunc)" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.frintScalar(true, .trunc, 0, 1);
    try expectWord(0x1E65C020, &code);
}

test "emit: CNT v0.8b, v1.8b" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.cnt8b(0, 1);
    try expectWord(0x0E205820, &code);
}

test "emit: ADDV b0, v1.8b" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.addvB8b(0, 1);
    try expectWord(0x0E31B820, &code);
}

test "emit: LDR q0, [x1]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.ldrQ(0, .x1);
    try expectWord(0x3DC00020, &code);
}

test "emit: STR q0, [x1]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.strQ(0, .x1);
    try expectWord(0x3D800020, &code);
}

test "emit: MOV v0.d[1], x17" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.insDFromGp64(0, 1, .x17);
    try expectWord(0x4E181E20, &code);
}

test "emit: DUP v0.4s, w1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.dup4sFromGp32(0, .x1);
    try expectWord(0x4E040C20, &code);
}

test "emit: DUP v0.8h, w1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.dup8hFromGp32(0, .x1);
    try expectWord(0x4E020C20, &code);
}

test "emit: DUP v0.16b, w3" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.dup16bFromGp32(0, .x3);
    try expectWord(0x4E010C60, &code);
}

test "emit: MOV v0.s[2], w17" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.insSFromGp32(0, 2, .x17);
    try expectWord(0x4E141E20, &code);
}

test "emit: MOV v0.h[5], w17" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.insHFromGp32(0, 5, .x17);
    try expectWord(0x4E161E20, &code);
}

test "emit: MOV v20.b[13], w4" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.insBFromGp32(20, 13, .x4);
    try expectWord(0x4E1B1C94, &code);
}

test "emit: MVN v0.16b, v1.16b" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.mvn16b(0, 1);
    try expectWord(0x6E205820, &code);
}

test "emit: EOR v0.16b, v1.16b, v2.16b" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.bitwise16b(.eor, 0, 1, 2);
    try expectWord(0x6E221C20, &code);
}

test "emit: ADD v0.4s, v1.4s, v2.4s" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.i32x4Op(.add, 0, 1, 2);
    try expectWord(0x4EA28420, &code);
}

test "emit: CMEQ v0.4s, v1.4s, v2.4s" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.i32x4Op(.cmeq, 0, 1, 2);
    try expectWord(0x6EA28C20, &code);
}

test "emit: MUL v0.4s, v1.4s, v2.4s" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.i32x4Op(.mul, 0, 1, 2);
    try expectWord(0x4EA29C20, &code);
}

test "emit: i8x16 vector ops" {
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i8x16Op(.add, 16, 17, 30);
        try expectWord(0x4E3E8630, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i8x16Op(.sub, 16, 17, 30);
        try expectWord(0x6E3E8630, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i8x16Op(.cmeq, 16, 17, 30);
        try expectWord(0x6E3E8E30, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i8x16Op(.cmgt, 16, 17, 30);
        try expectWord(0x4E3E3630, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i8x16Op(.cmge, 16, 17, 30);
        try expectWord(0x4E3E3E30, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i8x16Op(.cmhi, 16, 17, 30);
        try expectWord(0x6E3E3630, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i8x16Op(.cmhs, 16, 17, 30);
        try expectWord(0x6E3E3E30, &code);
    }
}

test "emit: i16x8 vector ops" {
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i16x8Op(.add, 0, 1, 2);
        try expectWord(0x4E628420, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i16x8Op(.sub, 3, 4, 5);
        try expectWord(0x6E658483, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i16x8Op(.mul, 6, 7, 8);
        try expectWord(0x4E689CE6, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i16x8Op(.cmeq, 9, 10, 11);
        try expectWord(0x6E6B8D49, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i16x8Op(.cmgt, 12, 13, 14);
        try expectWord(0x4E6E35AC, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i16x8Op(.cmge, 15, 16, 17);
        try expectWord(0x4E713E0F, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i16x8Op(.cmhi, 18, 19, 20);
        try expectWord(0x6E743672, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i16x8Op(.cmhs, 21, 22, 23);
        try expectWord(0x6E773ED5, &code);
    }
}

test "emit: i32x4 variable shifts" {
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.sshl4s(16, 17, 30);
        try expectWord(0x4EBE4630, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.neg4s(30, 30);
        try expectWord(0x6EA0BBDE, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.ushl4s(20, 21, 30);
        try expectWord(0x6EBE46B4, &code);
    }
}

test "emit: i16x8 variable shifts" {
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.sshl8h(16, 17, 30);
        try expectWord(0x4E7E4630, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.neg8h(30, 30);
        try expectWord(0x6E60BBDE, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.ushl8h(20, 21, 30);
        try expectWord(0x6E7E46B4, &code);
    }
}

test "emit: signed i32x4 comparisons" {
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i32x4Op(.cmgt, 0, 1, 2);
        try expectWord(0x4EA23420, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i32x4Op(.cmge, 0, 1, 2);
        try expectWord(0x4EA23C20, &code);
    }
}

test "emit: unsigned i32x4 comparisons" {
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i32x4Op(.cmhi, 0, 1, 2);
        try expectWord(0x6EA23420, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.i32x4Op(.cmhs, 0, 1, 2);
        try expectWord(0x6EA23C20, &code);
    }
}

test "emit: UMOV w0, v1.s[3]" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.umovWFromS(.x0, 1, 3);
    try expectWord(0x0E1C3C20, &code);
}

test "emit: UMOV/SMOV w from h lane" {
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.umovWFromH(.x2, 3, 6);
        try expectWord(0x0E1A3C62, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.smovWFromH(.x4, 5, 7);
        try expectWord(0x0E1E2CA4, &code);
    }
}

test "emit: UMOV/SMOV w from b lane" {
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.umovWFromB(.x5, 6, 15);
        try expectWord(0x0E1F3CC5, &code);
    }
    {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.smovWFromB(.x5, 6, 15);
        try expectWord(0x0E1F2CC5, &code);
    }
}

test "emit: DMB ISH" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.dmbIsh();
    try expectWord(0xD5033BBF, &code);
}

test "emit: LDAR Wt/Xt + LDARB/LDARH" {
    const sizes = [_]u8{ 1, 2, 4, 8 };
    const expected = [_]u32{ 0x08DFFC20, 0x48DFFC20, 0x88DFFC20, 0xC8DFFC20 };
    for (sizes, expected) |size, want| {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.ldarSized(.x0, .x1, size);
        try expectWord(want, &code);
    }
}

test "emit: STLR Wt/Xt + STLRB/STLRH" {
    const sizes = [_]u8{ 1, 2, 4, 8 };
    const expected = [_]u32{ 0x089FFC20, 0x489FFC20, 0x889FFC20, 0xC89FFC20 };
    for (sizes, expected) |size, want| {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.stlrSized(.x0, .x1, size);
        try expectWord(want, &code);
    }
}

test "emit: LDAXR sized" {
    const sizes = [_]u8{ 1, 2, 4, 8 };
    const expected = [_]u32{ 0x085FFC20, 0x485FFC20, 0x885FFC20, 0xC85FFC20 };
    for (sizes, expected) |size, want| {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.ldaxrSized(.x0, .x1, size);
        try expectWord(want, &code);
    }
}

test "emit: STLXR sized" {
    const sizes = [_]u8{ 1, 2, 4, 8 };
    // STLXR Ws=w2, Wt=w0, [x1]
    const expected = [_]u32{ 0x0802FC20, 0x4802FC20, 0x8802FC20, 0xC802FC20 };
    for (sizes, expected) |size, want| {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.stlxrSized(.x2, .x0, .x1, size);
        try expectWord(want, &code);
    }
}

test "emit: CBNZ W offset" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.cbnz32(.x2, 0);
    try expectWord(0x35000002, &code);
    var code2 = CodeBuffer.init(std.testing.allocator);
    defer code2.deinit();
    try code2.cbnz32(.x0, 1);
    try expectWord(0x35000020, &code2);
}

test "emit: CBZ X offset" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.cbz64(.x5, 4);
    try expectWord(0xB4000085, &code);
    var code2 = CodeBuffer.init(std.testing.allocator);
    defer code2.deinit();
    try code2.cbz64(.lr, 1);
    try expectWord(0xB400003E, &code2);
}

test "emit: LSE LDADDAL W" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.lseAtomic(.add, .x1, .x0, .x2, 4);
    try expectWord(0xB8E10040, &code);
}

test "emit: LSE LDCLRAL W" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.lseAtomic(.clr, .x1, .x0, .x2, 4);
    try expectWord(0xB8E11040, &code);
}

test "emit: LSE LDSETAL W" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.lseAtomic(.set, .x1, .x0, .x2, 4);
    try expectWord(0xB8E13040, &code);
}

test "emit: LSE LDEORAL W" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.lseAtomic(.eor, .x1, .x0, .x2, 4);
    try expectWord(0xB8E12040, &code);
}

test "emit: LSE SWPAL W" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.lseAtomic(.swp, .x1, .x0, .x2, 4);
    try expectWord(0xB8E18040, &code);
}

test "emit: LSE CASAL sizes" {
    const sizes = [_]u8{ 1, 2, 4, 8 };
    const expected = [_]u32{ 0x08E1FC40, 0x48E1FC40, 0x88E1FC40, 0xC8E1FC40 };
    for (sizes, expected) |size, want| {
        var code = CodeBuffer.init(std.testing.allocator);
        defer code.deinit();
        try code.casAl(.x1, .x0, .x2, size);
        try expectWord(want, &code);
    }
}

test "emit: NEG x0, x1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.negReg64(.x0, .x1);
    try expectWord(0xCB0103E0, &code);
}

test "emit: MVN x0, x1" {
    var code = CodeBuffer.init(std.testing.allocator);
    defer code.deinit();
    try code.mvnReg64(.x0, .x1);
    try expectWord(0xAA2103E0, &code);
}
