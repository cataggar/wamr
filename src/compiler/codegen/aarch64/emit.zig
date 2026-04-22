//! AArch64 Machine Code Emitter
//!
//! Translates compiler IR into AArch64 (ARM64) machine code.
//! AArch64 uses fixed 4-byte instruction encoding, which is simpler
//! than x86-64's variable-length encoding.

const std = @import("std");

/// AArch64 general-purpose registers (64-bit X registers).
pub const Reg = enum(u5) {
    x0 = 0, x1 = 1, x2 = 2, x3 = 3,
    x4 = 4, x5 = 5, x6 = 6, x7 = 7,
    x8 = 8, x9 = 9, x10 = 10, x11 = 11,
    x12 = 12, x13 = 13, x14 = 14, x15 = 15,
    x16 = 16, x17 = 17, x18 = 18, x19 = 19,
    x20 = 20, x21 = 21, x22 = 22, x23 = 23,
    x24 = 24, x25 = 25, x26 = 26, x27 = 27,
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

    /// SUB Xd, Xn, #imm12 (64-bit immediate sub). Rn may be SP.
    pub fn subImm(self: *CodeBuffer, rd: Reg, rn: Reg, imm12: u12) !void {
        // 1|10|100010|0|imm12|Rn|Rd
        try self.emit32(0xD1000000 | (@as(u32, imm12) << 10) |
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
        const opc: u2 = switch (op) { .lsl => 0b00, .lsr => 0b01, .asr => 0b10, .ror => 0b11 };
        try self.emit32(0x9AC02000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, opc) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// 32-bit variable shift (mask count by 5 per AArch64 semantics — matches wasm i32).
    pub fn shiftRegReg32(self: *CodeBuffer, rd: Reg, rn: Reg, rm: Reg, op: ShiftOp) !void {
        const opc: u2 = switch (op) { .lsl => 0b00, .lsr => 0b01, .asr => 0b10, .ror => 0b11 };
        try self.emit32(0x1AC02000 | (@as(u32, rm.encoding()) << 16) |
            (@as(u32, opc) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    pub const ShiftOp = enum { lsl, lsr, asr, ror };

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

    // ── Prologue / Epilogue ─────────────────────────────────────────

    /// Emit function prologue: STP FP, LR, [SP, #-frame_size]!; MOV FP, SP
    pub fn emitPrologue(self: *CodeBuffer, frame_size: u32) !void {
        // STP x29, x30, [sp, #-frame_size]!
        const scaled: i7 = @intCast(-@as(i8, @intCast(frame_size / 8)));
        try self.stpPre(.fp, .lr, .sp, scaled);
        // MOV x29, sp  (ADD x29, sp, #0 — NOT ORR, which would use XZR)
        try self.movFromSp(.fp);
    }

    /// Emit function epilogue: LDP FP, LR, [SP], #frame_size; RET
    pub fn emitEpilogue(self: *CodeBuffer, frame_size: u32) !void {
        const scaled: i7 = @intCast(@as(i8, @intCast(frame_size / 8)));
        try self.ldpPost(.fp, .lr, .sp, scaled);
        try self.ret();
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
