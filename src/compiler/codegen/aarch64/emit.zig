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

    /// ADD Xd, Xn, #imm12 (64-bit immediate add)
    pub fn addImm(self: *CodeBuffer, rd: Reg, rn: Reg, imm12: u12) !void {
        // 1|00|100010|0|imm12|Rn|Rd
        try self.emit32(0x91000000 | (@as(u32, imm12) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    /// SUB Xd, Xn, #imm12 (64-bit immediate sub)
    pub fn subImm(self: *CodeBuffer, rd: Reg, rn: Reg, imm12: u12) !void {
        // 1|10|100010|0|imm12|Rn|Rd
        try self.emit32(0xD1000000 | (@as(u32, imm12) << 10) |
            (@as(u32, rn.encoding()) << 5) | rd.encoding());
    }

    // ── Move ────────────────────────────────────────────────────────

    /// MOV Xd, Xn (alias for ORR Xd, XZR, Xn)
    pub fn movRegReg(self: *CodeBuffer, rd: Reg, rn: Reg) !void {
        try self.orrRegReg(rd, .sp, rn); // sp encodes as XZR in logic ops
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

    // ── Prologue / Epilogue ─────────────────────────────────────────

    /// Emit function prologue: STP FP, LR, [SP, #-frame_size]!; MOV FP, SP
    pub fn emitPrologue(self: *CodeBuffer, frame_size: u32) !void {
        // STP x29, x30, [sp, #-frame_size]!
        const scaled: i7 = @intCast(-@as(i8, @intCast(frame_size / 8)));
        try self.stpPre(.fp, .lr, .sp, scaled);
        // MOV x29, sp
        try self.movRegReg(.fp, .sp);
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
