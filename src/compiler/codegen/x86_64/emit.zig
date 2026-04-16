//! x86-64 Machine Code Emitter
//!
//! Translates compiler IR into x86-64 machine code.
//! Uses a simple linear-scan register allocator and direct
//! byte emission into a growable code buffer.

const std = @import("std");

/// x86-64 general-purpose registers.
pub const Reg = enum(u4) {
    rax = 0,
    rcx = 1,
    rdx = 2,
    rbx = 3,
    rsp = 4,
    rbp = 5,
    rsi = 6,
    rdi = 7,
    r8 = 8,
    r9 = 9,
    r10 = 10,
    r11 = 11,
    r12 = 12,
    r13 = 13,
    r14 = 14,
    r15 = 15,

    /// Returns the low 3 bits of the register encoding.
    pub fn low3(self: Reg) u3 {
        return @truncate(@intFromEnum(self));
    }

    /// Returns true if the register requires a REX prefix (r8–r15).
    pub fn isExtended(self: Reg) bool {
        return @intFromEnum(self) >= 8;
    }
};

/// Machine code buffer with x86-64 instruction encoding helpers.
pub const CodeBuffer = struct {
    bytes: std.ArrayList(u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CodeBuffer {
        return .{ .bytes = .empty, .allocator = allocator };
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

    // ── Raw byte emission ─────────────────────────────────────────────

    pub fn emitByte(self: *CodeBuffer, byte: u8) !void {
        try self.bytes.append(self.allocator, byte);
    }

    pub fn emitSlice(self: *CodeBuffer, data: []const u8) !void {
        try self.bytes.appendSlice(self.allocator, data);
    }

    pub fn emitU32(self: *CodeBuffer, val: u32) !void {
        const b: [4]u8 = @bitCast(val);
        try self.emitSlice(&b);
    }

    pub fn emitI32(self: *CodeBuffer, val: i32) !void {
        const b: [4]u8 = @bitCast(val);
        try self.emitSlice(&b);
    }

    pub fn emitU64(self: *CodeBuffer, val: u64) !void {
        const b: [8]u8 = @bitCast(val);
        try self.emitSlice(&b);
    }

    // ── REX prefix ────────────────────────────────────────────────────

    /// Emit a REX prefix. `w` sets REX.W (64-bit operand size).
    /// `r` and `b` supply the extension bits from the register operands.
    fn rex(self: *CodeBuffer, w: bool, r: Reg, b: Reg) !void {
        const val: u8 = 0x40 |
            (@as(u8, if (w) 1 else 0) << 3) |
            (@as(u8, @intFromEnum(r) >> 3) << 2) |
            (@as(u8, @intFromEnum(b) >> 3));
        if (val != 0x40) try self.emitByte(val);
    }

    /// Emit REX.W prefix (64-bit operand size).
    pub fn rexW(self: *CodeBuffer, r: Reg, b: Reg) !void {
        try self.rex(true, r, b);
    }

    // ── ModR/M byte ───────────────────────────────────────────────────

    pub fn modrm(self: *CodeBuffer, mod: u2, reg_op: u3, rm: u3) !void {
        try self.emitByte(@as(u8, mod) << 6 | @as(u8, reg_op) << 3 | rm);
    }

    // ── Common x86-64 instructions ────────────────────────────────────

    /// MOV reg, imm32 (sign-extended to 64-bit via REX.W + C7 /0).
    pub fn movRegImm32(self: *CodeBuffer, dst: Reg, imm: i32) !void {
        try self.rexW(.rax, dst);
        try self.emitByte(0xC7);
        try self.modrm(0b11, 0, dst.low3());
        try self.emitI32(imm);
    }

    /// MOV reg, imm64 (REX.W + B8+rd io).
    pub fn movRegImm64(self: *CodeBuffer, dst: Reg, imm: u64) !void {
        try self.rex(true, .rax, dst);
        try self.emitByte(0xB8 | @as(u8, dst.low3()));
        try self.emitU64(imm);
    }

    /// MOV dst, src (64-bit register-to-register).
    pub fn movRegReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        // Post-allocation peephole (B3): a self-move is a semantic no-op —
        // the 64-bit mov doesn't even clear upper bits (that's the 32-bit
        // form). Skip emission so allocator-introduced identity moves
        // vanish.
        if (dst == src) return;
        try self.rexW(src, dst);
        try self.emitByte(0x89);
        try self.modrm(0b11, src.low3(), dst.low3());
    }

    /// ADD dst, src (64-bit).
    pub fn addRegReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(src, dst);
        try self.emitByte(0x01);
        try self.modrm(0b11, src.low3(), dst.low3());
    }

    /// SUB dst, src (64-bit).
    pub fn subRegReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(src, dst);
        try self.emitByte(0x29);
        try self.modrm(0b11, src.low3(), dst.low3());
    }

    /// IMUL dst, src (64-bit, two-operand form: dst = dst * src).
    pub fn imulRegReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xAF);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// AND dst, src (64-bit).
    pub fn andRegReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(src, dst);
        try self.emitByte(0x21);
        try self.modrm(0b11, src.low3(), dst.low3());
    }

    /// OR dst, src (64-bit).
    pub fn orRegReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(src, dst);
        try self.emitByte(0x09);
        try self.modrm(0b11, src.low3(), dst.low3());
    }

    /// XOR dst, src (64-bit).
    pub fn xorRegReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(src, dst);
        try self.emitByte(0x31);
        try self.modrm(0b11, src.low3(), dst.low3());
    }

    /// CMP dst, src (64-bit).
    pub fn cmpRegReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(src, dst);
        try self.emitByte(0x39);
        try self.modrm(0b11, src.low3(), dst.low3());
    }

    /// CMP r32, r32 (32-bit compare, sets flags for i32 signed semantics).
    pub fn cmpRegReg32(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        if (dst.isExtended() or src.isExtended()) try self.rex(false, src, dst);
        try self.emitByte(0x39);
        try self.modrm(0b11, src.low3(), dst.low3());
    }

    /// PUSH reg (uses REX prefix only for r8–r15).
    pub fn pushReg(self: *CodeBuffer, reg: Reg) !void {
        if (reg.isExtended()) try self.emitByte(0x41);
        try self.emitByte(0x50 | @as(u8, reg.low3()));
    }

    /// POP reg (uses REX prefix only for r8–r15).
    pub fn popReg(self: *CodeBuffer, reg: Reg) !void {
        if (reg.isExtended()) try self.emitByte(0x41);
        try self.emitByte(0x58 | @as(u8, reg.low3()));
    }

    /// RET (near return).
    pub fn ret(self: *CodeBuffer) !void {
        try self.emitByte(0xC3);
    }

    /// NOP (single-byte).
    pub fn nop(self: *CodeBuffer) !void {
        try self.emitByte(0x90);
    }

    /// INT3 (software breakpoint).
    pub fn int3(self: *CodeBuffer) !void {
        try self.emitByte(0xCC);
    }

    /// CALL rel32 — emits a 5-byte near call with a 32-bit relative offset.
    pub fn callRel32(self: *CodeBuffer, rel: i32) !void {
        try self.emitByte(0xE8);
        try self.emitI32(rel);
    }

    /// CALL reg — emits an indirect call through a register.
    pub fn callReg(self: *CodeBuffer, reg: Reg) !void {
        if (reg.isExtended()) try self.emitByte(0x41); // REX.B
        try self.emitByte(0xFF);
        try self.modrm(0b11, 2, reg.low3());
    }

    /// JMP rel32 — emits a 5-byte near jump with a 32-bit relative offset.
    pub fn jmpRel32(self: *CodeBuffer, rel: i32) !void {
        try self.emitByte(0xE9);
        try self.emitI32(rel);
    }

    /// JMP rel8 — emits a 2-byte short jump with an 8-bit relative offset.
    pub fn jmpRel8(self: *CodeBuffer, rel: i8) !void {
        try self.emitByte(0xEB);
        try self.emitByte(@bitCast(rel));
    }

    // ── Conditional jumps (Jcc rel32) ─────────────────────────────────

    /// JE/JZ rel32.
    pub fn je(self: *CodeBuffer, rel: i32) !void {
        try self.emitByte(0x0F);
        try self.emitByte(0x84);
        try self.emitI32(rel);
    }

    /// JNE/JNZ rel32.
    pub fn jne(self: *CodeBuffer, rel: i32) !void {
        try self.emitByte(0x0F);
        try self.emitByte(0x85);
        try self.emitI32(rel);
    }

    /// JL rel32 (signed less than).
    pub fn jl(self: *CodeBuffer, rel: i32) !void {
        try self.emitByte(0x0F);
        try self.emitByte(0x8C);
        try self.emitI32(rel);
    }

    /// JGE rel32 (signed greater or equal).
    pub fn jge(self: *CodeBuffer, rel: i32) !void {
        try self.emitByte(0x0F);
        try self.emitByte(0x8D);
        try self.emitI32(rel);
    }

    // ── Memory access ─────────────────────────────────────────────────

    /// MOV reg, [base + disp32] (64-bit load from memory).
    pub fn movRegMem(self: *CodeBuffer, dst: Reg, base: Reg, disp: i32) !void {
        try self.rexW(dst, base);
        try self.emitByte(0x8B);
        try self.modrm(0b10, dst.low3(), base.low3());
        if (base.low3() == 4) try self.emitByte(0x24); // SIB for RSP-based
        try self.emitI32(disp);
    }

    /// MOV [base + disp32], reg (64-bit store to memory).
    pub fn movMemReg(self: *CodeBuffer, base: Reg, disp: i32, src: Reg) !void {
        try self.rexW(src, base);
        try self.emitByte(0x89);
        try self.modrm(0b10, src.low3(), base.low3());
        if (base.low3() == 4) try self.emitByte(0x24); // SIB for RSP-based
        try self.emitI32(disp);
    }

    // ── SETcc / MOVZX / TEST / CQO / DIV / CMOV ──────────────────────

    /// SETcc r/m8 — set byte based on condition code.
    /// Byte-register access requires a REX prefix for SPL/BPL/SIL/DIL
    /// (encodings 4–7) to distinguish them from AH/CH/DH/BH. We force
    /// the REX emission in that case even when no extension bits are set.
    pub fn setcc(self: *CodeBuffer, cc: u4, dst: Reg) !void {
        const idx = @intFromEnum(dst);
        if (idx >= 4 and idx < 8) {
            try self.emitByte(0x40); // mandatory REX for SPL/BPL/SIL/DIL
        } else {
            try self.rex(false, .rax, dst);
        }
        try self.emitByte(0x0F);
        try self.emitByte(0x90 | @as(u8, cc));
        try self.modrm(0b11, 0, dst.low3());
    }

    /// MOVZX r64, r/m8 — zero-extend byte to 64-bit.
    pub fn movzxByte(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xB6);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// TEST reg, reg (64-bit).
    pub fn testRegReg(self: *CodeBuffer, a: Reg, b: Reg) !void {
        try self.rexW(b, a);
        try self.emitByte(0x85);
        try self.modrm(0b11, b.low3(), a.low3());
    }

    /// CQO — sign-extend RAX into RDX:RAX (REX.W + 99).
    pub fn cqo(self: *CodeBuffer) !void {
        try self.emitByte(0x48); // REX.W
        try self.emitByte(0x99);
    }

    /// IDIV r/m64 — signed divide RDX:RAX by reg.
    pub fn idivReg(self: *CodeBuffer, src: Reg) !void {
        try self.rexW(.rax, src);
        try self.emitByte(0xF7);
        try self.modrm(0b11, 7, src.low3());
    }

    /// DIV r/m64 — unsigned divide RDX:RAX by reg.
    pub fn divReg(self: *CodeBuffer, src: Reg) !void {
        try self.rexW(.rax, src);
        try self.emitByte(0xF7);
        try self.modrm(0b11, 6, src.low3());
    }

    /// CMOVNZ dst, src (64-bit conditional move if not zero).
    pub fn cmovnz(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0x45);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// LZCNT dst, src — count leading zeros (BMI1).
    pub fn lzcnt(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.emitByte(0xF3); // mandatory prefix
        try self.rexW(dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xBD);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// TZCNT dst, src — count trailing zeros (BMI1).
    pub fn tzcnt(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.emitByte(0xF3); // mandatory prefix
        try self.rexW(dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xBC);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// POPCNT dst, src — population count.
    pub fn popcntReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.emitByte(0xF3); // mandatory prefix
        try self.rexW(dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xB8);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// LZCNT r32, r/m32 — 32-bit leading zero count (no REX.W).
    pub fn lzcnt32(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.emitByte(0xF3);
        if (dst.isExtended() or src.isExtended()) try self.rex(false, dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xBD);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// TZCNT r32, r/m32 — 32-bit trailing zero count (no REX.W).
    pub fn tzcnt32(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.emitByte(0xF3);
        if (dst.isExtended() or src.isExtended()) try self.rex(false, dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xBC);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// POPCNT r32, r/m32 — 32-bit population count (no REX.W).
    pub fn popcnt32(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.emitByte(0xF3);
        if (dst.isExtended() or src.isExtended()) try self.rex(false, dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xB8);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// MOV r32, r/m32 — 32-bit reg-reg move. Implicitly zero-extends to 64.
    pub fn movRegReg32(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        if (dst.isExtended() or src.isExtended()) try self.rex(false, src, dst);
        try self.emitByte(0x89);
        try self.modrm(0b11, src.low3(), dst.low3());
    }

    /// MOVSXD r64, r/m32 — sign-extend 32→64.
    pub fn movsxd(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(dst, src);
        try self.emitByte(0x63);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// MOVSX r64, r/m8 — sign-extend 8→64.
    pub fn movsxByteToReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xBE);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// MOVSX r64, r/m16 — sign-extend 16→64.
    pub fn movsxWordToReg(self: *CodeBuffer, dst: Reg, src: Reg) !void {
        try self.rexW(dst, src);
        try self.emitByte(0x0F);
        try self.emitByte(0xBF);
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    /// LEA dst, [base + index] — 3-operand 64-bit add without touching flags.
    /// Used for non-destructive `dst = base + index` when dst != base, saving
    /// a `mov dst, base` compared to the mov+add sequence.
    /// Precondition: index must not be RSP (SIB encodes rsp as "no index").
    pub fn leaRegBaseIndex64(self: *CodeBuffer, dst: Reg, base: Reg, index: Reg) !void {
        std.debug.assert(index != .rsp);
        // REX.W + R (dst) + X (index) + B (base).
        const rex_byte: u8 = 0x48 |
            (@as(u8, @intFromEnum(dst) >> 3) << 2) |
            (@as(u8, @intFromEnum(index) >> 3) << 1) |
            (@as(u8, @intFromEnum(base) >> 3));
        try self.emitByte(rex_byte);
        try self.emitByte(0x8D); // LEA
        // If base.low3 == 5 (rbp/r13), mod=00 would mean RIP-relative, so use
        // mod=01 with disp8=0 to encode a plain [base + index] form.
        const needs_disp8 = base.low3() == 5;
        const mod: u2 = if (needs_disp8) 0b01 else 0b00;
        try self.modrm(mod, dst.low3(), 0b100); // rm=100 → SIB follows
        // SIB: scale=00, index=index.low3, base=base.low3
        const sib: u8 = (@as(u8, 0) << 6) | (@as(u8, index.low3()) << 3) | @as(u8, base.low3());
        try self.emitByte(sib);
        if (needs_disp8) try self.emitByte(0x00);
    }

    /// ADD reg, imm32 (64-bit).
    /// ADD reg, imm (64-bit). Uses the imm8 form (opcode 0x83) when imm fits
    /// in a signed byte, saving 3 bytes vs the imm32 form (opcode 0x81).
    pub fn addRegImm32(self: *CodeBuffer, dst: Reg, imm: i32) !void {
        try self.rexW(.rax, dst);
        if (imm >= -128 and imm <= 127) {
            try self.emitByte(0x83);
            try self.modrm(0b11, 0, dst.low3());
            try self.emitByte(@bitCast(@as(i8, @intCast(imm))));
        } else {
            try self.emitByte(0x81);
            try self.modrm(0b11, 0, dst.low3());
            try self.emitI32(imm);
        }
    }

    /// SUB reg, imm (64-bit). Uses imm8 form when possible.
    pub fn subRegImm32(self: *CodeBuffer, dst: Reg, imm: i32) !void {
        try self.rexW(.rax, dst);
        if (imm >= -128 and imm <= 127) {
            try self.emitByte(0x83);
            try self.modrm(0b11, 5, dst.low3());
            try self.emitByte(@bitCast(@as(i8, @intCast(imm))));
        } else {
            try self.emitByte(0x81);
            try self.modrm(0b11, 5, dst.low3());
            try self.emitI32(imm);
        }
    }

    /// AND reg, imm (64-bit, sign-extended). Uses imm8 form when possible.
    pub fn andRegImm32(self: *CodeBuffer, dst: Reg, imm: i32) !void {
        try self.rexW(.rax, dst);
        if (imm >= -128 and imm <= 127) {
            try self.emitByte(0x83);
            try self.modrm(0b11, 4, dst.low3());
            try self.emitByte(@bitCast(@as(i8, @intCast(imm))));
        } else {
            try self.emitByte(0x81);
            try self.modrm(0b11, 4, dst.low3());
            try self.emitI32(imm);
        }
    }

    /// OR reg, imm (64-bit, sign-extended). Uses imm8 form when possible.
    pub fn orRegImm32(self: *CodeBuffer, dst: Reg, imm: i32) !void {
        try self.rexW(.rax, dst);
        if (imm >= -128 and imm <= 127) {
            try self.emitByte(0x83);
            try self.modrm(0b11, 1, dst.low3());
            try self.emitByte(@bitCast(@as(i8, @intCast(imm))));
        } else {
            try self.emitByte(0x81);
            try self.modrm(0b11, 1, dst.low3());
            try self.emitI32(imm);
        }
    }

    /// XOR reg, imm (64-bit, sign-extended). Uses imm8 form when possible.
    pub fn xorRegImm32(self: *CodeBuffer, dst: Reg, imm: i32) !void {
        try self.rexW(.rax, dst);
        if (imm >= -128 and imm <= 127) {
            try self.emitByte(0x83);
            try self.modrm(0b11, 6, dst.low3());
            try self.emitByte(@bitCast(@as(i8, @intCast(imm))));
        } else {
            try self.emitByte(0x81);
            try self.modrm(0b11, 6, dst.low3());
            try self.emitI32(imm);
        }
    }

    /// CMP reg, imm (64-bit, sign-extended). Uses imm8 form when possible.
    pub fn cmpRegImm32(self: *CodeBuffer, dst: Reg, imm: i32) !void {
        try self.rexW(.rax, dst);
        if (imm >= -128 and imm <= 127) {
            try self.emitByte(0x83);
            try self.modrm(0b11, 7, dst.low3());
            try self.emitByte(@bitCast(@as(i8, @intCast(imm))));
        } else {
            try self.emitByte(0x81);
            try self.modrm(0b11, 7, dst.low3());
            try self.emitI32(imm);
        }
    }

    /// XOR r32, r32 — zero register without REX.W (2 bytes, zero idiom).
    pub fn xorReg32(self: *CodeBuffer, reg: Reg) !void {
        if (reg.isExtended()) try self.rex(false, reg, reg);
        try self.emitByte(0x31);
        try self.modrm(0b11, reg.low3(), reg.low3());
    }

    /// MOV r32, r32 — zero-extend a 32-bit value to 64 bits by clearing
    /// the upper 32 bits. On x86-64, writing to a 32-bit register
    /// implicitly zeroes bits 63:32.
    pub fn zeroExtend32(self: *CodeBuffer, reg: Reg) !void {
        if (reg.isExtended()) try self.rex(false, reg, reg);
        try self.emitByte(0x89);
        try self.modrm(0b11, reg.low3(), reg.low3());
    }

    /// MOV r32, [base + disp32] — 32-bit load (no REX.W, zero-extends to 64).
    pub fn movRegMemNoRex(self: *CodeBuffer, dst: Reg, base: Reg, disp: i32) !void {
        if (dst.isExtended() or base.isExtended()) {
            try self.rex(false, dst, base);
        }
        try self.emitByte(0x8B);
        if (disp == 0 and base.low3() != 5) {
            try self.modrm(0b00, dst.low3(), base.low3());
            if (base.low3() == 4) try self.emitByte(0x24);
        } else {
            try self.modrm(0b10, dst.low3(), base.low3());
            if (base.low3() == 4) try self.emitByte(0x24);
            try self.emitI32(disp);
        }
    }

    /// MOV [base + disp32], r32 — 32-bit store (no REX.W).
    pub fn movMemRegNoRex(self: *CodeBuffer, base: Reg, disp: i32, src: Reg) !void {
        if (src.isExtended() or base.isExtended()) {
            try self.rex(false, src, base);
        }
        try self.emitByte(0x89);
        if (disp == 0 and base.low3() != 5) {
            try self.modrm(0b00, src.low3(), base.low3());
            if (base.low3() == 4) try self.emitByte(0x24);
        } else {
            try self.modrm(0b10, src.low3(), base.low3());
            if (base.low3() == 4) try self.emitByte(0x24);
            try self.emitI32(disp);
        }
    }

    // ── Atomic / LOCK-prefix instructions ─────────────────────────────

    /// Emit the LOCK prefix byte (0xF0) for atomic memory operations.
    pub fn lockPrefix(self: *CodeBuffer) !void {
        try self.emitByte(0xF0);
    }

    /// MFENCE — full memory barrier (0F AE /6).
    pub fn mfence(self: *CodeBuffer) !void {
        try self.emitSlice(&.{ 0x0F, 0xAE, 0xF0 });
    }

    /// Emit prefix bytes for sized operations: operand-size override (0x66)
    /// for 16-bit, REX for extended registers, REX.W for 64-bit.
    fn emitSizedPrefix(self: *CodeBuffer, r: Reg, b: Reg, size: u8) !void {
        switch (size) {
            1, 4 => {
                if (r.isExtended() or b.isExtended()) try self.rex(false, r, b);
            },
            2 => {
                try self.emitByte(0x66);
                if (r.isExtended() or b.isExtended()) try self.rex(false, r, b);
            },
            8 => try self.rexW(r, b),
            else => unreachable,
        }
    }

    /// Emit ModR/M + optional SIB + disp32 for a [base + disp32] memory operand.
    fn emitMemOperand(self: *CodeBuffer, reg_op: u3, base: Reg, disp: i32) !void {
        try self.modrm(0b10, reg_op, base.low3());
        if (base.low3() == 4) try self.emitByte(0x24); // SIB for RSP-based
        try self.emitI32(disp);
    }

    /// Sized MOV load: load `size` bytes from [base + disp] into dst.
    /// Sub-word sizes (1, 2) are zero-extended via MOVZX.
    /// Size 4 uses 32-bit MOV (implicit zero-extend to 64-bit).
    /// Size 8 uses 64-bit MOV (REX.W).
    pub fn movRegMemSized(self: *CodeBuffer, dst: Reg, base: Reg, disp: i32, size: u8) !void {
        switch (size) {
            1 => {
                // MOVZX r32, BYTE PTR [base+disp]: [REX] 0F B6 /r
                if (dst.isExtended() or base.isExtended()) try self.rex(false, dst, base);
                try self.emitSlice(&.{ 0x0F, 0xB6 });
            },
            2 => {
                // MOVZX r32, WORD PTR [base+disp]: [REX] 0F B7 /r
                if (dst.isExtended() or base.isExtended()) try self.rex(false, dst, base);
                try self.emitSlice(&.{ 0x0F, 0xB7 });
            },
            4 => {
                // MOV r32, [base+disp]: [REX] 8B /r
                if (dst.isExtended() or base.isExtended()) try self.rex(false, dst, base);
                try self.emitByte(0x8B);
            },
            8 => {
                // MOV r64, [base+disp]: REX.W 8B /r
                try self.rexW(dst, base);
                try self.emitByte(0x8B);
            },
            else => unreachable,
        }
        try self.emitMemOperand(dst.low3(), base, disp);
    }

    /// Sized MOV store: store `size` bytes from src into [base + disp].
    pub fn movMemRegSized(self: *CodeBuffer, base: Reg, disp: i32, src: Reg, size: u8) !void {
        const opcode: u8 = if (size == 1) 0x88 else 0x89;
        try self.emitSizedPrefix(src, base, size);
        try self.emitByte(opcode);
        try self.emitMemOperand(src.low3(), base, disp);
    }

    /// LOCK XADD [base + disp], src — atomic exchange-and-add.
    /// After execution, src contains the old memory value.
    pub fn lockXadd(self: *CodeBuffer, base: Reg, disp: i32, src: Reg, size: u8) !void {
        try self.emitByte(0xF0); // LOCK
        try self.emitSizedPrefix(src, base, size);
        const opcode: u8 = if (size == 1) 0xC0 else 0xC1;
        try self.emitSlice(&.{ 0x0F, opcode });
        try self.emitMemOperand(src.low3(), base, disp);
    }

    /// LOCK CMPXCHG [base + disp], src — atomic compare-and-exchange.
    /// Compares RAX with [base + disp]; if equal, stores src; otherwise loads into RAX.
    pub fn lockCmpxchg(self: *CodeBuffer, base: Reg, disp: i32, src: Reg, size: u8) !void {
        try self.emitByte(0xF0); // LOCK
        try self.emitSizedPrefix(src, base, size);
        const opcode: u8 = if (size == 1) 0xB0 else 0xB1;
        try self.emitSlice(&.{ 0x0F, opcode });
        try self.emitMemOperand(src.low3(), base, disp);
    }

    /// XCHG [base + disp], reg — atomic exchange (implicit LOCK for memory operands).
    pub fn xchgMemReg(self: *CodeBuffer, base: Reg, disp: i32, src: Reg, size: u8) !void {
        const opcode: u8 = if (size == 1) 0x86 else 0x87;
        try self.emitSizedPrefix(src, base, size);
        try self.emitByte(opcode);
        try self.emitMemOperand(src.low3(), base, disp);
    }

    /// NEG reg — two's complement negate.
    pub fn negReg(self: *CodeBuffer, reg: Reg, size: u8) !void {
        const opcode: u8 = if (size == 1) 0xF6 else 0xF7;
        try self.emitSizedPrefix(.rax, reg, size);
        try self.emitByte(opcode);
        try self.modrm(0b11, 3, reg.low3());
    }

    /// Zero-extend register value to 64-bit based on operand size.
    /// No-op for sizes 4 and 8 (32-bit writes auto-zero-extend; 64-bit is full width).
    pub fn zeroExtendReg(self: *CodeBuffer, reg: Reg, size: u8) !void {
        switch (size) {
            1 => {
                // MOVZX r32, r8: [REX] 0F B6 /r
                if (reg.isExtended()) try self.rex(false, reg, reg);
                try self.emitSlice(&.{ 0x0F, 0xB6 });
                try self.modrm(0b11, reg.low3(), reg.low3());
            },
            2 => {
                // MOVZX r32, r16: [REX] 0F B7 /r
                if (reg.isExtended()) try self.rex(false, reg, reg);
                try self.emitSlice(&.{ 0x0F, 0xB7 });
                try self.modrm(0b11, reg.low3(), reg.low3());
            },
            4, 8 => {}, // 32-bit writes auto-zero-extend; 64-bit is full width
            else => unreachable,
        }
    }

    // ── SSE/SSE2 floating-point instructions ─────────────────────────

    /// MOVSD xmm, [base + disp32] — load f64 from memory
    pub fn movsdLoad(self: *CodeBuffer, dst: Reg, base: Reg, disp: i32) !void {
        try self.emitByte(0xF2);
        if (dst.isExtended() or base.isExtended()) try self.rex(false, dst, base);
        try self.emitSlice(&.{ 0x0F, 0x10 });
        try self.modrm(0b10, dst.low3(), base.low3());
        if (base.low3() == 4) try self.emitByte(0x24);
        try self.emitI32(disp);
    }

    /// MOVSD [base + disp32], xmm — store f64 to memory
    pub fn movsdStore(self: *CodeBuffer, base: Reg, disp: i32, src: Reg) !void {
        try self.emitByte(0xF2);
        if (src.isExtended() or base.isExtended()) try self.rex(false, src, base);
        try self.emitSlice(&.{ 0x0F, 0x11 });
        try self.modrm(0b10, src.low3(), base.low3());
        if (base.low3() == 4) try self.emitByte(0x24);
        try self.emitI32(disp);
    }

    /// MOVSS xmm, [base + disp32] — load f32 from memory
    pub fn movssLoad(self: *CodeBuffer, dst: Reg, base: Reg, disp: i32) !void {
        try self.emitByte(0xF3);
        if (dst.isExtended() or base.isExtended()) try self.rex(false, dst, base);
        try self.emitSlice(&.{ 0x0F, 0x10 });
        try self.modrm(0b10, dst.low3(), base.low3());
        if (base.low3() == 4) try self.emitByte(0x24);
        try self.emitI32(disp);
    }

    /// MOVSS [base + disp32], xmm — store f32 to memory
    pub fn movssStore(self: *CodeBuffer, base: Reg, disp: i32, src: Reg) !void {
        try self.emitByte(0xF3);
        if (src.isExtended() or base.isExtended()) try self.rex(false, src, base);
        try self.emitSlice(&.{ 0x0F, 0x11 });
        try self.modrm(0b10, src.low3(), base.low3());
        if (base.low3() == 4) try self.emitByte(0x24);
        try self.emitI32(disp);
    }

    /// MOVQ xmm, r64 — move GPR to XMM (for bitcast/reinterpret)
    pub fn movqToXmm(self: *CodeBuffer, xmm: Reg, gpr: Reg) !void {
        try self.emitByte(0x66);
        try self.rexW(xmm, gpr);
        try self.emitSlice(&.{ 0x0F, 0x6E });
        try self.modrm(0b11, xmm.low3(), gpr.low3());
    }

    /// MOVQ r64, xmm — move XMM to GPR
    pub fn movqFromXmm(self: *CodeBuffer, gpr: Reg, xmm: Reg) !void {
        try self.emitByte(0x66);
        try self.rexW(xmm, gpr);
        try self.emitSlice(&.{ 0x0F, 0x7E });
        try self.modrm(0b11, xmm.low3(), gpr.low3());
    }

    /// MOVD xmm, r32 — move 32-bit GPR to XMM
    pub fn movdToXmm(self: *CodeBuffer, xmm: Reg, gpr: Reg) !void {
        try self.emitByte(0x66);
        if (xmm.isExtended() or gpr.isExtended()) try self.rex(false, xmm, gpr);
        try self.emitSlice(&.{ 0x0F, 0x6E });
        try self.modrm(0b11, xmm.low3(), gpr.low3());
    }

    /// MOVD r32, xmm — move XMM to 32-bit GPR
    pub fn movdFromXmm(self: *CodeBuffer, gpr: Reg, xmm: Reg) !void {
        try self.emitByte(0x66);
        if (xmm.isExtended() or gpr.isExtended()) try self.rex(false, xmm, gpr);
        try self.emitSlice(&.{ 0x0F, 0x7E });
        try self.modrm(0b11, xmm.low3(), gpr.low3());
    }

    /// Generic SSE binary op: prefix opcode xmm1, xmm2
    fn sseBinOp(self: *CodeBuffer, prefix: u8, opcode: u8, dst: Reg, src: Reg) !void {
        try self.emitByte(prefix);
        if (dst.isExtended() or src.isExtended()) try self.rex(false, dst, src);
        try self.emitSlice(&.{ 0x0F, opcode });
        try self.modrm(0b11, dst.low3(), src.low3());
    }

    // ── f64 binary (ADDSD, SUBSD, MULSD, DIVSD) ──
    pub fn addsd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF2, 0x58, dst, src); }
    pub fn subsd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF2, 0x5C, dst, src); }
    pub fn mulsd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF2, 0x59, dst, src); }
    pub fn divsd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF2, 0x5E, dst, src); }
    pub fn sqrtsd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF2, 0x51, dst, src); }
    pub fn minsd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF2, 0x5D, dst, src); }
    pub fn maxsd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF2, 0x5F, dst, src); }

    // ── f32 binary (ADDSS, SUBSS, MULSS, DIVSS) ──
    pub fn addss(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF3, 0x58, dst, src); }
    pub fn subss(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF3, 0x5C, dst, src); }
    pub fn mulss(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF3, 0x59, dst, src); }
    pub fn divss(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF3, 0x5E, dst, src); }
    pub fn sqrtss(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF3, 0x51, dst, src); }
    pub fn minss(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF3, 0x5D, dst, src); }
    pub fn maxss(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF3, 0x5F, dst, src); }

    // ── Comparisons ──
    pub fn ucomisd(self: *CodeBuffer, a: Reg, b: Reg) !void { try self.sseBinOp(0x66, 0x2E, a, b); }
    pub fn ucomiss(self: *CodeBuffer, a: Reg, b: Reg) !void {
        if (a.isExtended() or b.isExtended()) try self.rex(false, a, b);
        try self.emitSlice(&.{ 0x0F, 0x2E });
        try self.modrm(0b11, a.low3(), b.low3());
    }

    // ── Conversions ──
    /// CVTSI2SD xmm, r64 — convert signed i64 to f64
    pub fn cvtsi2sd(self: *CodeBuffer, xmm: Reg, gpr: Reg) !void {
        try self.emitByte(0xF2);
        try self.rexW(xmm, gpr);
        try self.emitSlice(&.{ 0x0F, 0x2A });
        try self.modrm(0b11, xmm.low3(), gpr.low3());
    }

    /// CVTSI2SS xmm, r64 — convert signed i64 to f32
    pub fn cvtsi2ss(self: *CodeBuffer, xmm: Reg, gpr: Reg) !void {
        try self.emitByte(0xF3);
        try self.rexW(xmm, gpr);
        try self.emitSlice(&.{ 0x0F, 0x2A });
        try self.modrm(0b11, xmm.low3(), gpr.low3());
    }

    /// CVTTSD2SI r64, xmm — truncate f64 to signed i64
    pub fn cvttsd2si(self: *CodeBuffer, gpr: Reg, xmm: Reg) !void {
        try self.emitByte(0xF2);
        try self.rexW(gpr, xmm);
        try self.emitSlice(&.{ 0x0F, 0x2C });
        try self.modrm(0b11, gpr.low3(), xmm.low3());
    }

    /// CVTTSS2SI r64, xmm — truncate f32 to signed i64
    pub fn cvttss2si(self: *CodeBuffer, gpr: Reg, xmm: Reg) !void {
        try self.emitByte(0xF3);
        try self.rexW(gpr, xmm);
        try self.emitSlice(&.{ 0x0F, 0x2C });
        try self.modrm(0b11, gpr.low3(), xmm.low3());
    }

    /// CVTSD2SS xmm, xmm — convert f64 to f32
    pub fn cvtsd2ss(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF2, 0x5A, dst, src); }

    /// CVTSS2SD xmm, xmm — convert f32 to f64
    pub fn cvtss2sd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0xF3, 0x5A, dst, src); }

    // ── Bitwise XMM ──
    /// XORPD xmm, xmm — bitwise XOR for f64
    pub fn xorpd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0x66, 0x57, dst, src); }
    /// ANDPD xmm, xmm — bitwise AND for f64
    pub fn andpd(self: *CodeBuffer, dst: Reg, src: Reg) !void { try self.sseBinOp(0x66, 0x54, dst, src); }

    // ── Function prologue / epilogue ──────────────────────────────────

    /// Emit standard function prologue: push rbp; mov rbp, rsp; sub rsp, frame_size.
    pub fn emitPrologue(self: *CodeBuffer, frame_size: u32) !void {
        try self.pushReg(.rbp);
        try self.movRegReg(.rbp, .rsp);
        if (frame_size > 0) {
            // SUB rsp, imm32
            try self.rexW(.rax, .rsp);
            try self.emitByte(0x81);
            try self.modrm(0b11, 5, Reg.rsp.low3());
            try self.emitU32(frame_size);
        }
    }

    /// Emit standard function epilogue: mov rsp, rbp; pop rbp; ret.
    pub fn emitEpilogue(self: *CodeBuffer) !void {
        try self.movRegReg(.rsp, .rbp);
        try self.popReg(.rbp);
        try self.ret();
    }

    /// Patch a previously emitted 32-bit value at `offset` in the buffer.
    pub fn patchI32(self: *CodeBuffer, offset: usize, val: i32) void {
        const b: [4]u8 = @bitCast(val);
        @memcpy(self.bytes.items[offset..][0..4], &b);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

fn hexEqual(actual: []const u8, expected: []const u8) !void {
    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "CodeBuffer init and deinit lifecycle" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try std.testing.expectEqual(@as(usize, 0), buf.len());
    try std.testing.expectEqualSlices(u8, &[_]u8{}, buf.getCode());
}

test "emit raw bytes and verify contents" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.emitByte(0x90);
    try buf.emitByte(0xCC);
    try buf.emitSlice(&.{ 0x48, 0x89 });

    try std.testing.expectEqual(@as(usize, 4), buf.len());
    try hexEqual(buf.getCode(), &.{ 0x90, 0xCC, 0x48, 0x89 });
}

test "emitPrologue with zero frame size" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.emitPrologue(0);
    // push rbp = 55
    // REX.W mov rbp, rsp = 48 89 E5
    try hexEqual(buf.getCode(), &.{ 0x55, 0x48, 0x89, 0xE5 });
}

test "emitPrologue with nonzero frame size" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.emitPrologue(32);
    // push rbp = 55
    // mov rbp, rsp = 48 89 E5
    // REX.W sub rsp, 32 = 48 81 EC 20 00 00 00
    try hexEqual(buf.getCode(), &.{
        0x55, 0x48, 0x89, 0xE5,
        0x48, 0x81, 0xEC, 0x20, 0x00, 0x00, 0x00,
    });
}

test "emitEpilogue" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.emitEpilogue();
    // mov rsp, rbp = 48 89 EC
    // pop rbp = 5D
    // ret = C3
    try hexEqual(buf.getCode(), &.{ 0x48, 0x89, 0xEC, 0x5D, 0xC3 });
}

test "movRegImm32 rax, 42" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.movRegImm32(.rax, 42);
    // REX.W C7 /0 rax, imm32
    // 48 C7 C0 2A 00 00 00
    try hexEqual(buf.getCode(), &.{ 0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00 });
}

test "movRegImm32 r8, -1" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.movRegImm32(.r8, -1);
    // REX.WB C7 /0 r8, 0xFFFFFFFF
    // 49 C7 C0 FF FF FF FF
    try hexEqual(buf.getCode(), &.{ 0x49, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF });
}

test "movRegImm64" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.movRegImm64(.rax, 0x123456789ABCDEF0);
    // REX.W B8 + rax, imm64
    // 48 B8 F0 DE BC 9A 78 56 34 12
    try hexEqual(buf.getCode(), &.{
        0x48, 0xB8,
        0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12,
    });
}

test "movRegReg" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.movRegReg(.rbx, .rcx);
    // REX.W MOV rbx, rcx = 48 89 CB
    try hexEqual(buf.getCode(), &.{ 0x48, 0x89, 0xCB });
}

test "addRegReg rax, rcx" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.addRegReg(.rax, .rcx);
    // REX.W ADD rax, rcx = 48 01 C8
    try hexEqual(buf.getCode(), &.{ 0x48, 0x01, 0xC8 });
}

test "subRegReg" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.subRegReg(.rax, .rdx);
    // REX.W SUB rax, rdx = 48 29 D0
    try hexEqual(buf.getCode(), &.{ 0x48, 0x29, 0xD0 });
}

test "imulRegReg" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.imulRegReg(.rax, .rcx);
    // REX.W 0F AF /r: 48 0F AF C1
    try hexEqual(buf.getCode(), &.{ 0x48, 0x0F, 0xAF, 0xC1 });
}

test "andRegReg" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.andRegReg(.rax, .rbx);
    // REX.W AND rax, rbx = 48 21 D8
    try hexEqual(buf.getCode(), &.{ 0x48, 0x21, 0xD8 });
}

test "orRegReg" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.orRegReg(.rax, .rcx);
    // REX.W OR rax, rcx = 48 09 C8
    try hexEqual(buf.getCode(), &.{ 0x48, 0x09, 0xC8 });
}

test "xorRegReg" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.xorRegReg(.rax, .rax);
    // REX.W XOR rax, rax = 48 31 C0
    try hexEqual(buf.getCode(), &.{ 0x48, 0x31, 0xC0 });
}

test "cmpRegReg" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.cmpRegReg(.rax, .rcx);
    // REX.W CMP rax, rcx = 48 39 C8
    try hexEqual(buf.getCode(), &.{ 0x48, 0x39, 0xC8 });
}

test "ret" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.ret();
    try hexEqual(buf.getCode(), &.{0xC3});
}

test "nop and int3" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.nop();
    try buf.int3();
    try hexEqual(buf.getCode(), &.{ 0x90, 0xCC });
}

test "pushReg / popReg low registers" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.pushReg(.rax);
    try buf.pushReg(.rbp);
    try buf.popReg(.rbp);
    try buf.popReg(.rax);
    try hexEqual(buf.getCode(), &.{ 0x50, 0x55, 0x5D, 0x58 });
}

test "pushReg / popReg extended registers (r8+)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.pushReg(.r8);
    try buf.pushReg(.r12);
    try buf.pushReg(.r15);
    try buf.popReg(.r15);
    try buf.popReg(.r12);
    try buf.popReg(.r8);
    try hexEqual(buf.getCode(), &.{
        0x41, 0x50, // push r8
        0x41, 0x54, // push r12
        0x41, 0x57, // push r15
        0x41, 0x5F, // pop r15
        0x41, 0x5C, // pop r12
        0x41, 0x58, // pop r8
    });
}

test "callRel32 and jmpRel32" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.callRel32(0x100);
    try buf.jmpRel32(-5);
    try hexEqual(buf.getCode(), &.{
        0xE8, 0x00, 0x01, 0x00, 0x00, // call +256
        0xE9, 0xFB, 0xFF, 0xFF, 0xFF, // jmp -5
    });
}

test "conditional jumps (je, jne)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.je(0x10);
    try buf.jne(0x20);
    try hexEqual(buf.getCode(), &.{
        0x0F, 0x84, 0x10, 0x00, 0x00, 0x00,
        0x0F, 0x85, 0x20, 0x00, 0x00, 0x00,
    });
}

test "patchI32 overwrites previously emitted bytes" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.jmpRel32(0); // placeholder
    try std.testing.expectEqual(@as(usize, 5), buf.len());

    buf.patchI32(1, 42);
    try hexEqual(buf.getCode(), &.{ 0xE9, 0x2A, 0x00, 0x00, 0x00 });
}

test "addRegReg with extended registers (r8, r9)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.addRegReg(.r8, .r9);
    // REX.WRB ADD r8, r9 = 4D 01 C8
    try hexEqual(buf.getCode(), &.{ 0x4D, 0x01, 0xC8 });
}

test "full prologue + epilogue round-trip" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.emitPrologue(0);
    try buf.nop();
    try buf.emitEpilogue();

    try hexEqual(buf.getCode(), &.{
        0x55, 0x48, 0x89, 0xE5, // prologue
        0x90, // nop
        0x48, 0x89, 0xEC, 0x5D, 0xC3, // epilogue
    });
}

// ═══════════════════════════════════════════════════════════════════════
// Atomic instruction tests
// ═══════════════════════════════════════════════════════════════════════

test "lockPrefix" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.lockPrefix();
    try hexEqual(buf.getCode(), &.{0xF0});
}

test "mfence" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.mfence();
    // 0F AE F0
    try hexEqual(buf.getCode(), &.{ 0x0F, 0xAE, 0xF0 });
}

test "movRegMemSized 8-bit (MOVZX byte)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movRegMemSized(.rax, .rcx, 0x10, 1);
    // MOVZX eax, BYTE PTR [rcx+0x10]: 0F B6 81 10000000
    try hexEqual(buf.getCode(), &.{ 0x0F, 0xB6, 0x81, 0x10, 0x00, 0x00, 0x00 });
}

test "movRegMemSized 16-bit (MOVZX word)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movRegMemSized(.rax, .rcx, 0x10, 2);
    // MOVZX eax, WORD PTR [rcx+0x10]: 0F B7 81 10000000
    try hexEqual(buf.getCode(), &.{ 0x0F, 0xB7, 0x81, 0x10, 0x00, 0x00, 0x00 });
}

test "movRegMemSized 32-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movRegMemSized(.rax, .rcx, 0x10, 4);
    // MOV eax, [rcx+0x10]: 8B 81 10000000
    try hexEqual(buf.getCode(), &.{ 0x8B, 0x81, 0x10, 0x00, 0x00, 0x00 });
}

test "movRegMemSized 64-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movRegMemSized(.rax, .rcx, 0x10, 8);
    // REX.W MOV rax, [rcx+0x10]: 48 8B 81 10000000
    try hexEqual(buf.getCode(), &.{ 0x48, 0x8B, 0x81, 0x10, 0x00, 0x00, 0x00 });
}

test "movMemRegSized 8-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movMemRegSized(.rcx, 0x10, .rax, 1);
    // MOV BYTE PTR [rcx+0x10], al: 88 81 10000000
    try hexEqual(buf.getCode(), &.{ 0x88, 0x81, 0x10, 0x00, 0x00, 0x00 });
}

test "setcc on legacy byte reg (al) omits REX" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.setcc(0x4, .rax); // sete al
    try hexEqual(buf.getCode(), &.{ 0x0F, 0x94, 0xC0 });
}

test "setcc on rsi emits mandatory REX for sil" {
    // Without the REX prefix, 0F 94 C6 encodes `sete DH`, which would be
    // incorrect — we need `sete SIL`. Ensure the 0x40 REX is emitted.
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.setcc(0x4, .rsi);
    try hexEqual(buf.getCode(), &.{ 0x40, 0x0F, 0x94, 0xC6 });
}

test "setcc on rdi emits mandatory REX for dil" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.setcc(0x5, .rdi); // setne dil
    try hexEqual(buf.getCode(), &.{ 0x40, 0x0F, 0x95, 0xC7 });
}

test "setcc on extended reg r8 uses REX.B" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.setcc(0x4, .r8); // sete r8b
    // REX.B = 0x41
    try hexEqual(buf.getCode(), &.{ 0x41, 0x0F, 0x94, 0xC0 });
}

test "lzcnt32 on rdx encodes without REX.W" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.lzcnt32(.rdx, .rdx);
    // F3 0F BD /r with ModR/M 11_010_010 = 0xD2 (no REX)
    try hexEqual(buf.getCode(), &.{ 0xF3, 0x0F, 0xBD, 0xD2 });
}

test "tzcnt32 on r9 emits REX.RB" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.tzcnt32(.r9, .r9);
    // F3 REX.RB(0x45) 0F BC /r with ModR/M 11_001_001 = 0xC9
    try hexEqual(buf.getCode(), &.{ 0xF3, 0x45, 0x0F, 0xBC, 0xC9 });
}

test "popcnt32 between different regs" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.popcnt32(.rsi, .rdi);
    // F3 0F B8 /r with ModR/M 11_110_111 = 0xF7 (no REX: both legacy)
    try hexEqual(buf.getCode(), &.{ 0xF3, 0x0F, 0xB8, 0xF7 });
}

test "movRegReg32 rdx to rsi" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movRegReg32(.rsi, .rdx);
    // MOV r/m32, r32 opcode 0x89 with reg=src=rdx=2, rm=dst=rsi=6 → 11_010_110 = D6
    try hexEqual(buf.getCode(), &.{ 0x89, 0xD6 });
}

test "movsxd rsi, edx" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movsxd(.rsi, .rdx);
    // REX.W 63 /r with ModR/M 11_110_010 = 0xF2
    try hexEqual(buf.getCode(), &.{ 0x48, 0x63, 0xF2 });
}

test "movsxByteToReg rdi, dl" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movsxByteToReg(.rdi, .rdx);
    // REX.W 0F BE /r with ModR/M 11_111_010 = 0xFA
    try hexEqual(buf.getCode(), &.{ 0x48, 0x0F, 0xBE, 0xFA });
}

test "movsxWordToReg r8, si" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movsxWordToReg(.r8, .rsi);
    // REX.WR (0x4C) 0F BF /r with ModR/M 11_000_110 = 0xC6
    try hexEqual(buf.getCode(), &.{ 0x4C, 0x0F, 0xBF, 0xC6 });
}

test "movMemRegSized 16-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movMemRegSized(.rcx, 0x10, .rax, 2);
    // MOV WORD PTR [rcx+0x10], ax: 66 89 81 10000000
    try hexEqual(buf.getCode(), &.{ 0x66, 0x89, 0x81, 0x10, 0x00, 0x00, 0x00 });
}

test "movMemRegSized 32-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movMemRegSized(.rcx, 0x10, .rax, 4);
    // MOV DWORD PTR [rcx+0x10], eax: 89 81 10000000
    try hexEqual(buf.getCode(), &.{ 0x89, 0x81, 0x10, 0x00, 0x00, 0x00 });
}

test "movMemRegSized 64-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movMemRegSized(.rcx, 0x10, .rax, 8);
    // REX.W MOV [rcx+0x10], rax: 48 89 81 10000000
    try hexEqual(buf.getCode(), &.{ 0x48, 0x89, 0x81, 0x10, 0x00, 0x00, 0x00 });
}

test "lockXadd 32-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.lockXadd(.rax, 0, .rcx, 4);
    // LOCK XADD [rax+0], ecx: F0 0F C1 88 00000000
    try hexEqual(buf.getCode(), &.{ 0xF0, 0x0F, 0xC1, 0x88, 0x00, 0x00, 0x00, 0x00 });
}

test "lockXadd 64-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.lockXadd(.rax, 0x10, .rcx, 8);
    // LOCK REX.W XADD [rax+0x10], rcx: F0 48 0F C1 88 10000000
    try hexEqual(buf.getCode(), &.{ 0xF0, 0x48, 0x0F, 0xC1, 0x88, 0x10, 0x00, 0x00, 0x00 });
}

test "lockXadd 8-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.lockXadd(.rax, 0, .rcx, 1);
    // LOCK XADD BYTE PTR [rax+0], cl: F0 0F C0 88 00000000
    try hexEqual(buf.getCode(), &.{ 0xF0, 0x0F, 0xC0, 0x88, 0x00, 0x00, 0x00, 0x00 });
}

test "lockCmpxchg 32-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.lockCmpxchg(.rax, 0, .rcx, 4);
    // LOCK CMPXCHG [rax+0], ecx: F0 0F B1 88 00000000
    try hexEqual(buf.getCode(), &.{ 0xF0, 0x0F, 0xB1, 0x88, 0x00, 0x00, 0x00, 0x00 });
}

test "lockCmpxchg 64-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.lockCmpxchg(.rax, 0x10, .rcx, 8);
    // LOCK REX.W CMPXCHG [rax+0x10], rcx: F0 48 0F B1 88 10000000
    try hexEqual(buf.getCode(), &.{ 0xF0, 0x48, 0x0F, 0xB1, 0x88, 0x10, 0x00, 0x00, 0x00 });
}

test "xchgMemReg 32-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.xchgMemReg(.rax, 0, .rcx, 4);
    // XCHG [rax+0], ecx: 87 88 00000000
    try hexEqual(buf.getCode(), &.{ 0x87, 0x88, 0x00, 0x00, 0x00, 0x00 });
}

test "xchgMemReg 64-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.xchgMemReg(.rax, 0x10, .rcx, 8);
    // REX.W XCHG [rax+0x10], rcx: 48 87 88 10000000
    try hexEqual(buf.getCode(), &.{ 0x48, 0x87, 0x88, 0x10, 0x00, 0x00, 0x00 });
}

test "negReg 64-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.negReg(.rcx, 8);
    // REX.W NEG rcx: 48 F7 D9
    try hexEqual(buf.getCode(), &.{ 0x48, 0xF7, 0xD9 });
}

test "negReg 32-bit" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.negReg(.rcx, 4);
    // NEG ecx: F7 D9
    try hexEqual(buf.getCode(), &.{ 0xF7, 0xD9 });
}

test "negReg extended register (r8, 64-bit)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.negReg(.r8, 8);
    // REX.WB NEG r8: 49 F7 D8
    try hexEqual(buf.getCode(), &.{ 0x49, 0xF7, 0xD8 });
}

test "zeroExtendReg 8-bit (MOVZX r32, r8)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.zeroExtendReg(.rax, 1);
    // MOVZX eax, al: 0F B6 C0
    try hexEqual(buf.getCode(), &.{ 0x0F, 0xB6, 0xC0 });
}

test "zeroExtendReg 16-bit (MOVZX r32, r16)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.zeroExtendReg(.rax, 2);
    // MOVZX eax, ax: 0F B7 C0
    try hexEqual(buf.getCode(), &.{ 0x0F, 0xB7, 0xC0 });
}

test "zeroExtendReg 32-bit (no-op)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.zeroExtendReg(.rax, 4);
    try hexEqual(buf.getCode(), &.{});
}

test "zeroExtendReg 64-bit (no-op)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.zeroExtendReg(.rax, 8);
    try hexEqual(buf.getCode(), &.{});
}

// ═══════════════════════════════════════════════════════════════════════
// Immediate-form instruction tests
// ═══════════════════════════════════════════════════════════════════════

test "subRegImm32 rax, 10 uses imm8 form" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.subRegImm32(.rax, 10);
    // REX.W 83 /5 rax, imm8: 48 83 E8 0A
    try hexEqual(buf.getCode(), &.{ 0x48, 0x83, 0xE8, 0x0A });
}

test "subRegImm32 rax, 1000 uses imm32 form" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.subRegImm32(.rax, 1000);
    // REX.W 81 /5 rax, imm32: 48 81 E8 E8 03 00 00
    try hexEqual(buf.getCode(), &.{ 0x48, 0x81, 0xE8, 0xE8, 0x03, 0x00, 0x00 });
}

test "andRegImm32 rax, 0xFF uses imm32 form (0xFF > i8 max)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.andRegImm32(.rax, 0xFF);
    // REX.W 81 /4 rax: 48 81 E0 FF 00 00 00 (0xFF doesn't fit in i8)
    try hexEqual(buf.getCode(), &.{ 0x48, 0x81, 0xE0, 0xFF, 0x00, 0x00, 0x00 });
}

test "andRegImm32 rax, 0x0F uses imm8 form" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.andRegImm32(.rax, 0x0F);
    // REX.W 83 /4 rax, imm8: 48 83 E0 0F
    try hexEqual(buf.getCode(), &.{ 0x48, 0x83, 0xE0, 0x0F });
}

test "orRegImm32 rax, 1 uses imm8 form" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.orRegImm32(.rax, 1);
    // REX.W 83 /1 rax, imm8: 48 83 C8 01
    try hexEqual(buf.getCode(), &.{ 0x48, 0x83, 0xC8, 0x01 });
}

test "xorRegImm32 rax, 0x55 uses imm8 form" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.xorRegImm32(.rax, 0x55);
    // REX.W 83 /6 rax, imm8: 48 83 F0 55
    try hexEqual(buf.getCode(), &.{ 0x48, 0x83, 0xF0, 0x55 });
}

test "cmpRegImm32 rax, 42 uses imm8 form" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.cmpRegImm32(.rax, 42);
    // REX.W 83 /7 rax, imm8: 48 83 F8 2A
    try hexEqual(buf.getCode(), &.{ 0x48, 0x83, 0xF8, 0x2A });
}

test "addRegImm32 rax, -1 uses imm8 form (sign-extended)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.addRegImm32(.rax, -1);
    // REX.W 83 /0 rax, imm8=0xFF: 48 83 C0 FF
    try hexEqual(buf.getCode(), &.{ 0x48, 0x83, 0xC0, 0xFF });
}

test "addRegImm32 rax, 128 uses imm32 form (boundary)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.addRegImm32(.rax, 128);
    // 128 doesn't fit in i8 — must use imm32: 48 81 C0 80 00 00 00
    try hexEqual(buf.getCode(), &.{ 0x48, 0x81, 0xC0, 0x80, 0x00, 0x00, 0x00 });
}

test "xorReg32 rax (zero idiom, 2 bytes)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.xorReg32(.rax);
    // XOR eax, eax: 31 C0
    try hexEqual(buf.getCode(), &.{ 0x31, 0xC0 });
}

test "xorReg32 r8 (extended, 3 bytes)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.xorReg32(.r8);
    // REX 45 31 C0 (XOR r8d, r8d)
    try hexEqual(buf.getCode(), &.{ 0x45, 0x31, 0xC0 });
}



test "leaRegBaseIndex64 rdx, rsi, rdi" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.leaRegBaseIndex64(.rdx, .rsi, .rdi);
    // REX.W=0x48 (no extension), opcode 0x8D, ModR/M 00_010_100=0x14,
    // SIB 00_111_110=0x3E (scale=0, index=rdi=7, base=rsi=6).
    try hexEqual(buf.getCode(), &.{ 0x48, 0x8D, 0x14, 0x3E });
}

test "leaRegBaseIndex64 r12, r13, rdx (base r13 needs disp8=0)" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.leaRegBaseIndex64(.r12, .r13, .rdx);
    // REX: W=1, R=1 (r12), X=0 (rdx low), B=1 (r13) → 0x4D.
    // Opcode 0x8D.
    // ModR/M: mod=01 (disp8), reg=r12.low3=4, rm=100 → 01_100_100 = 0x64.
    // SIB: scale=0, index=rdx=2, base=r13.low3=5 → 00_010_101 = 0x15.
    // disp8: 0x00.
    try hexEqual(buf.getCode(), &.{ 0x4D, 0x8D, 0x64, 0x15, 0x00 });
}

test "leaRegBaseIndex64 rax, r8, r9" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.leaRegBaseIndex64(.rax, .r8, .r9);
    // REX: W=1, R=0, X=1 (r9), B=1 (r8) → 0x4B.
    // Opcode 0x8D. ModR/M mod=00, reg=0, rm=100 → 0x04.
    // SIB scale=0, index=r9.low3=1, base=r8.low3=0 → 00_001_000 = 0x08.
    try hexEqual(buf.getCode(), &.{ 0x4B, 0x8D, 0x04, 0x08 });
}


test "movRegReg elides self-move" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movRegReg(.rax, .rax);
    try buf.movRegReg(.r12, .r12);
    // Both self-moves must produce zero bytes.
    try std.testing.expectEqual(@as(usize, 0), buf.getCode().len);
}

test "movRegReg still emits for distinct regs" {
    var buf = CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    try buf.movRegReg(.rax, .rdx);
    // REX.W 89 /r: 48 89 D0 (mov rax, rdx)
    try hexEqual(buf.getCode(), &.{ 0x48, 0x89, 0xD0 });
}
