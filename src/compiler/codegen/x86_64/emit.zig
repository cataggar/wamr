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
    fn rexW(self: *CodeBuffer, r: Reg, b: Reg) !void {
        try self.rex(true, r, b);
    }

    // ── ModR/M byte ───────────────────────────────────────────────────

    fn modrm(self: *CodeBuffer, mod: u2, reg_op: u3, rm: u3) !void {
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

test {
    _ = @import("compile.zig");
}
