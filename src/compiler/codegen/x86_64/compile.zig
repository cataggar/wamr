//! x86-64 IR Compiler
//!
//! Walks IR functions and emits x86-64 machine code via CodeBuffer.
//! Uses a simple register allocator that maps VRegs to physical registers.

const std = @import("std");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");

/// Simple VReg → physical register mapping.
/// Uses a linear scan over a fixed set of scratch registers.
/// Spills to stack when registers are exhausted.
const RegMap = struct {
    entries: std.AutoHashMap(ir.VReg, Location),
    reg_used: [scratch_regs.len]bool = [_]bool{false} ** scratch_regs.len,
    next_stack_offset: u32 = 0,

    const Location = union(enum) {
        reg: emit.Reg,
        stack: u32,
    };

    // Caller-saved scratch registers (excludes rsp/rbp)
    const scratch_regs = [_]emit.Reg{ .rax, .rcx, .rdx, .rsi, .rdi, .r8, .r9, .r10, .r11 };

    fn init(allocator: std.mem.Allocator) RegMap {
        return .{
            .entries = std.AutoHashMap(ir.VReg, Location).init(allocator),
        };
    }

    fn deinit(self: *RegMap) void {
        self.entries.deinit();
    }

    fn assign(self: *RegMap, vreg: ir.VReg) !Location {
        for (scratch_regs, 0..) |r, i| {
            if (!self.reg_used[i]) {
                self.reg_used[i] = true;
                const loc = Location{ .reg = r };
                try self.entries.put(vreg, loc);
                return loc;
            }
        }
        // Spill to stack
        const offset = self.next_stack_offset;
        self.next_stack_offset += 8;
        const loc = Location{ .stack = offset };
        try self.entries.put(vreg, loc);
        return loc;
    }

    fn get(self: *const RegMap, vreg: ir.VReg) ?Location {
        return self.entries.get(vreg);
    }
};

/// Compile an IR function to x86-64 machine code.
pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8 {
    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    var reg_map = RegMap.init(allocator);
    defer reg_map.deinit();

    const frame_size: u32 = func.local_count * 8 + 256;
    try code.emitPrologue(frame_size);

    // Spill parameters from SysV ABI registers to stack frame slots.
    // Params occupy locals[0..param_count], stored at [rbp - 8*(idx+1)].
    const param_regs = [_]emit.Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 };
    const spill_count = @min(func.param_count, param_regs.len);
    for (0..spill_count) |i| {
        try code.movMemReg(.rbp, -@as(i32, @intCast((i + 1) * 8)), param_regs[i]);
    }

    // Track block offsets for branch resolution
    var block_offsets = std.AutoHashMap(ir.BlockId, usize).init(allocator);
    defer block_offsets.deinit();

    var last_was_ret = false;
    // Two-pass: first collect block start offsets, then emit with patching.
    // For now, forward branches use placeholder + patch list.
    var branch_patches: std.ArrayList(BranchPatch) = .empty;
    defer branch_patches.deinit(allocator);

    for (func.blocks.items, 0..) |block, idx| {
        try block_offsets.put(@intCast(idx), code.len());
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInstEx(&code, inst, &reg_map, &branch_patches, func);
        }
    }

    // Patch forward branches
    for (branch_patches.items) |patch| {
        if (block_offsets.get(patch.target_block)) |target_off| {
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.patch_offset + 4)));
            code.patchI32(patch.patch_offset, rel);
        }
    }

    // Add epilogue if the function didn't end with a ret instruction
    if (!last_was_ret) {
        try code.emitEpilogue();
    }

    return code.bytes.toOwnedSlice(allocator);
}

fn isRet(op: ir.Inst.Op) bool {
    return switch (op) {
        .ret => true,
        else => false,
    };
}

const BranchPatch = struct {
    patch_offset: usize, // offset of the rel32 in code buffer
    target_block: ir.BlockId,
};

fn compileInstEx(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    reg_map: *RegMap,
    patches: *std.ArrayList(BranchPatch),
    func: *const ir.IrFunction,
) !void {
    _ = func;
    switch (inst.op) {
        .br => |target| {
            // JMP rel32 (placeholder)
            try code.emitByte(0xE9);
            const patch_off = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = patch_off, .target_block = target });
        },
        .br_if => |br| {
            const cond_loc = reg_map.get(br.cond) orelse return;
            const cond_reg = switch (cond_loc) { .reg => |r| r, .stack => return };
            // TEST cond, cond; JNZ then_block
            try code.testRegReg(cond_reg, cond_reg);
            try code.emitByte(0x0F);
            try code.emitByte(0x85); // JNE rel32
            const then_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = then_patch, .target_block = br.then_block });
            // Fall through to else_block (JMP rel32)
            try code.emitByte(0xE9);
            const else_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = else_patch, .target_block = br.else_block });
        },
        else => try compileInst(code, inst, reg_map),
    }
}

fn compileInst(code: *emit.CodeBuffer, inst: ir.Inst, reg_map: *RegMap) !void {
    switch (inst.op) {
        .iconst_32 => |val| {
            const dest = inst.dest orelse return;
            const loc = try reg_map.assign(dest);
            switch (loc) {
                .reg => |r| try code.movRegImm32(r, val),
                .stack => {},
            }
        },
        .iconst_64 => |val| {
            const dest = inst.dest orelse return;
            const loc = try reg_map.assign(dest);
            switch (loc) {
                .reg => |r| try code.movRegImm64(r, @bitCast(val)),
                .stack => {},
            }
        },
        .add => |bin| try emitBinOp(code, inst, bin, reg_map, .add),
        .sub => |bin| try emitBinOp(code, inst, bin, reg_map, .sub),
        .mul => |bin| try emitBinOp(code, inst, bin, reg_map, .mul),
        .@"and" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"and"),
        .@"or" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"or"),
        .xor => |bin| try emitBinOp(code, inst, bin, reg_map, .xor),
        .shl => |bin| try emitShift(code, inst, bin, reg_map, .shl),
        .shr_s => |bin| try emitShift(code, inst, bin, reg_map, .shr_s),
        .shr_u => |bin| try emitShift(code, inst, bin, reg_map, .shr_u),

        // Division/remainder: uses RAX:RDX pair
        .div_s, .div_u, .rem_s, .rem_u => |bin| try emitDivRem(code, inst, bin, reg_map),

        // Comparisons: cmp + setcc + movzx
        inline .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => |bin, tag| {
            try emitCmp(code, inst, bin, reg_map, tag);
        },

        // Unary ops
        .eqz => |src| try emitEqz(code, inst, src, reg_map),
        .clz => |src| try emitUnary(code, inst, src, reg_map, .clz),
        .ctz => |src| try emitUnary(code, inst, src, reg_map, .ctz),
        .popcnt => |src| try emitUnary(code, inst, src, reg_map, .popcnt),

        // Local variable access
        .local_get => |idx| {
            const dest = inst.dest orelse return;
            const loc = try reg_map.assign(dest);
            switch (loc) {
                .reg => |r| try code.movRegMem(r, .rbp, -@as(i32, @intCast((idx + 1) * 8))),
                .stack => {},
            }
        },
        .local_set => |ls| {
            const src_loc = reg_map.get(ls.val) orelse return;
            switch (src_loc) {
                .reg => |r| try code.movMemReg(.rbp, -@as(i32, @intCast((ls.idx + 1) * 8)), r),
                .stack => {},
            }
        },

        .ret => |maybe_val| {
            if (maybe_val) |val| {
                if (reg_map.get(val)) |loc| {
                    switch (loc) {
                        .reg => |r| {
                            if (r != .rax) try code.movRegReg(.rax, r);
                        },
                        .stack => {},
                    }
                }
            }
            try code.emitEpilogue();
        },
        .@"unreachable" => try code.int3(),
        .select => |sel| try emitSelect(code, inst, sel, reg_map),

        // Memory load: base_reg + offset → dest_reg
        .load => |ld| {
            const dest = inst.dest orelse return;
            const base_loc = reg_map.get(ld.base) orelse return;
            const dest_loc = try reg_map.assign(dest);
            const base_reg = switch (base_loc) { .reg => |r| r, .stack => return };
            const dest_reg = switch (dest_loc) { .reg => |r| r, .stack => return };
            // MOV dest, [base + offset] (32-bit load, zero-extended)
            if (ld.offset > 0) {
                try code.addRegImm32(base_reg, @intCast(ld.offset));
            }
            // Use 32-bit mov for i32 loads
            try code.movRegMemNoRex(dest_reg, base_reg, 0);
        },

        // Memory store: val → [base + offset]
        .store => |st| {
            const base_loc = reg_map.get(st.base) orelse return;
            const val_loc = reg_map.get(st.val) orelse return;
            const base_reg = switch (base_loc) { .reg => |r| r, .stack => return };
            const val_reg = switch (val_loc) { .reg => |r| r, .stack => return };
            if (st.offset > 0) {
                try code.addRegImm32(base_reg, @intCast(st.offset));
            }
            try code.movMemRegNoRex(base_reg, 0, val_reg);
        },

        // Function call: setup args, call rel32, collect result
        .call => |cl| {
            const dest = inst.dest orelse return;
            // For now: emit a stub that just returns 0 in the dest register.
            // Full inter-function calls require knowing function offsets at link time.
            _ = cl;
            const dest_loc = try reg_map.assign(dest);
            switch (dest_loc) {
                .reg => |r| try code.movRegImm32(r, 0),
                .stack => {},
            }
        },
        else => {},
    }
}

const BinOpKind = enum { add, sub, mul, @"and", @"or", xor };

fn emitBinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    kind: BinOpKind,
) !void {
    const dest = inst.dest orelse return;
    const lhs_loc = reg_map.get(bin.lhs) orelse return;
    const rhs_loc = reg_map.get(bin.rhs) orelse return;
    const dest_loc = try reg_map.assign(dest);

    const dest_reg = switch (dest_loc) {
        .reg => |r| r,
        .stack => return,
    };
    const lhs_reg = switch (lhs_loc) {
        .reg => |r| r,
        .stack => return,
    };
    const rhs_reg = switch (rhs_loc) {
        .reg => |r| r,
        .stack => return,
    };

    try code.movRegReg(dest_reg, lhs_reg);
    switch (kind) {
        .add => try code.addRegReg(dest_reg, rhs_reg),
        .sub => try code.subRegReg(dest_reg, rhs_reg),
        .mul => try code.imulRegReg(dest_reg, rhs_reg),
        .@"and" => try code.andRegReg(dest_reg, rhs_reg),
        .@"or" => try code.orRegReg(dest_reg, rhs_reg),
        .xor => try code.xorRegReg(dest_reg, rhs_reg),
    }
}

// ── Shift operations ──────────────────────────────────────────────────

const ShiftKind = enum { shl, shr_s, shr_u };

fn emitShift(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    kind: ShiftKind,
) !void {
    const dest = inst.dest orelse return;
    const lhs_loc = reg_map.get(bin.lhs) orelse return;
    const rhs_loc = reg_map.get(bin.rhs) orelse return;
    const dest_loc = try reg_map.assign(dest);

    const dest_reg = switch (dest_loc) { .reg => |r| r, .stack => return };
    const lhs_reg = switch (lhs_loc) { .reg => |r| r, .stack => return };
    const rhs_reg = switch (rhs_loc) { .reg => |r| r, .stack => return };

    // x86 shifts require count in CL
    if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
    if (dest_reg != lhs_reg) try code.movRegReg(dest_reg, lhs_reg);

    // REX.W + D3 /opcode
    const opcode_ext: u3 = switch (kind) {
        .shl => 4,
        .shr_u => 5,
        .shr_s => 7,
    };
    try code.rexW(.rax, dest_reg);
    try code.emitByte(0xD3);
    try code.modrm(0b11, opcode_ext, dest_reg.low3());
}

// ── Division / remainder ──────────────────────────────────────────────

fn emitDivRem(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const lhs_loc = reg_map.get(bin.lhs) orelse return;
    const rhs_loc = reg_map.get(bin.rhs) orelse return;
    const dest_loc = try reg_map.assign(dest);

    const lhs_reg = switch (lhs_loc) { .reg => |r| r, .stack => return };
    const rhs_reg = switch (rhs_loc) { .reg => |r| r, .stack => return };
    const dest_reg = switch (dest_loc) { .reg => |r| r, .stack => return };

    // Move dividend to RAX
    if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);

    const is_signed = (inst.op == .div_s or inst.op == .rem_s);
    const is_rem = (inst.op == .rem_s or inst.op == .rem_u);

    // Sign/zero extend RAX into RDX:RAX
    if (is_signed) {
        try code.cqo(); // sign-extend RAX → RDX:RAX
    } else {
        try code.xorRegReg(.rdx, .rdx); // zero-extend
    }

    // Divisor must not be in RAX or RDX
    var divisor = rhs_reg;
    if (divisor == .rax or divisor == .rdx) {
        divisor = .r10;
        try code.movRegReg(.r10, rhs_reg);
    }

    // IDIV/DIV r/m64
    if (is_signed) {
        try code.idivReg(divisor);
    } else {
        try code.divReg(divisor);
    }

    // Result: quotient in RAX, remainder in RDX
    const result_reg: emit.Reg = if (is_rem) .rdx else .rax;
    if (dest_reg != result_reg) try code.movRegReg(dest_reg, result_reg);
}

// ── Comparison operations ─────────────────────────────────────────────

fn emitCmp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    comptime op: std.meta.Tag(ir.Inst.Op),
) !void {
    const dest = inst.dest orelse return;
    const lhs_loc = reg_map.get(bin.lhs) orelse return;
    const rhs_loc = reg_map.get(bin.rhs) orelse return;
    const dest_loc = try reg_map.assign(dest);

    const lhs_reg = switch (lhs_loc) { .reg => |r| r, .stack => return };
    const rhs_reg = switch (rhs_loc) { .reg => |r| r, .stack => return };
    const dest_reg = switch (dest_loc) { .reg => |r| r, .stack => return };

    // CMP lhs, rhs
    try code.cmpRegReg(lhs_reg, rhs_reg);

    // SETcc to low byte of dest, then MOVZX to clear upper bytes
    // Map IR comparison to x86 condition code
    const cc: u4 = comptime switch (op) {
        .eq => 0x4,   // E/Z
        .ne => 0x5,   // NE/NZ
        .lt_s => 0xC, // L (SF≠OF)
        .lt_u => 0x2, // B (CF=1)
        .gt_s => 0xF, // G (ZF=0, SF=OF)
        .gt_u => 0x7, // A (CF=0, ZF=0)
        .le_s => 0xE, // LE (ZF=1 or SF≠OF)
        .le_u => 0x6, // BE (CF=1 or ZF=1)
        .ge_s => 0xD, // GE (SF=OF)
        .ge_u => 0x3, // AE (CF=0)
        else => unreachable,
    };
    try code.setcc(cc, dest_reg);
    try code.movzxByte(dest_reg, dest_reg);
}

// ── Unary operations ──────────────────────────────────────────────────

const UnaryKind = enum { clz, ctz, popcnt };

fn emitUnary(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    src: ir.VReg,
    reg_map: *RegMap,
    kind: UnaryKind,
) !void {
    const dest = inst.dest orelse return;
    const src_loc = reg_map.get(src) orelse return;
    const dest_loc = try reg_map.assign(dest);
    const src_reg = switch (src_loc) { .reg => |r| r, .stack => return };
    const dest_reg = switch (dest_loc) { .reg => |r| r, .stack => return };

    switch (kind) {
        .clz => try code.lzcnt(dest_reg, src_reg),
        .ctz => try code.tzcnt(dest_reg, src_reg),
        .popcnt => try code.popcntReg(dest_reg, src_reg),
    }
}

fn emitEqz(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    src: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src_loc = reg_map.get(src) orelse return;
    const dest_loc = try reg_map.assign(dest);
    const src_reg = switch (src_loc) { .reg => |r| r, .stack => return };
    const dest_reg = switch (dest_loc) { .reg => |r| r, .stack => return };

    // TEST src, src; SETZ dest
    try code.testRegReg(src_reg, src_reg);
    try code.setcc(0x4, dest_reg); // SETZ (ZF=1)
    try code.movzxByte(dest_reg, dest_reg);
}

// ── Select (conditional move) ─────────────────────────────────────────

fn emitSelect(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    sel: @TypeOf(@as(ir.Inst.Op, undefined).select),
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const cond_loc = reg_map.get(sel.cond) orelse return;
    const true_loc = reg_map.get(sel.if_true) orelse return;
    const false_loc = reg_map.get(sel.if_false) orelse return;
    const dest_loc = try reg_map.assign(dest);

    const cond_reg = switch (cond_loc) { .reg => |r| r, .stack => return };
    const true_reg = switch (true_loc) { .reg => |r| r, .stack => return };
    const false_reg = switch (false_loc) { .reg => |r| r, .stack => return };
    const dest_reg = switch (dest_loc) { .reg => |r| r, .stack => return };

    // dest = false_val; TEST cond; CMOVNZ dest, true_val
    try code.movRegReg(dest_reg, false_reg);
    try code.testRegReg(cond_reg, cond_reg);
    try code.cmovnz(dest_reg, true_reg);
}

/// Result of compiling an IR module.
pub const CompileResult = struct {
    code: []u8,
    offsets: []u32,
};

/// Compile all functions in an IR module to x86-64 machine code.
pub fn compileModule(ir_module: *const ir.IrModule, allocator: std.mem.Allocator) !CompileResult {
    var all_code: std.ArrayList(u8) = .empty;
    errdefer all_code.deinit(allocator);
    var offsets: std.ArrayList(u32) = .empty;
    errdefer offsets.deinit(allocator);

    for (ir_module.functions.items) |func| {
        try offsets.append(allocator, @intCast(all_code.items.len));
        const func_code = try compileFunction(&func, allocator);
        defer allocator.free(func_code);
        try all_code.appendSlice(allocator, func_code);
    }

    return .{
        .code = try all_code.toOwnedSlice(allocator),
        .offsets = try offsets.toOwnedSlice(allocator),
    };
}

// ── Tests ──────────────────────────────────────────────────────────

test "compileFunction: iconst_32 + ret produces mov and epilogue" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should have prologue + mov + epilogue
    try std.testing.expect(code.len > 10);
    // Verify the immediate 42 (0x2A) appears in the code
    var found_42 = false;
    for (code) |b| {
        if (b == 0x2A) {
            found_42 = true;
            break;
        }
    }
    try std.testing.expect(found_42);
    // Last byte should be ret (0xC3)
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: two iconsts + add + ret produces reasonable code" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Prologue + 2 mov-imm + mov+add + mov-to-rax + epilogue
    try std.testing.expect(code.len > 15);
    try std.testing.expect(code.len < 200);
    // Last byte should be ret
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileModule: two functions have correct offsets" {
    const allocator = std.testing.allocator;
    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    _ = try f1.newBlock();
    const v0 = f1.newVReg();
    try f1.getBlock(0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try f1.getBlock(0).append(.{ .op = .{ .ret = v0 } });
    _ = try ir_module.addFunction(f1);

    var f2 = ir.IrFunction.init(allocator, 0, 1, 0);
    _ = try f2.newBlock();
    const v1 = f2.newVReg();
    try f2.getBlock(0).append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
    try f2.getBlock(0).append(.{ .op = .{ .ret = v1 } });
    _ = try ir_module.addFunction(f2);

    const result = try compileModule(&ir_module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    try std.testing.expectEqual(@as(usize, 2), result.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), result.offsets[0]);
    try std.testing.expect(result.offsets[1] > 0);
}

test "compileFunction: empty function produces prologue and epilogue" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    _ = try func.newBlock();

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should have prologue + epilogue
    try std.testing.expect(code.len > 0);
    // Last byte should be ret (0xC3)
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}
