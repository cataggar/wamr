//! x86-64 IR Compiler
//!
//! Walks IR functions and emits x86-64 machine code via CodeBuffer.
//! Uses a stack-based operand model: all intermediate values live on the
//! stack frame (memory), using RAX and RCX as temporary scratch registers.

const std = @import("std");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");

/// Stack-based operand model. All intermediate values live on the stack
/// frame at [rbp + offset], using negative offsets that grow downward.
/// Locals occupy [rbp - 8*1] through [rbp - 8*N], and the operand stack
/// starts immediately below at [rbp - 8*(N+1)].
const OperandStack = struct {
    /// Offset from RBP to the next free operand stack slot (negative, grows down).
    top: i32,
    /// Base offset (marks bottom of operand stack area, i.e. the initial top).
    base: i32,

    fn init(local_count: u32) OperandStack {
        const base_off = -@as(i32, @intCast((local_count + 1) * 8));
        return .{ .top = base_off, .base = base_off };
    }

    fn push(self: *OperandStack, code: *emit.CodeBuffer, src: emit.Reg) !void {
        try code.movMemReg(.rbp, self.top, src);
        self.top -= 8;
    }

    fn pop(self: *OperandStack, code: *emit.CodeBuffer, dst: emit.Reg) !void {
        self.top += 8;
        try code.movRegMem(dst, .rbp, self.top);
    }

    fn depth(self: *const OperandStack) u32 {
        return @intCast(@divExact(self.base - self.top, 8));
    }

    /// Save the current stack depth so we can restore it later (e.g. across blocks).
    fn save(self: *const OperandStack) i32 {
        return self.top;
    }

    /// Restore a previously saved stack depth.
    fn restore(self: *OperandStack, saved: i32) void {
        self.top = saved;
    }
};

/// Compile an IR function to x86-64 machine code.
pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8 {
    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    // Frame: locals + 64 operand stack slots
    const frame_size: u32 = (func.local_count + 64) * 8;
    try code.emitPrologue(frame_size);

    // Spill parameters from SysV ABI registers to stack frame slots.
    const param_regs = [_]emit.Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 };
    const spill_count = @min(func.param_count, param_regs.len);
    for (0..spill_count) |i| {
        try code.movMemReg(.rbp, -@as(i32, @intCast((i + 1) * 8)), param_regs[i]);
    }

    var stack = OperandStack.init(func.local_count);

    // Track block offsets for branch resolution
    var block_offsets = std.AutoHashMap(ir.BlockId, usize).init(allocator);
    defer block_offsets.deinit();

    var last_was_ret = false;
    var branch_patches: std.ArrayList(BranchPatch) = .empty;
    defer branch_patches.deinit(allocator);
    var call_patches: std.ArrayList(CallPatch) = .empty;
    defer call_patches.deinit(allocator);

    for (func.blocks.items, 0..) |block, idx| {
        try block_offsets.put(@intCast(idx), code.len());
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInst(&code, inst, &stack, &branch_patches, &call_patches);
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
    patch_offset: usize,
    target_block: ir.BlockId,
};

const CallPatch = struct {
    patch_offset: usize,
    target_func_idx: u32,
};

fn compileInst(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    stack: *OperandStack,
    patches: *std.ArrayList(BranchPatch),
    call_patches: *std.ArrayList(CallPatch),
) !void {
    switch (inst.op) {
        // ── Constants ─────────────────────────────────────────────────
        .iconst_32 => |val| {
            try code.movRegImm32(.rax, val);
            try stack.push(code, .rax);
        },
        .iconst_64 => |val| {
            try code.movRegImm64(.rax, @bitCast(val));
            try stack.push(code, .rax);
        },
        .fconst_32 => |val| {
            try code.movRegImm32(.rax, @bitCast(val));
            try stack.push(code, .rax);
        },
        .fconst_64 => |val| {
            try code.movRegImm64(.rax, @bitCast(val));
            try stack.push(code, .rax);
        },

        // ── Binary arithmetic ─────────────────────────────────────────
        .add => {
            try stack.pop(code, .rcx); // rhs
            try stack.pop(code, .rax); // lhs
            try code.addRegReg(.rax, .rcx);
            try stack.push(code, .rax);
        },
        .sub => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            try code.subRegReg(.rax, .rcx);
            try stack.push(code, .rax);
        },
        .mul => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            try code.imulRegReg(.rax, .rcx);
            try stack.push(code, .rax);
        },
        .@"and" => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            try code.andRegReg(.rax, .rcx);
            try stack.push(code, .rax);
        },
        .@"or" => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            try code.orRegReg(.rax, .rcx);
            try stack.push(code, .rax);
        },
        .xor => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            try code.xorRegReg(.rax, .rcx);
            try stack.push(code, .rax);
        },

        // ── Shifts ────────────────────────────────────────────────────
        .shl => {
            try stack.pop(code, .rcx); // count → RCX (CL)
            try stack.pop(code, .rax); // value
            // SHL RAX, CL: REX.W D3 /4
            try code.rexW(.rax, .rax);
            try code.emitByte(0xD3);
            try code.modrm(0b11, 4, emit.Reg.rax.low3());
            try stack.push(code, .rax);
        },
        .shr_s => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            // SAR RAX, CL: REX.W D3 /7
            try code.rexW(.rax, .rax);
            try code.emitByte(0xD3);
            try code.modrm(0b11, 7, emit.Reg.rax.low3());
            try stack.push(code, .rax);
        },
        .shr_u => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            // SHR RAX, CL: REX.W D3 /5
            try code.rexW(.rax, .rax);
            try code.emitByte(0xD3);
            try code.modrm(0b11, 5, emit.Reg.rax.low3());
            try stack.push(code, .rax);
        },
        .rotl => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            // ROL RAX, CL: REX.W D3 /0
            try code.rexW(.rax, .rax);
            try code.emitByte(0xD3);
            try code.modrm(0b11, 0, emit.Reg.rax.low3());
            try stack.push(code, .rax);
        },
        .rotr => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            // ROR RAX, CL: REX.W D3 /1
            try code.rexW(.rax, .rax);
            try code.emitByte(0xD3);
            try code.modrm(0b11, 1, emit.Reg.rax.low3());
            try stack.push(code, .rax);
        },

        // ── Division / remainder ──────────────────────────────────────
        .div_s, .div_u, .rem_s, .rem_u => {
            try stack.pop(code, .rcx); // divisor
            try stack.pop(code, .rax); // dividend

            const is_signed = (inst.op == .div_s or inst.op == .rem_s);
            const is_rem = (inst.op == .rem_s or inst.op == .rem_u);

            if (is_signed) {
                try code.cqo();
            } else {
                try code.xorRegReg(.rdx, .rdx);
            }

            if (is_signed) {
                try code.idivReg(.rcx);
            } else {
                try code.divReg(.rcx);
            }

            if (is_rem) {
                try stack.push(code, .rdx);
            } else {
                try stack.push(code, .rax);
            }
        },

        // ── Comparisons ───────────────────────────────────────────────
        inline .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => |_, tag| {
            try stack.pop(code, .rcx); // rhs
            try stack.pop(code, .rax); // lhs
            try code.cmpRegReg(.rax, .rcx);

            const cc: u4 = comptime switch (tag) {
                .eq => 0x4,
                .ne => 0x5,
                .lt_s => 0xC,
                .lt_u => 0x2,
                .gt_s => 0xF,
                .gt_u => 0x7,
                .le_s => 0xE,
                .le_u => 0x6,
                .ge_s => 0xD,
                .ge_u => 0x3,
                else => unreachable,
            };
            try code.setcc(cc, .rax);
            try code.movzxByte(.rax, .rax);
            try stack.push(code, .rax);
        },

        // ── Unary ops ─────────────────────────────────────────────────
        .eqz => {
            try stack.pop(code, .rax);
            try code.testRegReg(.rax, .rax);
            try code.setcc(0x4, .rax); // SETZ
            try code.movzxByte(.rax, .rax);
            try stack.push(code, .rax);
        },
        .clz => {
            try stack.pop(code, .rax);
            try code.lzcnt(.rax, .rax);
            try stack.push(code, .rax);
        },
        .ctz => {
            try stack.pop(code, .rax);
            try code.tzcnt(.rax, .rax);
            try stack.push(code, .rax);
        },
        .popcnt => {
            try stack.pop(code, .rax);
            try code.popcntReg(.rax, .rax);
            try stack.push(code, .rax);
        },

        // ── Local variable access ─────────────────────────────────────
        .local_get => |idx| {
            try code.movRegMem(.rax, .rbp, -@as(i32, @intCast((idx + 1) * 8)));
            try stack.push(code, .rax);
        },
        .local_set => |ls| {
            try stack.pop(code, .rax);
            try code.movMemReg(.rbp, -@as(i32, @intCast((ls.idx + 1) * 8)), .rax);
        },

        // ── Return ────────────────────────────────────────────────────
        .ret => |maybe_val| {
            if (maybe_val != null) {
                try stack.pop(code, .rax);
            }
            try code.emitEpilogue();
        },

        .@"unreachable" => try code.int3(),

        // ── Select (conditional move) ─────────────────────────────────
        .select => {
            try stack.pop(code, .rax); // cond
            try stack.pop(code, .rcx); // if_false
            try stack.pop(code, .rdx); // if_true
            // TEST RAX, RAX; if cond!=0, pick true (rdx), else keep false (rcx)
            try code.testRegReg(.rax, .rax);
            try code.cmovnz(.rcx, .rdx);
            try stack.push(code, .rcx);
        },

        // ── Memory load ───────────────────────────────────────────────
        .load => |ld| {
            try stack.pop(code, .rax); // base address
            if (ld.offset > 0) {
                try code.addRegImm32(.rax, @intCast(ld.offset));
            }
            try code.movRegMemNoRex(.rax, .rax, 0);
            try stack.push(code, .rax);
        },

        // ── Memory store ──────────────────────────────────────────────
        .store => |st| {
            try stack.pop(code, .rcx); // value
            try stack.pop(code, .rax); // base address
            if (st.offset > 0) {
                try code.addRegImm32(.rax, @intCast(st.offset));
            }
            try code.movMemRegNoRex(.rax, 0, .rcx);
        },

        // ── Branches ──────────────────────────────────────────────────
        .br => |target| {
            try code.emitByte(0xE9); // JMP rel32
            const patch_off = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = patch_off, .target_block = target });
        },
        .br_if => |br| {
            // Pop condition from operand stack
            try stack.pop(code, .rax);
            try code.testRegReg(.rax, .rax);
            // JNE then_block
            try code.emitByte(0x0F);
            try code.emitByte(0x85);
            const then_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = then_patch, .target_block = br.then_block });
            // JMP else_block (fallthrough)
            try code.emitByte(0xE9);
            const else_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = else_patch, .target_block = br.else_block });
        },

        // ── Function calls ────────────────────────────────────────────
        .call => |cl| {
            const abi_regs = [_]emit.Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 };
            const n_args = cl.arg_count;

            // Pop args from operand stack in reverse order into ABI registers.
            // Wasm pushes args left-to-right, so first arg is deepest.
            // We pop in reverse (last arg first), store to temp slots, then
            // load in order to ABI regs.
            if (n_args > 0 and n_args <= abi_regs.len) {
                // Pop all args into temporary frame slots (reuse operand stack area)
                // We pop in reverse order (topmost = last arg)
                var i: u32 = n_args;
                while (i > 0) {
                    i -= 1;
                    try stack.pop(code, .rax);
                    // Store to a temp area: use [rbp - frame_temp_base - i*8]
                    // We use R11 as a scratch for the offset calculation
                    // Actually, simpler: just load into the right ABI reg directly
                    // Since we pop in reverse, arg index = i
                    try code.movRegReg(abi_regs[i], .rax);
                }
            } else if (n_args > abi_regs.len) {
                // More than 6 args: pop extras to stack, then first 6 to regs
                // For now, handle only register-passed args
                var i: u32 = n_args;
                while (i > abi_regs.len) {
                    i -= 1;
                    try stack.pop(code, .rax); // discard overflow args for now
                }
                while (i > 0) {
                    i -= 1;
                    try stack.pop(code, .rax);
                    try code.movRegReg(abi_regs[i], .rax);
                }
            }

            // Emit CALL rel32 (placeholder, patched later)
            try code.emitByte(0xE8);
            const patch_off = code.len();
            try code.emitI32(0);
            try call_patches.append(code.allocator, .{
                .patch_offset = patch_off,
                .target_func_idx = cl.func_idx,
            });

            // Push return value (RAX) to operand stack
            try stack.push(code, .rax);
        },

        // ── Global access (stubs) ─────────────────────────────────────
        .global_get => {
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .global_set => {
            try stack.pop(code, .rax);
        },

        // ── Type conversions (pass-through for integer types) ─────────
        .wrap_i64, .extend_i32_s, .extend_i32_u,
        .extend8_s, .extend16_s, .extend32_s,
        .reinterpret,
        => {
            // Pop, push back (value stays on stack, no-op for now)
            try stack.pop(code, .rax);
            try stack.push(code, .rax);
        },

        // ── Float/conversion stubs (pop input, push placeholder) ──────
        .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
        .convert_s, .convert_u, .demote_f64, .promote_f32,
        .trunc_sat_f32_s, .trunc_sat_f32_u, .trunc_sat_f64_s, .trunc_sat_f64_u,
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest,
        => {
            try stack.pop(code, .rax);
            try stack.push(code, .rax);
        },

        // ── Float binary stubs ────────────────────────────────────────
        .f_min, .f_max, .f_copysign => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            try stack.push(code, .rax);
        },

        // ── Atomic stubs ──────────────────────────────────────────────
        .atomic_fence => {},
        .atomic_load => {
            try stack.pop(code, .rax);
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .atomic_store => {
            try stack.pop(code, .rcx); // val
            try stack.pop(code, .rax); // base
        },
        .atomic_rmw => {
            try stack.pop(code, .rcx); // val
            try stack.pop(code, .rax); // base
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .atomic_cmpxchg => {
            try stack.pop(code, .rcx); // replacement
            try stack.pop(code, .rdx); // expected
            try stack.pop(code, .rax); // base
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .atomic_notify => {
            try stack.pop(code, .rcx); // count
            try stack.pop(code, .rax); // base
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .atomic_wait => {
            try stack.pop(code, .rcx); // timeout
            try stack.pop(code, .rdx); // expected
            try stack.pop(code, .rax); // base
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
    }
}

/// Result of compiling an IR module.
pub const CompileResult = struct {
    code: []u8,
    offsets: []u32,
};

/// Compile all functions in an IR module to x86-64 machine code.
/// After compiling all functions, patches inter-function call sites.
pub fn compileModule(ir_module: *const ir.IrModule, allocator: std.mem.Allocator) !CompileResult {
    var all_code: std.ArrayList(u8) = .empty;
    errdefer all_code.deinit(allocator);
    var offsets: std.ArrayList(u32) = .empty;
    errdefer offsets.deinit(allocator);

    // Per-function call patches, accumulated across all functions
    var global_call_patches: std.ArrayList(GlobalCallPatch) = .empty;
    defer global_call_patches.deinit(allocator);

    for (ir_module.functions.items) |func| {
        const func_start = all_code.items.len;
        try offsets.append(allocator, @intCast(func_start));

        // Compile function, collecting call patches
        var code = emit.CodeBuffer.init(allocator);
        errdefer code.deinit();

        var stack = OperandStack.init(func.local_count);

        const frame_size: u32 = (func.local_count + 64) * 8;
        try code.emitPrologue(frame_size);

        const param_regs = [_]emit.Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 };
        const spill_count = @min(func.param_count, param_regs.len);
        for (0..spill_count) |i| {
            try code.movMemReg(.rbp, -@as(i32, @intCast((i + 1) * 8)), param_regs[i]);
        }

        var block_offsets = std.AutoHashMap(ir.BlockId, usize).init(allocator);
        defer block_offsets.deinit();
        var branch_patches: std.ArrayList(BranchPatch) = .empty;
        defer branch_patches.deinit(allocator);
        var call_patches: std.ArrayList(CallPatch) = .empty;
        defer call_patches.deinit(allocator);

        var last_was_ret = false;
        for (func.blocks.items, 0..) |block, idx| {
            try block_offsets.put(@intCast(idx), code.len());
            for (block.instructions.items) |inst| {
                last_was_ret = isRet(inst.op);
                try compileInst(&code, inst, &stack, &branch_patches, &call_patches);
            }
        }

        // Patch intra-function branches
        for (branch_patches.items) |patch| {
            if (block_offsets.get(patch.target_block)) |target_off| {
                const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.patch_offset + 4)));
                code.patchI32(patch.patch_offset, rel);
            }
        }

        if (!last_was_ret) {
            try code.emitEpilogue();
        }

        // Record call patches with global offsets
        for (call_patches.items) |cp| {
            try global_call_patches.append(allocator, .{
                .patch_offset = func_start + cp.patch_offset,
                .target_func_idx = cp.target_func_idx,
            });
        }

        try all_code.appendSlice(allocator, code.getCode());
        code.deinit();
    }

    // Patch inter-function calls
    for (global_call_patches.items) |patch| {
        if (patch.target_func_idx < offsets.items.len) {
            const target_off = offsets.items[patch.target_func_idx];
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.patch_offset + 4)));
            const b: [4]u8 = @bitCast(rel);
            @memcpy(all_code.items[patch.patch_offset..][0..4], &b);
        }
    }

    return .{
        .code = try all_code.toOwnedSlice(allocator),
        .offsets = try offsets.toOwnedSlice(allocator),
    };
}

const GlobalCallPatch = struct {
    patch_offset: usize,
    target_func_idx: u32,
};

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

    // Should have prologue + mov + stack ops + epilogue
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

    try std.testing.expect(code.len > 15);
    try std.testing.expect(code.len < 300);
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

    try std.testing.expect(code.len > 0);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "OperandStack: push and pop maintain depth" {
    var code = emit.CodeBuffer.init(std.testing.allocator);
    defer code.deinit();

    var stack = OperandStack.init(2); // 2 locals
    try std.testing.expectEqual(@as(u32, 0), stack.depth());

    try stack.push(&code, .rax);
    try std.testing.expectEqual(@as(u32, 1), stack.depth());

    try stack.push(&code, .rcx);
    try std.testing.expectEqual(@as(u32, 2), stack.depth());

    try stack.pop(&code, .rax);
    try std.testing.expectEqual(@as(u32, 1), stack.depth());

    try stack.pop(&code, .rax);
    try std.testing.expectEqual(@as(u32, 0), stack.depth());
}
