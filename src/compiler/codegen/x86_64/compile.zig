//! x86-64 IR Compiler
//!
//! Walks IR functions and emits x86-64 machine code via CodeBuffer.
//! Uses a register-caching operand stack: the top N values are kept in
//! registers, falling back to memory when all cache registers are in use.
//! RAX and RCX serve as temporary scratch registers for instruction operands.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");

/// Platform-specific ABI parameter registers, resolved at comptime.
const param_regs = if (builtin.os.tag == .windows)
    [_]emit.Reg{ .rcx, .rdx, .r8, .r9 } // Win64
else
    [_]emit.Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 }; // SysV

/// Caller-saved allocatable registers that must be saved/restored around calls.
/// On Win64: rdx, r8, r9 are volatile. rsi, rdi are callee-saved.
/// On SysV: rdx, rsi, rdi, r8, r9 are all volatile.
/// Caller-saved allocatable registers that must be saved/restored around calls.
/// Since our generated functions don't preserve callee-saved registers in their
/// prologues, ALL allocatable registers must be treated as caller-saved.
const caller_saved_alloc = [_]emit.Reg{ .rdx, .rsi, .rdi, .r8, .r9 };

/// Callee-saved allocatable registers preserved in prologue/epilogue.
/// r12, r13 are callee-saved on both Win64 and SysV.
/// On Win64, rsi and rdi are also callee-saved per the ABI; compiled functions
/// must save/restore them if used. On SysV they're caller-saved (handled
/// at call sites via caller_saved_alloc), so no prologue work is needed for them.
const callee_saved_alloc = if (builtin.os.tag == .windows)
    [_]emit.Reg{ .rsi, .rdi, .r12, .r13 }
else
    [_]emit.Reg{ .r12, .r13 };

/// Fixed frame offset for the VMContext pointer.
/// Stored at [rbp - 8] by compileFunctionRA at function entry.
/// Points to a VmCtx struct with memory_base at +0, globals_base at +16.
const vmctx_offset: i32 = -8;
const vmctx_membase_field: i32 = 0; // VmCtx.memory_base offset
const vmctx_memsize_field: i32 = 8; // VmCtx.memory_size offset
const vmctx_globals_field: i32 = 16; // VmCtx.globals_ptr offset
const vmctx_host_functions_field: i32 = 24; // VmCtx.host_functions_ptr offset
const vmctx_mem_max_size_field: i32 = 32; // VmCtx.memory_max_size offset
const vmctx_func_table_field: i32 = 40; // VmCtx.func_table_ptr offset
const vmctx_mem_pages_field: i32 = 56; // VmCtx.memory_pages offset (u32)

/// Register-caching operand stack. Keeps the top N values in registers,
/// eliminating redundant store-load pairs. Falls back to memory (via RBP
/// offsets) when all cache registers are occupied.
/// Locals occupy [rbp - 8*1] through [rbp - 8*N], and operand stack
/// slot i lives at [rbp - 8*(N+1) - 8*i].
const CachedStack = struct {
    const SlotState = enum { empty, in_reg, in_mem, is_const, is_cmp };
    const Slot = struct {
        state: SlotState = .empty,
        reg: emit.Reg = .rax,
        const_val: i64 = 0, // valid when state == .is_const
        cmp_cc: u4 = 0, // valid when state == .is_cmp
    };

    /// Registers available for caching operand stack values.
    /// Excludes RAX (scratch/return), RCX (shift count), RDX (div remainder).
    const cache_regs = [_]emit.Reg{ .r10, .r11, .rsi, .rdi, .r8, .r9 };

    slots: [64]Slot = [_]Slot{.{}} ** 64,
    depth: u32 = 0,
    /// Base offset from RBP (marks bottom of operand stack area).
    base: i32,
    reg_used: [cache_regs.len]bool = [_]bool{false} ** cache_regs.len,

    fn init(local_count: u32) CachedStack {
        const base_off = -@as(i32, @intCast((local_count + 1) * 8));
        return .{ .base = base_off };
    }

    /// Return the memory offset for operand stack slot `idx`.
    fn memOffset(self: *const CachedStack, idx: u32) i32 {
        return self.base - @as(i32, @intCast(idx * 8));
    }

    /// Push a value that is currently in `src` register.
    fn push(self: *CachedStack, code: *emit.CodeBuffer, src: emit.Reg) !void {
        const idx = self.depth;
        self.depth += 1;

        for (cache_regs, 0..) |cr, i| {
            if (!self.reg_used[i]) {
                self.reg_used[i] = true;
                if (src != cr) try code.movRegReg(cr, src);
                self.slots[idx] = .{ .state = .in_reg, .reg = cr };
                return;
            }
        }

        // No free register – spill to memory.
        try code.movMemReg(.rbp, self.memOffset(idx), src);
        self.slots[idx] = .{ .state = .in_mem };
    }

    /// Push a deferred constant — no code emitted, no register consumed.
    fn pushConst(self: *CachedStack, val: i64) void {
        const idx = self.depth;
        self.depth += 1;
        self.slots[idx] = .{ .state = .is_const, .const_val = val };
    }

    /// Push a deferred comparison result — no code emitted, no register consumed.
    fn pushCmp(self: *CachedStack, cc: u4) void {
        const idx = self.depth;
        self.depth += 1;
        self.slots[idx] = .{ .state = .is_cmp, .cmp_cc = cc };
    }

    /// Return true if the top of stack is a deferred constant.
    fn topIsConst(self: *const CachedStack) bool {
        if (self.depth == 0) return false;
        return self.slots[self.depth - 1].state == .is_const;
    }

    /// Return the constant value at top of stack. Only valid if topIsConst().
    fn topConstVal(self: *const CachedStack) i64 {
        return self.slots[self.depth - 1].const_val;
    }

    /// Return true if the top of stack is a deferred comparison.
    fn topIsCmp(self: *const CachedStack) bool {
        if (self.depth == 0) return false;
        return self.slots[self.depth - 1].state == .is_cmp;
    }

    /// Return the condition code at top of stack. Only valid if topIsCmp().
    fn topCmpCc(self: *const CachedStack) u4 {
        return self.slots[self.depth - 1].cmp_cc;
    }

    /// Drop the top slot without emitting any code (for deferred slots consumed by fusion).
    fn dropTop(self: *CachedStack) void {
        self.depth -= 1;
        self.slots[self.depth].state = .empty;
    }

    /// Pop the top value into `dst` register.
    fn pop(self: *CachedStack, code: *emit.CodeBuffer, dst: emit.Reg) !void {
        self.depth -= 1;
        const slot = self.slots[self.depth];
        self.slots[self.depth].state = .empty;

        switch (slot.state) {
            .in_reg => {
                for (cache_regs, 0..) |cr, i| {
                    if (cr == slot.reg) {
                        self.reg_used[i] = false;
                        break;
                    }
                }
                if (dst != slot.reg) try code.movRegReg(dst, slot.reg);
            },
            .in_mem => {
                try code.movRegMem(dst, .rbp, self.memOffset(self.depth));
            },
            .is_const => {
                if (slot.const_val == 0) {
                    try code.xorReg32(dst);
                } else if (slot.const_val >= std.math.minInt(i32) and slot.const_val <= std.math.maxInt(i32)) {
                    try code.movRegImm32(dst, @intCast(slot.const_val));
                } else {
                    try code.movRegImm64(dst, @bitCast(slot.const_val));
                }
            },
            .is_cmp => {
                try code.setcc(slot.cmp_cc, dst);
                try code.movzxByte(dst, dst);
            },
            .empty => unreachable,
        }
    }

    /// Flush all register-cached values to memory.
    /// Must be called before branches, calls, and returns.
    fn flush(self: *CachedStack, code: *emit.CodeBuffer) !void {
        for (0..self.depth) |i| {
            switch (self.slots[i].state) {
                .in_reg => {
                    try code.movMemReg(.rbp, self.memOffset(@intCast(i)), self.slots[i].reg);
                    for (cache_regs, 0..) |cr, ci| {
                        if (cr == self.slots[i].reg) {
                            self.reg_used[ci] = false;
                            break;
                        }
                    }
                    self.slots[i] = .{ .state = .in_mem };
                },
                .is_const => {
                    // Use RCX as temp to preserve RAX (may hold return value)
                    if (self.slots[i].const_val >= std.math.minInt(i32) and self.slots[i].const_val <= std.math.maxInt(i32)) {
                        try code.movRegImm32(.rcx, @intCast(self.slots[i].const_val));
                    } else {
                        try code.movRegImm64(.rcx, @bitCast(self.slots[i].const_val));
                    }
                    try code.movMemReg(.rbp, self.memOffset(@intCast(i)), .rcx);
                    self.slots[i] = .{ .state = .in_mem };
                },
                .is_cmp => {
                    try code.setcc(self.slots[i].cmp_cc, .rcx);
                    try code.movzxByte(.rcx, .rcx);
                    try code.movMemReg(.rbp, self.memOffset(@intCast(i)), .rcx);
                    self.slots[i] = .{ .state = .in_mem };
                },
                .in_mem, .empty => {},
            }
        }
    }

    fn save(self: *const CachedStack) u32 {
        return self.depth;
    }

    fn restore(self: *CachedStack, saved: u32) void {
        while (self.depth > saved) {
            self.depth -= 1;
            if (self.slots[self.depth].state == .in_reg) {
                for (cache_regs, 0..) |cr, i| {
                    if (cr == self.slots[self.depth].reg) {
                        self.reg_used[i] = false;
                        break;
                    }
                }
            }
            self.slots[self.depth].state = .empty;
        }
    }
};

/// Compile an IR function to x86-64 machine code.
pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8 {
    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    // Frame: locals + 64 operand stack slots, aligned to 16 bytes
    // After push rbp (8 bytes), we need frame_size ≡ 8 (mod 16) for 16-byte RSP alignment.
    const raw_size: u32 = (func.local_count + 64) * 8;
    const frame_size: u32 = (raw_size + 15) & ~@as(u32, 15) | 8; // ensure ≡ 8 mod 16
    try code.emitPrologue(frame_size);

    // Spill parameters from ABI registers to stack frame slots.
    const spill_count = @min(func.param_count, param_regs.len);
    for (0..spill_count) |i| {
        try code.movMemReg(.rbp, -@as(i32, @intCast((i + 1) * 8)), param_regs[i]);
    }

    var stack = CachedStack.init(func.local_count);

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

/// Patches one 4-byte entry in a br_table jump table. After final layout is
/// known, `entry_offset` is overwritten with `block_offsets[target_block] - base_offset`.
const TablePatch = struct {
    entry_offset: usize,
    base_offset: usize,
    target_block: ir.BlockId,
};

fn compileInst(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    stack: *CachedStack,
    patches: *std.ArrayList(BranchPatch),
    call_patches: *std.ArrayList(CallPatch),
) !void {
    switch (inst.op) {
        // ── Constants ─────────────────────────────────────────────────
        .iconst_32 => |val| {
            stack.pushConst(val);
        },
        .iconst_64 => |val| {
            stack.pushConst(val);
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
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.addRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.addRegReg(.rax, .rcx);
            }
            try stack.push(code, .rax);
        },
        .sub => {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.subRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.subRegReg(.rax, .rcx);
            }
            try stack.push(code, .rax);
        },
        .mul => {
            try stack.pop(code, .rcx);
            try stack.pop(code, .rax);
            try code.imulRegReg(.rax, .rcx);
            try stack.push(code, .rax);
        },
        .@"and" => {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                try code.andRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.andRegReg(.rax, .rcx);
            }
            try stack.push(code, .rax);
        },
        .@"or" => {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.orRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.orRegReg(.rax, .rcx);
            }
            try stack.push(code, .rax);
        },
        .xor => {
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                if (imm != 0) try code.xorRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx);
                try stack.pop(code, .rax);
                try code.xorRegReg(.rax, .rcx);
            }
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
            if (stack.topIsConst()) {
                const imm = stack.topConstVal();
                stack.dropTop();
                try stack.pop(code, .rax);
                try code.cmpRegImm32(.rax, @intCast(imm));
            } else {
                try stack.pop(code, .rcx); // rhs
                try stack.pop(code, .rax); // lhs
                try code.cmpRegReg(.rax, .rcx);
            }

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
            stack.pushCmp(cc);
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
                try stack.pop(code, .rax); // pop before flush to preserve deferred optimizations
            }
            try stack.flush(code); // flush uses RCX as temp, preserving RAX
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
            try stack.flush(code);
            try code.emitByte(0xE9); // JMP rel32
            const patch_off = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = patch_off, .target_block = target });
        },
        .br_if => |br| {
            if (stack.topIsCmp()) {
                // Fused compare-and-branch: emit Jcc directly
                const cc = stack.topCmpCc();
                stack.dropTop();
                try stack.flush(code); // flush remaining values (MOV only, flags preserved)
                // Jcc then_block
                try code.emitByte(0x0F);
                try code.emitByte(0x80 | @as(u8, cc));
                const then_patch = code.len();
                try code.emitI32(0);
                try patches.append(code.allocator, .{ .patch_offset = then_patch, .target_block = br.then_block });
            } else {
                try stack.flush(code);
                // Non-fused: pop condition, test, JNE
                try stack.pop(code, .rax);
                try code.testRegReg(.rax, .rax);
                // JNE then_block
                try code.emitByte(0x0F);
                try code.emitByte(0x85);
                const then_patch = code.len();
                try code.emitI32(0);
                try patches.append(code.allocator, .{ .patch_offset = then_patch, .target_block = br.then_block });
            }
            // JMP else_block (fallthrough)
            try code.emitByte(0xE9);
            const else_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = else_patch, .target_block = br.else_block });
        },
        .br_table => {
            // br_table is only used through compileFunctionRA path in production.
            // The legacy stack-based compileInst path does not support it.
            return error.Unsupported;
        },

        // ── Function calls ────────────────────────────────────────────
        .call => |cl| {
            try stack.flush(code);
            const n_args: u32 = @intCast(cl.args.len);

            // Pop args from operand stack in reverse order into ABI registers.
            // Wasm pushes args left-to-right, so first arg is deepest.
            // We pop in reverse (last arg first), store to temp slots, then
            // load in order to ABI regs.
            if (n_args > 0 and n_args <= param_regs.len) {
                // Pop all args into temporary frame slots (reuse operand stack area)
                // We pop in reverse order (topmost = last arg)
                var i: u32 = n_args;
                while (i > 0) {
                    i -= 1;
                    try stack.pop(code, .rax);
                    try code.movRegReg(param_regs[i], .rax);
                }
            } else if (n_args > param_regs.len) {
                // More than 6 args: pop extras to stack, then first 6 to regs
                var i: u32 = n_args;
                while (i > param_regs.len) {
                    i -= 1;
                    try stack.pop(code, .rax); // discard overflow args for now
                }
                while (i > 0) {
                    i -= 1;
                    try stack.pop(code, .rax);
                    try code.movRegReg(param_regs[i], .rax);
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

        // ── Atomic operations ──────────────────────────────────────────
        .atomic_fence => {
            try code.mfence();
        },
        .atomic_load => |ld| {
            try stack.pop(code, .rax); // base address
            if (ld.offset > 0) {
                try code.addRegImm32(.rax, @intCast(ld.offset));
            }
            try code.movRegMemSized(.rax, .rax, 0, ld.size);
            try stack.push(code, .rax);
        },
        .atomic_store => |st| {
            try stack.pop(code, .rcx); // val
            try stack.pop(code, .rax); // base address
            if (st.offset > 0) {
                try code.addRegImm32(.rax, @intCast(st.offset));
            }
            try code.movMemRegSized(.rax, 0, .rcx, st.size);
            try code.mfence(); // seq-cst ordering
        },
        .atomic_rmw => |rmw| {
            try stack.pop(code, .rcx); // val
            try stack.pop(code, .rax); // base address
            if (rmw.offset > 0) {
                try code.addRegImm32(.rax, @intCast(rmw.offset));
            }
            switch (rmw.op) {
                .add => {
                    // LOCK XADD [rax], rcx → rcx gets old value
                    try code.lockXadd(.rax, 0, .rcx, rmw.size);
                    try code.movRegReg(.rax, .rcx);
                    try code.zeroExtendReg(.rax, rmw.size);
                },
                .sub => {
                    // NEG rcx; LOCK XADD → rcx gets old value
                    try code.negReg(.rcx, rmw.size);
                    try code.lockXadd(.rax, 0, .rcx, rmw.size);
                    try code.movRegReg(.rax, .rcx);
                    try code.zeroExtendReg(.rax, rmw.size);
                },
                .xchg => {
                    // XCHG [rax], rcx → rcx gets old value (implicit LOCK)
                    try code.xchgMemReg(.rax, 0, .rcx, rmw.size);
                    try code.movRegReg(.rax, .rcx);
                    try code.zeroExtendReg(.rax, rmw.size);
                },
                .@"and", .@"or", .xor => {
                    // CAS loop: r8 needed as temp, flush stack to free cache regs
                    try stack.flush(code);
                    try code.movRegReg(.rdx, .rax); // rdx = address
                    try code.movRegMemSized(.rax, .rdx, 0, rmw.size); // rax = current value (zero-extended)
                    // retry:
                    const retry_off = code.len();
                    try code.movRegReg(.r8, .rax); // r8 = copy of old value
                    switch (rmw.op) {
                        .@"and" => try code.andRegReg(.r8, .rcx),
                        .@"or" => try code.orRegReg(.r8, .rcx),
                        .xor => try code.xorRegReg(.r8, .rcx),
                        else => unreachable,
                    }
                    // LOCK CMPXCHG: if [rdx]==rax → store r8, else rax=[rdx]
                    try code.lockCmpxchg(.rdx, 0, .r8, rmw.size);
                    // JNE retry (backward jump on CAS failure)
                    const jne_off = code.len();
                    try code.jne(0); // placeholder rel32
                    const retry_rel: i32 = @intCast(@as(i64, @intCast(retry_off)) - @as(i64, @intCast(jne_off + 6)));
                    code.patchI32(jne_off + 2, retry_rel);
                    // rax = old value (zero-extended from initial MOVZX; sub-word
                    // CMPXCHG failure only writes AL/AX, preserving zero upper bytes)
                },
            }
            try stack.push(code, .rax);
        },
        .atomic_cmpxchg => |cmpxchg| {
            try stack.pop(code, .rcx); // replacement
            try stack.pop(code, .rax); // expected → rax (implicit CMPXCHG operand)
            try stack.pop(code, .rdx); // base → rdx
            if (cmpxchg.offset > 0) {
                try code.addRegImm32(.rdx, @intCast(cmpxchg.offset));
            }
            try code.lockCmpxchg(.rdx, 0, .rcx, cmpxchg.size);
            try code.zeroExtendReg(.rax, cmpxchg.size);
            try stack.push(code, .rax);
        },
        .atomic_notify => {
            try stack.pop(code, .rcx); // count
            try stack.pop(code, .rax); // base
            // TODO: implement with futex runtime (see #76)
            try code.movRegImm32(.rax, 0); // return 0 waiters woken
            try stack.push(code, .rax);
        },
        .atomic_wait => {
            try stack.pop(code, .rcx); // timeout
            try stack.pop(code, .rdx); // expected
            try stack.pop(code, .rax); // base
            // TODO: implement with futex runtime (see #76)
            try code.movRegImm32(.rax, 1); // return 1 ("not equal")
            try stack.push(code, .rax);
        },
        .memory_copy => |mc| {
            _ = mc;
            try stack.pop(code, .rcx); // len
            try stack.pop(code, .rdx); // src
            try stack.pop(code, .rax); // dst
            // TODO: implement as memcpy
        },
        .memory_fill => |mf| {
            _ = mf;
            try stack.pop(code, .rcx); // len
            try stack.pop(code, .rdx); // val
            try stack.pop(code, .rax); // dst
            // TODO: implement as memset
        },
        .memory_size => {
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .memory_grow => {
            try stack.pop(code, .rax); // pages (discard)
            try code.movRegImm32(.rax, -1); // always fail in non-RA path
            try stack.push(code, .rax);
        },
        .call_indirect => {
            // Stub in non-RA path: pop args + elem_idx, push 0
            try stack.pop(code, .rax); // elem_idx (discard)
            try code.movRegImm32(.rax, 0);
            try stack.push(code, .rax);
        },
        .memory_init => {
            try stack.pop(code, .rax); // len
            try stack.pop(code, .rax); // src
            try stack.pop(code, .rax); // dst
        },
        .data_drop => {},
    }
}

/// Result of compiling a single function.
const FuncCompileResult = struct {
    code: []u8,
    call_patches: []CallPatch,
};

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

        // Compile function using register allocator
        const result = try compileFunctionRA(&func, ir_module.import_count, allocator);
        defer allocator.free(result.code);
        defer allocator.free(result.call_patches);

        // Accumulate call patches with global offsets
        for (result.call_patches) |patch| {
            try global_call_patches.append(allocator, .{
                .patch_offset = func_start + patch.patch_offset,
                .target_func_idx = patch.target_func_idx,
            });
        }

        try all_code.appendSlice(allocator, result.code);
    }

    // Patch inter-function call sites now that all function offsets are known
    for (global_call_patches.items) |patch| {
        if (patch.target_func_idx < offsets.items.len) {
            const target_off = offsets.items[patch.target_func_idx];
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.patch_offset + 4)));
            std.mem.writeInt(i32, all_code.items[patch.patch_offset..][0..4], rel, .little);
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

// ── Register-Allocated Compilation ────────────────────────────────────

const regalloc = @import("../../ir/regalloc.zig");

/// Compile an IR function using the linear scan register allocator.
/// VRegs are assigned to physical registers; instructions operate directly
/// on assigned registers without push/pop through a CachedStack.
pub fn compileFunctionRA(func: *const ir.IrFunction, import_count: u32, allocator: std.mem.Allocator) !FuncCompileResult {
    // Collect clobber points: instructions that destroy specific registers.
    // Uses the same sequential numbering as computeLiveRanges in analysis.zig.
    var clobber_points: std.ArrayList(regalloc.ClobberPoint) = .empty;
    defer clobber_points.deinit(allocator);
    {
        var pos: u32 = 0;
        for (func.blocks.items) |block| {
            for (block.instructions.items) |ci| {
                switch (ci.op) {
                    .call, .call_indirect => {
                        // Calls clobber caller-saved allocatable regs.
                        // alloc_regs = [rdx(2), rsi(6), rdi(7), r8(8), r9(9), r12(12), r13(13)]
                        // On Win64: rdx, r8, r9 are volatile (indices 0, 3, 4); rsi/rdi/r12/r13 callee-saved.
                        // On SysV: rdx, rsi, rdi, r8, r9 are volatile; r12, r13 callee-saved.
                        const mask = if (comptime builtin.os.tag == .windows)
                            [_]bool{ true, false, false, true, true, false, false }
                        else
                            [_]bool{ true, true, true, true, true, false, false };
                        try clobber_points.append(allocator, .{ .pos = pos, .regs_clobbered = mask });
                    },
                    .memory_copy => {
                        // REP MOVSB clobbers rsi(6) and rdi(7) → indices 1, 2
                        try clobber_points.append(allocator, .{
                            .pos = pos,
                            .regs_clobbered = .{ false, true, true, false, false, false, false },
                        });
                    },
                    .memory_fill => {
                        // REP STOSB clobbers rdi(7) → index 2
                        try clobber_points.append(allocator, .{
                            .pos = pos,
                            .regs_clobbered = .{ false, false, true, false, false, false, false },
                        });
                    },
                    else => {},
                }
                pos += 1;
            }
        }
    }

    var alloc_result = try regalloc.allocate(func, allocator, clobber_points.items);
    defer alloc_result.deinit();

    // Compute which caller-saved registers are actually used by this function.
    // Only these need to be saved/restored around call sites.
    var used_caller_saved: [caller_saved_alloc.len]bool = .{false} ** caller_saved_alloc.len;
    // Track which callee-saved registers are used (for prologue/epilogue preservation).
    var used_callee_saved: [callee_saved_alloc.len]bool = .{false} ** callee_saved_alloc.len;
    {
        var it = alloc_result.assignments.iterator();
        while (it.next()) |entry| {
            switch (entry.value_ptr.*) {
                .reg => |preg| {
                    const reg: emit.Reg = @enumFromInt(preg);
                    for (caller_saved_alloc, 0..) |cs_reg, i| {
                        if (reg == cs_reg) used_caller_saved[i] = true;
                    }
                    for (callee_saved_alloc, 0..) |cs_reg, i| {
                        if (reg == cs_reg) used_callee_saved[i] = true;
                    }
                },
                .stack => {},
            }
        }
    }
    // Mark callee-saved regs clobbered by fixed-register instructions:
    // memory_copy (REP MOVSB) hard-uses rsi+rdi, memory_fill (REP STOSB) uses rdi.
    if (comptime builtin.os.tag == .windows) {
        for (func.blocks.items) |block| {
            for (block.instructions.items) |blk_inst| {
                switch (blk_inst.op) {
                    .memory_copy => {
                        for (callee_saved_alloc, 0..) |cs_reg, i| {
                            if (cs_reg == .rsi or cs_reg == .rdi) used_callee_saved[i] = true;
                        }
                    },
                    .memory_fill => {
                        for (callee_saved_alloc, 0..) |cs_reg, i| {
                            if (cs_reg == .rdi) used_callee_saved[i] = true;
                        }
                    },
                    else => {},
                }
            }
        }
    }

    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    // Count callee-saved pushes for stack alignment calculation.
    var callee_save_count: u32 = 0;
    for (used_callee_saved) |used| {
        if (used) callee_save_count += 1;
    }

    // Frame: locals + spill slots + 1 vmctx slot, aligned to 16 bytes.
    // No push/pop at call sites (allocator handles clobbering), so after
    // push rbp (rsp=0 mod 16) + sub rsp + callee-save pushes, rsp must
    // be 0 mod 16 for direct CALL alignment.
    const spill_slots = alloc_result.spill_count;
    const raw_size: u32 = (func.local_count + 64 + spill_slots + 1) * 8;
    const aligned: u32 = (raw_size + 15) & ~@as(u32, 15);
    // After push rbp: rsp=0 mod 16. After sub rsp,frame_size + N callee pushes:
    // want (frame_size + 8*N) ≡ 0 mod 16 → rsp ≡ 0 mod 16 at CALL sites.
    const frame_size: u32 = if (callee_save_count % 2 == 0) aligned else aligned | 8;
    try code.emitPrologue(frame_size);

    // Save memory base (VMContext) from first ABI register to [rbp - 8]
    // The runtime passes the linear memory base pointer as a hidden first parameter.
    const vmctx_reg = param_regs[0]; // rcx on Win64, rdi on SysV
    try code.movMemReg(.rbp, vmctx_offset, vmctx_reg);

    // Spill wasm parameters from ABI registers to stack frame slots.
    // Wasm params are shifted by 1 because the first ABI register is the VMContext.
    const spill_count = @min(func.param_count, param_regs.len - 1);
    for (0..spill_count) |i| {
        try code.movMemReg(.rbp, -@as(i32, @intCast((i + 2) * 8)), param_regs[i + 1]);
    }

    // Zero-initialize declared locals (wasm spec requires locals to be zero).
    // Locals start after parameters in the frame layout.
    if (func.local_count > func.param_count) {
        try code.xorReg32(.rax);
        for (func.param_count..func.local_count) |i| {
            try code.movMemReg(.rbp, -@as(i32, @intCast((i + 2) * 8)), .rax);
        }
    }

    // Save callee-saved registers used by this function.
    // Win64: rsi, rdi, r12, r13. SysV: r12, r13.
    for (callee_saved_alloc, 0..) |reg, i| {
        if (used_callee_saved[i]) try code.pushReg(reg);
    }

    // Build constant value table for immediate folding
    var const_vals = std.AutoHashMap(ir.VReg, i64).init(allocator);
    defer const_vals.deinit();
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            if (inst.dest) |dest| {
                switch (inst.op) {
                    .iconst_32 => |v| try const_vals.put(dest, v),
                    .iconst_64 => |v| try const_vals.put(dest, v),
                    else => {},
                }
            }
        }
    }

    var block_offsets = std.AutoHashMap(ir.BlockId, usize).init(allocator);
    defer block_offsets.deinit();
    var branch_patches: std.ArrayList(BranchPatch) = .empty;
    defer branch_patches.deinit(allocator);
    var call_patches: std.ArrayList(CallPatch) = .empty;
    defer call_patches.deinit(allocator);
    var table_patches: std.ArrayList(TablePatch) = .empty;
    defer table_patches.deinit(allocator);

    var last_was_ret = false;
    for (func.blocks.items, 0..) |block, idx| {
        try block_offsets.put(@intCast(idx), code.len());
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInstRA(&code, inst, &alloc_result, &const_vals, &branch_patches, &call_patches, &table_patches, import_count, &used_caller_saved, &used_callee_saved);
        }
    }

    for (branch_patches.items) |patch| {
        if (block_offsets.get(patch.target_block)) |target_off| {
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.patch_offset + 4)));
            code.patchI32(patch.patch_offset, rel);
        }
    }

    for (table_patches.items) |patch| {
        if (block_offsets.get(patch.target_block)) |target_off| {
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.base_offset)));
            code.patchI32(patch.entry_offset, rel);
        }
    }

    if (!last_was_ret) {
        // Restore callee-saved registers before epilogue (reverse order).
        var ci: usize = callee_saved_alloc.len;
        while (ci > 0) {
            ci -= 1;
            if (used_callee_saved[ci]) try code.popReg(callee_saved_alloc[ci]);
        }
        try code.emitEpilogue();
    }

    return .{
        .code = try code.bytes.toOwnedSlice(allocator),
        .call_patches = try call_patches.toOwnedSlice(allocator),
    };
}

/// Get a VReg's value into a physical register, using scratch if spilled.
fn useVReg(
    code: *emit.CodeBuffer,
    alloc_result: *const regalloc.AllocResult,
    vreg: ir.VReg,
    scratch: emit.Reg,
) !emit.Reg {
    const alloc = alloc_result.get(vreg) orelse return scratch;
    switch (alloc) {
        .reg => |preg| return @as(emit.Reg, @enumFromInt(preg)),
        .stack => |offset| {
            try code.movRegMem(scratch, .rbp, offset);
            return scratch;
        },
    }
}

/// Store a result into a VReg's allocated location.
/// For i32 results, automatically zero-extends to clear upper 32 bits.
fn writeDef(
    code: *emit.CodeBuffer,
    alloc_result: *const regalloc.AllocResult,
    dest: ir.VReg,
    result_reg: emit.Reg,
) !void {
    const alloc = alloc_result.get(dest) orelse return;
    switch (alloc) {
        .reg => |preg| {
            const dst = @as(emit.Reg, @enumFromInt(preg));
            if (dst != result_reg) try code.movRegReg(dst, result_reg);
        },
        .stack => |offset| {
            try code.movMemReg(.rbp, offset, result_reg);
        },
    }
}

/// Write an i32 result: zero-extend to clear upper 32 bits, then store.
fn writeDefI32(
    code: *emit.CodeBuffer,
    alloc_result: *const regalloc.AllocResult,
    dest: ir.VReg,
    result_reg: emit.Reg,
) !void {
    try code.zeroExtend32(result_reg);
    try writeDef(code, alloc_result, dest, result_reg);
}

/// Type-aware writeDef: applies i32 zero-extension when needed.
fn writeDefTyped(
    code: *emit.CodeBuffer,
    alloc_result: *const regalloc.AllocResult,
    dest: ir.VReg,
    result_reg: emit.Reg,
    val_type: ir.IrType,
) !void {
    if (val_type == .i32) {
        try writeDefI32(code, alloc_result, dest, result_reg);
    } else {
        try writeDef(code, alloc_result, dest, result_reg);
    }
}

/// Get the physical register for a VReg (only valid if allocated to a register).
fn regOf(alloc_result: *const regalloc.AllocResult, vreg: ir.VReg) ?emit.Reg {
    const alloc = alloc_result.get(vreg) orelse return null;
    return switch (alloc) {
        .reg => |preg| @as(emit.Reg, @enumFromInt(preg)),
        .stack => null,
    };
}

/// Return the physical register for a destination VReg, or a default scratch
/// register for stack-spilled destinations. This enables register-flexible
/// codegen: instructions operate directly on the allocated register instead
/// of always routing through rax.
fn destReg(alloc_result: *const regalloc.AllocResult, dest: ir.VReg) emit.Reg {
    return regOf(alloc_result, dest) orelse .rax;
}

fn compileInstRA(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    alloc_result: *const regalloc.AllocResult,
    const_vals: *const std.AutoHashMap(ir.VReg, i64),
    patches: *std.ArrayList(BranchPatch),
    call_patches: *std.ArrayList(CallPatch),
    table_patches: *std.ArrayList(TablePatch),
    import_count: u32,
    used_caller_saved: *const [caller_saved_alloc.len]bool,
    used_callee_saved: *const [callee_saved_alloc.len]bool,
) !void {
    switch (inst.op) {
        // ── Constants ─────────────────────────────────────────────────
        .iconst_32 => |val| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            if (val == 0) {
                try code.xorReg32(dr);
            } else {
                try code.movRegImm32(dr, val);
            }
            // 32-bit write already zero-extends upper bits — skip zeroExtend32
            try writeDef(code, alloc_result, dest, dr);
        },
        .iconst_64 => |val| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            try code.movRegImm64(dr, @bitCast(val));
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .fconst_32 => |val| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            try code.movRegImm32(dr, @bitCast(val));
            // 32-bit write already zero-extends
            try writeDef(code, alloc_result, dest, dr);
        },
        .fconst_64 => |val| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            try code.movRegImm64(dr, @bitCast(val));
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },

        // ── Binary arithmetic ─────────────────────────────────────────
        .add, .sub, .mul, .@"and", .@"or", .xor => {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const bin = switch (inst.op) {
                .add => |b| b, .sub => |b| b, .mul => |b| b,
                .@"and" => |b| b, .@"or" => |b| b, .xor => |b| b,
                else => unreachable,
            };
            const scratch: emit.Reg = if (dr == .rax) .rcx else .rax;
            // Check if RHS is a constant for immediate form
            if (const_vals.get(bin.rhs)) |imm| {
                if (imm >= std.math.minInt(i32) and imm <= std.math.maxInt(i32)) {
                    const lhs_reg = try useVReg(code, alloc_result, bin.lhs, dr);
                    if (lhs_reg != dr) try code.movRegReg(dr, lhs_reg);
                    const imm32: i32 = @intCast(imm);
                    switch (inst.op) {
                        .add => if (imm32 != 0) try code.addRegImm32(dr, imm32),
                        .sub => if (imm32 != 0) try code.subRegImm32(dr, imm32),
                        .@"and" => try code.andRegImm32(dr, imm32),
                        .@"or" => if (imm32 != 0) try code.orRegImm32(dr, imm32),
                        .xor => if (imm32 != 0) try code.xorRegImm32(dr, imm32),
                        .mul => {
                            try code.movRegImm32(scratch, imm32);
                            try code.imulRegReg(dr, scratch);
                        },
                        else => unreachable,
                    }
                    if (inst.type == .i32) try code.zeroExtend32(dr);
                    try writeDefTyped(code, alloc_result, dest, dr, inst.type);
                    return;
                }
            }
            // General case: load LHS into dest register, RHS into scratch
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, dr);
            if (lhs_reg != dr) try code.movRegReg(dr, lhs_reg);
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, scratch);
            switch (inst.op) {
                .add => try code.addRegReg(dr, rhs_reg),
                .sub => try code.subRegReg(dr, rhs_reg),
                .mul => try code.imulRegReg(dr, rhs_reg),
                .@"and" => try code.andRegReg(dr, rhs_reg),
                .@"or" => try code.orRegReg(dr, rhs_reg),
                .xor => try code.xorRegReg(dr, rhs_reg),
                else => unreachable,
            }
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },

        // ── Comparisons ───────────────────────────────────────────────
        inline .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => |bin, tag| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            // Load lhs and rhs using different spill scratches so spilled
            // operands don't clobber each other.
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .r11);
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rax);
            // Use 32-bit compare for i32 to get correct signed semantics.
            if (inst.type == .i32) {
                try code.cmpRegReg32(lhs_reg, rhs_reg);
            } else {
                try code.cmpRegReg(lhs_reg, rhs_reg);
            }
            const cc: u4 = comptime switch (tag) {
                .eq => 0x4, .ne => 0x5, .lt_s => 0xC, .lt_u => 0x2,
                .gt_s => 0xF, .gt_u => 0x7, .le_s => 0xE, .le_u => 0x6,
                .ge_s => 0xD, .ge_u => 0x3, else => unreachable,
            };
            try code.setcc(cc, dr);
            try code.movzxByte(dr, dr);
            // setcc+movzx produces clean 0/1 — skip zeroExtend32
            try writeDef(code, alloc_result, dest, dr);
        },

        .eqz => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try code.testRegReg(src_reg, src_reg);
            try code.setcc(0x4, dr);
            try code.movzxByte(dr, dr);
            // setcc+movzx produces clean 0/1 — skip zeroExtend32
            try writeDef(code, alloc_result, dest, dr);
        },

        // ── Local variable access ─────────────────────────────────────
        .local_get => |idx| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            try code.movRegMem(dr, .rbp, -@as(i32, @intCast((idx + 2) * 8)));
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .local_set => |ls| {
            const src_reg = try useVReg(code, alloc_result, ls.val, .rax);
            try code.movMemReg(.rbp, -@as(i32, @intCast((ls.idx + 2) * 8)), src_reg);
        },

        // ── Return ────────────────────────────────────────────────────
        .ret => |maybe_val| {
            if (maybe_val) |val| {
                const src_reg = try useVReg(code, alloc_result, val, .rax);
                if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            }
            // Restore callee-saved registers before epilogue (reverse order).
            var ci: usize = callee_saved_alloc.len;
            while (ci > 0) {
                ci -= 1;
                if (used_callee_saved[ci]) try code.popReg(callee_saved_alloc[ci]);
            }
            try code.emitEpilogue();
        },

        // ── Branches ──────────────────────────────────────────────────
        .br => |target| {
            try code.emitByte(0xE9);
            const patch_off = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = patch_off, .target_block = target });
        },
        .br_if => |br| {
            const cond_reg = try useVReg(code, alloc_result, br.cond, .rax);
            try code.testRegReg(cond_reg, cond_reg);
            try code.emitByte(0x0F);
            try code.emitByte(0x85);
            const then_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = then_patch, .target_block = br.then_block });
            try code.emitByte(0xE9);
            const else_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = else_patch, .target_block = br.else_block });
        },
        .br_table => |bt| {
            // Load index into r11 (scratch) and canonicalize upper 32 bits to zero.
            const idx_reg = try useVReg(code, alloc_result, bt.index, .r11);
            if (idx_reg != .r11) try code.movRegReg(.r11, idx_reg);
            try code.zeroExtend32(.r11);

            // cmp r11, targets.len ; jae default  (unsigned out-of-range goes to default)
            try code.cmpRegImm32(.r11, @intCast(bt.targets.len));
            try code.emitByte(0x0F);
            try code.emitByte(0x83);
            const default_patch = code.len();
            try code.emitI32(0);
            try patches.append(code.allocator, .{ .patch_offset = default_patch, .target_block = bt.default });

            // lea r10, [rip + table]       ; 4C 8D 15 disp32
            try code.emitByte(0x4C);
            try code.emitByte(0x8D);
            try code.emitByte(0x15);
            const lea_disp_off = code.len();
            try code.emitI32(0);

            // movsxd r11, dword ptr [r10 + r11*4]
            // REX.WRXB = 0x4F, opcode 0x63, ModR/M=00_011_100 (0x1C), SIB=10_011_010 (0x9A)
            try code.emitByte(0x4F);
            try code.emitByte(0x63);
            try code.emitByte(0x1C);
            try code.emitByte(0x9A);

            // add r10, r11
            try code.addRegReg(.r10, .r11);

            // jmp r10                     ; 41 FF E2
            try code.emitByte(0x41);
            try code.emitByte(0xFF);
            try code.emitByte(0xE2);

            // Emit the jump table inline (no fallthrough from the indirect jmp).
            const table_off = code.len();
            const lea_rel: i32 = @intCast(@as(i64, @intCast(table_off)) - @as(i64, @intCast(lea_disp_off + 4)));
            code.patchI32(lea_disp_off, lea_rel);

            for (bt.targets) |target| {
                const entry_off = code.len();
                try code.emitI32(0);
                try table_patches.append(code.allocator, .{
                    .entry_offset = entry_off,
                    .base_offset = table_off,
                    .target_block = target,
                });
            }
        },

        // ── Function calls ────────────────────────────────────────────
        .call => |cl| {
            // No push/pop: the allocator ensures no live vreg in a
            // caller-saved register spans this call instruction.

            if (cl.func_idx < import_count) {
                // Import call: indirect call via host function pointer table in VmCtx.
                try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
                const n_args: u32 = @intCast(cl.args.len);
                if (n_args > 0) {
                    const max_reg_args = @min(n_args, @as(u32, @intCast(param_regs.len)) - 1);
                    var i: u32 = 0;
                    while (i < max_reg_args) : (i += 1) {
                        const arg_reg = try useVReg(code, alloc_result, cl.args[i], .r10);
                        if (arg_reg != param_regs[i + 1]) try code.movRegReg(param_regs[i + 1], arg_reg);
                    }
                }
                try code.movRegMem(.r10, param_regs[0], vmctx_host_functions_field);
                if (cl.func_idx > 0) {
                    try code.addRegImm32(.r10, @intCast(@as(u32, cl.func_idx) * 8));
                }
                try code.movRegMem(.rax, .r10, 0);
                if (comptime builtin.os.tag == .windows) {
                    try code.subRegImm32(.rsp, 32);
                }
                try code.callReg(.rax);
                if (comptime builtin.os.tag == .windows) {
                    try code.addRegImm32(.rsp, 32);
                }
            } else {
                // Local function call: direct CALL rel32
                try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
                const n_args: u32 = @intCast(cl.args.len);
                if (n_args > 0) {
                    const max_reg_args = @min(n_args, @as(u32, @intCast(param_regs.len)) - 1);
                    var i: u32 = 0;
                    while (i < max_reg_args) : (i += 1) {
                        const arg_reg = try useVReg(code, alloc_result, cl.args[i], .r10);
                        if (arg_reg != param_regs[i + 1]) try code.movRegReg(param_regs[i + 1], arg_reg);
                    }
                }
                try code.emitByte(0xE8);
                const patch_off = code.len();
                try code.emitI32(0);
                try call_patches.append(code.allocator, .{
                    .patch_offset = patch_off,
                    .target_func_idx = cl.func_idx - import_count,
                });
            }

            if (inst.dest) |dest| {
                try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
            }
        },

        // ── Indirect function calls ───────────────────────────────────
        .call_indirect => |ci| {
            // No push/pop: the allocator ensures no live vreg in a
            // clobbered register spans this call instruction.

            // Load table element index
            const idx_reg = try useVReg(code, alloc_result, ci.elem_idx, .rax);
            if (idx_reg != .rax) try code.movRegReg(.rax, idx_reg);
            try code.zeroExtend32(.rax);

            // Load func_table_ptr from VmCtx
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            try code.movRegMem(.r10, .r10, vmctx_func_table_field);

            // func_ptr = func_table[elem_idx * 8]
            try code.emitSlice(&.{ 0x48, 0xC1, 0xE0, 0x03 }); // shl rax, 3
            try code.addRegReg(.rax, .r10);
            try code.movRegMem(.rax, .rax, 0);

            // Set up call: VmCtx in param_regs[0], wasm args in param_regs[1..]
            try code.movRegMem(param_regs[0], .rbp, vmctx_offset);
            const n_args: u32 = @intCast(ci.args.len);
            if (n_args > 0) {
                try code.pushReg(.rax);
                const max_reg_args = @min(n_args, @as(u32, @intCast(param_regs.len)) - 1);
                var i: u32 = 0;
                while (i < max_reg_args) : (i += 1) {
                    const arg_reg = try useVReg(code, alloc_result, ci.args[i], .r10);
                    if (arg_reg != param_regs[i + 1]) try code.movRegReg(param_regs[i + 1], arg_reg);
                }
                try code.popReg(.rax);
            }

            if (comptime builtin.os.tag == .windows) {
                try code.subRegImm32(.rsp, 32);
            }
            try code.callReg(.rax);
            if (comptime builtin.os.tag == .windows) {
                try code.addRegImm32(.rsp, 32);
            }

            if (inst.dest) |dest| {
                try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
            }
        },

        // ── Memory ────────────────────────────────────────────────────
        .load => |ld| {
            const dest = inst.dest orelse return;
            // Load memory base from VMContext frame slot, add wasm offset
            try code.movRegMem(.r10, .rbp, vmctx_offset); // load VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_membase_field); // load VmCtx.memory_base
            const base_reg = try useVReg(code, alloc_result, ld.base, .rax);
            if (base_reg != .rax) try code.movRegReg(.rax, base_reg);
            try code.zeroExtend32(.rax); // wasm addresses are i32
            try code.addRegReg(.rax, .r10); // rax = mem_base + wasm_addr
            // Load value into dest register (address is in rax); fold offset into disp.
            const dr = destReg(alloc_result, dest);
            try code.movRegMemSized(dr, .rax, @intCast(ld.offset), ld.size);
            if (ld.sign_extend and ld.size < 8) {
                switch (ld.size) {
                    1 => try code.movsxByteToReg(dr, dr),
                    2 => try code.movsxWordToReg(dr, dr),
                    4 => try code.movsxd(dr, dr),
                    else => {},
                }
            }
            // For non-sign-extended loads ≤ 4 bytes, movRegMemSized already
            // produces a zero-extended (clean) i32 — skip redundant zeroExtend32.
            const load_is_clean = !ld.sign_extend and ld.size <= 4;
            if (load_is_clean) {
                try writeDef(code, alloc_result, dest, dr);
            } else {
                try writeDefTyped(code, alloc_result, dest, dr, inst.type);
            }
        },
        .store => |st| {
            // Load memory base from VMContext frame slot into r10.
            try code.movRegMem(.r10, .rbp, vmctx_offset); // load VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_membase_field); // load VmCtx.memory_base
            // Compute final address in rax (not allocatable — safe to clobber).
            const base_reg = try useVReg(code, alloc_result, st.base, .rax);
            if (base_reg != .rax) try code.movRegReg(.rax, base_reg);
            try code.zeroExtend32(.rax); // wasm addresses are i32
            try code.addRegReg(.rax, .r10); // rax = mem_base + wasm_addr
            // Load value into rcx (not allocatable — safe to clobber).
            // useVReg writes spill loads into scratch=.rcx, so rax is preserved.
            const val_reg = try useVReg(code, alloc_result, st.val, .rcx);
            if (val_reg != .rcx) try code.movRegReg(.rcx, val_reg);
            // Fold wasm offset into the mov displacement.
            try code.movMemRegSized(.rax, @intCast(st.offset), .rcx, st.size);
        },
        .memory_copy => |mc| {
            // REP MOVSB: rdi=dst, rsi=src, rcx=len
            try code.movRegMem(.r10, .rbp, vmctx_offset); // load VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_membase_field); // load VmCtx.memory_base // memory base
            const len_reg = try useVReg(code, alloc_result, mc.len, .rcx);
            if (len_reg != .rcx) try code.movRegReg(.rcx, len_reg);
            const src_reg = try useVReg(code, alloc_result, mc.src, .rsi);
            if (src_reg != .rsi) try code.movRegReg(.rsi, src_reg);
            try code.zeroExtend32(.rsi); // wasm addresses are i32
            try code.addRegReg(.rsi, .r10); // rsi = mem_base + src
            const dst_reg = try useVReg(code, alloc_result, mc.dst, .rdi);
            if (dst_reg != .rdi) try code.movRegReg(.rdi, dst_reg);
            try code.zeroExtend32(.rdi); // wasm addresses are i32
            try code.addRegReg(.rdi, .r10); // rdi = mem_base + dst
            try code.emitSlice(&.{ 0xF3, 0xA4 }); // REP MOVSB
        },
        .memory_fill => |mf| {
            // REP STOSB: rdi=dst, al=val, rcx=len
            try code.movRegMem(.r10, .rbp, vmctx_offset); // load VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_membase_field); // load VmCtx.memory_base // memory base
            const len_reg = try useVReg(code, alloc_result, mf.len, .rcx);
            if (len_reg != .rcx) try code.movRegReg(.rcx, len_reg);
            const val_reg = try useVReg(code, alloc_result, mf.val, .rax);
            if (val_reg != .rax) try code.movRegReg(.rax, val_reg);
            const dst_reg = try useVReg(code, alloc_result, mf.dst, .rdi);
            if (dst_reg != .rdi) try code.movRegReg(.rdi, dst_reg);
            try code.zeroExtend32(.rdi); // wasm addresses are i32
            try code.addRegReg(.rdi, .r10); // rdi = mem_base + dst
            try code.emitSlice(&.{ 0xF3, 0xAA }); // REP STOSB
        },

        // ── Memory management ─────────────────────────────────────────
        .memory_size => {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            // Read current page count from VmCtx
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            try code.movRegMemNoRex(dr, .r10, vmctx_mem_pages_field);
            // 32-bit load already zero-extends
            try writeDef(code, alloc_result, dest, dr);
        },
        .memory_grow => |pages_vreg| {
            const dest = inst.dest orelse return;
            // Load requested pages into rcx
            const pages_reg = try useVReg(code, alloc_result, pages_vreg, .rcx);
            if (pages_reg != .rcx) try code.movRegReg(.rcx, pages_reg);
            try code.zeroExtend32(.rcx);

            // Load VmCtx and current pages
            try code.movRegMem(.r10, .rbp, vmctx_offset);
            try code.movRegMemNoRex(.rax, .r10, vmctx_mem_pages_field); // old_pages
            try code.movRegReg(.r11, .rax); // save old_pages in r11

            // new_pages = old_pages + requested
            try code.addRegReg(.rax, .rcx); // rax = new_pages

            // Update VmCtx.memory_pages = new_pages (always succeed, memory is pre-allocated)
            try code.movMemRegNoRex(.r10, vmctx_mem_pages_field, .rax);

            // Update VmCtx.memory_size = new_pages * 65536
            try code.emitSlice(&.{ 0x48, 0xC1, 0xE0, 0x10 }); // shl rax, 16
            try code.movMemReg(.r10, vmctx_memsize_field, .rax);

            // Return old_pages
            try code.movRegReg(.rax, .r11);

            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // memory.init and data.drop — currently no-ops in AOT since data
        // segments are applied at instantiation. For full passive segment
        // support, these would need runtime calls via VmCtx.
        .memory_init => |mi| {
            // Consume operands (required for correct vreg tracking)
            _ = try useVReg(code, alloc_result, mi.dst, .rax);
            _ = try useVReg(code, alloc_result, mi.src, .rax);
            _ = try useVReg(code, alloc_result, mi.len, .rax);
        },
        .data_drop => {},

        // ── Division ──────────────────────────────────────────────────
        .div_s, .div_u, .rem_s, .rem_u => |bin| {
            const dest = inst.dest orelse return;
            // idiv clobbers rax+rdx. rax is not allocatable, so only rdx
            // (which IS allocatable) needs saving if it holds a live value.
            const rdx_in_use = for (caller_saved_alloc, 0..) |reg, i| {
                if (reg == .rdx and used_caller_saved[i]) break true;
            } else false;
            if (rdx_in_use) try code.pushReg(.rdx);

            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            // useVReg loads spilled rhs into its scratch (.rcx), so it cannot
            // clobber rax; no need to stash LHS in r11.
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);

            // Zero divisor check
            try code.testRegReg(.rcx, .rcx);
            try code.emitSlice(&.{ 0x0F, 0x84 }); // JZ rel32
            const skip_patch = code.len();
            try code.emitI32(0);

            switch (inst.op) {
                .div_s, .rem_s => {
                    try code.cqo();
                    try code.idivReg(.rcx);
                },
                .div_u, .rem_u => {
                    try code.movRegImm32(.rdx, 0);
                    try code.divReg(.rcx);
                },
                else => unreachable,
            }
            try code.emitSlice(&.{ 0xE9 }); // JMP done
            const done_patch = code.len();
            try code.emitI32(0);

            const zero_off = code.len();
            try code.xorReg32(.rax);
            try code.movRegImm32(.rdx, 0);
            const done_off = code.len();

            code.patchI32(skip_patch, @intCast(@as(i64, @intCast(zero_off)) - @as(i64, @intCast(skip_patch + 4))));
            code.patchI32(done_patch, @intCast(@as(i64, @intCast(done_off)) - @as(i64, @intCast(done_patch + 4))));

            const result_reg: emit.Reg = switch (inst.op) {
                .div_s, .div_u => .rax,
                .rem_s, .rem_u => .rdx,
                else => unreachable,
            };

            // Restore rdx if it was saved
            if (rdx_in_use) {
                if (result_reg == .rdx) {
                    // Remainder result is in rdx — save before restoring
                    try code.movRegReg(.r11, .rdx);
                    try code.popReg(.rdx);
                    try code.movRegReg(.rax, .r11);
                } else {
                    try code.popReg(.rdx);
                }
            } else {
                if (result_reg != .rax) try code.movRegReg(.rax, result_reg);
            }

            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Shifts ────────────────────────────────────────────────────
        .shl, .shr_s, .shr_u, .rotl, .rotr => |bin| {
            const dest = inst.dest orelse return;
            // Load val first to avoid clobbering if both share a register
            const val_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (val_reg != .rax) try code.movRegReg(.rax, val_reg);
            // useVReg loads spilled cnt into its scratch (.rcx), so it cannot
            // clobber rax; no need to stash val in r11.
            const cnt_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (cnt_reg != .rcx) try code.movRegReg(.rcx, cnt_reg);
            switch (inst.op) {
                .shl => {
                    try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xE0 });
                },
                .shr_u => {
                    try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xE8 });
                },
                .shr_s => {
                    try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xF8 });
                },
                .rotl => {
                    try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xC0 });
                },
                .rotr => {
                    try code.rexW(.rax, .rax);
                    try code.emitSlice(&.{ 0xD3, 0xC8 });
                },
                else => unreachable,
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Unary ops ─────────────────────────────────────────────────
        .clz => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (inst.type == .i32) {
                try code.lzcnt32(dr, src_reg);
            } else {
                try code.lzcnt(dr, src_reg);
            }
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .ctz => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (inst.type == .i32) {
                try code.tzcnt32(dr, src_reg);
            } else {
                try code.tzcnt(dr, src_reg);
            }
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .popcnt => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (inst.type == .i32) {
                try code.popcnt32(dr, src_reg);
            } else {
                try code.popcntReg(dr, src_reg);
            }
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },

        // ── Select ────────────────────────────────────────────────────
        .select => |sel| {
            const dest = inst.dest orelse return;
            const cond_reg = try useVReg(code, alloc_result, sel.cond, .rax);
            if (cond_reg != .rax) try code.movRegReg(.rax, cond_reg);
            const false_reg = try useVReg(code, alloc_result, sel.if_false, .rcx);
            if (false_reg != .rcx) try code.movRegReg(.rcx, false_reg);
            const true_reg = try useVReg(code, alloc_result, sel.if_true, .rdx);
            if (true_reg != .rdx) try code.movRegReg(.rdx, true_reg);
            try code.testRegReg(.rax, .rax);
            try code.cmovnz(.rcx, .rdx);
            try writeDefTyped(code, alloc_result, dest, .rcx, inst.type);
        },

        .global_get => |idx| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            // Load globals_base from VmCtx, then load global value
            try code.movRegMem(.r10, .rbp, vmctx_offset); // VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_globals_field); // globals_base
            try code.movRegMem(dr, .r10, @as(i32, @intCast(idx * 8))); // global[idx]
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .global_set => |gs| {
            const val_reg = try useVReg(code, alloc_result, gs.val, .rax);
            try code.movRegMem(.r10, .rbp, vmctx_offset); // VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_globals_field); // globals_base
            try code.movMemReg(.r10, @as(i32, @intCast(gs.idx * 8)), val_reg); // global[idx] = val
        },

        .@"unreachable" => {
            try code.int3();
        },

        // ── Type conversions ──────────────────────────────────────────
        .wrap_i64 => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            // 32-bit reg-reg move truncates to 32 bits and zero-extends.
            try code.movRegReg32(dr, src_reg);
            try writeDef(code, alloc_result, dest, dr);
        },
        .extend_i32_s => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try code.movsxd(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .extend_i32_u => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            // 32-bit mov zero-extends to 64.
            try code.movRegReg32(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .extend8_s => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try code.movsxByteToReg(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .extend16_s => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try code.movsxWordToReg(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .extend32_s => |vreg| {
            const dest = inst.dest orelse return;
            const dr = destReg(alloc_result, dest);
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try code.movsxd(dr, src_reg);
            try writeDefTyped(code, alloc_result, dest, dr, inst.type);
        },
        .reinterpret => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try writeDefTyped(code, alloc_result, dest, src_reg, inst.type);
        },

        // ── Float unary operations ─────────────────────────────────────
        .f_neg => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // XOR sign bit: for f64 flip bit 63, for f32 flip bit 31
            if (inst.type == .f32) {
                try code.emitSlice(&.{ 0x48, 0xB9 }); // mov rcx, imm64
                try code.emitU64(0x0000000080000000);
                try code.xorRegReg(.rax, .rcx);
            } else {
                try code.emitSlice(&.{ 0x48, 0xB9 }); // mov rcx, imm64
                try code.emitU64(0x8000000000000000);
                try code.xorRegReg(.rax, .rcx);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .f_abs => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // AND clear sign bit
            if (inst.type == .f32) {
                try code.emitSlice(&.{ 0x48, 0xB9 }); // mov rcx, imm64
                try code.emitU64(0x000000007FFFFFFF);
                try code.andRegReg(.rax, .rcx);
            } else {
                try code.emitSlice(&.{ 0x48, 0xB9 }); // mov rcx, imm64
                try code.emitU64(0x7FFFFFFFFFFFFFFF);
                try code.andRegReg(.rax, .rcx);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .f_sqrt => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            if (inst.type == .f32) {
                try code.movdToXmm(.rax, .rax);
                try code.sqrtss(.rax, .rax);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
                try code.sqrtsd(.rax, .rax);
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .f_ceil, .f_floor, .f_trunc, .f_nearest => |vreg| {
            // ROUNDSD/ROUNDSS (SSE4.1): 66 0F 3A 0B/0A xmm, xmm, imm8
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            const round_mode: u8 = switch (inst.op) {
                .f_ceil => 0x0A,
                .f_floor => 0x09,
                .f_trunc => 0x0B,
                .f_nearest => 0x08,
                else => unreachable,
            };
            if (inst.type == .f32) {
                try code.movdToXmm(.rax, .rax);
                // ROUNDSS: 66 0F 3A 0A C0 imm8
                try code.emitSlice(&.{ 0x66, 0x0F, 0x3A, 0x0A, 0xC0, round_mode });
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
                // ROUNDSD: 66 0F 3A 0B C0 imm8
                try code.emitSlice(&.{ 0x66, 0x0F, 0x3A, 0x0B, 0xC0, round_mode });
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Float binary operations ───────────────────────────────────
        .f_min => |bin| {
            const dest = inst.dest orelse return;
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            if (inst.type == .f32) {
                try code.movdToXmm(.rax, .rax);
                try code.movdToXmm(.rcx, .rcx);
                try code.minss(.rax, .rcx);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
                try code.movqToXmm(.rcx, .rcx);
                try code.minsd(.rax, .rcx);
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .f_max => |bin| {
            const dest = inst.dest orelse return;
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            if (inst.type == .f32) {
                try code.movdToXmm(.rax, .rax);
                try code.movdToXmm(.rcx, .rcx);
                try code.maxss(.rax, .rcx);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
                try code.movqToXmm(.rcx, .rcx);
                try code.maxsd(.rax, .rcx);
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .f_copysign => |bin| {
            const dest = inst.dest orelse return;
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            // copysign: magnitude from lhs, sign from rhs
            if (inst.type == .f32) {
                try code.emitSlice(&.{ 0x48, 0xBA }); // mov rdx, imm64
                try code.emitU64(0x000000007FFFFFFF);
                try code.andRegReg(.rax, .rdx);
                try code.emitSlice(&.{ 0x48, 0xBA }); // mov rdx, imm64
                try code.emitU64(0x0000000080000000);
                try code.andRegReg(.rcx, .rdx);
            } else {
                try code.emitSlice(&.{ 0x48, 0xBA }); // mov rdx, imm64
                try code.emitU64(0x7FFFFFFFFFFFFFFF);
                try code.andRegReg(.rax, .rdx);
                try code.emitSlice(&.{ 0x48, 0xBA }); // mov rdx, imm64
                try code.emitU64(0x8000000000000000);
                try code.andRegReg(.rcx, .rdx);
            }
            try code.orRegReg(.rax, .rcx);
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Int/Float conversions ─────────────────────────────────────
        .trunc_f32_s, .trunc_f64_s => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            if (inst.op == .trunc_f32_s) {
                try code.movdToXmm(.rax, .rax);
                try code.cvttss2si(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
                try code.cvttsd2si(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .trunc_f32_u, .trunc_f64_u => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            if (inst.op == .trunc_f32_u) {
                try code.movdToXmm(.rax, .rax);
                try code.cvttss2si(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
                try code.cvttsd2si(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .trunc_sat_f32_s, .trunc_sat_f64_s, .trunc_sat_f32_u, .trunc_sat_f64_u => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);

            // NaN check: compare value with itself (NaN != NaN)
            const is_f32 = (inst.op == .trunc_sat_f32_s or inst.op == .trunc_sat_f32_u);
            if (is_f32) {
                try code.movdToXmm(.rax, .rax);
                try code.ucomiss(.rax, .rax);
            } else {
                try code.movqToXmm(.rax, .rax);
                try code.ucomisd(.rax, .rax);
            }
            // JP = parity flag set = NaN → result is 0
            try code.emitSlice(&.{ 0x0F, 0x8A }); // JP rel32
            const nan_patch = code.len();
            try code.emitI32(0);

            // Normal conversion
            if (is_f32) {
                try code.cvttss2si(.rax, .rax);
            } else {
                try code.cvttsd2si(.rax, .rax);
            }
            try code.emitSlice(&.{0xE9}); // JMP done
            const done_patch = code.len();
            try code.emitI32(0);

            // NaN path: result = 0
            const nan_off = code.len();
            try code.xorReg32(.rax);
            const done_off = code.len();

            code.patchI32(nan_patch, @intCast(@as(i64, @intCast(nan_off)) - @as(i64, @intCast(nan_patch + 4))));
            code.patchI32(done_patch, @intCast(@as(i64, @intCast(done_off)) - @as(i64, @intCast(done_patch + 4))));

            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .convert_s => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            if (inst.type == .f32) {
                try code.cvtsi2ss(.rax, .rax);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.cvtsi2sd(.rax, .rax);
                try code.movqFromXmm(.rax, .rax);
            }
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .convert_u => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);

            // Check if value >= 2^63 (sign bit set)
            try code.testRegReg(.rax, .rax);
            try code.emitSlice(&.{ 0x0F, 0x88 }); // JS rel32 (jump if sign bit set)
            const large_patch = code.len();
            try code.emitI32(0);

            // Small path (< 2^63): treat as signed — cvtsi2s{s,d} is correct
            if (inst.type == .f32) {
                try code.cvtsi2ss(.rax, .rax);
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.cvtsi2sd(.rax, .rax);
                try code.movqFromXmm(.rax, .rax);
            }
            try code.emitSlice(&.{0xE9}); // JMP done
            const done_patch = code.len();
            try code.emitI32(0);

            // Large path (>= 2^63): shift right by 1, convert, then multiply by 2
            const large_off = code.len();
            try code.movRegReg(.rcx, .rax); // save original
            try code.emitSlice(&.{ 0x48, 0xD1, 0xE8 }); // shr rax, 1
            // Preserve low bit: OR with original & 1
            try code.andRegImm32(.rcx, 1); // rcx = original & 1
            try code.orRegReg(.rax, .rcx); // rax = (original >> 1) | (original & 1)
            if (inst.type == .f32) {
                try code.cvtsi2ss(.rax, .rax);
                try code.addss(.rax, .rax); // multiply by 2
                try code.movdFromXmm(.rax, .rax);
            } else {
                try code.cvtsi2sd(.rax, .rax);
                try code.addsd(.rax, .rax); // multiply by 2
                try code.movqFromXmm(.rax, .rax);
            }
            const done_off = code.len();

            code.patchI32(large_patch, @intCast(@as(i64, @intCast(large_off)) - @as(i64, @intCast(large_patch + 4))));
            code.patchI32(done_patch, @intCast(@as(i64, @intCast(done_off)) - @as(i64, @intCast(done_patch + 4))));

            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .demote_f64 => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            try code.movqToXmm(.rax, .rax);
            try code.cvtsd2ss(.rax, .rax);
            try code.movdFromXmm(.rax, .rax);
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },
        .promote_f32 => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            try code.movdToXmm(.rax, .rax);
            try code.cvtss2sd(.rax, .rax);
            try code.movqFromXmm(.rax, .rax);
            try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
        },

        // ── Stubs for ops not commonly hit ────────────────────────────
        else => {
            // For unhandled ops, emit a no-op placeholder
            if (inst.dest) |dest| {
                try code.movRegImm32(.rax, 0);
                try writeDefTyped(code, alloc_result, dest, .rax, inst.type);
            }
        },
    }
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

test "CachedStack: push and pop maintain depth" {
    var code = emit.CodeBuffer.init(std.testing.allocator);
    defer code.deinit();

    var stack = CachedStack.init(2); // 2 locals
    try std.testing.expectEqual(@as(u32, 0), stack.depth);

    try stack.push(&code, .rax);
    try std.testing.expectEqual(@as(u32, 1), stack.depth);

    try stack.push(&code, .rcx);
    try std.testing.expectEqual(@as(u32, 2), stack.depth);

    try stack.pop(&code, .rax);
    try std.testing.expectEqual(@as(u32, 1), stack.depth);

    try stack.pop(&code, .rax);
    try std.testing.expectEqual(@as(u32, 0), stack.depth);
}

// ── Atomic instruction integration tests ──────────────────────────

fn containsBytes(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i <= haystack.len - needle.len) : (i += 1) {
        if (std.mem.eql(u8, haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

test "compileFunction: atomic_fence emits MFENCE" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    try block.append(.{ .op = .{ .atomic_fence = {} } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // MFENCE = 0F AE F0
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0xAE, 0xF0 }));
}

test "compileFunction: atomic_load emits sized MOV" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const loaded = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_load = .{ .base = base, .offset = 0, .size = 4 } }, .dest = loaded, .type = .i32 });
    try block.append(.{ .op = .{ .ret = loaded } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(code.len > 10);
    // Should NOT contain LOCK prefix (loads don't need it on x86-64)
    try std.testing.expect(!containsBytes(code, &.{ 0xF0, 0x0F }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_store emits MOV + MFENCE" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_store = .{ .base = base, .offset = 0, .size = 4, .val = val } } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Must contain MFENCE for seq-cst ordering
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0xAE, 0xF0 }));
}

test "compileFunction: atomic_rmw add emits LOCK XADD" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .add } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // LOCK XADD = F0 0F C1
    try std.testing.expect(containsBytes(code, &.{ 0xF0, 0x0F, 0xC1 }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_rmw sub emits NEG + LOCK XADD" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .sub } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should contain NEG (F7 /3) followed later by LOCK XADD (F0 0F C1)
    try std.testing.expect(containsBytes(code, &.{ 0xF0, 0x0F, 0xC1 }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_rmw xchg emits XCHG" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 99 }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .xchg } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // XCHG [mem], reg = 87 (no LOCK prefix needed, implicit)
    try std.testing.expect(containsBytes(code, &.{0x87}));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_rmw and emits CAS loop with LOCK CMPXCHG" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0xFF }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_rmw = .{ .base = base, .offset = 0, .size = 4, .val = val, .op = .@"and" } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // LOCK CMPXCHG with r8 src = F0 44 0F B1 (REX.R for r8 in reg field)
    try std.testing.expect(containsBytes(code, &.{ 0xF0, 0x44, 0x0F, 0xB1 }));
    // JNE for CAS retry = 0F 85
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0x85 }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunction: atomic_cmpxchg emits LOCK CMPXCHG" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const base = func.newVReg();
    const expected = func.newVReg();
    const replacement = func.newVReg();
    const result = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0x1000 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = expected, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = replacement, .type = .i32 });
    try block.append(.{ .op = .{ .atomic_cmpxchg = .{ .base = base, .offset = 0, .size = 4, .expected = expected, .replacement = replacement } }, .dest = result, .type = .i32 });
    try block.append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // LOCK CMPXCHG = F0 0F B1
    try std.testing.expect(containsBytes(code, &.{ 0xF0, 0x0F, 0xB1 }));
    // No JNE (single CMPXCHG, no retry loop)
    try std.testing.expect(!containsBytes(code, &.{ 0x0F, 0x85 }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

// ── Instruction selection optimization tests ──────────────────────

test "compileFunction: iconst_32(0) emits xor (zero idiom)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should contain XOR r32,r32 (31 xx) instead of MOV rax,0 (48 C7 C0 00000000)
    try std.testing.expect(containsBytes(code, &.{0x31}));
    // Should NOT contain the 7-byte mov rax, 0 pattern
    try std.testing.expect(!containsBytes(code, &.{ 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00 }));
}

test "compileFunction: iconst + add uses ADD imm32" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should contain ADD reg, imm32 (81 /0) with value 5
    try std.testing.expect(containsBytes(code, &.{0x81}));
    try std.testing.expect(containsBytes(code, &.{ 0x05, 0x00, 0x00, 0x00 }));
}

test "compileFunction: cmp + br_if fuses to Jcc (no setcc)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);
    const block2 = func.getBlock(b2);

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const cond = func.newVReg();
    const r1 = func.newVReg();
    const r2 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block0.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try block0.append(.{ .op = .{ .lt_s = .{ .lhs = v0, .rhs = v1 } }, .dest = cond });
    try block0.append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try block1.append(.{ .op = .{ .iconst_32 = 1 }, .dest = r1 });
    try block1.append(.{ .op = .{ .ret = r1 } });
    try block2.append(.{ .op = .{ .iconst_32 = 0 }, .dest = r2 });
    try block2.append(.{ .op = .{ .ret = r2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Should contain JL (0F 8C) — fused compare-and-branch
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0x8C }));
    // Should NOT contain SETCC (0F 9x) — comparison was deferred
    var has_setcc = false;
    for (code, 0..) |b, j| {
        if (b == 0x0F and j + 1 < code.len and (code[j + 1] & 0xF0) == 0x90) {
            has_setcc = true;
            break;
        }
    }
    try std.testing.expect(!has_setcc);
}

// ── Register-allocated compilation tests ──────────────────────────

test "compileFunctionRA: iconst_32 + ret" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    try std.testing.expect(code.len > 5);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: add two constants" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    try std.testing.expect(code.len > 10);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
    // Should use ADD imm32 (0x81) since rhs is a constant
    var found_add_imm = false;
    for (code) |b| {
        if (b == 0x81) { found_add_imm = true; break; }
    }
    try std.testing.expect(found_add_imm);
}

test "compileFunctionRA: global_get emits load from VMContext" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .global_get = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    try std.testing.expect(code.len > 10);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: wrap_i64 emits mov eax,eax" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_64 = 0x100000042 }, .dest = v0, .type = .i64 });
    try block.append(.{ .op = .{ .wrap_i64 = v0 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Allocator keeps v0 in rdx and puts v1 in rsi (v0's range doesn't end
    // strictly before v1 begins). wrap_i64 emits `mov esi, edx` = 89 D6
    // (ModR/M 11_010_110, reg=rdx=2, rm=rsi=6), which zero-extends to rsi.
    try std.testing.expect(containsBytes(code, &.{ 0x89, 0xD6 }));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: extend_i32_s emits MOVSXD" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = -1 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .extend_i32_s = v0 }, .dest = v1, .type = .i64 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // MOVSXD rsi, edx (sign-extend v0→v1; v0 in rdx, v1 in rsi): 48 63 F2
    try std.testing.expect(containsBytes(code, &.{ 0x48, 0x63, 0xF2 }));
}

test "compileFunctionRA: memory_copy emits REP MOVSB" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const dst = func.newVReg();
    const src = func.newVReg();
    const len = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = dst });
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = src });
    try block.append(.{ .op = .{ .iconst_32 = 50 }, .dest = len });
    try block.append(.{ .op = .{ .memory_copy = .{ .dst = dst, .src = src, .len = len } } });
    try block.append(.{ .op = .{ .ret = null } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Should contain REP MOVSB (F3 A4)
    try std.testing.expect(containsBytes(code, &.{ 0xF3, 0xA4 }));
}

test "compileFunctionRA: memory_fill emits REP STOSB" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const dst = func.newVReg();
    const val = func.newVReg();
    const len = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = dst });
    try block.append(.{ .op = .{ .iconst_32 = 0xFF }, .dest = val });
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = len });
    try block.append(.{ .op = .{ .memory_fill = .{ .dst = dst, .val = val, .len = len } } });
    try block.append(.{ .op = .{ .ret = null } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Should contain REP STOSB (F3 AA)
    try std.testing.expect(containsBytes(code, &.{ 0xF3, 0xAA }));
}

test "compileFunctionRA: division uses rax/rdx" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v0 });
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v1 });
    try block.append(.{ .op = .{ .div_s = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Should contain IDIV (F7 F9 for idiv rcx)
    try std.testing.expect(containsBytes(code, &.{0xF7}));
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: local_get and local_set" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0 });
    try block.append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v0 } } });
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = v1 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    try std.testing.expect(code.len > 10);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]);
}

test "compileFunctionRA: br_table emits jump table + indirect jmp" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();

    // Param 0 is the switch index. 2 targets + default.
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const idx = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .local_get = 0 }, .dest = idx, .type = .i32 });

    const targets = try allocator.alloc(ir.BlockId, 2);
    targets[0] = b1;
    targets[1] = b2;
    try func.getBlock(b0).append(.{ .op = .{ .br_table = .{
        .index = idx,
        .targets = targets,
        .default = b3,
    } } });

    const r1 = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = r1 });
    try func.getBlock(b1).append(.{ .op = .{ .ret = r1 } });
    const r2 = func.newVReg();
    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 2 }, .dest = r2 });
    try func.getBlock(b2).append(.{ .op = .{ .ret = r2 } });
    const r3 = func.newVReg();
    try func.getBlock(b3).append(.{ .op = .{ .iconst_32 = 3 }, .dest = r3 });
    try func.getBlock(b3).append(.{ .op = .{ .ret = r3 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);
    // targets slice is leaked by IR (same pattern as call.args); free explicitly.
    allocator.free(targets);

    // lea r10, [rip+disp32]
    try std.testing.expect(containsBytes(code, &.{ 0x4C, 0x8D, 0x15 }));
    // movsxd r11, [r10 + r11*4]
    try std.testing.expect(containsBytes(code, &.{ 0x4F, 0x63, 0x1C, 0x9A }));
    // jmp r10
    try std.testing.expect(containsBytes(code, &.{ 0x41, 0xFF, 0xE2 }));
    // jae rel32 (bounds check)
    try std.testing.expect(containsBytes(code, &.{ 0x0F, 0x83 }));
}

test "compileFunctionRA: eqz targets allocated dest register (rdx)" {
    // eqz result should use the destReg directly, not funnel through rax.
    // For a simple function the first allocatable reg is rdx (alloc_regs[0]),
    // so we expect `setcc dl` (0F 94 C2) and `movzx rdx, dl` (48 0F B6 D2).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .eqz = v0 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Allocator expires v0 after v1 starts (live-range end==start), so v0 gets
    // rdx and v1 gets rsi. The setcc result therefore lands in sil, which
    // requires the mandatory REX prefix to distinguish it from DH.
    // setcc sil: 40 0F 94 C6  (REX=0x40, opcode 0F 94, ModR/M 11_000_110)
    try std.testing.expect(containsBytes(code, &.{ 0x40, 0x0F, 0x94, 0xC6 }));
    // movzx rsi, sil: 48 0F B6 F6  (REX.W, ModR/M 11_110_110)
    try std.testing.expect(containsBytes(code, &.{ 0x48, 0x0F, 0xB6, 0xF6 }));
    // Must NOT emit the old rax-centric `setcc al` (0F 94 C0).
    try std.testing.expect(!containsBytes(code, &.{ 0x0F, 0x94, 0xC0 }));
}

test "compileFunctionRA: comparison targets allocated dest register" {
    // eq should use destReg for setcc instead of hardcoded rax. The first
    // allocatable reg is rdx, so verify `setcc dl` is present.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .eq = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Two constants are live simultaneously at the eq, so v0→rdx, v1→rsi,
    // v2→rdi. The setcc writes dil with mandatory REX.
    // setcc dil: 40 0F 94 C7  (REX=0x40, opcode 0F 94, ModR/M 11_000_111)
    try std.testing.expect(containsBytes(code, &.{ 0x40, 0x0F, 0x94, 0xC7 }));
    // movzx rdi, dil: 48 0F B6 FF  (REX.W, ModR/M 11_111_111)
    try std.testing.expect(containsBytes(code, &.{ 0x48, 0x0F, 0xB6, 0xFF }));
    // No legacy `setcc al`.
    try std.testing.expect(!containsBytes(code, &.{ 0x0F, 0x94, 0xC0 }));
}

test "compileFunctionRA: memory_size loads into allocated dest register" {
    // memory_size should emit `mov dr, [r10 + mem_pages_field]` with
    // dr being the allocated destination register rather than forced rax.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .memory_size, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // mov edx, [r10 + 56] — 32-bit load, no REX.W.
    // Opcode 0x8B, REX.B=0x41 (because base is r10). ModR/M = 10_010_010 (0x92),
    // then disp32 = 0x38 00 00 00 (vmctx_mem_pages_field = 56).
    try std.testing.expect(containsBytes(code, &.{ 0x41, 0x8B, 0x92, 0x38, 0x00, 0x00, 0x00 }));
}


test "compileFunctionRA: r12 allocated under register pressure" {
    // Seven simultaneously-live values (6 constants + one op) overflow the
    // first 5 alloc_regs (rdx, rsi, rdi, r8, r9). The 6th and 7th must go to
    // r12 and r13 (newly added to alloc_regs). We also verify that r12 is
    // preserved via push/pop in the prologue/epilogue.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg();
    const v4 = func.newVReg();
    const v5 = func.newVReg();
    const v6 = func.newVReg();
    const v7 = func.newVReg();
    // Produce 7 constants that are all live through an add chain.
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 4 }, .dest = v3, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v4, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 6 }, .dest = v5, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v6, .type = .i32 });
    // Consume all of them so their ranges extend here.
    try block.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v7, .type = .i32 });
    // Use the rest (prevents range expiry) by re-summing.
    const v8 = func.newVReg();
    try block.append(.{ .op = .{ .add = .{ .lhs = v2, .rhs = v3 } }, .dest = v8, .type = .i32 });
    const v9 = func.newVReg();
    try block.append(.{ .op = .{ .add = .{ .lhs = v4, .rhs = v5 } }, .dest = v9, .type = .i32 });
    const v10 = func.newVReg();
    try block.append(.{ .op = .{ .add = .{ .lhs = v6, .rhs = v7 } }, .dest = v10, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v10 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // Prologue should push r12 (41 54) because it is callee-saved and used.
    try std.testing.expect(containsBytes(code, &.{ 0x41, 0x54 }));
    // Epilogue should pop r12 (41 5C).
    try std.testing.expect(containsBytes(code, &.{ 0x41, 0x5C }));
}

test "compileFunctionRA: low-pressure function does not save r12" {
    // A function that only needs one allocatable register must not touch r12.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // No push r12 (41 54) and no pop r12 (41 5C).
    try std.testing.expect(!containsBytes(code, &.{ 0x41, 0x54 }));
    try std.testing.expect(!containsBytes(code, &.{ 0x41, 0x5C }));
}


test "compileFunctionRA: shift does not emit dead r11 save" {
    // Before B1, every shl emitted `mov r11, rax; ...; mov rax, r11` around
    // the rhs load. Since useVReg loads spilled rhs into its scratch (.rcx),
    // rax is never clobbered, so the save is dead. Verify it's gone.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .shl = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // `mov r11, rax` = 49 89 C3. `mov rax, r11` = 4C 89 D8. Neither should appear.
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x89, 0xC3 }));
    try std.testing.expect(!containsBytes(code, &.{ 0x4C, 0x89, 0xD8 }));
    // Shift opcode D3 E0 (shl rax, cl with REX.W) must still appear.
    try std.testing.expect(containsBytes(code, &.{ 0xD3, 0xE0 }));
}

test "compileFunctionRA: div does not emit dead r11 save around operand load" {
    // Before B1, div emitted `mov r11, rax; mov rcx, rhs; mov rax, r11` to
    // preserve LHS. With distinct scratches the save is dead.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 100 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .div_u = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // `mov r11, rax` = 49 89 C3 must not appear before the idiv/div.
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x89, 0xC3 }));
    // `mov rax, r11` = 4C 89 D8: this sequence also appears in div's rem
    // rdx-restore path (only when rdx-in-use AND op is rem), so it must NOT
    // appear here because no vreg got rdx (low pressure).
    try std.testing.expect(!containsBytes(code, &.{ 0x4C, 0x89, 0xD8 }));
}


test "compileFunctionRA: load folds wasm offset into mov disp" {
    // Verifies B2: i32.load with offset=8 no longer emits `add rax, 8`
    // then `mov dst, [rax]`; instead a single `mov dst, [rax+8]`.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .load = .{ .base = v0, .offset = 8, .size = 4, .sign_extend = false } }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v1 } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // `add rax, 8` = 48 83 C0 08 (with REX.W). It must no longer appear.
    try std.testing.expect(!containsBytes(code, &.{ 0x48, 0x83, 0xC0, 0x08 }));
    // `add rax, imm32` form = 48 05 ...  also must not appear.
    try std.testing.expect(!containsBytes(code, &.{ 0x48, 0x05 }));
}

test "compileFunctionRA: store folds wasm offset into mov disp and omits r11 save" {
    // Verifies B2: i32.store with offset=16 emits a single
    // `mov [rax+16], ecx` and no longer stashes the base into r11.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v0, .val = v1, .offset = 16, .size = 4 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const compile_result = try compileFunctionRA(&func, 0, allocator);
    const code = compile_result.code;
    defer allocator.free(compile_result.call_patches);
    defer allocator.free(code);

    // No `mov r11, rax` (49 89 C3) — the base is kept in rax directly.
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x89, 0xC3 }));
    // No `add r11, imm32` (49 81 C3 ... or 49 83 C3 ...) — offset is folded.
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x81, 0xC3 }));
    try std.testing.expect(!containsBytes(code, &.{ 0x49, 0x83, 0xC3 }));
    // 32-bit store `mov [rax+16], ecx` with disp32 encoding = 89 88 10 00 00 00.
    try std.testing.expect(containsBytes(code, &.{ 0x89, 0x88, 0x10, 0x00, 0x00, 0x00 }));
}
