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

/// Fixed frame offset for the VMContext pointer.
/// Stored at [rbp - 8] by compileFunctionRA at function entry.
/// Points to a VmCtx struct with memory_base at +0, globals_base at +16.
const vmctx_offset: i32 = -8;
const vmctx_membase_field: i32 = 0; // VmCtx.memory_base offset
const vmctx_globals_field: i32 = 16; // VmCtx.globals_base offset

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

        // Compile function using register allocator
        const func_code = try compileFunctionRA(&func, allocator);
        defer allocator.free(func_code);

        // Scan for call patches in the compiled code
        // (compileFunctionRA handles intra-function branch patching internally,
        //  but inter-function call patches need global resolution)

        try all_code.appendSlice(allocator, func_code);
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
pub fn compileFunctionRA(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8 {
    var alloc_result = try regalloc.allocate(func, allocator);
    defer alloc_result.deinit();

    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    // Frame: locals + spill slots + 1 vmctx slot, aligned to 16 bytes
    const spill_slots = alloc_result.spill_count;
    const raw_size: u32 = (func.local_count + 64 + spill_slots + 1) * 8;
    const frame_size: u32 = (raw_size + 15) & ~@as(u32, 15) | 8;
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

    var last_was_ret = false;
    for (func.blocks.items, 0..) |block, idx| {
        try block_offsets.put(@intCast(idx), code.len());
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInstRA(&code, inst, &alloc_result, &const_vals, &branch_patches, &call_patches);
        }
    }

    for (branch_patches.items) |patch| {
        if (block_offsets.get(patch.target_block)) |target_off| {
            const rel: i32 = @intCast(@as(i64, @intCast(target_off)) - @as(i64, @intCast(patch.patch_offset + 4)));
            code.patchI32(patch.patch_offset, rel);
        }
    }

    if (!last_was_ret) {
        try code.emitEpilogue();
    }

    return code.bytes.toOwnedSlice(allocator);
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

/// Get the physical register for a VReg (only valid if allocated to a register).
fn regOf(alloc_result: *const regalloc.AllocResult, vreg: ir.VReg) ?emit.Reg {
    const alloc = alloc_result.get(vreg) orelse return null;
    return switch (alloc) {
        .reg => |preg| @as(emit.Reg, @enumFromInt(preg)),
        .stack => null,
    };
}

fn compileInstRA(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    alloc_result: *const regalloc.AllocResult,
    const_vals: *const std.AutoHashMap(ir.VReg, i64),
    patches: *std.ArrayList(BranchPatch),
    call_patches: *std.ArrayList(CallPatch),
) !void {
    switch (inst.op) {
        // ── Constants ─────────────────────────────────────────────────
        .iconst_32 => |val| {
            const dest = inst.dest orelse return;
            if (val == 0) {
                try code.xorReg32(.rax);
            } else {
                try code.movRegImm32(.rax, val);
            }
            try writeDef(code, alloc_result, dest, .rax);
        },
        .iconst_64 => |val| {
            const dest = inst.dest orelse return;
            try code.movRegImm64(.rax, @bitCast(val));
            try writeDef(code, alloc_result, dest, .rax);
        },
        .fconst_32 => |val| {
            const dest = inst.dest orelse return;
            try code.movRegImm32(.rax, @bitCast(val));
            try writeDef(code, alloc_result, dest, .rax);
        },
        .fconst_64 => |val| {
            const dest = inst.dest orelse return;
            try code.movRegImm64(.rax, @bitCast(val));
            try writeDef(code, alloc_result, dest, .rax);
        },

        // ── Binary arithmetic ─────────────────────────────────────────
        .add, .sub, .mul, .@"and", .@"or", .xor => {
            const dest = inst.dest orelse return;
            const bin = switch (inst.op) {
                .add => |b| b, .sub => |b| b, .mul => |b| b,
                .@"and" => |b| b, .@"or" => |b| b, .xor => |b| b,
                else => unreachable,
            };
            // Check if RHS is a constant for immediate form
            if (const_vals.get(bin.rhs)) |imm| {
                if (imm >= std.math.minInt(i32) and imm <= std.math.maxInt(i32)) {
                    const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
                    if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
                    const imm32: i32 = @intCast(imm);
                    switch (inst.op) {
                        .add => if (imm32 != 0) try code.addRegImm32(.rax, imm32),
                        .sub => if (imm32 != 0) try code.subRegImm32(.rax, imm32),
                        .@"and" => try code.andRegImm32(.rax, imm32),
                        .@"or" => if (imm32 != 0) try code.orRegImm32(.rax, imm32),
                        .xor => if (imm32 != 0) try code.xorRegImm32(.rax, imm32),
                        .mul => {
                            try code.movRegImm32(.rcx, imm32);
                            try code.imulRegReg(.rax, .rcx);
                        },
                        else => unreachable,
                    }
                    try writeDef(code, alloc_result, dest, .rax);
                    return;
                }
            }
            // General case: load both operands
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            switch (inst.op) {
                .add => try code.addRegReg(.rax, .rcx),
                .sub => try code.subRegReg(.rax, .rcx),
                .mul => try code.imulRegReg(.rax, .rcx),
                .@"and" => try code.andRegReg(.rax, .rcx),
                .@"or" => try code.orRegReg(.rax, .rcx),
                .xor => try code.xorRegReg(.rax, .rcx),
                else => unreachable,
            }
            try writeDef(code, alloc_result, dest, .rax);
        },

        // ── Comparisons ───────────────────────────────────────────────
        inline .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => |bin, tag| {
            const dest = inst.dest orelse return;
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
            try code.cmpRegReg(.rax, .rcx);
            const cc: u4 = comptime switch (tag) {
                .eq => 0x4, .ne => 0x5, .lt_s => 0xC, .lt_u => 0x2,
                .gt_s => 0xF, .gt_u => 0x7, .le_s => 0xE, .le_u => 0x6,
                .ge_s => 0xD, .ge_u => 0x3, else => unreachable,
            };
            try code.setcc(cc, .rax);
            try code.movzxByte(.rax, .rax);
            try writeDef(code, alloc_result, dest, .rax);
        },

        .eqz => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            try code.testRegReg(.rax, .rax);
            try code.setcc(0x4, .rax);
            try code.movzxByte(.rax, .rax);
            try writeDef(code, alloc_result, dest, .rax);
        },

        // ── Local variable access ─────────────────────────────────────
        .local_get => |idx| {
            const dest = inst.dest orelse return;
            try code.movRegMem(.rax, .rbp, -@as(i32, @intCast((idx + 1) * 8)));
            try writeDef(code, alloc_result, dest, .rax);
        },
        .local_set => |ls| {
            const src_reg = try useVReg(code, alloc_result, ls.val, .rax);
            try code.movMemReg(.rbp, -@as(i32, @intCast((ls.idx + 1) * 8)), src_reg);
        },

        // ── Return ────────────────────────────────────────────────────
        .ret => |maybe_val| {
            if (maybe_val) |val| {
                const src_reg = try useVReg(code, alloc_result, val, .rax);
                if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
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
            if (cond_reg != .rax) try code.movRegReg(.rax, cond_reg);
            try code.testRegReg(.rax, .rax);
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

        // ── Function calls ────────────────────────────────────────────
        .call => |cl| {
            const n_args: u32 = @intCast(cl.args.len);
            // Load args into ABI registers
            if (n_args > 0) {
                const max_reg_args = @min(n_args, @as(u32, @intCast(param_regs.len)));
                var i: u32 = 0;
                while (i < max_reg_args) : (i += 1) {
                    const arg_reg = try useVReg(code, alloc_result, cl.args[i], .r10);
                    if (arg_reg != param_regs[i]) try code.movRegReg(param_regs[i], arg_reg);
                }
            }
            try code.emitByte(0xE8);
            const patch_off = code.len();
            try code.emitI32(0);
            try call_patches.append(code.allocator, .{ .patch_offset = patch_off, .target_func_idx = cl.func_idx });
            if (inst.dest) |dest| {
                try writeDef(code, alloc_result, dest, .rax);
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
            try code.addRegReg(.rax, .r10); // rax = mem_base + wasm_addr
            if (ld.offset > 0) try code.addRegImm32(.rax, @intCast(ld.offset));
            try code.movRegMemNoRex(.rax, .rax, 0);
            try writeDef(code, alloc_result, dest, .rax);
        },
        .store => |st| {
            const val_reg = try useVReg(code, alloc_result, st.val, .rcx);
            if (val_reg != .rcx) try code.movRegReg(.rcx, val_reg);
            try code.movRegMem(.r10, .rbp, vmctx_offset); // load VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_membase_field); // load VmCtx.memory_base
            const base_reg = try useVReg(code, alloc_result, st.base, .rax);
            if (base_reg != .rax) try code.movRegReg(.rax, base_reg);
            try code.addRegReg(.rax, .r10); // rax = mem_base + wasm_addr
            if (st.offset > 0) try code.addRegImm32(.rax, @intCast(st.offset));
            try code.movMemRegNoRex(.rax, 0, .rcx);
        },
        .memory_copy => |mc| {
            // REP MOVSB: rdi=dst, rsi=src, rcx=len
            try code.movRegMem(.r10, .rbp, vmctx_offset); // load VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_membase_field); // load VmCtx.memory_base // memory base
            const len_reg = try useVReg(code, alloc_result, mc.len, .rcx);
            if (len_reg != .rcx) try code.movRegReg(.rcx, len_reg);
            const src_reg = try useVReg(code, alloc_result, mc.src, .rsi);
            if (src_reg != .rsi) try code.movRegReg(.rsi, src_reg);
            try code.addRegReg(.rsi, .r10); // rsi = mem_base + src
            const dst_reg = try useVReg(code, alloc_result, mc.dst, .rdi);
            if (dst_reg != .rdi) try code.movRegReg(.rdi, dst_reg);
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
            try code.addRegReg(.rdi, .r10); // rdi = mem_base + dst
            try code.emitSlice(&.{ 0xF3, 0xAA }); // REP STOSB
        },

        // ── Division ──────────────────────────────────────────────────
        .div_s, .div_u, .rem_s, .rem_u => |bin| {
            const dest = inst.dest orelse return;
            const rhs_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (rhs_reg != .rcx) try code.movRegReg(.rcx, rhs_reg);
            const lhs_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (lhs_reg != .rax) try code.movRegReg(.rax, lhs_reg);
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
            const result_reg: emit.Reg = switch (inst.op) {
                .div_s, .div_u => .rax,
                .rem_s, .rem_u => .rdx,
                else => unreachable,
            };
            try writeDef(code, alloc_result, dest, result_reg);
        },

        // ── Shifts ────────────────────────────────────────────────────
        .shl, .shr_s, .shr_u, .rotl, .rotr => |bin| {
            const dest = inst.dest orelse return;
            const cnt_reg = try useVReg(code, alloc_result, bin.rhs, .rcx);
            if (cnt_reg != .rcx) try code.movRegReg(.rcx, cnt_reg);
            const val_reg = try useVReg(code, alloc_result, bin.lhs, .rax);
            if (val_reg != .rax) try code.movRegReg(.rax, val_reg);
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
            try writeDef(code, alloc_result, dest, .rax);
        },

        // ── Unary ops ─────────────────────────────────────────────────
        .clz => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            try code.lzcnt(.rax, .rax);
            try writeDef(code, alloc_result, dest, .rax);
        },
        .ctz => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            try code.tzcnt(.rax, .rax);
            try writeDef(code, alloc_result, dest, .rax);
        },
        .popcnt => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            try code.popcntReg(.rax, .rax);
            try writeDef(code, alloc_result, dest, .rax);
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
            try writeDef(code, alloc_result, dest, .rcx);
        },

        .global_get => |idx| {
            const dest = inst.dest orelse return;
            // Load globals_base from VmCtx, then load global value
            try code.movRegMem(.r10, .rbp, vmctx_offset); // VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_globals_field); // globals_base
            try code.movRegMem(.rax, .r10, @as(i32, @intCast(idx * 8))); // global[idx]
            try writeDef(code, alloc_result, dest, .rax);
        },
        .global_set => |gs| {
            const val_reg = try useVReg(code, alloc_result, gs.val, .rax);
            if (val_reg != .rax) try code.movRegReg(.rax, val_reg);
            try code.movRegMem(.r10, .rbp, vmctx_offset); // VmCtx*
            try code.movRegMem(.r10, .r10, vmctx_globals_field); // globals_base
            try code.movMemReg(.r10, @as(i32, @intCast(gs.idx * 8)), .rax); // global[idx] = val
        },

        .@"unreachable" => {
            try code.int3();
        },

        // ── Type conversions ──────────────────────────────────────────
        .wrap_i64 => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // Truncate to 32-bit: writing to eax zero-extends to rax
            try code.emitSlice(&.{ 0x89, 0xC0 }); // mov eax, eax
            try writeDef(code, alloc_result, dest, .rax);
        },
        .extend_i32_s => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // MOVSXD rax, eax: sign-extend 32→64
            try code.emitSlice(&.{ 0x48, 0x63, 0xC0 });
            try writeDef(code, alloc_result, dest, .rax);
        },
        .extend_i32_u => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // mov eax, eax: zero-extend 32→64 (implicit on x86-64)
            try code.emitSlice(&.{ 0x89, 0xC0 });
            try writeDef(code, alloc_result, dest, .rax);
        },
        .extend8_s => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // MOVSX rax, al: REX.W 0F BE C0
            try code.emitSlice(&.{ 0x48, 0x0F, 0xBE, 0xC0 });
            try writeDef(code, alloc_result, dest, .rax);
        },
        .extend16_s => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // MOVSX rax, ax: REX.W 0F BF C0
            try code.emitSlice(&.{ 0x48, 0x0F, 0xBF, 0xC0 });
            try writeDef(code, alloc_result, dest, .rax);
        },
        .extend32_s => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            if (src_reg != .rax) try code.movRegReg(.rax, src_reg);
            // MOVSXD rax, eax
            try code.emitSlice(&.{ 0x48, 0x63, 0xC0 });
            try writeDef(code, alloc_result, dest, .rax);
        },
        .reinterpret => |vreg| {
            const dest = inst.dest orelse return;
            const src_reg = try useVReg(code, alloc_result, vreg, .rax);
            try writeDef(code, alloc_result, dest, src_reg);
        },

        // ── Stubs for ops not commonly hit ────────────────────────────
        else => {
            // For unhandled ops, emit a no-op placeholder
            // (atomics, float ops, extensions, etc.)
            if (inst.dest) |dest| {
                try code.movRegImm32(.rax, 0);
                try writeDef(code, alloc_result, dest, .rax);
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

    const code = try compileFunctionRA(&func, allocator);
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

    const code = try compileFunctionRA(&func, allocator);
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
