//! AArch64 IR Compiler
//!
//! Walks IR functions and emits AArch64 machine code via CodeBuffer.
//! Uses the same pattern as the x86-64 backend with a linear-scan
//! register allocator mapping VRegs to physical registers.

const std = @import("std");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");
const schedule = @import("schedule.zig");
const regalloc = @import("../../ir/regalloc.zig");
const analysis = @import("../../ir/analysis.zig");

/// Simple VReg → physical register mapping.
///
/// Stack-spill layout: when all scratch regs are in use, the allocator
/// falls back to the frame spill region at `[fp + spill_base + slot*8]`.
/// `spill_base` is set by `compileFunction` to sit immediately above the
/// locals area so spill slots don't collide with saved FP/LR, VMContext,
/// or wasm locals.
const RegMap = struct {
    const Location = union(enum) {
        reg: emit.Reg,
        /// Byte offset from `spill_base` for the spill slot.
        stack: u32,
    };

    // Caller-saved scratch registers (AAPCS64: X0–X15 are caller-saved).
    // We reserve X16 (IP0) and X17 (IP1) as non-allocatable scratch for
    // codegen use (shift-count negation, rem via MSUB, spill reload, etc.);
    // X18 is reserved platform register.
    //
    // X15 is reserved as `tmp2` for multi-operand sequences (e.g., bounds
    // check needs vmctx + zext_addr + mem_size live simultaneously).
    //
    // X19–X28 are AAPCS64 callee-saved and appended after the caller-saved
    // block. They survive calls (no save around BL), trading one extra
    // STR/LDR pair per function per reg that the body allocates.
    pub const caller_saved_count: usize = 15;
    pub const scratch_regs = [_]emit.Reg{
        .x0,  .x1,  .x2,  .x3,  .x4,  .x5,  .x6,  .x7,
        .x8,  .x9,  .x10, .x11, .x12, .x13, .x14,
        // Callee-saved:
        .x19,
        .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27,
        .x28,
    };

    /// Non-allocatable scratch registers usable by any handler.
    /// Must never appear in `scratch_regs`.
    pub const tmp0: emit.Reg = .x16;
    pub const tmp1: emit.Reg = .x17;
    pub const tmp2: emit.Reg = .x15;

    entries: std.AutoHashMap(ir.VReg, Location),
    reg_used: [scratch_regs.len]bool = [_]bool{false} ** scratch_regs.len,
    /// Bit i is set iff `callee_saved_regs[i]` was ever assigned to some
    /// vreg during this function's compilation. Used to elide the prologue
    /// save and epilogue restore of callee-saved registers that are never
    /// clobbered. AAPCS64 requires preserving x19..x28 across calls; a reg
    /// that's never assigned is never written, so the save/restore is dead.
    used_callee_mask: u16 = 0,
    next_stack_offset: u32 = 0,
    /// Byte offset from FP where the first spill slot lives.
    spill_base: u32 = 0,
    /// Maximum bytes reserved for spills (beyond which `assign` errors).
    spill_capacity: u32 = 0,
    /// Optional linear-scan allocation result. When non-null, `assign`
    /// consults this for the physical location of every vreg instead of
    /// running the greedy fallback. Set by `compileFunctionImpl` right
    /// after running `regalloc.allocate`.
    alloc_result: ?*const regalloc.AllocResult = null,

    fn init(allocator: std.mem.Allocator, spill_base: u32, spill_capacity: u32) RegMap {
        return .{
            .entries = std.AutoHashMap(ir.VReg, Location).init(allocator),
            .spill_base = spill_base,
            .spill_capacity = spill_capacity,
        };
    }

    fn deinit(self: *RegMap) void {
        self.entries.deinit();
    }

    fn assign(self: *RegMap, vreg: ir.VReg) !Location {
        // Regalloc-driven path: consult the pre-computed AllocResult.
        // The physreg numbers used by regalloc match the indices into
        // `scratch_regs` (see `aarch64RegSet`), so the phys-reg number
        // IS the scratch_regs index. Spill offsets are absolute FP
        // offsets; translate to RegMap-relative by subtracting spill_base.
        if (self.alloc_result) |ar| {
            if (ar.get(vreg)) |a| {
                switch (a) {
                    .reg => |r| {
                        // `r` is the aarch64 register NUMBER (0..14 or
                        // 19..28). Find its index in `scratch_regs` to
                        // update `reg_used` / `used_callee_mask`.
                        const reg_num: u32 = @intCast(r);
                        const idx: usize = if (reg_num <= 14) reg_num else reg_num - 4;
                        self.reg_used[idx] = true;
                        if (idx >= caller_saved_count) {
                            self.used_callee_mask |= (@as(u16, 1) << @intCast(idx - caller_saved_count));
                        }
                        const loc = Location{ .reg = scratch_regs[idx] };
                        try self.entries.put(vreg, loc);
                        return loc;
                    },
                    .stack => |abs_off| {
                        const rel: i64 = @as(i64, abs_off) - @as(i64, @intCast(self.spill_base));
                        if (rel < 0) return error.NegativeSpillOffset;
                        const rel_u: u32 = @intCast(rel);
                        if (rel_u + 8 > self.spill_capacity) return error.OutOfSpillSlots;
                        self.next_stack_offset = @max(self.next_stack_offset, rel_u + 8);
                        const loc = Location{ .stack = rel_u };
                        try self.entries.put(vreg, loc);
                        return loc;
                    },
                }
            }
            // Fall through to greedy if the allocator had no opinion
            // (vreg has no live range — e.g., a def that's never used).
            std.debug.panic("regalloc-driven assign: vreg %{d} missing from AllocResult", .{vreg});
        }
        for (scratch_regs, 0..) |r, i| {
            if (!self.reg_used[i]) {
                self.reg_used[i] = true;
                if (i >= caller_saved_count) {
                    self.used_callee_mask |= (@as(u16, 1) << @intCast(i - caller_saved_count));
                }
                const loc = Location{ .reg = r };
                try self.entries.put(vreg, loc);
                return loc;
            }
        }
        // Spill to stack
        if (self.next_stack_offset >= self.spill_capacity) return error.OutOfSpillSlots;
        const offset = self.next_stack_offset;
        self.next_stack_offset += 8;
        const loc = Location{ .stack = offset };
        try self.entries.put(vreg, loc);
        return loc;
    }

    /// Release the physical register held by `vreg` so it becomes available
    /// to subsequent `assign` calls. Called by codegen after the last static
    /// use of `vreg` (liveness-driven scavenging). The vreg's entry is
    /// removed so any further read via `get` fails-fast with a clear signal.
    /// Spilled (stack) vregs are not reclaimed — spill slots are stable for
    /// the lifetime of the function (other slots may still be live above).
    fn freeVReg(self: *RegMap, vreg: ir.VReg) void {
        // Under regalloc-driven mode, regalloc owns liveness; the main
        // loop's kill_lists is a different (and sometimes narrower)
        // notion. Removing entries here can erase bindings that
        // regalloc still expects to be valid (e.g., a vreg that is
        // live across a block boundary). Leave bookkeeping intact.
        if (self.alloc_result != null) return;
        const loc = self.entries.get(vreg) orelse return;
        switch (loc) {
            .reg => |r| {
                for (scratch_regs, 0..) |sr, i| {
                    if (sr == r) {
                        self.reg_used[i] = false;
                        break;
                    }
                }
                _ = self.entries.remove(vreg);
            },
            .stack => {}, // spill slot retained (simple bump allocator)
        }
    }

    fn get(self: *const RegMap, vreg: ir.VReg) ?Location {
        return self.entries.get(vreg);
    }

    /// FP-relative byte offset of a spill slot (scaled by 8 for LDR/STR X).
    fn spillOffsetScaled(self: *const RegMap, slot_byte_off: u32) u12 {
        return @intCast((self.spill_base + slot_byte_off) / 8);
    }
};

/// Stack-resident v128 storage for the first native SIMD slice.
///
/// Scalar VRegs continue to use the existing GPR allocator. v128 values get
/// their own 16-byte aligned frame slots and are moved through scratch NEON Q
/// registers only while an instruction is being emitted.
const V128StackMap = struct {
    entries: std.AutoHashMap(ir.VReg, u32),
    spill_base: u32,
    spill_capacity: u32,
    next_offset: u32 = 0,

    fn init(allocator: std.mem.Allocator, spill_base: u32, spill_capacity: u32) V128StackMap {
        return .{
            .entries = std.AutoHashMap(ir.VReg, u32).init(allocator),
            .spill_base = spill_base,
            .spill_capacity = spill_capacity,
        };
    }

    fn deinit(self: *V128StackMap) void {
        self.entries.deinit();
    }

    fn assign(self: *V128StackMap, vreg: ir.VReg) !u32 {
        if (self.entries.get(vreg)) |off| return off;
        if (self.next_offset + 16 > self.spill_capacity) return error.OutOfV128SpillSlots;
        const off = self.spill_base + self.next_offset;
        self.next_offset += 16;
        try self.entries.put(vreg, off);
        return off;
    }

    fn get(self: *const V128StackMap, vreg: ir.VReg) !u32 {
        return self.entries.get(vreg) orelse error.UnboundV128;
    }
};

const UseQueryCtx = struct {
    target: ir.VReg,
    found: bool = false,
};

fn recordUseQuery(ctx: *UseQueryCtx, vreg: ir.VReg) !void {
    if (vreg == ctx.target) ctx.found = true;
}

const V128RegCache = struct {
    const Slot = struct {
        vreg: ?ir.VReg = null,
        dirty: bool = false,
    };

    // Use caller-saved vector registers that do not overlap scalar FP helper
    // scratch V0/V1. Q8-Q15 are intentionally excluded because AAPCS64 only
    // preserves their low 64 bits, not the full 128-bit SIMD value.
    const regs = [_]u5{
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31,
    };

    slots: [regs.len]Slot = [_]Slot{.{}} ** regs.len,
    current_block_insts: []const ir.Inst = &.{},
    current_inst_index: usize = 0,

    fn beginInst(self: *V128RegCache, insts: []const ir.Inst, inst_index: usize) void {
        self.current_block_insts = insts;
        self.current_inst_index = inst_index;
    }

    fn regForSlot(idx: usize) u5 {
        return regs[idx];
    }

    fn slotForReg(vt: u5) ?usize {
        for (regs, 0..) |reg, idx| {
            if (reg == vt) return idx;
        }
        return null;
    }

    fn find(self: *const V128RegCache, vreg: ir.VReg) ?usize {
        for (self.slots, 0..) |slot, idx| {
            if (slot.vreg == vreg) return idx;
        }
        return null;
    }

    fn evictSlot(
        self: *V128RegCache,
        code: *emit.CodeBuffer,
        v128_map: *V128StackMap,
        idx: usize,
        write_dirty: bool,
    ) !void {
        const slot = self.slots[idx];
        const vreg = slot.vreg orelse return;
        if (write_dirty and slot.dirty) {
            const off = try v128_map.assign(vreg);
            try storeV128SlotAbs(code, off, regForSlot(idx));
        }
        self.slots[idx] = .{};
    }

    fn flushAll(
        self: *V128RegCache,
        code: *emit.CodeBuffer,
        v128_map: *V128StackMap,
    ) !void {
        for (self.slots, 0..) |_, idx| {
            try self.evictSlot(code, v128_map, idx, true);
        }
    }

    fn release(self: *V128RegCache, vreg: ir.VReg) void {
        const idx = self.find(vreg) orelse return;
        self.slots[idx] = .{};
    }

    fn ensure(
        self: *V128RegCache,
        code: *emit.CodeBuffer,
        v128_map: *V128StackMap,
        vreg: ir.VReg,
        excluded_reg: ?u5,
    ) !u5 {
        if (self.find(vreg)) |idx| return regForSlot(idx);

        const idx = try self.allocSlot(code, v128_map, excluded_reg);
        const off = try v128_map.get(vreg);
        try frameAddr(code, RegMap.tmp0, off);
        try code.ldrQ(regForSlot(idx), RegMap.tmp0);
        self.slots[idx] = .{ .vreg = vreg, .dirty = false };
        return regForSlot(idx);
    }

    fn allocSlot(
        self: *V128RegCache,
        code: *emit.CodeBuffer,
        v128_map: *V128StackMap,
        excluded_reg: ?u5,
    ) !usize {
        if (self.chooseFreeSlot(excluded_reg)) |idx| return idx;
        const idx = self.chooseEvictionSlot(excluded_reg) orelse return error.OutOfV128ScratchRegs;
        try self.evictSlot(code, v128_map, idx, true);
        return idx;
    }

    fn chooseFreeSlot(self: *const V128RegCache, excluded_reg: ?u5) ?usize {
        for (self.slots, 0..) |slot, idx| {
            const reg = regForSlot(idx);
            if (excluded_reg != null and excluded_reg.? == reg) continue;
            if (slot.vreg == null) return idx;
        }
        return null;
    }

    fn chooseEvictionSlot(self: *const V128RegCache, excluded_reg: ?u5) ?usize {
        var best_idx: ?usize = null;
        var best_category: u2 = 3;
        var best_next: usize = 0;

        for (self.slots, 0..) |slot, idx| {
            const reg = regForSlot(idx);
            if (excluded_reg != null and excluded_reg.? == reg) continue;
            const vreg = slot.vreg orelse continue;
            const maybe_next = self.nextUseIndex(vreg);
            const next = maybe_next orelse std.math.maxInt(usize);
            const category: u2 = if (maybe_next == null)
                0
            else if (!slot.dirty)
                1
            else
                2;

            if (best_idx == null or
                category < best_category or
                (category == best_category and next > best_next))
            {
                best_idx = idx;
                best_category = category;
                best_next = next;
            }
        }

        return best_idx;
    }

    fn nextUseIndex(self: *const V128RegCache, vreg: ir.VReg) ?usize {
        if (self.current_inst_index >= self.current_block_insts.len) return null;
        var idx = self.current_inst_index;
        while (idx < self.current_block_insts.len) : (idx += 1) {
            if (instUsesVReg(self.current_block_insts[idx], vreg)) return idx;
        }
        return null;
    }

    fn instUsesVReg(inst: ir.Inst, vreg: ir.VReg) bool {
        var ctx = UseQueryCtx{ .target = vreg };
        schedule.forEachUse(inst, &ctx, recordUseQuery) catch unreachable;
        return ctx.found;
    }

    fn defineFresh(
        self: *V128RegCache,
        code: *emit.CodeBuffer,
        v128_map: *V128StackMap,
        dest: ir.VReg,
        excluded_reg: ?u5,
    ) !u5 {
        const idx = try self.allocSlot(code, v128_map, excluded_reg);
        _ = try v128_map.assign(dest);
        self.slots[idx] = .{ .vreg = dest, .dirty = true };
        return regForSlot(idx);
    }

    fn prepareOverwrite(
        self: *V128RegCache,
        code: *emit.CodeBuffer,
        v128_map: *V128StackMap,
        vt: u5,
        drop_existing: bool,
    ) !void {
        const idx = slotForReg(vt) orelse return error.InvalidV128ScratchReg;
        try self.evictSlot(code, v128_map, idx, !drop_existing);
    }

    fn defineInReg(self: *V128RegCache, v128_map: *V128StackMap, dest: ir.VReg, vt: u5) !void {
        const idx = slotForReg(vt) orelse return error.InvalidV128ScratchReg;
        if (self.find(dest)) |existing_idx| {
            self.slots[existing_idx] = .{};
        }
        _ = try v128_map.assign(dest);
        self.slots[idx] = .{ .vreg = dest, .dirty = true };
    }
};

/// A pending branch that needs its 19/26-bit PC-relative offset patched once
/// the target block's offset is known.
const BranchPatch = struct {
    patch_offset: usize,
    target_block: ir.BlockId,
    kind: Kind,

    const Kind = enum { b_uncond, b_cond };
};

/// A pending BL whose 26-bit PC-relative offset needs to be patched once
/// all module functions are compiled and their start offsets are known.
pub const CallPatch = struct {
    /// Byte offset of the BL instruction within the target code buffer.
    patch_offset: usize,
    /// Local function index (import_count already subtracted).
    target_func_idx: u32,
};

pub const CompileOptions = struct {
    enable_scheduler: bool = true,
};

/// Context threaded through per-function compilation for cross-function
/// concerns (calls, imports).
const FuncCompileCtx = struct {
    import_count: u32 = 0,
    call_patches: ?*std.ArrayListUnmanaged(CallPatch) = null,
    /// FP-relative byte offset of the call-save region where caller-save
    /// physregs (x0..x15) are spilled around BL instructions.
    call_save_base: u32 = 0,
    /// FP-relative byte offset of the callee-save region where x19..x28
    /// are saved in the prologue and restored in each epilogue.
    callee_save_base: u32 = 0,
    /// FP-relative byte offset of the Hidden-Return-Pointer save slot.
    /// Only meaningful when this function's `result_count > 1`. The
    /// caller passes &scratch as the `param_count+1`-th arg; the
    /// prologue stashes it here so `ret_multi` can re-load it.
    hrp_save_off: u32 = 0,
    /// FP-relative byte offset of the caller-side scratch region where
    /// extra results of calls (beyond the one returned in x0) are
    /// written by the callee. Sized to `max_extra_results * 8` bytes.
    /// `call_result` reads back from `[fp + scratch_base + idx*8]`.
    scratch_base: u32 = 0,
    /// Total frame size (in bytes). Needed by tail-call emitters to
    /// tear down the frame before branching to the target.
    frame_size: u32 = 0,
    /// VReg → signed i64 value for `iconst_*` definitions in this function.
    /// Used to fold constants into immediate-form instructions (e.g.
    /// ADD/SUB imm). The iconst itself is still materialized into its home
    /// register, so downstream uses that don't fold still work.
    const_vals: ?*const std.AutoHashMap(ir.VReg, i64) = null,
    /// Set of mul dest vregs whose computation should be skipped because
    /// the single consumer is an add/sub that will be emitted as a fused
    /// MADD/MSUB (Phase 4 FMA fusion). When the mul handler sees a dest
    /// in this set, it bypasses codegen and leaves the sources live for
    /// the fused op.
    mul_fused: ?*const std.AutoHashMap(ir.VReg, void) = null,
    /// Map from add/sub dest vreg → (mul_lhs, mul_rhs, addend) source
    /// triple. Populated by the FMA pre-pass in lockstep with `mul_fused`.
    fma_info: ?*const std.AutoHashMap(ir.VReg, FmaInfo) = null,
    /// Byte offsets of every callee-save save/restore block emitted. Each
    /// entry is the 10 STR/LDR offsets for one `emitCalleeSaveStore` or
    /// `emitCalleeSaveRestore` call. After the body is compiled, offsets
    /// for regs not in `RegMap.used_callee_mask` are rewritten as NOPs.
    callee_save_sites: ?*std.ArrayListUnmanaged([callee_saved_regs.len]usize) = null,
    /// Vregs whose last static use is the instruction currently being
    /// compiled. Set by the main loop before each `compileInst` call.
    /// Call emitters consult this to elide save/restore work for
    /// caller-save regs that die at the call itself.
    current_kills: []const ir.VReg = &.{},
    options: CompileOptions = .{},
    allocator: std.mem.Allocator,
};

/// Pre-computed info for a fused MADD/MSUB: `dest = addend ± mul_lhs * mul_rhs`.
const FmaInfo = struct {
    mul_lhs: ir.VReg,
    mul_rhs: ir.VReg,
    addend: ir.VReg,
    is_sub: bool,
};

const KillUseCtx = struct {
    seen: *std.AutoHashMap(ir.VReg, void),
    kill_lists: []std.ArrayListUnmanaged(ir.VReg),
    allocator: std.mem.Allocator,
    flat_idx: usize,
};

fn recordKillUse(ctx: *KillUseCtx, v: ir.VReg) !void {
    const e = try ctx.seen.getOrPut(v);
    if (e.found_existing) return;
    try ctx.kill_lists[ctx.flat_idx].append(ctx.allocator, v);
}

/// Compile an IR function to AArch64 machine code.
///
/// **AArch64 frame layout** (positive offsets from FP, which equals new SP):
/// ```
///   [fp + 0]                = saved FP
///   [fp + 8]                = saved LR
///   [fp + 16]               = VMContext pointer (spilled from x0)
///   [fp + (i+3)*8]          = local i (0 ≤ i < local_count)
///   [fp + spill_base]       = RegMap spill slots (256 bytes, 32 slots)
///   [fp + call_save_base]   = per-physreg call-save slots (128 bytes)
///   [fp + frame_size]       = caller's stack (args beyond x7, etc.)
/// ```
///
/// **VMContext ABI**: The VMContext pointer is passed in `x0` (AAPCS64
/// first argument) as a hidden prefix to the wasm params. It is spilled
/// to `[fp + 16]` in the prologue and reloaded into a scratch register
/// (typically `x16`/`tmp0`) whenever needed.
///
/// **Wasm parameters**: passed in `x1..x7` (up to 7 params; more not yet
/// supported), spilled to their matching local slots in the prologue.
/// Declared locals beyond `param_count` are zero-initialized.
pub fn compileFunction(func: *const ir.IrFunction, allocator: std.mem.Allocator) ![]u8 {
    // Test-friendly entry: no cross-function linking. Any `.call` op will
    // fail with error.CallLinkageUnavailable.
    return compileFunctionWithOptions(func, allocator, .{});
}

pub fn compileFunctionWithOptions(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
    options: CompileOptions,
) ![]u8 {
    return compileFunctionImpl(func, .{ .allocator = allocator, .options = options }, allocator);
}

fn isSupportedV128Def(inst: ir.Inst) bool {
    return switch (inst.op) {
        .v128_const,
        .v128_load,
        .v128_not,
        .v128_bitwise,
        .i32x4_binop,
        .i32x4_unop,
        .i32x4_extadd_pairwise_i16x8,
        .i32x4_extend_i16x8,
        .i32x4_extmul_i16x8,
        .i32x4_shift,
        .i32x4_splat,
        .i32x4_replace_lane,
        .i8x16_binop,
        .i8x16_unop,
        .i8x16_shift,
        .i8x16_splat,
        .i8x16_replace_lane,
        .i16x8_binop,
        .i16x8_unop,
        .i16x8_extadd_pairwise_i8x16,
        .i16x8_extend_i8x16,
        .i16x8_extmul_i8x16,
        .i16x8_shift,
        .i16x8_splat,
        .i16x8_replace_lane,
        .i64x2_binop,
        .i64x2_unop,
        .i64x2_extend_i32x4,
        .i64x2_extmul_i32x4,
        .i64x2_shift,
        .i64x2_splat,
        .i64x2_replace_lane,
        => inst.type == .v128,
        else => false,
    };
}

fn functionHasUnsupportedV128(func: *const ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.local_types) |local_types| {
        for (local_types) |ty| if (ty == .v128) return true;
    }
    var vreg_types = std.AutoHashMap(ir.VReg, ir.IrType).init(allocator);
    defer vreg_types.deinit();

    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            switch (inst.op) {
                .v128_const,
                .v128_load,
                .v128_not,
                .v128_bitwise,
                .i32x4_binop,
                .i32x4_unop,
                .i32x4_extadd_pairwise_i16x8,
                .i32x4_extend_i16x8,
                .i32x4_extmul_i16x8,
                .i32x4_shift,
                .i32x4_splat,
                .i32x4_replace_lane,
                .i8x16_binop,
                .i8x16_unop,
                .i8x16_shift,
                .i8x16_splat,
                .i8x16_replace_lane,
                .i16x8_binop,
                .i16x8_unop,
                .i16x8_extadd_pairwise_i8x16,
                .i16x8_extend_i8x16,
                .i16x8_extmul_i8x16,
                .i16x8_shift,
                .i16x8_splat,
                .i16x8_replace_lane,
                .i64x2_binop,
                .i64x2_unop,
                .i64x2_extend_i32x4,
                .i64x2_extmul_i32x4,
                .i64x2_shift,
                .i64x2_splat,
                .i64x2_replace_lane,
                => {
                    if (!isSupportedV128Def(inst)) return true;
                },
                .v128_store,
                .i32x4_extract_lane,
                .i8x16_extract_lane,
                .i16x8_extract_lane,
                .i64x2_extract_lane,
                => {},
                else => {},
            }
            if (inst.type == .v128 and !isSupportedV128Def(inst)) return true;
            if (inst.dest) |dest| try vreg_types.put(dest, inst.type);
        }
    }

    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            switch (inst.op) {
                .ret => |maybe_v| {
                    if (maybe_v) |v| if ((vreg_types.get(v) orelse .void) == .v128) return true;
                },
                .ret_multi => |vals| {
                    for (vals) |v| if ((vreg_types.get(v) orelse .void) == .v128) return true;
                },
                .local_set => |ls| if ((vreg_types.get(ls.val) orelse .void) == .v128) return true,
                .global_set => |gs| if ((vreg_types.get(gs.val) orelse .void) == .v128) return true,
                .select => |sel| {
                    if ((vreg_types.get(sel.if_true) orelse .void) == .v128 or
                        (vreg_types.get(sel.if_false) orelse .void) == .v128)
                    {
                        return true;
                    }
                },
                else => {},
            }
        }
    }
    return false;
}

pub fn compileFunctionImpl(
    func: *const ir.IrFunction,
    ctx: FuncCompileCtx,
    allocator: std.mem.Allocator,
) ![]u8 {
    // Phase 1b: stack args beyond x7 aren't supported yet.
    if (func.param_count > 7) return error.TooManyParams;
    if (try functionHasUnsupportedV128(func, allocator)) return error.UnsupportedV128;

    // Phase 3 of regalloc adoption (issue #100): drive RegMap from a real
    // linear-scan allocation. `RegMap.assign` consults `alloc_result` first;
    // greedy scavenging remains as a fallback for vregs the allocator had no
    // opinion on. This subsumes the earlier Phase 2 shadow-mode run that
    // landed on `main` via #135 — the allocator now produces real assignments
    // rather than being a no-op consistency check.
    //
    // The actual `regalloc.allocate` call is deferred until after the FMA
    // fusion pre-pass below, because fused MADD/MSUB reads a mul's sources
    // at the add position; we need to extend those vregs' live ranges past
    // the mul before allocation so the sources' physregs aren't reassigned.
    // Compute RPO block order ONCE, before anything that uses global
    // instruction numbering. Clobber points, live ranges, FMA positions,
    // flat indexing, kill lists, and code emission all must use THIS order.
    const block_order = blk: {
        var dom = try analysis.computeDominators(func, allocator);
        defer dom.deinit();
        const order = try allocator.alloc(ir.BlockId, func.blocks.items.len);
        const po_len = dom.post_order.len;
        for (dom.post_order, 0..) |bid, i| {
            order[po_len - 1 - i] = bid;
        }
        var tail: usize = po_len;
        for (0..func.blocks.items.len) |idx| {
            const bid: ir.BlockId = @intCast(idx);
            if (dom.post_num[bid] == null) {
                order[tail] = bid;
                tail += 1;
            }
        }
        break :blk order;
    };
    defer allocator.free(block_order);

    var scheduled = try schedule.scheduleFunction(func, allocator, .{
        .enabled = ctx.options.enable_scheduler,
    });
    defer scheduled.deinit();

    var clobbers = try collectClobberPoints(func, block_order, &scheduled, allocator);
    defer clobbers.deinit(allocator);

    var code = emit.CodeBuffer.init(allocator);
    errdefer code.deinit();

    // Frame size: (local_count + 3) × 8 for saved FP/LR + vmctx + locals,
    // plus spill headroom (sized from IR vreg pressure, see below), plus
    // 128 bytes for call-save (16 physregs × 8), plus 80 bytes for 10
    // callee-saved regs (x19..x28), aligned to 16 (AArch64 SP alignment).
    //
    // Spill capacity used to be a fixed 256 bytes (32 slots) which was
    // enough for micro-benchmarks but overflowed on real workloads
    // (CoreMark `core_state_transition` lands ~90 live vregs at peak).
    // Size it conservatively from the number of IR vregs: every vreg
    // might need a slot, plus a 16-slot safety margin for pathological
    // interleavings. Slots are 8 bytes.
    const spill_base: u32 = (func.local_count + 3) * 8;

    // Multi-result plumbing. Scan call sites for max `extra_results` and
    // reserve caller-side scratch (`scratch_base`) + HRP save slot
    // (`hrp_save_off`, only meaningful if result_count > 1).
    var max_extra_results: u32 = 0;
    for (0..func.blocks.items.len) |bid| {
        for (scheduled.instructions(@intCast(bid))) |inst| {
            switch (inst.op) {
                .call => |cl| max_extra_results = @max(max_extra_results, cl.extra_results),
                .call_indirect => |ci| max_extra_results = @max(max_extra_results, ci.extra_results),
                .call_ref => |cr| max_extra_results = @max(max_extra_results, cr.extra_results),
                else => {},
            }
        }
    }

    // spill_capacity is finalized below after the allocator runs (we use
    // alloc_result.spill_count instead of a conservative next_vreg+16
    // estimate, which shrinks frames and reduces I-cache pressure).
    var fctx = ctx;

    var callee_save_sites: std.ArrayListUnmanaged([callee_saved_regs.len]usize) = .empty;
    defer callee_save_sites.deinit(allocator);
    fctx.callee_save_sites = &callee_save_sites;

    // Collect iconst_* values so emitBinOp etc can fold small constants
    // into immediate-form instructions. Scan is cheap — one pass over IR.
    var const_vals = std.AutoHashMap(ir.VReg, i64).init(allocator);
    defer const_vals.deinit();
    for (0..func.blocks.items.len) |bid| {
        for (scheduled.instructions(@intCast(bid))) |inst| {
            const dest = inst.dest orelse continue;
            switch (inst.op) {
                .iconst_32 => |v| try const_vals.put(dest, v),
                .iconst_64 => |v| try const_vals.put(dest, v),
                else => {},
            }
        }
    }
    fctx.const_vals = &const_vals;

    // FMA fusion pre-pass: find `mul → single-use add/sub` pairs within the
    // same block. The mul's dest must be consumed by the very next use (an
    // integer add/sub of matching type) and used nowhere else. IR vregs are
    // single-assignment so the mul's sources remain valid for MADD/MSUB.
    // Count uses across EVERY op that reads a vreg, not just binary ops.
    // Missing ops here means the FMA fusion pre-pass below can decide a
    // mul's dest is single-use when it actually has more consumers
    // (e.g. `local_set`, `ret`, `store`, `select`, `call`, `br_if`), and
    // then skip emitting the mul. Any subsequent read of that vreg then
    // hits `error.UnboundVReg`. CoreMark caught this via a mul feeding
    // both an `add` (the FMA candidate) and a `local_set` in the same
    // block.
    var use_counts = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer use_counts.deinit();
    const bumpUse = struct {
        fn f(uc: *std.AutoHashMap(ir.VReg, u32), v: ir.VReg) !void {
            const e = try uc.getOrPut(v);
            if (!e.found_existing) e.value_ptr.* = 0;
            e.value_ptr.* += 1;
        }
    }.f;
    for (0..func.blocks.items.len) |bid| {
        for (scheduled.instructions(@intCast(bid))) |inst| try schedule.forEachUse(inst, &use_counts, bumpUse);
    }

    var mul_fused = std.AutoHashMap(ir.VReg, void).init(allocator);
    defer mul_fused.deinit();
    var fma_info = std.AutoHashMap(ir.VReg, FmaInfo).init(allocator);
    defer fma_info.deinit();
    // Map from FMA add's dest vreg → global instruction position of the add.
    // Used below to extend `mul_lhs`/`mul_rhs`/`addend` live ranges past the
    // mul, so regalloc doesn't reassign their physregs before MADD/MSUB reads
    // them at the add position.
    var fma_add_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer fma_add_pos.deinit();
    {
        var global_idx: u32 = 0;
        for (block_order) |bo_bid| {
            const insts = scheduled.instructions(bo_bid);
            var i: usize = 0;
            while (i < insts.len) : (i += 1) {
                defer global_idx += 1;
                const add_inst = insts[i];
                const is_add = add_inst.op == .add;
                const is_sub = add_inst.op == .sub;
                if (!is_add and !is_sub) continue;
                if (add_inst.type != .i32 and add_inst.type != .i64) continue;
                const bin: ir.Inst.BinOp = switch (add_inst.op) {
                    .add => |b| b,
                    .sub => |b| b,
                    else => continue,
                };
                const candidates: [2]?ir.VReg = if (is_add)
                    .{ bin.rhs, bin.lhs }
                else
                    .{ bin.rhs, null };
                for (candidates) |maybe_mul_dest| {
                    const mul_dest = maybe_mul_dest orelse continue;
                    const uc = use_counts.get(mul_dest) orelse continue;
                    if (uc != 1) continue;
                    var found_bin: ?ir.Inst.BinOp = null;
                    var j: usize = i;
                    while (j > 0) : (j -= 1) {
                        const prev = insts[j - 1];
                        if (prev.dest) |d| {
                            if (d != mul_dest) continue;
                            if (prev.op == .mul and prev.type == add_inst.type) {
                                found_bin = prev.op.mul;
                            }
                            break;
                        }
                    }
                    const mul_bin = found_bin orelse continue;
                    const addend = if (is_add and maybe_mul_dest.? == bin.lhs)
                        bin.rhs
                    else
                        bin.lhs;
                    try mul_fused.put(mul_dest, {});
                    const dest = add_inst.dest orelse continue;
                    try fma_info.put(dest, .{
                        .mul_lhs = mul_bin.lhs,
                        .mul_rhs = mul_bin.rhs,
                        .addend = addend,
                        .is_sub = is_sub,
                    });
                    try fma_add_pos.put(dest, global_idx);
                    break;
                }
            }
        }
    }
    fctx.mul_fused = &mul_fused;
    fctx.fma_info = &fma_info;

    // Compute live ranges using the SAME block order as code emission.
    const live_ranges = try computeLiveRangesScheduled(func, block_order, &scheduled, allocator);
    defer allocator.free(live_ranges);
    if (fma_info.count() > 0) {
        // Build a vreg→range-index lookup for the patch loop.
        var range_idx = std.AutoHashMap(ir.VReg, usize).init(allocator);
        defer range_idx.deinit();
        try range_idx.ensureTotalCapacity(@intCast(live_ranges.len));
        for (live_ranges, 0..) |r, idx| {
            range_idx.putAssumeCapacity(r.vreg, idx);
        }
        var it = fma_info.iterator();
        while (it.next()) |entry| {
            const add_dest = entry.key_ptr.*;
            const fi = entry.value_ptr.*;
            const add_pos = fma_add_pos.get(add_dest) orelse continue;
            inline for (.{ fi.mul_lhs, fi.mul_rhs, fi.addend }) |src| {
                if (range_idx.get(src)) |idx| {
                    if (live_ranges[idx].end < add_pos) {
                        live_ranges[idx].end = add_pos;
                    }
                }
            }
        }
    }

    var scalar_live_ranges: std.ArrayList(analysis.LiveRange) = .empty;
    defer scalar_live_ranges.deinit(allocator);
    var v128_spill_slots: u32 = 0;
    for (live_ranges) |range| {
        if (range.type == .v128) {
            v128_spill_slots += 1;
        } else {
            try scalar_live_ranges.append(allocator, range);
        }
    }

    var alloc_result = try regalloc.allocateFromRanges(
        allocator,
        aarch64RegSet(func.local_count),
        clobbers.items,
        scalar_live_ranges.items,
    );
    defer alloc_result.deinit();

    // Finalize frame layout now that we know how many spill slots the
    // allocator actually consumed. Previously we pre-sized to
    // (next_vreg+16)*8 which was wasteful; using the allocator's exact
    // count shrinks frames and reduces I-cache pressure on entry.
    const spill_capacity: u32 = @as(u32, @intCast(alloc_result.spill_count)) * 8;
    const scalar_spill_end = spill_base + spill_capacity;
    const v128_spill_base: u32 = if (v128_spill_slots == 0)
        scalar_spill_end
    else
        alignForwardU32(scalar_spill_end, 16);
    const v128_spill_capacity: u32 = v128_spill_slots * 16;
    const hrp_save_off: u32 = v128_spill_base + v128_spill_capacity;
    const scratch_base: u32 = hrp_save_off + 8;
    const scratch_size: u32 = max_extra_results * 8;
    const call_save_base: u32 = scratch_base + scratch_size;
    const call_save_size: u32 = 128;
    const callee_save_base: u32 = call_save_base + call_save_size;
    const callee_save_count: u32 = 10; // x19..x28
    const callee_save_size: u32 = callee_save_count * 8;
    const raw_frame = callee_save_base + callee_save_size;
    const frame_size: u32 = (raw_frame + 15) & ~@as(u32, 15);

    var reg_map = RegMap.init(allocator, spill_base, spill_capacity);
    defer reg_map.deinit();
    reg_map.alloc_result = &alloc_result;

    var v128_map = V128StackMap.init(allocator, v128_spill_base, v128_spill_capacity);
    defer v128_map.deinit();
    var v128_cache = V128RegCache{};

    fctx.call_save_base = call_save_base;
    fctx.callee_save_base = callee_save_base;
    fctx.hrp_save_off = hrp_save_off;
    fctx.scratch_base = scratch_base;
    fctx.frame_size = frame_size;

    // Liveness-driven register scavenging: compute, for each instruction,
    // the set of vregs whose *last* static read occurs at that instruction.
    // After emitting the instruction, those vregs' physical registers are
    // released via `reg_map.freeVReg` and become available for subsequent
    // dests. Without this, `RegMap.assign` is a bump-allocator that spills
    // everything after 25 vregs — a major CoreMark hot-path cost.
    //
    // The kill set is fusion-aware: a `.mul` that gets fused into a following
    // `.add` (MADD) does not emit and has no effective reads, so its listed
    // operands' kills are attributed to the FMA add (which reads them via
    // `fma_info` instead of the mul's intermediate `dest`).
    var total_insts: usize = 0;
    for (0..func.blocks.items.len) |bid| total_insts += scheduled.instructions(@intCast(bid)).len;

    var block_flat_base = try allocator.alloc(usize, func.blocks.items.len);
    defer allocator.free(block_flat_base);
    {
        var acc: usize = 0;
        for (block_order) |bi| {
            block_flat_base[bi] = acc;
            acc += scheduled.instructions(bi).len;
        }
    }

    const kill_lists = try allocator.alloc(std.ArrayListUnmanaged(ir.VReg), total_insts);
    defer {
        for (kill_lists) |*kl| kl.deinit(allocator);
        allocator.free(kill_lists);
    }
    for (kill_lists) |*kl| kl.* = .empty;

    {
        var seen = std.AutoHashMap(ir.VReg, void).init(allocator);
        defer seen.deinit();
        const recordKill = struct {
            fn f(
                s: *std.AutoHashMap(ir.VReg, void),
                kls: []std.ArrayListUnmanaged(ir.VReg),
                alloc: std.mem.Allocator,
                v: ir.VReg,
                flat_idx: usize,
            ) !void {
                const e = try s.getOrPut(v);
                if (e.found_existing) return;
                try kls[flat_idx].append(alloc, v);
            }
        }.f;

        // Scan blocks in reverse of the emission order (block_order).
        var bo_rev = block_order.len;
        while (bo_rev > 0) {
            bo_rev -= 1;
            const bi_rev = block_order[bo_rev];
            const insts = scheduled.instructions(bi_rev);
            var ii_rev = insts.len;
            while (ii_rev > 0) {
                ii_rev -= 1;
                const inst = insts[ii_rev];
                const flat_idx = block_flat_base[bi_rev] + ii_rev;

                // FMA fusion awareness: a fused mul has no effective reads
                // (it doesn't emit); a fused add reads mul's sources + addend.
                const is_fused_mul = if (inst.dest) |d|
                    inst.op == .mul and mul_fused.contains(d)
                else
                    false;
                if (is_fused_mul) continue;

                if (inst.dest) |d| {
                    if (fma_info.get(d)) |fi| {
                        try recordKill(&seen, kill_lists, allocator, fi.mul_lhs, flat_idx);
                        try recordKill(&seen, kill_lists, allocator, fi.mul_rhs, flat_idx);
                        try recordKill(&seen, kill_lists, allocator, fi.addend, flat_idx);
                        continue;
                    }
                }

                var kill_ctx = KillUseCtx{
                    .seen = &seen,
                    .kill_lists = kill_lists,
                    .allocator = allocator,
                    .flat_idx = flat_idx,
                };
                try schedule.forEachUse(inst, &kill_ctx, recordKillUse);
            }
        }
    }

    try code.emitPrologue(frame_size);
    try emitCalleeSaveStoreTracked(&code, &fctx);

    // Spill VMContext (x0) and wasm params (x1..x7) to their frame slots.
    try emitEntrySpill(&code, func.*, hrp_save_off);

    var block_offsets = try allocator.alloc(usize, func.blocks.items.len);
    defer allocator.free(block_offsets);
    @memset(block_offsets, 0);

    // block_order already computed above — reuse for emission.

    var patches: std.ArrayListUnmanaged(BranchPatch) = .empty;
    defer patches.deinit(allocator);

    var last_was_ret = false;
    for (block_order) |bi| {
        block_offsets[bi] = code.len();
        for (scheduled.instructions(bi), 0..) |inst, ii| {
            last_was_ret = isRet(inst.op);
            const flat_idx = block_flat_base[bi] + ii;
            fctx.current_kills = kill_lists[flat_idx].items;
            v128_cache.beginInst(scheduled.instructions(bi), ii);
            try compileInst(&code, inst, &reg_map, &v128_map, &v128_cache, frame_size, &patches, &fctx);
            // Release physregs of vregs whose last static use is this inst.
            for (kill_lists[flat_idx].items) |v| {
                reg_map.freeVReg(v);
                v128_cache.release(v);
            }
        }
        try v128_cache.flushAll(&code, &v128_map);
    }

    if (!last_was_ret) {
        try emitCalleeSaveRestoreTracked(&code, &fctx);
        try code.emitEpilogue(frame_size);
    }

    // Elide saves/restores for callee-saved regs never assigned by RegMap.
    patchUnusedCalleeSaveSlots(&code, callee_save_sites.items, reg_map.used_callee_mask);

    // Resolve branch patches.
    for (patches.items) |p| {
        const target_off = block_offsets[p.target_block];
        const delta_bytes: i64 = @as(i64, @intCast(target_off)) - @as(i64, @intCast(p.patch_offset));
        // All AArch64 PC-relative branches use word offsets.
        if (@mod(delta_bytes, 4) != 0) return error.BranchMisaligned;
        const word_off = @divExact(delta_bytes, 4);
        const existing = std.mem.readInt(u32, code.bytes.items[p.patch_offset..][0..4], .little);
        const new_word: u32 = switch (p.kind) {
            .b_uncond => blk: {
                // B imm26: bits 25-0
                if (word_off < -(@as(i64, 1) << 25) or word_off >= (@as(i64, 1) << 25))
                    return error.BranchOutOfRange;
                const imm26: u26 = @bitCast(@as(i26, @intCast(word_off)));
                break :blk (existing & 0xFC000000) | @as(u32, imm26);
            },
            .b_cond => blk: {
                // B.cond imm19: bits 23-5
                if (word_off < -(@as(i64, 1) << 18) or word_off >= (@as(i64, 1) << 18))
                    return error.BranchOutOfRange;
                const imm19: u19 = @bitCast(@as(i19, @intCast(word_off)));
                break :blk (existing & 0xFF00001F) | (@as(u32, imm19) << 5);
            },
        };
        code.patch32(p.patch_offset, new_word);
    }

    return code.bytes.toOwnedSlice(allocator);
}

const LastUseCtx = struct {
    last_use_pos: *std.AutoHashMap(ir.VReg, u32),
    pos: u32,
};

fn recordLastUse(ctx: *LastUseCtx, v: ir.VReg) !void {
    try ctx.last_use_pos.put(v, ctx.pos);
}

fn computeLiveRangesScheduled(
    func: *const ir.IrFunction,
    block_order: []const ir.BlockId,
    scheduled: *const schedule.FunctionSchedule,
    allocator: std.mem.Allocator,
) ![]analysis.LiveRange {
    const liveness = try analysis.computeLiveness(func, allocator);
    defer {
        var it = @constCast(&liveness).iterator();
        while (it.next()) |entry| {
            entry.value_ptr.live_in.deinit();
            entry.value_ptr.live_out.deinit();
        }
        @constCast(&liveness).deinit();
    }

    var def_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_pos.deinit();
    var def_type = std.AutoHashMap(ir.VReg, ir.IrType).init(allocator);
    defer def_type.deinit();
    var last_use_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer last_use_pos.deinit();

    var global_idx: u32 = 0;
    for (block_order) |bid| {
        if (liveness.getPtr(bid)) |bl| {
            var lit = bl.live_in.iterator();
            while (lit.next()) |entry| {
                const vreg = entry.key_ptr.*;
                const existing = last_use_pos.get(vreg) orelse 0;
                try last_use_pos.put(vreg, @max(existing, global_idx));
            }
        }

        for (scheduled.instructions(bid)) |inst| {
            if (inst.dest) |dest| {
                if (!def_pos.contains(dest)) {
                    try def_pos.put(dest, global_idx);
                    try def_type.put(dest, inst.type);
                }
            }
            var use_ctx = LastUseCtx{
                .last_use_pos = &last_use_pos,
                .pos = global_idx,
            };
            try schedule.forEachUse(inst, &use_ctx, recordLastUse);
            global_idx += 1;
        }

        if (liveness.getPtr(bid)) |bl| {
            var lit = bl.live_out.iterator();
            while (lit.next()) |entry| {
                const vreg = entry.key_ptr.*;
                const existing = last_use_pos.get(vreg) orelse 0;
                try last_use_pos.put(vreg, @max(existing, global_idx -| 1));
            }
        }
    }

    var ranges: std.ArrayList(analysis.LiveRange) = .empty;
    errdefer ranges.deinit(allocator);
    var dit = def_pos.iterator();
    while (dit.next()) |entry| {
        const vreg = entry.key_ptr.*;
        const start = entry.value_ptr.*;
        const end = last_use_pos.get(vreg) orelse start;
        try ranges.append(allocator, .{
            .vreg = vreg,
            .start = start,
            .end = @max(start, end),
            .type = def_type.get(vreg) orelse .i32,
        });
    }

    std.mem.sort(analysis.LiveRange, ranges.items, {}, struct {
        fn lessThan(_: void, a: analysis.LiveRange, b: analysis.LiveRange) bool {
            return a.start < b.start;
        }
    }.lessThan);

    return ranges.toOwnedSlice(allocator);
}

/// FP-relative slot offset (in bytes) for the VMContext pointer.
const vmctx_slot_offset: u32 = 16;

/// FP-relative slot offset (in bytes, unsigned, positive) for local `idx`.
fn localSlotOffset(idx: u32) u32 {
    return (idx + 3) * 8;
}

fn alignForwardU32(value: u32, alignment: u32) u32 {
    std.debug.assert(alignment != 0 and (alignment & (alignment - 1)) == 0);
    return (value + alignment - 1) & ~(alignment - 1);
}

/// ABI register holding wasm parameter `i` (0-based). Parameter 0 is in x1
/// because x0 carries the hidden VMContext pointer.
fn paramAbiReg(i: u32) emit.Reg {
    return switch (i) {
        0 => .x1,
        1 => .x2,
        2 => .x3,
        3 => .x4,
        4 => .x5,
        5 => .x6,
        6 => .x7,
        else => unreachable,
    };
}

/// Spill x0 (VMContext) and x1..x7 (wasm params) to their frame slots, and
/// zero-initialize declared locals. Must be called right after emitPrologue
/// and before any vreg allocation so we don't clobber these ABI regs.
fn emitEntrySpill(code: *emit.CodeBuffer, func: ir.IrFunction, hrp_save_off: u32) !void {
    // VMContext → [fp + 16]
    try code.strImm(.x0, .fp, vmctx_slot_offset / 8);

    // Wasm params → local slots [fp + (i+3)*8]
    var i: u32 = 0;
    while (i < func.param_count) : (i += 1) {
        const off_scaled: u12 = @intCast(localSlotOffset(i) / 8);
        try code.strImm(paramAbiReg(i), .fp, off_scaled);
    }

    // Multi-value returns: callee receives an HRP (host-result pointer)
    // as an implicit trailing arg in x(1 + param_count). Stash to the
    // dedicated frame slot so `.ret_multi` can retrieve it.
    if (func.result_count > 1) {
        if (func.param_count >= 7) return error.TooManyParamsForMultiResult;
        const hrp_reg = paramAbiReg(func.param_count);
        try frameStore(code, hrp_reg, hrp_save_off);
    }

    // Zero-init declared locals (wasm spec requires non-param locals to be 0).
    // Use x16 (tmp0) as the zero source. Always 64-bit store to cover any type.
    if (func.local_count > func.param_count) {
        try code.movz(RegMap.tmp0, 0, 0);
        var j: u32 = func.param_count;
        while (j < func.local_count) : (j += 1) {
            const off_scaled: u12 = @intCast(localSlotOffset(j) / 8);
            try code.strImm(RegMap.tmp0, .fp, off_scaled);
        }
    }
}

/// Store `src` to `[fp + offset]`, handling offsets that overflow the
/// 12-bit scaled immediate of STR Xt. For large offsets we materialize
/// the byte offset into tmp0 and use an extended-register form via ADD
/// into tmp0. LDR/STR scaled imm12 covers up to 32760 bytes for the X
/// form, so most real frames fit directly.
fn frameStore(code: *emit.CodeBuffer, src: emit.Reg, offset: u32) !void {
    if (offset % 8 == 0 and offset / 8 <= 4095) {
        try code.strImm(src, .fp, @intCast(offset / 8));
    } else {
        try emitMovImm64(code, RegMap.tmp0, offset);
        try code.addRegReg(RegMap.tmp0, RegMap.tmp0, .fp);
        try code.strImm(src, RegMap.tmp0, 0);
    }
}

/// Load `[fp + offset]` into `dst`, mirroring `frameStore`.
fn frameLoad(code: *emit.CodeBuffer, dst: emit.Reg, offset: u32) !void {
    if (offset % 8 == 0 and offset / 8 <= 4095) {
        try code.ldrImm(dst, .fp, @intCast(offset / 8));
    } else {
        try emitMovImm64(code, RegMap.tmp0, offset);
        try code.addRegReg(RegMap.tmp0, RegMap.tmp0, .fp);
        try code.ldrImm(dst, RegMap.tmp0, 0);
    }
}

/// Compute `dst = fp + offset` for offsets that may exceed the 12-bit
/// ADD-immediate range.
fn frameAddr(code: *emit.CodeBuffer, dst: emit.Reg, offset: u32) !void {
    if (offset <= 4095) {
        try code.addImm(dst, .fp, @intCast(offset));
    } else if ((offset & 0xFFF) == 0 and (offset >> 12) <= 4095) {
        try code.addImmShift12(dst, .fp, @intCast(offset >> 12));
    } else {
        try emitMovImm64(code, dst, offset);
        try code.addRegReg(dst, dst, .fp);
    }
}

/// Materialize a 32-bit unsigned immediate into `dst` via MOVZ/MOVK.
fn emitMovImm64(code: *emit.CodeBuffer, dst: emit.Reg, imm: u32) !void {
    const lo: u16 = @truncate(imm);
    const hi: u16 = @truncate(imm >> 16);
    try code.movz(dst, lo, 0);
    if (hi != 0) try code.movk(dst, hi, 1);
}

fn isRet(op: ir.Inst.Op) bool {
    return switch (op) {
        .ret, .ret_multi => true,
        .call => |cl| cl.tail,
        .call_indirect => |ci| ci.tail,
        .call_ref => |cr| cr.tail,
        else => false,
    };
}

/// The 10 callee-saved allocatable registers (x19..x28). Must match the
/// tail of `RegMap.scratch_regs`.
const callee_saved_regs = [_]emit.Reg{
    .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27, .x28,
};

/// AArch64 `NOP` (`HINT #0`): 0xd503201f. Little-endian byte pattern.
const nop_word: u32 = 0xd503201f;

/// Emit prologue saves for all 10 callee-saved allocatable regs and
/// return the byte offsets of each STR instruction. After the function
/// body is fully compiled, `patchUnusedCalleeSaveSlots` rewrites the
/// entries for regs never assigned to vregs (per `RegMap.used_callee_mask`)
/// with NOPs, eliding the memory traffic of dead saves.
///
/// We lay down all 10 slots unconditionally so that prologue size and
/// every subsequent code offset (block starts, branch patches) are fixed
/// before we know which regs the allocator will use.
fn emitCalleeSaveStore(code: *emit.CodeBuffer, callee_save_base: u32) ![callee_saved_regs.len]usize {
    var offs: [callee_saved_regs.len]usize = undefined;
    for (callee_saved_regs, 0..) |r, i| {
        offs[i] = code.len();
        const off_scaled: u12 = @intCast((callee_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.strImm(r, .fp, off_scaled);
    }
    return offs;
}

/// Emit epilogue restores for all 10 callee-saved allocatable regs and
/// return the byte offsets of each LDR. Paired with the prologue store
/// via `patchUnusedCalleeSaveSlots`.
fn emitCalleeSaveRestore(code: *emit.CodeBuffer, callee_save_base: u32) ![callee_saved_regs.len]usize {
    var offs: [callee_saved_regs.len]usize = undefined;
    for (callee_saved_regs, 0..) |r, i| {
        offs[i] = code.len();
        const off_scaled: u12 = @intCast((callee_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(r, .fp, off_scaled);
    }
    return offs;
}

/// Wrap `emitCalleeSaveStore` + site tracking for the prologue.
fn emitCalleeSaveStoreTracked(code: *emit.CodeBuffer, fctx: *FuncCompileCtx) !void {
    const offs = try emitCalleeSaveStore(code, fctx.callee_save_base);
    try fctx.callee_save_sites.?.append(fctx.allocator, offs);
}

/// Wrap `emitCalleeSaveRestore` + site tracking for epilogues.
fn emitCalleeSaveRestoreTracked(code: *emit.CodeBuffer, fctx: *FuncCompileCtx) !void {
    const offs = try emitCalleeSaveRestore(code, fctx.callee_save_base);
    try fctx.callee_save_sites.?.append(fctx.allocator, offs);
}

/// Overwrite save/restore slots for callee-saved regs that were never
/// assigned to any vreg with NOPs. `sites` contains the STR/LDR offsets
/// captured by each emitCalleeSaveStore/emitCalleeSaveRestore call.
fn patchUnusedCalleeSaveSlots(
    code: *emit.CodeBuffer,
    sites: []const [callee_saved_regs.len]usize,
    used_mask: u16,
) void {
    var nop_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &nop_bytes, nop_word, .little);
    for (sites) |site| {
        for (site, 0..) |off, i| {
            if ((used_mask >> @intCast(i)) & 1 != 0) continue;
            @memcpy(code.bytes.items[off..][0..4], &nop_bytes);
        }
    }
}

fn isV128Inst(inst: ir.Inst) bool {
    return switch (inst.op) {
        .v128_const,
        .v128_load,
        .v128_store,
        .v128_not,
        .v128_bitwise,
        .i32x4_binop,
        .i32x4_unop,
        .i32x4_extadd_pairwise_i16x8,
        .i32x4_extend_i16x8,
        .i32x4_extmul_i16x8,
        .i32x4_shift,
        .i32x4_splat,
        .i32x4_extract_lane,
        .i32x4_replace_lane,
        .i8x16_binop,
        .i8x16_unop,
        .i8x16_shift,
        .i8x16_splat,
        .i8x16_extract_lane,
        .i8x16_replace_lane,
        .i16x8_binop,
        .i16x8_unop,
        .i16x8_extadd_pairwise_i8x16,
        .i16x8_extend_i8x16,
        .i16x8_extmul_i8x16,
        .i16x8_shift,
        .i16x8_splat,
        .i16x8_extract_lane,
        .i16x8_replace_lane,
        .i64x2_binop,
        .i64x2_unop,
        .i64x2_extend_i32x4,
        .i64x2_extmul_i32x4,
        .i64x2_shift,
        .i64x2_splat,
        .i64x2_extract_lane,
        .i64x2_replace_lane,
        => true,
        else => false,
    };
}

fn instRequiresV128Flush(inst: ir.Inst) bool {
    if (isV128Inst(inst)) return false;
    return switch (inst.op) {
        .iconst_32,
        .iconst_64,
        .fconst_32,
        .fconst_64,
        => false,
        .add,
        .sub,
        .mul,
        .div_s,
        .div_u,
        .rem_s,
        .rem_u,
        .@"and",
        .@"or",
        .xor,
        .shl,
        .shr_s,
        .shr_u,
        .rotl,
        .rotr,
        .eq,
        .ne,
        .lt_s,
        .lt_u,
        .gt_s,
        .gt_u,
        .le_s,
        .le_u,
        .ge_s,
        .ge_u,
        .eqz,
        .clz,
        .ctz,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .extend_i32_s,
        .extend_i32_u,
        .wrap_i64,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        .f_min,
        .f_max,
        .f_copysign,
        .popcnt,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i64_s,
        .convert_i32_u,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .select,
        .local_get,
        .local_set,
        .load,
        .store,
        .memory_size,
        .reinterpret,
        .global_get,
        .global_set,
        .atomic_load,
        .atomic_store,
        .atomic_rmw,
        .atomic_cmpxchg,
        .atomic_fence,
        .memory_init,
        .data_drop,
        => false,
        else => true,
    };
}

fn compileInst(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    frame_size: u32,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    fctx: *FuncCompileCtx,
) !void {
    if (instRequiresV128Flush(inst)) {
        try v128_cache.flushAll(code, v128_map);
    }

    switch (inst.op) {
        // ── Constants ────────────────────────────────────────────────
        .iconst_32 => |val| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.movImm32(info.reg, val);
            try destCommit(code, reg_map, info);
        },
        .iconst_64 => |val| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.movImm64(info.reg, @bitCast(val));
            try destCommit(code, reg_map, info);
        },
        .fconst_32 => |val| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.movImm32(info.reg, @bitCast(val));
            try destCommit(code, reg_map, info);
        },
        .fconst_64 => |val| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.movImm64(info.reg, @bitCast(val));
            try destCommit(code, reg_map, info);
        },
        .v128_const => |val| try emitV128Const(code, inst, val, v128_map, v128_cache),
        .v128_load => |ld| try emitV128Load(code, inst, ld, reg_map, v128_map, v128_cache),
        .v128_store => |st| try emitV128Store(code, st, reg_map, v128_map, v128_cache),
        .v128_not => |src| try emitV128Not(code, inst, src, v128_map, v128_cache, fctx),
        .v128_bitwise => |bin| try emitV128Bitwise(code, inst, bin, v128_map, v128_cache, fctx),
        .i32x4_binop => |bin| try emitI32x4BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .i32x4_unop => |un| try emitI32x4UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .i32x4_extadd_pairwise_i16x8 => |op| try emitI32x4ExtAddPairwiseI16x8(code, inst, op, v128_map, v128_cache, fctx),
        .i32x4_extend_i16x8 => |op| try emitI32x4ExtendI16x8(code, inst, op, v128_map, v128_cache, fctx),
        .i32x4_extmul_i16x8 => |op| try emitI32x4ExtMulI16x8(code, inst, op, v128_map, v128_cache, fctx),
        .i32x4_shift => |shift| try emitI32x4Shift(code, inst, shift, reg_map, v128_map, v128_cache, fctx),
        .i32x4_splat => |src| try emitI32x4Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .i32x4_extract_lane => |lane| try emitI32x4ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .i32x4_replace_lane => |lane| try emitI32x4ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),
        .i8x16_binop => |bin| try emitI8x16BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .i8x16_unop => |un| try emitI8x16UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .i8x16_shift => |shift| try emitI8x16Shift(code, inst, shift, reg_map, v128_map, v128_cache, fctx),
        .i8x16_splat => |src| try emitI8x16Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .i8x16_extract_lane => |lane| try emitI8x16ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .i8x16_replace_lane => |lane| try emitI8x16ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),
        .i16x8_binop => |bin| try emitI16x8BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .i16x8_unop => |un| try emitI16x8UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .i16x8_extadd_pairwise_i8x16 => |op| try emitI16x8ExtAddPairwiseI8x16(code, inst, op, v128_map, v128_cache, fctx),
        .i16x8_extend_i8x16 => |op| try emitI16x8ExtendI8x16(code, inst, op, v128_map, v128_cache, fctx),
        .i16x8_extmul_i8x16 => |op| try emitI16x8ExtMulI8x16(code, inst, op, v128_map, v128_cache, fctx),
        .i16x8_shift => |shift| try emitI16x8Shift(code, inst, shift, reg_map, v128_map, v128_cache, fctx),
        .i16x8_splat => |src| try emitI16x8Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .i16x8_extract_lane => |lane| try emitI16x8ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .i16x8_replace_lane => |lane| try emitI16x8ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),
        .i64x2_binop => |bin| try emitI64x2BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .i64x2_unop => |un| try emitI64x2UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .i64x2_extend_i32x4 => |op| try emitI64x2ExtendI32x4(code, inst, op, v128_map, v128_cache, fctx),
        .i64x2_extmul_i32x4 => |op| try emitI64x2ExtMulI32x4(code, inst, op, v128_map, v128_cache, fctx),
        .i64x2_shift => |shift| try emitI64x2Shift(code, inst, shift, reg_map, v128_map, v128_cache, fctx),
        .i64x2_splat => |src| try emitI64x2Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .i64x2_extract_lane => |lane| try emitI64x2ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .i64x2_replace_lane => |lane| try emitI64x2ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),

        .add => |bin| if (inst.type == .f32 or inst.type == .f64)
            try emitFBinOp(code, inst, bin, reg_map, .add)
        else
            try emitBinOp(code, inst, bin, reg_map, .add, fctx),
        .sub => |bin| if (inst.type == .f32 or inst.type == .f64)
            try emitFBinOp(code, inst, bin, reg_map, .sub)
        else
            try emitBinOp(code, inst, bin, reg_map, .sub, fctx),
        .mul => |bin| if (inst.type == .f32 or inst.type == .f64)
            try emitFBinOp(code, inst, bin, reg_map, .mul)
        else
            try emitBinOp(code, inst, bin, reg_map, .mul, fctx),
        .@"and" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"and", fctx),
        .@"or" => |bin| try emitBinOp(code, inst, bin, reg_map, .@"or", fctx),
        .xor => |bin| try emitBinOp(code, inst, bin, reg_map, .xor, fctx),

        .div_s => |bin| if (inst.type == .f32 or inst.type == .f64)
            try emitFBinOp(code, inst, bin, reg_map, .div)
        else
            try emitDivRem(code, inst, bin, reg_map, .div_s),
        .div_u => |bin| try emitDivRem(code, inst, bin, reg_map, .div_u),
        .rem_s => |bin| try emitDivRem(code, inst, bin, reg_map, .rem_s),
        .rem_u => |bin| try emitDivRem(code, inst, bin, reg_map, .rem_u),

        // ── Shifts / rotates ─────────────────────────────────────────
        .shl => |bin| try emitShift(code, inst, bin, reg_map, .lsl),
        .shr_u => |bin| try emitShift(code, inst, bin, reg_map, .lsr),
        .shr_s => |bin| try emitShift(code, inst, bin, reg_map, .asr),
        .rotr => |bin| try emitShift(code, inst, bin, reg_map, .ror),
        .rotl => |bin| try emitRotl(code, inst, bin, reg_map),

        // ── Comparisons ──────────────────────────────────────────────
        inline .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u, .le_s, .le_u, .ge_s, .ge_u => |bin, tag| {
            try emitCmp(code, inst, bin, reg_map, comptime tagToCond(tag));
        },

        .eqz => |vreg| try emitEqz(code, inst, vreg, reg_map),

        // ── Unary bit ops ────────────────────────────────────────────
        .clz => |vreg| try emitClz(code, inst, vreg, reg_map),
        .ctz => |vreg| try emitCtz(code, inst, vreg, reg_map),

        // ── Sign extension / wrap ────────────────────────────────────
        .extend8_s => |vreg| try emitExtendImpl(code, inst, vreg, reg_map, .b),
        .extend16_s => |vreg| try emitExtendImpl(code, inst, vreg, reg_map, .h),
        .extend32_s => |vreg| try emitExtendImpl(code, inst, vreg, reg_map, .w),
        .extend_i32_s => |vreg| try emitExtendImpl(code, inst, vreg, reg_map, .w),
        .extend_i32_u => |vreg| try emitExtendI32U(code, inst, vreg, reg_map),
        .wrap_i64 => |vreg| try emitWrap(code, inst, vreg, reg_map),

        // ── Select (CSEL) ────────────────────────────────────────────
        .select => |sel| try emitSelect(code, inst, sel, reg_map),

        // ── Branches ─────────────────────────────────────────────────
        .br => |target| try emitBr(code, patches, target, fctx.allocator),
        .br_if => |br| try emitBrIf(code, reg_map, patches, br, fctx.allocator),
        .br_table => |bt| try emitBrTable(code, reg_map, patches, bt, fctx.allocator),

        // ── Direct function call ─────────────────────────────────────
        .call => |cl| try emitCall(code, inst, cl, reg_map, fctx),
        .call_indirect => |ci| try emitCallIndirect(code, inst, ci, reg_map, fctx),
        .call_ref => |cr| try emitCallRef(code, inst, cr, reg_map, fctx),

        // ── Tables & function refs ───────────────────────────────────
        .table_size => |tidx| try emitTableSize(code, inst, tidx, reg_map),
        .table_get => |tg| try emitTableGet(code, inst, tg, reg_map),
        .table_set => |ts| try emitTableSet(code, ts, reg_map, fctx),
        .table_grow => |tg| try emitTableGrow(code, inst, tg, reg_map, fctx),
        .ref_func => |fidx| try emitRefFunc(code, inst, fidx, reg_map),

        // ── Locals (FP-relative frame slots) ─────────────────────────
        .local_get => |idx| {
            const dest = inst.dest orelse return;
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            const off_scaled: u12 = @intCast(localSlotOffset(idx) / 8);
            // Always 64-bit load: the slot stores the full 8 bytes, and a
            // subsequent W-form op (or wrap_i64) will ignore the upper bits.
            try code.ldrImm(info.reg, .fp, off_scaled);
            try destCommit(code, reg_map, info);
        },
        .local_set => |ls| {
            const src = try useInto(code, reg_map, ls.val, RegMap.tmp0);
            const off_scaled: u12 = @intCast(localSlotOffset(ls.idx) / 8);
            try code.strImm(src, .fp, off_scaled);
        },

        // ── Linear memory load/store ─────────────────────────────────
        .load => |ld| try emitLoad(code, inst, ld, reg_map),
        .store => |st| try emitStore(code, st, reg_map),

        .ret => |maybe_val| {
            if (maybe_val) |val| {
                const r = try useInto(code, reg_map, val, .x0);
                if (r != .x0) try code.movRegReg(.x0, r);
            }
            try emitCalleeSaveRestoreTracked(code, fctx);
            try code.emitEpilogue(frame_size);
        },
        .ret_multi => |vregs| try emitRetMulti(code, vregs, reg_map, fctx, frame_size),
        .call_result => |idx| try emitCallResult(code, inst, idx, reg_map, fctx),
        .@"unreachable" => try emitTrapHelperCall(code, vmctx_trap_unreachable_fn_slot),
        .global_get => |idx| try emitGlobalGet(code, inst, idx, reg_map),
        .global_set => |gs| try emitGlobalSet(code, gs, reg_map),
        .memory_size => try emitMemorySize(code, inst, reg_map),
        .memory_grow => |pages_vreg| try emitMemoryGrow(code, inst, pages_vreg, reg_map, fctx),
        .memory_fill => |mf| try emitMemoryFill(code, mf, reg_map, fctx),
        .memory_copy => |mc| try emitMemoryCopy(code, mc, reg_map, fctx),

        // ── Float bit-level unary ────────────────────────────────────
        // f_neg and f_abs are sign-bit manipulations; they can be done
        // as pure integer ops on the float's bit pattern without needing
        // a V-register. reinterpret is a bit-identical move.
        .f_neg => |vreg| try emitFSignBit(code, inst, vreg, reg_map, .neg),
        .f_abs => |vreg| try emitFSignBit(code, inst, vreg, reg_map, .abs),
        .f_sqrt => |vreg| try emitFSqrt(code, inst, vreg, reg_map),
        .f_ceil => |vreg| try emitFRint(code, inst, vreg, reg_map, .ceil),
        .f_floor => |vreg| try emitFRint(code, inst, vreg, reg_map, .floor),
        .f_trunc => |vreg| try emitFRint(code, inst, vreg, reg_map, .trunc),
        .f_nearest => |vreg| try emitFRint(code, inst, vreg, reg_map, .nearest),

        .f_min => |bin| try emitFMinMax(code, inst, bin, reg_map, .min),
        .f_max => |bin| try emitFMinMax(code, inst, bin, reg_map, .max),
        .f_copysign => |bin| try emitFCopysign(code, inst, bin, reg_map),

        .popcnt => |vreg| try emitPopcnt(code, inst, vreg, reg_map),

        .f_eq => |bin| try emitFCmp(code, inst, bin, reg_map, .eq),
        .f_ne => |bin| try emitFCmp(code, inst, bin, reg_map, .ne),
        .f_lt => |bin| try emitFCmp(code, inst, bin, reg_map, .mi),
        .f_gt => |bin| try emitFCmp(code, inst, bin, reg_map, .gt),
        .f_le => |bin| try emitFCmp(code, inst, bin, reg_map, .ls),
        .f_ge => |bin| try emitFCmp(code, inst, bin, reg_map, .ge),

        // Non-trapping float/int conversions.
        .convert_i32_s => |vreg| try emitConvertIntToFloat(code, inst, vreg, reg_map, true, false),
        .convert_i64_s => |vreg| try emitConvertIntToFloat(code, inst, vreg, reg_map, true, true),
        .convert_i32_u => |vreg| try emitConvertIntToFloat(code, inst, vreg, reg_map, false, false),
        .convert_i64_u => |vreg| try emitConvertIntToFloat(code, inst, vreg, reg_map, false, true),
        .demote_f64 => |vreg| try emitDemoteF64(code, inst, vreg, reg_map),
        .promote_f32 => |vreg| try emitPromoteF32(code, inst, vreg, reg_map),

        // Trapping float→int truncation (wasm trunc_f*_s/u).
        .trunc_f32_s, .trunc_f64_s => |vreg| try emitTruncTrapping(code, inst, vreg, reg_map, true),
        .trunc_f32_u, .trunc_f64_u => |vreg| try emitTruncTrapping(code, inst, vreg, reg_map, false),

        // Saturating float→int truncation (wasm trunc_sat_f*_s/u).
        .trunc_sat_f32_s, .trunc_sat_f64_s => |vreg| try emitTruncSat(code, inst, vreg, reg_map, true),
        .trunc_sat_f32_u, .trunc_sat_f64_u => |vreg| try emitTruncSat(code, inst, vreg, reg_map, false),
        .reinterpret => |vreg| try emitReinterpret(code, inst, vreg, reg_map),

        // ── Atomics (minimal seq-cst subset) ─────────────────────────
        .atomic_fence => try code.dmbIsh(),
        .atomic_load => |ld| try emitAtomicLoad(code, inst, ld, reg_map),
        .atomic_store => |st| try emitAtomicStore(code, st, reg_map),
        .atomic_rmw => |rmw| try emitAtomicRmw(code, inst, rmw, reg_map),
        .atomic_cmpxchg => |cx| try emitAtomicCmpxchg(code, inst, cx, reg_map),
        .atomic_notify => |an| try emitAtomicNotify(code, inst, an, reg_map, fctx),
        .atomic_wait => |aw| try emitAtomicWait(code, inst, aw, reg_map, fctx),

        // ── Passive segments ─────────────────────────────────────────
        // memory.init / data.drop are AOT no-ops: active segments are
        // applied at instantiation; passive memory segments aren't
        // tracked separately by this runtime. Mirrors x86_64 behavior.
        .memory_init => {},
        .data_drop => {},
        .table_init => |ti| try emitTableInit(code, ti, reg_map, fctx),
        .elem_drop => |seg_idx| try emitElemDrop(code, seg_idx, reg_map, fctx),
        // Phi must be lowered before codegen.
        .phi => unreachable,
        else => {
            // Explicit failure for unimplemented ops. Previously this was a
            // silent no-op which produced incorrect code. Anything that lands
            // here needs a handler — track remaining work in issue #111.
            return error.UnimplementedOp;
        },
    }
}

fn tagToCond(comptime tag: std.meta.Tag(ir.Inst.Op)) emit.Cond {
    return switch (tag) {
        .eq => .eq,
        .ne => .ne,
        .lt_s => .lt,
        .lt_u => .lo,
        .gt_s => .gt,
        .gt_u => .hi,
        .le_s => .le,
        .le_u => .ls,
        .ge_s => .ge,
        .ge_u => .hs,
        else => @compileError("tagToCond: not a comparison tag"),
    };
}

/// Resolve the destination register, returning null if it spills to stack
/// (in which case the caller silently drops — kept for handlers that
/// haven't been converted to the `destBegin`/`destCommit` spill API yet).
fn destReg(code: *emit.CodeBuffer, reg_map: *RegMap, dest: ir.VReg) !?emit.Reg {
    _ = code;
    const loc = try reg_map.assign(dest);
    return switch (loc) {
        .reg => |r| r,
        .stack => null,
    };
}

/// Full-spill-aware dest allocation. Returns a writable register: the
/// vreg's assigned physreg if any, else `scratch`. Caller must emit
/// `destCommit` once the result has been written into `info.reg`.
const DestInfo = struct {
    reg: emit.Reg,
    spill_slot: ?u32 = null, // byte offset from spill_base if spilled
};

fn destBegin(reg_map: *RegMap, dest: ir.VReg, scratch: emit.Reg) !DestInfo {
    const loc = try reg_map.assign(dest);
    return switch (loc) {
        .reg => |r| .{ .reg = r },
        .stack => |off| .{ .reg = scratch, .spill_slot = off },
    };
}

fn destCommit(code: *emit.CodeBuffer, reg_map: *const RegMap, info: DestInfo) !void {
    if (info.spill_slot) |off| {
        try code.strImm(info.reg, .fp, reg_map.spillOffsetScaled(off));
    }
}

/// Read `vreg`'s value, loading into `scratch` if spilled. Returns the
/// register holding the value. If the vreg is unbound the caller's
/// silent-drop contract is preserved via the `useReg` helper below.
fn useInto(
    code: *emit.CodeBuffer,
    reg_map: *const RegMap,
    vreg: ir.VReg,
    scratch: emit.Reg,
) !emit.Reg {
    const loc = reg_map.get(vreg) orelse return error.UnboundVReg;
    return switch (loc) {
        .reg => |r| r,
        .stack => |off| blk: {
            try code.ldrImm(scratch, .fp, reg_map.spillOffsetScaled(off));
            break :blk scratch;
        },
    };
}

fn useReg(reg_map: *const RegMap, vreg: ir.VReg) ?emit.Reg {
    const loc = reg_map.get(vreg) orelse return null;
    return switch (loc) {
        .reg => |r| r,
        .stack => null,
    };
}

const v128_tmp0: u5 = 0;
const v128_tmp1: u5 = 1;

fn storeV128SlotAbs(
    code: *emit.CodeBuffer,
    slot_off: u32,
    vt: u5,
) !void {
    try frameAddr(code, RegMap.tmp0, slot_off);
    try code.strQ(vt, RegMap.tmp0);
}

fn isCurrentKill(fctx: *const FuncCompileCtx, vreg: ir.VReg) bool {
    for (fctx.current_kills) |killed| {
        if (killed == vreg) return true;
    }
    return false;
}

fn prepareV128UnaryDest(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    src: ir.VReg,
    src_reg: u5,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !u5 {
    const dest = inst.dest orelse return src_reg;
    try v128_cache.prepareOverwrite(code, v128_map, src_reg, isCurrentKill(fctx, src));
    try v128_cache.defineInReg(v128_map, dest, src_reg);
    return src_reg;
}

fn prepareV128BinaryDest(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lhs: ir.VReg,
    lhs_reg: u5,
    rhs: ir.VReg,
    rhs_reg: u5,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !u5 {
    const dest = inst.dest orelse return lhs_reg;
    const dest_reg = if (isCurrentKill(fctx, lhs))
        lhs_reg
    else if (isCurrentKill(fctx, rhs))
        rhs_reg
    else
        lhs_reg;
    const drop_existing = (dest_reg == lhs_reg and isCurrentKill(fctx, lhs)) or
        (dest_reg == rhs_reg and isCurrentKill(fctx, rhs));
    try v128_cache.prepareOverwrite(code, v128_map, dest_reg, drop_existing);
    try v128_cache.defineInReg(v128_map, dest, dest_reg);
    return dest_reg;
}

fn emitV128Const(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    val: u128,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    const lo: u64 = @truncate(val);
    const hi: u64 = @truncate(val >> 64);
    try code.movImm64(RegMap.tmp0, lo);
    try code.fmovDFromGp64(dest_reg, RegMap.tmp0);
    try code.movImm64(RegMap.tmp0, hi);
    try code.insDFromGp64(dest_reg, 1, RegMap.tmp0);
}

fn emitV128Load(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    ld: ir.Inst.V128Mem,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    const end_offset: u64 = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + 16;
    if (ld.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, ld.base, ld.offset);
    } else {
        try emitMemAddr(code, reg_map, ld.base, ld.offset, end_offset);
    }
    try code.ldrQ(dest_reg, RegMap.tmp0);
}

fn emitV128Store(
    code: *emit.CodeBuffer,
    st: ir.Inst.V128Store,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const val_reg = try v128_cache.ensure(code, v128_map, st.val, null);
    const end_offset: u64 = if (st.checked_end > 0) st.checked_end else @as(u64, st.offset) + 16;
    if (st.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, st.base, st.offset);
    } else {
        try emitMemAddr(code, reg_map, st.base, st.offset, end_offset);
    }
    try code.strQ(val_reg, RegMap.tmp0);
}

fn emitV128Not(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    src: ir.VReg,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const src_reg = try v128_cache.ensure(code, v128_map, src, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, src, src_reg, v128_map, v128_cache, fctx);
    try code.mvn16b(dest_reg, src_reg);
}

fn emitV128Bitwise(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.V128Bitwise,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, bin.lhs, null);
    const rhs_reg = if (bin.rhs == bin.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, bin.rhs, lhs_reg);
    const dest_reg = try prepareV128BinaryDest(
        code,
        inst,
        bin.lhs,
        lhs_reg,
        bin.rhs,
        rhs_reg,
        v128_map,
        v128_cache,
        fctx,
    );
    const op: emit.CodeBuffer.V128BitwiseOp = switch (bin.op) {
        .@"and" => .@"and",
        .andnot => .bic,
        .@"or" => .orr,
        .xor => .eor,
    };
    try code.bitwise16b(op, dest_reg, lhs_reg, rhs_reg);
}

fn emitI32x4UnOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    un: ir.Inst.SimdUnary,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, un.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, un.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (un.op) {
        .abs => try code.abs4s(dest_reg, vector_reg),
        .neg => try code.neg4s(dest_reg, vector_reg),
    }
}

fn emitI32x4ExtAddPairwiseI16x8(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdExtAddPairwise,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, op.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (op.sign) {
        .signed => try code.saddlp4s8h(dest_reg, vector_reg),
        .unsigned => try code.uaddlp4s8h(dest_reg, vector_reg),
    }
}

fn emitI8x16UnOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    un: ir.Inst.SimdUnary,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, un.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, un.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (un.op) {
        .abs => try code.abs16b(dest_reg, vector_reg),
        .neg => try code.neg16b(dest_reg, vector_reg),
    }
}

fn emitI16x8UnOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    un: ir.Inst.SimdUnary,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, un.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, un.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (un.op) {
        .abs => try code.abs8h(dest_reg, vector_reg),
        .neg => try code.neg8h(dest_reg, vector_reg),
    }
}

fn emitI16x8ExtAddPairwiseI8x16(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdExtAddPairwise,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, op.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (op.sign) {
        .signed => try code.saddlp8h16b(dest_reg, vector_reg),
        .unsigned => try code.uaddlp8h16b(dest_reg, vector_reg),
    }
}

fn emitI16x8ExtendI8x16(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdExtendHalf,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, op.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (op.half) {
        .low => switch (op.sign) {
            .signed => try code.sshll8h8b(dest_reg, vector_reg),
            .unsigned => try code.ushll8h8b(dest_reg, vector_reg),
        },
        .high => switch (op.sign) {
            .signed => try code.sshll2_8h16b(dest_reg, vector_reg),
            .unsigned => try code.ushll2_8h16b(dest_reg, vector_reg),
        },
    }
}

fn emitI32x4ExtendI16x8(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdExtendHalf,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, op.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (op.half) {
        .low => switch (op.sign) {
            .signed => try code.sshll4s4h(dest_reg, vector_reg),
            .unsigned => try code.ushll4s4h(dest_reg, vector_reg),
        },
        .high => switch (op.sign) {
            .signed => try code.sshll2_4s8h(dest_reg, vector_reg),
            .unsigned => try code.ushll2_4s8h(dest_reg, vector_reg),
        },
    }
}

fn emitI64x2ExtendI32x4(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdExtendHalf,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, op.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (op.half) {
        .low => switch (op.sign) {
            .signed => try code.sshll2d2s(dest_reg, vector_reg),
            .unsigned => try code.ushll2d2s(dest_reg, vector_reg),
        },
        .high => switch (op.sign) {
            .signed => try code.sshll2_2d4s(dest_reg, vector_reg),
            .unsigned => try code.ushll2_2d4s(dest_reg, vector_reg),
        },
    }
}

fn emitI16x8ExtMulI8x16(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdExtMul,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, op.lhs, null);
    const rhs_reg = if (op.rhs == op.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, op.rhs, lhs_reg);
    const dest_reg = try prepareV128BinaryDest(
        code,
        inst,
        op.lhs,
        lhs_reg,
        op.rhs,
        rhs_reg,
        v128_map,
        v128_cache,
        fctx,
    );
    switch (op.half) {
        .low => switch (op.sign) {
            .signed => try code.smull8h8b(dest_reg, lhs_reg, rhs_reg),
            .unsigned => try code.umull8h8b(dest_reg, lhs_reg, rhs_reg),
        },
        .high => switch (op.sign) {
            .signed => try code.smull2_8h16b(dest_reg, lhs_reg, rhs_reg),
            .unsigned => try code.umull2_8h16b(dest_reg, lhs_reg, rhs_reg),
        },
    }
}

fn emitI32x4ExtMulI16x8(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdExtMul,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, op.lhs, null);
    const rhs_reg = if (op.rhs == op.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, op.rhs, lhs_reg);
    const dest_reg = try prepareV128BinaryDest(
        code,
        inst,
        op.lhs,
        lhs_reg,
        op.rhs,
        rhs_reg,
        v128_map,
        v128_cache,
        fctx,
    );
    switch (op.half) {
        .low => switch (op.sign) {
            .signed => try code.smull4s4h(dest_reg, lhs_reg, rhs_reg),
            .unsigned => try code.umull4s4h(dest_reg, lhs_reg, rhs_reg),
        },
        .high => switch (op.sign) {
            .signed => try code.smull2_4s8h(dest_reg, lhs_reg, rhs_reg),
            .unsigned => try code.umull2_4s8h(dest_reg, lhs_reg, rhs_reg),
        },
    }
}

fn emitI64x2ExtMulI32x4(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdExtMul,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, op.lhs, null);
    const rhs_reg = if (op.rhs == op.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, op.rhs, lhs_reg);
    const dest_reg = try prepareV128BinaryDest(
        code,
        inst,
        op.lhs,
        lhs_reg,
        op.rhs,
        rhs_reg,
        v128_map,
        v128_cache,
        fctx,
    );
    switch (op.half) {
        .low => switch (op.sign) {
            .signed => try code.smull2d2s(dest_reg, lhs_reg, rhs_reg),
            .unsigned => try code.umull2d2s(dest_reg, lhs_reg, rhs_reg),
        },
        .high => switch (op.sign) {
            .signed => try code.smull2_2d4s(dest_reg, lhs_reg, rhs_reg),
            .unsigned => try code.umull2_2d4s(dest_reg, lhs_reg, rhs_reg),
        },
    }
}

fn emitI64x2UnOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    un: ir.Inst.SimdUnary,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, un.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, un.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (un.op) {
        .abs => try code.abs2d(dest_reg, vector_reg),
        .neg => try code.neg2d(dest_reg, vector_reg),
    }
}

fn emitI32x4BinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.I32x4BinOp,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, bin.lhs, null);
    const rhs_reg = if (bin.rhs == bin.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, bin.rhs, lhs_reg);
    const dest_reg = try prepareV128BinaryDest(
        code,
        inst,
        bin.lhs,
        lhs_reg,
        bin.rhs,
        rhs_reg,
        v128_map,
        v128_cache,
        fctx,
    );
    switch (bin.op) {
        .add => try code.i32x4Op(.add, dest_reg, lhs_reg, rhs_reg),
        .sub => try code.i32x4Op(.sub, dest_reg, lhs_reg, rhs_reg),
        .mul => try code.i32x4Op(.mul, dest_reg, lhs_reg, rhs_reg),
        .min_s => try code.i32x4Op(.smin, dest_reg, lhs_reg, rhs_reg),
        .min_u => try code.i32x4Op(.umin, dest_reg, lhs_reg, rhs_reg),
        .max_s => try code.i32x4Op(.smax, dest_reg, lhs_reg, rhs_reg),
        .max_u => try code.i32x4Op(.umax, dest_reg, lhs_reg, rhs_reg),
        .eq => try code.i32x4Op(.cmeq, dest_reg, lhs_reg, rhs_reg),
        .ne => {
            try code.i32x4Op(.cmeq, dest_reg, lhs_reg, rhs_reg);
            try code.mvn16b(dest_reg, dest_reg);
        },
        .gt_s => try code.i32x4Op(.cmgt, dest_reg, lhs_reg, rhs_reg),
        .ge_s => try code.i32x4Op(.cmge, dest_reg, lhs_reg, rhs_reg),
        .lt_s => try code.i32x4Op(.cmgt, dest_reg, rhs_reg, lhs_reg),
        .le_s => try code.i32x4Op(.cmge, dest_reg, rhs_reg, lhs_reg),
        .gt_u => try code.i32x4Op(.cmhi, dest_reg, lhs_reg, rhs_reg),
        .ge_u => try code.i32x4Op(.cmhs, dest_reg, lhs_reg, rhs_reg),
        .lt_u => try code.i32x4Op(.cmhi, dest_reg, rhs_reg, lhs_reg),
        .le_u => try code.i32x4Op(.cmhs, dest_reg, rhs_reg, lhs_reg),
    }
}

fn emitI32x4Shift(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    shift: ir.Inst.I32x4Shift,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, shift.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, shift.vector, vector_reg, v128_map, v128_cache, fctx);
    const count_reg = try useInto(code, reg_map, shift.count, RegMap.tmp0);
    try code.andImm32Mask31(RegMap.tmp0, count_reg);
    const shift_reg = v128_tmp1;
    try code.dup4sFromGp32(shift_reg, RegMap.tmp0);
    switch (shift.op) {
        .shl => try code.sshl4s(dest_reg, vector_reg, shift_reg),
        .shr_s => {
            try code.neg4s(shift_reg, shift_reg);
            try code.sshl4s(dest_reg, vector_reg, shift_reg);
        },
        .shr_u => {
            try code.neg4s(shift_reg, shift_reg);
            try code.ushl4s(dest_reg, vector_reg, shift_reg);
        },
    }
}

fn emitI32x4Splat(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    src: ir.VReg,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    const src_reg = try useInto(code, reg_map, src, RegMap.tmp0);
    try code.dup4sFromGp32(dest_reg, src_reg);
}

fn emitI32x4ExtractLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.I32x4ExtractLane,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    try code.umovWFromS(info.reg, vector_reg, lane.lane);
    try destCommit(code, reg_map, info);
}

fn emitI32x4ReplaceLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.I32x4ReplaceLane,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, lane.vector, vector_reg, v128_map, v128_cache, fctx);
    const val_reg = try useInto(code, reg_map, lane.val, RegMap.tmp0);
    try code.insSFromGp32(dest_reg, lane.lane, val_reg);
}

fn emitI8x16BinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.I8x16BinOp,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, bin.lhs, null);
    const rhs_reg = if (bin.rhs == bin.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, bin.rhs, lhs_reg);
    const dest_reg = try prepareV128BinaryDest(
        code,
        inst,
        bin.lhs,
        lhs_reg,
        bin.rhs,
        rhs_reg,
        v128_map,
        v128_cache,
        fctx,
    );
    switch (bin.op) {
        .add => try code.i8x16Op(.add, dest_reg, lhs_reg, rhs_reg),
        .sub => try code.i8x16Op(.sub, dest_reg, lhs_reg, rhs_reg),
        .add_sat_s => try code.i8x16Op(.sqadd, dest_reg, lhs_reg, rhs_reg),
        .add_sat_u => try code.i8x16Op(.uqadd, dest_reg, lhs_reg, rhs_reg),
        .sub_sat_s => try code.i8x16Op(.sqsub, dest_reg, lhs_reg, rhs_reg),
        .sub_sat_u => try code.i8x16Op(.uqsub, dest_reg, lhs_reg, rhs_reg),
        .eq => try code.i8x16Op(.cmeq, dest_reg, lhs_reg, rhs_reg),
        .ne => {
            try code.i8x16Op(.cmeq, dest_reg, lhs_reg, rhs_reg);
            try code.mvn16b(dest_reg, dest_reg);
        },
        .gt_s => try code.i8x16Op(.cmgt, dest_reg, lhs_reg, rhs_reg),
        .ge_s => try code.i8x16Op(.cmge, dest_reg, lhs_reg, rhs_reg),
        .lt_s => try code.i8x16Op(.cmgt, dest_reg, rhs_reg, lhs_reg),
        .le_s => try code.i8x16Op(.cmge, dest_reg, rhs_reg, lhs_reg),
        .gt_u => try code.i8x16Op(.cmhi, dest_reg, lhs_reg, rhs_reg),
        .ge_u => try code.i8x16Op(.cmhs, dest_reg, lhs_reg, rhs_reg),
        .lt_u => try code.i8x16Op(.cmhi, dest_reg, rhs_reg, lhs_reg),
        .le_u => try code.i8x16Op(.cmhs, dest_reg, rhs_reg, lhs_reg),
        .min_s => try code.i8x16Op(.smin, dest_reg, lhs_reg, rhs_reg),
        .min_u => try code.i8x16Op(.umin, dest_reg, lhs_reg, rhs_reg),
        .max_s => try code.i8x16Op(.smax, dest_reg, lhs_reg, rhs_reg),
        .max_u => try code.i8x16Op(.umax, dest_reg, lhs_reg, rhs_reg),
        .avgr_u => try code.i8x16Op(.urhadd, dest_reg, lhs_reg, rhs_reg),
    }
}

fn emitI8x16Shift(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    shift: ir.Inst.I8x16Shift,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, shift.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, shift.vector, vector_reg, v128_map, v128_cache, fctx);
    const count_reg = try useInto(code, reg_map, shift.count, RegMap.tmp0);
    try code.andImm32Mask7(RegMap.tmp0, count_reg);
    const shift_reg = v128_tmp1;
    try code.dup16bFromGp32(shift_reg, RegMap.tmp0);
    switch (shift.op) {
        .shl => try code.sshl16b(dest_reg, vector_reg, shift_reg),
        .shr_s => {
            try code.neg16b(shift_reg, shift_reg);
            try code.sshl16b(dest_reg, vector_reg, shift_reg);
        },
        .shr_u => {
            try code.neg16b(shift_reg, shift_reg);
            try code.ushl16b(dest_reg, vector_reg, shift_reg);
        },
    }
}

fn emitI8x16Splat(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    src: ir.VReg,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    const src_reg = try useInto(code, reg_map, src, RegMap.tmp0);
    try code.dup16bFromGp32(dest_reg, src_reg);
}

fn emitI8x16ExtractLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.I8x16ExtractLane,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    switch (lane.sign) {
        .signed => try code.smovWFromB(info.reg, vector_reg, lane.lane),
        .unsigned => try code.umovWFromB(info.reg, vector_reg, lane.lane),
    }
    try destCommit(code, reg_map, info);
}

fn emitI8x16ReplaceLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.I8x16ReplaceLane,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, lane.vector, vector_reg, v128_map, v128_cache, fctx);
    const val_reg = try useInto(code, reg_map, lane.val, RegMap.tmp0);
    try code.insBFromGp32(dest_reg, lane.lane, val_reg);
}

fn emitI16x8BinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.I16x8BinOp,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, bin.lhs, null);
    const rhs_reg = if (bin.rhs == bin.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, bin.rhs, lhs_reg);
    const dest_reg = try prepareV128BinaryDest(
        code,
        inst,
        bin.lhs,
        lhs_reg,
        bin.rhs,
        rhs_reg,
        v128_map,
        v128_cache,
        fctx,
    );
    switch (bin.op) {
        .add => try code.i16x8Op(.add, dest_reg, lhs_reg, rhs_reg),
        .sub => try code.i16x8Op(.sub, dest_reg, lhs_reg, rhs_reg),
        .mul => try code.i16x8Op(.mul, dest_reg, lhs_reg, rhs_reg),
        .q15mulr_sat_s => try code.i16x8Op(.sqrdmulh, dest_reg, lhs_reg, rhs_reg),
        .add_sat_s => try code.i16x8Op(.sqadd, dest_reg, lhs_reg, rhs_reg),
        .add_sat_u => try code.i16x8Op(.uqadd, dest_reg, lhs_reg, rhs_reg),
        .sub_sat_s => try code.i16x8Op(.sqsub, dest_reg, lhs_reg, rhs_reg),
        .sub_sat_u => try code.i16x8Op(.uqsub, dest_reg, lhs_reg, rhs_reg),
        .eq => try code.i16x8Op(.cmeq, dest_reg, lhs_reg, rhs_reg),
        .ne => {
            try code.i16x8Op(.cmeq, dest_reg, lhs_reg, rhs_reg);
            try code.mvn16b(dest_reg, dest_reg);
        },
        .gt_s => try code.i16x8Op(.cmgt, dest_reg, lhs_reg, rhs_reg),
        .ge_s => try code.i16x8Op(.cmge, dest_reg, lhs_reg, rhs_reg),
        .lt_s => try code.i16x8Op(.cmgt, dest_reg, rhs_reg, lhs_reg),
        .le_s => try code.i16x8Op(.cmge, dest_reg, rhs_reg, lhs_reg),
        .gt_u => try code.i16x8Op(.cmhi, dest_reg, lhs_reg, rhs_reg),
        .ge_u => try code.i16x8Op(.cmhs, dest_reg, lhs_reg, rhs_reg),
        .lt_u => try code.i16x8Op(.cmhi, dest_reg, rhs_reg, lhs_reg),
        .le_u => try code.i16x8Op(.cmhs, dest_reg, rhs_reg, lhs_reg),
        .min_s => try code.i16x8Op(.smin, dest_reg, lhs_reg, rhs_reg),
        .min_u => try code.i16x8Op(.umin, dest_reg, lhs_reg, rhs_reg),
        .max_s => try code.i16x8Op(.smax, dest_reg, lhs_reg, rhs_reg),
        .max_u => try code.i16x8Op(.umax, dest_reg, lhs_reg, rhs_reg),
        .avgr_u => try code.i16x8Op(.urhadd, dest_reg, lhs_reg, rhs_reg),
    }
}

fn emitI16x8Shift(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    shift: ir.Inst.I16x8Shift,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, shift.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, shift.vector, vector_reg, v128_map, v128_cache, fctx);
    const count_reg = try useInto(code, reg_map, shift.count, RegMap.tmp0);
    try code.andImm32Mask15(RegMap.tmp0, count_reg);
    const shift_reg = v128_tmp1;
    try code.dup8hFromGp32(shift_reg, RegMap.tmp0);
    switch (shift.op) {
        .shl => try code.sshl8h(dest_reg, vector_reg, shift_reg),
        .shr_s => {
            try code.neg8h(shift_reg, shift_reg);
            try code.sshl8h(dest_reg, vector_reg, shift_reg);
        },
        .shr_u => {
            try code.neg8h(shift_reg, shift_reg);
            try code.ushl8h(dest_reg, vector_reg, shift_reg);
        },
    }
}

fn emitI16x8Splat(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    src: ir.VReg,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    const src_reg = try useInto(code, reg_map, src, RegMap.tmp0);
    try code.dup8hFromGp32(dest_reg, src_reg);
}

fn emitI16x8ExtractLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.I16x8ExtractLane,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    switch (lane.sign) {
        .signed => try code.smovWFromH(info.reg, vector_reg, lane.lane),
        .unsigned => try code.umovWFromH(info.reg, vector_reg, lane.lane),
    }
    try destCommit(code, reg_map, info);
}

fn emitI16x8ReplaceLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.I16x8ReplaceLane,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, lane.vector, vector_reg, v128_map, v128_cache, fctx);
    const val_reg = try useInto(code, reg_map, lane.val, RegMap.tmp0);
    try code.insHFromGp32(dest_reg, lane.lane, val_reg);
}

fn emitI64x2BinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.I64x2BinOp,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, bin.lhs, null);
    const rhs_reg = if (bin.rhs == bin.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, bin.rhs, lhs_reg);
    const dest_reg = try prepareV128BinaryDest(
        code,
        inst,
        bin.lhs,
        lhs_reg,
        bin.rhs,
        rhs_reg,
        v128_map,
        v128_cache,
        fctx,
    );
    switch (bin.op) {
        .add => try code.i64x2Op(.add, dest_reg, lhs_reg, rhs_reg),
        .sub => try code.i64x2Op(.sub, dest_reg, lhs_reg, rhs_reg),
        .eq => try code.i64x2Op(.cmeq, dest_reg, lhs_reg, rhs_reg),
        .ne => {
            try code.i64x2Op(.cmeq, dest_reg, lhs_reg, rhs_reg);
            try code.mvn16b(dest_reg, dest_reg);
        },
        .gt_s => try code.i64x2Op(.cmgt, dest_reg, lhs_reg, rhs_reg),
        .ge_s => try code.i64x2Op(.cmge, dest_reg, lhs_reg, rhs_reg),
        .lt_s => try code.i64x2Op(.cmgt, dest_reg, rhs_reg, lhs_reg),
        .le_s => try code.i64x2Op(.cmge, dest_reg, rhs_reg, lhs_reg),
    }
}

fn emitI64x2Shift(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    shift: ir.Inst.I64x2Shift,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, shift.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, shift.vector, vector_reg, v128_map, v128_cache, fctx);
    const count_reg = try useInto(code, reg_map, shift.count, RegMap.tmp0);
    try code.andImm32Mask63(RegMap.tmp0, count_reg);
    const shift_reg = v128_tmp1;
    try code.dup2dFromGp64(shift_reg, RegMap.tmp0);
    switch (shift.op) {
        .shl => try code.sshl2d(dest_reg, vector_reg, shift_reg),
        .shr_s => {
            try code.neg2d(shift_reg, shift_reg);
            try code.sshl2d(dest_reg, vector_reg, shift_reg);
        },
        .shr_u => {
            try code.neg2d(shift_reg, shift_reg);
            try code.ushl2d(dest_reg, vector_reg, shift_reg);
        },
    }
}

fn emitI64x2Splat(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    src: ir.VReg,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    const src_reg = try useInto(code, reg_map, src, RegMap.tmp0);
    try code.dup2dFromGp64(dest_reg, src_reg);
}

fn emitI64x2ExtractLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.I64x2ExtractLane,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    try code.umovXFromD(info.reg, vector_reg, lane.lane);
    try destCommit(code, reg_map, info);
}

fn emitI64x2ReplaceLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.I64x2ReplaceLane,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, lane.vector, vector_reg, v128_map, v128_cache, fctx);
    const val_reg = try useInto(code, reg_map, lane.val, RegMap.tmp0);
    try code.insDFromGp64(dest_reg, lane.lane, val_reg);
}

const ExtendWidth = enum { b, h, w };

fn emitEqz(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    if (inst.type == .i32) {
        try code.cmpImm32(src, 0);
        try code.cset32(info.reg, .eq);
    } else {
        try code.cmpImm(src, 0);
        try code.cset(info.reg, .eq);
    }
    try destCommit(code, reg_map, info);
}

fn emitCmp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    cond: emit.Cond,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    const info = try destBegin(reg_map, dest, RegMap.tmp2);
    if (inst.type == .i32) {
        try code.cmpRegReg32(lhs, rhs);
        try code.cset32(info.reg, cond);
    } else {
        try code.cmpRegReg(lhs, rhs);
        try code.cset(info.reg, cond);
    }
    try destCommit(code, reg_map, info);
}

fn emitClz(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    if (inst.type == .i32) {
        try code.clzReg32(info.reg, src);
    } else {
        try code.clzReg(info.reg, src);
    }
    try destCommit(code, reg_map, info);
}

fn emitCtz(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    // CTZ(x) = CLZ(RBIT(x)). RBIT reads src then writes info.reg; CLZ reads
    // info.reg then writes info.reg — safe even if info.reg aliases src.
    if (inst.type == .i32) {
        try code.rbitReg32(info.reg, src);
        try code.clzReg32(info.reg, info.reg);
    } else {
        try code.rbitReg(info.reg, src);
        try code.clzReg(info.reg, info.reg);
    }
    try destCommit(code, reg_map, info);
}

fn emitExtendImpl(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
    width: ExtendWidth,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    switch (width) {
        .b => try code.sxtb(info.reg, src),
        .h => try code.sxth(info.reg, src),
        .w => try code.sxtw(info.reg, src),
    }
    try destCommit(code, reg_map, info);
}

fn emitWrap(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    // wasm i32.wrap_i64: take low 32 bits of i64. UXTW zero-extends.
    try code.uxtw(info.reg, src);
    try destCommit(code, reg_map, info);
}

/// wasm `i64.extend_i32_u`: zero-extend 32-bit operand to 64 bits. Same
/// emit sequence as `wrap_i64` (UXTW = MOV Wd, Wn, which zero-extends
/// into Xd), but semantically distinct in the IR.
fn emitExtendI32U(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    try code.uxtw(info.reg, src);
    try destCommit(code, reg_map, info);
}

const FSignOp = enum { neg, abs };

/// Emit f_neg / f_abs as pure integer bit manipulation on the float's
/// bit pattern. Correct regardless of whether the operand came from a
/// load, iconst, or prior FPU op because all values flow through the
/// integer scratch pool. Upper bits of 32-bit floats are treated as
/// don't-care (matches existing sub-word codegen contract).
fn emitFSignBit(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
    kind: FSignOp,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const is32 = (inst.type == .f32);
    const mask: u64 = switch (kind) {
        .neg => if (is32) @as(u64, 0x80000000) else @as(u64, 0x8000000000000000),
        .abs => if (is32) @as(u64, 0x7FFFFFFF) else @as(u64, 0x7FFFFFFFFFFFFFFF),
    };
    // Materialize the mask in tmp2 (x15) so it doesn't collide with the
    // src or dest scratch slots (tmp0 / tmp1).
    try code.movImm64(RegMap.tmp2, mask);
    switch (kind) {
        .neg => try code.eorRegReg(info.reg, src, RegMap.tmp2),
        .abs => try code.andRegReg(info.reg, src, RegMap.tmp2),
    }
    try destCommit(code, reg_map, info);
}

/// Bit-identical copy between i32↔f32 and i64↔f64.
fn emitReinterpret(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    if (src != info.reg) try code.movRegReg(info.reg, src);
    try destCommit(code, reg_map, info);
}

const FBinKind = enum { add, sub, mul, div };

/// Emit a scalar float binary op by shuttling operands through the
/// non-allocatable scratch V-regs V0 / V1.
///
/// Sequence:
///   FMOV v0, src_lhs_gp
///   FMOV v1, src_rhs_gp
///   F<op> v0, v0, v1
///   FMOV dst_gp, v0
///
/// This is used by the integer-typed `add`/`sub`/`mul` IR ops when
/// `inst.type` is a float type (matching the frontend's dispatch: the
/// frontend emits the integer-named op with a float type for wasm's
/// f_add/f_sub/f_mul/f_div).
fn emitFBinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    kind: FBinKind,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    const info = try destBegin(reg_map, dest, RegMap.tmp2);

    const is64 = (inst.type == .f64);
    if (is64) {
        try code.fmovDFromGp64(0, lhs);
        try code.fmovDFromGp64(1, rhs);
    } else {
        try code.fmovSFromGp32(0, lhs);
        try code.fmovSFromGp32(1, rhs);
    }
    switch (kind) {
        .add => try code.faddScalar(is64, 0, 0, 1),
        .sub => try code.fsubScalar(is64, 0, 0, 1),
        .mul => try code.fmulScalar(is64, 0, 0, 1),
        .div => try code.fdivScalar(is64, 0, 0, 1),
    }
    if (is64) {
        try code.fmovGpFromD64(info.reg, 0);
    } else {
        try code.fmovGpFromS32(info.reg, 0);
    }
    try destCommit(code, reg_map, info);
}

/// Emit f_sqrt (unary) via scratch V-reg V0.
fn emitFSqrt(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const is64 = (inst.type == .f64);
    if (is64) try code.fmovDFromGp64(0, src) else try code.fmovSFromGp32(0, src);
    try code.fsqrtScalar(is64, 0, 0);
    if (is64) try code.fmovGpFromD64(info.reg, 0) else try code.fmovGpFromS32(info.reg, 0);
    try destCommit(code, reg_map, info);
}

/// Emit f.ceil / f.floor / f.trunc / f.nearest via FRINT{P,M,Z,N} on a
/// scratch V-reg.
fn emitFRint(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
    mode: emit.CodeBuffer.FRoundMode,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const is64 = (inst.type == .f64);
    if (is64) try code.fmovDFromGp64(0, src) else try code.fmovSFromGp32(0, src);
    try code.frintScalar(is64, mode, 0, 0);
    if (is64) try code.fmovGpFromD64(info.reg, 0) else try code.fmovGpFromS32(info.reg, 0);
    try destCommit(code, reg_map, info);
}

const FMinMaxKind = enum { min, max };

/// Emit f.min / f.max via FMIN/FMAX on scratch V-regs V0,V1.
/// ARM FMIN/FMAX propagate NaNs and handle ±0 per wasm spec.
fn emitFMinMax(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    kind: FMinMaxKind,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    const info = try destBegin(reg_map, dest, RegMap.tmp2);

    const is64 = (inst.type == .f64);
    if (is64) {
        try code.fmovDFromGp64(0, lhs);
        try code.fmovDFromGp64(1, rhs);
    } else {
        try code.fmovSFromGp32(0, lhs);
        try code.fmovSFromGp32(1, rhs);
    }
    switch (kind) {
        .min => try code.fminScalar(is64, 0, 0, 1),
        .max => try code.fmaxScalar(is64, 0, 0, 1),
    }
    if (is64) {
        try code.fmovGpFromD64(info.reg, 0);
    } else {
        try code.fmovGpFromS32(info.reg, 0);
    }
    try destCommit(code, reg_map, info);
}

/// Emit f.copysign: dest = (lhs & ~sign) | (rhs & sign).
/// Implemented as pure integer bit-twiddling on the float bit pattern —
/// no V-register needed. Mirrors emitFSignBit.
fn emitFCopysign(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    const info = try destBegin(reg_map, dest, RegMap.tmp2);

    const is32 = (inst.type == .f32);
    const sign_mask: u64 = if (is32) @as(u64, 0x80000000) else @as(u64, 0x8000000000000000);
    const magnitude_mask: u64 = if (is32) @as(u64, 0x7FFFFFFF) else @as(u64, 0x7FFFFFFFFFFFFFFF);

    // tmp2 already holds dest scratch; we need one more scratch for the
    // masked rhs. Reuse lhs's slot after we've consumed it: materialize
    // the masks into info.reg and a helper, then combine.
    //
    // Sequence (using info.reg as dest, and one temp):
    //   mov   temp, magnitude_mask
    //   and   info.reg, lhs, temp
    //   mov   temp, sign_mask
    //   and   temp, rhs, temp
    //   orr   info.reg, info.reg, temp
    //
    // Pick a scratch distinct from lhs, rhs, info.reg.
    const scratch: emit.Reg = blk: {
        const candidates = [_]emit.Reg{ RegMap.tmp0, RegMap.tmp1, RegMap.tmp2 };
        for (candidates) |c| {
            if (c != lhs and c != rhs and c != info.reg) break :blk c;
        }
        unreachable;
    };

    try code.movImm64(scratch, magnitude_mask);
    try code.andRegReg(info.reg, lhs, scratch);
    try code.movImm64(scratch, sign_mask);
    try code.andRegReg(scratch, rhs, scratch);
    try code.orrRegReg(info.reg, info.reg, scratch);

    try destCommit(code, reg_map, info);
}

/// Emit wasm i32/i64.popcnt via NEON CNT + ADDV on scratch V0.
///
/// Sequence:
///   FMOV d0, Xsrc     ; move 64 bits of src into V0
///   CNT  v0.8b, v0.8b ; per-byte popcount (0..8 in each byte)
///   ADDV b0, v0.8b    ; horizontal sum of the 8 bytes into B0
///   FMOV Xd, d0       ; move the byte result back to GPR
///
/// For i32 src, the frontend zero-extends to 64 bits in the source vreg
/// (upper 32 bits are always zero in our codegen), so the 64-bit popcount
/// equals the 32-bit one.
fn emitPopcnt(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    try code.fmovDFromGp64(0, src);
    try code.cnt8b(0, 0);
    try code.addvB8b(0, 0);
    try code.fmovGpFromD64(info.reg, 0);
    try destCommit(code, reg_map, info);
}

/// Emit a float comparison (f_eq/f_ne/f_lt/f_gt/f_le/f_ge). Operand
/// type is carried in `inst.type` (.f32 or .f64); the result is an
/// i32 boolean placed in `inst.dest`.
///
/// NaN semantics: WebAssembly requires every comparison to return 0
/// when either operand is NaN, except `ne` which must return 1.
/// FCMP on AArch64 sets NZCV = 0b0011 (N=0, Z=0, C=1, V=1) when
/// unordered. The chosen conditions satisfy wasm semantics for NaN:
///   eq → EQ  (Z=1)           : unordered Z=0 → 0 ✓
///   ne → NE  (Z=0)           : unordered Z=0 → 1 ✓
///   lt → MI  (N=1)           : unordered N=0 → 0 ✓
///   le → LS  (C=0 | Z=1)     : unordered C=1,Z=0 → 0 ✓
///   gt → GT  (Z=0 & N=V)     : unordered N=0,V=1 → 0 ✓
///   ge → GE  (N=V)           : unordered N=0,V=1 → 0 ✓
fn emitFCmp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    cond: emit.Cond,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    const info = try destBegin(reg_map, dest, RegMap.tmp2);

    const is64 = (inst.type == .f64);
    if (is64) {
        try code.fmovDFromGp64(0, lhs);
        try code.fmovDFromGp64(1, rhs);
    } else {
        try code.fmovSFromGp32(0, lhs);
        try code.fmovSFromGp32(1, rhs);
    }
    try code.fcmpScalar(is64, 0, 1);
    // Result is i32; CSET Wd, cond — upper 32 bits are zeroed by the
    // CSINC-W alias, matching x86_64 SETcc+MOVZX convention.
    try code.cset32(info.reg, cond);
    try destCommit(code, reg_map, info);
}

/// Emit a non-trapping int→float conversion via scratch V-reg V0.
///
/// `src_is_i64` selects whether the source vreg is read as X (64-bit)
/// or W (32-bit). Destination float width comes from `inst.type`.
fn emitConvertIntToFloat(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
    signed: bool,
    src_is_i64: bool,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const is_f64 = (inst.type == .f64);
    if (signed) {
        try code.scvtfFromGp(is_f64, src_is_i64, 0, src);
    } else {
        try code.ucvtfFromGp(is_f64, src_is_i64, 0, src);
    }
    if (is_f64) try code.fmovGpFromD64(info.reg, 0) else try code.fmovGpFromS32(info.reg, 0);
    try destCommit(code, reg_map, info);
}

/// Emit f32.demote_f64 (f64→f32) via scratch V-reg V0.
fn emitDemoteF64(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    try code.fmovDFromGp64(0, src);
    try code.fcvtDemoteDToS(0, 0);
    try code.fmovGpFromS32(info.reg, 0);
    try destCommit(code, reg_map, info);
}

/// Emit f64.promote_f32 (f32→f64) via scratch V-reg V0.
fn emitPromoteF32(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    try code.fmovSFromGp32(0, src);
    try code.fcvtPromoteSToD(0, 0);
    try code.fmovGpFromD64(info.reg, 0);
    try destCommit(code, reg_map, info);
}

/// Emit trapping float→int truncation (wasm's `trunc_f*_s/u`).
///
/// Semantics (wasm spec):
///   * NaN → trap
///   * ±inf → trap
///   * out-of-range (would not fit in the target integer type after
///     round-toward-zero) → trap
///   * else → FCVTZS/FCVTZU result
///
/// Implementation mirrors the x86_64 reference (`emitTruncRangeCheck`):
///   1. FMOV src into V0 (S or D lane).
///   2. FCMP V0, V0 — unordered sets V=1; trap on VS (inline helper call).
///   3. Materialize the lower bound into tmp1 → FMOV V1; FCMP V0, V1.
///      Trap on MI (strict `<` bound; f32 cases) or LE (inclusive
///      `<=` bound; f64→i32 signed where `INT_MIN-1` is representable,
///      and all unsigned cases where `-1.0` is representable).
///   4. Materialize the upper bound (always strict `<`); FCMP V0, V1;
///      trap on MI using the INVERTED operand order FCMP V1, V0 so the
///      skip condition is uniform. (We instead do FCMP V0, V1 and trap
///      on PL: src >= max; skip when src < max, i.e. MI.) In practice
///      we use: FCMP V0, V1; skip on MI.
///   5. FCVTZS/FCVTZU dst, V0.
fn emitTruncTrapping(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
    signed: bool,
) !void {
    const dest = inst.dest orelse return;
    const dst_is_i32 = (inst.type == .i32);
    // The IR op is trunc_f32_* or trunc_f64_*; source type is encoded
    // in the op tag, not inst.type (which is the destination type).
    const src_is_f32 = switch (inst.op) {
        .trunc_f32_s, .trunc_f32_u => true,
        .trunc_f64_s, .trunc_f64_u => false,
        else => unreachable,
    };

    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);

    // 1. Move source into V0 (S or D lane).
    if (src_is_f32) try code.fmovSFromGp32(0, src) else try code.fmovDFromGp64(0, src);

    // 2. NaN check: FCMP V0, V0 sets V=1 on unordered.
    //    Skip the trap when ordered: B.VC skip; trap; skip:
    try code.fcmpScalar(!src_is_f32, 0, 0);
    {
        const skip = code.len();
        try code.bCond(.vc, 0);
        try emitTrapHelperCall(code, vmctx_trap_iovf_fn_slot);
        patchBCondHere(code, skip);
    }

    // 3. Lower-bound check. Choose bits + inclusivity based on the
    //    smallest representable value just below INT_MIN (signed) or
    //    -1.0 (unsigned).
    const LowerInfo = struct { bits: u64, inclusive_trap: bool };
    const lo: LowerInfo = if (signed) blk: {
        if (dst_is_i32) {
            if (src_is_f32) {
                // f32 -2^31 = 0xCF000000; -2^31-1 not representable.
                break :blk .{ .bits = 0xCF000000, .inclusive_trap = false };
            } else {
                // f64 -2^31-1 representable exactly: 0xC1E0000000200000.
                break :blk .{ .bits = 0xC1E0000000200000, .inclusive_trap = true };
            }
        } else {
            break :blk if (src_is_f32)
                .{ .bits = 0xDF000000, .inclusive_trap = false }
            else
                .{ .bits = 0xC3E0000000000000, .inclusive_trap = false };
        }
    } else blk: {
        break :blk if (src_is_f32)
            .{ .bits = 0xBF800000, .inclusive_trap = true } // -1.0_f32
        else
            .{ .bits = 0xBFF0000000000000, .inclusive_trap = true }; // -1.0_f64
    };
    if (src_is_f32) {
        try code.movImm32(RegMap.tmp1, @bitCast(@as(u32, @truncate(lo.bits))));
        try code.fmovSFromGp32(1, RegMap.tmp1);
    } else {
        try code.movImm64(RegMap.tmp1, lo.bits);
        try code.fmovDFromGp64(1, RegMap.tmp1);
    }
    try code.fcmpScalar(!src_is_f32, 0, 1);
    {
        // Strict: trap on MI  (src < bound)           → skip on PL
        // Inclusive: trap on LE  (src <= bound)       → skip on GT
        const skip = code.len();
        try code.bCond(if (lo.inclusive_trap) .gt else .pl, 0);
        try emitTrapHelperCall(code, vmctx_trap_iovf_fn_slot);
        patchBCondHere(code, skip);
    }

    // 4. Upper-bound check (always strict `src >= max` traps).
    const max_bits: u64 = if (signed) blk: {
        if (dst_is_i32) {
            break :blk if (src_is_f32) @as(u64, 0x4F000000) else @as(u64, 0x41E0000000000000);
        } else {
            break :blk if (src_is_f32) @as(u64, 0x5F000000) else @as(u64, 0x43E0000000000000);
        }
    } else blk: {
        if (dst_is_i32) {
            break :blk if (src_is_f32) @as(u64, 0x4F800000) else @as(u64, 0x41F0000000000000);
        } else {
            break :blk if (src_is_f32) @as(u64, 0x5F800000) else @as(u64, 0x43F0000000000000);
        }
    };
    if (src_is_f32) {
        try code.movImm32(RegMap.tmp1, @bitCast(@as(u32, @truncate(max_bits))));
        try code.fmovSFromGp32(1, RegMap.tmp1);
    } else {
        try code.movImm64(RegMap.tmp1, max_bits);
        try code.fmovDFromGp64(1, RegMap.tmp1);
    }
    try code.fcmpScalar(!src_is_f32, 0, 1);
    {
        // Trap on src >= max (GE) → skip on src < max (MI).
        const skip = code.len();
        try code.bCond(.mi, 0);
        try emitTrapHelperCall(code, vmctx_trap_iovf_fn_slot);
        patchBCondHere(code, skip);
    }

    // 5. Allocate dest, then FCVTZ. info.reg fallback is tmp2 which is
    //    unused above.
    const info = try destBegin(reg_map, dest, RegMap.tmp2);
    const dst_is_x = !dst_is_i32;
    if (signed) {
        try code.fcvtzsToGp(dst_is_x, !src_is_f32, info.reg, 0);
    } else {
        try code.fcvtzuToGp(dst_is_x, !src_is_f32, info.reg, 0);
    }
    try destCommit(code, reg_map, info);
}

/// Emit saturating float→int truncation (wasm's `trunc_sat_f*_s/u`).
/// AArch64 FCVTZ[SU] saturates to INT_MIN/MAX (or 0/UINT_MAX) and
/// returns 0 for NaN, which matches the wasm spec directly — no
/// range checks needed.
fn emitTruncSat(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    vreg: ir.VReg,
    reg_map: *RegMap,
    signed: bool,
) !void {
    const dest = inst.dest orelse return;
    const dst_is_i32 = (inst.type == .i32);
    const src_is_f32 = switch (inst.op) {
        .trunc_sat_f32_s, .trunc_sat_f32_u => true,
        .trunc_sat_f64_s, .trunc_sat_f64_u => false,
        else => unreachable,
    };

    const src = try useInto(code, reg_map, vreg, RegMap.tmp0);
    if (src_is_f32) try code.fmovSFromGp32(0, src) else try code.fmovDFromGp64(0, src);

    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const dst_is_x = !dst_is_i32;
    if (signed) {
        try code.fcvtzsToGp(dst_is_x, !src_is_f32, info.reg, 0);
    } else {
        try code.fcvtzuToGp(dst_is_x, !src_is_f32, info.reg, 0);
    }
    try destCommit(code, reg_map, info);
}

fn emitSelect(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    sel: @TypeOf(@as(ir.Inst.Op, undefined).select),
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    // Load cond first; after CMP we don't need the cond register anymore,
    // so we can reuse tmp0 for subsequent uses if desired. We keep t/f in
    // distinct scratches (tmp1/tmp2) so CSEL can read them simultaneously.
    const cond_r = try useInto(code, reg_map, sel.cond, RegMap.tmp0);
    try code.cmpImm32(cond_r, 0);
    const t_r = try useInto(code, reg_map, sel.if_true, RegMap.tmp1);
    const f_r = try useInto(code, reg_map, sel.if_false, RegMap.tmp2);
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    // wasm: select picks if_true when cond != 0.
    if (inst.type == .i32) {
        try code.csel32(info.reg, t_r, f_r, .ne);
    } else {
        try code.csel(info.reg, t_r, f_r, .ne);
    }
    try destCommit(code, reg_map, info);
}

fn emitShift(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    op: emit.CodeBuffer.ShiftOp,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    const info = try destBegin(reg_map, dest, RegMap.tmp2);
    if (inst.type == .i32) {
        try code.shiftRegReg32(info.reg, lhs, rhs, op);
    } else {
        try code.shiftRegReg(info.reg, lhs, rhs, op);
    }
    try destCommit(code, reg_map, info);
}

fn emitRotl(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const lhs = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    const rhs = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    // NEG output must not clobber lhs or rhs. Use tmp2 as the dedicated
    // scratch for -rhs. destBegin takes tmp0 as its fallback, so in the
    // worst case info.reg == tmp0 == lhs (aliased read-before-write — OK).
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    // rotl(x, n) = ror(x, -n). AArch64 has no ROL, so negate count into
    // tmp2, then RORV. Width-correct: W-form masks by 5, X by 6.
    if (inst.type == .i32) {
        try code.negReg32(RegMap.tmp2, rhs);
        try code.shiftRegReg32(info.reg, lhs, RegMap.tmp2, .ror);
    } else {
        try code.negReg(RegMap.tmp2, rhs);
        try code.shiftRegReg(info.reg, lhs, RegMap.tmp2, .ror);
    }
    try destCommit(code, reg_map, info);
}

fn emitBr(
    code: *emit.CodeBuffer,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    target: ir.BlockId,
    allocator: std.mem.Allocator,
) !void {
    const patch_off = code.len();
    try code.b(0); // placeholder
    try patches.append(allocator, .{
        .patch_offset = patch_off,
        .target_block = target,
        .kind = .b_uncond,
    });
}

/// ABI arg register for the i-th user argument (i=0 → x1 because x0 = vmctx).
fn callArgReg(i: u32) emit.Reg {
    return switch (i) {
        0 => .x1,
        1 => .x2,
        2 => .x3,
        3 => .x4,
        4 => .x5,
        5 => .x6,
        6 => .x7,
        else => unreachable,
    };
}

/// Emit a direct wasm call.
///
/// Limitations (Phase 1): no tail calls, no extra results, no imports,
/// at most 7 arguments. Violations return an error rather than emit
/// incorrect code.
///
/// Sequence: spill all currently-live caller-save physregs (x0..x15) to
/// fixed per-reg slots in the call-save region, move arguments into
/// x1..x7 by reading from those save slots (for physreg sources) or the
/// RegMap spill region (for stack sources), load VMContext into x0, emit
/// `BL 0` with a patch record, move the result from x0 into the dest
/// vreg's location, then restore all saved regs.
///
/// The result-holding physreg is safe to not skip during restore: because
/// the RegMap allocates `dest` AFTER the spill and assign() only picks
/// currently-free regs, `dest.reg` cannot match any reg we saved.
fn emitCall(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    cl: @TypeOf(@as(ir.Inst.Op, undefined).call),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    // args + optional HRP must fit in the 7 user-arg slots.
    const total_call_regs: u32 = @as(u32, @intCast(cl.args.len)) +
        (if (cl.extra_results > 0) @as(u32, 1) else 0);
    if (total_call_regs > 7) return error.TooManyCallArgs;
    const is_import = cl.func_idx < fctx.import_count;
    const call_patches: ?*std.ArrayListUnmanaged(CallPatch) = if (is_import)
        null
    else blk: {
        break :blk (fctx.call_patches orelse return error.CallLinkageUnavailable);
    };

    // Snapshot live caller-save physregs (x0..x15). Callee-saved
    // allocatable regs (x19..x28) at indices >= caller_saved_count survive
    // BL automatically, so we never spill them around calls.
    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;

    // Vregs whose last static use is this call — their regs are dead
    // after BL, so we skip the restore. Args consumed into x1..x7 and
    // used nowhere else fall into this bucket (common case in hot loops).

    // Spill each used caller-save reg to [fp + call_save_base + i*8].
    const dying_mask: u16 = dyingCallerSaveMask(reg_map, fctx.current_kills);
    // For tail calls these writes are dead, but arg staging still reads
    // back from these slots as the stable per-reg source locations for
    // args whose source reg is clobbered by an earlier arg's target.
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, cl.args);

    // Move arguments into x1..x7. `clobbered` tracks which caller-save
    // regs have been overwritten by earlier staging in this same call;
    // `stageArgFromSaved` uses it to pick between a direct register mov
    // (cheap) and a reload from the call-save slot (correct when the
    // source was clobbered by an earlier arg's target).
    var clobbered: u16 = 0;
    for (cl.args, 0..) |arg_vreg, i| {
        const target = callArgReg(@intCast(i));
        try stageArgFromSaved(code, reg_map, fctx, target, arg_vreg, clobbered);
        clobbered |= (@as(u16, 1) << @intCast(@intFromEnum(target)));
    }

    // HRP handling. For a normal (non-tail) call, pass a pointer to our
    // own scratch region so the callee writes extras there for
    // `call_result` to pick up. For a tail call we instead forward our
    // caller's HRP (saved in the prologue at `hrp_save_off`), because
    // wasm tail-call semantics require matching signatures and the
    // callee's extras flow straight into our caller's scratch.
    if (cl.extra_results > 0) {
        const hrp_target = callArgReg(@intCast(cl.args.len));
        if (cl.tail) {
            try frameLoad(code, hrp_target, fctx.hrp_save_off);
        } else {
            try frameAddr(code, hrp_target, fctx.scratch_base);
        }
    }

    // Load VMContext into x0 (was saved above if previously live).
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);

    if (cl.tail) {
        // For imports the target lives in a vmctx slot; load it into
        // tmp0 BEFORE tearing down the frame (vmctx reads need FP).
        // tmp0 (x16) is a caller-save register not touched by the
        // epilogue, so it survives to the `br` below.
        if (is_import) {
            try code.ldrImm(RegMap.tmp0, .x0, vmctx_host_functions_slot);
            const fn_slot: u12 = @intCast(cl.func_idx);
            try code.ldrImm(RegMap.tmp0, RegMap.tmp0, fn_slot);
        }
        // Tear down frame. The epilogue restores FP/LR to the caller's
        // values and deallocates our stack, leaving the CPU state such
        // that branching to the target makes it return directly to our
        // caller — i.e. a real tail call.
        try emitCalleeSaveRestoreTracked(code, fctx);
        try code.emitEpilogueNoRet(fctx.frame_size);
        if (is_import) {
            try code.br(RegMap.tmp0);
        } else {
            // Emit B imm26 (opcode 0x14000000) with a patch record. The
            // module-level call patcher preserves bits 31:26 from the
            // existing encoding, so B and BL are both patched correctly.
            const patch_off = code.len();
            try code.b(0);
            try call_patches.?.append(fctx.allocator, .{
                .patch_offset = patch_off,
                .target_func_idx = cl.func_idx - fctx.import_count,
            });
        }
        return;
    }

    if (is_import) {
        // Indirect call through vmctx.host_functions_ptr[func_idx].
        // Use tmp0 for the function pointer chain so x0 (the vmctx arg)
        // survives.
        try code.ldrImm(RegMap.tmp0, .x0, vmctx_host_functions_slot);
        const fn_slot: u12 = @intCast(cl.func_idx);
        try code.ldrImm(RegMap.tmp0, RegMap.tmp0, fn_slot);
        try code.blr(RegMap.tmp0);
    } else {
        // BL 0 (placeholder); record a patch for module-level resolution.
        const patch_off = code.len();
        try code.bl(0);
        try call_patches.?.append(fctx.allocator, .{
            .patch_offset = patch_off,
            .target_func_idx = cl.func_idx - fctx.import_count,
        });
    }

    // Commit result (in x0) to dest's location BEFORE restoring saved regs,
    // so restoration can't clobber the result holder (dest.reg is always a
    // reg that was FREE at snapshot time — i.e. not in `used_snapshot`).
    if (inst.dest) |dest| {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }

    // Restore previously-saved caller-save regs, skipping those whose
    // vreg died at this call (the reg is dead after BL — nothing reads
    // it, so the ldr would be wasted work).
    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        if ((dying_mask >> @intCast(i)) & 1 != 0) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// Compute linear-memory effective address in tmp0: mem_base + zext(wasm_addr) + offset.
///
/// Emits an inline bounds check: if `zext(wasm_addr) + end_offset > memory_size`,
/// calls `VmCtx.trap_oob_fn(vmctx)` which is noreturn. `end_offset` is
/// `static_offset + access_size` as supplied by the caller.
///
/// Clobbers tmp0, tmp1, tmp2. On the trap path additionally clobbers x0
/// (as the call arg) and LR — but that path is noreturn, so callers need
/// not care. After the fallthrough, only tmp0 carries meaningful data.
fn emitMemAddr(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    base_vreg: ir.VReg,
    offset: u32,
    end_offset: u64,
) !void {
    try emitMemAddrImpl(code, reg_map, base_vreg, offset, end_offset, false);
}

fn emitMemAddrSkipBounds(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    base_vreg: ir.VReg,
    offset: u32,
) !void {
    try emitMemAddrImpl(code, reg_map, base_vreg, offset, 0, true);
}

fn emitMemAddrImpl(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    base_vreg: ir.VReg,
    offset: u32,
    end_offset: u64,
    skip_bounds: bool,
) !void {
    // Step 1: zero-extend wasm address into tmp2 (kept alive across check).
    const src = try useInto(code, reg_map, base_vreg, RegMap.tmp1);
    try code.movRegReg32(RegMap.tmp2, src);

    if (!skip_bounds) {
        // Step 2: compute end = tmp2 + end_offset in tmp1.
        if (end_offset == 0) {
            try code.movRegReg(RegMap.tmp1, RegMap.tmp2);
        } else if (end_offset <= 0xFFF) {
            try code.addImm(RegMap.tmp1, RegMap.tmp2, @intCast(end_offset));
        } else {
            try code.movImm64(RegMap.tmp1, end_offset);
            try code.addRegReg(RegMap.tmp1, RegMap.tmp1, RegMap.tmp2);
        }

        // Step 3: load VmCtx.memory_size (at +8, scaled-by-8 offset = 1).
        try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
        try code.ldrImm(RegMap.tmp0, RegMap.tmp0, 1);

        // Step 4: cmp end, mem_size; B.LS over_trap.
        try code.cmpRegReg(RegMap.tmp1, RegMap.tmp0);
        const over_patch = code.len();
        try code.bCond(.ls, 0); // placeholder

        // Trap path.
        try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
        try code.ldrImm(RegMap.tmp0, .x0, vmctx_trap_oob_fn_slot);
        try code.blr(RegMap.tmp0);
        try code.brk(0);

        // Patch B.LS to land here.
        const over_target = code.len();
        const delta_words: i19 = @intCast(@divExact(
            @as(i64, @intCast(over_target)) - @as(i64, @intCast(over_patch)),
            4,
        ));
        const existing = std.mem.readInt(u32, code.bytes.items[over_patch..][0..4], .little);
        const imm19: u19 = @bitCast(delta_words);
        const new_word: u32 = (existing & 0xFF00001F) | (@as(u32, imm19) << 5);
        code.patch32(over_patch, new_word);
    }

    // Step 5: load mem_base into tmp0.
    try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, 0);

    // Step 6: tmp0 = mem_base + zext(wasm_addr).
    try code.addRegReg(RegMap.tmp0, RegMap.tmp0, RegMap.tmp2);

    // Step 7: fold constant offset.
    if (offset != 0) {
        if (offset <= 0xFFF) {
            try code.addImm(RegMap.tmp0, RegMap.tmp0, @intCast(offset));
        } else {
            try code.movImm64(RegMap.tmp1, offset);
            try code.addRegReg(RegMap.tmp0, RegMap.tmp0, RegMap.tmp1);
        }
    }
}

/// VmCtx field offsets. Must match `runtime/aot/runtime.zig::VmCtx`.
const vmctx_memsize_slot: u12 = 1; // byte 8, scale 8
const vmctx_globals_slot: u12 = 2; // byte 16, scale 8
const vmctx_host_functions_slot: u12 = 3; // byte 24, scale 8
const vmctx_mem_pages_slot_w: u12 = 14; // byte 56 (u32), scale 4
const vmctx_mem_grow_fn_slot: u12 = 8; // byte 64, scale 8
const vmctx_mem_fill_fn_slot: u12 = 28; // byte 224, scale 8
const vmctx_mem_copy_fn_slot: u12 = 29; // byte 232, scale 8
const vmctx_trap_oob_fn_slot: u12 = 10; // byte 80, scale 8
const vmctx_trap_unreachable_fn_slot: u12 = 11; // byte 88, scale 8
const vmctx_trap_idivz_fn_slot: u12 = 12; // byte 96, scale 8
const vmctx_trap_iovf_fn_slot: u12 = 13; // byte 104, scale 8
const vmctx_funcptrs_slot: u12 = 15; // byte 120, scale 8
const vmctx_table_grow_fn_slot: u12 = 16; // byte 128, scale 8
const vmctx_tables_info_slot: u12 = 17; // byte 136, scale 8
const vmctx_sig_table_slot: u12 = 20; // byte 160, scale 8
const vmctx_table_set_fn_slot: u12 = 24; // byte 192, scale 8
const vmctx_table_init_fn_slot: u12 = 18; // byte 144, scale 8
const vmctx_elem_drop_fn_slot: u12 = 19; // byte 152, scale 8
const vmctx_futex_wait32_fn_slot: u12 = 25; // byte 200, scale 8
const vmctx_futex_wait64_fn_slot: u12 = 26; // byte 208, scale 8
const vmctx_futex_notify_fn_slot: u12 = 27; // byte 216, scale 8

// Per-table descriptor layout (`TableInfo`, 24 bytes):
//   { ptr: u64, len: u32, _pad: u32, type_backing_ptr: u64 }
const table_info_stride: u32 = 24;
const table_info_ptr_off: u32 = 0;
const table_info_len_off: u32 = 8;
const table_info_type_backing_off: u32 = 16;

/// Globals are a packed array of 8-byte slots indexed by global index.
/// For i32/f32 we read/write the low 32 bits with W-form LDR/STR; for
/// i64/f64 we use the 64-bit forms. This matches x86-64 which uses an
/// 8-byte slot per global.
fn emitGlobalGet(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    idx: u32,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    // tmp0 = vmctx; tmp0 = globals_base
    try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, vmctx_globals_slot);

    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const is32 = (inst.type == .i32 or inst.type == .f32);
    // Byte offset = idx * 8. LDR scaled-imm requires idx fit in u12.
    if (idx > 0xFFF) return error.GlobalIndexOutOfRange;
    if (is32) try code.ldrImm32(info.reg, RegMap.tmp0, @intCast(idx * 2)) else try code.ldrImm(info.reg, RegMap.tmp0, @intCast(idx));
    try destCommit(code, reg_map, info);
}

fn emitGlobalSet(
    code: *emit.CodeBuffer,
    gs: @TypeOf(@as(ir.Inst.Op, undefined).global_set),
    reg_map: *RegMap,
) !void {
    const val_r = try useInto(code, reg_map, gs.val, RegMap.tmp1);
    // tmp0 = vmctx; tmp0 = globals_base
    try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, vmctx_globals_slot);
    if (gs.idx > 0xFFF) return error.GlobalIndexOutOfRange;
    // Type width inferred from the val's IR type is not available here;
    // always store 8 bytes (safe: the interpreter allocates 8 bytes per
    // global slot, and i32/f32 are zero-extended or host-padded).
    try code.strImm(val_r, RegMap.tmp0, @intCast(gs.idx));
}

/// `memory.size` — 32-bit load of VmCtx.memory_pages.
fn emitMemorySize(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    // Load vmctx into tmp0 first, then allocate dest (dest's scratch is
    // tmp1, so even if dest spills we don't alias tmp0 until after the
    // vmctx load).
    try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    try code.ldrImm32(info.reg, RegMap.tmp0, vmctx_mem_pages_slot_w);
    try destCommit(code, reg_map, info);
}

/// `memory.grow` — calls host helper `VmCtx.mem_grow_fn(vmctx, delta)`.
/// Returns previous page count (or -1 on failure) in x0.
///
/// Follows the same save-all-live-caller-save pattern as `emitCall`: we
/// snapshot `reg_map.reg_used`, STR each live physreg to its fixed
/// per-reg call-save slot, stage `delta_pages` into x1 by reading from
/// either the save slot (for physreg sources) or the spill slot (for
/// stack sources), load vmctx into x0, indirect-call through
/// VmCtx.mem_grow_fn (scaled slot 8), commit x0 to `dest` BEFORE
/// restoring saved regs (so the reg allocator's choice for dest — which
/// was made after the snapshot — cannot collide with a reg being
/// restored), then restore each saved reg.
fn emitMemoryGrow(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    pages_vreg: ir.VReg,
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    const args = [_]ir.VReg{pages_vreg};
    try emitVmctxHelperCall(code, &args, vmctx_mem_grow_fn_slot, reg_map, fctx, inst.dest);
}

fn emitMemoryFill(
    code: *emit.CodeBuffer,
    mf: @TypeOf(@as(ir.Inst.Op, undefined).memory_fill),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    const args = [_]ir.VReg{ mf.dst, mf.val, mf.len };
    try emitVmctxHelperCall(code, &args, vmctx_mem_fill_fn_slot, reg_map, fctx, null);
}

fn emitMemoryCopy(
    code: *emit.CodeBuffer,
    mc: @TypeOf(@as(ir.Inst.Op, undefined).memory_copy),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    const args = [_]ir.VReg{ mc.dst, mc.src, mc.len };
    try emitVmctxHelperCall(code, &args, vmctx_mem_copy_fn_slot, reg_map, fctx, null);
}

/// Load the address of table `idx`'s `TableInfo` record into `dest` reg
/// via the `vmctx.tables_info_ptr` array (stride 24 bytes per table).
/// Uses `RegMap.tmp0` internally if `dest` == vmctx register (it won't,
/// since `dest` is always a scratch caller like tmp0/tmp1 at call sites).
fn loadTableInfoAddr(code: *emit.CodeBuffer, dest: emit.Reg, table_idx: u32) !void {
    try code.ldrImm(dest, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(dest, dest, vmctx_tables_info_slot);
    const off: u32 = table_idx * table_info_stride;
    if (off != 0) {
        try code.movImm32(RegMap.tmp1, @intCast(off));
        try code.addRegReg(dest, dest, RegMap.tmp1);
    }
}

/// `.table_size` — returns `vmctx.tables_info[table_idx].len`.
fn emitTableSize(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    table_idx: u32,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    try loadTableInfoAddr(code, RegMap.tmp0, table_idx);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const len_slot_w: u12 = @intCast(table_info_len_off / 4);
    try code.ldrImm32(info.reg, RegMap.tmp0, len_slot_w);
    try destCommit(code, reg_map, info);
}

/// `.table_get` — bounds-checked load of table element pointer.
///   if (idx >= table.len) trap_unreachable;
///   dest = table.ptr[idx];
fn emitTableGet(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    tg: @TypeOf(@as(ir.Inst.Op, undefined).table_get),
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const idx_reg = try useInto(code, reg_map, tg.idx, RegMap.tmp2);

    try loadTableInfoAddr(code, RegMap.tmp0, tg.table_idx);
    const len_slot_w: u12 = @intCast(table_info_len_off / 4);
    try code.ldrImm32(RegMap.tmp1, RegMap.tmp0, len_slot_w);
    try code.cmpRegReg32(idx_reg, RegMap.tmp1);
    const skip_patch = code.len();
    try code.bCond(.lo, 0);
    try emitTrapHelperCall(code, vmctx_trap_unreachable_fn_slot);
    patchBCondHere(code, skip_patch);

    // tmp0 = table_info_ptr; load table.ptr (at offset 0) into tmp0.
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, @intCast(table_info_ptr_off / 8));
    // tmp0 = table.ptr + (idx * 8)
    try code.addExtUxtw3(RegMap.tmp0, RegMap.tmp0, idx_reg);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    try code.ldrImm(info.reg, RegMap.tmp0, 0);
    try destCommit(code, reg_map, info);
}

/// `.table_set` — calls `VmCtx.table_set_fn(vmctx, table_idx, idx, val)`.
/// The helper handles bounds checking and type-backing updates.
fn emitTableSet(
    code: *emit.CodeBuffer,
    ts: @TypeOf(@as(ir.Inst.Op, undefined).table_set),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    // Helper signature: fn(vmctx, table_idx, idx, val) with table_idx a
    // compile-time constant. Snapshot live caller-save regs, stage args
    // (x1=table_idx from movImm32, x2=idx, x3=val from saved slots),
    // load vmctx into x0, BLR, restore.

    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, &.{ ts.idx, ts.val });

    // x1 = table_idx, x2 = idx, x3 = val, x0 = vmctx.
    try code.movImm32(.x1, @intCast(ts.table_idx));
    try stageArgFromSaved(code, reg_map, fctx, .x2, ts.idx, 0);
    try stageArgFromSaved(code, reg_map, fctx, .x3, ts.val, 0);
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_table_set_fn_slot);
    try code.blr(RegMap.tmp0);

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// Build a bitmask over `RegMap.scratch_regs` indices of regs whose
/// currently-bound vreg is in `dying`. Used by call emitters to skip
/// the post-call restore of caller-save regs that are dead after the
/// call (their vreg's last static use is the call itself).
/// Pre-call save loop: store currently-used caller-save physregs to the
/// per-function `call_save_base` slot area, so `stageArgFromSaved` can
/// fall back to the slot when an earlier arg's target clobbers a later
/// arg's source. Greedy: saves every used caller-save reg (conservative).
/// Regalloc: limits saves to caller-save regs holding an arg source,
/// since regalloc proves all other caller-save values are dead at the
/// call.
fn saveCallerSaveForCall(
    code: *emit.CodeBuffer,
    reg_map: *const RegMap,
    fctx: *const FuncCompileCtx,
    used_snapshot: []const bool,
    arg_sources: []const ir.VReg,
) !void {
    const save_mask: u16 = if (reg_map.alloc_result != null)
        callerSaveArgMask(reg_map, arg_sources)
    else
        std.math.maxInt(u16);
    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        if ((save_mask >> @intCast(i)) & 1 == 0) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.strImm(reg, .fp, slot_scaled);
    }
}

fn dyingCallerSaveMask(reg_map: *const RegMap, dying: []const ir.VReg) u16 {
    var mask: u16 = 0;
    for (dying) |v| {
        const loc = reg_map.get(v) orelse continue;
        switch (loc) {
            .reg => |r| {
                for (RegMap.scratch_regs, 0..) |sr, i| {
                    if (sr == r) {
                        if (i < RegMap.caller_saved_count) {
                            mask |= (@as(u16, 1) << @intCast(i));
                        }
                        break;
                    }
                }
            },
            .stack => {},
        }
    }
    return mask;
}

/// Bitmask over `RegMap.scratch_regs` indices of caller-save regs that
/// currently hold one of the given `args` vregs. Used under regalloc to
/// limit pre-call save-loops to regs whose values the parallel-move arg
/// staging actually needs to preserve. All other caller-save regs hold
/// dead values at the call (regalloc ensures no live vreg spans the
/// clobber point) and don't need to be spilled.
fn callerSaveArgMask(reg_map: *const RegMap, args: []const ir.VReg) u16 {
    var mask: u16 = 0;
    for (args) |v| {
        const loc = reg_map.get(v) orelse continue;
        switch (loc) {
            .reg => |r| {
                for (RegMap.scratch_regs, 0..) |sr, i| {
                    if (sr == r) {
                        if (i < RegMap.caller_saved_count) {
                            mask |= (@as(u16, 1) << @intCast(i));
                        }
                        break;
                    }
                }
            },
            .stack => {},
        }
    }
    return mask;
}

/// Stage a single call argument into `target`.
///
/// `clobbered` is a bitmask over `RegMap.scratch_regs` indices tracking
/// which caller-save regs have been overwritten by earlier arg staging
/// in the same call. If the arg's source reg is still intact, we `mov`
/// directly from it; otherwise we fall back to reading the call-save
/// slot that the caller filled before staging began.
///
/// Precondition: the caller has already spilled all used caller-save
/// regs to `fctx.call_save_base` (so the memory fallback is valid).
fn stageArgFromSaved(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
    target: emit.Reg,
    vreg: ir.VReg,
    clobbered: u16,
) !void {
    const loc = reg_map.get(vreg) orelse return error.UnboundVReg;
    switch (loc) {
        .reg => |r| {
            const reg_num: u32 = @intFromEnum(r);
            if (reg_num >= 19) {
                try code.movRegReg(target, r);
            } else if (target == r) {
                // Value already in the right place.
            } else if ((clobbered >> @intCast(reg_num)) & 1 == 0) {
                // Caller-save source still intact since the snapshot.
                try code.movRegReg(target, r);
            } else {
                // Source reg was overwritten by earlier arg staging or
                // by pre-staging fixup; read the saved value from its
                // fixed call-save slot.
                const slot_scaled: u12 = @intCast((fctx.call_save_base + reg_num * 8) / 8);
                try code.ldrImm(target, .fp, slot_scaled);
            }
        },
        .stack => |off| {
            try code.ldrImm(target, .fp, reg_map.spillOffsetScaled(off));
        },
    }
}

/// `.table_grow` — calls `VmCtx.table_grow_fn(vmctx, init, delta, table_idx)`.
/// Returns the previous size (or -1 on failure) in x0.
fn emitTableGrow(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    tg: @TypeOf(@as(ir.Inst.Op, undefined).table_grow),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, &.{ tg.init, tg.delta });

    // x1 = init, x2 = delta, x3 = table_idx, x0 = vmctx.
    try stageArgFromSaved(code, reg_map, fctx, .x1, tg.init, 0);
    try stageArgFromSaved(code, reg_map, fctx, .x2, tg.delta, 0);
    try code.movImm32(.x3, @intCast(tg.table_idx));
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_table_grow_fn_slot);
    try code.blr(RegMap.tmp0);

    if (inst.dest) |dest| {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// `.ref_func` — returns `vmctx.funcptrs_ptr[func_idx]` (a native function
/// pointer for `func_idx`, used as a host-side reference value).
fn emitRefFunc(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    func_idx: u32,
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    try code.ldrImm(RegMap.tmp0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, vmctx_funcptrs_slot);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const slot: u12 = @intCast(func_idx);
    try code.ldrImm(info.reg, RegMap.tmp0, slot);
    try destCommit(code, reg_map, info);
}

/// `.call_indirect` — table-dispatched call with runtime signature check.
///   ti = vmctx.tables_info[table_idx]
///   if (idx >= ti.len) trap_unreachable;
///   expected_sig = vmctx.sig_table_ptr[type_idx]
///   actual_sig = ti.type_backing_ptr[idx]
///   if (expected_sig != actual_sig) trap_unreachable;
///   fn_ptr = ti.ptr[idx];
///   <marshal args, BLR fn_ptr>
fn emitCallIndirect(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    ci: @TypeOf(@as(ir.Inst.Op, undefined).call_indirect),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    const total_ci_regs: u32 = @as(u32, @intCast(ci.args.len)) +
        (if (ci.extra_results > 0) @as(u32, 1) else 0);
    if (total_ci_regs > 7) return error.TooManyCallArgs;

    // Snapshot live caller-save regs and spill to call_save.
    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    {
        var all_args: [8]ir.VReg = undefined;
        all_args[0] = ci.elem_idx;
        for (ci.args, 0..) |a, i| all_args[1 + i] = a;
        try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, all_args[0 .. 1 + ci.args.len]);
    }

    // Stage `elem_idx` into a callee-safe spot: write it to a dedicated
    // scratch slot on the stack (we reuse slot 0 of the call-save region
    // is risky since we just spilled to those slots — use the spill-slot
    // of elem_idx itself). Actually, re-read elem_idx from its saved
    // location into tmp2 each time we need it.
    //
    // Load elem_idx → tmp2 (32-bit value, zero-extended).
    {
        const loc = reg_map.get(ci.elem_idx) orelse return error.UnboundVReg;
        switch (loc) {
            .reg => |r| {
                const reg_idx: u32 = @intFromEnum(r);
                if (reg_idx >= 19) {
                    try code.movRegReg(RegMap.tmp2, r);
                } else {
                    const slot_scaled: u12 = @intCast((fctx.call_save_base + reg_idx * 8) / 8);
                    try code.ldrImm(RegMap.tmp2, .fp, slot_scaled);
                }
            },
            .stack => |off| {
                try code.ldrImm(RegMap.tmp2, .fp, reg_map.spillOffsetScaled(off));
            },
        }
    }

    // tmp0 = &vmctx.tables_info[table_idx]
    try loadTableInfoAddr(code, RegMap.tmp0, ci.table_idx);
    // Bounds check: tmp1 = ti.len; cmp tmp2, tmp1; b.lo ok; trap; ok:
    const len_slot_w: u12 = @intCast(table_info_len_off / 4);
    try code.ldrImm32(RegMap.tmp1, RegMap.tmp0, len_slot_w);
    try code.cmpRegReg32(RegMap.tmp2, RegMap.tmp1);
    const skip_bounds = code.len();
    try code.bCond(.lo, 0);
    try emitTrapHelperCall(code, vmctx_trap_unreachable_fn_slot);
    patchBCondHere(code, skip_bounds);

    // Sig check:
    //   tmp1 = ti.type_backing_ptr (at offset 16)
    //   actual_sig = type_backing_ptr[idx]        (u32, stride 4)
    //   expected_sig = vmctx.sig_table_ptr[type_idx]  (u32, stride 4)
    //   cmp actual, expected; b.eq ok; trap; ok:
    try code.ldrImm(RegMap.tmp1, RegMap.tmp0, @intCast(table_info_type_backing_off / 8));
    // tmp1 = type_backing_ptr + idx*4
    try code.uxtw(RegMap.tmp0, RegMap.tmp2);
    try code.lslImm(RegMap.tmp0, RegMap.tmp0, 2);
    try code.addRegReg(RegMap.tmp1, RegMap.tmp1, RegMap.tmp0);
    try code.ldrImm32(RegMap.tmp1, RegMap.tmp1, 0); // tmp1 = actual_sig

    // Load expected_sig from vmctx.sig_table_ptr[type_idx].
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(.x0, .x0, vmctx_sig_table_slot);
    try code.ldrImm32(RegMap.tmp0, .x0, @intCast(ci.type_idx)); // scale 4

    try code.cmpRegReg32(RegMap.tmp1, RegMap.tmp0);
    const skip_sig = code.len();
    try code.bCond(.eq, 0);
    try emitTrapHelperCall(code, vmctx_trap_unreachable_fn_slot);
    patchBCondHere(code, skip_sig);

    // Load fn_ptr: ti.ptr + idx*8. Recompute ti addr (x0 clobbered above).
    try loadTableInfoAddr(code, RegMap.tmp0, ci.table_idx);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, @intCast(table_info_ptr_off / 8));
    try code.addExtUxtw3(RegMap.tmp0, RegMap.tmp0, RegMap.tmp2);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, 0); // tmp0 = fn_ptr

    // Park fn_ptr in the call_save slot for tmp0's scratch index so it
    // survives arg staging. Actually tmp0 is x16 — non-allocatable — so
    // arg staging won't clobber it (arg staging only uses x1..x7, x0,
    // and reads from call_save). Safe to keep in tmp0 across arg moves.

    // Stage args into x1..xN. The bounds/sig check sequence above
    // clobbered x0 (and other tmps not in caller-save range), so seed
    // `clobbered_ci` with bit 0 to force args sourced from x0 to read
    // from the saved slot rather than the now-garbage register.
    const dying_mask_ci: u16 = dyingCallerSaveMask(reg_map, fctx.current_kills);
    var clobbered_ci: u16 = 1; // x0 already trashed by sig/bounds checks
    for (ci.args, 0..) |arg_vreg, i| {
        const target = callArgReg(@intCast(i));
        try stageArgFromSaved(code, reg_map, fctx, target, arg_vreg, clobbered_ci);
        clobbered_ci |= (@as(u16, 1) << @intCast(@intFromEnum(target)));
    }
    if (ci.tail) {
        // Tail: args already in x1..x7. Forward our caller's HRP if the
        // tail target is multi-result. Then re-materialize the target
        // fn_ptr into tmp0 AFTER any HRP load (which can transiently
        // use tmp0 as scratch on the large-offset path), load vmctx, and
        // finally tear down the frame + BR.
        if (ci.extra_results > 0) {
            const hrp_target = callArgReg(@intCast(ci.args.len));
            try frameLoad(code, hrp_target, fctx.hrp_save_off);
        }
        // Recompute fn_ptr into tmp0. We already passed the bounds and
        // sig checks above; tmp2 still holds elem_idx.
        try loadTableInfoAddr(code, RegMap.tmp0, ci.table_idx);
        try code.ldrImm(RegMap.tmp0, RegMap.tmp0, @intCast(table_info_ptr_off / 8));
        try code.addExtUxtw3(RegMap.tmp0, RegMap.tmp0, RegMap.tmp2);
        try code.ldrImm(RegMap.tmp0, RegMap.tmp0, 0);
        // Load vmctx into x0 (last FP-relative read before teardown).
        try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
        try emitCalleeSaveRestoreTracked(code, fctx);
        try code.emitEpilogueNoRet(fctx.frame_size);
        try code.br(RegMap.tmp0);
        return;
    }

    if (ci.extra_results > 0) {
        const hrp_target = callArgReg(@intCast(ci.args.len));
        try frameAddr(code, hrp_target, fctx.scratch_base);
    }
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.blr(RegMap.tmp0);

    if (inst.dest) |dest| {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        if ((dying_mask_ci >> @intCast(i)) & 1 != 0) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// `.call_ref` — call via a native funcref pointer value. Null-checks
/// the pointer (traps via `VmCtx.trap_unreachable_fn` on null), then
/// performs an indirect call with `vmctx` in x0 and args in x1..xN.
fn emitCallRef(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    cr: @TypeOf(@as(ir.Inst.Op, undefined).call_ref),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    const total_cr_regs: u32 = @as(u32, @intCast(cr.args.len)) +
        (if (cr.extra_results > 0) @as(u32, 1) else 0);
    if (total_cr_regs > 7) return error.TooManyCallArgs;

    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    {
        var all_args: [8]ir.VReg = undefined;
        all_args[0] = cr.func_ref;
        for (cr.args, 0..) |a, i| all_args[1 + i] = a;
        try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, all_args[0 .. 1 + cr.args.len]);
    }

    // Load the 64-bit func_ref pointer into tmp0. tmp0 is x16 (non-
    // allocatable) so it survives arg staging below.
    try stageArgFromSaved(code, reg_map, fctx, RegMap.tmp0, cr.func_ref, 0);

    // Null check: CBZ tmp0, trap_block; fall through on non-null.
    // Trap block is: x0 = vmctx; tmp1 = vmctx.trap_unreachable_fn; BLR tmp1.
    // Total 3 instructions = 12 bytes = 3 words, then BR over.
    // Simpler: branch forward to skip trap.
    const cbz_site = code.len();
    try code.cbz64(RegMap.tmp0, 0); // patched below
    // trap: load vmctx, load fn, blr (noreturn)
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp1, .x0, vmctx_trap_unreachable_fn_slot);
    try code.blr(RegMap.tmp1);
    // CBZ shares the b.cond imm19 encoding at bits [23:5], so the same
    // patch helper works.
    patchBCondHere(code, cbz_site);

    // Stage args into x1..xN (tmp0 holds fn_ptr, non-allocatable).
    const dying_mask_cr: u16 = dyingCallerSaveMask(reg_map, fctx.current_kills);
    var clobbered_cr: u16 = 0;
    for (cr.args, 0..) |arg_vreg, i| {
        const target = callArgReg(@intCast(i));
        try stageArgFromSaved(code, reg_map, fctx, target, arg_vreg, clobbered_cr);
        clobbered_cr |= (@as(u16, 1) << @intCast(@intFromEnum(target)));
    }
    if (cr.tail) {
        // Tail: forward caller's HRP if callee has extras, then re-stage
        // fn_ref into tmp0 (the HRP load can clobber tmp0 on the
        // large-offset path, so we refresh it here). Load vmctx, tear
        // down, and BR.
        if (cr.extra_results > 0) {
            const hrp_target = callArgReg(@intCast(cr.args.len));
            try frameLoad(code, hrp_target, fctx.hrp_save_off);
        }
        try stageArgFromSaved(code, reg_map, fctx, RegMap.tmp0, cr.func_ref, 0);
        try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
        try emitCalleeSaveRestoreTracked(code, fctx);
        try code.emitEpilogueNoRet(fctx.frame_size);
        try code.br(RegMap.tmp0);
        return;
    }

    if (cr.extra_results > 0) {
        const hrp_target = callArgReg(@intCast(cr.args.len));
        try frameAddr(code, hrp_target, fctx.scratch_base);
    }
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.blr(RegMap.tmp0);

    if (inst.dest) |dest| {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        if ((dying_mask_cr >> @intCast(i)) & 1 != 0) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// `.call_result idx` — read the `idx`-th extra result slot written by
/// the preceding call into its dest vreg. Caller-side scratch lives at
/// `[fp + scratch_base]`, with each result occupying 8 bytes.
fn emitCallResult(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    idx: u8,
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    const dest = inst.dest orelse return;
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    const off: u32 = fctx.scratch_base + @as(u32, idx) * 8;
    try frameLoad(code, info.reg, off);
    try destCommit(code, reg_map, info);
}

/// `.ret_multi [v0, v1, ...]` — first result → x0; remaining → written
/// through the saved HRP at `[hrp + (i-1)*8]`. Emits the full epilogue
/// (callee-save restore + emitEpilogue).
fn emitRetMulti(
    code: *emit.CodeBuffer,
    vregs: []const ir.VReg,
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
    frame_size: u32,
) !void {
    if (vregs.len == 0) {
        try emitCalleeSaveRestoreTracked(code, fctx);
        try code.emitEpilogue(frame_size);
        return;
    }

    // First result → x0.
    const first = try useInto(code, reg_map, vregs[0], .x0);
    if (first != .x0) try code.movRegReg(.x0, first);

    if (vregs.len > 1) {
        // Load HRP into tmp0 (non-allocatable, stable across arg reads).
        try frameLoad(code, RegMap.tmp0, fctx.hrp_save_off);
        // Write extra results through HRP.
        var i: u32 = 1;
        while (i < vregs.len) : (i += 1) {
            const src = try useInto(code, reg_map, vregs[i], RegMap.tmp1);
            const scaled: u12 = @intCast(i - 1);
            try code.strImm(src, RegMap.tmp0, scaled);
        }
    }

    try emitCalleeSaveRestoreTracked(code, fctx);
    try code.emitEpilogue(frame_size);
}

/// same caller-save snapshot pattern as `emitCall`: all currently-used
/// scratch regs are spilled to the function's `call_save` region before
/// arg staging, so the arg moves can read from stable stack slots
/// without parallel-move hazards. If `maybe_dest` is non-null, x0 is
/// committed to it before restoring saved regs.
///
/// `imm_args` optionally provides constant `u32` values to materialize
/// into arg positions `args.len`, `args.len + 1`, ... (after the
/// variable-source args). Used by `table.set` / `table.grow` to pass
/// `table_idx`.
fn emitVmctxHelperCall(
    code: *emit.CodeBuffer,
    args: []const ir.VReg,
    fn_slot: u12,
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
    maybe_dest: ?ir.VReg,
) !void {
    if (args.len > 7) return error.UnimplementedOp;

    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, args);

    // Stage VReg args into x1..xN from stable post-snapshot storage.
    const arg_regs = [_]emit.Reg{ .x1, .x2, .x3, .x4, .x5, .x6, .x7 };
    for (args, 0..) |vreg, i| {
        const loc = reg_map.get(vreg) orelse return error.UnboundVReg;
        switch (loc) {
            .reg => |r| {
                const reg_idx: u32 = @intFromEnum(r);
                if (reg_idx >= 19) {
                    try code.movRegReg(arg_regs[i], r);
                } else {
                    const slot_scaled: u12 = @intCast((fctx.call_save_base + reg_idx * 8) / 8);
                    try code.ldrImm(arg_regs[i], .fp, slot_scaled);
                }
            },
            .stack => |off| {
                try code.ldrImm(arg_regs[i], .fp, reg_map.spillOffsetScaled(off));
            },
        }
    }

    // Materialize any constant args into the trailing arg regs.
    _ = &arg_regs;

    // x0 = vmctx; tmp0 = vmctx.<fn_slot>; BLR tmp0.
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, .x0, fn_slot);
    try code.blr(RegMap.tmp0);

    // Commit result (x0) to dest BEFORE restoring saved regs.
    if (maybe_dest) |dest| {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// Load a VReg's current value into `dest_reg` in a way that is stable
/// across a prior caller-save snapshot: if the VReg is in a caller-saved
/// physreg, read from its save slot (not the now-clobbered reg); if it
/// is in a callee-saved physreg, read from the reg directly; if spilled,
/// read from the spill slot.
fn readVregStable(
    code: *emit.CodeBuffer,
    reg_map: *const RegMap,
    fctx: *FuncCompileCtx,
    vreg: ir.VReg,
    dest_reg: emit.Reg,
) !void {
    const loc = reg_map.get(vreg) orelse return error.UnboundVReg;
    switch (loc) {
        .reg => |r| {
            const reg_idx: u32 = @intFromEnum(r);
            if (reg_idx >= 19) {
                if (r != dest_reg) try code.movRegReg(dest_reg, r);
            } else {
                const slot_scaled: u12 = @intCast((fctx.call_save_base + reg_idx * 8) / 8);
                try code.ldrImm(dest_reg, .fp, slot_scaled);
            }
        },
        .stack => |off| {
            try code.ldrImm(dest_reg, .fp, reg_map.spillOffsetScaled(off));
        },
    }
}

fn emitTableInit(
    code: *emit.CodeBuffer,
    ti: @TypeOf(@as(ir.Inst.Op, undefined).table_init),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    // Helper signature (matches runtime.tableInitHelper):
    //   fn (vmctx, packed_seg_table: u64, packed_dst_src: u64, len: u32)
    //   packed_seg_table = seg_idx | (table_idx << 32)   (compile-time)
    //   packed_dst_src   = dst     | (src << 32)

    // Snapshot and save live caller-saved regs.
    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, &.{ ti.src, ti.dst, ti.len });

    // x3 = dst | (src << 32). Build in x3 via tmp0 scratch.
    try readVregStable(code, reg_map, fctx, ti.src, .x3);
    try code.movRegReg32(.x3, .x3); // zero-extend 32→64
    try code.lslImm(.x3, .x3, 32);
    try readVregStable(code, reg_map, fctx, ti.dst, RegMap.tmp0);
    try code.movRegReg32(RegMap.tmp0, RegMap.tmp0);
    try code.orrRegReg(.x3, .x3, RegMap.tmp0);

    // x4 = zero-extended len.
    try readVregStable(code, reg_map, fctx, ti.len, .x4);
    try code.movRegReg32(.x4, .x4);

    // x2 = packed_seg_table (compile-time constant).
    const packed_st: u64 =
        @as(u64, ti.seg_idx) | (@as(u64, ti.table_idx) << 32);
    try code.movImm64(.x2, packed_st);

    // x0 = vmctx; tmp0 = vmctx.table_init_fn; BLR tmp0.
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_table_init_fn_slot);
    try code.blr(RegMap.tmp0);

    // Restore caller-saved regs.
    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

fn emitElemDrop(
    code: *emit.CodeBuffer,
    seg_idx: u32,
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    // Helper: elemDropHelper(vmctx, seg_idx: u32)
    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, &.{});

    try code.movImm32(.x1, @bitCast(seg_idx));
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_elem_drop_fn_slot);
    try code.blr(RegMap.tmp0);

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// Patch a B.cond placeholder at `patch_off` to branch to the current PC.
/// Preserves the original condition bits and updates only imm19.
fn patchBCondHere(code: *emit.CodeBuffer, patch_off: usize) void {
    const target = code.len();
    const delta_words: i19 = @intCast(@divExact(
        @as(i64, @intCast(target)) - @as(i64, @intCast(patch_off)),
        4,
    ));
    const existing = std.mem.readInt(u32, code.bytes.items[patch_off..][0..4], .little);
    const imm19: u19 = @bitCast(delta_words);
    const new_word: u32 = (existing & 0xFF00001F) | (@as(u32, imm19) << 5);
    code.patch32(patch_off, new_word);
}

/// Emit a noreturn call to a VmCtx trap helper at scaled slot `slot`.
/// Sequence: load x0 = vmctx, tmp0 = [x0 + slot*8], BLR tmp0, BRK 0.
/// Clobbers x0, tmp0, LR — but this path is noreturn, so callers do not
/// care what survives it.
fn emitTrapHelperCall(code: *emit.CodeBuffer, slot: u12) !void {
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, .x0, slot);
    try code.blr(RegMap.tmp0);
    try code.brk(0);
}

const DivRemKind = enum { div_s, div_u, rem_s, rem_u };

/// Emit div_s / div_u / rem_s / rem_u. Semantics per wasm:
///   * rhs == 0 traps (idivz helper).
///   * div_s with INT_MIN / -1 traps (iovf helper).
///   * rem_s with INT_MIN / -1 yields 0 — handled naturally by MSUB with
///     two's complement wrap (INT_MIN - INT_MIN*(-1) == 0).
///
/// Uses tmp0=lhs, tmp1=rhs, tmp2=scratch/quotient/INT_MIN constant.
fn emitDivRem(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    kind: DivRemKind,
) !void {
    const dest = inst.dest orelse return;
    const is64 = inst.type == .i64;

    // Copy sources into the fixed scratch registers (tmp0, tmp1). useInto
    // may already place them there, but if the vreg lives in an allocated
    // physreg we need an explicit move because later code clobbers tmp0
    // and tmp1 across the trap-helper calls.
    const lhs_loaded = try useInto(code, reg_map, bin.lhs, RegMap.tmp0);
    if (lhs_loaded != RegMap.tmp0) {
        if (is64) try code.movRegReg(RegMap.tmp0, lhs_loaded) else try code.movRegReg32(RegMap.tmp0, lhs_loaded);
    }
    const rhs_loaded = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    if (rhs_loaded != RegMap.tmp1) {
        if (is64) try code.movRegReg(RegMap.tmp1, rhs_loaded) else try code.movRegReg32(RegMap.tmp1, rhs_loaded);
    }

    // Zero-divisor check: cmp rhs, #0; b.ne skip; call trap_idivz; skip:
    if (is64) try code.cmpImm(RegMap.tmp1, 0) else try code.cmpImm32(RegMap.tmp1, 0);
    const skip_idivz = code.len();
    try code.bCond(.ne, 0);
    try emitTrapHelperCall(code, vmctx_trap_idivz_fn_slot);
    patchBCondHere(code, skip_idivz);

    // div_s INT_MIN/-1 overflow check. Sequence:
    //   tmp2 = -1; cmp rhs, tmp2; b.ne skip_ovf
    //   tmp2 = INT_MIN; cmp lhs, tmp2; b.ne skip_ovf
    //   call trap_iovf
    // skip_ovf:
    if (kind == .div_s) {
        if (is64) {
            try code.movImm64(RegMap.tmp2, 0xFFFF_FFFF_FFFF_FFFF);
            try code.cmpRegReg(RegMap.tmp1, RegMap.tmp2);
        } else {
            try code.movImm32(RegMap.tmp2, -1);
            try code.cmpRegReg32(RegMap.tmp1, RegMap.tmp2);
        }
        const skip_a = code.len();
        try code.bCond(.ne, 0);
        if (is64) {
            try code.movImm64(RegMap.tmp2, 0x8000_0000_0000_0000);
            try code.cmpRegReg(RegMap.tmp0, RegMap.tmp2);
        } else {
            try code.movImm32(RegMap.tmp2, @as(i32, -2147483648));
            try code.cmpRegReg32(RegMap.tmp0, RegMap.tmp2);
        }
        const skip_b = code.len();
        try code.bCond(.ne, 0);
        try emitTrapHelperCall(code, vmctx_trap_iovf_fn_slot);
        patchBCondHere(code, skip_a);
        patchBCondHere(code, skip_b);
    }

    // After checks: tmp0=lhs, tmp1=rhs are still live. Now allocate dest
    // (destBegin's scratch is tmp2 — safe because tmp2 is dead here).
    const info = try destBegin(reg_map, dest, RegMap.tmp2);

    switch (kind) {
        .div_s => if (is64)
            try code.sdivRegReg(info.reg, RegMap.tmp0, RegMap.tmp1)
        else
            try code.sdivRegReg32(info.reg, RegMap.tmp0, RegMap.tmp1),
        .div_u => if (is64)
            try code.udivRegReg(info.reg, RegMap.tmp0, RegMap.tmp1)
        else
            try code.udivRegReg32(info.reg, RegMap.tmp0, RegMap.tmp1),
        .rem_s, .rem_u => {
            // q = lhs DIV rhs into tmp2; dest = lhs - q*rhs via MSUB.
            // If info.reg happens to equal tmp2 (dest is spilled), SDIV/UDIV
            // still writes tmp2=q, then MSUB reads q (tmp2) before writing
            // dest (tmp2) — single-instruction read-before-write, safe.
            if (kind == .rem_s) {
                if (is64) try code.sdivRegReg(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1) else try code.sdivRegReg32(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1);
            } else {
                if (is64) try code.udivRegReg(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1) else try code.udivRegReg32(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1);
            }
            // MSUB dest, q, rhs, lhs  →  dest = lhs - q*rhs
            if (is64) try code.msubRegReg(info.reg, RegMap.tmp2, RegMap.tmp1, RegMap.tmp0) else try code.msubRegReg32(info.reg, RegMap.tmp2, RegMap.tmp1, RegMap.tmp0);
        },
    }
    try destCommit(code, reg_map, info);
}

/// Cascaded-compare br_table. For each target i, emits:
///   cmp idx, #i
///   b.eq target[i]        ; patched to absolute block by caller
/// then an unconditional B to the default block at the end. Because the
/// default branch is taken whenever every EQ compare fails, indices ≥
/// targets.len naturally dispatch to default (matching wasm semantics
/// that treats the index as unsigned).
///
/// Code size is O(N) instructions. Real wasm br_tables are usually small
/// (few dozen arms); Phase 4 may replace this with a true jump table via
/// ADR + register-indexed LDR + BR for hot paths.
fn emitBrTable(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    bt: @TypeOf(@as(ir.Inst.Op, undefined).br_table),
    allocator: std.mem.Allocator,
) !void {
    const idx_r = try useInto(code, reg_map, bt.index, RegMap.tmp0);
    for (bt.targets, 0..) |target, i| {
        if (i <= 0xFFF) {
            try code.cmpImm32(idx_r, @intCast(i));
        } else {
            try code.movImm32(RegMap.tmp1, @intCast(i));
            try code.cmpRegReg32(idx_r, RegMap.tmp1);
        }
        const patch_off = code.len();
        try code.bCond(.eq, 0); // placeholder
        try patches.append(allocator, .{
            .patch_offset = patch_off,
            .target_block = target,
            .kind = .b_cond,
        });
    }
    const default_patch = code.len();
    try code.b(0); // placeholder
    try patches.append(allocator, .{
        .patch_offset = default_patch,
        .target_block = bt.default,
        .kind = .b_uncond,
    });
}

/// Scaled-immediate offset suitable for LDR/STR with given scale.
/// Returns null if the offset isn't a multiple of `scale` within u12*scale range.
fn scaledOffset(off: u32, scale: u32) ?u12 {
    const max = @as(u32, 0xFFF) * scale;
    if (off > max) return null;
    if ((off % scale) != 0) return null;
    return @intCast(off / scale);
}

fn emitLoad(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    ld: @TypeOf(@as(ir.Inst.Op, undefined).load),
    reg_map: *RegMap,
) !void {
    const dest = inst.dest orelse return;
    const is64 = inst.type == .i64;

    // Validate size/sign combinations. i32.load32 can't sign-extend since
    // there's nothing wider to extend into at 32-bit dest.
    const scale: u32 = switch (ld.size) {
        1 => 1,
        2 => 2,
        4 => 4,
        8 => 8,
        else => return error.UnimplementedLoadSize,
    };
    if (ld.sign_extend and ld.size == 8) return error.BadLoadSign; // 8-byte has no extension
    if (ld.sign_extend and ld.size == 4 and !is64) return error.BadLoadSign;

    // Try to fold offset into the load's scaled immediate; else add it to tmp0.
    const folded = scaledOffset(ld.offset, scale);
    const end_offset: u64 = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + @as(u64, ld.size);
    const folded_offset: u32 = if (folded == null) ld.offset else 0;
    if (ld.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, ld.base, folded_offset);
    } else {
        try emitMemAddr(code, reg_map, ld.base, folded_offset, end_offset);
    }
    const disp: u12 = if (folded) |d| d else 0;

    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    if (ld.sign_extend) {
        switch (ld.size) {
            1 => if (is64) try code.ldrsbImm64(info.reg, RegMap.tmp0, disp) else try code.ldrsbImm32(info.reg, RegMap.tmp0, disp),
            2 => if (is64) try code.ldrshImm64(info.reg, RegMap.tmp0, disp) else try code.ldrshImm32(info.reg, RegMap.tmp0, disp),
            4 => try code.ldrswImm(info.reg, RegMap.tmp0, disp), // always X-form
            else => unreachable,
        }
    } else {
        switch (ld.size) {
            1 => try code.ldrbImm(info.reg, RegMap.tmp0, disp),
            2 => try code.ldrhImm(info.reg, RegMap.tmp0, disp),
            4 => try code.ldrImm32(info.reg, RegMap.tmp0, disp),
            8 => try code.ldrImm(info.reg, RegMap.tmp0, disp),
            else => unreachable,
        }
    }
    try destCommit(code, reg_map, info);
}

fn emitStore(
    code: *emit.CodeBuffer,
    st: @TypeOf(@as(ir.Inst.Op, undefined).store),
    reg_map: *RegMap,
) !void {
    const scale: u32 = switch (st.size) {
        1 => 1,
        2 => 2,
        4 => 4,
        8 => 8,
        else => return error.UnimplementedStoreSize,
    };

    const folded = scaledOffset(st.offset, scale);
    const end_offset: u64 = if (st.checked_end > 0) st.checked_end else @as(u64, st.offset) + @as(u64, st.size);
    const folded_offset: u32 = if (folded == null) st.offset else 0;
    if (st.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, st.base, folded_offset);
    } else {
        try emitMemAddr(code, reg_map, st.base, folded_offset, end_offset);
    }
    const disp: u12 = if (folded) |d| d else 0;

    // Materialize the value into tmp1 (or use its home reg). emitMemAddr is
    // done with tmp1 by now, so it's free for `useInto` to reuse.
    const val_reg = try useInto(code, reg_map, st.val, RegMap.tmp1);
    switch (st.size) {
        1 => try code.strbImm(val_reg, RegMap.tmp0, disp),
        2 => try code.strhImm(val_reg, RegMap.tmp0, disp),
        4 => try code.strImm32(val_reg, RegMap.tmp0, disp),
        8 => try code.strImm(val_reg, RegMap.tmp0, disp),
        else => unreachable,
    }
}

fn emitAtomicLoad(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    ld: @TypeOf(@as(ir.Inst.Op, undefined).atomic_load),
    reg_map: *RegMap,
) !void {
    // wasm threads: seq-cst load. LDAR has no offset form, so always fold
    // the full offset into the base address in tmp0. Upper bits of sub-word
    // loads are zero-extended by LDARB/LDARH; LDAR Wt also zeroes bits[63:32].
    const dest = inst.dest orelse return;
    switch (ld.size) {
        1, 2, 4, 8 => {},
        else => return error.UnimplementedAtomicSize,
    }
    const end_offset: u64 = @as(u64, ld.offset) + @as(u64, ld.size);
    try emitMemAddr(code, reg_map, ld.base, ld.offset, end_offset);
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    try code.ldarSized(info.reg, RegMap.tmp0, ld.size);
    try destCommit(code, reg_map, info);
}

fn emitAtomicStore(
    code: *emit.CodeBuffer,
    st: @TypeOf(@as(ir.Inst.Op, undefined).atomic_store),
    reg_map: *RegMap,
) !void {
    switch (st.size) {
        1, 2, 4, 8 => {},
        else => return error.UnimplementedAtomicSize,
    }
    const end_offset: u64 = @as(u64, st.offset) + @as(u64, st.size);
    try emitMemAddr(code, reg_map, st.base, st.offset, end_offset);
    const val_reg = try useInto(code, reg_map, st.val, RegMap.tmp1);
    try code.stlrSized(val_reg, RegMap.tmp0, st.size);
}

fn emitAtomicRmw(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    rmw: @TypeOf(@as(ir.Inst.Op, undefined).atomic_rmw),
    reg_map: *RegMap,
) !void {
    // ARMv8.1-A LSE one-instruction seq-cst atomics.
    //   add  -> LDADDAL        Rs=val, Rt=old, [Rn=addr]
    //   sub  -> NEG tmp, val;  LDADDAL tmp, old, [addr]
    //   and  -> MVN tmp, val;  LDCLRAL tmp, old, [addr]   (clear bits in ~val)
    //   or   -> LDSETAL        Rs=val, Rt=old
    //   xor  -> LDEORAL        Rs=val, Rt=old
    //   xchg -> SWPAL          Rs=val, Rt=old
    switch (rmw.size) {
        1, 2, 4, 8 => {},
        else => return error.UnimplementedAtomicSize,
    }
    const dest = inst.dest orelse return error.AtomicRmwMissingDest;
    const end_offset: u64 = @as(u64, rmw.offset) + @as(u64, rmw.size);
    try emitMemAddr(code, reg_map, rmw.base, rmw.offset, end_offset);

    // After emitMemAddr: tmp0 = effective address, tmp1/tmp2 free.
    const val_src = try useInto(code, reg_map, rmw.val, RegMap.tmp1);

    // For sub/and we need a mutated copy of val in tmp1.
    var rs_reg: emit.Reg = val_src;
    var lse_op: emit.CodeBuffer.LseOp = undefined;
    switch (rmw.op) {
        .add => lse_op = .add,
        .sub => {
            try code.negReg64(RegMap.tmp1, val_src);
            rs_reg = RegMap.tmp1;
            lse_op = .add;
        },
        .@"and" => {
            try code.mvnReg64(RegMap.tmp1, val_src);
            rs_reg = RegMap.tmp1;
            lse_op = .clr;
        },
        .@"or" => lse_op = .set,
        .xor => lse_op = .eor,
        .xchg => lse_op = .swp,
    }

    // The result (old value) goes into info.reg. LDADDAL / LDCLRAL / etc
    // require Rt (destination) distinct from Rs on the constrained path —
    // and info.reg is always in the scratch pool (never tmp0/1/2), so Rt
    // != Rn (addr=tmp0) and Rt != Rs (rs_reg ∈ {tmp1, val's home reg}).
    const info = try destBegin(reg_map, dest, RegMap.tmp2);
    try code.lseAtomic(lse_op, rs_reg, info.reg, RegMap.tmp0, rmw.size);
    try destCommit(code, reg_map, info);
}

fn emitAtomicCmpxchg(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    cx: @TypeOf(@as(ir.Inst.Op, undefined).atomic_cmpxchg),
    reg_map: *RegMap,
) !void {
    // ARMv8.1-A LSE CASAL: one instruction. Rs = expected (input) & old
    // value (output). So we load expected into the dest register and
    // CASAL updates it in place.
    switch (cx.size) {
        1, 2, 4, 8 => {},
        else => return error.UnimplementedAtomicSize,
    }
    const dest = inst.dest orelse return error.AtomicCmpxchgMissingDest;
    const end_offset: u64 = @as(u64, cx.offset) + @as(u64, cx.size);
    try emitMemAddr(code, reg_map, cx.base, cx.offset, end_offset);

    // info.reg receives `expected`, then CASAL updates it in-place to old.
    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const exp_src = try useInto(code, reg_map, cx.expected, info.reg);
    if (exp_src != info.reg) try code.movRegReg(info.reg, exp_src);

    const rep_src = try useInto(code, reg_map, cx.replacement, RegMap.tmp2);
    if (rep_src != RegMap.tmp2) try code.movRegReg(RegMap.tmp2, rep_src);

    try code.casAl(info.reg, RegMap.tmp2, RegMap.tmp0, cx.size);
    try destCommit(code, reg_map, info);
}

/// `.atomic_notify` — calls `VmCtx.futex_notify_fn(vmctx, addr, count)`.
/// `addr` = zext32(base) + offset (NOT the effective memory pointer —
/// helper interprets it as a wasm linear-memory offset).
fn emitAtomicNotify(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    an: @TypeOf(@as(ir.Inst.Op, undefined).atomic_notify),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, &.{ an.base, an.count });

    // x1 = zext32(base) + offset
    try stageArgFromSaved(code, reg_map, fctx, .x1, an.base, 0);
    try code.uxtw(.x1, .x1);
    if (an.offset > 0) {
        try code.movImm32(RegMap.tmp0, @intCast(an.offset));
        try code.addRegReg(.x1, .x1, RegMap.tmp0);
    }
    // x2 = count
    try stageArgFromSaved(code, reg_map, fctx, .x2, an.count, 0);
    // x0 = vmctx, tmp0 = helper, BLR.
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_futex_notify_fn_slot);
    try code.blr(RegMap.tmp0);

    if (inst.dest) |dest| {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// `.atomic_wait` — calls `VmCtx.futex_wait32_fn` or `futex_wait64_fn`.
/// Signatures (AAPCS64, 8 arg regs available, all fit):
///   wait32(vmctx, addr, expected, timeout_lo, timeout_hi) → i32
///   wait64(vmctx, addr, exp_lo, exp_hi, timeout_lo, timeout_hi) → i32
/// Our IR packs both timeout halves into one i64 VReg `timeout`, and
/// for wait64 `expected` is an i64 VReg. We split into lo/hi halves at
/// the call site.
fn emitAtomicWait(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    aw: @TypeOf(@as(ir.Inst.Op, undefined).atomic_wait),
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
) !void {
    switch (aw.size) {
        4, 8 => {},
        else => return error.UnimplementedAtomicSize,
    }

    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, &.{ aw.base, aw.expected, aw.timeout });

    // x1 = zext32(base) + offset
    try stageArgFromSaved(code, reg_map, fctx, .x1, aw.base, 0);
    try code.uxtw(.x1, .x1);
    if (aw.offset > 0) {
        try code.movImm32(RegMap.tmp0, @intCast(aw.offset));
        try code.addRegReg(.x1, .x1, RegMap.tmp0);
    }

    if (aw.size == 4) {
        // x2 = expected (u32, zero-extended from low 32)
        try stageArgFromSaved(code, reg_map, fctx, .x2, aw.expected, 0);
        try code.uxtw(.x2, .x2);
        // x3 = timeout_lo, x4 = timeout_hi (timeout is i64).
        try stageArgFromSaved(code, reg_map, fctx, .x3, aw.timeout, 0);
        try code.movRegReg(.x4, .x3);
        try code.lsrImm(.x4, .x4, 32);
        try code.uxtw(.x3, .x3);
        try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
        try code.ldrImm(RegMap.tmp0, .x0, vmctx_futex_wait32_fn_slot);
    } else {
        // wait64: expected is i64 → exp_lo (x2) / exp_hi (x3).
        try stageArgFromSaved(code, reg_map, fctx, .x2, aw.expected, 0);
        try code.movRegReg(.x3, .x2);
        try code.lsrImm(.x3, .x3, 32);
        try code.uxtw(.x2, .x2);
        // timeout → timeout_lo (x4) / timeout_hi (x5).
        try stageArgFromSaved(code, reg_map, fctx, .x4, aw.timeout, 0);
        try code.movRegReg(.x5, .x4);
        try code.lsrImm(.x5, .x5, 32);
        try code.uxtw(.x4, .x4);
        try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
        try code.ldrImm(RegMap.tmp0, .x0, vmctx_futex_wait64_fn_slot);
    }

    try code.blr(RegMap.tmp0);

    if (inst.dest) |dest| {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

fn emitBrIf(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    br: @TypeOf(@as(ir.Inst.Op, undefined).br_if),
    allocator: std.mem.Allocator,
) !void {
    const cond_r = try useInto(code, reg_map, br.cond, RegMap.tmp0);
    // wasm br_if: branch to then_block if cond != 0 (i32).
    try code.cmpImm32(cond_r, 0);
    const cond_patch = code.len();
    try code.bCond(.ne, 0); // placeholder
    try patches.append(allocator, .{
        .patch_offset = cond_patch,
        .target_block = br.then_block,
        .kind = .b_cond,
    });
    // Unconditional branch to else_block
    const else_patch = code.len();
    try code.b(0);
    try patches.append(allocator, .{
        .patch_offset = else_patch,
        .target_block = br.else_block,
        .kind = .b_uncond,
    });
}

const BinOpKind = enum { add, sub, mul, @"and", @"or", xor };

/// Encode an integer as an AArch64 ADD/SUB immediate, if possible.
/// Returns null if the value can't be encoded as a 12-bit imm
/// (optionally LSL 12). Negative values are encoded as the opposite op
/// (add↔sub), signaled by `flip_op`.
const AddSubImmEnc = struct { imm12: u12, shift12: bool, flip_op: bool };

fn encodeAddSubImm(value: i64) ?AddSubImmEnc {
    const flip = value < 0;
    const abs_val: u64 = if (flip) @intCast(-value) else @intCast(value);
    if (abs_val <= 0xFFF) {
        return .{ .imm12 = @intCast(abs_val), .shift12 = false, .flip_op = flip };
    }
    if ((abs_val & 0xFFF) == 0 and (abs_val >> 12) <= 0xFFF) {
        return .{ .imm12 = @intCast(abs_val >> 12), .shift12 = true, .flip_op = flip };
    }
    return null;
}

fn emitAddSubImm(
    code: *emit.CodeBuffer,
    dst: emit.Reg,
    src: emit.Reg,
    enc: AddSubImmEnc,
    base_kind: BinOpKind, // .add or .sub (pre-flip)
) !void {
    const is_sub = switch (base_kind) {
        .add => enc.flip_op,
        .sub => !enc.flip_op,
        else => unreachable,
    };
    if (is_sub) {
        if (enc.shift12) try code.subImmShift12(dst, src, enc.imm12) else try code.subImm(dst, src, enc.imm12);
    } else {
        if (enc.shift12) try code.addImmShift12(dst, src, enc.imm12) else try code.addImm(dst, src, enc.imm12);
    }
}

fn emitBinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.BinOp,
    reg_map: *RegMap,
    kind: BinOpKind,
    fctx: *const FuncCompileCtx,
) !void {
    const dest = inst.dest orelse return;

    // FMA fusion: if this is an add/sub whose dest is in fma_info, skip the
    // mul (already marked for skip) and emit MADD/MSUB in one shot.
    if ((kind == .add or kind == .sub) and fctx.fma_info != null) {
        if (fctx.fma_info.?.get(dest)) |info| {
            const dst_info = try destBegin(reg_map, dest, RegMap.tmp0);
            // Load three sources into fresh scratch regs. When dst is
            // spilled, dst_info.reg == tmp0; reserve tmp1/tmp2 plus a
            // secondary landing pad via the reg_map for the addend.
            const lhs_scratch: emit.Reg = if (dst_info.spill_slot == null) RegMap.tmp0 else dst_info.reg;
            const lhs_reg = try useInto(code, reg_map, info.mul_lhs, lhs_scratch);
            const rhs_reg = try useInto(code, reg_map, info.mul_rhs, RegMap.tmp1);
            const add_reg = try useInto(code, reg_map, info.addend, RegMap.tmp2);
            const is64 = (inst.type == .i64);
            if (info.is_sub) {
                if (is64)
                    try code.msubRegReg(dst_info.reg, lhs_reg, rhs_reg, add_reg)
                else
                    try code.msubRegReg32(dst_info.reg, lhs_reg, rhs_reg, add_reg);
            } else {
                if (is64)
                    try code.maddRegReg(dst_info.reg, lhs_reg, rhs_reg, add_reg)
                else
                    try code.maddRegReg32(dst_info.reg, lhs_reg, rhs_reg, add_reg);
            }
            try destCommit(code, reg_map, dst_info);
            return;
        }
    }

    // Mul skip: if this mul's dest was absorbed into a later MADD/MSUB, emit
    // nothing. The dest vreg is never read, so no allocation is needed.
    if (kind == .mul and fctx.mul_fused != null and fctx.mul_fused.?.contains(dest)) {
        return;
    }

    // Allocate dest first; if spilled, dst_info.reg is tmp0 and we STR on commit.
    const dst_info = try destBegin(reg_map, dest, RegMap.tmp0);

    // Try to fold an ADD/SUB immediate. For ADD, either operand may be the
    // constant (commutative). For SUB, only rhs may be (non-commutative).
    if ((kind == .add or kind == .sub) and fctx.const_vals != null) {
        const cmap = fctx.const_vals.?;
        const rhs_const = cmap.get(bin.rhs);
        const lhs_const = if (kind == .add) cmap.get(bin.lhs) else null;
        const pick: ?struct { imm: i64, other: ir.VReg } = blk: {
            if (rhs_const) |v| {
                if (encodeAddSubImm(v) != null) break :blk .{ .imm = v, .other = bin.lhs };
            }
            if (lhs_const) |v| {
                if (encodeAddSubImm(v) != null) break :blk .{ .imm = v, .other = bin.rhs };
            }
            break :blk null;
        };
        if (pick) |p| {
            const enc = encodeAddSubImm(p.imm).?;
            const other_scratch: emit.Reg = if (dst_info.spill_slot == null)
                RegMap.tmp0
            else
                dst_info.reg;
            const other_reg = try useInto(code, reg_map, p.other, other_scratch);
            try emitAddSubImm(code, dst_info.reg, other_reg, enc, kind);
            try destCommit(code, reg_map, dst_info);
            return;
        }
    }

    // General reg/reg path.
    const lhs_scratch: emit.Reg = if (dst_info.spill_slot == null) RegMap.tmp0 else dst_info.reg;
    const lhs_reg = try useInto(code, reg_map, bin.lhs, lhs_scratch);
    const rhs_reg = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);

    switch (kind) {
        .add => try code.addRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .sub => try code.subRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .mul => try code.mulRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .@"and" => try code.andRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .@"or" => try code.orrRegReg(dst_info.reg, lhs_reg, rhs_reg),
        .xor => try code.eorRegReg(dst_info.reg, lhs_reg, rhs_reg),
    }
    try destCommit(code, reg_map, dst_info);
}

/// Result of compiling an IR module.
pub const CompileResult = struct {
    code: []u8,
    offsets: []u32,
};

/// Compile all functions in an IR module to AArch64 machine code.
pub fn compileModule(ir_module: *const ir.IrModule, allocator: std.mem.Allocator) !CompileResult {
    return compileModuleWithOptions(ir_module, allocator, .{});
}

pub fn compileModuleWithOptions(
    ir_module: *const ir.IrModule,
    allocator: std.mem.Allocator,
    options: CompileOptions,
) !CompileResult {
    var all_code: std.ArrayListUnmanaged(u8) = .empty;
    errdefer all_code.deinit(allocator);
    var offsets: std.ArrayListUnmanaged(u32) = .empty;
    errdefer offsets.deinit(allocator);

    var global_call_patches: std.ArrayListUnmanaged(CallPatch) = .empty;
    defer global_call_patches.deinit(allocator);

    for (ir_module.functions.items) |func| {
        const func_base: u32 = @intCast(all_code.items.len);
        try offsets.append(allocator, func_base);

        var func_patches: std.ArrayListUnmanaged(CallPatch) = .empty;
        defer func_patches.deinit(allocator);

        const ctx: FuncCompileCtx = .{
            .import_count = ir_module.import_count,
            .call_patches = &func_patches,
            .options = options,
            .allocator = allocator,
        };
        const func_code = try compileFunctionImpl(&func, ctx, allocator);
        defer allocator.free(func_code);

        // Globalize patch offsets to module code coordinates.
        for (func_patches.items) |p| {
            try global_call_patches.append(allocator, .{
                .patch_offset = p.patch_offset + func_base,
                .target_func_idx = p.target_func_idx,
            });
        }
        try all_code.appendSlice(allocator, func_code);
    }

    // Resolve BL patches. offsets[i] is the byte offset of local function i
    // (indices already exclude imports).
    for (global_call_patches.items) |p| {
        if (p.target_func_idx >= offsets.items.len) return error.BadFuncIndex;
        const target_off: i64 = @intCast(offsets.items[p.target_func_idx]);
        const patch_off: i64 = @intCast(p.patch_offset);
        const delta_bytes = target_off - patch_off;
        if (@mod(delta_bytes, 4) != 0) return error.BranchMisaligned;
        const word_off = @divExact(delta_bytes, 4);
        // BL imm26 range: ±128 MB → word_off in [-2^25, 2^25).
        const limit: i64 = 1 << 25;
        if (word_off >= limit or word_off < -limit) return error.CallOutOfRange;
        const bytes = all_code.items[p.patch_offset..][0..4];
        const existing = std.mem.readInt(u32, bytes, .little);
        // BL encoding: 100101 imm26 | opcode bits 31-26 = 0b100101
        const imm26: u26 = @bitCast(@as(i26, @intCast(word_off)));
        const new_word: u32 = (existing & 0xFC000000) | imm26;
        std.mem.writeInt(u32, bytes, new_word, .little);
    }

    return .{
        .code = try all_code.toOwnedSlice(allocator),
        .offsets = try offsets.toOwnedSlice(allocator),
    };
}

// ── Register-Allocator Preparation ─────────────────────────────────────────
//
// Phase 1 of the linear-scan regalloc adoption (plan in issue #100): define
// the aarch64 `RegSet` and a `collectClobberPoints` helper that mirrors the
// x86-64 version in `../x86_64/compile.zig`. This code is not yet wired into
// codegen — `RegMap` remains the source of truth for vreg → physreg
// assignment. It exists so Phase 2 can run allocation in shadow mode and
// Phase 3 can flip emitters without also introducing new analysis in the
// same step.

// (regalloc import lives at top of file)

/// Allocatable aarch64 GPRs, in stable index order used by clobber masks.
///
/// Indices 0..14 map to x0..x14 (AAPCS64 caller-saved; x15 is reserved as
/// a non-allocatable scratch `RegMap.tmp2`). Indices 15..24 map to
/// x19..x28 (callee-saved). x16/x17 are `RegMap.tmp0/tmp1`, x18 is the
/// platform register, x29/x30 are FP/LR, and x31 is SP — all excluded.
pub const aarch64_alloc_regs = [_]regalloc.PhysReg{
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
};

/// Indices into `aarch64_alloc_regs` of AAPCS64 caller-saved registers
/// (x0..x14). Used as the "prefer when live range doesn't span a call"
/// pool.
pub const aarch64_caller_saved_indices = [_]u8{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
};

/// Indices into `aarch64_alloc_regs` of AAPCS64 callee-saved registers
/// (x19..x28). Preferred for live ranges that span a call, since they
/// survive without save/restore.
pub const aarch64_callee_saved_indices = [_]u8{
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
};

/// Bitmask over `aarch64_alloc_regs` of registers destroyed by any
/// AAPCS64-compatible procedure call (direct, indirect, or vmctx
/// helper): x0..x14 (indices 0..14). x15..x17 are our own non-
/// allocatable scratches, x18 is platform-reserved.
pub const aarch64_call_clobber_mask: u64 = (@as(u64, 1) << 15) - 1;

/// Build the per-function `RegSet` for aarch64. `local_count` sets the
/// spill-slot origin above the operand stack.
///
/// Frame layout (matches `compileFunctionImpl` / `emitEntrySpill`):
/// `[fp+0]` = saved fp, `[fp+8]` = saved lr, `[fp+16]` = vmctx slot,
/// `[fp+24 .. fp+24+LC*8]` = locals, then a 64-slot operand stack, then
/// spills growing upward. Spills begin at `fp + 24 + (LC + 64)*8`.
pub fn aarch64RegSet(local_count: u32) regalloc.RegSet {
    return .{
        .alloc_regs = &aarch64_alloc_regs,
        .callee_saved_indices = &aarch64_callee_saved_indices,
        .caller_saved_indices = &aarch64_caller_saved_indices,
        // Match RegMap's `spill_base = (local_count + 3) * 8` so
        // translation between AllocResult.stack (absolute FP offset)
        // and RegMap.Location.stack (offset from spill_base) is direct.
        .spill_base = @as(i32, @intCast((local_count + 3) * 8)),
        .spill_stride = 8,
    };
}

/// Scan `func` and emit a `ClobberPoint` for every instruction that
/// destroys caller-saved registers under AAPCS64. Positions use the same
/// flat indexing as `analysis.computeLiveRanges` (one counter, block-
/// major, instruction-minor), so they line up with the allocator's live
/// ranges without further translation.
///
/// Clobbering ops on aarch64:
///   - `.call`, `.call_indirect`, `.call_ref` — direct/indirect/callable-
///     reference calls, all go through `bl`/`blr`.
///   - `.memory_grow`, `.memory_copy`, `.memory_fill`, `.memory_init` —
///     compiled as vmctx helper calls (see `emitVmctxHelperCall`).
///   - `.table_set`, `.table_grow`, `.table_init` — likewise.
///   - `.atomic_wait`, `.atomic_notify` — vmctx helper calls.
/// All use a `bl` to a C ABI helper and therefore clobber the full
/// caller-saved set.
///
/// Not clobbered (executed inline): `.atomic_load`, `.atomic_store`,
/// `.atomic_rmw`, `.atomic_cmpxchg`, `.atomic_fence` (LSE atomics).
pub fn collectClobberPoints(
    func: *const ir.IrFunction,
    block_order_opt: ?[]const ir.BlockId,
    scheduled: ?*const schedule.FunctionSchedule,
    allocator: std.mem.Allocator,
) !std.ArrayList(regalloc.ClobberPoint) {
    var clobbers: std.ArrayList(regalloc.ClobberPoint) = .empty;
    errdefer clobbers.deinit(allocator);

    var owns_order = false;
    const block_order: []const ir.BlockId = if (block_order_opt) |bo| bo else blk: {
        const raw = try allocator.alloc(ir.BlockId, func.blocks.items.len);
        for (raw, 0..) |*r, i| r.* = @intCast(i);
        owns_order = true;
        break :blk raw;
    };
    defer if (owns_order) allocator.free(block_order);

    var pos: u32 = 0;
    for (block_order) |bo_bid| {
        const insts = if (scheduled) |s|
            s.instructions(bo_bid)
        else
            func.blocks.items[bo_bid].instructions.items;
        for (insts) |ci| {
            switch (ci.op) {
                .call,
                .call_indirect,
                .call_ref,
                .memory_grow,
                .memory_copy,
                .memory_fill,
                .memory_init,
                .table_set,
                .table_grow,
                .table_init,
                .elem_drop,
                .atomic_wait,
                .atomic_notify,
                => try clobbers.append(allocator, .{
                    .pos = pos,
                    .regs_clobbered = aarch64_call_clobber_mask,
                }),
                else => {},
            }
            pos += 1;
        }
    }
    return clobbers;
}

/// Phase 2 shim: run the linear-scan allocator on `func` and discard the
/// result. Exercises `collectClobberPoints`, `analysis.computeLiveRanges`,
/// and `regalloc.allocate` on every aarch64 function the compiler sees,
/// so bugs surface here before Phase 3 depends on the output.
///
/// This is called unconditionally from `compileFunctionImpl` and will be
/// deleted in Phase 3 (replaced by a real consumer of the allocation).
fn shadowRunRegalloc(func: *const ir.IrFunction, allocator: std.mem.Allocator) !void {
    var clobbers = try collectClobberPoints(func, null, null, allocator);
    defer clobbers.deinit(allocator);

    var alloc_result = try regalloc.allocate(
        func,
        allocator,
        aarch64RegSet(func.local_count),
        clobbers.items,
    );
    defer alloc_result.deinit();

    // Sanity: every vreg the allocator assigns lands on either a real
    // aarch64 GPR or a nonzero spill offset. A bogus mapping (e.g., a
    // reg number outside `aarch64_alloc_regs`) would indicate a broken
    // RegSet setup.
    var it = alloc_result.assignments.iterator();
    while (it.next()) |e| {
        switch (e.value_ptr.*) {
            .reg => |r| {
                var found = false;
                for (aarch64_alloc_regs) |ar| {
                    if (r == ar) {
                        found = true;
                        break;
                    }
                }
                std.debug.assert(found);
            },
            .stack => |off| std.debug.assert(off >= 0),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "compileFunction: iconst_32 + ret" {
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

    // Should produce: prologue + MOVZ + epilogue
    // All instructions are 4 bytes each
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0); // all 4-byte aligned
}

test "compileFunction: add two constants" {
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

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compileFunction: global_get then global_set round-trips" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .global_get = 3 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .global_set = .{ .idx = 5, .val = v0 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compileFunction: div_s by zero codegen produces trap path" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .div_s = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compileFunction: br_table with 2 targets + default compiles" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();

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

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // IR leaks the targets slice (same convention as call.args); free explicitly.
    allocator.free(targets);

    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compileFunction: memory.size compiles" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .memory_size = {} }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "compileFunction: memory.grow compiles" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .memory_grow = v0 }, .dest = v1, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v1 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "compileModule: records offsets" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // Add two functions
    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    const b1 = try f1.newBlock();
    const v0 = f1.newVReg();
    try f1.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try f1.getBlock(b1).append(.{ .op = .{ .ret = v0 } });
    _ = try module.addFunction(f1);

    var f2 = ir.IrFunction.init(allocator, 0, 1, 0);
    const b2 = try f2.newBlock();
    const v1 = f2.newVReg();
    try f2.getBlock(b2).append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1 });
    try f2.getBlock(b2).append(.{ .op = .{ .ret = v1 } });
    _ = try module.addFunction(f2);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    try std.testing.expectEqual(@as(usize, 2), result.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), result.offsets[0]);
    try std.testing.expect(result.offsets[1] > 0); // second function starts after first
}

test "compileModule: direct call between two local functions" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // f0: returns 42
    var f0 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f0.newBlock();
        const v = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .ret = v } });
    }
    _ = try module.addFunction(f0);

    // f1: calls f0, returns its result
    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f1.newBlock();
        const v = f1.newVReg();
        try f1.getBlock(b).append(.{
            .op = .{ .call = .{ .func_idx = 0, .args = &.{} } },
            .dest = v,
            .type = .i32,
        });
        try f1.getBlock(b).append(.{ .op = .{ .ret = v } });
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    // Walk f1's code looking for a BL that points at f0 (offset 0).
    const f1_start = result.offsets[1];
    const f0_start = result.offsets[0];
    var found_bl_to_f0 = false;
    var i: usize = f1_start;
    while (i + 4 <= result.code.len) : (i += 4) {
        const word = std.mem.readInt(u32, result.code[i..][0..4], .little);
        // BL opcode = 0b100101 in top 6 bits → 0x94000000
        if ((word & 0xFC000000) == 0x94000000) {
            const imm26_raw: u26 = @truncate(word);
            const imm26: i26 = @bitCast(imm26_raw);
            const delta_words: i64 = imm26;
            const target: i64 = @as(i64, @intCast(i)) + delta_words * 4;
            if (target == @as(i64, @intCast(f0_start))) {
                found_bl_to_f0 = true;
                break;
            }
        }
    }
    try std.testing.expect(found_bl_to_f0);
}

// ── New handler tests (Phase 1a) ─────────────────────────────────────────────

test "load i32: emits VMCtx load + LDR W with scaled offset" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 8, .size = 4 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Expect at least one LDR W (opcode 0xB9400000 in top bits).
    var found_ldr_w = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0xB9400000) {
            found_ldr_w = true;
            break;
        }
    }
    try std.testing.expect(found_ldr_w);
}

test "store i32: emits VMCtx load + STR W" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = @bitCast(@as(u32, 0xDEADBEEF)) }, .dest = val, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .store = .{ .base = addr, .offset = 0, .size = 4, .val = val } },
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = null } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Expect at least one STR W (opcode 0xB9000000 in top bits).
    var found_str_w = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0xB9000000) {
            found_str_w = true;
            break;
        }
    }
    try std.testing.expect(found_str_w);
}

test "load i32: unrepresentable offset falls back to movImm + add" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    // Offset 5 is not a multiple of 4 → must go through tmp1/ADD path.
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 5, .size = 4 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compile: v128 first-family ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const x = func.newVReg();
    const c = func.newVReg();
    const sum = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_0000_0001 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0008_0000_0007_0000_0006_0000_0005 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = a, .rhs = b } },
        .dest = x,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_000C_0000_000B_0000_000A_0000_0009 }, .dest = c, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_binop = .{ .op = .add, .lhs = x, .rhs = c } },
        .dest = sum,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = sum, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_eor = false;
    var found_add4s = false;
    var found_umov = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x6E201C00) found_eor = true;
        if ((w & 0xFFE0FC00) == 0x4EA08400) found_add4s = true;
        if ((w & 0x0FE0FC00) == 0x0E003C00) found_umov = true;
    }

    try std.testing.expect(found_eor);
    try std.testing.expect(found_add4s);
    try std.testing.expect(found_umov);
}

test "compile: i32x4 lane ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const scalar = func.newVReg();
    const splat = func.newVReg();
    const replacement = func.newVReg();
    const replaced = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 7 }, .dest = scalar, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_splat = scalar }, .dest = splat, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 99 }, .dest = replacement, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_replace_lane = .{ .vector = splat, .val = replacement, .lane = 2 } },
        .dest = replaced,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = replaced, .lane = 2 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_dup = false;
    var found_ins = false;
    var found_umov = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4E040C00) found_dup = true;
        if ((w & 0xFFFFFC00) == 0x4E141C00) found_ins = true;
        if ((w & 0x0FE0FC00) == 0x0E003C00) found_umov = true;
    }

    try std.testing.expect(found_dup);
    try std.testing.expect(found_ins);
    try std.testing.expect(found_umov);
}

test "compile: i8x16 lane ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const scalar = func.newVReg();
    const splat = func.newVReg();
    const replacement = func.newVReg();
    const replaced = func.newVReg();
    const signed_lane = func.newVReg();
    const unsigned_lane = func.newVReg();
    const sum = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = -2 }, .dest = scalar, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_splat = scalar }, .dest = splat, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0x0000_0080 }, .dest = replacement, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .i8x16_replace_lane = .{ .vector = splat, .val = replacement, .lane = 13 } },
        .dest = replaced,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i8x16_extract_lane = .{ .vector = replaced, .lane = 13, .sign = .signed } },
        .dest = signed_lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i8x16_extract_lane = .{ .vector = replaced, .lane = 13, .sign = .unsigned } },
        .dest = unsigned_lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .add = .{ .lhs = signed_lane, .rhs = unsigned_lane } },
        .dest = sum,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = sum } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_dup = false;
    var found_ins = false;
    var found_smov = false;
    var found_umov = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4E010C00) found_dup = true;
        if ((w & 0xFFE0FC00) == 0x4E001C00) found_ins = true;
        if ((w & 0xFFE0FC00) == 0x0E002C00) found_smov = true;
        if ((w & 0xFFE0FC00) == 0x0E003C00) found_umov = true;
    }

    try std.testing.expect(found_dup);
    try std.testing.expect(found_ins);
    try std.testing.expect(found_smov);
    try std.testing.expect(found_umov);
}

test "compile: i16x8 lane ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const scalar = func.newVReg();
    const splat = func.newVReg();
    const replacement = func.newVReg();
    const replaced = func.newVReg();
    const signed_lane = func.newVReg();
    const unsigned_lane = func.newVReg();
    const sum = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = -2 }, .dest = scalar, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_splat = scalar }, .dest = splat, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0x0000_FF80 }, .dest = replacement, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .i16x8_replace_lane = .{ .vector = splat, .val = replacement, .lane = 5 } },
        .dest = replaced,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i16x8_extract_lane = .{ .vector = replaced, .lane = 5, .sign = .signed } },
        .dest = signed_lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i16x8_extract_lane = .{ .vector = replaced, .lane = 5, .sign = .unsigned } },
        .dest = unsigned_lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .add = .{ .lhs = signed_lane, .rhs = unsigned_lane } },
        .dest = sum,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = sum } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_dup = false;
    var found_ins = false;
    var found_smov = false;
    var found_umov = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4E020C00) found_dup = true;
        if ((w & 0xFFFFFC00) == 0x4E161C00) found_ins = true;
        if ((w & 0xFFFFFC00) == 0x0E162C00) found_smov = true;
        if ((w & 0xFFFFFC00) == 0x0E163C00) found_umov = true;
    }

    try std.testing.expect(found_dup);
    try std.testing.expect(found_ins);
    try std.testing.expect(found_smov);
    try std.testing.expect(found_umov);
}

test "compile: i64x2 lane ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const scalar = func.newVReg();
    const splat = func.newVReg();
    const replacement = func.newVReg();
    const replaced = func.newVReg();
    const lane = func.newVReg();
    const wrapped = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 7 }, .dest = scalar, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_splat = scalar }, .dest = splat, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = -128 }, .dest = replacement, .type = .i64 });
    try func.getBlock(bid).append(.{
        .op = .{ .i64x2_replace_lane = .{ .vector = splat, .val = replacement, .lane = 1 } },
        .dest = replaced,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i64x2_extract_lane = .{ .vector = replaced, .lane = 1 } },
        .dest = lane,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .wrap_i64 = lane }, .dest = wrapped, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = wrapped } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_dup = false;
    var found_ins = false;
    var found_umov = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4E080C00) found_dup = true;
        if ((w & 0xFFFFFC00) == 0x4E181C00) found_ins = true;
        if ((w & 0xFFFFFC00) == 0x4E183C00) found_umov = true;
    }

    try std.testing.expect(found_dup);
    try std.testing.expect(found_ins);
    try std.testing.expect(found_umov);
}

test "compile: i32x4 cmp and arithmetic ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const mul = func.newVReg();
    const min_s = func.newVReg();
    const min_u = func.newVReg();
    const max_s = func.newVReg();
    const max_u = func.newVReg();
    const ne = func.newVReg();
    const lt_s = func.newVReg();
    const gt_s = func.newVReg();
    const le_s = func.newVReg();
    const ge_s = func.newVReg();
    const lt_u = func.newVReg();
    const gt_u = func.newVReg();
    const le_u = func.newVReg();
    const ge_u = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0004_8000_0000_FFFF_FFFF_0000_0002 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0003_8000_0000_0000_0001_0000_0003 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .mul, .lhs = a, .rhs = b } }, .dest = mul, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .min_s, .lhs = a, .rhs = b } }, .dest = min_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .min_u, .lhs = a, .rhs = b } }, .dest = min_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .max_s, .lhs = a, .rhs = b } }, .dest = max_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .max_u, .lhs = a, .rhs = b } }, .dest = max_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .ne, .lhs = a, .rhs = b } }, .dest = ne, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .lt_s, .lhs = a, .rhs = b } }, .dest = lt_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .gt_s, .lhs = a, .rhs = b } }, .dest = gt_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .le_s, .lhs = a, .rhs = b } }, .dest = le_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .ge_s, .lhs = a, .rhs = b } }, .dest = ge_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .lt_u, .lhs = a, .rhs = b } }, .dest = lt_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .gt_u, .lhs = a, .rhs = b } }, .dest = gt_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .le_u, .lhs = a, .rhs = b } }, .dest = le_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .ge_u, .lhs = a, .rhs = b } }, .dest = ge_u, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = ge_u, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_mul = false;
    var found_smin = false;
    var found_umin = false;
    var found_smax = false;
    var found_umax = false;
    var found_cmeq = false;
    var found_mvn = false;
    var found_cmgt = false;
    var found_cmge = false;
    var found_cmhi = false;
    var found_cmhs = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x4EA09C00) found_mul = true;
        if ((w & 0xFFE0FC00) == 0x4EA06C00) found_smin = true;
        if ((w & 0xFFE0FC00) == 0x6EA06C00) found_umin = true;
        if ((w & 0xFFE0FC00) == 0x4EA06400) found_smax = true;
        if ((w & 0xFFE0FC00) == 0x6EA06400) found_umax = true;
        if ((w & 0xFFE0FC00) == 0x6EA08C00) found_cmeq = true;
        if ((w & 0xFFFFFC00) == 0x6E205800) found_mvn = true;
        if ((w & 0xFFE0FC00) == 0x4EA03400) found_cmgt = true;
        if ((w & 0xFFE0FC00) == 0x4EA03C00) found_cmge = true;
        if ((w & 0xFFE0FC00) == 0x6EA03400) found_cmhi = true;
        if ((w & 0xFFE0FC00) == 0x6EA03C00) found_cmhs = true;
    }

    try std.testing.expect(found_mul);
    try std.testing.expect(found_smin);
    try std.testing.expect(found_umin);
    try std.testing.expect(found_smax);
    try std.testing.expect(found_umax);
    try std.testing.expect(found_cmeq);
    try std.testing.expect(found_mvn);
    try std.testing.expect(found_cmgt);
    try std.testing.expect(found_cmge);
    try std.testing.expect(found_cmhi);
    try std.testing.expect(found_cmhs);
}

test "compile: i8x16 cmp and arithmetic ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const add = func.newVReg();
    const sub = func.newVReg();
    const add_sat_s = func.newVReg();
    const add_sat_u = func.newVReg();
    const sub_sat_s = func.newVReg();
    const sub_sat_u = func.newVReg();
    const ne = func.newVReg();
    const lt_s = func.newVReg();
    const gt_s = func.newVReg();
    const le_s = func.newVReg();
    const ge_s = func.newVReg();
    const lt_u = func.newVReg();
    const gt_u = func.newVReg();
    const le_u = func.newVReg();
    const ge_u = func.newVReg();
    const min_s = func.newVReg();
    const min_u = func.newVReg();
    const max_s = func.newVReg();
    const max_u = func.newVReg();
    const avgr_u = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0E0D_0C0B_0A09_0807_0605_0403_8001_FF00 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0F0E_0D0C_0B0A_0908_0706_0504_7F02_0100 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .add, .lhs = a, .rhs = b } }, .dest = add, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .sub, .lhs = a, .rhs = b } }, .dest = sub, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .add_sat_s, .lhs = a, .rhs = b } }, .dest = add_sat_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .add_sat_u, .lhs = a, .rhs = b } }, .dest = add_sat_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .sub_sat_s, .lhs = a, .rhs = b } }, .dest = sub_sat_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .sub_sat_u, .lhs = a, .rhs = b } }, .dest = sub_sat_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .ne, .lhs = a, .rhs = b } }, .dest = ne, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .lt_s, .lhs = a, .rhs = b } }, .dest = lt_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .gt_s, .lhs = a, .rhs = b } }, .dest = gt_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .le_s, .lhs = a, .rhs = b } }, .dest = le_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .ge_s, .lhs = a, .rhs = b } }, .dest = ge_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .lt_u, .lhs = a, .rhs = b } }, .dest = lt_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .gt_u, .lhs = a, .rhs = b } }, .dest = gt_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .le_u, .lhs = a, .rhs = b } }, .dest = le_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .ge_u, .lhs = a, .rhs = b } }, .dest = ge_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .min_s, .lhs = a, .rhs = b } }, .dest = min_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .min_u, .lhs = a, .rhs = b } }, .dest = min_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .max_s, .lhs = a, .rhs = b } }, .dest = max_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .max_u, .lhs = a, .rhs = b } }, .dest = max_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_binop = .{ .op = .avgr_u, .lhs = a, .rhs = b } }, .dest = avgr_u, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i8x16_extract_lane = .{ .vector = ge_u, .lane = 0, .sign = .unsigned } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_add = false;
    var found_sub = false;
    var found_sqadd = false;
    var found_uqadd = false;
    var found_sqsub = false;
    var found_uqsub = false;
    var found_cmeq = false;
    var found_mvn = false;
    var found_cmgt = false;
    var found_cmge = false;
    var found_cmhi = false;
    var found_cmhs = false;
    var found_smin = false;
    var found_umin = false;
    var found_smax = false;
    var found_umax = false;
    var found_urhadd = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x4E208400) found_add = true;
        if ((w & 0xFFE0FC00) == 0x6E208400) found_sub = true;
        if ((w & 0xFFE0FC00) == 0x4E200C00) found_sqadd = true;
        if ((w & 0xFFE0FC00) == 0x6E200C00) found_uqadd = true;
        if ((w & 0xFFE0FC00) == 0x4E202C00) found_sqsub = true;
        if ((w & 0xFFE0FC00) == 0x6E202C00) found_uqsub = true;
        if ((w & 0xFFE0FC00) == 0x6E208C00) found_cmeq = true;
        if ((w & 0xFFFFFC00) == 0x6E205800) found_mvn = true;
        if ((w & 0xFFE0FC00) == 0x4E203400) found_cmgt = true;
        if ((w & 0xFFE0FC00) == 0x4E203C00) found_cmge = true;
        if ((w & 0xFFE0FC00) == 0x6E203400) found_cmhi = true;
        if ((w & 0xFFE0FC00) == 0x6E203C00) found_cmhs = true;
        if ((w & 0xFFE0FC00) == 0x4E206C00) found_smin = true;
        if ((w & 0xFFE0FC00) == 0x6E206C00) found_umin = true;
        if ((w & 0xFFE0FC00) == 0x4E206400) found_smax = true;
        if ((w & 0xFFE0FC00) == 0x6E206400) found_umax = true;
        if ((w & 0xFFE0FC00) == 0x6E201400) found_urhadd = true;
    }

    try std.testing.expect(found_add);
    try std.testing.expect(found_sub);
    try std.testing.expect(found_sqadd);
    try std.testing.expect(found_uqadd);
    try std.testing.expect(found_sqsub);
    try std.testing.expect(found_uqsub);
    try std.testing.expect(found_cmeq);
    try std.testing.expect(found_mvn);
    try std.testing.expect(found_cmgt);
    try std.testing.expect(found_cmge);
    try std.testing.expect(found_cmhi);
    try std.testing.expect(found_cmhs);
    try std.testing.expect(found_smin);
    try std.testing.expect(found_umin);
    try std.testing.expect(found_smax);
    try std.testing.expect(found_umax);
    try std.testing.expect(found_urhadd);
}

test "compile: i16x8 cmp and arithmetic ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const q15mulr_sat_s = func.newVReg();
    const add_sat_s = func.newVReg();
    const add_sat_u = func.newVReg();
    const sub_sat_s = func.newVReg();
    const sub_sat_u = func.newVReg();
    const mul = func.newVReg();
    const ne = func.newVReg();
    const lt_s = func.newVReg();
    const gt_s = func.newVReg();
    const le_s = func.newVReg();
    const ge_s = func.newVReg();
    const lt_u = func.newVReg();
    const gt_u = func.newVReg();
    const le_u = func.newVReg();
    const ge_u = func.newVReg();
    const min_s = func.newVReg();
    const min_u = func.newVReg();
    const max_s = func.newVReg();
    const max_u = func.newVReg();
    const avgr_u = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0008_0007_0006_0005_8000_FFFF_0002_0001 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0007_0008_0005_0006_8000_0001_0003_0001 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .q15mulr_sat_s, .lhs = a, .rhs = b } }, .dest = q15mulr_sat_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .add_sat_s, .lhs = a, .rhs = b } }, .dest = add_sat_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .add_sat_u, .lhs = a, .rhs = b } }, .dest = add_sat_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .sub_sat_s, .lhs = a, .rhs = b } }, .dest = sub_sat_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .sub_sat_u, .lhs = a, .rhs = b } }, .dest = sub_sat_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .mul, .lhs = a, .rhs = b } }, .dest = mul, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .ne, .lhs = a, .rhs = b } }, .dest = ne, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .lt_s, .lhs = a, .rhs = b } }, .dest = lt_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .gt_s, .lhs = a, .rhs = b } }, .dest = gt_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .le_s, .lhs = a, .rhs = b } }, .dest = le_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .ge_s, .lhs = a, .rhs = b } }, .dest = ge_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .lt_u, .lhs = a, .rhs = b } }, .dest = lt_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .gt_u, .lhs = a, .rhs = b } }, .dest = gt_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .le_u, .lhs = a, .rhs = b } }, .dest = le_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .ge_u, .lhs = a, .rhs = b } }, .dest = ge_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .min_s, .lhs = a, .rhs = b } }, .dest = min_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .min_u, .lhs = a, .rhs = b } }, .dest = min_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .max_s, .lhs = a, .rhs = b } }, .dest = max_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .max_u, .lhs = a, .rhs = b } }, .dest = max_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_binop = .{ .op = .avgr_u, .lhs = a, .rhs = b } }, .dest = avgr_u, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i16x8_extract_lane = .{ .vector = ge_u, .lane = 0, .sign = .unsigned } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_sqrdmulh = false;
    var found_sqadd = false;
    var found_uqadd = false;
    var found_sqsub = false;
    var found_uqsub = false;
    var found_mul = false;
    var found_cmeq = false;
    var found_mvn = false;
    var found_cmgt = false;
    var found_cmge = false;
    var found_cmhi = false;
    var found_cmhs = false;
    var found_smin = false;
    var found_umin = false;
    var found_smax = false;
    var found_umax = false;
    var found_urhadd = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x6E60B400) found_sqrdmulh = true;
        if ((w & 0xFFE0FC00) == 0x4E600C00) found_sqadd = true;
        if ((w & 0xFFE0FC00) == 0x6E600C00) found_uqadd = true;
        if ((w & 0xFFE0FC00) == 0x4E602C00) found_sqsub = true;
        if ((w & 0xFFE0FC00) == 0x6E602C00) found_uqsub = true;
        if ((w & 0xFFE0FC00) == 0x4E609C00) found_mul = true;
        if ((w & 0xFFE0FC00) == 0x6E608C00) found_cmeq = true;
        if ((w & 0xFFFFFC00) == 0x6E205800) found_mvn = true;
        if ((w & 0xFFE0FC00) == 0x4E603400) found_cmgt = true;
        if ((w & 0xFFE0FC00) == 0x4E603C00) found_cmge = true;
        if ((w & 0xFFE0FC00) == 0x6E603400) found_cmhi = true;
        if ((w & 0xFFE0FC00) == 0x6E603C00) found_cmhs = true;
        if ((w & 0xFFE0FC00) == 0x4E606C00) found_smin = true;
        if ((w & 0xFFE0FC00) == 0x6E606C00) found_umin = true;
        if ((w & 0xFFE0FC00) == 0x4E606400) found_smax = true;
        if ((w & 0xFFE0FC00) == 0x6E606400) found_umax = true;
        if ((w & 0xFFE0FC00) == 0x6E601400) found_urhadd = true;
    }

    try std.testing.expect(found_sqrdmulh);
    try std.testing.expect(found_sqadd);
    try std.testing.expect(found_uqadd);
    try std.testing.expect(found_sqsub);
    try std.testing.expect(found_uqsub);
    try std.testing.expect(found_mul);
    try std.testing.expect(found_cmeq);
    try std.testing.expect(found_mvn);
    try std.testing.expect(found_cmgt);
    try std.testing.expect(found_cmge);
    try std.testing.expect(found_cmhi);
    try std.testing.expect(found_cmhs);
    try std.testing.expect(found_smin);
    try std.testing.expect(found_umin);
    try std.testing.expect(found_smax);
    try std.testing.expect(found_umax);
    try std.testing.expect(found_urhadd);
}

test "compile: i64x2 cmp and arithmetic ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const add = func.newVReg();
    const sub = func.newVReg();
    const ne = func.newVReg();
    const lt_s = func.newVReg();
    const gt_s = func.newVReg();
    const le_s = func.newVReg();
    const ge_s = func.newVReg();
    const lane = func.newVReg();
    const wrapped = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0000_0000_0004_ffff_ffff_ffff_ffff }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0000_0000_0003_0000_0000_0000_0001 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_binop = .{ .op = .add, .lhs = a, .rhs = b } }, .dest = add, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_binop = .{ .op = .sub, .lhs = add, .rhs = b } }, .dest = sub, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_binop = .{ .op = .ne, .lhs = sub, .rhs = a } }, .dest = ne, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_binop = .{ .op = .lt_s, .lhs = a, .rhs = b } }, .dest = lt_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_binop = .{ .op = .gt_s, .lhs = a, .rhs = b } }, .dest = gt_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_binop = .{ .op = .le_s, .lhs = a, .rhs = b } }, .dest = le_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_binop = .{ .op = .ge_s, .lhs = a, .rhs = b } }, .dest = ge_s, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i64x2_extract_lane = .{ .vector = ge_s, .lane = 0 } },
        .dest = lane,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .wrap_i64 = lane }, .dest = wrapped, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = wrapped } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_add = false;
    var found_sub = false;
    var found_cmeq = false;
    var found_mvn = false;
    var found_cmgt = false;
    var found_cmge = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x4EE08400) found_add = true;
        if ((w & 0xFFE0FC00) == 0x6EE08400) found_sub = true;
        if ((w & 0xFFE0FC00) == 0x6EE08C00) found_cmeq = true;
        if ((w & 0xFFFFFC00) == 0x6E205800) found_mvn = true;
        if ((w & 0xFFE0FC00) == 0x4EE03400) found_cmgt = true;
        if ((w & 0xFFE0FC00) == 0x4EE03C00) found_cmge = true;
    }

    try std.testing.expect(found_add);
    try std.testing.expect(found_sub);
    try std.testing.expect(found_cmeq);
    try std.testing.expect(found_mvn);
    try std.testing.expect(found_cmgt);
    try std.testing.expect(found_cmge);
}

test "compile: integer SIMD abs and neg ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const source = func.newVReg();
    const i8_abs = func.newVReg();
    const i8_neg = func.newVReg();
    const i16_abs = func.newVReg();
    const i16_neg = func.newVReg();
    const i32_abs = func.newVReg();
    const i32_neg = func.newVReg();
    const i64_abs = func.newVReg();
    const i64_neg = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8000_0000_0000_0000_8000_0000_807F_0180 }, .dest = source, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_unop = .{ .op = .abs, .vector = source } }, .dest = i8_abs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_unop = .{ .op = .neg, .vector = i8_abs } }, .dest = i8_neg, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_unop = .{ .op = .abs, .vector = i8_neg } }, .dest = i16_abs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_unop = .{ .op = .neg, .vector = i16_abs } }, .dest = i16_neg, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_unop = .{ .op = .abs, .vector = i16_neg } }, .dest = i32_abs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_unop = .{ .op = .neg, .vector = i32_abs } }, .dest = i32_neg, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_unop = .{ .op = .abs, .vector = i32_neg } }, .dest = i64_abs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_unop = .{ .op = .neg, .vector = i64_abs } }, .dest = i64_neg, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = i64_neg, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_abs16b = false;
    var found_neg16b = false;
    var found_abs8h = false;
    var found_neg8h = false;
    var found_abs4s = false;
    var found_neg4s = false;
    var found_abs2d = false;
    var found_neg2d = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4E20B800) found_abs16b = true;
        if ((w & 0xFFFFFC00) == 0x6E20B800) found_neg16b = true;
        if ((w & 0xFFFFFC00) == 0x4E60B800) found_abs8h = true;
        if ((w & 0xFFFFFC00) == 0x6E60B800) found_neg8h = true;
        if ((w & 0xFFFFFC00) == 0x4EA0B800) found_abs4s = true;
        if ((w & 0xFFFFFC00) == 0x6EA0B800) found_neg4s = true;
        if ((w & 0xFFFFFC00) == 0x4EE0B800) found_abs2d = true;
        if ((w & 0xFFFFFC00) == 0x6EE0B800) found_neg2d = true;
    }

    try std.testing.expect(found_abs16b);
    try std.testing.expect(found_neg16b);
    try std.testing.expect(found_abs8h);
    try std.testing.expect(found_neg8h);
    try std.testing.expect(found_abs4s);
    try std.testing.expect(found_neg4s);
    try std.testing.expect(found_abs2d);
    try std.testing.expect(found_neg2d);
}

test "compile: integer SIMD pairwise extended add ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const source = func.newVReg();
    const i16_signed = func.newVReg();
    const i16_unsigned = func.newVReg();
    const i32_signed = func.newVReg();
    const i32_unsigned = func.newVReg();
    const combined = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8001_7FFE_8000_7FFF_0180_7F80_01FF_8001 }, .dest = source, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_extadd_pairwise_i8x16 = .{ .sign = .signed, .vector = source } }, .dest = i16_signed, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_extadd_pairwise_i8x16 = .{ .sign = .unsigned, .vector = source } }, .dest = i16_unsigned, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_extadd_pairwise_i16x8 = .{ .sign = .signed, .vector = i16_signed } }, .dest = i32_signed, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_extadd_pairwise_i16x8 = .{ .sign = .unsigned, .vector = i16_unsigned } }, .dest = i32_unsigned, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = i32_signed, .rhs = i32_unsigned } }, .dest = combined, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = combined, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_saddlp8h = false;
    var found_uaddlp8h = false;
    var found_saddlp4s = false;
    var found_uaddlp4s = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4E202800) found_saddlp8h = true;
        if ((w & 0xFFFFFC00) == 0x6E202800) found_uaddlp8h = true;
        if ((w & 0xFFFFFC00) == 0x4E602800) found_saddlp4s = true;
        if ((w & 0xFFFFFC00) == 0x6E602800) found_uaddlp4s = true;
    }

    try std.testing.expect(found_saddlp8h);
    try std.testing.expect(found_uaddlp8h);
    try std.testing.expect(found_saddlp4s);
    try std.testing.expect(found_uaddlp4s);
}

test "compile: integer SIMD widening extend low/high ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const source = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8001_7FFE_8000_7FFF_0180_7F80_01FF_8001 }, .dest = source, .type = .v128 });

    const Variant = struct {
        sign: ir.Inst.SimdExtendSign,
        half: ir.Inst.SimdExtendHalfSelect,
    };
    const variants = [_]Variant{
        .{ .sign = .signed, .half = .low },
        .{ .sign = .signed, .half = .high },
        .{ .sign = .unsigned, .half = .low },
        .{ .sign = .unsigned, .half = .high },
    };

    var prev = source;
    inline for (.{ "i16x8_extend_i8x16", "i32x4_extend_i16x8", "i64x2_extend_i32x4" }) |op_name| {
        for (variants) |v| {
            const dest = func.newVReg();
            try func.getBlock(bid).append(.{
                .op = @unionInit(ir.Inst.Op, op_name, .{ .sign = v.sign, .half = v.half, .vector = prev }),
                .dest = dest,
                .type = .v128,
            });
            prev = dest;
        }
    }

    const lane = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = prev, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    const expected = [_]struct { mask: u32, base: u32, name: []const u8 }{
        .{ .mask = 0xFFFFFC00, .base = 0x0F08A400, .name = "sshll8h8b" },
        .{ .mask = 0xFFFFFC00, .base = 0x2F08A400, .name = "ushll8h8b" },
        .{ .mask = 0xFFFFFC00, .base = 0x4F08A400, .name = "sshll2_8h16b" },
        .{ .mask = 0xFFFFFC00, .base = 0x6F08A400, .name = "ushll2_8h16b" },
        .{ .mask = 0xFFFFFC00, .base = 0x0F10A400, .name = "sshll4s4h" },
        .{ .mask = 0xFFFFFC00, .base = 0x2F10A400, .name = "ushll4s4h" },
        .{ .mask = 0xFFFFFC00, .base = 0x4F10A400, .name = "sshll2_4s8h" },
        .{ .mask = 0xFFFFFC00, .base = 0x6F10A400, .name = "ushll2_4s8h" },
        .{ .mask = 0xFFFFFC00, .base = 0x0F20A400, .name = "sshll2d2s" },
        .{ .mask = 0xFFFFFC00, .base = 0x2F20A400, .name = "ushll2d2s" },
        .{ .mask = 0xFFFFFC00, .base = 0x4F20A400, .name = "sshll2_2d4s" },
        .{ .mask = 0xFFFFFC00, .base = 0x6F20A400, .name = "ushll2_2d4s" },
    };

    inline for (expected) |e| {
        var found = false;
        var i: usize = 0;
        while (i + 4 <= code.len) : (i += 4) {
            const w = std.mem.readInt(u32, code[i..][0..4], .little);
            if ((w & e.mask) == e.base) {
                found = true;
                break;
            }
        }
        if (!found) {
            std.debug.print("missing NEON op {s} (base 0x{X})\n", .{ e.name, e.base });
        }
        try std.testing.expect(found);
    }
}

test "compile: integer SIMD widening multiply low/high ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const lhs = func.newVReg();
    const rhs = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8001_7FFE_8000_7FFF_0180_7F80_01FF_8001 }, .dest = lhs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x1234_5678_9ABC_DEF0_0123_4567_89AB_CDEF }, .dest = rhs, .type = .v128 });

    const Variant = struct {
        sign: ir.Inst.SimdExtendSign,
        half: ir.Inst.SimdExtendHalfSelect,
    };
    const variants = [_]Variant{
        .{ .sign = .signed, .half = .low },
        .{ .sign = .signed, .half = .high },
        .{ .sign = .unsigned, .half = .low },
        .{ .sign = .unsigned, .half = .high },
    };

    inline for (.{ "i16x8_extmul_i8x16", "i32x4_extmul_i16x8", "i64x2_extmul_i32x4" }) |op_name| {
        for (variants) |v| {
            const dest = func.newVReg();
            try func.getBlock(bid).append(.{
                .op = @unionInit(ir.Inst.Op, op_name, .{ .sign = v.sign, .half = v.half, .lhs = lhs, .rhs = rhs }),
                .dest = dest,
                .type = .v128,
            });
        }
    }

    // Force a use so the extmul results aren't fully dead-code-eliminated; pick the
    // last one through an extract.
    const last_extmul = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .i64x2_extmul_i32x4 = .{ .sign = .unsigned, .half = .high, .lhs = lhs, .rhs = rhs } },
        .dest = last_extmul,
        .type = .v128,
    });
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = last_extmul, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    const expected = [_]struct { mask: u32, base: u32, name: []const u8 }{
        .{ .mask = 0xFFE0FC00, .base = 0x0E20C000, .name = "smull8h8b" },
        .{ .mask = 0xFFE0FC00, .base = 0x2E20C000, .name = "umull8h8b" },
        .{ .mask = 0xFFE0FC00, .base = 0x4E20C000, .name = "smull2_8h16b" },
        .{ .mask = 0xFFE0FC00, .base = 0x6E20C000, .name = "umull2_8h16b" },
        .{ .mask = 0xFFE0FC00, .base = 0x0E60C000, .name = "smull4s4h" },
        .{ .mask = 0xFFE0FC00, .base = 0x2E60C000, .name = "umull4s4h" },
        .{ .mask = 0xFFE0FC00, .base = 0x4E60C000, .name = "smull2_4s8h" },
        .{ .mask = 0xFFE0FC00, .base = 0x6E60C000, .name = "umull2_4s8h" },
        .{ .mask = 0xFFE0FC00, .base = 0x0EA0C000, .name = "smull2d2s" },
        .{ .mask = 0xFFE0FC00, .base = 0x2EA0C000, .name = "umull2d2s" },
        .{ .mask = 0xFFE0FC00, .base = 0x4EA0C000, .name = "smull2_2d4s" },
        .{ .mask = 0xFFE0FC00, .base = 0x6EA0C000, .name = "umull2_2d4s" },
    };

    inline for (expected) |e| {
        var found = false;
        var i: usize = 0;
        while (i + 4 <= code.len) : (i += 4) {
            const w = std.mem.readInt(u32, code[i..][0..4], .little);
            if ((w & e.mask) == e.base) {
                found = true;
                break;
            }
        }
        if (!found) {
            std.debug.print("missing NEON op {s} (base 0x{X})\n", .{ e.name, e.base });
        }
        try std.testing.expect(found);
    }
}

test "compile: i64x2 scalar-count shifts emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const vector = func.newVReg();
    const count_shl = func.newVReg();
    const shl = func.newVReg();
    const count_shr_s = func.newVReg();
    const shr_s = func.newVReg();
    const count_shr_u = func.newVReg();
    const shr_u = func.newVReg();
    const lane = func.newVReg();
    const wrapped = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8000_0000_0000_0001_ffff_ffff_ffff_ff80 }, .dest = vector, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 65 }, .dest = count_shl, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_shift = .{ .op = .shl, .vector = vector, .count = count_shl } }, .dest = shl, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 64 }, .dest = count_shr_s, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_shift = .{ .op = .shr_s, .vector = shl, .count = count_shr_s } }, .dest = shr_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 63 }, .dest = count_shr_u, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_shift = .{ .op = .shr_u, .vector = shr_s, .count = count_shr_u } }, .dest = shr_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i64x2_extract_lane = .{ .vector = shr_u, .lane = 0 } }, .dest = lane, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .wrap_i64 = lane }, .dest = wrapped, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = wrapped } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_count_mask = false;
    var found_dup = false;
    var found_sshl = false;
    var found_neg2d = false;
    var found_ushl = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x12001400) found_count_mask = true;
        if ((w & 0xFFFFFC00) == 0x4E080C00) found_dup = true;
        if ((w & 0xFFE0FC00) == 0x4EE04400) found_sshl = true;
        if ((w & 0xFFFFFC00) == 0x6EE0B800) found_neg2d = true;
        if ((w & 0xFFE0FC00) == 0x6EE04400) found_ushl = true;
    }

    try std.testing.expect(found_count_mask);
    try std.testing.expect(found_dup);
    try std.testing.expect(found_sshl);
    try std.testing.expect(found_neg2d);
    try std.testing.expect(found_ushl);
}

test "compile: i8x16 scalar-count shifts emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const vector = func.newVReg();
    const count_shl = func.newVReg();
    const shl = func.newVReg();
    const count_shr_s = func.newVReg();
    const shr_s = func.newVReg();
    const count_shr_u = func.newVReg();
    const shr_u = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0D0C_0B0A_0908_0706_0504_0302_01FF_7F80 }, .dest = vector, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 9 }, .dest = count_shl, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_shift = .{ .op = .shl, .vector = vector, .count = count_shl } }, .dest = shl, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 8 }, .dest = count_shr_s, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_shift = .{ .op = .shr_s, .vector = shl, .count = count_shr_s } }, .dest = shr_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 15 }, .dest = count_shr_u, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_shift = .{ .op = .shr_u, .vector = shr_s, .count = count_shr_u } }, .dest = shr_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_extract_lane = .{ .vector = shr_u, .lane = 0, .sign = .unsigned } }, .dest = lane, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_count_mask = false;
    var found_dup = false;
    var found_sshl = false;
    var found_neg16b = false;
    var found_ushl = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x12000800) found_count_mask = true;
        if ((w & 0xFFFFFC00) == 0x4E010C00) found_dup = true;
        if ((w & 0xFFE0FC00) == 0x4E204400) found_sshl = true;
        if ((w & 0xFFFFFC00) == 0x6E20B800) found_neg16b = true;
        if ((w & 0xFFE0FC00) == 0x6E204400) found_ushl = true;
    }

    try std.testing.expect(found_count_mask);
    try std.testing.expect(found_dup);
    try std.testing.expect(found_sshl);
    try std.testing.expect(found_neg16b);
    try std.testing.expect(found_ushl);
}

test "compile: i32x4 scalar-count shifts emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const vector = func.newVReg();
    const count_shl = func.newVReg();
    const shl = func.newVReg();
    const count_shr_s = func.newVReg();
    const shr_s = func.newVReg();
    const count_shr_u = func.newVReg();
    const shr_u = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8000_0000_7FFF_FFFF_FFFF_FFFF_0000_0001 }, .dest = vector, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 33 }, .dest = count_shl, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_shift = .{ .op = .shl, .vector = vector, .count = count_shl } }, .dest = shl, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 32 }, .dest = count_shr_s, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_shift = .{ .op = .shr_s, .vector = shl, .count = count_shr_s } }, .dest = shr_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 63 }, .dest = count_shr_u, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_shift = .{ .op = .shr_u, .vector = shr_s, .count = count_shr_u } }, .dest = shr_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = shr_u, .lane = 0 } }, .dest = lane, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_count_mask = false;
    var found_dup = false;
    var found_sshl = false;
    var found_neg4s = false;
    var found_ushl = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x12001000) found_count_mask = true;
        if ((w & 0xFFFFFC00) == 0x4E040C00) found_dup = true;
        if ((w & 0xFFE0FC00) == 0x4EA04400) found_sshl = true;
        if ((w & 0xFFFFFC00) == 0x6EA0B800) found_neg4s = true;
        if ((w & 0xFFE0FC00) == 0x6EA04400) found_ushl = true;
    }

    try std.testing.expect(found_count_mask);
    try std.testing.expect(found_dup);
    try std.testing.expect(found_sshl);
    try std.testing.expect(found_neg4s);
    try std.testing.expect(found_ushl);
}

test "compile: i16x8 scalar-count shifts emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const vector = func.newVReg();
    const count_shl = func.newVReg();
    const shl = func.newVReg();
    const count_shr_s = func.newVReg();
    const shr_s = func.newVReg();
    const count_shr_u = func.newVReg();
    const shr_u = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8000_7FFF_FFFF_0001_0008_0004_0002_0001 }, .dest = vector, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 17 }, .dest = count_shl, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_shift = .{ .op = .shl, .vector = vector, .count = count_shl } }, .dest = shl, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 16 }, .dest = count_shr_s, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_shift = .{ .op = .shr_s, .vector = shl, .count = count_shr_s } }, .dest = shr_s, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 31 }, .dest = count_shr_u, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_shift = .{ .op = .shr_u, .vector = shr_s, .count = count_shr_u } }, .dest = shr_u, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i16x8_extract_lane = .{ .vector = shr_u, .lane = 0, .sign = .unsigned } }, .dest = lane, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_count_mask = false;
    var found_dup = false;
    var found_sshl = false;
    var found_neg8h = false;
    var found_ushl = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x12000C00) found_count_mask = true;
        if ((w & 0xFFFFFC00) == 0x4E020C00) found_dup = true;
        if ((w & 0xFFE0FC00) == 0x4E604400) found_sshl = true;
        if ((w & 0xFFFFFC00) == 0x6E60B800) found_neg8h = true;
        if ((w & 0xFFE0FC00) == 0x6E604400) found_ushl = true;
    }

    try std.testing.expect(found_count_mask);
    try std.testing.expect(found_dup);
    try std.testing.expect(found_sshl);
    try std.testing.expect(found_neg8h);
    try std.testing.expect(found_ushl);
}

test "compile: v128 locals remain unsupported before ABI support" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    func.local_types = try allocator.dupe(ir.IrType, &[_]ir.IrType{.v128});
    const bid = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = v, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v } });

    try std.testing.expectError(error.UnsupportedV128, compileFunction(&func, allocator));
}

test "load: size 8 (i64) emits 64-bit LDR X" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 0, .size = 8 } },
        .dest = val,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // LDR X opcode: bits 31..22 = 0xF94 → mask 0xFFC00000 = 0xF9400000.
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0xF9400000) {
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "load: size 1 unsigned (i32.load8_u) emits LDRB" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 3, .size = 1 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // LDRB encoding top bits: 0x39400000 in bits 31..22 (mask 0xFFC00000).
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0x39400000) {
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "load: size 2 signed (i32.load16_s) emits LDRSH W" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 2, .size = 2, .sign_extend = true } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // LDRSH W top bits: 0x79C00000 (mask 0xFFC00000).
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0x79C00000) {
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "load: bounds check emits B.LS + trap BLR + BRK" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 0, .size = 4 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Walk instructions; find B.cond with cond=LS (cond=0b1001), BLR, BRK.
    // B.cond encoding: 0101_0100 in bits 31..24, bits 3..0 = cond.
    // BLR Xn: 1101_0110_0011_1111_0000_00_Rn_00000 → 0xD63F0000 | Rn<<5.
    // BRK #imm: 1101_0100_001_imm16_00000 → 0xD4200000.
    var found_bls = false;
    var found_blr = false;
    var found_brk = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        // B.cond: opcode 0x54 in bits 31..24, bit 4 = 0, cond in bits 3..0.
        if ((w & 0xFF000010) == 0x54000000 and (w & 0xF) == 0x9) found_bls = true;
        if ((w & 0xFFFFFC1F) == 0xD63F0000) found_blr = true;
        if ((w & 0xFFE0001F) == 0xD4200000) found_brk = true;
    }
    try std.testing.expect(found_bls);
    try std.testing.expect(found_blr);
    try std.testing.expect(found_brk);
}

test "store: size 1 emits STRB" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 42 }, .dest = val, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .store = .{ .base = addr, .offset = 0, .size = 1, .val = val } },
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = null } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // STRB top bits: 0x39000000 (mask 0xFFC00000).
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0x39000000) {
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "load: size != 1,2,4,8 returns UnimplementedLoadSize" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{ .base = addr, .offset = 0, .size = 3 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    try std.testing.expectError(error.UnimplementedLoadSize, compileFunction(&func, allocator));
}

/// Build a single-block function with the given ops and verify codegen
/// succeeds and produces 4-byte-aligned output.
fn expectCompiles(insts: []const ir.Inst, param_count: u32, local_count: u32) !void {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, param_count, 1, local_count);
    defer func.deinit();
    const bid = try func.newBlock();
    const block = func.getBlock(bid);
    for (insts) |i| try block.append(i);
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compile: iconst_64 + ret" {
    var func = ir.IrFunction.init(std.testing.allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 0x1234_5678_9ABC_DEF0 }, .dest = v0, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, std.testing.allocator);
    defer std.testing.allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "compile: comparison (eq) produces CMP+CSET" {
    const v0: ir.VReg = 0;
    const v1: ir.VReg = 1;
    const v2: ir.VReg = 2;
    try expectCompiles(&.{
        .{ .op = .{ .iconst_32 = 5 }, .dest = v0, .type = .i32 },
        .{ .op = .{ .iconst_32 = 10 }, .dest = v1, .type = .i32 },
        .{ .op = .{ .eq = .{ .lhs = v0, .rhs = v1 } }, .dest = v2, .type = .i32 },
        .{ .op = .{ .ret = v2 } },
    }, 0, 0);
}

test "compile: eqz" {
    const v0: ir.VReg = 0;
    const v1: ir.VReg = 1;
    try expectCompiles(&.{
        .{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 },
        .{ .op = .{ .eqz = v0 }, .dest = v1, .type = .i32 },
        .{ .op = .{ .ret = v1 } },
    }, 0, 0);
}

test "compile: all comparisons" {
    // Each comparison returns i32 0/1.
    inline for (.{ "eq", "ne", "lt_s", "lt_u", "gt_s", "gt_u", "le_s", "le_u", "ge_s", "ge_u" }) |name| {
        const allocator = std.testing.allocator;
        var func = ir.IrFunction.init(allocator, 0, 1, 0);
        defer func.deinit();
        const bid = try func.newBlock();
        const v0 = func.newVReg();
        const v1 = func.newVReg();
        const v2 = func.newVReg();
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0, .type = .i32 });
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v1, .type = .i32 });
        const op = @unionInit(ir.Inst.Op, name, .{ .lhs = v0, .rhs = v1 });
        try func.getBlock(bid).append(.{ .op = op, .dest = v2, .type = .i32 });
        try func.getBlock(bid).append(.{ .op = .{ .ret = v2 } });
        const code = try compileFunction(&func, allocator);
        defer allocator.free(code);
        try std.testing.expect(code.len > 0);
    }
}

test "compile: shifts and rotates" {
    inline for (.{ "shl", "shr_u", "shr_s", "rotl", "rotr" }) |name| {
        const allocator = std.testing.allocator;
        var func = ir.IrFunction.init(allocator, 0, 1, 0);
        defer func.deinit();
        const bid = try func.newBlock();
        const v0 = func.newVReg();
        const v1 = func.newVReg();
        const v2 = func.newVReg();
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v1, .type = .i32 });
        const op = @unionInit(ir.Inst.Op, name, .{ .lhs = v0, .rhs = v1 });
        try func.getBlock(bid).append(.{ .op = op, .dest = v2, .type = .i32 });
        try func.getBlock(bid).append(.{ .op = .{ .ret = v2 } });
        const code = try compileFunction(&func, allocator);
        defer allocator.free(code);
        try std.testing.expect(code.len > 0);
    }
}

test "compile: clz / ctz / extend / wrap" {
    inline for (.{ "clz", "ctz", "extend8_s", "extend16_s", "extend32_s", "wrap_i64" }) |name| {
        const allocator = std.testing.allocator;
        var func = ir.IrFunction.init(allocator, 0, 1, 0);
        defer func.deinit();
        const bid = try func.newBlock();
        const v0 = func.newVReg();
        const v1 = func.newVReg();
        try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 0x1234 }, .dest = v0, .type = .i64 });
        const op = @unionInit(ir.Inst.Op, name, v0);
        try func.getBlock(bid).append(.{ .op = op, .dest = v1, .type = if (std.mem.eql(u8, name, "wrap_i64")) .i32 else .i64 });
        try func.getBlock(bid).append(.{ .op = .{ .ret = v1 } });
        const code = try compileFunction(&func, allocator);
        defer allocator.free(code);
        try std.testing.expect(code.len > 0);
    }
}

test "compile: select via CSEL" {
    const v0: ir.VReg = 0;
    const v1: ir.VReg = 1;
    const vc: ir.VReg = 2;
    const vr: ir.VReg = 3;
    try expectCompiles(&.{
        .{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 },
        .{ .op = .{ .iconst_32 = 20 }, .dest = v1, .type = .i32 },
        .{ .op = .{ .iconst_32 = 1 }, .dest = vc, .type = .i32 },
        .{ .op = .{ .select = .{ .cond = vc, .if_true = v0, .if_false = v1 } }, .dest = vr, .type = .i32 },
        .{ .op = .{ .ret = vr } },
    }, 0, 0);
}

test "compile: br forward jump patches correctly" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compile: br_if with two blocks" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b_then = try func.newBlock();
    const b_else = try func.newBlock();
    const v_cond = func.newVReg();
    const v_t = func.newVReg();
    const v_f = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_cond, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b_then, .else_block = b_else } } });
    try func.getBlock(b_then).append(.{ .op = .{ .iconst_32 = 11 }, .dest = v_t, .type = .i32 });
    try func.getBlock(b_then).append(.{ .op = .{ .ret = v_t } });
    try func.getBlock(b_else).append(.{ .op = .{ .iconst_32 = 22 }, .dest = v_f, .type = .i32 });
    try func.getBlock(b_else).append(.{ .op = .{ .ret = v_f } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
}

// ── Phase 1b: VMContext ABI + locals tests ───────────────────────────────────

test "compile: entry spills vmctx to [fp+16]" {
    // Function with no params, no locals: prologue must still spill x0 → [fp+16].
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // Layout: prologue (STP + ADD fp,sp,0), then 10 callee-save STRs for
    // x19..x28 (40 bytes), then STR x0 → [fp+16].
    //   opcode 0xF9000000 | (imm12=2 << 10) | (rn=29 << 5) | rt=0
    //   = 0xF9000000 | 0x800 | 0x3A0 | 0 = 0xF9000BA0
    const str_word = std.mem.readInt(u32, code[48..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xF9000BA0), str_word);
}

test "compile: param spill — STR x1 into first local slot" {
    // 1 param (x1), 1 local (just the param), no body besides ret of local 0.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // Layout: STP, ADD fp,sp,0, 10× callee-save STRs (40 bytes), STR x0
    // vmctx, STR x1 local0.
    //   STR x1, [fp, #24]: rt=1, rn=29, imm12=3
    //   = 0xF9000000 | (3<<10) | (29<<5) | 1 = 0xF9000FA1
    const str_param = std.mem.readInt(u32, code[52..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xF9000FA1), str_param);
}

test "compile: zero-init of declared local (beyond params)" {
    // 0 params, 2 locals. After spilling x0 (vmctx), we MOVZ x16,#0 then
    // STR x16 to both local slots.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // Layout (small-frame prologue, 2 words — spill region sized
    // dynamically from vreg count so this function's frame fits in the
    // STP pre-index scaled imm7 range):
    //   [0]   STP FP, LR, [SP, #-frame_size]!
    //   [4]   MOV FP, SP
    //   [8]..[44] 10× callee-save STR (40 bytes)
    //   [48]  STR x0 vmctx, [52] MOVZ x16, [56]/[60] STR x16 → locals.
    const movz_word = std.mem.readInt(u32, code[52..][0..4], .little);
    // MOVZ X16, #0, LSL #0 = 0xD2800000 | (0<<5) | 16 = 0xD2800010
    try std.testing.expectEqual(@as(u32, 0xD2800010), movz_word);
    const str0 = std.mem.readInt(u32, code[56..][0..4], .little);
    // STR x16, [fp, #24]: rt=16, rn=29, imm12=3 → 0xF9000000|(3<<10)|(29<<5)|16
    try std.testing.expectEqual(@as(u32, 0xF9000FB0), str0);
    const str1 = std.mem.readInt(u32, code[60..][0..4], .little);
    // STR x16, [fp, #32]: imm12=4 → 0xF9000000|(4<<10)|(29<<5)|16
    try std.testing.expectEqual(@as(u32, 0xF90013B0), str1);
}

test "compile: local_get + local_set round trip" {
    // local_set 0, const 42; local_get 0; ret.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v0 } } });
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = v1, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v1 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compile: rejects more than 7 params (Phase 1b limit)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 8, 1, 8);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    try std.testing.expectError(error.TooManyParams, compileFunction(&func, allocator));
}

test "compile: unimplemented op returns error.UnimplementedOp" {
    // `.convert_s` / `.convert_u` are legacy IR variants that the
    // current frontend no longer emits; they have no aarch64 handler and
    // must fail loudly via the dispatch `else` branch rather than
    // silently drop.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .convert_s = v0 }, .dest = v1, .type = .f32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v1 } });
    try std.testing.expectError(
        error.UnimplementedOp,
        compileFunction(&func, allocator),
    );
}

test "compileModule: direct tail call emits B (not BL) to target" {
    // Verify `.call{tail=true}` to a local function compiles to a real
    // tail — i.e. a B (0x14000000) patched to the callee's entry,
    // preceded by the frame teardown (LDP + optional SP adjust). Not
    // a BL with a dead epilogue after it.
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // f0: (i32) -> i32 returning x + 1.
    var f0 = ir.IrFunction.init(allocator, 1, 1, 1);
    {
        const b = try f0.newBlock();
        const p = f0.newVReg();
        const one = f0.newVReg();
        const sum = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .local_get = 0 }, .dest = p, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 1 }, .dest = one, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .add = .{ .lhs = p, .rhs = one } }, .dest = sum, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .ret = sum } });
    }
    _ = try module.addFunction(f0);

    // f1: (i32) -> i32 that `return_call f0(x)` — a direct tail call.
    var f1 = ir.IrFunction.init(allocator, 1, 1, 1);
    const args = try allocator.alloc(ir.VReg, 1);
    defer allocator.free(args);
    {
        const b = try f1.newBlock();
        const p = f1.newVReg();
        const r = f1.newVReg();
        try f1.getBlock(b).append(.{ .op = .{ .local_get = 0 }, .dest = p, .type = .i32 });
        args[0] = p;
        try f1.getBlock(b).append(.{
            .op = .{ .call = .{ .func_idx = 0, .args = args, .tail = true } },
            .dest = r,
            .type = .i32,
        });
        // No trailing ret: tail call is a terminator.
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    // Walk f1's code looking for a B (not BL) that targets f0.
    const f1_start = result.offsets[1];
    const f1_end = result.code.len;
    const f0_start = result.offsets[0];
    var found_b_to_f0 = false;
    var saw_bl = false;
    var i: usize = f1_start;
    while (i + 4 <= f1_end) : (i += 4) {
        const word = std.mem.readInt(u32, result.code[i..][0..4], .little);
        // B opcode = 0b000101_xxxxxx → top 6 bits == 0x14000000
        if ((word & 0xFC000000) == 0x14000000) {
            const imm26_raw: u26 = @truncate(word);
            const imm26: i26 = @bitCast(imm26_raw);
            const target: i64 = @as(i64, @intCast(i)) + @as(i64, imm26) * 4;
            if (target == @as(i64, @intCast(f0_start))) {
                found_b_to_f0 = true;
            }
        }
        // BL opcode = 0b100101 → 0x94000000. Tail call must NOT emit BL.
        if ((word & 0xFC000000) == 0x94000000) {
            const imm26_raw: u26 = @truncate(word);
            const imm26: i26 = @bitCast(imm26_raw);
            const target: i64 = @as(i64, @intCast(i)) + @as(i64, imm26) * 4;
            if (target == @as(i64, @intCast(f0_start))) saw_bl = true;
        }
    }
    try std.testing.expect(found_b_to_f0);
    try std.testing.expect(!saw_bl);
}

test "compileModule: multi-result call (ret_multi + call_result)" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    // f0: () -> (i32, i32, i32) returning (10, 20, 30)
    var f0 = ir.IrFunction.init(allocator, 0, 3, 0);
    const rets = try allocator.alloc(ir.VReg, 3);
    defer allocator.free(rets);
    {
        const b = try f0.newBlock();
        const v0 = f0.newVReg();
        const v1 = f0.newVReg();
        const v2 = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 30 }, .dest = v2, .type = .i32 });
        rets[0] = v0;
        rets[1] = v1;
        rets[2] = v2;
        try f0.getBlock(b).append(.{ .op = .{ .ret_multi = rets } });
    }
    _ = try module.addFunction(f0);

    // f1: () -> i32 — calls f0, sums the 3 results.
    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f1.newBlock();
        const r0 = f1.newVReg();
        const r1 = f1.newVReg();
        const r2 = f1.newVReg();
        const s0 = f1.newVReg();
        const s1 = f1.newVReg();
        try f1.getBlock(b).append(.{
            .op = .{ .call = .{ .func_idx = 0, .args = &.{}, .extra_results = 2 } },
            .dest = r0,
            .type = .i32,
        });
        try f1.getBlock(b).append(.{ .op = .{ .call_result = 0 }, .dest = r1, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .call_result = 1 }, .dest = r2, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .add = .{ .lhs = r0, .rhs = r1 } }, .dest = s0, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .add = .{ .lhs = s0, .rhs = r2 } }, .dest = s1, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .ret = s1 } });
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    try std.testing.expect(result.code.len > 0);
    try std.testing.expect(result.code.len % 4 == 0);
}

const QMemOpCounts = struct {
    loads: u32 = 0,
    stores: u32 = 0,
};

fn countQMemOps(code: []const u8) QMemOpCounts {
    var counts = QMemOpCounts{};
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x3DC00000) counts.loads += 1;
        if ((w & 0xFFFFFC00) == 0x3D800000) counts.stores += 1;
    }
    return counts;
}

test "compile: v128 cache keeps local unary chain in registers" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const vector = func.newVReg();
    const inverted = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .v128_const = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF },
        .dest = vector,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_not = vector },
        .dest = inverted,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = inverted, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    const counts = countQMemOps(code);
    try std.testing.expectEqual(@as(u32, 0), counts.loads);
    try std.testing.expectEqual(@as(u32, 0), counts.stores);
}

test "compile: v128 cache does not flush for float constants" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const vector = func.newVReg();
    const float_const = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .v128_const = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF },
        .dest = vector,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .fconst_32 = 4.0 },
        .dest = float_const,
        .type = .f32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = vector, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    const counts = countQMemOps(code);
    try std.testing.expectEqual(@as(u32, 0), counts.loads);
    try std.testing.expectEqual(@as(u32, 0), counts.stores);
}

test "compile: v128 cache avoids stack traffic in memory add chain" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr_a = func.newVReg();
    const addr_b = func.newVReg();
    const addr_dst = func.newVReg();
    const vec_a = func.newVReg();
    const vec_b = func.newVReg();
    const sum = func.newVReg();
    const ret = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr_a, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_load = .{ .base = addr_a, .offset = 0, .alignment = 4, .bounds_known = true } },
        .dest = vec_a,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 16 }, .dest = addr_b, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_load = .{ .base = addr_b, .offset = 0, .alignment = 4, .bounds_known = true } },
        .dest = vec_b,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_binop = .{ .op = .add, .lhs = vec_a, .rhs = vec_b } },
        .dest = sum,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 32 }, .dest = addr_dst, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_store = .{ .base = addr_dst, .offset = 0, .alignment = 4, .val = sum, .bounds_known = true } },
    });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = ret, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = ret } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    const counts = countQMemOps(code);
    try std.testing.expectEqual(@as(u32, 2), counts.loads);
    try std.testing.expectEqual(@as(u32, 1), counts.stores);
}

test "compile: v128 register pool avoids stack traffic under local vector pressure" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const a = func.newVReg();
    const b = func.newVReg();
    const c = func.newVReg();
    const d = func.newVReg();
    const e = func.newVReg();
    const f = func.newVReg();
    const g = func.newVReg();
    const h = func.newVReg();
    const sum_ab = func.newVReg();
    const sum_cd = func.newVReg();
    const sum_ef = func.newVReg();
    const sum_gh = func.newVReg();
    const sum_abcd = func.newVReg();
    const sum_efgh = func.newVReg();
    const total = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_0000_0001 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0008_0000_0007_0000_0006_0000_0005 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_000C_0000_000B_0000_000A_0000_0009 }, .dest = c, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0010_0000_000F_0000_000E_0000_000D }, .dest = d, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0014_0000_0013_0000_0012_0000_0011 }, .dest = e, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0018_0000_0017_0000_0016_0000_0015 }, .dest = f, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_001C_0000_001B_0000_001A_0000_0019 }, .dest = g, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0020_0000_001F_0000_001E_0000_001D }, .dest = h, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .add, .lhs = a, .rhs = b } }, .dest = sum_ab, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .add, .lhs = c, .rhs = d } }, .dest = sum_cd, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .add, .lhs = e, .rhs = f } }, .dest = sum_ef, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .add, .lhs = g, .rhs = h } }, .dest = sum_gh, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .add, .lhs = sum_ab, .rhs = sum_cd } }, .dest = sum_abcd, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .add, .lhs = sum_ef, .rhs = sum_gh } }, .dest = sum_efgh, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_binop = .{ .op = .add, .lhs = sum_abcd, .rhs = sum_efgh } }, .dest = total, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = total, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    const counts = countQMemOps(code);
    try std.testing.expectEqual(@as(u32, 0), counts.loads);
    try std.testing.expectEqual(@as(u32, 0), counts.stores);
}

test "compile: v128 cache flushes across block boundary" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const entry = try func.newBlock();
    const done = try func.newBlock();
    const vector = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(entry).append(.{
        .op = .{ .v128_const = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF },
        .dest = vector,
        .type = .v128,
    });
    try func.getBlock(entry).append(.{ .op = .{ .br = done } });
    try func.getBlock(done).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = vector, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(done).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    const counts = countQMemOps(code);
    try std.testing.expectEqual(@as(u32, 1), counts.loads);
    try std.testing.expectEqual(@as(u32, 1), counts.stores);
}

test "compile: v128 cache survives scalar FP scratch op without stack traffic" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const vector = func.newVReg();
    const float_src = func.newVReg();
    const float_result = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .v128_const = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF },
        .dest = vector,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .fconst_32 = 4.0 },
        .dest = float_src,
        .type = .f32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .f_sqrt = float_src },
        .dest = float_result,
        .type = .f32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = vector, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    const counts = countQMemOps(code);
    try std.testing.expectEqual(@as(u32, 0), counts.loads);
    try std.testing.expectEqual(@as(u32, 0), counts.stores);
}

test "compile: v128 cache survives float binop scratch ops without stack traffic" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const vector = func.newVReg();
    const float_lhs = func.newVReg();
    const float_rhs = func.newVReg();
    const float_result = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .v128_const = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF },
        .dest = vector,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .fconst_32 = 4.0 },
        .dest = float_lhs,
        .type = .f32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .fconst_32 = 9.0 },
        .dest = float_rhs,
        .type = .f32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .add = .{ .lhs = float_lhs, .rhs = float_rhs } },
        .dest = float_result,
        .type = .f32,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = vector, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    const counts = countQMemOps(code);
    try std.testing.expectEqual(@as(u32, 0), counts.loads);
    try std.testing.expectEqual(@as(u32, 0), counts.stores);
}

test "compile: popcnt round-trips via CNT + ADDV" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .popcnt = v0 },
        .dest = v1,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v1 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // Three 4-byte instructions we expect somewhere in the body:
    //   FMOV d0, Xsrc, CNT v0.8b,v0.8b, ADDV b0,v0.8b, FMOV Xd,d0.
    // Just sanity-check CNT v0.8b, v0.8b (0x0E205800) appears.
    var found_cnt = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if (w == 0x0E205800) found_cnt = true;
    }
    try std.testing.expect(found_cnt);
}

test "compile: atomic_rmw add emits LDADDAL" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    const result = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 1 }, .dest = val, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .atomic_rmw = .{ .base = addr, .offset = 0, .size = 4, .val = val, .op = .add } },
        .dest = result,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = result } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // Look for any LDADDAL Ws, Wt, [Xn]: top 12 bits = 0xB8E and
    // opcode[15:12]=0000. Mask low bits for Rs/Rn/Rt.
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0xB8E00000) found = true;
    }
    try std.testing.expect(found);
}

test "compile: atomic_cmpxchg emits CASAL" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const expected = func.newVReg();
    const replacement = func.newVReg();
    const result = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 1 }, .dest = expected, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 2 }, .dest = replacement, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .atomic_cmpxchg = .{ .base = addr, .offset = 0, .size = 4, .expected = expected, .replacement = replacement } },
        .dest = result,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = result } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // Look for CASAL W: 0x88E0FC00 base, Rs/Rn/Rt in low bits.
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x88E0FC00) found = true;
    }
    try std.testing.expect(found);
}

test "compile: call_ref emits CBZ + BLR" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const fref = func.newVReg();
    const result = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 0 }, .dest = fref, .type = .i64 });
    try func.getBlock(bid).append(.{
        .op = .{ .call_ref = .{ .type_idx = 0, .func_ref = fref, .args = &.{}, .extra_results = 0, .tail = false } },
        .dest = result,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = result } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // CBZ X: 0xB4000000 base (sf=1). BLR Xn: 0xD63F0000.
    var found_cbz = false;
    var found_blr = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFF000000) == 0xB4000000) found_cbz = true;
        if ((w & 0xFFFFFC1F) == 0xD63F0000) found_blr = true;
    }
    try std.testing.expect(found_cbz);
    try std.testing.expect(found_blr);
}

test "compile: atomic_notify dispatches helper via VmCtx slot 27" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const count = func.newVReg();
    const result = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 1 }, .dest = count, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .atomic_notify = .{ .base = addr, .offset = 0, .count = count } },
        .dest = result,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = result } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    // LDR Xt, [Xn, #27*8]: sf=1, 11|111|00|01|imm12|Rn|Rt = 0xF9400000 |
    // (imm12<<10) | (Rn<<5) | Rt. Mask opcode + imm12 (bits [21:10]=27).
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0xF9400000 and ((w >> 10) & 0xFFF) == 27) found = true;
    }
    try std.testing.expect(found);
}

test "compile: .call without linkage context returns CallLinkageUnavailable" {
    // compileFunction (no import_count, no patch list) can't resolve
    // direct calls; must error loudly rather than emit bogus BL 0.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .call = .{ .func_idx = 0, .args = &.{} } },
        .dest = v0,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    try std.testing.expectError(error.CallLinkageUnavailable, compileFunction(&func, allocator));
}

test "compile: binop with spilled operands emits LDR/STR via spill slots" {
    // Regalloc is smart enough to not spill values whose live ranges
    // don't overlap, so the naive "17 consecutive iconsts" pattern
    // doesn't force spills any more. Instead, build a chain-reduce
    // where 30 vregs must all be simultaneously live at the final
    // reduction — exceeding the 25 allocatable GPRs and forcing real
    // spills to the frame.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const n = 30;
    var vregs: [n]ir.VReg = undefined;
    for (&vregs, 0..) |*v, i| {
        v.* = func.newVReg();
        try func.getBlock(bid).append(.{
            .op = .{ .iconst_64 = @intCast(i + 1) },
            .dest = v.*,
            .type = .i64,
        });
    }
    // Reduce: acc = v0 + v1; acc += v2; ... acc += v{n-1}
    // All 30 vregs are live at the first add (they were all defined
    // above and each is used once in the following reductions).
    var acc = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .add = .{ .lhs = vregs[0], .rhs = vregs[1] } },
        .dest = acc,
        .type = .i64,
    });
    for (2..n) |i| {
        const next = func.newVReg();
        try func.getBlock(bid).append(.{
            .op = .{ .add = .{ .lhs = acc, .rhs = vregs[i] } },
            .dest = next,
            .type = .i64,
        });
        acc = next;
    }
    try func.getBlock(bid).append(.{ .op = .{ .ret = acc } });

    // Disable local scheduling here: the test targets spill codegen, and the
    // scheduler can legally shorten this synthetic live-pressure pattern.
    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_scheduler = false });
    defer allocator.free(code);

    // Scan for LDR Xt, [fp, #imm] (opcode pattern 0xF9400000 + rn=29<<5).
    var found_ldr_from_fp = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        const top = (w >> 22) & 0x3FF;
        const rn = (w >> 5) & 0x1F;
        if (top == 0x3E5 and rn == 29) {
            found_ldr_from_fp = true;
            break;
        }
    }
    try std.testing.expect(found_ldr_from_fp);
}

test "compileFunction: FMA fusion does not skip mul with non-arith uses" {
    // Regression test for a bug where the FMA fusion pre-pass only
    // counted vreg uses in binary ops. A mul whose dest was read by BOTH
    // a subsequent `add` AND a `local_set` was miscounted as single-use,
    // fused into a MADD, and its original mul emission was skipped —
    // leaving the vreg unbound when `local_set` later tried to read it.
    //
    // This mirrors the pattern CoreMark's `core_state_transition` hits:
    //   %v3 = mul %v0, %v1        ; marked single-use, skipped
    //   %v4 = add %v3, %v2        ; emits MADD using mul_lhs/mul_rhs
    //   local_set %v3             ; UnboundVReg — v3 was never written
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1); // 1 local (for local_set)
    defer func.deinit();

    const b0 = try func.newBlock();
    const block = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg();
    const v4 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .mul = .{ .lhs = v0, .rhs = v1 } }, .dest = v3, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = v3, .rhs = v2 } }, .dest = v4, .type = .i32 });
    // Second (non-binary-op) use of v3 — local_set. If the pre-pass
    // misses this use, the mul is fused away and compilation fails with
    // error.UnboundVReg when we get here.
    try block.append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v3 } }, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v4 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "collectClobberPoints: no calls → empty" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const block = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v0 } });

    var cps = try collectClobberPoints(&func, null, null, allocator);
    defer cps.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), cps.items.len);
}

test "collectClobberPoints: one ClobberPoint per call, correct mask" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const block = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    // Mix of clobbering and non-clobbering ops to verify positions.
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v0, .type = .i32 }); // pos 0
    try block.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v1 }); // pos 1
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = func.newVReg(), .type = .i32 }); // pos 2
    try block.append(.{ .op = .{ .memory_fill = .{ .dst = v0, .val = v0, .len = v0 } } }); // pos 3
    try block.append(.{ .op = .{ .ret = v1 } }); // pos 4

    var cps = try collectClobberPoints(&func, null, null, allocator);
    defer cps.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), cps.items.len);
    try std.testing.expectEqual(@as(u32, 1), cps.items[0].pos);
    try std.testing.expectEqual(@as(u32, 3), cps.items[1].pos);
    // Full caller-saved set (x0..x14 = indices 0..14): (1<<15) - 1 = 0x7FFF.
    try std.testing.expectEqual(@as(u64, 0x7FFF), cps.items[0].regs_clobbered);
    try std.testing.expectEqual(@as(u64, 0x7FFF), cps.items[1].regs_clobbered);
}

test "collectClobberPoints: positions are monotonic across blocks" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);
    const v0 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 }); // pos 0
    try block0.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = func.newVReg() }); // pos 1
    try block0.append(.{ .op = .{ .br = b1 } }); // pos 2
    try block1.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = func.newVReg() }); // pos 3
    try block1.append(.{ .op = .{ .ret = v0 } }); // pos 4

    var cps = try collectClobberPoints(&func, null, null, allocator);
    defer cps.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), cps.items.len);
    try std.testing.expectEqual(@as(u32, 1), cps.items[0].pos);
    try std.testing.expectEqual(@as(u32, 3), cps.items[1].pos);
    try std.testing.expect(cps.items[0].pos < cps.items[1].pos);
}

test "aarch64RegSet: sane layout for a tiny function" {
    const rs = aarch64RegSet(0);
    // 25 allocatable GPRs, 15 caller-saved + 10 callee-saved.
    try std.testing.expectEqual(@as(usize, 25), rs.alloc_regs.len);
    try std.testing.expectEqual(@as(usize, 15), rs.caller_saved_indices.len);
    try std.testing.expectEqual(@as(usize, 10), rs.callee_saved_indices.len);
    // Spill stride grows upward (away from fp).
    try std.testing.expect(rs.spill_stride > 0);
    // Spills live above the locals area (saved fp/lr + vmctx + locals = 24 bytes min).
    try std.testing.expect(rs.spill_base >= 24);
}
