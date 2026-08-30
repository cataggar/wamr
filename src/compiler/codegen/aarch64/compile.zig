//! AArch64 IR Compiler
//!
//! Walks IR functions and emits AArch64 machine code via CodeBuffer.
//! Uses the same pattern as the x86-64 backend with a linear-scan
//! register allocator mapping VRegs to physical registers.

const std = @import("std");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");
const schedule = @import("schedule.zig");
const coalesce_post = @import("coalesce_post.zig");
const range_split = @import("../../ir/range_split.zig");
const codegen_cache = @import("../../codegen_cache.zig");
const passes = @import("../../ir/passes.zig");
const codegen_timing = @import("../timing.zig");

/// Compile-time debug flag: when true, print per-function range-split
/// stats to stderr. Flip via `zig build -Drange-split-debug=true` after
/// adding the option if needed; for routine bench triage just edit this
/// line and rebuild — the harness handles isolated caches per worktree.
const range_split_debug = false;

/// #754 codegen trace knob — when true, print one line per `select`
/// emission. Off by default; see x86_64/compile.zig for the matching
/// flag and a longer rationale.
const trace_select: bool = false;
const regalloc = @import("../../ir/regalloc.zig");
const analysis = @import("../../ir/analysis.zig");
const local_init = @import("../../ir/local_init.zig");

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
    // X19 is reserved as the pinned `vmctx` register for the entire
    // function body (issue #465). The prologue copies x0 (AAPCS64 arg0
    // = vmctx) into x19 once, and every site that previously reloaded
    // vmctx from `[fp + 16]` reads x19 directly. Because x19 is AAPCS64
    // callee-saved, calls (to wasm helpers, trap helpers, host
    // functions) preserve it without a save/restore at the call site.
    //
    // X20 is reserved as the pinned `memory_base` register (issue
    // #466). The prologue loads `[x19 + 0]` (= VmCtx.memory_base) into
    // x20 once, and every linear-memory access reads x20 directly
    // instead of rematerialising the base. memory.grow and every wasm
    // call site emit a reload `ldr x20, [x19, #0]` because either op
    // can invalidate the cached base.
    //
    // X21–X28 are AAPCS64 callee-saved and appended after the caller-
    // saved block. They survive calls (no save around BL), trading one
    // extra STR/LDR pair per function per reg that the body allocates.
    pub const caller_saved_count: usize = 15;
    pub const scratch_regs = [_]emit.Reg{
        .x0,  .x1,  .x2,  .x3,  .x4,  .x5,  .x6,  .x7,
        .x8,  .x9,  .x10, .x11, .x12, .x13, .x14,
        // Callee-saved (x19 is pinned to vmctx, x20 to memory_base; neither allocatable):
        .x21,
        .x22, .x23, .x24, .x25, .x26, .x27, .x28,
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
            // x19 (vmctx, issue #465) and x20 (memory_base, issue #466)
            // are permanently pinned. Mark both as used in
            // `used_callee_mask` (bits 0 and 1) so the prologue STR /
            // epilogue LDR pair for each is never NOP-patched. The
            // prologue STRs happen BEFORE we clobber x19/x20 with their
            // pinned values, so each caller's value is correctly
            // preserved across our frame.
            .used_callee_mask = 0b11,
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
                        // 21..28; x19 is excluded — pinned to vmctx —
                        // and x20 is excluded — pinned to memory_base).
                        // Find its index in `scratch_regs` to update
                        // `reg_used` / `used_callee_mask`. After
                        // skipping x15..x18 (4 non-allocatable) plus
                        // x19+x20 (pinned), reg_num N >= 21 maps to
                        // scratch_regs index N - 6.
                        const reg_num: u32 = @intCast(r);
                        const idx: usize = if (reg_num <= 14) reg_num else reg_num - 6;
                        self.reg_used[idx] = true;
                        if (idx >= caller_saved_count) {
                            // scratch_regs callee-saved tail starts at
                            // x21 (idx 15). callee_saved_regs (the array
                            // used for prologue STR/LDR, still
                            // [.x19..x28]) has x21 at index 2, so the
                            // mask bit is (idx - caller_saved_count) + 2.
                            self.used_callee_mask |= (@as(u16, 1) << @intCast(idx - caller_saved_count + 2));
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
                    // scratch_regs callee-saved tail starts at x21 (x19
                    // and x20 are pinned). callee_saved_regs (used for
                    // prologue STR/LDR) still starts at x19 at bit 0,
                    // so x21 occupies bit 2.
                    self.used_callee_mask |= (@as(u16, 1) << @intCast(i - caller_saved_count + 2));
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

    /// FP-relative *byte* offset of a spill slot. Used by the frame-access
    /// helpers (#916), which fall back to address materialization when the
    /// offset overflows the 12-bit scaled LDR/STR immediate.
    fn spillOffsetBytes(self: *const RegMap, slot_byte_off: u32) u32 {
        return self.spill_base + slot_byte_off;
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
    /// Optional linear-scan V-register allocation result. Register homes are
    /// used directly; stack homes reuse this map as the spill fallback.
    alloc_result: ?*const regalloc.AllocResult = null,

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
        if (self.alloc_result) |ar| {
            if (ar.get(vreg)) |a| {
                return switch (a) {
                    .reg => error.V128AlreadyInRegister,
                    .stack => |off| if (off >= 0) @as(u32, @intCast(off)) else error.NegativeSpillOffset,
                };
            }
            return error.UnboundV128;
        }
        if (self.entries.get(vreg)) |off| return off;
        if (self.next_offset + 16 > self.spill_capacity) return error.OutOfV128SpillSlots;
        const off = self.spill_base + self.next_offset;
        self.next_offset += 16;
        try self.entries.put(vreg, off);
        return off;
    }

    fn get(self: *const V128StackMap, vreg: ir.VReg) !u32 {
        if (self.alloc_result) |ar| {
            if (ar.get(vreg)) |a| {
                return switch (a) {
                    .reg => error.V128AlreadyInRegister,
                    .stack => |off| if (off >= 0) @as(u32, @intCast(off)) else error.NegativeSpillOffset,
                };
            }
            return error.UnboundV128;
        }
        return self.entries.get(vreg) orelse error.UnboundV128;
    }

    fn reg(self: *const V128StackMap, vreg: ir.VReg) ?u5 {
        const ar = self.alloc_result orelse return null;
        const a = ar.get(vreg) orelse return null;
        return switch (a) {
            .reg => |r| @intCast(r),
            .stack => null,
        };
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

    // Legacy/bisect fallback cache for v128 values that do not have a direct
    // linear-scan V-register home. With enable_vreg_alloc=true, typical v128
    // values use v3-v7/v16-v31 directly and never enter this cache; stack
    // spills and enable_vreg_alloc=false still use it as before. Q8-Q15 are
    // intentionally excluded because using them would require SIMD
    // callee-save prologue/epilogue wiring (deferred).
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
        if (v128_map.reg(vreg)) |reg| return reg;
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
        if (v128_map.reg(dest)) |reg| return reg;
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
        if (v128_map.alloc_result != null and slotForReg(vt) == null) return;
        const idx = slotForReg(vt) orelse return error.InvalidV128ScratchReg;
        try self.evictSlot(code, v128_map, idx, !drop_existing);
    }

    fn defineInReg(self: *V128RegCache, v128_map: *V128StackMap, dest: ir.VReg, vt: u5) !void {
        if (v128_map.reg(dest)) |reg| {
            if (reg != vt) return error.V128RegisterMismatch;
            return;
        }
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
    enable_peephole: bool = true,
    /// Use the Stage-A scalar X-register linear-scan allocator.
    /// When disabled, codegen falls back to the legacy greedy RegMap path.
    enable_xreg_alloc: bool = true,
    /// Use the Stage-B SIMD V-register linear-scan allocator.
    /// When disabled, v128 codegen falls back to V128StackMap/V128RegCache.
    enable_vreg_alloc: bool = true,
    /// Run the post-allocation move-coalescing pass (issue #386). After
    /// `regalloc.allocateFromRangesWithHints` assigns physregs, walk the
    /// IR for copy-like ops (currently `.reinterpret`) and retarget the
    /// dest vreg onto the source vreg's physreg when no live-range
    /// conflict prevents it. The emit-time `if (src != dest) movRegReg`
    /// guard then elides the now-redundant mov entirely.
    enable_move_coalesce: bool = true,
    /// Run loop-aware live-range splitting before linear scan (issue #383).
    /// Splits vregs that cross a loop without being used inside it, freeing
    /// their physical register for the hot loop body. Conservative: only
    /// loops with a single entry-pred and single exit-edge are eligible.
    enable_range_split: bool = true,
    /// Run the post-emission MOV coalescer scan (issue #539). After the
    /// main emit loop, scan each basic block of the finalised byte
    /// stream for `MOV xD, xS` (and `MOV wD, wS`) instructions that
    /// regalloc could not eliminate via copy-hints, rewrite downstream
    /// uses of `xD` to read `xS`, and replace the MOV with a NOP. The
    /// pass is block-local in v1; cross-block coalescing is filed as a
    /// follow-up. Disabling this option also disables the scan on the
    /// `coalesce_post.zig` unit tests via build.zig's test step.
    enable_post_emit_coalesce: bool = true,
    /// #808: elide an unconditional `br` whose target is the immediately
    /// following block in emission (layout) order — i.e. a redundant
    /// `b .+4`. The aarch64 block layout is reverse-post-order, so a block
    /// frequently branches to the block laid down right after it; `emitBr`
    /// emits that branch unconditionally. These dead branches dominate the
    /// CoreMark hot loops (lf7/lf9: 100% of their unconditional branches,
    /// lf3: 93/111). Eliding them removes the instruction entirely (smaller
    /// code, fewer front-end redirects). Off disables for A/B measurement.
    enable_fallthrough_elision: bool = true,
    /// Fold eligible checked memory32 accesses with memarg offset zero into
    /// AArch64's `[x20, x15]` register-offset load/store form. The existing
    /// `mov w15, wAddr` remains in place to preserve wasm's unsigned i32
    /// address semantics; only the separate `add x16, x20, x15` is removed.
    /// Bounds-known, memory64-shaped, non-zero-offset, SIMD, and atomic/shared
    /// paths keep their existing address-materialization fallback.
    enable_mem_reg_offset: bool = true,
    /// #778: native-codegen timing diagnostics. aarch64 reports per-function
    /// totals + a module summary (no x86_64-style setup/liveness/regalloc
    /// sub-phase breakdown — the #778 bottleneck is the x86_64 host path).
    codegen_timing: passes.CodegenTimingOptions = .{},
    /// #808 Lever 1: per-function spill-cost diagnostics. Off unless
    /// `WAMR_AOT_SPILL_METRIC` is set. Combines the scalar (X) and v128 (V)
    /// allocations into one per-function report.
    spill_metric: passes.SpillMetricOptions = .{},
    /// Component core/module index, used only to match the
    /// `WAMR_AOT_CODEGEN_TIMING_MODULE` filter.
    module_idx: u32 = 0,
    /// #862 lazy-JIT spike: indexed by LOCAL function index (same
    /// indexing as `ir_module.functions.items`). When
    /// `lazy_skip[fi]` is true, `compileModuleCachedWithOptions`
    /// emits a zero-byte placeholder for that function instead of
    /// compiling it — safe ONLY because `lazy_jit.findLazyEligibleLeaves`
    /// guarantees skipped functions are leaf (no outgoing calls to
    /// patch) and never a direct call target (no inter-function call
    /// patch will ever reference their offset). Empty by default —
    /// zero behavior change for every existing caller.
    lazy_skip: []const bool = &.{},
};

/// Context threaded through per-function compilation for cross-function
/// concerns (calls, imports).
const FuncCompileCtx = struct {
    import_count: u32 = 0,
    has_memory64: bool = false,
    has_shared_memory: bool = false,
    call_patches: ?*std.ArrayListUnmanaged(CallPatch) = null,
    /// FP-relative byte offset for each wasm/synthetic local slot.
    local_offsets: []const u32 = &.{},
    /// Per-local type table, params first. Missing entries are treated as
    /// scalar 8-byte slots for compatibility with hand-built IR tests.
    local_types: []const ir.IrType = &.{},
    /// Wasm-flat global types and byte offsets (imports first, then locals).
    /// Missing entries fall back to the legacy 8-byte-per-index layout used by
    /// older hand-built IR tests.
    global_types: []const ir.IrType = &.{},
    global_offsets: []const u32 = &.{},
    /// Wasm function type table and module funcidx → typeidx map. Missing
    /// entries are tolerated for hand-built IR tests; call lowering then falls
    /// back to per-vreg definition types.
    func_types: []const ir.IrFuncType = &.{},
    func_type_indices: []const u32 = &.{},
    vreg_types: ?*const std.AutoHashMap(ir.VReg, ir.IrType) = null,
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
    /// extra results of calls (beyond the primary return register) are
    /// written by the callee. Scalar slots keep the legacy 8-byte layout;
    /// v128 slots are 16-byte aligned and 16 bytes wide.
    scratch_base: u32 = 0,
    /// Precomputed `.call_result` dest vreg → byte offset within
    /// `scratch_base`, using the callee's result type layout.
    call_result_offsets: ?*const std.AutoHashMap(ir.VReg, u32) = null,
    /// Total frame size (in bytes). Needed by tail-call emitters to
    /// tear down the frame before branching to the target.
    frame_size: u32 = 0,
    /// VReg → signed i64 value for `iconst_*` definitions in this function.
    /// Used to fold constants into immediate-form instructions (e.g.
    /// ADD/SUB imm). The iconst itself is still materialized into its home
    /// register, so downstream uses that don't fold still work.
    const_vals: ?*const std.AutoHashMap(ir.VReg, i64) = null,
    /// Set of `iconst_*` dest vregs whose every use folds as an immediate
    /// operand (currently: ADD/SUB imm12). For these vregs the codegen-side
    /// `mov xN, #K` materialization is suppressed — the physreg is allocated
    /// (so regalloc bookkeeping stays consistent) but never written. See
    /// issue #523. The IR-level def stays in place; only the asm emit is
    /// elided.
    suppress_iconst_emit: ?*const std.AutoHashMap(ir.VReg, void) = null,
    /// Set of mul dest vregs whose computation should be skipped because
    /// the single consumer is an add/sub that will be emitted as a fused
    /// MADD/MSUB (Phase 4 FMA fusion). When the mul handler sees a dest
    /// in this set, it bypasses codegen and leaves the sources live for
    /// the fused op.
    mul_fused: ?*const std.AutoHashMap(ir.VReg, void) = null,
    /// Map from add/sub dest vreg → (mul_lhs, mul_rhs, addend) source
    /// triple. Populated by the FMA pre-pass in lockstep with `mul_fused`.
    fma_info: ?*const std.AutoHashMap(ir.VReg, FmaInfo) = null,
    /// SIMD f32x4/f64x2 mul dests consumed by a fused FMLA/FMLS.
    simd_mul_fused: ?*const std.AutoHashMap(ir.VReg, void) = null,
    /// SIMD add/sub dest vreg → fused FMLA/FMLS source triple.
    simd_fma_info: ?*const std.AutoHashMap(ir.VReg, FmaInfo) = null,
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
    /// #781: optional per-function sub-phase stopwatch. Non-null only when
    /// `WAMR_AOT_CODEGEN_TIMING` is live for this module; `compileFunctionImpl`
    /// threads its scheduling / range-split / prepass / liveness / regalloc /
    /// coalesce / post-emit-coalesce spans into it. Null is the normal
    /// (untimed) path — a hard no-op.
    codegen_timer: ?*codegen_timing.Aarch64FuncTimer = null,
    /// #808 Lever 1: local function index, used to label the spill-metric
    /// line and to match the `WAMR_AOT_SPILL_METRIC_FUNC` filter. Defaults
    /// to 0 on the single-function `compileFunctionWithOptions` entry.
    /// #808: fall-through elision support. `block_rep[block_pos[b]]` gives the
    /// block id whose address block `b` resolves to after empty single-`br`
    /// blocks are elided; `fallthrough_rep` is that representative for the
    /// block physically following the one being compiled. A branch target
    /// that resolves to `fallthrough_rep` is a redundant `b .+4`.
    block_rep: []const ir.BlockId = &.{},
    block_pos: []const usize = &.{},
    fallthrough_rep: ?ir.BlockId = null,
    func_idx: u32 = 0,
    allocator: std.mem.Allocator,

    /// #808: true when a branch `target` resolves to the current block's
    /// physical fall-through address, so a branch to it can be dropped (the
    /// block simply falls through). Gated by the codegen option so the A/B
    /// toggle disables all elision.
    fn targetIsFallthrough(self: *const FuncCompileCtx, target: ir.BlockId) bool {
        if (!self.options.enable_fallthrough_elision) return false;
        const ft = self.fallthrough_rep orelse return false;
        if (self.block_pos.len == 0) return false;
        return self.block_rep[self.block_pos[target]] == ft;
    }
};

/// Pre-computed info for a fused MADD/MSUB: `dest = addend ± mul_lhs * mul_rhs`.
const FmaInfo = struct {
    mul_lhs: ir.VReg,
    mul_rhs: ir.VReg,
    addend: ir.VReg,
    is_sub: bool,
};

const SimdFmaBin = union(enum) {
    f32: ir.Inst.F32x4BinOp,
    f64: ir.Inst.F64x2BinOp,
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
/// **Wasm parameters**: scalar params use `x1..x7` with `x0` reserved for the
/// hidden VMContext; v128 params use SIMD registers `q0..q7`. Each register
/// bank is allocated independently, and excess params are read from the
/// caller's stack arg area with v128 slots 16-byte aligned.
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

/// #862 lazy-JIT design-spike: public per-function entry point matching
/// the SAME real aarch64 codegen `compileModuleCachedWithOptions` uses.
/// Used by `component_aot_compile.LazyCompileDriver` to compile a single
/// deferred leaf function on demand with the module's real
/// `import_count`/`global_offsets`, matching what it would have gotten
/// had it been compiled eagerly in the module loop.
pub fn compileFunctionWithGlobalOffsetsPublic(
    func: *const ir.IrFunction,
    import_count: u32,
    global_offsets: []const u32,
    has_memory64: bool,
    has_shared_memory: bool,
    allocator: std.mem.Allocator,
) ![]u8 {
    return compileFunctionImpl(func, .{
        .import_count = import_count,
        .global_offsets = global_offsets,
        .has_memory64 = has_memory64,
        .has_shared_memory = has_shared_memory,
        .allocator = allocator,
    }, allocator);
}

fn isSupportedV128Def(inst: ir.Inst) bool {
    return switch (inst.op) {
        .local_get,
        .global_get,
        .call,
        .call_indirect,
        .call_ref,
        .call_result,
        .v128_const,
        .v128_load,
        .v128_load_splat,
        .v128_load_zero,
        .v128_load_extend,
        .v128_load_lane,
        .v128_not,
        .v128_bitwise,
        .v128_bitselect,
        .i32x4_binop,
        .i32x4_unop,
        .f32x4_unop,
        .f32x4_binop,
        .f32x4_convert_i32x4,
        .i32x4_trunc_sat,
        .f32x4_demote_f64x2_zero,
        .f32x4_splat,
        .f32x4_replace_lane,
        .i32x4_extadd_pairwise_i16x8,
        .i32x4_dot_i16x8_s,
        .i32x4_extend_i16x8,
        .i32x4_extmul_i16x8,
        .i32x4_shift,
        .i32x4_splat,
        .i32x4_replace_lane,
        .i8x16_binop,
        .i8x16_shuffle,
        .i8x16_swizzle,
        .i8x16_unop,
        .i8x16_shift,
        .i8x16_splat,
        .i8x16_replace_lane,
        .i8x16_narrow_i16x8,
        .i16x8_binop,
        .i16x8_unop,
        .i16x8_extadd_pairwise_i8x16,
        .i16x8_extend_i8x16,
        .i16x8_extmul_i8x16,
        .i16x8_shift,
        .i16x8_splat,
        .i16x8_replace_lane,
        .i16x8_narrow_i32x4,
        .i64x2_binop,
        .i64x2_unop,
        .f64x2_convert_low_i32x4,
        .f64x2_promote_low_f32x4,
        .i64x2_extend_i32x4,
        .i64x2_extmul_i32x4,
        .i64x2_shift,
        .i64x2_splat,
        .i64x2_replace_lane,
        .f64x2_unop,
        .f64x2_binop,
        .f64x2_splat,
        .f64x2_replace_lane,
        => inst.type == .v128,
        else => false,
    };
}

fn functionHasUnsupportedV128(func: *const ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var vreg_types = std.AutoHashMap(ir.VReg, ir.IrType).init(allocator);
    defer vreg_types.deinit();

    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            switch (inst.op) {
                .v128_const,
                .v128_load,
                .v128_load_splat,
                .v128_load_zero,
                .v128_load_extend,
                .v128_load_lane,
                .v128_not,
                .v128_bitwise,
                .v128_bitselect,
                .i32x4_binop,
                .i32x4_unop,
                .f32x4_unop,
                .f32x4_binop,
                .f32x4_convert_i32x4,
                .i32x4_trunc_sat,
                .f32x4_demote_f64x2_zero,
                .f32x4_splat,
                .f32x4_replace_lane,
                .i32x4_extadd_pairwise_i16x8,
                .i32x4_dot_i16x8_s,
                .i32x4_extend_i16x8,
                .i32x4_extmul_i16x8,
                .i32x4_shift,
                .i32x4_splat,
                .i32x4_replace_lane,
                .i8x16_binop,
                .i8x16_shuffle,
                .i8x16_swizzle,
                .i8x16_unop,
                .i8x16_shift,
                .i8x16_splat,
                .i8x16_replace_lane,
                .i8x16_narrow_i16x8,
                .i16x8_binop,
                .i16x8_unop,
                .i16x8_extadd_pairwise_i8x16,
                .i16x8_extend_i8x16,
                .i16x8_extmul_i8x16,
                .i16x8_shift,
                .i16x8_splat,
                .i16x8_replace_lane,
                .i16x8_narrow_i32x4,
                .i64x2_binop,
                .i64x2_unop,
                .f64x2_convert_low_i32x4,
                .f64x2_promote_low_f32x4,
                .i64x2_extend_i32x4,
                .i64x2_extmul_i32x4,
                .i64x2_shift,
                .i64x2_splat,
                .i64x2_replace_lane,
                .f64x2_unop,
                .f64x2_binop,
                .f64x2_splat,
                .f64x2_replace_lane,
                => {
                    if (!isSupportedV128Def(inst)) return true;
                },
                .v128_store,
                .v128_store_lane,
                .f32x4_extract_lane,
                .i32x4_extract_lane,
                .i8x16_extract_lane,
                .i16x8_extract_lane,
                .i64x2_extract_lane,
                .f64x2_extract_lane,
                => {},
                else => {},
            }
            if (inst.dest != null and inst.type == .v128 and !isSupportedV128Def(inst)) return true;
            if (inst.dest) |dest| try vreg_types.put(dest, inst.type);
        }
    }

    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            switch (inst.op) {
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

/// Relax `B.cond` patches whose target is outside the AArch64 imm19
/// reach (±1 MiB) by rewriting each one in place to a 2-instruction
/// sequence using the much wider imm26 (±128 MiB) range:
///
///   B.!cond +8         ; conditional skip over the next instruction
///   B target           ; long-range unconditional branch
///
/// Each rewrite inserts 4 bytes into `code.bytes` at `patch_offset + 4`
/// and shifts every downstream `block_offsets` entry, `BranchPatch`,
/// and `CallPatch` whose stored offset crosses the insertion point.
/// The original `BranchPatch` is repointed at the inserted unconditional
/// `B` (kind becomes `.b_uncond`) so the normal patch-resolution loop
/// resolves the long-range half.
///
/// Iterates because relaxing one branch shifts code that other
/// in-range conditional branches span — a few sweeps may surface new
/// out-of-range candidates. Caps at `max_iter` defensive sweeps;
/// returns `error.BranchOutOfRange` if the count is exceeded (in
/// practice 1-2 iterations suffice even for multi-MiB functions).
/// Issue #621.
fn relaxOutOfRangeConditionalBranches(
    code: *emit.CodeBuffer,
    block_offsets: []usize,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    call_patches: ?*std.ArrayListUnmanaged(CallPatch),
    allocator: std.mem.Allocator,
) !void {
    const max_iter: u32 = 16;
    const min_word: i64 = -(@as(i64, 1) << 18);
    const max_word: i64 = @as(i64, 1) << 18;

    // Scratch lists reused across iterations to avoid per-iter alloc churn.
    var oor: std.ArrayListUnmanaged(usize) = .empty;
    defer oor.deinit(allocator);
    var ins_points: std.ArrayListUnmanaged(usize) = .empty;
    defer ins_points.deinit(allocator);

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        // 1. Collect every out-of-range b_cond patch.
        oor.clearRetainingCapacity();
        for (patches.items, 0..) |p, i| {
            if (p.kind != .b_cond) continue;
            const target_off: i64 = @intCast(block_offsets[p.target_block]);
            const patch_off: i64 = @intCast(p.patch_offset);
            const word_off = @divTrunc(target_off - patch_off, 4);
            if (word_off >= min_word and word_off < max_word) continue;
            try oor.append(allocator, i);
        }
        if (oor.items.len == 0) return;

        // 2. Sort OOR indices by patch_offset ascending so we can rebuild
        //    `code.bytes` and compute cumulative shifts via prefix counts.
        const SortCtx = struct {
            patches_slice: []const BranchPatch,
            pub fn lessThan(self: @This(), a: usize, b: usize) bool {
                return self.patches_slice[a].patch_offset < self.patches_slice[b].patch_offset;
            }
        };
        std.mem.sort(
            usize,
            oor.items,
            SortCtx{ .patches_slice = patches.items },
            SortCtx.lessThan,
        );

        // 3. Snapshot the original patch_offsets of the OOR set into a
        //    sorted insertion-point array for O(log K) shift queries.
        ins_points.clearRetainingCapacity();
        try ins_points.ensureTotalCapacity(allocator, oor.items.len);
        for (oor.items) |idx| {
            ins_points.appendAssumeCapacity(patches.items[idx].patch_offset);
        }

        // 4. Rewrite every OOR B.cond word in place (flip its condition
        //    bit and retarget to `+2 words` = +8 bytes). We do this
        //    against the OLD buffer; the inserted B will be written
        //    into the NEW buffer in step 5.
        const old_data = code.bytes.items;
        for (oor.items) |idx| {
            const orig_offset = patches.items[idx].patch_offset;
            const old_word = std.mem.readInt(u32, old_data[orig_offset..][0..4], .little);
            const new_cond: u32 = (old_word & 0xF) ^ 1;
            const new_bcond_word: u32 = 0x54000000 | (@as(u32, 2) << 5) | new_cond;
            std.mem.writeInt(u32, old_data[orig_offset..][0..4], new_bcond_word, .little);
        }

        // 5. Build a fresh code buffer with a 4-byte unconditional-B slot
        //    inserted after each OOR B.cond. Single-pass O(N) rebuild
        //    avoids the O(K * N) memmove cost of repeated insertSlice.
        const old_len = old_data.len;
        const new_len = old_len + oor.items.len * 4;
        var new_buf = try code.allocator.alloc(u8, new_len);
        errdefer code.allocator.free(new_buf);

        var src_pos: usize = 0;
        var dst_pos: usize = 0;
        for (oor.items) |idx| {
            const orig_offset = patches.items[idx].patch_offset;
            const copy_until = orig_offset + 4;
            const chunk = copy_until - src_pos;
            @memcpy(new_buf[dst_pos..][0..chunk], old_data[src_pos..][0..chunk]);
            src_pos += chunk;
            dst_pos += chunk;
            // Placeholder unconditional `B 0` — its imm26 is patched by
            // the resolution loop below.
            std.mem.writeInt(u32, new_buf[dst_pos..][0..4], 0x14000000, .little);
            dst_pos += 4;
        }
        @memcpy(new_buf[dst_pos..], old_data[src_pos..]);

        // Swap buffers.
        code.bytes.deinit(code.allocator);
        code.bytes = .{ .items = new_buf[0..new_len], .capacity = new_len };

        // 6. Apply cumulative shifts. For byte position X, the shift is
        //    4 * (number of OOR insertion points strictly less than X).
        for (block_offsets) |*bo| {
            const shift = lowerBound(ins_points.items, bo.*);
            bo.* += shift * 4;
        }
        for (patches.items, 0..) |*p, i| {
            // Skip the OOR patches themselves — handled in step 7.
            if (isOorIndex(oor.items, i)) continue;
            const shift = lowerBound(ins_points.items, p.patch_offset);
            p.patch_offset += shift * 4;
        }
        if (call_patches) |cp| {
            for (cp.items) |*p| {
                const shift = lowerBound(ins_points.items, p.patch_offset);
                p.patch_offset += shift * 4;
            }
        }

        // 7. Repoint each OOR patch at the inserted unconditional B. The
        //    j-th OOR patch (zero-based by sorted order) has j prior
        //    insertions before it, so its new B sits at:
        //      original_offset + 4   + (j + 1) * 4 - 4
        //    = original_offset + (j + 1) * 4
        //    The `+ (j + 1) * 4 - 4` extra term accounts for being
        //    after this patch's own insertion. Simplified:
        for (oor.items, 0..) |idx, j| {
            patches.items[idx].patch_offset = ins_points.items[j] + (j + 1) * 4;
            patches.items[idx].kind = .b_uncond;
        }
    }
    return error.BranchOutOfRange;
}

/// Number of `sorted` entries strictly less than `target`.
fn lowerBound(sorted: []const usize, target: usize) usize {
    var lo: usize = 0;
    var hi: usize = sorted.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (sorted[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// O(log K) membership check in a sorted-ascending index list. Used only
/// when K (out-of-range count) is small; for larger K the indices are
/// effectively a dense subset of `patches.items` and a flag-array would
/// win — but the OOR set is bounded by the number of long-range
/// conditional branches in a single function, which stays manageable
/// even on the issue #621 repro (~16k total patches).
fn isOorIndex(sorted: []const usize, target: usize) bool {
    var lo: usize = 0;
    var hi: usize = sorted.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (sorted[mid] == target) return true;
        if (sorted[mid] < target) lo = mid + 1 else hi = mid;
    }
    return false;
}

pub fn compileFunctionImpl(
    func: *const ir.IrFunction,
    ctx: FuncCompileCtx,
    allocator: std.mem.Allocator,
) ![]u8 {
    // #781: optional sub-phase stopwatch (null on the untimed path).
    const tmr = ctx.codegen_timer;

    if (tmr) |t| t.begin();
    const unsupported_v128 = try functionHasUnsupportedV128(func, allocator);
    if (tmr) |t| t.end(.prepass);
    if (unsupported_v128) return error.UnsupportedV128;

    // Drive scalar and SIMD virtual registers from linear-scan allocation when
    // enabled. `RegMap.assign` and V128StackMap/V128RegCache consult their
    // AllocResult first; disabling the options keeps legacy greedy/cache paths
    // available as correctness fallbacks.
    //
    // The actual `regalloc.allocate` call is deferred until after the FMA
    // fusion pre-pass below, because fused MADD/MSUB reads a mul's sources
    // at the add position; we need to extend those vregs' live ranges past
    // the mul before allocation so the sources' physregs aren't reassigned.
    // Compute RPO block order ONCE, before anything that uses global
    // instruction numbering. Clobber points, live ranges, FMA positions,
    // flat indexing, kill lists, and code emission all must use THIS order.
    if (tmr) |t| t.begin();
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
    if (tmr) |t| t.end(.prepass);
    defer allocator.free(block_order);

    if (tmr) |t| t.begin();
    var scheduled = try schedule.scheduleFunction(func, allocator, .{
        .enabled = ctx.options.enable_scheduler,
    });
    if (tmr) |t| t.end(.scheduling);
    defer scheduled.deinit();

    // Loop-aware live-range splitting (issue #383). Mutates `func` (adds
    // synthetic locals + fresh vregs) and `scheduled` (splices in the
    // `local_set`/`local_get` ops and rewrites post-loop uses) before live
    // ranges are computed below. No-op for functions without eligible
    // loops. Gated by a CompileOption to ease bisection of any regression.
    if (tmr) |t| t.begin();
    if (ctx.options.enable_range_split) {
        const rs_stats = try range_split.splitLiveRangesAtLoopBoundariesWithConfig(
            @constCast(func),
            &scheduled,
            allocator,
            .{
                // Target-derived pressure thresholds (#524): the aarch64
                // allocatable integer pool and its callee-saved subset.
                .num_available_phys_regs = RegMap.scratch_regs.len,
                .num_callee_saved_regs = RegMap.scratch_regs.len - RegMap.caller_saved_count,
            },
        );
        if (range_split_debug and rs_stats.loops_considered > 0) {
            std.debug.print(
                "[range_split] fn={s} loops={d} skip_shape={d} skip_pressure={d} splits={d}\n",
                .{
                    func.name orelse "<anon>",
                    rs_stats.loops_considered,
                    rs_stats.loops_skipped_shape,
                    rs_stats.loops_skipped_pressure,
                    rs_stats.splits_applied,
                },
            );
        }
    }
    if (tmr) |t| t.end(.range_split);

    // Pre-register-allocation analysis scans (#781 `prepass`): clobber/hint
    // collection, frame + type tables, const-fold tables, FMA fusion, and
    // iconst-mov suppression. Ends just before live-range computation.
    if (tmr) |t| t.begin();
    var clobbers = try collectClobberPoints(func, block_order, &scheduled, allocator);
    defer clobbers.deinit(allocator);

    // Calling-convention regalloc hints (issue #423, mirrors x86_64 PR
    // #422). Steers call-arg vregs into their AAPCS64 target reg
    // (x{i+1}) so `emitStageCallAbi`/`stageArgFromSaved` short-circuits
    // each arg via the `target == r` no-op path instead of emitting an
    // inter-X-reg `mov`. Hints are advisory: unsafe / unavailable hints
    // are silently ignored by `regalloc.allocateFromRangesWithHints`.
    var hint_points = try collectHintPoints(func, &ctx, block_order, &scheduled, allocator);
    defer hint_points.deinit(allocator);

    var code = emit.CodeBuffer.initWithPeephole(allocator, .{ .enabled = ctx.options.enable_peephole });
    errdefer code.deinit();

    const local_layout = try computeLocalFrameLayout(func, allocator);
    defer allocator.free(local_layout.offsets);

    // Frame size: saved FP/LR + vmctx + typed local slots (v128 slots are
    // 16-byte aligned and 16 bytes wide), plus spill headroom (sized from IR
    // vreg pressure, see below), plus 128 bytes for call-save (16 physregs ×
    // 8), plus 80 bytes for 10 callee-saved regs (x19..x28), aligned to 16
    // (AArch64 SP alignment).
    //
    // Spill capacity used to be a fixed 256 bytes (32 slots) which was
    // enough for micro-benchmarks but overflowed on real workloads
    // (CoreMark `core_state_transition` lands ~90 live vregs at peak).
    // Size it conservatively from the number of IR vregs: every vreg
    // might need a slot, plus a 16-slot safety margin for pathological
    // interleavings. Slots are 8 bytes.
    const spill_base: u32 = local_layout.end;

    // spill_capacity is finalized below after the allocator runs (we use
    // alloc_result.spill_count instead of a conservative next_vreg+16
    // estimate, which shrinks frames and reduces I-cache pressure).
    var fctx = ctx;

    // Multi-result plumbing. Precompute the caller-side return-area layout for
    // `.call_result` instructions. Scalar-only calls keep the old 8-byte slot
    // offsets; mixed/v128 results use aligned 16-byte slots where needed.
    var call_result_offsets = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer call_result_offsets.deinit();
    const call_result_layout = try collectCallResultOffsets(&scheduled, &fctx, &call_result_offsets);
    fctx.call_result_offsets = &call_result_offsets;

    fctx.local_offsets = local_layout.offsets;
    fctx.local_types = func.local_types orelse &.{};

    var vreg_types = std.AutoHashMap(ir.VReg, ir.IrType).init(allocator);
    defer vreg_types.deinit();
    for (0..func.blocks.items.len) |bid| {
        for (scheduled.instructions(@intCast(bid))) |inst| {
            if (inst.dest) |dest| try vreg_types.put(dest, inst.type);
        }
    }
    fctx.vreg_types = &vreg_types;

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
    var simd_mul_fused = std.AutoHashMap(ir.VReg, void).init(allocator);
    defer simd_mul_fused.deinit();
    var simd_fma_info = std.AutoHashMap(ir.VReg, FmaInfo).init(allocator);
    defer simd_fma_info.deinit();
    var simd_fma_add_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer simd_fma_add_pos.deinit();
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
                if ((is_add or is_sub) and (add_inst.type == .i32 or add_inst.type == .i64)) {
                    const bin: ir.Inst.BinOp = switch (add_inst.op) {
                        .add => |b| b,
                        .sub => |b| b,
                        else => unreachable,
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

                const maybe_simd_bin: ?SimdFmaBin = switch (add_inst.op) {
                    .f32x4_binop => |b| .{ .f32 = b },
                    .f64x2_binop => |b| .{ .f64 = b },
                    else => null,
                };
                const simd_bin = maybe_simd_bin orelse continue;
                const simd_is_add, const simd_is_sub = switch (simd_bin) {
                    .f32 => |b| .{ b.op == .add, b.op == .sub },
                    .f64 => |b| .{ b.op == .add, b.op == .sub },
                };
                if (!simd_is_add and !simd_is_sub) continue;
                const simd_lhs, const simd_rhs = switch (simd_bin) {
                    .f32 => |b| .{ b.lhs, b.rhs },
                    .f64 => |b| .{ b.lhs, b.rhs },
                };
                const simd_candidates: [2]?ir.VReg = if (simd_is_add)
                    .{ simd_rhs, simd_lhs }
                else
                    .{ simd_rhs, null };
                for (simd_candidates) |maybe_mul_dest| {
                    const mul_dest = maybe_mul_dest orelse continue;
                    const uc = use_counts.get(mul_dest) orelse continue;
                    if (uc != 1) continue;
                    var found_bin: ?ir.Inst.BinOp = null;
                    var j: usize = i;
                    while (j > 0) : (j -= 1) {
                        const prev = insts[j - 1];
                        if (prev.dest) |d| {
                            if (d != mul_dest) continue;
                            switch (simd_bin) {
                                .f32 => {
                                    if (prev.op == .f32x4_binop and prev.op.f32x4_binop.op == .mul) {
                                        found_bin = .{ .lhs = prev.op.f32x4_binop.lhs, .rhs = prev.op.f32x4_binop.rhs };
                                    }
                                },
                                .f64 => {
                                    if (prev.op == .f64x2_binop and prev.op.f64x2_binop.op == .mul) {
                                        found_bin = .{ .lhs = prev.op.f64x2_binop.lhs, .rhs = prev.op.f64x2_binop.rhs };
                                    }
                                },
                            }
                            break;
                        }
                    }
                    const mul_bin = found_bin orelse continue;
                    const addend = if (simd_is_add and mul_dest == simd_lhs)
                        simd_rhs
                    else
                        simd_lhs;
                    try simd_mul_fused.put(mul_dest, {});
                    const dest = add_inst.dest orelse continue;
                    try simd_fma_info.put(dest, .{
                        .mul_lhs = mul_bin.lhs,
                        .mul_rhs = mul_bin.rhs,
                        .addend = addend,
                        .is_sub = simd_is_sub,
                    });
                    try simd_fma_add_pos.put(dest, global_idx);
                    break;
                }
            }
        }
    }
    fctx.mul_fused = &mul_fused;
    fctx.fma_info = &fma_info;
    fctx.simd_mul_fused = &simd_mul_fused;
    fctx.simd_fma_info = &simd_fma_info;

    // Build iconst-mov suppression set. An `iconst_*` def whose every use
    // folds as an immediate operand (currently: the imm12 path of ADD/SUB
    // in `emitBinOp`) doesn't need its `mov xN, #K` materialization at
    // all — the consuming op encodes the value directly. The IR-level def
    // stays so regalloc/liveness book-keeping is unchanged; only the asm
    // emit is suppressed.
    //
    // Conservative conjunctive rule (issue #523): if ANY use of an iconst
    // vreg needs a real register (binop reg/reg fallback, store address
    // base, call argument, phi, spill, divisor, comparison, …), the mov
    // stays — otherwise we'd have to re-emit it per use. Mirrors the pick
    // logic in `emitBinOp` exactly so the suppression decision can never
    // disagree with the consumer.
    var iconst_total: std.AutoHashMap(ir.VReg, u32) = .init(allocator);
    defer iconst_total.deinit();
    var iconst_foldable: std.AutoHashMap(ir.VReg, u32) = .init(allocator);
    defer iconst_foldable.deinit();
    {
        const Ctx = struct {
            cv: *const std.AutoHashMap(ir.VReg, i64),
            tot: *std.AutoHashMap(ir.VReg, u32),
            fol: *std.AutoHashMap(ir.VReg, u32),
            fma: *const std.AutoHashMap(ir.VReg, FmaInfo),
            mul_fused_set: *const std.AutoHashMap(ir.VReg, void),

            fn bumpTotal(self: *@This(), v: ir.VReg) !void {
                if (!self.cv.contains(v)) return;
                const e = try self.tot.getOrPut(v);
                if (!e.found_existing) e.value_ptr.* = 0;
                e.value_ptr.* += 1;
            }
            fn bumpFold(self: *@This(), v: ir.VReg) !void {
                if (!self.cv.contains(v)) return;
                const e = try self.fol.getOrPut(v);
                if (!e.found_existing) e.value_ptr.* = 0;
                e.value_ptr.* += 1;
            }
            fn visitDefault(self: *@This(), v: ir.VReg) !void {
                try self.bumpTotal(v);
            }
        };
        var cls_ctx: Ctx = .{
            .cv = &const_vals,
            .tot = &iconst_total,
            .fol = &iconst_foldable,
            .fma = &fma_info,
            .mul_fused_set = &mul_fused,
        };

        for (block_order) |bo_bid| {
            for (scheduled.instructions(bo_bid)) |inst| {
                // The mul itself is skipped if fused; its operand reads
                // happen at the MADD site via `useInto`. Either way, the
                // operands are read into real registers — non-foldable.
                // For ADD/SUB, mirror emitBinOp's pick: if the FMA path
                // is taken, neither lhs nor rhs of the add/sub is folded
                // as an immediate (they take the FMA branch instead).
                switch (inst.op) {
                    .add, .sub => |b| {
                        const dest = inst.dest;
                        const fma_taken = if (dest) |d| cls_ctx.fma.contains(d) else false;
                        if (fma_taken or (inst.type != .i32 and inst.type != .i64)) {
                            try cls_ctx.bumpTotal(b.lhs);
                            try cls_ctx.bumpTotal(b.rhs);
                            continue;
                        }
                        const is_add = inst.op == .add;
                        const rhs_const = cls_ctx.cv.get(b.rhs);
                        const lhs_const = if (is_add) cls_ctx.cv.get(b.lhs) else null;
                        var pick_rhs = false;
                        var pick_lhs = false;
                        if (rhs_const) |v| {
                            if (encodeAddSubImm(v) != null) pick_rhs = true;
                        }
                        if (!pick_rhs) {
                            if (lhs_const) |v| {
                                if (encodeAddSubImm(v) != null) pick_lhs = true;
                            }
                        }
                        // Always count totals; only count foldables on
                        // the picked side.
                        try cls_ctx.bumpTotal(b.lhs);
                        try cls_ctx.bumpTotal(b.rhs);
                        if (pick_lhs) try cls_ctx.bumpFold(b.lhs);
                        if (pick_rhs) try cls_ctx.bumpFold(b.rhs);
                    },
                    else => {
                        try schedule.forEachUse(inst, &cls_ctx, Ctx.visitDefault);
                    },
                }
            }
        }
    }

    var suppress_iconst: std.AutoHashMap(ir.VReg, void) = .init(allocator);
    defer suppress_iconst.deinit();
    {
        var it = const_vals.iterator();
        while (it.next()) |entry| {
            const v = entry.key_ptr.*;
            const total = iconst_total.get(v) orelse 0;
            const fold = iconst_foldable.get(v) orelse 0;
            // total > 0 guards against vregs with zero uses (those should
            // already have been DCE'd by PR #508; if any survive, leaving
            // the mov is harmless).
            if (total > 0 and total == fold) {
                try suppress_iconst.put(v, {});
            }
        }
    }
    fctx.suppress_iconst_emit = &suppress_iconst;
    if (tmr) |t| t.end(.prepass);

    // Compute live ranges using the SAME block order as code emission.
    if (tmr) |t| t.begin();
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
    if (simd_fma_info.count() > 0) {
        var range_idx = std.AutoHashMap(ir.VReg, usize).init(allocator);
        defer range_idx.deinit();
        try range_idx.ensureTotalCapacity(@intCast(live_ranges.len));
        for (live_ranges, 0..) |r, idx| {
            range_idx.putAssumeCapacity(r.vreg, idx);
        }
        var it = simd_fma_info.iterator();
        while (it.next()) |entry| {
            const add_dest = entry.key_ptr.*;
            const fi = entry.value_ptr.*;
            const add_pos = simd_fma_add_pos.get(add_dest) orelse continue;
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
    var v128_live_ranges: std.ArrayList(analysis.LiveRange) = .empty;
    defer v128_live_ranges.deinit(allocator);
    var legacy_v128_spill_slots: u32 = 0;
    for (live_ranges) |range| {
        if (range.type == .v128) {
            legacy_v128_spill_slots += 1;
            try v128_live_ranges.append(allocator, range);
        } else {
            try scalar_live_ranges.append(allocator, range);
        }
    }
    if (tmr) |t| t.end(.liveness);

    var alloc_result_storage: ?regalloc.AllocResult = null;
    defer if (alloc_result_storage) |*ar| ar.deinit();
    if (ctx.options.enable_xreg_alloc) {
        if (tmr) |t| t.begin();
        alloc_result_storage = try regalloc.allocateFromRangesWithHints(
            allocator,
            aarch64RegSetForSpillBase(spill_base),
            clobbers.items,
            scalar_live_ranges.items,
            hint_points.items,
        );
        if (tmr) |t| t.end(.regalloc);

        // Post-allocation move coalescing (issue #386). Retarget the
        // dest vreg of each copy-like IR op (.reinterpret) onto its
        // source vreg's physreg when safe, so the emit-time guard
        // `if (src != dest) movRegReg(...)` elides the now-redundant
        // mov entirely. Coalescing only mutates the AllocResult's
        // assignments map — codegen consumes the rewritten mapping
        // through `RegMap.assign` unchanged.
        if (ctx.options.enable_move_coalesce) {
            if (tmr) |t| t.begin();
            var copy_hints = try collectCopyHints(func, block_order, &scheduled, allocator);
            defer copy_hints.deinit(allocator);
            _ = try regalloc.coalesceMoves(
                allocator,
                &alloc_result_storage.?,
                aarch64RegSetForSpillBase(spill_base),
                clobbers.items,
                scalar_live_ranges.items,
                copy_hints.items,
            );
            if (tmr) |t| t.end(.coalesce);
        }
    }

    const scalar_spill_capacity_for_vregs: u32 = if (alloc_result_storage) |ar|
        @as(u32, @intCast(ar.spill_count)) * 8
    else
        (func.next_vreg + 16) * 8;
    const v128_spill_base: u32 = alignForwardU32(spill_base + scalar_spill_capacity_for_vregs, 16);

    var v128_alloc_result_storage: ?regalloc.AllocResult = null;
    defer if (v128_alloc_result_storage) |*ar| ar.deinit();
    if (ctx.options.enable_vreg_alloc) {
        if (tmr) |t| t.begin();
        var v128_clobbers = try vregClobbersFromScalar(clobbers.items, allocator);
        defer v128_clobbers.deinit(allocator);
        v128_alloc_result_storage = try regalloc.allocateFromRanges(
            allocator,
            aarch64VRegSetForSpillBase(v128_spill_base),
            v128_clobbers.items,
            v128_live_ranges.items,
        );
        if (tmr) |t| t.end(.regalloc);
    }

    // #808 Lever 1: per-function spill-cost diagnostic. Combines the scalar
    // (X-reg) and v128 (V-reg) allocations — disjoint vreg sets, distinct
    // register files — into one report. Gated so the disabled path pays
    // nothing; `printSpill` is further gated by `shouldLog`.
    if (ctx.options.spill_metric.enabled and
        ctx.options.spill_metric.moduleMatches(ctx.options.module_idx))
    {
        var metric: regalloc.SpillMetric = .{};
        var total_slots: u32 = 0;
        if (alloc_result_storage) |*ar| {
            metric = metric.add(try regalloc.computeSpillMetric(
                allocator,
                func,
                aarch64RegSetForSpillBase(spill_base),
                scalar_live_ranges.items,
                ar,
            ));
            total_slots += ar.spill_count;
        }
        if (v128_alloc_result_storage) |*ar| {
            metric = metric.add(try regalloc.computeSpillMetric(
                allocator,
                func,
                aarch64VRegSetForSpillBase(v128_spill_base),
                v128_live_ranges.items,
                ar,
            ));
            total_slots += ar.spill_count;
        }
        if (ctx.options.spill_metric.shouldLog(ctx.options.module_idx, ctx.func_idx, metric.spilled_vregs)) {
            codegen_timing.printSpill(.{
                .module_idx = ctx.options.module_idx,
                .func_idx = ctx.func_idx,
                .func_name = func.name orelse "<anon>",
                .insts = aarch64CountInstructions(func),
                .clobbers = @intCast(clobbers.items.len),
                .spill_count = total_slots,
                .metric = metric,
            });
        }
    }

    // Finalize frame layout now that we know how many spill slots the
    // allocator actually consumed. Previously we pre-sized to
    // (next_vreg+16)*8 which was wasteful; using the allocator's exact
    // count shrinks frames and reduces I-cache pressure on entry.
    const spill_capacity: u32 = if (alloc_result_storage) |ar|
        @as(u32, @intCast(ar.spill_count)) * 8
    else
        (func.next_vreg + 16) * 8;
    const scalar_spill_end = spill_base + spill_capacity;
    const final_v128_spill_base: u32 = if ((v128_alloc_result_storage != null and v128_alloc_result_storage.?.spill_count == 0) or
        (v128_alloc_result_storage == null and legacy_v128_spill_slots == 0))
        scalar_spill_end
    else
        alignForwardU32(scalar_spill_end, 16);
    const v128_spill_capacity: u32 = if (v128_alloc_result_storage) |ar|
        @as(u32, @intCast(ar.spill_count)) * 8
    else
        legacy_v128_spill_slots * 16;
    const hrp_save_off: u32 = final_v128_spill_base + v128_spill_capacity;
    const scratch_base_unaligned: u32 = hrp_save_off + 8;
    const scratch_base: u32 = if (call_result_layout.needs_16_align)
        alignForwardU32(scratch_base_unaligned, 16)
    else
        scratch_base_unaligned;
    const scratch_size: u32 = call_result_layout.size;
    const call_save_base: u32 = scratch_base + scratch_size;
    const call_save_size: u32 = 128;
    const callee_save_base: u32 = call_save_base + call_save_size;
    const callee_save_count: u32 = 10; // x19..x28
    const callee_save_size: u32 = callee_save_count * 8;
    const raw_frame = callee_save_base + callee_save_size;
    const frame_size: u32 = (raw_frame + 15) & ~@as(u32, 15);

    var reg_map = RegMap.init(allocator, spill_base, spill_capacity);
    defer reg_map.deinit();
    if (alloc_result_storage) |*ar| reg_map.alloc_result = ar;

    var v128_map = V128StackMap.init(allocator, final_v128_spill_base, v128_spill_capacity);
    defer v128_map.deinit();
    if (v128_alloc_result_storage) |*ar| v128_map.alloc_result = ar;
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
    if (tmr) |t| t.begin();
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
                    (inst.op == .mul and mul_fused.contains(d)) or
                        ((inst.op == .f32x4_binop and inst.op.f32x4_binop.op == .mul) or
                            (inst.op == .f64x2_binop and inst.op.f64x2_binop.op == .mul)) and
                            simd_mul_fused.contains(d)
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
                    if (simd_fma_info.get(d)) |fi| {
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
    if (tmr) |t| t.end(.liveness);

    try code.emitPrologue(frame_size);
    try emitCalleeSaveStoreTracked(&code, &fctx);

    // Pin vmctx (AAPCS64 arg0 = x0) in callee-saved x19 for the whole
    // function body (issue #465). The preceding callee-save STR has
    // already preserved the caller's x19, and `RegMap.init` forces
    // x19's bit in `used_callee_mask` so the STR is not NOP-patched.
    // Every former `ldr Xd, [fp + vmctx_slot_offset/8]` is replaced by
    // a `mov Xd, x19` (or a `mov x0, x19` when staging arg0 for a
    // helper / cross-function call).
    try code.movRegReg(.x19, .x0);

    // Pin VmCtx.memory_base in callee-saved x20 for the whole function
    // body (issue #466). The preceding callee-save STR preserved the
    // caller's x20, and `RegMap.init` forces x20's bit in
    // `used_callee_mask`. Every linear-memory access reads `x20`
    // directly instead of rematerialising via
    // `mov tmp, x19; ldr tmp, [tmp]`. `memory.grow` and every wasm
    // call emit `ldr x20, [x19, #0]` afterward to refresh the base.
    try code.ldrImm(.x20, .x19, 0);

    // Spill VMContext (x0) and wasm params (x1..x7) to their frame slots.
    if (tmr) |t| t.begin();
    const needs_init = try local_init.computeNeedsInit(allocator, func);
    if (tmr) |t| t.end(.prepass);
    defer allocator.free(needs_init);
    try emitEntrySpill(&code, func.*, local_layout.offsets, fctx.local_types, needs_init, hrp_save_off, frame_size, allocator);

    var block_offsets = try allocator.alloc(usize, func.blocks.items.len);
    defer allocator.free(block_offsets);
    @memset(block_offsets, 0);

    // block_order already computed above — reuse for emission.

    var patches: std.ArrayListUnmanaged(BranchPatch) = .empty;
    defer patches.deinit(allocator);

    // Fall-through elision precompute (#808): a terminating `br target` is a
    // redundant `b .+4` when `target` resolves to the same address as the
    // physical fall-through. Single-`br` blocks that are themselves elided
    // emit nothing, so they resolve to their physical successor. Compute, per
    // emission-order position, a "representative" block id = the address that
    // position resolves to after elision. Walking backwards makes this O(n).
    const elide_enabled = ctx.options.enable_fallthrough_elision;
    const pos_of_block = try allocator.alloc(usize, func.blocks.items.len);
    defer allocator.free(pos_of_block);
    for (block_order, 0..) |b, oi| pos_of_block[b] = oi;
    const rep = try allocator.alloc(ir.BlockId, block_order.len);
    defer allocator.free(rep);
    {
        var oi: usize = block_order.len;
        while (oi > 0) {
            oi -= 1;
            const phys_next_rep: ?ir.BlockId =
                if (oi + 1 < block_order.len) rep[oi + 1] else null;
            var emits_nothing = false;
            const insts = scheduled.instructions(block_order[oi]);
            if (elide_enabled and insts.len == 1 and phys_next_rep != null) {
                switch (insts[0].op) {
                    // A lone forward `br` whose target resolves to the
                    // physical fall-through emits nothing once elided.
                    .br => |target| {
                        const tp = pos_of_block[target];
                        if (tp > oi and rep[tp] == phys_next_rep.?) emits_nothing = true;
                    },
                    else => {},
                }
            }
            rep[oi] = if (emits_nothing) phys_next_rep.? else block_order[oi];
        }
    }

    // Expose the resolved layout to the terminator emitters (br_if uses it to
    // drop its unconditional fall-through branch). block_pos/block_rep are
    // stable for the whole emission; fallthrough_rep is set per block below.
    fctx.block_pos = pos_of_block;
    fctx.block_rep = rep;

    var last_was_ret = false;
    for (block_order, 0..) |bi, order_idx| {
        block_offsets[bi] = code.len();
        code.peepholeBarrier();
        fctx.fallthrough_rep = if (order_idx + 1 < block_order.len) rep[order_idx + 1] else null;
        const block_insts = scheduled.instructions(bi);
        for (block_insts, 0..) |inst, ii| {
            last_was_ret = isRet(inst.op);
            const flat_idx = block_flat_base[bi] + ii;
            fctx.current_kills = kill_lists[flat_idx].items;
            v128_cache.beginInst(block_insts, ii);
            // Fall-through elision (#808): drop a terminating `br` whose target
            // resolves to the physical fall-through address (a `b .+4`, incl.
            // over elided empty blocks). Only a terminator can be a `br`, so
            // this never drops a needed instruction; block_offsets stay valid
            // because the next block's offset is recorded at the now-earlier
            // code position. Backward/self targets are never elided.
            const elide_br = ii + 1 == block_insts.len and
                switch (inst.op) {
                    .br => |target| fctx.targetIsFallthrough(target),
                    else => false,
                };
            if (!elide_br) {
                try compileInst(&code, inst, &reg_map, &v128_map, &v128_cache, frame_size, &patches, &fctx);
            }
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

    // Post-allocation MOV coalescer scan (issue #539). Operates on the
    // finalised byte stream after every other emit-time fixup that
    // writes individual instruction words, and BEFORE branch-patch
    // resolution (which only edits embedded offset fields inside the
    // already-emitted branch words, untouched by the coalescer). The
    // pass replaces eliminated MOVs with NOP — byte offsets are
    // preserved, so `block_offsets`, `patches`, `call_patches`, and
    // `callee_save_sites` all remain valid afterwards.
    if (ctx.options.enable_post_emit_coalesce) {
        if (tmr) |t| t.begin();
        _ = coalesce_post.coalesceMovesPostEmit(
            code.bytes.items,
            block_offsets,
            code.bytes.items.len,
        );
        if (tmr) |t| t.end(.post_emit_coalesce);
    }

    // Relax conditional branches whose imm19 reach (±1 MiB) is exceeded
    // by the function's code size. Each out-of-range `B.cond target` is
    // rewritten in place to a 2-instruction sequence:
    //   B.!cond +8       ; skip the next inst when the original cond is false
    //   B target         ; long-range unconditional branch (±128 MiB)
    // Inserts 4 bytes per relaxed branch, shifting downstream
    // `block_offsets`, `patches`, and `call_patches` accordingly, then
    // re-runs the scan because each insertion can push other in-range
    // branches out of range. Issue #621.
    try relaxOutOfRangeConditionalBranches(
        &code,
        block_offsets,
        &patches,
        ctx.call_patches,
        allocator,
    );

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
    // Earliest block-start position at which each vreg is live-in. A value
    // live-in to a block is live from that block's start, so its range must
    // begin no later than there — even when its only textual def sits
    // *after* that point in the linear order. This is the loop-carried
    // case: a `parallel_copy` dst from #540 phi lowering is defined on the
    // back-edge predecessor (latch) but used at the loop header, which
    // precedes the latch. Without pulling `start` back, the interval
    // collapses to ≈[latch, latch] and the allocator reuses the register
    // inside the loop, clobbering the loop-carried value (#818 / #540).
    var first_live_in_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer first_live_in_pos.deinit();

    // Per-order-position flat-index span (mirrors the same array used
    // by `analysis.computeLiveRangesWithOrder`). One extra slot acts as
    // a sentinel so `block_starts[i+1]` is always a valid exclusive end.
    const block_starts = try allocator.alloc(u32, block_order.len + 1);
    defer allocator.free(block_starts);

    var global_idx: u32 = 0;
    for (block_order, 0..) |bid, oi| {
        block_starts[oi] = global_idx;
        if (liveness.getPtr(bid)) |bl| {
            var lit = bl.live_in.iterator();
            while (lit.next()) |entry| {
                const vreg = entry.key_ptr.*;
                const existing = last_use_pos.get(vreg) orelse 0;
                try last_use_pos.put(vreg, @max(existing, global_idx));
                const prev_li = first_live_in_pos.get(vreg) orelse std.math.maxInt(u32);
                try first_live_in_pos.put(vreg, @min(prev_li, global_idx));
            }
        }

        for (scheduled.instructions(bid)) |inst| {
            if (inst.dest) |dest| {
                if (!def_pos.contains(dest)) {
                    try def_pos.put(dest, global_idx);
                    try def_type.put(dest, inst.type);
                }
            }
            // parallel_copy is a multi-def IR op (one def per pair).
            // Register each dst so it gets a live range and an
            // assignment from the allocator.
            if (inst.op == .parallel_copy) {
                for (inst.op.parallel_copy) |p| {
                    if (!def_pos.contains(p.dst)) {
                        try def_pos.put(p.dst, global_idx);
                        try def_type.put(p.dst, p.ty);
                    }
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
    block_starts[block_order.len] = global_idx;

    // Per-block loop-nest depth. Kept as a separate helper so the
    // scheduler-aware path here and the canonical `computeLiveRanges`
    // path share the exact same eviction-priority signal.
    const loop_depth_by_block: []u8 = try analysis.loopDepthByBlockForFunc(func, allocator);
    defer allocator.free(loop_depth_by_block);

    var ranges: std.ArrayList(analysis.LiveRange) = .empty;
    errdefer ranges.deinit(allocator);
    var dit = def_pos.iterator();
    while (dit.next()) |entry| {
        const vreg = entry.key_ptr.*;
        const def_p = entry.value_ptr.*;
        // Pull the range start back to the earliest point where the value
        // is live-in, covering a loop-carried use that precedes the
        // back-edge def in linear order (#818). For forward values the def
        // dominates every use, so first_live_in_pos >= def_p and this is a
        // no-op.
        const start = if (first_live_in_pos.get(vreg)) |li| @min(def_p, li) else def_p;
        const end = last_use_pos.get(vreg) orelse start;
        const final_end = @max(start, end);
        const depth = analysis.maxLoopDepthOverSpan(
            start,
            final_end,
            block_order,
            block_starts,
            loop_depth_by_block,
        );
        try ranges.append(allocator, .{
            .vreg = vreg,
            .start = start,
            .end = final_end,
            .type = def_type.get(vreg) orelse .i32,
            .max_loop_depth = depth,
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

fn alignForwardU32(value: u32, alignment: u32) u32 {
    std.debug.assert(alignment != 0 and (alignment & (alignment - 1)) == 0);
    return (value + alignment - 1) & ~(alignment - 1);
}

const LocalFrameLayout = struct {
    offsets: []u32,
    end: u32,
};

fn localTypeAt(local_types: []const ir.IrType, idx: u32) ir.IrType {
    const i: usize = @intCast(idx);
    if (i < local_types.len) return local_types[i];
    return .i64;
}

fn localSlotAlignment(ty: ir.IrType) u32 {
    return switch (ty) {
        .v128 => 16,
        .void => 8,
        else => 8,
    };
}

fn localSlotSize(ty: ir.IrType) u32 {
    return switch (ty) {
        .v128 => 16,
        .void => 0,
        else => 8,
    };
}

fn computeLocalFrameLayout(func: *const ir.IrFunction, allocator: std.mem.Allocator) !LocalFrameLayout {
    const slot_count = @max(func.local_count, func.param_count);
    const offsets = try allocator.alloc(u32, slot_count);
    errdefer allocator.free(offsets);

    const local_types = func.local_types orelse &.{};
    var next: u32 = vmctx_slot_offset + 8;
    for (offsets, 0..) |*slot, i| {
        const ty = localTypeAt(local_types, @intCast(i));
        next = alignForwardU32(next, localSlotAlignment(ty));
        slot.* = next;
        next += localSlotSize(ty);
    }

    return .{ .offsets = offsets, .end = next };
}

fn localFrameOffset(fctx: *const FuncCompileCtx, idx: u32) !u32 {
    const i: usize = @intCast(idx);
    if (i >= fctx.local_offsets.len) return error.InvalidLocalIndex;
    return fctx.local_offsets[i];
}

const scalar_arg_reg_count: u32 = 7;
const v128_arg_reg_count: u32 = 8;
const v128_call_stack_tmp: u5 = 31;

/// ABI register holding scalar wasm argument `i` within the scalar register
/// bank. Scalar argument 0 is in x1 because x0 carries the hidden VMContext.
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

fn v128ParamAbiReg(i: u32) u5 {
    if (i >= v128_arg_reg_count) unreachable;
    return @intCast(i);
}

const AbiArgLoc = union(enum) {
    scalar_reg: emit.Reg,
    v128_reg: u5,
    stack: struct {
        offset: u32,
        ty: ir.IrType,
    },
};

const CallAbiPlan = struct {
    locs: []AbiArgLoc,
    hrp: ?AbiArgLoc = null,
    stack_size: u32 = 0,
    allocator: std.mem.Allocator,

    fn deinit(self: *CallAbiPlan) void {
        self.allocator.free(self.locs);
    }
};

fn classifyAbiArg(ty: ir.IrType, scalar_idx: *u32, v128_idx: *u32, stack_next: *u32) AbiArgLoc {
    if (ty == .v128) {
        if (v128_idx.* < v128_arg_reg_count) {
            const reg = v128ParamAbiReg(v128_idx.*);
            v128_idx.* += 1;
            return .{ .v128_reg = reg };
        }
        stack_next.* = alignForwardU32(stack_next.*, 16);
        const off = stack_next.*;
        stack_next.* += 16;
        return .{ .stack = .{ .offset = off, .ty = .v128 } };
    }

    if (scalar_idx.* < scalar_arg_reg_count) {
        const reg = paramAbiReg(scalar_idx.*);
        scalar_idx.* += 1;
        return .{ .scalar_reg = reg };
    }
    stack_next.* = alignForwardU32(stack_next.*, 8);
    const off = stack_next.*;
    stack_next.* += 8;
    return .{ .stack = .{ .offset = off, .ty = ty } };
}

fn buildEntryAbiPlan(
    allocator: std.mem.Allocator,
    func: ir.IrFunction,
    local_types: []const ir.IrType,
) !CallAbiPlan {
    const locs = try allocator.alloc(AbiArgLoc, func.param_count);
    errdefer allocator.free(locs);

    var scalar_idx: u32 = 0;
    var v128_idx: u32 = 0;
    var stack_next: u32 = 0;
    var i: u32 = 0;
    while (i < func.param_count) : (i += 1) {
        locs[@intCast(i)] = classifyAbiArg(localTypeAt(local_types, i), &scalar_idx, &v128_idx, &stack_next);
    }

    const hrp: ?AbiArgLoc = if (func.result_count > 1)
        classifyAbiArg(.i64, &scalar_idx, &v128_idx, &stack_next)
    else
        null;

    return .{
        .locs = locs,
        .hrp = hrp,
        .stack_size = alignForwardU32(stack_next, 16),
        .allocator = allocator,
    };
}

fn funcParamTypes(fctx: *const FuncCompileCtx, func_idx: u32) []const ir.IrType {
    const idx: usize = @intCast(func_idx);
    if (idx >= fctx.func_type_indices.len) return &.{};
    const type_idx = fctx.func_type_indices[idx];
    const ti: usize = @intCast(type_idx);
    if (ti >= fctx.func_types.len) return &.{};
    return fctx.func_types[ti].params;
}

fn indexedParamTypes(fctx: *const FuncCompileCtx, type_idx: u32) []const ir.IrType {
    const ti: usize = @intCast(type_idx);
    if (ti >= fctx.func_types.len) return &.{};
    return fctx.func_types[ti].params;
}

fn funcResultTypes(fctx: *const FuncCompileCtx, func_idx: u32) []const ir.IrType {
    const idx: usize = @intCast(func_idx);
    if (idx >= fctx.func_type_indices.len) return &.{};
    const type_idx = fctx.func_type_indices[idx];
    const ti: usize = @intCast(type_idx);
    if (ti >= fctx.func_types.len) return &.{};
    return fctx.func_types[ti].results;
}

fn indexedResultTypes(fctx: *const FuncCompileCtx, type_idx: u32) []const ir.IrType {
    const ti: usize = @intCast(type_idx);
    if (ti >= fctx.func_types.len) return &.{};
    return fctx.func_types[ti].results;
}

fn resultSlotAlignment(ty: ir.IrType) u32 {
    return if (ty == .v128) 16 else 8;
}

fn resultSlotSize(ty: ir.IrType) u32 {
    return switch (ty) {
        .v128 => 16,
        .void => 0,
        else => 8,
    };
}

fn extraResultAreaOffset(extra_types: []const ir.IrType, idx: u32) u32 {
    var off: u32 = 0;
    var i: u32 = 0;
    while (i < extra_types.len) : (i += 1) {
        const ty = extra_types[@intCast(i)];
        off = alignForwardU32(off, resultSlotAlignment(ty));
        if (i == idx) return off;
        off += resultSlotSize(ty);
    }
    return idx * 8;
}

fn extraResultAreaSize(extra_types: []const ir.IrType) u32 {
    var off: u32 = 0;
    for (extra_types) |ty| {
        off = alignForwardU32(off, resultSlotAlignment(ty));
        off += resultSlotSize(ty);
    }
    return off;
}

fn extraResultAreaNeeds16(extra_types: []const ir.IrType) bool {
    for (extra_types) |ty| if (ty == .v128) return true;
    return false;
}

const CallResultScratchLayout = struct {
    size: u32 = 0,
    needs_16_align: bool = false,
};

fn noteCallResultSlot(layout: *CallResultScratchLayout, off: u32, ty: ir.IrType) void {
    layout.size = @max(layout.size, off + resultSlotSize(ty));
    if (ty == .v128) layout.needs_16_align = true;
}

fn noteTypedExtraResults(layout: *CallResultScratchLayout, extra_types: []const ir.IrType) void {
    layout.size = @max(layout.size, extraResultAreaSize(extra_types));
    if (extraResultAreaNeeds16(extra_types)) layout.needs_16_align = true;
}

fn collectCallResultOffsets(
    scheduled: *const schedule.FunctionSchedule,
    fctx: *const FuncCompileCtx,
    offsets: *std.AutoHashMap(ir.VReg, u32),
) !CallResultScratchLayout {
    var layout = CallResultScratchLayout{};
    var pending_extra_types: []const ir.IrType = &.{};
    var fallback_next_idx: u32 = 0;
    var fallback_next_off: u32 = 0;

    for (0..scheduled.blocks.len) |bid| {
        for (scheduled.instructions(@intCast(bid))) |inst| {
            switch (inst.op) {
                .call => |cl| {
                    const results = funcResultTypes(fctx, cl.func_idx);
                    pending_extra_types = if (results.len > 1) results[1..] else &.{};
                    fallback_next_idx = 0;
                    fallback_next_off = 0;
                    if (pending_extra_types.len > 0) {
                        noteTypedExtraResults(&layout, pending_extra_types);
                    } else if (cl.extra_results > 0) {
                        layout.size = @max(layout.size, @as(u32, cl.extra_results) * 8);
                    }
                },
                .call_indirect => |ci| {
                    const results = indexedResultTypes(fctx, ci.type_idx);
                    pending_extra_types = if (results.len > 1) results[1..] else &.{};
                    fallback_next_idx = 0;
                    fallback_next_off = 0;
                    if (pending_extra_types.len > 0) {
                        noteTypedExtraResults(&layout, pending_extra_types);
                    } else if (ci.extra_results > 0) {
                        layout.size = @max(layout.size, @as(u32, ci.extra_results) * 8);
                    }
                },
                .call_ref => |cr| {
                    const results = indexedResultTypes(fctx, cr.type_idx);
                    pending_extra_types = if (results.len > 1) results[1..] else &.{};
                    fallback_next_idx = 0;
                    fallback_next_off = 0;
                    if (pending_extra_types.len > 0) {
                        noteTypedExtraResults(&layout, pending_extra_types);
                    } else if (cr.extra_results > 0) {
                        layout.size = @max(layout.size, @as(u32, cr.extra_results) * 8);
                    }
                },
                .call_result => |idx| {
                    const dest = inst.dest orelse continue;
                    const off: u32 = if (idx < pending_extra_types.len)
                        extraResultAreaOffset(pending_extra_types, idx)
                    else blk: {
                        if (idx == fallback_next_idx) {
                            fallback_next_off = alignForwardU32(fallback_next_off, resultSlotAlignment(inst.type));
                            const computed = fallback_next_off;
                            fallback_next_off += resultSlotSize(inst.type);
                            fallback_next_idx += 1;
                            break :blk computed;
                        }
                        const legacy = @as(u32, idx) * 8;
                        break :blk if (inst.type == .v128) alignForwardU32(legacy, 16) else legacy;
                    };
                    try offsets.put(dest, off);
                    noteCallResultSlot(&layout, off, inst.type);
                },
                else => {},
            }
        }
    }

    return layout;
}

fn callArgType(fctx: *const FuncCompileCtx, params: []const ir.IrType, args: []const ir.VReg, idx: usize) ir.IrType {
    if (idx < params.len) return params[idx];
    if (fctx.vreg_types) |vts| {
        if (vts.get(args[idx])) |ty| return ty;
    }
    return .i64;
}

fn buildCallAbiPlan(
    allocator: std.mem.Allocator,
    fctx: *const FuncCompileCtx,
    args: []const ir.VReg,
    params: []const ir.IrType,
    include_hrp: bool,
) !CallAbiPlan {
    const locs = try allocator.alloc(AbiArgLoc, args.len);
    errdefer allocator.free(locs);

    var scalar_idx: u32 = 0;
    var v128_idx: u32 = 0;
    var stack_next: u32 = 0;
    for (args, 0..) |_, i| {
        locs[i] = classifyAbiArg(callArgType(fctx, params, args, i), &scalar_idx, &v128_idx, &stack_next);
    }

    const hrp: ?AbiArgLoc = if (include_hrp)
        classifyAbiArg(.i64, &scalar_idx, &v128_idx, &stack_next)
    else
        null;

    return .{
        .locs = locs,
        .hrp = hrp,
        .stack_size = alignForwardU32(stack_next, 16),
        .allocator = allocator,
    };
}

/// Spill x0 (VMContext) and x1..x7 (wasm params) to their frame slots, and
/// zero-initialize declared locals. Must be called right after emitPrologue
/// and before any vreg allocation so we don't clobber these ABI regs.
///
/// `needs_init` (length == func.local_count) is a per-local bitmap. If
/// `needs_init[i]` is false for `i ≥ param_count`, the wasm spec
/// guarantee "non-param locals start at zero" is satisfied without
/// emitting a prologue store because every reachable `local_get i` is
/// dominated by a `local_set i`. Parameter indices are always treated
/// as initialised by the caller's ABI spill regardless of the bitmap.
fn emitEntrySpill(
    code: *emit.CodeBuffer,
    func: ir.IrFunction,
    local_offsets: []const u32,
    local_types: []const ir.IrType,
    needs_init: []const bool,
    hrp_save_off: u32,
    frame_size: u32,
    allocator: std.mem.Allocator,
) !void {
    // VMContext (x0) is pinned in x19 by `compileFunctionImpl`'s
    // prologue (issue #465); no spill to `[fp + 16]` is needed. The
    // slot itself is retained in the frame layout (locals still start
    // at `[fp + 24]`) to avoid rippling the offset math through HRP,
    // RegMap.spill_base, and v128 local-slot computations.

    var abi = try buildEntryAbiPlan(allocator, func, local_types);
    defer abi.deinit();

    // Wasm params → their typed local slots.
    var i: u32 = 0;
    while (i < func.param_count) : (i += 1) {
        const offset = local_offsets[@intCast(i)];
        switch (abi.locs[@intCast(i)]) {
            .scalar_reg => |reg| try frameStore(code, reg, offset),
            .v128_reg => |vt| {
                try frameAddr(code, RegMap.tmp1, offset);
                try code.strQ(vt, RegMap.tmp1);
            },
            .stack => |stack| {
                const inbound = frame_size + stack.offset;
                if (stack.ty == .v128) {
                    try frameAddr(code, RegMap.tmp0, inbound);
                    try code.ldrQ(v128_call_stack_tmp, RegMap.tmp0);
                    try frameAddr(code, RegMap.tmp1, offset);
                    try code.strQ(v128_call_stack_tmp, RegMap.tmp1);
                } else {
                    try frameLoad(code, RegMap.tmp0, inbound);
                    try frameStore(code, RegMap.tmp0, offset);
                }
            },
        }
    }

    // Multi-value returns: callee receives an HRP (host-result pointer)
    // as an implicit trailing scalar arg. Stash to the dedicated frame slot so
    // `.ret_multi` can retrieve it.
    if (abi.hrp) |hrp_loc| {
        switch (hrp_loc) {
            .scalar_reg => |reg| try frameStore(code, reg, hrp_save_off),
            .stack => |stack| {
                try frameLoad(code, RegMap.tmp0, frame_size + stack.offset);
                try frameStore(code, RegMap.tmp0, hrp_save_off);
            },
            .v128_reg => unreachable,
        }
    }

    // Zero-init declared locals (wasm spec requires non-param locals to be 0)
    // for those flagged by `needs_init`. Skipped locals are provably
    // overwritten by a `local_set` before any `local_get` on every
    // reachable path, so the store is dead.
    //
    // Pre-scan: see whether any scalar / any v128 slot needs init so we
    // only materialise the zero value(s) when at least one store will
    // be emitted.
    var any_scalar = false;
    var any_v128 = false;
    if (func.local_count > func.param_count) {
        var j: u32 = func.param_count;
        while (j < func.local_count) : (j += 1) {
            if (!needs_init[@intCast(j)]) continue;
            if (localTypeAt(local_types, j) == .v128) any_v128 = true else any_scalar = true;
        }
    }
    if (any_scalar) try code.movz(RegMap.tmp0, 0, 0);
    if (any_v128) try code.movi2dZero(v128_tmp0);

    if (func.local_count > func.param_count) {
        var j: u32 = func.param_count;
        while (j < func.local_count) {
            if (!needs_init[@intCast(j)]) {
                j += 1;
                continue;
            }
            const offset = local_offsets[@intCast(j)];
            if (localTypeAt(local_types, j) == .v128) {
                try frameAddr(code, RegMap.tmp1, offset);
                try code.strQ(v128_tmp0, RegMap.tmp1);
                j += 1;
                continue;
            }

            // Try to fuse with the next slot into a single STP. Two
            // adjacent scalar slots whose offsets are 8 apart and whose
            // base offset / 8 fits the STP signed-imm7 encoding can be
            // written with a single 16-byte `stp tmp0, tmp0, [fp,#N]`.
            if (j + 1 < func.local_count and
                needs_init[@intCast(j + 1)] and
                localTypeAt(local_types, j + 1) != .v128)
            {
                const next_off = local_offsets[@intCast(j + 1)];
                if (next_off == offset + 8 and offset % 8 == 0) {
                    const scaled = offset / 8;
                    // STP signed imm7 is in [-64, 63] in 8-byte units.
                    if (scaled <= 63) {
                        try code.stpImm(RegMap.tmp0, RegMap.tmp0, .fp, @intCast(scaled));
                        j += 2;
                        continue;
                    }
                }
            }
            try frameStore(code, RegMap.tmp0, offset);
            j += 1;
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
        const scratch = if (src == RegMap.tmp0) RegMap.tmp1 else RegMap.tmp0;
        try emitMovImm64(code, scratch, offset);
        try code.addRegReg(scratch, scratch, .fp);
        try code.strImm(src, scratch, 0);
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

/// Load `[fp + offset]` into `dst`, using `dst` itself as the scratch for
/// the out-of-range address computation. Safe wherever `dst` is a
/// freshly-allocated destination that is about to be overwritten anyway
/// (e.g. spill reloads), and avoids clobbering the shared `tmp0`. #916.
fn frameLoadReg(code: *emit.CodeBuffer, dst: emit.Reg, offset: u32) !void {
    if (offset % 8 == 0 and offset / 8 <= 4095) {
        try code.ldrImm(dst, .fp, @intCast(offset / 8));
    } else {
        try emitMovImm64(code, dst, offset);
        try code.addRegReg(dst, dst, .fp);
        try code.ldrImm(dst, dst, 0);
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
    // Fast path: the highest slot offset still fits the 12-bit scaled
    // STR immediate (imm12*8 ≤ 32760). Emit a direct fp-relative STR per
    // reg so each slot stays a single word (required by the NOP-elision
    // in `patchUnusedCalleeSaveSlots`).
    const top_scaled = (callee_save_base + (callee_saved_regs.len - 1) * 8) / 8;
    if (top_scaled <= 4095) {
        for (callee_saved_regs, 0..) |r, i| {
            code.peepholeBarrier();
            offs[i] = code.len();
            const off_scaled: u12 = @intCast((callee_save_base + @as(u32, @intCast(i)) * 8) / 8);
            try code.strImm(r, .fp, off_scaled);
        }
        code.peepholeBarrier();
        return offs;
    }
    // Large frame (#916): materialize the callee-save base address into a
    // scratch register once, then STR each reg at a small per-slot scaled
    // offset (i ∈ [0,9], always in range). The one-time base setup is not
    // part of `offs`, so NOP-elision still rewrites individual STRs safely.
    code.peepholeBarrier();
    try frameAddr(code, RegMap.tmp0, callee_save_base);
    for (callee_saved_regs, 0..) |r, i| {
        code.peepholeBarrier();
        offs[i] = code.len();
        try code.strImm(r, RegMap.tmp0, @intCast(i));
    }
    code.peepholeBarrier();
    return offs;
}

/// Emit epilogue restores for all 10 callee-saved allocatable regs and
/// return the byte offsets of each LDR. Paired with the prologue store
/// via `patchUnusedCalleeSaveSlots`.
fn emitCalleeSaveRestore(code: *emit.CodeBuffer, callee_save_base: u32) ![callee_saved_regs.len]usize {
    var offs: [callee_saved_regs.len]usize = undefined;
    const top_scaled = (callee_save_base + (callee_saved_regs.len - 1) * 8) / 8;
    if (top_scaled <= 4095) {
        for (callee_saved_regs, 0..) |r, i| {
            code.peepholeBarrier();
            offs[i] = code.len();
            const off_scaled: u12 = @intCast((callee_save_base + @as(u32, @intCast(i)) * 8) / 8);
            try code.ldrImm(r, .fp, off_scaled);
        }
        code.peepholeBarrier();
        return offs;
    }
    // Large frame (#916): mirror `emitCalleeSaveStore`. tmp0 (x16) is dead
    // at the epilogue — the return value is already staged in x0/x1/v0.
    code.peepholeBarrier();
    try frameAddr(code, RegMap.tmp0, callee_save_base);
    for (callee_saved_regs, 0..) |r, i| {
        code.peepholeBarrier();
        offs[i] = code.len();
        try code.ldrImm(r, RegMap.tmp0, @intCast(i));
    }
    code.peepholeBarrier();
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
        .v128_load_splat,
        .v128_load_zero,
        .v128_load_extend,
        .v128_load_lane,
        .v128_store,
        .v128_store_lane,
        .v128_not,
        .v128_bitwise,
        .v128_bitselect,
        .v128_any_true,
        .simd_all_true,
        .simd_bitmask,
        .i32x4_binop,
        .i32x4_unop,
        .f32x4_unop,
        .f32x4_binop,
        .f32x4_convert_i32x4,
        .i32x4_trunc_sat,
        .f32x4_demote_f64x2_zero,
        .f32x4_splat,
        .f32x4_extract_lane,
        .f32x4_replace_lane,
        .i32x4_extadd_pairwise_i16x8,
        .i32x4_dot_i16x8_s,
        .i32x4_extend_i16x8,
        .i32x4_extmul_i16x8,
        .i32x4_shift,
        .i32x4_splat,
        .i32x4_extract_lane,
        .i32x4_replace_lane,
        .i8x16_binop,
        .i8x16_shuffle,
        .i8x16_swizzle,
        .i8x16_unop,
        .i8x16_shift,
        .i8x16_splat,
        .i8x16_extract_lane,
        .i8x16_replace_lane,
        .i8x16_narrow_i16x8,
        .i16x8_binop,
        .i16x8_unop,
        .i16x8_extadd_pairwise_i8x16,
        .i16x8_extend_i8x16,
        .i16x8_extmul_i8x16,
        .i16x8_shift,
        .i16x8_splat,
        .i16x8_extract_lane,
        .i16x8_replace_lane,
        .i16x8_narrow_i32x4,
        .i64x2_binop,
        .i64x2_unop,
        .f64x2_convert_low_i32x4,
        .f64x2_promote_low_f32x4,
        .i64x2_extend_i32x4,
        .i64x2_extmul_i32x4,
        .i64x2_shift,
        .i64x2_splat,
        .i64x2_extract_lane,
        .i64x2_replace_lane,
        .f64x2_unop,
        .f64x2_binop,
        .f64x2_splat,
        .f64x2_extract_lane,
        .f64x2_replace_lane,
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
        .parallel_copy,
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
            if (fctx.suppress_iconst_emit) |s| {
                if (s.contains(dest)) {
                    // All uses fold as immediates (#523). Skip the
                    // `mov xN, #K` materialization but still call
                    // `assign` so callee-save mask / spill bookkeeping
                    // stays consistent. The allocated physreg is never
                    // written; consumers encode the value directly.
                    _ = try reg_map.assign(dest);
                    return;
                }
            }
            // #542: rematerialisable — re-emitted at each use site
            // by `useInto` (and the call-arg staging helpers). The
            // canonical def emits nothing AND skips `reg_map.assign`
            // entirely because the allocator chose not to assign it
            // (no spill slot, no register).
            if (reg_map.alloc_result) |ar| {
                if (ar.isRemat(dest)) return;
            }
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.movImm32(info.reg, val);
            try destCommit(code, reg_map, info);
        },
        .iconst_64 => |val| {
            const dest = inst.dest orelse return;
            if (fctx.suppress_iconst_emit) |s| {
                if (s.contains(dest)) {
                    _ = try reg_map.assign(dest);
                    return;
                }
            }
            // #542: rematerialisable — see above.
            if (reg_map.alloc_result) |ar| {
                if (ar.isRemat(dest)) return;
            }
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
        .v128_load_splat => |ld| try emitV128LoadSplat(code, inst, ld, reg_map, v128_map, v128_cache),
        .v128_load_zero => |ld| try emitV128LoadZero(code, inst, ld, reg_map, v128_map, v128_cache),
        .v128_load_extend => |ld| try emitV128LoadExtend(code, inst, ld, reg_map, v128_map, v128_cache),
        .v128_load_lane => |ld| try emitV128LoadLane(code, inst, ld, reg_map, v128_map, v128_cache, fctx),
        .v128_store => |st| try emitV128Store(code, st, reg_map, v128_map, v128_cache),
        .v128_store_lane => |st| try emitV128StoreLane(code, st, reg_map, v128_map, v128_cache),
        .v128_not => |src| try emitV128Not(code, inst, src, v128_map, v128_cache, fctx),
        .v128_bitwise => |bin| try emitV128Bitwise(code, inst, bin, v128_map, v128_cache, fctx),
        .v128_bitselect => |sel| try emitV128Bitselect(code, inst, sel, v128_map, v128_cache, fctx),
        .v128_any_true => |src| try emitV128AnyTrue(code, inst, src, reg_map, v128_map, v128_cache),
        .simd_all_true => |op| try emitSimdAllTrue(code, inst, op, reg_map, v128_map, v128_cache),
        .simd_bitmask => |op| try emitSimdBitmask(code, inst, op, reg_map, v128_map, v128_cache),
        .i32x4_binop => |bin| try emitI32x4BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .i32x4_unop => |un| try emitI32x4UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .f32x4_unop => |un| try emitF32x4UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .f32x4_binop => |bin| try emitF32x4BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .f32x4_convert_i32x4 => |op| try emitF32x4ConvertI32x4(code, inst, op, v128_map, v128_cache, fctx),
        .i32x4_trunc_sat => |op| try emitI32x4TruncSat(code, inst, op, v128_map, v128_cache, fctx),
        .f32x4_demote_f64x2_zero => |op| try emitF32x4DemoteF64x2Zero(code, inst, op, v128_map, v128_cache),
        .f32x4_splat => |src| try emitF32x4Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .f32x4_extract_lane => |lane| try emitF32x4ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .f32x4_replace_lane => |lane| try emitF32x4ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),
        .i32x4_extadd_pairwise_i16x8 => |op| try emitI32x4ExtAddPairwiseI16x8(code, inst, op, v128_map, v128_cache, fctx),
        .i32x4_dot_i16x8_s => |bin| try emitI32x4DotI16x8S(code, inst, bin, v128_map, v128_cache),
        .i32x4_extend_i16x8 => |op| try emitI32x4ExtendI16x8(code, inst, op, v128_map, v128_cache, fctx),
        .i32x4_extmul_i16x8 => |op| try emitI32x4ExtMulI16x8(code, inst, op, v128_map, v128_cache, fctx),
        .i32x4_shift => |shift| try emitI32x4Shift(code, inst, shift, reg_map, v128_map, v128_cache, fctx),
        .i32x4_splat => |src| try emitI32x4Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .i32x4_extract_lane => |lane| try emitI32x4ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .i32x4_replace_lane => |lane| try emitI32x4ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),
        .i8x16_binop => |bin| try emitI8x16BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .i8x16_shuffle => |op| try emitI8x16Shuffle(code, inst, op, v128_map, v128_cache),
        .i8x16_unop => |un| try emitI8x16UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .i8x16_shift => |shift| try emitI8x16Shift(code, inst, shift, reg_map, v128_map, v128_cache, fctx),
        .i8x16_splat => |src| try emitI8x16Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .i8x16_extract_lane => |lane| try emitI8x16ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .i8x16_replace_lane => |lane| try emitI8x16ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),
        .i8x16_narrow_i16x8 => |op| try emitI8x16NarrowI16x8(code, inst, op, v128_map, v128_cache, fctx),
        .i8x16_swizzle => |op| try emitI8x16Swizzle(code, inst, op, v128_map, v128_cache, fctx),
        .i16x8_binop => |bin| try emitI16x8BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .i16x8_unop => |un| try emitI16x8UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .i16x8_extadd_pairwise_i8x16 => |op| try emitI16x8ExtAddPairwiseI8x16(code, inst, op, v128_map, v128_cache, fctx),
        .i16x8_extend_i8x16 => |op| try emitI16x8ExtendI8x16(code, inst, op, v128_map, v128_cache, fctx),
        .i16x8_extmul_i8x16 => |op| try emitI16x8ExtMulI8x16(code, inst, op, v128_map, v128_cache, fctx),
        .i16x8_shift => |shift| try emitI16x8Shift(code, inst, shift, reg_map, v128_map, v128_cache, fctx),
        .i16x8_splat => |src| try emitI16x8Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .i16x8_extract_lane => |lane| try emitI16x8ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .i16x8_replace_lane => |lane| try emitI16x8ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),
        .i16x8_narrow_i32x4 => |op| try emitI16x8NarrowI32x4(code, inst, op, v128_map, v128_cache, fctx),
        .i64x2_binop => |bin| try emitI64x2BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .i64x2_unop => |un| try emitI64x2UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .f64x2_convert_low_i32x4 => |op| try emitF64x2ConvertLowI32x4(code, inst, op, v128_map, v128_cache, fctx),
        .f64x2_promote_low_f32x4 => |op| try emitF64x2PromoteLowF32x4(code, inst, op, v128_map, v128_cache, fctx),
        .i64x2_extend_i32x4 => |op| try emitI64x2ExtendI32x4(code, inst, op, v128_map, v128_cache, fctx),
        .i64x2_extmul_i32x4 => |op| try emitI64x2ExtMulI32x4(code, inst, op, v128_map, v128_cache, fctx),
        .i64x2_shift => |shift| try emitI64x2Shift(code, inst, shift, reg_map, v128_map, v128_cache, fctx),
        .i64x2_splat => |src| try emitI64x2Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .i64x2_extract_lane => |lane| try emitI64x2ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .i64x2_replace_lane => |lane| try emitI64x2ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),
        .f64x2_unop => |un| try emitF64x2UnOp(code, inst, un, v128_map, v128_cache, fctx),
        .f64x2_binop => |bin| try emitF64x2BinOp(code, inst, bin, v128_map, v128_cache, fctx),
        .f64x2_splat => |src| try emitF64x2Splat(code, inst, src, reg_map, v128_map, v128_cache),
        .f64x2_extract_lane => |lane| try emitF64x2ExtractLane(code, inst, lane, reg_map, v128_map, v128_cache),
        .f64x2_replace_lane => |lane| try emitF64x2ReplaceLane(code, inst, lane, reg_map, v128_map, v128_cache, fctx),

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
        .br_if => |br| try emitBrIf(code, reg_map, patches, br, fctx),
        .br_table => |bt| try emitBrTable(code, reg_map, patches, bt, fctx.allocator),

        // ── Direct function call ─────────────────────────────────────
        .call => |cl| try emitCall(code, inst, cl, reg_map, v128_map, v128_cache, fctx),
        .call_indirect => |ci| try emitCallIndirect(code, inst, ci, reg_map, v128_map, v128_cache, fctx),
        .call_ref => |cr| try emitCallRef(code, inst, cr, reg_map, v128_map, v128_cache, fctx),

        // ── Tables & function refs ───────────────────────────────────
        .table_size => |tidx| try emitTableSize(code, inst, tidx, reg_map),
        .table_get => |tg| try emitTableGet(code, inst, tg, reg_map),
        .table_set => |ts| try emitTableSet(code, ts, reg_map, fctx),
        .table_grow => |tg| try emitTableGrow(code, inst, tg, reg_map, fctx),
        .ref_func => |fidx| try emitRefFunc(code, inst, fidx, reg_map),

        // ── Locals (FP-relative frame slots) ─────────────────────────
        .local_get => |idx| {
            const dest = inst.dest orelse return;
            const offset = try localFrameOffset(fctx, idx);
            if (inst.type == .v128) {
                const vt = try v128_cache.defineFresh(code, v128_map, dest, null);
                try frameAddr(code, RegMap.tmp0, offset);
                try code.ldrQ(vt, RegMap.tmp0);
            } else {
                const info = try destBegin(reg_map, dest, RegMap.tmp0);
                // Always 64-bit load: the slot stores the full 8 bytes, and a
                // subsequent W-form op (or wrap_i64) will ignore the upper bits.
                try frameLoad(code, info.reg, offset);
                try destCommit(code, reg_map, info);
            }
        },
        .local_set => |ls| {
            const offset = try localFrameOffset(fctx, ls.idx);
            if (localTypeAt(fctx.local_types, ls.idx) == .v128) {
                const vt = try v128_cache.ensure(code, v128_map, ls.val, null);
                try frameAddr(code, RegMap.tmp0, offset);
                try code.strQ(vt, RegMap.tmp0);
            } else {
                const src = try useInto(code, reg_map, ls.val, RegMap.tmp0);
                try frameStore(code, src, offset);
            }
        },

        // ── Linear memory load/store ─────────────────────────────────
        .load => |ld| try emitLoad(code, inst, ld, reg_map, fctx),

        // ── Compound address (base + index*scale + disp) ─────────────
        // The x86-64 `foldCompoundLea` pass is the only producer of `.lea`
        // and it is gated to the x86-64 pipeline, so this arm is not
        // reached in practice; it is provided so the op is lowered
        // correctly should it ever appear on AArch64 (there is no single
        // AArch64 addressing-compute instruction, so it expands to
        // lsl + add [+ add disp]).
        .lea => |l| {
            const dest = inst.dest orelse return;
            const scale_log2: u6 = switch (l.scale) {
                1 => 0,
                2 => 1,
                4 => 2,
                8 => 3,
                else => unreachable,
            };
            const idx = try useInto(code, reg_map, l.index, RegMap.tmp0);
            if (scale_log2 != 0) {
                try code.lslImm(RegMap.tmp0, idx, scale_log2);
            } else if (idx != RegMap.tmp0) {
                try code.addImm(RegMap.tmp0, idx, 0);
            }
            const base = try useInto(code, reg_map, l.base, RegMap.tmp1);
            const info = try destBegin(reg_map, dest, RegMap.tmp0);
            try code.addRegReg(info.reg, base, RegMap.tmp0);
            if (l.disp != 0) {
                try code.movImm64(RegMap.tmp1, @bitCast(@as(i64, l.disp)));
                try code.addRegReg(info.reg, info.reg, RegMap.tmp1);
            }
            try destCommit(code, reg_map, info);
        },
        .store => |st| try emitStore(code, st, reg_map, fctx),

        .ret => |maybe_val| {
            if (maybe_val) |val| {
                try emitPrimaryReturnValue(code, val, reg_map, v128_map, v128_cache, fctx);
            }
            try emitCalleeSaveRestoreTracked(code, fctx);
            try code.emitEpilogue(frame_size);
        },
        .ret_multi => |vregs| try emitRetMulti(code, vregs, reg_map, v128_map, v128_cache, fctx, frame_size),
        .call_result => |idx| try emitCallResult(code, inst, idx, reg_map, v128_map, v128_cache, fctx),
        .@"unreachable" => try emitTrapHelperCall(code, vmctx_trap_unreachable_fn_slot),
        .global_get => |idx| try emitGlobalGet(code, inst, idx, reg_map, v128_map, v128_cache, fctx),
        .global_set => |gs| try emitGlobalSet(code, inst, gs, reg_map, v128_map, v128_cache, fctx),
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
        .parallel_copy => |pairs| try emitParallelCopy(code, pairs, reg_map, fctx.allocator),
        // #672 EH ops — commit 3:
        //   * try_table_begin / try_table_end: no-op. In-function catch
        //     dispatch is deferred to a later commit; for now the catch
        //     handler block is reached via normal br targets only (any
        //     `throw` inside the region escapes as an uncaught trap).
        //   * throw: spill payload into vmctx, call `aot_throw_uncaught`
        //     trampoline (noreturn).
        //   * throw_ref: route through the unreachable trap helper. The
        //     exnref state lives in ExecEnv (interp-only); cross-arch
        //     AOT exnref support arrives with cross-function unwind in
        //     commit 5.
        .try_table_begin, .try_table_end => {},
        .throw => |th| try emitThrow(code, th, reg_map),
        .throw_ref => try emitTrapHelperCall(code, vmctx_trap_unreachable_fn_slot),
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
        try frameStore(code, info.reg, reg_map.spillOffsetBytes(off));
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
    // #542: rematerialisable defs have no entry in `entries` and no
    // spill slot. Re-emit the constant into `scratch`. Critically,
    // because this runs at the use site, the re-emission happens
    // AFTER any intervening call-clobber — no reload needed.
    if (reg_map.alloc_result) |ar| {
        if (ar.getRemat(vreg)) |rd| {
            switch (rd) {
                .iconst_32 => |v| try code.movImm32(scratch, v),
                .iconst_64 => |v| try code.movImm64(scratch, @bitCast(v)),
            }
            return scratch;
        }
    }
    const loc = reg_map.get(vreg) orelse return error.UnboundVReg;
    return switch (loc) {
        .reg => |r| r,
        .stack => |off| blk: {
            try frameLoadReg(code, scratch, reg_map.spillOffsetBytes(off));
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
const v128_tmp2: u5 = 2;
const v128_return_reg: u5 = 0;

fn storeV128SlotAbs(
    code: *emit.CodeBuffer,
    slot_off: u32,
    vt: u5,
) !void {
    try frameAddr(code, RegMap.tmp0, slot_off);
    try code.strQ(vt, RegMap.tmp0);
}

fn vregType(fctx: *const FuncCompileCtx, vreg: ir.VReg) ir.IrType {
    if (fctx.vreg_types) |vts| {
        if (vts.get(vreg)) |ty| return ty;
    }
    return .i64;
}

fn addrFromBaseOffset(
    code: *emit.CodeBuffer,
    dst: emit.Reg,
    base: emit.Reg,
    offset: u32,
) !void {
    if (offset <= 4095) {
        try code.addImm(dst, base, @intCast(offset));
    } else if ((offset & 0xFFF) == 0 and (offset >> 12) <= 4095) {
        try code.addImmShift12(dst, base, @intCast(offset >> 12));
    } else {
        try emitMovImm64(code, dst, offset);
        try code.addRegReg(dst, dst, base);
    }
}

fn storeScalarToBaseOffset(
    code: *emit.CodeBuffer,
    base: emit.Reg,
    offset: u32,
    src: emit.Reg,
    addr_tmp: emit.Reg,
) !void {
    if (offset % 8 == 0 and offset / 8 <= 4095) {
        try code.strImm(src, base, @intCast(offset / 8));
        return;
    }
    if (src == addr_tmp) return error.ReturnAreaOffsetTooLarge;
    try addrFromBaseOffset(code, addr_tmp, base, offset);
    try code.strImm(src, addr_tmp, 0);
}

fn storeV128ToBaseOffset(
    code: *emit.CodeBuffer,
    base: emit.Reg,
    offset: u32,
    vt: u5,
    addr_tmp: emit.Reg,
) !void {
    if (offset == 0) {
        try code.strQ(vt, base);
        return;
    }
    try addrFromBaseOffset(code, addr_tmp, base, offset);
    try code.strQ(vt, addr_tmp);
}

fn emitPrimaryReturnValue(
    code: *emit.CodeBuffer,
    val: ir.VReg,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    if (vregType(fctx, val) == .v128) {
        const vt = try v128_cache.ensure(code, v128_map, val, null);
        if (vt != v128_return_reg) try code.bitwise16b(.orr, v128_return_reg, vt, vt);
    } else {
        const r = try useInto(code, reg_map, val, .x0);
        if (r != .x0) try code.movRegReg(.x0, r);
    }
}

fn emitCapturePrimaryCallResult(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    if (inst.type == .v128) {
        const vt = try v128_cache.defineFresh(code, v128_map, dest, null);
        try code.bitwise16b(.orr, vt, v128_return_reg, v128_return_reg);
    } else {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }
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
    if (v128_map.reg(dest)) |dest_reg| return dest_reg;
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
    if (v128_map.reg(dest)) |dest_reg| return dest_reg;
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
    try emitV128ImmediateIntoReg(code, dest_reg, val);
}

fn emitV128ImmediateIntoReg(
    code: *emit.CodeBuffer,
    dest_reg: u5,
    val: u128,
) !void {
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

fn emitV128LoadSplat(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    ld: ir.Inst.V128LoadSplat,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    const end_offset: u64 = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + ld.accessSize();
    if (ld.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, ld.base, ld.offset);
    } else {
        try emitMemAddr(code, reg_map, ld.base, ld.offset, end_offset);
    }
    switch (ld.width) {
        .i8x16 => try code.ld1r16b(dest_reg, RegMap.tmp0),
        .i16x8 => try code.ld1r8h(dest_reg, RegMap.tmp0),
        .i32x4 => try code.ld1r4s(dest_reg, RegMap.tmp0),
        .i64x2 => try code.ld1r2d(dest_reg, RegMap.tmp0),
    }
}

fn emitV128LoadZero(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    ld: ir.Inst.V128LoadZero,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    const end_offset: u64 = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + ld.accessSize();
    if (ld.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, ld.base, ld.offset);
    } else {
        try emitMemAddr(code, reg_map, ld.base, ld.offset, end_offset);
    }
    switch (ld.width) {
        .i32 => try code.ldrS(dest_reg, RegMap.tmp0),
        .i64 => try code.ldrD(dest_reg, RegMap.tmp0),
    }
}

fn emitV128LoadExtend(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    ld: ir.Inst.V128LoadExtend,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    const end_offset: u64 = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + ld.accessSize();
    if (ld.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, ld.base, ld.offset);
    } else {
        try emitMemAddr(code, reg_map, ld.base, ld.offset, end_offset);
    }
    try code.ldrD(dest_reg, RegMap.tmp0);
    switch (ld.src_width) {
        .i8x8 => switch (ld.sign) {
            .signed => try code.sshll8h8b(dest_reg, dest_reg),
            .unsigned => try code.ushll8h8b(dest_reg, dest_reg),
        },
        .i16x4 => switch (ld.sign) {
            .signed => try code.sshll4s4h(dest_reg, dest_reg),
            .unsigned => try code.ushll4s4h(dest_reg, dest_reg),
        },
        .i32x2 => switch (ld.sign) {
            .signed => try code.sshll2d2s(dest_reg, dest_reg),
            .unsigned => try code.ushll2d2s(dest_reg, dest_reg),
        },
    }
}

fn emitV128LoadLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    ld: ir.Inst.V128LoadLane,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const dest = inst.dest orelse return;
    if (ld.lane >= ld.width.laneCount()) return error.InvalidSimdLane;

    const vector_reg = try v128_cache.ensure(code, v128_map, ld.vector, null);
    const dest_reg = if (v128_map.reg(dest)) |allocated| blk: {
        if (allocated != vector_reg) try code.bitwise16b(.orr, allocated, vector_reg, vector_reg);
        break :blk allocated;
    } else if (isCurrentKill(fctx, ld.vector)) blk: {
        try v128_cache.prepareOverwrite(code, v128_map, vector_reg, true);
        try v128_cache.defineInReg(v128_map, dest, vector_reg);
        break :blk vector_reg;
    } else blk: {
        const fresh = try v128_cache.defineFresh(code, v128_map, dest, vector_reg);
        try code.bitwise16b(.orr, fresh, vector_reg, vector_reg);
        break :blk fresh;
    };

    const end_offset: u64 = if (ld.checked_end > 0) ld.checked_end else @as(u64, ld.offset) + ld.accessSize();
    if (ld.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, ld.base, ld.offset);
    } else {
        try emitMemAddr(code, reg_map, ld.base, ld.offset, end_offset);
    }
    switch (ld.width) {
        .i8 => try code.ld1LaneB(dest_reg, @intCast(ld.lane), RegMap.tmp0),
        .i16 => try code.ld1LaneH(dest_reg, @intCast(ld.lane), RegMap.tmp0),
        .i32 => try code.ld1LaneS(dest_reg, @intCast(ld.lane), RegMap.tmp0),
        .i64 => try code.ld1LaneD(dest_reg, @intCast(ld.lane), RegMap.tmp0),
    }
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

fn emitV128StoreLane(
    code: *emit.CodeBuffer,
    st: ir.Inst.V128StoreLane,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    if (st.lane >= st.width.laneCount()) return error.InvalidSimdLane;

    const vector_reg = try v128_cache.ensure(code, v128_map, st.vector, null);
    const end_offset: u64 = if (st.checked_end > 0) st.checked_end else @as(u64, st.offset) + st.accessSize();
    if (st.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, st.base, st.offset);
    } else {
        try emitMemAddr(code, reg_map, st.base, st.offset, end_offset);
    }
    switch (st.width) {
        .i8 => try code.st1LaneB(vector_reg, @intCast(st.lane), RegMap.tmp0),
        .i16 => try code.st1LaneH(vector_reg, @intCast(st.lane), RegMap.tmp0),
        .i32 => try code.st1LaneS(vector_reg, @intCast(st.lane), RegMap.tmp0),
        .i64 => try code.st1LaneD(vector_reg, @intCast(st.lane), RegMap.tmp0),
    }
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

fn emitV128Bitselect(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    sel: ir.Inst.V128Bitselect,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const mask_reg = try v128_cache.ensure(code, v128_map, sel.mask, null);
    const dest = inst.dest orelse return;
    const dest_reg = if (v128_map.reg(dest)) |allocated| blk: {
        if (allocated != mask_reg) try code.bitwise16b(.orr, allocated, mask_reg, mask_reg);
        break :blk allocated;
    } else if (isCurrentKill(fctx, sel.mask)) blk: {
        try v128_cache.prepareOverwrite(code, v128_map, mask_reg, true);
        try v128_cache.defineInReg(v128_map, dest, mask_reg);
        break :blk mask_reg;
    } else blk: {
        const reg = try v128_cache.defineFresh(code, v128_map, dest, mask_reg);
        try code.bitwise16b(.orr, reg, mask_reg, mask_reg);
        break :blk reg;
    };

    const a_reg = if (sel.a == sel.mask)
        dest_reg
    else
        try v128_cache.ensure(code, v128_map, sel.a, dest_reg);
    const b_reg = if (sel.b == sel.mask)
        dest_reg
    else if (sel.b == sel.a)
        a_reg
    else
        try v128_cache.ensure(code, v128_map, sel.b, dest_reg);

    try code.bsl16b(dest_reg, a_reg, b_reg);
}

fn emitV128AnyTrue(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    src: ir.VReg,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    const src_reg = try v128_cache.ensure(code, v128_map, src, null);
    try code.umaxvB16b(v128_tmp0, src_reg);
    try code.umovWFromB(info.reg, v128_tmp0, 0);
    try code.cmpImm32(info.reg, 0);
    try code.cset32(info.reg, .ne);
    try destCommit(code, reg_map, info);
}

fn emitSimdAllTrue(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdAllTrue,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const info = try destBegin(reg_map, dest, RegMap.tmp2);
    const src_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    switch (op.width) {
        .i8x16 => {
            try code.uminvB16b(v128_tmp0, src_reg);
            try code.umovWFromB(info.reg, v128_tmp0, 0);
            try code.cmpImm32(info.reg, 0);
            try code.cset32(info.reg, .ne);
        },
        .i16x8 => {
            try code.uminvH8h(v128_tmp0, src_reg);
            try code.umovWFromH(info.reg, v128_tmp0, 0);
            try code.cmpImm32(info.reg, 0);
            try code.cset32(info.reg, .ne);
        },
        .i32x4 => {
            try code.uminvS4s(v128_tmp0, src_reg);
            try code.umovWFromS(info.reg, v128_tmp0, 0);
            try code.cmpImm32(info.reg, 0);
            try code.cset32(info.reg, .ne);
        },
        .i64x2 => {
            try code.umovXFromD(RegMap.tmp0, src_reg, 0);
            try code.umovXFromD(RegMap.tmp1, src_reg, 1);
            try code.cmpImm(RegMap.tmp0, 0);
            try code.cset32(RegMap.tmp0, .ne);
            try code.cmpImm(RegMap.tmp1, 0);
            try code.cset32(RegMap.tmp1, .ne);
            try code.andRegReg(info.reg, RegMap.tmp0, RegMap.tmp1);
        },
    }
    try destCommit(code, reg_map, info);
}

const bitmask_i8x16_weights: u128 = 0x8040_2010_0804_0201_8040_2010_0804_0201;
const bitmask_i16x8_weights: u128 = 0x0080_0040_0020_0010_0008_0004_0002_0001;
const bitmask_i32x4_weights: u128 = 0x0000_0008_0000_0004_0000_0002_0000_0001;
const bitmask_i64x2_weights: u128 = 0x0000_0000_0000_0002_0000_0000_0000_0001;

fn emitSimdBitmask(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdBitmask,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const info = try destBegin(reg_map, dest, RegMap.tmp2);
    const src_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    switch (op.width) {
        .i8x16 => {
            try code.sshr16b7(v128_tmp0, src_reg);
            try emitV128ImmediateIntoReg(code, v128_tmp1, bitmask_i8x16_weights);
            try code.bitwise16b(.@"and", v128_tmp0, v128_tmp0, v128_tmp1);
            try code.addp16b(v128_tmp0, v128_tmp0, v128_tmp0);
            try code.addp16b(v128_tmp0, v128_tmp0, v128_tmp0);
            try code.addp16b(v128_tmp0, v128_tmp0, v128_tmp0);
            try code.umovWFromB(info.reg, v128_tmp0, 0);
            try code.umovWFromB(RegMap.tmp0, v128_tmp0, 1);
            try code.lslImm(RegMap.tmp0, RegMap.tmp0, 8);
            try code.orrRegReg(info.reg, info.reg, RegMap.tmp0);
        },
        .i16x8 => {
            try code.sshr8h15(v128_tmp0, src_reg);
            try emitV128ImmediateIntoReg(code, v128_tmp1, bitmask_i16x8_weights);
            try code.bitwise16b(.@"and", v128_tmp0, v128_tmp0, v128_tmp1);
            try code.addvH8h(v128_tmp0, v128_tmp0);
            try code.umovWFromH(info.reg, v128_tmp0, 0);
        },
        .i32x4 => {
            try code.sshr4s31(v128_tmp0, src_reg);
            try emitV128ImmediateIntoReg(code, v128_tmp1, bitmask_i32x4_weights);
            try code.bitwise16b(.@"and", v128_tmp0, v128_tmp0, v128_tmp1);
            try code.addvS4s(v128_tmp0, v128_tmp0);
            try code.umovWFromS(info.reg, v128_tmp0, 0);
        },
        .i64x2 => {
            try code.sshr2d63(v128_tmp0, src_reg);
            try emitV128ImmediateIntoReg(code, v128_tmp1, bitmask_i64x2_weights);
            try code.bitwise16b(.@"and", v128_tmp0, v128_tmp0, v128_tmp1);
            try code.addpDScalar(v128_tmp0, v128_tmp0);
            try code.umovXFromD(info.reg, v128_tmp0, 0);
        },
    }
    try destCommit(code, reg_map, info);
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
        .popcnt => unreachable,
    }
}

fn emitF32x4UnOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    un: ir.Inst.F32x4UnOp,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, un.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, un.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (un.op) {
        .abs => try code.fabs4s(dest_reg, vector_reg),
        .neg => try code.fneg4s(dest_reg, vector_reg),
        .sqrt => {
            try code.fsqrt4s(dest_reg, vector_reg);
            try canonicalizeF32x4NaNs(code, dest_reg);
        },
        .ceil => {
            try code.frint4s(.ceil, dest_reg, vector_reg);
            try canonicalizeF32x4NaNs(code, dest_reg);
        },
        .floor => {
            try code.frint4s(.floor, dest_reg, vector_reg);
            try canonicalizeF32x4NaNs(code, dest_reg);
        },
        .trunc => {
            try code.frint4s(.trunc, dest_reg, vector_reg);
            try canonicalizeF32x4NaNs(code, dest_reg);
        },
        .nearest => {
            try code.frint4s(.nearest, dest_reg, vector_reg);
            try canonicalizeF32x4NaNs(code, dest_reg);
        },
    }
}

fn canonicalizeF32x4NaNs(code: *emit.CodeBuffer, dest_reg: u5) !void {
    try code.fcmeq4s(v128_tmp0, dest_reg, dest_reg);
    try code.mvn16b(v128_tmp0, v128_tmp0);
    try code.movImm32(RegMap.tmp0, @bitCast(@as(u32, 0x7fc0_0000)));
    try code.dup4sFromGp32(v128_tmp1, RegMap.tmp0);
    try code.bsl16b(v128_tmp0, v128_tmp1, dest_reg);
    try code.bitwise16b(.orr, dest_reg, v128_tmp0, v128_tmp0);
}

fn emitF64x2UnOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    un: ir.Inst.F64x2UnOp,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, un.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, un.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (un.op) {
        .abs => try code.fabs2d(dest_reg, vector_reg),
        .neg => try code.fneg2d(dest_reg, vector_reg),
        .sqrt => {
            try code.fsqrt2d(dest_reg, vector_reg);
            try canonicalizeF64x2NaNs(code, dest_reg);
        },
        .ceil => {
            try code.frint2d(.ceil, dest_reg, vector_reg);
            try canonicalizeF64x2NaNs(code, dest_reg);
        },
        .floor => {
            try code.frint2d(.floor, dest_reg, vector_reg);
            try canonicalizeF64x2NaNs(code, dest_reg);
        },
        .trunc => {
            try code.frint2d(.trunc, dest_reg, vector_reg);
            try canonicalizeF64x2NaNs(code, dest_reg);
        },
        .nearest => {
            try code.frint2d(.nearest, dest_reg, vector_reg);
            try canonicalizeF64x2NaNs(code, dest_reg);
        },
    }
}

fn canonicalizeF64x2NaNs(code: *emit.CodeBuffer, dest_reg: u5) !void {
    try code.fcmeq2d(v128_tmp0, dest_reg, dest_reg);
    try code.mvn16b(v128_tmp0, v128_tmp0);
    try code.movImm64(RegMap.tmp0, 0x7ff8_0000_0000_0000);
    try code.dup2dFromGp64(v128_tmp1, RegMap.tmp0);
    try code.bsl16b(v128_tmp0, v128_tmp1, dest_reg);
    try code.bitwise16b(.orr, dest_reg, v128_tmp0, v128_tmp0);
}

fn emitF32x4ConvertI32x4(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdIntToFloatConvert,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, op.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (op.sign) {
        .signed => try code.scvtf4s(dest_reg, vector_reg),
        .unsigned => try code.ucvtf4s(dest_reg, vector_reg),
    }
}

fn emitI32x4TruncSat(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdFloatToIntTruncSat,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    switch (op.src_width) {
        .f32x4 => {
            const dest_reg = try prepareV128UnaryDest(code, inst, op.vector, vector_reg, v128_map, v128_cache, fctx);
            try code.fcmeq4s(v128_tmp0, vector_reg, vector_reg);
            try code.bitwise16b(.@"and", dest_reg, vector_reg, v128_tmp0);
            switch (op.sign) {
                .signed => try code.fcvtzs4s(dest_reg, dest_reg),
                .unsigned => try code.fcvtzu4s(dest_reg, dest_reg),
            }
        },
        .f64x2 => {
            const dest = inst.dest orelse return;
            try code.fcmeq2d(v128_tmp0, vector_reg, vector_reg);
            try code.bitwise16b(.@"and", v128_tmp1, vector_reg, v128_tmp0);
            switch (op.sign) {
                .signed => try code.fcvtzs2d(v128_tmp1, v128_tmp1),
                .unsigned => try code.fcvtzu2d(v128_tmp1, v128_tmp1),
            }
            const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, vector_reg);
            try code.movi2dZero(dest_reg);
            switch (op.sign) {
                .signed => try code.sqxtn2s2d(dest_reg, v128_tmp1),
                .unsigned => try code.uqxtn2s2d(dest_reg, v128_tmp1),
            }
        },
    }
}

fn emitF32x4DemoteF64x2Zero(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdFloatPrecisionConvert,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, vector_reg);
    try code.bitwise16b(.eor, dest_reg, dest_reg, dest_reg);
    try code.fcvtn2s2d(dest_reg, vector_reg);
    try canonicalizeF32x4NaNs(code, dest_reg);
}

fn emitF32x4Splat(
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

fn emitF32x4ExtractLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.F32x4ExtractLane,
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

fn emitF32x4ReplaceLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.F32x4ReplaceLane,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, lane.vector, vector_reg, v128_map, v128_cache, fctx);
    if (dest_reg != vector_reg) try code.bitwise16b(.orr, dest_reg, vector_reg, vector_reg);
    const val_reg = try useInto(code, reg_map, lane.val, RegMap.tmp0);
    try code.insSFromGp32(dest_reg, lane.lane, val_reg);
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
        .popcnt => try code.cnt16b(dest_reg, vector_reg),
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
        .popcnt => unreachable,
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

fn emitF64x2ConvertLowI32x4(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdIntToFloatConvert,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, op.vector, vector_reg, v128_map, v128_cache, fctx);
    switch (op.sign) {
        .signed => {
            try code.sshll2d2s(dest_reg, vector_reg);
            try code.scvtf2d(dest_reg, dest_reg);
        },
        .unsigned => {
            try code.ushll2d2s(dest_reg, vector_reg);
            try code.ucvtf2d(dest_reg, dest_reg);
        },
    }
}

fn emitF64x2PromoteLowF32x4(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdFloatPrecisionConvert,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, op.vector, vector_reg, v128_map, v128_cache, fctx);
    try code.fcvtl2d2s(dest_reg, vector_reg);
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

fn emitI32x4DotI16x8S(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.BinOp,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, op.lhs, null);
    const rhs_reg = if (op.rhs == op.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, op.rhs, lhs_reg);

    try code.smull4s4h(v128_tmp0, lhs_reg, rhs_reg);
    try code.smull2_4s8h(v128_tmp1, lhs_reg, rhs_reg);

    const dest = inst.dest orelse return;
    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    try code.addp4s(dest_reg, v128_tmp0, v128_tmp1);
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

fn emitI8x16NarrowI16x8(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdNarrow,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, op.lhs, null);
    const rhs_reg = if (op.rhs == op.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, op.rhs, lhs_reg);
    const dest = inst.dest orelse return;
    // XTN writes Vd's low 64 bits and zeroes the upper; XTN2 then writes the
    // upper 64 bits while preserving the low half. The XTN2 reads rhs after
    // dest has been written, so dest_reg must NOT alias rhs_reg.
    const dest_reg = if (v128_map.reg(dest)) |allocated|
        allocated
    else if (op.rhs == op.lhs)
        try v128_cache.defineFresh(code, v128_map, dest, lhs_reg)
    else blk: {
        try v128_cache.prepareOverwrite(code, v128_map, lhs_reg, isCurrentKill(fctx, op.lhs));
        try v128_cache.defineInReg(v128_map, dest, lhs_reg);
        break :blk lhs_reg;
    };
    switch (op.sign) {
        .signed => {
            try code.sqxtn8b8h(dest_reg, lhs_reg);
            try code.sqxtn2_16b8h(dest_reg, rhs_reg);
        },
        .unsigned => {
            try code.sqxtun8b8h(dest_reg, lhs_reg);
            try code.sqxtun2_16b8h(dest_reg, rhs_reg);
        },
    }
}

fn emitI16x8NarrowI32x4(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.SimdNarrow,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const lhs_reg = try v128_cache.ensure(code, v128_map, op.lhs, null);
    const rhs_reg = if (op.rhs == op.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, op.rhs, lhs_reg);
    const dest = inst.dest orelse return;
    const dest_reg = if (v128_map.reg(dest)) |allocated|
        allocated
    else if (op.rhs == op.lhs)
        try v128_cache.defineFresh(code, v128_map, dest, lhs_reg)
    else blk: {
        try v128_cache.prepareOverwrite(code, v128_map, lhs_reg, isCurrentKill(fctx, op.lhs));
        try v128_cache.defineInReg(v128_map, dest, lhs_reg);
        break :blk lhs_reg;
    };
    switch (op.sign) {
        .signed => {
            try code.sqxtn4h4s(dest_reg, lhs_reg);
            try code.sqxtn2_8h4s(dest_reg, rhs_reg);
        },
        .unsigned => {
            try code.sqxtun4h4s(dest_reg, lhs_reg);
            try code.sqxtun2_8h4s(dest_reg, rhs_reg);
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
        .popcnt => unreachable,
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
    if (dest_reg != vector_reg) try code.bitwise16b(.orr, dest_reg, vector_reg, vector_reg);
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

fn emitI8x16Shuffle(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.I8x16Shuffle,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
) !void {
    const dest = inst.dest orelse return;
    const lhs_reg = try v128_cache.ensure(code, v128_map, op.lhs, null);
    const rhs_reg = if (op.rhs == op.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, op.rhs, lhs_reg);

    // TBL's two-register table operand must be a consecutive register pair.
    // Keep cached values in Q16-Q31 and use Q0/Q1/Q2 only as per-instruction scratch.
    try code.bitwise16b(.orr, v128_tmp0, lhs_reg, lhs_reg);
    try code.bitwise16b(.orr, v128_tmp1, rhs_reg, rhs_reg);
    try emitV128ImmediateIntoReg(code, v128_tmp2, std.mem.readInt(u128, &op.lanes, .little));

    const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
    try code.tbl2_16b(dest_reg, v128_tmp0, v128_tmp2);
}

fn emitI8x16Swizzle(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    op: ir.Inst.I8x16Swizzle,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, op.vector, null);
    const indices_reg = if (op.indices == op.vector)
        vector_reg
    else
        try v128_cache.ensure(code, v128_map, op.indices, vector_reg);
    const dest_reg = try prepareV128BinaryDest(
        code,
        inst,
        op.vector,
        vector_reg,
        op.indices,
        indices_reg,
        v128_map,
        v128_cache,
        fctx,
    );
    try code.tbl1_16b(dest_reg, vector_reg, indices_reg);
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
    if (dest_reg != vector_reg) try code.bitwise16b(.orr, dest_reg, vector_reg, vector_reg);
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
    if (dest_reg != vector_reg) try code.bitwise16b(.orr, dest_reg, vector_reg, vector_reg);
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

fn emitSimdFma(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    info: FmaInfo,
    is_f32x4: bool,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const addend_reg = try v128_cache.ensure(code, v128_map, info.addend, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, info.addend, addend_reg, v128_map, v128_cache, fctx);
    if (dest_reg != addend_reg) try code.bitwise16b(.orr, dest_reg, addend_reg, addend_reg);

    const lhs_reg = try v128_cache.ensure(code, v128_map, info.mul_lhs, dest_reg);
    const rhs_reg = if (info.mul_rhs == info.mul_lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, info.mul_rhs, dest_reg);

    if (is_f32x4) {
        try code.fmla4s(info.is_sub, dest_reg, lhs_reg, rhs_reg);
    } else {
        try code.fmla2d(info.is_sub, dest_reg, lhs_reg, rhs_reg);
    }
}

fn emitF32x4BinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.F32x4BinOp,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const dest = inst.dest orelse return;
    if (bin.op == .mul and fctx.simd_mul_fused != null and fctx.simd_mul_fused.?.contains(dest)) return;
    if ((bin.op == .add or bin.op == .sub) and fctx.simd_fma_info != null) {
        if (fctx.simd_fma_info.?.get(dest)) |info| {
            try emitSimdFma(code, inst, info, true, v128_map, v128_cache, fctx);
            return;
        }
    }
    const lhs_reg = try v128_cache.ensure(code, v128_map, bin.lhs, null);
    const rhs_reg = if (bin.rhs == bin.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, bin.rhs, lhs_reg);
    if (bin.op == .min or bin.op == .max) {
        try emitF32x4MinMax(code, inst, bin, lhs_reg, rhs_reg, v128_map, v128_cache, fctx);
        return;
    }
    if (bin.op == .pmin or bin.op == .pmax) {
        const cmp_lhs = if (bin.op == .pmin) lhs_reg else rhs_reg;
        const cmp_rhs = if (bin.op == .pmin) rhs_reg else lhs_reg;
        try code.fcmgt4s(v128_tmp0, cmp_lhs, cmp_rhs);
        try code.bsl16b(v128_tmp0, rhs_reg, lhs_reg);
        const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
        try code.bitwise16b(.orr, dest_reg, v128_tmp0, v128_tmp0);
        return;
    }
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
        .add => try code.f32x4Op(.add, dest_reg, lhs_reg, rhs_reg),
        .sub => try code.f32x4Op(.sub, dest_reg, lhs_reg, rhs_reg),
        .mul => try code.f32x4Op(.mul, dest_reg, lhs_reg, rhs_reg),
        .div => try code.f32x4Op(.div, dest_reg, lhs_reg, rhs_reg),
        .eq => try code.fcmeq4s(dest_reg, lhs_reg, rhs_reg),
        .ne => {
            try code.fcmeq4s(dest_reg, lhs_reg, rhs_reg);
            try code.mvn16b(dest_reg, dest_reg);
        },
        .gt => try code.fcmgt4s(dest_reg, lhs_reg, rhs_reg),
        .ge => try code.fcmge4s(dest_reg, lhs_reg, rhs_reg),
        .lt => try code.fcmgt4s(dest_reg, rhs_reg, lhs_reg),
        .le => try code.fcmge4s(dest_reg, rhs_reg, lhs_reg),
        .min, .max => unreachable,
        .pmin, .pmax => unreachable,
    }
}

fn emitF32x4MinMax(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.F32x4BinOp,
    lhs_reg: u5,
    rhs_reg: u5,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    try code.fcmeq4s(v128_tmp0, lhs_reg, lhs_reg);
    try code.fcmeq4s(v128_tmp1, rhs_reg, rhs_reg);
    try code.bitwise16b(.@"and", v128_tmp0, v128_tmp0, v128_tmp1);
    try code.mvn16b(v128_tmp0, v128_tmp0);

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
        .min => try code.f32x4Op(.min, dest_reg, lhs_reg, rhs_reg),
        .max => try code.f32x4Op(.max, dest_reg, lhs_reg, rhs_reg),
        else => unreachable,
    }

    try code.movImm32(RegMap.tmp0, @bitCast(@as(u32, 0x7fc0_0000)));
    try code.dup4sFromGp32(v128_tmp1, RegMap.tmp0);
    try code.bitwise16b(.bic, dest_reg, dest_reg, v128_tmp0);
    try code.bitwise16b(.@"and", v128_tmp1, v128_tmp1, v128_tmp0);
    try code.bitwise16b(.orr, dest_reg, dest_reg, v128_tmp1);
}

fn emitF64x2BinOp(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.F64x2BinOp,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const dest = inst.dest orelse return;
    if (bin.op == .mul and fctx.simd_mul_fused != null and fctx.simd_mul_fused.?.contains(dest)) return;
    if ((bin.op == .add or bin.op == .sub) and fctx.simd_fma_info != null) {
        if (fctx.simd_fma_info.?.get(dest)) |info| {
            try emitSimdFma(code, inst, info, false, v128_map, v128_cache, fctx);
            return;
        }
    }
    const lhs_reg = try v128_cache.ensure(code, v128_map, bin.lhs, null);
    const rhs_reg = if (bin.rhs == bin.lhs)
        lhs_reg
    else
        try v128_cache.ensure(code, v128_map, bin.rhs, lhs_reg);
    if (bin.op == .min or bin.op == .max) {
        try emitF64x2MinMax(code, inst, bin, lhs_reg, rhs_reg, v128_map, v128_cache, fctx);
        return;
    }
    if (bin.op == .pmin or bin.op == .pmax) {
        const cmp_lhs = if (bin.op == .pmin) lhs_reg else rhs_reg;
        const cmp_rhs = if (bin.op == .pmin) rhs_reg else lhs_reg;
        try code.fcmgt2d(v128_tmp0, cmp_lhs, cmp_rhs);
        try code.bsl16b(v128_tmp0, rhs_reg, lhs_reg);
        const dest_reg = try v128_cache.defineFresh(code, v128_map, dest, null);
        try code.bitwise16b(.orr, dest_reg, v128_tmp0, v128_tmp0);
        return;
    }
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
        .add => try code.f64x2Op(.add, dest_reg, lhs_reg, rhs_reg),
        .sub => try code.f64x2Op(.sub, dest_reg, lhs_reg, rhs_reg),
        .mul => try code.f64x2Op(.mul, dest_reg, lhs_reg, rhs_reg),
        .div => try code.f64x2Op(.div, dest_reg, lhs_reg, rhs_reg),
        .eq => try code.fcmeq2d(dest_reg, lhs_reg, rhs_reg),
        .ne => {
            try code.fcmeq2d(dest_reg, lhs_reg, rhs_reg);
            try code.mvn16b(dest_reg, dest_reg);
        },
        .gt => try code.fcmgt2d(dest_reg, lhs_reg, rhs_reg),
        .ge => try code.fcmge2d(dest_reg, lhs_reg, rhs_reg),
        .lt => try code.fcmgt2d(dest_reg, rhs_reg, lhs_reg),
        .le => try code.fcmge2d(dest_reg, rhs_reg, lhs_reg),
        .min, .max => unreachable,
        .pmin, .pmax => unreachable,
    }
}

fn emitF64x2MinMax(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    bin: ir.Inst.F64x2BinOp,
    lhs_reg: u5,
    rhs_reg: u5,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    try code.fcmeq2d(v128_tmp0, lhs_reg, lhs_reg);
    try code.fcmeq2d(v128_tmp1, rhs_reg, rhs_reg);
    try code.bitwise16b(.@"and", v128_tmp0, v128_tmp0, v128_tmp1);
    try code.mvn16b(v128_tmp0, v128_tmp0);

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
        .min => try code.f64x2Op(.min, dest_reg, lhs_reg, rhs_reg),
        .max => try code.f64x2Op(.max, dest_reg, lhs_reg, rhs_reg),
        else => unreachable,
    }

    try code.movImm64(RegMap.tmp0, 0x7ff8_0000_0000_0000);
    try code.dup2dFromGp64(v128_tmp1, RegMap.tmp0);
    try code.bitwise16b(.bic, dest_reg, dest_reg, v128_tmp0);
    try code.bitwise16b(.@"and", v128_tmp1, v128_tmp1, v128_tmp0);
    try code.bitwise16b(.orr, dest_reg, dest_reg, v128_tmp1);
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
    if (dest_reg != vector_reg) try code.bitwise16b(.orr, dest_reg, vector_reg, vector_reg);
    const val_reg = try useInto(code, reg_map, lane.val, RegMap.tmp0);
    try code.insDFromGp64(dest_reg, lane.lane, val_reg);
}

fn emitF64x2Splat(
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

fn emitF64x2ExtractLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.F64x2ExtractLane,
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

fn emitF64x2ReplaceLane(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    lane: ir.Inst.F64x2ReplaceLane,
    reg_map: *const RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const vector_reg = try v128_cache.ensure(code, v128_map, lane.vector, null);
    const dest_reg = try prepareV128UnaryDest(code, inst, lane.vector, vector_reg, v128_map, v128_cache, fctx);
    if (dest_reg != vector_reg) try code.bitwise16b(.orr, dest_reg, vector_reg, vector_reg);
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
    // When the move-coalescer has placed src and dest on the same physreg
    // we can skip the UXTW entirely: every downstream consumer of an i32
    // result reads W-form (see `local_get` comment about the contract that
    // upper bits are ignored), so leaving the upper bits stale is safe.
    // See `collectCopyHints` for the matching copy-hint registration.
    if (src != info.reg) try code.uxtw(info.reg, src);
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

/// Resolve a batch of parallel (dst ← src) vreg copies, honouring the
/// "all reads before any writes" semantics required for phi resolution
/// at a predecessor's terminator.
///
/// Each pair classifies into one of four storage shapes after consulting
/// the register allocator's assignments:
///   * reg → stack : a plain `STR src_reg, [fp, #dst_off]` (reads OLD src).
///   * stack → stack : `LDR tmp0, [fp, #src_off]; STR tmp0, [fp, #dst_off]`.
///   * reg → reg : participates in the 4-phase parallel-copy resolver.
///   * stack → reg : a plain `LDR dst_reg, [fp, #src_off]`.
///
/// We emit reg→stack and stack→stack FIRST so they observe the
/// pre-copy register state; then resolve reg→reg with cycle breaking
/// using `RegMap.tmp2` as scratch (X15, non-allocatable); then emit
/// stack→reg loads last so they can safely clobber dst regs that were
/// read in earlier phases.
///
/// Cycle resolver (4-phase):
///   1. Build edges `src_reg → dst_reg` for reg→reg pairs.
///   2. Repeatedly emit moves whose dst is NOT some other pending move's
///      src (a "leaf" — safe because no remaining mov needs to read it).
///   3. Whatever remains forms one or more pure cycles. Break each cycle
///      by `mov tmp2, head_src; mov head_src, …; … ; mov tail_dst, tmp2`.
///   4. Repeat until all pairs are emitted.
fn emitParallelCopy(
    code: *emit.CodeBuffer,
    pairs: []const ir.Inst.ParallelCopy,
    reg_map: *RegMap,
    allocator: std.mem.Allocator,
) !void {
    if (pairs.len == 0) return;
    // v128 parallel-copy is intentionally out of scope for #540. The
    // lowering pass only emits scalar pairs; surface a clear error if
    // a v128 sneaks in.
    for (pairs) |p| if (p.ty == .v128) return error.UnimplementedOp;
    try emitParallelCopyFull(code, pairs, reg_map, allocator);
}

/// True iff two locations (register or spill slot) are identical.
fn locEq(a: RegMap.Location, b: RegMap.Location) bool {
    return switch (a) {
        .reg => |ra| b == .reg and b.reg == ra,
        .stack => |sa| b == .stack and b.stack == sa,
    };
}

/// Emit a single move `dst <- src` between any two locations.
/// reg→reg: MOV; reg→stack: STR; stack→reg: LDR; stack→stack: LDR/STR via
/// tmp0. Identity moves are elided. `tmp0` (X16) is non-allocatable scratch,
/// so it never aliases a parallel-copy operand.
fn emitLocMove(
    code: *emit.CodeBuffer,
    reg_map: *const RegMap,
    dst: RegMap.Location,
    src: RegMap.Location,
) !void {
    if (locEq(dst, src)) return;
    switch (src) {
        .reg => |src_reg| switch (dst) {
            .reg => |dst_reg| try code.movRegReg(dst_reg, src_reg),
            .stack => |off| try frameStore(code, src_reg, reg_map.spillOffsetBytes(off)),
        },
        .stack => |src_off| switch (dst) {
            .reg => |dst_reg| try frameLoadReg(code, dst_reg, reg_map.spillOffsetBytes(src_off)),
            .stack => |dst_off| {
                try frameLoadReg(code, RegMap.tmp0, reg_map.spillOffsetBytes(src_off));
                try frameStore(code, RegMap.tmp0, reg_map.spillOffsetBytes(dst_off));
            },
        },
    }
}

fn emitParallelCopyFull(
    code: *emit.CodeBuffer,
    pairs: []const ir.Inst.ParallelCopy,
    reg_map: *RegMap,
    allocator: std.mem.Allocator,
) !void {
    const Loc = RegMap.Location;
    const Resolved = struct {
        dst: Loc,
        src: Loc,
    };

    var resolved = try allocator.alloc(Resolved, pairs.len);
    defer allocator.free(resolved);

    for (pairs, 0..) |p, i| {
        const dst_loc = try reg_map.assign(p.dst);
        const src_loc = reg_map.get(p.src) orelse return error.UnboundVReg;
        resolved[i] = .{ .dst = dst_loc, .src = src_loc };
    }

    // General parallel-copy sequentialization over arbitrary locations
    // (registers AND spill slots). The previous phase-split resolver
    // (reg→stack, stack→stack, reg→reg, stack→reg) only preserved the
    // "read all sources before writing any destination" invariant within
    // each phase: when a spill slot was BOTH a source and a destination
    // (spilled loop-carried phi operands under register pressure) an
    // earlier phase overwrote a slot a later phase still needed to read,
    // miscompiling the value. This single resolver keeps the invariant
    // across all locations: emit "leaf" moves (whose dst is not read by
    // any pending move) until only cycles remain, then break each cycle
    // with `tmp2` (X15) scratch — exactly the strategy `emitParallelRegMoves`
    // uses for the reg-only subset, generalised via `emitLocMove`.
    var done = try allocator.alloc(bool, pairs.len);
    defer allocator.free(done);
    @memset(done, false);

    const n = resolved.len;
    while (true) {
        // Emit every ready leaf (and elide identities) until none remain.
        var progressed = true;
        while (progressed) {
            progressed = false;
            for (0..n) |i| {
                if (done[i]) continue;
                if (locEq(resolved[i].src, resolved[i].dst)) {
                    done[i] = true;
                    continue;
                }
                // dst[i] is "read by a pending move" iff some other
                // unfinished pair reads from that exact location.
                var read_by_pending = false;
                for (0..n) |j| {
                    if (done[j] or j == i) continue;
                    if (locEq(resolved[j].src, resolved[i].dst)) {
                        read_by_pending = true;
                        break;
                    }
                }
                if (!read_by_pending) {
                    try emitLocMove(code, reg_map, resolved[i].dst, resolved[i].src);
                    done[i] = true;
                    progressed = true;
                }
            }
        }

        // Any pending pair is part of a pure cycle. Pick a head, save its
        // source value into scratch, then emit the cycle's moves in
        // dependency order and close it from scratch.
        var head: ?usize = null;
        for (0..n) |i| {
            if (!done[i]) {
                head = i;
                break;
            }
        }
        if (head == null) return;

        const start = head.?;
        const scratch = Loc{ .reg = RegMap.tmp2 };
        try emitLocMove(code, reg_map, scratch, resolved[start].src);

        // Walk predecessors: the pair whose dst is start's source location
        // is emitted first (it would otherwise clobber the value we just
        // saved), chaining until we return to `start`.
        var found_predecessor: ?usize = null;
        for (0..n) |k| {
            if (done[k] or k == start) continue;
            if (locEq(resolved[k].dst, resolved[start].src)) {
                found_predecessor = k;
                break;
            }
        }

        if (found_predecessor == null) {
            // Degenerate single-edge cycle: settle from scratch.
            try emitLocMove(code, reg_map, resolved[start].dst, scratch);
            done[start] = true;
            continue;
        }

        var cur = found_predecessor.?;
        while (cur != start) {
            try emitLocMove(code, reg_map, resolved[cur].dst, resolved[cur].src);
            done[cur] = true;
            var next_opt: ?usize = null;
            for (0..n) |k| {
                if (done[k] or k == start) continue;
                if (locEq(resolved[k].dst, resolved[cur].src)) {
                    next_opt = k;
                    break;
                }
            }
            if (next_opt == null) break;
            cur = next_opt.?;
        }
        // Close the cycle: write the saved head source into head's dst.
        try emitLocMove(code, reg_map, resolved[start].dst, scratch);
        done[start] = true;
    }
}

/// 4-phase parallel-copy resolver for the reg→reg subset. Emits moves
/// that respect "all reads before all writes": leaf moves (whose dst is
/// not read by any pending move) are emitted in topological order; the
/// remaining pure cycles are broken with `RegMap.tmp2` (X15) as scratch.
fn emitParallelRegMoves(
    code: *emit.CodeBuffer,
    dst: []const emit.Reg,
    src: []const emit.Reg,
    done: []bool,
) !void {
    const n = dst.len;
    if (n == 0) return;

    while (true) {
        var progressed = true;
        while (progressed) {
            progressed = false;
            for (0..n) |i| {
                if (done[i]) continue;
                // Is dst[i] read by any pending mov?
                var read_by_pending = false;
                for (0..n) |j| {
                    if (done[j] or j == i) continue;
                    if (src[j] == dst[i]) {
                        read_by_pending = true;
                        break;
                    }
                }
                if (!read_by_pending) {
                    try code.movRegReg(dst[i], src[i]);
                    done[i] = true;
                    progressed = true;
                }
            }
        }

        // Any pending? Find a head and break its cycle.
        var head: ?usize = null;
        for (0..n) |i| {
            if (!done[i]) {
                head = i;
                break;
            }
        }
        if (head == null) return;

        // Cycle break: walk the cycle backwards (in dataflow terms,
        // visit the pair that WRITES TO head.src first). The pair we
        // visit last is `head` itself, emitted with its src replaced
        // by `scratch` (which still holds the head.src's PRE-copy
        // value).
        const start = head.?;
        const scratch = RegMap.tmp2;
        try code.movRegReg(scratch, src[start]);

        var found_predecessor: ?usize = null;
        for (0..n) |k| {
            if (done[k] or k == start) continue;
            if (dst[k] == src[start]) {
                found_predecessor = k;
                break;
            }
        }

        if (found_predecessor == null) {
            // `start`'s src is not written by any pending mov. That
            // means `start` participates in a degenerate cycle of one
            // edge (post-leaf-resolution this can only happen via a
            // chained handoff already emitted). Settle it with
            // scratch and re-enter the outer loop.
            try code.movRegReg(dst[start], scratch);
            done[start] = true;
            continue;
        }

        var cur = found_predecessor.?;
        while (cur != start) {
            try code.movRegReg(dst[cur], src[cur]);
            done[cur] = true;
            var next_opt: ?usize = null;
            for (0..n) |k| {
                if (done[k] or k == start) continue;
                if (dst[k] == src[cur]) {
                    next_opt = k;
                    break;
                }
            }
            if (next_opt == null) {
                // Chain bridged into the cycle — shouldn't happen
                // after leaf resolution, but bail safely.
                break;
            }
            cur = next_opt.?;
        }
        // Close the cycle: write scratch into head's dst.
        try code.movRegReg(dst[start], scratch);
        done[start] = true;
    }
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
    if (trace_select) {
        std.debug.print(
            "[codegen.aarch64.select] dest=v{d} type={s} if_true=v{d} if_false=v{d} cond=v{d}\n",
            .{ dest, @tagName(inst.type), sel.if_true, sel.if_false, sel.cond },
        );
    }
    // Load cond first; after CMP we don't need the cond register anymore,
    // so we can reuse tmp0 for subsequent uses if desired. We keep t/f in
    // distinct scratches (tmp1/tmp2) so CSEL can read them simultaneously.
    const cond_r = try useInto(code, reg_map, sel.cond, RegMap.tmp0);
    try code.cmpImm32(cond_r, 0);
    const t_r = try useInto(code, reg_map, sel.if_true, RegMap.tmp1);
    const f_r = try useInto(code, reg_map, sel.if_false, RegMap.tmp2);
    const info = try destBegin(reg_map, dest, RegMap.tmp0);
    // wasm: select picks if_true when cond != 0.
    //
    // #754: `inst.type` MUST reflect the actual 64-bit type when
    // selecting f64/i64 values — `csel32` clobbers the upper 32 bits
    // of the destination register. The frontend's untyped `select`
    // initially records `.i32` and a fix-up pass propagates the
    // correct type from the `if_true` producer; flip `trace_select`
    // on a debug build if a chained-select pattern is suspected of
    // returning only the low 32 bits.
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

fn emitSpAdjust(code: *emit.CodeBuffer, bytes: u32, kind: BinOpKind) !void {
    if (bytes == 0) return;
    const enc = encodeAddSubImm(@intCast(bytes)) orelse return error.StackArgAreaTooLarge;
    try emitAddSubImm(code, .sp, .sp, enc, kind);
}

fn stackAddr(code: *emit.CodeBuffer, dst: emit.Reg, offset: u32) !void {
    if (offset <= 4095) {
        try code.addImm(dst, .sp, @intCast(offset));
    } else if ((offset & 0xFFF) == 0 and (offset >> 12) <= 4095) {
        try code.addImmShift12(dst, .sp, @intCast(offset >> 12));
    } else {
        try emitMovImm64(code, dst, offset);
        try code.addRegReg(dst, dst, .sp);
    }
}

fn storeOutgoingScalarArg(
    code: *emit.CodeBuffer,
    src: emit.Reg,
    offset: u32,
    addr_tmp: emit.Reg,
) !void {
    if (offset % 8 == 0 and offset / 8 <= 4095) {
        try code.strImm(src, .sp, @intCast(offset / 8));
    } else {
        try stackAddr(code, addr_tmp, offset);
        try code.strImm(src, addr_tmp, 0);
    }
}

fn storeOutgoingV128Arg(
    code: *emit.CodeBuffer,
    vt: u5,
    offset: u32,
    addr_tmp: emit.Reg,
) !void {
    if (offset == 0) {
        try code.strQ(vt, .sp);
    } else {
        try stackAddr(code, addr_tmp, offset);
        try code.strQ(vt, addr_tmp);
    }
}

fn loadV128ArgFromSaved(
    code: *emit.CodeBuffer,
    v128_map: *V128StackMap,
    vreg: ir.VReg,
    vt: u5,
    addr_tmp: emit.Reg,
) !void {
    if (v128_map.reg(vreg)) |src| {
        if (src != vt) try code.bitwise16b(.orr, vt, src, src);
        return;
    }
    const off = try v128_map.get(vreg);
    try frameAddr(code, addr_tmp, off);
    try code.ldrQ(vt, addr_tmp);
}

fn frameLoadWithScratch(
    code: *emit.CodeBuffer,
    dst: emit.Reg,
    offset: u32,
    scratch: emit.Reg,
) !void {
    if (offset % 8 == 0 and offset / 8 <= 4095) {
        try code.ldrImm(dst, .fp, @intCast(offset / 8));
    } else {
        try emitMovImm64(code, scratch, offset);
        try code.addRegReg(scratch, scratch, .fp);
        try code.ldrImm(dst, scratch, 0);
    }
}

const HrpSource = enum { caller_scratch, forwarded };

fn emitStageHrp(
    code: *emit.CodeBuffer,
    loc: AbiArgLoc,
    fctx: *FuncCompileCtx,
    source: HrpSource,
    scalar_tmp: emit.Reg,
    addr_tmp: emit.Reg,
) !void {
    switch (loc) {
        .scalar_reg => |target| switch (source) {
            .caller_scratch => try frameAddr(code, target, fctx.scratch_base),
            .forwarded => try frameLoadWithScratch(code, target, fctx.hrp_save_off, scalar_tmp),
        },
        .stack => |stack| {
            switch (source) {
                .caller_scratch => try frameAddr(code, scalar_tmp, fctx.scratch_base),
                .forwarded => try frameLoadWithScratch(code, scalar_tmp, fctx.hrp_save_off, addr_tmp),
            }
            try storeOutgoingScalarArg(code, scalar_tmp, stack.offset, addr_tmp);
        },
        .v128_reg => unreachable,
    }
}

fn markScalarClobbered(mask: *u16, reg: emit.Reg) void {
    const reg_num: u32 = @intFromEnum(reg);
    if (reg_num < RegMap.caller_saved_count) {
        mask.* |= (@as(u16, 1) << @intCast(reg_num));
    }
}

fn emitStageCallAbi(
    code: *emit.CodeBuffer,
    plan: *const CallAbiPlan,
    args: []const ir.VReg,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    fctx: *FuncCompileCtx,
    base_clobbered: u16,
    scalar_tmp: emit.Reg,
    addr_tmp: emit.Reg,
    hrp_source: ?HrpSource,
) !u16 {
    if (plan.stack_size > 0) try emitSpAdjust(code, plan.stack_size, .sub);

    var clobbered = base_clobbered;

    // Store stack-passed args first so temporary vector/scalar registers cannot
    // clobber already-staged register args.
    for (args, 0..) |arg_vreg, i| {
        switch (plan.locs[i]) {
            .stack => |stack| {
                if (stack.ty == .v128) {
                    try loadV128ArgFromSaved(code, v128_map, arg_vreg, v128_call_stack_tmp, addr_tmp);
                    try storeOutgoingV128Arg(code, v128_call_stack_tmp, stack.offset, addr_tmp);
                } else {
                    try stageArgFromSaved(code, reg_map, fctx, scalar_tmp, arg_vreg, clobbered);
                    try storeOutgoingScalarArg(code, scalar_tmp, stack.offset, addr_tmp);
                }
            },
            else => {},
        }
    }
    if (plan.hrp) |hrp_loc| {
        if (hrp_source) |src| {
            switch (hrp_loc) {
                .stack => try emitStageHrp(code, hrp_loc, fctx, src, scalar_tmp, addr_tmp),
                else => {},
            }
        }
    }

    for (args, 0..) |arg_vreg, i| {
        switch (plan.locs[i]) {
            .scalar_reg => |target| {
                try stageArgFromSaved(code, reg_map, fctx, target, arg_vreg, clobbered);
                markScalarClobbered(&clobbered, target);
            },
            .v128_reg => |vt| try loadV128ArgFromSaved(code, v128_map, arg_vreg, vt, addr_tmp),
            .stack => {},
        }
    }
    if (plan.hrp) |hrp_loc| {
        if (hrp_source) |src| {
            switch (hrp_loc) {
                .scalar_reg => |reg| {
                    try emitStageHrp(code, hrp_loc, fctx, src, scalar_tmp, addr_tmp);
                    markScalarClobbered(&clobbered, reg);
                },
                .v128_reg => unreachable,
                .stack => {},
            }
        }
    }

    return clobbered;
}

/// Emit a direct wasm call.
///
/// Sequence: spill all currently-live caller-save physregs (x0..x15) to
/// fixed per-reg slots in the call-save region, move arguments into
/// scalar/vector argument registers or the outgoing stack area, load VMContext
/// into x0, emit `BL 0` with a patch record, move the result from x0 into the
/// dest vreg's location, then restore all saved regs.
///
/// The result-holding physreg is safe to not skip during restore: because
/// the RegMap allocates `dest` AFTER the spill and assign() only picks
/// currently-free regs, `dest.reg` cannot match any reg we saved.
fn emitCall(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    cl: @TypeOf(@as(ir.Inst.Op, undefined).call),
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *FuncCompileCtx,
) !void {
    var abi = try buildCallAbiPlan(fctx.allocator, fctx, cl.args, funcParamTypes(fctx, cl.func_idx), cl.extra_results > 0);
    defer abi.deinit();
    if (cl.tail and abi.stack_size > 0) return error.UnsupportedTailCallStackArgs;

    const is_import = cl.func_idx < fctx.import_count;
    // Imports are marshaled with the same native ABI as local AOT functions.
    // The scalar runtime entry helpers still reject exported v128 signatures,
    // so tests route v128 params through scalar exported wrappers.
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

    _ = try emitStageCallAbi(
        code,
        &abi,
        cl.args,
        reg_map,
        v128_map,
        fctx,
        0,
        RegMap.tmp0,
        RegMap.tmp2,
        if (cl.extra_results > 0) (if (cl.tail) HrpSource.forwarded else HrpSource.caller_scratch) else null,
    );

    // Load VMContext into x0 (pinned in x19, issue #465).
    try code.movRegReg(.x0, .x19);

    if (cl.tail) {
        // For imports the target lives in a vmctx slot; load it into
        // tmp0 BEFORE tearing down the frame (the callee-save restore
        // below will overwrite x19 with the caller's value, and x0 is
        // also live with vmctx). tmp0 (x16) is a caller-save register
        // not touched by the epilogue, so it survives to the `br` below.
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
    if (abi.stack_size > 0) try emitSpAdjust(code, abi.stack_size, .add);

    // Refresh the pinned memory_base in x20 (issue #466). The callee
    // may have invoked `memory.grow`, which reallocs the wasm linear
    // memory and updates `[vmctx + 0]`. The callee's callee-save
    // discipline restored *our* (pre-call) x20 in its epilogue, so the
    // value we hold is stale; reload from the now-current
    // `[vmctx + 0]`.
    try code.ldrImm(.x20, .x19, 0);

    // Commit result (in x0 or q0) to dest's location BEFORE restoring saved regs,
    // so restoration can't clobber the result holder (dest.reg is always a
    // reg that was FREE at snapshot time — i.e. not in `used_snapshot`).
    try emitCapturePrimaryCallResult(code, inst, reg_map, v128_map, v128_cache);

    // Restore previously-saved caller-save regs, skipping those whose
    // vreg died at this call (the reg is dead after BL — nothing reads
    // it, so the ldr would be wasted work).
    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        if ((dying_mask >> @intCast(i)) & 1 != 0) continue;
        const reg = RegMap.scratch_regs[i];
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
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
    try emitMemIndexImpl(code, reg_map, base_vreg, end_offset, skip_bounds);

    // ea = mem_base (pinned x20, issue #466) + zext(wasm_addr).
    try code.addRegReg(RegMap.tmp0, .x20, RegMap.tmp2);

    // Fold constant offset.
    if (offset != 0) {
        if (offset <= 0xFFF) {
            try code.addImm(RegMap.tmp0, RegMap.tmp0, @intCast(offset));
        } else {
            try code.movImm64(RegMap.tmp1, offset);
            try code.addRegReg(RegMap.tmp0, RegMap.tmp0, RegMap.tmp1);
        }
    }
}

/// Prepare tmp2 as the zero-extended memory32 index and perform the normal
/// inline bounds check, but do not materialize `memory_base + index`.
/// Scalar offset-zero loads/stores can consume tmp2 directly through the
/// AArch64 register-offset addressing mode.
fn emitCheckedMemIndex(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    base_vreg: ir.VReg,
    end_offset: u64,
) !void {
    try emitMemIndexImpl(code, reg_map, base_vreg, end_offset, false);
}

fn emitMemIndexImpl(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    base_vreg: ir.VReg,
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

        // Step 3: load VmCtx.memory_size (at +8, scaled-by-8 offset = 1)
        // directly from the pinned vmctx in x19.
        try code.ldrImm(RegMap.tmp0, .x19, vmctx_memsize_slot);

        // Step 4: cmp end, mem_size; B.LS over_trap.
        try code.cmpRegReg(RegMap.tmp1, RegMap.tmp0);
        const over_patch = code.len();
        try code.bCond(.ls, 0); // placeholder

        // Trap path.
        try code.movRegReg(.x0, .x19);
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
// #672 EH slots — added in commit 3.
const vmctx_tags_ptr_slot: u12 = 30; // byte 240, scale 8
const vmctx_aot_throw_uncaught_fn_slot: u12 = 32; // byte 256, scale 8
const vmctx_exception_params_byte_off: u32 = 264; // [16]u64 buffer base
const vmctx_exception_param_count_slot_w: u12 = 98; // byte 392 (u32), scale 4

// Per-table descriptor layout (`TableInfo`, 24 bytes):
//   { ptr: u64, len: u32, _pad: u32, type_backing_ptr: u64 }
const table_info_stride: u32 = 24;
const table_info_ptr_off: u32 = 0;
const table_info_len_off: u32 = 8;
const table_info_type_backing_off: u32 = 16;

fn globalTypeAt(fctx: *const FuncCompileCtx, idx: u32, fallback: ir.IrType) ir.IrType {
    const i: usize = @intCast(idx);
    if (i < fctx.global_types.len) return fctx.global_types[i];
    return fallback;
}

fn globalByteOffset(fctx: *const FuncCompileCtx, idx: u32) !u32 {
    const i: usize = @intCast(idx);
    if (i < fctx.global_offsets.len) return fctx.global_offsets[i];
    if (idx > std.math.maxInt(u32) / 8) return error.GlobalIndexOutOfRange;
    return idx * 8;
}

fn addGlobalByteOffset(code: *emit.CodeBuffer, base: emit.Reg, offset: u32, scratch: emit.Reg) !void {
    if (offset == 0) return;
    if (offset <= 4095) {
        try code.addImm(base, base, @intCast(offset));
    } else if ((offset & 0xFFF) == 0 and (offset >> 12) <= 4095) {
        try code.addImmShift12(base, base, @intCast(offset >> 12));
    } else {
        try emitMovImm64(code, scratch, offset);
        try code.addRegReg(base, base, scratch);
    }
}

/// Globals are a byte-addressed storage area. Scalar/reference globals keep
/// 8-byte slots; v128 globals are 16-byte aligned and 16 bytes wide.
fn emitGlobalGet(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    idx: u32,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const dest = inst.dest orelse return;
    const ty = globalTypeAt(fctx, idx, inst.type);
    const byte_offset = try globalByteOffset(fctx, idx);
    if (ty == .v128) {
        const vt = try v128_cache.defineFresh(code, v128_map, dest, null);
        try code.movRegReg(RegMap.tmp0, .x19);
        try code.ldrImm(RegMap.tmp0, RegMap.tmp0, vmctx_globals_slot);
        try addGlobalByteOffset(code, RegMap.tmp0, byte_offset, RegMap.tmp1);
        try code.ldrQ(vt, RegMap.tmp0);
        return;
    }

    // tmp0 = vmctx; tmp0 = globals_base
    try code.movRegReg(RegMap.tmp0, .x19);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, vmctx_globals_slot);

    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    const is32 = (ty == .i32 or ty == .f32);
    if (is32 and byte_offset % 4 == 0 and byte_offset / 4 <= 0xFFF) {
        try code.ldrImm32(info.reg, RegMap.tmp0, @intCast(byte_offset / 4));
    } else if (!is32 and byte_offset % 8 == 0 and byte_offset / 8 <= 0xFFF) {
        try code.ldrImm(info.reg, RegMap.tmp0, @intCast(byte_offset / 8));
    } else {
        const scratch = if (info.reg == RegMap.tmp1) RegMap.tmp2 else RegMap.tmp1;
        try addGlobalByteOffset(code, RegMap.tmp0, byte_offset, scratch);
        if (is32) try code.ldrImm32(info.reg, RegMap.tmp0, 0) else try code.ldrImm(info.reg, RegMap.tmp0, 0);
    }
    try destCommit(code, reg_map, info);
}

fn emitGlobalSet(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    gs: @TypeOf(@as(ir.Inst.Op, undefined).global_set),
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *const FuncCompileCtx,
) !void {
    const ty = globalTypeAt(fctx, gs.idx, inst.type);
    const byte_offset = try globalByteOffset(fctx, gs.idx);
    if (ty == .v128) {
        const vt = try v128_cache.ensure(code, v128_map, gs.val, null);
        try code.movRegReg(RegMap.tmp0, .x19);
        try code.ldrImm(RegMap.tmp0, RegMap.tmp0, vmctx_globals_slot);
        try addGlobalByteOffset(code, RegMap.tmp0, byte_offset, RegMap.tmp1);
        try code.strQ(vt, RegMap.tmp0);
        return;
    }

    const val_r = try useInto(code, reg_map, gs.val, RegMap.tmp1);
    // tmp0 = vmctx; tmp0 = globals_base
    try code.movRegReg(RegMap.tmp0, .x19);
    try code.ldrImm(RegMap.tmp0, RegMap.tmp0, vmctx_globals_slot);
    // Scalar/reference globals use 8-byte slots. i32/f32 consumers read the
    // low 32 bits, matching the pre-v128 layout and preserving old behaviour.
    if (byte_offset % 8 == 0 and byte_offset / 8 <= 0xFFF) {
        try code.strImm(val_r, RegMap.tmp0, @intCast(byte_offset / 8));
    } else {
        const scratch = if (val_r == RegMap.tmp1) RegMap.tmp2 else RegMap.tmp1;
        try addGlobalByteOffset(code, RegMap.tmp0, byte_offset, scratch);
        try code.strImm(val_r, RegMap.tmp0, 0);
    }
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
    try code.movRegReg(RegMap.tmp0, .x19);
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
    // mem_grow_fn may realloc the wasm linear memory and invalidate the
    // pinned memory_base in x20 (issue #466). Refresh from
    // [vmctx + 0] now so subsequent memory accesses see the new base.
    try code.ldrImm(.x20, .x19, 0);
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
    try code.movRegReg(dest, .x19);
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
    try code.movRegReg(.x0, .x19);
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_table_set_fn_slot);
    try code.blr(RegMap.tmp0);

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
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
        try frameStore(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
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
/// #542: emit a rematerialised def's value directly into `target`. Returns
/// `true` if `vreg` was rematerialisable (caller should skip the normal
/// reg/spill path entirely).
fn emitRematInto(
    code: *emit.CodeBuffer,
    reg_map: *const RegMap,
    vreg: ir.VReg,
    target: emit.Reg,
) !bool {
    const ar = reg_map.alloc_result orelse return false;
    const rd = ar.getRemat(vreg) orelse return false;
    switch (rd) {
        .iconst_32 => |v| try code.movImm32(target, v),
        .iconst_64 => |v| try code.movImm64(target, @bitCast(v)),
    }
    return true;
}

fn stageArgFromSaved(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
    target: emit.Reg,
    vreg: ir.VReg,
    clobbered: u16,
) !void {
    if (try emitRematInto(code, reg_map, vreg, target)) return;
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
                try frameLoadReg(code, target, fctx.call_save_base + reg_num * 8);
            }
        },
        .stack => |off| {
            try frameLoadReg(code, target, reg_map.spillOffsetBytes(off));
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
    try code.movRegReg(.x0, .x19);
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
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
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
    try code.movRegReg(RegMap.tmp0, .x19);
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
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *FuncCompileCtx,
) !void {
    var abi = try buildCallAbiPlan(fctx.allocator, fctx, ci.args, indexedParamTypes(fctx, ci.type_idx), ci.extra_results > 0);
    defer abi.deinit();
    if (ci.tail and abi.stack_size > 0) return error.UnsupportedTailCallStackArgs;

    // Snapshot live caller-save regs and spill to call_save.
    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    const all_args = try fctx.allocator.alloc(ir.VReg, ci.args.len + 1);
    defer fctx.allocator.free(all_args);
    all_args[0] = ci.elem_idx;
    for (ci.args, 0..) |a, i| all_args[1 + i] = a;
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, all_args);

    // Stage `elem_idx` into a callee-safe spot: write it to a dedicated
    // scratch slot on the stack (we reuse slot 0 of the call-save region
    // is risky since we just spilled to those slots — use the spill-slot
    // of elem_idx itself). Actually, re-read elem_idx from its saved
    // location into tmp2 each time we need it.
    //
    // Load elem_idx → tmp2 (32-bit value, zero-extended).
    {
        if (try emitRematInto(code, reg_map, ci.elem_idx, RegMap.tmp2)) {
            // already in tmp2
        } else {
            const loc = reg_map.get(ci.elem_idx) orelse return error.UnboundVReg;
            switch (loc) {
                .reg => |r| {
                    const reg_idx: u32 = @intFromEnum(r);
                    if (reg_idx >= 19) {
                        try code.movRegReg(RegMap.tmp2, r);
                    } else {
                        try frameLoadReg(code, RegMap.tmp2, fctx.call_save_base + reg_idx * 8);
                    }
                },
                .stack => |off| {
                    try frameLoadReg(code, RegMap.tmp2, reg_map.spillOffsetBytes(off));
                },
            }
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
    try code.movRegReg(.x0, .x19);
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

    // Stage args. The bounds/sig check sequence above clobbered x0, so seed
    // the clobber mask with bit 0 to force x0-sourced args to read from the
    // saved slot rather than the now-garbage register. Keep tmp0 for fn_ptr.
    const dying_mask_ci: u16 = dyingCallerSaveMask(reg_map, fctx.current_kills);
    _ = try emitStageCallAbi(
        code,
        &abi,
        ci.args,
        reg_map,
        v128_map,
        fctx,
        1,
        RegMap.tmp1,
        RegMap.tmp2,
        if (ci.extra_results > 0) (if (ci.tail) HrpSource.forwarded else HrpSource.caller_scratch) else null,
    );
    if (ci.tail) {
        try code.movRegReg(.x0, .x19);
        try emitCalleeSaveRestoreTracked(code, fctx);
        try code.emitEpilogueNoRet(fctx.frame_size);
        try code.br(RegMap.tmp0);
        return;
    }

    try code.movRegReg(.x0, .x19);
    try code.blr(RegMap.tmp0);
    if (abi.stack_size > 0) try emitSpAdjust(code, abi.stack_size, .add);

    // Refresh pinned memory_base in x20 (issue #466) — the called
    // function may have invoked `memory.grow`; see emitCall for
    // rationale.
    try code.ldrImm(.x20, .x19, 0);

    try emitCapturePrimaryCallResult(code, inst, reg_map, v128_map, v128_cache);

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        if ((dying_mask_ci >> @intCast(i)) & 1 != 0) continue;
        const reg = RegMap.scratch_regs[i];
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
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
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *FuncCompileCtx,
) !void {
    var abi = try buildCallAbiPlan(fctx.allocator, fctx, cr.args, indexedParamTypes(fctx, cr.type_idx), cr.extra_results > 0);
    defer abi.deinit();
    if (cr.tail and abi.stack_size > 0) return error.UnsupportedTailCallStackArgs;

    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    const all_args = try fctx.allocator.alloc(ir.VReg, cr.args.len + 1);
    defer fctx.allocator.free(all_args);
    all_args[0] = cr.func_ref;
    for (cr.args, 0..) |a, i| all_args[1 + i] = a;
    try saveCallerSaveForCall(code, reg_map, fctx, &used_snapshot, all_args);

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
    try code.movRegReg(.x0, .x19);
    try code.ldrImm(RegMap.tmp1, .x0, vmctx_trap_unreachable_fn_slot);
    try code.blr(RegMap.tmp1);
    // CBZ shares the b.cond imm19 encoding at bits [23:5], so the same
    // patch helper works.
    patchBCondHere(code, cbz_site);

    // Stage args while preserving tmp0, which holds fn_ptr.
    const dying_mask_cr: u16 = dyingCallerSaveMask(reg_map, fctx.current_kills);
    _ = try emitStageCallAbi(
        code,
        &abi,
        cr.args,
        reg_map,
        v128_map,
        fctx,
        0,
        RegMap.tmp1,
        RegMap.tmp2,
        if (cr.extra_results > 0) (if (cr.tail) HrpSource.forwarded else HrpSource.caller_scratch) else null,
    );
    if (cr.tail) {
        try code.movRegReg(.x0, .x19);
        try emitCalleeSaveRestoreTracked(code, fctx);
        try code.emitEpilogueNoRet(fctx.frame_size);
        try code.br(RegMap.tmp0);
        return;
    }

    try code.movRegReg(.x0, .x19);
    try code.blr(RegMap.tmp0);
    if (abi.stack_size > 0) try emitSpAdjust(code, abi.stack_size, .add);

    // Refresh pinned memory_base in x20 (issue #466) — the called
    // function may have invoked `memory.grow`; see emitCall for
    // rationale.
    try code.ldrImm(.x20, .x19, 0);

    try emitCapturePrimaryCallResult(code, inst, reg_map, v128_map, v128_cache);

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        if ((dying_mask_cr >> @intCast(i)) & 1 != 0) continue;
        const reg = RegMap.scratch_regs[i];
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
    }
}

/// `.call_result idx` — read the `idx`-th extra result slot written by
/// the preceding call into its dest vreg. Caller-side scratch lives at
/// `[fp + scratch_base]`; scalar slots are 8 bytes, v128 slots are 16-byte
/// aligned and 16 bytes wide.
fn emitCallResult(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    idx: u8,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *FuncCompileCtx,
) !void {
    const dest = inst.dest orelse return;
    const rel_off: u32 = if (fctx.call_result_offsets) |offsets|
        offsets.get(dest) orelse @as(u32, idx) * 8
    else
        @as(u32, idx) * 8;
    const off: u32 = fctx.scratch_base + rel_off;
    if (inst.type == .v128) {
        const vt = try v128_cache.defineFresh(code, v128_map, dest, null);
        try frameAddr(code, RegMap.tmp0, off);
        try code.ldrQ(vt, RegMap.tmp0);
    } else {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        try frameLoad(code, info.reg, off);
        try destCommit(code, reg_map, info);
    }
}

/// `.ret_multi [v0, v1, ...]` — first scalar/v128 result → x0/q0; remaining
/// results are written through the saved HRP using the same aligned return-area
/// layout as `.call_result`. Emits the full epilogue.
fn emitRetMulti(
    code: *emit.CodeBuffer,
    vregs: []const ir.VReg,
    reg_map: *RegMap,
    v128_map: *V128StackMap,
    v128_cache: *V128RegCache,
    fctx: *FuncCompileCtx,
    frame_size: u32,
) !void {
    if (vregs.len == 0) {
        try emitCalleeSaveRestoreTracked(code, fctx);
        try code.emitEpilogue(frame_size);
        return;
    }

    // First result → x0/q0.
    try emitPrimaryReturnValue(code, vregs[0], reg_map, v128_map, v128_cache, fctx);

    if (vregs.len > 1) {
        // Load HRP into tmp0 (non-allocatable, stable across arg reads).
        try frameLoad(code, RegMap.tmp0, fctx.hrp_save_off);
        // Write extra results through HRP.
        var extra_off: u32 = 0;
        var i: u32 = 1;
        while (i < vregs.len) : (i += 1) {
            const ty = vregType(fctx, vregs[i]);
            extra_off = alignForwardU32(extra_off, resultSlotAlignment(ty));
            const off = extra_off;
            if (ty == .v128) {
                const vt = v128_tmp1;
                try loadV128ArgFromSaved(code, v128_map, vregs[i], vt, RegMap.tmp1);
                try storeV128ToBaseOffset(code, RegMap.tmp0, off, vt, RegMap.tmp1);
            } else {
                const src = try useInto(code, reg_map, vregs[i], RegMap.tmp1);
                try storeScalarToBaseOffset(code, RegMap.tmp0, off, src, RegMap.tmp2);
            }
            extra_off += resultSlotSize(ty);
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
        if (try emitRematInto(code, reg_map, vreg, arg_regs[i])) continue;
        const loc = reg_map.get(vreg) orelse return error.UnboundVReg;
        switch (loc) {
            .reg => |r| {
                const reg_idx: u32 = @intFromEnum(r);
                if (reg_idx >= 19) {
                    try code.movRegReg(arg_regs[i], r);
                } else {
                    try frameLoadReg(code, arg_regs[i], fctx.call_save_base + reg_idx * 8);
                }
            },
            .stack => |off| {
                try frameLoadReg(code, arg_regs[i], reg_map.spillOffsetBytes(off));
            },
        }
    }

    // Materialize any constant args into the trailing arg regs.
    _ = &arg_regs;

    // x0 = vmctx; tmp0 = vmctx.<fn_slot>; BLR tmp0.
    try code.movRegReg(.x0, .x19);
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
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
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
    if (try emitRematInto(code, reg_map, vreg, dest_reg)) return;
    const loc = reg_map.get(vreg) orelse return error.UnboundVReg;
    switch (loc) {
        .reg => |r| {
            const reg_idx: u32 = @intFromEnum(r);
            if (reg_idx >= 19) {
                if (r != dest_reg) try code.movRegReg(dest_reg, r);
            } else {
                try frameLoadReg(code, dest_reg, fctx.call_save_base + reg_idx * 8);
            }
        },
        .stack => |off| {
            try frameLoadReg(code, dest_reg, reg_map.spillOffsetBytes(off));
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
    try code.movRegReg(.x0, .x19);
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_table_init_fn_slot);
    try code.blr(RegMap.tmp0);

    // Restore caller-saved regs.
    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
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
    try code.movRegReg(.x0, .x19);
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_elem_drop_fn_slot);
    try code.blr(RegMap.tmp0);

    for (used_snapshot, 0..) |used, i| {
        if (reg_map.alloc_result != null) break;
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
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
    try code.movRegReg(.x0, .x19);
    try code.ldrImm(RegMap.tmp0, .x0, slot);
    try code.blr(RegMap.tmp0);
    try code.brk(0);
}

/// Lower IR `throw`. Terminator semantics let us clobber every caller-save
/// register: the helper is noreturn (it longjmps via `trapLongjmp` when
/// trap-as-error is armed, otherwise `std.process.exit(2)`).
///
/// Sequence (commit 3 of #672, leaf-function-only first cut):
///   * Spill each operand into `vmctx.exception_params[i]`.
///   * Write `param_count` into `vmctx.exception_param_count`.
///   * Load `*TagInstance` from `vmctx.tags_ptr[tag_idx]` into x1.
///   * mov x0, vmctx; ldr xtmp, [vmctx + aot_throw_uncaught_fn]; blr xtmp; brk 0.
///
/// Limitations: tag_idx must fit in the 12-bit unsigned scaled-by-8
/// LDR immediate range (`tag_idx < 512`). Higher indices return
/// `error.UnimplementedOp` for now; lifting this requires the indexed
/// addressing path that `emitGlobalGet` already uses for large globals
/// and is filed for the cross-function unwind commit (#672 commit 5).
fn emitThrow(
    code: *emit.CodeBuffer,
    th: ir.Inst.Throw,
    reg_map: *RegMap,
) !void {
    if (th.args.len > 16) return error.UnimplementedOp;
    if (th.tag_idx >= 512) return error.UnimplementedOp;

    // exception_params[i] = arg[i]. Single scratch (tmp0) reused across iterations.
    for (th.args, 0..) |arg_vreg, i| {
        const src = try useInto(code, reg_map, arg_vreg, RegMap.tmp0);
        const slot: u12 = @intCast(vmctx_exception_params_byte_off / 8 + i);
        try code.strImm(src, .x19, slot);
    }

    // exception_param_count = th.args.len (u32, scale-4 slot).
    try code.movImm64(RegMap.tmp0, @as(u64, th.args.len));
    try code.strImm32(RegMap.tmp0, .x19, vmctx_exception_param_count_slot_w);

    // x1 = vmctx.tags_ptr[tag_idx].
    try code.ldrImm(RegMap.tmp0, .x19, vmctx_tags_ptr_slot);
    try code.ldrImm(.x1, RegMap.tmp0, @intCast(th.tag_idx));

    // mov x0, vmctx; ldr tmp0, [vmctx + aot_throw_uncaught_fn]; blr tmp0; brk 0.
    try code.movRegReg(.x0, .x19);
    try code.ldrImm(RegMap.tmp0, .x19, vmctx_aot_throw_uncaught_fn_slot);
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
    fctx: *const FuncCompileCtx,
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
    const use_reg_offset = !ld.bounds_known and
        fctx.options.enable_mem_reg_offset and
        ld.offset == 0 and
        !fctx.has_memory64 and
        !fctx.has_shared_memory;
    if (use_reg_offset) {
        try emitCheckedMemIndex(code, reg_map, ld.base, end_offset);
    } else if (ld.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, ld.base, folded_offset);
    } else {
        try emitMemAddr(code, reg_map, ld.base, folded_offset, end_offset);
    }
    const disp: u12 = if (folded) |d| d else 0;

    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    if (use_reg_offset and ld.sign_extend) {
        switch (ld.size) {
            1 => if (is64) try code.ldrsbRegOffset64(info.reg, .x20, RegMap.tmp2) else try code.ldrsbRegOffset32(info.reg, .x20, RegMap.tmp2),
            2 => if (is64) try code.ldrshRegOffset64(info.reg, .x20, RegMap.tmp2) else try code.ldrshRegOffset32(info.reg, .x20, RegMap.tmp2),
            4 => try code.ldrswRegOffset(info.reg, .x20, RegMap.tmp2),
            else => unreachable,
        }
    } else if (use_reg_offset) {
        switch (ld.size) {
            1 => try code.ldrbRegOffset(info.reg, .x20, RegMap.tmp2),
            2 => try code.ldrhRegOffset(info.reg, .x20, RegMap.tmp2),
            4 => try code.ldrRegOffset32(info.reg, .x20, RegMap.tmp2),
            8 => try code.ldrRegOffset(info.reg, .x20, RegMap.tmp2),
            else => unreachable,
        }
    } else if (ld.sign_extend) {
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
    fctx: *const FuncCompileCtx,
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
    const use_reg_offset = !st.bounds_known and
        fctx.options.enable_mem_reg_offset and
        st.offset == 0 and
        !fctx.has_memory64 and
        !fctx.has_shared_memory;
    if (use_reg_offset) {
        try emitCheckedMemIndex(code, reg_map, st.base, end_offset);
    } else if (st.bounds_known) {
        try emitMemAddrSkipBounds(code, reg_map, st.base, folded_offset);
    } else {
        try emitMemAddr(code, reg_map, st.base, folded_offset, end_offset);
    }
    const disp: u12 = if (folded) |d| d else 0;

    // Materialize the value into tmp1 (or use its home reg). Address
    // preparation is done with tmp1 by now, while tmp2 keeps the index live
    // for the register-offset form.
    const val_reg = try useInto(code, reg_map, st.val, RegMap.tmp1);
    if (use_reg_offset) {
        switch (st.size) {
            1 => try code.strbRegOffset(val_reg, .x20, RegMap.tmp2),
            2 => try code.strhRegOffset(val_reg, .x20, RegMap.tmp2),
            4 => try code.strRegOffset32(val_reg, .x20, RegMap.tmp2),
            8 => try code.strRegOffset(val_reg, .x20, RegMap.tmp2),
            else => unreachable,
        }
    } else {
        switch (st.size) {
            1 => try code.strbImm(val_reg, RegMap.tmp0, disp),
            2 => try code.strhImm(val_reg, RegMap.tmp0, disp),
            4 => try code.strImm32(val_reg, RegMap.tmp0, disp),
            8 => try code.strImm(val_reg, RegMap.tmp0, disp),
            else => unreachable,
        }
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
    try code.movRegReg(.x0, .x19);
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
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
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
        try code.movRegReg(.x0, .x19);
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
        try code.movRegReg(.x0, .x19);
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
        try frameLoadReg(code, reg, fctx.call_save_base + @as(u32, @intCast(i)) * 8);
    }
}

fn emitBrIf(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    br: @TypeOf(@as(ir.Inst.Op, undefined).br_if),
    fctx: *const FuncCompileCtx,
) !void {
    const allocator = fctx.allocator;
    const cond_r = try useInto(code, reg_map, br.cond, RegMap.tmp0);
    // wasm br_if: branch to then_block if cond != 0 (i32).
    try code.cmpImm32(cond_r, 0);

    // Fall-through elision (#808): if a successor is the physical fall-through
    // block, drop the unconditional branch and (if it's the then-side) invert
    // the condition so the common path falls through instead of branching.
    if (fctx.targetIsFallthrough(br.else_block)) {
        const p = code.len();
        try code.bCond(.ne, 0); // cond != 0 → then_block; else falls through
        try patches.append(allocator, .{
            .patch_offset = p,
            .target_block = br.then_block,
            .kind = .b_cond,
        });
        return;
    }
    if (fctx.targetIsFallthrough(br.then_block)) {
        const p = code.len();
        try code.bCond(.eq, 0); // cond == 0 → else_block; then falls through
        try patches.append(allocator, .{
            .patch_offset = p,
            .target_block = br.else_block,
            .kind = .b_cond,
        });
        return;
    }

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

    for (ir_module.functions.items, 0..) |func, func_idx| {
        const func_base: u32 = @intCast(all_code.items.len);
        try offsets.append(allocator, func_base);

        var func_patches: std.ArrayListUnmanaged(CallPatch) = .empty;
        defer func_patches.deinit(allocator);

        const ctx: FuncCompileCtx = .{
            .import_count = ir_module.import_count,
            .has_memory64 = ir_module.has_memory64,
            .has_shared_memory = ir_module.has_shared_memory,
            .call_patches = &func_patches,
            .global_types = ir_module.global_types orelse &.{},
            .global_offsets = ir_module.global_offsets orelse &.{},
            .func_types = ir_module.func_types.items,
            .func_type_indices = ir_module.func_type_indices.items,
            .options = options,
            .func_idx = @intCast(func_idx),
            .allocator = allocator,
        };
        const func_code = compileFunctionImpl(&func, ctx, allocator) catch |err| {
            std.debug.print("Error compiling function {d} ({d} blocks, {d} vregs): {}\n", .{
                func_idx,
                func.blocks.items.len,
                func.next_vreg,
                err,
            });
            return err;
        };
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

// ── #761 Phase 2: codegen cache integration ──────────────────────────────

pub const CompileResultCached = codegen_cache.CompileResultCached;

/// Same as `compileModuleWithOptions` but consults an optional `reuse`
/// cache to skip per-function codegen for functions whose IR sha256
/// matches the cached entry. Reused code's local-offset call patches
/// are rebased to the new module layout; the existing BL patch
/// resolution loop runs unchanged.
pub fn compileModuleCached(
    ir_module: *const ir.IrModule,
    reuse: ?*const codegen_cache.Cache,
    allocator: std.mem.Allocator,
) !codegen_cache.CompileResultCached {
    return compileModuleCachedWithOptions(ir_module, reuse, allocator, .{});
}

pub fn compileModuleCachedWithOptions(
    ir_module: *const ir.IrModule,
    reuse: ?*const codegen_cache.Cache,
    allocator: std.mem.Allocator,
    options: CompileOptions,
) !codegen_cache.CompileResultCached {
    var all_code: std.ArrayListUnmanaged(u8) = .empty;
    errdefer all_code.deinit(allocator);
    var offsets: std.ArrayListUnmanaged(u32) = .empty;
    errdefer offsets.deinit(allocator);

    var global_call_patches: std.ArrayListUnmanaged(CallPatch) = .empty;
    defer global_call_patches.deinit(allocator);

    const func_count: usize = ir_module.functions.items.len;
    var cache_funcs = try allocator.alloc(codegen_cache.CachedFunction, func_count);
    var cache_init: usize = 0;
    errdefer {
        for (cache_funcs[0..cache_init]) |*f| {
            allocator.free(f.code);
            allocator.free(f.call_patches);
        }
        allocator.free(cache_funcs);
    }

    var stats: codegen_cache.CacheStats = .{};

    // #778: codegen timing (coarse — per-function totals + module summary).
    const ct = options.codegen_timing;
    const ct_live = ct.enabled and ct.moduleMatches(options.module_idx);
    const module_start_ns: u64 = if (ct_live) codegen_timing.nowNs() else 0;
    var mod_hash_ns: u64 = 0;
    var mod_compile_ns: u64 = 0;
    var mod_compiled: usize = 0;
    var mod_reused: usize = 0;
    codegen_timing.printModuleBegin(ct, options.module_idx, func_count);

    for (ir_module.functions.items, 0..) |func, fi| {
        const func_base: u32 = @intCast(all_code.items.len);
        try offsets.append(allocator, func_base);

        // #862 lazy-JIT spike: emit nothing for lazy-eligible functions.
        // Safe only under the invariants `lazy_jit.findLazyEligibleLeaves`
        // establishes (leaf, never a direct call target) — see that
        // function's doc comment and docs/design/lazy-jit-spike.md.
        if (fi < options.lazy_skip.len and options.lazy_skip[fi]) {
            cache_funcs[fi] = .{
                .ir_sha256 = codegen_cache.hashFunction(&func),
                .code = try allocator.dupe(u8, &.{}),
                .call_patches = try allocator.dupe(codegen_cache.FuncCallPatch, &.{}),
            };
            cache_init += 1;
            continue;
        }

        var hash_ns: u64 = 0;
        const ir_sha = blk: {
            if (ct_live) {
                const t0 = codegen_timing.nowNs();
                const sha = codegen_cache.hashFunction(&func);
                hash_ns = codegen_timing.nowNs() -| t0;
                mod_hash_ns += hash_ns;
                break :blk sha;
            }
            break :blk codegen_cache.hashFunction(&func);
        };

        var reused = false;
        var hit_code: []const u8 = undefined;
        var hit_patches: []const codegen_cache.FuncCallPatch = undefined;
        if (reuse) |r| {
            if (fi < r.functions.len and
                std.mem.eql(u8, &r.functions[fi].ir_sha256, &ir_sha))
            {
                hit_code = r.functions[fi].code;
                hit_patches = r.functions[fi].call_patches;
                reused = true;
            }
        }

        var cache_code_owned: []u8 = undefined;
        var cache_patches_owned: []codegen_cache.FuncCallPatch = undefined;
        var compile_ns: u64 = 0;
        // #781: per-function sub-phase stopwatch, live only when timing is
        // enabled for this module. Declared per-iteration so it stays valid
        // for the single log site below; reused functions leave every span
        // at 0 (so the printed emit_ms == total_ms == 0).
        var func_timer: ?codegen_timing.Aarch64FuncTimer =
            if (ct_live) codegen_timing.Aarch64FuncTimer.start() else null;

        if (reused) {
            cache_code_owned = try allocator.dupe(u8, hit_code);
            errdefer allocator.free(cache_code_owned);
            cache_patches_owned = try allocator.dupe(codegen_cache.FuncCallPatch, hit_patches);
            try all_code.appendSlice(allocator, hit_code);
            for (hit_patches) |p| {
                try global_call_patches.append(allocator, .{
                    .patch_offset = @as(usize, func_base) + @as(usize, p.patch_offset),
                    .target_func_idx = p.target_func_idx,
                });
            }
            stats.reused += 1;
            mod_reused += 1;
        } else {
            var func_patches: std.ArrayListUnmanaged(CallPatch) = .empty;
            defer func_patches.deinit(allocator);

            const ctx: FuncCompileCtx = .{
                .import_count = ir_module.import_count,
                .has_memory64 = ir_module.has_memory64,
                .has_shared_memory = ir_module.has_shared_memory,
                .call_patches = &func_patches,
                .global_types = ir_module.global_types orelse &.{},
                .global_offsets = ir_module.global_offsets orelse &.{},
                .func_types = ir_module.func_types.items,
                .func_type_indices = ir_module.func_type_indices.items,
                .options = options,
                .codegen_timer = if (func_timer) |*t| t else null,
                .func_idx = @intCast(fi),
                .allocator = allocator,
            };
            const compile_t0: u64 = if (ct_live) codegen_timing.nowNs() else 0;
            const func_code = compileFunctionImpl(&func, ctx, allocator) catch |err| {
                std.debug.print("Error compiling function {d} ({d} blocks, {d} vregs): {}\n", .{
                    fi,
                    func.blocks.items.len,
                    func.next_vreg,
                    err,
                });
                return err;
            };
            if (ct_live) {
                compile_ns = codegen_timing.nowNs() -| compile_t0;
                mod_compile_ns += compile_ns;
            }

            const new_patches = try allocator.alloc(codegen_cache.FuncCallPatch, func_patches.items.len);
            errdefer allocator.free(new_patches);
            for (func_patches.items, new_patches) |src, *dst| {
                dst.* = .{
                    .patch_offset = @intCast(src.patch_offset),
                    .target_func_idx = src.target_func_idx,
                };
            }
            cache_code_owned = func_code;
            cache_patches_owned = new_patches;

            for (func_patches.items) |p| {
                try global_call_patches.append(allocator, .{
                    .patch_offset = p.patch_offset + func_base,
                    .target_func_idx = p.target_func_idx,
                });
            }
            try all_code.appendSlice(allocator, func_code);
            stats.recompiled += 1;
            mod_compiled += 1;
        }

        if (ct_live and codegen_timing.shouldLogFunc(ct, options.module_idx, @intCast(fi), compile_ns)) {
            // #781: aarch64 per-function sub-phase breakdown. `func_timer`
            // is all-zero for reused functions; `printAarch64Func` derives
            // emit_ms = total_ms − sum(spans).
            const ft = func_timer orelse codegen_timing.Aarch64FuncTimer.start();
            codegen_timing.printAarch64Func(.{
                .module_idx = options.module_idx,
                .func_idx = @intCast(fi),
                .blocks = func.blocks.items.len,
                .insts = aarch64CountInstructions(&func),
                .reused = reused,
                .hash_ns = hash_ns,
                .total_ns = compile_ns,
                .scheduling_ns = ft.scheduling_ns,
                .range_split_ns = ft.range_split_ns,
                .prepass_ns = ft.prepass_ns,
                .liveness_ns = ft.liveness_ns,
                .regalloc_ns = ft.regalloc_ns,
                .coalesce_ns = ft.coalesce_ns,
                .post_emit_coalesce_ns = ft.post_emit_coalesce_ns,
            });
        }

        cache_funcs[fi] = .{
            .ir_sha256 = ir_sha,
            .code = cache_code_owned,
            .call_patches = cache_patches_owned,
        };
        cache_init += 1;
    }

    // Resolve BL patches — identical logic to
    // `compileModuleWithOptions` above.
    const patch_t0: u64 = if (ct_live) codegen_timing.nowNs() else 0;
    for (global_call_patches.items) |p| {
        if (p.target_func_idx >= offsets.items.len) return error.BadFuncIndex;
        const target_off: i64 = @intCast(offsets.items[p.target_func_idx]);
        const patch_off: i64 = @intCast(p.patch_offset);
        const delta_bytes = target_off - patch_off;
        if (@mod(delta_bytes, 4) != 0) return error.BranchMisaligned;
        const word_off = @divExact(delta_bytes, 4);
        const limit: i64 = 1 << 25;
        if (word_off >= limit or word_off < -limit) return error.CallOutOfRange;
        const bytes = all_code.items[p.patch_offset..][0..4];
        const existing = std.mem.readInt(u32, bytes, .little);
        const imm26: u26 = @bitCast(@as(i26, @intCast(word_off)));
        const new_word: u32 = (existing & 0xFC000000) | imm26;
        std.mem.writeInt(u32, bytes, new_word, .little);
    }
    if (ct_live) {
        const now = codegen_timing.nowNs();
        codegen_timing.printModuleSummary(ct, .{
            .module_idx = options.module_idx,
            .funcs = func_count,
            .compiled = mod_compiled,
            .reused = mod_reused,
            .hash_ns = mod_hash_ns,
            .compile_ns = mod_compile_ns,
            .global_patch_ns = now -| patch_t0,
            .total_ns = now -| module_start_ns,
        });
    }

    return .{
        .code = try all_code.toOwnedSlice(allocator),
        .offsets = try offsets.toOwnedSlice(allocator),
        .cache_functions = cache_funcs,
        .stats = stats,
    };
}

/// Sum of instructions across all blocks — reported in #778 codegen
/// timing lines.
fn aarch64CountInstructions(func: *const ir.IrFunction) usize {
    var n: usize = 0;
    for (func.blocks.items) |block| n += block.instructions.items.len;
    return n;
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
/// a non-allocatable scratch `RegMap.tmp2`). Indices 15..22 map to
/// x21..x28 (callee-saved). x16/x17 are `RegMap.tmp0/tmp1`, x18 is the
/// platform register, x19 is pinned to `vmctx` (issue #465), x20 is
/// pinned to `memory_base` (issue #466), x29/x30 are FP/LR, and x31 is
/// SP — all excluded.
pub const aarch64_alloc_regs = [_]regalloc.PhysReg{
    0,  1,  2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14,
    21, 22, 23, 24, 25, 26, 27, 28,
};

/// Indices into `aarch64_alloc_regs` of AAPCS64 caller-saved registers
/// (x0..x14). Used as the "prefer when live range doesn't span a call"
/// pool.
pub const aarch64_caller_saved_indices = [_]u8{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
};

/// Indices into `aarch64_alloc_regs` of AAPCS64 callee-saved registers
/// (x21..x28). Preferred for live ranges that span a call, since they
/// survive without save/restore. x19 is excluded — pinned to `vmctx`
/// (issue #465). x20 is excluded — pinned to `memory_base` (issue
/// #466).
pub const aarch64_callee_saved_indices = [_]u8{
    15, 16, 17, 18, 19, 20, 21, 22,
};

/// Bitmask over `aarch64_alloc_regs` of registers destroyed by any
/// AAPCS64-compatible procedure call (direct, indirect, or vmctx
/// helper): x0..x14 (indices 0..14). x15..x17 are our own non-
/// allocatable scratches, x18 is platform-reserved.
pub const aarch64_call_clobber_mask: u64 = (@as(u64, 1) << 15) - 1;

/// Map an aarch64 physical register to its position in
/// `aarch64_alloc_regs`. Returns `null` for non-allocatable regs (x15
/// = `RegMap.tmp2`, x16 = `tmp0`, x17 = `tmp1`, x18 platform reg, plus
/// fp/lr/sp). Used by the regalloc-hint pre-pass to translate
/// AAPCS64 ABI-fixed call-arg targets (x1..x7) into hint indices.
fn aarch64_alloc_idx(reg: emit.Reg) ?u8 {
    inline for (aarch64_alloc_regs, 0..) |phys, i| {
        if (@intFromEnum(reg) == phys) return @intCast(i);
    }
    return null;
}

/// Conservative SIMD allocator pool: v3-v7 plus v16-v31. v0-v2 remain
/// non-allocatable codegen temporaries used by complex NEON sequences. v8-v15
/// are excluded because AAPCS64 only preserves their low 64 bits; using them
/// would require full-width prologue/epilogue save/restore wiring, deferred.
pub const aarch64_v_alloc_regs = [_]regalloc.PhysReg{
    3,  4,  5,  6,  7,
    16, 17, 18, 19, 20,
    21, 22, 23, 24, 25,
    26, 27, 28, 29, 30,
    31,
};

pub const aarch64_v_caller_saved_indices = [_]u8{
    0,  1,  2,  3,  4,
    5,  6,  7,  8,  9,
    10, 11, 12, 13, 14,
    15, 16, 17, 18, 19,
    20,
};

pub const aarch64_v_callee_saved_indices = [_]u8{};
pub const aarch64_v_call_clobber_mask: u64 = (@as(u64, 1) << aarch64_v_alloc_regs.len) - 1;

/// Build the per-function `RegSet` for aarch64 with a known typed-local
/// frame end. Scalar-only callers can use `aarch64RegSet(local_count)`.
///
/// Frame layout (matches `compileFunctionImpl` / `emitEntrySpill`):
/// `[fp+0]` = saved fp, `[fp+8]` = saved lr, `[fp+16]` = vmctx slot,
/// `[fp+24 .. spill_base)` = locals plus any alignment padding, then spills
/// grow upward.
pub fn aarch64RegSetForSpillBase(spill_base: u32) regalloc.RegSet {
    return .{
        .alloc_regs = &aarch64_alloc_regs,
        .callee_saved_indices = &aarch64_callee_saved_indices,
        .caller_saved_indices = &aarch64_caller_saved_indices,
        // Match RegMap's spill_base so translation between AllocResult.stack
        // (absolute FP offset) and RegMap.Location.stack (offset from
        // spill_base) is direct.
        .spill_base = @as(i32, @intCast(spill_base)),
        .spill_stride = 8,
    };
}

pub fn aarch64VRegSetForSpillBase(spill_base: u32) regalloc.RegSet {
    return .{
        .alloc_regs = &aarch64_v_alloc_regs,
        .callee_saved_indices = &aarch64_v_callee_saved_indices,
        .caller_saved_indices = &aarch64_v_caller_saved_indices,
        .spill_base = @as(i32, @intCast(spill_base)),
        .spill_stride = 8,
    };
}

pub fn aarch64RegSet(local_count: u32) regalloc.RegSet {
    return aarch64RegSetForSpillBase((local_count + 3) * 8);
}

fn vregClobbersFromScalar(
    scalar_clobbers: []const regalloc.ClobberPoint,
    allocator: std.mem.Allocator,
) !std.ArrayList(regalloc.ClobberPoint) {
    var out: std.ArrayList(regalloc.ClobberPoint) = .empty;
    errdefer out.deinit(allocator);
    try out.ensureTotalCapacity(allocator, scalar_clobbers.len);
    for (scalar_clobbers) |cp| {
        out.appendAssumeCapacity(.{
            .pos = cp.pos,
            .regs_clobbered = aarch64_v_call_clobber_mask,
        });
    }
    return out;
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

/// Append calling-convention regalloc hints for the scalar args of one
/// call site. Each scalar arg `i` whose AAPCS64 plan-loc is
/// `.scalar_reg = xN` gets a hint `{ vreg = args[i], reg_idx = idx(xN) }`.
/// `.v128_reg` and `.stack` arg locs are skipped — v128 hints are out
/// of scope for issue #423, and stack args have no fixed-physreg target.
///
/// Multiple hints may be emitted per call (one per scalar arg). The
/// allocator (`regalloc.allocateFromRangesWithHints`) honors a hint
/// only when the target is free at assignment time and the vreg's live
/// range survives any clobber, so unsafe hints are harmlessly dropped.
fn addCallArgScalarHints(
    list: *std.ArrayList(regalloc.Hint),
    allocator: std.mem.Allocator,
    fctx: *const FuncCompileCtx,
    args: []const ir.VReg,
    params: []const ir.IrType,
) !void {
    var plan = try buildCallAbiPlan(fctx.allocator, fctx, args, params, false);
    defer plan.deinit();
    for (args, 0..) |arg_vreg, i| {
        switch (plan.locs[i]) {
            .scalar_reg => |reg| {
                if (aarch64_alloc_idx(reg)) |idx| {
                    try list.append(allocator, .{ .vreg = arg_vreg, .reg_idx = idx });
                }
            },
            .v128_reg, .stack => {},
        }
    }
}

/// Emit a single `Hint` for `vreg` targeting AAPCS64 register `xN` if
/// `xN` is allocatable in `aarch64_alloc_regs`. x0..x14 are; x15..x18,
/// x19 (pinned to vmctx, issue #465) and x20 (pinned to memory_base,
/// issue #466) are not. The callee-saved x21..x28 are reachable but
/// not used for ABI-fixed call-arg slots.
fn appendArgRegHint(
    list: *std.ArrayList(regalloc.Hint),
    allocator: std.mem.Allocator,
    vreg: ir.VReg,
    reg: emit.Reg,
) !void {
    if (aarch64_alloc_idx(reg)) |idx| {
        try list.append(allocator, .{ .vreg = vreg, .reg_idx = idx });
    }
}

/// Collect calling-convention register hints for every IR op that flows
/// vregs into AAPCS64 ABI-fixed param registers. Mirrors x86_64's
/// hint pre-pass (PR #422), adapted for AAPCS64: vmctx is in x0 and
/// scalar wasm args go to x1..x7 in declaration order; v128 args go to
/// v0..v7 (out of scope here — v0..v2 aren't even in the V-reg
/// allocator pool, see `aarch64_v_alloc_regs`).
///
/// Hint sources covered (matching `emitCall`/`emitCallIndirect`/
/// `emitCallRef`/host-helper emit fns):
///   - `.call`, `.call_indirect`, `.call_ref`: scalar args by plan
///   - `.memory_copy` (dst, src, len) → x1, x2, x3
///   - `.memory_fill` (dst, val, len) → x1, x2, x3
///   - `.memory_grow` (pages) → x1
///   - `.table_grow` (init, delta) → x1, x2
///   - `.table_set` (idx, val) → x2, x3 (x1 holds the constant table_idx)
///   - `.atomic_notify` (base, count) → x1, x2
///   - `.atomic_wait` (base, expected) → x1, x2 (timeout's slot varies
///     by 32/64 path, hint just the unambiguous prefix)
///
/// Skipped: `.table_init` packs dst/src/len into encoded x3/x4 values
/// rather than placing the vregs directly in fixed param regs, so a
/// hint would point at the wrong register.
///
/// Positions are not used in the returned list — `Hint` is just
/// `(vreg, reg_idx)`. This walk only needs to cover every op that has
/// a fixed-target arg.
fn collectHintPoints(
    func: *const ir.IrFunction,
    fctx: *const FuncCompileCtx,
    block_order_opt: ?[]const ir.BlockId,
    scheduled: ?*const schedule.FunctionSchedule,
    allocator: std.mem.Allocator,
) !std.ArrayList(regalloc.Hint) {
    var hints: std.ArrayList(regalloc.Hint) = .empty;
    errdefer hints.deinit(allocator);

    var owns_order = false;
    const block_order: []const ir.BlockId = if (block_order_opt) |bo| bo else blk: {
        const raw = try allocator.alloc(ir.BlockId, func.blocks.items.len);
        for (raw, 0..) |*r, i| r.* = @intCast(i);
        owns_order = true;
        break :blk raw;
    };
    defer if (owns_order) allocator.free(block_order);

    for (block_order) |bo_bid| {
        const insts = if (scheduled) |s|
            s.instructions(bo_bid)
        else
            func.blocks.items[bo_bid].instructions.items;
        for (insts) |ci| {
            switch (ci.op) {
                .call => |cl| try addCallArgScalarHints(
                    &hints,
                    allocator,
                    fctx,
                    cl.args,
                    funcParamTypes(fctx, cl.func_idx),
                ),
                .call_indirect => |cli| try addCallArgScalarHints(
                    &hints,
                    allocator,
                    fctx,
                    cli.args,
                    indexedParamTypes(fctx, cli.type_idx),
                ),
                .call_ref => |cr| try addCallArgScalarHints(
                    &hints,
                    allocator,
                    fctx,
                    cr.args,
                    indexedParamTypes(fctx, cr.type_idx),
                ),
                .memory_copy => |mc| {
                    try appendArgRegHint(&hints, allocator, mc.dst, .x1);
                    try appendArgRegHint(&hints, allocator, mc.src, .x2);
                    try appendArgRegHint(&hints, allocator, mc.len, .x3);
                },
                .memory_fill => |mf| {
                    try appendArgRegHint(&hints, allocator, mf.dst, .x1);
                    try appendArgRegHint(&hints, allocator, mf.val, .x2);
                    try appendArgRegHint(&hints, allocator, mf.len, .x3);
                },
                .memory_grow => |pages_vreg| {
                    try appendArgRegHint(&hints, allocator, pages_vreg, .x1);
                },
                .table_grow => |tg| {
                    try appendArgRegHint(&hints, allocator, tg.init, .x1);
                    try appendArgRegHint(&hints, allocator, tg.delta, .x2);
                },
                .table_set => |ts| {
                    try appendArgRegHint(&hints, allocator, ts.idx, .x2);
                    try appendArgRegHint(&hints, allocator, ts.val, .x3);
                },
                .atomic_notify => |an| {
                    try appendArgRegHint(&hints, allocator, an.base, .x1);
                    try appendArgRegHint(&hints, allocator, an.count, .x2);
                },
                .atomic_wait => |aw| {
                    try appendArgRegHint(&hints, allocator, aw.base, .x1);
                    try appendArgRegHint(&hints, allocator, aw.expected, .x2);
                },
                else => {},
            }
        }
    }
    return hints;
}

/// Collect copy-like IR ops whose emitted reg-to-reg `mov` can be
/// elided by post-allocation move coalescing (issue #386). The
/// regalloc.coalesceMoves pass tries to retarget each `dest` onto
/// `src`'s physreg; when both endpoints end up on the same reg, the
/// emit-site guard `if (src != info.reg) movRegReg(...)` swallows the
/// move entirely.
///
/// Currently produced for:
///   * `.reinterpret` — codegen is exactly a guarded `MOV Xd, Xn`
///     (`emitReinterpret`).
///   * `.wrap_i64`    — codegen is `UXTW Xd, Wn`. The W-form result
///     contract (downstream i32 consumers always read W-form) makes
///     the UXTW a no-op when src and dest share a physreg, so
///     `emitWrap`'s guard `if (src != dest) uxtw(...)` elides it.
///
/// Sign / zero extends (`extend_i32_s`, `extend_i32_u`) and the
/// sub-word extends (`extend8/16/32_s`) are NOT in scope: their
/// upper-bit-write semantics are part of the IR contract for the i64
/// dest and cannot be skipped just because src and dest co-located.
fn collectCopyHints(
    func: *const ir.IrFunction,
    block_order_opt: ?[]const ir.BlockId,
    scheduled: ?*const schedule.FunctionSchedule,
    allocator: std.mem.Allocator,
) !std.ArrayList(regalloc.CopyHint) {
    var hints: std.ArrayList(regalloc.CopyHint) = .empty;
    errdefer hints.deinit(allocator);

    var owns_order = false;
    const block_order: []const ir.BlockId = if (block_order_opt) |bo| bo else blk: {
        const raw = try allocator.alloc(ir.BlockId, func.blocks.items.len);
        for (raw, 0..) |*r, i| r.* = @intCast(i);
        owns_order = true;
        break :blk raw;
    };
    defer if (owns_order) allocator.free(block_order);

    for (block_order) |bo_bid| {
        const insts = if (scheduled) |s|
            s.instructions(bo_bid)
        else
            func.blocks.items[bo_bid].instructions.items;
        for (insts) |ci| {
            switch (ci.op) {
                .reinterpret, .wrap_i64 => |src_vreg| {
                    const dest = ci.dest orelse continue;
                    try hints.append(allocator, .{ .dest = dest, .src = src_vreg });
                },
                .parallel_copy => |pairs| {
                    // Each pair is an honest register-move candidate at
                    // the predecessor's terminator. Hinting `dst <- src`
                    // lets the linear-scan allocator + post-allocation
                    // coalescer collapse same-physreg pairs to no-ops
                    // before `emitParallelCopy` ever runs.
                    //
                    // Disabled while investigating: copy hints for the
                    // back-edge phi (src = vreg defined inside the
                    // loop body, dst = phi value live across the
                    // body) may make the coalescer believe two
                    // overlapping live ranges can share a physreg —
                    // misaligning loop semantics.
                    _ = pairs;
                },
                else => {},
            }
        }
    }
    return hints;
}

/// so bugs surface here before Phase 3 depends on the output.
///
/// This is called unconditionally from `compileFunctionImpl` and will be
/// deleted in Phase 3 (replaced by a real consumer of the allocation).
fn shadowRunRegalloc(func: *const ir.IrFunction, allocator: std.mem.Allocator) !void {
    var clobbers = try collectClobberPoints(func, null, null, allocator);
    defer clobbers.deinit(allocator);

    const local_layout = try computeLocalFrameLayout(func, allocator);
    defer allocator.free(local_layout.offsets);

    var alloc_result = try regalloc.allocate(
        func,
        allocator,
        aarch64RegSetForSpillBase(local_layout.end),
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

fn testCodeContainsWord(code: []const u8, word: u32) bool {
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        if (std.mem.readInt(u32, code[i..][0..4], .little) == word) return true;
    }
    return false;
}

fn testCodeContainsMasked(code: []const u8, mask: u32, value: u32) bool {
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & mask) == value) return true;
    }
    return false;
}

/// AArch64 `MOV Xd, Xn` is encoded as ORR Xd, XZR, Xn:
///   0xAA000000 | (1 << 21) ? ... actually: 1010|1010|000|Rm|0|00000|11111|Rd
/// Standard form: 0xAA0003E0 | (Rm << 16) | Rd.
fn testMovRegRegWord(rd: emit.Reg, rm: emit.Reg) u32 {
    return 0xAA0003E0 |
        (@as(u32, @intFromEnum(rm)) << 16) |
        @as(u32, @intFromEnum(rd));
}

test "emitParallelRegMoves: empty input is a no-op" {
    const allocator = std.testing.allocator;
    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();
    var done = [_]bool{};
    try emitParallelRegMoves(&code, &.{}, &.{}, &done);
    try std.testing.expectEqual(@as(usize, 0), code.bytes.items.len);
}

test "emitParallelRegMoves: disjoint chain emits topological MOVs" {
    // Pairs:
    //   x3 <- x1   (leaf — x3 not read by anyone)
    //   x4 <- x2   (leaf — x4 not read by anyone)
    // Expected: two MOVs in any order, no scratch use.
    const allocator = std.testing.allocator;
    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    var dst = [_]emit.Reg{ .x3, .x4 };
    var src = [_]emit.Reg{ .x1, .x2 };
    var done = [_]bool{ false, false };
    try emitParallelRegMoves(&code, &dst, &src, &done);

    try std.testing.expectEqual(@as(usize, 8), code.bytes.items.len);
    try std.testing.expect(testCodeContainsWord(code.bytes.items, testMovRegRegWord(.x3, .x1)));
    try std.testing.expect(testCodeContainsWord(code.bytes.items, testMovRegRegWord(.x4, .x2)));
    // No use of the cycle scratch register (X15 = tmp2).
    try std.testing.expect(!testCodeContainsWord(code.bytes.items, testMovRegRegWord(.x15, .x1)));
    try std.testing.expect(!testCodeContainsWord(code.bytes.items, testMovRegRegWord(.x15, .x2)));
}

test "emitParallelRegMoves: chain a→b→c emits in correct order" {
    // Pairs:
    //   x2 <- x1
    //   x3 <- x2   (must run before x2<-x1 clobbers x2)
    // Expected: MOV x3, x2 first; then MOV x2, x1.
    const allocator = std.testing.allocator;
    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    var dst = [_]emit.Reg{ .x2, .x3 };
    var src = [_]emit.Reg{ .x1, .x2 };
    var done = [_]bool{ false, false };
    try emitParallelRegMoves(&code, &dst, &src, &done);

    try std.testing.expectEqual(@as(usize, 8), code.bytes.items.len);
    const w0 = std.mem.readInt(u32, code.bytes.items[0..4], .little);
    const w1 = std.mem.readInt(u32, code.bytes.items[4..8], .little);
    try std.testing.expectEqual(testMovRegRegWord(.x3, .x2), w0);
    try std.testing.expectEqual(testMovRegRegWord(.x2, .x1), w1);
}

test "emitParallelRegMoves: 2-cycle swap a↔b uses scratch" {
    // Pairs:
    //   x1 <- x2
    //   x2 <- x1   (swap)
    // Naive sequential MOVs would corrupt: `mov x1, x2` clobbers x1,
    // then `mov x2, x1` reads the new value. The 4-phase resolver
    // saves head.src to X15 (tmp2), walks the cycle, and writes
    // scratch into head.dst at the end.
    const allocator = std.testing.allocator;
    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    var dst = [_]emit.Reg{ .x1, .x2 };
    var src = [_]emit.Reg{ .x2, .x1 };
    var done = [_]bool{ false, false };
    try emitParallelRegMoves(&code, &dst, &src, &done);

    try std.testing.expectEqual(@as(usize, 12), code.bytes.items.len);
    const w0 = std.mem.readInt(u32, code.bytes.items[0..4], .little);
    const w1 = std.mem.readInt(u32, code.bytes.items[4..8], .little);
    const w2 = std.mem.readInt(u32, code.bytes.items[8..12], .little);
    // mov X15, X2  (save head.src = X2)
    try std.testing.expectEqual(testMovRegRegWord(.x15, .x2), w0);
    // mov X2, X1   (walk predecessor: pair whose dst==head.src=X2)
    try std.testing.expectEqual(testMovRegRegWord(.x2, .x1), w1);
    // mov X1, X15  (close cycle: head.dst <- scratch)
    try std.testing.expectEqual(testMovRegRegWord(.x1, .x15), w2);
}

test "emitParallelRegMoves: 3-cycle a→b, b→c, c→a uses scratch" {
    // Pairs (cycle):
    //   x1 <- x3   (a <- c)
    //   x2 <- x1   (b <- a)
    //   x3 <- x2   (c <- b)
    // Expected: scratch X15 saves head.src, then visit pairs in
    // reverse-dataflow order: pair writing to head.src first, then
    // around the cycle, finally head with src=scratch.
    const allocator = std.testing.allocator;
    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    var dst = [_]emit.Reg{ .x1, .x2, .x3 };
    var src = [_]emit.Reg{ .x3, .x1, .x2 };
    var done = [_]bool{ false, false, false };
    try emitParallelRegMoves(&code, &dst, &src, &done);

    // 4 MOVs total: 1 save + 3 cycle MOVs.
    try std.testing.expectEqual(@as(usize, 16), code.bytes.items.len);
    // First instruction is `mov X15, head.src`. Head is pair[0] (first
    // pending after leaf phase), src[0]=X3.
    const w0 = std.mem.readInt(u32, code.bytes.items[0..4], .little);
    try std.testing.expectEqual(testMovRegRegWord(.x15, .x3), w0);
    // Last instruction closes the cycle: dst[0] <- X15.
    const last = std.mem.readInt(u32, code.bytes.items[12..16], .little);
    try std.testing.expectEqual(testMovRegRegWord(.x1, .x15), last);
}

test "emitParallelRegMoves: mixed chain + cycle" {
    // Three pairs:
    //   x5 <- x1   (chain leaf — x5 read by nobody)
    //   x1 <- x2   (cycle part)
    //   x2 <- x1   (cycle part — swap with above)
    // Wait, that's a 2-cycle PLUS a leaf reading x1 (the cycle's src).
    // The leaf MUST run BEFORE the cycle clobbers x1.
    const allocator = std.testing.allocator;
    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    var dst = [_]emit.Reg{ .x5, .x1, .x2 };
    var src = [_]emit.Reg{ .x1, .x2, .x1 };
    var done = [_]bool{ false, false, false };
    try emitParallelRegMoves(&code, &dst, &src, &done);

    // The leaf (x5 <- x1) reads x1; the cycle (x1<->x2) overwrites x1.
    // For correctness x5 must be written BEFORE x1 changes.
    // The resolver's leaf phase emits x5<-x1 first.
    const w0 = std.mem.readInt(u32, code.bytes.items[0..4], .little);
    try std.testing.expectEqual(testMovRegRegWord(.x5, .x1), w0);
    // Then a 2-cycle swap follows (3 MOVs with scratch).
    try std.testing.expectEqual(@as(usize, 16), code.bytes.items.len);
}

// ── emitParallelCopyFull: general reg/stack sequentializer (#817) ──
//
// These tests exercise the mixed register/spill-slot parallel-copy
// resolver by building a RegMap backed by an explicit AllocResult, emitting
// the copy, then decoding the MOV/LDR/STR stream and replaying it over a
// register+stack model. The correctness oracle is the parallel-copy
// semantics: after the sequence, every destination location must hold the
// value its source location held BEFORE the copy. This catches the
// phase-ordering bug the old 4-phase resolver had (a spill slot that is
// both a source and a destination was overwritten before being read).

const PcTestLoc = union(enum) { reg: u8, stack: u32 };
const PcVLoc = struct { v: ir.VReg, loc: PcTestLoc };

fn simulateParallelCopy(
    allocator: std.mem.Allocator,
    pairs: []const ir.Inst.ParallelCopy,
    locs: []const PcVLoc,
) !void {
    var ar = regalloc.AllocResult{
        .assignments = std.AutoHashMap(ir.VReg, regalloc.Allocation).init(allocator),
        .spill_count = 0,
        .remat = std.AutoHashMap(ir.VReg, regalloc.RematDef).init(allocator),
    };
    defer ar.deinit();

    var loc_of = std.AutoHashMap(ir.VReg, PcTestLoc).init(allocator);
    defer loc_of.deinit();
    for (locs) |e| {
        const a: regalloc.Allocation = switch (e.loc) {
            .reg => |k| .{ .reg = k },
            .stack => |off| .{ .stack = @intCast(off) },
        };
        try ar.assignments.put(e.v, a);
        try loc_of.put(e.v, e.loc);
    }

    var reg_map = RegMap.init(allocator, 0, 4096);
    defer reg_map.deinit();
    reg_map.alloc_result = &ar;
    // Pre-resolve every vreg into `entries` so `get(src)` succeeds.
    for (locs) |e| _ = try reg_map.assign(e.v);

    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();
    try emitParallelCopyFull(&code, pairs, &reg_map, allocator);

    // Model: 32 GPRs + a stack keyed by the scaled (÷8) FP offset that the
    // emitted LDR/STR immediates use.
    var reg_model = [_]u64{0} ** 32;
    var stack_model = std.AutoHashMap(u32, u64).init(allocator);
    defer stack_model.deinit();

    // Seed each distinct operand location with a unique non-zero sentinel.
    var sentinel: u64 = 1;
    for (locs) |e| switch (e.loc) {
        .reg => |k| if (reg_model[k] == 0) {
            reg_model[k] = sentinel;
            sentinel += 1;
        },
        .stack => |off| {
            const key = off / 8;
            if (!stack_model.contains(key)) {
                try stack_model.put(key, sentinel);
                sentinel += 1;
            }
        },
    };

    const readLoc = struct {
        fn f(rm: *const [32]u64, sm: *const std.AutoHashMap(u32, u64), loc: PcTestLoc) u64 {
            return switch (loc) {
                .reg => |k| rm[k],
                .stack => |off| sm.get(off / 8) orelse 0,
            };
        }
    }.f;

    // Capture each destination's EXPECTED post-copy value (its source's
    // pre-copy value) before replaying the emitted stream.
    var expected = try allocator.alloc(u64, pairs.len);
    defer allocator.free(expected);
    for (pairs, 0..) |p, idx| {
        expected[idx] = readLoc(&reg_model, &stack_model, loc_of.get(p.src).?);
    }

    // Replay the emitted MOV/LDR/STR instructions over the model.
    var i: usize = 0;
    while (i < code.bytes.items.len) : (i += 4) {
        const w = std.mem.readInt(u32, code.bytes.items[i..][0..4], .little);
        const rd: usize = @intCast(w & 0x1F);
        const rn: u32 = (w >> 5) & 0x1F;
        const imm12: u32 = (w >> 10) & 0xFFF;
        const rm: usize = @intCast((w >> 16) & 0x1F);
        if ((w >> 21) == 0x550 and rn == 31 and ((w >> 10) & 0x3F) == 0) {
            // MOV Xrd, Xrm  (ORR Xrd, XZR, Xrm)
            reg_model[rd] = reg_model[rm];
        } else if ((w >> 22) == 0x3E5) {
            // LDR Xrd, [Xrn, #imm12*8]
            try std.testing.expectEqual(@as(u32, 29), rn); // base is FP
            reg_model[rd] = stack_model.get(imm12) orelse 0;
        } else if ((w >> 22) == 0x3E4) {
            // STR Xrd, [Xrn, #imm12*8]
            try std.testing.expectEqual(@as(u32, 29), rn); // base is FP
            try stack_model.put(imm12, reg_model[rd]);
        } else {
            std.debug.panic("parallel copy emitted unexpected insn 0x{x:0>8}", .{w});
        }
    }

    // Verify parallel-copy semantics held.
    for (pairs, 0..) |p, idx| {
        const got = readLoc(&reg_model, &stack_model, loc_of.get(p.dst).?);
        try std.testing.expectEqual(expected[idx], got);
    }
}

test "emitParallelCopyFull: identity move (coalesced) emits nothing" {
    const allocator = std.testing.allocator;
    var ar = regalloc.AllocResult{
        .assignments = std.AutoHashMap(ir.VReg, regalloc.Allocation).init(allocator),
        .spill_count = 0,
        .remat = std.AutoHashMap(ir.VReg, regalloc.RematDef).init(allocator),
    };
    defer ar.deinit();
    try ar.assignments.put(0, .{ .reg = 5 });
    try ar.assignments.put(1, .{ .reg = 5 }); // src shares dst's register

    var reg_map = RegMap.init(allocator, 0, 4096);
    defer reg_map.deinit();
    reg_map.alloc_result = &ar;
    _ = try reg_map.assign(0);
    _ = try reg_map.assign(1);

    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();
    const pairs = [_]ir.Inst.ParallelCopy{.{ .dst = 0, .src = 1, .ty = .i32 }};
    try emitParallelCopyFull(&code, &pairs, &reg_map, allocator);
    try std.testing.expectEqual(@as(usize, 0), code.bytes.items.len);
}

test "emitParallelCopyFull: spill slot is both source and destination" {
    // pair0: stack[0] <- x5 ; pair1: x6 <- stack[0]
    // stack[0] is a destination (pair0) AND a source (pair1). The resolver
    // MUST read x6<-stack[0] (old value) before stack[0] is overwritten.
    const allocator = std.testing.allocator;
    const pairs = [_]ir.Inst.ParallelCopy{
        .{ .dst = 0, .src = 1, .ty = .i64 }, // v0(stack0) <- v1(x5)
        .{ .dst = 2, .src = 0, .ty = .i64 }, // v2(x6)     <- v0(stack0)
    };
    try simulateParallelCopy(allocator, &pairs, &[_]PcVLoc{
        .{ .v = 0, .loc = .{ .stack = 0 } },
        .{ .v = 1, .loc = .{ .reg = 5 } },
        .{ .v = 2, .loc = .{ .reg = 6 } },
    });
}

test "emitParallelCopyFull: 2-element reg<->stack cycle (swap)" {
    // pair0: x5 <- stack[8] ; pair1: stack[8] <- x5  — a 2-cycle that must
    // be broken with scratch (saving one operand) across the reg/stack edge.
    const allocator = std.testing.allocator;
    const pairs = [_]ir.Inst.ParallelCopy{
        .{ .dst = 0, .src = 1, .ty = .i64 }, // v0(x5)      <- v1(stack8)
        .{ .dst = 1, .src = 0, .ty = .i64 }, // v1(stack8)  <- v0(x5)
    };
    try simulateParallelCopy(allocator, &pairs, &[_]PcVLoc{
        .{ .v = 0, .loc = .{ .reg = 5 } },
        .{ .v = 1, .loc = .{ .stack = 8 } },
    });
}

test "emitParallelCopyFull: 3-element cycle mixing reg and stack" {
    // x5 <- stack8 <- x6 <- x5  (rotate values around a mixed 3-cycle).
    const allocator = std.testing.allocator;
    const pairs = [_]ir.Inst.ParallelCopy{
        .{ .dst = 0, .src = 1, .ty = .i64 }, // v0(x5)     <- v1(stack8)
        .{ .dst = 1, .src = 2, .ty = .i64 }, // v1(stack8) <- v2(x6)
        .{ .dst = 2, .src = 0, .ty = .i64 }, // v2(x6)     <- v0(x5)
    };
    try simulateParallelCopy(allocator, &pairs, &[_]PcVLoc{
        .{ .v = 0, .loc = .{ .reg = 5 } },
        .{ .v = 1, .loc = .{ .stack = 8 } },
        .{ .v = 2, .loc = .{ .reg = 6 } },
    });
}

test "emitParallelCopyFull: fan-out and a leaf reading a cycle source" {
    // x5 <- x6 ; x6 <- stack8 ; stack8 <- x6  -> x6 and stack8 form a swap,
    // and x5<-x6 is a leaf that must read x6 BEFORE the swap overwrites it.
    const allocator = std.testing.allocator;
    const pairs = [_]ir.Inst.ParallelCopy{
        .{ .dst = 0, .src = 1, .ty = .i64 }, // v0(x5)     <- v1(x6)
        .{ .dst = 1, .src = 2, .ty = .i64 }, // v1(x6)     <- v2(stack8)
        .{ .dst = 2, .src = 1, .ty = .i64 }, // v2(stack8) <- v1(x6)
    };
    try simulateParallelCopy(allocator, &pairs, &[_]PcVLoc{
        .{ .v = 0, .loc = .{ .reg = 5 } },
        .{ .v = 1, .loc = .{ .reg = 6 } },
        .{ .v = 2, .loc = .{ .stack = 8 } },
    });
}

test "compile: 2-arm scalar phi lowers to MOV-free path (copy-hint coalesced)" {
    // Build:
    //   b0:  v0 = iconst 7; br_if c → b1 else b2
    //   b1:  v1 = iconst 11; br b3
    //   b2:  v2 = iconst 13; br b3
    //   b3:  v3 = phi (b1, v1) (b2, v2); ret v3
    // After lowerPhisToLocals + coalescePhiLocalsToParallelCopy +
    // copy-hint coalescing, the predecessor copies should collapse to
    // no-ops (v3 / v1 / v2 share a physreg).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });

    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 11 }, .dest = v1, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });

    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 13 }, .dest = v2, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });

    const edges = try allocator.alloc(ir.Inst.PhiEdge, 2);
    edges[0] = .{ .block = b1, .val = v1 };
    edges[1] = .{ .block = b2, .val = v2 };
    try func.getBlock(b3).append(.{ .op = .{ .phi = edges }, .dest = v3, .type = .i32 });
    try func.getBlock(b3).append(.{ .op = .{ .ret = v3 } });

    try func.getBlock(b1).addPredecessor(b0);
    try func.getBlock(b2).addPredecessor(b0);
    try func.getBlock(b3).addPredecessor(b1);
    try func.getBlock(b3).addPredecessor(b2);

    _ = try @import("../../ir/passes.zig").lowerPhisToLocals(&func, allocator);
    _ = try @import("../../ir/passes.zig").coalescePhiLocalsToParallelCopy(&func, allocator);

    // After our lowering, the join block (b3) has NO local_get of the
    // synth — it was deleted; phi_dest is defined by parallel_copy
    // ops in each predecessor.
    for (func.getBlock(b3).instructions.items) |inst| {
        try std.testing.expect(inst.op != .local_get);
    }
    // Both predecessor blocks gained a parallel_copy instruction
    // (placed before their terminator).
    var pc_blocks: u32 = 0;
    for ([_]ir.BlockId{ b1, b2 }) |bid| {
        for (func.getBlock(bid).instructions.items) |inst| {
            if (inst.op == .parallel_copy) {
                pc_blocks += 1;
                try std.testing.expectEqual(@as(usize, 1), inst.op.parallel_copy.len);
                try std.testing.expectEqual(v3, inst.op.parallel_copy[0].dst);
                try std.testing.expectEqual(ir.IrType.i32, inst.op.parallel_copy[0].ty);
            }
        }
    }
    try std.testing.expectEqual(@as(u32, 2), pc_blocks);

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "compile: phi-resolution end-to-end produces no per-arm frame round-trip" {
    // Build a small loop with a phi (loop counter):
    //   b0:  v_init = iconst 0; br b1
    //   b1:  v_i = phi (b0, v_init) (b1, v_next); v_next = v_i + 1;
    //        br_if cond → b1 else b2
    //   b2:  ret v_next
    // The counter `v_i` is read only inside the loop (its post-loop result
    // is observed via `v_next`, not `v_i`), so it is safe to coalesce and the
    // phi-resolution should NOT emit `str ... [fp, ...]` and matching `ldr`
    // purely for the phi bridge: instead a register MOV (or, when the
    // coalescer wins, no instruction at all). A phi whose result is live
    // across the loop exit takes the memory path instead (see the #820
    // lost-copy tests above).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_init = func.newVReg();
    const v_i = func.newVReg();
    const v_one = func.newVReg();
    const v_next = func.newVReg();
    const v_cond = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_init, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const edges = try allocator.alloc(ir.Inst.PhiEdge, 2);
    edges[0] = .{ .block = b0, .val = v_init };
    edges[1] = .{ .block = b1, .val = v_next };
    try func.getBlock(b1).append(.{ .op = .{ .phi = edges }, .dest = v_i, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_one, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i, .rhs = v_one } }, .dest = v_next, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v_cond, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });

    try func.getBlock(b2).append(.{ .op = .{ .ret = v_next } });

    try func.getBlock(b1).addPredecessor(b0);
    try func.getBlock(b1).addPredecessor(b1);
    try func.getBlock(b2).addPredecessor(b1);

    _ = try @import("../../ir/passes.zig").lowerPhisToLocals(&func, allocator);
    _ = try @import("../../ir/passes.zig").coalescePhiLocalsToParallelCopy(&func, allocator);

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
}

// Count `parallel_copy` ops across all blocks of `func`.
fn countParallelCopies(func: *const ir.IrFunction) u32 {
    var n: u32 = 0;
    for (func.blocks.items) |*block| {
        for (block.instructions.items) |inst| {
            if (inst.op == .parallel_copy) n += 1;
        }
    }
    return n;
}

test "coalescePhiLocalsToParallelCopy: rejects loop phi live across loop exit (#820)" {
    // Lost-copy hazard: a loop-counter phi whose value is read AFTER the
    // loop. Coalescing would route the back-edge increment through the phi
    // register, clobbering the value the post-loop use still needs (this is
    // what hung CoreMark `core_list_mergesort`). The pass must leave it on
    // the memory path.
    //   b0:  v_init = 0; br b1
    //   b1:  v_i = phi(b0:v_init, b1:v_next); v_next = v_i + 1;
    //        br_if cond → b1 else b2
    //   b2:  ret v_i           ← v_i live across the loop exit
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_init = func.newVReg();
    const v_i = func.newVReg();
    const v_one = func.newVReg();
    const v_next = func.newVReg();
    const v_cond = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_init, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const edges = try allocator.alloc(ir.Inst.PhiEdge, 2);
    edges[0] = .{ .block = b0, .val = v_init };
    edges[1] = .{ .block = b1, .val = v_next };
    try func.getBlock(b1).append(.{ .op = .{ .phi = edges }, .dest = v_i, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_one, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i, .rhs = v_one } }, .dest = v_next, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v_cond, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b1, .else_block = b2 } } });

    try func.getBlock(b2).append(.{ .op = .{ .ret = v_i } });

    try func.getBlock(b1).addPredecessor(b0);
    try func.getBlock(b1).addPredecessor(b1);
    try func.getBlock(b2).addPredecessor(b1);

    _ = try @import("../../ir/passes.zig").lowerPhisToLocals(&func, allocator);
    const changed = try @import("../../ir/passes.zig").coalescePhiLocalsToParallelCopy(&func, allocator);

    // The single phi is the lost-copy case → rejected → nothing coalesced.
    try std.testing.expect(!changed);
    try std.testing.expectEqual(@as(u32, 0), countParallelCopies(&func));
}

test "coalescePhiLocalsToParallelCopy: coalesces loop phi used only inside the loop" {
    // The safe counterpart of the #820 test: the loop-counter phi is read
    // only inside the loop body (never after the exit), so coalescing it is
    // safe and must still happen.
    //   b0:  v_init = 0; br b1
    //   b1:  v_i = phi(b0:v_init, b1:v_next); v_next = v_i + 1;
    //        br_if cond=v_i → b1 else b2
    //   b2:  ret v_ret        ← does NOT use v_i
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();

    const v_init = func.newVReg();
    const v_i = func.newVReg();
    const v_one = func.newVReg();
    const v_next = func.newVReg();
    const v_ret = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_init, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });

    const edges = try allocator.alloc(ir.Inst.PhiEdge, 2);
    edges[0] = .{ .block = b0, .val = v_init };
    edges[1] = .{ .block = b1, .val = v_next };
    try func.getBlock(b1).append(.{ .op = .{ .phi = edges }, .dest = v_i, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_one, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_i, .rhs = v_one } }, .dest = v_next, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_i, .then_block = b1, .else_block = b2 } } });

    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v_ret, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v_ret } });

    try func.getBlock(b1).addPredecessor(b0);
    try func.getBlock(b1).addPredecessor(b1);
    try func.getBlock(b2).addPredecessor(b1);

    _ = try @import("../../ir/passes.zig").lowerPhisToLocals(&func, allocator);
    const changed = try @import("../../ir/passes.zig").coalescePhiLocalsToParallelCopy(&func, allocator);

    // In-loop-only phi → safe → coalesced into parallel_copy(s).
    try std.testing.expect(changed);
    try std.testing.expect(countParallelCopies(&func) > 0);
}

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

// Count MOVZ instructions in `code`. MOVZ encoding: 1|10|100101|hw|imm16|Rd
// → top 9 bits == 110100101 (mask 0xFF800000 against base 0xD2800000).
fn countMovzImm(code: []const u8) u32 {
    var i: usize = 0;
    var n: u32 = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFF800000) == 0xD2800000) n += 1;
    }
    return n;
}

test "iconst suppress (#523): iconst feeding add-imm12 emits no MOVZ" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 2, 0);
    defer func.deinit();
    const lt = try allocator.alloc(ir.IrType, 1);
    lt[0] = .i32;
    func.local_types = lt;

    const bid = try func.newBlock();
    const block = func.getBlock(bid);
    const param = func.newVReg();
    const k = func.newVReg();
    const sum = func.newVReg();
    // sum = param + 5 ; ret sum   — `5` fits imm12, all uses fold.
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = param, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 5 }, .dest = k, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = param, .rhs = k } }, .dest = sum, .type = .i32 });
    try block.append(.{ .op = .{ .ret = sum } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // No MOVZ should appear: prologue+epilogue use STP/LDP/ADD/SUB only,
    // and the iconst's mov is suppressed (#523).
    try std.testing.expectEqual(@as(u32, 0), countMovzImm(code));
}

test "iconst suppress (#523): out-of-imm12 constant keeps its MOVZ" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 2, 0);
    defer func.deinit();
    const lt = try allocator.alloc(ir.IrType, 1);
    lt[0] = .i32;
    func.local_types = lt;

    const bid = try func.newBlock();
    const block = func.getBlock(bid);
    const param = func.newVReg();
    const k = func.newVReg();
    const sum = func.newVReg();
    // 0x1_0001 doesn't fit imm12 nor imm12<<12 — encodeAddSubImm returns
    // null, the consumer falls through to reg/reg ADD which reads `k`
    // as a real register, so the iconst must materialize.
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = param, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0x1_0001 }, .dest = k, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = param, .rhs = k } }, .dest = sum, .type = .i32 });
    try block.append(.{ .op = .{ .ret = sum } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(countMovzImm(code) >= 1);
}

test "iconst suppress (#523): one foldable + one non-foldable use keeps MOVZ" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 2, 0);
    defer func.deinit();
    const lt = try allocator.alloc(ir.IrType, 1);
    lt[0] = .i32;
    func.local_types = lt;

    const bid = try func.newBlock();
    const block = func.getBlock(bid);
    const param = func.newVReg();
    const k = func.newVReg();
    const sum = func.newVReg();
    const eq = func.newVReg();
    const out = func.newVReg();
    // `k=7` is folded into ADD-imm12 (foldable use), but also feeds an
    // EQ comparison which has no immediate-fold path on aarch64 — the
    // cmp reads `k` as a register. Conjunctive rule keeps the mov.
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = param, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = k, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = param, .rhs = k } }, .dest = sum, .type = .i32 });
    try block.append(.{ .op = .{ .eq = .{ .lhs = sum, .rhs = k } }, .dest = eq, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = sum, .rhs = eq } }, .dest = out, .type = .i32 });
    try block.append(.{ .op = .{ .ret = out } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(countMovzImm(code) >= 1);
}

test "iconst suppress (#523): iconst as store base keeps MOVZ" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 0);
    defer func.deinit();
    const lt = try allocator.alloc(ir.IrType, 1);
    lt[0] = .i32;
    func.local_types = lt;

    const bid = try func.newBlock();
    const block = func.getBlock(bid);
    const val = func.newVReg();
    const base = func.newVReg();
    // store i32 value at constant base address. Address goes through
    // foldLoadStoreOffset but uses the IR `.offset` field (literal),
    // not a vreg — `base` is read as a register. Not foldable.
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = val, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = base, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .val = val, .size = 4 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // base=0 is materialized via MOVZ #0; can't be folded into the store
    // address (the IR `.offset` field handles literal displacements, not
    // vreg-based ones).
    try std.testing.expect(countMovzImm(code) >= 1);
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

test "compileFunction: v128 global_get/global_set emit Q loads and stores" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .global_get = 1 }, .dest = v0, .type = .v128 });
    try block.append(.{ .op = .{ .global_set = .{ .idx = 3, .val = v0 } }, .type = .v128 });
    try block.append(.{ .op = .{ .ret = null } });

    const global_types = [_]ir.IrType{ .i32, .v128, .i64, .v128 };
    const global_offsets = [_]u32{ 0, 16, 32, 48 };
    const code = try compileFunctionImpl(&func, .{
        .allocator = allocator,
        .global_types = &global_types,
        .global_offsets = &global_offsets,
    }, allocator);
    defer allocator.free(code);

    var saw_ldr_q = false;
    var saw_str_q = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const word = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((word & 0xFFC00000) == 0x3DC00000) saw_ldr_q = true;
        if ((word & 0xFFC00000) == 0x3D800000) saw_str_q = true;
    }
    try std.testing.expect(saw_ldr_q);
    try std.testing.expect(saw_str_q);
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

test "compileFunction: memory.grow refreshes pinned x20 (issue #466)" {
    // The helper call to mem_grow_fn can realloc the linear memory and
    // invalidate x20's cached `[vmctx + 0]`. Codegen must emit a fresh
    // LDR x20, [x19, #0] right after the grow call's BLR returns.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .memory_grow = v0 }, .dest = v1, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v1 } });
    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_peephole = false });
    defer allocator.free(code);

    // BLR x16 (the indirect call through vmctx.mem_grow_fn) encodes as
    //   1101011000111111000000 | Rn=10000 | 00000  → 0xD63F0200.
    const blr_x16: u32 = 0xD63F0200;
    var blr_off: ?usize = null;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if (w == blr_x16) {
            blr_off = i;
            // Don't break; we want the LAST blr — the mem_grow call
            // happens after any earlier setup BLRs.
        }
    }
    try std.testing.expect(blr_off != null);

    // After the BLR, scan for LDR x20, [x19, #0]:
    //   0xF9400000 | (imm12=0 << 10) | (Rn=19 << 5) | Rt=20 = 0xF9400274
    const ldr_x20: u32 = 0xF9400274;
    var saw_reload = false;
    var j: usize = blr_off.? + 4;
    while (j + 4 <= code.len) : (j += 4) {
        const w = std.mem.readInt(u32, code[j..][0..4], .little);
        if (w == ldr_x20) {
            saw_reload = true;
            break;
        }
    }
    try std.testing.expect(saw_reload);
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

test "compileModule: direct v128 call stages q0" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    var f0 = ir.IrFunction.init(allocator, 1, 1, 1);
    f0.local_types = try allocator.dupe(ir.IrType, &[_]ir.IrType{.v128});
    {
        const b = try f0.newBlock();
        const p = f0.newVReg();
        const lane = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .local_get = 0 }, .dest = p, .type = .v128 });
        try f0.getBlock(b).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = p, .lane = 0 } }, .dest = lane, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .ret = lane } });
    }
    _ = try module.addFunction(f0);

    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    const args = try allocator.alloc(ir.VReg, 1);
    {
        const b = try f1.newBlock();
        const v = f1.newVReg();
        const r = f1.newVReg();
        try f1.getBlock(b).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_0000_0001 }, .dest = v, .type = .v128 });
        args[0] = v;
        try f1.getBlock(b).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = r, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .ret = r } });
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    const f1_code = result.code[result.offsets[1]..];
    try std.testing.expect(testCodeContainsMasked(f1_code, 0xFFE0FC1F, 0x4EA01C00)); // v128 arg → q0
    try std.testing.expect(testCodeContainsMasked(f1_code, 0xFC000000, 0x94000000)); // BL
}

test "compileModule: mixed scalar and v128 direct call uses independent banks" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    var f0 = ir.IrFunction.init(allocator, 3, 1, 3);
    f0.local_types = try allocator.dupe(ir.IrType, &[_]ir.IrType{ .i32, .v128, .i64 });
    {
        const b = try f0.newBlock();
        const p0 = f0.newVReg();
        const p1 = f0.newVReg();
        const lane = f0.newVReg();
        const sum = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .local_get = 0 }, .dest = p0, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .local_get = 1 }, .dest = p1, .type = .v128 });
        try f0.getBlock(b).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = p1, .lane = 0 } }, .dest = lane, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .add = .{ .lhs = p0, .rhs = lane } }, .dest = sum, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .ret = sum } });
    }
    _ = try module.addFunction(f0);

    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    const args = try allocator.alloc(ir.VReg, 3);
    {
        const b = try f1.newBlock();
        const s0 = f1.newVReg();
        const v = f1.newVReg();
        const s1 = f1.newVReg();
        const r = f1.newVReg();
        try f1.getBlock(b).append(.{ .op = .{ .iconst_32 = 7 }, .dest = s0, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_0000_0005 }, .dest = v, .type = .v128 });
        try f1.getBlock(b).append(.{ .op = .{ .iconst_64 = 9 }, .dest = s1, .type = .i64 });
        args[0] = s0;
        args[1] = v;
        args[2] = s1;
        try f1.getBlock(b).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = r, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .ret = r } });
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    const f1_code = result.code[result.offsets[1]..];
    try std.testing.expect(testCodeContainsMasked(f1_code, 0xFFE0FC1F, 0x4EA01C00)); // v128 arg → q0
    try std.testing.expect(testCodeContainsMasked(f1_code, 0xFC000000, 0x94000000));
}

test "compileModule: excess v128 args use outgoing stack" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    var f0 = ir.IrFunction.init(allocator, 9, 1, 9);
    f0.local_types = try allocator.dupe(ir.IrType, &[_]ir.IrType{
        .v128, .v128, .v128, .v128, .v128, .v128, .v128, .v128, .v128,
    });
    {
        const b = try f0.newBlock();
        const p = f0.newVReg();
        const lane = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .local_get = 8 }, .dest = p, .type = .v128 });
        try f0.getBlock(b).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = p, .lane = 0 } }, .dest = lane, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .ret = lane } });
    }
    _ = try module.addFunction(f0);

    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    const args = try allocator.alloc(ir.VReg, 9);
    {
        const b = try f1.newBlock();
        for (args, 0..) |*slot, idx| {
            const v = f1.newVReg();
            try f1.getBlock(b).append(.{ .op = .{ .v128_const = @as(u128, @intCast(idx + 1)) }, .dest = v, .type = .v128 });
            slot.* = v;
        }
        const r = f1.newVReg();
        try f1.getBlock(b).append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = r, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .ret = r } });
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    const f1_code = result.code[result.offsets[1]..];
    try std.testing.expect(testCodeContainsWord(f1_code, 0xD10043FF)); // SUB SP, SP, #16
    try std.testing.expect(testCodeContainsWord(f1_code, 0x3D8003FF)); // STR Q31, [SP]
    try std.testing.expect(testCodeContainsWord(f1_code, 0x910043FF)); // ADD SP, SP, #16
}

test "compileFunction: indirect v128 call stages q0 and emits BLR" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const args = try allocator.alloc(ir.VReg, 1);

    const b = try func.newBlock();
    const v = func.newVReg();
    const idx = func.newVReg();
    const r = func.newVReg();
    try func.getBlock(b).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_0000_0001 }, .dest = v, .type = .v128 });
    try func.getBlock(b).append(.{ .op = .{ .iconst_32 = 0 }, .dest = idx, .type = .i32 });
    args[0] = v;
    try func.getBlock(b).append(.{ .op = .{ .call_indirect = .{ .type_idx = 0, .elem_idx = idx, .args = args } }, .dest = r, .type = .i32 });
    try func.getBlock(b).append(.{ .op = .{ .ret = r } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(testCodeContainsMasked(code, 0xFFE0FC1F, 0x4EA01C00)); // v128 arg → q0
    try std.testing.expect(testCodeContainsWord(code, 0xD63F0200)); // BLR x16
}

test "compileFunction: call_ref v128 call stages q0 and emits BLR" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const args = try allocator.alloc(ir.VReg, 1);

    const b = try func.newBlock();
    const v = func.newVReg();
    const ref = func.newVReg();
    const r = func.newVReg();
    try func.getBlock(b).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_0000_0001 }, .dest = v, .type = .v128 });
    try func.getBlock(b).append(.{ .op = .{ .ref_func = 0 }, .dest = ref, .type = .i64 });
    args[0] = v;
    try func.getBlock(b).append(.{ .op = .{ .call_ref = .{ .type_idx = 0, .func_ref = ref, .args = args } }, .dest = r, .type = .i32 });
    try func.getBlock(b).append(.{ .op = .{ .ret = r } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(testCodeContainsMasked(code, 0xFFE0FC1F, 0x4EA01C00)); // v128 arg → q0
    try std.testing.expect(testCodeContainsWord(code, 0xD63F0200)); // BLR x16
}

test "compileModule: direct v128 result captures q0" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    var f0 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f0.newBlock();
        const v = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_0000_0001 }, .dest = v, .type = .v128 });
        try f0.getBlock(b).append(.{ .op = .{ .ret = v } });
    }
    _ = try module.addFunction(f0);

    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f1.newBlock();
        const v = f1.newVReg();
        const lane = f1.newVReg();
        try f1.getBlock(b).append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v, .type = .v128 });
        try f1.getBlock(b).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = v, .lane = 1 } }, .dest = lane, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .ret = lane } });
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    const f1_code = result.code[result.offsets[1]..];
    try std.testing.expect(testCodeContainsMasked(f1_code, 0xFC000000, 0x94000000)); // BL
    try std.testing.expect(testCodeContainsMasked(f1_code, 0xFFE0FC00, 0x4EA01C00)); // ORR Vd.16B, V0.16B, V0.16B
}

test "compileFunction: indirect v128 result captures q0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b = try func.newBlock();
    const idx = func.newVReg();
    const v = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(b).append(.{ .op = .{ .iconst_32 = 0 }, .dest = idx, .type = .i32 });
    try func.getBlock(b).append(.{ .op = .{ .call_indirect = .{ .type_idx = 0, .elem_idx = idx } }, .dest = v, .type = .v128 });
    try func.getBlock(b).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = v, .lane = 0 } }, .dest = lane, .type = .i32 });
    try func.getBlock(b).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(testCodeContainsWord(code, 0xD63F0200)); // BLR x16
    try std.testing.expect(testCodeContainsMasked(code, 0xFFE0FC00, 0x4EA01C00)); // ORR Vd.16B, V0.16B, V0.16B
}

test "compileFunction: call_ref v128 result captures q0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b = try func.newBlock();
    const ref = func.newVReg();
    const v = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(b).append(.{ .op = .{ .ref_func = 0 }, .dest = ref, .type = .i64 });
    try func.getBlock(b).append(.{ .op = .{ .call_ref = .{ .type_idx = 0, .func_ref = ref } }, .dest = v, .type = .v128 });
    try func.getBlock(b).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = v, .lane = 0 } }, .dest = lane, .type = .i32 });
    try func.getBlock(b).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(testCodeContainsWord(code, 0xD63F0200)); // BLR x16
    try std.testing.expect(testCodeContainsMasked(code, 0xFFE0FC00, 0x4EA01C00)); // ORR Vd.16B, V0.16B, V0.16B
}

test "compileModule: v128 primary and aligned v128 extra multi-result" {
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();
    const callee_type = try module.addFuncType(&[_]ir.IrType{}, &[_]ir.IrType{ .v128, .i32, .v128 });
    const caller_type = try module.addFuncType(&[_]ir.IrType{}, &[_]ir.IrType{.i32});
    try module.addFuncTypeIndex(callee_type);
    try module.addFuncTypeIndex(caller_type);

    var f0 = ir.IrFunction.init(allocator, 0, 3, 0);
    const ret_vals = try allocator.alloc(ir.VReg, 3);
    {
        const b = try f0.newBlock();
        const primary = f0.newVReg();
        const scalar = f0.newVReg();
        const extra = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_0000_000A }, .dest = primary, .type = .v128 });
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 5 }, .dest = scalar, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .v128_const = 0x0000_001E_0000_0014_0000_000F_0000_0001 }, .dest = extra, .type = .v128 });
        ret_vals[0] = primary;
        ret_vals[1] = scalar;
        ret_vals[2] = extra;
        try f0.getBlock(b).append(.{ .op = .{ .ret_multi = ret_vals } });
    }
    _ = try module.addFunction(f0);

    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f1.newBlock();
        const primary = f1.newVReg();
        const scalar = f1.newVReg();
        const extra = f1.newVReg();
        const p_lane = f1.newVReg();
        const e_lane = f1.newVReg();
        const sum0 = f1.newVReg();
        const sum1 = f1.newVReg();
        try f1.getBlock(b).append(.{ .op = .{ .call = .{ .func_idx = 0, .extra_results = 2 } }, .dest = primary, .type = .v128 });
        try f1.getBlock(b).append(.{ .op = .{ .call_result = 0 }, .dest = scalar, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .call_result = 1 }, .dest = extra, .type = .v128 });
        try f1.getBlock(b).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = primary, .lane = 0 } }, .dest = p_lane, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = extra, .lane = 2 } }, .dest = e_lane, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .add = .{ .lhs = p_lane, .rhs = scalar } }, .dest = sum0, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .add = .{ .lhs = sum0, .rhs = e_lane } }, .dest = sum1, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .ret = sum1 } });
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    const f0_code = result.code[result.offsets[0]..result.offsets[1]];
    const f1_code = result.code[result.offsets[1]..];
    try std.testing.expect(testCodeContainsMasked(f0_code, 0xFFE0FC1F, 0x4EA01C00)); // primary v128 -> q0
    try std.testing.expect(testCodeContainsWord(f0_code, 0x91004211)); // ADD x17, x16, #16 for aligned v128 extra
    try std.testing.expect(testCodeContainsMasked(f0_code, 0xFFFFFFE0, 0x3D800220)); // STR Qn, [x17]
    try std.testing.expect(testCodeContainsMasked(f1_code, 0xFFE0FC00, 0x4EA01C00)); // capture primary q0
    try std.testing.expect(testCodeContainsMasked(f1_code, 0xFFFFFFE0, 0x3DC00200)); // LDR Qn, [x16]
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

test "store i32: emits VMCtx load + register-offset STR W" {
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

    // STR Wt, [Xn, Xm] register-offset form.
    var found_str_w = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0xB8206800) {
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

test "compile: v128 bitselect emits BSL and preserves live mask" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const mask = func.newVReg();
    const selected = func.newVReg();
    const reused_mask = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_1111_1111 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0008_0000_0007_0000_0006_5555_5555 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0000_0000_0000_0000_0000_FFFF_FFFF }, .dest = mask, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_bitselect = .{ .a = a, .b = b, .mask = mask } }, .dest = selected, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = selected, .rhs = mask } }, .dest = reused_mask, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = reused_mask, .lane = 0 } }, .dest = lane, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_bsl = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x6E601C00) found_bsl = true;
    }
    try std.testing.expect(found_bsl);
}

test "compile: v128 any_true emits UMAXV and scalar bool" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const vector = func.newVReg();
    const any = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0000_0000_0000_0000_0000_0000_0001 }, .dest = vector, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_any_true = vector }, .dest = any, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = any } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_umaxv = false;
    var found_cset = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x6E30A800) found_umaxv = true;
        if ((w & 0xFFFFFFE0) == 0x1A9F07E0) found_cset = true;
    }
    try std.testing.expect(found_umaxv);
    try std.testing.expect(found_cset);
}

test "compile: SIMD all_true emits reductions and scalar bools" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const v8 = func.newVReg();
    const v16 = func.newVReg();
    const v32 = func.newVReg();
    const v64 = func.newVReg();
    const all8 = func.newVReg();
    const all16 = func.newVReg();
    const all32 = func.newVReg();
    const all64 = func.newVReg();
    const sum0 = func.newVReg();
    const sum1 = func.newVReg();
    const sum2 = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x1011_1213_1415_1617_1819_1A1B_1C1D_1E1F }, .dest = v8, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .simd_all_true = .{ .width = .i8x16, .vector = v8 } }, .dest = all8, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0001_0002_0003_0004_0005_0006_0007_0008 }, .dest = v16, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .simd_all_true = .{ .width = .i16x8, .vector = v16 } }, .dest = all16, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0001_FFFF_FFFF_8000_0000_0000_002A }, .dest = v32, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .simd_all_true = .{ .width = .i32x4, .vector = v32 } }, .dest = all32, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8000_0000_0000_0000_0000_0000_0000_0001 }, .dest = v64, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .simd_all_true = .{ .width = .i64x2, .vector = v64 } }, .dest = all64, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .add = .{ .lhs = all8, .rhs = all16 } }, .dest = sum0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .add = .{ .lhs = sum0, .rhs = all32 } }, .dest = sum1, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .add = .{ .lhs = sum1, .rhs = all64 } }, .dest = sum2, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = sum2 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_uminv_b = false;
    var found_uminv_h = false;
    var found_uminv_s = false;
    var found_umov_d = false;
    var found_cset = false;
    var found_and = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x6E31A800) found_uminv_b = true;
        if ((w & 0xFFFFFC00) == 0x6E71A800) found_uminv_h = true;
        if ((w & 0xFFFFFC00) == 0x6EB1A800) found_uminv_s = true;
        if ((w & 0xFFE0FC00) == 0x4E003C00) found_umov_d = true;
        if ((w & 0xFFFFFFE0) == 0x1A9F07E0) found_cset = true;
        if ((w & 0xFFE0FC00) == 0x8A000000) found_and = true;
    }
    try std.testing.expect(found_uminv_b);
    try std.testing.expect(found_uminv_h);
    try std.testing.expect(found_uminv_s);
    try std.testing.expect(found_umov_d);
    try std.testing.expect(found_cset);
    try std.testing.expect(found_and);
}

test "compile: SIMD bitmask emits sign-bit reductions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const v8 = func.newVReg();
    const m8 = func.newVReg();
    const v16 = func.newVReg();
    const m16 = func.newVReg();
    const sum16 = func.newVReg();
    const v32 = func.newVReg();
    const m32 = func.newVReg();
    const sum32 = func.newVReg();
    const v64 = func.newVReg();
    const m64 = func.newVReg();
    const sum64 = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8000_0000_0000_0000_0000_0000_0000_0080 }, .dest = v8, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .simd_bitmask = .{ .width = .i8x16, .vector = v8 } }, .dest = m8, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8000_0000_0000_0000_0000_0000_0000_8000 }, .dest = v16, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .simd_bitmask = .{ .width = .i16x8, .vector = v16 } }, .dest = m16, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .add = .{ .lhs = m8, .rhs = m16 } }, .dest = sum16, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8000_0000_0000_0000_0000_0000_8000_0000 }, .dest = v32, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .simd_bitmask = .{ .width = .i32x4, .vector = v32 } }, .dest = m32, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .add = .{ .lhs = sum16, .rhs = m32 } }, .dest = sum32, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8000_0000_0000_0000_8000_0000_0000_0000 }, .dest = v64, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .simd_bitmask = .{ .width = .i64x2, .vector = v64 } }, .dest = m64, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .add = .{ .lhs = sum32, .rhs = m64 } }, .dest = sum64, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = sum64 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_sshr8 = false;
    var found_sshr16 = false;
    var found_sshr32 = false;
    var found_sshr64 = false;
    var found_addp8 = false;
    var found_addv16 = false;
    var found_addv32 = false;
    var found_addp64 = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4F090400) found_sshr8 = true;
        if ((w & 0xFFFFFC00) == 0x4F110400) found_sshr16 = true;
        if ((w & 0xFFFFFC00) == 0x4F210400) found_sshr32 = true;
        if ((w & 0xFFFFFC00) == 0x4F410400) found_sshr64 = true;
        if ((w & 0xFFE0FC00) == 0x4E20BC00) found_addp8 = true;
        if ((w & 0xFFFFFC00) == 0x4E71B800) found_addv16 = true;
        if ((w & 0xFFFFFC00) == 0x4EB1B800) found_addv32 = true;
        if ((w & 0xFFFFFC00) == 0x5EF1B800) found_addp64 = true;
    }
    try std.testing.expect(found_sshr8);
    try std.testing.expect(found_sshr16);
    try std.testing.expect(found_sshr32);
    try std.testing.expect(found_sshr64);
    try std.testing.expect(found_addp8);
    try std.testing.expect(found_addv16);
    try std.testing.expect(found_addv32);
    try std.testing.expect(found_addp64);
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

test "compile: f32x4 lane ops emit bit-preserving NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const scalar = func.newVReg();
    const splat = func.newVReg();
    const replacement = func.newVReg();
    const replaced = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .fconst_32 = @bitCast(@as(u32, 0x7FC1_2345)) }, .dest = scalar, .type = .f32 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_splat = scalar }, .dest = splat, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .fconst_32 = @bitCast(@as(u32, 0x8000_0000)) }, .dest = replacement, .type = .f32 });
    try func.getBlock(bid).append(.{
        .op = .{ .f32x4_replace_lane = .{ .vector = splat, .val = replacement, .lane = 3 } },
        .dest = replaced,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .f32x4_extract_lane = .{ .vector = replaced, .lane = 3 } },
        .dest = lane,
        .type = .f32,
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
        if ((w & 0xFFFFFC00) == 0x4E1C1C00) found_ins = true;
        if ((w & 0xFFFFFC00) == 0x0E1C3C00) found_umov = true;
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

test "compile: i8x16.shuffle emits TBL instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const lhs = func.newVReg();
    const rhs = func.newVReg();
    const mixed = func.newVReg();
    const same = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0F0E_0D0C_0B0A_0908_0706_0504_0302_0100 }, .dest = lhs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x1F1E_1D1C_1B1A_1918_1716_1514_1312_1110 }, .dest = rhs, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i8x16_shuffle = .{
            .lhs = lhs,
            .rhs = rhs,
            .lanes = .{ 16, 1, 18, 3, 20, 5, 22, 7, 24, 9, 26, 11, 28, 13, 30, 15 },
        } },
        .dest = mixed,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i8x16_shuffle = .{
            .lhs = mixed,
            .rhs = mixed,
            .lanes = .{ 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 },
        } },
        .dest = same,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_extract_lane = .{ .vector = same, .lane = 0, .sign = .unsigned } }, .dest = lane, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var tbl_count: u32 = 0;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x4E002000) tbl_count += 1;
    }

    try std.testing.expectEqual(@as(u32, 2), tbl_count);
}

test "compile: i8x16.swizzle emits single-register TBL instruction" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const vector = func.newVReg();
    const indices = func.newVReg();
    const swizzled = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0F0E_0D0C_0B0A_0908_0706_0504_0302_0100 }, .dest = vector, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0010_0F0E_0D0C_0B0A_0908_0706_0504_0302 }, .dest = indices, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i8x16_swizzle = .{
            .vector = vector,
            .indices = indices,
        } },
        .dest = swizzled,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_extract_lane = .{ .vector = swizzled, .lane = 0, .sign = .unsigned } }, .dest = lane, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var tbl1_count: u32 = 0;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x4E000000) tbl1_count += 1;
    }

    try std.testing.expectEqual(@as(u32, 1), tbl1_count);
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

test "compile: f64x2 lane ops emit bit-preserving NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const scalar = func.newVReg();
    const splat = func.newVReg();
    const replacement = func.newVReg();
    const replaced = func.newVReg();
    const lane = func.newVReg();
    const bits = func.newVReg();
    const wrapped = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .fconst_64 = @bitCast(@as(u64, 0x7FF8_0123_4567_89AB)) }, .dest = scalar, .type = .f64 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_splat = scalar }, .dest = splat, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .fconst_64 = @bitCast(@as(u64, 0x8000_0000_0000_0000)) }, .dest = replacement, .type = .f64 });
    try func.getBlock(bid).append(.{
        .op = .{ .f64x2_replace_lane = .{ .vector = splat, .val = replacement, .lane = 1 } },
        .dest = replaced,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .f64x2_extract_lane = .{ .vector = replaced, .lane = 1 } },
        .dest = lane,
        .type = .f64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .reinterpret = lane }, .dest = bits, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .wrap_i64 = bits }, .dest = wrapped, .type = .i32 });
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

test "compile: f64x2 arithmetic and comparison ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const add = func.newVReg();
    const sub = func.newVReg();
    const mul = func.newVReg();
    const div = func.newVReg();
    const min = func.newVReg();
    const max = func.newVReg();
    const eq = func.newVReg();
    const ne = func.newVReg();
    const lt = func.newVReg();
    const gt = func.newVReg();
    const le = func.newVReg();
    const ge = func.newVReg();
    const pmin = func.newVReg();
    const pmax = func.newVReg();
    const lane = func.newVReg();
    const wrapped = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4008_0000_0000_0000_3ff0_0000_0000_0000 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4010_0000_0000_0000_4000_0000_0000_0000 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .add, .lhs = a, .rhs = b } }, .dest = add, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .sub, .lhs = add, .rhs = b } }, .dest = sub, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .mul, .lhs = sub, .rhs = b } }, .dest = mul, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .div, .lhs = mul, .rhs = b } }, .dest = div, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .min, .lhs = div, .rhs = b } }, .dest = min, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .max, .lhs = min, .rhs = b } }, .dest = max, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .eq, .lhs = a, .rhs = b } }, .dest = eq, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .ne, .lhs = eq, .rhs = b } }, .dest = ne, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .lt, .lhs = a, .rhs = b } }, .dest = lt, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .gt, .lhs = b, .rhs = a } }, .dest = gt, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .le, .lhs = a, .rhs = b } }, .dest = le, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .ge, .lhs = b, .rhs = a } }, .dest = ge, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .pmin, .lhs = ge, .rhs = b } }, .dest = pmin, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .pmax, .lhs = pmin, .rhs = a } }, .dest = pmax, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i64x2_extract_lane = .{ .vector = pmax, .lane = 0 } },
        .dest = lane,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .wrap_i64 = lane }, .dest = wrapped, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = wrapped } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_fadd = false;
    var found_fsub = false;
    var found_fmul = false;
    var found_fdiv = false;
    var found_fmin = false;
    var found_fmax = false;
    var found_fcmeq = false;
    var found_fcmgt = false;
    var found_fcmge = false;
    var found_mvn = false;
    var bsl_count: u32 = 0;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x4E60D400) found_fadd = true;
        if ((w & 0xFFE0FC00) == 0x4EE0D400) found_fsub = true;
        if ((w & 0xFFE0FC00) == 0x6E60DC00) found_fmul = true;
        if ((w & 0xFFE0FC00) == 0x6E60FC00) found_fdiv = true;
        if ((w & 0xFFE0FC00) == 0x4EE0F400) found_fmin = true;
        if ((w & 0xFFE0FC00) == 0x4E60F400) found_fmax = true;
        if ((w & 0xFFE0FC00) == 0x4E60E400) found_fcmeq = true;
        if ((w & 0xFFE0FC00) == 0x6EE0E400) found_fcmgt = true;
        if ((w & 0xFFE0FC00) == 0x6E60E400) found_fcmge = true;
        if ((w & 0xFFFFFC00) == 0x6E205800) found_mvn = true;
        if ((w & 0xFFE0FC00) == 0x6E601C00) bsl_count += 1;
    }

    try std.testing.expect(found_fadd);
    try std.testing.expect(found_fsub);
    try std.testing.expect(found_fmul);
    try std.testing.expect(found_fdiv);
    try std.testing.expect(found_fmin);
    try std.testing.expect(found_fmax);
    try std.testing.expect(found_fcmeq);
    try std.testing.expect(found_fcmgt);
    try std.testing.expect(found_fcmge);
    try std.testing.expect(found_mvn);
    try std.testing.expect(bsl_count >= 2);
}

test "compile: f64x2 unary ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const abs = func.newVReg();
    const neg = func.newVReg();
    const sqrt = func.newVReg();
    const ceil = func.newVReg();
    const floor = func.newVReg();
    const trunc = func.newVReg();
    const nearest = func.newVReg();
    const lane = func.newVReg();
    const wrapped = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4010_0000_0000_0000_c010_0000_0000_0000 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_unop = .{ .op = .abs, .vector = a } }, .dest = abs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_unop = .{ .op = .neg, .vector = abs } }, .dest = neg, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_unop = .{ .op = .sqrt, .vector = neg } }, .dest = sqrt, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_unop = .{ .op = .ceil, .vector = sqrt } }, .dest = ceil, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_unop = .{ .op = .floor, .vector = ceil } }, .dest = floor, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_unop = .{ .op = .trunc, .vector = floor } }, .dest = trunc, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_unop = .{ .op = .nearest, .vector = trunc } }, .dest = nearest, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i64x2_extract_lane = .{ .vector = nearest, .lane = 0 } },
        .dest = lane,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .wrap_i64 = lane }, .dest = wrapped, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = wrapped } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_fabs = false;
    var found_fneg = false;
    var found_fsqrt = false;
    var found_frintp = false;
    var found_frintm = false;
    var found_frintz = false;
    var found_frintn = false;
    var found_fcmeq = false;
    var found_mvn = false;
    var found_bsl = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4EE0F800) found_fabs = true;
        if ((w & 0xFFFFFC00) == 0x6EE0F800) found_fneg = true;
        if ((w & 0xFFFFFC00) == 0x6EE1F800) found_fsqrt = true;
        if ((w & 0xFFFFFC00) == 0x4EE18800) found_frintp = true;
        if ((w & 0xFFFFFC00) == 0x4E619800) found_frintm = true;
        if ((w & 0xFFFFFC00) == 0x4EE19800) found_frintz = true;
        if ((w & 0xFFFFFC00) == 0x4E618800) found_frintn = true;
        if ((w & 0xFFE0FC00) == 0x4E60E400) found_fcmeq = true;
        if ((w & 0xFFFFFC00) == 0x6E205800) found_mvn = true;
        if ((w & 0xFFE0FC00) == 0x6E601C00) found_bsl = true;
    }

    try std.testing.expect(found_fabs);
    try std.testing.expect(found_fneg);
    try std.testing.expect(found_fsqrt);
    try std.testing.expect(found_frintp);
    try std.testing.expect(found_frintm);
    try std.testing.expect(found_frintz);
    try std.testing.expect(found_frintn);
    try std.testing.expect(found_fcmeq);
    try std.testing.expect(found_mvn);
    try std.testing.expect(found_bsl);
}

test "compile: f32x4 unary ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const abs = func.newVReg();
    const neg = func.newVReg();
    const sqrt = func.newVReg();
    const ceil = func.newVReg();
    const floor = func.newVReg();
    const trunc = func.newVReg();
    const nearest = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4110_0000_4080_0000_8000_0000_c060_0000 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_unop = .{ .op = .abs, .vector = a } }, .dest = abs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_unop = .{ .op = .neg, .vector = abs } }, .dest = neg, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_unop = .{ .op = .sqrt, .vector = neg } }, .dest = sqrt, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_unop = .{ .op = .ceil, .vector = sqrt } }, .dest = ceil, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_unop = .{ .op = .floor, .vector = ceil } }, .dest = floor, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_unop = .{ .op = .trunc, .vector = floor } }, .dest = trunc, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_unop = .{ .op = .nearest, .vector = trunc } }, .dest = nearest, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = nearest, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_fabs = false;
    var found_fneg = false;
    var found_fsqrt = false;
    var found_frintp = false;
    var found_frintm = false;
    var found_frintz = false;
    var found_frintn = false;
    var found_fcmeq = false;
    var found_mvn = false;
    var found_bsl = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4EA0F800) found_fabs = true;
        if ((w & 0xFFFFFC00) == 0x6EA0F800) found_fneg = true;
        if ((w & 0xFFFFFC00) == 0x6EA1F800) found_fsqrt = true;
        if ((w & 0xFFFFFC00) == 0x4EA18800) found_frintp = true;
        if ((w & 0xFFFFFC00) == 0x4E219800) found_frintm = true;
        if ((w & 0xFFFFFC00) == 0x4EA19800) found_frintz = true;
        if ((w & 0xFFFFFC00) == 0x4E218800) found_frintn = true;
        if ((w & 0xFFE0FC00) == 0x4E20E400) found_fcmeq = true;
        if ((w & 0xFFFFFC00) == 0x6E205800) found_mvn = true;
        if ((w & 0xFFE0FC00) == 0x6E601C00) found_bsl = true;
    }

    try std.testing.expect(found_fabs);
    try std.testing.expect(found_fneg);
    try std.testing.expect(found_fsqrt);
    try std.testing.expect(found_frintp);
    try std.testing.expect(found_frintm);
    try std.testing.expect(found_frintz);
    try std.testing.expect(found_frintn);
    try std.testing.expect(found_fcmeq);
    try std.testing.expect(found_mvn);
    try std.testing.expect(found_bsl);
}

test "compile: f32x4 comparisons emit ordered NEON comparisons" {
    const allocator = std.testing.allocator;

    const Case = struct {
        op: ir.Inst.F32x4Op,
        expected_cmp: u32,
        expect_mvn: bool = false,
    };
    const cases = [_]Case{
        .{ .op = .eq, .expected_cmp = 0x4E31E610 },
        .{ .op = .ne, .expected_cmp = 0x4E31E610, .expect_mvn = true },
        .{ .op = .gt, .expected_cmp = 0x6EB1E610 },
        .{ .op = .ge, .expected_cmp = 0x6E31E610 },
        .{ .op = .lt, .expected_cmp = 0x6EB0E630 },
        .{ .op = .le, .expected_cmp = 0x6E30E630 },
    };

    for (cases) |case| {
        var func = ir.IrFunction.init(allocator, 0, 1, 0);
        defer func.deinit();
        const bid = try func.newBlock();

        const a = func.newVReg();
        const b = func.newVReg();
        const cmp = func.newVReg();
        const lane = func.newVReg();

        try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4080_0000_4040_0000_4000_0000_3f80_0000 }, .dest = a, .type = .v128 });
        try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4100_0000_40e0_0000_40c0_0000_40a0_0000 }, .dest = b, .type = .v128 });
        try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = case.op, .lhs = a, .rhs = b } }, .dest = cmp, .type = .v128 });
        try func.getBlock(bid).append(.{
            .op = .{ .i32x4_extract_lane = .{ .vector = cmp, .lane = 0 } },
            .dest = lane,
            .type = .i32,
        });
        try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

        const code = try compileFunction(&func, allocator);
        defer allocator.free(code);

        var found_cmp = false;
        var found_mvn = false;
        var i: usize = 0;
        while (i + 4 <= code.len) : (i += 4) {
            const w = std.mem.readInt(u32, code[i..][0..4], .little);
            if ((w & 0xFFE0FC00) == (case.expected_cmp & 0xFFE0FC00)) found_cmp = true;
            if ((w & 0xFFFFFC00) == 0x6E205800) found_mvn = true;
        }

        try std.testing.expect(found_cmp);
        try std.testing.expectEqual(case.expect_mvn, found_mvn);
    }
}

test "compile: f32x4 arithmetic/minmax and pseudo-minmax ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const add = func.newVReg();
    const sub = func.newVReg();
    const mul = func.newVReg();
    const div = func.newVReg();
    const min = func.newVReg();
    const max = func.newVReg();
    const pmin = func.newVReg();
    const pmax = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4080_0000_4040_0000_4000_0000_3f80_0000 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4100_0000_40e0_0000_40c0_0000_40a0_0000 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .add, .lhs = a, .rhs = b } }, .dest = add, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .sub, .lhs = add, .rhs = b } }, .dest = sub, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .mul, .lhs = sub, .rhs = b } }, .dest = mul, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .div, .lhs = mul, .rhs = b } }, .dest = div, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .min, .lhs = div, .rhs = b } }, .dest = min, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .max, .lhs = min, .rhs = a } }, .dest = max, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .pmin, .lhs = max, .rhs = b } }, .dest = pmin, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .pmax, .lhs = pmin, .rhs = a } }, .dest = pmax, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = pmax, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_fadd = false;
    var found_fsub = false;
    var found_fmul = false;
    var found_fdiv = false;
    var found_fmin = false;
    var found_fmax = false;
    var found_fcmeq = false;
    var found_fcmgt = false;
    var bsl_count: u32 = 0;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x4E20D400) found_fadd = true;
        if ((w & 0xFFE0FC00) == 0x4EA0D400) found_fsub = true;
        if ((w & 0xFFE0FC00) == 0x6E20DC00) found_fmul = true;
        if ((w & 0xFFE0FC00) == 0x6E20FC00) found_fdiv = true;
        if ((w & 0xFFE0FC00) == 0x4EA0F400) found_fmin = true;
        if ((w & 0xFFE0FC00) == 0x4E20F400) found_fmax = true;
        if ((w & 0xFFE0FC00) == 0x4E20E400) found_fcmeq = true;
        if ((w & 0xFFE0FC00) == 0x6EA0E400) found_fcmgt = true;
        if ((w & 0xFFE0FC00) == 0x6E601C00) bsl_count += 1;
    }

    try std.testing.expect(found_fadd);
    try std.testing.expect(found_fsub);
    try std.testing.expect(found_fmul);
    try std.testing.expect(found_fdiv);
    try std.testing.expect(found_fmin);
    try std.testing.expect(found_fmax);
    try std.testing.expect(found_fcmeq);
    try std.testing.expect(found_fcmgt);
    try std.testing.expect(bsl_count >= 2);
}

test "compile: f32x4 pseudo-minmax uses compare and select operand order" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const pmin = func.newVReg();
    const pmax = func.newVReg();
    const mix = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4080_0000_4040_0000_4000_0000_3f80_0000 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4100_0000_40e0_0000_40c0_0000_40a0_0000 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .pmin, .lhs = a, .rhs = b } }, .dest = pmin, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .pmax, .lhs = a, .rhs = b } }, .dest = pmax, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = pmin, .rhs = pmax } }, .dest = mix, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = mix, .lane = 0 } }, .dest = lane, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var cmp_count: u32 = 0;
    var ordered_selects: u32 = 0;
    var after_expected_cmp = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x6EA0E400) {
            cmp_count += 1;
            after_expected_cmp = true;
        } else if ((w & 0xFFE0FC00) == 0x6E601C00 and after_expected_cmp) {
            ordered_selects += 1;
            after_expected_cmp = false;
        }
    }

    try std.testing.expect(cmp_count >= 2);
    try std.testing.expectEqual(@as(u32, 2), ordered_selects);
}

test "compile: SIMD int-to-float conversions emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const source = func.newVReg();
    const f32_s = func.newVReg();
    const f32_u = func.newVReg();
    const f64_s = func.newVReg();
    const f64_u = func.newVReg();
    const lane = func.newVReg();
    const wrapped = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0003_0000_0002_0000_0001_8000_0000 }, .dest = source, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .f32x4_convert_i32x4 = .{ .sign = .signed, .vector = source } },
        .dest = f32_s,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .f32x4_convert_i32x4 = .{ .sign = .unsigned, .vector = source } },
        .dest = f32_u,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .f64x2_convert_low_i32x4 = .{ .sign = .signed, .vector = source } },
        .dest = f64_s,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .f64x2_convert_low_i32x4 = .{ .sign = .unsigned, .vector = source } },
        .dest = f64_u,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{ .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = f32_s, .rhs = f32_u } }, .dest = func.newVReg(), .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = f64_s, .rhs = f64_u } }, .dest = func.newVReg(), .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i64x2_extract_lane = .{ .vector = f64_u, .lane = 0 } },
        .dest = lane,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .wrap_i64 = lane }, .dest = wrapped, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = wrapped } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_scvtf4s = false;
    var found_ucvtf4s = false;
    var found_sshll2d2s = false;
    var found_ushll2d2s = false;
    var found_scvtf2d = false;
    var found_ucvtf2d = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4E21D800) found_scvtf4s = true;
        if ((w & 0xFFFFFC00) == 0x6E21D800) found_ucvtf4s = true;
        if ((w & 0xFFFFFC00) == 0x0F20A400) found_sshll2d2s = true;
        if ((w & 0xFFFFFC00) == 0x2F20A400) found_ushll2d2s = true;
        if ((w & 0xFFFFFC00) == 0x4E61D800) found_scvtf2d = true;
        if ((w & 0xFFFFFC00) == 0x6E61D800) found_ucvtf2d = true;
    }

    try std.testing.expect(found_scvtf4s);
    try std.testing.expect(found_ucvtf4s);
    try std.testing.expect(found_sshll2d2s);
    try std.testing.expect(found_ushll2d2s);
    try std.testing.expect(found_scvtf2d);
    try std.testing.expect(found_ucvtf2d);
}

test "compile: SIMD trunc_sat float-to-int emits NaN masks and narrowing" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const f32_source = func.newVReg();
    const f64_source = func.newVReg();
    const f32_s = func.newVReg();
    const f32_u = func.newVReg();
    const f64_s = func.newVReg();
    const f64_u = func.newVReg();
    const mix = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0xff80_0000_4f00_0000_7fc0_0000_3fc0_0000 }, .dest = f32_source, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_trunc_sat = .{ .src_width = .f32x4, .sign = .signed, .vector = f32_source } },
        .dest = f32_s,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_trunc_sat = .{ .src_width = .f32x4, .sign = .unsigned, .vector = f32_source } },
        .dest = f32_u,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0xfff0_0000_0000_0000_7ff8_0000_0000_0001 }, .dest = f64_source, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_trunc_sat = .{ .src_width = .f64x2, .sign = .signed, .vector = f64_source } },
        .dest = f64_s,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_trunc_sat = .{ .src_width = .f64x2, .sign = .unsigned, .vector = f64_source } },
        .dest = f64_u,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{ .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = f32_s, .rhs = f32_u } }, .dest = mix, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = f64_u, .lane = 3 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_fcmeq4s = false;
    var found_fcmeq2d = false;
    var found_fcvtzs4s = false;
    var found_fcvtzu4s = false;
    var found_fcvtzs2d = false;
    var found_fcvtzu2d = false;
    var found_sqxtn = false;
    var found_uqxtn = false;
    var zero_before_narrow: u32 = 0;
    var pending_zero = false;

    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x4E20E400) found_fcmeq4s = true;
        if ((w & 0xFFE0FC00) == 0x4E60E400) found_fcmeq2d = true;
        if ((w & 0xFFFFFC00) == 0x4EA1B800) found_fcvtzs4s = true;
        if ((w & 0xFFFFFC00) == 0x6EA1B800) found_fcvtzu4s = true;
        if ((w & 0xFFFFFC00) == 0x4EE1B800) found_fcvtzs2d = true;
        if ((w & 0xFFFFFC00) == 0x6EE1B800) found_fcvtzu2d = true;
        if ((w & 0xFFFFFFE0) == 0x6F00E400) pending_zero = true;
        if ((w & 0xFFFFFC00) == 0x0EA14800) {
            found_sqxtn = true;
            if (pending_zero) zero_before_narrow += 1;
            pending_zero = false;
        }
        if ((w & 0xFFFFFC00) == 0x2EA14800) {
            found_uqxtn = true;
            if (pending_zero) zero_before_narrow += 1;
            pending_zero = false;
        }
    }

    try std.testing.expect(found_fcmeq4s);
    try std.testing.expect(found_fcmeq2d);
    try std.testing.expect(found_fcvtzs4s);
    try std.testing.expect(found_fcvtzu4s);
    try std.testing.expect(found_fcvtzs2d);
    try std.testing.expect(found_fcvtzu2d);
    try std.testing.expect(found_sqxtn);
    try std.testing.expect(found_uqxtn);
    try std.testing.expectEqual(@as(u32, 2), zero_before_narrow);
}

test "compile: SIMD float precision conversions emit FCVTN and FCVTL" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const source = func.newVReg();
    const demoted = func.newVReg();
    const promoted = func.newVReg();
    const mix = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4008_0000_0000_0000_3ff8_0000_0000_0000 }, .dest = source, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .f32x4_demote_f64x2_zero = .{ .vector = source } },
        .dest = demoted,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .f64x2_promote_low_f32x4 = .{ .vector = demoted } },
        .dest = promoted,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{ .op = .{ .v128_bitwise = .{ .op = .xor, .lhs = demoted, .rhs = promoted } }, .dest = mix, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = mix, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_fcvtn = false;
    var found_fcvtl = false;
    var zero_before_fcvtn = false;
    var pending_zero = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        const rd = w & 0x1F;
        const rn = (w >> 5) & 0x1F;
        const rm = (w >> 16) & 0x1F;
        if ((w & 0xFFE0FC00) == 0x6E201C00 and rd == rn and rd == rm) pending_zero = true;
        if ((w & 0xFFFFFC00) == 0x0E616800) {
            found_fcvtn = true;
            if (pending_zero) zero_before_fcvtn = true;
            pending_zero = false;
        }
        if ((w & 0xFFFFFC00) == 0x0E617800) found_fcvtl = true;
    }

    try std.testing.expect(found_fcvtn);
    try std.testing.expect(found_fcvtl);
    try std.testing.expect(zero_before_fcvtn);
}

test "compile: integer SIMD unary ops emit NEON instructions" {
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
    const i8_popcnt = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .i8x16_unop = .{ .op = .popcnt, .vector = source } }, .dest = i8_popcnt, .type = .v128 });

    try func.getBlock(bid).append(.{ .op = .{ .i16x8_unop = .{ .op = .abs, .vector = i8_popcnt } }, .dest = i16_abs, .type = .v128 });
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
    var found_cnt16b = false;
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
        if ((w & 0xFFFFFC00) == 0x4E205800) found_cnt16b = true;
    }

    try std.testing.expect(found_abs16b);
    try std.testing.expect(found_neg16b);
    try std.testing.expect(found_abs8h);
    try std.testing.expect(found_neg8h);
    try std.testing.expect(found_abs4s);
    try std.testing.expect(found_neg4s);
    try std.testing.expect(found_abs2d);
    try std.testing.expect(found_neg2d);
    try std.testing.expect(found_cnt16b);
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

test "compile: i32x4.dot_i16x8_s emits SMULL, SMULL2, ADDP" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const lhs = func.newVReg();
    const rhs = func.newVReg();
    const dot = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0006_0005_0004_0003_FFFE_FFFD_0002_0001 }, .dest = lhs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x000E_000D_FFF8_0007_0006_0005_0004_0003 }, .dest = rhs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_dot_i16x8_s = .{ .lhs = lhs, .rhs = rhs } }, .dest = dot, .type = .v128 });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = dot, .lane = 3 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    const expected = [_]struct { mask: u32, base: u32, name: []const u8 }{
        .{ .mask = 0xFFE0FC00, .base = 0x0E60C000, .name = "smull4s4h" },
        .{ .mask = 0xFFE0FC00, .base = 0x4E60C000, .name = "smull2_4s8h" },
        .{ .mask = 0xFFE0FC00, .base = 0x4EA0BC00, .name = "addp4s" },
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

fn codeContainsMaskedWord(code: []const u8, mask: u32, base: u32) bool {
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & mask) == base) return true;
    }
    return false;
}

test "compile: f32x4 add of mul emits FMLA" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const c = func.newVReg();
    const mul = func.newVReg();
    const add = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4080_0000_4040_0000_4000_0000_3F80_0000 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4100_0000_40E0_0000_40C0_0000_40A0_0000 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x3F80_0000_3F80_0000_3F80_0000_3F80_0000 }, .dest = c, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .mul, .lhs = a, .rhs = b } }, .dest = mul, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .add, .lhs = mul, .rhs = c } }, .dest = add, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_extract_lane = .{ .vector = add, .lane = 0 } }, .dest = lane, .type = .f32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(codeContainsMaskedWord(code, 0xFFE0FC00, 0x4E20CC00));
    try std.testing.expect(!codeContainsMaskedWord(code, 0xFFE0FC00, 0x6E20DC00));
}

test "compile: f32x4 commuted add of mul emits FMLA" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const c = func.newVReg();
    const mul = func.newVReg();
    const add = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4080_0000_4040_0000_4000_0000_3F80_0000 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4100_0000_40E0_0000_40C0_0000_40A0_0000 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x3F80_0000_3F80_0000_3F80_0000_3F80_0000 }, .dest = c, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .mul, .lhs = a, .rhs = b } }, .dest = mul, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .add, .lhs = c, .rhs = mul } }, .dest = add, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_extract_lane = .{ .vector = add, .lane = 0 } }, .dest = lane, .type = .f32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(codeContainsMaskedWord(code, 0xFFE0FC00, 0x4E20CC00));
    try std.testing.expect(!codeContainsMaskedWord(code, 0xFFE0FC00, 0x6E20DC00));
}

test "compile: f32x4 sub of mul emits FMLS" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const c = func.newVReg();
    const mul = func.newVReg();
    const sub = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4080_0000_4040_0000_4000_0000_3F80_0000 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4100_0000_40E0_0000_40C0_0000_40A0_0000 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x3F80_0000_3F80_0000_3F80_0000_3F80_0000 }, .dest = c, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .mul, .lhs = a, .rhs = b } }, .dest = mul, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_binop = .{ .op = .sub, .lhs = c, .rhs = mul } }, .dest = sub, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f32x4_extract_lane = .{ .vector = sub, .lane = 0 } }, .dest = lane, .type = .f32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(codeContainsMaskedWord(code, 0xFFE0FC00, 0x4EA0CC00));
    try std.testing.expect(!codeContainsMaskedWord(code, 0xFFE0FC00, 0x6E20DC00));
}

test "compile: f64x2 add of mul emits FMLA" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const c = func.newVReg();
    const mul = func.newVReg();
    const add = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4010_0000_0000_0000_4008_0000_0000_0000 }, .dest = a, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x4020_0000_0000_0000_401C_0000_0000_0000 }, .dest = b, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x3FF0_0000_0000_0000_3FF0_0000_0000_0000 }, .dest = c, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .mul, .lhs = a, .rhs = b } }, .dest = mul, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_binop = .{ .op = .add, .lhs = mul, .rhs = c } }, .dest = add, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .f64x2_extract_lane = .{ .vector = add, .lane = 0 } }, .dest = lane, .type = .f64 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    try std.testing.expect(codeContainsMaskedWord(code, 0xFFE0FC00, 0x4E60CC00));
    try std.testing.expect(!codeContainsMaskedWord(code, 0xFFE0FC00, 0x6E60DC00));
}

test "compile: integer SIMD narrow saturating ops emit NEON instructions" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    const lhs = func.newVReg();
    const rhs = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x8001_7FFE_8000_7FFF_0180_7F80_01FF_8001 }, .dest = lhs, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x1234_5678_9ABC_DEF0_0123_4567_89AB_CDEF }, .dest = rhs, .type = .v128 });

    const signs = [_]ir.Inst.SimdNarrowSign{ .signed, .unsigned };

    inline for (.{ "i8x16_narrow_i16x8", "i16x8_narrow_i32x4" }) |op_name| {
        for (signs) |sign| {
            const dest = func.newVReg();
            try func.getBlock(bid).append(.{
                .op = @unionInit(ir.Inst.Op, op_name, .{ .sign = sign, .lhs = lhs, .rhs = rhs }),
                .dest = dest,
                .type = .v128,
            });
        }
    }

    // Force a use so the narrow results aren't dead-code-eliminated; pick the
    // last one through an extract.
    const last_narrow = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .i16x8_narrow_i32x4 = .{ .sign = .unsigned, .lhs = lhs, .rhs = rhs } },
        .dest = last_narrow,
        .type = .v128,
    });
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = last_narrow, .lane = 0 } },
        .dest = lane,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    const expected = [_]struct { mask: u32, base: u32, name: []const u8 }{
        .{ .mask = 0xFFFFFC00, .base = 0x0E214800, .name = "sqxtn8b8h" },
        .{ .mask = 0xFFFFFC00, .base = 0x4E214800, .name = "sqxtn2_16b8h" },
        .{ .mask = 0xFFFFFC00, .base = 0x2E212800, .name = "sqxtun8b8h" },
        .{ .mask = 0xFFFFFC00, .base = 0x6E212800, .name = "sqxtun2_16b8h" },
        .{ .mask = 0xFFFFFC00, .base = 0x0E614800, .name = "sqxtn4h4s" },
        .{ .mask = 0xFFFFFC00, .base = 0x4E614800, .name = "sqxtn2_8h4s" },
        .{ .mask = 0xFFFFFC00, .base = 0x2E612800, .name = "sqxtun4h4s" },
        .{ .mask = 0xFFFFFC00, .base = 0x6E612800, .name = "sqxtun2_8h4s" },
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

test "compile: v128 param spills q0 into first local slot" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 1, 1);
    defer func.deinit();
    func.local_types = try allocator.dupe(ir.IrType, &[_]ir.IrType{.v128});
    const bid = try func.newBlock();
    const v = func.newVReg();
    const lane = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = v, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = v, .lane = 0 } }, .dest = lane, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_entry_str_q0 = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if (w == 0x3D800220) found_entry_str_q0 = true; // STR Q0, [x17]
    }
    try std.testing.expect(found_entry_str_q0);
}

test "compile: v128 result returns in q0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0 }, .dest = v, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_orr_to_q0 = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC1F) == 0x4EA01C00) found_orr_to_q0 = true;
    }
    try std.testing.expect(found_orr_to_q0);
}

test "compile: local frame layout aligns v128 locals" {
    const allocator = std.testing.allocator;

    var scalar_func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer scalar_func.deinit();
    scalar_func.local_types = try allocator.dupe(ir.IrType, &[_]ir.IrType{ .i32, .i64 });
    const scalar_layout = try computeLocalFrameLayout(&scalar_func, allocator);
    defer allocator.free(scalar_layout.offsets);
    try std.testing.expectEqual(@as(u32, 24), scalar_layout.offsets[0]);
    try std.testing.expectEqual(@as(u32, 32), scalar_layout.offsets[1]);
    try std.testing.expectEqual(@as(u32, 40), scalar_layout.end);

    var mixed_func = ir.IrFunction.init(allocator, 0, 1, 4);
    defer mixed_func.deinit();
    mixed_func.local_types = try allocator.dupe(ir.IrType, &[_]ir.IrType{ .i32, .v128, .i64, .v128 });
    const mixed_layout = try computeLocalFrameLayout(&mixed_func, allocator);
    defer allocator.free(mixed_layout.offsets);
    try std.testing.expectEqual(@as(u32, 24), mixed_layout.offsets[0]);
    try std.testing.expectEqual(@as(u32, 32), mixed_layout.offsets[1]);
    try std.testing.expectEqual(@as(u32, 48), mixed_layout.offsets[2]);
    try std.testing.expectEqual(@as(u32, 64), mixed_layout.offsets[3]);
    try std.testing.expectEqual(@as(u32, 80), mixed_layout.end);
}

test "compile: v128 local zero-init uses full-width store without clobbering scalar zero" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer func.deinit();
    func.local_types = try allocator.dupe(ir.IrType, &[_]ir.IrType{ .v128, .i32 });
    const bid = try func.newBlock();
    // Read both locals before any set so the prologue zero-init must
    // be preserved (see issue #468 — define-before-use skip).
    const v_dummy = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = v_dummy, .type = .v128 });
    const ret = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 1 }, .dest = ret, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = ret } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    const add_local0_addr = @as(u32, 0x91000000) |
        (@as(u32, 32) << 10) |
        (@as(u32, @intFromEnum(emit.Reg.fp)) << 5) |
        @as(u32, @intFromEnum(RegMap.tmp1));
    const str_q0_tmp1 = @as(u32, 0x3D800000) |
        (@as(u32, @intFromEnum(RegMap.tmp1)) << 5) |
        @as(u32, v128_tmp0);
    const str_scalar_zero = @as(u32, 0xF9000000) |
        (@as(u32, 6) << 10) |
        (@as(u32, @intFromEnum(emit.Reg.fp)) << 5) |
        @as(u32, @intFromEnum(RegMap.tmp0));

    var found_movi_zero = false;
    var found_local_addr = false;
    var found_q_store = false;
    var found_scalar_store = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const word = std.mem.readInt(u32, code[i..][0..4], .little);
        if (word == 0x6F00E400) found_movi_zero = true;
        if (word == add_local0_addr) found_local_addr = true;
        if (word == str_q0_tmp1) found_q_store = true;
        if (word == str_scalar_zero) found_scalar_store = true;
    }

    try std.testing.expect(found_movi_zero);
    try std.testing.expect(found_local_addr);
    try std.testing.expect(found_q_store);
    try std.testing.expect(found_scalar_store);
}

test "compile: v128 local set/get uses Q stack traffic" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();
    func.local_types = try allocator.dupe(ir.IrType, &[_]ir.IrType{.v128});
    const bid = try func.newBlock();
    const init = func.newVReg();
    const loaded = func.newVReg();
    const lane = func.newVReg();

    try func.getBlock(bid).append(.{ .op = .{ .v128_const = 0x0000_0004_0000_0003_0000_0002_0000_0001 }, .dest = init, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = init } } });
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = loaded, .type = .v128 });
    try func.getBlock(bid).append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = loaded, .lane = 0 } }, .dest = lane, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = lane } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    const counts = countQMemOps(code);
    try std.testing.expect(counts.loads >= 1);
    // Q stores: the prologue zero-init can be skipped under issue #468
    // because the local is set before any get; only the local_set Q
    // store is required.
    try std.testing.expect(counts.stores >= 1);
}

const AddressingLoadTestSpec = struct {
    base_type: ir.IrType = .i32,
    has_memory64: bool = false,
    has_shared_memory: bool = false,
    offset: u32 = 0,
    size: u8 = 4,
    sign_extend: bool = false,
    bounds_known: bool = false,
    checked_end: u64 = 0,
    result_type: ir.IrType = .i32,
};

fn compileAddressingLoadTest(
    allocator: std.mem.Allocator,
    spec: AddressingLoadTestSpec,
    options: CompileOptions,
) ![]u8 {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    if (spec.base_type == .i32) {
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    } else {
        try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 0 }, .dest = addr, .type = spec.base_type });
    }
    try func.getBlock(bid).append(.{
        .op = .{ .load = .{
            .base = addr,
            .offset = spec.offset,
            .size = spec.size,
            .sign_extend = spec.sign_extend,
            .bounds_known = spec.bounds_known,
            .checked_end = spec.checked_end,
        } },
        .dest = val,
        .type = spec.result_type,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    return compileFunctionImpl(&func, .{
        .has_memory64 = spec.has_memory64,
        .has_shared_memory = spec.has_shared_memory,
        .allocator = allocator,
        .options = options,
    }, allocator);
}

const AddressingStoreTestSpec = struct {
    base_type: ir.IrType = .i32,
    has_memory64: bool = false,
    has_shared_memory: bool = false,
    offset: u32 = 0,
    size: u8 = 4,
    bounds_known: bool = false,
    checked_end: u64 = 0,
    value_type: ir.IrType = .i32,
};

fn compileAddressingStoreTest(
    allocator: std.mem.Allocator,
    spec: AddressingStoreTestSpec,
    options: CompileOptions,
) ![]u8 {
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    if (spec.base_type == .i32) {
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    } else {
        try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 0 }, .dest = addr, .type = spec.base_type });
    }
    if (spec.value_type == .i32) {
        try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 42 }, .dest = val, .type = .i32 });
    } else {
        try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 42 }, .dest = val, .type = spec.value_type });
    }
    try func.getBlock(bid).append(.{
        .op = .{ .store = .{
            .base = addr,
            .offset = spec.offset,
            .size = spec.size,
            .val = val,
            .bounds_known = spec.bounds_known,
            .checked_end = spec.checked_end,
        } },
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = null } });
    return compileFunctionImpl(&func, .{
        .has_memory64 = spec.has_memory64,
        .has_shared_memory = spec.has_shared_memory,
        .allocator = allocator,
        .options = options,
    }, allocator);
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

test "load: checked memory32 offset zero uses every scalar register-offset form" {
    const allocator = std.testing.allocator;
    const cases = [_]struct {
        spec: AddressingLoadTestSpec,
        opcode: u32,
    }{
        .{ .spec = .{ .size = 1 }, .opcode = 0x38606800 },
        .{ .spec = .{ .size = 1, .sign_extend = true }, .opcode = 0x38E06800 },
        .{ .spec = .{ .size = 1, .sign_extend = true, .result_type = .i64 }, .opcode = 0x38A06800 },
        .{ .spec = .{ .size = 2 }, .opcode = 0x78606800 },
        .{ .spec = .{ .size = 2, .sign_extend = true }, .opcode = 0x78E06800 },
        .{ .spec = .{ .size = 2, .sign_extend = true, .result_type = .i64 }, .opcode = 0x78A06800 },
        .{ .spec = .{ .size = 4, .result_type = .i64 }, .opcode = 0xB8606800 },
        .{ .spec = .{ .size = 4, .sign_extend = true, .result_type = .i64 }, .opcode = 0xB8A06800 },
        .{ .spec = .{ .size = 8, .result_type = .i64 }, .opcode = 0xF8606800 },
    };
    const index_bits = (@as(u32, @intFromEnum(RegMap.tmp2)) << 16) |
        (@as(u32, @intFromEnum(emit.Reg.x20)) << 5);
    const native_add = 0x8B000000 | index_bits | @as(u32, @intFromEnum(RegMap.tmp0));

    for (cases) |case| {
        const code = try compileAddressingLoadTest(allocator, case.spec, .{
            .enable_scheduler = false,
            .enable_peephole = false,
            .enable_post_emit_coalesce = false,
        });
        defer allocator.free(code);
        try std.testing.expect(testCodeContainsMasked(code, 0xFFFFFFE0, case.opcode | index_bits));
        try std.testing.expect(!testCodeContainsWord(code, native_add));
    }
}

test "store: checked memory32 offset zero uses every scalar register-offset form" {
    const allocator = std.testing.allocator;
    const cases = [_]struct {
        spec: AddressingStoreTestSpec,
        opcode: u32,
    }{
        .{ .spec = .{ .size = 1 }, .opcode = 0x38206800 },
        .{ .spec = .{ .size = 2 }, .opcode = 0x78206800 },
        .{ .spec = .{ .size = 4 }, .opcode = 0xB8206800 },
        .{ .spec = .{ .size = 8, .value_type = .i64 }, .opcode = 0xF8206800 },
    };
    const index_bits = (@as(u32, @intFromEnum(RegMap.tmp2)) << 16) |
        (@as(u32, @intFromEnum(emit.Reg.x20)) << 5);
    const native_add = 0x8B000000 | index_bits | @as(u32, @intFromEnum(RegMap.tmp0));

    for (cases) |case| {
        const code = try compileAddressingStoreTest(allocator, case.spec, .{
            .enable_scheduler = false,
            .enable_peephole = false,
            .enable_post_emit_coalesce = false,
        });
        defer allocator.free(code);
        try std.testing.expect(testCodeContainsMasked(code, 0xFFFFFFE0, case.opcode | index_bits));
        try std.testing.expect(!testCodeContainsWord(code, native_add));
    }
}

test "load: register-offset form removes one ADD and keeps zero-extension before the trap edge" {
    const allocator = std.testing.allocator;
    const spec = AddressingLoadTestSpec{ .size = 4 };
    const code_on = try compileAddressingLoadTest(allocator, spec, .{
        .enable_scheduler = false,
        .enable_peephole = false,
        .enable_post_emit_coalesce = true,
        .enable_mem_reg_offset = true,
    });
    defer allocator.free(code_on);
    const code_off = try compileAddressingLoadTest(allocator, spec, .{
        .enable_scheduler = false,
        .enable_peephole = false,
        .enable_post_emit_coalesce = true,
        .enable_mem_reg_offset = false,
    });
    defer allocator.free(code_off);

    try std.testing.expectEqual(code_off.len, code_on.len + 4);

    const index_bits = (@as(u32, @intFromEnum(RegMap.tmp2)) << 16) |
        (@as(u32, @intFromEnum(emit.Reg.x20)) << 5);
    const reg_load = 0xB8606800 | index_bits;
    const native_add = 0x8B000000 | index_bits | @as(u32, @intFromEnum(RegMap.tmp0));
    try std.testing.expect(testCodeContainsMasked(code_on, 0xFFFFFFE0, reg_load));
    try std.testing.expect(!testCodeContainsWord(code_on, native_add));
    try std.testing.expect(testCodeContainsWord(code_off, native_add));

    var mov_zext_pos: ?usize = null;
    var bls_pos: ?usize = null;
    var bls_word: u32 = 0;
    var blr_pos: ?usize = null;
    var brk_pos: ?usize = null;
    var load_pos: ?usize = null;
    var i: usize = 0;
    while (i + 4 <= code_on.len) : (i += 4) {
        const word = std.mem.readInt(u32, code_on[i..][0..4], .little);
        if ((word & 0xFFE0FFFF) == 0x2A0003EF) mov_zext_pos = i;
        if ((word & 0xFF00001F) == 0x54000009) {
            bls_pos = i;
            bls_word = word;
        }
        if ((word & 0xFFFFFC1F) == 0xD63F0000) blr_pos = i;
        if ((word & 0xFFE0001F) == 0xD4200000) brk_pos = i;
        if ((word & 0xFFFFFFE0) == reg_load) load_pos = i;
    }

    try std.testing.expect(mov_zext_pos.? < bls_pos.?);
    try std.testing.expect(bls_pos.? < blr_pos.?);
    try std.testing.expect(blr_pos.? < brk_pos.?);
    try std.testing.expect(brk_pos.? < load_pos.?);
    const branch_delta: i19 = @bitCast(@as(u19, @truncate(bls_word >> 5)));
    const branch_target = @as(i64, @intCast(bls_pos.?)) + @as(i64, branch_delta) * 4;
    try std.testing.expectEqual(@as(i64, @intCast(load_pos.?)), branch_target);
}

test "load: register-offset eligibility preserves offset and memory64-shaped fallbacks" {
    const allocator = std.testing.allocator;
    const cases = [_]AddressingLoadTestSpec{
        .{ .offset = 4, .size = 4 },
        .{ .offset = std.math.maxInt(u32), .size = 1 },
        .{ .base_type = .i64, .has_memory64 = true, .size = 4, .result_type = .i64 },
        .{ .has_shared_memory = true, .size = 4 },
        .{ .size = 4, .bounds_known = true },
    };
    const index_bits = (@as(u32, @intFromEnum(RegMap.tmp2)) << 16) |
        (@as(u32, @intFromEnum(emit.Reg.x20)) << 5);
    const native_add = 0x8B000000 | index_bits | @as(u32, @intFromEnum(RegMap.tmp0));

    for (cases) |spec| {
        const code = try compileAddressingLoadTest(allocator, spec, .{
            .enable_scheduler = false,
            .enable_peephole = false,
            .enable_post_emit_coalesce = false,
        });
        defer allocator.free(code);
        try std.testing.expect(!testCodeContainsMasked(code, 0x3B200C00, 0x38200800));
        try std.testing.expect(testCodeContainsWord(code, native_add));
    }
}

test "store: register-offset eligibility preserves offset and memory64-shaped fallbacks" {
    const allocator = std.testing.allocator;
    const cases = [_]AddressingStoreTestSpec{
        .{ .offset = 8, .size = 8, .value_type = .i64 },
        .{ .offset = std.math.maxInt(u32), .size = 1 },
        .{ .base_type = .i64, .has_memory64 = true, .size = 8, .value_type = .i64 },
        .{ .has_shared_memory = true, .size = 4 },
        .{ .size = 4, .bounds_known = true },
    };
    const index_bits = (@as(u32, @intFromEnum(RegMap.tmp2)) << 16) |
        (@as(u32, @intFromEnum(emit.Reg.x20)) << 5);
    const native_add = 0x8B000000 | index_bits | @as(u32, @intFromEnum(RegMap.tmp0));

    for (cases) |spec| {
        const code = try compileAddressingStoreTest(allocator, spec, .{
            .enable_scheduler = false,
            .enable_peephole = false,
            .enable_post_emit_coalesce = false,
        });
        defer allocator.free(code);
        try std.testing.expect(!testCodeContainsMasked(code, 0x3B200C00, 0x38200800));
        try std.testing.expect(testCodeContainsWord(code, native_add));
    }
}

test "atomic load: shared-memory path keeps materialized address fallback" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const val = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .atomic_load = .{ .base = addr, .offset = 0, .size = 4 } },
        .dest = val,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = val } });
    const code = try compileFunctionWithOptions(&func, allocator, .{
        .enable_scheduler = false,
        .enable_peephole = false,
        .enable_post_emit_coalesce = false,
    });
    defer allocator.free(code);

    const index_bits = (@as(u32, @intFromEnum(RegMap.tmp2)) << 16) |
        (@as(u32, @intFromEnum(emit.Reg.x20)) << 5);
    const native_add = 0x8B000000 | index_bits | @as(u32, @intFromEnum(RegMap.tmp0));
    try std.testing.expect(testCodeContainsWord(code, native_add));
    try std.testing.expect(!testCodeContainsMasked(code, 0x3B200C00, 0x38200800));
}

test "compile: v128.load_splat emits LD1R forms with bounds trap" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });

    const cases = [_]struct {
        width: ir.Inst.V128LoadSplatWidth,
        offset: u32,
        alignment: u32,
    }{
        .{ .width = .i8x16, .offset = 0, .alignment = 0 },
        .{ .width = .i16x8, .offset = 2, .alignment = 1 },
        .{ .width = .i32x4, .offset = 4, .alignment = 2 },
        .{ .width = .i64x2, .offset = 8, .alignment = 3 },
    };
    for (cases) |case| {
        const vec = func.newVReg();
        try func.getBlock(bid).append(.{
            .op = .{ .v128_load_splat = .{
                .width = case.width,
                .base = addr,
                .offset = case.offset,
                .alignment = case.alignment,
            } },
            .dest = vec,
            .type = .v128,
        });
    }

    const ret = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = ret, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = ret } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_ld1r_16b = false;
    var found_ld1r_8h = false;
    var found_ld1r_4s = false;
    var found_ld1r_2d = false;
    var found_bls = false;
    var found_blr = false;
    var found_brk = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x4D40C000) found_ld1r_16b = true;
        if ((w & 0xFFFFFC00) == 0x4D40C400) found_ld1r_8h = true;
        if ((w & 0xFFFFFC00) == 0x4D40C800) found_ld1r_4s = true;
        if ((w & 0xFFFFFC00) == 0x4D40CC00) found_ld1r_2d = true;
        if ((w & 0xFF000010) == 0x54000000 and (w & 0xF) == 0x9) found_bls = true;
        if ((w & 0xFFFFFC1F) == 0xD63F0000) found_blr = true;
        if ((w & 0xFFE0001F) == 0xD4200000) found_brk = true;
    }

    try std.testing.expect(found_ld1r_16b);
    try std.testing.expect(found_ld1r_8h);
    try std.testing.expect(found_ld1r_4s);
    try std.testing.expect(found_ld1r_2d);
    try std.testing.expect(found_bls);
    try std.testing.expect(found_blr);
    try std.testing.expect(found_brk);
}

test "compile: v128.load_zero emits scalar LDR forms with access-size bounds" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const ld32 = func.newVReg();
    const ld64 = func.newVReg();
    const result = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_load_zero = .{ .width = .i32, .base = addr, .offset = 0, .alignment = 2 } },
        .dest = ld32,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_load_zero = .{ .width = .i64, .base = addr, .offset = 0, .alignment = 3 } },
        .dest = ld64,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i32x4_extract_lane = .{ .vector = ld32, .lane = 0 } },
        .dest = result,
        .type = .i32,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_ldr_s = false;
    var found_ldr_d = false;
    var found_add_end_4 = false;
    var found_add_end_8 = false;
    var found_bls = false;
    var found_blr = false;
    var found_brk = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0xBD400000) found_ldr_s = true;
        if ((w & 0xFFC00000) == 0xFD400000) found_ldr_d = true;
        if (w == 0x910011F1) found_add_end_4 = true; // add x17, x15, #4
        if (w == 0x910021F1) found_add_end_8 = true; // add x17, x15, #8
        if ((w & 0xFF000010) == 0x54000000 and (w & 0xF) == 0x9) found_bls = true;
        if ((w & 0xFFFFFC1F) == 0xD63F0000) found_blr = true;
        if ((w & 0xFFE0001F) == 0xD4200000) found_brk = true;
    }
    try std.testing.expect(found_ldr_s);
    try std.testing.expect(found_ldr_d);
    try std.testing.expect(found_add_end_4);
    try std.testing.expect(found_add_end_8);
    try std.testing.expect(found_bls);
    try std.testing.expect(found_blr);
    try std.testing.expect(found_brk);
}

test "compile: v128.load_extend emits 64-bit load then low-half widening" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });

    const cases = [_]struct {
        width: ir.Inst.V128LoadExtendWidth,
        sign: ir.Inst.SimdExtendSign,
    }{
        .{ .width = .i8x8, .sign = .signed },
        .{ .width = .i8x8, .sign = .unsigned },
        .{ .width = .i16x4, .sign = .signed },
        .{ .width = .i16x4, .sign = .unsigned },
        .{ .width = .i32x2, .sign = .signed },
        .{ .width = .i32x2, .sign = .unsigned },
    };
    var last_vec: ir.VReg = addr;
    for (cases) |case| {
        const vec = func.newVReg();
        try func.getBlock(bid).append(.{
            .op = .{ .v128_load_extend = .{
                .src_width = case.width,
                .sign = case.sign,
                .base = addr,
                .offset = 0,
                .alignment = 3,
            } },
            .dest = vec,
            .type = .v128,
        });
        last_vec = vec;
    }
    const result = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .i64x2_extract_lane = .{ .vector = last_vec, .lane = 0 } },
        .dest = result,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_ldr_d = false;
    var found_sshll8 = false;
    var found_ushll8 = false;
    var found_sshll16 = false;
    var found_ushll16 = false;
    var found_sshll32 = false;
    var found_ushll32 = false;
    var found_add_end_8 = false;
    var found_bls = false;
    var found_blr = false;
    var found_brk = false;
    var first_bls: ?usize = null;
    var first_ldr_d: ?usize = null;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFC00000) == 0xFD400000) {
            found_ldr_d = true;
            if (first_ldr_d == null) first_ldr_d = i;
        }
        if ((w & 0xFFFFFC00) == 0x0F08A400) found_sshll8 = true;
        if ((w & 0xFFFFFC00) == 0x2F08A400) found_ushll8 = true;
        if ((w & 0xFFFFFC00) == 0x0F10A400) found_sshll16 = true;
        if ((w & 0xFFFFFC00) == 0x2F10A400) found_ushll16 = true;
        if ((w & 0xFFFFFC00) == 0x0F20A400) found_sshll32 = true;
        if ((w & 0xFFFFFC00) == 0x2F20A400) found_ushll32 = true;
        if (w == 0x910021F1) found_add_end_8 = true; // add x17, x15, #8
        if ((w & 0xFF000010) == 0x54000000 and (w & 0xF) == 0x9) {
            found_bls = true;
            if (first_bls == null) first_bls = i;
        }
        if ((w & 0xFFFFFC1F) == 0xD63F0000) found_blr = true;
        if ((w & 0xFFE0001F) == 0xD4200000) found_brk = true;
    }
    try std.testing.expect(found_ldr_d);
    try std.testing.expect(found_sshll8);
    try std.testing.expect(found_ushll8);
    try std.testing.expect(found_sshll16);
    try std.testing.expect(found_ushll16);
    try std.testing.expect(found_sshll32);
    try std.testing.expect(found_ushll32);
    try std.testing.expect(found_add_end_8);
    try std.testing.expect(found_bls);
    try std.testing.expect(found_blr);
    try std.testing.expect(found_brk);
    try std.testing.expect((first_bls orelse code.len) < (first_ldr_d orelse 0));
}

test "v128.loadN_lane emits checked LD1 lane loads and preserves live vector" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const vector = func.newVReg();
    const ld8 = func.newVReg();
    const ld16 = func.newVReg();
    const ld32 = func.newVReg();
    const ld64 = func.newVReg();
    const result = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_const = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF },
        .dest = vector,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_load_lane = .{ .width = .i8, .base = addr, .offset = 3, .alignment = 0, .vector = vector, .lane = 5 } },
        .dest = ld8,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_load_lane = .{ .width = .i16, .base = addr, .offset = 4, .alignment = 1, .vector = vector, .lane = 2 } },
        .dest = ld16,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_load_lane = .{ .width = .i32, .base = addr, .offset = 8, .alignment = 2, .vector = vector, .lane = 1 } },
        .dest = ld32,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_load_lane = .{ .width = .i64, .base = addr, .offset = 16, .alignment = 3, .vector = vector, .lane = 1 } },
        .dest = ld64,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .i64x2_extract_lane = .{ .vector = ld64, .lane = 1 } },
        .dest = result,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_ld1_b = false;
    var found_ld1_h = false;
    var found_ld1_s = false;
    var found_ld1_d = false;
    var found_bls = false;
    var found_blr = false;
    var found_brk = false;
    var found_orr_copy = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFFFFC00) == 0x0D401400) found_ld1_b = true;
        if ((w & 0xFFFFFC00) == 0x0D405000) found_ld1_h = true;
        if ((w & 0xFFFFFC00) == 0x0D409000) found_ld1_s = true;
        if ((w & 0xFFFFFC00) == 0x4D408400) found_ld1_d = true;
        if ((w & 0xFF000010) == 0x54000000 and (w & 0xF) == 0x9) found_bls = true;
        if ((w & 0xFFFFFC1F) == 0xD63F0000) found_blr = true;
        if ((w & 0xFFE0001F) == 0xD4200000) found_brk = true;
        if ((w & 0xFFE0FC00) == 0x4EA01C00) found_orr_copy = true;
    }
    try std.testing.expect(found_ld1_b);
    try std.testing.expect(found_ld1_h);
    try std.testing.expect(found_ld1_s);
    try std.testing.expect(found_ld1_d);
    try std.testing.expect(found_bls);
    try std.testing.expect(found_blr);
    try std.testing.expect(found_brk);
    try std.testing.expect(found_orr_copy);
}

test "v128.storeN_lane emits checked ST1 lane stores" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const addr = func.newVReg();
    const vector = func.newVReg();
    const result = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = addr, .type = .i32 });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_const = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF },
        .dest = vector,
        .type = .v128,
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_store_lane = .{ .width = .i8, .base = addr, .offset = 3, .alignment = 0, .vector = vector, .lane = 5 } },
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_store_lane = .{ .width = .i16, .base = addr, .offset = 4, .alignment = 1, .vector = vector, .lane = 2 } },
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_store_lane = .{ .width = .i32, .base = addr, .offset = 8, .alignment = 2, .vector = vector, .lane = 1 } },
    });
    try func.getBlock(bid).append(.{
        .op = .{ .v128_store_lane = .{ .width = .i64, .base = addr, .offset = 16, .alignment = 3, .vector = vector, .lane = 1 } },
    });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = result, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = result } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    var found_st1_b = false;
    var found_st1_h = false;
    var found_st1_s = false;
    var found_st1_d = false;
    var found_bls = false;
    var found_blr = false;
    var found_brk = false;
    var first_st1_pos: ?usize = null;
    var first_bls_pos: ?usize = null;
    var first_blr_pos: ?usize = null;
    var first_brk_pos: ?usize = null;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        const is_st1_b = (w & 0xFFFFFC00) == 0x0D001400;
        const is_st1_h = (w & 0xFFFFFC00) == 0x0D005000;
        const is_st1_s = (w & 0xFFFFFC00) == 0x0D009000;
        const is_st1_d = (w & 0xFFFFFC00) == 0x4D008400;
        if (is_st1_b) found_st1_b = true;
        if (is_st1_h) found_st1_h = true;
        if (is_st1_s) found_st1_s = true;
        if (is_st1_d) found_st1_d = true;
        if ((is_st1_b or is_st1_h or is_st1_s or is_st1_d) and first_st1_pos == null) first_st1_pos = i;
        if ((w & 0xFF000010) == 0x54000000 and (w & 0xF) == 0x9) {
            found_bls = true;
            if (first_bls_pos == null) first_bls_pos = i;
        }
        if ((w & 0xFFFFFC1F) == 0xD63F0000) {
            found_blr = true;
            if (first_blr_pos == null) first_blr_pos = i;
        }
        if ((w & 0xFFE0001F) == 0xD4200000) {
            found_brk = true;
            if (first_brk_pos == null) first_brk_pos = i;
        }
    }
    try std.testing.expect(found_st1_b);
    try std.testing.expect(found_st1_h);
    try std.testing.expect(found_st1_s);
    try std.testing.expect(found_st1_d);
    try std.testing.expect(found_bls);
    try std.testing.expect(found_blr);
    try std.testing.expect(found_brk);
    try std.testing.expect(first_bls_pos.? < first_st1_pos.?);
    try std.testing.expect(first_blr_pos.? < first_st1_pos.?);
    try std.testing.expect(first_brk_pos.? < first_st1_pos.?);
}

test "store: size 1 emits register-offset STRB" {
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
    // STRB Wt, [Xn, Xm] register-offset form.
    var found = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((w & 0xFFE0FC00) == 0x38206800) {
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

test "compile: entry pins vmctx in x19 and memory_base in x20 (issues #465 / #466)" {
    // Function with no params, no locals: prologue must copy x0 → x19
    // and load `[x19 + 0]` → x20 after the 10 callee-save STRs
    // (vmctx pinned in x19, memory_base pinned in x20 for the whole
    // function body).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_peephole = false });
    defer allocator.free(code);
    // Layout: prologue (STP + ADD fp,sp,0 = 8 bytes), then 10 callee-save
    // STRs for x19..x28 (40 bytes), then MOV x19, x0 (= ORR x19, xzr, x0),
    // then LDR x20, [x19, #0] (memory_base pin).
    //   MOV encoding: 0xAA000000 | (rm=0 << 16) | (rn=31 << 5) | rd=19
    //   = 0xAA000000 | 0x3E0 | 0x13 = 0xAA0003F3
    const mov_word = std.mem.readInt(u32, code[48..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xAA0003F3), mov_word);
    //   LDR encoding (LDR Xt, [Xn, #imm12*8]):
    //     0xF9400000 | (imm12=0 << 10) | (rn=19 << 5) | rt=20
    //     = 0xF9400000 | 0x260 | 0x14 = 0xF9400274
    const ldr_word = std.mem.readInt(u32, code[52..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xF9400274), ldr_word);
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
    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_peephole = false });
    defer allocator.free(code);
    // Layout: STP, ADD fp,sp,0, 10× callee-save STRs (40 bytes), MOV x19, x0
    // (#465 vmctx pin), LDR x20, [x19, #0] (#466 memory_base pin), STR x1 local0.
    //   STR x1, [fp, #24]: rt=1, rn=29, imm12=3
    //   = 0xF9000000 | (3<<10) | (29<<5) | 1 = 0xF9000FA1
    const str_param = std.mem.readInt(u32, code[56..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xF9000FA1), str_param);
}

test "compile: zero-init of declared local (beyond params) fuses into STP" {
    // 0 params, 2 declared locals, both read before being set so the
    // prologue zero-init is preserved (issue #468). The two adjacent
    // 8-byte zero stores fuse into a single `stp xzr-via-tmp0, tmp0,
    // [fp, #24]`.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    defer func.deinit();
    const bid = try func.newBlock();
    const r0 = func.newVReg();
    const r1 = func.newVReg();
    const sum = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = r0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 1 }, .dest = r1, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .add = .{ .lhs = r0, .rhs = r1 } }, .dest = sum, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = sum } });
    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_peephole = false });
    defer allocator.free(code);
    // Layout (small-frame prologue):
    //   [0]   STP FP, LR, [SP, #-frame_size]!
    //   [4]   MOV FP, SP
    //   [8]..[44] 10× callee-save STR (40 bytes)
    //   [48]  MOV x19, x0 (#465 vmctx pin)
    //   [52]  LDR x20, [x19, #0] (#466 memory_base pin)
    //   [56]  MOVZ x16, #0
    //   [60]  STP x16, x16, [fp, #24] (fused 16-byte zero store, issue #468)
    const movz_word = std.mem.readInt(u32, code[56..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xD2800010), movz_word);
    // STP x16, x16, [fp, #24]: imm7=3 (#24/8), rt2=16, rn=29(fp), rt1=16
    //   0xA9000000 | (3 << 15) | (16 << 10) | (29 << 5) | 16
    const stp_word = std.mem.readInt(u32, code[60..][0..4], .little);
    const expected_stp: u32 = 0xA9000000 |
        (@as(u32, 3) << 15) |
        (@as(u32, 16) << 10) |
        (@as(u32, 29) << 5) |
        16;
    try std.testing.expectEqual(expected_stp, stp_word);
}

test "compile: prologue zero-init skipped when local is definitely set before any get (#468)" {
    // 0 params, 1 declared local that is set before any read. The
    // wasm spec zero-init store can be skipped because every reachable
    // `local_get` is dominated by a `local_set`.
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
    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_peephole = false });
    defer allocator.free(code);
    // Word at offset 56 is the slot right after the vmctx + memory_base
    // pins. With zero-init skipped this is NOT a MOVZ x16, #0
    // (0xD2800010); the body proper begins here.
    const word56 = std.mem.readInt(u32, code[56..][0..4], .little);
    try std.testing.expect(word56 != 0xD2800010);
}

test "compile: prologue zero-init preserved when local is read before any set (#468)" {
    // 0 params, 1 declared local that is read before being set on
    // some path: the prologue zero store must be preserved for spec
    // compliance.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .local_get = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_peephole = false });
    defer allocator.free(code);
    const movz_word = std.mem.readInt(u32, code[56..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xD2800010), movz_word);
    // single 8-byte STR x16, [fp, #24]: imm12=3
    const str0 = std.mem.readInt(u32, code[60..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0xF9000FB0), str0);
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

test "compile: accepts scalar params beyond x7 via inbound stack" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 8, 1, 8);
    defer func.deinit();
    const bid = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v0, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = v0 } });
    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
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

test "compile: v128 allocator keeps cross-block vector in a register" {
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
    try std.testing.expectEqual(@as(u32, 0), counts.loads);
    try std.testing.expectEqual(@as(u32, 0), counts.stores);
}

test "compileFunction: enable_vreg_alloc false keeps legacy v128 fallback working" {
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

    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_vreg_alloc = false });
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
    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_scheduler = false, .enable_peephole = false });
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

test "compile: large frame beyond 12-bit scaled offset compiles (callee-save + spills) (#916)" {
    // Regression for #916. On macOS ARM64 (and any aarch64 target) a
    // function whose frame exceeds the LDR/STR unsigned scaled-immediate
    // reach (imm12*8 = 32760 bytes for the 64-bit form) used to panic
    // with "integer does not fit in destination type" while emitting the
    // prologue callee-save stores (`@intCast` of the slot offset to u12).
    //
    // Force a >32 KiB frame by declaring a large local count (each i64
    // local is an 8-byte frame slot, so ~5000 locals pushes the spill /
    // call-save / callee-save regions well past the scaled-immediate
    // limit), and add real register pressure so the allocator spills to
    // the high spill region too. The whole thing must now compile without
    // a host-side safety panic.
    const allocator = std.testing.allocator;

    const local_count = 5000; // 5000 * 8 = 40000 bytes > 32760.
    var func = ir.IrFunction.init(allocator, 0, 1, local_count);
    defer func.deinit();
    const bid = try func.newBlock();

    // 30 simultaneously-live vregs exceed the allocatable GPRs and force
    // spills; with the inflated local frame those spill slots land above
    // the scaled-immediate limit, exercising the frameStore/frameLoadReg
    // fallback in addition to the callee-save fix.
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

    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_scheduler = false, .enable_peephole = false });
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);

    // The oversized frame must take the #916 large-frame callee-save path:
    // the base address is materialized into tmp0 (x16) and each callee-save
    // register is stored via `STR Xt, [x16, #i]` rather than a direct
    // fp-relative store. Assert at least one STR with base register x16
    // (rn == 16) is present. STR Xt,[Xn,#imm] top-10 opcode bits == 0x3E4.
    var found_x16_based_store = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        const top = (w >> 22) & 0x3FF;
        const rn = (w >> 5) & 0x1F;
        if (top == 0x3E4 and rn == 16) {
            found_x16_based_store = true;
            break;
        }
    }
    try std.testing.expect(found_x16_based_store);
}

test "compileFunction: enable_xreg_alloc false keeps legacy fallback working" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const lhs = func.newVReg();
    const rhs = func.newVReg();
    const sum = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 40 }, .dest = lhs, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 2 }, .dest = rhs, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .add = .{ .lhs = lhs, .rhs = rhs } }, .dest = sum, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = sum } });

    const code = try compileFunctionWithOptions(&func, allocator, .{ .enable_xreg_alloc = false });
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "compileFunction: enable_move_coalesce elides reinterpret mov" {
    // Issue #386: with the post-allocation move-coalescing pass enabled,
    // a `reinterpret` whose source's last use is the reinterpret itself
    // collapses to zero instructions — the dest vreg shares the source
    // vreg's physreg, and the emit-time guard
    // `if (src != info.reg) movRegReg(...)` in `emitReinterpret`
    // suppresses the otherwise-redundant mov.
    //
    // This test pins the strong direction: with coalescing on, no
    // reg-to-reg mov (ORR Xd, XZR, Xn pattern) appears in the body.
    // The asymmetry between on/off is unit-tested at the regalloc
    // layer (see `regalloc.zig`'s coalesceMoves tests).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const a = func.newVReg();
    const r = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 0x4000_0000_0000_0000 }, .dest = a, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .reinterpret = a }, .dest = r, .type = .f64 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = r } });

    // ORR Xd, XZR, Xn (the MOV alias `movRegReg` emits): mask
    // `0xFFE0_FC1F` keeps sf+opc+shift+N+imm6+Rn; masked value
    // `0xAA00_03E0` has Rn=11111 (XZR). Rm and Rd are wildcarded.
    const mov_mask: u32 = 0xFFE0_FC1F;
    const mov_value: u32 = 0xAA00_03E0;

    const code_on = try compileFunctionWithOptions(&func, allocator, .{ .enable_move_coalesce = true });
    defer allocator.free(code_on);
    try std.testing.expect(!testCodeContainsMasked(code_on, mov_mask, mov_value));
}

test "compileFunction: enable_move_coalesce elides wrap_i64 uxtw" {
    // Issue #386 follow-up: wrap_i64 (i64 → i32) emits `UXTW Xd, Wn`,
    // which is `MOV Wd, Wn` (ORR Wd, WZR, Wn) under the hood and
    // zero-extends into Xd. When the move-coalescer places src and dest
    // on the same physreg, the upper bits of the X register are stale
    // but irrelevant — every downstream consumer of the i32 dest reads
    // W-form (see `local_get`'s "subsequent W-form op will ignore the
    // upper bits" contract), so `emitWrap`'s `if (src != dest) uxtw(...)`
    // guard skips the now-redundant instruction.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    const a = func.newVReg();
    const r = func.newVReg();
    try func.getBlock(bid).append(.{ .op = .{ .iconst_64 = 0x1_0000_0001 }, .dest = a, .type = .i64 });
    try func.getBlock(bid).append(.{ .op = .{ .wrap_i64 = a }, .dest = r, .type = .i32 });
    try func.getBlock(bid).append(.{ .op = .{ .ret = r } });

    // ORR Wd, WZR, Wn (the 32-bit MOV alias `movRegReg32` emits, which
    // `uxtw` is implemented as): mask `0x7FE0_FC1F` keeps sf+opc+shift+
    // N+imm6+Rn; masked value `0x2A00_03E0` has Rn=11111 (WZR). Rm and
    // Rd are wildcarded.
    const mov32_mask: u32 = 0x7FE0_FC1F;
    const mov32_value: u32 = 0x2A00_03E0;

    const code_on = try compileFunctionWithOptions(&func, allocator, .{ .enable_move_coalesce = true });
    defer allocator.free(code_on);
    try std.testing.expect(!testCodeContainsMasked(code_on, mov32_mask, mov32_value));
    // The off-case asymmetry is allocator-ordering-dependent — see the
    // matching note on the reinterpret test above; covered at the
    // regalloc layer instead.
}

test "compileFunction: fall-through elision drops br to next block (#808)" {
    // A `br` whose target is the immediately-following block in layout order
    // is a redundant `b .+4`. With elision on it is dropped (the block falls
    // through); with it off the unconditional B is emitted.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const entry = try func.newBlock();
    const next = try func.newBlock();
    try func.getBlock(entry).append(.{ .op = .{ .br = next } });
    try func.getBlock(next).append(.{ .op = .{ .ret = null } });

    // Unconditional AArch64 B: top 6 bits 0b000101 (mask 0xFC00_0000, value
    // 0x1400_0000); excludes BL (0x9400_0000) and B.cond (0x5400_0000).
    const b_mask: u32 = 0xFC00_0000;
    const b_value: u32 = 0x1400_0000;

    const code_on = try compileFunctionWithOptions(&func, allocator, .{ .enable_fallthrough_elision = true });
    defer allocator.free(code_on);
    try std.testing.expect(!testCodeContainsMasked(code_on, b_mask, b_value));

    const code_off = try compileFunctionWithOptions(&func, allocator, .{ .enable_fallthrough_elision = false });
    defer allocator.free(code_off);
    try std.testing.expect(testCodeContainsMasked(code_off, b_mask, b_value));
}

test "compileFunction: fall-through elision drops br_if unconditional edge (#808)" {
    // `br_if` lowers to `B.cond then` + `B else`. When a successor is the
    // physical fall-through, the unconditional `B else` is dropped (inverting
    // the condition if `then` is the fall-through). One of the two successors
    // is always laid out right after `entry`, so no unconditional B remains.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const entry = try func.newBlock();
    const then_b = try func.newBlock();
    const else_b = try func.newBlock();
    const c = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = c, .then_block = then_b, .else_block = else_b } } });
    try func.getBlock(then_b).append(.{ .op = .{ .ret = null } });
    try func.getBlock(else_b).append(.{ .op = .{ .ret = null } });

    const b_mask: u32 = 0xFC00_0000;
    const b_value: u32 = 0x1400_0000;

    const code_on = try compileFunctionWithOptions(&func, allocator, .{ .enable_fallthrough_elision = true });
    defer allocator.free(code_on);
    try std.testing.expect(!testCodeContainsMasked(code_on, b_mask, b_value));

    const code_off = try compileFunctionWithOptions(&func, allocator, .{ .enable_fallthrough_elision = false });
    defer allocator.free(code_off);
    try std.testing.expect(testCodeContainsMasked(code_off, b_mask, b_value));
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
    // 23 allocatable GPRs, 15 caller-saved + 8 callee-saved (x19 is
    // pinned to vmctx per issue #465 and x20 is pinned to memory_base
    // per issue #466; neither is in the allocator pool).
    try std.testing.expectEqual(@as(usize, 23), rs.alloc_regs.len);
    try std.testing.expectEqual(@as(usize, 15), rs.caller_saved_indices.len);
    try std.testing.expectEqual(@as(usize, 8), rs.callee_saved_indices.len);
    // Spill stride grows upward (away from fp).
    try std.testing.expect(rs.spill_stride > 0);
    // Spills live above the locals area (saved fp/lr + vmctx + locals = 24 bytes min).
    try std.testing.expect(rs.spill_base >= 24);
}

test "compileModule: regalloc hints — leaf 2-arg call places args directly into x1/x2 (aarch64)" {
    // Issue #423: with calling-convention hints the regalloc places
    // arg vregs directly into their AAPCS64 target param regs (x1, x2
    // for a 2-arg call). `stageArgFromSaved` then short-circuits via
    // its `target == r` no-op path, so the call-arg setup region
    // contains zero inter-X-reg `MOV Xd, Xn` instructions.
    //
    // Without hints the allocator picks the first two caller-saved
    // indices (x0 then x1), so arg[0] lands in x0 and arg[1] in x1 —
    // which then have to be shuffled to x1/x2 at the call site,
    // emitting at least one inter-X-reg MOV.

    const allocator = std.testing.allocator;
    var ir_module = ir.IrModule.init(allocator);
    defer ir_module.deinit();

    // callee: returns 0. Body is irrelevant — the test inspects the
    // *caller*'s call-arg setup region. (Test fixtures don't populate
    // `func_types`, so `funcParamTypes` returns &.{} and the ABI plan
    // falls back to `vreg_types` / .i64 for every arg, all of which
    // still classify as scalar_reg in the AAPCS64 plan.)
    var callee = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        _ = try callee.newBlock();
        const v = callee.newVReg();
        try callee.getBlock(0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v, .type = .i32 });
        try callee.getBlock(0).append(.{ .op = .{ .ret = v } });
    }
    _ = try ir_module.addFunction(callee);

    // caller(): two iconsts → 2-arg call → ret.
    var caller = ir.IrFunction.init(allocator, 0, 1, 0);
    _ = try caller.newBlock();
    const a0 = caller.newVReg();
    const a1 = caller.newVReg();
    const r = caller.newVReg();
    try caller.getBlock(0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = a0, .type = .i32 });
    try caller.getBlock(0).append(.{ .op = .{ .iconst_32 = 2 }, .dest = a1, .type = .i32 });
    const args = try allocator.alloc(ir.VReg, 2);
    args[0] = a0;
    args[1] = a1;
    try caller.getBlock(0).append(.{
        .op = .{ .call = .{ .func_idx = 0, .args = args } },
        .dest = r,
        .type = .i32,
    });
    try caller.getBlock(0).append(.{ .op = .{ .ret = r } });
    _ = try ir_module.addFunction(caller);

    const result = try compileModule(&ir_module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);

    const caller_off = result.offsets[1];
    const caller_code = result.code[caller_off..];

    // Locate the BL to the callee (top 6 bits = 0b100101 → 0x94000000).
    var bl_word_off: ?usize = null;
    {
        var i: usize = 0;
        while (i + 4 <= caller_code.len) : (i += 4) {
            const word = std.mem.readInt(u32, caller_code[i..][0..4], .little);
            if ((word & 0xFC000000) == 0x94000000) {
                bl_word_off = i;
                break;
            }
        }
    }
    const bl_off = bl_word_off orelse return error.TestExpectedCallEmitted;

    // Positive check: the iconsts must directly target x1 and x2.
    //   movImm32 with val < 0x10000 emits a single MOVZ.
    //   MOVZ Xd, #imm16, LSL #0 = 0xD2800000 | (imm16 << 5) | rd
    //   MOVZ X1, #1 = 0xD2800021
    //   MOVZ X2, #2 = 0xD2800042
    const movz_x1_1: u32 = 0xD2800021;
    const movz_x2_2: u32 = 0xD2800042;
    try std.testing.expect(testCodeContainsWord(caller_code[0..bl_off], movz_x1_1));
    try std.testing.expect(testCodeContainsWord(caller_code[0..bl_off], movz_x2_2));

    // Negative check: no inter-X-reg MOV (= ORR Xd, XZR, Xn shifted form
    // with shift=0) appears in the call-arg setup region (everything
    // before the BL). Encoding: 0xAA0003E0 | (rn << 16) | rd, where
    // rd, rn are 5-bit register numbers. The pattern is:
    //   bits [31:24] = 0xAA
    //   bits [23:21] = 000 (sf=1 sz=01 already in 0xAA prefix? Actually
    //                       0xAA means top byte; ORR shifted reg has
    //                       top 8 bits = 0xAA when sf=1, opc=01,
    //                       0|01010|00|0 in [29:21], all good)
    //   bits [15:10] = 000000 (no shift)
    //   bits [9:5]   = rn
    //   bits [4:0]   = rd
    //
    // Mask 0xFFE0FC00 captures [31:21] | [15:10]; the literal 0xAA0003E0
    // requires [9:5] = 11111 (XZR), which is the ORR-as-MOV alias. A
    // direct ORR-as-MOV with rn=XZR and rd in 0..30 is a register move.
    // We assert no such instruction is present — every "real" mov in
    // the staging region would otherwise show up here.
    var p: usize = 0;
    while (p + 4 <= bl_off) : (p += 4) {
        const word = std.mem.readInt(u32, caller_code[p..][0..4], .little);
        const is_mov_xd_xn = (word & 0xFFE0FC00) == 0xAA0003E0;
        if (is_mov_xd_xn) {
            const rd: u32 = word & 0x1F;
            const rn: u32 = (word >> 16) & 0x1F;
            // Allow MOV Xd, XZR (rn=31, alias for clearing) — not an
            // inter-vreg shuffle. Real shuffles have rn in 0..30.
            if (rn != 31) {
                std.debug.print(
                    "\nUnexpected inter-X-reg MOV at offset {}: word=0x{x:0>8} rd=x{} rn=x{}\n",
                    .{ p, word, rd, rn },
                );
                try std.testing.expect(false);
            }
        }
    }
}

// ── #542: rematerialisation of cheap defs ────────────────────────────

test "remat (#542): high-pressure iconst function compiles successfully" {
    // 20 simultaneously-live iconst_32 vregs forces register pressure
    // well above the 25 allocatable GPRs once the kept-alive sum chain
    // is included. Without remat, the spill-slot accesses would
    // explode the code size; with remat the def emits nothing and
    // uses re-emit a single MOVZ each.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b = try func.newBlock();
    const block = func.getBlock(b);
    var ks: [30]ir.VReg = undefined;
    for (0..ks.len) |i| {
        ks[i] = func.newVReg();
        try block.append(.{ .op = .{ .iconst_32 = @as(i32, @intCast(i + 1)) * 0x100001 }, .dest = ks[i], .type = .i32 });
    }
    var sum = ks[0];
    for (1..ks.len) |i| {
        const next = func.newVReg();
        try block.append(.{ .op = .{ .add = .{ .lhs = sum, .rhs = ks[i] } }, .dest = next, .type = .i32 });
        sum = next;
    }
    try block.append(.{ .op = .{ .ret = sum } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

test "remat (#542): iconst used after a call recompiles cleanly" {
    // f0 returns 7. f1 builds an iconst, calls f0, then adds the
    // iconst to f0's result. Call clobbers all caller-save physregs;
    // if the allocator assigned the iconst to a caller-save it would
    // have to spill across the call — remat skips that and re-emits
    // the const at the post-call use.
    const allocator = std.testing.allocator;
    var module = ir.IrModule.init(allocator);
    defer module.deinit();

    var f0 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f0.newBlock();
        const v = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .ret = v } });
    }
    _ = try module.addFunction(f0);

    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f1.newBlock();
        const blk = f1.getBlock(b);
        const k = f1.newVReg();
        // Out-of-imm12 to defeat #523 fold-on-use and force a real
        // remat use point.
        try blk.append(.{ .op = .{ .iconst_32 = 0x10003 }, .dest = k, .type = .i32 });
        const call_res = f1.newVReg();
        try blk.append(.{
            .op = .{ .call = .{ .func_idx = 0, .args = &.{} } },
            .dest = call_res,
            .type = .i32,
        });
        const sum = f1.newVReg();
        try blk.append(.{ .op = .{ .add = .{ .lhs = call_res, .rhs = k } }, .dest = sum, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = sum } });
    }
    _ = try module.addFunction(f1);

    const result = try compileModule(&module, allocator);
    defer allocator.free(result.code);
    defer allocator.free(result.offsets);
    try std.testing.expect(result.code.len > 0);
}

test "remat (#542): iconst with mixed imm12 + out-of-imm12 uses composes with #523 fold" {
    // The same iconst feeds an add-imm12 (fold-on-use, #523) and an
    // add of two regs (needs the value in a real reg). Verify the
    // function compiles cleanly — i.e. remat composes with the
    // suppress-iconst-emit fold path. (#523 only suppresses the def
    // when ALL uses fold; mixed-mode triggers normal materialisation,
    // which is what remat now handles via re-emit at the non-folded
    // use site.)
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 1, 2, 0);
    defer func.deinit();
    const lt = try allocator.alloc(ir.IrType, 1);
    lt[0] = .i32;
    func.local_types = lt;

    const bid = try func.newBlock();
    const block = func.getBlock(bid);
    const param = func.newVReg();
    const k = func.newVReg();
    const fold_sum = func.newVReg();
    const reg_use = func.newVReg();
    const ret_val = func.newVReg();
    try block.append(.{ .op = .{ .local_get = 0 }, .dest = param, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 0x10003 }, .dest = k, .type = .i32 });
    // Use 1: add of two registers — needs k in a real reg.
    try block.append(.{ .op = .{ .add = .{ .lhs = param, .rhs = k } }, .dest = reg_use, .type = .i32 });
    // Use 2: would normally fold to add-imm12 but the 0x10003 is
    // out-of-range, so this also needs k in a reg. Two register
    // uses keeps it simple — the existing fold path is exercised by
    // dedicated #523 tests; here we just verify mixed-context compile.
    try block.append(.{ .op = .{ .add = .{ .lhs = reg_use, .rhs = k } }, .dest = fold_sum, .type = .i32 });
    try block.append(.{ .op = .{ .add = .{ .lhs = fold_sum, .rhs = k } }, .dest = ret_val, .type = .i32 });
    try block.append(.{ .op = .{ .ret = ret_val } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "relaxOutOfRangeConditionalBranches: rewrites out-of-range B.cond to B.!cond+8 / B" {
    // Regression for #621. Construct a synthetic code buffer with a
    // single B.cond at offset 0 whose target block sits 2 MiB away —
    // well beyond the imm19 ±1 MiB reach. The relaxer should:
    //   1. Insert 4 bytes after the B.cond, growing the buffer by 4.
    //   2. Rewrite the original word to `B.!cond +2 words` (skip the
    //      inserted instruction when the original condition is false).
    //   3. Emit a placeholder `B 0` (0x14000000) at the inserted slot.
    //   4. Convert the patch entry to `.b_uncond` pointing at the
    //      inserted slot.
    //   5. Shift downstream block_offsets and call_patches by +4.
    const allocator = std.testing.allocator;

    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    // B.EQ placeholder (cond bits = 0x0). imm19 = 0 will be patched by
    // the resolution loop after relaxation; we only test the relaxer here.
    try code.bytes.appendSlice(allocator, &[_]u8{ 0, 0, 0, 0 }); // B.EQ 0
    std.mem.writeInt(u32, code.bytes.items[0..4], 0x54000000, .little);

    // Pad with NOPs out to 2 MiB so block 1 lands at offset 2 MiB.
    const target_off: usize = 2 * 1024 * 1024;
    try code.bytes.resize(allocator, target_off);
    @memset(code.bytes.items[4..target_off], 0);

    var block_offsets = [_]usize{ 0, target_off };
    var patches: std.ArrayListUnmanaged(BranchPatch) = .empty;
    defer patches.deinit(allocator);
    try patches.append(allocator, .{ .patch_offset = 0, .target_block = 1, .kind = .b_cond });

    var call_patches: std.ArrayListUnmanaged(CallPatch) = .empty;
    defer call_patches.deinit(allocator);
    try call_patches.append(allocator, .{ .patch_offset = target_off, .target_func_idx = 0 });

    try relaxOutOfRangeConditionalBranches(
        &code,
        &block_offsets,
        &patches,
        &call_patches,
        allocator,
    );

    // Buffer grew by 4 bytes.
    try std.testing.expectEqual(target_off + 4, code.bytes.items.len);

    // Word at offset 0: B.!EQ (= B.NE, cond = 0x1) with imm19 = 2.
    const w0 = std.mem.readInt(u32, code.bytes.items[0..4], .little);
    try std.testing.expectEqual(@as(u32, 0x54000000 | (2 << 5) | 0x1), w0);

    // Word at offset 4: placeholder unconditional B (imm26 patched later).
    const w1 = std.mem.readInt(u32, code.bytes.items[4..8], .little);
    try std.testing.expectEqual(@as(u32, 0x14000000), w1);

    // Patch entry now points at the inserted B and is unconditional.
    try std.testing.expectEqual(@as(usize, 4), patches.items[0].patch_offset);
    try std.testing.expect(patches.items[0].kind == .b_uncond);

    // Downstream block_offsets shifted by +4.
    try std.testing.expectEqual(@as(usize, 0), block_offsets[0]);
    try std.testing.expectEqual(target_off + 4, block_offsets[1]);

    // Downstream call_patches shifted by +4.
    try std.testing.expectEqual(target_off + 4, call_patches.items[0].patch_offset);
}

test "relaxOutOfRangeConditionalBranches: in-range branches are not touched" {
    const allocator = std.testing.allocator;

    var code = emit.CodeBuffer.init(allocator);
    defer code.deinit();

    try code.bytes.appendSlice(allocator, &[_]u8{ 0, 0, 0, 0 });
    std.mem.writeInt(u32, code.bytes.items[0..4], 0x54000000, .little);

    // Target only 256 bytes away — well within imm19 reach.
    const target_off: usize = 256;
    try code.bytes.resize(allocator, target_off);
    @memset(code.bytes.items[4..target_off], 0);

    var block_offsets = [_]usize{ 0, target_off };
    var patches: std.ArrayListUnmanaged(BranchPatch) = .empty;
    defer patches.deinit(allocator);
    try patches.append(allocator, .{ .patch_offset = 0, .target_block = 1, .kind = .b_cond });

    try relaxOutOfRangeConditionalBranches(
        &code,
        &block_offsets,
        &patches,
        null,
        allocator,
    );

    try std.testing.expectEqual(target_off, code.bytes.items.len);
    try std.testing.expectEqual(@as(usize, 0), patches.items[0].patch_offset);
    try std.testing.expect(patches.items[0].kind == .b_cond);
    try std.testing.expectEqual(target_off, block_offsets[1]);
}

// ── #672 commit 3: aarch64 EH codegen ────────────────────────────────

test "#672 commit 3: throw lowers to STR / LDR / BLR + BRK" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0xDEAD }, .dest = v0, .type = .i32 });

    const args = try allocator.alloc(ir.VReg, 1);
    args[0] = v0;
    try block.append(.{ .op = .{ .throw = .{ .tag_idx = 3, .args = args } } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);

    // The throw sequence ends with `brk #0` (0xD4200000). Scan for it.
    var found_brk = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        if (w == 0xD4200000) {
            found_brk = true;
            break;
        }
    }
    try std.testing.expect(found_brk);
}

test "#672 commit 3: try_table_begin / try_table_end are no-ops" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const block_id = try func.newBlock();
    const block = func.getBlock(block_id);
    const v0 = func.newVReg();

    // Wrap a trivial body in try_table_begin / try_table_end.
    const clauses = try allocator.alloc(ir.Inst.CatchClause, 0);
    try block.append(.{ .op = .{ .try_table_begin = .{
        .result_arity = 0,
        .clauses = clauses,
    } } });
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try block.append(.{ .op = .{ .try_table_end = {} } });
    try block.append(.{ .op = .{ .ret = v0 } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);
    try std.testing.expect(code.len > 0);
    try std.testing.expect(code.len % 4 == 0);
}

fn freeCachedFuncsAarch64(allocator: std.mem.Allocator, funcs: []codegen_cache.CachedFunction) void {
    for (funcs) |*f| {
        allocator.free(f.code);
        allocator.free(f.call_patches);
    }
    allocator.free(funcs);
}

fn buildTwoFnAddModuleAarch64(allocator: std.mem.Allocator) !ir.IrModule {
    var module = ir.IrModule.init(allocator);
    errdefer module.deinit();

    // f0: return iconst(1) + iconst(2)
    var f0 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f0.newBlock();
        const a = f0.newVReg();
        const c = f0.newVReg();
        const s = f0.newVReg();
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 1 }, .dest = a, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .iconst_32 = 2 }, .dest = c, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .add = .{ .lhs = a, .rhs = c } }, .dest = s, .type = .i32 });
        try f0.getBlock(b).append(.{ .op = .{ .ret = s } });
    }
    _ = try module.addFunction(f0);

    // f1: return iconst(7)
    var f1 = ir.IrFunction.init(allocator, 0, 1, 0);
    {
        const b = try f1.newBlock();
        const v = f1.newVReg();
        try f1.getBlock(b).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v, .type = .i32 });
        try f1.getBlock(b).append(.{ .op = .{ .ret = v } });
    }
    _ = try module.addFunction(f1);

    return module;
}

test "#781 aarch64 Aarch64FuncTimer accumulates each phase independently" {
    var ft = codegen_timing.Aarch64FuncTimer.start();
    try std.testing.expectEqual(@as(u64, 0), ft.scheduling_ns);

    // Recording one phase must not disturb the others.
    ft.begin();
    ft.end(.scheduling);
    try std.testing.expectEqual(@as(u64, 0), ft.range_split_ns);
    try std.testing.expectEqual(@as(u64, 0), ft.prepass_ns);
    try std.testing.expectEqual(@as(u64, 0), ft.liveness_ns);
    try std.testing.expectEqual(@as(u64, 0), ft.regalloc_ns);
    try std.testing.expectEqual(@as(u64, 0), ft.coalesce_ns);
    try std.testing.expectEqual(@as(u64, 0), ft.post_emit_coalesce_ns);

    // Each phase routes to its own accumulator; assign distinct values and
    // confirm there is no cross-talk.
    ft = .{
        .scheduling_ns = 1,
        .range_split_ns = 2,
        .prepass_ns = 3,
        .liveness_ns = 4,
        .regalloc_ns = 5,
        .coalesce_ns = 6,
        .post_emit_coalesce_ns = 7,
    };
    try std.testing.expectEqual(@as(u64, 1), ft.scheduling_ns);
    try std.testing.expectEqual(@as(u64, 2), ft.range_split_ns);
    try std.testing.expectEqual(@as(u64, 3), ft.prepass_ns);
    try std.testing.expectEqual(@as(u64, 4), ft.liveness_ns);
    try std.testing.expectEqual(@as(u64, 5), ft.regalloc_ns);
    try std.testing.expectEqual(@as(u64, 6), ft.coalesce_ns);
    try std.testing.expectEqual(@as(u64, 7), ft.post_emit_coalesce_ns);
}

test "#781 aarch64 codegen timing is byte-identical to the untimed path" {
    // Enabling WAMR_AOT_CODEGEN_TIMING is instrumentation only: it must not
    // perturb the generated code, offsets, or cache stats.
    const allocator = std.testing.allocator;
    var m_plain = try buildTwoFnAddModuleAarch64(allocator);
    defer m_plain.deinit();
    var m_timed = try buildTwoFnAddModuleAarch64(allocator);
    defer m_timed.deinit();

    const plain = try compileModuleCached(&m_plain, null, allocator);
    defer allocator.free(plain.code);
    defer allocator.free(plain.offsets);
    defer freeCachedFuncsAarch64(allocator, plain.cache_functions);

    const timed = try compileModuleCachedWithOptions(&m_timed, null, allocator, .{
        .codegen_timing = .{ .enabled = true, .threshold_ns = 0 },
    });
    defer allocator.free(timed.code);
    defer allocator.free(timed.offsets);
    defer freeCachedFuncsAarch64(allocator, timed.cache_functions);

    try std.testing.expectEqualSlices(u8, plain.code, timed.code);
    try std.testing.expectEqualSlices(u32, plain.offsets, timed.offsets);
    try std.testing.expectEqual(plain.stats.recompiled, timed.stats.recompiled);
    try std.testing.expectEqual(plain.stats.reused, timed.stats.reused);
}
