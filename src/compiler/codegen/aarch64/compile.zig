//! AArch64 IR Compiler
//!
//! Walks IR functions and emits AArch64 machine code via CodeBuffer.
//! Uses the same pattern as the x86-64 backend with a linear-scan
//! register allocator mapping VRegs to physical registers.

const std = @import("std");
const ir = @import("../../ir/ir.zig");
const emit = @import("emit.zig");

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
        .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27, .x28,
    };

    /// Non-allocatable scratch registers usable by any handler.
    /// Must never appear in `scratch_regs`.
    pub const tmp0: emit.Reg = .x16;
    pub const tmp1: emit.Reg = .x17;
    pub const tmp2: emit.Reg = .x15;

    entries: std.AutoHashMap(ir.VReg, Location),
    reg_used: [scratch_regs.len]bool = [_]bool{false} ** scratch_regs.len,
    next_stack_offset: u32 = 0,
    /// Byte offset from FP where the first spill slot lives.
    spill_base: u32 = 0,
    /// Maximum bytes reserved for spills (beyond which `assign` errors).
    spill_capacity: u32 = 0,

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
        for (scratch_regs, 0..) |r, i| {
            if (!self.reg_used[i]) {
                self.reg_used[i] = true;
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

    fn get(self: *const RegMap, vreg: ir.VReg) ?Location {
        return self.entries.get(vreg);
    }

    /// FP-relative byte offset of a spill slot (scaled by 8 for LDR/STR X).
    fn spillOffsetScaled(self: *const RegMap, slot_byte_off: u32) u12 {
        return @intCast((self.spill_base + slot_byte_off) / 8);
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
    allocator: std.mem.Allocator,
};

/// Pre-computed info for a fused MADD/MSUB: `dest = addend ± mul_lhs * mul_rhs`.
const FmaInfo = struct {
    mul_lhs: ir.VReg,
    mul_rhs: ir.VReg,
    addend: ir.VReg,
    is_sub: bool,
};

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
    return compileFunctionImpl(func, .{ .allocator = allocator }, allocator);
}

pub fn compileFunctionImpl(
    func: *const ir.IrFunction,
    ctx: FuncCompileCtx,
    allocator: std.mem.Allocator,
) ![]u8 {
    // Phase 1b: stack args beyond x7 aren't supported yet.
    if (func.param_count > 7) return error.TooManyParams;

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
    const spill_capacity: u32 = blk: {
        const vreg_slots = func.next_vreg + 16;
        break :blk @intCast(vreg_slots * 8);
    };
    const call_save_base: u32 = spill_base + spill_capacity;
    const call_save_size: u32 = 128;
    const callee_save_base: u32 = call_save_base + call_save_size;
    const callee_save_count: u32 = 10; // x19..x28
    const callee_save_size: u32 = callee_save_count * 8;
    const raw_frame = callee_save_base + callee_save_size;
    const frame_size: u32 = (raw_frame + 15) & ~@as(u32, 15);

    var reg_map = RegMap.init(allocator, spill_base, spill_capacity);
    defer reg_map.deinit();

    var fctx = ctx;
    fctx.call_save_base = call_save_base;
    fctx.callee_save_base = callee_save_base;

    // Collect iconst_* values so emitBinOp etc can fold small constants
    // into immediate-form instructions. Scan is cheap — one pass over IR.
    var const_vals = std.AutoHashMap(ir.VReg, i64).init(allocator);
    defer const_vals.deinit();
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
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
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            switch (inst.op) {
                // Binary ops: two vreg reads.
                .add, .sub, .mul, .@"and", .@"or", .xor,
                .div_s, .div_u, .rem_s, .rem_u,
                .shl, .shr_s, .shr_u, .rotl, .rotr,
                .eq, .ne, .lt_s, .lt_u, .gt_s, .gt_u,
                .le_s, .le_u, .ge_s, .ge_u,
                .f_eq, .f_ne, .f_lt, .f_gt, .f_le, .f_ge,
                => |b| {
                    try bumpUse(&use_counts, b.lhs);
                    try bumpUse(&use_counts, b.rhs);
                },
                // Single-operand ops.
                .local_set => |ls| try bumpUse(&use_counts, ls.val),
                .global_set => |gs| try bumpUse(&use_counts, gs.val),
                .eqz,
                .ctz, .clz, .popcnt,
                .extend8_s, .extend16_s, .extend32_s,
                .extend_i32_s, .extend_i32_u, .wrap_i64,
                .f_neg, .f_abs, .f_sqrt,
                .convert_i32_s, .convert_i32_u, .convert_i64_s, .convert_i64_u,
                .demote_f64, .promote_f32,
                .trunc_f32_s, .trunc_f32_u, .trunc_f64_s, .trunc_f64_u,
                .trunc_sat_f32_s, .trunc_sat_f32_u, .trunc_sat_f64_s, .trunc_sat_f64_u,
                .reinterpret,
                .memory_grow,
                => |v| try bumpUse(&use_counts, v),
                .ret => |maybe_v| if (maybe_v) |v| try bumpUse(&use_counts, v),
                .load => |ld| try bumpUse(&use_counts, ld.base),
                .store => |st| {
                    try bumpUse(&use_counts, st.base);
                    try bumpUse(&use_counts, st.val);
                },
                .atomic_load => |ald| try bumpUse(&use_counts, ald.base),
                .atomic_store => |ast| {
                    try bumpUse(&use_counts, ast.base);
                    try bumpUse(&use_counts, ast.val);
                },
                .select => |sel| {
                    try bumpUse(&use_counts, sel.cond);
                    try bumpUse(&use_counts, sel.if_true);
                    try bumpUse(&use_counts, sel.if_false);
                },
                .br_if => |bi| try bumpUse(&use_counts, bi.cond),
                .br_table => |bt| try bumpUse(&use_counts, bt.index),
                .call => |cl| for (cl.args) |a| try bumpUse(&use_counts, a),
                .call_indirect => |ci| {
                    try bumpUse(&use_counts, ci.elem_idx);
                    for (ci.args) |a| try bumpUse(&use_counts, a);
                },
                .call_ref => |cr| {
                    try bumpUse(&use_counts, cr.func_ref);
                    for (cr.args) |a| try bumpUse(&use_counts, a);
                },
                .memory_fill => |mf| {
                    try bumpUse(&use_counts, mf.dst);
                    try bumpUse(&use_counts, mf.val);
                    try bumpUse(&use_counts, mf.len);
                },
                .memory_copy => |mc| {
                    try bumpUse(&use_counts, mc.dst);
                    try bumpUse(&use_counts, mc.src);
                    try bumpUse(&use_counts, mc.len);
                },
                .table_get => |tg| try bumpUse(&use_counts, tg.idx),
                .table_set => |ts| {
                    try bumpUse(&use_counts, ts.idx);
                    try bumpUse(&use_counts, ts.val);
                },
                .table_grow => |tg| {
                    try bumpUse(&use_counts, tg.init);
                    try bumpUse(&use_counts, tg.delta);
                },
                // Zero-operand ops (producers only): iconst, fconst, local_get,
                // global_get, memory_size, ref_func, ref_null, table_size, br,
                // @"unreachable".
                else => {},
            }
        }
    }

    var mul_fused = std.AutoHashMap(ir.VReg, void).init(allocator);
    defer mul_fused.deinit();
    var fma_info = std.AutoHashMap(ir.VReg, FmaInfo).init(allocator);
    defer fma_info.deinit();
    for (func.blocks.items) |block| {
        const insts = block.instructions.items;
        var i: usize = 0;
        while (i < insts.len) : (i += 1) {
            const add_inst = insts[i];
            const is_add = add_inst.op == .add;
            const is_sub = add_inst.op == .sub;
            if (!is_add and !is_sub) continue;
            if (add_inst.type != .i32 and add_inst.type != .i64) continue;
            const bin: ir.Inst.BinOp = switch (add_inst.op) {
                .add => |b| b, .sub => |b| b, else => continue,
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
                break;
            }
        }
    }
    fctx.mul_fused = &mul_fused;
    fctx.fma_info = &fma_info;

    try code.emitPrologue(frame_size);
    try emitCalleeSaveStore(&code, callee_save_base);

    // Spill VMContext (x0) and wasm params (x1..x7) to their frame slots.
    try emitEntrySpill(&code, func.*);

    var block_offsets = try allocator.alloc(usize, func.blocks.items.len);
    defer allocator.free(block_offsets);
    @memset(block_offsets, 0);

    var patches: std.ArrayListUnmanaged(BranchPatch) = .empty;
    defer patches.deinit(allocator);

    var last_was_ret = false;
    for (func.blocks.items, 0..) |block, bi| {
        block_offsets[bi] = code.len();
        for (block.instructions.items) |inst| {
            last_was_ret = isRet(inst.op);
            try compileInst(&code, inst, &reg_map, frame_size, &patches, &fctx);
        }
    }

    if (!last_was_ret) {
        try emitCalleeSaveRestore(&code, callee_save_base);
        try code.emitEpilogue(frame_size);
    }

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

/// FP-relative slot offset (in bytes) for the VMContext pointer.
const vmctx_slot_offset: u32 = 16;

/// FP-relative slot offset (in bytes, unsigned, positive) for local `idx`.
fn localSlotOffset(idx: u32) u32 {
    return (idx + 3) * 8;
}

/// ABI register holding wasm parameter `i` (0-based). Parameter 0 is in x1
/// because x0 carries the hidden VMContext pointer.
fn paramAbiReg(i: u32) emit.Reg {
    return switch (i) {
        0 => .x1, 1 => .x2, 2 => .x3, 3 => .x4,
        4 => .x5, 5 => .x6, 6 => .x7,
        else => unreachable,
    };
}

/// Spill x0 (VMContext) and x1..x7 (wasm params) to their frame slots, and
/// zero-initialize declared locals. Must be called right after emitPrologue
/// and before any vreg allocation so we don't clobber these ABI regs.
fn emitEntrySpill(code: *emit.CodeBuffer, func: ir.IrFunction) !void {
    // VMContext → [fp + 16]
    try code.strImm(.x0, .fp, vmctx_slot_offset / 8);

    // Wasm params → local slots [fp + (i+3)*8]
    var i: u32 = 0;
    while (i < func.param_count) : (i += 1) {
        const off_scaled: u12 = @intCast(localSlotOffset(i) / 8);
        try code.strImm(paramAbiReg(i), .fp, off_scaled);
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

fn isRet(op: ir.Inst.Op) bool {
    return switch (op) {
        .ret => true,
        else => false,
    };
}

/// The 10 callee-saved allocatable registers (x19..x28). Must match the
/// tail of `RegMap.scratch_regs`.
const callee_saved_regs = [_]emit.Reg{
    .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27, .x28,
};

/// Save all callee-saved allocatable registers to `[fp + callee_save_base
/// + i*8]`. Called once from the prologue, unconditionally — the simple
/// linear-scan allocator doesn't track per-reg usage precisely enough to
/// emit targeted saves, and the overhead (10 STRs per function) is small
/// relative to typical wasm function bodies.
fn emitCalleeSaveStore(code: *emit.CodeBuffer, callee_save_base: u32) !void {
    for (callee_saved_regs, 0..) |r, i| {
        const off_scaled: u12 = @intCast((callee_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.strImm(r, .fp, off_scaled);
    }
}

/// Restore all callee-saved allocatable registers. Emitted before each
/// epilogue so every return path leaves the registers in the state the
/// caller expects.
fn emitCalleeSaveRestore(code: *emit.CodeBuffer, callee_save_base: u32) !void {
    for (callee_saved_regs, 0..) |r, i| {
        const off_scaled: u12 = @intCast((callee_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(r, .fp, off_scaled);
    }
}

fn compileInst(
    code: *emit.CodeBuffer,
    inst: ir.Inst,
    reg_map: *RegMap,
    frame_size: u32,
    patches: *std.ArrayListUnmanaged(BranchPatch),
    fctx: *FuncCompileCtx,
) !void {
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
            try emitCalleeSaveRestore(code, fctx.callee_save_base);
            try code.emitEpilogue(frame_size);
        },
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
    if (is64) try code.fmovDFromGp64(0, src)
    else try code.fmovSFromGp32(0, src);
    try code.fsqrtScalar(is64, 0, 0);
    if (is64) try code.fmovGpFromD64(info.reg, 0)
    else try code.fmovGpFromS32(info.reg, 0);
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
    if (is64) try code.fmovDFromGp64(0, src)
    else try code.fmovSFromGp32(0, src);
    try code.frintScalar(is64, mode, 0, 0);
    if (is64) try code.fmovGpFromD64(info.reg, 0)
    else try code.fmovGpFromS32(info.reg, 0);
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
    if (is_f64) try code.fmovGpFromD64(info.reg, 0)
    else try code.fmovGpFromS32(info.reg, 0);
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
    if (src_is_f32) try code.fmovSFromGp32(0, src)
    else try code.fmovDFromGp64(0, src);

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
    if (src_is_f32) try code.fmovSFromGp32(0, src)
    else try code.fmovDFromGp64(0, src);

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
        0 => .x1, 1 => .x2, 2 => .x3, 3 => .x4,
        4 => .x5, 5 => .x6, 6 => .x7,
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
    if (cl.tail) return error.UnimplementedTailCall;
    if (cl.extra_results > 0) return error.UnimplementedMultiResult;
    if (cl.args.len > 7) return error.TooManyCallArgs;
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

    // Spill each used caller-save reg to [fp + call_save_base + i*8].
    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.strImm(reg, .fp, slot_scaled);
    }

    // Move arguments into x1..x7 by reading from their current source
    // locations via memory. Physreg sources are read from the call-save
    // slot we just filled; stack sources from the RegMap spill slot.
    // This order-independent approach avoids parallel-move hazards.
    for (cl.args, 0..) |arg_vreg, i| {
        const target = callArgReg(@intCast(i));
        const loc = reg_map.get(arg_vreg) orelse return error.UnboundVReg;
        switch (loc) {
            .reg => |r| {
                const reg_idx: u32 = @intFromEnum(r);
                if (reg_idx >= 19) {
                    // Callee-saved source reg was not spilled across BL — just
                    // MOV it into the target arg reg directly.
                    try code.movRegReg(target, r);
                } else {
                    const slot_scaled: u12 = @intCast((fctx.call_save_base + reg_idx * 8) / 8);
                    try code.ldrImm(target, .fp, slot_scaled);
                }
            },
            .stack => |off| {
                try code.ldrImm(target, .fp, reg_map.spillOffsetScaled(off));
            },
        }
    }

    // Load VMContext into x0 (was saved above if previously live).
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);

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

    // Restore previously-saved caller-save regs.
    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
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
    // Step 1: zero-extend wasm address into tmp2 (kept alive across check).
    const src = try useInto(code, reg_map, base_vreg, RegMap.tmp1);
    try code.movRegReg32(RegMap.tmp2, src);

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

    // Step 4: cmp end, mem_size; B.LS over_trap (≤ is in-bounds since
    // accesses of size s are valid when end = addr+s ≤ memory_size).
    try code.cmpRegReg(RegMap.tmp1, RegMap.tmp0);
    const over_patch = code.len();
    try code.bCond(.ls, 0); // placeholder

    // Trap path: arg0=vmctx, call vmctx.trap_oob_fn (noreturn). We read
    // vmctx again because tmp0 now holds mem_size. BRK after BLR is
    // defensive — the helper is declared noreturn.
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    // trap_oob_fn is at VmCtx offset 80 = scale-8 slot 10.
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

    // Step 5: reload vmctx, then load memory_base (at +0).
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
const vmctx_memsize_slot: u12 = 1;       // byte 8, scale 8
const vmctx_globals_slot: u12 = 2;       // byte 16, scale 8
const vmctx_host_functions_slot: u12 = 3;// byte 24, scale 8
const vmctx_mem_pages_slot_w: u12 = 14;  // byte 56 (u32), scale 4
const vmctx_mem_grow_fn_slot: u12 = 8;   // byte 64, scale 8
const vmctx_mem_fill_fn_slot: u12 = 28;  // byte 224, scale 8
const vmctx_mem_copy_fn_slot: u12 = 29;  // byte 232, scale 8
const vmctx_trap_oob_fn_slot: u12 = 10;  // byte 80, scale 8
const vmctx_trap_unreachable_fn_slot: u12 = 11; // byte 88, scale 8
const vmctx_trap_idivz_fn_slot: u12 = 12; // byte 96, scale 8
const vmctx_trap_iovf_fn_slot: u12 = 13;  // byte 104, scale 8
const vmctx_funcptrs_slot: u12 = 15;     // byte 120, scale 8
const vmctx_table_grow_fn_slot: u12 = 16;// byte 128, scale 8
const vmctx_tables_info_slot: u12 = 17;  // byte 136, scale 8
const vmctx_sig_table_slot: u12 = 20;    // byte 160, scale 8
const vmctx_table_set_fn_slot: u12 = 24; // byte 192, scale 8

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
    if (is32) try code.ldrImm32(info.reg, RegMap.tmp0, @intCast(idx * 2))
    else try code.ldrImm(info.reg, RegMap.tmp0, @intCast(idx));
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
    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.strImm(reg, .fp, slot_scaled);
    }

    // x1 = table_idx, x2 = idx, x3 = val, x0 = vmctx.
    try code.movImm32(.x1, @intCast(ts.table_idx));
    try stageArgFromSaved(code, reg_map, fctx, .x2, ts.idx);
    try stageArgFromSaved(code, reg_map, fctx, .x3, ts.val);
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.ldrImm(RegMap.tmp0, .x0, vmctx_table_set_fn_slot);
    try code.blr(RegMap.tmp0);

    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// Helper used by `emitTableSet`/`emitTableGrow` to stage one VReg into
/// a specific arg register, reading from the call-save or spill slot.
/// Precondition: the caller has snapshotted and spilled all used
/// caller-save regs to `fctx.call_save_base`.
fn stageArgFromSaved(
    code: *emit.CodeBuffer,
    reg_map: *RegMap,
    fctx: *FuncCompileCtx,
    target: emit.Reg,
    vreg: ir.VReg,
) !void {
    const loc = reg_map.get(vreg) orelse return error.UnboundVReg;
    switch (loc) {
        .reg => |r| {
            const reg_idx: u32 = @intFromEnum(r);
            if (reg_idx >= 19) {
                try code.movRegReg(target, r);
            } else {
                const slot_scaled: u12 = @intCast((fctx.call_save_base + reg_idx * 8) / 8);
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
    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.strImm(reg, .fp, slot_scaled);
    }

    // x1 = init, x2 = delta, x3 = table_idx, x0 = vmctx.
    try stageArgFromSaved(code, reg_map, fctx, .x1, tg.init);
    try stageArgFromSaved(code, reg_map, fctx, .x2, tg.delta);
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
    if (ci.tail) return error.UnimplementedTailCall;
    if (ci.extra_results > 0) return error.UnimplementedMultiResult;
    if (ci.args.len > 7) return error.TooManyCallArgs;

    // Snapshot live caller-save regs and spill to call_save.
    var used_snapshot: [RegMap.scratch_regs.len]bool = undefined;
    for (&used_snapshot, reg_map.reg_used) |*dst, src| dst.* = src;
    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.strImm(reg, .fp, slot_scaled);
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

    // Stage args into x1..xN.
    for (ci.args, 0..) |arg_vreg, i| {
        try stageArgFromSaved(code, reg_map, fctx, callArgReg(@intCast(i)), arg_vreg);
    }
    try code.ldrImm(.x0, .fp, vmctx_slot_offset / 8);
    try code.blr(RegMap.tmp0);

    if (inst.dest) |dest| {
        const info = try destBegin(reg_map, dest, RegMap.tmp0);
        if (info.reg != .x0) try code.movRegReg(info.reg, .x0);
        try destCommit(code, reg_map, info);
    }

    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.ldrImm(reg, .fp, slot_scaled);
    }
}

/// Emit an indirect call to a host helper via a VmCtx slot, passing
/// `vmctx` as the first arg and `args` as x1..xN (up to 7). Uses the
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

    for (used_snapshot, 0..) |used, i| {
        if (!used) continue;
        if (i >= RegMap.caller_saved_count) continue;
        const reg = RegMap.scratch_regs[i];
        const slot_scaled: u12 = @intCast((fctx.call_save_base + @as(u32, @intCast(i)) * 8) / 8);
        try code.strImm(reg, .fp, slot_scaled);
    }

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
        if (is64) try code.movRegReg(RegMap.tmp0, lhs_loaded)
        else try code.movRegReg32(RegMap.tmp0, lhs_loaded);
    }
    const rhs_loaded = try useInto(code, reg_map, bin.rhs, RegMap.tmp1);
    if (rhs_loaded != RegMap.tmp1) {
        if (is64) try code.movRegReg(RegMap.tmp1, rhs_loaded)
        else try code.movRegReg32(RegMap.tmp1, rhs_loaded);
    }

    // Zero-divisor check: cmp rhs, #0; b.ne skip; call trap_idivz; skip:
    if (is64) try code.cmpImm(RegMap.tmp1, 0)
    else try code.cmpImm32(RegMap.tmp1, 0);
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
                if (is64) try code.sdivRegReg(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1)
                else try code.sdivRegReg32(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1);
            } else {
                if (is64) try code.udivRegReg(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1)
                else try code.udivRegReg32(RegMap.tmp2, RegMap.tmp0, RegMap.tmp1);
            }
            // MSUB dest, q, rhs, lhs  →  dest = lhs - q*rhs
            if (is64) try code.msubRegReg(info.reg, RegMap.tmp2, RegMap.tmp1, RegMap.tmp0)
            else try code.msubRegReg32(info.reg, RegMap.tmp2, RegMap.tmp1, RegMap.tmp0);
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
    const end_offset: u64 = @as(u64, ld.offset) + @as(u64, ld.size);
    try emitMemAddr(code, reg_map, ld.base, if (folded == null) ld.offset else 0, end_offset);
    const disp: u12 = if (folded) |d| d else 0;

    const info = try destBegin(reg_map, dest, RegMap.tmp1);
    if (ld.sign_extend) {
        switch (ld.size) {
            1 => if (is64) try code.ldrsbImm64(info.reg, RegMap.tmp0, disp)
                 else try code.ldrsbImm32(info.reg, RegMap.tmp0, disp),
            2 => if (is64) try code.ldrshImm64(info.reg, RegMap.tmp0, disp)
                 else try code.ldrshImm32(info.reg, RegMap.tmp0, disp),
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
    const end_offset: u64 = @as(u64, st.offset) + @as(u64, st.size);
    try emitMemAddr(code, reg_map, st.base, if (folded == null) st.offset else 0, end_offset);
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
        if (enc.shift12) try code.subImmShift12(dst, src, enc.imm12)
        else try code.subImm(dst, src, enc.imm12);
    } else {
        if (enc.shift12) try code.addImmShift12(dst, src, enc.imm12)
        else try code.addImm(dst, src, enc.imm12);
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
        if ((w & 0xFFC00000) == 0xF9400000) { found = true; break; }
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
        if ((w & 0xFFC00000) == 0x39400000) { found = true; break; }
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
        if ((w & 0xFFC00000) == 0x79C00000) { found = true; break; }
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
        if ((w & 0xFFC00000) == 0x39000000) { found = true; break; }
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
    // data_drop is not yet implemented on aarch64 — must fail loudly,
    // not silently drop.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const bid = try func.newBlock();
    try func.getBlock(bid).append(.{
        .op = .{ .data_drop = 0 },
        .dest = null,
        .type = .void,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = null } });
    try std.testing.expectError(error.UnimplementedOp, compileFunction(&func, allocator));
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
    // 17 iconsts keep x0..x15 live, then an ADD with v16+v17 forces spills
    // for the operands and possibly the dest. Verify code succeeds (no
    // silent drop) and contains at least one LDR/STR pair against FP.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const bid = try func.newBlock();

    // Allocate 17 vregs, each with a distinct iconst. The 17th (v16)
    // forces spill.
    var vregs: [18]ir.VReg = undefined;
    for (&vregs, 0..) |*v, i| {
        v.* = func.newVReg();
        try func.getBlock(bid).append(.{
            .op = .{ .iconst_64 = @intCast(i + 1) },
            .dest = v.*,
            .type = .i64,
        });
    }
    // dst = vregs[16] + vregs[17]   (both spilled)
    const dst = func.newVReg();
    try func.getBlock(bid).append(.{
        .op = .{ .add = .{ .lhs = vregs[16], .rhs = vregs[17] } },
        .dest = dst,
        .type = .i64,
    });
    try func.getBlock(bid).append(.{ .op = .{ .ret = dst } });

    const code = try compileFunction(&func, allocator);
    defer allocator.free(code);

    // Scan for LDR Xt, [fp, #imm] (opcode pattern 0xF9400000 + rn=29<<5).
    // We must see at least one such load (the spill reload).
    var found_ldr_from_fp = false;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const w = std.mem.readInt(u32, code[i..][0..4], .little);
        // LDR (imm) unsigned offset, 64-bit: top 10 bits == 0x3E5, rn bits 5-9
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
