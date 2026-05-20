//! IR invariant checker (Cranelift-style "verifier") run between IR passes.
//!
//! Detects SSA / CFG breakage at the *producing pass* instead of letting the
//! symptom surface deep inside aarch64 lowering as `error.UnboundVReg`. See
//! issue #624.
//!
//! Checks implemented (Phase 1 — issue #624 invariants 1, 2, 4, 5, 6):
//!
//!   1. **SSA def-before-use.** Every `VReg` operand read by an instruction
//!      must be either (a) a function parameter (vregs `0..param_count`),
//!      (b) defined by an earlier instruction in the same block, or
//!      (c) defined in a strictly-dominating block.
//!   2. **Def uniqueness.** Each `VReg` has at most one defining instruction.
//!   4. **Terminator legality.** Every reachable block ends in exactly one
//!      terminator (`br`, `br_if`, `br_table`, `ret`, `ret_multi`,
//!      `unreachable`, or a tail-call). Non-last instructions must not be
//!      terminators.
//!   5. **No dangling block refs.** Every `BlockId` named by a terminator
//!      refers to an existing block in `func.blocks`.
//!   6. **Predecessor consistency.** `BasicBlock.predecessors` matches the
//!      set of blocks whose terminator targets it (both directions — no
//!      stale preds and no missing preds).
//!
//! Check 3 (block-parameter arity) is vacuous on the current IR — phi is an
//! instruction, not a block parameter — and is therefore omitted. It will be
//! re-introduced if/when block-params land.
//!
//! Wiring: `passes.runPassesWithOptions` calls `verifyFunction` after every
//! pass invocation when `opts.verify_mode != .off`, and annotates the failure
//! with the pass name before propagating.

const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");

pub const VerifyMode = enum {
    /// Skip verification entirely (no-op, zero cost).
    off,
    /// Run checks 1, 2, 4, 5, 6 after every pass.
    after_each_pass,
    /// Reserved for future stricter checks (operand-width sanity, loop /
    /// dom-tree freshness, live-range monotonicity). Behaves like
    /// `after_each_pass` for now.
    paranoid,
};

pub const VerifyError = error{
    /// An instruction reads a VReg that has no dominating definition
    /// (check 1).
    UnboundVRegUse,
    /// A VReg appears as the `dest` of more than one instruction
    /// (check 2).
    VRegDefinedTwice,
    /// A reachable block has no terminator at its end (check 4).
    MissingTerminator,
    /// A block contains a terminator before its final instruction (check 4).
    MultipleTerminators,
    /// A terminator references a `BlockId` that doesn't exist in the
    /// function (check 5).
    DanglingBlockRef,
    /// `BasicBlock.predecessors` lists a block whose terminator does not
    /// target this block (check 6).
    StalePredecessor,
    /// A block whose terminator targets another block is missing from the
    /// target's `predecessors` list (check 6).
    MissingPredecessor,
} || std.mem.Allocator.Error;

/// Information about the most-recent verifier failure. Populated as a
/// best-effort companion to the returned `VerifyError` so callers (the
/// `runPasses` wrapper + the `wamrc` CLI) can produce a meaningful
/// "pass X broke invariant Y in func #N block #B" diagnostic without
/// hand-decoding the error.
///
/// Thread-local — the verifier is single-threaded (one IR module at a
/// time inside `runPasses`).
pub const LastFailure = struct {
    kind: VerifyError = error.OutOfMemory, // sentinel, overwritten on real failure
    func_index: ?u32 = null,
    block: ?ir.BlockId = null,
    inst_index: ?u32 = null,
    vreg: ?ir.VReg = null,
    pass_name: []const u8 = "",
    detail: []const u8 = "",

    pub fn reset(self: *LastFailure) void {
        self.* = .{};
    }

    pub fn format(self: LastFailure, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("IR verifier: {s}", .{@errorName(self.kind)});
        if (self.pass_name.len > 0) try writer.print(" after pass '{s}'", .{self.pass_name});
        if (self.func_index) |fi| try writer.print(" func #{d}", .{fi});
        if (self.block) |b| try writer.print(" block #{d}", .{b});
        if (self.inst_index) |i| try writer.print(" inst #{d}", .{i});
        if (self.vreg) |v| try writer.print(" vreg %{d}", .{v});
        if (self.detail.len > 0) try writer.print(" — {s}", .{self.detail});
    }
};

/// Reset before each `runPasses` invocation, populated on the first failing
/// check. Accessed by `passes.runPassesWithOptions` to surface a diagnostic.
pub threadlocal var last_failure: LastFailure = .{};

// ── Operand / successor iteration ───────────────────────────────────────

/// Invoke `cb(ctx, vreg)` for every `VReg` *read* by `inst` (operands,
/// not the `dest`). Comptime-exhaustive over every `Inst.Op` variant.
pub fn forEachOperand(
    inst: ir.Inst,
    ctx: anytype,
    comptime cb: fn (@TypeOf(ctx), ir.VReg) void,
) void {
    switch (inst.op) {
        // Pure constants / nullary ops — no VReg operands.
        .iconst_32,
        .iconst_64,
        .fconst_32,
        .fconst_64,
        .v128_const,
        .local_get,
        .global_get,
        .br,
        .@"unreachable",
        .atomic_fence,
        .memory_size,
        .table_size,
        .ref_func,
        .data_drop,
        .elem_drop,
        .call_result,
        => {},

        // BinOp shape (lhs, rhs).
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
        .f_min,
        .f_max,
        .f_copysign,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        => |bin| {
            cb(ctx, bin.lhs);
            cb(ctx, bin.rhs);
        },

        // SIMD bin/un/lane shapes. Each opcode's payload is a distinct type
        // so we can't combine them in one match arm — list them out instead.
        .v128_bitwise => |bin| {
            cb(ctx, bin.lhs);
            cb(ctx, bin.rhs);
        },
        .v128_bitselect => |sel| {
            cb(ctx, sel.a);
            cb(ctx, sel.b);
            cb(ctx, sel.mask);
        },
        .i32x4_binop => |bin| {
            cb(ctx, bin.lhs);
            cb(ctx, bin.rhs);
        },
        .f32x4_binop => |bin| {
            cb(ctx, bin.lhs);
            cb(ctx, bin.rhs);
        },
        .i8x16_binop => |bin| {
            cb(ctx, bin.lhs);
            cb(ctx, bin.rhs);
        },
        .i16x8_binop => |bin| {
            cb(ctx, bin.lhs);
            cb(ctx, bin.rhs);
        },
        .i64x2_binop => |bin| {
            cb(ctx, bin.lhs);
            cb(ctx, bin.rhs);
        },
        .f64x2_binop => |bin| {
            cb(ctx, bin.lhs);
            cb(ctx, bin.rhs);
        },
        .i32x4_unop => |un| cb(ctx, un.vector),
        .f32x4_unop => |un| cb(ctx, un.vector),
        .i8x16_unop => |un| cb(ctx, un.vector),
        .i16x8_unop => |un| cb(ctx, un.vector),
        .i64x2_unop => |un| cb(ctx, un.vector),
        .f64x2_unop => |un| cb(ctx, un.vector),
        .i32x4_extadd_pairwise_i16x8 => |op| cb(ctx, op.vector),
        .i16x8_extadd_pairwise_i8x16 => |op| cb(ctx, op.vector),
        .i32x4_extend_i16x8 => |op| cb(ctx, op.vector),
        .i16x8_extend_i8x16 => |op| cb(ctx, op.vector),
        .i64x2_extend_i32x4 => |op| cb(ctx, op.vector),
        .i32x4_trunc_sat => |op| cb(ctx, op.vector),
        .f32x4_convert_i32x4 => |op| cb(ctx, op.vector),
        .f32x4_demote_f64x2_zero => |op| cb(ctx, op.vector),
        .f64x2_convert_low_i32x4 => |op| cb(ctx, op.vector),
        .f64x2_promote_low_f32x4 => |op| cb(ctx, op.vector),
        .i32x4_dot_i16x8_s => |op| {
            cb(ctx, op.lhs);
            cb(ctx, op.rhs);
        },
        .i32x4_extmul_i16x8 => |op| {
            cb(ctx, op.lhs);
            cb(ctx, op.rhs);
        },
        .i16x8_extmul_i8x16 => |op| {
            cb(ctx, op.lhs);
            cb(ctx, op.rhs);
        },
        .i64x2_extmul_i32x4 => |op| {
            cb(ctx, op.lhs);
            cb(ctx, op.rhs);
        },
        .i8x16_shuffle => |op| {
            cb(ctx, op.lhs);
            cb(ctx, op.rhs);
        },
        .i8x16_narrow_i16x8 => |op| {
            cb(ctx, op.lhs);
            cb(ctx, op.rhs);
        },
        .i16x8_narrow_i32x4 => |op| {
            cb(ctx, op.lhs);
            cb(ctx, op.rhs);
        },
        .i8x16_swizzle => |op| {
            cb(ctx, op.vector);
            cb(ctx, op.indices);
        },
        .i8x16_shift => |shift| {
            cb(ctx, shift.vector);
            cb(ctx, shift.count);
        },
        .i16x8_shift => |shift| {
            cb(ctx, shift.vector);
            cb(ctx, shift.count);
        },
        .i32x4_shift => |shift| {
            cb(ctx, shift.vector);
            cb(ctx, shift.count);
        },
        .i64x2_shift => |shift| {
            cb(ctx, shift.vector);
            cb(ctx, shift.count);
        },
        .simd_all_true => |op| cb(ctx, op.vector),
        .simd_bitmask => |op| cb(ctx, op.vector),
        .i32x4_extract_lane => |lane| cb(ctx, lane.vector),
        .f32x4_extract_lane => |lane| cb(ctx, lane.vector),
        .i8x16_extract_lane => |lane| cb(ctx, lane.vector),
        .i16x8_extract_lane => |lane| cb(ctx, lane.vector),
        .i64x2_extract_lane => |lane| cb(ctx, lane.vector),
        .f64x2_extract_lane => |lane| cb(ctx, lane.vector),
        .i32x4_replace_lane => |lane| {
            cb(ctx, lane.vector);
            cb(ctx, lane.val);
        },
        .f32x4_replace_lane => |lane| {
            cb(ctx, lane.vector);
            cb(ctx, lane.val);
        },
        .i8x16_replace_lane => |lane| {
            cb(ctx, lane.vector);
            cb(ctx, lane.val);
        },
        .i16x8_replace_lane => |lane| {
            cb(ctx, lane.vector);
            cb(ctx, lane.val);
        },
        .i64x2_replace_lane => |lane| {
            cb(ctx, lane.vector);
            cb(ctx, lane.val);
        },
        .f64x2_replace_lane => |lane| {
            cb(ctx, lane.vector);
            cb(ctx, lane.val);
        },

        // Single-operand unary ops (where the variant payload is a bare VReg).
        .clz,
        .ctz,
        .popcnt,
        .eqz,
        .wrap_i64,
        .extend_i32_s,
        .extend_i32_u,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i64_s,
        .convert_i32_u,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .reinterpret,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .v128_not,
        .v128_any_true,
        .i32x4_splat,
        .f32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        .f64x2_splat,
        .memory_grow,
        => |vreg| cb(ctx, vreg),

        .local_set => |ls| cb(ctx, ls.val),
        .global_set => |gs| cb(ctx, gs.val),

        .load => |ld| cb(ctx, ld.base),
        .v128_load => |ld| cb(ctx, ld.base),
        .v128_load_splat => |ld| cb(ctx, ld.base),
        .v128_load_zero => |ld| cb(ctx, ld.base),
        .v128_load_extend => |ld| cb(ctx, ld.base),
        .v128_load_lane => |ld| {
            cb(ctx, ld.base);
            cb(ctx, ld.vector);
        },
        .store => |st| {
            cb(ctx, st.base);
            cb(ctx, st.val);
        },
        .v128_store => |st| {
            cb(ctx, st.base);
            cb(ctx, st.val);
        },
        .v128_store_lane => |st| {
            cb(ctx, st.base);
            cb(ctx, st.vector);
        },

        .br_if => |bi| cb(ctx, bi.cond),
        .br_table => |bt| cb(ctx, bt.index),

        .ret => |maybe| if (maybe) |v| cb(ctx, v),
        .ret_multi => |vregs| {
            for (vregs) |v| cb(ctx, v);
        },

        .call => |cl| {
            for (cl.args) |a| cb(ctx, a);
        },
        .call_indirect => |ci| {
            cb(ctx, ci.elem_idx);
            for (ci.args) |a| cb(ctx, a);
        },
        .call_ref => |cr| {
            cb(ctx, cr.func_ref);
            for (cr.args) |a| cb(ctx, a);
        },
        .select => |sel| {
            cb(ctx, sel.cond);
            cb(ctx, sel.if_true);
            cb(ctx, sel.if_false);
        },

        .atomic_load => |al| cb(ctx, al.base),
        .atomic_store => |ast| {
            cb(ctx, ast.base);
            cb(ctx, ast.val);
        },
        .atomic_rmw => |ar| {
            cb(ctx, ar.base);
            cb(ctx, ar.val);
        },
        .atomic_cmpxchg => |ac| {
            cb(ctx, ac.base);
            cb(ctx, ac.expected);
            cb(ctx, ac.replacement);
        },
        .atomic_notify => |an| {
            cb(ctx, an.base);
            cb(ctx, an.count);
        },
        .atomic_wait => |aw| {
            cb(ctx, aw.base);
            cb(ctx, aw.expected);
            cb(ctx, aw.timeout);
        },

        .memory_copy => |mc| {
            cb(ctx, mc.dst);
            cb(ctx, mc.src);
            cb(ctx, mc.len);
        },
        .memory_fill => |mf| {
            cb(ctx, mf.dst);
            cb(ctx, mf.val);
            cb(ctx, mf.len);
        },
        .memory_init => |mi| {
            cb(ctx, mi.dst);
            cb(ctx, mi.src);
            cb(ctx, mi.len);
        },
        .table_get => |tg| cb(ctx, tg.idx),
        .table_set => |ts| {
            cb(ctx, ts.idx);
            cb(ctx, ts.val);
        },
        .table_grow => |tg| {
            cb(ctx, tg.init);
            cb(ctx, tg.delta);
        },
        .table_init => |ti| {
            cb(ctx, ti.dst);
            cb(ctx, ti.src);
            cb(ctx, ti.len);
        },

        .phi => |edges| {
            for (edges) |e| cb(ctx, e.val);
        },
        .parallel_copy => |pairs| {
            for (pairs) |p| cb(ctx, p.src);
        },
    }
}

/// True iff `op` transfers control out of the current block.
pub fn isTerminator(op: ir.Inst.Op) bool {
    return switch (op) {
        .br,
        .br_if,
        .br_table,
        .ret,
        .ret_multi,
        .@"unreachable",
        => true,
        .call => |c| c.tail,
        .call_indirect => |c| c.tail,
        .call_ref => |c| c.tail,
        else => false,
    };
}

/// Invoke `cb(ctx, target)` for every successor `BlockId` named by `op`.
/// Non-terminators contribute no successors. Tail-calls have no in-function
/// successors.
pub fn forEachSuccessor(
    op: ir.Inst.Op,
    ctx: anytype,
    comptime cb: fn (@TypeOf(ctx), ir.BlockId) void,
) void {
    switch (op) {
        .br => |t| cb(ctx, t),
        .br_if => |bi| {
            cb(ctx, bi.then_block);
            cb(ctx, bi.else_block);
        },
        .br_table => |bt| {
            for (bt.targets) |t| cb(ctx, t);
            cb(ctx, bt.default);
        },
        else => {},
    }
}

// ── Module / function entry points ──────────────────────────────────────

pub fn verifyModule(
    module: *const ir.IrModule,
    mode: VerifyMode,
    allocator: std.mem.Allocator,
) VerifyError!void {
    if (mode == .off) return;
    for (module.functions.items, 0..) |*func, fi| {
        try verifyFunction(func, @intCast(fi), mode, allocator);
    }
}

pub fn verifyFunction(
    func: *const ir.IrFunction,
    func_index: u32,
    mode: VerifyMode,
    allocator: std.mem.Allocator,
) VerifyError!void {
    if (mode == .off) return;
    last_failure.reset();
    last_failure.func_index = func_index;

    if (func.blocks.items.len == 0) return;

    try checkDefUniqueness(func, func_index);
    try checkTerminators(func, func_index);
    try checkBlockRefs(func, func_index);
    try checkPredecessors(func, func_index, allocator);
    try checkSsaDominance(func, func_index, allocator);
}

// ── Check 2: def uniqueness ─────────────────────────────────────────────

fn checkDefUniqueness(func: *const ir.IrFunction, func_index: u32) VerifyError!void {
    var seen = std.AutoHashMap(ir.VReg, void).init(std.heap.page_allocator);
    defer seen.deinit();
    for (func.blocks.items) |block| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.dest) |d| {
                const gop = try seen.getOrPut(d);
                if (gop.found_existing) {
                    last_failure = .{
                        .kind = error.VRegDefinedTwice,
                        .func_index = func_index,
                        .block = block.id,
                        .inst_index = @intCast(ii),
                        .vreg = d,
                        .detail = "VReg has a second defining instruction",
                    };
                    return error.VRegDefinedTwice;
                }
            }
            // phi result is encoded as `dest`; parallel_copy multi-dests are
            // tracked separately.
            switch (inst.op) {
                .parallel_copy => |pairs| for (pairs) |p| {
                    const gop = try seen.getOrPut(p.dst);
                    if (gop.found_existing) {
                        last_failure = .{
                            .kind = error.VRegDefinedTwice,
                            .func_index = func_index,
                            .block = block.id,
                            .inst_index = @intCast(ii),
                            .vreg = p.dst,
                            .detail = "parallel_copy dst has a second definition",
                        };
                        return error.VRegDefinedTwice;
                    }
                },
                else => {},
            }
        }
    }
}

// ── Check 4: terminator legality ────────────────────────────────────────

fn checkTerminators(func: *const ir.IrFunction, func_index: u32) VerifyError!void {
    for (func.blocks.items) |block| {
        if (block.instructions.items.len == 0) {
            last_failure = .{
                .kind = error.MissingTerminator,
                .func_index = func_index,
                .block = block.id,
                .detail = "block has no instructions",
            };
            return error.MissingTerminator;
        }
        const last_idx = block.instructions.items.len - 1;
        for (block.instructions.items, 0..) |inst, ii| {
            const is_term = isTerminator(inst.op);
            if (is_term and ii != last_idx) {
                last_failure = .{
                    .kind = error.MultipleTerminators,
                    .func_index = func_index,
                    .block = block.id,
                    .inst_index = @intCast(ii),
                    .detail = "terminator before end of block",
                };
                return error.MultipleTerminators;
            }
            if (!is_term and ii == last_idx) {
                last_failure = .{
                    .kind = error.MissingTerminator,
                    .func_index = func_index,
                    .block = block.id,
                    .inst_index = @intCast(ii),
                    .detail = "last instruction is not a terminator",
                };
                return error.MissingTerminator;
            }
        }
    }
}

// ── Check 5: dangling block refs ────────────────────────────────────────

fn checkBlockRefs(func: *const ir.IrFunction, func_index: u32) VerifyError!void {
    const nblocks: u32 = @intCast(func.blocks.items.len);
    for (func.blocks.items) |block| {
        for (block.instructions.items, 0..) |inst, ii| {
            const Ctx = struct {
                ok: bool = true,
                bad: ir.BlockId = 0,
                nblocks: u32,
            };
            var c = Ctx{ .nblocks = nblocks };
            forEachSuccessor(inst.op, &c, struct {
                fn cb(ptr: *Ctx, target: ir.BlockId) void {
                    if (!ptr.ok) return;
                    if (target >= ptr.nblocks) {
                        ptr.ok = false;
                        ptr.bad = target;
                    }
                }
            }.cb);
            if (!c.ok) {
                last_failure = .{
                    .kind = error.DanglingBlockRef,
                    .func_index = func_index,
                    .block = block.id,
                    .inst_index = @intCast(ii),
                    .detail = "terminator targets a nonexistent block",
                };
                return error.DanglingBlockRef;
            }
        }
    }
}

// ── Check 6: predecessor consistency ────────────────────────────────────

fn checkPredecessors(
    func: *const ir.IrFunction,
    func_index: u32,
    allocator: std.mem.Allocator,
) VerifyError!void {
    const nblocks: u32 = @intCast(func.blocks.items.len);
    // Derive the canonical pred set from successor edges.
    var derived = try allocator.alloc(std.AutoHashMap(ir.BlockId, void), nblocks);
    defer {
        for (derived) |*s| s.deinit();
        allocator.free(derived);
    }
    for (derived) |*s| s.* = std.AutoHashMap(ir.BlockId, void).init(allocator);

    for (func.blocks.items) |block| {
        const last = block.instructions.items[block.instructions.items.len - 1];
        const Ctx = struct {
            derived: []std.AutoHashMap(ir.BlockId, void),
            src: ir.BlockId,
            err: ?std.mem.Allocator.Error = null,
        };
        var c = Ctx{ .derived = derived, .src = block.id };
        forEachSuccessor(last.op, &c, struct {
            fn cb(ptr: *Ctx, target: ir.BlockId) void {
                if (ptr.err != null) return;
                ptr.derived[target].put(ptr.src, {}) catch |e| {
                    ptr.err = e;
                };
            }
        }.cb);
        if (c.err) |e| return e;
    }

    // Compare each block's recorded preds to the derived set.
    for (func.blocks.items) |block| {
        var recorded = std.AutoHashMap(ir.BlockId, void).init(allocator);
        defer recorded.deinit();
        for (block.predecessors.items) |p| try recorded.put(p, {});

        // Stale: in `recorded` but not derived.
        var rit = recorded.iterator();
        while (rit.next()) |entry| {
            if (!derived[block.id].contains(entry.key_ptr.*)) {
                last_failure = .{
                    .kind = error.StalePredecessor,
                    .func_index = func_index,
                    .block = block.id,
                    .detail = "predecessor list contains a block whose terminator does not target this block",
                };
                return error.StalePredecessor;
            }
        }
        // Missing: in derived but not recorded.
        var dit = derived[block.id].iterator();
        while (dit.next()) |entry| {
            if (!recorded.contains(entry.key_ptr.*)) {
                last_failure = .{
                    .kind = error.MissingPredecessor,
                    .func_index = func_index,
                    .block = block.id,
                    .detail = "predecessor list omits a block whose terminator targets this block",
                };
                return error.MissingPredecessor;
            }
        }
    }
}

// ── Check 1: SSA def-before-use ─────────────────────────────────────────

fn checkSsaDominance(
    func: *const ir.IrFunction,
    func_index: u32,
    allocator: std.mem.Allocator,
) VerifyError!void {
    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();

    const nblocks = func.blocks.items.len;
    // For each VReg, record (block, inst_index) of its definition.
    var def_loc = std.AutoHashMap(ir.VReg, struct { block: ir.BlockId, idx: u32 }).init(allocator);
    defer def_loc.deinit();

    for (func.blocks.items) |block| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.dest) |d| {
                try def_loc.put(d, .{ .block = block.id, .idx = @intCast(ii) });
            }
            switch (inst.op) {
                .parallel_copy => |pairs| for (pairs) |p| {
                    try def_loc.put(p.dst, .{ .block = block.id, .idx = @intCast(ii) });
                },
                else => {},
            }
        }
    }

    const param_count = func.param_count;

    // Pre-collect predecessors per block (canonical from edges, since
    // checkPredecessors has already passed if we got here).
    for (func.blocks.items) |block| {
        // Only verify reachable blocks. `scrubUnreachableBlocks` is run by
        // `runPasses` and turns unreachable bodies into a single
        // `unreachable` op anyway, so an unreachable block with stale uses
        // would already fail check 4 / 5 first.
        if (block.id != 0 and dom.idom[block.id] == null) continue;

        for (block.instructions.items, 0..) |inst, ii| {
            const Ctx = struct {
                bad: ?ir.VReg = null,
                param_count: u32,
                def_loc: *const @TypeOf(def_loc),
                dom: *const analysis.DomTree,
                cur_block: ir.BlockId,
                cur_idx: u32,
                cur_op: ir.Inst.Op,
                nblocks: usize,
            };
            var c = Ctx{
                .param_count = param_count,
                .def_loc = &def_loc,
                .dom = &dom,
                .cur_block = block.id,
                .cur_idx = @intCast(ii),
                .cur_op = inst.op,
                .nblocks = nblocks,
            };
            forEachOperand(inst, &c, struct {
                fn cb(ptr: *Ctx, v: ir.VReg) void {
                    if (ptr.bad != null) return;
                    if (v < ptr.param_count) return; // function parameter
                    const def = ptr.def_loc.get(v) orelse {
                        ptr.bad = v;
                        return;
                    };
                    // For phi: the operand for edge `block = B` must dominate
                    // the END of B, not the phi's own block.
                    if (ptr.cur_op == .phi) {
                        // Find the matching edge to determine the predecessor.
                        // We re-scan since forEachOperand doesn't pass the
                        // edge metadata; phi edges are small in practice.
                        for (ptr.cur_op.phi) |edge| {
                            if (edge.val != v) continue;
                            if (def.block == edge.block) return; // same block: OK
                            if (ptr.dom.dominates(def.block, edge.block)) return;
                            ptr.bad = v;
                            return;
                        }
                        // Operand isn't on any edge — fall through to default
                        // dom check, which will likely fail.
                    }
                    if (def.block == ptr.cur_block) {
                        if (def.idx < ptr.cur_idx) return;
                        ptr.bad = v;
                        return;
                    }
                    if (!ptr.dom.dominates(def.block, ptr.cur_block)) {
                        ptr.bad = v;
                        return;
                    }
                }
            }.cb);
            if (c.bad) |v| {
                last_failure = .{
                    .kind = error.UnboundVRegUse,
                    .func_index = func_index,
                    .block = block.id,
                    .inst_index = @intCast(ii),
                    .vreg = v,
                    .detail = "use is not dominated by a definition",
                };
                return error.UnboundVRegUse;
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

fn newReturnBlock(func: *ir.IrFunction, ret_v: ?ir.VReg) !ir.BlockId {
    const b = try func.newBlock();
    try func.getBlock(b).append(.{ .op = .{ .ret = ret_v } });
    return b;
}

test "verifier: empty function passes" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    try verifyFunction(&func, 0, .after_each_pass, a);
}

test "verifier: minimal ret-only block passes" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    _ = try newReturnBlock(&func, null);
    try verifyFunction(&func, 0, .after_each_pass, a);
}

test "verifier: detects missing terminator" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v, .type = .i32, .op = .{ .iconst_32 = 7 } });
    try testing.expectError(error.MissingTerminator, verifyFunction(&func, 0, .after_each_pass, a));
    try testing.expectEqual(@as(?ir.BlockId, b), last_failure.block);
}

test "verifier: detects multiple terminators" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b0).append(.{ .op = .{ .@"unreachable" = {} } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b1).addPredecessor(b0);
    try testing.expectError(error.MultipleTerminators, verifyFunction(&func, 0, .after_each_pass, a));
}

test "verifier: detects dangling block ref" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = 99 } });
    try testing.expectError(error.DanglingBlockRef, verifyFunction(&func, 0, .after_each_pass, a));
}

test "verifier: detects stale predecessor" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    // b1 falsely claims b0 as a predecessor.
    try func.getBlock(b1).addPredecessor(b0);
    try testing.expectError(error.StalePredecessor, verifyFunction(&func, 0, .after_each_pass, a));
}

test "verifier: detects missing predecessor" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    // Deliberately omit b1.addPredecessor(b0).
    try testing.expectError(error.MissingPredecessor, verifyFunction(&func, 0, .after_each_pass, a));
}

test "verifier: detects def uniqueness violation" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(b).append(.{ .dest = v, .type = .i32, .op = .{ .iconst_32 = 2 } });
    try func.getBlock(b).append(.{ .op = .{ .ret = v } });
    try testing.expectError(error.VRegDefinedTwice, verifyFunction(&func, 0, .after_each_pass, a));
    try testing.expectEqual(@as(?ir.VReg, v), last_failure.vreg);
}

test "verifier: detects unbound vreg use (non-dominating def)" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    // CFG:
    //   b0 -> {b1, b2}; b1 defines v; b2 reads v (NOT dominated by b1).
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const cond = func.newVReg();
    const v = func.newVReg();
    try func.getBlock(b0).append(.{ .dest = cond, .type = .i32, .op = .{ .iconst_32 = 0 } });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .dest = v, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = v } }); // bad: v not defined here
    try func.getBlock(b1).addPredecessor(b0);
    try func.getBlock(b2).addPredecessor(b0);
    try testing.expectError(error.UnboundVRegUse, verifyFunction(&func, 0, .after_each_pass, a));
    try testing.expectEqual(@as(?ir.VReg, v), last_failure.vreg);
}

test "verifier: dominating def across blocks passes" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    // b0 defines v; b0 -> b1 returns v (b0 dominates b1).
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(b0).append(.{ .dest = v, .type = .i32, .op = .{ .iconst_32 = 42 } });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = v } });
    try func.getBlock(b1).addPredecessor(b0);
    try verifyFunction(&func, 0, .after_each_pass, a);
}

test "verifier: function parameter use without def is OK" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 1, 1, 1);
    defer func.deinit();
    // Param 0 is vreg 0 (no defining instruction); return it.
    _ = func.newVReg(); // reserve vreg 0 for the param
    const b = try func.newBlock();
    try func.getBlock(b).append(.{ .op = .{ .ret = 0 } });
    try verifyFunction(&func, 0, .after_each_pass, a);
}

test "verifier: off mode is a no-op even on broken IR" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    // Block with no terminator — would normally trip MissingTerminator.
    try func.getBlock(b).append(.{ .op = .{ .local_get = 0 } });
    try verifyFunction(&func, 0, .off, a);
}
