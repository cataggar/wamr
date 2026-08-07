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
//! Paranoid-mode checks (issue #624 stretch invariants 7, 8, 9, 10, plus #738):
//!
//!   7. **Operand-type sanity** (#628). Each VReg operand's recorded width
//!      matches the producing instruction's result width.
//!   8. **Loop-info consistency** (#629). The natural-loop forest is well
//!      formed: every header dominates every member block, and every latch
//!      is a member dominated by the header.
//!   9. **Dominator-tree structure** (#629). Entry has `idom == null`,
//!      reachable blocks have a post-order number, and dominance is
//!      reflexive. (A true freshness check vs a cross-pass cache is gated
//!      on a future pass-pipeline cache landing — see TODO in
//!      `checkDomTreeStructure`.)
//!  10. **Live-range monotonicity** (#629). For the default sequential
//!      block order, every `LiveRange` has `start <= end`.
//!  11. **Load-forwarding soundness** (#738, loop-complete #794). For a load
//!      result reused across a CFG edge, no aliasing store or coarse load
//!      barrier may run between the load and the use on any path —
//!      **including loop back-edges and the use block's own post-load
//!      effect** — unless the path also re-executes the load (the
//!      "avoid-def" refinement). Mirrors the runtime forwarder gate
//!      (`loadForwardCrossesEffect`, `passes.zig`) as an independent
//!      re-derivation. NOTE: this is a SOUND OVER-APPROXIMATION — a load
//!      value legitimately reused as a snapshot across an aliasing store is
//!      indistinguishable in final IR from an unsound forward, so it can
//!      false-positive on real (LLVM-optimised) wasm and must stay opt-in,
//!      never a default-compile or real-corpus CI gate. Runs under both
//!      `.paranoid` and the dedicated `.load_forwarding` mode.
//!
//! Paranoid checks re-derive analyses (dom, loops, liveness) for every
//! invocation, so they're intentionally opt-in: even safety builds default
//! to `after_each_pass` and only `wamrc --verify-ir=paranoid` (or a fuzz
//! lane) flips them on. The `.load_forwarding` mode runs the base checks plus
//! only check 11, so a corpus CI gate can exercise load-forwarding soundness
//! without the stricter operand-width / dom-tree re-derivations.
//!
//! Wiring: `passes.runPassesWithOptions` calls `verifyFunction` after every
//! pass invocation when `opts.verify_mode != .off`, and annotates the failure
//! with the pass name before propagating.

const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");
const alias_class = @import("alias_class.zig");

pub const VerifyMode = enum {
    /// Skip verification entirely (no-op, zero cost).
    off,
    /// Run checks 1, 2, 4, 5, 6 after every pass.
    after_each_pass,
    /// Run the base checks plus only the load-forwarding soundness check
    /// (#794). Cheaper than `paranoid` and, unlike `paranoid`, does not trip
    /// on the stricter operand-width / dom-tree re-derivations, so it can gate
    /// a real-corpus CI run dedicated to load-forwarding regressions.
    load_forwarding,
    /// Additionally run paranoid re-derivation checks: operand widths,
    /// loop / dom-tree consistency, live-range monotonicity, and
    /// load-forwarding soundness.
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
    /// A VReg operand's recorded type does not match the type the
    /// consuming instruction expects for that role (check 7, paranoid).
    OperandTypeMismatch,
    /// A loop header does not dominate a block listed in its body, or a
    /// latch is not contained / not dominated (check 8, paranoid).
    LoopInvariantBroken,
    /// The dominator tree fails a structural soundness invariant —
    /// entry has a non-null idom, a reachable block lacks a post-order
    /// number, or reflexivity fails (check 9, paranoid).
    DomTreeInconsistent,
    /// A live range has `end < start`, indicating the underlying live-
    /// range numbering disagrees with the program's def-before-use
    /// structure (check 10, paranoid).
    LiveRangeInverted,
    /// A load result is used across a CFG edge along a path containing an
    /// aliasing store or load barrier (check 11, paranoid).
    LoadForwardingUnsound,
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
        .lea => |l| {
            cb(ctx, l.base);
            cb(ctx, l.index);
        },
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

        // #672 EH ops.
        .try_table_begin, .try_table_end => {},
        .throw => |th| for (th.args) |a| cb(ctx, a),
        .throw_ref => |v| cb(ctx, v),
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
        .throw,
        .throw_ref,
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
    if (mode == .paranoid) {
        try checkOperandWidths(func, func_index, allocator);
        try checkDomTreeStructure(func, func_index, allocator);
        try checkLoopInfo(func, func_index, allocator);
        try checkLiveRangeMonotonicity(func, func_index, allocator);
    }
    if (mode == .paranoid or mode == .load_forwarding) {
        try verifyLoadForwardingSoundness(func, func_index, allocator);
    }
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

const BlockRefCtx = struct {
    ok: bool = true,
    bad: ir.BlockId = 0,
    nblocks: u32,
};

fn checkBlockRefs(func: *const ir.IrFunction, func_index: u32) VerifyError!void {
    const nblocks: u32 = @intCast(func.blocks.items.len);
    for (func.blocks.items) |block| {
        for (block.instructions.items, 0..) |inst, ii| {
            var c = BlockRefCtx{ .nblocks = nblocks };
            forEachSuccessor(inst.op, &c, struct {
                fn cb(ptr: *BlockRefCtx, target: ir.BlockId) void {
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
    const cfg_cache = analysis.currentCfgAnalysisCache();
    var owned_dom: ?analysis.DomTree = null;
    defer if (owned_dom) |*dom| dom.deinit();
    const dom: *const analysis.DomTree = if (cfg_cache) |cache|
        try cache.getDominators(func)
    else blk: {
        owned_dom = try analysis.computeDominators(func, allocator);
        break :blk &owned_dom.?;
    };

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
                .dom = dom,
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

// ── Check 7: operand-width sanity (paranoid only) ───────────────────────

/// Thread-local scratch used to format dynamic detail strings (e.g.
/// "operand %3 has type v128, expected i32"). The single-threaded
/// verifier guarantees no two checks share this buffer concurrently.
threadlocal var detail_buf: [512]u8 = undefined;

/// Build a dense `[]?ir.IrType` keyed by VReg. Params (vregs
/// `0..param_count`) get types from `func.local_types` when present;
/// otherwise their slot stays `null` and operand checks against them
/// are skipped. Defs from `Inst.dest` use `inst.type`; `parallel_copy`
/// dsts use the per-pair `ParallelCopy.ty`.
fn buildVRegTypeMap(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) std.mem.Allocator.Error![]?ir.IrType {
    const n: usize = @intCast(func.next_vreg);
    const types = try allocator.alloc(?ir.IrType, n);
    @memset(types, null);

    if (func.local_types) |lt| {
        const np: usize = @intCast(func.param_count);
        const lim = @min(np, n);
        for (0..lim) |i| types[i] = lt[i];
    }

    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            if (inst.dest) |d| {
                if (@as(usize, @intCast(d)) < n) types[@intCast(d)] = inst.type;
            }
            switch (inst.op) {
                .parallel_copy => |pairs| for (pairs) |p| {
                    if (@as(usize, @intCast(p.dst)) < n) types[@intCast(p.dst)] = p.ty;
                },
                else => {},
            }
        }
    }
    return types;
}

const OperandCheckCtx = struct {
    func: *const ir.IrFunction,
    func_index: u32,
    types: []const ?ir.IrType,
    block: ir.BlockId,
    inst_index: u32,

    fn typeOf(self: OperandCheckCtx, v: ir.VReg) ?ir.IrType {
        if (@as(usize, @intCast(v)) >= self.types.len) return null;
        return self.types[@intCast(v)];
    }

    fn expect(
        self: OperandCheckCtx,
        v: ir.VReg,
        expected: ir.IrType,
        role: []const u8,
    ) VerifyError!void {
        const got = self.typeOf(v) orelse return; // unknown (param w/o local_types, or unbound — flagged by check 1)
        if (got == expected) return;
        const written = std.fmt.bufPrint(
            &detail_buf,
            "operand %{d} ({s}) has type {s}, expected {s}",
            .{ v, role, @tagName(got), @tagName(expected) },
        ) catch &detail_buf;
        last_failure = .{
            .kind = error.OperandTypeMismatch,
            .func_index = self.func_index,
            .block = self.block,
            .inst_index = self.inst_index,
            .vreg = v,
            .detail = written,
        };
        return error.OperandTypeMismatch;
    }

    /// Both operands must share the same width as each other. Used for
    /// binops/cmps where `inst.type` is the operand width (int cmps record
    /// operand width in `Inst.type` even though the logical result is i32 —
    /// see frontend.zig for the convention).
    fn expectBin(
        self: OperandCheckCtx,
        lhs: ir.VReg,
        rhs: ir.VReg,
        expected: ir.IrType,
    ) VerifyError!void {
        try self.expect(lhs, expected, "lhs");
        try self.expect(rhs, expected, "rhs");
    }
};

fn checkOperandWidths(
    func: *const ir.IrFunction,
    func_index: u32,
    allocator: std.mem.Allocator,
) VerifyError!void {
    const types = try buildVRegTypeMap(func, allocator);
    defer allocator.free(types);

    for (func.blocks.items) |block| {
        for (block.instructions.items, 0..) |inst, ii| {
            const ctx = OperandCheckCtx{
                .func = func,
                .func_index = func_index,
                .types = types,
                .block = block.id,
                .inst_index = @intCast(ii),
            };
            try checkOneInst(ctx, inst);
        }
    }
}

fn checkOneInst(ctx: OperandCheckCtx, inst: ir.Inst) VerifyError!void {
    switch (inst.op) {
        // Nullary / constants — no operands.
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

        // Integer + float binops. `inst.type` is the operand width.
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
        => |bin| try ctx.expectBin(bin.lhs, bin.rhs, inst.type),

        // Single-operand integer/float unops where `inst.type` is the
        // operand (and result) type.
        .clz, .ctz, .popcnt, .eqz => |v| try ctx.expect(v, inst.type, "operand"),
        .extend8_s, .extend16_s, .extend32_s => |v| try ctx.expect(v, inst.type, "operand"),
        .f_neg, .f_abs, .f_sqrt, .f_ceil, .f_floor, .f_trunc, .f_nearest => |v| try ctx.expect(v, inst.type, "operand"),

        // Width-changing conversions: operand width fixed per-op, dest is
        // `inst.type`.
        .wrap_i64 => |v| try ctx.expect(v, .i64, "operand"),
        .extend_i32_s, .extend_i32_u => |v| try ctx.expect(v, .i32, "operand"),
        .trunc_f32_s, .trunc_f32_u, .trunc_sat_f32_s, .trunc_sat_f32_u => |v| try ctx.expect(v, .f32, "operand"),
        .trunc_f64_s, .trunc_f64_u, .trunc_sat_f64_s, .trunc_sat_f64_u => |v| try ctx.expect(v, .f64, "operand"),
        .convert_i32_s, .convert_i32_u => |v| try ctx.expect(v, .i32, "operand"),
        .convert_i64_s, .convert_i64_u => |v| try ctx.expect(v, .i64, "operand"),
        .demote_f64 => |v| try ctx.expect(v, .f64, "operand"),
        .promote_f32 => |v| try ctx.expect(v, .f32, "operand"),
        // `reinterpret`, `convert_s`, `convert_u` have ambiguous operand
        // widths under the current IR (the op is reused across i32↔f32 /
        // i64↔f64); skip rather than risk false positives.
        .reinterpret, .convert_s, .convert_u => {},

        // Locals / globals — value type isn't recorded on the op (only the
        // local/global index), so we can't check here. Mem2reg / promotion
        // passes elsewhere already enforce typed assignment.
        .local_set, .global_set => {},

        // Loads: `base` is the linear-memory pointer, always i32 in wasm32.
        .load => |ld| try ctx.expect(ld.base, .i32, "base"),
        // Fused LEA: base/index share the op's integer width (i32/i64).
        .lea => |l| {
            try ctx.expect(l.base, inst.type, "base");
            try ctx.expect(l.index, inst.type, "index");
        },
        .v128_load => |ld| try ctx.expect(ld.base, .i32, "base"),
        .v128_load_splat => |ld| try ctx.expect(ld.base, .i32, "base"),
        .v128_load_zero => |ld| try ctx.expect(ld.base, .i32, "base"),
        .v128_load_extend => |ld| try ctx.expect(ld.base, .i32, "base"),
        .v128_load_lane => |ld| {
            try ctx.expect(ld.base, .i32, "base");
            try ctx.expect(ld.vector, .v128, "vector");
        },

        // Stores: base is i32; value width is the store's `inst.type`.
        .store => |st| {
            try ctx.expect(st.base, .i32, "base");
            try ctx.expect(st.val, inst.type, "val");
        },
        .v128_store => |st| {
            try ctx.expect(st.base, .i32, "base");
            try ctx.expect(st.val, .v128, "val");
        },
        .v128_store_lane => |st| {
            try ctx.expect(st.base, .i32, "base");
            try ctx.expect(st.vector, .v128, "vector");
        },

        // Control flow.
        .br_if => |bi| try ctx.expect(bi.cond, .i32, "cond"),
        .br_table => |bt| try ctx.expect(bt.index, .i32, "index"),

        // Returns: would need the function signature to type-check; defer.
        .ret, .ret_multi => {},

        // Calls: arg types would need the module's `func_types`; defer to a
        // follow-up. (`elem_idx`/`func_ref` are integer table/func indices,
        // always i32 in wasm32 and worth checking.)
        .call => {},
        .call_indirect => |ci| try ctx.expect(ci.elem_idx, .i32, "elem_idx"),
        .call_ref => |cr| try ctx.expect(cr.func_ref, .i32, "func_ref"),

        // Select: cond i32, branches share `inst.type`.
        .select => |sel| {
            try ctx.expect(sel.cond, .i32, "cond");
            try ctx.expect(sel.if_true, inst.type, "if_true");
            try ctx.expect(sel.if_false, inst.type, "if_false");
        },

        // Atomics. base is i32; val/expected/replacement match the access
        // width via `inst.type`. atomic_notify count is i32; atomic_wait
        // timeout is always i64 (per spec) and `expected` matches inst.type.
        .atomic_load => |al| try ctx.expect(al.base, .i32, "base"),
        .atomic_store => |st| {
            try ctx.expect(st.base, .i32, "base");
            try ctx.expect(st.val, inst.type, "val");
        },
        .atomic_rmw => |ar| {
            try ctx.expect(ar.base, .i32, "base");
            try ctx.expect(ar.val, inst.type, "val");
        },
        .atomic_cmpxchg => |ac| {
            try ctx.expect(ac.base, .i32, "base");
            try ctx.expect(ac.expected, inst.type, "expected");
            try ctx.expect(ac.replacement, inst.type, "replacement");
        },
        .atomic_notify => |an| {
            try ctx.expect(an.base, .i32, "base");
            try ctx.expect(an.count, .i32, "count");
        },
        .atomic_wait => |aw| {
            try ctx.expect(aw.base, .i32, "base");
            try ctx.expect(aw.expected, inst.type, "expected");
            try ctx.expect(aw.timeout, .i64, "timeout");
        },

        // Bulk memory ops — all operands are i32 in wasm32.
        .memory_copy => |mc| {
            try ctx.expect(mc.dst, .i32, "dst");
            try ctx.expect(mc.src, .i32, "src");
            try ctx.expect(mc.len, .i32, "len");
        },
        .memory_fill => |mf| {
            try ctx.expect(mf.dst, .i32, "dst");
            try ctx.expect(mf.val, .i32, "val");
            try ctx.expect(mf.len, .i32, "len");
        },
        .memory_init => |mi| {
            try ctx.expect(mi.dst, .i32, "dst");
            try ctx.expect(mi.src, .i32, "src");
            try ctx.expect(mi.len, .i32, "len");
        },
        .memory_grow => |v| try ctx.expect(v, .i32, "delta"),

        // Tables. Index always i32; element-type-dependent operands skipped.
        .table_get => |tg| try ctx.expect(tg.idx, .i32, "idx"),
        .table_set => |ts| try ctx.expect(ts.idx, .i32, "idx"),
        .table_grow => |tg| try ctx.expect(tg.delta, .i32, "delta"),
        .table_init => |ti| {
            try ctx.expect(ti.dst, .i32, "dst");
            try ctx.expect(ti.src, .i32, "src");
            try ctx.expect(ti.len, .i32, "len");
        },

        // SIMD — vector operands are v128; scalar shift counts / splat
        // values use the documented scalar width per opcode.
        .v128_not => |v| try ctx.expect(v, .v128, "operand"),
        .v128_any_true => |v| try ctx.expect(v, .v128, "operand"),
        .v128_bitwise => |bin| try ctx.expectBin(bin.lhs, bin.rhs, .v128),
        .v128_bitselect => |sel| {
            try ctx.expect(sel.a, .v128, "a");
            try ctx.expect(sel.b, .v128, "b");
            try ctx.expect(sel.mask, .v128, "mask");
        },
        .simd_all_true => |op| try ctx.expect(op.vector, .v128, "vector"),
        .simd_bitmask => |op| try ctx.expect(op.vector, .v128, "vector"),

        .i32x4_binop => |bin| try ctx.expectBin(bin.lhs, bin.rhs, .v128),
        .f32x4_binop => |bin| try ctx.expectBin(bin.lhs, bin.rhs, .v128),
        .i8x16_binop => |bin| try ctx.expectBin(bin.lhs, bin.rhs, .v128),
        .i16x8_binop => |bin| try ctx.expectBin(bin.lhs, bin.rhs, .v128),
        .i64x2_binop => |bin| try ctx.expectBin(bin.lhs, bin.rhs, .v128),
        .f64x2_binop => |bin| try ctx.expectBin(bin.lhs, bin.rhs, .v128),

        .i32x4_unop => |un| try ctx.expect(un.vector, .v128, "vector"),
        .f32x4_unop => |un| try ctx.expect(un.vector, .v128, "vector"),
        .i8x16_unop => |un| try ctx.expect(un.vector, .v128, "vector"),
        .i16x8_unop => |un| try ctx.expect(un.vector, .v128, "vector"),
        .i64x2_unop => |un| try ctx.expect(un.vector, .v128, "vector"),
        .f64x2_unop => |un| try ctx.expect(un.vector, .v128, "vector"),

        .i32x4_extadd_pairwise_i16x8 => |op| try ctx.expect(op.vector, .v128, "vector"),
        .i16x8_extadd_pairwise_i8x16 => |op| try ctx.expect(op.vector, .v128, "vector"),
        .i32x4_extend_i16x8 => |op| try ctx.expect(op.vector, .v128, "vector"),
        .i16x8_extend_i8x16 => |op| try ctx.expect(op.vector, .v128, "vector"),
        .i64x2_extend_i32x4 => |op| try ctx.expect(op.vector, .v128, "vector"),
        .i32x4_trunc_sat => |op| try ctx.expect(op.vector, .v128, "vector"),
        .f32x4_convert_i32x4 => |op| try ctx.expect(op.vector, .v128, "vector"),
        .f32x4_demote_f64x2_zero => |op| try ctx.expect(op.vector, .v128, "vector"),
        .f64x2_convert_low_i32x4 => |op| try ctx.expect(op.vector, .v128, "vector"),
        .f64x2_promote_low_f32x4 => |op| try ctx.expect(op.vector, .v128, "vector"),

        .i32x4_dot_i16x8_s => |op| try ctx.expectBin(op.lhs, op.rhs, .v128),
        .i32x4_extmul_i16x8 => |op| try ctx.expectBin(op.lhs, op.rhs, .v128),
        .i16x8_extmul_i8x16 => |op| try ctx.expectBin(op.lhs, op.rhs, .v128),
        .i64x2_extmul_i32x4 => |op| try ctx.expectBin(op.lhs, op.rhs, .v128),
        .i8x16_shuffle => |op| try ctx.expectBin(op.lhs, op.rhs, .v128),
        .i8x16_narrow_i16x8 => |op| try ctx.expectBin(op.lhs, op.rhs, .v128),
        .i16x8_narrow_i32x4 => |op| try ctx.expectBin(op.lhs, op.rhs, .v128),

        .i8x16_swizzle => |op| {
            try ctx.expect(op.vector, .v128, "vector");
            try ctx.expect(op.indices, .v128, "indices");
        },

        .i8x16_shift => |shift| {
            try ctx.expect(shift.vector, .v128, "vector");
            try ctx.expect(shift.count, .i32, "count");
        },
        .i16x8_shift => |shift| {
            try ctx.expect(shift.vector, .v128, "vector");
            try ctx.expect(shift.count, .i32, "count");
        },
        .i32x4_shift => |shift| {
            try ctx.expect(shift.vector, .v128, "vector");
            try ctx.expect(shift.count, .i32, "count");
        },
        .i64x2_shift => |shift| {
            try ctx.expect(shift.vector, .v128, "vector");
            try ctx.expect(shift.count, .i32, "count");
        },

        .i32x4_extract_lane => |lane| try ctx.expect(lane.vector, .v128, "vector"),
        .f32x4_extract_lane => |lane| try ctx.expect(lane.vector, .v128, "vector"),
        .i8x16_extract_lane => |lane| try ctx.expect(lane.vector, .v128, "vector"),
        .i16x8_extract_lane => |lane| try ctx.expect(lane.vector, .v128, "vector"),
        .i64x2_extract_lane => |lane| try ctx.expect(lane.vector, .v128, "vector"),
        .f64x2_extract_lane => |lane| try ctx.expect(lane.vector, .v128, "vector"),
        .i32x4_replace_lane => |lane| {
            try ctx.expect(lane.vector, .v128, "vector");
            try ctx.expect(lane.val, .i32, "val");
        },
        .f32x4_replace_lane => |lane| {
            try ctx.expect(lane.vector, .v128, "vector");
            try ctx.expect(lane.val, .f32, "val");
        },
        .i8x16_replace_lane => |lane| {
            try ctx.expect(lane.vector, .v128, "vector");
            try ctx.expect(lane.val, .i32, "val"); // i8 lifted to i32
        },
        .i16x8_replace_lane => |lane| {
            try ctx.expect(lane.vector, .v128, "vector");
            try ctx.expect(lane.val, .i32, "val"); // i16 lifted to i32
        },
        .i64x2_replace_lane => |lane| {
            try ctx.expect(lane.vector, .v128, "vector");
            try ctx.expect(lane.val, .i64, "val");
        },
        .f64x2_replace_lane => |lane| {
            try ctx.expect(lane.vector, .v128, "vector");
            try ctx.expect(lane.val, .f64, "val");
        },

        .i32x4_splat => |v| try ctx.expect(v, .i32, "scalar"),
        .f32x4_splat => |v| try ctx.expect(v, .f32, "scalar"),
        .i8x16_splat => |v| try ctx.expect(v, .i32, "scalar"), // i8 lifted to i32
        .i16x8_splat => |v| try ctx.expect(v, .i32, "scalar"), // i16 lifted to i32
        .i64x2_splat => |v| try ctx.expect(v, .i64, "scalar"),
        .f64x2_splat => |v| try ctx.expect(v, .f64, "scalar"),

        // Phi: every incoming edge must produce a value of the phi's type.
        .phi => |edges| for (edges) |e| try ctx.expect(e.val, inst.type, "phi edge"),

        // Parallel copy: each pair carries its own width.
        .parallel_copy => |pairs| for (pairs) |p| try ctx.expect(p.src, p.ty, "src"),

        // #672 EH ops: try_table_begin / try_table_end carry no vreg
        // operands. `throw`'s args are popped at lowering time; their
        // widths are derived from the tag's func_type so we don't try
        // to re-typecheck them here in commit 2. `throw_ref` carries a
        // single exnref which we model as i32 for now (no dedicated
        // exnref width yet).
        .try_table_begin, .try_table_end, .throw => {},
        .throw_ref => |v| try ctx.expect(v, .i32, "exnref"),
    }
}

// ── Check 8: loop-info consistency (paranoid only, #629) ────────────────

fn checkLoopInfo(
    func: *const ir.IrFunction,
    func_index: u32,
    allocator: std.mem.Allocator,
) VerifyError!void {
    const cfg_cache = analysis.currentCfgAnalysisCache();
    var owned_dom: ?analysis.DomTree = null;
    defer if (owned_dom) |*dom| dom.deinit();
    const dom: *const analysis.DomTree = if (cfg_cache) |cache|
        cache.getDominators(func) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
        }
    else blk: {
        owned_dom = analysis.computeDominators(func, allocator) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
        };
        break :blk &owned_dom.?;
    };

    var forest = analysis.computeLoopsCached(func, dom, allocator, cfg_cache) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer forest.deinit();

    try verifyLoopForest(func_index, dom, &forest);
}

/// Validate a loop forest against a dominator tree. Split out so tests
/// can pass hand-rolled forests with deliberately broken invariants.
fn verifyLoopForest(
    func_index: u32,
    dom: *const analysis.DomTree,
    forest: *const analysis.LoopForest,
) VerifyError!void {
    for (forest.loops) |loop| {
        for (loop.blocks) |b| {
            if (!dom.dominates(loop.header, b)) {
                const written = std.fmt.bufPrint(
                    &detail_buf,
                    "loop header #{d} does not dominate body block #{d}",
                    .{ loop.header, b },
                ) catch &detail_buf;
                last_failure = .{
                    .kind = error.LoopInvariantBroken,
                    .func_index = func_index,
                    .block = b,
                    .detail = written,
                };
                return error.LoopInvariantBroken;
            }
        }
        for (loop.latches) |latch| {
            if (!loop.containsBlock(latch)) {
                const written = std.fmt.bufPrint(
                    &detail_buf,
                    "loop header #{d}: latch #{d} not in loop.blocks",
                    .{ loop.header, latch },
                ) catch &detail_buf;
                last_failure = .{
                    .kind = error.LoopInvariantBroken,
                    .func_index = func_index,
                    .block = latch,
                    .detail = written,
                };
                return error.LoopInvariantBroken;
            }
            if (!dom.dominates(loop.header, latch)) {
                const written = std.fmt.bufPrint(
                    &detail_buf,
                    "loop header #{d} does not dominate latch #{d}",
                    .{ loop.header, latch },
                ) catch &detail_buf;
                last_failure = .{
                    .kind = error.LoopInvariantBroken,
                    .func_index = func_index,
                    .block = latch,
                    .detail = written,
                };
                return error.LoopInvariantBroken;
            }
        }
    }
}

// ── Check 9: dominator-tree structural soundness (paranoid only, #629) ──

fn checkDomTreeStructure(
    func: *const ir.IrFunction,
    func_index: u32,
    allocator: std.mem.Allocator,
) VerifyError!void {
    const cfg_cache = analysis.currentCfgAnalysisCache();
    var owned_dom: ?analysis.DomTree = null;
    defer if (owned_dom) |*dom| dom.deinit();
    const dom: *const analysis.DomTree = if (cfg_cache) |cache|
        cache.getDominators(func) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
        }
    else blk: {
        owned_dom = analysis.computeDominators(func, allocator) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
        };
        break :blk &owned_dom.?;
    };
    try verifyDomTreeStructure(func, func_index, dom);

    // TODO: once any pass caches a `DomTree` on `IrFunction` across the
    // pipeline, also diff that cached tree against the freshly computed
    // one and emit `error.DomTreeInconsistent` with detail
    // "stale dominator cache" on mismatch.
}

fn verifyDomTreeStructure(
    func: *const ir.IrFunction,
    func_index: u32,
    dom: *const analysis.DomTree,
) VerifyError!void {
    // Entry block's idom convention: `computeDominators` sets `idom[0] = 0`
    // (entry dominates itself); a `null` here means an empty / unreachable
    // entry. Anything else is structurally invalid.
    if (func.blocks.items.len > 0) {
        const e0 = dom.idom[0];
        if (e0 != null and e0.? != 0) {
            const written = std.fmt.bufPrint(
                &detail_buf,
                "entry idom must be self or null, got #{d}",
                .{e0.?},
            ) catch &detail_buf;
            last_failure = .{
                .kind = error.DomTreeInconsistent,
                .func_index = func_index,
                .block = 0,
                .detail = written,
            };
            return error.DomTreeInconsistent;
        }
    }

    for (0..func.blocks.items.len) |i| {
        const b: ir.BlockId = @intCast(i);
        const reachable = dom.idom[i] != null;
        if (reachable) {
            // Reachable blocks must have a post-order number and dominate
            // themselves (reflexivity is encoded in `DomTree.dominates`).
            if (dom.post_num[i] == null) {
                const written = std.fmt.bufPrint(
                    &detail_buf,
                    "reachable block #{d} has no post-order number",
                    .{b},
                ) catch &detail_buf;
                last_failure = .{
                    .kind = error.DomTreeInconsistent,
                    .func_index = func_index,
                    .block = b,
                    .detail = written,
                };
                return error.DomTreeInconsistent;
            }
            if (!dom.dominates(b, b)) {
                const written = std.fmt.bufPrint(
                    &detail_buf,
                    "block #{d} does not dominate itself (reflexivity failed)",
                    .{b},
                ) catch &detail_buf;
                last_failure = .{
                    .kind = error.DomTreeInconsistent,
                    .func_index = func_index,
                    .block = b,
                    .detail = written,
                };
                return error.DomTreeInconsistent;
            }
        }
    }
}

// ── Check 10: live-range monotonicity (paranoid only, #629) ─────────────

fn checkLiveRangeMonotonicity(
    func: *const ir.IrFunction,
    func_index: u32,
    allocator: std.mem.Allocator,
) VerifyError!void {
    const ranges = analysis.computeLiveRangesWithOrder(func, null, allocator) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer allocator.free(ranges);
    try verifyLiveRangeMonotonicity(func_index, ranges);
}

fn verifyLiveRangeMonotonicity(
    func_index: u32,
    ranges: []const analysis.LiveRange,
) VerifyError!void {
    for (ranges) |r| {
        if (r.end < r.start) {
            const written = std.fmt.bufPrint(
                &detail_buf,
                "vreg %{d}: live range end={d} precedes start={d}",
                .{ r.vreg, r.end, r.start },
            ) catch &detail_buf;
            last_failure = .{
                .kind = error.LiveRangeInverted,
                .func_index = func_index,
                .vreg = r.vreg,
                .detail = written,
            };
            return error.LiveRangeInverted;
        }
    }
}

// ── Check 11: load-forwarding soundness (#738, loop-complete #794) ──────
//
// Re-derive the load-availability invariant from final IR only. For a load
// `v = load[p]` whose SSA result is consumed in a *different* block, the value
// is stale — and the load was therefore forwarded/elided unsoundly — if some
// aliasing memory effect (an aliasing `store` or a coarse load barrier such as
// `call`) can run after the load and before the use *without the load being
// re-executed in between*.
//
// This mirrors the per-forward CFG-reachability gate the runtime forwarder
// applies (`loadForwardCrossesEffect`, src/compiler/ir/passes.zig) — an
// effecting block on any def→use path, **including loop back-edges and the use
// block's own post-load effect** — but is an *independent* re-derivation so a
// regression in any forwarding pass is caught rather than mirrored.
//
// The one refinement over the runtime gate is the "avoid-def" rule. The
// runtime gate only ever forwards *from a dominator that is outside the loop*,
// so its `a` block is never re-executed on a back-edge. The verifier's `a` is
// the surviving load's own def block, which may itself sit inside the loop; a
// path from the effect back to the use that passes through `a` re-executes the
// load and refreshes the value, so such effects must NOT be flagged. We model
// this with a backward BFS from the use block that treats the def block as a
// wall: the stamped blocks are exactly those that reach the use without
// re-executing the load.
//
// NB: this is a heuristic, not a universal SSA invariant — a value loaded into
// a wasm local and legitimately reused as a snapshot across an aliasing store
// is indistinguishable in final IR from an unsound forward. It is therefore an
// opt-in (`paranoid` / `load_forwarding`) check, not a default-compile one.
// Correctness relies on `checkSsaDominance` (run earlier) guaranteeing the def
// block dominates every use block, so a block that reaches the use avoiding the
// def is necessarily reachable *from* the def after the load executes. Blocks
// unreachable from entry break that argument and can yield a spurious flag, but
// the normal pipeline runs `scrubUnreachableBlocks` (wiping them to a single
// effect-free `unreachable` op) before this check ever sees them.

const LoadUseSite = struct { block: ir.BlockId, inst: u32 };

fn instHasAliasEffect(load_key: alias_class.LoadKey, op: ir.Inst.Op) bool {
    return switch (op) {
        .store => |st| alias_class.storeAliasesLoad(load_key, st),
        .v128_store => |st| alias_class.storeAliasesLoad(load_key, alias_class.storeRangeFromV128Store(st)),
        .v128_store_lane => |st| alias_class.storeAliasesLoad(load_key, alias_class.storeRangeFromV128StoreLane(st)),
        else => alias_class.opIsLoadBarrier(op),
    };
}

/// Index of the first instruction in `block[start..end)` carrying an aliasing
/// memory effect for `load_key`, or null if the slice is clean.
fn aliasEffectInRange(
    block: *const ir.BasicBlock,
    load_key: alias_class.LoadKey,
    start: usize,
    end_excl: usize,
) ?usize {
    const len = block.instructions.items.len;
    const lo = @min(start, len);
    const hi = @min(end_excl, len);
    if (lo >= hi) return null;
    for (block.instructions.items[lo..hi], lo..) |inst, ii| {
        if (instHasAliasEffect(load_key, inst.op)) return ii;
    }
    return null;
}

const LoadForwardingChecker = struct {
    func: *const ir.IrFunction,
    func_index: u32,
    allocator: std.mem.Allocator,
    successors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    predecessors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    /// Generation-stamped marker, length == block count. A block stamped with
    /// the current `gen` reaches the current use block without passing through
    /// the current def block.
    reach_stamp: []u32,
    gen: u32,
    /// BFS worklist; after the backward BFS in `checkUse` it holds exactly the
    /// stamped blocks (those that reach the use without re-executing the load).
    queue: *std.ArrayList(ir.BlockId),

    fn succsOf(self: *const LoadForwardingChecker, b: ir.BlockId) []const ir.BlockId {
        return if (self.successors.get(b)) |s| s else &[_]ir.BlockId{};
    }
    fn predsOf(self: *const LoadForwardingChecker, b: ir.BlockId) []const ir.BlockId {
        return if (self.predecessors.get(b)) |p| p else &[_]ir.BlockId{};
    }
    fn reaches(self: *const LoadForwardingChecker, b: ir.BlockId) bool {
        return b < self.reach_stamp.len and self.reach_stamp[b] == self.gen;
    }

    /// Verify one cross-block use of a load result. `a`/`di` are the load's
    /// def block and instruction index; `d`/`ui` are the use block and index.
    fn checkUse(
        self: *LoadForwardingChecker,
        load_vreg: ir.VReg,
        load_key: alias_class.LoadKey,
        a: ir.BlockId,
        di: u32,
        d: ir.BlockId,
        ui: u32,
    ) VerifyError!void {
        if (a == d) return;
        if (a >= self.reach_stamp.len or d >= self.reach_stamp.len) return;

        // Backward BFS from the use block over predecessors, treating the def
        // block `a` as a wall (never traversed). The stamped set is the blocks
        // that reach the use without re-executing the load.
        self.gen += 1;
        self.queue.clearRetainingCapacity();
        self.reach_stamp[d] = self.gen;
        try self.queue.append(self.allocator, d);
        var head: usize = 0;
        while (head < self.queue.items.len) : (head += 1) {
            const x = self.queue.items[head];
            for (self.predsOf(x)) |p| {
                if (p == a) continue;
                if (p >= self.reach_stamp.len) continue;
                if (self.reach_stamp[p] == self.gen) continue;
                self.reach_stamp[p] = self.gen;
                try self.queue.append(self.allocator, p);
            }
        }

        // Def block: an aliasing effect after the load is stale if control can
        // still reach the use without re-executing the load.
        var a_reaches_d = false;
        for (self.succsOf(a)) |s| {
            if (self.reaches(s)) {
                a_reaches_d = true;
                break;
            }
        }
        if (a_reaches_d) {
            const blk = &self.func.blocks.items[a];
            if (aliasEffectInRange(blk, load_key, @as(usize, di) + 1, blk.instructions.items.len)) |idx|
                return self.report(load_vreg, blk.instructions.items[idx].op, a, @intCast(idx), a, d);
        }

        // Use block: an aliasing effect strictly before the use is always
        // stale on the def→use path.
        {
            const blk = &self.func.blocks.items[d];
            if (aliasEffectInRange(blk, load_key, 0, ui)) |idx|
                return self.report(load_vreg, blk.instructions.items[idx].op, d, @intCast(idx), a, d);
        }

        // Use block: an aliasing effect at/after the use only matters when the
        // use block can loop back to itself without re-executing the load, in
        // which case the next iteration's use observes the stale value.
        var d_reaches_d = false;
        for (self.succsOf(d)) |s| {
            if (self.reaches(s)) {
                d_reaches_d = true;
                break;
            }
        }
        if (d_reaches_d) {
            const blk = &self.func.blocks.items[d];
            if (aliasEffectInRange(blk, load_key, ui, blk.instructions.items.len)) |idx|
                return self.report(load_vreg, blk.instructions.items[idx].op, d, @intCast(idx), a, d);
        }

        // Interior corridor: every block that reaches the use avoiding the def
        // (the stamped set), excluding the use block itself (the def block is
        // never stamped — it is the BFS wall).
        for (self.queue.items) |b| {
            if (b == d) continue;
            const blk = &self.func.blocks.items[b];
            if (aliasEffectInRange(blk, load_key, 0, blk.instructions.items.len)) |idx|
                return self.report(load_vreg, blk.instructions.items[idx].op, b, @intCast(idx), a, d);
        }
    }

    fn report(
        self: *const LoadForwardingChecker,
        load_vreg: ir.VReg,
        offending_op: ir.Inst.Op,
        offending_block: ir.BlockId,
        offending_inst: u32,
        def_block: ir.BlockId,
        use_block: ir.BlockId,
    ) VerifyError!void {
        const written = std.fmt.bufPrint(
            &detail_buf,
            "load %{d} (def #{d}) crosses {s} at #{d}:{d} before use in #{d}",
            .{ load_vreg, def_block, @tagName(std.meta.activeTag(offending_op)), offending_block, offending_inst, use_block },
        ) catch &detail_buf;
        last_failure = .{
            .kind = error.LoadForwardingUnsound,
            .func_index = self.func_index,
            .block = offending_block,
            .inst_index = offending_inst,
            .vreg = load_vreg,
            .detail = written,
        };
        return error.LoadForwardingUnsound;
    }
};

fn verifyLoadForwardingSoundness(
    func: *const ir.IrFunction,
    func_index: u32,
    allocator: std.mem.Allocator,
) VerifyError!void {
    const n_blocks = func.blocks.items.len;
    if (n_blocks == 0) return;

    var successors = try analysis.buildSuccessors(func, allocator);
    defer analysis.freeBlockIdMap(&successors, allocator);
    var predecessors = try analysis.buildPredecessors(func, allocator);
    defer analysis.freeBlockIdMap(&predecessors, allocator);

    const reach_stamp = try allocator.alloc(u32, n_blocks);
    defer allocator.free(reach_stamp);
    @memset(reach_stamp, 0);

    var queue: std.ArrayList(ir.BlockId) = .empty;
    defer queue.deinit(allocator);

    // Collect the use sites of every load result in a single pass so the
    // per-load scan below is O(uses), not O(instructions). Only load-result
    // vregs get a list, so non-load operands cost nothing.
    var uses = std.AutoHashMap(ir.VReg, std.ArrayList(LoadUseSite)).init(allocator);
    defer {
        var it = uses.valueIterator();
        while (it.next()) |list| list.deinit(allocator);
        uses.deinit();
    }

    for (func.blocks.items) |*block| {
        for (block.instructions.items) |inst| {
            if (inst.op == .load) {
                if (inst.dest) |dst| {
                    const gop = try uses.getOrPut(dst);
                    if (!gop.found_existing) gop.value_ptr.* = .empty;
                }
            }
        }
    }

    const Collect = struct {
        uses: *std.AutoHashMap(ir.VReg, std.ArrayList(LoadUseSite)),
        allocator: std.mem.Allocator,
        site: LoadUseSite,
        err: ?std.mem.Allocator.Error = null,

        fn cb(self: *@This(), v: ir.VReg) void {
            if (self.err != null) return;
            const list = self.uses.getPtr(v) orelse return;
            // Dedup repeated operands of the same instruction (e.g. `add v, v`).
            if (list.items.len != 0) {
                const last = list.items[list.items.len - 1];
                if (last.block == self.site.block and last.inst == self.site.inst) return;
            }
            list.append(self.allocator, self.site) catch |e| {
                self.err = e;
            };
        }
    };

    for (func.blocks.items) |*block| {
        for (block.instructions.items, 0..) |inst, ii| {
            var collect = Collect{
                .uses = &uses,
                .allocator = allocator,
                .site = .{ .block = block.id, .inst = @intCast(ii) },
            };
            forEachOperand(inst, &collect, Collect.cb);
            if (collect.err) |e| return e;
        }
    }

    var checker = LoadForwardingChecker{
        .func = func,
        .func_index = func_index,
        .allocator = allocator,
        .successors = &successors,
        .predecessors = &predecessors,
        .reach_stamp = reach_stamp,
        .gen = 0,
        .queue = &queue,
    };

    for (func.blocks.items) |*block| {
        for (block.instructions.items, 0..) |inst, ii| {
            if (inst.op != .load) continue;
            const dest = inst.dest orelse continue;
            const load_key = alias_class.LoadKey{ .mem = alias_class.memKeyFromLoad(inst.op.load) };
            const list = uses.get(dest) orelse continue;
            for (list.items) |use| {
                if (use.block == block.id) continue; // same-block use: not a cross-block forward
                try checker.checkUse(dest, load_key, block.id, @intCast(ii), use.block, use.inst);
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

test "verifier: LastFailure format includes one-byte detail" {
    const failure = LastFailure{ .kind = error.MissingTerminator, .detail = "x" };
    var buf: [128]u8 = undefined;
    const rendered = try std.fmt.bufPrint(&buf, "{f}", .{failure});

    try testing.expectEqualStrings("IR verifier: MissingTerminator — x", rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, " — x") != null);
}

test "verifier: load forwarding soundness permits clean cross-block reuse" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 1, 0, 1);
    defer func.deinit();

    const base = func.newVReg();
    const entry = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(exit).addPredecessor(entry);

    const loaded = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .op = .{ .br = exit } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = loaded } });

    try verifyFunction(&func, 0, .paranoid, a);
}

test "verifier: load forwarding soundness rejects cross-block barrier" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 1, 0, 1);
    defer func.deinit();

    const base = func.newVReg();
    const entry = try func.newBlock();
    const call_block = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(call_block).addPredecessor(entry);
    try func.getBlock(exit).addPredecessor(call_block);

    const loaded = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .op = .{ .br = call_block } });
    try func.getBlock(call_block).append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try func.getBlock(call_block).append(.{ .op = .{ .br = exit } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = loaded } });

    try verifyFunction(&func, 0, .after_each_pass, a);
    try testing.expectError(error.LoadForwardingUnsound, verifyFunction(&func, 0, .paranoid, a));
    try testing.expect(std.mem.indexOf(u8, last_failure.detail, "call") != null);
}

test "verifier: load forwarding soundness catches sibling barrier regression" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 3, 0, 3);
    defer func.deinit();

    const base = func.newVReg();
    const cond = func.newVReg();
    const elem = func.newVReg();
    const entry = try func.newBlock();
    const barrier = try func.newBlock();
    const merge = try func.newBlock();
    try func.getBlock(barrier).addPredecessor(entry);
    try func.getBlock(merge).addPredecessor(entry);
    try func.getBlock(merge).addPredecessor(barrier);

    const loaded = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = barrier, .else_block = merge } } });
    try func.getBlock(barrier).append(.{ .op = .{ .call_indirect = .{ .type_idx = 0, .elem_idx = elem } } });
    try func.getBlock(barrier).append(.{ .op = .{ .br = merge } });
    try func.getBlock(merge).append(.{ .op = .{ .ret = loaded } });

    try testing.expectError(error.LoadForwardingUnsound, verifyFunction(&func, 0, .paranoid, a));
    try testing.expect(std.mem.indexOf(u8, last_failure.detail, "call_indirect") != null);
}

// #794 loop-completeness regression cases. Each hand-builds the *post-pass*
// IR a buggy forwarder would emit (a single surviving load whose value is
// reused across an effect on a loop). param_count covers the undefined input
// vregs (base, cond); defined vregs (the load result, store value) come after.

test "verifier(#794): barrier in loop body after the use is flagged (func7463 shape)" {
    // entry: v = load[p]; br header
    // header: use v; br_if cond, body, exit
    // body:   call;       br header        <-- barrier reachable after the use
    // exit:   ret
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 2, 0, 0);
    defer func.deinit();
    const base = func.newVReg(); // 0 (param)
    const cond = func.newVReg(); // 1 (param)

    const entry = try func.newBlock();
    const header = try func.newBlock();
    const body = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(header).addPredecessor(entry);
    try func.getBlock(header).addPredecessor(body);
    try func.getBlock(body).addPredecessor(header);
    try func.getBlock(exit).addPredecessor(header);

    const loaded = func.newVReg();
    const used = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .op = .{ .br = header } });
    try func.getBlock(header).append(.{ .dest = used, .type = .i32, .op = .{ .add = .{ .lhs = loaded, .rhs = loaded } } });
    try func.getBlock(header).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = body, .else_block = exit } } });
    try func.getBlock(body).append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try func.getBlock(body).append(.{ .op = .{ .br = header } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = null } });

    // The simple def->use path (entry->header) is clean, so only a
    // loop-complete check catches this.
    try verifyFunction(&func, 0, .after_each_pass, a); // Check 11 not run
    try testing.expectError(error.LoadForwardingUnsound, verifyFunction(&func, 0, .load_forwarding, a));
    try testing.expect(std.mem.indexOf(u8, last_failure.detail, "call") != null);
}

test "verifier(#794): use block's own post-load store on a loop is flagged (fn2178 shape)" {
    // entry: v = load[p]; val = const; br loop
    // loop:  use v; store[p] = val; br_if cond, loop, exit   <-- store after the use
    // exit:  ret
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 2, 0, 0);
    defer func.deinit();
    const base = func.newVReg(); // 0 (param)
    const cond = func.newVReg(); // 1 (param)

    const entry = try func.newBlock();
    const loop = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(loop).addPredecessor(entry);
    try func.getBlock(loop).addPredecessor(loop);
    try func.getBlock(exit).addPredecessor(loop);

    const loaded = func.newVReg();
    const val = func.newVReg();
    const used = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .dest = val, .type = .i32, .op = .{ .iconst_32 = 7 } });
    try func.getBlock(entry).append(.{ .op = .{ .br = loop } });
    try func.getBlock(loop).append(.{ .dest = used, .type = .i32, .op = .{ .add = .{ .lhs = loaded, .rhs = loaded } } });
    try func.getBlock(loop).append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = val } } });
    try func.getBlock(loop).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = loop, .else_block = exit } } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = null } });

    try testing.expectError(error.LoadForwardingUnsound, verifyFunction(&func, 0, .load_forwarding, a));
    try testing.expect(std.mem.indexOf(u8, last_failure.detail, "store") != null);
}

test "verifier(#794): store after use accepted when the back-edge re-executes the load (avoid-def)" {
    // entry: val = const; br A
    // A:     v = load[p]; br D
    // D:     use v; store[p] = val; br_if cond, A, exit   <-- back-edge to A reloads v
    // exit:  ret
    // Sound: every path from the store back to the use re-runs the load in A,
    // so the use never observes a stale value. The naive runtime "B==d" clause
    // (no avoid-def) would wrongly flag this; the verifier's avoid-def must not.
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 2, 0, 0);
    defer func.deinit();
    const base = func.newVReg(); // 0 (param)
    const cond = func.newVReg(); // 1 (param)

    const entry = try func.newBlock();
    const blk_a = try func.newBlock();
    const blk_d = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(blk_a).addPredecessor(entry);
    try func.getBlock(blk_a).addPredecessor(blk_d);
    try func.getBlock(blk_d).addPredecessor(blk_a);
    try func.getBlock(exit).addPredecessor(blk_d);

    const val = func.newVReg();
    const loaded = func.newVReg();
    const used = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = val, .type = .i32, .op = .{ .iconst_32 = 7 } });
    try func.getBlock(entry).append(.{ .op = .{ .br = blk_a } });
    try func.getBlock(blk_a).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(blk_a).append(.{ .op = .{ .br = blk_d } });
    try func.getBlock(blk_d).append(.{ .dest = used, .type = .i32, .op = .{ .add = .{ .lhs = loaded, .rhs = loaded } } });
    try func.getBlock(blk_d).append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = val } } });
    try func.getBlock(blk_d).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = blk_a, .else_block = exit } } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = null } });

    try verifyFunction(&func, 0, .load_forwarding, a);
}

test "verifier(#794): store after use flagged when the back-edge skips the load (avoid-def)" {
    // Same as above but the back-edge targets D, so the load in A is NOT
    // re-executed and the next iteration's use is stale — must be flagged.
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 2, 0, 0);
    defer func.deinit();
    const base = func.newVReg(); // 0 (param)
    const cond = func.newVReg(); // 1 (param)

    const entry = try func.newBlock();
    const blk_a = try func.newBlock();
    const blk_d = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(blk_a).addPredecessor(entry);
    try func.getBlock(blk_d).addPredecessor(blk_a);
    try func.getBlock(blk_d).addPredecessor(blk_d);
    try func.getBlock(exit).addPredecessor(blk_d);

    const val = func.newVReg();
    const loaded = func.newVReg();
    const used = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = val, .type = .i32, .op = .{ .iconst_32 = 7 } });
    try func.getBlock(entry).append(.{ .op = .{ .br = blk_a } });
    try func.getBlock(blk_a).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(blk_a).append(.{ .op = .{ .br = blk_d } });
    try func.getBlock(blk_d).append(.{ .dest = used, .type = .i32, .op = .{ .add = .{ .lhs = loaded, .rhs = loaded } } });
    try func.getBlock(blk_d).append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = val } } });
    try func.getBlock(blk_d).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = blk_d, .else_block = exit } } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = null } });

    try testing.expectError(error.LoadForwardingUnsound, verifyFunction(&func, 0, .load_forwarding, a));
    try testing.expect(std.mem.indexOf(u8, last_failure.detail, "store") != null);
}

test "verifier(#794): non-aliasing store in the loop body is accepted (alias-aware)" {
    // entry: v = load[p+0]; val = const; br loop
    // loop:  use v; store[p+64] = val; br_if cond, loop, exit  <-- disjoint range
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 2, 0, 0);
    defer func.deinit();
    const base = func.newVReg(); // 0 (param)
    const cond = func.newVReg(); // 1 (param)

    const entry = try func.newBlock();
    const loop = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(loop).addPredecessor(entry);
    try func.getBlock(loop).addPredecessor(loop);
    try func.getBlock(exit).addPredecessor(loop);

    const loaded = func.newVReg();
    const val = func.newVReg();
    const used = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .dest = val, .type = .i32, .op = .{ .iconst_32 = 7 } });
    try func.getBlock(entry).append(.{ .op = .{ .br = loop } });
    try func.getBlock(loop).append(.{ .dest = used, .type = .i32, .op = .{ .add = .{ .lhs = loaded, .rhs = loaded } } });
    try func.getBlock(loop).append(.{ .op = .{ .store = .{ .base = base, .offset = 64, .size = 4, .val = val } } });
    try func.getBlock(loop).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = loop, .else_block = exit } } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = null } });

    try verifyFunction(&func, 0, .load_forwarding, a);
}

test "verifier(#794): barrier-free loop reusing a dominating load is accepted" {
    // entry: v = load[p]; br header
    // header: use v; br_if cond, header, exit   <-- loop, no effects at all
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 2, 0, 0);
    defer func.deinit();
    const base = func.newVReg(); // 0 (param)
    const cond = func.newVReg(); // 1 (param)

    const entry = try func.newBlock();
    const header = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(header).addPredecessor(entry);
    try func.getBlock(header).addPredecessor(header);
    try func.getBlock(exit).addPredecessor(header);

    const loaded = func.newVReg();
    const used = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .op = .{ .br = header } });
    try func.getBlock(header).append(.{ .dest = used, .type = .i32, .op = .{ .add = .{ .lhs = loaded, .rhs = loaded } } });
    try func.getBlock(header).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = header, .else_block = exit } } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = null } });

    try verifyFunction(&func, 0, .load_forwarding, a);
}

test "verifier(#794): aliasing store after the load in the def block is flagged (straight path)" {
    // entry: v = load[p]; store[p] = val; br exit
    // exit:  use v; ret      <-- v is stale at the use (boundary-def, non-loop)
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 1, 0, 0);
    defer func.deinit();
    const base = func.newVReg(); // 0 (param)

    const entry = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(exit).addPredecessor(entry);

    const loaded = func.newVReg();
    const val = func.newVReg();
    const used = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .dest = val, .type = .i32, .op = .{ .iconst_32 = 9 } });
    try func.getBlock(entry).append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = val } } });
    try func.getBlock(entry).append(.{ .op = .{ .br = exit } });
    try func.getBlock(exit).append(.{ .dest = used, .type = .i32, .op = .{ .add = .{ .lhs = loaded, .rhs = loaded } } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = null } });

    try testing.expectError(error.LoadForwardingUnsound, verifyFunction(&func, 0, .load_forwarding, a));
    try testing.expect(std.mem.indexOf(u8, last_failure.detail, "store") != null);
}

test "verifier(#794): a load value carried through a loop-header phi across a store is checked" {
    // entry:  v = load[p]; val = const; br header
    // header: x = phi[entry: v, body: x2]; br_if cond, body, exit
    // body:   store[p] = val; x2 = x + 1; br header   <-- in-loop store
    // exit:   ret
    // `v` is used only as a phi operand; this locks in that phi-operand uses are
    // covered by Check 11 (the in-loop store is reachable on the header→header
    // cycle that does not re-execute the entry load).
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 2, 0, 0);
    defer func.deinit();
    const base = func.newVReg(); // 0 (param)
    const cond = func.newVReg(); // 1 (param)

    const entry = try func.newBlock();
    const header = try func.newBlock();
    const body = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(header).addPredecessor(entry);
    try func.getBlock(header).addPredecessor(body);
    try func.getBlock(body).addPredecessor(header);
    try func.getBlock(exit).addPredecessor(header);

    const loaded = func.newVReg();
    const val = func.newVReg();
    const phi_x = func.newVReg();
    const x2 = func.newVReg();
    const one = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .dest = val, .type = .i32, .op = .{ .iconst_32 = 3 } });
    try func.getBlock(entry).append(.{ .op = .{ .br = header } });

    const x_edges = try a.dupe(ir.Inst.PhiEdge, &.{ .{ .block = entry, .val = loaded }, .{ .block = body, .val = x2 } });
    try func.getBlock(header).append(.{ .dest = phi_x, .type = .i32, .op = .{ .phi = x_edges } });
    try func.getBlock(header).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = body, .else_block = exit } } });

    try func.getBlock(body).append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = val } } });
    try func.getBlock(body).append(.{ .dest = one, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(body).append(.{ .dest = x2, .type = .i32, .op = .{ .add = .{ .lhs = phi_x, .rhs = one } } });
    try func.getBlock(body).append(.{ .op = .{ .br = header } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = null } });

    try testing.expectError(error.LoadForwardingUnsound, verifyFunction(&func, 0, .load_forwarding, a));
    try testing.expect(std.mem.indexOf(u8, last_failure.detail, "store") != null);
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

test "verifier: block-ref callback context defaults bad block to entry" {
    const ctx = BlockRefCtx{ .nblocks = 1 };

    try testing.expect(ctx.ok);
    try testing.expectEqual(@as(ir.BlockId, 0), ctx.bad);
}

test "verifier: detects dangling block ref" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = 99 } });
    try testing.expectError(error.DanglingBlockRef, verifyFunction(&func, 0, .after_each_pass, a));
    try testing.expectEqual(error.DanglingBlockRef, last_failure.kind);
    try testing.expectEqual(@as(?ir.BlockId, b0), last_failure.block);
    try testing.expectEqual(@as(?u32, 0), last_failure.inst_index);
    try testing.expect(std.mem.indexOf(u8, last_failure.detail, "nonexistent block") != null);
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

test "verifier: cross-block non-dominance reports first bad operand" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();

    const entry = try func.newBlock();
    const left = try func.newBlock();
    const right = try func.newBlock();
    const cond = func.newVReg();
    const v_left = func.newVReg();
    const v_other = func.newVReg();
    const v_out = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = cond, .type = .i32, .op = .{ .iconst_32 = 0 } });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = left, .else_block = right } } });
    try func.getBlock(left).append(.{ .dest = v_left, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(left).append(.{ .dest = v_other, .type = .i32, .op = .{ .iconst_32 = 2 } });
    try func.getBlock(left).append(.{ .op = .{ .ret = v_left } });
    try func.getBlock(right).append(.{ .dest = v_out, .type = .i32, .op = .{ .add = .{ .lhs = v_left, .rhs = v_other } } });
    try func.getBlock(right).append(.{ .op = .{ .ret = v_out } });
    try func.getBlock(left).addPredecessor(entry);
    try func.getBlock(right).addPredecessor(entry);

    try testing.expectError(error.UnboundVRegUse, verifyFunction(&func, 0, .after_each_pass, a));
    try testing.expectEqual(@as(?ir.VReg, v_left), last_failure.vreg);
    try testing.expectEqual(@as(?ir.BlockId, right), last_failure.block);
    try testing.expectEqual(@as(?u32, 0), last_failure.inst_index);
}

test "verifier: same-block def-after-use reports first bad operand" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const v_future = func.newVReg();
    const v_other_bad = func.newVReg();
    const v_out = func.newVReg();
    try func.getBlock(b0).append(.{ .dest = v_out, .type = .i32, .op = .{ .add = .{ .lhs = v_future, .rhs = v_other_bad } } });
    try func.getBlock(b0).append(.{ .dest = v_future, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(b0).append(.{ .dest = v_other_bad, .type = .i32, .op = .{ .iconst_32 = 2 } });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v_out } });

    try testing.expectError(error.UnboundVRegUse, verifyFunction(&func, 0, .after_each_pass, a));
    try testing.expectEqual(@as(?ir.VReg, v_future), last_failure.vreg);
    try testing.expectEqual(@as(?ir.BlockId, b0), last_failure.block);
    try testing.expectEqual(@as(?u32, 0), last_failure.inst_index);
}

test "verifier: phi dominance reports first bad incoming operand" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();

    const entry = try func.newBlock();
    const left = try func.newBlock();
    const right = try func.newBlock();
    const merge = try func.newBlock();
    const cond = func.newVReg();
    const v_left = func.newVReg();
    const v_right = func.newVReg();
    const v_phi = func.newVReg();
    const edges = try a.alloc(ir.Inst.PhiEdge, 2);
    edges[0] = .{ .block = right, .val = v_left };
    edges[1] = .{ .block = left, .val = v_right };

    try func.getBlock(entry).append(.{ .dest = cond, .type = .i32, .op = .{ .iconst_32 = 0 } });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = left, .else_block = right } } });
    try func.getBlock(left).append(.{ .dest = v_left, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(left).append(.{ .op = .{ .br = merge } });
    try func.getBlock(right).append(.{ .dest = v_right, .type = .i32, .op = .{ .iconst_32 = 2 } });
    try func.getBlock(right).append(.{ .op = .{ .br = merge } });
    try func.getBlock(merge).append(.{ .dest = v_phi, .type = .i32, .op = .{ .phi = edges } });
    try func.getBlock(merge).append(.{ .op = .{ .ret = v_phi } });
    try func.getBlock(left).addPredecessor(entry);
    try func.getBlock(right).addPredecessor(entry);
    try func.getBlock(merge).addPredecessor(left);
    try func.getBlock(merge).addPredecessor(right);

    try testing.expectError(error.UnboundVRegUse, verifyFunction(&func, 0, .after_each_pass, a));
    try testing.expectEqual(@as(?ir.VReg, v_left), last_failure.vreg);
    try testing.expectEqual(@as(?ir.BlockId, merge), last_failure.block);
    try testing.expectEqual(@as(?u32, 0), last_failure.inst_index);
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

// ── Check 7 (operand-width sanity, paranoid only) ─────────────────────

test "verifier(paranoid): well-typed add i32 passes" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v0, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(b).append(.{ .dest = v1, .type = .i32, .op = .{ .iconst_32 = 2 } });
    try func.getBlock(b).append(.{ .dest = v2, .type = .i32, .op = .{ .add = .{ .lhs = v0, .rhs = v1 } } });
    try func.getBlock(b).append(.{ .op = .{ .ret = v2 } });
    try verifyFunction(&func, 0, .paranoid, a);
}

test "verifier(paranoid): detects scalar bin-op width mismatch" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v0, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(b).append(.{ .dest = v1, .type = .i64, .op = .{ .iconst_64 = 2 } });
    // i32 add reading an i64 operand.
    try func.getBlock(b).append(.{ .dest = v2, .type = .i32, .op = .{ .add = .{ .lhs = v0, .rhs = v1 } } });
    try func.getBlock(b).append(.{ .op = .{ .ret = v2 } });
    try testing.expectError(error.OperandTypeMismatch, verifyFunction(&func, 0, .paranoid, a));
    try testing.expectEqual(@as(?ir.VReg, v1), last_failure.vreg);
}

test "verifier(paranoid): detects load base wrong width" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const v_base = func.newVReg();
    const v_loaded = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v_base, .type = .i64, .op = .{ .iconst_64 = 0 } });
    try func.getBlock(b).append(.{
        .dest = v_loaded,
        .type = .i32,
        .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } },
    });
    try func.getBlock(b).append(.{ .op = .{ .ret = v_loaded } });
    try testing.expectError(error.OperandTypeMismatch, verifyFunction(&func, 0, .paranoid, a));
    try testing.expectEqual(@as(?ir.VReg, v_base), last_failure.vreg);
}

test "verifier(paranoid): detects simd swizzle indices not v128" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const v_vec = func.newVReg();
    const v_idx = func.newVReg();
    const v_out = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v_vec, .type = .v128, .op = .{ .v128_const = 0 } });
    try func.getBlock(b).append(.{ .dest = v_idx, .type = .i32, .op = .{ .iconst_32 = 0 } });
    try func.getBlock(b).append(.{
        .dest = v_out,
        .type = .v128,
        .op = .{ .i8x16_swizzle = .{ .vector = v_vec, .indices = v_idx } },
    });
    try func.getBlock(b).append(.{ .op = .{ .ret = v_out } });
    try testing.expectError(error.OperandTypeMismatch, verifyFunction(&func, 0, .paranoid, a));
    try testing.expectEqual(@as(?ir.VReg, v_idx), last_failure.vreg);
}

test "verifier(paranoid): detects f_neg on integer producer" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const v_i = func.newVReg();
    const v_out = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v_i, .type = .i32, .op = .{ .iconst_32 = 7 } });
    // f_neg with inst.type = .f32, reading an i32-typed producer.
    try func.getBlock(b).append(.{ .dest = v_out, .type = .f32, .op = .{ .f_neg = v_i } });
    try func.getBlock(b).append(.{ .op = .{ .ret = v_out } });
    try testing.expectError(error.OperandTypeMismatch, verifyFunction(&func, 0, .paranoid, a));
    try testing.expectEqual(@as(?ir.VReg, v_i), last_failure.vreg);
}

test "verifier(paranoid): after_each_pass mode skips operand-width check" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const v_i = func.newVReg();
    const v_out = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v_i, .type = .i32, .op = .{ .iconst_32 = 7 } });
    try func.getBlock(b).append(.{ .dest = v_out, .type = .f32, .op = .{ .f_neg = v_i } });
    try func.getBlock(b).append(.{ .op = .{ .ret = v_out } });
    // Same broken IR — paranoid trips, after_each_pass does not.
    try verifyFunction(&func, 0, .after_each_pass, a);
    try testing.expectError(error.OperandTypeMismatch, verifyFunction(&func, 0, .paranoid, a));
}

test "verifier(paranoid): parameter operand types respected via local_types" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 1, 1, 1);
    defer func.deinit();
    // Param 0 is vreg 0, typed i64 via local_types.
    const lt = try a.alloc(ir.IrType, 1);
    lt[0] = .i64;
    func.local_types = lt;
    _ = func.newVReg(); // reserve vreg 0 for the param
    const v_other = func.newVReg();
    const v_sum = func.newVReg();
    const b = try func.newBlock();
    try func.getBlock(b).append(.{ .dest = v_other, .type = .i32, .op = .{ .iconst_32 = 1 } });
    // i32 add reading the i64 param.
    try func.getBlock(b).append(.{ .dest = v_sum, .type = .i32, .op = .{ .add = .{ .lhs = 0, .rhs = v_other } } });
    try func.getBlock(b).append(.{ .op = .{ .ret = v_sum } });
    try testing.expectError(error.OperandTypeMismatch, verifyFunction(&func, 0, .paranoid, a));
    try testing.expectEqual(@as(?ir.VReg, 0), last_failure.vreg);
}

test "verifier(paranoid): parallel_copy per-pair type checked" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const v_src = func.newVReg();
    const v_dst = func.newVReg();
    const b = try func.newBlock();
    try func.getBlock(b).append(.{ .dest = v_src, .type = .i32, .op = .{ .iconst_32 = 1 } });
    const pairs = try a.alloc(ir.Inst.ParallelCopy, 1);
    // pair says ty=i64 but src is i32 — mismatch.
    pairs[0] = .{ .dst = v_dst, .src = v_src, .ty = .i64 };
    try func.getBlock(b).append(.{ .op = .{ .parallel_copy = pairs } });
    try func.getBlock(b).append(.{ .op = .{ .ret = null } });
    try testing.expectError(error.OperandTypeMismatch, verifyFunction(&func, 0, .paranoid, a));
    try testing.expectEqual(@as(?ir.VReg, v_src), last_failure.vreg);
}

// ── Check 8 / 9 / 10 tests (paranoid, #629) ─────────────────────────────

test "verifier(paranoid): simple loop passes loop-info check" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    // b0 -> b1 (header); b1 -> b1 (latch, self-loop); b1 -> b2 -> ret
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const cond = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .dest = cond, .type = .i32, .op = .{ .iconst_32 = 0 } });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b1).addPredecessor(b0);
    try func.getBlock(b1).addPredecessor(b1);
    try func.getBlock(b2).addPredecessor(b1);
    try verifyFunction(&func, 0, .paranoid, a);
}

test "verifier(paranoid): verifyLoopForest rejects header not dominating body" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    // Trivial straight-line: b0 -> b1 ret. Header #1 does NOT dominate b0.
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b1).addPredecessor(b0);

    var dom = try analysis.computeDominators(&func, a);
    defer dom.deinit();

    // Hand-rolled bad forest: claim there's a loop with header=b1 and
    // body={b0, b1}. b1 does not dominate b0 → check 8 must reject.
    const bad_blocks = try a.alloc(ir.BlockId, 2);
    defer a.free(bad_blocks);
    bad_blocks[0] = 0;
    bad_blocks[1] = 1;
    const bad_latches = try a.alloc(ir.BlockId, 0);
    defer a.free(bad_latches);
    const loops = [_]analysis.Loop{.{ .header = 1, .latches = bad_latches, .blocks = bad_blocks }};
    var hl = std.AutoHashMap(ir.BlockId, u32).init(a);
    defer hl.deinit();
    try hl.put(1, 0);
    const forest = analysis.LoopForest{
        .loops = @constCast(loops[0..]),
        .header_loop = hl,
        .allocator = a,
    };
    try testing.expectError(error.LoopInvariantBroken, verifyLoopForest(0, &dom, &forest));
    try testing.expectEqual(error.LoopInvariantBroken, last_failure.kind);
    try testing.expectEqual(@as(?ir.BlockId, b0), last_failure.block);
    try testing.expect(std.mem.indexOf(u8, last_failure.detail, "body block #0") != null);
}

test "verifier(paranoid): linear CFG passes dom-tree structure check" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b1).addPredecessor(b0);
    try verifyFunction(&func, 0, .paranoid, a);
}

test "verifier(paranoid): verifyDomTreeStructure rejects non-null idom on entry" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b1).addPredecessor(b0);
    var dom = try analysis.computeDominators(&func, a);
    defer dom.deinit();
    // Tamper: entry's idom must be self or null; force it to point at b1.
    dom.idom[0] = 1;
    try testing.expectError(error.DomTreeInconsistent, verifyDomTreeStructure(&func, 0, &dom));
}

test "verifier(paranoid): well-typed live ranges pass monotonicity" {
    const a = testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b).append(.{ .dest = v0, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(b).append(.{ .dest = v1, .type = .i32, .op = .{ .iconst_32 = 2 } });
    try func.getBlock(b).append(.{ .dest = v2, .type = .i32, .op = .{ .add = .{ .lhs = v0, .rhs = v1 } } });
    try func.getBlock(b).append(.{ .op = .{ .ret = v2 } });
    try verifyFunction(&func, 0, .paranoid, a);
}

test "verifier(paranoid): verifyLiveRangeMonotonicity rejects inverted range" {
    const ranges = [_]analysis.LiveRange{
        .{ .vreg = 7, .start = 10, .end = 3, .type = .i32 },
    };
    try testing.expectError(error.LiveRangeInverted, verifyLiveRangeMonotonicity(0, ranges[0..]));
    try testing.expectEqual(@as(?ir.VReg, 7), last_failure.vreg);
}
