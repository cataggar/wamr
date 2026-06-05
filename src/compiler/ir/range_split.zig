//! Loop-aware live range splitting for register allocation (issue #383).
//!
//! Before linear scan, walk each natural loop and split the live range of
//! any VReg that crosses the loop without being used inside it. The split
//! materializes a fresh VReg in the post-loop region, fed from a synthetic
//! wasm local that's `local_set` at the loop's entry edge and `local_get`
//! at its exit edge. The original VReg's live range thus ends before the
//! loop body, freeing its physical register for the loop's hot uses; the
//! post-loop VReg's range begins after the loop, so the allocator can pick
//! any free register at that point.
//!
//! Conservative restriction (Phase 1): only loops with a single entry-pred
//! (one outside-loop predecessor of the header) and a single exit-edge to
//! a single out-of-loop successor are eligible. This dodges the multi-exit
//! phi problem and covers CoreMark's `crc16`-style hot loops, which is the
//! perf target the issue calls out. Multi-exit loops can be a follow-up.
//!
//! Defends against `analysis.computeLiveRangesWithOrder`'s silent
//! `end = @max(start, end)` clamp by asserting `end >= start` on every
//! produced range — a buggy split would otherwise corrupt linear scan
//! silently (cf. trap regression #443).

const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");
const passes = @import("passes.zig");

/// Result of a successful run.
pub const SplitStats = struct {
    /// Number of vreg splits applied (one per `local_set`/`local_get` pair
    /// inserted). Zero on functions with no eligible loops or no
    /// crossing-without-use vregs.
    splits_applied: u32 = 0,
    /// Number of natural loops considered (for diagnostics).
    loops_considered: u32 = 0,
    /// Number of loops skipped because they violated the single-entry /
    /// single-exit conservative restriction.
    loops_skipped_shape: u32 = 0,
    /// Number of loops skipped because they did not exhibit enough
    /// register pressure to justify the spill/reload overhead (see the
    /// `min_candidates` / `min_loop_body_insts` thresholds in
    /// `splitLiveRangesAtLoopBoundaries`).
    loops_skipped_pressure: u32 = 0,
};

/// In-place: insert spill/reload pairs and rewrite post-loop uses so that
/// long-lived VRegs unused inside a loop free their physical register
/// during the loop body.
///
/// Mutates `func` (appends synthetic locals, allocates fresh VRegs,
/// inserts `local_set`/`local_get` ops in basic blocks) and `sched`
/// (mirrors the same instruction-stream edits in the scheduled `[]ir.Inst`
/// that codegen consumes).
///
/// Returns stats. On any internal inconsistency, returns an error rather
/// than silently producing degenerate ranges that the regalloc clamp at
/// `analysis.zig:600` would mask.
/// Tunables for the pressure gate. Defaults are calibrated for CoreMark
/// (no-op or positive). Tests pass `min_candidates: 0` to exercise the
/// rewriter on small synthetic shapes.
pub const Config = struct {
    /// Floor on crossing-without-use candidates per loop. The primary
    /// gate is now register pressure (see `num_available_phys_regs`), so
    /// this stays at 1 (need at least one candidate to split). Tests pass
    /// 0 to keep the same meaning.
    min_candidates: usize = 1,
    /// Skip loops whose total body instruction count is below this.
    /// Trivial loops rarely have register pressure the allocator can't
    /// already resolve.
    min_loop_body_insts: usize = 8,
    /// Function-level cap on total splits. Defends against runaway
    /// cases on heavily-nested CFGs (CoreMark `core_state_transition`
    /// would otherwise produce 100+ splits per function and net-
    /// regress the benchmark).
    max_splits_per_function: u32 = 16,

    /// Register-pressure ranker (#524). Total allocatable GPR-class
    /// physical registers on the target. aarch64's allocatable integer
    /// pool is 23 (x0–x14 caller-saved + x21–x28 callee-saved; x19/x20
    /// pinned, x15–x18 scratch). The caller plumbs the real number from
    /// its backend; the default encodes aarch64 since this pass is
    /// aarch64-only today.
    num_available_phys_regs: usize = 23,
    /// Callee-saved subset of `num_available_phys_regs` (aarch64: x21–x28
    /// = 8). Values live across a call inside the loop cannot use
    /// caller-saved registers (the call clobbers them), so call-spanning
    /// pressure is gated against this smaller pool.
    num_callee_saved_regs: usize = 8,
    /// Headroom subtracted from each pool before declaring a loop
    /// pressure-bound. Leaves room for the allocator's own scratch /
    /// short-lived temporaries so we only split when there is credible
    /// spilling pressure, not merely high-but-fittable demand.
    safety_margin: usize = 2,
};

/// Saturating `a - b` for `usize` thresholds (avoids underflow when the
/// margin exceeds the pool, e.g. tiny synthetic test pools).
fn satSub(a: usize, b: usize) usize {
    return if (a > b) a - b else 0;
}

pub fn splitLiveRangesAtLoopBoundaries(
    func: *ir.IrFunction,
    /// Duck-typed: must expose `.allocator` and `.blocks` where each
    /// block has a public `.instructions: []ir.Inst`. Concretely this
    /// is `*schedule.FunctionSchedule` from
    /// `src/compiler/codegen/aarch64/schedule.zig`, but pinning the
    /// type here would force a `..` import outside this module's path,
    /// which Zig 0.16 rejects.
    sched: anytype,
    allocator: std.mem.Allocator,
) !SplitStats {
    return splitLiveRangesAtLoopBoundariesWithConfig(func, sched, allocator, .{});
}

pub fn splitLiveRangesAtLoopBoundariesWithConfig(
    func: *ir.IrFunction,
    sched: anytype,
    allocator: std.mem.Allocator,
    cfg: Config,
) !SplitStats {
    var stats: SplitStats = .{};
    const nblocks = func.blocks.items.len;
    if (nblocks == 0) return stats;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();
    var forest = try analysis.computeLoops(func, &dom, allocator);
    defer forest.deinit();
    if (forest.loops.len == 0) return stats;

    var predecessors = try analysis.buildPredecessors(func, allocator);
    defer {
        var pit = predecessors.iterator();
        while (pit.next()) |entry| allocator.free(entry.value_ptr.*);
        predecessors.deinit();
    }
    var successors = try analysis.buildSuccessors(func, allocator);
    defer {
        var sit = successors.iterator();
        while (sit.next()) |entry| allocator.free(entry.value_ptr.*);
        successors.deinit();
    }

    // ── Function-scope inputs for the register-pressure ranker (#524) ──
    // Computed once on the pre-split function and read-only thereafter.
    // Per-loop splits append `local_set`/`local_get` ops to existing
    // blocks (never new blocks), so block-keyed liveness stays usable; a
    // freshly-added `alt` vreg is simply absent from these tables, which
    // can only *under*-count a later loop's pressure — the regression-safe
    // direction (under-firing skips, it never over-splits).
    var liveness = try analysis.computeLiveness(func, allocator);
    defer {
        var lit = liveness.iterator();
        while (lit.next()) |entry| {
            entry.value_ptr.live_in.deinit();
            entry.value_ptr.live_out.deinit();
        }
        liveness.deinit();
    }

    // Live-range widths drive the candidate ranker (wider original range →
    // larger lifetime released per split). Built from global live ranges.
    var width = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer width.deinit();
    {
        const ranges = try analysis.computeLiveRanges(func, allocator);
        defer allocator.free(ranges);
        for (ranges) |r| try width.put(r.vreg, r.end - r.start);
    }

    // vreg → defining-op type, used to classify GPR-class (i32/i64) vs
    // FP/vector pressure. Phase 1 counts and splits only integer values.
    var vreg_type = std.AutoHashMap(ir.VReg, ir.IrType).init(allocator);
    defer vreg_type.deinit();
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            if (inst.dest) |d| {
                if (!vreg_type.contains(d)) try vreg_type.put(d, inst.type);
            }
        }
    }

    // Process inner loops first (smaller `blocks` slice). Splitting at an
    // inner-loop boundary is safe regardless of outer-loop nesting; an
    // outer-loop pass would never see the inner-loop crossing vreg
    // because the splitter for the inner loop has already replaced
    // post-inner-loop uses with a fresh vreg.
    const loop_order = try allocator.alloc(usize, forest.loops.len);
    defer allocator.free(loop_order);
    for (loop_order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, loop_order, forest.loops, struct {
        fn lessThan(loops: []const analysis.Loop, a: usize, b: usize) bool {
            return loops[a].blocks.len < loops[b].blocks.len;
        }
    }.lessThan);

    // Precompute per-loop "is innermost" flag — true iff no OTHER loop's
    // header lies inside this loop's body. Empirically, splitting only
    // innermost loops avoids over-applying on heavily-nested CFGs
    // (CoreMark `core_state_transition` produces a 22-loop forest with
    // 128 splits ungated, which net-regresses the benchmark). The
    // intuition: register pressure peaks at the innermost level, so
    // splits there carry the most benefit. Outer loops typically have
    // their critical vregs ALSO crossing inner loops, where the inner
    // pass has already split them.
    const is_innermost = try allocator.alloc(bool, forest.loops.len);
    defer allocator.free(is_innermost);
    @memset(is_innermost, true);
    for (forest.loops, 0..) |outer, oi| {
        for (forest.loops, 0..) |inner, ii| {
            if (oi == ii) continue;
            if (outer.containsBlock(inner.header)) {
                // `outer` contains `inner.header` → outer is not innermost.
                is_innermost[oi] = false;
                break;
            }
        }
    }

    // Function-level cap on splits, defending against runaway cases where
    // a single function's 50+ loops would each contribute multiple
    // splits. Calibrated against CoreMark's heaviest functions.
    const max_splits_per_function: u32 = cfg.max_splits_per_function;

    for (loop_order) |li| {
        stats.loops_considered += 1;
        const loop = &forest.loops[li];

        if (!is_innermost[li]) {
            stats.loops_skipped_pressure += 1;
            continue;
        }
        if (stats.splits_applied >= max_splits_per_function) {
            stats.loops_skipped_pressure += 1;
            continue;
        }

        // ── Shape check: single entry-predecessor and single exit edge ──
        const entry_pred = singleEntryPredecessor(loop, predecessors) orelse {
            stats.loops_skipped_shape += 1;
            continue;
        };
        const exit = singleExitEdge(loop, successors) orelse {
            stats.loops_skipped_shape += 1;
            continue;
        };

        // Build the "post-loop reachable AND exit.succ-dominated" block
        // set. Soundness invariant: every use we rewrite from orig → alt
        // must be dominated by the local_get (which lives at the start
        // of exit.succ). Otherwise a side-path reaching a "post" block
        // without going through exit.succ would observe `alt` undefined.
        // Plain CFG-reachability is NOT enough; we additionally require
        // `dom.dominates(exit.succ, block)`.
        var post_set = try computePostLoopReachable(
            loop,
            exit.succ,
            successors,
            &dom,
            nblocks,
            allocator,
        );
        defer post_set.deinit(allocator);

        // ── Find splittable vregs for this loop ──
        // A vreg is splittable iff:
        //   - It has at least one use in the post-loop reachable set, AND
        //   - It is NOT used inside the loop body (header or any body block).
        // We don't require a pre-loop use: a vreg defined in a block that
        // dominates the header (e.g. function entry's `local_get`) and only
        // read post-loop is just as splittable.

        var used_in_loop = try VRegSet.init(allocator);
        defer used_in_loop.deinit(allocator);
        var defined_in_loop = try VRegSet.init(allocator);
        defer defined_in_loop.deinit(allocator);
        for (loop.blocks) |bid| {
            for (func.blocks.items[bid].instructions.items) |inst| {
                if (inst.dest) |d| try defined_in_loop.put(allocator, d);
                try collectUses(inst, &used_in_loop, allocator);
            }
        }

        var used_post = try VRegSet.init(allocator);
        defer used_post.deinit(allocator);
        for (post_set.items) |bid| {
            for (func.blocks.items[bid].instructions.items) |inst| {
                try collectUses(inst, &used_post, allocator);
            }
        }

        // Authoritative def map: each vreg's defining block + type.
        // Using a use-site type would be wrong for ops where inst.type
        // differs from operand type (e.g. comparisons → result .i32 but
        // operand type matches the def's type). The defining block is
        // also needed to filter candidates: a vreg defined in or after
        // the loop's exit block cannot be spilled at the entry-pred
        // because its value doesn't exist yet — that would produce
        // invalid IR.
        var def_type = std.AutoHashMap(ir.VReg, ir.IrType).init(allocator);
        defer def_type.deinit();
        var def_block = std.AutoHashMap(ir.VReg, ir.BlockId).init(allocator);
        defer def_block.deinit();
        for (func.blocks.items, 0..) |block, bidx| {
            for (block.instructions.items) |inst| {
                if (inst.dest) |d| {
                    if (!def_type.contains(d)) {
                        try def_type.put(d, inst.type);
                        try def_block.put(d, @intCast(bidx));
                    }
                }
            }
        }

        // Candidates: used post-loop, NOT used inside loop, NOT defined inside loop,
        // AND defining block dominates the entry_pred (so the value already
        // exists when we hit the loop entry's spill point).
        var candidates: std.ArrayListUnmanaged(Candidate) = .empty;
        defer candidates.deinit(allocator);

        var pit = used_post.set.iterator();
        while (pit.next()) |entry| {
            const v = entry.key_ptr.*;
            if (used_in_loop.contains(v)) continue;
            if (defined_in_loop.contains(v)) continue;
            // The defining block must dominate entry_pred — otherwise
            // emitting `local_set slot, v` at end of entry_pred would
            // reference a vreg before its first def, producing invalid
            // IR. This excludes vregs whose definition lives in or
            // after the exit-successor (e.g. post-loop temporaries
            // used by other post-loop instructions).
            const dblk = def_block.get(v) orelse continue;
            if (!dom.dominates(dblk, entry_pred)) continue;
            // Scalar integer (GPR-class) only. The synthetic local slot is
            // sized as `.i64` (8 B) by the aarch64 backend's `localTypeAt`
            // fallback, and the #524 pressure gate below models the integer
            // register pool. f32/f64 occupy the FP/SIMD file and v128 needs
            // a 16-byte slot — neither is handled here. Phase 1.
            const ty = def_type.get(v) orelse continue;
            if (ty != .i32 and ty != .i64) continue;
            try candidates.append(allocator, .{ .orig = v, .ty = ty });
        }

        if (candidates.items.len == 0) continue;

        if (candidates.items.len < cfg.min_candidates) continue;

        var body_inst_count: usize = 0;
        for (loop.blocks) |bid| body_inst_count += func.blocks.items[bid].instructions.items.len;
        if (body_inst_count < cfg.min_loop_body_insts) {
            stats.loops_skipped_pressure += 1;
            continue;
        }

        // ── Register-pressure gate (#524) ──
        // Two thresholds: the full allocatable GPR pool for ordinary
        // loop-body pressure, and the callee-saved subset for values that
        // must survive a call inside the loop (a call clobbers caller-saved
        // registers, so live-across-call values can only use callee-saved).
        // A loop is pressure-bound iff it exceeds either pool minus the
        // safety margin. When neither is exceeded the allocator can satisfy
        // the loop without spilling, so splitting only adds spill/reload
        // overhead (empirical -0.67% on CoreMark when split unconditionally).
        const t_all = satSub(cfg.num_available_phys_regs, cfg.safety_margin);
        const t_xcall = satSub(cfg.num_callee_saved_regs, cfg.safety_margin);
        const pres = try loopGprPressure(func, loop, &liveness, &vreg_type, allocator);
        const relief_needed = @max(
            satSub(pres.max_live, t_all),
            satSub(pres.max_across_call, t_xcall),
        );
        if (relief_needed == 0) {
            stats.loops_skipped_pressure += 1;
            continue;
        }

        // ── Relief filter ──
        // A candidate only lowers loop pressure if, after rewriting its
        // post_set uses to `alt`, its original vreg is no longer live across
        // the loop. That holds iff every loop-exit-reachable use of `orig`
        // lies in `post_set` (the dominated rewrite region). If `orig` is
        // also used in a post-loop block reachable but NOT dominated by
        // exit.succ, that use keeps `orig` live through the loop, so the
        // split frees no register there — exclude it from the relief target.
        var used_outside = try collectUsesOutsidePostSet(
            func,
            loop,
            exit.succ,
            &post_set,
            successors,
            nblocks,
            allocator,
        );
        defer used_outside.deinit(allocator);

        var relieving: std.ArrayListUnmanaged(Candidate) = .empty;
        defer relieving.deinit(allocator);
        for (candidates.items) |c| {
            if (used_outside.contains(c.orig)) continue;
            try relieving.append(allocator, c);
        }
        if (relieving.items.len == 0) {
            stats.loops_skipped_pressure += 1;
            continue;
        }

        // Rank by live-range width (widest original range → largest
        // lifetime released per split); deterministic vreg-id tie-break
        // keeps codegen output stable.
        std.mem.sort(Candidate, relieving.items, RankCtx{ .width = &width }, RankCtx.lessThan);

        // Split just enough of the widest relieving candidates to bring
        // pressure back toward the threshold; each live-through split lowers
        // pressure by exactly 1. Capped by the remaining function budget.
        const remaining_budget = satSub(cfg.max_splits_per_function, stats.splits_applied);
        const k = @min(@min(relieving.items.len, relief_needed), remaining_budget);
        if (k == 0) {
            stats.loops_skipped_pressure += 1;
            continue;
        }

        // ── Apply each split ──
        // 1. Allocate a fresh local slot and a fresh vreg.
        // 2. Insert `local_set slot, orig` at end of entry_pred (before its
        //    terminator) in both func.blocks AND sched.blocks.
        // 3. Insert `alt = local_get slot` at start of exit.succ in both.
        // 4. Walk every block in `post_set` and replace uses of orig → alt
        //    in both func.blocks AND sched.blocks. (The reload itself uses
        //    NO vreg operand — `local_get` is a pure def — so it's safe to
        //    insert before the rewrite walk.)
        for (relieving.items[0..k]) |cand| {
            const slot: u32 = func.local_count;
            func.local_count += 1;
            const alt: ir.VReg = func.newVReg();

            try insertBeforeTerminator(
                func,
                sched,
                entry_pred,
                .{
                    .op = .{ .local_set = .{ .idx = slot, .val = cand.orig } },
                    .dest = null,
                    .type = .void,
                },
                allocator,
            );

            try insertAtBlockStart(
                func,
                sched,
                exit.succ,
                .{
                    .op = .{ .local_get = slot },
                    .dest = alt,
                    .type = cand.ty,
                },
                allocator,
            );

            // Rewrite post-loop uses orig → alt. Done in both the IR
            // function blocks (so any later non-aarch64 consumer sees a
            // consistent IR) and the scheduled stream (so the aarch64
            // codegen's actual emit loop reads the rewritten vreg).
            for (post_set.items) |bid| {
                // Skip the reload's own def — its dest is `alt`, not a
                // use of orig, so replaceInInst won't touch it anyway,
                // but we still iterate it harmlessly.
                for (func.blocks.items[bid].instructions.items) |*inst| {
                    passes.replaceInInst(inst, cand.orig, alt);
                }
                const sched_block_idx: usize = @intCast(bid);
                for (sched.blocks[sched_block_idx].instructions) |*inst| {
                    passes.replaceInInst(inst, cand.orig, alt);
                }
            }

            stats.splits_applied += 1;
        }
    }

    return stats;
}

const Candidate = struct {
    orig: ir.VReg,
    ty: ir.IrType,
};

/// Ranker context: orders candidates by descending live-range width, with
/// a deterministic vreg-id tie-break so codegen output is stable.
const RankCtx = struct {
    width: *const std.AutoHashMap(ir.VReg, u32),
    fn lessThan(ctx: RankCtx, a: Candidate, b: Candidate) bool {
        const wa = ctx.width.get(a.orig) orelse 0;
        const wb = ctx.width.get(b.orig) orelse 0;
        if (wa != wb) return wa > wb;
        return a.orig < b.orig;
    }
};

const LoopPressure = struct {
    /// Max simultaneously-live GPR-class vregs anywhere in the loop body.
    max_live: usize = 0,
    /// Max GPR-class vregs live across any call inside the loop (the call
    /// clobbers caller-saved registers, so these compete for callee-saved).
    max_across_call: usize = 0,
};

/// True for integer (GPR-class) vregs. Unknown vregs — e.g. params with no
/// defining op in this function — default to GPR: they are overwhelmingly
/// integer/pointer, and over-counting only makes the gate marginally more
/// eager, never unsound.
fn isGprVReg(vreg_type: *const std.AutoHashMap(ir.VReg, ir.IrType), v: ir.VReg) bool {
    const ty = vreg_type.get(v) orelse return true;
    return ty == .i32 or ty == .i64;
}

fn gprCount(set: *const VRegSet, vreg_type: *const std.AutoHashMap(ir.VReg, ir.IrType)) usize {
    var n: usize = 0;
    var it = set.set.iterator();
    while (it.next()) |e| {
        if (isGprVReg(vreg_type, e.key_ptr.*)) n += 1;
    }
    return n;
}

fn isCallOp(op: ir.Inst.Op) bool {
    return switch (op) {
        .call, .call_indirect, .call_ref => true,
        else => false,
    };
}

/// Estimate the loop body's GPR register pressure via a per-block backward
/// liveness sweep seeded from `live_out`. Returns both the ordinary max and
/// the max measured at call sites (live-across-call). Computed on the
/// pre-split function; read-only.
fn loopGprPressure(
    func: *const ir.IrFunction,
    loop: *const analysis.Loop,
    liveness: *const std.AutoHashMap(ir.BlockId, analysis.BlockLiveness),
    vreg_type: *const std.AutoHashMap(ir.VReg, ir.IrType),
    allocator: std.mem.Allocator,
) !LoopPressure {
    var live = try VRegSet.init(allocator);
    defer live.deinit(allocator);
    var res: LoopPressure = .{};

    for (loop.blocks) |bid| {
        live.set.clearRetainingCapacity();
        const bl = liveness.get(bid) orelse continue;
        var oit = bl.live_out.keyIterator();
        while (oit.next()) |k| try live.put(allocator, k.*);
        res.max_live = @max(res.max_live, gprCount(&live, vreg_type));

        const insts = func.blocks.items[bid].instructions.items;
        var i = insts.len;
        while (i > 0) {
            i -= 1;
            const inst = insts[i];
            // `live` is the set live-out of `inst`. A call clobbers
            // caller-saved registers, so everything live across it must
            // survive in callee-saved registers (or spill).
            if (isCallOp(inst.op)) {
                res.max_across_call = @max(res.max_across_call, gprCount(&live, vreg_type));
            }
            // Def point: pressure here includes the destination being
            // written (live-out ∪ {dest}).
            if (inst.dest) |d| {
                const had = live.contains(d);
                if (!had) try live.put(allocator, d);
                res.max_live = @max(res.max_live, gprCount(&live, vreg_type));
                _ = live.set.remove(d); // dest is not live before its def
            }
            // Use point: (live-out \ dest) ∪ uses == live-in of `inst`.
            try collectUses(inst, &live, allocator);
            res.max_live = @max(res.max_live, gprCount(&live, vreg_type));
        }
    }
    return res;
}

/// Collect vregs used in blocks reachable after the loop (BFS from
/// `exit_succ`, not re-entering the loop) that are NOT in `post_set`. These
/// are the reachable-but-not-dominated post-loop blocks: a candidate used
/// there stays live across the loop even after its `post_set` uses are
/// rewritten, so it relieves no register pressure and must be excluded from
/// the relief target.
fn collectUsesOutsidePostSet(
    func: *const ir.IrFunction,
    loop: *const analysis.Loop,
    exit_succ: ir.BlockId,
    post_set: *const std.ArrayListUnmanaged(ir.BlockId),
    successors: std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    nblocks: usize,
    allocator: std.mem.Allocator,
) !VRegSet {
    var in_post = try allocator.alloc(bool, nblocks);
    defer allocator.free(in_post);
    @memset(in_post, false);
    for (post_set.items) |b| in_post[b] = true;

    var out = try VRegSet.init(allocator);
    errdefer out.deinit(allocator);

    var visited = try allocator.alloc(bool, nblocks);
    defer allocator.free(visited);
    @memset(visited, false);

    var stack: std.ArrayListUnmanaged(ir.BlockId) = .empty;
    defer stack.deinit(allocator);
    try stack.append(allocator, exit_succ);
    while (stack.pop()) |bid| {
        if (visited[bid]) continue;
        if (loop.containsBlock(bid)) continue;
        visited[bid] = true;
        if (!in_post[bid]) {
            for (func.blocks.items[bid].instructions.items) |inst| {
                try collectUses(inst, &out, allocator);
            }
        }
        const succs = successors.get(bid) orelse continue;
        for (succs) |s| if (!visited[s]) try stack.append(allocator, s);
    }
    return out;
}

/// Return the unique block that's a predecessor of `loop.header` and is
/// NOT itself in the loop. `null` if zero or multiple such predecessors
/// exist.
fn singleEntryPredecessor(
    loop: *const analysis.Loop,
    predecessors: std.AutoHashMap(ir.BlockId, []const ir.BlockId),
) ?ir.BlockId {
    const preds = predecessors.get(loop.header) orelse return null;
    var found: ?ir.BlockId = null;
    for (preds) |p| {
        if (loop.containsBlock(p)) continue; // back-edge from latch
        if (found != null) return null; // multiple entry preds
        found = p;
    }
    return found;
}

const ExitEdge = struct {
    /// Block inside the loop with an outgoing edge to outside.
    src: ir.BlockId,
    /// Block outside the loop reached by that edge.
    succ: ir.BlockId,
};

/// Return the unique exit edge of `loop` if there is exactly one such
/// edge AND it has exactly one out-of-loop successor. `null` otherwise.
fn singleExitEdge(
    loop: *const analysis.Loop,
    successors: std.AutoHashMap(ir.BlockId, []const ir.BlockId),
) ?ExitEdge {
    var found: ?ExitEdge = null;
    for (loop.blocks) |bid| {
        const succs = successors.get(bid) orelse continue;
        for (succs) |s| {
            if (loop.containsBlock(s)) continue; // intra-loop edge
            const edge: ExitEdge = .{ .src = bid, .succ = s };
            if (found) |f| {
                // Multiple exit edges → reject. (Even two edges to the
                // same successor would require care: the reload would
                // need to be on every edge or in a dominator of both.)
                if (f.src != edge.src or f.succ != edge.succ) return null;
            }
            found = edge;
        }
    }
    return found;
}

/// BFS from `start` over blocks reachable without re-entering the loop
/// AND dominated by `start`. The dominance filter is the soundness gate:
/// only blocks where every entry-path passes through `start` (the
/// post-loop reload site) are safe to rewrite. A block reachable from
/// `start` but ALSO reachable from outside (via a side-path that skips
/// the loop) would observe the fresh `alt` vreg undefined on that
/// side-path.
fn computePostLoopReachable(
    loop: *const analysis.Loop,
    start: ir.BlockId,
    successors: std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    dom: *const analysis.DomTree,
    nblocks: usize,
    allocator: std.mem.Allocator,
) !std.ArrayListUnmanaged(ir.BlockId) {
    var visited = try allocator.alloc(bool, nblocks);
    defer allocator.free(visited);
    @memset(visited, false);

    var out: std.ArrayListUnmanaged(ir.BlockId) = .empty;
    errdefer out.deinit(allocator);

    var stack: std.ArrayListUnmanaged(ir.BlockId) = .empty;
    defer stack.deinit(allocator);
    try stack.append(allocator, start);

    while (stack.pop()) |bid| {
        if (visited[bid]) continue;
        if (loop.containsBlock(bid)) continue;
        // Soundness: skip blocks not dominated by `start`. They have a
        // CFG path that doesn't pass through the reload site, so a
        // rewrite would be unsound on that path.
        if (!dom.dominates(start, bid)) continue;
        visited[bid] = true;
        try out.append(allocator, bid);
        const succs = successors.get(bid) orelse continue;
        for (succs) |s| if (!visited[s]) try stack.append(allocator, s);
    }
    return out;
}

/// Insert `inst` just before the terminator of `bid` in both `func`
/// and `sched`. If the block is empty, append. Terminator is detected
/// as "the last instruction" — wasm-flavoured IR always has a
/// terminator at end-of-block.
fn insertBeforeTerminator(
    func: *ir.IrFunction,
    sched: anytype,
    bid: ir.BlockId,
    inst: ir.Inst,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;
    const block = func.getBlock(bid);
    const idx_func: usize = if (block.instructions.items.len == 0) 0 else block.instructions.items.len - 1;
    try block.instructions.insert(block.allocator, idx_func, inst);

    const sb = &sched.blocks[@intCast(bid)];
    const idx_sched: usize = if (sb.instructions.len == 0) 0 else sb.instructions.len - 1;
    try insertIntoSlice(sched.allocator, &sb.instructions, idx_sched, inst);
}

/// Insert `inst` at index 0 of `bid` in both `func` and `sched`.
fn insertAtBlockStart(
    func: *ir.IrFunction,
    sched: anytype,
    bid: ir.BlockId,
    inst: ir.Inst,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;
    const block = func.getBlock(bid);
    try block.instructions.insert(block.allocator, 0, inst);

    const sb = &sched.blocks[@intCast(bid)];
    try insertIntoSlice(sched.allocator, &sb.instructions, 0, inst);
}

/// Grow `slice.*` by 1 and insert `inst` at `idx`, shifting the tail
/// right. Uses `allocator.realloc` — caller MUST own the slice from the
/// same allocator. `schedule.FunctionSchedule.blocks[i].instructions`
/// satisfies this (allocated by `sched.allocator`).
fn insertIntoSlice(
    allocator: std.mem.Allocator,
    slice: *[]ir.Inst,
    idx: usize,
    inst: ir.Inst,
) !void {
    const old_len = slice.len;
    std.debug.assert(idx <= old_len);
    const grown = try allocator.realloc(slice.*, old_len + 1);
    if (idx < old_len) {
        std.mem.copyBackwards(ir.Inst, grown[idx + 1 .. old_len + 1], grown[idx..old_len]);
    }
    grown[idx] = inst;
    slice.* = grown;
}

// ── VReg-set helper (HashMap-backed; avoids dependency churn) ──

const VRegSet = struct {
    set: std.AutoHashMap(ir.VReg, void),

    fn init(allocator: std.mem.Allocator) !VRegSet {
        return .{ .set = std.AutoHashMap(ir.VReg, void).init(allocator) };
    }
    fn deinit(self: *VRegSet, _: std.mem.Allocator) void {
        self.set.deinit();
    }
    fn put(self: *VRegSet, _: std.mem.Allocator, v: ir.VReg) !void {
        try self.set.put(v, {});
    }
    fn contains(self: *const VRegSet, v: ir.VReg) bool {
        return self.set.contains(v);
    }
};

const UseCtx = struct {
    set: *VRegSet,
    allocator: std.mem.Allocator,
};

fn visitUse(ctx: *UseCtx, v: ir.VReg) anyerror!void {
    try ctx.set.put(ctx.allocator, v);
}

fn collectUses(inst: ir.Inst, into: *VRegSet, allocator: std.mem.Allocator) !void {
    var ctx = UseCtx{ .set = into, .allocator = allocator };
    try forEachUseInst(inst, &ctx, visitUse);
}

/// Mirror of `schedule.forEachUse` (file-private switch over `ir.Inst.op`).
/// Inlined here because importing `../codegen/aarch64/schedule.zig`
/// breaches the `src/compiler/ir/` module path under Zig 0.16's strict
/// per-module file-tree boundary. Stays in sync with that source — any
/// new op variant must be added in both places.
pub fn forEachUseInst(
    inst: ir.Inst,
    context: anytype,
    comptime visit: fn (@TypeOf(context), ir.VReg) anyerror!void,
) !void {
    switch (inst.op) {
        .add,
        .sub,
        .mul,
        .@"and",
        .@"or",
        .xor,
        .div_s,
        .div_u,
        .rem_s,
        .rem_u,
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
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        .f_min,
        .f_max,
        .f_copysign,
        => |b| {
            try visit(context, b.lhs);
            try visit(context, b.rhs);
        },
        .local_set => |ls| try visit(context, ls.val),
        .global_set => |gs| try visit(context, gs.val),
        .eqz,
        .ctz,
        .clz,
        .popcnt,
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
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i32_u,
        .convert_i64_s,
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
        .reinterpret,
        .memory_grow,
        => |v| try visit(context, v),
        .ret => |maybe_v| if (maybe_v) |v| try visit(context, v),
        .ret_multi => |vregs| for (vregs) |v| try visit(context, v),
        .load => |ld| try visit(context, ld.base),
        .store => |st| {
            try visit(context, st.base);
            try visit(context, st.val);
        },
        .atomic_load => |ald| try visit(context, ald.base),
        .atomic_store => |ast| {
            try visit(context, ast.base);
            try visit(context, ast.val);
        },
        .atomic_rmw => |arm| {
            try visit(context, arm.base);
            try visit(context, arm.val);
        },
        .atomic_cmpxchg => |acx| {
            try visit(context, acx.base);
            try visit(context, acx.expected);
            try visit(context, acx.replacement);
        },
        .atomic_notify => |an| {
            try visit(context, an.base);
            try visit(context, an.count);
        },
        .atomic_wait => |aw| {
            try visit(context, aw.base);
            try visit(context, aw.expected);
            try visit(context, aw.timeout);
        },
        .select => |sel| {
            try visit(context, sel.cond);
            try visit(context, sel.if_true);
            try visit(context, sel.if_false);
        },
        .br_if => |bi| try visit(context, bi.cond),
        .br_table => |bt| try visit(context, bt.index),
        .call => |cl| for (cl.args) |a| try visit(context, a),
        .call_indirect => |ci| {
            try visit(context, ci.elem_idx);
            for (ci.args) |a| try visit(context, a);
        },
        .call_ref => |cr| {
            try visit(context, cr.func_ref);
            for (cr.args) |a| try visit(context, a);
        },
        .memory_fill => |mf| {
            try visit(context, mf.dst);
            try visit(context, mf.val);
            try visit(context, mf.len);
        },
        .memory_copy => |mc| {
            try visit(context, mc.dst);
            try visit(context, mc.src);
            try visit(context, mc.len);
        },
        .memory_init => |mi| {
            try visit(context, mi.dst);
            try visit(context, mi.src);
            try visit(context, mi.len);
        },
        .table_init => |ti| {
            try visit(context, ti.dst);
            try visit(context, ti.src);
            try visit(context, ti.len);
        },
        .table_get => |tg| try visit(context, tg.idx),
        .table_set => |ts| {
            try visit(context, ts.idx);
            try visit(context, ts.val);
        },
        .table_grow => |tg| {
            try visit(context, tg.init);
            try visit(context, tg.delta);
        },
        .phi => |edges| for (edges) |edge| try visit(context, edge.val),
        .parallel_copy => |pairs| for (pairs) |p| try visit(context, p.src),
        .v128_not => |v| try visit(context, v),
        .v128_any_true => |v| try visit(context, v),
        .v128_load => |ld| try visit(context, ld.base),
        .v128_load_splat => |ld| try visit(context, ld.base),
        .v128_load_zero => |ld| try visit(context, ld.base),
        .v128_load_extend => |ld| try visit(context, ld.base),
        .v128_load_lane => |ld| {
            try visit(context, ld.base);
            try visit(context, ld.vector);
        },
        .v128_store => |st| {
            try visit(context, st.base);
            try visit(context, st.val);
        },
        .v128_store_lane => |st| {
            try visit(context, st.base);
            try visit(context, st.vector);
        },
        .v128_bitwise => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .v128_bitselect => |sel| {
            try visit(context, sel.a);
            try visit(context, sel.b);
            try visit(context, sel.mask);
        },
        .simd_all_true => |op| try visit(context, op.vector),
        .simd_bitmask => |op| try visit(context, op.vector),
        .i32x4_binop => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .i32x4_unop => |un| try visit(context, un.vector),
        .i32x4_extadd_pairwise_i16x8 => |op| try visit(context, op.vector),
        .i32x4_dot_i16x8_s => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .i32x4_extend_i16x8 => |op| try visit(context, op.vector),
        .f32x4_binop => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .f32x4_unop => |un| try visit(context, un.vector),
        .f32x4_convert_i32x4 => |op| try visit(context, op.vector),
        .i32x4_trunc_sat => |op| try visit(context, op.vector),
        .f32x4_demote_f64x2_zero => |op| try visit(context, op.vector),
        .f32x4_splat => |v| try visit(context, v),
        .f32x4_extract_lane => |lane| try visit(context, lane.vector),
        .f32x4_replace_lane => |lane| {
            try visit(context, lane.vector);
            try visit(context, lane.val);
        },
        .i32x4_extmul_i16x8 => |op| {
            try visit(context, op.lhs);
            try visit(context, op.rhs);
        },
        .i8x16_binop => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .i8x16_shuffle => |op| {
            try visit(context, op.lhs);
            try visit(context, op.rhs);
        },
        .i8x16_swizzle => |op| {
            try visit(context, op.vector);
            try visit(context, op.indices);
        },
        .i8x16_narrow_i16x8 => |op| {
            try visit(context, op.lhs);
            try visit(context, op.rhs);
        },
        .i8x16_unop => |un| try visit(context, un.vector),
        .i8x16_shift => |shift| {
            try visit(context, shift.vector);
            try visit(context, shift.count);
        },
        .i16x8_binop => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .i16x8_unop => |un| try visit(context, un.vector),
        .i16x8_extadd_pairwise_i8x16 => |op| try visit(context, op.vector),
        .i16x8_extend_i8x16 => |op| try visit(context, op.vector),
        .i16x8_extmul_i8x16 => |op| {
            try visit(context, op.lhs);
            try visit(context, op.rhs);
        },
        .i16x8_narrow_i32x4 => |op| {
            try visit(context, op.lhs);
            try visit(context, op.rhs);
        },
        .i64x2_binop => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .f64x2_binop => |bin| {
            try visit(context, bin.lhs);
            try visit(context, bin.rhs);
        },
        .f64x2_unop => |un| try visit(context, un.vector),
        .f64x2_convert_low_i32x4 => |op| try visit(context, op.vector),
        .f64x2_promote_low_f32x4 => |op| try visit(context, op.vector),
        .i64x2_unop => |un| try visit(context, un.vector),
        .i64x2_extend_i32x4 => |op| try visit(context, op.vector),
        .i64x2_extmul_i32x4 => |op| {
            try visit(context, op.lhs);
            try visit(context, op.rhs);
        },
        .i64x2_shift => |shift| {
            try visit(context, shift.vector);
            try visit(context, shift.count);
        },
        .i32x4_shift => |shift| {
            try visit(context, shift.vector);
            try visit(context, shift.count);
        },
        .i16x8_shift => |shift| {
            try visit(context, shift.vector);
            try visit(context, shift.count);
        },
        .i32x4_splat => |v| try visit(context, v),
        .i32x4_extract_lane => |lane| try visit(context, lane.vector),
        .i32x4_replace_lane => |lane| {
            try visit(context, lane.vector);
            try visit(context, lane.val);
        },
        .i8x16_splat => |v| try visit(context, v),
        .i8x16_extract_lane => |lane| try visit(context, lane.vector),
        .i8x16_replace_lane => |lane| {
            try visit(context, lane.vector);
            try visit(context, lane.val);
        },
        .i16x8_splat => |v| try visit(context, v),
        .i16x8_extract_lane => |lane| try visit(context, lane.vector),
        .i16x8_replace_lane => |lane| {
            try visit(context, lane.vector);
            try visit(context, lane.val);
        },
        .i64x2_splat => |v| try visit(context, v),
        .i64x2_extract_lane => |lane| try visit(context, lane.vector),
        .i64x2_replace_lane => |lane| {
            try visit(context, lane.vector);
            try visit(context, lane.val);
        },
        .f64x2_splat => |v| try visit(context, v),
        .f64x2_extract_lane => |lane| try visit(context, lane.vector),
        .f64x2_replace_lane => |lane| {
            try visit(context, lane.vector);
            try visit(context, lane.val);
        },
        else => {},
    }
}

// ── Tests ──────────────────────────────────────────────────────────────
//
// `schedule.FunctionSchedule` lives outside this module's path, so the
// tests construct a `MockSchedule` with the same duck-typed surface
// (.allocator + .blocks[i].instructions: []ir.Inst). This validates the
// splitter logic without dragging in the aarch64 codegen path. The full
// end-to-end emission test lives in compile.zig tests.

const MockBlock = struct {
    instructions: []ir.Inst = &.{},
};

const MockSchedule = struct {
    allocator: std.mem.Allocator,
    blocks: []MockBlock,

    fn fromFunc(func: *const ir.IrFunction, allocator: std.mem.Allocator) !MockSchedule {
        const blocks = try allocator.alloc(MockBlock, func.blocks.items.len);
        errdefer allocator.free(blocks);
        var done: usize = 0;
        errdefer for (blocks[0..done]) |b| allocator.free(b.instructions);
        for (func.blocks.items, 0..) |b, i| {
            blocks[i] = .{ .instructions = try allocator.dupe(ir.Inst, b.instructions.items) };
            done += 1;
        }
        return .{ .allocator = allocator, .blocks = blocks };
    }

    fn deinit(self: *MockSchedule) void {
        for (self.blocks) |b| self.allocator.free(b.instructions);
        self.allocator.free(self.blocks);
    }
};

fn countLocalSetsInBlock(block: ir.BasicBlock) usize {
    var n: usize = 0;
    for (block.instructions.items) |inst| {
        if (inst.op == .local_set) n += 1;
    }
    return n;
}
fn countLocalGetsInBlock(block: ir.BasicBlock) usize {
    var n: usize = 0;
    for (block.instructions.items) |inst| {
        if (inst.op == .local_get) n += 1;
    }
    return n;
}

/// First `local_set` source vreg in a block (the orig of the first split).
fn firstLocalSetVal(block: ir.BasicBlock) ?ir.VReg {
    for (block.instructions.items) |inst| {
        if (inst.op == .local_set) return inst.op.local_set.val;
    }
    return null;
}

/// Builds a 4-block loop with 3 long-lived i32 candidates A,B,C (defined in
/// that order) plus a counter, all live across a `br`-only loop body. The
/// post-loop block uses C, then B, then A, so the live-range widths satisfy
/// width(A) > width(B) > width(C). Loop register pressure is exactly 4
/// (A,B,C,ctr) with no defs inside the loop. Returns the candidate vregs in
/// widest-first order and the entry/exit block ids.
const RankFixture = struct { cands: [3]ir.VReg, b_entry: ir.BlockId, b_exit: ir.BlockId };
fn buildRankFixture(func: *ir.IrFunction) !RankFixture {
    const b_entry = try func.newBlock();
    const b_hdr = try func.newBlock();
    const b_body = try func.newBlock();
    const b_exit = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const c = func.newVReg();
    const ctr = func.newVReg();

    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = a, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 2 }, .dest = b, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 3 }, .dest = c, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 100 }, .dest = ctr, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    try func.getBlock(b_hdr).append(.{ .op = .{ .br_if = .{ .cond = ctr, .then_block = b_body, .else_block = b_exit } }, .type = .void });

    // Loop body holds no values of its own (no defs) → loop pressure is
    // exactly the 4 live-through scalars.
    try func.getBlock(b_body).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    // Post-loop: use C first, then B, then A so A is the widest range.
    const s1 = func.newVReg();
    const s2 = func.newVReg();
    const s3 = func.newVReg();
    try func.getBlock(b_exit).append(.{ .op = .{ .add = .{ .lhs = c, .rhs = c } }, .dest = s1, .type = .i32 });
    try func.getBlock(b_exit).append(.{ .op = .{ .add = .{ .lhs = b, .rhs = s1 } }, .dest = s2, .type = .i32 });
    try func.getBlock(b_exit).append(.{ .op = .{ .add = .{ .lhs = a, .rhs = s2 } }, .dest = s3, .type = .i32 });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = s3 }, .type = .void });

    return .{ .cands = .{ a, b, c }, .b_entry = b_entry, .b_exit = b_exit };
}

test "range_split pressure gate: low pressure (P <= T) skips entirely" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    _ = try buildRankFixture(&func);

    var sched = try MockSchedule.fromFunc(&func, allocator);
    defer sched.deinit();

    // Pool of 100 GPRs ⇒ T = 100; loop pressure is 4 ⇒ no spilling pressure.
    const stats = try splitLiveRangesAtLoopBoundariesWithConfig(&func, &sched, allocator, .{
        .min_candidates = 0,
        .min_loop_body_insts = 0,
        .num_available_phys_regs = 100,
        .num_callee_saved_regs = 100,
        .safety_margin = 0,
    });
    try std.testing.expectEqual(@as(u32, 0), stats.splits_applied);
    try std.testing.expect(stats.loops_skipped_pressure >= 1);
}

test "range_split pressure gate: exactly at threshold (P == T) skips" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    _ = try buildRankFixture(&func);

    var sched = try MockSchedule.fromFunc(&func, allocator);
    defer sched.deinit();

    // T = 4 == P ⇒ satSub(P,T) == 0 ⇒ not pressure-bound ⇒ skip.
    const stats = try splitLiveRangesAtLoopBoundariesWithConfig(&func, &sched, allocator, .{
        .min_candidates = 0,
        .min_loop_body_insts = 0,
        .num_available_phys_regs = 4,
        .num_callee_saved_regs = 4,
        .safety_margin = 0,
    });
    try std.testing.expectEqual(@as(u32, 0), stats.splits_applied);
    try std.testing.expect(stats.loops_skipped_pressure >= 1);
}

test "range_split ranker: high pressure splits K = P - T widest candidates" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();
    const fx = try buildRankFixture(&func);

    var sched = try MockSchedule.fromFunc(&func, allocator);
    defer sched.deinit();

    // P = 4, T = 3 ⇒ relief_needed = 1 ⇒ split exactly the single widest
    // candidate (A), not all three.
    const stats = try splitLiveRangesAtLoopBoundariesWithConfig(&func, &sched, allocator, .{
        .min_candidates = 0,
        .min_loop_body_insts = 0,
        .num_available_phys_regs = 3,
        .num_callee_saved_regs = 3,
        .safety_margin = 0,
    });
    try std.testing.expectEqual(@as(u32, 1), stats.splits_applied);
    // The widest range (A = cands[0]) is the one spilled at the entry pred.
    const set_val = firstLocalSetVal(func.getBlock(fx.b_entry).*);
    try std.testing.expect(set_val != null);
    try std.testing.expectEqual(fx.cands[0], set_val.?);
}

test "range_split call-clobber gate: call-spanning pressure splits via callee-saved pool" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b_entry = try func.newBlock();
    const b_hdr = try func.newBlock();
    const b_body = try func.newBlock();
    const b_exit = try func.newBlock();

    const a = func.newVReg();
    const b = func.newVReg();
    const ctr = func.newVReg();

    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = a, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 2 }, .dest = b, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 100 }, .dest = ctr, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    try func.getBlock(b_hdr).append(.{ .op = .{ .br_if = .{ .cond = ctr, .then_block = b_body, .else_block = b_exit } }, .type = .void });

    // A call inside the loop clobbers caller-saved regs; A and B (live
    // across it) compete for the callee-saved pool.
    try func.getBlock(b_body).append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .type = .void });
    try func.getBlock(b_body).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    const s1 = func.newVReg();
    try func.getBlock(b_exit).append(.{ .op = .{ .add = .{ .lhs = a, .rhs = b } }, .dest = s1, .type = .i32 });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = s1 }, .type = .void });

    var sched = try MockSchedule.fromFunc(&func, allocator);
    defer sched.deinit();

    // Huge total GPR pool (ordinary gate never fires) but only 1 callee-
    // saved reg ⇒ the 2 call-spanning scalars trip the clobber gate.
    const stats = try splitLiveRangesAtLoopBoundariesWithConfig(&func, &sched, allocator, .{
        .min_candidates = 0,
        .min_loop_body_insts = 0,
        .num_available_phys_regs = 100,
        .num_callee_saved_regs = 1,
        .safety_margin = 0,
    });
    try std.testing.expectEqual(@as(u32, 2), stats.splits_applied);
}

test "splitLiveRangesAtLoopBoundaries: no-op on function without loops" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .ret = v0 }, .type = .void });

    var sched = try MockSchedule.fromFunc(&func, allocator);
    defer sched.deinit();

    const stats = try splitLiveRangesAtLoopBoundariesWithConfig(&func, &sched, allocator, .{ .min_candidates = 0, .min_loop_body_insts = 0, .num_available_phys_regs = 0, .num_callee_saved_regs = 0, .safety_margin = 0 });
    try std.testing.expectEqual(@as(u32, 0), stats.splits_applied);
    try std.testing.expectEqual(@as(u32, 0), stats.loops_considered);
}

test "splitLiveRangesAtLoopBoundaries: splits vreg crossing single-exit loop" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    // entry: v0 = const 7        (long-lived, used post-loop)
    //        v1 = const 100      (loop counter init)
    //        br hdr
    // hdr:   br_if (v1 != 0) -> body, exit
    // body:  v1' = v1 - v1
    //        br hdr              (back-edge — makes hdr a natural-loop header)
    // exit:  ret v0              ← post-loop use of v0
    //
    // v0 crosses the loop without being used inside ⇒ splittable.

    const b_entry = try func.newBlock();
    const b_hdr = try func.newBlock();
    const b_body = try func.newBlock();
    const b_exit = try func.newBlock();

    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v1_dec = func.newVReg();

    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 100 }, .dest = v1, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    try func.getBlock(b_hdr).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = b_body, .else_block = b_exit } }, .type = .void });

    try func.getBlock(b_body).append(.{ .op = .{ .sub = .{ .lhs = v1, .rhs = v1 } }, .dest = v1_dec, .type = .i32 });
    try func.getBlock(b_body).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    try func.getBlock(b_exit).append(.{ .op = .{ .ret = v0 }, .type = .void });

    var sched = try MockSchedule.fromFunc(&func, allocator);
    defer sched.deinit();

    const before_locals = func.local_count;
    const before_vregs = func.next_vreg;

    const stats = try splitLiveRangesAtLoopBoundariesWithConfig(&func, &sched, allocator, .{ .min_candidates = 0, .min_loop_body_insts = 0, .num_available_phys_regs = 0, .num_callee_saved_regs = 0, .safety_margin = 0 });

    try std.testing.expect(stats.splits_applied >= 1);
    try std.testing.expect(stats.loops_considered >= 1);

    // Side effects: one fresh local slot + one fresh vreg per split.
    try std.testing.expectEqual(before_locals + stats.splits_applied, func.local_count);
    try std.testing.expectEqual(before_vregs + stats.splits_applied, func.next_vreg);

    // local_set inserted in entry block (the single entry-pred); local_get
    // inserted in the exit block (the single exit-succ).
    try std.testing.expect(countLocalSetsInBlock(func.getBlock(b_entry).*) >= 1);
    try std.testing.expect(countLocalGetsInBlock(func.getBlock(b_exit).*) >= 1);

    // ret in exit block now uses the *fresh* vreg, not v0.
    const exit_block = func.getBlock(b_exit);
    var last_ret: ?ir.VReg = null;
    for (exit_block.instructions.items) |inst| if (inst.op == .ret) {
        last_ret = inst.op.ret;
    };
    try std.testing.expect(last_ret != null);
    try std.testing.expect(last_ret.? != v0);

    // Schedule mirrors the same edits.
    var sched_entry_sets: usize = 0;
    for (sched.blocks[b_entry].instructions) |inst| if (inst.op == .local_set) {
        sched_entry_sets += 1;
    };
    var sched_exit_gets: usize = 0;
    for (sched.blocks[b_exit].instructions) |inst| if (inst.op == .local_get) {
        sched_exit_gets += 1;
    };
    try std.testing.expect(sched_entry_sets >= 1);
    try std.testing.expect(sched_exit_gets >= 1);

    // Defends against the silent clamp at analysis.zig: every produced
    // live range must satisfy end >= start. The synthetic ret-of-fresh-vreg
    // would corrupt the range if range_split miscomputed.
    const ranges = try analysis.computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);
    for (ranges) |r| try std.testing.expect(r.end >= r.start);
}

test "splitLiveRangesAtLoopBoundaries: skips multi-exit loops" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b_entry = try func.newBlock();
    const b_hdr = try func.newBlock();
    const b_body = try func.newBlock();
    const b_exit_a = try func.newBlock();
    const b_exit_b = try func.newBlock();

    const v0 = func.newVReg();
    const c = func.newVReg();
    const c2 = func.newVReg();

    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    try func.getBlock(b_hdr).append(.{ .op = .{ .br_if = .{ .cond = c, .then_block = b_body, .else_block = b_exit_a } }, .type = .void });

    try func.getBlock(b_body).append(.{ .op = .{ .iconst_32 = 0 }, .dest = c2, .type = .i32 });
    try func.getBlock(b_body).append(.{ .op = .{ .br_if = .{ .cond = c2, .then_block = b_hdr, .else_block = b_exit_b } }, .type = .void });

    try func.getBlock(b_exit_a).append(.{ .op = .{ .ret = v0 }, .type = .void });
    try func.getBlock(b_exit_b).append(.{ .op = .{ .ret = v0 }, .type = .void });

    var sched = try MockSchedule.fromFunc(&func, allocator);
    defer sched.deinit();

    const stats = try splitLiveRangesAtLoopBoundariesWithConfig(&func, &sched, allocator, .{ .min_candidates = 0, .min_loop_body_insts = 0, .num_available_phys_regs = 0, .num_callee_saved_regs = 0, .safety_margin = 0 });
    try std.testing.expect(stats.loops_considered >= 1);
    try std.testing.expectEqual(@as(u32, 0), stats.splits_applied);
    try std.testing.expect(stats.loops_skipped_shape >= 1);
}

test "splitLiveRangesAtLoopBoundaries: leaves vregs used inside the loop alone" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b_entry = try func.newBlock();
    const b_hdr = try func.newBlock();
    const b_body = try func.newBlock();
    const b_exit = try func.newBlock();

    const v0 = func.newVReg();
    const ctr = func.newVReg();
    const sum = func.newVReg();

    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v0, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 100 }, .dest = ctr, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    try func.getBlock(b_hdr).append(.{ .op = .{ .br_if = .{ .cond = ctr, .then_block = b_body, .else_block = b_exit } }, .type = .void });

    try func.getBlock(b_body).append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = ctr } }, .dest = sum, .type = .i32 }); // uses v0
    try func.getBlock(b_body).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    try func.getBlock(b_exit).append(.{ .op = .{ .ret = v0 }, .type = .void });

    var sched = try MockSchedule.fromFunc(&func, allocator);
    defer sched.deinit();

    const stats = try splitLiveRangesAtLoopBoundariesWithConfig(&func, &sched, allocator, .{ .min_candidates = 0, .min_loop_body_insts = 0, .num_available_phys_regs = 0, .num_callee_saved_regs = 0, .safety_margin = 0 });
    try std.testing.expectEqual(@as(u32, 0), stats.splits_applied);
}

test "splitLiveRangesAtLoopBoundaries: N+1 hot-loop synthetic — fresh vreg redirects post-loop uses" {
    // Issue #383 acceptance pattern: a hot loop with N+1 simultaneously-
    // live scalars where N regs are available. The pre-loop scalar that's
    // unused inside the loop must be split so its register is free during
    // the loop body. The fresh vreg picked up by the post-loop reload
    // breaks the over-pressure interval.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b_entry = try func.newBlock();
    const b_hdr = try func.newBlock();
    const b_body = try func.newBlock();
    const b_exit = try func.newBlock();

    // Pre-loop: a few long-lived values that ARE used post-loop but NOT
    // inside the loop, plus the loop counter.
    const long1 = func.newVReg();
    const long2 = func.newVReg();
    const long3 = func.newVReg();
    const ctr = func.newVReg();

    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = long1, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 2 }, .dest = long2, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 3 }, .dest = long3, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 100 }, .dest = ctr, .type = .i32 });
    try func.getBlock(b_entry).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    try func.getBlock(b_hdr).append(.{ .op = .{ .br_if = .{ .cond = ctr, .then_block = b_body, .else_block = b_exit } }, .type = .void });

    // Body uses only ctr — none of long1/long2/long3.
    const tmp = func.newVReg();
    try func.getBlock(b_body).append(.{ .op = .{ .sub = .{ .lhs = ctr, .rhs = ctr } }, .dest = tmp, .type = .i32 });
    try func.getBlock(b_body).append(.{ .op = .{ .br = b_hdr }, .type = .void });

    // Post-loop uses long1+long2+long3.
    const sum_ab = func.newVReg();
    const sum_abc = func.newVReg();
    try func.getBlock(b_exit).append(.{ .op = .{ .add = .{ .lhs = long1, .rhs = long2 } }, .dest = sum_ab, .type = .i32 });
    try func.getBlock(b_exit).append(.{ .op = .{ .add = .{ .lhs = sum_ab, .rhs = long3 } }, .dest = sum_abc, .type = .i32 });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = sum_abc }, .type = .void });

    var sched = try MockSchedule.fromFunc(&func, allocator);
    defer sched.deinit();

    const before_locals = func.local_count;
    const stats = try splitLiveRangesAtLoopBoundariesWithConfig(&func, &sched, allocator, .{ .min_candidates = 0, .min_loop_body_insts = 0, .num_available_phys_regs = 0, .num_callee_saved_regs = 0, .safety_margin = 0 });

    // All three long-lived scalars should be split.
    try std.testing.expectEqual(@as(u32, 3), stats.splits_applied);
    try std.testing.expectEqual(before_locals + 3, func.local_count);

    // Three local_set in entry, three local_get at start of exit.
    try std.testing.expect(countLocalSetsInBlock(func.getBlock(b_entry).*) >= 3);
    try std.testing.expect(countLocalGetsInBlock(func.getBlock(b_exit).*) >= 3);

    // Crucially: no instruction in the body references long1/long2/long3.
    // (That was already the case before the split; the assertion verifies
    // the split didn't accidentally introduce a body reference.)
    for (func.getBlock(b_body).instructions.items) |inst| {
        var ctx = struct {
            forbidden: [3]ir.VReg,
            found: bool = false,
        }{ .forbidden = .{ long1, long2, long3 } };
        const visitor = struct {
            fn f(c: *@TypeOf(ctx), v: ir.VReg) anyerror!void {
                for (c.forbidden) |f_| if (v == f_) {
                    c.found = true;
                };
            }
        }.f;
        try forEachUseInst(inst, &ctx, visitor);
        try std.testing.expect(!ctx.found);
    }

    // Live ranges of long1/long2/long3 end before the loop; ranges of the
    // fresh post-loop vregs start at the exit block. Defend against the
    // silent clamp.
    const ranges = try analysis.computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);
    for (ranges) |r| try std.testing.expect(r.end >= r.start);

    // Find the splits: original long-vregs' ranges should end early
    // (before the exit block's first instruction); the fresh vregs' ranges
    // start inside the exit block.
    var orig_max_end: u32 = 0;
    var fresh_min_start: u32 = std.math.maxInt(u32);
    for (ranges) |r| {
        if (r.vreg == long1 or r.vreg == long2 or r.vreg == long3) {
            if (r.end > orig_max_end) orig_max_end = r.end;
        } else if (r.vreg >= 5) { // fresh vregs allocated after long1..ctr+tmp
            // approximate: the fresh post-loop vregs are the ones whose
            // range starts >= the local_get insertion point.
            if (r.start < fresh_min_start) fresh_min_start = r.start;
        }
    }
    // The split should have moved the originals' last-use BEFORE the
    // exit block (their last use is now the local_set, which lives in
    // the entry block). Concretely: orig_max_end < fresh_min_start (the
    // fresh vreg's range starts at the post-loop reload, strictly after
    // the originals' last use).
    try std.testing.expect(orig_max_end < fresh_min_start);
}
