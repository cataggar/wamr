//! Per-local "needs prologue zero-init" analysis.
//!
//! Wasm requires every non-parameter local to start at zero. Codegen
//! emits a prologue zero-init store for each declared local slot. When
//! we can prove every reachable `local_get i` is dominated by a
//! `local_set i` on every path from the function entry, the prologue
//! store for `i` is dead and may be skipped.
//!
//! The analysis is a forward "must-be-assigned" dataflow over the IR
//! CFG. For each block we compute:
//!   • `gen[b]`  — locals that are unconditionally set in `b` before any
//!                  read of them in `b` (i.e. the first observation in
//!                  block-order is a `local_set`).
//!   • `kill[b]` — locals that are read in `b` before any set of them in
//!                  `b` (i.e. the first observation is a `local_get`).
//!
//! Lattice element: a set of locals "known assigned". Top = universe.
//! Transfer: `out[b] = in[b] ∪ gen[b]`.
//! Merge:    `in[b] = ∩ out[p] for p ∈ preds(b)`, with `in[entry] = {params}`.
//!
//! A local `i ≥ param_count` needs prologue init iff there exists a
//! reachable block `b` such that `i ∈ kill[b]` and `i ∉ in[b]`.
//!
//! The dataflow is path-sensitive at merge points: a `local_set i` in
//! only one arm of an `if/else` does NOT imply definite assignment.
//! Unreachable blocks (no predecessors and not the entry) start at top
//! so they never force a local back into the needs-init set.

const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");

/// Fixpoint solver selector for `computeNeedsInitImpl`. `worklist` is the
/// production path; `round_robin` is the reference oracle a differential
/// test pins the worklist to. Both compute the identical greatest fixpoint
/// of the same monotone forward must-assigned framework, so the resulting
/// needs-init bitmap is identical — only the iteration strategy differs.
const SolverMode = enum { worklist, round_robin };

/// Compute the per-local "needs zero-init" bitmap for `func`.
///
/// The returned slice has length `func.local_count`. Index `i` is
/// `true` iff the prologue must still emit a zero store for local `i`.
/// Parameter slots (i < param_count) are always `false` — the caller's
/// ABI spill, not the prologue zero-init, initialises them.
///
/// Caller owns the returned slice and must `allocator.free` it.
pub fn computeNeedsInit(
    allocator: std.mem.Allocator,
    func: *const ir.IrFunction,
) ![]bool {
    return computeNeedsInitImpl(allocator, func, .worklist);
}

/// Reference round-robin solver retained as the differential oracle for
/// `computeNeedsInit` (the worklist version). Sweeps every block until a
/// full pass makes no change; both compute the same greatest fixpoint.
/// Used by tests only — do not call from codegen (it rescans every block
/// each round, which is super-linear in block count, issue #780).
fn computeNeedsInitRoundRobin(
    allocator: std.mem.Allocator,
    func: *const ir.IrFunction,
) ![]bool {
    return computeNeedsInitImpl(allocator, func, .round_robin);
}

fn computeNeedsInitImpl(
    allocator: std.mem.Allocator,
    func: *const ir.IrFunction,
    comptime mode: SolverMode,
) ![]bool {
    const n_locals: usize = func.local_count;
    const result = try allocator.alloc(bool, n_locals);
    @memset(result, false);

    // No declared locals → nothing to do.
    if (func.local_count <= func.param_count) return result;
    // No blocks → no IR; conservatively keep everything zero (nothing
    // to skip but also nothing to break).
    if (func.blocks.items.len == 0) {
        var j: usize = func.param_count;
        while (j < n_locals) : (j += 1) result[j] = true;
        return result;
    }

    const n_blocks: usize = func.blocks.items.len;
    const stride: usize = n_locals;

    // gen[b][i] = local i is definitely assigned somewhere in b before
    // any read of i in b (first in-block observation is a set).
    const gen = try allocator.alloc(bool, n_blocks * stride);
    defer allocator.free(gen);
    @memset(gen, false);
    // kill[b][i] = local i is read in b before any in-block set of i.
    const kill = try allocator.alloc(bool, n_blocks * stride);
    defer allocator.free(kill);
    @memset(kill, false);

    // First-observation scan per block.
    for (func.blocks.items, 0..) |block, bi| {
        // 0 = unobserved, 1 = first-was-set, 2 = first-was-get.
        const seen = try allocator.alloc(u2, stride);
        defer allocator.free(seen);
        @memset(seen, 0);
        for (block.instructions.items) |inst| {
            switch (inst.op) {
                .local_get => |idx| {
                    if (idx < n_locals and seen[idx] == 0) {
                        seen[idx] = 2;
                        kill[bi * stride + idx] = true;
                    }
                },
                .local_set => |ls| {
                    if (ls.idx < n_locals and seen[ls.idx] == 0) {
                        seen[ls.idx] = 1;
                        gen[bi * stride + ls.idx] = true;
                    }
                },
                else => {},
            }
        }
    }

    // Build predecessors using existing CFG helper.
    var preds = try analysis.buildPredecessors(func, allocator);
    defer {
        var it = preds.iterator();
        while (it.next()) |e| allocator.free(e.value_ptr.*);
        preds.deinit();
    }

    // Mark reachable blocks from entry (block 0) via BFS on successors.
    var successors = try analysis.buildSuccessors(func, allocator);
    defer {
        var sit = successors.iterator();
        while (sit.next()) |e| allocator.free(e.value_ptr.*);
        successors.deinit();
    }

    const reachable = try allocator.alloc(bool, n_blocks);
    defer allocator.free(reachable);
    @memset(reachable, false);
    {
        var queue: std.ArrayList(ir.BlockId) = .empty;
        defer queue.deinit(allocator);
        try queue.append(allocator, 0);
        reachable[0] = true;
        var head: usize = 0;
        while (head < queue.items.len) : (head += 1) {
            const b = queue.items[head];
            if (successors.get(b)) |succs| {
                for (succs) |s| {
                    if (s < n_blocks and !reachable[s]) {
                        reachable[s] = true;
                        try queue.append(allocator, s);
                    }
                }
            }
        }
    }

    // in[b][i] / out[b][i] — must-be-assigned sets.
    // Top element = all-true. Entry's `in` starts as {params}.
    const in_set = try allocator.alloc(bool, n_blocks * stride);
    defer allocator.free(in_set);
    const out_set = try allocator.alloc(bool, n_blocks * stride);
    defer allocator.free(out_set);
    @memset(in_set, true);
    @memset(out_set, true);

    // Entry block in_set = params only.
    {
        const base = 0 * stride;
        var i: usize = 0;
        while (i < n_locals) : (i += 1) {
            in_set[base + i] = (i < func.param_count);
        }
        // out[entry] = in[entry] ∪ gen[entry].
        i = 0;
        while (i < n_locals) : (i += 1) {
            out_set[base + i] = in_set[base + i] or gen[base + i];
        }
    }

    // Iterate to fixed point. `worklist` revisits a block only when a
    // predecessor's `out_set` changed; `round_robin` (the reference oracle)
    // sweeps every block each round. Both converge to the same greatest
    // fixpoint of this monotone forward must-assigned framework.
    switch (mode) {
        .round_robin => {
            // Simple round-robin over blocks in index order. Super-linear in
            // block count on large functions (issue #780) — kept only as the
            // differential oracle for the worklist solver below.
            var changed = true;
            var safety: usize = 0;
            const max_iters: usize = (n_blocks + 4) * 8;
            while (changed) {
                changed = false;
                safety += 1;
                if (safety > max_iters) break; // safety net; lattice height is bounded
                for (func.blocks.items, 0..) |_, bi| {
                    if (bi == 0) continue;
                    if (!reachable[bi]) continue;
                    const new_in = try allocator.alloc(bool, stride);
                    defer allocator.free(new_in);
                    // Initialize to top (all true) then intersect with each
                    // reachable predecessor's out_set.
                    @memset(new_in, true);
                    var have_pred = false;
                    if (preds.get(@intCast(bi))) |ps| {
                        for (ps) |p| {
                            if (p >= n_blocks) continue;
                            if (!reachable[p]) continue;
                            have_pred = true;
                            const pbase = @as(usize, p) * stride;
                            var i: usize = 0;
                            while (i < n_locals) : (i += 1) {
                                new_in[i] = new_in[i] and out_set[pbase + i];
                            }
                        }
                    }
                    // Reachable block with no reachable predecessor (besides
                    // entry which we already skipped) can only happen via the
                    // implicit fallthrough from entry handled elsewhere; treat
                    // such blocks as bottom-of-knowledge to stay conservative.
                    if (!have_pred) {
                        @memset(new_in, false);
                        // params are still known-assigned everywhere.
                        var i: usize = 0;
                        while (i < func.param_count) : (i += 1) new_in[i] = true;
                    }

                    const base = bi * stride;
                    var differs = false;
                    var i: usize = 0;
                    while (i < n_locals) : (i += 1) {
                        if (in_set[base + i] != new_in[i]) {
                            in_set[base + i] = new_in[i];
                            differs = true;
                        }
                    }
                    if (differs) {
                        changed = true;
                        i = 0;
                        while (i < n_locals) : (i += 1) {
                            out_set[base + i] = in_set[base + i] or gen[base + i];
                        }
                    }
                }
            }
        },
        .worklist => {
            // Worklist solver: only revisit a block when a predecessor's
            // `out_set` changed, instead of rescanning every block each
            // round. The entry block's in/out are fixed (params /
            // params∪gen) and are never recomputed. A single reusable
            // `new_in` buffer replaces the round-robin's per-block-per-round
            // heap allocation.
            var in_queue = try allocator.alloc(bool, n_blocks);
            defer allocator.free(in_queue);
            @memset(in_queue, false);

            var worklist: std.ArrayList(ir.BlockId) = .empty;
            defer worklist.deinit(allocator);

            // Seed every reachable non-entry block, high index first; LIFO
            // pop then visits low index first — i.e. ~reverse-postorder for
            // the usual (roughly topological) block numbering, the fast
            // order for a forward dataflow.
            var seed: usize = n_blocks;
            while (seed > 1) {
                seed -= 1;
                if (!reachable[seed]) continue;
                try worklist.append(allocator, @intCast(seed));
                in_queue[seed] = true;
            }

            const new_in = try allocator.alloc(bool, stride);
            defer allocator.free(new_in);

            while (worklist.pop()) |bi| {
                in_queue[bi] = false;

                // in[b] = ∩ out[reachable preds]; top when no preds processed.
                @memset(new_in, true);
                var have_pred = false;
                if (preds.get(bi)) |ps| {
                    for (ps) |p| {
                        if (p >= n_blocks) continue;
                        if (!reachable[p]) continue;
                        have_pred = true;
                        const pbase = @as(usize, p) * stride;
                        var i: usize = 0;
                        while (i < n_locals) : (i += 1) {
                            new_in[i] = new_in[i] and out_set[pbase + i];
                        }
                    }
                }
                if (!have_pred) {
                    @memset(new_in, false);
                    var i: usize = 0;
                    while (i < func.param_count) : (i += 1) new_in[i] = true;
                }

                const base = @as(usize, bi) * stride;
                var differs = false;
                var i: usize = 0;
                while (i < n_locals) : (i += 1) {
                    if (in_set[base + i] != new_in[i]) {
                        in_set[base + i] = new_in[i];
                        differs = true;
                    }
                }
                if (!differs) continue;

                // out[b] = in[b] ∪ gen[b]; if it changed, the block's
                // successors must be revisited.
                var out_changed = false;
                i = 0;
                while (i < n_locals) : (i += 1) {
                    const nv = in_set[base + i] or gen[base + i];
                    if (out_set[base + i] != nv) {
                        out_set[base + i] = nv;
                        out_changed = true;
                    }
                }
                if (!out_changed) continue;

                if (successors.get(bi)) |succs| {
                    for (succs) |s| {
                        if (s >= n_blocks) continue;
                        if (s == 0) continue; // entry is never recomputed
                        if (!reachable[s]) continue;
                        if (!in_queue[s]) {
                            in_queue[s] = true;
                            try worklist.append(allocator, @intCast(s));
                        }
                    }
                }
            }
        },
    }

    // A local needs init iff some reachable block reads it before any
    // in-block set AND it isn't in must-assigned-in for that block.
    var j: usize = func.param_count;
    while (j < n_locals) : (j += 1) {
        var needs = false;
        for (0..n_blocks) |bi| {
            if (!reachable[bi]) continue;
            if (kill[bi * stride + j] and !in_set[bi * stride + j]) {
                needs = true;
                break;
            }
        }
        result[j] = needs;
    }
    return result;
}

// ── Tests ───────────────────────────────────────────────────────────

const testing = std.testing;

test "computeNeedsInit: no locals beyond params → nothing needs init" {
    const A = testing.allocator;
    var func = ir.IrFunction.init(A, 2, 1, 2);
    defer func.deinit();
    _ = try func.newBlock();
    const ni = try computeNeedsInit(A, &func);
    defer A.free(ni);
    try testing.expect(!ni[0]);
    try testing.expect(!ni[1]);
}

test "computeNeedsInit: local set before any read can skip init" {
    const A = testing.allocator;
    var func = ir.IrFunction.init(A, 0, 1, 2);
    defer func.deinit();
    const b = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(b).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v, .type = .i32 });
    try func.getBlock(b).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v } } });
    try func.getBlock(b).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v } } });
    const v2 = func.newVReg();
    try func.getBlock(b).append(.{ .op = .{ .local_get = 0 }, .dest = v2, .type = .i32 });
    try func.getBlock(b).append(.{ .op = .{ .ret = v2 } });
    const ni = try computeNeedsInit(A, &func);
    defer A.free(ni);
    try testing.expectEqual(false, ni[0]);
    try testing.expectEqual(false, ni[1]);
}

test "computeNeedsInit: local read before set must keep init" {
    const A = testing.allocator;
    var func = ir.IrFunction.init(A, 0, 1, 1);
    defer func.deinit();
    const b = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(b).append(.{ .op = .{ .local_get = 0 }, .dest = v, .type = .i32 });
    try func.getBlock(b).append(.{ .op = .{ .ret = v } });
    const ni = try computeNeedsInit(A, &func);
    defer A.free(ni);
    try testing.expectEqual(true, ni[0]);
}

test "computeNeedsInit: set in only one arm of if is NOT definite" {
    const A = testing.allocator;
    var func = ir.IrFunction.init(A, 1, 1, 2); // 1 param, 1 declared local
    defer func.deinit();
    const entry = try func.newBlock();
    const then_b = try func.newBlock();
    const else_b = try func.newBlock();
    const join = try func.newBlock();
    try func.getBlock(then_b).addPredecessor(entry);
    try func.getBlock(else_b).addPredecessor(entry);
    try func.getBlock(join).addPredecessor(then_b);
    try func.getBlock(join).addPredecessor(else_b);

    const cond = func.newVReg();
    const v = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .local_get = 0 }, .dest = cond, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = then_b, .else_block = else_b } } });
    // Only the then-arm sets local 1.
    try func.getBlock(then_b).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v, .type = .i32 });
    try func.getBlock(then_b).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v } } });
    try func.getBlock(then_b).append(.{ .op = .{ .br = join } });
    try func.getBlock(else_b).append(.{ .op = .{ .br = join } });
    const r = func.newVReg();
    try func.getBlock(join).append(.{ .op = .{ .local_get = 1 }, .dest = r, .type = .i32 });
    try func.getBlock(join).append(.{ .op = .{ .ret = r } });

    const ni = try computeNeedsInit(A, &func);
    defer A.free(ni);
    try testing.expectEqual(false, ni[0]); // param
    try testing.expectEqual(true, ni[1]); // not definitely assigned on else path
}

test "computeNeedsInit: set in BOTH arms of if IS definite" {
    const A = testing.allocator;
    var func = ir.IrFunction.init(A, 1, 1, 2);
    defer func.deinit();
    const entry = try func.newBlock();
    const then_b = try func.newBlock();
    const else_b = try func.newBlock();
    const join = try func.newBlock();
    try func.getBlock(then_b).addPredecessor(entry);
    try func.getBlock(else_b).addPredecessor(entry);
    try func.getBlock(join).addPredecessor(then_b);
    try func.getBlock(join).addPredecessor(else_b);

    const cond = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .local_get = 0 }, .dest = cond, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = then_b, .else_block = else_b } } });
    const v1 = func.newVReg();
    try func.getBlock(then_b).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v1, .type = .i32 });
    try func.getBlock(then_b).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v1 } } });
    try func.getBlock(then_b).append(.{ .op = .{ .br = join } });
    const v2 = func.newVReg();
    try func.getBlock(else_b).append(.{ .op = .{ .iconst_32 = 9 }, .dest = v2, .type = .i32 });
    try func.getBlock(else_b).append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v2 } } });
    try func.getBlock(else_b).append(.{ .op = .{ .br = join } });
    const r = func.newVReg();
    try func.getBlock(join).append(.{ .op = .{ .local_get = 1 }, .dest = r, .type = .i32 });
    try func.getBlock(join).append(.{ .op = .{ .ret = r } });

    const ni = try computeNeedsInit(A, &func);
    defer A.free(ni);
    try testing.expectEqual(false, ni[0]);
    try testing.expectEqual(false, ni[1]);
}

test "computeNeedsInit: never-read local does not need init" {
    const A = testing.allocator;
    var func = ir.IrFunction.init(A, 0, 1, 3);
    defer func.deinit();
    const b = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(b).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v, .type = .i32 });
    try func.getBlock(b).append(.{ .op = .{ .ret = v } });
    const ni = try computeNeedsInit(A, &func);
    defer A.free(ni);
    try testing.expectEqual(false, ni[0]);
    try testing.expectEqual(false, ni[1]);
    try testing.expectEqual(false, ni[2]);
}

// ── #780: worklist vs round-robin differential ──────────────────────────────

test "computeNeedsInit: worklist matches round-robin across CFG shapes" {
    const A = testing.allocator;
    const builders = [_]*const fn (std.mem.Allocator) anyerror!ir.IrFunction{
        &buildNI_oneArmIf,
        &buildNI_bothArmsIf,
        &buildNI_loopSetBeforeBody,
        &buildNI_nestedLoop,
        &buildNI_forwardChain,
        &buildNI_reverseChain,
    };
    for (builders) |build| {
        var func = try build(A);
        defer func.deinit();
        const reference = try computeNeedsInitRoundRobin(A, &func);
        defer A.free(reference);
        const actual = try computeNeedsInit(A, &func);
        defer A.free(actual);
        try testing.expectEqualSlices(bool, reference, actual);
    }
}

test "computeNeedsInit: worklist matches round-robin on large reverse chain" {
    // A reverse-ordered chain is the round-robin's worst case (info
    // propagates one block per full sweep), and the case the worklist solver
    // handles directly. Both must produce the identical needs-init bitmap.
    const A = testing.allocator;
    var func = try buildNI_largeReverseChain(A, 1200);
    defer func.deinit();
    const reference = try computeNeedsInitRoundRobin(A, &func);
    defer A.free(reference);
    const actual = try computeNeedsInit(A, &func);
    defer A.free(actual);
    try testing.expectEqualSlices(bool, reference, actual);
}

// local 0 set only on the then-arm of an if → not definitely assigned at the
// join's read, so it needs init.
fn buildNI_oneArmIf(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const then_b = try func.newBlock();
    const else_b = try func.newBlock();
    const join = try func.newBlock();
    const cond = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = then_b, .else_block = else_b } } });
    const v = func.newVReg();
    try func.getBlock(then_b).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v, .type = .i32 });
    try func.getBlock(then_b).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v } } });
    try func.getBlock(then_b).append(.{ .op = .{ .br = join } });
    try func.getBlock(else_b).append(.{ .op = .{ .br = join } });
    const r = func.newVReg();
    try func.getBlock(join).append(.{ .op = .{ .local_get = 0 }, .dest = r, .type = .i32 });
    try func.getBlock(join).append(.{ .op = .{ .ret = r } });
    return func;
}

// local 0 set on both arms → definitely assigned at the join, no init needed.
fn buildNI_bothArmsIf(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const then_b = try func.newBlock();
    const else_b = try func.newBlock();
    const join = try func.newBlock();
    const cond = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = then_b, .else_block = else_b } } });
    const v1 = func.newVReg();
    try func.getBlock(then_b).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v1, .type = .i32 });
    try func.getBlock(then_b).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v1 } } });
    try func.getBlock(then_b).append(.{ .op = .{ .br = join } });
    const v2 = func.newVReg();
    try func.getBlock(else_b).append(.{ .op = .{ .iconst_32 = 9 }, .dest = v2, .type = .i32 });
    try func.getBlock(else_b).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v2 } } });
    try func.getBlock(else_b).append(.{ .op = .{ .br = join } });
    const r = func.newVReg();
    try func.getBlock(join).append(.{ .op = .{ .local_get = 0 }, .dest = r, .type = .i32 });
    try func.getBlock(join).append(.{ .op = .{ .ret = r } });
    return func;
}

// local 0 set in entry before the loop, read inside the loop body.
fn buildNI_loopSetBeforeBody(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const header = try func.newBlock();
    const body = try func.newBlock();
    const exit = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v } } });
    try func.getBlock(entry).append(.{ .op = .{ .br = header } });
    const c = func.newVReg();
    try func.getBlock(header).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c, .type = .i32 });
    try func.getBlock(header).append(.{ .op = .{ .br_if = .{ .cond = c, .then_block = body, .else_block = exit } } });
    const r0 = func.newVReg();
    // local 0 is definitely assigned (entry); local 1 is read but never set.
    try func.getBlock(body).append(.{ .op = .{ .local_get = 0 }, .dest = r0, .type = .i32 });
    const r1 = func.newVReg();
    try func.getBlock(body).append(.{ .op = .{ .local_get = 1 }, .dest = r1, .type = .i32 });
    try func.getBlock(body).append(.{ .op = .{ .br = header } });
    const rr = func.newVReg();
    try func.getBlock(exit).append(.{ .op = .{ .local_get = 0 }, .dest = rr, .type = .i32 });
    try func.getBlock(exit).append(.{ .op = .{ .ret = rr } });
    return func;
}

fn buildNI_nestedLoop(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const outer = try func.newBlock();
    const inner = try func.newBlock();
    const inner_body = try func.newBlock();
    const outer_tail = try func.newBlock();
    const exit = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v, .type = .i32 });
    try func.getBlock(entry).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v } } });
    try func.getBlock(entry).append(.{ .op = .{ .br = outer } });
    const c1 = func.newVReg();
    try func.getBlock(outer).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c1, .type = .i32 });
    try func.getBlock(outer).append(.{ .op = .{ .br_if = .{ .cond = c1, .then_block = inner, .else_block = exit } } });
    const c2 = func.newVReg();
    try func.getBlock(inner).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c2, .type = .i32 });
    try func.getBlock(inner).append(.{ .op = .{ .br_if = .{ .cond = c2, .then_block = inner_body, .else_block = outer_tail } } });
    const r0 = func.newVReg();
    try func.getBlock(inner_body).append(.{ .op = .{ .local_get = 0 }, .dest = r0, .type = .i32 });
    // local 1 read before any set inside the inner loop → needs init.
    const r1 = func.newVReg();
    try func.getBlock(inner_body).append(.{ .op = .{ .local_get = 1 }, .dest = r1, .type = .i32 });
    try func.getBlock(inner_body).append(.{ .op = .{ .br = inner } });
    try func.getBlock(outer_tail).append(.{ .op = .{ .br = outer } });
    const rr = func.newVReg();
    try func.getBlock(exit).append(.{ .op = .{ .local_get = 0 }, .dest = rr, .type = .i32 });
    try func.getBlock(exit).append(.{ .op = .{ .ret = rr } });
    return func;
}

// Forward chain: local set in block 0, read in the last block.
fn buildNI_forwardChain(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    errdefer func.deinit();
    var prev = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(prev).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v, .type = .i32 });
    try func.getBlock(prev).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v } } });
    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        const next = try func.newBlock();
        try func.getBlock(prev).append(.{ .op = .{ .br = next } });
        prev = next;
    }
    const r = func.newVReg();
    try func.getBlock(prev).append(.{ .op = .{ .local_get = 0 }, .dest = r, .type = .i32 });
    try func.getBlock(prev).append(.{ .op = .{ .ret = r } });
    return func;
}

// Reverse-ordered chain: entry (block 0) jumps to the highest-index block and
// the chain descends back to a low-index exit. Block indices run opposite to
// control flow, the round-robin solver's worst case.
fn buildNI_reverseChain(allocator: std.mem.Allocator) !ir.IrFunction {
    return buildNI_largeReverseChain(allocator, 24);
}

fn buildNI_largeReverseChain(allocator: std.mem.Allocator, n: u32) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 2);
    errdefer func.deinit();
    // Allocate n+1 blocks: block 0 is entry, blocks 1..n are the chain.
    var k: u32 = 0;
    while (k <= n) : (k += 1) _ = try func.newBlock();
    // Entry (block 0) sets local 0 (definitely assigned → stays known on the
    // whole chain), then jumps to the highest block n. Local 1 is never set,
    // so its "not-assigned" fact must propagate entry → exit across every
    // block of the reverse chain — the round-robin solver's worst case (one
    // block of progress per full sweep).
    const v = func.newVReg();
    try func.getBlock(0).append(.{ .op = .{ .iconst_32 = 5 }, .dest = v, .type = .i32 });
    try func.getBlock(0).append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v } } });
    try func.getBlock(0).append(.{ .op = .{ .br = @intCast(n) } });
    // Chain: block j → block j-1, for j from n down to 2.
    var j: u32 = n;
    while (j >= 2) : (j -= 1) {
        try func.getBlock(j).append(.{ .op = .{ .br = @intCast(j - 1) } });
    }
    // Block 1 is the exit: read local 0 (definitely assigned in entry → no
    // init) and local 1 (never assigned → needs init), then ret.
    const r0 = func.newVReg();
    try func.getBlock(1).append(.{ .op = .{ .local_get = 0 }, .dest = r0, .type = .i32 });
    const r1 = func.newVReg();
    try func.getBlock(1).append(.{ .op = .{ .local_get = 1 }, .dest = r1, .type = .i32 });
    try func.getBlock(1).append(.{ .op = .{ .ret = r0 } });
    return func;
}
