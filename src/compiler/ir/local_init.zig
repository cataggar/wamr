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

    // Iterate to fixed point. Order: simple round-robin over blocks
    // (reverse postorder would be faster but functions are usually
    // small; this stays simple and correct).
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
