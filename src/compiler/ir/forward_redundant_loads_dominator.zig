//! Cross-block / dominator-aware redundant-load forwarder (#391).
//!
//! Promotes the single-block `forwardRedundantLoads` pass to a
//! dominator-tree walk. The algorithm (per #391):
//!
//!   1. DFS the dominator tree from the entry block.
//!   2. On entry to each block, push a fresh `(MemKey -> VReg)` map.
//!   3. While scanning instructions, lookups walk the stack from
//!      innermost to outermost frame. A hit means an ancestor (or this
//!      block) already loaded the same `(base, offset, size, sign_extend)`
//!      and no aliasing store / call / barrier has run since.
//!   4. Stores invalidate aliasing entries in every active frame (a
//!      store anywhere on a dominator path kills cached values seen by
//!      dominated blocks). Calls and barriers clear every frame fully.
//!   5. On exit from a block, pop its frame.
//!
//! Invalidation rules mirror `forward_redundant_loads.zig`; the key
//! type lives in `alias_class.zig` so #467's wasm-local-slot extension
//! can drop a new union variant in one place.
//!
//! Sibling-block invariance: stores in one branch of a diamond do not
//! propagate up into the dominator's frame (we only invalidate frames
//! that are on the current DFS path), so the unrelated sibling-then-
//! merge block still observes the dominator's load.
//!
//! Sibling-block BARRIER soundness (#719): a barrier (call / atomic /
//! bulk-memory) on one diamond branch DOES need to wipe ancestor
//! cached loads, because a merge block dominated by the diamond head
//! is reachable via the barrier-containing branch on some execution
//! path. We achieve this with `passes.sortDomChildrenBarrierLast`,
//! which schedules barrier-containing subtrees to be DFS-visited
//! before their non-barrier siblings — so the barrier's `clearAll`
//! takes effect on ancestor frames before any non-barrier subtree
//! gets a chance to forward stale loads into a downstream merge.

const std = @import("std");
const ir = @import("ir.zig");
const passes = @import("passes.zig");
const analysis = @import("analysis.zig");
const alias_class = @import("alias_class.zig");

const MemKey = alias_class.MemKey;

pub fn forwardRedundantLoadsDominator(
    func: *ir.IrFunction,
    allocator: std.mem.Allocator,
) !bool {
    if (func.blocks.items.len == 0) return false;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();

    const nblocks = func.blocks.items.len;

    // Build dom-tree children list.
    var children = try allocator.alloc(std.ArrayList(ir.BlockId), nblocks);
    defer {
        for (children) |*list| list.deinit(allocator);
        allocator.free(children);
    }
    for (children) |*list| list.* = .empty;
    for (0..nblocks) |i| {
        const bid: ir.BlockId = @intCast(i);
        const idom_opt = dom.idom[bid];
        if (idom_opt == null) continue; // unreachable
        const idom = idom_opt.?;
        if (idom == bid) continue; // entry's idom is itself
        try children[idom].append(allocator, bid);
    }
    try passes.sortDomChildrenBarrierLast(func, &dom, children, allocator);

    if (dom.idom[0] == null) return false;

    var changed = false;

    // Stack of per-dominator-level value tables. Lookup walks top→bottom.
    var table_stack: std.ArrayList(std.AutoHashMap(MemKey, ir.VReg)) = .empty;
    defer {
        for (table_stack.items) |*t| t.deinit();
        table_stack.deinit(allocator);
    }

    // DFS frames. Each block is visited twice: phase=0 to scan and push
    // children, phase=1 to pop its table on the way back up.
    const Frame = struct { bid: ir.BlockId, phase: u1 };
    var dfs: std.ArrayList(Frame) = .empty;
    defer dfs.deinit(allocator);
    try dfs.append(allocator, .{ .bid = 0, .phase = 0 });

    while (dfs.items.len > 0) {
        const top = &dfs.items[dfs.items.len - 1];
        if (top.phase == 1) {
            var popped = table_stack.pop().?;
            popped.deinit();
            _ = dfs.pop();
            continue;
        }
        top.phase = 1;
        const bid = top.bid;

        try table_stack.append(allocator, std.AutoHashMap(MemKey, ir.VReg).init(allocator));
        var table = &table_stack.items[table_stack.items.len - 1];

        const block = &func.blocks.items[bid];

        var i: usize = 0;
        var alias_buf: std.ArrayList(MemKey) = .empty;
        defer alias_buf.deinit(allocator);

        while (i < block.instructions.items.len) {
            const inst = &block.instructions.items[i];
            switch (inst.op) {
                .load => |ld| {
                    const key = alias_class.memKeyFromLoad(ld);

                    var hit: ?ir.VReg = null;
                    var t = table_stack.items.len;
                    while (t > 0) : (t -= 1) {
                        if (table_stack.items[t - 1].get(key)) |held| {
                            hit = held;
                            break;
                        }
                    }

                    if (hit) |held_vreg| {
                        if (inst.dest) |dest| {
                            passes.replaceVReg(func, dest, held_vreg);
                        }
                        _ = block.instructions.orderedRemove(i);
                        changed = true;
                        continue;
                    }

                    if (inst.dest) |dest| {
                        try table.put(key, dest);
                    }
                },
                .store => |st| {
                    // Invalidate aliasing entries across every active
                    // frame on the current DFS path. Sibling branches
                    // not on the path are untouched (their frames are
                    // not present in the stack).
                    for (table_stack.items) |*frame| {
                        alias_buf.clearRetainingCapacity();
                        var it = frame.iterator();
                        while (it.next()) |entry| {
                            if (alias_class.storeAliases(entry.key_ptr.*, st)) {
                                try alias_buf.append(allocator, entry.key_ptr.*);
                            }
                        }
                        for (alias_buf.items) |k| _ = frame.remove(k);
                    }
                },
                .call,
                .call_indirect,
                .call_ref,
                .atomic_load,
                .atomic_store,
                .atomic_rmw,
                .atomic_cmpxchg,
                .atomic_fence,
                .atomic_notify,
                .atomic_wait,
                .memory_copy,
                .memory_fill,
                .memory_init,
                .memory_grow,
                => {
                    for (table_stack.items) |*frame| frame.clearRetainingCapacity();
                },
                else => {},
            }
            i += 1;
        }

        // Push children so each is visited under our just-populated frame.
        for (children[bid].items) |c| {
            try dfs.append(allocator, .{ .bid = c, .phase = 0 });
        }
    }

    return changed;
}

// ── Tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

/// Append a `br` terminator wiring `block.successors`-equivalent via the
/// only mechanism the IR actually uses: terminator ops.
fn terminateBr(blk: *ir.BasicBlock, target: ir.BlockId) !void {
    try blk.append(.{ .op = .{ .br = target }, .type = .i32 });
}

fn terminateBrIf(blk: *ir.BasicBlock, cond: ir.VReg, t: ir.BlockId, e: ir.BlockId) !void {
    try blk.append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = t, .else_block = e } }, .type = .i32 });
}

test "dominator FRL: forwards through linear chain (entry → mid → tail)" {
    const allocator = testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const entry = try func.newBlock();
    const mid = try func.newBlock();
    const tail = try func.newBlock();

    const v_base = func.newVReg();
    const v_load1 = func.newVReg();
    const v_load2 = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_load1, .type = .i32 });
        try terminateBr(blk, mid);
    }
    {
        const blk = &func.blocks.items[mid];
        try terminateBr(blk, tail);
    }
    {
        const blk = &func.blocks.items[tail];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_load2, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    try testing.expect(changed);

    // Tail's redundant load should be gone; only `ret` remains.
    const tail_blk = &func.blocks.items[tail];
    try testing.expectEqual(@as(usize, 1), tail_blk.instructions.items.len);
    try testing.expect(tail_blk.instructions.items[0].op == .ret);
}

test "dominator FRL: diamond merge — load forwarded from dominator" {
    const allocator = testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const entry = try func.newBlock();
    const left = try func.newBlock();
    const right = try func.newBlock();
    const merge = try func.newBlock();

    const v_base = func.newVReg();
    const v_cond = func.newVReg();
    const v_dom = func.newVReg();
    const v_merge_load = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try terminateBrIf(blk, v_cond, left, right);
    }
    {
        const blk = &func.blocks.items[left];
        try terminateBr(blk, merge);
    }
    {
        const blk = &func.blocks.items[right];
        try terminateBr(blk, merge);
    }
    {
        const blk = &func.blocks.items[merge];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_merge_load, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    try testing.expect(changed);
    const m = &func.blocks.items[merge];
    try testing.expectEqual(@as(usize, 1), m.instructions.items.len);
    try testing.expect(m.instructions.items[0].op == .ret);
}

test "dominator FRL: sibling-block store does NOT invalidate dominator-cached value" {
    // entry: load v_dom@base; br_if to left/right
    // left:  store @base (aliases v_dom) → would invalidate its own frame,
    //        but must NOT touch the entry frame so `right` still forwards.
    // right: load @base (should be forwarded from entry's v_dom).
    const allocator = testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const entry = try func.newBlock();
    const left = try func.newBlock();
    const right = try func.newBlock();

    const v_base = func.newVReg();
    const v_cond = func.newVReg();
    const v_dom = func.newVReg();
    const v_store_val = func.newVReg();
    const v_right_load = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try terminateBrIf(blk, v_cond, left, right);
    }
    {
        const blk = &func.blocks.items[left];
        try blk.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v_store_val } }, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }
    {
        const blk = &func.blocks.items[right];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_right_load, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    try testing.expect(changed);

    // Right block's load forwarded; only `ret` remains.
    const r = &func.blocks.items[right];
    try testing.expectEqual(@as(usize, 1), r.instructions.items.len);
    try testing.expect(r.instructions.items[0].op == .ret);

    // Left block's store survives.
    const l = &func.blocks.items[left];
    try testing.expect(l.instructions.items[0].op == .store);
}

test "dominator FRL: call between dominator-load and dominated-load clears scope" {
    const allocator = testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const entry = try func.newBlock();
    const tail = try func.newBlock();

    const v_base = func.newVReg();
    const v_dom = func.newVReg();
    const v_call_res = func.newVReg();
    const v_tail_load = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try blk.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v_call_res, .type = .i32 });
        try terminateBr(blk, tail);
    }
    {
        const blk = &func.blocks.items[tail];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_tail_load, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    try testing.expect(!changed);
    // Tail's load is preserved (still 2 insts including ret).
    const t = &func.blocks.items[tail];
    try testing.expectEqual(@as(usize, 2), t.instructions.items.len);
    try testing.expect(t.instructions.items[0].op == .load);
}

test "dominator FRL: call in sibling branch invalidates merge-block forwarding (#719)" {
    // entry:  v_dom = load p[0]; br_if cond, sib, tail
    // sib:    call f; br tail
    // tail:   v_tail = load p[0]; ret
    //
    // idom(tail) = entry. The path entry → sib → tail crosses a
    // barrier, so the load in `tail` must NOT be forwarded from `v_dom`.
    //
    // Regression test mirroring the GVN-side bug (issue #719): without
    // dom-DFS reordering, `tail` is popped before `sib` and forwards
    // the stale ancestor value.
    const allocator = testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const entry = try func.newBlock();
    const sib = try func.newBlock();
    const tail = try func.newBlock();

    const v_base = func.newVReg();
    const cond = func.newVReg();
    const v_dom = func.newVReg();
    const v_call = func.newVReg();
    const v_tail = func.newVReg();

    {
        const blk = &func.blocks.items[entry];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_dom, .type = .i32 });
        try terminateBrIf(blk, cond, sib, tail);
    }
    {
        const blk = &func.blocks.items[sib];
        try blk.append(.{ .op = .{ .call = .{ .func_idx = 0 } }, .dest = v_call, .type = .i32 });
        try terminateBr(blk, tail);
    }
    {
        const blk = &func.blocks.items[tail];
        try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_tail, .type = .i32 });
        try blk.append(.{ .op = .{ .ret = null }, .type = .i32 });
    }

    _ = try forwardRedundantLoadsDominator(&func, allocator);
    const t = &func.blocks.items[tail];
    try testing.expectEqual(@as(usize, 2), t.instructions.items.len);
    try testing.expect(t.instructions.items[0].op == .load);
}
