const std = @import("std");
const ir = @import("ir.zig");
const passes = @import("passes.zig");
const alias_class = @import("alias_class.zig");

const LoadKey = alias_class.LoadKey;

/// Forward redundant loads within a basic block.
///
/// Two alias classes are tracked in a single value-table keyed by
/// `alias_class.LoadKey`:
///
///   * `mem` — wasm linear-memory loads keyed by
///     `(base, offset, size, sign_extend)`. A redundant load at the
///     same key is rewritten to use the earlier load's dest vreg and
///     deleted. Removal is safe because the prior load already
///     performed (and either succeeded at, or trapped during) the
///     bounds check; if we reach the redundant load at run-time the
///     same access would be in-bounds again. No intervening op
///     modifies linear memory at the stored location, so the value is
///     unchanged.
///
///   * `local` — wasm locals keyed by `local_idx`. A `local_set i, v`
///     caches `local: i → v`; a later `local_get i` is rewritten to
///     use `v` and deleted. A first `local_get i` caches its own dest
///     so subsequent `local_get i`s in the same block coalesce.
///     `local_set j` (j ≠ i) does NOT invalidate `local: i` — wasm
///     locals are not aliased across distinct indices.
///
/// Cross-class invalidation rules (see `alias_class.storeAliasesLoad`):
///   * `store` invalidates only the overlapping `.mem` entries.
///   * Calls and other barriers (atomic ops, `memory_*` bulk ops)
///     clear the entire table. Wasm semantics technically allow
///     preserving `.local` entries across calls (a callee cannot
///     mutate the caller's locals), but we stay conservative here
///     until inter-procedural escape analysis lands. `forwardLocalGet`
///     runs earlier in the pipeline and already handles the call-safe
///     local-forwarding case; this pass picks up local-forwarding
///     opportunities exposed *after* later passes (GVN, LICM, IV
///     simplification, etc.) reshape the IR.
///
/// Single-block scope only — cross-block forwarding is handled by
/// `forward_redundant_loads_dominator.zig` (#391).
pub fn forwardRedundantLoads(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var value_table = std.AutoHashMap(LoadKey, ir.VReg).init(allocator);
    defer value_table.deinit();

    var aliasing_keys: std.ArrayList(LoadKey) = .empty;
    defer aliasing_keys.deinit(allocator);

    var changed = false;

    for (func.blocks.items) |*block| {
        value_table.clearRetainingCapacity();

        var i: usize = 0;
        while (i < block.instructions.items.len) {
            const inst = &block.instructions.items[i];
            switch (inst.op) {
                .load => |ld| {
                    const key: LoadKey = .{ .mem = alias_class.memKeyFromLoad(ld) };
                    if (value_table.get(key)) |held_vreg| {
                        if (inst.dest) |dest| {
                            passes.replaceVReg(func, dest, held_vreg);
                        }
                        _ = block.instructions.orderedRemove(i);
                        changed = true;
                        continue;
                    }
                    if (inst.dest) |dest| {
                        try value_table.put(key, dest);
                    }
                },
                .store => |st| {
                    aliasing_keys.clearRetainingCapacity();
                    var it = value_table.iterator();
                    while (it.next()) |entry| {
                        if (alias_class.storeAliasesLoad(entry.key_ptr.*, st)) {
                            try aliasing_keys.append(allocator, entry.key_ptr.*);
                        }
                    }
                    for (aliasing_keys.items) |k| _ = value_table.remove(k);
                },
                .local_set => |ls| {
                    // Only invalidate this exact slot; wasm locals are
                    // not aliased across distinct indices.
                    _ = value_table.remove(.{ .local = ls.idx });
                    try value_table.put(.{ .local = ls.idx }, ls.val);
                },
                .local_get => |idx| {
                    const key: LoadKey = .{ .local = idx };
                    if (value_table.get(key)) |held_vreg| {
                        if (inst.dest) |dest| {
                            passes.replaceVReg(func, dest, held_vreg);
                        }
                        _ = block.instructions.orderedRemove(i);
                        changed = true;
                        continue;
                    }
                    if (inst.dest) |dest| {
                        try value_table.put(key, dest);
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
                    value_table.clearRetainingCapacity();
                },
                else => {},
            }
            i += 1;
        }
    }
    return changed;
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "forwardRedundantLoads: local_set then local_get rewrites use to set's val" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const blk = &func.blocks.items[b];

    const v = func.newVReg();
    const g = func.newVReg();
    const r = func.newVReg();

    try blk.append(.{ .op = .{ .local_set = .{ .idx = 7, .val = v } } });
    try blk.append(.{ .op = .{ .local_get = 7 }, .dest = g, .type = .i32 });
    try blk.append(.{ .op = .{ .add = .{ .lhs = g, .rhs = g } }, .dest = r, .type = .i32 });

    const changed = try forwardRedundantLoads(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(usize, 2), blk.instructions.items.len);
    try std.testing.expect(blk.instructions.items[0].op == .local_set);
    try std.testing.expect(blk.instructions.items[1].op == .add);
    try std.testing.expectEqual(v, blk.instructions.items[1].op.add.lhs);
    try std.testing.expectEqual(v, blk.instructions.items[1].op.add.rhs);
}

test "forwardRedundantLoads: local_set, unrelated op, local_get still rewrites" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const blk = &func.blocks.items[b];

    const v = func.newVReg();
    const tmp = func.newVReg();
    const g = func.newVReg();
    const r = func.newVReg();

    try blk.append(.{ .op = .{ .local_set = .{ .idx = 3, .val = v } } });
    try blk.append(.{ .op = .{ .add = .{ .lhs = v, .rhs = v } }, .dest = tmp, .type = .i32 });
    try blk.append(.{ .op = .{ .local_get = 3 }, .dest = g, .type = .i32 });
    try blk.append(.{ .op = .{ .add = .{ .lhs = g, .rhs = tmp } }, .dest = r, .type = .i32 });

    const changed = try forwardRedundantLoads(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(usize, 3), blk.instructions.items.len);
    try std.testing.expectEqual(v, blk.instructions.items[2].op.add.lhs);
}

test "forwardRedundantLoads: local_set j (j != i) does not invalidate local i" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const blk = &func.blocks.items[b];

    const v = func.newVReg();
    const w = func.newVReg();
    const g = func.newVReg();
    const r = func.newVReg();

    try blk.append(.{ .op = .{ .local_set = .{ .idx = 1, .val = v } } });
    try blk.append(.{ .op = .{ .local_set = .{ .idx = 2, .val = w } } });
    try blk.append(.{ .op = .{ .local_get = 1 }, .dest = g, .type = .i32 });
    try blk.append(.{ .op = .{ .add = .{ .lhs = g, .rhs = w } }, .dest = r, .type = .i32 });

    const changed = try forwardRedundantLoads(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(usize, 3), blk.instructions.items.len);
    try std.testing.expectEqual(v, blk.instructions.items[2].op.add.lhs);
}

test "forwardRedundantLoads: call invalidates local table" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const blk = &func.blocks.items[b];

    const v = func.newVReg();
    const g = func.newVReg();
    const r = func.newVReg();

    try blk.append(.{ .op = .{ .local_set = .{ .idx = 4, .val = v } } });
    try blk.append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try blk.append(.{ .op = .{ .local_get = 4 }, .dest = g, .type = .i32 });
    try blk.append(.{ .op = .{ .add = .{ .lhs = g, .rhs = g } }, .dest = r, .type = .i32 });

    const before_len = blk.instructions.items.len;
    const changed = try forwardRedundantLoads(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before_len, blk.instructions.items.len);
    try std.testing.expectEqual(g, blk.instructions.items[3].op.add.lhs);
}

test "forwardRedundantLoads: memory store does not invalidate local entries" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b = try func.newBlock();
    const blk = &func.blocks.items[b];

    const v = func.newVReg();
    const base = func.newVReg();
    const sv = func.newVReg();
    const g = func.newVReg();
    const r = func.newVReg();

    try blk.append(.{ .op = .{ .local_set = .{ .idx = 5, .val = v } } });
    try blk.append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = sv } }, .type = .i32 });
    try blk.append(.{ .op = .{ .local_get = 5 }, .dest = g, .type = .i32 });
    try blk.append(.{ .op = .{ .add = .{ .lhs = g, .rhs = g } }, .dest = r, .type = .i32 });

    const changed = try forwardRedundantLoads(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(v, blk.instructions.items[2].op.add.lhs);
}
