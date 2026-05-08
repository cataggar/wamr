const std = @import("std");
const ir = @import("ir.zig");

test {
    _ = @import("passes_dead_store_elim_test.zig");
}

/// Dead Store Elimination — block-local, alias-conservative.
///
/// Removes a store iff a strictly later store within the same basic block
/// writes the same `(base, offset, size)` location, with NO intervening
/// instruction that could observe or alias linear memory:
///
///   * any load (any width) — without alias info, every load may read any
///     location
///   * atomic memory ops, fences, notifies, waits
///   * bulk memory ops (memory.copy / memory.fill / memory.init), memory.grow
///   * calls — opaque side effects
///
/// SSA gives free aliasing for stores that share the same base VReg: the
/// VReg holds a single value, so identical `(base, offset, size)` keys
/// always denote the same address. Two stores with different base VRegs
/// may still alias at run-time, but that does not affect correctness here:
/// when the later same-key store fully overwrites the location, whatever
/// any intermediate store wrote there is irrelevant — the location's
/// final value is determined by the later store. Loads and barriers in
/// between are what matter, and those clear the shadow set.
///
/// The store at the end of a block (or any store followed only by ops
/// that clear the shadow set) is preserved, since successor blocks may
/// read it.
pub fn deadStoreElimination(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    const Key = struct {
        base: ir.VReg,
        offset: u32,
        size: u8,
    };

    var shadowed = std.AutoHashMap(Key, void).init(allocator);
    defer shadowed.deinit();

    var changed = false;

    for (func.blocks.items) |*block| {
        shadowed.clearRetainingCapacity();

        var i: usize = block.instructions.items.len;
        while (i > 0) {
            i -= 1;
            const inst = &block.instructions.items[i];
            switch (inst.op) {
                .store => |st| {
                    const key: Key = .{ .base = st.base, .offset = st.offset, .size = st.size };
                    if (shadowed.contains(key)) {
                        _ = block.instructions.orderedRemove(i);
                        changed = true;
                    } else {
                        try shadowed.put(key, {});
                    }
                },
                .load,
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
                .call,
                .call_indirect,
                .call_ref,
                => {
                    shadowed.clearRetainingCapacity();
                },
                else => {},
            }
        }
    }
    return changed;
}
