const std = @import("std");
const ir = @import("ir.zig");
const passes = @import("passes.zig");
const alias_class = @import("alias_class.zig");

const LoadKey = alias_class.MemKey;

/// Forward redundant loads within a basic block: when a load at the
/// same `(base, offset, size, sign_extend)` location was already
/// performed earlier in the block with no intervening aliasing store,
/// barrier, or call, replace later loads' result vregs with the
/// earlier load's dest and remove the redundant load instructions.
///
/// Removal is safe because the prior load already performed (and
/// either succeeded at, or trapped during) the bounds check; if we
/// reach the redundant load at run-time, the same access would be
/// in-bounds again. No intervening op modifies linear memory at the
/// stored location, so the value is unchanged.
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
                    const key = alias_class.memKeyFromLoad(ld);
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
                        if (alias_class.storeAliases(entry.key_ptr.*, st)) {
                            try aliasing_keys.append(allocator, entry.key_ptr.*);
                        }
                    }
                    for (aliasing_keys.items) |k| _ = value_table.remove(k);
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
