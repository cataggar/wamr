const std = @import("std");
const ir = @import("ir.zig");

/// forwards redundant loads within a basic block
pub fn forwardRedundantLoads(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    var changed = false;
    const LoadKey = struct {
        base: ir.VReg,
        offset: u32,
        size: u8,
        sign_extend: bool,
    };
    var value_table = std.AutoHashMap(LoadKey, ir.VReg).init(allocator);
    defer value_table.deinit();

    // for each block
    for (func.blocks.items) |*block| {
        value_table.clearRetainingCapacity();
        // scan each instruction
        var i: usize = 0;
        while (i < block.instructions.items.len) : (i += 1) {
            const inst = &block.instructions.items[i];
            switch (inst.op) {
                .load => |ld| {
                    const key = LoadKey{
                        .base = ld.base,
                        .offset = ld.offset,
                        .size = ld.size,
                        .sign_extend = ld.sign_extend,
                    };
                    if (value_table.get(key)) |held_vreg| {
                        // replace this load with a simple move from held_vreg
                        inst.op = .{ .iconst_32 = 0 }; // dummy, prevents double-free
                        inst.* = ir.Inst{
                            .op = .{ .add = .{ .lhs = held_vreg, .rhs = 0 } },
                            .dest = inst.dest,
                            .type = inst.type,
                        };
                        changed = true;
                    } else if (inst.dest) |dest| {
                        try value_table.put(key, dest);
                    }
                },
                .store => |st| {
                    // Invalidate table entries that may alias this store
                    
                    // Zig 0.10+ doesn't allow mutation+remove during iteration.
// Workaround: collect keys to remove, then remove them.
var remove_keys = std.ArrayList(LoadKey).initCapacity(allocator, 4) catch unreachable;

                    // remove unused var it from store branch
defer remove_keys.deinit(allocator);
var entry_it = value_table.iterator();
while (entry_it.next()) |entry| {
    const key = entry.key_ptr.*;
    if (key.base == st.base and !(key.offset + key.size <= st.offset or st.offset + st.size <= key.offset)) {
        remove_keys.append(allocator, key) catch unreachable;
    }
}
for (remove_keys.items) |key| {
    _ = value_table.remove(key);
}

                },
                else => {},
            }
        }
    }
    return changed;
}
