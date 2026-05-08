const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");

// Cross-block/dominator-aware load forwarder
pub fn forwardRedundantLoadsDominator(func: *ir.IrFunction, allocator: std.mem.Allocator) !bool {
    if (func.blocks.items.len == 0) return false;

    var dom = try analysis.computeDominators(func, allocator);
    defer dom.deinit();

    const nblocks = func.blocks.items.len;
    var children = try allocator.alloc(std.ArrayList(ir.BlockId), nblocks);
    defer {
        for (children) |*list| list.deinit(allocator);
        allocator.free(children);
    }
    for (children) |*list| list.* = .empty;
    for (0..nblocks) |i| {
        const bid: ir.BlockId = @intCast(i);
        const idom = dom.idom[bid] orelse continue;
        if (idom == bid) continue;
        try children[idom].append(allocator, bid);
    }

    // Redundant load table: stack of maps, one per dom level.
    const LoadKey = struct {
        base: ir.VReg,
        offset: u32,
        size: u8,
        sign_extend: bool,
    };
    const LoadEntry = struct { vreg: ir.VReg };

    var table_stack = std.ArrayList(std.AutoHashMap(LoadKey, ir.VReg)).init(allocator);
    defer {
        for (table_stack.items) |*t| t.deinit();
        table_stack.deinit();
    }

    const Frame = struct { bid: ir.BlockId, phase: u1, table_len: usize };
    var dfs_stack = std.ArrayList(Frame).init(allocator);
    defer dfs_stack.deinit();

    var changed = false;

    if (dom.idom[0] == null) return false;
    try dfs_stack.append(.{ .bid = 0, .phase = 0, .table_len = 0 });
    try table_stack.append(.{ }); table_stack.items[0] = std.AutoHashMap(LoadKey, ir.VReg).init(allocator);

    while (dfs_stack.items.len > 0) {
        const top = &dfs_stack.items[dfs_stack.items.len - 1];
        if (top.phase == 1) {
            table_stack.items[table_stack.items.len - 1].deinit();
            _ = table_stack.pop();
            _ = dfs_stack.pop();
            continue;
        }
        const bid = top.bid;
        top.phase = 1;
        top.table_len = table_stack.items.len;
        // Push new dominator scope for this block
        var scope_map = std.AutoHashMap(LoadKey, ir.VReg).init(allocator);
        // Populate scope with previous values (shallow copy for lookup walk)
        if (table_stack.items.len > 0) {
            var prev = &table_stack.items[table_stack.items.len - 1];
            var it = prev.iterator();
            while (it.next()) |entry| {
                try scope_map.put(entry.key_ptr.*, entry.value_ptr.*);
            }
        }
        try table_stack.append(scope_map);
        var table = &table_stack.items[table_stack.items.len - 1];

        const block = &func.blocks.items[bid];
        // Scan this block's instructions
        for (block.instructions.items) |*inst| {
            switch (inst.op) {
                .load => |ld| {
                    const key = LoadKey {
                        .base = ld.base,
                        .offset = ld.offset,
                        .size = ld.size,
                        .sign_extend = ld.sign_extend,
                    };
                    var found = false;
                    // Walk table_stack from top to bottom (most local to least)
                    var t = table_stack.items.len;
                    while (t > 0) : (t -= 1) {
                        const tab = &table_stack.items[t - 1];
                        if (tab.get(key)) |held_vreg| {
                            // Replace this load with move/add from held_vreg
                            inst.op = .{ .add = .{ .lhs = held_vreg, .rhs = 0 } };
                            changed = true;
                            found = true;
                            break;
                        }
                    }
                    if (!found, inst.dest) |dest| {
                        try table.put(key, dest);
                    }
                },
                .store => |st| {
                    // Invalidate aliases in the current table only.
                    var remove_keys = std.ArrayList(LoadKey).init(allocator);
                    defer remove_keys.deinit();
                    var it = table.iterator();
                    while (it.next()) |entry| {
                        const key = entry.key_ptr.*;
                        const soff = ld_offset_(key);
                        const eoff = ld_offset_(st);
                        if (key.base == st.base and !(key.offset + key.size <= st.offset or st.offset + st.size <= key.offset)) {
                            remove_keys.append(key) catch unreachable;
                        }
                    }
                    for (remove_keys.items) |k| _ = table.remove(k);
                },
                else => {},
            }
        }

        // DFS push children
        for (children[bid].items) |child| {
            try dfs_stack.append(.{ .bid = child, .phase = 0, .table_len = 0 });
        }
    }

    return changed;
}

fn ld_offset_(k: anytype) u32 {
    return @field(k, "offset");
}
