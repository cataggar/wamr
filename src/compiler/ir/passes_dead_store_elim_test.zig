const std = @import("std");
const ir = @import("ir.zig");
const deadStoreElimination = @import("dead_store_elimination.zig").deadStoreElimination;

test "deadStoreElimination: same-key store followed by store removes earlier" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const block = func.getBlock(b0);

    const v_base = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 11 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 22 }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v1 } } });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v2 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const before = block.instructions.items.len;
    const changed = try deadStoreElimination(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(before - 1, block.instructions.items.len);
}

test "deadStoreElimination: store followed by load preserves the store" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const block = func.getBlock(b0);

    const v_base = func.newVReg();
    const v_val = func.newVReg();
    const v_loaded = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_val, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v_val } } });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try block.append(.{ .op = .{ .ret = v_loaded } });

    const before = block.instructions.items.len;
    const changed = try deadStoreElimination(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before, block.instructions.items.len);
}

test "deadStoreElimination: trailing store at end of block is preserved" {
    // The last store could feed a load in a successor block, so it must
    // never be eliminated even if no later instruction in the same block
    // reads it. Regression for the original aggressive pass that
    // eliminated trailing stores.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const block = func.getBlock(b0);

    const v_base = func.newVReg();
    const v_val = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 99 }, .dest = v_val, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v_val } } });
    try block.append(.{ .op = .{ .ret = null } });

    const before = block.instructions.items.len;
    const changed = try deadStoreElimination(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before, block.instructions.items.len);
}

test "deadStoreElimination: load between two same-key stores preserves both" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const block = func.getBlock(b0);

    const v_base = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v_loaded = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v1 } } });
    try block.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_loaded, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v2 } } });
    try block.append(.{ .op = .{ .ret = v_loaded } });

    const before = block.instructions.items.len;
    const changed = try deadStoreElimination(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before, block.instructions.items.len);
}

test "deadStoreElimination: call between two same-key stores preserves both" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const block = func.getBlock(b0);

    const v_base = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v1 } } });
    try block.append(.{ .op = .{ .call = .{ .func_idx = 0 } } });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v2 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const before = block.instructions.items.len;
    const changed = try deadStoreElimination(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before, block.instructions.items.len);
}

test "deadStoreElimination: stores at different offsets are unrelated" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const block = func.getBlock(b0);

    const v_base = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v1 } } });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 4, .size = 4, .val = v2 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const before = block.instructions.items.len;
    const changed = try deadStoreElimination(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before, block.instructions.items.len);
}

test "deadStoreElimination: chain of three same-key stores collapses to one" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const block = func.getBlock(b0);

    const v_base = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const v3 = func.newVReg();
    try block.append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 2 }, .dest = v2, .type = .i32 });
    try block.append(.{ .op = .{ .iconst_32 = 3 }, .dest = v3, .type = .i32 });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v1 } } });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v2 } } });
    try block.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v3 } } });
    try block.append(.{ .op = .{ .ret = null } });

    const before = block.instructions.items.len;
    const changed = try deadStoreElimination(&func, allocator);
    try std.testing.expect(changed);
    try std.testing.expectEqual(before - 2, block.instructions.items.len);
}

test "deadStoreElimination: shadowing does not cross block boundaries" {
    // A store at the end of block 0 must be preserved even if block 1
    // begins with a same-key store, because the pass treats blocks as
    // independent.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();

    const v_base = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = v_base, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v1 } } });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 2 }, .dest = v2, .type = .i32 });
    try func.getBlock(b1).append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });

    const before0 = func.getBlock(b0).instructions.items.len;
    const before1 = func.getBlock(b1).instructions.items.len;
    const changed = try deadStoreElimination(&func, allocator);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before0, func.getBlock(b0).instructions.items.len);
    try std.testing.expectEqual(before1, func.getBlock(b1).instructions.items.len);
}
