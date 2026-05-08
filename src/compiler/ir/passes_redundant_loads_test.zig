const std = @import("std");
const ir = @import("ir.zig");
const forwardRedundantLoads = @import("forward_redundant_loads.zig").forwardRedundantLoads;

// Test: two i32.load [base+0] separated by unrelated arithmetic: second load should be replaced by mov
// Also, test intervening store invalidates forwarding

pub fn main() void {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 2, 0);
    defer func.deinit();
    const block = try func.newBlock();
    var blk = &func.blocks.items[block];
    const v_base = func.newVReg();
    const v_load1 = func.newVReg();
    const v_tmp = func.newVReg();
    const v_load2 = func.newVReg();
    const v_val = func.newVReg();
    // Load 1
    try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_load1, .type = .i32 });
    // Unrelated op
    try blk.append(.{ .op = .{ .add = .{ .lhs = v_load1, .rhs = 42 } }, .dest = v_tmp, .type = .i32 });
    // Load 2 (redundant)
    try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_load2, .type = .i32 });

    const changed = try forwardRedundantLoads(&func, allocator);
    std.testing.expect(changed) catch @panic("pass didn't fire");

    // Expect second load to be replaced by move/add from v_load1
    const last = blk.instructions.items[2];
    std.testing.expect(last.op == .add) catch @panic("not replaced with mov/add");
    std.testing.expect(last.op.add.lhs == v_load1) catch @panic("not forwarding vreg");

    // Store-alias invalidation test
    const v_store_val = func.newVReg();
    try blk.append(.{ .op = .{ .store = .{ .base = v_base, .offset = 0, .size = 4, .val = v_store_val } }, .type = .i32 });
    const v_load3 = func.newVReg();
    try blk.append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_load3, .type = .i32 });
    const changed2 = try forwardRedundantLoads(&func, allocator);
    std.testing.expect(!changed2) catch @panic("table not invalidated on store");
}
