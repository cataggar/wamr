const std = @import("std");
const ir = @import("ir.zig");
const forwardRedundantLoadsDominator = @import("forward_redundant_loads_dominator.zig").forwardRedundantLoadsDominator;

// Test: cross-block (diamond) load forwarding and CSE through diamond
pub fn main() void {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 3, 0);
    defer func.deinit();

    // Block structure: entry -> b1, b2; both -> merge
    const entry = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const merge = try func.newBlock();
    var blocks = &func.blocks.items;
    // Connect CFG
    blocks[entry].successors = &[_]ir.BlockId{b1, b2};
    blocks[b1].successors = &[_]ir.BlockId{merge};
    blocks[b2].successors = &[_]ir.BlockId{merge};

    // VRegs
    const v_base = func.newVReg();
    const v1 = func.newVReg(); // load in b1
    const v2 = func.newVReg(); // load in b2
    const v_merge = func.newVReg(); // load in merge

    // entry: nothing
    // b1: load
    try blocks[b1].append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v1, .type = .i32 });
    // b2: load
    try blocks[b2].append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v2, .type = .i32 });
    // merge: load again (should be forwarded if no aliasing store)
    try blocks[merge].append(.{ .op = .{ .load = .{ .base = v_base, .offset = 0, .size = 4 } }, .dest = v_merge, .type = .i32 });

    const changed = try forwardRedundantLoadsDominator(&func, allocator);
    std.testing.expect(changed) catch @panic("forwarding did not fire");
    // v_merge load should be replaced with earlier (any dominates through diamond)
    const last = blocks[merge].instructions.items[0];
    std.testing.expect(last.op == .add) catch @panic("not replaced with mov/add");
}
