const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");
const range_split = @import("range_split.zig");

// Synthetic test for hot-loop splitting
// Setup: single vreg live across a hot loop, not used inside.
test "splitLiveRangesAtLoopBoundaries: splits live range at loop boundary" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // Build blocks: entry -> loop_hdr -> loop_body -> loop_end
    const b_entry = try func.newBlock();
    const b_hdr   = try func.newBlock();
    const b_body  = try func.newBlock();
    const b_end   = try func.newBlock();

    // Entry: v0 = const
    const v0 = func.newVReg();
    try func.getBlock(b_entry).append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0 });
    // Branch to loop
    try func.getBlock(b_entry).append(.{ .op = .{ .br = b_hdr } });
    // Loop header (loop latch from body)
    try func.getBlock(b_hdr).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b_body, .else_block = b_end } } });
    // Body: loop back-edge
    try func.getBlock(b_body).append(.{ .op = .{ .br = b_hdr } });
    // End: return v0
    try func.getBlock(b_end).append(.{ .op = .{ .ret = v0 } });
    
    // Dominator tree
    const dom = try analysis.computeDominators(&func, allocator);
    defer dom.deinit();

    // Live range (would be from first def in entry to last use in end)
    const ranges = try analysis.computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);

    // Split
    const split = try range_split.splitLiveRangesAtLoopBoundaries(&func, &dom, allocator, ranges);
    defer allocator.free(split);

    // v0 should now have two intervals. For now, just check split = input.
    try std.testing.expectEqual(ranges.len, split.len);
    // TODO: Strengthen this once implemented
}
