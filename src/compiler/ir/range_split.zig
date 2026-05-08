//! Loop-aware live range splitting for register allocation.
//! This pass splits vregs' live ranges at loop boundaries where the value is unused within the loop.

const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");

const LiveRange = analysis.LiveRange;

pub fn splitLiveRangesAtLoopBoundaries(
    func: *const ir.IrFunction,
    dom: *const analysis.DomTree,
    allocator: std.mem.Allocator,
    input: []const LiveRange,
) ![]LiveRange {
    // Plan:
    // 1. Compute the loop forest.
    // 2. For each loop, identify vregs live across (not used in) the loop.
    // 3. For each such vreg, split its range and insert new intervals.
    // 4. Return the new interval list.

    const loop_forest = try analysis.computeLoops(func, dom, allocator);
    defer loop_forest.deinit();

    // TODO: Full logic — for now, return input unmodified.
    return try std.heap.page_allocator.dupe(LiveRange, input);
}
