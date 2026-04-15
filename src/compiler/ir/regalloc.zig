//! Linear Scan Register Allocator
//!
//! Assigns physical x86-64 registers to VRegs based on live range intervals.
//! Uses the Poletto & Sarkar algorithm: sort intervals by start, walk in order,
//! assign from free pool, spill the longest-remaining interval when exhausted.

const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");

/// Physical register identifier (0-15 for x86-64 GPRs).
pub const PhysReg = u4;

/// Physical register or stack slot assignment.
pub const Allocation = union(enum) {
    reg: PhysReg,
    stack: i32, // offset from RBP
};

/// Result of register allocation for one function.
pub const AllocResult = struct {
    /// VReg → physical location mapping.
    assignments: std.AutoHashMap(ir.VReg, Allocation),
    /// Number of spill slots used.
    spill_count: u32,

    pub fn deinit(self: *AllocResult) void {
        self.assignments.deinit();
    }

    pub fn get(self: *const AllocResult, vreg: ir.VReg) ?Allocation {
        return self.assignments.get(vreg);
    }
};

/// Allocatable registers as PhysReg IDs.
/// Excludes RSP (4), RBP (5), and scratch regs R10 (10), R11 (11).
const alloc_regs = [_]PhysReg{ 0, 1, 2, 6, 7, 8, 9 };
// rax=0, rcx=1, rdx=2, rsi=6, rdi=7, r8=8, r9=9

/// Scratch registers for spill loads (not allocatable).
pub const scratch1: PhysReg = 10; // r10
pub const scratch2: PhysReg = 11; // r11

/// Run linear scan register allocation on a function.
pub fn allocate(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !AllocResult {
    // Compute live ranges (sorted by start position)
    const ranges = try analysis.computeLiveRanges(func, allocator);
    defer allocator.free(ranges);

    var assignments = std.AutoHashMap(ir.VReg, Allocation).init(allocator);

    // Track which registers are free
    var reg_free = [_]bool{true} ** alloc_regs.len;
    // Active intervals (currently assigned to a register), sorted by end position
    var active: std.ArrayList(ActiveInterval) = .empty;
    defer active.deinit(allocator);

    var spill_count: u32 = 0;

    for (ranges) |range| {
        // Expire old intervals that ended before this one starts
        expireOldIntervals(&active, range.start, &reg_free);

        // Try to find a free register
        if (findFreeReg(&reg_free)) |reg_idx| {
            reg_free[reg_idx] = false;
            try assignments.put(range.vreg, .{ .reg = alloc_regs[reg_idx] });
            try insertActive(&active, allocator, .{
                .vreg = range.vreg,
                .end = range.end,
                .reg_idx = reg_idx,
            });
        } else {
            // No free register — spill the interval with the longest remaining range
            if (active.items.len > 0) {
                const last = active.items.len - 1;
                const spill_candidate = active.items[last];
                if (spill_candidate.end > range.end) {
                    // Spill the active interval, assign its register to the new one
                    const stolen_reg = spill_candidate.reg_idx;
                    // Move spilled VReg to stack
                    const spill_offset = computeSpillOffset(spill_count);
                    try assignments.put(spill_candidate.vreg, .{ .stack = spill_offset });
                    spill_count += 1;
                    // Remove spilled from active
                    _ = active.orderedRemove(last);
                    // Assign stolen register to new interval
                    try assignments.put(range.vreg, .{ .reg = alloc_regs[stolen_reg] });
                    try insertActive(&active, allocator, .{
                        .vreg = range.vreg,
                        .end = range.end,
                        .reg_idx = stolen_reg,
                    });
                } else {
                    // New interval is shorter — spill the new one
                    const spill_offset = computeSpillOffset(spill_count);
                    try assignments.put(range.vreg, .{ .stack = spill_offset });
                    spill_count += 1;
                }
            } else {
                // No active intervals — spill the new one
                const spill_offset = computeSpillOffset(spill_count);
                try assignments.put(range.vreg, .{ .stack = spill_offset });
                spill_count += 1;
            }
        }
    }

    return .{
        .assignments = assignments,
        .spill_count = spill_count,
    };
}

const ActiveInterval = struct {
    vreg: ir.VReg,
    end: u32,
    reg_idx: usize,
};

/// Remove intervals from `active` whose end position is <= `pos`.
fn expireOldIntervals(
    active: *std.ArrayList(ActiveInterval),
    pos: u32,
    reg_free: *[alloc_regs.len]bool,
) void {
    // Active is sorted by end position; remove from front
    while (active.items.len > 0 and active.items[0].end < pos) {
        const expired = active.orderedRemove(0);
        reg_free[expired.reg_idx] = true;
    }
}

/// Find the first free register, or null if all are occupied.
fn findFreeReg(reg_free: *const [alloc_regs.len]bool) ?usize {
    for (reg_free, 0..) |free, i| {
        if (free) return i;
    }
    return null;
}

/// Insert into active list maintaining sorted order by end position.
fn insertActive(
    active: *std.ArrayList(ActiveInterval),
    allocator: std.mem.Allocator,
    interval: ActiveInterval,
) !void {
    // Find insertion point (keep sorted by end)
    var pos: usize = 0;
    while (pos < active.items.len and active.items[pos].end <= interval.end) {
        pos += 1;
    }
    try active.insert(allocator, pos, interval);
}

/// Compute spill slot offset from RBP.
/// Spill slots start after locals and operand stack area.
/// `local_count` is needed to position spills after the local variable area.
fn computeSpillOffset(spill_idx: u32) i32 {
    // Spill area starts at [rbp - 600] to avoid conflicts with locals and operand stack.
    // Each spill slot is 8 bytes.
    const spill_base: i32 = -600;
    return spill_base - @as(i32, @intCast(spill_idx * 8));
}

// ── Tests ───────────────────────────────────────────────────────────────

test "allocate: simple function gets registers" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 });
    try block0.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 });
    try block0.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 });
    try block0.append(.{ .op = .{ .ret = v2 } });

    var result = try allocate(&func, allocator);
    defer result.deinit();

    // All 3 VRegs should get registers (only 3 needed, 9 available)
    try std.testing.expect(result.get(v0) != null);
    try std.testing.expect(result.get(v1) != null);
    try std.testing.expect(result.get(v2) != null);
    try std.testing.expectEqual(@as(u32, 0), result.spill_count);

    // All should be in registers
    try std.testing.expect(result.get(v0).? == .reg);
    try std.testing.expect(result.get(v1).? == .reg);
    try std.testing.expect(result.get(v2).? == .reg);
}

test "allocate: no spills with few live values" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);

    // Create a chain of operations: each value used once then dead
    var prev = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 1 }, .dest = prev });
    for (0..8) |_| {
        const next_val = func.newVReg();
        const imm = func.newVReg();
        try block0.append(.{ .op = .{ .iconst_32 = 1 }, .dest = imm });
        try block0.append(.{ .op = .{ .add = .{ .lhs = prev, .rhs = imm } }, .dest = next_val });
        prev = next_val;
    }
    try block0.append(.{ .op = .{ .ret = prev } });

    var result = try allocate(&func, allocator);
    defer result.deinit();

    // Low register pressure — no spills expected
    try std.testing.expectEqual(@as(u32, 0), result.spill_count);
}

test "allocate: spills when pressure exceeds registers" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);

    // Create 15 values all live simultaneously (more than 9 allocatable regs)
    var vregs: [15]ir.VReg = undefined;
    for (0..15) |i| {
        vregs[i] = func.newVReg();
        try block0.append(.{ .op = .{ .iconst_32 = @intCast(i) }, .dest = vregs[i] });
    }
    // Use all of them in pairs to keep them live
    var sum = vregs[0];
    for (1..15) |i| {
        const next = func.newVReg();
        try block0.append(.{ .op = .{ .add = .{ .lhs = sum, .rhs = vregs[i] } }, .dest = next });
        sum = next;
    }
    try block0.append(.{ .op = .{ .ret = sum } });

    var result = try allocate(&func, allocator);
    defer result.deinit();

    // Should have some spills (15 values alive > 9 registers)
    try std.testing.expect(result.spill_count > 0);

    // All VRegs should still have an allocation (reg or stack)
    for (vregs) |v| {
        try std.testing.expect(result.get(v) != null);
    }
}
