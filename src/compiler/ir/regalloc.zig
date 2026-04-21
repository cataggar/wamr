//! Linear Scan Register Allocator
//!
//! Assigns physical x86-64 registers to VRegs based on live range intervals.
//! Uses the Poletto & Sarkar algorithm: sort intervals by start, walk in order,
//! assign from free pool, spill the longest-remaining interval when exhausted.
//!
//! Clobber-aware: instructions that destroy register contents (calls,
//! memory_copy, etc.) are modeled as ClobberPoints. The allocator ensures
//! no VReg assigned to a clobbered register has its live range span the
//! clobber, eliminating the need for push/pop at those sites.

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
/// Excludes RAX (0), RCX (1) — used as scratch temporaries by codegen,
/// RSP (4), RBP (5) — frame pointers, and R10 (10), R11 (11) — scratch regs.
/// Callee-saved regs (rbx, r12-r15) are preserved in prologue/epilogue when used.
const alloc_regs = [_]PhysReg{ 2, 3, 6, 7, 8, 9, 12, 13, 14, 15 };
// rdx=2, rbx=3, rsi=6, rdi=7, r8=8, r9=9, r12=12, r13=13, r14=14, r15=15

/// Scratch registers for spill loads (not allocatable).
pub const scratch1: PhysReg = 10; // r10
pub const scratch2: PhysReg = 11; // r11

/// A point in the instruction stream where specific registers are destroyed.
/// Used to model calls (clobber caller-saved), memory_copy (clobber rsi+rdi), etc.
pub const ClobberPoint = struct {
    pos: u32,
    /// Which alloc_regs indices are clobbered at this position.
    regs_clobbered: [alloc_regs.len]bool,
};

/// Run linear scan register allocation on a function.
/// `clobbers` lists positions where specific registers are destroyed.
pub fn allocate(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
    clobbers: []const ClobberPoint,
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

    // Spill area must start AFTER the operand-stack area in the frame.
    // Frame layout (compile.zig): [rbp-8]=VmCtx, [rbp-16..-(1+LC)*8]=locals (LC slots),
    // [-(2+LC)*8..-(65+LC)*8]=op-stack (64 slots). Spills begin at -(66+LC)*8.
    const spill_base: i32 = -@as(i32, @intCast((func.local_count + 66) * 8));

    for (ranges) |range| {
        // Expire old intervals that ended before this one starts
        expireOldIntervals(&active, range.start, &reg_free);

        // Try to find a free register that is safe (not clobbered during this range)
        if (findSafeReg(&reg_free, range.start, range.end, clobbers)) |reg_idx| {
            reg_free[reg_idx] = false;
            try assignments.put(range.vreg, .{ .reg = alloc_regs[reg_idx] });
            try insertActive(&active, allocator, .{
                .vreg = range.vreg,
                .end = range.end,
                .reg_idx = reg_idx,
            });
        } else {
            // No safe free register — try to evict an active interval
            // whose register IS safe for this range.
            var best_evict: ?usize = null;
            for (active.items, 0..) |ai, idx| {
                if (ai.end > range.end and
                    regSafeForRange(ai.reg_idx, range.start, range.end, clobbers))
                {
                    if (best_evict == null or ai.end > active.items[best_evict.?].end) {
                        best_evict = idx;
                    }
                }
            }

            if (best_evict) |evict_idx| {
                const evicted = active.orderedRemove(evict_idx);
                const stolen_reg = evicted.reg_idx;
                const spill_offset = spill_base - @as(i32, @intCast(spill_count * 8));
                try assignments.put(evicted.vreg, .{ .stack = spill_offset });
                spill_count += 1;
                try assignments.put(range.vreg, .{ .reg = alloc_regs[stolen_reg] });
                try insertActive(&active, allocator, .{
                    .vreg = range.vreg,
                    .end = range.end,
                    .reg_idx = stolen_reg,
                });
            } else {
                // No safe eviction candidate — spill the new interval
                const spill_offset = spill_base - @as(i32, @intCast(spill_count * 8));
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

/// Check if register at `reg_idx` is safe for a live range [start, end].
/// A register is unsafe if it's clobbered at any point strictly inside the range.
fn regSafeForRange(reg_idx: usize, start: u32, end: u32, clobbers: []const ClobberPoint) bool {
    for (clobbers) |cp| {
        if (cp.pos > start and cp.pos < end and cp.regs_clobbered[reg_idx]) return false;
    }
    return true;
}

/// Find a free register that is not clobbered during [start, end].
fn findSafeReg(
    reg_free: *const [alloc_regs.len]bool,
    start: u32,
    end: u32,
    clobbers: []const ClobberPoint,
) ?usize {
    for (reg_free, 0..) |free, i| {
        if (free and regSafeForRange(i, start, end, clobbers)) return i;
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

/// Compute spill slot offset from RBP (legacy helper; callers now use
/// `spill_base` computed per-function in `allocate`). Kept for potential
/// unit-test use with the default operand-stack budget of 64 slots.
fn computeSpillOffset(spill_idx: u32) i32 {
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

    var result = try allocate(&func, allocator, &.{});
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

    var result = try allocate(&func, allocator, &.{});
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

    var result = try allocate(&func, allocator, &.{});
    defer result.deinit();

    // Should have some spills (15 values alive > 9 registers)
    try std.testing.expect(result.spill_count > 0);

    // All VRegs should still have an allocation (reg or stack)
    for (vregs) |v| {
        try std.testing.expect(result.get(v) != null);
    }
}
