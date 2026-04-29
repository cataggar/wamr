//! Linear Scan Register Allocator
//!
//! Assigns physical registers to VRegs based on live range intervals.
//! Uses the Poletto & Sarkar algorithm: sort intervals by start, walk in order,
//! assign from free pool, spill the longest-remaining interval when exhausted.
//!
//! Clobber-aware: instructions that destroy register contents (calls,
//! memory_copy, etc.) are modeled as ClobberPoints. The allocator ensures
//! no VReg assigned to a clobbered register has its live range span the
//! clobber, eliminating the need for push/pop at those sites.
//!
//! Architecture-agnostic: the caller passes a `RegSet` describing the
//! allocatable registers, their caller/callee-saved partition, and the
//! spill-slot layout. x86-64 and aarch64 share this implementation.

const std = @import("std");
const ir = @import("ir.zig");
const analysis = @import("analysis.zig");

/// Physical register identifier. Widest architecture we target is aarch64
/// with 0..30 (v0..v31 is separate). `u8` leaves headroom.
pub const PhysReg = u8;

/// Maximum number of allocatable registers supported by the bitmasks below.
/// aarch64's allocatable GPR pool is 25; x86-64's is 10. 64 is ample.
pub const max_alloc_regs: usize = 64;

/// Physical register or stack slot assignment.
pub const Allocation = union(enum) {
    reg: PhysReg,
    /// Byte offset of the first spill byte from the frame pointer. Sign and
    /// stride come from `RegSet.spill_base`/`spill_stride`.
    stack: i32,
};

/// Describes the architecture's register file and spill-slot layout.
/// Caller constructs this per-function (spill_base may depend on the
/// locals area size, for example).
pub const RegSet = struct {
    /// Allocatable physical register numbers, in preference order within
    /// the caller- and callee-saved partitions. Length must be ≤
    /// `max_alloc_regs`.
    alloc_regs: []const PhysReg,
    /// Indices into `alloc_regs` of registers that survive a call
    /// without save/restore. Prefer these for live ranges that span a
    /// clobber point.
    callee_saved_indices: []const u8,
    /// Indices into `alloc_regs` of registers that do NOT survive a
    /// call. Prefer these for short-lived values to avoid the cost of
    /// preserving callee-saved regs in the prologue.
    caller_saved_indices: []const u8,
    /// Byte offset (from the frame pointer) of the first spill slot.
    spill_base: i32,
    /// Byte stride from one spill slot to the next. Negative on
    /// downward-growing frames (x86-64: -8), positive on upward
    /// (aarch64: +8).
    spill_stride: i32,
};

/// Result of register allocation for one function.
pub const AllocResult = struct {
    /// VReg → physical location mapping.
    assignments: std.AutoHashMap(ir.VReg, Allocation),
    /// Number of 8-byte spill slots used. v128 values consume two slots and
    /// are aligned to a 16-byte FP-relative offset.
    spill_count: u32,

    pub fn deinit(self: *AllocResult) void {
        self.assignments.deinit();
    }

    pub fn get(self: *const AllocResult, vreg: ir.VReg) ?Allocation {
        return self.assignments.get(vreg);
    }
};

/// A point in the instruction stream where specific registers are destroyed.
/// Used to model calls (clobber caller-saved), memory_copy (clobber rsi+rdi), etc.
///
/// `regs_clobbered` is a bitmask over the caller's `RegSet.alloc_regs`:
/// bit i set means `alloc_regs[i]` is destroyed at this position.
pub const ClobberPoint = struct {
    pos: u32,
    regs_clobbered: u64,
};

/// Run linear scan register allocation on a function.
/// `clobbers` lists positions where specific registers are destroyed.
pub fn allocate(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
    reg_set: RegSet,
    clobbers: []const ClobberPoint,
) !AllocResult {
    const ranges = try analysis.computeLiveRanges(func, allocator);
    defer allocator.free(ranges);
    return allocateFromRanges(allocator, reg_set, clobbers, ranges);
}

/// Variant of `allocate` that takes pre-computed live ranges. Used by
/// aarch64 to inject FMA-fusion awareness: the codegen's MADD/MSUB pre-pass
/// reads a fused mul's sources at the following add instruction, so those
/// vregs' live ranges must be extended past the mul before allocation.
/// `ranges` must be sorted by `.start` (as returned by `computeLiveRanges`).
pub fn allocateFromRanges(
    allocator: std.mem.Allocator,
    reg_set: RegSet,
    clobbers: []const ClobberPoint,
    ranges: []const analysis.LiveRange,
) !AllocResult {
    std.debug.assert(reg_set.alloc_regs.len <= max_alloc_regs);

    var assignments = std.AutoHashMap(ir.VReg, Allocation).init(allocator);

    // Track which register indices are free (bit i ↔ alloc_regs[i]).
    // Start with the low `alloc_regs.len` bits set.
    var reg_free: u64 = if (reg_set.alloc_regs.len == 64)
        std.math.maxInt(u64)
    else
        (@as(u64, 1) << @intCast(reg_set.alloc_regs.len)) - 1;

    // Active intervals (currently assigned to a register), sorted by end position
    var active: std.ArrayList(ActiveInterval) = .empty;
    defer active.deinit(allocator);

    var spill_slots_used: u32 = 0;

    for (ranges) |range| {
        // Expire old intervals that ended before this one starts
        expireOldIntervals(&active, range.start, &reg_free);

        // Try to find a free register that is safe (not clobbered during this range)
        if (findSafeReg(reg_set, reg_free, range.start, range.end, clobbers)) |reg_idx| {
            reg_free &= ~(@as(u64, 1) << @intCast(reg_idx));
            try assignments.put(range.vreg, .{ .reg = reg_set.alloc_regs[reg_idx] });
            try insertActive(&active, allocator, .{
                .vreg = range.vreg,
                .end = range.end,
                .reg_idx = reg_idx,
                .type = range.type,
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
                const spill_offset = allocateSpill(&spill_slots_used, reg_set, evicted.type);
                try assignments.put(evicted.vreg, .{ .stack = spill_offset });
                try assignments.put(range.vreg, .{ .reg = reg_set.alloc_regs[stolen_reg] });
                try insertActive(&active, allocator, .{
                    .vreg = range.vreg,
                    .end = range.end,
                    .reg_idx = stolen_reg,
                    .type = range.type,
                });
            } else {
                // No safe eviction candidate — spill the new interval
                const spill_offset = allocateSpill(&spill_slots_used, reg_set, range.type);
                try assignments.put(range.vreg, .{ .stack = spill_offset });
            }
        }
    }

    return .{
        .assignments = assignments,
        .spill_count = spill_slots_used,
    };
}

fn allocateSpill(spill_slots_used: *u32, reg_set: RegSet, ty: ir.IrType) i32 {
    const align_slots = @as(u32, ty.spillAlignSlots64());
    const needed_slots = @as(u32, ty.spillSlots64());
    while (!spillSlotAligned(reg_set, spill_slots_used.*, align_slots)) {
        spill_slots_used.* += 1;
    }
    const offset = reg_set.spill_base +
        @as(i32, @intCast(spill_slots_used.*)) * reg_set.spill_stride;
    spill_slots_used.* += needed_slots;
    return offset;
}

fn spillSlotAligned(reg_set: RegSet, slot_index: u32, align_slots: u32) bool {
    if (align_slots <= 1) return true;
    const offset = reg_set.spill_base +
        @as(i32, @intCast(slot_index)) * reg_set.spill_stride;
    const align_bytes = @as(i32, @intCast(align_slots * 8));
    const abs_offset = if (offset < 0) -offset else offset;
    return @mod(abs_offset, align_bytes) == 0;
}

const ActiveInterval = struct {
    vreg: ir.VReg,
    end: u32,
    reg_idx: u8,
    type: ir.IrType,
};

/// Remove intervals from `active` whose end position is <= `pos`.
fn expireOldIntervals(
    active: *std.ArrayList(ActiveInterval),
    pos: u32,
    reg_free: *u64,
) void {
    // Active is sorted by end position; remove from front
    while (active.items.len > 0 and active.items[0].end < pos) {
        const expired = active.orderedRemove(0);
        reg_free.* |= (@as(u64, 1) << @intCast(expired.reg_idx));
    }
}

/// Check if register at `reg_idx` is safe for a live range [start, end].
/// A register is unsafe if it's clobbered at any point strictly inside the range.
fn regSafeForRange(reg_idx: u8, start: u32, end: u32, clobbers: []const ClobberPoint) bool {
    const bit = @as(u64, 1) << @intCast(reg_idx);
    for (clobbers) |cp| {
        if (cp.pos > start and cp.pos < end and (cp.regs_clobbered & bit) != 0) return false;
    }
    return true;
}

/// Whether a vreg's live range spans any clobber point (e.g., a call).
/// If so, callee-saved registers are preferred to avoid save/restore.
fn spansClobber(start: u32, end: u32, clobbers: []const ClobberPoint) bool {
    for (clobbers) |cp| {
        if (cp.pos >= start and cp.pos < end) return true;
        if (cp.pos >= end) break; // clobbers are position-ordered
    }
    return false;
}

/// Find a free register, preferring callee-saved for long-lived values
/// (those spanning calls) and caller-saved for short-lived values.
fn findSafeReg(
    reg_set: RegSet,
    reg_free: u64,
    start: u32,
    end: u32,
    clobbers: []const ClobberPoint,
) ?u8 {
    const prefer_callee = spansClobber(start, end, clobbers);
    const first: []const u8 = if (prefer_callee) reg_set.callee_saved_indices else reg_set.caller_saved_indices;
    const second: []const u8 = if (prefer_callee) reg_set.caller_saved_indices else reg_set.callee_saved_indices;
    for (first) |i| {
        const bit = @as(u64, 1) << @intCast(i);
        if ((reg_free & bit) != 0 and regSafeForRange(i, start, end, clobbers)) return i;
    }
    for (second) |i| {
        const bit = @as(u64, 1) << @intCast(i);
        if ((reg_free & bit) != 0 and regSafeForRange(i, start, end, clobbers)) return i;
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

// ── Tests ───────────────────────────────────────────────────────────────

/// Register set used by the in-file tests. Mirrors the legacy x86-64
/// layout so the test expectations remain meaningful: 10 allocatable
/// GPRs, rbx+r12..r15 callee-saved, rdx+rsi+rdi+r8+r9 caller-saved,
/// spill area below rbp with 64-slot operand stack.
const test_reg_set: RegSet = .{
    .alloc_regs = &.{ 2, 3, 6, 7, 8, 9, 12, 13, 14, 15 },
    .callee_saved_indices = &.{ 1, 6, 7, 8, 9 },
    .caller_saved_indices = &.{ 0, 2, 3, 4, 5 },
    // func.local_count==1 for all tests: spill_base = -(1 + 66) * 8 = -536.
    .spill_base = -536,
    .spill_stride = -8,
};

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

    var result = try allocate(&func, allocator, test_reg_set, &.{});
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

    var result = try allocate(&func, allocator, test_reg_set, &.{});
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

    var result = try allocate(&func, allocator, test_reg_set, &.{});
    defer result.deinit();

    // Should have some spills (15 values alive > 9 registers)
    try std.testing.expect(result.spill_count > 0);

    // All VRegs should still have an allocation (reg or stack)
    for (vregs) |v| {
        try std.testing.expect(result.get(v) != null);
    }
}

test "allocateFromRanges: v128 spills consume two aligned slots" {
    const allocator = std.testing.allocator;
    const one_reg_set: RegSet = .{
        .alloc_regs = &.{0},
        .callee_saved_indices = &.{},
        .caller_saved_indices = &.{0},
        .spill_base = 8,
        .spill_stride = 8,
    };
    const ranges = [_]analysis.LiveRange{
        .{ .vreg = 0, .start = 0, .end = 10, .type = .v128 },
        .{ .vreg = 1, .start = 1, .end = 9, .type = .i64 },
        .{ .vreg = 2, .start = 2, .end = 8, .type = .v128 },
    };

    var result = try allocateFromRanges(allocator, one_reg_set, &.{}, &ranges);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 4), result.spill_count);
    try std.testing.expectEqual(Allocation{ .stack = 16 }, result.get(0).?);
    try std.testing.expectEqual(Allocation{ .stack = 32 }, result.get(1).?);
    try std.testing.expectEqual(Allocation{ .reg = 0 }, result.get(2).?);
}
