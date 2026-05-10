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
const range_split = @import("range_split.zig");

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

/// Hint that `vreg` should be allocated to a particular index in
/// `RegSet.alloc_regs` if at all possible. Hints are advisory: if the
/// hinted register is busy or unsafe (clobbered inside the range), the
/// allocator silently falls back to the normal caller/callee-saved
/// preference order.
///
/// Hints exist primarily to satisfy ABI constraints — e.g. an x86-64
/// vreg used as `args[i]` of a call site that is then placed into
/// `param_regs[i + 1]` benefits if the allocator put it there directly,
/// avoiding a `mov` at the call site.
pub const Hint = struct {
    vreg: ir.VReg,
    /// Index into `RegSet.alloc_regs` of the preferred register.
    reg_idx: u8,
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
    return allocateFromRangesWithHints(allocator, reg_set, clobbers, ranges, &.{});
}

/// Variant of `allocateFromRanges` that additionally accepts a list of
/// per-vreg register hints. See `Hint` for semantics.
///
/// Multiple hints for the same vreg are allowed; the first one wins.
/// Subsequent hints have no effect (they're not consulted as fallback).
pub fn allocateFromRangesWithHints(
    allocator: std.mem.Allocator,
    reg_set: RegSet,
    clobbers: []const ClobberPoint,
    ranges: []const analysis.LiveRange,
    hints: []const Hint,
) !AllocResult {
    std.debug.assert(reg_set.alloc_regs.len <= max_alloc_regs);

    var assignments = std.AutoHashMap(ir.VReg, Allocation).init(allocator);

    // Build a map of vreg → hinted alloc-regs index. First hint wins on
    // duplicates; subsequent ones are ignored.
    var hint_map = std.AutoHashMap(ir.VReg, u8).init(allocator);
    defer hint_map.deinit();
    try hint_map.ensureTotalCapacity(@intCast(hints.len));
    for (hints) |h| {
        const gop = hint_map.getOrPutAssumeCapacity(h.vreg);
        if (!gop.found_existing) gop.value_ptr.* = h.reg_idx;
    }

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

        const hint_idx: ?u8 = hint_map.get(range.vreg);

        // Try to find a free register that is safe (not clobbered during this range)
        if (findSafeReg(reg_set, reg_free, range.start, range.end, clobbers, hint_idx)) |reg_idx| {
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
///
/// If `hint_idx` is set and that register is free and safe for this
/// range, it is returned without consulting the preference lists. This
/// lets callers steer ABI-constrained vregs (e.g. call-arg vregs) into
/// the right register at allocation time. Hints are advisory: when the
/// hinted register is unavailable we silently fall back to the normal
/// caller/callee-saved scan.
fn findSafeReg(
    reg_set: RegSet,
    reg_free: u64,
    start: u32,
    end: u32,
    clobbers: []const ClobberPoint,
    hint_idx: ?u8,
) ?u8 {
    if (hint_idx) |i| {
        if (i < reg_set.alloc_regs.len) {
            const bit = @as(u64, 1) << @intCast(i);
            if ((reg_free & bit) != 0 and regSafeForRange(i, start, end, clobbers)) return i;
        }
    }
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

test "allocateFromRanges: non-overlapping intervals reuse a register" {
    const allocator = std.testing.allocator;
    const one_reg_set: RegSet = .{
        .alloc_regs = &.{7},
        .callee_saved_indices = &.{},
        .caller_saved_indices = &.{0},
        .spill_base = 64,
        .spill_stride = 8,
    };
    const ranges = [_]analysis.LiveRange{
        .{ .vreg = 0, .start = 0, .end = 1, .type = .i64 },
        .{ .vreg = 1, .start = 2, .end = 3, .type = .i64 },
    };

    var result = try allocateFromRanges(allocator, one_reg_set, &.{}, &ranges);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 0), result.spill_count);
    try std.testing.expectEqual(Allocation{ .reg = 7 }, result.get(0).?);
    try std.testing.expectEqual(Allocation{ .reg = 7 }, result.get(1).?);
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

test "allocateFromRanges: call-spanning intervals avoid caller-saved registers" {
    const allocator = std.testing.allocator;
    const mixed_reg_set: RegSet = .{
        .alloc_regs = &.{ 0, 19 },
        .callee_saved_indices = &.{1},
        .caller_saved_indices = &.{0},
        .spill_base = 128,
        .spill_stride = 8,
    };
    const ranges = [_]analysis.LiveRange{
        .{ .vreg = 0, .start = 0, .end = 4, .type = .i64 },
        .{ .vreg = 1, .start = 1, .end = 2, .type = .i64 },
    };
    const clobbers = [_]ClobberPoint{
        .{ .pos = 3, .regs_clobbered = 0b01 },
    };

    var result = try allocateFromRanges(allocator, mixed_reg_set, &clobbers, &ranges);
    defer result.deinit();

    try std.testing.expectEqual(Allocation{ .reg = 19 }, result.get(0).?);
    try std.testing.expectEqual(Allocation{ .reg = 0 }, result.get(1).?);
    try std.testing.expectEqual(@as(u32, 0), result.spill_count);
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

test "allocateFromRanges: 24-register v128 pool holds 12 live vectors without spills" {
    const allocator = std.testing.allocator;
    const regs = [_]PhysReg{
        0,  1,  2,  3,  4,  5,  6,  7,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
    };
    const indices = [_]u8{
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
    };
    const vreg_set: RegSet = .{
        .alloc_regs = &regs,
        .callee_saved_indices = &.{},
        .caller_saved_indices = &indices,
        .spill_base = 128,
        .spill_stride = 8,
    };
    var ranges: [12]analysis.LiveRange = undefined;
    for (&ranges, 0..) |*range, i| {
        range.* = .{
            .vreg = @intCast(i),
            .start = @intCast(i),
            .end = 100,
            .type = .v128,
        };
    }

    var result = try allocateFromRanges(allocator, vreg_set, &.{}, &ranges);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 0), result.spill_count);
    for (0..12) |i| {
        try std.testing.expect(result.get(@intCast(i)).? == .reg);
    }
}

test "allocateFromRangesWithHints: hint honored when target reg is free" {
    const allocator = std.testing.allocator;
    const ranges = [_]analysis.LiveRange{
        .{ .vreg = 0, .start = 0, .end = 1, .type = .i64 },
    };
    // Without a hint, the first caller-saved index (0 → physreg 2 = rdx)
    // would be picked. Hint to index 4 (physreg 8 = r8) instead.
    const hints = [_]Hint{.{ .vreg = 0, .reg_idx = 4 }};

    var result = try allocateFromRangesWithHints(
        allocator,
        test_reg_set,
        &.{},
        &ranges,
        &hints,
    );
    defer result.deinit();

    try std.testing.expectEqual(Allocation{ .reg = 8 }, result.get(0).?);
}

test "allocateFromRangesWithHints: hint silently ignored when target reg is busy" {
    const allocator = std.testing.allocator;
    // vreg 0 is allocated first into physreg 8 (idx 4) via a hint.
    // vreg 1 is then hinted to the same idx 4, but it's still in use,
    // so the allocator must fall back without erroring.
    const ranges = [_]analysis.LiveRange{
        .{ .vreg = 0, .start = 0, .end = 5, .type = .i64 },
        .{ .vreg = 1, .start = 1, .end = 3, .type = .i64 },
    };
    const hints = [_]Hint{
        .{ .vreg = 0, .reg_idx = 4 },
        .{ .vreg = 1, .reg_idx = 4 },
    };

    var result = try allocateFromRangesWithHints(
        allocator,
        test_reg_set,
        &.{},
        &ranges,
        &hints,
    );
    defer result.deinit();

    try std.testing.expectEqual(Allocation{ .reg = 8 }, result.get(0).?);
    // vreg 1 must have landed somewhere else (not idx 4 → physreg 8).
    const a1 = result.get(1).?;
    try std.testing.expect(a1 == .reg);
    try std.testing.expect(a1.reg != 8);
    try std.testing.expectEqual(@as(u32, 0), result.spill_count);
}

test "allocateFromRangesWithHints: hint ignored when target is clobbered inside range" {
    const allocator = std.testing.allocator;
    // Range [0, 4] spans a clobber at pos 2 that destroys idx 0
    // (caller-saved). The hint targets idx 0; it must be rejected and
    // the allocator must fall back to a callee-saved register.
    const ranges = [_]analysis.LiveRange{
        .{ .vreg = 0, .start = 0, .end = 4, .type = .i64 },
    };
    const clobbers = [_]ClobberPoint{
        .{ .pos = 2, .regs_clobbered = 0b1 }, // clobbers idx 0
    };
    const hints = [_]Hint{.{ .vreg = 0, .reg_idx = 0 }};

    var result = try allocateFromRangesWithHints(
        allocator,
        test_reg_set,
        &clobbers,
        &ranges,
        &hints,
    );
    defer result.deinit();

    const a = result.get(0).?;
    try std.testing.expect(a == .reg);
    // Hint (idx 0 → physreg 2 = rdx) was unsafe; allocator must have
    // chosen a callee-saved physreg. Callee-saved phys regs in
    // test_reg_set are alloc_regs[1, 6, 7, 8, 9] = {3, 12, 13, 14, 15}.
    try std.testing.expect(a.reg != 2);
}

test "allocateFromRangesWithHints: duplicate hints — first wins" {
    const allocator = std.testing.allocator;
    const ranges = [_]analysis.LiveRange{
        .{ .vreg = 0, .start = 0, .end = 1, .type = .i64 },
    };
    // Both hints target free safe registers; the first should be honored.
    const hints = [_]Hint{
        .{ .vreg = 0, .reg_idx = 4 }, // physreg 8 = r8
        .{ .vreg = 0, .reg_idx = 5 }, // physreg 9 = r9
    };

    var result = try allocateFromRangesWithHints(
        allocator,
        test_reg_set,
        &.{},
        &ranges,
        &hints,
    );
    defer result.deinit();

    try std.testing.expectEqual(Allocation{ .reg = 8 }, result.get(0).?);
}

test "allocateFromRangesWithHints: empty hint list behaves like allocateFromRanges" {
    const allocator = std.testing.allocator;
    const ranges = [_]analysis.LiveRange{
        .{ .vreg = 0, .start = 0, .end = 1, .type = .i64 },
        .{ .vreg = 1, .start = 0, .end = 1, .type = .i64 },
    };

    var result_a = try allocateFromRangesWithHints(
        allocator,
        test_reg_set,
        &.{},
        &ranges,
        &.{},
    );
    defer result_a.deinit();
    var result_b = try allocateFromRanges(allocator, test_reg_set, &.{}, &ranges);
    defer result_b.deinit();

    try std.testing.expectEqual(result_a.get(0).?, result_b.get(0).?);
    try std.testing.expectEqual(result_a.get(1).?, result_b.get(1).?);
}
