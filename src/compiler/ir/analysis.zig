//! IR analysis passes for register allocation.
//!
//! Provides CFG construction, liveness analysis, and live range computation
//! needed by the linear scan register allocator.

const std = @import("std");
const ir = @import("ir.zig");

pub const TimingOptions = struct {
    enabled: bool = false,
    threshold_ns: u64 = 100 * std.time.ns_per_ms,
    module_filter: ?u32 = null,
    func_filter: ?u32 = null,

    fn contextMatches(self: TimingOptions, ctx: TimingContext) bool {
        if (!self.enabled) return false;
        if (self.module_filter) |want| {
            if (ctx.module_idx == null or ctx.module_idx.? != want) return false;
        }
        if (self.func_filter) |want| {
            if (ctx.func_idx == null or ctx.func_idx.? != want) return false;
        }
        return true;
    }
};

pub const TimingContext = struct {
    module_idx: ?u32 = null,
    func_idx: ?u32 = null,
    phase: []const u8 = "-",
    pass_idx: ?u32 = null,
    pass_name: []const u8 = "-",
};

const TimingState = struct {
    options: TimingOptions = .{},
    context: TimingContext = .{},
    build_successors_calls: u64 = 0,
    compute_dominators_calls: u64 = 0,
};

threadlocal var timing_state: TimingState = .{};

pub fn setTimingOptions(options: TimingOptions) TimingOptions {
    const previous = timing_state.options;
    timing_state.options = options;
    if (!options.enabled) timing_state.context = .{};
    return previous;
}

pub const TimingContextScope = struct {
    previous: TimingContext,

    pub fn deinit(self: *TimingContextScope) void {
        timing_state.context = self.previous;
    }
};

pub fn pushTimingContext(context: TimingContext) TimingContextScope {
    const previous = timing_state.context;
    timing_state.context = context;
    return .{ .previous = previous };
}

const TimingKind = enum {
    build_successors,
    compute_dominators,

    fn label(self: TimingKind) []const u8 {
        return switch (self) {
            .build_successors => "buildSuccessors",
            .compute_dominators => "computeDominators",
        };
    }
};

const TimingSample = struct {
    kind: TimingKind,
    enabled: bool = false,
    call_index: u64 = 0,
    start_ns: u64 = 0,

    fn finish(self: TimingSample, func: *const ir.IrFunction) void {
        if (!self.enabled) return;
        const elapsed_ns = elapsedTimingNsSince(self.start_ns);
        if (elapsed_ns < timing_state.options.threshold_ns) return;
        printAnalysisTiming(self.kind, self.call_index, func, elapsed_ns);
    }
};

fn beginTiming(kind: TimingKind) TimingSample {
    const options = timing_state.options;
    if (!options.contextMatches(timing_state.context)) {
        return .{ .kind = kind };
    }
    const call_index = switch (kind) {
        .build_successors => blk: {
            timing_state.build_successors_calls += 1;
            break :blk timing_state.build_successors_calls;
        },
        .compute_dominators => blk: {
            timing_state.compute_dominators_calls += 1;
            break :blk timing_state.compute_dominators_calls;
        },
    };
    return .{
        .kind = kind,
        .enabled = true,
        .call_index = call_index,
        .start_ns = timingNowNs(),
    };
}

fn timingNowNs() u64 {
    return switch (comptime @import("builtin").os.tag) {
        .linux => blk: {
            const linux = std.os.linux;
            var ts: linux.timespec = undefined;
            const rc = linux.clock_gettime(.MONOTONIC, &ts);
            if (rc != 0) return 0;
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        .macos, .ios, .tvos, .watchos, .visionos => blk: {
            var ts: std.c.timespec = undefined;
            if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        .windows => blk: {
            const ntdll = std.os.windows.ntdll;
            var counter: std.os.windows.LARGE_INTEGER = undefined;
            var freq: std.os.windows.LARGE_INTEGER = undefined;
            _ = ntdll.RtlQueryPerformanceCounter(&counter);
            _ = ntdll.RtlQueryPerformanceFrequency(&freq);
            const ticks: u128 = @intCast(counter);
            const hz: u128 = @intCast(freq);
            break :blk @as(u64, @truncate(ticks * std.time.ns_per_s / hz));
        },
        else => 0,
    };
}

fn elapsedTimingNsSince(start_ns: u64) u64 {
    const now_ns = timingNowNs();
    return if (now_ns >= start_ns) now_ns - start_ns else 0;
}

fn countInstructions(func: *const ir.IrFunction) usize {
    var total: usize = 0;
    for (func.blocks.items) |*block| total += block.instructions.items.len;
    return total;
}

fn printAnalysisTiming(
    kind: TimingKind,
    call_index: u64,
    func: *const ir.IrFunction,
    elapsed_ns: u64,
) void {
    const ctx = timing_state.context;
    var module_buf: [16]u8 = undefined;
    const module_text = if (ctx.module_idx) |idx|
        std.fmt.bufPrint(&module_buf, "{d}", .{idx}) catch "?"
    else
        "-";
    var func_buf: [16]u8 = undefined;
    const func_text = if (ctx.func_idx) |idx|
        std.fmt.bufPrint(&func_buf, "{d}", .{idx}) catch "?"
    else
        "-";
    var pass_idx_buf: [16]u8 = undefined;
    const pass_idx_text = if (ctx.pass_idx) |idx|
        std.fmt.bufPrint(&pass_idx_buf, "{d}", .{idx}) catch "?"
    else
        "-";
    const func_name = func.name orelse "-";
    const elapsed_ms = elapsed_ns / std.time.ns_per_ms;
    const elapsed_ms_frac = (elapsed_ns % std.time.ns_per_ms) / std.time.ns_per_us;
    std.debug.print(
        "[aot-analysis-timing] analysis={s} call={d} mod={s} local_func={s} phase={s} pass={s} pass_name={s} func_name={s} elapsed_ms={d}.{d:0>3} blocks={d} insts={d}\n",
        .{
            kind.label(),
            call_index,
            module_text,
            func_text,
            ctx.phase,
            pass_idx_text,
            ctx.pass_name,
            func_name,
            elapsed_ms,
            elapsed_ms_frac,
            func.blocks.items.len,
            countInstructions(func),
        },
    );
}

// ── CFG: Successor computation ──────────────────────────────────────────

/// Compute successor block IDs for each block by scanning branch instructions.
pub fn buildSuccessors(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, []const ir.BlockId) {
    if (currentCfgAnalysisCache()) |cache| {
        return cloneBlockIdMap(try cache.getSuccessors(func), allocator);
    }
    return buildSuccessorsFresh(func, allocator);
}

fn buildSuccessorsFresh(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, []const ir.BlockId) {
    const timing = beginTiming(.build_successors);
    defer timing.finish(func);

    var successors = std.AutoHashMap(ir.BlockId, []const ir.BlockId).init(allocator);

    for (func.blocks.items, 0..) |block, idx| {
        var succs: std.ArrayList(ir.BlockId) = .empty;
        for (block.instructions.items) |inst| {
            switch (inst.op) {
                .br => |target| try succs.append(allocator, target),
                .br_if => |bi| {
                    try succs.append(allocator, bi.then_block);
                    try succs.append(allocator, bi.else_block);
                },
                .br_table => |bt| {
                    for (bt.targets) |t| try succs.append(allocator, t);
                    try succs.append(allocator, bt.default);
                },
                else => {},
            }
        }
        try successors.put(@intCast(idx), try succs.toOwnedSlice(allocator));
    }
    return successors;
}

pub fn freeBlockIdMap(map: *std.AutoHashMap(ir.BlockId, []const ir.BlockId), allocator: std.mem.Allocator) void {
    var it = map.iterator();
    while (it.next()) |entry| allocator.free(entry.value_ptr.*);
    map.deinit();
}

pub fn cloneBlockIdMap(
    source: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, []const ir.BlockId) {
    var result = std.AutoHashMap(ir.BlockId, []const ir.BlockId).init(allocator);
    errdefer freeBlockIdMap(&result, allocator);

    var it = @constCast(source).iterator();
    while (it.next()) |entry| {
        const owned = try allocator.dupe(ir.BlockId, entry.value_ptr.*);
        try result.put(entry.key_ptr.*, owned);
    }
    return result;
}

fn consumeCachedSuccessor(cached: []const ir.BlockId, index: *usize, target: ir.BlockId) bool {
    if (index.* >= cached.len or cached[index.*] != target) return false;
    index.* += 1;
    return true;
}

fn cachedBlockSuccessorsMatch(block: *const ir.BasicBlock, cached: []const ir.BlockId) bool {
    var index: usize = 0;
    for (block.instructions.items) |inst| {
        switch (inst.op) {
            .br => |target| {
                if (!consumeCachedSuccessor(cached, &index, target)) return false;
            },
            .br_if => |bi| {
                if (!consumeCachedSuccessor(cached, &index, bi.then_block)) return false;
                if (!consumeCachedSuccessor(cached, &index, bi.else_block)) return false;
            },
            .br_table => |bt| {
                for (bt.targets) |target| {
                    if (!consumeCachedSuccessor(cached, &index, target)) return false;
                }
                if (!consumeCachedSuccessor(cached, &index, bt.default)) return false;
            },
            else => {},
        }
    }
    return index == cached.len;
}

fn cachedSuccessorsStillMatch(
    func: *const ir.IrFunction,
    successors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
) bool {
    if (@as(usize, @intCast(successors.count())) != func.blocks.items.len) return false;
    for (0..func.blocks.items.len) |idx| {
        const bid: ir.BlockId = @intCast(idx);
        const cached = successors.get(bid) orelse return false;
        if (!cachedBlockSuccessorsMatch(&func.blocks.items[idx], cached)) return false;
    }
    return true;
}

// ── Liveness analysis ───────────────────────────────────────────────────

/// Per-block liveness sets.
pub const BlockLiveness = struct {
    /// VRegs live at the start of the block.
    live_in: std.AutoHashMap(ir.VReg, void),
    /// VRegs live at the end of the block.
    live_out: std.AutoHashMap(ir.VReg, void),
};

/// Compute liveness information for all blocks using backward dataflow analysis.
/// Returns a map from BlockId to BlockLiveness.
///
/// Treats a `phi` instruction's edge values as plain in-block uses (the
/// legacy handling). This is correct for the post-`lowerPhisToLocals` IR
/// the register allocator consumes (no phis present); use
/// `computeSsaLiveness` when analysing phi-form IR.
pub fn computeLiveness(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, BlockLiveness) {
    return computeLivenessImpl(func, allocator, null);
}

/// SSA-aware liveness (#392 step 1). A `phi` contributes no in-block uses;
/// instead each phi-arm `val` is live-out of its specific predecessor
/// `edge.block` — the value travels on that CFG edge to the join. The phi
/// dest is killed at the top of the join block. Use this on phi-form IR
/// (before `lowerPhisToLocals`).
pub fn computeSsaLiveness(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, BlockLiveness) {
    var phi_arms = try buildPhiArmsByPred(func, allocator);
    defer {
        var it = phi_arms.iterator();
        while (it.next()) |e| allocator.free(e.value_ptr.*);
        phi_arms.deinit();
    }
    return computeLivenessImpl(func, allocator, &phi_arms);
}

/// Group every phi-arm value by the predecessor block it travels out of.
/// For `dest = phi [(B0,v0),(B1,v1),...]` (in any block) this records that
/// `vi` is live-out of `Bi`. Returns a map predecessor-BlockId → owned
/// slice of VRegs (caller frees each slice). Duplicate arms are fine — the
/// liveness set union dedups.
fn buildPhiArmsByPred(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, []const ir.VReg) {
    var lists = std.AutoHashMap(ir.BlockId, std.ArrayListUnmanaged(ir.VReg)).init(allocator);
    defer {
        var it = lists.iterator();
        while (it.next()) |e| e.value_ptr.deinit(allocator);
        lists.deinit();
    }
    for (func.blocks.items) |block| {
        for (block.instructions.items) |inst| {
            if (inst.op == .phi) {
                for (inst.op.phi) |edge| {
                    const gop = try lists.getOrPut(edge.block);
                    if (!gop.found_existing) gop.value_ptr.* = .empty;
                    try gop.value_ptr.append(allocator, edge.val);
                }
            }
        }
    }

    var out = std.AutoHashMap(ir.BlockId, []const ir.VReg).init(allocator);
    errdefer {
        var it = out.iterator();
        while (it.next()) |e| allocator.free(e.value_ptr.*);
        out.deinit();
    }
    var it = lists.iterator();
    while (it.next()) |e| {
        try out.put(e.key_ptr.*, try e.value_ptr.toOwnedSlice(allocator));
    }
    return out;
}

/// Backward-dataflow liveness solver. When `phi_arms_by_pred` is non-null,
/// phi instructions are handled with SSA edge semantics (see
/// `computeSsaLiveness`); when null, the legacy in-block phi-use handling
/// is used and behaviour is byte-identical to the historical
/// `computeLiveness` (pinned by the round-robin differential test).
fn computeLivenessImpl(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
    phi_arms_by_pred: ?*const std.AutoHashMap(ir.BlockId, []const ir.VReg),
) !std.AutoHashMap(ir.BlockId, BlockLiveness) {
    const ssa = phi_arms_by_pred != null;
    const successors = try buildSuccessors(func, allocator);
    defer {
        var it = successors.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        @constCast(&successors).deinit();
    }

    var liveness = std.AutoHashMap(ir.BlockId, BlockLiveness).init(allocator);
    for (0..func.blocks.items.len) |idx| {
        try liveness.put(@intCast(idx), .{
            .live_in = std.AutoHashMap(ir.VReg, void).init(allocator),
            .live_out = std.AutoHashMap(ir.VReg, void).init(allocator),
        });
    }

    // Predecessors drive the worklist: when a block's live_in grows, its
    // predecessors' live_out may grow, so they must be revisited.
    var predecessors = try buildPredecessorsFromSuccessors(func, allocator, &successors);
    defer {
        var it = predecessors.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        predecessors.deinit();
    }

    // Worklist solver for the backward liveness dataflow. Converges to the
    // same least fixpoint as a round-robin iterate-until-stable sweep, but
    // only re-visits blocks whose successors changed instead of rescanning
    // every block each round (issue #778: liveness dominated codegen on
    // functions with thousands of blocks). `computeLivenessRoundRobin` is
    // the reference implementation a differential test pins this to.
    const nblocks = func.blocks.items.len;
    var in_queue = try allocator.alloc(bool, nblocks);
    defer allocator.free(in_queue);
    @memset(in_queue, false);

    // Tracks whether a block's transfer has run at least once. The first
    // visit must always recompute live_in (to seed use[B]); on later visits
    // an unchanged live_out means live_in cannot change, so the transfer
    // walk can be skipped (issue #782).
    var visited = try allocator.alloc(bool, nblocks);
    defer allocator.free(visited);
    @memset(visited, false);

    var worklist: std.ArrayList(ir.BlockId) = .empty;
    defer worklist.deinit(allocator);
    // Seed in reverse-postorder of the forward CFG so the LIFO worklist pops
    // blocks in postorder — each block after its successors — the order that
    // lets a backward-liveness sweep propagate furthest before re-enqueues,
    // minimizing total block visits to converge (issue #782). Seed order
    // only affects convergence speed, never the fixpoint (pinned by the
    // `computeLivenessRoundRobin` differential test).
    {
        const postorder = try forwardPostorder(func, allocator, &successors);
        defer allocator.free(postorder);
        var i: usize = postorder.len;
        while (i > 0) {
            i -= 1;
            const bid = postorder[i];
            try worklist.append(allocator, bid);
            in_queue[bid] = true;
        }
    }

    // Single reusable scratch set for the per-block transfer, cleared (not
    // freed) each visit to avoid allocator churn on the hot path (issue #782).
    var live = std.AutoHashMap(ir.VReg, void).init(allocator);
    defer live.deinit();

    while (worklist.pop()) |bid| {
        in_queue[bid] = false;
        const block = &func.blocks.items[bid];
        const bl = liveness.getPtr(bid).?;

        // live_out ∪= live_in[succ]  (monotone accumulate); note whether the
        // union added anything new.
        var live_out_grew = false;
        if (successors.get(bid)) |succs| {
            for (succs) |succ_id| {
                if (liveness.getPtr(succ_id)) |succ_bl| {
                    var sit = succ_bl.live_in.iterator();
                    while (sit.next()) |entry| {
                        const result = try bl.live_out.getOrPut(entry.key_ptr.*);
                        if (!result.found_existing) {
                            result.value_ptr.* = {};
                            live_out_grew = true;
                        }
                    }
                }
            }
        }

        // SSA edge semantics (#392): every phi-arm whose edge leaves THIS
        // block is live-out here, regardless of the join's live_in (the
        // join killed the phi dest and never lists the arm). This term is
        // constant across iterations; it must still flag `live_out_grew` on
        // first add so the new value propagates back to predecessors.
        if (phi_arms_by_pred) |pa| {
            if (pa.get(bid)) |arms| {
                for (arms) |val| {
                    const result = try bl.live_out.getOrPut(val);
                    if (!result.found_existing) {
                        result.value_ptr.* = {};
                        live_out_grew = true;
                    }
                }
            }
        }

        // live_in depends only on live_out and the fixed use/def of this
        // block, so once the block has been visited an unchanged live_out
        // means live_in cannot change — skip the transfer walk entirely.
        if (visited[bid] and !live_out_grew) continue;
        visited[bid] = true;

        // live_in = use[B] ∪ (live_out[B] - def[B]) — recomputed by the same
        // backward instruction walk the reference uses, so the per-block
        // transfer function (and therefore the fixpoint) is identical.
        live.clearRetainingCapacity();
        var lit = bl.live_out.iterator();
        while (lit.next()) |entry| try live.put(entry.key_ptr.*, {});

        var inst_idx: usize = block.instructions.items.len;
        while (inst_idx > 0) {
            inst_idx -= 1;
            const inst = block.instructions.items[inst_idx];
            if (inst.dest) |dest| _ = live.remove(dest);
            if (inst.op == .parallel_copy) {
                for (inst.op.parallel_copy) |p| _ = live.remove(p.dst);
            }
            // SSA mode: a phi contributes no in-block uses — its arms are
            // accounted on the predecessor edges above. Legacy mode keeps
            // the historical in-block phi-use handling.
            if (!(ssa and inst.op == .phi)) addInstUses(&live, inst);
        }

        // Merge into live_in; if it grew, revisit predecessors so their
        // live_out picks up the new entries.
        var grew = false;
        var wit = live.iterator();
        while (wit.next()) |entry| {
            const result = try bl.live_in.getOrPut(entry.key_ptr.*);
            if (!result.found_existing) {
                result.value_ptr.* = {};
                grew = true;
            }
        }
        if (grew) {
            if (predecessors.get(bid)) |preds| {
                for (preds) |p| {
                    if (!in_queue[p]) {
                        in_queue[p] = true;
                        try worklist.append(allocator, p);
                    }
                }
            }
        }
    }

    return liveness;
}

/// Compute a postorder of the forward CFG covering every block (length
/// `func.blocks.items.len`). A DFS is started from the entry block and then
/// from any still-unvisited block, in index order, so unreachable components
/// are included deterministically. Iterative (explicit stack) because IR
/// functions can have tens of thousands of blocks. Used only to seed the
/// liveness worklist; ordering affects convergence speed, not the result.
fn forwardPostorder(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
    successors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
) ![]ir.BlockId {
    const nblocks = func.blocks.items.len;
    const order = try allocator.alloc(ir.BlockId, nblocks);
    errdefer allocator.free(order);
    if (nblocks == 0) return order;

    var visited = try allocator.alloc(bool, nblocks);
    defer allocator.free(visited);
    @memset(visited, false);

    const Frame = struct { bid: ir.BlockId, next: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(allocator);

    var count: usize = 0;
    for (0..nblocks) |start| {
        if (visited[start]) continue;
        visited[start] = true;
        try stack.append(allocator, .{ .bid = @intCast(start), .next = 0 });
        while (stack.items.len > 0) {
            const ti = stack.items.len - 1;
            const bid = stack.items[ti].bid;
            const succs: []const ir.BlockId = successors.get(bid) orelse &[_]ir.BlockId{};
            if (stack.items[ti].next < succs.len) {
                const succ = succs[stack.items[ti].next];
                stack.items[ti].next += 1;
                if (succ < nblocks and !visited[succ]) {
                    visited[succ] = true;
                    try stack.append(allocator, .{ .bid = succ, .next = 0 });
                }
            } else {
                order[count] = bid;
                count += 1;
                _ = stack.pop();
            }
        }
    }
    std.debug.assert(count == nblocks);
    return order;
}

/// Reference round-robin liveness solver retained as the differential
/// oracle for `computeLiveness` (the worklist version). Both compute the
/// same least fixpoint of the backward liveness dataflow; this one simply
/// sweeps every block in reverse index order until a full pass makes no
/// change. Used by tests only — do not call from codegen.
fn computeLivenessRoundRobin(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, BlockLiveness) {
    const successors = try buildSuccessors(func, allocator);
    defer {
        var it = successors.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        @constCast(&successors).deinit();
    }

    var liveness = std.AutoHashMap(ir.BlockId, BlockLiveness).init(allocator);
    for (0..func.blocks.items.len) |idx| {
        try liveness.put(@intCast(idx), .{
            .live_in = std.AutoHashMap(ir.VReg, void).init(allocator),
            .live_out = std.AutoHashMap(ir.VReg, void).init(allocator),
        });
    }

    var changed = true;
    while (changed) {
        changed = false;
        var block_idx: usize = func.blocks.items.len;
        while (block_idx > 0) {
            block_idx -= 1;
            const bid: ir.BlockId = @intCast(block_idx);
            const block = &func.blocks.items[block_idx];
            const bl = liveness.getPtr(bid).?;

            if (successors.get(bid)) |succs| {
                for (succs) |succ_id| {
                    if (liveness.getPtr(succ_id)) |succ_bl| {
                        var sit = succ_bl.live_in.iterator();
                        while (sit.next()) |entry| {
                            const result = try bl.live_out.getOrPut(entry.key_ptr.*);
                            if (!result.found_existing) {
                                result.value_ptr.* = {};
                                changed = true;
                            }
                        }
                    }
                }
            }

            var live = std.AutoHashMap(ir.VReg, void).init(allocator);
            defer live.deinit();
            var lit = bl.live_out.iterator();
            while (lit.next()) |entry| try live.put(entry.key_ptr.*, {});

            var inst_idx: usize = block.instructions.items.len;
            while (inst_idx > 0) {
                inst_idx -= 1;
                const inst = block.instructions.items[inst_idx];
                if (inst.dest) |dest| _ = live.remove(dest);
                if (inst.op == .parallel_copy) {
                    for (inst.op.parallel_copy) |p| _ = live.remove(p.dst);
                }
                addInstUses(&live, inst);
            }

            var wit = live.iterator();
            while (wit.next()) |entry| {
                const result = try bl.live_in.getOrPut(entry.key_ptr.*);
                if (!result.found_existing) {
                    result.value_ptr.* = {};
                    changed = true;
                }
            }
        }
    }

    return liveness;
}

/// Add all VReg uses of an instruction to a live set.
fn addInstUses(live: *std.AutoHashMap(ir.VReg, void), inst: ir.Inst) void {
    switch (inst.op) {
        .iconst_32, .iconst_64, .fconst_32, .fconst_64, .v128_const => {},
        .local_get, .global_get => {},
        .br, .@"unreachable", .atomic_fence => {},

        .add,
        .sub,
        .mul,
        .div_s,
        .div_u,
        .rem_s,
        .rem_u,
        .@"and",
        .@"or",
        .xor,
        .shl,
        .shr_s,
        .shr_u,
        .rotl,
        .rotr,
        .eq,
        .ne,
        .lt_s,
        .lt_u,
        .gt_s,
        .gt_u,
        .le_s,
        .le_u,
        .ge_s,
        .ge_u,
        .f_min,
        .f_max,
        .f_copysign,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },

        .v128_bitwise => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },
        .v128_bitselect => |sel| {
            live.put(sel.a, {}) catch {};
            live.put(sel.b, {}) catch {};
            live.put(sel.mask, {}) catch {};
        },
        .i32x4_binop => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },
        .i32x4_unop => |un| live.put(un.vector, {}) catch {},
        .i32x4_extadd_pairwise_i16x8 => |op| live.put(op.vector, {}) catch {},
        .i32x4_dot_i16x8_s => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },
        .i32x4_extend_i16x8 => |op| live.put(op.vector, {}) catch {},
        .f32x4_binop => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },
        .f32x4_unop => |un| live.put(un.vector, {}) catch {},
        .f32x4_convert_i32x4 => |op| live.put(op.vector, {}) catch {},
        .i32x4_trunc_sat => |op| live.put(op.vector, {}) catch {},
        .f32x4_demote_f64x2_zero => |op| live.put(op.vector, {}) catch {},
        .i32x4_extmul_i16x8 => |op| {
            live.put(op.lhs, {}) catch {};
            live.put(op.rhs, {}) catch {};
        },
        .i8x16_binop => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },
        .i8x16_shuffle => |op| {
            live.put(op.lhs, {}) catch {};
            live.put(op.rhs, {}) catch {};
        },
        .i8x16_swizzle => |op| {
            live.put(op.vector, {}) catch {};
            live.put(op.indices, {}) catch {};
        },
        .i8x16_narrow_i16x8 => |op| {
            live.put(op.lhs, {}) catch {};
            live.put(op.rhs, {}) catch {};
        },
        .i8x16_unop => |un| live.put(un.vector, {}) catch {},
        .i8x16_shift => |shift| {
            live.put(shift.vector, {}) catch {};
            live.put(shift.count, {}) catch {};
        },
        .i16x8_binop => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },
        .i16x8_unop => |un| live.put(un.vector, {}) catch {},
        .i16x8_extadd_pairwise_i8x16 => |op| live.put(op.vector, {}) catch {},
        .i16x8_extend_i8x16 => |op| live.put(op.vector, {}) catch {},
        .i16x8_extmul_i8x16 => |op| {
            live.put(op.lhs, {}) catch {};
            live.put(op.rhs, {}) catch {};
        },
        .i16x8_narrow_i32x4 => |op| {
            live.put(op.lhs, {}) catch {};
            live.put(op.rhs, {}) catch {};
        },
        .i64x2_binop => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },
        .f64x2_binop => |bin| {
            live.put(bin.lhs, {}) catch {};
            live.put(bin.rhs, {}) catch {};
        },
        .f64x2_unop => |un| live.put(un.vector, {}) catch {},
        .f64x2_convert_low_i32x4 => |op| live.put(op.vector, {}) catch {},
        .f64x2_promote_low_f32x4 => |op| live.put(op.vector, {}) catch {},
        .i64x2_unop => |un| live.put(un.vector, {}) catch {},
        .i64x2_extend_i32x4 => |op| live.put(op.vector, {}) catch {},
        .i64x2_extmul_i32x4 => |op| {
            live.put(op.lhs, {}) catch {};
            live.put(op.rhs, {}) catch {};
        },
        .i64x2_shift => |shift| {
            live.put(shift.vector, {}) catch {};
            live.put(shift.count, {}) catch {};
        },
        .i32x4_shift => |shift| {
            live.put(shift.vector, {}) catch {};
            live.put(shift.count, {}) catch {};
        },
        .i16x8_shift => |shift| {
            live.put(shift.vector, {}) catch {};
            live.put(shift.count, {}) catch {};
        },

        .clz,
        .ctz,
        .popcnt,
        .eqz,
        .wrap_i64,
        .extend_i32_s,
        .extend_i32_u,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i64_s,
        .convert_i32_u,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .reinterpret,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .v128_not,
        .v128_any_true,
        .i32x4_splat,
        .f32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        .f64x2_splat,
        => |vreg| live.put(vreg, {}) catch {},
        .simd_all_true => |op| live.put(op.vector, {}) catch {},
        .simd_bitmask => |op| live.put(op.vector, {}) catch {},
        .i32x4_extract_lane => |lane| live.put(lane.vector, {}) catch {},
        .f32x4_extract_lane => |lane| live.put(lane.vector, {}) catch {},
        .i8x16_extract_lane => |lane| live.put(lane.vector, {}) catch {},
        .i16x8_extract_lane => |lane| live.put(lane.vector, {}) catch {},
        .i64x2_extract_lane => |lane| live.put(lane.vector, {}) catch {},
        .f64x2_extract_lane => |lane| live.put(lane.vector, {}) catch {},
        .i32x4_replace_lane => |lane| {
            live.put(lane.vector, {}) catch {};
            live.put(lane.val, {}) catch {};
        },
        .f32x4_replace_lane => |lane| {
            live.put(lane.vector, {}) catch {};
            live.put(lane.val, {}) catch {};
        },
        .i8x16_replace_lane => |lane| {
            live.put(lane.vector, {}) catch {};
            live.put(lane.val, {}) catch {};
        },
        .i16x8_replace_lane => |lane| {
            live.put(lane.vector, {}) catch {};
            live.put(lane.val, {}) catch {};
        },
        .i64x2_replace_lane => |lane| {
            live.put(lane.vector, {}) catch {};
            live.put(lane.val, {}) catch {};
        },
        .f64x2_replace_lane => |lane| {
            live.put(lane.vector, {}) catch {};
            live.put(lane.val, {}) catch {};
        },

        .local_set => |ls| live.put(ls.val, {}) catch {},
        .global_set => |gs| live.put(gs.val, {}) catch {},
        .load => |ld| live.put(ld.base, {}) catch {},
        .v128_load => |ld| live.put(ld.base, {}) catch {},
        .v128_load_splat => |ld| live.put(ld.base, {}) catch {},
        .v128_load_zero => |ld| live.put(ld.base, {}) catch {},
        .v128_load_extend => |ld| live.put(ld.base, {}) catch {},
        .v128_load_lane => |ld| {
            live.put(ld.base, {}) catch {};
            live.put(ld.vector, {}) catch {};
        },
        .store => |st| {
            live.put(st.base, {}) catch {};
            live.put(st.val, {}) catch {};
        },
        .v128_store => |st| {
            live.put(st.base, {}) catch {};
            live.put(st.val, {}) catch {};
        },
        .v128_store_lane => |st| {
            live.put(st.base, {}) catch {};
            live.put(st.vector, {}) catch {};
        },
        .br_if => |bi| live.put(bi.cond, {}) catch {},
        .br_table => |bt| live.put(bt.index, {}) catch {},
        .ret => |maybe_vreg| if (maybe_vreg) |v| live.put(v, {}) catch {},
        .ret_multi => |vregs| {
            for (vregs) |v| live.put(v, {}) catch {};
        },
        .call_result => {},
        .call => |cl| {
            for (cl.args) |arg| live.put(arg, {}) catch {};
        },
        .call_indirect => |ci| {
            live.put(ci.elem_idx, {}) catch {};
            for (ci.args) |arg| live.put(arg, {}) catch {};
        },
        .call_ref => |cr| {
            live.put(cr.func_ref, {}) catch {};
            for (cr.args) |arg| live.put(arg, {}) catch {};
        },
        .select => |sel| {
            live.put(sel.cond, {}) catch {};
            live.put(sel.if_true, {}) catch {};
            live.put(sel.if_false, {}) catch {};
        },

        .atomic_load => |al| live.put(al.base, {}) catch {},
        .atomic_store => |ast| {
            live.put(ast.base, {}) catch {};
            live.put(ast.val, {}) catch {};
        },
        .atomic_rmw => |ar| {
            live.put(ar.base, {}) catch {};
            live.put(ar.val, {}) catch {};
        },
        .atomic_cmpxchg => |ac| {
            live.put(ac.base, {}) catch {};
            live.put(ac.expected, {}) catch {};
            live.put(ac.replacement, {}) catch {};
        },
        .atomic_notify => |an| {
            live.put(an.base, {}) catch {};
            live.put(an.count, {}) catch {};
        },
        .atomic_wait => |aw| {
            live.put(aw.base, {}) catch {};
            live.put(aw.expected, {}) catch {};
            live.put(aw.timeout, {}) catch {};
        },
        .memory_copy => |mc| {
            live.put(mc.dst, {}) catch {};
            live.put(mc.src, {}) catch {};
            live.put(mc.len, {}) catch {};
        },
        .memory_fill => |mf| {
            live.put(mf.dst, {}) catch {};
            live.put(mf.val, {}) catch {};
            live.put(mf.len, {}) catch {};
        },
        .memory_size => {},
        .memory_grow => |pages| {
            live.put(pages, {}) catch {};
        },
        .table_size => {},
        .table_get => |tg| {
            live.put(tg.idx, {}) catch {};
        },
        .table_set => |ts| {
            live.put(ts.idx, {}) catch {};
            live.put(ts.val, {}) catch {};
        },
        .table_grow => |tg| {
            live.put(tg.init, {}) catch {};
            live.put(tg.delta, {}) catch {};
        },
        .ref_func => {},
        .memory_init => |mi| {
            live.put(mi.dst, {}) catch {};
            live.put(mi.src, {}) catch {};
            live.put(mi.len, {}) catch {};
        },
        .data_drop => {},
        .table_init => |ti| {
            live.put(ti.dst, {}) catch {};
            live.put(ti.src, {}) catch {};
            live.put(ti.len, {}) catch {};
        },
        .elem_drop => {},
        .phi => |edges| {
            for (edges) |edge| live.put(edge.val, {}) catch {};
        },
        .parallel_copy => |pairs| {
            for (pairs) |p| live.put(p.src, {}) catch {};
        },

        // #672 EH ops. `throw`'s args are the live uses; `throw_ref`'s
        // exnref is a single live use. `try_table_*` carry no vregs.
        .try_table_begin, .try_table_end => {},
        .throw => |th| for (th.args) |a| live.put(a, {}) catch {},
        .throw_ref => |v| live.put(v, {}) catch {},
    }
}

/// A live range interval for a VReg.
///
/// `max_loop_depth` is the loop-nest depth of the hottest block the range
/// overlaps. The register allocator uses it to bias eviction toward the
/// coldest active interval — a long-living, loop-invariant value carrying
/// `max_loop_depth = N` is preferred over a short-lived noise vreg at
/// depth `0` when the spill cost would otherwise be paid once per
/// iteration. See `computeLoopDepthByBlock` for how the value is derived.
pub const LiveRange = struct {
    vreg: ir.VReg,
    start: u32, // global instruction index of definition
    end: u32, // global instruction index of last use
    type: ir.IrType,
    /// Maximum loop-nest depth of any block the range overlaps. `0` means
    /// the range is never inside a loop (cheap to spill). Higher values
    /// indicate hotter code — every iteration of the enclosing loop pays
    /// the load/store cost of a spill, so the allocator should prefer to
    /// evict ranges with smaller `max_loop_depth`.
    max_loop_depth: u8 = 0,
};

/// Compute live ranges for all VRegs in a function.
/// Uses global instruction numbering across all blocks.
pub fn computeLiveRanges(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) ![]LiveRange {
    return computeLiveRangesWithOrder(func, null, allocator);
}

/// Compute live ranges with instruction numbering following `block_order`.
/// When provided, this MUST match the codegen emission order so that the
/// register allocator's interval arithmetic is consistent with actual
/// code layout.
pub fn computeLiveRangesWithOrder(
    func: *const ir.IrFunction,
    block_order: ?[]const ir.BlockId,
    allocator: std.mem.Allocator,
) ![]LiveRange {
    return computeLiveRangesImpl(func, block_order, allocator, false);
}

/// SSA-aware live ranges (#392 step 1). Computes intervals on phi-form IR:
/// each phi dest's range starts at the phi instruction; each phi-arm's
/// range ends at the terminator of its specific predecessor block (the arm
/// is live-out of that predecessor, never into the join). Built on
/// `computeSsaLiveness`. Intended to feed the future SSA-aware allocator
/// (step 2); the legacy `computeLiveRanges*` remain the input to today's
/// post-lowering linear scan.
pub fn computeSsaLiveRanges(
    func: *const ir.IrFunction,
    block_order: ?[]const ir.BlockId,
    allocator: std.mem.Allocator,
) ![]LiveRange {
    return computeLiveRangesImpl(func, block_order, allocator, true);
}

fn computeLiveRangesImpl(
    func: *const ir.IrFunction,
    block_order: ?[]const ir.BlockId,
    allocator: std.mem.Allocator,
    ssa: bool,
) ![]LiveRange {
    const liveness = if (ssa)
        try computeSsaLiveness(func, allocator)
    else
        try computeLiveness(func, allocator);
    defer {
        var it = @constCast(&liveness).iterator();
        while (it.next()) |entry| {
            entry.value_ptr.live_in.deinit();
            entry.value_ptr.live_out.deinit();
        }
        @constCast(&liveness).deinit();
    }

    // Global instruction numbering — follows block_order if provided.
    var def_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer def_pos.deinit();
    var def_type = std.AutoHashMap(ir.VReg, ir.IrType).init(allocator);
    defer def_type.deinit();
    var last_use_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer last_use_pos.deinit();
    // Earliest block-start position at which each vreg is live-in. A value
    // that is live-in to a block is live from that block's start, so its
    // range must begin no later than there — even when its only textual
    // def is positioned *after* that point. This is the loop-carried case:
    // a value defined on a back-edge predecessor (e.g. a `parallel_copy`
    // dst from #540 phi lowering, after the join's `local_get` was removed)
    // is used at the loop header, which precedes the latch def in linear
    // order. Without this the range would collapse to ≈[latch, latch] and
    // the allocator would reuse the register inside the loop, clobbering
    // the loop-carried value (#818 / #540 self-loop hang).
    var first_live_in_pos = std.AutoHashMap(ir.VReg, u32).init(allocator);
    defer first_live_in_pos.deinit();

    // Build default sequential order if none provided.
    const nblocks = func.blocks.items.len;
    var owns_order = false;
    const effective_order: []const ir.BlockId = if (block_order) |bo| bo else blk: {
        const raw = try allocator.alloc(ir.BlockId, nblocks);
        for (raw, 0..) |*r, i| r.* = @intCast(i);
        owns_order = true;
        break :blk raw;
    };
    defer if (owns_order) allocator.free(effective_order);

    // Per-order-position flat-index span: block at order position `i`
    // covers [block_starts[i], block_starts[i+1]). Used after the main
    // pass to annotate each live range with the max loop-nest depth of
    // any block it overlaps.
    const block_starts = try allocator.alloc(u32, effective_order.len + 1);
    defer allocator.free(block_starts);

    var global_idx: u32 = 0;
    for (effective_order, 0..) |bid, oi| {
        block_starts[oi] = global_idx;
        const block = func.blocks.items[bid];

        // VRegs in live_in are used before defined in this block — extend their range
        if (liveness.getPtr(bid)) |bl| {
            var lit = bl.live_in.iterator();
            while (lit.next()) |entry| {
                const vreg = entry.key_ptr.*;
                // Extend last use to at least this block's start
                const existing = last_use_pos.get(vreg) orelse 0;
                try last_use_pos.put(vreg, @max(existing, global_idx));
                // Record the earliest block-start where it is live-in so the
                // range start can be pulled back to cover a pre-def use.
                const prev_li = first_live_in_pos.get(vreg) orelse std.math.maxInt(u32);
                try first_live_in_pos.put(vreg, @min(prev_li, global_idx));
            }
        }

        for (block.instructions.items) |inst| {
            // Record definition position
            if (inst.dest) |dest| {
                if (!def_pos.contains(dest)) {
                    try def_pos.put(dest, global_idx);
                    try def_type.put(dest, inst.type);
                }
            }
            if (inst.op == .parallel_copy) {
                for (inst.op.parallel_copy) |p| {
                    if (!def_pos.contains(p.dst)) {
                        try def_pos.put(p.dst, global_idx);
                        try def_type.put(p.dst, p.ty);
                    }
                }
            }
            // Record last use position. In SSA mode a phi contributes no
            // in-block use; each arm's last use is the terminator of its
            // predecessor, applied via the live_out extension below.
            if (!(ssa and inst.op == .phi)) updateLastUse(&last_use_pos, inst, global_idx);
            global_idx += 1;
        }

        // VRegs in live_out extend to end of block
        if (liveness.getPtr(bid)) |bl| {
            var lit = bl.live_out.iterator();
            while (lit.next()) |entry| {
                const vreg = entry.key_ptr.*;
                const existing = last_use_pos.get(vreg) orelse 0;
                try last_use_pos.put(vreg, @max(existing, global_idx -| 1));
            }
        }
    }
    block_starts[effective_order.len] = global_idx;

    // Per-block loop-nest depth, computed once and reused for every
    // range. On functions without back-edges this stays all-zeros and
    // the eviction heuristic degrades to the pre-change "longest end"
    // rule. Failures here are propagated rather than silently masked —
    // dominator/loop computation is bounded by CFG size and very rarely
    // OOMs in practice.
    const loop_depth_by_block: []u8 = try computeLoopDepthByBlockForFunc(func, allocator);
    defer allocator.free(loop_depth_by_block);

    // Build sorted live ranges
    var ranges: std.ArrayList(LiveRange) = .empty;
    var dit = def_pos.iterator();
    while (dit.next()) |entry| {
        const vreg = entry.key_ptr.*;
        const def_p = entry.value_ptr.*;
        // A loop-carried value is live-in to its loop header before its
        // (back-edge) def; pull the range start back to that live-in point
        // so the interval spans the whole loop, not just [def, def]. For
        // forward values the def dominates every use, so first_live_in_pos
        // >= def_p and this is a no-op.
        const start = if (first_live_in_pos.get(vreg)) |li| @min(def_p, li) else def_p;
        const end = last_use_pos.get(vreg) orelse start;
        const final_end = @max(start, end);
        const depth = maxLoopDepthOverSpan(
            start,
            final_end,
            effective_order,
            block_starts,
            loop_depth_by_block,
        );
        try ranges.append(allocator, .{
            .vreg = vreg,
            .start = start,
            .end = final_end,
            .type = def_type.get(vreg) orelse .i32,
            .max_loop_depth = depth,
        });
    }

    // Sort by start position
    std.mem.sort(LiveRange, ranges.items, {}, struct {
        fn lessThan(_: void, a: LiveRange, b: LiveRange) bool {
            return a.start < b.start;
        }
    }.lessThan);

    return try ranges.toOwnedSlice(allocator);
}

/// Build the dominator tree and loop forest for `func` and return the
/// per-block loop-nest depth (slice of length `func.blocks.items.len`).
/// All zeros if the function has no natural loops or no blocks.
///
/// Exposed under both names so the aarch64 scheduler-aware live-range
/// path can call into the same helper as `computeLiveRangesWithOrder`,
/// keeping the eviction-priority signal byte-identical between
/// backends.
pub fn loopDepthByBlockForFunc(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) ![]u8 {
    return computeLoopDepthByBlockForFunc(func, allocator);
}

fn computeLoopDepthByBlockForFunc(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) ![]u8 {
    const nblocks = func.blocks.items.len;
    if (nblocks == 0) return try allocator.alloc(u8, 0);

    var dom = try computeDominators(func, allocator);
    defer dom.deinit();
    var forest = try computeLoops(func, &dom, allocator);
    defer forest.deinit();
    return try computeLoopDepthByBlock(func, &forest, allocator);
}

/// Walk the blocks whose flat-index spans overlap `[start, end]` and
/// return the largest `depth_by_block` value encountered. `block_starts`
/// is the sentinel-terminated array produced by
/// `computeLiveRangesWithOrder`. Falls back to depth 0 if the range
/// spans no recognized block (defensive — shouldn't happen for ranges
/// produced by this pass).
///
/// Exposed publicly so the aarch64 scheduler-aware live-range builder
/// can reuse the same overlap-walk logic.
pub fn maxLoopDepthOverSpan(
    start: u32,
    end: u32,
    block_order: []const ir.BlockId,
    block_starts: []const u32,
    depth_by_block: []const u8,
) u8 {
    if (block_order.len == 0) return 0;
    var max_depth: u8 = 0;
    for (block_order, 0..) |bid, oi| {
        const blk_begin = block_starts[oi];
        const blk_end = block_starts[oi + 1]; // exclusive
        // Range is [start, end] inclusive; block is [blk_begin, blk_end).
        // No overlap if blk_end <= start or blk_begin > end.
        if (blk_end <= start) continue;
        if (blk_begin > end) break; // block_starts is monotonic; rest are later
        if (bid < depth_by_block.len) {
            const d = depth_by_block[bid];
            if (d > max_depth) max_depth = d;
        }
    }
    return max_depth;
}

fn updateLastUse(last_use: *std.AutoHashMap(ir.VReg, u32), inst: ir.Inst, pos: u32) void {
    switch (inst.op) {
        .iconst_32, .iconst_64, .fconst_32, .fconst_64, .v128_const => {},
        .local_get, .global_get => {},
        .br, .@"unreachable", .atomic_fence => {},

        .add,
        .sub,
        .mul,
        .div_s,
        .div_u,
        .rem_s,
        .rem_u,
        .@"and",
        .@"or",
        .xor,
        .shl,
        .shr_s,
        .shr_u,
        .rotl,
        .rotr,
        .eq,
        .ne,
        .lt_s,
        .lt_u,
        .gt_s,
        .gt_u,
        .le_s,
        .le_u,
        .ge_s,
        .ge_u,
        .f_min,
        .f_max,
        .f_copysign,
        .f_eq,
        .f_ne,
        .f_lt,
        .f_gt,
        .f_le,
        .f_ge,
        => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },

        .v128_bitwise => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },
        .v128_bitselect => |sel| {
            last_use.put(sel.a, pos) catch {};
            last_use.put(sel.b, pos) catch {};
            last_use.put(sel.mask, pos) catch {};
        },
        .i32x4_binop => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },
        .i32x4_unop => |un| last_use.put(un.vector, pos) catch {},
        .i32x4_extadd_pairwise_i16x8 => |op| last_use.put(op.vector, pos) catch {},
        .i32x4_dot_i16x8_s => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },
        .i32x4_extend_i16x8 => |op| last_use.put(op.vector, pos) catch {},
        .f32x4_binop => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },
        .f32x4_unop => |un| last_use.put(un.vector, pos) catch {},
        .f32x4_convert_i32x4 => |op| last_use.put(op.vector, pos) catch {},
        .i32x4_trunc_sat => |op| last_use.put(op.vector, pos) catch {},
        .f32x4_demote_f64x2_zero => |op| last_use.put(op.vector, pos) catch {},
        .i32x4_extmul_i16x8 => |op| {
            last_use.put(op.lhs, pos) catch {};
            last_use.put(op.rhs, pos) catch {};
        },
        .i8x16_binop => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },
        .i8x16_shuffle => |op| {
            last_use.put(op.lhs, pos) catch {};
            last_use.put(op.rhs, pos) catch {};
        },
        .i8x16_swizzle => |op| {
            last_use.put(op.vector, pos) catch {};
            last_use.put(op.indices, pos) catch {};
        },
        .i8x16_narrow_i16x8 => |op| {
            last_use.put(op.lhs, pos) catch {};
            last_use.put(op.rhs, pos) catch {};
        },
        .i8x16_unop => |un| last_use.put(un.vector, pos) catch {},
        .i8x16_shift => |shift| {
            last_use.put(shift.vector, pos) catch {};
            last_use.put(shift.count, pos) catch {};
        },
        .i16x8_binop => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },
        .i16x8_unop => |un| last_use.put(un.vector, pos) catch {},
        .i16x8_extadd_pairwise_i8x16 => |op| last_use.put(op.vector, pos) catch {},
        .i16x8_extend_i8x16 => |op| last_use.put(op.vector, pos) catch {},
        .i16x8_extmul_i8x16 => |op| {
            last_use.put(op.lhs, pos) catch {};
            last_use.put(op.rhs, pos) catch {};
        },
        .i16x8_narrow_i32x4 => |op| {
            last_use.put(op.lhs, pos) catch {};
            last_use.put(op.rhs, pos) catch {};
        },
        .i64x2_binop => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },
        .f64x2_binop => |bin| {
            last_use.put(bin.lhs, pos) catch {};
            last_use.put(bin.rhs, pos) catch {};
        },
        .f64x2_unop => |un| last_use.put(un.vector, pos) catch {},
        .f64x2_convert_low_i32x4 => |op| last_use.put(op.vector, pos) catch {},
        .f64x2_promote_low_f32x4 => |op| last_use.put(op.vector, pos) catch {},
        .i64x2_unop => |un| last_use.put(un.vector, pos) catch {},
        .i64x2_extend_i32x4 => |op| last_use.put(op.vector, pos) catch {},
        .i64x2_extmul_i32x4 => |op| {
            last_use.put(op.lhs, pos) catch {};
            last_use.put(op.rhs, pos) catch {};
        },
        .i64x2_shift => |shift| {
            last_use.put(shift.vector, pos) catch {};
            last_use.put(shift.count, pos) catch {};
        },
        .i32x4_shift => |shift| {
            last_use.put(shift.vector, pos) catch {};
            last_use.put(shift.count, pos) catch {};
        },
        .i16x8_shift => |shift| {
            last_use.put(shift.vector, pos) catch {};
            last_use.put(shift.count, pos) catch {};
        },

        .clz,
        .ctz,
        .popcnt,
        .eqz,
        .wrap_i64,
        .extend_i32_s,
        .extend_i32_u,
        .extend8_s,
        .extend16_s,
        .extend32_s,
        .f_neg,
        .f_abs,
        .f_sqrt,
        .f_ceil,
        .f_floor,
        .f_trunc,
        .f_nearest,
        .trunc_f32_s,
        .trunc_f32_u,
        .trunc_f64_s,
        .trunc_f64_u,
        .convert_s,
        .convert_u,
        .convert_i32_s,
        .convert_i64_s,
        .convert_i32_u,
        .convert_i64_u,
        .demote_f64,
        .promote_f32,
        .reinterpret,
        .trunc_sat_f32_s,
        .trunc_sat_f32_u,
        .trunc_sat_f64_s,
        .trunc_sat_f64_u,
        .v128_not,
        .v128_any_true,
        .i32x4_splat,
        .f32x4_splat,
        .i8x16_splat,
        .i16x8_splat,
        .i64x2_splat,
        .f64x2_splat,
        => |vreg| last_use.put(vreg, pos) catch {},
        .simd_all_true => |op| last_use.put(op.vector, pos) catch {},
        .simd_bitmask => |op| last_use.put(op.vector, pos) catch {},
        .i32x4_extract_lane => |lane| last_use.put(lane.vector, pos) catch {},
        .f32x4_extract_lane => |lane| last_use.put(lane.vector, pos) catch {},
        .i8x16_extract_lane => |lane| last_use.put(lane.vector, pos) catch {},
        .i16x8_extract_lane => |lane| last_use.put(lane.vector, pos) catch {},
        .i64x2_extract_lane => |lane| last_use.put(lane.vector, pos) catch {},
        .f64x2_extract_lane => |lane| last_use.put(lane.vector, pos) catch {},
        .i32x4_replace_lane => |lane| {
            last_use.put(lane.vector, pos) catch {};
            last_use.put(lane.val, pos) catch {};
        },
        .f32x4_replace_lane => |lane| {
            last_use.put(lane.vector, pos) catch {};
            last_use.put(lane.val, pos) catch {};
        },
        .i8x16_replace_lane => |lane| {
            last_use.put(lane.vector, pos) catch {};
            last_use.put(lane.val, pos) catch {};
        },
        .i16x8_replace_lane => |lane| {
            last_use.put(lane.vector, pos) catch {};
            last_use.put(lane.val, pos) catch {};
        },
        .i64x2_replace_lane => |lane| {
            last_use.put(lane.vector, pos) catch {};
            last_use.put(lane.val, pos) catch {};
        },
        .f64x2_replace_lane => |lane| {
            last_use.put(lane.vector, pos) catch {};
            last_use.put(lane.val, pos) catch {};
        },

        .local_set => |ls| last_use.put(ls.val, pos) catch {},
        .global_set => |gs| last_use.put(gs.val, pos) catch {},
        .load => |ld| last_use.put(ld.base, pos) catch {},
        .v128_load => |ld| last_use.put(ld.base, pos) catch {},
        .v128_load_splat => |ld| last_use.put(ld.base, pos) catch {},
        .v128_load_zero => |ld| last_use.put(ld.base, pos) catch {},
        .v128_load_extend => |ld| last_use.put(ld.base, pos) catch {},
        .v128_load_lane => |ld| {
            last_use.put(ld.base, pos) catch {};
            last_use.put(ld.vector, pos) catch {};
        },
        .store => |st| {
            last_use.put(st.base, pos) catch {};
            last_use.put(st.val, pos) catch {};
        },
        .v128_store => |st| {
            last_use.put(st.base, pos) catch {};
            last_use.put(st.val, pos) catch {};
        },
        .v128_store_lane => |st| {
            last_use.put(st.base, pos) catch {};
            last_use.put(st.vector, pos) catch {};
        },
        .br_if => |bi| last_use.put(bi.cond, pos) catch {},
        .br_table => |bt| last_use.put(bt.index, pos) catch {},
        .ret => |maybe_vreg| if (maybe_vreg) |v| last_use.put(v, pos) catch {},
        .ret_multi => |vregs| {
            for (vregs) |v| last_use.put(v, pos) catch {};
        },
        .call_result => {},
        .call => |cl| {
            for (cl.args) |arg| last_use.put(arg, pos) catch {};
        },
        .call_indirect => |ci| {
            last_use.put(ci.elem_idx, pos) catch {};
            for (ci.args) |arg| last_use.put(arg, pos) catch {};
        },
        .call_ref => |cr| {
            last_use.put(cr.func_ref, pos) catch {};
            for (cr.args) |arg| last_use.put(arg, pos) catch {};
        },
        .select => |sel| {
            last_use.put(sel.cond, pos) catch {};
            last_use.put(sel.if_true, pos) catch {};
            last_use.put(sel.if_false, pos) catch {};
        },

        .atomic_load => |al| last_use.put(al.base, pos) catch {},
        .atomic_store => |ast| {
            last_use.put(ast.base, pos) catch {};
            last_use.put(ast.val, pos) catch {};
        },
        .atomic_rmw => |ar| {
            last_use.put(ar.base, pos) catch {};
            last_use.put(ar.val, pos) catch {};
        },
        .atomic_cmpxchg => |ac| {
            last_use.put(ac.base, pos) catch {};
            last_use.put(ac.expected, pos) catch {};
            last_use.put(ac.replacement, pos) catch {};
        },
        .atomic_notify => |an| {
            last_use.put(an.base, pos) catch {};
            last_use.put(an.count, pos) catch {};
        },
        .atomic_wait => |aw| {
            last_use.put(aw.base, pos) catch {};
            last_use.put(aw.expected, pos) catch {};
            last_use.put(aw.timeout, pos) catch {};
        },
        .memory_copy => |mc| {
            last_use.put(mc.dst, pos) catch {};
            last_use.put(mc.src, pos) catch {};
            last_use.put(mc.len, pos) catch {};
        },
        .memory_fill => |mf| {
            last_use.put(mf.dst, pos) catch {};
            last_use.put(mf.val, pos) catch {};
            last_use.put(mf.len, pos) catch {};
        },
        .memory_size => {},
        .memory_grow => |pages| {
            last_use.put(pages, pos) catch {};
        },
        .table_size => {},
        .table_get => |tg| {
            last_use.put(tg.idx, pos) catch {};
        },
        .table_set => |ts| {
            last_use.put(ts.idx, pos) catch {};
            last_use.put(ts.val, pos) catch {};
        },
        .table_grow => |tg| {
            last_use.put(tg.init, pos) catch {};
            last_use.put(tg.delta, pos) catch {};
        },
        .ref_func => {},
        .memory_init => |mi| {
            last_use.put(mi.dst, pos) catch {};
            last_use.put(mi.src, pos) catch {};
            last_use.put(mi.len, pos) catch {};
        },
        .data_drop => {},
        .table_init => |ti| {
            last_use.put(ti.dst, pos) catch {};
            last_use.put(ti.src, pos) catch {};
            last_use.put(ti.len, pos) catch {};
        },
        .elem_drop => {},
        .phi => |edges| {
            for (edges) |edge| last_use.put(edge.val, pos) catch {};
        },
        .parallel_copy => |pairs| {
            for (pairs) |p| last_use.put(p.src, pos) catch {};
        },

        // #672 EH ops.
        .try_table_begin, .try_table_end => {},
        .throw => |th| for (th.args) |a| last_use.put(a, pos) catch {},
        .throw_ref => |v| last_use.put(v, pos) catch {},
    }
}

// ── Dominator tree (Cooper-Harvey-Kennedy) ──────────────────────────────

/// Immediate-dominator tree for the function's CFG, rooted at the entry
/// block (block 0). Owns no block-ID storage — `idom`/`post_order` are
/// sized to `func.blocks.items.len`.
///
/// - `idom[b]` is the immediate dominator of block `b`. The entry block
///   dominates only itself, so `idom[entry] == entry`. Unreachable blocks
///   (not visited from entry) have `idom[b] == null`.
/// - `post_order` lists reachable blocks in DFS post-order from the entry.
///   Reverse post-order (RPO) is the natural iteration order for forward
///   dataflow and is used internally by this pass.
///
/// Algorithm: "A Simple, Fast Dominance Algorithm" — Cooper, Harvey,
/// Kennedy (2001). O(N²) worst-case on pathological CFGs, near-linear in
/// practice on structured wasm CFGs.
pub const DomTree = struct {
    idom: []?ir.BlockId,
    /// Post-order numbering for each block, or `null` if unreachable.
    /// Higher number ⇒ later in post-order ⇒ earlier in reverse post-order.
    post_num: []?u32,
    /// Reachable blocks in DFS post-order from entry (length ≤ nblocks).
    post_order: []ir.BlockId,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *DomTree) void {
        self.allocator.free(self.idom);
        self.allocator.free(self.post_num);
        self.allocator.free(self.post_order);
    }

    /// Returns true if `a` dominates `b` (reflexive: every block dominates
    /// itself). Unreachable blocks are dominated only by themselves.
    pub fn dominates(self: *const DomTree, a: ir.BlockId, b: ir.BlockId) bool {
        if (a == b) return true;
        if (self.idom[b] == null) return false;
        var cur: ir.BlockId = b;
        while (true) {
            const next = self.idom[cur] orelse return false;
            if (next == a) return true;
            // Entry's idom is itself; stop to avoid infinite loop.
            if (next == cur) return false;
            cur = next;
        }
    }
};

pub fn cloneDomTree(source: *const DomTree, allocator: std.mem.Allocator) !DomTree {
    const idom = try allocator.dupe(?ir.BlockId, source.idom);
    errdefer allocator.free(idom);
    const post_num = try allocator.dupe(?u32, source.post_num);
    errdefer allocator.free(post_num);
    const post_order = try allocator.dupe(ir.BlockId, source.post_order);
    errdefer allocator.free(post_order);
    return .{
        .idom = idom,
        .post_num = post_num,
        .post_order = post_order,
        .allocator = allocator,
    };
}

pub const CfgAnalysisCache = struct {
    allocator: std.mem.Allocator,
    cached_func: ?*const ir.IrFunction = null,
    dirty: bool = false,
    successors: ?std.AutoHashMap(ir.BlockId, []const ir.BlockId) = null,
    predecessors: ?std.AutoHashMap(ir.BlockId, []const ir.BlockId) = null,
    dominators: ?DomTree = null,

    pub fn init(allocator: std.mem.Allocator) CfgAnalysisCache {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *CfgAnalysisCache) void {
        self.clear();
    }

    fn clear(self: *CfgAnalysisCache) void {
        if (self.dominators) |*dom| {
            dom.deinit();
            self.dominators = null;
        }
        if (self.predecessors) |*preds| {
            freeBlockIdMap(preds, self.allocator);
            self.predecessors = null;
        }
        if (self.successors) |*succs| {
            freeBlockIdMap(succs, self.allocator);
            self.successors = null;
        }
        self.cached_func = null;
        self.dirty = false;
    }

    /// Mark cached CFG analyses as potentially stale. The next lookup compares
    /// the cached successor lists to the current function and reuses the
    /// analyses only when the block set and branch edges are unchanged.
    pub fn invalidate(self: *CfgAnalysisCache) void {
        if (self.successors != null or self.predecessors != null or self.dominators != null) {
            self.dirty = true;
        }
    }

    fn ensureFresh(self: *CfgAnalysisCache, func: *const ir.IrFunction) void {
        if (self.cached_func) |cached| {
            if (cached != func) {
                self.clear();
                return;
            }
        }
        if (!self.dirty) return;
        if (self.successors) |*succs| {
            if (cachedSuccessorsStillMatch(func, succs)) {
                self.dirty = false;
                return;
            }
        }
        self.clear();
    }

    pub fn getSuccessors(
        self: *CfgAnalysisCache,
        func: *const ir.IrFunction,
    ) !*const std.AutoHashMap(ir.BlockId, []const ir.BlockId) {
        self.ensureFresh(func);
        if (self.successors == null) {
            self.successors = try buildSuccessorsFresh(func, self.allocator);
            self.cached_func = func;
            self.dirty = false;
        }
        return &self.successors.?;
    }

    pub fn getPredecessors(
        self: *CfgAnalysisCache,
        func: *const ir.IrFunction,
    ) !*const std.AutoHashMap(ir.BlockId, []const ir.BlockId) {
        self.ensureFresh(func);
        if (self.predecessors == null) {
            const successors = try self.getSuccessors(func);
            self.predecessors = try buildPredecessorsFromSuccessors(func, self.allocator, successors);
        }
        return &self.predecessors.?;
    }

    pub fn getDominators(
        self: *CfgAnalysisCache,
        func: *const ir.IrFunction,
    ) !*const DomTree {
        self.ensureFresh(func);
        if (self.dominators == null) {
            const timing = beginTiming(.compute_dominators);
            defer timing.finish(func);
            const successors = try self.getSuccessors(func);
            const predecessors = try self.getPredecessors(func);
            self.dominators = try computeDominatorsFromCfg(func, self.allocator, successors, predecessors);
        }
        return &self.dominators.?;
    }
};

threadlocal var scoped_cfg_cache: ?*CfgAnalysisCache = null;

pub const CfgAnalysisCacheScope = struct {
    previous: ?*CfgAnalysisCache,

    pub fn deinit(self: *CfgAnalysisCacheScope) void {
        scoped_cfg_cache = self.previous;
    }
};

pub fn pushCfgAnalysisCache(cache: *CfgAnalysisCache) CfgAnalysisCacheScope {
    const previous = scoped_cfg_cache;
    scoped_cfg_cache = cache;
    return .{ .previous = previous };
}

pub fn currentCfgAnalysisCache() ?*CfgAnalysisCache {
    return scoped_cfg_cache;
}

/// Compute predecessors for each block from `buildSuccessors`. The
/// `BasicBlock.predecessors` field is not guaranteed to be populated by
/// all IR producers, so passes that need predecessors should use this.
pub fn buildPredecessors(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !std.AutoHashMap(ir.BlockId, []const ir.BlockId) {
    if (currentCfgAnalysisCache()) |cache| {
        return cloneBlockIdMap(try cache.getPredecessors(func), allocator);
    }
    var successors = try buildSuccessorsFresh(func, allocator);
    defer freeBlockIdMap(&successors, allocator);

    return buildPredecessorsFromSuccessors(func, allocator, &successors);
}

fn buildPredecessorsFromSuccessors(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
    successors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
) !std.AutoHashMap(ir.BlockId, []const ir.BlockId) {
    var lists = std.AutoHashMap(ir.BlockId, std.ArrayList(ir.BlockId)).init(allocator);
    defer {
        var it = lists.iterator();
        while (it.next()) |entry| entry.value_ptr.deinit(allocator);
        lists.deinit();
    }
    for (0..func.blocks.items.len) |idx| {
        try lists.put(@intCast(idx), .empty);
    }

    var sit = @constCast(successors).iterator();
    while (sit.next()) |entry| {
        const from = entry.key_ptr.*;
        for (entry.value_ptr.*) |to| {
            const list_ptr = lists.getPtr(to).?;
            // Deduplicate (br_table may list a target more than once).
            var already = false;
            for (list_ptr.items) |p| {
                if (p == from) {
                    already = true;
                    break;
                }
            }
            if (!already) try list_ptr.append(allocator, from);
        }
    }

    var result = std.AutoHashMap(ir.BlockId, []const ir.BlockId).init(allocator);
    errdefer {
        var it = result.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        result.deinit();
    }
    var lit = lists.iterator();
    while (lit.next()) |entry| {
        const owned = try entry.value_ptr.toOwnedSlice(allocator);
        try result.put(entry.key_ptr.*, owned);
    }
    return result;
}

/// Recompute each block's on-block `predecessors` field from the
/// canonical pred set derived from terminators.
///
/// Most analysis passes use `buildPredecessors` (which returns a fresh
/// map) and never touch `BasicBlock.predecessors`. The on-block field is
/// only consumed by the IR pretty-printer and the IR verifier
/// (check 6, `MissingPredecessor` / `StalePredecessor`). Passes that
/// mutate the CFG without rebuilding the on-block field will trip the
/// verifier; call this helper after such a pass to restore the
/// invariant.
///
/// Deduplicates within each pred list, matching `buildPredecessors`'
/// semantics for `br_table` targets that list the same block twice.
pub fn refreshBlockPredecessors(
    func: *ir.IrFunction,
    allocator: std.mem.Allocator,
) !void {
    var successors = try buildSuccessorsFresh(func, allocator);
    defer freeBlockIdMap(&successors, allocator);
    var preds = try buildPredecessorsFromSuccessors(func, allocator, &successors);
    defer freeBlockIdMap(&preds, allocator);

    for (func.blocks.items, 0..) |*block, idx| {
        block.predecessors.clearRetainingCapacity();
        const list = preds.get(@intCast(idx)) orelse continue;
        try block.predecessors.appendSlice(block.allocator, list);
    }
}

/// Compute the dominator tree for `func` rooted at block 0.
pub fn computeDominators(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) !DomTree {
    if (currentCfgAnalysisCache()) |cache| {
        return cloneDomTree(try cache.getDominators(func), allocator);
    }

    const timing = beginTiming(.compute_dominators);
    defer timing.finish(func);

    var successors = try buildSuccessorsFresh(func, allocator);
    defer freeBlockIdMap(&successors, allocator);
    var predecessors = try buildPredecessorsFromSuccessors(func, allocator, &successors);
    defer freeBlockIdMap(&predecessors, allocator);

    return computeDominatorsFromCfg(func, allocator, &successors, &predecessors);
}

fn computeDominatorsFromCfg(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
    successors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    predecessors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
) !DomTree {
    const nblocks = func.blocks.items.len;

    // ── Iterative DFS to produce post-order from entry (block 0) ──
    var post_order: std.ArrayList(ir.BlockId) = .empty;
    errdefer post_order.deinit(allocator);

    const post_num = try allocator.alloc(?u32, nblocks);
    errdefer allocator.free(post_num);
    @memset(post_num, null);

    if (nblocks > 0) {
        const StackEntry = struct { bid: ir.BlockId, next_succ: usize };
        var visited = try allocator.alloc(bool, nblocks);
        defer allocator.free(visited);
        @memset(visited, false);

        var stack: std.ArrayList(StackEntry) = .empty;
        defer stack.deinit(allocator);

        const entry: ir.BlockId = 0;
        visited[entry] = true;
        try stack.append(allocator, .{ .bid = entry, .next_succ = 0 });

        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            const succs = successors.get(top.bid) orelse &[_]ir.BlockId{};
            if (top.next_succ < succs.len) {
                const s = succs[top.next_succ];
                top.next_succ += 1;
                if (!visited[s]) {
                    visited[s] = true;
                    try stack.append(allocator, .{ .bid = s, .next_succ = 0 });
                }
            } else {
                const bid = top.bid;
                post_num[bid] = @intCast(post_order.items.len);
                try post_order.append(allocator, bid);
                _ = stack.pop();
            }
        }
    }

    // ── Cooper-Harvey-Kennedy iterative idom computation ──
    const idom = try allocator.alloc(?ir.BlockId, nblocks);
    errdefer allocator.free(idom);
    @memset(idom, null);

    if (nblocks > 0 and post_num[0] != null) {
        // Entry dominates itself.
        idom[0] = 0;

        // Reverse post-order excluding entry.
        const Intersect = struct {
            fn call(idom_slice: []const ?ir.BlockId, post: []const ?u32, b1_in: ir.BlockId, b2_in: ir.BlockId) ir.BlockId {
                var b1 = b1_in;
                var b2 = b2_in;
                while (b1 != b2) {
                    while (post[b1].? < post[b2].?) b1 = idom_slice[b1].?;
                    while (post[b2].? < post[b1].?) b2 = idom_slice[b2].?;
                }
                return b1;
            }
        };

        var changed = true;
        while (changed) {
            changed = false;
            // Iterate reverse post-order, skipping the entry node (last in post_order).
            var i: usize = post_order.items.len;
            while (i > 0) {
                i -= 1;
                const b = post_order.items[i];
                if (b == 0) continue;

                const preds = predecessors.get(b) orelse &[_]ir.BlockId{};
                // Pick first processed predecessor as the running idom.
                var new_idom_opt: ?ir.BlockId = null;
                var other_start: usize = 0;
                for (preds, 0..) |p, pi| {
                    if (idom[p] != null) {
                        new_idom_opt = p;
                        other_start = pi + 1;
                        break;
                    }
                }
                const new_idom_first = new_idom_opt orelse continue;
                var new_idom = new_idom_first;
                for (preds[other_start..]) |p| {
                    if (idom[p] != null) {
                        new_idom = Intersect.call(idom, post_num, p, new_idom);
                    }
                }
                if (idom[b] == null or idom[b].? != new_idom) {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }
    }

    return .{
        .idom = idom,
        .post_num = post_num,
        .post_order = try post_order.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

// ── Dominance frontiers ─────────────────────────────────────────────────

/// Compute the dominance frontier for every block in `func`.
///
/// DF(b) = { y | ∃ pred of y that b dominates, but b does not strictly
///             dominate y }. Uses the efficient "bottom-up" algorithm
/// from Cooper, Harvey & Kennedy (2001), §4.2.
///
/// Caller owns the returned slices; call `freeDominanceFrontiers` to release.
pub fn computeDominanceFrontiers(
    dom: *const DomTree,
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) ![][]const ir.BlockId {
    const nblocks = func.blocks.items.len;

    var preds = try buildPredecessors(func, allocator);
    defer {
        var pit = preds.iterator();
        while (pit.next()) |entry| allocator.free(entry.value_ptr.*);
        preds.deinit();
    }

    // Accumulate DF sets as ArrayLists, then convert to owned slices.
    var df_lists = try allocator.alloc(std.ArrayList(ir.BlockId), nblocks);
    defer allocator.free(df_lists);
    for (df_lists) |*l| l.* = .empty;

    for (0..nblocks) |idx| {
        const b: ir.BlockId = @intCast(idx);
        const pred_list = preds.get(b) orelse continue;
        if (pred_list.len < 2) continue; // join point iff ≥2 preds

        for (pred_list) |p| {
            var runner = p;
            while (runner != (dom.idom[b] orelse break)) {
                // Add b to DF(runner) if not already present.
                var dup = false;
                for (df_lists[runner].items) |existing| {
                    if (existing == b) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) try df_lists[runner].append(allocator, b);
                const next = dom.idom[runner] orelse break;
                if (next == runner) break;
                runner = next;
            }
        }
    }

    // Convert to owned slices.
    const result = try allocator.alloc([]const ir.BlockId, nblocks);
    errdefer allocator.free(result);
    for (df_lists, 0..) |*l, i| {
        result[i] = try l.toOwnedSlice(allocator);
    }
    return result;
}

/// Free the slices returned by `computeDominanceFrontiers`.
pub fn freeDominanceFrontiers(df: [][]const ir.BlockId, allocator: std.mem.Allocator) void {
    for (df) |s| allocator.free(s);
    allocator.free(df);
}

// ── Natural-loop detection ──────────────────────────────────────────────

/// A natural loop identified by a single header and one or more latches
/// (blocks with back-edges to the header). Multiple back-edges to the
/// same header are merged into one loop, matching the standard
/// "natural loops of a flow graph" definition.
///
/// Invariants:
///   - `header ∈ blocks` and every `latch ∈ blocks`.
///   - `header` dominates every block in `blocks`.
///   - `blocks` and `latches` are sorted ascending, no duplicates.
pub const Loop = struct {
    header: ir.BlockId,
    latches: []ir.BlockId,
    blocks: []ir.BlockId,

    /// O(log N) membership test (blocks is sorted).
    pub fn containsBlock(self: *const Loop, bid: ir.BlockId) bool {
        var lo: usize = 0;
        var hi: usize = self.blocks.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const v = self.blocks[mid];
            if (v == bid) return true;
            if (v < bid) lo = mid + 1 else hi = mid;
        }
        return false;
    }
};

/// A forest of natural loops for a function. Each loop is identified by
/// its index in `loops`. `header_loop` maps a header block ID back to
/// its loop index, so callers can answer "is this block a loop header?"
/// and "what loop does this header start?" in O(1).
pub const LoopForest = struct {
    loops: []Loop,
    header_loop: std.AutoHashMap(ir.BlockId, u32),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LoopForest) void {
        for (self.loops) |*loop| {
            self.allocator.free(loop.latches);
            self.allocator.free(loop.blocks);
        }
        self.allocator.free(self.loops);
        self.header_loop.deinit();
    }

    /// Returns true if `bid` is the header of some loop in this forest.
    pub fn isHeader(self: *const LoopForest, bid: ir.BlockId) bool {
        return self.header_loop.contains(bid);
    }
};

/// Compute the natural-loop forest for `func` using the supplied
/// dominator tree. Only reachable back-edges contribute to loops;
/// unreachable subgraphs (null idom) are ignored.
pub fn computeLoops(
    func: *const ir.IrFunction,
    dom: *const DomTree,
    allocator: std.mem.Allocator,
) !LoopForest {
    if (currentCfgAnalysisCache()) |cache| {
        return computeLoopsCached(func, dom, allocator, cache);
    }

    return computeLoopsFresh(func, dom, allocator);
}

fn computeLoopsFresh(
    func: *const ir.IrFunction,
    dom: *const DomTree,
    allocator: std.mem.Allocator,
) !LoopForest {
    var successors = try buildSuccessorsFresh(func, allocator);
    defer freeBlockIdMap(&successors, allocator);
    var predecessors = try buildPredecessorsFromSuccessors(func, allocator, &successors);
    defer freeBlockIdMap(&predecessors, allocator);

    return computeLoopsFromCfg(func, dom, allocator, &successors, &predecessors);
}

pub fn computeLoopsCached(
    func: *const ir.IrFunction,
    dom: *const DomTree,
    allocator: std.mem.Allocator,
    cache: ?*CfgAnalysisCache,
) !LoopForest {
    if (cache) |c| {
        const successors = try c.getSuccessors(func);
        const predecessors = try c.getPredecessors(func);
        return computeLoopsFromCfg(func, dom, allocator, successors, predecessors);
    }
    return computeLoopsFresh(func, dom, allocator);
}

fn computeLoopsFromCfg(
    func: *const ir.IrFunction,
    dom: *const DomTree,
    allocator: std.mem.Allocator,
    successors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
    predecessors: *const std.AutoHashMap(ir.BlockId, []const ir.BlockId),
) !LoopForest {
    const nblocks = func.blocks.items.len;

    // ── Collect back-edges per header ──
    // A back-edge is t → h where h dominates t. Both ends must be
    // reachable (have a non-null idom).
    var latches_by_header = std.AutoHashMap(ir.BlockId, std.ArrayList(ir.BlockId)).init(allocator);
    defer {
        var it = latches_by_header.iterator();
        while (it.next()) |entry| entry.value_ptr.deinit(allocator);
        latches_by_header.deinit();
    }

    var from_idx: usize = 0;
    while (from_idx < nblocks) : (from_idx += 1) {
        const from: ir.BlockId = @intCast(from_idx);
        if (dom.idom[from] == null) continue;
        const succs = successors.get(from) orelse continue;
        for (succs) |to| {
            if (dom.idom[to] == null) continue;
            if (!dom.dominates(to, from)) continue;
            const gop = try latches_by_header.getOrPut(to);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            // br_table / br_if may produce duplicate (from → to) edges.
            var dup = false;
            for (gop.value_ptr.items) |l| {
                if (l == from) {
                    dup = true;
                    break;
                }
            }
            if (!dup) try gop.value_ptr.append(allocator, from);
        }
    }

    // ── Materialize one Loop per header ──
    var loops: std.ArrayList(Loop) = .empty;
    errdefer {
        for (loops.items) |*loop| {
            allocator.free(loop.latches);
            allocator.free(loop.blocks);
        }
        loops.deinit(allocator);
    }

    var header_loop = std.AutoHashMap(ir.BlockId, u32).init(allocator);
    errdefer header_loop.deinit();

    // Scratch buffers reused across headers.
    var in_loop = try allocator.alloc(bool, nblocks);
    defer allocator.free(in_loop);
    var worklist: std.ArrayList(ir.BlockId) = .empty;
    defer worklist.deinit(allocator);

    var hit = latches_by_header.iterator();
    while (hit.next()) |entry| {
        const header = entry.key_ptr.*;
        const latch_list = entry.value_ptr.*;

        // Standard natural-loop body computation: start from header, add
        // each latch, then walk predecessors but never past the header.
        @memset(in_loop, false);
        in_loop[header] = true;
        worklist.clearRetainingCapacity();
        for (latch_list.items) |latch| {
            if (!in_loop[latch]) {
                in_loop[latch] = true;
                try worklist.append(allocator, latch);
            }
        }
        while (worklist.pop()) |bid| {
            const preds = predecessors.get(bid) orelse continue;
            for (preds) |p| {
                if (!in_loop[p]) {
                    in_loop[p] = true;
                    try worklist.append(allocator, p);
                }
            }
        }

        // Freeze in_loop → sorted blocks slice.
        var count: usize = 0;
        for (in_loop) |b| {
            if (b) count += 1;
        }
        const blocks_slice = try allocator.alloc(ir.BlockId, count);
        errdefer allocator.free(blocks_slice);
        var bi: usize = 0;
        for (in_loop, 0..) |b, idx| {
            if (b) {
                blocks_slice[bi] = @intCast(idx);
                bi += 1;
            }
        }

        // Sort latches ascending, no duplicates (duplicates already filtered above).
        const latches_slice = try allocator.dupe(ir.BlockId, latch_list.items);
        std.mem.sort(ir.BlockId, latches_slice, {}, std.sort.asc(ir.BlockId));

        try header_loop.put(header, @intCast(loops.items.len));
        try loops.append(allocator, .{
            .header = header,
            .latches = latches_slice,
            .blocks = blocks_slice,
        });
    }

    return .{
        .loops = try loops.toOwnedSlice(allocator),
        .header_loop = header_loop,
        .allocator = allocator,
    };
}

/// Compute per-block loop-nest depth: for each block `b`, the number of
/// natural loops (from `forest`) that contain `b`. Outer loops contribute
/// to the depth of every block they cover, so a block inside two nested
/// loops gets depth 2. Unreachable blocks and blocks outside any loop
/// have depth 0.
///
/// Saturates at `std.math.maxInt(u8)` — pathological CFGs with hundreds
/// of nesting levels would otherwise wrap.
///
/// Caller owns the returned slice (length `func.blocks.items.len`).
pub fn computeLoopDepthByBlock(
    func: *const ir.IrFunction,
    forest: *const LoopForest,
    allocator: std.mem.Allocator,
) ![]u8 {
    const nblocks = func.blocks.items.len;
    const depths = try allocator.alloc(u8, nblocks);
    @memset(depths, 0);
    for (forest.loops) |loop| {
        for (loop.blocks) |bid| {
            if (bid >= nblocks) continue;
            depths[bid] = std.math.add(u8, depths[bid], 1) catch std.math.maxInt(u8);
        }
    }
    return depths;
}

// ── Block frequency estimation ──────────────────────────────────────────

/// Maximum loop-depth amplification factor exponent. A block nested inside
/// `k` natural loops gets its base frequency multiplied by `10^min(k, MAX_LOOP_DEPTH_EXP)`.
/// Capped to keep the result well within `f32` range even for deeply
/// nested loops (10^6 still fits comfortably).
const MAX_LOOP_DEPTH_EXP: u8 = 6;

/// Per-loop trip count assumed by the heuristic. Mirrors what LLVM's
/// BlockFrequencyInfo uses as a default when no profile data is available.
const LOOP_TRIP_FACTOR: f32 = 10.0;

/// Estimate static execution frequency for every block of `func`,
/// relative to the entry block (which has frequency `1.0`).
///
/// The heuristic is a single-pass push-flow model:
///
///   1. Walk reachable blocks in reverse post-order from the entry.
///      Each block `b` distributes its frequency uniformly across its
///      forward-edge successors (`share = freq[b] / out_degree`).
///      Back-edges (those whose target dominates the source — i.e. the
///      back-edges of natural loops) are skipped so the flow is acyclic;
///      loop amplification is added in a second step.
///
///   2. For every natural loop, multiply `freq[b]` by `LOOP_TRIP_FACTOR`
///      for each loop that contains `b`. Loop depth is saturated at
///      `MAX_LOOP_DEPTH_EXP` to keep the values bounded.
///
/// Unreachable blocks (those with no incoming flow from the entry) keep
/// frequency `0.0`. The returned slice is indexed by `BlockId` and the
/// caller owns it.
///
/// This is a deliberately simple static heuristic — no edge profiling,
/// no branch-bias hints, no irreducible-cycle handling. It is intended
/// only as input to layout/cold-sinking passes; correctness of generated
/// code never depends on these numbers.
pub fn computeBlockFrequencies(
    func: *const ir.IrFunction,
    allocator: std.mem.Allocator,
) ![]f32 {
    const nblocks = func.blocks.items.len;
    const freq = try allocator.alloc(f32, nblocks);
    errdefer allocator.free(freq);
    @memset(freq, 0.0);
    if (nblocks == 0) return freq;

    // Successors are required to push flow.
    var successors = try buildSuccessors(func, allocator);
    defer {
        var sit = successors.iterator();
        while (sit.next()) |entry| allocator.free(entry.value_ptr.*);
        successors.deinit();
    }

    // Dominators identify back-edges (target dominates source).
    var dom = try computeDominators(func, allocator);
    defer dom.deinit();

    // Walk reachable blocks in reverse post-order. `dom.post_order` is
    // in DFS post-order (lowest index = first popped), so iterating it
    // back-to-front yields RPO with the entry first.
    freq[0] = 1.0;
    var i: usize = dom.post_order.len;
    while (i > 0) {
        i -= 1;
        const b = dom.post_order[i];
        const f = freq[b];
        if (f == 0.0) continue; // no flow to distribute

        const succs = successors.get(b) orelse continue;
        if (succs.len == 0) continue;

        // Count forward (non-back-edge) successors. A back-edge is one
        // whose target dominates the source.
        var forward_count: u32 = 0;
        for (succs) |s| {
            if (s >= nblocks) continue;
            if (dom.dominates(s, b)) continue; // back-edge
            forward_count += 1;
        }
        if (forward_count == 0) continue;

        const share = f / @as(f32, @floatFromInt(forward_count));
        for (succs) |s| {
            if (s >= nblocks) continue;
            if (dom.dominates(s, b)) continue; // back-edge
            freq[s] += share;
        }
    }

    // Apply loop-depth amplification. Each natural loop containing a
    // block multiplies its frequency by `LOOP_TRIP_FACTOR` (saturated
    // at `MAX_LOOP_DEPTH_EXP`).
    var lf = try computeLoops(func, &dom, allocator);
    defer lf.deinit();
    if (lf.loops.len > 0) {
        const depth = try allocator.alloc(u8, nblocks);
        defer allocator.free(depth);
        @memset(depth, 0);
        for (lf.loops) |*loop| {
            for (loop.blocks) |bid| {
                if (depth[bid] < MAX_LOOP_DEPTH_EXP) depth[bid] += 1;
            }
        }
        for (freq, 0..) |*f, idx| {
            if (f.* == 0.0) continue;
            var k: u8 = 0;
            while (k < depth[idx]) : (k += 1) f.* *= LOOP_TRIP_FACTOR;
        }
    }

    return freq;
}

// ── Tests ───────────────────────────────────────────────────────────────

test "buildSuccessors: linear block" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);
    try block0.append(.{ .op = .{ .ret = null } });

    var succs = try buildSuccessors(&func, allocator);
    defer {
        var it = succs.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        succs.deinit();
    }

    try std.testing.expectEqual(@as(usize, 0), succs.get(b0).?.len);
}

test "buildSuccessors: branch block" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const v0 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try block0.append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    var succs = try buildSuccessors(&func, allocator);
    defer {
        var it = succs.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        succs.deinit();
    }

    try std.testing.expectEqual(@as(usize, 2), succs.get(b0).?.len);
    try std.testing.expectEqual(@as(usize, 0), succs.get(b1).?.len);
}

test "computeLiveness: simple def-use in one block" {
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

    var liveness = try computeLiveness(&func, allocator);
    defer {
        var it = liveness.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.live_in.deinit();
            entry.value_ptr.live_out.deinit();
        }
        liveness.deinit();
    }

    // Nothing should be live_in at the entry block
    try std.testing.expectEqual(@as(u32, 0), liveness.get(b0).?.live_in.count());
    // Nothing should be live_out (ret terminates)
    try std.testing.expectEqual(@as(u32, 0), liveness.get(b0).?.live_out.count());
}

test "computeLiveRanges: basic ranges" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 }); // pos 0
    try block0.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 }); // pos 1
    try block0.append(.{ .op = .{ .add = .{ .lhs = v0, .rhs = v1 } }, .dest = v2 }); // pos 2
    try block0.append(.{ .op = .{ .ret = v2 } }); // pos 3

    const ranges = try computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);

    // Should have 3 live ranges (v0, v1, v2)
    try std.testing.expectEqual(@as(usize, 3), ranges.len);
    // Sorted by start: v0 at 0, v1 at 1, v2 at 2
    try std.testing.expectEqual(@as(u32, 0), ranges[0].start);
    try std.testing.expectEqual(@as(u32, 2), ranges[0].end); // v0 used at pos 2 (add)
    try std.testing.expectEqual(@as(u32, 1), ranges[1].start);
    try std.testing.expectEqual(@as(u32, 2), ranges[1].end); // v1 used at pos 2 (add)
    try std.testing.expectEqual(@as(u32, 2), ranges[2].start);
    try std.testing.expectEqual(@as(u32, 3), ranges[2].end); // v2 used at pos 3 (ret)
}

test "computeLiveRanges: v128 ranges retain type" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try block0.append(.{ .op = .{ .v128_const = 0xFFFF }, .dest = v0, .type = .v128 });
    try block0.append(.{ .op = .{ .v128_not = v0 }, .dest = v1, .type = .v128 });
    try block0.append(.{ .op = .{ .i32x4_extract_lane = .{ .vector = v1, .lane = 0 } }, .dest = v2, .type = .i32 });
    try block0.append(.{ .op = .{ .ret = v2 } });

    const ranges = try computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);

    try std.testing.expectEqual(@as(usize, 3), ranges.len);
    try std.testing.expectEqual(ir.IrType.v128, ranges[0].type);
    try std.testing.expectEqual(ir.IrType.v128, ranges[1].type);
    try std.testing.expectEqual(ir.IrType.i32, ranges[2].type);
    try std.testing.expectEqual(@as(u32, 1), ranges[0].end);
    try std.testing.expectEqual(@as(u32, 2), ranges[1].end);
}

test "computeLiveRanges: call with explicit args" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    const args = try allocator.alloc(ir.VReg, 2);
    args[0] = v0;
    args[1] = v1;
    try block0.append(.{ .op = .{ .iconst_32 = 10 }, .dest = v0 }); // pos 0
    try block0.append(.{ .op = .{ .iconst_32 = 20 }, .dest = v1 }); // pos 1
    try block0.append(.{ .op = .{ .call = .{ .func_idx = 0, .args = args } }, .dest = v2 }); // pos 2
    try block0.append(.{ .op = .{ .ret = v2 } }); // pos 3

    const ranges = try computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);

    try std.testing.expectEqual(@as(usize, 3), ranges.len);
    // v0: defined at 0, used at 2 (call arg)
    try std.testing.expectEqual(@as(u32, 0), ranges[0].start);
    try std.testing.expectEqual(@as(u32, 2), ranges[0].end);
    // v1: defined at 1, used at 2 (call arg)
    try std.testing.expectEqual(@as(u32, 1), ranges[1].start);
    try std.testing.expectEqual(@as(u32, 2), ranges[1].end);
}

test "computeLiveness: cross-block value is live" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 1, 1);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);

    const v0 = func.newVReg();
    try block0.append(.{ .op = .{ .iconst_32 = 42 }, .dest = v0 });
    try block0.append(.{ .op = .{ .local_set = .{ .idx = 0, .val = v0 } } });
    try block0.append(.{ .op = .{ .br = b1 } });

    const v1 = func.newVReg();
    try block1.append(.{ .op = .{ .local_get = 0 }, .dest = v1 });
    try block1.append(.{ .op = .{ .ret = v1 } });

    var liveness = try computeLiveness(&func, allocator);
    defer {
        var it = liveness.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.live_in.deinit();
            entry.value_ptr.live_out.deinit();
        }
        liveness.deinit();
    }

    // Block 0 should have nothing live_in (entry block)
    try std.testing.expectEqual(@as(u32, 0), liveness.get(b0).?.live_in.count());
}

const DiamondPhi = struct {
    b0: ir.BlockId,
    b1: ir.BlockId,
    b2: ir.BlockId,
    b3: ir.BlockId,
    b4: ir.BlockId,
    v_a: ir.VReg,
    v_cond: ir.VReg,
    v_b: ir.VReg,
    v_c: ir.VReg,
    v_phi: ir.VReg,
    v_res: ir.VReg,
};

/// 5-block diamond with a join phi:
///   b0: v_a=10; v_cond=1; br b1
///   b1: br_if v_cond -> b2, b3
///   b2: v_b=20; br b4
///   b3: v_c=30; br b4
///   b4: v_phi = phi[(b2,v_b),(b3,v_c)]; v_res = v_phi + v_a; ret v_res
fn buildDiamondPhi(func: *ir.IrFunction, allocator: std.mem.Allocator) !DiamondPhi {
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const b4 = try func.newBlock();

    const v_a = func.newVReg();
    const v_cond = func.newVReg();
    const v_b = func.newVReg();
    const v_c = func.newVReg();
    const v_phi = func.newVReg();
    const v_res = func.newVReg();

    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 10 }, .dest = v_a, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_cond, .type = .i32 });
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 }, .type = .void });

    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b2, .else_block = b3 } }, .type = .void });

    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 20 }, .dest = v_b, .type = .i32 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b4 }, .type = .void });

    try func.getBlock(b3).append(.{ .op = .{ .iconst_32 = 30 }, .dest = v_c, .type = .i32 });
    try func.getBlock(b3).append(.{ .op = .{ .br = b4 }, .type = .void });

    const edges = try allocator.dupe(ir.Inst.PhiEdge, &.{
        .{ .block = b2, .val = v_b },
        .{ .block = b3, .val = v_c },
    });
    try func.getBlock(b4).append(.{ .op = .{ .phi = edges }, .dest = v_phi, .type = .i32 });
    try func.getBlock(b4).append(.{ .op = .{ .add = .{ .lhs = v_phi, .rhs = v_a } }, .dest = v_res, .type = .i32 });
    try func.getBlock(b4).append(.{ .op = .{ .ret = v_res }, .type = .void });

    return .{ .b0 = b0, .b1 = b1, .b2 = b2, .b3 = b3, .b4 = b4, .v_a = v_a, .v_cond = v_cond, .v_b = v_b, .v_c = v_c, .v_phi = v_phi, .v_res = v_res };
}

fn findRange(ranges: []const LiveRange, vreg: ir.VReg) ?LiveRange {
    for (ranges) |r| if (r.vreg == vreg) return r;
    return null;
}

test "computeSsaLiveness: diamond phi — arms live only out of their own predecessor" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const d = try buildDiamondPhi(&func, allocator);

    var liveness = try computeSsaLiveness(&func, allocator);
    defer freeLiveness(&liveness);

    const lo_b2 = liveness.get(d.b2).?.live_out;
    const lo_b3 = liveness.get(d.b3).?.live_out;
    const li_b4 = liveness.get(d.b4).?.live_in;

    // Each arm is live-out of ITS predecessor only — the other branch's
    // value must NOT bleed in (the bug the legacy in-block handling has).
    try std.testing.expect(lo_b2.contains(d.v_b));
    try std.testing.expect(!lo_b2.contains(d.v_c));
    try std.testing.expect(lo_b3.contains(d.v_c));
    try std.testing.expect(!lo_b3.contains(d.v_b));

    // The join's live_in excludes both arms and the phi dest; only the
    // cross-diamond value v_a is live in.
    try std.testing.expect(li_b4.contains(d.v_a));
    try std.testing.expect(!li_b4.contains(d.v_b));
    try std.testing.expect(!li_b4.contains(d.v_c));
    try std.testing.expect(!li_b4.contains(d.v_phi));

    // The phi dest never leaks back into a predecessor's live_out.
    try std.testing.expect(!lo_b2.contains(d.v_phi));
    try std.testing.expect(!lo_b3.contains(d.v_phi));

    // v_a is live across the whole diamond.
    try std.testing.expect(liveness.get(d.b0).?.live_out.contains(d.v_a));
    try std.testing.expect(lo_b2.contains(d.v_a));
    try std.testing.expect(lo_b3.contains(d.v_a));
}

test "computeSsaLiveRanges: diamond phi reference ranges (#392 step 1)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const d = try buildDiamondPhi(&func, allocator);

    const ranges = try computeSsaLiveRanges(&func, null, allocator);
    defer allocator.free(ranges);

    // One interval per SSA value, each with a def.
    try std.testing.expectEqual(@as(usize, 6), ranges.len);

    // Global indices (sequential block order b0..b4):
    //   0 v_a=const   1 v_cond=const   2 br
    //   3 br_if
    //   4 v_b=const   5 br
    //   6 v_c=const   7 br
    //   8 v_phi=phi   9 v_res=add      10 ret
    const ra = findRange(ranges, d.v_a).?;
    try std.testing.expectEqual(@as(u32, 0), ra.start);
    try std.testing.expectEqual(@as(u32, 9), ra.end); // last use at the add

    const rcond = findRange(ranges, d.v_cond).?;
    try std.testing.expectEqual(@as(u32, 1), rcond.start);
    try std.testing.expectEqual(@as(u32, 3), rcond.end); // br_if

    // The crux: each phi arm ends at ITS predecessor's terminator, NOT at
    // the phi (index 8). v_b: def 4, live-out of b2 → b2 terminator = 5.
    const rb = findRange(ranges, d.v_b).?;
    try std.testing.expectEqual(@as(u32, 4), rb.start);
    try std.testing.expectEqual(@as(u32, 5), rb.end);

    const rc = findRange(ranges, d.v_c).?;
    try std.testing.expectEqual(@as(u32, 6), rc.start);
    try std.testing.expectEqual(@as(u32, 7), rc.end);

    const rphi = findRange(ranges, d.v_phi).?;
    try std.testing.expectEqual(@as(u32, 8), rphi.start);
    try std.testing.expectEqual(@as(u32, 9), rphi.end);

    const rres = findRange(ranges, d.v_res).?;
    try std.testing.expectEqual(@as(u32, 9), rres.start);
    try std.testing.expectEqual(@as(u32, 10), rres.end);
}

test "computeSsaLiveRanges vs legacy: phi arm not extended into the join block" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const d = try buildDiamondPhi(&func, allocator);

    const ssa = try computeSsaLiveRanges(&func, null, allocator);
    defer allocator.free(ssa);
    const legacy = try computeLiveRanges(&func, allocator);
    defer allocator.free(legacy);

    // Legacy counts the phi arm as an in-block use at the phi (index 8),
    // over-extending v_b's range into the join; SSA ends it at b2's
    // terminator (index 5).
    const ssa_b = findRange(ssa, d.v_b).?;
    const legacy_b = findRange(legacy, d.v_b).?;
    try std.testing.expectEqual(@as(u32, 5), ssa_b.end);
    try std.testing.expect(legacy_b.end > ssa_b.end);
}

test "computeLiveRanges: loop-carried parallel_copy dst spans the back-edge (#818)" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // CFG with a loop whose carried value `v_loop` is the dst of a
    // `parallel_copy` placed on the back-edge (latch) — i.e. defined LATE
    // in linear order — but used at the loop header (early). This mirrors
    // the post-#540 lowered shape where `coalescePhiLocalsToParallelCopy`
    // routes phi resolution through a register `parallel_copy` instead of a
    // frame round-trip.
    //
    //   b0 (entry):  br b1
    //   b1 (header): use v_loop ; br_if -> b2 (latch) | b3 (exit)
    //   b2 (latch):  parallel_copy (v_loop <- v_src) ; br b1   (back-edge)
    //   b3 (exit):   ret
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();

    const v_loop = func.newVReg();
    const v_cond = func.newVReg();
    const v_use = func.newVReg();
    const v_src = func.newVReg();

    // b0: pos0 br
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    // b1 (header): pos1 cond, pos2 use(v_loop), pos3 br_if
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_cond });
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = v_loop, .rhs = v_loop } }, .dest = v_use });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b2, .else_block = b3 } } });
    // b2 (latch): pos4 v_src, pos5 parallel_copy(v_loop<-v_src), pos6 br
    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_src });
    const copies = try allocator.dupe(ir.Inst.ParallelCopy, &.{.{ .dst = v_loop, .src = v_src, .ty = .i32 }});
    try func.getBlock(b2).append(.{ .op = .{ .parallel_copy = copies } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });
    // b3 (exit): ret
    try func.getBlock(b3).append(.{ .op = .{ .ret = v_use } });

    const ranges = try computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);

    const r_loop = findRange(ranges, v_loop).?;
    const header_use_pos: u32 = 2; // the `add` reading v_loop
    const latch_def_pos: u32 = 5; // the parallel_copy defining v_loop

    // Pre-#818 the builder set start = def_pos = latch (5) and only
    // extended `end`, collapsing the interval to ≈[5,6] and MISSING the
    // header use at pos 2 — the allocator would then reuse v_loop's
    // register inside the loop and clobber the loop-carried value. The fix
    // pulls `start` back to the earliest live-in, so the range must span
    // from at-or-before the header use through at-or-after the latch def.
    try std.testing.expect(r_loop.start <= header_use_pos);
    try std.testing.expect(r_loop.end >= latch_def_pos);
}

fn expectLivenessEqual(
    reference: *std.AutoHashMap(ir.BlockId, BlockLiveness),
    actual: *std.AutoHashMap(ir.BlockId, BlockLiveness),
) !void {
    try std.testing.expectEqual(reference.count(), actual.count());
    var it = reference.iterator();
    while (it.next()) |entry| {
        const bid = entry.key_ptr.*;
        const ref_bl = entry.value_ptr.*;
        const act_bl = actual.get(bid) orelse return error.MissingBlock;
        try std.testing.expectEqual(ref_bl.live_in.count(), act_bl.live_in.count());
        try std.testing.expectEqual(ref_bl.live_out.count(), act_bl.live_out.count());
        var iit = ref_bl.live_in.iterator();
        while (iit.next()) |e| try std.testing.expect(act_bl.live_in.contains(e.key_ptr.*));
        var oit = ref_bl.live_out.iterator();
        while (oit.next()) |e| try std.testing.expect(act_bl.live_out.contains(e.key_ptr.*));
    }
}

fn freeLiveness(liveness: *std.AutoHashMap(ir.BlockId, BlockLiveness)) void {
    var it = liveness.iterator();
    while (it.next()) |entry| {
        entry.value_ptr.live_in.deinit();
        entry.value_ptr.live_out.deinit();
    }
    liveness.deinit();
}

test "computeLiveness: worklist matches round-robin reference across CFG shapes" {
    const allocator = std.testing.allocator;

    // Each builder constructs a distinct CFG that stresses backward
    // liveness propagation: a diamond merge, a natural loop with a value
    // live across the backedge, a nested loop, and a long chain. The
    // worklist solver must agree with the reference round-robin solver on
    // live_in / live_out for every block.
    const builders = [_]*const fn (std.mem.Allocator) anyerror!ir.IrFunction{
        &buildDiamondLivenessFunc,
        &buildLoopLivenessFunc,
        &buildNestedLoopLivenessFunc,
        &buildChainLivenessFunc,
        &buildUnreachableBlockLivenessFunc,
        &buildUnreachableIntoReachableLivenessFunc,
        &buildMultiExitLivenessFunc,
        &buildSelfLoopLivenessFunc,
    };

    for (builders) |build| {
        var func = try build(allocator);
        defer func.deinit();

        var reference = try computeLivenessRoundRobin(&func, allocator);
        defer freeLiveness(&reference);
        var actual = try computeLiveness(&func, allocator);
        defer freeLiveness(&actual);

        try expectLivenessEqual(&reference, &actual);
    }
}

fn buildDiamondLivenessFunc(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    errdefer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const a = func.newVReg();
    const c = func.newVReg();
    const cond = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = a });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 2 }, .dest = c });
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    const t1 = func.newVReg();
    try func.getBlock(b1).append(.{ .op = .{ .add = .{ .lhs = a, .rhs = c } }, .dest = t1 });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    const t2 = func.newVReg();
    try func.getBlock(b2).append(.{ .op = .{ .sub = .{ .lhs = a, .rhs = c } }, .dest = t2 });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = a } });
    return func;
}

fn buildLoopLivenessFunc(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const header = try func.newBlock();
    const body = try func.newBlock();
    const exit = try func.newBlock();
    const acc = func.newVReg();
    const cond = func.newVReg();
    // acc defined in entry, used in body and exit (live across the loop).
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 10 }, .dest = acc });
    try func.getBlock(entry).append(.{ .op = .{ .br = header } });
    try func.getBlock(header).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(header).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = body, .else_block = exit } } });
    const t = func.newVReg();
    try func.getBlock(body).append(.{ .op = .{ .add = .{ .lhs = acc, .rhs = acc } }, .dest = t });
    try func.getBlock(body).append(.{ .op = .{ .br = header } }); // backedge
    try func.getBlock(exit).append(.{ .op = .{ .ret = acc } });
    return func;
}

fn buildNestedLoopLivenessFunc(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const outer = try func.newBlock();
    const inner = try func.newBlock();
    const inner_body = try func.newBlock();
    const outer_tail = try func.newBlock();
    const exit = try func.newBlock();
    const base = func.newVReg();
    const c1 = func.newVReg();
    const c2 = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 7 }, .dest = base });
    try func.getBlock(entry).append(.{ .op = .{ .br = outer } });
    try func.getBlock(outer).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c1 });
    try func.getBlock(outer).append(.{ .op = .{ .br_if = .{ .cond = c1, .then_block = inner, .else_block = exit } } });
    try func.getBlock(inner).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c2 });
    try func.getBlock(inner).append(.{ .op = .{ .br_if = .{ .cond = c2, .then_block = inner_body, .else_block = outer_tail } } });
    const t = func.newVReg();
    try func.getBlock(inner_body).append(.{ .op = .{ .add = .{ .lhs = base, .rhs = base } }, .dest = t });
    try func.getBlock(inner_body).append(.{ .op = .{ .br = inner } }); // inner backedge
    try func.getBlock(outer_tail).append(.{ .op = .{ .br = outer } }); // outer backedge
    try func.getBlock(exit).append(.{ .op = .{ .ret = base } });
    return func;
}

fn buildChainLivenessFunc(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    errdefer func.deinit();
    const v = func.newVReg();
    var prev = try func.newBlock();
    try func.getBlock(prev).append(.{ .op = .{ .iconst_32 = 3 }, .dest = v });
    // Long chain so the value is live across many blocks — the case where
    // reverse-index round-robin needs several sweeps but the worklist
    // converges directly. Both must agree.
    var i: u32 = 0;
    while (i < 12) : (i += 1) {
        const next = try func.newBlock();
        try func.getBlock(prev).append(.{ .op = .{ .br = next } });
        prev = next;
    }
    try func.getBlock(prev).append(.{ .op = .{ .ret = v } });
    return func;
}

fn buildUnreachableBlockLivenessFunc(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const exit = try func.newBlock();
    // Unreachable block with no predecessors; both solvers must still
    // compute its liveness from its own instructions.
    const unreachable_block = try func.newBlock();
    const v = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 5 }, .dest = v });
    try func.getBlock(entry).append(.{ .op = .{ .br = exit } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = v } });
    const w = func.newVReg();
    const t = func.newVReg();
    try func.getBlock(unreachable_block).append(.{ .op = .{ .iconst_32 = 9 }, .dest = w });
    try func.getBlock(unreachable_block).append(.{ .op = .{ .add = .{ .lhs = w, .rhs = w } }, .dest = t });
    try func.getBlock(unreachable_block).append(.{ .op = .{ .ret = t } });
    return func;
}

fn buildUnreachableIntoReachableLivenessFunc(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const mid = try func.newBlock();
    // Unreachable block that branches into a reachable block: `mid` lists it
    // as a predecessor, so when `mid.live_in` grows the worklist must
    // re-enqueue the unreachable block to grow its `live_out`.
    const unreachable_block = try func.newBlock();
    const a = func.newVReg();
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 3 }, .dest = a });
    try func.getBlock(entry).append(.{ .op = .{ .br = mid } });
    try func.getBlock(mid).append(.{ .op = .{ .ret = a } });
    const b = func.newVReg();
    try func.getBlock(unreachable_block).append(.{ .op = .{ .iconst_32 = 4 }, .dest = b });
    try func.getBlock(unreachable_block).append(.{ .op = .{ .br = mid } });
    return func;
}

fn buildMultiExitLivenessFunc(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const e1 = try func.newBlock();
    const e2 = try func.newBlock();
    const x = func.newVReg();
    const cond = func.newVReg();
    // x is live into e1 (ret x) but not e2 — the two exits diverge.
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 8 }, .dest = x });
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = e1, .else_block = e2 } } });
    try func.getBlock(e1).append(.{ .op = .{ .ret = x } });
    const y = func.newVReg();
    try func.getBlock(e2).append(.{ .op = .{ .iconst_32 = 2 }, .dest = y });
    try func.getBlock(e2).append(.{ .op = .{ .ret = y } });
    return func;
}

fn buildSelfLoopLivenessFunc(allocator: std.mem.Allocator) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, 0, 1, 0);
    errdefer func.deinit();
    const entry = try func.newBlock();
    const loop = try func.newBlock();
    const exit = try func.newBlock();
    const acc = func.newVReg();
    const cond = func.newVReg();
    const t = func.newVReg();
    // `loop` branches to itself; acc is live across the self-edge.
    try func.getBlock(entry).append(.{ .op = .{ .iconst_32 = 6 }, .dest = acc });
    try func.getBlock(entry).append(.{ .op = .{ .br = loop } });
    try func.getBlock(loop).append(.{ .op = .{ .add = .{ .lhs = acc, .rhs = acc } }, .dest = t });
    try func.getBlock(loop).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(loop).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = loop, .else_block = exit } } });
    try func.getBlock(exit).append(.{ .op = .{ .ret = acc } });
    return func;
}

test "buildSuccessors: loop backedge" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const block0 = func.getBlock(b0);
    const block1 = func.getBlock(b1);

    try block0.append(.{ .op = .{ .br = b1 } });
    const v0 = func.newVReg();
    try block1.append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try block1.append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b0, .else_block = b1 } } });

    var succs = try buildSuccessors(&func, allocator);
    defer {
        var it = succs.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        succs.deinit();
    }

    // Block 0 has successor b1
    try std.testing.expectEqual(@as(usize, 1), succs.get(b0).?.len);
    // Block 1 has successors b0 and b1 (loop)
    try std.testing.expectEqual(@as(usize, 2), succs.get(b1).?.len);
}

test "buildPredecessors: diamond" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // entry → {b1, b2} → b3
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    var preds = try buildPredecessors(&func, allocator);
    defer {
        var it = preds.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        preds.deinit();
    }

    try std.testing.expectEqual(@as(usize, 0), preds.get(b0).?.len);
    try std.testing.expectEqual(@as(usize, 1), preds.get(b1).?.len);
    try std.testing.expectEqual(@as(usize, 1), preds.get(b2).?.len);
    try std.testing.expectEqual(@as(usize, 2), preds.get(b3).?.len);
}

fn expectDomTreeEqual(expected: *const DomTree, actual: *const DomTree) !void {
    try std.testing.expectEqual(expected.idom.len, actual.idom.len);
    for (expected.idom, actual.idom) |e, a| try std.testing.expectEqual(e, a);
    try std.testing.expectEqual(expected.post_num.len, actual.post_num.len);
    for (expected.post_num, actual.post_num) |e, a| try std.testing.expectEqual(e, a);
    try std.testing.expectEqual(expected.post_order.len, actual.post_order.len);
    for (expected.post_order, actual.post_order) |e, a| try std.testing.expectEqual(e, a);
}

test "CfgAnalysisCache: predecessor cache deduplicates repeated br_table targets" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const idx = func.newVReg();
    const targets = try allocator.dupe(ir.BlockId, &.{ b1, b1 });
    try func.owned_br_table_targets.append(allocator, targets);
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 0 }, .dest = idx });
    try func.getBlock(b0).append(.{ .op = .{ .br_table = .{ .index = idx, .targets = targets, .default = b1 } } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });

    var cache = CfgAnalysisCache.init(allocator);
    defer cache.deinit();

    const preds = try cache.getPredecessors(&func);
    try std.testing.expectEqual(@as(usize, 1), preds.get(b1).?.len);
    try std.testing.expectEqual(b0, preds.get(b1).?[0]);
}

test "CfgAnalysisCache: invalidate keeps dominators after non-CFG edit" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const cond = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    var cache = CfgAnalysisCache.init(allocator);
    defer cache.deinit();

    const dom_before = try cache.getDominators(&func);
    const idom_ptr = dom_before.idom.ptr;
    const post_num_ptr = dom_before.post_num.ptr;
    const post_order_ptr = dom_before.post_order.ptr;

    func.getBlock(b0).instructions.items[0].op = .{ .iconst_32 = 7 };
    cache.invalidate();
    try std.testing.expect(cache.dirty);

    const dom_after = try cache.getDominators(&func);
    try std.testing.expect(!cache.dirty);
    try std.testing.expect(idom_ptr == dom_after.idom.ptr);
    try std.testing.expect(post_num_ptr == dom_after.post_num.ptr);
    try std.testing.expect(post_order_ptr == dom_after.post_order.ptr);

    var fresh = try computeDominators(&func, allocator);
    defer fresh.deinit();
    try expectDomTreeEqual(&fresh, dom_after);
}

test "CfgAnalysisCache: invalidate refreshes dominators after branch retarget" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const cond = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    var cache = CfgAnalysisCache.init(allocator);
    defer cache.deinit();

    const dom_before = try cache.getDominators(&func);
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom_before.idom[b3]);

    func.getBlock(b0).instructions.items[1].op.br_if.else_block = b1;
    cache.invalidate();
    try std.testing.expect(cache.dirty);

    const dom_after = try cache.getDominators(&func);
    try std.testing.expect(!cache.dirty);
    try std.testing.expectEqual(@as(?ir.BlockId, b1), dom_after.idom[b3]);

    var fresh = try computeDominators(&func, allocator);
    defer fresh.deinit();
    try expectDomTreeEqual(&fresh, dom_after);
}

test "computeDominators: linear chain" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b2 } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();

    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b0]);
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b1]);
    try std.testing.expectEqual(@as(?ir.BlockId, b1), dom.idom[b2]);
    try std.testing.expect(dom.dominates(b0, b2));
    try std.testing.expect(dom.dominates(b1, b2));
    try std.testing.expect(!dom.dominates(b2, b0));
}

test "computeDominators: diamond idom is entry for merge" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();

    // Entry dominates itself.
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b0]);
    // Sides' idom is entry.
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b1]);
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b2]);
    // Merge's idom is entry (neither side dominates the merge).
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b3]);

    try std.testing.expect(dom.dominates(b0, b3));
    try std.testing.expect(!dom.dominates(b1, b3));
    try std.testing.expect(!dom.dominates(b2, b3));
}

test "computeDominators: simple loop" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // b0 → b1 → (self-loop or to b2)
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();

    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b0]);
    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b1]);
    // b1 is the only predecessor of b2 that is reachable; b1 dominates b2.
    try std.testing.expectEqual(@as(?ir.BlockId, b1), dom.idom[b2]);
    try std.testing.expect(dom.dominates(b0, b2));
    try std.testing.expect(dom.dominates(b1, b2));
}

test "computeDominators: unreachable block has null idom" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock(); // unreachable
    try func.getBlock(b0).append(.{ .op = .{ .ret = null } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();

    try std.testing.expectEqual(@as(?ir.BlockId, b0), dom.idom[b0]);
    try std.testing.expectEqual(@as(?ir.BlockId, null), dom.idom[b1]);
    try std.testing.expect(!dom.dominates(b0, b1));
}

test "computeLoops: no loops in DAG" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 0), lf.loops.len);
    try std.testing.expect(!lf.isHeader(b0));
    try std.testing.expect(!lf.isHeader(b1));
}

test "computeLoops: self-loop" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    const b1 = try func.newBlock(); // self-loop
    const b2 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 1), lf.loops.len);
    const loop = &lf.loops[0];
    try std.testing.expectEqual(b1, loop.header);
    try std.testing.expectEqual(@as(usize, 1), loop.latches.len);
    try std.testing.expectEqual(b1, loop.latches[0]);
    try std.testing.expect(loop.containsBlock(b1));
    try std.testing.expect(!loop.containsBlock(b0));
    try std.testing.expect(!loop.containsBlock(b2));
    try std.testing.expect(lf.isHeader(b1));
}

test "computeLoops: while-loop body" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // b0 → b1(header) → b2(body) → b1 (back-edge) | b3(exit)
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b2, .else_block = b3 } } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 1), lf.loops.len);
    const loop = &lf.loops[0];
    try std.testing.expectEqual(b1, loop.header);
    try std.testing.expectEqual(@as(usize, 2), loop.blocks.len);
    try std.testing.expect(loop.containsBlock(b1));
    try std.testing.expect(loop.containsBlock(b2));
    try std.testing.expect(!loop.containsBlock(b0));
    try std.testing.expect(!loop.containsBlock(b3));
    try std.testing.expectEqual(@as(usize, 1), loop.latches.len);
    try std.testing.expectEqual(b2, loop.latches[0]);
}

test "computeLoops: multiple latches share header" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // b0 → h; h branches to b2 or b3; both jump back to h; h has exit b4.
    const b0 = try func.newBlock();
    const h = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const b4 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = h } });
    try func.getBlock(h).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(h).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b2, .else_block = b3 } } });
    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1 });
    try func.getBlock(b2).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = h, .else_block = b4 } } });
    try func.getBlock(b3).append(.{ .op = .{ .br = h } });
    try func.getBlock(b4).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 1), lf.loops.len);
    const loop = &lf.loops[0];
    try std.testing.expectEqual(h, loop.header);
    try std.testing.expectEqual(@as(usize, 2), loop.latches.len);
    // Sorted ascending.
    try std.testing.expectEqual(b2, loop.latches[0]);
    try std.testing.expectEqual(b3, loop.latches[1]);
    try std.testing.expect(loop.containsBlock(h));
    try std.testing.expect(loop.containsBlock(b2));
    try std.testing.expect(loop.containsBlock(b3));
    try std.testing.expect(!loop.containsBlock(b0));
    try std.testing.expect(!loop.containsBlock(b4));
}

test "computeLoops: nested loops" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // outer: h_o → h_i → body → h_i (inner back-edge) | exit_i → h_o (outer back-edge) | exit_o
    const b0 = try func.newBlock();
    const h_o = try func.newBlock();
    const h_i = try func.newBlock();
    const body = try func.newBlock();
    const exit_i = try func.newBlock();
    const exit_o = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = h_o } });
    try func.getBlock(h_o).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(h_o).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = h_i, .else_block = exit_o } } });
    try func.getBlock(h_i).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1 });
    try func.getBlock(h_i).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = body, .else_block = exit_i } } });
    try func.getBlock(body).append(.{ .op = .{ .br = h_i } });
    try func.getBlock(exit_i).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v2 });
    try func.getBlock(exit_i).append(.{ .op = .{ .br_if = .{ .cond = v2, .then_block = h_o, .else_block = exit_o } } });
    try func.getBlock(exit_o).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    try std.testing.expectEqual(@as(usize, 2), lf.loops.len);
    try std.testing.expect(lf.isHeader(h_o));
    try std.testing.expect(lf.isHeader(h_i));

    const outer = &lf.loops[lf.header_loop.get(h_o).?];
    const inner = &lf.loops[lf.header_loop.get(h_i).?];

    // Inner loop: h_i and body only.
    try std.testing.expectEqual(@as(usize, 2), inner.blocks.len);
    try std.testing.expect(inner.containsBlock(h_i));
    try std.testing.expect(inner.containsBlock(body));

    // Outer loop contains all inner blocks plus h_o and exit_i.
    try std.testing.expect(outer.containsBlock(h_o));
    try std.testing.expect(outer.containsBlock(h_i));
    try std.testing.expect(outer.containsBlock(body));
    try std.testing.expect(outer.containsBlock(exit_i));
    try std.testing.expect(!outer.containsBlock(exit_o));
    try std.testing.expect(!outer.containsBlock(b0));
}

test "computeLoops: irreducible-ish (no back-edge without dominator) produces no loops" {
    // Ensure we don't spuriously report a loop when CFG has a cycle but
    // no edge target dominates its source (dominator-based natural-loop
    // detection ignores irreducible cycles by design).
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // b0 → {b1, b2}; b1 → b2; b2 → b1. Neither b1 nor b2 dominates
    // the other because both are entered directly from b0.
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1 });
    try func.getBlock(b1).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = b2, .else_block = b0 } } });
    try func.getBlock(b2).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v2 });
    try func.getBlock(b2).append(.{ .op = .{ .br_if = .{ .cond = v2, .then_block = b1, .else_block = b0 } } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var lf = try computeLoops(&func, &dom, allocator);
    defer lf.deinit();

    // Only b0 dominates b1 and b2. There's an edge b1→b0 and b2→b0
    // (b0 dominates both), so those are back-edges ⇒ one natural loop
    // headed at b0. The cycle b1↔b2 is irreducible and not reported.
    try std.testing.expectEqual(@as(usize, 1), lf.loops.len);
    try std.testing.expectEqual(b0, lf.loops[0].header);
    try std.testing.expectEqual(@as(usize, 3), lf.loops[0].blocks.len);
}

// ── Loop-depth annotation on live ranges (issue #382) ──────────────────

test "computeLoopDepthByBlock: nested loop yields depth 2 in body" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // outer header → inner header → body → inner header → outer header.
    const b0 = try func.newBlock();
    const h_o = try func.newBlock();
    const h_i = try func.newBlock();
    const body = try func.newBlock();
    const exit_i = try func.newBlock();
    const exit_o = try func.newBlock();
    const c_o = func.newVReg();
    const c_i = func.newVReg();
    const c_e = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = h_o } });
    try func.getBlock(h_o).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c_o });
    try func.getBlock(h_o).append(.{ .op = .{ .br_if = .{ .cond = c_o, .then_block = h_i, .else_block = exit_o } } });
    try func.getBlock(h_i).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c_i });
    try func.getBlock(h_i).append(.{ .op = .{ .br_if = .{ .cond = c_i, .then_block = body, .else_block = exit_i } } });
    try func.getBlock(body).append(.{ .op = .{ .br = h_i } });
    try func.getBlock(exit_i).append(.{ .op = .{ .iconst_32 = 1 }, .dest = c_e });
    try func.getBlock(exit_i).append(.{ .op = .{ .br_if = .{ .cond = c_e, .then_block = h_o, .else_block = exit_o } } });
    try func.getBlock(exit_o).append(.{ .op = .{ .ret = null } });

    var dom = try computeDominators(&func, allocator);
    defer dom.deinit();
    var forest = try computeLoops(&func, &dom, allocator);
    defer forest.deinit();
    const depths = try computeLoopDepthByBlock(&func, &forest, allocator);
    defer allocator.free(depths);

    try std.testing.expectEqual(@as(u8, 0), depths[b0]);
    try std.testing.expectEqual(@as(u8, 1), depths[h_o]);
    try std.testing.expectEqual(@as(u8, 2), depths[h_i]);
    try std.testing.expectEqual(@as(u8, 2), depths[body]);
    try std.testing.expectEqual(@as(u8, 1), depths[exit_i]);
    try std.testing.expectEqual(@as(u8, 0), depths[exit_o]);
}

test "computeLiveRanges: max_loop_depth annotated for loop body vreg" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    // Single-block self-loop: vreg defined inside the loop must end up
    // with depth 1. A vreg defined before the loop and dead after it
    // would not get depth — we only need to confirm the inside-loop
    // case here.
    const b0 = try func.newBlock();
    const b_hdr = try func.newBlock();
    const b_exit = try func.newBlock();
    const v_cond = func.newVReg();
    const v_loop = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = b_hdr } });
    try func.getBlock(b_hdr).append(.{ .op = .{ .iconst_32 = 7 }, .dest = v_loop });
    try func.getBlock(b_hdr).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v_cond });
    try func.getBlock(b_hdr).append(.{ .op = .{ .br_if = .{ .cond = v_cond, .then_block = b_hdr, .else_block = b_exit } } });
    try func.getBlock(b_exit).append(.{ .op = .{ .ret = null } });

    const ranges = try computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);

    // Find vreg defined inside the loop. The header is at depth 1 in
    // its own natural loop.
    var seen = false;
    for (ranges) |r| {
        if (r.vreg == v_loop) {
            try std.testing.expectEqual(@as(u8, 1), r.max_loop_depth);
            seen = true;
        }
    }
    try std.testing.expect(seen);
}

test "computeLiveRanges: max_loop_depth is 0 for ranges outside any loop" {
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

    const ranges = try computeLiveRanges(&func, allocator);
    defer allocator.free(ranges);

    for (ranges) |r| {
        try std.testing.expectEqual(@as(u8, 0), r.max_loop_depth);
    }
}

test "computeBlockFrequencies: single block has frequency 1.0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();

    const b0 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .ret = null } });

    const freq = try computeBlockFrequencies(&func, allocator);
    defer allocator.free(freq);

    try std.testing.expectEqual(@as(usize, 1), freq.len);
    try std.testing.expectEqual(@as(f32, 1.0), freq[0]);
}

test "computeBlockFrequencies: linear chain preserves frequency" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b2 } });
    try func.getBlock(b2).append(.{ .op = .{ .ret = null } });

    const freq = try computeBlockFrequencies(&func, allocator);
    defer allocator.free(freq);

    try std.testing.expectEqual(@as(f32, 1.0), freq[0]);
    try std.testing.expectEqual(@as(f32, 1.0), freq[1]);
    try std.testing.expectEqual(@as(f32, 1.0), freq[2]);
}

test "computeBlockFrequencies: diamond splits flow then rejoins" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    const b2 = try func.newBlock();
    const b3 = try func.newBlock();
    const cond = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .iconst_32 = 1 }, .dest = cond });
    try func.getBlock(b0).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = b1, .else_block = b2 } } });
    try func.getBlock(b1).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b2).append(.{ .op = .{ .br = b3 } });
    try func.getBlock(b3).append(.{ .op = .{ .ret = null } });

    const freq = try computeBlockFrequencies(&func, allocator);
    defer allocator.free(freq);

    try std.testing.expectEqual(@as(f32, 1.0), freq[0]);
    try std.testing.expectEqual(@as(f32, 0.5), freq[1]);
    try std.testing.expectEqual(@as(f32, 0.5), freq[2]);
    try std.testing.expectEqual(@as(f32, 1.0), freq[3]);
}

test "computeBlockFrequencies: while-loop body is hotter than entry" {
    // b0 → h; h ⇄ body; h → exit.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const h = try func.newBlock();
    const body = try func.newBlock();
    const exit_b = try func.newBlock();
    const v0 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = h } });
    try func.getBlock(h).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(h).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = body, .else_block = exit_b } } });
    try func.getBlock(body).append(.{ .op = .{ .br = h } });
    try func.getBlock(exit_b).append(.{ .op = .{ .ret = null } });

    const freq = try computeBlockFrequencies(&func, allocator);
    defer allocator.free(freq);

    try std.testing.expectEqual(@as(f32, 1.0), freq[b0]);
    // Header gets full inflow from b0; loop factor 10 applied once.
    try std.testing.expectEqual(@as(f32, 10.0), freq[h]);
    // Body receives half of header's pre-loop-factor flow (0.5), then ×10.
    try std.testing.expectEqual(@as(f32, 5.0), freq[body]);
    // Exit is cold (no loop nesting).
    try std.testing.expectEqual(@as(f32, 0.5), freq[exit_b]);
    // Entry < header.
    try std.testing.expect(freq[b0] < freq[h]);
    // Body hotter than entry.
    try std.testing.expect(freq[body] > freq[b0]);
}

test "computeBlockFrequencies: nested loops multiply" {
    // Outer header h_o; inner header h_i nested inside; body inside inner.
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const h_o = try func.newBlock();
    const h_i = try func.newBlock();
    const body = try func.newBlock();
    const exit_i = try func.newBlock();
    const exit_o = try func.newBlock();
    const v0 = func.newVReg();
    const v1 = func.newVReg();
    const v2 = func.newVReg();
    try func.getBlock(b0).append(.{ .op = .{ .br = h_o } });
    try func.getBlock(h_o).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v0 });
    try func.getBlock(h_o).append(.{ .op = .{ .br_if = .{ .cond = v0, .then_block = h_i, .else_block = exit_o } } });
    try func.getBlock(h_i).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v1 });
    try func.getBlock(h_i).append(.{ .op = .{ .br_if = .{ .cond = v1, .then_block = body, .else_block = exit_i } } });
    try func.getBlock(body).append(.{ .op = .{ .br = h_i } });
    try func.getBlock(exit_i).append(.{ .op = .{ .iconst_32 = 1 }, .dest = v2 });
    try func.getBlock(exit_i).append(.{ .op = .{ .br_if = .{ .cond = v2, .then_block = h_o, .else_block = exit_o } } });
    try func.getBlock(exit_o).append(.{ .op = .{ .ret = null } });

    const freq = try computeBlockFrequencies(&func, allocator);
    defer allocator.free(freq);

    // body lies inside both loops ⇒ 100× amplification applied to its
    // pre-amplification inflow (≈0.25), while h_o sees only 10×. The
    // resulting ratio is dominated by the depth difference.
    try std.testing.expect(freq[body] > freq[h_o] * 2.0);
    // h_i lies inside the outer loop and is itself a header ⇒ depth 2,
    // hotter than h_o (depth 1).
    try std.testing.expect(freq[h_i] > freq[h_o] * 2.0);
    // h_o is hotter than its preheader b0.
    try std.testing.expect(freq[h_o] > freq[0]);
    // exit_o is cold (outside every loop) and stays close to entry.
    try std.testing.expect(freq[exit_o] < freq[h_o]);
    try std.testing.expect(freq[exit_o] < 2.0);
}

test "computeBlockFrequencies: unreachable block stays at 0" {
    const allocator = std.testing.allocator;
    var func = ir.IrFunction.init(allocator, 0, 0, 0);
    defer func.deinit();
    const b0 = try func.newBlock();
    const b1 = try func.newBlock();
    _ = try func.newBlock(); // b2: unreachable, no edges to it
    try func.getBlock(b0).append(.{ .op = .{ .br = b1 } });
    try func.getBlock(b1).append(.{ .op = .{ .ret = null } });

    const freq = try computeBlockFrequencies(&func, allocator);
    defer allocator.free(freq);

    try std.testing.expectEqual(@as(f32, 1.0), freq[0]);
    try std.testing.expectEqual(@as(f32, 1.0), freq[1]);
    try std.testing.expectEqual(@as(f32, 0.0), freq[2]);
}
