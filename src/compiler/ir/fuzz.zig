//! Deterministic IR generators for optimizer property tests (#736).
//!
//! The generators intentionally stay within the scalar subset supported by
//! `interp.zig`: i32/i64 integer ops, locals/globals, memory, branches, phis,
//! and mock calls as barriers. Every generated function is expected to pass the
//! verifier before it is used as an optimizer oracle input.

const std = @import("std");
const ir = @import("ir.zig");
const interp = @import("interp.zig");
const verifier = @import("verifier.zig");

pub const Shape = enum {
    linear,
    diamond,
    nested_diamond,
    counted_loop,
    memory_barrier,
    /// #794: a load that dominates a loop whose body reloads and then stores
    /// the same address. Exercises the #743 load-forwarding-across-a-loop-effect
    /// bug class with an execution-observable divergence (the in-loop store
    /// changes the value an unsound forward would staleify).
    loop_forwarded_load,

    pub const all = [_]Shape{ .linear, .diamond, .nested_diamond, .counted_loop, .memory_barrier, .loop_forwarded_load };
};

pub const Case = struct {
    seed: u64,
    shape: Shape,
    func: ir.IrFunction,
    inputs: []interp.Value,
    memory: []u8,

    pub fn deinit(self: *Case, allocator: std.mem.Allocator) void {
        self.func.deinit();
        allocator.free(self.inputs);
        allocator.free(self.memory);
        self.* = undefined;
    }
};

pub fn generate(allocator: std.mem.Allocator, seed: u64, shape: Shape) !Case {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    return switch (shape) {
        .linear => generateLinear(allocator, seed, random),
        .diamond => generateDiamond(allocator, seed, random),
        .nested_diamond => generateNestedDiamond(allocator, seed, random),
        .counted_loop => generateCountedLoop(allocator, seed, random),
        .memory_barrier => generateMemoryBarrier(allocator, seed, random),
        .loop_forwarded_load => generateLoopForwardedLoad(allocator, seed, random),
    };
}

fn initFunc(allocator: std.mem.Allocator, param_count: u32, result_count: u32, local_count: u32) !ir.IrFunction {
    var func = ir.IrFunction.init(allocator, param_count, result_count, local_count);
    errdefer func.deinit();
    const local_types = try allocator.alloc(ir.IrType, local_count);
    @memset(local_types, .i32);
    func.local_types = local_types;
    var i: u32 = 0;
    while (i < param_count) : (i += 1) _ = func.newVReg();
    return func;
}

fn makeInputs(allocator: std.mem.Allocator, random: std.Random, n: usize) ![]interp.Value {
    const inputs = try allocator.alloc(interp.Value, n);
    for (inputs) |*v| {
        // Keep values small to avoid accidental OOB addresses in generated IR;
        // arithmetic still wraps once passes start rewriting expressions.
        v.* = interp.Value.u32v(random.intRangeLessThan(u32, 0, 16));
    }
    return inputs;
}

fn makeMemory(allocator: std.mem.Allocator, random: std.Random) ![]u8 {
    const memory = try allocator.alloc(u8, 64);
    for (memory) |*b| b.* = random.int(u8);
    return memory;
}

fn finishCase(
    allocator: std.mem.Allocator,
    seed: u64,
    shape: Shape,
    func: ir.IrFunction,
    inputs: []interp.Value,
    memory: []u8,
) !Case {
    try verifier.verifyFunction(&func, 0, .after_each_pass, allocator);
    return .{ .seed = seed, .shape = shape, .func = func, .inputs = inputs, .memory = memory };
}

fn generateLinear(allocator: std.mem.Allocator, seed: u64, random: std.Random) !Case {
    var func = try initFunc(allocator, 2, 1, 4);
    errdefer func.deinit();
    const inputs = try makeInputs(allocator, random, 2);
    errdefer allocator.free(inputs);
    const memory = try makeMemory(allocator, random);
    errdefer allocator.free(memory);

    const b0 = try func.newBlock();
    const k = func.newVReg();
    const sum = func.newVReg();
    const mix = func.newVReg();
    const shifted = func.newVReg();
    const reloaded = func.newVReg();
    const out = func.newVReg();
    try func.getBlock(b0).append(.{ .dest = k, .type = .i32, .op = .{ .iconst_32 = @intCast(random.intRangeLessThan(u32, 1, 8)) } });
    try func.getBlock(b0).append(.{ .dest = sum, .type = .i32, .op = .{ .add = .{ .lhs = 0, .rhs = k } } });
    try func.getBlock(b0).append(.{ .dest = mix, .type = .i32, .op = .{ .xor = .{ .lhs = sum, .rhs = 1 } } });
    try func.getBlock(b0).append(.{ .op = .{ .local_set = .{ .idx = 2, .val = mix } } });
    try func.getBlock(b0).append(.{ .dest = reloaded, .type = .i32, .op = .{ .local_get = 2 } });
    try func.getBlock(b0).append(.{ .dest = shifted, .type = .i32, .op = .{ .shl = .{ .lhs = reloaded, .rhs = k } } });
    try func.getBlock(b0).append(.{ .dest = out, .type = .i32, .op = .{ .@"or" = .{ .lhs = shifted, .rhs = reloaded } } });
    try func.getBlock(b0).append(.{ .op = .{ .ret = out } });

    return finishCase(allocator, seed, .linear, func, inputs, memory);
}

fn generateDiamond(allocator: std.mem.Allocator, seed: u64, random: std.Random) !Case {
    var func = try initFunc(allocator, 2, 1, 3);
    errdefer func.deinit();
    const inputs = try makeInputs(allocator, random, 2);
    errdefer allocator.free(inputs);
    const memory = try makeMemory(allocator, random);
    errdefer allocator.free(memory);

    const entry = try func.newBlock();
    const left = try func.newBlock();
    const right = try func.newBlock();
    const merge = try func.newBlock();
    try func.getBlock(left).addPredecessor(entry);
    try func.getBlock(right).addPredecessor(entry);
    try func.getBlock(merge).addPredecessor(left);
    try func.getBlock(merge).addPredecessor(right);

    const cond = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = cond, .type = .i32, .op = .{ .eqz = 0 } });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = left, .else_block = right } } });

    const left_v = func.newVReg();
    try func.getBlock(left).append(.{ .dest = left_v, .type = .i32, .op = .{ .add = .{ .lhs = 0, .rhs = 1 } } });
    try func.getBlock(left).append(.{ .op = .{ .br = merge } });

    const right_v = func.newVReg();
    try func.getBlock(right).append(.{ .dest = right_v, .type = .i32, .op = .{ .sub = .{ .lhs = 0, .rhs = 1 } } });
    try func.getBlock(right).append(.{ .op = .{ .br = merge } });

    const edges = try allocator.dupe(ir.Inst.PhiEdge, &.{ .{ .block = left, .val = left_v }, .{ .block = right, .val = right_v } });
    const phi = func.newVReg();
    const out = func.newVReg();
    try func.getBlock(merge).append(.{ .dest = phi, .type = .i32, .op = .{ .phi = edges } });
    try func.getBlock(merge).append(.{ .dest = out, .type = .i32, .op = .{ .xor = .{ .lhs = phi, .rhs = 1 } } });
    try func.getBlock(merge).append(.{ .op = .{ .ret = out } });

    return finishCase(allocator, seed, .diamond, func, inputs, memory);
}

fn generateNestedDiamond(allocator: std.mem.Allocator, seed: u64, random: std.Random) !Case {
    var func = try initFunc(allocator, 2, 1, 3);
    errdefer func.deinit();
    const inputs = try makeInputs(allocator, random, 2);
    errdefer allocator.free(inputs);
    const memory = try makeMemory(allocator, random);
    errdefer allocator.free(memory);

    const entry = try func.newBlock();
    const outer_then = try func.newBlock();
    const outer_else = try func.newBlock();
    const inner_left = try func.newBlock();
    const inner_right = try func.newBlock();
    const inner_merge = try func.newBlock();
    const final_merge = try func.newBlock();
    try func.getBlock(outer_then).addPredecessor(entry);
    try func.getBlock(outer_else).addPredecessor(entry);
    try func.getBlock(inner_left).addPredecessor(outer_then);
    try func.getBlock(inner_right).addPredecessor(outer_then);
    try func.getBlock(inner_merge).addPredecessor(inner_left);
    try func.getBlock(inner_merge).addPredecessor(inner_right);
    try func.getBlock(final_merge).addPredecessor(inner_merge);
    try func.getBlock(final_merge).addPredecessor(outer_else);

    const outer_cond = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = outer_cond, .type = .i32, .op = .{ .@"and" = .{ .lhs = 0, .rhs = 1 } } });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = outer_cond, .then_block = outer_then, .else_block = outer_else } } });

    const inner_cond = func.newVReg();
    try func.getBlock(outer_then).append(.{ .dest = inner_cond, .type = .i32, .op = .{ .eqz = 1 } });
    try func.getBlock(outer_then).append(.{ .op = .{ .br_if = .{ .cond = inner_cond, .then_block = inner_left, .else_block = inner_right } } });

    const left_v = func.newVReg();
    try func.getBlock(inner_left).append(.{ .dest = left_v, .type = .i32, .op = .{ .mul = .{ .lhs = 0, .rhs = 1 } } });
    try func.getBlock(inner_left).append(.{ .op = .{ .br = inner_merge } });

    const right_v = func.newVReg();
    try func.getBlock(inner_right).append(.{ .dest = right_v, .type = .i32, .op = .{ .xor = .{ .lhs = 0, .rhs = 1 } } });
    try func.getBlock(inner_right).append(.{ .op = .{ .br = inner_merge } });

    const inner_edges = try allocator.dupe(ir.Inst.PhiEdge, &.{ .{ .block = inner_left, .val = left_v }, .{ .block = inner_right, .val = right_v } });
    const inner_phi = func.newVReg();
    try func.getBlock(inner_merge).append(.{ .dest = inner_phi, .type = .i32, .op = .{ .phi = inner_edges } });
    try func.getBlock(inner_merge).append(.{ .op = .{ .br = final_merge } });

    const else_v = func.newVReg();
    try func.getBlock(outer_else).append(.{ .dest = else_v, .type = .i32, .op = .{ .@"or" = .{ .lhs = 0, .rhs = 1 } } });
    try func.getBlock(outer_else).append(.{ .op = .{ .br = final_merge } });

    const final_edges = try allocator.dupe(ir.Inst.PhiEdge, &.{ .{ .block = inner_merge, .val = inner_phi }, .{ .block = outer_else, .val = else_v } });
    const final_phi = func.newVReg();
    try func.getBlock(final_merge).append(.{ .dest = final_phi, .type = .i32, .op = .{ .phi = final_edges } });
    try func.getBlock(final_merge).append(.{ .op = .{ .ret = final_phi } });

    return finishCase(allocator, seed, .nested_diamond, func, inputs, memory);
}

fn generateCountedLoop(allocator: std.mem.Allocator, seed: u64, random: std.Random) !Case {
    var func = try initFunc(allocator, 1, 1, 2);
    errdefer func.deinit();
    const inputs = try makeInputs(allocator, random, 1);
    errdefer allocator.free(inputs);
    const memory = try makeMemory(allocator, random);
    errdefer allocator.free(memory);

    const bound_value: i32 = @intCast(random.intRangeLessThan(u32, 1, 6));
    const entry = try func.newBlock();
    const header = try func.newBlock();
    const body = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(header).addPredecessor(entry);
    try func.getBlock(header).addPredecessor(body);
    try func.getBlock(body).addPredecessor(header);
    try func.getBlock(exit).addPredecessor(header);

    const zero = func.newVReg();
    const bound = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = zero, .type = .i32, .op = .{ .iconst_32 = 0 } });
    try func.getBlock(entry).append(.{ .dest = bound, .type = .i32, .op = .{ .iconst_32 = bound_value } });
    try func.getBlock(entry).append(.{ .op = .{ .br = header } });

    const phi_i = func.newVReg();
    const phi_sum = func.newVReg();
    const inc = func.newVReg();
    const sum_next = func.newVReg();
    const i_edges = try allocator.dupe(ir.Inst.PhiEdge, &.{ .{ .block = entry, .val = zero }, .{ .block = body, .val = inc } });
    const sum_edges = try allocator.dupe(ir.Inst.PhiEdge, &.{ .{ .block = entry, .val = 0 }, .{ .block = body, .val = sum_next } });
    const cond = func.newVReg();
    try func.getBlock(header).append(.{ .dest = phi_i, .type = .i32, .op = .{ .phi = i_edges } });
    try func.getBlock(header).append(.{ .dest = phi_sum, .type = .i32, .op = .{ .phi = sum_edges } });
    try func.getBlock(header).append(.{ .dest = cond, .type = .i32, .op = .{ .lt_u = .{ .lhs = phi_i, .rhs = bound } } });
    try func.getBlock(header).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = body, .else_block = exit } } });

    const one = func.newVReg();
    try func.getBlock(body).append(.{ .dest = one, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(body).append(.{ .dest = sum_next, .type = .i32, .op = .{ .add = .{ .lhs = phi_sum, .rhs = phi_i } } });
    try func.getBlock(body).append(.{ .dest = inc, .type = .i32, .op = .{ .add = .{ .lhs = phi_i, .rhs = one } } });
    try func.getBlock(body).append(.{ .op = .{ .br = header } });

    try func.getBlock(exit).append(.{ .op = .{ .ret = phi_sum } });

    return finishCase(allocator, seed, .counted_loop, func, inputs, memory);
}

fn generateMemoryBarrier(allocator: std.mem.Allocator, seed: u64, random: std.Random) !Case {
    var func = try initFunc(allocator, 1, 1, 1);
    errdefer func.deinit();
    const inputs = try makeInputs(allocator, random, 1);
    errdefer allocator.free(inputs);
    const memory = try makeMemory(allocator, random);
    errdefer allocator.free(memory);

    const entry = try func.newBlock();
    const call_block = try func.newBlock();
    const direct_block = try func.newBlock();
    const merge = try func.newBlock();
    try func.getBlock(call_block).addPredecessor(entry);
    try func.getBlock(direct_block).addPredecessor(entry);
    try func.getBlock(merge).addPredecessor(call_block);
    try func.getBlock(merge).addPredecessor(direct_block);

    const base = func.newVReg();
    const cond = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = base, .type = .i32, .op = .{ .iconst_32 = 0 } });
    try func.getBlock(entry).append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = 0 } } });
    try func.getBlock(entry).append(.{ .dest = cond, .type = .i32, .op = .{ .eqz = 0 } });
    try func.getBlock(entry).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = direct_block, .else_block = call_block } } });

    try func.getBlock(call_block).append(.{ .op = .{ .call = .{ .func_idx = 3, .args = &.{} } }, .type = .void });
    try func.getBlock(call_block).append(.{ .op = .{ .br = merge } });

    try func.getBlock(direct_block).append(.{ .op = .{ .br = merge } });

    const loaded = func.newVReg();
    try func.getBlock(merge).append(.{ .dest = loaded, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(merge).append(.{ .op = .{ .ret = loaded } });

    return finishCase(allocator, seed, .memory_barrier, func, inputs, memory);
}

/// #794 / #743 regression generator. Builds:
///
///     entry:  store[0] = p0; dom = load[0]; br header
///     header: phi_i, phi_sum; cond = i < bound; br_if cond, body, exit
///     body:   cur = load[0]            ; redundant w.r.t. `dom` (forward target)
///             sum_next = phi_sum + cur
///             store[0] = cur + 1       ; aliasing store -> staleifies `dom`
///             i_next = i + 1; br header
///     exit:   ret phi_sum
///
/// `cur` is a load-forwarding candidate for the dominating `dom`. The #793 gate
/// must refuse that forward because the in-loop store lies on the body→body
/// back-edge corridor. If a regression forwards it, the interpreter on the
/// optimized IR returns a stale sum (and stale final memory), diverging from
/// the original — which the differential property test flags. `bound >= 2`
/// guarantees the store runs before a reuse so the divergence is observable.
fn generateLoopForwardedLoad(allocator: std.mem.Allocator, seed: u64, random: std.Random) !Case {
    var func = try initFunc(allocator, 1, 1, 0);
    errdefer func.deinit();
    const inputs = try makeInputs(allocator, random, 1);
    errdefer allocator.free(inputs);
    const memory = try makeMemory(allocator, random);
    errdefer allocator.free(memory);

    const bound_value: i32 = @intCast(random.intRangeLessThan(u32, 2, 6));

    const entry = try func.newBlock();
    const header = try func.newBlock();
    const body = try func.newBlock();
    const exit = try func.newBlock();
    try func.getBlock(header).addPredecessor(entry);
    try func.getBlock(header).addPredecessor(body);
    try func.getBlock(body).addPredecessor(header);
    try func.getBlock(exit).addPredecessor(header);

    const base = func.newVReg();
    const zero = func.newVReg();
    const bound = func.newVReg();
    const dom = func.newVReg();
    try func.getBlock(entry).append(.{ .dest = base, .type = .i32, .op = .{ .iconst_32 = 0 } });
    // Seed memory[0..4] from the input parameter so the value differs across
    // input vectors, then take the dominating load.
    try func.getBlock(entry).append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = 0 } } });
    try func.getBlock(entry).append(.{ .dest = dom, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(entry).append(.{ .dest = zero, .type = .i32, .op = .{ .iconst_32 = 0 } });
    try func.getBlock(entry).append(.{ .dest = bound, .type = .i32, .op = .{ .iconst_32 = bound_value } });
    try func.getBlock(entry).append(.{ .op = .{ .br = header } });

    const phi_i = func.newVReg();
    const phi_sum = func.newVReg();
    const inc = func.newVReg();
    const sum_next = func.newVReg();
    const i_edges = try allocator.dupe(ir.Inst.PhiEdge, &.{ .{ .block = entry, .val = zero }, .{ .block = body, .val = inc } });
    // `dom` seeds the accumulator so the returned value depends on the
    // dominating load as well as each in-loop reload.
    const sum_edges = try allocator.dupe(ir.Inst.PhiEdge, &.{ .{ .block = entry, .val = dom }, .{ .block = body, .val = sum_next } });
    const cond = func.newVReg();
    try func.getBlock(header).append(.{ .dest = phi_i, .type = .i32, .op = .{ .phi = i_edges } });
    try func.getBlock(header).append(.{ .dest = phi_sum, .type = .i32, .op = .{ .phi = sum_edges } });
    try func.getBlock(header).append(.{ .dest = cond, .type = .i32, .op = .{ .lt_u = .{ .lhs = phi_i, .rhs = bound } } });
    try func.getBlock(header).append(.{ .op = .{ .br_if = .{ .cond = cond, .then_block = body, .else_block = exit } } });

    const cur = func.newVReg();
    const one = func.newVReg();
    const next_mem = func.newVReg();
    try func.getBlock(body).append(.{ .dest = cur, .type = .i32, .op = .{ .load = .{ .base = base, .offset = 0, .size = 4 } } });
    try func.getBlock(body).append(.{ .dest = sum_next, .type = .i32, .op = .{ .add = .{ .lhs = phi_sum, .rhs = cur } } });
    try func.getBlock(body).append(.{ .dest = one, .type = .i32, .op = .{ .iconst_32 = 1 } });
    try func.getBlock(body).append(.{ .dest = next_mem, .type = .i32, .op = .{ .add = .{ .lhs = cur, .rhs = one } } });
    try func.getBlock(body).append(.{ .op = .{ .store = .{ .base = base, .offset = 0, .size = 4, .val = next_mem } } });
    try func.getBlock(body).append(.{ .dest = inc, .type = .i32, .op = .{ .add = .{ .lhs = phi_i, .rhs = one } } });
    try func.getBlock(body).append(.{ .op = .{ .br = header } });

    try func.getBlock(exit).append(.{ .op = .{ .ret = phi_sum } });

    return finishCase(allocator, seed, .loop_forwarded_load, func, inputs, memory);
}

test "fuzz: generated cases verify and interpret" {
    const a = std.testing.allocator;
    for (Shape.all, 0..) |shape, i| {
        var case = try generate(a, 0x7360 + i, shape);
        defer case.deinit(a);

        var outcome = try interp.run(a, &case.func, .{
            .params = case.inputs,
            .memory = case.memory,
            .fuel = 1_000,
        });
        defer outcome.deinit(a);
        try std.testing.expect(outcome == .returned);
        try std.testing.expect(outcome.returned.results.len <= 1);
    }
}
