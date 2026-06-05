//! Property tests for optimizer passes using the in-process IR interpreter.
//! Completes the #736 oracle loop: generate verifier-valid IR, execute it,
//! clone + optimize, verify again, execute again, and compare observables.

const std = @import("std");
const ir = @import("ir.zig");
const interp = @import("interp.zig");
const fuzz = @import("fuzz.zig");
const passes = @import("passes.zig");
const verifier = @import("verifier.zig");
const dead_store = @import("dead_store_elimination.zig");
const frl_dom = @import("forward_redundant_loads_dominator.zig");
const property_options = @import("ir_property_options");

const PassFn = *const fn (*ir.IrFunction, std.mem.Allocator) anyerror!bool;

const PassCase = struct {
    name: []const u8,
    func: PassFn,
};

const property_passes = [_]PassCase{
    .{ .name = "constantFold", .func = passes.constantFold },
    .{ .name = "algebraicSimplify", .func = passes.algebraicSimplify },
    .{ .name = "deadCodeElimination", .func = passes.deadCodeElimination },
    .{ .name = "deadStoreElimination", .func = dead_store.deadStoreElimination },
    .{ .name = "commonSubexprElimination", .func = passes.commonSubexprElimination },
    .{ .name = "globalValueNumbering", .func = passes.globalValueNumbering },
    .{ .name = "forwardRedundantLoadsDominator", .func = frl_dom.forwardRedundantLoadsDominator },
};

fn setInputVector(inputs: []interp.Value, seed: u64, vector: u32) void {
    var x = seed ^ (@as(u64, vector) *% 0x9e37_79b9_7f4a_7c15);
    for (inputs, 0..) |*v, i| {
        x = x *% 0xd134_2543_de82_ef95 +% 0x94d0_49bb_1331_11eb +% i;
        var value: u32 = @truncate(x >> 32);
        value &= 0x0f;
        // Force both sides of branch-heavy shapes across the input vectors.
        if (i == 0) value = if ((vector & 1) == 0) 0 else @max(value, 1);
        v.* = interp.Value.u32v(value);
    }
}

fn expectEquivalent(
    pass_name: []const u8,
    seed: u64,
    shape: fuzz.Shape,
    expected: interp.Outcome,
    observed: interp.Outcome,
) !void {
    if (!outcomesEqual(expected, observed)) {
        std.debug.print(
            "IR property mismatch: pass={s} seed=0x{x} shape={s}\n",
            .{ pass_name, seed, @tagName(shape) },
        );
        try std.testing.expect(false);
    }
}

fn outcomesEqual(a: interp.Outcome, b: interp.Outcome) bool {
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    return switch (a) {
        .returned => |ar| blk: {
            const br = b.returned;
            if (ar.results.len != br.results.len) break :blk false;
            for (ar.results, br.results) |av, bv| {
                if (av.ty != bv.ty or av.bits != bv.bits) break :blk false;
            }
            break :blk std.mem.eql(u8, ar.memory, br.memory);
        },
        .trapped => |at| at == b.trapped,
        .unsupported => true,
        .inconclusive => true,
    };
}

fn checkPassPreservesCase(
    allocator: std.mem.Allocator,
    pass: PassCase,
    seed: u64,
    shape: fuzz.Shape,
    vector_count: u32,
) !void {
    var case = try fuzz.generate(allocator, seed, shape);
    defer case.deinit(allocator);

    var optimized = try case.func.clone(allocator);
    defer optimized.deinit();
    _ = try pass.func(&optimized, allocator);
    try verifier.verifyFunction(&optimized, 0, .after_each_pass, allocator);

    var vector: u32 = 0;
    while (vector < vector_count) : (vector += 1) {
        setInputVector(case.inputs, seed, vector);

        var expected = try interp.run(allocator, &case.func, .{
            .params = case.inputs,
            .memory = case.memory,
            .fuel = 10_000,
        });
        defer expected.deinit(allocator);

        var observed = try interp.run(allocator, &optimized, .{
            .params = case.inputs,
            .memory = case.memory,
            .fuel = 10_000,
        });
        defer observed.deinit(allocator);

        try expectEquivalent(pass.name, seed, shape, expected, observed);
    }
}

test "property: scalar optimizer passes preserve interpreted behavior" {
    const allocator = std.testing.allocator;
    const iterations = property_options.iterations;
    const vectors_per_case: u32 = 4;

    for (property_passes) |pass| {
        var iter: u32 = 0;
        while (iter < iterations) : (iter += 1) {
            for (fuzz.Shape.all, 0..) |shape, shape_idx| {
                const seed = 0x7360_0000 + (@as(u64, iter) << 8) + shape_idx;
                try checkPassPreservesCase(allocator, pass, seed, shape, vectors_per_case);
            }
        }
    }
}

test "property: named memory-barrier regression seed exercises call path" {
    const allocator = std.testing.allocator;
    const seed = 0x7360_00ff;
    try checkPassPreservesCase(
        allocator,
        .{ .name = "forwardRedundantLoadsDominator", .func = frl_dom.forwardRedundantLoadsDominator },
        seed,
        .memory_barrier,
        2,
    );
}

test "property(#794): load-forwarding passes preserve behavior across a loop effect" {
    // #743 regression: a load dominating a loop whose body reloads + stores the
    // same address must not be forwarded into the loop. The #793 gate enforces
    // that; a regression would surface here as a stale interpreted result.
    const allocator = std.testing.allocator;
    const forwarders = [_]PassCase{
        .{ .name = "commonSubexprElimination", .func = passes.commonSubexprElimination },
        .{ .name = "globalValueNumbering", .func = passes.globalValueNumbering },
        .{ .name = "forwardRedundantLoadsDominator", .func = frl_dom.forwardRedundantLoadsDominator },
    };
    for (forwarders) |pass| {
        var iter: u32 = 0;
        while (iter < 32) : (iter += 1) {
            const seed = 0x7943_0000 + iter;
            try checkPassPreservesCase(allocator, pass, seed, .loop_forwarded_load, 4);
        }
    }
}
