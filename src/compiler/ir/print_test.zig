//! Unit tests for the IR pretty-printer in `print.zig`.

const std = @import("std");
const ir = @import("ir.zig");
const ir_print = @import("print.zig");

/// Helper — render an instruction into a freshly-allocated owned slice.
fn renderInst(allocator: std.mem.Allocator, inst: ir.Inst) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    try ir_print.formatInst(inst, &aw.writer);
    var list = aw.toArrayList();
    return list.toOwnedSlice(allocator);
}

/// Helper — render an entire function into a freshly-allocated owned slice.
fn renderFunc(allocator: std.mem.Allocator, func: *const ir.IrFunction, idx: u32) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    try ir_print.formatFunc(func, idx, &aw.writer);
    var list = aw.toArrayList();
    return list.toOwnedSlice(allocator);
}

test "formatInst: iconst_32 produces dest + tag + payload" {
    const a = std.testing.allocator;
    const s = try renderInst(a, .{
        .op = .{ .iconst_32 = 42 },
        .dest = 0,
        .type = .i32,
    });
    defer a.free(s);
    try std.testing.expectEqualStrings("v0 = iconst_32.i32 42", s);
}

test "formatInst: local_get / local_set render local index and val" {
    const a = std.testing.allocator;

    const get_s = try renderInst(a, .{
        .op = .{ .local_get = 7 },
        .dest = 3,
        .type = .i64,
    });
    defer a.free(get_s);
    try std.testing.expectEqualStrings("v3 = local_get.i64 local[7]", get_s);

    const set_s = try renderInst(a, .{
        .op = .{ .local_set = .{ .idx = 7, .val = 3 } },
        .type = .void,
    });
    defer a.free(set_s);
    try std.testing.expectEqualStrings("local_set local[7], v3", set_s);
}

test "formatInst: load / store include base, offset, size, sext, bounds_known" {
    const a = std.testing.allocator;

    const ld_s = try renderInst(a, .{
        .op = .{ .load = .{ .base = 1, .offset = 16, .size = 4, .sign_extend = true, .bounds_known = true } },
        .dest = 2,
        .type = .i32,
    });
    defer a.free(ld_s);
    try std.testing.expectEqualStrings(
        "v2 = load.i32 base=v1, offset=16, size=4, sext, bounds_known",
        ld_s,
    );

    const st_s = try renderInst(a, .{
        .op = .{ .store = .{ .base = 1, .offset = 8, .size = 8, .val = 4 } },
        .type = .void,
    });
    defer a.free(st_s);
    try std.testing.expectEqualStrings("store base=v1, offset=8, size=8, val=v4", st_s);
}

test "formatInst: br_if / br_table render block ids" {
    const a = std.testing.allocator;

    const brif_s = try renderInst(a, .{
        .op = .{ .br_if = .{ .cond = 5, .then_block = 3, .else_block = 9 } },
        .type = .void,
    });
    defer a.free(brif_s);
    try std.testing.expectEqualStrings("br_if cond=v5, then=b3, else=b9", brif_s);

    const targets = [_]ir.BlockId{ 4, 5, 6 };
    const brt_s = try renderInst(a, .{
        .op = .{ .br_table = .{ .index = 2, .targets = &targets, .default = 7 } },
        .type = .void,
    });
    defer a.free(brt_s);
    try std.testing.expectEqualStrings(
        "br_table index=v2, default=b7, targets=[b4, b5, b6]",
        brt_s,
    );
}

test "formatInst: ret with and without value, ret_multi list" {
    const a = std.testing.allocator;

    const ret_void = try renderInst(a, .{ .op = .{ .ret = null }, .type = .void });
    defer a.free(ret_void);
    try std.testing.expectEqualStrings("ret", ret_void);

    const ret_v = try renderInst(a, .{ .op = .{ .ret = 11 }, .type = .void });
    defer a.free(ret_v);
    try std.testing.expectEqualStrings("ret v11", ret_v);

    const vregs = [_]ir.VReg{ 1, 2, 3 };
    const ret_m = try renderInst(a, .{ .op = .{ .ret_multi = &vregs }, .type = .void });
    defer a.free(ret_m);
    try std.testing.expectEqualStrings("ret_multi [v1, v2, v3]", ret_m);
}

test "formatInst: call lists args and extra_results, call_result has index" {
    const a = std.testing.allocator;

    const args = [_]ir.VReg{ 4, 5 };
    const call_s = try renderInst(a, .{
        .op = .{ .call = .{ .func_idx = 12, .args = &args, .extra_results = 2 } },
        .dest = 10,
        .type = .i32,
    });
    defer a.free(call_s);
    try std.testing.expectEqualStrings(
        "v10 = call.i32 func=12, args=[v4, v5], extra_results=2",
        call_s,
    );

    const call_tail = try renderInst(a, .{
        .op = .{ .call = .{ .func_idx = 12, .args = &args, .tail = true } },
        .type = .void,
    });
    defer a.free(call_tail);
    try std.testing.expectEqualStrings("call func=12, tail, args=[v4, v5]", call_tail);

    const cr_s = try renderInst(a, .{
        .op = .{ .call_result = 1 },
        .dest = 11,
        .type = .i64,
    });
    defer a.free(cr_s);
    try std.testing.expectEqualStrings("v11 = call_result.i64 #1", cr_s);
}

test "formatInst: phi prints predecessor-value pairs" {
    const a = std.testing.allocator;
    const edges = [_]ir.Inst.PhiEdge{
        .{ .block = 0, .val = 4 },
        .{ .block = 2, .val = 9 },
    };
    const s = try renderInst(a, .{
        .op = .{ .phi = &edges },
        .dest = 12,
        .type = .i32,
    });
    defer a.free(s);
    try std.testing.expectEqualStrings("v12 = phi.i32 [b0:v4, b2:v9]", s);
}

test "formatInst: scalar binop renders lhs/rhs" {
    const a = std.testing.allocator;
    const s = try renderInst(a, .{
        .op = .{ .add = .{ .lhs = 1, .rhs = 2 } },
        .dest = 3,
        .type = .i32,
    });
    defer a.free(s);
    try std.testing.expectEqualStrings("v3 = add.i32 v1, v2", s);
}

test "formatInst: SIMD op uses generic vregs=[…] rendering" {
    const a = std.testing.allocator;
    const s = try renderInst(a, .{
        .op = .{ .i32x4_binop = .{ .op = .add, .lhs = 5, .rhs = 6 } },
        .dest = 7,
        .type = .v128,
    });
    defer a.free(s);
    try std.testing.expectEqualStrings("v7 = i32x4_binop.v128 (vregs=[v5, v6])", s);
}

test "formatFunc: header + blocks + predecessor lists" {
    const a = std.testing.allocator;
    var func = ir.IrFunction.init(a, 1, 1, 3);
    defer func.deinit();
    func.name = "demo";
    func.next_vreg = 4;

    // b0: v0 = local_get local[0]; v1 = iconst_32 1; v2 = add v0, v1; br_if cond=v2 then=b1 else=b2
    _ = try func.newBlock();
    try func.blocks.items[0].append(.{ .op = .{ .local_get = 0 }, .dest = 0, .type = .i32 });
    try func.blocks.items[0].append(.{ .op = .{ .iconst_32 = 1 }, .dest = 1, .type = .i32 });
    try func.blocks.items[0].append(.{ .op = .{ .add = .{ .lhs = 0, .rhs = 1 } }, .dest = 2, .type = .i32 });
    try func.blocks.items[0].append(.{ .op = .{ .br_if = .{ .cond = 2, .then_block = 1, .else_block = 2 } }, .type = .void });

    // b1: ret v2
    _ = try func.newBlock();
    try func.blocks.items[1].addPredecessor(0);
    try func.blocks.items[1].append(.{ .op = .{ .ret = 2 }, .type = .void });

    // b2: unreachable
    _ = try func.newBlock();
    try func.blocks.items[2].addPredecessor(0);
    try func.blocks.items[2].append(.{ .op = .{ .@"unreachable" = {} }, .type = .void });

    const out = try renderFunc(a, &func, 5);
    defer a.free(out);

    const expected =
        \\func #5 demo (params=1, results=1, locals=3, vregs=4) {
        \\b0:
        \\  v0 = local_get.i32 local[0]
        \\  v1 = iconst_32.i32 1
        \\  v2 = add.i32 v0, v1
        \\  br_if cond=v2, then=b1, else=b2
        \\b1  ; preds=[b0]:
        \\  ret v2
        \\b2  ; preds=[b0]:
        \\  unreachable
        \\}
        \\
    ;
    try std.testing.expectEqualStrings(expected, out);
}

test "formatFunc: unnamed function header" {
    const a = std.testing.allocator;
    var func = ir.IrFunction.init(a, 0, 0, 0);
    defer func.deinit();
    _ = try func.newBlock();
    try func.blocks.items[0].append(.{ .op = .{ .@"unreachable" = {} }, .type = .void });

    const out = try renderFunc(a, &func, 0);
    defer a.free(out);

    const expected =
        \\func #0 <unnamed> (params=0, results=0, locals=0, vregs=0) {
        \\b0:
        \\  unreachable
        \\}
        \\
    ;
    try std.testing.expectEqualStrings(expected, out);
}
