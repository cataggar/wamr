//! Admission policy for --profile=unikraft-x86_64. This runs before optimization:
//! unsupported operations cannot disappear and accidentally acquire a supported
//! runtime-contract stamp. The runtime independently validates all metadata.
const wamr = struct {
    pub const ir = @import("ir/ir.zig");
    pub const types = @import("../runtime/common/types.zig");
    pub const frontend = @import("frontend.zig");
};
const ir = wamr.ir;
const std = @import("std");

test "native profile: rejects shared memory memory64 threads and non-scalar signatures" {
    var lowered = ir.IrModule.init(std.testing.allocator);
    defer lowered.deinit();
    var module: wamr.types.WasmModule = .{};
    try validate(&module, &lowered);
    lowered.has_shared_memory = true;
    try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
    lowered.has_shared_memory = false;
    lowered.has_memory64 = true;
    try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
    lowered.has_memory64 = false;
    lowered.spawns_threads = true;
    try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
    lowered.spawns_threads = false;
    module.types = &.{.{ .params = &.{.v128}, .results = &.{} }};
    try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
    module.types = &.{.{ .params = &.{}, .results = &.{ .i32, .i32 } }};
    try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
}

test "native profile: rejects ref.as_non_null erased by lowering including dead code" {
    const signature: wamr.types.FuncType = .{ .params = &.{}, .results = &.{.i32} };
    for ([_][]const u8{
        &.{ 0xd0, 0x70, 0xd4, 0x1a, 0x41, 0x01, 0x0b },
        &.{ 0x00, 0xd0, 0x70, 0xd4, 0x1a, 0x41, 0x01, 0x0b },
    }) |code| {
        const module: wamr.types.WasmModule = .{
            .types = &.{signature},
            .functions = &.{.{
                .type_idx = 0,
                .func_type = signature,
                .local_count = 0,
                .locals = &.{},
                .code = code,
            }},
        };
        var lowered = try wamr.frontend.lowerModule(&module, std.testing.allocator);
        defer lowered.deinit();
        try std.testing.expect(lowered.functions.items[0].has_ref_as_non_null);
        try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
    }
    // 0xd4 in an immediate is not a reference instruction.
    const module: wamr.types.WasmModule = .{
        .types = &.{signature},
        .functions = &.{.{
            .type_idx = 0,
            .func_type = signature,
            .local_count = 0,
            .locals = &.{},
            .code = &.{ 0x41, 0xd4, 0x01, 0x0b },
        }},
    };
    var lowered = try wamr.frontend.lowerModule(&module, std.testing.allocator);
    defer lowered.deinit();
    try std.testing.expect(!lowered.functions.items[0].has_ref_as_non_null);
    try validate(&module, &lowered);
}

test "native profile: rejects omitted element segments before source indices change" {
    const signature: wamr.types.FuncType = .{ .params = &.{}, .results = &.{} };
    var segments = [_]wamr.types.ElemSegment{
        .{ .table_idx = 0, .offset = null, .kind = .func_ref, .func_indices = &.{0}, .is_declarative = true },
        .{ .table_idx = 0, .offset = null, .kind = .func_ref, .func_indices = &.{ null, 0 }, .is_passive = true },
    };
    const module: wamr.types.WasmModule = .{
        .types = &.{signature},
        .tables = &.{.{ .elem_type = .funcref, .limits = .{ .min = 2 } }},
        .elements = &segments,
        .functions = &.{.{
            .type_idx = 0,
            .func_type = signature,
            .local_count = 0,
            .locals = &.{},
            // table.init element 1, then elem.drop element 1.
            .code = &.{ 0x41, 0, 0x41, 0, 0x41, 0, 0xfc, 0x0c, 1, 0, 0xfc, 0x0d, 1, 0x0b },
        }},
    };
    var lowered = try wamr.frontend.lowerModule(&module, std.testing.allocator);
    defer lowered.deinit();
    try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
    segments[0].is_declarative = false;
    segments[0].is_passive = true;
    try validate(&module, &lowered);
    segments[0].is_passive = false;
    try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
    segments[0].offset = .{ .global_get = 0 };
    try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
    segments[0].offset = .{ .i32_const = 0 };
    try validate(&module, &lowered);
    segments[0].elem_exprs = &.{.{ .global_get = 0 }};
    try std.testing.expectError(error.UnsupportedNativeFeature, validate(&module, &lowered));
}

pub fn validateModule(module: anytype) error{UnsupportedNativeFeature}!void {
    if (module.memories.len > 1 or module.tag_types.len != 0) return error.UnsupportedNativeFeature;
    if (module.imports.len > 64) return error.UnsupportedNativeFeature;
    for (module.memories) |memory| {
        if (memory.is_memory64 or memory.is_shared) return error.UnsupportedNativeFeature;
    }

    for (module.imports) |imp| {
        if (imp.kind != .function) return error.UnsupportedNativeFeature;
        const signature = module.types[imp.func_type_idx.?];
        if (signature.params.len > 5) return error.UnsupportedNativeFeature;
    }
    for (module.types) |signature| {
        if (signature.kind != .func or signature.params.len > 16 or signature.results.len > 1) return error.UnsupportedNativeFeature;
        for (signature.params) |t| if (!t.isNumeric()) return error.UnsupportedNativeFeature;
        for (signature.results) |t| if (!t.isNumeric()) return error.UnsupportedNativeFeature;
    }
    for (module.globals) |g| if (!g.global_type.val_type.isNumeric()) return error.UnsupportedNativeFeature;
    for (module.tables) |t| {
        if (t.elem_type != .funcref or t.is_table64 or t.init_expr != null) return error.UnsupportedNativeFeature;
    }
    for (module.elements) |segment| {
        // The existing container emitter omits declarative segments, which
        // would change the source index space used by table.init/elem.drop.
        if (segment.is_declarative) return error.UnsupportedNativeFeature;
        if (segment.kind != .func_ref) return error.UnsupportedNativeFeature;
        if (!segment.is_passive) {
            const offset = segment.offset orelse return error.UnsupportedNativeFeature;
            if (offset != .i32_const) return error.UnsupportedNativeFeature;
        }
        for (segment.elem_exprs) |expression| {
            if (expression != null) return error.UnsupportedNativeFeature;
        }
    }
}

pub fn validate(module: anytype, lowered: *const ir.IrModule) error{UnsupportedNativeFeature}!void {
    try validateModule(module);
    if (lowered.has_memory64 or lowered.has_shared_memory or lowered.spawns_threads)
        return error.UnsupportedNativeFeature;
    for (lowered.functions.items) |function| {
        if (function.has_ref_as_non_null) return error.UnsupportedNativeFeature;
        for (function.blocks.items) |block| {
            for (block.instructions.items) |instruction| {
                switch (instruction.op) {
                    .iconst_32,
                    .iconst_64,
                    .fconst_32,
                    .fconst_64,
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
                    .lea,
                    .clz,
                    .ctz,
                    .popcnt,
                    .eqz,
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
                    .local_get,
                    .local_set,
                    .load,
                    .store,
                    .br,
                    .br_if,
                    .br_table,
                    .ret,
                    .@"unreachable",
                    .call,
                    .call_indirect,
                    .select,
                    .global_get,
                    .global_set,
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
                    .f_min,
                    .f_max,
                    .f_copysign,
                    .f_eq,
                    .f_ne,
                    .f_lt,
                    .f_gt,
                    .f_le,
                    .f_ge,
                    .wrap_i64,
                    .extend_i32_s,
                    .extend_i32_u,
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
                    .memory_copy,
                    .memory_fill,
                    .memory_init,
                    .data_drop,
                    .memory_size,
                    .memory_grow,
                    .table_size,
                    .table_get,
                    .table_set,
                    .table_grow,
                    .table_init,
                    .elem_drop,
                    .ref_func,
                    .phi,
                    .parallel_copy,
                    => {},
                    else => return error.UnsupportedNativeFeature,
                }
            }
        }
    }
}

/// Fuel bounds loops, not native stack depth. The optional compiler profile
/// therefore admits only statically bounded direct call graphs.
pub fn validateBoundedCalls(module: *const ir.IrModule, allocator: std.mem.Allocator) error{ OutOfMemory, UnsupportedNativeFeature }!void {
    const depth = try allocator.alloc(u8, module.functions.items.len);
    defer allocator.free(depth);
    @memset(depth, 0);
    for (0..depth.len) |index| _ = try callDepth(module, index, depth, 0);
}

fn callDepth(module: *const ir.IrModule, index: usize, memo: []u8, level: usize) error{UnsupportedNativeFeature}!u8 {
    const limit = @import("../runtime/aot/native_abi.zig").max_fuel_call_depth;
    if (level >= limit or memo[index] == 255) return error.UnsupportedNativeFeature;
    if (memo[index] != 0) return memo[index];
    memo[index] = 255;
    var depth: u8 = 1;
    for (module.functions.items[index].blocks.items) |block| {
        for (block.instructions.items) |instruction| switch (instruction.op) {
            .call => |call| {
                if (call.func_idx < module.import_count) continue;
                const target = call.func_idx - module.import_count;
                if (target >= memo.len) return error.UnsupportedNativeFeature;
                depth = @max(depth, 1 + try callDepth(module, target, memo, level + 1));
            },
            .call_indirect, .call_ref => return error.UnsupportedNativeFeature,
            else => {},
        };
    }
    if (depth > limit) return error.UnsupportedNativeFeature;
    memo[index] = depth;
    return depth;
}

test "native profile: rejects scalar ABI overflow before lowering" {
    var module: wamr.types.WasmModule = .{};
    module.types = &.{.{ .params = &([_]wamr.types.ValType{.i32} ** 17), .results = &.{} }};
    try std.testing.expectError(error.UnsupportedNativeFeature, validateModule(&module));
    module.types = &.{.{ .params = &([_]wamr.types.ValType{.i32} ** 6), .results = &.{} }};
    module.imports = &.{.{ .module_name = "env", .field_name = "too_many", .kind = .function, .func_type_idx = 0 }};
    try std.testing.expectError(error.UnsupportedNativeFeature, validateModule(&module));
}

test "native profile: fuel rejects cycles indirect calls and excessive static depth" {
    var module = ir.IrModule.init(std.testing.allocator);
    defer module.deinit();
    for (0..33) |i| {
        var func = ir.IrFunction.init(std.testing.allocator, 0, 0, 0);
        const block = try func.newBlock();
        if (i < 32) try func.getBlock(block).append(.{ .op = .{ .call = .{ .func_idx = @intCast(i + 1) } } });
        try func.getBlock(block).append(.{ .op = .{ .ret = null } });
        _ = try module.addFunction(func);
    }
    try std.testing.expectError(error.UnsupportedNativeFeature, validateBoundedCalls(&module, std.testing.allocator));
    module.functions.items[0].blocks.items[0].instructions.items[0].op = .{ .call = .{ .func_idx = 0 } };
    try std.testing.expectError(error.UnsupportedNativeFeature, validateBoundedCalls(&module, std.testing.allocator));
    module.functions.items[0].blocks.items[0].instructions.items[0].op = .{ .call_indirect = .{ .type_idx = 0, .table_idx = 0, .elem_idx = 0 } };
    try std.testing.expectError(error.UnsupportedNativeFeature, validateBoundedCalls(&module, std.testing.allocator));
}
