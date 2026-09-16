//! Admission policy for --profile=unikraft-x86_64. This runs before optimization:
//! unsupported operations cannot disappear and accidentally acquire a supported
//! runtime-contract stamp. The runtime independently validates all metadata.
const wamr = @import("wamr");
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

pub fn validate(module: anytype, lowered: *const ir.IrModule) error{UnsupportedNativeFeature}!void {
    if (lowered.has_memory64 or lowered.has_shared_memory or lowered.spawns_threads)
        return error.UnsupportedNativeFeature;
    if (module.memories.len > 1 or module.tag_types.len != 0) return error.UnsupportedNativeFeature;
    for (module.memories) |memory| {
        if (memory.is_memory64 or memory.is_shared) return error.UnsupportedNativeFeature;
    }

    for (module.imports) |imp| {
        if (imp.kind != .function) return error.UnsupportedNativeFeature;
        const signature = module.types[imp.func_type_idx.?];
        if (signature.params.len > 5) return error.UnsupportedNativeFeature;
    }
    for (module.types) |signature| {
        if (signature.params.len > 16 or signature.results.len > 1) return error.UnsupportedNativeFeature;
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
