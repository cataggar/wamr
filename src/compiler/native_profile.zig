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
    for (lowered.functions.items) |function| {
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
