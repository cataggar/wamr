//! Lazy-JIT regression coverage for #887's non-leaf x86_64 extension.
const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

const config = wamr.config;
const component_aot_compile = wamr.component_aot_compile;
const aot_loader_mod = wamr.aot_loader;
const aot_runtime_mod = wamr.aot_runtime;

const can_exec_aot = switch (builtin.cpu.arch) {
    .x86_64 => true,
    else => false,
};

const eager_entry_local: usize = 0;
const lazy_add_local: usize = 1;
const eager_callee_local: usize = 2;
const lazy_to_eager_local: usize = 3;
const nested_callee_local: usize = 4;
const nested_tail_local: usize = 5;
const unused_leaf_local: usize = 6;
const unused_nonleaf_local: usize = 7;

const lazy_nonleaf_fixture_wasm = @embedFile("lazy_nonleaf_fixture.wasm");

fn localCodeLen(module: *const aot_loader_mod.AotModule, local_idx: usize) usize {
    const text = module.text_section orelse return 0;
    const start = module.func_offsets[local_idx];
    const end: u32 = if (local_idx + 1 < module.func_offsets.len)
        module.func_offsets[local_idx + 1]
    else
        @intCast(text.len);
    return @intCast(end - start);
}

fn callI32(inst: *aot_runtime_mod.AotInstance, func_idx: u32, arg: i32) !i32 {
    var results_buf: [1]aot_runtime_mod.ScalarResult = undefined;
    const results = try aot_runtime_mod.callFuncScalar(
        inst,
        func_idx,
        &.{.i32},
        &.{.i32},
        &.{.{ .i32 = arg }},
        &results_buf,
    );
    return results[0].i32;
}

test "#887 lazy-JIT: stable stubs and lazy-body indirect calls cover non-leaf direct-call graphs" {
    if (comptime !config.lazy_jit) return error.SkipZigTest;
    if (comptime !can_exec_aot) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    var lazy_out: component_aot_compile.LazyJitOut = .{};
    const cwasm = try component_aot_compile.compileCoreWasmCached(
        gpa,
        lazy_nonleaf_fixture_wasm,
        .{ .lazy_jit = true },
        .{ .lazy_jit_out = &lazy_out },
    );
    defer gpa.free(cwasm);

    try std.testing.expectEqualSlices(
        u32,
        &.{ 1, 3, 4, 5, 6, 7 },
        lazy_out.lazy_local_indices,
    );

    var module = try aot_loader_mod.load(cwasm, gpa);
    defer aot_loader_mod.unload(&module, gpa);

    try std.testing.expect(module.text_section != null);
    for (lazy_out.lazy_local_indices) |local_idx| {
        try std.testing.expect(localCodeLen(&module, local_idx) > 0);
    }

    const inst = try aot_runtime_mod.instantiate(&module, gpa);
    try aot_runtime_mod.mapCodeExecutable(inst);

    const driver = try component_aot_compile.setupLazyJit(inst, lazy_out, gpa);
    defer {
        aot_runtime_mod.destroy(inst);
        driver.deinit();
    }

    try std.testing.expect(!inst.lazy_jit.pending[eager_entry_local]);
    try std.testing.expect(inst.lazy_jit.pending[lazy_add_local]);
    try std.testing.expect(!inst.lazy_jit.pending[eager_callee_local]);
    try std.testing.expect(inst.lazy_jit.pending[lazy_to_eager_local]);
    try std.testing.expect(inst.lazy_jit.pending[nested_callee_local]);
    try std.testing.expect(inst.lazy_jit.pending[nested_tail_local]);
    try std.testing.expect(inst.lazy_jit.pending[unused_leaf_local]);
    try std.testing.expect(inst.lazy_jit.pending[unused_nonleaf_local]);

    const eager_entry_idx = aot_runtime_mod.findExportFunc(inst, "eager_entry") orelse return error.ExportNotFound;
    const lazy_to_eager_idx = aot_runtime_mod.findExportFunc(inst, "lazy_to_eager") orelse return error.ExportNotFound;
    const nested_tail_idx = aot_runtime_mod.findExportFunc(inst, "nested_tail") orelse return error.ExportNotFound;

    try std.testing.expectEqual(@as(i32, 42), try callI32(inst, eager_entry_idx, 41));
    try std.testing.expect(!inst.lazy_jit.pending[lazy_add_local]);
    const lazy_add_addr = inst.lazy_jit.compiled[lazy_add_local].?.addr;
    try std.testing.expectEqual(
        @intFromPtr(lazy_add_addr),
        inst.funcptrs[@as(usize, module.import_function_count) + lazy_add_local],
    );
    try std.testing.expectEqual(@as(i32, 10), try callI32(inst, eager_entry_idx, 9));
    try std.testing.expectEqual(lazy_add_addr, inst.lazy_jit.compiled[lazy_add_local].?.addr);

    try std.testing.expectEqual(@as(i32, 42), try callI32(inst, lazy_to_eager_idx, 21));
    try std.testing.expect(!inst.lazy_jit.pending[lazy_to_eager_local]);
    try std.testing.expect(inst.lazy_jit.compiled[lazy_to_eager_local] != null);
    try std.testing.expect(inst.lazy_jit.compiled[eager_callee_local] == null);

    try std.testing.expectEqual(@as(i32, 42), try callI32(inst, nested_tail_idx, 37));
    try std.testing.expect(!inst.lazy_jit.pending[nested_callee_local]);
    try std.testing.expect(!inst.lazy_jit.pending[nested_tail_local]);
    const nested_callee_addr = inst.lazy_jit.compiled[nested_callee_local].?.addr;
    const nested_tail_addr = inst.lazy_jit.compiled[nested_tail_local].?.addr;
    try std.testing.expectEqual(@as(i32, 105), try callI32(inst, nested_tail_idx, 100));
    try std.testing.expectEqual(nested_callee_addr, inst.lazy_jit.compiled[nested_callee_local].?.addr);
    try std.testing.expectEqual(nested_tail_addr, inst.lazy_jit.compiled[nested_tail_local].?.addr);

    try std.testing.expect(inst.lazy_jit.pending[unused_leaf_local]);
    try std.testing.expect(inst.lazy_jit.pending[unused_nonleaf_local]);
    try std.testing.expect(inst.lazy_jit.compiled[unused_leaf_local] == null);
    try std.testing.expect(inst.lazy_jit.compiled[unused_nonleaf_local] == null);
}
