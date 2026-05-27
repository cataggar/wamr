//! #625 phase 1 — AOT-backed core instance smoke test.
//!
//! Drives the new `instantiateWithOptions` API end-to-end against a
//! tiny core module precompiled in-process via `aot_harness`. Lives in
//! its own test step because `aot_harness.zig` is owned by the
//! test-runner module — importing it from inside `wamr` would trigger
//! Zig 0.16's "file exists in modules X and Y" error (cf. the
//! `differential.zig` comment in `build.zig`).

const std = @import("std");
const wamr = @import("wamr");
const aot_harness = @import("aot_harness.zig");

const instance = wamr.component_instance;
const ctypes = wamr.component_types;
const core_types = wamr.types;
const aot_loader_mod = wamr.aot_loader;
const aot_runtime_mod = wamr.aot_runtime;

test "#649 phase 1: AOT loader retains imported table descriptors" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // import section: (import "env" "tbl" (table 8 8 funcref))
        0x02, 0x0e,
        0x01, 0x03, 'e',  'n',  'v',  0x03, 't',  'b',
        'l',  0x01, 0x70, 0x01, 0x08, 0x08,
        // function section: 1 local function of type 0
        0x03, 0x02,
        0x01, 0x00,
        // export section: export local func 0 as "run"
        0x07, 0x07, 0x01, 0x03, 'r',  'u',
        'n',  0x00, 0x00,
        // code section: empty body
        0x0a, 0x04, 0x01, 0x02, 0x00,
        0x0b,
    };

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &core_wasm);
    defer allocator.free(cwasm_bytes);

    const module = try aot_loader_mod.load(cwasm_bytes, allocator);
    defer aot_loader_mod.unload(&module, allocator);

    try std.testing.expectEqual(@as(usize, 1), module.imports.len);
    try std.testing.expectEqual(core_types.ExternalKind.table, module.imports[0].kind);
    try std.testing.expectEqual(@as(u32, 0), module.import_function_count);
    try std.testing.expectEqual(@as(usize, 1), module.importedTables().len);
    const table = module.importedTables()[0];
    try std.testing.expect(std.mem.eql(u8, table.module_name, "env"));
    try std.testing.expect(std.mem.eql(u8, table.name, "tbl"));
    try std.testing.expectEqual(core_types.ValType.funcref, table.elem_type);
    try std.testing.expectEqual(@as(u32, 8), table.min);
    try std.testing.expectEqual(@as(?u32, 8), table.max);
}

test "#649 phase 2: instantiateWithOverrides shares imported tables across AOT cores" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const exporter_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> i32
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
        // function section: 1 local function of type 0
        0x03,
        0x02, 0x01, 0x00,
        // table section: (table (export "tbl") 1 1 funcref)
        0x04, 0x05, 0x01, 0x70, 0x01,
        0x01, 0x01,
        // export section: export table 0 as "tbl"
        0x07, 0x07, 0x01, 0x03, 't',  'b',
        'l',  0x01, 0x00,
        // elem section: active elem writing func 0 at slot 0
        0x09, 0x07, 0x01, 0x00, 0x41,
        0x00, 0x0b, 0x01, 0x00,
        // code section: i32.const 7; end
        0x0a, 0x06, 0x01, 0x04,
        0x00, 0x41, 0x07, 0x0b,
    };
    const importer_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> i32
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
        // import section: (import "env" "tbl" (table 1 1 funcref))
        0x02,
        0x0e, 0x01, 0x03, 'e',  'n',  'v',  0x03, 't',
        'b',  'l',  0x01, 0x70, 0x01, 0x01, 0x01,
        // function section: 1 local function of type 0
        0x03,
        0x02, 0x01, 0x00,
        // export section: export local func 0 as "invoke"
        0x07, 0x0a, 0x01, 0x06, 'i',
        'n',  'v',  'o',  'k',  'e',  0x00, 0x00,
        // code section: i32.const 0; call_indirect (type 0) table 0; end
        0x0a,
        0x09, 0x01, 0x07, 0x00, 0x41, 0x00, 0x11, 0x00,
        0x00, 0x0b,
    };

    const allocator = std.testing.allocator;
    const exporter_cwasm = try aot_harness.compileWasmToAot(allocator, &exporter_wasm);
    defer allocator.free(exporter_cwasm);
    const importer_cwasm = try aot_harness.compileWasmToAot(allocator, &importer_wasm);
    defer allocator.free(importer_cwasm);

    var exporter_module = try aot_loader_mod.load(exporter_cwasm, allocator);
    defer aot_loader_mod.unload(&exporter_module, allocator);
    var importer_module = try aot_loader_mod.load(importer_cwasm, allocator);
    defer aot_loader_mod.unload(&importer_module, allocator);

    const exporter_inst = try aot_runtime_mod.instantiate(&exporter_module, allocator);
    defer aot_runtime_mod.destroy(exporter_inst);
    try aot_runtime_mod.mapCodeExecutable(exporter_inst);

    const overrides = [_]?*core_types.TableInstance{exporter_inst.tables[0]};
    const importer_inst = try aot_runtime_mod.instantiateWithOverrides(&importer_module, allocator, &overrides, &.{}, &.{}, &.{});
    defer aot_runtime_mod.destroy(importer_inst);
    try std.testing.expectEqual(exporter_inst.tables[0], importer_inst.tables[0]);
    try std.testing.expect(!importer_inst.tables_owned[0]);
    try aot_runtime_mod.mapCodeExecutable(importer_inst);

    const fn_idx = aot_runtime_mod.findExportFunc(importer_inst, "invoke") orelse return error.TestFailed;
    const no_params = [_]core_types.ValType{};
    const result_types = [_]core_types.ValType{.i32};
    const no_args = [_]core_types.Value{};
    var results_buf: [1]aot_runtime_mod.ScalarResult = .{.{ .i32 = 0 }};
    const results = try aot_runtime_mod.callFuncScalar(
        importer_inst,
        fn_idx,
        &no_params,
        &result_types,
        &no_args,
        &results_buf,
    );
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(i32, 7), results[0].i32);
}

test "#660 item 2: AOT mutable imported global writes back across component cores" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const exporter_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> i32, () -> ()
        0x01, 0x08, 0x02, 0x60, 0x00, 0x01, 0x7f, 0x60, 0x00, 0x00,
        // function section: readG type 0, write_one type 1
        0x03, 0x03, 0x02, 0x00, 0x01,
        // global section: (global (mut i32) (i32.const 0))
        0x06, 0x06, 0x01, 0x7f, 0x01, 0x41, 0x00, 0x0b,
        // export section: global "g", funcs "readG" and "write_one"
        0x07, 0x19, 0x03,
        0x01, 'g', 0x03, 0x00,
        0x05, 'r', 'e', 'a', 'd', 'G', 0x00, 0x00,
        0x09, 'w', 'r', 'i', 't', 'e', '_', 'o', 'n', 'e', 0x00, 0x01,
        // code section: readG => global.get 0; write_one => global.set 0 to 1
        0x0a, 0x0d, 0x02,
        0x04, 0x00, 0x23, 0x00, 0x0b,
        0x06, 0x00, 0x41, 0x01, 0x24, 0x00, 0x0b,
    };
    const importer_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> (), () -> i32
        0x01, 0x08, 0x02, 0x60, 0x00, 0x00, 0x60, 0x00, 0x01, 0x7f,
        // import section: (import "a" "g" (global (mut i32)))
        0x02, 0x08, 0x01, 0x01, 'a', 0x01, 'g', 0x03, 0x7f, 0x01,
        // function section: write42 type 0, readG type 1
        0x03, 0x03, 0x02, 0x00, 0x01,
        // export section: funcs "write42" and "readG"
        0x07, 0x13, 0x02,
        0x07, 'w', 'r', 'i', 't', 'e', '4', '2', 0x00, 0x00,
        0x05, 'r', 'e', 'a', 'd', 'G', 0x00, 0x01,
        // code section: write42 => global.set 0 to 42; readG => global.get 0
        0x0a, 0x0d, 0x02,
        0x06, 0x00, 0x41, 0x2a, 0x24, 0x00, 0x0b,
        0x04, 0x00, 0x23, 0x00, 0x0b,
    };

    const allocator = std.testing.allocator;
    const exporter_cwasm = try aot_harness.compileWasmToAot(allocator, &exporter_wasm);
    defer allocator.free(exporter_cwasm);
    const importer_cwasm = try aot_harness.compileWasmToAot(allocator, &importer_wasm);
    defer allocator.free(importer_cwasm);

    const core_modules = [_]ctypes.CoreModule{
        .{ .data = &exporter_wasm },
        .{ .data = &importer_wasm },
    };
    const importer_args = [_]ctypes.CoreInstantiateArg{.{ .name = "a", .instance_idx = 0 }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
        .{ .instantiate = .{ .module_idx = 1, .args = &importer_args } },
    };
    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const pcs = [_]instance.PrecompiledCore{
        .{ .module_idx = 0, .cwasm_bytes = exporter_cwasm },
        .{ .module_idx = 1, .cwasm_bytes = importer_cwasm },
    };
    const inst = try instance.instantiateWithOptions(&component, allocator, .{
        .precompiled_cores = &pcs,
        .aot_only = true,
    });
    defer inst.deinit();

    try std.testing.expectEqual(@as(usize, 2), inst.core_instances.len);
    const exporter_ai = inst.core_instances[0].aot_inst orelse return error.TestFailed;
    const importer_ai = inst.core_instances[1].aot_inst orelse return error.TestFailed;
    try std.testing.expectEqual(exporter_ai.globals[0], importer_ai.globals[0]);
    try std.testing.expect(!importer_ai.globals_owned[0]);

    const a_read = aot_runtime_mod.findExportFunc(exporter_ai, "readG") orelse return error.TestFailed;
    const a_write_one = aot_runtime_mod.findExportFunc(exporter_ai, "write_one") orelse return error.TestFailed;
    const b_write42 = aot_runtime_mod.findExportFunc(importer_ai, "write42") orelse return error.TestFailed;
    const b_read = aot_runtime_mod.findExportFunc(importer_ai, "readG") orelse return error.TestFailed;
    const no_params = [_]core_types.ValType{};
    const no_results = [_]core_types.ValType{};
    const i32_results = [_]core_types.ValType{.i32};
    const no_args = [_]core_types.Value{};
    var results_buf: [1]aot_runtime_mod.ScalarResult = .{.{ .i32 = 0 }};

    _ = try aot_runtime_mod.callFuncScalar(importer_ai, b_write42, &no_params, &no_results, &no_args, &results_buf);
    const a_after_b_write = try aot_runtime_mod.callFuncScalar(exporter_ai, a_read, &no_params, &i32_results, &no_args, &results_buf);
    try std.testing.expectEqual(@as(i32, 42), a_after_b_write[0].i32);

    _ = try aot_runtime_mod.callFuncScalar(exporter_ai, a_write_one, &no_params, &no_results, &no_args, &results_buf);
    const b_after_a_write = try aot_runtime_mod.callFuncScalar(importer_ai, b_read, &no_params, &i32_results, &no_args, &results_buf);
    try std.testing.expectEqual(@as(i32, 1), b_after_a_write[0].i32);
}

test "#625 phase 1: instantiateWithOptions loads + runs an AOT core" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    // Minimal core module exporting one `i32 -> i32` function and a memory.
    //   (module
    //     (memory (export "memory") 1)
    //     (func (export "add42") (param i32) (result i32)
    //       local.get 0 i32.const 42 i32.add)
    //   )
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: (i32) -> i32
        0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
        // function section: 1 fn, type 0
        0x03, 0x02, 0x01, 0x00,
        // memory section: 1 memory, min=1
        0x05, 0x03, 0x01, 0x00,
        0x01,
        // export section: "memory" (mem 0), "add42" (func 0)
        0x07, 0x12, 0x02, 0x06, 'm',  'e',  'm',
        'o',  'r',  'y',  0x02, 0x00, 0x05, 'a',  'd',
        'd',  '4',  '2',  0x00, 0x00,
        // code section: local.get 0; i32.const 42; i32.add; end
        0x0a, 0x09, 0x01,
        0x07, 0x00, 0x20, 0x00, 0x41, 0x2a, 0x6a, 0x0b,
    };

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &core_wasm);
    defer allocator.free(cwasm_bytes);

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };
    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };

    const pcs = [_]instance.PrecompiledCore{
        .{ .module_idx = 0, .cwasm_bytes = cwasm_bytes },
    };
    const inst = try instance.instantiateWithOptions(&component, allocator, .{
        .precompiled_cores = &pcs,
    });
    defer inst.deinit();

    // The AOT backend should be populated, the interp one should not.
    try std.testing.expectEqual(@as(usize, 1), inst.core_instances.len);
    try std.testing.expect(inst.core_instances[0].module_inst == null);
    const ai = inst.core_instances[0].aot_inst orelse return error.TestFailed;

    // Backend-agnostic memory lookup must surface the AOT memory.
    const mem = inst.firstBackendMemory() orelse return error.TestFailed;
    try std.testing.expect(mem.data.len >= core_types.MemoryInstance.page_size);

    // The exported function should be discoverable through the backend
    // adapter — same call shape as a future canon-ABI dispatch would use.
    const be = inst.core_instances[0].backend() orelse return error.TestFailed;
    const fn_idx = be.findExportFunc("add42") orelse return error.TestFailed;

    // And we can drive it through the AOT runtime directly. This proves
    // the loaded artifact is actually executable end-to-end on this
    // host; the canon-ABI dispatch that ties this into
    // `callComponentFuncByLocal` lands in #625 phase 3.
    const param_types = [_]core_types.ValType{.i32};
    const result_types = [_]core_types.ValType{.i32};
    const args = [_]core_types.Value{.{ .i32 = 100 }};
    var results_buf: [1]aot_runtime_mod.ScalarResult = .{.{ .i32 = 0 }};
    const results = try aot_runtime_mod.callFuncScalar(
        ai,
        fn_idx,
        &param_types,
        &result_types,
        &args,
        &results_buf,
    );
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(i32, 142), results[0].i32);
}

test "#649 phase 3: AOT loader retains imported memory descriptors" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    // (module
    //   (import "env" "mem" (memory 2 7))
    //   (func (export "noop"))
    // )
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // import section: "env" "mem" memory min=2 max=7
        0x02, 0x0d,
        0x01, 0x03, 'e',  'n',  'v',  0x03, 'm',  'e',
        'm',  0x02, 0x01, 0x02, 0x07,
        // function section: 1 fn type 0
        0x03, 0x02, 0x01, 0x00,
        // export section: "noop" func 0
        0x07, 0x08, 0x01, 0x04, 'n',  'o',  'o',  'p',
        0x00, 0x00,
        // code section: empty body
        0x0a, 0x04, 0x01, 0x02, 0x00, 0x0b,
    };

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &core_wasm);
    defer allocator.free(cwasm_bytes);

    const module = try aot_loader_mod.load(cwasm_bytes, allocator);
    defer aot_loader_mod.unload(&module, allocator);

    try std.testing.expectEqual(@as(usize, 1), module.imports.len);
    try std.testing.expectEqual(core_types.ExternalKind.memory, module.imports[0].kind);
    try std.testing.expectEqual(@as(u32, 0), module.import_function_count);
    try std.testing.expectEqual(@as(usize, 1), module.importedMemories().len);
    const mem = module.importedMemories()[0];
    try std.testing.expect(std.mem.eql(u8, mem.module_name, "env"));
    try std.testing.expect(std.mem.eql(u8, mem.name, "mem"));
    try std.testing.expectEqual(@as(u32, 2), mem.min);
    try std.testing.expectEqual(@as(?u32, 7), mem.max);
    try std.testing.expectEqual(false, mem.is64);
}

test "#649 phase 3: instantiateWithOverrides shares imported memory across AOT cores" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    // exporter:
    //   (module
    //     (memory (export "mem") 1)
    //     (data (i32.const 0) "\55\44\33\22")
    //   )
    const exporter_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // memory section: 1 memory min=1
        0x05, 0x03, 0x01, 0x00, 0x01,
        // export section: "mem" mem 0
        0x07, 0x07, 0x01, 0x03, 'm',  'e',  'm',  0x02,
        0x00,
        // data section: 1 active segment, memory 0, offset 0, 4 bytes
        0x0b, 0x0a,
        0x01, 0x00, 0x41, 0x00, 0x0b, 0x04, 0x55, 0x44,
        0x33, 0x22,
    };

    // importer:
    //   (module
    //     (import "env" "mem" (memory 1))
    //     (func (export "read") (result i32) i32.const 0 i32.load)
    //   )
    const importer_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> i32
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
        // import section: "env" "mem" memory min=1
        0x02, 0x0c,
        0x01, 0x03, 'e',  'n',  'v',  0x03, 'm',  'e',
        'm',  0x02, 0x00, 0x01,
        // function section: 1 fn type 0
        0x03, 0x02, 0x01, 0x00,
        // export section: "read" func 0
        0x07, 0x08, 0x01, 0x04, 'r',  'e',  'a',  'd',
        0x00, 0x00,
        // code section: locals=0; i32.const 0; i32.load align=2 offset=0; end
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x41, 0x00, 0x28,
        0x02, 0x00, 0x0b,
    };

    const allocator = std.testing.allocator;
    const exporter_cwasm = try aot_harness.compileWasmToAot(allocator, &exporter_wasm);
    defer allocator.free(exporter_cwasm);
    const importer_cwasm = try aot_harness.compileWasmToAot(allocator, &importer_wasm);
    defer allocator.free(importer_cwasm);

    var exporter_module = try aot_loader_mod.load(exporter_cwasm, allocator);
    defer aot_loader_mod.unload(&exporter_module, allocator);
    var importer_module = try aot_loader_mod.load(importer_cwasm, allocator);
    defer aot_loader_mod.unload(&importer_module, allocator);

    const exporter_inst = try aot_runtime_mod.instantiate(&exporter_module, allocator);
    defer aot_runtime_mod.destroy(exporter_inst);

    // Sentinel from the exporter's data segment must be live in its memory.
    try std.testing.expectEqual(@as(u8, 0x55), exporter_inst.memories[0].data[0]);
    try std.testing.expectEqual(@as(u8, 0x22), exporter_inst.memories[0].data[3]);

    const overrides = [_]?*core_types.MemoryInstance{exporter_inst.memories[0]};
    const importer_inst = try aot_runtime_mod.instantiateWithOverrides(&importer_module, allocator, &.{}, &overrides, &.{}, &.{});
    defer aot_runtime_mod.destroy(importer_inst);
    try std.testing.expectEqual(exporter_inst.memories[0], importer_inst.memories[0]);
    try std.testing.expect(!importer_inst.memories_owned[0]);
    try aot_runtime_mod.mapCodeExecutable(importer_inst);

    const fn_idx = aot_runtime_mod.findExportFunc(importer_inst, "read") orelse return error.TestFailed;
    const no_params = [_]core_types.ValType{};
    const result_types = [_]core_types.ValType{.i32};
    const no_args = [_]core_types.Value{};
    var results_buf: [1]aot_runtime_mod.ScalarResult = .{.{ .i32 = 0 }};
    const results = try aot_runtime_mod.callFuncScalar(
        importer_inst,
        fn_idx,
        &no_params,
        &result_types,
        &no_args,
        &results_buf,
    );
    try std.testing.expectEqual(@as(usize, 1), results.len);
    // little-endian {0x55, 0x44, 0x33, 0x22} = 0x22334455
    try std.testing.expectEqual(@as(i32, 0x22334455), results[0].i32);
}

test "#649 phase 4: AOT loader retains imported global descriptors" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    // (module
    //   (import "env" "g" (global i32))
    //   (func (export "noop"))
    // )
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // import section: "env" "g" global i32 immutable
        0x02, 0x0a,
        0x01, 0x03, 'e',  'n',  'v',  0x01, 'g',  0x03,
        0x7f, 0x00,
        // function section: 1 fn type 0
        0x03, 0x02, 0x01, 0x00,
        // export section: "noop" func 0
        0x07, 0x08, 0x01, 0x04, 'n',  'o',  'o',  'p',
        0x00, 0x00,
        // code section: empty body
        0x0a, 0x04, 0x01, 0x02, 0x00, 0x0b,
    };

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &core_wasm);
    defer allocator.free(cwasm_bytes);

    const module = try aot_loader_mod.load(cwasm_bytes, allocator);
    defer aot_loader_mod.unload(&module, allocator);

    try std.testing.expectEqual(@as(usize, 1), module.imports.len);
    try std.testing.expectEqual(core_types.ExternalKind.global, module.imports[0].kind);
    try std.testing.expectEqual(@as(u32, 0), module.import_function_count);
    try std.testing.expectEqual(@as(usize, 1), module.importedGlobals().len);
    const g = module.importedGlobals()[0];
    try std.testing.expect(std.mem.eql(u8, g.module_name, "env"));
    try std.testing.expect(std.mem.eql(u8, g.name, "g"));
    try std.testing.expectEqual(core_types.ValType.i32, g.val_type);
    try std.testing.expectEqual(false, g.mutable);
}

test "#649 phase 4: instantiateWithOverrides shares imported globals across AOT cores" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    // exporter:
    //   (module
    //     (global (export "g") i32 (i32.const 0x55))
    //   )
    const exporter_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // global section: 1 global, i32 immutable, init i32.const 0x55 (LEB: 0xD5 0x00)
        0x06, 0x07, 0x01, 0x7f, 0x00, 0x41, 0xd5, 0x00,
        0x0b,
        // export section: "g" global 0
        0x07, 0x05, 0x01, 0x01, 'g',  0x03, 0x00,
    };

    // importer:
    //   (module
    //     (import "env" "g" (global i32))
    //     (func (export "read") (result i32) global.get 0)
    //   )
    const importer_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> i32
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
        // import section: "env" "g" global i32 immutable
        0x02, 0x0a,
        0x01, 0x03, 'e',  'n',  'v',  0x01, 'g',  0x03,
        0x7f, 0x00,
        // function section: 1 fn type 0
        0x03, 0x02, 0x01, 0x00,
        // export section: "read" func 0
        0x07, 0x08, 0x01, 0x04, 'r',  'e',  'a',  'd',
        0x00, 0x00,
        // code section: locals=0; global.get 0; end
        0x0a, 0x06, 0x01, 0x04, 0x00, 0x23, 0x00, 0x0b,
    };

    const allocator = std.testing.allocator;
    const exporter_cwasm = try aot_harness.compileWasmToAot(allocator, &exporter_wasm);
    defer allocator.free(exporter_cwasm);
    const importer_cwasm = try aot_harness.compileWasmToAot(allocator, &importer_wasm);
    defer allocator.free(importer_cwasm);

    var exporter_module = try aot_loader_mod.load(exporter_cwasm, allocator);
    defer aot_loader_mod.unload(&exporter_module, allocator);
    var importer_module = try aot_loader_mod.load(importer_cwasm, allocator);
    defer aot_loader_mod.unload(&importer_module, allocator);

    const exporter_inst = try aot_runtime_mod.instantiate(&exporter_module, allocator);
    defer aot_runtime_mod.destroy(exporter_inst);

    // Exporter's global must hold the i32.const literal value after init.
    try std.testing.expectEqual(@as(i32, 0x55), exporter_inst.globals[0].value.i32);

    const overrides = [_]?*core_types.GlobalInstance{exporter_inst.globals[0]};
    const importer_inst = try aot_runtime_mod.instantiateWithOverrides(&importer_module, allocator, &.{}, &.{}, &overrides, &.{});
    defer aot_runtime_mod.destroy(importer_inst);
    try std.testing.expectEqual(exporter_inst.globals[0], importer_inst.globals[0]);
    try std.testing.expect(!importer_inst.globals_owned[0]);
    try aot_runtime_mod.mapCodeExecutable(importer_inst);

    const fn_idx = aot_runtime_mod.findExportFunc(importer_inst, "read") orelse return error.TestFailed;
    const no_params = [_]core_types.ValType{};
    const result_types = [_]core_types.ValType{.i32};
    const no_args = [_]core_types.Value{};
    var results_buf: [1]aot_runtime_mod.ScalarResult = .{.{ .i32 = 0 }};
    const results = try aot_runtime_mod.callFuncScalar(
        importer_inst,
        fn_idx,
        &no_params,
        &result_types,
        &no_args,
        &results_buf,
    );
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(i32, 0x55), results[0].i32);
}

test "#649 phase 5: cross-AOT shared memory + table + global in one importer" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    // exporter:
    //   (module
    //     (func (result i32) i32.const 100)
    //     (table (export "tbl") 1 1 funcref)
    //     (memory (export "mem") 1)
    //     (global (export "g") i32 (i32.const 3))
    //     (export "f" (func 0))
    //     (elem (i32.const 0) func 0)
    //     (data (i32.const 0) "\07\00\00\00")
    //   )
    const exporter_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> i32
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
        // function section: 1 fn type 0
        0x03, 0x02, 0x01, 0x00,
        // table section: 1 table funcref min=1 max=1
        0x04, 0x05, 0x01, 0x70, 0x01, 0x01, 0x01,
        // memory section: 1 memory min=1
        0x05, 0x03, 0x01, 0x00, 0x01,
        // global section: 1 global i32 immutable init=3
        0x06, 0x06, 0x01, 0x7f, 0x00, 0x41, 0x03, 0x0b,
        // export section: "mem", "tbl", "g"
        0x07, 0x11, 0x03,
        0x03, 'm',  'e',  'm',  0x02, 0x00,
        0x03, 't',  'b',  'l',  0x01, 0x00,
        0x01, 'g',  0x03, 0x00,
        // elem section: 1 active segment table 0 offset 0, func [0]
        0x09, 0x07, 0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x00,
        // code section: 1 fn, body = i32.const 100; end (LEB 0xE4 0x00)
        0x0a, 0x07, 0x01, 0x05, 0x00, 0x41, 0xe4, 0x00, 0x0b,
        // data section: 1 active segment mem 0 offset 0 bytes [0x07 0x00 0x00 0x00]
        0x0b, 0x0a, 0x01, 0x00, 0x41, 0x00, 0x0b, 0x04, 0x07, 0x00, 0x00, 0x00,
    };

    // importer:
    //   (module
    //     (type (func (result i32)))
    //     (import "env" "mem" (memory 1))
    //     (import "env" "tbl" (table 1 1 funcref))
    //     (import "env" "g" (global i32))
    //     (func (export "compute") (result i32)
    //       i32.const 0 i32.load              ;; mem[0] = 7
    //       global.get 0                       ;; g = 3
    //       i32.add                            ;; 10
    //       i32.const 0 call_indirect (type 0) ;; tbl[0]() = 100
    //       i32.add)                           ;; 110
    //   )
    const importer_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> i32
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
        // import section: 3 imports
        //   "env" "mem" memory min=1                          (11 bytes)
        //   "env" "tbl" table funcref min=1 max=1             (13 bytes)
        //   "env" "g" global i32 immutable                    ( 9 bytes)
        // payload = count(1) + 11 + 13 + 9 = 34 = 0x22
        0x02, 0x22,
        0x03,
        0x03, 'e',  'n',  'v',  0x03, 'm',  'e',  'm',  0x02, 0x00, 0x01,
        0x03, 'e',  'n',  'v',  0x03, 't',  'b',  'l',  0x01, 0x70, 0x01, 0x01, 0x01,
        0x03, 'e',  'n',  'v',  0x01, 'g',  0x03, 0x7f, 0x00,
        // function section: 1 fn type 0
        0x03, 0x02, 0x01, 0x00,
        // export section: "compute" func 0
        0x07, 0x0b, 0x01, 0x07, 'c',  'o',  'm',  'p',  'u',  't',  'e',  0x00, 0x00,
        // code section: locals=0; (load + add + call_indirect + add)
        0x0a, 0x12, 0x01, 0x10, 0x00,
        0x41, 0x00, 0x28, 0x02, 0x00,
        0x23, 0x00,
        0x6a,
        0x41, 0x00, 0x11, 0x00, 0x00,
        0x6a,
        0x0b,
    };

    const allocator = std.testing.allocator;
    const exporter_cwasm = try aot_harness.compileWasmToAot(allocator, &exporter_wasm);
    defer allocator.free(exporter_cwasm);
    const importer_cwasm = try aot_harness.compileWasmToAot(allocator, &importer_wasm);
    defer allocator.free(importer_cwasm);

    var exporter_module = try aot_loader_mod.load(exporter_cwasm, allocator);
    defer aot_loader_mod.unload(&exporter_module, allocator);
    var importer_module = try aot_loader_mod.load(importer_cwasm, allocator);
    defer aot_loader_mod.unload(&importer_module, allocator);

    const exporter_inst = try aot_runtime_mod.instantiate(&exporter_module, allocator);
    defer aot_runtime_mod.destroy(exporter_inst);
    try aot_runtime_mod.mapCodeExecutable(exporter_inst);

    const table_overrides = [_]?*core_types.TableInstance{exporter_inst.tables[0]};
    const memory_overrides = [_]?*core_types.MemoryInstance{exporter_inst.memories[0]};
    const global_overrides = [_]?*core_types.GlobalInstance{exporter_inst.globals[0]};
    const importer_inst = try aot_runtime_mod.instantiateWithOverrides(
        &importer_module,
        allocator,
        &table_overrides,
        &memory_overrides,
        &global_overrides,
        &.{},
    );
    defer aot_runtime_mod.destroy(importer_inst);
    try std.testing.expectEqual(exporter_inst.tables[0], importer_inst.tables[0]);
    try std.testing.expectEqual(exporter_inst.memories[0], importer_inst.memories[0]);
    try std.testing.expectEqual(exporter_inst.globals[0], importer_inst.globals[0]);
    try std.testing.expect(!importer_inst.tables_owned[0]);
    try std.testing.expect(!importer_inst.memories_owned[0]);
    try std.testing.expect(!importer_inst.globals_owned[0]);
    try aot_runtime_mod.mapCodeExecutable(importer_inst);

    const fn_idx = aot_runtime_mod.findExportFunc(importer_inst, "compute") orelse return error.TestFailed;
    const no_params = [_]core_types.ValType{};
    const result_types = [_]core_types.ValType{.i32};
    const no_args = [_]core_types.Value{};
    var results_buf: [1]aot_runtime_mod.ScalarResult = .{.{ .i32 = 0 }};
    const results = try aot_runtime_mod.callFuncScalar(
        importer_inst,
        fn_idx,
        &no_params,
        &result_types,
        &no_args,
        &results_buf,
    );
    try std.testing.expectEqual(@as(usize, 1), results.len);
    // mem[0]=7 + global=3 + tbl[0]()=100 = 110
    try std.testing.expectEqual(@as(i32, 110), results[0].i32);
}
