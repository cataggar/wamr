//! #625 phase 3 — AOT canon.lift dispatch smoke test.
//!
//! Builds a component literal with one core module (the same
//! `add42` AOT smoke fixture used in phase 1/2), a `canon.lift` of
//! its `add42` export typed as `(func (param "x" s32) (result s32))`,
//! and a top-level `.func` export pointing at that canon. Drives the
//! exported function through `callComponentFuncByLocal`, asserting
//! the dispatch flows through the new AOT scalar fast path in
//! `executor.zig`.
//!
//! Pre-phase-3, the same call would hit
//! `error.CoreInstanceNotAvailable` because the executor's
//! `module_inst orelse` guard refused AOT-only cores.

const std = @import("std");
const wamr = @import("wamr");
const aot_harness = @import("aot_harness.zig");

const instance = wamr.component_instance;
const ctypes = wamr.component_types;
const executor = wamr.component_executor;
const abi = wamr.canonical_abi;

const core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x05, 0x03, 0x01, 0x00, 0x01,
    0x07, 0x12, 0x02,
    0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00,
    0x05, 'a', 'd', 'd', '4', '2', 0x00, 0x00,
    0x0a, 0x09, 0x01, 0x07, 0x00,
    0x20, 0x00,
    0x41, 0x2a,
    0x6a,
    0x0b,
};

test "#625 phase 3: canon.lift dispatches onto AOT core" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &core_wasm);
    defer allocator.free(cwasm_bytes);

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };

    // Component-level types: one func type `(s32) -> s32`.
    const params = [_]ctypes.NamedValType{.{ .name = "x", .type = .s32 }};
    const types = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };

    // canon.lift of core_func 0 inside core_inst 0 → component func 0,
    // typed by component type 0.
    const lift_opts = [_]ctypes.CanonOpt{};
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 0, .opts = &lift_opts } },
    };

    // Export the lifted func as "add42" on the component.
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "add42", .desc = .{ .func = 0 }, .sort_idx = .{ .sort = .func, .idx = 0 } },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &types,
        .canons = &canons,
        .imports = &.{},
        .exports = &exports,
    };

    const pcs = [_]instance.PrecompiledCore{
        .{ .module_idx = 0, .cwasm_bytes = cwasm_bytes },
    };
    const inst = try instance.instantiateWithOptions(&component, allocator, .{
        .precompiled_cores = &pcs,
    });
    defer inst.deinit();

    // Confirm the AOT branch was actually chosen.
    try std.testing.expect(inst.core_instances[0].module_inst == null);
    try std.testing.expect(inst.core_instances[0].aot_inst != null);

    const ef = inst.getExport("add42") orelse return error.TestFailed;
    const local = switch (ef) {
        .local => |l| l,
        else => return error.TestFailed,
    };

    const args = [_]abi.InterfaceValue{.{ .s32 = 100 }};
    var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
    try executor.callComponentFuncByLocal(inst, local, &args, &results, allocator);
    try std.testing.expectEqual(@as(i32, 142), results[0].s32);
}

test "#625 phase 3: AOT path rejects unsupported shapes cleanly" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &core_wasm);
    defer allocator.free(cwasm_bytes);

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };

    // Declare the lift as taking a `string` param (which flattens to
    // 2 i32 slots + needs memory) so the scalar fast path bails with
    // AotPathUnsupported rather than producing wrong results.
    const params = [_]ctypes.NamedValType{.{ .name = "s", .type = .string }};
    const types = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };
    const lift_opts = [_]ctypes.CanonOpt{};
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 0, .opts = &lift_opts } },
    };
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "go", .desc = .{ .func = 0 }, .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &types,
        .canons = &canons,
        .imports = &.{},
        .exports = &exports,
    };

    const pcs = [_]instance.PrecompiledCore{
        .{ .module_idx = 0, .cwasm_bytes = cwasm_bytes },
    };
    const inst = try instance.instantiateWithOptions(&component, allocator, .{
        .precompiled_cores = &pcs,
    });
    defer inst.deinit();

    const ef = inst.getExport("go") orelse return error.TestFailed;
    const local = switch (ef) {
        .local => |l| l,
        else => return error.TestFailed,
    };

    const args = [_]abi.InterfaceValue{.{ .string = .{ .ptr = 0, .len = 0 } }};
    var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
    try std.testing.expectError(
        error.AotPathUnsupported,
        executor.callComponentFuncByLocal(inst, local, &args, &results, allocator),
    );
}

// #650 phase A — `result<(),()>` lifts onto AOT scalar fast path.
//
// Core wasm exports `(func "ok" (result i32) i32.const 0)` and
// `(func "err" (result i32) i32.const 1)`; canon.lift retypes them
// as `(func) -> result<(), ()>`, which flattens to a single i32
// discriminant slot. Pre-#650, `liftScalarResult` only knew
// primitives so this hit `error.AotPathUnsupported` even though
// the shape lives in one i32. After #650 phase A, both calls
// surface as `result_val { is_ok=true/false, payload=null }`.
//
// This is the headline case for wit-bindgen-emitted CLI
// components whose `wasi:cli/run.run` lifts as
// `func() -> result<_, _>`.
const result_core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    // type: () -> i32
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    // funcs: 2 funcs, both type 0
    0x03, 0x03, 0x02, 0x00, 0x00,
    // memory: 1 page
    0x05, 0x03, 0x01, 0x00, 0x01,
    // exports: memory, ok, err  (count=3, payload = 9+5+6 = 20 -> total 21)
    0x07, 0x15, 0x03,
    0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00,
    0x02, 'o', 'k', 0x00, 0x00,
    0x03, 'e', 'r', 'r', 0x00, 0x01,
    // code: 2 funcs, each body 4 bytes (locals=0, i32.const X, end) -> count + 2*(1+4) = 11
    0x0a, 0x0b, 0x02,
    0x04, 0x00, 0x41, 0x00, 0x0b,
    0x04, 0x00, 0x41, 0x01, 0x0b,
};

test "#650 phase A: AOT lifts result<(),()> via scalar fast path" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &result_core_wasm);
    defer allocator.free(cwasm_bytes);

    const core_modules = [_]ctypes.CoreModule{.{ .data = &result_core_wasm }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };

    // type 0: result<(),()>; type 1: () -> result<(),()> (referencing type 0).
    const types = [_]ctypes.TypeDef{
        .{ .result = .{ .ok = null, .err = null } },
        .{ .func = .{
            .params = &.{},
            .results = .{ .unnamed = .{ .result = 0 } },
        } },
    };
    const lift_opts = [_]ctypes.CanonOpt{};
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 1, .opts = &lift_opts } },
        .{ .lift = .{ .core_func_idx = 1, .type_idx = 1, .opts = &lift_opts } },
    };
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "ok", .desc = .{ .func = 1 }, .sort_idx = .{ .sort = .func, .idx = 0 } },
        .{ .name = "err", .desc = .{ .func = 1 }, .sort_idx = .{ .sort = .func, .idx = 1 } },
    };
    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &types,
        .canons = &canons,
        .imports = &.{},
        .exports = &exports,
    };

    const pcs = [_]instance.PrecompiledCore{
        .{ .module_idx = 0, .cwasm_bytes = cwasm_bytes },
    };
    const inst = try instance.instantiateWithOptions(&component, allocator, .{
        .precompiled_cores = &pcs,
    });
    defer inst.deinit();

    try std.testing.expect(inst.core_instances[0].aot_inst != null);

    // ok arm
    {
        const ef = inst.getExport("ok") orelse return error.TestFailed;
        const local = switch (ef) {
            .local => |l| l,
            else => return error.TestFailed,
        };
        var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
        try executor.callComponentFuncByLocal(inst, local, &.{}, &results, allocator);
        const rv = results[0].result_val;
        try std.testing.expect(rv.is_ok);
        try std.testing.expect(rv.payload == null);
    }

    // err arm
    {
        const ef = inst.getExport("err") orelse return error.TestFailed;
        const local = switch (ef) {
            .local => |l| l,
            else => return error.TestFailed,
        };
        var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
        try executor.callComponentFuncByLocal(inst, local, &.{}, &results, allocator);
        const rv = results[0].result_val;
        try std.testing.expect(!rv.is_ok);
        try std.testing.expect(rv.payload == null);
    }
}
