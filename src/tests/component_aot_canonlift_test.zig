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
