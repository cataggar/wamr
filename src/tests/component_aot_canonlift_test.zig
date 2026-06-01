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
const builtin = @import("builtin");
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

    // After commit 2 (CallFrame refactor) compound aggregate params are
    // handled by the unified canon-ABI walk, so the previous
    // `tuple<u32,u32>` rejection no longer applies. The remaining
    // guard-rail is the canon-ABI requirement that spilling params to
    // memory (>MAX_FLAT_PARAMS) needs a `realloc` binding. Declare a
    // wide param tuple (17 u32 fields → 17 flat slots > MAX_FLAT_PARAMS)
    // with NO realloc in lift_opts; the AOT path must surface
    // `error.ReallocNotAvailable`.
    const tuple_fields = [_]ctypes.ValType{
        .u32, .u32, .u32, .u32, .u32, .u32, .u32, .u32, .u32,
        .u32, .u32, .u32, .u32, .u32, .u32, .u32, .u32,
    };
    const params = [_]ctypes.NamedValType{.{ .name = "p", .type = .{ .tuple = 0 } }};
    const types = [_]ctypes.TypeDef{
        .{ .tuple = .{ .fields = &tuple_fields } },
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };
    const lift_opts = [_]ctypes.CanonOpt{}; // no realloc → spill must fail
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 1, .opts = &lift_opts } },
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

    var tup_field: [17]abi.InterfaceValue = undefined;
    for (&tup_field) |*v| v.* = .{ .u32 = 0 };
    const args = [_]abi.InterfaceValue{.{ .tuple_val = &tup_field }};
    var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
    try std.testing.expectError(
        error.ReallocNotAvailable,
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

// #650 phase B.1 — retptr-based compound returns on AOT.
//
// Core wasm exports two functions:
//   * `realloc(orig, sz, align, new_sz)` — fixed-bump allocator
//      that always returns 16, so the lifted return tuple lands
//      at offset 16 in linear memory.
//   * `wp()` — writes 0x000000AA at offset 16 and 0x000000BB at
//      offset 20, returns the retptr (16) as a single i32 result.
//
// canon.lift retypes `wp` as `func() -> tuple<u32, u32>`. The lifted
// type flattens to 2 i32 slots, exceeding `MAX_FLAT_RESULTS=1`, so
// canon ABI v1 spilled-result convention applies: the core function
// is emitted as `() -> i32` (CALLEE-allocates retptr) — the callee
// allocates the buffer via its own realloc and returns the pointer.
// The lift trampoline pops the i32 retptr and reads both u32 fields
// back from linear memory via `loadInterfaceValue`.
//
// Pre-phase-B.1 this lift hit `error.AotPathUnsupported` on
// `result_types.len > MAX_FLAT_RESULTS` (which conflated interface
// arity with flat slot count anyway). Pre-#719 the AOT path used
// caller-allocates (passing retptr as a trailing arg, expecting void
// return) which mismatched what wit-bindgen actually emits.
const retptr_core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    // type section: 2 types (size = 9 + 4 = 13)
    0x01, 0x0d, 0x02,
    0x60, 0x04, 0x7f, 0x7f, 0x7f, 0x7f, 0x01, 0x7f, // realloc: (i32,i32,i32,i32) -> i32
    0x60, 0x00, 0x01, 0x7f, // wp: () -> i32
    // func section: 2 funcs
    0x03, 0x03, 0x02, 0x00, 0x01,
    // memory section: 1 page
    0x05, 0x03, 0x01, 0x00, 0x01,
    // export section: memory, realloc, wp (payload=9+10+5+count(1)=25)
    0x07, 0x19, 0x03,
    0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00,
    0x07, 'r', 'e', 'a', 'l', 'l', 'o', 'c', 0x00, 0x00,
    0x02, 'w', 'p', 0x00, 0x01,
    // code section: 2 funcs (payload = count(1) + body0(5) + body1(21) = 27)
    0x0a, 0x1b, 0x02,
    0x04, 0x00, 0x41, 0x10, 0x0b, // realloc: i32.const 16; end
    // wp body (size=20): locals(0) +
    //   i32.const 16; i32.const 0xAA; i32.store offset=0
    //   i32.const 16; i32.const 0xBB; i32.store offset=4
    //   i32.const 16          ;; return retptr
    //   end
    0x14,
    0x00,
    0x41, 0x10, 0x41, 0xaa, 0x01, 0x36, 0x02, 0x00,
    0x41, 0x10, 0x41, 0xbb, 0x01, 0x36, 0x02, 0x04,
    0x41, 0x10,
    0x0b,
};

test "#650 phase B.1: AOT lifts tuple<u32,u32> via retptr" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &retptr_core_wasm);
    defer allocator.free(cwasm_bytes);

    const core_modules = [_]ctypes.CoreModule{.{ .data = &retptr_core_wasm }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };

    // type 0: tuple<u32, u32>; type 1: () -> tuple<u32, u32> (referencing type 0).
    const tuple_fields = [_]ctypes.ValType{ .u32, .u32 };
    const types = [_]ctypes.TypeDef{
        .{ .tuple = .{ .fields = &tuple_fields } },
        .{ .func = .{
            .params = &.{},
            .results = .{ .unnamed = .{ .tuple = 0 } },
        } },
    };
    // Canon-lift `wp` (core_func_idx=1) with realloc bound to core_func_idx=0.
    const lift_opts = [_]ctypes.CanonOpt{
        .{ .realloc = 0 },
    };
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 1, .type_idx = 1, .opts = &lift_opts } },
    };
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "wp", .desc = .{ .func = 1 }, .sort_idx = .{ .sort = .func, .idx = 0 } },
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

    const ef = inst.getExport("wp") orelse return error.TestFailed;
    const local = switch (ef) {
        .local => |l| l,
        else => return error.TestFailed,
    };
    var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
    try executor.callComponentFuncByLocal(inst, local, &.{}, &results, allocator);
    defer results[0].deinit(allocator);

    const tup = results[0].tuple_val;
    try std.testing.expectEqual(@as(usize, 2), tup.len);
    try std.testing.expectEqual(@as(u32, 0xAA), tup[0].u32);
    try std.testing.expectEqual(@as(u32, 0xBB), tup[1].u32);
}

// #650 phase B.2 — multi-slot flat params on AOT.
//
// Core wasm exports `(func "len" (param i32 i32) (result i32))` that
// returns its second i32 arg (i.e. the string length). canon.lift
// retypes it as `func(s: string) -> u32`. The string param flattens
// to 2 i32 slots, exceeding what the pre-B.2 single-slot lowerScalarArg
// could emit. Phase B.2's `lowerFlatRecur` writes both slots into the
// AOT arg buffer.
const string_param_core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    // type: (i32 i32) -> i32
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
    // funcs: 1 func type 0
    0x03, 0x02, 0x01, 0x00,
    // memory: 1 page
    0x05, 0x03, 0x01, 0x00, 0x01,
    // exports: memory, len  (count=3 -> wait 2; payload: 9 + 6 = 15 + 1)
    0x07, 0x10, 0x02,
    0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00,
    0x03, 'l', 'e', 'n', 0x00, 0x00,
    // code: 1 func, body = locals(1 byte) + local.get 1 + end (4 bytes)
    0x0a, 0x06, 0x01,
    0x04, 0x00, 0x20, 0x01, 0x0b,
};

test "#650 phase B.2: AOT lowers string param (multi-slot flat)" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &string_param_core_wasm);
    defer allocator.free(cwasm_bytes);

    const core_modules = [_]ctypes.CoreModule{.{ .data = &string_param_core_wasm }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };

    const params = [_]ctypes.NamedValType{.{ .name = "s", .type = .string }};
    const types = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .u32 } } },
    };
    const lift_opts = [_]ctypes.CanonOpt{};
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 0, .opts = &lift_opts } },
    };
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "len", .desc = .{ .func = 0 }, .sort_idx = .{ .sort = .func, .idx = 0 } },
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

    const ef = inst.getExport("len") orelse return error.TestFailed;
    const local = switch (ef) {
        .local => |l| l,
        else => return error.TestFailed,
    };
    const args = [_]abi.InterfaceValue{.{ .string = .{ .ptr = 0x100, .len = 42 } }};
    var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
    try executor.callComponentFuncByLocal(inst, local, &args, &results, allocator);
    try std.testing.expectEqual(@as(u32, 42), results[0].u32);
}

// #650 commit 2 — tuple<u32,u32> as PARAM (multi-slot compound param)
// on AOT via the unified CallFrame path. Phase B.1 only tested tuple
// as a RESULT (retptr-based). Phase B.2 tested string-as-param (2 i32
// slots) but not a true compound. With the CallFrame refactor the
// AOT path delegates to the same `pushInterfaceValue` /
// `abi.lowerFlatReg` walk that interp uses, so compound aggregate
// params work for free.
//
// Reuses `string_param_core_wasm` (`(i32 i32) -> i32` returning
// `local.get 1`). canon.lift retypes it as
// `func(p: tuple<u32, u32>) -> u32`; the tuple flattens to 2 i32
// slots in order, the function returns the second element, which we
// assert is the tuple's second u32 field.
test "#650 commit 2: AOT lowers tuple<u32,u32> as multi-slot compound param" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &string_param_core_wasm);
    defer allocator.free(cwasm_bytes);

    const core_modules = [_]ctypes.CoreModule{.{ .data = &string_param_core_wasm }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };

    // type 0: tuple<u32,u32>; type 1: (p: tuple<u32,u32>) -> u32
    const tuple_fields = [_]ctypes.ValType{ .u32, .u32 };
    const params = [_]ctypes.NamedValType{.{ .name = "p", .type = .{ .tuple = 0 } }};
    const types = [_]ctypes.TypeDef{
        .{ .tuple = .{ .fields = &tuple_fields } },
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .u32 } } },
    };
    const lift_opts = [_]ctypes.CanonOpt{};
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 1, .opts = &lift_opts } },
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

    const tup_field = [_]abi.InterfaceValue{ .{ .u32 = 100 }, .{ .u32 = 200 } };
    const args = [_]abi.InterfaceValue{.{ .tuple_val = &tup_field }};
    var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
    try executor.callComponentFuncByLocal(inst, local, &args, &results, allocator);
    // core returns local.get 1 (the second flat slot of the tuple).
    try std.testing.expectEqual(@as(u32, 200), results[0].u32);
}

// #650 commit 2 — headline acceptance: `func(list<u8>) -> string`.
//
// Two-stage lowering on the AOT path:
//   1. list<u8> flattens to 2 i32 slots (ptr, len) — fits flat, no spill.
//   2. string flattens to 2 i32 slots — > MAX_FLAT_RESULTS=1 → spilled.
//      AOT uses callee-allocates (canon-ABI v1 lift convention,
//      matching wit-bindgen): core is emitted as
//      `(list_ptr, list_len) -> retptr`. Core allocates the
//      (str_ptr, str_len) buffer via its own realloc and returns
//      the pointer as a single i32 result.
//
// Core wasm:
//   func[0] realloc:  (i32 i32 i32 i32) -> i32      ;; returns 16
//   func[1] cast:     (list_ptr list_len) -> i32    ;; stores list_ptr/len at 16+{0,4}, returns 16
const list_to_string_core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    // type section: 2 types (size = 9 + 6 = 15)
    0x01, 0x0f, 0x02,
    0x60, 0x04, 0x7f, 0x7f, 0x7f, 0x7f, 0x01, 0x7f, // realloc: (i32,i32,i32,i32) -> i32 (9 bytes)
    0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, // cast: (i32,i32) -> i32 (6 bytes)
    // func section: 2 funcs (types 0 and 1)
    0x03, 0x03, 0x02, 0x00, 0x01,
    // memory section: 1 page
    0x05, 0x03, 0x01, 0x00, 0x01,
    // export section: memory, realloc, cast
    // payload = 9 ("memory") + 10 ("realloc") + 7 ("cast") + 1(count) = 27
    0x07, 0x1b, 0x03,
    0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00,
    0x07, 'r', 'e', 'a', 'l', 'l', 'o', 'c', 0x00, 0x00,
    0x04, 'c', 'a', 's', 't', 0x00, 0x01,
    // code section: 2 funcs
    // body0 (realloc): 0x00 0x41 0x10 0x0b = 4 bytes (preceded by size 0x04)
    // body1 (cast, callee-allocates): locals(0) +
    //   i32.const 16; local.get 0; i32.store offset=0
    //   i32.const 16; local.get 1; i32.store offset=4
    //   i32.const 16              ;; return retptr
    //   end
    //   = 1 + (2+2+3) + (2+2+3) + 2 + 1 = 18 bytes (preceded by size 0x12)
    // section payload = count(1) + len-prefix(1)+body0(4) + len-prefix(1)+body1(18) = 25
    0x0a, 0x19, 0x02,
    0x04, 0x00, 0x41, 0x10, 0x0b,
    0x12, 0x00,
    0x41, 0x10, 0x20, 0x00, 0x36, 0x02, 0x00,
    0x41, 0x10, 0x20, 0x01, 0x36, 0x02, 0x04,
    0x41, 0x10,
    0x0b,
};

test "#650 commit 2: AOT lifts func(list<u8>) -> string (acceptance)" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &list_to_string_core_wasm);
    defer allocator.free(cwasm_bytes);

    const core_modules = [_]ctypes.CoreModule{.{ .data = &list_to_string_core_wasm }};
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };

    // type 0: list<u8>; type 1: func(b: list<u8>) -> string
    const params = [_]ctypes.NamedValType{.{ .name = "b", .type = .{ .list = 0 } }};
    const types = [_]ctypes.TypeDef{
        .{ .list = .{ .element = .u8 } },
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .string } } },
    };
    // canon.lift cast (core_func 1) with realloc bound to core_func 0.
    const lift_opts = [_]ctypes.CanonOpt{
        .{ .realloc = 0 },
    };
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 1, .type_idx = 1, .opts = &lift_opts } },
    };
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "cast", .desc = .{ .func = 0 }, .sort_idx = .{ .sort = .func, .idx = 0 } },
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

    const ef = inst.getExport("cast") orelse return error.TestFailed;
    const local = switch (ef) {
        .local => |l| l,
        else => return error.TestFailed,
    };
    const args = [_]abi.InterfaceValue{.{ .list = .{ .ptr = 0x100, .len = 5 } }};
    var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
    try executor.callComponentFuncByLocal(inst, local, &args, &results, allocator);
    defer results[0].deinit(allocator);
    try std.testing.expectEqual(@as(u32, 0x100), results[0].string.ptr);
    try std.testing.expectEqual(@as(u32, 5), results[0].string.len);
}

// ── #687: canon-lower-backed cross-instance bridging ─────────────────────

/// AOT module that imports `env.increment: (i32) -> i32` and exports
/// `go: (i32) -> i32` which simply forwards to the import. Used by the
/// #687 fixture to exercise the canon-lower-backed cross-instance thunk.
const aot_forwarder_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    // type section: 1 type, (i32)->i32
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    // import section: "env"."increment" (func type 0)
    0x02, 0x11, 0x01,
    0x03, 'e', 'n', 'v',
    0x09, 'i', 'n', 'c', 'r', 'e', 'm', 'e', 'n', 't',
    0x00, 0x00,
    // function section: 1 func of type 0
    0x03, 0x02, 0x01, 0x00,
    // memory section: 1 memory min 1
    0x05, 0x03, 0x01, 0x00, 0x01,
    // export section: "memory" (mem 0) + "go" (func 1)
    0x07, 0x0f, 0x02,
    0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00,
    0x02, 'g', 'o', 0x00, 0x01,
    // code section: 1 func body — local.get 0; call 0; end (6 body bytes)
    0x0a, 0x08, 0x01,
    0x06,
    0x00,
    0x20, 0x00,
    0x10, 0x00,
    0x0b,
};

const Incr687Ctx = struct {
    observed: i32 = 0,
    return_value: i32 = 0,

    fn call(
        ctx: ?*anyopaque,
        _: *instance.ComponentInstance,
        args: []const abi.InterfaceValue,
        results: []abi.InterfaceValue,
        _: std.mem.Allocator,
    ) anyerror!void {
        const self: *Incr687Ctx = @ptrCast(@alignCast(ctx.?));
        self.observed = args[0].s32;
        self.return_value = args[0].s32 + 1;
        results[0] = .{ .s32 = self.return_value };
    }
};

test "#687: AOT core imports a canon-lower-bridged sibling export" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;
    // Cross-instance AOT bridging needs `TrampolinePool` (RWX page
    // allocator). Skip on platforms where the pool ctor is comptime-
    // disabled (`runtime/aot/host_trampolines.zig:144-151`): Windows
    // and macOS aarch64. On those targets the runtime falls back to
    // the trap-on-call stub, which the fixture intentionally exercises.
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    if (comptime builtin.os.tag == .macos and builtin.cpu.arch == .aarch64) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &aot_forwarder_wasm);
    defer allocator.free(cwasm_bytes);

    // Component shape:
    //   import "incr": (i32)->i32   = comp-func 0
    //   core_inst 0 = (exports "increment" -> core-func 0 (the lower))
    //   core_inst 1 = (instantiate $aot (with "env" $core_inst_0))
    //   alias core-func ("go" of core_inst 1)            = core-func 1
    //   canons[0] = lower(comp-func 0)                    -> core-func 0
    //   canons[1] = lift(core-func 1, type 0)             = comp-func 1
    //   export "go" = sort_idx{ .func, 1 }
    //
    // Without #687, the AOT import "env.increment" cannot be resolved
    // (its sibling source is a canon.lower, not an alias-of-AOT-export),
    // so the core call traps. With the fix, the import is bridged
    // through `dispatchAotComponentTrampoline` to the bound HostFunc.

    const core_modules = [_]ctypes.CoreModule{.{ .data = &aot_forwarder_wasm }};

    const inst0_exports = [_]ctypes.CoreInlineExport{
        .{ .name = "increment", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const inst1_args = [_]ctypes.CoreInstantiateArg{
        .{ .name = "env", .instance_idx = 0 },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .exports = &inst0_exports },
        .{ .instantiate = .{ .module_idx = 0, .args = &inst1_args } },
    };

    const aliases = [_]ctypes.Alias{
        .{ .instance_export = .{ .sort = .{ .core = .func }, .instance_idx = 1, .name = "go" } },
    };

    const params = [_]ctypes.NamedValType{.{ .name = "x", .type = .s32 }};
    const types = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };

    const empty_opts = [_]ctypes.CanonOpt{};
    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &empty_opts } },
        .{ .lift = .{ .core_func_idx = 1, .type_idx = 0, .opts = &empty_opts } },
    };

    const imports = [_]ctypes.ImportDecl{
        .{ .name = "incr", .desc = .{ .func = 0 } },
    };
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "go", .desc = .{ .func = 0 }, .sort_idx = .{ .sort = .func, .idx = 1 } },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases,
        .types = &types,
        .canons = &canons,
        .imports = &imports,
        .exports = &exports,
    };

    const pcs = [_]instance.PrecompiledCore{
        .{ .module_idx = 0, .cwasm_bytes = cwasm_bytes },
    };
    const inst = try instance.instantiateWithOptions(&component, allocator, .{
        .precompiled_cores = &pcs,
    });
    defer inst.deinit();

    try std.testing.expect(inst.core_instances[1].aot_inst != null);

    // Bind the parent "incr" import to a host function that increments
    // its argument. linkImports walks `inst.trampoline_ctxs` and rebinds
    // the canon-lower ctx that #687 registered there.
    var host_ctx = Incr687Ctx{};
    var providers: std.StringHashMapUnmanaged(instance.ImportBinding) = .{};
    defer providers.deinit(allocator);
    try providers.put(allocator, "incr", .{
        .host_func = .{ .context = &host_ctx, .call = Incr687Ctx.call },
    });
    try inst.linkImports(providers);

    const ef = inst.getExport("go") orelse return error.TestFailed;
    const local = switch (ef) {
        .local => |l| l,
        else => return error.TestFailed,
    };

    const args = [_]abi.InterfaceValue{.{ .s32 = 41 }};
    var results: [1]abi.InterfaceValue = .{.{ .s32 = 0 }};
    try executor.callComponentFuncByLocal(inst, local, &args, &results, allocator);
    try std.testing.expectEqual(@as(i32, 41), host_ctx.observed);
    try std.testing.expectEqual(@as(i32, 42), results[0].s32);
}
