//! C-compatible API for WAMR embedding.
//!
//! This module exports functions with C calling convention and naming
//! matching the original wasm_export.h API, allowing existing C embedders
//! to link against the Zig-built library unchanged.

const std = @import("std");
const config = @import("../config.zig");

/// Runtime initialization arguments
pub const RuntimeInitArgs = extern struct {
    mem_alloc_type: MemAllocType = .allocator_system,
    mem_alloc_option: MemAllocOption = .{},
    // TODO: expand as migration progresses
};

pub const MemAllocType = enum(c_int) {
    allocator_system = 0,
    allocator_pool = 1,
    allocator_user = 2,
};

pub const MemAllocOption = extern struct {
    reserved: u64 = 0,
    // TODO: union of pool/user allocator options
};

/// Module handle (opaque pointer for C callers)
pub const WasmModule = opaque {};

/// Module instance handle
pub const WasmModuleInstance = opaque {};

/// Execution environment handle
pub const WasmExecEnv = opaque {};

/// Wasm value
pub const WasmVal = extern struct {
    kind: ValKind,
    of: extern union {
        i32: i32,
        i64: i64,
        f32: f32,
        f64: f64,
    },
};

pub const ValKind = enum(u8) {
    i32 = 0x7F,
    i64 = 0x7E,
    f32 = 0x7D,
    f64 = 0x7C,
    v128 = 0x7B,
    funcref = 0x70,
    externref = 0x6F,
};

// ---------------------------------------------------------------------------
// Exported C API functions (stubs for Phase 0)
// ---------------------------------------------------------------------------

/// Initialize the WAMR runtime
export fn wasm_runtime_init() bool {
    // TODO: implement in Phase 1
    return true;
}

/// Destroy the WAMR runtime
export fn wasm_runtime_destroy() void {
    // TODO: implement in Phase 1
}

/// Get the version string
export fn wasm_runtime_get_version() [*:0]const u8 {
    return "0.1.0-zig";
}

test "c_api: init and destroy" {
    const ok = wasm_runtime_init();
    try std.testing.expect(ok);
    wasm_runtime_destroy();
}

test "c_api: version string" {
    const ver = wasm_runtime_get_version();
    const slice = std.mem.span(ver);
    try std.testing.expectEqualStrings("0.1.0-zig", slice);
}

// ---------------------------------------------------------------------------
// Extended C API stubs (to be connected in a future phase)
// ---------------------------------------------------------------------------

/// Load a module from binary data.
export fn wasm_runtime_load(buf: [*]const u8, size: u32, error_buf: [*]u8, error_buf_size: u32) ?*WasmModule {
    // TODO: implement using loader
    _ = .{ buf, size, error_buf, error_buf_size };
    return null;
}

/// Instantiate a loaded module.
export fn wasm_runtime_instantiate(module: *WasmModule, stack_size: u32, heap_size: u32, error_buf: [*]u8, error_buf_size: u32) ?*WasmModuleInstance {
    _ = .{ module, stack_size, heap_size, error_buf, error_buf_size };
    return null;
}

/// Deinstantiate a module instance.
export fn wasm_runtime_deinstantiate(inst: *WasmModuleInstance) void {
    _ = inst;
}

/// Unload a module.
export fn wasm_runtime_unload(module: *WasmModule) void {
    _ = module;
}

/// Look up an exported function.
export fn wasm_runtime_lookup_function(inst: *WasmModuleInstance, name: [*:0]const u8) ?*anyopaque {
    _ = .{ inst, name };
    return null;
}

test "c_api: load stub returns null" {
    var err_buf: [64]u8 = undefined;
    const result = wasm_runtime_load("", 0, &err_buf, err_buf.len);
    try std.testing.expect(result == null);
}

test "c_api: instantiate stub returns null" {
    var err_buf: [64]u8 = undefined;
    const result = wasm_runtime_instantiate(undefined, 0, 0, &err_buf, err_buf.len);
    try std.testing.expect(result == null);
}

test "c_api: lookup function stub returns null" {
    const result = wasm_runtime_lookup_function(undefined, "test");
    try std.testing.expect(result == null);
}
