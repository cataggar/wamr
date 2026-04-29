//! C-compatible API for WAMR embedding.
//!
//! This module exports functions with C calling convention and naming
//! matching the original wasm_export.h API, allowing existing C embedders
//! to link against the Zig-built library unchanged.

const std = @import("std");
const config = @import("../config.zig");
const loader = @import("../runtime/interpreter/loader.zig");
const instance_mod = @import("../runtime/interpreter/instance.zig");
const types = @import("../runtime/common/types.zig");

const backing_allocator = std.heap.page_allocator;
const version_cstr = config.version ++ "\x00";

/// Internal wrapper that pairs a WasmModule with its arena.
const ModuleWrapper = struct {
    module: types.WasmModule,
    arena: std.heap.ArenaAllocator,
};

/// Internal wrapper for an instance + back-pointer to its module wrapper.
const InstanceWrapper = struct {
    instance: *types.ModuleInstance,
    module_wrapper: *ModuleWrapper,
};

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
// Exported C API functions
// ---------------------------------------------------------------------------

/// Initialize the WAMR runtime.
export fn wasm_runtime_init() bool {
    return true;
}

/// Destroy the WAMR runtime.
export fn wasm_runtime_destroy() void {}

/// Get the version string.
export fn wasm_runtime_get_version() [*:0]const u8 {
    return version_cstr;
}

/// Load a module from binary data.
export fn wasm_runtime_load(buf: [*]const u8, size: u32, error_buf: [*]u8, error_buf_size: u32) ?*WasmModule {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    const module = loader.load(buf[0..size], arena.allocator()) catch |err| {
        writeError(error_buf, error_buf_size, @errorName(err));
        arena.deinit();
        return null;
    };
    const wrapper = backing_allocator.create(ModuleWrapper) catch {
        writeError(error_buf, error_buf_size, "out of memory");
        arena.deinit();
        return null;
    };
    wrapper.* = .{ .module = module, .arena = arena };
    return @ptrCast(wrapper);
}

/// Unload a module.
export fn wasm_runtime_unload(module_ptr: ?*WasmModule) void {
    const wrapper: *ModuleWrapper = @ptrCast(@alignCast(module_ptr orelse return));
    wrapper.arena.deinit();
    backing_allocator.destroy(wrapper);
}

/// Instantiate a loaded module.
export fn wasm_runtime_instantiate(
    module_ptr: ?*WasmModule,
    stack_size: u32,
    heap_size: u32,
    error_buf: [*]u8,
    error_buf_size: u32,
) ?*WasmModuleInstance {
    _ = .{ stack_size, heap_size };
    const mod_wrapper: *ModuleWrapper = @ptrCast(@alignCast(module_ptr orelse return null));
    const inst = instance_mod.instantiate(&mod_wrapper.module, backing_allocator) catch |err| {
        writeError(error_buf, error_buf_size, @errorName(err));
        return null;
    };
    const wrapper = backing_allocator.create(InstanceWrapper) catch {
        writeError(error_buf, error_buf_size, "out of memory");
        return null;
    };
    wrapper.* = .{ .instance = inst, .module_wrapper = mod_wrapper };
    return @ptrCast(wrapper);
}

/// Deinstantiate a module instance.
export fn wasm_runtime_deinstantiate(inst_ptr: ?*WasmModuleInstance) void {
    const wrapper: *InstanceWrapper = @ptrCast(@alignCast(inst_ptr orelse return));
    instance_mod.destroy(wrapper.instance);
    backing_allocator.destroy(wrapper);
}

/// Look up an exported function. Returns a non-null opaque handle on success.
export fn wasm_runtime_lookup_function(inst_ptr: ?*WasmModuleInstance, name: [*:0]const u8) ?*anyopaque {
    const wrapper: *InstanceWrapper = @ptrCast(@alignCast(inst_ptr orelse return null));
    const name_slice = std.mem.span(name);
    const exp = wrapper.instance.module.findExport(name_slice, .function) orelse return null;
    // Encode function index as a pointer (+1 to avoid null).
    return @ptrFromInt(@as(usize, exp.index) + 1);
}

/// Write a NUL-terminated error message into a caller-provided buffer.
fn writeError(buf: [*]u8, size: u32, msg: []const u8) void {
    if (size == 0) return;
    const len = @min(msg.len, size - 1);
    @memcpy(buf[0..len], msg[0..len]);
    buf[len] = 0;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "c_api: init and destroy" {
    const ok = wasm_runtime_init();
    try std.testing.expect(ok);
    wasm_runtime_destroy();
}

test "c_api: version string" {
    const ver = wasm_runtime_get_version();
    const slice = std.mem.span(ver);
    try std.testing.expectEqualStrings(config.version, slice);
}

/// (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add)
const add_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
    // type section: 1 type — (param i32 i32) (result i32)
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7F, 0x7F, 0x01,
    0x7F,
    // function section: 1 function, type index 0
    0x03, 0x02, 0x01, 0x00,
    // export section: "add" -> func 0
    0x07, 0x07, 0x01,
    0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
    // code section: 1 body
    0x0A, 0x09, 0x01, // section id, size, count
    0x07, 0x00, // body size, local decl count
    0x20, 0x00, // local.get 0
    0x20, 0x01, // local.get 1
    0x6A, // i32.add
    0x0B, // end
};

test "c_api: full lifecycle — init, load, instantiate, lookup, teardown" {
    try std.testing.expect(wasm_runtime_init());

    var err_buf: [128]u8 = undefined;

    const mod = wasm_runtime_load(&add_wasm, add_wasm.len, &err_buf, err_buf.len);
    try std.testing.expect(mod != null);

    const inst = wasm_runtime_instantiate(mod.?, 0, 0, &err_buf, err_buf.len);
    try std.testing.expect(inst != null);

    const func = wasm_runtime_lookup_function(inst.?, "add");
    try std.testing.expect(func != null);

    wasm_runtime_deinstantiate(inst.?);
    wasm_runtime_unload(mod.?);
    wasm_runtime_destroy();
}

test "c_api: load invalid wasm returns null with error message" {
    var err_buf: [128]u8 = [_]u8{0} ** 128;
    const bad_data = [_]u8{ 0xDE, 0xAD };
    const result = wasm_runtime_load(&bad_data, bad_data.len, &err_buf, err_buf.len);
    try std.testing.expect(result == null);
    // Error buffer should contain a non-empty NUL-terminated string.
    const msg = std.mem.sliceTo(&err_buf, 0);
    try std.testing.expect(msg.len > 0);
}

test "c_api: load and unload roundtrip" {
    var err_buf: [128]u8 = undefined;
    const mod = wasm_runtime_load(&add_wasm, add_wasm.len, &err_buf, err_buf.len);
    try std.testing.expect(mod != null);
    wasm_runtime_unload(mod.?);
}

test "c_api: lookup nonexistent function returns null" {
    var err_buf: [128]u8 = undefined;
    const mod = wasm_runtime_load(&add_wasm, add_wasm.len, &err_buf, err_buf.len);
    try std.testing.expect(mod != null);
    const inst = wasm_runtime_instantiate(mod.?, 0, 0, &err_buf, err_buf.len);
    try std.testing.expect(inst != null);

    const func = wasm_runtime_lookup_function(inst.?, "nonexistent");
    try std.testing.expect(func == null);

    wasm_runtime_deinstantiate(inst.?);
    wasm_runtime_unload(mod.?);
}
