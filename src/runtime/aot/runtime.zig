//! AOT module instantiation and execution.
//!
//! Creates a runnable AotInstance from an AotModule by allocating memories,
//! tables, and globals according to the module specifications, and optionally
//! mapping the compiled native code as executable.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../common/types.zig");
const aot_loader = @import("loader.zig");
const host_bridge = @import("host_bridge.zig");
const platform = @import("../../platform/platform.zig");

/// Compact context passed to AOT-compiled functions as a hidden first parameter.
/// Laid out as a flat struct so compiled code can load fields at known offsets.
pub const VmCtx = extern struct {
    /// Base pointer to linear memory (memory 0).
    memory_base: usize = 0,
    /// Size of linear memory in bytes (current, may grow).
    memory_size: usize = 0,
    /// Pointer to flat globals values array (each global is an i64 at index * 8).
    globals_ptr: usize = 0,
    /// Pointer to array of host function pointers (one per import).
    host_functions_ptr: usize = 0,
    /// Maximum allocated memory region size in bytes (for grow bounds checking).
    memory_max_size: usize = 0,
    /// Number of globals.
    globals_count: u32 = 0,
    /// Number of host functions.
    host_functions_count: u32 = 0,
    /// Current memory size in pages (for memory.size instruction).
    memory_pages: u32 = 0,
};

// ─── Comptime target validation ─────────────────────────────────────────────

/// The native machine architecture, resolved at comptime.
const native_arch: enum { x86_64, aarch64, unsupported } = switch (builtin.cpu.arch) {
    .x86_64 => .x86_64,
    .aarch64 => .aarch64,
    else => .unsupported,
};

/// Whether the current target can execute AOT code.
const can_execute_native = native_arch != .unsupported;

// ─── Instance ───────────────────────────────────────────────────────────────

pub const AotInstance = struct {
    module: *const aot_loader.AotModule,
    memories: []*types.MemoryInstance,
    tables: []*types.TableInstance,
    globals: []*types.GlobalInstance,
    allocator: std.mem.Allocator,
    /// Base address of the mapped executable code (null if not yet mapped).
    code_base: ?[*]const u8 = null,
    /// Size of the mapped executable region (for cleanup).
    code_size: usize = 0,
    /// Resolved AOT host function pointers (one per import).
    host_functions: []const ?*const anyopaque = &.{},
};

// ─── Errors ─────────────────────────────────────────────────────────────────

pub const RuntimeError = error{
    OutOfMemory,
    CodeMappingFailed,
    FunctionNotFound,
    ExecutionFailed,
    TableAllocationFailed,
};

// ─── Public API ─────────────────────────────────────────────────────────────

/// Instantiate an AOT module, producing a runnable AotInstance.
pub fn instantiate(module: *const aot_loader.AotModule, allocator: std.mem.Allocator) RuntimeError!*AotInstance {
    var inst = allocator.create(AotInstance) catch return error.OutOfMemory;
    errdefer allocator.destroy(inst);

    inst.* = .{
        .module = module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
        .allocator = allocator,
    };

    inst.memories = try allocateMemories(module, allocator);
    errdefer freeMemories(inst.memories, allocator);

    // Apply data segments to linear memory
    for (module.data_segments) |seg| {
        if (seg.memory_idx >= inst.memories.len) continue;
        const mem = inst.memories[seg.memory_idx];
        const end = @as(usize, seg.offset) + seg.data.len;
        if (end > mem.data.len) continue;
        @memcpy(mem.data[seg.offset..][0..seg.data.len], seg.data);
    }

    inst.tables = try allocateTables(module, allocator);
    errdefer freeTables(inst.tables, allocator);

    inst.globals = try allocateGlobals(module, allocator);
    errdefer freeGlobals(inst.globals, allocator);

    // Resolve AOT host functions for imports
    inst.host_functions = try resolveHostFunctions(module, allocator);

    return inst;
}

/// Destroy an AOT instance, freeing all allocated resources.
pub fn destroy(inst: *AotInstance) void {
    const allocator = inst.allocator;
    // Unmap executable code if mapped.
    if (inst.code_base) |base| {
        platform.munmap(@constCast(@ptrCast(base)), inst.code_size);
    }
    freeMemories(inst.memories, allocator);
    freeTables(inst.tables, allocator);
    freeGlobals(inst.globals, allocator);
    if (inst.host_functions.len > 0) allocator.free(inst.host_functions);
    allocator.destroy(inst);
}

/// Look up an exported function by name, returning its function index.
pub fn findExportFunc(inst: *const AotInstance, name: []const u8) ?u32 {
    for (inst.module.exports) |exp| {
        if (exp.kind == .function and std.mem.eql(u8, exp.name, name)) return exp.index;
    }
    return null;
}

/// Get the native code pointer for a function by module-level index.
/// Import functions (func_idx < import_function_count) have no native code
/// and return null.  Local functions are looked up in func_offsets after
/// subtracting the import count.
pub fn getFuncAddr(inst: *const AotInstance, func_idx: u32) ?[*]const u8 {
    const module = inst.module;
    const import_count = module.import_function_count;

    // Import functions don't have native code
    if (func_idx < import_count) return null;

    const local_idx = func_idx - import_count;
    if (local_idx >= module.func_count) return null;
    const offset = module.func_offsets[local_idx];
    // Prefer the executable mapping.
    if (inst.code_base) |base| {
        if (offset >= inst.code_size) return null;
        return base + offset;
    }
    // Fall back to raw text section (read-only, not executable).
    const text = module.text_section orelse return null;
    if (offset >= text.len) return null;
    return text.ptr + offset;
}

// ─── Native execution ───────────────────────────────────────────────────────

/// Map the module's native code into executable memory.
/// After this call, `getFuncAddr` returns pointers suitable for execution.
pub fn mapCodeExecutable(inst: *AotInstance) RuntimeError!void {
    const text = inst.module.text_section orelse return;
    if (text.len == 0) return;

    // 1. Allocate RW pages
    const mem = platform.mmap(null, text.len, .{ .read = true, .write = true }, .{}) orelse
        return error.CodeMappingFailed;

    // 2. Copy native code
    @memcpy(mem[0..text.len], text);

    // 3. Flush instruction cache (required on AArch64, no-op on x86-64)
    if (comptime native_arch == .aarch64) {
        platform.icacheFlush(mem, text.len);
    }

    // 4. Transition to RX (W^X)
    platform.mprotect(mem, text.len, .{ .read = true, .exec = true }) catch
        return error.CodeMappingFailed;

    inst.code_base = mem;
    inst.code_size = text.len;
}

/// Call an AOT-compiled function by index.
/// The code must have been mapped via `mapCodeExecutable` first.
///
/// Uses comptime to select the correct function pointer type based on `Result`.
pub fn callFunc(inst: *AotInstance, func_idx: u32, comptime Result: type) RuntimeError!Result {
    comptime if (!can_execute_native) @compileError("AOT execution not supported on this architecture");

    if (inst.code_base == null) return error.CodeMappingFailed;
    const addr = getFuncAddr(inst, func_idx) orelse return error.FunctionNotFound;

    // Build flat globals array for AOT access
    var globals_buf: [256]i64 = std.mem.zeroes([256]i64);
    const n_globals = @min(inst.globals.len, globals_buf.len);
    for (0..n_globals) |i| {
        globals_buf[i] = inst.globals[i].value.i64;
    }

    // Build VMContext for the compiled function
    var vmctx = VmCtx{};
    if (inst.memories.len > 0) {
        vmctx.memory_base = @intFromPtr(inst.memories[0].data.ptr);
        vmctx.memory_size = @as(usize, inst.memories[0].current_pages) * types.MemoryInstance.page_size;
        vmctx.memory_max_size = inst.memories[0].data.len;
        vmctx.memory_pages = inst.memories[0].current_pages;
    }
    // Always provide a valid globals pointer — compiled code may access globals
    // even if none are explicitly initialized (they default to zero).
    vmctx.globals_ptr = @intFromPtr(&globals_buf);
    vmctx.globals_count = @intCast(n_globals);
    if (inst.host_functions.len > 0) {
        vmctx.host_functions_ptr = @intFromPtr(inst.host_functions.ptr);
        vmctx.host_functions_count = @intCast(inst.host_functions.len);
    }

    // AOT-compiled functions receive a VmCtx pointer as hidden first parameter.
    const FnPtr = *const fn (*VmCtx) callconv(.c) Result;
    const func_ptr: FnPtr = @ptrCast(@alignCast(addr));
    const result = func_ptr(&vmctx);

    // Sync globals back from flat array to GlobalInstance objects
    for (0..n_globals) |i| {
        inst.globals[i].value = .{ .i64 = globals_buf[i] };
    }

    return result;
}

// ─── Host function resolution ───────────────────────────────────────────────

/// Resolve AOT host function adapters for each import in the module.
/// Returns a slice of optional function pointers indexed by import index.
fn resolveHostFunctions(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
) RuntimeError![]const ?*const anyopaque {
    if (module.import_function_count == 0) return &.{};

    const host_fns = allocator.alloc(?*const anyopaque, module.import_function_count) catch
        return error.OutOfMemory;
    @memset(host_fns, null);

    var func_idx: u32 = 0;
    for (module.imports) |imp| {
        if (imp.kind == .function) {
            if (func_idx < module.import_function_count and
                host_bridge.isWasiModule(imp.module_name))
            {
                host_fns[func_idx] = host_bridge.resolveAotHostFunction(imp.field_name);
            }
            func_idx += 1;
        }
    }

    return host_fns;
}

// ─── Allocation helpers ─────────────────────────────────────────────────────

fn allocateMemories(module: *const aot_loader.AotModule, allocator: std.mem.Allocator) RuntimeError![]*types.MemoryInstance {
    if (module.memories.len == 0) return &.{};

    const memories = allocator.alloc(*types.MemoryInstance, module.memories.len) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| memories[i].release(allocator);
        allocator.free(memories);
    }

    for (module.memories, 0..) |mem_type, i| {
        const initial_pages: u32 = @intCast(@min(mem_type.limits.min, 65536));
        const max_pages: u32 = @intCast(@min(mem_type.limits.max orelse 65536, 65536));
        // Pre-allocate for memory.grow: use max_pages but cap at a reasonable default
        const alloc_pages = @min(max_pages, @max(initial_pages, 256));
        const size = @as(usize, alloc_pages) * types.MemoryInstance.page_size;

        const data = allocator.alloc(u8, size) catch return error.OutOfMemory;
        @memset(data, 0);
        const mem = allocator.create(types.MemoryInstance) catch {
            allocator.free(data);
            return error.OutOfMemory;
        };
        mem.* = .{
            .memory_type = mem_type,
            .data = data,
            .current_pages = initial_pages,
            .max_pages = max_pages,
        };
        memories[i] = mem;
        initialized += 1;
    }

    return memories;
}

fn allocateTables(module: *const aot_loader.AotModule, allocator: std.mem.Allocator) RuntimeError![]*types.TableInstance {
    if (module.tables.len == 0) return &.{};

    const tables = allocator.alloc(*types.TableInstance, module.tables.len) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| tables[i].release(allocator);
        allocator.free(tables);
    }

    for (module.tables, 0..) |table_type, i| {
        const elements = allocator.alloc(types.TableElement, table_type.limits.min) catch return error.TableAllocationFailed;
        for (elements) |*e| e.* = types.TableElement.nullForType(table_type.elem_type);
        const tbl = allocator.create(types.TableInstance) catch {
            allocator.free(elements);
            return error.TableAllocationFailed;
        };
        tbl.* = .{ .table_type = table_type, .elements = elements };
        tables[i] = tbl;
        initialized += 1;
    }

    return tables;
}

fn allocateGlobals(module: *const aot_loader.AotModule, allocator: std.mem.Allocator) RuntimeError![]*types.GlobalInstance {
    if (module.global_inits.len == 0) return &.{};

    const globals = allocator.alloc(*types.GlobalInstance, module.global_inits.len) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| allocator.destroy(globals[i]);
        allocator.free(globals);
    }

    for (module.global_inits, 0..) |ginit, i| {
        const g = allocator.create(types.GlobalInstance) catch return error.OutOfMemory;
        g.* = .{
            .global_type = .{
                .val_type = @enumFromInt(ginit.val_type),
                .mutability = if (ginit.mutability != 0) .mutable else .immutable,
            },
            .value = .{ .i64 = ginit.init_i64 },
        };
        globals[i] = g;
        initialized += 1;
    }

    return globals;
}

fn freeMemories(memories: []*types.MemoryInstance, allocator: std.mem.Allocator) void {
    for (memories) |m| m.release(allocator);
    if (memories.len > 0) allocator.free(memories);
}

fn freeTables(tables: []*types.TableInstance, allocator: std.mem.Allocator) void {
    for (tables) |t| t.release(allocator);
    if (tables.len > 0) allocator.free(tables);
}

fn freeGlobals(globals: []*types.GlobalInstance, allocator: std.mem.Allocator) void {
    for (globals) |g| allocator.destroy(g);
    if (globals.len > 0) allocator.free(globals);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

test "instantiate: empty module" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 0), inst.memories.len);
    try std.testing.expectEqual(@as(usize, 0), inst.tables.len);
    try std.testing.expectEqual(@as(usize, 0), inst.globals.len);
    try std.testing.expectEqual(@as(?[*]const u8, null), inst.code_base);
}

test "findExportFunc: returns null for missing export" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(?u32, null), findExportFunc(inst, "nonexistent"));
}

test "findExportFunc: finds exported function" {
    const exports = [_]types.ExportDesc{
        .{ .name = "memory", .kind = .memory, .index = 0 },
        .{ .name = "_start", .kind = .function, .index = 3 },
    };
    const module = aot_loader.AotModule{ .exports = &exports };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(?u32, 3), findExportFunc(inst, "_start"));
    try std.testing.expectEqual(@as(?u32, null), findExportFunc(inst, "missing"));
}

test "destroy: cleans up without leaks" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    destroy(inst);
    // If the testing allocator doesn't report leaks, we're good.
}

test "instantiate: module with memory" {
    const mem_types = [_]types.MemoryType{
        .{ .limits = .{ .min = 1, .max = 4 } },
    };
    const module = aot_loader.AotModule{ .memories = &mem_types };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 1), inst.memories.len);
    try std.testing.expectEqual(@as(u32, 1), inst.memories[0].current_pages);
    // Pre-allocated to max_pages (4) for memory.grow support
    try std.testing.expectEqual(@as(usize, 4 * 65536), inst.memories[0].data.len);
}

test "instantiate: module with table" {
    const tbl_types = [_]types.TableType{
        .{ .elem_type = .funcref, .limits = .{ .min = 10, .max = 100 } },
    };
    const module = aot_loader.AotModule{ .tables = &tbl_types };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 1), inst.tables.len);
    try std.testing.expectEqual(@as(usize, 10), inst.tables[0].elements.len);
    // All elements should be null-initialized
    for (inst.tables[0].elements) |elem| {
        try std.testing.expect(elem.isNull());
    }
}

test "getFuncAddr: returns null without text section" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(?[*]const u8, null), getFuncAddr(inst, 0));
}

test "getFuncAddr: returns correct address" {
    const text = [_]u8{ 0xCC, 0x90, 0xC3, 0x55, 0x48, 0x89, 0xE5, 0xC3 };
    const offsets = [_]u32{ 0, 3 };
    const module = aot_loader.AotModule{
        .text_section = &text,
        .func_offsets = &offsets,
        .func_count = 2,
    };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    const addr0 = getFuncAddr(inst, 0);
    try std.testing.expect(addr0 != null);
    try std.testing.expectEqual(@as(u8, 0xCC), addr0.?[0]);

    const addr1 = getFuncAddr(inst, 1);
    try std.testing.expect(addr1 != null);
    try std.testing.expectEqual(@as(u8, 0x55), addr1.?[0]);

    // Out of range
    try std.testing.expectEqual(@as(?[*]const u8, null), getFuncAddr(inst, 99));
}

test "VmCtx layout: fields at expected offsets" {
    // Verify extern struct field offsets match what the codegen expects
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(VmCtx, "memory_base"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(VmCtx, "memory_size"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(VmCtx, "globals_ptr"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(VmCtx, "host_functions_ptr"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(VmCtx, "memory_max_size"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(VmCtx, "globals_count"));
    try std.testing.expectEqual(@as(usize, 44), @offsetOf(VmCtx, "host_functions_count"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(VmCtx, "memory_pages"));
}

test "getFuncAddr: import indices return null" {
    const text = [_]u8{ 0xCC, 0x90, 0xC3, 0x55, 0x48, 0x89, 0xE5, 0xC3 };
    const offsets = [_]u32{ 0, 3 };
    const module = aot_loader.AotModule{
        .text_section = &text,
        .func_offsets = &offsets,
        .func_count = 2,
        .import_function_count = 2,
    };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    // Import indices (0, 1) should return null (no native code)
    try std.testing.expectEqual(@as(?[*]const u8, null), getFuncAddr(inst, 0));
    try std.testing.expectEqual(@as(?[*]const u8, null), getFuncAddr(inst, 1));

    // Local indices (2, 3) map to func_offsets[0], func_offsets[1]
    const addr2 = getFuncAddr(inst, 2);
    try std.testing.expect(addr2 != null);
    try std.testing.expectEqual(@as(u8, 0xCC), addr2.?[0]);

    const addr3 = getFuncAddr(inst, 3);
    try std.testing.expect(addr3 != null);
    try std.testing.expectEqual(@as(u8, 0x55), addr3.?[0]);
}

test "resolveHostFunctions: resolves WASI imports" {
    const imports = [_]aot_loader.AotImportDesc{
        .{ .module_name = "wasi_snapshot_preview1", .field_name = "fd_write", .kind = .function, .func_type_idx = 0 },
        .{ .module_name = "wasi_snapshot_preview1", .field_name = "clock_time_get", .kind = .function, .func_type_idx = 1 },
        .{ .module_name = "env", .field_name = "some_func", .kind = .function, .func_type_idx = 2 },
    };
    const module = aot_loader.AotModule{
        .import_function_count = 3,
        .imports = &imports,
    };
    const result = try resolveHostFunctions(&module, std.testing.allocator);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expect(result[0] != null); // fd_write resolved
    try std.testing.expect(result[1] != null); // clock_time_get resolved
    try std.testing.expect(result[2] == null); // env.some_func not resolved
}

test "instantiate: module with WASI imports resolves host functions" {
    const imports = [_]aot_loader.AotImportDesc{
        .{ .module_name = "wasi_snapshot_preview1", .field_name = "fd_write", .kind = .function, .func_type_idx = 0 },
    };
    const module = aot_loader.AotModule{
        .import_function_count = 1,
        .imports = &imports,
    };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 1), inst.host_functions.len);
    try std.testing.expect(inst.host_functions[0] != null);
}
