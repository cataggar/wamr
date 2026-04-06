//! AOT module instantiation and execution.
//!
//! Creates a runnable AotInstance from an AotModule by allocating memories,
//! tables, and globals according to the module specifications, and optionally
//! mapping the compiled native code as executable.

const std = @import("std");
const types = @import("../common/types.zig");
const aot_loader = @import("loader.zig");

// ─── Instance ───────────────────────────────────────────────────────────────

pub const AotInstance = struct {
    module: *const aot_loader.AotModule,
    memories: []types.MemoryInstance,
    tables: []types.TableInstance,
    globals: []types.GlobalInstance,
    allocator: std.mem.Allocator,
    /// Base address of the mapped executable code (null if not yet mapped).
    code_base: ?[*]const u8 = null,
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

    inst.tables = try allocateTables(module, allocator);
    errdefer freeTables(inst.tables, allocator);

    inst.globals = try allocateGlobals(module, allocator);
    errdefer freeGlobals(inst.globals, allocator);

    return inst;
}

/// Destroy an AOT instance, freeing all allocated resources.
pub fn destroy(inst: *AotInstance) void {
    const allocator = inst.allocator;
    freeMemories(inst.memories, allocator);
    freeTables(inst.tables, allocator);
    freeGlobals(inst.globals, allocator);
    allocator.destroy(inst);
}

/// Look up an exported function by name, returning its function index.
pub fn findExportFunc(inst: *const AotInstance, name: []const u8) ?u32 {
    for (inst.module.exports) |exp| {
        if (exp.kind == .function and std.mem.eql(u8, exp.name, name)) return exp.index;
    }
    return null;
}

/// Get the native code pointer for a function by index.
/// Returns the text_section base offset by the function's offset.
pub fn getFuncAddr(inst: *const AotInstance, func_idx: u32) ?[*]const u8 {
    const module = inst.module;
    const text = module.text_section orelse return null;
    if (func_idx >= module.func_count) return null;
    const offset = module.func_offsets[func_idx];
    if (offset >= text.len) return null;
    return text.ptr + offset;
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
        const initial_pages = mem_type.limits.min;
        const max_pages = mem_type.limits.max orelse 65536;
        const size = @as(usize, initial_pages) * types.MemoryInstance.page_size;

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
        const elements = allocator.alloc(?u32, table_type.limits.min) catch return error.TableAllocationFailed;
        @memset(elements, null);
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
    // AOT modules don't carry global init exprs in this loader yet;
    // return empty for now. Full init_data parsing will populate these.
    _ = module;
    _ = allocator;
    return &.{};
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
    try std.testing.expectEqual(@as(usize, 65536), inst.memories[0].data.len);
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
        try std.testing.expectEqual(@as(?u32, null), elem);
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
