//! Module instantiation — creates a runnable ModuleInstance from a WasmModule.
//!
//! Follows the WebAssembly spec §4.5.4: given a validated WasmModule, this
//! allocates memories, tables, and globals, then applies data and element
//! segments to produce a ready-to-execute ModuleInstance.

const std = @import("std");
const types = @import("../common/types.zig");
const leb128_mod = @import("../../shared/utils/leb128.zig");
const ExecEnv = @import("../common/exec_env.zig").ExecEnv;
const interp = @import("interp.zig");

pub const InstantiationError = error{
    OutOfMemory,
    MemoryAllocationFailed,
    TableAllocationFailed,
    InvalidInitExpr,
    DataSegmentOutOfBounds,
    ElemSegmentOutOfBounds,
    ImportResolutionFailed,
    InvalidGlobalIndex,
    UnknownImport,
    StartFunctionFailed,
};

/// Resolved imports passed during instantiation.
/// When a module has imports, the caller must supply this context
/// with enough entries to cover each import category count.
pub const ImportContext = struct {
    globals: []const *types.GlobalInstance = &.{},
    memories: []const *types.MemoryInstance = &.{},
    tables: []const *types.TableInstance = &.{},
    functions: []const types.ImportedFunction = &.{},
};

/// Instantiate a module, producing a runnable ModuleInstance.
/// If the module has imports and no `import_ctx` is provided, returns ImportResolutionFailed.
pub fn instantiate(module: *const types.WasmModule, allocator: std.mem.Allocator) InstantiationError!*types.ModuleInstance {
    return instantiateWithImports(module, allocator, null);
}

/// Instantiate a module with optional pre-resolved imports.
pub fn instantiateWithImports(
    module: *const types.WasmModule,
    allocator: std.mem.Allocator,
    import_ctx: ?ImportContext,
) InstantiationError!*types.ModuleInstance {
    const has_imports = module.import_function_count > 0 or
        module.import_global_count > 0 or
        module.import_memory_count > 0 or
        module.import_table_count > 0;

    if (has_imports and import_ctx == null) {
        return error.ImportResolutionFailed;
    }

    var inst = allocator.create(types.ModuleInstance) catch return error.OutOfMemory;
    errdefer allocator.destroy(inst);

    inst.* = .{
        .module = module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
        .allocator = allocator,
    };

    inst.memories = try allocateMemories(module, allocator, import_ctx);
    errdefer freeMemories(inst.memories, allocator);

    inst.tables = try allocateTables(module, allocator, import_ctx);
    errdefer freeTables(inst.tables, allocator);

    inst.globals = try initializeGlobals(module, allocator, import_ctx);
    errdefer freeGlobals(inst.globals, module.import_global_count, allocator);

    // Store imported function references
    if (import_ctx) |ctx| {
        if (ctx.functions.len > 0) {
            inst.import_functions = allocator.dupe(types.ImportedFunction, ctx.functions) catch return error.OutOfMemory;
        }
    }

    try applyDataSegments(module, inst.memories, inst.globals);
    try applyElemSegments(module, inst.tables, inst, inst.globals);

    // Set source_module for locally-created funcref globals
    for (module.globals, 0..) |global, i| {
        if (global.init_expr == .ref_func) {
            inst.globals[module.import_global_count + i].source_module = inst;
        }
    }

    // Mark active and declarative elem segments as dropped (§4.5.4 step 13)
    if (module.elements.len > 0) {
        inst.dropped_elems = allocator.alloc(bool, module.elements.len) catch return error.OutOfMemory;
        for (module.elements, 0..) |seg, i| {
            inst.dropped_elems[i] = !seg.is_passive; // active and declarative are dropped
        }
    }

    // Execute start function if present (§4.5.4 step 15)
    if (module.start_function) |start_idx| {
        if (start_idx >= module.import_function_count) {
            var env = ExecEnv.create(inst, 4096, allocator) catch return error.StartFunctionFailed;
            defer env.destroy();
            interp.executeFunction(env, start_idx) catch return error.StartFunctionFailed;
        }
    }

    return inst;
}

/// Destroy a module instance, freeing all allocated resources.
pub fn destroy(inst: *types.ModuleInstance) void {
    const allocator = inst.allocator;
    freeMemories(inst.memories, allocator);
    freeTables(inst.tables, allocator);
    freeGlobals(inst.globals, inst.module.import_global_count, allocator);
    if (inst.import_functions.len > 0) allocator.free(inst.import_functions);
    if (inst.dropped_elems.len > 0) allocator.free(inst.dropped_elems);
    allocator.destroy(inst);
}

// ─── Allocation helpers ─────────────────────────────────────────────────────

fn allocateMemories(module: *const types.WasmModule, allocator: std.mem.Allocator, import_ctx: ?ImportContext) InstantiationError![]*types.MemoryInstance {
    const import_count = module.import_memory_count;
    const total_count = import_count + @as(u32, @intCast(module.memories.len));
    if (total_count == 0) return &.{};

    const mems = allocator.alloc(*types.MemoryInstance, total_count) catch
        return error.MemoryAllocationFailed;
    errdefer allocator.free(mems);

    var local_init: usize = 0;
    errdefer for (0..local_init) |i| mems[import_count + i].release(allocator);

    // Imported memories: share pointer and bump refcount
    if (import_count > 0) {
        if (import_ctx) |ctx| {
            if (ctx.memories.len >= import_count) {
                for (0..import_count) |i| {
                    const m = ctx.memories[i];
                    @constCast(m).retain();
                    mems[i] = @constCast(m);
                }
            }
        }
    }

    // Heap-allocate local memories
    for (module.memories, 0..) |mem_type, i| {
        const min_pages = mem_type.limits.min;
        const max_pages = mem_type.limits.max orelse 65536;
        const initial_size = @as(usize, min_pages) * types.MemoryInstance.page_size;

        const data = allocator.alloc(u8, initial_size) catch
            return error.MemoryAllocationFailed;
        @memset(data, 0);

        const mem = allocator.create(types.MemoryInstance) catch {
            allocator.free(data);
            return error.MemoryAllocationFailed;
        };
        mem.* = .{
            .memory_type = mem_type,
            .data = data,
            .current_pages = min_pages,
            .max_pages = max_pages,
        };
        mems[import_count + i] = mem;
        local_init += 1;
    }

    return mems;
}

fn allocateTables(module: *const types.WasmModule, allocator: std.mem.Allocator, import_ctx: ?ImportContext) InstantiationError![]*types.TableInstance {
    const import_count = module.import_table_count;
    const total_count = import_count + @as(u32, @intCast(module.tables.len));
    if (total_count == 0) return &.{};

    const tables = allocator.alloc(*types.TableInstance, total_count) catch
        return error.TableAllocationFailed;
    errdefer allocator.free(tables);

    var local_init: usize = 0;
    errdefer for (0..local_init) |i| tables[import_count + i].release(allocator);

    // Imported tables: share pointer and bump refcount
    if (import_count > 0) {
        if (import_ctx) |ctx| {
            if (ctx.tables.len >= import_count) {
                for (0..import_count) |i| {
                    const t = ctx.tables[i];
                    @constCast(t).retain();
                    tables[i] = @constCast(t);
                }
            }
        }
    }

    // Heap-allocate local tables
    for (module.tables, 0..) |table_type, i| {
        const min_elems = table_type.limits.min;
        const elems = allocator.alloc(?types.FuncRef, min_elems) catch
            return error.TableAllocationFailed;
        @memset(elems, null);

        const tbl = allocator.create(types.TableInstance) catch {
            allocator.free(elems);
            return error.TableAllocationFailed;
        };
        tbl.* = .{ .table_type = table_type, .elements = elems };
        tables[import_count + i] = tbl;
        local_init += 1;
    }

    return tables;
}

fn initializeGlobals(module: *const types.WasmModule, allocator: std.mem.Allocator, import_ctx: ?ImportContext) InstantiationError![]*types.GlobalInstance {
    const import_count = module.import_global_count;
    const total_count = import_count + @as(u32, @intCast(module.globals.len));
    if (total_count == 0) return &.{};

    const globals = allocator.alloc(*types.GlobalInstance, total_count) catch
        return error.OutOfMemory;

    // Copy imported globals
    if (import_count > 0) {
        if (import_ctx) |ctx| {
            const count = @min(import_count, @as(u32, @intCast(ctx.globals.len)));
            for (0..count) |i| {
                globals[i] = ctx.globals[i];
            }
        }
    }

    // Initialize local globals (can reference imported globals via global.get)
    for (module.globals, 0..) |global, i| {
        const g = allocator.create(types.GlobalInstance) catch return error.OutOfMemory;
        g.* = .{
            .global_type = global.global_type,
            .value = try evalInitExpr(global.init_expr, globals[0 .. import_count + i]),
        };
        // For ref_func globals, source_module will be set after inst is fully created
        // For global.get, inherit source_module from the referenced global
        if (global.init_expr == .global_get) {
            const src_idx = global.init_expr.global_get;
            if (src_idx < import_count + i) {
                g.source_module = globals[src_idx].source_module;
            }
        }
        globals[import_count + i] = g;
    }

    return globals;
}

/// Evaluate a constant init expression.
fn evalInitExpr(expr: types.InitExpr, preceding_globals: []const *types.GlobalInstance) InstantiationError!types.Value {
    return switch (expr) {
        .i32_const => |v| .{ .i32 = v },
        .i64_const => |v| .{ .i64 = v },
        .f32_const => |v| .{ .f32 = v },
        .f64_const => |v| .{ .f64 = v },
        .global_get => |idx| {
            if (idx >= preceding_globals.len) return error.InvalidGlobalIndex;
            return preceding_globals[idx].value;
        },
        .ref_null => |vt| switch (vt) {
            .funcref, .nonfuncref => return .{ .funcref = null },
            .externref, .nonexternref => return .{ .externref = null },
            else => return error.InvalidInitExpr,
        },
        .ref_func => |idx| .{ .nonfuncref = idx },
        .bytecode => |code| evalInitBytecode(code, preceding_globals),
    };
}

/// Evaluate compound constant expression bytecode using a mini stack machine.
fn evalInitBytecode(code: []const u8, globals: []const *types.GlobalInstance) InstantiationError!types.Value {
    var stack: [16]types.Value = undefined;
    var sp: u32 = 0;
    var ip: usize = 0;
    while (ip < code.len) {
        const op = code[ip];
        ip += 1;
        switch (op) {
            0x41 => { // i32.const
                const r = leb128_mod.readSigned(i32, code[ip..]) catch return error.InvalidInitExpr;
                ip += r.bytes_read;
                if (sp >= stack.len) return error.InvalidInitExpr;
                stack[sp] = .{ .i32 = r.value };
                sp += 1;
            },
            0x42 => { // i64.const
                const r = leb128_mod.readSigned(i64, code[ip..]) catch return error.InvalidInitExpr;
                ip += r.bytes_read;
                if (sp >= stack.len) return error.InvalidInitExpr;
                stack[sp] = .{ .i64 = r.value };
                sp += 1;
            },
            0x43 => { // f32.const
                if (ip + 4 > code.len) return error.InvalidInitExpr;
                const bits = std.mem.readInt(u32, code[ip..][0..4], .little);
                ip += 4;
                if (sp >= stack.len) return error.InvalidInitExpr;
                stack[sp] = .{ .f32 = @bitCast(bits) };
                sp += 1;
            },
            0x44 => { // f64.const
                if (ip + 8 > code.len) return error.InvalidInitExpr;
                const bits = std.mem.readInt(u64, code[ip..][0..8], .little);
                ip += 8;
                if (sp >= stack.len) return error.InvalidInitExpr;
                stack[sp] = .{ .f64 = @bitCast(bits) };
                sp += 1;
            },
            0x23 => { // global.get
                const r = leb128_mod.readUnsigned(u32, code[ip..]) catch return error.InvalidInitExpr;
                ip += r.bytes_read;
                if (r.value >= globals.len) return error.InvalidGlobalIndex;
                if (sp >= stack.len) return error.InvalidInitExpr;
                stack[sp] = globals[r.value].value;
                sp += 1;
            },
            0xD0 => { // ref.null
                if (ip >= code.len) return error.InvalidInitExpr;
                const ht = code[ip];
                ip += 1;
                if (sp >= stack.len) return error.InvalidInitExpr;
                if (ht == 0x6F or ht == 0x72) {
                    stack[sp] = .{ .externref = null };
                } else {
                    stack[sp] = .{ .funcref = null };
                }
                sp += 1;
            },
            0xD2 => { // ref.func
                const r = leb128_mod.readUnsigned(u32, code[ip..]) catch return error.InvalidInitExpr;
                ip += r.bytes_read;
                if (sp >= stack.len) return error.InvalidInitExpr;
                stack[sp] = .{ .nonfuncref = r.value };
                sp += 1;
            },
            // i32 arithmetic
            0x6A => { if (sp < 2) return error.InvalidInitExpr; sp -= 1; stack[sp - 1] = .{ .i32 = stack[sp - 1].i32 +% stack[sp].i32 }; },
            0x6B => { if (sp < 2) return error.InvalidInitExpr; sp -= 1; stack[sp - 1] = .{ .i32 = stack[sp - 1].i32 -% stack[sp].i32 }; },
            0x6C => { if (sp < 2) return error.InvalidInitExpr; sp -= 1; stack[sp - 1] = .{ .i32 = stack[sp - 1].i32 *% stack[sp].i32 }; },
            // i64 arithmetic
            0x7C => { if (sp < 2) return error.InvalidInitExpr; sp -= 1; stack[sp - 1] = .{ .i64 = stack[sp - 1].i64 +% stack[sp].i64 }; },
            0x7D => { if (sp < 2) return error.InvalidInitExpr; sp -= 1; stack[sp - 1] = .{ .i64 = stack[sp - 1].i64 -% stack[sp].i64 }; },
            0x7E => { if (sp < 2) return error.InvalidInitExpr; sp -= 1; stack[sp - 1] = .{ .i64 = stack[sp - 1].i64 *% stack[sp].i64 }; },
            else => return error.InvalidInitExpr,
        }
    }
    if (sp != 1) return error.InvalidInitExpr;
    return stack[0];
}

// ─── Segment application ────────────────────────────────────────────────────

fn applyDataSegments(module: *const types.WasmModule, memories: []*types.MemoryInstance, globals: []const *types.GlobalInstance) InstantiationError!void {
    for (module.data_segments) |seg| {
        if (seg.is_passive) continue;

        const mem_idx = seg.memory_idx;
        if (mem_idx >= memories.len) return error.DataSegmentOutOfBounds;
        const mem = memories[mem_idx];

        const offset = evalInitExprAsU32(seg.offset, globals) catch
            return error.DataSegmentOutOfBounds;

        const end = @as(u64, offset) + seg.data.len;
        if (end > mem.data.len) return error.DataSegmentOutOfBounds;

        @memcpy(mem.data[offset..][0..seg.data.len], seg.data);
    }
}

fn applyElemSegments(module: *const types.WasmModule, tables: []*types.TableInstance, inst: *types.ModuleInstance, globals: []const *types.GlobalInstance) InstantiationError!void {
    for (module.elements) |seg| {
        if (seg.is_passive or seg.is_declarative) continue;

        const table_idx = seg.table_idx;
        if (table_idx >= tables.len) return error.ElemSegmentOutOfBounds;
        var table = tables[table_idx];

        const offset_expr = seg.offset orelse continue;
        const offset = evalInitExprAsU32(offset_expr, globals) catch
            return error.ElemSegmentOutOfBounds;

        const end = @as(u64, offset) + seg.func_indices.len;
        if (end > table.elements.len) return error.ElemSegmentOutOfBounds;

        for (seg.func_indices, 0..) |mfunc_idx, i| {
            // Check if this element has a runtime init expression
            if (seg.elem_exprs.len > i) {
                if (seg.elem_exprs[i]) |expr| {
                    switch (expr) {
                        .ref_func => |fidx| {
                            table.elements[offset + i] = .{ .func_idx = fidx, .module_inst = inst };
                        },
                        .global_get => |gidx| {
                            // Evaluate global to get funcref value
                            if (gidx < globals.len) {
                                const global = globals[gidx];
                                const gval = global.value;
                                // Use the source module that owns the function
                                const src_inst = global.source_module orelse inst;
                                if (gval.funcref) |fidx| {
                                    table.elements[offset + i] = .{ .func_idx = fidx, .module_inst = src_inst };
                                } else if (gval.nonfuncref) |fidx| {
                                    table.elements[offset + i] = .{ .func_idx = fidx, .module_inst = src_inst };
                                } else {
                                    table.elements[offset + i] = null;
                                }
                            } else {
                                table.elements[offset + i] = null;
                            }
                        },
                        else => {
                            table.elements[offset + i] = if (mfunc_idx) |func_idx|
                                .{ .func_idx = func_idx, .module_inst = inst }
                            else
                                null;
                        },
                    }
                    continue;
                }
            }
            // Fallback: use func_indices directly
            table.elements[offset + i] = if (mfunc_idx) |func_idx|
                .{ .func_idx = func_idx, .module_inst = inst }
            else
                null;
        }
    }
}

/// Helper: evaluate an init expr, returning the result as a u32 offset.
fn evalInitExprAsU32(expr: types.InitExpr, globals: []const *types.GlobalInstance) InstantiationError!u32 {
    const val = try evalInitExpr(expr, globals);
    return switch (val) {
        .i32 => |v| if (v < 0) return error.DataSegmentOutOfBounds else @intCast(v),
        .i64 => |v| if (v < 0 or v > std.math.maxInt(u32)) return error.DataSegmentOutOfBounds else @intCast(v),
        else => return error.InvalidInitExpr,
    };
}

// ─── Cleanup helpers ────────────────────────────────────────────────────────

fn freeMemories(memories: []*types.MemoryInstance, allocator: std.mem.Allocator) void {
    for (memories) |m| m.release(allocator);
    if (memories.len > 0) allocator.free(memories);
}

fn freeTables(tables: []*types.TableInstance, allocator: std.mem.Allocator) void {
    for (tables) |t| t.release(allocator);
    if (tables.len > 0) allocator.free(tables);
}

fn freeGlobals(globals: []*types.GlobalInstance, import_count: u32, allocator: std.mem.Allocator) void {
    // Imported globals use refcounting; locally-created globals are destroyed directly
    for (globals[0..import_count]) |g| g.release(allocator);
    for (globals[import_count..]) |g| allocator.destroy(g);
    if (globals.len > 0) allocator.free(globals);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;
const loader = @import("loader.zig");

/// Wasm header: \0asm followed by version 1.
const wasm_header = [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 };

test "instantiate: empty module" {
    // Minimal valid wasm with no sections.
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const module = try loader.load(&wasm_header, arena.allocator());

    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    try testing.expectEqual(@as(usize, 0), inst.memories.len);
    try testing.expectEqual(@as(usize, 0), inst.tables.len);
    try testing.expectEqual(@as(usize, 0), inst.globals.len);
}

test "instantiate: module with one memory page" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const data = wasm_header ++ [_]u8{
        // memory section: 1 memory, limits flag=0 (no max), min=1
        0x05, 0x03, 0x01, 0x00, 0x01,
    };
    const module = try loader.load(&data, arena.allocator());
    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    try testing.expectEqual(@as(usize, 1), inst.memories.len);
    try testing.expectEqual(@as(u32, 1), inst.memories[0].current_pages);
    try testing.expectEqual(@as(usize, 65536), inst.memories[0].data.len);
    // Memory should be zero-initialized.
    for (inst.memories[0].data) |b| try testing.expectEqual(@as(u8, 0), b);
}

test "instantiate: module with table" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const data = wasm_header ++ [_]u8{
        // table section: 1 table, funcref, limits no max, min=4
        0x04, 0x04, 0x01, 0x70, 0x00, 0x04,
    };
    const module = try loader.load(&data, arena.allocator());
    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    try testing.expectEqual(@as(usize, 1), inst.tables.len);
    try testing.expectEqual(@as(usize, 4), inst.tables[0].elements.len);
    // All elements should be null.
    for (inst.tables[0].elements) |e| try testing.expectEqual(@as(?types.FuncRef, null), e);
}

test "instantiate: module with global i32.const 42" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const data = wasm_header ++ [_]u8{
        // global section: 1 global, i32 mutable, init=i32.const(42), end
        0x06, 0x06, 0x01, 0x7F, 0x01, 0x41, 0x2A, 0x0B,
    };
    const module = try loader.load(&data, arena.allocator());
    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    try testing.expectEqual(@as(usize, 1), inst.globals.len);
    try testing.expectEqual(@as(i32, 42), inst.globals[0].value.i32);
}

test "instantiate: data segment copied to memory" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const data = wasm_header ++ [_]u8{
        // memory section: 1 memory, min=1
        0x05, 0x03, 0x01, 0x00, 0x01,
        // data section: 1 segment, flags=0, i32.const(0), end, 2 bytes "hi"
        0x0B, 0x08, 0x01, 0x00, 0x41, 0x00, 0x0B, 0x02, 'h', 'i',
    };
    const module = try loader.load(&data, arena.allocator());
    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    try testing.expectEqual(@as(u8, 'h'), inst.memories[0].data[0]);
    try testing.expectEqual(@as(u8, 'i'), inst.memories[0].data[1]);
    // Rest should still be zero.
    try testing.expectEqual(@as(u8, 0), inst.memories[0].data[2]);
}

test "instantiate: data segment with nonzero offset" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const data = wasm_header ++ [_]u8{
        // memory section: 1 memory, min=1
        0x05, 0x03, 0x01, 0x00, 0x01,
        // data section: 1 segment, flags=0, i32.const(16), end, 3 bytes "abc"
        0x0B, 0x09, 0x01, 0x00, 0x41, 0x10, 0x0B, 0x03, 'a', 'b', 'c',
    };
    const module = try loader.load(&data, arena.allocator());
    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    try testing.expectEqual(@as(u8, 'a'), inst.memories[0].data[16]);
    try testing.expectEqual(@as(u8, 'b'), inst.memories[0].data[17]);
    try testing.expectEqual(@as(u8, 'c'), inst.memories[0].data[18]);
}

// TODO: Fix test binary encoding to match loader expectations
// test "instantiate: elem segment fills table"
// test "instantiate: data segment out of bounds"
// test "instantiate: destroy cleans up without leaks"

test "evalInitExpr: all constant types" {
    const i32_val = try evalInitExpr(.{ .i32_const = -5 }, &.{});
    try testing.expectEqual(@as(i32, -5), i32_val.i32);

    const i64_val = try evalInitExpr(.{ .i64_const = 100 }, &.{});
    try testing.expectEqual(@as(i64, 100), i64_val.i64);

    const f32_val = try evalInitExpr(.{ .f32_const = 3.14 }, &.{});
    try testing.expectApproxEqAbs(@as(f32, 3.14), f32_val.f32, 0.001);

    const f64_val = try evalInitExpr(.{ .f64_const = 2.718 }, &.{});
    try testing.expectApproxEqAbs(@as(f64, 2.718), f64_val.f64, 0.001);

    const ref_null_val = try evalInitExpr(.{ .ref_null = .funcref }, &.{});
    try testing.expectEqual(@as(?u32, null), ref_null_val.funcref);

    const ref_func_val = try evalInitExpr(.{ .ref_func = 5 }, &.{});
    try testing.expectEqual(@as(?u32, 5), ref_func_val.nonfuncref);
}

test "evalInitExpr: global_get references preceding global" {
    var global = types.GlobalInstance{
        .global_type = .{ .val_type = .i32, .mutability = .immutable },
        .value = .{ .i32 = 99 },
    };
    const globals = [_]*types.GlobalInstance{&global};
    const val = try evalInitExpr(.{ .global_get = 0 }, &globals);
    try testing.expectEqual(@as(i32, 99), val.i32);
}

test "evalInitExpr: global_get out of range" {
    const result = evalInitExpr(.{ .global_get = 5 }, &.{});
    try testing.expectError(error.InvalidGlobalIndex, result);
}
