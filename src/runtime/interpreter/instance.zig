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
    tags: []const *types.TagInstance = &.{},
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
    return instantiateImpl(module, allocator, import_ctx, null);
}

/// Instantiate a module with host (native) functions from a comptime HostImports table.
/// Custom host functions take priority; unmatched imports fall back to WASI.
pub fn instantiateWithHosts(
    module: *const types.WasmModule,
    allocator: std.mem.Allocator,
    comptime HostImportsT: type,
) InstantiationError!*types.ModuleInstance {
    return instantiateImpl(module, allocator, null, HostImportsT);
}

fn instantiateImpl(
    module: *const types.WasmModule,
    allocator: std.mem.Allocator,
    import_ctx: ?ImportContext,
    comptime HostImportsT: ?type,
) InstantiationError!*types.ModuleInstance {
    const has_non_func_imports =
        module.import_global_count > 0 or
        module.import_memory_count > 0 or
        module.import_table_count > 0 or
        module.import_tag_count > 0;

    // Non-function imports still require an explicit ImportContext
    if (has_non_func_imports and import_ctx == null) {
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

    // Initialize waiter queues for shared memories
    for (inst.memories) |mem| {
        if (mem.memory_type.is_shared and mem.waiter_queue == null) {
            const wq = allocator.create(types.WaiterQueue) catch return error.OutOfMemory;
            wq.* = .{};
            mem.waiter_queue = wq;
        }
    }

    inst.tables = try allocateTables(module, allocator, import_ctx);
    errdefer freeTables(inst.tables, allocator);

    inst.globals = try initializeGlobals(module, allocator, import_ctx, inst);
    errdefer freeGlobals(inst.globals, module.import_global_count, allocator);

    // Store imported function references
    if (import_ctx) |ctx| {
        if (ctx.functions.len > 0) {
            inst.import_functions = allocator.dupe(types.ImportedFunction, ctx.functions) catch return error.OutOfMemory;
        }
    }

    // Resolve host functions for function imports.
    // Priority: custom HostImports (if provided) > WASI auto-resolution.
    if (module.import_function_count > 0) {
        const host_fns = allocator.alloc(?types.HostFn, module.import_function_count) catch return error.OutOfMemory;
        @memset(host_fns, null);

        // Layer 1: custom host functions from HostImports (comptime-resolved)
        if (HostImportsT) |HI| {
            var func_idx: u32 = 0;
            for (module.imports) |imp| {
                if (imp.kind == .function) {
                    if (func_idx < module.import_function_count) {
                        if (HI.resolve(imp.module_name, imp.field_name)) |entry| {
                            host_fns[func_idx] = entry.interp_fn;
                        }
                    }
                    func_idx += 1;
                }
            }
        }

        // Layer 2: WASI auto-resolution for any remaining nulls
        const wasi_host = @import("../../wasi/host_functions.zig");
        const wasi_fns = wasi_host.resolveWasiHostFunctions(module, allocator) catch return error.OutOfMemory;
        defer allocator.free(wasi_fns);
        for (wasi_fns, 0..) |wasi_fn, i| {
            if (host_fns[i] == null) host_fns[i] = wasi_fn;
        }

        inst.host_functions = host_fns;
        inst.owns_host_functions = true;
    }

    // Allocate tags: imported tags + locally defined tags
    const total_tags = module.import_tag_count + @as(u32, @intCast(module.tag_types.len));
    if (total_tags > 0) {
        inst.tags = allocator.alloc(*types.TagInstance, total_tags) catch return error.OutOfMemory;
        // Imported tags: shared pointers from exporting instances
        if (import_ctx) |ctx| {
            for (ctx.tags, 0..) |t, i| {
                inst.tags[i] = t;
            }
        }
        // Locally defined tags: create new instances
        for (module.tag_types, 0..) |type_idx, i| {
            const tag = allocator.create(types.TagInstance) catch return error.OutOfMemory;
            const arity: u32 = if (type_idx < module.types.len) @intCast(module.types[type_idx].params.len) else 0;
            tag.* = .{ .param_arity = arity };
            inst.tags[module.import_tag_count + i] = tag;
        }
    }

    try applyTableInitExprs(module, inst.tables, inst, inst.globals);
    try applyElemSegments(module, inst.tables, inst, inst.globals);
    try applyDataSegments(module, inst.memories, inst.globals);

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

    // Cache evaluated element segment values (spec requires one-time evaluation)
    if (module.elements.len > 0) {
        inst.cached_elem_values = allocator.alloc(?[]types.Value, module.elements.len) catch return error.OutOfMemory;
        for (module.elements, 0..) |elem, seg_i| {
            if (elem.elem_exprs.len == 0 and elem.func_indices.len == 0) {
                inst.cached_elem_values[seg_i] = null;
                continue;
            }
            const count = @max(elem.elem_exprs.len, elem.func_indices.len);
            const vals = allocator.alloc(types.Value, count) catch {
                inst.cached_elem_values[seg_i] = null;
                continue;
            };
            for (0..count) |i| {
                const si = @as(u32, @intCast(i));
                if (elem.elem_exprs.len > si) {
                    if (elem.elem_exprs[si]) |expr| {
                        switch (expr) {
                            .ref_func => |fidx| {
                                vals[i] = .{ .nonfuncref = fidx };
                                continue;
                            },
                            .bytecode => |bc| {
                                vals[i] = evalInitBytecode(bc, &.{}, inst) catch .{ .funcref = null };
                                continue;
                            },
                            else => {},
                        }
                    }
                }
                if (si < elem.func_indices.len) {
                    vals[i] = if (elem.func_indices[si]) |fi| .{ .nonfuncref = fi } else .{ .funcref = null };
                } else {
                    vals[i] = .{ .funcref = null };
                }
            }
            inst.cached_elem_values[seg_i] = vals;
        }
    }

    // Initialize dropped data segments tracking
    if (module.data_segments.len > 0) {
        inst.dropped_data = allocator.alloc(bool, module.data_segments.len) catch return error.OutOfMemory;
        @memset(inst.dropped_data, false);
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

/// Attach a slice of context-carrying host function entries to an already
/// instantiated module. Used by higher layers (e.g. the component-model
/// canon-lower trampoline) to install per-slot host state.
///
/// The slice length must equal `inst.module.import_function_count`; each slot
/// maps to an imported function index. `null` entries fall through to the
/// existing legacy `host_functions` slot and the WASI resolver.
///
/// Ownership transfers: `inst` will free `entries` on destroy. Caller must
/// not free the slice separately.
///
/// NOTE: entries installed this way are NOT visible to a core module's start
/// function, which runs during `instantiate` before this call. Modules that
/// need host dispatch during start should extend the `instantiate` API to
/// accept entries up front (Phase 2A follow-up).
pub fn attachHostFuncEntries(
    inst: *types.ModuleInstance,
    entries: []?types.HostFnEntry,
) void {
    const allocator = inst.allocator;
    if (inst.owns_host_func_entries and inst.host_func_entries.len > 0) {
        allocator.free(inst.host_func_entries);
    }
    inst.host_func_entries = entries;
    inst.owns_host_func_entries = true;
}

/// Destroy a module instance, freeing all allocated resources.
pub fn destroy(inst: *types.ModuleInstance) void {
    const allocator = inst.allocator;
    freeMemories(inst.memories, allocator);
    freeTables(inst.tables, allocator);
    freeGlobals(inst.globals, inst.module.import_global_count, allocator);
    if (inst.import_functions.len > 0) allocator.free(inst.import_functions);
    if (inst.owns_host_functions and inst.host_functions.len > 0)
        allocator.free(inst.host_functions);
    if (inst.owns_host_func_entries and inst.host_func_entries.len > 0)
        allocator.free(inst.host_func_entries);
    // Free locally defined tags (imported tags are owned by their source instance)
    for (inst.tags[inst.module.import_tag_count..]) |t| allocator.destroy(t);
    if (inst.tags.len > 0) allocator.free(inst.tags);
    if (inst.dropped_elems.len > 0) allocator.free(inst.dropped_elems);
    if (inst.dropped_data.len > 0) allocator.free(inst.dropped_data);
    // Free cached element segment values
    for (inst.cached_elem_values) |maybe_vals| {
        if (maybe_vals) |vals| allocator.free(vals);
    }
    if (inst.cached_elem_values.len > 0) allocator.free(inst.cached_elem_values);
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
        const min_pages: u32 = @intCast(@min(mem_type.limits.min, 65536));
        const max_pages: u32 = @intCast(@min(mem_type.limits.max orelse 65536, 65536));
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
        const min_elems: usize = @intCast(@min(table_type.limits.min, std.math.maxInt(usize)));
        const elems = allocator.alloc(types.TableElement, min_elems) catch
            return error.TableAllocationFailed;
        for (elems) |*e| e.* = types.TableElement.nullForType(table_type.elem_type);

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

fn initializeGlobals(module: *const types.WasmModule, allocator: std.mem.Allocator, import_ctx: ?ImportContext, inst: *types.ModuleInstance) InstantiationError![]*types.GlobalInstance {
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
            .value = try evalInitExpr(global.init_expr, globals[0 .. import_count + i], inst),
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
pub fn evalInitExpr(expr: types.InitExpr, preceding_globals: []const *types.GlobalInstance, inst: ?*types.ModuleInstance) InstantiationError!types.Value {
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
            .anyref => return .{ .anyref = null },
            .eqref => return .{ .eqref = null },
            .i31ref => return .{ .i31ref = null },
            .structref => return .{ .structref = null },
            .arrayref => return .{ .arrayref = null },
            .nullref => return .{ .nullref = null },
            .exnref => return .{ .exnref = null },
            else => return error.InvalidInitExpr,
        },
        .ref_func => |idx| .{ .nonfuncref = idx },
        .bytecode => |code| evalInitBytecode(code, preceding_globals, inst),
    };
}

/// Evaluate compound constant expression bytecode using a mini stack machine.
pub fn evalInitBytecode(code: []const u8, globals: []const *types.GlobalInstance, inst: ?*types.ModuleInstance) InstantiationError!types.Value {
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
                // For concrete type indices, consume remaining LEB128 bytes
                if (ht & 0x80 != 0) {
                    while (ip < code.len) {
                        if (code[ip] & 0x80 == 0) { ip += 1; break; }
                        ip += 1;
                    }
                }
                if (sp >= stack.len) return error.InvalidInitExpr;
                stack[sp] = switch (ht) {
                    0x6F, 0x72, 0x74 => .{ .externref = null },
                    0x6E => .{ .anyref = null },
                    0x6D => .{ .eqref = null },
                    0x6C => .{ .i31ref = null },
                    0x6B => .{ .structref = null },
                    0x6A => .{ .arrayref = null },
                    0x65 => .{ .nullref = null },
                    0x69, 0x68 => .{ .exnref = null },
                    else => .{ .funcref = null },
                };
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
            // GC prefix opcodes
            0xFB => {
                const r = leb128_mod.readUnsigned(u32, code[ip..]) catch return error.InvalidInitExpr;
                ip += r.bytes_read;
                switch (r.value) {
                    0x1C => { // ref.i31: pop i32, push i31ref
                        if (sp < 1) return error.InvalidInitExpr;
                        const i32_val = stack[sp - 1].i32;
                        stack[sp - 1] = .{ .i31ref = @as(u32, @bitCast(i32_val)) & 0x7FFF_FFFF };
                    },
                    0x1A => { // any.convert_extern: pop externref, push anyref
                        if (sp < 1) return error.InvalidInitExpr;
                        const val: ?u32 = switch (stack[sp - 1]) {
                            .externref, .nonexternref => |v| v,
                            .funcref, .nonfuncref => |v| v,
                            else => null,
                        };
                        stack[sp - 1] = .{ .anyref = val };
                    },
                    0x1B => { // extern.convert_any: pop anyref, push externref
                        if (sp < 1) return error.InvalidInitExpr;
                        const val: ?u32 = switch (stack[sp - 1]) {
                            .funcref, .nonfuncref => |v| v,
                            .externref, .nonexternref => |v| v,
                            else => null,
                        };
                        stack[sp - 1] = .{ .externref = val };
                    },
                    0x00, 0x01 => { // struct.new, struct.new_default
                        const type_idx_r = leb128_mod.readUnsigned(u32, code[ip..]) catch return error.InvalidInitExpr;
                        ip += type_idx_r.bytes_read;
                        const type_idx = type_idx_r.value;
                        if (inst) |mi| {
                            const module = mi.module;
                            if (type_idx >= module.types.len) return error.InvalidInitExpr;
                            const ft = module.types[type_idx];
                            const field_count = ft.field_types.len;
                            var fields_buf: [256]types.Value = undefined;
                            if (field_count > fields_buf.len) return error.InvalidInitExpr;
                            if (r.value == 0x01) { // struct.new_default
                                for (ft.field_types, 0..) |fvt, fi| {
                                    fields_buf[fi] = switch (fvt) {
                                        .i32 => .{ .i32 = 0 },
                                        .i64 => .{ .i64 = 0 },
                                        .f32 => .{ .f32 = 0.0 },
                                        .f64 => .{ .f64 = 0.0 },
                                        .funcref => .{ .funcref = null },
                                        .externref => .{ .externref = null },
                                        .anyref => .{ .anyref = null },
                                        .eqref => .{ .eqref = null },
                                        .i31ref => .{ .i31ref = null },
                                        .structref => .{ .structref = null },
                                        .arrayref => .{ .arrayref = null },
                                        .nullref => .{ .nullref = null },
                                        .v128 => .{ .v128 = 0 },
                                        else => .{ .i32 = 0 },
                                    };
                                }
                            } else { // struct.new: pop fields in reverse
                                if (sp < field_count) return error.InvalidInitExpr;
                                var fi = field_count;
                                while (fi > 0) {
                                    fi -= 1;
                                    sp -= 1;
                                    fields_buf[fi] = stack[sp];
                                }
                            }
                            const fields_copy = mi.allocator.alloc(types.Value, field_count) catch return error.InvalidInitExpr;
                            @memcpy(fields_copy, fields_buf[0..field_count]);
                            const obj_idx: u32 = @intCast(mi.gc_objects.items.len);
                            mi.gc_objects.append(mi.allocator, .{ .type_idx = type_idx, .fields = fields_copy }) catch return error.InvalidInitExpr;
                            if (sp >= stack.len) return error.InvalidInitExpr;
                            stack[sp] = .{ .structref = obj_idx };
                            sp += 1;
                        } else {
                            // No module instance — push null placeholder
                            if (r.value == 0x00) {
                                // struct.new: need to pop fields but don't know count without types
                                return error.InvalidInitExpr;
                            }
                            if (sp >= stack.len) return error.InvalidInitExpr;
                            stack[sp] = .{ .structref = null };
                            sp += 1;
                        }
                    },
                    0x08 => { // array.new_fixed: type_idx + count
                        const type_idx_r = leb128_mod.readUnsigned(u32, code[ip..]) catch return error.InvalidInitExpr;
                        ip += type_idx_r.bytes_read;
                        const count_r = leb128_mod.readUnsigned(u32, code[ip..]) catch return error.InvalidInitExpr;
                        ip += count_r.bytes_read;
                        const elem_count = count_r.value;
                        if (sp < elem_count) return error.InvalidInitExpr;
                        if (inst) |mi| {
                            var elems_buf: [65536]types.Value = undefined;
                            if (elem_count > elems_buf.len) return error.InvalidInitExpr;
                            // Pop elements in reverse
                            var ei = elem_count;
                            while (ei > 0) {
                                ei -= 1;
                                sp -= 1;
                                elems_buf[ei] = stack[sp];
                            }
                            const fields_copy = mi.allocator.alloc(types.Value, elem_count) catch return error.InvalidInitExpr;
                            @memcpy(fields_copy, elems_buf[0..elem_count]);
                            const obj_idx: u32 = @intCast(mi.gc_objects.items.len);
                            mi.gc_objects.append(mi.allocator, .{ .type_idx = type_idx_r.value, .fields = fields_copy }) catch return error.InvalidInitExpr;
                            if (sp >= stack.len) return error.InvalidInitExpr;
                            stack[sp] = .{ .arrayref = obj_idx };
                            sp += 1;
                        } else {
                            sp -= elem_count;
                            if (sp >= stack.len) return error.InvalidInitExpr;
                            stack[sp] = .{ .arrayref = null };
                            sp += 1;
                        }
                    },
                    0x06 => { // array.new: pop init_val + length, allocate array
                        const type_idx_r2 = leb128_mod.readUnsigned(u32, code[ip..]) catch return error.InvalidInitExpr;
                        ip += type_idx_r2.bytes_read;
                        if (sp < 2) return error.InvalidInitExpr;
                        sp -= 1;
                        const len: u32 = @bitCast(stack[sp].i32);
                        sp -= 1;
                        const init_val = stack[sp];
                        if (inst) |mi| {
                            const fields_copy = mi.allocator.alloc(types.Value, len) catch return error.InvalidInitExpr;
                            for (fields_copy) |*f| f.* = init_val;
                            const obj_idx: u32 = @intCast(mi.gc_objects.items.len);
                            mi.gc_objects.append(mi.allocator, .{ .type_idx = type_idx_r2.value, .fields = fields_copy }) catch return error.InvalidInitExpr;
                            if (sp >= stack.len) return error.InvalidInitExpr;
                            stack[sp] = .{ .arrayref = obj_idx };
                            sp += 1;
                        } else {
                            if (sp >= stack.len) return error.InvalidInitExpr;
                            stack[sp] = .{ .arrayref = null };
                            sp += 1;
                        }
                    },
                    0x07 => { // array.new_default: pop length, allocate zero-initialized array
                        const type_idx_r2 = leb128_mod.readUnsigned(u32, code[ip..]) catch return error.InvalidInitExpr;
                        ip += type_idx_r2.bytes_read;
                        if (sp < 1) return error.InvalidInitExpr;
                        sp -= 1;
                        const len: u32 = @bitCast(stack[sp].i32);
                        if (inst) |mi| {
                            const module = mi.module;
                            const elem_vt: types.ValType = if (type_idx_r2.value < module.types.len) blk: {
                                const ft = module.types[type_idx_r2.value];
                                break :blk if (ft.field_types.len > 0) ft.field_types[0] else .i32;
                            } else .i32;
                            const default_val: types.Value = switch (elem_vt) {
                                .i32 => .{ .i32 = 0 },
                                .i64 => .{ .i64 = 0 },
                                .f32 => .{ .f32 = 0.0 },
                                .f64 => .{ .f64 = 0.0 },
                                .funcref => .{ .funcref = null },
                                .externref => .{ .externref = null },
                                .anyref => .{ .anyref = null },
                                else => .{ .i32 = 0 },
                            };
                            const fields_copy = mi.allocator.alloc(types.Value, len) catch return error.InvalidInitExpr;
                            for (fields_copy) |*f| f.* = default_val;
                            const obj_idx: u32 = @intCast(mi.gc_objects.items.len);
                            mi.gc_objects.append(mi.allocator, .{ .type_idx = type_idx_r2.value, .fields = fields_copy }) catch return error.InvalidInitExpr;
                            if (sp >= stack.len) return error.InvalidInitExpr;
                            stack[sp] = .{ .arrayref = obj_idx };
                            sp += 1;
                        } else {
                            if (sp >= stack.len) return error.InvalidInitExpr;
                            stack[sp] = .{ .arrayref = null };
                            sp += 1;
                        }
                    },
                    else => return error.InvalidInitExpr,
                }
            },
            // SIMD prefix: v128.const
            0xFD => {
                const r = leb128_mod.readUnsigned(u32, code[ip..]) catch return error.InvalidInitExpr;
                ip += r.bytes_read;
                switch (r.value) {
                    0x0C => { // v128.const: 16 bytes
                        if (ip + 16 > code.len) return error.InvalidInitExpr;
                        const bits = std.mem.readInt(u128, code[ip..][0..16], .little);
                        ip += 16;
                        if (sp >= stack.len) return error.InvalidInitExpr;
                        stack[sp] = .{ .v128 = bits };
                        sp += 1;
                    },
                    else => return error.InvalidInitExpr,
                }
            },
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

        // Use u64 offset for memory64, u32 for memory32
        const offset: u64 = if (mem.memory_type.is_memory64)
            evalInitExprAsU64(seg.offset, globals) catch return error.DataSegmentOutOfBounds
        else
            @as(u64, evalInitExprAsU32(seg.offset, globals) catch return error.DataSegmentOutOfBounds);

        const end = offset + seg.data.len;
        if (end > mem.data.len) return error.DataSegmentOutOfBounds;

        const off: usize = @intCast(offset);
        @memcpy(mem.data[off..][0..seg.data.len], seg.data);
    }
}

fn applyTableInitExprs(module: *const types.WasmModule, tables: []*types.TableInstance, inst: *types.ModuleInstance, globals: []const *types.GlobalInstance) InstantiationError!void {
    const import_count = module.import_table_count;
    for (module.tables, 0..) |_, i| {
        const table_type = module.tables[i];
        const init_expr = table_type.init_expr orelse continue;
        const table = tables[import_count + i];
        const val = evalInitExpr(init_expr, globals, inst) catch continue;
        const elem = types.TableElement.fromValue(val, inst);
        for (table.elements) |*e| {
            e.* = elem;
        }
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
                            table.elements[offset + i] = .{
                                .value = .{ .nonfuncref = fidx },
                                .module_inst = inst,
                            };
                        },
                        .global_get => |gidx| {
                            if (gidx < globals.len) {
                                const global = globals[gidx];
                                const gval = global.value;
                                const src_inst = global.source_module orelse inst;
                                table.elements[offset + i] = types.TableElement.fromValue(gval, src_inst);
                            } else {
                                table.elements[offset + i] = types.TableElement.nullForType(table.table_type.elem_type);
                            }
                        },
                        .bytecode => {
                            const bc_val = evalInitExpr(expr, globals, inst) catch {
                                table.elements[offset + i] = types.TableElement.nullForType(table.table_type.elem_type);
                                continue;
                            };
                            table.elements[offset + i] = types.TableElement.fromValue(bc_val, inst);
                        },
                        else => {
                            table.elements[offset + i] = if (mfunc_idx) |func_idx|
                                .{ .value = .{ .nonfuncref = func_idx }, .module_inst = inst }
                            else
                                types.TableElement.nullForType(table.table_type.elem_type);
                        },
                    }
                    continue;
                }
            }
            // Fallback: use func_indices directly
            table.elements[offset + i] = if (mfunc_idx) |func_idx|
                .{ .value = .{ .nonfuncref = func_idx }, .module_inst = inst }
            else
                types.TableElement.nullForType(table.table_type.elem_type);
        }
    }
}

/// Helper: evaluate an init expr, returning the result as a u32 offset.
fn evalInitExprAsU32(expr: types.InitExpr, globals: []const *types.GlobalInstance) InstantiationError!u32 {
    const val = try evalInitExpr(expr, globals, null);
    return switch (val) {
        .i32 => |v| if (v < 0) return error.DataSegmentOutOfBounds else @intCast(v),
        .i64 => |v| if (v < 0 or v > std.math.maxInt(u32)) return error.DataSegmentOutOfBounds else @intCast(v),
        else => return error.InvalidInitExpr,
    };
}

/// Helper: evaluate an init expr, returning the result as a u64 offset (memory64).
fn evalInitExprAsU64(expr: types.InitExpr, globals: []const *types.GlobalInstance) InstantiationError!u64 {
    const val = try evalInitExpr(expr, globals, null);
    return switch (val) {
        .i64 => |v| if (v < 0) return error.DataSegmentOutOfBounds else @intCast(v),
        .i32 => |v| if (v < 0) return error.DataSegmentOutOfBounds else @intCast(v),
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
    for (inst.tables[0].elements) |e| try testing.expect(e.isNull());
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
    const i32_val = try evalInitExpr(.{ .i32_const = -5 }, &.{}, null);
    try testing.expectEqual(@as(i32, -5), i32_val.i32);

    const i64_val = try evalInitExpr(.{ .i64_const = 100 }, &.{}, null);
    try testing.expectEqual(@as(i64, 100), i64_val.i64);

    const f32_val = try evalInitExpr(.{ .f32_const = 3.14 }, &.{}, null);
    try testing.expectApproxEqAbs(@as(f32, 3.14), f32_val.f32, 0.001);

    const f64_val = try evalInitExpr(.{ .f64_const = 2.718 }, &.{}, null);
    try testing.expectApproxEqAbs(@as(f64, 2.718), f64_val.f64, 0.001);

    const ref_null_val = try evalInitExpr(.{ .ref_null = .funcref }, &.{}, null);
    try testing.expectEqual(@as(?u32, null), ref_null_val.funcref);

    const ref_func_val = try evalInitExpr(.{ .ref_func = 5 }, &.{}, null);
    try testing.expectEqual(@as(?u32, 5), ref_func_val.nonfuncref);
}

test "evalInitExpr: global_get references preceding global" {
    var global = types.GlobalInstance{
        .global_type = .{ .val_type = .i32, .mutability = .immutable },
        .value = .{ .i32 = 99 },
    };
    const globals = [_]*types.GlobalInstance{&global};
    const val = try evalInitExpr(.{ .global_get = 0 }, &globals, null);
    try testing.expectEqual(@as(i32, 99), val.i32);
}

test "evalInitExpr: global_get out of range" {
    const result = evalInitExpr(.{ .global_get = 5 }, &.{}, null);
    try testing.expectError(error.InvalidGlobalIndex, result);
}

test "instantiate: host functions resolved for wasi thread-spawn import" {
    const imports = [_]types.ImportDesc{
        .{
            .module_name = "wasi",
            .field_name = "thread-spawn",
            .kind = .function,
            .func_type_idx = 0,
        },
    };
    const func_types = [_]types.FuncType{
        .{ .params = &.{.i32}, .results = &.{.i32} },
    };
    var module = types.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
    };
    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    // Host functions should be resolved
    try testing.expectEqual(@as(usize, 1), inst.host_functions.len);
    try testing.expect(inst.host_functions[0] != null);
    try testing.expect(inst.owns_host_functions);
}

test "instantiate: non-wasi imports have null host functions" {
    const imports = [_]types.ImportDesc{
        .{
            .module_name = "env",
            .field_name = "some_func",
            .kind = .function,
            .func_type_idx = 0,
        },
    };
    const func_types = [_]types.FuncType{
        .{ .params = &.{}, .results = &.{} },
    };
    var module = types.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
    };
    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    try testing.expectEqual(@as(usize, 1), inst.host_functions.len);
    try testing.expect(inst.host_functions[0] == null);
}

test "attachHostFuncEntries: context-carrying host fn receives ctx and takes priority" {
    const imports = [_]types.ImportDesc{
        .{
            .module_name = "host",
            .field_name = "addn",
            .kind = .function,
            .func_type_idx = 0,
        },
    };
    const func_types = [_]types.FuncType{
        .{ .params = &.{ .i32, .i32 }, .results = &.{.i32} },
    };
    var module = types.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
    };
    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    const TestCtx = struct {
        bias: i32,
        const Self = @This();
        fn trampoline(env_opaque: *anyopaque, ctx: ?*anyopaque) types.HostFnError!void {
            const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
            const self: *Self = @ptrCast(@alignCast(ctx.?));
            // Stack is LIFO: pop b first, then a.
            const b = env.popI32() catch return error.StackUnderflow;
            const a = env.popI32() catch return error.StackUnderflow;
            env.pushI32(a + b + self.bias) catch return error.StackOverflow;
        }
    };
    var tctx = TestCtx{ .bias = 100 };
    const entries = try testing.allocator.alloc(?types.HostFnEntry, 1);
    entries[0] = .{ .func = &TestCtx.trampoline, .ctx = @ptrCast(&tctx) };
    attachHostFuncEntries(inst, entries);

    // Execute import function index 0 with args (3, 4) -> 3 + 4 + 100 = 107.
    const env = try ExecEnv.create(inst, 256, testing.allocator);
    defer env.destroy();
    try env.pushI32(3);
    try env.pushI32(4);
    try interp.executeFunction(env, 0);
    const result = try env.popI32();
    try testing.expectEqual(@as(i32, 107), result);
}

test "attachHostFuncEntries: null entry falls through to legacy host_functions" {
    const imports = [_]types.ImportDesc{
        .{
            .module_name = "wasi",
            .field_name = "thread-spawn",
            .kind = .function,
            .func_type_idx = 0,
        },
    };
    const func_types = [_]types.FuncType{
        .{ .params = &.{.i32}, .results = &.{.i32} },
    };
    var module = types.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
        .types = &func_types,
    };
    const inst = try instantiate(&module, testing.allocator);
    defer destroy(inst);

    const entries = try testing.allocator.alloc(?types.HostFnEntry, 1);
    entries[0] = null; // explicit null — must fall through
    attachHostFuncEntries(inst, entries);

    // Legacy WASI resolver should still be active at slot 0.
    try testing.expect(inst.host_functions[0] != null);
    try testing.expect(inst.host_func_entries[0] == null);
}
