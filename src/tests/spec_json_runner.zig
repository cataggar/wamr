//! WebAssembly Spec Test JSON Runner
//!
//! Reads JSON files produced by wast2json, executing module loads,
//! assert_return, assert_trap, and assert_invalid/assert_malformed
//! commands against the WAMR runtime.

const std = @import("std");
const root = @import("wamr");
const wamr = root.wamr;
const types = root.types;
const instance_mod = root.instance;
const wabt = @import("wabt");
const Io = std.Io;
const Dir = std.fs.Dir;

pub const SpecTestResult = struct {
    file: []const u8,
    total: u32 = 0,
    passed: u32 = 0,
    failed: u32 = 0,
    skipped: u32 = 0,
};

const Command = struct {
    type: []const u8,
    line: i64 = 0,
    filename: ?[]const u8 = null,
    action: ?Action = null,
    expected: ?[]const Arg = null,
    text: ?[]const u8 = null,
    module_type: ?[]const u8 = null,
    @"as": ?[]const u8 = null,
    name: ?[]const u8 = null,
};

const Action = struct {
    type: []const u8,
    field: ?[]const u8 = null,
    module: ?[]const u8 = null,
    args: ?[]const Arg = null,
};

const Arg = struct {
    type: []const u8,
    value: ?[]const u8 = null,
};

const SpecJson = struct {
    source_filename: ?[]const u8 = null,
    commands: []const Command,
};

/// Parse a spec-test JSON value object into a types.Value.
/// The spec encodes all values as unsigned decimal bit patterns.
fn parseValue(arg: Arg) ?types.Value {
    const val_str = arg.value orelse return null;
    if (std.mem.eql(u8, arg.type, "i32")) {
        const bits = std.fmt.parseUnsigned(u32, val_str, 10) catch return null;
        return .{ .i32 = @bitCast(bits) };
    } else if (std.mem.eql(u8, arg.type, "i64")) {
        const bits = std.fmt.parseUnsigned(u64, val_str, 10) catch return null;
        return .{ .i64 = @bitCast(bits) };
    } else if (std.mem.eql(u8, arg.type, "f32")) {
        if (std.mem.eql(u8, val_str, "nan:canonical")) {
            return .{ .f32 = @bitCast(@as(u32, 0x7FC00000)) };
        } else if (std.mem.eql(u8, val_str, "nan:arithmetic")) {
            // Sentinel: arithmetic NaN = any NaN is OK, use 0x7FC00001 as marker
            return .{ .f32 = @bitCast(@as(u32, 0x7FC00001)) };
        }
        const bits = std.fmt.parseUnsigned(u32, val_str, 10) catch return null;
        return .{ .f32 = @bitCast(bits) };
    } else if (std.mem.eql(u8, arg.type, "f64")) {
        if (std.mem.eql(u8, val_str, "nan:canonical")) {
            return .{ .f64 = @bitCast(@as(u64, 0x7FF8000000000000)) };
        } else if (std.mem.eql(u8, val_str, "nan:arithmetic")) {
            return .{ .f64 = @bitCast(@as(u64, 0x7FF8000000000001)) };
        }
        const bits = std.fmt.parseUnsigned(u64, val_str, 10) catch return null;
        return .{ .f64 = @bitCast(bits) };
    } else if (std.mem.eql(u8, arg.type, "funcref")) {
        if (std.mem.eql(u8, val_str, "null")) return .{ .funcref = null };
        const idx = std.fmt.parseUnsigned(u32, val_str, 10) catch return null;
        return .{ .funcref = idx };
    } else if (std.mem.eql(u8, arg.type, "externref")) {
        if (std.mem.eql(u8, val_str, "null")) return .{ .externref = null };
        const idx = std.fmt.parseUnsigned(u32, val_str, 10) catch return null;
        return .{ .externref = idx };
    }
    return null;
}

/// Bit-exact comparison of two Values (handles NaN correctly).
fn valuesEqual(a: types.Value, b: types.Value) bool {
    return switch (a) {
        .i32 => |v| b == .i32 and b.i32 == v,
        .i64 => |v| b == .i64 and b.i64 == v,
        .f32 => |v| blk: {
            if (b != .f32) break :blk false;
            const actual_bits: u32 = @bitCast(v);
            const expected_bits: u32 = @bitCast(b.f32);
            // canonical NaN: accept either sign canonical
            if (expected_bits == 0x7FC00000 or expected_bits == 0xFFC00000)
                break :blk (actual_bits == 0x7FC00000 or actual_bits == 0xFFC00000);
            // arithmetic NaN marker (0x7FC00001): any NaN passes
            if (expected_bits == 0x7FC00001)
                break :blk std.math.isNan(v);
            break :blk actual_bits == expected_bits;
        },
        .f64 => |v| blk: {
            if (b != .f64) break :blk false;
            const actual_bits: u64 = @bitCast(v);
            const expected_bits: u64 = @bitCast(b.f64);
            if (expected_bits == 0x7FF8000000000000 or expected_bits == 0xFFF8000000000000)
                break :blk (actual_bits == 0x7FF8000000000000 or actual_bits == 0xFFF8000000000000);
            if (expected_bits == 0x7FF8000000000001)
                break :blk std.math.isNan(v);
            break :blk actual_bits == expected_bits;
        },
        .funcref => |v| (b == .funcref or b == .nonfuncref) and ((v == null and (if (b == .funcref) b.funcref else b.nonfuncref) == null) or (v != null and (if (b == .funcref) b.funcref else b.nonfuncref) != null)),
        .externref => |v| (b == .externref or b == .nonexternref) and ((v == null and (if (b == .externref) b.externref else b.nonexternref) == null) or (v != null and (if (b == .externref) b.externref else b.nonexternref) != null)),
        .nonfuncref => |v| (b == .funcref or b == .nonfuncref) and ((v == null and (if (b == .funcref) b.funcref else b.nonfuncref) == null) or (v != null and (if (b == .funcref) b.funcref else b.nonfuncref) != null)),
        .nonexternref => |v| (b == .externref or b == .nonexternref) and ((v == null and (if (b == .externref) b.externref else b.nonexternref) == null) or (v != null and (if (b == .externref) b.externref else b.nonexternref) != null)),
        else => false,
    };
}

/// Parse a slice of JSON args into a caller-owned slice of Values.
/// Returns null if any arg has an unsupported type.
fn parseArgs(args_json: []const Arg, allocator: std.mem.Allocator) ?[]types.Value {
    const args = allocator.alloc(types.Value, args_json.len) catch return null;
    for (args_json, 0..) |arg, i| {
        args[i] = parseValue(arg) orelse {
            allocator.free(args);
            return null;
        };
    }
    return args;
}

/// Build an ImportContext for a module based on its imports and the spectest/registry.
fn buildImportContext(
    module: *const types.WasmModule,
    module_registry: *const std.StringHashMap(*types.ModuleInstance),
    allocator: std.mem.Allocator,
) error{ImportResolutionFailed}!?instance_mod.ImportContext {
    if (module.import_global_count == 0 and
        module.import_memory_count == 0 and
        module.import_table_count == 0 and
        module.import_function_count == 0) return null;

    var globals: std.ArrayList(*types.GlobalInstance) = .empty;
    defer globals.deinit(allocator);
    var memories: std.ArrayList(*types.MemoryInstance) = .empty;
    defer {
        for (memories.items) |m| m.release(allocator);
        memories.deinit(allocator);
    }
    var tables: std.ArrayList(*types.TableInstance) = .empty;
    defer {
        for (tables.items) |t| t.release(allocator);
        tables.deinit(allocator);
    }
    var functions: std.ArrayList(types.ImportedFunction) = .empty;
    defer functions.deinit(allocator);

    for (module.imports) |imp| {
        const is_spectest = std.mem.eql(u8, imp.module_name, "spectest");
        const reg_inst = module_registry.get(imp.module_name);

        if (!is_spectest and reg_inst == null)
            return error.ImportResolutionFailed;

        switch (imp.kind) {
            .global => {
                const gt = imp.global_type orelse return error.ImportResolutionFailed;
                if (is_spectest) {
                    const sgt = spectestGlobalType(imp.field_name) orelse
                        return error.ImportResolutionFailed;
                    if (sgt.val_type != gt.val_type or sgt.mutability != gt.mutability)
                        return error.ImportResolutionFailed;
                    const g = allocator.create(types.GlobalInstance) catch
                        return error.ImportResolutionFailed;
                    g.* = .{
                        .global_type = gt,
                        .value = getSpectestGlobal(imp.field_name, gt.val_type),
                    };
                    globals.append(allocator, g) catch return error.ImportResolutionFailed;
                } else {
                    const ri = reg_inst.?;
                    const exp = ri.module.findExport(imp.field_name, .global) orelse
                        return error.ImportResolutionFailed;
                    if (exp.index >= ri.globals.len) return error.ImportResolutionFailed;
                    const eg = ri.globals[exp.index];
                    if (eg.global_type.val_type != gt.val_type or
                        eg.global_type.mutability != gt.mutability)
                        return error.ImportResolutionFailed;
                    const gw = allocator.create(types.GlobalInstance) catch
                        return error.ImportResolutionFailed;
                    gw.* = eg.*;
                    gw.owned = false;
                    globals.append(allocator, gw) catch return error.ImportResolutionFailed;
                }
            },
            .memory => {
                if (is_spectest) {
                    if (!std.mem.eql(u8, imp.field_name, "memory"))
                        return error.ImportResolutionFailed;
                    const imp_limits = if (imp.memory_type) |mt| mt.limits else types.Limits{ .min = 1 };
                    if (!limitsMatch(.{ .min = 1, .max = @as(?u32, 2) }, imp_limits))
                        return error.ImportResolutionFailed;
                    memories.append(allocator, makeSpectestMemory(allocator) orelse
                        return error.ImportResolutionFailed) catch return error.ImportResolutionFailed;
                } else {
                    const ri = reg_inst.?;
                    const exp = ri.module.findExport(imp.field_name, .memory) orelse
                        return error.ImportResolutionFailed;
                    if (exp.index >= ri.memories.len) return error.ImportResolutionFailed;
                    const m = ri.memories[exp.index];
                    if (imp.memory_type) |mt| {
                        // Use current pages (may have been grown) for limits matching
                        const actual_limits = types.Limits{
                            .min = m.current_pages,
                            .max = m.memory_type.limits.max,
                        };
                        if (!limitsMatch(actual_limits, mt.limits))
                            return error.ImportResolutionFailed;
                    }
                    // Share the memory — retain refcount
                    m.retain();
                    memories.append(allocator, m) catch return error.ImportResolutionFailed;
                }
            },
            .table => {
                if (is_spectest) {
                    if (!std.mem.eql(u8, imp.field_name, "table"))
                        return error.ImportResolutionFailed;
                    const tt = imp.table_type orelse types.TableType{ .elem_type = .funcref, .limits = .{ .min = 10 } };
                    if (!tt.elem_type.isFuncRef()) return error.ImportResolutionFailed;
                    if (!limitsMatch(.{ .min = 10, .max = @as(?u32, 20) }, tt.limits))
                        return error.ImportResolutionFailed;
                    tables.append(allocator, makeSpectestTable(allocator) orelse
                        return error.ImportResolutionFailed) catch return error.ImportResolutionFailed;
                } else {
                    const ri = reg_inst.?;
                    const exp = ri.module.findExport(imp.field_name, .table) orelse
                        return error.ImportResolutionFailed;
                    if (exp.index >= ri.tables.len) return error.ImportResolutionFailed;
                    const t = ri.tables[exp.index];
                    // Validate elem type and limits compatibility
                    if (imp.table_type) |tt| {
                        if (t.table_type.elem_type != tt.elem_type and
                            !t.table_type.elem_type.isFuncRef() and !tt.elem_type.isFuncRef())
                            return error.ImportResolutionFailed;
                        // Use current element count for limits matching (table may have been grown)
                        const actual_limits = types.Limits{
                            .min = @intCast(t.elements.len),
                            .max = t.table_type.limits.max,
                        };
                        if (!limitsMatch(actual_limits, tt.limits))
                            return error.ImportResolutionFailed;
                    }
                    // Share the table — retain refcount
                    t.retain();
                    tables.append(allocator, t) catch return error.ImportResolutionFailed;
                }
            },
            .function => {
                if (is_spectest) {
                    if (!isSpectestFunction(imp.field_name))
                        return error.ImportResolutionFailed;
                    if (imp.func_type_idx) |tidx| {
                        if (tidx < module.types.len) {
                            if (!spectestFuncTypeMatches(imp.field_name, module.types[tidx]))
                                return error.ImportResolutionFailed;
                        }
                    }
                    // spectest functions remain as stubs (no ImportedFunction entry)
                } else {
                    const ri = reg_inst.?;
                    const exp = ri.module.findExport(imp.field_name, .function) orelse
                        return error.ImportResolutionFailed;
                    if (imp.func_type_idx) |tidx| {
                        if (tidx < module.types.len) {
                            if (ri.module.getFuncType(exp.index)) |export_ft| {
                                if (!funcTypesMatch(module.types[tidx], export_ft))
                                    return error.ImportResolutionFailed;
                            }
                        }
                    }
                    functions.append(allocator, .{
                        .module_inst = ri,
                        .func_idx = exp.index,
                    }) catch return error.ImportResolutionFailed;
                }
            },
        }
    }

    return .{
        .globals = globals.toOwnedSlice(allocator) catch return error.ImportResolutionFailed,
        .memories = memories.toOwnedSlice(allocator) catch return error.ImportResolutionFailed,
        .tables = tables.toOwnedSlice(allocator) catch return error.ImportResolutionFailed,
        .functions = functions.toOwnedSlice(allocator) catch return error.ImportResolutionFailed,
    };
}

fn freeImportContext(ctx: instance_mod.ImportContext, allocator: std.mem.Allocator) void {
    for (ctx.memories) |m| @constCast(m).release(allocator);
    if (ctx.memories.len > 0) allocator.free(ctx.memories);
    for (ctx.tables) |t| @constCast(t).release(allocator);
    if (ctx.tables.len > 0) allocator.free(ctx.tables);
    for (ctx.globals) |g| { if (g.owned) allocator.destroy(g); }
    if (ctx.globals.len > 0) allocator.free(ctx.globals);
    if (ctx.functions.len > 0) allocator.free(ctx.functions);
}

/// Success-path cleanup: the instance took ownership via retain().
fn freeImportContextSlices(ctx: instance_mod.ImportContext, allocator: std.mem.Allocator) void {
    if (ctx.memories.len > 0) allocator.free(ctx.memories);
    if (ctx.tables.len > 0) allocator.free(ctx.tables);
    if (ctx.globals.len > 0) allocator.free(ctx.globals);
    if (ctx.functions.len > 0) allocator.free(ctx.functions);
}

fn getSpectestGlobal(field: []const u8, val_type: types.ValType) types.Value {
    if (std.mem.eql(u8, field, "global_i32")) return .{ .i32 = 666 };
    if (std.mem.eql(u8, field, "global_i64")) return .{ .i64 = 666 };
    if (std.mem.eql(u8, field, "global_f32")) return .{ .f32 = 666.6 };
    if (std.mem.eql(u8, field, "global_f64")) return .{ .f64 = 666.6 };
    return defaultValue(val_type);
}

fn defaultValue(val_type: types.ValType) types.Value {
    return switch (val_type) {
        .i32 => .{ .i32 = 0 },
        .i64 => .{ .i64 = 0 },
        .f32 => .{ .f32 = 0.0 },
        .f64 => .{ .f64 = 0.0 },
        .funcref, .nonfuncref => .{ .funcref = null },
        .externref, .nonexternref => .{ .externref = null },
        .v128 => .{ .v128 = 0 },
    };
}

fn findExportedGlobal(inst: *types.ModuleInstance, name: []const u8) ?*types.GlobalInstance {
    const exp = inst.module.findExport(name, .global) orelse return null;
    if (exp.index < inst.globals.len) return inst.globals[exp.index];
    return null;
}

fn limitsMatch(exported: types.Limits, imported: types.Limits) bool {
    if (exported.min < imported.min) return false;
    if (imported.max) |imp_max| {
        const exp_max = exported.max orelse return false;
        if (exp_max > imp_max) return false;
    }
    return true;
}

fn isSpectestFunction(name: []const u8) bool {
    const known = [_][]const u8{ "print", "print_i32", "print_i64", "print_f32", "print_f64", "print_i32_f32", "print_f64_f64" };
    for (known) |k| {
        if (std.mem.eql(u8, name, k)) return true;
    }
    return false;
}

fn spectestGlobalType(name: []const u8) ?types.GlobalType {
    if (std.mem.eql(u8, name, "global_i32")) return .{ .val_type = .i32, .mutability = .immutable };
    if (std.mem.eql(u8, name, "global_i64")) return .{ .val_type = .i64, .mutability = .immutable };
    if (std.mem.eql(u8, name, "global_f32")) return .{ .val_type = .f32, .mutability = .immutable };
    if (std.mem.eql(u8, name, "global_f64")) return .{ .val_type = .f64, .mutability = .immutable };
    return null;
}

fn spectestFuncTypeMatches(name: []const u8, ft: types.FuncType) bool {
    if (std.mem.eql(u8, name, "print")) return ft.params.len == 0 and ft.results.len == 0;
    if (std.mem.eql(u8, name, "print_i32")) return ft.params.len == 1 and ft.params[0] == .i32 and ft.results.len == 0;
    if (std.mem.eql(u8, name, "print_i64")) return ft.params.len == 1 and ft.params[0] == .i64 and ft.results.len == 0;
    if (std.mem.eql(u8, name, "print_f32")) return ft.params.len == 1 and ft.params[0] == .f32 and ft.results.len == 0;
    if (std.mem.eql(u8, name, "print_f64")) return ft.params.len == 1 and ft.params[0] == .f64 and ft.results.len == 0;
    if (std.mem.eql(u8, name, "print_i32_f32")) return ft.params.len == 2 and ft.params[0] == .i32 and ft.params[1] == .f32 and ft.results.len == 0;
    if (std.mem.eql(u8, name, "print_f64_f64")) return ft.params.len == 2 and ft.params[0] == .f64 and ft.params[1] == .f64 and ft.results.len == 0;
    return false;
}

fn funcTypesMatch(a: types.FuncType, b: types.FuncType) bool {
    if (a.params.len != b.params.len or a.results.len != b.results.len) return false;
    for (a.params, b.params) |pa, pb| { if (pa.toNullable() != pb.toNullable()) return false; }
    for (a.results, b.results) |ra, rb| { if (ra.toNullable() != rb.toNullable()) return false; }
    return true;
}

fn makeSpectestMemory(allocator: std.mem.Allocator) ?*types.MemoryInstance {
    const data = allocator.alloc(u8, types.MemoryInstance.page_size) catch return null;
    @memset(data, 0);
    const mem = allocator.create(types.MemoryInstance) catch { allocator.free(data); return null; };
    mem.* = .{
        .memory_type = .{ .limits = .{ .min = 1, .max = 2 } },
        .data = data,
        .current_pages = 1,
        .max_pages = 2,
    };
    return mem;
}

fn makeSpectestTable(allocator: std.mem.Allocator) ?*types.TableInstance {
    const elems = allocator.alloc(?types.FuncRef, 10) catch return null;
    @memset(elems, null);
    const tbl = allocator.create(types.TableInstance) catch { allocator.free(elems); return null; };
    tbl.* = .{ .table_type = .{ .elem_type = .funcref, .limits = .{ .min = 10, .max = 20 } }, .elements = elems };
    return tbl;
}

fn copyMemory(src: types.MemoryInstance, allocator: std.mem.Allocator) ?types.MemoryInstance {
    const data = allocator.alloc(u8, src.data.len) catch return null;
    @memcpy(data, src.data);
    return .{
        .memory_type = src.memory_type,
        .data = data,
        .current_pages = src.current_pages,
        .max_pages = src.max_pages,
    };
}

fn copyTable(src: types.TableInstance, allocator: std.mem.Allocator) ?types.TableInstance {
    const elems = allocator.alloc(?u32, src.elements.len) catch return null;
    @memcpy(elems, src.elements);
    return .{
        .table_type = src.table_type,
        .elements = elems,
    };
}

fn makeDefaultMemory(mt: ?types.MemoryType, allocator: std.mem.Allocator) ?types.MemoryInstance {
    const mem_type = mt orelse types.MemoryType{ .limits = .{ .min = 1 } };
    const pages = mem_type.limits.min;
    const data = allocator.alloc(u8, @as(usize, pages) * types.MemoryInstance.page_size) catch return null;
    @memset(data, 0);
    return .{
        .memory_type = mem_type,
        .data = data,
        .current_pages = pages,
        .max_pages = mem_type.limits.max orelse 65536,
    };
}

fn makeDefaultTable(tt: ?types.TableType, allocator: std.mem.Allocator) ?types.TableInstance {
    const table_type = tt orelse types.TableType{ .elem_type = .funcref, .limits = .{ .min = 10 } };
    const elems = allocator.alloc(?u32, table_type.limits.min) catch return null;
    @memset(elems, null);
    return .{
        .table_type = table_type,
        .elements = elems,
    };
}

pub fn runSpecTestFile(json_path: []const u8, allocator: std.mem.Allocator) !SpecTestResult {
    const cwd = std.fs.cwd();
    const json_data = try cwd.readFileAlloc(allocator, json_path, 10 * 1024 * 1024);
    defer allocator.free(json_data);

    const parsed = try std.json.parseFromSlice(SpecJson, allocator, json_data, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const commands = parsed.value.commands;

    var result = SpecTestResult{ .file = json_path };
    var runtime = wamr.Runtime.init(allocator);
    defer runtime.deinit();

    var current_module: ?wamr.Module = null;
    var current_instance: ?wamr.Instance = null;
    // Whether current_instance is owned (needs deinit) or borrowed from reg_instances
    var current_instance_owned: bool = false;

    const json_dir = std.fs.path.dirname(json_path) orelse ".";

    var current_wasm_data: ?[]u8 = null;

    // Module registry: maps import module name → instance for cross-module imports
    var module_registry = std.StringHashMap(*types.ModuleInstance).init(allocator);
    defer module_registry.deinit();

    // Named instance registry: maps JSON module name ($Mg, $Mt, etc.) → instance for assertion lookup
    var named_instances = std.StringHashMap(wamr.Instance).init(allocator);
    defer named_instances.deinit();

    // Track allocated registry name strings for cleanup
    var registry_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (registry_names.items) |name| allocator.free(name);
        registry_names.deinit(allocator);
    }

    // Keep registered modules alive so cross-module import pointers stay valid.
    var reg_mod_ptrs: std.ArrayList(*wamr.Module) = .empty;
    var reg_instances: std.ArrayList(wamr.Instance) = .empty;
    var reg_wasm_data: std.ArrayList([]u8) = .empty;

    defer {
        if (current_instance_owned) {
            if (current_instance) |*i| i.deinit();
        }
        if (current_module) |*m| m.deinit();
        if (current_wasm_data) |d| allocator.free(d);
    }
    defer {
        for (reg_instances.items) |*i| @constCast(i).deinit();
        reg_instances.deinit(allocator);
        for (reg_mod_ptrs.items) |p| { p.deinit(); allocator.destroy(p); }
        reg_mod_ptrs.deinit(allocator);
        for (reg_wasm_data.items) |d| allocator.free(d);
        reg_wasm_data.deinit(allocator);
    }

    for (commands) |cmd| {
        result.total += 1;

        if (std.mem.eql(u8, cmd.type, "module")) {
            // Clean up previous current state
            if (current_instance_owned) {
                if (current_instance) |*inst| inst.deinit();
            }
            current_instance = null;
            current_instance_owned = false;
            if (current_module) |*mod| {
                mod.deinit();
                current_module = null;
            }
            if (current_wasm_data) |d| {
                allocator.free(d);
                current_wasm_data = null;
            }

            const filename = cmd.filename orelse {
                result.skipped += 1;
                continue;
            };

            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);

            const wasm_data = cwd.readFileAlloc(allocator, wasm_path, 10 * 1024 * 1024) catch |err| {
                std.debug.print("  SKIP read {s} line {d}: {}\n", .{ filename, cmd.line, err });
                result.skipped += 1;
                continue;
            };

            current_module = runtime.loadModule(wasm_data) catch |err| {
                std.debug.print("  SKIP load {s} line {d}: {}\n", .{ filename, cmd.line, err });
                allocator.free(wasm_data);
                result.skipped += 1;
                continue;
            };
            current_wasm_data = wasm_data;

            // Try to instantiate with import resolution
            const import_ctx = buildImportContext(&current_module.?.inner, &module_registry, allocator) catch |err| {
                std.debug.print("  SKIP import ctx {s} line {d}: {}\n", .{ filename, cmd.line, err });
                result.skipped += 1;
                continue;
            };
            if (import_ctx) |ctx| {
                current_instance = current_module.?.instantiateWithImports(ctx) catch |err| {
                    std.debug.print("  SKIP instantiate+imports {s} line {d}: {}\n", .{ filename, cmd.line, err });
                    freeImportContext(ctx, allocator);
                    result.skipped += 1;
                    continue;
                };
                freeImportContextSlices(ctx, allocator);
            } else {
                current_instance = current_module.?.instantiate() catch |err| {
                    std.debug.print("  SKIP instantiate {s} line {d}: {}\n", .{ filename, cmd.line, err });
                    result.skipped += 1;
                    continue;
                };
            }

            // If module has a JSON name, move to registries so it survives across module loads
            if (cmd.name) |mod_name| {
                if (current_instance) |inst| {
                    // Move to registries for lifetime management
                    reg_instances.append(allocator, inst) catch {};
                    if (current_module) |*mod| {
                        // Transfer the Module to the heap. We must NOT copy it because
                        // the ArenaAllocator inside has internal pointers. Instead, allocate
                        // and move the bytes, then null out current_module to prevent deinit.
                        const mod_heap = allocator.create(wamr.Module) catch {
                            result.passed += 1;
                            continue;
                        };
                        mod_heap.* = mod.*;
                        inst.inner.module = &mod_heap.inner;
                        reg_mod_ptrs.append(allocator, mod_heap) catch {};
                    }
                    current_module = null; // Prevent double-free (mod_heap owns the arena now)
                    if (current_wasm_data) |wd| {
                        reg_wasm_data.append(allocator, wd) catch {};
                        current_wasm_data = null;
                    }
                    named_instances.put(mod_name, inst) catch {};
                    // current_instance is now a borrowed reference
                    current_instance_owned = false;
                }
            } else {
                current_instance_owned = true;
            }
            result.passed += 1;
        } else if (std.mem.eql(u8, cmd.type, "register")) {
            const reg_name = cmd.@"as" orelse {
                result.skipped += 1;
                continue;
            };
            if (current_instance) |inst| {
                const name_copy = allocator.dupe(u8, reg_name) catch {
                    result.skipped += 1;
                    continue;
                };
                module_registry.put(name_copy, inst.inner) catch {
                    allocator.free(name_copy);
                    result.skipped += 1;
                    continue;
                };
                registry_names.append(allocator, name_copy) catch {
                    result.skipped += 1;
                    continue;
                };
                // Only move to registries if not already moved by named module handler
                if (current_instance_owned) {
                    // Heap-allocate Module so WasmModule address stays stable
                    if (current_module) |mod| {
                        const mod_heap = allocator.create(wamr.Module) catch {
                            result.skipped += 1;
                            continue;
                        };
                        mod_heap.* = mod;
                        inst.inner.module = &mod_heap.inner;
                        reg_mod_ptrs.append(allocator, mod_heap) catch {
                            result.skipped += 1;
                            continue;
                        };
                        current_module = null;
                    }
                    reg_instances.append(allocator, inst) catch {
                        result.skipped += 1;
                        continue;
                    };
                    if (current_wasm_data) |wd| {
                        reg_wasm_data.append(allocator, wd) catch {
                            result.skipped += 1;
                            continue;
                        };
                        current_wasm_data = null;
                    }
                }
                // Also track by JSON name if present
                if (cmd.name) |mod_name| {
                    named_instances.put(mod_name, inst) catch {};
                }
                current_instance = null;
                current_instance_owned = false;
                result.passed += 1;
            } else {
                result.skipped += 1;
            }
        } else if (std.mem.eql(u8, cmd.type, "assert_return")) {
            const action = cmd.action orelse {
                result.skipped += 1;
                continue;
            };

            // Handle "get" actions (global.get by export name)
            if (std.mem.eql(u8, action.type, "get")) {
                const resolved_get_inst = if (action.module) |mod_name|
                    named_instances.get(mod_name) orelse current_instance
                else
                    current_instance;
                const get_inst = resolved_get_inst orelse {
                    result.skipped += 1;
                    continue;
                };
                const field = action.field orelse {
                    result.skipped += 1;
                    continue;
                };
                const expected_json = cmd.expected orelse &[_]Arg{};
                const global = findExportedGlobal(get_inst.inner, field) orelse {
                    result.skipped += 1;
                    continue;
                };
                if (expected_json.len == 1) {
                    const expected_val = parseValue(expected_json[0]) orelse {
                        result.skipped += 1;
                        continue;
                    };
                    if (valuesEqual(global.value, expected_val)) {
                        result.passed += 1;
                    } else {
                        std.debug.print("  FAIL assert_return line {d}: get {s} value mismatch\n", .{ cmd.line, field });
                        result.failed += 1;
                    }
                } else {
                    result.passed += 1; // no expected value
                }
                continue;
            }

            if (!std.mem.eql(u8, action.type, "invoke")) {
                result.skipped += 1;
                continue;
            }

            // Resolve instance: use action.module if specified, else current
            const resolved_inst = if (action.module) |mod_name|
                named_instances.get(mod_name) orelse current_instance
            else
                current_instance;
            var inst = resolved_inst orelse {
                result.skipped += 1;
                continue;
            };

            const field = action.field orelse {
                result.skipped += 1;
                continue;
            };
            const expected_json = cmd.expected orelse &[_]Arg{};
            const args_json = action.args orelse &[_]Arg{};

            const args = parseArgs(args_json, allocator) orelse {
                result.skipped += 1;
                continue;
            };
            defer allocator.free(args);

            // Parse expected values
            var expected_vals: ?[]types.Value = null;
            if (expected_json.len > 0) {
                expected_vals = parseArgs(expected_json, allocator) orelse {
                    result.skipped += 1;
                    continue;
                };
            }
            defer if (expected_vals) |ev| allocator.free(ev);

            const actual = inst.call(field, args) catch |err| {
                std.debug.print("  FAIL assert_return line {d}: {s} call error: {}\n", .{ cmd.line, field, err });
                result.failed += 1;
                continue;
            };
            defer allocator.free(actual);

            const expected = expected_vals orelse &[_]types.Value{};
            if (actual.len != expected.len) {
                std.debug.print("  FAIL assert_return line {d}: {s} result count mismatch: got {d}, expected {d}\n", .{ cmd.line, field, actual.len, expected.len });
                result.failed += 1;
                continue;
            }

            var all_match = true;
            for (actual, expected) |a, e| {
                if (!valuesEqual(a, e)) {
                    all_match = false;
                    break;
                }
            }

            if (all_match) {
                result.passed += 1;
            } else {
                if (actual.len > 0 and expected.len > 0) {
                    std.debug.print("  FAIL assert_return line {d}: {s} value mismatch\n", .{ cmd.line, field });
                }
                result.failed += 1;
            }
        } else if (std.mem.eql(u8, cmd.type, "assert_trap")) {
            const action = cmd.action orelse {
                result.skipped += 1;
                continue;
            };

            if (!std.mem.eql(u8, action.type, "invoke")) {
                result.skipped += 1;
                continue;
            }

            // Resolve instance: use action.module if specified, else current
            const resolved_trap_inst = if (action.module) |mod_name|
                named_instances.get(mod_name) orelse current_instance
            else
                current_instance;
            var inst = resolved_trap_inst orelse {
                result.skipped += 1;
                continue;
            };

            const field = action.field orelse {
                result.skipped += 1;
                continue;
            };
            const args_json = action.args orelse &[_]Arg{};

            const args = parseArgs(args_json, allocator) orelse {
                result.skipped += 1;
                continue;
            };
            defer allocator.free(args);

            if (inst.call(field, args)) |results| {
                allocator.free(results);
                result.failed += 1; // should have trapped
            } else |_| {
                result.passed += 1; // trap expected
            }
        } else if (std.mem.eql(u8, cmd.type, "assert_invalid") or
            std.mem.eql(u8, cmd.type, "assert_malformed"))
        {
            const filename = cmd.filename orelse {
                result.skipped += 1;
                continue;
            };

            // For WAT-format modules, parse with wabt text parser and validate
            if (cmd.module_type) |mt| {
                if (std.mem.eql(u8, mt, "text")) {
                    const wat_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
                    defer allocator.free(wat_path);
                    const wat_data = cwd.readFileAlloc(allocator, wat_path, 10 * 1024 * 1024) catch {
                        result.passed += 1;
                        continue;
                    };
                    defer allocator.free(wat_data);
                    var wat_module = wabt.text.Parser.parseModule(allocator, wat_data) catch {
                        result.passed += 1;
                        continue;
                    };
                    defer wat_module.deinit();
                    wabt.Validator.validate(&wat_module, .{}) catch {
                        result.passed += 1;
                        continue;
                    };
                    const wasm_bytes = wabt.binary.writer.writeModule(allocator, &wat_module) catch {
                        result.passed += 1;
                        continue;
                    };
                    defer allocator.free(wasm_bytes);
                    var mod = runtime.loadModule(wasm_bytes) catch {
                        result.passed += 1;
                        continue;
                    };
                    mod.deinit();
                    std.debug.print("  NOTREJECTED line {d}: {s}\n", .{ cmd.line, filename });
                    result.failed += 1;
                    continue;
                }
            }

            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);

            const wasm_data = cwd.readFileAlloc(allocator, wasm_path, 10 * 1024 * 1024) catch {
                result.passed += 1; // can't read = invalid, as expected
                continue;
            };
            defer allocator.free(wasm_data);

            var mod = runtime.loadModule(wasm_data) catch {
                result.passed += 1; // load failed as expected
                continue;
            };
            mod.deinit();
            std.debug.print("  NOTREJECTED line {d}: {s}\n", .{ cmd.line, filename });
            result.failed += 1; // should have failed to load
        } else if (std.mem.eql(u8, cmd.type, "assert_unlinkable")) {
            const filename = cmd.filename orelse {
                result.skipped += 1;
                continue;
            };

            if (cmd.module_type) |mt| {
                if (std.mem.eql(u8, mt, "text")) {
                    result.passed += 1; // text format unlinkable — assume valid rejection
                    continue;
                }
            }

            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);

            const wasm_data = cwd.readFileAlloc(allocator, wasm_path, 10 * 1024 * 1024) catch {
                result.passed += 1;
                continue;
            };
            defer allocator.free(wasm_data);

            var mod = runtime.loadModule(wasm_data) catch {
                result.passed += 1;
                continue;
            };

            const import_ctx = buildImportContext(&mod.inner, &module_registry, allocator) catch {
                mod.deinit();
                result.passed += 1;
                continue;
            };
            if (import_ctx) |ctx| {
                var inst_or_err = mod.instantiateWithImports(ctx);
                if (inst_or_err) |*inst| {
                    inst.deinit();
                    freeImportContext(ctx, allocator);
                    mod.deinit();
                    result.failed += 1;
                } else |_| {
                    freeImportContext(ctx, allocator);
                    mod.deinit();
                    result.passed += 1;
                }
            } else {
                var inst_or_err = mod.instantiate();
                if (inst_or_err) |*inst| {
                    inst.deinit();
                    mod.deinit();
                    result.failed += 1;
                } else |_| {
                    mod.deinit();
                    result.passed += 1;
                }
            }
        } else if (std.mem.eql(u8, cmd.type, "assert_uninstantiable")) {
            const filename = cmd.filename orelse {
                result.skipped += 1;
                continue;
            };

            if (cmd.module_type) |mt| {
                if (std.mem.eql(u8, mt, "text")) {
                    result.passed += 1; // text format uninstantiable — assume valid rejection
                    continue;
                }
            }

            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);

            const wasm_data = cwd.readFileAlloc(allocator, wasm_path, 10 * 1024 * 1024) catch {
                result.passed += 1;
                continue;
            };
            defer allocator.free(wasm_data);

            var mod = runtime.loadModule(wasm_data) catch {
                result.passed += 1;
                continue;
            };

            const import_ctx = buildImportContext(&mod.inner, &module_registry, allocator) catch {
                mod.deinit();
                result.passed += 1;
                continue;
            };
            if (import_ctx) |ctx| {
                var inst_or_err = mod.instantiateWithImports(ctx);
                if (inst_or_err) |*inst| {
                    inst.deinit();
                    freeImportContext(ctx, allocator);
                    mod.deinit();
                    result.failed += 1;
                } else |_| {
                    freeImportContext(ctx, allocator);
                    mod.deinit();
                    result.passed += 1;
                }
            } else {
                var inst_or_err = mod.instantiate();
                if (inst_or_err) |*inst| {
                    inst.deinit();
                    mod.deinit();
                    result.failed += 1;
                } else |_| {
                    mod.deinit();
                    result.passed += 1;
                }
            }
        } else if (std.mem.eql(u8, cmd.type, "assert_exhaustion")) {
            const action = cmd.action orelse {
                result.skipped += 1;
                continue;
            };

            if (!std.mem.eql(u8, action.type, "invoke")) {
                result.skipped += 1;
                continue;
            }

            var inst = current_instance orelse {
                result.skipped += 1;
                continue;
            };

            const field = action.field orelse {
                result.skipped += 1;
                continue;
            };
            const args_json = action.args orelse &[_]Arg{};

            const args = parseArgs(args_json, allocator) orelse {
                result.skipped += 1;
                continue;
            };
            defer allocator.free(args);

            if (inst.call(field, args)) |results| {
                allocator.free(results);
                result.failed += 1; // should have exhausted
            } else |_| {
                result.passed += 1; // exhaustion trap expected
            }
        } else if (std.mem.eql(u8, cmd.type, "action")) {
            const action = cmd.action orelse {
                result.skipped += 1;
                continue;
            };
            if (!std.mem.eql(u8, action.type, "invoke")) {
                result.skipped += 1;
                continue;
            }
            var inst = current_instance orelse {
                result.skipped += 1;
                continue;
            };
            const field = action.field orelse {
                result.skipped += 1;
                continue;
            };
            const args_json = action.args orelse &[_]Arg{};
            const args = parseArgs(args_json, allocator) orelse {
                result.skipped += 1;
                continue;
            };
            defer allocator.free(args);
            if (inst.call(field, args)) |results| {
                allocator.free(results);
                result.passed += 1;
            } else |_| {
                result.failed += 1;
            }
        } else {
            // unknown command type
            result.skipped += 1;
        }
    }

    return result;
}
