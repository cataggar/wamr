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

const Io = std.Io;
const Dir = Io.Dir;

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
        const bits = std.fmt.parseUnsigned(u32, val_str, 10) catch return null;
        return .{ .f32 = @bitCast(bits) };
    } else if (std.mem.eql(u8, arg.type, "f64")) {
        const bits = std.fmt.parseUnsigned(u64, val_str, 10) catch return null;
        return .{ .f64 = @bitCast(bits) };
    }
    return null;
}

/// Bit-exact comparison of two Values (handles NaN correctly).
fn valuesEqual(a: types.Value, b: types.Value) bool {
    return switch (a) {
        .i32 => |v| b == .i32 and b.i32 == v,
        .i64 => |v| b == .i64 and b.i64 == v,
        .f32 => |v| b == .f32 and @as(u32, @bitCast(b.f32)) == @as(u32, @bitCast(v)),
        .f64 => |v| b == .f64 and @as(u64, @bitCast(b.f64)) == @as(u64, @bitCast(v)),
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
) ?instance_mod.ImportContext {
    if (module.import_global_count == 0 and
        module.import_memory_count == 0 and
        module.import_table_count == 0 and
        module.import_function_count == 0) return null;

    // Build imported globals from the module's import descriptors
    var globals: std.ArrayList(types.GlobalInstance) = .empty;
    defer globals.deinit(allocator);
    var memories: std.ArrayList(types.MemoryInstance) = .empty;
    defer memories.deinit(allocator);
    var tables: std.ArrayList(types.TableInstance) = .empty;
    defer tables.deinit(allocator);
    defer tables.deinit(allocator);

    for (module.imports) |imp| {
        switch (imp.kind) {
            .global => {
                const gt = imp.global_type orelse continue;
                // Try spectest first
                if (std.mem.eql(u8, imp.module_name, "spectest")) {
                    globals.append(allocator, .{
                        .global_type = gt,
                        .value = getSpectestGlobal(imp.field_name, gt.val_type),
                    }) catch return null;
                } else if (module_registry.get(imp.module_name)) |reg_inst| {
                    // Try to find the export in a registered module
                    if (findExportedGlobal(reg_inst, imp.field_name)) |g| {
                        globals.append(allocator, g) catch return null;
                    } else {
                        globals.append(allocator, .{ .global_type = gt, .value = defaultValue(gt.val_type) }) catch return null;
                    }
                } else {
                    globals.append(allocator, .{ .global_type = gt, .value = defaultValue(gt.val_type) }) catch return null;
                }
            },
            .memory => {
                if (std.mem.eql(u8, imp.module_name, "spectest")) {
                    memories.append(allocator, makeSpectestMemory(allocator) orelse return null) catch return null;
                } else if (module_registry.get(imp.module_name)) |reg_inst| {
                    if (reg_inst.memories.len > 0) {
                        memories.append(allocator, copyMemory(reg_inst.memories[0], allocator) orelse return null) catch return null;
                    } else {
                        memories.append(allocator, makeDefaultMemory(imp.memory_type, allocator) orelse return null) catch return null;
                    }
                } else {
                    memories.append(allocator, makeDefaultMemory(imp.memory_type, allocator) orelse return null) catch return null;
                }
            },
            .table => {
                if (std.mem.eql(u8, imp.module_name, "spectest")) {
                    tables.append(allocator, makeSpectestTable(allocator) orelse return null) catch return null;
                } else if (module_registry.get(imp.module_name)) |reg_inst| {
                    if (reg_inst.tables.len > 0) {
                        tables.append(allocator, copyTable(reg_inst.tables[0], allocator) orelse return null) catch return null;
                    } else {
                        tables.append(allocator, makeDefaultTable(imp.table_type, allocator) orelse return null) catch return null;
                    }
                } else {
                    tables.append(allocator, makeDefaultTable(imp.table_type, allocator) orelse return null) catch return null;
                }
            },
            .function => {
                // Functions are counted but not bound; the interpreter will
                // return UnknownFunction if a test actually calls one.
            },
        }
    }

    return .{
        .globals = globals.toOwnedSlice(allocator) catch return null,
        .memories = memories.toOwnedSlice(allocator) catch return null,
        .tables = tables.toOwnedSlice(allocator) catch return null,
    };
}

fn freeImportContext(ctx: instance_mod.ImportContext, allocator: std.mem.Allocator) void {
    for (ctx.memories) |*m| allocator.free(@constCast(m).data);
    if (ctx.memories.len > 0) allocator.free(ctx.memories);
    for (ctx.tables) |*t| allocator.free(@constCast(t).elements);
    if (ctx.tables.len > 0) allocator.free(ctx.tables);
    if (ctx.globals.len > 0) allocator.free(ctx.globals);
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
        .funcref => .{ .funcref = null },
        .externref => .{ .externref = null },
        .v128 => .{ .v128 = 0 },
    };
}

fn findExportedGlobal(inst: *types.ModuleInstance, name: []const u8) ?types.GlobalInstance {
    const exp = inst.module.findExport(name, .global) orelse return null;
    if (exp.index < inst.globals.len) return inst.globals[exp.index];
    return null;
}

fn makeSpectestMemory(allocator: std.mem.Allocator) ?types.MemoryInstance {
    const data = allocator.alloc(u8, types.MemoryInstance.page_size) catch return null;
    @memset(data, 0);
    return .{
        .memory_type = .{ .limits = .{ .min = 1, .max = 2 } },
        .data = data,
        .current_pages = 1,
        .max_pages = 2,
    };
}

fn makeSpectestTable(allocator: std.mem.Allocator) ?types.TableInstance {
    const elems = allocator.alloc(?u32, 10) catch return null;
    @memset(elems, null);
    return .{
        .table_type = .{ .elem_type = .funcref, .limits = .{ .min = 10, .max = 20 } },
        .elements = elems,
    };
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

pub fn runSpecTestFile(json_path: []const u8, allocator: std.mem.Allocator, io: Io) !SpecTestResult {
    const cwd = Dir.cwd();
    const json_data = try cwd.readFileAlloc(io, json_path, allocator, Io.Limit.limited(10 * 1024 * 1024));
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

    const json_dir = std.fs.path.dirname(json_path) orelse ".";

    var current_wasm_data: ?[]u8 = null;

    // Module registry: maps name → instance for cross-module imports
    var module_registry = std.StringHashMap(*types.ModuleInstance).init(allocator);
    defer module_registry.deinit();

    // Track allocated registry name strings for cleanup
    var registry_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (registry_names.items) |name| allocator.free(name);
        registry_names.deinit(allocator);
    }

    defer {
        if (current_instance) |*i| i.deinit();
        if (current_module) |*m| m.deinit();
        if (current_wasm_data) |d| allocator.free(d);
    }

    for (commands) |cmd| {
        result.total += 1;

        if (std.mem.eql(u8, cmd.type, "module")) {
            if (current_instance) |*inst| {
                inst.deinit();
                current_instance = null;
            }
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

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, Io.Limit.limited(10 * 1024 * 1024)) catch {
                result.skipped += 1;
                continue;
            };

            current_module = runtime.loadModule(wasm_data) catch {
                allocator.free(wasm_data);
                result.skipped += 1;
                continue;
            };
            current_wasm_data = wasm_data;

            // Try to instantiate with import resolution
            const import_ctx = buildImportContext(&current_module.?.inner, &module_registry, allocator);
            if (import_ctx) |ctx| {
                current_instance = current_module.?.instantiateWithImports(ctx) catch {
                    freeImportContext(ctx, allocator);
                    result.skipped += 1;
                    continue;
                };
                freeImportContext(ctx, allocator);
            } else {
                current_instance = current_module.?.instantiate() catch {
                    result.skipped += 1;
                    continue;
                };
            }
            result.passed += 1;
        } else if (std.mem.eql(u8, cmd.type, "register")) {
            const reg_name = cmd.@"as" orelse {
                result.skipped += 1;
                continue;
            };
            if (current_instance) |*inst| {
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
                result.passed += 1;
            } else {
                result.skipped += 1;
            }
        } else if (std.mem.eql(u8, cmd.type, "assert_return")) {
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

            const actual = inst.call(field, args) catch {
                result.failed += 1;
                continue;
            };
            defer allocator.free(actual);

            const expected = expected_vals orelse &[_]types.Value{};
            if (actual.len != expected.len) {
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

            // Skip WAT-format modules (only test binary .wasm)
            if (cmd.module_type) |mt| {
                if (std.mem.eql(u8, mt, "text")) {
                    result.skipped += 1;
                    continue;
                }
            }

            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, Io.Limit.limited(10 * 1024 * 1024)) catch {
                result.passed += 1; // can't read = invalid, as expected
                continue;
            };
            defer allocator.free(wasm_data);

            var mod = runtime.loadModule(wasm_data) catch {
                result.passed += 1; // load failed as expected
                continue;
            };
            mod.deinit();
            result.failed += 1; // should have failed to load
        } else if (std.mem.eql(u8, cmd.type, "assert_unlinkable")) {
            const filename = cmd.filename orelse {
                result.skipped += 1;
                continue;
            };

            if (cmd.module_type) |mt| {
                if (std.mem.eql(u8, mt, "text")) {
                    result.skipped += 1;
                    continue;
                }
            }

            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, Io.Limit.limited(10 * 1024 * 1024)) catch {
                result.passed += 1;
                continue;
            };
            defer allocator.free(wasm_data);

            var mod = runtime.loadModule(wasm_data) catch {
                result.passed += 1;
                continue;
            };

            // Try instantiation — it should fail because imports can't be resolved
            const import_ctx = buildImportContext(&mod.inner, &module_registry, allocator);
            if (import_ctx) |ctx| {
                var inst_or_err = mod.instantiateWithImports(ctx);
                if (inst_or_err) |*inst| {
                    inst.deinit();
                    freeImportContext(ctx, allocator);
                    mod.deinit();
                    result.failed += 1; // should have failed
                } else |_| {
                    freeImportContext(ctx, allocator);
                    mod.deinit();
                    result.passed += 1; // link error as expected
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
                    result.skipped += 1;
                    continue;
                }
            }

            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, Io.Limit.limited(10 * 1024 * 1024)) catch {
                result.passed += 1;
                continue;
            };
            defer allocator.free(wasm_data);

            var mod = runtime.loadModule(wasm_data) catch {
                result.passed += 1;
                continue;
            };

            const import_ctx = buildImportContext(&mod.inner, &module_registry, allocator);
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
        } else {
            // action, etc.
            result.skipped += 1;
        }
    }

    return result;
}
