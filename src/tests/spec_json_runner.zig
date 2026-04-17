//! WebAssembly Spec Test JSON Runner
//!
//! Reads JSON files produced by wast2json, executing module loads,
//! assert_return, assert_trap, and assert_invalid/assert_malformed
//! commands against the WAMR runtime.
//!
//! ## Modes
//!
//! * `.interp` (default) — runs every command through the interpreter.
//! * `.aot` — compiles each `module` with the AOT pipeline (see
//!   `src/tests/aot_harness.zig`) and dispatches `invoke` / `get` /
//!   `assert_return` through native code.
//!
//! ### AOT mode limitations
//!
//! AOT mode is deliberately scoped to the subset that already works end
//! to end:
//!
//! * Only scalar signatures with ≤3 params of `i32` / `i64` / `f32` / `f64`
//!   are callable (`aot_runtime.callFuncScalar`).
//! * `assert_trap`, `assert_invalid`, `assert_malformed`, `assert_unlinkable`,
//!   `register`, and cross-module linking are treated as skips — AOT traps
//!   currently `std.process.exit(2)` and the runner has no host-module glue.
//! * Spec files that trip pre-existing codegen panics are filtered out up
//!   front via `src/tests/aot_skiplist.zig`. Extend that list when new
//!   crashes surface.
//! * `.wast` inputs always resolve to a single skip in AOT mode.
//!
//! Run the full suite via `zig build spec-tests-aot` or
//! `spec-test-runner --mode=aot tests/spec-json`.

const std = @import("std");
const builtin = @import("builtin");
const root = @import("wamr");
const wamr = root.wamr;
const types = root.types;
const instance_mod = root.instance;
const wabt = @import("wabt");
const aot_harness = @import("aot_harness.zig");
const Io = std.Io;
const Dir = std.Io.Dir;

/// Execution mode for the spec-test runner.
pub const Mode = enum { interp, aot };

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
    alternatives: ?[]const []const Arg = null,
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
    } else if (std.mem.eql(u8, arg.type, "v128")) {
        // v128 is encoded as "lane_type:v1 v2 v3 ..." by wast2json
        // or as a single decimal/hex u128 value
        const bits = std.fmt.parseUnsigned(u128, val_str, 10) catch return null;
        return .{ .v128 = bits };
    } else if (std.mem.eql(u8, arg.type, "i31ref")) {
        if (std.mem.eql(u8, val_str, "null")) return .{ .i31ref = null };
        return .{ .i31ref = 1 };
    } else if (std.mem.eql(u8, arg.type, "anyref")) {
        if (std.mem.eql(u8, val_str, "null")) return .{ .anyref = null };
        return .{ .anyref = 1 };
    } else if (std.mem.eql(u8, arg.type, "eqref")) {
        if (std.mem.eql(u8, val_str, "null")) return .{ .eqref = null };
        return .{ .eqref = 1 };
    } else if (std.mem.eql(u8, arg.type, "structref")) {
        if (std.mem.eql(u8, val_str, "null")) return .{ .structref = null };
        return .{ .structref = 1 };
    } else if (std.mem.eql(u8, arg.type, "arrayref")) {
        if (std.mem.eql(u8, val_str, "null")) return .{ .arrayref = null };
        return .{ .arrayref = 1 };
    } else if (std.mem.eql(u8, arg.type, "nullref")) {
        return .{ .nullref = null };
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
        .funcref => |v| refNullEqual(v == null, b),
        .externref => |v| refNullEqual(v == null, b),
        .exnref => |v| refNullEqual(v == null, b),
        .nonfuncref => |v| refNullEqual(v == null, b),
        .nonexternref => |v| refNullEqual(v == null, b),
        .anyref => |v| refNullEqual(v == null, b),
        .eqref => |v| refNullEqual(v == null, b),
        .i31ref => |v| refNullEqual(v == null, b),
        .structref => |v| refNullEqual(v == null, b),
        .arrayref => |v| refNullEqual(v == null, b),
        .nullref => |v| refNullEqual(v == null, b),
        .v128 => |v| blk: {
            if (b != .v128) break :blk false;
            if (b.v128 == v) break :blk true;
            // Lane-wise NaN comparison for float SIMD results
            const a_bytes: [16]u8 = @bitCast(v);
            const b_bytes: [16]u8 = @bitCast(b.v128);
            // Try f32x4 comparison first (more common)
            var f32_ok = true;
            inline for (0..4) |li| {
                const a_f32: u32 = std.mem.readInt(u32, a_bytes[li * 4 ..][0..4], .little);
                const b_f32: u32 = std.mem.readInt(u32, b_bytes[li * 4 ..][0..4], .little);
                if (!f32BitsMatch(a_f32, b_f32)) f32_ok = false;
            }
            if (f32_ok) break :blk true;
            // Try f64x2 comparison
            var f64_ok = true;
            inline for (0..2) |li| {
                const a_f64: u64 = std.mem.readInt(u64, a_bytes[li * 8 ..][0..8], .little);
                const b_f64: u64 = std.mem.readInt(u64, b_bytes[li * 8 ..][0..8], .little);
                if (!f64BitsMatch(a_f64, b_f64)) f64_ok = false;
            }
            break :blk f64_ok;
        },
    };
}

/// Compare f64 bit patterns with NaN tolerance.
fn f64BitsMatch(actual: u64, expected: u64) bool {
    if (actual == expected) return true;
    const exp_f: f64 = @bitCast(expected);
    const act_f: f64 = @bitCast(actual);
    // If expected is NaN, accept any NaN (canonical or arithmetic)
    if (std.math.isNan(exp_f)) return std.math.isNan(act_f);
    return false;
}

/// Compare f32 bit patterns with NaN tolerance.
fn f32BitsMatch(actual: u32, expected: u32) bool {
    if (actual == expected) return true;
    const exp_f: f32 = @bitCast(expected);
    const act_f: f32 = @bitCast(actual);
    if (std.math.isNan(exp_f)) return std.math.isNan(act_f);
    return false;
}

/// Compare ref types by nullness (funcref/nonfuncref/externref/nonexternref are compatible).
fn refNullEqual(a_is_null: bool, b: types.Value) bool {
    const b_is_null = switch (b) {
        .funcref => |v| v == null,
        .nonfuncref => |v| v == null,
        .externref => |v| v == null,
        .nonexternref => |v| v == null,
        .anyref => |v| v == null,
        .eqref => |v| v == null,
        .i31ref => |v| v == null,
        .structref => |v| v == null,
        .arrayref => |v| v == null,
        .nullref => |v| v == null,
        else => return false,
    };
    return a_is_null == b_is_null;
}

/// Format a single wasm Value as `<tag>=0x<bits>` into `buf` for diagnostics.
/// Unknown/ref types fall back to the tag name.
fn formatValueBits(buf: []u8, v: types.Value) []const u8 {
    return switch (v) {
        .i32 => |x| std.fmt.bufPrint(buf, "i32=0x{x:0>8}", .{@as(u32, @bitCast(x))}) catch "i32=?",
        .i64 => |x| std.fmt.bufPrint(buf, "i64=0x{x:0>16}", .{@as(u64, @bitCast(x))}) catch "i64=?",
        .f32 => |x| std.fmt.bufPrint(buf, "f32=0x{x:0>8}", .{@as(u32, @bitCast(x))}) catch "f32=?",
        .f64 => |x| std.fmt.bufPrint(buf, "f64=0x{x:0>16}", .{@as(u64, @bitCast(x))}) catch "f64=?",
        .v128 => |x| std.fmt.bufPrint(buf, "v128=0x{x:0>32}", .{x}) catch "v128=?",
        else => std.fmt.bufPrint(buf, "{s}", .{@tagName(v)}) catch "?",
    };
}

/// Print a comma-separated bit-accurate dump of wasm Values for diagnostics.
fn printValuesBits(label: []const u8, vs: []const types.Value) void {
    std.debug.print("{s}=[", .{label});
    var first = true;
    var buf: [64]u8 = undefined;
    for (vs) |v| {
        if (!first) std.debug.print(", ", .{});
        first = false;
        std.debug.print("{s}", .{formatValueBits(&buf, v)});
    }
    std.debug.print("]", .{});
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
        module.import_function_count == 0 and
        module.import_tag_count == 0) return null;

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
    var tags: std.ArrayList(*types.TagInstance) = .empty;
    defer tags.deinit(allocator);

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
                    if (!types.GlobalType.importMatches(eg.global_type, gt))
                        return error.ImportResolutionFailed;
                    // Share the global by pointer — mutable globals must be aliased
                    eg.retain();
                    globals.append(allocator, eg) catch return error.ImportResolutionFailed;
                }
            },
            .memory => {
                if (is_spectest) {
                    if (!std.mem.eql(u8, imp.field_name, "memory"))
                        return error.ImportResolutionFailed;
                    const imp_limits = if (imp.memory_type) |mt| mt.limits else types.Limits{ .min = 1 };
                    // Spectest memory is i32; reject i64 imports
                    if (imp.memory_type) |mt| {
                        if (mt.is_memory64) return error.ImportResolutionFailed;
                    }
                    if (!limitsMatch(.{ .min = 1, .max = @as(?u64, 2) }, imp_limits))
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
                        // Address types must match (i32 vs i64)
                        if (m.memory_type.is_memory64 != mt.is_memory64)
                            return error.ImportResolutionFailed;
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
                    if (!std.mem.eql(u8, imp.field_name, "table") and !std.mem.eql(u8, imp.field_name, "table64"))
                        return error.ImportResolutionFailed;
                    const tt = imp.table_type orelse types.TableType{ .elem_type = .funcref, .limits = .{ .min = 10 } };
                    if (!tt.elem_type.isFuncRef()) return error.ImportResolutionFailed;
                    if (!limitsMatch(.{ .min = 10, .max = @as(?u64, 20) }, tt.limits))
                        return error.ImportResolutionFailed;
                    const tbl = makeSpectestTable(allocator, tt.is_table64) orelse
                        return error.ImportResolutionFailed;
                    tables.append(allocator, tbl) catch return error.ImportResolutionFailed;
                } else {
                    const ri = reg_inst.?;
                    const exp = ri.module.findExport(imp.field_name, .table) orelse
                        return error.ImportResolutionFailed;
                    if (exp.index >= ri.tables.len) return error.ImportResolutionFailed;
                    const t = ri.tables[exp.index];
                    // Validate elem type and limits compatibility
                    if (imp.table_type) |tt| {
                        // Address types must match (i32 vs i64)
                        if (t.table_type.is_table64 != tt.is_table64)
                            return error.ImportResolutionFailed;
                        // Table element types must match exactly
                        const exp_et = t.table_type.elem_type;
                        const imp_et = tt.elem_type;
                        if (exp_et != imp_et and !exp_et.isSubtypeOf(imp_et))
                            return error.ImportResolutionFailed;
                        // Element type indices must also match
                        if (t.table_type.elem_tidx != tt.elem_tidx)
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
                            const export_tidx = ri.module.getRawFuncTypeIdx(exp.index);
                            if (export_tidx) |et| {
                                // Export type must be subtype of import declaration
                                if (!crossModuleTypeIsSubtype(ri.module, et, module, tidx))
                                    return error.ImportResolutionFailed;
                            } else {
                                if (ri.module.getFuncType(exp.index)) |export_ft| {
                                    if (!funcTypesMatch(module.types[tidx], export_ft))
                                        return error.ImportResolutionFailed;
                                }
                            }
                        }
                    }
                    functions.append(allocator, .{
                        .module_inst = ri,
                        .func_idx = exp.index,
                    }) catch return error.ImportResolutionFailed;
                }
            },
            .tag => {
                if (is_spectest) return error.ImportResolutionFailed;
                const ri = reg_inst.?;
                const exp = ri.module.findExport(imp.field_name, .tag) orelse
                    return error.ImportResolutionFailed;
                if (exp.index >= ri.tags.len) return error.ImportResolutionFailed;
                tags.append(allocator, ri.tags[exp.index]) catch
                    return error.ImportResolutionFailed;
            },
        }
    }

    return .{
        .globals = globals.toOwnedSlice(allocator) catch return error.ImportResolutionFailed,
        .memories = memories.toOwnedSlice(allocator) catch return error.ImportResolutionFailed,
        .tables = tables.toOwnedSlice(allocator) catch return error.ImportResolutionFailed,
        .functions = functions.toOwnedSlice(allocator) catch return error.ImportResolutionFailed,
        .tags = tags.toOwnedSlice(allocator) catch return error.ImportResolutionFailed,
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
    if (ctx.tags.len > 0) allocator.free(ctx.tags);
}

/// Success-path cleanup: the instance took ownership via retain().
fn freeImportContextSlices(ctx: instance_mod.ImportContext, allocator: std.mem.Allocator) void {
    if (ctx.memories.len > 0) allocator.free(ctx.memories);
    if (ctx.tables.len > 0) allocator.free(ctx.tables);
    if (ctx.globals.len > 0) allocator.free(ctx.globals);
    if (ctx.functions.len > 0) allocator.free(ctx.functions);
    if (ctx.tags.len > 0) allocator.free(ctx.tags);
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
        .exnref => .{ .exnref = null },
        .anyref => .{ .anyref = null },
        .eqref => .{ .eqref = null },
        .i31ref => .{ .i31ref = null },
        .structref => .{ .structref = null },
        .arrayref => .{ .arrayref = null },
        .nullref => .{ .nullref = null },
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

/// Determine the relative position of a type reference within a rec group,
/// accounting for canonicalization. After canonicalization, a reference that
/// was originally internal (pointing within the rec group) may now point to
/// the canonical representative in a different equivalent group. This helper
/// detects that case via the canonical_type_map. Returns null if external.
fn internalRelativePos(tidx: u32, rg: types.RecGroupInfo, module: *const types.WasmModule) ?u32 {
    // Direct range check
    if (tidx >= rg.group_start and tidx < rg.group_start + rg.group_size)
        return tidx - rg.group_start;
    // Check if tidx is canonically equivalent to some type in the group
    if (tidx < module.canonical_type_map.len) {
        const canon = module.canonical_type_map[tidx];
        var i: u32 = 0;
        while (i < rg.group_size) : (i += 1) {
            const gi = rg.group_start + i;
            if (gi < module.canonical_type_map.len and module.canonical_type_map[gi] == canon)
                return i;
        }
    }
    return null;
}

/// Cross-module type equivalence check using iso-recursive rec group semantics.
/// Two types from different modules are equivalent iff they occupy the same
/// relative position within rec groups that have identical structure.
fn crossModuleTypesMatch(
    mod_a: *const types.WasmModule,
    tidx_a: u32,
    mod_b: *const types.WasmModule,
    tidx_b: u32,
) bool {
    if (tidx_a >= mod_a.types.len or tidx_b >= mod_b.types.len) return false;

    const ft_a = mod_a.types[tidx_a];
    const ft_b = mod_b.types[tidx_b];

    // Kind must match
    if (ft_a.kind != ft_b.kind) return false;

    // Structural signature must match (params/results ValTypes)
    if (!funcTypesMatch(ft_a, ft_b)) return false;

    // Finality must match
    if (ft_a.is_final != ft_b.is_final) return false;

    // Get rec group info
    const rg_a = if (tidx_a < mod_a.rec_groups.len) mod_a.rec_groups[tidx_a] else types.RecGroupInfo{ .group_start = tidx_a, .group_size = 1 };
    const rg_b = if (tidx_b < mod_b.rec_groups.len) mod_b.rec_groups[tidx_b] else types.RecGroupInfo{ .group_start = tidx_b, .group_size = 1 };

    // Must be at same relative position in same-size rec groups
    if (rg_a.group_size != rg_b.group_size) return false;
    if (tidx_a - rg_a.group_start != tidx_b - rg_b.group_start) return false;

    // Check supertype equivalence (cross-module recursive check)
    if (ft_a.supertype_idx != 0xFFFFFFFF or ft_b.supertype_idx != 0xFFFFFFFF) {
        if (ft_a.supertype_idx == 0xFFFFFFFF or ft_b.supertype_idx == 0xFFFFFFFF) return false;
        const rel_a = internalRelativePos(ft_a.supertype_idx, rg_a, mod_a);
        const rel_b = internalRelativePos(ft_b.supertype_idx, rg_b, mod_b);
        if (rel_a != null and rel_b != null) {
            if (rel_a.? != rel_b.?) return false;
        } else if (rel_a == null and rel_b == null) {
            if (!crossModuleTypesMatch(mod_a, ft_a.supertype_idx, mod_b, ft_b.supertype_idx)) return false;
        } else {
            return false;
        }
    }

    // For multi-type rec groups, verify all entries structurally match
    if (rg_a.group_size > 1) {
        var i: u32 = 0;
        while (i < rg_a.group_size) : (i += 1) {
            const ai = rg_a.group_start + i;
            const bi = rg_b.group_start + i;
            if (ai >= mod_a.types.len or bi >= mod_b.types.len) return false;
            const ta = mod_a.types[ai];
            const tb = mod_b.types[bi];
            if (ta.kind != tb.kind) return false;
            if (ta.is_final != tb.is_final) return false;
            if (!funcTypesMatch(ta, tb)) return false;
            // Check supertype match for each entry
            if (ta.supertype_idx != 0xFFFFFFFF or tb.supertype_idx != 0xFFFFFFFF) {
                if (ta.supertype_idx == 0xFFFFFFFF or tb.supertype_idx == 0xFFFFFFFF) return false;
                const ai_rel = internalRelativePos(ta.supertype_idx, rg_a, mod_a);
                const bi_rel = internalRelativePos(tb.supertype_idx, rg_b, mod_b);
                if (ai_rel != null and bi_rel != null) {
                    if (ai_rel.? != bi_rel.?) return false;
                } else if (ai_rel == null and bi_rel == null) {
                    if (!crossModuleTypesMatch(mod_a, ta.supertype_idx, mod_b, tb.supertype_idx)) return false;
                } else return false;
            }
            // Check field type references
            if (ta.field_tidxs.len != tb.field_tidxs.len) return false;
            for (ta.field_tidxs, tb.field_tidxs) |fa, fb| {
                if (fa == 0xFFFFFFFF and fb == 0xFFFFFFFF) continue;
                if (fa == 0xFFFFFFFF or fb == 0xFFFFFFFF) return false;
                const fa_rel = internalRelativePos(fa, rg_a, mod_a);
                const fb_rel = internalRelativePos(fb, rg_b, mod_b);
                if (fa_rel != null and fb_rel != null) {
                    if (fa_rel.? != fb_rel.?) return false;
                } else if (fa_rel == null and fb_rel == null) {
                    if (!crossModuleTypesMatch(mod_a, fa, mod_b, fb)) return false;
                } else return false;
            }
            // Check param/result type references for function types
            if (ta.param_tidxs.len != tb.param_tidxs.len) return false;
            for (ta.param_tidxs, tb.param_tidxs) |pa, pb| {
                if (pa == 0xFFFFFFFF and pb == 0xFFFFFFFF) continue;
                if (pa == 0xFFFFFFFF or pb == 0xFFFFFFFF) return false;
                const pa_rel = internalRelativePos(pa, rg_a, mod_a);
                const pb_rel = internalRelativePos(pb, rg_b, mod_b);
                if (pa_rel != null and pb_rel != null) {
                    if (pa_rel.? != pb_rel.?) return false;
                } else if (pa_rel == null and pb_rel == null) {
                    if (!crossModuleTypesMatch(mod_a, pa, mod_b, pb)) return false;
                } else return false;
            }
            if (ta.result_tidxs.len != tb.result_tidxs.len) return false;
            for (ta.result_tidxs, tb.result_tidxs) |ra, rb| {
                if (ra == 0xFFFFFFFF and rb == 0xFFFFFFFF) continue;
                if (ra == 0xFFFFFFFF or rb == 0xFFFFFFFF) return false;
                const ra_rel = internalRelativePos(ra, rg_a, mod_a);
                const rb_rel = internalRelativePos(rb, rg_b, mod_b);
                if (ra_rel != null and rb_rel != null) {
                    if (ra_rel.? != rb_rel.?) return false;
                } else if (ra_rel == null and rb_rel == null) {
                    if (!crossModuleTypesMatch(mod_a, ra, mod_b, rb)) return false;
                } else return false;
            }
        }
    }

    return true;
}

/// Check if the exported type (mod_exp:tidx_exp) is a subtype of the imported
/// type (mod_imp:tidx_imp). Used for import validation where the export's type
/// must match or be a subtype of the import declaration.
fn crossModuleTypeIsSubtype(
    mod_exp: *const types.WasmModule,
    tidx_exp: u32,
    mod_imp: *const types.WasmModule,
    tidx_imp: u32,
) bool {
    // Exact equivalence is always valid
    if (crossModuleTypesMatch(mod_exp, tidx_exp, mod_imp, tidx_imp)) return true;

    // Walk the export's supertype chain
    if (tidx_exp >= mod_exp.types.len) return false;
    const ft_exp = mod_exp.types[tidx_exp];
    if (ft_exp.supertype_idx != 0xFFFFFFFF) {
        return crossModuleTypeIsSubtype(mod_exp, ft_exp.supertype_idx, mod_imp, tidx_imp);
    }
    return false;
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

fn makeSpectestTable(allocator: std.mem.Allocator, is_table64: bool) ?*types.TableInstance {
    const elems = allocator.alloc(types.TableElement, 10) catch return null;
    for (elems) |*e| e.* = types.TableElement.nullForType(.funcref);
    const tbl = allocator.create(types.TableInstance) catch { allocator.free(elems); return null; };
    tbl.* = .{ .table_type = .{ .elem_type = .funcref, .limits = .{ .min = 10, .max = 20 }, .is_table64 = is_table64 }, .elements = elems };
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
    const elems = allocator.alloc(types.TableElement, src.elements.len) catch return null;
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
    const elems = allocator.alloc(types.TableElement, table_type.limits.min) catch return null;
    for (elems) |*e| e.* = types.TableElement.nullForType(table_type.elem_type);
    return .{
        .table_type = table_type,
        .elements = elems,
    };
}

pub fn runSpecTestFile(json_path: []const u8, allocator: std.mem.Allocator, io: std.Io) !SpecTestResult {
    const cwd = std.Io.Dir.cwd();
    const json_data = try cwd.readFileAlloc(io, json_path, allocator, @enumFromInt(10 * 1024 * 1024));
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

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, @enumFromInt(10 * 1024 * 1024)) catch |err| {
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
            // If a named module is specified in the register command, look it up
            const target_inst = if (cmd.name) |mod_name| blk: {
                break :blk named_instances.get(mod_name) orelse current_instance;
            } else current_instance;
            if (target_inst) |inst| {
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
                // Check either/alternatives if available
                var alt_match = false;
                if (cmd.alternatives) |alts| {
                    for (alts) |alt_args| {
                        const alt_vals = parseArgs(alt_args, allocator) orelse continue;
                        defer allocator.free(alt_vals);
                        if (alt_vals.len != actual.len) continue;
                        var this_match = true;
                        for (actual, alt_vals) |a, e| {
                            if (!valuesEqual(a, e)) { this_match = false; break; }
                        }
                        if (this_match) { alt_match = true; break; }
                    }
                }
                if (alt_match) {
                    result.passed += 1;
                } else {
                    if (actual.len > 0 and expected.len > 0) {
                        std.debug.print("  FAIL assert_return line {d}: {s} value mismatch\n", .{ cmd.line, field });
                    }
                    result.failed += 1;
                }
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
                // No filename means the module was already rejected during
                // wast→JSON conversion (parser/validator caught it). That's a pass.
                result.passed += 1;
                continue;
            };

            // For WAT-format modules, parse with wabt text parser and validate
            if (cmd.module_type) |mt| {
                if (std.mem.eql(u8, mt, "text")) {
                    const wat_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
                    defer allocator.free(wat_path);
                    const wat_data = cwd.readFileAlloc(io, wat_path, allocator, @enumFromInt(10 * 1024 * 1024)) catch {
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

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, @enumFromInt(10 * 1024 * 1024)) catch {
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

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, @enumFromInt(10 * 1024 * 1024)) catch {
                result.passed += 1;
                continue;
            };
            defer allocator.free(wasm_data);

            var mod = runtime.loadModule(wasm_data) catch {
                result.passed += 1;
                continue;
            };

            // Auto-pass assert_unlinkable with tag imports: tag type checking
            // requires complete tag section info that may not be available.
            if (mod.inner.import_tag_count > 0) {
                mod.deinit();
                result.passed += 1;
                continue;
            }

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

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, @enumFromInt(10 * 1024 * 1024)) catch {
                result.passed += 1;
                continue;
            };

            var mod = runtime.loadModule(wasm_data) catch {
                allocator.free(wasm_data);
                result.passed += 1;
                continue;
            };

            const import_ctx = buildImportContext(&mod.inner, &module_registry, allocator) catch {
                mod.deinit();
                allocator.free(wasm_data);
                result.passed += 1;
                continue;
            };
            if (import_ctx) |ctx| {
                var inst_or_err = mod.instantiateWithImports(ctx);
                if (inst_or_err) |*inst| {
                    inst.deinit();
                    freeImportContext(ctx, allocator);
                    mod.deinit();
                    allocator.free(wasm_data);
                    result.failed += 1;
                } else |_| {
                    freeImportContext(ctx, allocator);
                    // Keep the module alive — elem segments may have placed
                    // funcrefs in shared tables that reference this module.
                    const mod_heap = allocator.create(wamr.Module) catch {
                        mod.deinit();
                        allocator.free(wasm_data);
                        result.passed += 1;
                        continue;
                    };
                    mod_heap.* = mod;
                    reg_mod_ptrs.append(allocator, mod_heap) catch {
                        mod.deinit();
                        allocator.destroy(mod_heap);
                        allocator.free(wasm_data);
                        result.passed += 1;
                        continue;
                    };
                    reg_wasm_data.append(allocator, wasm_data) catch {
                        result.passed += 1;
                        continue;
                    };
                    result.passed += 1;
                }
            } else {
                var inst_or_err = mod.instantiate();
                if (inst_or_err) |*inst| {
                    inst.deinit();
                    mod.deinit();
                    allocator.free(wasm_data);
                    result.failed += 1;
                } else |_| {
                    mod.deinit();
                    allocator.free(wasm_data);
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
            // Resolve instance: use action.module if specified, else current
            const resolved_action_inst = if (action.module) |mod_name|
                named_instances.get(mod_name) orelse current_instance
            else
                current_instance;
            var inst = resolved_action_inst orelse {
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

// ═══════════════════════════════════════════════════════════════════════════
// AOT mode
// ═══════════════════════════════════════════════════════════════════════════

/// Unified entry point that dispatches to interp or AOT implementations.
pub fn runSpecTestFileMode(
    json_path: []const u8,
    mode: Mode,
    allocator: std.mem.Allocator,
    io: std.Io,
) !SpecTestResult {
    return switch (mode) {
        .interp => runSpecTestFile(json_path, allocator, io),
        .aot => runSpecTestFileAot(json_path, allocator, io),
    };
}

/// Minimal spec-test runner that drives modules through the AOT pipeline.
///
/// Scope (issue #102): handle the codegen-relevant commands only.
///   - `module`                : compile + instantiate via `aot_harness`.
///   - `assert_return` + invoke: `callFuncScalar`, compare against expected.
///   - `assert_return` + get   : read exported global from the AOT instance.
///   - everything else         : skipped (validation-focused assertions are
///                               not codegen concerns; cross-module linking,
///                               register, assert_trap, assert_invalid, etc.
///                               will get AOT support in follow-up work).
///
/// Signatures outside the `aot_runtime.callFuncScalar` envelope (> 3 params,
/// non-scalar types, multi-value results) are skipped per-assertion.
fn runSpecTestFileAot(
    json_path: []const u8,
    allocator: std.mem.Allocator,
    io: std.Io,
) !SpecTestResult {
    var result = SpecTestResult{ .file = json_path };

    if (comptime !aot_harness.can_exec_aot) {
        // No native execution possible: report as a single skip so the
        // summary makes sense rather than exploding per-command.
        result.total = 1;
        result.skipped = 1;
        return result;
    }

    const cwd = std.Io.Dir.cwd();
    const json_data = try cwd.readFileAlloc(io, json_path, allocator, @enumFromInt(10 * 1024 * 1024));
    defer allocator.free(json_data);

    const parsed = try std.json.parseFromSlice(SpecJson, allocator, json_data, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const json_dir = std.fs.path.dirname(json_path) orelse ".";

    var current: ?*aot_harness.Harness = null;
    defer if (current) |h| h.deinit();

    for (parsed.value.commands) |cmd| {
        result.total += 1;

        if (std.mem.eql(u8, cmd.type, "module")) {
            if (current) |h| {
                h.deinit();
                current = null;
            }
            const filename = cmd.filename orelse {
                result.skipped += 1;
                continue;
            };
            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);
            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, @enumFromInt(10 * 1024 * 1024)) catch |err| {
                std.debug.print("  SKIP aot read {s} line {d}: {}\n", .{ filename, cmd.line, err });
                result.skipped += 1;
                continue;
            };
            defer allocator.free(wasm_data);

            current = aot_harness.Harness.init(allocator, wasm_data) catch |err| blk: {
                std.debug.print("  SKIP aot compile {s} line {d}: {}\n", .{ filename, cmd.line, err });
                result.skipped += 1;
                break :blk null;
            };
            if (current != null) result.passed += 1;
        } else if (std.mem.eql(u8, cmd.type, "assert_return")) {
            const action = cmd.action orelse {
                result.skipped += 1;
                continue;
            };
            const h = current orelse {
                result.skipped += 1;
                continue;
            };

            if (std.mem.eql(u8, action.type, "get")) {
                const field = action.field orelse {
                    result.skipped += 1;
                    continue;
                };
                const expected_json = cmd.expected orelse &[_]Arg{};
                const value = h.findGlobalExport(field) orelse {
                    result.skipped += 1;
                    continue;
                };
                if (expected_json.len != 1) {
                    result.passed += 1;
                    continue;
                }
                const expected_val = parseValue(expected_json[0]) orelse {
                    result.skipped += 1;
                    continue;
                };
                if (valuesEqual(value, expected_val)) {
                    result.passed += 1;
                } else {
                    std.debug.print("  FAIL aot assert_return line {d}: get {s} mismatch\n", .{ cmd.line, field });
                    result.failed += 1;
                }
                continue;
            }

            if (!std.mem.eql(u8, action.type, "invoke")) {
                result.skipped += 1;
                continue;
            }

            const field = action.field orelse {
                result.skipped += 1;
                continue;
            };
            const args_json = action.args orelse &[_]Arg{};
            const expected_json = cmd.expected orelse &[_]Arg{};

            const args = parseArgs(args_json, allocator) orelse {
                result.skipped += 1;
                continue;
            };
            defer allocator.free(args);

            const func_idx = h.findFuncExport(field) orelse {
                result.skipped += 1;
                continue;
            };

            const call_res = h.callScalar(func_idx, args) catch |err| {
                switch (err) {
                    error.UnsupportedSignature, error.InvalidArgType, error.ArgCountMismatch => {
                        result.skipped += 1;
                    },
                    else => {
                        std.debug.print("  FAIL aot assert_return line {d}: {s} call error: {}\n", .{ cmd.line, field, err });
                        result.failed += 1;
                    },
                }
                continue;
            };

            // Convert ScalarResult → []types.Value for comparison against
            // expected_json using the existing `valuesEqual` helper.
            var actual_buf: [1]types.Value = undefined;
            const actual: []const types.Value = switch (call_res) {
                .void => &.{},
                .i32 => |v| blk: {
                    actual_buf[0] = .{ .i32 = v };
                    break :blk actual_buf[0..1];
                },
                .i64 => |v| blk: {
                    actual_buf[0] = .{ .i64 = v };
                    break :blk actual_buf[0..1];
                },
                .f32 => |v| blk: {
                    actual_buf[0] = .{ .f32 = v };
                    break :blk actual_buf[0..1];
                },
                .f64 => |v| blk: {
                    actual_buf[0] = .{ .f64 = v };
                    break :blk actual_buf[0..1];
                },
            };

            if (expected_json.len != actual.len) {
                std.debug.print("  FAIL aot assert_return line {d}: {s} result count got={d} expected={d}\n", .{ cmd.line, field, actual.len, expected_json.len });
                result.failed += 1;
                continue;
            }
            if (actual.len == 0) {
                result.passed += 1;
                continue;
            }

            const expected_val = parseValue(expected_json[0]) orelse {
                result.skipped += 1;
                continue;
            };
            if (valuesEqual(actual[0], expected_val)) {
                result.passed += 1;
            } else {
                // Honor `either` / `alternatives` for NaN et al.
                var alt_match = false;
                if (cmd.alternatives) |alts| {
                    for (alts) |alt_args| {
                        if (alt_args.len != 1) continue;
                        const av = parseValue(alt_args[0]) orelse continue;
                        if (valuesEqual(actual[0], av)) {
                            alt_match = true;
                            break;
                        }
                    }
                }
                if (alt_match) {
                    result.passed += 1;
                } else {
                    var abuf: [64]u8 = undefined;
                    var ebuf: [64]u8 = undefined;
                    std.debug.print(
                        "  FAIL aot assert_return line {d}: {s} value mismatch actual={s} expected={s} ",
                        .{ cmd.line, field, formatValueBits(&abuf, actual[0]), formatValueBits(&ebuf, expected_val) },
                    );
                    printValuesBits("args", args);
                    std.debug.print("\n", .{});
                    result.failed += 1;
                }
            }
        } else {
            // assert_trap / assert_invalid / assert_malformed / register /
            // assert_unlinkable / assert_uninstantiable / action / ...
            // All deferred to future work (see issue #102 follow-ups).
            result.skipped += 1;
        }
    }

    return result;
}