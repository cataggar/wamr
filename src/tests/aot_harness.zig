//! Reusable in-memory AOT compile+instantiate harness.
//!
//! Wraps the end-to-end AOT pipeline (interpreter loader → IR frontend →
//! IR passes → x86_64/aarch64 codegen → emit_aot → aot_loader →
//! aot_runtime.instantiate → mapCodeExecutable) behind a small typed API
//! usable by test runners.
//!
//! Factored out of `src/tests/differential.zig` so the spec-test runner
//! can drive the same pipeline (see issue #102).

const std = @import("std");
const builtin = @import("builtin");

const root = @import("wamr");
const types = root.types;
const loader_mod = root.loader;
const frontend = root.frontend;
const passes = root.passes;
const x86_64_compile = root.x86_64_compile;
const aarch64_compile = root.aarch64_compile;
const emit_aot = root.emit_aot;
const aot_loader = root.aot_loader;
const aot_runtime = root.aot_runtime;

/// True on targets where the AOT runtime can execute generated code.
pub const can_exec_aot = switch (builtin.cpu.arch) {
    .x86_64, .aarch64 => true,
    else => false,
};

pub const Error = error{
    AotUnsupportedArch,
    CompileFailed,
    LoadFailed,
    InstantiateFailed,
    MapExecutableFailed,
} || aot_runtime.ScalarCallError;

/// A fully instantiated AOT module ready to invoke.
///
/// Owns the interpreter-loaded `WasmModule` (for function-type lookup),
/// the emitted AOT binary, the loaded `AotModule`, and the `AotInstance`.
pub const Harness = struct {
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    wasm_module: types.WasmModule,
    aot_bin: []u8,
    aot_module: aot_loader.AotModule,
    inst: *aot_runtime.AotInstance,

    /// Compile `wasm_bytes` through the full AOT pipeline and instantiate it.
    /// On success the caller owns the harness and must call `deinit` then
    /// free the pointer with the same allocator.
    pub fn init(allocator: std.mem.Allocator, wasm_bytes: []const u8) Error!*Harness {
        if (comptime !can_exec_aot) return error.AotUnsupportedArch;

        const h = allocator.create(Harness) catch return error.OutOfMemory;
        errdefer allocator.destroy(h);

        const arena = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
        errdefer allocator.destroy(arena);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const a = arena.allocator();

        const wasm_module = loader_mod.load(wasm_bytes, a) catch |err| {
            std.debug.print("aot_harness: loader failed: {}\n", .{err});
            return error.CompileFailed;
        };

        const aot_bin = compileToAot(allocator, arena, &wasm_module) catch |err| {
            std.debug.print("aot_harness: compile failed: {}\n", .{err});
            return error.CompileFailed;
        };
        errdefer allocator.free(aot_bin);

        const aot_module = aot_loader.load(aot_bin, allocator) catch return error.LoadFailed;

        h.* = .{
            .allocator = allocator,
            .arena = arena,
            .wasm_module = wasm_module,
            .aot_bin = aot_bin,
            .aot_module = aot_module,
            .inst = undefined,
        };
        errdefer aot_loader.unload(&h.aot_module, allocator);

        // `&h.aot_module` is stable for the lifetime of the heap-allocated
        // Harness, so `inst.module` will remain valid until `deinit`.
        h.inst = aot_runtime.instantiate(&h.aot_module, allocator) catch
            return error.InstantiateFailed;
        errdefer aot_runtime.destroy(h.inst);

        aot_runtime.mapCodeExecutable(h.inst) catch return error.MapExecutableFailed;

        // Invoke the start function, if any. The start function has type
        // `() -> ()` per the wasm spec, so we pass no params/results.
        if (h.aot_module.start_function) |start_idx| {
            const start_ft = h.wasm_module.getFuncType(start_idx);
            if (start_ft != null) {
                _ = aot_runtime.callFuncScalar(
                    h.inst,
                    start_idx,
                    start_ft.?.params,
                    null,
                    &.{},
                ) catch {};
            }
        }

        return h;
    }

    pub fn deinit(self: *Harness) void {
        const allocator = self.allocator;
        aot_runtime.destroy(self.inst);
        aot_loader.unload(&self.aot_module, allocator);
        allocator.free(self.aot_bin);
        self.arena.deinit();
        allocator.destroy(self.arena);
        allocator.destroy(self);
    }

    /// Look up an exported function by name, returning the module-level
    /// function index (same index space as `aot_runtime.findExportFunc`).
    pub fn findFuncExport(self: *const Harness, name: []const u8) ?u32 {
        return aot_runtime.findExportFunc(self.inst, name);
    }

    /// Look up the function type of a module function by index, using the
    /// interpreter-loaded `WasmModule` (the AOT module drops type info).
    pub fn getFuncType(self: *const Harness, func_idx: u32) ?types.FuncType {
        return self.wasm_module.getFuncType(func_idx);
    }

    /// Find an exported global by name and return its current value.
    pub fn findGlobalExport(self: *const Harness, name: []const u8) ?types.Value {
        for (self.wasm_module.exports) |exp| {
            if (exp.kind == .global and std.mem.eql(u8, exp.name, name)) {
                if (exp.index >= self.inst.globals.len) return null;
                return self.inst.globals[exp.index].value;
            }
        }
        return null;
    }

    /// Invoke an AOT function by index with runtime-typed scalar args.
    /// See `aot_runtime.callFuncScalar` for the supported signature envelope.
    pub fn callScalar(
        self: *Harness,
        func_idx: u32,
        args: []const types.Value,
    ) aot_runtime.ScalarCallError!aot_runtime.ScalarResult {
        const ft = self.getFuncType(func_idx) orelse return error.FunctionNotFound;
        const result_type: ?types.ValType = if (ft.results.len == 0)
            null
        else if (ft.results.len == 1)
            ft.results[0]
        else
            return error.UnsupportedSignature;
        return aot_runtime.callFuncScalar(self.inst, func_idx, ft.params, result_type, args);
    }
};

fn compileToAot(
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    module: *const types.WasmModule,
) ![]u8 {
    const a = arena.allocator();

    var ir_module = try frontend.lowerModule(module, a);
    defer ir_module.deinit();

    _ = try passes.runPasses(&ir_module, passes.default_passes, a);

    const code: []const u8, const offsets: []const u32 = switch (builtin.cpu.arch) {
        .aarch64 => blk: {
            const r = try aarch64_compile.compileModule(&ir_module, a);
            break :blk .{ r.code, r.offsets };
        },
        else => blk: {
            const r = try x86_64_compile.compileModule(&ir_module, a);
            break :blk .{ r.code, r.offsets };
        },
    };

    var exports: std.ArrayList(emit_aot.ExportEntry) = .empty;
    for (module.exports) |exp| {
        try exports.append(a, .{
            .name = exp.name,
            .kind = @enumFromInt(@intFromEnum(exp.kind)),
            .index = exp.index,
        });
    }

    // Propagate the wasm module's imports so the loaded AOT module has the
    // correct `import_function_count`. Without this, exported local-function
    // indices (which include imports in their index space) fail to resolve in
    // `getFuncAddr` — e.g. names.3.wasm exports `print32` at func_idx=2 after
    // two spectest imports; skipping imports turns that into out-of-range.
    var import_entries: std.ArrayList(emit_aot.ImportEntry) = .empty;
    for (module.imports) |imp| {
        try import_entries.append(a, .{
            .module_name = imp.module_name,
            .field_name = imp.field_name,
            .kind = @enumFromInt(@intFromEnum(imp.kind)),
            .func_type_idx = imp.func_type_idx orelse 0,
        });
    }

    var mem_entries: std.ArrayList(emit_aot.MemoryEntry) = .empty;
    for (module.memories) |mem| {
        try mem_entries.append(a, .{
            .min_pages = @intCast(mem.limits.min),
            .max_pages = if (mem.limits.max) |m| @as(?u32, @intCast(m)) else null,
        });
    }

    // Build the global entries with wasm-flat indexing: imported globals
    // come first (at wasm indices [0, import_global_count)), then locals.
    // This matches what the x86_64 codegen expects: `global.get N` emits
    // a load at offset N*8 from the globals base, where N is the raw
    // wasm index (imports + locals).
    //
    // For imports we resolve values from the spectest module (the spec
    // suite's canonical host module). Unknown imports fall back to 0.
    //
    // For locals we run `evalInitExpr` against all preceding globals so
    // that `global.get` / arithmetic bytecode inits produce correct
    // values (e.g. global.wast's $z1..$z6 depend on imported globals).
    var global_entries: std.ArrayList(emit_aot.GlobalEntry) = .empty;

    // Temporary GlobalInstance list to feed evalInitExpr.
    var tmp_globals: std.ArrayList(*types.GlobalInstance) = .empty;
    defer {
        for (tmp_globals.items) |g| a.destroy(g);
        tmp_globals.deinit(a);
    }

    // Imports first (only globals; other kinds are skipped).
    for (module.imports) |imp| {
        if (imp.kind != .global) continue;
        const gt = imp.global_type orelse continue;
        const val: types.Value = spectestGlobalInitValue(imp.module_name, imp.field_name, gt.val_type) orelse
            defaultZeroValue(gt.val_type);
        const g = try a.create(types.GlobalInstance);
        g.* = .{ .global_type = gt, .value = val };
        try tmp_globals.append(a, g);
        try global_entries.append(a, .{
            .val_type = @intFromEnum(gt.val_type),
            .mutability = if (gt.mutability == .mutable) @as(u8, 1) else @as(u8, 0),
            .init_i64 = valueToI64(val),
        });
    }

    // Locals: evaluate init expressions against the preceding globals.
    for (module.globals) |g| {
        const val: types.Value = root.instance.evalInitExpr(g.init_expr, tmp_globals.items, null) catch defaultZeroValue(g.global_type.val_type);
        const gi = try a.create(types.GlobalInstance);
        gi.* = .{ .global_type = g.global_type, .value = val };
        try tmp_globals.append(a, gi);
        try global_entries.append(a, .{
            .val_type = @intFromEnum(g.global_type.val_type),
            .mutability = if (g.global_type.mutability == .mutable) @as(u8, 1) else @as(u8, 0),
            .init_i64 = valueToI64(val),
        });
    }

    // Element segments: forward active segments whose offset is an
    // i32.const literal and whose entries are all concrete funcidx. This
    // is sufficient for the bulk of the spec suite; passive/declarative
    // and init-expr segments are deferred.
    var elem_entries: std.ArrayList(emit_aot.ElemEntry) = .empty;
    for (module.elements) |seg| {
        if (seg.is_passive or seg.is_declarative) continue;
        const offset_expr = seg.offset orelse continue;
        const offset_val: u32 = switch (offset_expr) {
            .i32_const => |v| @bitCast(v),
            else => continue,
        };
        var all_concrete = true;
        for (seg.func_indices) |fi| {
            if (fi == null) { all_concrete = false; break; }
        }
        if (!all_concrete) continue;
        const indices = try a.alloc(u32, seg.func_indices.len);
        for (seg.func_indices, 0..) |fi, k| indices[k] = fi.?;
        try elem_entries.append(a, .{
            .table_idx = seg.table_idx,
            .offset = offset_val,
            .func_indices = indices,
        });
    }

    // Data segments: forward active segments with i32.const offsets.
    var data_entries: std.ArrayList(emit_aot.DataSegmentEntry) = .empty;
    for (module.data_segments) |seg| {
        if (seg.is_passive) continue;
        const offset_val: u32 = switch (seg.offset) {
            .i32_const => |v| @bitCast(v),
            else => continue,
        };
        try data_entries.append(a, .{
            .memory_idx = seg.memory_idx,
            .offset = offset_val,
            .data = seg.data,
        });
    }

    var arch_name = std.mem.zeroes([16]u8);
    switch (builtin.cpu.arch) {
        .aarch64 => @memcpy(arch_name[0..7], "aarch64"),
        else => @memcpy(arch_name[0..6], "x86-64"),
    }

    return try emit_aot.emit(
        allocator,
        code,
        offsets,
        exports.items,
        .{ .arch = arch_name },
        if (data_entries.items.len > 0) data_entries.items else null,
        if (import_entries.items.len > 0) import_entries.items else null,
        if (mem_entries.items.len > 0) mem_entries.items else null,
        if (global_entries.items.len > 0) global_entries.items else null,
        if (elem_entries.items.len > 0) elem_entries.items else null,
        module.start_function,
    );
}

// ─── Helpers for global init value resolution ──────────────────────────────

/// Default zero value of the given wasm type. Used as a fallback when an
/// imported global isn't resolvable or an init expression fails to evaluate.
fn defaultZeroValue(vt: types.ValType) types.Value {
    return switch (vt) {
        .i32 => .{ .i32 = 0 },
        .i64 => .{ .i64 = 0 },
        .f32 => .{ .f32 = 0 },
        .f64 => .{ .f64 = 0 },
        else => .{ .i64 = 0 },
    };
}

/// Pack a typed Value into an i64 for the globals_buf layout expected by
/// AOT code. Must stay in sync with `globalValueToI64` in
/// runtime/aot/runtime.zig.
fn valueToI64(v: types.Value) i64 {
    return switch (v) {
        .i32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .i64 => |x| x,
        .f32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .f64 => |x| @as(i64, @bitCast(x)),
        else => 0,
    };
}

/// Resolve a `spectest.*` imported global to its canonical value. Mirrors
/// the values in `src/tests/spec_json_runner.zig` (global_i32=666,
/// global_i64=666, global_f32=666.6, global_f64=666.6). Returns null for
/// unrecognized imports so the caller can fall back to a default.
fn spectestGlobalInitValue(module_name: []const u8, field: []const u8, vt: types.ValType) ?types.Value {
    if (!std.mem.eql(u8, module_name, "spectest")) return null;
    if (std.mem.eql(u8, field, "global_i32") and vt == .i32) return .{ .i32 = 666 };
    if (std.mem.eql(u8, field, "global_i64") and vt == .i64) return .{ .i64 = 666 };
    if (std.mem.eql(u8, field, "global_f32") and vt == .f32) return .{ .f32 = 666.6 };
    if (std.mem.eql(u8, field, "global_f64") and vt == .f64) return .{ .f64 = 666.6 };
    return null;
}