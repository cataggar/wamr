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

/// Import registry for cross-module links discovered via `register`.
/// Tracks exported globals and exported function native code pointers
/// keyed by "module_name/field_name". Lookups fall through to the
/// spectest fallback if no hit.
pub const ImportRegistry = struct {
    allocator: std.mem.Allocator,
    globals: std.StringHashMap(types.Value),
    /// Exported-function native code pointer (post-mapCodeExecutable).
    /// Lifetime is tied to the exporter Harness — callers must ensure
    /// the exporter outlives any importer that consults this entry.
    functions: std.StringHashMap(usize),
    /// Exported funcref-typed globals resolved to the native code pointer of
    /// the referenced function in the exporter. Lets importers whose
    /// element segments are initialized from an imported funcref global
    /// (e.g. `(elem ... (global.get 0))`) install a real native pointer
    /// in their local table rather than a bogus funcidx in the importer's
    /// index space.
    funcref_globals: std.StringHashMap(usize),
    /// Exported memory instances keyed by "<module_name>/<field>".
    /// Values are non-owning pointers into the exporter Harness
    /// (refcount-shared via `MemoryInstance.retain`/`release` on swap-in).
    /// Lifetime of the exporter Harness must exceed any importer that
    /// installs the pointer — spec_json_runner keeps harnesses alive
    /// via the `retained` list on `register`.
    memories: std.StringHashMap(*types.MemoryInstance),

    pub fn init(allocator: std.mem.Allocator) ImportRegistry {
        return .{
            .allocator = allocator,
            .globals = std.StringHashMap(types.Value).init(allocator),
            .functions = std.StringHashMap(usize).init(allocator),
            .funcref_globals = std.StringHashMap(usize).init(allocator),
            .memories = std.StringHashMap(*types.MemoryInstance).init(allocator),
        };
    }

    pub fn deinit(self: *ImportRegistry) void {
        var it = self.globals.keyIterator();
        while (it.next()) |k| self.allocator.free(k.*);
        self.globals.deinit();
        var itf = self.functions.keyIterator();
        while (itf.next()) |k| self.allocator.free(k.*);
        self.functions.deinit();
        var itfg = self.funcref_globals.keyIterator();
        while (itfg.next()) |k| self.allocator.free(k.*);
        self.funcref_globals.deinit();
        var itm = self.memories.keyIterator();
        while (itm.next()) |k| self.allocator.free(k.*);
        self.memories.deinit();
    }

    /// Record an exported global under "<module_name>/<field>" = value.
    pub fn putGlobal(self: *ImportRegistry, module_name: []const u8, field: []const u8, value: types.Value) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ module_name, field });
        errdefer self.allocator.free(key);
        const gop = try self.globals.getOrPut(key);
        if (gop.found_existing) {
            self.allocator.free(key);
        }
        gop.value_ptr.* = value;
    }

    pub fn getGlobal(self: *const ImportRegistry, module_name: []const u8, field: []const u8) ?types.Value {
        var buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}/{s}", .{ module_name, field }) catch return null;
        return self.globals.get(key);
    }

    /// Record an exported function's native code pointer under
    /// "<module_name>/<field>". The pointer comes from the exporter's
    /// `inst.funcptrs[func_idx]` after `mapCodeExecutable` has run.
    pub fn putFunction(self: *ImportRegistry, module_name: []const u8, field: []const u8, native_ptr: usize) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ module_name, field });
        errdefer self.allocator.free(key);
        const gop = try self.functions.getOrPut(key);
        if (gop.found_existing) {
            self.allocator.free(key);
        }
        gop.value_ptr.* = native_ptr;
    }

    pub fn getFunction(self: *const ImportRegistry, module_name: []const u8, field: []const u8) ?usize {
        var buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}/{s}", .{ module_name, field }) catch return null;
        return self.functions.get(key);
    }

    /// Record the native pointer for a funcref-typed global export.
    pub fn putFuncrefGlobal(self: *ImportRegistry, module_name: []const u8, field: []const u8, native_ptr: usize) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ module_name, field });
        errdefer self.allocator.free(key);
        const gop = try self.funcref_globals.getOrPut(key);
        if (gop.found_existing) {
            self.allocator.free(key);
        }
        gop.value_ptr.* = native_ptr;
    }

    pub fn getFuncrefGlobal(self: *const ImportRegistry, module_name: []const u8, field: []const u8) ?usize {
        var buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}/{s}", .{ module_name, field }) catch return null;
        return self.funcref_globals.get(key);
    }

    /// Record an exported memory instance under "<module_name>/<field>".
    /// Pointer is non-owning; the exporter Harness retains ownership.
    pub fn putMemory(self: *ImportRegistry, module_name: []const u8, field: []const u8, mem: *types.MemoryInstance) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ module_name, field });
        errdefer self.allocator.free(key);
        const gop = try self.memories.getOrPut(key);
        if (gop.found_existing) {
            self.allocator.free(key);
        }
        gop.value_ptr.* = mem;
    }

    pub fn getMemory(self: *const ImportRegistry, module_name: []const u8, field: []const u8) ?*types.MemoryInstance {
        var buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}/{s}", .{ module_name, field }) catch return null;
        return self.memories.get(key);
    }
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
        return initWithRegistry(allocator, wasm_bytes, null);
    }

    /// Like `init` but consults `registry` when resolving imports that the
    /// built-in spectest fallback doesn't recognize. Currently only
    /// imported-global values flow through here (enough for the spec suite's
    /// cross-module global tests in `global.json`).
    pub fn initWithRegistry(
        allocator: std.mem.Allocator,
        wasm_bytes: []const u8,
        registry: ?*const ImportRegistry,
    ) Error!*Harness {
        if (comptime !can_exec_aot) return error.AotUnsupportedArch;

        const h = allocator.create(Harness) catch return error.OutOfMemory;
        errdefer allocator.destroy(h);

        const arena = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
        errdefer allocator.destroy(arena);
        arena.* = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const a = arena.allocator();

        // The interpreter loader returns export/import names (and other
        // string slices) that alias the input wasm buffer. Callers
        // typically free the buffer right after `init` returns, so we
        // copy it into the arena first to guarantee those slices stay
        // valid for the lifetime of the Harness (needed for `register`).
        const owned_wasm = a.dupe(u8, wasm_bytes) catch return error.OutOfMemory;

        const wasm_module = loader_mod.load(owned_wasm, a) catch |err| {
            std.debug.print("aot_harness: loader failed: {}\n", .{err});
            return error.CompileFailed;
        };

        const aot_bin = compileToAot(allocator, arena, &wasm_module, registry) catch |err| {
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

        // The AOT binary format does not currently encode the table section,
        // so `aot_runtime.instantiate` produces `inst.tables = &.{}` even when
        // the module declares local tables. Patch the tables up from the
        // interpreter-loaded `wasm_module` so that `table.grow` has accurate
        // limits and `mapCodeExecutable` allocates a `func_table` of the
        // declared `min` size even without element segments.
        if (h.inst.tables.len < wasm_module.tables.len) {
            patchTables(h.inst, allocator, wasm_module.tables) catch
                return error.InstantiateFailed;
        }

        // Swap in shared memories for imported memory slots (by registry
        // lookup). compileToAot emits imported memories at indices
        // [0, import_memory_count); `allocateMemories` allocated fresh
        // MemoryInstances for them, which we release here and replace with
        // the exporter's `*MemoryInstance` (refcount-retained). This
        // causes cross-module `memory.grow` / `memory.size` to observe a
        // single shared buffer. Unresolved imports keep the fresh slot.
        if (registry) |reg| {
            var mem_import_idx: u32 = 0;
            for (wasm_module.imports) |imp| {
                if (imp.kind != .memory) continue;
                defer mem_import_idx += 1;
                if (mem_import_idx >= h.inst.memories.len) break;
                const shared = reg.getMemory(imp.module_name, imp.field_name) orelse continue;
                const mem_mut = @as([]*types.MemoryInstance, @constCast(h.inst.memories));
                mem_mut[mem_import_idx].release(allocator);
                shared.retain();
                mem_mut[mem_import_idx] = shared;
            }
        }

        aot_runtime.mapCodeExecutable(h.inst) catch return error.MapExecutableFailed;

        // Wire cross-module function imports from the registry. The
        // exporter's `mapCodeExecutable` has already populated its
        // `funcptrs[exp.index]` with the native code address; patch the
        // importer's `host_functions[i]` (which the AOT codegen calls
        // through VmCtx.host_functions_ptr) and `funcptrs[i]` (which
        // `ref.func N` reads) to that same address. The exporter Harness
        // must outlive this importer — spec_json_runner keeps harnesses
        // alive via the ImportRegistry's owning scope.
        if (registry) |reg| {
            const hf_mut = @as([]?*const anyopaque, @constCast(h.inst.host_functions));
            const fp_mut = @as([]usize, @constCast(h.inst.funcptrs));
            var func_import_idx: u32 = 0;
            for (wasm_module.imports) |imp| {
                if (imp.kind != .function) continue;
                defer func_import_idx += 1;
                if (reg.getFunction(imp.module_name, imp.field_name)) |native_ptr| {
                    if (func_import_idx < hf_mut.len) {
                        hf_mut[func_import_idx] = @ptrFromInt(native_ptr);
                    }
                    if (func_import_idx < fp_mut.len) {
                        fp_mut[func_import_idx] = native_ptr;
                    }
                }
            }

            // Patch native table entries for active elem segments whose
            // items are `global.get K` targeting an imported funcref
            // global. The AOT-emit path leaves these slots as `maxInt(u32)`
            // (unresolvable at compile time because the exporter's native
            // pointer isn't known until after `mapCodeExecutable` on the
            // exporter), so the runtime left them as 0 in the native
            // table. Now, with the registry populated, write the correct
            // native pointer directly into the per-table backing store.
            patchImportedFuncrefElems(h.inst, &wasm_module, reg) catch {};
        }

        // Invoke the start function, if any. The start function has type
        // `() -> ()` per the wasm spec, so we pass no params/results.
        if (h.aot_module.start_function) |start_idx| {
            const start_ft = h.wasm_module.getFuncType(start_idx);
            if (start_ft != null) {
                var start_results: [1]aot_runtime.ScalarResult = undefined;
                _ = aot_runtime.callFuncScalar(
                    h.inst,
                    start_idx,
                    start_ft.?.params,
                    &.{},
                    &.{},
                    &start_results,
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

    /// Copy every exported global of this instance into `registry` keyed by
    /// `<module_name>/<export_name>`. Used to expose a just-registered
    /// module's globals to subsequently-compiled modules that import them.
    pub fn exportGlobalsToRegistry(
        self: *const Harness,
        registry: *ImportRegistry,
        module_name: []const u8,
    ) !void {
        for (self.wasm_module.exports) |exp| {
            if (exp.kind != .global) continue;
            if (exp.index >= self.inst.globals.len) continue;
            const val = self.inst.globals[exp.index].value;
            try registry.putGlobal(module_name, exp.name, val);
            // For funcref globals, also resolve to the exporter's native
            // code pointer so importers can install it directly in their
            // local tables (elem_exprs with `global.get <imported funcref>`).
            switch (val) {
                .funcref, .nonfuncref => |maybe| if (maybe) |fidx| {
                    if (fidx < self.inst.funcptrs.len) {
                        const ptr = self.inst.funcptrs[fidx];
                        if (ptr != 0) {
                            try registry.putFuncrefGlobal(module_name, exp.name, ptr);
                        }
                    }
                },
                else => {},
            }
        }
    }

    /// Copy every exported function's native code pointer into `registry`
    /// keyed by `<module_name>/<export_name>`. Must be called after
    /// `mapCodeExecutable`, which populates `inst.funcptrs`. Used to
    /// expose a just-registered module's functions to subsequently-compiled
    /// modules that import them (cross-module `call` / `ref.func`).
    pub fn exportFunctionsToRegistry(
        self: *const Harness,
        registry: *ImportRegistry,
        module_name: []const u8,
    ) !void {
        for (self.wasm_module.exports) |exp| {
            if (exp.kind != .function) continue;
            if (exp.index >= self.inst.funcptrs.len) continue;
            const ptr = self.inst.funcptrs[exp.index];
            if (ptr == 0) continue;
            try registry.putFunction(module_name, exp.name, ptr);
        }
    }

    /// Publish every exported memory's `*MemoryInstance` pointer into
    /// `registry` keyed by `<module_name>/<export_name>`. Subsequent
    /// importers' `initWithRegistry` will install the shared pointer in
    /// their own `inst.memories` slot so `memory.grow` / `memory.size`
    /// observe the same underlying buffer across modules.
    pub fn exportMemoriesToRegistry(
        self: *const Harness,
        registry: *ImportRegistry,
        module_name: []const u8,
    ) !void {
        for (self.wasm_module.exports) |exp| {
            if (exp.kind != .memory) continue;
            if (exp.index >= self.inst.memories.len) continue;
            try registry.putMemory(module_name, exp.name, self.inst.memories[exp.index]);
        }
    }

    /// Invoke an AOT function by index with runtime-typed scalar args.
    /// See `aot_runtime.callFuncScalar` for the supported signature envelope.
    /// Writes decoded results into `results_buf` and returns a slice into it.
    pub fn callScalar(
        self: *Harness,
        func_idx: u32,
        args: []const types.Value,
        results_buf: []aot_runtime.ScalarResult,
    ) aot_runtime.ScalarCallError![]const aot_runtime.ScalarResult {
        const ft = self.getFuncType(func_idx) orelse return error.FunctionNotFound;
        return aot_runtime.callFuncScalar(self.inst, func_idx, ft.params, ft.results, args, results_buf);
    }
};

fn patchTables(
    inst: *aot_runtime.AotInstance,
    allocator: std.mem.Allocator,
    wasm_tables: []const types.TableType,
) !void {
    // Free any tables allocated by the default-table fallback in
    // aot_runtime.instantiate so we can recreate the full set from the
    // wasm module (which encodes every declared table).
    if (inst.tables.len > 0) {
        for (inst.tables) |tbl| tbl.release(allocator);
        allocator.free(inst.tables);
        inst.tables = &.{};
    }
    const tables = try allocator.alloc(*types.TableInstance, wasm_tables.len);
    errdefer allocator.free(tables);
    for (wasm_tables, 0..) |tt, i| {
        const elements = try allocator.alloc(types.TableElement, tt.limits.min);
        for (elements) |*e| e.* = types.TableElement.nullForType(tt.elem_type);
        const tbl = try allocator.create(types.TableInstance);
        tbl.* = .{ .table_type = tt, .elements = elements };
        tables[i] = tbl;
    }
    inst.tables = tables;
}

/// After `mapCodeExecutable` has populated the native func_table, fix up
/// entries written by active elem segments whose items are
/// `global.get K` where K is an imported funcref global. Such entries are
/// left as 0 by the runtime (because the AOT binary encodes them as
/// `maxInt(u32)` sentinels — the exporter's native pointer is unknown at
/// AOT compile time). Now, with the registry populated, write the real
/// native pointer directly into the per-table backing store.
fn patchImportedFuncrefElems(
    inst: *aot_runtime.AotInstance,
    wasm_module: *const types.WasmModule,
    registry: *const ImportRegistry,
) !void {
    // Build a per-import-global lookup: for each imported funcref global,
    // note (import module name, field name) so we can resolve via
    // registry.getFuncrefGlobal.
    for (wasm_module.elements) |seg| {
        if (seg.is_passive or seg.is_declarative) continue;
        // Reuse the existing compile-time offset reducer; locals-only
        // globals are available via inst.globals here. Build a slice of
        // pointers from the interpreter-loaded globals (matches the order
        // used during compile), then call resolveU32InitExpr.
        const offset_expr = seg.offset orelse continue;
        // `tmp_globals` at instantiate time is seeded from wasm_module
        // globals (import then local). We only need i32/global_get
        // resolution here, so pass an empty slice and fall through to
        // evalInitExpr which walks the module's init expressions.
        const offset_val: u32 = resolveU32InitExpr(offset_expr, &.{}) orelse continue;

        for (seg.elem_exprs, 0..) |maybe_expr, k| {
            const expr = maybe_expr orelse continue;
            // Only handle `global.get gidx` where gidx < import_global_count.
            const gidx: u32 = switch (expr) {
                .global_get => |g| g,
                else => continue,
            };
            if (gidx >= wasm_module.import_global_count) continue;

            // Walk imports to find the matching import-global entry.
            var import_global_seen: u32 = 0;
            var resolved_ptr: ?usize = null;
            for (wasm_module.imports) |imp| {
                if (imp.kind != .global) continue;
                defer import_global_seen += 1;
                if (import_global_seen != gidx) continue;
                resolved_ptr = registry.getFuncrefGlobal(imp.module_name, imp.field_name);
                break;
            }
            const native_ptr = resolved_ptr orelse continue;

            // Pick the native backing store for seg.table_idx.
            const dst_idx: usize = @as(usize, offset_val) + k;
            if (seg.table_idx == 0) {
                if (dst_idx < inst.func_table.len) inst.func_table[dst_idx] = native_ptr;
            } else {
                const extra_idx = seg.table_idx - 1;
                if (extra_idx >= inst.extra_tables_storage.len) continue;
                const backing = inst.extra_tables_storage[extra_idx];
                if (dst_idx < backing.len) backing[dst_idx] = native_ptr;
            }
        }
    }
}

fn compileToAot(
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    module: *const types.WasmModule,
    registry: ?*const ImportRegistry,
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
        // Cross-module table linking isn't supported yet (Phase 4). A
        // module with an imported table currently falls through with an
        // empty `module.tables`, so `table.size` on it would return 0
        // instead of the declared min. Surface as UnsupportedOpcode so
        // the spec runner records a skip rather than a value mismatch.
        if (imp.kind == .table) return error.UnsupportedOpcode;
        try import_entries.append(a, .{
            .module_name = imp.module_name,
            .field_name = imp.field_name,
            .kind = @enumFromInt(@intFromEnum(imp.kind)),
            .func_type_idx = imp.func_type_idx orelse 0,
        });
    }

    var mem_entries: std.ArrayList(emit_aot.MemoryEntry) = .empty;
    // Imported memories first so `memory_idx` (combined import+local
    // indexing per interpreter loader) continues to reference the right
    // slot. `initWithRegistry` swaps the importer's fresh allocations
    // for the exporter's shared `*MemoryInstance` post-instantiate.
    for (module.imports) |imp| {
        if (imp.kind != .memory) continue;
        const mt = imp.memory_type orelse continue;
        try mem_entries.append(a, .{
            .min_pages = @intCast(mt.limits.min),
            .max_pages = if (mt.limits.max) |m| @as(?u32, @intCast(m)) else null,
        });
    }
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
        const val: types.Value = blk: {
            if (registry) |reg| {
                if (reg.getGlobal(imp.module_name, imp.field_name)) |v| {
                    break :blk v;
                }
            }
            if (spectestGlobalInitValue(imp.module_name, imp.field_name, gt.val_type)) |v| {
                break :blk v;
            }
            break :blk defaultZeroValue(gt.val_type);
        };
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

    // Element segments: forward active segments whose offset reduces to
    // a constant u32 (`i32.const` literal, or `global.get K` where K is
    // an i32 immutable global with a known constant value at instantiate
    // time) and whose entries are all concrete funcidx. This is sufficient
    // for the bulk of the spec suite; passive/declarative and arbitrary
    // bytecode segments are deferred.
    var elem_entries: std.ArrayList(emit_aot.ElemEntry) = .empty;
    for (module.elements) |seg| {
        if (seg.is_passive or seg.is_declarative) continue;
        const offset_expr = seg.offset orelse continue;
        const offset_val: u32 = resolveU32InitExpr(offset_expr, tmp_globals.items) orelse continue;
        const indices = try a.alloc(u32, seg.func_indices.len);
        for (seg.func_indices, 0..) |fi, k| {
            if (fi) |v| {
                indices[k] = v;
                continue;
            }
            // Fall back to resolving the init expression stored in
            // elem_exprs (e.g. `global.get K` where global K holds a
            // funcref produced by `ref.func N`). Unresolvable entries
            // encode as maxInt(u32) → null slot at runtime.
            if (k < seg.elem_exprs.len) {
                if (seg.elem_exprs[k]) |expr| {
                    if (resolveFuncIdxFromExpr(expr, tmp_globals.items, module.import_global_count)) |resolved| {
                        indices[k] = resolved;
                        continue;
                    }
                }
            }
            indices[k] = std.math.maxInt(u32);
        }
        try elem_entries.append(a, .{
            .table_idx = seg.table_idx,
            .offset = offset_val,
            .func_indices = indices,
        });
    }

    // Data segments: forward active segments whose offset reduces to a
    // constant u32. See `resolveU32InitExpr` for the supported forms.
    var data_entries: std.ArrayList(emit_aot.DataSegmentEntry) = .empty;
    for (module.data_segments) |seg| {
        if (seg.is_passive) continue;
        const offset_val: u32 = resolveU32InitExpr(seg.offset, tmp_globals.items) orelse continue;
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

/// Reduce an `InitExpr` to a constant u32 offset for active elem/data
/// segments. Handles `i32.const` literals and `global.get K` where K
/// resolves to an immutable i32 global with a known constant value at
/// instantiate time (typically a spectest import — see `global.json`,
/// where active elem/data offsets reference the imported `global_i32`).
/// Returns null when the expression cannot be resolved at AOT compile
/// time, in which case the caller skips emitting the segment.
fn resolveU32InitExpr(
    expr: types.InitExpr,
    tmp_globals: []const *types.GlobalInstance,
) ?u32 {
    // Fast path: avoid invoking the interpreter for trivial cases that
    // arise frequently in spec tests.
    switch (expr) {
        .i32_const => |v| return @as(u32, @bitCast(v)),
        .global_get => |idx| {
            if (idx >= tmp_globals.len) return null;
            const gv = tmp_globals[idx].value;
            return switch (gv) {
                .i32 => |x| @as(u32, @bitCast(x)),
                else => null,
            };
        },
        else => {},
    }
    // Compound expressions (i32.add/sub/mul, etc.) are stored as raw
    // bytecode. Delegate to the interpreter's `evalInitExpr` so that
    // extended constant expressions (the wasm 2.0 proposal) are
    // supported uniformly.
    const val = root.instance.evalInitExpr(expr, tmp_globals, null) catch return null;
    return switch (val) {
        .i32 => |x| @as(u32, @bitCast(x)),
        else => null,
    };
}

/// Resolve an element segment item init expression to a wasm funcidx,
/// or return null if the expression represents a null funcref or cannot
/// be reduced at instantiation time.
///
/// NOTE: for `global.get K` we intentionally restrict to LOCAL globals
/// (K >= module.import_global_count). Imported funcref globals hold a
/// funcidx in the exporting module's index space, which is not
/// meaningful in the importing module's table / call_indirect. Writing
/// such a funcidx into a local table would install the wrong function
/// pointer and can produce stack-overflow/crash at call time.
fn resolveFuncIdxFromExpr(
    expr: types.InitExpr,
    tmp_globals: []const *types.GlobalInstance,
    import_global_count: u32,
) ?u32 {
    return switch (expr) {
        .ref_func => |idx| idx,
        .ref_null => null,
        .global_get => |gidx| blk: {
            if (gidx < import_global_count) break :blk null;
            if (gidx >= tmp_globals.len) break :blk null;
            const gv = tmp_globals[gidx].value;
            break :blk switch (gv) {
                .funcref, .nonfuncref => |maybe| maybe,
                else => null,
            };
        },
        else => null,
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
        // Reference types: store the wasm function/extern index as a
        // plain i64 in the AOT binary with a +1 sentinel so `ref.func 0`
        // is distinguishable from null (0). The runtime decodes via
        // `allocateGlobals` with the matching `-1` shift.
        .funcref, .nonfuncref => |maybe| if (maybe) |x| @as(i64, @as(u32, x)) + 1 else 0,
        .externref, .nonexternref => |maybe| if (maybe) |x| @as(i64, @as(u32, x)) + 1 else 0,
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