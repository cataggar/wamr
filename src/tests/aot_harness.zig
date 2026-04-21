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
const platform = root.platform;

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
    pub const FuncrefGlobalEntry = struct { ptr: usize, sig_id: u32 };

    allocator: std.mem.Allocator,
    globals: std.StringHashMap(types.Value),
    /// Exported-function native code pointer (post-mapCodeExecutable).
    /// Lifetime is tied to the exporter Harness — callers must ensure
    /// the exporter outlives any importer that consults this entry.
    functions: std.StringHashMap(usize),
    /// Exported funcref-typed globals resolved to the native code pointer of
    /// the referenced function in the exporter, plus its canonical sig_id.
    /// Lets importers whose element segments are initialized from an
    /// imported funcref global (e.g. `(elem ... (global.get 0))`) install
    /// a real native pointer AND correct sig_id in their local table.
    funcref_globals: std.StringHashMap(FuncrefGlobalEntry),
    /// Exported memory instances keyed by "<module_name>/<field>".
    /// Values are non-owning pointers into the exporter Harness
    /// (refcount-shared via `MemoryInstance.retain`/`release` on swap-in).
    /// Lifetime of the exporter Harness must exceed any importer that
    /// installs the pointer — spec_json_runner keeps harnesses alive
    /// via the `retained` list on `register`.
    memories: std.StringHashMap(*types.MemoryInstance),
    /// Exported table instances keyed by "<module_name>/<field>".
    /// Same lifetime rules as `memories` (shared via
    /// `TableInstance.retain`/`release`).
    tables: std.StringHashMap(*types.TableInstance),
    /// Exported mutable globals as shared `*GlobalInstance` pointers,
    /// keyed by "<module_name>/<field>". Importers swap their local
    /// GlobalInstance for this pointer so mutations are visible across
    /// modules.
    shared_globals: std.StringHashMap(*types.GlobalInstance),

    pub fn init(allocator: std.mem.Allocator) ImportRegistry {
        return .{
            .allocator = allocator,
            .globals = std.StringHashMap(types.Value).init(allocator),
            .functions = std.StringHashMap(usize).init(allocator),
            .funcref_globals = std.StringHashMap(FuncrefGlobalEntry).init(allocator),
            .memories = std.StringHashMap(*types.MemoryInstance).init(allocator),
            .tables = std.StringHashMap(*types.TableInstance).init(allocator),
            .shared_globals = std.StringHashMap(*types.GlobalInstance).init(allocator),
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
        var itt = self.tables.keyIterator();
        while (itt.next()) |k| self.allocator.free(k.*);
        self.tables.deinit();
        var itsg = self.shared_globals.keyIterator();
        while (itsg.next()) |k| self.allocator.free(k.*);
        self.shared_globals.deinit();
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

    /// Record the native pointer + sig_id for a funcref-typed global export.
    pub fn putFuncrefGlobal(self: *ImportRegistry, module_name: []const u8, field: []const u8, native_ptr: usize, sig_id: u32) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ module_name, field });
        errdefer self.allocator.free(key);
        const gop = try self.funcref_globals.getOrPut(key);
        if (gop.found_existing) {
            self.allocator.free(key);
        }
        gop.value_ptr.* = .{ .ptr = native_ptr, .sig_id = sig_id };
    }

    pub fn getFuncrefGlobal(self: *const ImportRegistry, module_name: []const u8, field: []const u8) ?FuncrefGlobalEntry {
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

    /// Record an exported table instance under "<module_name>/<field>".
    /// Pointer is non-owning; the exporter Harness retains ownership.
    pub fn putTable(self: *ImportRegistry, module_name: []const u8, field: []const u8, tbl: *types.TableInstance) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ module_name, field });
        errdefer self.allocator.free(key);
        const gop = try self.tables.getOrPut(key);
        if (gop.found_existing) {
            self.allocator.free(key);
        }
        gop.value_ptr.* = tbl;
    }

    pub fn getTable(self: *const ImportRegistry, module_name: []const u8, field: []const u8) ?*types.TableInstance {
        var buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}/{s}", .{ module_name, field }) catch return null;
        return self.tables.get(key);
    }

    pub fn putSharedGlobal(self: *ImportRegistry, module_name: []const u8, field: []const u8, g: *types.GlobalInstance) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ module_name, field });
        errdefer self.allocator.free(key);
        const gop = try self.shared_globals.getOrPut(key);
        if (gop.found_existing) {
            self.allocator.free(key);
        }
        gop.value_ptr.* = g;
    }

    pub fn getSharedGlobal(self: *const ImportRegistry, module_name: []const u8, field: []const u8) ?*types.GlobalInstance {
        var buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}/{s}", .{ module_name, field }) catch return null;
        return self.shared_globals.get(key);
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
    /// Persistent VmCtx for cross-module trampolines. Built once after
    /// full initialization; imported functions can jump through a
    /// trampoline that loads this pointer so the callee sees the
    /// exporter's memory/tables/globals instead of the caller's.
    persistent_vmctx: ?*aot_runtime.VmCtx = null,
    persistent_globals: ?[]i64 = null,
    /// Executable trampoline stubs (mmap'd). Each stub loads
    /// persistent_vmctx into param_regs[0] and jumps to the real func.
    trampoline_pages: ?[*]u8 = null,
    trampoline_size: usize = 0,

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
        var import_table_count: u32 = 0;
        for (wasm_module.imports) |imp| {
            if (imp.kind == .table) import_table_count += 1;
        }
        const expected_tables = import_table_count + wasm_module.tables.len;
        if (expected_tables > 0) {
            patchTables(h.inst, allocator, wasm_module.imports, wasm_module.tables) catch
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

        // Re-apply active data segments to the (possibly swapped-in)
        // memories so that modules importing a shared memory still get
        // their own data segments written into it.
        for (h.aot_module.data_segments) |seg| {
            if (seg.memory_idx >= h.inst.memories.len) continue;
            const mem = h.inst.memories[seg.memory_idx];
            const end = @as(usize, seg.offset) + seg.data.len;
            if (end > mem.data.len) continue;
            @memcpy(mem.data[seg.offset..][0..seg.data.len], seg.data);
        }

        // Swap in shared mutable globals for imported global slots.
        // The wasm spec requires mutable global imports to alias the
        // exporter's storage so cross-module mutations are visible.
        if (registry) |reg| {
            var glob_import_idx: u32 = 0;
            for (wasm_module.imports) |imp| {
                if (imp.kind != .global) continue;
                defer glob_import_idx += 1;
                const gt = imp.global_type orelse continue;
                if (gt.mutability != .mutable) continue;
                if (glob_import_idx >= h.inst.globals.len) break;
                const shared = reg.getSharedGlobal(imp.module_name, imp.field_name) orelse continue;
                const g_mut = @as([]*types.GlobalInstance, @constCast(h.inst.globals));
                g_mut[glob_import_idx].release(allocator);
                shared.retain();
                g_mut[glob_import_idx] = shared;
            }
        }

        // Swap in shared tables for imported table slots (by registry
        // lookup). `patchTables` reserved slots at indices
        // [0, import_table_count) with fresh local TableInstances; replace
        // each with the exporter's `*TableInstance` (refcount-retained).
        // Cross-module `table.get`/`table.set` observe the shared
        // `TableInstance.elements` array. Note: `mapCodeExecutable` reads
        // the table size/limits from `inst.tables[*]` so this must happen
        // before that call. Unresolved imports keep the fresh local slot.
        if (registry) |reg| {
            var tbl_import_idx: u32 = 0;
            for (wasm_module.imports) |imp| {
                if (imp.kind != .table) continue;
                defer tbl_import_idx += 1;
                if (tbl_import_idx >= h.inst.tables.len) break;
                const shared = reg.getTable(imp.module_name, imp.field_name) orelse continue;
                const tbl_mut = @as([]*types.TableInstance, @constCast(h.inst.tables));
                tbl_mut[tbl_import_idx].release(allocator);
                shared.retain();
                tbl_mut[tbl_import_idx] = shared;
            }
        }

        // Wire cross-module function imports into host_functions BEFORE
        // mapCodeExecutable, because mapCodeExecutable builds the native
        // table for call_indirect from inst.host_functions. Without this,
        // elem segment slots referencing imported functions are left as 0
        // in the native table, causing call_indirect to trap.
        if (registry) |reg| {
            const hf_mut = @as([]?*const anyopaque, @constCast(h.inst.host_functions));
            var func_import_idx: u32 = 0;
            for (wasm_module.imports) |imp| {
                if (imp.kind != .function) continue;
                defer func_import_idx += 1;
                if (reg.getFunction(imp.module_name, imp.field_name)) |native_ptr| {
                    if (func_import_idx < hf_mut.len) {
                        hf_mut[func_import_idx] = @ptrFromInt(native_ptr);
                    }
                }
            }
        }

        // Re-intern module types with recursive group context from the
        // interpreter-loaded WasmModule. The AOT binary format doesn't
        // carry rec_group_size/position, so instantiate() interns all
        // types as singletons. This patch adds the group context so
        // call_indirect distinguishes types in different rec groups.
        if (wasm_module.types.len > 0 and h.inst.sig_table.len > 0) {
            const reg = root.sig_registry.global();
            for (wasm_module.types, 0..) |ft, i| {
                if (i >= h.inst.sig_table.len) break;
                const rg = if (i < wasm_module.rec_groups.len)
                    wasm_module.rec_groups[i]
                else
                    types.RecGroupInfo{ .group_start = @intCast(i), .group_size = 1 };
                if (rg.group_size <= 1) continue; // singleton — already correct
                const pos: u16 = @intCast(@as(u32, @intCast(i)) - rg.group_start);
                const augmented = types.FuncType{
                    .params = ft.params,
                    .results = ft.results,
                    .kind = ft.kind,
                    .field_types = ft.field_types,
                    .field_muts = ft.field_muts,
                    .rec_group_size = @intCast(rg.group_size),
                    .rec_group_position = pos,
                };
                h.inst.sig_table[i] = reg.intern(&augmented) catch continue;
            }
            // Rebuild func_sig_ids from the corrected sig_table.
            if (h.inst.func_sig_ids.len > 0) {
                const import_count = h.aot_module.import_function_count;
                var slot: u32 = 0;
                for (wasm_module.imports) |imp| {
                    if (imp.kind != .function) continue;
                    if (slot >= import_count) break;
                    const ft_idx = imp.func_type_idx orelse 0;
                    if (ft_idx < h.inst.sig_table.len and slot < h.inst.func_sig_ids.len) {
                        h.inst.func_sig_ids[slot] = h.inst.sig_table[ft_idx];
                    }
                    slot += 1;
                }
                for (0..h.aot_module.func_count) |li| {
                    const tidx: u32 = if (li < h.aot_module.local_func_type_indices.len)
                        h.aot_module.local_func_type_indices[li]
                    else
                        0;
                    const fsi_idx = import_count + @as(u32, @intCast(li));
                    if (tidx < h.inst.sig_table.len and fsi_idx < h.inst.func_sig_ids.len) {
                        h.inst.func_sig_ids[fsi_idx] = h.inst.sig_table[tidx];
                    }
                }
            }
        }

        aot_runtime.mapCodeExecutable(h.inst) catch return error.MapExecutableFailed;

        // Post-mapCodeExecutable: funcptrs is now allocated. Patch import
        // slots so ref.func on imported indices returns the exporter's
        // native pointer (mapCodeExecutable already copied host_functions
        // into funcptrs, but re-affirm here for clarity). Also patch
        // funcref-elem table entries.
        if (registry) |reg| {
            const fp_mut = @as([]usize, @constCast(h.inst.funcptrs));
            var func_import_idx: u32 = 0;
            for (wasm_module.imports) |imp| {
                if (imp.kind != .function) continue;
                defer func_import_idx += 1;
                if (reg.getFunction(imp.module_name, imp.field_name)) |native_ptr| {
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

        // Apply table fill values (init_expr) for locally-declared tables.
        // Tables may declare a fill expression like `(table 10 funcref (ref.func $f))`.
        // patchTables null-initializes all elements; now that funcptrs are
        // populated we can resolve funcref fill values to native pointers.
        {
            var import_table_count2: u32 = 0;
            for (wasm_module.imports) |imp| {
                if (imp.kind == .table) import_table_count2 += 1;
            }
            for (wasm_module.tables, 0..) |tt, ti| {
                const tbl_idx = import_table_count2 + @as(u32, @intCast(ti));
                const init_e = tt.init_expr orelse continue;
                if (tbl_idx >= h.inst.tables.len) continue;
                const tbl = h.inst.tables[tbl_idx];
                // Resolve the fill value to a funcref index.
                const fill_fidx: ?u32 = switch (init_e) {
                    .ref_func => |fidx| fidx,
                    .global_get => |gidx| blk: {
                        if (gidx < h.inst.globals.len) {
                            switch (h.inst.globals[gidx].value) {
                                .funcref, .nonfuncref => |maybe| break :blk maybe,
                                else => break :blk null,
                            }
                        }
                        break :blk null;
                    },
                    .ref_null => continue,
                    else => continue,
                };
                const fidx = fill_fidx orelse continue;
                const native_ptr: usize = if (fidx < h.inst.funcptrs.len) h.inst.funcptrs[fidx] else 0;
                if (native_ptr == 0) continue;
                const sig_id: u32 = if (fidx < h.inst.func_sig_ids.len) h.inst.func_sig_ids[fidx] else 0;
                // Fill native_backing, type_backing, and elements.
                if (tbl_idx < h.inst.tables_info.len) {
                    const info = &h.inst.tables_info[tbl_idx];
                    if (info.ptr != 0) {
                        const backing: [*]usize = @ptrFromInt(@as(usize, @intCast(info.ptr)));
                        var k: u32 = 0;
                        while (k < info.len) : (k += 1) {
                            backing[k] = native_ptr;
                        }
                    }
                    if (info.type_backing_ptr != 0) {
                        const tb: [*]u32 = @ptrFromInt(@as(usize, @intCast(info.type_backing_ptr)));
                        var k: u32 = 0;
                        while (k < info.len) : (k += 1) {
                            tb[k] = sig_id;
                        }
                    }
                }
                for (tbl.elements) |*e| {
                    e.* = .{ .value = .{ .funcref = fidx } };
                }
            }
        }

        // Build persistent VmCtx for cross-module trampoline support.
        // Must happen after all wiring (funcptrs, tables_info, sig_table).
        h.buildPersistentVmCtx();

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
        if (self.trampoline_pages) |pages| {
            platform.munmap(pages, self.trampoline_size);
        }
        if (self.persistent_globals) |g| allocator.free(g);
        if (self.persistent_vmctx) |v| allocator.destroy(v);
        aot_runtime.destroy(self.inst);
        aot_loader.unload(&self.aot_module, allocator);
        allocator.free(self.aot_bin);
        self.arena.deinit();
        allocator.destroy(self.arena);
        allocator.destroy(self);
    }

    /// Build a heap-allocated VmCtx snapshot for this instance. Used by
    /// cross-module trampolines so imported functions run with the
    /// exporter's memory/tables/globals. Must be called after
    /// mapCodeExecutable and all import wiring.
    pub fn buildPersistentVmCtx(self: *Harness) void {
        const inst = self.inst;
        const allocator = self.allocator;
        const vmctx = allocator.create(aot_runtime.VmCtx) catch return;
        vmctx.* = .{};
        if (inst.memories.len > 0) {
            vmctx.memory_base = @intFromPtr(inst.memories[0].data.ptr);
            vmctx.memory_size = @as(usize, inst.memories[0].current_pages) * types.MemoryInstance.page_size;
            vmctx.memory_max_size = inst.memories[0].data.len;
            vmctx.memory_pages = inst.memories[0].current_pages;
        }
        const n_globals = @min(inst.globals.len, @as(usize, 256));
        const globals_buf = allocator.alloc(i64, 256) catch {
            allocator.destroy(vmctx);
            return;
        };
        @memset(globals_buf, 0);
        for (0..n_globals) |i| {
            globals_buf[i] = aot_runtime.globalValueToI64(inst, inst.globals[i].value);
        }
        vmctx.globals_ptr = @intFromPtr(globals_buf.ptr);
        vmctx.globals_count = @intCast(n_globals);
        if (inst.host_functions.len > 0) {
            vmctx.host_functions_ptr = @intFromPtr(inst.host_functions.ptr);
            vmctx.host_functions_count = @intCast(inst.host_functions.len);
        }
        if (inst.func_table.len > 0) {
            vmctx.func_table_ptr = @intFromPtr(inst.func_table.ptr);
            vmctx.func_table_len = @intCast(inst.func_table.len);
        }
        if (inst.funcptrs.len > 0) vmctx.funcptrs_ptr = @intFromPtr(inst.funcptrs.ptr);
        vmctx.instance_ptr = @intFromPtr(inst);
        vmctx.mem_grow_fn = @intFromPtr(&aot_runtime.memGrowHelper);
        vmctx.trap_oob_fn = @intFromPtr(&aot_runtime.aotTrapOOB);
        vmctx.trap_unreachable_fn = @intFromPtr(&aot_runtime.aotTrapUnreachable);
        vmctx.trap_idivz_fn = @intFromPtr(&aot_runtime.aotTrapIntDivZero);
        vmctx.trap_iovf_fn = @intFromPtr(&aot_runtime.aotTrapIntOverflow);
        vmctx.trap_ivc_fn = @intFromPtr(&aot_runtime.aotTrapInvalidConversion);
        vmctx.table_grow_fn = @intFromPtr(&aot_runtime.tableGrowHelper);
        if (inst.tables_info.len > 0) vmctx.tables_info_ptr = @intFromPtr(inst.tables_info.ptr);
        vmctx.table_init_fn = @intFromPtr(&aot_runtime.tableInitHelper);
        vmctx.elem_drop_fn = @intFromPtr(&aot_runtime.elemDropHelper);
        vmctx.table_set_fn = @intFromPtr(&aot_runtime.tableSetHelper);
        if (inst.sig_table.len > 0) vmctx.sig_table_ptr = @intFromPtr(inst.sig_table.ptr);
        if (inst.func_sig_ids.len > 0) vmctx.func_sig_ids_ptr = @intFromPtr(inst.func_sig_ids.ptr);
        if (inst.ptr_to_sig.len > 0) {
            vmctx.ptr_to_sig_ptr = @intFromPtr(inst.ptr_to_sig.ptr);
            vmctx.ptr_to_sig_len = @intCast(inst.ptr_to_sig.len);
        }
        self.persistent_vmctx = vmctx;
        self.persistent_globals = globals_buf;
    }

    /// Replace host_function entries with trampolines that switch to
    /// the exporter's persistent VmCtx before jumping to the real function.
    /// Call after initWithRegistry on the IMPORTER, passing a lookup
    /// function that resolves import module names to exporter harnesses.
    pub fn wireImportTrampolines(
        self: *Harness,
        resolve_exporter: anytype,
    ) void {
        const import_count = self.wasm_module.import_function_count;
        if (import_count == 0) return;

        // Count how many trampolines we need.
        var tramp_count: u32 = 0;
        var func_import_idx2: u32 = 0;
        for (self.wasm_module.imports) |imp| {
            if (imp.kind != .function) continue;
            defer func_import_idx2 += 1;
            if (func_import_idx2 >= import_count) break;
            const exp_h = resolve_exporter.get(imp.module_name) orelse continue;
            if (exp_h.persistent_vmctx == null) continue;
            tramp_count += 1;
        }
        if (tramp_count == 0) return;

        // Trampoline stub (22 bytes each on x86_64):
        //   movabs param_regs[0], <vmctx_ptr>   ; 10 bytes
        //   movabs rax, <func_ptr>              ; 10 bytes
        //   jmp rax                             ; 2 bytes
        const tramp_size: usize = 22;
        const page_size: usize = 4096;
        const total = ((tramp_count * tramp_size) + page_size - 1) & ~(page_size - 1);
        const mem = platform.mmap(null, total, .{ .read = true, .write = true }, .{}) orelse return;

        var offset: usize = 0;
        var func_import_idx3: u32 = 0;
        const hf_mut = @as([]?*const anyopaque, @constCast(self.inst.host_functions));
        for (self.wasm_module.imports) |imp| {
            if (imp.kind != .function) continue;
            defer func_import_idx3 += 1;
            if (func_import_idx3 >= import_count) break;
            const exp_h = resolve_exporter.get(imp.module_name) orelse continue;
            const pvmctx = exp_h.persistent_vmctx orelse continue;
            const real_ptr = if (func_import_idx3 < hf_mut.len)
                (if (hf_mut[func_import_idx3]) |p| @intFromPtr(p) else continue)
            else
                continue;

            const vmctx_addr: u64 = @intFromPtr(pvmctx);
            const func_addr: u64 = @as(u64, real_ptr);
            const stub = mem + offset;

            // movabs rcx/rdi, <vmctx_addr>
            const vmctx_reg_prefix: u8 = if (comptime builtin.os.tag == .windows) 0xB9 else 0xBF;
            stub[0] = 0x48;
            stub[1] = vmctx_reg_prefix;
            @as(*align(1) u64, @ptrCast(stub + 2)).* = vmctx_addr;
            // movabs rax, <func_addr>
            stub[10] = 0x48;
            stub[11] = 0xB8;
            @as(*align(1) u64, @ptrCast(stub + 12)).* = func_addr;
            // jmp rax
            stub[20] = 0xFF;
            stub[21] = 0xE0;

            hf_mut[func_import_idx3] = @ptrFromInt(@intFromPtr(stub));
            offset += tramp_size;
        }

        platform.mprotect(mem, total, .{ .read = true, .exec = true }) catch {
            platform.munmap(mem, total);
            return;
        };
        platform.icacheFlush(mem, total);
        self.trampoline_pages = mem;
        self.trampoline_size = total;
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
            const g = self.inst.globals[exp.index];
            const val = g.value;
            try registry.putGlobal(module_name, exp.name, val);
            // Share mutable globals by pointer so cross-module mutations
            // are visible (wasm spec requires mutable global imports to
            // alias the same storage as the exporter).
            if (g.global_type.mutability == .mutable) {
                try registry.putSharedGlobal(module_name, exp.name, g);
            }
            // For funcref globals, also resolve to the exporter's native
            // code pointer so importers can install it directly in their
            // local tables (elem_exprs with `global.get <imported funcref>`).
            switch (val) {
                .funcref, .nonfuncref => |maybe| if (maybe) |fidx| {
                    if (fidx < self.inst.funcptrs.len) {
                        const ptr = self.inst.funcptrs[fidx];
                        if (ptr != 0) {
                            const sid: u32 = if (fidx < self.inst.func_sig_ids.len) self.inst.func_sig_ids[fidx] else 0;
                            try registry.putFuncrefGlobal(module_name, exp.name, ptr, sid);
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

    /// Publish every exported table's `*TableInstance` pointer into
    /// `registry` keyed by `<module_name>/<export_name>`. Subsequent
    /// importers' `initWithRegistry` will install the shared pointer in
    /// their own `inst.tables` slot so cross-module `table.get`/`table.set`
    /// observe the same underlying elements array.
    pub fn exportTablesToRegistry(
        self: *const Harness,
        registry: *ImportRegistry,
        module_name: []const u8,
    ) !void {
        for (self.wasm_module.exports) |exp| {
            if (exp.kind != .table) continue;
            if (exp.index >= self.inst.tables.len) continue;
            try registry.putTable(module_name, exp.name, self.inst.tables[exp.index]);
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
    wasm_imports: []const types.ImportDesc,
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

    var import_count: u32 = 0;
    for (wasm_imports) |imp| {
        if (imp.kind == .table) import_count += 1;
    }

    const total = import_count + wasm_tables.len;
    const tables = try allocator.alloc(*types.TableInstance, total);
    errdefer allocator.free(tables);

    // Imported tables first (they occupy indices [0, import_count)).
    // Initialize with a fresh local TableInstance sized to the import's
    // declared `min`; `initWithRegistry` may swap in a shared exporter
    // instance if the registry has a matching entry.
    var idx: usize = 0;
    for (wasm_imports) |imp| {
        if (imp.kind != .table) continue;
        const tt = imp.table_type orelse return error.InstantiateFailed;
        const elements = try allocator.alloc(types.TableElement, tt.limits.min);
        for (elements) |*e| e.* = types.TableElement.nullForType(tt.elem_type);
        const tbl = try allocator.create(types.TableInstance);
        tbl.* = .{ .table_type = tt, .elements = elements };
        tables[idx] = tbl;
        idx += 1;
    }

    for (wasm_tables) |tt| {
        const elements = try allocator.alloc(types.TableElement, tt.limits.min);
        for (elements) |*e| e.* = types.TableElement.nullForType(tt.elem_type);
        const tbl = try allocator.create(types.TableInstance);
        tbl.* = .{ .table_type = tt, .elements = elements };
        tables[idx] = tbl;
        idx += 1;
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
            var resolved: ?ImportRegistry.FuncrefGlobalEntry = null;
            for (wasm_module.imports) |imp| {
                if (imp.kind != .global) continue;
                defer import_global_seen += 1;
                if (import_global_seen != gidx) continue;
                resolved = registry.getFuncrefGlobal(imp.module_name, imp.field_name);
                break;
            }
            const entry = resolved orelse continue;

            // Pick the native backing store for seg.table_idx.
            const dst_idx: usize = @as(usize, offset_val) + k;
            if (seg.table_idx == 0) {
                if (dst_idx < inst.func_table.len) inst.func_table[dst_idx] = entry.ptr;
            } else {
                const extra_idx = seg.table_idx - 1;
                if (extra_idx >= inst.extra_tables_storage.len) continue;
                const backing = inst.extra_tables_storage[extra_idx];
                if (dst_idx < backing.len) backing[dst_idx] = entry.ptr;
            }
            // Also write the sig_id to type_backing so call_indirect
            // sig checks see the correct value.
            if (seg.table_idx < inst.tables_info.len) {
                const ti = &inst.tables_info[seg.table_idx];
                if (ti.type_backing_ptr != 0 and dst_idx < ti.len) {
                    const tb: [*]u32 = @ptrFromInt(@as(usize, @intCast(ti.type_backing_ptr)));
                    tb[dst_idx] = entry.sig_id;
                }
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
        // emit_aot.ExternalKind lacks .tag (0x04); exception handling is not
        // supported in AOT. Drop tag exports rather than panicking on the enum
        // conversion below.
        if (exp.kind == .tag) continue;
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
        // Skip tag imports — AOT does not support exception handling, and
        // emit_aot.ExternalKind has no .tag variant.
        if (imp.kind == .tag) continue;
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
        if (seg.is_declarative) continue;
        const offset_val: u32 = if (seg.is_passive) 0 else blk: {
            const offset_expr = seg.offset orelse continue;
            break :blk resolveU32InitExpr(offset_expr, tmp_globals.items) orelse continue;
        };
        // Some elem segments (e.g. externref segments with `ref.null extern`
        // or `ref.extern K` expressions) carry their values in `elem_exprs`
        // rather than `func_indices`. Size the emitted index array to the
        // larger of the two so expression-only entries still emit a slot
        // (encoded as maxInt(u32) → null if unresolvable).
        const entry_count = @max(seg.func_indices.len, seg.elem_exprs.len);
        const indices = try a.alloc(u32, entry_count);
        for (0..entry_count) |k| {
            if (k < seg.func_indices.len) {
                if (seg.func_indices[k]) |v| {
                    indices[k] = v;
                    continue;
                }
            }
            // Fall back to resolving the init expression stored in
            // elem_exprs (e.g. `global.get K` where global K holds a
            // funcref produced by `ref.func N`). Unresolvable entries
            // (including `ref.null extern`, `ref.extern K`) encode as
            // maxInt(u32) → null slot at runtime.
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
            .is_passive = seg.is_passive,
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

    // Sig-check tables: per-module FuncType descriptors + per-local-func
    // type index. Required for call_indirect's inline sig equality check.
    var ft_entries: std.ArrayList(emit_aot.FuncTypeEntry) = .empty;
    for (module.types) |ft| {
        const pbytes = try a.alloc(u8, ft.params.len);
        for (ft.params, 0..) |vt, i| pbytes[i] = @intFromEnum(vt);
        const rbytes = try a.alloc(u8, ft.results.len);
        for (ft.results, 0..) |vt, i| rbytes[i] = @intFromEnum(vt);
        try ft_entries.append(a, .{ .params = pbytes, .results = rbytes });
    }
    const tidxs = try a.alloc(u32, module.functions.len);
    for (module.functions, 0..) |f, i| tidxs[i] = f.type_idx;

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
        if (ft_entries.items.len > 0) ft_entries.items else null,
        if (tidxs.len > 0) tidxs else null,
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