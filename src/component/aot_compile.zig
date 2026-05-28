//! Component / core-module AOT precompilation (compile-only half).
//!
//! Split out of `src/component/aot.zig` so the runtime CLI (`wamr`)
//! can link against the manifest loader without dragging in the
//! compiler crate. The runtime-only declarations
//! (`LoadedManifest`, `loadManifest`, `defaultManifestPathFor`,
//! manifest JSON schema, `isComponent`) live in `aot.zig`; this file
//! owns `compileCoreWasm`, `precompileComponent`, and the
//! `emit_aot` field-table builders that only the compile side uses.
//!
//! `Manifest` / `ManifestModuleEntry` / `manifest_format_version` are
//! re-exported from `aot.zig` here so the on-disk JSON shape stays
//! single-source-of-truth in `aot.zig`.

const std = @import("std");
const builtin = @import("builtin");
const ctypes = @import("types.zig");
const component_loader = @import("loader.zig");
const aot = @import("aot.zig");
const core_loader = @import("../runtime/interpreter/loader.zig");
const frontend = @import("../compiler/frontend.zig");
const passes = @import("../compiler/ir/passes.zig");
const x86_64_compile = @import("../compiler/codegen/x86_64/compile.zig");
const aarch64_compile = @import("../compiler/codegen/aarch64/compile.zig");
const emit_aot = @import("../compiler/emit_aot.zig");
const core_types = @import("../runtime/common/types.zig");
const interp_instance = @import("../runtime/interpreter/instance.zig");
const config = @import("../config.zig");

// Re-export the on-disk JSON schema from `aot.zig` so `precompileComponent`
// and `loadManifest` agree on layout without duplicating the schema.
pub const Manifest = aot.Manifest;
pub const ManifestModuleEntry = aot.ManifestModuleEntry;
pub const manifest_format_version = aot.manifest_format_version;

pub const PrecompileError = error{
    InvalidComponent,
    CoreCompileFailed,
    WriteFailed,
    OutOfMemory,
    OpenDirFailed,
    JsonSerializationFailed,
};

/// Options controlling `precompileComponent`. Today only the target
/// arch is exposed; future options (opt level, dump-ir hooks, …) slot
/// in here without breaking the API.
pub const PrecompileOptions = struct {
    target_arch: passes.TargetArch = switch (builtin.cpu.arch) {
        .aarch64 => .aarch64,
        else => .x86_64,
    },
};

/// AOT-compile a single core wasm module. Returns freshly-allocated
/// `.cwasm` bytes owned by the caller (free with `allocator.free`).
///
/// Pure data in / data out — no filesystem I/O — so the same helper
/// services both `precompileComponent` and any future programmatic
/// caller that wants per-core artifacts in memory.
pub fn compileCoreWasm(
    allocator: std.mem.Allocator,
    wasm_bytes: []const u8,
    opts: PrecompileOptions,
) PrecompileError![]u8 {
    // Lifetime split mirrors `src/compiler/main.zig` (`wamrc compile`)
    // to bound peak memory on large modules (issue #640). The parsed
    // module lives in a transient `module_arena`; IR + passes +
    // codegen go on the outer GPA so per-function intermediates are
    // freed back as each pass completes; emit-side field tables live
    // in a short-lived `emit_arena`. Lumping everything on one arena
    // (the previous implementation) retained tens of GB on a
    // 12 610-function core.
    var module_arena = std.heap.ArenaAllocator.init(allocator);
    defer module_arena.deinit();
    const ma = module_arena.allocator();

    // Don't dupe `wasm_bytes`: `core_loader.load` only borrows from
    // the slice (returned `module` fields are slices into it), and
    // the caller's bytes outlive this call.
    const module = core_loader.load(wasm_bytes, ma) catch return error.CoreCompileFailed;

    var ir_module = frontend.lowerModule(&module, allocator) catch return error.CoreCompileFailed;
    defer ir_module.deinit();

    _ = passes.runPassesWithOptions(
        &ir_module,
        passes.defaultPassesForTarget(opts.target_arch),
        allocator,
        .{ .verify_mode = .off },
    ) catch return error.CoreCompileFailed;

    const code: []u8, const offsets: []u32 = switch (opts.target_arch) {
        .aarch64 => blk: {
            const r = aarch64_compile.compileModule(&ir_module, allocator) catch return error.CoreCompileFailed;
            break :blk .{ r.code, r.offsets };
        },
        .x86_64 => blk: {
            const r = x86_64_compile.compileModule(&ir_module, allocator) catch return error.CoreCompileFailed;
            break :blk .{ r.code, r.offsets };
        },
    };
    defer allocator.free(code);
    defer allocator.free(offsets);

    // Short-lived arena for the emit-side field tables. `emit_aot.emit`
    // copies what it needs into its caller-owned output buffer, so
    // everything in here (and in `module_arena`) is safe to drop on
    // return.
    var emit_arena = std.heap.ArenaAllocator.init(allocator);
    defer emit_arena.deinit();
    const ea = emit_arena.allocator();

    var exports: std.ArrayList(emit_aot.ExportEntry) = .empty;
    for (module.exports) |exp| {
        if (exp.kind == .tag) continue;
        exports.append(ea, .{
            .name = exp.name,
            .kind = @enumFromInt(@intFromEnum(exp.kind)),
            .index = exp.index,
        }) catch return error.OutOfMemory;
    }

    var imports: std.ArrayList(emit_aot.ImportEntry) = .empty;
    for (module.imports) |imp| {
        switch (imp.kind) {
            .function => imports.append(ea, .{
                .module_name = imp.module_name,
                .field_name = imp.field_name,
                .kind = .function,
                .func_type_idx = imp.func_type_idx orelse 0,
            }) catch return error.OutOfMemory,
            .table => {
                const table_type = imp.table_type orelse continue;
                imports.append(ea, .{
                    .module_name = imp.module_name,
                    .field_name = imp.field_name,
                    .kind = .table,
                    .table_elem_type = table_type.elem_type,
                    .table_min = @intCast(table_type.limits.min),
                    .table_max = if (table_type.limits.max) |m| @as(?u32, @intCast(m)) else null,
                }) catch return error.OutOfMemory;
            },
            .memory => {
                const memory_type = imp.memory_type orelse continue;
                imports.append(ea, .{
                    .module_name = imp.module_name,
                    .field_name = imp.field_name,
                    .kind = .memory,
                    .memory_min = @intCast(memory_type.limits.min),
                    .memory_max = if (memory_type.limits.max) |m| @as(?u32, @intCast(m)) else null,
                    .memory_is64 = memory_type.is_memory64,
                }) catch return error.OutOfMemory;
            },
            .global => {
                const global_type = imp.global_type orelse continue;
                imports.append(ea, .{
                    .module_name = imp.module_name,
                    .field_name = imp.field_name,
                    .kind = .global,
                    .global_val_type = global_type.val_type,
                    .global_mutable = global_type.mutability == .mutable,
                }) catch return error.OutOfMemory;
            },
            .tag => {
                const type_idx = imp.tag_type_idx orelse continue;
                imports.append(ea, .{
                    .module_name = imp.module_name,
                    .field_name = imp.field_name,
                    .kind = .tag,
                    .tag_type_idx = type_idx,
                }) catch return error.OutOfMemory;
            },
        }
    }

    var mem_entries: std.ArrayList(emit_aot.MemoryEntry) = .empty;
    for (module.memories) |mem| {
        mem_entries.append(ea, .{
            .min_pages = @intCast(mem.limits.min),
            .max_pages = if (mem.limits.max) |m| @as(?u32, @intCast(m)) else null,
        }) catch return error.OutOfMemory;
    }

    // Locally-defined tables (#681). Imported tables already round-trip
    // via `import_entries`; local tables need section 15 so the loader
    // populates `module.tables` and `allocateTables` creates the matching
    // `TableInstance`s. wit-component cores rely on this for
    // `(table $imports)`.
    var table_entries: std.ArrayList(emit_aot.TableEntry) = .empty;
    for (module.tables) |t| {
        table_entries.append(ea, .{
            .elem_type = t.elem_type,
            .min = @intCast(t.limits.min),
            .max = if (t.limits.max) |m| @as(?u32, @intCast(m)) else null,
        }) catch return error.OutOfMemory;
    }

    var data_segs: std.ArrayList(emit_aot.DataSegmentEntry) = .empty;
    for (module.data_segments) |seg| {
        if (seg.is_passive) continue;
        const offset: u32 = switch (seg.offset) {
            .i32_const => |v| @bitCast(v),
            else => continue,
        };
        data_segs.append(ea, .{
            .memory_idx = seg.memory_idx,
            .offset = offset,
            .data = seg.data,
        }) catch return error.OutOfMemory;
    }

    var func_type_entries: std.ArrayList(emit_aot.FuncTypeEntry) = .empty;
    for (module.types) |ft| {
        if (ft.kind != .func) {
            func_type_entries.append(ea, .{ .params = &.{}, .results = &.{} }) catch return error.OutOfMemory;
            continue;
        }
        const params_bytes = ea.alloc(u8, ft.params.len) catch return error.OutOfMemory;
        for (ft.params, 0..) |p, j| params_bytes[j] = @intFromEnum(p);
        const results_bytes = ea.alloc(u8, ft.results.len) catch return error.OutOfMemory;
        for (ft.results, 0..) |r, j| results_bytes[j] = @intFromEnum(r);
        func_type_entries.append(ea, .{ .params = params_bytes, .results = results_bytes }) catch return error.OutOfMemory;
    }

    var local_func_tidx_list: std.ArrayList(u32) = .empty;
    for (module.functions) |f| {
        local_func_tidx_list.append(ea, f.type_idx) catch return error.OutOfMemory;
    }

    // Locally-declared tags (#672). `module.tag_types` is parallel to the
    // wasm tag section: each entry is a function-type index describing the
    // exception parameters.
    var tag_entries: std.ArrayList(emit_aot.TagEntry) = .empty;
    for (module.tag_types) |type_idx| {
        tag_entries.append(ea, .{ .type_idx = type_idx }) catch return error.OutOfMemory;
    }

    // Build global entries in wasm-flat order (imported globals first,
    // then local globals). `evalInitExpr` is shared with the on-disk
    // wamrc path so the resulting `.cwasm` is byte-identical for this
    // section. `tmp_globals` is the running prefix `evalInitExpr` uses
    // to resolve `global.get` in a `global` init expression.
    var global_entries: std.ArrayList(emit_aot.GlobalEntry) = .empty;
    var tmp_globals: std.ArrayList(*core_types.GlobalInstance) = .empty;
    defer {
        for (tmp_globals.items) |g| allocator.destroy(g);
        tmp_globals.deinit(allocator);
    }
    for (module.imports) |imp| {
        if (imp.kind != .global) continue;
        const gt = imp.global_type orelse continue;
        const val = defaultZeroValue(gt.val_type);
        const gi = allocator.create(core_types.GlobalInstance) catch return error.OutOfMemory;
        gi.* = .{ .global_type = gt, .value = val };
        tmp_globals.append(allocator, gi) catch return error.OutOfMemory;
        global_entries.append(ea, .{
            .val_type = @intFromEnum(gt.val_type),
            .mutability = if (gt.mutability == .mutable) @as(u8, 1) else @as(u8, 0),
            .init_i64 = valueToI64(val),
            .init_v128 = valueToV128(val),
        }) catch return error.OutOfMemory;
    }
    for (module.globals) |g| {
        const val = interp_instance.evalInitExpr(g.init_expr, tmp_globals.items, null) catch defaultZeroValue(g.global_type.val_type);
        const gi = allocator.create(core_types.GlobalInstance) catch return error.OutOfMemory;
        gi.* = .{ .global_type = g.global_type, .value = val };
        tmp_globals.append(allocator, gi) catch return error.OutOfMemory;
        global_entries.append(ea, .{
            .val_type = @intFromEnum(g.global_type.val_type),
            .mutability = if (g.global_type.mutability == .mutable) @as(u8, 1) else @as(u8, 0),
            .init_i64 = valueToI64(val),
            .init_v128 = valueToV128(val),
        }) catch return error.OutOfMemory;
    }

    // Build element segment entries. Declarative segments contribute
    // nothing at runtime; passive segments emit with offset = 0 and
    // `is_passive = true` (only usable via `table.init`). Funcidx
    // null sentinels are encoded as `0xFFFFFFFF`, matching the loader.
    var elem_entries: std.ArrayList(emit_aot.ElemEntry) = .empty;
    for (module.elements) |seg| {
        if (seg.is_declarative) continue;
        const offset: u32 = if (seg.is_passive) 0 else blk: {
            const off = seg.offset orelse continue;
            break :blk switch (off) {
                .i32_const => |v| @as(u32, @bitCast(v)),
                else => continue,
            };
        };
        const indices = ea.alloc(u32, seg.func_indices.len) catch return error.OutOfMemory;
        for (seg.func_indices, 0..) |fi, j| {
            indices[j] = fi orelse 0xFFFFFFFF;
        }
        elem_entries.append(ea, .{
            .table_idx = seg.table_idx,
            .offset = offset,
            .func_indices = indices,
            .is_passive = seg.is_passive,
        }) catch return error.OutOfMemory;
    }

    var arch_name = std.mem.zeroes([16]u8);
    switch (opts.target_arch) {
        .x86_64 => @memcpy(arch_name[0..6], "x86-64"),
        .aarch64 => @memcpy(arch_name[0..7], "aarch64"),
    }

    return emit_aot.emit(
        allocator,
        code,
        offsets,
        exports.items,
        .{ .arch = arch_name },
        if (data_segs.items.len > 0) data_segs.items else null,
        if (imports.items.len > 0) imports.items else null,
        if (mem_entries.items.len > 0) mem_entries.items else null,
        // Plain wasm32-wasip1 modules (AssemblyScript / Rust / clang
        // wasi-libc) carry a mutable `i32` shadow stack pointer global
        // that every function prologue reads/writes; failing to emit it
        // leaves SP=0 at startup and AS aborts inside its runtime with
        // `abort:  in (1:1)`. Componentize-js cores tolerate empty
        // globals/elems because canon-lift wraps them; standalone cores
        // do not. See `phase-b-diagnosis.md` (#662 Phase B).
        if (global_entries.items.len > 0) global_entries.items else null,
        if (elem_entries.items.len > 0) elem_entries.items else null,
        module.start_function,
        if (func_type_entries.items.len > 0) func_type_entries.items else null,
        if (local_func_tidx_list.items.len > 0) local_func_tidx_list.items else null,
        if (tag_entries.items.len > 0) tag_entries.items else null,
        if (table_entries.items.len > 0) table_entries.items else null,
    ) catch return error.CoreCompileFailed;
}

// ── Global / element entry builders ─────────────────────────────────────
//
// Mirrors `src/compiler/main.zig`'s `runCompile` so the in-process AOT
// compile produces the same `.cwasm` layout the on-disk path does. The
// helpers here are deliberately self-contained (no cross-imports of
// file-private functions in `compiler/main.zig`) to keep this fix
// surgical; the duplication can be deduplicated in a follow-up by
// hoisting them into `emit_aot.zig` or a new shared file.

fn defaultZeroValue(vt: core_types.ValType) core_types.Value {
    return switch (vt) {
        .i32 => .{ .i32 = 0 },
        .i64 => .{ .i64 = 0 },
        .f32 => .{ .f32 = 0 },
        .f64 => .{ .f64 = 0 },
        .v128 => .{ .v128 = 0 },
        .funcref => .{ .funcref = null },
        .externref => .{ .externref = null },
        .nonfuncref => .{ .nonfuncref = null },
        .nonexternref => .{ .nonexternref = null },
        else => .{ .i64 = 0 },
    };
}

fn valueToI64(v: core_types.Value) i64 {
    return switch (v) {
        .i32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .i64 => |x| x,
        .f32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .f64 => |x| @as(i64, @bitCast(x)),
        .funcref, .nonfuncref => |maybe| if (maybe) |x| @as(i64, @as(u32, x)) + 1 else 0,
        .externref, .nonexternref => |maybe| if (maybe) |x| @as(i64, @as(u32, x)) + 1 else 0,
        else => 0,
    };
}

fn valueToV128(v: core_types.Value) u128 {
    return switch (v) {
        .v128 => |x| x,
        else => 0,
    };
}

/// Hex-encode a sha256 digest into a fresh allocator-owned string.
fn hexSha256(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = try allocator.alloc(u8, digest.len * 2);
    const hex_chars = "0123456789abcdef";
    for (digest, 0..) |b, i| {
        hex[i * 2] = hex_chars[b >> 4];
        hex[i * 2 + 1] = hex_chars[b & 0x0f];
    }
    return hex;
}

/// Result of `precompileComponent`. The returned `Manifest`'s strings
/// are arena-allocated and freed by `deinit` together with the arena.
pub const PrecompileResult = struct {
    manifest: Manifest,
    arena: *std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PrecompileResult) void {
        self.arena.deinit();
        self.allocator.destroy(self.arena);
    }
};

/// Precompile every embedded core module of `component_bytes` and
/// write them as `<stem>.<idx>.cwasm` next to `manifest_path`, with
/// the manifest JSON itself at `manifest_path`. The "stem" is the
/// basename of `manifest_path` with the `.cwasm.json` suffix stripped
/// (or the whole basename if the suffix is absent); cores share that
/// stem so multiple components can coexist in the same directory.
///
/// The parent directory of `manifest_path` is created if missing.
/// Existing files at the target paths are overwritten — callers
/// responsible for stale-cache management beyond the manifest's
/// build-id check.
pub fn precompileComponent(
    allocator: std.mem.Allocator,
    component_bytes: []const u8,
    manifest_path: []const u8,
    opts: PrecompileOptions,
) PrecompileError!PrecompileResult {
    // Snapshot per-core byte slices off the parsed component, then
    // drop `load_arena` before the compile loop so the parsed
    // component (large for componentize-js workloads) isn't live in
    // memory while each core is being AOT-compiled. The `core_mod.data`
    // slices borrow from `component_bytes` (owned by the caller), so
    // they remain valid after `load_arena.deinit()`.
    //
    // Recurses into `component.components` so cores inside nested
    // sub-components (the dominant shape of `wabt component compose -d`
    // / `wasm-tools compose` output) are precompiled too — without
    // this the manifest is always empty for composed components and
    // `--precompiled-manifest` falls back to in-memory AOT on every
    // cold start. (#676)
    const CoreRef = struct {
        data: []const u8,
        /// Position within the `core_modules[]` of the (sub-)component
        /// that directly contains this core. Persisted as
        /// `ManifestModuleEntry.idx` for human debugging and for the
        /// v1-fallback loader path.
        local_idx: u32,
    };
    var core_data_list: std.ArrayList(CoreRef) = .empty;
    defer core_data_list.deinit(allocator);
    {
        var load_arena = std.heap.ArenaAllocator.init(allocator);
        defer load_arena.deinit();
        const component = component_loader.load(component_bytes, load_arena.allocator()) catch
            return error.InvalidComponent;
        const Walker = struct {
            fn walk(
                comp: *const ctypes.Component,
                list: *std.ArrayList(CoreRef),
                alloc: std.mem.Allocator,
            ) PrecompileError!void {
                for (comp.core_modules, 0..) |cm, mi| {
                    list.append(alloc, .{ .data = cm.data, .local_idx = @intCast(mi) }) catch
                        return error.OutOfMemory;
                }
                for (comp.components) |child| {
                    try walk(child, list, alloc);
                }
            }
        };
        try Walker.walk(&component, &core_data_list, allocator);
    }

    // Result-owned arena holds all strings the caller might read off
    // the returned Manifest (paths, hex hashes, build id).
    const result_arena = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(result_arena);
    result_arena.* = std.heap.ArenaAllocator.init(allocator);
    errdefer result_arena.deinit();
    const ra = result_arena.allocator();

    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const parent_dir_path = std.fs.path.dirname(manifest_path) orelse ".";
    const manifest_filename = std.fs.path.basename(manifest_path);
    const core_stem = if (std.mem.endsWith(u8, manifest_filename, ".cwasm.json"))
        manifest_filename[0 .. manifest_filename.len - ".cwasm.json".len]
    else
        manifest_filename;
    cwd.createDirPath(io, parent_dir_path) catch return error.OpenDirFailed;
    var dir = cwd.openDir(io, parent_dir_path, .{}) catch return error.OpenDirFailed;
    defer dir.close(io);

    var entries: std.ArrayList(ManifestModuleEntry) = .empty;
    entries.ensureTotalCapacity(ra, core_data_list.items.len) catch return error.OutOfMemory;

    for (core_data_list.items, 0..) |core_ref, idx| {
        const cwasm = compileCoreWasm(allocator, core_ref.data, opts) catch |err| {
            std.log.err("precompileComponent: core {d} compile failed: {s}", .{ idx, @errorName(err) });
            return err;
        };
        defer allocator.free(cwasm);

        const rel_path = std.fmt.allocPrint(ra, "{s}.{d}.cwasm", .{ core_stem, idx }) catch return error.OutOfMemory;

        dir.writeFile(io, .{ .sub_path = rel_path, .data = cwasm }) catch |err| {
            std.log.err("precompileComponent: write {s}/{s} failed: {s}", .{ parent_dir_path, rel_path, @errorName(err) });
            return error.WriteFailed;
        };

        const hex = hexSha256(ra, cwasm) catch return error.OutOfMemory;
        const core_hex = hexSha256(ra, core_ref.data) catch return error.OutOfMemory;
        entries.append(ra, .{
            .idx = core_ref.local_idx,
            .path = rel_path,
            .sha256 = hex,
            .core_sha256 = core_hex,
        }) catch return error.OutOfMemory;
    }

    const component_hex = hexSha256(ra, component_bytes) catch return error.OutOfMemory;
    const build_id = ra.dupe(u8, config.version) catch return error.OutOfMemory;

    const manifest = Manifest{
        .version = manifest_format_version,
        .wamr_build_id = build_id,
        .component_sha256 = component_hex,
        .modules = entries.items,
    };

    // Serialize the manifest sidecar.
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var stringify: std.json.Stringify = .{ .writer = &aw.writer, .options = .{ .whitespace = .indent_2 } };
    stringify.write(manifest) catch return error.JsonSerializationFailed;
    dir.writeFile(io, .{ .sub_path = manifest_filename, .data = aw.written() }) catch return error.WriteFailed;

    return .{
        .manifest = manifest,
        .arena = result_arena,
        .allocator = allocator,
    };
}
