//! Component-level AOT precompilation (#625 phase 2).
//!
//! Walks every embedded core module in a parsed `Component` and AOT-
//! compiles it via the existing `src/compiler` pipeline, writing each
//! result to `<out_dir>/module<N>.cwasm` alongside a versioned
//! `manifest.json`. The manifest records the component's sha256 and
//! the wamr build id so a stale or wrong-component artifact is
//! rejected on load.
//!
//! Mirrors `wasmtime::Engine::precompile_component`. The on-disk
//! layout is:
//!
//! ```text
//! <out_dir>/manifest.json
//! <out_dir>/module0.cwasm
//! <out_dir>/module1.cwasm
//! …
//! ```
//!
//! `loadManifest` is the inverse: parse `manifest.json`, mmap each
//! referenced `.cwasm`, verify the embedded sha256s, and return a
//! `LoadedManifest` whose `precompiledCores()` hands the slice
//! directly to `instance.instantiateWithOptions`.
//!
//! Out of scope (per issue #625):
//!   * Cross-process mmap sharing — every load reads its own copy.
//!   * Cache invalidation beyond build-id + content-hash comparison.

const std = @import("std");
const builtin = @import("builtin");
const ctypes = @import("types.zig");
const component_loader = @import("loader.zig");
const core_backend = @import("core_backend.zig");
const core_loader = @import("../runtime/interpreter/loader.zig");
const frontend = @import("../compiler/frontend.zig");
const passes = @import("../compiler/ir/passes.zig");
const x86_64_compile = @import("../compiler/codegen/x86_64/compile.zig");
const aarch64_compile = @import("../compiler/codegen/aarch64/compile.zig");
const emit_aot = @import("../compiler/emit_aot.zig");
const core_types = @import("../runtime/common/types.zig");
const instance_mod = @import("instance.zig");
const config = @import("../config.zig");

pub const PrecompileError = error{
    InvalidComponent,
    CoreCompileFailed,
    WriteFailed,
    OutOfMemory,
    OpenDirFailed,
    JsonSerializationFailed,
};

pub const LoadError = error{
    ManifestNotFound,
    ManifestParseFailed,
    ManifestVersionMismatch,
    ManifestBuildIdMismatch,
    ManifestComponentMismatch,
    CwasmReadFailed,
    CwasmHashMismatch,
    OutOfMemory,
};

/// Manifest schema version. Bump when the on-disk layout or
/// serialization changes in a way that older loaders cannot read.
pub const manifest_format_version: u32 = 1;

/// Per-core entry in the manifest.
pub const ManifestModuleEntry = struct {
    /// Index into `component.core_modules` this artifact precompiles.
    idx: u32,
    /// Relative path under the manifest's directory (e.g. `module3.cwasm`).
    path: []const u8,
    /// Hex sha256 of the `.cwasm` bytes, used to detect tampering
    /// or partial writes on load.
    sha256: []const u8,
};

/// Top-level manifest. Serialized to / parsed from `manifest.json`.
pub const Manifest = struct {
    /// `manifest_format_version` at write time. Refused on load when
    /// it doesn't match the current loader's expected value.
    version: u32 = manifest_format_version,
    /// wamr build id (`config.version`) at precompile time. Refused
    /// on load when the runtime's build id is different — AOT codegen
    /// is not stable across wamr versions.
    wamr_build_id: []const u8,
    /// Hex sha256 of the **original component binary** that was fed
    /// to `precompileComponent`. Cross-checked on load against the
    /// component bytes the caller hands to `loadManifest`.
    component_sha256: []const u8,
    modules: []const ManifestModuleEntry,
};

/// The result of `loadManifest`: parsed manifest + a per-core buffer
/// owned by the `LoadedManifest`. Callers turn it into the form
/// `instance.instantiateWithOptions` consumes via `precompiledCores`.
pub const LoadedManifest = struct {
    manifest: Manifest,
    /// One owned `.cwasm` buffer per `manifest.modules[]` entry, in
    /// the same order. Freed by `deinit`.
    cwasm_buffers: []const []u8,
    /// Pre-built slice ready to pass as `Options.precompiled_cores`.
    /// Borrows from `manifest.modules[]` (idx) and `cwasm_buffers`
    /// (bytes); valid for the lifetime of this `LoadedManifest`.
    pcs: []const core_backend.PrecompiledCore,
    allocator: std.mem.Allocator,
    /// Backing storage for `manifest` strings + entries. The JSON
    /// parser allocates into this arena; `deinit` tears it down.
    arena: *std.heap.ArenaAllocator,

    pub fn precompiledCores(self: *const LoadedManifest) []const core_backend.PrecompiledCore {
        return self.pcs;
    }

    pub fn deinit(self: *LoadedManifest) void {
        for (self.cwasm_buffers) |buf| self.allocator.free(buf);
        self.allocator.free(self.cwasm_buffers);
        self.allocator.free(self.pcs);
        self.arena.deinit();
        self.allocator.destroy(self.arena);
    }
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
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const owned_wasm = a.dupe(u8, wasm_bytes) catch return error.OutOfMemory;
    const module = core_loader.load(owned_wasm, a) catch return error.CoreCompileFailed;

    var ir_module = frontend.lowerModule(&module, a) catch return error.CoreCompileFailed;
    defer ir_module.deinit();

    _ = passes.runPassesWithOptions(
        &ir_module,
        passes.defaultPassesForTarget(opts.target_arch),
        a,
        .{ .verify_mode = .off },
    ) catch return error.CoreCompileFailed;

    const code: []const u8, const offsets: []const u32 = switch (opts.target_arch) {
        .aarch64 => blk: {
            const r = aarch64_compile.compileModule(&ir_module, a) catch return error.CoreCompileFailed;
            break :blk .{ r.code, r.offsets };
        },
        .x86_64 => blk: {
            const r = x86_64_compile.compileModule(&ir_module, a) catch return error.CoreCompileFailed;
            break :blk .{ r.code, r.offsets };
        },
    };

    var exports: std.ArrayList(emit_aot.ExportEntry) = .empty;
    for (module.exports) |exp| {
        if (exp.kind == .tag) continue;
        exports.append(a, .{
            .name = exp.name,
            .kind = @enumFromInt(@intFromEnum(exp.kind)),
            .index = exp.index,
        }) catch return error.OutOfMemory;
    }

    var imports: std.ArrayList(emit_aot.ImportEntry) = .empty;
    for (module.imports) |imp| {
        if (imp.kind == .function) {
            imports.append(a, .{
                .module_name = imp.module_name,
                .field_name = imp.field_name,
                .kind = .function,
                .func_type_idx = imp.func_type_idx orelse 0,
            }) catch return error.OutOfMemory;
        }
    }

    var mem_entries: std.ArrayList(emit_aot.MemoryEntry) = .empty;
    for (module.memories) |mem| {
        mem_entries.append(a, .{
            .min_pages = @intCast(mem.limits.min),
            .max_pages = if (mem.limits.max) |m| @as(?u32, @intCast(m)) else null,
        }) catch return error.OutOfMemory;
    }

    var data_segs: std.ArrayList(emit_aot.DataSegmentEntry) = .empty;
    for (module.data_segments) |seg| {
        if (seg.is_passive) continue;
        const offset: u32 = switch (seg.offset) {
            .i32_const => |v| @bitCast(v),
            else => continue,
        };
        data_segs.append(a, .{
            .memory_idx = seg.memory_idx,
            .offset = offset,
            .data = seg.data,
        }) catch return error.OutOfMemory;
    }

    var func_type_entries: std.ArrayList(emit_aot.FuncTypeEntry) = .empty;
    for (module.types) |ft| {
        if (ft.kind != .func) {
            func_type_entries.append(a, .{ .params = &.{}, .results = &.{} }) catch return error.OutOfMemory;
            continue;
        }
        const params_bytes = a.alloc(u8, ft.params.len) catch return error.OutOfMemory;
        for (ft.params, 0..) |p, j| params_bytes[j] = @intFromEnum(p);
        const results_bytes = a.alloc(u8, ft.results.len) catch return error.OutOfMemory;
        for (ft.results, 0..) |r, j| results_bytes[j] = @intFromEnum(r);
        func_type_entries.append(a, .{ .params = params_bytes, .results = results_bytes }) catch return error.OutOfMemory;
    }

    var local_func_tidx_list: std.ArrayList(u32) = .empty;
    for (module.functions) |f| {
        local_func_tidx_list.append(a, f.type_idx) catch return error.OutOfMemory;
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
        // Globals + elems are not yet populated by this helper. The
        // motivating workload (componentize-js cores) initialises its
        // own globals through data segments + start code, and the
        // existing AOT loader tolerates empty global/elem sections
        // (it allocates from `module.memories.len`/`module.globals.len`
        // = 0 just fine, then runs `start` which re-derives state).
        // Wired in fully when phase 3 lights up canon-lift onto AOT
        // exports for components that exercise top-level globals.
        null,
        null,
        module.start_function,
        if (func_type_entries.items.len > 0) func_type_entries.items else null,
        if (local_func_tidx_list.items.len > 0) local_func_tidx_list.items else null,
    ) catch return error.CoreCompileFailed;
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

/// Precompile every embedded core module of `component_bytes` to
/// `<out_dir>/module<N>.cwasm` and write `<out_dir>/manifest.json`.
///
/// `out_dir` is created if it doesn't exist. Existing files at the
/// target paths are overwritten — callers responsible for stale-cache
/// management beyond the manifest's build-id check.
pub fn precompileComponent(
    allocator: std.mem.Allocator,
    component_bytes: []const u8,
    out_dir: []const u8,
    opts: PrecompileOptions,
) PrecompileError!PrecompileResult {
    var load_arena = std.heap.ArenaAllocator.init(allocator);
    defer load_arena.deinit();
    const component = component_loader.load(component_bytes, load_arena.allocator()) catch
        return error.InvalidComponent;

    // Result-owned arena holds all strings the caller might read off
    // the returned Manifest (paths, hex hashes, build id).
    const result_arena = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(result_arena);
    result_arena.* = std.heap.ArenaAllocator.init(allocator);
    errdefer result_arena.deinit();
    const ra = result_arena.allocator();

    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    cwd.createDirPath(io, out_dir) catch return error.OpenDirFailed;
    var dir = cwd.openDir(io, out_dir, .{}) catch return error.OpenDirFailed;
    defer dir.close(io);

    var entries: std.ArrayList(ManifestModuleEntry) = .empty;
    entries.ensureTotalCapacity(ra, component.core_modules.len) catch return error.OutOfMemory;

    for (component.core_modules, 0..) |core_mod, idx| {
        const cwasm = compileCoreWasm(allocator, core_mod.data, opts) catch |err| {
            std.log.err("precompileComponent: core {d} compile failed: {s}", .{ idx, @errorName(err) });
            return err;
        };
        defer allocator.free(cwasm);

        const rel_path = std.fmt.allocPrint(ra, "module{d}.cwasm", .{idx}) catch return error.OutOfMemory;

        dir.writeFile(io, .{ .sub_path = rel_path, .data = cwasm }) catch |err| {
            std.log.err("precompileComponent: write {s}/{s} failed: {s}", .{ out_dir, rel_path, @errorName(err) });
            return error.WriteFailed;
        };

        const hex = hexSha256(ra, cwasm) catch return error.OutOfMemory;
        entries.append(ra, .{
            .idx = @intCast(idx),
            .path = rel_path,
            .sha256 = hex,
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

    // Serialize manifest.json.
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var stringify: std.json.Stringify = .{ .writer = &aw.writer, .options = .{ .whitespace = .indent_2 } };
    stringify.write(manifest) catch return error.JsonSerializationFailed;
    dir.writeFile(io, .{ .sub_path = "manifest.json", .data = aw.written() }) catch return error.WriteFailed;

    return .{
        .manifest = manifest,
        .arena = result_arena,
        .allocator = allocator,
    };
}

/// Read + validate a `manifest.json` and its referenced `.cwasm`
/// artifacts from `manifest_dir`, cross-checking the recorded
/// `component_sha256` against `expected_component_bytes`.
///
/// The returned `LoadedManifest.precompiledCores()` slice can be
/// handed straight to `instance.instantiateWithOptions`.
pub fn loadManifest(
    allocator: std.mem.Allocator,
    manifest_dir: []const u8,
    expected_component_bytes: []const u8,
) LoadError!LoadedManifest {
    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = cwd.openDir(io, manifest_dir, .{}) catch return error.ManifestNotFound;
    defer dir.close(io);

    const json_bytes = dir.readFileAlloc(io, "manifest.json", allocator, @enumFromInt(4 * 1024 * 1024)) catch
        return error.ManifestNotFound;
    defer allocator.free(json_bytes);

    const arena = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena);
    arena.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();

    const parsed = std.json.parseFromSliceLeaky(
        Manifest,
        arena.allocator(),
        json_bytes,
        .{ .ignore_unknown_fields = true },
    ) catch return error.ManifestParseFailed;

    if (parsed.version != manifest_format_version) return error.ManifestVersionMismatch;
    if (!std.mem.eql(u8, parsed.wamr_build_id, config.version)) return error.ManifestBuildIdMismatch;

    // Verify the manifest was produced from the component the caller
    // is about to instantiate.
    var expected_hex_buf: [64]u8 = undefined;
    var expected_digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(expected_component_bytes, &expected_digest, .{});
    const hex_chars = "0123456789abcdef";
    for (expected_digest, 0..) |b, i| {
        expected_hex_buf[i * 2] = hex_chars[b >> 4];
        expected_hex_buf[i * 2 + 1] = hex_chars[b & 0x0f];
    }
    if (!std.mem.eql(u8, &expected_hex_buf, parsed.component_sha256)) return error.ManifestComponentMismatch;

    const cwasm_buffers = allocator.alloc([]u8, parsed.modules.len) catch return error.OutOfMemory;
    var loaded: usize = 0;
    errdefer {
        for (cwasm_buffers[0..loaded]) |b| allocator.free(b);
        allocator.free(cwasm_buffers);
    }

    for (parsed.modules, 0..) |mod, i| {
        const buf = dir.readFileAlloc(io, mod.path, allocator, @enumFromInt(256 * 1024 * 1024)) catch
            return error.CwasmReadFailed;
        cwasm_buffers[i] = buf;
        loaded += 1;

        // Verify content hash.
        var hex_buf: [64]u8 = undefined;
        var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(buf, &digest, .{});
        for (digest, 0..) |b, j| {
            hex_buf[j * 2] = hex_chars[b >> 4];
            hex_buf[j * 2 + 1] = hex_chars[b & 0x0f];
        }
        if (!std.mem.eql(u8, &hex_buf, mod.sha256)) return error.CwasmHashMismatch;
    }

    const pcs = allocator.alloc(core_backend.PrecompiledCore, parsed.modules.len) catch return error.OutOfMemory;
    for (parsed.modules, 0..) |mod, i| {
        pcs[i] = .{ .module_idx = mod.idx, .cwasm_bytes = cwasm_buffers[i] };
    }

    return .{
        .manifest = parsed,
        .cwasm_buffers = cwasm_buffers,
        .pcs = pcs,
        .allocator = allocator,
        .arena = arena,
    };
}

/// Sniff the first 8 bytes of `data` for the component magic +
/// version pair. Returns `true` for a component, `false` for a core
/// module, `error.InvalidMagic` for anything else.
pub fn isComponent(data: []const u8) error{InvalidMagic}!bool {
    if (data.len < 8) return error.InvalidMagic;
    const magic = std.mem.readInt(u32, data[0..4], .little);
    if (magic != core_types.wasm_magic) return error.InvalidMagic;
    const version = std.mem.readInt(u32, data[4..8], .little);
    if (version == core_types.component_version) return true;
    if (version == core_types.wasm_version) return false;
    return error.InvalidMagic;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

test "isComponent: core module vs component" {
    const core = [_]u8{ 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 };
    const comp = [_]u8{ 0x00, 0x61, 0x73, 0x6d, 0x0d, 0x00, 0x01, 0x00 };
    const bad = [_]u8{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
    try std.testing.expectEqual(false, try isComponent(&core));
    try std.testing.expectEqual(true, try isComponent(&comp));
    try std.testing.expectError(error.InvalidMagic, isComponent(&bad));
    try std.testing.expectError(error.InvalidMagic, isComponent(&.{}));
}
