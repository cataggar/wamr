//! Component-level AOT precompilation (#625 phase 2).
//!
//! Walks every embedded core module in a parsed `Component` and AOT-
//! compiles it via the existing `src/compiler` pipeline, writing each
//! result next to the source `.wasm` as `<stem>.<idx>.cwasm`, with a
//! versioned `<stem>.cwasm.json` manifest sidecar. The manifest
//! records the component's sha256 and the wamr build id so a stale
//! or wrong-component artifact is rejected on load.
//!
//! Mirrors `wasmtime::Engine::precompile_component`. The on-disk
//! layout for `foo.wasm` is:
//!
//! ```text
//! foo.wasm            (source component)
//! foo.cwasm.json      (manifest sidecar)
//! foo.0.cwasm         (per-core AOT artifacts)
//! foo.1.cwasm
//! …
//! ```
//!
//! `loadManifest` is the inverse: parse `<stem>.cwasm.json`, mmap
//! each referenced `.cwasm`, verify the embedded sha256s, and return
//! a `LoadedManifest` whose `precompiledCores()` hands the slice
//! directly to `instance.instantiateWithOptions`.
//!
//! Out of scope (per issue #625):
//!   * Cross-process mmap sharing — every load reads its own copy.
//!   * Cache invalidation beyond build-id + content-hash comparison.

const std = @import("std");
const ctypes = @import("types.zig");
const component_loader = @import("loader.zig");
const core_backend = @import("core_backend.zig");
const name_section_mod = @import("../runtime/common/name_section.zig");
const core_types = @import("../runtime/common/types.zig");
const config = @import("../config.zig");

pub const LoadError = error{
    ManifestNotFound,
    ManifestParseFailed,
    ManifestVersionMismatch,
    ManifestBuildIdMismatch,
    ManifestComponentMismatch,
    /// v2 manifest references a `core_sha256` that no core in the
    /// supplied component bytes matches (or vice versa) — the
    /// manifest was produced from a different component shape than
    /// the one being loaded.
    ManifestCoreMismatch,
    /// v2 loader couldn't re-parse `expected_component_bytes` to
    /// walk its nested-component tree.
    ComponentParseFailed,
    CwasmReadFailed,
    CwasmHashMismatch,
    OutOfMemory,
};

/// Convention for the default manifest sidecar path next to a
/// component file: strip a trailing `.wasm` (if present) and append
/// `.cwasm.json`. Used both by `wamrc compile-component` (default
/// `-o`) and by `wamr run`'s auto-detect probe — they must stay in
/// sync (issue #645). Caller owns the returned slice.
pub fn defaultManifestPathFor(allocator: std.mem.Allocator, in_path: []const u8) ![]u8 {
    const stem = if (std.mem.endsWith(u8, in_path, ".wasm"))
        in_path[0 .. in_path.len - ".wasm".len]
    else
        in_path;
    return std.mem.concat(allocator, u8, &.{ stem, ".cwasm.json" });
}

/// Manifest schema version. Bump when the on-disk layout or
/// serialization changes in a way that older loaders cannot read.
///
/// History:
///   * v1 — single top-level loop, one `ManifestModuleEntry` per
///     `component.core_modules[idx]`. No `core_sha256`; matching at
///     load time was `module_idx`-only against the root component.
///   * v2 — recursive walker; one entry per *leaf* core anywhere in
///     the nested-component tree. `core_sha256` is the primary key.
///     Required for `wabt component compose -d` /
///     `wasm-tools compose` output whose cores live inside
///     sub-components. (#676)
pub const manifest_format_version: u32 = 2;

/// Per-core entry in the manifest.
pub const ManifestModuleEntry = struct {
    /// Leaf-local core-module index — i.e. the position within the
    /// `core_modules` slice of the (sub-)component that *directly*
    /// contains this core. Used by the v1 fallback loader (which
    /// matches against the root component); v2 manifests carry it
    /// for debugging only and match by `core_sha256` instead.
    idx: u32,
    /// Relative path under the manifest's directory (e.g. `module3.cwasm`).
    path: []const u8,
    /// Hex sha256 of the `.cwasm` bytes, used to detect tampering
    /// or partial writes on load.
    sha256: []const u8,
    /// Hex sha256 of the **raw core-module bytes** (`core_mod.data`)
    /// that produced this cwasm. Primary key for cross-parse
    /// identity in v2 manifests — `loadManifest` recomputes this
    /// per visited core in the live component tree and matches
    /// against this field. Null/empty on v1-shaped JSON read from
    /// disk (back-compat for fixtures persisted before #676).
    core_sha256: ?[]const u8 = null,
};

/// Top-level manifest. Serialized to / parsed from `<stem>.cwasm.json`.
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
///
/// Lives in `aot_compile.zig`; aliased here for callers that only
/// hold a `wamr.component_aot` import.
pub const PrecompileOptions = @import("aot_compile.zig").PrecompileOptions;

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

/// Read + validate a `<stem>.cwasm.json` manifest sidecar and its
/// referenced `.cwasm` artifacts (resolved relative to the manifest
/// file's parent directory), cross-checking the recorded
/// `component_sha256` against `expected_component_bytes`.
///
/// The returned `LoadedManifest.precompiledCores()` slice can be
/// handed straight to `instance.instantiateWithOptions`.
pub fn loadManifest(
    allocator: std.mem.Allocator,
    manifest_path: []const u8,
    expected_component_bytes: []const u8,
) LoadError!LoadedManifest {
    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const parent = std.fs.path.dirname(manifest_path) orelse ".";
    const filename = std.fs.path.basename(manifest_path);
    var dir = cwd.openDir(io, parent, .{}) catch return error.ManifestNotFound;
    defer dir.close(io);

    const json_bytes = dir.readFileAlloc(io, filename, allocator, @enumFromInt(4 * 1024 * 1024)) catch
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

    // v1 was top-level-only and matched by root `module_idx`; v2
    // (#676) recurses into nested sub-components and matches by raw
    // core-bytes sha256. Both still rejected when `wamr_build_id`
    // doesn't match — AOT codegen isn't stable across versions.
    if (parsed.version != 1 and parsed.version != manifest_format_version)
        return error.ManifestVersionMismatch;
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
    errdefer allocator.free(pcs);

    // v2 (or any manifest entry carrying `core_sha256`) lets us
    // recurse into nested sub-components and match each on-disk
    // entry to the live core_modules[i].data slice by raw-bytes
    // sha256. The runtime's `findPrecompiled` then matches by
    // slice identity, sidestepping module_idx collisions across
    // sibling sub-components. (#676)
    const any_core_sha = blk: {
        for (parsed.modules) |mod| if (mod.core_sha256 != null) break :blk true;
        break :blk false;
    };

    if (any_core_sha) {
        // Re-parse the component into a scratch arena to walk its
        // nested-component tree. `core_mod.data` slices reference
        // `expected_component_bytes` (owned by the caller) so they
        // remain valid after `scratch.deinit()`; only the parse
        // bookkeeping lives in the arena.
        var scratch = std.heap.ArenaAllocator.init(allocator);
        defer scratch.deinit();
        const live = component_loader.load(expected_component_bytes, scratch.allocator()) catch
            return error.ComponentParseFailed;

        const Visit = struct { data: []const u8, local_idx: u32, hex: [64]u8 };
        var visits: std.ArrayList(Visit) = .empty;
        defer visits.deinit(scratch.allocator());

        const Walker = struct {
            fn walk(
                comp: *const ctypes.Component,
                list: *std.ArrayList(Visit),
                alloc: std.mem.Allocator,
            ) error{OutOfMemory}!void {
                for (comp.core_modules, 0..) |cm, mi| {
                    var d: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
                    std.crypto.hash.sha2.Sha256.hash(cm.data, &d, .{});
                    var h: [64]u8 = undefined;
                    const hc = "0123456789abcdef";
                    for (d, 0..) |b, j| {
                        h[j * 2] = hc[b >> 4];
                        h[j * 2 + 1] = hc[b & 0x0f];
                    }
                    try list.append(alloc, .{ .data = cm.data, .local_idx = @intCast(mi), .hex = h });
                }
                for (comp.components) |child| try walk(child, list, alloc);
            }
        };
        Walker.walk(&live, &visits, scratch.allocator()) catch return error.OutOfMemory;

        // For every manifest entry, locate the matching live core by
        // core_sha256 and stamp a `PrecompiledCore` with slice
        // identity (`core_wasm = core_mod.data`).
        for (parsed.modules, 0..) |mod, i| {
            const want = mod.core_sha256 orelse return error.ManifestCoreMismatch;
            const match = blk: {
                for (visits.items) |v| {
                    if (std.mem.eql(u8, &v.hex, want)) break :blk v;
                }
                break :blk null;
            };
            const v = match orelse return error.ManifestCoreMismatch;
            pcs[i] = .{
                .module_idx = v.local_idx,
                .cwasm_bytes = cwasm_buffers[i],
                .core_wasm = v.data,
            };
        }
    } else {
        // v1 fallback: legacy on-disk shape with top-level-only
        // cores keyed by root-local `module_idx`.
        for (parsed.modules, 0..) |mod, i| {
            pcs[i] = .{ .module_idx = mod.idx, .cwasm_bytes = cwasm_buffers[i] };
        }
    }

    return .{
        .manifest = parsed,
        .cwasm_buffers = cwasm_buffers,
        .pcs = pcs,
        .allocator = allocator,
        .arena = arena,
    };
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

test "defaultManifestPathFor: strips .wasm and appends .cwasm.json (#645)" {
    const alloc = std.testing.allocator;

    const a = try defaultManifestPathFor(alloc, "/tmp/stdio-echo.wasm");
    defer alloc.free(a);
    try std.testing.expectEqualStrings("/tmp/stdio-echo.cwasm.json", a);

    // No .wasm suffix → append directly (preserves the prior fallback).
    const b = try defaultManifestPathFor(alloc, "/tmp/stdio-echo");
    defer alloc.free(b);
    try std.testing.expectEqualStrings("/tmp/stdio-echo.cwasm.json", b);

    // .wasm appears mid-path but not as suffix → don't strip.
    const c = try defaultManifestPathFor(alloc, "/tmp/a.wasm.bak");
    defer alloc.free(c);
    try std.testing.expectEqualStrings("/tmp/a.wasm.bak.cwasm.json", c);
}
