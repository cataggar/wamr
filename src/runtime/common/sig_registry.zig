//! Process-global canonical signature registry for AOT call_indirect sig checks.
//!
//! Maps structurally-equal FuncTypes to a compact `u32` id so that the AOT
//! code-generator can emit a single 32-bit compare to validate that an
//! indirect-call target matches the static type expected at the call site.
//!
//! Design notes:
//!  * `id = 0` is reserved for "null funcref" / uninitialized slots.
//!  * Ids are assigned sequentially from 1 in insertion order and are stable
//!    for the lifetime of the registry.
//!  * Only structural equality of the function-type shape is considered; this
//!    is sufficient for Wasm MVP / reference-types call_indirect semantics.
//!    GC subtyping is intentionally not modelled here — the AOT inline compare
//!    targets non-GC parity only.
//!  * Access is serialised with a simple spinlock mutex so the registry can be
//!    shared across instantiation of multiple modules in the same process.

const std = @import("std");
const types = @import("types.zig");

const FuncType = types.FuncType;
const ValType = types.ValType;

/// Spinlock mutex (mirrors the style in `types.zig`, avoids std.Thread.Mutex
/// which is IO-gated in Zig 0.16).
const Mutex = struct {
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

    pub fn lock(self: *Mutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null)
            std.atomic.spinLoopHint();
    }

    pub fn unlock(self: *Mutex) void {
        self.state.store(0, .release);
    }
};

/// A deep-copied FuncType owned by the registry.
const Entry = struct {
    func_type: FuncType,
    /// Canonical byte-encoded key backing the hash-map entry.
    key: []const u8,
};

pub const SigRegistry = struct {
    allocator: std.mem.Allocator,
    mutex: Mutex = .{},
    entries: std.ArrayList(Entry) = .empty,
    /// Maps canonical key bytes → id (1-based).
    index: std.StringHashMapUnmanaged(u32) = .{},

    pub fn init(allocator: std.mem.Allocator) SigRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SigRegistry) void {
        for (self.entries.items) |*e| {
            self.allocator.free(e.func_type.params);
            self.allocator.free(e.func_type.results);
            if (e.func_type.param_tidxs.len != 0) self.allocator.free(e.func_type.param_tidxs);
            if (e.func_type.result_tidxs.len != 0) self.allocator.free(e.func_type.result_tidxs);
            if (e.func_type.field_tidxs.len != 0) self.allocator.free(e.func_type.field_tidxs);
            if (e.func_type.field_types.len != 0) self.allocator.free(e.func_type.field_types);
            if (e.func_type.field_muts.len != 0) self.allocator.free(e.func_type.field_muts);
            self.allocator.free(e.key);
        }
        self.entries.deinit(self.allocator);
        self.index.deinit(self.allocator);
        self.* = undefined;
    }

    /// Intern a FuncType and return its canonical id (≥ 1).
    /// On a cache hit no allocation occurs; on miss the FuncType is deep-copied.
    pub fn intern(self: *SigRegistry, ft: *const FuncType) !u32 {
        var key_buf: std.ArrayList(u8) = .empty;
        defer key_buf.deinit(self.allocator);
        try encodeKey(self.allocator, &key_buf, ft);

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.index.get(key_buf.items)) |id| return id;

        // Miss — deep-copy entry and key into registry-owned storage.
        const owned_key = try self.allocator.dupe(u8, key_buf.items);
        errdefer self.allocator.free(owned_key);

        const owned_ft = try copyFuncType(self.allocator, ft);
        errdefer freeFuncType(self.allocator, owned_ft);

        try self.entries.append(self.allocator, .{ .func_type = owned_ft, .key = owned_key });
        errdefer _ = self.entries.pop();

        const id: u32 = @intCast(self.entries.items.len);
        try self.index.put(self.allocator, owned_key, id);
        return id;
    }

    /// Return the FuncType previously interned with `id`, or null if out of range.
    pub fn get(self: *SigRegistry, id: u32) ?*const FuncType {
        if (id == 0) return null;
        self.mutex.lock();
        defer self.mutex.unlock();
        const idx = id - 1;
        if (idx >= self.entries.items.len) return null;
        return &self.entries.items[idx].func_type;
    }

    pub fn len(self: *SigRegistry) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return @intCast(self.entries.items.len);
    }
};

// ── Process-global singleton ───────────────────────────────────────────────
//
// A single registry is shared by all AOT instances so that canonical ids are
// stable across modules and can be compared directly at call_indirect sites.
// Lazily initialised on first `global()` call using the page allocator so it
// is available before `main` (e.g. during test setup).

var g_registry_storage: SigRegistry = undefined;
var g_registry_inited: bool = false;
var g_registry_mutex: Mutex = .{};

/// Return the process-global `SigRegistry`, initialising it on first use.
pub fn global() *SigRegistry {
    g_registry_mutex.lock();
    defer g_registry_mutex.unlock();
    if (!g_registry_inited) {
        g_registry_storage = SigRegistry.init(std.heap.page_allocator);
        g_registry_inited = true;
    }
    return &g_registry_storage;
}

fn copyFuncType(allocator: std.mem.Allocator, ft: *const FuncType) !FuncType {
    const params = try allocator.dupe(ValType, ft.params);
    errdefer allocator.free(params);
    const results = try allocator.dupe(ValType, ft.results);
    errdefer allocator.free(results);

    const param_tidxs = if (ft.param_tidxs.len != 0)
        try allocator.dupe(u32, ft.param_tidxs)
    else
        &[_]u32{};
    errdefer if (param_tidxs.len != 0) allocator.free(param_tidxs);

    const result_tidxs = if (ft.result_tidxs.len != 0)
        try allocator.dupe(u32, ft.result_tidxs)
    else
        &[_]u32{};
    errdefer if (result_tidxs.len != 0) allocator.free(result_tidxs);

    const field_tidxs = if (ft.field_tidxs.len != 0)
        try allocator.dupe(u32, ft.field_tidxs)
    else
        &[_]u32{};
    errdefer if (field_tidxs.len != 0) allocator.free(field_tidxs);

    const field_types = if (ft.field_types.len != 0)
        try allocator.dupe(ValType, ft.field_types)
    else
        &[_]ValType{};
    errdefer if (field_types.len != 0) allocator.free(field_types);

    const field_muts = if (ft.field_muts.len != 0)
        try allocator.dupe(u8, ft.field_muts)
    else
        &[_]u8{};

    return .{
        .params = params,
        .results = results,
        .param_tidxs = param_tidxs,
        .result_tidxs = result_tidxs,
        .kind = ft.kind,
        .field_tidxs = field_tidxs,
        .field_types = field_types,
        .field_muts = field_muts,
        .supertype_idx = ft.supertype_idx,
        .is_final = ft.is_final,
    };
}

fn freeFuncType(allocator: std.mem.Allocator, ft: FuncType) void {
    allocator.free(ft.params);
    allocator.free(ft.results);
    if (ft.param_tidxs.len != 0) allocator.free(ft.param_tidxs);
    if (ft.result_tidxs.len != 0) allocator.free(ft.result_tidxs);
    if (ft.field_tidxs.len != 0) allocator.free(ft.field_tidxs);
    if (ft.field_types.len != 0) allocator.free(ft.field_types);
    if (ft.field_muts.len != 0) allocator.free(ft.field_muts);
}

/// Encode a FuncType into its canonical structural key.
/// Layout (all little-endian):
///   u8   kind
///   u16  params_len   u8*   params  (ValType bytes)
///   u16  results_len  u8*   results
///   u16  fields_len   u8*   field_types   u8* field_muts
///
/// `param_tidxs` / `result_tidxs` / `field_tidxs` are deliberately NOT part of
/// the key: they reference module-local type indices and have no meaning in
/// the process-wide registry. Structural equality over ValTypes is what the
/// Wasm MVP call_indirect check requires.
fn encodeKey(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), ft: *const FuncType) !void {
    try buf.append(allocator, @intFromEnum(ft.kind));
    try appendU16(allocator, buf, @intCast(ft.params.len));
    for (ft.params) |v| try buf.append(allocator, @intFromEnum(v));
    try appendU16(allocator, buf, @intCast(ft.results.len));
    for (ft.results) |v| try buf.append(allocator, @intFromEnum(v));
    try appendU16(allocator, buf, @intCast(ft.field_types.len));
    for (ft.field_types) |v| try buf.append(allocator, @intFromEnum(v));
    for (ft.field_muts) |m| try buf.append(allocator, m);
    // Recursive group context: types in different groups or at different
    // positions within their group are distinct for call_indirect.
    try appendU16(allocator, buf, ft.rec_group_size);
    try appendU16(allocator, buf, ft.rec_group_position);
}

fn appendU16(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), v: u16) !void {
    try buf.append(allocator, @intCast(v & 0xFF));
    try buf.append(allocator, @intCast((v >> 8) & 0xFF));
}

// ── Tests ──────────────────────────────────────────────────────────────────

test "SigRegistry: dedupes structurally-equal func types" {
    var reg = SigRegistry.init(std.testing.allocator);
    defer reg.deinit();

    const a = FuncType{
        .params = &.{ .i32, .i64 },
        .results = &.{.f32},
    };
    const b = FuncType{
        .params = &.{ .i32, .i64 },
        .results = &.{.f32},
    };

    const id_a = try reg.intern(&a);
    const id_b = try reg.intern(&b);
    try std.testing.expect(id_a >= 1);
    try std.testing.expectEqual(id_a, id_b);
    try std.testing.expectEqual(@as(u32, 1), reg.len());
}

test "SigRegistry: distinct params → distinct ids" {
    var reg = SigRegistry.init(std.testing.allocator);
    defer reg.deinit();

    const a = FuncType{ .params = &.{.i32}, .results = &.{} };
    const b = FuncType{ .params = &.{.i64}, .results = &.{} };
    const c = FuncType{ .params = &.{.i32}, .results = &.{.i32} };

    const id_a = try reg.intern(&a);
    const id_b = try reg.intern(&b);
    const id_c = try reg.intern(&c);
    try std.testing.expect(id_a != id_b);
    try std.testing.expect(id_a != id_c);
    try std.testing.expect(id_b != id_c);
    try std.testing.expectEqual(@as(u32, 3), reg.len());
}

test "SigRegistry: funcref and externref distinguished" {
    var reg = SigRegistry.init(std.testing.allocator);
    defer reg.deinit();

    const a = FuncType{ .params = &.{.funcref}, .results = &.{} };
    const b = FuncType{ .params = &.{.externref}, .results = &.{} };
    const id_a = try reg.intern(&a);
    const id_b = try reg.intern(&b);
    try std.testing.expect(id_a != id_b);
}

test "SigRegistry: nullable vs non-nullable kept distinct" {
    var reg = SigRegistry.init(std.testing.allocator);
    defer reg.deinit();

    const a = FuncType{ .params = &.{.funcref}, .results = &.{} };
    const b = FuncType{ .params = &.{.nonfuncref}, .results = &.{} };
    const id_a = try reg.intern(&a);
    const id_b = try reg.intern(&b);
    try std.testing.expect(id_a != id_b);
}

test "SigRegistry: get() returns the stored type" {
    var reg = SigRegistry.init(std.testing.allocator);
    defer reg.deinit();

    const ft = FuncType{ .params = &.{ .i32, .f64 }, .results = &.{.i64} };
    const id = try reg.intern(&ft);
    const stored = reg.get(id).?;
    try std.testing.expectEqual(@as(usize, 2), stored.params.len);
    try std.testing.expectEqual(ValType.i32, stored.params[0]);
    try std.testing.expectEqual(ValType.f64, stored.params[1]);
    try std.testing.expectEqual(@as(usize, 1), stored.results.len);
    try std.testing.expectEqual(ValType.i64, stored.results[0]);

    try std.testing.expectEqual(@as(?*const FuncType, null), reg.get(0));
    try std.testing.expectEqual(@as(?*const FuncType, null), reg.get(999));
}

test "SigRegistry: many types stay distinct and re-intern is stable" {
    var reg = SigRegistry.init(std.testing.allocator);
    defer reg.deinit();

    var ids: [32]u32 = undefined;
    inline for (0..32) |i| {
        const params = [_]ValType{ .i32, .i64, if (i % 2 == 0) .f32 else .f64 };
        const results = [_]ValType{if (i % 3 == 0) .i32 else .i64};
        var ft: FuncType = .{ .params = &params, .results = &results };
        _ = &ft;
        ids[i] = try reg.intern(&ft);
    }
    // Re-intern the same set, expect identical ids.
    inline for (0..32) |i| {
        const params = [_]ValType{ .i32, .i64, if (i % 2 == 0) .f32 else .f64 };
        const results = [_]ValType{if (i % 3 == 0) .i32 else .i64};
        var ft: FuncType = .{ .params = &params, .results = &results };
        _ = &ft;
        const id2 = try reg.intern(&ft);
        try std.testing.expectEqual(ids[i], id2);
    }
    // 4 distinct shapes (2 param variants * 2 result variants).
    try std.testing.expectEqual(@as(u32, 4), reg.len());
}
