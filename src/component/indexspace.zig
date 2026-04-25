//! Component-model index-space resolvers.
//!
//! The component-model spec defines several flat index spaces (component
//! func, component instance, core func, core instance, type, …). Each is
//! contributed to by a specific subset of declaration kinds, and entries
//! are appended in declaration order as the binary is walked.
//!
//! Phase 1A's loader stores declarations in *separate per-section arrays*
//! (`component.imports`, `component.aliases`, `component.canons`, …)
//! rather than the original mixed-section ordering. We assume the canonical
//! section order produced by both `wasm-tools` and `wit-bindgen`:
//!
//!   imports → instances → aliases → canons (… → exports)
//!
//! That assumption is sufficient for every component we currently care
//! about (hand-authored Phase 2B fixtures *and* `stdio-echo.wasm` from
//! `cargo build --target wasm32-wasip2`). A future slice can replace this
//! with binary-decl-order tagging if non-canonical layouts ever appear.
//!
//! The resolvers in this module are pure functions over the parsed
//! `Component` AST — they don't touch a `ComponentInstance`, allowing them
//! to be reused for both pre-instantiation analysis and runtime resolution.

const std = @import("std");
const ctypes = @import("types.zig");

// ── Component-func index space ────────────────────────────────────────────
//
// Contributors, in section order:
//   1. `import` decls of kind `.func` (top-level component func imports —
//      rare in real components but used by hand-authored fixtures).
//   2. `alias` decls with `sort = .func` (member aliases of imported or
//      local component instances — every wit-bindgen-generated component
//      uses this to surface WASI member funcs).
//   3. `canon.lift` entries (the only contributor that produces *callable*
//      component funcs from core funcs).

pub const CompFuncRef = union(enum) {
    /// Top-level `import (… (func type_idx))` — index into `component.imports`.
    imported: u32,
    /// `alias … (func)` — index into `component.aliases`. Resolves to a
    /// member of an instance, which itself sits in the component-instance
    /// index space (resolve via `resolveCompInstance` against
    /// `instance_export.instance_idx`).
    aliased: u32,
    /// `canon lift core_func_idx … type_idx` — index into `component.canons`.
    lifted: u32,
};

pub fn resolveCompFunc(component: *const ctypes.Component, idx: u32) ?CompFuncRef {
    var n: u32 = 0;
    for (component.imports, 0..) |imp, i| {
        if (imp.desc != .func) continue;
        if (n == idx) return .{ .imported = @intCast(i) };
        n += 1;
    }
    for (component.aliases, 0..) |a, i| {
        if (!aliasContributesTo(a, .comp_func)) continue;
        if (n == idx) return .{ .aliased = @intCast(i) };
        n += 1;
    }
    for (component.canons, 0..) |c, i| {
        switch (c) {
            .lift => {
                if (n == idx) return .{ .lifted = @intCast(i) };
                n += 1;
            },
            else => {},
        }
    }
    return null;
}

// ── Component-instance index space ────────────────────────────────────────
//
// Contributors, in section order:
//   1. `import` decls of kind `.instance`.
//   2. `instance` expressions (locally instantiated or `from_exports`-style
//      bundles synthesizing an instance from already-defined members).
//   3. `alias` decls with `sort = .instance`.

pub const CompInstanceRef = union(enum) {
    /// Index into `component.imports` (filtered to instance-typed).
    imported: u32,
    /// Index into `component.instances`.
    local: u32,
    /// Index into `component.aliases`.
    aliased: u32,
};

pub fn resolveCompInstance(component: *const ctypes.Component, idx: u32) ?CompInstanceRef {
    var n: u32 = 0;
    for (component.imports, 0..) |imp, i| {
        if (imp.desc != .instance) continue;
        if (n == idx) return .{ .imported = @intCast(i) };
        n += 1;
    }
    for (component.instances, 0..) |_, i| {
        if (n == idx) return .{ .local = @intCast(i) };
        n += 1;
    }
    for (component.aliases, 0..) |a, i| {
        if (!aliasContributesTo(a, .comp_instance)) continue;
        if (n == idx) return .{ .aliased = @intCast(i) };
        n += 1;
    }
    return null;
}

// ── Core-func index space ─────────────────────────────────────────────────
//
// Contributors, in section order:
//   1. Canon entries that produce a core func — `canon.lower` plus the
//      `canon.resource.{new,drop,rep}` family. Each contributes one slot.
//   2. `alias` decls with `sort = .core(.func)` (exposing a core instance's
//      core func export under a top-level core-func index).
//
// The component-model spec is explicit that `canon.lower` and the resource
// canons all produce core functions and are counted in the same indexspace
// in the order they appear in `canons[]`. Real wit-component output (e.g.
// the stdio-echo binary) interleaves them with abandon: if we miscount
// resource.drop slots, every later alias resolves to the wrong target.

pub const CoreFuncRef = union(enum) {
    /// Index into `component.canons` for a `.lower` entry.
    lowered: u32,
    /// Index into `component.canons` for a `.resource_drop` entry.
    resource_drop: u32,
    /// Index into `component.canons` for a `.resource_new` entry.
    resource_new: u32,
    /// Index into `component.canons` for a `.resource_rep` entry.
    resource_rep: u32,
    /// Index into `component.aliases`.
    aliased: u32,
};

pub fn resolveCoreFunc(component: *const ctypes.Component, idx: u32) ?CoreFuncRef {
    var n: u32 = 0;
    for (component.canons, 0..) |c, i| {
        switch (c) {
            .lower => {
                if (n == idx) return .{ .lowered = @intCast(i) };
                n += 1;
            },
            .resource_drop => {
                if (n == idx) return .{ .resource_drop = @intCast(i) };
                n += 1;
            },
            .resource_new => {
                if (n == idx) return .{ .resource_new = @intCast(i) };
                n += 1;
            },
            .resource_rep => {
                if (n == idx) return .{ .resource_rep = @intCast(i) };
                n += 1;
            },
            .lift => {}, // contributes to component-func indexspace, not core-func
        }
    }
    for (component.aliases, 0..) |a, i| {
        if (!aliasContributesTo(a, .core_func)) continue;
        if (n == idx) return .{ .aliased = @intCast(i) };
        n += 1;
    }
    return null;
}

// ── Top-level core table / memory / global index spaces ──────────────────
//
// Contributors today: only `alias core export ... (core {table,memory,global})`.
// Real-world wit-component output uses these to lift `$main`'s memory and
// `$shim`'s table to the component scope so a later inline-exports core
// instance can bundle them and pass them into a third core module
// (`$fixup` in stdio-echo). A future slice can add core imports and core
// instance defs once those use cases appear.

pub const CoreItemAliasRef = struct {
    /// Index into `component.aliases`.
    aliased: u32,
};

fn resolveCoreItem(
    component: *const ctypes.Component,
    idx: u32,
    target: AliasTarget,
) ?CoreItemAliasRef {
    var n: u32 = 0;
    for (component.aliases, 0..) |a, i| {
        if (!aliasContributesTo(a, target)) continue;
        if (n == idx) return .{ .aliased = @intCast(i) };
        n += 1;
    }
    return null;
}

pub fn resolveCoreTable(component: *const ctypes.Component, idx: u32) ?CoreItemAliasRef {
    return resolveCoreItem(component, idx, .core_table);
}

pub fn resolveCoreMemory(component: *const ctypes.Component, idx: u32) ?CoreItemAliasRef {
    return resolveCoreItem(component, idx, .core_memory);
}

pub fn resolveCoreGlobal(component: *const ctypes.Component, idx: u32) ?CoreItemAliasRef {
    return resolveCoreItem(component, idx, .core_global);
}

// ── Member lookup ─────────────────────────────────────────────────────────

/// Look up a named member in a locally-instantiated `from_exports`-style
/// component instance. Returns the member's `SortIdx` (back-pointer into
/// the appropriate index space) or null.
///
/// Only the `.exports` form is supported; `.instantiate` would require
/// us to recursively resolve another component's exports, which we do
/// not yet do (the only real-world contributor is wit-bindgen's
/// synthesized `wasi:cli/run` instance, which always uses `.exports`).
pub fn lookupLocalInstanceMember(
    component: *const ctypes.Component,
    local_inst_idx: u32,
    name: []const u8,
) ?ctypes.SortIdx {
    if (local_inst_idx >= component.instances.len) return null;
    const expr = component.instances[local_inst_idx];
    return switch (expr) {
        .exports => |exports| blk: {
            for (exports) |e| {
                if (std.mem.eql(u8, e.name, name)) break :blk e.sort_idx;
            }
            break :blk null;
        },
        .instantiate => null,
    };
}

// ── Internal: "which index space does this alias contribute to?" ─────────

const AliasTarget = enum {
    comp_func,
    comp_instance,
    core_func,
    core_instance,
    core_table,
    core_memory,
    core_global,
    type_x,
    value,
};

fn aliasContributesTo(a: ctypes.Alias, target: AliasTarget) bool {
    return switch (a) {
        .instance_export => |ie| switch (ie.sort) {
            .func => target == .comp_func,
            .instance => target == .comp_instance,
            .core => |cs| switch (cs) {
                .func => target == .core_func,
                .instance => target == .core_instance,
                .type => target == .type_x,
                .table => target == .core_table,
                .memory => target == .core_memory,
                .global => target == .core_global,
                .tag, .module => false,
            },
            .type => target == .type_x,
            .value => target == .value,
            .component => false,
        },
        // Outer / core aliases — not yet observed; default to false. A
        // future slice handling nested components will need to extend this.
        else => false,
    };
}

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

test "resolveCompFunc: imports → aliases → lifts in section order" {
    // Synthesized component with one func import, one alias.func, one
    // canon.lift. Expected comp-func index space: 0=import, 1=alias, 2=lift.
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "host:fn", .desc = .{ .func = 0 } },
    };
    const aliases = [_]ctypes.Alias{
        .{ .instance_export = .{ .sort = .func, .instance_idx = 0, .name = "m" } },
    };
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 0, .opts = &.{} } },
    };
    const comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases,
        .types = &.{},
        .canons = &canons,
        .imports = &imports,
        .exports = &.{},
    };
    try testing.expect(resolveCompFunc(&comp, 0).? == .imported);
    try testing.expect(resolveCompFunc(&comp, 1).? == .aliased);
    try testing.expect(resolveCompFunc(&comp, 2).? == .lifted);
    try testing.expect(resolveCompFunc(&comp, 3) == null);
}

test "resolveCompInstance: imports → locals → aliases" {
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "wasi:io/streams", .desc = .{ .instance = 0 } },
    };
    const inline_exp = [_]ctypes.InlineExport{
        .{ .name = "run", .sort_idx = .{ .sort = .func, .idx = 5 } },
    };
    const instances = [_]ctypes.InstanceExpr{
        .{ .exports = &inline_exp },
    };
    const aliases = [_]ctypes.Alias{
        .{ .instance_export = .{ .sort = .instance, .instance_idx = 0, .name = "x" } },
    };
    const comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &instances,
        .aliases = &aliases,
        .types = &.{},
        .canons = &.{},
        .imports = &imports,
        .exports = &.{},
    };
    try testing.expect(resolveCompInstance(&comp, 0).? == .imported);
    try testing.expect(resolveCompInstance(&comp, 1).? == .local);
    try testing.expect(resolveCompInstance(&comp, 2).? == .aliased);
    try testing.expect(resolveCompInstance(&comp, 3) == null);
}

test "resolveCoreFunc: canon.lowers → core(.func) aliases" {
    const aliases = [_]ctypes.Alias{
        // A non-core-func alias (sort=.func) — should be ignored here.
        .{ .instance_export = .{ .sort = .func, .instance_idx = 0, .name = "skip" } },
        // Core func alias #1 — contributes at idx 2.
        .{ .instance_export = .{ .sort = .{ .core = .func }, .instance_idx = 0, .name = "x" } },
    };
    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &.{} } },
        .{ .lower = .{ .func_idx = 1, .opts = &.{} } },
    };
    const comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases,
        .types = &.{},
        .canons = &canons,
        .imports = &.{},
        .exports = &.{},
    };
    try testing.expect(resolveCoreFunc(&comp, 0).? == .lowered);
    try testing.expect(resolveCoreFunc(&comp, 1).? == .lowered);
    try testing.expect(resolveCoreFunc(&comp, 2).? == .aliased);
    try testing.expect(resolveCoreFunc(&comp, 3) == null);
}

test "lookupLocalInstanceMember: finds named export in inline-exports instance" {
    const inline_exp = [_]ctypes.InlineExport{
        .{ .name = "run", .sort_idx = .{ .sort = .func, .idx = 7 } },
        .{ .name = "other", .sort_idx = .{ .sort = .func, .idx = 8 } },
    };
    const instances = [_]ctypes.InstanceExpr{.{ .exports = &inline_exp }};
    const comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &instances,
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const m = lookupLocalInstanceMember(&comp, 0, "run").?;
    try testing.expectEqual(@as(u32, 7), m.idx);
    try testing.expect(lookupLocalInstanceMember(&comp, 0, "missing") == null);
}
