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
    // Section-order-aware path: when the loader populated
    // `comp_instance_indexspace`, dispatch directly. Composed
    // components need this because instance-section and alias-section
    // (sort .instance) entries interleave (issue #355).
    if (component.comp_instance_indexspace.len > 0) {
        if (idx >= component.comp_instance_indexspace.len) return null;
        return switch (component.comp_instance_indexspace[idx]) {
            .import => |i| .{ .imported = i },
            .instance => |i| .{ .local = i },
            .alias => |i| .{ .aliased = i },
        };
    }
    // Legacy path: hand-authored fixtures (no loader) and old
    // pre-#355 binaries that don't set the indexspace. Walk imports,
    // then instances, then aliases — correct only when sections do
    // not interleave.
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

/// What an `(alias export ...)` chain ultimately points at, for an entry
/// in the component-instance index space. Used by `registerInstanceExport`
/// (issue #355) to follow the `wasm-tools compose` pattern where the
/// top-level `wasi:cli/run@0.2.x` export is an aliased export of a
/// sub-component instance:
///
/// ```
/// (instance (;10;) (instantiate 0 ...))
/// (alias export 10 "wasi:cli/run@0.2.6" (instance (;11;)))
/// (export "wasi:cli/run@0.2.6" (instance 11))
/// ```
///
/// Resolving `instance 11` yields `.sub_export { source = .local{10},
/// name = "wasi:cli/run@0.2.6" }`.
pub const InstanceExprRef = union(enum) {
    /// Parent component import; satisfied at link time.
    imported: u32,
    /// `component.instances[i]` directly — no alias hop in play.
    local: u32,
    /// The chain bottoms out by selecting a named instance export of
    /// some other resolved component instance.
    sub_export: SubExport,

    pub const SubExport = struct {
        /// Where the chain landed before the final name lookup.
        source: Source,
        /// The named instance export to look up on `source`.
        name: []const u8,

        pub const Source = union(enum) {
            imported: u32,
            local: u32,
        };
    };
};

pub const ResolveInstanceExprError = error{
    /// The alias chain exceeded a defensive depth bound or contains
    /// a cycle.
    AliasDepthExceeded,
    /// The alias chain has more than one hop. The wasm-tools compose
    /// output that motivates #355 only ever produces single-hop alias
    /// chains; deeper chains require a chain-of-names representation
    /// that is not yet implemented.
    MultiHopAliasUnsupported,
    /// An alias in the chain has the wrong shape (e.g. `outer` alias
    /// where `instance_export` is required, or non-instance sort).
    InvalidAliasShape,
};

/// Walk an `(alias export ...)` chain in the component-instance index
/// space until it bottoms out at an import, a local, or a named export
/// of an import/local. Returns `null` if `idx` is out of range. Returns
/// an error on cycle / multi-hop / malformed alias.
pub fn resolveInstanceExpr(
    component: *const ctypes.Component,
    idx: u32,
) ResolveInstanceExprError!?InstanceExprRef {
    const initial = resolveCompInstance(component, idx) orelse return null;
    return switch (initial) {
        .imported => |i| .{ .imported = i },
        .local => |i| .{ .local = i },
        .aliased => |alias_idx| try followInstanceAlias(component, alias_idx),
    };
}

fn followInstanceAlias(
    component: *const ctypes.Component,
    alias_idx: u32,
) ResolveInstanceExprError!?InstanceExprRef {
    if (alias_idx >= component.aliases.len) return error.InvalidAliasShape;
    const alias = component.aliases[alias_idx];
    const ie = switch (alias) {
        .instance_export => |x| x,
        .outer => return error.InvalidAliasShape,
    };
    if (ie.sort != .instance) return error.InvalidAliasShape;

    const inner = resolveCompInstance(component, ie.instance_idx) orelse return null;
    return switch (inner) {
        .imported => |i| .{ .sub_export = .{
            .source = .{ .imported = i },
            .name = ie.name,
        } },
        .local => |i| .{ .sub_export = .{
            .source = .{ .local = i },
            .name = ie.name,
        } },
        // Multi-hop alias chains (alias of alias of …) require a chain
        // of names. Real wasm-tools compose output only ever emits
        // single-hop chains for the composition pattern that motivates
        // #355, so refuse rather than silently truncate the chain.
        .aliased => |next_alias_idx| {
            if (next_alias_idx == alias_idx) return error.AliasDepthExceeded;
            return error.MultiHopAliasUnsupported;
        },
    };
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
    // Prefer the loader-provided binary-order indexspace when present
    // (real components from wit-component / wasm-tools). Hand-authored
    // fixtures that bypass the loader fall back to the section-order
    // heuristic below.
    if (component.core_func_indexspace.len > 0) {
        if (idx >= component.core_func_indexspace.len) return null;
        const c = component.core_func_indexspace[idx];
        return switch (c) {
            .alias => |a| .{ .aliased = a },
            .canon => |canon_idx| switch (component.canons[canon_idx]) {
                .lower => .{ .lowered = canon_idx },
                .resource_drop => .{ .resource_drop = canon_idx },
                .resource_new => .{ .resource_new = canon_idx },
                .resource_rep => .{ .resource_rep = canon_idx },
                .lift => null, // lift never contributes to core-func indexspace
            },
        };
    }
    // Fallback (hand-authored fixtures, no loader): assume canon
    // entries declared first, aliases second.
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
            .lift => {},
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

test "resolveInstanceExpr: local instance returns .local" {
    const inline_exp = [_]ctypes.InlineExport{
        .{ .name = "run", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const instances = [_]ctypes.InstanceExpr{
        .{ .exports = &inline_exp },
    };
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
    const got = (try resolveInstanceExpr(&comp, 0)).?;
    try testing.expect(got == .local);
    try testing.expectEqual(@as(u32, 0), got.local);
}

test "resolveInstanceExpr: imported instance returns .imported" {
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "wasi:io/streams", .desc = .{ .instance = 0 } },
    };
    const comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &imports,
        .exports = &.{},
    };
    const got = (try resolveInstanceExpr(&comp, 0)).?;
    try testing.expect(got == .imported);
    try testing.expectEqual(@as(u32, 0), got.imported);
}

test "resolveInstanceExpr: alias of local returns .sub_export over .local" {
    // Mirrors `wasm-tools compose` shape:
    //   instance 0: (instantiate sub-component)
    //   alias  0:   (alias export 0 "wasi:cli/run@0.2.6" (instance 1))
    //   instance-index space: 0=local, 1=aliased.
    const instances = [_]ctypes.InstanceExpr{
        .{ .instantiate = .{ .component_idx = 0, .args = &.{} } },
    };
    const aliases = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .instance,
            .instance_idx = 0,
            .name = "wasi:cli/run@0.2.6",
        } },
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
        .imports = &.{},
        .exports = &.{},
    };
    const got = (try resolveInstanceExpr(&comp, 1)).?;
    try testing.expect(got == .sub_export);
    try testing.expect(got.sub_export.source == .local);
    try testing.expectEqual(@as(u32, 0), got.sub_export.source.local);
    try testing.expectEqualStrings("wasi:cli/run@0.2.6", got.sub_export.name);
}

test "resolveInstanceExpr: alias of imported returns .sub_export over .imported" {
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "host:bundle", .desc = .{ .instance = 0 } },
    };
    const aliases = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .instance,
            .instance_idx = 0,
            .name = "inner",
        } },
    };
    const comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases,
        .types = &.{},
        .canons = &.{},
        .imports = &imports,
        .exports = &.{},
    };
    // index space: 0=imported, 1=aliased.
    const got = (try resolveInstanceExpr(&comp, 1)).?;
    try testing.expect(got == .sub_export);
    try testing.expect(got.sub_export.source == .imported);
    try testing.expectEqual(@as(u32, 0), got.sub_export.source.imported);
    try testing.expectEqualStrings("inner", got.sub_export.name);
}

test "resolveInstanceExpr: alias of alias errors with MultiHopAliasUnsupported" {
    // Two-hop chain: instance 1 = aliased(0), instance 2 = aliased(1).
    // The chain `2 → alias 1 → alias 0 → local 0` is not yet supported;
    // resolveInstanceExpr must surface a clear error rather than
    // silently truncate the chain.
    const instances = [_]ctypes.InstanceExpr{
        .{ .instantiate = .{ .component_idx = 0, .args = &.{} } },
    };
    const aliases = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .instance,
            .instance_idx = 0,
            .name = "outer",
        } },
        .{ .instance_export = .{
            .sort = .instance,
            .instance_idx = 1, // points at the prior alias
            .name = "inner",
        } },
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
        .imports = &.{},
        .exports = &.{},
    };
    // index space: 0=local, 1=alias[0], 2=alias[1].
    try testing.expectError(
        error.MultiHopAliasUnsupported,
        resolveInstanceExpr(&comp, 2),
    );
}

test "resolveInstanceExpr: self-referential alias errors with AliasDepthExceeded" {
    // Pathological self-loop: alias 0 points at instance idx 1 which
    // resolves back to alias 0.
    const aliases = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .instance,
            .instance_idx = 1, // index space slot for this very alias
            .name = "loop",
        } },
    };
    const comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases,
        .types = &.{},
        .canons = &.{},
        .imports = &.{ .{ .name = "host", .desc = .{ .instance = 0 } } },
        .exports = &.{},
    };
    // index space: 0=imported, 1=alias[0]. Resolving alias[0] looks up
    // its instance_idx=1 which is the alias itself → cycle.
    try testing.expectError(
        error.AliasDepthExceeded,
        resolveInstanceExpr(&comp, 1),
    );
}

test "resolveInstanceExpr: out-of-range idx returns null" {
    const comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    try testing.expect((try resolveInstanceExpr(&comp, 0)) == null);
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
