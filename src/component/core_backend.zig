//! Backend abstraction for a component's core-module instances.
//!
//! A `ComponentInstance` historically held one `*core_types.ModuleInstance`
//! per `core_instances[…]` slot — every embedded core module was loaded
//! by `runtime/interpreter/loader.zig` and executed by the interpreter.
//! Issue #625 introduces an alternate AOT path so a precompiled
//! `.cwasm` artifact for a core module can be instantiated and executed
//! by `runtime/aot/runtime.zig`.
//!
//! This module defines the two unions that bridge the two backends:
//!
//!   * `CoreModuleBackend` — the parsed module form
//!     (`*WasmModule` or `*AotModule`)
//!   * `CoreInstanceBackend` — the runnable form
//!     (`*ModuleInstance` or `*AotInstance`)
//!
//! Backend-agnostic adapter helpers (`memory`, `findExportFunc`, …)
//! let callers in `instance.zig` and `executor.zig` resolve cross-
//! instance memory/table/global sharing and export lookup without
//! caring which backend produced the instance — the underlying
//! `MemoryInstance`, `TableInstance` and `GlobalInstance` types
//! already live in `runtime/common` and are shared between
//! interp and AOT.
//!
//! Rollout (see issue #625):
//!   * Phase 1 (this commit): types + `Options.precompiled_cores`
//!     plumbing + `firstBackendMemory` lookup. AOT-backed cores can be
//!     loaded and exposed for memory/table/global sharing but their
//!     exported functions are not yet callable from the canon-ABI
//!     lift path (phase 3).
//!   * Phase 2: `wamrc compile-component` + manifest format.
//!   * Phase 3: canon-ABI dispatch onto `*AotInstance` (executor.zig
//!     `callComponentFuncByLocal`) + cross-instance canon-lower into
//!     an AOT core via per-import native arg-marshalling stubs.

const std = @import("std");
const core_types = @import("../runtime/common/types.zig");
const aot_loader = @import("../runtime/aot/loader.zig");
const aot_runtime = @import("../runtime/aot/runtime.zig");

/// The parsed-but-not-yet-runnable form of a core module.
///
/// Kept on `ComponentInstance.module_arena` next to whatever produced
/// it (the interpreter loader or the AOT loader). The lifetime is the
/// `ComponentInstance`'s.
pub const CoreModuleBackend = union(enum) {
    interp: *core_types.WasmModule,
    aot: *aot_loader.AotModule,
};

/// The runnable form of a core module.
///
/// Sibling to (not replacement of) `CoreInstanceEntry.module_inst` in
/// phase 1: existing call sites continue to read `module_inst` and the
/// vast majority of paths stay interp-only. New AOT-aware lookups
/// (`firstBackendMemory`, the executor's memory resolver) consult both.
pub const CoreInstanceBackend = union(enum) {
    interp: *core_types.ModuleInstance,
    aot: *aot_runtime.AotInstance,

    /// Return the `MemoryInstance` at `mem_idx` (0 is the canonical
    /// memory) on this backend, or null if the backend has no memory
    /// at that index.
    pub fn memory(self: CoreInstanceBackend, mem_idx: u32) ?*core_types.MemoryInstance {
        switch (self) {
            .interp => |mi| return mi.getMemory(mem_idx),
            .aot => |ai| {
                if (mem_idx >= ai.memories.len) return null;
                return ai.memories[mem_idx];
            },
        }
    }

    /// Return the `TableInstance` at `tbl_idx` on this backend.
    pub fn table(self: CoreInstanceBackend, tbl_idx: u32) ?*core_types.TableInstance {
        switch (self) {
            .interp => |mi| {
                if (tbl_idx >= mi.tables.len) return null;
                return mi.tables[tbl_idx];
            },
            .aot => |ai| {
                if (tbl_idx >= ai.tables.len) return null;
                return ai.tables[tbl_idx];
            },
        }
    }

    /// Return the `GlobalInstance` at `glob_idx` on this backend.
    pub fn global(self: CoreInstanceBackend, glob_idx: u32) ?*core_types.GlobalInstance {
        switch (self) {
            .interp => |mi| {
                if (glob_idx >= mi.globals.len) return null;
                return mi.globals[glob_idx];
            },
            .aot => |ai| {
                if (glob_idx >= ai.globals.len) return null;
                return ai.globals[glob_idx];
            },
        }
    }

    /// Look up an exported function by name. Returns the local funcidx
    /// (imports + locals — same convention both backends use).
    pub fn findExportFunc(self: CoreInstanceBackend, name: []const u8) ?u32 {
        switch (self) {
            .interp => |mi| {
                for (mi.module.exports) |exp| {
                    if (exp.kind == .function and std.mem.eql(u8, exp.name, name)) {
                        return exp.index;
                    }
                }
                return null;
            },
            .aot => |ai| return aot_runtime.findExportFunc(ai, name),
        }
    }
};

/// A caller-supplied precompiled-core artifact, used to opt a single
/// `core_modules[module_idx]` into the AOT backend at instantiate time.
///
/// `cwasm_bytes` must outlive the resulting `ComponentInstance` —
/// `AotModule` parsing is zero-copy on data sections, so the bytes are
/// borrowed.
pub const PrecompiledCore = struct {
    module_idx: u32,
    cwasm_bytes: []const u8,
    /// Optional raw-bytes identity of the core module that produced
    /// this cwasm. Both `wamr.component_loader.load` invocations
    /// against the same `wasm_data` buffer produce identical slices
    /// for `core_modules[i].data`, so slice (ptr+len) equality is a
    /// stable cross-parse key. When set, `findPrecompiled` matches
    /// by `core_wasm`; when null, it falls back to matching by
    /// `module_idx` against the *root* component (legacy on-disk
    /// manifests + test fixtures that only have top-level cores).
    ///
    /// `wamr run`'s auto-precompile walks composed-component trees
    /// (`wabt component compose -d` output) and stamps `core_wasm`
    /// on every emitted entry so cores nested inside sub-components
    /// can be found during recursive `instantiateWithOptions` of
    /// those sub-components without their inner local `module_idx`
    /// colliding with the root's. (#662 phase D)
    core_wasm: ?[]const u8 = null,
};

/// Process-global toggle for AOT debug diagnostics. Off by default;
/// `main.zig` sets it during startup when the `WAMR_AOT_DEBUG` env var
/// is set to a non-empty / non-`0` / non-`false` value. Read via
/// `debugAotEnabled` from any code path that wants to surface AOT
/// instantiation / call / trap details that would otherwise be
/// swallowed by the `error.Trap` envelope (see #644).
///
/// #859 thread-safety note: genuinely shared, unsynchronized,
/// process-wide mutable state (not `threadlocal`), read during AOT
/// dispatch/instantiation — not compilation itself, but audited
/// alongside `aot_bisect.global` since it's the same class of risk.
/// `main.zig` only ever calls `setDebugAotEnabled` once, at
/// single-threaded startup, before any compile/run/dispatch activity
/// begins. An embedder must preserve that invariant — configure this
/// before spawning concurrent work, not while other threads may be
/// reading it via `debugAotEnabled`. No lock guards this field.
var aot_debug_enabled: bool = false;

pub fn setDebugAotEnabled(on: bool) void {
    aot_debug_enabled = on;
}

pub fn debugAotEnabled() bool {
    return aot_debug_enabled;
}

/// Process-global toggle that converts the cross-instance AOT fast-thunk's
/// "caller memory != target memory" warning into a typed trap. Off by
/// default; `main.zig` sets it during startup when `WAMR_TRAP_CROSS_MEMORY_THUNK`
/// is set to a non-empty / non-`0` / non-`false` value. See the dispatcher
/// in `executor.dispatchAotCrossInstance` and #719 Bug B for context.
///
/// #859 thread-safety note: same shape and same contract as
/// `aot_debug_enabled` above — configure once at startup, before any
/// concurrent compile/run/dispatch activity, never mid-flight.
var trap_cross_memory_enabled: bool = false;

pub fn setTrapCrossMemoryEnabled(on: bool) void {
    trap_cross_memory_enabled = on;
}

pub fn trapCrossMemoryEnabled() bool {
    return trap_cross_memory_enabled;
}

/// Caller-supplied instantiation options.
pub const Options = struct {
    precompiled_cores: []const PrecompiledCore = &.{},

    /// When true, the instantiation refuses to silently fall back to
    /// the interpreter on AOT-unresolvable imports / cross-instance
    /// wiring gaps. Instead the failing core surfaces a typed
    /// `error.AotImportUnresolvable` (instance.zig) and the
    /// component-level "force every core to interp on any single
    /// gap" policy is skipped.
    ///
    /// The `wamr run` CLI sets this; library callers (tests,
    /// embedders) leave it `false` to keep the legacy "best effort,
    /// fall back to interp" behaviour. See issue #644.
    aot_only: bool = false,

    /// Find a precompiled artifact for a given core-module slot.
    /// `core_wasm` is the raw module bytes from
    /// `component.core_modules[module_idx].data`. Entries with
    /// `core_wasm` set are matched by slice identity (stable across
    /// re-parses of the same input); entries without (legacy
    /// callers) are matched by `module_idx` alone.
    pub fn findPrecompiled(
        self: Options,
        core_wasm: []const u8,
        module_idx: u32,
    ) ?[]const u8 {
        for (self.precompiled_cores) |pc| {
            if (pc.core_wasm) |cw| {
                if (cw.ptr == core_wasm.ptr and cw.len == core_wasm.len)
                    return pc.cwasm_bytes;
            } else {
                if (pc.module_idx == module_idx) return pc.cwasm_bytes;
            }
        }
        return null;
    }
};

test "Options.findPrecompiled hits + misses" {
    const bytes_a = [_]u8{ 1, 2, 3 };
    const bytes_b = [_]u8{ 4, 5 };
    const pcs = [_]PrecompiledCore{
        .{ .module_idx = 0, .cwasm_bytes = &bytes_a },
        .{ .module_idx = 2, .cwasm_bytes = &bytes_b },
    };
    const opts = Options{ .precompiled_cores = &pcs };
    // Legacy entries (no `core_wasm`) match by `module_idx`; the
    // `core_wasm` arg is ignored for those.
    const dummy: []const u8 = &.{};
    try std.testing.expectEqualSlices(u8, &bytes_a, opts.findPrecompiled(dummy, 0).?);
    try std.testing.expectEqualSlices(u8, &bytes_b, opts.findPrecompiled(dummy, 2).?);
    try std.testing.expectEqual(@as(?[]const u8, null), opts.findPrecompiled(dummy, 1));
    try std.testing.expectEqual(@as(?[]const u8, null), opts.findPrecompiled(dummy, 99));

    const empty_opts = Options{};
    try std.testing.expectEqual(@as(?[]const u8, null), empty_opts.findPrecompiled(dummy, 0));
}

test "Options.findPrecompiled scoped by core_wasm slice identity" {
    // Two modules with the *same* local `module_idx = 0` (composed
    // component case: each sub-component has its own indexing).
    // Distinguished by `core_wasm` slice identity.
    const mod_x_src = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF };
    const mod_y_src = [_]u8{ 0xCA, 0xFE, 0xBA, 0xBE };
    const cwasm_x = [_]u8{ 1, 1, 1 };
    const cwasm_y = [_]u8{ 2, 2, 2 };
    const pcs = [_]PrecompiledCore{
        .{ .module_idx = 0, .cwasm_bytes = &cwasm_x, .core_wasm = &mod_x_src },
        .{ .module_idx = 0, .cwasm_bytes = &cwasm_y, .core_wasm = &mod_y_src },
    };
    const opts = Options{ .precompiled_cores = &pcs };
    try std.testing.expectEqualSlices(u8, &cwasm_x, opts.findPrecompiled(&mod_x_src, 0).?);
    try std.testing.expectEqualSlices(u8, &cwasm_y, opts.findPrecompiled(&mod_y_src, 0).?);
    // A third slice that matches neither (different ptr) returns null
    // even though `module_idx` matches both entries.
    const mod_z_src = [_]u8{ 0, 0, 0, 0 };
    try std.testing.expectEqual(@as(?[]const u8, null), opts.findPrecompiled(&mod_z_src, 0));
}
