//! Component instance — runtime state for an instantiated component.
//!
//! Manages resource tables, canonical function wrappers, and links
//! between component instances and their underlying core module instances.

const std = @import("std");
const ctypes = @import("types.zig");
const core_types = @import("../runtime/common/types.zig");
const executor_mod = @import("executor.zig");
const indexspace = @import("indexspace.zig");

// ── Resource Table ──────────────────────────────────────────────────────────

/// A resource table maps integer handles to host-side representations.
/// Each component instance has its own resource table per resource type.
pub const ResourceTable = struct {
    /// Slot in the resource table.
    const Slot = struct {
        /// The host-side representation value.
        rep: u32,
        /// Whether this handle is currently valid.
        active: bool = true,
        /// Borrow depth — number of outstanding borrows of this handle.
        borrow_count: u32 = 0,
        /// Whether this is an owned handle (vs borrowed).
        owned: bool = true,
    };

    slots: std.ArrayListUnmanaged(Slot) = .empty,
    /// Free list of slot indices for reuse.
    free_list: std.ArrayListUnmanaged(u32) = .empty,

    /// Allocate a new handle for a representation. Returns the handle index.
    pub fn new(self: *ResourceTable, representation: u32, owned: bool, allocator: std.mem.Allocator) !u32 {
        if (self.free_list.items.len > 0) {
            const idx = self.free_list.items[self.free_list.items.len - 1];
            self.free_list.items.len -= 1;
            self.slots.items[idx] = .{ .rep = representation, .owned = owned };
            return idx;
        }
        const idx: u32 = @intCast(self.slots.items.len);
        try self.slots.append(allocator, .{ .rep = representation, .owned = owned });
        return idx;
    }

    /// Get the representation for a handle. Returns null if invalid.
    pub fn rep(self: *const ResourceTable, handle: u32) ?u32 {
        if (handle >= self.slots.items.len) return null;
        const slot = self.slots.items[handle];
        if (!slot.active) return null;
        return slot.rep;
    }

    /// Drop a handle, marking it inactive. Returns the rep for destructor call.
    /// Returns null if the handle was already dropped or is a borrow with outstanding refs.
    pub fn drop(self: *ResourceTable, handle: u32, allocator: std.mem.Allocator) ?u32 {
        if (handle >= self.slots.items.len) return null;
        const slot = &self.slots.items[handle];
        if (!slot.active) return null;
        if (slot.borrow_count > 0) return null; // can't drop with outstanding borrows
        const r = slot.rep;
        slot.active = false;
        // Add to free list for reuse
        self.free_list.append(allocator, handle) catch {};
        return r;
    }

    /// Increment borrow count for a handle.
    pub fn borrow(self: *ResourceTable, handle: u32) bool {
        if (handle >= self.slots.items.len) return false;
        const slot = &self.slots.items[handle];
        if (!slot.active) return false;
        slot.borrow_count += 1;
        return true;
    }

    /// Decrement borrow count for a handle.
    pub fn returnBorrow(self: *ResourceTable, handle: u32) void {
        if (handle >= self.slots.items.len) return;
        const slot = &self.slots.items[handle];
        if (slot.borrow_count > 0) slot.borrow_count -= 1;
    }

    pub fn deinit(self: *ResourceTable, allocator: std.mem.Allocator) void {
        self.slots.deinit(allocator);
        self.free_list.deinit(allocator);
    }
};

// ── Component Instance ──────────────────────────────────────────────────────

/// Host-facing type of an interface value passed across a component boundary.
pub const InterfaceValue = @import("canonical_abi.zig").InterfaceValue;

/// A host-provided function that satisfies a component `func` import.
///
/// Phase 2A will invoke this from a canon-lowered core trampoline, after
/// the trampoline has decoded the core ABI representation into component-level
/// `InterfaceValue`s. For Phase 1B the field is captured but never called.
///
/// Memory ownership for compound result values (strings, lists, records)
/// follows the standard canonical-ABI convention: the callee allocates
/// into `allocator`; the trampoline that invoked the host func owns the
/// resulting values and is responsible for lifting them back into core memory.
pub const HostFunc = struct {
    /// Opaque context pointer forwarded to `call`.
    context: ?*anyopaque = null,
    /// Host-side implementation. Null is legal in tests where the call path
    /// is not exercised; `linkImports` does not require it to be set.
    call: ?*const fn (
        ctx: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: std.mem.Allocator,
    ) anyerror!void = null,
    /// Optional component-level function type index, for Phase 2B validation.
    type_idx: ?u32 = null,
};

/// A member of a host-provided instance binding.
pub const HostInstanceMember = union(enum) {
    func: HostFunc,
    /// Placeholder for host-side resource-type identity. For Phase 1B we
    /// carry only the raw component resource-type index the host claims to
    /// implement; real opaque identity / lift/lower glue lands in Phase 2A.
    resource_type: u32,
};

/// A host binding for an instance-typed component import. The members map
/// names (e.g. `"[method]output-stream.write"`) to host implementations.
///
/// `ComponentInstance` stores these by borrowed pointer — callers must keep
/// the `HostInstance` alive for at least the lifetime of every
/// `ComponentInstance` whose imports point at it.
pub const HostInstance = struct {
    members: std.StringHashMapUnmanaged(HostInstanceMember) = .empty,

    pub fn deinit(self: *HostInstance, allocator: std.mem.Allocator) void {
        self.members.deinit(allocator);
    }
};

/// Binding for a component import — either a host-provided callback,
/// a host-provided instance (member map), or a reference to another
/// component instance's export.
pub const ImportBinding = union(enum) {
    /// A host-provided function.
    host_func: HostFunc,
    /// A host-provided instance (WASI p2 top-level imports are always
    /// instance-typed; this is the common Phase 2B path).
    host_instance: *const HostInstance,
    /// A reference to another ComponentInstance's exported function.
    component_export: struct {
        instance: *const ComponentInstance,
        func_name: []const u8,
    },
};

/// A runtime component instance — the result of instantiating a Component.
pub const ComponentInstance = struct {
    /// The parsed component this instance was created from.
    component: *const ctypes.Component,
    /// Core module instances created during instantiation.
    core_instances: []CoreInstanceEntry,
    /// Resource tables, keyed by the raw component resource type index
    /// (as referenced by `canon resource.{new,drop,rep}`). Allocated
    /// lazily on first access so the dense `[]ResourceTable` layout (which
    /// silently assumed resource indices were dense over locally declared
    /// resources) no longer corrupts on aliased or imported resources.
    resource_tables: std.AutoHashMapUnmanaged(u32, ResourceTable),
    /// Exported functions (component-level func index → core func index + instance).
    exported_funcs: std.StringHashMapUnmanaged(ExportedFunc),
    /// Resolved imports keyed by import name.
    imports: std.StringHashMapUnmanaged(ImportBinding),
    /// Arena used for core-module loader allocations. Core wasm binaries
    /// parsed from this component have their types/imports/exports/etc.
    /// arrays allocated into this arena, which is destroyed as one unit on
    /// `deinit`. Mirrors the pattern `api/wamr.zig` uses for top-level
    /// module loads.
    module_arena: std.heap.ArenaAllocator,
    /// Canon-lower trampoline contexts owned by this instance. Each entry
    /// is referenced as the `ctx` of an installed `HostFnEntry` on some core
    /// module instance; we keep the slice here so lifetimes are tied to the
    /// `ComponentInstance` and freed together on `deinit`.
    trampoline_ctxs: std.ArrayListUnmanaged(*executor_mod.ComponentTrampolineCtx) = .empty,
    /// Pending core-module start functions whose execution was deferred
    /// during `instantiate` so canon-lower trampoline `host_funcs` can be
    /// bound by `linkImports` first. Drained by `linkImports` in core-instance
    /// order; see `runDeferredCoreStarts` (issue #308).
    pending_core_starts: std.ArrayListUnmanaged(*core_types.ModuleInstance) = .empty,
    /// Whether the start function has been executed.
    started: bool = false,
    /// Allocator for instance lifetime.
    allocator: std.mem.Allocator,

    pub const CoreInstanceEntry = struct {
        module_inst: ?*core_types.ModuleInstance = null,
        /// When this entry corresponds to a `CoreInstanceExpr.exports` (an
        /// inline instance bundling named core items rather than an actual
        /// core-module instantiation), the named items live here. `module_inst`
        /// is null in that case.
        inline_exports: []const ctypes.CoreInlineExport = &.{},
    };

    pub const ExportedFunc = struct {
        core_instance_idx: u32,
        core_func_idx: u32,
        /// Component-level function type index (into component.types).
        func_type_idx: u32 = 0,
        /// Canonical options from the canon lift definition.
        opts: []const ctypes.CanonOpt = &.{},
    };

    /// Look up an exported function by name.
    pub fn getExport(self: *const ComponentInstance, name: []const u8) ?ExportedFunc {
        return self.exported_funcs.get(name);
    }

    /// Look up a resolved import by name.
    pub fn getImport(self: *const ComponentInstance, name: []const u8) ?ImportBinding {
        return self.imports.get(name);
    }

    /// Get-or-create the resource table for a given component resource type
    /// index. Resource tables are lazily allocated on first use.
    pub fn getOrCreateResourceTable(self: *ComponentInstance, type_idx: u32) !*ResourceTable {
        const gop = try self.resource_tables.getOrPut(self.allocator, type_idx);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        return gop.value_ptr;
    }

    /// Find the first core instance entry with a real `ModuleInstance`.
    /// The component's "canonical" memory always lives on a real
    /// instance — the inline-exports entries used to wire imports never
    /// own a module. This is the lookup both `componentTrampoline` and
    /// `readGuestBytes` use to resolve guest memory.
    pub fn firstModuleInst(self: *const ComponentInstance) ?*core_types.ModuleInstance {
        for (self.core_instances) |entry| {
            if (entry.module_inst) |mi| return mi;
        }
        return null;
    }

    /// Resolve a top-level core memory indexspace index to the underlying
    /// `*MemoryInstance`. Used by `componentTrampoline` and host-side
    /// helpers to find the memory referenced by a `(memory N)` canonical
    /// option, where `N` is the component-level core memory index — which
    /// may be contributed by an `alias core export` decl pointing at a
    /// memory exported by a different core instance than the "first" one.
    ///
    /// Resolution order:
    ///   1. If `N` is contributed by an `alias core export`, follow it
    ///      through the source core instance to the underlying memory.
    ///   2. Otherwise, fall back to `firstModuleInst().getMemory(N)` —
    ///      preserves behavior for hand-authored fixtures with a single
    ///      core module (where the local memory idx matches N).
    pub fn resolveTopLevelMemory(self: *const ComponentInstance, idx: u32) ?*core_types.MemoryInstance {
        const ref = indexspace.resolveCoreMemory(self.component, idx) orelse {
            const mi = self.firstModuleInst() orelse return null;
            return mi.getMemory(idx);
        };
        const ie = self.component.aliases[ref.aliased].instance_export;
        if (ie.instance_idx >= self.core_instances.len) return null;
        const src_mi = self.core_instances[ie.instance_idx].module_inst orelse return null;
        const exp = src_mi.module.findExport(ie.name, .memory) orelse return null;
        if (exp.index >= src_mi.memories.len) return null;
        return src_mi.memories[exp.index];
    }

    /// Resolve a top-level core func indexspace index to a callable
    /// `(*ModuleInstance, local_func_idx)` pair, suitable for invoking
    /// from a host context (e.g. calling `cabi_realloc`). Only the
    /// alias-core-export contributors yield a directly-callable function;
    /// canon.lower / resource.* canons are imports and return null.
    pub fn resolveTopLevelCoreFunc(
        self: *const ComponentInstance,
        idx: u32,
    ) ?struct { mi: *core_types.ModuleInstance, local_idx: u32 } {
        const ref = indexspace.resolveCoreFunc(self.component, idx) orelse return null;
        switch (ref) {
            .lowered, .resource_drop, .resource_new, .resource_rep => return null,
            .aliased => |alias_idx| {
                const ie = self.component.aliases[alias_idx].instance_export;
                if (ie.instance_idx >= self.core_instances.len) return null;
                const mi = self.core_instances[ie.instance_idx].module_inst orelse return null;
                const local = mi.getExportFunc(ie.name) orelse return null;
                return .{ .mi = mi, .local_idx = local };
            },
        }
    }

    /// Read `len` bytes starting at guest linear-memory offset `ptr` from the
    /// canonical memory of this component. Used by host adapter callbacks
    /// invoked from `componentTrampoline` to materialize `list<u8>` /
    /// `string` arguments whose flat representation is `(ptr, len)` into
    /// guest memory.
    ///
    /// Phase 2B narrow assumption: the canonical memory lives on the
    /// first core instance with a real `module_inst` at memory index 0.
    /// Returns null if no such instance is present, the memory is missing,
    /// or the slice is out of bounds.
    pub fn readGuestBytes(self: *const ComponentInstance, ptr: u32, len: u32) ?[]const u8 {
        const mem = self.canonicalMemory() orelse return null;
        const end = @as(usize, ptr) + @as(usize, len);
        if (end > mem.data.len) return null;
        return mem.data[ptr..end];
    }

    /// Return the "canonical" guest memory: the one that
    /// `cabi_realloc` allocates into and that lift/lower of compound
    /// types reads/writes through. Prefers a top-level core memory 0
    /// (which under wit-component output is `alias core export $main
    /// "memory"`); falls back to the first module instance's memory[0]
    /// for legacy hand-authored fixtures.
    pub fn canonicalMemory(self: *const ComponentInstance) ?*core_types.MemoryInstance {
        if (self.resolveTopLevelMemory(0)) |m| return m;
        const mi = self.firstModuleInst() orelse return null;
        return mi.getMemory(0);
    }

    /// Locate the module instance that owns `cabi_realloc`. wit-component
    /// emits `cabi_realloc` on the same core module that owns the
    /// canonical memory (`$main` in stdio-echo); legacy fixtures put it
    /// on the only module instance.
    fn reallocOwner(self: *const ComponentInstance) ?*core_types.ModuleInstance {
        // First try: walk all core instances looking for one that exports
        // `cabi_realloc`. This is robust regardless of which instance the
        // canonical memory aliases through.
        for (self.core_instances) |entry| {
            const mi = entry.module_inst orelse continue;
            if (mi.getExportFunc("cabi_realloc") != null) return mi;
        }
        return null;
    }

    /// Allocate `len` bytes inside the canonical guest linear memory and
    /// copy `bytes` into them. Returns the guest-side pointer or null on
    /// failure (no `cabi_realloc` export, OOM, or invocation error).
    ///
    /// Used by host-side callbacks (e.g. `wasi:io/streams.[method]
    /// input-stream.blocking-read`) that must materialize a `list<u8>`
    /// or `string` value into guest memory before the canonical ABI
    /// stores its `(ptr, len)` representation in a spilled result tuple.
    ///
    /// Convention: wit-bindgen emits a single `cabi_realloc` export on
    /// the main core module; we call it with `(0, 0, align=1, len)` to
    /// allocate fresh space.
    pub fn hostAllocAndWrite(self: *ComponentInstance, bytes: []const u8) ?u32 {
        const realloc_owner = self.reallocOwner() orelse return null;
        const realloc_local = realloc_owner.getExportFunc("cabi_realloc") orelse return null;
        const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
        const executor = @import("executor.zig");
        const env = ExecEnv.create(realloc_owner, 4096, self.allocator) catch return null;
        defer env.destroy();
        const ptr = executor.callRealloc(env, realloc_local, 0, 0, 1, @intCast(bytes.len)) catch return null;
        const mem = self.canonicalMemory() orelse return null;
        const end = @as(usize, ptr) + bytes.len;
        if (end > mem.data.len) return null;
        @memcpy(mem.data[ptr..end], bytes);
        return ptr;
    }

    /// Kind-classify a top-level component import. Pure-type imports (those
    /// whose sole purpose is to introduce a type index) do not need a runtime
    /// binding; every other kind must be satisfied by `linkImports`.
    fn importIsRuntime(imp: ctypes.ImportDecl) bool {
        return switch (imp.desc) {
            .type => false,
            .module, .func, .value, .component, .instance => true,
        };
    }

    /// Validate that `binding` is compatible with `imp.desc`'s kind. Returns
    /// `error.ImportKindMismatch` for an outright mismatch. Cross-component
    /// wiring via `component_export` is accepted for any runtime kind and is
    /// validated more thoroughly at call time.
    fn importKindMatches(imp: ctypes.ImportDecl, binding: ImportBinding) bool {
        return switch (binding) {
            .component_export => true,
            .host_func => imp.desc == .func,
            .host_instance => imp.desc == .instance,
        };
    }

    /// Link imports against a set of provided bindings.
    ///
    /// Every runtime (non-type-only) top-level import must have a provider;
    /// missing ones fail with `error.MissingImport`. Kind mismatches (e.g.
    /// binding a `host_func` to an instance-typed import) fail with
    /// `error.ImportKindMismatch`. Pure `.type` imports are satisfied by the
    /// type system and require no runtime binding.
    ///
    /// Side effect: after binding, this drains `pending_core_starts` and
    /// runs each deferred core-module `(start ...)` directive in original
    /// declaration order. This means `linkImports` can return guest-trap
    /// errors (surfaced as `error.StartFunctionFailed` plus a
    /// `[component init trap] ...` line on stderr), and a failure leaves
    /// the instance only partially started — callers should treat such an
    /// instance as poisoned and `deinit` it. The deferral exists so that
    /// trampoline `host_funcs` are bound before any `(start)` runs;
    /// otherwise wasi-using `_initialize` traps with `HostFuncNotBound`
    /// (issue #308).
    pub fn linkImports(
        self: *ComponentInstance,
        providers: std.StringHashMapUnmanaged(ImportBinding),
    ) !void {
        for (self.component.imports) |imp| {
            const maybe_binding = providers.get(imp.name);
            if (maybe_binding) |binding| {
                if (!importKindMatches(imp, binding)) return error.ImportKindMismatch;
                self.imports.put(self.allocator, imp.name, binding) catch
                    return error.OutOfMemory;
            } else if (importIsRuntime(imp)) {
                return error.MissingImport;
            }
        }

        // Fill in trampoline host_funcs now that bindings are in place.
        // Each trampoline records the component func index it lowers; we
        // walk the component's imports to find the matching host binding.
        //
        // Note: `resolveComponentFuncToHostFunc` only resolves bindings of
        // kind `.host_func` and `.host_instance.members.func`. Imports bound
        // via `.component_export` (cross-component composition) leave the
        // trampoline `host_func` unset, which would surface as
        // `error.HostFuncNotBound` if a deferred core start calls them.
        // Cross-component composition with deferred starts is not currently
        // exercised by any caller; revisit when that combination is wired.
        for (self.trampoline_ctxs.items) |ctx| {
            if (resolveComponentFuncToHostFunc(self, self.component, ctx.component_func_idx)) |hf| {
                ctx.host_func = hf;
            } else {
            }
        }

        // Now run any core-module `(start ...)` directives that were
        // deferred during `instantiate`. Trampoline `host_funcs` are bound
        // above, so wasi imports invoked from within `_initialize` etc.
        // resolve correctly (issue #308).
        try self.runDeferredCoreStarts();
    }

    /// Execute any core-module `(start ...)` directives whose dispatch was
    /// deferred during `instantiate`, in the original core-instance order.
    /// Drains `pending_core_starts`. Surfaces the underlying trap as
    /// `error.StartFunctionFailed` (with the diagnostic on `env.host_trap`
    /// preserved on the per-start `ExecEnv` printed by `callComponentFunc`
    /// downstream — but for instantiation-time failures the diagnostic is
    /// printed here so a failed `_initialize` doesn't masquerade as a
    /// later runtime error).
    fn runDeferredCoreStarts(self: *ComponentInstance) !void {
        const inst_mod = @import("../runtime/interpreter/instance.zig");
        const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
        const interp = @import("../runtime/interpreter/interp.zig");
        defer self.pending_core_starts.clearRetainingCapacity();

        for (self.pending_core_starts.items) |mi| {
            const start_idx = mi.module.start_function orelse continue;
            const env = ExecEnv.create(mi, 4096, self.allocator) catch return error.OutOfMemory;
            defer env.destroy();
            interp.executeFunction(env, start_idx) catch {
                if (env.host_trap) |ht| {
                    std.debug.print(
                        "[component init trap] core_func_idx={d} import='{s}.{s}' stage={s} error={s}\n",
                        .{ ht.core_func_idx, ht.import_module_name, ht.import_field_name, @tagName(ht.stage), ht.err_name },
                    );
                }
                return inst_mod.InstantiationError.StartFunctionFailed;
            };
        }
    }

    /// Execute the component's start function if one is defined and not yet run.
    pub fn executeStart(self: *ComponentInstance) !void {
        if (self.started) return;
        self.started = true;

        const start = self.component.start orelse return;

        // The start function references a canon index which should be
        // a canon lift that we've already mapped to an exported func.
        // Walk exports to find the matching canon func.
        if (start.func_idx < self.component.canons.len) {
            const canon = self.component.canons[start.func_idx];
            switch (canon) {
                .lift => |lift| {
                    if (self.core_instances.len > 0) {
                        if (self.core_instances[0].module_inst) |mod_inst| {
                            const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
                            const interp = @import("../runtime/interpreter/interp.zig");
                            const env = ExecEnv.create(mod_inst, 8192, self.allocator) catch return;
                            defer env.destroy();
                            interp.executeFunction(env, lift.core_func_idx) catch return;
                        }
                    }
                },
                else => {},
            }
        }
    }

    pub fn deinit(self: *ComponentInstance) void {
        var rt_it = self.resource_tables.valueIterator();
        while (rt_it.next()) |rt| rt.deinit(self.allocator);
        self.resource_tables.deinit(self.allocator);
        for (self.trampoline_ctxs.items) |ctx| {
            ctx.deinit(self.allocator);
            self.allocator.destroy(ctx);
        }
        self.trampoline_ctxs.deinit(self.allocator);
        self.pending_core_starts.deinit(self.allocator);
        if (self.core_instances.len > 0) {
            const inst_mod = @import("../runtime/interpreter/instance.zig");
            for (self.core_instances) |entry| {
                if (entry.module_inst) |mi| inst_mod.destroy(mi);
            }
            self.allocator.free(self.core_instances);
        }
        self.module_arena.deinit();
        self.exported_funcs.deinit(self.allocator);
        self.imports.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};

// ── Component Instantiation ─────────────────────────────────────────────────

pub const InstantiationError = error{
    OutOfMemory,
    InvalidComponent,
    CoreModuleLoadFailed,
    CoreModuleInstantiateFailed,
    ImportResolutionFailed,
    MissingImport,
    ImportKindMismatch,
};

/// Instantiate a parsed component, producing a runnable ComponentInstance.
///
/// When `component.core_instances` is populated, each expression is
/// processed in order:
///   - `.exports` contributes an inline instance whose named members are
///     recorded on the ComponentInstance entry (no core module instantiation).
///   - `.instantiate { module_idx, args }` loads and instantiates the
///     referenced core module, resolving each of its imports against the
///     prior core-instance exports named by `args`. Whenever an imported
///     core function is satisfied by a `canon.lower`, the runtime installs
///     a `componentTrampoline` + per-slot `ComponentTrampolineCtx` on that
///     import slot so future core calls bridge back to the host `HostFunc`.
///
/// For legacy callers with `core_instances.len == 0`, falls back to the
/// pre-2A behaviour of instantiating each `core_module` exactly once.
pub fn instantiate(
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
) InstantiationError!*ComponentInstance {
    const inst = allocator.create(ComponentInstance) catch return error.OutOfMemory;
    errdefer allocator.destroy(inst);

    inst.* = .{
        .component = component,
        .module_arena = std.heap.ArenaAllocator.init(allocator),
        .core_instances = &.{},
        .resource_tables = .empty,
        .exported_funcs = .{},
        .imports = .{},
        .allocator = allocator,
    };

    const loader = @import("../runtime/interpreter/loader.zig");
    const inst_mod = @import("../runtime/interpreter/instance.zig");

    if (component.core_instances.len > 0) {
        // Section-aware path: walk core_instances expressions in order.
        const cis = allocator.alloc(ComponentInstance.CoreInstanceEntry, component.core_instances.len) catch return error.OutOfMemory;
        for (cis) |*entry| entry.* = .{};
        inst.core_instances = cis;

        for (component.core_instances, 0..) |expr, ci_idx| {
            switch (expr) {
                .exports => |inline_exports| {
                    cis[ci_idx] = .{ .inline_exports = inline_exports };
                },
                .instantiate => |ie| {
                    if (ie.module_idx >= component.core_modules.len) continue;
                    const core_mod = component.core_modules[ie.module_idx];

                    const mod_alloc = inst.module_arena.allocator();
                    const loaded = loader.load(core_mod.data, mod_alloc) catch continue;
                    const module_ptr = mod_alloc.create(core_types.WasmModule) catch continue;
                    module_ptr.* = loaded;

                    // Resolve every import against `with` args BEFORE
                    // instantiation. Three backends:
                    //   * Cross-instance: the source `with`-arg is a real core
                    //     module instance; wire the appropriate per-kind slot
                    //     (`import_functions` / `memories` / `tables` /
                    //     `globals`) so the interpreter dispatches into the
                    //     source's body / shares the same MemoryInstance etc.
                    //   * Canon.lower (function-only): the source is an
                    //     inline-exports bundle pointing at a `canon lower`
                    //     core func; install a `componentTrampoline` on
                    //     `host_func_entries[i]`.
                    //   * Unresolved: function slots fall through to a no-op
                    //     stub; non-function unresolved imports trap at
                    //     instantiation (current `instantiateWithImports`
                    //     contract — caller must satisfy them).
                    const import_func_count = module_ptr.import_function_count;
                    const import_mem_count = module_ptr.import_memory_count;
                    const import_tbl_count = module_ptr.import_table_count;
                    const import_glob_count = module_ptr.import_global_count;
                    var entries: []?core_types.HostFnEntry = &.{};
                    var imps_buf: []core_types.ImportedFunction = &.{};
                    var is_cross: []bool = &.{};
                    var mems_buf: []*core_types.MemoryInstance = &.{};
                    var tbls_buf: []*core_types.TableInstance = &.{};
                    var globs_buf: []*core_types.GlobalInstance = &.{};
                    var first_cross_src: ?*core_types.ModuleInstance = null;
                    var has_imports_resolved = false;
                    if (import_func_count > 0) {
                        entries = allocator.alloc(?core_types.HostFnEntry, import_func_count) catch continue;
                        @memset(entries, null);
                        imps_buf = allocator.alloc(core_types.ImportedFunction, import_func_count) catch {
                            allocator.free(entries);
                            continue;
                        };
                        is_cross = allocator.alloc(bool, import_func_count) catch {
                            allocator.free(entries);
                            allocator.free(imps_buf);
                            continue;
                        };
                        @memset(is_cross, false);
                    }
                    // Per-kind resolution buffers; allocated lazily so a module
                    // with e.g. zero memory imports never touches the allocator.
                    if (import_mem_count > 0) {
                        mems_buf = allocator.alloc(*core_types.MemoryInstance, import_mem_count) catch {
                            if (entries.len > 0) allocator.free(entries);
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            continue;
                        };
                    }
                    if (import_tbl_count > 0) {
                        tbls_buf = allocator.alloc(*core_types.TableInstance, import_tbl_count) catch {
                            if (entries.len > 0) allocator.free(entries);
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            if (mems_buf.len > 0) allocator.free(mems_buf);
                            continue;
                        };
                    }
                    if (import_glob_count > 0) {
                        globs_buf = allocator.alloc(*core_types.GlobalInstance, import_glob_count) catch {
                            if (entries.len > 0) allocator.free(entries);
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            if (mems_buf.len > 0) allocator.free(mems_buf);
                            if (tbls_buf.len > 0) allocator.free(tbls_buf);
                            continue;
                        };
                    }

                    // Walk imports once, dispatching by kind. Per-kind running
                    // index counters track the position within the per-kind
                    // import sequence — the interpreter's `ImportContext`
                    // fields are indexed by these (not by the global import
                    // index).
                    var imp_func_idx: u32 = 0;
                    var imp_mem_idx: u32 = 0;
                    var imp_tbl_idx: u32 = 0;
                    var imp_glob_idx: u32 = 0;
                    for (module_ptr.imports) |imp| {
                        // Find the `with` arg whose name matches this import's
                        // wasm "module" string (component-model `with` keys).
                        const source_inst_idx: u32 = arg_blk: {
                            for (ie.args) |arg| {
                                if (std.mem.eql(u8, arg.name, imp.module_name)) break :arg_blk arg.instance_idx;
                            }
                            break :arg_blk std.math.maxInt(u32);
                        };

                        switch (imp.kind) {
                            .function => {
                                defer imp_func_idx += 1;
                                if (source_inst_idx == std.math.maxInt(u32)) continue;
                                if (source_inst_idx >= ci_idx) continue;
                                const source_entry = cis[source_inst_idx];

                                if (source_entry.module_inst) |src_mi| {
                                    const target_func_idx = src_mi.getExportFunc(imp.field_name) orelse continue;
                                    imps_buf[imp_func_idx] = .{ .module_inst = src_mi, .func_idx = target_func_idx };
                                    is_cross[imp_func_idx] = true;
                                    if (first_cross_src == null) first_cross_src = src_mi;
                                    has_imports_resolved = true;
                                    continue;
                                }

                                // Inline-exports source — member is a core func that is either:
                                //   (a) a `canon.lower` (host trampoline), or
                                //   (b) an `alias core export` of another core instance's
                                //       func (cross-instance wiring, e.g. shim's exports
                                //       routed into $main via wit-component's shim/fixup
                                //       pattern).
                                var member_sort_idx: ?ctypes.CoreSort = null;
                                var member_idx: u32 = 0;
                                for (source_entry.inline_exports) |mem| {
                                    if (std.mem.eql(u8, mem.name, imp.field_name)) {
                                        member_sort_idx = mem.sort_idx.sort;
                                        member_idx = mem.sort_idx.idx;
                                        break;
                                    }
                                }
                                if (member_sort_idx == null) continue;
                                if (member_sort_idx.? != .func) continue;

                                const cfref = indexspace.resolveCoreFunc(component, member_idx) orelse continue;
                                // Aliased core func — resolve to the underlying
                                // {module_inst, func_idx} pair from a previously
                                // instantiated core instance.
                                switch (cfref) {
                                    .aliased => |alias_idx| {
                                        const al = component.aliases[alias_idx];
                                        const ie_al = switch (al) {
                                            .instance_export => |x| x,
                                            else => continue,
                                        };
                                        if (ie_al.instance_idx >= ci_idx) continue;
                                        const al_src = cis[ie_al.instance_idx];
                                        const al_mi = al_src.module_inst orelse continue;
                                        const al_func_idx = al_mi.getExportFunc(ie_al.name) orelse continue;
                                        imps_buf[imp_func_idx] = .{ .module_inst = al_mi, .func_idx = al_func_idx };
                                        is_cross[imp_func_idx] = true;
                                        if (first_cross_src == null) first_cross_src = al_mi;
                                        has_imports_resolved = true;
                                        continue;
                                    },
                                    .lowered => {},
                                    else => continue,
                                }
                                const canon_idx = cfref.lowered;
                                const canon = component.canons[canon_idx];
                                const lower = switch (canon) {
                                    .lower => |l| l,
                                    else => continue,
                                };

                                const ctx_ptr = allocator.create(executor_mod.ComponentTrampolineCtx) catch continue;
                                // Prefer name-based lookup (correct for real components).
                                // Fall back to direct types[lower.func_idx] indexing for
                                // hand-authored fixtures that put the FuncType at that
                                // top-level type slot without a nested instance-type body.
                                const rft_opt: ?ResolvedFuncType = resolveCompFuncType(component, lower.func_idx) orelse blk: {
                                    if (lower.func_idx >= component.types.len) break :blk null;
                                    break :blk switch (component.types[lower.func_idx]) {
                                        .func => |f| ResolvedFuncType{ .ft = f },
                                        else => null,
                                    };
                                };
                                const rft = rft_opt orelse {
                                    allocator.destroy(ctx_ptr);
                                    continue;
                                };
                                // When the FuncType came from an instance-type body, its
                                // param/result `.type_idx` references — and any nested
                                // structural type indices — are local to that body's
                                // type indexspace. Build a per-trampoline TypeRegistry
                                // extension that materializes the local type space at
                                // an absolute offset, then rebase param/result ValTypes
                                // to absolute indices that the registry can resolve.
                                const ext_base: u32 = if (component.type_indexspace.len > 0)
                                    @intCast(component.type_indexspace.len)
                                else
                                    @intCast(component.types.len);
                                const ext: InstanceTypeExtension = if (rft.decls) |decls|
                                    buildInstanceTypeExtension(allocator, decls, ext_base) catch {
                                        allocator.destroy(ctx_ptr);
                                        continue;
                                    }
                                else
                                    InstanceTypeExtension.empty();
                                const ft: ctypes.FuncType = rft.ft;
                                const params = allocator.alloc(ctypes.ValType, ft.params.len) catch {
                                    ext.deinit(allocator, true);
                                    allocator.destroy(ctx_ptr);
                                    continue;
                                };
                                for (ft.params, 0..) |p, i| {
                                    params[i] = if (rft.decls != null)
                                        rewriteValTypeAbsolute(ext_base, p.type)
                                    else
                                        p.type;
                                }
                                const results = switch (ft.results) {
                                    .none => allocator.alloc(ctypes.ValType, 0) catch {
                                        allocator.free(params);
                                        ext.deinit(allocator, true);
                                        allocator.destroy(ctx_ptr);
                                        continue;
                                    },
                                    .unnamed => |t| blk2: {
                                        const r = allocator.alloc(ctypes.ValType, 1) catch {
                                            allocator.free(params);
                                            ext.deinit(allocator, true);
                                            allocator.destroy(ctx_ptr);
                                            continue;
                                        };
                                        r[0] = if (rft.decls != null) rewriteValTypeAbsolute(ext_base, t) else t;
                                        break :blk2 r;
                                    },
                                    .named => |named| blk2: {
                                        const r = allocator.alloc(ctypes.ValType, named.len) catch {
                                            allocator.free(params);
                                            ext.deinit(allocator, true);
                                            allocator.destroy(ctx_ptr);
                                            continue;
                                        };
                                        for (named, 0..) |n, i| {
                                            r[i] = if (rft.decls != null)
                                                rewriteValTypeAbsolute(ext_base, n.type)
                                            else
                                                n.type;
                                        }
                                        break :blk2 r;
                                    },
                                };
                                ctx_ptr.* = .{
                                    .comp_inst = inst,
                                    .host_func = .{},
                                    .component_func_idx = lower.func_idx,
                                    .param_types = params,
                                    .result_types = results,
                                    .lower_opts = executor_mod.LowerOptions.fromOpts(lower.opts),
                                    .extended_types = ext.extension_types,
                                    .extended_indexspace = ext.extension_indexspace,
                                };
                                inst.trampoline_ctxs.append(allocator, ctx_ptr) catch {
                                    ctx_ptr.deinit(allocator);
                                    allocator.destroy(ctx_ptr);
                                    continue;
                                };

                                entries[imp_func_idx] = .{
                                    .func = &executor_mod.componentTrampoline,
                                    .ctx = @ptrCast(ctx_ptr),
                                };
                            },
                            .memory => {
                                defer imp_mem_idx += 1;
                                if (source_inst_idx == std.math.maxInt(u32)) continue;
                                if (source_inst_idx >= ci_idx) continue;
                                const source_entry = cis[source_inst_idx];
                                if (source_entry.module_inst) |src_mi| {
                                    const exp = src_mi.module.findExport(imp.field_name, .memory) orelse continue;
                                    if (exp.index >= src_mi.memories.len) continue;
                                    mems_buf[imp_mem_idx] = src_mi.memories[exp.index];
                                    if (first_cross_src == null) first_cross_src = src_mi;
                                    has_imports_resolved = true;
                                    continue;
                                }
                                // Inline-exports source: member references a
                                // top-level core memory contributed by an
                                // `alias core export` decl. Follow the alias
                                // back to the original module instance.
                                for (source_entry.inline_exports) |mem| {
                                    if (!std.mem.eql(u8, mem.name, imp.field_name)) continue;
                                    if (mem.sort_idx.sort != .memory) break;
                                    const mi_ptr = resolveCoreMemoryToMI(inst, component, mem.sort_idx.idx) orelse break;
                                    mems_buf[imp_mem_idx] = mi_ptr;
                                    has_imports_resolved = true;
                                    break;
                                }
                            },
                            .table => {
                                defer imp_tbl_idx += 1;
                                if (source_inst_idx == std.math.maxInt(u32)) continue;
                                if (source_inst_idx >= ci_idx) continue;
                                const source_entry = cis[source_inst_idx];
                                if (source_entry.module_inst) |src_mi| {
                                    const exp = src_mi.module.findExport(imp.field_name, .table) orelse continue;
                                    if (exp.index >= src_mi.tables.len) continue;
                                    tbls_buf[imp_tbl_idx] = src_mi.tables[exp.index];
                                    if (first_cross_src == null) first_cross_src = src_mi;
                                    has_imports_resolved = true;
                                    continue;
                                }
                                for (source_entry.inline_exports) |mem| {
                                    if (!std.mem.eql(u8, mem.name, imp.field_name)) continue;
                                    if (mem.sort_idx.sort != .table) break;
                                    const t_ptr = resolveCoreTableToMI(inst, component, mem.sort_idx.idx) orelse break;
                                    tbls_buf[imp_tbl_idx] = t_ptr;
                                    has_imports_resolved = true;
                                    break;
                                }
                            },
                            .global => {
                                defer imp_glob_idx += 1;
                                if (source_inst_idx == std.math.maxInt(u32)) continue;
                                if (source_inst_idx >= ci_idx) continue;
                                const source_entry = cis[source_inst_idx];
                                if (source_entry.module_inst) |src_mi| {
                                    const exp = src_mi.module.findExport(imp.field_name, .global) orelse continue;
                                    if (exp.index >= src_mi.globals.len) continue;
                                    globs_buf[imp_glob_idx] = src_mi.globals[exp.index];
                                    if (first_cross_src == null) first_cross_src = src_mi;
                                    has_imports_resolved = true;
                                    continue;
                                }
                                for (source_entry.inline_exports) |mem| {
                                    if (!std.mem.eql(u8, mem.name, imp.field_name)) continue;
                                    if (mem.sort_idx.sort != .global) break;
                                    const g_ptr = resolveCoreGlobalToMI(inst, component, mem.sort_idx.idx) orelse break;
                                    globs_buf[imp_glob_idx] = g_ptr;
                                    has_imports_resolved = true;
                                    break;
                                }
                            },
                            else => {},
                        }
                    }

                    // Instantiate, optionally seeding `import_functions` for
                    // cross-instance wiring. Slots that aren't cross-instance
                    // get a safe placeholder pointing at any cross-source we
                    // saw — interp dispatch never reaches them because their
                    // `host_func_entries[i]` is non-null (canon.lower) or the
                    // import is unresolved (caught by the no-op stub before
                    // `import_functions` is consulted).
                    const mi = blk: {
                        if (has_imports_resolved or first_cross_src != null) {
                            if (first_cross_src) |placeholder| {
                                for (imps_buf, 0..) |*slot, i| {
                                    if (!is_cross[i]) {
                                        slot.* = .{ .module_inst = placeholder, .func_idx = 0 };
                                    }
                                }
                            }
                            const ctx = inst_mod.ImportContext{
                                .functions = imps_buf,
                                .memories = mems_buf,
                                .tables = tbls_buf,
                                .globals = globs_buf,
                            };
                            // Defer core `(start ...)` execution so canon-lower
                            // trampoline `host_funcs` are bound by `linkImports`
                            // before any start runs (issue #308).
                            break :blk inst_mod.instantiateWithOptions(module_ptr, allocator, .{
                                .import_ctx = ctx,
                                .defer_start = true,
                            }) catch {
                                if (entries.len > 0) allocator.free(entries);
                                if (imps_buf.len > 0) allocator.free(imps_buf);
                                if (is_cross.len > 0) allocator.free(is_cross);
                                if (mems_buf.len > 0) allocator.free(mems_buf);
                                if (tbls_buf.len > 0) allocator.free(tbls_buf);
                                if (globs_buf.len > 0) allocator.free(globs_buf);
                                continue;
                            };
                        }
                        break :blk inst_mod.instantiateWithOptions(module_ptr, allocator, .{
                            .defer_start = true,
                        }) catch {
                            if (entries.len > 0) allocator.free(entries);
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            if (mems_buf.len > 0) allocator.free(mems_buf);
                            if (tbls_buf.len > 0) allocator.free(tbls_buf);
                            if (globs_buf.len > 0) allocator.free(globs_buf);
                            continue;
                        };
                    };
                    cis[ci_idx] = .{ .module_inst = mi };

                    if (entries.len > 0) inst_mod.attachHostFuncEntries(mi, entries);
                    if (module_ptr.start_function != null) {
                        inst.pending_core_starts.append(allocator, mi) catch {
                            // OOM here is fatal — without the start running,
                            // the component cannot reach a usable state.
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            if (mems_buf.len > 0) allocator.free(mems_buf);
                            if (tbls_buf.len > 0) allocator.free(tbls_buf);
                            if (globs_buf.len > 0) allocator.free(globs_buf);
                            return error.OutOfMemory;
                        };
                    }
                    {
                        var nset: u32 = 0;
                        for (entries) |e| if (e != null) { nset += 1; };
                    }
                    if (imps_buf.len > 0) allocator.free(imps_buf);
                    if (is_cross.len > 0) allocator.free(is_cross);
                    if (mems_buf.len > 0) allocator.free(mems_buf);
                    if (tbls_buf.len > 0) allocator.free(tbls_buf);
                    if (globs_buf.len > 0) allocator.free(globs_buf);
                },
            }
        }
    } else if (component.core_modules.len > 0) {
        // Legacy path: one core instance per core module.
        const cis = allocator.alloc(ComponentInstance.CoreInstanceEntry, component.core_modules.len) catch return error.OutOfMemory;
        for (component.core_modules, 0..) |core_mod, i| {
            cis[i] = .{};
            const mod_alloc = inst.module_arena.allocator();
            const module = loader.load(core_mod.data, mod_alloc) catch continue;
            const module_ptr = mod_alloc.create(core_types.WasmModule) catch continue;
            module_ptr.* = module;
            const module_inst = inst_mod.instantiate(module_ptr, allocator) catch continue;
            cis[i].module_inst = module_inst;
        }
        inst.core_instances = cis;
    }

    // Build export map: walk top-level component exports, resolve each
    // `.func` export through the component-func index space to its backing
    // `canon.lift`, and register an entry in `exported_funcs` keyed by the
    // export name.
    //
    // Top-level *instance* exports (e.g. `wasi:cli/run@0.2.6`) are
    // handled by walking the locally-instantiated instance's inline-exports
    // and registering each member func under both the dotted name
    // (`<instance-name>/<member>`) and — for the canonical `wasi:cli/run`
    // shape — the bare member name so the existing `runComponent` adapter
    // can locate `"run"` without knowing the version suffix.
    for (component.exports) |exp| {
        const si = exp.sort_idx orelse continue;
        switch (si.sort) {
            .func => registerLiftedExport(inst, component, allocator, exp.name, si.idx),
            .instance => registerInstanceExport(inst, component, allocator, exp.name, si.idx),
            else => {},
        }
    }

    return inst;
}

fn registerLiftedExport(
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
    name: []const u8,
    func_idx: u32,
) void {
    const ref = indexspace.resolveCompFunc(component, func_idx) orelse return;
    const canon_idx = switch (ref) {
        .lifted => |i| i,
        else => return,
    };
    switch (component.canons[canon_idx]) {
        .lift => |lift| {
            const resolved = resolveLiftedCoreFunc(inst, component, lift.core_func_idx);
            inst.exported_funcs.put(allocator, name, .{
                .core_instance_idx = if (resolved) |r| r.core_instance_idx else 0,
                .core_func_idx = if (resolved) |r| r.local_func_idx else lift.core_func_idx,
                .func_type_idx = lift.type_idx,
                .opts = lift.opts,
            }) catch {};
        },
        else => {},
    }
}

fn registerInstanceExport(
    inst: *ComponentInstance,
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
    instance_name: []const u8,
    instance_idx: u32,
) void {
    const ref = indexspace.resolveCompInstance(component, instance_idx) orelse return;
    const local_idx = switch (ref) {
        .local => |i| i,
        // Re-exported imported instances and aliased instances aren't
        // backed by lifts inside this component — nothing to register.
        else => return,
    };
    if (local_idx >= component.instances.len) return;
    const expr = component.instances[local_idx];
    const expose_bare = isWasiCliRunName(instance_name);

    switch (expr) {
        .exports => |inline_exports| {
            for (inline_exports) |mem| {
                if (mem.sort_idx.sort != .func) continue;
                const dotted = std.fmt.allocPrint(inst.module_arena.allocator(), "{s}/{s}", .{ instance_name, mem.name }) catch continue;
                registerLiftedExport(inst, component, allocator, dotted, mem.sort_idx.idx);
                if (expose_bare and std.mem.eql(u8, mem.name, "run")) {
                    registerLiftedExport(inst, component, allocator, "run", mem.sort_idx.idx);
                }
            }
        },
        .instantiate => |inst_expr| {
            // The wit-bindgen "0.2.0-shim" pattern: the wasi:cli/run
            // export is an instance produced by instantiating a tiny
            // sub-component whose only purpose is to re-export an
            // imported func ("import-func-run") under the canonical
            // member name "run". We resolve such an instance member by:
            //   1. looking up the sub-component's named export,
            //   2. mapping its func sort_idx into the sub-component's
            //      func index space (it must be an imported func),
            //   3. matching that import's name against the parent
            //      `with` arg list, and
            //   4. resolving the parent's argument through the parent's
            //      indexspace. The parent func ref is the one we
            //      register under `<instance>/<member>`.
            if (inst_expr.component_idx >= component.components.len) return;
            const subcomp = component.components[inst_expr.component_idx];
            for (subcomp.exports) |sub_exp| {
                if (sub_exp.desc != .func) continue;
                const sub_si = sub_exp.sort_idx orelse continue;
                if (sub_si.sort != .func) continue;
                const sub_ref = indexspace.resolveCompFunc(subcomp, sub_si.idx) orelse continue;
                const sub_imp_idx: u32 = switch (sub_ref) {
                    .imported => |i| i,
                    else => continue,
                };
                if (sub_imp_idx >= subcomp.imports.len) continue;
                const import_name = subcomp.imports[sub_imp_idx].name;
                // Find matching `with` arg in the parent instantiate.
                const parent_func_idx: u32 = blk: {
                    for (inst_expr.args) |arg| {
                        if (arg.sort_idx.sort != .func) continue;
                        if (std.mem.eql(u8, arg.name, import_name)) {
                            break :blk arg.sort_idx.idx;
                        }
                    }
                    continue;
                };
                const dotted = std.fmt.allocPrint(inst.module_arena.allocator(), "{s}/{s}", .{ instance_name, sub_exp.name }) catch continue;
                registerLiftedExport(inst, component, allocator, dotted, parent_func_idx);
                if (expose_bare and std.mem.eql(u8, sub_exp.name, "run")) {
                    registerLiftedExport(inst, component, allocator, "run", parent_func_idx);
                }
            }
        },
    }
}

/// Match `wasi:cli/run` and `wasi:cli/run@<version>` instance export names.
fn isWasiCliRunName(name: []const u8) bool {
    const prefix = "wasi:cli/run";
    if (!std.mem.startsWith(u8, name, prefix)) return false;
    const rest = name[prefix.len..];
    return rest.len == 0 or rest[0] == '@';
}

// ── Index-space helpers ─────────────────────────────────────────────────────
//
// These resolvers implement the narrow subset needed for the Phase 2A.2b
// hand-authored fixture: no core aliases, no imported core funcs, every
// core func in the core-func-index-space comes from a `canon.lower`.
// A later slice will replace them with a section-order-aware resolver that
// handles arbitrary component layouts including `stdio-echo.wasm`.

/// Map a core-func-index-space index back to the canon.lower that
/// produced it. Returns null when the index does not point to a lower
/// (e.g. when it refers to an aliased core func).
fn resolveCoreFuncLower(component: *const ctypes.Component, core_func_idx: u32) ?u32 {
    return switch (indexspace.resolveCoreFunc(component, core_func_idx) orelse return null) {
        .lowered => |i| i,
        .resource_drop, .resource_new, .resource_rep, .aliased => null,
    };
}

/// Find which `ComponentInstance.core_instances[i]` hosts the core function
/// referenced by `core_func_idx`, by searching each inline-exports instance
/// for a `func`-sort export of that idx. Falls back to 0 on miss so the
/// legacy single-module layout continues to work.
fn resolveCoreFuncToInstance(component: *const ctypes.Component, core_func_idx: u32) ?u32 {
    // A canon.lift's core_func_idx typically references a function exposed
    // by the main `.instantiate` core instance (not an inline instance
    // wrapper). For now we pick the last `.instantiate` expression and
    // fall through to 0 otherwise.
    _ = core_func_idx;
    var i: usize = component.core_instances.len;
    while (i > 0) {
        i -= 1;
        switch (component.core_instances[i]) {
            .instantiate => return @intCast(i),
            else => {},
        }
    }
    return null;
}

/// Resolve a top-level core memory index to the underlying source
/// `MemoryInstance` it aliases. Only `alias core export` is currently
/// modeled.
fn resolveCoreMemoryToMI(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    core_mem_idx: u32,
) ?*core_types.MemoryInstance {
    const ref = indexspace.resolveCoreMemory(component, core_mem_idx) orelse return null;
    const ie = component.aliases[ref.aliased].instance_export;
    if (ie.instance_idx >= inst.core_instances.len) return null;
    const src_mi = inst.core_instances[ie.instance_idx].module_inst orelse return null;
    const exp = src_mi.module.findExport(ie.name, .memory) orelse return null;
    if (exp.index >= src_mi.memories.len) return null;
    return src_mi.memories[exp.index];
}

fn resolveCoreTableToMI(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    core_tbl_idx: u32,
) ?*core_types.TableInstance {
    const ref = indexspace.resolveCoreTable(component, core_tbl_idx) orelse return null;
    const ie = component.aliases[ref.aliased].instance_export;
    if (ie.instance_idx >= inst.core_instances.len) return null;
    const src_mi = inst.core_instances[ie.instance_idx].module_inst orelse return null;
    const exp = src_mi.module.findExport(ie.name, .table) orelse return null;
    if (exp.index >= src_mi.tables.len) return null;
    return src_mi.tables[exp.index];
}

fn resolveCoreGlobalToMI(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    core_glob_idx: u32,
) ?*core_types.GlobalInstance {
    const ref = indexspace.resolveCoreGlobal(component, core_glob_idx) orelse return null;
    const ie = component.aliases[ref.aliased].instance_export;
    if (ie.instance_idx >= inst.core_instances.len) return null;
    const src_mi = inst.core_instances[ie.instance_idx].module_inst orelse return null;
    const exp = src_mi.module.findExport(ie.name, .global) orelse return null;
    if (exp.index >= src_mi.globals.len) return null;
    return src_mi.globals[exp.index];
}

/// Walk `component.type_indexspace[idx]` (loader-populated) to find the
/// local entry in `component.types`. Falls back to direct indexing for
/// hand-authored fixtures that bypass the loader.
fn resolveTypeDef(component: *const ctypes.Component, type_idx: u32) ?ctypes.TypeDef {
    if (component.type_indexspace.len > 0) {
        if (type_idx >= component.type_indexspace.len) return null;
        const local = component.type_indexspace[type_idx] orelse return null;
        if (local >= component.types.len) return null;
        return component.types[local];
    }
    if (type_idx >= component.types.len) return null;
    return component.types[type_idx];
}

/// Resolve a component-func index to its function type. Handles imports,
/// aliases of imported-instance members, and `canon.lift` entries.
/// Inside an instance-type body, type indices are local. This helper
/// walks decls in order, resolving the Nth type-producing declarator
/// to its concrete `TypeDef`. Used by the canonical-ABI trampoline to
/// dereference param/result `.type_idx` references.
fn resolveInstanceTypeLocal(decls: []const ctypes.Decl, idx: u32) ?ctypes.TypeDef {
    var n: u32 = 0;
    for (decls) |d| switch (d) {
        .type => |td| {
            if (n == idx) return td;
            n += 1;
        },
        .alias => {
            // Aliases of types produce a binding but we don't track the
            // outer-resolved type here; return null to surface a fallback.
            if (n == idx) return null;
            n += 1;
        },
        .@"export" => |e| {
            if (e.desc == .type) {
                // Resource exports / type-eq exports — caller treats as
                // an opaque resource handle.
                if (n == idx) return null;
                n += 1;
            }
        },
        else => {},
    };
    return null;
}

/// Lower a possibly-local `ValType` referring into an instance-type body
/// to a `ValType` that the trampoline can lower without a TypeRegistry.
/// Superseded by `rewriteValTypeAbsolute` + `buildInstanceTypeExtension`
/// for the canon.lower trampoline path; kept for tests / external callers.
fn rewriteInstanceTypeValType(decls: []const ctypes.Decl, vt: ctypes.ValType) ctypes.ValType {
    _ = decls;
    return vt;
}

/// Rewrite an instance-type-local `ValType` so that any local type index
/// becomes an absolute index in the per-trampoline extended indexspace.
///
/// `base` is the offset where the extension starts in the absolute
/// indexspace (i.e. `component.type_indexspace.len`, or `types.len` when
/// the component has no indexspace).
///
/// Resource handle indices (`.own` / `.borrow`) carry resource identity
/// rather than structural type info and are left unchanged.
fn rewriteValTypeAbsolute(base: u32, vt: ctypes.ValType) ctypes.ValType {
    return switch (vt) {
        // Resource identity: do not rewrite.
        .own, .borrow => vt,
        // Structural compound refs: rebase index into extension.
        .record => |i| .{ .record = base + i },
        .variant => |i| .{ .variant = base + i },
        .list => |i| .{ .list = base + i },
        .tuple => |i| .{ .tuple = base + i },
        .flags => |i| .{ .flags = base + i },
        .enum_ => |i| .{ .enum_ = base + i },
        .option => |i| .{ .option = base + i },
        .result => |i| .{ .result = base + i },
        .type_idx => |i| .{ .type_idx = base + i },
        else => vt,
    };
}

/// Deep-copy a `TypeDef` from an instance-type body, rewriting all nested
/// `ValType` references through `rewriteValTypeAbsolute`. Allocations are
/// owned by `allocator` and freed when the trampoline ctx tears down.
fn rewriteTypeDefAbsolute(
    allocator: std.mem.Allocator,
    base: u32,
    td: ctypes.TypeDef,
) !ctypes.TypeDef {
    return switch (td) {
        .val => |v| .{ .val = rewriteValTypeAbsolute(base, v) },
        .list => |l| .{ .list = .{ .element = rewriteValTypeAbsolute(base, l.element) } },
        .option => |o| .{ .option = .{ .inner = rewriteValTypeAbsolute(base, o.inner) } },
        .result => |r| .{ .result = .{
            .ok = if (r.ok) |ok| rewriteValTypeAbsolute(base, ok) else null,
            .err = if (r.err) |er| rewriteValTypeAbsolute(base, er) else null,
        } },
        .record => |rec| blk: {
            const new_fields = try allocator.alloc(ctypes.Field, rec.fields.len);
            for (rec.fields, 0..) |f, i| {
                new_fields[i] = .{ .name = f.name, .type = rewriteValTypeAbsolute(base, f.type) };
            }
            break :blk .{ .record = .{ .fields = new_fields } };
        },
        .tuple => |tup| blk: {
            const new_fields = try allocator.alloc(ctypes.ValType, tup.fields.len);
            for (tup.fields, 0..) |f, i| new_fields[i] = rewriteValTypeAbsolute(base, f);
            break :blk .{ .tuple = .{ .fields = new_fields } };
        },
        .variant => |v| blk: {
            const new_cases = try allocator.alloc(ctypes.Case, v.cases.len);
            for (v.cases, 0..) |c, i| {
                new_cases[i] = .{
                    .name = c.name,
                    .type = if (c.type) |ct| rewriteValTypeAbsolute(base, ct) else null,
                    .refines = c.refines,
                };
            }
            break :blk .{ .variant = .{ .cases = new_cases } };
        },
        // Primitives + `.flags` / `.enum_` / `.resource` / func / component / instance
        // carry no nested ValType refs that need rewriting at this layer.
        else => td,
    };
}

const InstanceTypeExtension = struct {
    extension_types: []const ctypes.TypeDef,
    extension_indexspace: []const ?u32,

    pub fn empty() InstanceTypeExtension {
        return .{ .extension_types = &.{}, .extension_indexspace = &.{} };
    }

    pub fn deinit(self: InstanceTypeExtension, allocator: std.mem.Allocator, deep: bool) void {
        if (deep) {
            // Free the per-typedef allocations made by rewriteTypeDefAbsolute.
            for (self.extension_types) |td| switch (td) {
                .record => |rec| allocator.free(rec.fields),
                .tuple => |tup| allocator.free(tup.fields),
                .variant => |v| allocator.free(v.cases),
                else => {},
            };
        }
        if (self.extension_types.len > 0) allocator.free(self.extension_types);
        if (self.extension_indexspace.len > 0) allocator.free(self.extension_indexspace);
    }
};

/// Materialize the per-trampoline TypeRegistry extension covering an
/// instance-type body's local type space. Walks `decls` in declaration
/// order, mirroring `resolveInstanceTypeLocal`'s slot-counting rules:
/// `.type`, `.alias`, and `.@"export"`-with-type each contribute one
/// indexspace slot. Only `.type` slots materialize a structural typedef;
/// `.alias` and exported-type slots map to `null` (the trampoline path
/// for them was already null-fallback under the prior local-only walker).
///
/// The caller is responsible for `deinit`'ing the returned extension.
fn buildInstanceTypeExtension(
    allocator: std.mem.Allocator,
    decls: []const ctypes.Decl,
    base: u32,
) !InstanceTypeExtension {
    // First pass: count slots and type entries.
    var slot_count: u32 = 0;
    var type_count: u32 = 0;
    for (decls) |d| switch (d) {
        .type => {
            slot_count += 1;
            type_count += 1;
        },
        .alias => slot_count += 1,
        .@"export" => |e| if (e.desc == .type) {
            slot_count += 1;
        },
        else => {},
    };

    if (slot_count == 0) return InstanceTypeExtension.empty();

    const types_buf = try allocator.alloc(ctypes.TypeDef, type_count);
    errdefer allocator.free(types_buf);
    const idxspace_buf = try allocator.alloc(?u32, slot_count);
    errdefer allocator.free(idxspace_buf);

    var slot_i: u32 = 0;
    var type_i: u32 = 0;
    var rewrite_failed: bool = false;
    for (decls) |d| switch (d) {
        .type => |td| {
            const rewritten = rewriteTypeDefAbsolute(allocator, base, td) catch {
                rewrite_failed = true;
                break;
            };
            types_buf[type_i] = rewritten;
            idxspace_buf[slot_i] = type_i;
            type_i += 1;
            slot_i += 1;
        },
        .alias => {
            idxspace_buf[slot_i] = null;
            slot_i += 1;
        },
        .@"export" => |e| if (e.desc == .type) {
            idxspace_buf[slot_i] = null;
            slot_i += 1;
        },
        else => {},
    };

    if (rewrite_failed) {
        // Free any nested allocations made before the failure.
        var i: u32 = 0;
        while (i < type_i) : (i += 1) switch (types_buf[i]) {
            .record => |rec| allocator.free(rec.fields),
            .tuple => |tup| allocator.free(tup.fields),
            .variant => |v| allocator.free(v.cases),
            else => {},
        };
        allocator.free(types_buf);
        allocator.free(idxspace_buf);
        return error.OutOfMemory;
    }

    return .{
        .extension_types = types_buf,
        .extension_indexspace = idxspace_buf,
    };
}

/// Same as `rewriteInstanceTypeValType` but rewrites the params and
/// results of a `FuncType` and allocates a fresh slice via `allocator`.
/// (Superseded by `buildInstanceTypeExtension` + `rewriteValTypeAbsolute`.
/// Retained for any callers outside the canon.lower trampoline path.)
fn rewriteInstanceFuncType(
    allocator: std.mem.Allocator,
    decls: []const ctypes.Decl,
    ft: ctypes.FuncType,
) !ctypes.FuncType {
    _ = allocator;
    _ = decls;
    return ft;
}

const ResolvedFuncType = struct {
    ft: ctypes.FuncType,
    /// When the FuncType came from an instance-type body, its
    /// param/result `.type_idx` references are local to that body's
    /// type space. Callers must rewrite via `rewriteInstanceTypeValType`
    /// using this `decls` list before consuming the types.
    decls: ?[]const ctypes.Decl = null,
};

fn resolveCompFuncType(component: *const ctypes.Component, func_idx: u32) ?ResolvedFuncType {
    const ref = indexspace.resolveCompFunc(component, func_idx) orelse {
        return null;
    };
    switch (ref) {
        .imported => |imp_idx| {
            const imp = component.imports[imp_idx];
            const tidx = switch (imp.desc) {
                .func => |t| t,
                else => return null,
            };
            const td = resolveTypeDef(component, tidx) orelse return null;
            return switch (td) {
                .func => |ft| .{ .ft = ft },
                else => null,
            };
        },
        .aliased => |alias_idx| {
            const ie = component.aliases[alias_idx].instance_export;
            const inst_ref = indexspace.resolveCompInstance(component, ie.instance_idx) orelse {
                return null;
            };
            const inst_type_idx: u32 = switch (inst_ref) {
                .imported => |i| switch (component.imports[i].desc) {
                    .instance => |t| t,
                    else => return null,
                },
                else => {
                    return null;
                },
            };
            const inst_td = resolveTypeDef(component, inst_type_idx) orelse {
                return null;
            };
            const decls = switch (inst_td) {
                .instance => |it| it.decls,
                else => return null,
            };
            for (decls) |d| switch (d) {
                .@"export" => |e| {
                    if (!std.mem.eql(u8, e.name, ie.name)) continue;
                    const tidx = switch (e.desc) {
                        .func => |t| t,
                        else => return null,
                    };
                    var n: u32 = 0;
                    var found: ?ctypes.FuncType = null;
                    for (decls) |d2| {
                        switch (d2) {
                            .type => |td2| {
                                if (n == tidx) {
                                    found = switch (td2) {
                                        .func => |ft| ft,
                                        else => null,
                                    };
                                }
                                n += 1;
                            },
                            .alias => n += 1,
                            .@"export" => |e2| {
                                if (e2.desc == .type) n += 1;
                            },
                            else => {},
                        }
                        if (found != null) break;
                    }
                    if (found) |ft| return .{ .ft = ft, .decls = decls };
                    return null;
                },
                else => {},
            };
            return null;
        },
        .lifted => |canon_idx| {
            const lift = switch (component.canons[canon_idx]) {
                .lift => |l| l,
                else => return null,
            };
            const td = resolveTypeDef(component, lift.type_idx) orelse return null;
            return switch (td) {
                .func => |ft| .{ .ft = ft },
                else => null,
            };
        },
    }
}

/// Resolve a `canon.lift.core_func_idx` (component core-func-index-space) to
/// the (core_instance_idx, local_func_idx) pair where the function actually
/// lives, so the executor can call it via `interp.executeFunction` with the
/// module-local index.
///
/// Phase 2A.2c layout assumption: the core-func-index-space is built up by
/// (in this order) all `canon.lower` entries, then all `Alias.instance_export`
/// entries with `sort = .core(.func)`. A full section-order-aware resolver
/// will replace this once the loader emits ordered per-index-space streams.
fn resolveLiftedCoreFunc(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    core_func_idx: u32,
) ?struct { core_instance_idx: u32, local_func_idx: u32 } {
    const ref = indexspace.resolveCoreFunc(component, core_func_idx) orelse return null;
    switch (ref) {
        // Canon entries (lowers and resource.{new,drop,rep}) all produce
        // imports/host-bound core funcs — not exported callables — so a
        // canon.lift pointing at one is malformed.
        .lowered, .resource_drop, .resource_new, .resource_rep => return null,
        .aliased => |alias_idx| {
            const a = component.aliases[alias_idx];
            const ie = a.instance_export;
            if (ie.instance_idx >= inst.core_instances.len) return null;
            const target = inst.core_instances[ie.instance_idx];
            const mi = target.module_inst orelse return null;
            const local = mi.getExportFunc(ie.name) orelse return null;
            return .{
                .core_instance_idx = ie.instance_idx,
                .local_func_idx = local,
            };
        },
    }
}

/// Resolve a component-level func index to a bound `HostFunc`.
///
/// The component-level func index space is contributed to by, in section
/// order: `import (kind=func)` decls, then component-level aliases of
/// `Sort.func` exports of imported component instances, then canon.lifts,
/// and so on. Phase 2B narrow assumption: the prefix we care about is
/// `imports (kind=func)` followed by `aliases (Sort.func, instance_export)`
/// pointing into imported component instances. canon.lifts come later in
/// the index space and are not host-bound.
fn resolveComponentFuncToHostFunc(
    inst: *const ComponentInstance,
    component: *const ctypes.Component,
    func_idx: u32,
) ?HostFunc {
    const ref = indexspace.resolveCompFunc(component, func_idx) orelse return null;
    switch (ref) {
        .imported => |imp_idx| {
            const imp = component.imports[imp_idx];
            const binding = inst.imports.get(imp.name) orelse return null;
            return switch (binding) {
                .host_func => |hf| hf,
                else => null,
            };
        },
        .aliased => |alias_idx| {
            const ie = component.aliases[alias_idx].instance_export;
            const inst_ref = indexspace.resolveCompInstance(component, ie.instance_idx) orelse return null;
            // Only aliases pointing at imported component instances are
            // host-bound. Aliases of locally-instantiated instances are
            // resolved to their underlying canon.lift / canon.lower elsewhere.
            const imp_decl = switch (inst_ref) {
                .imported => |i| component.imports[i],
                else => return null,
            };
            const binding = inst.imports.get(imp_decl.name) orelse return null;
            const host_inst = switch (binding) {
                .host_instance => |hi| hi,
                else => return null,
            };
            const member = host_inst.members.get(ie.name) orelse return null;
            return switch (member) {
                .func => |hf| hf,
                .resource_type => null,
            };
        },
        // canon.lifts are not host-bound — they are component-defined
        // funcs implemented by core code.
        .lifted => return null,
    }
}

/// Resolve a component-level instance index to its `ImportDecl` if it
/// refers to an imported component instance. Returns null for locally
/// instantiated or aliased instances (callers wanting the broader form
/// should use `indexspace.resolveCompInstance` directly).
fn resolveImportedInstance(
    component: *const ctypes.Component,
    instance_idx: u32,
) ?ctypes.ImportDecl {
    const ref = indexspace.resolveCompInstance(component, instance_idx) orelse return null;
    return switch (ref) {
        .imported => |i| component.imports[i],
        else => null,
    };
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "ResourceTable: new and rep" {
    const allocator = std.testing.allocator;
    var rt = ResourceTable{};
    defer rt.deinit(allocator);

    const h0 = try rt.new(42, true, allocator);
    const h1 = try rt.new(99, true, allocator);

    try std.testing.expectEqual(@as(u32, 0), h0);
    try std.testing.expectEqual(@as(u32, 1), h1);
    try std.testing.expectEqual(@as(?u32, 42), rt.rep(h0));
    try std.testing.expectEqual(@as(?u32, 99), rt.rep(h1));
}

test "ResourceTable: drop and reuse" {
    const allocator = std.testing.allocator;
    var rt = ResourceTable{};
    defer rt.deinit(allocator);

    const h0 = try rt.new(10, true, allocator);
    const dropped_rep = rt.drop(h0, allocator);
    try std.testing.expectEqual(@as(?u32, 10), dropped_rep);
    try std.testing.expectEqual(@as(?u32, null), rt.rep(h0));

    // Reuse the slot
    const h2 = try rt.new(20, true, allocator);
    try std.testing.expectEqual(h0, h2); // reused slot
    try std.testing.expectEqual(@as(?u32, 20), rt.rep(h2));
}

test "ResourceTable: borrow prevents drop" {
    const allocator = std.testing.allocator;
    var rt = ResourceTable{};
    defer rt.deinit(allocator);

    const h = try rt.new(55, true, allocator);
    try std.testing.expect(rt.borrow(h));
    try std.testing.expectEqual(@as(?u32, null), rt.drop(h, allocator)); // can't drop

    rt.returnBorrow(h);
    try std.testing.expectEqual(@as(?u32, 55), rt.drop(h, allocator)); // now it works
}

test "ResourceTable: double drop returns null" {
    const allocator = std.testing.allocator;
    var rt = ResourceTable{};
    defer rt.deinit(allocator);

    const h = try rt.new(77, true, allocator);
    _ = rt.drop(h, allocator);
    try std.testing.expectEqual(@as(?u32, null), rt.drop(h, allocator));
}

test "ImportBinding: host func creation" {
    const binding = ImportBinding{ .host_func = .{ .context = null } };
    try std.testing.expect(binding == .host_func);
}

test "ComponentInstance: linkImports resolves known imports" {
    const allocator = std.testing.allocator;

    // Create a minimal component with one import
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "my-import", .desc = .{ .func = 0 } },
    };
    const component = ctypes.Component{
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

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    // Provide a binding for the import
    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try providers.put(allocator, "my-import", .{ .host_func = .{ .context = null } });

    try inst.linkImports(providers);

    // Verify the import was resolved
    const resolved = inst.getImport("my-import");
    try std.testing.expect(resolved != null);
    try std.testing.expect(resolved.? == .host_func);
}

test "ComponentInstance: getImport returns null for unknown" {
    const allocator = std.testing.allocator;

    const component = ctypes.Component{
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

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    try std.testing.expectEqual(@as(?ImportBinding, null), inst.getImport("nonexistent"));
}

test "ComponentInstance: executeStart is idempotent" {
    const allocator = std.testing.allocator;

    // Component with no start function — executeStart should be a no-op
    const component = ctypes.Component{
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

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    try inst.executeStart(); // first call
    try std.testing.expect(inst.started);
    try inst.executeStart(); // second call — should be idempotent
    try std.testing.expect(inst.started);
}

test "linkImports: missing runtime import returns MissingImport" {
    const allocator = std.testing.allocator;

    const imports = [_]ctypes.ImportDecl{
        .{ .name = "my-func", .desc = .{ .func = 0 } },
    };
    const component = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},
        .imports = &imports,      .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try std.testing.expectError(error.MissingImport, inst.linkImports(providers));
}

test "linkImports: kind mismatch returns ImportKindMismatch" {
    const allocator = std.testing.allocator;

    // Instance-typed import must be satisfied with host_instance, not host_func.
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "wasi:io/streams", .desc = .{ .instance = 0 } },
    };
    const component = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},
        .imports = &imports,      .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try providers.put(allocator, "wasi:io/streams", .{ .host_func = .{} });
    try std.testing.expectError(error.ImportKindMismatch, inst.linkImports(providers));
}

test "linkImports: host_instance binding satisfies instance import" {
    const allocator = std.testing.allocator;

    const imports = [_]ctypes.ImportDecl{
        .{ .name = "wasi:io/streams", .desc = .{ .instance = 0 } },
    };
    const component = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},
        .imports = &imports,      .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    var host: HostInstance = .{};
    defer host.deinit(allocator);

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try providers.put(allocator, "wasi:io/streams", .{ .host_instance = &host });

    try inst.linkImports(providers);
    const resolved = inst.getImport("wasi:io/streams") orelse return error.TestUnexpectedResult;
    try std.testing.expect(resolved == .host_instance);
    try std.testing.expectEqual(&host, resolved.host_instance);
}

test "linkImports: type import needs no binding" {
    const allocator = std.testing.allocator;

    const imports = [_]ctypes.ImportDecl{
        .{ .name = "T", .desc = .{ .type = .sub_resource } },
    };
    const component = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},
        .imports = &imports,      .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .{};
    defer providers.deinit(allocator);
    try inst.linkImports(providers); // no error despite empty providers
}

test "ComponentInstance: resource tables are lazy and keyed by typeidx" {
    const allocator = std.testing.allocator;

    const component = ctypes.Component{
        .core_modules = &.{},     .core_instances = &.{}, .core_types = &.{},
        .components = &.{},       .instances = &.{},      .aliases = &.{},
        .types = &.{},            .canons = &.{},
        .imports = &.{},          .exports = &.{},
    };

    const inst = try instantiate(&component, allocator);
    defer inst.deinit();

    // No tables allocated up front.
    try std.testing.expectEqual(@as(u32, 0), inst.resource_tables.count());

    // Sparse resource type indices are fine.
    const rt_a = try inst.getOrCreateResourceTable(5);
    const rt_b = try inst.getOrCreateResourceTable(42);
    try std.testing.expect(rt_a != rt_b);
    try std.testing.expectEqual(@as(u32, 2), inst.resource_tables.count());

    // Repeated access returns the same table.
    const rt_a2 = try inst.getOrCreateResourceTable(5);
    try std.testing.expectEqual(rt_a, rt_a2);
}

test "instantiate: canon.lower wires host func into core import (2A.2b)" {

    // Minimal core module:
    //   (type (func (param i32 i32) (result i32)))
    //   (type (func (result i32)))
    //   (import "host" "sub" (func $sub (type 0)))
    //   (func $run (type 1) i32.const 7 i32.const 2 call $sub)
    //   (export "run" (func $run))
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section
        0x01, 0x0b, 0x02,
        0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
        0x60, 0x00, 0x01, 0x7f,
        // import section: host.sub (func type 0)
        0x02, 0x0c, 0x01,
        0x04, 'h', 'o', 's', 't',
        0x03, 's', 'u', 'b',
        0x00, 0x00,
        // function section: 1 local fn, type 1
        0x03, 0x02, 0x01, 0x01,
        // export section: "run" -> func 1
        0x07, 0x07, 0x01,
        0x03, 'r', 'u', 'n',
        0x00, 0x01,
        // code section
        0x0a, 0x0a, 0x01,
        0x08, 0x00,
        0x41, 0x07, // i32.const 7
        0x41, 0x02, // i32.const 2
        0x10, 0x00, // call 0 (imported sub)
        0x0b, // end
    };

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};

    // Component func type 0: (s32, s32) -> s32
    const params = [_]ctypes.NamedValType{
        .{ .name = "a", .type = .s32 },
        .{ .name = "b", .type = .s32 },
    };
    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };
    const imports_decl = [_]ctypes.ImportDecl{
        .{ .name = "host-sub", .desc = .{ .func = 0 } },
    };
    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &.{} } },
    };
    const inline_exports = [_]ctypes.CoreInlineExport{
        .{ .name = "sub", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const inst_args = [_]ctypes.CoreInstantiateArg{
        .{ .name = "host", .instance_idx = 0 },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .exports = &inline_exports },
        .{ .instantiate = .{ .module_idx = 0, .args = &inst_args } },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &type_defs,
        .canons = &canons,
        .imports = &imports_decl,
        .exports = &.{},
    };

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    // Register a host sub: returns a - b. Non-commutative to catch argument
    // order reversal in the trampoline.
    const Host = struct {
        fn sub(
            _: ?*anyopaque,
            _: *ComponentInstance,
            in: []const InterfaceValue,
            out: []InterfaceValue,
            _: std.mem.Allocator,
        ) anyerror!void {
            out[0] = .{ .s32 = in[0].s32 - in[1].s32 };
        }
    };
    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try providers.put(std.testing.allocator, "host-sub", .{
        .host_func = .{ .call = &Host.sub },
    });
    try inst.linkImports(providers);

    // After linkImports, trampolines should be wired to the host fn.
    try std.testing.expect(inst.core_instances.len == 2);
    const mi = inst.core_instances[1].module_inst orelse return error.TestFailed;
    try std.testing.expect(mi.host_func_entries.len >= 1);
    try std.testing.expect(inst.trampoline_ctxs.items.len == 1);
    try std.testing.expect(inst.trampoline_ctxs.items[0].host_func.call != null);

    // Invoke the exported "run" core function.
    const run_idx = mi.getExportFunc("run") orelse return error.TestFailed;
    const env = try @import("../runtime/common/exec_env.zig").ExecEnv.create(mi, 512, std.testing.allocator);
    defer env.destroy();
    try @import("../runtime/interpreter/interp.zig").executeFunction(env, run_idx);
    try std.testing.expectEqual(@as(i32, 5), try env.popI32());
}

test "callComponentFunc: invokes lifted export through alias (2A.2c)" {
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Same shape as the 2A.2b fixture, but `run` takes the args instead of
    // hard-coding constants. A canon.lift then exposes it as a component
    // export, and callComponentFunc invokes it through the lift.
    //
    //   (module
    //     (type (func (param i32 i32) (result i32)))
    //     (import "host" "sub" (func (type 0)))
    //     (func (type 0)
    //       local.get 0 local.get 1 call 0)
    //     (export "run" (func 1)))
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section (1 type)
        0x01, 0x07, 0x01,
        0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
        // import section (1 import)
        0x02, 0x0c, 0x01,
        0x04, 'h', 'o', 's', 't',
        0x03, 's', 'u', 'b',
        0x00, 0x00,
        // function section (1 local fn)
        0x03, 0x02, 0x01, 0x00,
        // export section: "run" -> func 1
        0x07, 0x07, 0x01,
        0x03, 'r', 'u', 'n',
        0x00, 0x01,
        // code section (1 body, 8 bytes)
        0x0a, 0x0a, 0x01,
        0x08, 0x00,
        0x20, 0x00, 0x20, 0x01,
        0x10, 0x00,
        0x0b,
    };

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};

    const params = [_]ctypes.NamedValType{
        .{ .name = "a", .type = .s32 },
        .{ .name = "b", .type = .s32 },
    };
    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };
    const imports_decl = [_]ctypes.ImportDecl{
        .{ .name = "host-sub", .desc = .{ .func = 0 } },
    };
    // Canon order: lower first (core-func 0), then lift (component func 1).
    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &.{} } },
        .{ .lift = .{ .core_func_idx = 1, .type_idx = 0, .opts = &.{} } },
    };
    const inline_exports = [_]ctypes.CoreInlineExport{
        .{ .name = "sub", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const inst_args = [_]ctypes.CoreInstantiateArg{
        .{ .name = "host", .instance_idx = 0 },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .exports = &inline_exports },
        .{ .instantiate = .{ .module_idx = 0, .args = &inst_args } },
    };
    const aliases_decl = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .{ .core = .func },
            .instance_idx = 1,
            .name = "run",
        } },
    };
    const exports_decl = [_]ctypes.ExportDecl{
        .{ .name = "run", .desc = .{ .func = 1 }, .sort_idx = .{ .sort = .func, .idx = 1 } },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases_decl,
        .types = &type_defs,
        .canons = &canons,
        .imports = &imports_decl,
        .exports = &exports_decl,
    };

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const Host = struct {
        fn sub(
            _: ?*anyopaque,
            _: *ComponentInstance,
            in: []const abi_mod.InterfaceValue,
            out: []abi_mod.InterfaceValue,
            _: std.mem.Allocator,
        ) anyerror!void {
            out[0] = .{ .s32 = in[0].s32 - in[1].s32 };
        }
    };
    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try providers.put(std.testing.allocator, "host-sub", .{
        .host_func = .{ .call = &Host.sub },
    });
    try inst.linkImports(providers);

    // Confirm export resolution found the right (instance, local) pair.
    const exported = inst.getExport("run") orelse return error.TestFailed;
    try std.testing.expectEqual(@as(u32, 1), exported.core_instance_idx);
    try std.testing.expectEqual(@as(u32, 1), exported.core_func_idx);

    var args = [_]abi_mod.InterfaceValue{
        .{ .s32 = 7 },
        .{ .s32 = 2 },
    };
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "run", &args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(i32, 5), results[0].s32);
}

test "instantiate: H1 micro-fixture — multi-core-module composition with cross-instance call (#156 H1)" {
    const loader_mod = @import("loader.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Two-module composition (see fixtures/h1-compose.wat):
    //   $A exports func "f" returning 7.
    //   $B imports "a"."f", exports "g" returning f()+1 (==8).
    //   The component aliases $b's "g" and lifts it as export "g" : u32.
    // Exercises:
    //   * `(core instance (instantiate $B (with "a" (instance $a))))` where
    //     the source instance is a real module_inst, not an inline-exports
    //     bundle — currently the resolver only consults inline_exports.
    //   * `(alias core export $b "g")` driving the lifted export through
    //     the second core instance.
    const data = @embedFile("fixtures/h1-compose.wasm");
    // The loader has no Component.deinit yet (see #142 Phase 1B); allocate
    // its small slices into an arena so the test doesn't leak.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    // No host imports needed — both core modules are self-contained.
    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try inst.linkImports(providers);

    var args: [0]abi_mod.InterfaceValue = .{};
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "g", &args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 8), results[0].u32);
}

test "instantiate: H1.2 micro-fixture — cross-instance memory wiring (#156 H1.2)" {
    const loader_mod = @import("loader.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Two-module composition (see fixtures/h1-mem.wat):
    //   $A exports memory "mem".
    //   $B imports "a"."mem", stores 42 at offset 0 then loads it.
    //   Component lifts $b's "g" as export "g" : u32 → must be 42.
    // Exercises `(with NAME (instance N))` matching against a real source
    // instance's *memory* export, populating ImportContext.memories so the
    // shared MemoryInstance is seen by both core modules.
    const data = @embedFile("fixtures/h1-mem.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try inst.linkImports(providers);

    // Both core instances must share the same MemoryInstance.
    try std.testing.expect(inst.core_instances.len == 2);
    const mi_a = inst.core_instances[0].module_inst orelse return error.TestFailed;
    const mi_b = inst.core_instances[1].module_inst orelse return error.TestFailed;
    try std.testing.expect(mi_a.memories.len >= 1);
    try std.testing.expect(mi_b.memories.len >= 1);
    try std.testing.expectEqual(mi_a.memories[0], mi_b.memories[0]);

    var args: [0]abi_mod.InterfaceValue = .{};
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "g", &args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 42), results[0].u32);
}

test "instantiate: H1.3 micro-fixture — alias core export of memory through inline-exports (#156 H1.3)" {
    const loader_mod = @import("loader.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Three-step composition (see fixtures/h1-alias.wat):
    //   $A exports memory "mem" and func "init" (writes 7 at addr 99).
    //   `(alias core export $a "mem" (core memory))` → top-level core mem 0.
    //   `(core instance $args (export "mem" (memory 0)))` — inline-exports
    //   bundle that re-exports the aliased memory via the SortIdx path.
    //   $B imports "src"."mem" and exports "read" (loads addr 99).
    //   $b instantiated `(with "src" (instance $args))`.
    // Exercises:
    //   * `aliasContributesTo` for `.core_memory`.
    //   * `resolveCoreMemory` ordering.
    //   * Memory import resolution against an inline-exports source whose
    //     member's SortIdx points at a top-level core memory contributed by
    //     an alias-core-export — the path stdio-echo's `$fixup` takes for
    //     the lifted `$main.memory`.
    const data = @embedFile("fixtures/h1-alias.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try inst.linkImports(providers);

    // Both real module instances must see the same MemoryInstance even
    // though the wiring goes through an inline-exports bundle.
    try std.testing.expect(inst.core_instances.len >= 3);
    const mi_a = inst.core_instances[0].module_inst orelse return error.TestFailed;
    // core_instances[1] is the inline-exports `$args` bundle (no module_inst).
    const mi_b = inst.core_instances[2].module_inst orelse return error.TestFailed;
    try std.testing.expect(mi_a.memories.len >= 1);
    try std.testing.expect(mi_b.memories.len >= 1);
    try std.testing.expectEqual(mi_a.memories[0], mi_b.memories[0]);

    // Run init via $A so memory has 7 at offset 99, then read via $B.
    var no_args: [0]abi_mod.InterfaceValue = .{};
    var no_results: [0]abi_mod.InterfaceValue = .{};
    try executor.callComponentFunc(inst, "init", &no_args, &no_results, std.testing.allocator);

    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "read", &no_args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 7), results[0].u32);
}

test "instantiate: H2 micro-fixture — table.set + call_indirect via canon.lower trampoline (#156 H2)" {
    const loader_mod = @import("loader.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Three-module composition (see fixtures/h2-trampoline.wat):
    //   $A exports table "t" (1 funcref) and func "call0" which call_indirect's
    //     element 0 with the i32 arg passed in.
    //   `(alias core export $a "t" (core table))` → top-level core table 0.
    //   `(canon lower (func $dbl))` produces a core func bound via trampoline
    //     to host_func host:double (HostFunc.call doubles its u32 arg).
    //   `(core instance $args (export "t" (table 0)) (export "f" (func ...)))`
    //   $B imports the table and the lowered func; its `start` runs
    //     `i32.const 0  ref.func $f  table.set 0`, i.e. installs the
    //     trampoline-backed funcref into the imported table at offset 0.
    // After instantiation the lifted `call0(x)` exercises:
    //   * cross-module call_indirect against an imported, post-instantiation
    //     populated table;
    //   * funcref dispatch into a host_func_entries[]-backed canon.lower
    //     trampoline;
    //   * host return-value lift back into the calling core module.
    const data = @embedFile("fixtures/h2-trampoline.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const Host = struct {
        fn double(
            _: ?*anyopaque,
            _: *ComponentInstance,
            args: []const abi_mod.InterfaceValue,
            results: []abi_mod.InterfaceValue,
            _: std.mem.Allocator,
        ) anyerror!void {
            results[0] = .{ .u32 = args[0].u32 *% 2 };
        }
    };

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try providers.put(std.testing.allocator, "my:host/double", .{ .host_func = .{ .call = &Host.double } });
    try inst.linkImports(providers);

    var args: [1]abi_mod.InterfaceValue = .{.{ .u32 = 21 }};
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "call0", &args, &results, std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 42), results[0].u32);
}

test "instantiate: registers nested wasi:cli/run instance member as 'run' (#151)" {
    // Hand-authored component:
    //   - 1 core module exporting "run" (no imports, no host calls)
    //   - 1 core instance instantiating it
    //   - 1 alias of the core "run" → core-func-idx 0
    //   - 1 canon.lift over that core func → comp-func-idx 0
    //   - 1 local component-instance bundling { "run": comp-func 0 }
    //     (instance-idx 0)
    //   - 1 top-level export "wasi:cli/run@0.2.6" → instance 0
    //
    // After instantiate(), inst.getExport("run") and
    // inst.getExport("wasi:cli/run@0.2.6/run") must both resolve to
    // the lifted core export.
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: () -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // function section: 1 fn of type 0
        0x03, 0x02, 0x01, 0x00,
        // export section: "run" -> func 0
        0x07, 0x07, 0x01, 0x03, 'r', 'u', 'n', 0x00, 0x00,
        // code section: empty body
        0x0a, 0x04, 0x01, 0x02, 0x00, 0x0b,
    };

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};
    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &.{}, .results = .none } },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .instantiate = .{ .module_idx = 0, .args = &.{} } },
    };
    const aliases_decl = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .{ .core = .func },
            .instance_idx = 0,
            .name = "run",
        } },
    };
    const canons = [_]ctypes.Canon{
        .{ .lift = .{ .core_func_idx = 0, .type_idx = 0, .opts = &.{} } },
    };
    const inline_exp = [_]ctypes.InlineExport{
        .{ .name = "run", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const instances = [_]ctypes.InstanceExpr{
        .{ .exports = &inline_exp },
    };
    const exports_decl = [_]ctypes.ExportDecl{
        .{
            .name = "wasi:cli/run@0.2.6",
            .desc = .{ .instance = 0 },
            .sort_idx = .{ .sort = .instance, .idx = 0 },
        },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &instances,
        .aliases = &aliases_decl,
        .types = &type_defs,
        .canons = &canons,
        .imports = &.{},
        .exports = &exports_decl,
    };

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    const bare = inst.getExport("run") orelse return error.TestFailed;
    try std.testing.expectEqual(@as(u32, 0), bare.core_instance_idx);
    try std.testing.expectEqual(@as(u32, 0), bare.core_func_idx);

    const dotted = inst.getExport("wasi:cli/run@0.2.6/run") orelse return error.TestFailed;
    try std.testing.expectEqual(@as(u32, 0), dotted.core_instance_idx);
    try std.testing.expectEqual(@as(u32, 0), dotted.core_func_idx);
}

test "instantiate: core (start ...) calling canon-lowered host import sees bound host_func (#308)" {
    const loader_mod = @import("loader.zig");
    const abi_mod = @import("canonical_abi.zig");

    // Regression for #308. The fixture has a core module whose
    // `(start ...)` directive calls a canon-lowered host import.
    //   (component
    //     (import "host:nop/run" (func $run (param "x" u32)))
    //     (core module $A
    //       (import "host" "f" (func $f (param i32)))
    //       (start $start)
    //       (func $start  i32.const 42  call $f))
    //     (core func $f_low (canon lower (func $run)))
    //     (core instance $args (export "f" (func $f_low)))
    //     (core instance $a (instantiate $A (with "host" (instance $args)))))
    //
    // Before the deferred-start fix the (start) ran during instantiate(),
    // before linkImports() bound the trampoline `host_func` — and trapped
    // with `HostFuncNotBound`. After the fix, instantiate() defers the
    // start; linkImports() binds the trampoline and then drains the
    // pending starts, so the host fn is invoked with 42 exactly once.
    const data = @embedFile("fixtures/h3-start-host-call.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    // ─ Before linkImports: the deferred start has NOT run yet.
    try std.testing.expect(inst.pending_core_starts.items.len == 1);

    const Host = struct {
        var calls: u32 = 0;
        var last_arg: u32 = 0;
        fn run(
            _: ?*anyopaque,
            _: *ComponentInstance,
            args: []const abi_mod.InterfaceValue,
            _: []abi_mod.InterfaceValue,
            _: std.mem.Allocator,
        ) anyerror!void {
            calls += 1;
            last_arg = args[0].u32;
        }
    };
    Host.calls = 0;
    Host.last_arg = 0;

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(std.testing.allocator);
    try providers.put(std.testing.allocator, "host:nop/run", .{ .host_func = .{ .call = &Host.run } });

    // linkImports binds the trampoline AND drains pending starts.
    try inst.linkImports(providers);

    try std.testing.expectEqual(@as(u32, 1), Host.calls);
    try std.testing.expectEqual(@as(u32, 42), Host.last_arg);
    try std.testing.expectEqual(@as(usize, 0), inst.pending_core_starts.items.len);
}

test "instantiate: core (start ...) calling host import without linkImports leaves start un-run (#308)" {
    const loader_mod = @import("loader.zig");

    // Same fixture — but the caller never invokes linkImports, so the
    // pending start remains queued. We just verify that instantiate()
    // alone does NOT trap and that the deferred start is observable.
    // This is the contract `WasiCliAdapter.runLoadedComponent` relies on
    // (instantiate must succeed even when the core start would call a
    // not-yet-bound host import).
    const data = @embedFile("fixtures/h3-start-host-call.wasm");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const component_owned = try loader_mod.load(data, arena.allocator());
    var component = component_owned;

    const inst = try instantiate(&component, std.testing.allocator);
    defer inst.deinit();

    try std.testing.expectEqual(@as(usize, 1), inst.pending_core_starts.items.len);
}
