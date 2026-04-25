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
        const mi = self.firstModuleInst() orelse return null;
        const mem = mi.getMemory(0) orelse return null;
        const end = @as(usize, ptr) + @as(usize, len);
        if (end > mem.data.len) return null;
        return mem.data[ptr..end];
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
        const mi = self.firstModuleInst() orelse return null;
        const realloc_local = mi.getExportFunc("cabi_realloc") orelse return null;
        const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
        const executor = @import("executor.zig");
        const env = ExecEnv.create(mi, 4096, self.allocator) catch return null;
        defer env.destroy();
        const ptr = executor.callRealloc(env, realloc_local, 0, 0, 1, @intCast(bytes.len)) catch return null;
        const mem = mi.getMemory(0) orelse return null;
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
        for (self.trampoline_ctxs.items) |ctx| {
            if (resolveComponentFuncToHostFunc(self, self.component, ctx.component_func_idx)) |hf| {
                ctx.host_func = hf;
            }
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

                    // Resolve every function import against `with` args BEFORE
                    // instantiation. Two backends:
                    //   * Cross-instance: the source `with`-arg is a real core
                    //     module instance; wire `import_functions[i]` so the
                    //     interpreter dispatches into the source's body
                    //     (interp.executeFunction switches `env.module_inst`).
                    //   * Canon.lower: the source is an inline-exports bundle
                    //     pointing at a `canon lower` core func; install a
                    //     `componentTrampoline` on `host_func_entries[i]`.
                    // Slots not satisfied by either backend remain unresolved
                    // (interp falls through to a no-op stub).
                    const import_func_count = module_ptr.import_function_count;
                    var entries: []?core_types.HostFnEntry = &.{};
                    var imps_buf: []core_types.ImportedFunction = &.{};
                    var is_cross: []bool = &.{};
                    var first_cross_src: ?*core_types.ModuleInstance = null;
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

                    var imp_func_idx: u32 = 0;
                    for (module_ptr.imports) |imp| {
                        if (imp.kind != .function) continue;
                        defer imp_func_idx += 1;

                        const source_inst_idx: u32 = arg_blk: {
                            for (ie.args) |arg| {
                                if (std.mem.eql(u8, arg.name, imp.module_name)) break :arg_blk arg.instance_idx;
                            }
                            break :arg_blk std.math.maxInt(u32);
                        };
                        if (source_inst_idx == std.math.maxInt(u32)) continue;
                        if (source_inst_idx >= ci_idx) continue;
                        const source_entry = cis[source_inst_idx];

                        if (source_entry.module_inst) |src_mi| {
                            // Cross-instance core wiring: look up the named
                            // export in the source module instance.
                            const target_func_idx = src_mi.getExportFunc(imp.field_name) orelse continue;
                            imps_buf[imp_func_idx] = .{ .module_inst = src_mi, .func_idx = target_func_idx };
                            is_cross[imp_func_idx] = true;
                            if (first_cross_src == null) first_cross_src = src_mi;
                            continue;
                        }

                        // Inline-exports source — existing canon.lower path.
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

                        // Resolve member_idx in core-func-index-space back to
                        // a canon.lower. Simplified layout: we assume the
                        // space is populated exclusively by canon.lower
                        // entries in canons[] order (no core aliases, no
                        // module-import funcs at top level). This covers the
                        // hand-authored 2A.2b fixture and common Rust emit
                        // patterns; the full resolver lands in a later slice.
                        const canon_idx = resolveCoreFuncLower(component, member_idx) orelse continue;
                        const canon = component.canons[canon_idx];
                        const lower = switch (canon) {
                            .lower => |l| l,
                            else => continue,
                        };

                        // Build and own the trampoline context. The actual
                        // `HostFunc.call` is resolved later by `linkImports`
                        // when the caller supplies the import providers;
                        // here we just record which component func index
                        // this trampoline is lowering.
                        const ctx_ptr = allocator.create(executor_mod.ComponentTrampolineCtx) catch continue;
                        if (lower.func_idx >= component.types.len) {
                            allocator.destroy(ctx_ptr);
                            continue;
                        }
                        const ft = switch (component.types[lower.func_idx]) {
                            .func => |ft_v| ft_v,
                            else => {
                                allocator.destroy(ctx_ptr);
                                continue;
                            },
                        };
                        const params = allocator.alloc(ctypes.ValType, ft.params.len) catch {
                            allocator.destroy(ctx_ptr);
                            continue;
                        };
                        for (ft.params, 0..) |p, i| params[i] = p.type;
                        const results = switch (ft.results) {
                            .none => allocator.alloc(ctypes.ValType, 0) catch {
                                allocator.free(params);
                                allocator.destroy(ctx_ptr);
                                continue;
                            },
                            .unnamed => |t| blk: {
                                const r = allocator.alloc(ctypes.ValType, 1) catch {
                                    allocator.free(params);
                                    allocator.destroy(ctx_ptr);
                                    continue;
                                };
                                r[0] = t;
                                break :blk r;
                            },
                            .named => |named| blk: {
                                const r = allocator.alloc(ctypes.ValType, named.len) catch {
                                    allocator.free(params);
                                    allocator.destroy(ctx_ptr);
                                    continue;
                                };
                                for (named, 0..) |n, i| r[i] = n.type;
                                break :blk r;
                            },
                        };
                        ctx_ptr.* = .{
                            .comp_inst = inst,
                            .host_func = .{}, // filled in by linkImports
                            .component_func_idx = lower.func_idx,
                            .param_types = params,
                            .result_types = results,
                            .lower_opts = executor_mod.LowerOptions.fromOpts(lower.opts),
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
                    }

                    // Instantiate, optionally seeding `import_functions` for
                    // cross-instance wiring. Slots that aren't cross-instance
                    // get a safe placeholder pointing at any cross-source we
                    // saw — interp dispatch never reaches them because their
                    // `host_func_entries[i]` is non-null (canon.lower) or the
                    // import is unresolved (caught by the no-op stub before
                    // `import_functions` is consulted).
                    const mi = blk: {
                        if (first_cross_src) |placeholder| {
                            for (imps_buf, 0..) |*slot, i| {
                                if (!is_cross[i]) {
                                    slot.* = .{ .module_inst = placeholder, .func_idx = 0 };
                                }
                            }
                            const ctx = inst_mod.ImportContext{ .functions = imps_buf };
                            break :blk inst_mod.instantiateWithImports(module_ptr, allocator, ctx) catch {
                                if (entries.len > 0) allocator.free(entries);
                                if (imps_buf.len > 0) allocator.free(imps_buf);
                                if (is_cross.len > 0) allocator.free(is_cross);
                                continue;
                            };
                        }
                        break :blk inst_mod.instantiate(module_ptr, allocator) catch {
                            if (entries.len > 0) allocator.free(entries);
                            if (imps_buf.len > 0) allocator.free(imps_buf);
                            if (is_cross.len > 0) allocator.free(is_cross);
                            continue;
                        };
                    };
                    cis[ci_idx] = .{ .module_inst = mi };

                    if (entries.len > 0) inst_mod.attachHostFuncEntries(mi, entries);
                    if (imps_buf.len > 0) allocator.free(imps_buf);
                    if (is_cross.len > 0) allocator.free(is_cross);
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
        .aliased => null,
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
        // canon.lowers are imports into core modules — not callable as
        // exports — so a canon.lift pointing at one is malformed.
        .lowered => return null,
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
