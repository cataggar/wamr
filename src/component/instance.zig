//! Component instance — runtime state for an instantiated component.
//!
//! Manages resource tables, canonical function wrappers, and links
//! between component instances and their underlying core module instances.

const std = @import("std");
const ctypes = @import("types.zig");
const core_types = @import("../runtime/common/types.zig");

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
    /// Whether the start function has been executed.
    started: bool = false,
    /// Allocator for instance lifetime.
    allocator: std.mem.Allocator,

    pub const CoreInstanceEntry = struct {
        module_inst: ?*core_types.ModuleInstance,
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
        if (self.core_instances.len > 0) self.allocator.free(self.core_instances);
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
/// This walks the component's sections in order, creating core module
/// instances, resolving aliases, and wiring up canonical functions.
pub fn instantiate(
    component: *const ctypes.Component,
    allocator: std.mem.Allocator,
) InstantiationError!*ComponentInstance {
    const inst = allocator.create(ComponentInstance) catch return error.OutOfMemory;
    errdefer allocator.destroy(inst);

    // Count resource types
    // Resource tables are allocated lazily on first use (see
    // ComponentInstance.getOrCreateResourceTable). Nothing to do here.

    // Instantiate core modules
    const core_module_count = component.core_modules.len;
    const core_instances: []ComponentInstance.CoreInstanceEntry = if (core_module_count > 0) blk: {
        const cis = allocator.alloc(ComponentInstance.CoreInstanceEntry, core_module_count) catch return error.OutOfMemory;
        for (component.core_modules, 0..) |core_mod, i| {
            const loader = @import("../runtime/interpreter/loader.zig");
            const inst_mod = @import("../runtime/interpreter/instance.zig");

            const module = loader.load(core_mod.data, allocator) catch {
                cis[i] = .{ .module_inst = null };
                continue;
            };
            const module_ptr = allocator.create(core_types.WasmModule) catch {
                cis[i] = .{ .module_inst = null };
                continue;
            };
            module_ptr.* = module;
            const module_inst = inst_mod.instantiate(module_ptr, allocator) catch {
                cis[i] = .{ .module_inst = null };
                continue;
            };
            cis[i] = .{ .module_inst = module_inst };
        }
        break :blk cis;
    } else &.{};

    // Build export map
    var exported_funcs: std.StringHashMapUnmanaged(ComponentInstance.ExportedFunc) = .{};
    for (component.exports) |exp| {
        switch (exp.desc) {
            .func => |func_idx| {
                // Map component function exports. For now, assume canon lift
                // links directly to core functions in the first core instance.
                if (func_idx < component.canons.len) {
                    const canon = component.canons[func_idx];
                    switch (canon) {
                        .lift => |lift| {
                            exported_funcs.put(allocator, exp.name, .{
                                .core_instance_idx = 0,
                                .core_func_idx = lift.core_func_idx,
                                .func_type_idx = lift.type_idx,
                                .opts = lift.opts,
                            }) catch {};
                        },
                        else => {},
                    }
                }
            },
            else => {},
        }
    }

    inst.* = .{
        .component = component,
        .core_instances = core_instances,
        .resource_tables = .empty,
        .exported_funcs = exported_funcs,
        .imports = .{},
        .allocator = allocator,
    };

    return inst;
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
