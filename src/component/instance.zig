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

/// A runtime component instance — the result of instantiating a Component.
pub const ComponentInstance = struct {
    /// The parsed component this instance was created from.
    component: *const ctypes.Component,
    /// Core module instances created during instantiation.
    core_instances: []CoreInstanceEntry,
    /// Resource tables, one per resource type defined in the component.
    resource_tables: []ResourceTable,
    /// Exported functions (component-level func index → core func index + instance).
    exported_funcs: std.StringHashMapUnmanaged(ExportedFunc),
    /// Allocator for instance lifetime.
    allocator: std.mem.Allocator,

    pub const CoreInstanceEntry = struct {
        module_inst: ?*core_types.ModuleInstance,
    };

    pub const ExportedFunc = struct {
        core_instance_idx: u32,
        core_func_idx: u32,
    };

    /// Look up an exported function by name.
    pub fn getExport(self: *const ComponentInstance, name: []const u8) ?ExportedFunc {
        return self.exported_funcs.get(name);
    }

    pub fn deinit(self: *ComponentInstance) void {
        for (self.resource_tables) |*rt| rt.deinit(self.allocator);
        if (self.resource_tables.len > 0) self.allocator.free(self.resource_tables);
        if (self.core_instances.len > 0) self.allocator.free(self.core_instances);
        self.exported_funcs.deinit(self.allocator);
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
    var resource_count: u32 = 0;
    for (component.types) |td| {
        if (td == .resource) resource_count += 1;
    }

    // Allocate resource tables
    const resource_tables: []ResourceTable = if (resource_count > 0) blk: {
        const rts = allocator.alloc(ResourceTable, resource_count) catch return error.OutOfMemory;
        for (rts) |*rt| rt.* = .{};
        break :blk rts;
    } else &.{};

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
        .resource_tables = resource_tables,
        .exported_funcs = exported_funcs,
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
