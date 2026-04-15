//! Component composition — linking components by matching exports to imports.
//!
//! Implements the component linker that wires one component's exports
//! to another component's imports based on interface name matching.

const std = @import("std");
const ctypes = @import("types.zig");

/// A composition plan describes how to link multiple components together.
pub const CompositionPlan = struct {
    /// Components to instantiate, in dependency order.
    entries: []const Entry,

    pub const Entry = struct {
        component: *const ctypes.Component,
        /// Import resolutions: import name → source (entry index + export name).
        import_bindings: []const ImportBinding,
    };

    pub const ImportBinding = struct {
        import_name: []const u8,
        source_entry: u32, // index into entries
        export_name: []const u8,
    };
};

/// Validate that two components can be linked: every import of `consumer`
/// must be satisfied by an export of `provider` with a matching name.
pub fn validateLink(
    consumer: *const ctypes.Component,
    provider: *const ctypes.Component,
) LinkResult {
    var unresolved: u32 = 0;
    var resolved: u32 = 0;

    for (consumer.imports) |imp| {
        var found = false;
        for (provider.exports) |exp| {
            if (std.mem.eql(u8, imp.name, exp.name)) {
                found = true;
                break;
            }
        }
        if (found) {
            resolved += 1;
        } else {
            unresolved += 1;
        }
    }

    return .{
        .resolved = resolved,
        .unresolved = unresolved,
        .total_imports = @intCast(consumer.imports.len),
    };
}

pub const LinkResult = struct {
    resolved: u32,
    unresolved: u32,
    total_imports: u32,

    pub fn isFullyResolved(self: LinkResult) bool {
        return self.unresolved == 0;
    }
};

/// Find matching exports for a component's imports from a set of providers.
pub fn resolveImports(
    consumer: *const ctypes.Component,
    providers: []const *const ctypes.Component,
    allocator: std.mem.Allocator,
) ![]const CompositionPlan.ImportBinding {
    var bindings: std.ArrayListUnmanaged(CompositionPlan.ImportBinding) = .empty;

    for (consumer.imports) |imp| {
        for (providers, 0..) |provider, provider_idx| {
            for (provider.exports) |exp| {
                if (std.mem.eql(u8, imp.name, exp.name)) {
                    try bindings.append(allocator, .{
                        .import_name = imp.name,
                        .source_entry = @intCast(provider_idx),
                        .export_name = exp.name,
                    });
                    break;
                }
            }
        }
    }

    return try bindings.toOwnedSlice(allocator);
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "validateLink: fully resolved" {
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "my-func", .desc = .{ .func = 0 } },
    };
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "my-func", .desc = .{ .func = 0 } },
    };
    const consumer = ctypes.Component{
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
    const provider = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &exports,
    };

    const result = validateLink(&consumer, &provider);
    try std.testing.expect(result.isFullyResolved());
    try std.testing.expectEqual(@as(u32, 1), result.resolved);
}

test "validateLink: unresolved import" {
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "missing", .desc = .{ .func = 0 } },
    };
    const consumer = ctypes.Component{
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
    const provider = ctypes.Component{
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

    const result = validateLink(&consumer, &provider);
    try std.testing.expect(!result.isFullyResolved());
    try std.testing.expectEqual(@as(u32, 1), result.unresolved);
}

test "resolveImports: finds matching exports" {
    const allocator = std.testing.allocator;

    const imports = [_]ctypes.ImportDecl{
        .{ .name = "do-work", .desc = .{ .func = 0 } },
    };
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "do-work", .desc = .{ .func = 0 } },
    };
    const consumer = ctypes.Component{
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
    const provider = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &exports,
    };

    const providers = [_]*const ctypes.Component{&provider};
    const bindings = try resolveImports(&consumer, &providers, allocator);
    defer allocator.free(bindings);

    try std.testing.expectEqual(@as(usize, 1), bindings.len);
    try std.testing.expectEqualStrings("do-work", bindings[0].import_name);
}
