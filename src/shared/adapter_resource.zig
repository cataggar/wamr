//! Conditional stable ownership for adapter-local WASI resource handles.
//!
//! The disabled specialization keeps the historical compact array fast path.
//! Its leases are scoped pointers and must not cross mutation of the same
//! table. Thread-enabled builds publish stable nodes and require callers to
//! hold a retained lease while using a resource. Directory locks are never
//! held while user destructors run.

const std = @import("std");
const config = @import("config");
const stable_resource = @import("stable_resource.zig");

pub fn ResourceTable(
    comptime T: type,
    comptime Context: type,
    comptime destroy: fn (Context, *T) void,
    comptime reserve_zero: bool,
) type {
    return ResourceTableFor(
        config.lib_wasi_threads,
        T,
        Context,
        destroy,
        reserve_zero,
        64,
    );
}

pub fn ResourceTableFor(
    comptime enabled: bool,
    comptime T: type,
    comptime Context: type,
    comptime destroy: fn (Context, *T) void,
    comptime reserve_zero: bool,
    comptime chunk_capacity: usize,
) type {
    const NodeMutex = stable_resource.ConditionalMutexFor(
        enabled,
        stable_resource.LockRank.resource_node,
    );
    const InitMutex = stable_resource.ConditionalMutexFor(
        enabled,
        stable_resource.LockRank.resource_registry,
    );
    const DirectoryMutex = stable_resource.ConditionalMutexFor(
        enabled,
        stable_resource.LockRank.resource_directory,
    );

    const Entry = struct {
        mutex: NodeMutex = .init,
        value: T,
    };

    const Destroyer = struct {
        fn run(context: Context, entry: *Entry) void {
            stable_resource.assertNoLocksHeldFor(enabled);
            destroy(context, &entry.value);
        }
    };

    const Stable = stable_resource.StableHandleTableFor(
        enabled,
        Entry,
        Context,
        Destroyer.run,
        chunk_capacity,
    );

    return struct {
        const Self = @This();

        pub const Lease = struct {
            storage: if (enabled) Stable.Lease else ?*Entry,
            locked: bool = false,

            pub fn value(self: *Lease) *T {
                const node_entry = self.entry();
                return &node_entry.value;
            }

            pub fn lock(self: *Lease) *T {
                std.debug.assert(!self.locked);
                const node_entry = self.entry();
                node_entry.mutex.lock();
                self.locked = true;
                return &node_entry.value;
            }

            pub fn unlock(self: *Lease) void {
                std.debug.assert(self.locked);
                self.entry().mutex.unlock();
                self.locked = false;
            }

            pub fn isClosing(self: *const Lease) bool {
                if (comptime enabled) return self.storage.isClosing();
                return false;
            }

            pub fn retain(self: *const Lease) Lease {
                if (comptime enabled) {
                    return .{ .storage = self.storage.retain() };
                }
                return .{ .storage = self.storage };
            }

            pub fn release(self: *Lease) void {
                std.debug.assert(!self.locked);
                if (comptime enabled) {
                    self.storage.release();
                } else {
                    self.storage = null;
                }
            }

            pub fn deinit(self: *Lease) void {
                self.release();
            }

            fn entry(self: *Lease) *Entry {
                if (comptime enabled) return self.storage.value();
                return self.storage.?;
            }
        };

        pub const Publication = struct {
            handle: u32,
            lease: Lease,
        };

        allocator: std.mem.Allocator,
        context: Context,
        init_mutex: InitMutex = .init,
        directory_mutex: DirectoryMutex = .init,
        stable: if (enabled) ?Stable else void = if (enabled) null else {},
        entries: if (enabled) void else std.ArrayListUnmanaged(?Entry) =
            if (enabled) {} else .empty,
        published: if (enabled) void else usize = if (enabled) {} else 0,
        shutting_down: bool = false,

        pub fn init(allocator: std.mem.Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .context = context,
            };
        }

        /// Publish a resource and take ownership only on success.
        pub fn publish(self: *Self, value: T) !u32 {
            if (comptime enabled) {
                var table = try self.stableTable();
                const internal = try table.publish(.{ .value = value });
                if (comptime reserve_zero) {
                    if (internal >= std.math.maxInt(u32) - 1) {
                        std.debug.assert(table.withdraw(internal) != null);
                        return error.HandleExhausted;
                    }
                    return internal + 1;
                }
                return internal;
            }

            self.directory_mutex.lock();
            defer self.directory_mutex.unlock();
            if (self.shutting_down) return error.TableShuttingDown;

            const start: usize = if (reserve_zero) 1 else 0;
            if (reserve_zero and self.entries.items.len == 0) {
                try self.entries.append(self.allocator, null);
            }
            for (self.entries.items[start..], start..) |slot, index| {
                if (slot == null) {
                    self.entries.items[index] = .{ .value = value };
                    self.published += 1;
                    return @intCast(index);
                }
            }
            if (self.entries.items.len > std.math.maxInt(u32)) {
                return error.HandleExhausted;
            }
            const handle: u32 = @intCast(self.entries.items.len);
            try self.entries.append(self.allocator, .{ .value = value });
            self.published += 1;
            return handle;
        }

        /// Publish and return a retained lease that is established before
        /// shutdown/removal can retire the new handle.
        pub fn publishLeased(self: *Self, value: T) !Publication {
            if (comptime enabled) {
                var table = try self.stableTable();
                var published = try table.publishLeased(.{ .value = value });
                if (comptime reserve_zero) {
                    if (published.handle >= std.math.maxInt(u32) - 1) {
                        published.lease.release();
                        std.debug.assert(table.withdraw(published.handle) != null);
                        return error.HandleExhausted;
                    }
                    return .{
                        .handle = published.handle + 1,
                        .lease = .{ .storage = published.lease },
                    };
                }
                return .{
                    .handle = published.handle,
                    .lease = .{ .storage = published.lease },
                };
            }

            const handle = try self.publish(value);
            return .{
                .handle = handle,
                .lease = self.acquire(handle).?,
            };
        }

        pub fn acquire(self: *Self, handle: u32) ?Lease {
            if (comptime enabled) {
                const internal = externalToInternal(handle) orelse return null;
                self.init_mutex.lock();
                if (self.shutting_down) {
                    self.init_mutex.unlock();
                    return null;
                }
                var table = self.stable orelse {
                    self.init_mutex.unlock();
                    return null;
                };
                self.init_mutex.unlock();
                const lease = table.acquire(internal) orelse return null;
                return .{ .storage = lease };
            }

            self.directory_mutex.lock();
            defer self.directory_mutex.unlock();
            if (self.shutting_down or handle >= self.entries.items.len) return null;
            if (self.entries.items[handle]) |*entry| {
                return .{ .storage = entry };
            }
            return null;
        }

        /// Remove a published resource. Destruction is deferred until the
        /// final enabled-build lease and always runs without a table lock.
        pub fn remove(self: *Self, handle: u32) bool {
            if (comptime enabled) {
                const internal = externalToInternal(handle) orelse return false;
                self.init_mutex.lock();
                var table = self.stable orelse {
                    self.init_mutex.unlock();
                    return false;
                };
                self.init_mutex.unlock();
                return table.remove(internal);
            }

            self.directory_mutex.lock();
            if (handle >= self.entries.items.len or self.entries.items[handle] == null) {
                self.directory_mutex.unlock();
                return false;
            }
            var entry = self.entries.items[handle].?;
            self.entries.items[handle] = null;
            self.published -= 1;
            self.directory_mutex.unlock();
            Destroyer.run(self.context, &entry);
            return true;
        }

        /// Roll back an unpublished-to-the-guest insertion without invoking
        /// the destructor. The returned value remains caller-owned.
        pub fn withdraw(self: *Self, handle: u32) ?T {
            if (comptime enabled) {
                const internal = externalToInternal(handle) orelse return null;
                self.init_mutex.lock();
                var table = self.stable orelse {
                    self.init_mutex.unlock();
                    return null;
                };
                self.init_mutex.unlock();
                const entry = table.withdraw(internal) orelse return null;
                return entry.value;
            }

            self.directory_mutex.lock();
            if (handle >= self.entries.items.len) {
                self.directory_mutex.unlock();
                return null;
            }
            const entry = self.entries.items[handle] orelse {
                self.directory_mutex.unlock();
                return null;
            };
            self.entries.items[handle] = null;
            self.published -= 1;
            self.directory_mutex.unlock();
            return entry.value;
        }

        pub fn contains(self: *Self, handle: u32) bool {
            var lease = self.acquire(handle) orelse return false;
            lease.release();
            return true;
        }

        /// Test/debug-only immediate pointer view. The pointer remains valid
        /// while the handle stays published, but carries no lease and must
        /// never be retained across concurrent removal.
        pub fn unsafeGetPtrForTest(self: *Self, handle: u32) ?*T {
            if (comptime enabled) {
                var lease = self.acquire(handle) orelse return null;
                const pointer = lease.value();
                lease.release();
                return pointer;
            }
            self.directory_mutex.lock();
            defer self.directory_mutex.unlock();
            if (self.shutting_down or handle >= self.entries.items.len) return null;
            if (self.entries.items[handle]) |*entry| return &entry.value;
            return null;
        }

        pub fn publishedCount(self: *Self) usize {
            if (comptime enabled) {
                self.init_mutex.lock();
                var table = self.stable;
                self.init_mutex.unlock();
                if (table) |*active| return active.stats().published;
                return 0;
            } else {
                self.directory_mutex.lock();
                defer self.directory_mutex.unlock();
                return self.published;
            }
        }

        pub fn snapshotHandles(self: *Self, allocator: std.mem.Allocator) ![]u32 {
            if (comptime enabled) {
                self.init_mutex.lock();
                var table = self.stable orelse {
                    self.init_mutex.unlock();
                    return allocator.alloc(u32, 0);
                };
                self.init_mutex.unlock();
                const handles = try table.snapshotHandles(allocator);
                if (comptime reserve_zero) {
                    for (handles) |*handle| handle.* += 1;
                }
                return handles;
            }

            self.directory_mutex.lock();
            const count = self.published;
            self.directory_mutex.unlock();

            const handles = try allocator.alloc(u32, count);
            self.directory_mutex.lock();
            var filled: usize = 0;
            for (self.entries.items, 0..) |slot, index| {
                if (slot != null and filled < handles.len) {
                    handles[filled] = @intCast(index);
                    filled += 1;
                }
            }
            self.directory_mutex.unlock();
            if (filled == handles.len) return handles;
            return try allocator.realloc(handles, filled);
        }

        pub fn shutdown(self: *Self) void {
            if (comptime enabled) {
                self.init_mutex.lock();
                if (self.shutting_down) {
                    self.init_mutex.unlock();
                    return;
                }
                self.shutting_down = true;
                var table = self.stable;
                self.init_mutex.unlock();
                if (table) |*active| active.shutdown();
                return;
            }

            self.directory_mutex.lock();
            if (self.shutting_down) {
                self.directory_mutex.unlock();
                return;
            }
            self.shutting_down = true;
            self.directory_mutex.unlock();

            while (true) {
                self.directory_mutex.lock();
                var entry_to_destroy: ?Entry = null;
                for (self.entries.items) |*slot| {
                    if (slot.*) |entry| {
                        entry_to_destroy = entry;
                        slot.* = null;
                        self.published -= 1;
                        break;
                    }
                }
                self.directory_mutex.unlock();
                if (entry_to_destroy) |*entry| {
                    Destroyer.run(self.context, entry);
                } else {
                    break;
                }
            }
        }

        pub fn isQuiescent(self: *Self) bool {
            if (comptime enabled) {
                self.init_mutex.lock();
                var table = self.stable;
                self.init_mutex.unlock();
                if (table) |*active| return active.isQuiescent();
                return true;
            } else {
                self.directory_mutex.lock();
                defer self.directory_mutex.unlock();
                return self.published == 0;
            }
        }

        pub fn leakCount(self: *Self) usize {
            if (comptime enabled) {
                self.init_mutex.lock();
                var table = self.stable;
                self.init_mutex.unlock();
                if (table) |*active| return active.leakCount();
                return 0;
            } else {
                self.directory_mutex.lock();
                defer self.directory_mutex.unlock();
                return self.published;
            }
        }

        pub fn deinit(self: *Self) !void {
            self.shutdown();
            if (comptime enabled) {
                self.init_mutex.lock();
                var table = self.stable;
                self.init_mutex.unlock();
                if (table) |*active| {
                    try active.deinit();
                    self.init_mutex.lock();
                    self.stable = null;
                    self.init_mutex.unlock();
                }
            } else {
                self.entries.deinit(self.allocator);
            }
        }

        fn stableTable(self: *Self) !Stable {
            self.init_mutex.lock();
            if (self.shutting_down) {
                self.init_mutex.unlock();
                return error.TableShuttingDown;
            }
            if (self.stable) |table| {
                self.init_mutex.unlock();
                return table;
            }
            self.init_mutex.unlock();

            var candidate = try Stable.init(self.allocator, self.context);
            var installed = false;
            self.init_mutex.lock();
            if (self.shutting_down) {
                self.init_mutex.unlock();
                try candidate.deinit();
                return error.TableShuttingDown;
            }
            if (self.stable) |table| {
                self.init_mutex.unlock();
                try candidate.deinit();
                return table;
            }
            self.stable = candidate;
            installed = true;
            const table = self.stable.?;
            self.init_mutex.unlock();
            std.debug.assert(installed);
            return table;
        }

        fn externalToInternal(handle: u32) ?u32 {
            if (comptime reserve_zero) {
                if (handle == 0) return null;
                return handle - 1;
            }
            return handle;
        }
    };
}

const TestResource = struct {
    value: usize,
};

fn destroyTestResource(counter: *std.atomic.Value(usize), _: *TestResource) void {
    _ = counter.fetchAdd(1, .monotonic);
}

test "adapter resource table preserves disabled and enabled handle conventions" {
    inline for (.{ false, true }) |enabled| {
        const Table = ResourceTableFor(
            enabled,
            TestResource,
            *std.atomic.Value(usize),
            destroyTestResource,
            true,
            2,
        );
        var destroyed = std.atomic.Value(usize).init(0);
        var table = Table.init(std.testing.allocator, &destroyed);
        const first = try table.publish(.{ .value = 11 });
        const second = try table.publish(.{ .value = 22 });
        try std.testing.expectEqual(@as(u32, 1), first);
        try std.testing.expectEqual(@as(u32, 2), second);
        try std.testing.expect(!table.contains(0));

        var lease = table.acquire(first).?;
        try std.testing.expectEqual(@as(usize, 11), lease.value().value);
        lease.release();
        try std.testing.expect(table.remove(first));
        try std.testing.expect(!table.remove(first));
        try std.testing.expectEqual(@as(usize, 1), destroyed.load(.monotonic));
        try table.deinit();
        try std.testing.expectEqual(@as(usize, 2), destroyed.load(.monotonic));
    }
}

test "enabled adapter resource removal waits for the final lease" {
    const Table = ResourceTableFor(
        true,
        TestResource,
        *std.atomic.Value(usize),
        destroyTestResource,
        false,
        2,
    );
    var destroyed = std.atomic.Value(usize).init(0);
    var table = Table.init(std.testing.allocator, &destroyed);
    const handle = try table.publish(.{ .value = 41 });
    var lease = table.acquire(handle).?;

    const Runner = struct {
        fn run(target: *Table, target_handle: u32) void {
            std.debug.assert(target.remove(target_handle));
        }
    };
    const thread = try std.Thread.spawn(.{}, Runner.run, .{ &table, handle });
    thread.join();

    try std.testing.expectEqual(@as(usize, 0), destroyed.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 41), lease.value().value);
    try std.testing.expect(lease.isClosing());
    lease.release();
    try std.testing.expectEqual(@as(usize, 1), destroyed.load(.monotonic));
    try table.deinit();
}

test "adapter publishLeased survives immediate retirement" {
    const Table = ResourceTableFor(
        true,
        TestResource,
        *std.atomic.Value(usize),
        destroyTestResource,
        true,
        2,
    );
    var destroyed = std.atomic.Value(usize).init(0);
    var table = Table.init(std.testing.allocator, &destroyed);
    var published = try table.publishLeased(.{ .value = 29 });
    try std.testing.expectEqual(@as(u32, 1), published.handle);
    try std.testing.expect(table.remove(published.handle));
    try std.testing.expect(published.lease.isClosing());
    try std.testing.expectEqual(@as(usize, 29), published.lease.value().value);
    published.lease.release();
    try std.testing.expectEqual(@as(usize, 1), destroyed.load(.monotonic));
    try table.deinit();
}

test "adapter resource handles preserve disabled reuse and reject enabled ABA" {
    inline for (.{ false, true }) |enabled| {
        const Table = ResourceTableFor(
            enabled,
            TestResource,
            *std.atomic.Value(usize),
            destroyTestResource,
            true,
            2,
        );
        var destroyed = std.atomic.Value(usize).init(0);
        var table = Table.init(std.testing.allocator, &destroyed);
        const stale = try table.publish(.{ .value = 1 });
        try std.testing.expect(table.remove(stale));
        const replacement = try table.publish(.{ .value = 2 });
        if (enabled) {
            try std.testing.expect(stale != replacement);
            try std.testing.expect(table.acquire(stale) == null);
        } else {
            try std.testing.expectEqual(stale, replacement);
        }
        try table.deinit();
        try std.testing.expectEqual(@as(usize, 2), destroyed.load(.monotonic));
    }
}

test "adapter resource shutdown reports outstanding enabled leases" {
    const Table = ResourceTableFor(
        true,
        TestResource,
        *std.atomic.Value(usize),
        destroyTestResource,
        false,
        2,
    );
    var destroyed = std.atomic.Value(usize).init(0);
    var table = Table.init(std.testing.allocator, &destroyed);
    const handle = try table.publish(.{ .value = 7 });
    var lease = table.acquire(handle).?;
    table.shutdown();
    try std.testing.expectEqual(@as(usize, 1), table.leakCount());
    try std.testing.expectError(error.LeasesOutstanding, table.deinit());
    lease.release();
    try table.deinit();
    try std.testing.expectEqual(@as(usize, 1), destroyed.load(.monotonic));
}
