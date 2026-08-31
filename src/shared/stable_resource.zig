//! Conditional synchronization and stable lease-based handle tables.
//!
//! The configured aliases erase synchronization and reference-count storage
//! when WASI threads are disabled. In that mode a lease is a scoped borrow:
//! callers must release it before removing its handle. Thread-enabled tables
//! retain retired nodes until the final lease is released.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");

pub const Handle = u32;

/// Lock ranks increase from outer to inner locks.
pub const LockRank = struct {
    pub const resource_registry: u16 = 50;
    pub const resource_directory: u16 = 100;
    pub const resource_node: u16 = 200;
    pub const adapter_relation: u16 = 210;
    pub const adapter_input_stream: u16 = 220;
    pub const adapter_output_stream: u16 = 221;
    pub const adapter_resource: u16 = 230;
    pub const adapter_state: u16 = 240;
    pub const core_instance: u16 = 250;
    pub const core_table: u16 = 300;
};

const debug_lock_tracking = builtin.mode == .Debug;
const max_debug_lock_depth = 32;

const DebugLockStack = struct {
    ranks: [max_debug_lock_depth]u16 = @splat(0),
    depth: usize = 0,
};

threadlocal var debug_lock_stack: DebugLockStack = .{};

pub fn debugCanAcquireFor(comptime enabled: bool, rank: u16) bool {
    if (comptime !enabled or !debug_lock_tracking) return true;
    if (debug_lock_stack.depth == max_debug_lock_depth) return false;
    if (debug_lock_stack.depth == 0) return true;
    return rank > debug_lock_stack.ranks[debug_lock_stack.depth - 1];
}

pub fn debugNoLocksHeldFor(comptime enabled: bool) bool {
    if (comptime !enabled or !debug_lock_tracking) return true;
    return debug_lock_stack.depth == 0;
}

pub fn assertNoLocksHeldFor(comptime enabled: bool) void {
    if (comptime enabled and debug_lock_tracking) {
        std.debug.assert(debugNoLocksHeldFor(enabled));
    }
}

pub fn assertNoLocksHeld() void {
    assertNoLocksHeldFor(config.lib_wasi_threads);
}

fn debugRecordAcquire(comptime enabled: bool, rank: u16) void {
    if (comptime enabled and debug_lock_tracking) {
        std.debug.assert(debugCanAcquireFor(enabled, rank));
        debug_lock_stack.ranks[debug_lock_stack.depth] = rank;
        debug_lock_stack.depth += 1;
    }
}

fn debugRecordRelease(comptime enabled: bool, rank: u16) void {
    if (comptime enabled and debug_lock_tracking) {
        std.debug.assert(debug_lock_stack.depth > 0);
        std.debug.assert(debug_lock_stack.ranks[debug_lock_stack.depth - 1] == rank);
        debug_lock_stack.depth -= 1;
    }
}

/// A ranked spin mutex whose disabled specialization has zero size.
pub fn ConditionalMutexFor(comptime enabled: bool, comptime rank: u16) type {
    return if (enabled) struct {
        const Self = @This();

        state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

        pub const init: Self = .{};

        pub fn tryLock(self: *Self) bool {
            if (comptime debug_lock_tracking) {
                std.debug.assert(debugCanAcquireFor(true, rank));
            }
            if (self.state.cmpxchgStrong(0, 1, .acquire, .monotonic) != null) {
                return false;
            }
            debugRecordAcquire(true, rank);
            return true;
        }

        pub fn lock(self: *Self) void {
            if (comptime debug_lock_tracking) {
                std.debug.assert(debugCanAcquireFor(true, rank));
            }
            while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
                std.atomic.spinLoopHint();
            }
            debugRecordAcquire(true, rank);
        }

        pub fn unlock(self: *Self) void {
            self.state.store(0, .release);
            debugRecordRelease(true, rank);
        }
    } else struct {
        const Self = @This();

        pub const init: Self = .{};

        pub inline fn tryLock(_: *Self) bool {
            return true;
        }

        pub inline fn lock(_: *Self) void {}

        pub inline fn unlock(_: *Self) void {}
    };
}

pub fn ConditionalMutex(comptime rank: u16) type {
    return ConditionalMutexFor(config.lib_wasi_threads, rank);
}

/// A zero-sized disabled / ordinary blocking mutex enabled specialization
/// for resource operation gates. Unlike table mutexes, operation gates are
/// intentionally unranked because they may cover host I/O but must never be
/// held by table directory code or resource destructors.
pub fn ConditionalOperationMutexFor(comptime enabled: bool) type {
    return if (enabled) struct {
        state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

        pub const init: @This() = .{};

        pub fn lock(self: *@This()) void {
            while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
                std.atomic.spinLoopHint();
            }
        }

        pub fn unlock(self: *@This()) void {
            self.state.store(0, .release);
        }
    } else struct {
        pub const init: @This() = .{};

        pub inline fn lock(_: *@This()) void {}
        pub inline fn unlock(_: *@This()) void {}
    };
}

pub const ConditionalOperationMutex =
    ConditionalOperationMutexFor(config.lib_wasi_threads);

/// An atomic reference count whose disabled specialization has no state.
///
/// `release` always reports the sole reference in the disabled specialization;
/// containers using it must compile out lease retains as StableHandleTable does.
pub fn ConditionalRefCountFor(comptime enabled: bool) type {
    return if (enabled) struct {
        const Self = @This();

        value: std.atomic.Value(usize),

        pub fn init(initial: usize) Self {
            std.debug.assert(initial > 0);
            return .{ .value = std.atomic.Value(usize).init(initial) };
        }

        pub fn retain(self: *Self) void {
            var current = self.value.load(.monotonic);
            while (true) {
                std.debug.assert(current > 0);
                std.debug.assert(current < std.math.maxInt(usize));
                if (self.value.cmpxchgWeak(
                    current,
                    current + 1,
                    .monotonic,
                    .monotonic,
                )) |observed| {
                    current = observed;
                } else {
                    return;
                }
            }
        }

        pub fn release(self: *Self) bool {
            const previous = self.value.fetchSub(1, .release);
            std.debug.assert(previous > 0);
            if (previous != 1) return false;
            _ = self.value.load(.acquire);
            return true;
        }

        pub fn count(self: *const Self) usize {
            return self.value.load(.monotonic);
        }
    } else struct {
        const Self = @This();

        pub inline fn init(_: usize) Self {
            return .{};
        }

        pub inline fn retain(_: *Self) void {}

        pub inline fn release(_: *Self) bool {
            return true;
        }

        pub inline fn count(_: *const Self) usize {
            return 1;
        }
    };
}

pub const ConditionalRefCount = ConditionalRefCountFor(config.lib_wasi_threads);

/// An ownership reference count that is atomic only when sharing is enabled.
///
/// Unlike `ConditionalRefCountFor`, the disabled specialization keeps a
/// plain counter because cross-instance ownership still needs exact lifetime
/// accounting even in a single-threaded build.
pub fn ConditionalLifetimeRefCountFor(comptime enabled: bool) type {
    return if (enabled) struct {
        const Self = @This();

        value: std.atomic.Value(u32),

        pub fn init(initial: usize) Self {
            std.debug.assert(initial > 0 and initial <= std.math.maxInt(u32));
            return .{ .value = std.atomic.Value(u32).init(@intCast(initial)) };
        }

        pub fn retain(self: *Self) void {
            var current = self.value.load(.monotonic);
            while (true) {
                std.debug.assert(current > 0);
                std.debug.assert(current < std.math.maxInt(u32));
                if (self.value.cmpxchgWeak(
                    current,
                    current + 1,
                    .monotonic,
                    .monotonic,
                )) |observed| {
                    current = observed;
                } else {
                    return;
                }
            }
        }

        pub fn release(self: *Self) bool {
            const previous = self.value.fetchSub(1, .release);
            std.debug.assert(previous > 0);
            if (previous != 1) return false;
            _ = self.value.load(.acquire);
            return true;
        }

        pub fn count(self: *const Self) usize {
            return @intCast(self.value.load(.monotonic));
        }
    } else struct {
        const Self = @This();

        value: u32,

        pub fn init(initial: usize) Self {
            std.debug.assert(initial > 0 and initial <= std.math.maxInt(u32));
            return .{ .value = @intCast(initial) };
        }

        pub inline fn retain(self: *Self) void {
            std.debug.assert(self.value > 0);
            std.debug.assert(self.value < std.math.maxInt(u32));
            self.value += 1;
        }

        pub inline fn release(self: *Self) bool {
            std.debug.assert(self.value > 0);
            self.value -= 1;
            return self.value == 0;
        }

        pub inline fn count(self: *const Self) usize {
            return @intCast(self.value);
        }
    };
}

pub const ConditionalLifetimeRefCount =
    ConditionalLifetimeRefCountFor(config.lib_wasi_threads);

pub const NodeState = enum(u8) {
    published,
    closing,
    destroying,
};

pub const TableStats = struct {
    published: usize,
    retired: usize,
    live_nodes: usize,
    chunks: usize,
    shutting_down: bool,
};

/// A stable-address handle table selected by `config.lib_wasi_threads`.
pub fn StableHandleTable(
    comptime T: type,
    comptime Context: type,
    comptime destroy: fn (Context, *T) void,
) type {
    return StableHandleTableFor(
        config.lib_wasi_threads,
        T,
        Context,
        destroy,
        64,
    );
}

/// Implementation hook for focused tests and users with unusual directory
/// sizing requirements. Normal users should prefer `StableHandleTable`.
pub fn StableHandleTableFor(
    comptime enabled: bool,
    comptime T: type,
    comptime Context: type,
    comptime destroy: fn (Context, *T) void,
    comptime chunk_capacity: usize,
) type {
    comptime {
        if (chunk_capacity == 0) @compileError("chunk_capacity must be non-zero");
        if (chunk_capacity > std.math.maxInt(Handle)) {
            @compileError("chunk_capacity does not fit in a handle");
        }
    }

    return struct {
        const Self = @This();
        const RefCount = ConditionalRefCountFor(enabled);
        const DirectoryMutex = ConditionalMutexFor(enabled, LockRank.resource_directory);

        const Node = struct {
            owner: *Control,
            handle: Handle = 0,
            refs: RefCount = RefCount.init(1),
            state: if (enabled) std.atomic.Value(u8) else NodeState =
                if (enabled)
                    std.atomic.Value(u8).init(@intFromEnum(NodeState.published))
                else
                    .published,
            value: T,

            fn setState(self: *Node, new_state: NodeState) void {
                if (comptime enabled) {
                    self.state.store(@intFromEnum(new_state), .release);
                } else {
                    self.state = new_state;
                }
            }

            fn getState(self: *const Node) NodeState {
                if (comptime enabled) {
                    return @enumFromInt(self.state.load(.acquire));
                }
                return self.state;
            }
        };

        const Slot = union(enum) {
            never,
            published: *Node,
            retired: *Node,
            free: ?Handle,
        };

        const Chunk = struct {
            next: ?*Chunk = null,
            base: Handle = 0,
            used: usize = 0,
            slots: [chunk_capacity]Slot = @splat(.never),
        };

        const Control = struct {
            allocator: std.mem.Allocator,
            context: Context,
            directory: DirectoryMutex = .init,
            head: ?*Chunk = null,
            tail: ?*Chunk = null,
            free_head: ?Handle = null,
            published: usize = 0,
            retired: usize = 0,
            live_nodes: usize = 0,
            chunk_count: usize = 0,
            shutting_down: bool = false,
        };

        /// A lease keeps an enabled-build node alive after its handle closes.
        /// It is move-only by convention; call `release` exactly once.
        pub const Lease = struct {
            node: ?*Node,

            pub fn value(self: *Lease) *T {
                return &self.node.?.value;
            }

            pub fn isClosing(self: *const Lease) bool {
                return self.node.?.getState() != .published;
            }

            pub fn release(self: *Lease) void {
                const node = self.node orelse return;
                self.node = null;
                if (comptime enabled) {
                    if (node.refs.release()) finalizeNode(node);
                }
            }

            pub fn deinit(self: *Lease) void {
                self.release();
            }
        };

        control: ?*Control,

        pub fn init(allocator: std.mem.Allocator, context: Context) !Self {
            assertNoLocksHeldFor(enabled);
            const control = try allocator.create(Control);
            control.* = .{
                .allocator = allocator,
                .context = context,
            };
            return .{ .control = control };
        }

        /// Publishes `value`, taking ownership only on success.
        ///
        /// Node and directory growth allocations occur before the directory
        /// lock. On failure all table allocations are rolled back and the
        /// caller still owns `value`.
        pub fn publish(self: *Self, value: T) !Handle {
            const control = self.control.?;
            assertNoLocksHeldFor(enabled);

            const node = try control.allocator.create(Node);
            var committed = false;
            defer if (!committed) control.allocator.destroy(node);
            node.* = .{
                .owner = control,
                .value = value,
            };

            var spare_chunk: ?*Chunk = null;
            defer if (spare_chunk) |chunk| control.allocator.destroy(chunk);

            while (true) {
                control.directory.lock();
                if (control.shutting_down) {
                    control.directory.unlock();
                    return error.TableShuttingDown;
                }

                if (control.free_head) |handle| {
                    const slot = findSlot(control, handle).?;
                    const next = switch (slot.*) {
                        .free => |free_next| free_next,
                        else => unreachable,
                    };
                    control.free_head = next;
                    node.handle = handle;
                    slot.* = .{ .published = node };
                    recordPublish(control);
                    control.directory.unlock();
                    committed = true;
                    return handle;
                }

                if (control.tail) |tail| {
                    if (tail.used < chunk_capacity) {
                        const offset = tail.used;
                        const handle: Handle = tail.base + @as(Handle, @intCast(offset));
                        tail.used += 1;
                        tail.slots[offset] = .{ .published = node };
                        node.handle = handle;
                        recordPublish(control);
                        control.directory.unlock();
                        committed = true;
                        return handle;
                    }
                }

                if (spare_chunk) |chunk| {
                    const base: Handle = if (control.tail) |tail|
                        std.math.add(
                            Handle,
                            tail.base,
                            @as(Handle, @intCast(chunk_capacity)),
                        ) catch {
                            control.directory.unlock();
                            return error.HandleExhausted;
                        }
                    else
                        0;
                    if (base > std.math.maxInt(Handle) - (chunk_capacity - 1)) {
                        control.directory.unlock();
                        return error.HandleExhausted;
                    }
                    chunk.base = base;
                    chunk.used = 1;
                    chunk.slots[0] = .{ .published = node };
                    if (control.tail) |tail| {
                        tail.next = chunk;
                    } else {
                        control.head = chunk;
                    }
                    control.tail = chunk;
                    control.chunk_count += 1;
                    spare_chunk = null;
                    node.handle = base;
                    recordPublish(control);
                    control.directory.unlock();
                    committed = true;
                    return base;
                }

                if (control.tail) |tail| {
                    const next_base = std.math.add(
                        Handle,
                        tail.base,
                        @as(Handle, @intCast(chunk_capacity)),
                    ) catch {
                        control.directory.unlock();
                        return error.HandleExhausted;
                    };
                    if (next_base > std.math.maxInt(Handle) - (chunk_capacity - 1)) {
                        control.directory.unlock();
                        return error.HandleExhausted;
                    }
                }
                control.directory.unlock();

                assertNoLocksHeldFor(enabled);
                spare_chunk = try control.allocator.create(Chunk);
                spare_chunk.?.* = .{};
            }
        }

        /// Returns only stable, retained nodes; no directory-owned pointer is
        /// exposed without a lease.
        pub fn acquire(self: *Self, handle: Handle) ?Lease {
            const control = self.control.?;
            control.directory.lock();
            defer control.directory.unlock();
            if (control.shutting_down) return null;
            const slot = findSlot(control, handle) orelse return null;
            const node = switch (slot.*) {
                .published => |published| published,
                else => return null,
            };
            if (comptime enabled) node.refs.retain();
            return .{ .node = node };
        }

        /// Roll back a successful publication without invoking `destroy`.
        ///
        /// This succeeds only while the table owns the sole reference, so it
        /// is intended for callers that have not exposed the handle yet. The
        /// returned value remains caller-owned.
        pub fn withdraw(self: *Self, handle: Handle) ?T {
            const control = self.control.?;
            control.directory.lock();
            const slot = findSlot(control, handle) orelse {
                control.directory.unlock();
                return null;
            };
            const node = switch (slot.*) {
                .published => |published| published,
                else => {
                    control.directory.unlock();
                    return null;
                },
            };
            if (comptime enabled) {
                if (node.refs.count() != 1) {
                    control.directory.unlock();
                    return null;
                }
            }

            node.setState(.destroying);
            slot.* = .{ .free = control.free_head };
            control.free_head = node.handle;
            control.published -= 1;
            control.live_nodes -= 1;
            control.directory.unlock();

            const value = node.value;
            control.allocator.destroy(node);
            return value;
        }

        /// Unpublishes a handle. Destruction happens after unlocking, and in
        /// enabled builds is deferred until every lease has been released.
        pub fn remove(self: *Self, handle: Handle) bool {
            const control = self.control.?;
            control.directory.lock();
            const slot = findSlot(control, handle) orelse {
                control.directory.unlock();
                return false;
            };
            const node = switch (slot.*) {
                .published => |published| published,
                else => {
                    control.directory.unlock();
                    return false;
                },
            };
            node.setState(.closing);
            slot.* = .{ .retired = node };
            control.published -= 1;
            control.retired += 1;
            control.directory.unlock();

            if (comptime enabled) {
                if (node.refs.release()) finalizeNode(node);
            } else {
                finalizeNode(node);
            }
            return true;
        }

        /// Stops publication and acquisition, and retires every published
        /// node. This never waits and never invokes a destructor under a lock.
        pub fn shutdown(self: *Self) void {
            const control = self.control.?;
            control.directory.lock();
            control.shutting_down = true;
            control.directory.unlock();

            while (true) {
                control.directory.lock();
                const node = retireOnePublished(control);
                control.directory.unlock();
                const retired_node = node orelse break;
                if (comptime enabled) {
                    if (retired_node.refs.release()) finalizeNode(retired_node);
                } else {
                    finalizeNode(retired_node);
                }
            }
        }

        pub fn isQuiescent(self: *Self) bool {
            return self.leakCount() == 0;
        }

        /// Number of nodes that would leak if the table storage disappeared.
        pub fn leakCount(self: *Self) usize {
            const control = self.control.?;
            control.directory.lock();
            defer control.directory.unlock();
            return control.live_nodes;
        }

        /// Number of unpublished nodes still awaiting destruction.
        pub fn retiredNodeCount(self: *Self) usize {
            const control = self.control.?;
            control.directory.lock();
            defer control.directory.unlock();
            return control.retired;
        }

        pub fn stats(self: *Self) TableStats {
            const control = self.control.?;
            control.directory.lock();
            defer control.directory.unlock();
            return .{
                .published = control.published,
                .retired = control.retired,
                .live_nodes = control.live_nodes,
                .chunks = control.chunk_count,
                .shutting_down = control.shutting_down,
            };
        }

        /// Return a point-in-time copy of all currently-published handles.
        /// Allocation and caller work happen outside the directory lock.
        pub fn snapshotHandles(
            self: *Self,
            allocator: std.mem.Allocator,
        ) ![]Handle {
            const control = self.control.?;
            while (true) {
                control.directory.lock();
                const capacity = control.published;
                control.directory.unlock();

                const handles = try allocator.alloc(Handle, capacity);
                var filled: usize = 0;
                control.directory.lock();
                if (control.published > handles.len) {
                    control.directory.unlock();
                    allocator.free(handles);
                    continue;
                }
                var chunk = control.head;
                while (chunk) |current| : (chunk = current.next) {
                    for (current.slots[0..current.used]) |slot| {
                        switch (slot) {
                            .published => |node| {
                                handles[filled] = node.handle;
                                filled += 1;
                            },
                            else => {},
                        }
                    }
                }
                control.directory.unlock();
                if (filled == handles.len) return handles;
                return try allocator.realloc(handles, filled);
            }
        }

        /// Shuts down and frees an already-quiescent table. Outstanding
        /// enabled-build leases leave the table intact and return an error.
        pub fn deinit(self: *Self) !void {
            const control = self.control orelse return;
            self.shutdown();
            if (!self.isQuiescent()) return error.LeasesOutstanding;

            assertNoLocksHeldFor(enabled);
            var chunk = control.head;
            while (chunk) |current| {
                const next = current.next;
                control.allocator.destroy(current);
                chunk = next;
            }
            const allocator = control.allocator;
            allocator.destroy(control);
            self.control = null;
        }

        fn recordPublish(control: *Control) void {
            control.published += 1;
            control.live_nodes += 1;
        }

        fn findSlot(control: *Control, handle: Handle) ?*Slot {
            var chunk = control.head;
            while (chunk) |current| : (chunk = current.next) {
                if (handle < current.base) return null;
                const offset = handle - current.base;
                if (offset < current.used) {
                    return &current.slots[@intCast(offset)];
                }
            }
            return null;
        }

        fn retireOnePublished(control: *Control) ?*Node {
            var chunk = control.head;
            while (chunk) |current| : (chunk = current.next) {
                for (current.slots[0..current.used]) |*slot| {
                    switch (slot.*) {
                        .published => |node| {
                            node.setState(.closing);
                            slot.* = .{ .retired = node };
                            control.published -= 1;
                            control.retired += 1;
                            return node;
                        },
                        else => {},
                    }
                }
            }
            return null;
        }

        fn finalizeNode(node: *Node) void {
            const control = node.owner;
            const allocator = control.allocator;
            assertNoLocksHeldFor(enabled);
            node.setState(.destroying);
            destroy(control.context, &node.value);

            control.directory.lock();
            const slot = findSlot(control, node.handle).?;
            switch (slot.*) {
                .retired => |retired| std.debug.assert(retired == node),
                else => unreachable,
            }
            slot.* = .{ .free = control.free_head };
            control.free_head = node.handle;
            control.retired -= 1;
            control.live_nodes -= 1;
            control.directory.unlock();

            allocator.destroy(node);
        }
    };
}

comptime {
    if (@sizeOf(ConditionalMutexFor(false, LockRank.resource_directory)) != 0) {
        @compileError("disabled ConditionalMutex must have zero size");
    }
    if (@sizeOf(ConditionalRefCountFor(false)) != 0) {
        @compileError("disabled ConditionalRefCount must have zero size");
    }
    if (@sizeOf(ConditionalOperationMutexFor(false)) != 0) {
        @compileError("disabled ConditionalOperationMutex must have zero size");
    }
}

const TestResource = struct {
    value: usize,
};

const TestDestroyContext = *std.atomic.Value(usize);

fn countTestDestroy(context: TestDestroyContext, _: *TestResource) void {
    _ = context.fetchAdd(1, .monotonic);
}

fn TestTable(comptime enabled: bool, comptime capacity: usize) type {
    return StableHandleTableFor(
        enabled,
        TestResource,
        TestDestroyContext,
        countTestDestroy,
        capacity,
    );
}

test "disabled conditional synchronization compiles to zero-sized no-ops" {
    try std.testing.expectEqual(
        @as(usize, 0),
        @sizeOf(ConditionalMutexFor(false, LockRank.resource_directory)),
    );
    try std.testing.expectEqual(@as(usize, 0), @sizeOf(ConditionalRefCountFor(false)));

    var mutex: ConditionalMutexFor(false, LockRank.resource_directory) = .init;
    try std.testing.expect(mutex.tryLock());
    mutex.unlock();
    mutex.lock();
    mutex.unlock();

    var refs = ConditionalRefCountFor(false).init(99);
    refs.retain();
    try std.testing.expectEqual(@as(usize, 1), refs.count());
    try std.testing.expect(refs.release());
    try std.testing.expect(debugNoLocksHeldFor(false));
}

test "configured aliases follow lib_wasi_threads" {
    const ConfiguredMutex = ConditionalMutex(LockRank.resource_directory);
    const expected_mutex_size = if (config.lib_wasi_threads)
        @sizeOf(ConditionalMutexFor(true, LockRank.resource_directory))
    else
        0;
    const expected_ref_size = if (config.lib_wasi_threads)
        @sizeOf(ConditionalRefCountFor(true))
    else
        0;
    try std.testing.expectEqual(expected_mutex_size, @sizeOf(ConfiguredMutex));
    try std.testing.expectEqual(expected_ref_size, @sizeOf(ConditionalRefCount));
}

test "conditional lifetime refcount preserves disabled ownership accounting" {
    try std.testing.expectEqual(@sizeOf(u32), @sizeOf(ConditionalLifetimeRefCountFor(false)));
    try std.testing.expectEqual(@sizeOf(u32), @sizeOf(ConditionalLifetimeRefCountFor(true)));

    var disabled = ConditionalLifetimeRefCountFor(false).init(1);
    disabled.retain();
    try std.testing.expectEqual(@as(usize, 2), disabled.count());
    try std.testing.expect(!disabled.release());
    try std.testing.expect(disabled.release());

    var enabled = ConditionalLifetimeRefCountFor(true).init(1);
    enabled.retain();
    try std.testing.expectEqual(@as(usize, 2), enabled.count());
    try std.testing.expect(!enabled.release());
    try std.testing.expect(enabled.release());
}

test "stable table grows without moving a leased node" {
    const Table = TestTable(true, 2);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);

    const first = try table.publish(.{ .value = 41 });
    var lease = table.acquire(first).?;
    const original_address = lease.value();

    var handles: [9]Handle = undefined;
    for (&handles, 0..) |*handle, i| {
        handle.* = try table.publish(.{ .value = i + 100 });
    }
    try std.testing.expectEqual(original_address, lease.value());
    try std.testing.expectEqual(@as(usize, 41), lease.value().value);

    table.shutdown();
    try std.testing.expectEqual(@as(usize, 1), table.retiredNodeCount());
    lease.release();
    try std.testing.expect(table.isQuiescent());
    try std.testing.expectEqual(@as(usize, 10), destroyed.load(.monotonic));
    try table.deinit();
}

test "remove racing a lease leaves its node usable and closing" {
    const Table = TestTable(true, 4);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);
    const handle = try table.publish(.{ .value = 73 });
    var lease = table.acquire(handle).?;
    var removed = std.atomic.Value(bool).init(false);

    const Runner = struct {
        fn run(target: *Table, target_handle: Handle, result: *std.atomic.Value(bool)) void {
            result.store(target.remove(target_handle), .release);
        }
    };
    const thread = try std.Thread.spawn(.{}, Runner.run, .{ &table, handle, &removed });
    thread.join();

    try std.testing.expect(removed.load(.acquire));
    try std.testing.expect(table.acquire(handle) == null);
    try std.testing.expect(lease.isClosing());
    try std.testing.expectEqual(@as(usize, 73), lease.value().value);
    try std.testing.expectEqual(@as(usize, 0), destroyed.load(.monotonic));
    lease.release();
    try std.testing.expectEqual(@as(usize, 1), destroyed.load(.monotonic));
    try table.deinit();
}

test "retired handles are reused only after final release" {
    const Table = TestTable(true, 4);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);

    const first = try table.publish(.{ .value = 1 });
    var lease = table.acquire(first).?;
    try std.testing.expect(table.remove(first));
    const second = try table.publish(.{ .value = 2 });
    try std.testing.expect(first != second);

    lease.release();
    const reused = try table.publish(.{ .value = 3 });
    try std.testing.expectEqual(first, reused);

    table.shutdown();
    try std.testing.expectEqual(@as(usize, 3), destroyed.load(.monotonic));
    try table.deinit();
}

test "concurrent final releases run the destructor exactly once" {
    const Table = TestTable(true, 4);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);
    const handle = try table.publish(.{ .value = 5 });
    var first = table.acquire(handle).?;
    var second = table.acquire(handle).?;
    try std.testing.expect(table.remove(handle));

    const Runner = struct {
        fn run(lease: *Table.Lease) void {
            lease.release();
        }
    };
    const first_thread = try std.Thread.spawn(.{}, Runner.run, .{&first});
    const second_thread = try std.Thread.spawn(.{}, Runner.run, .{&second});
    first_thread.join();
    second_thread.join();

    try std.testing.expectEqual(@as(usize, 1), destroyed.load(.monotonic));
    try std.testing.expect(table.isQuiescent());
    try table.deinit();
}

test "shutdown reports outstanding leases and reaches quiescence" {
    const Table = TestTable(true, 4);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);
    const leased_handle = try table.publish(.{ .value = 10 });
    _ = try table.publish(.{ .value = 20 });
    var lease = table.acquire(leased_handle).?;

    table.shutdown();
    const stats = table.stats();
    try std.testing.expect(stats.shutting_down);
    try std.testing.expectEqual(@as(usize, 0), stats.published);
    try std.testing.expectEqual(@as(usize, 1), stats.retired);
    try std.testing.expectEqual(@as(usize, 1), stats.live_nodes);
    try std.testing.expectError(error.TableShuttingDown, table.publish(.{ .value = 30 }));
    try std.testing.expectError(error.LeasesOutstanding, table.deinit());

    lease.release();
    try std.testing.expect(table.isQuiescent());
    try std.testing.expectEqual(@as(usize, 2), destroyed.load(.monotonic));
    try table.deinit();
}

test "publish OOM rolls back node and directory growth deterministically" {
    const Table = TestTable(true, 4);
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{
        // Control and node allocations succeed; first directory chunk fails.
        .fail_index = 2,
    });
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(failing.allocator(), &destroyed);

    try std.testing.expectError(error.OutOfMemory, table.publish(.{ .value = 99 }));
    const stats = table.stats();
    try std.testing.expectEqual(@as(usize, 0), stats.published);
    try std.testing.expectEqual(@as(usize, 0), stats.retired);
    try std.testing.expectEqual(@as(usize, 0), stats.live_nodes);
    try std.testing.expectEqual(@as(usize, 0), stats.chunks);
    try std.testing.expectEqual(@as(usize, 0), destroyed.load(.monotonic));
    try table.deinit();
}

test "withdraw returns unpublished ownership without running the destructor" {
    const Table = TestTable(true, 2);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);

    const handle = try table.publish(.{ .value = 123 });
    const value = table.withdraw(handle).?;
    try std.testing.expectEqual(@as(usize, 123), value.value);
    try std.testing.expectEqual(@as(usize, 0), destroyed.load(.monotonic));
    try std.testing.expect(table.acquire(handle) == null);

    const reused = try table.publish(.{ .value = 456 });
    try std.testing.expectEqual(handle, reused);
    table.shutdown();
    try std.testing.expectEqual(@as(usize, 1), destroyed.load(.monotonic));
    try table.deinit();
}

test "Debug lock checks reject inversion and track no-lock-held regions" {
    if (!debug_lock_tracking) return error.SkipZigTest;

    const Outer = ConditionalMutexFor(true, LockRank.resource_directory);
    const Inner = ConditionalMutexFor(true, LockRank.resource_node);
    var outer: Outer = .init;
    var inner: Inner = .init;

    outer.lock();
    try std.testing.expect(!debugNoLocksHeldFor(true));
    try std.testing.expect(debugCanAcquireFor(true, LockRank.resource_node));
    inner.lock();
    try std.testing.expect(!debugCanAcquireFor(true, LockRank.resource_directory));
    try std.testing.expect(!debugCanAcquireFor(true, LockRank.resource_node));
    inner.unlock();
    outer.unlock();
    try std.testing.expect(debugNoLocksHeldFor(true));
}

test "disabled table keeps ordinary scoped operations and handle reuse" {
    const Table = TestTable(false, 2);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);

    const first = try table.publish(.{ .value = 7 });
    var lease = table.acquire(first).?;
    try std.testing.expectEqual(@as(usize, 7), lease.value().value);
    lease.release();
    try std.testing.expect(table.remove(first));
    const reused = try table.publish(.{ .value = 8 });
    try std.testing.expectEqual(first, reused);

    table.shutdown();
    try std.testing.expectEqual(@as(usize, 2), destroyed.load(.monotonic));
    try table.deinit();
}
