//! Conditional synchronization and stable lease-based handle tables.
//!
//! The configured aliases erase synchronization and atomic reference-count
//! storage when WASI threads are disabled. Generational tables then use scoped
//! borrows, while keyed tables defer callback-facing node destruction across
//! reentrant removal without touching an atomic or per-lookup reference count.
//! Thread-enabled tables retain retired nodes until the final lease is
//! released. Reused directory slots advance a generation embedded in the
//! public handle, preventing ABA.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");

pub const Handle = u32;

/// Stable handles reserve the high byte for a generation and the low
/// 24 bits for the directory slot. A slot is retired permanently instead of
/// wrapping its generation, so an old handle can never alias a later occupant.
pub const handle_index_bits = 24;
pub const handle_generation_bits = @bitSizeOf(Handle) - handle_index_bits;
pub const max_handle_index: Handle = (1 << handle_index_bits) - 1;
pub const HandleGeneration = std.meta.Int(.unsigned, handle_generation_bits);

pub fn handleIndex(handle: Handle) Handle {
    return handle & max_handle_index;
}

pub fn handleGeneration(handle: Handle) HandleGeneration {
    return @truncate(handle >> handle_index_bits);
}

pub fn makeHandle(index: Handle, generation: HandleGeneration) Handle {
    std.debug.assert(index <= max_handle_index);
    return index | (@as(Handle, generation) << handle_index_bits);
}

/// Lock ranks increase from outer to inner locks.
pub const LockRank = struct {
    pub const resource_registry: u16 = 50;
    pub const resource_directory: u16 = 100;
    pub const resource_node: u16 = 200;
    pub const waitable_set: u16 = 225;
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
    return StableHandleTableForStart(
        enabled,
        T,
        Context,
        destroy,
        chunk_capacity,
        0,
    );
}

/// Stable table variant whose first generation-zero slot has `first_index`.
/// Component async handles use `1` because zero is their ABI sentinel.
pub fn StableHandleTableForStart(
    comptime enabled: bool,
    comptime T: type,
    comptime Context: type,
    comptime destroy: fn (Context, *T) void,
    comptime chunk_capacity: usize,
    comptime first_index: Handle,
) type {
    comptime {
        if (chunk_capacity == 0) @compileError("chunk_capacity must be non-zero");
        if (chunk_capacity > max_handle_index + 1) {
            @compileError("chunk_capacity does not fit in a handle");
        }
        if (first_index > max_handle_index) @compileError("first_index does not fit in a handle");
    }

    return struct {
        const Self = @This();
        const RefCount = ConditionalRefCountFor(enabled);
        const DirectoryMutex = ConditionalMutexFor(enabled, LockRank.resource_directory);

        const Node = struct {
            owner: *Control,
            handle: Handle = 0,
            slot_index: Handle = 0,
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
            free: struct {
                next: ?Handle,
                generation: HandleGeneration,
            },
            exhausted,
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
                    const slot = findSlotByIndex(control, handle).?;
                    const next = switch (slot.*) {
                        .free => |free_slot| free_slot.next,
                        else => unreachable,
                    };
                    const generation = slot.free.generation;
                    control.free_head = next;
                    node.slot_index = handle;
                    node.handle = makeHandle(handle, generation);
                    slot.* = .{ .published = node };
                    recordPublish(control);
                    control.directory.unlock();
                    committed = true;
                    return node.handle;
                }

                if (control.tail) |tail| {
                    if (tail.used < chunk_capacity) {
                        const offset = tail.used;
                        const slot_index: Handle = tail.base + @as(Handle, @intCast(offset));
                        tail.used += 1;
                        tail.slots[offset] = .{ .published = node };
                        node.slot_index = slot_index;
                        node.handle = makeHandle(slot_index, 0);
                        recordPublish(control);
                        control.directory.unlock();
                        committed = true;
                        return node.handle;
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
                        first_index;
                    if (base > max_handle_index - (chunk_capacity - 1)) {
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
                    node.slot_index = base;
                    node.handle = makeHandle(base, 0);
                    recordPublish(control);
                    control.directory.unlock();
                    committed = true;
                    return node.handle;
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
                    if (next_base > max_handle_index - (chunk_capacity - 1)) {
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
            if (node.handle != handle) return null;
            if (comptime enabled) node.refs.retain();
            return .{ .node = node };
        }

        /// Test-only escape hatch for legacy single-threaded fixtures.
        /// Production code must use `acquire`; this pointer carries no lease.
        pub fn unsafeValuePtrForTesting(self: *Self, handle: Handle) ?*T {
            if (!builtin.is_test) @compileError("use acquire() outside tests");
            const control = self.control.?;
            control.directory.lock();
            defer control.directory.unlock();
            if (control.shutting_down) return null;
            const slot = findSlot(control, handle) orelse return null;
            const node = switch (slot.*) {
                .published => |published| published,
                else => return null,
            };
            if (node.handle != handle) return null;
            return &node.value;
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
            if (node.handle != handle) {
                control.directory.unlock();
                return null;
            }
            if (comptime enabled) {
                if (node.refs.count() != 1) {
                    control.directory.unlock();
                    return null;
                }
            }

            node.setState(.destroying);
            recycleSlot(control, slot, node);
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
            if (node.handle != handle) {
                control.directory.unlock();
                return false;
            }
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

        /// Return a point-in-time copy of all currently published handles.
        /// No table-owned pointers escape the directory lock.
        pub fn snapshotHandles(self: *Self, allocator: std.mem.Allocator) ![]Handle {
            const control = self.control.?;
            control.directory.lock();
            defer control.directory.unlock();
            if (control.shutting_down) return allocator.alloc(Handle, 0);

            const handles = try allocator.alloc(Handle, control.published);
            var len: usize = 0;
            var chunk = control.head;
            while (chunk) |current| : (chunk = current.next) {
                for (current.slots[0..current.used]) |slot| {
                    switch (slot) {
                        .published => |node| {
                            handles[len] = node.handle;
                            len += 1;
                        },
                        else => {},
                    }
                }
            }
            std.debug.assert(len == handles.len);
            return handles;
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
            return findSlotByIndex(control, handleIndex(handle));
        }

        fn findSlotByIndex(control: *Control, index: Handle) ?*Slot {
            var chunk = control.head;
            while (chunk) |current| : (chunk = current.next) {
                if (index < current.base) return null;
                const offset = index - current.base;
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
            const slot = findSlotByIndex(control, node.slot_index).?;
            switch (slot.*) {
                .retired => |retired| std.debug.assert(retired == node),
                else => unreachable,
            }
            recycleSlot(control, slot, node);
            control.retired -= 1;
            control.live_nodes -= 1;
            control.directory.unlock();

            allocator.destroy(node);
        }

        fn recycleSlot(control: *Control, slot: *Slot, node: *Node) void {
            const generation = handleGeneration(node.handle);
            if (generation == std.math.maxInt(HandleGeneration)) {
                slot.* = .exhausted;
                return;
            }
            slot.* = .{ .free = .{
                .next = control.free_head,
                .generation = generation + 1,
            } };
            control.free_head = node.slot_index;
        }
    };
}

/// A conditionally-stable map from caller-owned public handles to values.
///
/// Enabled builds keep map values in the shared stable-node directory.
/// Disabled builds use heap-stable plain nodes and a value-specific busy
/// predicate. Ordinary leases remain scoped borrows; busy nodes defer
/// same-handle destruction until the callback clears its in-flight marker,
/// without a per-lookup reference update.
///
/// `publishAt` takes ownership only on success. Removal first unlinks the
/// public handle, then runs `destroy` outside both directory and value locks.
pub fn StableKeyedHandleTableFor(
    comptime enabled: bool,
    comptime T: type,
    comptime Context: type,
    comptime destroy: fn (Context, *T) void,
    comptime lock_values: bool,
    comptime defer_destroy: ?fn (*const T) bool,
) type {
    return struct {
        const Self = @This();
        const DirectoryMutex = ConditionalMutexFor(enabled, LockRank.resource_registry);
        const ValueMutex = ConditionalMutexFor(enabled and lock_values, LockRank.resource_node);

        const Stored = struct {
            owner: if (enabled) void else *Self = if (enabled) {} else undefined,
            mutex: ValueMutex = .init,
            closing: if (enabled) void else bool = if (enabled) {} else false,
            retired_next: if (enabled) void else ?*Stored =
                if (enabled) {} else null,
            value: T,
        };

        const Destroyer = struct {
            fn run(context: Context, stored: *Stored) void {
                assertNoLocksHeldFor(enabled);
                destroy(context, &stored.value);
            }
        };

        const Resources = StableHandleTableFor(
            enabled,
            Stored,
            Context,
            Destroyer.run,
            64,
        );
        const DirectoryValue = if (enabled) Handle else *Stored;

        pub const Lease = struct {
            storage: if (enabled) Resources.Lease else ?*Stored,
            locked: bool = true,
            active: bool = true,

            pub fn value(self: *Lease) *T {
                std.debug.assert(self.active and self.locked);
                return &self.stored().value;
            }

            pub fn isClosing(self: *const Lease) bool {
                std.debug.assert(self.active);
                if (comptime enabled) return self.storage.isClosing();
                return self.storage.?.closing;
            }

            pub fn unlock(self: *Lease) void {
                std.debug.assert(self.active and self.locked);
                self.stored().mutex.unlock();
                self.locked = false;
            }

            pub fn lock(self: *Lease) void {
                std.debug.assert(self.active and !self.locked);
                self.stored().mutex.lock();
                self.locked = true;
            }

            pub fn release(self: *Lease) void {
                if (!self.active) return;
                if (self.locked) self.unlock();
                self.active = false;
                if (comptime enabled) {
                    self.storage.release();
                } else {
                    self.storage = null;
                }
            }

            pub fn deinit(self: *Lease) void {
                self.release();
            }

            fn stored(self: *Lease) *Stored {
                if (comptime enabled) return self.storage.value();
                return self.storage.?;
            }
        };

        allocator: std.mem.Allocator,
        context: Context,
        directory: DirectoryMutex = .init,
        entries: std.AutoHashMapUnmanaged(Handle, DirectoryValue) = .empty,
        resources: if (enabled) Resources else void,
        shutting_down: bool = false,
        live_nodes: if (enabled) void else usize = if (enabled) {} else 0,
        retired_head: if (enabled) void else ?*Stored =
            if (enabled) {} else null,

        pub fn init(allocator: std.mem.Allocator, context: Context) !Self {
            return .{
                .allocator = allocator,
                .context = context,
                .resources = if (enabled) try Resources.init(allocator, context) else {},
            };
        }

        pub fn publishAt(self: *Self, public_handle: Handle, value: T) !void {
            assertNoLocksHeldFor(enabled);
            const resource_handle = if (comptime enabled)
                try self.resources.publish(.{ .value = value })
            else {};
            const disabled_node = if (comptime !enabled) blk: {
                const node = try self.allocator.create(Stored);
                node.* = .{
                    .owner = self,
                    .value = value,
                };
                break :blk node;
            } else {};
            var disabled_committed = false;
            defer {
                if (comptime !enabled) {
                    if (!disabled_committed) self.allocator.destroy(disabled_node);
                }
            }

            self.directory.lock();
            if (self.shutting_down) {
                self.directory.unlock();
                if (comptime enabled) {
                    std.debug.assert(self.resources.withdraw(resource_handle) != null);
                }
                return error.TableShuttingDown;
            }
            if (self.entries.contains(public_handle)) {
                self.directory.unlock();
                if (comptime enabled) {
                    std.debug.assert(self.resources.withdraw(resource_handle) != null);
                }
                return error.HandleAlreadyExists;
            }

            if (comptime enabled) {
                self.entries.put(
                    self.allocator,
                    public_handle,
                    resource_handle,
                ) catch |err| {
                    self.directory.unlock();
                    std.debug.assert(self.resources.withdraw(resource_handle) != null);
                    return err;
                };
            } else {
                self.entries.put(
                    self.allocator,
                    public_handle,
                    disabled_node,
                ) catch |err| {
                    self.directory.unlock();
                    return err;
                };
                self.live_nodes += 1;
                disabled_committed = true;
            }
            self.directory.unlock();
        }

        pub fn put(
            self: *Self,
            allocator: std.mem.Allocator,
            public_handle: Handle,
            value: T,
        ) !void {
            _ = allocator;
            return self.publishAt(public_handle, value);
        }

        pub fn acquire(self: *Self, public_handle: Handle) ?Lease {
            self.directory.lock();
            if (self.shutting_down) {
                self.directory.unlock();
                return null;
            }
            if (comptime enabled) {
                const resource_handle = self.entries.get(public_handle) orelse {
                    self.directory.unlock();
                    return null;
                };
                const resource_lease = self.resources.acquire(resource_handle) orelse {
                    self.directory.unlock();
                    return null;
                };
                self.directory.unlock();
                var lease = Lease{ .storage = resource_lease };
                lease.stored().mutex.lock();
                return lease;
            }
            const stored = self.entries.get(public_handle) orelse {
                self.directory.unlock();
                return null;
            };
            std.debug.assert(!stored.closing);
            self.directory.unlock();
            stored.mutex.lock();
            return .{ .storage = stored };
        }

        /// Test-only compatibility for old single-threaded fixtures. New
        /// concurrency coverage must use leases so removal cannot race use.
        pub fn getPtr(self: *Self, public_handle: Handle) ?*T {
            if (!builtin.is_test) @compileError("use acquire() outside tests");
            self.directory.lock();
            defer self.directory.unlock();
            if (self.shutting_down) return null;
            if (comptime enabled) {
                const resource_handle = self.entries.get(public_handle) orelse
                    return null;
                const stored = self.resources.unsafeValuePtrForTesting(
                    resource_handle,
                ) orelse return null;
                return &stored.value;
            }
            const stored = self.entries.get(public_handle) orelse return null;
            return &stored.value;
        }

        pub fn get(self: *Self, public_handle: Handle) ?T {
            if (!builtin.is_test) @compileError("use acquire() outside tests");
            const value = self.getPtr(public_handle) orelse return null;
            return value.*;
        }

        pub fn remove(self: *Self, public_handle: Handle) bool {
            self.directory.lock();
            const removed = if (self.shutting_down)
                null
            else
                self.entries.fetchRemove(public_handle);
            self.directory.unlock();

            const entry = removed orelse return false;
            if (comptime enabled) {
                std.debug.assert(self.resources.remove(entry.value));
            } else {
                const stored = entry.value;
                stored.closing = true;
                self.retireDisabled(stored);
                self.collectRetired();
            }
            return true;
        }

        pub fn count(self: *Self) usize {
            self.directory.lock();
            defer self.directory.unlock();
            return self.entries.count();
        }

        pub fn snapshotHandles(
            self: *Self,
            allocator: std.mem.Allocator,
        ) ![]Handle {
            self.directory.lock();
            defer self.directory.unlock();
            if (self.shutting_down) return allocator.alloc(Handle, 0);

            const handles = try allocator.alloc(Handle, self.entries.count());
            var iterator = self.entries.keyIterator();
            var len: usize = 0;
            while (iterator.next()) |handle| {
                handles[len] = handle.*;
                len += 1;
            }
            std.debug.assert(len == handles.len);
            return handles;
        }

        pub fn shutdown(self: *Self) void {
            self.directory.lock();
            if (self.shutting_down) {
                self.directory.unlock();
                return;
            }
            self.shutting_down = true;
            var detached = self.entries;
            self.entries = .empty;
            self.directory.unlock();

            if (comptime enabled) {
                detached.deinit(self.allocator);
                self.resources.shutdown();
            } else {
                var values = detached.valueIterator();
                while (values.next()) |stored_ptr| {
                    const stored = stored_ptr.*;
                    stored.closing = true;
                    self.retireDisabled(stored);
                }
                detached.deinit(self.allocator);
                self.collectRetired();
            }
        }

        /// Finalize disabled-build nodes whose value-specific busy predicate
        /// is no longer active. Enabled builds retire through stable leases.
        pub fn collectRetired(self: *Self) void {
            if (comptime enabled) return;

            while (true) {
                self.directory.lock();
                var link = &self.retired_head;
                var ready: ?*Stored = null;
                while (link.*) |stored| {
                    if (!shouldDeferDestroy(stored)) {
                        link.* = stored.retired_next;
                        stored.retired_next = null;
                        ready = stored;
                        break;
                    }
                    link = &stored.retired_next;
                }
                self.directory.unlock();

                const stored = ready orelse return;
                finalizeDisabled(stored);
            }
        }

        pub fn isQuiescent(self: *Self) bool {
            if (comptime enabled) return self.resources.isQuiescent();
            return self.live_nodes == 0;
        }

        pub fn deinit(self: *Self) !void {
            self.shutdown();
            if (comptime enabled) {
                try self.resources.deinit();
            } else {
                self.collectRetired();
                if (!self.isQuiescent()) return error.LeasesOutstanding;
            }
        }

        fn finalizeDisabled(stored: *Stored) void {
            const owner = stored.owner;
            Destroyer.run(owner.context, stored);
            owner.live_nodes -= 1;
            owner.allocator.destroy(stored);
        }

        fn retireDisabled(self: *Self, stored: *Stored) void {
            stored.retired_next = self.retired_head;
            self.retired_head = stored;
        }

        fn shouldDeferDestroy(stored: *const Stored) bool {
            if (defer_destroy) |is_busy| return is_busy(&stored.value);
            return false;
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
}

const TestResource = struct {
    value: usize,
    callback_inflight: bool = false,
};

const TestDestroyContext = *std.atomic.Value(usize);

fn countTestDestroy(context: TestDestroyContext, _: *TestResource) void {
    _ = context.fetchAdd(1, .monotonic);
}

fn testCallbackInFlight(resource: *const TestResource) bool {
    return resource.callback_inflight;
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

fn TestKeyedTable(comptime enabled: bool) type {
    return StableKeyedHandleTableFor(
        enabled,
        TestResource,
        TestDestroyContext,
        countTestDestroy,
        true,
        testCallbackInFlight,
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

test "retired slots advance generation only after final release" {
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
    try std.testing.expectEqual(handleIndex(first), handleIndex(reused));
    try std.testing.expect(handleGeneration(reused) > handleGeneration(first));
    try std.testing.expect(table.acquire(first) == null);
    var reused_lease = table.acquire(reused).?;
    try std.testing.expectEqual(@as(usize, 3), reused_lease.value().value);
    reused_lease.release();

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
    try std.testing.expectEqual(handleIndex(handle), handleIndex(reused));
    try std.testing.expect(handleGeneration(reused) > handleGeneration(handle));
    try std.testing.expect(table.acquire(handle) == null);
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

test "disabled table keeps ordinary scoped operations and stale generations invalid" {
    const Table = TestTable(false, 2);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);

    const first = try table.publish(.{ .value = 7 });
    var lease = table.acquire(first).?;
    try std.testing.expectEqual(@as(usize, 7), lease.value().value);
    lease.release();
    try std.testing.expect(table.remove(first));
    const reused = try table.publish(.{ .value = 8 });
    try std.testing.expectEqual(handleIndex(first), handleIndex(reused));
    try std.testing.expect(handleGeneration(reused) > handleGeneration(first));
    try std.testing.expect(table.acquire(first) == null);

    table.shutdown();
    try std.testing.expectEqual(@as(usize, 2), destroyed.load(.monotonic));
    try table.deinit();
}

test "generation exhaustion retires a slot instead of wrapping ABA" {
    const Table = TestTable(true, 2);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);

    const stale = try table.publish(.{ .value = 0 });
    try std.testing.expect(table.remove(stale));
    var last = stale;
    var generation: usize = 1;
    while (generation <= std.math.maxInt(HandleGeneration)) : (generation += 1) {
        last = try table.publish(.{ .value = generation });
        try std.testing.expectEqual(handleIndex(stale), handleIndex(last));
        try std.testing.expectEqual(
            @as(HandleGeneration, @intCast(generation)),
            handleGeneration(last),
        );
        try std.testing.expect(table.remove(last));
    }

    const replacement = try table.publish(.{ .value = 999 });
    try std.testing.expect(handleIndex(replacement) != handleIndex(stale));
    try std.testing.expect(table.acquire(stale) == null);
    try std.testing.expect(table.acquire(last) == null);
    try std.testing.expect(table.remove(replacement));
    try std.testing.expectEqual(
        @as(usize, std.math.maxInt(HandleGeneration)) + 2,
        destroyed.load(.monotonic),
    );
    try table.deinit();
}

test "keyed stable table unlinks before exact-once out-of-lock destruction" {
    const Table = TestKeyedTable(true);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);
    try table.publishAt(77, .{ .value = 123 });
    var lease = table.acquire(77).?;

    const Runner = struct {
        fn run(target: *Table) void {
            std.debug.assert(target.remove(77));
        }
    };
    const thread = try std.Thread.spawn(.{}, Runner.run, .{&table});
    thread.join();

    lease.unlock();
    try std.testing.expect(table.acquire(77) == null);
    try std.testing.expect(lease.isClosing());
    lease.lock();
    try std.testing.expectEqual(@as(usize, 123), lease.value().value);
    try std.testing.expectEqual(@as(usize, 0), destroyed.load(.monotonic));
    lease.release();
    try std.testing.expectEqual(@as(usize, 1), destroyed.load(.monotonic));
    try table.deinit();
}

test "keyed stable table publication failure keeps caller ownership" {
    const Table = TestKeyedTable(true);
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{
        // Stable control, node, and first chunk succeed; public map growth fails.
        .fail_index = 3,
    });
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(failing.allocator(), &destroyed);

    try std.testing.expectError(
        error.OutOfMemory,
        table.publishAt(9, .{ .value = 44 }),
    );
    try std.testing.expectEqual(@as(usize, 0), table.count());
    try std.testing.expectEqual(@as(usize, 0), destroyed.load(.monotonic));
    try table.deinit();
}

test "disabled keyed table defers destruction across callback reentry" {
    const Table = TestKeyedTable(false);
    var destroyed = std.atomic.Value(usize).init(0);
    var table = try Table.init(std.testing.allocator, &destroyed);

    try table.publishAt(5, .{ .value = 8 });
    var lease = table.acquire(5).?;
    try std.testing.expectEqual(@as(usize, 8), lease.value().value);
    lease.value().callback_inflight = true;
    lease.unlock();
    try std.testing.expect(table.remove(5));
    try std.testing.expect(table.acquire(5) == null);
    try std.testing.expect(lease.isClosing());
    try std.testing.expectEqual(@as(usize, 0), destroyed.load(.monotonic));
    try std.testing.expectError(error.LeasesOutstanding, table.deinit());
    lease.lock();
    lease.value().callback_inflight = false;
    lease.release();
    table.collectRetired();
    try std.testing.expectEqual(@as(usize, 1), destroyed.load(.monotonic));
    try table.deinit();
}
