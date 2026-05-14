//! Component Model async ABI — tasks, futures, and streams.
//!
//! Implements the WASIp3 cooperative task model where component function
//! calls can be non-blocking. Each async call produces a subtask that
//! the caller can poll for completion via waitable sets.

const std = @import("std");

// ── Task state ──────────────────────────────────────────────────────────────

pub const TaskState = enum(u8) {
    /// Task has been created but not yet started.
    created = 0,
    /// Task is running (started but not yet returned).
    started = 1,
    /// Task has produced its return value.
    returned = 2,
    /// Task has been cancelled.
    cancelled = 3,
};

/// Per-task `context.{get,set} i32 i` slots. The spec allows arbitrary
/// `valtype` and arbitrary index range; sub-PR 1 of #478 caps both
/// (`i32` only, `slot < N_CONTEXT_SLOTS`). Wasmtime currently exposes a
/// single `i32` slot; pick a small headroom value here so we don't have
/// to revisit the constant on the first conformance suite that uses 2.
pub const N_CONTEXT_SLOTS: u32 = 2;

/// A task represents an in-flight async component function call.
pub const Task = struct {
    id: u32,
    state: TaskState = .created,
    /// Return value buffer (set when state transitions to .returned).
    return_values: []u32 = &.{},
    /// Waiters to notify when state changes.
    waitable_set: ?*WaitableSet = null,
    /// Per-task `context.{get,set} i32` slots. Default-initialised to 0
    /// to match Wasmtime's behaviour for a freshly-started task.
    context_slots: [N_CONTEXT_SLOTS]u32 = [_]u32{0} ** N_CONTEXT_SLOTS,
};

// ── Waitable Set ────────────────────────────────────────────────────────────

/// A waitable set multiplexes readiness notifications across subtasks,
/// streams, and futures. Callers use wait/poll to discover which
/// registered items are ready.
pub const WaitableSet = struct {
    /// Registered items that can become ready.
    items: std.ArrayListUnmanaged(WaitableItem) = .empty,
    /// Items that have become ready since the last wait/poll.
    ready_queue: std.ArrayListUnmanaged(u32) = .empty, // indices into items

    pub const WaitableItem = struct {
        kind: Kind,
        handle: u32, // task/stream/future handle
        ready: bool = false,

        pub const Kind = enum { subtask, stream_read, stream_write, future_read, future_write };
    };

    /// Register an item for waiting.
    pub fn register(self: *WaitableSet, item: WaitableItem, allocator: std.mem.Allocator) !u32 {
        const idx: u32 = @intCast(self.items.items.len);
        try self.items.append(allocator, item);
        return idx;
    }

    /// Mark an item as ready (called by the runtime when a subtask completes, etc.).
    pub fn setReady(self: *WaitableSet, idx: u32, allocator: std.mem.Allocator) void {
        if (idx < self.items.items.len) {
            self.items.items[idx].ready = true;
            self.ready_queue.append(allocator, idx) catch {};
        }
    }

    /// Poll for ready items without blocking. Returns indices of ready items.
    pub fn pollReady(self: *WaitableSet, out: []u32) u32 {
        var count: u32 = 0;
        for (self.items.items, 0..) |*item, i| {
            if (item.ready) {
                if (count < out.len) {
                    out[count] = @intCast(i);
                    count += 1;
                }
                item.ready = false; // consume readiness
            }
        }
        return count;
    }

    pub fn deinit(self: *WaitableSet, allocator: std.mem.Allocator) void {
        self.items.deinit(allocator);
        self.ready_queue.deinit(allocator);
    }
};

// ── Task Manager ────────────────────────────────────────────────────────────

/// Manages the lifecycle of async tasks within a component instance.
pub const TaskManager = struct {
    tasks: std.ArrayListUnmanaged(Task) = .empty,
    next_id: u32 = 1,
    /// Handle of the task currently executing a core-wasm body, or `null`
    /// when no async task is on the dispatch stack. `context.{get,set}`
    /// and `task.yield` consult this to find their target task; when it's
    /// `null` (synchronous canon-lift call path), callers should fall
    /// back to `ComponentInstance.implicit_task`.
    current_task: ?u32 = null,

    /// Create a new task. Returns the task handle.
    pub fn createTask(self: *TaskManager, allocator: std.mem.Allocator) !u32 {
        const id = self.next_id;
        self.next_id += 1;
        const idx: u32 = @intCast(self.tasks.items.len);
        try self.tasks.append(allocator, .{ .id = id });
        return idx;
    }

    /// Transition a task to the started state.
    pub fn startTask(self: *TaskManager, handle: u32) void {
        if (handle < self.tasks.items.len) {
            self.tasks.items[handle].state = .started;
        }
    }

    /// Transition a task to the returned state with values.
    pub fn returnTask(self: *TaskManager, handle: u32, values: []u32) void {
        if (handle < self.tasks.items.len) {
            const task = &self.tasks.items[handle];
            task.state = .returned;
            task.return_values = values;
            // Notify waitable set
            if (task.waitable_set) |ws| {
                ws.setReady(handle, std.heap.page_allocator);
            }
        }
    }

    /// Cancel a task.
    pub fn cancelTask(self: *TaskManager, handle: u32) void {
        if (handle < self.tasks.items.len) {
            self.tasks.items[handle].state = .cancelled;
        }
    }

    /// Get the state of a task.
    pub fn getState(self: *const TaskManager, handle: u32) ?TaskState {
        if (handle >= self.tasks.items.len) return null;
        return self.tasks.items[handle].state;
    }

    /// Read a per-task `context.get i32 i` slot. Returns `null` if the
    /// task handle is unknown or the slot index is out of range.
    pub fn getContextSlot(self: *const TaskManager, handle: u32, slot: u32) ?u32 {
        if (handle >= self.tasks.items.len) return null;
        if (slot >= N_CONTEXT_SLOTS) return null;
        return self.tasks.items[handle].context_slots[slot];
    }

    /// Write a per-task `context.set i32 i` slot. Returns `false` if the
    /// task handle is unknown or the slot index is out of range; the
    /// caller surfaces this as a component trap.
    pub fn setContextSlot(self: *TaskManager, handle: u32, slot: u32, value: u32) bool {
        if (handle >= self.tasks.items.len) return false;
        if (slot >= N_CONTEXT_SLOTS) return false;
        self.tasks.items[handle].context_slots[slot] = value;
        return true;
    }

    pub fn deinit(self: *TaskManager, allocator: std.mem.Allocator) void {
        self.tasks.deinit(allocator);
    }
};

// ── Futures ─────────────────────────────────────────────────────────────────

/// A future represents a single async value (one-shot channel) parameterised
/// on element type `T`. Per the component-model spec, a future has two
/// ends — a readable side and a writable side — and a single value flows
/// from writer to reader (or the future is dropped before either occurs).
///
/// The host-side representation buffers the lowered bytes of T between the
/// producer `future.write` and consumer `future.read` rendezvous. If the
/// reader arrives first it parks, recording its destination guest pointer;
/// the next writer copies straight into that location instead of allocating
/// a buffer.
pub const Future = struct {
    /// Type index of `T`, captured from the originating `future.new t`'s
    /// `type_idx`. Used by `future.read` / `future.write` to compute the
    /// byte size of the payload via `canonical_abi.sizeOfType`.
    elem_type_idx: u32 = 0,
    /// Buffered payload bytes, set when a writer arrives before a reader.
    /// Owned by this `Future`; freed by the reader (on consumption) or by
    /// the `future_drop_*` arm once both ends are closed.
    payload: ?[]u8 = null,
    /// Set when a `future.read` arrives before any writer. The destination
    /// guest pointer is stashed so that a subsequent `future.write` can
    /// copy straight into the parked reader's memory and complete.
    pending_read: ?PendingRead = null,
    /// Waitable plumbing for `waitable.join` integration.
    waitable_set: ?*WaitableSet = null,
    read_waitable_idx: ?u32 = null,
    write_waitable_idx: ?u32 = null,
    state: State = .pending,
    /// Both ends are closed only after both `drop-readable` and
    /// `drop-writable`. Tracked separately so cancellation observes the
    /// correct end (see `dispatchAsyncCanon.future_read`/`future_write`).
    read_closed: bool = false,
    write_closed: bool = false,

    pub const State = enum { pending, ready, closed };

    pub const PendingRead = struct { guest_ptr: u32 };

    /// Free any heap-owned state (currently just the buffered `payload`).
    /// Safe to call multiple times.
    pub fn deinit(self: *Future, allocator: std.mem.Allocator) void {
        if (self.payload) |b| {
            allocator.free(b);
            self.payload = null;
        }
    }
};

// ── Streams (async multi-value channel) ─────────────────────────────────────

/// A component-level async stream — FIFO byte channel parameterised on
/// element type `T`. Mirrors the rendezvous-driven model used by `Future`
/// but with a multi-value buffer instead of a one-shot payload.
///
/// The buffer holds raw lowered bytes; element boundaries are computed at
/// each `stream.read` / `stream.write` op from
/// `canonical_abi.sizeOfType(elem_type_idx)`.
pub const AsyncStream = struct {
    /// Type index of `T`, captured from `stream.new t`'s `type_idx`. Used
    /// by `stream.read` / `stream.write` to compute the per-element byte
    /// size via `canonical_abi.sizeOfType`.
    elem_type_idx: u32 = 0,
    /// FIFO of raw lowered bytes. Element boundaries are recomputed at
    /// op time from `elem_size = sizeOfType(...)`.
    buffer: std.ArrayListUnmanaged(u8) = .empty,

    /// A reader that ran while the buffer was empty. Resolved by the next
    /// `write` (direct memcpy into the parked dst, no buffering).
    pending_read: ?PendingRead = null,
    /// A writer that ran with no parked reader and exhausted the buffer
    /// cap. The initial implementation has no cap, so this stays `null`;
    /// kept for the future high-water-mark backpressure PR.
    pending_write: ?PendingWrite = null,

    /// Waitable plumbing for `waitable.join` integration.
    waitable_set: ?*WaitableSet = null,
    read_waitable_idx: ?u32 = null,
    write_waitable_idx: ?u32 = null,

    state: State = .open,
    /// Both ends are closed only after both `drop-readable` and
    /// `drop-writable`. Tracked separately so cancellation observes the
    /// correct end.
    read_closed: bool = false,
    write_closed: bool = false,

    pub const State = enum { open, closed };
    pub const PendingRead = struct { guest_ptr: u32, max_count: u32 };
    pub const PendingWrite = struct { guest_ptr: u32, count: u32 };

    /// Free any heap-owned state (currently just the FIFO `buffer`).
    /// Safe to call multiple times.
    pub fn deinit(self: *AsyncStream, allocator: std.mem.Allocator) void {
        self.buffer.deinit(allocator);
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "TaskManager: create and lifecycle" {
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    defer tm.deinit(allocator);

    const h = try tm.createTask(allocator);
    try std.testing.expectEqual(TaskState.created, tm.getState(h).?);

    tm.startTask(h);
    try std.testing.expectEqual(TaskState.started, tm.getState(h).?);

    tm.returnTask(h, &.{});
    try std.testing.expectEqual(TaskState.returned, tm.getState(h).?);
}

test "TaskManager: cancel" {
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    defer tm.deinit(allocator);

    const h = try tm.createTask(allocator);
    tm.cancelTask(h);
    try std.testing.expectEqual(TaskState.cancelled, tm.getState(h).?);
}

test "TaskManager: context slot get/set round-trip" {
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    defer tm.deinit(allocator);

    const h = try tm.createTask(allocator);

    // Fresh task → all slots zero.
    try std.testing.expectEqual(@as(?u32, 0), tm.getContextSlot(h, 0));
    try std.testing.expectEqual(@as(?u32, 0), tm.getContextSlot(h, 1));

    try std.testing.expect(tm.setContextSlot(h, 0, 0xDEAD_BEEF));
    try std.testing.expect(tm.setContextSlot(h, 1, 42));
    try std.testing.expectEqual(@as(?u32, 0xDEAD_BEEF), tm.getContextSlot(h, 0));
    try std.testing.expectEqual(@as(?u32, 42), tm.getContextSlot(h, 1));
}

test "TaskManager: context slot bounds" {
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    defer tm.deinit(allocator);

    const h = try tm.createTask(allocator);

    // Out-of-range slot index.
    try std.testing.expect(tm.getContextSlot(h, N_CONTEXT_SLOTS) == null);
    try std.testing.expect(!tm.setContextSlot(h, N_CONTEXT_SLOTS, 1));

    // Unknown handle.
    try std.testing.expect(tm.getContextSlot(h + 999, 0) == null);
    try std.testing.expect(!tm.setContextSlot(h + 999, 0, 1));
}

test "TaskManager: context slots are per-task" {
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    defer tm.deinit(allocator);

    const a = try tm.createTask(allocator);
    const b = try tm.createTask(allocator);

    try std.testing.expect(tm.setContextSlot(a, 0, 10));
    try std.testing.expect(tm.setContextSlot(b, 0, 20));
    try std.testing.expectEqual(@as(?u32, 10), tm.getContextSlot(a, 0));
    try std.testing.expectEqual(@as(?u32, 20), tm.getContextSlot(b, 0));
}

test "WaitableSet: register and poll" {
    const allocator = std.testing.allocator;
    var ws = WaitableSet{};
    defer ws.deinit(allocator);

    const idx0 = try ws.register(.{ .kind = .subtask, .handle = 0 }, allocator);
    const idx1 = try ws.register(.{ .kind = .subtask, .handle = 1 }, allocator);
    _ = idx0;

    ws.setReady(idx1, allocator);
    var out: [4]u32 = undefined;
    const count = ws.pollReady(&out);
    try std.testing.expectEqual(@as(u32, 1), count);
    try std.testing.expectEqual(idx1, out[0]);
}

test "Future: state transitions through pending/ready" {
    var f = Future{};
    try std.testing.expectEqual(Future.State.pending, f.state);
    f.state = .ready;
    try std.testing.expectEqual(Future.State.ready, f.state);
}

test "Future: deinit frees buffered payload" {
    const allocator = std.testing.allocator;
    var f = Future{};
    f.payload = try allocator.alloc(u8, 8);
    f.deinit(allocator);
    try std.testing.expect(f.payload == null);
    // Second deinit is a no-op.
    f.deinit(allocator);
}

test "Future: pending_read records guest_ptr" {
    var f = Future{};
    f.pending_read = .{ .guest_ptr = 0x1234 };
    try std.testing.expectEqual(@as(u32, 0x1234), f.pending_read.?.guest_ptr);
}

test "AsyncStream: deinit frees buffer" {
    const allocator = std.testing.allocator;
    var s = AsyncStream{};
    try s.buffer.appendSlice(allocator, &[_]u8{ 1, 2, 3, 4 });
    s.deinit(allocator);
}

test "AsyncStream: pending_read records guest_ptr and max_count" {
    var s = AsyncStream{};
    s.pending_read = .{ .guest_ptr = 0x2000, .max_count = 7 };
    try std.testing.expectEqual(@as(u32, 0x2000), s.pending_read.?.guest_ptr);
    try std.testing.expectEqual(@as(u32, 7), s.pending_read.?.max_count);
}
