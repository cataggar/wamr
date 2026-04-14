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

/// A task represents an in-flight async component function call.
pub const Task = struct {
    id: u32,
    state: TaskState = .created,
    /// Return value buffer (set when state transitions to .returned).
    return_values: []u32 = &.{},
    /// Waiters to notify when state changes.
    waitable_set: ?*WaitableSet = null,
};

// ── Waitable Set ────────────────────────────────────────────────────────────

/// A waitable set multiplexes readiness notifications across subtasks,
/// streams, and futures. Callers use wait/poll to discover which
/// registered items are ready.
pub const WaitableSet = struct {
    /// Registered items that can become ready.
    items: std.ArrayListUnmanaged(WaitableItem) = .{},
    /// Items that have become ready since the last wait/poll.
    ready_queue: std.ArrayListUnmanaged(u32) = .{}, // indices into items

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
    tasks: std.ArrayListUnmanaged(Task) = .{},
    next_id: u32 = 1,

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

    pub fn deinit(self: *TaskManager, allocator: std.mem.Allocator) void {
        self.tasks.deinit(allocator);
    }
};

// ── Futures ─────────────────────────────────────────────────────────────────

/// A future represents a single async value (one-shot channel).
pub const Future = struct {
    state: State = .pending,
    value: ?u32 = null,
    waitable_set: ?*WaitableSet = null,
    waitable_idx: ?u32 = null,

    pub const State = enum { pending, ready, closed };

    /// Write a value to the future (producer side).
    pub fn write(self: *Future, val: u32, allocator: std.mem.Allocator) bool {
        if (self.state != .pending) return false;
        self.value = val;
        self.state = .ready;
        if (self.waitable_set) |ws| {
            if (self.waitable_idx) |idx| {
                ws.setReady(idx, allocator);
            }
        }
        return true;
    }

    /// Read the value from the future (consumer side).
    pub fn read(self: *Future) ?u32 {
        if (self.state == .ready) {
            self.state = .closed;
            return self.value;
        }
        return null;
    }
};

// ── Streams (async multi-value channel) ─────────────────────────────────────

/// A component-level async stream — multi-value channel with backpressure.
pub const AsyncStream = struct {
    buffer: std.ArrayListUnmanaged(u32) = .{},
    state: State = .open,
    waitable_set: ?*WaitableSet = null,
    read_waitable_idx: ?u32 = null,

    pub const State = enum { open, closed };

    /// Write values to the stream (producer side).
    pub fn writeValues(self: *AsyncStream, values: []const u32, allocator: std.mem.Allocator) !usize {
        if (self.state == .closed) return 0;
        try self.buffer.appendSlice(allocator, values);
        // Notify reader
        if (self.waitable_set) |ws| {
            if (self.read_waitable_idx) |idx| {
                ws.setReady(idx, allocator);
            }
        }
        return values.len;
    }

    /// Read values from the stream (consumer side).
    pub fn readValues(self: *AsyncStream, out: []u32) usize {
        const avail = @min(self.buffer.items.len, out.len);
        if (avail == 0) return 0;
        @memcpy(out[0..avail], self.buffer.items[0..avail]);
        // Remove consumed items from front
        std.mem.copyForwards(u32, self.buffer.items[0 .. self.buffer.items.len - avail], self.buffer.items[avail..]);
        self.buffer.items.len -= avail;
        return avail;
    }

    /// Close the stream.
    pub fn close(self: *AsyncStream) void {
        self.state = .closed;
    }

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

test "Future: write then read" {
    var f = Future{};
    try std.testing.expect(f.read() == null);

    try std.testing.expect(f.write(42, std.testing.allocator));
    try std.testing.expectEqual(@as(?u32, 42), f.read());
    // Second read returns null (closed)
    try std.testing.expect(f.read() == null);
}

test "Future: double write fails" {
    var f = Future{};
    try std.testing.expect(f.write(1, std.testing.allocator));
    try std.testing.expect(!f.write(2, std.testing.allocator));
}

test "AsyncStream: write and read" {
    const allocator = std.testing.allocator;
    var s = AsyncStream{};
    defer s.deinit(allocator);

    const written = try s.writeValues(&.{ 10, 20, 30 }, allocator);
    try std.testing.expectEqual(@as(usize, 3), written);

    var out: [2]u32 = undefined;
    const n = s.readValues(&out);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(u32, 10), out[0]);
    try std.testing.expectEqual(@as(u32, 20), out[1]);

    // Read remaining
    var out2: [4]u32 = undefined;
    const n2 = s.readValues(&out2);
    try std.testing.expectEqual(@as(usize, 1), n2);
    try std.testing.expectEqual(@as(u32, 30), out2[0]);
}

test "AsyncStream: close prevents write" {
    const allocator = std.testing.allocator;
    var s = AsyncStream{};
    defer s.deinit(allocator);

    s.close();
    const written = try s.writeValues(&.{1}, allocator);
    try std.testing.expectEqual(@as(usize, 0), written);
}
