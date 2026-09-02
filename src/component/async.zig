//! Component Model async ABI — tasks, futures, and streams.
//!
//! Implements the WASIp3 cooperative task model where component function
//! calls can be non-blocking. Each async call produces a subtask that
//! the caller can poll for completion via waitable sets.

const std = @import("std");
const config = @import("../config.zig");
const stable_resource = @import("../shared/stable_resource.zig");
const execution_context = @import("../runtime/common/execution_context.zig");
const task_cancellation = @import("../runtime/common/task_cancellation.zig");

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
pub const N_CONTEXT_SLOTS: u32 = execution_context.task_context_slot_count;

pub const WaitableRegistration = struct {
    set_handle: u32,
    item_index: u32,
};

/// A task represents an in-flight async component function call.
pub const Task = struct {
    id: u32,
    state: TaskState = .created,
    cancellation_source: *task_cancellation.Source,
    /// Return value buffer (set when state transitions to .returned).
    return_values: []u32 = &.{},
    return_values_allocator: ?std.mem.Allocator = null,
    /// Waiters to notify when state changes.
    waitable_set: ?*WaitableSet = null,
    waitable_idx: ?u32 = null,
    /// Per-task `context.{get,set} i32` slots. Default-initialised to 0
    /// to match Wasmtime's behaviour for a freshly-started task.
    context_slots: [N_CONTEXT_SLOTS]u32 = [_]u32{0} ** N_CONTEXT_SLOTS,
};

// ── Waitable Set ────────────────────────────────────────────────────────────

/// A waitable set multiplexes readiness notifications across subtasks,
/// streams, and futures. Callers use wait/poll to discover which
/// registered items are ready.
pub const WaitableSet = struct {
    mutex: stable_resource.ConditionalMutex(stable_resource.LockRank.waitable_set) = .init,
    /// Registered items that can become ready.
    items: std.ArrayListUnmanaged(WaitableItem) = .empty,
    /// FIFO of indices that have become ready since the last
    /// `popReadyEvent`. Used by `waitable-set.{wait,poll}` to surface
    /// the oldest pending event to the guest. The matching `ready`
    /// flag on the `WaitableItem` lets `register`/`setReady` avoid
    /// double-enqueueing the same item.
    ready_queue: std.ArrayListUnmanaged(u32) = .empty, // indices into items

    pub const WaitableItem = struct {
        kind: Kind,
        handle: u32, // task/stream/future handle
        active: bool = true,
        ready: bool = false,
        /// Packed status word delivered as the event payload2 when this
        /// item is surfaced by `waitable-set.{wait,poll}`. For
        /// stream/future events this is the `packStatus(...)` value
        /// the corresponding `{stream,future}.{read,write}` op would
        /// return when re-issued at the time the event was fired.
        /// For subtask events it carries the post-transition task
        /// state. Populated by `setReady`; stale across multiple
        /// readiness cycles only if the producer forgets to refresh.
        code: u32 = 0,

        pub const Kind = enum { subtask, stream_read, stream_write, future_read, future_write };
    };

    /// Register an item for waiting. Returns the per-set index of the
    /// new entry, which the caller stashes on the underlying
    /// stream/future entry so subsequent `setReady` calls can find
    /// the right slot.
    pub fn register(self: *WaitableSet, item: WaitableItem, allocator: std.mem.Allocator) !u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const idx: u32 = @intCast(self.items.items.len);
        try self.items.append(allocator, item);
        return idx;
    }

    /// Invalidate a registration without shifting later indices.
    pub fn unregister(self: *WaitableSet, idx: u32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (idx >= self.items.items.len) return;
        self.items.items[idx].active = false;
        self.items.items[idx].ready = false;
    }

    /// Mark an item as ready and enqueue it for the next
    /// `popReadyEvent` / `pollReady` consumer. `code` is delivered as
    /// the event payload2 (e.g. `packStatus(.completed, n)` for stream
    /// / future events; task state for subtask events).
    ///
    /// Idempotent for already-queued items — the `ready` flag prevents
    /// double-enqueueing — but always refreshes `code` to reflect the
    /// most recent event payload.
    pub fn setReady(
        self: *WaitableSet,
        idx: u32,
        allocator: std.mem.Allocator,
        code: u32,
    ) void {
        _ = self.trySetReady(idx, allocator, code);
    }

    /// `setReady` with explicit OOM reporting. On allocation failure the
    /// item remains unqueued and can be retried without a stale ready flag.
    pub fn trySetReady(
        self: *WaitableSet,
        idx: u32,
        allocator: std.mem.Allocator,
        code: u32,
    ) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (idx >= self.items.items.len) return false;
        const item = &self.items.items[idx];
        if (!item.active) return false;
        item.code = code;
        if (!item.ready) {
            self.ready_queue.append(allocator, idx) catch return false;
            item.ready = true;
        }
        return true;
    }

    /// Pop the oldest ready item from the queue, clearing its `ready`
    /// flag so a subsequent `setReady` re-enqueues it. Returns a copy
    /// of the WaitableItem (kind/handle/code) so callers can lift the
    /// event payload without re-indexing. `null` when no ready items
    /// remain — the wait/poll arm signals NONE in that case.
    pub fn popReadyEvent(self: *WaitableSet) ?WaitableItem {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.ready_queue.items.len > 0) {
            const idx = self.ready_queue.orderedRemove(0);
            if (idx >= self.items.items.len) continue;
            const item = &self.items.items[idx];
            if (!item.active or !item.ready) continue;
            item.ready = false;
            return item.*;
        }
        return null;
    }

    /// Poll for ready items without blocking. Returns the count of
    /// ready entries, writing their `items[]` indices into `out` (up
    /// to `out.len`). Drains the `ready_queue` of those indices.
    ///
    /// Kept for the smoke-tests that assert "some waitable woke" —
    /// real event delivery goes through `popReadyEvent` so the
    /// guest receives `(kind, handle, code)`.
    pub fn pollReady(self: *WaitableSet, out: []u32) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var count: u32 = 0;
        for (self.items.items, 0..) |*item, i| {
            if (item.active and item.ready) {
                if (count < out.len) {
                    out[count] = @intCast(i);
                    count += 1;
                }
                item.ready = false; // consume readiness
            }
        }
        // The ready_queue is now stale — clear it so subsequent
        // `popReadyEvent` callers don't re-surface drained items.
        self.ready_queue.clearRetainingCapacity();
        return count;
    }

    pub fn deinit(self: *WaitableSet, allocator: std.mem.Allocator) void {
        self.mutex.lock();
        var items = self.items;
        var ready_queue = self.ready_queue;
        self.items = .empty;
        self.ready_queue = .empty;
        self.mutex.unlock();
        items.deinit(allocator);
        ready_queue.deinit(allocator);
    }
};

// ── Task Manager ────────────────────────────────────────────────────────────

/// Manages the lifecycle of async tasks within a component instance.
pub const TaskManager = struct {
    tasks: std.ArrayListUnmanaged(Task) = .empty,
    next_id: u32 = 0,
    allocator: ?std.mem.Allocator = null,
    mutex: stable_resource.ConditionalMutex(stable_resource.LockRank.resource_registry) = .init,

    pub const CurrentGuard = struct {
        previous: ?CurrentTaskBinding,
        active: bool = true,

        pub fn deinit(self: *CurrentGuard) void {
            if (!self.active) return;
            current_task_binding = self.previous;
            self.active = false;
        }
    };

    /// Create a new task. Returns the task handle.
    pub fn createTask(self: *TaskManager, allocator: std.mem.Allocator) !u32 {
        const cancellation_source = try task_cancellation.Source.create(allocator);
        errdefer cancellation_source.release();

        self.mutex.lock();
        defer self.mutex.unlock();
        const task_allocator = self.allocator orelse allocator;
        if (self.allocator == null) self.allocator = task_allocator;
        const id = self.next_id;
        try self.tasks.append(task_allocator, .{
            .id = id,
            .cancellation_source = cancellation_source,
        });
        self.next_id += 1;
        return id;
    }

    fn taskIndexLocked(self: *const TaskManager, handle: u32) ?usize {
        if (self.tasks.items.len == 0) return null;
        const first_id = self.tasks.items[0].id;
        if (handle < first_id) return null;
        const index: usize = @intCast(handle - first_id);
        if (index >= self.tasks.items.len) return null;
        if (self.tasks.items[index].id != handle) return null;
        return index;
    }

    /// Transition a task to the started state.
    pub fn startTask(self: *TaskManager, handle: u32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.taskIndexLocked(handle)) |index| {
            if (self.tasks.items[index].state == .created)
                self.tasks.items[index].state = .started;
        }
    }

    /// Transition a task to the returned state with values.
    pub fn returnTask(self: *TaskManager, handle: u32, values: []u32) void {
        const allocator = self.allocator orelse std.heap.page_allocator;
        const owned = allocator.dupe(u32, values) catch {
            self.cancelTask(handle);
            return;
        };
        self.returnTaskOwned(handle, owned, allocator);
    }

    /// Transition a task to returned and take ownership of `values`.
    pub fn returnTaskOwned(
        self: *TaskManager,
        handle: u32,
        values: []u32,
        values_allocator: std.mem.Allocator,
    ) void {
        var waitable_set: ?*WaitableSet = null;
        var waitable_idx: ?u32 = null;
        var old_values: []u32 = &.{};
        var old_values_allocator: ?std.mem.Allocator = null;
        var accepted = false;
        self.mutex.lock();
        if (self.taskIndexLocked(handle)) |index| {
            const task = &self.tasks.items[index];
            if (task.state == .created or task.state == .started) {
                old_values = task.return_values;
                old_values_allocator = task.return_values_allocator;
                task.state = .returned;
                task.return_values = values;
                task.return_values_allocator = values_allocator;
                waitable_set = task.waitable_set;
                waitable_idx = task.waitable_idx;
                accepted = true;
            }
        }
        const allocator = self.allocator;
        self.mutex.unlock();

        const owned_allocator = allocator orelse std.heap.page_allocator;
        if (!accepted) {
            values_allocator.free(values);
            return;
        }
        if (old_values.len > 0) old_values_allocator.?.free(old_values);
        if (waitable_set) |ws| if (waitable_idx) |idx| {
            ws.setReady(
                idx,
                owned_allocator,
                @intFromEnum(TaskState.returned),
            );
        };
    }

    /// Cancel a task.
    pub fn cancelTask(self: *TaskManager, handle: u32) void {
        _ = self.tryCancelTask(handle);
    }

    /// Cancel a task unless it already returned. Returns true when the task is
    /// cancelled after the call (including an idempotent repeated cancel).
    /// Return and cancellation are first-terminal under the manager lock.
    pub fn tryCancelTask(self: *TaskManager, handle: u32) bool {
        var source_ref: ?task_cancellation.Source.Ref = null;
        self.mutex.lock();
        const index = self.taskIndexLocked(handle) orelse {
            self.mutex.unlock();
            return false;
        };
        const task = &self.tasks.items[index];
        const cancelled = switch (task.state) {
            .created, .started => blk: {
                task.state = .cancelled;
                break :blk true;
            },
            .cancelled => true,
            .returned => false,
        };
        if (cancelled)
            source_ref = task.cancellation_source.acquire();
        self.mutex.unlock();

        if (source_ref) |*source| {
            source.source.cancel();
            source.deinit();
        }
        return cancelled;
    }

    /// Get the state of a task.
    pub fn getState(self: *const TaskManager, handle: u32) ?TaskState {
        const mutable: *TaskManager = @constCast(self);
        mutable.mutex.lock();
        defer mutable.mutex.unlock();
        const index = mutable.taskIndexLocked(handle) orelse return null;
        return self.tasks.items[index].state;
    }

    /// Read a per-task `context.get i32 i` slot. Returns `null` if the
    /// task handle is unknown or the slot index is out of range.
    pub fn getContextSlot(self: *const TaskManager, handle: u32, slot: u32) ?u32 {
        const mutable: *TaskManager = @constCast(self);
        mutable.mutex.lock();
        defer mutable.mutex.unlock();
        const index = mutable.taskIndexLocked(handle) orelse return null;
        if (slot >= N_CONTEXT_SLOTS) return null;
        return self.tasks.items[index].context_slots[slot];
    }

    /// Write a per-task `context.set i32 i` slot. Returns `false` if the
    /// task handle is unknown or the slot index is out of range; the
    /// caller surfaces this as a component trap.
    pub fn setContextSlot(self: *TaskManager, handle: u32, slot: u32, value: u32) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        const index = self.taskIndexLocked(handle) orelse return false;
        if (slot >= N_CONTEXT_SLOTS) return false;
        self.tasks.items[index].context_slots[slot] = value;
        return true;
    }

    pub fn deinit(self: *TaskManager, allocator: std.mem.Allocator) void {
        self.mutex.lock();
        var tasks = self.tasks;
        const task_allocator = self.allocator orelse allocator;
        self.tasks = .empty;
        self.allocator = null;
        self.mutex.unlock();
        for (tasks.items) |task| {
            if (task.return_values.len > 0) {
                task.return_values_allocator.?.free(task.return_values);
            }
            task.cancellation_source.release();
        }
        tasks.deinit(task_allocator);
    }

    pub fn bindCurrent(self: *TaskManager, handle: u32) CurrentGuard {
        const previous = current_task_binding;
        current_task_binding = .{ .manager = self, .handle = handle };
        return .{ .previous = previous };
    }

    pub fn currentTask(self: *TaskManager) ?u32 {
        const binding = current_task_binding orelse return null;
        if (binding.manager != self) return null;
        return binding.handle;
    }

    pub const AcquireCancellationTicketError = error{InvalidTaskHandle};

    /// Capture a durable cancellation ticket while the task handle is valid.
    ///
    /// The caller owns the returned ticket and must call `deinit` exactly
    /// once. It can be cloned, moved into an HTTP body/trailer settlement
    /// owner, and queried after the frame, task owner, or TaskManager is
    /// destroyed. The ticket retains only the immutable task source
    /// generation, never a Task or TaskManager pointer.
    pub fn acquireCancellationTicket(
        self: *TaskManager,
        handle: u32,
    ) AcquireCancellationTicketError!task_cancellation.Source.Ticket {
        self.mutex.lock();
        defer self.mutex.unlock();
        const index = self.taskIndexLocked(handle) orelse
            return error.InvalidTaskHandle;
        return self.tasks.items[index].cancellation_source.acquire();
    }

    pub fn setWaitable(
        self: *TaskManager,
        handle: u32,
        waitable_set: *WaitableSet,
        waitable_idx: u32,
    ) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        const index = self.taskIndexLocked(handle) orelse return false;
        self.tasks.items[index].waitable_set = waitable_set;
        self.tasks.items[index].waitable_idx = waitable_idx;
        return true;
    }

    pub fn taskCount(self: *const TaskManager) usize {
        const mutable: *TaskManager = @constCast(self);
        mutable.mutex.lock();
        defer mutable.mutex.unlock();
        return self.tasks.items.len;
    }

    pub fn copyReturnValues(
        self: *const TaskManager,
        handle: u32,
        allocator: std.mem.Allocator,
    ) !?[]u32 {
        const mutable: *TaskManager = @constCast(self);
        mutable.mutex.lock();
        defer mutable.mutex.unlock();
        const index = mutable.taskIndexLocked(handle) orelse return null;
        const task = self.tasks.items[index];
        if (task.state != .returned) return null;
        return try allocator.dupe(u32, task.return_values);
    }

    pub fn returnValuesForTesting(
        self: *const TaskManager,
        handle: u32,
    ) ?[]u32 {
        if (!@import("builtin").is_test)
            @compileError("use copyReturnValues() outside tests");
        const mutable: *TaskManager = @constCast(self);
        mutable.mutex.lock();
        defer mutable.mutex.unlock();
        const index = mutable.taskIndexLocked(handle) orelse return null;
        const task = self.tasks.items[index];
        if (task.state != .returned) return null;
        return task.return_values;
    }

    pub fn yieldTask(self: *TaskManager, handle: u32, cancellable: bool) bool {
        var waitable_set: ?*WaitableSet = null;
        self.mutex.lock();
        const index = self.taskIndexLocked(handle) orelse {
            self.mutex.unlock();
            return false;
        };
        const task = &self.tasks.items[index];
        if (cancellable and task.state == .cancelled) {
            self.mutex.unlock();
            return true;
        }
        waitable_set = task.waitable_set;
        if (task.state == .created) task.state = .started;
        self.mutex.unlock();

        if (waitable_set) |ws| {
            var sink: [16]u32 = undefined;
            _ = ws.pollReady(&sink);
        }
        return false;
    }
};

const CurrentTaskBinding = struct {
    manager: *TaskManager,
    handle: u32,
};

threadlocal var current_task_binding: ?CurrentTaskBinding = null;

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
    /// Stable waitable-set registrations. Handles carry generations, so a
    /// dropped set can never redirect a later notification to a new set.
    read_waitable: ?WaitableRegistration = null,
    write_waitable: ?WaitableRegistration = null,
    state: State = .pending,
    /// Both ends are closed only after both `drop-readable` and
    /// `drop-writable`. Tracked separately so cancellation observes the
    /// correct end (see `dispatchAsyncCanon.future_read`/`future_write`).
    read_closed: bool = false,
    write_closed: bool = false,
    /// True iff the future was minted by a canon-lower-of-async-func
    /// path (#551) — i.e. by a host adapter that wants the WaitableSet
    /// wakeup to be surfaced through the `(handle << 4) | STATUS`
    /// subtask shape rather than the `future.read` return-code
    /// convention. Read by `executor.joinWaitable` /
    /// `executor.popReadyEvent` to route the right wakeup channel.
    ///
    /// Distinct from the existing `future.read` / `future.write`
    /// rendezvous flow — that path doesn't go through
    /// `waitable-set.wait` event delivery (it uses `BLOCKED_STATUS`
    /// status-word parking instead), and emitting an EVENT_SUBTASK
    /// event for a `future.read` waiter would mis-decode the
    /// `STATUS_RETURNED=2` payload as `ReturnCode::Cancelled` per
    /// `wit-bindgen ≥ 0.53`'s `crates/guest-rust/src/rt/async_support`
    /// constants — see the cli-stdio-roundtrip regression note in
    /// `executor.popReadyEvent`.
    subtask_managed: bool = false,

    /// Set by `componentTrampoline` on the canon-lower-of-async-func
    /// path (#564) when the host returns a `.pending` future for a
    /// non-empty-result async func. The host settle path (whoever
    /// eventually transitions `state` to `.ready`/`.closed`) must copy
    /// `payload` bytes into `guest_mem[async_lower_retptr..]` so the
    /// guest observes the canonical-ABI lifted result at the address
    /// it passed in to the lower trampoline.
    ///
    /// `null` for futures minted outside the canon-lower-async path
    /// (e.g. `future.new` / `wasi:clocks` timer-futures whose lifted
    /// result is unit and so have no payload). Today every P3 host
    /// adapter completes synchronously and writes `mem[retptr..]`
    /// directly from inside the trampoline; this field exists so any
    /// genuinely-async host body (HTTP request inflight, DNS resolve,
    /// long-blocking connect) can wire deferred completion later
    /// without changing the trampoline contract.
    async_lower_retptr: ?u32 = null,

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

/// Outcome of a `HostStreamDriver` callback. Tells the executor what to
/// do after invoking the driver.
pub const HostStreamAction = enum {
    /// Progress made — the driver appended bytes to the FIFO (for
    /// `on_read`) or successfully consumed the caller's bytes (for
    /// `on_write`). The executor's stream op will treat this as a
    /// completed operation.
    progressed,
    /// No progress yet — no host-side data available (read) or the
    /// host sink is temporarily not ready (write). The executor should
    /// fall back to its default behaviour (park reader / buffer write).
    would_block,
    /// The host-side source / sink is exhausted (peer closed, fd
    /// hit EOF). The executor should mark the corresponding stream end
    /// closed so subsequent ops surface stream-end naturally.
    eof,
    /// The host-side driver encountered an unrecoverable error.
    /// Equivalent to `eof` for the executor: close the appropriate end.
    err,
};

/// Result returned by a zero-copy `on_read_into` driver callback.
/// Distinct from the buffer-appending `on_read` shape because the
/// driver no longer hands the executor an *implicit* byte count via
/// `stream.buffer.items.len`; instead it states explicitly how many
/// bytes it wrote into the guest-supplied destination.
///
/// Invariants:
///   * `bytes_written` MUST be ≤ the `dst.len` passed to the callback.
///   * On `.would_block` / `.eof` / `.err`, `bytes_written` MUST be 0.
///   * On `.progressed` with `bytes_written == 0`, the executor treats
///     it as a defensive `.would_block` (no spinning).
pub const HostStreamReadInto = struct {
    action: HostStreamAction,
    bytes_written: u32 = 0,
};

/// Optional host-driven I/O attached to an `AsyncStream`. When set, the
/// executor's `stream.read` / `stream.write` paths invoke the driver
/// before parking / buffering so a long-lived host source / sink (a
/// TCP fd, a `listen()` accept queue, a UDP socket) can continuously
/// service the stream FIFO without per-call host fn dispatches.
///
/// Both callbacks are optional: a read-only stream sets `on_read`; a
/// write-only stream sets `on_write`; bidirectional models would set
/// both (none of the current sockets bindings do).
///
/// The executor holds the canonical FIFO in `AsyncStream.buffer` — the
/// driver appends to / drains from it directly via the `*AsyncStream`
/// pointer the executor passes in.
///
/// ## Zero-copy (#583 B2)
///
/// Drivers that can read straight into a caller-supplied byte slice may
/// additionally set `on_read_into`. When the executor's `stream.read`
/// arm sees a sufficiently-aligned guest destination and a positive
/// `max_count`, it borrows a slice of guest linmem (via
/// `comp_inst.writableGuestBytes`, which validates `ptr + len ≤
/// memory.size`) and passes that slice to the driver — skipping the
/// `stream.buffer` scratch allocation and the second `@memcpy` from
/// the FIFO into guest linmem. The slice is only valid for the
/// synchronous duration of the call; the driver must not retain it.
///
/// The symmetric `stream.write` path already passes the driver a
/// borrowed slice of guest linmem via `comp_inst.readGuestBytes`, so
/// the write side is implicitly zero-copy already (no scratch FIFO
/// allocation, no `@memcpy` between the legacy `on_write` callback
/// and the host syscall). Drivers can however set `on_write_from`
/// (#583 B2 follow-up) for a thinner callback signature that drops
/// the unused `*AsyncStream` and `Allocator` parameters — the
/// executor prefers it over `on_write` when both are installed and
/// the API matches `on_read_into`'s shape, keeping the driver
/// surface symmetric end-to-end.
pub const HostStreamDriver = struct {
    /// Opaque context (typically `*WasiCliAdapter` plus a per-socket
    /// fd captured inline). Passed back to each callback verbatim.
    context: ?*anyopaque = null,
    /// Called by `stream.read` when the FIFO is empty and the writable
    /// end is not yet closed. The driver should attempt to append more
    /// bytes to `stream.buffer` and return whether progress was made.
    on_read: ?*const fn (
        ctx: ?*anyopaque,
        stream: *AsyncStream,
        allocator: std.mem.Allocator,
    ) HostStreamAction = null,
    /// Zero-copy variant of `on_read` (#583 B2). When set, the executor
    /// invokes this in preference to `on_read` and passes a borrowed
    /// slice into guest linmem. The driver writes bytes directly into
    /// `dst[0..]` — no scratch FIFO allocation, no second memcpy.
    ///
    /// `dst.len` is bounded by the guest's `max_count * elem_size` and
    /// has already been validated against `memory.size`. Drivers
    /// should issue a single non-blocking syscall per invocation and
    /// return `.would_block` when no data is available — the executor
    /// will park the read.
    ///
    /// Falling back: when `on_read_into` is set but the guest's
    /// destination is not addressable as a contiguous slice (e.g. the
    /// computed length overflows or extends past `memory.size`), the
    /// executor falls back to `on_read` if also set, otherwise parks
    /// the read — the driver does not see the zero-copy call in that
    /// case.
    on_read_into: ?*const fn (
        ctx: ?*anyopaque,
        dst: []u8,
    ) HostStreamReadInto = null,
    /// Called by `stream.write` when there's no parked reader and a
    /// guest write has arrived. The driver should attempt to consume
    /// the bytes (push them onto a host fd, etc.). On `would_block`
    /// the executor falls back to buffering the bytes in the FIFO.
    on_write: ?*const fn (
        ctx: ?*anyopaque,
        stream: *AsyncStream,
        bytes: []const u8,
        allocator: std.mem.Allocator,
    ) HostStreamAction = null,
    /// Zero-copy variant of `on_write` (#583 B2 follow-up). When set,
    /// the executor invokes this in preference to `on_write` and
    /// passes only the borrowed source slice — no `*AsyncStream`, no
    /// `Allocator`. Both legacy parameters are unused by every
    /// in-tree write driver (`tcpSendStreamOnWrite`,
    /// `fsWriteViaStreamOnWrite`) because they push the bytes
    /// straight to a host fd / file via `writeAll` / `pwriteAll`.
    ///
    /// `src` is a borrowed view of guest linmem already validated by
    /// `readGuestBytes(guest_ptr, byte_len)` against `memory.size`
    /// before the call — the slice stays valid for the synchronous
    /// duration of the call (the executor never yields to a
    /// `memory.grow` between the bounds check and the driver
    /// return). Drivers must not retain the slice.
    ///
    /// Action semantics match `on_write`:
    ///   * `.progressed` — all `src.len` bytes accepted by the host
    ///     sink; the executor reports `completed(count)` to the
    ///     guest.
    ///   * `.would_block` — sink not ready; executor falls back to
    ///     FIFO buffering so a later read / drain can pick up the
    ///     bytes.
    ///   * `.eof` / `.err` — sink is gone / failed; executor flips
    ///     `read_closed = true` and surfaces `dropped(0)`.
    on_write_from: ?*const fn (
        ctx: ?*anyopaque,
        src: []const u8,
    ) HostStreamAction = null,
    /// Called when the guest drops the readable end of a host-driven
    /// stream. Socket sources use this to propagate `shutdown(SHUT_RD)`.
    on_drop_readable: ?*const fn (ctx: ?*anyopaque) void = null,
    /// Called when the guest drops the writable end of a host-driven
    /// stream. Socket sinks use this to propagate EOF with a half-close
    /// after all preceding writes have reached the host fd.
    on_drop_writable: ?*const fn (ctx: ?*anyopaque) void = null,
    /// Releases the host-owned context after both stream ends are gone.
    /// Called exactly once by `AsyncStream.deinit`, outside table locks.
    on_destroy: ?*const fn (ctx: ?*anyopaque) void = null,
};

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
    /// Optional per-stream override of the per-element byte size used
    /// when the executor's `stream.read t` instruction carries a
    /// `type_idx` that doesn't resolve in the host-side
    /// `TypeRegistry` (typically because the element type lives in
    /// another instance / wraps an unsupported recursive form).
    ///
    /// Host-side eager-lowering producers (`fsDescriptorReadDirectoryP3`,
    /// etc.) set this to the byte stride they actually appended to
    /// `buffer`, so the executor drains in the correct stride even
    /// when guest-side resolution would have returned a stale fallback
    /// size. (#571 — read-directory + cross-instance element types.)
    elem_size_hint: ?u32 = null,
    /// FIFO of raw lowered bytes. Element boundaries are recomputed at
    /// op time from `elem_size = sizeOfType(...)`.
    buffer: std.ArrayListUnmanaged(u8) = .empty,
    /// Optional FIFO high-water mark. When full, `stream.write` parks
    /// without copying the pending source; a reader wake makes the guest
    /// re-issue the write against newly available capacity.
    buffer_limit: ?usize = null,

    /// A reader that ran while the buffer was empty. Resolved by the next
    /// `write` (direct memcpy into the parked dst, no buffering).
    pending_read: ?PendingRead = null,
    /// A writer that ran with no parked reader and exhausted the buffer
    /// cap. The source remains in guest memory; after a reader frees space,
    /// the write waitable wakes with zero completed elements so the guest
    /// re-issues against the newly available capacity.
    pending_write: ?PendingWrite = null,
    /// A pending writer was made runnable before it joined a waitable set.
    /// The next join observes this latched edge and queues completion.
    write_ready: bool = false,
    /// Cancellation retired the reader after a writer returned BLOCKED but
    /// before its waitable joined. The stream remains as a zero-buffer
    /// tombstone until that join observes DROPPED, then the executor removes
    /// it. Arbitrary unknown handles still remain inert.
    terminal_write_dropped: bool = false,

    /// Optional host-side I/O hook for long-lived sockets (#535). When
    /// set, the executor's stream ops invoke the driver before parking
    /// / buffering so a long-lived TCP fd, accept queue, or UDP socket
    /// can keep the FIFO fed without a per-call host fn dispatch.
    host_driver: ?HostStreamDriver = null,
    /// Optional host-attached sink/source for `wasi:cli` stdio (#537).
    /// When set, `stream.write` / `stream.read` from the guest is
    /// forwarded directly to the host (via `on_write` / `on_read`)
    /// instead of being buffered. This is how
    /// `wasi:cli/{stdout,stderr,stdin}@0.3.x.{write,read}-via-stream`
    /// keep WAMR's single-threaded synchronous model in sync with the
    /// guest's `futures::join!` style concurrent writers — the host
    /// effectively stays "parked" on its end indefinitely, draining /
    /// producing synchronously at every guest op.
    ///
    /// Distinct from `host_driver` (#535): `host_driver` integrates a
    /// long-lived socket fd into the rendezvous loop with optional
    /// would-block fallback to buffering, whereas `host_handler`
    /// targets synchronous-once stdio with future-coupled completion.
    host_handler: ?HostStreamHandler = null,

    /// Reserve host callbacks under the stream lock, then invoke them after
    /// unlocking. Reentrant operations observe the reservation and never
    /// recursively invoke the same callback.
    host_read_inflight: bool = false,
    host_write_inflight: bool = false,

    /// Stable waitable-set registrations.
    read_waitable: ?WaitableRegistration = null,
    write_waitable: ?WaitableRegistration = null,

    state: State = .open,
    /// Both ends are closed only after both `drop-readable` and
    /// `drop-writable`. Tracked separately so cancellation observes the
    /// correct end.
    read_closed: bool = false,
    write_closed: bool = false,

    pub const State = enum { open, closed };
    pub const PendingRead = struct {
        guest_ptr: u32,
        max_count: u32,
        /// Canonical byte width captured when `stream.read` parks.
        /// Host event drivers need this to complete the original read
        /// before delivering its waitable event.
        elem_size: u32,
    };
    pub const PendingWrite = struct {
        guest_ptr: u32,
        count: u32,
        elem_size: u32 = 1,
    };

    /// Free any heap-owned state (currently just the FIFO `buffer`).
    /// Safe to call multiple times.
    pub fn deinit(self: *AsyncStream, allocator: std.mem.Allocator) void {
        if (self.host_driver) |driver| {
            if (driver.on_destroy) |destroy| destroy(driver.context);
            self.host_driver = null;
        }
        self.buffer.deinit(allocator);
    }
};

/// Host-side callbacks attached to an `AsyncStream` so the host can
/// participate in guest stream I/O without an asynchronous scheduler.
/// A stream has two ends; the host always attaches to the end opposite
/// the guest:
///
///   * `wasi:cli/stdout@0.3.x.write-via-stream` — host is on the READ
///     end. `on_write` fires from `stream.write` to drain guest data
///     directly into a host sink; `on_drop_writable` fires when the
///     guest closes its writer.
///   * `wasi:cli/stdin@0.3.x.read-via-stream` — host is on the WRITE
///     end. `on_read` fires from `stream.read` (when the buffer is
///     empty) so the host can produce bytes from a real fd.
///
/// Installed by the corresponding adapter functions in
/// `wasi_cli_adapter.zig` (#537).
pub const HostStreamHandler = struct {
    /// Acquire/release a callback-scoped reference while the stream table
    /// lock is dropped. `retain_context` runs before unlocking; returning
    /// false rejects a callback after the host has begun detaching.
    retain_context: ?*const fn (ctx: ?*anyopaque) bool = null,
    release_context: ?*const fn (ctx: ?*anyopaque) void = null,
    /// Host-on-read-end: drain guest write directly to a host sink.
    /// Returns `true` on success, `false` if the sink rejected
    /// (closed / errored).
    on_write: ?*const fn (ctx: ?*anyopaque, bytes: []const u8) bool = null,
    /// Host-on-read-end: notification fired from `stream.drop-writable`
    /// when the guest closes its writer end. Used to settle the
    /// companion `future<result<_,error-code>>`.
    on_drop_writable: ?*const fn (ctx: ?*anyopaque) void = null,

    /// Host-on-write-end: produce bytes for a guest read. Returns:
    ///   * `> 0` — number of bytes written into `dst`.
    ///   * `0`   — EOF; caller should mark the stream `write_closed`.
    ///   * `< 0` — error; caller should surface as DROPPED to the guest.
    on_read: ?*const fn (ctx: ?*anyopaque, dst: []u8) i32 = null,

    /// Optional destructor — fired when the stream is fully closed
    /// (both ends dropped) and removed from the table. Lets the host
    /// release its `ctx` allocation without leaking on Debug builds.
    on_destroy: ?*const fn (ctx: ?*anyopaque) void = null,

    /// Opaque user context passed back to the callbacks.
    ctx: ?*anyopaque = null,
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

test "TaskManager: handle cancellation fans out every registered target" {
    if (!config.lib_wasi_threads) return error.SkipZigTest;
    const Counter = struct {
        hits: usize = 0,

        fn wake(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.hits += 1;
        }
    };
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    defer tm.deinit(allocator);
    const handle = try tm.createTask(allocator);
    var source_ref = try tm.acquireCancellationTicket(handle);
    defer source_ref.deinit();
    var first = Counter{};
    var second = Counter{};
    var first_registration = task_cancellation.Registration{};
    defer first_registration.unregister();
    var second_registration = task_cancellation.Registration{};
    defer second_registration.unregister();
    try std.testing.expectEqual(
        task_cancellation.RegisterResult.registered,
        source_ref.source.register(&first_registration, .{
            .ctx = @ptrCast(&first),
            .wake = Counter.wake,
        }),
    );
    try std.testing.expectEqual(
        task_cancellation.RegisterResult.registered,
        source_ref.source.register(&second_registration, .{
            .ctx = @ptrCast(&second),
            .wake = Counter.wake,
        }),
    );

    tm.cancelTask(handle);
    try std.testing.expectEqual(@as(usize, 1), first.hits);
    try std.testing.expectEqual(@as(usize, 1), second.hits);
    tm.cancelTask(handle);
    try std.testing.expectEqual(@as(usize, 2), first.hits);
    try std.testing.expectEqual(@as(usize, 2), second.hits);
}

test "TaskManager: registered cancellation source outlives manager teardown" {
    if (!config.lib_wasi_threads) return error.SkipZigTest;
    const Counter = struct {
        hits: usize = 0,

        fn wake(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.hits += 1;
        }
    };
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    const handle = try tm.createTask(allocator);
    var source_ref = try tm.acquireCancellationTicket(handle);
    defer source_ref.deinit();
    var counter = Counter{};
    var registration = task_cancellation.Registration{};
    defer registration.unregister();
    if (comptime config.lib_wasi_threads) {
        try std.testing.expectEqual(
            task_cancellation.RegisterResult.registered,
            source_ref.source.register(&registration, .{
                .ctx = @ptrCast(&counter),
                .wake = Counter.wake,
            }),
        );
    }

    tm.deinit(allocator);
    source_ref.source.cancel();
    try std.testing.expectEqual(@as(usize, 1), counter.hits);
}

test "TaskManager: cancellation sources distinguish task generations" {
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    const first_handle = try tm.createTask(allocator);
    var first_source = try tm.acquireCancellationTicket(first_handle);
    defer first_source.deinit();
    tm.deinit(allocator);

    tm = .{};
    defer tm.deinit(allocator);
    const second_handle = try tm.createTask(allocator);
    var second_source = try tm.acquireCancellationTicket(second_handle);
    defer second_source.deinit();
    try std.testing.expect(first_source.source.id != second_source.source.id);
}

test "TaskManager: durable cancellation ticket survives owner teardown" {
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    const handle = try tm.createTask(allocator);
    var ticket = try tm.acquireCancellationTicket(handle);
    var clone = ticket.clone();

    tm.cancelTask(handle);
    tm.deinit(allocator);

    try std.testing.expect(ticket.isCancelled());
    try std.testing.expect(clone.isCancelled());
    try std.testing.expectEqual(ticket.identity(), clone.identity());
    ticket.deinit();
    clone.deinit();
}

test "TaskManager: ticket acquired after cancellation observes terminal flag" {
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    defer tm.deinit(allocator);
    const handle = try tm.createTask(allocator);
    tm.cancelTask(handle);
    var ticket = try tm.acquireCancellationTicket(handle);
    defer ticket.deinit();
    try std.testing.expect(ticket.isCancelled());
}

test "TaskManager: invalid and stale handles cannot acquire another task ticket" {
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    const stale = try tm.createTask(allocator);
    tm.deinit(allocator);

    defer tm.deinit(allocator);
    const current = try tm.createTask(allocator);
    try std.testing.expect(stale != current);
    try std.testing.expectError(
        error.InvalidTaskHandle,
        tm.acquireCancellationTicket(stale),
    );
    try std.testing.expectError(
        error.InvalidTaskHandle,
        tm.acquireCancellationTicket(std.math.maxInt(u32)),
    );
    var ticket = try tm.acquireCancellationTicket(current);
    defer ticket.deinit();
    try std.testing.expect(!ticket.isCancelled());
}

test "TaskManager: return and cancel are first-terminal under races" {
    if (@import("builtin").single_threaded)
        return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var tm = TaskManager{};
    defer tm.deinit(allocator);

    const Racer = struct {
        manager: *TaskManager,
        handle: u32,
        values: []u32,
        gate: *std.atomic.Value(bool),

        fn cancel(self: *@This()) void {
            while (!self.gate.load(.acquire)) std.atomic.spinLoopHint();
            _ = self.manager.tryCancelTask(self.handle);
        }

        fn complete(self: *@This()) void {
            while (!self.gate.load(.acquire)) std.atomic.spinLoopHint();
            self.manager.returnTaskOwned(
                self.handle,
                self.values,
                std.testing.allocator,
            );
        }
    };
    const Counter = struct {
        hits: std.atomic.Value(usize) = .init(0),

        fn wake(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.hits.fetchAdd(1, .acq_rel);
        }
    };

    var round: usize = 0;
    while (round < 64) : (round += 1) {
        const handle = try tm.createTask(allocator);
        tm.startTask(handle);
        var source_ref = try tm.acquireCancellationTicket(handle);
        defer source_ref.deinit();
        var counter = Counter{};
        var registration = task_cancellation.Registration{};
        defer registration.unregister();
        if (comptime config.lib_wasi_threads) {
            try std.testing.expectEqual(
                task_cancellation.RegisterResult.registered,
                source_ref.source.register(&registration, .{
                    .ctx = @ptrCast(&counter),
                    .wake = Counter.wake,
                }),
            );
        }
        const values = try allocator.dupe(u32, &.{@intCast(round)});
        var gate = std.atomic.Value(bool).init(false);
        var racer = Racer{
            .manager = &tm,
            .handle = handle,
            .values = values,
            .gate = &gate,
        };
        const cancel_thread = try std.Thread.spawn(.{}, Racer.cancel, .{&racer});
        const return_thread = try std.Thread.spawn(.{}, Racer.complete, .{&racer});
        gate.store(true, .release);
        cancel_thread.join();
        return_thread.join();

        switch (tm.getState(handle).?) {
            .cancelled => {
                try std.testing.expect(tm.tryCancelTask(handle));
                tm.startTask(handle);
                try std.testing.expectEqual(TaskState.cancelled, tm.getState(handle).?);
                try std.testing.expect(source_ref.isCancelled());
                if (comptime config.lib_wasi_threads)
                    try std.testing.expect(counter.hits.load(.acquire) >= 1);
            },
            .returned => {
                try std.testing.expect(!tm.tryCancelTask(handle));
                tm.startTask(handle);
                try std.testing.expectEqual(TaskState.returned, tm.getState(handle).?);
                try std.testing.expect(!source_ref.isCancelled());
                try std.testing.expectEqual(@as(usize, 0), counter.hits.load(.acquire));
            },
            else => return error.TestUnexpectedResult,
        }
    }
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

    ws.setReady(idx1, allocator, 0);
    var out: [4]u32 = undefined;
    const count = ws.pollReady(&out);
    try std.testing.expectEqual(@as(u32, 1), count);
    try std.testing.expectEqual(idx1, out[0]);
}

test "WaitableSet: popReadyEvent surfaces kind/handle/code in FIFO order" {
    const allocator = std.testing.allocator;
    var ws = WaitableSet{};
    defer ws.deinit(allocator);

    const idx_w = try ws.register(.{ .kind = .stream_write, .handle = 7 }, allocator);
    const idx_r = try ws.register(.{ .kind = .future_read, .handle = 9 }, allocator);

    // No ready items yet.
    try std.testing.expect(ws.popReadyEvent() == null);

    // Refreshing `code` on the same item must not double-enqueue.
    ws.setReady(idx_w, allocator, 0x40);
    ws.setReady(idx_w, allocator, 0x42);
    ws.setReady(idx_r, allocator, 0x10);

    const first = ws.popReadyEvent().?;
    try std.testing.expectEqual(WaitableSet.WaitableItem.Kind.stream_write, first.kind);
    try std.testing.expectEqual(@as(u32, 7), first.handle);
    try std.testing.expectEqual(@as(u32, 0x42), first.code);

    const second = ws.popReadyEvent().?;
    try std.testing.expectEqual(WaitableSet.WaitableItem.Kind.future_read, second.kind);
    try std.testing.expectEqual(@as(u32, 9), second.handle);
    try std.testing.expectEqual(@as(u32, 0x10), second.code);

    // Queue drained.
    try std.testing.expect(ws.popReadyEvent() == null);
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

test "AsyncStream: pending_read records read shape" {
    var s = AsyncStream{};
    s.pending_read = .{ .guest_ptr = 0x2000, .max_count = 7, .elem_size = 4 };
    try std.testing.expectEqual(@as(u32, 0x2000), s.pending_read.?.guest_ptr);
    try std.testing.expectEqual(@as(u32, 7), s.pending_read.?.max_count);
    try std.testing.expectEqual(@as(u32, 4), s.pending_read.?.elem_size);
}

test "AsyncStream: host_driver field defaults to null and can be installed (#535)" {
    var s = AsyncStream{};
    try std.testing.expect(s.host_driver == null);

    const Cb = struct {
        fn onRead(
            _: ?*anyopaque,
            stream: *AsyncStream,
            allocator: std.mem.Allocator,
        ) HostStreamAction {
            stream.buffer.appendSlice(allocator, &[_]u8{ 0xAA, 0xBB }) catch return .err;
            return .progressed;
        }
    };
    s.host_driver = .{ .context = null, .on_read = &Cb.onRead };
    try std.testing.expect(s.host_driver != null);
    try std.testing.expect(s.host_driver.?.on_read != null);

    // Drive the callback manually (mirrors what the executor will do).
    const action = s.host_driver.?.on_read.?(null, &s, std.testing.allocator);
    try std.testing.expectEqual(HostStreamAction.progressed, action);
    try std.testing.expectEqual(@as(usize, 2), s.buffer.items.len);
    s.deinit(std.testing.allocator);
}

test "component resource safety: concurrent task handles and TLS binding" {
    if (!config.lib_wasi_threads) return error.SkipZigTest;

    const thread_count = 4;
    const tasks_per_thread = 128;
    var manager = TaskManager{};
    defer manager.deinit(std.testing.allocator);
    var failed = std.atomic.Value(bool).init(false);

    const Runner = struct {
        fn run(target: *TaskManager, did_fail: *std.atomic.Value(bool), seed: u32) void {
            var i: u32 = 0;
            while (i < tasks_per_thread) : (i += 1) {
                const handle = target.createTask(std.testing.allocator) catch {
                    did_fail.store(true, .release);
                    return;
                };
                var current = target.bindCurrent(handle);
                defer current.deinit();
                if (target.currentTask() != handle) {
                    did_fail.store(true, .release);
                    return;
                }
                target.startTask(handle);
                if (!target.setContextSlot(handle, 0, seed + i)) {
                    did_fail.store(true, .release);
                    return;
                }
                if (target.getContextSlot(handle, 0) != seed + i) {
                    did_fail.store(true, .release);
                    return;
                }
            }
        }
    };

    var threads: [thread_count]std.Thread = undefined;
    for (&threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(
            .{},
            Runner.run,
            .{ &manager, &failed, @as(u32, @intCast(i * tasks_per_thread)) },
        );
    }
    for (threads) |thread| thread.join();

    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expectEqual(
        @as(usize, thread_count * tasks_per_thread),
        manager.taskCount(),
    );
    try std.testing.expect(manager.currentTask() == null);
}

test "component resource safety: waitable set producer consumer race" {
    if (!config.lib_wasi_threads) return error.SkipZigTest;

    const producer_count = 4;
    var waitable_set = WaitableSet{};
    defer waitable_set.deinit(std.testing.allocator);
    var indices: [producer_count]u32 = undefined;
    for (&indices, 0..) |*index, i| {
        index.* = try waitable_set.register(.{
            .kind = .future_read,
            .handle = @intCast(i + 1),
        }, std.testing.allocator);
    }

    var completed = std.atomic.Value(usize).init(0);
    const Producer = struct {
        fn run(
            target: *WaitableSet,
            index: u32,
            done: *std.atomic.Value(usize),
        ) void {
            var i: u32 = 0;
            while (i < 2000) : (i += 1) {
                target.setReady(index, std.testing.allocator, i);
            }
            _ = done.fetchAdd(1, .release);
        }
    };

    var threads: [producer_count]std.Thread = undefined;
    for (&threads, indices) |*thread, index| {
        thread.* = try std.Thread.spawn(
            .{},
            Producer.run,
            .{ &waitable_set, index, &completed },
        );
    }

    var seen: u8 = 0;
    while (completed.load(.acquire) != producer_count) {
        while (waitable_set.popReadyEvent()) |event| {
            if (event.handle >= 1 and event.handle <= producer_count) {
                seen |= @as(u8, 1) << @intCast(event.handle - 1);
            }
        }
        std.Thread.yield() catch {};
    }
    for (threads) |thread| thread.join();
    while (waitable_set.popReadyEvent()) |event| {
        if (event.handle >= 1 and event.handle <= producer_count) {
            seen |= @as(u8, 1) << @intCast(event.handle - 1);
        }
    }
    try std.testing.expectEqual(
        @as(u8, (1 << producer_count) - 1),
        seen,
    );
}
