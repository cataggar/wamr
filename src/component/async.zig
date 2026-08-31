//! Component Model async ABI — tasks, futures, and streams.
//!
//! Implements the WASIp3 cooperative task model where component function
//! calls can be non-blocking. Each async call produces a subtask that
//! the caller can poll for completion via waitable sets.

const std = @import("std");
const execution_context = @import("../runtime/common/execution_context.zig");

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
    /// FIFO of indices that have become ready since the last
    /// `popReadyEvent`. Used by `waitable-set.{wait,poll}` to surface
    /// the oldest pending event to the guest. The matching `ready`
    /// flag on the `WaitableItem` lets `register`/`setReady` avoid
    /// double-enqueueing the same item.
    ready_queue: std.ArrayListUnmanaged(u32) = .empty, // indices into items

    pub const WaitableItem = struct {
        kind: Kind,
        handle: u32, // task/stream/future handle
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
        const idx: u32 = @intCast(self.items.items.len);
        try self.items.append(allocator, item);
        return idx;
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
        if (idx >= self.items.items.len) return;
        const item = &self.items.items[idx];
        item.code = code;
        if (!item.ready) {
            item.ready = true;
            self.ready_queue.append(allocator, idx) catch {};
        }
    }

    /// Pop the oldest ready item from the queue, clearing its `ready`
    /// flag so a subsequent `setReady` re-enqueues it. Returns a copy
    /// of the WaitableItem (kind/handle/code) so callers can lift the
    /// event payload without re-indexing. `null` when no ready items
    /// remain — the wait/poll arm signals NONE in that case.
    pub fn popReadyEvent(self: *WaitableSet) ?WaitableItem {
        while (self.ready_queue.items.len > 0) {
            const idx = self.ready_queue.orderedRemove(0);
            if (idx >= self.items.items.len) continue;
            const item = &self.items.items[idx];
            if (!item.ready) continue;
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
        // The ready_queue is now stale — clear it so subsequent
        // `popReadyEvent` callers don't re-surface drained items.
        self.ready_queue.clearRetainingCapacity();
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
                ws.setReady(handle, std.heap.page_allocator, @intFromEnum(TaskState.returned));
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

    /// A reader that ran while the buffer was empty. Resolved by the next
    /// `write` (direct memcpy into the parked dst, no buffering).
    pending_read: ?PendingRead = null,
    /// A writer that ran with no parked reader and exhausted the buffer
    /// cap. The initial implementation has no cap, so this stays `null`;
    /// kept for the future high-water-mark backpressure PR.
    pending_write: ?PendingWrite = null,

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
    pub const PendingRead = struct {
        guest_ptr: u32,
        max_count: u32,
        /// Canonical byte width captured when `stream.read` parks.
        /// Host event drivers need this to complete the original read
        /// before delivering its waitable event.
        elem_size: u32,
    };
    pub const PendingWrite = struct { guest_ptr: u32, count: u32 };

    /// Free any heap-owned state (currently just the FIFO `buffer`).
    /// Safe to call multiple times.
    pub fn deinit(self: *AsyncStream, allocator: std.mem.Allocator) void {
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
