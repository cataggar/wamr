//! Async canonical ABI extensions — async lift/lower with subtask handles.
//!
//! Extends the synchronous canon lift/lower with async support. An async
//! lifted function returns a subtask handle that the caller polls via
//! a waitable set, rather than blocking until the callee returns.

const std = @import("std");
const ctypes = @import("types.zig");
const abi = @import("canonical_abi.zig");
const async_mod = @import("async.zig");

/// Options for an async canonical call.
pub const AsyncCanonOptions = struct {
    /// The waitable set to register the subtask on.
    waitable_set: ?*async_mod.WaitableSet = null,
    /// Task manager for lifecycle tracking.
    task_manager: *async_mod.TaskManager,
    allocator: std.mem.Allocator,
};

/// Result of an async canon lift — a subtask handle that can be polled.
pub const AsyncLiftResult = struct {
    subtask_handle: u32,
};

/// Perform an async canon lift: create a subtask for the function call,
/// register it with the waitable set, and return the handle.
///
/// The caller polls the waitable set to discover when the subtask
/// completes, then reads the return values from the task.
pub fn asyncLift(
    opts: AsyncCanonOptions,
) !AsyncLiftResult {
    const handle = try opts.task_manager.createTask(opts.allocator);
    opts.task_manager.startTask(handle);

    // Register with waitable set if provided
    if (opts.waitable_set) |ws| {
        _ = try ws.register(.{
            .kind = .subtask,
            .handle = handle,
        }, opts.allocator);
    }

    return .{ .subtask_handle = handle };
}

/// Complete an async subtask by providing return values.
/// Notifies the waitable set that the subtask is ready.
pub fn asyncReturn(
    task_manager: *async_mod.TaskManager,
    handle: u32,
    return_values: []u32,
) void {
    task_manager.returnTask(handle, return_values);
}

/// Cancel an async subtask.
pub fn asyncCancel(
    task_manager: *async_mod.TaskManager,
    handle: u32,
) void {
    task_manager.cancelTask(handle);
}

/// Outcome of `task.yield` (`canon thread.yield`, Binary.md tag 0x0c).
/// The component-model spec encodes the result as a single i32: zero on
/// a normal resumption, non-zero when the task observed a cancellation
/// request while parked (only possible if invoked with `cancellable`).
pub const YieldOutcome = enum(u32) {
    resumed = 0,
    cancelled = 1,
};

/// Discriminant for `future.read` / `future.write` / `future.cancel-*`
/// status words (Binary.md "Async Calls"). The on-the-wire status word
/// is `packStatus(status, count)` — bits 0..3 carry the discriminant,
/// bits 4..31 carry the number of elements transferred (for `future<T>`
/// always 0 or 1 since the channel is one-shot).
pub const FutureStatus = enum(u32) {
    starting = 0,
    started = 1,
    returned = 2,
    cancelled = 3,
};

/// Pack a future read/write status discriminant + element count into the
/// single i32 status word the spec returns from `future.{read,write}` and
/// `future.cancel-{read,write}`.
pub fn packStatus(status: FutureStatus, count: u32) u32 {
    return @intFromEnum(status) | (count << 4);
}

/// Execute `canon thread.yield cancel?` for the currently executing task.
///
/// Single-threaded runtime semantics: a "yield" is the smallest possible
/// cooperative-scheduling primitive. We don't (yet) have a fiber stack to
/// suspend onto, so we drain one round of readiness on any waitable set
/// the task is registered with and immediately resume. Sub-PR 3 of #478
/// replaces this with the real `future<T>` / `stream<T>` integration; the
/// surface this function exposes is stable.
///
/// When `cancellable == true` and the task is `.cancelled`, returns
/// `.cancelled` so the dispatcher can lower the right discriminant onto
/// the core wasm stack.
pub fn taskYield(
    task_manager: *async_mod.TaskManager,
    handle: u32,
    cancellable: bool,
    allocator: std.mem.Allocator,
) YieldOutcome {
    _ = allocator; // reserved for future scheduler hooks (sub-PR 3)

    if (handle >= task_manager.tasks.items.len) return .resumed;
    const task = &task_manager.tasks.items[handle];

    // Cancellation gets priority: if a cancellation arrived while the task
    // was on the dispatch stack, surface it before we drain readiness.
    if (cancellable and task.state == .cancelled) return .cancelled;

    // Drain one round of readiness on the task's waitable set, if any.
    // The dispatcher consumes the ready_queue via `pollReady` separately;
    // here we only need to clear the per-task suspend point — the
    // single-threaded scheduler is implicit (re-entry into wasm resumes
    // the task immediately on return from this built-in).
    if (task.waitable_set) |ws| {
        var sink: [16]u32 = undefined;
        _ = ws.pollReady(&sink);
    }

    // Re-arm. If the task was somehow flipped to `.returned` while we
    // were not looking, leave that alone — only `.created` would be
    // surprising, but defensively re-mark started just in case.
    if (task.state == .created) task.state = .started;

    return .resumed;
}

/// Check if a subtask has completed and retrieve its return values.
pub fn asyncPollResult(
    task_manager: *const async_mod.TaskManager,
    handle: u32,
) ?[]u32 {
    const state = task_manager.getState(handle) orelse return null;
    if (state != .returned) return null;
    if (handle < task_manager.tasks.items.len) {
        return task_manager.tasks.items[handle].return_values;
    }
    return null;
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "asyncLift: creates subtask and registers" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);
    var ws = async_mod.WaitableSet{};
    defer ws.deinit(allocator);

    const result = try asyncLift(.{
        .waitable_set = &ws,
        .task_manager = &tm,
        .allocator = allocator,
    });

    try std.testing.expectEqual(async_mod.TaskState.started, tm.getState(result.subtask_handle).?);
    try std.testing.expectEqual(@as(usize, 1), ws.items.items.len);
}

test "asyncReturn: completes subtask" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);

    const result = try asyncLift(.{
        .task_manager = &tm,
        .allocator = allocator,
    });

    var vals = [_]u32{ 42, 99 };
    asyncReturn(&tm, result.subtask_handle, &vals);

    const ret = asyncPollResult(&tm, result.subtask_handle);
    try std.testing.expect(ret != null);
    try std.testing.expectEqual(@as(u32, 42), ret.?[0]);
}

test "asyncCancel: cancels subtask" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);

    const result = try asyncLift(.{
        .task_manager = &tm,
        .allocator = allocator,
    });

    asyncCancel(&tm, result.subtask_handle);
    try std.testing.expectEqual(async_mod.TaskState.cancelled, tm.getState(result.subtask_handle).?);
    try std.testing.expect(asyncPollResult(&tm, result.subtask_handle) == null);
}

test "taskYield: resume on a live task" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);
    var ws = async_mod.WaitableSet{};
    defer ws.deinit(allocator);

    const result = try asyncLift(.{
        .waitable_set = &ws,
        .task_manager = &tm,
        .allocator = allocator,
    });

    const outcome = taskYield(&tm, result.subtask_handle, false, allocator);
    try std.testing.expectEqual(YieldOutcome.resumed, outcome);
    try std.testing.expectEqual(async_mod.TaskState.started, tm.getState(result.subtask_handle).?);
}

test "taskYield: cancellable observes cancellation" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);

    const result = try asyncLift(.{
        .task_manager = &tm,
        .allocator = allocator,
    });

    asyncCancel(&tm, result.subtask_handle);

    // Plain yield still reports `.resumed` (spec: non-cancellable yield
    // is opaque to cancellation observation).
    try std.testing.expectEqual(YieldOutcome.resumed, taskYield(&tm, result.subtask_handle, false, allocator));
    // Cancellable yield surfaces the pending cancellation.
    try std.testing.expectEqual(YieldOutcome.cancelled, taskYield(&tm, result.subtask_handle, true, allocator));
}

test "taskYield: unknown handle is a no-op" {
    const allocator = std.testing.allocator;
    var tm = async_mod.TaskManager{};
    defer tm.deinit(allocator);

    try std.testing.expectEqual(YieldOutcome.resumed, taskYield(&tm, 999, true, allocator));
}

test "packStatus: encodes discriminant + element count" {
    try std.testing.expectEqual(@as(u32, 0), packStatus(.starting, 0));
    try std.testing.expectEqual(@as(u32, 1), packStatus(.started, 0));
    try std.testing.expectEqual(@as(u32, 2), packStatus(.returned, 0));
    // `future<T>` only ever transfers 0 or 1 elements; verify the count
    // lands in bits 4..31.
    try std.testing.expectEqual(@as(u32, 2 | (1 << 4)), packStatus(.returned, 1));
    try std.testing.expectEqual(@as(u32, 3), packStatus(.cancelled, 0));
}
