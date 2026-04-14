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
