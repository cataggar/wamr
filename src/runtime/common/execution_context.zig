//! Shared-process and per-thread execution-context primitives.
//!
//! Core runtime modules cannot depend directly on `wasi.zig`, so process
//! ownership is represented by a small type-erased retained reference.
//! `WasiProcessState.processStateRef()` supplies the concrete retain/release
//! operations. Every interpreter `ExecEnv` and AOT execution instance owns
//! its own `ThreadExecutionContext`; only the process reference inside it is
//! shared.

const std = @import("std");
const config = @import("config");

pub const task_context_slot_count: usize = 2;

pub const ProcessStateOps = struct {
    retain: *const fn (*anyopaque) void,
    release: *const fn (*anyopaque) void,
};

/// An owned reference to process-scoped host state.
///
/// Copying this value directly does not retain it. Use `acquire()` whenever a
/// new owner is created, and pair every acquired reference with `release()`.
pub const ProcessStateRef = struct {
    ptr: *anyopaque,
    ops: *const ProcessStateOps,

    pub fn init(ptr: *anyopaque, ops: *const ProcessStateOps) ProcessStateRef {
        return .{ .ptr = ptr, .ops = ops };
    }

    pub fn acquire(self: ProcessStateRef) ProcessStateRef {
        self.ops.retain(self.ptr);
        return self;
    }

    pub fn release(self: ProcessStateRef) void {
        self.ops.release(self.ptr);
    }

    pub fn cast(self: ProcessStateRef, comptime T: type) *T {
        return @ptrCast(@alignCast(self.ptr));
    }
};

fn ConditionalFlagFor(comptime enabled: bool) type {
    return if (enabled) struct {
        const Self = @This();

        value: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        pub fn set(self: *Self) void {
            self.value.store(true, .release);
        }

        pub fn clear(self: *Self) void {
            self.value.store(false, .release);
        }

        pub fn isSet(self: *const Self) bool {
            return self.value.load(.acquire);
        }
    } else struct {
        const Self = @This();

        value: bool = false,

        pub inline fn set(self: *Self) void {
            self.value = true;
        }

        pub inline fn clear(self: *Self) void {
            self.value = false;
        }

        pub inline fn isSet(self: *const Self) bool {
            return self.value;
        }
    };
}

pub const ConditionalFlag = ConditionalFlagFor(config.lib_wasi_threads);

pub const AuxiliaryStack = struct {
    bottom: u32,
    top: u32,

    pub fn fromTop(top: u32, size: u32) ?AuxiliaryStack {
        if (size > top) return null;
        return .{ .bottom = top - size, .top = top };
    }
};

/// State that must never be copied from one guest thread to another.
///
/// The operand, call-frame, and label stacks remain owned directly by
/// `ExecEnv`; this structure holds the backend-neutral thread metadata and
/// dynamic host-call state shared by interpreter and AOT entry paths.
pub const ThreadExecutionContext = struct {
    process_state: ?ProcessStateRef = null,
    tid: i32 = 0,
    /// Opaque `start-arg` from the Preview-1 `thread-spawn` ABI. The runtime
    /// preserves its bits and never interprets the wasi-libc payload.
    start_arg: u32 = 0,
    /// Guest TLS base, when a future binding supplies one explicitly.
    /// Preview-1 wasi-libc initializes `__tls_base` from `start_arg` itself.
    tls_base: ?u32 = null,
    auxiliary_stack: ?AuxiliaryStack = null,
    thread_group: ?*anyopaque = null,
    task_manager: ?*anyopaque = null,
    runtime_call_context: ?*anyopaque = null,
    /// Request-scoped host coordination state carried through nested
    /// component calls and async callbacks without storing it globally.
    host_task_context: ?*anyopaque = null,
    backend_context: ?*anyopaque = null,
    implicit_task_context: [task_context_slot_count]u32 =
        [_]u32{0} ** task_context_slot_count,
    cancellation_requested: ConditionalFlag = .{},
    trap_observed: ConditionalFlag = .{},

    pub fn init(process_state: ?ProcessStateRef) ThreadExecutionContext {
        return .{
            .process_state = if (process_state) |state| state.acquire() else null,
        };
    }

    pub fn deinit(self: *ThreadExecutionContext) void {
        if (self.process_state) |state| state.release();
        self.process_state = null;
        self.thread_group = null;
        self.task_manager = null;
        self.runtime_call_context = null;
        self.host_task_context = null;
        self.backend_context = null;
    }

    pub fn replaceProcessState(
        self: *ThreadExecutionContext,
        process_state: ?ProcessStateRef,
    ) void {
        const replacement = if (process_state) |state| state.acquire() else null;
        if (self.process_state) |state| state.release();
        self.process_state = replacement;
    }

    pub fn process(self: *const ThreadExecutionContext, comptime T: type) ?*T {
        const state = self.process_state orelse return null;
        return state.cast(T);
    }

    pub fn configureWasiThread(
        self: *ThreadExecutionContext,
        tid: i32,
        start_arg: u32,
        auxiliary_stack: ?AuxiliaryStack,
    ) void {
        self.tid = tid;
        self.start_arg = start_arg;
        self.auxiliary_stack = auxiliary_stack;
    }

    pub fn setTlsBase(self: *ThreadExecutionContext, tls_base: ?u32) void {
        self.tls_base = tls_base;
    }

    pub fn setThreadGroup(
        self: *ThreadExecutionContext,
        thread_group: ?*anyopaque,
    ) void {
        self.thread_group = thread_group;
    }

    pub fn threadGroup(self: *const ThreadExecutionContext, comptime T: type) ?*T {
        const ptr = self.thread_group orelse return null;
        return @ptrCast(@alignCast(ptr));
    }

    pub fn bindTaskManager(
        self: *ThreadExecutionContext,
        task_manager: ?*anyopaque,
    ) OpaqueBinding {
        const previous = self.task_manager;
        self.task_manager = task_manager;
        return .{ .slot = &self.task_manager, .previous = previous };
    }

    pub fn taskManager(self: *const ThreadExecutionContext, comptime T: type) ?*T {
        const ptr = self.task_manager orelse return null;
        return @ptrCast(@alignCast(ptr));
    }

    pub fn bindRuntimeCallContext(
        self: *ThreadExecutionContext,
        runtime_call_context: ?*anyopaque,
    ) OpaqueBinding {
        const previous = self.runtime_call_context;
        self.runtime_call_context = runtime_call_context;
        return .{ .slot = &self.runtime_call_context, .previous = previous };
    }

    pub fn runtimeCallContext(
        self: *const ThreadExecutionContext,
        comptime T: type,
    ) ?*T {
        const ptr = self.runtime_call_context orelse return null;
        return @ptrCast(@alignCast(ptr));
    }

    pub fn bindHostTaskContext(
        self: *ThreadExecutionContext,
        host_task_context: ?*anyopaque,
    ) OpaqueBinding {
        const previous = self.host_task_context;
        self.host_task_context = host_task_context;
        return .{ .slot = &self.host_task_context, .previous = previous };
    }

    pub fn hostTaskContext(
        self: *const ThreadExecutionContext,
        comptime T: type,
    ) ?*T {
        const ptr = self.host_task_context orelse return null;
        return @ptrCast(@alignCast(ptr));
    }

    pub fn bindBackendContext(
        self: *ThreadExecutionContext,
        backend_context: ?*anyopaque,
    ) OpaqueBinding {
        const previous = self.backend_context;
        self.backend_context = backend_context;
        return .{ .slot = &self.backend_context, .previous = previous };
    }

    pub fn backendContext(
        self: *const ThreadExecutionContext,
        comptime T: type,
    ) ?*T {
        const ptr = self.backend_context orelse return null;
        return @ptrCast(@alignCast(ptr));
    }

    pub fn requestCancellation(self: *ThreadExecutionContext) void {
        self.cancellation_requested.set();
    }

    pub fn clearCancellation(self: *ThreadExecutionContext) void {
        self.cancellation_requested.clear();
    }

    pub fn isCancellationRequested(self: *const ThreadExecutionContext) bool {
        return self.cancellation_requested.isSet();
    }

    pub fn markTrap(self: *ThreadExecutionContext) void {
        self.trap_observed.set();
    }

    pub fn clearTrap(self: *ThreadExecutionContext) void {
        self.trap_observed.clear();
    }

    pub fn hasTrapped(self: *const ThreadExecutionContext) bool {
        return self.trap_observed.isSet();
    }

    pub fn enter(self: *ThreadExecutionContext) ActiveScope {
        const previous = active_context;
        active_context = self;
        return .{ .installed = self, .previous = previous };
    }
};

pub const OpaqueBinding = struct {
    slot: *?*anyopaque,
    previous: ?*anyopaque,

    pub fn deinit(self: OpaqueBinding) void {
        self.slot.* = self.previous;
    }
};

threadlocal var active_context: ?*ThreadExecutionContext = null;

pub const ActiveScope = struct {
    installed: *ThreadExecutionContext,
    previous: ?*ThreadExecutionContext,

    pub fn deinit(self: ActiveScope) void {
        std.debug.assert(active_context == self.installed);
        active_context = self.previous;
    }
};

pub fn current() ?*ThreadExecutionContext {
    return active_context;
}

test "thread execution contexts retain only shared process state" {
    const Tracker = struct {
        refs: usize = 1,

        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.refs += 1;
        }

        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            std.debug.assert(self.refs > 0);
            self.refs -= 1;
        }
    };
    const ops = ProcessStateOps{
        .retain = Tracker.retain,
        .release = Tracker.release,
    };

    var tracker = Tracker{};
    const process_ref = ProcessStateRef.init(@ptrCast(&tracker), &ops);
    var parent = ThreadExecutionContext.init(process_ref);
    defer parent.deinit();
    var child = ThreadExecutionContext.init(parent.process_state);
    defer child.deinit();

    try std.testing.expectEqual(@as(usize, 3), tracker.refs);
    try std.testing.expectEqual(parent.process_state.?.ptr, child.process_state.?.ptr);

    parent.configureWasiThread(1, 0x1234, AuxiliaryStack.fromTop(8192, 4096));
    parent.setTlsBase(0x2000);
    parent.implicit_task_context[0] = 7;
    parent.requestCancellation();
    parent.markTrap();

    child.configureWasiThread(2, 0x5678, AuxiliaryStack.fromTop(16384, 4096));
    child.setTlsBase(0x3000);
    child.implicit_task_context[0] = 9;

    try std.testing.expectEqual(@as(i32, 1), parent.tid);
    try std.testing.expectEqual(@as(i32, 2), child.tid);
    try std.testing.expectEqual(@as(u32, 0x1234), parent.start_arg);
    try std.testing.expectEqual(@as(u32, 0x5678), child.start_arg);
    try std.testing.expectEqual(@as(?u32, 0x2000), parent.tls_base);
    try std.testing.expectEqual(@as(?u32, 0x3000), child.tls_base);
    try std.testing.expectEqual(@as(u32, 7), parent.implicit_task_context[0]);
    try std.testing.expectEqual(@as(u32, 9), child.implicit_task_context[0]);
    try std.testing.expect(parent.isCancellationRequested());
    try std.testing.expect(!child.isCancellationRequested());
    try std.testing.expect(parent.hasTrapped());
    try std.testing.expect(!child.hasTrapped());
}

test "active execution context and opaque bindings restore when nested" {
    var outer = ThreadExecutionContext{};
    var inner = ThreadExecutionContext{};
    var outer_task: u8 = 1;
    var inner_task: u8 = 2;
    var outer_host: u8 = 3;
    var inner_host: u8 = 4;

    var outer_scope = outer.enter();
    defer outer_scope.deinit();
    try std.testing.expectEqual(&outer, current().?);

    var task_binding = outer.bindTaskManager(@ptrCast(&outer_task));
    defer task_binding.deinit();
    var host_binding = outer.bindHostTaskContext(@ptrCast(&outer_host));
    defer host_binding.deinit();
    try std.testing.expectEqual(&outer_task, outer.taskManager(u8).?);
    try std.testing.expectEqual(&outer_host, outer.hostTaskContext(u8).?);

    {
        var inner_scope = inner.enter();
        defer inner_scope.deinit();
        var inner_binding = inner.bindTaskManager(@ptrCast(&inner_task));
        defer inner_binding.deinit();
        var inner_host_binding = inner.bindHostTaskContext(
            @ptrCast(&inner_host),
        );
        defer inner_host_binding.deinit();
        try std.testing.expectEqual(&inner, current().?);
        try std.testing.expectEqual(&inner_task, inner.taskManager(u8).?);
        try std.testing.expectEqual(&inner_host, inner.hostTaskContext(u8).?);
    }

    try std.testing.expectEqual(&outer, current().?);
    try std.testing.expectEqual(&outer_task, outer.taskManager(u8).?);
    try std.testing.expectEqual(&outer_host, outer.hostTaskContext(u8).?);
}
