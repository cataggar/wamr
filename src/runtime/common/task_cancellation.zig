//! Task-owned cancellation source with lifetime-safe wake registrations.
//!
//! Component tasks can span several ComponentInstances, each with its own
//! ThreadManager. A Source is owned by the task and retained by every
//! registered manager-local group. Cancellation latches once, then fans out
//! synchronously to every registration. Register/cancel/unregister serialize
//! under one lock, so a target is either present for the cancel sweep or sees
//! the latched state and is woken before registration returns.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("../../config.zig");

const SpinMutex = struct {
    state: std.atomic.Value(u8) = .init(0),

    fn lock(self: *SpinMutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null)
            std.atomic.spinLoopHint();
    }

    fn unlock(self: *SpinMutex) void {
        self.state.store(0, .release);
    }
};

var next_source_id: std.atomic.Value(u32) = .init(1);

fn allocateSourceId() u32 {
    var current = next_source_id.load(.acquire);
    while (true) {
        if (current == std.math.maxInt(u32))
            @panic("task cancellation source identity exhausted");
        current = next_source_id.cmpxchgWeak(
            current,
            current + 1,
            .acq_rel,
            .acquire,
        ) orelse return current;
    }
}

pub const WakeTarget = struct {
    ctx: *anyopaque,
    wake: *const fn (*anyopaque) void,

    fn invoke(self: WakeTarget) void {
        self.wake(self.ctx);
    }
};

pub const RegisterResult = enum {
    registered,
    cancelled,
};

pub const Registration = struct {
    source: ?*Source = null,
    next: ?*Registration = null,
    target: WakeTarget = undefined,

    pub fn unregister(self: *Registration) void {
        const source = self.source orelse return;
        source.unregister(self);
    }
};

pub const Source = struct {
    allocator: std.mem.Allocator,
    id: u32,
    refs: std.atomic.Value(u32) = .init(1),
    cancelled: std.atomic.Value(bool) = .init(false),
    mutex: if (config.lib_wasi_threads) SpinMutex else void =
        if (config.lib_wasi_threads) .{} else {},
    registrations: if (config.lib_wasi_threads) ?*Registration else void =
        if (config.lib_wasi_threads) null else {},
    register_test_hook: if (builtin.is_test) ?*RegisterTestHook else void =
        if (builtin.is_test) null else {},

    /// Durable observation handle for one task cancellation generation.
    ///
    /// A ticket owns one source reference, may outlive the task, frame, and
    /// TaskManager that created it, and never retains a raw Task pointer. It
    /// is a linear owner despite Zig permitting bitwise copies: ordinary
    /// assignment is unsupported. Transfer ownership with `take`, create an
    /// additional owner with `clone`, and release each owner with `deinit`.
    pub const Ticket = struct {
        source: ?*Source = null,

        pub const Error = error{InactiveTicket};

        /// Transfer this ticket's one owned reference to the result. Repeated
        /// calls return an inert ticket and never touch the source.
        pub fn take(self: *Ticket) Ticket {
            const source = self.source orelse return .{};
            self.source = null;
            return .{ .source = source };
        }

        /// Create an independent owner. This is the only supported operation
        /// that adds a ticket owner/reference.
        pub fn clone(self: *const Ticket) Error!Ticket {
            const source = self.source orelse return error.InactiveTicket;
            return source.acquire();
        }

        pub fn isCancelled(self: *const Ticket) Error!bool {
            const source = self.source orelse return error.InactiveTicket;
            return source.isCancelled();
        }

        pub fn identity(self: *const Ticket) Error!u32 {
            const source = self.source orelse return error.InactiveTicket;
            return source.id;
        }

        pub fn isActive(self: *const Ticket) bool {
            return self.source != null;
        }

        /// Internal bridge for thread wake registration. Callers must not
        /// retain the returned pointer beyond this ticket's lifetime.
        pub fn sourceForRegistration(self: *const Ticket) Error!*Source {
            return self.source orelse error.InactiveTicket;
        }

        pub fn deinit(self: *Ticket) void {
            const source = self.source orelse return;
            self.source = null;
            source.release();
        }
    };

    pub fn create(allocator: std.mem.Allocator) !*Source {
        const self = try allocator.create(Source);
        self.* = .{
            .allocator = allocator,
            .id = allocateSourceId(),
        };
        return self;
    }

    pub fn acquire(self: *Source) Ticket {
        const previous = self.refs.fetchAdd(1, .acq_rel);
        std.debug.assert(previous > 0 and previous < std.math.maxInt(u32));
        return .{ .source = self };
    }

    pub fn release(self: *Source) void {
        const previous = self.refs.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
        if (previous != 1) return;
        if (comptime config.lib_wasi_threads)
            std.debug.assert(self.registrations == null);
        self.allocator.destroy(self);
    }

    pub fn isCancelled(self: *const Source) bool {
        return self.cancelled.load(.acquire);
    }

    pub fn register(
        self: *Source,
        registration: *Registration,
        target: WakeTarget,
    ) RegisterResult {
        if (comptime !config.lib_wasi_threads) {
            return .cancelled;
        }
        self.mutex.lock();
        if (comptime builtin.is_test) {
            if (self.register_test_hook) |hook| {
                hook.reached.store(true, .release);
                while (!hook.resume_flag.load(.acquire))
                    std.atomic.spinLoopHint();
            }
        }
        std.debug.assert(registration.source == null);
        std.debug.assert(registration.next == null);
        if (self.cancelled.load(.acquire)) {
            self.mutex.unlock();
            target.invoke();
            return .cancelled;
        }

        _ = self.acquire();
        registration.source = self;
        registration.target = target;
        registration.next = self.registrations;
        self.registrations = registration;
        self.mutex.unlock();
        return .registered;
    }

    /// Latch cancellation and synchronously wake every registered target.
    /// Repeated calls re-arm the wakeups without changing terminal state.
    pub fn cancel(self: *Source) void {
        if (comptime !config.lib_wasi_threads) {
            self.cancelled.store(true, .release);
            return;
        }
        self.mutex.lock();
        self.cancelled.store(true, .release);
        var registration = self.registrations;
        while (registration) |current| : (registration = current.next)
            current.target.invoke();
        self.mutex.unlock();
    }

    fn unregister(self: *Source, registration: *Registration) void {
        if (comptime !config.lib_wasi_threads) {
            return;
        }
        self.mutex.lock();
        if (registration.source != self) {
            self.mutex.unlock();
            return;
        }
        var link = &self.registrations;
        while (link.*) |candidate| {
            if (candidate == registration) {
                link.* = candidate.next;
                registration.source = null;
                registration.next = null;
                self.mutex.unlock();
                self.release();
                return;
            }
            link = &candidate.next;
        }
        self.mutex.unlock();
        unreachable;
    }

    pub fn registrationCount(self: *Source) usize {
        if (comptime !config.lib_wasi_threads) {
            return 0;
        }
        self.mutex.lock();
        defer self.mutex.unlock();
        var count: usize = 0;
        var registration = self.registrations;
        while (registration) |current| : (registration = current.next)
            count += 1;
        return count;
    }
};

const RegisterTestHook = struct {
    reached: std.atomic.Value(bool) = .init(false),
    resume_flag: std.atomic.Value(bool) = .init(false),
};

test "task cancellation ticket: take transfers one owner and leaves source inert" {
    const source = try Source.create(std.testing.allocator);
    var ticket = source.acquire();
    var transferred = ticket.take();
    var repeated = ticket.take();

    try std.testing.expect(!ticket.isActive());
    try std.testing.expect(!repeated.isActive());
    try std.testing.expectError(error.InactiveTicket, ticket.clone());
    try std.testing.expectError(error.InactiveTicket, ticket.isCancelled());
    try std.testing.expectError(error.InactiveTicket, ticket.identity());
    try std.testing.expectError(
        error.InactiveTicket,
        ticket.sourceForRegistration(),
    );

    source.cancel();
    try std.testing.expect(try transferred.isCancelled());
    source.release();
    ticket.deinit();
    ticket.deinit();
    repeated.deinit();
    transferred.deinit();
}

test "task cancellation ticket: cancellation before take remains observable" {
    const source = try Source.create(std.testing.allocator);
    var ticket = source.acquire();
    source.cancel();
    var transferred = ticket.take();
    source.release();

    try std.testing.expect(try transferred.isCancelled());
    transferred.deinit();
}

test "task cancellation ticket: clone creates an independent reference" {
    const source = try Source.create(std.testing.allocator);
    var ticket = source.acquire();
    var clone = try ticket.clone();
    try std.testing.expectEqual(@as(u32, 3), source.refs.load(.acquire));

    source.release();
    ticket.deinit();
    try std.testing.expect(clone.isActive());
    clone.deinit();
}

test "task cancellation ticket: raw value copy is not a retained owner" {
    const source = try Source.create(std.testing.allocator);
    defer source.release();
    var ticket = source.acquire();
    defer ticket.deinit();
    var unsupported_copy = ticket;
    try std.testing.expectEqual(@as(u32, 2), source.refs.load(.acquire));

    // Direct copies are intentionally unsupported: disarm this test-only
    // copy rather than releasing the same reference twice.
    unsupported_copy.source = null;
    unsupported_copy.deinit();
}

test "task cancellation: register racing a latched cancel wakes immediately" {
    if (comptime !config.lib_wasi_threads) return error.SkipZigTest;
    const Counter = struct {
        hits: usize = 0,

        fn wake(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.hits += 1;
        }
    };

    const source = try Source.create(std.testing.allocator);
    defer source.release();
    source.cancel();
    var counter = Counter{};
    var registration = Registration{};
    try std.testing.expectEqual(
        RegisterResult.cancelled,
        source.register(&registration, .{
            .ctx = @ptrCast(&counter),
            .wake = Counter.wake,
        }),
    );
    try std.testing.expectEqual(@as(usize, 1), counter.hits);
    try std.testing.expectEqual(@as(usize, 0), source.registrationCount());
}

test "task cancellation: registrations retain source through owner release" {
    if (comptime !config.lib_wasi_threads) return error.SkipZigTest;
    const Counter = struct {
        hits: usize = 0,

        fn wake(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.hits += 1;
        }
    };

    const source = try Source.create(std.testing.allocator);
    var counter = Counter{};
    var registration = Registration{};
    try std.testing.expectEqual(
        RegisterResult.registered,
        source.register(&registration, .{
            .ctx = @ptrCast(&counter),
            .wake = Counter.wake,
        }),
    );
    source.release();
    source.cancel();
    try std.testing.expectEqual(@as(usize, 1), counter.hits);
    registration.unregister();
}

test "task cancellation: cancel begun during registration cannot miss target" {
    if (builtin.single_threaded or !config.lib_wasi_threads)
        return error.SkipZigTest;
    const Counter = struct {
        hits: std.atomic.Value(usize) = .init(0),

        fn wake(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.hits.fetchAdd(1, .acq_rel);
        }
    };
    const RegisterCtx = struct {
        source: *Source,
        registration: *Registration,
        counter: *Counter,
        result: RegisterResult = .cancelled,

        fn run(self: *@This()) void {
            self.result = self.source.register(self.registration, .{
                .ctx = @ptrCast(self.counter),
                .wake = Counter.wake,
            });
        }
    };
    const CancelCtx = struct {
        source: *Source,
        started: *std.atomic.Value(bool),

        fn run(self: @This()) void {
            self.started.store(true, .release);
            self.source.cancel();
        }
    };

    const source = try Source.create(std.testing.allocator);
    defer source.release();
    var hook = RegisterTestHook{};
    source.register_test_hook = &hook;
    var counter = Counter{};
    var registration = Registration{};
    var register_ctx = RegisterCtx{
        .source = source,
        .registration = &registration,
        .counter = &counter,
    };
    const register_thread = try std.Thread.spawn(.{}, RegisterCtx.run, .{&register_ctx});
    while (!hook.reached.load(.acquire)) std.atomic.spinLoopHint();

    var cancel_started = std.atomic.Value(bool).init(false);
    const cancel_thread = try std.Thread.spawn(.{}, CancelCtx.run, .{CancelCtx{
        .source = source,
        .started = &cancel_started,
    }});
    while (!cancel_started.load(.acquire)) std.atomic.spinLoopHint();
    hook.resume_flag.store(true, .release);
    register_thread.join();
    cancel_thread.join();

    try std.testing.expectEqual(RegisterResult.registered, register_ctx.result);
    try std.testing.expect(source.isCancelled());
    try std.testing.expectEqual(@as(usize, 1), counter.hits.load(.acquire));
    registration.unregister();
}

test "task cancellation: unregister racing cancel keeps target lifetime safe" {
    if (builtin.single_threaded or !config.lib_wasi_threads)
        return error.SkipZigTest;
    const Counter = struct {
        hits: std.atomic.Value(usize) = .init(0),

        fn wake(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.hits.fetchAdd(1, .acq_rel);
        }
    };
    const CancelCtx = struct {
        source: *Source,
        gate: *std.atomic.Value(bool),

        fn run(self: @This()) void {
            while (!self.gate.load(.acquire)) std.atomic.spinLoopHint();
            self.source.cancel();
        }
    };
    const UnregisterCtx = struct {
        registration: *Registration,
        gate: *std.atomic.Value(bool),

        fn run(self: @This()) void {
            while (!self.gate.load(.acquire)) std.atomic.spinLoopHint();
            self.registration.unregister();
        }
    };

    var round: usize = 0;
    while (round < 256) : (round += 1) {
        const source = try Source.create(std.testing.allocator);
        defer source.release();
        var counter = Counter{};
        var registration = Registration{};
        try std.testing.expectEqual(
            RegisterResult.registered,
            source.register(&registration, .{
                .ctx = @ptrCast(&counter),
                .wake = Counter.wake,
            }),
        );
        var gate = std.atomic.Value(bool).init(false);
        const cancel_thread = try std.Thread.spawn(.{}, CancelCtx.run, .{CancelCtx{
            .source = source,
            .gate = &gate,
        }});
        const unregister_thread = try std.Thread.spawn(
            .{},
            UnregisterCtx.run,
            .{UnregisterCtx{
                .registration = &registration,
                .gate = &gate,
            }},
        );
        gate.store(true, .release);
        cancel_thread.join();
        unregister_thread.join();
        try std.testing.expect(source.isCancelled());
        try std.testing.expect(counter.hits.load(.acquire) <= 1);
        try std.testing.expectEqual(@as(usize, 0), source.registrationCount());
    }
}
