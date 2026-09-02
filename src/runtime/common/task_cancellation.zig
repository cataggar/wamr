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
    mutex: SpinMutex = .{},
    registrations: ?*Registration = null,
    register_test_hook: if (builtin.is_test) ?*RegisterTestHook else void =
        if (builtin.is_test) null else {},

    pub const Ref = struct {
        source: *Source,
        active: bool = true,

        pub fn deinit(self: *Ref) void {
            if (!self.active) return;
            self.source.release();
            self.active = false;
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

    pub fn acquire(self: *Source) Ref {
        const previous = self.refs.fetchAdd(1, .acq_rel);
        std.debug.assert(previous > 0 and previous < std.math.maxInt(u32));
        return .{ .source = self };
    }

    pub fn release(self: *Source) void {
        const previous = self.refs.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
        if (previous != 1) return;
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
        self.mutex.lock();
        self.cancelled.store(true, .release);
        var registration = self.registrations;
        while (registration) |current| : (registration = current.next)
            current.target.invoke();
        self.mutex.unlock();
    }

    fn unregister(self: *Source, registration: *Registration) void {
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

test "task cancellation: register racing a latched cancel wakes immediately" {
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
    if (builtin.single_threaded) return error.SkipZigTest;
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
    if (builtin.single_threaded) return error.SkipZigTest;
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
