//! Backend-neutral parking for WebAssembly atomic wait/notify.
//!
//! Waiters are keyed by the guest address. The value comparison and queue
//! insertion happen while holding the same bucket lock, which closes the
//! notify-before-enqueue race. Each waiter has a private OS wait word so an
//! exact number of matching waiters can be selected and woken.

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("platform.zig");

const is_linux = builtin.os.tag == .linux;
const is_macos = builtin.os.tag == .macos;
const is_windows = builtin.os.tag == .windows;
const parking_supported = !builtin.single_threaded and (is_linux or is_macos or is_windows);

pub const WaitResult = enum(u32) {
    notified = 0,
    not_equal = 1,
    timed_out = 2,
    cancelled = 3,
    closed = 4,
};

pub const BackendError = error{
    InvalidAddress,
    InvalidArgument,
    Unsupported,
    SystemFailure,
};

const OsWaitResult = enum {
    awakened,
    timed_out,
    spurious,
};

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

const Outcome = enum(u32) {
    waiting,
    notified,
    cancelled,
    closed,
};

const Waiter = struct {
    next: ?*Waiter = null,
    key: usize,
    outcome: std.atomic.Value(u32) = .init(@intFromEnum(Outcome.waiting)),
};

const Bucket = struct {
    mutex: SpinMutex = .{},
    head: ?*Waiter = null,
};

/// A keyed parking lot with explicit cancellation and quiescent shutdown.
///
/// `deinit` first prevents new waits, then wakes every queued waiter and
/// waits until all active `wait32`/`wait64` calls have returned. The object
/// may therefore be embedded in another refcounted control block.
pub const ParkingLot = struct {
    const bucket_count = 64;
    const closing_bit: u32 = 1 << 31;
    const active_mask: u32 = closing_bit - 1;

    buckets: [bucket_count]Bucket = @splat(.{}),
    lifecycle: std.atomic.Value(u32) = .init(0),

    pub fn init() ParkingLot {
        return .{};
    }

    pub fn wait32(
        self: *ParkingLot,
        address: *align(@alignOf(u32)) const u32,
        expected: u32,
        timeout_ns: i64,
    ) BackendError!WaitResult {
        if (comptime parking_supported) {
            return self.waitValue(u32, address, expected, timeout_ns);
        } else {
            return error.Unsupported;
        }
    }

    pub fn wait64(
        self: *ParkingLot,
        address: *align(@alignOf(u64)) const u64,
        expected: u64,
        timeout_ns: i64,
    ) BackendError!WaitResult {
        if (comptime parking_supported and @bitSizeOf(usize) >= 64) {
            return self.waitValue(u64, address, expected, timeout_ns);
        } else {
            return error.Unsupported;
        }
    }

    fn waitValue(
        self: *ParkingLot,
        comptime T: type,
        address: *align(@alignOf(T)) const T,
        expected: T,
        timeout_ns: i64,
    ) BackendError!WaitResult {
        if (!self.enter()) return .closed;
        defer self.leave();

        const key = @intFromPtr(address);
        const bucket = self.bucketFor(key);
        var waiter = Waiter{ .key = key };

        bucket.mutex.lock();
        if (self.lifecycle.load(.acquire) & closing_bit != 0) {
            bucket.mutex.unlock();
            return .closed;
        }
        if (@atomicLoad(T, address, .seq_cst) != expected) {
            bucket.mutex.unlock();
            return .not_equal;
        }
        if (timeout_ns == 0) {
            bucket.mutex.unlock();
            return .timed_out;
        }
        waiter.next = bucket.head;
        bucket.head = &waiter;
        bucket.mutex.unlock();

        const deadline_ns: ?u64 = if (timeout_ns < 0)
            null
        else
            std.math.add(u64, monotonicNowNs(), @intCast(timeout_ns)) catch
                std.math.maxInt(u64);

        while (true) {
            const outcome: Outcome = @enumFromInt(waiter.outcome.load(.acquire));
            if (outcome != .waiting) return outcomeToResult(outcome);

            const remaining_ns: ?u64 = if (deadline_ns) |deadline| remaining: {
                const now = monotonicNowNs();
                if (now >= deadline) {
                    bucket.mutex.lock();
                    const final = self.removeTimedOutLocked(bucket, &waiter);
                    bucket.mutex.unlock();
                    return final;
                }
                break :remaining deadline - now;
            } else null;

            const os_result = osWait(
                &waiter.outcome.raw,
                @intFromEnum(Outcome.waiting),
                remaining_ns,
            ) catch |err| {
                bucket.mutex.lock();
                const final_outcome: Outcome = @enumFromInt(waiter.outcome.load(.acquire));
                if (final_outcome != .waiting) {
                    bucket.mutex.unlock();
                    return outcomeToResult(final_outcome);
                }
                self.removeLocked(bucket, &waiter);
                waiter.outcome.store(@intFromEnum(Outcome.cancelled), .release);
                bucket.mutex.unlock();
                return err;
            };
            switch (os_result) {
                .awakened, .spurious => continue,
                .timed_out => {
                    if (deadline_ns) |deadline| {
                        if (monotonicNowNs() < deadline) continue;
                    }
                    bucket.mutex.lock();
                    const final = self.removeTimedOutLocked(bucket, &waiter);
                    bucket.mutex.unlock();
                    return final;
                },
            }
        }
    }

    fn removeTimedOutLocked(self: *ParkingLot, bucket: *Bucket, waiter: *Waiter) WaitResult {
        const outcome: Outcome = @enumFromInt(waiter.outcome.load(.acquire));
        if (outcome != .waiting) return outcomeToResult(outcome);

        self.removeLocked(bucket, waiter);
        waiter.outcome.store(@intFromEnum(Outcome.cancelled), .release);
        return .timed_out;
    }

    fn removeLocked(self: *ParkingLot, bucket: *Bucket, waiter: *Waiter) void {
        _ = self;
        var link = &bucket.head;
        while (link.*) |candidate| {
            if (candidate == waiter) {
                link.* = candidate.next;
                return;
            }
            link = &candidate.next;
        }
    }

    /// Wake up to `count` waiters at exactly `address`.
    pub fn notify(self: *ParkingLot, address: *const anyopaque, count: u32) BackendError!u32 {
        if (comptime parking_supported) {
            if (count == 0) return 0;
            return self.wakeKey(@intFromPtr(address), count, .notified);
        } else {
            return error.Unsupported;
        }
    }

    /// Cancel every waiter at exactly `address`.
    pub fn cancel(self: *ParkingLot, address: *const anyopaque) BackendError!u32 {
        if (comptime parking_supported) {
            return self.wakeKey(@intFromPtr(address), std.math.maxInt(u32), .cancelled);
        } else {
            return error.Unsupported;
        }
    }

    /// Cancel all waiters. This is the group-cancellation primitive.
    pub fn cancelAll(self: *ParkingLot) BackendError!u32 {
        if (comptime !parking_supported) {
            return error.Unsupported;
        }
        var total: u32 = 0;
        for (0..bucket_count) |i| {
            const bucket = &self.buckets[i];
            bucket.mutex.lock();
            while (bucket.head) |w| {
                bucket.head = w.next;
                w.outcome.store(@intFromEnum(Outcome.cancelled), .release);
                osWake(&w.outcome.raw) catch |err| {
                    bucket.mutex.unlock();
                    return err;
                };
                total +|= 1;
            }
            bucket.mutex.unlock();
        }
        return total;
    }

    fn wakeKey(self: *ParkingLot, key: usize, count: u32, outcome: Outcome) BackendError!u32 {
        const bucket = self.bucketFor(key);
        bucket.mutex.lock();
        defer bucket.mutex.unlock();

        var woken: u32 = 0;
        var link = &bucket.head;
        while (link.*) |waiter| {
            if (waiter.key != key or woken >= count) {
                link = &waiter.next;
                continue;
            }
            link.* = waiter.next;
            waiter.outcome.store(@intFromEnum(outcome), .release);
            try osWake(&waiter.outcome.raw);
            woken += 1;
        }
        return woken;
    }

    /// Number of currently queued waiters for a key. Intended for
    /// diagnostics and deterministic native tests.
    pub fn waiterCount(self: *ParkingLot, address: *const anyopaque) u32 {
        if (comptime !parking_supported) {
            return 0;
        }
        const key = @intFromPtr(address);
        const bucket = self.bucketFor(key);
        bucket.mutex.lock();
        defer bucket.mutex.unlock();

        var count: u32 = 0;
        var waiter = bucket.head;
        while (waiter) |w| : (waiter = w.next) {
            if (w.key == key) count += 1;
        }
        return count;
    }

    /// Wake all waiters and wait for their stack-allocated queue nodes to
    /// leave the parking lot.
    pub fn deinit(self: *ParkingLot) void {
        if (comptime !parking_supported) {
            return;
        }
        _ = self.lifecycle.fetchOr(closing_bit, .acq_rel);
        self.closeAll();

        while (self.lifecycle.load(.acquire) & active_mask != 0) {
            const observed = self.lifecycle.load(.acquire);
            _ = osWait(&self.lifecycle.raw, observed, null) catch |err| switch (err) {
                error.InvalidAddress,
                error.InvalidArgument,
                error.Unsupported,
                error.SystemFailure,
                => platform.usleep(100),
            };
        }
    }

    fn closeAll(self: *ParkingLot) void {
        for (0..bucket_count) |i| {
            const bucket = &self.buckets[i];
            bucket.mutex.lock();
            var waiter = bucket.head;
            bucket.head = null;
            while (waiter) |w| {
                waiter = w.next;
                w.outcome.store(@intFromEnum(Outcome.closed), .release);
                osWake(&w.outcome.raw) catch |err| switch (err) {
                    error.InvalidAddress,
                    error.InvalidArgument,
                    error.Unsupported,
                    error.SystemFailure,
                    => {},
                };
            }
            bucket.mutex.unlock();
        }
    }

    fn enter(self: *ParkingLot) bool {
        var state = self.lifecycle.load(.acquire);
        while (true) {
            if (state & closing_bit != 0) return false;
            if (state & active_mask == active_mask) return false;
            state = self.lifecycle.cmpxchgWeak(
                state,
                state + 1,
                .acquire,
                .monotonic,
            ) orelse return true;
        }
    }

    fn leave(self: *ParkingLot) void {
        const old = self.lifecycle.fetchSub(1, .release);
        if (old == closing_bit | 1) {
            _ = self.lifecycle.load(.acquire);
            osWakeAll(&self.lifecycle.raw) catch |err| switch (err) {
                error.InvalidAddress,
                error.InvalidArgument,
                error.Unsupported,
                error.SystemFailure,
                => {},
            };
        }
    }

    fn bucketFor(self: *ParkingLot, key: usize) *Bucket {
        const multiplier: usize = @truncate(0x9E3779B97F4A7C15);
        const hash = key *% multiplier;
        const shift = @bitSizeOf(usize) - @ctz(@as(usize, bucket_count));
        return &self.buckets[hash >> shift];
    }
};

fn outcomeToResult(outcome: Outcome) WaitResult {
    return switch (outcome) {
        .waiting => unreachable,
        .notified => .notified,
        .cancelled => .cancelled,
        .closed => .closed,
    };
}

fn monotonicNowNs() u64 {
    return std.math.mul(u64, platform.timeGetBootUs(), std.time.ns_per_us) catch
        std.math.maxInt(u64);
}

fn osWait(word: *const u32, expected: u32, timeout_ns: ?u64) BackendError!OsWaitResult {
    if (comptime is_linux) return linuxWait(word, expected, timeout_ns);
    if (comptime is_macos) return macosWait(word, expected, timeout_ns);
    if (comptime is_windows) return windowsWait(word, expected, timeout_ns);
    return error.Unsupported;
}

fn osWake(word: *const u32) BackendError!void {
    if (comptime is_linux) return linuxWake(word, 1);
    if (comptime is_macos) return macosWake(word, false);
    if (comptime is_windows) {
        std.os.windows.ntdll.RtlWakeAddressSingle(word);
        return;
    }
    return error.Unsupported;
}

fn osWakeAll(word: *const u32) BackendError!void {
    if (comptime is_linux) return linuxWake(word, std.math.maxInt(i32));
    if (comptime is_macos) return macosWake(word, true);
    if (comptime is_windows) {
        std.os.windows.ntdll.RtlWakeAddressAll(word);
        return;
    }
    return error.Unsupported;
}

fn linuxWait(word: *const u32, expected: u32, timeout_ns: ?u64) BackendError!OsWaitResult {
    if (!is_linux) unreachable;
    const linux = std.os.linux;
    var ts_buffer: linux.timespec = undefined;
    const ts: ?*linux.timespec = if (timeout_ns) |ns| ts: {
        if (ns == 0) return .timed_out;
        ts_buffer = .{
            .sec = @intCast(ns / std.time.ns_per_s),
            .nsec = @intCast(ns % std.time.ns_per_s),
        };
        break :ts &ts_buffer;
    } else null;
    const rc = linux.futex_4arg(word, .{ .cmd = .WAIT, .private = true }, expected, ts);
    return mapLinuxWaitErrno(linux.errno(rc));
}

fn mapLinuxWaitErrno(err: std.os.linux.E) BackendError!OsWaitResult {
    return switch (err) {
        .SUCCESS => .awakened,
        .AGAIN, .INTR => .spurious,
        .TIMEDOUT => .timed_out,
        .FAULT => error.InvalidAddress,
        .INVAL => error.InvalidArgument,
        .NOSYS => error.Unsupported,
        else => error.SystemFailure,
    };
}

fn linuxWake(word: *const u32, count: u32) BackendError!void {
    if (!is_linux) unreachable;
    const linux = std.os.linux;
    const rc = linux.futex_3arg(word, .{ .cmd = .WAKE, .private = true }, count);
    switch (linux.errno(rc)) {
        .SUCCESS => {},
        .FAULT => return error.InvalidAddress,
        .INVAL => return error.InvalidArgument,
        .NOSYS => return error.Unsupported,
        else => return error.SystemFailure,
    }
}

fn macosWait(word: *const u32, expected: u32, timeout_ns: ?u64) BackendError!OsWaitResult {
    if (!is_macos) unreachable;
    const flags: std.c.UL = .{ .op = .COMPARE_AND_WAIT, .NO_ERRNO = true };
    const timeout_us: u32 = if (timeout_ns) |ns|
        @intCast(@min(
            @max(@as(u64, 1), (ns +| (std.time.ns_per_us - 1)) / std.time.ns_per_us),
            std.math.maxInt(u32),
        ))
    else
        0;
    const status = std.c.__ulock_wait(flags, word, expected, timeout_us);
    if (status >= 0) return .awakened;
    return mapDarwinWaitErrno(@enumFromInt(-status));
}

fn mapDarwinWaitErrno(err: std.c.E) BackendError!OsWaitResult {
    return switch (err) {
        .INTR, .CANCELED, .FAULT => .spurious,
        .TIMEDOUT => .timed_out,
        .INVAL => error.InvalidArgument,
        .NOSYS => error.Unsupported,
        else => error.SystemFailure,
    };
}

fn macosWake(word: *const u32, all: bool) BackendError!void {
    if (!is_macos) unreachable;
    const flags: std.c.UL = .{
        .op = .COMPARE_AND_WAIT,
        .NO_ERRNO = true,
        .WAKE_ALL = all,
    };
    while (true) {
        const status = std.c.__ulock_wake(flags, word, 0);
        if (status >= 0) return;
        switch (@as(std.c.E, @enumFromInt(-status))) {
            .INTR, .CANCELED => continue,
            .NOENT => return,
            .FAULT => return error.InvalidAddress,
            .INVAL => return error.InvalidArgument,
            .NOSYS => return error.Unsupported,
            else => return error.SystemFailure,
        }
    }
}

fn windowsWait(word: *const u32, expected: u32, timeout_ns: ?u64) BackendError!OsWaitResult {
    if (!is_windows) unreachable;
    const windows = std.os.windows;
    var expected_copy = expected;
    var timeout: windows.LARGE_INTEGER = undefined;
    const timeout_ptr: ?*const windows.LARGE_INTEGER = if (timeout_ns) |ns| timeout: {
        if (ns == 0) return .timed_out;
        const ticks = @min(
            (ns +| 99) / 100,
            @as(u64, @intCast(std.math.maxInt(i64))),
        );
        timeout = -@as(i64, @intCast(@max(ticks, 1)));
        break :timeout &timeout;
    } else null;
    const status = windows.ntdll.RtlWaitOnAddress(
        word,
        &expected_copy,
        @sizeOf(u32),
        timeout_ptr,
    );
    return mapWindowsWaitStatus(status);
}

fn mapWindowsWaitStatus(status: std.os.windows.NTSTATUS) BackendError!OsWaitResult {
    return switch (status) {
        .SUCCESS => .awakened,
        .TIMEOUT => .timed_out,
        .ALERTED, .USER_APC => .spurious,
        .ACCESS_VIOLATION => error.InvalidAddress,
        .INVALID_PARAMETER => error.InvalidArgument,
        .NOT_IMPLEMENTED => error.Unsupported,
        else => error.SystemFailure,
    };
}

const WaitThreadCtx = struct {
    lot: *ParkingLot,
    word32: ?*align(4) u32 = null,
    word64: ?*align(8) u64 = null,
    expected32: u32 = 7,
    expected64: u64 = 11,
    result: *WaitResult,
};

fn waitThread(ctx: WaitThreadCtx) void {
    ctx.result.* = if (ctx.word32) |word|
        ctx.lot.wait32(word, ctx.expected32, -1) catch |err| switch (err) {
            error.InvalidAddress,
            error.InvalidArgument,
            error.Unsupported,
            error.SystemFailure,
            => .closed,
        }
    else
        ctx.lot.wait64(ctx.word64.?, ctx.expected64, -1) catch |err| switch (err) {
            error.InvalidAddress,
            error.InvalidArgument,
            error.Unsupported,
            error.SystemFailure,
            => .closed,
        };
}

fn waitUntilQueued(lot: *ParkingLot, address: *const anyopaque, count: u32) !void {
    const deadline = monotonicNowNs() +| (2 * std.time.ns_per_s);
    while (lot.waiterCount(address) != count) {
        if (monotonicNowNs() >= deadline) return error.TestExpectedEqual;
        platform.usleep(50);
    }
}

test "ParkingLot: wait mismatch does not enqueue" {
    var lot = ParkingLot.init();
    defer lot.deinit();
    var word: u32 align(4) = 9;
    try std.testing.expectEqual(WaitResult.not_equal, try lot.wait32(&word, 7, -1));
    try std.testing.expectEqual(@as(u32, 0), lot.waiterCount(&word));
}

test "ParkingLot: wait32 and wait64 use monotonic accurate timeouts" {
    var lot = ParkingLot.init();
    defer lot.deinit();
    var word32: u32 align(4) = 7;
    var word64: u64 align(8) = 11;

    const lower = 10 * std.time.ns_per_ms;
    const upper = 1 * std.time.ns_per_s;
    const timeout = 30 * std.time.ns_per_ms;

    var start = monotonicNowNs();
    try std.testing.expectEqual(WaitResult.timed_out, try lot.wait32(&word32, 7, timeout));
    var elapsed = monotonicNowNs() - start;
    try std.testing.expect(elapsed >= lower and elapsed <= upper);

    start = monotonicNowNs();
    try std.testing.expectEqual(WaitResult.timed_out, try lot.wait64(&word64, 11, timeout));
    elapsed = monotonicNowNs() - start;
    try std.testing.expect(elapsed >= lower and elapsed <= upper);
}

test "ParkingLot: notify returns exact counts" {
    var lot = ParkingLot.init();
    defer lot.deinit();
    var word: u32 align(4) = 7;
    var results: [4]WaitResult = @splat(.closed);
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(.{}, waitThread, .{WaitThreadCtx{
            .lot = &lot,
            .word32 = &word,
            .result = &results[i],
        }});
    }
    try waitUntilQueued(&lot, &word, 4);
    try std.testing.expectEqual(@as(u32, 2), try lot.notify(&word, 2));
    try std.testing.expectEqual(@as(u32, 2), try lot.notify(&word, 20));
    try std.testing.expectEqual(@as(u32, 0), try lot.notify(&word, 1));
    for (threads) |thread| thread.join();
    for (results) |result| try std.testing.expectEqual(WaitResult.notified, result);
}

test "ParkingLot: no lost wakeup under stress" {
    var lot = ParkingLot.init();
    defer lot.deinit();
    var word: u32 align(4) = 7;

    for (0..500) |_| {
        @atomicStore(u32, &word, 7, .seq_cst);
        var result: WaitResult = .closed;
        const thread = try std.Thread.spawn(.{}, waitThread, .{WaitThreadCtx{
            .lot = &lot,
            .word32 = &word,
            .result = &result,
        }});
        @atomicStore(u32, &word, 9, .seq_cst);
        const woken = try lot.notify(&word, 1);
        thread.join();
        try std.testing.expect(woken <= 1);
        try std.testing.expect(result == .notified or result == .not_equal);
    }
}

test "ParkingLot: cancellation wakes waiters" {
    var lot = ParkingLot.init();
    defer lot.deinit();
    var word: u64 align(8) = 11;
    var result: WaitResult = .closed;
    const thread = try std.Thread.spawn(.{}, waitThread, .{WaitThreadCtx{
        .lot = &lot,
        .word64 = &word,
        .result = &result,
    }});
    try waitUntilQueued(&lot, &word, 1);
    try std.testing.expectEqual(@as(u32, 1), try lot.cancelAll());
    thread.join();
    try std.testing.expectEqual(WaitResult.cancelled, result);
}

test "ParkingLot: deinit wakes and drains waiters" {
    var lot = ParkingLot.init();
    var word: u32 align(4) = 7;
    var result: WaitResult = .not_equal;
    const thread = try std.Thread.spawn(.{}, waitThread, .{WaitThreadCtx{
        .lot = &lot,
        .word32 = &word,
        .result = &result,
    }});
    try waitUntilQueued(&lot, &word, 1);
    lot.deinit();
    thread.join();
    try std.testing.expectEqual(WaitResult.closed, result);
}

test "ParkingLot: platform error mapping" {
    if (comptime is_linux) {
        try std.testing.expectError(error.InvalidAddress, mapLinuxWaitErrno(.FAULT));
        try std.testing.expectError(error.InvalidArgument, mapLinuxWaitErrno(.INVAL));
        try std.testing.expectError(error.Unsupported, mapLinuxWaitErrno(.NOSYS));
        try std.testing.expectEqual(OsWaitResult.spurious, try mapLinuxWaitErrno(.INTR));
        try std.testing.expectEqual(OsWaitResult.timed_out, try mapLinuxWaitErrno(.TIMEDOUT));
    } else if (comptime is_macos) {
        try std.testing.expectError(error.InvalidArgument, mapDarwinWaitErrno(.INVAL));
        try std.testing.expectError(error.Unsupported, mapDarwinWaitErrno(.NOSYS));
        try std.testing.expectEqual(OsWaitResult.spurious, try mapDarwinWaitErrno(.INTR));
        try std.testing.expectEqual(OsWaitResult.timed_out, try mapDarwinWaitErrno(.TIMEDOUT));
    } else if (comptime is_windows) {
        try std.testing.expectError(error.InvalidAddress, mapWindowsWaitStatus(.ACCESS_VIOLATION));
        try std.testing.expectError(error.InvalidArgument, mapWindowsWaitStatus(.INVALID_PARAMETER));
        try std.testing.expectError(error.Unsupported, mapWindowsWaitStatus(.NOT_IMPLEMENTED));
        try std.testing.expectEqual(OsWaitResult.spurious, try mapWindowsWaitStatus(.ALERTED));
        try std.testing.expectEqual(OsWaitResult.timed_out, try mapWindowsWaitStatus(.TIMEOUT));
    }
}
