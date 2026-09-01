//! First-wins terminal outcome for a WASI thread group.
//!
//! A process made of several guest threads has exactly one terminal result:
//! the first thread that calls `proc_exit` or traps establishes it, and every
//! later concurrent termination is recorded as a no-op. The winner's outcome
//! is what the embedder observes, so a trap can never be masked by a racing
//! `proc_exit(0)` (or the other way around).
//!
//! The state also carries the wakeup hook that the thread group installs. The
//! hook fires exactly once, on the claiming thread, as soon as the terminal
//! outcome is published — that is the signal siblings blocked in
//! `memory.atomic.wait`, `poll_oneoff`, or blocking WASI I/O use to unwind
//! promptly instead of running to natural completion.
//!
//! Only 32-bit atomics are used so the state compiles for `wasm32` hosts,
//! and the whole structure degrades to plain fields when threads are off.

const std = @import("std");
const config = @import("config");

pub const Kind = enum(u8) {
    /// `proc_exit(code)` — the guest asked for an orderly exit.
    exit,
    /// A trap (including a host-detected fatal condition) ended the group.
    trap,
};

pub const Outcome = struct {
    kind: Kind,
    /// Exit status for `.exit`; an opaque, embedder-defined trap code for
    /// `.trap` (the CLI reports every trap as status 1).
    code: u32,
};

pub const WakeHook = struct {
    ctx: *anyopaque,
    wake: *const fn (*anyopaque) void,

    fn invoke(self: WakeHook) void {
        self.wake(self.ctx);
    }
};

/// Publication states for the claim word.
const unclaimed: u32 = 0;
const claiming: u32 = 1;
const published: u32 = 2;

fn StateFor(comptime enabled: bool) type {
    return if (enabled) struct {
        const Self = @This();

        claim_state: std.atomic.Value(u32) = std.atomic.Value(u32).init(unclaimed),
        /// Written by the winning claimer before `claim_state` is published.
        kind: Kind = .exit,
        code: u32 = 0,
        hook_lock: HookLock = .{},
        hook: ?WakeHook = null,

        /// Record `proc_exit(code)`. Returns true when this call won.
        pub fn claimExit(self: *Self, code: u32) bool {
            return self.claim(.exit, code);
        }

        /// Record a trap. Returns true when this call won.
        pub fn claimTrap(self: *Self, code: u32) bool {
            return self.claim(.trap, code);
        }

        fn claim(self: *Self, kind: Kind, code: u32) bool {
            if (self.claim_state.cmpxchgStrong(
                unclaimed,
                claiming,
                .acq_rel,
                .acquire,
            ) != null) return false;
            self.kind = kind;
            self.code = code;
            self.claim_state.store(published, .release);
            self.wakeGroup();
            return true;
        }

        /// True once any thread has claimed the terminal outcome. Cheap
        /// enough for blocking-I/O slice checks and wait-entry guards.
        pub fn isTerminating(self: *const Self) bool {
            return self.claim_state.load(.acquire) != unclaimed;
        }

        /// The winning outcome, or null while the group is still running.
        ///
        /// A claim that is mid-publication is resolved by a bounded spin:
        /// the winner publishes with no intervening blocking work.
        pub fn outcome(self: *const Self) ?Outcome {
            while (true) {
                switch (self.claim_state.load(.acquire)) {
                    unclaimed => return null,
                    claiming => std.atomic.spinLoopHint(),
                    else => return .{ .kind = self.kind, .code = self.code },
                }
            }
        }

        /// Exit status when `proc_exit` won, null when the group is running
        /// or a trap won the race.
        pub fn exitCode(self: *const Self) ?u32 {
            const result = self.outcome() orelse return null;
            return switch (result.kind) {
                .exit => result.code,
                .trap => null,
            };
        }

        /// Install the group wakeup hook. If the group already terminated,
        /// the hook fires immediately so a late binder still tears down.
        pub fn bindWake(self: *Self, hook: WakeHook) void {
            self.hook_lock.lock();
            self.hook = hook;
            const terminating = self.isTerminating();
            self.hook_lock.unlock();
            if (terminating) hook.invoke();
        }

        /// Drop the hook. Callers must unbind before the hook owner dies.
        pub fn unbindWake(self: *Self, ctx: *anyopaque) void {
            self.hook_lock.lock();
            defer self.hook_lock.unlock();
            if (self.hook) |hook| {
                if (hook.ctx == ctx) self.hook = null;
            }
        }

        fn wakeGroup(self: *Self) void {
            self.hook_lock.lock();
            const hook = self.hook;
            self.hook_lock.unlock();
            if (hook) |installed| installed.invoke();
        }
    } else struct {
        const Self = @This();

        claimed: bool = false,
        kind: Kind = .exit,
        code: u32 = 0,

        pub fn claimExit(self: *Self, code: u32) bool {
            return self.claim(.exit, code);
        }

        pub fn claimTrap(self: *Self, code: u32) bool {
            return self.claim(.trap, code);
        }

        fn claim(self: *Self, kind: Kind, code: u32) bool {
            if (self.claimed) return false;
            self.claimed = true;
            self.kind = kind;
            self.code = code;
            return true;
        }

        pub fn isTerminating(self: *const Self) bool {
            return self.claimed;
        }

        pub fn outcome(self: *const Self) ?Outcome {
            if (!self.claimed) return null;
            return .{ .kind = self.kind, .code = self.code };
        }

        pub fn exitCode(self: *const Self) ?u32 {
            const result = self.outcome() orelse return null;
            return switch (result.kind) {
                .exit => result.code,
                .trap => null,
            };
        }

        pub fn bindWake(self: *Self, hook: WakeHook) void {
            _ = self;
            _ = hook;
        }

        pub fn unbindWake(self: *Self, ctx: *anyopaque) void {
            _ = self;
            _ = ctx;
        }
    };
}

/// Spin lock guarding hook installation. Hook binding happens during setup
/// and teardown only; claims take it for the duration of one store read.
const HookLock = struct {
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

    fn lock(self: *HookLock) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null)
            std.atomic.spinLoopHint();
    }

    fn unlock(self: *HookLock) void {
        self.state.store(0, .release);
    }
};

pub const State = StateFor(config.lib_wasi_threads);

/// Opaque trap code used when a trap has no embedder-visible status.
pub const generic_trap_code: u32 = 1;

// ── Tests ────────────────────────────────────────────────────────────────

test "termination: the first claim wins and later claims are no-ops" {
    var state = State{};
    try std.testing.expect(!state.isTerminating());
    try std.testing.expect(state.outcome() == null);
    try std.testing.expect(state.exitCode() == null);

    try std.testing.expect(state.claimExit(7));
    try std.testing.expect(state.isTerminating());
    try std.testing.expect(!state.claimExit(9));
    try std.testing.expect(!state.claimTrap(generic_trap_code));

    const result = state.outcome().?;
    try std.testing.expectEqual(Kind.exit, result.kind);
    try std.testing.expectEqual(@as(u32, 7), result.code);
    try std.testing.expectEqual(@as(?u32, 7), state.exitCode());
}

test "termination: a winning trap is not masked by a later proc_exit" {
    var state = State{};
    try std.testing.expect(state.claimTrap(generic_trap_code));
    try std.testing.expect(!state.claimExit(0));

    try std.testing.expectEqual(Kind.trap, state.outcome().?.kind);
    // The embedder must not see a success exit code behind the trap.
    try std.testing.expect(state.exitCode() == null);
}

test "termination: the wake hook fires once, on claim or on late binding" {
    const Counter = struct {
        hits: usize = 0,

        fn wake(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.hits += 1;
        }
    };

    var counter = Counter{};
    var state = State{};
    state.bindWake(.{ .ctx = @ptrCast(&counter), .wake = Counter.wake });
    try std.testing.expectEqual(@as(usize, 0), counter.hits);

    _ = state.claimExit(3);
    const after_claim: usize = if (config.lib_wasi_threads) 1 else 0;
    try std.testing.expectEqual(after_claim, counter.hits);

    _ = state.claimExit(4);
    try std.testing.expectEqual(after_claim, counter.hits);

    var late = Counter{};
    var terminated = State{};
    _ = terminated.claimTrap(generic_trap_code);
    terminated.bindWake(.{ .ctx = @ptrCast(&late), .wake = Counter.wake });
    try std.testing.expectEqual(after_claim, late.hits);

    terminated.unbindWake(@ptrCast(&late));
}

test "termination: concurrent claims publish exactly one outcome" {
    if (!config.lib_wasi_threads or @import("builtin").single_threaded)
        return error.SkipZigTest;

    const Racer = struct {
        state: *State,
        code: u32,
        wins: *std.atomic.Value(usize),
        gate: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            while (!self.gate.load(.acquire)) std.atomic.spinLoopHint();
            const won = if (self.code == 0)
                self.state.claimTrap(generic_trap_code)
            else
                self.state.claimExit(self.code);
            if (won) _ = self.wins.fetchAdd(1, .monotonic);
        }
    };

    var round: usize = 0;
    while (round < 64) : (round += 1) {
        var state = State{};
        var wins = std.atomic.Value(usize).init(0);
        var gate = std.atomic.Value(bool).init(false);
        var racers: [4]Racer = undefined;
        var threads: [4]std.Thread = undefined;
        for (&racers, 0..) |*racer, i| {
            racer.* = .{
                .state = &state,
                .code = @intCast(i),
                .wins = &wins,
                .gate = &gate,
            };
            threads[i] = try std.Thread.spawn(.{}, Racer.run, .{racer});
        }
        gate.store(true, .release);
        for (threads) |thread| thread.join();

        try std.testing.expectEqual(@as(usize, 1), wins.load(.acquire));
        const result = state.outcome().?;
        // Whatever won must be a code some racer actually asked for.
        switch (result.kind) {
            .exit => try std.testing.expect(result.code >= 1 and result.code <= 3),
            .trap => try std.testing.expectEqual(generic_trap_code, result.code),
        }
        // The published outcome never changes after the fact.
        try std.testing.expectEqual(result.kind, state.outcome().?.kind);
        try std.testing.expectEqual(result.code, state.outcome().?.code);
    }
}
