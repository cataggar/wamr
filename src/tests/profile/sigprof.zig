//! POSIX SIGPROF sampling backend (linux + aarch64/x86_64).
//!
//! Fires `SIGPROF` every `interval_us` microseconds via `setitimer
//! ITIMER_PROF` and records the interrupted PC into a fixed-size ring
//! buffer. Designed to be installed around a single AOT call site (`arm`)
//! and torn down immediately after (`disarm`).
//!
//! The handler is async-signal-safe: it does only atomic ring-buffer
//! writes — no allocation, no formatting, no blocking syscalls. Linux only
//! to keep the ucontext_t / mcontext_t layout pinned to a single set of
//! kernel ABIs (the runner host for `coremark-profile` is the AArch64
//! self-hosted Linux runner; x86_64-linux is supported as a cross-check).

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

pub const supported = builtin.os.tag == .linux and
    (builtin.cpu.arch == .aarch64 or builtin.cpu.arch == .x86_64);

pub const Sample = u64;

/// Fixed capacity. 64 K samples × 8 bytes = 512 KiB; ample for ~60 s of
/// 1 ms sampling and easily fits in BSS.
pub const Capacity: usize = 64 * 1024;

const State = struct {
    pcs: [Capacity]Sample = undefined,
    /// Monotonically increasing sample count. Atomic store from handler.
    count: std.atomic.Value(u64) = .init(0),
    /// Number of samples that overflowed the ring (count > Capacity).
    dropped: std.atomic.Value(u64) = .init(0),
};

var g_state: State = .{};

var g_prev_action: linux.Sigaction = undefined;
var g_prev_itimer: itimerval = .{
    .it_interval = .{ .sec = 0, .usec = 0 },
    .it_value = .{ .sec = 0, .usec = 0 },
};

const itimerval = extern struct {
    it_interval: timeval,
    it_value: timeval,
    pub const timeval = extern struct { sec: i64, usec: i64 };
};

const ITIMER_PROF: usize = 2;

/// Direct `setitimer` syscall — avoids a libc link dependency for callers
/// that prefer to keep the runtime self-contained. The kernel ABI takes
/// `struct itimerval` (sec, usec) and the `itimerval` extern struct above
/// is byte-compatible.
fn rawSetitimer(which: usize, new_value: *const itimerval, old_value: ?*itimerval) usize {
    return std.os.linux.syscall3(
        .setitimer,
        which,
        @intFromPtr(new_value),
        @intFromPtr(old_value),
    );
}

// ─── Linux mcontext layouts (PC field only) ────────────────────────────
// Pulled from std/debug/cpu_context.zig — reproduced here so we don't
// depend on a private std API.

const UcontextAarch64 = extern struct {
    _flags: usize,
    _link: ?*anyopaque,
    _stack: linux.stack_t,
    _sigmask: linux.sigset_t,
    _unused: [120]u8,
    mcontext: extern struct {
        _fault_address: u64 align(16),
        x: [30]u64,
        lr: u64,
        sp: u64,
        pc: u64,
    },
};

const UcontextX86_64 = extern struct {
    _flags: usize,
    _link: ?*anyopaque,
    _stack: linux.stack_t,
    mcontext: extern struct {
        r8: u64,
        r9: u64,
        r10: u64,
        r11: u64,
        r12: u64,
        r13: u64,
        r14: u64,
        r15: u64,
        rdi: u64,
        rsi: u64,
        rbp: u64,
        rbx: u64,
        rdx: u64,
        rax: u64,
        rcx: u64,
        rsp: u64,
        rip: u64,
    },
};

inline fn pcFromUcontext(ctx: ?*anyopaque) u64 {
    const p = ctx orelse return 0;
    return switch (builtin.cpu.arch) {
        .aarch64 => blk: {
            const u: *const UcontextAarch64 = @ptrCast(@alignCast(p));
            break :blk u.mcontext.pc;
        },
        .x86_64 => blk: {
            const u: *const UcontextX86_64 = @ptrCast(@alignCast(p));
            break :blk u.mcontext.rip;
        },
        else => 0,
    };
}

fn handler(_: linux.SIG, _: *const linux.siginfo_t, ctx: ?*anyopaque) callconv(.c) void {
    const pc = pcFromUcontext(ctx);
    const idx = g_state.count.fetchAdd(1, .acq_rel);
    if (idx < Capacity) {
        g_state.pcs[idx] = pc;
    } else {
        _ = g_state.dropped.fetchAdd(1, .acq_rel);
    }
}

pub const ArmError = error{
    Unsupported,
    SetitimerFailed,
};

/// Reset counters, install the SIGPROF handler (preserving the previous
/// disposition), and arm `ITIMER_PROF` at `interval_us` microseconds.
pub fn arm(interval_us: u32) ArmError!void {
    if (!supported) return error.Unsupported;
    g_state.count.store(0, .release);
    g_state.dropped.store(0, .release);

    const act: linux.Sigaction = .{
        .handler = .{ .sigaction = handler },
        .mask = linux.sigemptyset(),
        .flags = linux.SA.SIGINFO | linux.SA.RESTART,
    };
    std.posix.sigaction(.PROF, &act, &g_prev_action);

    const sec: i64 = @intCast(interval_us / 1_000_000);
    const usec: i64 = @intCast(interval_us % 1_000_000);
    const new_it: itimerval = .{
        .it_interval = .{ .sec = sec, .usec = usec },
        .it_value = .{ .sec = sec, .usec = usec },
    };
    if (rawSetitimer(ITIMER_PROF, &new_it, &g_prev_itimer) != 0) {
        std.posix.sigaction(.PROF, &g_prev_action, null);
        return error.SetitimerFailed;
    }
}

/// Disarm the timer and restore the previous SIGPROF disposition.
pub fn disarm() void {
    if (!supported) return;
    const zero: itimerval = .{
        .it_interval = .{ .sec = 0, .usec = 0 },
        .it_value = .{ .sec = 0, .usec = 0 },
    };
    _ = rawSetitimer(ITIMER_PROF, &zero, null);
    std.posix.sigaction(.PROF, &g_prev_action, null);
}

/// Captured PC samples. Length is clamped to `Capacity` even when the ring
/// overflowed; `droppedCount` reports the overflow.
pub fn samples() []const Sample {
    const cnt = @min(g_state.count.load(.acquire), Capacity);
    return g_state.pcs[0..cnt];
}

pub fn droppedCount() u64 {
    return g_state.dropped.load(.acquire);
}
