//! Hard-timeout wrapper for the Preview-1 thread fixtures.
//!
//! Group-termination fixtures assert that a runtime *stops*; a regression
//! makes them hang rather than fail. Running them through this wrapper turns
//! a hang into a deterministic failure instead of a stuck CI job.
//!
//! Usage: run-bounded <timeout-seconds> [--max-elapsed-ms=N] <exe> [args...]
//!
//! Exits with the child's status, or 124 (the `timeout(1)` convention) when
//! the deadline expired and the child had to be killed. The optional elapsed
//! bound rejects a child that exits normally but only after a runtime fallback
//! deadline; it returns 126 so this failure is distinct from the hard timeout.

const std = @import("std");
const builtin = @import("builtin");

const is_posix = builtin.os.tag != .windows;
const max_elapsed_prefix = "--max-elapsed-ms=";

const Watchdog = struct {
    child: *std.process.Child,
    io: std.Io,
    timeout_ns: u64,
    finished: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    expired: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    fn run(self: *Watchdog) void {
        const step_ns: u64 = 20 * std.time.ns_per_ms;
        var waited: u64 = 0;
        while (waited < self.timeout_ns) : (waited += step_ns) {
            if (self.finished.load(.acquire)) return;
            sleepNs(step_ns);
        }
        if (self.finished.load(.acquire)) return;
        self.expired.store(true, .release);
        if (comptime is_posix) {
            // Signal the process group so a nested `wamrc run` takes its
            // spawned `wamr` with it.
            if (self.child.id) |pid| {
                std.posix.kill(-pid, std.posix.SIG.KILL) catch {
                    std.posix.kill(pid, std.posix.SIG.KILL) catch {};
                };
            }
        } else {
            self.child.kill(self.io);
        }
    }
};

fn sleepNs(ns: u64) void {
    if (comptime builtin.os.tag == .linux) {
        const ts: std.os.linux.timespec = .{
            .sec = @intCast(ns / std.time.ns_per_s),
            .nsec = @intCast(ns % std.time.ns_per_s),
        };
        _ = std.os.linux.nanosleep(&ts, null);
    } else if (comptime builtin.os.tag == .windows) {
        // NtDelayExecution takes 100 ns units, negative for a relative delay
        // — the same primitive `src/platform/platform.zig` uses.
        const hundred_ns: u64 = @min(ns / 100, @as(u64, @intCast(std.math.maxInt(i64))));
        const delay: std.os.windows.LARGE_INTEGER = -@as(i64, @intCast(hundred_ns));
        _ = std.os.windows.ntdll.NtDelayExecution(.FALSE, &delay);
    } else {
        // macOS and other POSIX hosts: busy-yield in coarse steps. The
        // watchdog is idle bookkeeping, so precision does not matter.
        var spins: usize = 0;
        while (spins < ns / (100 * std.time.ns_per_us)) : (spins += 1)
            std.Thread.yield() catch {};
    }
}

fn monotonicNs() u64 {
    return switch (comptime builtin.os.tag) {
        .linux => blk: {
            const linux = std.os.linux;
            var ts: linux.timespec = undefined;
            if (linux.clock_gettime(.MONOTONIC, &ts) != 0) break :blk 0;
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s +
                @as(u64, @intCast(ts.nsec));
        },
        .macos, .ios, .tvos, .watchos, .visionos => blk: {
            var ts: std.c.timespec = undefined;
            if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) break :blk 0;
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s +
                @as(u64, @intCast(ts.nsec));
        },
        .windows => blk: {
            const ntdll = std.os.windows.ntdll;
            var counter: std.os.windows.LARGE_INTEGER = undefined;
            var freq: std.os.windows.LARGE_INTEGER = undefined;
            _ = ntdll.RtlQueryPerformanceCounter(&counter);
            _ = ntdll.RtlQueryPerformanceFrequency(&freq);
            const ticks: u128 = @intCast(counter);
            const hz: u128 = @intCast(freq);
            if (hz == 0) break :blk 0;
            break :blk @as(u64, @truncate(ticks * std.time.ns_per_s / hz));
        },
        else => 0,
    };
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print(
            "usage: run-bounded <timeout-seconds> [--max-elapsed-ms=N] <exe> [args...]\n",
            .{},
        );
        return 2;
    }

    const timeout_s = std.fmt.parseInt(u64, args[1], 10) catch {
        std.debug.print("run-bounded: invalid timeout '{s}'\n", .{args[1]});
        return 2;
    };

    var child_arg_index: usize = 2;
    var max_elapsed_ns: ?u64 = null;
    if (std.mem.startsWith(u8, args[child_arg_index], max_elapsed_prefix)) {
        const raw_ms = args[child_arg_index][max_elapsed_prefix.len..];
        const max_elapsed_ms = std.fmt.parseInt(u64, raw_ms, 10) catch {
            std.debug.print("run-bounded: invalid elapsed bound '{s}'\n", .{raw_ms});
            return 2;
        };
        if (max_elapsed_ms == 0) {
            std.debug.print("run-bounded: elapsed bound must be positive\n", .{});
            return 2;
        }
        max_elapsed_ns = std.math.mul(
            u64,
            max_elapsed_ms,
            std.time.ns_per_ms,
        ) catch {
            std.debug.print("run-bounded: elapsed bound is too large\n", .{});
            return 2;
        };
        child_arg_index += 1;
        if (child_arg_index >= args.len) {
            std.debug.print("run-bounded: missing executable\n", .{});
            return 2;
        }
    }

    const started_ns = if (max_elapsed_ns != null) monotonicNs() else 0;
    if (max_elapsed_ns != null and started_ns == 0) {
        std.debug.print("run-bounded: monotonic clock unavailable\n", .{});
        return 2;
    }

    var child = try std.process.spawn(io, .{
        .argv = args[child_arg_index..],
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });

    var watchdog = Watchdog{
        .child = &child,
        .io = io,
        .timeout_ns = timeout_s * std.time.ns_per_s,
    };
    const watcher = try std.Thread.spawn(.{}, Watchdog.run, .{&watchdog});

    const term = child.wait(io) catch |err| {
        watchdog.finished.store(true, .release);
        watcher.join();
        std.debug.print("run-bounded: wait failed: {s}\n", .{@errorName(err)});
        return 2;
    };
    const finished_ns = if (max_elapsed_ns != null) monotonicNs() else 0;
    watchdog.finished.store(true, .release);
    watcher.join();

    if (watchdog.expired.load(.acquire)) {
        std.debug.print(
            "run-bounded: '{s}' did not exit within {d}s — killed\n",
            .{ args[child_arg_index], timeout_s },
        );
        return 124;
    }

    if (max_elapsed_ns) |limit_ns| {
        if (finished_ns == 0 or finished_ns < started_ns) {
            std.debug.print("run-bounded: monotonic clock failed during run\n", .{});
            return 2;
        }
        const elapsed_ns = finished_ns - started_ns;
        if (elapsed_ns > limit_ns) {
            std.debug.print(
                "run-bounded: '{s}' took {d}ms, exceeding the {d}ms bound\n",
                .{
                    args[child_arg_index],
                    elapsed_ns / std.time.ns_per_ms,
                    limit_ns / std.time.ns_per_ms,
                },
            );
            return 126;
        }
    }

    return switch (term) {
        .exited => |code| code,
        .signal => |sig| blk: {
            std.debug.print("run-bounded: child died from signal {t}\n", .{sig});
            break :blk 125;
        },
        else => 125,
    };
}
