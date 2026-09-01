//! Hard-timeout wrapper for the Preview-1 thread fixtures.
//!
//! Group-termination fixtures assert that a runtime *stops*; a regression
//! makes them hang rather than fail. Running them through this wrapper turns
//! a hang into a deterministic failure instead of a stuck CI job.
//!
//! Usage: run-bounded <timeout-seconds> <exe> [args...]
//!
//! Exits with the child's status, or 124 (the `timeout(1)` convention) when
//! the deadline expired and the child had to be killed.

const std = @import("std");
const builtin = @import("builtin");

const is_posix = builtin.os.tag != .windows;

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

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: run-bounded <timeout-seconds> <exe> [args...]\n", .{});
        return 2;
    }

    const timeout_s = std.fmt.parseInt(u64, args[1], 10) catch {
        std.debug.print("run-bounded: invalid timeout '{s}'\n", .{args[1]});
        return 2;
    };

    var child = try std.process.spawn(io, .{
        .argv = args[2..],
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
    watchdog.finished.store(true, .release);
    watcher.join();

    if (watchdog.expired.load(.acquire)) {
        std.debug.print(
            "run-bounded: '{s}' did not exit within {d}s — killed\n",
            .{ args[2], timeout_s },
        );
        return 124;
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
