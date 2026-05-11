//! Integration-test driver for the WASI sockets plumbing (#437).
//!
//! Spawns `wamr run --listen=127.0.0.1:<port> <echo.wasm>` as a subprocess
//! and verifies that the guest's `sock_accept` / `sock_recv` / `sock_send`
//! roundtrip a single payload back to a host client.
//!
//! Usage: `wasi-sock-driver <wamr_exe> <wasm_path> <port>`
//!
//! Exit code 0 on success; non-zero (and a message on stderr) on failure.
//! The driver is the run target for the `test-wasi-sock` build step.

const std = @import("std");
const linux = std.os.linux;
const builtin = @import("builtin");

const message = "hello wamr sockets";
const usage = "usage: wasi-sock-driver <wamr_exe> <wasm_path> <port>\n";

pub fn main(init: std.process.Init) !void {
    if (comptime builtin.os.tag != .linux) {
        std.debug.print("wasi-sock-driver only runs on Linux\n", .{});
        std.process.exit(1);
    }

    const io = init.io;
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    const wamr_exe = args[1];
    const wasm_path = args[2];
    const port = std.fmt.parseInt(u16, args[3], 10) catch |err| {
        std.debug.print("invalid port '{s}': {t}\n", .{ args[3], err });
        std.process.exit(2);
    };

    const listen_arg = try std.fmt.allocPrint(allocator, "--listen=127.0.0.1:{d}", .{port});
    defer allocator.free(listen_arg);

    var child = try std.process.spawn(io, .{
        .argv = &.{ wamr_exe, "run", listen_arg, wasm_path },
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    var child_owned = true;
    defer if (child_owned) {
        child.kill(io);
    };

    // Open a host-side client socket and retry connect while the guest
    // initialises. The bind happens before fork+exec of wamr's interpreter
    // loop (eager-bind contract documented in main.zig), so the kernel
    // accepts the connect as soon as wamr is past arg parsing.
    var connected_fd: i32 = -1;
    {
        const dest: linux.sockaddr.in = .{
            .port = std.mem.nativeToBig(u16, port),
            .addr = @bitCast([4]u8{ 127, 0, 0, 1 }),
        };
        const max_attempts: u32 = 60; // ~3s @ 50ms
        var attempt: u32 = 0;
        while (attempt < max_attempts) : (attempt += 1) {
            const fd_rc = linux.socket(
                linux.AF.INET,
                linux.SOCK.STREAM | linux.SOCK.CLOEXEC,
                linux.IPPROTO.TCP,
            );
            if (linux.errno(fd_rc) != .SUCCESS) return error.SocketFailed;
            const fd: i32 = @intCast(@as(isize, @bitCast(fd_rc)));

            const rc = linux.connect(fd, @ptrCast(&dest), @sizeOf(@TypeOf(dest)));
            if (linux.errno(rc) == .SUCCESS) {
                connected_fd = fd;
                break;
            }
            _ = linux.close(fd);
            const ts: linux.timespec = .{ .sec = 0, .nsec = 50 * std.time.ns_per_ms };
            _ = linux.nanosleep(&ts, null);
        }
        if (connected_fd < 0) {
            std.debug.print(
                "driver: failed to connect to 127.0.0.1:{d} after {d} attempts\n",
                .{ port, max_attempts },
            );
            return error.ConnectTimeout;
        }
    }
    defer _ = linux.close(connected_fd);

    {
        const w_rc = linux.write(connected_fd, message.ptr, message.len);
        if (linux.errno(w_rc) != .SUCCESS) return error.WriteFailed;
        const wrote: usize = @intCast(@as(isize, @bitCast(w_rc)));
        if (wrote != message.len) return error.ShortWrite;
    }

    var recv_buf: [128]u8 = undefined;
    var total: usize = 0;
    while (total < message.len) {
        const r_rc = linux.read(
            connected_fd,
            recv_buf[total..].ptr,
            recv_buf.len - total,
        );
        if (linux.errno(r_rc) != .SUCCESS) return error.ReadFailed;
        const n: usize = @intCast(@as(isize, @bitCast(r_rc)));
        if (n == 0) break; // EOF
        total += n;
    }
    if (!std.mem.eql(u8, recv_buf[0..total], message)) {
        std.debug.print(
            "driver: echo mismatch — expected '{s}', got '{s}'\n",
            .{ message, recv_buf[0..total] },
        );
        return error.MismatchedEcho;
    }

    const term = try child.wait(io);
    child_owned = false;
    switch (term) {
        .exited => |code| if (code != 0) {
            std.debug.print("driver: wamr exited with code {d}\n", .{code});
            return error.WamrExitedNonZero;
        },
        else => {
            std.debug.print("driver: wamr terminated abnormally: {any}\n", .{term});
            return error.WamrAbnormalTerm;
        },
    }
}
