//! End-to-end SIGINT graceful-shutdown driver for `wamr serve` (#918).
//!
//! Spawns `<wamr_exe> serve --addr=127.0.0.1:0 <component.wasm>` as a
//! subprocess, discovers the kernel-assigned ephemeral port by scraping
//! the `Listening on 127.0.0.1:<port>` line from the child's stdout,
//! confirms at least one successful HTTP request (`GET /` -> 200), then
//! sends `SIGINT` and requires a *bounded, clean* `exit 0`.
//!
//! This is the regression net for #918: before the fix `wamr serve`
//! installed a graceful-shutdown handler only on Linux (so native macOS
//! died with signal 2 / `Popen(-2)`), and even on Linux a SIGINT that
//! raced the JIT compile was lost — the process wedged in `accept` until
//! force-killed. After the fix the SIGINT/SIGTERM handler is portable and
//! async-signal-safe (it only sets an atomic flag + wakes the accept loop
//! via a self-pipe), and all teardown runs on the normal control path.
//!
//! Mirrors `tests/component-http-smoke/driver.zig` (the serve-loop smoke
//! driver). Uses the `http-service.wasm` P3 fixture, which serves
//! `GET / -> 200 "hey\n"`.
//!
//! Usage: `component-http-shutdown-driver <wamr_exe> <component.wasm>`
//!
//! Exit 0 on success; non-zero (with a message on stderr) otherwise.

const std = @import("std");
const linux = std.os.linux;
const builtin = @import("builtin");

const usage = "usage: component-http-shutdown-driver <wamr_exe> <component.wasm>\n";

/// Force-kills the server if it fails to exit within the deadline, so a
/// regression can never hang the test suite indefinitely.
const Watchdog = struct {
    pid: linux.pid_t,
    done: std.atomic.Value(bool) = .init(false),

    fn run(self: *Watchdog) void {
        const timeout_ms: usize = 10_000;
        var waited: usize = 0;
        while (waited < timeout_ms) : (waited += 50) {
            if (self.done.load(.acquire)) return;
            const ts: linux.timespec = .{ .sec = 0, .nsec = 50 * std.time.ns_per_ms };
            _ = linux.nanosleep(&ts, null);
        }
        if (self.done.load(.acquire)) return;
        std.debug.print("driver: TIMEOUT waiting for clean shutdown; force-killing\n", .{});
        _ = linux.kill(self.pid, .KILL);
    }

    fn cancel(self: *Watchdog) void {
        self.done.store(true, .release);
    }
};

pub fn main(init: std.process.Init) !void {
    if (comptime builtin.os.tag != .linux) {
        // This driver uses raw Linux syscalls; it is only wired into the
        // build on Linux (see build.zig). Skip cleanly elsewhere.
        std.debug.print("component-http-shutdown-driver only runs on Linux; skipping\n", .{});
        std.process.exit(0);
    }

    const io = init.io;
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    const wamr_exe = args[1];
    const component_path = args[2];

    var child = try std.process.spawn(io, .{
        .argv = &.{ wamr_exe, "serve", "--addr=127.0.0.1:0", component_path },
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .inherit,
    });
    var killed = false;
    defer if (!killed) child.kill(io);

    // Discover the ephemeral port from the child's stdout `Listening on
    // 127.0.0.1:<port>` line. `read` blocks until the line is produced;
    // JIT-compiling the ~3 MB component can take a couple of seconds.
    const out_fd = child.stdout.?.handle;
    var acc: std.ArrayListUnmanaged(u8) = .empty;
    defer acc.deinit(allocator);
    const port: u16 = blk: {
        var buf: [512]u8 = undefined;
        while (true) {
            const rc = linux.read(out_fd, &buf, buf.len);
            if (linux.errno(rc) != .SUCCESS) {
                std.debug.print("driver: read(server stdout) failed\n", .{});
                std.process.exit(1);
            }
            const n: isize = @bitCast(rc);
            if (n <= 0) {
                std.debug.print("driver: server closed stdout before announcing a port\n", .{});
                std.process.exit(1);
            }
            try acc.appendSlice(allocator, buf[0..@intCast(n)]);
            if (parseListeningPort(acc.items)) |p| break :blk p;
            if (acc.items.len > 64 * 1024) {
                std.debug.print("driver: no `Listening on` line in server stdout\n", .{});
                std.process.exit(1);
            }
        }
    };

    // ── One successful request: GET / -> 200 "hey\n" ──────────────────
    {
        var resp_buf: [4096]u8 = undefined;
        const resp = sendReceiveOnce(port, "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n", resp_buf[0..]) catch |err| {
            std.debug.print("driver: request failed: {t}\n", .{err});
            std.process.exit(1);
        };
        assertStatus(resp, 200) catch std.process.exit(1);
        assertBodyEquals(resp, "hey\n") catch std.process.exit(1);
    }

    // ── Send SIGINT; require a bounded, clean exit 0 ──────────────────
    const pid = child.id.?;
    var wd: Watchdog = .{ .pid = pid };
    const wd_thread = try std.Thread.spawn(.{}, Watchdog.run, .{&wd});

    _ = linux.kill(pid, .INT);

    const term = child.wait(io) catch |err| {
        wd.cancel();
        wd_thread.join();
        std.debug.print("driver: wait failed: {t}\n", .{err});
        std.process.exit(1);
    };
    killed = true; // `wait` reaped the child; do not `kill` it in `defer`.
    wd.cancel();
    wd_thread.join();

    switch (term) {
        .exited => |code| {
            if (code != 0) {
                std.debug.print("driver: serve exited {d} on SIGINT, expected 0\n", .{code});
                std.process.exit(1);
            }
        },
        else => {
            std.debug.print("driver: serve did not exit cleanly on SIGINT: {any}\n", .{term});
            std.process.exit(1);
        },
    }
}

/// Scrape the `Listening on 127.0.0.1:<port>` line the CLI prints once the
/// listener is bound (only emitted for the ephemeral `:0` bind path).
fn parseListeningPort(bytes: []const u8) ?u16 {
    const marker = "Listening on 127.0.0.1:";
    const idx = std.mem.indexOf(u8, bytes, marker) orelse return null;
    const rest = bytes[idx + marker.len ..];
    var end: usize = 0;
    while (end < rest.len and rest[end] >= '0' and rest[end] <= '9') : (end += 1) {}
    if (end == 0) return null;
    // Require the newline so we don't parse a truncated port mid-read.
    if (end >= rest.len or (rest[end] != '\n' and rest[end] != '\r')) return null;
    return std.fmt.parseInt(u16, rest[0..end], 10) catch null;
}

fn tcpConnect(port: u16) !i32 {
    const dest: linux.sockaddr.in = .{
        .port = std.mem.nativeToBig(u16, port),
        .addr = @bitCast([4]u8{ 127, 0, 0, 1 }),
    };
    const fd_rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, linux.IPPROTO.TCP);
    if (linux.errno(fd_rc) != .SUCCESS) return error.SocketFailed;
    const fd: i32 = @intCast(@as(isize, @bitCast(fd_rc)));
    const rc = linux.connect(fd, @ptrCast(&dest), @sizeOf(@TypeOf(dest)));
    if (linux.errno(rc) != .SUCCESS) {
        _ = linux.close(fd);
        return error.ConnectFailed;
    }
    return fd;
}

fn sendReceiveOnce(
    port: u16,
    request: []const u8,
    resp_buf: []u8,
) ![]const u8 {
    const fd = try tcpConnect(port);
    defer _ = linux.close(fd);

    var sent: usize = 0;
    while (sent < request.len) {
        const rc = linux.write(fd, request[sent..].ptr, request.len - sent);
        if (linux.errno(rc) != .SUCCESS) return error.SendFailed;
        const n: isize = @bitCast(rc);
        if (n <= 0) return error.SendFailed;
        sent += @intCast(n);
    }

    // Read until the connection closes — wamr serves one response then
    // closes (per `serveOneHttpConnection`).
    var total: usize = 0;
    while (true) {
        if (total >= resp_buf.len) break;
        const rc = linux.read(fd, resp_buf[total..].ptr, resp_buf.len - total);
        if (linux.errno(rc) != .SUCCESS) return error.RecvFailed;
        const n: isize = @bitCast(rc);
        if (n == 0) break;
        if (n < 0) return error.RecvFailed;
        total += @intCast(n);
    }
    return resp_buf[0..total];
}

fn assertStatus(resp: []const u8, expected: u16) !void {
    const prefix = "HTTP/1.1 ";
    if (!std.mem.startsWith(u8, resp, prefix)) {
        std.debug.print("driver: bad status line — got: {s}\n", .{resp[0..@min(resp.len, 80)]});
        return error.BadStatusLine;
    }
    const after = resp[prefix.len..];
    var i: usize = 0;
    while (i < after.len and after[i] >= '0' and after[i] <= '9') : (i += 1) {}
    if (i == 0) return error.BadStatusLine;
    const code = std.fmt.parseInt(u16, after[0..i], 10) catch return error.BadStatusLine;
    if (code != expected) {
        std.debug.print("driver: expected {d}, got {d} — full status line: {s}\n", .{
            expected,
            code,
            after[0..@min(after.len, 60)],
        });
        return error.WrongStatus;
    }
}

fn assertBodyEquals(resp: []const u8, expected: []const u8) !void {
    const sep = "\r\n\r\n";
    const sep_idx = std.mem.indexOf(u8, resp, sep) orelse return error.MissingHeaderTerminator;
    const body = resp[sep_idx + sep.len ..];
    if (!std.mem.eql(u8, body, expected)) {
        std.debug.print("driver: body mismatch.\n  expected: {s}\n  got:      {s}\n", .{ expected, body });
        return error.BodyMismatch;
    }
}
