//! End-to-end smoke driver for `examples/components/zig-http`.
//!
//! Spawns `<wamr_exe> run --listen=127.0.0.1:<port> <component.wasm>` as
//! a subprocess and verifies the canonical HTTP-tutorial behaviour:
//!
//!   GET /         -> 200 "Hello, world!\n"
//!   GET /missing  -> 404 ""
//!
//! Mirrors `tests/wasi-sock/driver.zig` (the WASI-sockets driver wired
//! by `#437`). The driver is the run target for the
//! `component-examples-run` build step's `zig-http` slice.
//!
//! Usage: `component-http-smoke-driver <wamr_exe> <component.wasm> <port>`
//!
//! Exit 0 on success; non-zero (with a message on stderr) on any
//! mismatch or RPC failure.

const std = @import("std");
const linux = std.os.linux;
const builtin = @import("builtin");

const usage = "usage: component-http-smoke-driver <wamr_exe> <component.wasm> <port>\n";

pub fn main(init: std.process.Init) !void {
    if (comptime builtin.os.tag != .linux) {
        std.debug.print("component-http-smoke-driver only runs on Linux\n", .{});
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
    const component_path = args[2];
    const port = std.fmt.parseInt(u16, args[3], 10) catch |err| {
        std.debug.print("invalid port '{s}': {t}\n", .{ args[3], err });
        std.process.exit(2);
    };

    const listen_arg = try std.fmt.allocPrint(allocator, "--listen=127.0.0.1:{d}", .{port});
    defer allocator.free(listen_arg);

    var child = try std.process.spawn(io, .{
        .argv = &.{ wamr_exe, "run", listen_arg, component_path },
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    defer child.kill(io);

    // Wait for the server to bind. The wamr CLI eagerly binds (per
    // main.zig contract) before the component is instantiated, so a
    // few retries is plenty.
    var ready = false;
    {
        const max_attempts: u32 = 60; // ~3s @ 50ms
        var attempt: u32 = 0;
        while (attempt < max_attempts) : (attempt += 1) {
            if (tcpConnect(port)) |fd| {
                _ = linux.close(fd);
                ready = true;
                break;
            } else |_| {}
            const ts: linux.timespec = .{ .sec = 0, .nsec = 50 * std.time.ns_per_ms };
            _ = linux.nanosleep(&ts, null);
        }
    }
    if (!ready) {
        std.debug.print("driver: server never came up on 127.0.0.1:{d}\n", .{port});
        std.process.exit(1);
    }

    // ── Case 1: GET / -> 200 "Hello, world!\n" ────────────────────
    {
        var resp_buf: [4096]u8 = undefined;
        const resp = try sendReceiveOnce(port, "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n", resp_buf[0..]);
        try assertStatus(resp, 200);
        try assertBodyEquals(resp, "Hello, world!\n");
    }

    // ── Case 2: GET /missing -> 404 (empty body) ──────────────────
    {
        var resp_buf: [4096]u8 = undefined;
        const resp = try sendReceiveOnce(port, "GET /missing HTTP/1.1\r\nHost: localhost\r\n\r\n", resp_buf[0..]);
        try assertStatus(resp, 404);
        try assertBodyEquals(resp, "");
    }
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
    // Expect status line `HTTP/1.1 <code> ...\r\n`.
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
    // Find header terminator.
    const sep = "\r\n\r\n";
    const sep_idx = std.mem.indexOf(u8, resp, sep) orelse return error.MissingHeaderTerminator;
    const body = resp[sep_idx + sep.len ..];
    if (!std.mem.eql(u8, body, expected)) {
        std.debug.print("driver: body mismatch.\n  expected: {s}\n  got:      {s}\n", .{ expected, body });
        return error.BodyMismatch;
    }
}
