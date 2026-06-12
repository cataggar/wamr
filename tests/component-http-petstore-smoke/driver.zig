//! End-to-end smoke driver for `examples/zig-http-petstore`.
//!
//! Spawns `<wamr_exe> serve --addr=127.0.0.1:<port> <component.wasm>` as
//! a subprocess and verifies the TypeSpec petstore API behaviour:
//!
//!   GET    /pets          -> 200, JSON list containing "Fluffy" + "Rex"
//!   GET    /pets/1        -> 200 {"name":"Fluffy","tag":"cat","age":3}
//!   GET    /pets/99       -> 404 Error ("code":404)
//!   POST   /pets          -> 200, echoes the created pet
//!   GET    /pets/1/toys   -> 200, JSON list containing "Ball"
//!   DELETE /pets/2        -> 200 (empty body)
//!
//! Mirrors `tests/component-http-smoke/driver.zig` (the zig-http
//! driver). Exit 0 on success; non-zero (with a message on stderr) on
//! any mismatch or RPC failure.
//!
//! Usage: `component-http-petstore-smoke-driver <wamr_exe> <component.wasm> <port>`

const std = @import("std");
const linux = std.os.linux;
const builtin = @import("builtin");

const usage = "usage: component-http-petstore-smoke-driver <wamr_exe> <component.wasm> <port>\n";

pub fn main(init: std.process.Init) !void {
    if (comptime builtin.os.tag != .linux) {
        std.debug.print("component-http-petstore-smoke-driver only runs on Linux\n", .{});
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

    const addr_arg = try std.fmt.allocPrint(allocator, "--addr=127.0.0.1:{d}", .{port});
    defer allocator.free(addr_arg);

    var child = try std.process.spawn(io, .{
        .argv = &.{ wamr_exe, "serve", addr_arg, component_path },
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    defer child.kill(io);

    // Wait for the server to bind.
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

    // ── GET /pets -> 200, list with both seeded pets ──────────────
    {
        var buf: [4096]u8 = undefined;
        const resp = try sendReceiveOnce(port, "GET /pets HTTP/1.1\r\nHost: localhost\r\n\r\n", buf[0..]);
        try assertStatus(resp, 200);
        try assertBodyContains(resp, "\"Fluffy\"");
        try assertBodyContains(resp, "\"Rex\"");
    }

    // ── GET /pets/1 -> 200 exact Pet ──────────────────────────────
    {
        var buf: [4096]u8 = undefined;
        const resp = try sendReceiveOnce(port, "GET /pets/1 HTTP/1.1\r\nHost: localhost\r\n\r\n", buf[0..]);
        try assertStatus(resp, 200);
        try assertBodyEquals(resp, "{\"name\":\"Fluffy\",\"tag\":\"cat\",\"age\":3}");
    }

    // ── GET /pets/99 -> 404 Error ─────────────────────────────────
    {
        var buf: [4096]u8 = undefined;
        const resp = try sendReceiveOnce(port, "GET /pets/99 HTTP/1.1\r\nHost: localhost\r\n\r\n", buf[0..]);
        try assertStatus(resp, 404);
        try assertBodyContains(resp, "\"code\":404");
    }

    // ── POST /pets -> 200, echoes created pet ─────────────────────
    {
        var buf: [4096]u8 = undefined;
        const body = "{\"name\":\"Buddy\",\"tag\":\"dog\",\"age\":2}";
        const req = try std.fmt.allocPrint(
            allocator,
            "POST /pets HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}",
            .{ body.len, body },
        );
        defer allocator.free(req);
        const resp = try sendReceiveOnce(port, req, buf[0..]);
        try assertStatus(resp, 200);
        try assertBodyEquals(resp, "{\"name\":\"Buddy\",\"tag\":\"dog\",\"age\":2}");
    }

    // ── GET /pets/1/toys -> 200, list with "Ball" ─────────────────
    {
        var buf: [4096]u8 = undefined;
        const resp = try sendReceiveOnce(port, "GET /pets/1/toys HTTP/1.1\r\nHost: localhost\r\n\r\n", buf[0..]);
        try assertStatus(resp, 200);
        try assertBodyContains(resp, "\"Ball\"");
    }

    // ── GET /pets/1/toys?nameFilter=Mouse -> 200, only "Mouse" ────
    {
        var buf: [4096]u8 = undefined;
        const resp = try sendReceiveOnce(port, "GET /pets/1/toys?nameFilter=Mouse HTTP/1.1\r\nHost: localhost\r\n\r\n", buf[0..]);
        try assertStatus(resp, 200);
        try assertBodyContains(resp, "\"Mouse\"");
        try assertBodyExcludes(resp, "\"Ball\"");
    }

    // ── DELETE /pets/2 -> 200 (empty body) ────────────────────────
    {
        var buf: [4096]u8 = undefined;
        const resp = try sendReceiveOnce(port, "DELETE /pets/2 HTTP/1.1\r\nHost: localhost\r\n\r\n", buf[0..]);
        try assertStatus(resp, 200);
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

fn bodyOf(resp: []const u8) ![]const u8 {
    const sep = "\r\n\r\n";
    const sep_idx = std.mem.indexOf(u8, resp, sep) orelse return error.MissingHeaderTerminator;
    return resp[sep_idx + sep.len ..];
}

fn assertBodyEquals(resp: []const u8, expected: []const u8) !void {
    const body = try bodyOf(resp);
    if (!std.mem.eql(u8, body, expected)) {
        std.debug.print("driver: body mismatch.\n  expected: {s}\n  got:      {s}\n", .{ expected, body });
        return error.BodyMismatch;
    }
}

fn assertBodyContains(resp: []const u8, needle: []const u8) !void {
    const body = try bodyOf(resp);
    if (std.mem.indexOf(u8, body, needle) == null) {
        std.debug.print("driver: body missing {s}\n  got: {s}\n", .{ needle, body });
        return error.BodyMissingNeedle;
    }
}

fn assertBodyExcludes(resp: []const u8, needle: []const u8) !void {
    const body = try bodyOf(resp);
    if (std.mem.indexOf(u8, body, needle) != null) {
        std.debug.print("driver: body unexpectedly contains {s}\n  got: {s}\n", .{ needle, body });
        return error.BodyHasNeedle;
    }
}
