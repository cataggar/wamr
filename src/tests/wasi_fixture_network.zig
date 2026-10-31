//! Deadline-bound TCP operations for WASI fixtures.
const std = @import("std");
const Io = std.Io;
const net = Io.net;

const ConnectResult = net.IpAddress.ConnectError || net.HostName.LookupError || net.HostName.ValidateError;
const Race = union(enum) {
    connected: ConnectResult!net.Stream,
    timer: Io.Cancelable!void,
};

fn resolveAndConnect(io: Io, host: []const u8, port: u16) ConnectResult!net.Stream {
    const address: net.IpAddress = if (net.Ip4Address.parse(host, port)) |ip|
        .{ .ip4 = ip }
    else |_| blk: {
        const name = try net.HostName.init(host);
        var buffer: [16]net.HostName.LookupResult = undefined;
        var queue: Io.Queue(net.HostName.LookupResult) = .init(&buffer);
        try name.lookup(io, &queue, .{ .port = port, .family = .ip4 });
        while (queue.getOneUncancelable(io)) |result| switch (result) {
            .address => |ip| break :blk ip,
            .canonical_name => {},
        } else |_| return error.NoAddressReturned;
    };
    return address.connect(io, .{ .mode = .stream });
}

fn sleep(io: Io, milliseconds: u32) Io.Cancelable!void {
    try io.sleep(.fromMilliseconds(milliseconds), .awake);
}

/// Resolves IPv4 names and connects within one deadline.
pub fn connect(io: Io, host: []const u8, port: u16, milliseconds: u32) !net.Stream {
    var buffer: [2]Race = undefined;
    var select: Io.Select(Race) = .init(io, &buffer);
    defer while (select.cancel()) |result| switch (result) {
        .connected => |connection| {
            if (connection) |stream| stream.close(io) else |_| {}
        },
        .timer => {},
    };
    try select.concurrent(.connected, resolveAndConnect, .{ io, host, port });
    try select.concurrent(.timer, sleep, .{ io, milliseconds });
    return switch (try select.await()) {
        .connected => |result| try result,
        .timer => |result| {
            try result;
            return error.Timeout;
        },
    };
}

pub fn withDeadline(
    comptime T: type,
    io: Io,
    milliseconds: u32,
    comptime function: anytype,
    args: std.meta.ArgsTuple(@TypeOf(function)),
) !T {
    const Result = union(enum) { operation: anyerror!T, timer: Io.Cancelable!void };
    var buffer: [2]Result = undefined;
    var select: Io.Select(Result) = .init(io, &buffer);
    defer select.cancelDiscard();
    try select.concurrent(.operation, function, args);
    try select.concurrent(.timer, sleep, .{ io, milliseconds });
    return switch (try select.await()) {
        .operation => |result| try result,
        .timer => |result| {
            try result;
            return error.Timeout;
        },
    };
}

fn sendAll(io: Io, stream: net.Stream, bytes: []const u8) anyerror!void {
    var written: usize = 0;
    while (written < bytes.len) {
        const data = [_][]const u8{bytes[written..]};
        const result = try io.operate(.{ .net_write = .{ .socket_handle = stream.socket.handle, .data = &data } });
        const count = try result.net_write;
        if (count == 0) return error.WriteDidNotProgress;
        written += count;
    }
}

/// Sends all bytes under one deadline, as Python's socket.sendall does.
pub fn send(io: Io, stream: net.Stream, bytes: []const u8, milliseconds: u32) !void {
    try withDeadline(void, io, milliseconds, sendAll, .{ io, stream, bytes });
}

fn recvOnce(io: Io, stream: net.Stream, buffer: []u8) anyerror!usize {
    var data = [_][]u8{buffer};
    const result = try io.operate(.{ .net_read = .{ .socket_handle = stream.socket.handle, .data = &data } });
    return (try result.net_read).data_len;
}

/// Performs one socket read, which may return less than the requested length.
pub fn recv(io: Io, stream: net.Stream, buffer: []u8, milliseconds: u32) !usize {
    if (buffer.len == 0) return 0;
    return withDeadline(usize, io, milliseconds, recvOnce, .{ io, stream, buffer });
}

test "TCP send, single receive, EOF, and receive timeout" {
    const io = std.testing.io;
    const address: net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try address.listen(io, .{});
    defer server.deinit(io);
    const client = try connect(io, "127.0.0.1", server.socket.address.getPort(), 5000);
    defer client.close(io);
    const peer = try server.accept(io);
    var peer_open = true;
    defer if (peer_open) peer.close(io);

    try send(io, client, "hello", 5000);
    var buffer: [16]u8 = undefined;
    const received = try recv(io, peer, &buffer, 5000);
    try std.testing.expectEqualStrings("hello", buffer[0..received]);
    try std.testing.expectError(error.Timeout, recv(io, client, &buffer, 100));
    try send(io, peer, "ok", 5000);
    const reply = try recv(io, client, &buffer, 5000);
    try std.testing.expectEqualStrings("ok", buffer[0..reply]);
    try std.testing.expectEqual(@as(usize, 0), try recv(io, client, buffer[0..0], 5000));
    peer.close(io);
    peer_open = false;
    try std.testing.expectEqual(@as(usize, 0), try recv(io, client, &buffer, 5000));
}

test "connection failures do not leak a socket" {
    const io = std.testing.io;
    try std.testing.expectError(error.InvalidHostName, connect(io, "bad hostname", 1, 5000));
}

test "blocked TCP sends honor their deadline" {
    const io = std.testing.io;
    const address: net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try address.listen(io, .{});
    defer server.deinit(io);
    const client = try connect(io, "localhost", server.socket.address.getPort(), 5000);
    defer client.close(io);
    const peer = try server.accept(io);
    defer peer.close(io);
    const bytes = try std.testing.allocator.alloc(u8, 16 * 1024 * 1024);
    defer std.testing.allocator.free(bytes);
    @memset(bytes, 'x');
    try std.testing.expectError(error.Timeout, send(io, client, bytes, 100));
}
