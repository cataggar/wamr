//! WASI sockets — TCP, UDP, and IP name lookup.
//!
//! Implements wasi:sockets/tcp, wasi:sockets/udp, and
//! wasi:sockets/ip-name-lookup as resource types.

const std = @import("std");

// ── IP Address types ────────────────────────────────────────────────────────

pub const IpAddress = union(enum) {
    ipv4: [4]u8,
    ipv6: [16]u8,

    pub fn format(self: IpAddress, buf: []u8) []const u8 {
        switch (self) {
            .ipv4 => |addr| {
                const n = std.fmt.bufPrint(buf, "{}.{}.{}.{}", .{
                    addr[0], addr[1], addr[2], addr[3],
                }) catch return "";
                return n;
            },
            .ipv6 => return "::1", // simplified
        }
    }
};

pub const IpSocketAddress = struct {
    address: IpAddress,
    port: u16,
};

// ── TCP Socket ──────────────────────────────────────────────────────────────

pub const TcpSocket = struct {
    state: State = .closed,
    local_address: ?IpSocketAddress = null,
    remote_address: ?IpSocketAddress = null,

    pub const State = enum {
        closed,
        bound,
        listening,
        connected,
    };

    pub const Error = error{
        WouldBlock,
        ConnectionRefused,
        AddressInUse,
        InvalidState,
        NotConnected,
    };

    pub fn bind(self: *TcpSocket, addr: IpSocketAddress) Error!void {
        if (self.state != .closed) return error.InvalidState;
        self.local_address = addr;
        self.state = .bound;
    }

    pub fn listen(self: *TcpSocket) Error!void {
        if (self.state != .bound) return error.InvalidState;
        self.state = .listening;
    }

    pub fn connect(self: *TcpSocket, addr: IpSocketAddress) Error!void {
        if (self.state != .closed and self.state != .bound) return error.InvalidState;
        self.remote_address = addr;
        self.state = .connected;
    }

    pub fn shutdown(self: *TcpSocket) void {
        self.state = .closed;
    }
};

// ── UDP Socket ──────────────────────────────────────────────────────────────

pub const UdpSocket = struct {
    state: State = .closed,
    local_address: ?IpSocketAddress = null,

    pub const State = enum { closed, bound };

    pub const Datagram = struct {
        data: []const u8,
        remote_address: IpSocketAddress,
    };

    pub fn bind(self: *UdpSocket, addr: IpSocketAddress) !void {
        self.local_address = addr;
        self.state = .bound;
    }
};

// ── IP Name Lookup ──────────────────────────────────────────────────────────

pub const IpNameLookup = struct {
    /// Resolve a hostname to IP addresses (stub — returns loopback).
    pub fn resolveAddresses(_: []const u8) []const IpAddress {
        return &[_]IpAddress{.{ .ipv4 = .{ 127, 0, 0, 1 } }};
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "TcpSocket: bind-listen lifecycle" {
    var sock = TcpSocket{};
    try sock.bind(.{ .address = .{ .ipv4 = .{ 0, 0, 0, 0 } }, .port = 8080 });
    try std.testing.expectEqual(TcpSocket.State.bound, sock.state);
    try sock.listen();
    try std.testing.expectEqual(TcpSocket.State.listening, sock.state);
}

test "TcpSocket: connect" {
    var sock = TcpSocket{};
    try sock.connect(.{ .address = .{ .ipv4 = .{ 127, 0, 0, 1 } }, .port = 80 });
    try std.testing.expectEqual(TcpSocket.State.connected, sock.state);
    sock.shutdown();
    try std.testing.expectEqual(TcpSocket.State.closed, sock.state);
}

test "TcpSocket: invalid state transitions" {
    var sock = TcpSocket{};
    try std.testing.expectError(error.InvalidState, sock.listen()); // can't listen before bind
}

test "IpNameLookup: resolves to loopback" {
    const addrs = IpNameLookup.resolveAddresses("localhost");
    try std.testing.expectEqual(@as(usize, 1), addrs.len);
    try std.testing.expectEqual([4]u8{ 127, 0, 0, 1 }, addrs[0].ipv4);
}
