//! WASI HTTP types and handler interface.
//!
//! Implements wasi:http/types and wasi:http/handler for incoming and
//! outgoing HTTP request/response handling within components.

const std = @import("std");
const streams = @import("streams.zig");

// ── HTTP Method ─────────────────────────────────────────────────────────────

pub const Method = enum {
    GET,
    HEAD,
    POST,
    PUT,
    DELETE,
    CONNECT,
    OPTIONS,
    TRACE,
    PATCH,
};

// ── HTTP Headers ────────────────────────────────────────────────────────────

pub const HeaderEntry = struct {
    name: []const u8,
    value: []const u8,
};

pub const Headers = struct {
    entries: std.ArrayListUnmanaged(HeaderEntry) = .empty,

    pub fn append(self: *Headers, name: []const u8, value: []const u8, allocator: std.mem.Allocator) !void {
        try self.entries.append(allocator, .{ .name = name, .value = value });
    }

    pub fn get(self: *const Headers, name: []const u8) ?[]const u8 {
        for (self.entries.items) |e| {
            if (std.ascii.eqlIgnoreCase(e.name, name)) return e.value;
        }
        return null;
    }

    pub fn deinit(self: *Headers, allocator: std.mem.Allocator) void {
        self.entries.deinit(allocator);
    }
};

// ── HTTP Request ────────────────────────────────────────────────────────────

pub const Request = struct {
    method: Method = .GET,
    path: []const u8 = "/",
    headers: Headers = .{},
    body: ?*streams.InputStream = null,
    scheme: Scheme = .http,
    authority: []const u8 = "",

    pub const Scheme = enum { http, https };
};

// ── HTTP Response ───────────────────────────────────────────────────────────

pub const Response = struct {
    status: u16 = 200,
    headers: Headers = .{},
    body: ?*streams.OutputStream = null,
};

// ── HTTP Handler ────────────────────────────────────────────────────────────

/// Handler function type for incoming HTTP requests.
pub const IncomingHandler = *const fn (request: *const Request, allocator: std.mem.Allocator) Response;

/// A simple handler registry that routes requests to handlers by path prefix.
pub const Router = struct {
    routes: std.ArrayListUnmanaged(Route) = .empty,

    pub const Route = struct {
        path_prefix: []const u8,
        handler: IncomingHandler,
    };

    pub fn addRoute(self: *Router, prefix: []const u8, handler: IncomingHandler, allocator: std.mem.Allocator) !void {
        try self.routes.append(allocator, .{ .path_prefix = prefix, .handler = handler });
    }

    pub fn handle(self: *const Router, request: *const Request, allocator: std.mem.Allocator) Response {
        for (self.routes.items) |route| {
            if (std.mem.startsWith(u8, request.path, route.path_prefix)) {
                return route.handler(request, allocator);
            }
        }
        return .{ .status = 404 };
    }

    pub fn deinit(self: *Router, allocator: std.mem.Allocator) void {
        self.routes.deinit(allocator);
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "Headers: append and get" {
    const allocator = std.testing.allocator;
    var h = Headers{};
    defer h.deinit(allocator);

    try h.append("Content-Type", "text/plain", allocator);
    try h.append("X-Custom", "value", allocator);

    try std.testing.expectEqualStrings("text/plain", h.get("content-type").?);
    try std.testing.expect(h.get("missing") == null);
}

test "Router: routes by prefix" {
    const allocator = std.testing.allocator;
    var router = Router{};
    defer router.deinit(allocator);

    try router.addRoute("/api", &handleApi, allocator);

    const req = Request{ .path = "/api/users" };
    const resp = router.handle(&req, allocator);
    try std.testing.expectEqual(@as(u16, 200), resp.status);

    const req2 = Request{ .path = "/unknown" };
    const resp2 = router.handle(&req2, allocator);
    try std.testing.expectEqual(@as(u16, 404), resp2.status);
}

fn handleApi(_: *const Request, _: std.mem.Allocator) Response {
    return .{ .status = 200 };
}
