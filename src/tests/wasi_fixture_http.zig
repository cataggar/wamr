//! HTTP response collection for the WASI fixture runner.
const std = @import("std");
const network = @import("wasi_fixture_network.zig");
const fixture = @import("wasi_fixture_config.zig");
const Allocator = std.mem.Allocator;

pub const Header = struct { name: []const u8, value: []const u8 };

pub const Response = struct {
    arena: std.heap.ArenaAllocator,
    status: u16,
    headers: []const Header,
    body: []const u8,

    pub fn deinit(self: *Response) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const Mismatch = union(enum) {
    status: struct { expected: i64, actual: u16 },
    missing_header: []const u8,
    header: struct { name: []const u8, expected: []const u8, actual: []const u8 },
    body: struct { expected: []const u8, actual: []const u8 },
};

fn headerValue(allocator: Allocator, headers: []const Header, name: []const u8) !?[]const u8 {
    var value: std.ArrayList(u8) = .empty;
    var found = false;
    for (headers) |header| {
        if (!std.ascii.eqlIgnoreCase(header.name, name)) continue;
        if (found) try value.appendSlice(allocator, ", ");
        try value.appendSlice(allocator, header.value);
        found = true;
    }
    return if (found) try value.toOwnedSlice(allocator) else null;
}

fn decodeUtf8Replacing(allocator: Allocator, bytes: []const u8, ascii: bool) ![]const u8 {
    var output: std.ArrayList(u8) = .empty;
    var index: usize = 0;
    while (index < bytes.len) {
        const first = bytes[index];
        if (first < 0x80) {
            try output.append(allocator, first);
            index += 1;
            continue;
        }
        const length: usize = if (ascii or first < 0xc2 or first > 0xf4)
            0
        else
            std.unicode.utf8ByteSequenceLength(first) catch 0;
        var prefix: usize = 1;
        if (length >= 2 and index + 1 < bytes.len) {
            const second = bytes[index + 1];
            const valid_second = second >= 0x80 and second <= 0xbf and
                (first != 0xe0 or second >= 0xa0) and
                (first != 0xed or second <= 0x9f) and
                (first != 0xf0 or second >= 0x90) and
                (first != 0xf4 or second <= 0x8f);
            if (valid_second) {
                prefix = 2;
                while (prefix < length and index + prefix < bytes.len and
                    bytes[index + prefix] >= 0x80 and bytes[index + prefix] <= 0xbf) : (prefix += 1)
                {}
            }
        }
        if (length > 1 and prefix == length) {
            try output.appendSlice(allocator, bytes[index .. index + length]);
        } else {
            try output.appendSlice(allocator, "\xef\xbf\xbd");
        }
        index += prefix;
    }
    return output.toOwnedSlice(allocator);
}

fn decodeUtf16Replacing(allocator: Allocator, input: []const u8, order: std.builtin.Endian) ![]const u8 {
    var output: std.ArrayList(u8) = .empty;
    var index: usize = 0;
    while (index < input.len) {
        var codepoint: u21 = 0xfffd;
        if (input.len - index >= 2) {
            const first = std.mem.readInt(u16, input[index..][0..2], order);
            index += 2;
            if (first >= 0xd800 and first <= 0xdbff) {
                if (input.len - index >= 2) {
                    const second = std.mem.readInt(u16, input[index..][0..2], order);
                    if (second >= 0xdc00 and second <= 0xdfff) {
                        codepoint = 0x10000 + (@as(u21, first - 0xd800) << 10) + (second - 0xdc00);
                        index += 2;
                    }
                } else if (input.len - index == 1) {
                    index += 1;
                }
            } else if (first < 0xdc00 or first > 0xdfff) {
                codepoint = first;
            }
        } else {
            index += 1;
        }
        var bytes: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(codepoint, &bytes);
        try output.appendSlice(allocator, bytes[0..len]);
    }
    return output.toOwnedSlice(allocator);
}

fn decodeUtf32Replacing(allocator: Allocator, input: []const u8, order: std.builtin.Endian) ![]const u8 {
    var output: std.ArrayList(u8) = .empty;
    var index: usize = 0;
    while (index < input.len) {
        var value: u32 = 0xfffd;
        if (input.len - index >= 4) {
            value = std.mem.readInt(u32, input[index..][0..4], order);
            index += 4;
            if (value > 0x10ffff or (value >= 0xd800 and value <= 0xdfff)) value = 0xfffd;
        } else {
            index = input.len;
        }
        var bytes: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(@intCast(value), &bytes);
        try output.appendSlice(allocator, bytes[0..len]);
    }
    return output.toOwnedSlice(allocator);
}

fn charsetEqual(value: []const u8, expected: []const u8) bool {
    if (value.len != expected.len) return false;
    for (value, expected) |actual, wanted| {
        const normalized = if (actual == '_' or actual == ' ') '-' else std.ascii.toLower(actual);
        if (normalized != wanted) return false;
    }
    return true;
}

fn decodeWindows1252(allocator: Allocator, input: []const u8) ![]const u8 {
    const controls = [_]u21{
        0x20ac, 0xfffd, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021,
        0x02c6, 0x2030, 0x0160, 0x2039, 0x0152, 0xfffd, 0x017d, 0xfffd,
        0xfffd, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014,
        0x02dc, 0x2122, 0x0161, 0x203a, 0x0153, 0xfffd, 0x017e, 0x0178,
    };
    var output: std.ArrayList(u8) = .empty;
    for (input) |byte| {
        const codepoint: u21 = if (byte >= 0x80 and byte < 0xa0) controls[byte - 0x80] else byte;
        var encoded: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(codepoint, &encoded);
        try output.appendSlice(allocator, encoded[0..len]);
    }
    return output.toOwnedSlice(allocator);
}

fn decodeLatin9(allocator: Allocator, input: []const u8) ![]const u8 {
    var output: std.ArrayList(u8) = .empty;
    for (input) |byte| {
        const codepoint: u21 = switch (byte) {
            0xa4 => 0x20ac,
            0xa6 => 0x0160,
            0xa8 => 0x0161,
            0xb4 => 0x017d,
            0xb8 => 0x017e,
            0xbc => 0x0152,
            0xbd => 0x0153,
            0xbe => 0x0178,
            else => byte,
        };
        var encoded: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(codepoint, &encoded);
        try output.appendSlice(allocator, encoded[0..len]);
    }
    return output.toOwnedSlice(allocator);
}

fn responseText(allocator: Allocator, response: *const Response) ![]const u8 {
    const content_type = try headerValue(allocator, response.headers, "content-type");
    var charset: ?[]const u8 = null;
    var invalid_charset = false;
    var media_type: []const u8 = "";
    if (content_type) |value| {
        var parts = std.mem.splitScalar(u8, value, ';');
        media_type = std.mem.trim(u8, parts.next().?, " \t");
        while (parts.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t");
            const eq = std.mem.indexOfScalar(u8, trimmed, '=') orelse {
                if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, trimmed, " \t\"'"), "charset"))
                    invalid_charset = true;
                continue;
            };
            if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, trimmed[0..eq], " \t\"'"), "charset")) {
                charset = std.mem.trim(u8, trimmed[eq + 1 ..], " \t\"'");
                invalid_charset = false;
            }
        }
    }
    if (invalid_charset) return error.InvalidHttpCharset;
    if (charset == null) {
        if (std.mem.indexOf(u8, media_type, "text") != null) {
            charset = "iso-8859-1";
        } else if (std.mem.indexOf(u8, media_type, "application/json") != null) {
            charset = "utf-8";
        }
    }
    if (response.body.len == 0) return "";
    if (charset) |value| {
        if (charsetEqual(value, "cp1252") or charsetEqual(value, "windows-1252"))
            return decodeWindows1252(allocator, response.body);
        if (charsetEqual(value, "iso-8859-15") or charsetEqual(value, "iso8859-15") or
            charsetEqual(value, "latin9") or charsetEqual(value, "l9"))
            return decodeLatin9(allocator, response.body);
        if (charsetEqual(value, "utf-8-sig")) {
            const body = if (std.mem.startsWith(u8, response.body, "\xef\xbb\xbf")) response.body[3..] else response.body;
            if (std.unicode.utf8ValidateSlice(body)) return body;
            return decodeUtf8Replacing(allocator, body, false);
        }
        const auto = charsetEqual(value, "utf-16") or charsetEqual(value, "utf16");
        const little = charsetEqual(value, "utf-16le") or charsetEqual(value, "utf-16-le");
        const big = charsetEqual(value, "utf-16be") or charsetEqual(value, "utf-16-be");
        if (auto or little or big) {
            var body = response.body;
            var order: std.builtin.Endian = if (auto) @import("builtin").target.cpu.arch.endian() else if (big) .big else .little;
            if (auto and body.len >= 2) {
                if (std.mem.eql(u8, body[0..2], "\xff\xfe")) {
                    order = .little;
                    body = body[2..];
                } else if (std.mem.eql(u8, body[0..2], "\xfe\xff")) {
                    order = .big;
                    body = body[2..];
                }
            }
            return decodeUtf16Replacing(allocator, body, order);
        }
        const auto32 = charsetEqual(value, "utf-32") or charsetEqual(value, "utf32");
        const little32 = charsetEqual(value, "utf-32le") or charsetEqual(value, "utf-32-le");
        const big32 = charsetEqual(value, "utf-32be") or charsetEqual(value, "utf-32-be");
        if (auto32 or little32 or big32) {
            var body = response.body;
            var order: std.builtin.Endian = if (auto32) @import("builtin").target.cpu.arch.endian() else if (big32) .big else .little;
            if (auto32 and body.len >= 4) {
                if (std.mem.eql(u8, body[0..4], "\xff\xfe\x00\x00")) {
                    order = .little;
                    body = body[4..];
                } else if (std.mem.eql(u8, body[0..4], "\x00\x00\xfe\xff")) {
                    order = .big;
                    body = body[4..];
                }
            }
            return decodeUtf32Replacing(allocator, body, order);
        }
    }
    const latin1 = if (charset) |value|
        charsetEqual(value, "iso-8859-1") or charsetEqual(value, "iso8859-1") or
            charsetEqual(value, "latin-1") or charsetEqual(value, "latin1")
    else
        false;
    const utf8 = if (charset) |value|
        charsetEqual(value, "utf-8") or charsetEqual(value, "utf8") or
            charsetEqual(value, "ascii") or charsetEqual(value, "us-ascii") or value.len == 0
    else
        !latin1;
    if (!latin1 and !utf8) return error.UnsupportedHttpCharset;
    if (latin1) {
        var text: std.ArrayList(u8) = .empty;
        for (response.body) |byte| {
            var encoded: [2]u8 = undefined;
            const len = try std.unicode.utf8Encode(byte, &encoded);
            try text.appendSlice(allocator, encoded[0..len]);
        }
        return text.toOwnedSlice(allocator);
    }
    if (charset == null) {
        for (response.body) |byte| if (byte >= 0x80) return error.UnsupportedHttpDecoding;
        return response.body;
    }
    const ascii = charsetEqual(charset.?, "ascii") or charsetEqual(charset.?, "us-ascii");
    if (!ascii and std.unicode.utf8ValidateSlice(response.body)) return response.body;
    return decodeUtf8Replacing(allocator, response.body, ascii);
}

/// Checks common response encodings. Ambiguous text decoding remains explicit.
pub fn compare(response: *Response, expected: fixture.Request) !?Mismatch {
    if (response.status != expected.status) return .{ .status = .{
        .expected = expected.status,
        .actual = response.status,
    } };
    const allocator = response.arena.allocator();
    for (expected.headers) |header| {
        const value = try headerValue(allocator, response.headers, header.name) orelse
            return .{ .missing_header = header.name };
        if (!std.mem.eql(u8, header.value, value)) return .{ .header = .{
            .name = header.name,
            .expected = header.value,
            .actual = value,
        } };
    }
    const text = try responseText(allocator, response);
    if (!std.mem.eql(u8, expected.body, text)) return .{ .body = .{ .expected = expected.body, .actual = text } };
    return null;
}

const Collected = struct {
    status: u16 = undefined,
    headers: []const Header = undefined,
    body: []const u8 = undefined,
};

fn unreserved(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or std.mem.indexOfScalar(u8, "-._~", byte) != null;
}

fn requoteUri(allocator: Allocator, input: []const u8) ![]const u8 {
    if (!std.unicode.utf8ValidateSlice(input)) return error.InvalidHttpRedirectLocation;
    var invalid_escape = false;
    for (input, 0..) |byte, index| {
        if (byte == '%' and ((index + 1 < input.len and input[index + 1] >= 0x80) or
            (index + 2 < input.len and input[index + 2] >= 0x80)))
            return error.UnsupportedHttpRedirectLocation;
        if (byte == '%' and input.len - index >= 3 and
            std.ascii.isAlphanumeric(input[index + 1]) and std.ascii.isAlphanumeric(input[index + 2]) and
            (!std.ascii.isHex(input[index + 1]) or !std.ascii.isHex(input[index + 2])))
        {
            invalid_escape = true;
            break;
        }
    }
    var result: std.ArrayList(u8) = .empty;
    var index: usize = 0;
    while (index < input.len) {
        var byte = input[index];
        if (!invalid_escape and byte == '%' and input.len - index >= 3 and
            std.ascii.isHex(input[index + 1]) and std.ascii.isHex(input[index + 2]))
        {
            const decoded = try std.fmt.parseInt(u8, input[index + 1 ..][0..2], 16);
            if (unreserved(decoded)) {
                byte = decoded;
                index += 2;
            }
        }
        if (unreserved(byte) or std.mem.indexOfScalar(u8, "!#$&'()*+,/:;=?@[]", byte) != null or
            (byte == '%' and !invalid_escape))
        {
            try result.append(allocator, byte);
        } else {
            const hex = std.fmt.bytesToHex([_]u8{byte}, .upper);
            try result.appendSlice(allocator, &.{ '%', hex[0], hex[1] });
        }
        index += 1;
    }
    return result.toOwnedSlice(allocator);
}

fn redirectUri(allocator: Allocator, base: std.Uri, raw_location: []const u8) !std.Uri {
    const location = try requoteUri(allocator, raw_location);
    const location_space = try std.math.mul(usize, location.len, 2);
    const buffer_size = try std.math.add(usize, try std.math.add(usize, location_space, base.path.percent_encoded.len), 2);
    const buffer = try allocator.alloc(u8, buffer_size);
    @memcpy(buffer[0..location.len], location);
    var remaining = buffer;
    var uri = try base.resolveInPlace(location.len, &remaining);
    if (uri.user != null or uri.password != null) return error.UnsupportedHttpRedirectCredentials;
    if (uri.fragment == null) uri.fragment = base.fragment;
    return uri;
}

fn redirectMethod(status: u16, method: std.http.Method) std.http.Method {
    if ((status == 302 or status == 303) and method != .HEAD) return .GET;
    if (status == 301 and method == .POST) return .GET;
    return method;
}

fn collect(
    allocator: Allocator,
    io: std.Io,
    url: []const u8,
    method_name: []const u8,
    out: *Collected,
) anyerror!void {
    const upper = try allocator.alloc(u8, method_name.len);
    _ = std.ascii.upperString(upper, method_name);
    var method = std.meta.stringToEnum(std.http.Method, upper) orelse return error.UnsupportedHttpMethod;
    var uri = try std.Uri.parse(try requoteUri(allocator, url));
    if (uri.user != null or uri.password != null) return error.UnsupportedHttpRequestCredentials;
    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var redirects: usize = 0;
    while (true) {
        var req = try client.request(method, uri, .{
            .keep_alive = false,
            .redirect_behavior = .unhandled,
        });
        defer {
            if (req.connection) |connection| connection.closing = true;
            req.deinit();
        }
        if (method.requestHasBody()) {
            try req.sendBodyComplete(&.{});
        } else {
            try req.sendBodiless();
        }
        var response = try req.receiveHead(&.{});
        const status: u16 = @backingInt(response.head.status);
        const is_redirect = switch (status) {
            301, 302, 303, 307, 308 => true,
            else => false,
        };
        const location = if (is_redirect and response.head.location != null and response.head.location.?.len != 0)
            try allocator.dupe(u8, response.head.location.?)
        else
            null;
        var headers: std.ArrayList(Header) = .empty;
        var iter = response.head.iterateHeaders();
        while (iter.next()) |header| {
            if (iter.is_trailer) break;
            if (location != null and std.ascii.eqlIgnoreCase(header.name, "set-cookie"))
                return error.UnsupportedHttpRedirectCookie;
            try headers.append(allocator, .{
                .name = try allocator.dupe(u8, header.name),
                .value = try allocator.dupe(u8, header.value),
            });
        }
        var body: std.Io.Writer.Allocating = .init(allocator);
        if (method != .HEAD and (status < 100 or status >= 200) and status != 204 and status != 304) {
            const decompression: []u8 = switch (response.head.content_encoding) {
                .identity => &.{},
                .zstd => try allocator.alloc(u8, std.compress.zstd.default_window_len),
                .deflate, .gzip => try allocator.alloc(u8, std.compress.flate.max_window_len),
                .compress => return error.UnsupportedCompressionMethod,
            };
            var transfer_buffer: [64]u8 = undefined;
            var decompress: std.http.Decompress = undefined;
            const reader = response.readerDecompressing(&transfer_buffer, &decompress, decompression);
            _ = reader.streamRemaining(&body.writer) catch |err| switch (err) {
                error.ReadFailed => {
                    try io.checkCancel();
                    return response.bodyErr() orelse error.HttpBodyReadFailed;
                },
                else => |e| return e,
            };
        }
        if (location) |target| {
            if (redirects == 30) return error.TooManyHttpRedirects;
            redirects += 1;
            uri = try redirectUri(allocator, uri, target);
            method = redirectMethod(status, method);
            continue;
        }
        out.* = .{
            .status = status,
            .headers = try headers.toOwnedSlice(allocator),
            .body = try body.toOwnedSlice(),
        };
        return;
    }
}

/// Collect a response within one deadline; the caller owns the result.
pub fn request(io: std.Io, url: []const u8, method: []const u8, milliseconds: u32) !Response {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    errdefer arena.deinit();
    var collected: Collected = .{};
    try network.withDeadline(void, io, milliseconds, collect, .{
        arena.allocator(), io, url, method, &collected,
    });
    return .{
        .arena = arena,
        .status = collected.status,
        .headers = collected.headers,
        .body = collected.body,
    };
}

fn serveOne(io: std.Io, server: *std.Io.net.Server) !void {
    const peer = try server.accept(io);
    defer peer.close(io);
    var request_buffer: [4096]u8 = undefined;
    _ = try network.recv(io, peer, &request_buffer, 5000);
    try network.send(io, peer, "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 3\r\nConnection: close\r\n\r\nhey", 5000);
}

fn serveQuotedRequest(io: std.Io, server: *std.Io.net.Server) !void {
    const peer = try server.accept(io);
    defer peer.close(io);
    var request_buffer: [4096]u8 = undefined;
    const count = try network.recv(io, peer, &request_buffer, 5000);
    if (!std.mem.startsWith(u8, request_buffer[0..count], "GET /a%20b/~/%2F/caf%C3%A9 HTTP/1.1\r\n"))
        return error.UnexpectedRequest;
    try network.send(io, peer, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", 5000);
}

fn serveStalled(io: std.Io, server: *std.Io.net.Server, send_headers: bool) !void {
    const peer = try server.accept(io);
    defer peer.close(io);
    var request_buffer: [4096]u8 = undefined;
    _ = try network.recv(io, peer, &request_buffer, 5000);
    if (send_headers) {
        try network.send(io, peer, "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial", 5000);
    }
    try io.sleep(.fromSeconds(5), .awake);
}

fn serveBodyless(io: std.Io, server: *std.Io.net.Server, status: u16) !void {
    const peer = try server.accept(io);
    defer peer.close(io);
    var request_buffer: [4096]u8 = undefined;
    _ = try network.recv(io, peer, &request_buffer, 5000);
    var response_buffer: [128]u8 = undefined;
    const response = try std.fmt.bufPrint(&response_buffer, "HTTP/1.1 {d} Test\r\nContent-Length: 100\r\nContent-Encoding: gzip\r\n\r\n", .{status});
    try network.send(io, peer, response, 5000);
    try io.sleep(.fromMilliseconds(500), .awake);
}

fn serveRedirect(io: std.Io, server: *std.Io.net.Server, status: u16, original: []const u8, redirected: []const u8, location: []const u8, final_path: []const u8) !void {
    for (0..2) |index| {
        const peer = try server.accept(io);
        defer peer.close(io);
        var request_buffer: [4096]u8 = undefined;
        const count = try network.recv(io, peer, &request_buffer, 5000);
        var expected_buffer: [128]u8 = undefined;
        const expected = try std.fmt.bufPrint(&expected_buffer, "{s} {s} HTTP/1.1\r\n", .{
            if (index == 0) original else redirected,
            if (index == 0) "/parent/start" else final_path,
        });
        if (!std.mem.startsWith(u8, request_buffer[0..count], expected)) return error.UnexpectedRequest;
        var response_buffer: [256]u8 = undefined;
        const response = if (index == 0)
            try std.fmt.bufPrint(&response_buffer, "HTTP/1.1 {d} Redirect\r\nLocation: {s}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{ status, location })
        else
            "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        try network.send(io, peer, response, 5000);
    }
}

fn serveRedirectCookie(io: std.Io, server: *std.Io.net.Server) !void {
    const peer = try server.accept(io);
    defer peer.close(io);
    var request_buffer: [4096]u8 = undefined;
    _ = try network.recv(io, peer, &request_buffer, 5000);
    try network.send(io, peer, "HTTP/1.1 302 Found\r\nLocation: /next\r\nSet-Cookie: token=abc\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", 5000);
}

fn serveRedirectLoop(io: std.Io, server: *std.Io.net.Server) !void {
    for (0..31) |_| {
        const peer = try server.accept(io);
        defer peer.close(io);
        var request_buffer: [4096]u8 = undefined;
        _ = try network.recv(io, peer, &request_buffer, 5000);
        try network.send(io, peer, "HTTP/1.1 302 Found\r\nLocation: /loop\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", 5000);
    }
}

test "collect HTTP status, headers, and body" {
    const io = std.testing.io;
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try address.listen(io, .{});
    defer server.deinit(io);
    var task = try io.concurrent(serveOne, .{ io, &server });
    var awaited = false;
    defer if (!awaited) {
        task.cancel(io) catch |err| {
            if (err != error.Canceled) std.debug.print("HTTP test server failed: {s}\n", .{@errorName(err)});
        };
    };
    var url_buffer: [100]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/", .{server.socket.address.getPort()});
    var result = try request(io, url, "GET", 5000);
    defer result.deinit();
    try std.testing.expectEqual(@as(u16, 200), result.status);
    try std.testing.expectEqualStrings("hey", result.body);
    try std.testing.expectEqualStrings("Content-Type", result.headers[0].name);
    try std.testing.expectEqualStrings("text/plain", result.headers[0].value);
    try task.await(io);
    awaited = true;
}

test "HEAD and bodyless status responses do not wait for advertised bodies" {
    const io = std.testing.io;
    for ([_]struct { method: []const u8, status: u16 }{
        .{ .method = "HEAD", .status = 200 },
        .{ .method = "GET", .status = 204 },
        .{ .method = "GET", .status = 304 },
    }) |case| {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
        var server = try address.listen(io, .{});
        defer server.deinit(io);
        var task = try io.concurrent(serveBodyless, .{ io, &server, case.status });
        defer task.cancel(io) catch |err| {
            if (err != error.Canceled) std.debug.print("HTTP test server failed: {s}\n", .{@errorName(err)});
        };
        var url_buffer: [100]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/", .{server.socket.address.getPort()});
        var response = try request(io, url, case.method, 250);
        defer response.deinit();
        try std.testing.expectEqual(case.status, response.status);
        try std.testing.expectEqualStrings("", response.body);
    }
}

test "initial request paths are quoted before sending" {
    const io = std.testing.io;
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try address.listen(io, .{});
    defer server.deinit(io);
    var task = try io.concurrent(serveQuotedRequest, .{ io, &server });
    var awaited = false;
    defer if (!awaited) {
        task.cancel(io) catch |err| {
            if (err != error.Canceled) std.debug.print("HTTP test server failed: {s}\n", .{@errorName(err)});
        };
    };
    var url_buffer: [128]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/a b/%7e/%2F/caf\xc3\xa9", .{server.socket.address.getPort()});
    var response = try request(io, url, "GET", 5000);
    defer response.deinit();
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try task.await(io);
    awaited = true;
}

test "HTTP deadlines interrupt stalled headers and bodies" {
    const io = std.testing.io;
    for ([_]bool{ false, true }) |send_headers| {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
        var server = try address.listen(io, .{});
        defer server.deinit(io);
        var task = try io.concurrent(serveStalled, .{ io, &server, send_headers });
        defer task.cancel(io) catch |err| {
            if (err != error.Canceled) std.debug.print("HTTP test server failed: {s}\n", .{@errorName(err)});
        };
        var url_buffer: [100]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/", .{server.socket.address.getPort()});
        try std.testing.expectError(error.Timeout, request(io, url, "GET", 100));
    }
}

test "redirects resolve relative locations and preserve requests method rules" {
    const io = std.testing.io;
    const cases = [_]struct {
        status: u16,
        method: []const u8,
        next: []const u8,
        location: []const u8 = "../final?ok",
        final_path: []const u8 = "/final?ok",
    }{
        .{ .status = 301, .method = "POST", .next = "GET" },
        .{ .status = 301, .method = "PUT", .next = "PUT" },
        .{ .status = 302, .method = "PUT", .next = "GET" },
        .{ .status = 303, .method = "HEAD", .next = "HEAD" },
        .{ .status = 307, .method = "POST", .next = "POST" },
        .{ .status = 308, .method = "DELETE", .next = "DELETE" },
        .{ .status = 302, .method = "GET", .next = "GET", .location = "/a b/%7e", .final_path = "/a%20b/~" },
    };
    for (cases) |case| {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
        var server = try address.listen(io, .{});
        defer server.deinit(io);
        var task = try io.concurrent(serveRedirect, .{ io, &server, case.status, case.method, case.next, case.location, case.final_path });
        var awaited = false;
        defer if (!awaited) {
            task.cancel(io) catch |err| {
                if (err != error.Canceled) std.debug.print("HTTP test server failed: {s}\n", .{@errorName(err)});
            };
        };
        var url_buffer: [100]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/parent/start", .{server.socket.address.getPort()});
        var response = try request(io, url, case.method, 5000);
        defer response.deinit();
        try std.testing.expectEqual(@as(u16, 200), response.status);
        try task.await(io);
        awaited = true;
    }
}

test "redirects with cookies are rejected until cookie handling is implemented" {
    const io = std.testing.io;
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try address.listen(io, .{});
    defer server.deinit(io);
    var task = try io.concurrent(serveRedirectCookie, .{ io, &server });
    var awaited = false;
    defer if (!awaited) {
        task.cancel(io) catch |err| {
            if (err != error.Canceled) std.debug.print("HTTP test server failed: {s}\n", .{@errorName(err)});
        };
    };
    var url_buffer: [100]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/", .{server.socket.address.getPort()});
    try std.testing.expectError(error.UnsupportedHttpRedirectCookie, request(io, url, "GET", 5000));
    try task.await(io);
    awaited = true;
}

test "redirect loops have the same maximum as requests" {
    const io = std.testing.io;
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try address.listen(io, .{});
    defer server.deinit(io);
    var task = try io.concurrent(serveRedirectLoop, .{ io, &server });
    var awaited = false;
    defer if (!awaited) {
        task.cancel(io) catch |err| {
            if (err != error.Canceled) std.debug.print("HTTP test server failed: {s}\n", .{@errorName(err)});
        };
    };
    var url_buffer: [100]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/loop", .{server.socket.address.getPort()});
    try std.testing.expectError(error.TooManyHttpRedirects, request(io, url, "GET", 5000));
    try task.await(io);
    awaited = true;
}

test "redirect URL quoting matches requests and credentials remain unsupported" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const base = try std.Uri.parse("http://example.test/parent");
    const uri = try redirectUri(allocator, base, "/a b/%7e/%2F/caf\xc3\xa9");
    try std.testing.expectEqualStrings("/a%20b/~/%2F/caf%C3%A9", uri.path.percent_encoded);
    try std.testing.expectEqualStrings("/%25GG/%257e", try requoteUri(allocator, "/%GG/%7e"));
    try std.testing.expectError(error.InvalidHttpRedirectLocation, redirectUri(allocator, base, "/\xff"));
    try std.testing.expectError(error.UnsupportedHttpRedirectLocation, redirectUri(allocator, base, "/%\xc3\xa9"));
    try std.testing.expectError(error.UnsupportedHttpRedirectCredentials, redirectUri(arena.allocator(), base, "http://user@example.test/"));
}

test "compare status, case-insensitive combined headers, and Latin-1 text" {
    var response: Response = .{
        .arena = .init(std.testing.allocator),
        .status = 200,
        .headers = &.{
            .{ .name = "Content-Type", .value = "text/plain" },
            .{ .name = "X-Example", .value = "first" },
            .{ .name = "x-example", .value = "second" },
        },
        .body = "caf\xe9",
    };
    defer response.deinit();
    const expected: fixture.Request = .{
        .headers = &.{.{ .name = "X-EXAMPLE", .value = "first, second" }},
        .body = "caf\xc3\xa9",
    };
    try std.testing.expect((try compare(&response, expected)) == null);
    try std.testing.expect((try compare(&response, .{ .status = 201 })).? == .status);
    try std.testing.expect((try compare(&response, .{
        .headers = &.{.{ .name = "missing", .value = "value" }},
    })).? == .missing_header);
    try std.testing.expect((try compare(&response, .{
        .headers = &.{.{ .name = "x-example", .value = "wrong" }},
    })).? == .header);
    try std.testing.expect((try compare(&response, .{ .body = "wrong" })).? == .body);
}

test "comparison rejects unsupported or ambiguous encodings" {
    var response: Response = .{
        .arena = .init(std.testing.allocator),
        .status = 200,
        .headers = &.{.{ .name = "content-type", .value = "text/plain; charset=utf-7" }},
        .body = "a",
    };
    defer response.deinit();
    try std.testing.expectError(error.UnsupportedHttpCharset, compare(&response, .{ .body = "a" }));
    response.headers = &.{};
    response.body = "\xc3\xa9";
    try std.testing.expectError(error.UnsupportedHttpDecoding, compare(&response, .{ .body = "\xc3\xa9" }));
    response.headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=\"UTF-8\"" }};
    try std.testing.expect((try compare(&response, .{ .body = "\xc3\xa9" })) == null);
}

test "declared UTF-8 and ASCII decode invalid bytes with replacement" {
    var response: Response = .{
        .arena = .init(std.testing.allocator),
        .status = 200,
        .headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=utf-8" }},
        .body = "\xe2\x82x \xe0\x80\x80 \xf0\x9f\x92",
    };
    defer response.deinit();
    try std.testing.expect((try compare(&response, .{
        .body = "\xef\xbf\xbdx \xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd \xef\xbf\xbd",
    })) == null);
    response.body = "\xc0\xaf\xf5\x80\x80\x80";
    try std.testing.expect((try compare(&response, .{
        .body = "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd",
    })) == null);
    response.headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=ascii" }};
    response.body = "a\xc3\xa9";
    try std.testing.expect((try compare(&response, .{ .body = "a\xef\xbf\xbd\xef\xbf\xbd" })) == null);
}

test "declared Windows-1252 and UTF-8 signature decoding" {
    var response: Response = .{
        .arena = .init(std.testing.allocator),
        .status = 200,
        .headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=windows_1252" }},
        .body = "\x80\x81\x9f\xe9",
    };
    defer response.deinit();
    try std.testing.expect((try compare(&response, .{ .body = "\xe2\x82\xac\xef\xbf\xbd\xc5\xb8\xc3\xa9" })) == null);
    response.headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=utf_8_sig" }};
    response.body = "\xef\xbb\xbfhi\xff";
    try std.testing.expect((try compare(&response, .{ .body = "hi\xef\xbf\xbd" })) == null);
    response.headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=utf-8" }};
    response.body = "\xef\xbb\xbfhi";
    try std.testing.expect((try compare(&response, .{ .body = "\xef\xbb\xbfhi" })) == null);
}

test "declared ISO-8859-15 and its aliases decode like the reference" {
    var response: Response = .{
        .arena = .init(std.testing.allocator),
        .status = 200,
        .headers = &.{},
        .body = "\xa4\xa6\xa8\xb4\xb8\xbc\xbd\xbe",
    };
    defer response.deinit();
    for ([_][]const u8{ "iso-8859-15", "iso8859-15", "iso_8859_15", "latin9", "l9" }) |charset| {
        const content_type = try std.fmt.allocPrint(response.arena.allocator(), "text/plain; charset={s}", .{charset});
        response.headers = &.{.{ .name = "Content-Type", .value = content_type }};
        try std.testing.expect((try compare(&response, .{ .body = "\xe2\x82\xac\xc5\xa0\xc5\xa1\xc5\xbd\xc5\xbe\xc5\x92\xc5\x93\xc5\xb8" })) == null);
    }
    var bytes: [256]u8 = undefined;
    for (&bytes, 0..) |*byte, index| byte.* = @intCast(index);
    const text = try decodeLatin9(response.arena.allocator(), &bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    try std.testing.expectEqualStrings("9b58b26dbd8fbff2917ab21d989323703946ba491a1eb15cdb2af7ecf9581e97", &std.fmt.bytesToHex(digest, .lower));
}

test "JSON defaults to UTF-8 and the final charset parameter wins" {
    var response: Response = .{
        .arena = .init(std.testing.allocator),
        .status = 200,
        .headers = &.{.{ .name = "Content-Type", .value = "application/json" }},
        .body = "{\"value\":\"caf\xc3\xa9\"}",
    };
    defer response.deinit();
    try std.testing.expect((try compare(&response, .{ .body = "{\"value\":\"caf\xc3\xa9\"}" })) == null);
    response.body = "\xff";
    try std.testing.expect((try compare(&response, .{ .body = "\xef\xbf\xbd" })) == null);
    response.headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=latin-1; charset=utf-8" }};
    response.body = "\xc3\xa9";
    try std.testing.expect((try compare(&response, .{ .body = "\xc3\xa9" })) == null);
    response.headers = &.{.{ .name = "Content-Type", .value = "TEXT/plain" }};
    try std.testing.expectError(error.UnsupportedHttpDecoding, compare(&response, .{ .body = "\xc3\xa9" }));
    response.headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset" }};
    response.body = "";
    try std.testing.expectError(error.InvalidHttpCharset, compare(&response, .{ .body = "" }));
}

test "UTF-16 text handles byte order marks and malformed sequences" {
    var response: Response = .{
        .arena = .init(std.testing.allocator),
        .status = 200,
        .headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=utf-16" }},
        .body = "\xff\xfeA\x00\x3d\xd8\x00\xde\x00\xd8x\x00\xff",
    };
    defer response.deinit();
    try std.testing.expect((try compare(&response, .{ .body = "A\xf0\x9f\x98\x80\xef\xbf\xbdx\xef\xbf\xbd" })) == null);
    response.headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=utf-16be" }};
    response.body = "\xfe\xff\x00A\xdc\x00";
    try std.testing.expect((try compare(&response, .{ .body = "\xef\xbb\xbfA\xef\xbf\xbd" })) == null);
    response.headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=utf-16le" }};
    response.body = "\x00\xd8\x00";
    try std.testing.expect((try compare(&response, .{ .body = "\xef\xbf\xbd" })) == null);
    response.headers = &.{.{ .name = "Content-Type", .value = "text/plain; charset=utf-7" }};
    response.body = "";
    try std.testing.expect((try compare(&response, .{ .body = "" })) == null);
}

test "UTF-32 text handles byte order marks and malformed sequences" {
    var response: Response = .{
        .arena = .init(std.testing.allocator),
        .status = 200,
        .headers = &.{.{ .name = "Content-Type", .value = "application/json; charset=UTF_32" }},
        .body = "\x00\x00\xfe\xff\x00\x00\x00A\x00\x00\xd8\x00\x00\x11\x00\x00\xff",
    };
    defer response.deinit();
    try std.testing.expect((try compare(&response, .{ .body = "A\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd" })) == null);
    response.headers = &.{.{ .name = "Content-Type", .value = "application/json; charset=utf_32_le" }};
    response.body = "\xff\xfe\x00\x00A\x00\x00\x00";
    try std.testing.expect((try compare(&response, .{ .body = "\xef\xbb\xbfA" })) == null);
}
