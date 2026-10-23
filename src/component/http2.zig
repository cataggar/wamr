//! RFC 9113 HTTP/2 Frame Codec and RFC 7541 HPACK Header Compression Codec (#972 E1).
//!
//! Provides connection preface detection, 9-byte frame header encoding/decoding,
//! control and stream frame helpers (`SETTINGS`, `PING`, `WINDOW_UPDATE`,
//! `RST_STREAM`, `GOAWAY`, `HEADERS`, `CONTINUATION`, `DATA`), and an RFC 7541
//! HPACK static/dynamic table + Huffman header block decoder and encoder for
//! `wasi:http/incoming-handler` (`h2c` prior-knowledge and TLS `h2` ALPN).

const std = @import("std");
const Allocator = std.mem.Allocator;

/// 24-byte HTTP/2 client connection preface (RFC 9113 §3.4).
pub const client_preface = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// Request-line prefix of the HTTP/2 client connection preface.
pub const client_preface_prefix = "PRI * HTTP/2.0";

/// Size of an HTTP/2 frame header in octets (RFC 9113 §4.1).
pub const frame_header_len: usize = 9;

/// Default `SETTINGS_MAX_FRAME_SIZE` (RFC 9113 §6.5.2): 16,384 octets.
pub const default_max_frame_size: u24 = 16384;

/// Maximum permitted `SETTINGS_MAX_FRAME_SIZE` (RFC 9113 §6.5.2): 2^24 - 1 octets.
pub const max_allowed_frame_size: u24 = 16_777_215;

/// Default `SETTINGS_INITIAL_WINDOW_SIZE` (RFC 9113 §6.5.2): 65,535 octets.
pub const default_initial_window_size: u32 = 65_535;

/// Maximum legal flow-control window size (RFC 9113 §6.9.1): 2^31 - 1 octets.
pub const max_flow_control_window: i64 = 0x7fff_ffff;

/// Default HPACK dynamic table maximum size (RFC 7541 §4.2 / RFC 9113 §6.5.2).
pub const default_header_table_size: usize = 4096;

/// Returns true when `bytes` is a non-empty prefix of `client_preface` or starts
/// with `client_preface`.
pub fn isPrefacePrefix(bytes: []const u8) bool {
    if (bytes.len == 0) return false;
    if (bytes.len <= client_preface.len) {
        return std.mem.eql(u8, bytes, client_preface[0..bytes.len]);
    }
    return std.mem.startsWith(u8, bytes, client_preface);
}

/// RFC 9113 §6 HTTP/2 frame types.
pub const FrameType = enum(u8) {
    data = 0x0,
    headers = 0x1,
    priority = 0x2,
    rst_stream = 0x3,
    settings = 0x4,
    push_promise = 0x5,
    ping = 0x6,
    goaway = 0x7,
    window_update = 0x8,
    continuation = 0x9,
    _,
};

/// RFC 9113 §6 HTTP/2 frame flags.
pub const Flags = struct {
    pub const END_STREAM: u8 = 0x1;
    pub const ACK: u8 = 0x1;
    pub const END_HEADERS: u8 = 0x4;
    pub const PADDED: u8 = 0x8;
    pub const PRIORITY: u8 = 0x20;
};

pub const END_STREAM: u8 = Flags.END_STREAM;
pub const ACK: u8 = Flags.ACK;
pub const END_HEADERS: u8 = Flags.END_HEADERS;
pub const PADDED: u8 = Flags.PADDED;
pub const PRIORITY: u8 = Flags.PRIORITY;

/// RFC 9113 §7 HTTP/2 error codes.
pub const ErrorCode = enum(u32) {
    no_error = 0x0,
    protocol_error = 0x1,
    internal_error = 0x2,
    flow_control_error = 0x3,
    settings_timeout = 0x4,
    stream_closed = 0x5,
    frame_size_error = 0x6,
    refused_stream = 0x7,
    cancel = 0x8,
    compression_error = 0x9,
    connect_error = 0xa,
    enhance_your_calm = 0xb,
    inadequate_security = 0xc,
    http_1_1_required = 0xd,
    _,
};

/// RFC 9113 §6.5.2 HTTP/2 setting identifiers.
pub const SettingId = enum(u16) {
    header_table_size = 0x1,
    enable_push = 0x2,
    max_concurrent_streams = 0x3,
    initial_window_size = 0x4,
    max_frame_size = 0x5,
    max_header_list_size = 0x6,
    _,
};

pub const Setting = struct {
    id: SettingId,
    value: u32,
};

pub const CodecError = error{
    BufferTooShort,
    ProtocolError,
    FrameSizeError,
    FlowControlError,
    CompressionError,
    OutOfMemory,
};

/// 9-byte HTTP/2 frame header (RFC 9113 §4.1).
pub const FrameHeader = struct {
    length: u24,
    frame_type: FrameType,
    flags: u8,
    stream_id: u31,

    pub fn encode(self: FrameHeader) [frame_header_len]u8 {
        var out: [frame_header_len]u8 = undefined;
        out[0] = @intCast((self.length >> 16) & 0xff);
        out[1] = @intCast((self.length >> 8) & 0xff);
        out[2] = @intCast(self.length & 0xff);
        out[3] = @intFromEnum(self.frame_type);
        out[4] = self.flags;
        std.mem.writeInt(u32, out[5..9], @as(u32, self.stream_id), .big);
        return out;
    }

    pub fn decode(bytes: [frame_header_len]u8) FrameHeader {
        const length: u24 = (@as(u24, bytes[0]) << 16) |
            (@as(u24, bytes[1]) << 8) |
            @as(u24, bytes[2]);
        const raw_stream_id = std.mem.readInt(u32, bytes[5..9], .big);
        return .{
            .length = length,
            .frame_type = @enumFromInt(bytes[3]),
            .flags = bytes[4],
            .stream_id = @intCast(raw_stream_id & 0x7fff_ffff),
        };
    }

    pub fn decodeSlice(bytes: []const u8) CodecError!FrameHeader {
        if (bytes.len < frame_header_len) return error.BufferTooShort;
        return decode(bytes[0..frame_header_len].*);
    }
};

/// Encode a `SETTINGS` frame (RFC 9113 §6.5).
pub fn encodeSettings(allocator: Allocator, settings: []const Setting) ![]u8 {
    if (settings.len * 6 > max_allowed_frame_size) return error.FrameSizeError;
    const payload_len: u24 = @intCast(settings.len * 6);
    var out = try allocator.alloc(u8, frame_header_len + settings.len * 6);
    const hdr = (FrameHeader{
        .length = payload_len,
        .frame_type = .settings,
        .flags = 0,
        .stream_id = 0,
    }).encode();
    @memcpy(out[0..frame_header_len], &hdr);
    for (settings, 0..) |s, i| {
        const off = frame_header_len + i * 6;
        std.mem.writeInt(u16, out[off..][0..2], @intFromEnum(s.id), .big);
        std.mem.writeInt(u32, out[off + 2 ..][0..4], s.value, .big);
    }
    return out;
}

/// Encode an empty `SETTINGS` frame with `ACK` set (RFC 9113 §6.5).
pub fn encodeSettingsAck() [frame_header_len]u8 {
    return (FrameHeader{
        .length = 0,
        .frame_type = .settings,
        .flags = Flags.ACK,
        .stream_id = 0,
    }).encode();
}

/// Decode and validate a `SETTINGS` frame payload (RFC 9113 §6.5).
pub fn decodeSettings(
    allocator: Allocator,
    payload: []const u8,
    flags: u8,
    stream_id: u31,
) CodecError![]Setting {
    if (stream_id != 0) return error.ProtocolError;
    if ((flags & Flags.ACK) != 0) {
        if (payload.len != 0) return error.FrameSizeError;
        return allocator.alloc(Setting, 0);
    }
    if (payload.len % 6 != 0) return error.FrameSizeError;
    const count = payload.len / 6;
    var settings = try allocator.alloc(Setting, count);
    errdefer allocator.free(settings);
    for (0..count) |i| {
        const off = i * 6;
        const id: SettingId = @enumFromInt(std.mem.readInt(u16, payload[off..][0..2], .big));
        const val = std.mem.readInt(u32, payload[off + 2 ..][0..4], .big);
        switch (id) {
            .enable_push => {
                if (val > 1) return error.ProtocolError;
            },
            .initial_window_size => {
                if (val > 0x7fff_ffff) return error.FlowControlError;
            },
            .max_frame_size => {
                if (val < default_max_frame_size or val > max_allowed_frame_size) {
                    return error.ProtocolError;
                }
            },
            else => {},
        }
        settings[i] = .{ .id = id, .value = val };
    }
    return settings;
}

/// Encode a `PING` or `PING ACK` frame (RFC 9113 §6.7).
pub fn encodePing(opaque_data: [8]u8, ack: bool) [frame_header_len + 8]u8 {
    var out: [frame_header_len + 8]u8 = undefined;
    const hdr = (FrameHeader{
        .length = 8,
        .frame_type = .ping,
        .flags = if (ack) Flags.ACK else 0,
        .stream_id = 0,
    }).encode();
    @memcpy(out[0..frame_header_len], &hdr);
    @memcpy(out[frame_header_len..][0..8], &opaque_data);
    return out;
}

/// Decode a `PING` frame payload (RFC 9113 §6.7).
pub fn decodePing(payload: []const u8, stream_id: u31) CodecError![8]u8 {
    if (stream_id != 0) return error.ProtocolError;
    if (payload.len != 8) return error.FrameSizeError;
    return payload[0..8].*;
}

/// Encode a `WINDOW_UPDATE` frame (RFC 9113 §6.9).
pub fn encodeWindowUpdate(stream_id: u31, increment: u31) [frame_header_len + 4]u8 {
    var out: [frame_header_len + 4]u8 = undefined;
    const hdr = (FrameHeader{
        .length = 4,
        .frame_type = .window_update,
        .flags = 0,
        .stream_id = stream_id,
    }).encode();
    @memcpy(out[0..frame_header_len], &hdr);
    std.mem.writeInt(u32, out[frame_header_len..][0..4], @as(u32, increment), .big);
    return out;
}

/// Decode a `WINDOW_UPDATE` frame payload (RFC 9113 §6.9).
pub fn decodeWindowUpdate(payload: []const u8) CodecError!u31 {
    if (payload.len != 4) return error.FrameSizeError;
    const raw = std.mem.readInt(u32, payload[0..4], .big) & 0x7fff_ffff;
    if (raw == 0) return error.ProtocolError;
    return @intCast(raw);
}

/// Encode a `RST_STREAM` frame (RFC 9113 §6.4).
pub fn encodeRstStream(stream_id: u31, error_code: ErrorCode) [frame_header_len + 4]u8 {
    var out: [frame_header_len + 4]u8 = undefined;
    const hdr = (FrameHeader{
        .length = 4,
        .frame_type = .rst_stream,
        .flags = 0,
        .stream_id = stream_id,
    }).encode();
    @memcpy(out[0..frame_header_len], &hdr);
    std.mem.writeInt(u32, out[frame_header_len..][0..4], @intFromEnum(error_code), .big);
    return out;
}

/// Decode a `RST_STREAM` frame payload (RFC 9113 §6.4).
pub fn decodeRstStream(payload: []const u8, stream_id: u31) CodecError!ErrorCode {
    if (stream_id == 0) return error.ProtocolError;
    if (payload.len != 4) return error.FrameSizeError;
    return @enumFromInt(std.mem.readInt(u32, payload[0..4], .big));
}

pub const GoawayFrame = struct {
    last_stream_id: u31,
    error_code: ErrorCode,
    debug_data: []const u8,
};

/// Encode a `GOAWAY` frame (RFC 9113 §6.8).
pub fn encodeGoaway(
    allocator: Allocator,
    last_stream_id: u31,
    error_code: ErrorCode,
    debug_data: []const u8,
) ![]u8 {
    if (8 + debug_data.len > max_allowed_frame_size) return error.FrameSizeError;
    const payload_len: u24 = @intCast(8 + debug_data.len);
    var out = try allocator.alloc(u8, frame_header_len + 8 + debug_data.len);
    const hdr = (FrameHeader{
        .length = payload_len,
        .frame_type = .goaway,
        .flags = 0,
        .stream_id = 0,
    }).encode();
    @memcpy(out[0..frame_header_len], &hdr);
    std.mem.writeInt(u32, out[frame_header_len..][0..4], @as(u32, last_stream_id), .big);
    std.mem.writeInt(u32, out[frame_header_len + 4 ..][0..4], @intFromEnum(error_code), .big);
    if (debug_data.len > 0) {
        @memcpy(out[frame_header_len + 8 ..], debug_data);
    }
    return out;
}

/// Decode a `GOAWAY` frame payload (RFC 9113 §6.8).
pub fn decodeGoaway(payload: []const u8, stream_id: u31) CodecError!GoawayFrame {
    if (stream_id != 0) return error.ProtocolError;
    if (payload.len < 8) return error.FrameSizeError;
    const raw_last = std.mem.readInt(u32, payload[0..4], .big) & 0x7fff_ffff;
    const err_code: ErrorCode = @enumFromInt(std.mem.readInt(u32, payload[4..8], .big));
    return .{
        .last_stream_id = @intCast(raw_last),
        .error_code = err_code,
        .debug_data = payload[8..],
    };
}

/// Encode a `HEADERS` frame (RFC 9113 §6.2).
pub fn encodeHeadersFrame(
    allocator: Allocator,
    stream_id: u31,
    header_block: []const u8,
    end_stream: bool,
    end_headers: bool,
) ![]u8 {
    if (stream_id == 0) return error.ProtocolError;
    if (header_block.len > max_allowed_frame_size) return error.FrameSizeError;
    var flags: u8 = 0;
    if (end_stream) flags |= Flags.END_STREAM;
    if (end_headers) flags |= Flags.END_HEADERS;
    var out = try allocator.alloc(u8, frame_header_len + header_block.len);
    const hdr = (FrameHeader{
        .length = @intCast(header_block.len),
        .frame_type = .headers,
        .flags = flags,
        .stream_id = stream_id,
    }).encode();
    @memcpy(out[0..frame_header_len], &hdr);
    @memcpy(out[frame_header_len..], header_block);
    return out;
}

/// Encode a `CONTINUATION` frame (RFC 9113 §6.10).
pub fn encodeContinuationFrame(
    allocator: Allocator,
    stream_id: u31,
    header_block: []const u8,
    end_headers: bool,
) ![]u8 {
    if (stream_id == 0) return error.ProtocolError;
    if (header_block.len > max_allowed_frame_size) return error.FrameSizeError;
    const flags: u8 = if (end_headers) Flags.END_HEADERS else 0;
    var out = try allocator.alloc(u8, frame_header_len + header_block.len);
    const hdr = (FrameHeader{
        .length = @intCast(header_block.len),
        .frame_type = .continuation,
        .flags = flags,
        .stream_id = stream_id,
    }).encode();
    @memcpy(out[0..frame_header_len], &hdr);
    @memcpy(out[frame_header_len..], header_block);
    return out;
}

/// Extract the header block fragment from a `HEADERS` frame payload, stripping
/// optional `PADDED` and `PRIORITY` fields (RFC 9113 §6.2).
pub fn extractHeadersFragment(payload: []const u8, flags: u8) CodecError![]const u8 {
    var start: usize = 0;
    var pad_len: usize = 0;
    if ((flags & Flags.PADDED) != 0) {
        if (payload.len < 1) return error.ProtocolError;
        pad_len = payload[0];
        start += 1;
    }
    if ((flags & Flags.PRIORITY) != 0) {
        if (payload.len < start + 5) return error.FrameSizeError;
        start += 5;
    }
    if (payload.len < start + pad_len) return error.ProtocolError;
    return payload[start .. payload.len - pad_len];
}

/// Encode a `DATA` frame (RFC 9113 §6.1).
pub fn encodeDataFrame(
    allocator: Allocator,
    stream_id: u31,
    data: []const u8,
    end_stream: bool,
) ![]u8 {
    if (stream_id == 0) return error.ProtocolError;
    if (data.len > max_allowed_frame_size) return error.FrameSizeError;
    var out = try allocator.alloc(u8, frame_header_len + data.len);
    const hdr = (FrameHeader{
        .length = @intCast(data.len),
        .frame_type = .data,
        .flags = if (end_stream) Flags.END_STREAM else 0,
        .stream_id = stream_id,
    }).encode();
    @memcpy(out[0..frame_header_len], &hdr);
    @memcpy(out[frame_header_len..], data);
    return out;
}

/// Extract the application data slice from a `DATA` frame payload, stripping
/// optional `PADDED` padding (RFC 9113 §6.1).
pub fn extractDataPayload(payload: []const u8, flags: u8) CodecError![]const u8 {
    if ((flags & Flags.PADDED) == 0) return payload;
    if (payload.len < 1) return error.ProtocolError;
    const pad_len: usize = payload[0];
    if (1 + pad_len > payload.len) return error.ProtocolError;
    return payload[1 .. payload.len - pad_len];
}

// ============================================================================
// RFC 7541 HPACK Header Compression
// ============================================================================

pub const HeaderField = struct {
    name: []const u8,
    value: []const u8,
};

pub const OwnedHeaderField = struct {
    name: []u8,
    value: []u8,

    pub fn deinit(self: *OwnedHeaderField, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.value);
    }
};

/// RFC 7541 Appendix A Static Table (61 entries, 1-indexed).
pub const static_table = [_]HeaderField{
    .{ .name = ":authority", .value = "" }, // 1
    .{ .name = ":method", .value = "GET" }, // 2
    .{ .name = ":method", .value = "POST" }, // 3
    .{ .name = ":path", .value = "/" }, // 4
    .{ .name = ":path", .value = "/index.html" }, // 5
    .{ .name = ":scheme", .value = "http" }, // 6
    .{ .name = ":scheme", .value = "https" }, // 7
    .{ .name = ":status", .value = "200" }, // 8
    .{ .name = ":status", .value = "204" }, // 9
    .{ .name = ":status", .value = "206" }, // 10
    .{ .name = ":status", .value = "304" }, // 11
    .{ .name = ":status", .value = "400" }, // 12
    .{ .name = ":status", .value = "404" }, // 13
    .{ .name = ":status", .value = "500" }, // 14
    .{ .name = "accept-charset", .value = "" }, // 15
    .{ .name = "accept-encoding", .value = "gzip, deflate" }, // 16
    .{ .name = "accept-language", .value = "" }, // 17
    .{ .name = "accept-ranges", .value = "" }, // 18
    .{ .name = "accept", .value = "" }, // 19
    .{ .name = "access-control-allow-origin", .value = "" }, // 20
    .{ .name = "age", .value = "" }, // 21
    .{ .name = "allow", .value = "" }, // 22
    .{ .name = "authorization", .value = "" }, // 23
    .{ .name = "cache-control", .value = "" }, // 24
    .{ .name = "content-disposition", .value = "" }, // 25
    .{ .name = "content-encoding", .value = "" }, // 26
    .{ .name = "content-language", .value = "" }, // 27
    .{ .name = "content-length", .value = "" }, // 28
    .{ .name = "content-location", .value = "" }, // 29
    .{ .name = "content-range", .value = "" }, // 30
    .{ .name = "content-type", .value = "" }, // 31
    .{ .name = "cookie", .value = "" }, // 32
    .{ .name = "date", .value = "" }, // 33
    .{ .name = "etag", .value = "" }, // 34
    .{ .name = "expect", .value = "" }, // 35
    .{ .name = "expires", .value = "" }, // 36
    .{ .name = "from", .value = "" }, // 37
    .{ .name = "host", .value = "" }, // 38
    .{ .name = "if-match", .value = "" }, // 39
    .{ .name = "if-modified-since", .value = "" }, // 40
    .{ .name = "if-none-match", .value = "" }, // 41
    .{ .name = "if-range", .value = "" }, // 42
    .{ .name = "if-unmodified-since", .value = "" }, // 43
    .{ .name = "last-modified", .value = "" }, // 44
    .{ .name = "link", .value = "" }, // 45
    .{ .name = "location", .value = "" }, // 46
    .{ .name = "max-forwards", .value = "" }, // 47
    .{ .name = "proxy-authenticate", .value = "" }, // 48
    .{ .name = "proxy-authorization", .value = "" }, // 49
    .{ .name = "range", .value = "" }, // 50
    .{ .name = "referer", .value = "" }, // 51
    .{ .name = "refresh", .value = "" }, // 52
    .{ .name = "retry-after", .value = "" }, // 53
    .{ .name = "server", .value = "" }, // 54
    .{ .name = "set-cookie", .value = "" }, // 55
    .{ .name = "strict-transport-security", .value = "" }, // 56
    .{ .name = "transfer-encoding", .value = "" }, // 57
    .{ .name = "user-agent", .value = "" }, // 58
    .{ .name = "vary", .value = "" }, // 59
    .{ .name = "via", .value = "" }, // 60
    .{ .name = "www-authenticate", .value = "" }, // 61
};

/// RFC 7541 §4 Dynamic Table with FIFO eviction and size updates.
pub const DynamicTable = struct {
    allocator: Allocator,
    entries: std.ArrayListUnmanaged(OwnedHeaderField) = .empty,
    current_size: usize = 0,
    max_size: usize = default_header_table_size,
    allowed_max_size: usize = default_header_table_size,

    pub fn init(allocator: Allocator) DynamicTable {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *DynamicTable) void {
        for (self.entries.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.entries.deinit(self.allocator);
        self.current_size = 0;
    }

    pub fn entrySize(name: []const u8, value: []const u8) usize {
        return name.len + value.len + 32;
    }

    pub fn setAllowedMaxSize(self: *DynamicTable, new_allowed: usize) void {
        self.allowed_max_size = new_allowed;
        if (self.max_size > new_allowed) {
            self.updateMaxSize(new_allowed) catch {};
        }
    }

    pub fn updateMaxSize(self: *DynamicTable, new_max: usize) CodecError!void {
        if (new_max > self.allowed_max_size) return error.CompressionError;
        self.max_size = new_max;
        self.evictToSize(self.max_size);
    }

    fn evictToSize(self: *DynamicTable, target_size: usize) void {
        while (self.current_size > target_size and self.entries.items.len > 0) {
            var popped = self.entries.pop().?;
            const sz = entrySize(popped.name, popped.value);
            self.current_size -= sz;
            popped.deinit(self.allocator);
        }
    }

    /// Insert a new `(name, value)` at the front of the dynamic table (index 62).
    pub fn insert(self: *DynamicTable, name: []const u8, value: []const u8) CodecError!void {
        const sz = entrySize(name, value);
        if (sz > self.max_size) {
            self.evictToSize(0);
            return;
        }
        self.evictToSize(self.max_size - sz);
        const dup_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(dup_name);
        const dup_value = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(dup_value);
        try self.entries.insert(self.allocator, 0, .{
            .name = dup_name,
            .value = dup_value,
        });
        self.current_size += sz;
    }

    /// Lookup 1-based combined static + dynamic index (RFC 7541 §2.3.3).
    pub fn get(self: *const DynamicTable, index: usize) CodecError!HeaderField {
        if (index == 0) return error.CompressionError;
        if (index <= static_table.len) {
            return static_table[index - 1];
        }
        const dyn_idx = index - static_table.len - 1;
        if (dyn_idx >= self.entries.items.len) return error.CompressionError;
        const entry = self.entries.items[dyn_idx];
        return .{ .name = entry.name, .value = entry.value };
    }
};

/// Encode an integer with an N-bit prefix (RFC 7541 §5.1).
pub fn encodeInt(
    out: *std.ArrayListUnmanaged(u8),
    allocator: Allocator,
    first_byte_bits: u8,
    prefix_bits: u4,
    value: usize,
) !void {
    std.debug.assert(prefix_bits >= 1 and prefix_bits <= 8);
    const max_prefix: usize = (@as(usize, 1) << prefix_bits) - 1;
    if (value < max_prefix) {
        try out.append(allocator, first_byte_bits | @as(u8, @intCast(value)));
        return;
    }
    try out.append(allocator, first_byte_bits | @as(u8, @intCast(max_prefix)));
    var rem = value - max_prefix;
    while (rem >= 128) {
        try out.append(allocator, @as(u8, @intCast((rem & 0x7f) | 0x80)));
        rem >>= 7;
    }
    try out.append(allocator, @as(u8, @intCast(rem)));
}

/// Decode an integer with an N-bit prefix (RFC 7541 §5.1).
pub fn decodeInt(bytes: []const u8, pos: *usize, prefix_bits: u4) CodecError!usize {
    std.debug.assert(prefix_bits >= 1 and prefix_bits <= 8);
    if (pos.* >= bytes.len) return error.CompressionError;
    const max_prefix: usize = (@as(usize, 1) << prefix_bits) - 1;
    var value: usize = bytes[pos.*] & max_prefix;
    pos.* += 1;
    if (value < max_prefix) return value;

    var shift: u6 = 0;
    while (true) {
        if (pos.* >= bytes.len) return error.CompressionError;
        const b = bytes[pos.*];
        pos.* += 1;
        if (shift >= @bitSizeOf(usize) - 7) return error.CompressionError;
        const addend = (@as(usize, b & 0x7f)) << shift;
        value = std.math.add(usize, value, addend) catch return error.CompressionError;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return value;
}

/// RFC 7541 Appendix B Huffman code entry.
pub const HuffmanCode = struct {
    code: u32,
    bits: u6,
};

/// RFC 7541 Appendix B Huffman Code Table (symbols 0..255 + EOS=256).
pub const huffman_table = [257]HuffmanCode{
    .{ .code = 0x1ff8, .bits = 13 }, // 0
    .{ .code = 0x7fffd8, .bits = 23 }, // 1
    .{ .code = 0xfffffe2, .bits = 28 }, // 2
    .{ .code = 0xfffffe3, .bits = 28 }, // 3
    .{ .code = 0xfffffe4, .bits = 28 }, // 4
    .{ .code = 0xfffffe5, .bits = 28 }, // 5
    .{ .code = 0xfffffe6, .bits = 28 }, // 6
    .{ .code = 0xfffffe7, .bits = 28 }, // 7
    .{ .code = 0xfffffe8, .bits = 28 }, // 8
    .{ .code = 0xffffea, .bits = 24 }, // 9
    .{ .code = 0x3ffffffc, .bits = 30 }, // 10
    .{ .code = 0xfffffe9, .bits = 28 }, // 11
    .{ .code = 0xfffffea, .bits = 28 }, // 12
    .{ .code = 0x3ffffffd, .bits = 30 }, // 13
    .{ .code = 0xfffffeb, .bits = 28 }, // 14
    .{ .code = 0xfffffec, .bits = 28 }, // 15
    .{ .code = 0xfffffed, .bits = 28 }, // 16
    .{ .code = 0xfffffee, .bits = 28 }, // 17
    .{ .code = 0xfffffef, .bits = 28 }, // 18
    .{ .code = 0xffffff0, .bits = 28 }, // 19
    .{ .code = 0xffffff1, .bits = 28 }, // 20
    .{ .code = 0xffffff2, .bits = 28 }, // 21
    .{ .code = 0x3ffffffe, .bits = 30 }, // 22
    .{ .code = 0xffffff3, .bits = 28 }, // 23
    .{ .code = 0xffffff4, .bits = 28 }, // 24
    .{ .code = 0xffffff5, .bits = 28 }, // 25
    .{ .code = 0xffffff6, .bits = 28 }, // 26
    .{ .code = 0xffffff7, .bits = 28 }, // 27
    .{ .code = 0xffffff8, .bits = 28 }, // 28
    .{ .code = 0xffffff9, .bits = 28 }, // 29
    .{ .code = 0xffffffa, .bits = 28 }, // 30
    .{ .code = 0xffffffb, .bits = 28 }, // 31
    .{ .code = 0x14, .bits = 6 }, // 32 ' '
    .{ .code = 0x3f8, .bits = 10 }, // 33 '!'
    .{ .code = 0x3f9, .bits = 10 }, // 34 '"'
    .{ .code = 0xffa, .bits = 12 }, // 35 '#'
    .{ .code = 0x1ff9, .bits = 13 }, // 36 '$'
    .{ .code = 0x15, .bits = 6 }, // 37 '%'
    .{ .code = 0xf8, .bits = 8 }, // 38 '&'
    .{ .code = 0x7fa, .bits = 11 }, // 39 '\''
    .{ .code = 0x3fa, .bits = 10 }, // 40 '('
    .{ .code = 0x3fb, .bits = 10 }, // 41 ')'
    .{ .code = 0xf9, .bits = 8 }, // 42 '*'
    .{ .code = 0x7fb, .bits = 11 }, // 43 '+'
    .{ .code = 0xfa, .bits = 8 }, // 44 ','
    .{ .code = 0x16, .bits = 6 }, // 45 '-'
    .{ .code = 0x17, .bits = 6 }, // 46 '.'
    .{ .code = 0x18, .bits = 6 }, // 47 '/'
    .{ .code = 0x0, .bits = 5 }, // 48 '0'
    .{ .code = 0x1, .bits = 5 }, // 49 '1'
    .{ .code = 0x2, .bits = 5 }, // 50 '2'
    .{ .code = 0x19, .bits = 6 }, // 51 '3'
    .{ .code = 0x1a, .bits = 6 }, // 52 '4'
    .{ .code = 0x1b, .bits = 6 }, // 53 '5'
    .{ .code = 0x1c, .bits = 6 }, // 54 '6'
    .{ .code = 0x1d, .bits = 6 }, // 55 '7'
    .{ .code = 0x1e, .bits = 6 }, // 56 '8'
    .{ .code = 0x1f, .bits = 6 }, // 57 '9'
    .{ .code = 0x5c, .bits = 7 }, // 58 ':'
    .{ .code = 0xfb, .bits = 8 }, // 59 ';'
    .{ .code = 0x7ffc, .bits = 15 }, // 60 '<'
    .{ .code = 0x20, .bits = 6 }, // 61 '='
    .{ .code = 0xffb, .bits = 12 }, // 62 '>'
    .{ .code = 0x3fc, .bits = 10 }, // 63 '?'
    .{ .code = 0x1ffa, .bits = 13 }, // 64 '@'
    .{ .code = 0x21, .bits = 6 }, // 65 'A'
    .{ .code = 0x5d, .bits = 7 }, // 66 'B'
    .{ .code = 0x5e, .bits = 7 }, // 67 'C'
    .{ .code = 0x5f, .bits = 7 }, // 68 'D'
    .{ .code = 0x60, .bits = 7 }, // 69 'E'
    .{ .code = 0x61, .bits = 7 }, // 70 'F'
    .{ .code = 0x62, .bits = 7 }, // 71 'G'
    .{ .code = 0x63, .bits = 7 }, // 72 'H'
    .{ .code = 0x64, .bits = 7 }, // 73 'I'
    .{ .code = 0x65, .bits = 7 }, // 74 'J'
    .{ .code = 0x66, .bits = 7 }, // 75 'K'
    .{ .code = 0x67, .bits = 7 }, // 76 'L'
    .{ .code = 0x68, .bits = 7 }, // 77 'M'
    .{ .code = 0x69, .bits = 7 }, // 78 'N'
    .{ .code = 0x6a, .bits = 7 }, // 79 'O'
    .{ .code = 0x6b, .bits = 7 }, // 80 'P'
    .{ .code = 0x6c, .bits = 7 }, // 81 'Q'
    .{ .code = 0x6d, .bits = 7 }, // 82 'R'
    .{ .code = 0x6e, .bits = 7 }, // 83 'S'
    .{ .code = 0x6f, .bits = 7 }, // 84 'T'
    .{ .code = 0x70, .bits = 7 }, // 85 'U'
    .{ .code = 0x71, .bits = 7 }, // 86 'V'
    .{ .code = 0x72, .bits = 7 }, // 87 'W'
    .{ .code = 0xfc, .bits = 8 }, // 88 'X'
    .{ .code = 0x73, .bits = 7 }, // 89 'Y'
    .{ .code = 0xfd, .bits = 8 }, // 90 'Z'
    .{ .code = 0x1ffb, .bits = 13 }, // 91 '['
    .{ .code = 0x7fff0, .bits = 19 }, // 92 '\\'
    .{ .code = 0x1ffc, .bits = 13 }, // 93 ']'
    .{ .code = 0x3ffc, .bits = 14 }, // 94 '^'
    .{ .code = 0x22, .bits = 6 }, // 95 '_'
    .{ .code = 0x7ffd, .bits = 15 }, // 96 '`'
    .{ .code = 0x3, .bits = 5 }, // 97 'a'
    .{ .code = 0x23, .bits = 6 }, // 98 'b'
    .{ .code = 0x4, .bits = 5 }, // 99 'c'
    .{ .code = 0x24, .bits = 6 }, // 100 'd'
    .{ .code = 0x5, .bits = 5 }, // 101 'e'
    .{ .code = 0x25, .bits = 6 }, // 102 'f'
    .{ .code = 0x26, .bits = 6 }, // 103 'g'
    .{ .code = 0x27, .bits = 6 }, // 104 'h'
    .{ .code = 0x6, .bits = 5 }, // 105 'i'
    .{ .code = 0x74, .bits = 7 }, // 106 'j'
    .{ .code = 0x75, .bits = 7 }, // 107 'k'
    .{ .code = 0x28, .bits = 6 }, // 108 'l'
    .{ .code = 0x29, .bits = 6 }, // 109 'm'
    .{ .code = 0x2a, .bits = 6 }, // 110 'n'
    .{ .code = 0x7, .bits = 5 }, // 111 'o'
    .{ .code = 0x2b, .bits = 6 }, // 112 'p'
    .{ .code = 0x76, .bits = 7 }, // 113 'q'
    .{ .code = 0x2c, .bits = 6 }, // 114 'r'
    .{ .code = 0x8, .bits = 5 }, // 115 's'
    .{ .code = 0x9, .bits = 5 }, // 116 't'
    .{ .code = 0x2d, .bits = 6 }, // 117 'u'
    .{ .code = 0x77, .bits = 7 }, // 118 'v'
    .{ .code = 0x78, .bits = 7 }, // 119 'w'
    .{ .code = 0x79, .bits = 7 }, // 120 'x'
    .{ .code = 0x7a, .bits = 7 }, // 121 'y'
    .{ .code = 0x7b, .bits = 7 }, // 122 'z'
    .{ .code = 0x7ffe, .bits = 15 }, // 123 '{'
    .{ .code = 0x7fc, .bits = 11 }, // 124 '|'
    .{ .code = 0x3ffd, .bits = 14 }, // 125 '}'
    .{ .code = 0x1ffd, .bits = 13 }, // 126 '~'
    .{ .code = 0xffffffc, .bits = 28 }, // 127
    .{ .code = 0xfffe6, .bits = 20 }, // 128
    .{ .code = 0x3fffd2, .bits = 22 }, // 129
    .{ .code = 0xfffe7, .bits = 20 }, // 130
    .{ .code = 0xfffe8, .bits = 20 }, // 131
    .{ .code = 0x3fffd3, .bits = 22 }, // 132
    .{ .code = 0x3fffd4, .bits = 22 }, // 133
    .{ .code = 0x3fffd5, .bits = 22 }, // 134
    .{ .code = 0x7fffd9, .bits = 23 }, // 135
    .{ .code = 0x3fffd6, .bits = 22 }, // 136
    .{ .code = 0x7fffda, .bits = 23 }, // 137
    .{ .code = 0x7fffdb, .bits = 23 }, // 138
    .{ .code = 0x7fffdc, .bits = 23 }, // 139
    .{ .code = 0x7fffdd, .bits = 23 }, // 140
    .{ .code = 0x7fffde, .bits = 23 }, // 141
    .{ .code = 0xffffeb, .bits = 24 }, // 142
    .{ .code = 0x7fffdf, .bits = 23 }, // 143
    .{ .code = 0xffffec, .bits = 24 }, // 144
    .{ .code = 0xffffed, .bits = 24 }, // 145
    .{ .code = 0x3fffd7, .bits = 22 }, // 146
    .{ .code = 0x7fffe0, .bits = 23 }, // 147
    .{ .code = 0xffffee, .bits = 24 }, // 148
    .{ .code = 0x7fffe1, .bits = 23 }, // 149
    .{ .code = 0x7fffe2, .bits = 23 }, // 150
    .{ .code = 0x7fffe3, .bits = 23 }, // 151
    .{ .code = 0x7fffe4, .bits = 23 }, // 152
    .{ .code = 0x1fffdc, .bits = 21 }, // 153
    .{ .code = 0x3fffd8, .bits = 22 }, // 154
    .{ .code = 0x7fffe5, .bits = 23 }, // 155
    .{ .code = 0x3fffd9, .bits = 22 }, // 156
    .{ .code = 0x7fffe6, .bits = 23 }, // 157
    .{ .code = 0x7fffe7, .bits = 23 }, // 158
    .{ .code = 0xffffef, .bits = 24 }, // 159
    .{ .code = 0x3fffda, .bits = 22 }, // 160
    .{ .code = 0x1fffdd, .bits = 21 }, // 161
    .{ .code = 0xfffe9, .bits = 20 }, // 162
    .{ .code = 0x3fffdb, .bits = 22 }, // 163
    .{ .code = 0x3fffdc, .bits = 22 }, // 164
    .{ .code = 0x7fffe8, .bits = 23 }, // 165
    .{ .code = 0x7fffe9, .bits = 23 }, // 166
    .{ .code = 0x1fffde, .bits = 21 }, // 167
    .{ .code = 0x7fffea, .bits = 23 }, // 168
    .{ .code = 0x3fffdd, .bits = 22 }, // 169
    .{ .code = 0x3fffde, .bits = 22 }, // 170
    .{ .code = 0xfffff0, .bits = 24 }, // 171
    .{ .code = 0x1fffdf, .bits = 21 }, // 172
    .{ .code = 0x3fffdf, .bits = 22 }, // 173
    .{ .code = 0x7fffeb, .bits = 23 }, // 174
    .{ .code = 0x7fffec, .bits = 23 }, // 175
    .{ .code = 0x1fffe0, .bits = 21 }, // 176
    .{ .code = 0x1fffe1, .bits = 21 }, // 177
    .{ .code = 0x3fffe0, .bits = 22 }, // 178
    .{ .code = 0x1fffe2, .bits = 21 }, // 179
    .{ .code = 0x7fffed, .bits = 23 }, // 180
    .{ .code = 0x3fffe1, .bits = 22 }, // 181
    .{ .code = 0x7fffee, .bits = 23 }, // 182
    .{ .code = 0x7fffef, .bits = 23 }, // 183
    .{ .code = 0xfffea, .bits = 20 }, // 184
    .{ .code = 0x3fffe2, .bits = 22 }, // 185
    .{ .code = 0x3fffe3, .bits = 22 }, // 186
    .{ .code = 0x3fffe4, .bits = 22 }, // 187
    .{ .code = 0x7ffff0, .bits = 23 }, // 188
    .{ .code = 0x3fffe5, .bits = 22 }, // 189
    .{ .code = 0x3fffe6, .bits = 22 }, // 190
    .{ .code = 0x7ffff1, .bits = 23 }, // 191
    .{ .code = 0x3ffffe0, .bits = 26 }, // 192
    .{ .code = 0x3ffffe1, .bits = 26 }, // 193
    .{ .code = 0xfffeb, .bits = 20 }, // 194
    .{ .code = 0x7fff1, .bits = 19 }, // 195
    .{ .code = 0x3fffe7, .bits = 22 }, // 196
    .{ .code = 0x7ffff2, .bits = 23 }, // 197
    .{ .code = 0x3fffe8, .bits = 22 }, // 198
    .{ .code = 0x1ffffec, .bits = 25 }, // 199
    .{ .code = 0x3ffffe2, .bits = 26 }, // 200
    .{ .code = 0x3ffffe3, .bits = 26 }, // 201
    .{ .code = 0x3ffffe4, .bits = 26 }, // 202
    .{ .code = 0x7ffffde, .bits = 27 }, // 203
    .{ .code = 0x7ffffdf, .bits = 27 }, // 204
    .{ .code = 0x3ffffe5, .bits = 26 }, // 205
    .{ .code = 0xfffff1, .bits = 24 }, // 206
    .{ .code = 0x1ffffed, .bits = 25 }, // 207
    .{ .code = 0x7fff2, .bits = 19 }, // 208
    .{ .code = 0x1fffe3, .bits = 21 }, // 209
    .{ .code = 0x3ffffe6, .bits = 26 }, // 210
    .{ .code = 0x7ffffe0, .bits = 27 }, // 211
    .{ .code = 0x7ffffe1, .bits = 27 }, // 212
    .{ .code = 0x3ffffe7, .bits = 26 }, // 213
    .{ .code = 0x7ffffe2, .bits = 27 }, // 214
    .{ .code = 0xfffff2, .bits = 24 }, // 215
    .{ .code = 0x1fffe4, .bits = 21 }, // 216
    .{ .code = 0x1fffe5, .bits = 21 }, // 217
    .{ .code = 0x3ffffe8, .bits = 26 }, // 218
    .{ .code = 0x3ffffe9, .bits = 26 }, // 219
    .{ .code = 0xffffffd, .bits = 28 }, // 220
    .{ .code = 0x7ffffe3, .bits = 27 }, // 221
    .{ .code = 0x7ffffe4, .bits = 27 }, // 222
    .{ .code = 0x7ffffe5, .bits = 27 }, // 223
    .{ .code = 0xfffec, .bits = 20 }, // 224
    .{ .code = 0xfffff3, .bits = 24 }, // 225
    .{ .code = 0xfffed, .bits = 20 }, // 226
    .{ .code = 0x1fffe6, .bits = 21 }, // 227
    .{ .code = 0x3fffe9, .bits = 22 }, // 228
    .{ .code = 0x1fffe7, .bits = 21 }, // 229
    .{ .code = 0x1fffe8, .bits = 21 }, // 230
    .{ .code = 0x7ffff3, .bits = 23 }, // 231
    .{ .code = 0x3fffea, .bits = 22 }, // 232
    .{ .code = 0x3fffeb, .bits = 22 }, // 233
    .{ .code = 0x1ffffee, .bits = 25 }, // 234
    .{ .code = 0x1ffffef, .bits = 25 }, // 235
    .{ .code = 0xfffff4, .bits = 24 }, // 236
    .{ .code = 0xfffff5, .bits = 24 }, // 237
    .{ .code = 0x3ffffea, .bits = 26 }, // 238
    .{ .code = 0x7ffff4, .bits = 23 }, // 239
    .{ .code = 0x3ffffeb, .bits = 26 }, // 240
    .{ .code = 0x7ffffe6, .bits = 27 }, // 241
    .{ .code = 0x3ffffec, .bits = 26 }, // 242
    .{ .code = 0x3ffffed, .bits = 26 }, // 243
    .{ .code = 0x7ffffe7, .bits = 27 }, // 244
    .{ .code = 0x7ffffe8, .bits = 27 }, // 245
    .{ .code = 0x7ffffe9, .bits = 27 }, // 246
    .{ .code = 0x7ffffea, .bits = 27 }, // 247
    .{ .code = 0x7ffffeb, .bits = 27 }, // 248
    .{ .code = 0xffffffe, .bits = 28 }, // 249
    .{ .code = 0x7ffffec, .bits = 27 }, // 250
    .{ .code = 0x7ffffed, .bits = 27 }, // 251
    .{ .code = 0x7ffffee, .bits = 27 }, // 252
    .{ .code = 0x7ffffef, .bits = 27 }, // 253
    .{ .code = 0x7fffff0, .bits = 27 }, // 254
    .{ .code = 0x3ffffee, .bits = 26 }, // 255
    .{ .code = 0x3fffffff, .bits = 30 }, // 256 EOS
};

/// Encode a byte slice using RFC 7541 Appendix B Huffman codes, padded with
/// 1-bits (the MSB prefix of EOS) to an octet boundary.
pub fn encodeHuffman(allocator: Allocator, input: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    var acc: u64 = 0;
    var acc_bits: u7 = 0;
    for (input) |b| {
        const entry = huffman_table[b];
        acc = (acc << entry.bits) | entry.code;
        acc_bits += entry.bits;
        while (acc_bits >= 8) {
            acc_bits -= 8;
            const shift: u6 = @intCast(acc_bits);
            try out.append(allocator, @intCast((acc >> shift) & 0xff));
        }
    }
    if (acc_bits > 0) {
        const pad_bits: u6 = @intCast(8 - acc_bits);
        const pad_mask: u64 = (@as(u64, 1) << pad_bits) - 1;
        acc = (acc << pad_bits) | pad_mask;
        try out.append(allocator, @intCast(acc & 0xff));
    }
    return out.toOwnedSlice(allocator);
}

/// Decode an RFC 7541 Appendix B Huffman-encoded byte slice, validating EOS
/// padding rules (RFC 7541 §5.2).
pub fn decodeHuffman(allocator: Allocator, encoded: []const u8) CodecError![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    var code: u32 = 0;
    var code_bits: u6 = 0;

    for (encoded) |byte| {
        var bit_idx: u4 = 8;
        while (bit_idx > 0) {
            bit_idx -= 1;
            const bit: u32 = (byte >> @as(u3, @intCast(bit_idx))) & 1;
            if (code_bits >= 30) return error.CompressionError;
            code = (code << 1) | bit;
            code_bits += 1;

            if (code_bits >= 5) {
                for (huffman_table, 0..) |entry, sym| {
                    if (entry.bits == code_bits and entry.code == code) {
                        if (sym == 256) {
                            // Explicit EOS symbol in stream is a compression error (RFC 7541 §5.2).
                            return error.CompressionError;
                        }
                        try out.append(allocator, @intCast(sym));
                        code = 0;
                        code_bits = 0;
                        break;
                    }
                }
            }
        }
    }

    if (code_bits > 7) return error.CompressionError;
    if (code_bits > 0) {
        const expected_pad: u32 = (@as(u32, 1) << @as(u5, @intCast(code_bits))) - 1;
        if (code != expected_pad) return error.CompressionError;
    }

    return out.toOwnedSlice(allocator);
}

/// Encode an HPACK string literal (RFC 7541 §5.2).
pub fn encodeString(
    out: *std.ArrayListUnmanaged(u8),
    allocator: Allocator,
    str: []const u8,
    use_huffman: bool,
) !void {
    if (use_huffman) {
        const huff = try encodeHuffman(allocator, str);
        defer allocator.free(huff);
        try encodeInt(out, allocator, 0x80, 7, huff.len);
        try out.appendSlice(allocator, huff);
    } else {
        try encodeInt(out, allocator, 0x00, 7, str.len);
        try out.appendSlice(allocator, str);
    }
}

/// Decode an HPACK string literal (RFC 7541 §5.2), returning an owned slice.
pub fn decodeString(allocator: Allocator, bytes: []const u8, pos: *usize) CodecError![]u8 {
    if (pos.* >= bytes.len) return error.CompressionError;
    const is_huffman = (bytes[pos.*] & 0x80) != 0;
    const str_len = try decodeInt(bytes, pos, 7);
    if (pos.* + str_len > bytes.len) return error.CompressionError;
    const raw = bytes[pos.* .. pos.* + str_len];
    pos.* += str_len;
    if (is_huffman) {
        return decodeHuffman(allocator, raw);
    }
    return allocator.dupe(u8, raw);
}

pub const DecodedRequestHeaders = struct {
    method: []u8,
    scheme: ?[]u8 = null,
    authority: ?[]u8 = null,
    path: []u8,
    headers: std.ArrayListUnmanaged(OwnedHeaderField) = .empty,

    pub fn deinit(self: *DecodedRequestHeaders, allocator: Allocator) void {
        allocator.free(self.method);
        if (self.scheme) |s| allocator.free(s);
        if (self.authority) |a| allocator.free(a);
        allocator.free(self.path);
        for (self.headers.items) |*h| h.deinit(allocator);
        self.headers.deinit(allocator);
    }
};

fn validateHeaderNameLowercase(name: []const u8) CodecError!void {
    if (name.len == 0) return error.ProtocolError;
    for (name) |c| {
        if (c >= 'A' and c <= 'Z') return error.ProtocolError;
        if (c <= 0x20 or c == 0x7f) return error.ProtocolError;
    }
}

/// Stateful RFC 7541 HPACK Decoder.
pub const Decoder = struct {
    allocator: Allocator,
    dynamic_table: DynamicTable,

    pub fn init(allocator: Allocator) Decoder {
        return .{
            .allocator = allocator,
            .dynamic_table = DynamicTable.init(allocator),
        };
    }

    pub fn deinit(self: *Decoder) void {
        self.dynamic_table.deinit();
    }

    pub fn setAllowedMaxTableSize(self: *Decoder, max_size: usize) void {
        self.dynamic_table.setAllowedMaxSize(max_size);
    }

    /// Decode an HPACK header block into an ordered list of `(name, value)` fields.
    pub fn decodeHeaderList(
        self: *Decoder,
        block: []const u8,
    ) CodecError!std.ArrayListUnmanaged(OwnedHeaderField) {
        var list: std.ArrayListUnmanaged(OwnedHeaderField) = .empty;
        errdefer {
            for (list.items) |*item| item.deinit(self.allocator);
            list.deinit(self.allocator);
        }

        var pos: usize = 0;
        var seen_header = false;
        while (pos < block.len) {
            const first = block[pos];
            if ((first & 0x80) != 0) {
                // §6.1 Indexed Header Field Representation (7-bit prefix).
                seen_header = true;
                const idx = try decodeInt(block, &pos, 7);
                const entry = try self.dynamic_table.get(idx);
                const name = try self.allocator.dupe(u8, entry.name);
                errdefer self.allocator.free(name);
                const val = try self.allocator.dupe(u8, entry.value);
                errdefer self.allocator.free(val);
                try list.append(self.allocator, .{ .name = name, .value = val });
            } else if ((first & 0xc0) == 0x40) {
                // §6.2.1 Literal Header Field with Incremental Indexing (6-bit prefix).
                seen_header = true;
                const name_idx = try decodeInt(block, &pos, 6);
                const name = if (name_idx == 0)
                    try decodeString(self.allocator, block, &pos)
                else blk: {
                    const entry = try self.dynamic_table.get(name_idx);
                    break :blk try self.allocator.dupe(u8, entry.name);
                };
                errdefer self.allocator.free(name);
                const val = try decodeString(self.allocator, block, &pos);
                errdefer self.allocator.free(val);
                try self.dynamic_table.insert(name, val);
                try list.append(self.allocator, .{ .name = name, .value = val });
            } else if ((first & 0xe0) == 0x20) {
                // §6.3 Dynamic Table Size Update (5-bit prefix).
                if (seen_header) return error.CompressionError;
                const new_max = try decodeInt(block, &pos, 5);
                try self.dynamic_table.updateMaxSize(new_max);
            } else {
                // §6.2.2 Literal without Indexing (`0000xxxx`) or §6.2.3 Never Indexed (`0001xxxx`).
                seen_header = true;
                const name_idx = try decodeInt(block, &pos, 4);
                const name = if (name_idx == 0)
                    try decodeString(self.allocator, block, &pos)
                else blk: {
                    const entry = try self.dynamic_table.get(name_idx);
                    break :blk try self.allocator.dupe(u8, entry.name);
                };
                errdefer self.allocator.free(name);
                const val = try decodeString(self.allocator, block, &pos);
                errdefer self.allocator.free(val);
                try list.append(self.allocator, .{ .name = name, .value = val });
            }
        }
        return list;
    }

    /// Decode an HTTP/2 request header block into `:method`, `:scheme`,
    /// `:authority`, `:path`, and regular headers (RFC 9113 §8.3.1).
    pub fn decodeRequestHeaders(self: *Decoder, block: []const u8) CodecError!DecodedRequestHeaders {
        var raw_list = try self.decodeHeaderList(block);
        defer {
            for (raw_list.items) |*item| item.deinit(self.allocator);
            raw_list.deinit(self.allocator);
        }

        var method: ?[]u8 = null;
        errdefer if (method) |m| self.allocator.free(m);
        var scheme: ?[]u8 = null;
        errdefer if (scheme) |s| self.allocator.free(s);
        var authority: ?[]u8 = null;
        errdefer if (authority) |a| self.allocator.free(a);
        var path: ?[]u8 = null;
        errdefer if (path) |p| self.allocator.free(p);
        var headers: std.ArrayListUnmanaged(OwnedHeaderField) = .empty;
        errdefer {
            for (headers.items) |*h| h.deinit(self.allocator);
            headers.deinit(self.allocator);
        }

        var regular_started = false;
        for (raw_list.items) |item| {
            if (std.mem.startsWith(u8, item.name, ":")) {
                if (regular_started) return error.ProtocolError;
                try validateHeaderNameLowercase(item.name[1..]);
                if (std.mem.eql(u8, item.name, ":method")) {
                    if (method != null or item.value.len == 0) return error.ProtocolError;
                    method = try self.allocator.dupe(u8, item.value);
                } else if (std.mem.eql(u8, item.name, ":scheme")) {
                    if (scheme != null or item.value.len == 0) return error.ProtocolError;
                    scheme = try self.allocator.dupe(u8, item.value);
                } else if (std.mem.eql(u8, item.name, ":authority")) {
                    if (authority != null) return error.ProtocolError;
                    authority = try self.allocator.dupe(u8, item.value);
                } else if (std.mem.eql(u8, item.name, ":path")) {
                    if (path != null or item.value.len == 0) return error.ProtocolError;
                    path = try self.allocator.dupe(u8, item.value);
                } else {
                    return error.ProtocolError;
                }
            } else {
                regular_started = true;
                try validateHeaderNameLowercase(item.name);
                const dup_name = try self.allocator.dupe(u8, item.name);
                errdefer self.allocator.free(dup_name);
                const dup_val = try self.allocator.dupe(u8, item.value);
                errdefer self.allocator.free(dup_val);
                try headers.append(self.allocator, .{ .name = dup_name, .value = dup_val });
            }
        }

        if (method == null or path == null) return error.ProtocolError;
        return .{
            .method = method.?,
            .scheme = scheme,
            .authority = authority,
            .path = path.?,
            .headers = headers,
        };
    }

    /// Decode an HTTP/2 trailer header block (no pseudo-headers permitted per RFC 9113 §8.1).
    pub fn decodeTrailers(
        self: *Decoder,
        block: []const u8,
    ) CodecError!std.ArrayListUnmanaged(OwnedHeaderField) {
        var raw_list = try self.decodeHeaderList(block);
        errdefer {
            for (raw_list.items) |*item| item.deinit(self.allocator);
            raw_list.deinit(self.allocator);
        }
        for (raw_list.items) |item| {
            if (std.mem.startsWith(u8, item.name, ":")) return error.ProtocolError;
            try validateHeaderNameLowercase(item.name);
        }
        return raw_list;
    }
};

/// Stateful RFC 7541 HPACK Encoder.
pub const Encoder = struct {
    allocator: Allocator,
    dynamic_table: DynamicTable,
    use_huffman: bool = false,

    pub fn init(allocator: Allocator) Encoder {
        return .{
            .allocator = allocator,
            .dynamic_table = DynamicTable.init(allocator),
        };
    }

    pub fn deinit(self: *Encoder) void {
        self.dynamic_table.deinit();
    }

    fn findExactMatch(self: *const Encoder, name: []const u8, value: []const u8) ?usize {
        for (static_table, 1..) |entry, idx| {
            if (std.mem.eql(u8, entry.name, name) and std.mem.eql(u8, entry.value, value)) {
                return idx;
            }
        }
        for (self.dynamic_table.entries.items, 0..) |entry, dyn_i| {
            if (std.mem.eql(u8, entry.name, name) and std.mem.eql(u8, entry.value, value)) {
                return static_table.len + 1 + dyn_i;
            }
        }
        return null;
    }

    fn findNameMatch(self: *const Encoder, name: []const u8) ?usize {
        for (static_table, 1..) |entry, idx| {
            if (std.mem.eql(u8, entry.name, name)) {
                return idx;
            }
        }
        for (self.dynamic_table.entries.items, 0..) |entry, dyn_i| {
            if (std.mem.eql(u8, entry.name, name)) {
                return static_table.len + 1 + dyn_i;
            }
        }
        return null;
    }

    /// Encode a single header field into `out` with incremental indexing.
    pub fn encodeField(
        self: *Encoder,
        out: *std.ArrayListUnmanaged(u8),
        raw_name: []const u8,
        value: []const u8,
    ) !void {
        var lower_buf: [256]u8 = undefined;
        var heap_name: ?[]u8 = null;
        defer if (heap_name) |hn| self.allocator.free(hn);

        const name: []const u8 = if (raw_name.len <= lower_buf.len) blk: {
            for (raw_name, 0..) |c, i| {
                lower_buf[i] = std.ascii.toLower(c);
            }
            break :blk lower_buf[0..raw_name.len];
        } else blk: {
            const dup = try self.allocator.dupe(u8, raw_name);
            for (dup) |*c| c.* = std.ascii.toLower(c.*);
            heap_name = dup;
            break :blk dup;
        };

        if (self.findExactMatch(name, value)) |idx| {
            try encodeInt(out, self.allocator, 0x80, 7, idx);
            return;
        }
        if (self.findNameMatch(name)) |name_idx| {
            try encodeInt(out, self.allocator, 0x40, 6, name_idx);
            try encodeString(out, self.allocator, value, self.use_huffman);
        } else {
            try out.append(self.allocator, 0x40);
            try encodeString(out, self.allocator, name, self.use_huffman);
            try encodeString(out, self.allocator, value, self.use_huffman);
        }
        try self.dynamic_table.insert(name, value);
    }

    /// Encode an HTTP/2 request header block (`:method`, `:scheme`, `:authority`, `:path` + regular headers).
    pub fn encodeRequestHeaders(
        self: *Encoder,
        method: []const u8,
        scheme: []const u8,
        authority: []const u8,
        path: []const u8,
        headers: []const HeaderField,
    ) ![]u8 {
        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(self.allocator);
        try self.encodeField(&out, ":method", method);
        try self.encodeField(&out, ":scheme", scheme);
        if (authority.len > 0) {
            try self.encodeField(&out, ":authority", authority);
        }
        try self.encodeField(&out, ":path", path);
        for (headers) |h| {
            try self.encodeField(&out, h.name, h.value);
        }
        return out.toOwnedSlice(self.allocator);
    }

    /// Encode an HTTP/2 response header block (`:status` pseudo-header + regular headers).
    pub fn encodeResponseHeaders(
        self: *Encoder,
        status: u16,
        headers: []const HeaderField,
    ) ![]u8 {
        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(self.allocator);

        var status_buf: [16]u8 = undefined;
        const status_str = try std.fmt.bufPrint(&status_buf, "{d}", .{status});
        try self.encodeField(&out, ":status", status_str);

        for (headers) |h| {
            if (std.ascii.eqlIgnoreCase(h.name, "connection") or
                std.ascii.eqlIgnoreCase(h.name, "transfer-encoding") or
                std.ascii.eqlIgnoreCase(h.name, "keep-alive") or
                std.ascii.eqlIgnoreCase(h.name, "upgrade") or
                std.ascii.eqlIgnoreCase(h.name, "proxy-connection"))
            {
                continue;
            }
            try self.encodeField(&out, h.name, h.value);
        }
        return out.toOwnedSlice(self.allocator);
    }

    /// Encode an HTTP/2 trailer header block (regular headers only).
    pub fn encodeTrailerHeaders(
        self: *Encoder,
        trailers: []const HeaderField,
    ) ![]u8 {
        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(self.allocator);
        for (trailers) |t| {
            try self.encodeField(&out, t.name, t.value);
        }
        return out.toOwnedSlice(self.allocator);
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "http2: frame header encode and decode masks reserved bit (#972)" {
    const testing = std.testing;
    const hdr = FrameHeader{
        .length = 0x010203,
        .frame_type = .headers,
        .flags = Flags.END_STREAM | Flags.END_HEADERS,
        .stream_id = 17,
    };
    var encoded = hdr.encode();
    // Set reserved bit 31 in the wire representation.
    encoded[5] |= 0x80;
    const decoded = try FrameHeader.decodeSlice(&encoded);
    try testing.expectEqual(@as(u24, 0x010203), decoded.length);
    try testing.expectEqual(FrameType.headers, decoded.frame_type);
    try testing.expectEqual(Flags.END_STREAM | Flags.END_HEADERS, decoded.flags);
    try testing.expectEqual(@as(u31, 17), decoded.stream_id);
}

test "http2: SETTINGS, PING, WINDOW_UPDATE, RST_STREAM, GOAWAY, HEADERS, DATA codecs (#972)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // SETTINGS & SETTINGS ACK
    const settings_in = [_]Setting{
        .{ .id = .max_concurrent_streams, .value = 100 },
        .{ .id = .initial_window_size, .value = 65535 },
    };
    const settings_frame = try encodeSettings(allocator, &settings_in);
    defer allocator.free(settings_frame);
    const settings_hdr = try FrameHeader.decodeSlice(settings_frame);
    try testing.expectEqual(FrameType.settings, settings_hdr.frame_type);
    try testing.expectEqual(@as(u24, 12), settings_hdr.length);
    const settings_out = try decodeSettings(
        allocator,
        settings_frame[frame_header_len..],
        settings_hdr.flags,
        settings_hdr.stream_id,
    );
    defer allocator.free(settings_out);
    try testing.expectEqual(@as(usize, 2), settings_out.len);
    try testing.expectEqual(SettingId.max_concurrent_streams, settings_out[0].id);
    try testing.expectEqual(@as(u32, 100), settings_out[0].value);
    try testing.expectEqual(SettingId.initial_window_size, settings_out[1].id);
    try testing.expectEqual(@as(u32, 65535), settings_out[1].value);

    const ack_frame = encodeSettingsAck();
    const ack_hdr = FrameHeader.decode(ack_frame);
    try testing.expectEqual(FrameType.settings, ack_hdr.frame_type);
    try testing.expectEqual(Flags.ACK, ack_hdr.flags);
    try testing.expectEqual(@as(u24, 0), ack_hdr.length);

    // PING & PING ACK
    const ping_payload = [8]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const ping_frame = encodePing(ping_payload, true);
    const ping_hdr = try FrameHeader.decodeSlice(&ping_frame);
    try testing.expectEqual(FrameType.ping, ping_hdr.frame_type);
    try testing.expectEqual(Flags.ACK, ping_hdr.flags);
    const ping_decoded = try decodePing(ping_frame[frame_header_len..], ping_hdr.stream_id);
    try testing.expectEqualSlices(u8, &ping_payload, &ping_decoded);

    // WINDOW_UPDATE
    const wu_frame = encodeWindowUpdate(3, 4096);
    const wu_hdr = try FrameHeader.decodeSlice(&wu_frame);
    try testing.expectEqual(FrameType.window_update, wu_hdr.frame_type);
    try testing.expectEqual(@as(u31, 3), wu_hdr.stream_id);
    const wu_inc = try decodeWindowUpdate(wu_frame[frame_header_len..]);
    try testing.expectEqual(@as(u31, 4096), wu_inc);

    // RST_STREAM
    const rst_frame = encodeRstStream(5, .cancel);
    const rst_hdr = try FrameHeader.decodeSlice(&rst_frame);
    try testing.expectEqual(FrameType.rst_stream, rst_hdr.frame_type);
    try testing.expectEqual(@as(u31, 5), rst_hdr.stream_id);
    const rst_code = try decodeRstStream(rst_frame[frame_header_len..], rst_hdr.stream_id);
    try testing.expectEqual(ErrorCode.cancel, rst_code);

    // GOAWAY
    const goaway_frame = try encodeGoaway(allocator, 7, .no_error, "graceful");
    defer allocator.free(goaway_frame);
    const goaway_hdr = try FrameHeader.decodeSlice(goaway_frame);
    try testing.expectEqual(FrameType.goaway, goaway_hdr.frame_type);
    const goaway_decoded = try decodeGoaway(goaway_frame[frame_header_len..], goaway_hdr.stream_id);
    try testing.expectEqual(@as(u31, 7), goaway_decoded.last_stream_id);
    try testing.expectEqual(ErrorCode.no_error, goaway_decoded.error_code);
    try testing.expectEqualStrings("graceful", goaway_decoded.debug_data);

    // HEADERS with PADDED + PRIORITY stripping
    const padded_prio_payload = [_]u8{
        2, // pad length
        0x80, 0x00, 0x00, 0x01, // exclusive stream dep
        16, // weight
        0x82, 0x84, // HPACK fragment (:method GET, :path /)
        0x00, 0x00, // 2 pad bytes
    };
    const frag = try extractHeadersFragment(&padded_prio_payload, Flags.PADDED | Flags.PRIORITY);
    try testing.expectEqualSlices(u8, &[_]u8{ 0x82, 0x84 }, frag);

    // DATA with PADDED stripping
    const padded_data = [_]u8{ 3, 'h', 'i', 0, 0, 0 };
    const data_slice = try extractDataPayload(&padded_data, Flags.PADDED);
    try testing.expectEqualStrings("hi", data_slice);
}

test "http2: HPACK Huffman encode/decode and RFC 7541 Appendix C vectors (#972)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // RFC 7541 C.4.1 Huffman vector for "www.example.com"
    const rfc_www = [_]u8{ 0xf1, 0xe3, 0xc2, 0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab, 0x90, 0xf4, 0xff };
    const decoded_www = try decodeHuffman(allocator, &rfc_www);
    defer allocator.free(decoded_www);
    try testing.expectEqualStrings("www.example.com", decoded_www);

    const reencoded_www = try encodeHuffman(allocator, "www.example.com");
    defer allocator.free(reencoded_www);
    try testing.expectEqualSlices(u8, &rfc_www, reencoded_www);

    // RFC 7541 C.4.2 Huffman vector for "no-cache"
    const rfc_nocache = [_]u8{ 0xa8, 0xeb, 0x10, 0x64, 0x9c, 0xbf };
    const decoded_nocache = try decodeHuffman(allocator, &rfc_nocache);
    defer allocator.free(decoded_nocache);
    try testing.expectEqualStrings("no-cache", decoded_nocache);

    // Round-trip all 256 byte values.
    var all_bytes: [256]u8 = undefined;
    for (&all_bytes, 0..) |*b, i| b.* = @intCast(i);
    const enc_all = try encodeHuffman(allocator, &all_bytes);
    defer allocator.free(enc_all);
    const dec_all = try decodeHuffman(allocator, enc_all);
    defer allocator.free(dec_all);
    try testing.expectEqualSlices(u8, &all_bytes, dec_all);
}

test "http2: HPACK static/dynamic table indexing, eviction, request decode, and response encode (#972)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var decoder = Decoder.init(allocator);
    defer decoder.deinit();

    // RFC 7541 C.4.1 Request 1 with Huffman:
    // :method: GET (0x82), :scheme: http (0x86), :path: / (0x84),
    // :authority: www.example.com (0x41, 0x8c, f1 e3 c2 e5 f2 3a 6b a0 ab 90 f4 ff)
    const req1_block = [_]u8{
        0x82, 0x86, 0x84, 0x41, 0x8c,
        0xf1, 0xe3, 0xc2, 0xe5, 0xf2,
        0x3a, 0x6b, 0xa0, 0xab, 0x90,
        0xf4, 0xff,
    };
    var req1 = try decoder.decodeRequestHeaders(&req1_block);
    defer req1.deinit(allocator);
    try testing.expectEqualStrings("GET", req1.method);
    try testing.expectEqualStrings("http", req1.scheme.?);
    try testing.expectEqualStrings("/", req1.path);
    try testing.expectEqualStrings("www.example.com", req1.authority.?);
    try testing.expectEqual(@as(usize, 1), decoder.dynamic_table.entries.items.len);

    // RFC 7541 C.4.2 Request 2 referencing dynamic index 62 (0xbe) + cache-control: no-cache
    const req2_block = [_]u8{
        0x82, 0x86, 0x84, 0xbe, 0x58,
        0x86, 0xa8, 0xeb, 0x10, 0x64,
        0x9c, 0xbf,
    };
    var req2 = try decoder.decodeRequestHeaders(&req2_block);
    defer req2.deinit(allocator);
    try testing.expectEqualStrings("www.example.com", req2.authority.?);
    try testing.expectEqual(@as(usize, 1), req2.headers.items.len);
    try testing.expectEqualStrings("cache-control", req2.headers.items[0].name);
    try testing.expectEqualStrings("no-cache", req2.headers.items[0].value);
    try testing.expectEqual(@as(usize, 2), decoder.dynamic_table.entries.items.len);

    // Dynamic table size update to 60 bytes evicts the older entry (size 57) leaving 1 entry (size 53).
    const size_update_and_req = [_]u8{
        0x3f, 0x1d, // Dynamic Table Size Update to 60 (31 + 29)
        0x82, 0x86, 0x84,
    };
    var req3 = try decoder.decodeRequestHeaders(&size_update_and_req);
    defer req3.deinit(allocator);
    try testing.expectEqual(@as(usize, 1), decoder.dynamic_table.entries.items.len);
    try testing.expectEqualStrings("cache-control", decoder.dynamic_table.entries.items[0].name);

    // Encode response headers and decode with a fresh decoder.
    var encoder = Encoder.init(allocator);
    defer encoder.deinit();
    const resp_headers = [_]HeaderField{
        .{ .name = "Content-Type", .value = "application/json" },
        .{ .name = "X-Stream-Id", .value = "1" },
    };
    const resp_block = try encoder.encodeResponseHeaders(200, &resp_headers);
    defer allocator.free(resp_block);

    var resp_decoder = Decoder.init(allocator);
    defer resp_decoder.deinit();
    var decoded_resp = try resp_decoder.decodeHeaderList(resp_block);
    defer {
        for (decoded_resp.items) |*item| item.deinit(allocator);
        decoded_resp.deinit(allocator);
    }
    try testing.expectEqual(@as(usize, 3), decoded_resp.items.len);
    try testing.expectEqualStrings(":status", decoded_resp.items[0].name);
    try testing.expectEqualStrings("200", decoded_resp.items[0].value);
    try testing.expectEqualStrings("content-type", decoded_resp.items[1].name);
    try testing.expectEqualStrings("application/json", decoded_resp.items[1].value);
    try testing.expectEqualStrings("x-stream-id", decoded_resp.items[2].name);
    try testing.expectEqualStrings("1", decoded_resp.items[2].value);
}
