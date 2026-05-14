//! Polyfill: virtualize wasi:io@0.2.x `input-stream` / `output-stream`
//! resources by wrapping a 0.3 `stream<u8>`.
//!
//! Per the wasi.dev roadmap, "implementations may continue to support
//! 0.2 by virtualizing 0.2 in terms of 0.3". When a component imports
//! BOTH `wasi:io/streams@0.2.x` AND a P3 interface that yields a
//! `stream<u8>`, the polyfill activates so a 0.2 `input-stream` handle
//! is virtualized over a 0.3 stream. If only 0.2 is imported, the
//! existing direct 0.2 implementation is unchanged.
//!
//! The 0.3 `stream<u8>` host-side rendezvous machinery (canon
//! `stream.read` / `stream.write` from #478/#505) provides the
//! underlying transport — these virtual wrappers just translate
//! 0.2-style synchronous-result method calls into reads/writes against
//! a `ComponentInstance.streams` entry.

const std = @import("std");
const async_mod = @import("../../component/async.zig");

/// Outcome of a `read` against an `input-stream` (0.2 surface):
///
///     result<list<u8>, stream-error>
///
/// The `closed` variant maps to the 0.2 `stream-error.closed` case;
/// `last-operation-failed` is reserved for a future error-context
/// hook (the host has no failure path yet because the underlying
/// `AsyncStream` only signals open/closed).
pub const ReadOutcome = union(enum) {
    ok: []u8,
    closed,
};

/// Outcome of a `write` against an `output-stream` (0.2 surface):
///
///     result<_, stream-error>
pub const WriteOutcome = union(enum) {
    /// `ok` carries the number of bytes accepted into the 0.3
    /// stream's FIFO. The 0.2 spec lets `write` accept fewer bytes
    /// than offered when `check-write`'s permit was smaller, but the
    /// polyfill's FIFO is unbounded so every byte is always taken.
    ok: usize,
    closed,
};

/// 0.2 `input-stream` resource virtualized over a 0.3 `stream<u8>`.
///
/// `stream_handle` indexes `ComponentInstance.streams`. The virtual
/// stream does NOT own the underlying entry: closing the 0.2 handle
/// only marks the read end as closed via `markReadClosed`, mirroring
/// the canon-ABI `stream.drop-readable` semantics from #505.
pub const VirtualInputStream = struct {
    stream_handle: u32,

    /// Synchronous read of up to `dst.len` bytes from the wrapped
    /// 0.3 stream's FIFO. Returns the slice actually filled, or
    /// `closed` if the writable end has been dropped and the FIFO
    /// is empty (matching 0.2 `stream-error.closed`).
    pub fn read(self: VirtualInputStream, s: *async_mod.AsyncStream, dst: []u8) ReadOutcome {
        _ = self;
        if (s.buffer.items.len == 0) {
            if (s.write_closed) return .closed;
            // No bytes available yet but writer end is still live.
            // Surface as a zero-length read; the 0.2 spec allows
            // `read` to return `0` bytes for "would-block" since
            // `read` is the non-blocking variant.
            return .{ .ok = dst[0..0] };
        }
        const n = @min(dst.len, s.buffer.items.len);
        @memcpy(dst[0..n], s.buffer.items[0..n]);
        // Shift the FIFO. AsyncStream's buffer is a plain ArrayList
        // FIFO; we drain from the front via orderedRemove-equivalent
        // copy (no allocator needed because the spare capacity is
        // preserved).
        std.mem.copyForwards(u8, s.buffer.items[0 .. s.buffer.items.len - n], s.buffer.items[n..]);
        s.buffer.shrinkRetainingCapacity(s.buffer.items.len - n);
        return .{ .ok = dst[0..n] };
    }

    /// Mark the read end as closed (called from 0.2
    /// `[resource-drop]input-stream` when the virtual handle is
    /// destroyed).
    pub fn markReadClosed(self: VirtualInputStream, s: *async_mod.AsyncStream) void {
        _ = self;
        s.read_closed = true;
        if (s.write_closed) s.state = .closed;
    }
};

/// 0.2 `output-stream` resource virtualized over a 0.3 `stream<u8>`.
///
/// Same ownership rules as `VirtualInputStream`.
pub const VirtualOutputStream = struct {
    stream_handle: u32,

    /// Append `src` to the wrapped 0.3 stream's FIFO. Returns the
    /// number of bytes accepted (the polyfill's FIFO is unbounded so
    /// every byte is taken unless the stream is closed). Allocates
    /// into `s.buffer` using `allocator`.
    pub fn write(
        self: VirtualOutputStream,
        s: *async_mod.AsyncStream,
        allocator: std.mem.Allocator,
        src: []const u8,
    ) !WriteOutcome {
        _ = self;
        if (s.read_closed or s.state == .closed) return .closed;
        try s.buffer.appendSlice(allocator, src);
        return .{ .ok = src.len };
    }

    /// Mark the write end as closed (0.2 drop semantics).
    pub fn markWriteClosed(self: VirtualOutputStream, s: *async_mod.AsyncStream) void {
        _ = self;
        s.write_closed = true;
        if (s.read_closed) s.state = .closed;
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "VirtualInputStream.read: drains FIFO from a 0.3 stream" {
    const testing = std.testing;
    var s = async_mod.AsyncStream{};
    defer s.deinit(testing.allocator);

    try s.buffer.appendSlice(testing.allocator, "hello, world!");

    const vis = VirtualInputStream{ .stream_handle = 42 };
    var buf: [5]u8 = undefined;
    const out = vis.read(&s, &buf);

    try testing.expectEqual(@as(usize, 5), out.ok.len);
    try testing.expectEqualStrings("hello", out.ok);
    try testing.expectEqualStrings(", world!", s.buffer.items);

    // Second read pulls remainder.
    var buf2: [16]u8 = undefined;
    const out2 = vis.read(&s, &buf2);
    try testing.expectEqualStrings(", world!", out2.ok);
    try testing.expectEqual(@as(usize, 0), s.buffer.items.len);
}

test "VirtualInputStream.read: empty + writer-closed surfaces as closed" {
    const testing = std.testing;
    var s = async_mod.AsyncStream{};
    defer s.deinit(testing.allocator);
    s.write_closed = true;

    const vis = VirtualInputStream{ .stream_handle = 0 };
    var buf: [4]u8 = undefined;
    try testing.expect(vis.read(&s, &buf) == .closed);
}

test "VirtualOutputStream.write: appends to FIFO" {
    const testing = std.testing;
    var s = async_mod.AsyncStream{};
    defer s.deinit(testing.allocator);

    const vos = VirtualOutputStream{ .stream_handle = 1 };
    const out = try vos.write(&s, testing.allocator, "abc");
    try testing.expectEqual(@as(usize, 3), out.ok);
    try testing.expectEqualStrings("abc", s.buffer.items);

    const out2 = try vos.write(&s, testing.allocator, "def");
    try testing.expectEqual(@as(usize, 3), out2.ok);
    try testing.expectEqualStrings("abcdef", s.buffer.items);
}

test "VirtualOutputStream.write: closed when reader dropped" {
    const testing = std.testing;
    var s = async_mod.AsyncStream{};
    defer s.deinit(testing.allocator);
    s.read_closed = true;

    const vos = VirtualOutputStream{ .stream_handle = 2 };
    const out = try vos.write(&s, testing.allocator, "x");
    try testing.expect(out == .closed);
}

test "stream<u8> round-trip: write then read through both virtual ends" {
    const testing = std.testing;
    var s = async_mod.AsyncStream{};
    defer s.deinit(testing.allocator);

    const vos = VirtualOutputStream{ .stream_handle = 7 };
    const vis = VirtualInputStream{ .stream_handle = 7 };

    const w = try vos.write(&s, testing.allocator, "ping-pong");
    try testing.expectEqual(@as(usize, 9), w.ok);

    var buf: [9]u8 = undefined;
    const r = vis.read(&s, &buf);
    try testing.expectEqualStrings("ping-pong", r.ok);

    vos.markWriteClosed(&s);
    const r2 = vis.read(&s, &buf);
    try testing.expect(r2 == .closed);
    try testing.expectEqual(async_mod.AsyncStream.State.open, s.state);

    vis.markReadClosed(&s);
    try testing.expectEqual(async_mod.AsyncStream.State.closed, s.state);
}
