//! WASI Preview 2 I/O streams and poll.
//!
//! Implements wasi:io/streams (input-stream, output-stream) and
//! wasi:io/poll (pollable) as resource types with read/write operations.

const std = @import("std");

// ── wasi:io/streams — input-stream ──────────────────────────────────────────

/// An input stream resource — a readable byte source.
pub const InputStream = struct {
    source: Source,

    pub const Source = union(enum) {
        /// Backed by a fixed buffer (e.g., stdin capture).
        buffer: struct {
            data: []const u8,
            pos: usize = 0,
        },
        /// Backed by a host file descriptor.
        fd: std.posix.fd_t,
        /// Backed by a host file. Reads use positional `pread` so multiple
        /// streams over the same file are independent. The `file` pointer
        /// is borrowed from a `wasi:filesystem` descriptor table slot —
        /// the stream does not close it on drop.
        host_file: HostFile,
        /// Closed / exhausted.
        closed,
    };

    pub const HostFile = struct {
        file: std.Io.File,
        offset: u64 = 0,
    };

    /// Read up to `len` bytes. Returns the bytes read (may be fewer than len).
    pub fn read(self: *InputStream, buf: []u8) StreamResult {
        switch (self.source) {
            .buffer => |*b| {
                const avail = b.data.len - b.pos;
                if (avail == 0) return .{ .closed = {} };
                const n = @min(avail, buf.len);
                @memcpy(buf[0..n], b.data[b.pos..][0..n]);
                b.pos += n;
                return .{ .ok = n };
            },
            .fd => {
                // Host fd reading would go here
                return .{ .ok = 0 };
            },
            .host_file => |*hf| {
                const io = std.Io.Threaded.global_single_threaded.io();
                const n = hf.file.readPositionalAll(io, buf, hf.offset) catch
                    return .{ .err = .io_error };
                if (n == 0) return .{ .closed = {} };
                hf.offset += n;
                return .{ .ok = n };
            },
            .closed => return .{ .closed = {} },
        }
    }

    /// Create an input stream from a byte buffer.
    pub fn fromBuffer(data: []const u8) InputStream {
        return .{ .source = .{ .buffer = .{ .data = data } } };
    }

    /// Create an input stream that reads from a host file at the given offset.
    /// The `file` value is borrowed; the stream does not close it.
    pub fn fromHostFile(file: std.Io.File, offset: u64) InputStream {
        return .{ .source = .{ .host_file = .{ .file = file, .offset = offset } } };
    }
};

// ── wasi:io/streams — output-stream ─────────────────────────────────────────

/// An output stream resource — a writable byte sink.
pub const OutputStream = struct {
    sink: Sink,

    pub const Sink = union(enum) {
        /// Backed by a growable buffer (e.g., stdout capture).
        buffer: std.ArrayListUnmanaged(u8),
        /// Backed by a host file descriptor.
        fd: std.posix.fd_t,
        /// Backed by a host file. Writes use positional `pwrite`. When
        /// `append` is true, every write seeks to end-of-file first
        /// (sampled via `getEndPos`) so concurrent appenders interleave at
        /// record granularity. The `file` pointer is borrowed from a
        /// `wasi:filesystem` descriptor table slot — the stream does not
        /// close it on drop.
        host_file: HostFile,
        /// Closed.
        closed,
    };

    pub const HostFile = struct {
        file: std.Io.File,
        offset: u64 = 0,
        append: bool = false,
        /// When true, `flush` calls `file.sync(io)` after the most
        /// recent write so any buffered host-side data is persisted.
        /// Threaded from `wasi:filesystem` `descriptor-flags` bits
        /// `file-integrity-sync` / `data-integrity-sync` (#181).
        sync_on_flush: bool = false,
    };

    /// Write bytes to the stream. Returns number of bytes written.
    pub fn write(self: *OutputStream, data: []const u8, allocator: std.mem.Allocator) StreamResult {
        switch (self.sink) {
            .buffer => |*b| {
                b.appendSlice(allocator, data) catch return .{ .err = .would_block };
                return .{ .ok = data.len };
            },
            .fd => {
                // fd-backed sinks aren't yet used in production. Treating
                // every write as a successful no-op keeps the API shape
                // intact for future use without dragging in
                // platform-specific write syscalls.
                return .{ .ok = data.len };
            },
            .host_file => |*hf| {
                const io = std.Io.Threaded.global_single_threaded.io();
                if (hf.append) {
                    hf.offset = hf.file.length(io) catch
                        return .{ .err = .io_error };
                }
                hf.file.writePositionalAll(io, data, hf.offset) catch
                    return .{ .err = .io_error };
                hf.offset += data.len;
                return .{ .ok = data.len };
            },
            .closed => return .{ .closed = {} },
        }
    }

    /// Create an output stream backed by a growable buffer.
    pub fn toBuffer() OutputStream {
        return .{ .sink = .{ .buffer = .empty } };
    }

    /// Create an output stream that writes to a host file descriptor
    /// (e.g. real stdout/stderr). The fd is borrowed; the stream does
    /// not close it on `deinit`.
    pub fn toFd(fd: std.posix.fd_t) OutputStream {
        return .{ .sink = .{ .fd = fd } };
    }

    /// Create an output stream that writes to a host file at the given
    /// offset. If `append` is true, each write seeks to end-of-file
    /// first. The `file` value is borrowed; the stream does not close
    /// it on `deinit`. When `sync_on_flush` is true, `flush()` calls
    /// `file.sync()` to persist host-side buffers (#181).
    pub fn toHostFile(file: std.Io.File, offset: u64, append: bool, sync_on_flush: bool) OutputStream {
        return .{ .sink = .{ .host_file = .{
            .file = file,
            .offset = offset,
            .append = append,
            .sync_on_flush = sync_on_flush,
        } } };
    }

    /// Flush any host-side buffering. For host-file sinks with
    /// `sync_on_flush` set, this issues `file.sync()` so writes
    /// reach stable storage. Buffer / fd / closed sinks are no-ops.
    pub fn flush(self: *OutputStream) StreamResult {
        switch (self.sink) {
            .host_file => |*hf| {
                if (!hf.sync_on_flush) return .{ .ok = 0 };
                const io = std.Io.Threaded.global_single_threaded.io();
                hf.file.sync(io) catch return .{ .err = .io_error };
                return .{ .ok = 0 };
            },
            .closed => return .{ .closed = {} },
            else => return .{ .ok = 0 },
        }
    }

    /// Get the buffer contents (only valid for buffer-backed streams).
    pub fn getBufferContents(self: *const OutputStream) []const u8 {
        return switch (self.sink) {
            .buffer => |b| b.items,
            else => &.{},
        };
    }

    pub fn deinit(self: *OutputStream, allocator: std.mem.Allocator) void {
        switch (self.sink) {
            .buffer => |*b| b.deinit(allocator),
            else => {},
        }
    }
};

// ── Stream result ───────────────────────────────────────────────────────────

pub const StreamResult = union(enum) {
    ok: usize,
    closed,
    err: StreamError,
};

pub const StreamError = enum {
    would_block,
    broken_pipe,
    io_error,
};

// ── wasi:io/poll — pollable ─────────────────────────────────────────────────

/// A pollable resource — represents an async readiness notification.
pub const Pollable = struct {
    source: PollSource,

    pub const PollSource = union(enum) {
        /// Ready when a timer expires.
        timer: u64, // absolute monotonic nanoseconds
        /// Ready when an input stream has data.
        input_stream: *InputStream,
        /// Ready when an output stream can accept data.
        output_stream: *OutputStream,
        /// Always ready.
        immediate,
    };

    /// Check if this pollable is currently ready.
    pub fn isReady(self: *const Pollable) bool {
        return switch (self.source) {
            .timer => |deadline| blk: {
                const core = @import("core.zig");
                break :blk core.MonotonicClock.now() >= deadline;
            },
            .input_stream => |s| switch (s.source) {
                .buffer => |b| b.pos < b.data.len,
                .closed => true,
                else => false,
            },
            .output_stream => |s| switch (s.sink) {
                .buffer => true, // buffer can always accept
                .closed => true,
                else => false,
            },
            .immediate => true,
        };
    }
};

/// Poll a list of pollables. Returns the indices of ready pollables.
pub fn poll(pollables: []const *Pollable, out: []u32) u32 {
    var count: u32 = 0;
    for (pollables, 0..) |p, i| {
        if (p.isReady()) {
            if (count < out.len) {
                out[count] = @intCast(i);
                count += 1;
            }
        }
    }
    return count;
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "InputStream: read from buffer" {
    var stream = InputStream.fromBuffer("hello");
    var buf: [10]u8 = undefined;
    const r = stream.read(&buf);
    try std.testing.expectEqual(@as(usize, 5), r.ok);
    try std.testing.expectEqualSlices(u8, "hello", buf[0..5]);

    // Second read returns closed
    const r2 = stream.read(&buf);
    try std.testing.expect(r2 == .closed);
}

test "OutputStream: write to buffer" {
    var stream = OutputStream.toBuffer();
    defer stream.deinit(std.testing.allocator);

    const r = stream.write("world", std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 5), r.ok);
    try std.testing.expectEqualSlices(u8, "world", stream.getBufferContents());
}

test "Pollable: immediate is always ready" {
    const p = Pollable{ .source = .immediate };
    try std.testing.expect(p.isReady());
}

test "Pollable: timer in past is ready" {
    const p = Pollable{ .source = .{ .timer = 0 } };
    try std.testing.expect(p.isReady());
}

test "poll: returns ready indices" {
    var p1 = Pollable{ .source = .immediate };
    var p2 = Pollable{ .source = .{ .timer = std.math.maxInt(u64) } }; // far future
    var p3 = Pollable{ .source = .immediate };
    const pollables = [_]*Pollable{ &p1, &p2, &p3 };
    var out: [4]u32 = undefined;
    const count = poll(&pollables, &out);
    try std.testing.expectEqual(@as(u32, 2), count);
    try std.testing.expectEqual(@as(u32, 0), out[0]);
    try std.testing.expectEqual(@as(u32, 2), out[1]);
}
