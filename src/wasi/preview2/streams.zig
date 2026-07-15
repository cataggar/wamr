//! WASI Preview 2 I/O streams and poll.
//!
//! Implements wasi:io/streams (input-stream, output-stream) and
//! wasi:io/poll (pollable) as resource types with read/write operations.

const std = @import("std");
const builtin = @import("builtin");
const tls = @import("tls");

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
        /// Backed by a TCP connection. The fd is borrowed from a
        /// `Socket.tcp_stream` — the stream does not close it on drop.
        /// Only `Socket.closeAll` closes the underlying connection.
        tcp_stream: std.posix.fd_t,
        /// Backed by a TLS server connection (HTTPS termination, #609).
        /// The `*tls.Connection` is borrowed — the underlying socket and
        /// the connection are owned by the caller (`serveOneHttpConnectionP3`);
        /// the stream neither closes the fd nor sends `close_notify` on drop.
        tls: *tls.Connection,
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
            .fd => |fd| {
                // Raw fd source for live host stdio (#474). Uses
                // `std.Io.File.readStreaming` (the cross-platform
                // streaming reader), matching the symmetric `.fd`
                // write path. A zero-byte read surfaces as
                // `error.EndOfStream` and maps to `.closed`.
                const io = std.Io.Threaded.global_single_threaded.io();
                const file: std.Io.File = .{ .handle = fd, .flags = .{ .nonblocking = false } };
                const bufs = [_][]u8{buf};
                const n = file.readStreaming(io, &bufs) catch |err| return switch (err) {
                    error.EndOfStream => .{ .closed = {} },
                    error.WouldBlock => .{ .err = .would_block },
                    error.NotOpenForReading, error.SocketUnconnected, error.ConnectionResetByPeer => .{ .closed = {} },
                    else => .{ .err = .io_error },
                };
                if (n == 0) return .{ .closed = {} };
                return .{ .ok = n };
            },
            .host_file => |*hf| {
                const io = std.Io.Threaded.global_single_threaded.io();
                const n = hf.file.readPositionalAll(io, buf, hf.offset) catch
                    return .{ .err = .io_error };
                if (n == 0) return .{ .closed = {} };
                hf.offset += n;
                return .{ .ok = n };
            },
            .tcp_stream => |fd| {
                const io = std.Io.Threaded.global_single_threaded.io();
                var iovecs = [_][]u8{buf};
                const n = io.vtable.netRead(io.userdata, fd, &iovecs) catch
                    return .{ .err = .io_error };
                if (n == 0) return .{ .closed = {} };
                return .{ .ok = n };
            },
            .tls => |conn| {
                // `tls.Connection.read` returns 0 on a clean TLS close
                // (`close_notify`) or peer EOF; TLS / record errors surface
                // as `.io_error` (the library has already emitted an alert
                // on the wire for protocol-level failures).
                const n = conn.read(buf) catch return .{ .err = .io_error };
                if (n == 0) return .{ .closed = {} };
                return .{ .ok = n };
            },
            .closed => return .{ .closed = {} },
        }
    }

    /// Create an input stream from a byte buffer.
    pub fn fromBuffer(data: []const u8) InputStream {
        return .{ .source = .{ .buffer = .{ .data = data } } };
    }

    /// Create an input stream that reads from a host file descriptor
    /// using `std.posix.read` (non-positional). Use for streaming
    /// sources like stdin or a pipe read-end — for seekable files,
    /// prefer `fromHostFile` so multiple streams over the same file
    /// stay independent. The fd is borrowed; the stream does not close
    /// it on drop. Added for #474 live host stdio.
    pub fn fromFd(fd: std.posix.fd_t) InputStream {
        return .{ .source = .{ .fd = fd } };
    }

    /// Create an input stream that reads from a host file at the given offset.
    /// The `file` value is borrowed; the stream does not close it.
    pub fn fromHostFile(file: std.Io.File, offset: u64) InputStream {
        return .{ .source = .{ .host_file = .{ .file = file, .offset = offset } } };
    }

    /// Create an input stream backed by a TCP connection fd.
    /// The fd is borrowed; the stream does not close it.
    pub fn fromTcpStream(fd: std.posix.fd_t) InputStream {
        return .{ .source = .{ .tcp_stream = fd } };
    }

    /// Create an input stream backed by a TLS server connection (#609).
    /// The `*tls.Connection` is borrowed; the stream neither closes the
    /// underlying fd nor sends `close_notify` on drop.
    pub fn fromTlsConn(conn: *tls.Connection) InputStream {
        return .{ .source = .{ .tls = conn } };
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
        /// Backed by a TCP connection. The fd is borrowed from a
        /// `Socket.tcp_stream` — the stream does not close it on drop.
        tcp_stream: std.posix.fd_t,
        /// Backed by a TLS server connection (HTTPS termination, #609).
        /// The `*tls.Connection` is borrowed — owned by the caller
        /// (`serveOneHttpConnectionP3`); the stream does not close it.
        tls: *tls.Connection,
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
            .fd => |fd| {
                // Raw fd sink for live host stdio (#474). Uses
                // `std.Io.File.writeStreamingAll` which is the
                // non-positional streaming writer (vs `pwrite`-style
                // `host_file`), suitable for stdout/stderr/pipes. EAGAIN
                // surfaces as `error.WouldBlock`; `BrokenPipe` /
                // `NotOpenForWriting` are treated as `.closed`.
                const io = std.Io.Threaded.global_single_threaded.io();
                const file: std.Io.File = .{ .handle = fd, .flags = .{ .nonblocking = false } };
                file.writeStreamingAll(io, data) catch |err| return switch (err) {
                    error.WouldBlock => .{ .err = .would_block },
                    error.BrokenPipe, error.NotOpenForWriting => .{ .closed = {} },
                    else => .{ .err = .io_error },
                };
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
            .tcp_stream => |fd| {
                // A zero-length write is a no-op on a socket, and on
                // Windows `netWrite` with an empty buffer fails with
                // `INVALID_PARAMETER` — short-circuit it (e.g. an empty
                // HTTP response body, the `404, ""` case).
                if (data.len == 0) return .{ .ok = 0 };
                const io = std.Io.Threaded.global_single_threaded.io();
                const slices = [_][]const u8{data};
                _ = io.vtable.netWrite(io.userdata, fd, &.{}, &slices, 1) catch
                    return .{ .err = .io_error };
                return .{ .ok = data.len };
            },
            .tls => |conn| {
                // `writeAll` encrypts `data` into one or more TLS records and
                // flushes them to the underlying socket. Any failure (record
                // overflow, socket error) maps to `.io_error`.
                conn.writeAll(data) catch return .{ .err = .io_error };
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

    /// Create an output stream backed by a TCP connection fd.
    /// The fd is borrowed; the stream does not close it.
    pub fn toTcpStream(fd: std.posix.fd_t) OutputStream {
        return .{ .sink = .{ .tcp_stream = fd } };
    }

    /// Create an output stream backed by a TLS server connection (#609).
    /// The `*tls.Connection` is borrowed; the stream does not close it.
    pub fn toTlsConn(conn: *tls.Connection) OutputStream {
        return .{ .sink = .{ .tls = conn } };
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

/// Result of attempting the Linux zero-copy stream transfer. `unsupported`
/// means no bytes moved and the caller may safely use its buffered path.
pub const SpliceResult = union(enum) {
    ok: usize,
    closed,
    unsupported,
    err: StreamError,
};

pub const SpliceMode = enum {
    nonblocking,
    blocking,
};

const LinuxSpliceEndpoint = struct {
    fd: std.posix.fd_t,
    offset: ?*u64 = null,
};

fn linuxInputSpliceEndpoint(stream: *InputStream) ?LinuxSpliceEndpoint {
    return switch (stream.source) {
        .fd => |fd| .{ .fd = fd },
        .host_file => |*hf| .{ .fd = hf.file.handle, .offset = &hf.offset },
        .tcp_stream => |fd| .{ .fd = fd },
        else => null,
    };
}

fn linuxOutputSpliceEndpoint(stream: *OutputStream) ?LinuxSpliceEndpoint {
    return switch (stream.sink) {
        .fd => |fd| .{ .fd = fd },
        .host_file => |*hf| .{ .fd = hf.file.handle, .offset = &hf.offset },
        .tcp_stream => |fd| .{ .fd = fd },
        else => null,
    };
}

fn linuxIsPipe(fd: std.posix.fd_t) bool {
    const linux = std.os.linux;
    return linux.errno(linux.fcntl(fd, linux.F.GETPIPE_SZ, 0)) == .SUCCESS;
}

const LinuxReady = enum {
    ready,
    not_ready,
    err,
};

fn linuxFdReady(fd: std.posix.fd_t, events: i16, timeout_ms: i32) LinuxReady {
    var fds = [_]std.posix.pollfd{.{
        .fd = fd,
        .events = events,
        .revents = 0,
    }};
    const n = std.posix.poll(&fds, timeout_ms) catch return .err;
    if (n == 0) return .not_ready;
    const invalid: i16 = @intCast(std.c.POLL.NVAL);
    if ((fds[0].revents & invalid) != 0) return .err;
    const ready_events = events |
        @as(i16, @intCast(std.c.POLL.ERR)) |
        @as(i16, @intCast(std.c.POLL.HUP));
    return if ((fds[0].revents & ready_events) != 0) .ready else .not_ready;
}

fn linuxSpliceReady(in_fd: std.posix.fd_t, out_fd: std.posix.fd_t, timeout_ms: i32) LinuxReady {
    const write_ready = linuxFdReady(out_fd, @intCast(std.c.POLL.OUT), timeout_ms);
    if (write_ready != .ready) return write_ready;
    return linuxFdReady(in_fd, @intCast(std.c.POLL.IN), timeout_ms);
}

const SigpipeTestMutex = struct {
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

    pub fn lock(self: *SigpipeTestMutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null)
            std.atomic.spinLoopHint();
    }

    pub fn unlock(self: *SigpipeTestMutex) void {
        self.state.store(0, .release);
    }
};

pub const sigpipe_test_state = if (builtin.is_test) struct {
    pub var mutex: SigpipeTestMutex = .{};
} else struct {};

fn linuxSpliceNoSigpipe(
    in_fd: std.posix.fd_t,
    in_offset_arg: usize,
    out_fd: std.posix.fd_t,
    out_offset_arg: usize,
    len: usize,
    flags: usize,
) usize {
    const linux = std.os.linux;
    var pipe_set = linux.sigemptyset();
    linux.sigaddset(&pipe_set, .PIPE);

    // SIGPIPE is synchronous to this thread. Block it only around splice,
    // preserving both the caller's mask and any already-pending SIGPIPE.
    var old_mask: linux.sigset_t = undefined;
    const block_rc = linux.sigprocmask(linux.SIG.BLOCK, &pipe_set, &old_mask);
    if (linux.errno(block_rc) != .SUCCESS) return block_rc;

    var pending_before = linux.sigemptyset();
    const pending_rc = linux.syscall2(
        .rt_sigpending,
        @intFromPtr(&pending_before),
        linux.NSIG / 8,
    );
    if (linux.errno(pending_rc) != .SUCCESS) {
        _ = linux.sigprocmask(linux.SIG.SETMASK, &old_mask, null);
        return pending_rc;
    }
    const pipe_was_pending = linux.sigismember(&pending_before, .PIPE);

    const rc = linux.syscall6(
        .splice,
        @as(usize, @bitCast(@as(isize, in_fd))),
        in_offset_arg,
        @as(usize, @bitCast(@as(isize, out_fd))),
        out_offset_arg,
        len,
        flags,
    );

    // Consume only the SIGPIPE generated by this call. If one was already
    // pending, standard-signal coalescing means consuming it would steal the
    // caller's signal, so leave it pending.
    if (linux.errno(rc) == .PIPE and !pipe_was_pending) {
        const zero_timeout: linux.timespec = .{ .sec = 0, .nsec = 0 };
        while (true) {
            const wait_rc = linux.syscall4(
                .rt_sigtimedwait,
                @intFromPtr(&pipe_set),
                0,
                @intFromPtr(&zero_timeout),
                linux.NSIG / 8,
            );
            if (linux.errno(wait_rc) != .INTR) break;
        }
    }

    _ = linux.sigprocmask(linux.SIG.SETMASK, &old_mask, null);
    return rc;
}

/// Try Linux `splice(2)` for descriptor-backed streams. Nonblocking mode is
/// restricted to pipe-to-pipe transfers because `SPLICE_F_NONBLOCK` does not
/// guarantee nonblocking I/O on a non-pipe endpoint. The transfer is limited
/// to one successful syscall, matching the short-read behavior of `read`.
pub fn splice(src: *InputStream, dst: *OutputStream, len: usize, mode: SpliceMode) SpliceResult {
    if (src.source == .closed or dst.sink == .closed) return .closed;
    if (len == 0) return .{ .ok = 0 };
    if (comptime builtin.os.tag != .linux) return .unsupported;

    const in = linuxInputSpliceEndpoint(src) orelse return .unsupported;
    const out = linuxOutputSpliceEndpoint(dst) orelse return .unsupported;
    const in_is_pipe = linuxIsPipe(in.fd);
    const out_is_pipe = linuxIsPipe(out.fd);

    if (mode == .nonblocking) {
        if (!in_is_pipe or !out_is_pipe) return .unsupported;
    }

    if (dst.sink == .host_file and dst.sink.host_file.append) {
        const io = std.Io.Threaded.global_single_threaded.io();
        dst.sink.host_file.offset = dst.sink.host_file.file.length(io) catch
            return .{ .err = .io_error };
    }

    if (in.offset != null and in.offset.?.* > std.math.maxInt(i64)) return .unsupported;
    if (out.offset != null and out.offset.?.* > std.math.maxInt(i64)) return .unsupported;

    var in_offset: i64 = if (in.offset) |offset| @intCast(offset.*) else 0;
    var out_offset: i64 = if (out.offset) |offset| @intCast(offset.*) else 0;
    const in_offset_arg: usize = if (in.offset != null) @intFromPtr(&in_offset) else 0;
    const out_offset_arg: usize = if (out.offset != null) @intFromPtr(&out_offset) else 0;
    const linux = std.os.linux;
    const splice_f_nonblock: usize = 2;

    while (true) {
        const rc = linuxSpliceNoSigpipe(
            in.fd,
            in_offset_arg,
            out.fd,
            out_offset_arg,
            len,
            if (mode == .nonblocking) splice_f_nonblock else 0,
        );
        switch (linux.errno(rc)) {
            .SUCCESS => {
                const n: usize = @intCast(rc);
                if (n == 0) return .closed;
                if (in.offset) |offset| offset.* = @intCast(in_offset);
                if (out.offset) |offset| offset.* = @intCast(out_offset);
                return .{ .ok = n };
            },
            .INTR => continue,
            // These errors describe an endpoint pair that splice(2) cannot
            // handle. The syscall transferred nothing, so buffered I/O is
            // still safe.
            .INVAL, .NOSYS, .OPNOTSUPP, .SPIPE => return .unsupported,
            .PIPE => return .closed,
            .AGAIN => switch (mode) {
                .nonblocking => return .{ .ok = 0 },
                .blocking => switch (linuxSpliceReady(in.fd, out.fd, -1)) {
                    .ready => continue,
                    .not_ready => unreachable,
                    .err => return .{ .err = .io_error },
                },
            },
            else => return .{ .err = .io_error },
        }
    }
}

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

test "OutputStream: write to fd writes through to peer (#474)" {
    if (comptime @import("builtin").os.tag != .linux) return error.SkipZigTest;
    if (comptime @import("builtin").os.tag == .linux) {
        const linux = std.os.linux;
        var fds: [2]i32 = undefined;
        if (linux.errno(linux.pipe2(&fds, .{})) != .SUCCESS) return error.SkipZigTest;
        defer _ = linux.close(fds[0]);
        defer _ = linux.close(fds[1]);

        var stream = OutputStream.toFd(fds[1]);
        const r = stream.write("hello", std.testing.allocator);
        try std.testing.expectEqual(@as(usize, 5), r.ok);

        var buf: [16]u8 = undefined;
        const n = try std.posix.read(fds[0], &buf);
        try std.testing.expectEqualSlices(u8, "hello", buf[0..n]);
    }
}

test "OutputStream: write to fd returns closed when peer closed (#474)" {
    if (comptime @import("builtin").os.tag != .linux) return error.SkipZigTest;
    if (comptime @import("builtin").os.tag == .linux) {
        const linux = std.os.linux;
        sigpipe_test_state.mutex.lock();
        defer sigpipe_test_state.mutex.unlock();
        var fds: [2]i32 = undefined;
        if (linux.errno(linux.pipe2(&fds, .{})) != .SUCCESS) return error.SkipZigTest;
        _ = linux.close(fds[0]); // close read-end first
        defer _ = linux.close(fds[1]);

        // Ignore SIGPIPE so the write surfaces as EPIPE instead of killing
        // the test process.
        var act: linux.Sigaction = .{
            .handler = .{ .handler = linux.SIG.IGN },
            .mask = std.posix.sigemptyset(),
            .flags = 0,
        };
        var oact: linux.Sigaction = undefined;
        std.posix.sigaction(linux.SIG.PIPE, &act, &oact);
        defer std.posix.sigaction(linux.SIG.PIPE, &oact, null);

        var stream = OutputStream.toFd(fds[1]);
        const r = stream.write("hello", std.testing.allocator);
        try std.testing.expect(r == .closed);
    }
}

test "InputStream: read from fd returns bytes written by peer (#474)" {
    if (comptime @import("builtin").os.tag != .linux) return error.SkipZigTest;
    if (comptime @import("builtin").os.tag == .linux) {
        const linux = std.os.linux;
        var fds: [2]i32 = undefined;
        if (linux.errno(linux.pipe2(&fds, .{})) != .SUCCESS) return error.SkipZigTest;
        defer _ = linux.close(fds[0]);
        defer _ = linux.close(fds[1]);

        // Preload bytes into the pipe write-end via the raw syscall
        // (std.posix has no `write` wrapper in 0.16).
        _ = linux.write(fds[1], "world", 5);

        var stream = InputStream.fromFd(fds[0]);
        var buf: [16]u8 = undefined;
        const r = stream.read(&buf);
        try std.testing.expectEqual(@as(usize, 5), r.ok);
        try std.testing.expectEqualSlices(u8, "world", buf[0..5]);
    }
}

test "InputStream: read from fd returns closed at EOF (#474)" {
    if (comptime @import("builtin").os.tag != .linux) return error.SkipZigTest;
    if (comptime @import("builtin").os.tag == .linux) {
        const linux = std.os.linux;
        var fds: [2]i32 = undefined;
        if (linux.errno(linux.pipe2(&fds, .{})) != .SUCCESS) return error.SkipZigTest;
        defer _ = linux.close(fds[0]);
        _ = linux.close(fds[1]); // close write-end → next read sees EOF

        var stream = InputStream.fromFd(fds[0]);
        var buf: [16]u8 = undefined;
        const r = stream.read(&buf);
        try std.testing.expect(r == .closed);
    }
}

test "splice: Linux pipe fast path honors requested length (#616 A2)" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    if (comptime builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var source_fds: [2]i32 = undefined;
        var dest_fds: [2]i32 = undefined;
        if (linux.errno(linux.pipe2(&source_fds, .{})) != .SUCCESS) return error.SkipZigTest;
        defer _ = linux.close(source_fds[0]);
        defer _ = linux.close(source_fds[1]);
        if (linux.errno(linux.pipe2(&dest_fds, .{})) != .SUCCESS) return error.SkipZigTest;
        defer _ = linux.close(dest_fds[0]);
        defer _ = linux.close(dest_fds[1]);

        try std.testing.expectEqual(.SUCCESS, linux.errno(linux.write(source_fds[1], "abcdef", 6)));

        var src = InputStream.fromFd(source_fds[0]);
        var dst = OutputStream.toFd(dest_fds[1]);
        const result = splice(&src, &dst, 3, .blocking);
        try std.testing.expectEqual(@as(usize, 3), result.ok);

        var buf: [8]u8 = undefined;
        const dest_n = linux.read(dest_fds[0], &buf, buf.len);
        try std.testing.expectEqual(.SUCCESS, linux.errno(dest_n));
        try std.testing.expectEqualStrings("abc", buf[0..dest_n]);

        const source_n = linux.read(source_fds[0], &buf, buf.len);
        try std.testing.expectEqual(.SUCCESS, linux.errno(source_n));
        try std.testing.expectEqualStrings("def", buf[0..source_n]);
    }
}

test "splice: Linux pipe EOF reports closed (#616 A2)" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    if (comptime builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var source_fds: [2]i32 = undefined;
        var dest_fds: [2]i32 = undefined;
        if (linux.errno(linux.pipe2(&source_fds, .{})) != .SUCCESS) return error.SkipZigTest;
        defer _ = linux.close(source_fds[0]);
        _ = linux.close(source_fds[1]);
        if (linux.errno(linux.pipe2(&dest_fds, .{})) != .SUCCESS) return error.SkipZigTest;
        defer _ = linux.close(dest_fds[0]);
        defer _ = linux.close(dest_fds[1]);

        var src = InputStream.fromFd(source_fds[0]);
        var dst = OutputStream.toFd(dest_fds[1]);
        try std.testing.expect(splice(&src, &dst, 16, .blocking) == .closed);
    }
}

test "splice: Linux regular files request safe fallback (#616 A2)" {
    if (comptime builtin.os.tag != .linux) return error.SkipZigTest;
    if (comptime builtin.os.tag == .linux) {
        const io = std.Io.Threaded.global_single_threaded.io();
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();

        const source_file = try tmp.dir.createFile(io, "source", .{ .read = true });
        defer source_file.close(io);
        try source_file.writePositionalAll(io, "payload", 0);
        const dest_file = try tmp.dir.createFile(io, "dest", .{ .read = true });
        defer dest_file.close(io);

        var src = InputStream.fromHostFile(source_file, 0);
        var dst = OutputStream.toHostFile(dest_file, 0, false, false);
        try std.testing.expect(splice(&src, &dst, 7, .blocking) == .unsupported);
        try std.testing.expectEqual(@as(u64, 0), src.source.host_file.offset);
        try std.testing.expectEqual(@as(u64, 0), dst.sink.host_file.offset);
    }
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
