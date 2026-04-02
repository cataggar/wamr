//! WASI Preview1 Implementation
//!
//! Pure Zig implementation of the WASI preview1 syscall interface,
//! replacing libuv + uvwasi + Wasmtime SSP with `std.Io` (file I/O,
//! clocks, secure random) from the Zig standard library.

const std = @import("std");
const Io = std.Io;
const File = Io.File;

// ── WASI Error Codes ────────────────────────────────────────────────────

/// WASI errno values per the WASI preview1 specification.
pub const Errno = enum(u16) {
    success = 0,
    toobig = 1,
    acces = 2,
    addrinuse = 3,
    addrnotavail = 4,
    afnosupport = 5,
    again = 6,
    already = 7,
    badf = 8,
    badmsg = 9,
    busy = 10,
    canceled = 11,
    child = 12,
    connaborted = 13,
    connrefused = 14,
    connreset = 15,
    deadlk = 16,
    destaddrreq = 17,
    dom = 18,
    dquot = 19,
    exist = 20,
    fault = 21,
    fbig = 22,
    hostunreach = 23,
    idrm = 24,
    ilseq = 25,
    inprogress = 26,
    intr = 27,
    inval = 28,
    io = 29,
    isconn = 30,
    isdir = 31,
    loop = 32,
    mfile = 33,
    mlink = 34,
    msgsize = 35,
    multihop = 36,
    nametoolong = 37,
    netdown = 38,
    netreset = 39,
    netunreach = 40,
    nfile = 41,
    nobufs = 42,
    nodev = 43,
    noent = 44,
    noexec = 45,
    nolck = 46,
    nolink = 47,
    nomem = 48,
    nomsg = 49,
    noprotoopt = 50,
    nospc = 51,
    nosys = 52,
    notconn = 53,
    notdir = 54,
    notempty = 55,
    notrecoverable = 56,
    notsock = 57,
    notsup = 58,
    notty = 59,
    nxio = 60,
    overflow = 61,
    ownerdead = 62,
    perm = 63,
    pipe = 64,
    proto = 65,
    protonosupport = 66,
    prototype = 67,
    range = 68,
    rofs = 69,
    spipe = 70,
    srch = 71,
    stale = 72,
    timedout = 73,
    txtbsy = 74,
    xdev = 75,
    notcapable = 76,
};

// ── WASI Clock IDs ──────────────────────────────────────────────────────

pub const ClockId = enum(u32) {
    realtime = 0,
    monotonic = 1,
    process_cputime_id = 2,
    thread_cputime_id = 3,
};

// ── WASI Whence values ──────────────────────────────────────────────────

pub const Whence = enum(u8) {
    set = 0,
    cur = 1,
    end = 2,
};

// ── IoVec ───────────────────────────────────────────────────────────────

pub const IoVec = struct {
    buf: [*]u8,
    len: u32,

    pub fn slice(self: IoVec) []u8 {
        return self.buf[0..self.len];
    }
};

// ── Preopen ─────────────────────────────────────────────────────────────

pub const Preopen = struct {
    fd: u32,
    path: []const u8,
};

// ── File Descriptor Table ───────────────────────────────────────────────

pub const FdEntry = struct {
    kind: FdKind,
    host_fd: ?std.posix.fd_t = null,

    pub const FdKind = enum {
        stdin,
        stdout,
        stderr,
        regular_file,
        directory,
        socket,
    };
};

pub const FdTable = struct {
    entries: std.AutoHashMap(u32, FdEntry),
    next_fd: u32 = 3,

    pub fn init(allocator: std.mem.Allocator) FdTable {
        return .{ .entries = std.AutoHashMap(u32, FdEntry).init(allocator) };
    }

    pub fn deinit(self: *FdTable) void {
        self.entries.deinit();
    }

    pub fn insert(self: *FdTable, fd: u32, entry: FdEntry) !void {
        try self.entries.put(fd, entry);
    }

    pub fn get(self: *const FdTable, fd: u32) ?FdEntry {
        return self.entries.get(fd);
    }

    pub fn remove(self: *FdTable, fd: u32) void {
        _ = self.entries.remove(fd);
    }

    pub fn allocateFd(self: *FdTable) u32 {
        const fd = self.next_fd;
        self.next_fd += 1;
        return fd;
    }
};

// ── WASI Context ────────────────────────────────────────────────────────

/// WASI execution context — tracks file descriptors, args, env, preopens.
pub const WasiCtx = struct {
    allocator: std.mem.Allocator,
    io: Io,
    args: []const []const u8 = &.{},
    env_vars: []const []const u8 = &.{},
    preopens: []const Preopen = &.{},
    fd_table: FdTable,
    exit_code: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, io: Io) !*WasiCtx {
        const ctx = try allocator.create(WasiCtx);
        ctx.* = .{
            .allocator = allocator,
            .io = io,
            .fd_table = FdTable.init(allocator),
        };
        // Pre-populate stdin(0), stdout(1), stderr(2)
        try ctx.fd_table.insert(0, .{ .kind = .stdin });
        try ctx.fd_table.insert(1, .{ .kind = .stdout });
        try ctx.fd_table.insert(2, .{ .kind = .stderr });
        return ctx;
    }

    pub fn deinit(self: *WasiCtx) void {
        self.fd_table.deinit();
        self.allocator.destroy(self);
    }

    pub fn setArgs(self: *WasiCtx, args: []const []const u8) void {
        self.args = args;
    }

    pub fn setEnv(self: *WasiCtx, env: []const []const u8) void {
        self.env_vars = env;
    }

    pub fn addPreopen(self: *WasiCtx, fd: u32, path: []const u8) !void {
        _ = self;
        _ = fd;
        _ = path;
        // TODO: implement preopened directory
    }

    // ── args ────────────────────────────────────────────────────────

    pub fn args_sizes_get(self: *const WasiCtx) struct { count: u32, buf_size: u32 } {
        var buf_size: u32 = 0;
        for (self.args) |arg| {
            // Each arg is NUL-terminated in the WASI buffer
            buf_size += @as(u32, @intCast(arg.len)) + 1;
        }
        return .{
            .count = @intCast(self.args.len),
            .buf_size = buf_size,
        };
    }

    pub fn args_get(self: *const WasiCtx, argv_buf: []u8) []const u8 {
        var offset: usize = 0;
        for (self.args) |arg| {
            if (offset + arg.len + 1 > argv_buf.len) break;
            @memcpy(argv_buf[offset..][0..arg.len], arg);
            argv_buf[offset + arg.len] = 0; // NUL terminator
            offset += arg.len + 1;
        }
        return argv_buf[0..offset];
    }

    // ── environ ─────────────────────────────────────────────────────

    pub fn environ_sizes_get(self: *const WasiCtx) struct { count: u32, buf_size: u32 } {
        var buf_size: u32 = 0;
        for (self.env_vars) |env| {
            buf_size += @as(u32, @intCast(env.len)) + 1;
        }
        return .{
            .count = @intCast(self.env_vars.len),
            .buf_size = buf_size,
        };
    }

    // ── clock ───────────────────────────────────────────────────────

    pub fn clock_time_get(self: *const WasiCtx, clock_id: u32, precision: u64) !u64 {
        _ = precision;
        const id = std.enums.fromInt(ClockId, clock_id) orelse return error.InvalidClockId;
        const clock: Io.Clock = switch (id) {
            .realtime => .real,
            .monotonic => .awake,
            .process_cputime_id => .cpu_process,
            .thread_cputime_id => .cpu_thread,
        };
        const ts = clock.now(self.io);
        const ns = ts.nanoseconds;
        if (ns < 0) return error.InvalidClockId;
        return @intCast(ns);
    }

    // ── fd operations ───────────────────────────────────────────────

    pub fn fd_write(self: *WasiCtx, fd: u32, iovs: []const IoVec) !struct { nwritten: u32 } {
        const entry = self.fd_table.get(fd) orelse return error.BadFd;
        var total_written: u32 = 0;

        switch (entry.kind) {
            .stdout => {
                var buf: [4096]u8 = undefined;
                var w = File.stdout().writer(self.io, &buf);
                for (iovs) |iov| {
                    w.interface.writeAll(iov.slice()) catch return error.IoError;
                    total_written += iov.len;
                }
                w.flush() catch return error.IoError;
            },
            .stderr => {
                var buf: [4096]u8 = undefined;
                var w = File.stderr().writer(self.io, &buf);
                for (iovs) |iov| {
                    w.interface.writeAll(iov.slice()) catch return error.IoError;
                    total_written += iov.len;
                }
                w.flush() catch return error.IoError;
            },
            .regular_file => {
                if (entry.host_fd) |host_fd| {
                    const file = File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
                    var buf: [4096]u8 = undefined;
                    var w = file.writer(self.io, &buf);
                    for (iovs) |iov| {
                        w.interface.writeAll(iov.slice()) catch return error.IoError;
                        total_written += iov.len;
                    }
                    w.flush() catch return error.IoError;
                } else {
                    return error.BadFd;
                }
            },
            else => return error.BadFd,
        }

        return .{ .nwritten = total_written };
    }

    pub fn fd_read(self: *WasiCtx, fd: u32, iovs: []const IoVec) !struct { nread: u32 } {
        const entry = self.fd_table.get(fd) orelse return error.BadFd;
        var total_read: u32 = 0;

        switch (entry.kind) {
            .stdin => {
                var buf: [4096]u8 = undefined;
                var r = File.stdin().reader(self.io, &buf);
                for (iovs) |iov| {
                    const data = iov.slice();
                    const n = r.interface.read(data) catch return error.IoError;
                    total_read += @intCast(n);
                    if (n < data.len) break;
                }
            },
            .regular_file => {
                if (entry.host_fd) |host_fd| {
                    const file = File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
                    var buf: [4096]u8 = undefined;
                    var r = file.reader(self.io, &buf);
                    for (iovs) |iov| {
                        const data = iov.slice();
                        const n = r.interface.read(data) catch return error.IoError;
                        total_read += @intCast(n);
                        if (n < data.len) break;
                    }
                } else {
                    return error.BadFd;
                }
            },
            else => return error.BadFd,
        }

        return .{ .nread = total_read };
    }

    pub fn fd_close(self: *WasiCtx, fd: u32) Errno {
        const entry = self.fd_table.get(fd) orelse return .badf;

        // Don't allow closing stdio
        switch (entry.kind) {
            .stdin, .stdout, .stderr => return .badf,
            else => {},
        }

        if (entry.host_fd) |host_fd| {
            const file = File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
            file.close(self.io);
        }

        self.fd_table.remove(fd);
        return .success;
    }

    pub fn fd_seek(self: *WasiCtx, fd: u32, offset: i64, whence: u8) !u64 {
        const entry = self.fd_table.get(fd) orelse return error.BadFd;

        switch (entry.kind) {
            .regular_file => {
                if (entry.host_fd) |host_fd| {
                    const w = std.enums.fromInt(Whence, whence) orelse return error.InvalidWhence;
                    const file = File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
                    var buf: [4096]u8 = undefined;
                    var reader = file.reader(self.io, &buf);
                    switch (w) {
                        .set => {
                            if (offset < 0) return error.InvalidWhence;
                            reader.seekTo(@intCast(offset)) catch return error.IoError;
                        },
                        .cur => reader.seekBy(offset) catch return error.IoError,
                        .end => {
                            const stat = file.stat(self.io) catch return error.IoError;
                            const size: i64 = @intCast(stat.size);
                            const new_pos = size + offset;
                            if (new_pos < 0) return error.InvalidWhence;
                            reader.seekTo(@intCast(new_pos)) catch return error.IoError;
                        },
                    }
                    // Return current position by seeking by 0
                    return error.NoSys; // TODO: position tracking
                } else {
                    return error.BadFd;
                }
            },
            .stdin, .stdout, .stderr => return error.SpPipe,
            else => return error.BadFd,
        }
    }

    // ── proc ────────────────────────────────────────────────────────

    pub fn proc_exit(self: *WasiCtx, code: u32) void {
        self.exit_code = code;
    }

    // ── random ──────────────────────────────────────────────────────

    pub fn random_get(self: *const WasiCtx, buf: []u8) void {
        self.io.random(buf);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing_io = std.testing.io;

test "WasiCtx init/deinit lifecycle" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    // Verify stdio fds are pre-populated
    try std.testing.expect(ctx.fd_table.get(0) != null);
    try std.testing.expect(ctx.fd_table.get(1) != null);
    try std.testing.expect(ctx.fd_table.get(2) != null);
    try std.testing.expect(ctx.fd_table.get(3) == null);
    try std.testing.expect(ctx.exit_code == null);
}

test "FdTable insert, get, remove" {
    var table = FdTable.init(std.testing.allocator);
    defer table.deinit();

    try table.insert(10, .{ .kind = .regular_file });
    const entry = table.get(10);
    try std.testing.expect(entry != null);
    try std.testing.expectEqual(FdEntry.FdKind.regular_file, entry.?.kind);

    table.remove(10);
    try std.testing.expect(table.get(10) == null);
}

test "FdTable allocateFd" {
    var table = FdTable.init(std.testing.allocator);
    defer table.deinit();

    const fd1 = table.allocateFd();
    const fd2 = table.allocateFd();
    try std.testing.expectEqual(@as(u32, 3), fd1);
    try std.testing.expectEqual(@as(u32, 4), fd2);
}

test "args_sizes_get with known args" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const args = [_][]const u8{ "hello", "world" };
    ctx.setArgs(&args);

    const sizes = ctx.args_sizes_get();
    try std.testing.expectEqual(@as(u32, 2), sizes.count);
    // "hello\0" (6) + "world\0" (6) = 12
    try std.testing.expectEqual(@as(u32, 12), sizes.buf_size);
}

test "args_get writes NUL-terminated args" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const args = [_][]const u8{ "ab", "cd" };
    ctx.setArgs(&args);

    var buf: [6]u8 = undefined;
    const result = ctx.args_get(&buf);
    try std.testing.expectEqual(@as(usize, 6), result.len);
    try std.testing.expectEqualSlices(u8, "ab\x00cd\x00", result);
}

test "environ_sizes_get" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const env = [_][]const u8{ "FOO=bar", "BAZ=qux" };
    ctx.setEnv(&env);

    const sizes = ctx.environ_sizes_get();
    try std.testing.expectEqual(@as(u32, 2), sizes.count);
    // "FOO=bar\0" (8) + "BAZ=qux\0" (8) = 16
    try std.testing.expectEqual(@as(u32, 16), sizes.buf_size);
}

test "fd_write to stdout does not crash" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    var data = "test output\n".*;
    const iovs = [_]IoVec{.{ .buf = &data, .len = @intCast(data.len) }};
    const result = try ctx.fd_write(1, &iovs);
    try std.testing.expectEqual(@as(u32, 12), result.nwritten);
}

test "fd_write to stderr does not crash" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    var data = "err output\n".*;
    const iovs = [_]IoVec{.{ .buf = &data, .len = @intCast(data.len) }};
    const result = try ctx.fd_write(2, &iovs);
    try std.testing.expectEqual(@as(u32, 11), result.nwritten);
}

test "fd_write to invalid fd returns error" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    var data = "nope".*;
    const iovs = [_]IoVec{.{ .buf = &data, .len = 4 }};
    const result = ctx.fd_write(999, &iovs);
    try std.testing.expectError(error.BadFd, result);
}

test "fd_close on stdio returns badf" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    try std.testing.expectEqual(Errno.badf, ctx.fd_close(0));
    try std.testing.expectEqual(Errno.badf, ctx.fd_close(1));
    try std.testing.expectEqual(Errno.badf, ctx.fd_close(2));
}

test "fd_close on missing fd returns badf" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    try std.testing.expectEqual(Errno.badf, ctx.fd_close(42));
}

test "clock_time_get returns increasing values" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const t1 = try ctx.clock_time_get(@intFromEnum(ClockId.monotonic), 0);
    // Small busy wait
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        std.mem.doNotOptimizeAway(i);
    }
    const t2 = try ctx.clock_time_get(@intFromEnum(ClockId.monotonic), 0);
    try std.testing.expect(t2 >= t1);
}

test "clock_time_get realtime returns nonzero" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const t = try ctx.clock_time_get(@intFromEnum(ClockId.realtime), 0);
    try std.testing.expect(t > 0);
}

test "clock_time_get invalid clock returns error" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const result = ctx.clock_time_get(99, 0);
    try std.testing.expectError(error.InvalidClockId, result);
}

test "random_get fills buffer with non-zero bytes (probabilistic)" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    var buf = [_]u8{0} ** 64;
    ctx.random_get(&buf);

    // It's astronomically unlikely that 64 random bytes are all zero
    var all_zero = true;
    for (buf) |b| {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    try std.testing.expect(!all_zero);
}

test "proc_exit sets exit code" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    try std.testing.expect(ctx.exit_code == null);
    ctx.proc_exit(42);
    try std.testing.expectEqual(@as(u32, 42), ctx.exit_code.?);
}

test "proc_exit with zero" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    ctx.proc_exit(0);
    try std.testing.expectEqual(@as(u32, 0), ctx.exit_code.?);
}

test "Errno values match WASI spec" {
    try std.testing.expectEqual(@as(u16, 0), @intFromEnum(Errno.success));
    try std.testing.expectEqual(@as(u16, 1), @intFromEnum(Errno.toobig));
    try std.testing.expectEqual(@as(u16, 2), @intFromEnum(Errno.acces));
    try std.testing.expectEqual(@as(u16, 8), @intFromEnum(Errno.badf));
    try std.testing.expectEqual(@as(u16, 28), @intFromEnum(Errno.inval));
    try std.testing.expectEqual(@as(u16, 29), @intFromEnum(Errno.io));
    try std.testing.expectEqual(@as(u16, 44), @intFromEnum(Errno.noent));
    try std.testing.expectEqual(@as(u16, 48), @intFromEnum(Errno.nomem));
    try std.testing.expectEqual(@as(u16, 52), @intFromEnum(Errno.nosys));
    try std.testing.expectEqual(@as(u16, 63), @intFromEnum(Errno.perm));
    try std.testing.expectEqual(@as(u16, 76), @intFromEnum(Errno.notcapable));
}

test "IoVec slice" {
    var data = "hello".*;
    const iov = IoVec{ .buf = &data, .len = 5 };
    const s = iov.slice();
    try std.testing.expectEqualSlices(u8, "hello", s);
}
