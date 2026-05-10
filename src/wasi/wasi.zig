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

// ── WASI Filetype ───────────────────────────────────────────────────────

/// `filetype` per the WASI preview1 spec — used in `fdstat.fs_filetype`
/// and `filestat.filetype`. Values match the witx layout exactly.
pub const Filetype = enum(u8) {
    unknown = 0,
    block_device = 1,
    character_device = 2,
    directory = 3,
    regular_file = 4,
    socket_dgram = 5,
    socket_stream = 6,
    symbolic_link = 7,
};

// ── WASI fdflags (bitset, u16) ──────────────────────────────────────────

pub const FDFLAGS_APPEND: u16 = 0x0001;
pub const FDFLAGS_DSYNC: u16 = 0x0002;
pub const FDFLAGS_NONBLOCK: u16 = 0x0004;
pub const FDFLAGS_RSYNC: u16 = 0x0008;
pub const FDFLAGS_SYNC: u16 = 0x0010;
pub const FDFLAGS_ALL: u16 =
    FDFLAGS_APPEND | FDFLAGS_DSYNC | FDFLAGS_NONBLOCK | FDFLAGS_RSYNC | FDFLAGS_SYNC;

// ── WASI poll_oneoff types (#420 phase 7) ───────────────────────────────

pub const CLOCKID_REALTIME: u32 = 0;
pub const CLOCKID_MONOTONIC: u32 = 1;
pub const CLOCKID_PROCESS_CPUTIME_ID: u32 = 2;
pub const CLOCKID_THREAD_CPUTIME_ID: u32 = 3;

pub const EVENTTYPE_CLOCK: u8 = 0;
pub const EVENTTYPE_FD_READ: u8 = 1;
pub const EVENTTYPE_FD_WRITE: u8 = 2;

pub const SUBSCRIPTION_CLOCK_ABSTIME: u16 = 0x0001;

pub const EVENT_FD_READWRITE_HANGUP: u16 = 0x0001;

/// Per-witx ABI: subscription is 48 bytes, event is 32 bytes, both
/// align 8. Used by ctxPollOneoffCore for guest-memory bounds checks.
pub const SUBSCRIPTION_SIZE: usize = 48;
pub const EVENT_SIZE: usize = 32;

// ── WASI rights (bitset, u64) — only the bits we currently consult ──────
// Full taxonomy in the preview1 spec; extend on demand.

pub const RIGHTS_FD_DATASYNC: u64 = 0x0000_0000_0000_0001;
pub const RIGHTS_FD_READ: u64 = 0x0000_0000_0000_0002;
pub const RIGHTS_FD_SEEK: u64 = 0x0000_0000_0000_0004;
pub const RIGHTS_FD_FDSTAT_SET_FLAGS: u64 = 0x0000_0000_0000_0008;
pub const RIGHTS_FD_SYNC: u64 = 0x0000_0000_0000_0010;
pub const RIGHTS_FD_TELL: u64 = 0x0000_0000_0000_0020;
pub const RIGHTS_FD_WRITE: u64 = 0x0000_0000_0000_0040;
pub const RIGHTS_FD_ADVISE: u64 = 0x0000_0000_0000_0080;
pub const RIGHTS_FD_ALLOCATE: u64 = 0x0000_0000_0000_0100;
pub const RIGHTS_PATH_CREATE_DIRECTORY: u64 = 0x0000_0000_0000_0200;
pub const RIGHTS_PATH_CREATE_FILE: u64 = 0x0000_0000_0000_0400;
pub const RIGHTS_PATH_LINK_SOURCE: u64 = 0x0000_0000_0000_0800;
pub const RIGHTS_PATH_LINK_TARGET: u64 = 0x0000_0000_0000_1000;
pub const RIGHTS_PATH_OPEN: u64 = 0x0000_0000_0000_2000;
pub const RIGHTS_FD_READDIR: u64 = 0x0000_0000_0000_4000;
pub const RIGHTS_PATH_READLINK: u64 = 0x0000_0000_0000_8000;
pub const RIGHTS_PATH_RENAME_SOURCE: u64 = 0x0000_0000_0001_0000;
pub const RIGHTS_PATH_RENAME_TARGET: u64 = 0x0000_0000_0002_0000;
pub const RIGHTS_PATH_FILESTAT_GET: u64 = 0x0000_0000_0004_0000;
pub const RIGHTS_PATH_FILESTAT_SET_SIZE: u64 = 0x0000_0000_0008_0000;
pub const RIGHTS_PATH_FILESTAT_SET_TIMES: u64 = 0x0000_0000_0010_0000;
pub const RIGHTS_FD_FILESTAT_GET: u64 = 0x0000_0000_0020_0000;
pub const RIGHTS_FD_FILESTAT_SET_SIZE: u64 = 0x0000_0000_0040_0000;
pub const RIGHTS_FD_FILESTAT_SET_TIMES: u64 = 0x0000_0000_0080_0000;
pub const RIGHTS_PATH_SYMLINK: u64 = 0x0000_0000_0100_0000;
pub const RIGHTS_PATH_REMOVE_DIRECTORY: u64 = 0x0000_0000_0200_0000;
pub const RIGHTS_PATH_UNLINK_FILE: u64 = 0x0000_0000_0400_0000;
pub const RIGHTS_POLL_FD_READWRITE: u64 = 0x0000_0000_0800_0000;
pub const RIGHTS_SOCK_SHUTDOWN: u64 = 0x0000_0000_1000_0000;
pub const RIGHTS_SOCK_ACCEPT: u64 = 0x0000_0000_2000_0000;

/// Rights that apply to a directory file descriptor. wasi-libc and
/// wasmtime mask `path_open` results that yield a directory against
/// this set so e.g. `FD_READ`/`FD_WRITE`/`FD_SEEK` never appear on a
/// dir's `fs_rights_base`.
pub const DIRECTORY_BASE_RIGHTS: u64 =
    RIGHTS_FD_FDSTAT_SET_FLAGS |
    RIGHTS_FD_SYNC |
    RIGHTS_PATH_CREATE_DIRECTORY |
    RIGHTS_PATH_CREATE_FILE |
    RIGHTS_PATH_LINK_SOURCE |
    RIGHTS_PATH_LINK_TARGET |
    RIGHTS_PATH_OPEN |
    RIGHTS_FD_READDIR |
    RIGHTS_PATH_READLINK |
    RIGHTS_PATH_RENAME_SOURCE |
    RIGHTS_PATH_RENAME_TARGET |
    RIGHTS_PATH_FILESTAT_GET |
    RIGHTS_PATH_FILESTAT_SET_SIZE |
    RIGHTS_PATH_FILESTAT_SET_TIMES |
    RIGHTS_FD_FILESTAT_GET |
    RIGHTS_FD_FILESTAT_SET_TIMES |
    RIGHTS_PATH_SYMLINK |
    RIGHTS_PATH_REMOVE_DIRECTORY |
    RIGHTS_PATH_UNLINK_FILE |
    RIGHTS_POLL_FD_READWRITE;

/// Rights that apply to a regular file descriptor (everything that's
/// not directory-only). Used as the inheriting-rights mask for files
/// opened beneath a directory.
pub const FILE_BASE_RIGHTS: u64 =
    RIGHTS_FD_DATASYNC |
    RIGHTS_FD_READ |
    RIGHTS_FD_SEEK |
    RIGHTS_FD_FDSTAT_SET_FLAGS |
    RIGHTS_FD_SYNC |
    RIGHTS_FD_TELL |
    RIGHTS_FD_WRITE |
    RIGHTS_FD_ADVISE |
    RIGHTS_FD_ALLOCATE |
    RIGHTS_FD_FILESTAT_GET |
    RIGHTS_FD_FILESTAT_SET_SIZE |
    RIGHTS_FD_FILESTAT_SET_TIMES |
    RIGHTS_POLL_FD_READWRITE;

/// Rights a directory passes through to children opened beneath it via
/// `path_open`. Same as base + the file rights so opening a regular
/// file inside still gets read/write/seek capabilities.
pub const DIRECTORY_INHERITING_RIGHTS: u64 = DIRECTORY_BASE_RIGHTS | FILE_BASE_RIGHTS;

// ── WASI fstflags (bitset, u16) for `*_filestat_set_times` ──────────────

pub const FSTFLAGS_ATIM: u16 = 0x0001;
pub const FSTFLAGS_ATIM_NOW: u16 = 0x0002;
pub const FSTFLAGS_MTIM: u16 = 0x0004;
pub const FSTFLAGS_MTIM_NOW: u16 = 0x0008;

// ── WASI advice (u8 enum) for `fd_advise` ──────────────────────────────

pub const Advice = enum(u8) {
    normal = 0,
    sequential = 1,
    random = 2,
    willneed = 3,
    dontneed = 4,
    noreuse = 5,
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
    /// Owned host directory handle for `directory` entries (preopens and the
    /// results of `path_open` on directories). `null` for non-directory
    /// entries. WasiCtx owns this and closes it on `fd_close` / `deinit`.
    host_dir: ?std.Io.Dir = null,
    /// Tracks the byte offset for `regular_file` entries so `fd_seek` /
    /// `fd_read` / `fd_write` can advance the position without relying on
    /// host-side seek (which the std.Io.File reader/writer wraps).
    pos: u64 = 0,
    /// Cached preview1 fdflags (APPEND/DSYNC/NONBLOCK/RSYNC/SYNC). Updated
    /// by `fd_fdstat_set_flags` and surfaced by `fd_fdstat_get`. The
    /// authoritative state still lives on the host fd via `fcntl` for
    /// flags that map to host O_*, but we cache here so `fd_fdstat_get`
    /// reads cheaply and so flags that don't have an O_ analogue (none
    /// in preview1) are still preserved.
    fdflags: u16 = 0,
    /// Preview1 rights cap. Defaults to "all bits set" so existing
    /// callers see unconstrained rights. `fd_fdstat_set_rights` narrows
    /// these (widening returns `notcapable`).
    rights_base: u64 = 0xFFFF_FFFF_FFFF_FFFF,
    rights_inheriting: u64 = 0xFFFF_FFFF_FFFF_FFFF,

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
    preopens: std.ArrayListUnmanaged(Preopen) = .empty,
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
        // Close any host Dir / file handles owned by the table before tearing
        // it down so the OS doesn't see a leak in long-lived embedders.
        var it = self.fd_table.entries.iterator();
        while (it.next()) |kv| {
            const entry = kv.value_ptr.*;
            if (entry.host_dir) |dir| {
                var d = dir;
                d.close(self.io);
            }
            if (entry.host_fd) |host_fd| {
                if (entry.kind == .regular_file) {
                    const file = File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
                    file.close(self.io);
                }
            }
        }
        // Free the duplicated guest-name strings stored on each Preopen.
        for (self.preopens.items) |p| self.allocator.free(p.path);
        self.preopens.deinit(self.allocator);
        self.fd_table.deinit();
        self.allocator.destroy(self);
    }

    pub fn setArgs(self: *WasiCtx, args: []const []const u8) void {
        self.args = args;
    }

    pub fn setEnv(self: *WasiCtx, env: []const []const u8) void {
        self.env_vars = env;
    }

    /// Register an already-opened host directory under `guest_name` as a
    /// preopen. Allocates a fresh fd ≥ 3 and takes ownership of `dir`
    /// (closed on `WasiCtx.deinit`). Returns the assigned fd.
    pub fn addPreopen(self: *WasiCtx, guest_name: []const u8, dir: std.Io.Dir) !u32 {
        const fd = self.fd_table.allocateFd();
        const owned_name = try self.allocator.dupe(u8, guest_name);
        errdefer self.allocator.free(owned_name);
        // Preopens are directories: mask the default all-ones rights down
        // to the directory rights set so wasi-libc / wasi-tests inheritance
        // stays consistent (e.g. `path_open(preopen, OFLAGS_DIRECTORY,
        // base, ...)` doesn't carry FD_WRITE/FD_SEEK that would conflict
        // with the new fd's directory-only nature).
        try self.fd_table.insert(fd, .{
            .kind = .directory,
            .host_dir = dir,
            .rights_base = DIRECTORY_BASE_RIGHTS,
            .rights_inheriting = DIRECTORY_INHERITING_RIGHTS,
        });
        try self.preopens.append(self.allocator, .{ .fd = fd, .path = owned_name });
        return fd;
    }

    /// Open `host_path` on the host and register it as a preopen exposed to
    /// the guest under `guest_name`. Used by the CLI's `--map-dir` flag.
    pub fn openMappedDir(self: *WasiCtx, host_path: []const u8, guest_name: []const u8) !u32 {
        const dir = try std.Io.Dir.cwd().openDir(self.io, host_path, .{ .iterate = true });
        errdefer {
            var d = dir;
            d.close(self.io);
        }
        return try self.addPreopen(guest_name, dir);
    }

    /// Look up a preopen by its assigned fd; returns the guest name or null.
    pub fn preopenName(self: *const WasiCtx, fd: u32) ?[]const u8 {
        for (self.preopens.items) |p| {
            if (p.fd == fd) return p.path;
        }
        return null;
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

test "fd_write to a regular file writes all iovs" {
    // NOTE: deliberately does NOT exercise stdout/stderr (fd 1/2). When run
    // under `zig build test` the test binary's stdout is a pipe carrying the
    // Zig test event protocol, so writing raw bytes there desynchronises the
    // orchestrator and hangs CI. The real stdio path is covered end-to-end by
    // the wasi-testsuite suite (`zig build wasi-testsuite`).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile(testing_io, "fd_write.bin", .{ .read = true });
    defer file.close(testing_io);

    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const fd = ctx.fd_table.allocateFd();
    try ctx.fd_table.insert(fd, .{
        .kind = .regular_file,
        .host_fd = file.handle,
    });

    var part1 = "hello ".*;
    var part2 = "world".*;
    const iovs = [_]IoVec{
        .{ .buf = &part1, .len = @intCast(part1.len) },
        .{ .buf = &part2, .len = @intCast(part2.len) },
    };
    const result = try ctx.fd_write(fd, &iovs);
    try std.testing.expectEqual(@as(u32, 11), result.nwritten);

    // Drop the entry so deinit doesn't re-close the host fd we still own.
    ctx.fd_table.remove(fd);

    var buf: [16]u8 = undefined;
    const n = try file.readPositionalAll(testing_io, &buf, 0);
    try std.testing.expectEqualStrings("hello world", buf[0..n]);
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
