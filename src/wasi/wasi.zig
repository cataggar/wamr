//! WASI Preview1 Implementation
//!
//! Pure Zig implementation of the WASI preview1 syscall interface,
//! replacing libuv + uvwasi + Wasmtime SSP with `std.Io` (file I/O,
//! clocks, secure random) from the Zig standard library.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");
const stable_resource = @import("../shared/stable_resource.zig");
const execution_context = @import("../runtime/common/execution_context.zig");
const termination = @import("../runtime/common/termination.zig");
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

// ── WASI Signal numbers ─────────────────────────────────────────────────

/// WASI preview1 `signal` witx enum. The numeric ABI is *not* identical
/// to POSIX `SIG*` on Linux: WASI compresses the range and shifts the
/// values from `chld` onward to keep the enum dense. Translation to the
/// host POSIX number lives in `wasiSignalToPosix` below.
pub const Signal = enum(u8) {
    none = 0,
    hup = 1,
    int = 2,
    quit = 3,
    ill = 4,
    trap = 5,
    abrt = 6,
    bus = 7,
    fpe = 8,
    kill = 9,
    usr1 = 10,
    segv = 11,
    usr2 = 12,
    pipe = 13,
    alrm = 14,
    term = 15,
    chld = 16,
    cont = 17,
    stop = 18,
    tstp = 19,
    ttin = 20,
    ttou = 21,
    urg = 22,
    xcpu = 23,
    xfsz = 24,
    vtalrm = 25,
    prof = 26,
    winch = 27,
    poll = 28,
    pwr = 29,
    sys = 30,
};

/// Translate a WASI signal value to the host POSIX `SIG*` integer.
/// Returns null for `Signal.none` (`0`) and for values outside the witx
/// `signal` enum (≥ 31). The mapping is identity for 1..15 (HUP..TERM)
/// then shifts forward by one on Linux from CHLD onward.
pub fn wasiSignalToPosix(sig: u8) ?u8 {
    return switch (sig) {
        // 0 = `none`: not deliverable.
        0 => null,
        // 1..15 — HUP..TERM — match POSIX numbering exactly.
        1...15 => sig,
        // 16..30 — shift by one to match Linux `SIG*` numbering.
        // 16 CHLD→17, 17 CONT→18, 18 STOP→19, 19 TSTP→20, 20 TTIN→21,
        // 21 TTOU→22, 22 URG→23, 23 XCPU→24, 24 XFSZ→25, 25 VTALRM→26,
        // 26 PROF→27, 27 WINCH→28, 28 POLL→29 (SIGIO), 29 PWR→30,
        // 30 SYS→31.
        16...30 => sig + 1,
        else => null,
    };
}

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

// ── sock_shutdown sdflags ───────────────────────────────────────────────
// Witx encodes the direction as a bitfield; both bits set ≡ SHUT_RDWR.
pub const SDFLAGS_RD: u16 = 0x0001;
pub const SDFLAGS_WR: u16 = 0x0002;

// ── sock_recv riflags ───────────────────────────────────────────────────
// Witx `riflags` is a bitset. Only the two below are defined; any other
// bit must be rejected with `EINVAL`. `WAITALL` maps to `MSG_WAITALL`,
// `PEEK` to `MSG_PEEK`.
pub const RIFLAGS_RECV_PEEK: u16 = 0x0001;
pub const RIFLAGS_RECV_WAITALL: u16 = 0x0002;
pub const RIFLAGS_ALL: u16 = RIFLAGS_RECV_PEEK | RIFLAGS_RECV_WAITALL;

// ── sock_recv roflags ───────────────────────────────────────────────────
// Currently only `DATA_TRUNCATED` is defined (set when the host kernel
// reports `MSG_TRUNC`). wamr's `sock_recv` does not surface this yet —
// the bit is defined for forward compatibility and for the guest-side
// header layout.
pub const ROFLAGS_RECV_DATA_TRUNCATED: u16 = 0x0001;

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

/// Rights granted to a connected stream socket — what `sock_accept`
/// installs on the new fd. wasi-libc expects an accepted socket to
/// support `read`/`write`/`poll`/`shutdown`; it does not expect the
/// child fd to be able to `accept` again.
pub const SOCKET_BASE_RIGHTS: u64 =
    RIGHTS_FD_READ |
    RIGHTS_FD_WRITE |
    RIGHTS_FD_FDSTAT_SET_FLAGS |
    RIGHTS_POLL_FD_READWRITE |
    RIGHTS_SOCK_SHUTDOWN;

/// Rights granted to a listening socket preopen fd. It accepts new
/// connections and is poll-able; it cannot itself be read from or
/// written to, mirroring wasi-libc's expectation for a passive
/// `socket(2)` + `listen(2)` fd.
pub const SOCKET_LISTEN_RIGHTS: u64 =
    RIGHTS_SOCK_ACCEPT |
    RIGHTS_POLL_FD_READWRITE |
    RIGHTS_SOCK_SHUTDOWN |
    RIGHTS_FD_FDSTAT_SET_FLAGS;

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

// ── File Descriptor Table ───────────────────────────────────────────────

pub const FdKind = enum {
    stdin,
    stdout,
    stderr,
    regular_file,
    directory,
    socket,
};

pub const FdEntrySnapshot = struct {
    kind: FdKind,
    host_fd: ?std.posix.fd_t,
    host_dir: ?std.Io.Dir,
    pos: u64,
    fdflags: u16,
    rights_base: u64,
    rights_inheriting: u64,
};

pub fn FdEntryFor(comptime enabled: bool) type {
    const EntryMutex = stable_resource.ConditionalMutexFor(
        enabled,
        stable_resource.LockRank.resource_node,
    );
    const Kind = FdKind;

    return struct {
        const Self = @This();

        mutex: EntryMutex = .init,
        kind: Kind,
        host_fd: ?std.posix.fd_t = null,
        /// Owned host directory handle for `directory` entries (preopens and
        /// `path_open` results). The descriptor table closes it exactly once.
        host_dir: ?std.Io.Dir = null,
        /// Cached byte offset for regular files.
        pos: u64 = 0,
        /// Cached Preview-1 fdflags.
        fdflags: u16 = 0,
        /// Preview-1 rights caps. They may only be narrowed after publication.
        rights_base: u64 = 0xFFFF_FFFF_FFFF_FFFF,
        rights_inheriting: u64 = 0xFFFF_FFFF_FFFF_FFFF,

        pub const FdKind = Kind;

        inline fn snapshot(self: *Self) FdEntrySnapshot {
            self.mutex.lock();
            defer self.mutex.unlock();
            return .{
                .kind = self.kind,
                .host_fd = self.host_fd,
                .host_dir = self.host_dir,
                .pos = self.pos,
                .fdflags = self.fdflags,
                .rights_base = self.rights_base,
                .rights_inheriting = self.rights_inheriting,
            };
        }
    };
}

pub const FdEntry = FdEntryFor(config.lib_wasi_threads);

/// Query the shared host-file cursor without changing it.
pub fn hostFilePosition(host_fd: std.posix.fd_t) ?u64 {
    if (comptime builtin.os.tag == .windows) {
        const windows = std.os.windows;
        var iosb: windows.IO_STATUS_BLOCK = undefined;
        var info: windows.FILE.POSITION_INFORMATION = undefined;
        if (windows.ntdll.NtQueryInformationFile(
            host_fd,
            &iosb,
            &info,
            @sizeOf(windows.FILE.POSITION_INFORMATION),
            .Position,
        ) != .SUCCESS) return null;
        if (info.CurrentByteOffset < 0) return null;
        return @intCast(info.CurrentByteOffset);
    }

    if (comptime std.posix.SEEK == void) return null;
    const result = std.posix.system.lseek(host_fd, 0, std.posix.SEEK.CUR);
    if (std.posix.errno(result) != .SUCCESS) return null;
    return std.math.cast(u64, result);
}

const WindowsSeek = struct {
    extern "kernel32" fn SetFilePointerEx(
        file: std.os.windows.HANDLE,
        distance: std.os.windows.LARGE_INTEGER,
        new_position: *std.os.windows.LARGE_INTEGER,
        move_method: std.os.windows.DWORD,
    ) callconv(.winapi) std.os.windows.BOOL;
};

pub fn seekHostFile(io: Io, file: File, offset: i64, whence: Whence) !u64 {
    if (comptime builtin.os.tag == .windows) {
        var position: std.os.windows.LARGE_INTEGER = undefined;
        const move_method: std.os.windows.DWORD = switch (whence) {
            .set => 0,
            .cur => 1,
            .end => 2,
        };
        if (!WindowsSeek.SetFilePointerEx(file.handle, offset, &position, move_method).toBool())
            return error.Unseekable;
        if (position < 0) return error.InvalidOffset;
        return @intCast(position);
    }

    if (comptime builtin.os.tag == .linux) {
        const origin: usize = switch (whence) {
            .set => 0,
            .cur => 1,
            .end => 2,
        };
        const result = std.os.linux.lseek(file.handle, offset, origin);
        if (std.os.linux.errno(result) != .SUCCESS) return error.Unseekable;
        return @intCast(result);
    }

    if (comptime builtin.os.tag != .wasi and std.posix.SEEK != void) {
        const origin: std.c.whence_t = switch (whence) {
            .set => 0,
            .cur => 1,
            .end => 2,
        };
        const result = std.c.lseek(file.handle, @intCast(offset), origin);
        if (result < 0) return error.Unseekable;
        return @intCast(result);
    }

    switch (whence) {
        .set => {
            if (offset < 0) return error.InvalidOffset;
            try io.vtable.fileSeekTo(io.userdata, file, @intCast(offset));
        },
        .cur => try io.vtable.fileSeekBy(io.userdata, file, offset),
        .end => {
            const stat = try file.stat(io);
            const position = @as(i64, @intCast(stat.size)) + offset;
            if (position < 0) return error.InvalidOffset;
            try io.vtable.fileSeekTo(io.userdata, file, @intCast(position));
        },
    }
    return hostFilePosition(file.handle) orelse error.Unseekable;
}

pub fn FdTableFor(comptime enabled: bool) type {
    const Entry = FdEntryFor(enabled);
    const DirectoryMutex = stable_resource.ConditionalMutexFor(
        enabled,
        stable_resource.LockRank.resource_registry,
    );
    const DestroyContext = struct {
        allocator: std.mem.Allocator,
        io: Io,
    };
    const Destroyer = struct {
        fn destroy(context: DestroyContext, entry: *Entry) void {
            stable_resource.assertNoLocksHeldFor(enabled);
            if (entry.host_dir) |dir| {
                var owned_dir = dir;
                owned_dir.close(context.io);
                entry.host_dir = null;
            }
            if (entry.host_fd) |host_fd| {
                if (entry.kind == .regular_file or entry.kind == .socket) {
                    const file = File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
                    file.close(context.io);
                }
                entry.host_fd = null;
            }
        }
    };

    const ResourceTable = stable_resource.StableHandleTableFor(
        enabled,
        Entry,
        DestroyContext,
        Destroyer.destroy,
        64,
    );
    const ThreadedSlot = struct {
        handle: stable_resource.Handle,
        kind: FdKind,
    };
    const Slot = if (enabled) ThreadedSlot else Entry;

    return struct {
        const Self = @This();

        pub const Lease = struct {
            storage: if (enabled) ResourceTable.Lease else ?*Entry,

            inline fn value(self: *Lease) *Entry {
                if (comptime enabled) return self.storage.value();
                return self.storage.?;
            }

            pub inline fn snapshot(self: *Lease) FdEntrySnapshot {
                return self.value().snapshot();
            }

            pub fn isClosing(self: *const Lease) bool {
                if (comptime enabled) return self.storage.isClosing();
                return false;
            }

            pub fn setPosition(self: *Lease, position: u64) void {
                const entry = self.value();
                entry.mutex.lock();
                defer entry.mutex.unlock();
                entry.pos = position;
            }

            pub fn advancePosition(self: *Lease, amount: usize) void {
                const entry = self.value();
                entry.mutex.lock();
                defer entry.mutex.unlock();
                entry.pos +|= @as(u64, @intCast(amount));
            }

            pub fn setFdFlags(self: *Lease, flags: u16) void {
                const entry = self.value();
                entry.mutex.lock();
                defer entry.mutex.unlock();
                entry.fdflags = flags;
            }

            pub fn narrowRights(self: *Lease, base: u64, inheriting: u64) bool {
                const entry = self.value();
                entry.mutex.lock();
                defer entry.mutex.unlock();
                if ((base & ~entry.rights_base) != 0) return false;
                if ((inheriting & ~entry.rights_inheriting) != 0) return false;
                entry.rights_base = base;
                entry.rights_inheriting = inheriting;
                return true;
            }

            /// Transfer a raw file handle back to the caller. Intended for
            /// embedder ownership handoff and tests, not ordinary guest close.
            pub fn detachHostFd(self: *Lease) ?std.posix.fd_t {
                const entry = self.value();
                entry.mutex.lock();
                defer entry.mutex.unlock();
                const host_fd = entry.host_fd;
                entry.host_fd = null;
                return host_fd;
            }

            /// Transfer an owned directory handle back to the caller.
            pub fn detachHostDir(self: *Lease) ?std.Io.Dir {
                const entry = self.value();
                entry.mutex.lock();
                defer entry.mutex.unlock();
                const host_dir = entry.host_dir;
                entry.host_dir = null;
                return host_dir;
            }

            pub inline fn release(self: *Lease) void {
                if (comptime enabled) {
                    self.storage.release();
                } else {
                    self.storage = null;
                }
            }

            pub fn deinit(self: *Lease) void {
                self.release();
            }
        };

        allocator: std.mem.Allocator,
        io: Io,
        entries: std.AutoHashMap(u32, Slot),
        preopens: std.AutoHashMap(u32, []u8),
        next_fd: u32 = 3,
        directory_mutex: DirectoryMutex = .init,
        resources: if (enabled) ResourceTable else void,

        pub fn init(allocator: std.mem.Allocator, io: Io) !Self {
            return .{
                .allocator = allocator,
                .io = io,
                .entries = std.AutoHashMap(u32, Slot).init(allocator),
                .preopens = std.AutoHashMap(u32, []u8).init(allocator),
                .resources = if (enabled)
                    try ResourceTable.init(allocator, .{ .allocator = allocator, .io = io })
                else {},
            };
        }

        pub fn deinit(self: *Self) !void {
            if (comptime enabled) {
                self.resources.shutdown();
                if (!self.resources.isQuiescent()) return error.LeasesOutstanding;
            } else {
                var entries = self.entries.valueIterator();
                while (entries.next()) |entry| Destroyer.destroy(
                    .{ .allocator = self.allocator, .io = self.io },
                    entry,
                );
            }

            var preopens = self.preopens.valueIterator();
            while (preopens.next()) |path| self.allocator.free(path.*);
            self.preopens.deinit();
            self.entries.deinit();
            if (comptime enabled) try self.resources.deinit();
        }

        /// Insert at an explicit guest fd, taking ownership only on success.
        /// Replacing a descriptor preserves any preopen label attached to the
        /// numeric target fd and destroys the old descriptor after unlocking.
        pub fn insert(self: *Self, fd: u32, entry: Entry) !void {
            if (comptime enabled) {
                const handle = try self.resources.publish(entry);
                self.directory_mutex.lock();
                const old = self.entries.get(fd);
                self.entries.put(fd, .{ .handle = handle, .kind = entry.kind }) catch |err| {
                    self.directory_mutex.unlock();
                    std.debug.assert(self.resources.withdraw(handle) != null);
                    return err;
                };
                self.noteExplicitFd(fd);
                self.directory_mutex.unlock();
                if (old) |old_entry| std.debug.assert(self.resources.remove(old_entry.handle));
            } else {
                self.directory_mutex.lock();
                const old = self.entries.get(fd);
                self.entries.put(fd, entry) catch |err| {
                    self.directory_mutex.unlock();
                    return err;
                };
                self.noteExplicitFd(fd);
                self.directory_mutex.unlock();
                if (old) |old_entry| {
                    var owned = old_entry;
                    Destroyer.destroy(.{ .allocator = self.allocator, .io = self.io }, &owned);
                }
            }
        }

        /// Allocate and publish a descriptor as one operation.
        pub fn create(self: *Self, entry: Entry) !u32 {
            const handle = if (comptime enabled)
                try self.resources.publish(entry)
            else {};
            self.directory_mutex.lock();
            const fd = self.nextAvailableFdLocked() catch |err| {
                self.directory_mutex.unlock();
                if (comptime enabled) std.debug.assert(self.resources.withdraw(handle) != null);
                return err;
            };

            if (comptime enabled) {
                self.entries.put(fd, .{ .handle = handle, .kind = entry.kind }) catch |err| {
                    self.directory_mutex.unlock();
                    std.debug.assert(self.resources.withdraw(handle) != null);
                    return err;
                };
            } else {
                self.entries.put(fd, entry) catch |err| {
                    self.directory_mutex.unlock();
                    return err;
                };
            }
            self.advanceNextFd(fd);
            self.directory_mutex.unlock();
            return fd;
        }

        /// Publish a directory descriptor and its owned guest preopen name
        /// atomically. Both values remain caller-owned on failure.
        pub fn createPreopen(self: *Self, entry: Entry, owned_name: []u8) !u32 {
            const handle = if (comptime enabled)
                try self.resources.publish(entry)
            else {};
            self.directory_mutex.lock();
            const fd = self.nextAvailableFdLocked() catch |err| {
                self.directory_mutex.unlock();
                if (comptime enabled) std.debug.assert(self.resources.withdraw(handle) != null);
                return err;
            };
            self.entries.ensureUnusedCapacity(1) catch |err| {
                self.directory_mutex.unlock();
                if (comptime enabled) std.debug.assert(self.resources.withdraw(handle) != null);
                return err;
            };
            self.preopens.ensureUnusedCapacity(1) catch |err| {
                self.directory_mutex.unlock();
                if (comptime enabled) std.debug.assert(self.resources.withdraw(handle) != null);
                return err;
            };

            if (comptime enabled) {
                self.entries.putAssumeCapacity(fd, .{ .handle = handle, .kind = entry.kind });
            } else {
                self.entries.putAssumeCapacity(fd, entry);
            }
            self.preopens.putAssumeCapacity(fd, owned_name);
            self.advanceNextFd(fd);
            self.directory_mutex.unlock();
            return fd;
        }

        pub inline fn acquire(self: *Self, fd: u32) ?Lease {
            self.directory_mutex.lock();
            defer self.directory_mutex.unlock();
            if (comptime enabled) {
                const slot = self.entries.get(fd) orelse return null;
                const lease = self.resources.acquire(slot.handle) orelse return null;
                return .{ .storage = lease };
            }
            const entry = self.entries.getPtr(fd) orelse return null;
            return .{ .storage = entry };
        }

        pub fn contains(self: *Self, fd: u32) bool {
            self.directory_mutex.lock();
            defer self.directory_mutex.unlock();
            return self.entries.contains(fd);
        }

        /// Return a metadata snapshot. The snapshot does not keep host
        /// handles alive; I/O callers must hold an explicit lease instead.
        pub fn snapshot(self: *Self, fd: u32) ?FdEntrySnapshot {
            var lease = self.acquire(fd) orelse return null;
            defer lease.release();
            return lease.snapshot();
        }

        /// Remove a descriptor and any preopen label. Destructors and frees
        /// always run after the directory lock is released.
        pub fn remove(self: *Self, fd: u32) bool {
            self.directory_mutex.lock();
            const removed = self.entries.fetchRemove(fd);
            const preopen = self.preopens.fetchRemove(fd);
            self.directory_mutex.unlock();

            if (preopen) |kv| self.allocator.free(kv.value);
            if (removed) |kv| {
                if (comptime enabled) {
                    std.debug.assert(self.resources.remove(kv.value.handle));
                } else {
                    var entry = kv.value;
                    Destroyer.destroy(.{ .allocator = self.allocator, .io = self.io }, &entry);
                }
                return true;
            }
            return false;
        }

        pub const CloseResult = enum {
            closed,
            badf,
            protected_stdio,
        };

        /// Atomically classify and unlink a guest-closeable descriptor so a
        /// concurrent renumber cannot make the close target a new occupant.
        pub fn closeGuest(self: *Self, fd: u32) CloseResult {
            self.directory_mutex.lock();
            const current = self.entries.get(fd) orelse {
                self.directory_mutex.unlock();
                return .badf;
            };
            const kind = if (comptime enabled) current.kind else current.kind;
            switch (kind) {
                .stdin, .stdout, .stderr => {
                    self.directory_mutex.unlock();
                    return .protected_stdio;
                },
                else => {},
            }
            const removed = self.entries.fetchRemove(fd).?;
            const preopen = self.preopens.fetchRemove(fd);
            self.directory_mutex.unlock();

            if (preopen) |kv| self.allocator.free(kv.value);
            if (comptime enabled) {
                std.debug.assert(self.resources.remove(removed.value.handle));
            } else {
                var entry = removed.value;
                Destroyer.destroy(.{ .allocator = self.allocator, .io = self.io }, &entry);
            }
            return .closed;
        }

        pub const RenumberError = error{
            BadFd,
            TargetIsStdio,
        };

        /// Move the descriptor at `from` onto the already-open `to` slot.
        /// Existing leases on either descriptor remain valid. The target's
        /// numeric preopen label is preserved; a source label is retired.
        pub fn renumber(self: *Self, from: u32, to: u32) RenumberError!void {
            self.directory_mutex.lock();
            if (from == to) {
                const present = self.entries.contains(from);
                self.directory_mutex.unlock();
                if (!present) return error.BadFd;
                return;
            }

            const source = self.entries.get(from) orelse {
                self.directory_mutex.unlock();
                return error.BadFd;
            };
            const target = self.entries.get(to) orelse {
                self.directory_mutex.unlock();
                return error.BadFd;
            };
            const source_kind = if (comptime enabled) source.kind else source.kind;
            const target_kind = if (comptime enabled) target.kind else target.kind;
            switch (target_kind) {
                .stdin, .stdout, .stderr => {
                    self.directory_mutex.unlock();
                    return error.TargetIsStdio;
                },
                else => {},
            }

            if (comptime enabled) {
                self.entries.putAssumeCapacity(to, source);
            } else {
                self.entries.putAssumeCapacity(to, source);
            }
            _ = self.entries.remove(from);
            const source_preopen = self.preopens.fetchRemove(from);
            const invalid_target_preopen = if (source_kind == .directory)
                null
            else
                self.preopens.fetchRemove(to);
            self.directory_mutex.unlock();

            if (source_preopen) |kv| self.allocator.free(kv.value);
            if (invalid_target_preopen) |kv| self.allocator.free(kv.value);
            if (comptime enabled) {
                std.debug.assert(self.resources.remove(target.handle));
            } else {
                var old_target = target;
                Destroyer.destroy(.{ .allocator = self.allocator, .io = self.io }, &old_target);
            }
        }

        pub fn preopenNameLen(self: *Self, fd: u32) ?usize {
            self.directory_mutex.lock();
            defer self.directory_mutex.unlock();
            if (!self.entries.contains(fd)) return null;
            const path = self.preopens.get(fd) orelse return null;
            return path.len;
        }

        pub fn copyPreopenName(self: *Self, fd: u32, dest: []u8) ?usize {
            self.directory_mutex.lock();
            defer self.directory_mutex.unlock();
            if (!self.entries.contains(fd)) return null;
            const path = self.preopens.get(fd) orelse return null;
            if (dest.len < path.len) return null;
            @memcpy(dest[0..path.len], path);
            return path.len;
        }

        pub fn preopenCount(self: *Self) usize {
            self.directory_mutex.lock();
            defer self.directory_mutex.unlock();
            return self.preopens.count();
        }

        pub fn leakCount(self: *Self) usize {
            if (comptime enabled) return self.resources.leakCount();
            return self.entries.count();
        }

        fn nextAvailableFdLocked(self: *Self) !u32 {
            var fd = self.next_fd;
            while (self.entries.contains(fd)) {
                if (fd == std.math.maxInt(u32)) return error.FdExhausted;
                fd += 1;
            }
            return fd;
        }

        fn noteExplicitFd(self: *Self, fd: u32) void {
            if (fd < self.next_fd or fd == std.math.maxInt(u32)) return;
            self.next_fd = fd + 1;
        }

        fn advanceNextFd(self: *Self, fd: u32) void {
            if (fd == std.math.maxInt(u32)) {
                self.next_fd = fd;
            } else {
                self.next_fd = fd + 1;
            }
        }
    };
}

pub const FdTable = FdTableFor(config.lib_wasi_threads);

// The Preview-1 process exit status is one facet of the group's first-wins
// terminal outcome; see `runtime/common/termination.zig`.

// ── WASI Context ────────────────────────────────────────────────────────

/// Shared WASI process state.
///
/// Descriptors, preopens, arguments, environment, clocks, random source, and
/// the process-wide terminal outcome are shared by every guest thread.
/// Per-thread stack, TLS, task, cancellation, and trap state lives in
/// `execution_context.ThreadExecutionContext`.
pub const WasiProcessState = struct {
    allocator: std.mem.Allocator,
    io: Io,
    refs: stable_resource.ConditionalLifetimeRefCount =
        stable_resource.ConditionalLifetimeRefCount.init(1),
    args: []const []const u8 = &.{},
    env_vars: []const []const u8 = &.{},
    fd_table: FdTable,
    /// First-wins terminal result for the whole thread group.
    termination: termination.State = .{},

    pub fn init(allocator: std.mem.Allocator, io: Io) !*WasiProcessState {
        const ctx = try allocator.create(WasiProcessState);
        errdefer allocator.destroy(ctx);
        ctx.* = .{
            .allocator = allocator,
            .io = io,
            .fd_table = try FdTable.init(allocator, io),
        };
        errdefer ctx.fd_table.deinit() catch unreachable;
        // Pre-populate stdin(0), stdout(1), stderr(2)
        try ctx.fd_table.insert(0, .{ .kind = .stdin });
        try ctx.fd_table.insert(1, .{ .kind = .stdout });
        try ctx.fd_table.insert(2, .{ .kind = .stderr });
        return ctx;
    }

    pub fn retain(self: *WasiProcessState) void {
        self.refs.retain();
    }

    pub fn deinit(self: *WasiProcessState) void {
        if (!self.refs.release()) return;
        self.fd_table.deinit() catch @panic("WasiProcessState destroyed with outstanding descriptor leases");
        self.freeStringList(self.args);
        self.freeStringList(self.env_vars);
        self.allocator.destroy(self);
    }

    pub fn processStateRef(self: *WasiProcessState) execution_context.ProcessStateRef {
        return execution_context.ProcessStateRef.init(@ptrCast(self), &process_state_ops);
    }

    pub fn referenceCount(self: *const WasiProcessState) usize {
        return self.refs.count();
    }

    fn retainOpaque(raw: *anyopaque) void {
        const self: *WasiProcessState = @ptrCast(@alignCast(raw));
        self.retain();
    }

    fn releaseOpaque(raw: *anyopaque) void {
        const self: *WasiProcessState = @ptrCast(@alignCast(raw));
        self.deinit();
    }

    const process_state_ops = execution_context.ProcessStateOps{
        .retain = retainOpaque,
        .release = releaseOpaque,
    };

    fn duplicateStringList(
        self: *WasiProcessState,
        values: []const []const u8,
    ) ![]const []const u8 {
        if (values.len == 0) return &.{};
        const owned = try self.allocator.alloc([]const u8, values.len);
        var initialized: usize = 0;
        errdefer {
            for (owned[0..initialized]) |value| self.allocator.free(value);
            self.allocator.free(owned);
        }
        for (values, 0..) |value, i| {
            owned[i] = try self.allocator.dupe(u8, value);
            initialized += 1;
        }
        return owned;
    }

    fn freeStringList(self: *WasiProcessState, values: []const []const u8) void {
        if (values.len == 0) return;
        for (values) |value| self.allocator.free(value);
        self.allocator.free(values);
    }

    pub fn setArgs(self: *WasiProcessState, args: []const []const u8) !void {
        // Startup-only owned configuration; freeze before sharing.
        std.debug.assert(self.refs.count() == 1);
        const replacement = try self.duplicateStringList(args);
        self.freeStringList(self.args);
        self.args = replacement;
    }

    pub fn setEnv(self: *WasiProcessState, env: []const []const u8) !void {
        // Startup-only owned configuration; freeze before sharing.
        std.debug.assert(self.refs.count() == 1);
        const replacement = try self.duplicateStringList(env);
        self.freeStringList(self.env_vars);
        self.env_vars = replacement;
    }

    /// Register an already-opened host directory under `guest_name` as a
    /// preopen. Allocates a fresh fd ≥ 3 and takes ownership of `dir` only
    /// on success (closed on `WasiCtx.deinit`); the caller retains ownership
    /// on error. Returns the assigned fd.
    pub fn addPreopen(self: *WasiProcessState, guest_name: []const u8, dir: std.Io.Dir) !u32 {
        std.debug.assert(self.refs.count() == 1);
        const owned_name = try self.allocator.dupe(u8, guest_name);
        errdefer self.allocator.free(owned_name);
        // Preopens are directories: mask the default all-ones rights down
        // to the directory rights set so wasi-libc / wasi-tests inheritance
        // stays consistent (e.g. `path_open(preopen, OFLAGS_DIRECTORY,
        // base, ...)` doesn't carry FD_WRITE/FD_SEEK that would conflict
        // with the new fd's directory-only nature).
        return try self.fd_table.createPreopen(.{
            .kind = .directory,
            .host_dir = dir,
            .rights_base = DIRECTORY_BASE_RIGHTS,
            .rights_inheriting = DIRECTORY_INHERITING_RIGHTS,
        }, owned_name);
    }

    /// Open `host_path` on the host and register it as a preopen exposed to
    /// the guest under `guest_name`. Used by the CLI's `--map-dir` flag.
    pub fn openMappedDir(self: *WasiProcessState, host_path: []const u8, guest_name: []const u8) !u32 {
        const dir = try std.Io.Dir.cwd().openDir(self.io, host_path, .{ .iterate = true });
        errdefer {
            var d = dir;
            d.close(self.io);
        }
        return try self.addPreopen(guest_name, dir);
    }

    /// Register an already-listening host TCP socket as a socket preopen.
    /// Allocates a fresh fd ≥ 3 and takes ownership of `host_fd` on success
    /// (closed on `WasiCtx.deinit`); the caller retains it on error.
    /// Preview1's `fd_prestat_*` surface only exposes
    /// directory preopens; socket preopens are discoverable by convention
    /// (wasi-libc walks fds 3.. and reports `ENOTDIR` for non-dir prestats
    /// to enumerate them). Returns the assigned fd.
    pub fn addPreopenSocket(self: *WasiProcessState, host_fd: std.posix.fd_t) !u32 {
        std.debug.assert(self.refs.count() == 1);
        return try self.fd_table.create(.{
            .kind = .socket,
            .host_fd = host_fd,
            .rights_base = SOCKET_LISTEN_RIGHTS,
            .rights_inheriting = SOCKET_BASE_RIGHTS,
        });
    }

    pub fn preopenNameLen(self: *WasiProcessState, fd: u32) ?usize {
        return self.fd_table.preopenNameLen(fd);
    }

    pub fn copyPreopenName(self: *WasiProcessState, fd: u32, dest: []u8) ?usize {
        return self.fd_table.copyPreopenName(fd, dest);
    }

    pub fn getExitCode(self: *const WasiProcessState) ?u32 {
        return self.termination.exitCode();
    }

    /// The group's first-wins terminal outcome, or null while it is running.
    pub fn terminalOutcome(self: *const WasiProcessState) ?termination.Outcome {
        return self.termination.outcome();
    }

    /// True once any thread claimed the terminal outcome. Blocking host
    /// operations poll this to unwind instead of waiting for completion.
    pub fn isTerminating(self: *const WasiProcessState) bool {
        return self.termination.isTerminating();
    }

    // ── args ────────────────────────────────────────────────────────

    pub fn args_sizes_get(self: *const WasiProcessState) struct { count: u32, buf_size: u32 } {
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

    pub fn args_get(self: *const WasiProcessState, argv_buf: []u8) []const u8 {
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

    pub fn environ_sizes_get(self: *const WasiProcessState) struct { count: u32, buf_size: u32 } {
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

    pub fn clock_time_get(self: *const WasiProcessState, clock_id: u32, precision: u64) !u64 {
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

    pub fn fd_write(self: *WasiProcessState, fd: u32, iovs: []const IoVec) !struct { nwritten: u32 } {
        var lease = self.fd_table.acquire(fd) orelse return error.BadFd;
        defer lease.release();
        const entry = lease.snapshot();
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
                    for (iovs) |iov| {
                        file.writeStreamingAll(self.io, iov.slice()) catch return error.IoError;
                        total_written += iov.len;
                    }
                    if (hostFilePosition(host_fd)) |position| {
                        lease.setPosition(position);
                    } else {
                        lease.advancePosition(total_written);
                    }
                } else {
                    return error.BadFd;
                }
            },
            else => return error.BadFd,
        }

        return .{ .nwritten = total_written };
    }

    pub fn fd_read(self: *WasiProcessState, fd: u32, iovs: []const IoVec) !struct { nread: u32 } {
        var lease = self.fd_table.acquire(fd) orelse return error.BadFd;
        defer lease.release();
        const entry = lease.snapshot();
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
                    for (iovs) |iov| {
                        const data = iov.slice();
                        const n = file.readStreaming(self.io, &.{data}) catch |err| switch (err) {
                            error.EndOfStream => 0,
                            else => return error.IoError,
                        };
                        total_read += @intCast(n);
                        if (n < data.len) break;
                    }
                    if (hostFilePosition(host_fd)) |position| {
                        lease.setPosition(position);
                    } else {
                        lease.advancePosition(total_read);
                    }
                } else {
                    return error.BadFd;
                }
            },
            else => return error.BadFd,
        }

        return .{ .nread = total_read };
    }

    pub fn fd_close(self: *WasiProcessState, fd: u32) Errno {
        return switch (self.fd_table.closeGuest(fd)) {
            .closed => .success,
            .badf, .protected_stdio => .badf,
        };
    }

    pub fn fd_seek(self: *WasiProcessState, fd: u32, offset: i64, whence: u8) !u64 {
        var lease = self.fd_table.acquire(fd) orelse return error.BadFd;
        defer lease.release();
        const entry = lease.snapshot();

        switch (entry.kind) {
            .regular_file => {
                if (entry.host_fd) |host_fd| {
                    const w = std.enums.fromInt(Whence, whence) orelse return error.InvalidWhence;
                    const file = File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
                    const position = seekHostFile(self.io, file, offset, w) catch return error.IoError;
                    lease.setPosition(position);
                    return position;
                } else {
                    return error.BadFd;
                }
            },
            .stdin, .stdout, .stderr => return error.SpPipe,
            else => return error.BadFd,
        }
    }

    // ── proc ────────────────────────────────────────────────────────

    /// Preview-1 `proc_exit`. Only the first terminating thread in the group
    /// establishes the status; a racing trap or `proc_exit` cannot overwrite
    /// it. Claiming also wakes siblings blocked in the host.
    pub fn proc_exit(self: *WasiProcessState, code: u32) void {
        _ = self.termination.claimExit(code);
    }

    // ── random ──────────────────────────────────────────────────────

    pub fn random_get(self: *const WasiProcessState, buf: []u8) void {
        self.io.random(buf);
    }
};

/// Compatibility name retained for existing embedders.
pub const WasiCtx = WasiProcessState;

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing_io = std.testing.io;

test "WasiProcessState shares descriptors preopens args and environment across thread contexts" {
    const allocator = std.testing.allocator;
    const process = try WasiProcessState.init(allocator, testing_io);
    defer process.deinit();
    var program_buf = [_]u8{ 'p', 'r', 'o', 'g', 'r', 'a', 'm' };
    var env_buf = [_]u8{ 'K', 'E', 'Y', '=', 'v', 'a', 'l', 'u', 'e' };
    const args = [_][]const u8{ &program_buf, "child" };
    const env = [_][]const u8{&env_buf};
    try process.setArgs(&args);
    try process.setEnv(&env);
    program_buf[0] = 'X';
    env_buf[0] = 'X';

    const owned_name = try allocator.dupe(u8, "/shared");
    const preopen_fd = try process.fd_table.createPreopen(
        .{
            .kind = .directory,
            .rights_base = DIRECTORY_BASE_RIGHTS,
            .rights_inheriting = DIRECTORY_INHERITING_RIGHTS,
        },
        owned_name,
    );
    try process.fd_table.insert(10, .{ .kind = .regular_file });

    var parent = execution_context.ThreadExecutionContext.init(process.processStateRef());
    defer parent.deinit();
    var child = execution_context.ThreadExecutionContext.init(parent.process_state);
    defer child.deinit();

    const parent_process = parent.process(WasiProcessState).?;
    const child_process = child.process(WasiProcessState).?;
    try std.testing.expectEqual(parent_process, child_process);
    try std.testing.expectEqual(@as(usize, 3), process.referenceCount());
    try std.testing.expectEqual(@as(u32, 2), parent_process.args_sizes_get().count);
    try std.testing.expectEqual(@as(u32, 1), child_process.environ_sizes_get().count);
    var args_buf: [32]u8 = undefined;
    const copied_args = parent_process.args_get(&args_buf);
    try std.testing.expectEqualStrings("program\x00child\x00", copied_args);
    try std.testing.expectEqualStrings("KEY=value", child_process.env_vars[0]);

    var name_buf: [32]u8 = undefined;
    const copied = child_process.copyPreopenName(preopen_fd, &name_buf).?;
    try std.testing.expectEqualStrings("/shared", name_buf[0..copied]);
    try std.testing.expectEqual(Errno.success, child_process.fd_close(10));
    try std.testing.expect(!parent_process.fd_table.contains(10));
    try std.testing.expectEqual(Errno.success, child_process.fd_close(preopen_fd));
    try std.testing.expect(parent_process.preopenNameLen(preopen_fd) == null);
}

test "WasiProcessState survives root and parent release until child completion" {
    const process = try WasiProcessState.init(std.testing.allocator, testing_io);
    try process.setArgs(&.{"owned-after-parent"});
    try process.fd_table.insert(9, .{ .kind = .regular_file, .pos = 77 });

    var parent = execution_context.ThreadExecutionContext.init(process.processStateRef());
    var child = execution_context.ThreadExecutionContext.init(parent.process_state);
    try std.testing.expectEqual(@as(usize, 3), process.referenceCount());

    process.deinit();
    parent.deinit();
    const child_process = child.process(WasiProcessState).?;
    try std.testing.expectEqual(@as(usize, 1), child_process.referenceCount());
    try std.testing.expectEqual(@as(u32, 1), child_process.args_sizes_get().count);
    var lease = child_process.fd_table.acquire(9).?;
    try std.testing.expectEqual(@as(u64, 77), lease.snapshot().pos);
    lease.release();

    child.deinit();
}

test "FdTable threaded stale lease cannot alias a reused guest fd" {
    const Table = FdTableFor(true);
    var table = try Table.init(std.testing.allocator, testing_io);
    defer table.deinit() catch unreachable;

    try table.insert(10, .{ .kind = .regular_file, .pos = 11 });
    var stale = table.acquire(10).?;
    try std.testing.expect(table.remove(10));
    try table.insert(10, .{ .kind = .regular_file, .pos = 22 });

    var current = table.acquire(10).?;
    defer current.release();
    try std.testing.expectEqual(@as(u64, 11), stale.snapshot().pos);
    try std.testing.expect(stale.isClosing());
    try std.testing.expectEqual(@as(u64, 22), current.snapshot().pos);
    stale.release();
}

test "FdTable threaded removal defers descriptor close until final lease" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const linux = std.os.linux;
    var fds: [2]i32 = undefined;
    if (linux.errno(linux.pipe2(&fds, .{})) != .SUCCESS) return error.SkipZigTest;
    defer _ = linux.close(fds[1]);

    const Table = FdTableFor(true);
    var table = try Table.init(std.testing.allocator, testing_io);
    defer table.deinit() catch unreachable;
    try table.insert(7, .{ .kind = .regular_file, .host_fd = fds[0] });
    var lease = table.acquire(7).?;

    const Remover = struct {
        fn run(target: *Table) void {
            std.debug.assert(target.remove(7));
        }
    };
    const thread = try std.Thread.spawn(.{}, Remover.run, .{&table});
    thread.join();

    try std.testing.expect(table.acquire(7) == null);
    try std.testing.expect(lease.isClosing());
    const before = linux.fcntl(fds[0], linux.F.GETFD, 0);
    try std.testing.expectEqual(linux.E.SUCCESS, linux.errno(before));
    lease.release();
    const after = linux.fcntl(fds[0], linux.F.GETFD, 0);
    try std.testing.expectEqual(linux.E.BADF, linux.errno(after));
}

test "FdTable threaded concurrent lookup and removal are race free" {
    const Table = FdTableFor(true);
    var table = try Table.init(std.testing.allocator, testing_io);
    defer table.deinit() catch unreachable;
    try table.insert(9, .{ .kind = .regular_file, .pos = 99 });

    var start = std.atomic.Value(bool).init(false);
    var stop = std.atomic.Value(bool).init(false);
    var acquired = std.atomic.Value(usize).init(0);
    const Reader = struct {
        fn run(
            target: *Table,
            start_flag: *std.atomic.Value(bool),
            stop_flag: *std.atomic.Value(bool),
            count: *std.atomic.Value(usize),
        ) void {
            while (!start_flag.load(.acquire)) std.atomic.spinLoopHint();
            while (!stop_flag.load(.acquire)) {
                if (target.acquire(9)) |held| {
                    var lease = held;
                    std.debug.assert(lease.snapshot().pos == 99);
                    _ = count.fetchAdd(1, .monotonic);
                    lease.release();
                }
            }
        }
    };

    var readers: [4]std.Thread = undefined;
    for (&readers) |*reader| {
        reader.* = try std.Thread.spawn(
            .{},
            Reader.run,
            .{ &table, &start, &stop, &acquired },
        );
    }
    start.store(true, .release);
    while (acquired.load(.acquire) == 0) std.atomic.spinLoopHint();
    try std.testing.expect(table.remove(9));
    stop.store(true, .release);
    for (readers) |reader| reader.join();

    try std.testing.expect(table.acquire(9) == null);
    try std.testing.expectEqual(@as(usize, 0), table.leakCount());
}

test "FdTable guest close cannot remove a concurrently renumbered stdio entry" {
    const Table = FdTableFor(true);
    const Runner = struct {
        fn close(target: *Table, start: *std.atomic.Value(bool)) void {
            while (!start.load(.acquire)) std.atomic.spinLoopHint();
            _ = target.closeGuest(10);
        }

        fn renumber(target: *Table, start: *std.atomic.Value(bool)) void {
            while (!start.load(.acquire)) std.atomic.spinLoopHint();
            target.renumber(0, 10) catch {};
        }
    };

    for (0..100) |_| {
        var table = try Table.init(std.testing.allocator, testing_io);
        defer table.deinit() catch unreachable;
        try table.insert(0, .{ .kind = .stdin });
        try table.insert(10, .{ .kind = .regular_file });
        var start = std.atomic.Value(bool).init(false);
        const closer = try std.Thread.spawn(.{}, Runner.close, .{ &table, &start });
        const renumberer = try std.Thread.spawn(.{}, Runner.renumber, .{ &table, &start });
        start.store(true, .release);
        closer.join();
        renumberer.join();

        const at_zero = table.snapshot(0);
        const at_ten = table.snapshot(10);
        try std.testing.expect((at_zero == null) != (at_ten == null));
        const survivor = at_zero orelse at_ten.?;
        try std.testing.expectEqual(FdKind.stdin, survivor.kind);
    }
}

test "FdTable preopen label follows the numeric target and closes with it" {
    const Table = FdTableFor(true);
    var table = try Table.init(std.testing.allocator, testing_io);
    defer table.deinit() catch unreachable;

    const name = try std.testing.allocator.dupe(u8, "/shared");
    const preopen_fd = try table.createPreopen(
        .{ .kind = .directory },
        name,
    );
    const replacement_fd = try table.create(.{ .kind = .directory });
    try table.renumber(replacement_fd, preopen_fd);

    try std.testing.expect(!table.contains(replacement_fd));
    var buf: [16]u8 = undefined;
    const len = table.copyPreopenName(preopen_fd, &buf).?;
    try std.testing.expectEqualStrings("/shared", buf[0..len]);
    try std.testing.expectEqual(@as(usize, 1), table.preopenCount());

    try std.testing.expect(table.remove(preopen_fd));
    try std.testing.expect(table.copyPreopenName(preopen_fd, &buf) == null);
    try std.testing.expectEqual(@as(usize, 0), table.preopenCount());
}

test "FdTable renumber drops a preopen label for a non-directory source" {
    const Table = FdTableFor(true);
    var table = try Table.init(std.testing.allocator, testing_io);
    defer table.deinit() catch unreachable;

    const name = try std.testing.allocator.dupe(u8, "/dir");
    const preopen_fd = try table.createPreopen(.{ .kind = .directory }, name);
    const file_fd = try table.create(.{ .kind = .regular_file });
    try table.renumber(file_fd, preopen_fd);

    try std.testing.expectEqual(FdKind.regular_file, table.snapshot(preopen_fd).?.kind);
    try std.testing.expectEqual(@as(usize, 0), table.preopenCount());
    var buf: [8]u8 = undefined;
    try std.testing.expect(table.copyPreopenName(preopen_fd, &buf) == null);
}

test "FdTable publication failure returns descriptor ownership to caller" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const linux = std.os.linux;
    var fds: [2]i32 = undefined;
    if (linux.errno(linux.pipe2(&fds, .{})) != .SUCCESS) return error.SkipZigTest;
    defer _ = linux.close(fds[0]);
    defer _ = linux.close(fds[1]);

    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{
        // Control, resource node, and first stable chunk succeed. The guest
        // fd directory allocation then fails after publication.
        .fail_index = 3,
    });
    const Table = FdTableFor(true);
    var table = try Table.init(failing.allocator(), testing_io);
    defer table.deinit() catch unreachable;

    try std.testing.expectError(
        error.OutOfMemory,
        table.create(.{ .kind = .regular_file, .host_fd = fds[0] }),
    );
    try std.testing.expectEqual(@as(usize, 0), table.leakCount());
    const still_open = linux.fcntl(fds[0], linux.F.GETFD, 0);
    try std.testing.expectEqual(linux.E.SUCCESS, linux.errno(still_open));
}

test "FdTable preopen publication rolls back both directories on OOM" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{
        // Control, resource node/chunk, and descriptor-directory capacity
        // succeed; preopen-directory capacity fails.
        .fail_index = 4,
    });
    const Table = FdTableFor(true);
    var table = try Table.init(failing.allocator(), testing_io);
    defer table.deinit() catch unreachable;
    const name = try std.testing.allocator.dupe(u8, "/rollback");
    defer std.testing.allocator.free(name);

    try std.testing.expectError(
        error.OutOfMemory,
        table.createPreopen(.{ .kind = .directory }, name),
    );
    try std.testing.expectEqual(@as(usize, 0), table.leakCount());
    try std.testing.expectEqual(@as(usize, 0), table.preopenCount());
}

test "FdTable disabled specialization preserves direct behavior" {
    const Table = FdTableFor(false);
    const Entry = FdEntryFor(false);
    try std.testing.expectEqual(@as(usize, 0), @sizeOf(@TypeOf(@as(Entry, .{
        .kind = .regular_file,
    }).mutex)));
    try std.testing.expectEqual(@sizeOf(FdEntrySnapshot), @sizeOf(Entry));

    var table = try Table.init(std.testing.allocator, testing_io);
    defer table.deinit() catch unreachable;
    const fd = try table.create(.{ .kind = .regular_file, .pos = 5 });
    try std.testing.expectEqual(@as(u64, 5), table.snapshot(fd).?.pos);
    try std.testing.expect(table.remove(fd));
    try std.testing.expectEqual(@as(usize, 0), table.leakCount());
}

test "WasiCtx concurrent final releases destroy exactly once" {
    if (!config.lib_wasi_threads) return error.SkipZigTest;
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    ctx.retain();

    const Releaser = struct {
        fn run(target: *WasiCtx) void {
            target.deinit();
        }
    };
    const first = try std.Thread.spawn(.{}, Releaser.run, .{ctx});
    const second = try std.Thread.spawn(.{}, Releaser.run, .{ctx});
    first.join();
    second.join();
}

test "WasiCtx init/deinit lifecycle" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    // Verify stdio fds are pre-populated
    try std.testing.expect(ctx.fd_table.contains(0));
    try std.testing.expect(ctx.fd_table.contains(1));
    try std.testing.expect(ctx.fd_table.contains(2));
    try std.testing.expect(!ctx.fd_table.contains(3));
    try std.testing.expect(ctx.getExitCode() == null);
}

test "FdTable insert, snapshot, remove" {
    var table = try FdTable.init(std.testing.allocator, testing_io);
    defer table.deinit() catch unreachable;

    try table.insert(10, .{ .kind = .regular_file });
    const entry = table.snapshot(10);
    try std.testing.expect(entry != null);
    try std.testing.expectEqual(FdEntry.FdKind.regular_file, entry.?.kind);

    try std.testing.expect(table.remove(10));
    try std.testing.expect(table.snapshot(10) == null);
}

test "FdTable create allocates sequential guest fds" {
    var table = try FdTable.init(std.testing.allocator, testing_io);
    defer table.deinit() catch unreachable;

    const fd1 = try table.create(.{ .kind = .regular_file });
    const fd2 = try table.create(.{ .kind = .regular_file });
    try std.testing.expectEqual(@as(u32, 3), fd1);
    try std.testing.expectEqual(@as(u32, 4), fd2);
}

test "addPreopen: directory rights default omits fd file-side bits (#476)" {
    // wasi-libc and wasmtime mask `path_open` results that yield a directory
    // so the four file-side rights — FD_READ, FD_WRITE, FD_TELL, FD_SEEK —
    // never appear on a preopen's `rights_base`. Regression-guard the
    // default mask used by `WasiCtx.addPreopen`.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const owned_dir = try tmp.dir.openDir(testing_io, ".", .{ .iterate = true });
    const fd = try ctx.addPreopen("/pre", owned_dir);
    const entry = ctx.fd_table.snapshot(fd) orelse return error.MissingFd;

    // The four file-side bits must be off on a preopen directory fd.
    try std.testing.expectEqual(@as(u64, 0), entry.rights_base & RIGHTS_FD_READ);
    try std.testing.expectEqual(@as(u64, 0), entry.rights_base & RIGHTS_FD_WRITE);
    try std.testing.expectEqual(@as(u64, 0), entry.rights_base & RIGHTS_FD_TELL);
    try std.testing.expectEqual(@as(u64, 0), entry.rights_base & RIGHTS_FD_SEEK);

    // Sanity: the canonical preopen mask is exactly DIRECTORY_BASE_RIGHTS
    // and the inheriting mask exposes file rights so children opened via
    // `path_open` from this preopen still get read/write/seek caps.
    try std.testing.expectEqual(DIRECTORY_BASE_RIGHTS, entry.rights_base);
    try std.testing.expectEqual(DIRECTORY_INHERITING_RIGHTS, entry.rights_inheriting);

    try std.testing.expectEqual(@as(?usize, 4), ctx.preopenNameLen(fd));
}

test "args_sizes_get with known args" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const args = [_][]const u8{ "hello", "world" };
    try ctx.setArgs(&args);

    const sizes = ctx.args_sizes_get();
    try std.testing.expectEqual(@as(u32, 2), sizes.count);
    // "hello\0" (6) + "world\0" (6) = 12
    try std.testing.expectEqual(@as(u32, 12), sizes.buf_size);
}

test "args_get writes NUL-terminated args" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const args = [_][]const u8{ "ab", "cd" };
    try ctx.setArgs(&args);

    var buf: [6]u8 = undefined;
    const result = ctx.args_get(&buf);
    try std.testing.expectEqual(@as(usize, 6), result.len);
    try std.testing.expectEqualSlices(u8, "ab\x00cd\x00", result);
}

test "environ_sizes_get" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const env = [_][]const u8{ "FOO=bar", "BAZ=qux" };
    try ctx.setEnv(&env);

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

    const fd = try ctx.fd_table.create(.{
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

    // Transfer the test-owned handle back before removing the guest entry.
    var lease = ctx.fd_table.acquire(fd).?;
    try std.testing.expectEqual(file.handle, lease.detachHostFd().?);
    lease.release();
    try std.testing.expect(ctx.fd_table.remove(fd));

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

    try std.testing.expect(ctx.getExitCode() == null);
    ctx.proc_exit(42);
    try std.testing.expectEqual(@as(u32, 42), ctx.getExitCode().?);
}

test "proc_exit with zero" {
    const ctx = try WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    ctx.proc_exit(0);
    try std.testing.expectEqual(@as(u32, 0), ctx.getExitCode().?);
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

test "Signal values match WASI witx" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(Signal.none));
    try std.testing.expectEqual(@as(u8, 6), @intFromEnum(Signal.abrt));
    try std.testing.expectEqual(@as(u8, 9), @intFromEnum(Signal.kill));
    try std.testing.expectEqual(@as(u8, 15), @intFromEnum(Signal.term));
    try std.testing.expectEqual(@as(u8, 16), @intFromEnum(Signal.chld));
    try std.testing.expectEqual(@as(u8, 27), @intFromEnum(Signal.winch));
    try std.testing.expectEqual(@as(u8, 30), @intFromEnum(Signal.sys));
}

test "wasiSignalToPosix: known signals map to POSIX numbering" {
    // 1..15 identity (HUP..TERM).
    try std.testing.expectEqual(@as(?u8, 1), wasiSignalToPosix(1));
    try std.testing.expectEqual(@as(?u8, 6), wasiSignalToPosix(6));
    try std.testing.expectEqual(@as(?u8, 9), wasiSignalToPosix(9));
    try std.testing.expectEqual(@as(?u8, 15), wasiSignalToPosix(15));
    // 16..30 shifted by one to match Linux POSIX numbering.
    try std.testing.expectEqual(@as(?u8, 17), wasiSignalToPosix(16)); // CHLD
    try std.testing.expectEqual(@as(?u8, 18), wasiSignalToPosix(17)); // CONT
    try std.testing.expectEqual(@as(?u8, 19), wasiSignalToPosix(18)); // STOP
    try std.testing.expectEqual(@as(?u8, 28), wasiSignalToPosix(27)); // WINCH
    try std.testing.expectEqual(@as(?u8, 31), wasiSignalToPosix(30)); // SYS
}

test "wasiSignalToPosix: none and out-of-range return null" {
    try std.testing.expectEqual(@as(?u8, null), wasiSignalToPosix(0));
    try std.testing.expectEqual(@as(?u8, null), wasiSignalToPosix(31));
    try std.testing.expectEqual(@as(?u8, null), wasiSignalToPosix(100));
    try std.testing.expectEqual(@as(?u8, null), wasiSignalToPosix(255));
}

test "IoVec slice" {
    var data = "hello".*;
    const iov = IoVec{ .buf = &data, .len = 5 };
    const s = iov.slice();
    try std.testing.expectEqualSlices(u8, "hello", s);
}
