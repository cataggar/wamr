//! Instance-local, allocation-free WASI snapshot-0 bindings for native embedding.
//! No hosted WASI process, filesystem, threads, or implicit platform services.
const std = @import("std");

/// WASI errno numbers, not host errno numbers. Other defined WASI errors may
/// also be returned by callbacks; numbers outside 0..76 are rejected as EIO.
pub const Errno = enum(u16) {
    success = 0,
    again = 6,
    badf = 8,
    fault = 21,
    intr = 27,
    inval = 28,
    io = 29,
    nospc = 51,
    nosys = 52,
    notsup = 58,
    overflow = 61,
    pipe = 64,
    notcapable = 76,
    _,
};

pub const ClockId = enum(u32) {
    realtime = 0,
    monotonic = 1,
    process_cputime = 2,
    thread_cputime = 3,
};

/// Unlike preview1 (SET=0, CUR=1, END=2), snapshot-0 uses this ordering.
pub const LegacyWhence = enum(u32) { cur = 0, end = 1, set = 2 };

pub const WriteResult = struct {
    written: usize = 0,
    errno: Errno = .success,
};

pub const Output = struct {
    userdata: ?*anyopaque,
    /// Consume at most bytes.len bytes synchronously. Never retain the slice or
    /// reenter the guest. Report actual consumption even if an error accompanies it.
    write: *const fn (?*anyopaque, fd: u32, bytes: []const u8) WriteResult,
};

pub const ClockResult = union(enum) {
    timestamp_ns: u64,
    failure: Errno,
};

pub const Clock = struct {
    userdata: ?*anyopaque,
    /// Zero means unsupported. Nonzero entries are the real platform resolution
    /// in nanoseconds for IDs 0..3, not a synthetic clock or precision promise.
    resolution_ns: [4]u64,
    /// All inputs/outputs use nanoseconds. Realtime is Unix-epoch time; monotonic
    /// has an unspecified fixed origin; CPU clocks measure actual consumed CPU.
    read: *const fn (?*anyopaque, ClockId, precision_ns: u64) ClockResult,
};

pub const Descriptor = struct {
    open: bool = true,
    /// Whether the output callback represents a terminal character device.
    terminal: bool = false,
};

pub const Options = struct {
    /// Borrowed immutable strings, excluding their terminating NUL. The caller
    /// owns their storage for the entire instance lifetime.
    args: []const []const u8,
    environment: []const []const u8,
    descriptors: [3]Descriptor = .{ .{}, .{}, .{} },
    output: ?Output,
    clock: ?Clock,
};

pub const Outcome = union(enum) {
    returned: Errno,
    /// Terminal: an adapter MUST unwind guest execution, including exit code 0.
    exited: u32,
};

pub const Context = struct {
    args: []const []const u8,
    environment: []const []const u8,
    descriptors: [3]Descriptor,
    output: ?Output,
    clock: ?Clock,
    exit_code: ?u32 = null,
    deferred_write_errors: [3]?Errno = .{ null, null, null },

    pub fn init(options: Options) error{InvalidStrings}!Context {
        _ = stringSizes(options.args) catch return error.InvalidStrings;
        _ = stringSizes(options.environment) catch return error.InvalidStrings;
        return .{
            .args = options.args,
            .environment = options.environment,
            .descriptors = options.descriptors,
            .output = options.output,
            .clock = options.clock,
        };
    }

    pub fn argsSizesGet(self: *const Context, mem: []u8, count: u32, size: u32) Errno {
        return writeSizes(mem, self.args, count, size);
    }

    pub fn argsGet(self: *const Context, mem: []u8, pointers: u32, buffer: u32) Errno {
        return writeStrings(mem, self.args, pointers, buffer);
    }

    pub fn environSizesGet(self: *const Context, mem: []u8, count: u32, size: u32) Errno {
        return writeSizes(mem, self.environment, count, size);
    }

    pub fn environGet(self: *const Context, mem: []u8, pointers: u32, buffer: u32) Errno {
        return writeStrings(mem, self.environment, pointers, buffer);
    }

    fn descriptor(self: *const Context, fd: u32) ?Descriptor {
        if (fd >= self.descriptors.len or !self.descriptors[fd].open) return null;
        return self.descriptors[fd];
    }

    fn rights(self: *const Context, fd: u32) u64 {
        return if (fd != 0 and self.output != null) @as(u64, 1) << 6 else 0;
    }

    pub fn fdFdstatGet(self: *const Context, mem: []u8, fd: u32, pointer: u32) Errno {
        const entry = self.descriptor(fd) orelse return .badf;
        const buffer = span(mem, pointer, 24) orelse return .fault;
        @memset(buffer, 0);
        buffer[0] = if (entry.terminal) 2 else 0;
        std.mem.writeInt(u64, buffer[8..16], self.rights(fd), .little);
        return .success;
    }

    pub fn fdClose(self: *Context, fd: u32) Errno {
        _ = self.descriptor(fd) orelse return .badf;
        self.descriptors[fd].open = false;
        return .success;
    }

    pub fn fdSeek(self: *const Context, mem: []u8, fd: u32, offset: i64, whence: u32, result: u32) Errno {
        _ = offset;
        _ = self.descriptor(fd) orelse return .badf;
        _ = std.enums.fromInt(LegacyWhence, whence) orelse return .inval;
        _ = span(mem, result, 8) orelse return .fault;
        // No descriptor grants FD_SEEK; do not pretend a stream is seekable.
        return .notcapable;
    }

    /// Inspect a late output failure, including after close or proc_exit.
    pub fn pendingWriteError(self: *const Context, fd: u32) ?Errno {
        if (fd >= self.deferred_write_errors.len) return null;
        return self.deferred_write_errors[fd];
    }

    /// Consume a diagnostic explicitly instead of delivering it on the next
    /// validated write to this descriptor. No callback failure is overwritten.
    pub fn takeWriteError(self: *Context, fd: u32) ?Errno {
        const errno = self.pendingWriteError(fd) orelse return null;
        self.deferred_write_errors[fd] = null;
        return errno;
    }

    fn writeFailure(self: *Context, fd: u32, written: u32, errno: Errno) Errno {
        if (written == 0) return errno;
        // wasi-libc ignores nwritten on errno: report progress as a short write
        // so retrying the remainder cannot duplicate already consumed bytes.
        std.debug.assert(self.deferred_write_errors[fd] == null);
        self.deferred_write_errors[fd] = errno;
        return .success;
    }

    pub fn fdWrite(self: *Context, mem: []u8, fd: u32, iovs: u32, count: u32, nwritten: u32) Errno {
        _ = self.descriptor(fd) orelse return .badf;
        if (fd == 0) return .notcapable;
        const sink = self.output orelse return .notcapable;
        const result = span(mem, nwritten, 4) orelse return .fault;
        const vectors = span(mem, iovs, @as(u64, count) * 8) orelse return .fault;

        // Validate the whole operation before emitting any externally visible bytes.
        var requested: u64 = 0;
        var at: usize = 0;
        while (at < vectors.len) : (at += 8) {
            const pointer = std.mem.readInt(u32, vectors[at..][0..4], .little);
            const length = std.mem.readInt(u32, vectors[at + 4 ..][0..4], .little);
            _ = span(mem, pointer, length) orelse return .fault;
            requested += length;
            if (requested > std.math.maxInt(u32)) return .overflow;
        }

        if (self.takeWriteError(fd)) |errno| {
            std.mem.writeInt(u32, result[0..4], 0, .little);
            return errno;
        }

        var written: u32 = 0;
        at = 0;
        while (at < vectors.len) : (at += 8) {
            const pointer = std.mem.readInt(u32, vectors[at..][0..4], .little);
            const length = std.mem.readInt(u32, vectors[at + 4 ..][0..4], .little);
            if (length == 0) continue;
            const bytes = span(mem, pointer, length).?;
            const response = sink.write(sink.userdata, fd, bytes);
            if (response.written > bytes.len) {
                std.mem.writeInt(u32, result[0..4], written, .little);
                return self.writeFailure(fd, written, .io);
            }
            written += @intCast(response.written);
            const errno = checkedErrno(response.errno);
            if (errno != .success or response.written < bytes.len) {
                std.mem.writeInt(u32, result[0..4], written, .little);
                return if (errno == .success) .success else self.writeFailure(fd, written, errno);
            }
        }
        std.mem.writeInt(u32, result[0..4], written, .little);
        return .success;
    }

    pub fn clockTimeGet(self: *const Context, mem: []u8, id: u32, precision: u64, pointer: u32) Errno {
        const clock_id = std.enums.fromInt(ClockId, id) orelse return .inval;
        const result = span(mem, pointer, 8) orelse return .fault;
        const clock = self.clock orelse return .notsup;
        if (clock.resolution_ns[id] == 0) return .notsup;
        switch (clock.read(clock.userdata, clock_id, precision)) {
            .timestamp_ns => |timestamp| std.mem.writeInt(u64, result[0..8], timestamp, .little),
            .failure => |errno| return if (errno == .success) .io else checkedErrno(errno),
        }
        return .success;
    }

    pub fn procExit(self: *Context, code: u32) Outcome {
        if (self.exit_code == null) self.exit_code = code;
        return .{ .exited = self.exit_code.? };
    }

    /// Raw wasm value bits in parameter order; i32 parameters use their low
    /// 32 bits. Validate the import signature with resolve before dispatching.
    pub fn dispatch(self: *Context, mem: []u8, function: Function, args: []const u64) error{InvalidArguments}!Outcome {
        if (self.exit_code) |code| return .{ .exited = code };
        if (args.len != imports[@intFromEnum(function)].params.len) return error.InvalidArguments;
        const errno: Errno = switch (function) {
            .fd_prestat_get, .fd_prestat_dir_name => .badf,
            .environ_sizes_get => self.environSizesGet(mem, low(args[0]), low(args[1])),
            .environ_get => self.environGet(mem, low(args[0]), low(args[1])),
            .args_sizes_get => self.argsSizesGet(mem, low(args[0]), low(args[1])),
            .args_get => self.argsGet(mem, low(args[0]), low(args[1])),
            .clock_time_get => self.clockTimeGet(mem, low(args[0]), args[1], low(args[2])),
            .proc_exit => return self.procExit(low(args[0])),
            .fd_fdstat_get => self.fdFdstatGet(mem, low(args[0]), low(args[1])),
            .fd_close => self.fdClose(low(args[0])),
            .fd_seek => self.fdSeek(mem, low(args[0]), @bitCast(args[1]), low(args[2]), low(args[3])),
            .fd_write => self.fdWrite(mem, low(args[0]), low(args[1]), low(args[2]), low(args[3])),
        };
        return .{ .returned = errno };
    }
};

pub const Function = enum {
    fd_prestat_get,
    fd_prestat_dir_name,
    environ_sizes_get,
    environ_get,
    args_sizes_get,
    args_get,
    clock_time_get,
    proc_exit,
    fd_fdstat_get,
    fd_close,
    fd_seek,
    fd_write,
};

pub const Import = struct {
    namespace: []const u8 = "wasi_unstable",
    name: []const u8,
    /// Wasm value-type bytes, not C signatures or preview1 aliases.
    params: []const u8,
    results: []const u8 = "\x7f",
};

pub const imports = [_]Import{
    .{ .name = "fd_prestat_get", .params = "\x7f\x7f" },
    .{ .name = "fd_prestat_dir_name", .params = "\x7f\x7f\x7f" },
    .{ .name = "environ_sizes_get", .params = "\x7f\x7f" },
    .{ .name = "environ_get", .params = "\x7f\x7f" },
    .{ .name = "args_sizes_get", .params = "\x7f\x7f" },
    .{ .name = "args_get", .params = "\x7f\x7f" },
    .{ .name = "clock_time_get", .params = "\x7f\x7e\x7f" },
    .{ .name = "proc_exit", .params = "\x7f", .results = "" },
    .{ .name = "fd_fdstat_get", .params = "\x7f\x7f" },
    .{ .name = "fd_close", .params = "\x7f" },
    .{ .name = "fd_seek", .params = "\x7f\x7e\x7f\x7f" },
    .{ .name = "fd_write", .params = "\x7f\x7f\x7f\x7f" },
};

pub fn resolve(namespace: []const u8, name: []const u8, params: []const u8, results: []const u8) ?Function {
    for (imports, 0..) |entry, index| {
        if (std.mem.eql(u8, namespace, entry.namespace) and
            std.mem.eql(u8, name, entry.name) and
            std.mem.eql(u8, params, entry.params) and
            std.mem.eql(u8, results, entry.results))
            return @enumFromInt(index);
    }
    return null;
}

fn low(value: u64) u32 {
    return @truncate(value);
}

fn checkedErrno(errno: Errno) Errno {
    return if (@intFromEnum(errno) <= 76) errno else .io;
}

fn span(mem: []u8, pointer: u32, length: u64) ?[]u8 {
    if (length > (@as(u64, 1) << 32) - pointer) return null;
    const end = @as(u64, pointer) + length;
    if (end > mem.len) return null;
    return mem[@intCast(pointer)..@intCast(end)];
}

const StringSizes = struct { count: u32, bytes: u32 };

fn stringSizes(entries: []const []const u8) error{ InvalidStrings, Overflow }!StringSizes {
    if (entries.len > std.math.maxInt(u32)) return error.Overflow;
    var size: u32 = 0;
    for (entries) |entry| {
        if (entry.len >= std.math.maxInt(u32)) return error.Overflow;
        if (std.mem.indexOfScalar(u8, entry, 0) != null) return error.InvalidStrings;
        size = std.math.add(u32, size, @as(u32, @intCast(entry.len)) + 1) catch return error.Overflow;
    }
    return .{ .count = @intCast(entries.len), .bytes = size };
}

fn writeSizes(mem: []u8, entries: []const []const u8, count: u32, size: u32) Errno {
    const sizes = stringSizes(entries) catch return .overflow;
    const count_out = span(mem, count, 4) orelse return .fault;
    const size_out = span(mem, size, 4) orelse return .fault;
    std.mem.writeInt(u32, count_out[0..4], sizes.count, .little);
    std.mem.writeInt(u32, size_out[0..4], sizes.bytes, .little);
    return .success;
}

fn writeStrings(mem: []u8, entries: []const []const u8, pointers: u32, buffer: u32) Errno {
    const sizes = stringSizes(entries) catch return .overflow;
    const table = span(mem, pointers, @as(u64, sizes.count) * 4) orelse return .fault;
    const bytes = span(mem, buffer, sizes.bytes) orelse return .fault;
    // Two overlapping output arrays cannot represent a valid string table.
    if (table.len != 0 and bytes.len != 0 and
        @as(u64, pointers) < @as(u64, buffer) + bytes.len and
        @as(u64, buffer) < @as(u64, pointers) + table.len) return .inval;
    var offset: usize = 0;
    for (entries, 0..) |entry, index| {
        std.mem.writeInt(u32, table[index * 4 ..][0..4], @intCast(@as(u64, buffer) + offset), .little);
        @memcpy(bytes[offset..][0..entry.len], entry);
        bytes[offset + entry.len] = 0;
        offset += entry.len + 1;
    }
    return .success;
}

test {
    _ = @import("minimal_test.zig");
}
