const std = @import("std");
const wasi = @import("minimal.zig");
const testing = std.testing;

fn context() wasi.Context {
    return wasi.Context.init(.{ .args = &.{}, .environment = &.{}, .output = null, .clock = null }) catch unreachable;
}

fn put32(mem: []u8, offset: usize, value: u32) void {
    std.mem.writeInt(u32, mem[offset..][0..4], value, .little);
}

fn get32(mem: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, mem[offset..][0..4], .little);
}

test "minimal WASI exact argument and environment string tables" {
    var ctx = try wasi.Context.init(.{
        .args = &.{ "coremark", "", "123" },
        .environment = &.{ "A=b", "EMPTY=" },
        .output = null,
        .clock = null,
    });
    var mem = [_]u8{0xaa} ** 128;
    try testing.expectEqual(.success, ctx.argsSizesGet(&mem, 0, 4));
    try testing.expectEqual(@as(u32, 3), get32(&mem, 0));
    try testing.expectEqual(@as(u32, 14), get32(&mem, 4));
    try testing.expectEqual(.success, ctx.argsGet(&mem, 8, 32));
    try testing.expectEqual(@as(u32, 32), get32(&mem, 8));
    try testing.expectEqual(@as(u32, 41), get32(&mem, 12));
    try testing.expectEqual(@as(u32, 42), get32(&mem, 16));
    try testing.expectEqualStrings("coremark\x00\x00123\x00", mem[32..46]);
    try testing.expectEqual(@as(u8, 0xaa), mem[20]);
    try testing.expectEqual(.success, ctx.environSizesGet(&mem, 0, 4));
    try testing.expectEqual(@as(u32, 2), get32(&mem, 0));
    try testing.expectEqual(@as(u32, 11), get32(&mem, 4));
    try testing.expectEqual(.success, ctx.environGet(&mem, 64, 80));
    try testing.expectEqual(@as(u32, 80), get32(&mem, 64));
    try testing.expectEqual(@as(u32, 84), get32(&mem, 68));
    try testing.expectEqualStrings("A=b\x00EMPTY=\x00", mem[80..91]);
}

test "minimal WASI rejects invalid strings and atomically checks output ranges" {
    try testing.expectError(error.InvalidStrings, wasi.Context.init(.{
        .args = &.{"a\x00b"},
        .environment = &.{},
        .output = null,
        .clock = null,
    }));
    var ctx = context();
    var mem = [_]u8{0xaa} ** 64;
    try testing.expectEqual(.success, ctx.argsSizesGet(&mem, 0, 4));
    try testing.expectEqual(@as(u32, 0), get32(&mem, 0));
    try testing.expectEqual(@as(u32, 0), get32(&mem, 4));
    try testing.expectEqual(.success, ctx.argsGet(&mem, 64, 64));
    try testing.expectEqual(.fault, ctx.argsGet(&mem, 65, 64));
    @memset(&mem, 0xaa);
    ctx.args = &.{ "a", "bc" };
    try testing.expectEqual(.fault, ctx.argsSizesGet(&mem, 0, 0xffff_ffff));
    try testing.expectEqual(.fault, ctx.argsGet(&mem, 0xffff_ffff, 20));
    try testing.expectEqual(.fault, ctx.argsGet(&mem, 0, 61));
    try testing.expectEqual(.inval, ctx.argsGet(&mem, 0, 4));
    for (mem) |byte| try testing.expectEqual(@as(u8, 0xaa), byte);
}

const Sink = struct {
    bytes: [64]u8 = undefined,
    used: usize = 0,
    calls: usize = 0,
    fd: u32 = 0,
    limit: usize = 64,
    fail_call: usize = 0,
    failure: wasi.Errno = .io,
    failure_limit: ?usize = null,
    overreport: bool = false,

    fn write(raw: ?*anyopaque, fd: u32, bytes: []const u8) wasi.WriteResult {
        const self: *Sink = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        self.fd = fd;
        const limit = if (self.calls == self.fail_call) self.failure_limit orelse self.limit else self.limit;
        const count = @min(bytes.len, limit);
        @memcpy(self.bytes[self.used..][0..count], bytes[0..count]);
        self.used += count;
        return .{
            .written = if (self.overreport) bytes.len + 1 else count,
            .errno = if (self.calls == self.fail_call) self.failure else .success,
        };
    }

    fn output(self: *Sink) wasi.Output {
        return .{ .userdata = self, .write = write };
    }
};

fn ioMemory() [64]u8 {
    var mem = [_]u8{0xaa} ** 64;
    put32(&mem, 0, 32);
    put32(&mem, 4, 3);
    put32(&mem, 8, 40);
    put32(&mem, 12, 4);
    @memcpy(mem[32..35], "abc");
    @memcpy(mem[40..44], "d\x00ef");
    return mem;
}

test "minimal WASI descriptor rights lifecycle and absent preopens" {
    var sink: Sink = .{};
    var ctx = context();
    ctx.output = sink.output();
    var other = context();
    var mem = ioMemory();
    for (0..3) |fd| {
        try testing.expectEqual(.success, ctx.fdFdstatGet(&mem, @intCast(fd), 16));
        try testing.expectEqual(@as(u8, 0), mem[16]);
        try testing.expectEqual(@as(u64, if (fd == 0) 0 else 64), std.mem.readInt(u64, mem[24..32], .little));
        try testing.expectEqual(@as(u64, 0), std.mem.readInt(u64, mem[32..40], .little));
        try testing.expectEqual(.badf, (try ctx.dispatch(&mem, .fd_prestat_get, &.{ fd, 0xffff_ffff })).returned);
        try testing.expectEqual(.badf, (try ctx.dispatch(&mem, .fd_prestat_dir_name, &.{ fd, 0, 0 })).returned);
    }
    ctx.descriptors[2].terminal = true;
    try testing.expectEqual(.success, ctx.fdFdstatGet(&mem, 2, 16));
    try testing.expectEqual(@as(u8, 2), mem[16]);
    try testing.expectEqual(.fault, ctx.fdFdstatGet(&mem, 1, 0xffff_ffff));
    try testing.expectEqual(.notcapable, ctx.fdWrite(&mem, 0, 0, 0, 16));
    try testing.expectEqual(.notcapable, other.fdWrite(&mem, 1, 0, 0, 16));
    for ([_]u32{ 3, 0xffff_ffff }) |fd| {
        try testing.expectEqual(.badf, ctx.fdFdstatGet(&mem, fd, 16));
        try testing.expectEqual(.badf, ctx.fdClose(fd));
        try testing.expectEqual(.badf, ctx.fdWrite(&mem, fd, 0, 0, 16));
        try testing.expectEqual(.badf, ctx.fdSeek(&mem, fd, 0, 2, 16));
    }
    for (0..3) |whence| try testing.expectEqual(.notcapable, ctx.fdSeek(&mem, 1, -3, @intCast(whence), 16));
    try testing.expectEqual(.inval, ctx.fdSeek(&mem, 1, 0, 3, 16));
    try testing.expectEqual(.fault, ctx.fdSeek(&mem, 1, 0, 2, 0xffff_ffff));
    try testing.expectEqual(@as(u32, 0), @intFromEnum(wasi.LegacyWhence.cur));
    try testing.expectEqual(@as(u32, 1), @intFromEnum(wasi.LegacyWhence.end));
    try testing.expectEqual(@as(u32, 2), @intFromEnum(wasi.LegacyWhence.set));
    try testing.expectEqual(.success, ctx.fdClose(1));
    try testing.expectEqual(.badf, ctx.fdClose(1));
    try testing.expectEqual(.badf, ctx.fdFdstatGet(&mem, 1, 16));
    try testing.expectEqual(.badf, ctx.fdWrite(&mem, 1, 0, 0, 16));
    try testing.expectEqual(.badf, ctx.fdSeek(&mem, 1, 0, 2, 16));
    try testing.expectEqual(.success, other.fdFdstatGet(&mem, 1, 16));
}

test "minimal WASI fd_write preserves exact bytes partial zero and failed writes" {
    var mem = ioMemory();
    var sink: Sink = .{};
    var ctx = context();
    ctx.output = sink.output();
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 2, 0, 2, 16));
    try testing.expectEqualStrings("abcd\x00ef", sink.bytes[0..sink.used]);
    try testing.expectEqual(@as(u32, 2), sink.fd);
    try testing.expectEqual(@as(u32, 7), get32(&mem, 16));
    sink = .{ .limit = 2 };
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 2, 16));
    try testing.expectEqualStrings("ab", sink.bytes[0..sink.used]);
    try testing.expectEqual(@as(u32, 2), get32(&mem, 16));
    try testing.expectEqual(@as(usize, 1), sink.calls);
    sink = .{ .limit = 0 };
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 2, 16));
    try testing.expectEqual(@as(u32, 0), get32(&mem, 16));
    try testing.expectEqual(@as(usize, 1), sink.calls);
    sink = .{ .fail_call = 2, .failure = .nospc };
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 2, 16));
    try testing.expectEqual(@as(u32, 7), get32(&mem, 16));
    try testing.expectEqualStrings("abcd\x00ef", sink.bytes[0..sink.used]);
    try testing.expectEqual(wasi.Errno.nospc, ctx.pendingWriteError(1).?);
    try testing.expectEqual(wasi.Errno.nospc, ctx.takeWriteError(1).?);
    try testing.expectEqual(null, ctx.takeWriteError(1));
    sink = .{ .fail_call = 1, .failure = .again, .limit = 0 };
    try testing.expectEqual(.again, ctx.fdWrite(&mem, 1, 0, 2, 16));
    try testing.expectEqual(@as(u32, 0), get32(&mem, 16));
    sink = .{ .fail_call = 1, .failure = .pipe, .limit = 1 };
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 2, 16));
    try testing.expectEqual(@as(u32, 1), get32(&mem, 16));
    try testing.expectEqualStrings("a", sink.bytes[0..sink.used]);
    try testing.expectEqual(wasi.Errno.pipe, ctx.takeWriteError(1).?);
    sink = .{ .overreport = true };
    try testing.expectEqual(.io, ctx.fdWrite(&mem, 1, 0, 2, 16));
    sink = .{ .fail_call = 1, .failure = @enumFromInt(65535) };
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 2, 16));
    try testing.expectEqual(@as(u32, 3), get32(&mem, 16));
    try testing.expectEqual(wasi.Errno.io, ctx.takeWriteError(1).?);
}

test "minimal WASI late EAGAIN and EINTR preserve progress and exact remainder retries" {
    for ([_]wasi.Errno{ .again, .intr }) |failure| {
        for ([_]u32{ 0, 2 }) |late_progress| {
            var mem = ioMemory();
            var sink: Sink = .{ .fail_call = 2, .failure = failure, .failure_limit = late_progress };
            var ctx = context();
            ctx.output = sink.output();
            try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 2, 16));
            try testing.expectEqual(@as(u32, 3) + late_progress, get32(&mem, 16));
            try testing.expectEqualStrings("abcd\x00ef"[0 .. 3 + late_progress], sink.bytes[0..sink.used]);
            try testing.expectEqual(failure, ctx.pendingWriteError(1).?);

            // A writev caller skips the completed first iovec and any consumed
            // bytes in the second. Delivery of the deferred error adds no output.
            put32(&mem, 8, 40 + late_progress);
            put32(&mem, 12, 4 - late_progress);
            try testing.expectEqual(failure, ctx.fdWrite(&mem, 1, 8, 1, 16));
            try testing.expectEqual(@as(u32, 0), get32(&mem, 16));
            try testing.expectEqual(@as(usize, 2), sink.calls);
            try testing.expectEqual(null, ctx.pendingWriteError(1));
            try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 8, 1, 16));
            try testing.expectEqual(@as(u32, 4) - late_progress, get32(&mem, 16));
            try testing.expectEqualStrings("abcd\x00ef", sink.bytes[0..sink.used]);
            try testing.expectEqual(@as(usize, 3), sink.calls);
        }
    }
}

test "minimal WASI zero progress output errors remain immediate" {
    for ([_]wasi.Errno{ .again, .intr, .pipe }) |failure| {
        var mem = ioMemory();
        var sink: Sink = .{ .fail_call = 1, .failure = failure, .failure_limit = 0 };
        var ctx = context();
        ctx.output = sink.output();
        try testing.expectEqual(failure, ctx.fdWrite(&mem, 1, 0, 2, 16));
        try testing.expectEqual(@as(u32, 0), get32(&mem, 16));
        try testing.expectEqual(@as(usize, 0), sink.used);
        try testing.expectEqual(null, ctx.pendingWriteError(1));
        try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 2, 16));
        try testing.expectEqual(@as(u32, 7), get32(&mem, 16));
        try testing.expectEqualStrings("abcd\x00ef", sink.bytes[0..sink.used]);
    }
}

test "minimal WASI deferred output diagnostics survive invalid requests close and exit" {
    var mem = ioMemory();
    var sink: Sink = .{ .fail_call = 2, .failure = .io, .failure_limit = 0 };
    var ctx = context();
    ctx.output = sink.output();
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 2, 16));
    try testing.expectEqual(.fault, ctx.fdWrite(&mem, 1, 8, 1, 0xffff_ffff));
    try testing.expectEqual(.fault, ctx.fdWrite(&mem, 1, 0xffff_ffff, 1, 16));
    try testing.expectEqual(wasi.Errno.io, ctx.pendingWriteError(1).?);
    try testing.expectEqual(null, ctx.pendingWriteError(2));
    try testing.expectEqual(null, ctx.pendingWriteError(0xffff_ffff));
    try testing.expectEqual(null, ctx.takeWriteError(0xffff_ffff));
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 2, 8, 1, 16));
    try testing.expectEqualStrings("abcd\x00ef", sink.bytes[0..sink.used]);
    try testing.expectEqual(wasi.Errno.io, ctx.pendingWriteError(1).?);
    try testing.expectEqual(.success, ctx.fdClose(1));
    try testing.expectEqual(.badf, ctx.fdWrite(&mem, 1, 8, 1, 16));
    try testing.expectEqual(@as(u32, 0), ctx.procExit(0).exited);
    try testing.expectEqual(wasi.Errno.io, ctx.takeWriteError(1).?);
    try testing.expectEqual(null, ctx.pendingWriteError(1));
}

test "minimal WASI fd_write validates all iovecs before side effects including overflow" {
    var mem = ioMemory();
    var sink: Sink = .{};
    var ctx = context();
    ctx.output = sink.output();
    try testing.expectEqual(.fault, ctx.fdWrite(&mem, 1, 0xffff_ffff, 1, 16));
    try testing.expectEqual(.fault, ctx.fdWrite(&mem, 1, 0, 0xffff_ffff, 16));
    try testing.expectEqual(.fault, ctx.fdWrite(&mem, 1, 0, 2, 0xffff_ffff));
    put32(&mem, 8, 0xffff_ffff);
    try testing.expectEqual(.fault, ctx.fdWrite(&mem, 1, 0, 2, 16));
    put32(&mem, 8, 40);
    put32(&mem, 12, 0xffff_ffff);
    try testing.expectEqual(.fault, ctx.fdWrite(&mem, 1, 0, 2, 16));
    try testing.expectEqual(@as(usize, 0), sink.calls);
    try testing.expectEqual(@as(u32, 0xaaaa_aaaa), get32(&mem, 16));
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 64, 0, 16));
    try testing.expectEqual(@as(u32, 0), get32(&mem, 16));
    put32(&mem, 0, 64);
    put32(&mem, 4, 0);
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 1, 16));
    try testing.expectEqual(@as(usize, 0), sink.calls);
    // nwritten may alias an iovec: it must not be written during traversal.
    mem = ioMemory();
    try testing.expectEqual(.success, ctx.fdWrite(&mem, 1, 0, 2, 8));
    try testing.expectEqualStrings("abcd\x00ef", sink.bytes[0..sink.used]);
}

test "minimal WASI rejects aggregate write count overflow without output" {
    const count = 65536;
    const result_offset = count * 8;
    const mem = try testing.allocator.alloc(u8, result_offset + 4);
    defer testing.allocator.free(mem);
    for (0..count) |index| {
        put32(mem, index * 8, 0);
        put32(mem, index * 8 + 4, 65536);
    }
    put32(mem, result_offset, 0xaaaa_aaaa);
    var sink: Sink = .{};
    var ctx = context();
    ctx.output = sink.output();
    try testing.expectEqual(.overflow, ctx.fdWrite(mem, 1, 0, count, result_offset));
    try testing.expectEqual(@as(usize, 0), sink.calls);
    try testing.expectEqual(@as(u32, 0xaaaa_aaaa), get32(mem, result_offset));
}

const TestClock = struct {
    calls: usize = 0,
    id: wasi.ClockId = .realtime,
    precision: u64 = 0,
    result: wasi.ClockResult = .{ .timestamp_ns = 123_456_789 },

    fn read(raw: ?*anyopaque, id: wasi.ClockId, precision: u64) wasi.ClockResult {
        const self: *TestClock = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        self.id = id;
        self.precision = precision;
        return self.result;
    }
};

test "minimal WASI platform clock capabilities precision errors and checked output" {
    var mem = [_]u8{0xaa} ** 16;
    var clock: TestClock = .{};
    var ctx = context();
    try testing.expectEqual(.notsup, ctx.clockTimeGet(&mem, 1, 0, 0));
    ctx.clock = .{ .userdata = &clock, .resolution_ns = .{ 0, 100, 0, 0 }, .read = TestClock.read };
    try testing.expectEqual(.notsup, ctx.clockTimeGet(&mem, 0, 0, 0));
    try testing.expectEqual(.notsup, ctx.clockTimeGet(&mem, 2, 0, 0));
    try testing.expectEqual(.notsup, ctx.clockTimeGet(&mem, 3, 0, 0));
    try testing.expectEqual(.inval, ctx.clockTimeGet(&mem, 4, 0, 0));
    try testing.expectEqual(.inval, ctx.clockTimeGet(&mem, 0xffff_ffff, 0, 0));
    try testing.expectEqual(.fault, ctx.clockTimeGet(&mem, 1, 0, 0xffff_ffff));
    try testing.expectEqual(@as(usize, 0), clock.calls);
    try testing.expectEqual(.success, ctx.clockTimeGet(&mem, 1, 999, 8));
    try testing.expectEqual(@as(u64, 123_456_789), std.mem.readInt(u64, mem[8..16], .little));
    try testing.expectEqual(wasi.ClockId.monotonic, clock.id);
    try testing.expectEqual(@as(u64, 999), clock.precision);
    for ([_]wasi.Errno{ .io, .intr, .success, @enumFromInt(65535) }) |errno| {
        clock.result = .{ .failure = errno };
        try testing.expectEqual(if (errno == .intr) wasi.Errno.intr else wasi.Errno.io, ctx.clockTimeGet(&mem, 1, 0, 0));
        try testing.expectEqual(@as(u64, 0xaaaa_aaaa_aaaa_aaaa), std.mem.readInt(u64, mem[0..8], .little));
    }
}

test "minimal WASI exact import resolution and terminal zero nonzero exits" {
    for (wasi.imports, 0..) |entry, index| {
        try testing.expectEqual(@as(wasi.Function, @enumFromInt(index)), wasi.resolve(entry.namespace, entry.name, entry.params, entry.results).?);
        try testing.expectEqual(null, wasi.resolve("wasi_snapshot_preview1", entry.name, entry.params, entry.results));
        try testing.expectEqual(null, wasi.resolve(entry.namespace, entry.name, "", entry.results));
        try testing.expectEqual(null, wasi.resolve(entry.namespace, entry.name, entry.params, "\x7e"));
    }
    var mem = [_]u8{0} ** 8;
    for ([_]u32{ 0, 7, 0xffff_ffff }) |code| {
        var ctx = context();
        try testing.expectError(error.InvalidArguments, ctx.dispatch(&mem, .proc_exit, &.{}));
        try testing.expectEqual(code, (try ctx.dispatch(&mem, .proc_exit, &.{code})).exited);
        try testing.expectEqual(code, ctx.exit_code.?);
        try testing.expectEqual(code, (try ctx.dispatch(&mem, .fd_close, &.{1})).exited);
        try testing.expect(ctx.descriptors[1].open);
        try testing.expectEqual(code, ctx.procExit(code +% 1).exited);
    }
}
