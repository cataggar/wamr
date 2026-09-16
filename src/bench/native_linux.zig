//! Linux capabilities for the compiler-free embedding benchmark.
const std = @import("std");
const aot = @import("../api/aot.zig");
const wasi = @import("minimal-wasi");

pub fn now() aot.PlatformError!u64 {
    return clockRead(.MONOTONIC);
}

fn clockRead(id: std.os.linux.clockid_t) aot.PlatformError!u64 {
    var ts: std.os.linux.timespec = undefined;
    if (std.posix.errno(std.os.linux.clock_gettime(id, &ts)) != .SUCCESS or ts.sec < 0)
        return error.ClockFailed;
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn resolution(id: std.os.linux.clockid_t) aot.PlatformError!u64 {
    var ts: std.os.linux.timespec = undefined;
    if (std.posix.errno(std.os.linux.clock_getres(id, &ts)) != .SUCCESS or ts.sec < 0)
        return error.ClockFailed;
    const ns = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    if (ns == 0) return error.ClockFailed;
    return ns;
}

pub fn wasiClock() !wasi.Clock {
    return .{
        .userdata = null,
        .resolution_ns = .{
            try resolution(.REALTIME),           try resolution(.MONOTONIC),
            try resolution(.PROCESS_CPUTIME_ID), try resolution(.THREAD_CPUTIME_ID),
        },
        .read = readClock,
    };
}

fn readClock(_: ?*anyopaque, id: wasi.ClockId, _: u64) wasi.ClockResult {
    const native_id: std.os.linux.clockid_t = switch (id) {
        .realtime => .REALTIME,
        .monotonic => .MONOTONIC,
        .process_cputime => .PROCESS_CPUTIME_ID,
        .thread_cputime => .THREAD_CPUTIME_ID,
    };
    return .{ .timestamp_ns = clockRead(native_id) catch return .{ .failure = .io } };
}

/// Counts only native code/linear-memory mmap reservations and RW/RX accessible
/// committed ranges. This is NOT RSS, allocator usage, or complete process RAM.
pub const Pages = struct {
    const Region = struct { base: [*]align(4096) u8, size: usize, committed: usize = 0 };
    regions: [2]?Region = .{ null, null },
    fail_reserve: bool = false,
    fail_commit: bool = false,
    fail_protect: bool = false,
    fail_protect_after_transition: bool = false,
    partial_protection_applied: bool = false,
    fail_clock_at: ?usize = null,
    clock_reads: usize = 0,

    pub fn platform(self: *Pages) aot.Platform {
        return .{
            .context = self,
            .reserve = reserve,
            .commit = commit,
            .protect = protect,
            .unmap = unmap,
            .monotonic_ns = monotonic,
        };
    }

    pub fn reserved(self: *const Pages) usize {
        var size: usize = 0;
        for (self.regions) |entry| if (entry) |region| {
            size += region.size;
        };
        return size;
    }

    pub fn committed(self: *const Pages) usize {
        var size: usize = 0;
        for (self.regions) |entry| if (entry) |region| {
            size += region.committed;
        };
        return size;
    }

    fn reserve(raw: *anyopaque, size: usize) aot.PlatformError![*]align(4096) u8 {
        const self: *Pages = @ptrCast(@alignCast(raw));
        if (self.fail_reserve) return error.OutOfMemory;
        for (&self.regions) |*entry| if (entry.* == null) {
            const bytes = std.posix.mmap(null, size, .{}, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch return error.OutOfMemory;
            const base: [*]align(4096) u8 = @alignCast(bytes.ptr);
            entry.* = .{ .base = base, .size = size };
            return base;
        };
        return error.OutOfMemory;
    }

    fn commit(raw: *anyopaque, address: [*]align(4096) u8, size: usize) aot.PlatformError!void {
        const self: *Pages = @ptrCast(@alignCast(raw));
        if (self.fail_commit) return error.OutOfMemory;
        for (&self.regions) |*entry| if (entry.*) |*region| {
            const offset = std.math.sub(usize, @intFromPtr(address), @intFromPtr(region.base)) catch continue;
            if (offset > region.committed or size > region.size -| offset) continue;
            if (std.posix.errno(std.os.linux.mprotect(address, size, .{ .READ = true, .WRITE = true })) != .SUCCESS)
                return error.OutOfMemory;
            region.committed = @max(region.committed, offset + size);
            return;
        };
        return error.InvalidMapping;
    }

    fn protect(raw: *anyopaque, address: [*]align(4096) u8, size: usize, permission: aot.platform.Protection) aot.PlatformError!void {
        const self: *Pages = @ptrCast(@alignCast(raw));
        if (self.fail_protect) return error.ProtectionFailed;
        const prot: std.posix.PROT = switch (permission) {
            .none => .{},
            .read_write => .{ .READ = true, .WRITE = true },
            .read_execute => .{ .READ = true, .EXEC = true },
        };
        if (permission == .none and self.fail_protect_after_transition) {
            if (std.posix.errno(std.os.linux.mprotect(address, @min(size, 4096), prot)) != .SUCCESS)
                return error.ProtectionFailed;
            self.partial_protection_applied = true;
            return error.ProtectionFailed;
        }
        if (std.posix.errno(std.os.linux.mprotect(address, size, prot)) != .SUCCESS)
            return error.ProtectionFailed;
        if (permission == .none) {
            for (&self.regions) |*entry| if (entry.*) |*region| {
                const offset = std.math.sub(usize, @intFromPtr(address), @intFromPtr(region.base)) catch continue;
                if (offset <= region.committed and size == region.committed - offset) {
                    region.committed = offset;
                    return;
                }
            };
        }
    }

    fn unmap(raw: *anyopaque, address: [*]align(4096) u8, size: usize) void {
        const self: *Pages = @ptrCast(@alignCast(raw));
        for (&self.regions) |*entry| if (entry.*) |region| {
            if (region.base == address and region.size == size) {
                std.posix.munmap(address[0..size]);
                entry.* = null;
                return;
            }
        };
        unreachable;
    }

    fn monotonic(raw: *anyopaque) aot.PlatformError!u64 {
        const self: *Pages = @ptrCast(@alignCast(raw));
        self.clock_reads += 1;
        if (self.fail_clock_at == self.clock_reads) return error.ClockFailed;
        return now();
    }
};

test "Linux benchmark real clock and reservation cleanup" {
    const before = try now();
    try std.testing.expect(try now() >= before);
    try std.testing.expect(try resolution(.MONOTONIC) > 0);
    const clock = try wasiClock();
    try std.testing.expect(clock.read(null, .monotonic, 0) == .timestamp_ns);
    var pages: Pages = .{};
    const p = pages.platform();
    const base = try p.reserve(p.context, 8192);
    try p.commit(p.context, base, 4096);
    base[0] = 42;
    try std.testing.expectEqual(@as(usize, 8192), pages.reserved());
    try std.testing.expectEqual(@as(usize, 4096), pages.committed());
    try p.commit(p.context, base, 4096);
    try std.testing.expectEqual(@as(usize, 4096), pages.committed());
    p.unmap(p.context, base, 8192);
    try std.testing.expectEqual(@as(usize, 0), pages.reserved());
    try std.testing.expectEqual(@as(usize, 0), pages.committed());
}
