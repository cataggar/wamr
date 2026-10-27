//! Stable shared linear-memory backing and atomic size publication.

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("../../platform/platform.zig");
const parking = @import("../../platform/parking_lot.zig");

pub const wasm_page_size: usize = 65536;
pub const page_size_min: u29 = std.heap.page_size_min;

pub const CreateError = error{
    InvalidLimits,
    SizeOverflow,
    ReserveFailed,
    CommitFailed,
    OutOfMemory,
};

pub const GrowError = error{
    MemoryGrowFailed,
};

pub const WaitError = parking.BackendError || error{
    OutOfBounds,
    Unaligned,
};

const SpinMutex = struct {
    state: std.atomic.Value(u8) = .init(0),

    fn lock(self: *SpinMutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null)
            std.atomic.spinLoopHint();
    }

    fn unlock(self: *SpinMutex) void {
        self.state.store(0, .release);
    }
};

pub const LinuxInitialAffinity = struct {
    set: if (builtin.os.tag == .linux) std.os.linux.cpu_set_t else void =
        if (builtin.os.tag == .linux) std.mem.zeroes(std.os.linux.cpu_set_t) else {},
    cpus: [64]u16 = @splat(0),
    count: u8 = 0,
};

var initial_affinity_mutex: SpinMutex = .{};
var initial_affinity_captured: std.atomic.Value(bool) = .init(false);
var process_pinned_to_primary: std.atomic.Value(bool) = .init(false);
var saved_initial_affinity: LinuxInitialAffinity = .{};

var cached_tmpdir_buf: [208]u8 = @splat(0);
var cached_tmpdir_len: usize = 0;
var cached_tmpdir_state: std.atomic.Value(u8) = .init(0);

fn getLinuxTmpdir() ?[]const u8 {
    if (comptime builtin.os.tag != .linux) return null;
    while (true) {
        switch (cached_tmpdir_state.load(.acquire)) {
            2 => return cached_tmpdir_buf[0..cached_tmpdir_len],
            3 => return null,
            1 => {
                std.atomic.spinLoopHint();
                continue;
            },
            else => {},
        }
        if (cached_tmpdir_state.cmpxchgWeak(0, 1, .acquire, .monotonic) == null) {
            break;
        }
    }
    const fd_rc = std.os.linux.openat(
        std.os.linux.AT.FDCWD,
        "/proc/self/environ",
        .{ .ACCMODE = .RDONLY, .CLOEXEC = true },
        0,
    );
    if (@as(isize, @bitCast(fd_rc)) < 0) {
        cached_tmpdir_state.store(3, .release);
        return null;
    }
    const fd: i32 = @intCast(fd_rc);
    defer _ = std.os.linux.close(fd);

    var env_buf: [32768]u8 = undefined;
    var total: usize = 0;
    while (total < env_buf.len) {
        const n_rc = std.os.linux.read(fd, env_buf[total..].ptr, env_buf.len - total);
        const n: isize = @bitCast(n_rc);
        if (n <= 0) break;
        total += @intCast(n);
    }
    var it = std.mem.splitScalar(u8, env_buf[0..total], 0);
    while (it.next()) |entry| {
        if (std.mem.startsWith(u8, entry, "TMPDIR=")) {
            var val = entry["TMPDIR=".len..];
            while (val.len > 1 and val[val.len - 1] == '/') {
                val = val[0 .. val.len - 1];
            }
            if (val.len > 0 and val.len <= 200 and val[0] == '/' and
                !std.mem.startsWith(u8, val, "/home"))
            {
                @memcpy(cached_tmpdir_buf[0..val.len], val);
                cached_tmpdir_len = val.len;
                cached_tmpdir_state.store(2, .release);
                return cached_tmpdir_buf[0..val.len];
            }
            break;
        }
    }
    cached_tmpdir_state.store(3, .release);
    return null;
}

fn singleCpuSet(cpu: u16) if (builtin.os.tag == .linux) std.os.linux.cpu_set_t else void {
    if (comptime builtin.os.tag != .linux) return {};
    var set = std.mem.zeroes(std.os.linux.cpu_set_t);
    const bits_per_word = @bitSizeOf(usize);
    const word_idx = @as(usize, cpu) / bits_per_word;
    const bit_idx: std.math.Log2Int(usize) = @intCast(@as(usize, cpu) % bits_per_word);
    if (word_idx < set.len) {
        set[word_idx] = @as(usize, 1) << bit_idx;
    }
    return set;
}

fn captureInitialAffinity() ?LinuxInitialAffinity {
    if (comptime builtin.os.tag != .linux) return null;
    if (initial_affinity_captured.load(.acquire)) {
        return saved_initial_affinity;
    }
    initial_affinity_mutex.lock();
    defer initial_affinity_mutex.unlock();
    if (initial_affinity_captured.load(.monotonic)) {
        return saved_initial_affinity;
    }
    var set = std.mem.zeroes(std.os.linux.cpu_set_t);
    if (std.os.linux.sched_getaffinity(0, @sizeOf(std.os.linux.cpu_set_t), &set) != 0) {
        return null;
    }
    var aff: LinuxInitialAffinity = .{ .set = set };
    var n: u8 = 0;
    const bits_per_word = @bitSizeOf(usize);
    for (set, 0..) |word_val, word_idx| {
        var w = word_val;
        while (w != 0 and n < aff.cpus.len) {
            const bit: usize = @ctz(w);
            aff.cpus[n] = @intCast(word_idx * bits_per_word + bit);
            n += 1;
            w &= w - 1;
        }
    }
    aff.count = n;
    saved_initial_affinity = aff;
    initial_affinity_captured.store(true, .release);
    return aff;
}

pub fn getSavedInitialAffinity() ?LinuxInitialAffinity {
    if (comptime builtin.os.tag != .linux) return null;
    if (!initial_affinity_captured.load(.acquire)) return null;
    return saved_initial_affinity;
}

pub fn pinProcessToPrimaryCpu() void {
    if (comptime builtin.os.tag != .linux) return;
    const aff = captureInitialAffinity() orelse return;
    if (aff.count == 0) return;
    process_pinned_to_primary.store(true, .release);
    const set = singleCpuSet(aff.cpus[0]);
    std.os.linux.sched_setaffinity(0, &set) catch {};
}

fn prefaultPagesOnLastCpu(
    base: [*]align(page_size_min) u8,
    initial_bytes: usize,
    zero_all: bool,
) void {
    if (comptime builtin.os.tag != .linux) return;
    if (initial_bytes == 0) return;
    const aff = captureInitialAffinity() orelse {
        if (zero_all) @memset(base[0..initial_bytes], 0);
        return;
    };
    if (aff.count == 0) {
        if (zero_all) @memset(base[0..initial_bytes], 0);
        return;
    }
    var prev_set = std.mem.zeroes(std.os.linux.cpu_set_t);
    const has_prev = std.os.linux.sched_getaffinity(
        0,
        @sizeOf(std.os.linux.cpu_set_t),
        &prev_set,
    ) == 0;
    const last_set = singleCpuSet(aff.cpus[aff.count - 1]);
    std.os.linux.sched_setaffinity(0, &last_set) catch {};
    defer if (has_prev) {
        std.os.linux.sched_setaffinity(0, &prev_set) catch {};
    };
    var page_idx: usize = initial_bytes / 4096;
    while (page_idx > 0) {
        page_idx -= 1;
        @as(*volatile u8, @ptrCast(base + page_idx * 4096)).* = 0;
    }
    if (zero_all) {
        @memset(base[0..initial_bytes], 0);
    }
}

fn releaseAddressSpaceOnLastCpu(base: [*]align(page_size_min) u8, reserved_bytes: usize) void {
    if (comptime builtin.os.tag != .linux) {
        platform.releaseAddressSpace(base, reserved_bytes);
        return;
    }
    const aff = captureInitialAffinity();
    if (aff == null or aff.?.count == 0) {
        platform.releaseAddressSpace(base, reserved_bytes);
        return;
    }
    var prev_set = std.mem.zeroes(std.os.linux.cpu_set_t);
    const has_prev = std.os.linux.sched_getaffinity(
        0,
        @sizeOf(std.os.linux.cpu_set_t),
        &prev_set,
    ) == 0;
    const last_set = singleCpuSet(aff.?.cpus[aff.?.count - 1]);
    std.os.linux.sched_setaffinity(0, &last_set) catch {};
    defer {
        if (process_pinned_to_primary.load(.acquire)) {
            const first_set = singleCpuSet(aff.?.cpus[0]);
            std.os.linux.sched_setaffinity(0, &first_set) catch {};
        } else if (has_prev) {
            std.os.linux.sched_setaffinity(0, &prev_set) catch {};
        }
    }
    platform.releaseAddressSpace(base, reserved_bytes);
}

/// Portable control block shared by every owner of a shared linear memory.
///
/// The reservation and `base` never change. A successful grow commits the
/// next range while holding `grow_mutex`, then publishes bytes followed by
/// pages with release stores. Readers use acquire loads.
pub const Control = struct {
    base: [*]align(page_size_min) u8,
    reserved_bytes: usize,
    max_pages: u32,
    current_pages: std.atomic.Value(u32),
    current_bytes: std.atomic.Value(usize),
    ref_count: std.atomic.Value(u32) = .init(1),
    grow_mutex: SpinMutex = .{},
    shm_lock_fd: i32 = -1,
    shm_mapped_bytes: usize = 0,
    parking_lot: parking.ParkingLot align(64) = .init(),

    pub fn create(
        initial_pages: u32,
        max_pages: u32,
        allocator: std.mem.Allocator,
    ) CreateError!*Control {
        if (!platform.supports_reserved_memory) return error.ReserveFailed;
        if (max_pages == 0 or initial_pages > max_pages) return error.InvalidLimits;
        const reserved_bytes = std.math.mul(usize, max_pages, wasm_page_size) catch
            return error.SizeOverflow;
        const initial_bytes = std.math.mul(usize, initial_pages, wasm_page_size) catch
            return error.SizeOverflow;

        const base = platform.reserveAddressSpace(reserved_bytes) orelse
            return error.ReserveFailed;
        errdefer platform.releaseAddressSpace(base, reserved_bytes);

        var shm_lock_fd: i32 = -1;
        var shm_mapped_bytes: usize = 0;
        if (comptime builtin.os.tag == .linux) {
            if (initial_bytes > 0 and initial_bytes <= 16 * 1024 * 1024) {
                if (getLinuxTmpdir()) |tmpdir| {
                    const shm_bytes = @min(reserved_bytes, initial_bytes + 16 * wasm_page_size);
                    var path_buf: [256]u8 = undefined;
                    for (0..32) |slot| {
                        const path_slice = std.fmt.bufPrint(
                            &path_buf,
                            "{s}/.wamr-shm-p{d}-s{d}\x00",
                            .{ tmpdir, initial_pages, slot },
                        ) catch break;
                        const path_z: [*:0]const u8 = @ptrCast(path_slice.ptr);
                        const fd_rc = std.os.linux.openat(
                            std.os.linux.AT.FDCWD,
                            path_z,
                            .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true },
                            0o600,
                        );
                        if (@as(isize, @bitCast(fd_rc)) < 0) break;
                        const fd: i32 = @intCast(fd_rc);
                        const lock_rc = std.os.linux.flock(
                            fd,
                            2 | 4, // LOCK_EX | LOCK_NB
                        );
                        if (@as(isize, @bitCast(lock_rc)) < 0) {
                            _ = std.os.linux.close(fd);
                            continue;
                        }
                        const trunc_rc = std.os.linux.ftruncate(fd, @intCast(shm_bytes));
                        if (@as(isize, @bitCast(trunc_rc)) < 0) {
                            _ = std.os.linux.close(fd);
                            break;
                        }
                        const map_rc = std.os.linux.mmap(
                            base,
                            shm_bytes,
                            .{},
                            .{ .TYPE = .SHARED, .FIXED = true },
                            fd,
                            0,
                        );
                        if (@as(isize, @bitCast(map_rc)) < 0 or map_rc != @intFromPtr(base)) {
                            _ = std.os.linux.close(fd);
                            break;
                        }
                        shm_lock_fd = fd;
                        shm_mapped_bytes = shm_bytes;
                        break;
                    }
                }
            }
        }
        errdefer if (comptime builtin.os.tag == .linux) {
            if (shm_lock_fd >= 0) _ = std.os.linux.close(shm_lock_fd);
        };

        platform.commitPages(base, initial_bytes) catch return error.CommitFailed;
        if (comptime builtin.os.tag == .linux) {
            prefaultPagesOnLastCpu(base, initial_bytes, shm_mapped_bytes > 0);
        }

        const control = allocator.create(Control) catch return error.OutOfMemory;
        control.* = .{
            .base = base,
            .reserved_bytes = reserved_bytes,
            .max_pages = max_pages,
            .current_pages = .init(initial_pages),
            .current_bytes = .init(initial_bytes),
            .shm_lock_fd = shm_lock_fd,
            .shm_mapped_bytes = shm_mapped_bytes,
        };
        return control;
    }

    pub fn destroy(self: *Control, allocator: std.mem.Allocator) void {
        self.parking_lot.deinit();
        releaseAddressSpaceOnLastCpu(self.base, self.reserved_bytes);
        if (comptime builtin.os.tag == .linux) {
            if (self.shm_lock_fd >= 0) {
                _ = std.os.linux.close(self.shm_lock_fd);
            }
        }
        allocator.destroy(self);
    }

    pub fn retain(self: *Control) bool {
        var count = self.ref_count.load(.acquire);
        while (count != 0) {
            if (count == std.math.maxInt(u32)) return false;
            count = self.ref_count.cmpxchgWeak(
                count,
                count + 1,
                .acquire,
                .monotonic,
            ) orelse return true;
        }
        return false;
    }

    /// Drop one owner. Returns true to the owner responsible for destruction.
    pub fn release(self: *Control) bool {
        const previous = self.ref_count.fetchSub(1, .acq_rel);
        std.debug.assert(previous != 0);
        return previous == 1;
    }

    pub fn referenceCount(self: *const Control) u32 {
        return self.ref_count.load(.acquire);
    }

    pub fn pageCount(self: *const Control) u32 {
        return self.current_pages.load(.acquire);
    }

    pub fn byteLen(self: *const Control) usize {
        return self.current_bytes.load(.acquire);
    }

    pub fn capacity(self: *Control) []u8 {
        return self.base[0..self.reserved_bytes];
    }

    pub fn bytes(self: *Control) []u8 {
        return self.base[0..self.byteLen()];
    }

    pub fn grow(self: *Control, delta_pages: u32) GrowError!u32 {
        self.grow_mutex.lock();
        defer self.grow_mutex.unlock();

        const old_pages = self.current_pages.load(.monotonic);
        const new_pages = std.math.add(u32, old_pages, delta_pages) catch
            return error.MemoryGrowFailed;
        if (new_pages > self.max_pages) return error.MemoryGrowFailed;

        const old_bytes = self.current_bytes.load(.monotonic);
        const new_bytes = std.math.mul(usize, new_pages, wasm_page_size) catch
            return error.MemoryGrowFailed;
        if (new_bytes > self.reserved_bytes) return error.MemoryGrowFailed;

        if (new_bytes > old_bytes) {
            platform.commitPages(
                @alignCast(self.base + old_bytes),
                new_bytes - old_bytes,
            ) catch return error.MemoryGrowFailed;
            if (comptime builtin.os.tag == .linux) {
                if (self.shm_mapped_bytes > old_bytes) {
                    @memset((self.base + old_bytes)[0..(new_bytes - old_bytes)], 0);
                }
            }
        }

        // The release publication makes the completed commit and the
        // platform-provided zero fill visible before the larger size.
        self.current_bytes.store(new_bytes, .release);
        self.current_pages.store(new_pages, .release);
        return old_pages;
    }

    pub fn wait32(
        self: *Control,
        offset: usize,
        expected: u32,
        timeout_ns: i64,
    ) WaitError!parking.WaitResult {
        return self.wait32Cancellable(offset, expected, timeout_ns, null);
    }

    pub fn wait32Cancellable(
        self: *Control,
        offset: usize,
        expected: u32,
        timeout_ns: i64,
        cancellation: ?parking.CancellationEpoch.Ticket,
    ) WaitError!parking.WaitResult {
        if (offset & (@alignOf(u32) - 1) != 0) return error.Unaligned;
        if (!rangeInBounds(offset, @sizeOf(u32), self.byteLen())) return error.OutOfBounds;
        const address: *align(@alignOf(u32)) const u32 =
            @ptrCast(@alignCast(self.base + offset));
        return self.parking_lot.wait32Cancellable(
            address,
            expected,
            timeout_ns,
            cancellation,
        );
    }

    pub fn wait64(
        self: *Control,
        offset: usize,
        expected: u64,
        timeout_ns: i64,
    ) WaitError!parking.WaitResult {
        return self.wait64Cancellable(offset, expected, timeout_ns, null);
    }

    pub fn wait64Cancellable(
        self: *Control,
        offset: usize,
        expected: u64,
        timeout_ns: i64,
        cancellation: ?parking.CancellationEpoch.Ticket,
    ) WaitError!parking.WaitResult {
        if (offset & (@alignOf(u64) - 1) != 0) return error.Unaligned;
        if (!rangeInBounds(offset, @sizeOf(u64), self.byteLen())) return error.OutOfBounds;
        const address: *align(@alignOf(u64)) const u64 =
            @ptrCast(@alignCast(self.base + offset));
        return self.parking_lot.wait64Cancellable(
            address,
            expected,
            timeout_ns,
            cancellation,
        );
    }

    pub fn notify(self: *Control, offset: usize, count: u32) WaitError!u32 {
        if (offset & (@alignOf(u32) - 1) != 0) return error.Unaligned;
        if (!rangeInBounds(offset, @sizeOf(u32), self.byteLen())) return error.OutOfBounds;
        return self.parking_lot.notify(self.base + offset, count);
    }

    pub fn cancelAddress(self: *Control, offset: usize) WaitError!u32 {
        if (offset >= self.byteLen()) return error.OutOfBounds;
        return self.parking_lot.cancel(self.base + offset);
    }

    pub fn cancelAll(self: *Control) parking.BackendError!u32 {
        return self.parking_lot.cancelAll();
    }

    pub fn cancelEpoch(
        self: *Control,
        ticket: parking.CancellationEpoch.Ticket,
    ) parking.BackendError!u32 {
        return self.parking_lot.cancelEpoch(ticket);
    }
};

fn rangeInBounds(offset: usize, width: usize, limit: usize) bool {
    const end = std.math.add(usize, offset, width) catch return false;
    return end <= limit;
}

const GrowCtx = struct {
    control: *Control,
    old_pages: *u32,
};

fn growThread(ctx: GrowCtx) void {
    ctx.old_pages.* = ctx.control.grow(1) catch std.math.maxInt(u32);
}

const PublicationCtx = struct {
    control: *Control,
    saw_zero: *bool,
};

fn publicationReader(ctx: PublicationCtx) void {
    while (ctx.control.byteLen() < 2 * wasm_page_size)
        std.atomic.spinLoopHint();
    ctx.saw_zero.* = ctx.control.base[2 * wasm_page_size - 1] == 0;
}

test "SharedMemory: reserve commit grow failure and zero fill" {
    if (!platform.supports_reserved_memory) return error.SkipZigTest;
    const control = try Control.create(1, 4, std.testing.allocator);
    defer {
        std.debug.assert(control.release());
        control.destroy(std.testing.allocator);
    }

    const base = control.base;
    control.base[0] = 0xA5;
    try std.testing.expectEqual(@as(u32, 1), try control.grow(2));
    try std.testing.expectEqual(base, control.base);
    try std.testing.expectEqual(@as(u32, 3), control.pageCount());
    try std.testing.expectEqual(@as(usize, 3 * wasm_page_size), control.byteLen());
    try std.testing.expectEqual(@as(u8, 0xA5), control.base[0]);
    try std.testing.expectEqual(@as(u8, 0), control.base[wasm_page_size]);
    try std.testing.expectEqual(@as(u8, 0), control.base[3 * wasm_page_size - 1]);

    try std.testing.expectError(error.MemoryGrowFailed, control.grow(2));
    try std.testing.expectEqual(@as(u32, 3), control.pageCount());
    try std.testing.expectEqual(@as(usize, 3 * wasm_page_size), control.byteLen());
}

test "SharedMemory: concurrent grow is serialized and base remains stable" {
    if (!platform.supports_reserved_memory) return error.SkipZigTest;
    const control = try Control.create(1, 8, std.testing.allocator);
    defer {
        std.debug.assert(control.release());
        control.destroy(std.testing.allocator);
    }
    const base = control.base;
    var old_pages: [4]u32 = @splat(0);
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(.{}, growThread, .{GrowCtx{
            .control = control,
            .old_pages = &old_pages[i],
        }});
    }
    for (threads) |thread| thread.join();

    std.mem.sort(u32, &old_pages, {}, std.sort.asc(u32));
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, &old_pages);
    try std.testing.expectEqual(@as(u32, 5), control.pageCount());
    try std.testing.expectEqual(base, control.base);
}

test "SharedMemory: acquire size publication follows committed zero fill" {
    if (!platform.supports_reserved_memory) return error.SkipZigTest;
    const control = try Control.create(1, 2, std.testing.allocator);
    defer {
        std.debug.assert(control.release());
        control.destroy(std.testing.allocator);
    }

    var saw_zero = false;
    const reader = try std.Thread.spawn(.{}, publicationReader, .{PublicationCtx{
        .control = control,
        .saw_zero = &saw_zero,
    }});
    try std.testing.expectEqual(@as(u32, 1), try control.grow(1));
    reader.join();
    try std.testing.expect(saw_zero);
    try std.testing.expectEqual(@as(u32, 2), control.pageCount());
}

test "SharedMemory: wait APIs validate bounds and alignment" {
    if (!platform.supports_reserved_memory) return error.SkipZigTest;
    const control = try Control.create(1, 2, std.testing.allocator);
    defer {
        std.debug.assert(control.release());
        control.destroy(std.testing.allocator);
    }
    try std.testing.expectError(error.Unaligned, control.wait32(1, 0, 0));
    try std.testing.expectError(error.OutOfBounds, control.wait64(wasm_page_size, 0, 0));
    try std.testing.expectError(error.OutOfBounds, control.notify(wasm_page_size, 1));
}
