//! Stable shared linear-memory backing and atomic size publication.

const std = @import("std");
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
    parking_lot: parking.ParkingLot = .init(),

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
        platform.commitPages(base, initial_bytes) catch return error.CommitFailed;

        const control = allocator.create(Control) catch return error.OutOfMemory;
        control.* = .{
            .base = base,
            .reserved_bytes = reserved_bytes,
            .max_pages = max_pages,
            .current_pages = .init(initial_pages),
            .current_bytes = .init(initial_bytes),
        };
        return control;
    }

    pub fn destroy(self: *Control, allocator: std.mem.Allocator) void {
        self.parking_lot.deinit();
        platform.releaseAddressSpace(self.base, self.reserved_bytes);
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
        if (offset & (@alignOf(u32) - 1) != 0) return error.Unaligned;
        if (!rangeInBounds(offset, @sizeOf(u32), self.byteLen())) return error.OutOfBounds;
        const address: *align(@alignOf(u32)) const u32 =
            @ptrCast(@alignCast(self.base + offset));
        return self.parking_lot.wait32(address, expected, timeout_ns);
    }

    pub fn wait64(
        self: *Control,
        offset: usize,
        expected: u64,
        timeout_ns: i64,
    ) WaitError!parking.WaitResult {
        if (offset & (@alignOf(u64) - 1) != 0) return error.Unaligned;
        if (!rangeInBounds(offset, @sizeOf(u64), self.byteLen())) return error.OutOfBounds;
        const address: *align(@alignOf(u64)) const u64 =
            @ptrCast(@alignCast(self.base + offset));
        return self.parking_lot.wait64(address, expected, timeout_ns);
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
