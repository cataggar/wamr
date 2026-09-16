//! Requested-byte accounting for allocations made through the embedding caller.
//! Not allocator backing-page capacity, physical residency or whole-guest RAM.
const std = @import("std");

pub const Counter = struct {
    child: std.mem.Allocator,
    live: usize = 0,
    peak: usize = 0,
    fail_allocations: bool = false,

    pub fn allocator(self: *Counter) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{
            .alloc = alloc,
            .resize = resize,
            .remap = remap,
            .free = free,
        } };
    }

    fn update(self: *Counter, old: usize, new: usize) void {
        self.live = self.live - old + new;
        self.peak = @max(self.peak, self.live);
    }

    fn alloc(raw: *anyopaque, length: usize, alignment: std.mem.Alignment, address: usize) ?[*]u8 {
        const self: *Counter = @ptrCast(@alignCast(raw));
        if (self.fail_allocations) return null;
        const result = self.child.rawAlloc(length, alignment, address) orelse return null;
        self.update(0, length);
        return result;
    }

    fn resize(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, length: usize, address: usize) bool {
        const self: *Counter = @ptrCast(@alignCast(raw));
        if (!self.child.rawResize(memory, alignment, length, address)) return false;
        self.update(memory.len, length);
        return true;
    }

    fn remap(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, length: usize, address: usize) ?[*]u8 {
        const self: *Counter = @ptrCast(@alignCast(raw));
        const result = self.child.rawRemap(memory, alignment, length, address) orelse return null;
        self.update(memory.len, length);
        return result;
    }

    fn free(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, address: usize) void {
        const self: *Counter = @ptrCast(@alignCast(raw));
        self.child.rawFree(memory, alignment, address);
        self.update(memory.len, 0);
    }
};

test "native requested allocation counter tracks realloc and complete release" {
    var counter: Counter = .{ .child = std.testing.allocator };
    const allocator = counter.allocator();
    var bytes = try allocator.alloc(u8, 64);
    try std.testing.expectEqual(@as(usize, 64), counter.live);
    bytes = try allocator.realloc(bytes, 128);
    try std.testing.expectEqual(@as(usize, 128), counter.live);
    try std.testing.expect(counter.peak >= counter.live);
    allocator.free(bytes);
    try std.testing.expectEqual(@as(usize, 0), counter.live);
}
