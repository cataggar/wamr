//! Requested backing-allocation accounting; no page/OS or compiler dependency.
const std = @import("std");

pub const Allocator = struct {
    parent: std.mem.Allocator,
    limit: usize,
    live: usize = 0,
    peak: usize = 0,

    pub fn allocator(self: *Allocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn account(self: *Allocator, old: usize, new: usize) void {
        self.live = self.live - old + new;
        self.peak = @max(self.peak, self.live);
    }
    fn alloc(ctx: *anyopaque, length: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *Allocator = @ptrCast(@alignCast(ctx));
        if (length > self.limit - self.live) return null;
        const p = self.parent.rawAlloc(length, alignment, ra) orelse return null;
        self.account(0, length);
        return p;
    }
    fn resize(ctx: *anyopaque, bytes: []u8, alignment: std.mem.Alignment, length: usize, ra: usize) bool {
        const self: *Allocator = @ptrCast(@alignCast(ctx));
        if (length > bytes.len and length - bytes.len > self.limit - self.live) return false;
        if (!self.parent.rawResize(bytes, alignment, length, ra)) return false;
        self.account(bytes.len, length);
        return true;
    }
    fn remap(ctx: *anyopaque, bytes: []u8, alignment: std.mem.Alignment, length: usize, ra: usize) ?[*]u8 {
        const self: *Allocator = @ptrCast(@alignCast(ctx));
        if (length > bytes.len and length - bytes.len > self.limit - self.live) return null;
        const p = self.parent.rawRemap(bytes, alignment, length, ra) orelse return null;
        self.account(bytes.len, length);
        return p;
    }
    fn free(ctx: *anyopaque, bytes: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *Allocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(bytes, alignment, ra);
        self.account(bytes.len, 0);
    }
};
