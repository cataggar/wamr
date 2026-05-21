const std = @import("std");
const builtin = @import("builtin");
const host_trampolines = @import("../runtime/aot/host_trampolines.zig");

fn linuxMappingContainsAddress(allocator: std.mem.Allocator, addr: usize) !bool {
    if (builtin.os.tag != .linux) return true;

    const fd = try std.posix.openat(std.posix.AT.FDCWD, "/proc/self/maps", .{}, 0);
    defer _ = std.posix.system.close(fd);

    const contents = try allocator.alloc(u8, 1 << 20);
    defer allocator.free(contents);

    var total: usize = 0;
    while (total < contents.len) {
        const amt = try std.posix.read(fd, contents[total..]);
        if (amt == 0) break;
        total += amt;
    }

    var lines = std.mem.tokenizeScalar(u8, contents[0..total], '\n');
    while (lines.next()) |line| {
        var fields = std.mem.tokenizeScalar(u8, line, ' ');
        const range = fields.next() orelse continue;
        const dash = std.mem.indexOfScalar(u8, range, '-') orelse continue;
        const start = std.fmt.parseUnsigned(usize, range[0..dash], 16) catch continue;
        const end = std.fmt.parseUnsigned(usize, range[dash + 1 ..], 16) catch continue;
        if (addr >= start and addr < end) return true;
    }

    return false;
}

test "#648 phase 1: trampoline pool allocates mmap-backed stub slots" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var pool = try host_trampolines.TrampolinePool.init(allocator);

    const base_addr = @intFromPtr(pool.memory.ptr);
    try std.testing.expectEqual(@as(usize, 0), base_addr & (std.heap.page_size_min - 1));
    try std.testing.expect(try linuxMappingContainsAddress(allocator, base_addr));

    var fake_component_0: usize = 0x10;
    var fake_component_1: usize = 0x20;
    var fake_component_2: usize = 0x30;
    var fake_component_3: usize = 0x40;

    const stub0 = try pool.allocSlot(@ptrCast(&fake_component_0), 11);
    const stub1 = try pool.allocSlot(@ptrCast(&fake_component_1), 22);
    const stub2 = try pool.allocSlot(@ptrCast(&fake_component_2), 33);
    const stub3 = try pool.allocSlot(@ptrCast(&fake_component_3), 44);

    host_trampolines.setActivePool(&pool);
    defer host_trampolines.setActivePool(null);

    const addrs = [_]usize{
        @intFromPtr(stub0),
        @intFromPtr(stub1),
        @intFromPtr(stub2),
        @intFromPtr(stub3),
    };
    inline for (0..addrs.len) |i| {
        inline for (i + 1..addrs.len) |j| {
            try std.testing.expect(addrs[i] != addrs[j]);
        }
    }

    for (addrs, 0..) |addr, i| {
        try std.testing.expect(addr >= base_addr);
        try std.testing.expect(addr < base_addr + pool.memory.len);
        try std.testing.expectEqual(@as(usize, i) * host_trampolines.STUB_BYTES, addr - base_addr);
        const byte: *const u8 = @ptrFromInt(addr);
        _ = byte.*;
    }

    try std.testing.expectEqual(@as(u32, 4), pool.next_slot);
    try std.testing.expectEqual(@as(u32, 22), pool.slots[1].canon_lower_idx);
    try std.testing.expectEqual(@intFromPtr(&fake_component_2), @intFromPtr(pool.slots[2].component_inst));
    try std.testing.expectEqual(@as(u64, 0), host_trampolines.genericDispatcher(3, 1, 2, 3, 4, 5, 6));

    host_trampolines.setActivePool(null);
    pool.deinit(allocator);
    try std.testing.expect(!(try linuxMappingContainsAddress(allocator, base_addr)));
}
