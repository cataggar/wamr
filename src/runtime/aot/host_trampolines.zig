//! AOT host-import trampoline pool.
//!
//! Each slot contains a tiny arch-specific shim that bakes in the slot index
//! and forwards the flat core ABI arguments to `genericDispatcher`.

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("../../platform/platform.zig");
const core_types = @import("../common/types.zig");

const page_size = std.heap.page_size_min;
const x86_64_stub_bytes: usize = 39;
const aarch64_stub_bytes: usize = 48;

pub const STUB_BYTES: usize = switch (builtin.cpu.arch) {
    .x86_64 => x86_64_stub_bytes,
    .aarch64 => aarch64_stub_bytes,
    else => 1,
};
pub const MAX_SLOTS: u32 = 256;

const StubFn = *const fn () callconv(.c) void;

pub const LoweredSig = struct {
    param_types: []const core_types.ValType,
    result_types: []const core_types.ValType,
    has_retptr: bool = false,
};

pub const DispatchResult = extern struct {
    status: u32,
    value: u64,
};

extern fn wamrAotDispatchComponentTrampoline(
    ctx_opaque: *anyopaque,
    lowered_sig: *const LoweredSig,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
) callconv(.c) DispatchResult;

pub const Slot = struct {
    component_inst: *anyopaque,
    canon_lower_idx: u32,
    ctx: ?*anyopaque = null,
    lowered_sig: LoweredSig = .{
        .param_types = &.{},
        .result_types = &.{},
        .has_retptr = false,
    },
};

var g_active_pool: ?*TrampolinePool = null;

pub fn setActivePool(pool: ?*TrampolinePool) void {
    g_active_pool = pool;
}

pub fn genericDispatcher(slot: u32, a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) callconv(.c) u64 {
    const pool = g_active_pool orelse return 0;
    if (slot >= pool.next_slot) return 0;

    const entry = pool.slots[slot];
    const ctx = entry.ctx orelse return 0;
    const dispatched = wamrAotDispatchComponentTrampoline(ctx, &entry.lowered_sig, a0, a1, a2, a3, a4, a5);
    if (dispatched.status != 0) return 0;
    return dispatched.value;
}

pub const TrampolinePool = struct {
    memory: []align(page_size) u8,
    slots: []Slot,
    next_slot: u32 = 0,

    pub fn init(allocator: std.mem.Allocator) !TrampolinePool {
        if (builtin.os.tag == .windows) return error.UnsupportedPlatform;
        switch (builtin.cpu.arch) {
            .x86_64, .aarch64 => {},
            else => return error.UnsupportedPlatform,
        }
        // macOS aarch64 forbids RWX mmap without MAP_JIT + pthread_jit_write_protect_np;
        // not worth wiring up for stdio-echo's needs. AOT layer falls back to interp.
        if (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64) return error.UnsupportedPlatform;

        const slots = try allocator.alloc(Slot, MAX_SLOTS);
        errdefer allocator.free(slots);

        const map_len = std.mem.alignForward(usize, @as(usize, MAX_SLOTS) * STUB_BYTES, page_size);
        const memory = try std.posix.mmap(
            null,
            map_len,
            .{ .READ = true, .WRITE = true, .EXEC = true },
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        );
        @memset(memory, 0);

        return .{
            .memory = memory,
            .slots = slots,
        };
    }

    pub fn allocSlotWithCtx(self: *TrampolinePool, ctx: *anyopaque, lowered_sig: LoweredSig) !StubFn {
        const slot = self.next_slot;
        if (slot >= MAX_SLOTS) return error.OutOfTrampolineSlots;

        self.slots[slot] = .{
            .component_inst = ctx,
            .canon_lower_idx = 0,
            .ctx = ctx,
            .lowered_sig = lowered_sig,
        };
        self.next_slot += 1;
        writeStub(self.stubBytes(slot), slot);
        platform.icacheFlush(self.stubPtr(slot), STUB_BYTES);
        g_active_pool = self;

        return @ptrFromInt(@intFromPtr(self.stubPtr(slot)));
    }

    pub fn allocSlot(self: *TrampolinePool, component_inst: *anyopaque, canon_lower_idx: u32) !StubFn {
        const slot = self.next_slot;
        if (slot >= MAX_SLOTS) return error.OutOfTrampolineSlots;

        self.slots[slot] = .{
            .component_inst = component_inst,
            .canon_lower_idx = canon_lower_idx,
        };
        self.next_slot += 1;
        writeStub(self.stubBytes(slot), slot);
        platform.icacheFlush(self.stubPtr(slot), STUB_BYTES);
        g_active_pool = self;

        return @ptrFromInt(@intFromPtr(self.stubPtr(slot)));
    }

    pub fn deinit(self: *TrampolinePool, allocator: std.mem.Allocator) void {
        if (g_active_pool == self) g_active_pool = null;
        std.posix.munmap(self.memory);
        allocator.free(self.slots);
        self.* = undefined;
    }

    fn stubPtr(self: *TrampolinePool, slot: u32) [*]u8 {
        return self.memory.ptr + (@as(usize, slot) * STUB_BYTES);
    }

    fn stubBytes(self: *TrampolinePool, slot: u32) []u8 {
        const start = @as(usize, slot) * STUB_BYTES;
        return self.memory[start .. start + STUB_BYTES];
    }
};

fn writeStub(bytes: []u8, slot: u32) void {
    @memset(bytes, 0);
    switch (builtin.cpu.arch) {
        .x86_64 => encodeX8664Stub(bytes, slot, dispatcherAddr()),
        .aarch64 => encodeAarch64Stub(bytes, slot, dispatcherAddr()),
        else => unreachable,
    }
}

fn dispatcherAddr() usize {
    return @intFromPtr(&genericDispatcher);
}

fn encodeX8664Stub(bytes: []u8, slot: u32, dispatcher: usize) void {
    std.debug.assert(bytes.len >= STUB_BYTES);

    const prologue = [_]u8{
        0x41, 0x51,
        0x4D, 0x89,
        0xC1, 0x49,
        0x89, 0xC8,
        0x48, 0x89,
        0xD1, 0x48,
        0x89, 0xF2,
        0x48, 0x89,
        0xFE, 0xBF,
    };
    const movabs = [_]u8{ 0x48, 0xB8 };
    const epilogue = [_]u8{
        0xFF, 0xD0,
        0x48, 0x83,
        0xC4, 0x08,
        0xC3,
    };

    var cursor: usize = 0;
    @memcpy(bytes[cursor .. cursor + prologue.len], &prologue);
    cursor += prologue.len;
    writeIntLittle(u32, bytes[cursor .. cursor + 4], slot);
    cursor += 4;
    @memcpy(bytes[cursor .. cursor + movabs.len], &movabs);
    cursor += movabs.len;
    writeIntLittle(u64, bytes[cursor .. cursor + 8], @intCast(dispatcher));
    cursor += 8;
    @memcpy(bytes[cursor .. cursor + epilogue.len], &epilogue);
    cursor += epilogue.len;

    std.debug.assert(cursor == x86_64_stub_bytes);
}

fn encodeAarch64Stub(bytes: []u8, slot: u32, dispatcher: usize) void {
    std.debug.assert(bytes.len >= STUB_BYTES);

    var cursor: usize = 0;
    inline for ([_]struct { dst: u5, src: u5 }{
        .{ .dst = 6, .src = 5 },
        .{ .dst = 5, .src = 4 },
        .{ .dst = 4, .src = 3 },
        .{ .dst = 3, .src = 2 },
        .{ .dst = 2, .src = 1 },
        .{ .dst = 1, .src = 0 },
    }) |move| {
        emitAarch64(bytes, &cursor, 0xAA0003E0 | (@as(u32, move.src) << 16) | move.dst);
    }

    emitAarch64(bytes, &cursor, movz32(0, @truncate(slot), 0));

    const addr = @as(u64, @intCast(dispatcher));
    emitAarch64(bytes, &cursor, movz64(16, @truncate(addr), 0));
    emitAarch64(bytes, &cursor, movk64(16, @truncate(addr >> 16), 1));
    emitAarch64(bytes, &cursor, movk64(16, @truncate(addr >> 32), 2));
    emitAarch64(bytes, &cursor, movk64(16, @truncate(addr >> 48), 3));
    emitAarch64(bytes, &cursor, 0xD61F0200);

    std.debug.assert(cursor == aarch64_stub_bytes);
}

fn emitAarch64(bytes: []u8, cursor: *usize, word: u32) void {
    writeIntLittle(u32, bytes[cursor.* .. cursor.* + 4], word);
    cursor.* += 4;
}

fn movz32(rd: u5, imm16: u16, shift: u2) u32 {
    return 0x52800000 | (@as(u32, shift) << 21) | (@as(u32, imm16) << 5) | rd;
}

fn movz64(rd: u5, imm16: u16, shift: u2) u32 {
    return 0xD2800000 | (@as(u32, shift) << 21) | (@as(u32, imm16) << 5) | rd;
}

fn movk64(rd: u5, imm16: u16, shift: u2) u32 {
    return 0xF2800000 | (@as(u32, shift) << 21) | (@as(u32, imm16) << 5) | rd;
}

fn writeIntLittle(comptime T: type, bytes: []u8, value: T) void {
    std.mem.writeInt(T, bytes[0..@sizeOf(T)], value, .little);
}

test "#648 phase 2: x86_64 trampoline encoder emits slot and dispatcher immediates" {
    var bytes: [aarch64_stub_bytes]u8 = undefined;
    @memset(&bytes, 0);

    encodeX8664Stub(&bytes, 0x11223344, 0x1122334455667788);

    const expected = [_]u8{
        0x41, 0x51,
        0x4D, 0x89,
        0xC1, 0x49,
        0x89, 0xC8,
        0x48, 0x89,
        0xD1, 0x48,
        0x89, 0xF2,
        0x48, 0x89,
        0xFE, 0xBF,
        0x44, 0x33,
        0x22, 0x11,
        0x48, 0xB8,
        0x88, 0x77,
        0x66, 0x55,
        0x44, 0x33,
        0x22, 0x11,
        0xFF, 0xD0,
        0x48, 0x83,
        0xC4, 0x08,
        0xC3,
    };

    try std.testing.expectEqualSlices(u8, &expected, bytes[0..expected.len]);
    for (bytes[expected.len..]) |byte| {
        try std.testing.expectEqual(@as(u8, 0), byte);
    }
}

test "#648 phase 2: aarch64 trampoline encoder emits slot and dispatcher immediates" {
    var bytes: [aarch64_stub_bytes]u8 = undefined;
    @memset(&bytes, 0);

    encodeAarch64Stub(&bytes, 0x1234, 0x1122334455667788);

    const expected = [_]u32{
        0xAA0503E6,
        0xAA0403E5,
        0xAA0303E4,
        0xAA0203E3,
        0xAA0103E2,
        0xAA0003E1,
        0x52824680,
        0xD28EF110,
        0xF2AAACD0,
        0xF2C66890,
        0xF2E22450,
        0xD61F0200,
    };

    for (expected, 0..) |word, idx| {
        const start = idx * @sizeOf(u32);
        try std.testing.expectEqual(word, std.mem.readInt(u32, bytes[start..][0..4], .little));
    }
}
