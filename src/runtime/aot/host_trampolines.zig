//! AOT host-import trampoline pool scaffold.
//!
//! Phase 1 keeps the mapping RW-only and returns stub function pointers; later
//! phases will write real machine code and flip the mapping executable.

const std = @import("std");
const builtin = @import("builtin");
const core_types = @import("../common/types.zig");

const page_size = std.heap.page_size_min;

pub const STUB_BYTES: usize = 32;
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

        const slots = try allocator.alloc(Slot, MAX_SLOTS);
        errdefer allocator.free(slots);

        const map_len = std.mem.alignForward(usize, @as(usize, MAX_SLOTS) * STUB_BYTES, page_size);
        const memory = try std.posix.mmap(
            null,
            map_len,
            .{ .READ = true, .WRITE = true },
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
        writeStub(self.stubBytes(slot));
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
        writeStub(self.stubBytes(slot));
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

fn writeStub(bytes: []u8) void {
    @memset(bytes, 0);
    const head = switch (builtin.cpu.arch) {
        .x86_64 => &[_]u8{0xC3},
        .aarch64 => &[_]u8{ 0xC0, 0x03, 0x5F, 0xD6 },
        else => &[_]u8{0x00},
    };
    @memcpy(bytes[0..head.len], head);
}
