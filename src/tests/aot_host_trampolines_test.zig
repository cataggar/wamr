const std = @import("std");
const builtin = @import("builtin");
const host_trampolines = @import("../runtime/aot/host_trampolines.zig");
const executor = @import("../component/executor.zig");
const instance = @import("../component/instance.zig");
const ctypes = @import("../component/types.zig");
const core_types = @import("../runtime/common/types.zig");

const memory_core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d,
    0x01, 0x00, 0x00, 0x00,
    0x05, 0x03, 0x01, 0x00,
    0x01,
};
const empty_core_inst_args = [_]ctypes.CoreInstantiateArg{};
const memory_core_modules = [_]ctypes.CoreModule{.{ .data = &memory_core_wasm }};
const memory_core_insts = [_]ctypes.CoreInstanceExpr{.{ .instantiate = .{ .module_idx = 0, .args = &empty_core_inst_args } }};
const memory_component = ctypes.Component{
    .core_modules = &memory_core_modules,
    .core_instances = &memory_core_insts,
    .core_types = &.{},
    .components = &.{},
    .instances = &.{},
    .aliases = &.{},
    .types = &.{},
    .canons = &.{},
    .imports = &.{},
    .exports = &.{},
};

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

fn instantiateMemoryComponent() !*instance.ComponentInstance {
    return instance.instantiate(&memory_component, std.testing.allocator);
}

fn expectStoredPtrLen(mem: []const u8, offset: u32, expected_ptr: u32, expected_len: u32) !void {
    try std.testing.expectEqual(expected_ptr, std.mem.readInt(u32, mem[offset..][0..4], .little));
    try std.testing.expectEqual(expected_len, std.mem.readInt(u32, mem[offset + 4 ..][0..4], .little));
}

test "#648 phase 1: trampoline pool allocates mmap-backed stub slots" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;

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
    // Slot 3 was allocated via `allocSlot` (no ctx), so genericDispatcher
    // hits the `entry.ctx orelse` arm and returns DISPATCH_FAILURE_SENTINEL
    // (#708). Pre-708 this returned 0, which looked like a legitimate empty
    // handle / zero-length value to the guest.
    try std.testing.expectEqual(
        host_trampolines.DISPATCH_FAILURE_SENTINEL,
        host_trampolines.genericDispatcher(3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    );

    host_trampolines.setActivePool(null);
    pool.deinit(allocator);
    if (builtin.os.tag == .linux) {
        try std.testing.expect(!(try linuxMappingContainsAddress(allocator, base_addr)));
    }
}

test "#648 phase 2: trampoline slots execute through genericDispatcher" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;
    switch (builtin.cpu.arch) {
        .x86_64, .aarch64 => {},
        else => return error.SkipZigTest,
    }

    const Host = struct {
        const State = struct {
            base: u64,
            called: bool = false,
        };

        fn sum(ctx: ?*anyopaque, _: *instance.ComponentInstance, args: []const instance.InterfaceValue, results: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            const state: *State = @ptrCast(@alignCast(ctx.?));
            try std.testing.expectEqual(@as(usize, 6), args.len);
            try std.testing.expectEqual(@as(usize, 1), results.len);

            var total = state.base;
            for (args, 1..) |arg, expected| {
                try std.testing.expectEqual(@as(u64, expected), arg.u64);
                total += arg.u64;
            }

            state.called = true;
            results[0] = .{ .u64 = total };
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    const param_types = [_]ctypes.ValType{ .u64, .u64, .u64, .u64, .u64, .u64 };
    const result_types = [_]ctypes.ValType{.u64};
    const lowered_params = [_]core_types.ValType{ .i64, .i64, .i64, .i64, .i64, .i64 };
    const lowered_results = [_]core_types.ValType{.i64};

    var state0 = Host.State{ .base = 10 };
    var ctx0 = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .context = @ptrCast(&state0), .call = &Host.sum },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{},
    };
    const stub0 = try pool.allocSlotWithCtx(@ptrCast(&ctx0), .{ .param_types = &lowered_params, .result_types = &lowered_results });

    var state1 = Host.State{ .base = 20 };
    var ctx1 = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .context = @ptrCast(&state1), .call = &Host.sum },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{},
    };
    const stub1 = try pool.allocSlotWithCtx(@ptrCast(&ctx1), .{ .param_types = &lowered_params, .result_types = &lowered_results });

    const TrampolineFn = *const fn (u64, u64, u64, u64, u64, u64) callconv(.c) u64;
    const tramp0: TrampolineFn = @ptrCast(stub0);
    const tramp1: TrampolineFn = @ptrCast(stub1);

    try std.testing.expectEqual(@as(u64, 31), tramp0(1, 2, 3, 4, 5, 6));
    try std.testing.expectEqual(@as(u64, 41), tramp1(1, 2, 3, 4, 5, 6));
    try std.testing.expect(state0.called);
    try std.testing.expect(state1.called);
}

test "#648 phase 3: genericDispatcher handles (i32) -> ()" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;

    const Host = struct {
        const State = struct { called: bool = false, arg: i32 = 0 };

        fn drop(ctx: ?*anyopaque, _: *instance.ComponentInstance, args: []const instance.InterfaceValue, results: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            const state: *State = @ptrCast(@alignCast(ctx.?));
            try std.testing.expectEqual(@as(usize, 1), args.len);
            try std.testing.expectEqual(@as(usize, 0), results.len);
            state.called = true;
            state.arg = args[0].s32;
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    var state = Host.State{};
    const param_types = [_]ctypes.ValType{.s32};
    const result_types = [_]ctypes.ValType{};
    var ctx = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .context = @ptrCast(&state), .call = &Host.drop },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{},
    };
    const lowered_params = [_]core_types.ValType{.i32};
    const lowered_results = [_]core_types.ValType{};
    _ = try pool.allocSlotWithCtx(@ptrCast(&ctx), .{ .param_types = &lowered_params, .result_types = &lowered_results });

    try std.testing.expectEqual(@as(u64, 0), host_trampolines.genericDispatcher(0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    try std.testing.expect(state.called);
    try std.testing.expectEqual(@as(i32, 41), state.arg);
}

test "#648 phase 3: genericDispatcher handles () -> i32" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;

    const Host = struct {
        fn get(_: ?*anyopaque, _: *instance.ComponentInstance, args: []const instance.InterfaceValue, results: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            try std.testing.expectEqual(@as(usize, 0), args.len);
            try std.testing.expectEqual(@as(usize, 1), results.len);
            results[0] = .{ .s32 = 77 };
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    const param_types = [_]ctypes.ValType{};
    const result_types = [_]ctypes.ValType{.s32};
    var ctx = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .call = &Host.get },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{},
    };
    const lowered_params = [_]core_types.ValType{};
    const lowered_results = [_]core_types.ValType{.i32};
    _ = try pool.allocSlotWithCtx(@ptrCast(&ctx), .{ .param_types = &lowered_params, .result_types = &lowered_results });

    try std.testing.expectEqual(@as(u64, 77), host_trampolines.genericDispatcher(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

test "#648 phase 3: genericDispatcher handles (i32) -> i32" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;

    const Host = struct {
        fn subscribe(_: ?*anyopaque, _: *instance.ComponentInstance, args: []const instance.InterfaceValue, results: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            try std.testing.expectEqual(@as(i32, 12), args[0].s32);
            results[0] = .{ .s32 = args[0].s32 + 5 };
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    const param_types = [_]ctypes.ValType{.s32};
    const result_types = [_]ctypes.ValType{.s32};
    var ctx = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .call = &Host.subscribe },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{},
    };
    const lowered_params = [_]core_types.ValType{.i32};
    const lowered_results = [_]core_types.ValType{.i32};
    _ = try pool.allocSlotWithCtx(@ptrCast(&ctx), .{ .param_types = &lowered_params, .result_types = &lowered_results });

    try std.testing.expectEqual(@as(u64, 17), host_trampolines.genericDispatcher(0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

test "#648 phase 3: genericDispatcher handles (i32 i32) -> () retptr" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;

    const Host = struct {
        fn checkWrite(_: ?*anyopaque, _: *instance.ComponentInstance, args: []const instance.InterfaceValue, results: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            try std.testing.expectEqual(@as(i32, 9), args[0].s32);
            results[0] = .{ .string = .{ .ptr = 0x55, .len = 7 } };
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    const param_types = [_]ctypes.ValType{.s32};
    const result_types = [_]ctypes.ValType{.string};
    var ctx = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .call = &Host.checkWrite },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{ .memory_idx = 0 },
    };
    const lowered_params = [_]core_types.ValType{.i32};
    _ = try pool.allocSlotWithCtx(@ptrCast(&ctx), .{ .param_types = &lowered_params, .result_types = &.{}, .has_retptr = true });

    const retptr: u32 = 24;
    try std.testing.expectEqual(@as(u64, 0), host_trampolines.genericDispatcher(0, 9, retptr, 0, 0, 0, 0, 0, 0, 0, 0));
    const mem = inst.resolveTopLevelMemory(0).?.data;
    try expectStoredPtrLen(mem, retptr, 0x55, 7);
}

test "#648 phase 3: genericDispatcher handles (i32 i32 i32 i32) -> ()" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;

    const Host = struct {
        fn write(_: ?*anyopaque, _: *instance.ComponentInstance, args: []const instance.InterfaceValue, results: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            try std.testing.expectEqual(@as(i32, 3), args[0].s32);
            try std.testing.expectEqual(@as(u32, 0x40), args[1].list.ptr);
            try std.testing.expectEqual(@as(u32, 5), args[1].list.len);
            results[0] = .{ .string = .{ .ptr = 0x80, .len = 5 } };
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    const param_types = [_]ctypes.ValType{ .s32, .{ .list = 0 } };
    const result_types = [_]ctypes.ValType{.string};
    var ctx = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .call = &Host.write },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{ .memory_idx = 0 },
    };
    const lowered_params = [_]core_types.ValType{ .i32, .i32, .i32 };
    _ = try pool.allocSlotWithCtx(@ptrCast(&ctx), .{ .param_types = &lowered_params, .result_types = &.{}, .has_retptr = true });

    const retptr: u32 = 40;
    try std.testing.expectEqual(@as(u64, 0), host_trampolines.genericDispatcher(0, 3, 0x40, 5, retptr, 0, 0, 0, 0, 0, 0));
    const mem = inst.resolveTopLevelMemory(0).?.data;
    try expectStoredPtrLen(mem, retptr, 0x80, 5);
}

test "#648 phase 3: genericDispatcher handles (i32 i64 i32) -> ()" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;

    const Host = struct {
        fn blockingRead(_: ?*anyopaque, _: *instance.ComponentInstance, args: []const instance.InterfaceValue, results: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            try std.testing.expectEqual(@as(i32, 4), args[0].s32);
            try std.testing.expectEqual(@as(u64, 0x1_0000_0002), args[1].u64);
            results[0] = .{ .string = .{ .ptr = 0x90, .len = 2 } };
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    const param_types = [_]ctypes.ValType{ .s32, .u64 };
    const result_types = [_]ctypes.ValType{.string};
    var ctx = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .call = &Host.blockingRead },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{ .memory_idx = 0 },
    };
    const lowered_params = [_]core_types.ValType{ .i32, .i64 };
    _ = try pool.allocSlotWithCtx(@ptrCast(&ctx), .{ .param_types = &lowered_params, .result_types = &.{}, .has_retptr = true });

    const retptr: u32 = 56;
    try std.testing.expectEqual(@as(u64, 0), host_trampolines.genericDispatcher(0, 4, 0x1_0000_0002, retptr, 0, 0, 0, 0, 0, 0, 0));
    const mem = inst.resolveTopLevelMemory(0).?.data;
    try expectStoredPtrLen(mem, retptr, 0x90, 2);
}

test "#689: trampoline stub forwards 7 i32 args (widest WASIp2 method shape)" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;
    switch (builtin.cpu.arch) {
        .x86_64, .aarch64 => {},
        else => return error.SkipZigTest,
    }

    // 7 i32 wasm params is what wasi:filesystem/types.[method]descriptor.link-at
    // lowers to: enough to push caller_a6 onto the dispatcher's stack frame on
    // x86_64 SysV (only 6 reg args). On AArch64 all 7 fit in x0..x6 but the
    // injected slot pushes x6 -> x7, exercising the widened shift sequence.
    const Host = struct {
        const State = struct { received: [7]i32 = [_]i32{0} ** 7, called: bool = false };

        fn capture(ctx: ?*anyopaque, _: *instance.ComponentInstance, args: []const instance.InterfaceValue, results: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            const state: *State = @ptrCast(@alignCast(ctx.?));
            try std.testing.expectEqual(@as(usize, 7), args.len);
            try std.testing.expectEqual(@as(usize, 1), results.len);
            var sum: i32 = 0;
            for (args, 0..) |a, i| {
                state.received[i] = a.s32;
                sum +%= a.s32;
            }
            state.called = true;
            results[0] = .{ .s32 = sum };
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    const param_types = [_]ctypes.ValType{ .s32, .s32, .s32, .s32, .s32, .s32, .s32 };
    const result_types = [_]ctypes.ValType{.s32};
    const lowered_params = [_]core_types.ValType{ .i32, .i32, .i32, .i32, .i32, .i32, .i32 };
    const lowered_results = [_]core_types.ValType{.i32};

    var state = Host.State{};
    var ctx = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .context = @ptrCast(&state), .call = &Host.capture },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{},
    };
    const stub = try pool.allocSlotWithCtx(@ptrCast(&ctx), .{ .param_types = &lowered_params, .result_types = &lowered_results });

    const TrampolineFn = *const fn (u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
    const tramp: TrampolineFn = @ptrCast(stub);

    const result = tramp(101, 202, 303, 404, 505, 606, 707);

    try std.testing.expect(state.called);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 101, 202, 303, 404, 505, 606, 707 }, &state.received);
    // 101+202+303+404+505+606+707 = 2828
    try std.testing.expectEqual(@as(u64, 2828), result);
}

test "#689: trampoline stub forwards 8 i32 args (cap)" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;
    switch (builtin.cpu.arch) {
        .x86_64, .aarch64 => {},
        else => return error.SkipZigTest,
    }

    // 8 i32 wasm params is the new cap installed by #689 — exercises both
    // stack-arg slots on x86_64 (caller a6, a7) and the stack-spill arg on
    // AArch64 (caller a7 lands at [sp+8] after the stub's push).
    const Host = struct {
        const State = struct { received: [8]i32 = [_]i32{0} ** 8, called: bool = false };

        fn capture(ctx: ?*anyopaque, _: *instance.ComponentInstance, args: []const instance.InterfaceValue, results: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            const state: *State = @ptrCast(@alignCast(ctx.?));
            try std.testing.expectEqual(@as(usize, 8), args.len);
            try std.testing.expectEqual(@as(usize, 1), results.len);
            var sum: i32 = 0;
            for (args, 0..) |a, i| {
                state.received[i] = a.s32;
                sum +%= a.s32;
            }
            state.called = true;
            results[0] = .{ .s32 = sum };
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    const param_types = [_]ctypes.ValType{ .s32, .s32, .s32, .s32, .s32, .s32, .s32, .s32 };
    const result_types = [_]ctypes.ValType{.s32};
    const lowered_params = [_]core_types.ValType{ .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32 };
    const lowered_results = [_]core_types.ValType{.i32};

    var state = Host.State{};
    var ctx = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .context = @ptrCast(&state), .call = &Host.capture },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{},
    };
    const stub = try pool.allocSlotWithCtx(@ptrCast(&ctx), .{ .param_types = &lowered_params, .result_types = &lowered_results });

    const TrampolineFn = *const fn (u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
    const tramp: TrampolineFn = @ptrCast(stub);

    const result = tramp(1, 2, 4, 8, 16, 32, 64, 128);

    try std.testing.expect(state.called);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 4, 8, 16, 32, 64, 128 }, &state.received);
    try std.testing.expectEqual(@as(u64, 255), result);
}

// --- #708: non-zero sentinel + loud state-machine warns ---

test "#708: genericDispatcher returns sentinel on host-fn error" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;

    const Host = struct {
        fn failing(_: ?*anyopaque, _: *instance.ComponentInstance, _: []const instance.InterfaceValue, _: []instance.InterfaceValue, _: std.mem.Allocator) !void {
            return error.OutOfMemory;
        }
    };

    const inst = try instantiateMemoryComponent();
    defer inst.deinit();
    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    host_trampolines.setActivePool(&pool);
    defer host_trampolines.setActivePool(null);

    const param_types = [_]ctypes.ValType{.s32};
    const result_types = [_]ctypes.ValType{};
    var ctx = executor.ComponentTrampolineCtx{
        .comp_inst = inst,
        .host_func = .{ .call = &Host.failing },
        .param_types = &param_types,
        .result_types = &result_types,
        .lower_opts = .{},
    };
    const lowered_params = [_]core_types.ValType{.i32};
    const lowered_results = [_]core_types.ValType{};
    _ = try pool.allocSlotWithCtx(@ptrCast(&ctx), .{ .param_types = &lowered_params, .result_types = &lowered_results });

    // Pre-708 this collapsed to 0; the guest saw a legitimate-looking zero
    // handle / length / discriminant and threaded the failure several more
    // ops before crashing somewhere unrelated. Sentinel makes the first
    // downstream use trip a wit-bindgen `unreachable` with a stack
    // pointing at this import.
    try std.testing.expectEqual(
        host_trampolines.DISPATCH_FAILURE_SENTINEL,
        host_trampolines.genericDispatcher(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    );
}

test "#708: genericDispatcher returns sentinel when called with no active pool" {
    // Belt-and-braces — `setActivePool(null)` simulates the AOT trampoline
    // firing after instance teardown.
    host_trampolines.setActivePool(null);
    try std.testing.expectEqual(
        host_trampolines.DISPATCH_FAILURE_SENTINEL,
        host_trampolines.genericDispatcher(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    );
}

test "#708: genericDispatcher returns sentinel for out-of-range slot index" {
    if (builtin.os.tag == .windows or (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64)) return error.SkipZigTest;

    var pool = try host_trampolines.TrampolinePool.init(std.testing.allocator);
    defer pool.deinit(std.testing.allocator);

    host_trampolines.setActivePool(&pool);
    defer host_trampolines.setActivePool(null);

    // next_slot is 0 — any slot index is OOR.
    try std.testing.expectEqual(
        host_trampolines.DISPATCH_FAILURE_SENTINEL,
        host_trampolines.genericDispatcher(7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    );
}
