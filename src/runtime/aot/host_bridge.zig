//! AOT host function bridge.
//!
//! Provides C-calling-convention adapter functions that bridge AOT-compiled
//! code to the shared WASI core logic.  Each adapter receives a VmCtx pointer
//! as first argument (same as all AOT functions), followed by the WASI
//! function's arguments in registers, and returns the WASI errno as i32.

const std = @import("std");
const VmCtx = @import("runtime.zig").VmCtx;
const wasi_core = @import("../../wasi/wasi_core.zig");

// ── Helpers ───────────────────────────────────────────────────────────

/// Reconstruct a mutable memory slice from the VmCtx fields.
fn getMemoryFromCtx(vmctx: *VmCtx) ?[]u8 {
    if (vmctx.memory_base == 0 or vmctx.memory_size == 0) return null;
    const ptr: [*]u8 = @ptrFromInt(vmctx.memory_base);
    return ptr[0..vmctx.memory_size];
}

// ── AOT adapters ──────────────────────────────────────────────────────

pub fn aotFdWrite(vmctx: *VmCtx, fd: i32, iovs_ptr: i32, iovs_len: i32, nwritten_ptr: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return wasi_core.fdWriteCore(mem, fd, @bitCast(iovs_ptr), @bitCast(iovs_len), @bitCast(nwritten_ptr));
}

pub fn aotFdSeek(vmctx: *VmCtx, _: i32, _: i64, _: i32, _: i32) callconv(.c) i32 {
    _ = vmctx;
    return wasi_core.fdSeekCore();
}

pub fn aotFdClose(vmctx: *VmCtx, fd: i32) callconv(.c) i32 {
    _ = vmctx;
    return wasi_core.fdCloseCore(fd);
}

pub fn aotFdFdstatGet(vmctx: *VmCtx, fd: i32, buf_ptr: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return wasi_core.fdFdstatGetCore(mem, fd, @bitCast(buf_ptr));
}

pub fn aotFdPrestatGet(vmctx: *VmCtx, _: i32, _: i32) callconv(.c) i32 {
    _ = vmctx;
    return wasi_core.fdPrestatGetCore();
}

pub fn aotFdPrestatDirName(vmctx: *VmCtx, _: i32, _: i32, _: i32) callconv(.c) i32 {
    _ = vmctx;
    return wasi_core.fdPrestatDirNameCore();
}

pub fn aotClockTimeGet(vmctx: *VmCtx, clock_id: i32, _: i64, time_ptr: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return wasi_core.clockTimeGetCore(mem, clock_id, @bitCast(time_ptr));
}

pub fn aotEnvironSizesGet(vmctx: *VmCtx, count_ptr: i32, buf_size_ptr: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return wasi_core.environSizesGetCore(mem, @bitCast(count_ptr), @bitCast(buf_size_ptr));
}

pub fn aotEnvironGet(vmctx: *VmCtx, _: i32, _: i32) callconv(.c) i32 {
    _ = vmctx;
    return wasi_core.environGetCore();
}

pub fn aotArgsSizesGet(vmctx: *VmCtx, count_ptr: i32, buf_size_ptr: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return wasi_core.argsSizesGetCore(mem, @bitCast(count_ptr), @bitCast(buf_size_ptr));
}

pub fn aotArgsGet(vmctx: *VmCtx, _: i32, _: i32) callconv(.c) i32 {
    _ = vmctx;
    return wasi_core.argsGetCore();
}

pub fn aotProcExit(vmctx: *VmCtx, _: i32) callconv(.c) void {
    _ = vmctx;
    wasi_core.procExitCore();
    // In AOT context, proc_exit is a no-op (trap cannot propagate via return value).
    // The compiled code will simply continue/return.
}

// ── Resolver ──────────────────────────────────────────────────────────

/// Resolve a WASI function name to an AOT adapter function pointer.
/// Returns null for unrecognised names.
pub fn resolveAotHostFunction(name: []const u8) ?*const anyopaque {
    const map = .{
        .{ "fd_write", @as(*const anyopaque, @ptrCast(&aotFdWrite)) },
        .{ "fd_seek", @as(*const anyopaque, @ptrCast(&aotFdSeek)) },
        .{ "fd_close", @as(*const anyopaque, @ptrCast(&aotFdClose)) },
        .{ "fd_fdstat_get", @as(*const anyopaque, @ptrCast(&aotFdFdstatGet)) },
        .{ "fd_prestat_get", @as(*const anyopaque, @ptrCast(&aotFdPrestatGet)) },
        .{ "fd_prestat_dir_name", @as(*const anyopaque, @ptrCast(&aotFdPrestatDirName)) },
        .{ "clock_time_get", @as(*const anyopaque, @ptrCast(&aotClockTimeGet)) },
        .{ "environ_sizes_get", @as(*const anyopaque, @ptrCast(&aotEnvironSizesGet)) },
        .{ "environ_get", @as(*const anyopaque, @ptrCast(&aotEnvironGet)) },
        .{ "args_sizes_get", @as(*const anyopaque, @ptrCast(&aotArgsSizesGet)) },
        .{ "args_get", @as(*const anyopaque, @ptrCast(&aotArgsGet)) },
        .{ "proc_exit", @as(*const anyopaque, @ptrCast(&aotProcExit)) },
    };

    inline for (map) |entry| {
        if (std.mem.eql(u8, name, entry[0])) return entry[1];
    }
    return null;
}

/// Check whether a module name indicates a WASI import.
pub fn isWasiModule(module_name: []const u8) bool {
    return std.mem.eql(u8, module_name, "wasi_snapshot_preview1") or
        std.mem.eql(u8, module_name, "wasi_unstable") or
        std.mem.eql(u8, module_name, "wasi");
}

// ── Tests ─────────────────────────────────────────────────────────────

test "resolveAotHostFunction: all known functions resolve" {
    const names = [_][]const u8{
        "fd_write",      "fd_seek",           "fd_close",
        "fd_fdstat_get", "fd_prestat_get",    "fd_prestat_dir_name",
        "clock_time_get", "environ_sizes_get", "environ_get",
        "args_sizes_get", "args_get",          "proc_exit",
    };
    for (names) |name| {
        try std.testing.expect(resolveAotHostFunction(name) != null);
    }
}

test "resolveAotHostFunction: unknown returns null" {
    try std.testing.expect(resolveAotHostFunction("nonexistent") == null);
}

test "isWasiModule: recognises both module names" {
    try std.testing.expect(isWasiModule("wasi_snapshot_preview1"));
    try std.testing.expect(isWasiModule("wasi"));
    try std.testing.expect(!isWasiModule("env"));
}

test "aotFdWrite: returns EINVAL when no memory" {
    var vmctx = VmCtx{};
    const result = aotFdWrite(&vmctx, 1, 0, 0, 0);
    try std.testing.expectEqual(wasi_core.WASI_EINVAL, result);
}

test "aotFdWrite: writes to stdout with valid memory" {
    var mem = [_]u8{0} ** 128;
    // Set up one iov: buf_ptr=32, buf_len=5
    std.mem.writeInt(u32, mem[0..4], 32, .little);
    std.mem.writeInt(u32, mem[4..8], 5, .little);
    @memcpy(mem[32..37], "hello");

    var vmctx = VmCtx{
        .memory_base = @intFromPtr(&mem),
        .memory_size = mem.len,
    };

    const result = aotFdWrite(&vmctx, 1, 0, 1, 120);
    try std.testing.expectEqual(wasi_core.WASI_ESUCCESS, result);
    try std.testing.expectEqual(@as(u32, 5), wasi_core.memReadU32(&mem, 120).?);
}

test "aotFdClose: valid fds" {
    var vmctx = VmCtx{};
    try std.testing.expectEqual(wasi_core.WASI_ESUCCESS, aotFdClose(&vmctx, 1));
    try std.testing.expectEqual(wasi_core.WASI_EBADF, aotFdClose(&vmctx, 99));
}

test "aotClockTimeGet: returns time" {
    var mem = [_]u8{0} ** 16;
    var vmctx = VmCtx{
        .memory_base = @intFromPtr(&mem),
        .memory_size = mem.len,
    };
    const result = aotClockTimeGet(&vmctx, wasi_core.WASI_CLOCK_MONOTONIC, 0, 0);
    try std.testing.expectEqual(wasi_core.WASI_ESUCCESS, result);
    const nanos = std.mem.readInt(u64, mem[0..8], .little);
    try std.testing.expect(nanos > 0);
}

test "aotEnvironSizesGet: writes zeroes" {
    var mem = [_]u8{0xFF} ** 16;
    var vmctx = VmCtx{
        .memory_base = @intFromPtr(&mem),
        .memory_size = mem.len,
    };
    const result = aotEnvironSizesGet(&vmctx, 0, 4);
    try std.testing.expectEqual(wasi_core.WASI_ESUCCESS, result);
    try std.testing.expectEqual(@as(u32, 0), wasi_core.memReadU32(&mem, 0).?);
    try std.testing.expectEqual(@as(u32, 0), wasi_core.memReadU32(&mem, 4).?);
}
