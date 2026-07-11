//! #879 M4.6 (phase 1): standalone tests for `lazy_call_trampoline`'s
//! native trampoline mechanism.
//!
//! These deliberately do NOT go through any real wasm compile/execute
//! path (that's phase 2, not yet wired) -- they validate the raw stub
//! mechanism directly, the same way `runtime.zig`'s `JitCodeCache`
//! tests exercise `mapCodeExecutable` with a hand-written
//! `minimal_ret_body` payload instead of real codegen output.
const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

const trampoline_mod = wamr.lazy_call_trampoline;
const platform = wamr.platform;

const can_run = builtin.cpu.arch == .x86_64;

/// Hand-written native payload: `rax = rdi + rsi + rdx + rcx + r8 + r9;
/// ret`. Used to prove every argument-passing register survives the
/// trampoline's save-dispatch-restore-jmp round trip untouched, not
/// just the first one or two.
///
///   mov rax, rdi   48 89 F8
///   add rax, rsi   48 01 F0
///   add rax, rdx   48 01 D0
///   add rax, rcx   48 01 C8
///   add rax, r8    4C 01 C0
///   add rax, r9    4C 01 C8
///   ret            C3
const sum_regs_payload = [_]u8{
    0x48, 0x89, 0xF8,
    0x48, 0x01, 0xF0,
    0x48, 0x01, 0xD0,
    0x48, 0x01, 0xC8,
    0x4C, 0x01, 0xC0,
    0x4C, 0x01, 0xC8,
    0xC3,
};

/// Hand-written native payload: `rax = 0x2A (=42); ret`. Used where the
/// test only needs a distinct, easily-recognized target -- not
/// argument-passing behaviour.
///
///   mov eax, 42    B8 2A 00 00 00
///   ret            C3
const const42_payload = [_]u8{ 0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3 };

/// Test-owned dispatch context: maps `local_idx` to a mapped-executable
/// target address, and records the most recent `(ctx, local_idx)` the
/// dispatch function was actually invoked with, so a test can assert
/// on both "did the trampoline jump to the right place" and "did it
/// call the dispatcher with the arguments it was configured with".
const TestCtx = struct {
    targets: [4]usize = .{ 0, 0, 0, 0 },
    calls: u32 = 0,
    last_local_idx: u32 = 0,
};

fn testDispatch(ctx_opaque: *anyopaque, local_idx: u32) callconv(.c) usize {
    const ctx: *TestCtx = @ptrCast(@alignCast(ctx_opaque));
    ctx.calls += 1;
    ctx.last_local_idx = local_idx;
    return ctx.targets[local_idx];
}

test "lazy_call_trampoline: dispatch is invoked with the baked-in (ctx, local_idx) and jumps to its resolved address" {
    if (comptime !can_run) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    const mapped_const42 = platform.mapExecutableCode(&const42_payload) orelse return error.MapFailed;
    defer platform.munmap(mapped_const42, const42_payload.len);

    var ctx = TestCtx{};
    ctx.targets[2] = @intFromPtr(mapped_const42);

    var pool = try trampoline_mod.LazyCallTrampolinePool.init(gpa);
    defer pool.deinit();

    const stub = try pool.allocSlot(&ctx, 2, &testDispatch);
    const as_fn: *const fn () callconv(.c) i64 = @ptrCast(stub);

    try std.testing.expectEqual(@as(i64, 42), as_fn());
    try std.testing.expectEqual(@as(u32, 1), ctx.calls);
    try std.testing.expectEqual(@as(u32, 2), ctx.last_local_idx);
}

test "lazy_call_trampoline: every argument-passing register survives the save/dispatch/restore/jmp round trip" {
    if (comptime !can_run) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    const mapped_sum = platform.mapExecutableCode(&sum_regs_payload) orelse return error.MapFailed;
    defer platform.munmap(mapped_sum, sum_regs_payload.len);

    var ctx = TestCtx{};
    ctx.targets[0] = @intFromPtr(mapped_sum);

    var pool = try trampoline_mod.LazyCallTrampolinePool.init(gpa);
    defer pool.deinit();

    const stub = try pool.allocSlot(&ctx, 0, &testDispatch);
    const as_fn: *const fn (i64, i64, i64, i64, i64, i64) callconv(.c) i64 = @ptrCast(stub);

    // Distinct prime-ish values per slot so a transposition bug (e.g.
    // rdx and rcx swapped) would still change the sum in a way that's
    // easy to notice, even though the payload just sums everything.
    const result = as_fn(1, 20, 300, 4_000, 50_000, 600_000);
    try std.testing.expectEqual(@as(i64, 1 + 20 + 300 + 4_000 + 50_000 + 600_000), result);
}

test "lazy_call_trampoline: two independently-allocated slots resolve independently" {
    if (comptime !can_run) return error.SkipZigTest;

    const gpa = std.testing.allocator;

    const mapped_sum = platform.mapExecutableCode(&sum_regs_payload) orelse return error.MapFailed;
    defer platform.munmap(mapped_sum, sum_regs_payload.len);
    const mapped_const42 = platform.mapExecutableCode(&const42_payload) orelse return error.MapFailed;
    defer platform.munmap(mapped_const42, const42_payload.len);

    var ctx = TestCtx{};
    ctx.targets[0] = @intFromPtr(mapped_sum);
    ctx.targets[1] = @intFromPtr(mapped_const42);

    var pool = try trampoline_mod.LazyCallTrampolinePool.init(gpa);
    defer pool.deinit();

    const stub_sum = try pool.allocSlot(&ctx, 0, &testDispatch);
    const stub_const = try pool.allocSlot(&ctx, 1, &testDispatch);

    const sum_fn: *const fn (i64, i64, i64, i64, i64, i64) callconv(.c) i64 = @ptrCast(stub_sum);
    const const_fn: *const fn () callconv(.c) i64 = @ptrCast(stub_const);

    try std.testing.expectEqual(@as(i64, 1 + 2 + 3 + 4 + 5 + 6), sum_fn(1, 2, 3, 4, 5, 6));
    try std.testing.expectEqual(@as(i64, 42), const_fn());
    try std.testing.expectEqual(@as(u32, 2), ctx.calls);
    try std.testing.expectEqual(@as(u32, 1), ctx.last_local_idx); // most recent call was slot 1
}

test "lazy_call_trampoline: pool refuses to allocate past its configured cap" {
    if (comptime !can_run) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    var ctx = TestCtx{};

    var pool = try trampoline_mod.LazyCallTrampolinePool.initWithCap(gpa, 1);
    defer pool.deinit();

    _ = try pool.allocSlot(&ctx, 0, &testDispatch);
    try std.testing.expectError(error.OutOfTrampolineSlots, pool.allocSlot(&ctx, 0, &testDispatch));
}
