const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");
const aot_harness = @import("aot_harness.zig");

const allocator = std.testing.allocator;

const ThreadStartKind = enum {
    normal,
    immediate,
    trap,
    exit,
    missing,
    wrong_signature,
};

fn encodeUleb(out: *std.ArrayList(u8), value: u32) !void {
    var remaining = value;
    while (true) {
        var byte: u8 = @intCast(remaining & 0x7f);
        remaining >>= 7;
        if (remaining != 0) byte |= 0x80;
        try out.append(allocator, byte);
        if (remaining == 0) return;
    }
}

fn encodeSleb(out: *std.ArrayList(u8), value: i32) !void {
    var remaining: i64 = value;
    while (true) {
        const byte: u8 = @truncate(@as(u64, @bitCast(remaining)));
        const payload = byte & 0x7f;
        remaining >>= 7;
        const done = (remaining == 0 and payload & 0x40 == 0) or
            (remaining == -1 and payload & 0x40 != 0);
        const continuation: u8 = if (done) 0 else 0x80;
        try out.append(allocator, payload | continuation);
        if (done) return;
    }
}

fn appendName(out: *std.ArrayList(u8), name: []const u8) !void {
    try encodeUleb(out, @intCast(name.len));
    try out.appendSlice(allocator, name);
}

fn appendSection(out: *std.ArrayList(u8), id: u8, payload: []const u8) !void {
    try out.append(allocator, id);
    try encodeUleb(out, @intCast(payload.len));
    try out.appendSlice(allocator, payload);
}

fn appendI32Const(out: *std.ArrayList(u8), value: i32) !void {
    try out.append(allocator, 0x41);
    try encodeSleb(out, value);
}

fn appendLocalGet(out: *std.ArrayList(u8), index: u32) !void {
    try out.append(allocator, 0x20);
    try encodeUleb(out, index);
}

fn appendCall(out: *std.ArrayList(u8), index: u32) !void {
    try out.append(allocator, 0x10);
    try encodeUleb(out, index);
}

fn appendI32Store(out: *std.ArrayList(u8)) !void {
    try out.appendSlice(allocator, &.{ 0x36, 0x02, 0x00 });
}

fn appendI32Load(out: *std.ArrayList(u8)) !void {
    try out.appendSlice(allocator, &.{ 0x28, 0x02, 0x00 });
}

fn appendStoreConst(out: *std.ArrayList(u8), address: i32, value: i32) !void {
    try appendI32Const(out, address);
    try appendI32Const(out, value);
    try appendI32Store(out);
}

fn appendStoreLocal(out: *std.ArrayList(u8), address: i32, local: u32) !void {
    try appendI32Const(out, address);
    try appendLocalGet(out, local);
    try appendI32Store(out);
}

fn appendStorePayloadWord(
    out: *std.ArrayList(u8),
    destination: i32,
    start_arg_local: u32,
    payload_offset: i32,
) !void {
    try appendI32Const(out, destination);
    try appendLocalGet(out, start_arg_local);
    if (payload_offset != 0) {
        try appendI32Const(out, payload_offset);
        try out.append(allocator, 0x6a); // i32.add
    }
    try appendI32Load(out);
    try appendI32Store(out);
}

fn appendFdstatResult(out: *std.ArrayList(u8), destination: i32) !void {
    try appendI32Const(out, destination);
    try appendI32Const(out, 1);
    try appendI32Const(out, 64);
    try appendCall(out, 1);
    try appendI32Store(out);
}

fn appendAtomicCoverage(out: *std.ArrayList(u8)) !void {
    try appendI32Const(out, 48);
    try appendI32Const(out, 5);
    try out.appendSlice(allocator, &.{ 0xfe, 0x17, 0x02, 0x00 }); // store

    try appendI32Const(out, 52);
    try appendI32Const(out, 48);
    try out.appendSlice(allocator, &.{ 0xfe, 0x10, 0x02, 0x00 }); // load
    try appendI32Store(out);

    try appendI32Const(out, 56);
    try appendI32Const(out, 48);
    try appendI32Const(out, 5);
    try appendI32Const(out, 9);
    try out.appendSlice(allocator, &.{ 0xfe, 0x48, 0x02, 0x00 }); // cmpxchg
    try appendI32Store(out);

    try appendI32Const(out, 60);
    try appendI32Const(out, 48);
    try out.appendSlice(allocator, &.{ 0xfe, 0x10, 0x02, 0x00 }); // load
    try appendI32Store(out);

    // Keep unrelated values live across cmpxchg's rdx scratch use.
    try appendI32Const(out, 96);
    try appendI32Const(out, 10);
    try appendI32Const(out, 20);
    try appendI32Const(out, 48);
    try appendI32Const(out, 9);
    try appendI32Const(out, 11);
    try out.appendSlice(allocator, &.{ 0xfe, 0x48, 0x02, 0x00 });
    try out.append(allocator, 0x1a);
    try out.appendSlice(allocator, &.{ 0x6a, 0x36, 0x02, 0x00 });

    // Logical RMW uses rdx+r8 for its CAS loop; keep three values live.
    try appendI32Const(out, 100);
    try appendI32Const(out, 1);
    try appendI32Const(out, 2);
    try appendI32Const(out, 3);
    try appendI32Const(out, 48);
    try appendI32Const(out, 3);
    try out.appendSlice(allocator, &.{ 0xfe, 0x3a, 0x02, 0x00 });
    try out.append(allocator, 0x1a);
    try out.appendSlice(allocator, &.{ 0x6a, 0x6a, 0x36, 0x02, 0x00 });

    try appendI32Const(out, 104);
    try appendI32Const(out, 48);
    try out.appendSlice(allocator, &.{ 0xfe, 0x10, 0x02, 0x00 });
    try appendI32Store(out);
    try out.appendSlice(allocator, &.{ 0xfe, 0x03, 0x00 }); // fence
}

fn buildThreadFixture(kind: ThreadStartKind) ![]u8 {
    var wasm: std.ArrayList(u8) = .empty;
    errdefer wasm.deinit(allocator);
    try wasm.appendSlice(allocator, &.{
        0x00, 0x61, 0x73, 0x6d,
        0x01, 0x00, 0x00, 0x00,
    });

    var types: std.ArrayList(u8) = .empty;
    defer types.deinit(allocator);
    try encodeUleb(&types, 5);
    try types.appendSlice(allocator, &.{ 0x60, 0x01, 0x7f, 0x01, 0x7f });
    try types.appendSlice(allocator, &.{ 0x60, 0x00, 0x00 });
    try types.appendSlice(allocator, &.{ 0x60, 0x02, 0x7f, 0x7f, 0x00 });
    try types.appendSlice(allocator, &.{ 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f });
    try types.appendSlice(allocator, &.{ 0x60, 0x01, 0x7f, 0x00 });
    try appendSection(&wasm, 1, types.items);

    var imports: std.ArrayList(u8) = .empty;
    defer imports.deinit(allocator);
    try encodeUleb(&imports, 3);
    try appendName(&imports, "wasi");
    try appendName(&imports, "thread-spawn");
    try imports.appendSlice(allocator, &.{ 0x00, 0x00 });
    try appendName(&imports, "wasi_snapshot_preview1");
    try appendName(&imports, "fd_fdstat_get");
    try imports.appendSlice(allocator, &.{ 0x00, 0x03 });
    try appendName(&imports, "wasi_snapshot_preview1");
    try appendName(&imports, "proc_exit");
    try imports.appendSlice(allocator, &.{ 0x00, 0x04 });
    try appendSection(&wasm, 2, imports.items);

    var functions: std.ArrayList(u8) = .empty;
    defer functions.deinit(allocator);
    try encodeUleb(&functions, 2);
    try functions.append(allocator, 1);
    try functions.append(allocator, if (kind == .wrong_signature) 1 else 2);
    try appendSection(&wasm, 3, functions.items);

    var memory: std.ArrayList(u8) = .empty;
    defer memory.deinit(allocator);
    try memory.appendSlice(allocator, &.{ 0x01, 0x03, 0x01, 0x01 });
    try appendSection(&wasm, 5, memory.items);

    var exports: std.ArrayList(u8) = .empty;
    defer exports.deinit(allocator);
    try encodeUleb(&exports, if (kind == .missing) 2 else 3);
    try appendName(&exports, "memory");
    try exports.appendSlice(allocator, &.{ 0x02, 0x00 });
    try appendName(&exports, "_start");
    try exports.appendSlice(allocator, &.{ 0x00, 0x03 });
    if (kind != .missing) {
        try appendName(&exports, "wasi_thread_start");
        try exports.appendSlice(allocator, &.{ 0x00, 0x04 });
    }
    try appendSection(&wasm, 7, exports.items);

    var start_body: std.ArrayList(u8) = .empty;
    defer start_body.deinit(allocator);
    try start_body.append(allocator, 0); // local decls
    try appendStoreConst(&start_body, 256, 0x4000);
    try appendStoreConst(&start_body, 260, 0x5000);
    try appendStoreConst(&start_body, 264, 0x6000);
    try appendStoreConst(&start_body, 268, 0x7000);
    try appendI32Const(&start_body, 28);
    try appendI32Const(&start_body, 256);
    try appendCall(&start_body, 0);
    try appendI32Store(&start_body);
    try start_body.append(allocator, 0x0b);

    var thread_body: std.ArrayList(u8) = .empty;
    defer thread_body.deinit(allocator);
    try thread_body.append(allocator, 0); // local decls
    if (kind == .trap) {
        try thread_body.appendSlice(allocator, &.{ 0x00, 0x0b });
    } else if (kind == .exit) {
        try appendI32Const(&thread_body, 7);
        try appendCall(&thread_body, 2);
        try thread_body.append(allocator, 0x0b);
    } else if (kind == .wrong_signature or kind == .immediate) {
        try thread_body.append(allocator, 0x0b);
    } else {
        try thread_body.appendSlice(allocator, &.{
            0x41, 0x00, // i32.const counter address
            0x41, 0x01, // i32.const increment
            0xfe, 0x1e, 0x02, 0x00, // i32.atomic.rmw.add
            0x1a, // drop old value
        });
        try appendLocalGet(&thread_body, 1);
        try appendI32Const(&thread_body, 256);
        try thread_body.appendSlice(allocator, &.{ 0x46, 0x04, 0x40 }); // eq; if
        try appendStoreLocal(&thread_body, 4, 0);
        try appendStoreLocal(&thread_body, 8, 1);
        try appendStorePayloadWord(&thread_body, 32, 1, 0);
        try appendStorePayloadWord(&thread_body, 36, 1, 4);
        try appendAtomicCoverage(&thread_body);
        try appendI32Const(&thread_body, 12);
        try appendI32Const(&thread_body, 264);
        try appendCall(&thread_body, 0);
        try appendI32Store(&thread_body);
        try appendFdstatResult(&thread_body, 20);
        try thread_body.append(allocator, 0x05); // else
        try appendStoreLocal(&thread_body, 16, 1);
        try appendStorePayloadWord(&thread_body, 40, 1, 0);
        try appendStorePayloadWord(&thread_body, 44, 1, 4);
        try appendFdstatResult(&thread_body, 24);
        try thread_body.appendSlice(allocator, &.{ 0x0b, 0x0b }); // end if/function
    }

    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try encodeUleb(&code, 2);
    try encodeUleb(&code, @intCast(start_body.items.len));
    try code.appendSlice(allocator, start_body.items);
    try encodeUleb(&code, @intCast(thread_body.items.len));
    try code.appendSlice(allocator, thread_body.items);
    try appendSection(&wasm, 10, code.items);

    return wasm.toOwnedSlice(allocator);
}

fn readU32(memory: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, memory[offset..][0..4], .little);
}

fn runStart(harness: *aot_harness.Harness) !void {
    const start = harness.findFuncExport("_start") orelse
        return error.FunctionNotFound;
    var results: [0]wamr.aot_runtime.ScalarResult = .{};
    _ = try harness.callScalar(start, &.{}, &results);
}

fn requireAotThreads() !void {
    if (!wamr.config.lib_wasi_threads or !wamr.config.aot or
        !aot_harness.can_exec_aot or
        builtin.single_threaded)
        return error.SkipZigTest;
}

test "Preview-1 AOT spawn shares memory, WASI descriptors, and start/TLS payloads" {
    try requireAotThreads();
    const wasm = try buildThreadFixture(.normal);
    defer allocator.free(wasm);
    const harness = try aot_harness.Harness.init(allocator, wasm);
    defer harness.deinit();

    const wasi_ctx = try wamr.WasiCtx.init(allocator, std.testing.io);
    defer wasi_ctx.deinit();
    try wasi_ctx.setArgs(&.{ "aot-thread-test", "arg" });
    harness.inst.attachProcessState(wasi_ctx.processStateRef());

    var manager = wamr.thread_manager.ThreadManager.init(allocator);
    defer manager.deinit();
    harness.inst.setThreadManager(&manager);

    try runStart(harness);
    const summary = manager.joinAllWithSummary();
    try std.testing.expectEqual(@as(usize, 2), summary.joined);
    try std.testing.expectEqual(@as(usize, 0), summary.trapped);
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());

    const memory = harness.inst.memories[0].data;
    try std.testing.expectEqual(@as(u32, 2), readU32(memory, 0));
    const outer_tid = readU32(memory, 4);
    const nested_tid = readU32(memory, 12);
    try std.testing.expect(outer_tid > 0 and outer_tid < (1 << 29));
    try std.testing.expect(nested_tid > 0 and nested_tid < (1 << 29));
    try std.testing.expect(outer_tid != nested_tid);
    try std.testing.expectEqual(@as(u32, 256), readU32(memory, 8));
    try std.testing.expectEqual(@as(u32, 264), readU32(memory, 16));
    try std.testing.expectEqual(@as(u32, 0), readU32(memory, 20));
    try std.testing.expectEqual(@as(u32, 0), readU32(memory, 24));
    try std.testing.expectEqual(outer_tid, readU32(memory, 28));
    try std.testing.expectEqual(@as(u32, 0x4000), readU32(memory, 32));
    try std.testing.expectEqual(@as(u32, 0x5000), readU32(memory, 36));
    try std.testing.expectEqual(@as(u32, 0x6000), readU32(memory, 40));
    try std.testing.expectEqual(@as(u32, 0x7000), readU32(memory, 44));
    try std.testing.expectEqual(@as(u32, 8), readU32(memory, 48));
    try std.testing.expectEqual(@as(u32, 5), readU32(memory, 52));
    try std.testing.expectEqual(@as(u32, 5), readU32(memory, 56));
    try std.testing.expectEqual(@as(u32, 9), readU32(memory, 60));
    try std.testing.expectEqual(@as(u32, 30), readU32(memory, 96));
    try std.testing.expectEqual(@as(u32, 6), readU32(memory, 100));
    try std.testing.expectEqual(@as(u32, 8), readU32(memory, 104));
}

test "Preview-1 AOT child traps translate to a joined trapped outcome" {
    try requireAotThreads();
    const wasm = try buildThreadFixture(.trap);
    defer allocator.free(wasm);
    const harness = try aot_harness.Harness.init(allocator, wasm);
    defer harness.deinit();
    var manager = wamr.thread_manager.ThreadManager.init(allocator);
    defer manager.deinit();
    harness.inst.setThreadManager(&manager);

    try runStart(harness);
    const summary = manager.joinAllWithSummary();
    try std.testing.expectEqual(@as(usize, 1), summary.joined);
    try std.testing.expectEqual(@as(usize, 1), summary.trapped);
    try std.testing.expect(manager.hasTrap());
}

test "Preview-1 AOT immediate child exit remains joinable" {
    try requireAotThreads();
    const wasm = try buildThreadFixture(.immediate);
    defer allocator.free(wasm);
    const harness = try aot_harness.Harness.init(allocator, wasm);
    defer harness.deinit();
    var manager = wamr.thread_manager.ThreadManager.init(allocator);
    defer manager.deinit();
    harness.inst.setThreadManager(&manager);

    try runStart(harness);
    const summary = manager.shutdownWithSummary();
    try std.testing.expectEqual(@as(usize, 1), summary.joined);
    try std.testing.expectEqual(@as(usize, 0), summary.trapped);
    try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
    try std.testing.expect(manager.isShuttingDown());
}

test "Preview-1 AOT child proc_exit traps without terminating the host" {
    try requireAotThreads();
    const wasm = try buildThreadFixture(.exit);
    defer allocator.free(wasm);
    const harness = try aot_harness.Harness.init(allocator, wasm);
    defer harness.deinit();
    const wasi_ctx = try wamr.WasiCtx.init(allocator, std.testing.io);
    defer wasi_ctx.deinit();
    harness.inst.attachProcessState(wasi_ctx.processStateRef());
    var manager = wamr.thread_manager.ThreadManager.init(allocator);
    defer manager.deinit();
    harness.inst.setThreadManager(&manager);

    try runStart(harness);
    const summary = manager.joinAllWithSummary();
    try std.testing.expectEqual(@as(usize, 1), summary.joined);
    try std.testing.expectEqual(@as(usize, 1), summary.trapped);
    try std.testing.expectEqual(@as(?u32, 7), wasi_ctx.getExitCode());
    try std.testing.expect(manager.hasTrap());
}

test "Preview-1 AOT spawn rejects missing and invalid thread start exports" {
    try requireAotThreads();
    inline for (.{ ThreadStartKind.missing, ThreadStartKind.wrong_signature }) |kind| {
        const wasm = try buildThreadFixture(kind);
        defer allocator.free(wasm);
        const harness = try aot_harness.Harness.init(allocator, wasm);
        defer harness.deinit();
        var manager = wamr.thread_manager.ThreadManager.init(allocator);
        defer manager.deinit();
        harness.inst.setThreadManager(&manager);

        try runStart(harness);
        try std.testing.expectEqual(std.math.maxInt(u32), readU32(harness.inst.memories[0].data, 28));
        try std.testing.expectEqual(@as(usize, 0), manager.retainedCount());
    }
}
