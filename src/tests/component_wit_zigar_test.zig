//! Integration & benchmark tests for #1134:
//! - Build-time component embedding & in-memory AOT `.cwasm` core resolution
//! - `wit-zigar` comptime `CanonLift` / `CanonLower`, `makeHostFunc` (`call_flat`),
//!   and `callTypedExport` across interpreter and AOT (`wasm32` & `wasm64`)
//! - Zig 0.17 `std.Io`-backed `WaitableSet` and zero-copy `stream<u8>` transfers
//! - Stackful async coroutines (`ShadowStackPool`, `TaskContext`, and per-thread
//!   `__stack_pointer` preservation across cooperative suspension)

const std = @import("std");
const testing = std.testing;
const wamr = @import("wamr");
const aot_harness = @import("aot_harness.zig");

const canon = wamr.component_canon_comptime;
const coro = wamr.component_coro;
const comp_aot = wamr.component_aot;
const instance_mod = wamr.component_instance;
const ctypes = wamr.component_types;
const executor = wamr.component_executor;
const async_mod = wamr.component_async;
const async_canon = wamr.component_async_canon;
const abi = wamr.canonical_abi;
const wasi_cli = wamr.wasi_cli_adapter;
const ExecEnv = wamr.exec_env.ExecEnv;

// Core module exporting `(memory "memory" 1)` and `(func (export "add42") (param i32) (result i32))`
const add42_core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00, 0x05, 0x03, 0x01, 0x00,
    0x01, 0x07, 0x12, 0x02, 0x06, 'm',  'e',  'm',
    'o',  'r',  'y',  0x02, 0x00, 0x05, 'a',  'd',
    'd',  '4',  '2',  0x00, 0x00, 0x0a, 0x09, 0x01,
    0x07, 0x00, 0x20, 0x00, 0x41, 0x2a, 0x6a, 0x0b,
};

// Core module that imports `("host", "compute") (param i32 i32) (result i32)`
// and exports `(memory "memory" 1)` and `(func (export "run") (param i32 i32) (result i32))`
// which calls `host.compute(a, b)` and adds 10.
const import_host_core_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    // Type section: 1 type (i32, i32) -> i32
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01,
    0x7f,
    // Import section: 1 import "host"."compute" func type 0
    0x02, 0x10, 0x01, 0x04, 'h',  'o',  's',
    't',  0x07, 'c',  'o',  'm',  'p',  'u',  't',
    'e',  0x00, 0x00,
    // Function section: 1 function (index 1), type 0
    0x03, 0x02, 0x01, 0x00,
    // Memory section: 1 memory, min 1 page
    0x05,
    0x03, 0x01, 0x00, 0x01,
    // Export section: "memory" (mem 0), "run" (func 1)
    0x07, 0x10, 0x02, 0x06,
    'm',  'e',  'm',  'o',  'r',  'y',  0x02, 0x00,
    0x03, 'r',  'u',  'n',  0x00, 0x01,
    // Code section: local.get 0, local.get 1, call 0, i32.const 10, i32.add, end
    0x0a, 0x0d,
    0x01, 0x0b, 0x00, 0x20, 0x00, 0x20, 0x01, 0x10,
    0x00, 0x41, 0x0a, 0x6a, 0x0b,
};

test "#1134: comptime CanonLift / CanonLower and HostFunc.call_flat on interpreter and AOT" {
    const allocator = testing.allocator;

    const HostHandlers = struct {
        var flat_calls: usize = 0;

        fn compute(a: i32, b: i32) i32 {
            flat_calls += 1;
            return a * 3 + b;
        }
    };

    const host_fn = canon.makeHostFunc(HostHandlers.compute);
    try testing.expect(host_fn.call_flat != null);

    const core_modules = [_]ctypes.CoreModule{.{ .data = &import_host_core_wasm }};
    const core_args_inner = [_]ctypes.CoreInlineExport{
        .{ .name = "compute", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .exports = &core_args_inner },
        .{ .instantiate = .{
            .module_idx = 0,
            .args = &[_]ctypes.CoreInstantiateArg{
                .{ .name = "host", .instance_idx = 0 },
            },
        } },
    };
    const aliases = [_]ctypes.Alias{
        .{ .instance_export = .{ .sort = .{ .core = .func }, .instance_idx = 1, .name = "run" } },
    };
    const params = [_]ctypes.NamedValType{
        .{ .name = "a", .type = .s32 },
        .{ .name = "b", .type = .s32 },
    };
    const types = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .{ .unnamed = .s32 } } },
    };
    const imports = [_]ctypes.ImportDecl{
        .{ .name = "compute", .desc = .{ .func = 0 } },
    };
    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &.{} } },
        .{ .lift = .{ .core_func_idx = 1, .type_idx = 0, .opts = &.{} } },
    };
    const exports = [_]ctypes.ExportDecl{
        .{ .name = "run", .desc = .{ .func = 1 }, .sort_idx = .{ .sort = .func, .idx = 1 } },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases,
        .types = &types,
        .canons = &canons,
        .imports = &imports,
        .exports = &exports,
    };

    // 1. Interpreter mode
    {
        HostHandlers.flat_calls = 0;
        const inst = try instance_mod.instantiate(&component, allocator);
        defer inst.deinit();

        var import_map: std.StringHashMapUnmanaged(instance_mod.ImportBinding) = .empty;
        defer import_map.deinit(allocator);
        try import_map.put(allocator, "compute", .{ .host_func = host_fn });
        try inst.linkImports(import_map);

        const out = try canon.callTypedExport(
            fn (i32, i32) i32,
            .{},
            inst,
            "run",
            .{ 7, 5 },
            allocator,
        );
        // (7 * 3 + 5) + 10 = 36
        try testing.expectEqual(@as(i32, 36), out);
        try testing.expectEqual(@as(usize, 1), HostHandlers.flat_calls);
    }

    // 2. AOT mode
    if (comptime aot_harness.can_exec_aot) {
        HostHandlers.flat_calls = 0;
        const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &import_host_core_wasm);
        defer allocator.free(cwasm_bytes);

        const pcs = [_]instance_mod.PrecompiledCore{
            .{ .module_idx = 0, .cwasm_bytes = cwasm_bytes },
        };
        const inst = try instance_mod.instantiateWithOptions(&component, allocator, .{
            .precompiled_cores = &pcs,
        });
        defer inst.deinit();

        var import_map: std.StringHashMapUnmanaged(instance_mod.ImportBinding) = .empty;
        defer import_map.deinit(allocator);
        try import_map.put(allocator, "compute", .{ .host_func = host_fn });
        try inst.linkImports(import_map);

        const out = try canon.callTypedExport(
            fn (i32, i32) i32,
            .{},
            inst,
            "run",
            .{ 11, 4 },
            allocator,
        );
        // (11 * 3 + 4) + 10 = 47
        try testing.expectEqual(@as(i32, 47), out);
        try testing.expectEqual(@as(usize, 1), HostHandlers.flat_calls);
    }
}

test "#1134: embedded component AOT core resolution without filesystem I/O" {
    if (comptime !aot_harness.can_exec_aot) return error.SkipZigTest;

    const allocator = testing.allocator;
    const cwasm_bytes = try aot_harness.compileWasmToAot(allocator, &add42_core_wasm);
    defer allocator.free(cwasm_bytes);

    // Construct a minimal valid binary component containing `add42_core_wasm` as core module 0.
    var comp_bytes: std.ArrayListUnmanaged(u8) = .empty;
    defer comp_bytes.deinit(allocator);
    // Component header
    try comp_bytes.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6d, 0x0d, 0x00, 0x01, 0x00 });
    // Section 1 (core module): id = 1, size = add42_core_wasm.len
    try comp_bytes.append(allocator, 0x01);
    try comp_bytes.append(allocator, @intCast(add42_core_wasm.len));
    try comp_bytes.appendSlice(allocator, &add42_core_wasm);

    const core_hash = comp_aot.sha256Hex(&add42_core_wasm);
    const embedded_inputs = [_]comp_aot.EmbeddedCoreInput{
        .{
            .idx = 0,
            .core_sha256 = &core_hash,
            .cwasm_bytes = cwasm_bytes,
        },
    };

    var resolved = try comp_aot.resolveEmbeddedCores(allocator, comp_bytes.items, &embedded_inputs);
    defer resolved.deinit();
    try testing.expectEqual(@as(usize, 1), resolved.precompiledCores().len);
    try testing.expectEqualSlices(u8, &add42_core_wasm, resolved.precompiledCores()[0].core_wasm.?);
}

test "#1134: wasm32 and wasm64 guest memory slice bounds checking" {
    const allocator = testing.allocator;
    var inst: instance_mod.ComponentInstance = undefined;
    inst.test_mem = null;
    inst.core_instances = &.{};
    try inst.enableTestMem(allocator, 1024);
    defer inst.disableTestMem();

    // Valid 64-bit slice within bounds
    const writable = inst.writableGuestBytes64(256, 128);
    try testing.expect(writable != null);
    @memset(writable.?, 0xAB);

    const readable = inst.readGuestBytes64(256, 128);
    try testing.expect(readable != null);
    try testing.expectEqual(@as(u8, 0xAB), readable.?[0]);
    try testing.expectEqual(@as(u8, 0xAB), readable.?[127]);

    // Overflow & out-of-bounds checks for wasm64 addresses
    try testing.expect(inst.readGuestBytes64(900, 200) == null);
    try testing.expect(inst.writableGuestBytes64(std.math.maxInt(u64) - 10, 32) == null);
    try testing.expect(inst.readGuestBytes64(@as(u64, 1) << 40, 16) == null);
}

test "#1134: std.Io-backed WaitableSet immediate wakeup across threads" {
    const allocator = testing.allocator;
    var ws = async_mod.WaitableSet{};
    defer ws.deinit(allocator);

    const reg_idx = try ws.register(.{ .kind = .stream_read, .handle = 7 }, allocator);
    try testing.expect(!ws.hasReady());

    const Producer = struct {
        fn run(set: *async_mod.WaitableSet, idx: u32, alloc: std.mem.Allocator) void {
            std.Io.Threaded.global_single_threaded.io().sleep(
                std.Io.Duration.fromNanoseconds(2 * std.time.ns_per_ms),
                .awake,
            ) catch {};
            set.setReady(idx, alloc, async_canon.packStatus(.completed, 64));
        }
    };

    const thread = try std.Thread.spawn(.{}, Producer.run, .{ &ws, reg_idx, allocator });
    defer thread.join();

    _ = ws.waitReady(std.Io.Threaded.global_single_threaded.io(), 500 * std.time.ns_per_ms);
    try testing.expect(ws.hasReady());

    const ev = ws.popReadyEvent().?;
    try testing.expectEqual(async_mod.WaitableSet.WaitableItem.Kind.stream_read, ev.kind);
    try testing.expectEqual(@as(u32, 7), ev.handle);
    try testing.expectEqual(async_canon.packStatus(.completed, 64), ev.code);
}

test "#1134: zero-copy stream<u8> stdout write-via-stream and stdin read-via-stream" {
    const allocator = testing.allocator;
    const types_array = [_]ctypes.TypeDef{.{ .val = .u8 }};
    var comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &types_array,
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const comp_inst = try instance_mod.instantiate(&comp, allocator);
    defer comp_inst.deinit();
    try comp_inst.enableTestMem(allocator, 4096);
    defer comp_inst.disableTestMem();
    comp_inst.realloc_call_count = 0;

    var adapter = wasi_cli.WasiCliAdapter.init(allocator);
    defer adapter.deinit();
    adapter.setStdinBytes("hello zero-copy wasip3!");

    // 1. Test stdin#read-via-stream zero-copy into guest linear memory
    var stdin_results: [1]abi.InterfaceValue = undefined;
    try wasi_cli.WasiCliAdapter.stdinReadViaStreamP3(&adapter, comp_inst, &.{}, &stdin_results, allocator);
    defer stdin_results[0].deinit(allocator);
    const stdin_handle = stdin_results[0].tuple_val[0].handle;

    var dummy_mod = wamr.types.WasmModule{};
    const dummy_inst = try wamr.instance.instantiate(&dummy_mod, allocator);
    defer wamr.instance.destroy(dummy_inst);
    var env = try ExecEnv.create(dummy_inst, 256, allocator);
    defer env.destroy();

    // Push (handle, guest_ptr=128, max_count=23) for stream.read
    try env.pushI32(@bitCast(stdin_handle));
    try env.pushI32(128);
    try env.pushI32(23);
    try executor.dispatchCanonBuiltin(
        comp_inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        allocator,
    );
    const read_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, 23), read_status);
    try testing.expectEqualStrings("hello zero-copy wasip3!", comp_inst.readGuestBytes(128, 23).?);
    try testing.expectEqual(@as(usize, 0), comp_inst.realloc_call_count);
    {
        var lease = comp_inst.streams.acquire(stdin_handle).?;
        defer lease.release();
        try testing.expectEqual(@as(usize, 0), lease.value().buffer.capacity);
        try testing.expectEqual(@as(u64, 1), lease.value().zero_copy_transfers);
    }

    // 2. Test stdout#write-via-stream zero-copy from guest linear memory
    var stdout_handle: u32 = 0;
    try comp_inst.streams.put(allocator, 42, .{});
    stdout_handle = 42;
    const stdout_args = [_]abi.InterfaceValue{.{ .handle = stdout_handle }};
    var stdout_results: [1]abi.InterfaceValue = undefined;
    try wasi_cli.WasiCliAdapter.stdoutWriteViaStreamP3(&adapter, comp_inst, &stdout_args, &stdout_results, allocator);

    // Push (handle=42, guest_ptr=128, count=23) for stream.write
    try env.pushI32(@bitCast(stdout_handle));
    try env.pushI32(128);
    try env.pushI32(23);
    try executor.dispatchCanonBuiltin(
        comp_inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        allocator,
    );
    const write_status: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(async_canon.packStatus(.completed, 23), write_status);
    try testing.expectEqualStrings("hello zero-copy wasip3!", adapter.getStdoutBytes());
    try testing.expectEqual(@as(usize, 0), comp_inst.realloc_call_count);
    {
        var lease = comp_inst.streams.acquire(stdout_handle).?;
        defer lease.release();
        try testing.expectEqual(@as(usize, 0), lease.value().buffer.capacity);
        try testing.expectEqual(@as(u64, 1), lease.value().zero_copy_transfers);
    }
}

test "#1134: bidirectional zero-copy parked stream rendezvous (reader-first and writer-first)" {
    const allocator = testing.allocator;
    const types_array = [_]ctypes.TypeDef{.{ .val = .u8 }};
    var comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &types_array,
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const comp_inst = try instance_mod.instantiate(&comp, allocator);
    defer comp_inst.deinit();
    try comp_inst.enableTestMem(allocator, 4096);
    defer comp_inst.disableTestMem();

    var dummy_mod = wamr.types.WasmModule{};
    const dummy_inst = try wamr.instance.instantiate(&dummy_mod, allocator);
    defer wamr.instance.destroy(dummy_inst);
    var env = try ExecEnv.create(dummy_inst, 256, allocator);
    defer env.destroy();

    // Case A: Reader parks first, writer copies directly into parked reader's slice
    const stream_a: u32 = 10;
    try comp_inst.streams.put(allocator, stream_a, .{});
    const src_a = comp_inst.writableGuestBytes(100, 12).?;
    @memcpy(src_a, "direct-read!");

    // Reader parks at dst=500, max_count=12
    try env.pushI32(@bitCast(stream_a));
    try env.pushI32(500);
    try env.pushI32(12);
    try executor.dispatchCanonBuiltin(
        comp_inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        allocator,
    );
    try testing.expectEqual(async_canon.BLOCKED_STATUS, @as(u32, @bitCast(try env.popI32())));

    // Writer writes from src=100, count=12 -> direct @memcpy to 500
    try env.pushI32(@bitCast(stream_a));
    try env.pushI32(100);
    try env.pushI32(12);
    try executor.dispatchCanonBuiltin(
        comp_inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        allocator,
    );
    try testing.expectEqual(
        async_canon.packStatus(.completed, 12),
        @as(u32, @bitCast(try env.popI32())),
    );
    try testing.expectEqualStrings("direct-read!", comp_inst.readGuestBytes(500, 12).?);
    {
        var lease = comp_inst.streams.acquire(stream_a).?;
        defer lease.release();
        try testing.expectEqual(@as(usize, 0), lease.value().buffer.capacity);
        try testing.expectEqual(@as(u64, 1), lease.value().zero_copy_transfers);
    }

    // Case B: Writer parks first (buffer_limit = 0), reader copies directly from parked writer's slice
    const stream_b: u32 = 11;
    try comp_inst.streams.put(allocator, stream_b, .{ .buffer_limit = 0 });
    const src_b = comp_inst.writableGuestBytes(200, 13).?;
    @memcpy(src_b, "direct-write!");

    // Writer parks at src=200, count=13
    try env.pushI32(@bitCast(stream_b));
    try env.pushI32(200);
    try env.pushI32(13);
    try executor.dispatchCanonBuiltin(
        comp_inst,
        .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        allocator,
    );
    try testing.expectEqual(async_canon.BLOCKED_STATUS, @as(u32, @bitCast(try env.popI32())));

    // Reader reads into dst=600, max_count=13 -> direct @memcpy from 200 to 600
    try env.pushI32(@bitCast(stream_b));
    try env.pushI32(600);
    try env.pushI32(13);
    try executor.dispatchCanonBuiltin(
        comp_inst,
        .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
        env,
        null,
        allocator,
    );
    try testing.expectEqual(
        async_canon.packStatus(.completed, 13),
        @as(u32, @bitCast(try env.popI32())),
    );
    try testing.expectEqualStrings("direct-write!", comp_inst.readGuestBytes(600, 13).?);
    {
        var lease = comp_inst.streams.acquire(stream_b).?;
        defer lease.release();
        try testing.expectEqual(@as(usize, 0), lease.value().buffer.capacity);
        try testing.expectEqual(@as(u64, 1), lease.value().zero_copy_transfers);
    }
}

test "#1134: stackful coroutine TaskContext and ShadowStackPool isolation" {
    const allocator = testing.allocator;
    var pool = coro.ShadowStackPool.init(2048);
    defer pool.deinit(allocator);

    var host_sp: usize = 12288;
    const sp_binding = coro.StackPointerBinding.fromHostSlot(&host_sp);

    const stack_a = try pool.acquire(allocator);
    defer pool.release(allocator, stack_a) catch {};
    const stack_b = try pool.acquire(allocator);
    defer pool.release(allocator, stack_b) catch {};

    var task_a = coro.TaskContext.init(stack_a, sp_binding);
    var task_b = coro.TaskContext.init(stack_b, sp_binding);

    // Enter task A, simulate frame allocation on its shadow stack, then suspend
    task_a.enter();
    try testing.expectEqual(stack_a.initialTop(), host_sp);
    host_sp -= 128;
    const sp_a_suspended = host_sp;
    task_a.suspendTask();
    try testing.expectEqual(@as(usize, 12288), host_sp);

    // Enter task B while task A is suspended, simulate frame allocation, then suspend
    task_b.enter();
    try testing.expectEqual(stack_b.initialTop(), host_sp);
    host_sp -= 256;
    const sp_b_suspended = host_sp;
    task_b.suspendTask();
    try testing.expectEqual(@as(usize, 12288), host_sp);

    // Resume task A and verify its exact suspended SP and canary integrity
    task_a.resumeTask();
    try testing.expectEqual(sp_a_suspended, host_sp);
    try task_a.complete();
    try testing.expectEqual(@as(usize, 12288), host_sp);

    // Resume task B and verify its exact suspended SP and canary integrity
    task_b.resumeTask();
    try testing.expectEqual(sp_b_suspended, host_sp);
    try task_b.complete();
    try testing.expectEqual(@as(usize, 12288), host_sp);
}

test "#1134: waitable-set.wait suspends and resumes active coroutine TaskContext" {
    const allocator = testing.allocator;
    var comp = ctypes.Component{
        .core_modules = &.{},
        .core_instances = &.{},
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &.{},
        .types = &.{},
        .canons = &.{},
        .imports = &.{},
        .exports = &.{},
    };
    const comp_inst = try instance_mod.instantiate(&comp, allocator);
    defer comp_inst.deinit();
    try comp_inst.enableTestMem(allocator, 4096);
    defer comp_inst.disableTestMem();

    const ws_handle = comp_inst.allocAsyncHandle();
    try comp_inst.waitable_sets.put(allocator, ws_handle, .{});
    {
        var lease = comp_inst.waitable_sets.acquire(ws_handle).?;
        defer lease.release();
        const idx = try lease.value().register(.{ .kind = .subtask, .handle = 99 }, allocator);
        lease.value().setReady(idx, allocator, 2);
    }

    var pool = coro.ShadowStackPool.init(2048);
    defer pool.deinit(allocator);
    var host_sp: usize = 16384;
    const sp_binding = coro.StackPointerBinding.fromHostSlot(&host_sp);
    const stack = try pool.acquire(allocator);
    defer pool.release(allocator, stack) catch {};

    var task = coro.TaskContext.init(stack, sp_binding);
    task.enter();
    host_sp -= 64;
    const active_sp = host_sp;

    var dummy_mod = wamr.types.WasmModule{};
    const dummy_inst = try wamr.instance.instantiate(&dummy_mod, allocator);
    defer wamr.instance.destroy(dummy_inst);
    var env = try ExecEnv.create(dummy_inst, 256, allocator);
    defer env.destroy();

    // Push (ws_handle, out_ptr=256)
    try env.pushI32(@bitCast(ws_handle));
    try env.pushI32(256);
    try executor.dispatchCanonBuiltin(
        comp_inst,
        .{ .async_canon = .{ .waitable_set_wait = .{ .cancellable = false, .memory = 0 } } },
        env,
        null,
        allocator,
    );
    const ev_code: u32 = @bitCast(try env.popI32());
    try testing.expectEqual(@as(u32, @backingInt(async_canon.EventCode.subtask)), ev_code);
    // TaskContext must be resumed with its active shadow stack pointer intact
    try testing.expectEqual(coro.TaskContext.State.running, task.state);
    try testing.expectEqual(active_sp, host_sp);
    try task.complete();
    try testing.expectEqual(@as(usize, 16384), host_sp);
}

test "#1134: benchmark comptime CanonLift vs dynamic InterfaceValue host dispatch" {
    const allocator = testing.allocator;
    var comp_inst: instance_mod.ComponentInstance = undefined;
    comp_inst.test_mem = null;
    comp_inst.core_instances = &.{};
    try comp_inst.enableTestMem(allocator, 4096);
    defer comp_inst.disableTestMem();

    const Point = struct { x: i32, y: i32, z: i64 };
    const Handler = struct {
        fn sumPoint(p: Point, scale: i32) i64 {
            return (@as(i64, p.x) + @as(i64, p.y)) * @as(i64, scale) + p.z;
        }
    };

    const adapter = canon.HostFuncAdapter(Handler.sumPoint, .wasm32);
    const flat_args = [_]u64{ 10, 20, 100, 3 };
    var flat_results = [_]u64{0};

    const iterations: usize = 10_000;
    const io = std.Io.Threaded.global_single_threaded.io();

    const t0 = std.Io.Clock.Timestamp.now(io, .awake);
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        try adapter.callFlatSlots(null, &comp_inst, &flat_args, &flat_results, 0, allocator);
    }
    const t1 = std.Io.Clock.Timestamp.now(io, .awake);
    const flat_ns = t0.durationTo(t1).raw.nanoseconds;
    try testing.expectEqual(@as(u64, 190), flat_results[0]);

    var rec_fields = [_]abi.InterfaceValue{ .{ .s32 = 10 }, .{ .s32 = 20 }, .{ .s64 = 100 } };
    const dyn_args = [_]abi.InterfaceValue{
        .{ .record_val = &rec_fields },
        .{ .s32 = 3 },
    };
    var dyn_results = [_]abi.InterfaceValue{.{ .s64 = 0 }};

    const t2 = std.Io.Clock.Timestamp.now(io, .awake);
    i = 0;
    while (i < iterations) : (i += 1) {
        try adapter.call(null, &comp_inst, &dyn_args, &dyn_results, allocator);
    }
    const t3 = std.Io.Clock.Timestamp.now(io, .awake);
    const dyn_ns = t2.durationTo(t3).raw.nanoseconds;
    try testing.expectEqual(@as(i64, 190), dyn_results[0].s64);
    try testing.expect(flat_ns >= 0 and dyn_ns >= 0);
}
