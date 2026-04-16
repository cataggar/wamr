//! WASI host function implementations for the interpreter.
//!
//! These are native functions callable from Wasm via the import mechanism.
//! Each function follows the HostFn signature: it receives an opaque pointer
//! to an ExecEnv, pops arguments from the operand stack, and pushes results.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../runtime/common/types.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
const wasi_core = @import("wasi_core.zig");

const is_single_threaded = builtin.single_threaded;

/// Get linear memory (memory index 0) from an ExecEnv.
fn getMemory(env: *ExecEnv) ?[]u8 {
    const inst = env.module_inst;
    if (inst.memories.len == 0) return null;
    return inst.memories[0].data;
}

// ── WASI host functions ───────────────────────────────────────────────

/// Host function for the `wasi.thread-spawn` import.
pub fn wasiThreadSpawn(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));

    if (is_single_threaded) {
        _ = env.popI32() catch return error.StackUnderflow;
        env.pushI32(-1) catch return error.StackOverflow;
        return;
    }

    const start_arg = env.popI32() catch return error.StackUnderflow;

    const tm = env.module_inst.thread_manager orelse {
        env.pushI32(-1) catch return error.StackOverflow;
        return;
    };

    const tid = tm.spawnThread(env.module_inst, start_arg) catch {
        env.pushI32(-1) catch return error.StackOverflow;
        return;
    };

    env.pushI32(tid) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.proc_exit` — exit the process.
pub fn wasiProcExit(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    _ = env.popI32() catch return error.StackUnderflow;

    if (env.module_inst.thread_manager) |tm| {
        tm.signalTrap();
    }

    return error.Trap;
}

/// `wasi_snapshot_preview1.fd_write` — write to a file descriptor.
/// Signature: (fd: i32, iovs: i32, iovs_len: i32, nwritten_ptr: i32) -> i32
pub fn wasiFdWrite(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const nwritten_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const iovs_len: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const iovs_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    const result = wasi_core.fdWriteCore(mem, fd, iovs_ptr, iovs_len, nwritten_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_seek` — seek on a file descriptor.
pub fn wasiFdSeek(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    _ = env.popI32() catch return error.StackUnderflow; // newoffset_ptr
    _ = env.popI32() catch return error.StackUnderflow; // whence
    _ = env.popI64() catch return error.StackUnderflow; // offset
    _ = env.popI32() catch return error.StackUnderflow; // fd
    env.pushI32(wasi_core.fdSeekCore()) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_close` — close a file descriptor.
pub fn wasiFdClose(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const fd = env.popI32() catch return error.StackUnderflow;
    env.pushI32(wasi_core.fdCloseCore(fd)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_fdstat_get` — get fd status.
pub fn wasiFdFdstatGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const buf_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    const result = wasi_core.fdFdstatGetCore(mem, fd, buf_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_prestat_get` — get preopened fd info.
pub fn wasiFdPrestatGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    _ = env.popI32() catch return error.StackUnderflow; // buf_ptr
    _ = env.popI32() catch return error.StackUnderflow; // fd
    env.pushI32(wasi_core.fdPrestatGetCore()) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_prestat_dir_name` — get preopened dir name.
pub fn wasiFdPrestatDirName(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    _ = env.popI32() catch return error.StackUnderflow; // path_len
    _ = env.popI32() catch return error.StackUnderflow; // path_ptr
    _ = env.popI32() catch return error.StackUnderflow; // fd
    env.pushI32(wasi_core.fdPrestatDirNameCore()) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.clock_time_get` — get clock time.
/// Signature: (clock_id: i32, precision: i64, time_ptr: i32) -> i32
pub fn wasiClockTimeGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const time_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    _ = env.popI64() catch return error.StackUnderflow; // precision
    const clock_id = env.popI32() catch return error.StackUnderflow;

    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    const result = wasi_core.clockTimeGetCore(mem, clock_id, time_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.environ_sizes_get` — get environment variable sizes.
pub fn wasiEnvironSizesGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const buf_size_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const count_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    const result = wasi_core.environSizesGetCore(mem, count_ptr, buf_size_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.environ_get` — get environment variables.
pub fn wasiEnvironGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    _ = env.popI32() catch return error.StackUnderflow; // environ_buf
    _ = env.popI32() catch return error.StackUnderflow; // environ
    env.pushI32(wasi_core.environGetCore()) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.args_sizes_get` — get argument sizes.
pub fn wasiArgsSizesGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const buf_size_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const count_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    const result = wasi_core.argsSizesGetCore(mem, count_ptr, buf_size_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.args_get` — get arguments.
pub fn wasiArgsGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    _ = env.popI32() catch return error.StackUnderflow; // argv_buf
    _ = env.popI32() catch return error.StackUnderflow; // argv
    env.pushI32(wasi_core.argsGetCore()) catch return error.StackOverflow;
}

// ── Import resolution ─────────────────────────────────────────────────

/// Resolve WASI host functions for a module's imports.
/// Returns a slice of optional HostFn pointers indexed by import function index.
pub fn resolveWasiHostFunctions(
    module: *const types.WasmModule,
    allocator: std.mem.Allocator,
) ![]const ?types.HostFn {
    if (module.import_function_count == 0) return &.{};

    const host_fns = try allocator.alloc(?types.HostFn, module.import_function_count);
    @memset(host_fns, null);

    var func_idx: u32 = 0;
    for (module.imports) |imp| {
        if (imp.kind == .function) {
            const is_wasi = std.mem.eql(u8, imp.module_name, "wasi_snapshot_preview1") or
                std.mem.eql(u8, imp.module_name, "wasi_unstable") or
                std.mem.eql(u8, imp.module_name, "wasi");

            if (is_wasi) {
                host_fns[func_idx] = resolveWasiFunction(imp.field_name);

                if (host_fns[func_idx] == null) {
                    std.debug.print("WASI: unresolved import: {s}.{s}\n", .{ imp.module_name, imp.field_name });
                }
            }
            func_idx += 1;
        }
    }

    return host_fns;
}

fn resolveWasiFunction(name: []const u8) ?types.HostFn {
    const map = .{
        .{ "proc_exit", &wasiProcExit },
        .{ "thread-spawn", &wasiThreadSpawn },
        .{ "fd_write", &wasiFdWrite },
        .{ "fd_seek", &wasiFdSeek },
        .{ "fd_close", &wasiFdClose },
        .{ "fd_fdstat_get", &wasiFdFdstatGet },
        .{ "fd_prestat_get", &wasiFdPrestatGet },
        .{ "fd_prestat_dir_name", &wasiFdPrestatDirName },
        .{ "clock_time_get", &wasiClockTimeGet },
        .{ "environ_sizes_get", &wasiEnvironSizesGet },
        .{ "environ_get", &wasiEnvironGet },
        .{ "args_sizes_get", &wasiArgsSizesGet },
        .{ "args_get", &wasiArgsGet },
    };

    inline for (map) |entry| {
        if (std.mem.eql(u8, name, entry[0])) return entry[1];
    }
    return null;
}

// ── Tests ──────────────────────────────────────────────────────────────────

test "resolveWasiHostFunctions: empty module" {
    const module = types.WasmModule{};
    const result = try resolveWasiHostFunctions(&module, std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "resolveWasiHostFunctions: module with thread-spawn import" {
    const imports = [_]types.ImportDesc{
        .{
            .module_name = "wasi",
            .field_name = "thread-spawn",
            .kind = .function,
            .func_type_idx = 0,
        },
    };
    const module = types.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
    };
    const result = try resolveWasiHostFunctions(&module, std.testing.allocator);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expect(result[0] != null);
}

test "resolveWasiHostFunctions: non-wasi import returns null" {
    const imports = [_]types.ImportDesc{
        .{
            .module_name = "env",
            .field_name = "some_func",
            .kind = .function,
            .func_type_idx = 0,
        },
    };
    const module = types.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
    };
    const result = try resolveWasiHostFunctions(&module, std.testing.allocator);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expect(result[0] == null);
}

test "resolveWasiHostFunctions: proc_exit resolved" {
    const imports = [_]types.ImportDesc{
        .{
            .module_name = "wasi_snapshot_preview1",
            .field_name = "proc_exit",
            .kind = .function,
            .func_type_idx = 0,
        },
    };
    const module = types.WasmModule{
        .imports = &imports,
        .import_function_count = 1,
    };
    const result = try resolveWasiHostFunctions(&module, std.testing.allocator);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expect(result[0] != null);
}

test "wasiProcExit: signals trap flag" {
    const ThreadManager = @import("thread_manager.zig").ThreadManager;
    const allocator = std.testing.allocator;

    // Set up a minimal module instance with a thread manager
    var tm = ThreadManager.init(allocator);
    defer tm.deinit();

    const wasm_module = types.WasmModule{};
    const inst = try allocator.create(types.ModuleInstance);
    defer allocator.destroy(inst);
    inst.* = .{
        .module = &wasm_module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
        .allocator = allocator,
        .thread_manager = &tm,
    };

    // Create an ExecEnv and push exit code argument
    var env = try ExecEnv.create(inst, 256, allocator);
    defer env.destroy();
    try env.pushI32(0); // exit code

    // Verify trap flag is not set
    try std.testing.expect(!tm.hasTrap());

    // Call wasiProcExit — should signal trap and return error.Trap
    const result = wasiProcExit(@ptrCast(env));
    try std.testing.expectError(error.Trap, result);

    // Trap flag should now be set
    try std.testing.expect(tm.hasTrap());
}

test "resolveWasiFunction: fd_write resolves" {
    const result = resolveWasiFunction("fd_write");
    try std.testing.expect(result != null);
}

test "resolveWasiFunction: clock_time_get resolves" {
    const result = resolveWasiFunction("clock_time_get");
    try std.testing.expect(result != null);
}

test "resolveWasiFunction: unknown returns null" {
    const result = resolveWasiFunction("nonexistent_function");
    try std.testing.expect(result == null);
}

test "resolveWasiFunction: all 13 functions resolve" {
    const names = [_][]const u8{
        "proc_exit",     "thread-spawn",       "fd_write",
        "fd_seek",       "fd_close",           "fd_fdstat_get",
        "fd_prestat_get", "fd_prestat_dir_name", "clock_time_get",
        "environ_sizes_get", "environ_get",    "args_sizes_get",
        "args_get",
    };
    for (names) |name| {
        const result = resolveWasiFunction(name);
        try std.testing.expect(result != null);
    }
}
