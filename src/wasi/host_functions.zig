//! WASI host function implementations for the interpreter.
//!
//! These are native functions callable from Wasm via the import mechanism.
//! Each function follows the HostFn signature: it receives an opaque pointer
//! to an ExecEnv, pops arguments from the operand stack, and pushes results.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../runtime/common/types.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;

const is_single_threaded = builtin.single_threaded;

/// Host function for the `wasi.thread-spawn` import.
/// Pops start_arg (i32) from the operand stack, spawns a new thread,
/// and pushes the new TID (positive) or a negative errno on failure.
pub fn wasiThreadSpawn(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));

    if (is_single_threaded) {
        // Threads not available on this target — return error to guest
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

/// Host function for `wasi_snapshot_preview1.proc_exit`.
/// Pops the exit code (i32), signals all threads to stop via the trap flag,
/// and returns a Trap error to unwind the calling thread.
pub fn wasiProcExit(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    _ = env.popI32() catch return error.StackUnderflow; // exit code

    // Signal all threads to terminate
    if (env.module_inst.thread_manager) |tm| {
        tm.signalTrap();
    }

    return error.Trap;
}

/// Resolve WASI host functions for a module's imports.
/// Returns a slice of optional HostFn pointers indexed by import function index.
/// Caller owns the returned slice.
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
            if (std.mem.eql(u8, imp.module_name, "wasi") and
                std.mem.eql(u8, imp.field_name, "thread-spawn"))
            {
                host_fns[func_idx] = &wasiThreadSpawn;
            }
            if ((std.mem.eql(u8, imp.module_name, "wasi_snapshot_preview1") or
                std.mem.eql(u8, imp.module_name, "wasi")) and
                std.mem.eql(u8, imp.field_name, "proc_exit"))
            {
                host_fns[func_idx] = &wasiProcExit;
            }
            func_idx += 1;
        }
    }

    return host_fns;
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
