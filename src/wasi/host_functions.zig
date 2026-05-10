//! WASI host function implementations for the interpreter.
//!
//! Each function follows the HostFn signature: it receives an opaque pointer
//! to an ExecEnv, pops arguments from the operand stack, and pushes results.
//!
//! Two dispatch tiers are supported per host fn:
//!   - **ctx-aware**: when `ExecEnv.wasi_ctx` is non-null we forward to the
//!     real `wasi.WasiCtx` so args / env / preopens / file I/O all work.
//!   - **legacy stub**: when no ctx is attached (unit tests, embedders that
//!     never call `setWasiCtx`) we fall back to the existing
//!     `wasi_core.zig` behavior so test fixtures and fuzz harnesses keep
//!     working unchanged.
//!
//! `runWasm` in `src/main.zig` is the canonical caller that attaches a ctx;
//! see issue #400 for the wasi-testsuite integration this enables.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../runtime/common/types.zig");
const ExecEnv = @import("../runtime/common/exec_env.zig").ExecEnv;
const wasi_core = @import("wasi_core.zig");
const wasi = @import("wasi.zig");

const is_single_threaded = builtin.single_threaded;

/// Get linear memory (memory index 0) from an ExecEnv.
fn getMemory(env: *ExecEnv) ?[]u8 {
    const inst = env.module_inst;
    if (inst.memories.len == 0) return null;
    return inst.memories[0].data;
}

/// Cast `env.wasi_ctx` to `*wasi.WasiCtx`. Null when no ctx attached.
fn getCtx(env: *ExecEnv) ?*wasi.WasiCtx {
    const opaque_ptr = env.wasi_ctx orelse return null;
    return @ptrCast(@alignCast(opaque_ptr));
}

/// Translate `wasi.Errno` into the i32 errno value placed on the stack.
fn errnoVal(e: wasi.Errno) i32 {
    return @intCast(@intFromEnum(e));
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
    const code = env.popI32() catch return error.StackUnderflow;

    if (getCtx(env)) |ctx| {
        ctx.exit_code = @bitCast(code);
    }

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

    if (getCtx(env)) |ctx| {
        const result = ctxFdIoCore(ctx, mem, fd, iovs_ptr, iovs_len, nwritten_ptr, .write);
        env.pushI32(result) catch return error.StackOverflow;
        return;
    }

    const result = wasi_core.fdWriteCore(mem, fd, iovs_ptr, iovs_len, nwritten_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_read` — read from a file descriptor.
/// Signature: (fd: i32, iovs: i32, iovs_len: i32, nread_ptr: i32) -> i32
pub fn wasiFdRead(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const nread_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const iovs_len: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const iovs_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    if (getCtx(env)) |ctx| {
        const result = ctxFdIoCore(ctx, mem, fd, iovs_ptr, iovs_len, nread_ptr, .read);
        env.pushI32(result) catch return error.StackOverflow;
        return;
    }

    // Legacy: no real fd_read implementation off the ctx path.
    env.pushI32(wasi_core.WASI_EBADF) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_pread` — positional read at `offset` without
/// affecting the cached fd position.
/// Signature: (fd: i32, iovs_ptr: i32, iovs_len: i32, offset: i64, nread_ptr: i32) -> i32
pub fn wasiFdPread(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const nread_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const offset = env.popI64() catch return error.StackUnderflow;
    const iovs_len: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const iovs_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };
    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdPreadCore(ctx, mem, fd, iovs_ptr, iovs_len, offset, nread_ptr)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_pwrite` — positional write at `offset` without
/// affecting the cached fd position.
/// Signature: (fd: i32, iovs_ptr: i32, iovs_len: i32, offset: i64, nwritten_ptr: i32) -> i32
pub fn wasiFdPwrite(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const nwritten_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const offset = env.popI64() catch return error.StackUnderflow;
    const iovs_len: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const iovs_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };
    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdPwriteCore(ctx, mem, fd, iovs_ptr, iovs_len, offset, nwritten_ptr)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_readdir` — encode directory entries as
/// preview1 `dirent` records into the guest buffer.
/// Signature: (fd: i32, buf_ptr: i32, buf_len: i32, cookie: i64, bufused_ptr: i32) -> i32
pub fn wasiFdReaddir(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const bufused_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const cookie: u64 = @bitCast(env.popI64() catch return error.StackUnderflow);
    const buf_len: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const buf_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };
    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdReaddirCore(ctx, mem, fd, buf_ptr, buf_len, cookie, bufused_ptr)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_seek` — seek on a file descriptor.
pub fn wasiFdSeek(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const newoffset_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const whence: u8 = @intCast(@as(u32, @bitCast(env.popI32() catch return error.StackUnderflow)) & 0xff);
    const offset = env.popI64() catch return error.StackUnderflow;
    const fd = env.popI32() catch return error.StackUnderflow;

    if (getCtx(env)) |ctx| {
        const mem = getMemory(env) orelse {
            env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
            return;
        };
        const result = ctxFdSeekCore(ctx, mem, fd, offset, whence, newoffset_ptr);
        env.pushI32(result) catch return error.StackOverflow;
        return;
    }

    env.pushI32(wasi_core.fdSeekCore()) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_close` — close a file descriptor.
pub fn wasiFdClose(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const fd = env.popI32() catch return error.StackUnderflow;

    if (getCtx(env)) |ctx| {
        if (fd < 0) {
            env.pushI32(@intCast(@intFromEnum(wasi.Errno.badf))) catch return error.StackOverflow;
            return;
        }
        const result = ctx.fd_close(@intCast(fd));
        env.pushI32(errnoVal(result)) catch return error.StackOverflow;
        return;
    }

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

    if (getCtx(env)) |ctx| {
        const result = ctxFdFdstatGetCore(ctx, mem, fd, buf_ptr);
        env.pushI32(result) catch return error.StackOverflow;
        return;
    }

    const result = wasi_core.fdFdstatGetCore(mem, fd, buf_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_prestat_get` — get preopened fd info.
pub fn wasiFdPrestatGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const buf_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    if (getCtx(env)) |ctx| {
        const mem = getMemory(env) orelse {
            env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
            return;
        };
        const result = ctxFdPrestatGetCore(ctx, mem, fd, buf_ptr);
        env.pushI32(result) catch return error.StackOverflow;
        return;
    }

    env.pushI32(wasi_core.fdPrestatGetCore()) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_prestat_dir_name` — get preopened dir name.
pub fn wasiFdPrestatDirName(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const path_len: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const path_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    if (getCtx(env)) |ctx| {
        const mem = getMemory(env) orelse {
            env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
            return;
        };
        const result = ctxFdPrestatDirNameCore(ctx, mem, fd, path_ptr, path_len);
        env.pushI32(result) catch return error.StackOverflow;
        return;
    }

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

    if (getCtx(env)) |ctx| {
        const sizes = ctx.environ_sizes_get();
        if (!wasi_core.memWriteU32(mem, count_ptr, sizes.count) or
            !wasi_core.memWriteU32(mem, buf_size_ptr, sizes.buf_size))
        {
            env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
            return;
        }
        env.pushI32(wasi_core.WASI_ESUCCESS) catch return error.StackOverflow;
        return;
    }

    const result = wasi_core.environSizesGetCore(mem, count_ptr, buf_size_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.environ_get` — get environment variables.
pub fn wasiEnvironGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const environ_buf: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const environ_ptrs: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

    if (getCtx(env)) |ctx| {
        const mem = getMemory(env) orelse {
            env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
            return;
        };
        const result = writeStringTable(mem, ctx.env_vars, environ_ptrs, environ_buf);
        env.pushI32(result) catch return error.StackOverflow;
        return;
    }

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

    if (getCtx(env)) |ctx| {
        const sizes = ctx.args_sizes_get();
        if (!wasi_core.memWriteU32(mem, count_ptr, sizes.count) or
            !wasi_core.memWriteU32(mem, buf_size_ptr, sizes.buf_size))
        {
            env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
            return;
        }
        env.pushI32(wasi_core.WASI_ESUCCESS) catch return error.StackOverflow;
        return;
    }

    const result = wasi_core.argsSizesGetCore(mem, count_ptr, buf_size_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.args_get` — get arguments.
pub fn wasiArgsGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const argv_buf: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const argv_ptrs: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

    if (getCtx(env)) |ctx| {
        const mem = getMemory(env) orelse {
            env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
            return;
        };
        const result = writeStringTable(mem, ctx.args, argv_ptrs, argv_buf);
        env.pushI32(result) catch return error.StackOverflow;
        return;
    }

    env.pushI32(wasi_core.argsGetCore()) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.random_get` — fill `buf` with secure random bytes.
/// Signature: (buf: i32, buf_len: i32) -> i32
pub fn wasiRandomGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const buf_len: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const buf_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);

    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };
    if (@as(u64, buf_ptr) + buf_len > mem.len) {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    }

    if (getCtx(env)) |ctx| {
        ctx.random_get(mem[buf_ptr..][0..buf_len]);
    } else {
        // Best-effort fallback: zero-fill so downstream code is deterministic
        // rather than reading uninitialized linear memory.
        @memset(mem[buf_ptr..][0..buf_len], 0);
    }
    env.pushI32(wasi_core.WASI_ESUCCESS) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.path_open` — open a path relative to a preopened
/// directory. Signature:
/// (dirfd: i32, dirflags: i32, path_ptr: i32, path_len: i32, oflags: i32,
///  fs_rights_base: i64, fs_rights_inh: i64, fdflags: i32, fd_ptr: i32) -> i32
pub fn wasiPathOpen(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const fd_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fdflags: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    _ = env.popI64() catch return error.StackUnderflow; // fs_rights_inheriting
    _ = env.popI64() catch return error.StackUnderflow; // fs_rights_base
    const oflags: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const path_len: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const path_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    _ = env.popI32() catch return error.StackUnderflow; // dirflags
    const dirfd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };
    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    const result = ctxPathOpenCore(ctx, mem, dirfd, path_ptr, path_len, oflags, fdflags, fd_ptr);
    env.pushI32(result) catch return error.StackOverflow;
}

// ── fd metadata host functions (issue #420 phase 1) ───────────────────

/// `wasi_snapshot_preview1.fd_filestat_get` — populate a 64-byte
/// `filestat` struct in linear memory.
/// Signature: (fd: i32, buf_ptr: i32) -> i32
pub fn wasiFdFilestatGet(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const buf_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };
    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdFilestatGetCore(ctx, mem, fd, buf_ptr)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_filestat_set_size` — truncate a regular
/// file to the requested length.
/// Signature: (fd: i32, size: i64) -> i32
pub fn wasiFdFilestatSetSize(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const size = env.popI64() catch return error.StackUnderflow;
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdFilestatSetSizeCore(ctx, fd, size)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_filestat_set_times` — set atim / mtim on
/// the file referenced by `fd`. `fst_flags` selects which timestamp(s)
/// to set and whether to use `now`.
/// Signature: (fd: i32, atim: i64, mtim: i64, fst_flags: i32) -> i32
pub fn wasiFdFilestatSetTimes(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const fst_flags: u16 = blk: {
        const v = env.popI32() catch return error.StackUnderflow;
        break :blk @intCast(@as(u32, @bitCast(v)) & 0xffff);
    };
    const mtim: u64 = @bitCast(env.popI64() catch return error.StackUnderflow);
    const atim: u64 = @bitCast(env.popI64() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdFilestatSetTimesCore(ctx, fd, atim, mtim, fst_flags)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_fdstat_set_flags` — apply preview1 fdflags
/// to the host fd. Currently maps APPEND and NONBLOCK to host O_*; the
/// SYNC family is accepted only when zero (no-op) since we don't open
/// host fds with O_DSYNC/O_RSYNC/O_SYNC and toggling them post-open
/// isn't portable.
/// Signature: (fd: i32, fdflags: i32) -> i32
pub fn wasiFdFdstatSetFlags(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const fdflags_raw = env.popI32() catch return error.StackUnderflow;
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };

    const fdflags: u16 = @intCast(@as(u32, @bitCast(fdflags_raw)) & 0xffff);
    env.pushI32(ctxFdFdstatSetFlagsCore(ctx, fd, fdflags)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_fdstat_set_rights` — narrow the rights cap
/// recorded on the fd. Widening (setting any bit not currently set)
/// returns `notcapable` per the witx spec.
/// Signature: (fd: i32, rights_base: i64, rights_inheriting: i64) -> i32
pub fn wasiFdFdstatSetRights(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const rights_inheriting: u64 = @bitCast(env.popI64() catch return error.StackUnderflow);
    const rights_base: u64 = @bitCast(env.popI64() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdFdstatSetRightsCore(ctx, fd, rights_base, rights_inheriting)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_advise` — pass a posix_fadvise hint.
/// Signature: (fd: i32, offset: i64, len: i64, advice: i32) -> i32
pub fn wasiFdAdvise(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const advice_raw = env.popI32() catch return error.StackUnderflow;
    const len = env.popI64() catch return error.StackUnderflow;
    const offset = env.popI64() catch return error.StackUnderflow;
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };

    const advice: u8 = @intCast(@as(u32, @bitCast(advice_raw)) & 0xff);
    env.pushI32(ctxFdAdviseCore(ctx, fd, offset, len, advice)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_allocate` — extend a regular file so that
/// `[offset, offset+len)` is allocated. Falls back to `setLength` when
/// the host's `fallocate` is unavailable.
/// Signature: (fd: i32, offset: i64, len: i64) -> i32
pub fn wasiFdAllocate(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const len = env.popI64() catch return error.StackUnderflow;
    const offset = env.popI64() catch return error.StackUnderflow;
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdAllocateCore(ctx, fd, offset, len)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_datasync` — flush file contents (not
/// necessarily metadata) to disk.
/// Signature: (fd: i32) -> i32
pub fn wasiFdDatasync(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdSyncCore(ctx, fd, .data)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_sync` — flush file contents and metadata
/// to disk.
/// Signature: (fd: i32) -> i32
pub fn wasiFdSync(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdSyncCore(ctx, fd, .full)) catch return error.StackOverflow;
}

/// `wasi_snapshot_preview1.fd_tell` — write the current file position to
/// `offset_ptr`.
/// Signature: (fd: i32, offset_ptr: i32) -> i32
pub fn wasiFdTell(env_opaque: *anyopaque) types.HostFnError!void {
    const env: *ExecEnv = @ptrCast(@alignCast(env_opaque));
    const offset_ptr: u32 = @bitCast(env.popI32() catch return error.StackUnderflow);
    const fd = env.popI32() catch return error.StackUnderflow;

    const ctx = getCtx(env) orelse {
        env.pushI32(wasi_core.WASI_ENOSYS) catch return error.StackOverflow;
        return;
    };
    const mem = getMemory(env) orelse {
        env.pushI32(wasi_core.WASI_EINVAL) catch return error.StackOverflow;
        return;
    };

    env.pushI32(ctxFdTellCore(ctx, mem, fd, offset_ptr)) catch return error.StackOverflow;
}

// ── ctx-aware core implementations ────────────────────────────────────

/// Layout for `args_get` / `environ_get`: write `argv_ptrs` (one i32 pointer
/// per entry) and the concatenated NUL-terminated strings to `argv_buf`.
fn writeStringTable(mem: []u8, entries: []const []const u8, argv_ptrs: u32, argv_buf: u32) i32 {
    var buf_offset: u32 = 0;
    for (entries, 0..) |entry, i| {
        const ptr_slot = argv_ptrs + @as(u32, @intCast(i)) * 4;
        const buf_pos = argv_buf + buf_offset;
        if (buf_pos + entry.len + 1 > mem.len) return wasi_core.WASI_EINVAL;
        if (!wasi_core.memWriteU32(mem, ptr_slot, buf_pos)) return wasi_core.WASI_EINVAL;
        @memcpy(mem[buf_pos..][0..entry.len], entry);
        mem[buf_pos + entry.len] = 0;
        buf_offset += @as(u32, @intCast(entry.len)) + 1;
    }
    return wasi_core.WASI_ESUCCESS;
}

/// fdstat layout (24 bytes): fs_filetype(u8) + 1 pad + fs_flags(u16) +
/// 4 pad + fs_rights_base(u64) + fs_rights_inheriting(u64).
fn writeFdstat(mem: []u8, buf_ptr: u32, entry: wasi.FdEntry) i32 {
    if (buf_ptr + 24 > mem.len) return wasi_core.WASI_EINVAL;
    @memset(mem[buf_ptr..][0..24], 0);
    mem[buf_ptr] = @intFromEnum(filetypeForEntry(entry));
    _ = wasi_core.memWriteU16(mem, buf_ptr + 2, entry.fdflags);
    _ = wasi_core.memWriteU64(mem, buf_ptr + 8, entry.rights_base);
    _ = wasi_core.memWriteU64(mem, buf_ptr + 16, entry.rights_inheriting);
    return wasi_core.WASI_ESUCCESS;
}

/// Resolve the WASI `filetype` for an fd entry. Stdio fds are looked up
/// via `std.posix.isatty` so `wasi-libc`'s `isatty(3)` (which checks
/// `fs_filetype == character_device`) returns the right answer when the
/// host stdio is/isn't a TTY. For regular files, sockets and directories
/// the kind is determined statically by the FdEntry; for an unmapped
/// stdio fd that isn't a TTY we report `unknown` so guests don't
/// incorrectly treat a pipe as a TTY.
fn filetypeForEntry(entry: wasi.FdEntry) wasi.Filetype {
    return switch (entry.kind) {
        .stdin => stdioFiletype(0),
        .stdout => stdioFiletype(1),
        .stderr => stdioFiletype(2),
        .regular_file => .regular_file,
        .directory => .directory,
        .socket => .socket_stream,
    };
}

/// Probe whether the given POSIX-style stdio descriptor is a TTY. Only
/// meaningful on Linux; other platforms return `.unknown` so the caller
/// falls back to the FdEntry kind. The parameter is `i32` (rather than
/// `std.posix.fd_t`) because Windows uses HANDLE-based fd_t but doesn't
/// expose positional stdio fds — keeping the type as a plain integer
/// keeps the helper buildable on every target.
fn stdioFiletype(host_fd: i32) wasi.Filetype {
    if (builtin.os.tag == .linux) {
        // F_GETFL doesn't tell us whether the fd is a tty; use isatty()
        // syscall via the standard linux ioctl wrapper. std.os.linux
        // exposes `isatty` indirectly through the TIOCGWINSZ probe used
        // by `tcgetattr`, so use the dedicated linux syscall path here.
        const linux = std.os.linux;
        var termios: linux.termios = undefined;
        const rc = linux.tcgetattr(host_fd, &termios);
        if (linux.errno(rc) == .SUCCESS) return .character_device;
        return .unknown;
    }
    return .unknown;
}

fn ctxFdFdstatGetCore(ctx: *wasi.WasiCtx, mem: []u8, fd: i32, buf_ptr: u32) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const u_fd: u32 = @intCast(fd);
    var entry = ctx.fd_table.get(u_fd) orelse return wasi_core.WASI_EBADF;
    // Refresh fdflags from the host fd when applicable so changes that
    // happened outside `fd_fdstat_set_flags` (e.g. inheritance) are
    // reflected. Stdio + regular files honour O_APPEND/O_NONBLOCK; the
    // remaining preview1 sync flags don't have a portable readback
    // path, so we trust whatever was stashed on the entry.
    if (entryHostFd(entry)) |host_fd| {
        if (builtin.os.tag == .linux) {
            const flags = std.os.linux.fcntl(host_fd, std.os.linux.F.GETFL, 0);
            if (std.os.linux.errno(flags) == .SUCCESS) {
                const o: std.os.linux.O = @bitCast(@as(u32, @intCast(flags & 0xFFFF_FFFF)));
                var fd_flags: u16 = entry.fdflags & ~(wasi.FDFLAGS_APPEND | wasi.FDFLAGS_NONBLOCK);
                if (o.APPEND) fd_flags |= wasi.FDFLAGS_APPEND;
                if (o.NONBLOCK) fd_flags |= wasi.FDFLAGS_NONBLOCK;
                entry.fdflags = fd_flags;
            }
        }
    }
    return writeFdstat(mem, buf_ptr, entry);
}

/// Look up the host fd backing an FdEntry. Returns null for directory
/// entries (no host fd to act on directly — use `host_dir`). On
/// Windows we don't have positional stdio fds (`std.posix.fd_t` is a
/// HANDLE there), so stdio entries also return null — none of the
/// host-fd consumers (Linux fcntl/fadvise/etc.) run on Windows anyway.
fn entryHostFd(entry: wasi.FdEntry) ?std.posix.fd_t {
    return switch (entry.kind) {
        .stdin => stdio_in_fd,
        .stdout => stdio_out_fd,
        .stderr => stdio_err_fd,
        .regular_file, .socket => entry.host_fd,
        .directory => null,
    };
}

const stdio_in_fd: ?std.posix.fd_t = if (builtin.os.tag == .windows) null else std.posix.STDIN_FILENO;
const stdio_out_fd: ?std.posix.fd_t = if (builtin.os.tag == .windows) null else std.posix.STDOUT_FILENO;
const stdio_err_fd: ?std.posix.fd_t = if (builtin.os.tag == .windows) null else std.posix.STDERR_FILENO;

fn ctxFdPrestatGetCore(ctx: *wasi.WasiCtx, mem: []u8, fd: i32, buf_ptr: u32) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const u_fd: u32 = @intCast(fd);
    const name = ctx.preopenName(u_fd) orelse return wasi_core.WASI_EBADF;
    // prestat: u8 type (0 = dir) + 3 pad + u32 name_len = 8 bytes.
    if (buf_ptr + 8 > mem.len) return wasi_core.WASI_EINVAL;
    @memset(mem[buf_ptr..][0..8], 0);
    mem[buf_ptr] = 0;
    if (!wasi_core.memWriteU32(mem, buf_ptr + 4, @intCast(name.len))) return wasi_core.WASI_EINVAL;
    return wasi_core.WASI_ESUCCESS;
}

fn ctxFdPrestatDirNameCore(ctx: *wasi.WasiCtx, mem: []u8, fd: i32, path_ptr: u32, path_len: u32) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const u_fd: u32 = @intCast(fd);
    const name = ctx.preopenName(u_fd) orelse return wasi_core.WASI_EBADF;
    if (path_len < name.len) return wasi_core.WASI_EINVAL;
    if (path_ptr + path_len > mem.len) return wasi_core.WASI_EINVAL;
    @memcpy(mem[path_ptr..][0..name.len], name);
    return wasi_core.WASI_ESUCCESS;
}

const FdIoOp = enum { read, write };

/// Linear-memory-driven fd_read/fd_write: parse the iov array, dispatch to
/// posix syscalls on host_fd (regular files) or std.Io stream helpers
/// (stdio). Updates `pos` for regular files and writes the byte count to
/// `nresult_ptr`.
fn ctxFdIoCore(
    ctx: *wasi.WasiCtx,
    mem: []u8,
    fd: i32,
    iovs_ptr: u32,
    iovs_len: u32,
    nresult_ptr: u32,
    op: FdIoOp,
) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const u_fd: u32 = @intCast(fd);
    const entry_ptr = ctx.fd_table.entries.getPtr(u_fd) orelse return wasi_core.WASI_EBADF;

    var total: u32 = 0;
    var i: u32 = 0;
    while (i < iovs_len) : (i += 1) {
        const iov_offset = iovs_ptr + i * 8;
        const buf_ptr = wasi_core.memReadU32(mem, iov_offset) orelse
            return wasi_core.WASI_EINVAL;
        const buf_len = wasi_core.memReadU32(mem, iov_offset + 4) orelse
            return wasi_core.WASI_EINVAL;
        if (@as(u64, buf_ptr) + buf_len > mem.len) return wasi_core.WASI_EINVAL;
        const slice = mem[buf_ptr..][0..buf_len];

        switch (op) {
            .write => {
                const n = doWrite(ctx, entry_ptr, slice) catch |e| return errnoToI32(e);
                total += @intCast(n);
                if (n < slice.len) break;
            },
            .read => {
                const n = doRead(ctx, entry_ptr, slice) catch |e| return errnoToI32(e);
                total += @intCast(n);
                if (n < slice.len) break;
            },
        }
    }

    if (!wasi_core.memWriteU32(mem, nresult_ptr, total)) return wasi_core.WASI_EINVAL;
    return wasi_core.WASI_ESUCCESS;
}

fn doWrite(ctx: *wasi.WasiCtx, entry_ptr: *wasi.FdEntry, data: []const u8) !usize {
    switch (entry_ptr.kind) {
        .stdout => {
            const file = std.Io.File.stdout();
            try file.writeStreamingAll(ctx.io, data);
            return data.len;
        },
        .stderr => {
            const file = std.Io.File.stderr();
            try file.writeStreamingAll(ctx.io, data);
            return data.len;
        },
        .regular_file => {
            const host_fd = entry_ptr.host_fd orelse return error.BadFd;
            const file = std.Io.File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
            var buf: [4096]u8 = undefined;
            var w = file.writer(ctx.io, &buf);
            w.seekTo(entry_ptr.pos) catch return error.IoError;
            w.interface.writeAll(data) catch return error.IoError;
            w.flush() catch return error.IoError;
            entry_ptr.pos += data.len;
            return data.len;
        },
        else => return error.BadFd,
    }
}

fn doRead(ctx: *wasi.WasiCtx, entry_ptr: *wasi.FdEntry, data: []u8) !usize {
    switch (entry_ptr.kind) {
        .stdin => {
            const file = std.Io.File.stdin();
            var buf: [4096]u8 = undefined;
            var r = file.reader(ctx.io, &buf);
            const n = r.interface.readSliceShort(data) catch return error.IoError;
            return n;
        },
        .regular_file => {
            const host_fd = entry_ptr.host_fd orelse return error.BadFd;
            const file = std.Io.File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
            var buf: [4096]u8 = undefined;
            var r = file.reader(ctx.io, &buf);
            r.seekTo(entry_ptr.pos) catch return error.IoError;
            const n = r.interface.readSliceShort(data) catch return error.IoError;
            entry_ptr.pos += n;
            return n;
        },
        else => return error.BadFd,
    }
}

fn errnoToI32(e: anyerror) i32 {
    return switch (e) {
        error.BadFd => wasi_core.WASI_EBADF,
        error.AccessDenied => @intCast(@intFromEnum(wasi.Errno.acces)),
        error.NoSpaceLeft => @intCast(@intFromEnum(wasi.Errno.nospc)),
        else => wasi_core.WASI_EINVAL,
    };
}

fn ctxFdSeekCore(
    ctx: *wasi.WasiCtx,
    mem: []u8,
    fd: i32,
    offset: i64,
    whence: u8,
    newoffset_ptr: u32,
) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const u_fd: u32 = @intCast(fd);
    const entry_ptr = ctx.fd_table.entries.getPtr(u_fd) orelse return wasi_core.WASI_EBADF;
    if (entry_ptr.kind != .regular_file) {
        return @intCast(@intFromEnum(wasi.Errno.spipe));
    }
    const host_fd = entry_ptr.host_fd orelse return wasi_core.WASI_EBADF;
    const file = std.Io.File{ .handle = host_fd, .flags = .{ .nonblocking = false } };

    const new_pos: i64 = switch (whence) {
        0 => offset, // SET
        1 => @as(i64, @intCast(entry_ptr.pos)) + offset, // CUR
        2 => blk: {
            const stat = file.stat(ctx.io) catch return wasi_core.WASI_EINVAL;
            const size: i64 = @intCast(stat.size);
            break :blk size + offset;
        },
        else => return wasi_core.WASI_EINVAL,
    };
    if (new_pos < 0) return wasi_core.WASI_EINVAL;
    entry_ptr.pos = @intCast(new_pos);
    if (!wasi_core.memWriteU64(mem, newoffset_ptr, entry_ptr.pos)) return wasi_core.WASI_EINVAL;
    return wasi_core.WASI_ESUCCESS;
}

/// Translate a Linux `getdents64` `d_type` to a WASI preview1 `filetype`.
/// `DT_FIFO`, `DT_WHT`, and any unknown values fall through to `unknown`
/// since preview1 has no dedicated FIFO/whiteout filetype.
fn wasiFiletypeFromDt(dt: u8) wasi.Filetype {
    const linux = std.os.linux;
    return switch (dt) {
        linux.DT.REG => .regular_file,
        linux.DT.DIR => .directory,
        linux.DT.CHR => .character_device,
        linux.DT.BLK => .block_device,
        linux.DT.LNK => .symbolic_link,
        linux.DT.SOCK => .socket_stream,
        else => .unknown,
    };
}

/// Validate `fd` for fd_pread / fd_pwrite, returning the FdEntry pointer
/// or a WASI errno. The host-fd null check is deliberately deferred so
/// callers can apply per-flavour rejections (e.g. fd_pwrite refuses
/// append-mode fds with `notsup`) before bailing out with `EBADF`.
fn pIoLookup(
    ctx: *wasi.WasiCtx,
    fd: i32,
    offset: i64,
) union(enum) {
    ok: *wasi.FdEntry,
    err: i32,
} {
    if (fd < 0) return .{ .err = wasi_core.WASI_EBADF };
    if (offset < 0) return .{ .err = wasi_core.WASI_EINVAL };
    const u_fd: u32 = @intCast(fd);
    const entry_ptr = ctx.fd_table.entries.getPtr(u_fd) orelse
        return .{ .err = wasi_core.WASI_EBADF };
    switch (entry_ptr.kind) {
        .stdin, .stdout, .stderr, .socket => return .{ .err = @intCast(@intFromEnum(wasi.Errno.spipe)) },
        .directory => return .{ .err = @intCast(@intFromEnum(wasi.Errno.isdir)) },
        .regular_file => {},
    }
    return .{ .ok = entry_ptr };
}

fn ctxFdPreadCore(
    ctx: *wasi.WasiCtx,
    mem: []u8,
    fd: i32,
    iovs_ptr: u32,
    iovs_len: u32,
    offset: i64,
    nread_ptr: u32,
) i32 {
    const entry_ptr = switch (pIoLookup(ctx, fd, offset)) {
        .err => |e| return e,
        .ok => |p| p,
    };
    const host_fd = entry_ptr.host_fd orelse return wasi_core.WASI_EBADF;

    if (builtin.os.tag != .linux) return wasi_core.WASI_ENOSYS;
    const linux = std.os.linux;

    var total: u32 = 0;
    var cur_off: i64 = offset;
    var i: u32 = 0;
    while (i < iovs_len) : (i += 1) {
        const iov_offset = iovs_ptr + i * 8;
        const buf_ptr = wasi_core.memReadU32(mem, iov_offset) orelse
            return wasi_core.WASI_EINVAL;
        const buf_len = wasi_core.memReadU32(mem, iov_offset + 4) orelse
            return wasi_core.WASI_EINVAL;
        if (@as(u64, buf_ptr) + buf_len > mem.len) return wasi_core.WASI_EINVAL;
        if (buf_len == 0) continue;

        const dst = mem[buf_ptr..][0..buf_len];
        const rc = linux.pread(host_fd, dst.ptr, dst.len, cur_off);
        if (linux.errno(rc) != .SUCCESS) return mapLinuxErrno(rc);
        const n: u32 = @intCast(rc);
        total += n;
        cur_off += @intCast(n);
        if (n < dst.len) break; // EOF or short read
    }

    if (!wasi_core.memWriteU32(mem, nread_ptr, total)) return wasi_core.WASI_EINVAL;
    return wasi_core.WASI_ESUCCESS;
}

fn ctxFdPwriteCore(
    ctx: *wasi.WasiCtx,
    mem: []u8,
    fd: i32,
    iovs_ptr: u32,
    iovs_len: u32,
    offset: i64,
    nwritten_ptr: u32,
) i32 {
    const entry_ptr = switch (pIoLookup(ctx, fd, offset)) {
        .err => |e| return e,
        .ok => |p| p,
    };
    // POSIX requires pwrite to ignore the file's append-mode (Linux's
    // implementation in fact respects O_APPEND, contradicting POSIX).
    // Rather than silently differ from the witx contract, refuse pwrite
    // on append-mode fds with `notsup`. Guests that want positional
    // writes shouldn't be opening with APPEND in the first place.
    if ((entry_ptr.fdflags & wasi.FDFLAGS_APPEND) != 0) {
        return @intCast(@intFromEnum(wasi.Errno.notsup));
    }
    const host_fd = entry_ptr.host_fd orelse return wasi_core.WASI_EBADF;

    if (builtin.os.tag != .linux) return wasi_core.WASI_ENOSYS;
    const linux = std.os.linux;

    var total: u32 = 0;
    var cur_off: i64 = offset;
    var i: u32 = 0;
    while (i < iovs_len) : (i += 1) {
        const iov_offset = iovs_ptr + i * 8;
        const buf_ptr = wasi_core.memReadU32(mem, iov_offset) orelse
            return wasi_core.WASI_EINVAL;
        const buf_len = wasi_core.memReadU32(mem, iov_offset + 4) orelse
            return wasi_core.WASI_EINVAL;
        if (@as(u64, buf_ptr) + buf_len > mem.len) return wasi_core.WASI_EINVAL;
        if (buf_len == 0) continue;

        const src = mem[buf_ptr..][0..buf_len];
        const rc = linux.pwrite(host_fd, src.ptr, src.len, cur_off);
        if (linux.errno(rc) != .SUCCESS) return mapLinuxErrno(rc);
        const n: u32 = @intCast(rc);
        total += n;
        cur_off += @intCast(n);
        if (n < src.len) break; // partial write — unusual but possible
    }

    if (!wasi_core.memWriteU32(mem, nwritten_ptr, total)) return wasi_core.WASI_EINVAL;
    return wasi_core.WASI_ESUCCESS;
}

/// Encode `getdents64`-derived directory entries into the guest buffer
/// using the preview1 `dirent` layout (24-byte header + name bytes, no
/// NUL). Truncates on overflow, matching the wasi-libc expectation that
/// `bufused == buf_len` signals "more entries may exist; retry with the
/// last complete entry's `d_next` cookie".
fn ctxFdReaddirCore(
    ctx: *wasi.WasiCtx,
    mem: []u8,
    fd: i32,
    buf_ptr: u32,
    buf_len: u32,
    cookie: u64,
    bufused_ptr: u32,
) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    if (@as(u64, buf_ptr) + buf_len > mem.len) return wasi_core.WASI_EINVAL;

    const u_fd: u32 = @intCast(fd);
    const entry = ctx.fd_table.get(u_fd) orelse return wasi_core.WASI_EBADF;
    if (entry.kind != .directory) return @intCast(@intFromEnum(wasi.Errno.notdir));
    const dir = entry.host_dir orelse return wasi_core.WASI_EBADF;

    if (builtin.os.tag != .linux) return wasi_core.WASI_ENOSYS;
    const linux = std.os.linux;

    // Cookie 0 = restart from the top; otherwise it's the kernel d_off
    // we returned for the previous entry.
    const seek_off: i64 = @bitCast(cookie);
    const lrc = linux.lseek(dir.handle, seek_off, linux.SEEK.SET);
    if (linux.errno(lrc) != .SUCCESS) return mapLinuxErrno(lrc);

    var bufused: u32 = 0;
    var staging: [4096]u8 = undefined;

    outer: while (true) {
        const grc = linux.getdents64(dir.handle, &staging, staging.len);
        if (linux.errno(grc) != .SUCCESS) return mapLinuxErrno(grc);
        if (grc == 0) break; // end of directory

        var pos: usize = 0;
        while (pos < grc) {
            // linux_dirent64: u64 d_ino, s64 d_off, u16 d_reclen, u8 d_type, char d_name[].
            const ino = std.mem.readInt(u64, staging[pos..][0..8], .little);
            const off_raw = std.mem.readInt(u64, staging[pos + 8 ..][0..8], .little);
            const reclen = std.mem.readInt(u16, staging[pos + 16 ..][0..2], .little);
            const dt = staging[pos + 18];
            const name_start = pos + 19;
            const name_end_max = pos + reclen;
            var name_len: usize = 0;
            while (name_start + name_len < name_end_max and staging[name_start + name_len] != 0) {
                name_len += 1;
            }
            const name = staging[name_start..][0..name_len];

            // Encode the 24-byte preview1 dirent header.
            var hdr: [24]u8 = @splat(0);
            std.mem.writeInt(u64, hdr[0..8], off_raw, .little);
            std.mem.writeInt(u64, hdr[8..16], ino, .little);
            std.mem.writeInt(u32, hdr[16..20], @intCast(name_len), .little);
            hdr[20] = @intFromEnum(wasiFiletypeFromDt(dt));

            // Header — copy as much as fits.
            const remaining_h = buf_len - bufused;
            if (remaining_h == 0) break :outer;
            const hdr_to_copy = @min(@as(u32, @intCast(hdr.len)), remaining_h);
            @memcpy(mem[buf_ptr + bufused ..][0..hdr_to_copy], hdr[0..hdr_to_copy]);
            bufused += hdr_to_copy;
            if (hdr_to_copy < hdr.len) break :outer;

            // Name — copy as much as fits.
            const remaining_n = buf_len - bufused;
            if (remaining_n == 0) break :outer;
            const name_to_copy = @min(@as(u32, @intCast(name_len)), remaining_n);
            @memcpy(mem[buf_ptr + bufused ..][0..name_to_copy], name[0..name_to_copy]);
            bufused += name_to_copy;
            if (name_to_copy < name_len) break :outer;

            pos += reclen;
        }
    }

    if (!wasi_core.memWriteU32(mem, bufused_ptr, bufused)) return wasi_core.WASI_EINVAL;
    return wasi_core.WASI_ESUCCESS;
}

/// `path_open` core: resolve `path` relative to the dirfd's host Dir, open
/// it via std.Io.Dir.openFile (read or read+write), allocate a new fd
/// pointing at the host fd, and write the new fd to `fd_ptr`.
fn ctxPathOpenCore(
    ctx: *wasi.WasiCtx,
    mem: []u8,
    dirfd: i32,
    path_ptr: u32,
    path_len: u32,
    oflags: u32,
    fdflags: u32,
    fd_ptr: u32,
) i32 {
    _ = fdflags;
    if (dirfd < 0) return wasi_core.WASI_EBADF;
    if (path_ptr + path_len > mem.len) return wasi_core.WASI_EINVAL;
    const path = mem[path_ptr..][0..path_len];

    const dir_entry = ctx.fd_table.get(@intCast(dirfd)) orelse return wasi_core.WASI_EBADF;
    const host_dir = dir_entry.host_dir orelse return wasi_core.WASI_EBADF;

    // Detect the OFLAGS_DIRECTORY flag (bit 1, value 0x2). If set, open a
    // directory; otherwise open a file. We don't yet honor CREAT/TRUNC/EXCL.
    const want_dir = (oflags & 0x2) != 0;

    if (want_dir) {
        var new_dir = host_dir.openDir(ctx.io, path, .{ .iterate = true }) catch
            return wasi_core.WASI_EBADF;
        const new_fd = ctx.fd_table.allocateFd();
        ctx.fd_table.insert(new_fd, .{
            .kind = .directory,
            .host_dir = new_dir,
        }) catch {
            new_dir.close(ctx.io);
            return wasi_core.WASI_EINVAL;
        };
        if (!wasi_core.memWriteU32(mem, fd_ptr, new_fd)) return wasi_core.WASI_EINVAL;
        return wasi_core.WASI_ESUCCESS;
    }

    var file = host_dir.openFile(ctx.io, path, .{ .mode = .read_write }) catch |err| switch (err) {
        error.FileNotFound => return @intCast(@intFromEnum(wasi.Errno.noent)),
        error.AccessDenied => host_dir.openFile(ctx.io, path, .{ .mode = .read_only }) catch
            return @intCast(@intFromEnum(wasi.Errno.acces)),
        else => return wasi_core.WASI_EINVAL,
    };
    const new_fd = ctx.fd_table.allocateFd();
    ctx.fd_table.insert(new_fd, .{
        .kind = .regular_file,
        .host_fd = file.handle,
    }) catch {
        file.close(ctx.io);
        return wasi_core.WASI_EINVAL;
    };
    if (!wasi_core.memWriteU32(mem, fd_ptr, new_fd)) return wasi_core.WASI_EINVAL;
    return wasi_core.WASI_ESUCCESS;
}

// ── fd metadata core helpers (issue #420 phase 1) ─────────────────────

/// `filestat` layout (64 bytes): dev(u64) + ino(u64) + filetype(u8 + 7 pad)
/// + nlink(u64) + size(u64) + atim(u64) + mtim(u64) + ctim(u64).
fn writeFilestat(mem: []u8, buf_ptr: u32, stat: anytype, filetype: wasi.Filetype) i32 {
    if (buf_ptr + 64 > mem.len) return wasi_core.WASI_EINVAL;
    @memset(mem[buf_ptr..][0..64], 0);
    // dev: stable per-mount id; std.Io.File.Stat doesn't expose it, so
    // synthesize zero (wasi guests typically only use ino for tracking).
    _ = wasi_core.memWriteU64(mem, buf_ptr + 0, 0);
    _ = wasi_core.memWriteU64(mem, buf_ptr + 8, @intCast(stat.inode));
    mem[buf_ptr + 16] = @intFromEnum(filetype);
    _ = wasi_core.memWriteU64(mem, buf_ptr + 24, @intCast(stat.nlink));
    _ = wasi_core.memWriteU64(mem, buf_ptr + 32, stat.size);
    const atim_ns: u64 = if (stat.atime) |t| timestampToNs(t) else 0;
    _ = wasi_core.memWriteU64(mem, buf_ptr + 40, atim_ns);
    _ = wasi_core.memWriteU64(mem, buf_ptr + 48, timestampToNs(stat.mtime));
    _ = wasi_core.memWriteU64(mem, buf_ptr + 56, timestampToNs(stat.ctime));
    return wasi_core.WASI_ESUCCESS;
}

fn timestampToNs(ts: std.Io.Timestamp) u64 {
    if (ts.nanoseconds < 0) return 0;
    return @intCast(ts.nanoseconds);
}

fn filetypeFromIoKind(kind: std.Io.File.Kind) wasi.Filetype {
    return switch (kind) {
        .block_device => .block_device,
        .character_device => .character_device,
        .directory => .directory,
        .named_pipe, .unix_domain_socket => .socket_stream,
        .sym_link => .symbolic_link,
        .file => .regular_file,
        else => .unknown,
    };
}

fn ctxFdFilestatGetCore(ctx: *wasi.WasiCtx, mem: []u8, fd: i32, buf_ptr: u32) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const u_fd: u32 = @intCast(fd);
    const entry = ctx.fd_table.get(u_fd) orelse return wasi_core.WASI_EBADF;

    if (entry.kind == .directory) {
        if (entry.host_dir) |dir| {
            const stat = dir.stat(ctx.io) catch return errnoToI32(error.IoError);
            return writeFilestat(mem, buf_ptr, stat, .directory);
        }
        return wasi_core.WASI_EBADF;
    }

    const host_fd = entryHostFd(entry) orelse return wasi_core.WASI_EBADF;
    const file = std.Io.File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
    const stat = file.stat(ctx.io) catch |err| switch (err) {
        error.Streaming => {
            // Pipes / ttys can't be stat'd. Synthesise a zeroed filestat
            // and best-effort filetype.
            return writeFilestatSynthesised(mem, buf_ptr, filetypeForEntry(entry));
        },
        else => return errnoToI32(err),
    };
    const filetype: wasi.Filetype = switch (entry.kind) {
        .regular_file => filetypeFromIoKind(stat.kind),
        .socket => .socket_stream,
        else => filetypeForEntry(entry),
    };
    return writeFilestat(mem, buf_ptr, stat, filetype);
}

fn writeFilestatSynthesised(mem: []u8, buf_ptr: u32, filetype: wasi.Filetype) i32 {
    if (buf_ptr + 64 > mem.len) return wasi_core.WASI_EINVAL;
    @memset(mem[buf_ptr..][0..64], 0);
    mem[buf_ptr + 16] = @intFromEnum(filetype);
    return wasi_core.WASI_ESUCCESS;
}

fn ctxFdFilestatSetSizeCore(ctx: *wasi.WasiCtx, fd: i32, size: i64) i32 {
    if (size < 0) return wasi_core.WASI_EINVAL;
    if (fd < 0) return wasi_core.WASI_EBADF;

    const u_fd: u32 = @intCast(fd);
    const entry = ctx.fd_table.get(u_fd) orelse return wasi_core.WASI_EBADF;
    if (entry.kind == .directory) return @intCast(@intFromEnum(wasi.Errno.isdir));
    if (entry.kind != .regular_file) return @intCast(@intFromEnum(wasi.Errno.inval));

    if (builtin.os.tag != .linux) return wasi_core.WASI_ENOSYS;

    const linux = std.os.linux;
    const host_fd = entryHostFd(entry) orelse return wasi_core.WASI_EBADF;
    const rc = linux.ftruncate(@intCast(host_fd), size);
    return mapLinuxErrno(rc);
}

fn ctxFdFilestatSetTimesCore(ctx: *wasi.WasiCtx, fd: i32, atim: u64, mtim: u64, fst_flags: u16) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const exclusive = wasi.FSTFLAGS_ATIM | wasi.FSTFLAGS_ATIM_NOW;
    if ((fst_flags & exclusive) == exclusive) return wasi_core.WASI_EINVAL;
    const exclusive_m = wasi.FSTFLAGS_MTIM | wasi.FSTFLAGS_MTIM_NOW;
    if ((fst_flags & exclusive_m) == exclusive_m) return wasi_core.WASI_EINVAL;

    const u_fd: u32 = @intCast(fd);
    const entry = ctx.fd_table.get(u_fd) orelse return wasi_core.WASI_EBADF;

    if (builtin.os.tag != .linux) return wasi_core.WASI_ENOSYS;
    const linux = std.os.linux;
    var times: [2]linux.timespec = undefined;
    times[0] = nsToFutimens(atim, fst_flags, wasi.FSTFLAGS_ATIM, wasi.FSTFLAGS_ATIM_NOW);
    times[1] = nsToFutimens(mtim, fst_flags, wasi.FSTFLAGS_MTIM, wasi.FSTFLAGS_MTIM_NOW);

    const host_fd = entryHostFd(entry) orelse return wasi_core.WASI_EBADF;
    const rc = linux.futimens(@intCast(host_fd), &times);
    return mapLinuxErrno(rc);
}

fn nsToFutimens(ns: u64, flags: u16, set_bit: u16, now_bit: u16) std.os.linux.timespec {
    if ((flags & now_bit) != 0) return std.os.linux.UTIME.NOW;
    if ((flags & set_bit) == 0) return std.os.linux.UTIME.OMIT;
    const sec: isize = @intCast(ns / std.time.ns_per_s);
    const nsec: isize = @intCast(ns % std.time.ns_per_s);
    return .{ .sec = sec, .nsec = nsec };
}

fn ctxFdFdstatSetFlagsCore(ctx: *wasi.WasiCtx, fd: i32, fdflags: u16) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    if ((fdflags & ~wasi.FDFLAGS_ALL) != 0) return wasi_core.WASI_EINVAL;

    const u_fd: u32 = @intCast(fd);
    const entry_ptr = ctx.fd_table.entries.getPtr(u_fd) orelse return wasi_core.WASI_EBADF;
    if (entry_ptr.kind == .directory) return wasi_core.WASI_EBADF;

    // SYNC/DSYNC/RSYNC can't be toggled via F_SETFL on Linux, and we
    // have no portable way to apply them on macOS/Windows either, so
    // reject any request that tries to change them on every platform.
    // Otherwise guests would see a silent success-on-no-op.
    if ((fdflags & (wasi.FDFLAGS_DSYNC | wasi.FDFLAGS_RSYNC | wasi.FDFLAGS_SYNC)) != 0) {
        return @intCast(@intFromEnum(wasi.Errno.notsup));
    }

    if (builtin.os.tag == .linux) {
        const linux = std.os.linux;
        const host_fd = entryHostFd(entry_ptr.*) orelse return wasi_core.WASI_EBADF;
        const cur = linux.fcntl(host_fd, linux.F.GETFL, 0);
        if (linux.errno(cur) != .SUCCESS) return mapLinuxErrno(cur);

        var o: linux.O = @bitCast(@as(u32, @intCast(cur & 0xFFFF_FFFF)));
        o.APPEND = (fdflags & wasi.FDFLAGS_APPEND) != 0;
        o.NONBLOCK = (fdflags & wasi.FDFLAGS_NONBLOCK) != 0;
        const new_flags: u32 = @bitCast(o);

        const rc = linux.fcntl(host_fd, linux.F.SETFL, new_flags);
        if (linux.errno(rc) != .SUCCESS) return mapLinuxErrno(rc);
    }

    entry_ptr.fdflags = fdflags;
    return wasi_core.WASI_ESUCCESS;
}

fn ctxFdFdstatSetRightsCore(ctx: *wasi.WasiCtx, fd: i32, base: u64, inheriting: u64) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const u_fd: u32 = @intCast(fd);
    const entry_ptr = ctx.fd_table.entries.getPtr(u_fd) orelse return wasi_core.WASI_EBADF;
    if ((base & ~entry_ptr.rights_base) != 0) {
        return @intCast(@intFromEnum(wasi.Errno.notcapable));
    }
    if ((inheriting & ~entry_ptr.rights_inheriting) != 0) {
        return @intCast(@intFromEnum(wasi.Errno.notcapable));
    }
    entry_ptr.rights_base = base;
    entry_ptr.rights_inheriting = inheriting;
    return wasi_core.WASI_ESUCCESS;
}

fn ctxFdAdviseCore(ctx: *wasi.WasiCtx, fd: i32, offset: i64, len: i64, advice: u8) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    if (offset < 0 or len < 0) return wasi_core.WASI_EINVAL;
    if (advice > @intFromEnum(wasi.Advice.noreuse)) return wasi_core.WASI_EINVAL;

    const u_fd: u32 = @intCast(fd);
    const entry = ctx.fd_table.get(u_fd) orelse return wasi_core.WASI_EBADF;
    if (entry.kind != .regular_file) return @intCast(@intFromEnum(wasi.Errno.spipe));

    if (builtin.os.tag != .linux) return wasi_core.WASI_ESUCCESS;

    const linux = std.os.linux;
    const linux_advice: usize = switch (@as(wasi.Advice, @enumFromInt(advice))) {
        .normal => linux.POSIX_FADV.NORMAL,
        .sequential => linux.POSIX_FADV.SEQUENTIAL,
        .random => linux.POSIX_FADV.RANDOM,
        .willneed => linux.POSIX_FADV.WILLNEED,
        .dontneed => linux.POSIX_FADV.DONTNEED,
        .noreuse => linux.POSIX_FADV.NOREUSE,
    };
    const host_fd = entryHostFd(entry) orelse return wasi_core.WASI_EBADF;
    const rc = linux.fadvise(host_fd, offset, len, linux_advice);
    return mapLinuxErrno(rc);
}

fn ctxFdAllocateCore(ctx: *wasi.WasiCtx, fd: i32, offset: i64, len: i64) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    if (offset < 0 or len < 0) return wasi_core.WASI_EINVAL;

    const u_fd: u32 = @intCast(fd);
    const entry = ctx.fd_table.get(u_fd) orelse return wasi_core.WASI_EBADF;
    if (entry.kind != .regular_file) return @intCast(@intFromEnum(wasi.Errno.spipe));

    if (builtin.os.tag != .linux) return wasi_core.WASI_ENOSYS;

    const linux = std.os.linux;
    const host_fd = entryHostFd(entry) orelse return wasi_core.WASI_EBADF;
    // mode = 0 means "extend the file as needed".
    const rc = linux.fallocate(@intCast(host_fd), 0, offset, len);
    if (linux.errno(rc) == .SUCCESS) return wasi_core.WASI_ESUCCESS;
    // Fall back to ftruncate-extend on filesystems where fallocate is
    // unsupported (e.g. tmpfs on some kernels reports ENOTSUP).
    const e = linux.errno(rc);
    if (e == .OPNOTSUPP) {
        const file = std.Io.File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
        const cur_size: u64 = (file.length(ctx.io) catch return mapLinuxErrno(rc));
        const new_end: u64 = @intCast(offset + len);
        if (new_end > cur_size) {
            file.setLength(ctx.io, new_end) catch return wasi_core.WASI_EINVAL;
        }
        return wasi_core.WASI_ESUCCESS;
    }
    return mapLinuxErrno(rc);
}

const SyncMode = enum { data, full };

fn ctxFdSyncCore(ctx: *wasi.WasiCtx, fd: i32, mode: SyncMode) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const u_fd: u32 = @intCast(fd);
    const entry = ctx.fd_table.get(u_fd) orelse return wasi_core.WASI_EBADF;

    if (entry.kind == .directory) {
        if (entry.host_dir) |dir| {
            if (builtin.os.tag == .linux) {
                const rc = std.os.linux.fsync(dir.handle);
                return mapLinuxErrno(rc);
            }
            return wasi_core.WASI_ESUCCESS;
        }
        return wasi_core.WASI_EBADF;
    }

    const host_fd = entryHostFd(entry) orelse return wasi_core.WASI_EBADF;
    if (builtin.os.tag == .linux and mode == .data) {
        std.posix.fdatasync(host_fd) catch |err| return errnoToI32(err);
        return wasi_core.WASI_ESUCCESS;
    }
    const file = std.Io.File{ .handle = host_fd, .flags = .{ .nonblocking = false } };
    file.sync(ctx.io) catch |err| return errnoToI32(err);
    return wasi_core.WASI_ESUCCESS;
}

fn ctxFdTellCore(ctx: *wasi.WasiCtx, mem: []u8, fd: i32, offset_ptr: u32) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const u_fd: u32 = @intCast(fd);
    const entry = ctx.fd_table.get(u_fd) orelse return wasi_core.WASI_EBADF;
    if (entry.kind != .regular_file) {
        return @intCast(@intFromEnum(wasi.Errno.spipe));
    }
    if (!wasi_core.memWriteU64(mem, offset_ptr, entry.pos)) return wasi_core.WASI_EINVAL;
    return wasi_core.WASI_ESUCCESS;
}

/// Map a raw linux syscall return into a WASI errno (or success).
fn mapLinuxErrno(rc: usize) i32 {
    const linux = std.os.linux;
    return switch (linux.errno(rc)) {
        .SUCCESS => wasi_core.WASI_ESUCCESS,
        .BADF => wasi_core.WASI_EBADF,
        .INVAL => wasi_core.WASI_EINVAL,
        .ACCES => @intCast(@intFromEnum(wasi.Errno.acces)),
        .PERM => @intCast(@intFromEnum(wasi.Errno.perm)),
        .NOSPC => @intCast(@intFromEnum(wasi.Errno.nospc)),
        .ROFS => @intCast(@intFromEnum(wasi.Errno.rofs)),
        .ISDIR => @intCast(@intFromEnum(wasi.Errno.isdir)),
        .NOTDIR => @intCast(@intFromEnum(wasi.Errno.notdir)),
        .NOENT => @intCast(@intFromEnum(wasi.Errno.noent)),
        .EXIST => @intCast(@intFromEnum(wasi.Errno.exist)),
        .FBIG => @intCast(@intFromEnum(wasi.Errno.fbig)),
        .IO => @intCast(@intFromEnum(wasi.Errno.io)),
        .SPIPE => @intCast(@intFromEnum(wasi.Errno.spipe)),
        .OPNOTSUPP => @intCast(@intFromEnum(wasi.Errno.notsup)),
        .DQUOT => @intCast(@intFromEnum(wasi.Errno.dquot)),
        .NXIO => @intCast(@intFromEnum(wasi.Errno.nxio)),
        else => wasi_core.WASI_EINVAL,
    };
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
        .{ "fd_read", &wasiFdRead },
        .{ "fd_pread", &wasiFdPread },
        .{ "fd_pwrite", &wasiFdPwrite },
        .{ "fd_readdir", &wasiFdReaddir },
        .{ "fd_seek", &wasiFdSeek },
        .{ "fd_close", &wasiFdClose },
        .{ "fd_fdstat_get", &wasiFdFdstatGet },
        .{ "fd_fdstat_set_flags", &wasiFdFdstatSetFlags },
        .{ "fd_fdstat_set_rights", &wasiFdFdstatSetRights },
        .{ "fd_filestat_get", &wasiFdFilestatGet },
        .{ "fd_filestat_set_size", &wasiFdFilestatSetSize },
        .{ "fd_filestat_set_times", &wasiFdFilestatSetTimes },
        .{ "fd_advise", &wasiFdAdvise },
        .{ "fd_allocate", &wasiFdAllocate },
        .{ "fd_datasync", &wasiFdDatasync },
        .{ "fd_sync", &wasiFdSync },
        .{ "fd_tell", &wasiFdTell },
        .{ "fd_prestat_get", &wasiFdPrestatGet },
        .{ "fd_prestat_dir_name", &wasiFdPrestatDirName },
        .{ "clock_time_get", &wasiClockTimeGet },
        .{ "environ_sizes_get", &wasiEnvironSizesGet },
        .{ "environ_get", &wasiEnvironGet },
        .{ "args_sizes_get", &wasiArgsSizesGet },
        .{ "args_get", &wasiArgsGet },
        .{ "random_get", &wasiRandomGet },
        .{ "path_open", &wasiPathOpen },
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

test "resolveWasiFunction: all 29 functions resolve" {
    const names = [_][]const u8{
        "proc_exit",          "thread-spawn",         "fd_write",
        "fd_read",            "fd_pread",             "fd_pwrite",
        "fd_readdir",         "fd_seek",              "fd_close",
        "fd_fdstat_get",      "fd_fdstat_set_flags",  "fd_fdstat_set_rights",
        "fd_filestat_get",    "fd_filestat_set_size", "fd_filestat_set_times",
        "fd_advise",          "fd_allocate",          "fd_datasync",
        "fd_sync",            "fd_tell",              "fd_prestat_get",
        "fd_prestat_dir_name", "clock_time_get",      "environ_sizes_get",
        "environ_get",        "args_sizes_get",       "args_get",
        "random_get",         "path_open",
    };
    for (names) |name| {
        const result = resolveWasiFunction(name);
        try std.testing.expect(result != null);
    }
}

const testing_io = std.testing.io;

test "ctxFdFdstatSetFlagsCore: bad fd" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdFdstatSetFlagsCore(ctx, -1, 0));
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdFdstatSetFlagsCore(ctx, 99, 0));
}

test "ctxFdFdstatSetFlagsCore: invalid bits return EINVAL" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    try std.testing.expectEqual(wasi_core.WASI_EINVAL, ctxFdFdstatSetFlagsCore(ctx, 1, 0x8000));
}

test "ctxFdFdstatSetFlagsCore: SYNC bits return notsup" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    const expected: i32 = @intCast(@intFromEnum(wasi.Errno.notsup));
    try std.testing.expectEqual(expected, ctxFdFdstatSetFlagsCore(ctx, 1, wasi.FDFLAGS_SYNC));
    try std.testing.expectEqual(expected, ctxFdFdstatSetFlagsCore(ctx, 1, wasi.FDFLAGS_DSYNC));
    try std.testing.expectEqual(expected, ctxFdFdstatSetFlagsCore(ctx, 1, wasi.FDFLAGS_RSYNC));
}

test "ctxFdFdstatSetRightsCore: narrow ok, widen rejected" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    try ctx.fd_table.insert(50, .{
        .kind = .regular_file,
        .rights_base = 0xFF,
        .rights_inheriting = 0xFF,
    });
    try std.testing.expectEqual(
        wasi_core.WASI_ESUCCESS,
        ctxFdFdstatSetRightsCore(ctx, 50, 0x0F, 0x0F),
    );
    const after_narrow = ctx.fd_table.get(50).?;
    try std.testing.expectEqual(@as(u64, 0x0F), after_narrow.rights_base);
    try std.testing.expectEqual(@as(u64, 0x0F), after_narrow.rights_inheriting);
    const expected: i32 = @intCast(@intFromEnum(wasi.Errno.notcapable));
    try std.testing.expectEqual(expected, ctxFdFdstatSetRightsCore(ctx, 50, 0xFF, 0x0F));
    try std.testing.expectEqual(@as(u64, 0x0F), ctx.fd_table.get(50).?.rights_base);
}

test "ctxFdFdstatSetRightsCore: bad fd" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdFdstatSetRightsCore(ctx, -1, 0, 0));
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdFdstatSetRightsCore(ctx, 99, 0, 0));
}

test "ctxFdAdviseCore: invalid advice returns EINVAL" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    try ctx.fd_table.insert(60, .{ .kind = .regular_file });
    try std.testing.expectEqual(wasi_core.WASI_EINVAL, ctxFdAdviseCore(ctx, 60, 0, 0, 99));
}

test "ctxFdAdviseCore: bad fd" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdAdviseCore(ctx, -1, 0, 0, 0));
}

test "ctxFdSyncCore: bad fd" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdSyncCore(ctx, -1, .full));
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdSyncCore(ctx, 99, .data));
}

test "ctxFdTellCore: bad fd / spipe / cached pos" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    var mem: [16]u8 = @splat(0);

    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdTellCore(ctx, &mem, -1, 0));

    const spipe: i32 = @intCast(@intFromEnum(wasi.Errno.spipe));
    try std.testing.expectEqual(spipe, ctxFdTellCore(ctx, &mem, 1, 0));

    try ctx.fd_table.insert(70, .{ .kind = .regular_file, .pos = 0xDEADBEEF });
    try std.testing.expectEqual(wasi_core.WASI_ESUCCESS, ctxFdTellCore(ctx, &mem, 70, 0));
    try std.testing.expectEqual(@as(u64, 0xDEADBEEF), wasi_core.memReadU64(&mem, 0).?);
}

test "ctxFdFilestatSetTimesCore: conflicting fst_flags return EINVAL" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    try ctx.fd_table.insert(80, .{ .kind = .regular_file });
    try std.testing.expectEqual(
        wasi_core.WASI_EINVAL,
        ctxFdFilestatSetTimesCore(ctx, 80, 0, 0, wasi.FSTFLAGS_ATIM | wasi.FSTFLAGS_ATIM_NOW),
    );
    try std.testing.expectEqual(
        wasi_core.WASI_EINVAL,
        ctxFdFilestatSetTimesCore(ctx, 80, 0, 0, wasi.FSTFLAGS_MTIM | wasi.FSTFLAGS_MTIM_NOW),
    );
}

test "ctxFdFilestatGetCore: bad fd" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    var mem: [128]u8 = @splat(0);
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdFilestatGetCore(ctx, &mem, -1, 0));
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdFilestatGetCore(ctx, &mem, 99, 0));
}

test "ctxFdFilestatSetSizeCore: bad fd / directory rejected" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdFilestatSetSizeCore(ctx, -1, 0));
    try ctx.fd_table.insert(90, .{ .kind = .directory });
    const isdir: i32 = @intCast(@intFromEnum(wasi.Errno.isdir));
    try std.testing.expectEqual(isdir, ctxFdFilestatSetSizeCore(ctx, 90, 0));
}

test "ctxFdPreadCore: bad fd / negative offset / spipe / isdir" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    var mem: [32]u8 = @splat(0);

    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdPreadCore(ctx, &mem, -1, 0, 0, 0, 0));
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdPreadCore(ctx, &mem, 99, 0, 0, 0, 0));
    try std.testing.expectEqual(wasi_core.WASI_EINVAL, ctxFdPreadCore(ctx, &mem, 1, 0, 0, -1, 0));

    const spipe: i32 = @intCast(@intFromEnum(wasi.Errno.spipe));
    try std.testing.expectEqual(spipe, ctxFdPreadCore(ctx, &mem, 0, 0, 0, 0, 0));
    try std.testing.expectEqual(spipe, ctxFdPreadCore(ctx, &mem, 1, 0, 0, 0, 0));
    try std.testing.expectEqual(spipe, ctxFdPreadCore(ctx, &mem, 2, 0, 0, 0, 0));

    try ctx.fd_table.insert(90, .{ .kind = .directory });
    const isdir: i32 = @intCast(@intFromEnum(wasi.Errno.isdir));
    try std.testing.expectEqual(isdir, ctxFdPreadCore(ctx, &mem, 90, 0, 0, 0, 0));
}

test "ctxFdPwriteCore: bad fd / negative offset / spipe / isdir / append rejected" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    var mem: [32]u8 = @splat(0);

    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdPwriteCore(ctx, &mem, -1, 0, 0, 0, 0));
    try std.testing.expectEqual(wasi_core.WASI_EINVAL, ctxFdPwriteCore(ctx, &mem, 1, 0, 0, -1, 0));

    const spipe: i32 = @intCast(@intFromEnum(wasi.Errno.spipe));
    try std.testing.expectEqual(spipe, ctxFdPwriteCore(ctx, &mem, 1, 0, 0, 0, 0));

    try ctx.fd_table.insert(91, .{ .kind = .directory });
    const isdir: i32 = @intCast(@intFromEnum(wasi.Errno.isdir));
    try std.testing.expectEqual(isdir, ctxFdPwriteCore(ctx, &mem, 91, 0, 0, 0, 0));

    // Append-mode regular file rejected with notsup, but only after the
    // pre-checks pass (so the fdflags branch actually runs).
    try ctx.fd_table.insert(92, .{ .kind = .regular_file, .fdflags = wasi.FDFLAGS_APPEND });
    const notsup: i32 = @intCast(@intFromEnum(wasi.Errno.notsup));
    try std.testing.expectEqual(notsup, ctxFdPwriteCore(ctx, &mem, 92, 0, 0, 0, 0));
}

test "ctxFdReaddirCore: bad fd / notdir / inval buf" {
    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();
    var mem: [16]u8 = @splat(0);

    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdReaddirCore(ctx, &mem, -1, 0, 0, 0, 0));
    try std.testing.expectEqual(wasi_core.WASI_EBADF, ctxFdReaddirCore(ctx, &mem, 99, 0, 0, 0, 0));

    // Regular file → notdir.
    try ctx.fd_table.insert(93, .{ .kind = .regular_file });
    const notdir: i32 = @intCast(@intFromEnum(wasi.Errno.notdir));
    try std.testing.expectEqual(notdir, ctxFdReaddirCore(ctx, &mem, 93, 0, 0, 0, 0));

    // buf_ptr + buf_len out of bounds → einval. Use a fresh dir entry
    // (host_dir = null is fine since the bound-check runs first).
    try ctx.fd_table.insert(94, .{ .kind = .directory });
    try std.testing.expectEqual(wasi_core.WASI_EINVAL, ctxFdReaddirCore(ctx, &mem, 94, 8, 9, 0, 0));
}

test "ctxFdPreadCore: reads at offset without moving cached pos" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile(testing_io, "pread.bin", .{ .read = true });
    defer file.close(testing_io);
    try file.writePositionalAll(testing_io, "abcdefghij", 0);

    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const fd = ctx.fd_table.allocateFd();
    try ctx.fd_table.insert(fd, .{
        .kind = .regular_file,
        .host_fd = file.handle,
        .pos = 5,
    });
    defer ctx.fd_table.remove(fd);

    // mem layout: [0..4] iov0.buf_ptr=12, [4..8] iov0.buf_len=4,
    //             [8..12] nread, [12..16] dst.
    var mem: [32]u8 = @splat(0);
    _ = wasi_core.memWriteU32(&mem, 0, 12);
    _ = wasi_core.memWriteU32(&mem, 4, 4);

    try std.testing.expectEqual(
        wasi_core.WASI_ESUCCESS,
        ctxFdPreadCore(ctx, &mem, @intCast(fd), 0, 1, 3, 8),
    );
    try std.testing.expectEqualStrings("defg", mem[12..16]);
    try std.testing.expectEqual(@as(u32, 4), wasi_core.memReadU32(&mem, 8).?);

    // Cached pos must be unchanged — fd_pread is positional.
    const after = ctx.fd_table.get(fd).?;
    try std.testing.expectEqual(@as(u64, 5), after.pos);
}

test "ctxFdPwriteCore: writes at offset without moving cached pos" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile(testing_io, "pwrite.bin", .{ .read = true });
    defer file.close(testing_io);
    // Seed 10 bytes of zeros so we can pwrite into the middle.
    try file.writePositionalAll(testing_io, "..........", 0);

    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const fd = ctx.fd_table.allocateFd();
    try ctx.fd_table.insert(fd, .{
        .kind = .regular_file,
        .host_fd = file.handle,
        .pos = 0,
    });
    defer ctx.fd_table.remove(fd);

    // mem: [0..4] iov.buf_ptr=12, [4..8] iov.buf_len=3, [8..12] nwritten,
    //      [12..16] src "XYZ".
    var mem: [32]u8 = @splat(0);
    _ = wasi_core.memWriteU32(&mem, 0, 12);
    _ = wasi_core.memWriteU32(&mem, 4, 3);
    @memcpy(mem[12..15], "XYZ");

    try std.testing.expectEqual(
        wasi_core.WASI_ESUCCESS,
        ctxFdPwriteCore(ctx, &mem, @intCast(fd), 0, 1, 4, 8),
    );
    try std.testing.expectEqual(@as(u32, 3), wasi_core.memReadU32(&mem, 8).?);
    try std.testing.expectEqual(@as(u64, 0), ctx.fd_table.get(fd).?.pos);

    var read_back: [10]u8 = undefined;
    const n = try file.readPositionalAll(testing_io, &read_back, 0);
    try std.testing.expectEqualStrings("....XYZ...", read_back[0..n]);
}

test "ctxFdReaddirCore: encodes preview1 dirents" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // Populate a small set of entries; getdents64 also returns "." and "..".
    const f_a = try tmp.dir.createFile(testing_io, "alpha", .{});
    f_a.close(testing_io);
    const f_b = try tmp.dir.createFile(testing_io, "beta", .{});
    f_b.close(testing_io);

    const ctx = try wasi.WasiCtx.init(std.testing.allocator, testing_io);
    defer ctx.deinit();

    const fd = ctx.fd_table.allocateFd();
    try ctx.fd_table.insert(fd, .{
        .kind = .directory,
        .host_dir = tmp.dir,
    });
    // Drop the entry before deinit so we don't double-close the dir.
    defer ctx.fd_table.remove(fd);

    // 4 KiB scratch buffer in linear memory.
    var mem: [4096]u8 = @splat(0);
    const buf_ptr: u32 = 8;
    const buf_len: u32 = 4096 - 8;
    const bufused_ptr: u32 = 0;

    try std.testing.expectEqual(
        wasi_core.WASI_ESUCCESS,
        ctxFdReaddirCore(ctx, &mem, @intCast(fd), buf_ptr, buf_len, 0, bufused_ptr),
    );

    const bufused = wasi_core.memReadU32(&mem, bufused_ptr).?;
    try std.testing.expect(bufused > 0);
    try std.testing.expect(bufused <= buf_len);

    // Walk the encoded entries and verify we see "alpha" and "beta" with
    // filetype = regular_file. We don't insist on order; getdents64 may
    // also surface "." / ".." entries.
    var saw_alpha = false;
    var saw_beta = false;
    var off: u32 = 0;
    while (off + 24 <= bufused) {
        const namlen = wasi_core.memReadU32(&mem, buf_ptr + off + 16).?;
        const dt = mem[buf_ptr + off + 20];
        const name_start = off + 24;
        if (name_start + namlen > bufused) break; // truncated tail
        const name = mem[buf_ptr + name_start ..][0..namlen];
        if (std.mem.eql(u8, name, "alpha")) {
            saw_alpha = true;
            try std.testing.expectEqual(
                @intFromEnum(wasi.Filetype.regular_file),
                dt,
            );
        } else if (std.mem.eql(u8, name, "beta")) {
            saw_beta = true;
            try std.testing.expectEqual(
                @intFromEnum(wasi.Filetype.regular_file),
                dt,
            );
        }
        off = name_start + namlen;
    }
    try std.testing.expect(saw_alpha);
    try std.testing.expect(saw_beta);
}
