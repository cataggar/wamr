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
fn writeFdstat(mem: []u8, buf_ptr: u32, kind: wasi.FdEntry.FdKind) i32 {
    if (buf_ptr + 24 > mem.len) return wasi_core.WASI_EINVAL;
    @memset(mem[buf_ptr..][0..24], 0);
    const filetype: u8 = switch (kind) {
        .stdin, .stdout, .stderr => 2, // character device
        .regular_file => 4,
        .directory => 3,
        .socket => 6,
    };
    mem[buf_ptr] = filetype;
    // Permissive rights so downstream tests that read fs_rights_base before
    // read/write don't trip on missing capabilities.
    _ = wasi_core.memWriteU64(mem, buf_ptr + 8, 0xFFFF_FFFF_FFFF_FFFF);
    _ = wasi_core.memWriteU64(mem, buf_ptr + 16, 0xFFFF_FFFF_FFFF_FFFF);
    return wasi_core.WASI_ESUCCESS;
}

fn ctxFdFdstatGetCore(ctx: *wasi.WasiCtx, mem: []u8, fd: i32, buf_ptr: u32) i32 {
    if (fd < 0) return wasi_core.WASI_EBADF;
    const entry = ctx.fd_table.get(@intCast(fd)) orelse return wasi_core.WASI_EBADF;
    return writeFdstat(mem, buf_ptr, entry.kind);
}

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

test "resolveWasiFunction: all 16 functions resolve" {
    const names = [_][]const u8{
        "proc_exit",     "thread-spawn",       "fd_write",
        "fd_read",       "fd_seek",            "fd_close",
        "fd_fdstat_get", "fd_prestat_get",     "fd_prestat_dir_name",
        "clock_time_get", "environ_sizes_get", "environ_get",
        "args_sizes_get", "args_get",          "random_get",
        "path_open",
    };
    for (names) |name| {
        const result = resolveWasiFunction(name);
        try std.testing.expect(result != null);
    }
}
