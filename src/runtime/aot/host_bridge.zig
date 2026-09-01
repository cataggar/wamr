//! AOT host function bridge.
//!
//! Provides C-calling-convention adapter functions that bridge AOT-compiled
//! code to the shared WASI core logic.  Each adapter receives a VmCtx pointer
//! as first argument (same as all AOT functions), followed by the WASI
//! function's arguments in registers, and returns the WASI errno as i32.
//!
//! Two execution tiers, mirroring `src/wasi/host_functions.zig`:
//!   - **ctx-aware**: when `VmCtx.wasi_ctx != 0`, forward to the retained
//!     `WasiProcessState` through the same `ctxXxxCore` helpers used by the
//!     interpreter so args / env / preopens / fd-table / sockets all behave
//!     identically.
//!   - **legacy stub**: when no `WasiCtx` is attached (CoreMark / unit
//!     tests) we fall back to the stateless `wasi_core.zig` defaults
//!     (zero args, stdout-only) so older callers keep working.

const std = @import("std");
const aot_runtime = @import("runtime.zig");
const VmCtx = aot_runtime.VmCtx;
const wasi_core = @import("../../wasi/wasi_core.zig");
const wasi = @import("../../wasi/wasi.zig");
const host_functions = @import("../../wasi/host_functions.zig");
const host_trampolines = @import("host_trampolines.zig");

pub const TrampolinePool = host_trampolines.TrampolinePool;

var g_trampoline_pool: ?*TrampolinePool = null;

/// When true, every AOT WASI adapter prints a one-line trace on entry.
/// Set from `main.zig` via the `--trace-aot-wasi` CLI flag *before* the
/// instance is run. Module-level so adapters can read it cheaply without
/// threading a config struct through every call site.
pub var trace_enabled: bool = false;

inline fn traceAdapter(comptime name: []const u8, vmctx: *VmCtx, args: anytype) void {
    if (!trace_enabled) return;
    std.debug.print(
        "[aot-wasi] {s} vmctx=0x{x} mem_base=0x{x} mem_size=0x{x} wasi_ctx=0x{x} args={any}\n",
        .{ name, @intFromPtr(vmctx), vmctx.memory_base, vmctx.memory_size, vmctx.wasi_ctx, args },
    );
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Reconstruct a mutable memory slice from the VmCtx fields.
fn getMemoryFromCtx(vmctx: *VmCtx) ?[]u8 {
    if (vmctx.memory_base == 0 or vmctx.memory_size == 0) return null;
    const ptr: [*]u8 = @ptrFromInt(vmctx.memory_base);
    return ptr[0..vmctx.memory_size];
}

/// Resolve the process-scoped WASI state. Null when no state is attached.
fn getCtx(vmctx: *VmCtx) ?*wasi.WasiProcessState {
    if (vmctx.wasi_ctx == 0) return null;
    return @ptrFromInt(vmctx.wasi_ctx);
}

inline fn u16FromI32(v: i32) u16 {
    return @intCast(@as(u32, @bitCast(v)) & 0xFFFF);
}

inline fn u8FromI32(v: i32) u8 {
    return @intCast(@as(u32, @bitCast(v)) & 0xFF);
}

// ── AOT adapters ──────────────────────────────────────────────────────

pub fn aotFdWrite(vmctx: *VmCtx, fd: i32, iovs_ptr: i32, iovs_len: i32, nwritten_ptr: i32) callconv(.c) i32 {
    traceAdapter("fd_write", vmctx, .{ fd, iovs_ptr, iovs_len, nwritten_ptr });
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    if (getCtx(vmctx)) |ctx| {
        return host_functions.ctxFdIoCore(ctx, mem, fd, @bitCast(iovs_ptr), @bitCast(iovs_len), @bitCast(nwritten_ptr), .write);
    }
    return wasi_core.fdWriteCore(mem, fd, @bitCast(iovs_ptr), @bitCast(iovs_len), @bitCast(nwritten_ptr));
}

pub fn aotFdRead(vmctx: *VmCtx, fd: i32, iovs_ptr: i32, iovs_len: i32, nread_ptr: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_EBADF;
    const result = host_functions.ctxFdIoCore(ctx, mem, fd, @bitCast(iovs_ptr), @bitCast(iovs_len), @bitCast(nread_ptr), .read);
    // Parity with the interpreter: a read interrupted by the group's
    // terminal outcome unwinds this thread instead of returning an errno.
    if (result == host_functions.terminated_result)
        aot_runtime.terminateAotThread(vmctx);
    return result;
}

pub fn aotFdPread(vmctx: *VmCtx, fd: i32, iovs_ptr: i32, iovs_len: i32, offset: i64, nread_ptr: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxFdPreadCore(ctx, mem, fd, @bitCast(iovs_ptr), @bitCast(iovs_len), offset, @bitCast(nread_ptr));
}

pub fn aotFdPwrite(vmctx: *VmCtx, fd: i32, iovs_ptr: i32, iovs_len: i32, offset: i64, nwritten_ptr: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxFdPwriteCore(ctx, mem, fd, @bitCast(iovs_ptr), @bitCast(iovs_len), offset, @bitCast(nwritten_ptr));
}

pub fn aotFdReaddir(vmctx: *VmCtx, fd: i32, buf_ptr: i32, buf_len: i32, cookie: i64, bufused_ptr: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxFdReaddirCore(ctx, mem, fd, @bitCast(buf_ptr), @bitCast(buf_len), @bitCast(cookie), @bitCast(bufused_ptr));
}

pub fn aotFdSeek(vmctx: *VmCtx, fd: i32, offset: i64, whence: i32, newoffset_ptr: i32) callconv(.c) i32 {
    traceAdapter("fd_seek", vmctx, .{ fd, offset, whence, newoffset_ptr });
    if (getCtx(vmctx)) |ctx| {
        const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
        return host_functions.ctxFdSeekCore(ctx, mem, fd, offset, u8FromI32(whence), @bitCast(newoffset_ptr));
    }
    return wasi_core.fdSeekCore();
}

pub fn aotFdClose(vmctx: *VmCtx, fd: i32) callconv(.c) i32 {
    traceAdapter("fd_close", vmctx, .{fd});
    if (getCtx(vmctx)) |ctx| {
        if (fd < 0) return @intCast(@intFromEnum(wasi.Errno.badf));
        return @intCast(@intFromEnum(ctx.fd_close(@intCast(fd))));
    }
    return wasi_core.fdCloseCore(fd);
}

pub fn aotFdRenumber(vmctx: *VmCtx, from: i32, to: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxFdRenumberCore(ctx, from, to);
}

pub fn aotFdFdstatGet(vmctx: *VmCtx, fd: i32, buf_ptr: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    if (getCtx(vmctx)) |ctx| {
        return host_functions.ctxFdFdstatGetCore(ctx, mem, fd, @bitCast(buf_ptr));
    }
    return wasi_core.fdFdstatGetCore(mem, fd, @bitCast(buf_ptr));
}

pub fn aotFdFdstatSetFlags(vmctx: *VmCtx, fd: i32, fdflags: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxFdFdstatSetFlagsCore(ctx, fd, u16FromI32(fdflags));
}

pub fn aotFdFdstatSetRights(vmctx: *VmCtx, fd: i32, rights_base: i64, rights_inh: i64) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxFdFdstatSetRightsCore(ctx, fd, @bitCast(rights_base), @bitCast(rights_inh));
}

pub fn aotFdFilestatGet(vmctx: *VmCtx, fd: i32, buf_ptr: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxFdFilestatGetCore(ctx, mem, fd, @bitCast(buf_ptr));
}

pub fn aotFdFilestatSetSize(vmctx: *VmCtx, fd: i32, size: i64) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxFdFilestatSetSizeCore(ctx, fd, size);
}

pub fn aotFdFilestatSetTimes(vmctx: *VmCtx, fd: i32, atim: i64, mtim: i64, fst_flags: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxFdFilestatSetTimesCore(ctx, fd, @bitCast(atim), @bitCast(mtim), u16FromI32(fst_flags));
}

pub fn aotFdAdvise(vmctx: *VmCtx, fd: i32, offset: i64, len: i64, advice: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxFdAdviseCore(ctx, fd, offset, len, u8FromI32(advice));
}

pub fn aotFdAllocate(vmctx: *VmCtx, fd: i32, offset: i64, len: i64) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxFdAllocateCore(ctx, fd, offset, len);
}

pub fn aotFdDatasync(vmctx: *VmCtx, fd: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxFdSyncCore(ctx, fd, .data);
}

pub fn aotFdSync(vmctx: *VmCtx, fd: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxFdSyncCore(ctx, fd, .full);
}

pub fn aotFdTell(vmctx: *VmCtx, fd: i32, offset_ptr: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxFdTellCore(ctx, mem, fd, @bitCast(offset_ptr));
}

pub fn aotFdPrestatGet(vmctx: *VmCtx, fd: i32, buf_ptr: i32) callconv(.c) i32 {
    if (getCtx(vmctx)) |ctx| {
        const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
        return host_functions.ctxFdPrestatGetCore(ctx, mem, fd, @bitCast(buf_ptr));
    }
    return wasi_core.fdPrestatGetCore();
}

pub fn aotFdPrestatDirName(vmctx: *VmCtx, fd: i32, path_ptr: i32, path_len: i32) callconv(.c) i32 {
    if (getCtx(vmctx)) |ctx| {
        const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
        return host_functions.ctxFdPrestatDirNameCore(ctx, mem, fd, @bitCast(path_ptr), @bitCast(path_len));
    }
    return wasi_core.fdPrestatDirNameCore();
}

pub fn aotPathOpen(
    vmctx: *VmCtx,
    dirfd: i32,
    dirflags: i32,
    path_ptr: i32,
    path_len: i32,
    oflags: i32,
    fs_rights_base: i64,
    fs_rights_inh: i64,
    fdflags: i32,
    fd_ptr: i32,
) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathOpenCore(
        ctx,
        mem,
        dirfd,
        @bitCast(dirflags),
        @bitCast(path_ptr),
        @bitCast(path_len),
        @bitCast(oflags),
        @bitCast(fs_rights_base),
        @bitCast(fs_rights_inh),
        @bitCast(fdflags),
        @bitCast(fd_ptr),
    );
}

pub fn aotPathFilestatGet(
    vmctx: *VmCtx,
    fd: i32,
    lookup_flags: i32,
    path_ptr: i32,
    path_len: i32,
    buf_ptr: i32,
) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathFilestatGetCore(
        ctx,
        mem,
        fd,
        @bitCast(lookup_flags),
        @bitCast(path_ptr),
        @bitCast(path_len),
        @bitCast(buf_ptr),
    );
}

pub fn aotPathFilestatSetTimes(
    vmctx: *VmCtx,
    fd: i32,
    lookup_flags: i32,
    path_ptr: i32,
    path_len: i32,
    atim: i64,
    mtim: i64,
    fst_flags: i32,
) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathFilestatSetTimesCore(
        ctx,
        mem,
        fd,
        @bitCast(lookup_flags),
        @bitCast(path_ptr),
        @bitCast(path_len),
        @bitCast(atim),
        @bitCast(mtim),
        u16FromI32(fst_flags),
    );
}

pub fn aotPathCreateDirectory(vmctx: *VmCtx, fd: i32, path_ptr: i32, path_len: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathCreateDirectoryCore(ctx, mem, fd, @bitCast(path_ptr), @bitCast(path_len));
}

pub fn aotPathRemoveDirectory(vmctx: *VmCtx, fd: i32, path_ptr: i32, path_len: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathRemoveDirectoryCore(ctx, mem, fd, @bitCast(path_ptr), @bitCast(path_len));
}

pub fn aotPathUnlinkFile(vmctx: *VmCtx, fd: i32, path_ptr: i32, path_len: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathUnlinkFileCore(ctx, mem, fd, @bitCast(path_ptr), @bitCast(path_len));
}

pub fn aotPathLink(
    vmctx: *VmCtx,
    old_fd: i32,
    old_flags: i32,
    old_path_ptr: i32,
    old_path_len: i32,
    new_fd: i32,
    new_path_ptr: i32,
    new_path_len: i32,
) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathLinkCore(
        ctx,
        mem,
        old_fd,
        @bitCast(old_flags),
        @bitCast(old_path_ptr),
        @bitCast(old_path_len),
        new_fd,
        @bitCast(new_path_ptr),
        @bitCast(new_path_len),
    );
}

pub fn aotPathRename(
    vmctx: *VmCtx,
    old_fd: i32,
    old_path_ptr: i32,
    old_path_len: i32,
    new_fd: i32,
    new_path_ptr: i32,
    new_path_len: i32,
) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathRenameCore(
        ctx,
        mem,
        old_fd,
        @bitCast(old_path_ptr),
        @bitCast(old_path_len),
        new_fd,
        @bitCast(new_path_ptr),
        @bitCast(new_path_len),
    );
}

pub fn aotPathSymlink(
    vmctx: *VmCtx,
    old_path_ptr: i32,
    old_path_len: i32,
    fd: i32,
    new_path_ptr: i32,
    new_path_len: i32,
) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathSymlinkCore(
        ctx,
        mem,
        @bitCast(old_path_ptr),
        @bitCast(old_path_len),
        fd,
        @bitCast(new_path_ptr),
        @bitCast(new_path_len),
    );
}

pub fn aotPathReadlink(
    vmctx: *VmCtx,
    fd: i32,
    path_ptr: i32,
    path_len: i32,
    buf_ptr: i32,
    buf_len: i32,
    bufused_ptr: i32,
) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxPathReadlinkCore(
        ctx,
        mem,
        fd,
        @bitCast(path_ptr),
        @bitCast(path_len),
        @bitCast(buf_ptr),
        @bitCast(buf_len),
        @bitCast(bufused_ptr),
    );
}

pub fn aotClockTimeGet(vmctx: *VmCtx, clock_id: i32, _: i64, time_ptr: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return wasi_core.clockTimeGetCore(mem, clock_id, @bitCast(time_ptr));
}

pub fn aotClockResGet(vmctx: *VmCtx, clock_id: i32, resolution_ptr: i32) callconv(.c) i32 {
    traceAdapter("clock_res_get", vmctx, .{ clock_id, resolution_ptr });
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return wasi_core.clockResGetCore(mem, clock_id, @bitCast(resolution_ptr));
}

pub fn aotEnvironSizesGet(vmctx: *VmCtx, count_ptr: i32, buf_size_ptr: i32) callconv(.c) i32 {
    traceAdapter("environ_sizes_get", vmctx, .{ count_ptr, buf_size_ptr });
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    if (getCtx(vmctx)) |ctx| {
        const sizes = ctx.environ_sizes_get();
        if (!wasi_core.memWriteU32(mem, @bitCast(count_ptr), sizes.count) or
            !wasi_core.memWriteU32(mem, @bitCast(buf_size_ptr), sizes.buf_size))
        {
            return wasi_core.WASI_EINVAL;
        }
        return wasi_core.WASI_ESUCCESS;
    }
    return wasi_core.environSizesGetCore(mem, @bitCast(count_ptr), @bitCast(buf_size_ptr));
}

pub fn aotEnvironGet(vmctx: *VmCtx, environ_ptrs: i32, environ_buf: i32) callconv(.c) i32 {
    traceAdapter("environ_get", vmctx, .{ environ_ptrs, environ_buf });
    if (getCtx(vmctx)) |ctx| {
        const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
        return host_functions.writeStringTable(mem, ctx.env_vars, @bitCast(environ_ptrs), @bitCast(environ_buf));
    }
    return wasi_core.environGetCore();
}

pub fn aotArgsSizesGet(vmctx: *VmCtx, count_ptr: i32, buf_size_ptr: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    if (getCtx(vmctx)) |ctx| {
        const sizes = ctx.args_sizes_get();
        if (!wasi_core.memWriteU32(mem, @bitCast(count_ptr), sizes.count) or
            !wasi_core.memWriteU32(mem, @bitCast(buf_size_ptr), sizes.buf_size))
        {
            return wasi_core.WASI_EINVAL;
        }
        return wasi_core.WASI_ESUCCESS;
    }
    return wasi_core.argsSizesGetCore(mem, @bitCast(count_ptr), @bitCast(buf_size_ptr));
}

pub fn aotArgsGet(vmctx: *VmCtx, argv_ptrs: i32, argv_buf: i32) callconv(.c) i32 {
    if (getCtx(vmctx)) |ctx| {
        const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
        return host_functions.writeStringTable(mem, ctx.args, @bitCast(argv_ptrs), @bitCast(argv_buf));
    }
    return wasi_core.argsGetCore();
}

pub fn aotRandomGet(vmctx: *VmCtx, buf_ptr: i32, buf_len: i32) callconv(.c) i32 {
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    const off: u32 = @bitCast(buf_ptr);
    const len: u32 = @bitCast(buf_len);
    if (@as(u64, off) + len > mem.len) return wasi_core.WASI_EINVAL;

    if (getCtx(vmctx)) |ctx| {
        ctx.random_get(mem[off..][0..len]);
    } else {
        @memset(mem[off..][0..len], 0);
    }
    return wasi_core.WASI_ESUCCESS;
}

pub fn aotSchedYield(vmctx: *VmCtx) callconv(.c) i32 {
    traceAdapter("sched_yield", vmctx, .{});
    // Guest spin/yield loops are the one AOT construct that polls the host
    // often enough to be interrupted without codegen support.
    if (aot_runtime.threadGroupTerminating(vmctx))
        aot_runtime.terminateAotThread(vmctx);
    return wasi_core.schedYieldCore();
}

pub fn aotProcRaise(vmctx: *VmCtx, sig: i32) callconv(.c) i32 {
    _ = vmctx;
    return wasi_core.procRaiseCore(sig);
}

pub fn aotProcExit(vmctx: *VmCtx, code: i32) callconv(.c) void {
    traceAdapter("proc_exit", vmctx, .{code});
    var status: u32 = @bitCast(code);
    if (getCtx(vmctx)) |ctx| {
        ctx.proc_exit(status);
        // Losing a `proc_exit`/trap race must not change what the embedder
        // sees: unwind with whatever outcome actually won.
        if (ctx.terminalOutcome()) |winner| {
            status = switch (winner.kind) {
                .exit => winner.code,
                .trap => 1,
            };
        }
    }
    aot_runtime.signalThreadGroupTrap(vmctx);
    aot_runtime.aotTrapHost(vmctx, @intCast(status & 0xff));
}

pub fn aotThreadSpawn(vmctx: *VmCtx, start_arg: i32) callconv(.c) i32 {
    if (vmctx.instance_ptr == 0) return -1;
    const inst: *aot_runtime.AotInstance = @ptrFromInt(vmctx.instance_ptr);
    return aot_runtime.spawnWasiThread(inst, start_arg) catch -1;
}

pub fn aotPollOneoff(vmctx: *VmCtx, in_ptr: i32, out_ptr: i32, nsubs: i32, ret_ptr: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    const result = host_functions.ctxPollOneoffCore(ctx, mem, in_ptr, out_ptr, nsubs, ret_ptr);
    if (result == host_functions.terminated_result)
        aot_runtime.terminateAotThread(vmctx);
    return result;
}

pub fn aotSockShutdown(vmctx: *VmCtx, fd: i32, sdflags: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    return host_functions.ctxSockShutdownCore(ctx, fd, sdflags);
}

pub fn aotSockAccept(vmctx: *VmCtx, fd: i32, fdflags: i32, ro_fd_ptr: i32) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxSockAcceptCore(ctx, mem, fd, fdflags, @bitCast(ro_fd_ptr));
}

pub fn aotSockRecv(
    vmctx: *VmCtx,
    fd: i32,
    ri_data_ptr: i32,
    ri_data_len: i32,
    ri_flags: i32,
    ro_datalen_ptr: i32,
    ro_flags_ptr: i32,
) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxSockRecvCore(
        ctx,
        mem,
        fd,
        @bitCast(ri_data_ptr),
        @bitCast(ri_data_len),
        ri_flags,
        @bitCast(ro_datalen_ptr),
        @bitCast(ro_flags_ptr),
    );
}

pub fn aotSockSend(
    vmctx: *VmCtx,
    fd: i32,
    si_data_ptr: i32,
    si_data_len: i32,
    si_flags: i32,
    so_datalen_ptr: i32,
) callconv(.c) i32 {
    const ctx = getCtx(vmctx) orelse return wasi_core.WASI_ENOSYS;
    const mem = getMemoryFromCtx(vmctx) orelse return wasi_core.WASI_EINVAL;
    return host_functions.ctxSockSendCore(
        ctx,
        mem,
        fd,
        @bitCast(si_data_ptr),
        @bitCast(si_data_len),
        si_flags,
        @bitCast(so_datalen_ptr),
    );
}

// ── Resolver ──────────────────────────────────────────────────────────

/// Resolve a WASI function name to an AOT adapter function pointer.
/// Returns null for unrecognised names.
pub fn resolveAotHostFunction(name: []const u8) ?*const anyopaque {
    const map = .{
        // Core I/O
        .{ "fd_write", @as(*const anyopaque, @ptrCast(&aotFdWrite)) },
        .{ "fd_read", @as(*const anyopaque, @ptrCast(&aotFdRead)) },
        .{ "fd_pread", @as(*const anyopaque, @ptrCast(&aotFdPread)) },
        .{ "fd_pwrite", @as(*const anyopaque, @ptrCast(&aotFdPwrite)) },
        .{ "fd_readdir", @as(*const anyopaque, @ptrCast(&aotFdReaddir)) },
        .{ "fd_seek", @as(*const anyopaque, @ptrCast(&aotFdSeek)) },
        .{ "fd_close", @as(*const anyopaque, @ptrCast(&aotFdClose)) },
        .{ "fd_renumber", @as(*const anyopaque, @ptrCast(&aotFdRenumber)) },
        .{ "fd_tell", @as(*const anyopaque, @ptrCast(&aotFdTell)) },
        // fd metadata
        .{ "fd_fdstat_get", @as(*const anyopaque, @ptrCast(&aotFdFdstatGet)) },
        .{ "fd_fdstat_set_flags", @as(*const anyopaque, @ptrCast(&aotFdFdstatSetFlags)) },
        .{ "fd_fdstat_set_rights", @as(*const anyopaque, @ptrCast(&aotFdFdstatSetRights)) },
        .{ "fd_filestat_get", @as(*const anyopaque, @ptrCast(&aotFdFilestatGet)) },
        .{ "fd_filestat_set_size", @as(*const anyopaque, @ptrCast(&aotFdFilestatSetSize)) },
        .{ "fd_filestat_set_times", @as(*const anyopaque, @ptrCast(&aotFdFilestatSetTimes)) },
        .{ "fd_advise", @as(*const anyopaque, @ptrCast(&aotFdAdvise)) },
        .{ "fd_allocate", @as(*const anyopaque, @ptrCast(&aotFdAllocate)) },
        .{ "fd_datasync", @as(*const anyopaque, @ptrCast(&aotFdDatasync)) },
        .{ "fd_sync", @as(*const anyopaque, @ptrCast(&aotFdSync)) },
        // preopens
        .{ "fd_prestat_get", @as(*const anyopaque, @ptrCast(&aotFdPrestatGet)) },
        .{ "fd_prestat_dir_name", @as(*const anyopaque, @ptrCast(&aotFdPrestatDirName)) },
        // path_*
        .{ "path_open", @as(*const anyopaque, @ptrCast(&aotPathOpen)) },
        .{ "path_filestat_get", @as(*const anyopaque, @ptrCast(&aotPathFilestatGet)) },
        .{ "path_filestat_set_times", @as(*const anyopaque, @ptrCast(&aotPathFilestatSetTimes)) },
        .{ "path_create_directory", @as(*const anyopaque, @ptrCast(&aotPathCreateDirectory)) },
        .{ "path_remove_directory", @as(*const anyopaque, @ptrCast(&aotPathRemoveDirectory)) },
        .{ "path_unlink_file", @as(*const anyopaque, @ptrCast(&aotPathUnlinkFile)) },
        .{ "path_link", @as(*const anyopaque, @ptrCast(&aotPathLink)) },
        .{ "path_rename", @as(*const anyopaque, @ptrCast(&aotPathRename)) },
        .{ "path_symlink", @as(*const anyopaque, @ptrCast(&aotPathSymlink)) },
        .{ "path_readlink", @as(*const anyopaque, @ptrCast(&aotPathReadlink)) },
        // Clocks
        .{ "clock_time_get", @as(*const anyopaque, @ptrCast(&aotClockTimeGet)) },
        .{ "clock_res_get", @as(*const anyopaque, @ptrCast(&aotClockResGet)) },
        // Args / env / random
        .{ "environ_sizes_get", @as(*const anyopaque, @ptrCast(&aotEnvironSizesGet)) },
        .{ "environ_get", @as(*const anyopaque, @ptrCast(&aotEnvironGet)) },
        .{ "args_sizes_get", @as(*const anyopaque, @ptrCast(&aotArgsSizesGet)) },
        .{ "args_get", @as(*const anyopaque, @ptrCast(&aotArgsGet)) },
        .{ "random_get", @as(*const anyopaque, @ptrCast(&aotRandomGet)) },
        // Scheduling / signals
        .{ "sched_yield", @as(*const anyopaque, @ptrCast(&aotSchedYield)) },
        .{ "proc_raise", @as(*const anyopaque, @ptrCast(&aotProcRaise)) },
        .{ "proc_exit", @as(*const anyopaque, @ptrCast(&aotProcExit)) },
        .{ "thread-spawn", @as(*const anyopaque, @ptrCast(&aotThreadSpawn)) },
        // poll
        .{ "poll_oneoff", @as(*const anyopaque, @ptrCast(&aotPollOneoff)) },
        // sockets
        .{ "sock_shutdown", @as(*const anyopaque, @ptrCast(&aotSockShutdown)) },
        .{ "sock_accept", @as(*const anyopaque, @ptrCast(&aotSockAccept)) },
        .{ "sock_recv", @as(*const anyopaque, @ptrCast(&aotSockRecv)) },
        .{ "sock_send", @as(*const anyopaque, @ptrCast(&aotSockSend)) },
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

// ── Spectest stubs ────────────────────────────────────────────────────
//
// The WebAssembly reference-test suite imports a small set of `spectest.*`
// functions (see tests/spec-json/*.wasm). The test runner treats these as
// no-ops: the spec only asserts that calls don't trap. A single `ret` is
// sufficient for every print_* signature because x86-64 callers own their
// stack args (no callee cleanup) and unused arg registers can be left alone.

pub fn aotSpectestNoop(_: *VmCtx) callconv(.c) void {}

/// True if `module_name` is the `spectest` module used by the WebAssembly
/// reference test suite.
pub fn isSpectestModule(module_name: []const u8) bool {
    return std.mem.eql(u8, module_name, "spectest");
}

pub fn setTrampolinePool(pool: ?*TrampolinePool) void {
    g_trampoline_pool = pool;
    host_trampolines.setActivePool(pool);
}

pub fn getTrampolinePool() ?*TrampolinePool {
    return g_trampoline_pool;
}

/// Resolve a `spectest.*` function name to a no-op AOT adapter. Returns null
/// for names outside the spec's standard surface (print/print_i32/... etc).
pub fn resolveAotSpectestFunction(name: []const u8) ?*const anyopaque {
    const known = [_][]const u8{
        "print",
        "print_i32",
        "print_i64",
        "print_f32",
        "print_f64",
        "print_i32_f32",
        "print_f64_f64",
    };
    for (known) |k| {
        if (std.mem.eql(u8, name, k))
            return @as(*const anyopaque, @ptrCast(&aotSpectestNoop));
    }
    return null;
}

// ── Tests ─────────────────────────────────────────────────────────────

test "resolveAotHostFunction: all known functions resolve" {
    const names = [_][]const u8{
        "fd_write",              "fd_read",                "fd_pread",
        "fd_pwrite",             "fd_readdir",             "fd_seek",
        "fd_close",              "fd_renumber",            "fd_tell",
        "fd_fdstat_get",         "fd_fdstat_set_flags",    "fd_fdstat_set_rights",
        "fd_filestat_get",       "fd_filestat_set_size",   "fd_filestat_set_times",
        "fd_advise",             "fd_allocate",            "fd_datasync",
        "fd_sync",               "fd_prestat_get",         "fd_prestat_dir_name",
        "path_open",             "path_filestat_get",      "path_filestat_set_times",
        "path_create_directory", "path_remove_directory",  "path_unlink_file",
        "path_link",             "path_rename",            "path_symlink",
        "path_readlink",         "clock_time_get",         "clock_res_get",
        "environ_sizes_get",     "environ_get",            "args_sizes_get",
        "args_get",              "random_get",             "sched_yield",
        "proc_raise",            "proc_exit",              "thread-spawn",
        "poll_oneoff",           "sock_shutdown",          "sock_accept",
        "sock_recv",             "sock_send",
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

test "trampoline pool getter/setter roundtrip" {
    var pool: TrampolinePool = undefined;

    setTrampolinePool(&pool);
    try std.testing.expectEqual(&pool, getTrampolinePool().?);
    setTrampolinePool(null);
    try std.testing.expect(getTrampolinePool() == null);
}

test {
    _ = @import("../../tests/aot_host_trampolines_test.zig");
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

test "aotRandomGet: fills buffer with no ctx (zeroes)" {
    var mem = [_]u8{0xAA} ** 32;
    var vmctx = VmCtx{
        .memory_base = @intFromPtr(&mem),
        .memory_size = mem.len,
    };
    const result = aotRandomGet(&vmctx, 0, 16);
    try std.testing.expectEqual(wasi_core.WASI_ESUCCESS, result);
    for (mem[0..16]) |b| try std.testing.expectEqual(@as(u8, 0), b);
}

test "aotRandomGet: rejects OOB" {
    var mem = [_]u8{0} ** 8;
    var vmctx = VmCtx{
        .memory_base = @intFromPtr(&mem),
        .memory_size = mem.len,
    };
    try std.testing.expectEqual(wasi_core.WASI_EINVAL, aotRandomGet(&vmctx, 0, 16));
}
