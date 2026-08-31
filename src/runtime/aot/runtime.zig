//! AOT module instantiation and execution.
//!
//! Creates a runnable AotInstance from an AotModule by allocating memories,
//! tables, and globals according to the module specifications, and optionally
//! mapping the compiled native code as executable.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../common/types.zig");
const aot_loader = @import("loader.zig");
const host_bridge = @import("host_bridge.zig");
const host_trampolines = @import("host_trampolines.zig");
const platform = @import("../../platform/platform.zig");
const sig_registry = @import("../common/sig_registry.zig");
const trap_jmp = @import("trap_jmp.zig");
const config = @import("../../config.zig");
const execution_context = @import("../common/execution_context.zig");
const thread_manager = @import("../../wasi/thread_manager.zig");

// ─── Windows crash handler (debug only) ─────────────────────────────────────
const windows = std.os.windows;

/// #857: lightweight process-wide registry tracking currently-mapped
/// JIT/AOT executable code so a long-lived process that repeatedly
/// compiles+drops modules (e.g. an embedding host, a dev-server
/// hot-reload loop, or a test harness JIT-compiling many small modules
/// back to back) can introspect total resident code size and,
/// optionally, cap it rather than silently growing unbounded or OOMing.
///
/// This is bookkeeping only — it doesn't retain or reuse compiled code
/// across a `destroy` (that would be an actual *cache* in the reuse
/// sense; see the design spike tracked by issue #862 for lazy/tiered
/// compilation, which is a different feature). `mapCodeExecutable`,
/// lazy first-call compilation, and their matching teardown paths
/// already individually `mmap`/`munmap` correctly — verified by
/// inspection and by the stress test in `runtime_test.zig` — this
/// struct just lets that be observed and bounded in aggregate across
/// every live executable mapping in the process.
pub const JitCodeCache = struct {
    /// Total resident bytes across every currently-mapped executable
    /// region tracked through this registry (whole-module text blobs
    /// and lazily-compiled per-function mappings). Default `0` =
    /// unlimited (existing behavior unchanged). Opt in via
    /// `WAMR_JIT_CODE_BUDGET_BYTES` (see `main.zig`) or set directly
    /// before compiling.
    pub var budget_bytes: usize = 0;

    var resident_bytes: usize = 0;
    var mapping_count: usize = 0;

    /// Currently resident JIT/AOT executable code, summed across every
    /// live tracked executable mapping in this process.
    pub fn residentBytes() usize {
        return resident_bytes;
    }

    /// Number of currently-live tracked executable mappings in this
    /// process.
    pub fn mappingCount() usize {
        return mapping_count;
    }

    fn register(size: usize) void {
        resident_bytes += size;
        mapping_count += 1;
    }

    fn unregister(size: usize) void {
        resident_bytes -= size;
        mapping_count -= 1;
    }

    /// Returns `error.CodeBudgetExceeded` if mapping `additional_bytes`
    /// more code would push `residentBytes()` past `budget_bytes`, when
    /// a nonzero budget is configured. Called before the `mmap`, so a
    /// caller gets a clear typed error instead of the process
    /// eventually running out of memory from unbounded JIT growth.
    fn checkBudget(additional_bytes: usize) error{CodeBudgetExceeded}!void {
        if (budget_bytes == 0) return;
        if (resident_bytes + additional_bytes > budget_bytes) return error.CodeBudgetExceeded;
    }
};

// ── Trap-as-error plumbing (Windows x86_64 only) ─────────────────────
//
// Acts like setjmp/longjmp. `callFuncScalar` calls `RtlCaptureContext`
// immediately before invoking generated code; if a trap occurs (OOB
// via `aotTrapOOB`, or any access violation / illegal instruction /
// divide-by-zero / stack overflow caught by `vehHandler`), we set
// the active call state's `trap_occurred` flag and use
// `RtlRestoreContext` to resume execution at the capture site. The check
// then returns `error.WasmTrap` out of `callFuncScalar`.
//
// The VEH body dereferences x86_64-specific `CONTEXT` fields
// (`Rip`, `Rax`, ...) so the whole block is gated on x86_64 too;
// on aarch64-windows the runtime still works, traps just aren't
// catchable as errors (they'll abort the process, same as Linux).
//
// Per-call trap state lives behind the active `ThreadExecutionContext`.
// Windows TLS stores only that context pointer; the aligned CONTEXT itself
// stays in ordinary stack storage.
const windows_trap_supported = builtin.os.tag == .windows and builtin.cpu.arch == .x86_64;
/// #798 Lever 1: catch AOT traps as `error.WasmTrap` on POSIX x86_64 and
/// AArch64 (Linux / macOS), the analogue of the Windows x86_64
/// `RtlCaptureContext` path.
/// Uses the hand-rolled `trap_jmp` setjmp/longjmp (no libc on Linux). When
/// false (other targets), AOT traps abort the process as before.
const posix_trap_supported = !windows_trap_supported and trap_jmp.supported;
/// Forensic dump targets parsed from `WAMR_TRAP_OOB_DUMP`; populated by
/// `main.zig` at startup. See `aotTrapOobDumpMem` (#719).
pub var g_trap_oob_dump_env: ?[]const u8 = null;

// ─── #719 watch-addr SIGUSR1-driven address watcher ─────────────────────
//
// When the env var `WAMR_WATCH_ADDR=<hex_off>:<lo>:<hi>[:<min_mem_mb>]`
// is set, a background thread polls `*(u32*)(host_base + wasm_off)`
// and signals the main thread via SIGUSR1 the first time the watched
// value enters the half-open range `[lo, hi)`. The handler decodes
// the interrupted RIP into `local_func[N] + rel_off` so we can pin
// down the wasm function performing the rogue write.
//
// Intended for chasing the deep #719 trap where `tcgc-core` dlmalloc's
// `treebins[0]` (wasm 0x247bb8) gets overwritten with a static-data
// pointer (< 0x800000). Inert when the env var is unset.
const WatchAddrConfig = struct {
    wasm_off: u32,
    lo: u32,
    hi: u32,
    min_mem_bytes: usize,
};
var g_watch_cfg: ?WatchAddrConfig = null;
var g_watch_host_base: std.atomic.Value(usize) = .init(0);
var g_watch_armed: std.atomic.Value(bool) = .init(false);
var g_watch_caught: std.atomic.Value(u32) = .init(0);
var g_watch_main_tid: i32 = 0;
var g_watch_pid: i32 = 0;
var g_watch_prev_action: std.os.linux.Sigaction = undefined;
const WATCH_MAX_CATCHES: u32 = 16;
const linux_watch = std.os.linux;
const watch_supported = builtin.os.tag == .linux and builtin.cpu.arch == .x86_64;

const WatchUcontextX86_64 = extern struct {
    _flags: usize,
    _link: ?*anyopaque,
    _stack: linux_watch.stack_t,
    mcontext: extern struct {
        r8: u64,
        r9: u64,
        r10: u64,
        r11: u64,
        r12: u64,
        r13: u64,
        r14: u64,
        r15: u64,
        rdi: u64,
        rsi: u64,
        rbp: u64,
        rbx: u64,
        rdx: u64,
        rax: u64,
        rcx: u64,
        rsp: u64,
        rip: u64,
    },
};

fn watchSigusr1(_: linux_watch.SIG, _: *const linux_watch.siginfo_t, ctx: ?*anyopaque) callconv(.c) void {
    if (ctx == null) return;
    if (!watch_supported) return;
    const u: *const WatchUcontextX86_64 = @ptrCast(@alignCast(ctx.?));
    const rip = u.mcontext.rip;
    const cfg = g_watch_cfg orelse return;
    const base = g_watch_host_base.load(.acquire);
    var val: u32 = 0;
    if (base != 0) {
        const p: *const u32 = @ptrFromInt(base + cfg.wasm_off);
        val = p.*;
    }
    const hits = g_watch_caught.load(.acquire);
    const loc = decodeTrapReturnAddress(rip);
    if (loc.name) |name| {
        std.debug.print(
            "[watch-caught #{d}] val=0x{x} rip=0x{x} code+0x{x} local_func[{d}] \"{s}\"+0x{x}\n",
            .{ hits, val, rip, loc.code_off, loc.func_idx, name, loc.rel_off },
        );
    } else {
        std.debug.print(
            "[watch-caught #{d}] val=0x{x} rip=0x{x} code+0x{x} local_func[{d}]+0x{x}\n",
            .{ hits, val, rip, loc.code_off, loc.func_idx, loc.rel_off },
        );
    }
}

fn watchSleepNs(ns: u64) void {
    const sec: i64 = @intCast(ns / std.time.ns_per_s);
    const nsec: i64 = @intCast(ns % std.time.ns_per_s);
    const ts = std.posix.timespec{ .sec = sec, .nsec = nsec };
    _ = std.posix.system.nanosleep(&ts, null);
}

fn watchPoller() void {
    var prev: u32 = 0xFFFFFFFF;
    var first_read = true;
    while (g_watch_armed.load(.monotonic)) {
        const base = g_watch_host_base.load(.acquire);
        if (base == 0) {
            std.Thread.yield() catch {};
            watchSleepNs(1_000_000); // 1ms, wait for memory attach
            continue;
        }
        const cfg = g_watch_cfg orelse return;
        const p: *const u32 = @ptrFromInt(base + cfg.wasm_off);
        const v = p.*;
        if (first_read or v != prev) {
            first_read = false;
            prev = v;
            if (v >= cfg.lo and v < cfg.hi) {
                const cnt = g_watch_caught.fetchAdd(1, .acq_rel);
                if (cnt < WATCH_MAX_CATCHES and watch_supported) {
                    // Signal main thread to capture its RIP.
                    _ = linux_watch.tgkill(g_watch_pid, g_watch_main_tid, linux_watch.SIG.USR1);
                    // Sleep so handler runs before we keep polling.
                    watchSleepNs(2_000_000); // 2ms
                } else if (cnt == WATCH_MAX_CATCHES) {
                    std.debug.print("[watch-addr] hit catch-cap; muting further signals\n", .{});
                }
            }
        }
        // tight poll — yield to let main thread run
        std.atomic.spinLoopHint();
    }
}

/// Parse `WAMR_WATCH_ADDR=<hex_off>:<lo>:<hi>[:<min_mem_mb>]` and start
/// the watcher thread. Address fields accept `0x`-prefixed hex; the
/// optional `<min_mem_mb>` arms only for memories at least that large
/// (decimal). Inert if the platform isn't supported.
pub fn initWatchAddrFromEnv(env_val: []const u8) !void {
    if (!watch_supported) {
        std.debug.print("[watch-addr] unsupported on this platform; ignoring\n", .{});
        return;
    }
    var it = std.mem.splitScalar(u8, env_val, ':');
    const off_s = it.next() orelse return error.BadFormat;
    const lo_s = it.next() orelse return error.BadFormat;
    const hi_s = it.next() orelse return error.BadFormat;
    const mb_s = it.next();
    const off = try parseWatchU32Hex(off_s);
    const lo = try parseWatchU32Hex(lo_s);
    const hi = try parseWatchU32Hex(hi_s);
    const mb: usize = if (mb_s) |s| try parseWatchUsizeDec(s) else 0;

    g_watch_cfg = .{
        .wasm_off = off,
        .lo = lo,
        .hi = hi,
        .min_mem_bytes = mb * 1024 * 1024,
    };
    g_watch_pid = @intCast(linux_watch.getpid());
    g_watch_main_tid = @intCast(linux_watch.gettid());

    const act: linux_watch.Sigaction = .{
        .handler = .{ .sigaction = watchSigusr1 },
        .mask = linux_watch.sigemptyset(),
        .flags = linux_watch.SA.SIGINFO | linux_watch.SA.RESTART,
    };
    // Use the raw linux sigaction syscall (not std.posix.sigaction): the libc
    // wrapper expects c.common_linux_Sigaction, which has a different layout
    // than std.os.linux.Sigaction and rejects this struct when libc is linked
    // (musl). The raw syscall takes std.os.linux.Sigaction directly and works
    // on every linux libc/no-libc target.
    const rc = linux_watch.sigaction(.USR1, &act, &g_watch_prev_action);
    if (linux_watch.errno(rc) != .SUCCESS) return error.SigactionFailed;

    g_watch_armed.store(true, .release);
    _ = try std.Thread.spawn(.{}, watchPoller, .{});

    std.debug.print(
        "[watch-addr] armed wasm_off=0x{x} bad_range=[0x{x}..0x{x}) min_mem={d}MB main_tid={d}\n",
        .{ off, lo, hi, mb, g_watch_main_tid },
    );
}

fn parseWatchU32Hex(s: []const u8) !u32 {
    const trimmed = std.mem.trim(u8, s, " \t\r\n");
    const hex = if (std.mem.startsWith(u8, trimmed, "0x") or std.mem.startsWith(u8, trimmed, "0X"))
        trimmed[2..]
    else
        trimmed;
    return std.fmt.parseInt(u32, hex, 16);
}

fn parseWatchUsizeDec(s: []const u8) !usize {
    const trimmed = std.mem.trim(u8, s, " \t\r\n");
    return std.fmt.parseInt(usize, trimmed, 10);
}

/// Called from `refreshVmCtxMemory` so the watcher always reads the
/// freshest host base, including across `memory.grow` remappings.
fn watchAddrNoteMemory(host_base: usize, wasm_size: usize) void {
    const cfg = g_watch_cfg orelse return;
    if (wasm_size < cfg.min_mem_bytes) return;
    g_watch_host_base.store(host_base, .release);
}
extern "kernel32" fn RtlCaptureContext(ContextRecord: *windows.CONTEXT) callconv(.winapi) void;
extern "kernel32" fn RtlRestoreContext(ContextRecord: *windows.CONTEXT, ExceptionRecord: ?*anyopaque) callconv(.winapi) noreturn;

// Win32 memory-protect flags and APIs used by resetStackGuardPage.
// Declared locally so we don't depend on std.os.windows exposing them.
const PAGE_READWRITE: u32 = 0x04;
const PAGE_GUARD: u32 = 0x100;
const MEM_COMMIT: u32 = 0x1000;
const MEM_DECOMMIT: u32 = 0x4000;

const MEMORY_BASIC_INFORMATION = extern struct {
    BaseAddress: ?*anyopaque,
    AllocationBase: ?*anyopaque,
    AllocationProtect: u32,
    PartitionId: u16,
    _pad: u16,
    RegionSize: usize,
    State: u32,
    Protect: u32,
    Type: u32,
};

extern "kernel32" fn VirtualProtect(
    lpAddress: ?*anyopaque,
    dwSize: usize,
    flNewProtect: u32,
    lpflOldProtect: *u32,
) callconv(.winapi) windows.BOOL;

extern "kernel32" fn VirtualQuery(
    lpAddress: ?*const anyopaque,
    lpBuffer: *MEMORY_BASIC_INFORMATION,
    dwLength: usize,
) callconv(.winapi) usize;

extern "kernel32" fn VirtualFree(
    lpAddress: ?*anyopaque,
    dwSize: usize,
    dwFreeType: u32,
) callconv(.winapi) windows.BOOL;

extern "kernel32" fn SetThreadStackGuarantee(
    StackSizeInBytes: *u32,
) callconv(.winapi) windows.BOOL;

/// Re-arm the current thread's stack guard page after a caught
/// `STATUS_STACK_OVERFLOW`. The OS removed `PAGE_GUARD` from the page
/// that was hit and committed it as ordinary R/W memory, so a subsequent
/// overflow in the same thread walks past the end of the stack and the
/// process aborts with `STATUS_ACCESS_VIOLATION` — bypassing our VEH.
///
/// Mirrors the behaviour of MSVC CRT's `_resetstkoflw`: find the lowest
/// committed page in the current thread's stack allocation and mark it
/// `PAGE_READWRITE | PAGE_GUARD` so the OS raises another
/// `STATUS_STACK_OVERFLOW` on the next overrun.
///
/// Must be called on a clean stack — i.e. after `RtlRestoreContext` has
/// returned us to the capture site in `callFuncScalar`, well above the
/// former-guard page we're about to touch.
fn resetStackGuardPage() void {
    if (comptime builtin.os.tag != .windows) return;

    // Probe any on-stack address to locate the stack's allocation base.
    var mbi: MEMORY_BASIC_INFORMATION = undefined;
    var probe: usize = 0;
    const probe_ptr: *const anyopaque = @ptrCast(&probe);
    if (VirtualQuery(probe_ptr, &mbi, @sizeOf(MEMORY_BASIC_INFORMATION)) == 0) return;
    const alloc_base = mbi.AllocationBase orelse return;

    // Walk up from the allocation base skipping uncommitted pages to
    // find the first committed page — that's where the guard belongs.
    var cursor: usize = @intFromPtr(alloc_base);
    while (true) {
        if (VirtualQuery(@ptrFromInt(cursor), &mbi, @sizeOf(MEMORY_BASIC_INFORMATION)) == 0) return;
        if (mbi.AllocationBase != alloc_base) return;
        if (mbi.State == MEM_COMMIT) break;
        cursor +%= mbi.RegionSize;
        if (mbi.RegionSize == 0) return;
    }

    const page_size: usize = std.heap.page_size_min;

    // After a stack overflow the OS committed the former guard page and
    // additional pages below it for SetThreadStackGuarantee.  These extra
    // committed pages reduce the reservoir of reserved pages needed by
    // the next overflow's exception dispatch.  Decommit a few pages back
    // to MEM_RESERVE so the OS can reuse them for the next guarantee.
    const reserve_pages = 8; // 32 KB — comfortably covers a 16 KB guarantee
    const pages_in_region = mbi.RegionSize / page_size;
    if (pages_in_region >= 2) {
        const decommit_pages = @min(reserve_pages, pages_in_region - 1);
        const decommit_bytes = decommit_pages * page_size;
        _ = VirtualFree(@ptrFromInt(cursor), decommit_bytes, MEM_DECOMMIT);
        var old_protect: u32 = 0;
        _ = VirtualProtect(
            @ptrFromInt(cursor + decommit_bytes),
            page_size,
            PAGE_READWRITE | PAGE_GUARD,
            &old_protect,
        );
    } else {
        // Single committed page: just re-arm as guard (pre-existing
        // behaviour; may not survive a second overflow).
        var old_protect: u32 = 0;
        _ = VirtualProtect(@ptrFromInt(cursor), page_size, PAGE_READWRITE | PAGE_GUARD, &old_protect);
    }
}

fn trapLongjmp() noreturn {
    const state = activeAotCallState() orelse std.process.exit(2);
    state.trap_occurred.store(true, .seq_cst);
    if (comptime windows_trap_supported) {
        RtlRestoreContext(&state.saved_ctx, null);
    }
    if (comptime posix_trap_supported) {
        // Resume at the `trap_jmp.capture` site in callFuncScalar, which
        // then returns `error.WasmTrap`.
        trap_jmp.restore(&state.posix_trap_buf, 1);
    }
    // No trap-catch support on this target: fall back to exit.
    std.process.exit(2);
}

fn vehHandler(info: *windows.EXCEPTION_POINTERS) callconv(.winapi) c_long {
    const rec = info.ExceptionRecord;
    const ctx = info.ContextRecord;
    const code = rec.ExceptionCode;
    // Wasm-like traps we want to turn into error.WasmTrap when armed:
    //   0xC0000005 STATUS_ACCESS_VIOLATION  (OOB / null deref)
    //   0xC0000094 STATUS_INTEGER_DIVIDE_BY_ZERO
    //   0xC0000095 STATUS_INTEGER_OVERFLOW
    //   0xC000001D STATUS_ILLEGAL_INSTRUCTION (unreachable → ud2)
    //   0xC00000FD STATUS_STACK_OVERFLOW
    const is_wasm_fault = code == 0xC0000005 or
        code == 0xC0000094 or
        code == 0xC0000095 or
        code == 0xC000001D or
        code == 0xC00000FD;
    const state = activeAotCallState();
    const frame = if (state) |active| active.trap_decode else TrapDecodeFrame{};
    const rip: usize = @intCast(ctx.Rip);
    const in_code = frame.code_base != 0 and frame.code_size != 0 and
        rip >= frame.code_base and rip < frame.code_base + frame.code_size;
    // If armed, redirect any wasm-like fault to trapLongjmp. We used to
    // only redirect when RIP was inside the generated code, but a null
    // table entry in call_indirect causes RIP=0 at fault time (the
    // `call r11` has already transferred control), so the fault site is
    // outside the code region. The armed check is sufficient to
    // distinguish wasm traps from unrelated process-wide faults.
    if (is_wasm_fault and state != null and state.?.trap_catching.load(.seq_cst)) {
        _ = in_code;
        state.?.trap_occurred.store(true, .seq_cst);
        state.?.last_trap_code.store(code, .seq_cst);
        if (code == 0xC00000FD) {
            // Stack overflow: restore the full saved context so we
            // resume at the RtlCaptureContext site on a healthy stack.
            // Using trapLongjmp here is fragile because it would run
            // on the nearly-exhausted overflowed stack.
            ctx.* = state.?.saved_ctx;
        } else {
            ctx.Rip = @intFromPtr(&trapLongjmp);
        }
        return -1; // EXCEPTION_CONTINUE_EXECUTION
    }
    if (rec.ExceptionCode == 0xC0000005) { // STATUS_ACCESS_VIOLATION
        const fault: usize = @intCast(rec.ExceptionInformation[1]);
        std.debug.print(
            "\n=== VEH CRASH === RIP=0x{x} (code+0x{x}) fault=0x{x}",
            .{ rip, rip -% frame.code_base, fault },
        );
        if (frame.mem_base != 0 and fault >= frame.mem_base and fault < frame.mem_base +% frame.mem_size) {
            std.debug.print(" (wasm mem[0x{x}])", .{fault - frame.mem_base});
        } else if (frame.mem_base != 0) {
            const delta: isize = @as(isize, @bitCast(fault)) - @as(isize, @bitCast(frame.mem_base));
            std.debug.print(" (mem_base+0x{x} delta={d})", .{ fault -% frame.mem_base, delta });
        }
        std.debug.print("\n", .{});
        std.debug.print("RAX=0x{x} RCX=0x{x} RDX=0x{x} RBX=0x{x}\n", .{ ctx.Rax, ctx.Rcx, ctx.Rdx, ctx.Rbx });
        std.debug.print("RSI=0x{x} RDI=0x{x} RBP=0x{x} RSP=0x{x}\n", .{ ctx.Rsi, ctx.Rdi, ctx.Rbp, ctx.Rsp });
        std.debug.print("R8=0x{x} R9=0x{x} R10=0x{x} R11=0x{x}\n", .{ ctx.R8, ctx.R9, ctx.R10, ctx.R11 });
        std.debug.print("R12=0x{x} R13=0x{x} R14=0x{x} R15=0x{x}\n", .{ ctx.R12, ctx.R13, ctx.R14, ctx.R15 });
        if (frame.code_base != 0 and frame.code_size != 0) {
            const rip_off: usize = rip -% frame.code_base;
            if (rip_off < frame.code_size) {
                const start: usize = if (rip_off > 32) rip_off - 32 else 0;
                const end: usize = @min(rip_off + 16, frame.code_size);
                const p: [*]const u8 = @ptrFromInt(frame.code_base + start);
                std.debug.print("code@[0x{x}..0x{x}]:", .{ start, end });
                var i: usize = 0;
                while (i < end - start) : (i += 1) {
                    const marker: u8 = if (start + i == rip_off) '>' else ' ';
                    std.debug.print("{c}{x:0>2}", .{ marker, p[i] });
                }
                std.debug.print("\n", .{});
            }
        }
    }
    return 0; // EXCEPTION_CONTINUE_SEARCH
}

extern "kernel32" fn AddVectoredExceptionHandler(
    First: u32,
    Handler: *const fn (*windows.EXCEPTION_POINTERS) callconv(.winapi) c_long,
) callconv(.winapi) ?*anyopaque;

/// Compact context passed to AOT-compiled functions as a hidden first parameter.
/// Laid out as a flat struct so compiled code can load fields at known offsets.
pub const VmCtx = extern struct {
    /// Base pointer to linear memory (memory 0).
    memory_base: usize = 0,
    /// Size of linear memory in bytes (current, may grow).
    memory_size: usize = 0,
    /// Pointer to flat globals storage. Scalar/reference globals use 8-byte
    /// slots; v128 globals use 16-byte aligned, 16-byte slots.
    globals_ptr: usize = 0,
    /// Pointer to array of host function pointers (one per import).
    host_functions_ptr: usize = 0,
    /// Maximum allocated memory region size in bytes (for grow bounds checking).
    memory_max_size: usize = 0,
    /// Pointer to function pointer table (for call_indirect).
    /// Entry i is the native code address for module function i.
    func_table_ptr: usize = 0,
    /// Number of globals.
    globals_count: u32 = 0,
    /// Number of host functions.
    host_functions_count: u32 = 0,
    /// Current memory size in pages (for memory.size instruction).
    memory_pages: u32 = 0,
    /// Number of entries in the func_table (indexed by call_indirect elem idx).
    /// Used by inline bounds checks emitted for call_indirect to trap on
    /// out-of-range indices rather than dereferencing past the table.
    func_table_len: u32 = 0,
    /// Native function pointer for memory.grow host helper.
    /// Pointer to host function invoked by `memory.grow` in AOT-compiled code.
    /// Signature: fn (vmctx: *VmCtx, delta_pages: i32) callconv(.c) i32
    /// Returns previous page count on success, -1 on failure.
    mem_grow_fn: usize = 0,
    /// Opaque pointer to the owning AotInstance (used by host helpers).
    instance_ptr: usize = 0,
    /// Native function pointer for the out-of-bounds memory trap helper.
    /// Signature: fn (vmctx: *VmCtx) callconv(.c) noreturn
    /// Called from inline bounds checks emitted by AOT load/store codegen
    /// when a wasm memory access would exceed `memory_size`.
    trap_oob_fn: usize = 0,
    /// `fn (*VmCtx) noreturn` — called by AOT code for wasm `unreachable`.
    trap_unreachable_fn: usize = 0,
    /// `fn (*VmCtx) noreturn` — called for integer divide-by-zero.
    trap_idivz_fn: usize = 0,
    /// `fn (*VmCtx) noreturn` — called for signed INT_MIN/-1 overflow.
    trap_iovf_fn: usize = 0,
    /// `fn (*VmCtx) noreturn` — called for invalid float→int conversion
    /// (NaN or out-of-range) in `trunc_f*_*` opcodes.
    trap_ivc_fn: usize = 0,
    /// Pointer to native function pointer array indexed by module funcidx.
    /// Populated in `mapCodeExecutable` and read by AOT code generated for
    /// `ref.func`. Length is `module.import_function_count + module.func_count`.
    funcptrs_ptr: usize = 0,
    /// Native function pointer for the table.grow host helper.
    /// Signature: fn (vmctx: *VmCtx, init_val: i64, delta: i32) callconv(.c) i32
    /// Returns previous table size on success, -1 on failure.
    table_grow_fn: usize = 0,
    /// Pointer to an array of per-table descriptors, one per declared table:
    /// `extern struct { ptr: u64, len: u32, _pad: u32 }` (16 bytes).
    /// Used by table_get/table_set/table_size codegen for multi-table support.
    /// For table 0, `ptr` aliases `func_table_ptr` and `len` aliases
    /// `func_table_len`.
    tables_info_ptr: usize = 0,
    /// Native function pointer for the `table.init` host helper.
    /// Signature:
    ///   fn (vmctx: *VmCtx,
    ///       packed_seg_table: u64,   // seg_idx | (table_idx << 32)
    ///       packed_dst_src: u64,     // dst     | (src << 32)
    ///       len: u32) callconv(.c) void
    /// Traps on OOB (src+len > seg.len, dst+len > table.len) or
    /// already-dropped passive segment.
    table_init_fn: usize = 0,
    /// Native function pointer for the `elem.drop` host helper.
    /// Signature: fn (vmctx: *VmCtx, seg_idx: u32) callconv(.c) void
    /// Marks the passive element segment as dropped (idempotent).
    elem_drop_fn: usize = 0,
    /// Pointer to `[]u32` of length `module.func_types.len`.
    /// `sig_table[type_idx]` is the process-global canonical sig_id
    /// (from `sig_registry.global()`) for that module type. AOT
    /// codegen for `call_indirect (type $t)` loads the expected
    /// sig_id from `sig_table[$t]` and compares it to the slot's
    /// sig_id read from `TableInstance.type_backing`.
    sig_table_ptr: usize = 0,
    /// Pointer to `[]u32` of length
    /// `import_function_count + func_count`.
    /// `func_sig_ids[funcidx]` is the canonical sig_id for that
    /// function's declared type. Used by writer sites that know the
    /// module-level funcidx (e.g. active elem-segment copy,
    /// `table.init` applied from a passive elem segment) to populate
    /// `TableInstance.type_backing` in lockstep with a funcref store.
    func_sig_ids_ptr: usize = 0,
    /// Pointer to `[]PtrSigEntry` sorted ascending by `ptr`. One
    /// entry per resolved funcptr in this instance (imports + locals).
    /// Used by writer sites that receive a raw funcptr (e.g.
    /// `table.set` with a funcref value, cross-table `table.copy`)
    /// and must derive the matching sig_id via binary search.
    ptr_to_sig_ptr: usize = 0,
    /// Number of entries in `ptr_to_sig`.
    ptr_to_sig_len: u32 = 0,
    /// Padding to align table_set_fn to 8 bytes.
    _pad_pts: u32 = 0,
    /// Native function pointer for the `table.set` host helper.
    /// Signature: fn (vmctx: *VmCtx, table_idx: u32, elem_idx: u32,
    ///               value: usize) callconv(.c) void
    /// Updates both native_backing and type_backing (via ptr_to_sig
    /// binary search) so a subsequent call_indirect sees the correct
    /// sig_id.
    table_set_fn: usize = 0,
    /// Native function pointer for `memory.atomic.wait32`.
    /// Signature: fn (vmctx: *VmCtx, addr: u32, expected: u32, timeout_ns: i64) callconv(.c) i32
    /// Returns 0 (ok/woken), 1 (not-equal), 2 (timed-out).
    futex_wait32_fn: usize = 0,
    /// Native function pointer for `memory.atomic.wait64`.
    /// Signature: fn (vmctx: *VmCtx, addr: u32, exp_lo: u32, exp_hi: u32, timeout_ns: i64) callconv(.c) i32
    futex_wait64_fn: usize = 0,
    /// Native function pointer for `memory.atomic.notify`.
    /// Signature: fn (vmctx: *VmCtx, addr: u32, count: u32) callconv(.c) i32
    /// Returns number of waiters woken.
    futex_notify_fn: usize = 0,
    /// Native function pointer for `memory.fill` host helper.
    /// Signature: fn (vmctx: *VmCtx, dst: u32, val: u32, len: u32) callconv(.c) void
    /// Performs bounds check against `memory_size`; traps via `trap_oob_fn`
    /// if `dst + len > memory_size`. Otherwise writes `val & 0xFF` to
    /// `len` bytes starting at `memory_base + dst`.
    mem_fill_fn: usize = 0,
    /// Native function pointer for `memory.copy` host helper.
    /// Signature: fn (vmctx: *VmCtx, dst: u32, src: u32, len: u32) callconv(.c) void
    /// Performs bounds checks against `memory_size` for both ranges;
    /// traps via `trap_oob_fn` if either `dst + len` or `src + len`
    /// exceeds `memory_size`. Handles overlapping regions (memmove
    /// semantics).
    mem_copy_fn: usize = 0,
    /// Pointer to `[]*TagInstance` for this instance — `inst.tags.ptr`.
    /// AOT codegen for `throw` indexes this to resolve `tag_idx` →
    /// `*TagInstance` at runtime. Length matches `tags_count`.
    /// Issue #672.
    tags_ptr: usize = 0,
    /// Number of entries in `tags_ptr`.
    tags_count: u32 = 0,
    /// Padding to keep the following 8-byte field aligned.
    _pad_tags: u32 = 0,
    /// `fn (*VmCtx, *TagInstance) noreturn` — invoked by AOT-compiled
    /// `throw` when no in-function catch handler claims the exception.
    /// Reads `exception_params` / `exception_param_count`, records the
    /// pending exception, and longjmps via `trapLongjmp` (Windows) or
    /// exits the process (other platforms) — mirroring `trap_unreachable_fn`.
    /// Issue #672.
    aot_throw_uncaught_fn: usize = 0,
    /// Scratch buffer where AOT-compiled `throw` writes the wasm operand
    /// stack values that constitute the exception payload, one per slot.
    /// 16 slots = max exception arity supported by the leaf-function
    /// path. Read by `aotThrowUncaught` to relay them to the embedder.
    /// Per-thread isolation is the embedder's responsibility (one VmCtx
    /// per call-frame chain).
    exception_params: [16]u64 = [_]u64{0} ** 16,
    /// Valid-prefix length of `exception_params` for the most recent throw.
    exception_param_count: u32 = 0,
    /// Padding so `wasi_ctx` stays 8-byte aligned.
    _pad_exc: u32 = 0,
    /// Opaque pointer to the `WasiCtx` driving the WASI host imports
    /// for this AOT instance (issue #644 + Approach A). `0` means no
    /// WASI context was attached — AOT WASI adapters in
    /// `src/runtime/aot/host_bridge.zig` then fall back to the
    /// stateless `wasi_core.zig` defaults (zero-args, stdout-only)
    /// instead of the full filesystem / preopen / fd-table surface.
    /// Set during `wamr run <core.wasm>` after `instantiate` so the
    /// AOT path matches the interpreter's WASI semantics.
    wasi_ctx: usize = 0,
    /// Helper used by lazy-JIT entry stubs. Signature:
    /// `fn (vmctx: *VmCtx, local_idx: u32) callconv(.c) usize`.
    /// Returns the native address of the resolved body (compiling on
    /// demand if still pending), or 0 on failure.
    lazy_compile_fn: usize = 0,
    /// Pointer to this execution instance's per-thread context. Appended
    /// after every codegen-addressed field so existing AOT offsets remain
    /// unchanged.
    thread_context: usize = 0,
};

/// Entry in the sorted `ptr_to_sig` array. 16 bytes per entry.
pub const PtrSigEntry = extern struct {
    ptr: u64 = 0,
    sig_id: u32 = 0,
    _pad: u32 = 0,
};

/// Host helper invoked from AOT-compiled memory loads/stores when an
/// out-of-bounds access is detected. Mirrors the interpreter's
/// `error.OutOfBoundsMemoryAccess` trap. Exits the process with code 2
/// rather than allowing the native CPU to SIGSEGV on unmapped memory.
///
/// NOTE: This terminates the host process. A future change could thread
/// the trap back through a setjmp/longjmp path so that `callFunc` can
/// return `error.OutOfBoundsMemoryAccess`, matching interp semantics
/// for embedded usage. For the current CLI this is sufficient.
const TrapDecodeFrame = struct {
    code_base: usize = 0,
    code_size: usize = 0,
    func_offsets: []const u32 = &.{},
    func_names: []const types.NameSection.FunctionName = &.{},
    imported_function_count: u32 = 0,
    mem_base: usize = 0,
    mem_size: usize = 0,
};

/// Per-native-call AOT bookkeeping. A stack-local instance is bound to the
/// active `ThreadExecutionContext`, so concurrent AOT calls never share trap
/// decode, jump-buffer, or cancellation/trap state.
const AotCallState = struct {
    trap_decode: TrapDecodeFrame = .{},
    posix_trap_buf: if (posix_trap_supported) trap_jmp.JmpBuf else void =
        if (posix_trap_supported) undefined else {},
    saved_ctx: (if (windows_trap_supported) windows.CONTEXT else void) align(16) =
        if (windows_trap_supported) undefined else {},
    trap_catching: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    trap_occurred: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    last_trap_code: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
};

// 0 = uninitialized, 1 = one thread is installing, 2 = ready.
var g_veh_install_state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0);

fn ensureVehInstalled() void {
    if (comptime !windows_trap_supported) return;
    while (true) {
        switch (g_veh_install_state.load(.acquire)) {
            0 => {
                if (g_veh_install_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null)
                    continue;
                if (AddVectoredExceptionHandler(1, vehHandler) == null) {
                    g_veh_install_state.store(0, .release);
                } else {
                    g_veh_install_state.store(2, .release);
                }
                return;
            },
            1 => std.atomic.spinLoopHint(),
            2 => return,
            else => unreachable,
        }
    }
}

fn activeAotCallState() ?*AotCallState {
    const thread_ctx = execution_context.current() orelse return null;
    return thread_ctx.backendContext(AotCallState);
}

fn isTrapCatching() bool {
    const state = activeAotCallState() orelse return false;
    return state.trap_catching.load(.seq_cst);
}

/// Install the calling AOT instance's diagnostic identity. Caller is
/// expected to have already bound an `AotCallState`.
fn installTrapDecodeFrameFor(inst: *const AotInstance) void {
    const state = activeAotCallState() orelse return;
    const frame = &state.trap_decode;
    if (inst.code_base) |cb| {
        frame.code_base = @intFromPtr(cb);
        frame.code_size = inst.code_size;
        frame.func_offsets = inst.module.func_offsets;
    }
    frame.func_names = inst.module.function_names;
    frame.imported_function_count = inst.module.import_function_count;
}

/// Resolve `local_func[N]` back to the wasm-side symbol when the AOT
/// artifact preserves a `name` custom section. `local_idx` is the
/// `func_offsets` index reported by the trap helper's PC decode; the
/// wasm function index space prefixes imports, so we shift by the active
/// frame's imported-function count before looking up.
fn lookupLocalFuncName(local_idx: isize) ?[]const u8 {
    if (local_idx < 0) return null;
    const frame = &(activeAotCallState() orelse return null).trap_decode;
    if (frame.func_names.len == 0) return null;
    const wasm_idx: u32 = @intCast(
        @as(usize, @intCast(local_idx)) + frame.imported_function_count,
    );
    for (frame.func_names) |entry| {
        if (entry.index == wasm_idx) return entry.name;
    }
    return null;
}

/// Probe entry-point intended to be called from gdb / watchpoint scripts
/// to map an arbitrary native PC to (local_func, rel_off, name) using the
/// currently-installed trap-decode globals. Prints one line on stderr.
pub export fn aotProbeDecodePc(pc: usize) callconv(.c) void {
    const loc = decodeTrapReturnAddress(pc);
    if (loc.name) |name| {
        std.debug.print(
            "[probe] pc=0x{x} code+0x{x} local_func[{d}] \"{s}\"+0x{x}\n",
            .{ pc, loc.code_off, loc.func_idx, name, loc.rel_off },
        );
    } else {
        std.debug.print(
            "[probe] pc=0x{x} code+0x{x} local_func[{d}]+0x{x}\n",
            .{ pc, loc.code_off, loc.func_idx, loc.rel_off },
        );
    }
}

pub fn aotTrapOOB(vmctx: *VmCtx) callconv(.c) noreturn {
    const loc = decodeTrapReturnAddress(@returnAddress());
    if (isTrapCatching()) {
        // Caller has armed trap-as-error; unwind instead of exiting.
        trapLongjmp();
    }
    // Flush any buffered stdout from the guest before we tear down the
    // process so user-visible output isn't lost. Best-effort.
    if (loc.name) |name| {
        std.debug.print(
            "wasm trap: out of bounds memory access (code+0x{x}, local_func[{d}] \"{s}\"+0x{x}, mem_size=0x{x})\n",
            .{ loc.code_off, loc.func_idx, name, loc.rel_off, vmctx.memory_size },
        );
    } else {
        std.debug.print(
            "wasm trap: out of bounds memory access (code+0x{x}, local_func[{d}]+0x{x}, mem_size=0x{x})\n",
            .{ loc.code_off, loc.func_idx, loc.rel_off, vmctx.memory_size },
        );
    }
    aotTrapOobDumpMem(vmctx);
    std.process.exit(2);
}

/// Forensic dump for #719. Enabled by `WAMR_TRAP_OOB_DUMP=<hex>[,<hex>...]`
/// (each value is a wasm linear-memory address). For every address we
/// print the surrounding 64 bytes, and if the 32-bit value there looks
/// like a valid wasm-memory pointer we follow it and print 64 bytes
/// there too. Intentionally side-effect free if the env var is unset.
fn aotTrapOobDumpMem(vmctx: *VmCtx) void {
    const env = g_trap_oob_dump_env orelse return;
    if (vmctx.memory_base == 0 or vmctx.memory_size == 0) return;
    const mem: [*]const u8 = @ptrFromInt(vmctx.memory_base);

    var it = std.mem.splitScalar(u8, env, ',');
    while (it.next()) |tok| {
        const trimmed = std.mem.trim(u8, tok, " \t\r\n");
        if (trimmed.len == 0) continue;
        const hex = if (std.mem.startsWith(u8, trimmed, "0x") or std.mem.startsWith(u8, trimmed, "0X"))
            trimmed[2..]
        else
            trimmed;
        const addr = std.fmt.parseInt(u64, hex, 16) catch {
            std.debug.print("[trap-dump] bad hex token: '{s}'\n", .{trimmed});
            continue;
        };
        dumpAddr(mem, vmctx.memory_size, addr, 0);
    }
}

fn dumpAddr(mem: [*]const u8, mem_size: usize, addr: u64, depth: u8) void {
    const start: u64 = if (addr >= 32) addr - 32 else 0;
    const end: u64 = @min(addr + 64, mem_size);
    if (start >= end) {
        std.debug.print("[trap-dump] addr=0x{x} OUT OF RANGE (mem_size=0x{x})\n", .{ addr, mem_size });
        return;
    }
    std.debug.print("[trap-dump] addr=0x{x} window [0x{x}..0x{x}):\n", .{ addr, start, end });
    var off: u64 = start;
    while (off < end) : (off += 16) {
        const row_end = @min(off + 16, end);
        std.debug.print("  0x{x:0>8}:", .{off});
        var i: u64 = off;
        while (i < row_end) : (i += 1) {
            std.debug.print(" {x:0>2}", .{mem[@intCast(i)]});
        }
        var pad: u64 = row_end;
        while (pad < off + 16) : (pad += 1) std.debug.print("   ", .{});
        std.debug.print("  |", .{});
        i = off;
        while (i < row_end) : (i += 1) {
            const c = mem[@intCast(i)];
            std.debug.print("{c}", .{if (c >= 0x20 and c < 0x7f) c else @as(u8, '.')});
        }
        std.debug.print("|\n", .{});
    }

    if (depth >= 2) return;
    // Try interpreting the aligned u32 at `addr` as a wasm-memory pointer
    // and recurse one level so we can chase treebin-slot → chunk-header.
    if (addr + 4 <= mem_size and addr % 4 == 0) {
        const ptr_le = std.mem.readInt(u32, @as(*const [4]u8, @ptrCast(mem + addr)), .little);
        if (ptr_le != 0 and ptr_le < mem_size) {
            std.debug.print("[trap-dump]   *(u32)0x{x} = 0x{x} (in-range; following)\n", .{ addr, ptr_le });
            dumpAddr(mem, mem_size, ptr_le, depth + 1);
        }
    }
}

/// Resolve the trapping wasm function from `@returnAddress()` for the
/// non-OOB trap helpers (`aotTrapUnreachable`, `aotTrapInt*`,
/// `aotTrapInvalidConversion`). Returns `(local_idx, rel_off, name_opt)`
/// where `local_idx == -1` when no `func_offsets` entry contains the PC.
const TrapPcLocation = struct {
    code_off: usize,
    func_idx: isize,
    rel_off: usize,
    name: ?[]const u8,
};

fn decodeTrapReturnAddress(ret_addr: usize) TrapPcLocation {
    const frame = if (activeAotCallState()) |state|
        state.trap_decode
    else
        TrapDecodeFrame{};
    const code_off_s: isize = @as(isize, @bitCast(ret_addr)) -
        @as(isize, @bitCast(frame.code_base));
    const code_off: usize = if (code_off_s >= 0) @intCast(code_off_s) else 0;
    // PC falls outside the currently-installed code region (e.g. a trap
    // raised from a different AOT instance whose decode frame was just
    // restored, or a PC corruption). Refuse to scan the function offsets so we
    // don't report a stale last-index match.
    if (frame.code_size == 0 or code_off >= frame.code_size) {
        printTrapDecodeForensic(ret_addr, "ret-addr above code blob or no code installed");
        return .{ .code_off = code_off, .func_idx = -1, .rel_off = 0, .name = null };
    }
    if (code_off_s < 0) {
        // #406: ret-addr below the AOT code blob is anomalous — the trap
        // helper expects to be called from inside AOT code, so `@returnAddress()`
        // should land inside the active call's code blob.
        // Falling below indicates either host-stack corruption (the return
        // slot got overwritten before the trap helper read it) or the code base
        // being stale. Surface the raw values so the next time this fires we
        // have actionable data instead of the misleading degenerate decode
        // (`local_func[0] "__wasm_call_ctors"+0x0`) below.
        printTrapDecodeForensic(ret_addr, "ret-addr below code blob (host stack corruption?)");
    }
    var func_idx: isize = -1;
    var func_start: usize = 0;
    for (frame.func_offsets, 0..) |off, idx| {
        if (off <= code_off) {
            func_idx = @intCast(idx);
            func_start = off;
        } else break;
    }
    const rel_off: usize = if (code_off >= func_start) code_off - func_start else 0;
    return .{
        .code_off = code_off,
        .func_idx = func_idx,
        .rel_off = rel_off,
        .name = lookupLocalFuncName(func_idx),
    };
}

/// #406 forensic: when the trap-PC decoder hits a fallback path
/// (ret_addr outside the AOT code blob), print the raw inputs so the
/// next time a flake fires we get actionable data instead of just the
/// misleading symbolic decode the caller is about to print. Tagged
/// `[#406]` for grep-ability across CI logs.
fn printTrapDecodeForensic(ret_addr: usize, why: []const u8) void {
    const frame = if (activeAotCallState()) |state|
        state.trap_decode
    else
        TrapDecodeFrame{};
    std.debug.print(
        "[#406] trap decoder fallback: {s} — raw ret=0x{x} code_base=0x{x} code_size=0x{x}\n",
        .{ why, ret_addr, frame.code_base, frame.code_size },
    );
}

fn printTrapWithPc(kind: []const u8, loc: TrapPcLocation) void {
    if (loc.name) |name| {
        std.debug.print(
            "wasm trap: {s} (code+0x{x}, local_func[{d}] \"{s}\"+0x{x})\n",
            .{ kind, loc.code_off, loc.func_idx, name, loc.rel_off },
        );
    } else {
        std.debug.print(
            "wasm trap: {s} (code+0x{x}, local_func[{d}]+0x{x})\n",
            .{ kind, loc.code_off, loc.func_idx, loc.rel_off },
        );
    }
}

/// Host helper invoked from AOT-compiled code for `unreachable`,
/// integer divide-by-zero, INT_MIN/-1 overflow, and invalid float→int
/// conversion. When the caller has armed trap-as-error
/// (the active call state's `trap_catching` is true), longjmps back to
/// `callFuncScalar`, which
/// returns `error.WasmTrap`. Otherwise prints a diagnostic and exits.
pub fn aotTrapUnreachable(vmctx: *VmCtx) callconv(.c) noreturn {
    _ = vmctx;
    const loc = decodeTrapReturnAddress(@returnAddress());
    if (isTrapCatching()) trapLongjmp();
    printTrapWithPc("unreachable", loc);
    std.process.exit(2);
}

pub fn aotTrapIntDivZero(vmctx: *VmCtx) callconv(.c) noreturn {
    _ = vmctx;
    const loc = decodeTrapReturnAddress(@returnAddress());
    if (isTrapCatching()) trapLongjmp();
    printTrapWithPc("integer divide by zero", loc);
    std.process.exit(2);
}

pub fn aotTrapIntOverflow(vmctx: *VmCtx) callconv(.c) noreturn {
    _ = vmctx;
    const loc = decodeTrapReturnAddress(@returnAddress());
    if (isTrapCatching()) trapLongjmp();
    printTrapWithPc("integer overflow", loc);
    std.process.exit(2);
}

pub fn aotTrapInvalidConversion(vmctx: *VmCtx) callconv(.c) noreturn {
    _ = vmctx;
    const loc = decodeTrapReturnAddress(@returnAddress());
    if (isTrapCatching()) trapLongjmp();
    printTrapWithPc("invalid conversion to integer", loc);
    std.process.exit(2);
}

/// Unwind a trap raised by a native host adapter. Unlike generated-code trap
/// helpers, this deliberately skips return-address decoding because the
/// caller lives in host code rather than the AOT text mapping.
pub fn aotTrapHost(vmctx: *VmCtx, fallback_exit_code: u8) noreturn {
    _ = vmctx;
    if (isTrapCatching()) trapLongjmp();
    std.process.exit(fallback_exit_code);
}

/// Host helper invoked from AOT-compiled `throw` when control flow
/// cannot find a matching catch handler inside the current function
/// (the only case lowered by commit 3 of #672). Reads the payload
/// AOT codegen stored in `vmctx.exception_params` (count =
/// `vmctx.exception_param_count`) and surfaces an uncaught-exception
/// trap. Cross-function unwind + in-function catch dispatch are
/// scheduled for later commits in PR #674; once they land, this helper
/// becomes the leaf fallback only.
pub fn aotThrowUncaught(vmctx: *VmCtx, tag_inst: *types.TagInstance) callconv(.c) noreturn {
    if (isTrapCatching()) trapLongjmp();
    std.debug.print(
        "wasm trap: uncaught exception (tag=0x{x}, payload_words={d})\n",
        .{ @intFromPtr(tag_inst), vmctx.exception_param_count },
    );
    std.process.exit(2);
}

// ── Futex helpers for atomic.wait / atomic.notify ────────────────────

/// Host helper for `memory.atomic.wait32`.
/// Blocks the calling thread if `mem[addr] == expected`.
/// Returns: 0 = woken by notify, 1 = not-equal, 2 = timed-out.
pub fn aotAtomicWait32(vmctx: *VmCtx, addr: u32, expected: u32, timeout_lo: u32, timeout_hi: u32) callconv(.c) i32 {
    const timeout_ns: i64 = @bitCast(@as(u64, timeout_hi) << 32 | @as(u64, timeout_lo));
    const mem = aotWaitMemory(vmctx, addr, 4) orelse return 1;
    const result = mem.wait32(addr, expected, timeout_ns) catch |err| switch (err) {
        error.NotShared => return 1,
        error.OutOfBounds,
        error.Unaligned,
        error.InvalidAddress,
        error.InvalidArgument,
        error.Unsupported,
        error.SystemFailure,
        => return 2,
    };
    return switch (result) {
        .notified, .not_equal, .timed_out => @intCast(@intFromEnum(result)),
        .cancelled, .closed => 2,
    };
}

/// Host helper for `memory.atomic.wait64`.
pub fn aotAtomicWait64(vmctx: *VmCtx, addr: u32, exp_lo: u32, exp_hi: u32, timeout_lo: u32, timeout_hi: u32) callconv(.c) i32 {
    const expected: u64 = @as(u64, exp_hi) << 32 | @as(u64, exp_lo);
    const timeout_ns: i64 = @bitCast(@as(u64, timeout_hi) << 32 | @as(u64, timeout_lo));
    const mem = aotWaitMemory(vmctx, addr, 8) orelse return 1;
    const result = mem.wait64(addr, expected, timeout_ns) catch |err| switch (err) {
        error.NotShared => return 1,
        error.OutOfBounds,
        error.Unaligned,
        error.InvalidAddress,
        error.InvalidArgument,
        error.Unsupported,
        error.SystemFailure,
        => return 2,
    };
    return switch (result) {
        .notified, .not_equal, .timed_out => @intCast(@intFromEnum(result)),
        .cancelled, .closed => 2,
    };
}

/// Host helper for `memory.atomic.notify`.
/// Wakes up to `count` threads waiting on `mem[addr]`.
/// Returns the number of threads actually woken.
pub fn aotAtomicNotify(vmctx: *VmCtx, addr: u32, count: u32) callconv(.c) i32 {
    const mem = aotWaitMemory(vmctx, addr, 4) orelse return 0;
    return @intCast(mem.notify(addr, count) catch |err| switch (err) {
        error.NotShared,
        error.OutOfBounds,
        error.Unaligned,
        error.InvalidAddress,
        error.InvalidArgument,
        error.Unsupported,
        error.SystemFailure,
        => return 0,
    });
}

fn aotWaitMemory(vmctx: *VmCtx, addr: u32, width: u32) ?*types.MemoryInstance {
    if (vmctx.instance_ptr == 0) return null;
    const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
    if (inst.memories.len == 0) return null;
    const mem = inst.memories[0];
    if (@as(u64, addr) + width > mem.byteLen()) return null;
    return mem;
}

/// Host helper invoked from AOT-compiled `memory.grow` sites.
/// Grows `inst.memories[0]` by `delta_pages`, reallocating or
/// committing the host buffer if needed (see `MemoryInstance.grow`),
/// updates `vmctx` mirror fields (memory_base/size/pages) so
/// subsequent loads/stores see the new buffer, and returns the previous
/// page count. Returns -1 on failure (OOM or exceeds max).
pub fn memGrowHelper(vmctx: *VmCtx, delta_pages: i32) callconv(.c) i32 {
    if (vmctx.instance_ptr == 0) return -1;
    const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
    if (inst.memories.len == 0) return -1;
    const mem = inst.memories[0];
    if (delta_pages < 0) return -1;
    const delta: u32 = @intCast(delta_pages);
    const old_data_ptr = mem.data.ptr;
    // Route through `MemoryInstance.grow` so reserved (mmap-backed)
    // memories take the `mprotect`/commit path and legacy ones take
    // the `allocator.realloc` path. (#752: the legacy path may move
    // `data.ptr`; the reserved path keeps it pinned, preserving
    // external aliases held outside the vmctx-subscriber mechanism.)
    const old_pages = mem.grow(delta, inst.allocator) catch return -1;
    const new_pages = mem.pageCount();
    refreshVmCtxMemory(vmctx, mem);

    if (mem.data.ptr != old_data_ptr or new_pages != old_pages) {
        mem.subscriber_mutex.lock();
        defer mem.subscriber_mutex.unlock();
        for (mem.vmctx_subscribers.items) |subscriber_opaque| {
            const subscriber: *VmCtx = @ptrCast(@alignCast(subscriber_opaque));
            if (subscriber == vmctx) continue;
            refreshVmCtxMemory(subscriber, mem);
        }
    }

    if (comptime windows_trap_supported) {
        // Subscriber vmctxs may be inactive or on another thread, so only
        // the active grow caller updates its own VEH diagnostics frame.
        if (activeAotCallState()) |state| {
            state.trap_decode.mem_base = vmctx.memory_base;
            state.trap_decode.mem_size = vmctx.memory_size;
        }
    }
    return @intCast(old_pages);
}

/// Host helper invoked from AOT-compiled `memory.fill` sites.
/// wasm: `memory.fill(dst, val, len)` → writes `val & 0xFF` to `len`
/// bytes starting at `dst`. Traps (via `vmctx.trap_oob_fn`) if
/// `dst + len > memory_size`. `len == 0` is a no-op even at the
/// boundary, matching the wasm spec's pure-inequality check.
pub fn memFillHelper(vmctx: *VmCtx, dst: u32, val: u32, len: u32) callconv(.c) void {
    const end: u64 = @as(u64, dst) + @as(u64, len);
    if (end > vmctx.memory_size) {
        const trap_fn: *const fn (*VmCtx) callconv(.c) noreturn =
            @ptrFromInt(vmctx.trap_oob_fn);
        trap_fn(vmctx);
    }
    if (len == 0) return;
    const base: [*]u8 = @ptrFromInt(vmctx.memory_base);
    const b: u8 = @truncate(val);
    @memset(base[dst..@intCast(end)], b);
}

/// Host helper invoked from AOT-compiled `memory.copy` sites.
/// wasm: `memory.copy(dst, src, len)` — bounds-checks both ranges
/// then performs a memmove-style copy (correct for overlapping
/// regions).
pub fn memCopyHelper(vmctx: *VmCtx, dst: u32, src: u32, len: u32) callconv(.c) void {
    const d_end: u64 = @as(u64, dst) + @as(u64, len);
    const s_end: u64 = @as(u64, src) + @as(u64, len);
    if (d_end > vmctx.memory_size or s_end > vmctx.memory_size) {
        const trap_fn: *const fn (*VmCtx) callconv(.c) noreturn =
            @ptrFromInt(vmctx.trap_oob_fn);
        trap_fn(vmctx);
    }
    if (len == 0) return;
    const base: [*]u8 = @ptrFromInt(vmctx.memory_base);
    const d_slice = base[dst..@intCast(d_end)];
    const s_slice = base[src..@intCast(s_end)];
    if (dst <= src) {
        std.mem.copyForwards(u8, d_slice, s_slice);
    } else {
        std.mem.copyBackwards(u8, d_slice, s_slice);
    }
}

/// AOT host helper for table.grow.
/// Grows table 0 by `delta` entries (each holding a native function pointer),
/// initializing new slots with `init_val`. Updates `vmctx.func_table_ptr` and
/// `vmctx.func_table_len` on success. Returns previous table size, or -1 on
/// failure (allocation failure or max-size violation).
pub fn tableGrowHelper(vmctx: *VmCtx, init_val: i64, delta: i32, table_idx: u32) callconv(.c) i32 {
    if (vmctx.instance_ptr == 0) return -1;
    const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
    if (delta < 0) return -1;
    const delta_u: u32 = @intCast(delta);
    const fill_ptr: usize = @as(usize, @bitCast(@as(i64, init_val)));

    // Table 0: resize the shared `func_table` so call_indirect/call_ref
    // keep seeing the growth. Other tables: resize the backing storage
    // in `extra_tables_storage`.
    if (table_idx == 0) {
        const shared0 = if (inst.tables.len > 0) inst.tables[0] else null;
        if (shared0) |shared| shared.lock();
        defer if (shared0) |shared| shared.unlock();
        const old_size: u32 = @intCast(inst.func_table.len);
        const new_size_u64: u64 = @as(u64, old_size) + @as(u64, delta_u);
        const max_cap: u64 = blk: {
            if (inst.tables.len > 0) {
                break :blk inst.tables[0].table_type.limits.max orelse 0xFFFF_FFFF;
            } else break :blk 0xFFFF_FFFF;
        };
        if (new_size_u64 > max_cap) return -1;
        const new_size: usize = @intCast(new_size_u64);
        // Realloc the shared backing (owned by TableInstance). Also updates
        // inst.func_table (which aliases it).
        const old_slice: []usize = if (shared0) |s| s.native_backing else inst.func_table;
        const new_table = inst.allocator.realloc(old_slice, new_size) catch return -1;
        var i: usize = old_size;
        while (i < new_size) : (i += 1) new_table[i] = fill_ptr;
        inst.func_table = new_table;
        if (shared0) |s| s.native_backing = new_table;
        // Keep type_backing sized in lockstep. New tail slots are 0
        // (null sig_id); call_indirect treats 0 as "uninitialized
        // element" trap. A follow-up patch will populate new tail
        // slots with the sig_id of `fill_ptr` when it is non-null.
        if (shared0) |s| {
            const old_tb: []u32 = s.type_backing;
            const new_tb = inst.allocator.realloc(old_tb, new_size) catch return -1;
            var k: usize = old_size;
            while (k < new_size) : (k += 1) new_tb[k] = 0;
            s.type_backing = new_tb;
        }
        vmctx.func_table_ptr = @intFromPtr(new_table.ptr);
        vmctx.func_table_len = @intCast(new_size);
        if (inst.tables_info.len > 0) {
            inst.tables_info[0].ptr = @intFromPtr(new_table.ptr);
            inst.tables_info[0].len = @intCast(new_size);
            if (shared0) |s| {
                inst.tables_info[0].type_backing_ptr = @intFromPtr(s.type_backing.ptr);
            }
        }
        // Keep the shared `TableInstance` in sync with the current size.
        // Future importers that receive this `TableInstance` (via the
        // ImportRegistry table-swap in the test harness) size their own
        // `func_table` from `elements.len` in `mapCodeExecutable`; without
        // reallocating `elements` the importer would see the stale pre-grow
        // size and return wrong results from `table.size`/`table.grow`.
        if (inst.tables.len > 0) {
            const shared = inst.tables[0];
            shared.table_type.limits.min = @intCast(new_size);
            const new_elements = inst.allocator.realloc(shared.elements, new_size) catch return -1;
            var j: usize = old_size;
            while (j < new_size) : (j += 1) {
                new_elements[j] = types.TableElement.nullForType(shared.table_type.elem_type);
            }
            shared.elements = new_elements;
            refreshTableSubscribers(shared);
        }
        return @intCast(old_size);
    }

    if (table_idx >= inst.tables_info.len) return -1;
    if (table_idx - 1 >= inst.extra_tables_storage.len) return -1;
    const ti = &inst.tables_info[table_idx];
    const store = &inst.extra_tables_storage[table_idx - 1];
    const old_size: u32 = ti.len;
    const new_size_u64: u64 = @as(u64, old_size) + @as(u64, delta_u);
    const max_cap: u64 = blk: {
        if (table_idx < inst.tables.len) {
            break :blk inst.tables[table_idx].table_type.limits.max orelse 0xFFFF_FFFF;
        } else break :blk 0xFFFF_FFFF;
    };
    if (new_size_u64 > max_cap) return -1;
    const new_size: usize = @intCast(new_size_u64);
    const shared_n = if (table_idx < inst.tables.len) inst.tables[table_idx] else null;
    if (shared_n) |shared| shared.lock();
    defer if (shared_n) |shared| shared.unlock();
    const old_slice: []usize = if (shared_n) |s| s.native_backing else store.*;
    const new_store = inst.allocator.realloc(old_slice, new_size) catch return -1;
    var i: usize = old_size;
    while (i < new_size) : (i += 1) new_store[i] = fill_ptr;
    store.* = new_store;
    if (shared_n) |s| s.native_backing = new_store;
    // Keep type_backing sized in lockstep (same caveat as table 0).
    if (shared_n) |s| {
        const old_tb: []u32 = s.type_backing;
        const new_tb = inst.allocator.realloc(old_tb, new_size) catch return -1;
        var k: usize = old_size;
        while (k < new_size) : (k += 1) new_tb[k] = 0;
        s.type_backing = new_tb;
    }
    ti.ptr = @intFromPtr(new_store.ptr);
    ti.len = @intCast(new_size);
    if (shared_n) |s| {
        ti.type_backing_ptr = @intFromPtr(s.type_backing.ptr);
    }
    if (table_idx < inst.tables.len) {
        const shared = inst.tables[table_idx];
        shared.table_type.limits.min = @intCast(new_size);
        const new_elements = inst.allocator.realloc(shared.elements, new_size) catch return -1;
        var j: usize = old_size;
        while (j < new_size) : (j += 1) {
            new_elements[j] = types.TableElement.nullForType(shared.table_type.elem_type);
        }
        shared.elements = new_elements;
        refreshTableSubscribers(shared);
    }
    return @intCast(old_size);
}

/// Host helper invoked from AOT-compiled `table.init` sites.
///
/// Copies `len` function references from element segment `seg_idx` starting
/// at offset `src` into table `table_idx` starting at offset `dst`. Traps
/// (via `trapLongjmp` when armed, otherwise `process.exit(2)`) on:
///   - `seg_idx >= module.elem_segments.len`
///   - segment already dropped AND (`src != 0` or `len != 0`)
///   - `src + len > segment.func_indices.len`
///   - `table_idx >= tables_info.len`
///   - `dst + len > table.len`
///
/// Arguments are packed to fit the 4-register fast path of Win64 and SysV
/// so codegen can avoid stack spills for the typical call site:
///   packed_seg_table = seg_idx | (table_idx << 32)
///   packed_dst_src   = dst     | (src << 32)
pub fn tableInitHelper(
    vmctx: *VmCtx,
    packed_seg_table: u64,
    packed_dst_src: u64,
    len: u32,
) callconv(.c) void {
    if (vmctx.instance_ptr == 0) {
        aotTrapUnreachable(vmctx);
    }
    const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
    const module = inst.module_ref orelse inst.module;

    const seg_idx: u32 = @truncate(packed_seg_table);
    const table_idx: u32 = @truncate(packed_seg_table >> 32);
    const dst: u32 = @truncate(packed_dst_src);
    const src: u32 = @truncate(packed_dst_src >> 32);

    if (seg_idx >= module.elem_segments.len) aotTrapUnreachable(vmctx);
    const dropped = inst.elem_segments_dropped.len > seg_idx and inst.elem_segments_dropped[seg_idx];
    const seg = module.elem_segments[seg_idx];

    // Spec: table.init with a dropped segment traps iff src or len is non-zero.
    const seg_len: u64 = if (dropped) 0 else @as(u64, @intCast(seg.func_indices.len));
    if (@as(u64, src) + @as(u64, len) > seg_len) aotTrapUnreachable(vmctx);

    if (table_idx >= inst.tables_info.len) aotTrapUnreachable(vmctx);
    const ti = &inst.tables_info[table_idx];
    if (@as(u64, dst) + @as(u64, len) > @as(u64, ti.len)) aotTrapUnreachable(vmctx);

    if (len == 0) return;

    // Copy native function addresses into the table descriptor's backing.
    const backing: [*]usize = @ptrFromInt(@as(usize, @intCast(ti.ptr)));
    const shared_opt: ?*types.TableInstance =
        if (table_idx < inst.tables.len) inst.tables[table_idx] else null;
    if (shared_opt) |shared| shared.lock();
    defer if (shared_opt) |shared| shared.unlock();
    var i: u32 = 0;
    while (i < len) : (i += 1) {
        const fi = seg.func_indices[src + i];
        const addr: usize = if (fi == std.math.maxInt(u32) or fi >= inst.funcptrs.len)
            0
        else
            inst.funcptrs[fi];
        backing[dst + i] = addr;
        // Mirror canonical sig_id into type_backing so a subsequent
        // call_indirect on this slot sees the correct sig (or 0 for null).
        if (shared_opt) |shared| {
            if (dst + i < shared.type_backing.len) {
                if (fi == std.math.maxInt(u32) or fi >= inst.func_sig_ids.len) {
                    shared.type_backing[dst + i] = 0;
                } else {
                    shared.type_backing[dst + i] = inst.func_sig_ids[fi];
                }
            }
        }
    }

    // Mirror into the shared `TableInstance.elements` for importer consistency
    // (matches tableGrowHelper's pattern). Funcref elements are
    // `{ .value = .{ .funcref = ?u32 }, .module_inst = ... }`.
    if (table_idx < inst.tables.len) {
        const shared = inst.tables[table_idx];
        if (shared.elements.len >= @as(usize, dst) + @as(usize, len)) {
            i = 0;
            while (i < len) : (i += 1) {
                const fi = seg.func_indices[src + i];
                if (fi == std.math.maxInt(u32)) {
                    shared.elements[dst + i] = types.TableElement.nullForType(shared.table_type.elem_type);
                } else {
                    shared.elements[dst + i] = .{ .value = .{ .funcref = fi } };
                }
            }
        }
    }
}

/// Host helper invoked from AOT-compiled `elem.drop` sites. Marks the
/// passive element segment as dropped. Idempotent. Out-of-range indices
/// are treated as a no-op (validation guarantees they don't appear in
/// well-formed modules).
pub fn elemDropHelper(vmctx: *VmCtx, seg_idx: u32) callconv(.c) void {
    if (vmctx.instance_ptr == 0) return;
    const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
    if (seg_idx < inst.elem_segments_dropped.len) {
        inst.elem_segments_dropped[seg_idx] = true;
    }
}

/// Host helper invoked from AOT-compiled `table.set` sites.
/// Writes the funcptr into `tables_info[table_idx].ptr[elem_idx]`
/// (native_backing) and derives + writes the matching sig_id into
/// `type_backing[elem_idx]` via binary search on `inst.ptr_to_sig`.
/// Traps on out-of-bounds elem_idx.
pub fn tableSetHelper(vmctx: *VmCtx, table_idx: u32, elem_idx: u32, value: usize) callconv(.c) void {
    if (vmctx.instance_ptr == 0) {
        aotTrapUnreachable(vmctx);
    }
    const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
    if (table_idx >= inst.tables_info.len) aotTrapUnreachable(vmctx);
    const ti = &inst.tables_info[table_idx];
    if (elem_idx >= ti.len) aotTrapUnreachable(vmctx);
    const shared_opt: ?*types.TableInstance =
        if (table_idx < inst.tables.len) inst.tables[table_idx] else null;
    if (shared_opt) |shared| shared.lock();
    defer if (shared_opt) |shared| shared.unlock();

    // Write native pointer into backing store.
    const backing: [*]usize = @ptrFromInt(@as(usize, @intCast(ti.ptr)));
    backing[elem_idx] = value;

    // Derive sig_id from ptr_to_sig via binary search.
    const sig_id: u32 = if (value == 0) 0 else blk: {
        const entries = inst.ptr_to_sig;
        const needle: u64 = @as(u64, value);
        var lo: usize = 0;
        var hi: usize = entries.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (entries[mid].ptr < needle) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        break :blk if (lo < entries.len and entries[lo].ptr == needle)
            entries[lo].sig_id
        else
            0;
    };

    // Write sig_id into type_backing.
    if (ti.type_backing_ptr != 0) {
        const type_backing: [*]u32 = @ptrFromInt(@as(usize, @intCast(ti.type_backing_ptr)));
        type_backing[elem_idx] = sig_id;
    }

    // Mirror into the shared `TableInstance.elements` for consistency.
    if (shared_opt) |shared| {
        if (elem_idx < shared.elements.len) {
            if (value == 0) {
                shared.elements[elem_idx] = types.TableElement.nullForType(shared.table_type.elem_type);
            } else {
                // Derive funcidx from funcptrs.
                var funcidx: ?u32 = null;
                for (inst.funcptrs, 0..) |p, fi| {
                    if (p == value) {
                        funcidx = @intCast(fi);
                        break;
                    }
                }
                shared.elements[elem_idx] = .{ .value = .{ .funcref = funcidx } };
            }
        }
    }
}

/// The native machine architecture, resolved at comptime.
const native_arch: enum { x86_64, aarch64, unsupported } = switch (builtin.cpu.arch) {
    .x86_64 => .x86_64,
    .aarch64 => .aarch64,
    else => .unsupported,
};

/// Whether the current target can execute AOT code.
const can_execute_native = native_arch != .unsupported;

/// A single lazily-compiled function's tracked executable mapping, so
/// `AotInstance.destroy()` can `munmap` + unregister it (each lazy
/// function gets its own small mapping, separate from the instance's
/// main `code_base` blob — see the design doc's "Deferred to
/// follow-up" section on batching these).
pub const LazyCompiledFunc = struct {
    addr: [*]const u8,
    size: usize,
};

const LazySlotStateAtomic = std.atomic.Value(u8);

/// Lazy-JIT per-instance state. Only ever populated (non-empty) when
/// `config.lazy_jit` is true AND the instance was produced by the
/// lazy-JIT-aware compile path (`component_aot_compile.compileCoreWasmCached`
/// with `opts.lazy_jit = true`). Deferred locals resolve through either a
/// #887 text-section entry stub (direct-call-graph functions) or a #888
/// per-instance native trampoline (table/`ref.func`/`call_indirect`-reachable
/// leaf functions) whose stable pointer is published in `funcptrs` / tables
/// for the lifetime of the instance. Comptime-gated to a zero-cost `void`
/// field on `AotInstance` for every other build — see that field's doc
/// comment.
///
/// `AotInstance`/`runtime.zig` must not depend on compiler types
/// directly (the AOT-only `wamr` binary links no compiler at all,
/// #695), so the actual "compile function N now" logic is a
/// type-erased callback supplied by the JIT-side driver in
/// `aot_compile.zig`, mirroring the existing `TrampolinePool.ctx:
/// *anyopaque` pattern in `host_trampolines.zig`.
pub const LazyJitState = struct {
    pub const SlotState = enum(u8) {
        /// Not lazy-eligible; resolve through the instance's eagerly
        /// mapped `code_base` as normal.
        inactive = 0,
        /// Lazy-eligible but not yet compiled.
        pending = 1,
        /// One thread won the right to compile this local now.
        compiling = 2,
        /// The lazily compiled mapping in `compiled[local_idx]` has
        /// been fully published and can be reused by all callers.
        ready = 3,
    };

    /// LOCAL function idx → per-slot lazy state. Only lazy-eligible
    /// locals ever leave `inactive`; they transition
    /// `pending -> compiling -> ready`. A failed compile stores
    /// `pending` again so a later caller can retry.
    slot_states: []LazySlotStateAtomic = &.{},
    /// LOCAL function idx → compiled code once resolved. Parallel to
    /// `slot_states`; entries only meaningful where
    /// `slot_states[i] == .ready`. Published before the corresponding
    /// `.ready` release-store and read after an acquire-load of that
    /// state, so waiters never observe a torn/half-written
    /// `LazyCompiledFunc`. Freed (munmapped) by `AotInstance.destroy()`.
    compiled: []?LazyCompiledFunc = &.{},
    /// Per-instance pool owning the executable trampoline stubs
    /// published for deferred locals. Freed on `AotInstance.destroy()`.
    trampoline_pool: ?*host_trampolines.TrampolinePool = null,
    /// LOCAL function idx → stable trampoline pointer published into
    /// `funcptrs` / tables for still-pending lazy locals.
    trampolines: []const usize = &.{},
    /// Opaque context for `compile_fn`, owned by whoever set up this
    /// `LazyJitState` (the JIT driver in `aot_compile.zig`). Freed by
    /// that same driver, not by `AotInstance.destroy()` — see
    /// `docs/design/lazy-jit-spike.md`.
    compile_ctx: ?*anyopaque = null,
    /// Compile local function `local_idx` now; returns its
    /// mapped-executable native code (address + byte size, so
    /// `destroy()` can `munmap` + unregister it later). Preserves
    /// `error.CodeBudgetExceeded` (and any other `RuntimeError`) from
    /// the tracked mapping path, so callers can distinguish "budget
    /// exceeded, retryable" from a hard mapping failure. Same-slot
    /// contenders are serialized by `slot_states`; at most one thread
    /// enters `compile_fn` for a given `local_idx` at a time, while
    /// different lazy locals may still compile independently. Any
    /// error resets the slot back to `pending` so a later caller can
    /// retry.
    compile_fn: ?*const fn (ctx: *anyopaque, local_idx: u32) RuntimeError!LazyCompiledFunc = null,

    pub fn slotState(self: *const LazyJitState, local_idx: usize) SlotState {
        if (local_idx >= self.slot_states.len) return .inactive;
        const state: SlotState = @enumFromInt(self.slot_states[local_idx].load(.acquire));
        return state;
    }

    fn resolveLocalAddr(self: *LazyJitState, local_idx: usize) RuntimeError!?[*]const u8 {
        if (local_idx >= self.slot_states.len) return null;

        const slot = &self.slot_states[local_idx];
        var spins: u32 = 0;
        while (true) {
            const state: SlotState = @enumFromInt(slot.load(.acquire));
            switch (state) {
                .inactive => return null,
                .ready => {
                    const compiled = self.compiled[local_idx] orelse return error.CodeMappingFailed;
                    return compiled.addr;
                },
                .pending => {
                    if (slot.cmpxchgWeak(
                        @intFromEnum(SlotState.pending),
                        @intFromEnum(SlotState.compiling),
                        .acquire,
                        .acquire,
                    ) != null) continue;

                    const compile_ctx = self.compile_ctx orelse {
                        slot.store(@intFromEnum(SlotState.pending), .release);
                        return error.CodeMappingFailed;
                    };
                    const compile_fn = self.compile_fn orelse {
                        slot.store(@intFromEnum(SlotState.pending), .release);
                        return error.CodeMappingFailed;
                    };
                    const compiled = compile_fn(compile_ctx, @intCast(local_idx)) catch |err| {
                        slot.store(@intFromEnum(SlotState.pending), .release);
                        return err;
                    };
                    self.compiled[local_idx] = compiled;
                    slot.store(@intFromEnum(SlotState.ready), .release);
                    return compiled.addr;
                },
                .compiling => {
                    if (spins < 1024) {
                        spins += 1;
                        std.atomic.spinLoopHint();
                    } else {
                        spins = 0;
                        std.Thread.yield() catch {};
                    }
                },
            }
        }
    }

    fn free(self: *LazyJitState, allocator: std.mem.Allocator) void {
        for (self.compiled) |maybe| {
            if (maybe) |c| {
                platform.munmap(@constCast(c.addr), c.size);
                JitCodeCache.unregister(c.size);
            }
        }
        if (self.trampoline_pool) |pool| {
            pool.deinit(allocator);
            allocator.destroy(pool);
        }
        allocator.free(self.slot_states);
        allocator.free(self.compiled);
        if (self.trampolines.len > 0) allocator.free(self.trampolines);
    }
};

fn resolveLazyCompiledAddr(inst: *AotInstance, local_idx: u32) ?[*]const u8 {
    if (comptime !config.lazy_jit) return null;
    if (local_idx >= inst.module.func_count) return null;

    const addr = (inst.lazy_jit.resolveLocalAddr(local_idx) catch return null) orelse return null;

    const func_idx = inst.module.import_function_count + local_idx;
    if (func_idx < inst.funcptrs.len) {
        inst.funcptrs[func_idx] = @intFromPtr(addr);
    }

    return addr;
}

pub fn lazyCompileHelper(vmctx: *VmCtx, local_idx: u32) callconv(.c) usize {
    if (comptime !config.lazy_jit) return 0;
    if (vmctx.instance_ptr == 0) return 0;
    const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
    const addr = resolveLazyCompiledAddr(inst, local_idx) orelse return 0;
    return @intFromPtr(addr);
}

const SharedCodeMapping = struct {
    addr: [*]const u8,
    size: usize,
    allocator: std.mem.Allocator,
    references: std.atomic.Value(usize) = std.atomic.Value(usize).init(1),

    fn retain(self: *SharedCodeMapping) void {
        _ = self.references.fetchAdd(1, .acq_rel);
    }

    fn release(self: *SharedCodeMapping) void {
        const previous = self.references.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
        if (previous != 1) return;
        platform.munmap(@ptrCast(@constCast(self.addr)), self.size);
        JitCodeCache.unregister(self.size);
        self.allocator.destroy(self);
    }

    fn referenceCount(self: *const SharedCodeMapping) usize {
        return self.references.load(.acquire);
    }
};

// ─── Instance ───────────────────────────────────────────────────────────────

pub const AotInstance = struct {
    module: *const aot_loader.AotModule,
    memories: []*types.MemoryInstance,
    /// #862 lazy-JIT design-spike hook (see `LazyJitState`'s doc
    /// comment). `void` (zero size, zero cost) in every build except
    /// `-Dlazy_jit=true`.
    lazy_jit: if (config.lazy_jit) LazyJitState else void = if (config.lazy_jit) .{} else {},
    /// True when the matching `memories[i]` entry was allocated by this
    /// instance. Borrowed imported-memory overrides leave this false, but are
    /// still retain()'d on borrow and release()'d on destroy.
    memories_owned: []bool = &.{},
    tables: []*types.TableInstance,
    /// True when the matching `tables[i]` entry was allocated by this
    /// instance. Borrowed imported-table overrides leave this false, but are
    /// still retain()'d on borrow and release()'d on destroy.
    tables_owned: []bool = &.{},
    globals: []*types.GlobalInstance,
    /// True when the matching `globals[i]` entry was allocated by this
    /// instance. Borrowed imported-global overrides leave this false, but are
    /// still retain()'d on borrow and release()'d on destroy.
    globals_owned: []bool = &.{},
    /// Tag instances (#672). `tags[0..import_tag_count]` are imported (and may
    /// be borrowed from a sibling instance via `imported_tag_overrides`);
    /// `tags[import_tag_count..]` are locally declared and own-allocated
    /// from `module.tag_types`. `tags_owned[i]` distinguishes the two for
    /// cleanup purposes (borrowed entries are NOT freed on destroy).
    tags: []*types.TagInstance = &.{},
    tags_owned: []bool = &.{},
    /// Byte offset for each wasm-flat global in `VmCtx.globals_ptr`.
    global_offsets: []u32 = &.{},
    /// Total byte size of the globals storage described by `global_offsets`.
    global_storage_size: u32 = 0,
    allocator: std.mem.Allocator,
    /// Stable vmctx storage used by AOT calls and MemoryInstance subscriber lists.
    vmctx: VmCtx = .{},
    /// Base address of the mapped executable code (null if not yet mapped).
    code_base: ?[*]const u8 = null,
    /// Size of the mapped executable region (for cleanup).
    code_size: usize = 0,
    /// Refcounted immutable text mapping shared by thread clones. Every clone
    /// has its own VmCtx and mutable execution state, while native table and
    /// funcref entries can safely keep one stable code address.
    code_mapping: if (config.lib_wasi_threads) ?*SharedCodeMapping else void =
        if (config.lib_wasi_threads) null else {},
    /// Resolved AOT host function pointers (one per import).
    host_functions: []const ?*const anyopaque = &.{},
    /// Native function pointer table for call_indirect (one per module function).
    func_table: []usize = &.{},
    /// Native function pointer array indexed by module funcidx (imports + locals).
    /// Used by `ref.func` which must yield a function's native address even when
    /// the function was never placed in a wasm table by an element segment.
    funcptrs: []usize = &.{},
    /// Per-table native descriptor array (one 16-byte slot per declared table):
    /// `extern struct { ptr: u64, len: u32, _pad: u32 }`.
    /// Slot 0 aliases `func_table`; slots 1+ back additional wasm tables so
    /// multi-table programs can do table.get/set/size/grow per-table without
    /// cross-table corruption.
    tables_info: []TableInfo = &.{},
    /// Backing storage for each per-table `usize` array (excluding table 0,
    /// which shares `func_table`). Entry `i-1` holds the backing slice for
    /// wasm table index `i`.
    extra_tables_storage: [][]usize = &.{},
    /// Per element-segment drop flag. `elem_segments_dropped[i]` is true when
    /// segment `i` has been consumed (either implicitly for active segments
    /// at instantiation, or by a successful `elem.drop`/`table.init` for
    /// passive segments). A dropped segment behaves as length-0 — `table.init`
    /// with `src>0` or `len>0` traps.
    elem_segments_dropped: []bool = &.{},
    /// The underlying module — kept on the instance so host helpers invoked
    /// from AOT code (e.g. `tableInitHelper`) can recover the passive
    /// segment data via `vmctx.instance_ptr`.
    module_ref: ?*const aot_loader.AotModule = null,
    /// Module type_idx → canonical sig_id (interned in the process-global
    /// SigRegistry at instantiate-time). Empty for modules with no types.
    sig_table: []u32 = &.{},
    /// Module funcidx (imports + locals) → canonical sig_id. Derived from
    /// `sig_table` and `module.local_func_type_indices` / import descriptors.
    func_sig_ids: []u32 = &.{},
    /// Sorted-by-ptr map from resolved native funcptr → sig_id. Populated in
    /// `mapCodeExecutable` once `funcptrs` hold real addresses.
    ptr_to_sig: []PtrSigEntry = &.{},
    /// One execution-local context per AOT instance/thread. Only its retained
    /// process-state reference is shared with sibling instances.
    thread_context: execution_context.ThreadExecutionContext = .{},

    pub fn attachProcessState(
        self: *AotInstance,
        process_state: execution_context.ProcessStateRef,
    ) void {
        self.thread_context.replaceProcessState(process_state);
        refreshVmCtxForInstance(self, null);
    }

    /// Narrow hook for the future AOT thread-clone path: inherit only the
    /// process reference, leaving task/cancel/trap/TLS metadata fresh.
    pub fn inheritProcessStateFrom(
        self: *AotInstance,
        parent: *const AotInstance,
    ) void {
        self.thread_context.replaceProcessState(parent.thread_context.process_state);
        refreshVmCtxForInstance(self, null);
    }

    pub fn setThreadManager(
        self: *AotInstance,
        manager: ?*thread_manager.ThreadManager,
    ) void {
        self.thread_context.setThreadGroup(if (manager) |value|
            @ptrCast(value)
        else
            null);
        refreshVmCtxForInstance(self, null);
    }
};

pub const TableInfo = extern struct {
    ptr: u64 = 0,
    len: u32 = 0,
    _pad: u32 = 0,
    /// Pointer to the parallel `u32` sig_id array (TableInstance.type_backing).
    /// `sig_id[i]` is the canonical sig_id of the function currently in slot
    /// `i`, or 0 for null/uninitialized. AOT `call_indirect` codegen reads
    /// this to compare against the expected sig_id from `VmCtx.sig_table_ptr`.
    /// 0 means "no type_backing set" (empty table); never dereferenced because
    /// bounds check (`len`) rejects all indices first.
    type_backing_ptr: u64 = 0,
};

// ─── Errors ─────────────────────────────────────────────────────────────────

pub const RuntimeError = error{
    OutOfMemory,
    CodeMappingFailed,
    FunctionNotFound,
    ExecutionFailed,
    TableAllocationFailed,
    WasmTrap,
    /// AOT execution is unavailable on this build's target architecture
    /// (currently only x86_64 and aarch64 are supported). The symbol is
    /// still linked so importers of `src/root.zig` build cleanly on
    /// riscv64 / etc., but invoking it at runtime fails fast.
    UnsupportedArchitecture,
    /// The selected AOT mode cannot provide an independent thread instance
    /// (currently lazy-JIT clones).
    WasiThreadsAotNotImplemented,
    /// #857: mapping this instance's code would push total resident
    /// JIT/AOT executable code past `JitCodeCache.budget_bytes`. Only
    /// possible when a nonzero budget is configured (default is
    /// unlimited); see `JitCodeCache` below.
    CodeBudgetExceeded,
};

// ─── Public API ─────────────────────────────────────────────────────────────

/// Instantiate an AOT module, producing a runnable AotInstance.
pub fn instantiate(module: *const aot_loader.AotModule, allocator: std.mem.Allocator) RuntimeError!*AotInstance {
    return instantiateWithOverrides(module, allocator, &.{}, &.{}, &.{}, &.{}, &.{});
}

/// Instantiate an AOT module, optionally borrowing `TableInstance`s,
/// `MemoryInstance`s, `GlobalInstance`s, and `TagInstance`s for each imported
/// slot. `imported_table_overrides[i]` maps to `module.importedTables()[i]`,
/// `imported_memory_overrides[i]` to `module.importedMemories()[i]`,
/// `imported_global_overrides[i]` to `module.importedGlobals()[i]`,
/// `imported_tag_overrides[i]` to `module.importedTags()[i]`;
/// null leaves that slot locally allocated (with a zero-valued default).
///
/// `imported_function_overrides[i]` maps to the *i-th function import*
/// in `module.imports` declaration order. A non-null entry replaces the
/// host-bridge / spectest resolution (used by #662 Phase C to wire
/// cross-instance core-to-core fn imports + trap-on-call stubs through
/// `host_trampolines`).
pub fn instantiateWithOverrides(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
    imported_table_overrides: []const ?*types.TableInstance,
    imported_memory_overrides: []const ?*types.MemoryInstance,
    imported_global_overrides: []const ?*types.GlobalInstance,
    imported_function_overrides: []const ?*const anyopaque,
    imported_tag_overrides: []const ?*types.TagInstance,
) RuntimeError!*AotInstance {
    if (comptime config.lib_wasi_threads and !config.wasi_threads.implementation.aot_thread_spawning) {
        for (module.imports) |imp| {
            if (imp.kind == .function and config.threads_feature.isThreadSpawnImport(imp.module_name, imp.field_name))
                return error.WasiThreadsAotNotImplemented;
        }
    }

    std.debug.assert(imported_table_overrides.len == 0 or imported_table_overrides.len == module.importedTables().len);
    std.debug.assert(imported_memory_overrides.len == 0 or imported_memory_overrides.len == module.importedMemories().len);
    std.debug.assert(imported_global_overrides.len == 0 or imported_global_overrides.len == module.importedGlobals().len);
    std.debug.assert(imported_function_overrides.len == 0 or imported_function_overrides.len == module.import_function_count);
    std.debug.assert(imported_tag_overrides.len == 0 or imported_tag_overrides.len == module.importedTags().len);

    var inst = allocator.create(AotInstance) catch return error.OutOfMemory;
    errdefer allocator.destroy(inst);

    inst.* = .{
        .module = module,
        .memories = &.{},
        .memories_owned = &.{},
        .tables = &.{},
        .tables_owned = &.{},
        .globals = &.{},
        .globals_owned = &.{},
        .tags = &.{},
        .tags_owned = &.{},
        .allocator = allocator,
        .module_ref = module,
    };

    // Per-segment drop flag. Active segments are marked dropped immediately
    // since their bytes have already been applied at instantiation — the spec
    // treats their post-instantiation state as "as if elem.drop had already
    // executed", meaning a subsequent table.init with a non-zero src/len must
    // trap.
    if (module.elem_segments.len > 0) {
        const dropped = allocator.alloc(bool, module.elem_segments.len) catch return error.OutOfMemory;
        for (module.elem_segments, 0..) |seg, i| {
            dropped[i] = !seg.is_passive;
        }
        inst.elem_segments_dropped = dropped;
    }
    errdefer if (inst.elem_segments_dropped.len > 0) allocator.free(inst.elem_segments_dropped);

    const mem_alloc = try allocateMemories(module, allocator, imported_memory_overrides);
    inst.memories = mem_alloc.memories;
    inst.memories_owned = mem_alloc.owned;
    errdefer freeMemories(inst.memories, inst.memories_owned, allocator);

    // Apply data segments to locally-allocated linear memory only.
    // Writing into a *borrowed* memory would scribble the exporter's bytes
    // during the importer's instantiation, since both VmCtx's see the same
    // backing buffer.
    for (module.data_segments) |seg| {
        if (seg.memory_idx >= inst.memories.len) continue;
        if (seg.memory_idx < inst.memories_owned.len and !inst.memories_owned[seg.memory_idx]) continue;
        const mem = inst.memories[seg.memory_idx];
        const end = @as(usize, seg.offset) + seg.data.len;
        if (end > mem.byteLen()) continue;
        @memcpy(mem.data[seg.offset..][0..seg.data.len], seg.data);
    }

    const table_alloc = try allocateTables(module, allocator, imported_table_overrides);
    inst.tables = table_alloc.tables;
    inst.tables_owned = table_alloc.owned;
    errdefer freeTables(inst.tables, inst.tables_owned, allocator);

    // If no tables but we have element segments, create a default table
    if (inst.tables.len == 0 and module.elem_segments.len > 0) {
        // Compute required table size from element segments
        var max_size: u32 = 0;
        for (module.elem_segments) |seg| {
            const end = seg.offset + @as(u32, @intCast(seg.func_indices.len));
            if (end > max_size) max_size = end;
        }
        if (max_size > 0) {
            const fallback = blk: {
                const tables = allocator.alloc(*types.TableInstance, 1) catch return error.OutOfMemory;
                errdefer allocator.free(tables);
                const owned = allocator.alloc(bool, 1) catch return error.OutOfMemory;
                errdefer allocator.free(owned);
                const elements = allocator.alloc(types.TableElement, max_size) catch return error.OutOfMemory;
                errdefer allocator.free(elements);
                for (elements) |*e| e.* = types.TableElement.nullForType(.funcref);
                const tbl = allocator.create(types.TableInstance) catch return error.OutOfMemory;
                errdefer allocator.destroy(tbl);
                tbl.* = .{
                    .table_type = .{ .elem_type = .funcref, .limits = .{ .min = max_size, .max = max_size } },
                    .elements = elements,
                };
                tables[0] = tbl;
                owned[0] = true;
                break :blk TablesAllocation{ .tables = tables, .owned = owned };
            };
            inst.tables = fallback.tables;
            inst.tables_owned = fallback.owned;
        }
    }

    const global_alloc = try allocateGlobals(module, imported_global_overrides, allocator);
    inst.globals = global_alloc.globals;
    inst.globals_owned = global_alloc.owned;
    errdefer freeGlobals(inst.globals, inst.globals_owned, allocator);
    const global_layout = try computeGlobalLayout(module.importedGlobals(), module.global_inits, allocator);
    inst.global_offsets = global_layout.offsets;
    inst.global_storage_size = global_layout.size;
    errdefer if (inst.global_offsets.len > 0) allocator.free(inst.global_offsets);

    // #672 commit 1: allocate tag instances. Imported entries borrow when
    // an override is supplied (cross-instance tag identity, future #670);
    // otherwise we synthesize a fresh local instance derived from the
    // declared param arity so the instance is self-consistent even when
    // no exporter has been wired yet.
    const tag_alloc = try allocateTags(module, imported_tag_overrides, allocator);
    inst.tags = tag_alloc.tags;
    inst.tags_owned = tag_alloc.owned;
    errdefer freeTags(inst.tags, inst.tags_owned, allocator);

    // Resolve AOT host functions for imports
    inst.host_functions = try resolveHostFunctionsWithOverrides(module, allocator, imported_function_overrides);

    // Intern each declared module type into the process-global registry and
    // build a module-local sig_table (type_idx → canonical u32 sig_id). Only
    // `.func`-kind entries intern a real signature; struct/array placeholders
    // (serialised as empty params/results in the AOT format) map to 0 which
    // is also the "null sig_id" sentinel — safe because AOT codegen only
    // queries sig_table for `call_indirect` whose operand is a func type.
    if (module.func_types.len > 0) {
        const reg = sig_registry.global();
        const sig_table = allocator.alloc(u32, module.func_types.len) catch return error.OutOfMemory;
        errdefer allocator.free(sig_table);
        for (module.func_types, 0..) |aot_ft, i| {
            if (aot_ft.params.len == 0 and aot_ft.results.len == 0) {
                // Either an empty `() -> ()` func type *or* a non-func
                // placeholder (struct/array). Intern the empty func type —
                // placeholders will share id 1 but are never queried.
            }
            const ft = types.FuncType{ .params = aot_ft.params, .results = aot_ft.results };
            sig_table[i] = reg.intern(&ft) catch return error.OutOfMemory;
        }
        inst.sig_table = sig_table;
    }

    // Build func_sig_ids indexed by module funcidx (imports + locals). Uses
    // `sig_table` above for the canonical ids. Import entries that are not
    // `.function`-kind (memory/global/table imports) are skipped over in the
    // imports loop — only function imports consume a funcidx slot.
    const total_funcs = module.import_function_count + module.func_count;
    if (total_funcs > 0) {
        const fsi = allocator.alloc(u32, total_funcs) catch return error.OutOfMemory;
        errdefer allocator.free(fsi);
        @memset(fsi, 0);
        var slot: u32 = 0;
        for (module.imports) |imp| {
            if (imp.kind != .function) continue;
            if (slot >= module.import_function_count) break;
            if (imp.func_type_idx < inst.sig_table.len) {
                fsi[slot] = inst.sig_table[imp.func_type_idx];
            }
            slot += 1;
        }
        for (0..module.func_count) |li| {
            const tidx: u32 = if (li < module.local_func_type_indices.len)
                module.local_func_type_indices[li]
            else
                0;
            if (tidx < inst.sig_table.len) {
                fsi[module.import_function_count + li] = inst.sig_table[tidx];
            }
        }
        inst.func_sig_ids = fsi;
    }

    refreshVmCtxForInstance(inst, null);
    subscribeVmCtxToMemories(inst) catch return error.OutOfMemory;

    return inst;
}

/// Clone an executing AOT instance for the Preview-1 instance-per-thread
/// model.
///
/// Shared, retained state: linear memories, tables, immutable native code,
/// host-function bindings, and process state. Per-thread state: VmCtx,
/// globals, segment-drop flags, trap/cancel/task bindings, TLS/start_arg, and
/// transient call bookkeeping.
/// The parent owns immutable module/link/tag metadata and must outlive the
/// clone; `ThreadManager` shutdown enforces that for production callers.
fn snapshotGlobalForThread(
    parent: *const AotInstance,
    global: *const types.GlobalInstance,
    index: usize,
) types.Value {
    if (parent.vmctx.globals_ptr == 0) return global.value;
    const offset = globalOffsetAt(parent, index) orelse return global.value;
    const storage = @as(
        [*]const u8,
        @ptrFromInt(parent.vmctx.globals_ptr),
    )[0 .. globalStorageWordCount(parent) * @sizeOf(u128)];
    return switch (global.value) {
        .v128 => if (offset + 16 <= storage.len)
            .{ .v128 = std.mem.readInt(u128, storage[offset..][0..16], .little) }
        else
            global.value,
        else => if (offset + 8 <= storage.len)
            globalValueFromI64(
                parent,
                global.value,
                std.mem.readInt(i64, storage[offset..][0..8], .little),
            )
        else
            global.value,
    };
}

pub fn cloneForThread(
    parent: *const AotInstance,
    allocator: std.mem.Allocator,
) RuntimeError!*AotInstance {
    if (comptime !config.lib_wasi_threads)
        return error.WasiThreadsAotNotImplemented;
    if (comptime config.lazy_jit) return error.WasiThreadsAotNotImplemented;
    if (parent.code_base != null and parent.code_mapping == null)
        return error.CodeMappingFailed;

    const child = allocator.create(AotInstance) catch return error.OutOfMemory;
    errdefer allocator.destroy(child);

    const memories: []*types.MemoryInstance = if (parent.memories.len > 0)
        allocator.alloc(*types.MemoryInstance, parent.memories.len) catch
            return error.OutOfMemory
    else
        &.{};
    var memories_retained: usize = 0;
    errdefer {
        for (memories[0..memories_retained]) |memory| memory.release(allocator);
        if (memories.len > 0) allocator.free(memories);
    }
    for (parent.memories, 0..) |memory, i| {
        memory.retain();
        memories[i] = memory;
        memories_retained += 1;
    }
    const memories_owned: []bool = if (memories.len > 0)
        allocator.alloc(bool, memories.len) catch return error.OutOfMemory
    else
        &.{};
    errdefer if (memories_owned.len > 0) allocator.free(memories_owned);
    @memset(memories_owned, false);

    const tables: []*types.TableInstance = if (parent.tables.len > 0)
        allocator.alloc(*types.TableInstance, parent.tables.len) catch
            return error.OutOfMemory
    else
        &.{};
    var tables_retained: usize = 0;
    errdefer {
        for (tables[0..tables_retained]) |table| table.release(allocator);
        if (tables.len > 0) allocator.free(tables);
    }
    for (parent.tables, 0..) |table, i| {
        table.retain();
        tables[i] = table;
        tables_retained += 1;
    }
    const tables_owned: []bool = if (tables.len > 0)
        allocator.alloc(bool, tables.len) catch return error.OutOfMemory
    else
        &.{};
    errdefer if (tables_owned.len > 0) allocator.free(tables_owned);
    @memset(tables_owned, false);

    const globals: []*types.GlobalInstance = if (parent.globals.len > 0)
        allocator.alloc(*types.GlobalInstance, parent.globals.len) catch
            return error.OutOfMemory
    else
        &.{};
    var globals_created: usize = 0;
    errdefer {
        for (globals[0..globals_created]) |global| global.release(allocator);
        if (globals.len > 0) allocator.free(globals);
    }
    for (parent.globals, 0..) |global, i| {
        const clone = allocator.create(types.GlobalInstance) catch
            return error.OutOfMemory;
        clone.* = .{
            .global_type = global.global_type,
            .value = snapshotGlobalForThread(parent, global, i),
            .owned = global.owned,
            .source_module = global.source_module,
        };
        globals[i] = clone;
        globals_created += 1;
    }
    const globals_owned: []bool = if (globals.len > 0)
        allocator.alloc(bool, globals.len) catch return error.OutOfMemory
    else
        &.{};
    errdefer if (globals_owned.len > 0) allocator.free(globals_owned);
    @memset(globals_owned, true);

    const tags: []*types.TagInstance = if (parent.tags.len > 0)
        allocator.alloc(*types.TagInstance, parent.tags.len) catch
            return error.OutOfMemory
    else
        &.{};
    errdefer if (tags.len > 0) allocator.free(tags);
    if (tags.len > 0) @memcpy(tags, parent.tags);
    const tags_owned: []bool = if (tags.len > 0)
        allocator.alloc(bool, tags.len) catch return error.OutOfMemory
    else
        &.{};
    errdefer if (tags_owned.len > 0) allocator.free(tags_owned);
    @memset(tags_owned, false);

    const global_offsets: []u32 = if (parent.global_offsets.len > 0)
        allocator.dupe(u32, parent.global_offsets) catch return error.OutOfMemory
    else
        &.{};
    errdefer if (global_offsets.len > 0) allocator.free(global_offsets);
    const host_functions: []const ?*const anyopaque = if (parent.host_functions.len > 0)
        allocator.dupe(?*const anyopaque, parent.host_functions) catch
            return error.OutOfMemory
    else
        &.{};
    errdefer if (host_functions.len > 0) allocator.free(host_functions);
    const funcptrs: []usize = if (parent.funcptrs.len > 0)
        allocator.dupe(usize, parent.funcptrs) catch return error.OutOfMemory
    else
        &.{};
    errdefer if (funcptrs.len > 0) allocator.free(funcptrs);
    const elem_segments_dropped: []bool = if (parent.elem_segments_dropped.len > 0)
        allocator.dupe(bool, parent.elem_segments_dropped) catch
            return error.OutOfMemory
    else
        &.{};
    errdefer if (elem_segments_dropped.len > 0)
        allocator.free(elem_segments_dropped);
    const sig_table: []u32 = if (parent.sig_table.len > 0)
        allocator.dupe(u32, parent.sig_table) catch return error.OutOfMemory
    else
        &.{};
    errdefer if (sig_table.len > 0) allocator.free(sig_table);
    const func_sig_ids: []u32 = if (parent.func_sig_ids.len > 0)
        allocator.dupe(u32, parent.func_sig_ids) catch return error.OutOfMemory
    else
        &.{};
    errdefer if (func_sig_ids.len > 0) allocator.free(func_sig_ids);
    const ptr_to_sig: []PtrSigEntry = if (parent.ptr_to_sig.len > 0)
        allocator.dupe(PtrSigEntry, parent.ptr_to_sig) catch
            return error.OutOfMemory
    else
        &.{};
    errdefer if (ptr_to_sig.len > 0) allocator.free(ptr_to_sig);
    const tables_info: []TableInfo = if (parent.tables_info.len > 0)
        allocator.dupe(TableInfo, parent.tables_info) catch return error.OutOfMemory
    else
        &.{};
    errdefer if (tables_info.len > 0) allocator.free(tables_info);
    const extra_tables_storage: [][]usize = if (parent.extra_tables_storage.len > 0)
        allocator.dupe([]usize, parent.extra_tables_storage) catch
            return error.OutOfMemory
    else
        &.{};
    errdefer if (extra_tables_storage.len > 0)
        allocator.free(extra_tables_storage);

    const code_mapping = parent.code_mapping;
    if (code_mapping) |mapping| mapping.retain();
    errdefer if (code_mapping) |mapping| mapping.release();

    var child_thread_context =
        execution_context.ThreadExecutionContext.init(parent.thread_context.process_state);
    errdefer child_thread_context.deinit();

    child.* = .{
        .module = parent.module,
        .memories = memories,
        .memories_owned = memories_owned,
        .tables = tables,
        .tables_owned = tables_owned,
        .globals = globals,
        .globals_owned = globals_owned,
        .tags = tags,
        .tags_owned = tags_owned,
        .global_offsets = global_offsets,
        .global_storage_size = parent.global_storage_size,
        .allocator = allocator,
        .code_base = parent.code_base,
        .code_size = parent.code_size,
        .code_mapping = code_mapping,
        .host_functions = host_functions,
        .func_table = parent.func_table,
        .funcptrs = funcptrs,
        .tables_info = tables_info,
        .extra_tables_storage = extra_tables_storage,
        .elem_segments_dropped = elem_segments_dropped,
        .module_ref = parent.module_ref,
        .sig_table = sig_table,
        .func_sig_ids = func_sig_ids,
        .ptr_to_sig = ptr_to_sig,
        .thread_context = child_thread_context,
    };
    refreshVmCtxForInstance(child, null);
    subscribeVmCtxToMemories(child) catch return error.OutOfMemory;
    errdefer unsubscribeVmCtxFromMemories(child);
    subscribeVmCtxToTables(child) catch return error.OutOfMemory;
    refreshVmCtxTablesForInstance(child);
    return child;
}

/// Destroy an AOT instance, freeing all allocated resources.
pub fn destroy(inst: *AotInstance) void {
    const allocator = inst.allocator;
    if (comptime config.lib_wasi_threads) {
        if (inst.code_mapping) |mapping| mapping.release();
    } else if (inst.code_base) |base| {
        platform.munmap(@ptrCast(@constCast(base)), inst.code_size);
        JitCodeCache.unregister(inst.code_size);
    }
    if (comptime config.lazy_jit) {
        inst.lazy_jit.free(allocator);
    }
    unsubscribeVmCtxFromTables(inst);
    unsubscribeVmCtxFromMemories(inst);
    freeMemories(inst.memories, inst.memories_owned, allocator);
    freeTables(inst.tables, inst.tables_owned, allocator);
    freeGlobals(inst.globals, inst.globals_owned, allocator);
    freeTags(inst.tags, inst.tags_owned, allocator);
    if (inst.global_offsets.len > 0) allocator.free(inst.global_offsets);
    if (inst.host_functions.len > 0) allocator.free(inst.host_functions);
    // inst.func_table aliases tables[0].native_backing (freed by TableInstance.release).
    if (inst.funcptrs.len > 0) allocator.free(inst.funcptrs);
    if (inst.elem_segments_dropped.len > 0) allocator.free(inst.elem_segments_dropped);
    if (inst.sig_table.len > 0) allocator.free(inst.sig_table);
    if (inst.func_sig_ids.len > 0) allocator.free(inst.func_sig_ids);
    if (inst.ptr_to_sig.len > 0) allocator.free(inst.ptr_to_sig);
    if (inst.tables_info.len > 0) allocator.free(inst.tables_info);
    if (inst.extra_tables_storage.len > 0) allocator.free(inst.extra_tables_storage);
    inst.thread_context.deinit();
    allocator.destroy(inst);
}

fn refreshVmCtxMemory(vmctx: *VmCtx, mem: *types.MemoryInstance) void {
    vmctx.memory_base = @intFromPtr(mem.data.ptr);
    vmctx.memory_size = mem.byteLen();
    vmctx.memory_max_size = if (mem.shared_control) |control| control.reserved_bytes else mem.data.len;
    vmctx.memory_pages = mem.pageCount();
    // #719 forensic aid: when the trap-OOB dump env var is set, also log
    // the host base addr for every memory we attach so a gdb hardware
    // watchpoint can be placed on `mem_base + wasm_offset` cheaply.
    if (g_trap_oob_dump_env != null) {
        std.debug.print(
            "[mem-base] host_base=0x{x} wasm_size=0x{x} max=0x{x}\n",
            .{ vmctx.memory_base, vmctx.memory_size, vmctx.memory_max_size },
        );
    }
    // Keep the WAMR_WATCH_ADDR poller pointed at the current mapping;
    // memory.grow can move us via mremap so this must run every refresh.
    watchAddrNoteMemory(vmctx.memory_base, vmctx.memory_size);
}

fn refreshVmCtxForInstance(inst: *AotInstance, globals_buf: ?[]u8) void {
    const vmctx = &inst.vmctx;
    if (inst.memories.len > 0) {
        refreshVmCtxMemory(vmctx, inst.memories[0]);
    } else {
        vmctx.memory_base = 0;
        vmctx.memory_size = 0;
        vmctx.memory_max_size = 0;
        vmctx.memory_pages = 0;
    }

    vmctx.globals_ptr = if (globals_buf) |buf| @intFromPtr(buf.ptr) else 0;
    vmctx.globals_count = @intCast(inst.globals.len);
    if (inst.host_functions.len > 0) {
        vmctx.host_functions_ptr = @intFromPtr(inst.host_functions.ptr);
        vmctx.host_functions_count = @intCast(inst.host_functions.len);
    } else {
        vmctx.host_functions_ptr = 0;
        vmctx.host_functions_count = 0;
    }
    if (inst.func_table.len > 0) {
        vmctx.func_table_ptr = @intFromPtr(inst.func_table.ptr);
        vmctx.func_table_len = @intCast(inst.func_table.len);
    } else {
        vmctx.func_table_ptr = 0;
        vmctx.func_table_len = 0;
    }
    vmctx.funcptrs_ptr = if (inst.funcptrs.len > 0) @intFromPtr(inst.funcptrs.ptr) else 0;
    vmctx.instance_ptr = @intFromPtr(inst);
    vmctx.mem_grow_fn = @intFromPtr(&memGrowHelper);
    vmctx.mem_fill_fn = @intFromPtr(&memFillHelper);
    vmctx.mem_copy_fn = @intFromPtr(&memCopyHelper);
    vmctx.trap_oob_fn = @intFromPtr(&aotTrapOOB);
    vmctx.trap_unreachable_fn = @intFromPtr(&aotTrapUnreachable);
    vmctx.trap_idivz_fn = @intFromPtr(&aotTrapIntDivZero);
    vmctx.trap_iovf_fn = @intFromPtr(&aotTrapIntOverflow);
    vmctx.trap_ivc_fn = @intFromPtr(&aotTrapInvalidConversion);
    vmctx.aot_throw_uncaught_fn = @intFromPtr(&aotThrowUncaught);
    if (inst.tags.len > 0) {
        vmctx.tags_ptr = @intFromPtr(inst.tags.ptr);
        vmctx.tags_count = @intCast(inst.tags.len);
    } else {
        vmctx.tags_ptr = 0;
        vmctx.tags_count = 0;
    }
    vmctx.table_grow_fn = @intFromPtr(&tableGrowHelper);
    vmctx.tables_info_ptr = if (inst.tables_info.len > 0) @intFromPtr(inst.tables_info.ptr) else 0;
    vmctx.table_init_fn = @intFromPtr(&tableInitHelper);
    vmctx.elem_drop_fn = @intFromPtr(&elemDropHelper);
    vmctx.table_set_fn = @intFromPtr(&tableSetHelper);
    vmctx.futex_wait32_fn = @intFromPtr(&aotAtomicWait32);
    vmctx.futex_wait64_fn = @intFromPtr(&aotAtomicWait64);
    vmctx.futex_notify_fn = @intFromPtr(&aotAtomicNotify);
    vmctx.sig_table_ptr = if (inst.sig_table.len > 0) @intFromPtr(inst.sig_table.ptr) else 0;
    vmctx.func_sig_ids_ptr = if (inst.func_sig_ids.len > 0) @intFromPtr(inst.func_sig_ids.ptr) else 0;
    if (inst.ptr_to_sig.len > 0) {
        vmctx.ptr_to_sig_ptr = @intFromPtr(inst.ptr_to_sig.ptr);
        vmctx.ptr_to_sig_len = @intCast(inst.ptr_to_sig.len);
    } else {
        vmctx.ptr_to_sig_ptr = 0;
        vmctx.ptr_to_sig_len = 0;
    }
    vmctx.wasi_ctx = if (inst.thread_context.process_state) |state|
        @intFromPtr(state.ptr)
    else
        0;
    vmctx.lazy_compile_fn = if (comptime config.lazy_jit) @intFromPtr(&lazyCompileHelper) else 0;
    vmctx.thread_context = @intFromPtr(&inst.thread_context);
}

fn subscribeVmCtxToMemories(inst: *AotInstance) !void {
    // Subscribe owned and borrowed slots: if an importer grows a borrowed
    // memory, the exporter's active vmctx must be refreshed too.
    const vmctx_opaque: *anyopaque = @ptrCast(&inst.vmctx);
    var subscribed: usize = 0;
    errdefer {
        for (inst.memories[0..subscribed]) |mem| mem.unsubscribeVmCtx(vmctx_opaque);
    }
    for (inst.memories) |mem| {
        try mem.subscribeVmCtx(vmctx_opaque, inst.allocator);
        subscribed += 1;
    }
}

fn unsubscribeVmCtxFromMemories(inst: *AotInstance) void {
    const vmctx_opaque: *anyopaque = @ptrCast(&inst.vmctx);
    for (inst.memories) |mem| mem.unsubscribeVmCtx(vmctx_opaque);
}

fn subscribeVmCtxToTables(inst: *AotInstance) !void {
    if (comptime !config.lib_wasi_threads) return;
    const vmctx_opaque: *anyopaque = @ptrCast(&inst.vmctx);
    var subscribed: usize = 0;
    errdefer {
        for (inst.tables[0..subscribed]) |table|
            table.unsubscribeVmCtx(vmctx_opaque);
    }
    for (inst.tables) |table| {
        try table.subscribeVmCtx(vmctx_opaque, inst.allocator);
        subscribed += 1;
    }
}

fn unsubscribeVmCtxFromTables(inst: *AotInstance) void {
    if (comptime !config.lib_wasi_threads) return;
    const vmctx_opaque: *anyopaque = @ptrCast(&inst.vmctx);
    for (inst.tables) |table| table.unsubscribeVmCtx(vmctx_opaque);
}

fn refreshVmCtxTable(
    inst: *AotInstance,
    table_idx: usize,
    table: *types.TableInstance,
) void {
    const backing = table.native_backing;
    const type_backing_ptr = if (table.type_backing.len > 0)
        @intFromPtr(table.type_backing.ptr)
    else
        0;
    if (table_idx == 0) {
        inst.func_table = backing;
        inst.vmctx.func_table_ptr = if (backing.len > 0)
            @intFromPtr(backing.ptr)
        else
            0;
        inst.vmctx.func_table_len = @intCast(backing.len);
    } else if (table_idx - 1 < inst.extra_tables_storage.len) {
        inst.extra_tables_storage[table_idx - 1] = backing;
    }
    if (table_idx < inst.tables_info.len) {
        inst.tables_info[table_idx] = .{
            .ptr = if (backing.len > 0) @intFromPtr(backing.ptr) else 0,
            .len = @intCast(backing.len),
            .type_backing_ptr = type_backing_ptr,
        };
    }
}

fn refreshVmCtxTablesForInstance(inst: *AotInstance) void {
    for (inst.tables, 0..) |table, table_idx| {
        table.lock();
        refreshVmCtxTable(inst, table_idx, table);
        table.unlock();
    }
}

fn refreshTableSubscribers(table: *types.TableInstance) void {
    if (comptime !config.lib_wasi_threads) return;
    table.subscriber_mutex.lock();
    defer table.subscriber_mutex.unlock();
    for (table.vmctx_subscribers.items) |subscriber_opaque| {
        const vmctx: *VmCtx = @ptrCast(@alignCast(subscriber_opaque));
        if (vmctx.instance_ptr == 0) continue;
        const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
        for (inst.tables, 0..) |candidate, table_idx| {
            if (candidate == table) refreshVmCtxTable(inst, table_idx, table);
        }
    }
}

/// Look up an exported function by name, returning its function index.
pub fn findExportFunc(inst: *const AotInstance, name: []const u8) ?u32 {
    for (inst.module.exports) |exp| {
        if (exp.kind == .function and std.mem.eql(u8, exp.name, name)) return exp.index;
    }
    return null;
}

fn functionTypeForIndex(
    module: *const aot_loader.AotModule,
    func_idx: u32,
) ?*const aot_loader.AotFuncType {
    if (func_idx < module.import_function_count) {
        var imported_idx: u32 = 0;
        for (module.imports) |*imp| {
            if (imp.kind != .function) continue;
            if (imported_idx == func_idx) {
                if (imp.func_type_idx >= module.func_types.len) return null;
                return &module.func_types[imp.func_type_idx];
            }
            imported_idx += 1;
        }
        return null;
    }
    const local_idx = func_idx - module.import_function_count;
    if (local_idx >= module.local_func_type_indices.len) return null;
    const type_idx = module.local_func_type_indices[local_idx];
    if (type_idx >= module.func_types.len) return null;
    return &module.func_types[type_idx];
}

fn isWasiThreadStartType(func_type: *const aot_loader.AotFuncType) bool {
    return func_type.params.len == 2 and
        func_type.params[0] == .i32 and
        func_type.params[1] == .i32 and
        func_type.results.len == 0;
}

const AotThreadContext = struct {
    instance: *AotInstance,
    func_idx: u32,
    allocator: std.mem.Allocator,
};

fn createAotThreadContext(
    parent_opaque: *anyopaque,
    allocator: std.mem.Allocator,
) thread_manager.SpawnError!*anyopaque {
    const parent: *AotInstance = @ptrCast(@alignCast(parent_opaque));
    const func_idx = findExportFunc(parent, "wasi_thread_start") orelse
        return error.MissingThreadStart;
    const func_type = functionTypeForIndex(parent.module, func_idx) orelse
        return error.InvalidThreadStartSignature;
    if (!isWasiThreadStartType(func_type))
        return error.InvalidThreadStartSignature;

    const child = cloneForThread(parent, parent.allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.ChildInitializationFailed,
    };
    errdefer destroy(child);
    const source_context = execution_context.current() orelse
        &parent.thread_context;
    child.thread_context.replaceProcessState(source_context.process_state);
    const context = allocator.create(AotThreadContext) catch
        return error.OutOfMemory;
    context.* = .{
        .instance = child,
        .func_idx = func_idx,
        .allocator = allocator,
    };
    return @ptrCast(context);
}

fn configureAotThreadContext(
    child_opaque: *anyopaque,
    manager: *thread_manager.ThreadManager,
    tid: i32,
    start_arg: u32,
    auxiliary_stack: ?execution_context.AuxiliaryStack,
) thread_manager.SpawnError!void {
    const child: *AotThreadContext = @ptrCast(@alignCast(child_opaque));
    if (auxiliary_stack) |stack| {
        if (child.instance.module.findExport("__stack_pointer", .global)) |exp| {
            if (exp.index < child.instance.globals.len) {
                child.instance.globals[exp.index].value = .{
                    .i32 = @bitCast(stack.top),
                };
            }
        }
    }
    child.instance.setThreadManager(manager);
    child.instance.thread_context.configureWasiThread(
        tid,
        start_arg,
        auxiliary_stack,
    );
    refreshVmCtxForInstance(child.instance, null);
}

fn runAotThreadContext(child_opaque: *anyopaque) thread_manager.ThreadOutcome {
    const child: *AotThreadContext = @ptrCast(@alignCast(child_opaque));
    const params = [_]types.ValType{ .i32, .i32 };
    const args = [_]types.Value{
        .{ .i32 = child.instance.thread_context.tid },
        .{ .i32 = @bitCast(child.instance.thread_context.start_arg) },
    };
    var results: [0]ScalarResult = .{};
    _ = callFuncScalar(
        child.instance,
        child.func_idx,
        &params,
        &.{},
        &args,
        &results,
    ) catch return .trapped;
    return .completed;
}

fn destroyAotThreadContext(child_opaque: *anyopaque) void {
    const child: *AotThreadContext = @ptrCast(@alignCast(child_opaque));
    const allocator = child.allocator;
    destroy(child.instance);
    allocator.destroy(child);
}

const aot_thread_ops = thread_manager.ThreadBackendOps{
    .create = createAotThreadContext,
    .configure = configureAotThreadContext,
    .run = runAotThreadContext,
    .destroy = destroyAotThreadContext,
    .uses_auxiliary_stack = true,
};

pub fn usesWasiThreads(module: *const aot_loader.AotModule) bool {
    for (module.imports) |imp| {
        if (imp.kind == .function and
            config.threads_feature.isThreadSpawnImport(
                imp.module_name,
                imp.field_name,
            ))
            return true;
    }
    return false;
}

pub fn prepareWasiThreads(
    inst: *AotInstance,
    manager: *thread_manager.ThreadManager,
) thread_manager.PrepareError!void {
    if (inst.memories.len == 0) return error.SharedMemoryRequired;
    const heap_global: ?*types.GlobalInstance =
        if (inst.module.findExport("__heap_base", .global)) |exp| blk: {
            if (exp.index >= inst.globals.len) return error.InvalidHeapBase;
            break :blk inst.globals[exp.index];
        } else null;
    try manager.prepareSharedMemory(inst.memories[0], heap_global);
    inst.setThreadManager(manager);
}

/// Spawn `wasi_thread_start(tid, start_arg)` on a manager-owned native
/// thread. The host import maps every lifecycle error to the Preview-1
/// negative-TID failure result.
pub fn spawnWasiThread(parent: *AotInstance, start_arg: i32) thread_manager.SpawnError!i32 {
    if (comptime !config.lib_wasi_threads or !can_execute_native)
        return error.ThreadFeatureDisabled;
    const source_context = execution_context.current() orelse
        &parent.thread_context;
    const manager = source_context.threadGroup(thread_manager.ThreadManager) orelse
        return error.ThreadFeatureDisabled;
    return manager.spawnWithBackend(
        @ptrCast(parent),
        start_arg,
        &aot_thread_ops,
    );
}

pub fn signalThreadGroupTrap(vmctx: *VmCtx) void {
    if (vmctx.thread_context == 0) return;
    const thread_context_ptr: *execution_context.ThreadExecutionContext =
        @ptrFromInt(vmctx.thread_context);
    thread_context_ptr.markTrap();
    if (thread_context_ptr.threadGroup(thread_manager.ThreadManager)) |manager| {
        manager.signalTrap();
    }
}

/// Look up an exported memory by name, returning the underlying
/// `*MemoryInstance` from the AOT instance's `memories` array.
/// Used by `ComponentInstance.resolveTopLevelMemory` to follow an
/// `alias core export "name"` decl through an AOT sibling core
/// instance for canon-lower(aot) retptr stores. Issue #707.
pub fn findExportMemory(inst: *const AotInstance, name: []const u8) ?*types.MemoryInstance {
    const exp = inst.module.findExport(name, .memory) orelse return null;
    if (exp.index >= inst.memories.len) return null;
    return inst.memories[exp.index];
}

/// Get the native code pointer for a function by module-level index.
/// Import functions (func_idx < import_function_count) have no native code
/// and return null.  Local functions are looked up in func_offsets after
/// subtracting the import count.
pub fn getFuncAddr(inst: *const AotInstance, func_idx: u32) ?[*]const u8 {
    const module = inst.module;
    const import_count = module.import_function_count;

    // Import functions don't have native code
    if (func_idx < import_count) return null;

    const local_idx = func_idx - import_count;
    if (local_idx >= module.func_count) return null;
    const offset = module.func_offsets[local_idx];
    // Prefer the executable mapping.
    if (inst.code_base) |base| {
        if (offset >= inst.code_size) return null;
        return base + offset;
    }
    // Fall back to raw text section (read-only, not executable).
    const text = module.text_section orelse return null;
    if (offset >= text.len) return null;
    return text.ptr + offset;
}

fn getCallableAddr(inst: *const AotInstance, func_idx: u32) ?[*]const u8 {
    if (func_idx < inst.funcptrs.len and inst.funcptrs[func_idx] != 0) {
        return @ptrFromInt(inst.funcptrs[func_idx]);
    }
    if (inst.code_base == null) return null;
    return getFuncAddr(inst, func_idx);
}

// ─── Native execution ───────────────────────────────────────────────────────

/// Map arbitrary machine-code bytes into executable memory and register
/// the resulting region with `JitCodeCache`, so budget enforcement and
/// resident-byte accounting cover both eager module text blobs and
/// deferred lazy-JIT function bodies.
pub fn mapTrackedExecutableCode(code: []const u8) RuntimeError!LazyCompiledFunc {
    try JitCodeCache.checkBudget(code.len);

    // #858: `platform.mapExecutableCode` owns the W^X mapping strategy
    // (plain RW→RX `mprotect` on most targets; `MAP_JIT` +
    // per-thread `pthread_jit_write_protect_np` toggling on macOS
    // aarch64, where a post-hoc `mprotect` can't re-grant exec on a
    // MAP_JIT region) and flushes the instruction cache internally, so
    // callers no longer need an arch-specific `icacheFlush` branch of
    // their own.
    const mapped = platform.mapExecutableCode(code) orelse return error.CodeMappingFailed;
    JitCodeCache.register(code.len);
    return .{ .addr = mapped, .size = code.len };
}

/// Map the module's native code into executable memory.
/// After this call, `getFuncAddr` returns pointers suitable for execution.
pub fn mapCodeExecutable(inst: *AotInstance) RuntimeError!void {
    const module = inst.module;
    const text_opt = module.text_section;
    const has_text = text_opt != null and text_opt.?.len > 0;

    if (has_text) {
        const text = text_opt.?;
        const mapped = try mapTrackedExecutableCode(text);
        inst.code_base = mapped.addr;
        inst.code_size = mapped.size;
        if (comptime config.lib_wasi_threads) {
            const mapping = inst.allocator.create(SharedCodeMapping) catch {
                platform.munmap(@ptrCast(@constCast(mapped.addr)), mapped.size);
                JitCodeCache.unregister(mapped.size);
                inst.code_base = null;
                inst.code_size = 0;
                return error.OutOfMemory;
            };
            mapping.* = .{
                .addr = mapped.addr,
                .size = mapped.size,
                .allocator = inst.allocator,
            };
            inst.code_mapping = mapping;
        }
    }

    // Build function pointer table for call_indirect
    const import_count = module.import_function_count;

    // Build module func idx → native address mapping (temporary).
    //
    // #694: must be sized to `total_funcs`, NOT a fixed 256-entry stack
    // buffer. With a fixed cap, any module whose import_count +
    // func_count exceeds the cap silently dropped both the funcptr
    // and the type_backing update for high-funcidx active elem
    // segments, causing call_indirect through those slots to fail the
    // sig-id type check and trap on `unreachable`.
    const total_funcs = import_count + module.func_count;
    const func_addrs: []usize = if (total_funcs > 0)
        inst.allocator.alloc(usize, total_funcs) catch return error.OutOfMemory
    else
        &.{};
    defer if (func_addrs.len > 0) inst.allocator.free(func_addrs);
    @memset(func_addrs, 0);
    const n_addrs = func_addrs.len;
    // Import functions → host function pointers
    for (0..@min(import_count, @min(inst.host_functions.len, n_addrs))) |i| {
        func_addrs[i] = if (inst.host_functions[i]) |ptr| @intFromPtr(ptr) else 0;
    }
    // Local functions → either their eager code address (or #887 stub, both
    // stored via `code_base` + offset) or, for still-pending trampoline-
    // mechanism lazy locals, the stable trampoline stub published by
    // `setupLazyJit()`.
    const local_slots = n_addrs - @min(import_count, n_addrs);
    for (0..@min(module.func_count, local_slots)) |i| {
        if (comptime config.lazy_jit) {
            if (i < inst.lazy_jit.trampolines.len and inst.lazy_jit.trampolines[i] != 0) {
                func_addrs[import_count + i] = inst.lazy_jit.trampolines[i];
                continue;
            }
        }
        if (has_text) {
            const code_base = inst.code_base orelse unreachable;
            const offset = module.func_offsets[i];
            func_addrs[import_count + i] = @intFromPtr(code_base) + offset;
        }
    }

    // Persist the funcidx → native address map on the instance for ref.func.
    if (n_addrs > 0) {
        const persistent = inst.allocator.alloc(usize, n_addrs) catch return error.OutOfMemory;
        @memcpy(persistent, func_addrs[0..n_addrs]);
        inst.funcptrs = persistent;
    }

    // Build sorted ptr→sig_id map. Writer sites that receive a raw funcptr
    // (e.g. table.set with a funcref value produced by ref.func or by
    // reading another table) look up the matching sig_id here with binary
    // search. Zero-valued funcptrs (e.g. unresolved host imports, padding)
    // are intentionally skipped — a stored zero already means "null".
    if (n_addrs > 0 and inst.func_sig_ids.len > 0) {
        var n_entries: usize = 0;
        for (inst.funcptrs) |p| {
            if (p != 0) n_entries += 1;
        }
        if (n_entries > 0) {
            const arr = inst.allocator.alloc(PtrSigEntry, n_entries) catch return error.OutOfMemory;
            var j: usize = 0;
            for (inst.funcptrs, 0..) |p, fi| {
                if (p == 0) continue;
                const sid: u32 = if (fi < inst.func_sig_ids.len) inst.func_sig_ids[fi] else 0;
                arr[j] = .{ .ptr = @as(u64, p), .sig_id = sid };
                j += 1;
            }
            std.mem.sort(PtrSigEntry, arr, {}, struct {
                fn lessThan(_: void, a: PtrSigEntry, b: PtrSigEntry) bool {
                    return a.ptr < b.ptr;
                }
            }.lessThan);
            inst.ptr_to_sig = arr;
        }
    }

    // Build wasm table → native address table for call_indirect.
    //
    // Native backings live on the shared `TableInstance` so that multiple
    // modules importing the same table read/write the same slice. The
    // exporter allocates + publishes; importers alias. Active elem segments
    // from the importing module are still applied on top (using this
    // module's funcptrs), mutating the shared backing directly so the
    // exporter's compiled call_indirect sees the writes.
    if (inst.tables.len > 0) {
        const tbl = inst.tables[0];
        tbl.lock();
        defer tbl.unlock();
        const tbl_size = tbl.elements.len;
        if (tbl_size > 0) {
            var native_table: []usize = undefined;
            if (tbl.native_backing.len == tbl_size) {
                // Exporter already published a backing; alias it.
                native_table = tbl.native_backing;
            } else {
                native_table = inst.allocator.alloc(usize, tbl_size) catch return error.OutOfMemory;
                @memset(native_table, 0);
                tbl.native_backing = native_table;
            }
            // Size type_backing in lockstep (zero-init = null sig_id).
            // Filled in by later patches as writer sites start mirroring
            // sig_ids alongside native pointers.
            if (tbl.type_backing.len != tbl_size) {
                const tb = inst.allocator.alloc(u32, tbl_size) catch return error.OutOfMemory;
                @memset(tb, 0);
                if (tbl.type_backing.len > 0) inst.allocator.free(tbl.type_backing);
                tbl.type_backing = tb;
            }

            // Apply this module's active element segments (skip passive —
            // only usable by table.init). `0xFFFFFFFF` encodes a null
            // element (emitted by the compiler when the source segment
            // contained `ref.null` or an externref literal we can't resolve
            // statically); explicitly zero that slot so an importer's
            // `(elem (i32.const k) externref (ref.null extern))` can
            // overwrite an exporter's previously-written value.
            for (module.elem_segments) |seg| {
                if (seg.is_passive) continue;
                if (seg.table_idx != 0) continue;
                // Wasm v2: if the segment extends past the table, skip it
                // entirely (all-or-nothing). Only prior segments persist.
                const seg_end = @as(u64, seg.offset) + @as(u64, seg.func_indices.len);
                if (seg_end > tbl_size) continue;
                for (seg.func_indices, 0..) |func_idx, j| {
                    const dst = seg.offset + @as(u32, @intCast(j));
                    if (func_idx == std.math.maxInt(u32)) {
                        native_table[dst] = 0;
                        if (dst < tbl.type_backing.len) tbl.type_backing[dst] = 0;
                    } else if (func_idx < n_addrs) {
                        native_table[dst] = func_addrs[func_idx];
                        if (dst < tbl.type_backing.len and func_idx < inst.func_sig_ids.len) {
                            tbl.type_backing[dst] = inst.func_sig_ids[func_idx];
                        }
                    }
                }
            }

            inst.func_table = native_table;
        }
    }

    // Build per-table native descriptor array for multi-table support.
    // Slot 0 aliases `inst.func_table`. Additional slots alias their
    // `TableInstance.native_backing` (allocating on first use).
    if (inst.tables.len > 0) {
        const info = inst.allocator.alloc(TableInfo, inst.tables.len) catch return error.OutOfMemory;
        @memset(info, .{});
        const extra = inst.allocator.alloc([]usize, if (inst.tables.len > 1) inst.tables.len - 1 else 0) catch return error.OutOfMemory;
        for (extra) |*e| e.* = &.{};

        // Slot 0: alias inst.func_table.
        inst.tables[0].lock();
        info[0] = .{
            .ptr = @intFromPtr(inst.func_table.ptr),
            .len = @intCast(inst.func_table.len),
            .type_backing_ptr = if (inst.tables.len > 0 and inst.tables[0].type_backing.len > 0)
                @intFromPtr(inst.tables[0].type_backing.ptr)
            else
                0,
        };
        inst.tables[0].unlock();

        // Slots 1..n.
        for (inst.tables[1..], 1..) |tbl_i, idx| {
            {
                tbl_i.lock();
                defer tbl_i.unlock();
                const sz = tbl_i.elements.len;
                if (sz == 0) continue;
                var backing: []usize = undefined;
                if (tbl_i.native_backing.len == sz) {
                    backing = tbl_i.native_backing;
                } else {
                    backing = inst.allocator.alloc(usize, sz) catch return error.OutOfMemory;
                    @memset(backing, 0);
                    tbl_i.native_backing = backing;
                }
                if (tbl_i.type_backing.len != sz) {
                    const tb = inst.allocator.alloc(u32, sz) catch return error.OutOfMemory;
                    @memset(tb, 0);
                    if (tbl_i.type_backing.len > 0) inst.allocator.free(tbl_i.type_backing);
                    tbl_i.type_backing = tb;
                }
                for (module.elem_segments) |seg| {
                    if (seg.is_passive) continue;
                    if (seg.table_idx != idx) continue;
                    const seg_end = @as(u64, seg.offset) + @as(u64, seg.func_indices.len);
                    if (seg_end > sz) continue;
                    for (seg.func_indices, 0..) |func_idx, j| {
                        const dst = seg.offset + @as(u32, @intCast(j));
                        if (func_idx == std.math.maxInt(u32)) {
                            backing[dst] = 0;
                            if (dst < tbl_i.type_backing.len) tbl_i.type_backing[dst] = 0;
                        } else if (func_idx < n_addrs) {
                            backing[dst] = func_addrs[func_idx];
                            if (dst < tbl_i.type_backing.len and func_idx < inst.func_sig_ids.len) {
                                tbl_i.type_backing[dst] = inst.func_sig_ids[func_idx];
                            }
                        }
                    }
                }
                extra[idx - 1] = backing;
                info[idx] = .{
                    .ptr = @intFromPtr(backing.ptr),
                    .len = @intCast(sz),
                    .type_backing_ptr = if (tbl_i.type_backing.len > 0) @intFromPtr(tbl_i.type_backing.ptr) else 0,
                };
            }
        }

        inst.tables_info = info;
        inst.extra_tables_storage = extra;
    }

    try subscribeVmCtxToTables(inst);
}

/// Call an AOT-compiled function by index.
/// The code must have been mapped via `mapCodeExecutable` first.
///
/// Uses comptime to select the correct function pointer type based on `Result`.
pub fn callFunc(inst: *AotInstance, func_idx: u32, comptime Result: type) RuntimeError!Result {
    if (comptime !can_execute_native) return error.UnsupportedArchitecture;
    const call_thread_context = execution_context.current() orelse &inst.thread_context;
    var execution_scope = call_thread_context.enter();
    defer execution_scope.deinit();

    // Lazy-JIT: resolve through the per-slot atomic state machine first (see
    // `callFuncScalar`'s matching logic and `LazyJitState.resolveLocalAddr`'s
    // doc comment) before falling back to the instance's already-published
    // callable-pointer table.
    var lazy_resolved_addr: ?[*]const u8 = null;
    if (comptime config.lazy_jit) {
        const import_count = inst.module.import_function_count;
        if (func_idx >= import_count) {
            const local_idx: usize = @intCast(func_idx - import_count);
            lazy_resolved_addr = try inst.lazy_jit.resolveLocalAddr(local_idx);
            if (lazy_resolved_addr) |addr| {
                if (func_idx < inst.funcptrs.len) inst.funcptrs[func_idx] = @intFromPtr(addr);
            }
        }
    }

    const addr = lazy_resolved_addr orelse (getCallableAddr(inst, func_idx) orelse {
        if (inst.code_base == null) return error.CodeMappingFailed;
        return error.FunctionNotFound;
    });

    const previous_globals_ptr = inst.vmctx.globals_ptr;
    const vmctx_storage_addr = @intFromPtr(&inst.vmctx);
    // Host imports may re-enter the same AOT instance (notably cabi_realloc).
    // Reuse the active globals slab so mutable globals are not forked.
    const reuse_globals = previous_globals_ptr != 0;
    const globals_word_count = globalStorageWordCount(inst);
    var globals_words: []u128 = &.{};
    if (!reuse_globals) {
        globals_words = inst.allocator.alloc(u128, globals_word_count) catch return error.OutOfMemory;
    }
    defer if (!reuse_globals) inst.allocator.free(globals_words);
    const globals_buf: []u8 = if (reuse_globals)
        @as([*]u8, @ptrFromInt(previous_globals_ptr))[0 .. globals_word_count * @sizeOf(u128)]
    else
        std.mem.sliceAsBytes(globals_words);
    if (!reuse_globals) writeGlobalsToStorage(inst, globals_buf);

    // Always provide a valid globals pointer — compiled code may access globals
    // even if none are explicitly initialized (they default to zero).
    refreshVmCtxForInstance(inst, globals_buf);
    const vmctx: *VmCtx = @ptrFromInt(vmctx_storage_addr);
    defer @as(*VmCtx, @ptrFromInt(vmctx_storage_addr)).globals_ptr = previous_globals_ptr;
    const default_thread_context = vmctx.thread_context;
    const default_wasi_ctx = vmctx.wasi_ctx;
    vmctx.thread_context = @intFromPtr(call_thread_context);
    if (call_thread_context.process_state) |state| {
        vmctx.wasi_ctx = @intFromPtr(state.ptr);
    }
    defer {
        vmctx.thread_context = default_thread_context;
        vmctx.wasi_ctx = default_wasi_ctx;
    }

    // AOT-compiled functions receive a VmCtx pointer as hidden first parameter.
    const FnPtr = *const fn (*VmCtx) callconv(.c) Result;
    const func_ptr: FnPtr = @ptrCast(@alignCast(addr));
    var aot_call_state = AotCallState{};
    var backend_scope = call_thread_context.bindBackendContext(@ptrCast(&aot_call_state));
    defer backend_scope.deinit();
    installTrapDecodeFrameFor(inst);
    if (comptime windows_trap_supported) {
        aot_call_state.trap_decode.mem_base = vmctx.memory_base;
        aot_call_state.trap_decode.mem_size = vmctx.memory_size;
        ensureVehInstalled();
    }
    const result = func_ptr(vmctx);

    // Sync globals back from flat storage to GlobalInstance objects.
    const vmctx_after: *VmCtx = @ptrFromInt(vmctx_storage_addr);
    const inst_after: *AotInstance = @ptrFromInt(vmctx_after.instance_ptr);
    const globals_buf_after: []u8 = if (vmctx_after.globals_ptr != 0)
        @as([*]u8, @ptrFromInt(vmctx_after.globals_ptr))[0 .. globals_word_count * @sizeOf(u128)]
    else
        globals_buf;
    readGlobalsFromStorage(inst_after, globals_buf_after);

    return result;
}

// ─── Typed scalar call ──────────────────────────────────────────────────────

/// Scalar result value from a typed AOT call.
pub const ScalarResult = union(enum) {
    void,
    i32: i32,
    i64: i64,
    f32: f32,
    f64: f64,
    funcref: ?u64,
    externref: ?u64,
};

/// Errors callFuncScalar may return beyond the standard RuntimeError set.
pub const ScalarCallError = error{
    UnsupportedSignature,
    InvalidArgType,
    ArgCountMismatch,
} || RuntimeError;

/// ABI note: AOT-compiled functions expect all wasm scalar params (i32/i64/
/// f32/f64) to be passed in integer registers as raw bit patterns (see
/// `param_regs` in src/compiler/codegen/x86_64/compile.zig). This does NOT
/// match the System V / Win64 C ABI for floats, so we call through a
/// function pointer typed with integer args of matching width. The return
/// value always lands in RAX, so the "result" register is also treated as
/// a u64 bit pattern and reinterpreted by the caller.
///
/// Register and stack args follow the platform C ABI after the leading
/// `VmCtx*`; this typed bridge supports up to 16 scalar wasm params on both
/// platforms. Anything wider → `error.UnsupportedSignature`.
const CallFn0 = *const fn (*VmCtx) callconv(.c) u64;
const CallFn1 = *const fn (*VmCtx, u64) callconv(.c) u64;
const CallFn2 = *const fn (*VmCtx, u64, u64) callconv(.c) u64;
const CallFn3 = *const fn (*VmCtx, u64, u64, u64) callconv(.c) u64;
const CallFn4 = *const fn (*VmCtx, u64, u64, u64, u64) callconv(.c) u64;
const CallFn5 = *const fn (*VmCtx, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn6 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn7 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn8 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn9 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn10 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn11 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn12 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn13 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn14 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn15 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const CallFn16 = *const fn (*VmCtx, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) callconv(.c) u64;
const MaxScalarArgs: usize = 16;
/// Max number of results a multi-value return may produce via the
/// `callFuncScalar` path. Bounded by the HRP stack buffer size below.
pub const MaxScalarResults: usize = 16;

fn isScalarValType(t: types.ValType) bool {
    return switch (t) {
        .i32, .i64, .f32, .f64, .funcref, .externref => true,
        else => false,
    };
}

fn invokeScalarCallable(addr: [*]const u8, vmctx: *VmCtx, raw_args: []const u64) u64 {
    return switch (raw_args.len) {
        0 => blk: {
            const f: CallFn0 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx);
        },
        1 => blk: {
            const f: CallFn1 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0]);
        },
        2 => blk: {
            const f: CallFn2 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1]);
        },
        3 => blk: {
            const f: CallFn3 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2]);
        },
        4 => blk: {
            const f: CallFn4 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3]);
        },
        5 => blk: {
            const f: CallFn5 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4]);
        },
        6 => blk: {
            const f: CallFn6 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5]);
        },
        7 => blk: {
            const f: CallFn7 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6]);
        },
        8 => blk: {
            const f: CallFn8 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6], raw_args[7]);
        },
        9 => blk: {
            const f: CallFn9 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6], raw_args[7], raw_args[8]);
        },
        10 => blk: {
            const f: CallFn10 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6], raw_args[7], raw_args[8], raw_args[9]);
        },
        11 => blk: {
            const f: CallFn11 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6], raw_args[7], raw_args[8], raw_args[9], raw_args[10]);
        },
        12 => blk: {
            const f: CallFn12 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6], raw_args[7], raw_args[8], raw_args[9], raw_args[10], raw_args[11]);
        },
        13 => blk: {
            const f: CallFn13 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6], raw_args[7], raw_args[8], raw_args[9], raw_args[10], raw_args[11], raw_args[12]);
        },
        14 => blk: {
            const f: CallFn14 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6], raw_args[7], raw_args[8], raw_args[9], raw_args[10], raw_args[11], raw_args[12], raw_args[13]);
        },
        15 => blk: {
            const f: CallFn15 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6], raw_args[7], raw_args[8], raw_args[9], raw_args[10], raw_args[11], raw_args[12], raw_args[13], raw_args[14]);
        },
        16 => blk: {
            const f: CallFn16 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw_args[0], raw_args[1], raw_args[2], raw_args[3], raw_args[4], raw_args[5], raw_args[6], raw_args[7], raw_args[8], raw_args[9], raw_args[10], raw_args[11], raw_args[12], raw_args[13], raw_args[14], raw_args[15]);
        },
        else => unreachable,
    };
}

fn dispatchAotLazyLocal(
    inst: *AotInstance,
    lowered_sig: host_trampolines.LoweredSig,
    local_func_idx: u32,
    caller_vmctx_raw: u64,
    raw_args: []const u64,
) !u64 {
    if (comptime !config.lazy_jit) return error.LazyJitDisabled;
    if (caller_vmctx_raw == 0) return error.InvalidVmCtx;
    if (lowered_sig.has_retptr) return error.UnsupportedSignature;
    if (lowered_sig.param_types.len != raw_args.len) return error.UnsupportedSignature;
    if (lowered_sig.param_types.len > 9) return error.UnsupportedSignature;
    if (lowered_sig.result_types.len > 1) return error.UnsupportedSignature;
    for (lowered_sig.param_types) |pt| {
        if (!isScalarValType(pt)) return error.UnsupportedSignature;
    }
    for (lowered_sig.result_types) |rt| {
        if (!isScalarValType(rt)) return error.UnsupportedSignature;
    }
    if (local_func_idx >= inst.lazy_jit.slot_states.len) return error.FunctionNotFound;
    const addr = try inst.lazy_jit.resolveLocalAddr(local_func_idx) orelse return error.CodeMappingFailed;
    const caller_vmctx: *VmCtx = @ptrFromInt(@as(usize, @intCast(caller_vmctx_raw)));
    const raw_result = invokeScalarCallable(addr, caller_vmctx, raw_args);
    return if (lowered_sig.result_types.len == 0) 0 else raw_result;
}

pub export fn wamrAotDispatchLazyLocalAot(
    ctx_opaque: *anyopaque,
    lowered_sig: *const host_trampolines.LoweredSig,
    local_func_idx: u32,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
    a9: u64,
) callconv(.c) host_trampolines.DispatchResult {
    const inst: *AotInstance = @ptrCast(@alignCast(ctx_opaque));
    const raw_args = [_]u64{ a1, a2, a3, a4, a5, a6, a7, a8, a9 };
    const result = dispatchAotLazyLocal(
        inst,
        lowered_sig.*,
        local_func_idx,
        a0,
        raw_args[0..lowered_sig.param_types.len],
    ) catch |err| {
        return .{ .status = 1, .value = 0, .err_name = @errorName(err).ptr };
    };
    return .{ .status = 0, .value = result };
}

pub fn globalStorageWordCount(inst: *const AotInstance) usize {
    const bytes = if (inst.global_storage_size == 0) @as(u32, 16) else inst.global_storage_size;
    return (@as(usize, bytes) + 15) / 16;
}

fn globalOffsetAt(inst: *const AotInstance, idx: usize) ?usize {
    if (idx < inst.global_offsets.len) return inst.global_offsets[idx];
    const off = idx * 8;
    if (off + 8 <= inst.global_storage_size) return off;
    return null;
}

pub fn writeGlobalsToStorage(inst: *const AotInstance, storage: []u8) void {
    @memset(storage, 0);
    for (inst.globals, 0..) |g, i| {
        const off = globalOffsetAt(inst, i) orelse continue;
        switch (g.value) {
            .v128 => |x| {
                if (off + 16 <= storage.len) std.mem.writeInt(u128, storage[off..][0..16], x, .little);
            },
            else => {
                if (off + 8 <= storage.len) std.mem.writeInt(i64, storage[off..][0..8], globalValueToI64(inst, g.value), .little);
            },
        }
    }
}

pub fn readGlobalsFromStorage(inst: *AotInstance, storage: []const u8) void {
    const imported_count = inst.module.importedGlobals().len;
    for (inst.globals, 0..) |g, i| {
        if (i < imported_count and i < inst.globals_owned.len and !inst.globals_owned[i]) continue;
        readGlobalSlotFromStorage(inst, g, i, storage);
    }
    flushImportedGlobalsToStorage(inst, storage);
}

fn readGlobalSlotFromStorage(inst: *const AotInstance, g: *types.GlobalInstance, i: usize, storage: []const u8) void {
    const off = globalOffsetAt(inst, i) orelse return;
    g.value = switch (g.value) {
        .v128 => if (off + 16 <= storage.len)
            .{ .v128 = std.mem.readInt(u128, storage[off..][0..16], .little) }
        else
            g.value,
        else => if (off + 8 <= storage.len)
            globalValueFromI64(inst, g.value, std.mem.readInt(i64, storage[off..][0..8], .little))
        else
            g.value,
    };
}

pub fn flushImportedGlobalsToStorage(inst: *AotInstance, storage: []const u8) void {
    const imported = inst.module.importedGlobals();
    for (imported, 0..) |desc, i| {
        if (!desc.mutable) continue;
        if (i >= inst.globals.len) continue;
        // Imported globals are per-call snapshots in AOT code. Mutable borrowed
        // imports must be written back on call exit so sibling importers see the
        // canonical GlobalInstance update. Concurrent host re-entry is
        // last-writer-wins at the call boundary (#660); true interleaving would
        // require per-access indirection instead of the slab fast path.
        readGlobalSlotFromStorage(inst, inst.globals[i], i, storage);
    }
}

/// Pack a global's typed value into a raw 64-bit scalar/reference slot for the
/// flat globals storage. The tag determines the bit-cast; using `.value.i64`
/// would be UB when the active tag is `.f32` / `.f64`.
pub fn globalValueToI64(inst: *const AotInstance, v: types.Value) i64 {
    const r: i64 = switch (v) {
        .i32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .i64 => |x| x,
        .f32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .f64 => |x| @as(i64, @bitCast(x)),
        .funcref, .nonfuncref => |maybe| blk: {
            const idx = maybe orelse break :blk 0;
            if (idx >= inst.funcptrs.len) break :blk 0;
            break :blk @as(i64, @bitCast(@as(u64, inst.funcptrs[idx])));
        },
        // externref values are opaque host-supplied integer tags. The
        // wasm-spec test suite uses `(ref.extern 0)` as a non-null handle,
        // which collides with our convention that 0 == null. Encode as
        // `N + 1` so the value 0 becomes raw 1 and remains non-null when
        // passed through `ref.is_null`. The reverse decoding lives in
        // `globalValueFromI64` and the `ScalarResult` packing in
        // `callFuncScalar`.
        .externref, .nonexternref => |maybe| if (maybe) |n| @as(i64, @as(u32, n)) + 1 else 0,
        else => 0,
    };
    return r;
}

/// Unpack a raw 64-bit slot from globals_buf back into a typed Value,
/// preserving the active tag.
fn globalValueFromI64(inst: *const AotInstance, old: types.Value, raw: i64) types.Value {
    return switch (old) {
        .i32 => .{ .i32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(raw))))) },
        .i64 => .{ .i64 = raw },
        .f32 => .{ .f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(raw))))) },
        .f64 => .{ .f64 = @bitCast(raw) },
        .funcref => blk: {
            const ptr: u64 = @bitCast(raw);
            const idx = funcPtrToIndex(inst, ptr);
            break :blk .{ .funcref = if (idx) |x| @as(u32, @truncate(x)) else null };
        },
        .nonfuncref => blk: {
            const ptr: u64 = @bitCast(raw);
            const idx = funcPtrToIndex(inst, ptr);
            break :blk .{ .nonfuncref = if (idx) |x| @as(u32, @truncate(x)) else null };
        },
        .externref => .{ .externref = if (raw == 0) null else @as(u32, @truncate(@as(u64, @bitCast(raw)) - 1)) },
        .nonexternref => .{ .nonexternref = if (raw == 0) null else @as(u32, @truncate(@as(u64, @bitCast(raw)) - 1)) },
        else => old,
    };
}

fn valueToRawBits(inst: *const AotInstance, pt: types.ValType, v: types.Value) ScalarCallError!u64 {
    return switch (pt) {
        .i32 => blk: {
            if (v != .i32) return error.InvalidArgType;
            break :blk @as(u64, @as(u32, @bitCast(v.i32)));
        },
        .i64 => blk: {
            if (v != .i64) return error.InvalidArgType;
            break :blk @as(u64, @bitCast(v.i64));
        },
        .f32 => blk: {
            if (v != .f32) return error.InvalidArgType;
            break :blk @as(u64, @as(u32, @bitCast(v.f32)));
        },
        .f64 => blk: {
            if (v != .f64) return error.InvalidArgType;
            break :blk @as(u64, @bitCast(v.f64));
        },
        // funcref in AOT is the native code pointer, not the wasm func
        // index. Translate the test-harness `.funcref = <idx>` form to
        // `inst.funcptrs[idx]` so the callee can `call r11` it directly.
        .funcref => switch (v) {
            .funcref => |maybe| blk: {
                const idx = maybe orelse break :blk 0;
                if (idx >= inst.funcptrs.len) return error.InvalidArgType;
                break :blk @as(u64, inst.funcptrs[idx]);
            },
            .nonfuncref => |maybe| blk: {
                const idx = maybe orelse break :blk 0;
                if (idx >= inst.funcptrs.len) return error.InvalidArgType;
                break :blk @as(u64, inst.funcptrs[idx]);
            },
            else => error.InvalidArgType,
        },
        // externref is opaque in AOT; tests pass small integer tags (see
        // `globalValueToI64` for the +1 tagging that disambiguates 0 from
        // null) and expect the same value back unchanged.
        .externref => switch (v) {
            .externref => |maybe| if (maybe) |n| @as(u64, n) + 1 else 0,
            .nonexternref => |maybe| if (maybe) |n| @as(u64, n) + 1 else 0,
            else => error.InvalidArgType,
        },
        else => error.UnsupportedSignature,
    };
}

/// Reverse-lookup a funcref native pointer back to the wasm function index
/// so test-harness comparisons against `.funcref = N` work transparently.
/// Returns null if `ptr` is 0 (null funcref); returns the u64 ptr unchanged
/// when no matching function exists (will compare unequal to any index).
fn funcPtrToIndex(inst: *const AotInstance, ptr: u64) ?u64 {
    if (ptr == 0) return null;
    for (inst.funcptrs, 0..) |fp, i| {
        if (@as(u64, fp) == ptr) return @as(u64, @intCast(i));
    }
    return ptr;
}

/// Call an AOT-compiled function by index with runtime-typed scalar args.
///
/// Supports up to 12 params and up to `MaxScalarResults` results of type
/// i32/i64/f32/f64/funcref/externref. Multi-value returns use the hidden
/// return pointer (HRP) ABI emitted by the x86_64 codegen: first result in
/// RAX, remaining stored via a caller-supplied buffer passed as an implicit
/// trailing arg after the wasm user params.
///
/// `results_out` must have capacity >= `result_types.len`. Returns a slice
/// into `results_out` with the decoded results. Wider or non-scalar
/// signatures return `error.UnsupportedSignature` and should be skipped by
/// the caller.
pub fn callFuncScalar(
    inst: *AotInstance,
    func_idx: u32,
    param_types: []const types.ValType,
    result_types: []const types.ValType,
    args: []const types.Value,
    results_out: []ScalarResult,
) ScalarCallError![]const ScalarResult {
    if (comptime !can_execute_native) return error.UnsupportedArchitecture;
    const call_thread_context = execution_context.current() orelse &inst.thread_context;
    var execution_scope = call_thread_context.enter();
    defer execution_scope.deinit();

    if (param_types.len != args.len) return error.ArgCountMismatch;
    if (param_types.len > MaxScalarArgs) return error.UnsupportedSignature;
    for (param_types) |pt| {
        if (!isScalarValType(pt)) return error.UnsupportedSignature;
    }
    for (result_types) |rt| {
        if (!isScalarValType(rt)) return error.UnsupportedSignature;
    }
    if (result_types.len > MaxScalarResults) return error.UnsupportedSignature;
    if (results_out.len < result_types.len) return error.UnsupportedSignature;

    // Multi-value returns require an extra HRP slot appended after user args.
    const needs_hrp = result_types.len > 1;
    const effective_args: usize = args.len + @as(usize, if (needs_hrp) 1 else 0);
    if (effective_args > MaxScalarArgs) return error.UnsupportedSignature;

    // Lazy-JIT: if `func_idx` names a deferred local function, resolve it
    // through the per-slot state machine before falling back to the
    // instance's already-published callable-pointer table. Lazy-eligible
    // slots transition `pending -> compiling -> ready` with acquire/release
    // ordering so same-slot first-call races serialize correctly while
    // other lazy locals remain independent. Checked BEFORE the
    // `getCallableAddr` fallback below because a module where EVERY
    // function is lazy-eligible (e.g. this spike's own test fixture) never
    // maps any code up front, so `code_base` legitimately stays null until
    // the first lazy compile publishes a `ready` slot.
    var lazy_resolved_addr: ?[*]const u8 = null;
    if (comptime config.lazy_jit) {
        const import_count = inst.module.import_function_count;
        if (func_idx >= import_count) {
            const local_idx: usize = @intCast(func_idx - import_count);
            lazy_resolved_addr = try inst.lazy_jit.resolveLocalAddr(local_idx);
            if (lazy_resolved_addr) |addr| {
                if (func_idx < inst.funcptrs.len) inst.funcptrs[func_idx] = @intFromPtr(addr);
            }
        }
    }

    const addr = lazy_resolved_addr orelse (getCallableAddr(inst, func_idx) orelse {
        if (inst.code_base == null) return error.CodeMappingFailed;
        return error.FunctionNotFound;
    });

    const previous_globals_ptr = inst.vmctx.globals_ptr;
    const vmctx_storage_addr = @intFromPtr(&inst.vmctx);
    // Host imports may re-enter the same AOT instance (notably cabi_realloc).
    // Reuse the active globals slab so mutable globals are not forked.
    const reuse_globals = previous_globals_ptr != 0;
    const globals_word_count = globalStorageWordCount(inst);
    var globals_words: []u128 = &.{};
    if (!reuse_globals) {
        globals_words = inst.allocator.alloc(u128, globals_word_count) catch return error.OutOfMemory;
    }
    defer if (!reuse_globals) inst.allocator.free(globals_words);
    const globals_buf: []u8 = if (reuse_globals)
        @as([*]u8, @ptrFromInt(previous_globals_ptr))[0 .. globals_word_count * @sizeOf(u128)]
    else
        std.mem.sliceAsBytes(globals_words);
    if (!reuse_globals) writeGlobalsToStorage(inst, globals_buf);

    refreshVmCtxForInstance(inst, globals_buf);
    const vmctx: *VmCtx = @ptrFromInt(vmctx_storage_addr);
    defer @as(*VmCtx, @ptrFromInt(vmctx_storage_addr)).globals_ptr = previous_globals_ptr;
    const default_thread_context = vmctx.thread_context;
    const default_wasi_ctx = vmctx.wasi_ctx;
    vmctx.thread_context = @intFromPtr(call_thread_context);
    if (call_thread_context.process_state) |state| {
        vmctx.wasi_ctx = @intFromPtr(state.ptr);
    }
    defer {
        vmctx.thread_context = default_thread_context;
        vmctx.wasi_ctx = default_wasi_ctx;
    }

    // Marshal args to raw 64-bit bit patterns.Multi-value calls append a
    // hidden return pointer (HRP) at raw[args.len] pointing at `hrp_buf`;
    // the callee stores results[1..] there (codegen writes RAX for results[0]
    // and `[HRP + (i-1)*8]` for i in [1, result_count)).
    var raw: [MaxScalarArgs]u64 = [_]u64{0} ** MaxScalarArgs;
    for (args, param_types, 0..) |v, pt, i| {
        raw[i] = try valueToRawBits(inst, pt, v);
    }
    var hrp_buf: [MaxScalarResults - 1]u64 = [_]u64{0} ** (MaxScalarResults - 1);
    if (needs_hrp) {
        raw[args.len] = @intFromPtr(&hrp_buf);
    }

    var aot_call_state = AotCallState{};
    var backend_scope = call_thread_context.bindBackendContext(@ptrCast(&aot_call_state));
    defer backend_scope.deinit();
    installTrapDecodeFrameFor(inst);
    if (comptime windows_trap_supported) {
        aot_call_state.trap_decode.mem_base = vmctx.memory_base;
        aot_call_state.trap_decode.mem_size = vmctx.memory_size;
        ensureVehInstalled();
    }

    // Arm the trap-as-error path. Trap helpers called from generated code
    // consult this call-local state and longjmp back to the capture site.
    //
    // We do NOT arm this for the hardware-VEH path (ud2/int3 traps inside
    // generated code); those are still routed via the VEH, which proved
    // unstable for our use case. All wasm traps now go through explicit
    // helper calls, so the VEH is effectively unused.
    if (comptime windows_trap_supported) {
        aot_call_state.trap_occurred.store(false, .seq_cst);
        aot_call_state.last_trap_code.store(0, .seq_cst);
        // Reserve extra stack headroom so the VEH and trapLongjmp can
        // run safely after a STATUS_STACK_OVERFLOW consumes the guard
        // page. Without this, the OS leaves ~4KB of space below the
        // former guard for user-mode dispatch — enough for the VEH to
        // fire, but RtlRestoreContext and the subsequent return path
        // may overflow. 16 KB is generous and idempotent across calls.
        var guarantee: u32 = 16 * 1024;
        _ = SetThreadStackGuarantee(&guarantee);
        aot_call_state.trap_catching.store(true, .seq_cst);
        RtlCaptureContext(&aot_call_state.saved_ctx);
        if (aot_call_state.trap_occurred.load(.seq_cst)) {
            aot_call_state.trap_catching.store(false, .seq_cst);
            call_thread_context.markTrap();
            // If the trap was a stack overflow, the OS consumed the
            // thread's guard page. Re-arm it here so a subsequent
            // overflow in this process is also catchable rather than
            // silently aborting. Runs on the post-longjmp stack, which
            // is well clear of the former-guard region.
            if (aot_call_state.last_trap_code.load(.seq_cst) == 0xC00000FD) {
                resetStackGuardPage();
            }
            readGlobalsFromStorage(inst, globals_buf);
            return error.WasmTrap;
        }
    }

    // #798 Lever 1: POSIX x86_64 analogue. The trap helpers (aotTrapOOB,
    // aotTrapUnreachable, ...) longjmp back to the `trap_jmp.capture` site
    // below when the call-local trap state is armed; we then return
    // `error.WasmTrap` instead of aborting the process.
    if (comptime posix_trap_supported) {
        aot_call_state.trap_occurred.store(false, .seq_cst);
        aot_call_state.trap_catching.store(true, .seq_cst);
        if (trap_jmp.capture(&aot_call_state.posix_trap_buf) != 0) {
            // A trap helper unwound back here.
            aot_call_state.trap_catching.store(false, .seq_cst);
            call_thread_context.markTrap();
            readGlobalsFromStorage(inst, globals_buf);
            return error.WasmTrap;
        }
    }

    const raw_result: u64 = switch (effective_args) {
        0 => blk: {
            const f: CallFn0 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx);
        },
        1 => blk: {
            const f: CallFn1 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0]);
        },
        2 => blk: {
            const f: CallFn2 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1]);
        },
        3 => blk: {
            const f: CallFn3 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2]);
        },
        4 => blk: {
            const f: CallFn4 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3]);
        },
        5 => blk: {
            const f: CallFn5 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4]);
        },
        6 => blk: {
            const f: CallFn6 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]);
        },
        7 => blk: {
            const f: CallFn7 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6]);
        },
        8 => blk: {
            const f: CallFn8 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]);
        },
        9 => blk: {
            const f: CallFn9 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8]);
        },
        10 => blk: {
            const f: CallFn10 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9]);
        },
        11 => blk: {
            const f: CallFn11 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9], raw[10]);
        },
        12 => blk: {
            const f: CallFn12 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9], raw[10], raw[11]);
        },
        13 => blk: {
            const f: CallFn13 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9], raw[10], raw[11], raw[12]);
        },
        14 => blk: {
            const f: CallFn14 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9], raw[10], raw[11], raw[12], raw[13]);
        },
        15 => blk: {
            const f: CallFn15 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14]);
        },
        16 => blk: {
            const f: CallFn16 = @ptrCast(@alignCast(addr));
            break :blk f(vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15]);
        },
        else => unreachable,
    };

    if (comptime windows_trap_supported) {
        aot_call_state.trap_catching.store(false, .seq_cst);
    }
    if (comptime posix_trap_supported) {
        aot_call_state.trap_catching.store(false, .seq_cst);
    }

    // Sync globals back.
    const vmctx_after: *VmCtx = @ptrFromInt(vmctx_storage_addr);
    const inst_after: *AotInstance = @ptrFromInt(vmctx_after.instance_ptr);
    const globals_buf_after: []u8 = if (vmctx_after.globals_ptr != 0)
        @as([*]u8, @ptrFromInt(vmctx_after.globals_ptr))[0 .. globals_word_count * @sizeOf(u128)]
    else
        globals_buf;
    readGlobalsFromStorage(inst_after, globals_buf_after);

    if (result_types.len == 0) return results_out[0..0];

    results_out[0] = decodeScalarResult(inst_after, result_types[0], raw_result);
    var i: usize = 1;
    while (i < result_types.len) : (i += 1) {
        results_out[i] = decodeScalarResult(inst_after, result_types[i], hrp_buf[i - 1]);
    }
    return results_out[0..result_types.len];
}

/// Decode a raw 64-bit result slot into a typed ScalarResult.
fn decodeScalarResult(inst: *AotInstance, rt: types.ValType, raw_result: u64) ScalarResult {
    return switch (rt) {
        .i32 => ScalarResult{ .i32 = @bitCast(@as(u32, @truncate(raw_result))) },
        .i64 => ScalarResult{ .i64 = @bitCast(raw_result) },
        .f32 => ScalarResult{ .f32 = @bitCast(@as(u32, @truncate(raw_result))) },
        .f64 => ScalarResult{ .f64 = @bitCast(raw_result) },
        .funcref => ScalarResult{ .funcref = funcPtrToIndex(inst, raw_result) },
        .externref => ScalarResult{ .externref = if (raw_result == 0) null else raw_result - 1 },
        else => unreachable,
    };
}

// ─── Host function resolution ───────────────────────────────────────────────

/// Resolve AOT host function adapters for each import in the module.
/// Returns a slice of optional function pointers indexed by import index.
fn resolveHostFunctions(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
) RuntimeError![]const ?*const anyopaque {
    return resolveHostFunctionsImpl(module, allocator, null, &.{});
}

fn resolveHostFunctionsWithOverrides(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
    overrides: []const ?*const anyopaque,
) RuntimeError![]const ?*const anyopaque {
    return resolveHostFunctionsImpl(module, allocator, null, overrides);
}

/// Resolve host functions with optional custom HostImports (comptime-typed).
pub fn resolveHostFunctionsWithHosts(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
    comptime HostImportsT: ?type,
) RuntimeError![]const ?*const anyopaque {
    return resolveHostFunctionsImpl(module, allocator, HostImportsT, &.{});
}

fn resolveHostFunctionsImpl(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
    comptime HostImportsT: ?type,
    overrides: []const ?*const anyopaque,
) RuntimeError![]const ?*const anyopaque {
    if (module.import_function_count == 0) return &.{};

    const host_fns = allocator.alloc(?*const anyopaque, module.import_function_count) catch
        return error.OutOfMemory;
    @memset(host_fns, null);

    var func_idx: u32 = 0;
    for (module.imports) |imp| {
        if (imp.kind == .function) {
            if (func_idx < module.import_function_count) {
                // Layer 0: caller-supplied per-import override (#662 Phase C
                // cross-instance fn imports + trap-on-call stubs).
                if (func_idx < overrides.len) {
                    if (overrides[func_idx]) |entry| host_fns[func_idx] = entry;
                }
                // Layer 1: custom HostImports (comptime-resolved)
                if (host_fns[func_idx] == null) {
                    if (HostImportsT) |HI| {
                        if (HI.resolve(imp.module_name, imp.field_name)) |entry| {
                            host_fns[func_idx] = entry.aot_fn;
                        }
                    }
                }
                // Layer 2: WASI / spectest (only if not already resolved)
                if (host_fns[func_idx] == null) {
                    if (host_bridge.isWasiModule(imp.module_name)) {
                        host_fns[func_idx] = host_bridge.resolveAotHostFunction(imp.field_name);
                    } else if (host_bridge.isSpectestModule(imp.module_name)) {
                        host_fns[func_idx] = host_bridge.resolveAotSpectestFunction(imp.field_name);
                    }
                }
            }
            func_idx += 1;
        }
    }

    return host_fns;
}

// ─── Allocation helpers ─────────────────────────────────────────────────────

const MemoriesAllocation = struct {
    memories: []*types.MemoryInstance,
    owned: []bool,
};

/// Allocate the memory slots for an instance. Imported memory slots are
/// kept at the **front** of `memories` (so `memories[0]` still aliases the
/// active linear memory the way `VmCtx.memory_base` wiring expects).
/// Slots with a non-null entry in `imported_memory_overrides` are borrowed
/// from the exporting sibling (retain()'d until destroy); everything else
/// (and unoverriden imports) is locally allocated.
fn allocateMemories(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
    imported_memory_overrides: []const ?*types.MemoryInstance,
) RuntimeError!MemoriesAllocation {
    const imported = module.importedMemories();
    const total_count = imported.len + module.memories.len;
    if (total_count == 0) return .{ .memories = &.{}, .owned = &.{} };

    const memories = allocator.alloc(*types.MemoryInstance, total_count) catch return error.OutOfMemory;
    errdefer allocator.free(memories);
    const owned = allocator.alloc(bool, total_count) catch return error.OutOfMemory;
    errdefer allocator.free(owned);
    @memset(owned, false);

    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| memories[i].release(allocator);
    }

    for (imported, 0..) |desc, i| {
        if (i < imported_memory_overrides.len) {
            if (imported_memory_overrides[i]) |override| {
                @constCast(override).retain();
                memories[i] = override;
                initialized += 1;
                continue;
            }
        }
        const mem_type = types.MemoryType{
            .limits = .{
                .min = desc.min,
                .max = if (desc.max) |max| @as(u64, max) else null,
            },
            .is_memory64 = desc.is64,
            .is_shared = desc.is_shared,
        };
        memories[i] = try allocateOneMemory(mem_type, allocator);
        owned[i] = true;
        initialized += 1;
    }

    for (module.memories, 0..) |mem_type, local_i| {
        const i = imported.len + local_i;
        memories[i] = try allocateOneMemory(mem_type, allocator);
        owned[i] = true;
        initialized += 1;
    }

    return .{ .memories = memories, .owned = owned };
}

fn allocateOneMemory(mem_type: types.MemoryType, allocator: std.mem.Allocator) RuntimeError!*types.MemoryInstance {
    if (mem_type.is_shared) {
        // Shared memories must never fall back to allocator.realloc: every
        // owner and parked address depends on an immutable base.
        return types.MemoryInstance.createShared(mem_type, allocator) catch
            return error.OutOfMemory;
    }

    const initial_pages: u32 = @intCast(@min(mem_type.limits.min, 65536));
    const max_pages: u32 = @intCast(@min(mem_type.limits.max orelse 65536, 65536));

    // Prefer stable-address backing on platforms that support it (#752).
    // Reserving `max_pages * 64 KiB` of virtual address space up front
    // means `memory.grow` only `mprotect`s additional pages — the
    // `data.ptr` never moves. This keeps any external aliases into
    // this memory valid across grows, including SpiderMonkey/
    // StarlingMonkey "external string" char pointers that
    // componentize-js wraps around canon-lifted strings. Falls back to
    // the allocator-realloc path if reservation isn't supported on the
    // host (Windows for now) or the reservation fails.
    if (types.MemoryInstance.createReserved(mem_type, initial_pages, max_pages, allocator)) |mem| {
        return mem;
    }

    // Fallback: legacy allocator-backed memory. Pre-size to a reasonable
    // chunk so the first few `memory.grow` calls hit the no-op path
    // (still copies via `allocator.realloc` once the cap is exceeded).
    // `data.len` intentionally tracks the *allocated* capacity, not
    // `current_pages * page_size` — `MemoryInstance.grow` keys its
    // "do we need to realloc?" check on `data.len`.
    const alloc_pages = @min(max_pages, @max(initial_pages, 256));
    const size = @as(usize, alloc_pages) * types.MemoryInstance.page_size;
    const data = allocator.alloc(u8, size) catch return error.OutOfMemory;
    @memset(data, 0);
    const mem = allocator.create(types.MemoryInstance) catch {
        allocator.free(data);
        return error.OutOfMemory;
    };
    mem.* = .{
        .memory_type = mem_type,
        .data = data,
        .current_pages = initial_pages,
        .max_pages = max_pages,
    };
    return mem;
}

const TablesAllocation = struct {
    tables: []*types.TableInstance,
    owned: []bool,
};

fn allocateTables(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
    imported_table_overrides: []const ?*types.TableInstance,
) RuntimeError!TablesAllocation {
    const imported = module.importedTables();
    const total_count = imported.len + module.tables.len;
    if (total_count == 0) return .{ .tables = &.{}, .owned = &.{} };

    const tables = allocator.alloc(*types.TableInstance, total_count) catch return error.OutOfMemory;
    errdefer allocator.free(tables);
    const owned = allocator.alloc(bool, total_count) catch return error.OutOfMemory;
    errdefer allocator.free(owned);
    @memset(owned, false);

    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| tables[i].release(allocator);
    }

    for (imported, 0..) |desc, i| {
        if (i < imported_table_overrides.len) {
            if (imported_table_overrides[i]) |override| {
                @constCast(override).retain();
                tables[i] = override;
                initialized += 1;
                continue;
            }
        }

        const table_type = types.TableType{
            .elem_type = desc.elem_type,
            .limits = .{
                .min = desc.min,
                .max = if (desc.max) |max| @as(u64, max) else null,
            },
        };
        const elements = allocator.alloc(types.TableElement, table_type.limits.min) catch return error.TableAllocationFailed;
        for (elements) |*e| e.* = types.TableElement.nullForType(table_type.elem_type);
        const tbl = allocator.create(types.TableInstance) catch {
            allocator.free(elements);
            return error.TableAllocationFailed;
        };
        tbl.* = .{ .table_type = table_type, .elements = elements };
        tables[i] = tbl;
        owned[i] = true;
        initialized += 1;
    }

    for (module.tables, 0..) |table_type, local_i| {
        const i = imported.len + local_i;
        const elements = allocator.alloc(types.TableElement, table_type.limits.min) catch return error.TableAllocationFailed;
        for (elements) |*e| e.* = types.TableElement.nullForType(table_type.elem_type);
        const tbl = allocator.create(types.TableInstance) catch {
            allocator.free(elements);
            return error.TableAllocationFailed;
        };
        tbl.* = .{ .table_type = table_type, .elements = elements };
        tables[i] = tbl;
        owned[i] = true;
        initialized += 1;
    }

    return .{ .tables = tables, .owned = owned };
}

const GlobalLayout = struct {
    offsets: []u32,
    size: u32,
};

fn alignForwardU32(value: u32, alignment: u32) u32 {
    std.debug.assert(alignment != 0 and (alignment & (alignment - 1)) == 0);
    return (value + alignment - 1) & ~(alignment - 1);
}

fn globalSlotAlignment(vt: types.ValType) u32 {
    return switch (vt) {
        .v128 => 16,
        else => 8,
    };
}

fn globalSlotSize(vt: types.ValType) u32 {
    return switch (vt) {
        .v128 => 16,
        else => 8,
    };
}

fn computeGlobalLayout(
    imports: []const aot_loader.ImportedGlobalDesc,
    inits: []const aot_loader.AotGlobalInit,
    allocator: std.mem.Allocator,
) RuntimeError!GlobalLayout {
    const total = imports.len + inits.len;
    if (total == 0) return .{ .offsets = &.{}, .size = 0 };

    const offsets = allocator.alloc(u32, total) catch return error.OutOfMemory;
    errdefer allocator.free(offsets);

    var next: u32 = 0;
    // Imported globals come first in the wasm-flat global index space, so
    // their slab offsets must match what the codegen baked into
    // `global.get`/`global.set` instructions (frontend.zig also lays them
    // out imports-first when computing `ir_module.global_offsets`).
    for (imports, 0..) |g, i| {
        next = alignForwardU32(next, globalSlotAlignment(g.val_type));
        offsets[i] = next;
        next += globalSlotSize(g.val_type);
    }
    for (inits, 0..) |ginit, i| {
        const vt: types.ValType = @enumFromInt(ginit.val_type);
        next = alignForwardU32(next, globalSlotAlignment(vt));
        offsets[imports.len + i] = next;
        next += globalSlotSize(vt);
    }

    return .{ .offsets = offsets, .size = next };
}

/// Allocate the wasm-flat `[]*GlobalInstance` slice for an AOT instance.
///
/// Imported globals come first (matching codegen's wasm-flat indexing
/// established in `frontend.zig`); for each imported slot we either
/// borrow the override's `GlobalInstance` (retained) or, when no override
/// is provided, allocate a fresh zero-valued local stand-in. Locally
/// defined globals follow, materialised from `module.global_inits`.
///
/// The returned `GlobalsAllocation.owned[i]` is true exactly when this
/// instance allocated `globals[i]` itself (and therefore must release it
/// in `destroy`). Borrowed import slots leave `owned[i] = false` so the
/// exporting sibling retains lifetime ownership.
const GlobalsAllocation = struct {
    globals: []*types.GlobalInstance,
    owned: []bool,
};

fn allocateGlobals(
    module: *const aot_loader.AotModule,
    imported_global_overrides: []const ?*types.GlobalInstance,
    allocator: std.mem.Allocator,
) RuntimeError!GlobalsAllocation {
    const imports = module.importedGlobals();
    const total = imports.len + module.global_inits.len;
    if (total == 0) return .{ .globals = &.{}, .owned = &.{} };

    const globals = allocator.alloc(*types.GlobalInstance, total) catch return error.OutOfMemory;
    errdefer allocator.free(globals);
    const owned = allocator.alloc(bool, total) catch return error.OutOfMemory;
    errdefer allocator.free(owned);

    var initialized: usize = 0;
    errdefer {
        for (globals[0..initialized]) |global| global.release(allocator);
    }

    for (imports, 0..) |imp, i| {
        if (imported_global_overrides.len > i and imported_global_overrides[i] != null) {
            const shared = imported_global_overrides[i].?;
            shared.retain();
            // Exact exporter GlobalInstance; call-exit flush updates the
            // canonical value observed by sibling importers (#660).
            globals[i] = shared;
            owned[i] = false;
        } else {
            const g = allocator.create(types.GlobalInstance) catch return error.OutOfMemory;
            g.* = .{
                .global_type = .{
                    .val_type = imp.val_type,
                    .mutability = if (imp.mutable) .mutable else .immutable,
                },
                .value = defaultGlobalZero(imp.val_type),
                .owned = true,
            };
            globals[i] = g;
            owned[i] = true;
        }
        initialized += 1;
    }

    for (module.global_inits, 0..) |ginit, i| {
        const slot = imports.len + i;
        const g = allocator.create(types.GlobalInstance) catch return error.OutOfMemory;
        const vt: types.ValType = @enumFromInt(ginit.val_type);
        // Tag the stored Value per the declared val_type so later
        // packing/unpacking through globals_buf preserves ref types
        // (funcref idx ↔ native ptr conversion happens in
        // globalValueToI64 / globalValueFromI64).
        const val: types.Value = switch (vt) {
            .i32 => .{ .i32 = @as(i32, @truncate(ginit.init_i64)) },
            .i64 => .{ .i64 = ginit.init_i64 },
            .f32 => .{ .f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(ginit.init_i64))))) },
            .f64 => .{ .f64 = @bitCast(ginit.init_i64) },
            .v128 => .{ .v128 = ginit.init_v128 },
            .funcref => .{ .funcref = if (ginit.init_i64 == 0) null else @as(u32, @truncate(@as(u64, @bitCast(ginit.init_i64)) - 1)) },
            .nonfuncref => .{ .nonfuncref = if (ginit.init_i64 == 0) null else @as(u32, @truncate(@as(u64, @bitCast(ginit.init_i64)) - 1)) },
            .externref => .{ .externref = if (ginit.init_i64 == 0) null else @as(u32, @truncate(@as(u64, @bitCast(ginit.init_i64)) - 1)) },
            .nonexternref => .{ .nonexternref = if (ginit.init_i64 == 0) null else @as(u32, @truncate(@as(u64, @bitCast(ginit.init_i64)) - 1)) },
            else => .{ .i64 = ginit.init_i64 },
        };
        g.* = .{
            .global_type = .{
                .val_type = vt,
                .mutability = if (ginit.mutability != 0) .mutable else .immutable,
            },
            .value = val,
        };
        globals[slot] = g;
        owned[slot] = true;
        initialized += 1;
    }

    return .{ .globals = globals, .owned = owned };
}

fn defaultGlobalZero(vt: types.ValType) types.Value {
    return switch (vt) {
        .i32 => .{ .i32 = 0 },
        .i64 => .{ .i64 = 0 },
        .f32 => .{ .f32 = 0 },
        .f64 => .{ .f64 = 0 },
        .v128 => .{ .v128 = 0 },
        .funcref => .{ .funcref = null },
        .nonfuncref => .{ .nonfuncref = null },
        .externref => .{ .externref = null },
        .nonexternref => .{ .nonexternref = null },
        else => .{ .i64 = 0 },
    };
}

fn freeMemories(memories: []*types.MemoryInstance, owned: []bool, allocator: std.mem.Allocator) void {
    std.debug.assert(owned.len == 0 or owned.len == memories.len);
    for (memories) |m| {
        m.release(allocator);
    }
    if (memories.len > 0) allocator.free(memories);
    if (owned.len > 0) allocator.free(owned);
}

fn freeTables(tables: []*types.TableInstance, owned: []bool, allocator: std.mem.Allocator) void {
    std.debug.assert(owned.len == 0 or owned.len == tables.len);
    for (tables) |t| {
        t.release(allocator);
    }
    if (tables.len > 0) allocator.free(tables);
    if (owned.len > 0) allocator.free(owned);
}

fn freeGlobals(globals: []*types.GlobalInstance, owned: []bool, allocator: std.mem.Allocator) void {
    std.debug.assert(owned.len == 0 or owned.len == globals.len);
    for (globals) |global| global.release(allocator);
    if (globals.len > 0) allocator.free(globals);
    if (owned.len > 0) allocator.free(owned);
}

// ─── Tags (#672) ────────────────────────────────────────────────────────────

const TagsAllocation = struct {
    tags: []*types.TagInstance,
    owned: []bool,
};

/// Allocate (or borrow) `TagInstance`s for every imported + declared tag.
/// `imported_tag_overrides[i]` (if non-null) borrows a sibling instance's
/// tag for `module.importedTags()[i]`; otherwise a fresh `TagInstance` is
/// synthesized so the slot is well-formed even with no exporter wired.
/// Local (declared) tag arity is read from `module.func_types[type_idx]`.
fn allocateTags(
    module: *const aot_loader.AotModule,
    imported_tag_overrides: []const ?*types.TagInstance,
    allocator: std.mem.Allocator,
) RuntimeError!TagsAllocation {
    const import_tags = module.importedTags();
    const total = import_tags.len + module.tag_types.len;
    if (total == 0) return .{ .tags = &.{}, .owned = &.{} };

    const tags = allocator.alloc(*types.TagInstance, total) catch return error.OutOfMemory;
    errdefer allocator.free(tags);
    const owned = allocator.alloc(bool, total) catch return error.OutOfMemory;
    errdefer allocator.free(owned);

    // Synthesize a placeholder for failure paths so freeTags can iterate
    // before any real allocation has happened.
    var created: usize = 0;
    errdefer for (tags[0..created], owned[0..created]) |t, o| {
        if (o) allocator.destroy(t);
    };

    for (import_tags, 0..) |imp_desc, i| {
        if (i < imported_tag_overrides.len) {
            if (imported_tag_overrides[i]) |borrowed| {
                tags[i] = borrowed;
                owned[i] = false;
                created += 1;
                continue;
            }
        }
        // No override → synthesize a local placeholder with the declared
        // import's arity. Cross-instance identity is resolved later
        // (commit 6, closes #670).
        const arity = arityForTypeIdx(module, imp_desc.type_idx);
        const t = allocator.create(types.TagInstance) catch return error.OutOfMemory;
        t.* = .{ .param_arity = arity };
        tags[i] = t;
        owned[i] = true;
        created += 1;
    }

    for (module.tag_types, 0..) |type_idx, j| {
        const slot = import_tags.len + j;
        const t = allocator.create(types.TagInstance) catch return error.OutOfMemory;
        t.* = .{ .param_arity = arityForTypeIdx(module, type_idx) };
        tags[slot] = t;
        owned[slot] = true;
        created += 1;
    }

    return .{ .tags = tags, .owned = owned };
}

fn arityForTypeIdx(module: *const aot_loader.AotModule, type_idx: u32) u32 {
    if (type_idx >= module.func_types.len) return 0;
    return @intCast(module.func_types[type_idx].params.len);
}

fn freeTags(tags: []*types.TagInstance, owned: []bool, allocator: std.mem.Allocator) void {
    std.debug.assert(owned.len == 0 or owned.len == tags.len);
    for (tags, 0..) |t, i| {
        if (owned.len == 0 or owned[i]) allocator.destroy(t);
    }
    if (tags.len > 0) allocator.free(tags);
    if (owned.len > 0) allocator.free(owned);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

test "core resource AOT global rollback releases borrowed override" {
    const allocator = std.testing.allocator;
    const shared = try allocator.create(types.GlobalInstance);
    shared.* = .{
        .global_type = .{ .val_type = .i32, .mutability = .mutable },
        .value = .{ .i32 = 9 },
    };
    defer shared.release(allocator);

    const imports = [_]aot_loader.ImportedGlobalDesc{.{
        .module_name = "env",
        .name = "g",
        .val_type = .i32,
        .mutable = true,
    }};
    const locals = [_]aot_loader.AotGlobalInit{.{
        .val_type = @intFromEnum(types.ValType.i32),
        .mutability = 1,
        .init_i64 = 1,
    }};
    var module = aot_loader.AotModule{
        .imported_globals = &imports,
        .global_inits = &locals,
    };
    const overrides = [_]?*types.GlobalInstance{shared};
    var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 2 });

    try std.testing.expectError(
        error.OutOfMemory,
        allocateGlobals(&module, &overrides, failing.allocator()),
    );
    try std.testing.expectEqual(@as(usize, 1), shared.referenceCount());
}

test "instantiate resolves configured AOT thread-spawn imports" {
    if (comptime !config.lib_wasi_threads) return error.SkipZigTest;

    const imports = [_]aot_loader.AotImportDesc{.{
        .module_name = "wasi",
        .field_name = "thread-spawn",
        .kind = .function,
    }};
    const module = aot_loader.AotModule{
        .imports = &imports,
        .import_function_count = 1,
    };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);
    try std.testing.expectEqual(@as(usize, 1), inst.host_functions.len);
    try std.testing.expect(inst.host_functions[0] != null);
}

test "instantiate: empty module" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 0), inst.memories.len);
    try std.testing.expectEqual(@as(usize, 0), inst.tables.len);
    try std.testing.expectEqual(@as(usize, 0), inst.globals.len);
    try std.testing.expectEqual(@as(?[*]const u8, null), inst.code_base);
}

test "globals storage packs v128 globals on 16-byte aligned offsets" {
    const inits = [_]aot_loader.AotGlobalInit{
        .{ .val_type = @intFromEnum(types.ValType.i32), .mutability = 0, .init_i64 = 5 },
        .{ .val_type = @intFromEnum(types.ValType.v128), .mutability = 1, .init_v128 = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF },
        .{ .val_type = @intFromEnum(types.ValType.i64), .mutability = 0, .init_i64 = 9 },
    };
    const module = aot_loader.AotModule{ .global_inits = &inits };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqualSlices(u32, &.{ 0, 16, 32 }, inst.global_offsets);
    try std.testing.expectEqual(@as(u32, 40), inst.global_storage_size);
    try std.testing.expectEqual(@as(usize, 3), globalStorageWordCount(inst));

    const storage_words = try std.testing.allocator.alloc(u128, globalStorageWordCount(inst));
    defer std.testing.allocator.free(storage_words);
    const storage = std.mem.sliceAsBytes(storage_words);
    writeGlobalsToStorage(inst, storage);

    try std.testing.expectEqual(@as(i64, 5), std.mem.readInt(i64, storage[0..8], .little));
    try std.testing.expectEqual(inits[1].init_v128, std.mem.readInt(u128, storage[16..][0..16], .little));
    try std.testing.expectEqual(@as(i64, 9), std.mem.readInt(i64, storage[32..][0..8], .little));

    const updated: u128 = 0xFFEEDDCC_BBAA9988_77665544_33221100;
    std.mem.writeInt(u128, storage[16..][0..16], updated, .little);
    readGlobalsFromStorage(inst, storage);
    try std.testing.expectEqual(updated, inst.globals[1].value.v128);
}

test "findExportFunc: returns null for missing export" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(?u32, null), findExportFunc(inst, "nonexistent"));
}

test "#672: instantiate allocates TagInstance for declared tags" {
    // A module declaring two tags with different param-arity should
    // surface `inst.tags` parallel to `module.tag_types`, with each
    // entry's `param_arity` taken from `module.func_types[type_idx]`.
    const params0 = [_]types.ValType{ .i32, .i32 };
    const params1 = [_]types.ValType{.i64};
    const ftypes = [_]aot_loader.AotFuncType{
        .{ .params = &params0, .results = &.{} },
        .{ .params = &params1, .results = &.{} },
    };
    const tags = [_]u32{ 0, 1 };
    const module = aot_loader.AotModule{
        .func_types = &ftypes,
        .tag_types = &tags,
    };

    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 2), inst.tags.len);
    try std.testing.expectEqual(@as(usize, 2), inst.tags_owned.len);
    try std.testing.expect(inst.tags_owned[0]);
    try std.testing.expect(inst.tags_owned[1]);
    try std.testing.expectEqual(@as(u32, 2), inst.tags[0].param_arity);
    try std.testing.expectEqual(@as(u32, 1), inst.tags[1].param_arity);
}

test "#672: instantiate borrows imported tag overrides" {
    // An imported tag whose `imported_tag_overrides[i]` is non-null
    // should borrow that `*TagInstance` rather than synthesizing a fresh
    // one; the borrowed entry must NOT be freed on destroy.
    const params = [_]types.ValType{.i32};
    const ftypes = [_]aot_loader.AotFuncType{
        .{ .params = &params, .results = &.{} },
    };
    const imported = [_]aot_loader.ImportedTagDesc{
        .{ .module_name = "env", .name = "exn", .type_idx = 0 },
    };
    const module = aot_loader.AotModule{
        .func_types = &ftypes,
        .imported_tags = &imported,
    };

    // Stand-in exporter tag — allocated by the test, not by the runtime.
    const exporter_tag = try std.testing.allocator.create(types.TagInstance);
    defer std.testing.allocator.destroy(exporter_tag);
    exporter_tag.* = .{ .param_arity = 1 };

    const overrides = [_]?*types.TagInstance{exporter_tag};
    const inst = try instantiateWithOverrides(
        &module,
        std.testing.allocator,
        &.{},
        &.{},
        &.{},
        &.{},
        &overrides,
    );
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 1), inst.tags.len);
    try std.testing.expectEqual(exporter_tag, inst.tags[0]);
    try std.testing.expect(!inst.tags_owned[0]);
}

test "findExportFunc: finds exported function" {
    const exports = [_]types.ExportDesc{
        .{ .name = "memory", .kind = .memory, .index = 0 },
        .{ .name = "_start", .kind = .function, .index = 3 },
    };
    const module = aot_loader.AotModule{ .exports = &exports };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(?u32, 3), findExportFunc(inst, "_start"));
    try std.testing.expectEqual(@as(?u32, null), findExportFunc(inst, "missing"));
}

test "destroy: cleans up without leaks" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    destroy(inst);
    // If the testing allocator doesn't report leaks, we're good.
}

test "instantiate: module with memory" {
    const mem_types = [_]types.MemoryType{
        .{ .limits = .{ .min = 1, .max = 4 } },
    };
    const module = aot_loader.AotModule{ .memories = &mem_types };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 1), inst.memories.len);
    try std.testing.expectEqual(@as(u32, 1), inst.memories[0].current_pages);
    try std.testing.expectEqual(@as(u32, 4), inst.memories[0].max_pages);
    // `data.len` tracks the currently-committed window. With the
    // stable-address (mmap-reserved) backing introduced for #752,
    // that is `current_pages * page_size`; with the legacy
    // allocator-realloc fallback it is the pre-allocated capacity
    // (`alloc_pages * page_size`, ≥ `current_pages * page_size`).
    // Either way, the slice must cover at least the logical
    // current-pages range.
    try std.testing.expect(
        inst.memories[0].data.len >= @as(usize, 1) * types.MemoryInstance.page_size,
    );
    try std.testing.expect(
        inst.memories[0].data.len <= @as(usize, 4) * types.MemoryInstance.page_size,
    );
}

test "memory.grow propagates to cross-instance AOT memory subscribers" {
    const mem_types = [_]types.MemoryType{
        .{ .limits = .{ .min = 1, .max = 4 } },
    };
    const exporter_module = aot_loader.AotModule{ .memories = &mem_types };
    const exporter = try instantiate(&exporter_module, std.testing.allocator);
    defer destroy(exporter);

    const imported_mems = [_]aot_loader.ImportedMemoryDesc{
        .{ .module_name = "exporter", .name = "memory", .min = 1, .max = 4 },
    };
    const importer_module = aot_loader.AotModule{ .imported_memories = &imported_mems };
    const memory_overrides = [_]?*types.MemoryInstance{exporter.memories[0]};
    const importer = try instantiateWithOverrides(&importer_module, std.testing.allocator, &.{}, &memory_overrides, &.{}, &.{}, &.{});
    defer destroy(importer);

    try std.testing.expectEqual(exporter.memories[0], importer.memories[0]);
    try std.testing.expectEqual(@as(usize, 2), exporter.memories[0].vmctx_subscribers.items.len);

    try std.testing.expectEqual(@as(i32, 1), memGrowHelper(&exporter.vmctx, 1));
    try std.testing.expectEqual(@as(u32, 2), exporter.vmctx.memory_pages);
    try std.testing.expectEqual(@as(u32, 2), importer.vmctx.memory_pages);
    try std.testing.expectEqual(@as(usize, 2 * types.MemoryInstance.page_size), importer.vmctx.memory_size);

    const grown_offset = types.MemoryInstance.page_size;
    const importer_mem: [*]u8 = @ptrFromInt(importer.vmctx.memory_base);
    try std.testing.expectEqual(@as(u8, 0), importer_mem[grown_offset]);
    importer_mem[grown_offset] = 0x5A;

    const exporter_mem: [*]u8 = @ptrFromInt(exporter.vmctx.memory_base);
    try std.testing.expectEqual(@as(u8, 0x5A), exporter_mem[grown_offset]);

    try std.testing.expectEqual(@as(i32, 2), memGrowHelper(&importer.vmctx, 1));
    try std.testing.expectEqual(@as(u32, 3), exporter.vmctx.memory_pages);
    try std.testing.expectEqual(@as(u32, 3), importer.vmctx.memory_pages);
}

test "instantiate: module with table" {
    const tbl_types = [_]types.TableType{
        .{ .elem_type = .funcref, .limits = .{ .min = 10, .max = 100 } },
    };
    const module = aot_loader.AotModule{ .tables = &tbl_types };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 1), inst.tables.len);
    try std.testing.expectEqual(@as(usize, 10), inst.tables[0].elements.len);
    // All elements should be null-initialized
    for (inst.tables[0].elements) |elem| {
        try std.testing.expect(elem.isNull());
    }
}

test "#660 item 4: borrowed memory overrides are retained until importer destroy" {
    const mem_types = [_]types.MemoryType{
        .{ .limits = .{ .min = 1, .max = 4 } },
    };
    const data = [_]u8{0xa5};
    const data_segments = [_]aot_loader.AotDataSegment{
        .{ .memory_idx = 0, .offset = 7, .data = &data },
    };
    const exporter_module = aot_loader.AotModule{
        .memories = &mem_types,
        .data_segments = &data_segments,
    };
    const imported = [_]aot_loader.ImportedMemoryDesc{
        .{ .module_name = "env", .name = "mem", .min = 1, .max = 4 },
    };
    const importer_module = aot_loader.AotModule{ .imported_memories = &imported };

    const exporter_inst = try instantiate(&exporter_module, std.testing.allocator);
    defer destroy(exporter_inst);
    try std.testing.expectEqual(@as(usize, 1), exporter_inst.memories[0].referenceCount());
    try std.testing.expectEqual(@as(u8, 0xa5), exporter_inst.memories[0].data[7]);

    const overrides = [_]?*types.MemoryInstance{exporter_inst.memories[0]};
    const importer_inst = try instantiateWithOverrides(&importer_module, std.testing.allocator, &.{}, &overrides, &.{}, &.{}, &.{});
    try std.testing.expectEqual(exporter_inst.memories[0], importer_inst.memories[0]);
    try std.testing.expect(!importer_inst.memories_owned[0]);
    try std.testing.expectEqual(@as(usize, 2), exporter_inst.memories[0].referenceCount());

    destroy(importer_inst);
    try std.testing.expectEqual(@as(usize, 1), exporter_inst.memories[0].referenceCount());

    const vmctx = VmCtx{
        .memory_base = @intFromPtr(exporter_inst.memories[0].data.ptr),
        .memory_size = @as(usize, exporter_inst.memories[0].current_pages) * types.MemoryInstance.page_size,
    };
    const memory: [*]u8 = @ptrFromInt(vmctx.memory_base);
    try std.testing.expectEqual(@as(u8, 0xa5), memory[7]);
}

test "#660 item 4: borrowed table overrides are retained until importer destroy" {
    const tbl_types = [_]types.TableType{
        .{ .elem_type = .funcref, .limits = .{ .min = 2, .max = 2 } },
    };
    const exporter_module = aot_loader.AotModule{ .tables = &tbl_types };
    const imported = [_]aot_loader.ImportedTableDesc{
        .{ .module_name = "env", .name = "tbl", .elem_type = .funcref, .min = 2, .max = 2 },
    };
    const importer_module = aot_loader.AotModule{ .imported_tables = &imported };

    const exporter_inst = try instantiate(&exporter_module, std.testing.allocator);
    defer destroy(exporter_inst);
    exporter_inst.tables[0].elements[1] = .{ .value = .{ .funcref = 42 } };
    try std.testing.expectEqual(@as(usize, 1), exporter_inst.tables[0].referenceCount());

    const overrides = [_]?*types.TableInstance{exporter_inst.tables[0]};
    const importer_inst = try instantiateWithOverrides(&importer_module, std.testing.allocator, &overrides, &.{}, &.{}, &.{}, &.{});
    try std.testing.expectEqual(exporter_inst.tables[0], importer_inst.tables[0]);
    try std.testing.expect(!importer_inst.tables_owned[0]);
    try std.testing.expectEqual(@as(usize, 2), exporter_inst.tables[0].referenceCount());

    destroy(importer_inst);
    try std.testing.expectEqual(@as(usize, 1), exporter_inst.tables[0].referenceCount());
    try std.testing.expectEqual(@as(?u32, 42), exporter_inst.tables[0].elements[1].value.funcref);
}

test "getFuncAddr: returns null without text section" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(?[*]const u8, null), getFuncAddr(inst, 0));
}

test "getFuncAddr: returns correct address" {
    const text = [_]u8{ 0xCC, 0x90, 0xC3, 0x55, 0x48, 0x89, 0xE5, 0xC3 };
    const offsets = [_]u32{ 0, 3 };
    const module = aot_loader.AotModule{
        .text_section = &text,
        .func_offsets = &offsets,
        .func_count = 2,
    };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    const addr0 = getFuncAddr(inst, 0);
    try std.testing.expect(addr0 != null);
    try std.testing.expectEqual(@as(u8, 0xCC), addr0.?[0]);

    const addr1 = getFuncAddr(inst, 1);
    try std.testing.expect(addr1 != null);
    try std.testing.expectEqual(@as(u8, 0x55), addr1.?[0]);

    // Out of range
    try std.testing.expectEqual(@as(?[*]const u8, null), getFuncAddr(inst, 99));
}

test "VmCtx layout: fields at expected offsets" {
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(VmCtx, "memory_base"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(VmCtx, "memory_size"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(VmCtx, "globals_ptr"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(VmCtx, "host_functions_ptr"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(VmCtx, "memory_max_size"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(VmCtx, "func_table_ptr"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(VmCtx, "globals_count"));
    try std.testing.expectEqual(@as(usize, 52), @offsetOf(VmCtx, "host_functions_count"));
    try std.testing.expectEqual(@as(usize, 56), @offsetOf(VmCtx, "memory_pages"));
    try std.testing.expectEqual(@as(usize, 60), @offsetOf(VmCtx, "func_table_len"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(VmCtx, "mem_grow_fn"));
    try std.testing.expectEqual(@as(usize, 72), @offsetOf(VmCtx, "instance_ptr"));
    try std.testing.expectEqual(@as(usize, 80), @offsetOf(VmCtx, "trap_oob_fn"));
    try std.testing.expectEqual(@as(usize, 88), @offsetOf(VmCtx, "trap_unreachable_fn"));
    try std.testing.expectEqual(@as(usize, 120), @offsetOf(VmCtx, "funcptrs_ptr"));
    try std.testing.expectEqual(@as(usize, 128), @offsetOf(VmCtx, "table_grow_fn"));
    try std.testing.expectEqual(@as(usize, 136), @offsetOf(VmCtx, "tables_info_ptr"));
    try std.testing.expectEqual(@as(usize, 144), @offsetOf(VmCtx, "table_init_fn"));
    try std.testing.expectEqual(@as(usize, 152), @offsetOf(VmCtx, "elem_drop_fn"));
    try std.testing.expectEqual(@as(usize, 160), @offsetOf(VmCtx, "sig_table_ptr"));
    try std.testing.expectEqual(@as(usize, 168), @offsetOf(VmCtx, "func_sig_ids_ptr"));
    try std.testing.expectEqual(@as(usize, 176), @offsetOf(VmCtx, "ptr_to_sig_ptr"));
    try std.testing.expectEqual(@as(usize, 184), @offsetOf(VmCtx, "ptr_to_sig_len"));
    try std.testing.expectEqual(@as(usize, 192), @offsetOf(VmCtx, "table_set_fn"));
    try std.testing.expectEqual(@as(usize, 200), @offsetOf(VmCtx, "futex_wait32_fn"));
    try std.testing.expectEqual(@as(usize, 208), @offsetOf(VmCtx, "futex_wait64_fn"));
    try std.testing.expectEqual(@as(usize, 216), @offsetOf(VmCtx, "futex_notify_fn"));
    try std.testing.expectEqual(@as(usize, 224), @offsetOf(VmCtx, "mem_fill_fn"));
    try std.testing.expectEqual(@as(usize, 232), @offsetOf(VmCtx, "mem_copy_fn"));
    try std.testing.expectEqual(@as(usize, 240), @offsetOf(VmCtx, "tags_ptr"));
    try std.testing.expectEqual(@as(usize, 248), @offsetOf(VmCtx, "tags_count"));
    try std.testing.expectEqual(@as(usize, 256), @offsetOf(VmCtx, "aot_throw_uncaught_fn"));
    try std.testing.expectEqual(@as(usize, 264), @offsetOf(VmCtx, "exception_params"));
    try std.testing.expectEqual(@as(usize, 392), @offsetOf(VmCtx, "exception_param_count"));
    try std.testing.expectEqual(@as(usize, 400), @offsetOf(VmCtx, "wasi_ctx"));
    try std.testing.expectEqual(@as(usize, 408), @offsetOf(VmCtx, "lazy_compile_fn"));
    try std.testing.expectEqual(@as(usize, 416), @offsetOf(VmCtx, "thread_context"));
    try std.testing.expectEqual(@as(usize, 424), @sizeOf(VmCtx));
}

test "AOT thread context inherits retained process state without thread-local flags" {
    const Tracker = struct {
        refs: usize = 1,

        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.refs += 1;
        }

        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.refs -= 1;
        }
    };
    const ops = execution_context.ProcessStateOps{
        .retain = Tracker.retain,
        .release = Tracker.release,
    };
    var tracker = Tracker{};
    const root_ref = execution_context.ProcessStateRef.init(@ptrCast(&tracker), &ops);
    var module = aot_loader.AotModule{};

    const parent = try instantiate(&module, std.testing.allocator);
    parent.attachProcessState(root_ref);
    parent.thread_context.configureWasiThread(1, 0x1111, null);
    parent.thread_context.setTlsBase(0x2000);
    parent.thread_context.requestCancellation();
    parent.thread_context.markTrap();

    const child = try instantiate(&module, std.testing.allocator);
    child.inheritProcessStateFrom(parent);
    child.thread_context.configureWasiThread(2, 0x2222, null);
    child.thread_context.setTlsBase(0x3000);

    try std.testing.expectEqual(@as(usize, 3), tracker.refs);
    try std.testing.expectEqual(parent.vmctx.wasi_ctx, child.vmctx.wasi_ctx);
    try std.testing.expectEqual(
        @intFromPtr(&parent.thread_context),
        parent.vmctx.thread_context,
    );
    try std.testing.expectEqual(
        @intFromPtr(&child.thread_context),
        child.vmctx.thread_context,
    );
    try std.testing.expect(parent.thread_context.isCancellationRequested());
    try std.testing.expect(!child.thread_context.isCancellationRequested());
    try std.testing.expect(parent.thread_context.hasTrapped());
    try std.testing.expect(!child.thread_context.hasTrapped());

    destroy(parent);
    root_ref.release();
    try std.testing.expectEqual(@as(usize, 1), tracker.refs);
    try std.testing.expectEqual(
        @as(*Tracker, @ptrCast(@alignCast(child.thread_context.process_state.?.ptr))),
        &tracker,
    );
    destroy(child);
    try std.testing.expectEqual(@as(usize, 0), tracker.refs);
}

test "AOT thread clone shares code memory and tables while isolating globals and VmCtx" {
    if (comptime !config.lib_wasi_threads or !can_execute_native)
        return error.SkipZigTest;

    const Tracker = struct {
        refs: usize = 1,

        fn retain(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.refs += 1;
        }

        fn release(raw: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            std.debug.assert(self.refs > 0);
            self.refs -= 1;
        }
    };
    const process_ops = execution_context.ProcessStateOps{
        .retain = Tracker.retain,
        .release = Tracker.release,
    };
    const text = switch (builtin.cpu.arch) {
        .x86_64 => &[_]u8{0xC3},
        .aarch64 => &[_]u8{ 0xC0, 0x03, 0x5F, 0xD6 },
        else => unreachable,
    };
    const offsets = [_]u32{0};
    const type_indices = [_]u32{0};
    const func_types = [_]aot_loader.AotFuncType{.{
        .params = &.{},
        .results = &.{},
    }};
    const memories = [_]types.MemoryType{.{
        .limits = .{ .min = 1, .max = 2 },
        .is_shared = true,
    }};
    const tables = [_]types.TableType{.{
        .elem_type = .funcref,
        .limits = .{ .min = 1, .max = 2 },
    }};
    const globals = [_]aot_loader.AotGlobalInit{.{
        .val_type = @intFromEnum(types.ValType.i32),
        .mutability = 1,
        .init_i64 = 17,
    }};
    const tag_types = [_]u32{0};
    const module = aot_loader.AotModule{
        .text_section = text,
        .func_offsets = &offsets,
        .local_func_type_indices = &type_indices,
        .func_count = 1,
        .func_types = &func_types,
        .memories = &memories,
        .tables = &tables,
        .global_inits = &globals,
        .tag_types = &tag_types,
    };

    var tracker = Tracker{};
    const root_ref =
        execution_context.ProcessStateRef.init(@ptrCast(&tracker), &process_ops);
    const parent = try instantiate(&module, std.testing.allocator);
    try mapCodeExecutable(parent);
    parent.attachProcessState(root_ref);
    parent.thread_context.configureWasiThread(1, 0x1111, null);
    parent.thread_context.requestCancellation();

    var active_globals = [_]u128{0};
    writeGlobalsToStorage(parent, std.mem.sliceAsBytes(&active_globals));
    std.mem.writeInt(
        i32,
        std.mem.sliceAsBytes(&active_globals)[0..4],
        99,
        .little,
    );
    parent.vmctx.globals_ptr = @intFromPtr(&active_globals);
    const child = try cloneForThread(parent, std.testing.allocator);
    parent.vmctx.globals_ptr = 0;
    child.thread_context.configureWasiThread(2, 0x2222, null);

    try std.testing.expectEqual(parent.memories[0], child.memories[0]);
    try std.testing.expectEqual(parent.tables[0], child.tables[0]);
    try std.testing.expectEqual(parent.tags[0], child.tags[0]);
    try std.testing.expectEqual(@as(usize, 2), parent.memories[0].referenceCount());
    try std.testing.expectEqual(@as(usize, 2), parent.tables[0].referenceCount());
    try std.testing.expect(parent.globals[0] != child.globals[0]);
    try std.testing.expectEqual(@as(i32, 17), parent.globals[0].value.i32);
    try std.testing.expectEqual(@as(i32, 99), child.globals[0].value.i32);
    try std.testing.expect(&parent.vmctx != &child.vmctx);
    try std.testing.expectEqual(parent.vmctx.memory_base, child.vmctx.memory_base);
    try std.testing.expectEqual(parent.code_base, child.code_base);
    try std.testing.expectEqual(parent.funcptrs[0], child.funcptrs[0]);
    try std.testing.expectEqual(@as(usize, 2), parent.code_mapping.?.referenceCount());
    try std.testing.expectEqual(
        @as(usize, 2),
        parent.tables[0].vmctx_subscribers.items.len,
    );
    try std.testing.expectEqual(@as(usize, 3), tracker.refs);
    try std.testing.expect(parent.thread_context.isCancellationRequested());
    try std.testing.expect(!child.thread_context.isCancellationRequested());

    try std.testing.expectEqual(@as(i32, 1), tableGrowHelper(&parent.vmctx, 0, 1, 0));
    try std.testing.expectEqual(@as(u32, 2), parent.vmctx.func_table_len);
    try std.testing.expectEqual(@as(u32, 2), child.vmctx.func_table_len);
    try std.testing.expectEqual(parent.vmctx.func_table_ptr, child.vmctx.func_table_ptr);

    var no_results: [0]ScalarResult = .{};
    _ = try callFuncScalar(child, 0, &.{}, &.{}, &.{}, &no_results);
    destroy(child);
    try std.testing.expectEqual(@as(usize, 1), parent.memories[0].referenceCount());
    try std.testing.expectEqual(@as(usize, 1), parent.tables[0].referenceCount());
    try std.testing.expectEqual(
        @as(usize, 1),
        parent.tables[0].vmctx_subscribers.items.len,
    );
    try std.testing.expectEqual(@as(usize, 1), parent.code_mapping.?.referenceCount());
    try std.testing.expectEqual(@as(usize, 2), tracker.refs);
    destroy(parent);
    root_ref.release();
    try std.testing.expectEqual(@as(usize, 0), tracker.refs);
}

const AotCloneRollbackTracker = struct {
    refs: usize = 1,

    fn retain(raw: *anyopaque) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        self.refs += 1;
    }

    fn release(raw: *anyopaque) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        std.debug.assert(self.refs > 0);
        self.refs -= 1;
    }
};

fn exerciseAotCloneAllocation(
    failing_allocator: std.mem.Allocator,
    parent: *AotInstance,
) !void {
    const child = cloneForThread(parent, failing_allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return err,
    };
    destroy(child);
}

test "AOT thread clone rolls back every partial retain and allocation" {
    if (comptime !config.lib_wasi_threads or !can_execute_native)
        return error.SkipZigTest;

    const text = switch (builtin.cpu.arch) {
        .x86_64 => &[_]u8{0xC3},
        .aarch64 => &[_]u8{ 0xC0, 0x03, 0x5F, 0xD6 },
        else => unreachable,
    };
    const offsets = [_]u32{0};
    const type_indices = [_]u32{0};
    const func_types = [_]aot_loader.AotFuncType{.{
        .params = &.{},
        .results = &.{},
    }};
    const memories = [_]types.MemoryType{.{
        .limits = .{ .min = 1, .max = 2 },
        .is_shared = true,
    }};
    const tables = [_]types.TableType{.{
        .elem_type = .funcref,
        .limits = .{ .min = 1, .max = 2 },
    }};
    const globals = [_]aot_loader.AotGlobalInit{.{
        .val_type = @intFromEnum(types.ValType.i32),
        .mutability = 1,
        .init_i64 = 7,
    }};
    const module = aot_loader.AotModule{
        .text_section = text,
        .func_offsets = &offsets,
        .local_func_type_indices = &type_indices,
        .func_count = 1,
        .func_types = &func_types,
        .memories = &memories,
        .tables = &tables,
        .global_inits = &globals,
    };
    const process_ops = execution_context.ProcessStateOps{
        .retain = AotCloneRollbackTracker.retain,
        .release = AotCloneRollbackTracker.release,
    };
    var tracker = AotCloneRollbackTracker{};
    const root_ref =
        execution_context.ProcessStateRef.init(@ptrCast(&tracker), &process_ops);
    const parent = try instantiate(&module, std.testing.allocator);
    defer destroy(parent);
    try mapCodeExecutable(parent);
    parent.attachProcessState(root_ref);

    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        exerciseAotCloneAllocation,
        .{parent},
    );
    try std.testing.expectEqual(@as(usize, 1), parent.memories[0].referenceCount());
    try std.testing.expectEqual(@as(usize, 1), parent.tables[0].referenceCount());
    try std.testing.expectEqual(@as(usize, 1), parent.code_mapping.?.referenceCount());
    try std.testing.expectEqual(@as(usize, 2), tracker.refs);
    root_ref.release();
}

test "AOT execution context: no-WASI instance keeps process pointer null" {
    var module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expect(inst.thread_context.process_state == null);
    try std.testing.expectEqual(@as(usize, 0), inst.vmctx.wasi_ctx);
    try std.testing.expectEqual(
        @intFromPtr(&inst.thread_context),
        inst.vmctx.thread_context,
    );
}

test "AOT execution context keeps trap decode bookkeeping per native thread" {
    if (builtin.single_threaded) return error.SkipZigTest;

    const Worker = struct {
        fn run(
            thread_ctx: *execution_context.ThreadExecutionContext,
            marker: usize,
            ready: *std.atomic.Value(u32),
            go: *std.atomic.Value(bool),
            observed: *usize,
        ) void {
            var active_scope = thread_ctx.enter();
            defer active_scope.deinit();
            var call_state = AotCallState{};
            call_state.trap_decode.code_base = marker;
            var backend_scope = thread_ctx.bindBackendContext(@ptrCast(&call_state));
            defer backend_scope.deinit();
            _ = ready.fetchAdd(1, .acq_rel);
            while (!go.load(.acquire)) std.atomic.spinLoopHint();
            observed.* = activeAotCallState().?.trap_decode.code_base;
        }
    };

    var first_ctx = execution_context.ThreadExecutionContext{};
    var second_ctx = execution_context.ThreadExecutionContext{};
    var ready = std.atomic.Value(u32).init(0);
    var go = std.atomic.Value(bool).init(false);
    var first_observed: usize = 0;
    var second_observed: usize = 0;
    const first = try std.Thread.spawn(
        .{},
        Worker.run,
        .{ &first_ctx, 0x1111, &ready, &go, &first_observed },
    );
    const second = try std.Thread.spawn(
        .{},
        Worker.run,
        .{ &second_ctx, 0x2222, &ready, &go, &second_observed },
    );
    while (ready.load(.acquire) != 2) std.atomic.spinLoopHint();
    go.store(true, .release);
    first.join();
    second.join();

    try std.testing.expectEqual(@as(usize, 0x1111), first_observed);
    try std.testing.expectEqual(@as(usize, 0x2222), second_observed);
    try std.testing.expect(execution_context.current() == null);
}

test "getFuncAddr: import indices return null" {
    const text = [_]u8{ 0xCC, 0x90, 0xC3, 0x55, 0x48, 0x89, 0xE5, 0xC3 };
    const offsets = [_]u32{ 0, 3 };
    const module = aot_loader.AotModule{
        .text_section = &text,
        .func_offsets = &offsets,
        .func_count = 2,
        .import_function_count = 2,
    };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    // Import indices (0, 1) should return null (no native code)
    try std.testing.expectEqual(@as(?[*]const u8, null), getFuncAddr(inst, 0));
    try std.testing.expectEqual(@as(?[*]const u8, null), getFuncAddr(inst, 1));

    // Local indices (2, 3) map to func_offsets[0], func_offsets[1]
    const addr2 = getFuncAddr(inst, 2);
    try std.testing.expect(addr2 != null);
    try std.testing.expectEqual(@as(u8, 0xCC), addr2.?[0]);

    const addr3 = getFuncAddr(inst, 3);
    try std.testing.expect(addr3 != null);
    try std.testing.expectEqual(@as(u8, 0x55), addr3.?[0]);
}

test "resolveHostFunctions: resolves WASI imports" {
    const imports = [_]aot_loader.AotImportDesc{
        .{ .module_name = "wasi_snapshot_preview1", .field_name = "fd_write", .kind = .function, .func_type_idx = 0 },
        .{ .module_name = "wasi_snapshot_preview1", .field_name = "clock_time_get", .kind = .function, .func_type_idx = 1 },
        .{ .module_name = "env", .field_name = "some_func", .kind = .function, .func_type_idx = 2 },
    };
    const module = aot_loader.AotModule{
        .import_function_count = 3,
        .imports = &imports,
    };
    const result = try resolveHostFunctions(&module, std.testing.allocator);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expect(result[0] != null); // fd_write resolved
    try std.testing.expect(result[1] != null); // clock_time_get resolved
    try std.testing.expect(result[2] == null); // env.some_func not resolved
}

test "instantiate: module with WASI imports resolves host functions" {
    const imports = [_]aot_loader.AotImportDesc{
        .{ .module_name = "wasi_snapshot_preview1", .field_name = "fd_write", .kind = .function, .func_type_idx = 0 },
    };
    const module = aot_loader.AotModule{
        .import_function_count = 1,
        .imports = &imports,
    };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 1), inst.host_functions.len);
    try std.testing.expect(inst.host_functions[0] != null);
}

// ─── #857: JitCodeCache registry tests ─────────────────────────────────────

/// Minimal valid native function body for the host arch: a handful of
/// `nop`s followed by `ret`/`RET`, immediately returning. Enough to
/// exercise `mapCodeExecutable`'s full mmap → memcpy → icache-flush →
/// mprotect → call round trip without needing real codegen output —
/// this suite is testing the `JitCodeCache` registry's bookkeeping,
/// not codegen. Padded to a few bytes (rather than a single `ret`) so
/// the budget-rejection test below has room to pick a budget that's
/// unambiguously nonzero yet still smaller than this body's size —
/// with a 1-byte body and a 0-baseline, no such value exists, since
/// `0` doubles as the registry's "unlimited" sentinel.
const minimal_ret_body: []const u8 = switch (native_arch) {
    .x86_64 => &[_]u8{ 0x90, 0x90, 0x90, 0x90, 0xC3 }, // nop*4; ret
    .aarch64 => &[_]u8{ 0x1F, 0x20, 0x03, 0xD5, 0xC0, 0x03, 0x5F, 0xD6 }, // nop; ret
    .unsupported => &[_]u8{},
};

test "JitCodeCache: tracked eager code mapping round-trip leaves no residual mapping" {
    if (comptime !can_execute_native) return error.SkipZigTest;

    const before = JitCodeCache.residentBytes();
    const offsets = [_]u32{0};
    const module = aot_loader.AotModule{
        .text_section = minimal_ret_body,
        .func_offsets = &offsets,
        .func_count = 1,
    };
    const inst = try instantiate(&module, std.testing.allocator);
    try mapCodeExecutable(inst);

    try std.testing.expectEqual(before + minimal_ret_body.len, JitCodeCache.residentBytes());

    var results_buf: [0]ScalarResult = .{};
    _ = try callFuncScalar(inst, 0, &.{}, &.{}, &.{}, &results_buf);

    destroy(inst);
    try std.testing.expectEqual(before, JitCodeCache.residentBytes());
}

test "JitCodeCache: repeated eager mapping cycles never accumulate residual mappings" {
    if (comptime !can_execute_native) return error.SkipZigTest;

    const before_count = JitCodeCache.mappingCount();
    const before_bytes = JitCodeCache.residentBytes();

    // #857 acceptance: "JIT-compiles + runs + drops N small modules in
    // a loop in one process ... mapped code doesn't grow unbounded."
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        const offsets = [_]u32{0};
        const module = aot_loader.AotModule{
            .text_section = minimal_ret_body,
            .func_offsets = &offsets,
            .func_count = 1,
        };
        const inst = try instantiate(&module, std.testing.allocator);
        try mapCodeExecutable(inst);

        // Never more than one live mapping from this loop at a time —
        // this is the "doesn't grow unbounded" assertion.
        try std.testing.expectEqual(before_count + 1, JitCodeCache.mappingCount());
        try std.testing.expectEqual(before_bytes + minimal_ret_body.len, JitCodeCache.residentBytes());

        var results_buf: [0]ScalarResult = .{};
        _ = try callFuncScalar(inst, 0, &.{}, &.{}, &.{}, &results_buf);

        destroy(inst);
        // Back to baseline after every single drop — no accumulation.
        try std.testing.expectEqual(before_count, JitCodeCache.mappingCount());
        try std.testing.expectEqual(before_bytes, JitCodeCache.residentBytes());
    }
}

test "JitCodeCache: tracked eager mapping rejects a configured budget overrun" {
    if (comptime !can_execute_native) return error.SkipZigTest;

    // Budget smaller than the minimal function body guarantees rejection
    // regardless of the current residual baseline from other tests.
    // Must stay nonzero (0 doubles as the registry's "unlimited"
    // sentinel) — `minimal_ret_body` is padded specifically so this
    // value (baseline + 1) is always both nonzero and still short of
    // baseline + the body's full size.
    JitCodeCache.budget_bytes = JitCodeCache.residentBytes() + 1;
    defer JitCodeCache.budget_bytes = 0; // restore "unlimited" for later tests

    const offsets = [_]u32{0};
    const module = aot_loader.AotModule{
        .text_section = minimal_ret_body,
        .func_offsets = &offsets,
        .func_count = 1,
    };
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectError(error.CodeBudgetExceeded, mapCodeExecutable(inst));
    // Rejected before any mmap happened — no residual mapping, and the
    // instance's code_base stays unset so `destroy` above is a clean no-op
    // on the code-mapping front.
    try std.testing.expectEqual(@as(?[*]const u8, null), inst.code_base);
}
