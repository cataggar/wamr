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
const platform = @import("../../platform/platform.zig");
const sig_registry = @import("../common/sig_registry.zig");

// ─── Windows crash handler (debug only) ─────────────────────────────────────
const windows = std.os.windows;

var g_code_base: usize = 0;
var g_code_size: usize = 0;
var g_mem_base: usize = 0;
var g_mem_size: usize = 0;

// ── Trap-as-error plumbing (Windows x86_64 only) ─────────────────────
//
// Acts like setjmp/longjmp. `callFuncScalar` calls `RtlCaptureContext`
// immediately before invoking generated code; if a trap occurs (OOB
// via `aotTrapOOB`, or any access violation / illegal instruction /
// divide-by-zero / stack overflow caught by `vehHandler`), we set
// `g_trap_occurred` and use `RtlRestoreContext` to resume execution
// at the capture site. The post-capture check of `g_trap_occurred`
// then returns `error.WasmTrap` out of `callFuncScalar`.
//
// Not thread-safe — but neither is the rest of this runtime.
// Not thread-safe (neither is the rest of this runtime). Using a module-level
// var rather than threadlocal so Windows TLS alignment quirks don't bite us.
var g_saved_ctx: windows.CONTEXT align(16) = undefined;
var g_trap_catching: bool = false;
var g_trap_occurred: bool = false;
/// Exception code of the last fault vehHandler redirected to trapLongjmp.
/// Sampled by `callFuncScalar` after trap return to decide whether to
/// re-arm the thread's stack guard page (see `resetStackGuardPage`).
var g_last_trap_code: u32 = 0;

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
    @atomicStore(bool, &g_trap_occurred, true, .seq_cst);
    if (comptime builtin.os.tag == .windows) {
        RtlRestoreContext(&g_saved_ctx, null);
    }
    // Non-Windows: trap-as-error not yet supported; fall back to exit.
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
    const rip: usize = @intCast(ctx.Rip);
    const in_code = g_code_base != 0 and g_code_size != 0 and
        rip >= g_code_base and rip < g_code_base + g_code_size;
    // If armed, redirect any wasm-like fault to trapLongjmp. We used to
    // only redirect when RIP was inside the generated code, but a null
    // table entry in call_indirect causes RIP=0 at fault time (the
    // `call r11` has already transferred control), so the fault site is
    // outside the code region. The armed check is sufficient to
    // distinguish wasm traps from unrelated process-wide faults.
    if (is_wasm_fault and @atomicLoad(bool, &g_trap_catching, .seq_cst)) {
        _ = in_code;
        @atomicStore(bool, &g_trap_occurred, true, .seq_cst);
        @atomicStore(u32, &g_last_trap_code, code, .seq_cst);
        if (code == 0xC00000FD) {
            // Stack overflow: restore the full saved context so we
            // resume at the RtlCaptureContext site on a healthy stack.
            // Using trapLongjmp here is fragile because it would run
            // on the nearly-exhausted overflowed stack.
            ctx.* = g_saved_ctx;
        } else {
            ctx.Rip = @intFromPtr(&trapLongjmp);
        }
        return -1; // EXCEPTION_CONTINUE_EXECUTION
    }
    if (rec.ExceptionCode == 0xC0000005) { // STATUS_ACCESS_VIOLATION
        const fault: usize = @intCast(rec.ExceptionInformation[1]);
        std.debug.print(
            "\n=== VEH CRASH === RIP=0x{x} (code+0x{x}) fault=0x{x}",
            .{ rip, rip -% g_code_base, fault },
        );
        if (g_mem_base != 0 and fault >= g_mem_base and fault < g_mem_base +% g_mem_size) {
            std.debug.print(" (wasm mem[0x{x}])", .{fault - g_mem_base});
        } else if (g_mem_base != 0) {
            const delta: isize = @as(isize, @bitCast(fault)) - @as(isize, @bitCast(g_mem_base));
            std.debug.print(" (mem_base+0x{x} delta={d})", .{ fault -% g_mem_base, delta });
        }
        std.debug.print("\n", .{});
        std.debug.print("RAX=0x{x} RCX=0x{x} RDX=0x{x} RBX=0x{x}\n", .{ ctx.Rax, ctx.Rcx, ctx.Rdx, ctx.Rbx });
        std.debug.print("RSI=0x{x} RDI=0x{x} RBP=0x{x} RSP=0x{x}\n", .{ ctx.Rsi, ctx.Rdi, ctx.Rbp, ctx.Rsp });
        std.debug.print("R8=0x{x} R9=0x{x} R10=0x{x} R11=0x{x}\n", .{ ctx.R8, ctx.R9, ctx.R10, ctx.R11 });
        std.debug.print("R12=0x{x} R13=0x{x} R14=0x{x} R15=0x{x}\n", .{ ctx.R12, ctx.R13, ctx.R14, ctx.R15 });
        if (g_code_base != 0 and g_code_size != 0) {
            const rip_off: usize = rip -% g_code_base;
            if (rip_off < g_code_size) {
                const start: usize = if (rip_off > 32) rip_off - 32 else 0;
                const end: usize = @min(rip_off + 16, g_code_size);
                const p: [*]const u8 = @ptrFromInt(g_code_base + start);
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
    /// Pointer to flat globals values array (each global is an i64 at index * 8).
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
/// Module-level storage populated by callFunc so the trap helper can map
/// a return address back to a function index (purely diagnostic).
var g_func_offsets: []const u32 = &.{};
var g_veh_installed: bool = false;

pub fn aotTrapOOB(vmctx: *VmCtx) callconv(.c) noreturn {
    const ret_addr: usize = @returnAddress();
    const code_off_s: isize = @as(isize, @bitCast(ret_addr)) - @as(isize, @bitCast(g_code_base));
    const code_off: usize = if (code_off_s >= 0) @intCast(code_off_s) else 0;
    var func_idx: isize = -1;
    for (g_func_offsets, 0..) |off, idx| {
        if (off <= code_off) func_idx = @intCast(idx) else break;
    }
    if (@atomicLoad(bool, &g_trap_catching, .seq_cst)) {
        // Caller has armed trap-as-error; unwind instead of exiting.
        trapLongjmp();
    }
    // Flush any buffered stdout from the guest before we tear down the
    // process so user-visible output isn't lost. Best-effort.
    std.debug.print(
        "wasm trap: out of bounds memory access (code+0x{x}, local_func[{d}], mem_size=0x{x})\n",
        .{ code_off, func_idx, vmctx.memory_size },
    );
    std.process.exit(2);
}

/// Host helper invoked from AOT-compiled code for `unreachable`,
/// integer divide-by-zero, INT_MIN/-1 overflow, and invalid float→int
/// conversion. When the caller has armed trap-as-error
/// (`g_trap_catching` true), longjmps back to `callFuncScalar` which
/// returns `error.WasmTrap`. Otherwise prints a diagnostic and exits.
pub fn aotTrapUnreachable(vmctx: *VmCtx) callconv(.c) noreturn {
    _ = vmctx;
    if (@atomicLoad(bool, &g_trap_catching, .seq_cst)) trapLongjmp();
    std.debug.print("wasm trap: unreachable\n", .{});
    std.process.exit(2);
}

pub fn aotTrapIntDivZero(vmctx: *VmCtx) callconv(.c) noreturn {
    _ = vmctx;
    if (@atomicLoad(bool, &g_trap_catching, .seq_cst)) trapLongjmp();
    std.debug.print("wasm trap: integer divide by zero\n", .{});
    std.process.exit(2);
}

pub fn aotTrapIntOverflow(vmctx: *VmCtx) callconv(.c) noreturn {
    _ = vmctx;
    if (@atomicLoad(bool, &g_trap_catching, .seq_cst)) trapLongjmp();
    std.debug.print("wasm trap: integer overflow\n", .{});
    std.process.exit(2);
}

pub fn aotTrapInvalidConversion(vmctx: *VmCtx) callconv(.c) noreturn {
    _ = vmctx;
    if (@atomicLoad(bool, &g_trap_catching, .seq_cst)) trapLongjmp();
    std.debug.print("wasm trap: invalid conversion to integer\n", .{});
    std.process.exit(2);
}

// ── Futex helpers for atomic.wait / atomic.notify ────────────────────

/// Host helper for `memory.atomic.wait32`.
/// Blocks the calling thread if `mem[addr] == expected`.
/// Returns: 0 = woken by notify, 1 = not-equal, 2 = timed-out.
pub fn aotAtomicWait32(vmctx: *VmCtx, addr: u32, expected: u32, timeout_lo: u32, timeout_hi: u32) callconv(.c) i32 {
    const timeout_ns: i64 = @bitCast(@as(u64, timeout_hi) << 32 | @as(u64, timeout_lo));
    if (vmctx.memory_base == 0) return 1;
    if (@as(u64, addr) + 4 > vmctx.memory_size) return 1;
    const mem: [*]u8 = @ptrFromInt(vmctx.memory_base);
    const current = std.mem.readInt(u32, @as(*const [4]u8, @ptrCast(mem + addr)), .little);
    if (current != expected) return 1; // not-equal

    // In single-threaded mode, no other thread can wake us, so we'd block
    // forever. Return timed-out (2) immediately for timeout >= 0, or
    // block forever for timeout == -1 (infinite).
    // TODO: with real threading, use OS futex to block.
    if (timeout_ns < 0) {
        // Infinite wait in single-threaded mode would deadlock.
        // Spec says this is valid behavior (trap or block indefinitely).
        return 2; // timed-out
    }
    return 2; // timed-out
}

/// Host helper for `memory.atomic.wait64`.
pub fn aotAtomicWait64(vmctx: *VmCtx, addr: u32, exp_lo: u32, exp_hi: u32, timeout_lo: u32, timeout_hi: u32) callconv(.c) i32 {
    const expected: u64 = @as(u64, exp_hi) << 32 | @as(u64, exp_lo);
    const timeout_ns: i64 = @bitCast(@as(u64, timeout_hi) << 32 | @as(u64, timeout_lo));
    if (vmctx.memory_base == 0) return 1;
    if (@as(u64, addr) + 8 > vmctx.memory_size) return 1;
    const mem: [*]u8 = @ptrFromInt(vmctx.memory_base);
    const current = std.mem.readInt(u64, @as(*const [8]u8, @ptrCast(mem + addr)), .little);
    if (current != expected) return 1;
    _ = timeout_ns;
    return 2; // timed-out (single-threaded)
}

/// Host helper for `memory.atomic.notify`.
/// Wakes up to `count` threads waiting on `mem[addr]`.
/// Returns the number of threads actually woken.
pub fn aotAtomicNotify(vmctx: *VmCtx, addr: u32, count: u32) callconv(.c) i32 {
    // In single-threaded mode, no threads are ever waiting.
    _ = vmctx;
    _ = addr;
    _ = count;
    return 0;
}

/// Host helper invoked from AOT-compiled `memory.grow` sites.
/// Grows `inst.memories[0]` by `delta_pages`, reallocating the host buffer
/// if needed, updates `vmctx` mirror fields (memory_base/size/pages) so
/// subsequent loads/stores see the new buffer, and returns the previous
/// page count. Returns -1 on failure (OOM or exceeds max).
pub fn memGrowHelper(vmctx: *VmCtx, delta_pages: i32) callconv(.c) i32 {
    if (vmctx.instance_ptr == 0) return -1;
    const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
    if (inst.memories.len == 0) return -1;
    const mem = inst.memories[0];
    const old_pages: u32 = mem.current_pages;
    if (delta_pages < 0) return -1;
    const delta: u32 = @intCast(delta_pages);
    const new_pages_u64: u64 = @as(u64, old_pages) + @as(u64, delta);
    const cap: u64 = @min(mem.max_pages, 65536);
    if (new_pages_u64 > cap) return -1;
    const new_pages: u32 = @intCast(new_pages_u64);
    const new_size: usize = @as(usize, new_pages) * types.MemoryInstance.page_size;
    if (new_size > mem.data.len) {
        const new_data = inst.allocator.realloc(mem.data, new_size) catch return -1;
        @memset(new_data[mem.data.len..], 0);
        mem.data = new_data;
    }
    mem.current_pages = new_pages;
    vmctx.memory_base = @intFromPtr(mem.data.ptr);
    vmctx.memory_size = @as(usize, new_pages) * types.MemoryInstance.page_size;
    vmctx.memory_pages = new_pages;
    if (comptime builtin.os.tag == .windows) {
        g_mem_base = vmctx.memory_base;
        g_mem_size = vmctx.memory_size;
    }
    return @intCast(old_pages);
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
        const old_size: u32 = @intCast(inst.func_table.len);
        const new_size_u64: u64 = @as(u64, old_size) + @as(u64, delta_u);
        const max_cap: u64 = blk: {
            if (inst.tables.len > 0) {
                break :blk inst.tables[0].table_type.limits.max orelse 0xFFFF_FFFF;
            } else break :blk 0xFFFF_FFFF;
        };
        if (new_size_u64 > max_cap) return -1;
        const new_size: usize = @intCast(new_size_u64);
        const shared0 = if (inst.tables.len > 0) inst.tables[0] else null;
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
    if (table_idx < inst.tables.len) {
        const shared = inst.tables[table_idx];
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

// ─── Instance ───────────────────────────────────────────────────────────────

pub const AotInstance = struct {
    module: *const aot_loader.AotModule,
    memories: []*types.MemoryInstance,
    tables: []*types.TableInstance,
    globals: []*types.GlobalInstance,
    allocator: std.mem.Allocator,
    /// Base address of the mapped executable code (null if not yet mapped).
    code_base: ?[*]const u8 = null,
    /// Size of the mapped executable region (for cleanup).
    code_size: usize = 0,
    /// Resolved AOT host function pointers (one per import).
    host_functions: []const ?*const anyopaque = &.{},
    /// Native function pointer table for call_indirect (one per module function).
    func_table: []usize = &.{},
    /// Native function pointer array indexed by module funcidx (imports + locals).
    /// Used by `ref.func` which must yield a function's native address even when
    /// the function was never placed in a wasm table by an element segment.
    funcptrs: []const usize = &.{},
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
};

// ─── Public API ─────────────────────────────────────────────────────────────

/// Instantiate an AOT module, producing a runnable AotInstance.
pub fn instantiate(module: *const aot_loader.AotModule, allocator: std.mem.Allocator) RuntimeError!*AotInstance {
    var inst = allocator.create(AotInstance) catch return error.OutOfMemory;
    errdefer allocator.destroy(inst);

    inst.* = .{
        .module = module,
        .memories = &.{},
        .tables = &.{},
        .globals = &.{},
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

    inst.memories = try allocateMemories(module, allocator);
    errdefer freeMemories(inst.memories, allocator);

    // Apply data segments to linear memory
    for (module.data_segments) |seg| {
        if (seg.memory_idx >= inst.memories.len) continue;
        const mem = inst.memories[seg.memory_idx];
        const end = @as(usize, seg.offset) + seg.data.len;
        if (end > mem.data.len) continue;
        @memcpy(mem.data[seg.offset..][0..seg.data.len], seg.data);
    }

    inst.tables = try allocateTables(module, allocator);
    errdefer freeTables(inst.tables, allocator);

    // If no tables but we have element segments, create a default table
    if (inst.tables.len == 0 and module.elem_segments.len > 0) {
        // Compute required table size from element segments
        var max_size: u32 = 0;
        for (module.elem_segments) |seg| {
            const end = seg.offset + @as(u32, @intCast(seg.func_indices.len));
            if (end > max_size) max_size = end;
        }
        if (max_size > 0) {
            const tables = allocator.alloc(*types.TableInstance, 1) catch return error.OutOfMemory;
            const elements = allocator.alloc(types.TableElement, max_size) catch return error.OutOfMemory;
            for (elements) |*e| e.* = types.TableElement.nullForType(.funcref);
            const tbl = allocator.create(types.TableInstance) catch return error.OutOfMemory;
            tbl.* = .{
                .table_type = .{ .elem_type = .funcref, .limits = .{ .min = max_size, .max = max_size } },
                .elements = elements,
            };
            tables[0] = tbl;
            inst.tables = tables;
        }
    }

    inst.globals = try allocateGlobals(module, allocator);
    errdefer freeGlobals(inst.globals, allocator);

    // Resolve AOT host functions for imports
    inst.host_functions = try resolveHostFunctions(module, allocator);

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

    return inst;
}

/// Destroy an AOT instance, freeing all allocated resources.
pub fn destroy(inst: *AotInstance) void {
    const allocator = inst.allocator;
    // Unmap executable code if mapped.
    if (inst.code_base) |base| {
        platform.munmap(@constCast(@ptrCast(base)), inst.code_size);
    }
    freeMemories(inst.memories, allocator);
    freeTables(inst.tables, allocator);
    freeGlobals(inst.globals, allocator);
    if (inst.host_functions.len > 0) allocator.free(inst.host_functions);
    // inst.func_table aliases tables[0].native_backing (freed by TableInstance.release).
    if (inst.funcptrs.len > 0) allocator.free(inst.funcptrs);
    if (inst.elem_segments_dropped.len > 0) allocator.free(inst.elem_segments_dropped);
    if (inst.sig_table.len > 0) allocator.free(inst.sig_table);
    if (inst.func_sig_ids.len > 0) allocator.free(inst.func_sig_ids);
    if (inst.ptr_to_sig.len > 0) allocator.free(inst.ptr_to_sig);
    allocator.destroy(inst);
}

/// Look up an exported function by name, returning its function index.
pub fn findExportFunc(inst: *const AotInstance, name: []const u8) ?u32 {
    for (inst.module.exports) |exp| {
        if (exp.kind == .function and std.mem.eql(u8, exp.name, name)) return exp.index;
    }
    return null;
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

// ─── Native execution ───────────────────────────────────────────────────────

/// Map the module's native code into executable memory.
/// After this call, `getFuncAddr` returns pointers suitable for execution.
pub fn mapCodeExecutable(inst: *AotInstance) RuntimeError!void {
    const module = inst.module;
    const text_opt = module.text_section;
    const has_text = text_opt != null and text_opt.?.len > 0;

    var mem: [*]u8 = undefined;
    if (has_text) {
        const text = text_opt.?;
        // 1. Allocate RW pages
        mem = platform.mmap(null, text.len, .{ .read = true, .write = true }, .{}) orelse
            return error.CodeMappingFailed;

        // 2. Copy native code
        @memcpy(mem[0..text.len], text);

        // 3. Flush instruction cache (required on AArch64, no-op on x86-64)
        if (comptime native_arch == .aarch64) {
            platform.icacheFlush(mem, text.len);
        }

        // 4. Transition to RX (W^X)
        platform.mprotect(mem, text.len, .{ .read = true, .exec = true }) catch
            return error.CodeMappingFailed;

        inst.code_base = mem;
        inst.code_size = text.len;
    }

    // Build function pointer table for call_indirect
    const import_count = module.import_function_count;

    // Build module func idx → native address mapping (temporary)
    const total_funcs = import_count + module.func_count;
    var func_addrs: [256]usize = std.mem.zeroes([256]usize);
    const n_addrs = @min(total_funcs, func_addrs.len);
    // Import functions → host function pointers
    for (0..@min(import_count, @min(inst.host_functions.len, n_addrs))) |i| {
        func_addrs[i] = if (inst.host_functions[i]) |ptr| @intFromPtr(ptr) else 0;
    }
    // Local functions → code_base + offset
    if (has_text) {
        for (0..@min(module.func_count, n_addrs - @min(import_count, n_addrs))) |i| {
            const offset = module.func_offsets[i];
            func_addrs[import_count + i] = @intFromPtr(mem) + offset;
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
        info[0] = .{
            .ptr = @intFromPtr(inst.func_table.ptr),
            .len = @intCast(inst.func_table.len),
            .type_backing_ptr = if (inst.tables.len > 0 and inst.tables[0].type_backing.len > 0)
                @intFromPtr(inst.tables[0].type_backing.ptr)
            else
                0,
        };

        // Slots 1..n.
        for (inst.tables[1..], 1..) |tbl_i, idx| {
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

        inst.tables_info = info;
        inst.extra_tables_storage = extra;
    }
}

/// Call an AOT-compiled function by index.
/// The code must have been mapped via `mapCodeExecutable` first.
///
/// Uses comptime to select the correct function pointer type based on `Result`.
pub fn callFunc(inst: *AotInstance, func_idx: u32, comptime Result: type) RuntimeError!Result {
    comptime if (!can_execute_native) @compileError("AOT execution not supported on this architecture");

    if (inst.code_base == null) return error.CodeMappingFailed;
    const addr = getFuncAddr(inst, func_idx) orelse blk: {
        // Import functions have no native code in this module, but
        // their funcptrs slot may have been patched with the exporter's
        // native code pointer (cross-module import wiring). Use that.
        if (func_idx < inst.funcptrs.len and inst.funcptrs[func_idx] != 0)
            break :blk @as([*]const u8, @ptrFromInt(inst.funcptrs[func_idx]))
        else
            return error.FunctionNotFound;
    };

    // Build flat globals array for AOT access
    var globals_buf: [256]i64 = std.mem.zeroes([256]i64);
    const n_globals = @min(inst.globals.len, globals_buf.len);
    for (0..n_globals) |i| {
        globals_buf[i] = globalValueToI64(inst, inst.globals[i].value);
    }

    // Build VMContext for the compiled function
    var vmctx = VmCtx{};
    if (inst.memories.len > 0) {
        vmctx.memory_base = @intFromPtr(inst.memories[0].data.ptr);
        vmctx.memory_size = @as(usize, inst.memories[0].current_pages) * types.MemoryInstance.page_size;
        vmctx.memory_max_size = inst.memories[0].data.len;
        vmctx.memory_pages = inst.memories[0].current_pages;
    }
    // Always provide a valid globals pointer — compiled code may access globals
    // even if none are explicitly initialized (they default to zero).
    vmctx.globals_ptr = @intFromPtr(&globals_buf);
    vmctx.globals_count = @intCast(n_globals);
    if (inst.host_functions.len > 0) {
        vmctx.host_functions_ptr = @intFromPtr(inst.host_functions.ptr);
        vmctx.host_functions_count = @intCast(inst.host_functions.len);
    }
    if (inst.func_table.len > 0) {
        vmctx.func_table_ptr = @intFromPtr(inst.func_table.ptr);
        vmctx.func_table_len = @intCast(inst.func_table.len);
    }
    if (inst.funcptrs.len > 0) {
        vmctx.funcptrs_ptr = @intFromPtr(inst.funcptrs.ptr);
    }
    vmctx.instance_ptr = @intFromPtr(inst);
    vmctx.mem_grow_fn = @intFromPtr(&memGrowHelper);
    vmctx.trap_oob_fn = @intFromPtr(&aotTrapOOB);
    vmctx.trap_unreachable_fn = @intFromPtr(&aotTrapUnreachable);
    vmctx.trap_idivz_fn = @intFromPtr(&aotTrapIntDivZero);
    vmctx.trap_iovf_fn = @intFromPtr(&aotTrapIntOverflow);
    vmctx.trap_ivc_fn = @intFromPtr(&aotTrapInvalidConversion);
    vmctx.table_grow_fn = @intFromPtr(&tableGrowHelper);
    if (inst.tables_info.len > 0) {
        vmctx.tables_info_ptr = @intFromPtr(inst.tables_info.ptr);
    }
    vmctx.table_init_fn = @intFromPtr(&tableInitHelper);
    vmctx.elem_drop_fn = @intFromPtr(&elemDropHelper);
    vmctx.table_set_fn = @intFromPtr(&tableSetHelper);
    vmctx.futex_wait32_fn = @intFromPtr(&aotAtomicWait32);
    vmctx.futex_wait64_fn = @intFromPtr(&aotAtomicWait64);
    vmctx.futex_notify_fn = @intFromPtr(&aotAtomicNotify);
    if (inst.sig_table.len > 0) vmctx.sig_table_ptr = @intFromPtr(inst.sig_table.ptr);
    if (inst.func_sig_ids.len > 0) vmctx.func_sig_ids_ptr = @intFromPtr(inst.func_sig_ids.ptr);
    if (inst.ptr_to_sig.len > 0) {
        vmctx.ptr_to_sig_ptr = @intFromPtr(inst.ptr_to_sig.ptr);
        vmctx.ptr_to_sig_len = @intCast(inst.ptr_to_sig.len);
    }

    // AOT-compiled functions receive a VmCtx pointer as hidden first parameter.
    const FnPtr = *const fn (*VmCtx) callconv(.c) Result;
    const func_ptr: FnPtr = @ptrCast(@alignCast(addr));
    if (comptime builtin.os.tag == .windows) {
        if (inst.code_base) |cb| {
            g_code_base = @intFromPtr(cb);
            g_code_size = inst.code_size;
            g_func_offsets = inst.module.func_offsets;
        }
        g_mem_base = vmctx.memory_base;
        g_mem_size = vmctx.memory_size;
        if (!g_veh_installed) {
            _ = AddVectoredExceptionHandler(1, vehHandler);
            g_veh_installed = true;
        }
    }
    const result = func_ptr(&vmctx);

    // Sync globals back from flat array to GlobalInstance objects
    for (0..n_globals) |i| {
        inst.globals[i].value = globalValueFromI64(inst, inst.globals[i].value, globals_buf[i]);
    }

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
/// Registers available for args:
///   - Win64: RCX/RDX/R8/R9 → VmCtx + up to 3 wasm params.
///   - SysV AMD64: RDI/RSI/RDX/RCX/R8/R9 → VmCtx + up to 5 wasm params.
/// We conservatively cap at 3 wasm params so the same typed harness works
/// on both platforms. Anything outside → `error.UnsupportedSignature`.
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
const MaxScalarArgs: usize = 12;
/// Max number of results a multi-value return may produce via the
/// `callFuncScalar` path. Bounded by the HRP stack buffer size below.
pub const MaxScalarResults: usize = 16;

fn isScalarValType(t: types.ValType) bool {
    return switch (t) {
        .i32, .i64, .f32, .f64, .funcref, .externref => true,
        else => false,
    };
}

/// Pack a global's typed value into a raw 64-bit slot for the flat globals_buf
/// that AOT code accesses via `movRegMem(dr, globals_base, idx*8)`. The tag
/// determines the bit-cast; using `.value.i64` would be UB when the active
/// tag is `.f32` / `.f64`.
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
    comptime if (!can_execute_native) @compileError("AOT execution not supported on this architecture");

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

    if (inst.code_base == null) return error.CodeMappingFailed;
    const addr = getFuncAddr(inst, func_idx) orelse blk: {
        if (func_idx < inst.funcptrs.len and inst.funcptrs[func_idx] != 0)
            break :blk @as([*]const u8, @ptrFromInt(inst.funcptrs[func_idx]))
        else
            return error.FunctionNotFound;
    };

    // Build flat globals array for AOT access (mirrors callFunc).
    var globals_buf: [256]i64 = std.mem.zeroes([256]i64);
    const n_globals = @min(inst.globals.len, globals_buf.len);
    for (0..n_globals) |i| {
        globals_buf[i] = globalValueToI64(inst, inst.globals[i].value);
    }

    var vmctx = VmCtx{};
    if (inst.memories.len > 0) {
        vmctx.memory_base = @intFromPtr(inst.memories[0].data.ptr);
        vmctx.memory_size = @as(usize, inst.memories[0].current_pages) * types.MemoryInstance.page_size;
        vmctx.memory_max_size = inst.memories[0].data.len;
        vmctx.memory_pages = inst.memories[0].current_pages;
    }
    vmctx.globals_ptr = @intFromPtr(&globals_buf);
    vmctx.globals_count = @intCast(n_globals);
    if (inst.host_functions.len > 0) {
        vmctx.host_functions_ptr = @intFromPtr(inst.host_functions.ptr);
        vmctx.host_functions_count = @intCast(inst.host_functions.len);
    }
    if (inst.func_table.len > 0) {
        vmctx.func_table_ptr = @intFromPtr(inst.func_table.ptr);
        vmctx.func_table_len = @intCast(inst.func_table.len);
    }
    if (inst.funcptrs.len > 0) {
        vmctx.funcptrs_ptr = @intFromPtr(inst.funcptrs.ptr);
    }
    vmctx.instance_ptr = @intFromPtr(inst);
    vmctx.mem_grow_fn = @intFromPtr(&memGrowHelper);
    vmctx.trap_oob_fn = @intFromPtr(&aotTrapOOB);
    vmctx.trap_unreachable_fn = @intFromPtr(&aotTrapUnreachable);
    vmctx.trap_idivz_fn = @intFromPtr(&aotTrapIntDivZero);
    vmctx.trap_iovf_fn = @intFromPtr(&aotTrapIntOverflow);
    vmctx.trap_ivc_fn = @intFromPtr(&aotTrapInvalidConversion);
    vmctx.table_grow_fn = @intFromPtr(&tableGrowHelper);
    if (inst.tables_info.len > 0) {
        vmctx.tables_info_ptr = @intFromPtr(inst.tables_info.ptr);
    }
    vmctx.table_init_fn = @intFromPtr(&tableInitHelper);
    vmctx.elem_drop_fn = @intFromPtr(&elemDropHelper);
    vmctx.table_set_fn = @intFromPtr(&tableSetHelper);
    vmctx.futex_wait32_fn = @intFromPtr(&aotAtomicWait32);
    vmctx.futex_wait64_fn = @intFromPtr(&aotAtomicWait64);
    vmctx.futex_notify_fn = @intFromPtr(&aotAtomicNotify);
    if (inst.sig_table.len > 0) vmctx.sig_table_ptr = @intFromPtr(inst.sig_table.ptr);
    if (inst.func_sig_ids.len > 0) vmctx.func_sig_ids_ptr = @intFromPtr(inst.func_sig_ids.ptr);
    if (inst.ptr_to_sig.len > 0) {
        vmctx.ptr_to_sig_ptr = @intFromPtr(inst.ptr_to_sig.ptr);
        vmctx.ptr_to_sig_len = @intCast(inst.ptr_to_sig.len);
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

    if (comptime builtin.os.tag == .windows) {
        if (inst.code_base) |cb| {
            g_code_base = @intFromPtr(cb);
            g_code_size = inst.code_size;
            g_func_offsets = inst.module.func_offsets;
        }
        g_mem_base = vmctx.memory_base;
        g_mem_size = vmctx.memory_size;
        if (!g_veh_installed) {
            _ = AddVectoredExceptionHandler(1, vehHandler);
            g_veh_installed = true;
        }
    }

    // Arm the trap-as-error path. Trap helpers (aotTrapOOB, aotTrapUnreachable,
    // ...) called from generated code check `g_trap_catching` and longjmp
    // back to the `RtlCaptureContext` site below with `g_trap_occurred = true`
    // when armed; we then return `error.WasmTrap`.
    //
    // We do NOT arm this for the hardware-VEH path (ud2/int3 traps inside
    // generated code); those are still routed via the VEH, which proved
    // unstable for our use case. All wasm traps now go through explicit
    // helper calls, so the VEH is effectively unused.
    if (comptime builtin.os.tag == .windows) {
        @atomicStore(bool, &g_trap_occurred, false, .seq_cst);
        @atomicStore(u32, &g_last_trap_code, 0, .seq_cst);
        // Reserve extra stack headroom so the VEH and trapLongjmp can
        // run safely after a STATUS_STACK_OVERFLOW consumes the guard
        // page. Without this, the OS leaves ~4KB of space below the
        // former guard for user-mode dispatch — enough for the VEH to
        // fire, but RtlRestoreContext and the subsequent return path
        // may overflow. 16 KB is generous and idempotent across calls.
        var guarantee: u32 = 16 * 1024;
        _ = SetThreadStackGuarantee(&guarantee);
        @atomicStore(bool, &g_trap_catching, true, .seq_cst);
        RtlCaptureContext(&g_saved_ctx);
        if (@atomicLoad(bool, &g_trap_occurred, .seq_cst)) {
            @atomicStore(bool, &g_trap_catching, false, .seq_cst);
            // If the trap was a stack overflow, the OS consumed the
            // thread's guard page. Re-arm it here so a subsequent
            // overflow in this process is also catchable rather than
            // silently aborting. Runs on the post-longjmp stack, which
            // is well clear of the former-guard region.
            if (@atomicLoad(u32, &g_last_trap_code, .seq_cst) == 0xC00000FD) {
                resetStackGuardPage();
            }
            for (0..n_globals) |i| {
                inst.globals[i].value = globalValueFromI64(inst, inst.globals[i].value, globals_buf[i]);
            }
            return error.WasmTrap;
        }
    }

    const raw_result: u64 = switch (effective_args) {
        0 => blk: {
            const f: CallFn0 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx);
        },
        1 => blk: {
            const f: CallFn1 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0]);
        },
        2 => blk: {
            const f: CallFn2 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1]);
        },
        3 => blk: {
            const f: CallFn3 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2]);
        },
        4 => blk: {
            const f: CallFn4 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2], raw[3]);
        },
        5 => blk: {
            const f: CallFn5 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2], raw[3], raw[4]);
        },
        6 => blk: {
            const f: CallFn6 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]);
        },
        7 => blk: {
            const f: CallFn7 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6]);
        },
        8 => blk: {
            const f: CallFn8 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]);
        },
        9 => blk: {
            const f: CallFn9 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8]);
        },
        10 => blk: {
            const f: CallFn10 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9]);
        },
        11 => blk: {
            const f: CallFn11 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9], raw[10]);
        },
        12 => blk: {
            const f: CallFn12 = @ptrCast(@alignCast(addr));
            break :blk f(&vmctx, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9], raw[10], raw[11]);
        },
        else => unreachable,
    };

    if (comptime builtin.os.tag == .windows) {
        @atomicStore(bool, &g_trap_catching, false, .seq_cst);
    }

    // Sync globals back.
    for (0..n_globals) |i| {
        inst.globals[i].value = globalValueFromI64(inst, inst.globals[i].value, globals_buf[i]);
    }

    if (result_types.len == 0) return results_out[0..0];

    results_out[0] = decodeScalarResult(inst, result_types[0], raw_result);
    var i: usize = 1;
    while (i < result_types.len) : (i += 1) {
        results_out[i] = decodeScalarResult(inst, result_types[i], hrp_buf[i - 1]);
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
    return resolveHostFunctionsImpl(module, allocator, null);
}

/// Resolve host functions with optional custom HostImports (comptime-typed).
pub fn resolveHostFunctionsWithHosts(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
    comptime HostImportsT: ?type,
) RuntimeError![]const ?*const anyopaque {
    return resolveHostFunctionsImpl(module, allocator, HostImportsT);
}

fn resolveHostFunctionsImpl(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
    comptime HostImportsT: ?type,
) RuntimeError![]const ?*const anyopaque {
    if (module.import_function_count == 0) return &.{};

    const host_fns = allocator.alloc(?*const anyopaque, module.import_function_count) catch
        return error.OutOfMemory;
    @memset(host_fns, null);

    var func_idx: u32 = 0;
    for (module.imports) |imp| {
        if (imp.kind == .function) {
            if (func_idx < module.import_function_count) {
                // Layer 1: custom HostImports (comptime-resolved)
                if (HostImportsT) |HI| {
                    if (HI.resolve(imp.module_name, imp.field_name)) |entry| {
                        host_fns[func_idx] = entry.aot_fn;
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

fn allocateMemories(module: *const aot_loader.AotModule, allocator: std.mem.Allocator) RuntimeError![]*types.MemoryInstance {
    if (module.memories.len == 0) return &.{};

    const memories = allocator.alloc(*types.MemoryInstance, module.memories.len) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| memories[i].release(allocator);
        allocator.free(memories);
    }

    for (module.memories, 0..) |mem_type, i| {
        const initial_pages: u32 = @intCast(@min(mem_type.limits.min, 65536));
        const max_pages: u32 = @intCast(@min(mem_type.limits.max orelse 65536, 65536));
        // Pre-allocate for memory.grow: use max_pages but cap at a reasonable default
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
        memories[i] = mem;
        initialized += 1;
    }

    return memories;
}

fn allocateTables(module: *const aot_loader.AotModule, allocator: std.mem.Allocator) RuntimeError![]*types.TableInstance {
    if (module.tables.len == 0) return &.{};

    const tables = allocator.alloc(*types.TableInstance, module.tables.len) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| tables[i].release(allocator);
        allocator.free(tables);
    }

    for (module.tables, 0..) |table_type, i| {
        const elements = allocator.alloc(types.TableElement, table_type.limits.min) catch return error.TableAllocationFailed;
        for (elements) |*e| e.* = types.TableElement.nullForType(table_type.elem_type);
        const tbl = allocator.create(types.TableInstance) catch {
            allocator.free(elements);
            return error.TableAllocationFailed;
        };
        tbl.* = .{ .table_type = table_type, .elements = elements };
        tables[i] = tbl;
        initialized += 1;
    }

    return tables;
}

fn allocateGlobals(module: *const aot_loader.AotModule, allocator: std.mem.Allocator) RuntimeError![]*types.GlobalInstance {
    if (module.global_inits.len == 0) return &.{};

    const globals = allocator.alloc(*types.GlobalInstance, module.global_inits.len) catch return error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| allocator.destroy(globals[i]);
        allocator.free(globals);
    }

    for (module.global_inits, 0..) |ginit, i| {
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
        globals[i] = g;
        initialized += 1;
    }

    return globals;
}

fn freeMemories(memories: []*types.MemoryInstance, allocator: std.mem.Allocator) void {
    for (memories) |m| m.release(allocator);
    if (memories.len > 0) allocator.free(memories);
}

fn freeTables(tables: []*types.TableInstance, allocator: std.mem.Allocator) void {
    for (tables) |t| t.release(allocator);
    if (tables.len > 0) allocator.free(tables);
}

fn freeGlobals(globals: []*types.GlobalInstance, allocator: std.mem.Allocator) void {
    for (globals) |g| allocator.destroy(g);
    if (globals.len > 0) allocator.free(globals);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

test "instantiate: empty module" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(usize, 0), inst.memories.len);
    try std.testing.expectEqual(@as(usize, 0), inst.tables.len);
    try std.testing.expectEqual(@as(usize, 0), inst.globals.len);
    try std.testing.expectEqual(@as(?[*]const u8, null), inst.code_base);
}

test "findExportFunc: returns null for missing export" {
    const module = aot_loader.AotModule{};
    const inst = try instantiate(&module, std.testing.allocator);
    defer destroy(inst);

    try std.testing.expectEqual(@as(?u32, null), findExportFunc(inst, "nonexistent"));
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
    // Pre-allocated to max_pages (4) for memory.grow support
    try std.testing.expectEqual(@as(usize, 4 * 65536), inst.memories[0].data.len);
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
