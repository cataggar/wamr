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

extern "kernel32" fn RtlCaptureContext(ContextRecord: *windows.CONTEXT) callconv(.winapi) void;
extern "kernel32" fn RtlRestoreContext(ContextRecord: *windows.CONTEXT, ExceptionRecord: ?*anyopaque) callconv(.winapi) noreturn;

fn trapLongjmp() noreturn {
    @atomicStore(bool, &g_trap_occurred, true, .seq_cst);
    RtlRestoreContext(&g_saved_ctx, null);
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
        ctx.Rip = @intFromPtr(&trapLongjmp);
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
pub fn tableGrowHelper(vmctx: *VmCtx, init_val: i64, delta: i32) callconv(.c) i32 {
    if (vmctx.instance_ptr == 0) return -1;
    const inst: *AotInstance = @ptrFromInt(vmctx.instance_ptr);
    if (delta < 0) return -1;
    const old_size: u32 = @intCast(inst.func_table.len);
    const delta_u: u32 = @intCast(delta);
    const new_size_u64: u64 = @as(u64, old_size) + @as(u64, delta_u);
    // Use table metadata if available for an accurate max; otherwise cap at u32 max.
    const max_cap: u64 = blk: {
        if (inst.tables.len > 0) {
            break :blk inst.tables[0].table_type.limits.max orelse 0xFFFF_FFFF;
        } else {
            break :blk 0xFFFF_FFFF;
        }
    };
    if (new_size_u64 > max_cap) return -1;
    const new_size: usize = @intCast(new_size_u64);
    const new_table = inst.allocator.realloc(inst.func_table, new_size) catch return -1;
    const fill_ptr: usize = @as(usize, @bitCast(@as(i64, init_val)));
    var i: usize = old_size;
    while (i < new_size) : (i += 1) new_table[i] = fill_ptr;
    inst.func_table = new_table;
    vmctx.func_table_ptr = @intFromPtr(new_table.ptr);
    vmctx.func_table_len = @intCast(new_size);
    return @intCast(old_size);
}

// ─── Comptime target validation ─────────────────────────────────────────────

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
    };

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
    if (inst.func_table.len > 0) allocator.free(inst.func_table);
    if (inst.funcptrs.len > 0) allocator.free(inst.funcptrs);
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
    const text = inst.module.text_section orelse return;
    if (text.len == 0) return;

    // 1. Allocate RW pages
    const mem = platform.mmap(null, text.len, .{ .read = true, .write = true }, .{}) orelse
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

    // Build function pointer table for call_indirect
    const module = inst.module;
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
    for (0..@min(module.func_count, n_addrs - @min(import_count, n_addrs))) |i| {
        const offset = module.func_offsets[i];
        func_addrs[import_count + i] = @intFromPtr(mem) + offset;
    }

    // Persist the funcidx → native address map on the instance for ref.func.
    if (n_addrs > 0) {
        const persistent = inst.allocator.alloc(usize, n_addrs) catch return error.OutOfMemory;
        @memcpy(persistent, func_addrs[0..n_addrs]);
        inst.funcptrs = persistent;
    }

    // Build wasm table → native address table for call_indirect
    if (inst.tables.len > 0) {
        const tbl = inst.tables[0];
        const tbl_size = tbl.elements.len;
        if (tbl_size > 0) {
            const native_table = inst.allocator.alloc(usize, tbl_size) catch return error.OutOfMemory;
            @memset(native_table, 0);

            // Apply element segments
            for (module.elem_segments) |seg| {
                if (seg.table_idx != 0) continue;
                for (seg.func_indices, 0..) |func_idx, j| {
                    const dst = seg.offset + @as(u32, @intCast(j));
                    if (dst < tbl_size and func_idx < n_addrs) {
                        native_table[dst] = func_addrs[func_idx];
                    }
                }
            }

            inst.func_table = native_table;
        }
    }
}

/// Call an AOT-compiled function by index.
/// The code must have been mapped via `mapCodeExecutable` first.
///
/// Uses comptime to select the correct function pointer type based on `Result`.
pub fn callFunc(inst: *AotInstance, func_idx: u32, comptime Result: type) RuntimeError!Result {
    comptime if (!can_execute_native) @compileError("AOT execution not supported on this architecture");

    if (inst.code_base == null) return error.CodeMappingFailed;
    const addr = getFuncAddr(inst, func_idx) orelse return error.FunctionNotFound;

    // Build flat globals array for AOT access
    var globals_buf: [256]i64 = std.mem.zeroes([256]i64);
    const n_globals = @min(inst.globals.len, globals_buf.len);
    for (0..n_globals) |i| {
        globals_buf[i] = globalValueToI64(inst.globals[i].value);
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
        inst.globals[i].value = globalValueFromI64(inst.globals[i].value, globals_buf[i]);
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
fn globalValueToI64(v: types.Value) i64 {
    return switch (v) {
        .i32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .i64 => |x| x,
        .f32 => |x| @as(i64, @as(u32, @bitCast(x))),
        .f64 => |x| @as(i64, @bitCast(x)),
        else => 0,
    };
}

/// Unpack a raw 64-bit slot from globals_buf back into a typed Value,
/// preserving the active tag.
fn globalValueFromI64(old: types.Value, raw: i64) types.Value {
    return switch (old) {
        .i32 => .{ .i32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(raw))))) },
        .i64 => .{ .i64 = raw },
        .f32 => .{ .f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(raw))))) },
        .f64 => .{ .f64 = @bitCast(raw) },
        else => old,
    };
}

fn valueToRawBits(pt: types.ValType, v: types.Value) ScalarCallError!u64 {
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
        // Reference types: null -> 0, indexed -> index as u64 (tests use small
        // integer "externref N" literals and null; AOT stores a raw pointer).
        .funcref => switch (v) {
            .funcref => |maybe| @as(u64, maybe orelse 0),
            .nonfuncref => |maybe| @as(u64, maybe orelse 0),
            else => error.InvalidArgType,
        },
        .externref => switch (v) {
            .externref => |maybe| @as(u64, maybe orelse 0),
            .nonexternref => |maybe| @as(u64, maybe orelse 0),
            else => error.InvalidArgType,
        },
        else => error.UnsupportedSignature,
    };
}

/// Call an AOT-compiled function by index with runtime-typed scalar args.
///
/// Supports up to 3 params of type i32/i64/f32/f64 and a result of
/// void/i32/i64/f32/f64. Wider or non-scalar signatures return
/// `error.UnsupportedSignature` and should be skipped by the caller.
pub fn callFuncScalar(
    inst: *AotInstance,
    func_idx: u32,
    param_types: []const types.ValType,
    result_type: ?types.ValType,
    args: []const types.Value,
) ScalarCallError!ScalarResult {
    comptime if (!can_execute_native) @compileError("AOT execution not supported on this architecture");

    if (param_types.len != args.len) return error.ArgCountMismatch;
    if (param_types.len > MaxScalarArgs) return error.UnsupportedSignature;
    for (param_types) |pt| {
        if (!isScalarValType(pt)) return error.UnsupportedSignature;
    }
    if (result_type) |rt| {
        if (!isScalarValType(rt)) return error.UnsupportedSignature;
    }

    if (inst.code_base == null) return error.CodeMappingFailed;
    const addr = getFuncAddr(inst, func_idx) orelse return error.FunctionNotFound;

    // Build flat globals array for AOT access (mirrors callFunc).
    var globals_buf: [256]i64 = std.mem.zeroes([256]i64);
    const n_globals = @min(inst.globals.len, globals_buf.len);
    for (0..n_globals) |i| {
        globals_buf[i] = globalValueToI64(inst.globals[i].value);
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

    // Marshal args to raw 64-bit bit patterns.
    var raw: [MaxScalarArgs]u64 = [_]u64{0} ** MaxScalarArgs;
    for (args, param_types, 0..) |v, pt, i| {
        raw[i] = try valueToRawBits(pt, v);
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
        @atomicStore(bool, &g_trap_catching, true, .seq_cst);
        RtlCaptureContext(&g_saved_ctx);
        if (@atomicLoad(bool, &g_trap_occurred, .seq_cst)) {
            @atomicStore(bool, &g_trap_catching, false, .seq_cst);
            for (0..n_globals) |i| {
                inst.globals[i].value = globalValueFromI64(inst.globals[i].value, globals_buf[i]);
            }
            return error.WasmTrap;
        }
    }

    const raw_result: u64 = switch (args.len) {
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
        inst.globals[i].value = globalValueFromI64(inst.globals[i].value, globals_buf[i]);
    }

    if (result_type) |rt| {
        return switch (rt) {
            .i32 => ScalarResult{ .i32 = @bitCast(@as(u32, @truncate(raw_result))) },
            .i64 => ScalarResult{ .i64 = @bitCast(raw_result) },
            .f32 => ScalarResult{ .f32 = @bitCast(@as(u32, @truncate(raw_result))) },
            .f64 => ScalarResult{ .f64 = @bitCast(raw_result) },
            .funcref => ScalarResult{ .funcref = if (raw_result == 0) null else raw_result },
            .externref => ScalarResult{ .externref = if (raw_result == 0) null else raw_result },
            else => unreachable,
        };
    }
    return .void;
}

// ─── Host function resolution ───────────────────────────────────────────────

/// Resolve AOT host function adapters for each import in the module.
/// Returns a slice of optional function pointers indexed by import index.
fn resolveHostFunctions(
    module: *const aot_loader.AotModule,
    allocator: std.mem.Allocator,
) RuntimeError![]const ?*const anyopaque {
    if (module.import_function_count == 0) return &.{};

    const host_fns = allocator.alloc(?*const anyopaque, module.import_function_count) catch
        return error.OutOfMemory;
    @memset(host_fns, null);

    var func_idx: u32 = 0;
    for (module.imports) |imp| {
        if (imp.kind == .function) {
            if (func_idx < module.import_function_count) {
                if (host_bridge.isWasiModule(imp.module_name)) {
                    host_fns[func_idx] = host_bridge.resolveAotHostFunction(imp.field_name);
                } else if (host_bridge.isSpectestModule(imp.module_name)) {
                    host_fns[func_idx] = host_bridge.resolveAotSpectestFunction(imp.field_name);
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
        g.* = .{
            .global_type = .{
                .val_type = @enumFromInt(ginit.val_type),
                .mutability = if (ginit.mutability != 0) .mutable else .immutable,
            },
            .value = .{ .i64 = ginit.init_i64 },
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
