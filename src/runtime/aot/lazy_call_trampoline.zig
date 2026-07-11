//! #879 M4.6 (phase 1 of 2): native trampolines for lazily-compiled
//! functions reachable via `call_indirect` / `ref.func`+`call_ref`.
//!
//! The #862 lazy-JIT spike only defers functions that are leaf, never
//! a direct call target, AND never referenced by any element segment
//! (see `lazy_jit.findLazyEligibleLeaves`'s doc comment) — because
//! `func_addrs[i]`/a table's native backing must always be a genuinely
//! callable address for anything the runtime might jump to directly,
//! and the spike had no such address to offer for a not-yet-compiled
//! function. `lazy_jit.findLazyEligibleWithTrampoline` (phase 1, same
//! PR as this file) lifts that restriction at the analysis level; this
//! file provides the actual callable address those newly-eligible
//! functions need.
//!
//! ## Design
//!
//! A `LazyCallTrampolinePool` slot is a small blob of hand-encoded
//! machine code that:
//!
//!   1. saves every argument-passing register untouched — it never
//!      needs to interpret them. This codegen's wasm-to-wasm calling
//!      convention passes every scalar argument through the same
//!      fixed sequence of general-purpose registers regardless of
//!      wasm type (see `x86_64/compile.zig`'s `param_regs`: `rdi`
//!      always carries the callee's `VmCtx*`, `rsi..r9` carry up to
//!      five more args, anything beyond that — plus a hidden
//!      multi-result return pointer, if any — is passed on the
//!      stack). The trampoline is agnostic to which of those
//!      registers are actually "live" for a given signature: saving
//!      and restoring an unused one is harmless.
//!   2. calls back into a caller-supplied `LazyDispatchFn` with a
//!      `(ctx, local_idx)` pair baked into this specific stub
//!      instance at `allocSlot` time,
//!   3. restores every saved register,
//!   4. unconditionally `jmp`s to the address `LazyDispatchFn`
//!      returned.
//!
//! Step 4 is a tail-jump, not a call: the trampoline never pushes a
//! new return address, so it works identically whether the
//! `call_indirect`/`call_ref` site that reached it used a real `call`
//! (return address already on the stack, untouched) or was itself
//! emitted as a tail call (in which case the trampoline just forwards
//! the *caller's* inherited return address, exactly as the real
//! function would have). Stack-passed arguments (beyond the register
//! budget) and an on-stack hidden-return-pointer slot are never
//! touched either — the trampoline's own `sub rsp` / `add rsp` pair
//! around the dispatch call is fully self-balanced, so anything the
//! caller already placed above the return address stays exactly where
//! the freshly-compiled callee's prologue expects to find it.
//!
//! `LazyDispatchFn` must never return an unusable address: on compile
//! failure it must abort the process itself (there is no calling
//! convention here through which the trampoline could propagate a Zig
//! error, or trap cleanly back to the guest). This matches the
//! "should never happen" nature of a compile failure for a function
//! that already passed eligibility analysis and IR verification.
//!
//! ## Scope of this phase
//!
//! This is a standalone, independently-tested primitive — NOT yet
//! wired into `compileCoreWasmCached`/`mapCodeExecutable`/`AotInstance`
//! (that's phase 2, left as explicit follow-up work; see
//! `docs/design/lazy-jit-spike.md`). x86_64 only (SysV and Win64 ABIs
//! both supported, matching `host_trampolines.TrampolinePool`'s own
//! platform coverage); aarch64 needs the same additive change but is
//! unverified without real hardware (cf. #874/#886's aarch64 gap
//! notes).
//!
//! Unlike `host_trampolines.TrampolinePool` — whose stubs resolve
//! through a single process-wide `g_active_pool` plus a baked-in slot
//! index, because at most one component instance's host-import
//! trampolines are ever "active" at a time — every stub here bakes
//! its full `(ctx, local_idx, dispatch_fn)` triple directly as
//! immediates. There is no shared mutable global state at all: two
//! `AotInstance`s with their own lazy-eligible tabled functions can
//! execute concurrently on different threads without needing to
//! coordinate over which one is "active".

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("../../platform/platform.zig");

const is_windows = builtin.os.tag == .windows;
const page_size = std.heap.page_size_min;

/// Type-erased "compile local function `local_idx` now; return its
/// callable native address" callback. See this file's doc comment for
/// the "must never return an unusable address" contract.
pub const LazyDispatchFn = *const fn (ctx: *anyopaque, local_idx: u32) callconv(.c) usize;

const StubFn = *const fn () callconv(.c) void;

// SysV: 6 pushes (8B) + sub rsp,8 (4B) + movabs rdi,ctx (10B) +
// mov esi,local_idx (5B) + movabs rax,dispatch (10B) + call rax (2B) +
// add rsp,8 (4B) + 6 pops (8B) + jmp rax (2B) = 53B. See
// `encodeX8664SysvStub` for the byte-exact derivation.
const x86_64_sysv_stub_bytes: usize = 53;
// Win64: 4 pushes (6B) + sub rsp,40 (4B) + movabs rcx,ctx (10B) +
// mov edx,local_idx (5B) + movabs rax,dispatch (10B) + call rax (2B) +
// add rsp,40 (4B) + 4 pops (6B) + jmp rax (2B) = 49B.
const x86_64_win64_stub_bytes: usize = 49;

pub const STUB_BYTES: usize = switch (builtin.cpu.arch) {
    .x86_64 => if (is_windows) x86_64_win64_stub_bytes else x86_64_sysv_stub_bytes,
    else => 0,
};

/// Default per-pool slot count. Deliberately small — this phase-1
/// primitive has no production caller yet (phase 2 will size this the
/// way `host_trampolines.DEFAULT_MAX_SLOTS` is sized: configurable,
/// with a structured exhaustion error).
pub const DEFAULT_CAP: u32 = 256;

const supports_pool: bool = builtin.cpu.arch == .x86_64;

pub const LazyCallTrampolinePool = struct {
    memory: []align(page_size) u8,
    cap: u32,
    next_slot: u32 = 0,

    pub fn init(allocator: std.mem.Allocator) !LazyCallTrampolinePool {
        return initWithCap(allocator, DEFAULT_CAP);
    }

    pub fn initWithCap(allocator: std.mem.Allocator, cap_req: u32) !LazyCallTrampolinePool {
        _ = allocator; // no per-slot metadata array to allocate (unlike TrampolinePool) -- everything is baked into the stub bytes themselves.
        if (comptime !supports_pool) return error.UnsupportedPlatform;
        const cap: u32 = @max(cap_req, 1);

        const map_len = std.mem.alignForward(usize, @as(usize, cap) * STUB_BYTES, page_size);
        const memory: []align(page_size) u8 = if (comptime is_windows) win: {
            const ptr = platform.mmap(
                null,
                map_len,
                .{ .read = true, .write = true, .exec = true },
                .{},
            ) orelse return error.OutOfMemory;
            break :win @as([*]align(page_size) u8, @alignCast(ptr))[0..map_len];
        } else posix: {
            // macOS aarch64 needs MAP_JIT; see `host_trampolines.TrampolinePool.initWithCap`
            // for the same pattern and its rationale.
            const map_flags: std.posix.MAP = if (comptime platform.macos_jit)
                .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .JIT = true }
            else
                .{ .TYPE = .PRIVATE, .ANONYMOUS = true };
            break :posix try std.posix.mmap(
                null,
                map_len,
                .{ .READ = true, .WRITE = true, .EXEC = true },
                map_flags,
                -1,
                0,
            );
        };
        platform.jitWriteProtect(false);
        @memset(memory, 0);
        platform.jitWriteProtect(true);

        return .{ .memory = memory, .cap = cap };
    }

    pub fn deinit(self: *LazyCallTrampolinePool) void {
        if (comptime !supports_pool) {
            self.* = undefined;
            return;
        }
        if (comptime is_windows) {
            platform.munmap(self.memory.ptr, self.memory.len);
        } else {
            std.posix.munmap(self.memory);
        }
        self.* = undefined;
    }

    /// Allocate and install a new trampoline stub bound to
    /// `(ctx, local_idx, dispatch_fn)`. Returns the stub's own address,
    /// castable to whatever function-pointer type the caller's
    /// `func_addrs[i]`/table slot expects — the stub itself never
    /// inspects its own incoming arguments, so it is safe to call
    /// through any signature.
    pub fn allocSlot(
        self: *LazyCallTrampolinePool,
        ctx: *anyopaque,
        local_idx: u32,
        dispatch_fn: LazyDispatchFn,
    ) !StubFn {
        if (comptime !supports_pool) return error.UnsupportedPlatform;
        const slot = self.next_slot;
        if (slot >= self.cap) return error.OutOfTrampolineSlots;
        self.next_slot += 1;

        platform.jitWriteProtect(false);
        writeStub(self.stubBytes(slot), ctx, local_idx, dispatch_fn);
        platform.jitWriteProtect(true);
        platform.icacheFlush(self.stubPtr(slot), STUB_BYTES);

        return @ptrFromInt(@intFromPtr(self.stubPtr(slot)));
    }

    fn stubPtr(self: *LazyCallTrampolinePool, slot: u32) [*]u8 {
        return self.memory.ptr + (@as(usize, slot) * STUB_BYTES);
    }

    fn stubBytes(self: *LazyCallTrampolinePool, slot: u32) []u8 {
        const start = @as(usize, slot) * STUB_BYTES;
        return self.memory[start .. start + STUB_BYTES];
    }
};

fn writeStub(bytes: []u8, ctx: *anyopaque, local_idx: u32, dispatch_fn: LazyDispatchFn) void {
    @memset(bytes, 0);
    switch (builtin.cpu.arch) {
        .x86_64 => if (comptime is_windows)
            encodeX8664Win64Stub(bytes, ctx, local_idx, dispatch_fn)
        else
            encodeX8664SysvStub(bytes, ctx, local_idx, dispatch_fn),
        else => unreachable,
    }
}

fn writeIntLittle(comptime T: type, bytes: []u8, value: T) void {
    std.mem.writeInt(T, bytes[0..@sizeOf(T)], value, .little);
}

/// SysV x86_64 ABI (Linux, macOS, *BSD). Incoming: rdi=VmCtx*,
/// rsi/rdx/rcx/r8/r9 = up to 5 more wasm args (raw bit patterns
/// regardless of wasm type — see this file's doc comment), anything
/// beyond that on the caller's stack above the return address.
fn encodeX8664SysvStub(bytes: []u8, ctx: *anyopaque, local_idx: u32, dispatch_fn: LazyDispatchFn) void {
    std.debug.assert(bytes.len >= x86_64_sysv_stub_bytes);

    // Save every argument register. SysV alignment: at stub entry
    // rsp%16==8 (the `call`/inherited-tail-call already pushed/kept
    // one 8-byte return address). 6 pushes keep rsp%16==8 (48 bytes is
    // a multiple of 16); `sub rsp,8` below re-aligns to 0 mod 16
    // immediately before `call rax`.
    const save = [_]u8{
        0x57, // push rdi
        0x56, // push rsi
        0x52, // push rdx
        0x51, // push rcx
        0x41, 0x50, // push r8
        0x41, 0x51, // push r9
        0x48, 0x83, 0xEC, 0x08, // sub rsp, 8
    };
    // movabs rdi, ctx  (first arg to dispatch_fn, SysV C ABI)
    const mov_rdi = [_]u8{ 0x48, 0xBF };
    // mov esi, local_idx  (second arg; writing the 32-bit half
    // zero-extends into rsi, and local_idx is always non-negative)
    const mov_esi = [_]u8{0xBE};
    // movabs rax, dispatch_fn ; call rax
    const movabs_rax = [_]u8{ 0x48, 0xB8 };
    const call_rax = [_]u8{ 0xFF, 0xD0 };
    // Undo the alignment pad, then restore every saved register in
    // reverse order. `rax` (the dispatch call's return value, the
    // resolved callable address) is untouched by every pop here.
    const restore = [_]u8{
        0x48, 0x83, 0xC4, 0x08, // add rsp, 8
        0x41, 0x59, // pop r9
        0x41, 0x58, // pop r8
        0x59, // pop rcx
        0x5A, // pop rdx
        0x5E, // pop rsi
        0x5F, // pop rdi
    };
    // jmp rax — tail-jump into the freshly compiled function with
    // every original argument register (and the untouched stack/
    // return-address layout) exactly as this stub was entered with.
    const jmp_rax = [_]u8{ 0xFF, 0xE0 };

    var cursor: usize = 0;
    @memcpy(bytes[cursor .. cursor + save.len], &save);
    cursor += save.len;
    @memcpy(bytes[cursor .. cursor + mov_rdi.len], &mov_rdi);
    cursor += mov_rdi.len;
    writeIntLittle(u64, bytes[cursor .. cursor + 8], @intFromPtr(ctx));
    cursor += 8;
    @memcpy(bytes[cursor .. cursor + mov_esi.len], &mov_esi);
    cursor += mov_esi.len;
    writeIntLittle(u32, bytes[cursor .. cursor + 4], local_idx);
    cursor += 4;
    @memcpy(bytes[cursor .. cursor + movabs_rax.len], &movabs_rax);
    cursor += movabs_rax.len;
    writeIntLittle(u64, bytes[cursor .. cursor + 8], @intFromPtr(dispatch_fn));
    cursor += 8;
    @memcpy(bytes[cursor .. cursor + call_rax.len], &call_rax);
    cursor += call_rax.len;
    @memcpy(bytes[cursor .. cursor + restore.len], &restore);
    cursor += restore.len;
    @memcpy(bytes[cursor .. cursor + jmp_rax.len], &jmp_rax);
    cursor += jmp_rax.len;

    std.debug.assert(cursor == x86_64_sysv_stub_bytes);
}

/// Win64 ABI (Windows x86_64). Incoming: rcx=VmCtx*, rdx/r8/r9 = up to
/// 3 more wasm args, anything beyond that on the caller's stack (past
/// the mandatory 32-byte shadow space).
fn encodeX8664Win64Stub(bytes: []u8, ctx: *anyopaque, local_idx: u32, dispatch_fn: LazyDispatchFn) void {
    std.debug.assert(bytes.len >= x86_64_win64_stub_bytes);

    const save = [_]u8{
        0x51, // push rcx
        0x52, // push rdx
        0x41, 0x50, // push r8
        0x41, 0x51, // push r9
        // Win64 alignment: at stub entry rsp%16==8. The 4 pushes above
        // add 32 bytes (rsp%16 stays 8). `sub rsp,40` below both
        // satisfies the mandatory 32-byte shadow space AND re-aligns
        // to 0 mod 16 immediately before `call rax` (8 + 32 + 40 == 80,
        // a multiple of 16).
        0x48, 0x83, 0xEC, 0x28, // sub rsp, 40 (0x28)
    };
    const mov_rcx = [_]u8{ 0x48, 0xB9 }; // movabs rcx, ctx
    const mov_edx = [_]u8{0xBA}; // mov edx, local_idx
    const movabs_rax = [_]u8{ 0x48, 0xB8 };
    const call_rax = [_]u8{ 0xFF, 0xD0 };
    const restore = [_]u8{
        0x48, 0x83, 0xC4, 0x28, // add rsp, 40
        0x41, 0x59, // pop r9
        0x41, 0x58, // pop r8
        0x5A, // pop rdx
        0x59, // pop rcx
    };
    const jmp_rax = [_]u8{ 0xFF, 0xE0 };

    var cursor: usize = 0;
    @memcpy(bytes[cursor .. cursor + save.len], &save);
    cursor += save.len;
    @memcpy(bytes[cursor .. cursor + mov_rcx.len], &mov_rcx);
    cursor += mov_rcx.len;
    writeIntLittle(u64, bytes[cursor .. cursor + 8], @intFromPtr(ctx));
    cursor += 8;
    @memcpy(bytes[cursor .. cursor + mov_edx.len], &mov_edx);
    cursor += mov_edx.len;
    writeIntLittle(u32, bytes[cursor .. cursor + 4], local_idx);
    cursor += 4;
    @memcpy(bytes[cursor .. cursor + movabs_rax.len], &movabs_rax);
    cursor += movabs_rax.len;
    writeIntLittle(u64, bytes[cursor .. cursor + 8], @intFromPtr(dispatch_fn));
    cursor += 8;
    @memcpy(bytes[cursor .. cursor + call_rax.len], &call_rax);
    cursor += call_rax.len;
    @memcpy(bytes[cursor .. cursor + restore.len], &restore);
    cursor += restore.len;
    @memcpy(bytes[cursor .. cursor + jmp_rax.len], &jmp_rax);
    cursor += jmp_rax.len;

    std.debug.assert(cursor == x86_64_win64_stub_bytes);
}
