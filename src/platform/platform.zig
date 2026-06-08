//! Cross-platform abstraction layer for WAMR.
//!
//! Replaces the 16 OS-specific C backends in `core/shared/platform/` with a
//! single Zig module built on `std`.  Covers the APIs from both
//! `platform_api_vmcore.h` (memory mapping, mutexes, time, thread identity)
//! and `platform_api_extension.h` (threads, condvars, rwlocks).

const std = @import("std");
const builtin = @import("builtin");

// When wired through build.zig, import config for conditional compilation:
//   const config = @import("../config.zig");
// Omitted here so the file can be tested standalone with `zig test`.

const is_windows = builtin.os.tag == .windows;
const is_linux = builtin.os.tag == .linux;
const is_macos = builtin.os.tag == .macos;
const page_size = std.heap.page_size_min;

// ── Windows NT API imports ──────────────────────────────────────────────

const win = if (is_windows) std.os.windows else undefined;
const ntdll = if (is_windows) @import("std").os.windows.ntdll else undefined;

// Win32 functions not in Zig's std that we need.
const win_extern = if (is_windows) struct {
    extern "kernel32" fn GetCurrentThreadStackLimits(
        LowLimit: *usize,
        HighLimit: *usize,
    ) callconv(.winapi) void;

    extern "kernel32" fn GetThreadTimes(
        hThread: std.os.windows.HANDLE,
        lpCreationTime: *std.os.windows.FILETIME,
        lpExitTime: *std.os.windows.FILETIME,
        lpKernelTime: *std.os.windows.FILETIME,
        lpUserTime: *std.os.windows.FILETIME,
    ) callconv(.winapi) std.os.windows.BOOL;
} else struct {};

// ── 1. Memory mapping ───────────────────────────────────────────────────

pub const MemProt = packed struct {
    read: bool = false,
    write: bool = false,
    exec: bool = false,
    _padding: u5 = 0,
};

pub const MapFlags = packed struct {
    map_32bit: bool = false,
    map_fixed: bool = false,
    _padding: u6 = 0,
};

/// Map memory pages.  Returns the mapped region or null.
pub fn mmap(hint: ?[*]u8, size: usize, prot: MemProt, flags: MapFlags) ?[*]u8 {
    if (size == 0) return null;

    if (is_windows) {
        return mmapWindows(hint, size, prot, flags);
    } else {
        return mmapPosix(hint, size, prot, flags);
    }
}

/// Unmap previously mapped memory.
pub fn munmap(addr: [*]u8, size: usize) void {
    if (is_windows) {
        munmapWindows(addr);
    } else {
        munmapPosix(addr, size);
    }
}

/// Change protection on mapped memory.
pub fn mprotect(addr: [*]u8, size: usize, prot: MemProt) !void {
    if (is_windows) {
        try mprotectWindows(addr, size, prot);
    } else {
        try mprotectPosix(addr, size, prot);
    }
}

// ── Windows memory mapping (NT API) ─────────────────────────────────────

fn mmapWindows(hint: ?[*]u8, size: usize, prot: MemProt, flags: MapFlags) ?[*]u8 {
    if (!is_windows) unreachable;

    var alloc_flags: win.MEM.ALLOCATE = .{ .RESERVE = true };
    if (prot.read or prot.write or prot.exec) {
        alloc_flags.COMMIT = true;
    }

    const page_prot = memProtToWindowsPage(prot);

    var base_addr: ?[*]u8 = if (flags.map_fixed) hint else null;
    var region_size: usize = size;

    const status = ntdll.NtAllocateVirtualMemory(
        win.GetCurrentProcess(),
        @ptrCast(&base_addr),
        0,
        &region_size,
        alloc_flags,
        page_prot,
    );

    if (status == .SUCCESS) {
        return base_addr;
    }
    return null;
}

fn munmapWindows(addr: [*]u8) void {
    if (!is_windows) unreachable;

    var base: ?[*]u8 = addr;
    var region_size: usize = 0;

    _ = ntdll.NtFreeVirtualMemory(
        win.GetCurrentProcess(),
        @ptrCast(&base),
        &region_size,
        .{ .RELEASE = true },
    );
}

fn mprotectWindows(addr: [*]u8, size: usize, prot: MemProt) !void {
    if (!is_windows) unreachable;

    const new_prot = memProtToWindowsPage(prot);
    var old_prot: win.PAGE = undefined;
    var base: ?*anyopaque = @ptrCast(addr);
    var region_size: usize = size;

    const status = ntdll.NtProtectVirtualMemory(
        win.GetCurrentProcess(),
        &base,
        &region_size,
        new_prot,
        &old_prot,
    );

    if (status != .SUCCESS) return error.MprotectFailed;
}

fn memProtToWindowsPage(prot: MemProt) win.PAGE {
    if (!is_windows) unreachable;

    if (prot.exec) {
        if (prot.write) return .{ .EXECUTE_READWRITE = true };
        if (prot.read) return .{ .EXECUTE_READ = true };
        return .{ .EXECUTE = true };
    }
    if (prot.write) return .{ .READWRITE = true };
    if (prot.read) return .{ .READONLY = true };
    return .{ .NOACCESS = true };
}

// ── POSIX memory mapping implementation ─────────────────────────────────

fn mmapPosix(hint: ?[*]u8, size: usize, prot: MemProt, flags: MapFlags) ?[*]u8 {
    if (is_windows) unreachable;

    const posix = std.posix;

    var posix_prot: std.posix.PROT = .{};
    if (prot.read) posix_prot.READ = true;
    if (prot.write) posix_prot.WRITE = true;
    if (prot.exec) posix_prot.EXEC = true;

    var map_flags: std.posix.system.MAP = .{ .TYPE = .PRIVATE, .ANONYMOUS = true };
    if (flags.map_fixed) map_flags.FIXED = true;

    // MAP_32BIT is Linux-only x86_64.
    if (flags.map_32bit and is_linux and builtin.cpu.arch == .x86_64) {
        map_flags.@"32BIT" = true;
    }

    const hint_aligned: ?[*]align(page_size) u8 = if (hint) |h|
        @alignCast(h)
    else
        null;

    const result = posix.mmap(
        hint_aligned,
        size,
        posix_prot,
        map_flags,
        -1,
        0,
    );

    if (result) |slice| {
        return slice.ptr;
    } else |_| {
        return null;
    }
}

fn munmapPosix(addr: [*]u8, size: usize) void {
    if (is_windows) unreachable;

    const aligned: [*]align(page_size) u8 = @alignCast(addr);
    std.posix.munmap(aligned[0..size]);
}

fn mprotectPosix(addr: [*]u8, size: usize, prot: MemProt) !void {
    if (is_windows) unreachable;

    var posix_prot: std.posix.PROT = .{};
    if (prot.read) posix_prot.READ = true;
    if (prot.write) posix_prot.WRITE = true;
    if (prot.exec) posix_prot.EXEC = true;

    const aligned: [*]align(page_size) u8 = @alignCast(addr);
    const rc = std.posix.system.mprotect(aligned, size, posix_prot);
    const err = std.posix.errno(rc);
    if (err != .SUCCESS) return error.MprotectFailed;
}

// ── Reserved address-space helpers (linear-memory backing) ──────────────
//
// `MemoryInstance` (src/runtime/common/types.zig) backs each wasm linear
// memory with a `[]u8`. Historically `MemoryInstance.grow` used
// `allocator.realloc`, which can **relocate** the buffer. wamr's
// vmctx-subscriber machinery refreshes every AOT vmctx's `memory_base`
// mirror after a grow, so generated code keeps working — but **external
// pointers held outside the vmctx mechanism** (StarlingMonkey /
// SpiderMonkey "external strings" referencing wasm memory; host-held
// slices captured across a cross-instance marshal) become dangling.
// Issue #752: TCGC's componentize-js engine reads a 1.3 MiB
// `__spec_files` JSON string as an external string. Subsequent
// `Map.set` inside `compileInner` triggers a `memory.grow` that
// relocates the backing; the external string's char pointer dangles
// and TCGC's module resolver sees partially-corrupted UTF-8 → fails to
// find `@typespec/compiler` with the cryptic "Resolved to / which is
// outside package file://" error.
//
// `reserveAddressSpace` reserves `size` bytes of virtual address space
// (POSIX `mmap(MAP_PRIVATE|MAP_ANONYMOUS)` with `PROT_NONE`; Windows
// `VirtualAlloc(MEM_RESERVE)`). The returned region is **not** backed
// by physical memory and cannot be read or written until `commitPages`
// changes the protection on a sub-range. `releaseAddressSpace`
// releases the entire reservation.
//
// `commitPages(addr, size)` enables read/write on `[addr, addr+size)`.
// On POSIX this is `mprotect(PROT_READ|PROT_WRITE)`, which on Linux
// triggers lazy demand-paging — physical pages are only allocated as
// the wasm guest first touches them. On Windows this is
// `VirtualAlloc(MEM_COMMIT, PAGE_READWRITE)` on top of the existing
// reservation.
//
// Stability contract: the returned base pointer is valid for the
// lifetime of the reservation (until `releaseAddressSpace`). The
// caller may keep slices into the address range across any number of
// `commitPages` calls — extending the committed range never moves
// existing addresses.

/// True on platforms where `reserveAddressSpace` + `commitPages` are
/// supported. On unsupported platforms callers must fall back to the
/// pre-#752 `allocator.realloc` path (which moves the buffer and is
/// unsafe for external-string aliases).
pub const supports_reserved_memory: bool = !is_windows;

/// Reserve `size` bytes of virtual address space, no physical pages
/// backed. Returns the base pointer or null on failure. Free with
/// `releaseAddressSpace`.
pub fn reserveAddressSpace(size: usize) ?[*]align(page_size) u8 {
    if (size == 0) return null;
    if (is_windows) {
        // Windows: reservation needs a separate commit step — gate at
        // call sites via `supports_reserved_memory` and fall back to
        // the allocator path here. (Tracking a Windows-native
        // reserve/commit implementation as a follow-up.)
        return null;
    }
    const ptr = mmap(null, size, .{}, .{}) orelse return null;
    return @alignCast(ptr);
}

/// Release a previously reserved address range. `size` must match the
/// `reserveAddressSpace` call's `size`.
pub fn releaseAddressSpace(addr: [*]align(page_size) u8, size: usize) void {
    if (size == 0) return;
    if (is_windows) unreachable;
    munmap(addr, size);
}

/// Commit physical-page backing for `[addr, addr+size)` within a
/// previously reserved range. On POSIX this is
/// `mprotect(PROT_READ|PROT_WRITE)`; pages are demand-paged on first
/// access. Returns `error.MprotectFailed` on failure.
pub fn commitPages(addr: [*]align(page_size) u8, size: usize) !void {
    if (size == 0) return;
    if (is_windows) unreachable;
    try mprotect(addr, size, .{ .read = true, .write = true });
}

// ── 2. Threading ────────────────────────────────────────────────────────

pub const Thread = std.Thread;

/// Simple spinlock mutex that doesn't require Io (Zig 0.16 moved std.Thread.Mutex behind Io).
pub const Mutex = struct {
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

    pub const init: Mutex = .{ .state = std.atomic.Value(u8).init(0) };

    pub fn lock(self: *Mutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn unlock(self: *Mutex) void {
        self.state.store(0, .release);
    }
};

/// Simple condition variable stub (Zig 0.16 moved std.Thread.Condition behind Io).
pub const Condition = struct {
    flag: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

    pub fn wait(self: *Condition, mutex: *Mutex) void {
        mutex.unlock();
        while (self.flag.load(.acquire) == 0) {
            std.atomic.spinLoopHint();
        }
        self.flag.store(0, .release);
        mutex.lock();
    }

    pub fn signal(self: *Condition) void {
        self.flag.store(1, .release);
    }

    pub fn broadcast(self: *Condition) void {
        self.flag.store(1, .release);
    }
};

/// Get the current thread ID.
pub fn selfThread() std.Thread.Id {
    return std.Thread.getCurrentId();
}

/// Get the current thread's stack boundary (lowest valid address).
/// Returns null if not implemented for this platform.
pub fn threadGetStackBoundary() ?[*]u8 {
    if (is_windows) {
        return threadGetStackBoundaryWindows();
    } else if (is_linux) {
        return threadGetStackBoundaryLinux();
    } else if (is_macos) {
        return threadGetStackBoundaryMacos();
    }
    return null;
}

fn threadGetStackBoundaryWindows() ?[*]u8 {
    if (!is_windows) unreachable;

    var low_limit: usize = 0;
    var high_limit: usize = 0;
    win_extern.GetCurrentThreadStackLimits(&low_limit, &high_limit);

    if (low_limit == 0) return null;

    // Skip past the guard pages: 4 system guard pages + 1 safety page.
    const guard_offset = 5 * page_size;
    const boundary = low_limit + guard_offset;
    return @ptrFromInt(boundary);
}

fn threadGetStackBoundaryLinux() ?[*]u8 {
    if (!is_linux) unreachable;

    // Use pthread_attr_getstack via the libc interface.
    const c = @cImport({
        @cInclude("pthread.h");
    });

    var attr: c.pthread_attr_t = undefined;
    if (c.pthread_getattr_np(c.pthread_self(), &attr) != 0) return null;
    defer _ = c.pthread_attr_destroy(&attr);

    var stack_addr: ?*anyopaque = null;
    var stack_size: usize = 0;
    if (c.pthread_attr_getstack(&attr, &stack_addr, &stack_size) != 0) return null;

    var guard_size: usize = 0;
    _ = c.pthread_attr_getguardsize(&attr, &guard_size);

    const base = @intFromPtr(stack_addr) + guard_size;
    return @ptrFromInt(base);
}

fn threadGetStackBoundaryMacos() ?[*]u8 {
    if (!is_macos) unreachable;

    const c = @cImport({
        @cInclude("pthread.h");
    });

    const self = c.pthread_self();
    const stack_addr = @intFromPtr(c.pthread_get_stackaddr_np(self));
    const stack_size = c.pthread_get_stacksize_np(self);

    if (stack_addr == 0 or stack_size == 0) return null;

    // On macOS stack_addr is the *top* (highest address) of the stack.
    return @ptrFromInt(stack_addr - stack_size);
}

/// Sleep for the specified number of microseconds.
pub fn usleep(us: u64) void {
    if (is_windows) {
        usleepWindows(us);
    } else {
        usleepPosix(us);
    }
}

fn usleepWindows(us: u64) void {
    if (!is_windows) unreachable;

    // NtDelayExecution takes a LARGE_INTEGER in 100ns units, negative for relative delay.
    const hundred_ns = @as(u64, @min(us * 10, @as(u64, @intCast(std.math.maxInt(i64)))));
    const delay: win.LARGE_INTEGER = -@as(i64, @intCast(hundred_ns));
    _ = ntdll.NtDelayExecution(.FALSE, &delay);
}

fn usleepPosix(us: u64) void {
    if (is_windows) unreachable;
    const ns = us * std.time.ns_per_us;
    const ts = std.posix.timespec{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.posix.system.nanosleep(&ts, null);
}

// ── 3. Time ─────────────────────────────────────────────────────────────

/// Monotonic time since boot in microseconds.
pub fn timeGetBootUs() u64 {
    if (is_windows) {
        return timeGetBootUsWindows();
    } else {
        return timeGetBootUsPosix();
    }
}

fn timeGetBootUsWindows() u64 {
    if (!is_windows) unreachable;

    var counter: i64 = 0;
    _ = ntdll.RtlQueryPerformanceCounter(&counter);
    var freq: i64 = 0;
    _ = ntdll.RtlQueryPerformanceFrequency(&freq);

    if (freq == 0) return 0;

    const counter_u: u64 = @intCast(counter);
    const freq_u: u64 = @intCast(freq);
    // Convert ticks to microseconds: ticks * 1_000_000 / freq.
    // Use 128-bit multiply to avoid overflow.
    const wide = @as(u128, counter_u) * std.time.us_per_s;
    return @intCast(wide / freq_u);
}

fn timeGetBootUsPosix() u64 {
    if (is_windows) unreachable;
    var ts: std.posix.timespec = .{ .sec = 0, .nsec = 0 };
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    const ns: u64 = @intCast(ts.sec * std.time.ns_per_s + ts.nsec);
    return ns / std.time.ns_per_us;
}

/// Current thread CPU time in microseconds (best-effort).
/// Falls back to monotonic time when per-thread CPU time is unavailable.
pub fn timeThreadCputimeUs() u64 {
    if (is_windows) {
        return timeThreadCputimeWindows();
    } else {
        return timeThreadCputimePosix();
    }
}

fn timeThreadCputimeWindows() u64 {
    if (!is_windows) unreachable;

    var creation: win.FILETIME = undefined;
    var exit: win.FILETIME = undefined;
    var kernel: win.FILETIME = undefined;
    var user: win.FILETIME = undefined;

    const handle = win.GetCurrentThread();
    const rc = win_extern.GetThreadTimes(handle, &creation, &exit, &kernel, &user);
    if (rc == .FALSE) return 0;

    const k_ticks = @as(u64, kernel.dwHighDateTime) << 32 | kernel.dwLowDateTime;
    const u_ticks = @as(u64, user.dwHighDateTime) << 32 | user.dwLowDateTime;
    // 100-ns ticks → microseconds.
    return (k_ticks + u_ticks) / 10;
}

fn timeThreadCputimePosix() u64 {
    if (is_windows) unreachable;
    const ns = std.time.nanoTimestamp();
    return @intCast(@divFloor(ns, std.time.ns_per_us));
}

// ── 4. Console I/O ──────────────────────────────────────────────────────

/// Print to stderr (for runtime logging).
pub fn print(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt, args);
}

// ── 5. Flush caches ─────────────────────────────────────────────────────

/// Flush data cache (no-op on x86).
pub fn dcacheFlush() void {}

/// Flush instruction cache after writing code.
pub fn icacheFlush(start: [*]u8, len: usize) void {
    switch (builtin.cpu.arch) {
        .aarch64 => icacheFlushAarch64(start, len),
        .arm, .thumb => icacheFlushArm(start, len),
        else => {
            // x86/x86_64: instruction cache is coherent with data cache.
        },
    }
}

/// Whether the platform allocates JIT/trampoline code via macOS's
/// per-thread MAP_JIT write-protection (Apple Silicon). On such
/// targets an RWX region cannot simply be `mmap`ed and written; the
/// region is mapped `MAP_JIT` and each thread flips between
/// write-enabled and execute-enabled with `jitWriteProtect`.
pub const macos_jit = is_macos and builtin.cpu.arch == .aarch64;

extern "c" fn pthread_jit_write_protect_np(enabled: c_int) void;

/// Toggle the calling thread's view of MAP_JIT pages between
/// executable (`enable = true`) and writable (`enable = false`).
/// No-op on every target except macOS aarch64. Callers writing into a
/// MAP_JIT region must wrap the write in
/// `jitWriteProtect(false) … jitWriteProtect(true)` and then
/// `icacheFlush` the modified range. The protection is per-thread, so
/// the thread that later *executes* the code must be in the
/// execute-enabled state (which `jitWriteProtect(true)` leaves it in).
pub fn jitWriteProtect(enable: bool) void {
    if (comptime macos_jit) {
        pthread_jit_write_protect_np(if (enable) 1 else 0);
    }
}

fn icacheFlushAarch64(start: [*]u8, len: usize) void {
    if (is_macos) {
        // macOS: use sys_icache_invalidate from libsystem.
        const c = @cImport({
            @cInclude("libkern/OSCacheControl.h");
        });
        c.sys_icache_invalidate(@ptrCast(start), len);
    } else if (is_linux) {
        // Linux AArch64: clear d-cache and invalidate i-cache line by line.
        const cache_line: usize = 64;
        const base = @intFromPtr(start);
        const end = base + len;

        var addr = base & ~(cache_line - 1);
        while (addr < end) : (addr += cache_line) {
            asm volatile ("dc cvau, %[addr]"
                :
                : [addr] "r" (addr),
                : .{ .memory = true }
            );
        }
        asm volatile ("dsb ish" ::: .{ .memory = true });

        addr = base & ~(cache_line - 1);
        while (addr < end) : (addr += cache_line) {
            asm volatile ("ic ivau, %[addr]"
                :
                : [addr] "r" (addr),
                : .{ .memory = true }
            );
        }
        asm volatile ("dsb ish" ::: .{ .memory = true });
        asm volatile ("isb" ::: .{ .memory = true });
    }
}

fn icacheFlushArm(start: [*]u8, len: usize) void {
    if (is_linux) {
        const base = @intFromPtr(start);
        const end_addr = base + len;
        // ARM Linux cacheflush syscall: syscall 0xf0002 (ARM_NR_cacheflush)
        _ = std.os.linux.syscall3(
            @enumFromInt(0xf0002),
            base,
            end_addr,
            0,
        );
    } else {
        // No cache flush needed on non-Linux ARM.
    }
}

// ── 6. Tests ────────────────────────────────────────────────────────────

test "mmap/munmap roundtrip" {
    const size = page_size;
    const ptr = mmap(null, size, .{ .read = true, .write = true }, .{}) orelse
        return error.MmapFailed;

    // Write a pattern and read it back.
    ptr[0] = 0xAB;
    ptr[size - 1] = 0xCD;
    try std.testing.expectEqual(@as(u8, 0xAB), ptr[0]);
    try std.testing.expectEqual(@as(u8, 0xCD), ptr[size - 1]);

    munmap(ptr, size);
}

test "mprotect changes permissions" {
    const size = page_size;
    const ptr = mmap(null, size, .{ .read = true, .write = true }, .{}) orelse
        return error.MmapFailed;
    defer munmap(ptr, size);

    // Write while writable.
    ptr[0] = 42;
    try std.testing.expectEqual(@as(u8, 42), ptr[0]);

    // Make read-only.
    try mprotect(ptr, size, .{ .read = true });

    // Make read-write again so we can verify the call succeeded without crashing.
    try mprotect(ptr, size, .{ .read = true, .write = true });
    ptr[0] = 99;
    try std.testing.expectEqual(@as(u8, 99), ptr[0]);
}

test "selfThread returns non-zero" {
    const tid = selfThread();
    try std.testing.expect(tid != 0);
}

test "usleep sleeps approximately correct duration" {
    const target_us: u64 = 50_000; // 50 ms
    const tolerance_us: u64 = 200_000; // generous 200 ms tolerance for CI

    const before = timeGetBootUs();
    usleep(target_us);
    const after = timeGetBootUs();

    const elapsed = after - before;
    // Must have slept at least ~half the target (kernel scheduling jitter).
    try std.testing.expect(elapsed >= target_us / 2);
    // Must not have slept absurdly long.
    try std.testing.expect(elapsed < target_us + tolerance_us);
}

test "timeGetBootUs returns increasing values" {
    const t1 = timeGetBootUs();
    usleep(1_000); // 1 ms
    const t2 = timeGetBootUs();
    try std.testing.expect(t2 > t1);
}
