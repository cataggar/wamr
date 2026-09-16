//! Explicit capabilities supplied by a native Unikraft embedding application.
//! No libc, syscalls, signals, TLS, process exit, or global allocator.
const std = @import("std");

pub const Error = error{ OutOfMemory, Unsupported, InvalidMapping, ProtectionFailed, ClockFailed };
pub const Protection = enum { none, read_write, read_execute };

pub const Platform = struct {
    context: *anyopaque,
    page_size: usize = 4096,
    /// Reserve an inaccessible, page-aligned, stable address range.
    reserve: *const fn (*anyopaque, usize) Error![*]align(4096) u8,
    /// Commit previously inaccessible pages RW. Failure must be atomic.
    /// Pages must be zero-filled; the runtime additionally clears new memory.
    commit: *const fn (*anyopaque, [*]align(4096) u8, usize) Error!void,
    /// Transition pages to exactly the requested protection. Never RWX.
    protect: *const fn (*anyopaque, [*]align(4096) u8, usize, Protection) Error!void,
    /// Release the complete reservation, including any uncommitted suffix.
    /// Infallible ownership-release contract: the adapter must ensure teardown.
    unmap: *const fn (*anyopaque, [*]align(4096) u8, usize) void,
    monotonic_ns: *const fn (*anyopaque) Error!u64,

    pub fn validate(self: Platform) Error!void {
        if (self.page_size != 4096) return error.Unsupported;
    }

    pub fn rounded(self: Platform, size: usize) Error!usize {
        try self.validate();
        const n = std.math.add(usize, size, self.page_size - 1) catch return error.OutOfMemory;
        return n & ~(self.page_size - 1);
    }

    pub fn monotonicNs(self: Platform) Error!u64 {
        const ns = try self.monotonic_ns(self.context);
        // The pinned Hyper-V clock uses UINT64_MAX as its saturation sentinel.
        if (ns == std.math.maxInt(u64)) return error.ClockFailed;
        return ns;
    }
};

/// CPU feature bits use native_abi.cpu_features, not Zig's internal feature IDs.
/// CPUID is available at CPL0 and does not depend on hosted OS facilities.
pub fn detectedCpuFeatures() u64 {
    if (@import("builtin").cpu.arch != .x86_64) return 0;
    const base = cpuid(0, 0);
    var features: u64 = 0;
    if (base.a >= 1) {
        const leaf = cpuid(1, 0);
        if (leaf.d & (1 << 26) != 0) features |= 1;
        if (leaf.c & (1 << 19) != 0) features |= 2;
        if (leaf.c & (1 << 23) != 0) features |= 4;
    }
    if (base.a >= 7 and cpuid(7, 0).b & (1 << 3) != 0) features |= 8;
    if (cpuid(0x80000000, 0).a >= 0x80000001 and cpuid(0x80000001, 0).c & (1 << 5) != 0) features |= 16;
    return features;
}

fn cpuid(leaf: u32, subleaf: u32) struct { a: u32, b: u32, c: u32, d: u32 } {
    var a: u32 = undefined;
    var b: u32 = undefined;
    var c: u32 = undefined;
    var d: u32 = undefined;
    asm volatile ("cpuid"
        : [a] "={eax}" (a),
          [b] "={ebx}" (b),
          [c] "={ecx}" (c),
          [d] "={edx}" (d),
        : [leaf] "{eax}" (leaf),
          [subleaf] "{ecx}" (subleaf),
    );
    return .{ .a = a, .b = b, .c = c, .d = d };
}
