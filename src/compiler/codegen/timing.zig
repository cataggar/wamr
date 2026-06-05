//! Shared native-codegen timing instrumentation (issue #778).
//!
//! Mirrors the optimizer's `WAMR_AOT_PASS_TIMING*` diagnostics for the
//! native-codegen phase so a super-linear per-function cost (instruction
//! selection / register allocation / emit) can be attributed without a
//! profiler. Off unless the codegen timing options are enabled, which the
//! `wamrc` driver parses from `WAMR_AOT_CODEGEN_TIMING*` via
//! `passes.codegenTimingOptionsFromEnv`. Disabled is a hard no-op: the
//! per-function backends only construct a `FuncTimer` when logging is live.
//!
//! Output is single-threaded stderr (`std.debug.print`); codegen runs one
//! function at a time per module today, so per-phase wall-clock is
//! meaningful and the lines do not interleave.

const std = @import("std");
const builtin = @import("builtin");
const passes = @import("../ir/passes.zig");

pub const Options = passes.CodegenTimingOptions;

const ns_per_ms: u64 = std.time.ns_per_ms;
const ns_per_us: u64 = std.time.ns_per_us;

/// Monotonic nanosecond clock. Mirrors `passes.passTimingNowNs` — this Zig
/// build has no `std.time.Timer`, so timing uses `clock_gettime(MONOTONIC)`
/// directly. Returns 0 if the platform clock is unavailable.
pub fn nowNs() u64 {
    return switch (comptime builtin.os.tag) {
        .linux => blk: {
            const linux = std.os.linux;
            var ts: linux.timespec = undefined;
            const rc = linux.clock_gettime(.MONOTONIC, &ts);
            if (rc != 0) break :blk 0;
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        .macos, .ios, .tvos, .watchos, .visionos => blk: {
            var ts: std.c.timespec = undefined;
            if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) break :blk 0;
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        .windows => blk: {
            const ntdll = std.os.windows.ntdll;
            var counter: std.os.windows.LARGE_INTEGER = undefined;
            var freq: std.os.windows.LARGE_INTEGER = undefined;
            _ = ntdll.RtlQueryPerformanceCounter(&counter);
            _ = ntdll.RtlQueryPerformanceFrequency(&freq);
            const ticks: u128 = @intCast(counter);
            const hz: u128 = @intCast(freq);
            break :blk @as(u64, @truncate(ticks * std.time.ns_per_s / hz));
        },
        else => 0,
    };
}

/// Per-function sub-phase stopwatch threaded into the x86-64 per-function
/// compile. Records the contiguous `setup` / `liveness` / `regalloc`
/// spans; the caller derives `emit` (everything else, including the emit
/// loop and patch finalisation) as `total - setup - liveness - regalloc`
/// so the four buckets always partition the function's compile time.
pub const Phase = enum { setup, liveness, regalloc };

pub const FuncTimer = struct {
    span_start: u64 = 0,
    setup_ns: u64 = 0,
    liveness_ns: u64 = 0,
    regalloc_ns: u64 = 0,

    pub fn start() FuncTimer {
        return .{};
    }

    pub fn begin(self: *FuncTimer) void {
        self.span_start = nowNs();
    }

    pub fn end(self: *FuncTimer, phase: Phase) void {
        const dt = nowNs() -| self.span_start;
        switch (phase) {
            .setup => self.setup_ns += dt,
            .liveness => self.liveness_ns += dt,
            .regalloc => self.regalloc_ns += dt,
        }
    }
};

/// Decide whether a per-function line should be emitted. Gated at the
/// function level (not per sub-phase) so a slow function always prints its
/// full breakdown: matched `func_filter`, the every-N cadence, or the
/// total exceeding the threshold all qualify.
pub fn shouldLogFunc(opts: Options, module_idx: u32, func_idx: u32, total_ns: u64) bool {
    if (!opts.enabled or !opts.moduleMatches(module_idx)) return false;
    if (opts.func_filter) |f| {
        if (f == func_idx) return true;
    }
    if (opts.every_n_functions != 0 and func_idx % opts.every_n_functions == 0) return true;
    return total_ns >= opts.threshold_ns;
}

pub const FuncReport = struct {
    module_idx: u32,
    func_idx: u32,
    blocks: usize,
    insts: usize,
    reused: bool,
    hash_ns: u64,
    /// Per-function compile call wall-clock (0 for cache reuse hits).
    total_ns: u64,
    setup_ns: u64 = 0,
    liveness_ns: u64 = 0,
    regalloc_ns: u64 = 0,
};

pub fn printFunc(r: FuncReport) void {
    const emit_ns = r.total_ns -| r.setup_ns -| r.liveness_ns -| r.regalloc_ns;
    std.debug.print(
        "[aot-codegen-timing] local_func={d} mod={d} blocks={d} insts={d} reused={} " ++
            "hash_ms={d}.{d:0>3} total_ms={d}.{d:0>3} setup_ms={d}.{d:0>3} " ++
            "liveness_ms={d}.{d:0>3} regalloc_ms={d}.{d:0>3} emit_ms={d}.{d:0>3}\n",
        .{
            r.func_idx,            r.module_idx,              r.blocks,               r.insts,                   r.reused,
            r.hash_ns / ns_per_ms, msFrac(r.hash_ns),         r.total_ns / ns_per_ms, msFrac(r.total_ns),        r.setup_ns / ns_per_ms,
            msFrac(r.setup_ns),    r.liveness_ns / ns_per_ms, msFrac(r.liveness_ns),  r.regalloc_ns / ns_per_ms, msFrac(r.regalloc_ns),
            emit_ns / ns_per_ms,   msFrac(emit_ns),
        },
    );
}

/// aarch64 per-function sub-phase stopwatch (issue #781). The aarch64
/// backend's pipeline differs from x86-64's, so it gets its own phase set
/// rather than reusing `Phase`. Spans accumulate per phase via `begin()` /
/// `end(phase)`; some phases are measured across two disjoint source
/// regions (e.g. `prepass` covers both the dominator/block-order setup and
/// the later clobber/hint/const/FMA scans; `liveness` covers the live-range
/// solve and the kill-list build; `regalloc` covers scalar and v128 alloc).
/// The caller derives `emit` (frame layout + prologue/body/epilogue emit +
/// branch relaxation + patch finalisation) as `total - sum(all spans)` so
/// the buckets partition the function's compile time. `prepass` means
/// "pre-register-allocation analysis", not only instruction scans.
pub const Aarch64Phase = enum {
    scheduling,
    range_split,
    prepass,
    liveness,
    regalloc,
    coalesce,
    post_emit_coalesce,
};

pub const Aarch64FuncTimer = struct {
    span_start: u64 = 0,
    scheduling_ns: u64 = 0,
    range_split_ns: u64 = 0,
    prepass_ns: u64 = 0,
    liveness_ns: u64 = 0,
    regalloc_ns: u64 = 0,
    coalesce_ns: u64 = 0,
    post_emit_coalesce_ns: u64 = 0,

    pub fn start() Aarch64FuncTimer {
        return .{};
    }

    pub fn begin(self: *Aarch64FuncTimer) void {
        self.span_start = nowNs();
    }

    pub fn end(self: *Aarch64FuncTimer, phase: Aarch64Phase) void {
        const dt = nowNs() -| self.span_start;
        switch (phase) {
            .scheduling => self.scheduling_ns += dt,
            .range_split => self.range_split_ns += dt,
            .prepass => self.prepass_ns += dt,
            .liveness => self.liveness_ns += dt,
            .regalloc => self.regalloc_ns += dt,
            .coalesce => self.coalesce_ns += dt,
            .post_emit_coalesce => self.post_emit_coalesce_ns += dt,
        }
    }
};

pub const Aarch64FuncReport = struct {
    module_idx: u32,
    func_idx: u32,
    blocks: usize,
    insts: usize,
    reused: bool,
    hash_ns: u64,
    /// Per-function compile call wall-clock (0 for cache reuse hits).
    total_ns: u64,
    scheduling_ns: u64 = 0,
    range_split_ns: u64 = 0,
    prepass_ns: u64 = 0,
    liveness_ns: u64 = 0,
    regalloc_ns: u64 = 0,
    coalesce_ns: u64 = 0,
    post_emit_coalesce_ns: u64 = 0,
};

pub fn printAarch64Func(r: Aarch64FuncReport) void {
    // Derive emit as the remainder. Sequential saturating subtraction
    // keeps emit_ms >= 0 even if clock jitter makes the spans sum slightly
    // above total.
    const emit_ns = r.total_ns -| r.scheduling_ns -| r.range_split_ns -|
        r.prepass_ns -| r.liveness_ns -| r.regalloc_ns -| r.coalesce_ns -|
        r.post_emit_coalesce_ns;
    std.debug.print(
        "[aot-codegen-timing] local_func={d} mod={d} blocks={d} insts={d} reused={} " ++
            "hash_ms={d}.{d:0>3} total_ms={d}.{d:0>3} sched_ms={d}.{d:0>3} " ++
            "range_split_ms={d}.{d:0>3} prepass_ms={d}.{d:0>3} liveness_ms={d}.{d:0>3} " ++
            "regalloc_ms={d}.{d:0>3} coalesce_ms={d}.{d:0>3} post_coalesce_ms={d}.{d:0>3} " ++
            "emit_ms={d}.{d:0>3}\n",
        .{
            r.func_idx,                       r.module_idx,                        r.blocks,                       r.insts,                     r.reused,
            r.hash_ns / ns_per_ms,            msFrac(r.hash_ns),                   r.total_ns / ns_per_ms,         msFrac(r.total_ns),          r.scheduling_ns / ns_per_ms,
            msFrac(r.scheduling_ns),          r.range_split_ns / ns_per_ms,        msFrac(r.range_split_ns),       r.prepass_ns / ns_per_ms,    msFrac(r.prepass_ns),
            r.liveness_ns / ns_per_ms,        msFrac(r.liveness_ns),               r.regalloc_ns / ns_per_ms,      msFrac(r.regalloc_ns),       r.coalesce_ns / ns_per_ms,
            msFrac(r.coalesce_ns),            r.post_emit_coalesce_ns / ns_per_ms, msFrac(r.post_emit_coalesce_ns), emit_ns / ns_per_ms,        msFrac(emit_ns),
        },
    );
}

pub fn printModuleBegin(opts: Options, module_idx: u32, funcs: usize) void {
    if (!opts.enabled or !opts.moduleMatches(module_idx)) return;
    std.debug.print(
        "[aot-codegen-timing] begin mod={d} funcs={d} threshold_ms={d} every_n_funcs={d}\n",
        .{ module_idx, funcs, opts.threshold_ns / ns_per_ms, opts.every_n_functions },
    );
}

pub const ModuleReport = struct {
    module_idx: u32,
    funcs: usize,
    compiled: usize,
    reused: usize,
    hash_ns: u64,
    compile_ns: u64,
    global_patch_ns: u64,
    total_ns: u64,
};

pub fn printModuleSummary(opts: Options, r: ModuleReport) void {
    if (!opts.enabled or !opts.moduleMatches(r.module_idx)) return;
    std.debug.print(
        "[aot-codegen-timing] module-summary mod={d} funcs={d} compiled={d} reused={d} " ++
            "hash_ms={d}.{d:0>3} compile_ms={d}.{d:0>3} global_patch_ms={d}.{d:0>3} total_ms={d}.{d:0>3}\n",
        .{
            r.module_idx,                  r.funcs,                   r.compiled,               r.reused,
            r.hash_ns / ns_per_ms,         msFrac(r.hash_ns),         r.compile_ns / ns_per_ms, msFrac(r.compile_ns),
            r.global_patch_ns / ns_per_ms, msFrac(r.global_patch_ns), r.total_ns / ns_per_ms,   msFrac(r.total_ns),
        },
    );
}

fn msFrac(ns: u64) u64 {
    return (ns % ns_per_ms) / ns_per_us;
}
