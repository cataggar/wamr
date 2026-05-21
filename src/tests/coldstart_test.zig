//! In-process cold-start budget tests (issue #395).
//!
//! Companion to the subprocess timing harness in #394. Subprocess timing
//! measures the user-visible total but is too noisy on hosted CI to gate
//! small (< 100 µs) regressions, and bundles in process-launch overhead
//! from the kernel/libc/Zig startup that's outside WAMR's control. This
//! file isolates the WAMR-internal slice of cold-start so even GitHub-
//! hosted x86 runners can reliably catch regressions there.
//!
//! Measured baseline (subprocess median, AArch64 D16pds_v6 `ab887c59`):
//!
//!     noop.cwasm    ~ 0.78 ms
//!     noop.wasm     ~ 0.82 ms
//!
//! After subtracting Linux process spawn (~200–400 µs), Zig stdlib init
//! (allocator setup, args slicing, arena init), and the noop wasm/cwasm
//! file read, the WAMR-attributable slice is plausibly:
//!
//!     noop.cwasm    < 100 µs   (load + instantiate + map + call)
//!     noop.wasm     < 200 µs   (parse + load + instantiate + interp)
//!
//! Budgets are deliberately set to **3×** the realistic upper bound so
//! routine timer jitter on shared CI runners does not fire false
//! positives, while a real ≥3× regression — the kind a stray eager init
//! would introduce — fails the test loudly.
//!
//!     wasm budget   500 µs   (1000 µs on macOS — see #395 jitter note)
//!     cwasm budget  200 µs   (400 µs on macOS — see #395 jitter note)
//!
//! Skip from the build with `-Dskip-coldstart=true`. See
//! https://github.com/cataggar/wamr/issues/395 for context and #394 for
//! the subprocess companion.

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const wamr = @import("wamr");
const aot_loader = wamr.aot_loader;
const aot_runtime = wamr.aot_runtime;

const coldstart_options = @import("coldstart_options");

/// 36-byte `(module (func (export "_start")))`. Committed at
/// `tests/coldstart/noop.wasm` and embedded at compile time so the
/// timed loop never touches the filesystem.
const NOOP_WASM = @embedFile("noop_wasm");

/// `wamrc compile`-produced AOT image of `NOOP_WASM`, generated at
/// build time via `b.addRunArtifact(wamrc)` (see `build.zig`).
const NOOP_CWASM = @embedFile("noop_cwasm");

/// Number of untimed iterations executed before sampling. Warms page
/// cache, branch predictors, and any lazy allocator state.
const WARMUP = 5;

/// Number of timed samples per test. Median is `samples[12]`.
const SAMPLES = 25;

/// Wasm-path budget — see top-of-file rationale.
///
/// macOS GHA runners exhibit higher variance than Linux/Windows
/// counterparts; the macos-arm64 job hit `median=344µs vs budget=200µs`
/// on PR #496's cwasm test despite no compiler change touching the
/// cold-start path. Double the budgets on macOS to absorb runner
/// jitter without losing the real-regression detection on the more
/// stable archs.
const WASM_BUDGET_NS: u64 = if (builtin.os.tag == .macos) 1_000_000 else 500_000;

/// Cwasm-path budget — see top-of-file rationale (and macOS note on
/// `WASM_BUDGET_NS`).
const CWASM_BUDGET_NS: u64 = if (builtin.os.tag == .macos) 600_000 else 200_000;

/// Runtime arch gate for the cwasm half. Mirrors `aot_supported` in
/// `differential.zig`. On non-AOT-executable targets the cwasm test
/// returns `error.SkipZigTest`.
const aot_supported = switch (builtin.cpu.arch) {
    .x86_64, .aarch64 => true,
    else => false,
};

/// Read `CLOCK_MONOTONIC` in nanoseconds. Mirrors the cross-platform
/// timer pattern used by `src/compiler/bench_codegen.zig`: Linux uses
/// the raw syscall (no libc dependency); Darwin uses libc (which the
/// test module gets via `clock_gettime` on libSystem when it links
/// libc); Windows converts `QueryPerformanceCounter` ticks to ns.
fn nowNanos() u64 {
    return switch (comptime builtin.os.tag) {
        .linux => blk: {
            var ts: std.os.linux.timespec = undefined;
            _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s +
                @as(u64, @intCast(ts.nsec));
        },
        .macos, .ios, .tvos, .watchos, .visionos => blk: {
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s +
                @as(u64, @intCast(ts.nsec));
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
        else => @compileError("coldstart test needs a portable monotonic timer for this target"),
    };
}

/// One end-to-end load+invoke of `NOOP_WASM` through the interpreter
/// pipeline. Mirrors the call sequence in `src/main.zig::runWasm`
/// (Runtime → loadModule → instantiate → callVoid("_start")) so the
/// budget tracks the actual CLI cold-start path.
fn runOneWasm(allocator: std.mem.Allocator) !void {
    var runtime = wamr.wamr.Runtime.init(allocator);
    defer runtime.deinit();
    var module = try runtime.loadModule(NOOP_WASM);
    defer module.deinit();
    var instance = try module.instantiate();
    defer instance.deinit();
    try instance.callVoid("_start", &.{});
}

/// One end-to-end load+invoke of `NOOP_CWASM` through the AOT pipeline.
/// Mirrors `src/main.zig::runAotReal` (load → instantiate →
/// mapCodeExecutable → findExportFunc → callFunc) and additionally
/// pairs `aot_loader.load` with `aot_loader.unload` so the loop is
/// leak-clean under `std.testing.allocator`.
fn runOneCwasm(allocator: std.mem.Allocator) !void {
    var aot_module = try aot_loader.load(NOOP_CWASM, allocator);
    defer aot_loader.unload(&aot_module, allocator);

    const aot_inst = try aot_runtime.instantiate(&aot_module, allocator);
    defer aot_runtime.destroy(aot_inst);

    try aot_runtime.mapCodeExecutable(aot_inst);

    const func_idx = aot_runtime.findExportFunc(aot_inst, "_start") orelse
        return error.FunctionNotFound;
    try aot_runtime.callFunc(aot_inst, func_idx, void);
}

/// Run `runOne` `SAMPLES` times under `clock_gettime(MONOTONIC)`,
/// preceded by `WARMUP` untimed iterations, and assert the median is
/// at most `budget_ns`. On a budget overrun, prints the median/budget
/// plus min/max for diagnosis and returns `error.ColdstartRegression`.
fn measureAndAssert(
    comptime label: []const u8,
    runOne: fn (std.mem.Allocator) anyerror!void,
    budget_ns: u64,
) !void {
    const allocator = testing.allocator;

    for (0..WARMUP) |_| {
        try runOne(allocator);
    }

    var samples: [SAMPLES]u64 = undefined;
    for (&samples) |*s| {
        const t0 = nowNanos();
        try runOne(allocator);
        const t1 = nowNanos();
        s.* = t1 - t0;
    }

    std.sort.heap(u64, &samples, {}, std.sort.asc(u64));
    const median = samples[SAMPLES / 2];

    if (median > budget_ns) {
        std.debug.print(
            "[coldstart] {s} regressed: median={d}ns budget={d}ns " ++
                "min={d}ns max={d}ns (samples=" ++
                std.fmt.comptimePrint("{d}", .{SAMPLES}) ++ ")\n",
            .{ label, median, budget_ns, samples[0], samples[SAMPLES - 1] },
        );
        return error.ColdstartRegression;
    }
}

test "coldstart: noop.wasm load+invoke under budget" {
    if (coldstart_options.skip) return error.SkipZigTest;
    try measureAndAssert("noop.wasm", runOneWasm, WASM_BUDGET_NS);
}

test "coldstart: noop.cwasm load+invoke under budget" {
    if (coldstart_options.skip) return error.SkipZigTest;
    if (!aot_supported) return error.SkipZigTest;
    try measureAndAssert("noop.cwasm", runOneCwasm, CWASM_BUDGET_NS);
}
