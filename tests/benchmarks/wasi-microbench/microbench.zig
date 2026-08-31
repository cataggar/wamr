//! WASI host-path micro-benchmark / regression detector (#583, W11-6).
//!
//! Parallel to the CoreMark + cold-start benches: this harness exercises
//! WAMR's `executor.dispatchCanonBuiltin` → `AsyncStream` → host_driver
//! pipeline — the canonical lowering used by every Preview-3 socket /
//! filesystem stream binding in `src/component/wasi_cli_adapter.zig`
//! (TCP send/receive, UDP receive, fs read/write-via-stream).
//!
//! Synthetic host_drivers stand in for the production
//! `wasiCliAdapter.tcpReceiveStreamOnReadInto` / `udpReceiveStreamOnRead`
//! / `fsReadStreamOnReadInto` / `fsWriteViaStreamOnWriteFrom` callbacks.
//! The drivers' cost shape (one `@memcpy` of `bytes_per_call` into / out
//! of the borrowed guest slice, no syscall) intentionally isolates the
//! WAMR-side overhead from kernel jitter so the bench produces a stable
//! signal on shared CI runners. The bench does *not* go through real
//! sockets / `pwrite(2)` — that's covered by the conformance gates;
//! this is a regression detector for the canonical lowering itself.
//!
//! ## Scenarios
//!
//! | name                              | shape                          |
//! |-----------------------------------|--------------------------------|
//! | `http-service-keepalive-100rt`    | 100 × (read 256 B + write 256 B) — request/response keep-alive cost |
//! | `udp-receive-1mb`                 | 128 × `stream.read` 8 KiB datagrams (1 MiB) |
//! | `fs-write-via-stream-1mb`         |  16 × `stream.write` 64 KiB (1 MiB) |
//! | `fs-read-via-stream-1mb`          |  16 × `stream.read` 64 KiB (1 MiB) |
//!
//! Each scenario runs `--samples` times (default 10). The harness
//! reports median + p95 wall-clock and peak RSS, then compares the
//! median against the budget in `tests/benchmarks/wasi-microbench/budget.json`.
//! A scenario whose median exceeds `median_ns_budget × (1 + threshold)`
//! is flagged as a regression and the process exits non-zero.
//!
//! ## Usage
//!
//! ```
//! zig build wasi-microbench
//! zig build wasi-microbench -- --samples 20
//! zig build wasi-microbench -- --budget tests/benchmarks/wasi-microbench/budget.json
//! zig build wasi-microbench -- --json out.json
//! zig build wasi-microbench -- --no-budget   # skip regression check
//! ```
//!
//! Update the budget when an intentional perf change lands: see
//! `docs/wasi.md` § "Updating the wasi-microbench budget".

const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");

const async_mod = wamr.component_async;
const ctypes = wamr.component_types;
const instance_mod = wamr.component_instance;
const executor = wamr.component_executor;
const core_types_mod = wamr.types;
const inst_mod_core = wamr.instance;
const ExecEnv = wamr.exec_env.ExecEnv;

fn nowNs() u64 {
    return wamr.platform.timeGetBootUs() * std.time.ns_per_us;
}

// ── Scenario table ────────────────────────────────────────────────────

const ScenarioKind = enum { read, write, rt };

const Scenario = struct {
    name: []const u8,
    kind: ScenarioKind,
    iters_per_sample: u32,
    bytes_per_iter: u32,
};

const scenarios = [_]Scenario{
    .{ .name = "http-service-keepalive-100rt", .kind = .rt, .iters_per_sample = 100, .bytes_per_iter = 256 },
    .{ .name = "udp-receive-1mb", .kind = .read, .iters_per_sample = 128, .bytes_per_iter = 8 * 1024 },
    .{ .name = "fs-write-via-stream-1mb", .kind = .write, .iters_per_sample = 16, .bytes_per_iter = 64 * 1024 },
    .{ .name = "fs-read-via-stream-1mb", .kind = .read, .iters_per_sample = 16, .bytes_per_iter = 64 * 1024 },
};

// ── Synthetic host drivers ────────────────────────────────────────────

const SyntheticSource = struct {
    pattern: []const u8,
    bytes_per_call: u32,

    fn fill(self: *SyntheticSource, dst: []u8) u32 {
        const n = @min(self.bytes_per_call, dst.len);
        const m = @min(n, self.pattern.len);
        @memcpy(dst[0..m], self.pattern[0..m]);
        return @intCast(m);
    }
};

const SyntheticSink = struct {
    drained_bytes: u64 = 0,
    last_checksum: u64 = 0,

    fn drain(self: *SyntheticSink, src: []const u8) void {
        self.drained_bytes += src.len;
        // Touch the bytes so the optimiser doesn't elide the executor's
        // borrowed-slice argument — we want to measure the cost of
        // *passing* the slice through, not let LLVM fold it away.
        var s: u64 = self.last_checksum;
        for (src) |b| s +%= b;
        self.last_checksum = s;
    }
};

fn driverReadInto(ctx: ?*anyopaque, dst: []u8) async_mod.HostStreamReadInto {
    const src: *SyntheticSource = @ptrCast(@alignCast(ctx.?));
    if (dst.len == 0) return .{ .action = .would_block };
    const n = src.fill(dst);
    if (n == 0) return .{ .action = .would_block };
    return .{ .action = .progressed, .bytes_written = n };
}

fn driverWriteFrom(ctx: ?*anyopaque, src: []const u8) async_mod.HostStreamAction {
    const sink: *SyntheticSink = @ptrCast(@alignCast(ctx.?));
    sink.drain(src);
    return .progressed;
}

// ── Bench scaffolding (single sample) ─────────────────────────────────

const StreamFixture = struct {
    inst: *instance_mod.ComponentInstance,
    core_inst: *core_types_mod.ModuleInstance,
    env: *ExecEnv,
    handle: u32,
    src: SyntheticSource,
    sink: SyntheticSink,
    pattern: []u8,
    allocator: std.mem.Allocator,

    fn init(
        allocator: std.mem.Allocator,
        bytes_per_iter: u32,
        comp: *ctypes.Component,
    ) !StreamFixture {
        const inst = try instance_mod.instantiate(comp, allocator);
        errdefer inst.deinit();

        const mem_size: usize = std.math.ceilPowerOfTwo(usize, @as(usize, bytes_per_iter) * 2) catch
            @as(usize, bytes_per_iter) * 2;
        try inst.enableTestMem(allocator, mem_size);

        const module_storage = try allocator.create(core_types_mod.WasmModule);
        module_storage.* = .{};
        const core_inst = try inst_mod_core.instantiate(module_storage, allocator);
        errdefer inst_mod_core.destroy(core_inst);

        const env = try ExecEnv.create(core_inst, 64, allocator);
        errdefer env.destroy();

        try executor.dispatchCanonBuiltin(
            inst,
            .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
            env,
            null,
            allocator,
        );
        const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

        const pattern = try allocator.alloc(u8, bytes_per_iter);
        var pi: u32 = 0;
        while (pi < pattern.len) : (pi += 1) pattern[pi] = @truncate(pi);

        return .{
            .inst = inst,
            .core_inst = core_inst,
            .env = env,
            .handle = handle,
            .src = .{ .pattern = pattern, .bytes_per_call = bytes_per_iter },
            .sink = .{},
            .pattern = pattern,
            .allocator = allocator,
        };
    }

    fn deinit(self: *StreamFixture) void {
        self.allocator.free(self.pattern);
        self.env.destroy();
        // core_inst owns a heap-allocated WasmModule; recover the
        // backing pointer before destroying.
        const module_ptr = self.core_inst.module;
        inst_mod_core.destroy(self.core_inst);
        self.allocator.destroy(module_ptr);
        self.inst.disableTestMem();
        self.inst.deinit();
    }

    fn installReadDriver(self: *StreamFixture) void {
        var stream_lease = self.inst.streams.acquire(self.handle).?;
        defer stream_lease.release();
        const s = stream_lease.value();
        s.host_driver = .{
            .context = @as(*anyopaque, @ptrCast(&self.src)),
            .on_read_into = &driverReadInto,
        };
    }

    fn installWriteDriver(self: *StreamFixture) void {
        var stream_lease = self.inst.streams.acquire(self.handle).?;
        defer stream_lease.release();
        const s = stream_lease.value();
        s.host_driver = .{
            .context = @as(*anyopaque, @ptrCast(&self.sink)),
            .on_write_from = &driverWriteFrom,
        };
    }
};

// A persistent per-scenario Component shell. The instantiate path doesn't
// retain a reference to it, so a single shared one is safe to reuse.
var component_shell_types = [_]ctypes.TypeDef{.{ .val = .u8 }};
var component_shell: ctypes.Component = .{
    .core_modules = &.{},
    .core_instances = &.{},
    .core_types = &.{},
    .components = &.{},
    .instances = &.{},
    .aliases = &.{},
    .types = &component_shell_types,
    .canons = &.{},
    .imports = &.{},
    .exports = &.{},
};

fn runReadSample(allocator: std.mem.Allocator, scn: Scenario) !u64 {
    var fx = try StreamFixture.init(allocator, scn.bytes_per_iter, &component_shell);
    defer fx.deinit();
    fx.installReadDriver();

    const dst_ptr: u32 = 0;
    const start = nowNs();
    var i: u32 = 0;
    while (i < scn.iters_per_sample) : (i += 1) {
        try fx.env.pushI32(@bitCast(fx.handle));
        try fx.env.pushI32(@bitCast(dst_ptr));
        try fx.env.pushI32(@bitCast(scn.bytes_per_iter));
        try executor.dispatchCanonBuiltin(
            fx.inst,
            .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
            fx.env,
            null,
            allocator,
        );
        _ = try fx.env.popI32();
    }
    return nowNs() - start;
}

fn runWriteSample(allocator: std.mem.Allocator, scn: Scenario) !u64 {
    var fx = try StreamFixture.init(allocator, scn.bytes_per_iter, &component_shell);
    defer fx.deinit();
    fx.installWriteDriver();

    const src_ptr: u32 = 0;
    // Seed the guest source bytes once.
    {
        const src_bytes = fx.inst.writableGuestBytes(src_ptr, scn.bytes_per_iter).?;
        var pi: u32 = 0;
        while (pi < scn.bytes_per_iter) : (pi += 1) src_bytes[pi] = @truncate(pi);
    }

    const start = nowNs();
    var i: u32 = 0;
    while (i < scn.iters_per_sample) : (i += 1) {
        try fx.env.pushI32(@bitCast(fx.handle));
        try fx.env.pushI32(@bitCast(src_ptr));
        try fx.env.pushI32(@bitCast(scn.bytes_per_iter));
        try executor.dispatchCanonBuiltin(
            fx.inst,
            .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
            fx.env,
            null,
            allocator,
        );
        _ = try fx.env.popI32();
    }
    return nowNs() - start;
}

fn runRtSample(allocator: std.mem.Allocator, scn: Scenario) !u64 {
    // Round-trip = (request write into write-stream + response read
    // from read-stream) per RT. We keep one read stream + one write
    // stream alive across the entire sample to model keep-alive.
    var rx = try StreamFixture.init(allocator, scn.bytes_per_iter, &component_shell);
    defer rx.deinit();
    rx.installReadDriver();

    var tx = try StreamFixture.init(allocator, scn.bytes_per_iter, &component_shell);
    defer tx.deinit();
    tx.installWriteDriver();

    const src_ptr: u32 = 0;
    const dst_ptr: u32 = 0;
    {
        const src_bytes = tx.inst.writableGuestBytes(src_ptr, scn.bytes_per_iter).?;
        var pi: u32 = 0;
        while (pi < scn.bytes_per_iter) : (pi += 1) src_bytes[pi] = @truncate(pi);
    }

    const start = nowNs();
    var i: u32 = 0;
    while (i < scn.iters_per_sample) : (i += 1) {
        // request → write
        try tx.env.pushI32(@bitCast(tx.handle));
        try tx.env.pushI32(@bitCast(src_ptr));
        try tx.env.pushI32(@bitCast(scn.bytes_per_iter));
        try executor.dispatchCanonBuiltin(
            tx.inst,
            .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
            tx.env,
            null,
            allocator,
        );
        _ = try tx.env.popI32();

        // response → read
        try rx.env.pushI32(@bitCast(rx.handle));
        try rx.env.pushI32(@bitCast(dst_ptr));
        try rx.env.pushI32(@bitCast(scn.bytes_per_iter));
        try executor.dispatchCanonBuiltin(
            rx.inst,
            .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
            rx.env,
            null,
            allocator,
        );
        _ = try rx.env.popI32();
    }
    return nowNs() - start;
}

fn runOneSample(allocator: std.mem.Allocator, scn: Scenario) !u64 {
    return switch (scn.kind) {
        .read => runReadSample(allocator, scn),
        .write => runWriteSample(allocator, scn),
        .rt => runRtSample(allocator, scn),
    };
}

// ── RSS sampling ──────────────────────────────────────────────────────

/// Returns the peak RSS observed by the process so far, in bytes.
/// Linux + macOS expose this via `getrusage(SELF).ru_maxrss`; Linux
/// reports KiB, macOS reports bytes; other targets return 0.
fn peakRssBytes() u64 {
    switch (builtin.os.tag) {
        .linux, .macos => {
            const ru = std.posix.getrusage(std.posix.rusage.SELF);
            const maxrss: u64 = @intCast(@max(ru.maxrss, 0));
            return if (builtin.os.tag == .macos) maxrss else maxrss * 1024;
        },
        else => return 0,
    }
}

// ── Sample aggregation ────────────────────────────────────────────────

const SampleStats = struct {
    samples: u32,
    min_ns: u64,
    median_ns: u64,
    p95_ns: u64,
    max_ns: u64,
    rss_peak_bytes: u64,
};

fn analyse(samples: []u64, rss: u64) SampleStats {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const n = samples.len;
    const median = samples[n / 2];
    // p95: lower index so we don't go out of bounds on small n.
    const p95_idx_f: f64 = @as(f64, @floatFromInt(n - 1)) * 0.95;
    const p95_idx: usize = @intFromFloat(@floor(p95_idx_f));
    return .{
        .samples = @intCast(n),
        .min_ns = samples[0],
        .median_ns = median,
        .p95_ns = samples[p95_idx],
        .max_ns = samples[n - 1],
        .rss_peak_bytes = rss,
    };
}

// ── Budget loading + regression detection ─────────────────────────────

const BudgetEntry = struct {
    median_ns_budget: u64,
    samples: u32 = 10,
};

fn loadBudget(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    out: *std.StringHashMap(BudgetEntry),
) !void {
    const dir = std.Io.Dir.cwd();
    const buf = dir.readFileAlloc(io, path, allocator, .unlimited) catch |err| {
        std.debug.print("warn: budget file '{s}' unreadable ({s}); regression check disabled.\n", .{ path, @errorName(err) });
        return;
    };
    defer allocator.free(buf);

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, buf, .{}) catch |err| {
        std.debug.print("warn: budget file '{s}' parse error ({s}); regression check disabled.\n", .{ path, @errorName(err) });
        return;
    };
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return;
    var it = root.object.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        const v = kv.value_ptr.*;
        if (v != .object) continue;
        const median_v = v.object.get("median_ns_budget") orelse continue;
        if (median_v != .integer) continue;
        const samples_v = v.object.get("samples");
        const samples_i: u32 = if (samples_v) |sv|
            (if (sv == .integer) @intCast(sv.integer) else 10)
        else
            10;
        try out.put(
            try allocator.dupe(u8, key),
            .{
                .median_ns_budget = @intCast(median_v.integer),
                .samples = samples_i,
            },
        );
    }
}

// ── Main ──────────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var samples: u32 = 10;
    var warmup: u32 = 2;
    var budget_path: ?[]const u8 = "tests/benchmarks/wasi-microbench/budget.json";
    var regression_threshold_pct: f64 = 10.0;
    var json_out_path: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--samples")) {
            i += 1;
            if (i >= args.len) return error.MissingArg;
            samples = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            i += 1;
            if (i >= args.len) return error.MissingArg;
            warmup = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--budget")) {
            i += 1;
            if (i >= args.len) return error.MissingArg;
            budget_path = args[i];
        } else if (std.mem.eql(u8, arg, "--no-budget")) {
            budget_path = null;
        } else if (std.mem.eql(u8, arg, "--threshold-pct")) {
            i += 1;
            if (i >= args.len) return error.MissingArg;
            regression_threshold_pct = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--json")) {
            i += 1;
            if (i >= args.len) return error.MissingArg;
            json_out_path = args[i];
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            std.debug.print(
                \\wasi-microbench — host-path regression detector (#583, W11-6).
                \\
                \\Usage:
                \\  wasi-microbench [--samples N] [--warmup N] [--budget PATH]
                \\                  [--no-budget] [--threshold-pct N] [--json PATH]
                \\
                \\Defaults: --samples 10 --warmup 2 --threshold-pct 10
                \\          --budget tests/benchmarks/wasi-microbench/budget.json
                \\
            , .{});
            return;
        } else {
            std.debug.print("warn: ignoring unknown arg '{s}'\n", .{arg});
        }
    }

    if (samples < 3) samples = 3; // need ≥3 for median + p95 to be meaningful

    var budgets = std.StringHashMap(BudgetEntry).init(a);
    defer {
        var bit = budgets.iterator();
        while (bit.next()) |kv| a.free(kv.key_ptr.*);
        budgets.deinit();
    }
    if (budget_path) |p| try loadBudget(a, io, p, &budgets);

    std.debug.print(
        "wasi-microbench: {d} scenarios × {d} samples (warmup {d}) — threshold {d:.1}%\n\n",
        .{ scenarios.len, samples, warmup, regression_threshold_pct },
    );

    var json_buf = std.Io.Writer.Allocating.init(a);
    defer json_buf.deinit();
    try json_buf.writer.writeAll("{\n  \"scenarios\": [\n");

    var any_regression = false;
    for (scenarios, 0..) |scn, scn_idx| {
        // Warmup (discarded).
        var w_i: u32 = 0;
        while (w_i < warmup) : (w_i += 1) {
            _ = try runOneSample(a, scn);
        }

        const buf = try a.alloc(u64, samples);
        defer a.free(buf);
        var s_i: u32 = 0;
        while (s_i < samples) : (s_i += 1) {
            buf[s_i] = try runOneSample(a, scn);
        }
        const rss = peakRssBytes();
        const stats = analyse(buf, rss);

        const ns_per_iter: f64 = @as(f64, @floatFromInt(stats.median_ns)) /
            @as(f64, @floatFromInt(scn.iters_per_sample));
        const total_bytes: u64 = @as(u64, scn.iters_per_sample) * @as(u64, scn.bytes_per_iter);
        const mib_per_s: f64 = @as(f64, @floatFromInt(total_bytes)) /
            (@as(f64, @floatFromInt(stats.median_ns)) / 1e9) / (1024.0 * 1024.0);

        const verdict: []const u8 = blk: {
            if (budgets.get(scn.name)) |budget| {
                const limit_ns: f64 = @as(f64, @floatFromInt(budget.median_ns_budget)) *
                    (1.0 + regression_threshold_pct / 100.0);
                if (@as(f64, @floatFromInt(stats.median_ns)) > limit_ns) {
                    any_regression = true;
                    break :blk "REGRESSED";
                }
                break :blk "ok";
            }
            break :blk "no-budget";
        };

        std.debug.print(
            "  [{s:^9}] {s:<32}  median={d:>10} ns  p95={d:>10} ns  " ++
                "min={d:>10} ns  iter={d:>8.1} ns  thr={d:>7.1} MiB/s  rss_peak={d:>6} MiB\n",
            .{
                verdict,
                scn.name,
                stats.median_ns,
                stats.p95_ns,
                stats.min_ns,
                ns_per_iter,
                mib_per_s,
                stats.rss_peak_bytes / (1024 * 1024),
            },
        );

        if (budgets.get(scn.name)) |budget| {
            const delta_pct = (@as(f64, @floatFromInt(stats.median_ns)) -
                @as(f64, @floatFromInt(budget.median_ns_budget))) /
                @as(f64, @floatFromInt(budget.median_ns_budget)) * 100.0;
            const sign: []const u8 = if (delta_pct >= 0) "+" else "";
            std.debug.print(
                "                budget median_ns={d}  Δ {s}{d:.1}%  (threshold +{d:.1}%)\n",
                .{ budget.median_ns_budget, sign, delta_pct, regression_threshold_pct },
            );
        }

        if (scn_idx > 0) try json_buf.writer.writeAll(",\n");
        try json_buf.writer.print(
            "    {{\"name\":\"{s}\",\"samples\":{d},\"iters_per_sample\":{d}," ++
                "\"bytes_per_iter\":{d},\"min_ns\":{d},\"median_ns\":{d}," ++
                "\"p95_ns\":{d},\"max_ns\":{d},\"rss_peak_bytes\":{d}," ++
                "\"verdict\":\"{s}\"}}",
            .{
                scn.name,
                stats.samples,
                scn.iters_per_sample,
                scn.bytes_per_iter,
                stats.min_ns,
                stats.median_ns,
                stats.p95_ns,
                stats.max_ns,
                stats.rss_peak_bytes,
                verdict,
            },
        );
    }
    try json_buf.writer.writeAll("\n  ]\n}\n");

    if (json_out_path) |p| {
        const dir = std.Io.Dir.cwd();
        try dir.writeFile(io, .{ .sub_path = p, .data = json_buf.written() });
        std.debug.print("\nWrote machine-readable summary to {s}\n", .{p});
    } else {
        std.debug.print("\n{s}", .{json_buf.written()});
    }

    if (any_regression) {
        std.debug.print(
            "\n❌ One or more scenarios exceeded budget median × {d:.2}.\n" ++
                "   Re-run on an idle host; if the regression is intentional, see\n" ++
                "   `docs/wasi.md` § \"Updating the wasi-microbench budget\".\n",
            .{1.0 + regression_threshold_pct / 100.0},
        );
        std.process.exit(1);
    }
    std.debug.print("\n✅ All scenarios within budget (or no budget provided).\n", .{});
}
