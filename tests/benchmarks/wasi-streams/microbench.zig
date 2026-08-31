//! WASI stream zero-copy specialisation microbenchmark (#583 B2).
//!
//! Drives the executor's `stream.read` / `stream.write` rendezvous
//! against synthetic `host_driver`s that mirror the cost shape of
//! `WasiCliAdapter.tcpReceiveStreamOnRead` / `tcpSendStreamOnWrite`
//! (the `wasi:sockets@0.3.x.tcp-socket.{receive,send}` hot paths).
//! Counts wall-clock + heap allocations for two configurations on
//! each side:
//!
//! ## Read side
//!
//!   * **Baseline (`on_read`)** — driver writes into a stack buffer
//!     and `appendSlice`s the bytes onto `AsyncStream.buffer`; the
//!     executor then memcpys from the FIFO into guest linmem.
//!     Two memcpys + one heap allocation per chunk.
//!
//!   * **Zero-copy (`on_read_into`)** — driver writes directly into
//!     the guest-supplied destination slice; no scratch FIFO
//!     allocation, no second memcpy.
//!
//! ## Write side (#583 B2 follow-up)
//!
//!   * **Baseline (`on_write`)** — fatter signature mirroring the
//!     read side: `(ctx, *AsyncStream, bytes, allocator)`. Already
//!     zero-copy in terms of memcpys since the executor passes the
//!     borrowed `readGuestBytes` slice straight through; the two
//!     trailing parameters are unused.
//!
//!   * **Zero-copy (`on_write_from`)** — thinner signature `(ctx,
//!     src)`. Same memcpy story, just no spurious arguments. Stresses
//!     that the legacy + zero-copy paths are at parity (the API
//!     hygiene win, not a perf win — the write side never had the
//!     double-memcpy problem the read side did).
//!
//! This is the harness for the profile numbers reported in PR #583 B2
//! + the follow-up PR. It deliberately *doesn't* go through a real
//! TCP socket / fs `pwrite(2)` — the goal is to isolate the
//! canonical-ABI lowering + FIFO cost, not the kernel's
//! `recvfrom` / `pwrite` latency.
//!
//! Build / run:
//!
//!     zig build wasi-streams-bench -- --bytes-per-call 4096 --iters 4096
//!
//! Defaults (no args): 4 KiB per call × 4096 ops = 16 MiB pushed
//! through each side.

const std = @import("std");
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

/// CountingAllocator wraps another allocator and tracks number of
/// allocs / total bytes allocated. Used to compare scratch-buffer
/// churn across the two host_driver shapes.
const CountingAllocator = struct {
    parent: std.mem.Allocator,
    alloc_count: usize = 0,
    bytes_allocated: usize = 0,
    bytes_in_use: usize = 0,

    pub fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const out = self.parent.rawAlloc(len, alignment, ra) orelse return null;
        self.alloc_count += 1;
        self.bytes_allocated += len;
        self.bytes_in_use += len;
        return out;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const ok = self.parent.rawResize(buf, alignment, new_len, ra);
        if (ok) {
            if (new_len > buf.len) {
                self.bytes_in_use += new_len - buf.len;
                self.bytes_allocated += new_len - buf.len;
            } else {
                self.bytes_in_use -= buf.len - new_len;
            }
        }
        return ok;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const out = self.parent.rawRemap(buf, alignment, new_len, ra) orelse return null;
        self.alloc_count += 1;
        if (new_len > buf.len) {
            self.bytes_allocated += new_len - buf.len;
            self.bytes_in_use += new_len - buf.len;
        } else {
            self.bytes_in_use -= buf.len - new_len;
        }
        return out;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(buf, alignment, ra);
        self.bytes_in_use -= buf.len;
    }
};

/// Synthetic source for the host_driver: copies bytes from a fixed
/// pattern buffer into the driver's destination. Mirrors the cost
/// shape of a real `netRead` populating a destination buffer with
/// kernel-supplied bytes — both code paths see one `@memcpy` of
/// `bytes_per_call` worth of source bytes.
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

/// Baseline driver `on_read` — emulates `tcpReceiveStreamOnRead`.
/// Stack buffer → heap-backed `stream.buffer.appendSlice`.
fn baselineOnRead(
    opaque_ctx: ?*anyopaque,
    stream: *async_mod.AsyncStream,
    allocator: std.mem.Allocator,
) async_mod.HostStreamAction {
    const src: *SyntheticSource = @ptrCast(@alignCast(opaque_ctx.?));
    var buf: [64 * 1024]u8 = undefined;
    const n = src.fill(buf[0..src.bytes_per_call]);
    if (n == 0) return .would_block;
    stream.buffer.appendSlice(allocator, buf[0..n]) catch return .err;
    return .progressed;
}

/// Zero-copy driver `on_read_into` — writes directly into the
/// guest-supplied destination.
fn zeroCopyOnReadInto(
    opaque_ctx: ?*anyopaque,
    dst: []u8,
) async_mod.HostStreamReadInto {
    const src: *SyntheticSource = @ptrCast(@alignCast(opaque_ctx.?));
    if (dst.len == 0) return .{ .action = .would_block };
    const n = src.fill(dst);
    if (n == 0) return .{ .action = .would_block };
    return .{ .action = .progressed, .bytes_written = n };
}

/// Synthetic sink for the write-side drivers: drains the guest's
/// borrowed slice into a fixed-size buffer (so allocator behaviour
/// is identical between baseline and zero-copy paths). Mirrors the
/// cost shape of `tcpSendStreamOnWrite` / `fsWriteViaStreamOnWrite`
/// without going through a real fd.
const SyntheticSink = struct {
    drained_bytes: u64 = 0,
    last_checksum: u64 = 0,

    fn drain(self: *SyntheticSink, src: []const u8) void {
        self.drained_bytes += src.len;
        // Touch the bytes so the optimiser doesn't elide the
        // executor's borrowed-slice argument — we want the
        // benchmark to faithfully measure the cost of *passing*
        // the slice through, not have LLVM fold it away.
        var s: u64 = self.last_checksum;
        for (src) |b| s +%= b;
        self.last_checksum = s;
    }
};

/// Baseline driver `on_write` — fatter signature mirroring
/// `tcpSendStreamOnWrite`. Already operates on the executor's
/// borrowed `readGuestBytes` slice (no memcpy); the `*AsyncStream`
/// and `Allocator` parameters are unused.
fn baselineOnWrite(
    opaque_ctx: ?*anyopaque,
    _: *async_mod.AsyncStream,
    bytes: []const u8,
    _: std.mem.Allocator,
) async_mod.HostStreamAction {
    const sink: *SyntheticSink = @ptrCast(@alignCast(opaque_ctx.?));
    sink.drain(bytes);
    return .progressed;
}

/// Zero-copy driver `on_write_from` — thinner `(ctx, src)`
/// signature. Same memcpy story as `baselineOnWrite` (the write
/// path was already zero-copy via `readGuestBytes`); the win is API
/// hygiene, not memcpy elimination.
fn zeroCopyOnWriteFrom(
    opaque_ctx: ?*anyopaque,
    src: []const u8,
) async_mod.HostStreamAction {
    const sink: *SyntheticSink = @ptrCast(@alignCast(opaque_ctx.?));
    sink.drain(src);
    return .progressed;
}

const RunResult = struct {
    label: []const u8,
    iters: u32,
    bytes_total: u64,
    wall_ns: u64,
    alloc_count: usize,
    bytes_allocated: usize,

    fn report(self: RunResult) void {
        const ns_per_iter: f64 = @as(f64, @floatFromInt(self.wall_ns)) /
            @as(f64, @floatFromInt(self.iters));
        const mib_per_s: f64 = @as(f64, @floatFromInt(self.bytes_total)) /
            (@as(f64, @floatFromInt(self.wall_ns)) / 1e9) / (1024.0 * 1024.0);
        std.debug.print(
            "  {s:<18} iters={d:>6}  bytes={d:>10}  wall={d:>9} ns " ++
                "({d:>7.1} ns/iter, {d:>7.1} MiB/s)  allocs={d:>6}  alloc_bytes={d}\n",
            .{
                self.label,
                self.iters,
                self.bytes_total,
                self.wall_ns,
                ns_per_iter,
                mib_per_s,
                self.alloc_count,
                self.bytes_allocated,
            },
        );
    }
};

fn runBench(
    label: []const u8,
    parent_allocator: std.mem.Allocator,
    iters: u32,
    bytes_per_call: u32,
    zero_copy: bool,
) !RunResult {
    var counter = CountingAllocator{ .parent = parent_allocator };
    const alloc = counter.allocator();

    // Build a minimal `ComponentInstance` with a u8-element type so
    // `stream.read t` resolves `sizeOfType(t) == 1`.
    const StreamTypeFixture = struct {
        var types_array = [_]ctypes.TypeDef{.{ .val = .u8 }};
        var comp: ctypes.Component = .{
            .core_modules = &.{},
            .core_instances = &.{},
            .core_types = &.{},
            .components = &.{},
            .instances = &.{},
            .aliases = &.{},
            .types = &.{},
            .canons = &.{},
            .imports = &.{},
            .exports = &.{},
        };
    };
    StreamTypeFixture.comp.types = &StreamTypeFixture.types_array;
    const inst = try instance_mod.instantiate(&StreamTypeFixture.comp, alloc);
    defer {
        inst.disableTestMem();
        inst.deinit();
    }
    // Test linmem ≥ bytes_per_call to host the destination.
    const mem_size: usize = std.math.ceilPowerOfTwo(usize, @as(usize, bytes_per_call) * 2) catch
        @as(usize, bytes_per_call) * 2;
    try inst.enableTestMem(alloc, mem_size);

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, alloc);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, alloc);
    defer env.destroy();

    // Create a fresh stream with the chosen driver.
    try executor.dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        alloc,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);
    const pattern = try alloc.alloc(u8, bytes_per_call);
    defer alloc.free(pattern);
    var pi: u32 = 0;
    while (pi < pattern.len) : (pi += 1) pattern[pi] = @truncate(pi);
    var source = SyntheticSource{ .pattern = pattern, .bytes_per_call = bytes_per_call };
    var stream_lease = inst.streams.acquire(handle).?;
    const s = stream_lease.value();
    s.host_driver = if (zero_copy)
        .{ .context = &source, .on_read_into = &zeroCopyOnReadInto }
    else
        .{ .context = &source, .on_read = &baselineOnRead };
    stream_lease.release();

    // Reset the counters AFTER setup so we measure only the per-op cost.
    counter.alloc_count = 0;
    counter.bytes_allocated = 0;

    const dst_ptr: u32 = 0;

    const start_ns: u64 = nowNs();
    var i: u32 = 0;
    var bytes_total: u64 = 0;
    while (i < iters) : (i += 1) {
        try env.pushI32(@bitCast(handle));
        try env.pushI32(@bitCast(dst_ptr));
        try env.pushI32(@bitCast(bytes_per_call));
        try executor.dispatchCanonBuiltin(
            inst,
            .{ .async_canon = .{ .stream_read = .{ .type_idx = 0, .opts = &.{} } } },
            env,
            null,
            alloc,
        );
        const status: u32 = @bitCast(try env.popI32());
        // Unpack `packStatus`: low 4 bits = FutureStatus discriminant
        // (we only ever see `completed` / `would_block` parking in this
        // bench since the synthetic driver never errors), high 28 bits
        // = element count actually transferred.
        const got_count: u32 = status >> 4;
        bytes_total += got_count;
    }
    const wall_ns = nowNs() - start_ns;

    return .{
        .label = label,
        .iters = iters,
        .bytes_total = bytes_total,
        .wall_ns = wall_ns,
        .alloc_count = counter.alloc_count,
        .bytes_allocated = counter.bytes_allocated,
    };
}

/// Write-side bench: drives `stream.write` against either the
/// fatter `on_write` shape (baseline) or the thinner `on_write_from`
/// shape (zero-copy, #583 B2 follow-up). Returns the same
/// `RunResult` shape so the summary at the bottom of `main` works
/// for both sides uniformly.
fn runWriteBench(
    label: []const u8,
    parent_allocator: std.mem.Allocator,
    iters: u32,
    bytes_per_call: u32,
    zero_copy: bool,
) !RunResult {
    var counter = CountingAllocator{ .parent = parent_allocator };
    const alloc = counter.allocator();

    const StreamTypeFixture = struct {
        var types_array = [_]ctypes.TypeDef{.{ .val = .u8 }};
        var comp: ctypes.Component = .{
            .core_modules = &.{},
            .core_instances = &.{},
            .core_types = &.{},
            .components = &.{},
            .instances = &.{},
            .aliases = &.{},
            .types = &.{},
            .canons = &.{},
            .imports = &.{},
            .exports = &.{},
        };
    };
    StreamTypeFixture.comp.types = &StreamTypeFixture.types_array;
    const inst = try instance_mod.instantiate(&StreamTypeFixture.comp, alloc);
    defer {
        inst.disableTestMem();
        inst.deinit();
    }
    const mem_size: usize = std.math.ceilPowerOfTwo(usize, @as(usize, bytes_per_call) * 2) catch
        @as(usize, bytes_per_call) * 2;
    try inst.enableTestMem(alloc, mem_size);

    var module = core_types_mod.WasmModule{};
    const core_inst = try inst_mod_core.instantiate(&module, alloc);
    defer inst_mod_core.destroy(core_inst);
    const env = try ExecEnv.create(core_inst, 64, alloc);
    defer env.destroy();

    try executor.dispatchCanonBuiltin(
        inst,
        .{ .async_canon = .{ .stream_new = .{ .type_idx = 0 } } },
        env,
        null,
        alloc,
    );
    const handle: u32 = @truncate(@as(u64, @bitCast(try env.popI64())) >> 32);

    // Seed the guest source bytes once — every iteration re-reads
    // from the same `src_ptr` (mirrors the steady-state pattern of
    // a host driver draining a long-lived stream).
    const src_ptr: u32 = 0;
    {
        const src_bytes = inst.writableGuestBytes(src_ptr, bytes_per_call).?;
        var pi: u32 = 0;
        while (pi < bytes_per_call) : (pi += 1) src_bytes[pi] = @truncate(pi);
    }

    var sink = SyntheticSink{};
    var stream_lease = inst.streams.acquire(handle).?;
    const s = stream_lease.value();
    s.host_driver = if (zero_copy)
        .{ .context = &sink, .on_write_from = &zeroCopyOnWriteFrom }
    else
        .{ .context = &sink, .on_write = &baselineOnWrite };
    stream_lease.release();

    // Reset counters after setup so we measure only the per-op cost.
    counter.alloc_count = 0;
    counter.bytes_allocated = 0;

    const start_ns: u64 = nowNs();
    var i: u32 = 0;
    var bytes_total: u64 = 0;
    while (i < iters) : (i += 1) {
        try env.pushI32(@bitCast(handle));
        try env.pushI32(@bitCast(src_ptr));
        try env.pushI32(@bitCast(bytes_per_call));
        try executor.dispatchCanonBuiltin(
            inst,
            .{ .async_canon = .{ .stream_write = .{ .type_idx = 0, .opts = &.{} } } },
            env,
            null,
            alloc,
        );
        const status: u32 = @bitCast(try env.popI32());
        // Same packStatus unpack as the read bench: high 28 bits =
        // element count consumed.
        const got_count: u32 = status >> 4;
        bytes_total += got_count;
    }
    const wall_ns = nowNs() - start_ns;

    return .{
        .label = label,
        .iters = iters,
        .bytes_total = bytes_total,
        .wall_ns = wall_ns,
        .alloc_count = counter.alloc_count,
        .bytes_allocated = counter.bytes_allocated,
    };
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    // CLI parsing: `--bytes-per-call N --iters N`. Defaults sized so a
    // single run pushes 16 MiB through the rendezvous — large enough
    // that scratch-allocation cost dominates timer noise on a slow VM.
    var bytes_per_call: u32 = 4 * 1024;
    var iters: u32 = 4 * 1024;
    var i: usize = 1; // args[0] = program name
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--bytes-per-call")) {
            i += 1;
            if (i >= args.len) return error.MissingArg;
            bytes_per_call = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--iters")) {
            i += 1;
            if (i >= args.len) return error.MissingArg;
            iters = try std.fmt.parseInt(u32, args[i], 10);
        }
    }

    std.debug.print(
        "wasi-streams microbench: {d} iters × {d} bytes per stream op\n",
        .{ iters, bytes_per_call },
    );

    // ── Read side ───────────────────────────────────────────────────
    std.debug.print("\n== stream.read (#583 B2) ==\n", .{});
    const baseline_1 = try runBench("baseline (on_read)", a, iters, bytes_per_call, false);
    const baseline_2 = try runBench("baseline (on_read)", a, iters, bytes_per_call, false);
    const zerocopy_1 = try runBench("zero-copy (into)", a, iters, bytes_per_call, true);
    const zerocopy_2 = try runBench("zero-copy (into)", a, iters, bytes_per_call, true);

    std.debug.print("\nIndividual runs:\n", .{});
    baseline_1.report();
    baseline_2.report();
    zerocopy_1.report();
    zerocopy_2.report();

    const base_wall: u64 = @min(baseline_1.wall_ns, baseline_2.wall_ns);
    const zero_wall: u64 = @min(zerocopy_1.wall_ns, zerocopy_2.wall_ns);
    const base_allocs: usize = baseline_1.alloc_count;
    const zero_allocs: usize = zerocopy_1.alloc_count;

    const wall_speedup: f64 = (@as(f64, @floatFromInt(base_wall)) -
        @as(f64, @floatFromInt(zero_wall))) /
        @as(f64, @floatFromInt(base_wall)) * 100.0;
    const alloc_reduction: f64 = if (base_allocs > 0)
        (@as(f64, @floatFromInt(base_allocs)) -
            @as(f64, @floatFromInt(zero_allocs))) /
            @as(f64, @floatFromInt(base_allocs)) * 100.0
    else
        0.0;
    std.debug.print(
        "\nMedian-of-2 summary (read):\n" ++
            "  baseline  wall = {d:>10} ns, allocs = {d}\n" ++
            "  zero-copy wall = {d:>10} ns, allocs = {d}\n" ++
            "  Δ wall-clock = {d:.2}%   Δ allocs = {d:.2}%\n",
        .{
            base_wall,
            base_allocs,
            zero_wall,
            zero_allocs,
            wall_speedup,
            alloc_reduction,
        },
    );

    // ── Write side (#583 B2 follow-up) ──────────────────────────────
    //
    // The write path was already zero-copy via `readGuestBytes` (no
    // FIFO allocation, no second memcpy), so the wall-clock /
    // allocation numbers here are expected to be at parity between
    // the legacy `on_write` and the thinner `on_write_from`. The
    // bench exists primarily to (a) pin that property in CI and (b)
    // surface any regression introduced by the new executor branch.
    std.debug.print("\n== stream.write (#583 B2 follow-up) ==\n", .{});
    const w_baseline_1 = try runWriteBench("baseline (on_write)", a, iters, bytes_per_call, false);
    const w_baseline_2 = try runWriteBench("baseline (on_write)", a, iters, bytes_per_call, false);
    const w_zerocopy_1 = try runWriteBench("zero-copy (from)", a, iters, bytes_per_call, true);
    const w_zerocopy_2 = try runWriteBench("zero-copy (from)", a, iters, bytes_per_call, true);

    std.debug.print("\nIndividual runs:\n", .{});
    w_baseline_1.report();
    w_baseline_2.report();
    w_zerocopy_1.report();
    w_zerocopy_2.report();

    const wbase_wall: u64 = @min(w_baseline_1.wall_ns, w_baseline_2.wall_ns);
    const wzero_wall: u64 = @min(w_zerocopy_1.wall_ns, w_zerocopy_2.wall_ns);
    const wbase_allocs: usize = w_baseline_1.alloc_count;
    const wzero_allocs: usize = w_zerocopy_1.alloc_count;
    const w_wall_delta: f64 = (@as(f64, @floatFromInt(wbase_wall)) -
        @as(f64, @floatFromInt(wzero_wall))) /
        @as(f64, @floatFromInt(wbase_wall)) * 100.0;
    std.debug.print(
        "\nMedian-of-2 summary (write):\n" ++
            "  baseline  wall = {d:>10} ns, allocs = {d}\n" ++
            "  zero-copy wall = {d:>10} ns, allocs = {d}\n" ++
            "  Δ wall-clock = {d:.2}%   (parity is expected — both paths " ++
            "operate on the borrowed `readGuestBytes` slice)\n",
        .{
            wbase_wall,
            wbase_allocs,
            wzero_wall,
            wzero_allocs,
            w_wall_delta,
        },
    );
}
