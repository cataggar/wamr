//! CoreMark sampling-profile runner for the Zig AOT backend.
//!
//! Companion to `coremark_aot_runner.zig`. Wraps the same load → AOT
//! compile → instantiate → invoke `_start` pipeline with a SIGPROF-driven
//! sampling profiler, then prints a top-N hot-function table and the raw
//! disassembly of the top-3 hot functions.
//!
//! The harness deliberately injects nothing into the AOT-emitted code
//! itself: kernel-driven timer interrupts hit the running thread inside
//! the codegen, the signal handler peeks the interrupted PC out of
//! `ucontext_t.mcontext.{pc,rip}`, and that PC is bucketed against the
//! AOT instance's already-known `func_offsets` table. So the codegen we
//! are *trying* to measure is unperturbed.
//!
//! Linux + aarch64/x86_64 only (the SIGPROF backend is gated on those —
//! see `profile/sigprof.zig`). On any unsupported host the step prints a
//! clear "skipping" message and exits 0, matching the existing
//! `coremark_aot_runner.zig` skip behaviour.
//!
//! Usage:
//!     zig build coremark-profile
//!     # or, after `zig build`:
//!     ./zig-out/bin/coremark-profile-runner path/to/coremark_wasi_nofp.wasm

const std = @import("std");
const builtin = @import("builtin");
const aot_harness = @import("aot_harness.zig");
const sigprof = @import("profile/sigprof.zig");

const wamr = @import("wamr");
const aot_runtime = wamr.aot_runtime;
const name_section_mod = wamr.name_section;

const TopN: usize = 10;
const HotDisassemble: usize = 3;
const DefaultIntervalUs: u32 = 1000;

const Bucket = struct {
    /// Wasm-level funcidx (imports + locals). For synthetic buckets this
    /// is `std.math.maxInt(u32)`.
    func_idx: u32,
    samples: u64,
    /// Display name. For real wasm funcs this is the name-section entry
    /// or `func_<localidx>` fallback. For synthetic buckets it's e.g.
    /// `<helper>` or `<outside>`.
    name: []const u8,
    /// Local-func offset within the text section, or 0 for synthetic
    /// buckets. Used by the disassembler step.
    text_offset: u32 = 0,
    /// Length of the function's machine code, or 0 for synthetic buckets.
    code_len: u32 = 0,
    /// True for real wasm functions (eligible for disassembly).
    real: bool = false,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len < 2) {
        std.debug.print(
            \\usage: coremark-profile-runner <path-to-coremark.wasm>
            \\
            \\Runs CoreMark through the Zig AOT backend with an in-process
            \\SIGPROF sampling profiler, then prints a top-{d} hot-function
            \\table and the disassembly of the top-{d}.
            \\
        , .{ TopN, HotDisassemble });
        std.process.exit(2);
    }
    const wasm_path = args[1];

    if (!aot_harness.can_exec_aot) {
        std.debug.print(
            "coremark-profile-runner: AOT execution not supported on this target ({s}); skipping.\n",
            .{@tagName(builtin.cpu.arch)},
        );
        return;
    }
    if (!sigprof.supported) {
        std.debug.print(
            "coremark-profile-runner: SIGPROF profiling not supported on this OS/arch ({s}-{s}); skipping.\n",
            .{ @tagName(builtin.os.tag), @tagName(builtin.cpu.arch) },
        );
        return;
    }

    const wasm_bytes = std.Io.Dir.cwd().readFileAlloc(io, wasm_path, allocator, @enumFromInt(128 * 1024 * 1024)) catch |err| {
        std.debug.print("coremark-profile-runner: failed to read {s}: {s}\n", .{ wasm_path, @errorName(err) });
        std.process.exit(2);
    };
    defer allocator.free(wasm_bytes);

    std.debug.print("============> profile {s} (zig AOT)\n", .{wasm_path});

    const h = aot_harness.Harness.initWithOptions(
        allocator,
        wasm_bytes,
        null,
        .{ .invoke_start = true },
    ) catch |err| {
        std.debug.print("coremark-profile-runner: harness init failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer h.deinit();

    const start_idx = h.findFuncExport("_start") orelse {
        std.debug.print("coremark-profile-runner: no `_start` export found\n", .{});
        std.process.exit(1);
    };

    // Resolve function names once; profile aggregation is O(top_n) so
    // a linear scan is fine, but we cache the slice to keep the per-row
    // lookup constant.
    const name_entries = name_section_mod.parseFunctionNames(wasm_bytes, allocator) catch |err| switch (err) {
        // A malformed name section in a CoreMark binary would mean the
        // toolchain that produced it was broken. Treat it as "no names"
        // and continue rather than fail the profile run.
        else => blk: {
            std.debug.print(
                "coremark-profile-runner: name section parse failed ({s}); falling back to numeric names\n",
                .{@errorName(err)},
            );
            break :blk allocator.alloc(name_section_mod.FunctionName, 0) catch &.{};
        },
    };
    defer allocator.free(name_entries);

    // Arm SIGPROF, invoke `_start`, disarm.
    const interval_us = DefaultIntervalUs;
    sigprof.arm(interval_us) catch |err| {
        std.debug.print("coremark-profile-runner: arm failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };

    const wall_start = monotonicNs();

    var results: [1]aot_runtime.ScalarResult = undefined;
    const ft = h.getFuncType(start_idx) orelse {
        sigprof.disarm();
        std.debug.print("coremark-profile-runner: _start has no type info\n", .{});
        std.process.exit(1);
    };
    _ = aot_runtime.callFuncScalar(
        h.inst,
        start_idx,
        ft.params,
        &.{},
        &.{},
        &results,
    ) catch |err| {
        // Same rationale as coremark_aot_runner: `proc_exit` traps are
        // expected; surface anything else but let aggregation continue
        // — partial samples are still useful.
        std.debug.print("coremark-profile-runner: _start returned error: {s}\n", .{@errorName(err)});
    };

    const wall_end = monotonicNs();
    sigprof.disarm();

    const wall_ms: u64 = (wall_end -% wall_start) / std.time.ns_per_ms;
    const samples = sigprof.samples();
    const dropped = sigprof.droppedCount();

    std.debug.print(
        "[coremark-profile] total_samples={d} dropped={d} wall_ms={d} interval_us={d}\n",
        .{ samples.len, dropped, wall_ms, interval_us },
    );

    var buckets = aggregate(allocator, h.inst, samples, name_entries) catch |err| {
        std.debug.print("coremark-profile-runner: aggregate failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer buckets.deinit(allocator);

    sortBuckets(buckets.items);
    printTopTable(buckets.items, samples.len, interval_us);

    dumpHotFunctions(allocator, io, h.inst, buckets.items) catch |err| {
        std.debug.print("coremark-profile-runner: disassembly skipped: {s}\n", .{@errorName(err)});
    };

    std.debug.print("============> {s} profile complete\n", .{wasm_path});
}

// ─── Aggregation ───────────────────────────────────────────────────────

const Buckets = std.ArrayList(Bucket);

fn aggregate(
    gpa: std.mem.Allocator,
    inst: *const aot_runtime.AotInstance,
    samples: []const sigprof.Sample,
    name_entries: []const name_section_mod.FunctionName,
) !Buckets {
    const module = inst.module;
    const code_base_opt = inst.code_base;
    const code_size = inst.code_size;
    const code_base: usize = if (code_base_opt) |p| @intFromPtr(p) else 0;
    const import_count = module.import_function_count;
    const func_count = module.func_count;
    const func_offsets = module.func_offsets;

    // One bucket per local function plus three synthetic buckets:
    //   <outside>: PC outside [code_base, code_base+code_size)
    //   <helper>:  inside the runtime helper functions (memGrow, traps, …)
    //   <unmapped>: code_base unset (shouldn't happen for a runnable AOT)
    var list: Buckets = .empty;
    errdefer list.deinit(gpa);

    try list.ensureTotalCapacityPrecise(gpa, func_count + 3);

    // Real wasm funcs.
    var i: u32 = 0;
    while (i < func_count) : (i += 1) {
        const local_idx = i;
        const wasm_idx = import_count + local_idx;
        const off = func_offsets[local_idx];
        const next_off: u32 = if (local_idx + 1 < func_count)
            func_offsets[local_idx + 1]
        else
            @intCast(code_size);
        const len: u32 = if (next_off > off) next_off - off else 0;
        list.appendAssumeCapacity(.{
            .func_idx = wasm_idx,
            .samples = 0,
            .name = nameFor(name_entries, wasm_idx, local_idx),
            .text_offset = off,
            .code_len = len,
            .real = true,
        });
    }

    // Synthetic buckets: keep stable indices at the end.
    const outside_idx = list.items.len;
    list.appendAssumeCapacity(.{
        .func_idx = std.math.maxInt(u32),
        .samples = 0,
        .name = "<outside>",
    });
    const helper_idx = list.items.len;
    list.appendAssumeCapacity(.{
        .func_idx = std.math.maxInt(u32),
        .samples = 0,
        .name = "<helper>",
    });
    const unmapped_idx = list.items.len;
    list.appendAssumeCapacity(.{
        .func_idx = std.math.maxInt(u32),
        .samples = 0,
        .name = "<unmapped>",
    });

    for (samples) |pc_raw| {
        const pc: usize = @intCast(pc_raw);
        if (code_base == 0) {
            list.items[unmapped_idx].samples += 1;
            continue;
        }
        if (pc < code_base or pc >= code_base + code_size) {
            // Could be a runtime helper (e.g. memGrowHelper, aotTrap*) or
            // libc / kernel return paths. Use the funcptrs table to spot
            // helpers; otherwise it's <outside>.
            if (isHelperPc(inst, pc)) {
                list.items[helper_idx].samples += 1;
            } else {
                list.items[outside_idx].samples += 1;
            }
            continue;
        }
        const off: u32 = @intCast(pc - code_base);
        const local_idx = funcOffsetIndex(func_offsets, off);
        list.items[local_idx].samples += 1;
    }

    return list;
}

fn nameFor(
    entries: []const name_section_mod.FunctionName,
    wasm_idx: u32,
    local_idx: u32,
) []const u8 {
    if (name_section_mod.lookup(entries, wasm_idx)) |n| return n;
    // Fallback. Stored in a static buffer per-call site is awkward in
    // Zig; emit a stable placeholder that the formatter prints
    // alongside the numeric `func_idx` column.
    _ = local_idx;
    return "(no name section)";
}

fn funcOffsetIndex(func_offsets: []const u32, off: u32) usize {
    // Linear scan beats binary search for the typical CoreMark binary
    // (~100 funcs) on memory hierarchy effects, but use binary search to
    // stay scalable for larger workloads.
    if (func_offsets.len == 0) return 0;
    var lo: usize = 0;
    var hi: usize = func_offsets.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (func_offsets[mid] <= off) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

fn isHelperPc(inst: *const aot_runtime.AotInstance, pc: usize) bool {
    // Match the funcptrs table for imports — those are runtime helper
    // pointers (host_bridge AOT shims) installed at instantiation time.
    const import_count = inst.module.import_function_count;
    const total = @min(inst.funcptrs.len, import_count);
    var i: usize = 0;
    while (i < total) : (i += 1) {
        // Helpers are short (well under 4 KiB); a wide window is fine
        // and avoids needing per-helper sizes.
        const base = inst.funcptrs[i];
        if (base != 0 and pc >= base and pc < base +% 4096) return true;
    }
    return false;
}

fn sortBuckets(buckets: []Bucket) void {
    const Cmp = struct {
        fn lt(_: void, a: Bucket, b: Bucket) bool {
            if (a.samples != b.samples) return a.samples > b.samples;
            // Tie-break: put real functions before synthetic buckets,
            // then ascending wasm_idx for stable bisect output.
            if (a.real != b.real) return a.real and !b.real;
            return a.func_idx < b.func_idx;
        }
    };
    std.sort.pdq(Bucket, buckets, {}, Cmp.lt);
}

// ─── Output ────────────────────────────────────────────────────────────

fn printTopTable(buckets: []const Bucket, total: usize, interval_us: u32) void {
    std.debug.print(
        "\n[coremark-profile] top-{d} hot functions:\n",
        .{TopN},
    );
    std.debug.print("  {s:>5}  {s:>9}  {s:>10}  {s:>9}  {s}\n", .{ "idx", "samples", "self_ms", "self_pct", "name" });
    std.debug.print("  {s:>5}  {s:>9}  {s:>10}  {s:>9}  {s}\n", .{ "-----", "---------", "----------", "---------", "----" });

    const limit = @min(buckets.len, TopN);
    var i: usize = 0;
    while (i < limit) : (i += 1) {
        const b = buckets[i];
        if (b.samples == 0) break; // stop once we hit zero-sample rows
        const self_ms: f64 = @as(f64, @floatFromInt(b.samples)) *
            @as(f64, @floatFromInt(interval_us)) / 1000.0;
        const pct: f64 = if (total == 0) 0 else 100.0 *
            @as(f64, @floatFromInt(b.samples)) / @as(f64, @floatFromInt(total));
        if (b.real) {
            std.debug.print(
                "  {d:>5}  {d:>9}  {d:>10.1}  {d:>8.2}%  {s}\n",
                .{ b.func_idx, b.samples, self_ms, pct, b.name },
            );
        } else {
            std.debug.print(
                "  {s:>5}  {d:>9}  {d:>10.1}  {d:>8.2}%  {s}\n",
                .{ "-", b.samples, self_ms, pct, b.name },
            );
        }
    }
}

// ─── Disassembly ───────────────────────────────────────────────────────

fn dumpHotFunctions(
    gpa: std.mem.Allocator,
    io: std.Io,
    inst: *const aot_runtime.AotInstance,
    buckets: []const Bucket,
) !void {
    const code_base_opt = inst.code_base;
    if (code_base_opt == null) return;
    const code_base: usize = @intFromPtr(code_base_opt.?);

    var dumped: usize = 0;
    var i: usize = 0;
    while (i < buckets.len and dumped < HotDisassemble) : (i += 1) {
        const b = buckets[i];
        if (!b.real or b.samples == 0 or b.code_len == 0) continue;
        const fn_addr = code_base + b.text_offset;
        const fn_bytes = @as([*]const u8, @ptrFromInt(fn_addr))[0..b.code_len];
        try dumpOneFunction(gpa, io, b, fn_bytes);
        dumped += 1;
    }
}

fn dumpOneFunction(
    gpa: std.mem.Allocator,
    io: std.Io,
    b: Bucket,
    fn_bytes: []const u8,
) !void {
    std.debug.print(
        "\n[coremark-profile] disassembly: idx={d} ({s}) text_offset=0x{x} size={d}\n",
        .{ b.func_idx, b.name, b.text_offset, b.code_len },
    );

    // GNU `objdump -D -b binary -m <arch>` accepts a flat byte stream;
    // `llvm-objdump` lacks an equivalent flag. We pick the GNU machine
    // name based on the host arch (the runner runs the AOT natively).
    const machine = switch (builtin.cpu.arch) {
        .aarch64 => "aarch64",
        .x86_64 => "i386:x86-64",
        else => return,
    };

    // Write bytes to a temp file under /tmp (sufficient for the
    // self-hosted runner; we don't need TMPDIR overrides here).
    const tmp_path = try std.fmt.allocPrint(
        gpa,
        "/tmp/coremark-profile-fn-{d}.bin",
        .{b.func_idx},
    );
    defer gpa.free(tmp_path);

    std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = tmp_path,
        .data = fn_bytes,
        .flags = .{ .truncate = true },
    }) catch |err| {
        std.debug.print("  (could not write {s}: {s}; raw hex dump follows)\n", .{ tmp_path, @errorName(err) });
        hexDump(fn_bytes);
        return;
    };
    defer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};

    const result = std.process.run(gpa, io, .{
        .argv = &[_][]const u8{
            "objdump",
            "-D",
            "-b",
            "binary",
            "-m",
            machine,
            "--no-show-raw-insn",
            tmp_path,
        },
    }) catch |err| {
        std.debug.print(
            "  (objdump unavailable — {s}; raw hex dump follows)\n",
            .{@errorName(err)},
        );
        hexDump(fn_bytes);
        return;
    };
    defer gpa.free(result.stdout);
    defer gpa.free(result.stderr);

    // GNU objdump prints to stdout; surface stderr only on non-zero
    // exit, mirroring how the user would run it manually.
    const exited_ok = switch (result.term) {
        .exited => |c| c == 0,
        else => false,
    };
    if (!exited_ok and result.stderr.len > 0) {
        std.debug.print("  (objdump stderr: {s})\n", .{result.stderr});
    }
    std.debug.print("{s}", .{result.stdout});
}

fn hexDump(bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) : (off += 16) {
        const end = @min(off + 16, bytes.len);
        std.debug.print("    {x:0>4}: ", .{off});
        for (bytes[off..end]) |byte| {
            std.debug.print("{x:0>2} ", .{byte});
        }
        std.debug.print("\n", .{});
    }
}

fn monotonicNs() u64 {
    var ts: std.posix.timespec = .{ .sec = 0, .nsec = 0 };
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    const sec_u: u64 = @intCast(ts.sec);
    const nsec_u: u64 = @intCast(ts.nsec);
    return sec_u *% std.time.ns_per_s +% nsec_u;
}
