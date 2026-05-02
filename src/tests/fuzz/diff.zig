//! fuzz-diff — interp vs AOT differential oracle.
//!
//! For each valid import-free module, compare interpreter and AOT
//! results for a conservative subset of nullary scalar exports. Typed
//! load/compile/trap outcomes are expected; result mismatches are saved
//! as named crashers.

const std = @import("std");
const wamr = @import("wamr");
const common = @import("common.zig");
const invoke = @import("invoke.zig");
const aot_harness = @import("aot_harness");

pub fn main(init: std.process.Init) !void {
    if (comptime !aot_harness.can_exec_aot) {
        std.log.err("fuzz-diff requires x86_64 or aarch64 host", .{});
        return error.UnsupportedArch;
    }
    const allocator = init.gpa;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());

    const args = try common.Args.parse(argv);
    var corpus = try common.Corpus.load(allocator, io, args.corpus_dir);
    defer corpus.deinit();

    if (corpus.count() == 0) {
        std.log.err("empty corpus at {s}", .{args.corpus_dir});
        return error.EmptyCorpus;
    }

    const deadline = common.Deadline.init(io, args.duration_ms);
    var iter: u64 = 0;
    var idx: usize = 0;
    while (!deadline.expired(io)) : (iter += 1) {
        const input = corpus.get(idx);
        idx +%= 1;

        try common.markInFlight(io, args.crashes_dir, input);
        try runOnce(allocator, io, args.crashes_dir, input, args.fuel orelse invoke.default_fuel_per_export);
        common.clearInFlight(io, args.crashes_dir);
    }

    std.log.info("fuzz-diff: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}

fn runOnce(allocator: std.mem.Allocator, io: std.Io, crashes_dir: []const u8, bytes: []const u8, fuel_per_export: u32) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const module = wamr.loader.load(bytes, arena.allocator()) catch return;
    if (!invoke.modulePolicyAllows(&module)) return;

    var targets: [invoke.max_exports_per_input]invoke.ExportTarget = undefined;
    var target_count: usize = 0;
    for (module.exports) |exp| {
        const target = invoke.selectExport(&module, exp) orelse continue;
        if (!invoke.isAotSafeExport(&module, target.func_idx)) continue;
        targets[target_count] = target;
        target_count += 1;
        if (target_count >= targets.len) break;
    }
    if (target_count == 0) return;

    const interp_inst = wamr.instance.instantiate(&module, allocator) catch return;
    defer wamr.instance.destroy(interp_inst);

    // Full AOT compile + instantiate, but do NOT invoke start. Native AOT
    // invocation below is limited to the statically safe subset selected above.
    const h = aot_harness.Harness.initWithOptions(allocator, bytes, null, .{ .invoke_start = false }) catch return;
    defer h.deinit();

    for (targets[0..target_count]) |target| {
        var interp_buf: [invoke.max_results]invoke.ScalarValue = undefined;
        const interp_results = invoke.invokeInterpExport(
            allocator,
            interp_inst,
            target.func_idx,
            target.func_type,
            fuel_per_export,
            &interp_buf,
        ) catch |err| switch (err) {
            error.ExpectedTrap, error.OutOfFuel => continue,
            else => return err,
        };

        var aot_raw_buf: [wamr.aot_runtime.MaxScalarResults]wamr.aot_runtime.ScalarResult = undefined;
        const aot_raw = h.callScalar(target.func_idx, &.{}, &aot_raw_buf) catch continue;
        if (aot_raw.len > invoke.max_results) {
            try common.saveCrasher(io, crashes_dir, bytes, "diff-mismatch");
            return;
        }

        var aot_buf: [invoke.max_results]invoke.ScalarValue = undefined;
        for (aot_raw, 0..) |raw, i| {
            aot_buf[i] = invoke.scalarFromAot(raw) orelse {
                try common.saveCrasher(io, crashes_dir, bytes, "diff-mismatch");
                return;
            };
        }
        const aot_results = aot_buf[0..aot_raw.len];

        if (!invoke.scalarSlicesEqual(interp_results, aot_results)) {
            try common.saveCrasher(io, crashes_dir, bytes, "diff-mismatch");
            return;
        }
    }
}
