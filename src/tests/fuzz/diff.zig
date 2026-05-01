//! fuzz-diff — interp vs AOT differential oracle (SCAFFOLD).
//!
//! For each valid wasm input, drive BOTH the interpreter load path
//! and the full AOT compile+instantiate path. Crashes / panics in
//! either pipeline surface via the sentinel-file mechanism.
//!
//! A true result-comparison oracle requires a shared "invoke export
//! N, capture typed results" helper that neither interp nor AOT
//! currently expose uniformly. For v1 the harness only exercises
//! the compile paths. See tests/fuzz/README.md for the v2 scope.

const std = @import("std");
const wamr = @import("wamr");
const common = @import("common.zig");
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
        runOnce(allocator, input) catch {};
        common.clearInFlight(io, args.crashes_dir);
    }

    std.log.info("fuzz-diff: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}

fn runOnce(allocator: std.mem.Allocator, bytes: []const u8) !void {
    // Interp side — loader only for v1.
    var iarena = std.heap.ArenaAllocator.init(allocator);
    defer iarena.deinit();
    _ = wamr.loader.load(bytes, iarena.allocator()) catch return;

    // AOT side — full compile + instantiate, but do NOT invoke start:
    // attacker-supplied start functions can SEGV on OOB access.
    const h = aot_harness.Harness.initWithOptions(allocator, bytes, null, .{ .invoke_start = false }) catch return;
    defer h.deinit();
}
