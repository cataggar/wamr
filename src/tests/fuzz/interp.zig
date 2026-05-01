//! fuzz-interp — load + instantiate + invoke every nullary export
//! through the interpreter.
//!
//! Oracle: any panic, safety-checked UB, or unhandled error is a bug.
//! Runtime traps surface as typed errors and are OK.
//!
//! This is a scaffold. Until a bounded execution helper exists, this
//! target deliberately stays on loader + validation coverage. See
//! tests/fuzz/README.md for follow-up scope.

const std = @import("std");
const wamr = @import("wamr");
const common = @import("common.zig");

pub fn main(init: std.process.Init) !void {
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

    std.log.info("fuzz-interp: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}

fn runOnce(allocator: std.mem.Allocator, bytes: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Just exercise loader + downstream validation for now. A full
    // invocation path requires bounded execution and host-import setup.
    _ = wamr.loader.load(bytes, a) catch return;
}
