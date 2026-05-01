//! fuzz-component-loader — feed arbitrary bytes to the component loader.
//!
//! Oracle: the component loader must return either a valid `Component`
//! or a typed loader error. Any panic, safety-checked overflow, or OOM
//! on a small input is a bug and leaves the offending input at
//! `<crashes>/in-flight.wasm` when the process aborts.

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

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        if (wamr.component_loader.load(input, arena.allocator())) |_| {
            // Valid component — fine.
        } else |_| {
            // Typed loader error — fine.
        }

        common.clearInFlight(io, args.crashes_dir);
    }

    std.log.info("fuzz-component-loader: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}
