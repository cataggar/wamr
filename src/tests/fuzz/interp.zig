//! fuzz-interp — load, instantiate, and invoke safe nullary exports
//! through the interpreter under a fuel budget.
//!
//! Oracle: loader/instantiation errors, guest traps, fuel exhaustion,
//! and unsupported signatures are expected outcomes. Any panic,
//! safety-checked UB, or process abort is a bug.

const std = @import("std");
const common = @import("common.zig");
const invoke = @import("invoke.zig");

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
        try invoke.runInterpModule(allocator, input, args.fuel orelse invoke.default_fuel_per_export);
        common.clearInFlight(io, args.crashes_dir);
    }

    std.log.info("fuzz-interp: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}
