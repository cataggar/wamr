//! fuzz-loader — feed arbitrary bytes to the wasm loader.
//!
//! Oracle: the loader must return either a valid `WasmModule` or a
//! typed error. Any panic, integer overflow caught by ReleaseSafe,
//! or OOM on a small input is a bug and will leave the offending
//! input at `<crashes>/in-flight.wasm` when the process aborts.

const std = @import("std");
const wamr = @import("wamr");
const common = @import("common.zig");

/// Run the loader once over `input`. Shared between the CLI corpus
/// replay below and the OSS-Fuzz `LLVMFuzzerTestOneInput` shim in
/// `oss_loader.zig`. Typed loader errors are an expected outcome.
pub fn runOnce(allocator: std.mem.Allocator, input: []const u8) void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    if (wamr.loader.load(input, arena.allocator())) |_| {
        // Valid module — fine.
    } else |_| {
        // Typed error — fine.
    }
}

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
        runOnce(allocator, input);
        common.clearInFlight(io, args.crashes_dir);
    }

    std.log.info("fuzz-loader: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}
