//! fuzz-aot — AOT compile (+ instantiate + invoke) every input.
//!
//! Oracle: any panic, safety-checked UB in the compiler, codegen
//! writing outside its buffer, or generated code SEGVing is a bug.
//! Catches the `alloc_regs` scratch-corruption class of bugs where
//! a setcc / byte-op silently clobbers a live vreg.
//!
//! Scaffold: for v1 we only drive the compile path through the
//! harness. Full invoke-every-export comparison lives in fuzz-diff.

const std = @import("std");
const wamr = @import("wamr");
const common = @import("common.zig");
const aot_harness = @import("aot_harness");

pub fn main(init: std.process.Init) !void {
    if (comptime !aot_harness.can_exec_aot) {
        std.log.err("fuzz-aot requires x86_64 or aarch64 host", .{});
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

    std.log.info("fuzz-aot: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}

fn runOnce(allocator: std.mem.Allocator, bytes: []const u8) !void {
    // Fuzz targets must NOT execute attacker-supplied start functions —
    // arbitrary wasm bytecode can SEGV on OOB memory/table access, and
    // the runtime does not sandbox those faults. Compile + instantiate
    // only.
    const h = aot_harness.Harness.initWithOptions(allocator, bytes, null, .{ .invoke_start = false }) catch return;
    defer h.deinit();
}
