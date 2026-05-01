//! fuzz-interp — load, instantiate, and invoke safe nullary exports
//! through the interpreter under a fuel budget.
//!
//! Oracle: loader/instantiation errors, guest traps, fuel exhaustion,
//! and unsupported signatures are expected outcomes. Any panic,
//! safety-checked UB, or process abort is a bug.

const std = @import("std");
const wamr = @import("wamr");
const common = @import("common.zig");
const types = wamr.types;
const ExecEnv = wamr.exec_env.ExecEnv;

const default_fuel_per_export: u32 = 100_000;
const exec_stack_size: u32 = 4096;
const max_exports_per_input: usize = 8;

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
        try runOnce(allocator, input, args.fuel orelse default_fuel_per_export);
        common.clearInFlight(io, args.crashes_dir);
    }

    std.log.info("fuzz-interp: {d} iterations over {d} inputs", .{ iter, corpus.count() });
}

fn runOnce(allocator: std.mem.Allocator, bytes: []const u8, fuel_per_export: u32) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const module = wamr.loader.load(bytes, a) catch return;
    if (module.imports.len != 0 or module.start_function != null) return;

    const inst = wamr.instance.instantiate(&module, allocator) catch return;
    defer wamr.instance.destroy(inst);

    var invoked: usize = 0;
    for (module.exports) |exp| {
        if (invoked >= max_exports_per_input) break;
        if (exp.kind != .function) continue;
        if (exp.index < module.import_function_count) continue;
        const func_type = module.getFuncType(exp.index) orelse continue;
        if (!supportedFuncType(func_type)) continue;

        try invokeExport(allocator, inst, exp.index, func_type, fuel_per_export);
        invoked += 1;
    }
}

fn supportedFuncType(func_type: types.FuncType) bool {
    if (func_type.params.len != 0) return false;
    if (func_type.results.len > 1) return false;
    for (func_type.results) |result| {
        switch (result) {
            .i32, .i64, .f32, .f64 => {},
            else => return false,
        }
    }
    return true;
}

fn invokeExport(
    allocator: std.mem.Allocator,
    inst: *types.ModuleInstance,
    func_idx: u32,
    func_type: types.FuncType,
    fuel_per_export: u32,
) !void {
    var env = try ExecEnv.create(inst, exec_stack_size, allocator);
    defer env.destroy();

    wamr.interp.executeFunctionWithOptions(env, func_idx, .{ .fuel = fuel_per_export }) catch return;
    try validateResults(env, func_type.results);
}

fn validateResults(env: *ExecEnv, results: []const types.ValType) !void {
    if (@as(usize, @intCast(env.sp)) != results.len) return error.InvalidResultStack;

    var i = results.len;
    while (i > 0) {
        i -= 1;
        const value = try env.pop();
        if (std.meta.activeTag(value) != results[i]) return error.InvalidResultStack;
    }

    if (env.sp != 0) return error.InvalidResultStack;
}
