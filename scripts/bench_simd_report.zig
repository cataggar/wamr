//! Report rendering for SIMD benchmark measurements.
const std = @import("std");
const parser = @import("bench_simd_parse.zig");
const optimize = @import("benchmark_optimize.zig");
const integer_median = @import("integer_median.zig");
const fixed = @import("decimal_format.zig").fixed;
const Allocator = std.mem.Allocator;
const Measurement = parser.Measurement;

const Key = struct { case: []const u8, engine: []const u8 };
const Number = union(enum) { integer: i128, float: f64 };
const Summary = struct {
    status: []const u8,
    result: ?i128,
    run_ns: ?Number,
    compile_ns: ?Number,
    iterations: i128,
    code_size: ?i128,
};

fn same(key: Key, row: Measurement) bool {
    return std.mem.eql(u8, key.case, row.case) and std.mem.eql(u8, key.engine, row.engine);
}

fn lessKey(_: void, a: Key, b: Key) bool {
    const order = std.mem.order(u8, a.case, b.case);
    return if (order == .eq) std.mem.lessThan(u8, a.engine, b.engine) else order == .lt;
}

fn keys(allocator: Allocator, first: []const Measurement, second: []const Measurement) ![]const Key {
    var result: std.ArrayList(Key) = .empty;
    for ([_][]const Measurement{ first, second }) |rows| {
        for (rows) |row| {
            const key: Key = .{ .case = row.case, .engine = row.engine };
            var found = false;
            for (result.items) |existing| {
                if (std.mem.eql(u8, existing.case, key.case) and std.mem.eql(u8, existing.engine, key.engine)) {
                    found = true;
                    break;
                }
            }
            if (!found) try result.append(allocator, key);
        }
    }
    std.mem.sort(Key, result.items, {}, lessKey);
    return result.toOwnedSlice(allocator);
}

fn median(values: []i128) !?Number {
    if (values.len == 0) return null;
    std.mem.sort(i128, values, {}, std.sort.asc(i128));
    const middle = values.len / 2;
    if (values.len % 2 != 0) return .{ .integer = values[middle] };
    return .{ .float = integer_median.average(values[middle - 1], values[middle]) };
}

fn summarize(allocator: Allocator, rows: []const Measurement, key: Key) !?Summary {
    var has_ok = false;
    var present = false;
    for (rows) |row| if (same(key, row)) {
        present = true;
        if (std.mem.eql(u8, row.status, "ok")) has_ok = true;
    };
    if (!present) return null;
    var times: std.ArrayList(i128) = .empty;
    var compile: std.ArrayList(i128) = .empty;
    var first: ?Measurement = null;
    var result: ?i128 = null;
    var size: ?i128 = null;
    for (rows) |row| {
        if (!same(key, row) or (has_ok and !std.mem.eql(u8, row.status, "ok"))) continue;
        if (first == null) first = row;
        if (result == null) result = row.result;
        if (size == null) size = row.code_size;
        if (row.run_ns) |value| try times.append(allocator, value);
        if (row.compile_ns) |value| try compile.append(allocator, value);
    }
    return .{
        .status = if (has_ok) "ok" else first.?.status,
        .result = result,
        .run_ns = try median(times.items),
        .compile_ns = try median(compile.items),
        .iterations = first.?.iterations,
        .code_size = size,
    };
}

fn numberFloat(number: Number) f64 {
    return switch (number) {
        .integer => |value| @floatFromInt(value),
        .float => |value| value,
    };
}

fn renderFixed(allocator: Allocator, value: f64, comptime precision: u8) ![]const u8 {
    const magnitude = try fixed(allocator, @abs(value), precision);
    const negative = @as(u64, @bitCast(value)) >> 63 != 0;
    return if (negative) try std.fmt.allocPrint(allocator, "-{s}", .{magnitude}) else magnitude;
}

fn renderNs(allocator: Allocator, value: ?Number) ![]const u8 {
    const number = numberFloat(value orelse return "-");
    if (number >= 1_000_000) return std.fmt.allocPrint(allocator, "{s} ms", .{try renderFixed(allocator, number / 1_000_000, 3)});
    if (number >= 1_000) return std.fmt.allocPrint(allocator, "{s} us", .{try renderFixed(allocator, number / 1_000, 3)});
    return std.fmt.allocPrint(allocator, "{s} ns", .{try renderFixed(allocator, number, 0)});
}

fn integerText(allocator: Allocator, value: ?i128) ![]const u8 {
    return if (value) |number| try std.fmt.allocPrint(allocator, "{d}", .{number}) else "-";
}

/// Use an arena for rendering; returned text and intermediate values use `allocator`.
pub fn renderComparison(allocator: Allocator, baseline_ref: []const u8, baseline: []const Measurement, target_ref: []const u8, target: []const Measurement, mode: []const u8) ![]const u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    const w = &output.writer;
    if (std.mem.eql(u8, mode, "ReleaseFast")) {
        try w.writeAll("### SIMD AOT benchmark comparison\n");
    } else {
        try w.print("### SIMD AOT benchmark comparison ({s})\n", .{mode});
    }
    try w.writeAll(
        \\
        \\| Case | Engine | Ref | Status | Result | Median run | Median compile | Code size | Iterations |
        \\|---|---|---|---|---:|---:|---:|---:|---:|
        \\
    );
    for (try keys(allocator, baseline, target)) |key| {
        for ([_][]const Measurement{ baseline, target }, [_][]const u8{ baseline_ref, target_ref }, [_][]const u8{ "baseline", "target" }) |rows, ref, role| {
            if (try summarize(allocator, rows, key)) |summary| {
                try w.print("| `{s}` | `{s}` | `{s} ({s})` | {s} | {s} | {s} | {s} | {s} | {d} |\n", .{
                    key.case,                                key.engine,                                  ref,                                           role,               summary.status, try integerText(allocator, summary.result),
                    try renderNs(allocator, summary.run_ns), try renderNs(allocator, summary.compile_ns), try integerText(allocator, summary.code_size), summary.iterations,
                });
            } else {
                try w.print("| `{s}` | `{s}` | `{s} ({s})` | missing | - | - | - | - | - |\n", .{ key.case, key.engine, ref, role });
            }
        }
    }
    try w.writeAll(
        \\
        \\AOT rows with `unsupported` are expected for refs or SIMD opcode families without native v128 lowering.
        \\CoreMark is scalar, so this harness is the SIMD-specific signal for issue #220.
    );
    return output.toOwnedSlice();
}

pub fn renderOptimizeComparison(allocator: Allocator, target_ref: []const u8, fast: []const Measurement, safe: []const Measurement) ![]const u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    const w = &output.writer;
    try w.writeAll(
        \\### SIMD AOT optimize-mode comparison
        \\
        \\| Case | Engine | Ref | Fast status | Safe status | Fast median run | Safe median run | Safe/Fast run | Fast median compile | Safe median compile | Safe/Fast compile |
        \\|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
        \\
    );
    for (try keys(allocator, fast, safe)) |key| {
        const fast_summary = try summarize(allocator, fast, key);
        const safe_summary = try summarize(allocator, safe, key);
        const fast_run = if (fast_summary) |value| value.run_ns else null;
        const safe_run = if (safe_summary) |value| value.run_ns else null;
        const fast_compile = if (fast_summary) |value| value.compile_ns else null;
        const safe_compile = if (safe_summary) |value| value.compile_ns else null;
        const run_ratio = try optimize.formatRatio(allocator, if (safe_run) |value| numberFloat(value) else null, if (fast_run) |value| numberFloat(value) else null);
        const compile_ratio = try optimize.formatRatio(allocator, if (safe_compile) |value| numberFloat(value) else null, if (fast_compile) |value| numberFloat(value) else null);
        try w.print("| `{s}` | `{s}` | `{s}` (target) | {s} | {s} | {s} | {s} | {s} | {s} | {s} | {s} |\n", .{
            key.case,                                              key.engine,                                            target_ref,
            if (fast_summary) |value| value.status else "missing", if (safe_summary) |value| value.status else "missing", try renderNs(allocator, fast_run),
            try renderNs(allocator, safe_run),                     run_ratio,                                             try renderNs(allocator, fast_compile),
            try renderNs(allocator, safe_compile),                 compile_ratio,
        });
    }
    try w.writeAll(
        \\
        \\Safe/Fast ratios compare ReleaseSafe timing divided by ReleaseFast timing for the same target/case/engine.
    );
    return output.toOwnedSlice();
}

test "comparison uses successful rows, medians, and missing counterparts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const baseline = try parser.parse(allocator, "bench\tadd\taot\tunsupported\t-\t-\t-\t10\t-\n" ++
        "bench\tadd\taot\tok\t42\t1000\t2000\t10\t30\n" ++
        "bench\tadd\taot\tok\t43\t3000\t4000\t10\t31\n", 1);
    const target = try parser.parse(allocator, "bench\tmul\tjit\tok\t7\t100\t400\t5\t-\n", 1);
    const report = try renderComparison(allocator, "old", baseline, "new", target, "ReleaseFast");
    try std.testing.expect(std.mem.indexOf(u8, report, "| `add` | `aot` | `old (baseline)` | ok | 42 | 3.000 us | 2.000 us | 30 | 10 |") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "| `add` | `aot` | `new (target)` | missing | - | - | - | - | - |") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "| `mul` | `jit` | `new (target)` | ok | 7 | 400 ns | 100 ns | - | 5 |") != null);
}

test "optimize comparison uses safe over fast timing and missing markers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const fast = try parser.parse(allocator, "bench\tadd\taot\tok\t1\t100\t400\t5\t-\n", 1);
    const safe = try parser.parse(allocator, "bench\tadd\taot\tok\t1\t200\t600\t5\t-\n" ++
        "bench\tmul\tjit\tunsupported\t-\t-\t-\t5\t-\n", 1);
    const report = try renderOptimizeComparison(allocator, "new", fast, safe);
    try std.testing.expect(std.mem.indexOf(u8, report, "| `add` | `aot` | `new` (target) | ok | ok | 400 ns | 600 ns | \xc3\x971.50 | 100 ns | 200 ns | \xc3\x972.00 |") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "| `mul` | `jit` | `new` (target) | missing | unsupported | - | - | \xe2\x80\x94 | - | - | \xe2\x80\x94 |") != null);
}

test "even medians retain the full sum of representable integer samples" {
    const maximum = std.math.maxInt(i128);
    const minimum = std.math.minInt(i128);
    var upper = [_]i128{ maximum, maximum };
    var lower = [_]i128{ minimum, minimum };
    var mixed = [_]i128{ minimum, maximum };
    try std.testing.expectEqual(@as(f64, @floatFromInt(maximum)), (try median(&upper)).?.float);
    try std.testing.expectEqual(@as(f64, @floatFromInt(minimum)), (try median(&lower)).?.float);
    try std.testing.expectEqual(@as(f64, -0.5), (try median(&mixed)).?.float);
}

test "comparison renders extreme even medians like the reference" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const maximum = std.math.maxInt(i128);
    const minimum = std.math.minInt(i128);
    const input = try std.fmt.allocPrint(allocator, "bench\thigh\taot\tok\t1\t{d}\t{d}\t1\t-\n" ++
        "bench\thigh\taot\tok\t1\t{d}\t{d}\t1\t-\n" ++
        "bench\tlow\taot\tok\t1\t{d}\t{d}\t1\t-\n" ++
        "bench\tlow\taot\tok\t1\t{d}\t{d}\t1\t-\n" ++
        "bench\tmix\taot\tok\t1\t{d}\t{d}\t1\t-\n" ++
        "bench\tmix\taot\tok\t1\t{d}\t{d}\t1\t-\n", .{ maximum, maximum, maximum, maximum, minimum, minimum, minimum, minimum, minimum, minimum, maximum, maximum });
    const rows = try parser.parse(allocator, input, 1);
    const rendered = try renderComparison(allocator, "old", rows, "new", &.{}, "ReleaseFast");
    try std.testing.expect(std.mem.indexOf(u8, rendered, "| `high` | `aot` | `old (baseline)` | ok | 1 | 170141183460469224032477495951360.000 ms |") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "| `low` | `aot` | `old (baseline)` | ok | 1 | -170141183460469231731687303715884105728 ns |") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "| `mix` | `aot` | `old (baseline)` | ok | 1 | -0 ns |") != null);
}
