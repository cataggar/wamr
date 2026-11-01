//! Validated fixtures and AOT size accounting for the loop-pass benchmark.
const std = @import("std");
const benchmark_stats = @import("benchmark_stats.zig");
const fixed = @import("decimal_format.zig").fixed;
const Allocator = std.mem.Allocator;

pub const Fixture = struct {
    name: []const u8,
    path: []const u8,
    sha256: []const u8,
};

pub const fixtures = [_]Fixture{
    .{ .name = "iv_store", .path = "tests/benchmarks/loop-passes/iv_store.wasm", .sha256 = "b1979dd330c14d5f898b8a7c8c313e58f6521ad6eb7db9a6e7d7e78788ac6e72" },
    .{ .name = "unroll4", .path = "tests/benchmarks/loop-passes/unroll4.wasm", .sha256 = "6870b3373e4098117c82b6736d0ca7cbcc7d8d747fe87ca5b7d1ebf0e4d12890" },
};

pub fn validateFixtures(allocator: Allocator, io: std.Io, repo: []const u8) !void {
    for (fixtures) |fixture| {
        const path = try std.fs.path.join(allocator, &.{ repo, fixture.path });
        defer allocator.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), fixture.sha256))
            return error.FixtureChecksumMismatch;
    }
}

pub const AotSize = struct { file_bytes: usize, text_bytes: u32 };

pub fn parseAotSize(bytes: []const u8) !AotSize {
    if (bytes.len < 8 or !std.mem.eql(u8, bytes[0..4], "\x00aot")) return error.InvalidAotFile;
    var pos: usize = 8;
    var text_bytes: ?u32 = null;
    while (bytes.len - pos >= 8) {
        const section = std.mem.readInt(u32, bytes[pos..][0..4], .little);
        const size = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        pos += 8;
        if (size > bytes.len - pos) return error.TruncatedAotSection;
        if (section == 2) text_bytes = size;
        pos += size;
    }
    return .{ .file_bytes = bytes.len, .text_bytes = text_bytes orelse return error.MissingAotText };
}

pub const Counts = struct { warmups: u64, runs: u64 };

pub fn counts(profile: []const u8, warmups: ?i64, runs: ?i64) !Counts {
    const defaults: Counts = if (std.mem.eql(u8, profile, "authoritative"))
        .{ .warmups = 2, .runs = 10 }
    else if (std.mem.eql(u8, profile, "ci"))
        .{ .warmups = 0, .runs = 3 }
    else
        return error.InvalidProfile;
    if (warmups) |value| if (value < 0) return error.InvalidWarmups;
    if (runs) |value| if (value <= 0) return error.InvalidRuns;
    return .{
        .warmups = if (warmups) |value| @intCast(value) else defaults.warmups,
        .runs = if (runs) |value| @intCast(value) else defaults.runs,
    };
}

pub const Stats = benchmark_stats.Stats;
pub const stats = benchmark_stats.summarize;

pub fn speedupPct(allocator: Allocator, baseline: []const f64, target: []const f64) !f64 {
    const baseline_mean = (try stats(allocator, baseline)).mean;
    const target_mean = (try stats(allocator, target)).mean;
    if (target_mean == 0) return error.ZeroTargetMean;
    return (baseline_mean / target_mean - 1.0) * 100.0;
}

pub const Executable = struct { cwd: []const u8, wamr: []const u8, artifact: []const u8 };
pub const Measurements = struct { baseline: []f64, target: []f64 };

fn measureOnce(allocator: Allocator, io: std.Io, executable: Executable) !f64 {
    const started = std.Io.Clock.awake.now(io);
    const args = [_][]const u8{ executable.wamr, "run", executable.artifact };
    const result = try std.process.run(allocator, io, .{ .argv = &args, .cwd = .{ .path = executable.cwd } });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("[harness] command failed ({f}): {s} run {s}\n[harness] cwd: {s}\n[harness] --- stdout ---\n{s}\n[harness] --- stderr ---\n{s}\n", .{
            result.term, executable.wamr, executable.artifact, executable.cwd, result.stdout, result.stderr,
        });
        return error.BenchmarkCommandFailed;
    }
    return @as(f64, @floatFromInt(started.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) / std.time.ns_per_s;
}

pub fn measureFixture(allocator: Allocator, io: std.Io, baseline: Executable, target: Executable, name: []const u8, warmups: usize, runs: usize) !Measurements {
    if (runs == 0) return error.EmptyMeasurements;
    const total = std.math.add(usize, warmups, runs) catch return error.TooManyMeasurements;
    const baseline_samples = try allocator.alloc(f64, runs);
    errdefer allocator.free(baseline_samples);
    const target_samples = try allocator.alloc(f64, runs);
    errdefer allocator.free(target_samples);
    for (0..total) |index| {
        const warmup = index < warmups;
        const phase = if (warmup) "warmup" else "run";
        const phase_index = if (warmup) index + 1 else index - warmups + 1;
        for (0..2) |round| {
            const is_baseline = (index + round) % 2 == 0;
            const elapsed = try measureOnce(allocator, io, if (is_baseline) baseline else target);
            const formatted = try fixed(allocator, elapsed, 6);
            defer allocator.free(formatted);
            std.debug.print("[harness]   {s} {s} {s} {d}/{d}: {s} s{s}\n", .{
                name,                          if (is_baseline) "baseline" else "target", phase,                              phase_index,
                if (warmup) warmups else runs, formatted,                                 if (warmup) " (discarded)" else "",
            });
            if (!warmup) (if (is_baseline) baseline_samples else target_samples)[index - warmups] = elapsed;
        }
    }
    return .{ .baseline = baseline_samples, .target = target_samples };
}

pub const ReportRef = struct {
    ref: []const u8,
    sha: []const u8,
    sizes: [fixtures.len]AotSize,
};

pub const Report = struct {
    baseline: ReportRef,
    target: ReportRef,
    measurements: [fixtures.len][2][]const f64,
    profile: []const u8,
    sample_counts: Counts,
    optimize: []const u8,
    architecture: []const u8,
    cpu_count: []const u8,
    cpu_model: []const u8,
};

pub fn renderReport(allocator: Allocator, report: Report) ![]u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    errdefer output.deinit();
    const w = &output.writer;
    try w.print(
        \\### Focused AOT loop-pass comparison
        \\
        \\- Measurement profile: `{s}` ({d} warmups discarded, {d} measured runs per fixture/ref)
        \\- WAMR optimize mode: `{s}`
        \\
        \\| Fixture | Ref | Mean s | Median s | Range s | Text bytes | AOT bytes | Samples s |
        \\|---|---|---:|---:|---:|---:|---:|---|
        \\
    , .{ report.profile, report.sample_counts.warmups, report.sample_counts.runs, report.optimize });
    for (fixtures, 0..) |fixture, index| {
        const samples = report.measurements[index];
        for ([_]ReportRef{ report.baseline, report.target }, [_][]const u8{ "baseline", "target" }, 0..) |reference, role, role_index| {
            const summary = try stats(allocator, samples[role_index]);
            const mean = try fixed(allocator, summary.mean, 6);
            defer allocator.free(mean);
            const median = try fixed(allocator, summary.median, 6);
            defer allocator.free(median);
            const min = try fixed(allocator, summary.min, 6);
            defer allocator.free(min);
            const max = try fixed(allocator, summary.max, 6);
            defer allocator.free(max);
            try w.print("| `{s}` | `{s} ({s}, {s})` | {s} | {s} | {s}–{s} | {d} | {d} | ", .{
                fixture.name,                      reference.ref,                     role, reference.sha[0..@min(12, reference.sha.len)],
                mean,                              median,                            min,  max,
                reference.sizes[index].text_bytes, reference.sizes[index].file_bytes,
            });
            for (samples[role_index], 0..) |sample, sample_index| {
                if (sample_index != 0) try w.writeAll(", ");
                const text = try fixed(allocator, sample, 6);
                defer allocator.free(text);
                try w.writeAll(text);
            }
            try w.writeAll(" |\n");
        }
        const speedup = try speedupPct(allocator, samples[0], samples[1]);
        const magnitude = try fixed(allocator, @abs(speedup), 2);
        defer allocator.free(magnitude);
        const negative = @as(u64, @bitCast(speedup)) >> 63 != 0;
        try w.print("| `{s}` | **target speedup** | **{s}{s}%** |  |  |  |  |  |\n", .{ fixture.name, if (negative) "-" else "+", magnitude });
    }
    try w.print(
        \\
        \\A sample is valid only when the benchmark `_start` returns normally; each fixture traps on an incorrect result.
        \\
        \\_Host: arch `{s}` · {s} vCPU · `{s}`_
    , .{ report.architecture, report.cpu_count, report.cpu_model });
    return output.toOwnedSlice();
}

test "tracked benchmark fixtures match their pinned hashes" {
    try validateFixtures(std.testing.allocator, std.testing.io, ".");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.createDirPath(io, "tests/benchmarks/loop-passes");
    try temp.dir.writeFile(io, .{ .sub_path = fixtures[0].path, .data = "corrupted" });
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    try std.testing.expectError(error.FixtureChecksumMismatch, validateFixtures(allocator, io, root));
}

test "AOT text sizes reject malformed files and use the final text section" {
    const valid = "\x00aot\x01\x00\x00\x00" ++
        "\x02\x00\x00\x00\x02\x00\x00\x00ab" ++
        "\x01\x00\x00\x00\x01\x00\x00\x00x" ++
        "\x02\x00\x00\x00\x03\x00\x00\x00xyz";
    try std.testing.expectEqualDeep(AotSize{ .file_bytes = valid.len, .text_bytes = 3 }, try parseAotSize(valid));
    try std.testing.expectError(error.InvalidAotFile, parseAotSize("bad"));
    try std.testing.expectError(error.TruncatedAotSection, parseAotSize(valid[0 .. valid.len - 1]));
    try std.testing.expectError(error.MissingAotText, parseAotSize("\x00aot\x01\x00\x00\x00"));
    try std.testing.expectEqualDeep(AotSize{ .file_bytes = valid.len + 1, .text_bytes = 3 }, try parseAotSize(valid ++ "\x00"));
}

test "benchmark sampling counts enforce each profile" {
    try std.testing.expectEqualDeep(Counts{ .warmups = 2, .runs = 10 }, try counts("authoritative", null, null));
    try std.testing.expectEqualDeep(Counts{ .warmups = 0, .runs = 3 }, try counts("ci", null, null));
    try std.testing.expectEqualDeep(Counts{ .warmups = 1, .runs = 2 }, try counts("ci", 1, 2));
    try std.testing.expectError(error.InvalidWarmups, counts("ci", -1, null));
    try std.testing.expectError(error.InvalidRuns, counts("ci", null, 0));
}

test "benchmark statistics use accurately rounded means and elapsed-time ratios" {
    const allocator = std.testing.allocator;
    const summary = try stats(allocator, &.{ 1e16, 1, 1, 1 });
    try std.testing.expectEqual(@as(f64, 2500000000000001), summary.mean);
    try std.testing.expectEqual(@as(f64, 1), summary.median);
    try std.testing.expectEqual(@as(f64, 1), summary.min);
    try std.testing.expectEqual(@as(f64, 1e16), summary.max);
    try std.testing.expectEqual(@as(f64, 100), try speedupPct(allocator, &.{ 2, 2 }, &.{ 1, 1 }));
    try std.testing.expectEqual(@as(f64, -50), try speedupPct(allocator, &.{ 1, 1 }, &.{ 2, 2 }));
    try std.testing.expectError(error.EmptyMeasurements, stats(allocator, &.{}));
    try std.testing.expectError(error.ZeroTargetMean, speedupPct(allocator, &.{1}, &.{0}));
}

test "benchmark report retains sample order, sizes, and signed speedups" {
    const allocator = std.testing.allocator;
    const report = try renderReport(allocator, .{
        .baseline = .{ .ref = "old", .sha = "123456789012345", .sizes = .{ .{ .file_bytes = 12, .text_bytes = 4 }, .{ .file_bytes = 20, .text_bytes = 8 } } },
        .target = .{ .ref = "new", .sha = "abcdefabcdef123", .sizes = .{ .{ .file_bytes = 10, .text_bytes = 3 }, .{ .file_bytes = 24, .text_bytes = 9 } } },
        .measurements = .{ .{ &.{ 2, 2 }, &.{ 1, 1 } }, .{ &.{ 1, 1 }, &.{ 2, 2 } } },
        .profile = "ci",
        .sample_counts = .{ .warmups = 0, .runs = 2 },
        .optimize = "ReleaseFast",
        .architecture = "arm64",
        .cpu_count = "4",
        .cpu_model = "test CPU",
    });
    defer allocator.free(report);
    try std.testing.expect(std.mem.indexOf(u8, report, "| `iv_store` | `old (baseline, 123456789012)` | 2.000000 | 2.000000 | 2.000000–2.000000 | 4 | 12 | 2.000000, 2.000000 |") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "| `iv_store` | **target speedup** | **+100.00%**") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "| `unroll4` | **target speedup** | **-50.00%**") != null);
    try std.testing.expect(std.mem.endsWith(u8, report, "_Host: arch `arm64` · 4 vCPU · `test CPU`_"));
}

test "benchmark sampling alternates execution order and propagates command failures" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(io, .{ .sub_path = "run", .data = "printf %s \"$1\" >> order\n[ \"$1\" != F ]\n" });
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const baseline: Executable = .{ .cwd = root, .wamr = "/bin/sh", .artifact = "B" };
    const target: Executable = .{ .cwd = root, .wamr = "/bin/sh", .artifact = "T" };
    const measured = try measureFixture(allocator, io, baseline, target, "fixture", 1, 2);
    defer allocator.free(measured.baseline);
    defer allocator.free(measured.target);
    try std.testing.expectEqual(@as(usize, 2), measured.baseline.len);
    try std.testing.expectEqual(@as(usize, 2), measured.target.len);
    const order = try temp.dir.readFileAlloc(io, "order", allocator, .unlimited);
    defer allocator.free(order);
    try std.testing.expectEqualStrings("BTTBBT", order);
    try std.testing.expectError(error.BenchmarkCommandFailed, measureFixture(allocator, io, baseline, .{ .cwd = root, .wamr = "/bin/sh", .artifact = "F" }, "fixture", 0, 1));
}
