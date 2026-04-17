//! CLI runner for WebAssembly spec tests.
//!
//! Usage: spec-test-runner <dir>
//!
//! Accepts a directory containing either:
//! - .json files (produced by wast2json) — legacy format
//! - .wast files (WebAssembly spec test format) — native, no conversion needed
//!
//! Runs each test file and prints a summary.

const std = @import("std");
const spec_json_runner = @import("spec_json_runner.zig");
const aot_skiplist = @import("aot_skiplist.zig");

const Dir = std.Io.Dir;
const print = std.debug.print;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    // Parse `--mode=interp|aot` (optional; default interp). Remaining
    // positional arg is the test path.
    var mode: spec_json_runner.Mode = .interp;
    var test_path_opt: ?[]const u8 = null;
    var verbose_skips = false;
    for (args[1..]) |a| {
        if (std.mem.startsWith(u8, a, "--mode=")) {
            const v = a["--mode=".len..];
            if (std.mem.eql(u8, v, "interp")) {
                mode = .interp;
            } else if (std.mem.eql(u8, v, "aot")) {
                mode = .aot;
            } else {
                print("Unknown mode '{s}' (expected interp|aot)\n", .{v});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, a, "--mode")) {
            print("--mode requires =interp or =aot\n", .{});
            std.process.exit(1);
        } else if (std.mem.eql(u8, a, "--verbose-skips")) {
            verbose_skips = true;
        } else if (test_path_opt == null) {
            test_path_opt = a;
        } else {
            print("Unexpected argument: {s}\n", .{a});
            std.process.exit(1);
        }
    }

    if (test_path_opt == null) {
        print("Usage: spec-test-runner [--mode=interp|aot] <dir-or-file>\n", .{});
        print("  Accepts a directory of .json/.wast files or a single .wast file\n", .{});
        std.process.exit(1);
    }
    const test_path = test_path_opt.?;

    print("\n=== WebAssembly Spec Test Runner ({s}) ===\n\n", .{@tagName(mode)});

    var skip_histogram = spec_json_runner.SkipHistogram.init(allocator);
    defer skip_histogram.deinit();
    if (verbose_skips and mode == .aot) {
        spec_json_runner.setSkipHistogram(&skip_histogram);
    }
    defer spec_json_runner.setSkipHistogram(null);

    // Check if argument is a single file
    if (std.mem.endsWith(u8, test_path, ".wast") or std.mem.endsWith(u8, test_path, ".json")) {
        const result = if (std.mem.endsWith(u8, test_path, ".wast"))
            runWastFile(test_path, mode, allocator, io)
        else
            spec_json_runner.runSpecTestFileMode(test_path, mode, allocator, io) catch |err| {
                print("ERROR: {}\n", .{err});
                std.process.exit(1);
            };

        const name = std.fs.path.basename(test_path);
        printResult(name, result);
        print("\n--- Summary ---\nPassed: {d}\nFailed: {d}\nSkipped: {d}\nTotal: {d}\n\n", .{
            result.passed, result.failed, result.skipped, result.total,
        });
        return;
    }

    var dir = std.Io.Dir.cwd().openDir(io, test_path, .{ .iterate = true }) catch |err| {
        print("Error: cannot open directory '{s}': {}\n", .{ test_path, err });
        std.process.exit(1);
    };
    defer dir.close(io);

    var total_passed: u32 = 0;
    var total_failed: u32 = 0;
    var total_skipped: u32 = 0;
    var total_tests: u32 = 0;
    var file_count: u32 = 0;

    // Collect test file names (.json or .wast)
    var test_files: std.ArrayList([]const u8) = .empty;
    defer {
        for (test_files.items) |name| allocator.free(name);
        test_files.deinit(allocator);
    }

    var iter = dir.iterate();
    while (iter.next(io) catch null) |entry| {
        if (entry.kind == .file and
            (std.mem.endsWith(u8, entry.name, ".json") or
            std.mem.endsWith(u8, entry.name, ".wast")))
        {
            const name_copy = try allocator.dupe(u8, entry.name);
            try test_files.append(allocator, name_copy);
        }
    }

    // Sort for deterministic output
    std.mem.sort([]const u8, test_files.items, {}, struct {
        fn cmp(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.cmp);

    for (test_files.items) |name| {
        const full_path = try std.fs.path.join(allocator, &.{ test_path, name });
        defer allocator.free(full_path);

        // Skip files known to cause panics (IP misalignment issues for
        // interp, codegen assertion failures for AOT).
        const skip_files = [_][]const u8{"memory_trap64.wast"};
        var should_skip = false;
        for (skip_files) |skip| {
            if (std.mem.eql(u8, name, skip)) {
                should_skip = true;
                break;
            }
        }
        if (!should_skip and mode == .aot and aot_skiplist.isSkippedInAot(name)) {
            should_skip = true;
        }
        if (should_skip) {
            print("  {s:<25} SKIP (known panic)\n", .{name});
            continue;
        }

        if (std.mem.endsWith(u8, name, ".json")) {
            // Legacy JSON format
            const result = spec_json_runner.runSpecTestFileMode(full_path, mode, allocator, io) catch |err| {
                print("  {s:<25} ERROR: {}\n", .{ name, err });
                continue;
            };
            printResult(name, result);
            total_passed += result.passed;
            total_failed += result.failed;
            total_skipped += result.skipped;
            total_tests += result.total;
            file_count += 1;
        } else if (std.mem.endsWith(u8, name, ".wast")) {
            // Native .wast format — use wabt to run
            const result = runWastFile(full_path, mode, allocator, io);
            printResult(name, result);
            total_passed += result.passed;
            total_failed += result.failed;
            total_skipped += result.skipped;
            total_tests += result.total;
            file_count += 1;
        }
    }

    print("\n--- Summary ---\n", .{});
    print("Files:   {d}\n", .{file_count});
    print("Passed:  {d}\n", .{total_passed});
    print("Failed:  {d}\n", .{total_failed});
    print("Skipped: {d}\n", .{total_skipped});
    print("Total:   {d}\n", .{total_tests});

    if (total_tests > 0) {
        const pass_rate = @as(f64, @floatFromInt(total_passed)) / @as(f64, @floatFromInt(total_tests)) * 100.0;
        print("Pass rate: {d:.1}%\n", .{pass_rate});
    }

    if (verbose_skips and mode == .aot) {
        printSkipHistogram(&skip_histogram, allocator);
    }

    print("\n", .{});
}

fn printSkipHistogram(h: *spec_json_runner.SkipHistogram, allocator: std.mem.Allocator) void {
    const Entry = struct { reason: []const u8, count: u32 };
    var entries: std.ArrayList(Entry) = .empty;
    defer entries.deinit(allocator);

    var it = h.iterator();
    while (it.next()) |e| {
        entries.append(allocator, .{ .reason = e.key_ptr.*, .count = e.value_ptr.* }) catch return;
    }
    std.mem.sort(Entry, entries.items, {}, struct {
        fn cmp(_: void, a: Entry, b: Entry) bool {
            if (a.count != b.count) return a.count > b.count;
            return std.mem.order(u8, a.reason, b.reason) == .lt;
        }
    }.cmp);

    print("\n--- Skip reasons (AOT) ---\n", .{});
    var total: u32 = 0;
    for (entries.items) |e| {
        print("  {s:<22} {d}\n", .{ e.reason, e.count });
        total += e.count;
    }
    print("  {s:<22} {d}\n", .{ "TOTAL", total });
}

const TestResult = spec_json_runner.SpecTestResult;

fn printResult(name: []const u8, result: TestResult) void {
    const status = if (result.failed == 0) "OK" else "FAIL";
    print("  {s:<25} {s:>4}  pass={d:<5} fail={d:<5} skip={d:<5} total={d}\n", .{
        name,
        status,
        result.passed,
        result.failed,
        result.skipped,
        result.total,
    });
}

/// Run a .wast file using wabt's parser + interpreter for conformance.
/// In AOT mode, .wast files are currently skipped (see issue #102 —
/// a .wast-aware AOT harness is follow-up work).
fn runWastFile(path: []const u8, mode: spec_json_runner.Mode, allocator: std.mem.Allocator, io: std.Io) TestResult {
    if (mode == .aot) {
        // Reported as a single skip so summary counters make sense.
        return .{ .file = path, .passed = 0, .failed = 0, .skipped = 1, .total = 1 };
    }
    const wast_runner = @import("wast_runner.zig");
    const source = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, @enumFromInt(64 * 1024 * 1024)) catch {
        return .{ .file = path, .passed = 0, .failed = 0, .skipped = 1, .total = 1 };
    };
    defer allocator.free(source);

    const base = std.fs.path.stem(path);
    const result = wast_runner.run(allocator, source, base, io);
    return .{
        .file = path,
        .passed = result.passed,
        .failed = result.failed,
        .skipped = result.skipped,
        .total = result.total(),
    };
}
