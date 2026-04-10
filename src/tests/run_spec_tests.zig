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

const Dir = std.fs.Dir;
const print = std.debug.print;

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        print("Usage: spec-test-runner <dir-or-file>\n", .{});
        print("  Accepts a directory of .json/.wast files or a single .wast file\n", .{});
        std.process.exit(1);
    }
    const test_path = args[1];

    print("\n=== WebAssembly Spec Test Runner ===\n\n", .{});

    // Check if argument is a single file
    if (std.mem.endsWith(u8, test_path, ".wast") or std.mem.endsWith(u8, test_path, ".json")) {
        const result = if (std.mem.endsWith(u8, test_path, ".wast"))
            runWastFile(test_path, allocator)
        else
            spec_json_runner.runSpecTestFile(test_path, allocator) catch |err| {
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

    var dir = std.fs.cwd().openDir(test_path, .{ .iterate = true }) catch |err| {
        print("Error: cannot open directory '{s}': {}\n", .{ test_path, err });
        std.process.exit(1);
    };
    defer dir.close();

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
    while (iter.next() catch null) |entry| {
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

        if (std.mem.endsWith(u8, name, ".json")) {
            // Legacy JSON format
            const result = spec_json_runner.runSpecTestFile(full_path, allocator) catch |err| {
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
            const result = runWastFile(full_path, allocator);
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

    print("\n", .{});
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
fn runWastFile(path: []const u8, allocator: std.mem.Allocator) TestResult {
    const wast_runner = @import("wast_runner.zig");
    const source = std.fs.cwd().readFileAlloc(allocator, path, 64 * 1024 * 1024) catch {
        return .{ .file = path, .passed = 0, .failed = 0, .skipped = 1, .total = 1 };
    };
    defer allocator.free(source);

    const base = std.fs.path.stem(path);
    const result = wast_runner.run(allocator, source, base);
    return .{
        .file = path,
        .passed = result.passed,
        .failed = result.failed,
        .skipped = result.skipped,
        .total = result.total(),
    };
}
