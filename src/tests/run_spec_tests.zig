//! CLI runner for WebAssembly spec tests in JSON format.
//!
//! Usage: spec-test-runner <json-dir>
//!
//! Scans the given directory for .json files produced by wast2json,
//! runs each through the spec test runner, and prints a summary.

const std = @import("std");
const spec_json_runner = @import("spec_json_runner.zig");

const Io = std.Io;
const Dir = Io.Dir;
const print = std.debug.print;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_iter.deinit();

    _ = args_iter.next(); // skip program name
    const json_dir = args_iter.next() orelse {
        print("Usage: spec-test-runner <json-dir>\n", .{});
        std.process.exit(1);
    };

    print("\n=== WebAssembly Spec Test Runner ===\n\n", .{});

    var dir = Dir.cwd().openDir(init.io, json_dir, .{ .iterate = true }) catch |err| {
        print("Error: cannot open directory '{s}': {}\n", .{ json_dir, err });
        std.process.exit(1);
    };
    defer dir.close(init.io);

    var total_passed: u32 = 0;
    var total_failed: u32 = 0;
    var total_skipped: u32 = 0;
    var total_tests: u32 = 0;
    var file_count: u32 = 0;

    // Collect JSON file names
    var json_files: std.ArrayList([]const u8) = .empty;
    defer {
        for (json_files.items) |name| allocator.free(name);
        json_files.deinit(allocator);
    }

    var iter = dir.iterate();
    while (iter.next(init.io) catch null) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".json")) {
            const name_copy = try allocator.dupe(u8, entry.name);
            try json_files.append(allocator, name_copy);
        }
    }

    // Sort for deterministic output
    std.mem.sort([]const u8, json_files.items, {}, struct {
        fn cmp(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.cmp);

    for (json_files.items) |name| {
        const json_path = try std.fs.path.join(allocator, &.{ json_dir, name });
        defer allocator.free(json_path);

        const result = spec_json_runner.runSpecTestFile(json_path, allocator, init.io) catch |err| {
            print("  {s:<20} ERROR: {}\n", .{ name, err });
            continue;
        };

        const status = if (result.failed == 0) "OK" else "FAIL";
        print("  {s:<20} {s:>4}  pass={d:<5} fail={d:<5} skip={d:<5} total={d}\n", .{
            name,
            status,
            result.passed,
            result.failed,
            result.skipped,
            result.total,
        });

        total_passed += result.passed;
        total_failed += result.failed;
        total_skipped += result.skipped;
        total_tests += result.total;
        file_count += 1;
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
