//! WebAssembly Spec Test Runner
//!
//! Provides infrastructure for running Wasm conformance tests.
//! Tests can assert function return values, trap conditions,
//! and module validation errors.

const std = @import("std");
const wamr = @import("../api/wamr.zig");
const types = @import("../runtime/common/types.zig");

pub const TestResult = union(enum) {
    pass,
    fail: []const u8,
    skip: []const u8,
};

pub const Assertion = union(enum) {
    /// Assert a function returns specific i32 values.
    assert_return_i32: struct {
        func_name: []const u8,
        args: []const i32,
        expected: i32,
    },
    /// Assert a function traps.
    assert_trap: struct {
        func_name: []const u8,
        args: []const i32,
    },
    /// Assert a module fails to load (invalid).
    assert_invalid: struct {
        wasm: []const u8,
    },
};

/// A spec test case — a module + a list of assertions.
pub const TestCase = struct {
    name: []const u8,
    wasm: []const u8,
    assertions: []const Assertion,
};

/// Run a single test case and return results.
pub fn runTestCase(tc: *const TestCase, allocator: std.mem.Allocator) ![]TestResult {
    var results = try allocator.alloc(TestResult, tc.assertions.len);

    var runtime = wamr.Runtime.init(allocator);
    defer runtime.deinit();

    // Try to load the module
    var module = runtime.loadModule(tc.wasm) catch |err| {
        // If load fails, check if any assertions expect invalid
        for (tc.assertions, 0..) |assertion, i| {
            results[i] = switch (assertion) {
                .assert_invalid => .pass,
                else => .{ .fail = @errorName(err) },
            };
        }
        return results;
    };
    defer module.deinit();

    var instance = module.instantiate() catch |err| {
        for (tc.assertions, 0..) |_, i| {
            results[i] = .{ .fail = @errorName(err) };
        }
        return results;
    };
    defer instance.deinit();

    for (tc.assertions, 0..) |assertion, i| {
        results[i] = runAssertion(&instance, &assertion);
    }

    return results;
}

fn runAssertion(instance: *wamr.Instance, assertion: *const Assertion) TestResult {
    switch (assertion.*) {
        .assert_return_i32 => |a| {
            const result = instance.callI32(a.func_name, a.args) catch |err| {
                return .{ .fail = @errorName(err) };
            };
            if (result == a.expected) return .pass;
            return .{ .fail = "unexpected return value" };
        },
        .assert_trap => |a| {
            _ = instance.callI32(a.func_name, a.args) catch {
                return .pass; // trap is expected
            };
            return .{ .fail = "expected trap but function returned normally" };
        },
        .assert_invalid => {
            return .{ .fail = "module loaded but was expected to be invalid" };
        },
    }
}

/// Summary statistics.
pub const Stats = struct {
    passed: u32 = 0,
    failed: u32 = 0,
    skipped: u32 = 0,

    pub fn total(self: Stats) u32 {
        return self.passed + self.failed + self.skipped;
    }
};

pub fn summarize(results: []const TestResult) Stats {
    var stats = Stats{};
    for (results) |r| {
        switch (r) {
            .pass => stats.passed += 1,
            .fail => stats.failed += 1,
            .skip => stats.skipped += 1,
        }
    }
    return stats;
}

// ── Tests ──────────────────────────────────────────────────────────────────

// Helper: build the "add" Wasm module
// (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add)
const add_wasm = [_]u8{
    0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7F, 0x7F, 0x01, 0x7F,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
    0x0A, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B,
};

test "spec: assert_return i32.add" {
    const tc = TestCase{
        .name = "i32.add",
        .wasm = &add_wasm,
        .assertions = &.{
            .{ .assert_return_i32 = .{ .func_name = "add", .args = &.{ 1, 2 }, .expected = 3 } },
            .{ .assert_return_i32 = .{ .func_name = "add", .args = &.{ 0, 0 }, .expected = 0 } },
            .{ .assert_return_i32 = .{ .func_name = "add", .args = &.{ -1, 1 }, .expected = 0 } },
        },
    };
    const results = try runTestCase(&tc, std.testing.allocator);
    defer std.testing.allocator.free(results);
    const stats = summarize(results);
    try std.testing.expectEqual(@as(u32, 3), stats.passed);
    try std.testing.expectEqual(@as(u32, 0), stats.failed);
}

test "spec: assert_trap unreachable" {
    // Module with: (func (export "trap") unreachable)
    const trap_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        0x03, 0x02, 0x01, 0x00,
        0x07, 0x08, 0x01, 0x04, 0x74, 0x72, 0x61, 0x70, 0x00, 0x00,
        0x0A, 0x05, 0x01, 0x03, 0x00, 0x00, 0x0B,
    };
    const tc = TestCase{
        .name = "unreachable trap",
        .wasm = &trap_wasm,
        .assertions = &.{
            .{ .assert_trap = .{ .func_name = "trap", .args = &.{} } },
        },
    };
    const results = try runTestCase(&tc, std.testing.allocator);
    defer std.testing.allocator.free(results);
    const stats = summarize(results);
    try std.testing.expectEqual(@as(u32, 1), stats.passed);
}

test "spec: assert_invalid bad magic" {
    const bad = [_]u8{ 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00 };
    const tc = TestCase{
        .name = "invalid magic",
        .wasm = &bad,
        .assertions = &.{
            .{ .assert_invalid = .{ .wasm = &bad } },
        },
    };
    const results = try runTestCase(&tc, std.testing.allocator);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(u32, 1), summarize(results).passed);
}

test "spec: summarize" {
    const results = [_]TestResult{ .pass, .pass, .{ .fail = "err" }, .{ .skip = "todo" } };
    const stats = summarize(&results);
    try std.testing.expectEqual(@as(u32, 2), stats.passed);
    try std.testing.expectEqual(@as(u32, 1), stats.failed);
    try std.testing.expectEqual(@as(u32, 1), stats.skipped);
    try std.testing.expectEqual(@as(u32, 4), stats.total());
}
