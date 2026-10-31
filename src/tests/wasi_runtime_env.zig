//! Adapter environment command parsing.
const std = @import("std");
const Allocator = std.mem.Allocator;
const runtime = @import("wasi_fixture_runtime.zig");

pub const Timeouts = struct {
    wait_ms: u32,
    operation_ms: u32,
    compile_ms: u32,
};

fn parseTimeout(raw: []const u8) !?u32 {
    const seconds = std.fmt.parseFloat(f64, std.mem.trim(u8, raw, " \t\r\n")) catch 0;
    if (!std.math.isFinite(seconds) or seconds <= 0) return null;
    const milliseconds = @ceil(seconds * 1000);
    if (milliseconds > std.math.maxInt(u32)) return error.TimeoutOutOfRange;
    return @intFromFloat(@max(1, milliseconds));
}

fn timeout(env: *const std.process.Environ.Map, name: []const u8, fallback_ms: u32) !u32 {
    const raw = env.get(name) orelse return fallback_ms;
    if (raw.len == 0) return fallback_ms;
    return try parseTimeout(raw) orelse {
        std.debug.print("warning: ignoring invalid {s}={s}\n", .{ name, raw });
        return fallback_ms;
    };
}

pub fn timeouts(env: *const std.process.Environ.Map, adapter: runtime.Runtime) !Timeouts {
    const wait = if (adapter == .wamr) try timeout(env, "WAMR_TESTSUITE_TIMEOUT", 5000) else 5000;
    return .{
        .wait_ms = wait,
        .operation_ms = try timeout(env, "WAMR_TESTSUITE_OP_TIMEOUT", wait),
        .compile_ms = if (adapter == .wamr) try timeout(env, "WAMR_COMPILE_TIMEOUT", 600_000) else 600_000,
    };
}

/// Equivalent to Python's `shlex.split(value)` without shell expansion.
pub fn splitCommand(allocator: Allocator, value: []const u8) ![][]u8 {
    var tokens: std.ArrayList([]u8) = .empty;
    errdefer {
        for (tokens.items) |token| allocator.free(token);
        tokens.deinit(allocator);
    }
    var token: std.ArrayList(u8) = .empty;
    defer token.deinit(allocator);
    var quote: enum { none, single, double } = .none;
    var started = false;
    var index: usize = 0;
    while (index < value.len) : (index += 1) {
        const byte = value[index];
        switch (quote) {
            .single => {
                if (byte == '\'') {
                    quote = .none;
                } else {
                    try token.append(allocator, byte);
                }
            },
            .double => switch (byte) {
                '"' => quote = .none,
                '\\' => {
                    index += 1;
                    if (index == value.len) return error.NoEscapedCharacter;
                    const escaped = value[index];
                    if (escaped != '"' and escaped != '\\') try token.append(allocator, '\\');
                    try token.append(allocator, escaped);
                },
                else => try token.append(allocator, byte),
            },
            .none => switch (byte) {
                ' ', '\t', '\r', '\n' => {
                    if (started) {
                        const owned = try token.toOwnedSlice(allocator);
                        tokens.append(allocator, owned) catch |err| {
                            allocator.free(owned);
                            return err;
                        };
                        started = false;
                    }
                },
                '\'' => {
                    started = true;
                    quote = .single;
                },
                '"' => {
                    started = true;
                    quote = .double;
                },
                '\\' => {
                    started = true;
                    index += 1;
                    if (index == value.len) return error.NoEscapedCharacter;
                    try token.append(allocator, value[index]);
                },
                else => {
                    started = true;
                    try token.append(allocator, byte);
                },
            },
        }
    }
    if (quote != .none) return error.UnclosedQuote;
    if (started) {
        const owned = try token.toOwnedSlice(allocator);
        tokens.append(allocator, owned) catch |err| {
            allocator.free(owned);
            return err;
        };
    }
    return tokens.toOwnedSlice(allocator);
}

pub fn freeCommand(allocator: Allocator, tokens: [][]u8) void {
    for (tokens) |token| allocator.free(token);
    allocator.free(tokens);
}

test "adapter commands honor shlex quoting, escapes, and empty arguments" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { source: []const u8, expected: []const []const u8 }{
        .{ .source = " x\\ y \"hi there\" 'x'\"y\" #literal", .expected = &.{ "x y", "hi there", "xy", "#literal" } },
        .{ .source = "\"a\\$b\" \"a\\\\b\" a\\\nb ''\"\"", .expected = &.{ "a\\$b", "a\\b", "a\nb", "" } },
        .{ .source = "\t \n", .expected = &.{} },
        .{ .source = "'/path with spaces/wamr' --flag=\"two words\"", .expected = &.{ "/path with spaces/wamr", "--flag=two words" } },
    };
    for (cases) |case| {
        const actual = try splitCommand(allocator, case.source);
        defer freeCommand(allocator, actual);
        try std.testing.expectEqual(case.expected.len, actual.len);
        for (case.expected, actual) |expected, item| try std.testing.expectEqualStrings(expected, item);
    }

    try std.testing.expectError(error.UnclosedQuote, splitCommand(allocator, "first 'unfinished"));
    try std.testing.expectError(error.NoEscapedCharacter, splitCommand(allocator, "first trailing\\"));
    try std.testing.expectError(error.NoEscapedCharacter, splitCommand(allocator, "\"trailing\\"));
}

test "timeouts honor adapter defaults and independent overrides" {
    var environment = std.process.Environ.Map.init(std.testing.allocator);
    defer environment.deinit();
    try std.testing.expectEqualDeep(Timeouts{ .wait_ms = 5000, .operation_ms = 5000, .compile_ms = 600_000 }, try timeouts(&environment, .wamr));
    try environment.put("WAMR_TESTSUITE_TIMEOUT", "2.5");
    try environment.put("WAMR_TESTSUITE_OP_TIMEOUT", "0.0011");
    try environment.put("WAMR_COMPILE_TIMEOUT", "120");
    try std.testing.expectEqualDeep(Timeouts{ .wait_ms = 2500, .operation_ms = 2, .compile_ms = 120_000 }, try timeouts(&environment, .wamr));
    try std.testing.expectEqualDeep(Timeouts{ .wait_ms = 5000, .operation_ms = 2, .compile_ms = 600_000 }, try timeouts(&environment, .wasmtime));
    try std.testing.expectEqual(@as(?u32, null), try parseTimeout("0"));
    try environment.put("WAMR_COMPILE_TIMEOUT", "1e100");
    try std.testing.expectError(error.TimeoutOutOfRange, timeouts(&environment, .wamr));
}
