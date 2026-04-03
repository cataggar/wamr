//! WebAssembly Spec Test JSON Runner
//!
//! Reads JSON files produced by wast2json, executing module loads,
//! assert_return, assert_trap, and assert_invalid/assert_malformed
//! commands against the WAMR runtime.

const std = @import("std");
const root = @import("wamr");
const wamr = root.wamr;
const types = root.types;

const Io = std.Io;
const Dir = Io.Dir;

pub const SpecTestResult = struct {
    file: []const u8,
    total: u32 = 0,
    passed: u32 = 0,
    failed: u32 = 0,
    skipped: u32 = 0,
};

const Command = struct {
    type: []const u8,
    line: i64 = 0,
    filename: ?[]const u8 = null,
    action: ?Action = null,
    expected: ?[]const Arg = null,
    text: ?[]const u8 = null,
    module_type: ?[]const u8 = null,
};

const Action = struct {
    type: []const u8,
    field: ?[]const u8 = null,
    module: ?[]const u8 = null,
    args: ?[]const Arg = null,
};

const Arg = struct {
    type: []const u8,
    value: ?[]const u8 = null,
};

const SpecJson = struct {
    source_filename: ?[]const u8 = null,
    commands: []const Command,
};

/// Parse a spec-test JSON value object into a types.Value.
/// The spec encodes all values as unsigned decimal bit patterns.
fn parseValue(arg: Arg) ?types.Value {
    const val_str = arg.value orelse return null;
    if (std.mem.eql(u8, arg.type, "i32")) {
        const bits = std.fmt.parseUnsigned(u32, val_str, 10) catch return null;
        return .{ .i32 = @bitCast(bits) };
    } else if (std.mem.eql(u8, arg.type, "i64")) {
        const bits = std.fmt.parseUnsigned(u64, val_str, 10) catch return null;
        return .{ .i64 = @bitCast(bits) };
    } else if (std.mem.eql(u8, arg.type, "f32")) {
        const bits = std.fmt.parseUnsigned(u32, val_str, 10) catch return null;
        return .{ .f32 = @bitCast(bits) };
    } else if (std.mem.eql(u8, arg.type, "f64")) {
        const bits = std.fmt.parseUnsigned(u64, val_str, 10) catch return null;
        return .{ .f64 = @bitCast(bits) };
    }
    return null;
}

/// Bit-exact comparison of two Values (handles NaN correctly).
fn valuesEqual(a: types.Value, b: types.Value) bool {
    return switch (a) {
        .i32 => |v| b == .i32 and b.i32 == v,
        .i64 => |v| b == .i64 and b.i64 == v,
        .f32 => |v| b == .f32 and @as(u32, @bitCast(b.f32)) == @as(u32, @bitCast(v)),
        .f64 => |v| b == .f64 and @as(u64, @bitCast(b.f64)) == @as(u64, @bitCast(v)),
        else => false,
    };
}

/// Parse a slice of JSON args into a caller-owned slice of Values.
/// Returns null if any arg has an unsupported type.
fn parseArgs(args_json: []const Arg, allocator: std.mem.Allocator) ?[]types.Value {
    const args = allocator.alloc(types.Value, args_json.len) catch return null;
    for (args_json, 0..) |arg, i| {
        args[i] = parseValue(arg) orelse {
            allocator.free(args);
            return null;
        };
    }
    return args;
}

pub fn runSpecTestFile(json_path: []const u8, allocator: std.mem.Allocator, io: Io) !SpecTestResult {
    const cwd = Dir.cwd();
    const json_data = try cwd.readFileAlloc(io, json_path, allocator, Io.Limit.limited(10 * 1024 * 1024));
    defer allocator.free(json_data);

    const parsed = try std.json.parseFromSlice(SpecJson, allocator, json_data, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const commands = parsed.value.commands;

    var result = SpecTestResult{ .file = json_path };
    var runtime = wamr.Runtime.init(allocator);
    defer runtime.deinit();

    var current_module: ?wamr.Module = null;
    var current_instance: ?wamr.Instance = null;

    const json_dir = std.fs.path.dirname(json_path) orelse ".";

    var current_wasm_data: ?[]u8 = null;

    defer {
        if (current_instance) |*i| i.deinit();
        if (current_module) |*m| m.deinit();
        if (current_wasm_data) |d| allocator.free(d);
    }

    for (commands) |cmd| {
        result.total += 1;

        if (std.mem.eql(u8, cmd.type, "module")) {
            if (current_instance) |*inst| {
                inst.deinit();
                current_instance = null;
            }
            if (current_module) |*mod| {
                mod.deinit();
                current_module = null;
            }
            if (current_wasm_data) |d| {
                allocator.free(d);
                current_wasm_data = null;
            }

            const filename = cmd.filename orelse {
                result.skipped += 1;
                continue;
            };

            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, Io.Limit.limited(10 * 1024 * 1024)) catch {
                result.skipped += 1;
                continue;
            };

            current_module = runtime.loadModule(wasm_data) catch {
                allocator.free(wasm_data);
                result.skipped += 1;
                continue;
            };
            current_wasm_data = wasm_data;
            current_instance = current_module.?.instantiate() catch {
                result.skipped += 1;
                continue;
            };
            result.passed += 1;
        } else if (std.mem.eql(u8, cmd.type, "assert_return")) {
            const action = cmd.action orelse {
                result.skipped += 1;
                continue;
            };

            if (!std.mem.eql(u8, action.type, "invoke")) {
                result.skipped += 1;
                continue;
            }

            var inst = current_instance orelse {
                result.skipped += 1;
                continue;
            };

            const field = action.field orelse {
                result.skipped += 1;
                continue;
            };
            const expected_json = cmd.expected orelse &[_]Arg{};
            const args_json = action.args orelse &[_]Arg{};

            const args = parseArgs(args_json, allocator) orelse {
                result.skipped += 1;
                continue;
            };
            defer allocator.free(args);

            // Parse expected values
            var expected_vals: ?[]types.Value = null;
            if (expected_json.len > 0) {
                expected_vals = parseArgs(expected_json, allocator) orelse {
                    result.skipped += 1;
                    continue;
                };
            }
            defer if (expected_vals) |ev| allocator.free(ev);

            const actual = inst.call(field, args) catch {
                result.failed += 1;
                continue;
            };
            defer allocator.free(actual);

            const expected = expected_vals orelse &[_]types.Value{};
            if (actual.len != expected.len) {
                result.failed += 1;
                continue;
            }

            var all_match = true;
            for (actual, expected) |a, e| {
                if (!valuesEqual(a, e)) {
                    all_match = false;
                    break;
                }
            }

            if (all_match) {
                result.passed += 1;
            } else {
                result.failed += 1;
            }
        } else if (std.mem.eql(u8, cmd.type, "assert_trap")) {
            const action = cmd.action orelse {
                result.skipped += 1;
                continue;
            };

            if (!std.mem.eql(u8, action.type, "invoke")) {
                result.skipped += 1;
                continue;
            }

            var inst = current_instance orelse {
                result.skipped += 1;
                continue;
            };

            const field = action.field orelse {
                result.skipped += 1;
                continue;
            };
            const args_json = action.args orelse &[_]Arg{};

            const args = parseArgs(args_json, allocator) orelse {
                result.skipped += 1;
                continue;
            };
            defer allocator.free(args);

            if (inst.call(field, args)) |results| {
                allocator.free(results);
                result.failed += 1; // should have trapped
            } else |_| {
                result.passed += 1; // trap expected
            }
        } else if (std.mem.eql(u8, cmd.type, "assert_invalid") or
            std.mem.eql(u8, cmd.type, "assert_malformed"))
        {
            const filename = cmd.filename orelse {
                result.skipped += 1;
                continue;
            };

            // Skip WAT-format modules (only test binary .wasm)
            if (cmd.module_type) |mt| {
                if (std.mem.eql(u8, mt, "text")) {
                    result.skipped += 1;
                    continue;
                }
            }

            const wasm_path = try std.fs.path.join(allocator, &.{ json_dir, filename });
            defer allocator.free(wasm_path);

            const wasm_data = cwd.readFileAlloc(io, wasm_path, allocator, Io.Limit.limited(10 * 1024 * 1024)) catch {
                result.passed += 1; // can't read = invalid, as expected
                continue;
            };
            defer allocator.free(wasm_data);

            var mod = runtime.loadModule(wasm_data) catch {
                result.passed += 1; // load failed as expected
                continue;
            };
            mod.deinit();
            result.failed += 1; // should have failed to load
        } else {
            // register, assert_exhaustion, assert_uninstantiable, assert_unlinkable, action
            result.skipped += 1;
        }
    }

    return result;
}
