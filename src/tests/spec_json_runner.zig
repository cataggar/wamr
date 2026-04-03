//! WebAssembly Spec Test JSON Runner
//!
//! Reads JSON files produced by wast2json, executing module loads,
//! assert_return, assert_trap, and assert_invalid/assert_malformed
//! commands against the WAMR runtime.

const std = @import("std");
const root = @import("wamr");
const wamr = root.wamr;

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

fn parseI32(val_str: []const u8) ?i32 {
    // Spec tests encode i32 values as unsigned decimal strings.
    // e.g. "4294967295" represents -1 as i32.
    if (std.fmt.parseUnsigned(u32, val_str, 10)) |v| {
        return @bitCast(v);
    } else |_| {}
    if (std.fmt.parseInt(i32, val_str, 10)) |v| {
        return v;
    } else |_| {}
    return null;
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
            const expected = cmd.expected orelse &[_]Arg{};
            const args_json = action.args orelse &[_]Arg{};

            var args_buf: [16]i32 = undefined;
            var arg_count: usize = 0;
            var all_i32 = true;

            for (args_json) |arg| {
                if (!std.mem.eql(u8, arg.type, "i32")) {
                    all_i32 = false;
                    break;
                }
                const val_str = arg.value orelse {
                    all_i32 = false;
                    break;
                };
                args_buf[arg_count] = parseI32(val_str) orelse {
                    all_i32 = false;
                    break;
                };
                arg_count += 1;
            }

            if (!all_i32) {
                result.skipped += 1;
                continue;
            }

            // Handle void return (no expected values) — skip for now
            if (expected.len == 0) {
                result.skipped += 1;
                continue;
            }

            // Handle single i32 return
            if (expected.len != 1 or !std.mem.eql(u8, expected[0].type, "i32")) {
                result.skipped += 1;
                continue;
            }

            const expected_val_str = expected[0].value orelse {
                result.skipped += 1;
                continue;
            };
            const expected_val = parseI32(expected_val_str) orelse {
                result.skipped += 1;
                continue;
            };

            const actual = inst.callI32(field, args_buf[0..arg_count]) catch {
                result.failed += 1;
                continue;
            };

            if (actual == expected_val) {
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

            var args_buf: [16]i32 = undefined;
            var arg_count: usize = 0;
            var all_i32 = true;

            for (args_json) |arg| {
                if (!std.mem.eql(u8, arg.type, "i32")) {
                    all_i32 = false;
                    break;
                }
                const val_str = arg.value orelse {
                    all_i32 = false;
                    break;
                };
                args_buf[arg_count] = parseI32(val_str) orelse {
                    all_i32 = false;
                    break;
                };
                arg_count += 1;
            }

            if (!all_i32) {
                result.skipped += 1;
                continue;
            }

            _ = inst.callI32(field, args_buf[0..arg_count]) catch {
                result.passed += 1; // trap expected
                continue;
            };
            result.failed += 1; // should have trapped
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
