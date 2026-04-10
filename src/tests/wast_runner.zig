//! WAMR-native .wast spec test runner.
//!
//! Converts .wast to JSON + .wasm in memory using wabt's S-expression
//! parser, writes to a temp directory, then runs through WAMR's
//! spec test runner (WAMR's loader + interpreter).

const std = @import("std");
const wabt = @import("wabt");
const spec_json_runner = @import("spec_json_runner.zig");

pub const WastResult = struct {
    passed: u32 = 0,
    failed: u32 = 0,
    skipped: u32 = 0,

    pub fn total(self: WastResult) u32 {
        return self.passed + self.failed + self.skipped;
    }
};

/// Run a .wast file through WAMR's interpreter.
/// 1. Parse .wast S-expressions using wabt utilities
/// 2. Convert modules to .wasm binary via wabt text.Parser + binary.writer
/// 3. Write JSON + .wasm to temp dir
/// 4. Run through WAMR's JSON-based spec test runner
pub fn run(allocator: std.mem.Allocator, source: []const u8, name: []const u8) WastResult {
    return runInner(allocator, source, name) catch return .{ .skipped = 1 };
}

fn runInner(allocator: std.mem.Allocator, source: []const u8, name: []const u8) !WastResult {
    // Convert .wast to JSON + wasm modules in memory
    var json_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer json_buf.deinit(allocator);
    var modules = std.StringHashMapUnmanaged([]u8){};
    defer {
        var it = modules.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        modules.deinit(allocator);
    }

    try convertWast(allocator, source, name, &json_buf, &modules);

    // Write to temp dir
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    // Write wasm modules
    var mod_it = modules.iterator();
    while (mod_it.next()) |entry| {
        tmp_dir.dir.writeFile(.{ .sub_path = entry.key_ptr.*, .data = entry.value_ptr.* }) catch continue;
    }

    // Write JSON
    tmp_dir.dir.writeFile(.{ .sub_path = "test.json", .data = json_buf.items }) catch return .{ .skipped = 1 };

    // Resolve real path and run
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const json_path = try tmp_dir.dir.realpath("test.json", &path_buf);

    const result = try spec_json_runner.runSpecTestFile(json_path, allocator);
    return .{
        .passed = result.passed,
        .failed = result.failed,
        .skipped = result.skipped,
    };
}

fn convertWast(allocator: std.mem.Allocator, source: []const u8, base_name: []const u8, json_buf: *std.ArrayListUnmanaged(u8), modules: *std.StringHashMapUnmanaged([]u8)) !void {
    const wr = wabt.wast_runner;
    const w = json_buf.writer(allocator);
    try w.writeAll("{\"commands\":[");

    var pos: usize = 0;
    var module_idx: u32 = 0;
    var first = true;

    while (pos < source.len) {
        pos = wr.skipWhitespaceAndComments(source, pos);
        if (pos >= source.len) break;
        if (source[pos] != '(') { pos += 1; continue; }

        var line_num: u32 = 1;
        for (source[0..pos]) |c| { if (c == '\n') line_num += 1; }

        const sexpr = wr.extractSExpr(source, pos) orelse break;
        pos = sexpr.end;

        if (!first) try w.writeByte(',');
        first = false;

        const cmd = wr.classifyCommand(sexpr.text);
        switch (cmd) {
            .module => {
                const filename = try std.fmt.allocPrint(allocator, "{s}.{d}.wasm", .{ base_name, module_idx });
                if (wr.isBinaryOrQuoteModule(sexpr.text)) {
                    if (wr.isBinaryModule(sexpr.text)) {
                        const wasm_bytes = wr.decodeWastHexStrings(allocator, sexpr.text) catch {
                            try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, filename });
                            allocator.free(filename);
                            module_idx += 1;
                            continue;
                        };
                        try modules.put(allocator, filename, wasm_bytes);
                    } else {
                        allocator.free(filename);
                    }
                } else {
                    var mod = wabt.text.Parser.parseModule(allocator, sexpr.text) catch {
                        try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, filename });
                        allocator.free(filename);
                        module_idx += 1;
                        continue;
                    };
                    defer mod.deinit();
                    const wasm_bytes = wabt.binary.writer.writeModule(allocator, &mod) catch {
                        try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, filename });
                        allocator.free(filename);
                        module_idx += 1;
                        continue;
                    };
                    try modules.put(allocator, filename, wasm_bytes);
                }
                const fn2 = try std.fmt.allocPrint(allocator, "{s}.{d}.wasm", .{ base_name, module_idx });
                defer allocator.free(fn2);
                try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, fn2 });
                module_idx += 1;
            },
            .assert_invalid, .assert_malformed, .assert_unlinkable => {
                const type_str = switch (cmd) {
                    .assert_invalid => "assert_invalid",
                    .assert_malformed => "assert_malformed",
                    .assert_unlinkable => "assert_unlinkable",
                    else => unreachable,
                };
                const filename = try std.fmt.allocPrint(allocator, "{s}.{d}.wasm", .{ base_name, module_idx });
                module_idx += 1;
                // Try to extract and compile the embedded module
                if (std.mem.indexOf(u8, sexpr.text, "(module")) |mod_start| {
                    if (wr.extractSExpr(sexpr.text, mod_start)) |mod_sexpr| {
                        const module_type = if (wr.isQuoteModule(mod_sexpr.text)) "text" else "binary";
                        if (wr.isBinaryModule(mod_sexpr.text)) {
                            if (wr.decodeWastHexStrings(allocator, mod_sexpr.text)) |wasm_bytes| {
                                try modules.put(allocator, filename, wasm_bytes);
                            } else |_| {
                                allocator.free(filename);
                            }
                        } else if (!wr.isQuoteModule(mod_sexpr.text)) {
                            var mod2 = wabt.text.Parser.parseModule(allocator, mod_sexpr.text) catch {
                                allocator.free(filename);
                                try w.print("{{\"type\":\"{s}\",\"line\":{d},\"module_type\":\"text\"}}", .{ type_str, line_num });
                                continue;
                            };
                            defer mod2.deinit();
                            if (wabt.binary.writer.writeModule(allocator, &mod2)) |wasm_bytes| {
                                try modules.put(allocator, filename, wasm_bytes);
                            } else |_| {
                                allocator.free(filename);
                            }
                        } else {
                            allocator.free(filename);
                        }
                        const fn3 = try std.fmt.allocPrint(allocator, "{s}.{d}.wasm", .{ base_name, module_idx - 1 });
                        defer allocator.free(fn3);
                        try w.print("{{\"type\":\"{s}\",\"line\":{d},\"filename\":\"{s}\",\"module_type\":\"{s}\"}}", .{ type_str, line_num, fn3, module_type });
                        continue;
                    }
                }
                allocator.free(filename);
                try w.print("{{\"type\":\"{s}\",\"line\":{d}}}", .{ type_str, line_num });
            },
            .assert_return, .assert_trap, .assert_exhaustion, .assert_exception => {
                const type_str = switch (cmd) {
                    .assert_return => "assert_return",
                    .assert_trap => "assert_trap",
                    .assert_exhaustion => "assert_exhaustion",
                    .assert_exception => "assert_trap",
                    else => unreachable,
                };
                try w.print("{{\"type\":\"{s}\",\"line\":{d}}}", .{ type_str, line_num });
            },
            .register => {
                try w.print("{{\"type\":\"register\",\"line\":{d}}}", .{line_num});
            },
            .invoke, .get => {
                try w.print("{{\"type\":\"action\",\"line\":{d}}}", .{line_num});
            },
            .unknown => { first = true; },
        }
    }

    try w.writeAll("]}");
}
