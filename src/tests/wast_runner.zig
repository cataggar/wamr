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
            .assert_return => {
                try writeAssertReturn(w, sexpr.text, line_num);
            },
            .assert_trap, .assert_exhaustion, .assert_exception => {
                const type_str = switch (cmd) {
                    .assert_trap => "assert_trap",
                    .assert_exhaustion => "assert_exhaustion",
                    .assert_exception => "assert_trap",
                    else => unreachable,
                };
                try writeAssertTrap(w, sexpr.text, type_str, line_num);
            },
            .register => {
                try writeRegister(w, sexpr.text, line_num);
            },
            .invoke, .get => {
                try w.print("{{\"type\":\"action\",\"line\":{d}}}", .{line_num});
            },
            .unknown => { first = true; },
        }
    }

    try w.writeAll("]}");
}

// ── S-expression parsers for assert commands ─────────────────────────────

fn writeAssertReturn(w: anytype, text: []const u8, line: u32) !void {
    // (assert_return (invoke "name" args...) expected...)
    // (assert_return (invoke $Mod "name" args...) expected...)
    // (assert_return (get "name") expected...)
    const wr = wabt.wast_runner;
    try w.print("{{\"type\":\"assert_return\",\"line\":{d}", .{line});

    // Find the action (invoke or get)
    if (std.mem.indexOf(u8, text, "(invoke")) |inv_start| {
        if (wr.extractSExpr(text, inv_start)) |inv_sexpr| {
            try w.writeAll(",\"action\":{\"type\":\"invoke\"");
            try writeInvokeFields(w, inv_sexpr.text);
            try w.writeByte('}');
            // Expected values come after the invoke sexpr
            const after_invoke = inv_start + inv_sexpr.text.len;
            try writeExpected(w, text[after_invoke..]);
        }
    } else if (std.mem.indexOf(u8, text, "(get")) |get_start| {
        if (wr.extractSExpr(text, get_start)) |get_sexpr| {
            try w.writeAll(",\"action\":{\"type\":\"get\"");
            try writeGetFields(w, get_sexpr.text);
            try w.writeByte('}');
            const after_get = get_start + get_sexpr.text.len;
            try writeExpected(w, text[after_get..]);
        }
    }
    try w.writeByte('}');
}

fn writeAssertTrap(w: anytype, text: []const u8, type_str: []const u8, line: u32) !void {
    // (assert_trap (invoke "name" args...) "trap message")
    const wr = wabt.wast_runner;
    try w.print("{{\"type\":\"{s}\",\"line\":{d}", .{ type_str, line });

    if (std.mem.indexOf(u8, text, "(invoke")) |inv_start| {
        if (wr.extractSExpr(text, inv_start)) |inv_sexpr| {
            try w.writeAll(",\"action\":{\"type\":\"invoke\"");
            try writeInvokeFields(w, inv_sexpr.text);
            try w.writeByte('}');
            // Extract trap message (last quoted string after invoke)
            const after = inv_start + inv_sexpr.text.len;
            if (findQuotedString(text[after..])) |msg| {
                try w.print(",\"text\":\"{s}\"", .{msg});
            }
        }
    } else if (std.mem.indexOf(u8, text, "(module")) |_| {
        // assert_trap with module (uninstantiable)
        if (findLastQuotedString(text)) |msg| {
            try w.print(",\"text\":\"{s}\"", .{msg});
        }
    }
    try w.writeByte('}');
}

fn writeRegister(w: anytype, text: []const u8, line: u32) !void {
    // (register "name") or (register "name" $mod)
    try w.print("{{\"type\":\"register\",\"line\":{d}", .{line});
    if (findQuotedString(text)) |name| {
        try w.print(",\"as\":\"{s}\"", .{name});
    }
    // Check for $name after the quoted string
    if (findDollarName(text)) |mod_name| {
        try w.print(",\"name\":\"{s}\"", .{mod_name});
    }
    try w.writeByte('}');
}

fn writeInvokeFields(w: anytype, invoke_text: []const u8) !void {
    // (invoke "name" args...) or (invoke $Mod "name" args...)
    // Extract module name if present
    if (findDollarName(invoke_text)) |mod_name| {
        try w.print(",\"module\":\"{s}\"", .{mod_name});
    }
    // Extract function name
    if (findQuotedString(invoke_text)) |name| {
        try w.print(",\"field\":\"{s}\"", .{name});
    }
    // Extract args
    try w.writeAll(",\"args\":[");
    try writeConstValues(w, invoke_text);
    try w.writeByte(']');
}

fn writeGetFields(w: anytype, get_text: []const u8) !void {
    if (findDollarName(get_text)) |mod_name| {
        try w.print(",\"module\":\"{s}\"", .{mod_name});
    }
    if (findQuotedString(get_text)) |name| {
        try w.print(",\"field\":\"{s}\"", .{name});
    }
}

fn writeExpected(w: anytype, text: []const u8) !void {
    try w.writeAll(",\"expected\":[");
    try writeConstValues(w, text);
    try w.writeByte(']');
}

fn writeConstValues(w: anytype, text: []const u8) !void {
    // Find all (i32.const N), (i64.const N), (f32.const N), (f64.const N), (ref.null ...), (ref.func ...)
    var first_val = true;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] != '(') continue;
        // Check for const patterns
        const remaining = text[i..];
        if (parseConst(remaining, "i32.const", "i32")) |result| {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            try w.print("{{\"type\":\"i32\",\"value\":\"{s}\"}}", .{result.value});
            i += result.len - 1;
        } else if (parseConst(remaining, "i64.const", "i64")) |result| {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            try w.print("{{\"type\":\"i64\",\"value\":\"{s}\"}}", .{result.value});
            i += result.len - 1;
        } else if (parseConst(remaining, "f32.const", "f32")) |result| {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            try w.print("{{\"type\":\"f32\",\"value\":\"{s}\"}}", .{result.value});
            i += result.len - 1;
        } else if (parseConst(remaining, "f64.const", "f64")) |result| {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            try w.print("{{\"type\":\"f64\",\"value\":\"{s}\"}}", .{result.value});
            i += result.len - 1;
        } else if (std.mem.startsWith(u8, remaining, "(ref.null")) {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            if (std.mem.indexOf(u8, remaining, "extern")) |_| {
                try w.writeAll("{\"type\":\"externref\",\"value\":\"null\"}");
            } else {
                try w.writeAll("{\"type\":\"funcref\",\"value\":\"null\"}");
            }
            // Skip to closing paren
            while (i < text.len and text[i] != ')') : (i += 1) {}
        } else if (std.mem.startsWith(u8, remaining, "(ref.func")) {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            try w.writeAll("{\"type\":\"funcref\",\"value\":\"0\"}");
            while (i < text.len and text[i] != ')') : (i += 1) {}
        }
    }
}

const ConstResult = struct { value: []const u8, len: usize };

fn parseConst(text: []const u8, prefix: []const u8, _: []const u8) ?ConstResult {
    // Match "(i32.const VALUE)"
    if (!std.mem.startsWith(u8, text, "(")) return null;
    if (text.len < prefix.len + 3) return null; // minimum: "(prefix V)"
    if (!std.mem.startsWith(u8, text[1..], prefix)) return null;
    const after_prefix = 1 + prefix.len;
    if (after_prefix >= text.len or text[after_prefix] != ' ') return null;
    var val_start = after_prefix + 1;
    while (val_start < text.len and text[val_start] == ' ') : (val_start += 1) {}
    var val_end = val_start;
    while (val_end < text.len and text[val_end] != ')' and text[val_end] != ' ') : (val_end += 1) {}
    if (val_end >= text.len) return null;
    const close = std.mem.indexOfScalarPos(u8, text, val_end, ')') orelse return null;
    return .{ .value = text[val_start..val_end], .len = close + 1 };
}

fn findQuotedString(text: []const u8) ?[]const u8 {
    const start = (std.mem.indexOf(u8, text, "\"") orelse return null) + 1;
    var end = start;
    while (end < text.len and text[end] != '"') : (end += 1) {
        if (text[end] == '\\' and end + 1 < text.len) end += 1;
    }
    return text[start..end];
}

fn findLastQuotedString(text: []const u8) ?[]const u8 {
    var last: ?[]const u8 = null;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] == '"') {
            const start = i + 1;
            i += 1;
            while (i < text.len and text[i] != '"') : (i += 1) {
                if (text[i] == '\\' and i + 1 < text.len) i += 1;
            }
            last = text[start..i];
        }
    }
    return last;
}

fn findDollarName(text: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] == '$') {
            const start = i;
            i += 1;
            while (i < text.len and text[i] != ' ' and text[i] != ')' and text[i] != '\n') : (i += 1) {}
            return text[start..i];
        }
        if (text[i] == '"') { // skip quoted strings
            i += 1;
            while (i < text.len and text[i] != '"') : (i += 1) {
                if (text[i] == '\\' and i + 1 < text.len) i += 1;
            }
        }
    }
    return null;
}
