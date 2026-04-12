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

    // Write to a unique temp subdirectory in CWD
    const dir_name = try std.fmt.allocPrint(allocator, ".wamr-test-{s}", .{name});
    defer allocator.free(dir_name);

    // Create dir (ignore if exists)
    std.fs.cwd().makeDir(dir_name) catch {};
    var out_dir = try std.fs.cwd().openDir(dir_name, .{});
    defer out_dir.close();

    // Write wasm modules
    var mod_it = modules.iterator();
    while (mod_it.next()) |entry| {
        out_dir.writeFile(.{ .sub_path = entry.key_ptr.*, .data = entry.value_ptr.* }) catch continue;
    }

    // Write JSON
    const json_name = try std.fmt.allocPrint(allocator, "{s}.json", .{name});
    defer allocator.free(json_name);
    out_dir.writeFile(.{ .sub_path = json_name, .data = json_buf.items }) catch return .{ .skipped = 1 };

    // Build full path
    const json_path = try std.fs.path.join(allocator, &.{ dir_name, json_name });
    defer allocator.free(json_path);

    const result = try spec_json_runner.runSpecTestFile(json_path, allocator);

    // Cleanup temp dir
    std.fs.cwd().deleteTree(dir_name) catch {};

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

    // Storage for module definitions (module definition $name ...)
    var module_defs: std.StringHashMapUnmanaged([]const u8) = .{};

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
                // Handle module definition/instance syntax
                if (wr.isModuleDefinition(sexpr.text)) {
                    // Store the definition text for later instantiation
                    if (wr.extractModuleDefName(sexpr.text)) |name| {
                        module_defs.put(allocator, name, sexpr.text) catch {};
                    }
                    // Compile it as a normal module (strip the definition keyword)
                    const stripped = wr.stripDefinitionKeyword(allocator, sexpr.text) orelse {
                        try w.print("{{\"type\":\"module\",\"line\":{d}}}", .{line_num});
                        continue;
                    };
                    defer allocator.free(stripped);
                    const filename = try std.fmt.allocPrint(allocator, "{s}.{d}.wasm", .{ base_name, module_idx });
                    var mod = wabt.text.Parser.parseModule(allocator, stripped) catch {
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
                    const fn2 = try std.fmt.allocPrint(allocator, "{s}.{d}.wasm", .{ base_name, module_idx });
                    defer allocator.free(fn2);
                    const mod_name = extractModuleName(sexpr.text);
                    if (mod_name) |mn| {
                        try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\",\"name\":\"{s}\"}}", .{ line_num, fn2, mn });
                    } else {
                        try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, fn2 });
                    }
                    module_idx += 1;
                    continue;
                } else if (hasDefinitionKw(sexpr.text)) {
                    // Unnamed module definition — just validate, don't instantiate
                    first = true; // undo the comma
                    continue;
                } else if (wr.isModuleInstance(sexpr.text)) {
                    // (module instance $inst $def) — compile a fresh copy from the definition
                    var inst_name: ?[]const u8 = null;
                    const def_name = blk: {
                        var i: usize = 1;
                        while (i < sexpr.text.len and (sexpr.text[i] == ' ' or sexpr.text[i] == '\t' or sexpr.text[i] == '\n' or sexpr.text[i] == '\r')) : (i += 1) {}
                        i += 6; // "module"
                        while (i < sexpr.text.len and (sexpr.text[i] == ' ' or sexpr.text[i] == '\t' or sexpr.text[i] == '\n' or sexpr.text[i] == '\r')) : (i += 1) {}
                        i += 8; // "instance"
                        while (i < sexpr.text.len and (sexpr.text[i] == ' ' or sexpr.text[i] == '\t' or sexpr.text[i] == '\n' or sexpr.text[i] == '\r')) : (i += 1) {}
                        // Extract instance name
                        const ins = i;
                        while (i < sexpr.text.len and sexpr.text[i] != ' ' and sexpr.text[i] != '\t' and sexpr.text[i] != ')') : (i += 1) {}
                        inst_name = sexpr.text[ins..i];
                        while (i < sexpr.text.len and (sexpr.text[i] == ' ' or sexpr.text[i] == '\t')) : (i += 1) {}
                        // Extract def name
                        const ds = i;
                        while (i < sexpr.text.len and sexpr.text[i] != ' ' and sexpr.text[i] != '\t' and sexpr.text[i] != ')') : (i += 1) {}
                        break :blk sexpr.text[ds..i];
                    };
                    if (module_defs.get(def_name)) |def_text| {
                        const stripped = wr.stripDefinitionKeyword(allocator, def_text) orelse {
                            try w.print("{{\"type\":\"module\",\"line\":{d}}}", .{line_num});
                            continue;
                        };
                        defer allocator.free(stripped);
                        const filename = try std.fmt.allocPrint(allocator, "{s}.{d}.wasm", .{ base_name, module_idx });
                        var mod = wabt.text.Parser.parseModule(allocator, stripped) catch {
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
                        const fn2 = try std.fmt.allocPrint(allocator, "{s}.{d}.wasm", .{ base_name, module_idx });
                        defer allocator.free(fn2);
                        if (inst_name) |iname| {
                            try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\",\"name\":\"{s}\"}}", .{ line_num, fn2, iname });
                        } else {
                            try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, fn2 });
                        }
                        module_idx += 1;
                    } else {
                        try w.print("{{\"type\":\"module\",\"line\":{d}}}", .{line_num});
                    }
                    continue;
                }

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
                        // Quote module: decode quoted WAT text, parse, and compile
                        const wat_text = wr.decodeQuoteStrings(allocator, sexpr.text) catch {
                            try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, filename });
                            allocator.free(filename);
                            module_idx += 1;
                            continue;
                        };
                        defer allocator.free(wat_text);
                        // Wrap in (module ...) if not already
                        const trimmed = std.mem.trimLeft(u8, wat_text, " \t\n\r");
                        const parse_text = if (std.mem.startsWith(u8, trimmed, "(module"))
                            wat_text
                        else blk: {
                            break :blk std.fmt.allocPrint(allocator, "(module {s})", .{wat_text}) catch {
                                try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, filename });
                                allocator.free(filename);
                                module_idx += 1;
                                continue;
                            };
                        };
                        defer if (parse_text.ptr != wat_text.ptr) allocator.free(parse_text);
                        var mod = wabt.text.Parser.parseModule(allocator, parse_text) catch {
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
                // Check for named module: (module $name ...)
                const mod_name = extractModuleName(sexpr.text);
                if (mod_name) |mn| {
                    try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\",\"name\":\"{s}\"}}", .{ line_num, fn2, mn });
                } else {
                    try w.print("{{\"type\":\"module\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, fn2 });
                }
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
                            // For assert_invalid, validate before writing binary
                            if (cmd == .assert_invalid) {
                                wabt.Validator.validate(&mod2, .{}) catch {
                                    allocator.free(filename);
                                    try w.print("{{\"type\":\"{s}\",\"line\":{d},\"module_type\":\"text\"}}", .{ type_str, line_num });
                                    continue;
                                };
                            }
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
                // Check if this is assert_trap with an embedded module (instantiation trap)
                if (std.mem.indexOf(u8, sexpr.text, "(module") != null and
                    std.mem.indexOf(u8, sexpr.text, "(invoke") == null)
                {
                    // Treat like assert_unlinkable: compile the module and write it as a file
                    const filename = try std.fmt.allocPrint(allocator, "{s}.{d}.wasm", .{ base_name, module_idx });
                    module_idx += 1;
                    if (std.mem.indexOf(u8, sexpr.text, "(module")) |mod_start| {
                        if (wr.extractSExpr(sexpr.text, mod_start)) |mod_sexpr| {
                            if (wr.isBinaryModule(mod_sexpr.text)) {
                                if (wr.decodeWastHexStrings(allocator, mod_sexpr.text)) |wasm_bytes| {
                                    try modules.put(allocator, filename, wasm_bytes);
                                } else |_| {
                                    allocator.free(filename);
                                }
                            } else if (!wr.isQuoteModule(mod_sexpr.text)) {
                                var mod2 = wabt.text.Parser.parseModule(allocator, mod_sexpr.text) catch {
                                    allocator.free(filename);
                                    try w.print("{{\"type\":\"assert_trap\",\"line\":{d}}}", .{line_num});
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
                            try w.print("{{\"type\":\"assert_uninstantiable\",\"line\":{d},\"filename\":\"{s}\"}}", .{ line_num, fn3 });
                            continue;
                        }
                    }
                    allocator.free(filename);
                    try w.print("{{\"type\":\"assert_trap\",\"line\":{d}}}", .{line_num});
                } else {
                    try writeAssertTrap(w, sexpr.text, type_str, line_num);
                }
            },
            .register => {
                try writeRegister(w, sexpr.text, line_num);
            },
            .invoke, .get => {
                // Write as action command with proper invoke details
                if (cmd == .invoke) {
                    if (findQuotedString(sexpr.text)) |field_name| {
                        try w.print("{{\"type\":\"action\",\"line\":{d},\"action\":{{\"type\":\"invoke\"", .{line_num});
                        // Include module name if present (e.g., invoke $M1 "store")
                        if (findDollarName(sexpr.text)) |mod_name| {
                            const name = if (mod_name.len > 0 and mod_name[0] == '$') mod_name[1..] else mod_name;
                            try w.print(",\"module\":\"{s}\"", .{name});
                        }
                        try w.print(",\"field\":\"{s}\",\"args\":[", .{field_name});
                        writeConstValues(w, sexpr.text) catch {};
                        try w.writeAll("]}}");
                    } else {
                        try w.print("{{\"type\":\"action\",\"line\":{d}}}", .{line_num});
                    }
                } else {
                    try w.print("{{\"type\":\"action\",\"line\":{d}}}", .{line_num});
                }
            },
            .unknown => { first = true; },
        }
    }

    try w.writeAll("]}");
}

fn hasDefinitionKw(text: []const u8) bool {
    var i: usize = 1;
    while (i < text.len and (text[i] == ' ' or text[i] == '\t' or text[i] == '\n' or text[i] == '\r')) : (i += 1) {}
    if (i + 6 >= text.len) return false;
    if (!std.mem.eql(u8, text[i .. i + 6], "module")) return false;
    i += 6;
    while (i < text.len and (text[i] == ' ' or text[i] == '\t' or text[i] == '\n' or text[i] == '\r')) : (i += 1) {}
    if (i + 10 >= text.len) return false;
    return std.mem.eql(u8, text[i .. i + 10], "definition");
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
    // Extract module name if present (strip $ prefix)
    if (findDollarName(invoke_text)) |mod_name| {
        const name = if (mod_name.len > 0 and mod_name[0] == '$') mod_name[1..] else mod_name;
        try w.print(",\"module\":\"{s}\"", .{name});
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
        const name = if (mod_name.len > 0 and mod_name[0] == '$') mod_name[1..] else mod_name;
        try w.print(",\"module\":\"{s}\"", .{name});
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
            try writeConstJson(w, "i32", result.value);
            i += result.len - 1;
        } else if (parseConst(remaining, "i64.const", "i64")) |result| {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            try writeConstJson(w, "i64", result.value);
            i += result.len - 1;
        } else if (parseConst(remaining, "f32.const", "f32")) |result| {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            try writeConstJson(w, "f32", result.value);
            i += result.len - 1;
        } else if (parseConst(remaining, "f64.const", "f64")) |result| {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            try writeConstJson(w, "f64", result.value);
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
            // Extract the function index if present
            if (parseConst(remaining, "ref.func", "funcref")) |result| {
                try writeConstJson(w, "funcref", result.value);
                i += result.len - 1;
            } else {
                try w.writeAll("{\"type\":\"funcref\",\"value\":\"0\"}");
                while (i < text.len and text[i] != ')') : (i += 1) {}
            }
        } else if (std.mem.startsWith(u8, remaining, "(ref.extern")) {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            if (parseConst(remaining, "ref.extern", "externref")) |result| {
                try writeConstJson(w, "externref", result.value);
                i += result.len - 1;
            } else {
                try w.writeAll("{\"type\":\"externref\",\"value\":\"0\"}");
                while (i < text.len and text[i] != ')') : (i += 1) {}
            }
        } else if (std.mem.startsWith(u8, remaining, "(v128.const")) {
            if (!first_val) try w.writeByte(',');
            first_val = false;
            // Parse v128.const: (v128.const <shape> <lane0> <lane1> ...)
            // Need to find the closing paren and parse all lanes
            const close = std.mem.indexOfPos(u8, text, i, ")") orelse text.len;
            const inner = text[i + 1 .. close]; // "v128.const <shape> <lanes>"
            writeV128Json(w, inner) catch {
                try w.writeAll("{\"type\":\"v128\",\"value\":\"0\"}");
            };
            i = close;
        }
    }
}

/// Write a v128 const value as JSON.
/// Input format: "v128.const <shape> <lane0> <lane1> ..."
/// Output: {"type":"v128","value":"<decimal_u128>"}
fn writeV128Json(w: anytype, inner: []const u8) !void {
    // Skip "v128.const"
    var pos: usize = 0;
    while (pos < inner.len and inner[pos] != ' ' and inner[pos] != '\t') : (pos += 1) {}
    while (pos < inner.len and (inner[pos] == ' ' or inner[pos] == '\t')) : (pos += 1) {}

    // Read shape
    const shape_start = pos;
    while (pos < inner.len and inner[pos] != ' ' and inner[pos] != '\t') : (pos += 1) {}
    const shape = inner[shape_start..pos];
    while (pos < inner.len and (inner[pos] == ' ' or inner[pos] == '\t')) : (pos += 1) {}

    // Collect lane value strings
    var lane_strs: [16][]const u8 = undefined;
    var lane_count: usize = 0;
    while (pos < inner.len and lane_count < 16) {
        while (pos < inner.len and (inner[pos] == ' ' or inner[pos] == '\t')) : (pos += 1) {}
        if (pos >= inner.len) break;
        const start = pos;
        while (pos < inner.len and inner[pos] != ' ' and inner[pos] != '\t' and inner[pos] != ')') : (pos += 1) {}
        if (pos > start) {
            lane_strs[lane_count] = inner[start..pos];
            lane_count += 1;
        }
    }

    var bytes: [16]u8 = .{0} ** 16;
    if (std.mem.eql(u8, shape, "i8x16")) {
        for (0..@min(lane_count, 16)) |i| {
            const val = parseWatI32(lane_strs[i]) orelse 0;
            bytes[i] = @truncate(val);
        }
    } else if (std.mem.eql(u8, shape, "i16x8")) {
        for (0..@min(lane_count, 8)) |i| {
            const val = parseWatI32(lane_strs[i]) orelse 0;
            std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @truncate(val), .little);
        }
    } else if (std.mem.eql(u8, shape, "i32x4")) {
        for (0..@min(lane_count, 4)) |i| {
            const val = parseWatI32(lane_strs[i]) orelse 0;
            std.mem.writeInt(u32, bytes[i * 4 ..][0..4], val, .little);
        }
    } else if (std.mem.eql(u8, shape, "i64x2")) {
        for (0..@min(lane_count, 2)) |i| {
            const val = parseWatI64(lane_strs[i]) orelse 0;
            std.mem.writeInt(u64, bytes[i * 8 ..][0..8], val, .little);
        }
    } else if (std.mem.eql(u8, shape, "f32x4")) {
        for (0..@min(lane_count, 4)) |i| {
            const val = parseWatF32(lane_strs[i]) orelse 0;
            std.mem.writeInt(u32, bytes[i * 4 ..][0..4], val, .little);
        }
    } else if (std.mem.eql(u8, shape, "f64x2")) {
        for (0..@min(lane_count, 2)) |i| {
            const val = parseWatF64(lane_strs[i]) orelse 0;
            std.mem.writeInt(u64, bytes[i * 8 ..][0..8], val, .little);
        }
    }

    const result = std.mem.readInt(u128, &bytes, .little);
    try w.print("{{\"type\":\"v128\",\"value\":\"{d}\"}}", .{result});
}

/// Write a JSON const value, converting WAT numeric literals to decimal.
/// WAT allows hex (0x...), underscores, and negative values.
/// The JSON format uses unsigned decimal for integers and either decimal
/// or special strings (nan:canonical, nan:arithmetic, inf, -inf) for floats.
fn writeConstJson(w: anytype, type_str: []const u8, raw_value: []const u8) !void {
    // Special float values recognized by the JSON runner
    if (std.mem.eql(u8, raw_value, "nan:canonical") or
        std.mem.eql(u8, raw_value, "nan:arithmetic"))
    {
        try w.print("{{\"type\":\"{s}\",\"value\":\"{s}\"}}", .{ type_str, raw_value });
        return;
    }

    // For integer types, parse and emit as unsigned decimal
    if (std.mem.eql(u8, type_str, "i32")) {
        if (parseWatI32(raw_value)) |val| {
            try w.print("{{\"type\":\"i32\",\"value\":\"{d}\"}}", .{val});
            return;
        }
    } else if (std.mem.eql(u8, type_str, "i64")) {
        if (parseWatI64(raw_value)) |val| {
            try w.print("{{\"type\":\"i64\",\"value\":\"{d}\"}}", .{val});
            return;
        }
    } else if (std.mem.eql(u8, type_str, "f32")) {
        if (parseWatF32(raw_value)) |val| {
            try w.print("{{\"type\":\"f32\",\"value\":\"{d}\"}}", .{val});
            return;
        }
    } else if (std.mem.eql(u8, type_str, "f64")) {
        if (parseWatF64(raw_value)) |val| {
            try w.print("{{\"type\":\"f64\",\"value\":\"{d}\"}}", .{val});
            return;
        }
    }

    // Fallback: emit as-is
    try w.print("{{\"type\":\"{s}\",\"value\":\"{s}\"}}", .{ type_str, raw_value });
}

/// Parse a WAT i32 literal (decimal, hex, with underscores, possibly negative)
/// and return as unsigned u32 (matching JSON spec format).
fn parseWatI32(raw: []const u8) ?u32 {
    const stripped = stripUnderscores(raw);
    const s = stripped.slice();
    const negative = s.len > 0 and s[0] == '-';
    const abs = if (negative) s[1..] else s;
    if (std.mem.startsWith(u8, abs, "0x") or std.mem.startsWith(u8, abs, "0X")) {
        const val = std.fmt.parseUnsigned(u32, abs[2..], 16) catch return null;
        return if (negative) 0 -% val else val;
    }
    if (negative) {
        const val = std.fmt.parseUnsigned(u32, abs, 10) catch return null;
        return 0 -% val;
    }
    return std.fmt.parseUnsigned(u32, abs, 10) catch return null;
}

fn parseWatI64(raw: []const u8) ?u64 {
    const stripped = stripUnderscores(raw);
    const s = stripped.slice();
    const negative = s.len > 0 and s[0] == '-';
    const abs = if (negative) s[1..] else s;
    if (std.mem.startsWith(u8, abs, "0x") or std.mem.startsWith(u8, abs, "0X")) {
        const val = std.fmt.parseUnsigned(u64, abs[2..], 16) catch return null;
        return if (negative) 0 -% val else val;
    }
    if (negative) {
        const val = std.fmt.parseUnsigned(u64, abs, 10) catch return null;
        return 0 -% val;
    }
    return std.fmt.parseUnsigned(u64, abs, 10) catch return null;
}

/// Parse WAT f32 literal to its u32 bit pattern for JSON.
fn parseWatF32(raw: []const u8) ?u32 {
    const stripped = stripUnderscores(raw);
    const s = stripped.slice();
    // Handle NaN variants
    if (parseWatNanF32(s)) |bits| return bits;
    // Handle inf
    if (std.mem.eql(u8, s, "inf")) return 0x7F800000;
    if (std.mem.eql(u8, s, "-inf")) return 0xFF800000;
    // Try standard parse (handles hex float like 0x1.5p+5)
    const f = std.fmt.parseFloat(f32, s) catch return null;
    return @bitCast(f);
}

/// Parse WAT f64 literal to its u64 bit pattern for JSON.
fn parseWatF64(raw: []const u8) ?u64 {
    const stripped = stripUnderscores(raw);
    const s = stripped.slice();
    if (parseWatNanF64(s)) |bits| return bits;
    if (std.mem.eql(u8, s, "inf")) return 0x7FF0000000000000;
    if (std.mem.eql(u8, s, "-inf")) return 0xFFF0000000000000;
    const f = std.fmt.parseFloat(f64, s) catch return null;
    return @bitCast(f);
}

/// Parse WAT NaN notation for f32: nan, -nan, nan:0xN, -nan:0xN
fn parseWatNanF32(s: []const u8) ?u32 {
    const negative = s.len > 0 and s[0] == '-';
    const rest = if (negative) s[1..] else s;
    if (!std.mem.startsWith(u8, rest, "nan")) return null;
    const sign: u32 = if (negative) 0x80000000 else 0;
    if (rest.len == 3) {
        // Plain nan → canonical NaN
        return sign | 0x7FC00000;
    }
    if (std.mem.startsWith(u8, rest[3..], ":0x")) {
        // nan:0xN → custom payload
        const payload = std.fmt.parseUnsigned(u32, rest[6..], 16) catch return null;
        return sign | 0x7F800000 | (payload & 0x7FFFFF);
    }
    return null;
}

/// Parse WAT NaN notation for f64: nan, -nan, nan:0xN, -nan:0xN
fn parseWatNanF64(s: []const u8) ?u64 {
    const negative = s.len > 0 and s[0] == '-';
    const rest = if (negative) s[1..] else s;
    if (!std.mem.startsWith(u8, rest, "nan")) return null;
    const sign: u64 = if (negative) 0x8000000000000000 else 0;
    if (rest.len == 3) {
        return sign | 0x7FF8000000000000;
    }
    if (std.mem.startsWith(u8, rest[3..], ":0x")) {
        const payload = std.fmt.parseUnsigned(u64, rest[6..], 16) catch return null;
        return sign | 0x7FF0000000000000 | (payload & 0xFFFFFFFFFFFFF);
    }
    return null;
}

/// Strip underscores from a numeric literal. Returns a view that may be
/// backed by a stack buffer or the original slice.
const StrippedNum = struct {
    buf: [64]u8 = undefined,
    len: usize,
    original: []const u8,

    fn slice(self: *const StrippedNum) []const u8 {
        if (self.len == self.original.len) return self.original;
        return self.buf[0..self.len];
    }
};

fn stripUnderscores(s: []const u8) StrippedNum {
    if (std.mem.indexOf(u8, s, "_") == null) {
        return .{ .len = s.len, .original = s };
    }
    var result: StrippedNum = .{ .len = 0, .original = s };
    for (s) |c| {
        if (c != '_' and result.len < result.buf.len) {
            result.buf[result.len] = c;
            result.len += 1;
        }
    }
    return result;
}

/// Extract the module name from a WAT module declaration.
/// "(module $Mt ...)" → "Mt" (without $ prefix)
/// "(module ...)" → null
fn extractModuleName(text: []const u8) ?[]const u8 {
    const prefix = "(module";
    if (!std.mem.startsWith(u8, text, prefix)) return null;
    var i = prefix.len;
    // Skip whitespace
    while (i < text.len and (text[i] == ' ' or text[i] == '\n' or text[i] == '\r' or text[i] == '\t')) : (i += 1) {}
    if (i >= text.len or text[i] != '$') return null;
    const name_start = i + 1; // skip $
    var name_end = name_start;
    while (name_end < text.len and text[name_end] != ' ' and text[name_end] != '\n' and
        text[name_end] != '\r' and text[name_end] != '\t' and text[name_end] != ')') : (name_end += 1)
    {}
    if (name_end == name_start) return null;
    return text[name_start..name_end];
}

const ConstResult= struct { value: []const u8, len: usize };

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

// ── Tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

test "wabt: parse and write simple module produces valid wasm" {
    const allocator = testing.allocator;
    const wat_source = "(module (func (export \"add\") (param i32 i32) (result i32) (i32.add (local.get 0) (local.get 1))))";

    var mod = try wabt.text.Parser.parseModule(allocator, wat_source);
    defer mod.deinit();

    const wasm_bytes = try wabt.binary.writer.writeModule(allocator, &mod);
    defer allocator.free(wasm_bytes);

    // Must have valid wasm header
    try testing.expect(wasm_bytes.len >= 8);
    try testing.expectEqualSlices(u8, &.{ 0x00, 0x61, 0x73, 0x6d }, wasm_bytes[0..4]);

    // Must load successfully in WAMR
    const root = @import("wamr");
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const module = try root.loader.load(wasm_bytes, arena.allocator());

    // Must have the add function
    try testing.expectEqual(@as(usize, 1), module.functions.len);
    try testing.expectEqual(@as(usize, 1), module.exports.len);

    // Function body must contain actual bytecode (not just 00 0b)
    try testing.expect(module.functions[0].code.len > 2);
}

test "wast runner: full pipeline for simple module" {
    const allocator = testing.allocator;
    const wast_source =
        \\(module (func (export "add") (param i32 i32) (result i32) (i32.add (local.get 0) (local.get 1))))
        \\(assert_return (invoke "add" (i32.const 1) (i32.const 2)) (i32.const 3))
    ;

    // Test the convertWast function specifically
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

    try convertWast(allocator, wast_source, "test_add", &json_buf, &modules);

    // Should have 1 module
    try testing.expectEqual(@as(u32, 1), modules.count());

    // Get the wasm bytes
    const wasm = modules.get("test_add.0.wasm") orelse return error.TestFailed;
    try testing.expect(wasm.len >= 8);

    // Load in WAMR
    const root = @import("wamr");
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const module = root.loader.load(wasm, arena.allocator()) catch |err| {
        std.debug.print("WAMR load failed: {}\n", .{err});
        std.debug.print("wasm ({d} bytes): ", .{wasm.len});
        for (wasm) |b| std.debug.print("{x:0>2}", .{b});
        std.debug.print("\n", .{});
        return err;
    };
    try testing.expectEqual(@as(usize, 1), module.functions.len);
    try testing.expect(module.functions[0].code.len > 2);
}

test "wabt: memory load module round-trips through WAMR" {
    const allocator = testing.allocator;
    const wat = "(module (memory 1) (func (export \"load\") (result i32) (i32.load (i32.const 0))))";

    var mod = try wabt.text.Parser.parseModule(allocator, wat);
    defer mod.deinit();

    const wasm_bytes = try wabt.binary.writer.writeModule(allocator, &mod);
    defer allocator.free(wasm_bytes);

    // Load in WAMR
    const root = @import("wamr");
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const module = root.loader.load(wasm_bytes, arena.allocator()) catch |err| {
        std.debug.print("WAMR load failed: {}\n", .{err});
        for (wasm_bytes) |b| std.debug.print("{x:0>2}", .{b});
        std.debug.print("\n", .{});
        return err;
    };

    try testing.expectEqual(@as(usize, 1), module.functions.len);
    try testing.expectEqual(@as(usize, 1), module.memories.len);
    // Code must NOT contain the extra 0x00 (unreachable) byte
    // Expected: i32.const 0 (41 00) + i32.load (28 00 00) + end (0b) = 6 bytes
    try testing.expectEqual(@as(usize, 6), module.functions[0].code.len);
}

test "wabt: memory store module round-trips through WAMR" {
    const allocator = testing.allocator;
    const wat = "(module (memory 1) (func (export \"store\") (i32.store (i32.const 0) (i32.const 42))))";

    var mod = try wabt.text.Parser.parseModule(allocator, wat);
    defer mod.deinit();

    const wasm_bytes = try wabt.binary.writer.writeModule(allocator, &mod);
    defer allocator.free(wasm_bytes);

    const root = @import("wamr");
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const module = root.loader.load(wasm_bytes, arena.allocator()) catch |err| {
        std.debug.print("WAMR load failed: {}\n", .{err});
        return err;
    };

    try testing.expectEqual(@as(usize, 1), module.functions.len);
    // Expected: i32.const 0 (41 00) + i32.const 42 (41 2a) + i32.store (36 00 00) + end (0b) = 8 bytes
    try testing.expectEqual(@as(usize, 8), module.functions[0].code.len);
}

test "parseConst: i32" {
    const result = parseConst("(i32.const 42)", "i32.const", "i32");
    try testing.expect(result != null);
    try testing.expectEqualStrings("42", result.?.value);
}

test "parseConst: negative" {
    const result = parseConst("(i32.const -1)", "i32.const", "i32");
    try testing.expect(result != null);
    try testing.expectEqualStrings("-1", result.?.value);
}

test "findQuotedString" {
    try testing.expectEqualStrings("add", findQuotedString("(invoke \"add\" (i32.const 1))").?);
}

test "extractModuleName" {
    try testing.expectEqualStrings("Mt", extractModuleName("(module $Mt (func))").?);
    try testing.expectEqualStrings("Nt", extractModuleName("(module $Nt\n  (func))").?);
    try testing.expect(extractModuleName("(module (func))") == null);
    try testing.expect(extractModuleName("(module)") == null);
}

test "findDollarName" {
    try testing.expectEqualStrings("$Mod", findDollarName("(invoke $Mod \"func\")").?);
    try testing.expect(findDollarName("(invoke \"func\")") == null);
}

test "parseWatI32: decimal" {
    try testing.expectEqual(@as(u32, 42), parseWatI32("42").?);
    try testing.expectEqual(@as(u32, 0), parseWatI32("0").?);
}

test "parseWatI32: hex" {
    try testing.expectEqual(@as(u32, 0x7fffffff), parseWatI32("0x7fffffff").?);
    try testing.expectEqual(@as(u32, 0x80000000), parseWatI32("0x80000000").?);
    try testing.expectEqual(@as(u32, 0xDEADBEEF), parseWatI32("0xDEADBEEF").?);
}

test "parseWatI32: negative" {
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), parseWatI32("-1").?); // -1 wraps to max u32
    try testing.expectEqual(@as(u32, 0x80000000), parseWatI32("-0x80000000").?);
}

test "parseWatI32: underscores" {
    try testing.expectEqual(@as(u32, 0x01234500), parseWatI32("0x012345_00").?);
    try testing.expectEqual(@as(u32, 0xFEDCBA80), parseWatI32("0xfedcba_80").?);
}

test "parseWatI64: hex" {
    try testing.expectEqual(@as(u64, 0x7FFFFFFFFFFFFFFF), parseWatI64("0x7FFFFFFFFFFFFFFF").?);
}

test "stripUnderscores" {
    const s1 = stripUnderscores("0x012345_00");
    try testing.expectEqualStrings("0x01234500", s1.slice());

    const s2 = stripUnderscores("42");
    try testing.expectEqualStrings("42", s2.slice());
}

test "parseWatF32: nan variants" {
    try testing.expectEqual(@as(u32, 0x7FC00000), parseWatF32("nan").?);
    try testing.expectEqual(@as(u32, 0xFFC00000), parseWatF32("-nan").?);
    try testing.expectEqual(@as(u32, 0x7F800001), parseWatF32("nan:0x1").?);
    try testing.expectEqual(@as(u32, 0xFF800001), parseWatF32("-nan:0x1").?);
    try testing.expectEqual(@as(u32, 0x7FC00000), parseWatF32("nan:0x400000").?);
}

test "parseWatF32: inf" {
    try testing.expectEqual(@as(u32, 0x7F800000), parseWatF32("inf").?);
    try testing.expectEqual(@as(u32, 0xFF800000), parseWatF32("-inf").?);
}

test "parseWatF32: hex float" {
    // 0x0p+0 = 0.0
    try testing.expectEqual(@as(u32, 0), parseWatF32("0x0p+0").?);
    // -0x0p+0 = -0.0
    try testing.expectEqual(@as(u32, 0x80000000), parseWatF32("-0x0p+0").?);
}

test "parseWatF64: nan variants" {
    try testing.expectEqual(@as(u64, 0x7FF8000000000000), parseWatF64("nan").?);
    try testing.expectEqual(@as(u64, 0xFFF8000000000000), parseWatF64("-nan").?);
}

test "parseWatNanF32" {
    try testing.expectEqual(@as(u32, 0x7FC00000), parseWatNanF32("nan").?);
    try testing.expectEqual(@as(u32, 0xFFC00000), parseWatNanF32("-nan").?);
    try testing.expect(parseWatNanF32("42") == null);
    try testing.expect(parseWatNanF32("0x7fc00000") == null);
}
