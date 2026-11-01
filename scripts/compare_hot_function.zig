const std = @import("std");
const builtin = @import("builtin");
const json = std.json;
const Value = json.Value;
const ObjectMap = json.ObjectMap;
const Array = json.Array;

pub const CONFIG_SCHEMA_VERSION: i64 = 1;
pub const CAPTURE_SCHEMA_VERSION: i64 = 1;
pub const REPORT_SCHEMA_VERSION: i64 = 1;
pub const AOT_MAGIC: u32 = 0x746F6100;
pub const AOT_VERSION: u32 = 11;
pub const AOT_TEXT_SECTION: u32 = 2;
pub const AOT_FUNCTION_SECTION: u32 = 3;

pub const METRIC_ORDER = [_][]const u8{
    "native_instructions",
    "code_size_bytes",
    "frame_size_bytes",
    "frame_loads",
    "frame_stores",
    "reg_reg_moves",
    "address_generation",
    "branches",
    "indirect_dispatch",
    "calls",
};

pub fn metricLabel(metric: []const u8) []const u8 {
    if (std.mem.eql(u8, metric, "native_instructions")) return "Native instructions";
    if (std.mem.eql(u8, metric, "code_size_bytes")) return "Code size";
    if (std.mem.eql(u8, metric, "frame_size_bytes")) return "Fixed frame size";
    if (std.mem.eql(u8, metric, "frame_loads")) return "Frame-load instructions";
    if (std.mem.eql(u8, metric, "frame_stores")) return "Frame-store instructions";
    if (std.mem.eql(u8, metric, "reg_reg_moves")) return "Register-to-register moves";
    if (std.mem.eql(u8, metric, "address_generation")) return "Address-generation instructions";
    if (std.mem.eql(u8, metric, "branches")) return "Branch instructions";
    if (std.mem.eql(u8, metric, "indirect_dispatch")) return "Indirect dispatch instructions";
    if (std.mem.eql(u8, metric, "calls")) return "Call instructions";
    return metric;
}

pub fn metricUnit(metric: []const u8) []const u8 {
    if (std.mem.endsWith(u8, metric, "_bytes")) return "bytes";
    return "instructions";
}

pub fn isGateMetric(metric: []const u8) bool {
    for ([_][]const u8{
        "frame_loads",
        "frame_stores",
        "reg_reg_moves",
        "address_generation",
        "branches",
        "indirect_dispatch",
        "calls",
    }) |m| {
        if (std.mem.eql(u8, metric, m)) return true;
    }
    return false;
}

pub fn leverForMetric(metric: []const u8) []const u8 {
    if (std.mem.eql(u8, metric, "frame_loads") or std.mem.eql(u8, metric, "frame_stores"))
        return "register-pressure / frame-access reduction";
    if (std.mem.eql(u8, metric, "reg_reg_moves"))
        return "move coalescing";
    if (std.mem.eql(u8, metric, "address_generation"))
        return "address-generation simplification";
    if (std.mem.eql(u8, metric, "branches"))
        return "branch / bounds-check layout";
    if (std.mem.eql(u8, metric, "indirect_dispatch"))
        return "indirect-dispatch tightening";
    if (std.mem.eql(u8, metric, "calls"))
        return "call / helper boundary reduction";
    return "unknown";
}

pub const ComparisonError = error{
    InvalidComparison,
    OutOfMemory,
};

threadlocal var last_error_buf: [2048]u8 = undefined;
threadlocal var last_error_len: usize = 0;

pub fn fail(comptime fmt: []const u8, args: anytype) ComparisonError {
    const msg = std.fmt.bufPrint(&last_error_buf, fmt, args) catch "comparison error";
    last_error_len = msg.len;
    return error.InvalidComparison;
}

pub fn lastErrorMessage() []const u8 {
    return last_error_buf[0..last_error_len];
}

pub fn sha256Bytes(allocator: std.mem.Allocator, data: []const u8) ![]const u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

pub fn sha256File(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch {
        return fail("file not found: {s}", .{path});
    };
    defer allocator.free(data);
    return sha256Bytes(allocator, data);
}

pub fn isLowerHex(text: []const u8, expected_len: usize) bool {
    if (text.len != expected_len) return false;
    var all_zero = true;
    for (text) |ch| {
        if (!((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f'))) return false;
        if (ch != '0') all_zero = false;
    }
    return !all_zero;
}

pub fn requireSha256(allocator: std.mem.Allocator, val: ?Value, label: []const u8) ![]const u8 {
    const raw = try requireString(val, label);
    const lower = try allocator.dupe(u8, raw);
    for (lower) |*c| c.* = std.ascii.toLower(c.*);
    if (!isLowerHex(lower, 64)) {
        return fail("{s} must be a non-zero lowercase SHA-256", .{label});
    }
    return lower;
}

pub fn requireObject(val: ?Value, label: []const u8) !ObjectMap {
    const v = val orelse return fail("{s} must be a JSON object", .{label});
    if (v != .object) return fail("{s} must be a JSON object", .{label});
    return v.object;
}

pub fn requireString(val: ?Value, label: []const u8) ![]const u8 {
    const v = val orelse return fail("{s} must be a non-empty string", .{label});
    if (v != .string or v.string.len == 0) return fail("{s} must be a non-empty string", .{label});
    return v.string;
}

pub fn requireInteger(val: ?Value, label: []const u8, minimum: i64) !i64 {
    const v = val orelse return fail("{s} must be an integer >= {d}", .{ label, minimum });
    if (v != .integer or v.integer < minimum) return fail("{s} must be an integer >= {d}", .{ label, minimum });
    return v.integer;
}

pub fn requireNumber(val: ?Value, label: []const u8, minimum: f64) !f64 {
    const v = val orelse return fail("{s} must be a number >= {d}", .{ label, minimum });
    const num: f64 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return fail("{s} must be a number >= {d}", .{ label, minimum }),
    };
    if (!std.math.isFinite(num) or num < minimum) {
        return fail("{s} must be a number >= {d}", .{ label, minimum });
    }
    return num;
}

pub fn resolvePath(allocator: std.mem.Allocator, io: std.Io, base: []const u8, val: ?Value, label: []const u8) ![]const u8 {
    const raw = try requireString(val, label);
    const joined = if (std.fs.path.isAbsolute(raw))
        try allocator.dupe(u8, raw)
    else
        try std.fs.path.join(allocator, &.{ base, raw });
    if (std.Io.Dir.cwd().realPathFileAlloc(io, joined, allocator)) |real| {
        return real;
    } else |_| {
        return std.fs.path.resolve(allocator, &.{joined});
    }
}

pub fn loadJsonObject(allocator: std.mem.Allocator, io: std.Io, path: []const u8, label: []const u8) !ObjectMap {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch {
        return fail("{s} not found: {s}", .{ label, path });
    };
    const parsed = json.parseFromSlice(Value, allocator, bytes, .{}) catch |err| {
        return fail("invalid JSON in {s} {s}: {s}", .{ label, path, @errorName(err) });
    };
    if (parsed.value != .object) {
        return fail("{s} must contain a JSON object", .{label});
    }
    return parsed.value.object;
}

const ByteReader = struct {
    data: []const u8,
    label: []const u8,
    pos: usize = 0,

    fn remaining(self: ByteReader) usize {
        return self.data.len - self.pos;
    }

    fn readByte(self: *ByteReader) !u8 {
        if (self.pos >= self.data.len) return fail("{s}: unexpected end of input", .{self.label});
        const v = self.data[self.pos];
        self.pos += 1;
        return v;
    }

    fn readBytes(self: *ByteReader, size: usize) ![]const u8 {
        if (self.pos + size > self.data.len) return fail("{s}: unexpected end of input", .{self.label});
        const v = self.data[self.pos .. self.pos + size];
        self.pos += size;
        return v;
    }

    fn readU32(self: *ByteReader) !u32 {
        var value: u64 = 0;
        var shift: u6 = 0;
        for (0..5) |_| {
            const byte = try self.readByte();
            value |= (@as(u64, byte & 0x7F) << shift);
            if ((byte & 0x80) == 0) {
                if (value > 0xFFFF_FFFF) break;
                return @intCast(value);
            }
            shift += 7;
        }
        return fail("{s}: invalid u32 LEB128", .{self.label});
    }

    fn readU64(self: *ByteReader) !u64 {
        var value: u128 = 0;
        var shift: u7 = 0;
        for (0..10) |_| {
            const byte = try self.readByte();
            value |= (@as(u128, byte & 0x7F) << shift);
            if ((byte & 0x80) == 0) {
                if (value > std.math.maxInt(u64)) break;
                return @intCast(value);
            }
            shift += 7;
        }
        return fail("{s}: invalid u64 LEB128", .{self.label});
    }

    fn readS33(self: *ByteReader) !i64 {
        var value: i64 = 0;
        var shift: u6 = 0;
        for (0..5) |_| {
            const byte = try self.readByte();
            value |= (@as(i64, byte & 0x7F) << shift);
            shift += 7;
            if ((byte & 0x80) == 0) {
                if (shift < 33 and (byte & 0x40) != 0) {
                    value |= -(@as(i64, 1) << shift);
                }
                return value;
            }
        }
        return fail("{s}: invalid s33 LEB128", .{self.label});
    }

    fn readName(self: *ByteReader) ![]const u8 {
        const len = try self.readU32();
        const raw = try self.readBytes(len);
        if (!std.unicode.utf8ValidateSlice(raw)) return fail("{s}: invalid UTF-8 name", .{self.label});
        return raw;
    }
};

pub const WasmModuleIdentity = struct {
    sha256: []const u8,
    imported_function_count: usize,
    local_function_count: usize,
    function_names: std.AutoArrayHashMapUnmanaged(usize, []const u8),

    pub fn functionCount(self: WasmModuleIdentity) usize {
        return self.imported_function_count + self.local_function_count;
    }
};

fn readValueType(reader: *ByteReader) !void {
    const first = try reader.readByte();
    if (first == 0x63 or first == 0x64) {
        _ = try reader.readS33();
    }
}

fn readLimits(reader: *ByteReader) !void {
    const flags = try reader.readU32();
    if ((flags & 0x4) != 0) {
        _ = try reader.readU64();
        if ((flags & 0x1) != 0) _ = try reader.readU64();
    } else {
        _ = try reader.readU32();
        if ((flags & 0x1) != 0) _ = try reader.readU32();
    }
}

fn parseImports(payload: []const u8, label: []const u8) !usize {
    var reader = ByteReader{ .data = payload, .label = label };
    const count = try reader.readU32();
    var function_count: usize = 0;
    for (0..count) |_| {
        _ = try reader.readName();
        _ = try reader.readName();
        const kind = try reader.readByte();
        switch (kind) {
            0 => {
                _ = try reader.readU32();
                function_count += 1;
            },
            1 => {
                try readValueType(&reader);
                try readLimits(&reader);
            },
            2 => try readLimits(&reader),
            3 => {
                try readValueType(&reader);
                _ = try reader.readByte();
            },
            4 => {
                _ = try reader.readByte();
                _ = try reader.readU32();
            },
            else => return fail("{s}: unsupported import kind {d}", .{ label, kind }),
        }
    }
    if (reader.remaining() != 0) return fail("{s}: trailing bytes in import section", .{label});
    return function_count;
}

fn parseFunctionNames(
    allocator: std.mem.Allocator,
    payload: []const u8,
    label: []const u8,
    names: *std.AutoArrayHashMapUnmanaged(usize, []const u8),
) !void {
    var reader = ByteReader{ .data = payload, .label = label };
    const sec_name = try reader.readName();
    if (!std.mem.eql(u8, sec_name, "name")) return;
    while (reader.remaining() > 0) {
        const subsection_id = try reader.readByte();
        const sub_len = try reader.readU32();
        const subsection = try reader.readBytes(sub_len);
        if (subsection_id != 1) continue;
        const sub_label = try std.fmt.allocPrint(allocator, "{s} function names", .{label});
        var sub = ByteReader{ .data = subsection, .label = sub_label };
        const count = try sub.readU32();
        for (0..count) |_| {
            const index: usize = @intCast(try sub.readU32());
            const name = try allocator.dupe(u8, try sub.readName());
            if (names.contains(index)) {
                return fail("{s}: duplicate function-name entry for wasm index {d}", .{ label, index });
            }
            try names.put(allocator, index, name);
        }
        if (sub.remaining() != 0) {
            return fail("{s}: trailing bytes in function-name map", .{label});
        }
    }
}

pub fn parseCoreWasm(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !WasmModuleIdentity {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch {
        return fail("raw core wasm not found: {s}", .{path});
    };
    if (data.len < 8 or !std.mem.eql(u8, data[0..8], "\x00asm\x01\x00\x00\x00")) {
        return fail("{s}: expected a core wasm v1 module", .{path});
    }
    var reader = ByteReader{ .data = data[8..], .label = path };
    var imports: usize = 0;
    var saw_import_section = false;
    var local_functions: ?usize = null;
    var code_function_count: ?usize = null;
    var names: std.AutoArrayHashMapUnmanaged(usize, []const u8) = .empty;
    while (reader.remaining() > 0) {
        const section_id = try reader.readByte();
        const sec_len = try reader.readU32();
        const payload = try reader.readBytes(sec_len);
        switch (section_id) {
            0 => try parseFunctionNames(allocator, payload, path, &names),
            2 => {
                if (saw_import_section) return fail("{s}: duplicate import section", .{path});
                saw_import_section = true;
                const imp_label = try std.fmt.allocPrint(allocator, "{s} import section", .{path});
                imports += try parseImports(payload, imp_label);
            },
            3 => {
                const fn_label = try std.fmt.allocPrint(allocator, "{s} function section", .{path});
                var sec = ByteReader{ .data = payload, .label = fn_label };
                const count: usize = @intCast(try sec.readU32());
                for (0..count) |_| _ = try sec.readU32();
                if (sec.remaining() != 0) return fail("{s}: trailing bytes in function section", .{path});
                if (local_functions != null) return fail("{s}: duplicate function section", .{path});
                local_functions = count;
            },
            10 => {
                const code_label = try std.fmt.allocPrint(allocator, "{s} code section", .{path});
                var sec = ByteReader{ .data = payload, .label = code_label };
                const count: usize = @intCast(try sec.readU32());
                for (0..count) |_| {
                    const body_len = try sec.readU32();
                    _ = try sec.readBytes(body_len);
                }
                if (sec.remaining() != 0) return fail("{s}: trailing bytes in code section", .{path});
                if (code_function_count != null) return fail("{s}: duplicate code section", .{path});
                code_function_count = count;
            },
            else => {},
        }
    }
    const locals = local_functions orelse 0;
    const codes = code_function_count orelse 0;
    if (codes != locals) {
        return fail("{s}: function/code count mismatch ({d} != {d})", .{ path, locals, codes });
    }
    const total = imports + locals;
    var it = names.iterator();
    while (it.next()) |entry| {
        if (entry.key_ptr.* >= total) {
            return fail("{s}: function-name index {d} is outside 0..{d}", .{ path, entry.key_ptr.*, if (total > 0) total - 1 else 0 });
        }
    }
    return .{
        .sha256 = try sha256Bytes(allocator, data),
        .imported_function_count = imports,
        .local_function_count = locals,
        .function_names = names,
    };
}

pub const ResolvedWasmFunction = struct {
    wasm_index: usize,
    name: ?[]const u8,
};

pub fn resolveWasmFunction(
    allocator: std.mem.Allocator,
    identity: WasmModuleIdentity,
    wasm_index_opt: ?usize,
    name_opt: ?[]const u8,
) !ResolvedWasmFunction {
    var wasm_index: usize = undefined;
    if (wasm_index_opt) |idx| {
        wasm_index = idx;
    } else {
        const name = name_opt orelse return fail("function must provide wasm_index, name, or both", .{});
        var matches: std.ArrayList(usize) = .empty;
        var it = identity.function_names.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.value_ptr.*, name)) {
                try matches.append(allocator, entry.key_ptr.*);
            }
        }
        std.mem.sort(usize, matches.items, {}, std.sort.asc(usize));
        if (matches.items.len == 0) {
            return fail("function name '{s}' is absent from the name section", .{name});
        }
        if (matches.items.len != 1) {
            return fail("function name '{s}' is ambiguous; provide function.wasm_index", .{name});
        }
        wasm_index = matches.items[0];
    }
    if (wasm_index < identity.imported_function_count) {
        return fail("wasm function {d} is imported; a WAMR local_func must be defined", .{wasm_index});
    }
    if (wasm_index >= identity.functionCount()) {
        return fail("wasm function {d} is outside 0..{d}", .{
            wasm_index,
            if (identity.functionCount() > 0) identity.functionCount() - 1 else 0,
        });
    }
    const actual_name = identity.function_names.get(wasm_index);
    if (name_opt) |expected| {
        if (actual_name == null or !std.mem.eql(u8, actual_name.?, expected)) {
            return fail("wasm function {d} name mismatch: expected '{s}', found '{s}'", .{
                wasm_index,
                expected,
                actual_name orelse "<none>",
            });
        }
    }
    return .{ .wasm_index = wasm_index, .name = actual_name };
}

pub const CwasmFunction = struct {
    code: []const u8,
    code_size: usize,
    text_offset: usize,
    function_count: usize,
};

pub fn readWamrFunction(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    local_func: usize,
) !CwasmFunction {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch {
        return fail("{s}: cannot read WAMR cwasm", .{path});
    };
    if (data.len < 8) return fail("{s}: truncated WAMR cwasm", .{path});
    const magic = std.mem.readInt(u32, data[0..4], .little);
    const version = std.mem.readInt(u32, data[4..8], .little);
    if (magic != AOT_MAGIC) return fail("{s}: bad WAMR cwasm magic 0x{x}", .{ path, magic });
    if (version != AOT_VERSION) {
        return fail("{s}: WAMR cwasm version {d} is unsupported; expected {d}", .{ path, version, AOT_VERSION });
    }
    var pos: usize = 8;
    var text: ?[]const u8 = null;
    var offsets: ?[]usize = null;
    while (pos + 8 <= data.len) {
        const section_type = std.mem.readInt(u32, data[pos..][0..4], .little);
        const section_size: usize = std.mem.readInt(u32, data[pos + 4 ..][0..4], .little);
        pos += 8;
        if (pos + section_size > data.len) return fail("{s}: truncated cwasm section {d}", .{ path, section_type });
        const payload = data[pos .. pos + section_size];
        if (section_type == AOT_TEXT_SECTION) {
            text = payload;
        } else if (section_type == AOT_FUNCTION_SECTION) {
            if (payload.len < 4) return fail("{s}: truncated function section", .{path});
            const count: usize = std.mem.readInt(u32, payload[0..4], .little);
            const expected = 4 + count * 8;
            if (payload.len != expected) {
                return fail("{s}: function section size {d} != {d}", .{ path, payload.len, expected });
            }
            const offs = try allocator.alloc(usize, count);
            for (0..count) |i| {
                offs[i] = std.mem.readInt(u32, payload[4 + i * 8 ..][0..4], .little);
            }
            offsets = offs;
        }
        pos += section_size;
    }
    if (pos != data.len) return fail("{s}: trailing truncated cwasm header", .{path});
    const txt = text orelse return fail("{s}: missing text or function section", .{path});
    const offs = offsets orelse return fail("{s}: missing text or function section", .{path});
    if (offs.len == 0 or offs[0] != 0) return fail("{s}: invalid function offsets", .{path});
    for (1..offs.len) |i| {
        if (offs[i] <= offs[i - 1]) return fail("{s}: invalid function offsets", .{path});
    }
    if (offs[offs.len - 1] >= txt.len) return fail("{s}: final function offset is outside text", .{path});
    if (local_func >= offs.len) {
        return fail("WAMR local_func {d} is outside 0..{d}", .{ local_func, offs.len - 1 });
    }
    const start = offs[local_func];
    const end = if (local_func + 1 < offs.len) offs[local_func + 1] else txt.len;
    return .{
        .code = try allocator.dupe(u8, txt[start..end]),
        .code_size = end - start,
        .text_offset = start,
        .function_count = offs.len,
    };
}

pub const Instruction = struct {
    offset: usize,
    size: usize,
    mnemonic: []const u8,
    operands: []const u8,
    text: []const u8,
    raw_bytes: []const u8,
};

pub const WasmtimeHeader = struct {
    address: ?usize,
    module_index: usize,
    function_index: usize,
    name: ?[]const u8,
};

fn isHexToken(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |ch| {
        if (!std.ascii.isHex(ch)) return false;
    }
    return true;
}

pub fn parseWasmtimeHeader(line: []const u8) ?WasmtimeHeader {
    const trimmed = std.mem.trim(u8, line, " \t\r\n");
    if (trimmed.len == 0 or trimmed[trimmed.len - 1] != ':') return null;
    var rest = trimmed[0 .. trimmed.len - 1];
    var address: ?usize = null;
    if (!std.mem.startsWith(u8, rest, "wasm[") and !std.mem.startsWith(u8, rest, "<wasm[")) {
        const ws = std.mem.indexOfAny(u8, rest, " \t") orelse return null;
        const addr_tok = rest[0..ws];
        if (!isHexToken(addr_tok)) return null;
        address = std.fmt.parseInt(usize, addr_tok, 16) catch return null;
        rest = std.mem.trimStart(u8, rest[ws..], " \t");
    }
    if (rest.len > 0 and rest[0] == '<') rest = rest[1..];
    if (!std.mem.startsWith(u8, rest, "wasm[")) return null;
    rest = rest["wasm[".len..];
    const mod_end = std.mem.indexOf(u8, rest, "]::function[") orelse return null;
    const mod_str = rest[0..mod_end];
    if (mod_str.len == 0) return null;
    for (mod_str) |ch| if (!std.ascii.isDigit(ch)) return null;
    const module_index = std.fmt.parseInt(usize, mod_str, 10) catch return null;
    rest = rest[mod_end + "]::function[".len ..];
    const fn_end = std.mem.indexOfScalar(u8, rest, ']') orelse return null;
    const fn_str = rest[0..fn_end];
    if (fn_str.len == 0) return null;
    for (fn_str) |ch| if (!std.ascii.isDigit(ch)) return null;
    const function_index = std.fmt.parseInt(usize, fn_str, 10) catch return null;
    var tail = rest[fn_end + 1 ..];
    if (tail.len > 0 and tail[0] == '>') tail = tail[1..];
    if (tail.len > 0 and tail[tail.len - 1] == '>') tail = tail[0 .. tail.len - 1];
    var name: ?[]const u8 = null;
    if (tail.len > 0) {
        if (!std.mem.startsWith(u8, tail, "::")) return null;
        name = tail[2..];
        if (name.?.len == 0 or std.mem.indexOfAny(u8, name.?, ">:") != null) return null;
    }
    return .{
        .address = address,
        .module_index = module_index,
        .function_index = function_index,
        .name = name,
    };
}

pub fn normalizeMnemonic(
    allocator: std.mem.Allocator,
    text: []const u8,
) !struct { mnemonic: []const u8, operands: []const u8 } {
    var trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return .{ .mnemonic = "", .operands = "" };
    const first_ws = std.mem.indexOfAny(u8, trimmed, " \t");
    var first_word = if (first_ws) |idx| trimmed[0..idx] else trimmed;
    var rest = if (first_ws) |idx| std.mem.trimStart(u8, trimmed[idx..], " \t") else "";

    var lower_first_buf: [64]u8 = undefined;
    if (first_word.len <= lower_first_buf.len) {
        for (first_word, 0..) |c, i| lower_first_buf[i] = std.ascii.toLower(c);
        const lf = lower_first_buf[0..first_word.len];
        if (std.mem.eql(u8, lf, "lock") or std.mem.eql(u8, lf, "rep") or
            std.mem.eql(u8, lf, "repe") or std.mem.eql(u8, lf, "repz") or
            std.mem.eql(u8, lf, "repne") or std.mem.eql(u8, lf, "repnz"))
        {
            trimmed = rest;
            if (trimmed.len == 0) return .{ .mnemonic = "", .operands = "" };
            const ws2 = std.mem.indexOfAny(u8, trimmed, " \t");
            first_word = if (ws2) |idx| trimmed[0..idx] else trimmed;
            rest = if (ws2) |idx| std.mem.trimStart(u8, trimmed[idx..], " \t") else "";
        }
    }

    const lower = try allocator.dupe(u8, first_word);
    for (lower) |*c| c.* = std.ascii.toLower(c.*);

    var mnemonic: []const u8 = lower;
    const aliases = .{
        .{ "callq", "call" },
        .{ "jmpq", "jmp" },
        .{ "pushq", "push" },
        .{ "popq", "pop" },
        .{ "retq", "ret" },
        .{ "leaveq", "leave" },
        .{ "leaq", "lea" },
        .{ "movq", "mov" },
        .{ "movl", "mov" },
        .{ "movw", "mov" },
        .{ "movb", "mov" },
    };
    inline for (aliases) |pair| {
        if (std.mem.eql(u8, mnemonic, pair[0])) {
            mnemonic = pair[1];
            break;
        }
    }

    if (mnemonic.len > 2 and std.mem.indexOfScalar(u8, "bwlq", mnemonic[mnemonic.len - 1]) != null) {
        const stem = mnemonic[0 .. mnemonic.len - 1];
        for ([_][]const u8{
            "add", "sub", "and", "or",  "xor", "cmp", "test", "inc",
            "dec", "neg", "not", "shl", "shr", "sar", "imul",
        }) |cand| {
            if (std.mem.eql(u8, stem, cand)) {
                mnemonic = stem;
                break;
            }
        }
    }

    return .{
        .mnemonic = try allocator.dupe(u8, mnemonic),
        .operands = try allocator.dupe(u8, std.mem.trim(u8, rest, " \t")),
    };
}

fn isHexByteGroup(tok: []const u8) bool {
    if (tok.len != 2 and tok.len != 4 and tok.len != 8) return false;
    return isHexToken(tok);
}

fn appendHexByteGroup(allocator: std.mem.Allocator, list: *std.ArrayList(u8), tok: []const u8) !void {
    var i: usize = 0;
    while (i + 2 <= tok.len) : (i += 2) {
        const b = std.fmt.parseInt(u8, tok[i .. i + 2], 16) catch unreachable;
        try list.append(allocator, b);
    }
}

fn parseInstructionLine(
    allocator: std.mem.Allocator,
    line: []const u8,
) !?struct { offset: usize, bytes: []u8, assembly: []const u8 } {
    const trimmed_start = std.mem.trimStart(u8, line, " \t");
    const colon = std.mem.indexOfScalar(u8, trimmed_start, ':') orelse return null;
    const addr_str = trimmed_start[0..colon];
    if (!isHexToken(addr_str)) return null;
    var rest = trimmed_start[colon + 1 ..];
    if (rest.len == 0 or (rest[0] != ' ' and rest[0] != '\t')) return null;
    rest = std.mem.trimStart(u8, rest, " \t");

    var raw_bytes: std.ArrayList(u8) = .empty;
    var cursor: usize = 0;
    while (cursor < rest.len) {
        var tok_end = cursor;
        while (tok_end < rest.len and rest[tok_end] != ' ' and rest[tok_end] != '\t' and rest[tok_end] != '\r') : (tok_end += 1) {}
        const tok = rest[cursor..tok_end];
        if (!isHexByteGroup(tok)) break;
        try appendHexByteGroup(allocator, &raw_bytes, tok);
        const after_tok = rest[tok_end..];
        var ws_len: usize = 0;
        var has_tab = false;
        while (ws_len < after_tok.len and (after_tok[ws_len] == ' ' or after_tok[ws_len] == '\t')) : (ws_len += 1) {
            if (after_tok[ws_len] == '\t') has_tab = true;
        }
        cursor = tok_end + ws_len;
        if (ws_len >= 2 or has_tab or cursor >= rest.len) {
            if (cursor < rest.len) break;
        }
    }
    if (raw_bytes.items.len == 0) return null;
    const offset = std.fmt.parseInt(usize, addr_str, 16) catch return null;
    const assembly = std.mem.trim(u8, rest[cursor..], " \t\r\n");
    return .{
        .offset = offset,
        .bytes = try raw_bytes.toOwnedSlice(allocator),
        .assembly = assembly,
    };
}

fn parseContinuationLine(allocator: std.mem.Allocator, line: []const u8) !?[]u8 {
    const trimmed = std.mem.trim(u8, line, " \t\r\n");
    if (trimmed.len == 0) return null;
    var bytes: std.ArrayList(u8) = .empty;
    var tok_it = std.mem.tokenizeAny(u8, trimmed, " \t");
    while (tok_it.next()) |tok| {
        if (!isHexByteGroup(tok)) return null;
        try appendHexByteGroup(allocator, &bytes, tok);
    }
    if (bytes.items.len == 0) return null;
    return try bytes.toOwnedSlice(allocator);
}

pub fn parseDisassembly(
    allocator: std.mem.Allocator,
    text: []const u8,
    wasmtime_wasm_index: ?usize,
) ![]Instruction {
    var instructions: std.ArrayList(Instruction) = .empty;
    var selected = (wasmtime_wasm_index == null);
    var saw_header = false;
    var saw_selected_header = false;
    var base_offset: ?usize = null;

    var line_it = std.mem.splitScalar(u8, text, '\n');
    while (line_it.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (parseWasmtimeHeader(line)) |hdr| {
            saw_header = true;
            selected = (wasmtime_wasm_index != null and hdr.module_index == 0 and hdr.function_index == wasmtime_wasm_index.?);
            if (selected) {
                if (saw_selected_header) {
                    return fail("Wasmtime objdump contained duplicate matching function headers", .{});
                }
                saw_selected_header = true;
            }
            continue;
        }
        if (saw_header and !selected) continue;

        if (try parseInstructionLine(allocator, line)) |inst| {
            if (inst.assembly.len == 0) {
                if (instructions.items.len > 0) {
                    var last = &instructions.items[instructions.items.len - 1];
                    last.size += inst.bytes.len;
                    last.raw_bytes = try std.mem.concat(allocator, u8, &.{ last.raw_bytes, inst.bytes });
                }
                continue;
            }
            const norm = try normalizeMnemonic(allocator, inst.assembly);
            if (norm.mnemonic.len == 0) continue;
            if (base_offset == null) base_offset = inst.offset;
            if (inst.offset < base_offset.?) {
                return fail("disassembly instruction offsets are not increasing", .{});
            }
            try instructions.append(allocator, .{
                .offset = inst.offset - base_offset.?,
                .size = inst.bytes.len,
                .mnemonic = norm.mnemonic,
                .operands = norm.operands,
                .text = try allocator.dupe(u8, inst.assembly),
                .raw_bytes = inst.bytes,
            });
            continue;
        }
        if (try parseContinuationLine(allocator, line)) |extra| {
            if (instructions.items.len > 0) {
                var last = &instructions.items[instructions.items.len - 1];
                last.size += extra.len;
                last.raw_bytes = try std.mem.concat(allocator, u8, &.{ last.raw_bytes, extra });
            }
        }
    }

    if (wasmtime_wasm_index) |widx| {
        if (!saw_selected_header) {
            return fail("Wasmtime objdump did not contain exactly wasm[0]::function[{d}]", .{widx});
        }
    }
    if (instructions.items.len == 0) {
        return fail("disassembly contained no instructions", .{});
    }
    for (1..instructions.items.len) |i| {
        if (instructions.items[i].offset <= instructions.items[i - 1].offset) {
            return fail("disassembly instruction offsets are not increasing", .{});
        }
    }
    return instructions.toOwnedSlice(allocator);
}

fn extractJsonObjectSlice(text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trimStart(u8, text, " \t\r\n");
    if (trimmed.len == 0 or trimmed[0] != '{') return null;
    var depth: usize = 0;
    var in_string = false;
    var escaped = false;
    for (trimmed, 0..) |ch, idx| {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            depth += 1;
        } else if (ch == '}') {
            depth -= 1;
            if (depth == 0) return trimmed[0 .. idx + 1];
        }
    }
    return null;
}

pub fn parseWasmtimeExplore(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    wasm_index: usize,
) !struct { all: []Instruction, body: []Instruction } {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch {
        return fail("Wasmtime explore file not found: {s}", .{path});
    };
    const marker = "window.ASM =";
    const start_idx = std.mem.indexOf(u8, text, marker) orelse {
        return fail("Wasmtime explore output has no window.ASM payload", .{});
    };
    const obj_slice = extractJsonObjectSlice(text[start_idx + marker.len ..]) orelse {
        return fail("Wasmtime explore window.ASM payload is malformed", .{});
    };
    const parsed = json.parseFromSlice(Value, allocator, obj_slice, .{}) catch |err| {
        return fail("Wasmtime explore window.ASM payload is malformed: {s}", .{@errorName(err)});
    };
    const root = try requireObject(parsed.value, "Wasmtime explore ASM");
    const funcs_val = root.get("functions") orelse return fail("Wasmtime explore ASM functions must be an array", .{});
    if (funcs_val != .array) return fail("Wasmtime explore ASM functions must be an array", .{});

    var matched_fn: ?ObjectMap = null;
    var match_count: usize = 0;
    for (funcs_val.array.items) |item| {
        if (item == .object) {
            if (item.object.get("func_index")) |fi| {
                if (fi == .integer and fi.integer == @as(i64, @intCast(wasm_index))) {
                    matched_fn = item.object;
                    match_count += 1;
                }
            }
        }
    }
    if (match_count != 1) {
        return fail("Wasmtime explore must contain exactly one function with wasm index {d}, found {d}", .{ wasm_index, match_count });
    }
    const raw_insts_val = matched_fn.?.get("instructions") orelse return fail("Wasmtime explore function has no instructions", .{});
    if (raw_insts_val != .array or raw_insts_val.array.items.len == 0) {
        return fail("Wasmtime explore function has no instructions", .{});
    }

    var first_address: ?usize = null;
    var all_instructions: std.ArrayList(Instruction) = .empty;
    var mapped_indices: std.ArrayList(usize) = .empty;

    for (raw_insts_val.array.items, 0..) |raw, idx| {
        const item = try requireObject(raw, "Wasmtime explore instruction");
        const address: usize = @intCast(try requireInteger(item.get("address"), "explore instruction.address", 0));
        const raw_bytes_val = item.get("bytes") orelse return fail("explore instruction {d}.bytes must be a non-empty byte array", .{idx});
        if (raw_bytes_val != .array or raw_bytes_val.array.items.len == 0) {
            return fail("explore instruction {d}.bytes must be a non-empty byte array", .{idx});
        }
        const raw_bytes = try allocator.alloc(u8, raw_bytes_val.array.items.len);
        for (raw_bytes_val.array.items, 0..) |bval, bi| {
            if (bval != .integer or bval.integer < 0 or bval.integer > 255) {
                return fail("explore instruction {d}.bytes must be a non-empty byte array", .{idx});
            }
            raw_bytes[bi] = @intCast(bval.integer);
        }
        const raw_mnem = try requireString(item.get("mnemonic"), "explore instruction.mnemonic");
        const lower_mnem = try allocator.dupe(u8, raw_mnem);
        for (lower_mnem) |*c| c.* = std.ascii.toLower(c.*);
        const ops_val = item.get("operands") orelse return fail("explore instruction.operands must be a string", .{});
        if (ops_val != .string) return fail("explore instruction.operands must be a string", .{});
        const operands = ops_val.string;
        const full_text = std.mem.trimEnd(u8, try std.fmt.allocPrint(allocator, "{s} {s}", .{ lower_mnem, operands }), " \t");
        const norm = try normalizeMnemonic(allocator, full_text);
        if (first_address == null) first_address = address;
        if (address < first_address.?) return fail("Wasmtime explore instruction bytes are not contiguous", .{});
        const inst = Instruction{
            .offset = address - first_address.?,
            .size = raw_bytes.len,
            .mnemonic = norm.mnemonic,
            .operands = operands,
            .text = full_text,
            .raw_bytes = raw_bytes,
        };
        if (all_instructions.items.len > 0) {
            const prev = all_instructions.items[all_instructions.items.len - 1];
            if (inst.offset != prev.offset + prev.size) {
                return fail("Wasmtime explore instruction bytes are not contiguous", .{});
            }
        }
        try all_instructions.append(allocator, inst);
        if (item.get("wasm_offset")) |wo| {
            if (wo != .null) {
                _ = try requireInteger(wo, "explore instruction.wasm_offset", 0);
                try mapped_indices.append(allocator, idx);
            }
        }
    }
    if (mapped_indices.items.len == 0) {
        return fail("Wasmtime explore emitted no wasm address mappings for the function", .{});
    }
    var final = mapped_indices.items[mapped_indices.items.len - 1];
    const isTerminator = struct {
        fn check(m: []const u8) bool {
            for ([_][]const u8{ "ret", "ud2", "jmp", "hlt", "int3" }) |t| {
                if (std.mem.eql(u8, m, t)) return true;
            }
            return false;
        }
    }.check;
    while (final + 1 < all_instructions.items.len and !isTerminator(all_instructions.items[final].mnemonic)) {
        final += 1;
    }
    if (!isTerminator(all_instructions.items[final].mnemonic)) {
        return fail("cannot separate Wasmtime trailing data: no final mapped terminator", .{});
    }
    const all_slice = try all_instructions.toOwnedSlice(allocator);
    return .{
        .all = all_slice,
        .body = all_slice[0 .. final + 1],
    };
}

pub fn splitOperands(allocator: std.mem.Allocator, operands: []const u8) ![][]const u8 {
    var result: std.ArrayList([]const u8) = .empty;
    var start: usize = 0;
    var depth: usize = 0;
    for (operands, 0..) |ch, idx| {
        if (ch == '[' or ch == '(') {
            depth += 1;
        } else if (ch == ']' or ch == ')') {
            if (depth > 0) depth -= 1;
        } else if (ch == ',' and depth == 0) {
            const piece = std.mem.trim(u8, operands[start..idx], " \t");
            try result.append(allocator, piece);
            start = idx + 1;
        }
    }
    const tail = std.mem.trim(u8, operands[start..], " \t");
    if (tail.len > 0) {
        try result.append(allocator, tail);
    }
    return result.toOwnedSlice(allocator);
}

pub fn isFrameMemory(operand: []const u8) bool {
    var compact_buf: [256]u8 = undefined;
    var compact_len: usize = 0;
    for (operand) |ch| {
        if (ch == ' ' or ch == '\t') continue;
        if (compact_len >= compact_buf.len) return false;
        compact_buf[compact_len] = std.ascii.toLower(ch);
        compact_len += 1;
    }
    const compact = compact_buf[0..compact_len];

    // Check Intel bracket form: \[(?:[^\]]*[+\-*])?(?:rbp|rsp)(?:[+\-*][^\]]*)?\]
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, compact, pos, "[")) |open| {
        const close = std.mem.indexOfPos(u8, compact, open + 1, "]") orelse break;
        const inner = compact[open + 1 .. close];
        for ([_][]const u8{ "rbp", "rsp" }) |reg| {
            var rpos: usize = 0;
            while (std.mem.indexOfPos(u8, inner, rpos, reg)) |idx| {
                const before_ok = (idx == 0) or (inner[idx - 1] == '+' or inner[idx - 1] == '-' or inner[idx - 1] == '*');
                const after_idx = idx + reg.len;
                const after_ok = (after_idx == inner.len) or (inner[after_idx] == '+' or inner[after_idx] == '-' or inner[after_idx] == '*');
                if (before_ok and after_ok) return true;
                rpos = idx + 1;
            }
        }
        pos = close + 1;
    }

    // Check AT&T paren form: \([^)]*%(?:rbp|rsp)(?:,[^)]*)?\)
    pos = 0;
    while (std.mem.indexOfPos(u8, compact, pos, "(")) |open| {
        const close = std.mem.indexOfPos(u8, compact, open + 1, ")") orelse break;
        const inner = compact[open + 1 .. close];
        for ([_][]const u8{ "%rbp", "%rsp" }) |reg| {
            var rpos: usize = 0;
            while (std.mem.indexOfPos(u8, inner, rpos, reg)) |idx| {
                const after_idx = idx + reg.len;
                if (after_idx == inner.len or inner[after_idx] == ',') return true;
                rpos = idx + 1;
            }
        }
        pos = close + 1;
    }
    return false;
}

pub fn isRegister(operand: []const u8) bool {
    var lower_buf: [128]u8 = undefined;
    const trimmed = std.mem.trim(u8, operand, " \t");
    if (trimmed.len == 0 or trimmed.len > lower_buf.len) return false;
    for (trimmed, 0..) |ch, i| lower_buf[i] = std.ascii.toLower(ch);
    var value: []const u8 = lower_buf[0..trimmed.len];

    for ([_][]const u8{
        "byte", "word", "dword", "qword", "xmmword", "ymmword", "zmmword",
    }) |sz| {
        if (std.mem.startsWith(u8, value, sz) and value.len > sz.len and (value[sz.len] == ' ' or value[sz.len] == '\t')) {
            const after_sz = std.mem.trimStart(u8, value[sz.len..], " \t");
            if (std.mem.startsWith(u8, after_sz, "ptr") and after_sz.len > 3 and (after_sz[3] == ' ' or after_sz[3] == '\t')) {
                value = std.mem.trimStart(u8, after_sz[3..], " \t");
                break;
            }
        }
    }
    value = std.mem.trimStart(u8, value, "*%");
    if (value.len == 0) return false;

    for ([_][]const u8{
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
        "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
        "ax",  "bx",  "cx",  "dx",  "si",  "di",  "bp",  "sp",
        "sil", "dil", "bpl", "spl", "al",  "ah",  "bl",  "bh",
        "cl",  "ch",  "dl",  "dh",
    }) |reg| {
        if (std.mem.eql(u8, value, reg)) return true;
    }

    if (value[0] == 'r') {
        var rest = value[1..];
        if (rest.len > 0 and (rest[rest.len - 1] == 'b' or rest[rest.len - 1] == 'w' or rest[rest.len - 1] == 'd')) {
            rest = rest[0 .. rest.len - 1];
        }
        if (rest.len >= 1 and rest.len <= 2) {
            if (std.fmt.parseInt(u32, rest, 10)) |num| {
                if (num <= 15 and (rest.len == 1 or rest[0] == '1')) return true;
            } else |_| {}
        }
    }

    for ([_][]const u8{ "xmm", "ymm", "zmm", "mm", "k" }) |prefix| {
        if (std.mem.startsWith(u8, value, prefix)) {
            const digits = value[prefix.len..];
            if (digits.len > 0) {
                var all_digits = true;
                for (digits) |d| if (!std.ascii.isDigit(d)) {
                    all_digits = false;
                };
                if (all_digits) return true;
            }
        }
    }
    return false;
}

fn isMove(mnemonic: []const u8) bool {
    return std.mem.startsWith(u8, mnemonic, "mov") or
        std.mem.startsWith(u8, mnemonic, "vmov") or
        std.mem.startsWith(u8, mnemonic, "kmov");
}

fn isAttSyntax(instruction: Instruction) bool {
    return std.mem.indexOfAny(u8, instruction.operands, "%$") != null;
}

fn frameAccesses(allocator: std.mem.Allocator, instruction: Instruction) !struct { load: bool, store: bool } {
    if (!isMove(instruction.mnemonic)) return .{ .load = false, .store = false };
    const operands = try splitOperands(allocator, instruction.operands);
    if (operands.len == 0) return .{ .load = false, .store = false };
    const destination: usize = if (isAttSyntax(instruction)) operands.len - 1 else 0;
    var load = false;
    var store = false;
    for (operands, 0..) |op, idx| {
        if (isFrameMemory(op)) {
            if (idx == destination) {
                store = true;
            } else {
                load = true;
            }
        }
    }
    return .{ .load = load, .store = store };
}

fn parseImmediate(value: []const u8) ?i64 {
    var text = std.mem.trim(u8, value, " \t");
    if (text.len > 0 and text[0] == '$') text = std.mem.trim(u8, text[1..], " \t");
    if (text.len == 0) return null;
    return std.fmt.parseInt(i64, text, 0) catch null;
}

pub fn inferFrameSize(allocator: std.mem.Allocator, instructions: []const Instruction) !Value {
    var fixed: i64 = 0;
    var saw_adjustment = false;
    var saw_sub_adjustment = false;
    var prologue_open = true;

    for (instructions) |inst| {
        const operands = try splitOperands(allocator, inst.operands);
        const att = isAttSyntax(inst);
        const dest_op = if (operands.len == 0) "" else if (att) operands[operands.len - 1] else operands[0];
        const dest_trim = std.mem.trimStart(u8, dest_op, "%");
        if (prologue_open) {
            if (std.mem.eql(u8, inst.mnemonic, "endbr64") or std.mem.eql(u8, inst.mnemonic, "nop")) {
                continue;
            }
            if (std.mem.eql(u8, inst.mnemonic, "push")) {
                fixed += 8;
                saw_adjustment = true;
                continue;
            }
            if (std.mem.eql(u8, inst.mnemonic, "sub") and operands.len == 2 and std.ascii.eqlIgnoreCase(dest_trim, "rsp")) {
                if (saw_sub_adjustment) {
                    return try unavailableMetric(allocator, "multiple prologue stack allocations are ambiguous");
                }
                const amount = parseImmediate(if (att) operands[0] else operands[1]);
                if (amount == null or amount.? < 0) {
                    return try unavailableMetric(allocator, "non-constant stack adjustment in function prologue");
                }
                fixed += amount.?;
                saw_adjustment = true;
                saw_sub_adjustment = true;
                continue;
            }
            if (std.mem.eql(u8, inst.mnemonic, "and") and operands.len > 0 and std.ascii.eqlIgnoreCase(dest_trim, "rsp")) {
                return try unavailableMetric(allocator, "dynamic stack alignment prevents fixed-frame recovery");
            }
            if (std.mem.eql(u8, inst.mnemonic, "mov") or std.mem.eql(u8, inst.mnemonic, "lea")) {
                continue;
            }
            prologue_open = false;
        }
    }

    var map: ObjectMap = .empty;
    try map.put(allocator, "status", .{ .string = "measured" });
    try map.put(allocator, "value", .{ .integer = if (saw_adjustment) fixed else 0 });
    try map.put(allocator, "unit", .{ .string = "bytes" });
    try map.put(allocator, "method", .{ .string = "fixed x86_64 prologue stack adjustment" });
    return .{ .object = map };
}

fn measuredStaticMetric(allocator: std.mem.Allocator, value: i64, unit: []const u8, method: []const u8) !Value {
    var map: ObjectMap = .empty;
    try map.put(allocator, "status", .{ .string = "measured" });
    try map.put(allocator, "value", .{ .integer = value });
    try map.put(allocator, "unit", .{ .string = unit });
    try map.put(allocator, "method", .{ .string = method });
    return .{ .object = map };
}

fn unavailableMetric(allocator: std.mem.Allocator, reason: []const u8) !Value {
    var map: ObjectMap = .empty;
    try map.put(allocator, "status", .{ .string = "unavailable" });
    try map.put(allocator, "reason", .{ .string = reason });
    return .{ .object = map };
}

pub fn aggregateStaticMetrics(
    allocator: std.mem.Allocator,
    instructions: []const Instruction,
    code_size: usize,
) !ObjectMap {
    var frame_loads: i64 = 0;
    var frame_stores: i64 = 0;
    var reg_reg_moves: i64 = 0;
    var address_generation: i64 = 0;
    var branches: i64 = 0;
    var indirect_dispatch: i64 = 0;
    var calls: i64 = 0;

    for (instructions) |inst| {
        const operands = try splitOperands(allocator, inst.operands);
        const fa = try frameAccesses(allocator, inst);
        if (fa.load) frame_loads += 1;
        if (fa.store) frame_stores += 1;
        if (isMove(inst.mnemonic) and operands.len >= 2 and isRegister(operands[0]) and isRegister(operands[1])) {
            reg_reg_moves += 1;
        }
        if (std.mem.eql(u8, inst.mnemonic, "lea")) {
            address_generation += 1;
        }
        const is_branch = std.mem.eql(u8, inst.mnemonic, "jmp") or
            (inst.mnemonic.len > 1 and inst.mnemonic[0] == 'j');
        if (is_branch) branches += 1;
        if (std.mem.eql(u8, inst.mnemonic, "call")) calls += 1;
        if ((std.mem.eql(u8, inst.mnemonic, "call") or std.mem.eql(u8, inst.mnemonic, "jmp")) and operands.len > 0) {
            const target = std.mem.trimStart(u8, operands[0], "*");
            if (isRegister(target) or std.mem.indexOfAny(u8, target, "[(") != null) {
                indirect_dispatch += 1;
            }
        }
    }

    var metrics: ObjectMap = .empty;
    try metrics.put(allocator, "native_instructions", try measuredStaticMetric(allocator, @intCast(instructions.len), "instructions", "parsed native disassembly"));
    try metrics.put(allocator, "code_size_bytes", try measuredStaticMetric(allocator, @intCast(code_size), "bytes", "function byte extent"));
    try metrics.put(allocator, "frame_size_bytes", try inferFrameSize(allocator, instructions));
    try metrics.put(allocator, "frame_loads", try measuredStaticMetric(allocator, frame_loads, "instructions", "classified native disassembly"));
    try metrics.put(allocator, "frame_stores", try measuredStaticMetric(allocator, frame_stores, "instructions", "classified native disassembly"));
    try metrics.put(allocator, "reg_reg_moves", try measuredStaticMetric(allocator, reg_reg_moves, "instructions", "classified native disassembly"));
    try metrics.put(allocator, "address_generation", try measuredStaticMetric(allocator, address_generation, "instructions", "classified native disassembly"));
    try metrics.put(allocator, "branches", try measuredStaticMetric(allocator, branches, "instructions", "classified native disassembly"));
    try metrics.put(allocator, "indirect_dispatch", try measuredStaticMetric(allocator, indirect_dispatch, "instructions", "classified native disassembly"));
    try metrics.put(allocator, "calls", try measuredStaticMetric(allocator, calls, "instructions", "classified native disassembly"));
    return metrics;
}

fn sampleMetric(allocator: std.mem.Allocator, samples: i64, function_samples: i64, total_samples: i64) !Value {
    var map: ObjectMap = .empty;
    try map.put(allocator, "status", .{ .string = "measured" });
    try map.put(allocator, "samples", .{ .integer = samples });
    const fn_pct = if (function_samples != 0) 100.0 * @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(function_samples)) else 0.0;
    const run_pct = if (total_samples != 0) 100.0 * @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(total_samples)) else 0.0;
    try map.put(allocator, "percent_of_function_samples", .{ .float = fn_pct });
    try map.put(allocator, "percent_of_run_samples", .{ .float = run_pct });
    try map.put(allocator, "method", .{ .string = "perf self samples classified by sampled native instruction" });
    return .{ .object = map };
}

pub fn wamrDynamicMetrics(
    allocator: std.mem.Allocator,
    attribution: ObjectMap,
) !struct { dynamic: ObjectMap, hotness: ObjectMap } {
    const total = try requireInteger(attribution.get("total_samples"), "attribution.total_samples", 1);
    const classified = try requireObject(attribution.get("classified_function"), "attribution.classified_function");
    const function_samples = try requireInteger(classified.get("samples"), "attribution.classified_function.samples", 1);
    const share = try requireNumber(classified.get("percent_of_run"), "attribution.classified_function.percent_of_run", 0.0);
    const classes = try requireObject(classified.get("classes"), "attribution.classified_function.classes");
    if (function_samples > total) {
        return fail("attribution classified function samples exceed total samples", .{});
    }
    const expected_share = 100.0 * @as(f64, @floatFromInt(function_samples)) / @as(f64, @floatFromInt(total));
    if (@abs(share - expected_share) > 0.05) {
        return fail("attribution classified function percent_of_run is inconsistent with its sample counts", .{});
    }
    var class_total: i64 = 0;
    var cit = classes.iterator();
    while (cit.next()) |entry| {
        const val = try requireObject(entry.value_ptr.*, "attribution class");
        class_total += try requireInteger(val.get("samples"), "attribution class.samples", 0);
    }
    if (class_total != function_samples) {
        return fail("attribution instruction-class samples do not sum to function samples", .{});
    }

    const samplesFor = struct {
        fn get(cls: ObjectMap, name: []const u8) !i64 {
            const raw = cls.get(name) orelse return 0;
            const obj = try requireObject(raw, name);
            return requireInteger(obj.get("samples"), name, 0);
        }
    }.get;

    var dynamic: ObjectMap = .empty;
    try dynamic.put(allocator, "code_size_bytes", try unavailableMetric(allocator, "this metric is static metadata, not an instruction class"));
    try dynamic.put(allocator, "frame_size_bytes", try unavailableMetric(allocator, "this metric is static metadata, not an instruction class"));
    try dynamic.put(allocator, "native_instructions", try sampleMetric(allocator, function_samples, function_samples, total));
    try dynamic.put(allocator, "reg_reg_moves", try sampleMetric(allocator, try samplesFor(classes, "regmov"), function_samples, total));
    try dynamic.put(allocator, "indirect_dispatch", try sampleMetric(allocator, try samplesFor(classes, "dispatch_jmp"), function_samples, total));
    try dynamic.put(allocator, "calls", try sampleMetric(allocator, try samplesFor(classes, "call"), function_samples, total));

    const frame_opt = classified.get("frame_attribution");
    if (frame_opt == null or frame_opt.? == .null) {
        try dynamic.put(allocator, "frame_loads", try unavailableMetric(allocator, "no exact-image frame sidecar; frame accesses cannot be assigned sound origins"));
        try dynamic.put(allocator, "frame_stores", try unavailableMetric(allocator, "no exact-image frame sidecar; frame accesses cannot be assigned sound origins"));
    } else {
        const frame = try requireObject(frame_opt, "frame_attribution");
        const coverage = try requireObject(frame.get("coverage"), "frame_attribution.coverage");
        const origins = try requireObject(frame.get("origins"), "frame_attribution.origins");
        const proven_origins = [_][]const u8{
            "allocator_spill",
            "wasm_local_or_phi",
            "explicit_frame_storage",
            "fixed_runtime_frame_state",
        };
        var oit = origins.iterator();
        while (oit.next()) |entry| {
            var ok = std.mem.eql(u8, entry.key_ptr.*, "unknown");
            for (proven_origins) |po| {
                if (std.mem.eql(u8, entry.key_ptr.*, po)) ok = true;
            }
            if (!ok) return fail("unrecognized frame origin in attribution", .{});
        }
        var proven_samples: i64 = 0;
        for (proven_origins) |po| {
            for ([_][]const u8{ "load", "store" }) |kind| {
                const cname = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ po, kind });
                proven_samples += try samplesFor(classes, cname);
            }
        }
        var frame_samples_match = true;
        if (coverage.get("frame_samples")) |fs_val| {
            const fs = try requireInteger(fs_val, "frame_attribution.coverage.frame_samples", 0);
            if (proven_samples > fs) return fail("proven frame class samples exceed frame coverage", .{});
            if (proven_samples != fs) frame_samples_match = false;
        }
        const oc_ok = if (coverage.get("origin_coverage_pct")) |v| (try requireNumber(v, "origin_coverage_pct", 0.0)) == 100.0 else false;
        const osc_ok = if (coverage.get("origin_sample_coverage_pct")) |v| (try requireNumber(v, "origin_sample_coverage_pct", 0.0)) == 100.0 else false;
        const ufi_ok = if (coverage.get("unknown_frame_instructions")) |v| (try requireInteger(v, "unknown_frame_instructions", 0)) == 0 else false;
        const ufs_ok = if (coverage.get("unknown_frame_samples")) |v| (try requireInteger(v, "unknown_frame_samples", 0)) == 0 else false;

        for ([_][2][]const u8{ .{ "frame_loads", "load" }, .{ "frame_stores", "store" } }) |pair| {
            const metric = pair[0];
            const kind = pair[1];
            var count: i64 = 0;
            for (proven_origins) |po| {
                const cname = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ po, kind });
                count += try samplesFor(classes, cname);
            }
            const suffix = try std.fmt.allocPrint(allocator, "_{s}", .{kind});
            var has_unattributed = false;
            var cl_it = classes.iterator();
            while (cl_it.next()) |entry| {
                const k = entry.key_ptr.*;
                if (std.mem.endsWith(u8, k, suffix)) {
                    var in_proven = false;
                    for (proven_origins) |po| {
                        const cname = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ po, kind });
                        if (std.mem.eql(u8, k, cname)) in_proven = true;
                    }
                    if (!in_proven and (std.mem.startsWith(u8, k, "frame_") or std.mem.startsWith(u8, k, "unknown_frame_") or std.mem.startsWith(u8, k, "unknown_"))) {
                        has_unattributed = true;
                    }
                }
            }
            if (!oc_ok or !osc_ok or !ufi_ok or !ufs_ok or !frame_samples_match or has_unattributed) {
                try dynamic.put(allocator, metric, try unavailableMetric(allocator, "frame sidecar has ambiguous or unknown frame origins"));
            } else {
                try dynamic.put(allocator, metric, try sampleMetric(allocator, count, function_samples, total));
            }
        }
    }

    try dynamic.put(allocator, "address_generation", try unavailableMetric(
        allocator,
        "aot_jit_attr groups LEA with ALU, so address-generation samples cannot be separated without changing the concurrently-owned attribution tool",
    ));
    try dynamic.put(allocator, "branches", try unavailableMetric(
        allocator,
        "aot_jit_attr combines bounds compares with bounds branches, so exact branch-only samples are unavailable",
    ));

    var hotness: ObjectMap = .empty;
    try hotness.put(allocator, "status", .{ .string = "measured" });
    try hotness.put(allocator, "source", .{ .string = "WAMR perf self samples" });
    try hotness.put(allocator, "total_samples", .{ .integer = total });
    try hotness.put(allocator, "function_samples", .{ .integer = function_samples });
    try hotness.put(allocator, "percent_of_run", .{ .float = share });

    return .{ .dynamic = dynamic, .hotness = hotness };
}

pub fn loadExplicitProfile(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    expected_sha256: []const u8,
    engine: []const u8,
    module_sha256: []const u8,
    wasm_index: usize,
    function_name: ?[]const u8,
) !ObjectMap {
    const actual_sha = try sha256File(allocator, io, path);
    if (!std.mem.eql(u8, actual_sha, expected_sha256)) {
        return fail("{s} profile SHA-256 mismatch: expected {s}, got {s}", .{ engine, expected_sha256, actual_sha });
    }
    const profile_label = try std.fmt.allocPrint(allocator, "{s} profile", .{engine});
    const profile = try loadJsonObject(allocator, io, path, profile_label);
    const sv = profile.get("schema_version") orelse return fail("{s} profile has an unsupported schema or kind", .{engine});
    const kind = profile.get("kind") orelse return fail("{s} profile has an unsupported schema or kind", .{engine});
    if (sv != .integer or sv.integer != 1 or kind != .string or !std.mem.eql(u8, kind.string, "wasm-hot-function-profile")) {
        return fail("{s} profile has an unsupported schema or kind", .{engine});
    }
    const eng = profile.get("engine") orelse return fail("profile engine mismatch", .{});
    if (eng != .string or !std.mem.eql(u8, eng.string, engine)) {
        return fail("profile engine mismatch: expected '{s}'", .{engine});
    }
    const mv = profile.get("mapping_verified") orelse return fail("{s} profile must set mapping_verified=true", .{engine});
    if (mv != .bool or !mv.bool) return fail("{s} profile must set mapping_verified=true", .{engine});
    _ = try requireString(profile.get("mapping_evidence"), "profile.mapping_evidence");
    const mod_sha = try requireSha256(allocator, profile.get("module_sha256"), "profile.module_sha256");
    if (!std.mem.eql(u8, mod_sha, module_sha256)) {
        return fail("{s} profile module identity does not match the core wasm", .{engine});
    }
    const prof_widx: usize = @intCast(try requireInteger(profile.get("wasm_function_index"), "profile.wasm_function_index", 0));
    if (prof_widx != wasm_index) {
        return fail("{s} profile function index does not match", .{engine});
    }
    const fn_val = profile.get("function_name");
    if (function_name) |expected_fn| {
        if (fn_val == null or fn_val.? != .string or !std.mem.eql(u8, fn_val.?.string, expected_fn)) {
            return fail("{s} profile function name does not match", .{engine});
        }
    } else {
        if (fn_val != null and fn_val.? != .null) {
            return fail("{s} profile function name does not match", .{engine});
        }
    }
    const total = try requireInteger(profile.get("total_samples"), "profile.total_samples", 1);
    const function_samples = try requireInteger(profile.get("function_samples"), "profile.function_samples", 1);
    if (function_samples > total) {
        return fail("profile.function_samples cannot exceed total_samples", .{});
    }
    const metrics = try requireObject(profile.get("metrics"), "profile.metrics");
    var result: ObjectMap = .empty;
    try result.put(allocator, "native_instructions", try sampleMetric(allocator, function_samples, function_samples, total));
    try result.put(allocator, "code_size_bytes", try unavailableMetric(allocator, "this metric is static metadata, not an instruction class"));
    try result.put(allocator, "frame_size_bytes", try unavailableMetric(allocator, "this metric is static metadata, not an instruction class"));
    for (METRIC_ORDER) |name| {
        if (!isGateMetric(name)) continue;
        const item_opt = metrics.get(name);
        if (item_opt == null or item_opt.? == .null) {
            try result.put(allocator, name, try unavailableMetric(allocator, "metric omitted by explicit profile"));
            continue;
        }
        const value = try requireObject(item_opt, "profile.metrics");
        const status = try requireString(value.get("status"), "profile.metrics.status");
        if (std.mem.eql(u8, status, "measured")) {
            const s = try requireInteger(value.get("samples"), "profile.metrics.samples", 0);
            try result.put(allocator, name, try sampleMetric(allocator, s, function_samples, total));
        } else if (std.mem.eql(u8, status, "unavailable") or std.mem.eql(u8, status, "partial")) {
            const reason = try requireString(value.get("reason"), "profile.metrics.reason");
            var m: ObjectMap = .empty;
            try m.put(allocator, "status", .{ .string = status });
            try m.put(allocator, "reason", .{ .string = reason });
            try result.put(allocator, name, .{ .object = m });
        } else {
            return fail("profile.metrics.{s}.status must be measured, partial, or unavailable", .{name});
        }
    }
    return result;
}

pub const JumpTableRange = struct {
    start: usize,
    end: usize,
    entries: usize,
};

pub fn findWamrJumpTables(allocator: std.mem.Allocator, code: []const u8) ![]JumpTableRange {
    const dispatch_tail = [_]u8{ 0x4F, 0x63, 0x1C, 0x9A, 0x4D, 0x01, 0xDA, 0x41, 0xFF, 0xE2 };
    var ranges: std.ArrayList(JumpTableRange) = .empty;
    var search_from: usize = 0;
    while (std.mem.indexOfPos(u8, code, search_from, &dispatch_tail)) |tail| {
        search_from = tail + 1;
        var inside_existing = false;
        for (ranges.items) |table| {
            if (table.start <= tail and tail < table.end) {
                inside_existing = true;
                break;
            }
        }
        if (inside_existing) continue;
        if (tail < 13) return fail("WAMR br_table dispatch tail lacks its expected RIP-relative LEA", .{});
        const lea = tail - 7;
        if (!std.mem.eql(u8, code[lea .. lea + 3], &[_]u8{ 0x4C, 0x8D, 0x15 })) {
            return fail("WAMR br_table dispatch tail lacks its expected RIP-relative LEA", .{});
        }
        const jae = lea - 6;
        if (!std.mem.eql(u8, code[jae .. jae + 2], &[_]u8{ 0x0F, 0x83 })) {
            return fail("WAMR br_table dispatch lacks its expected unsigned bounds branch", .{});
        }
        var count_opt: ?usize = null;
        if (jae >= 4 and std.mem.eql(u8, code[jae - 4 .. jae - 1], &[_]u8{ 0x49, 0x83, 0xFB })) {
            count_opt = code[jae - 1];
        } else if (jae >= 7 and std.mem.eql(u8, code[jae - 7 .. jae - 4], &[_]u8{ 0x49, 0x81, 0xFB })) {
            count_opt = std.mem.readInt(u32, code[jae - 4 ..][0..4], .little);
        }
        const count = count_opt orelse return fail("WAMR br_table dispatch target count could not be recovered", .{});
        const displacement = std.mem.readInt(i32, code[lea + 3 ..][0..4], .little);
        const table_start_signed: i64 = @as(i64, @intCast(lea + 7)) + displacement;
        const dispatch_end = tail + dispatch_tail.len;
        if (table_start_signed != @as(i64, @intCast(dispatch_end))) {
            return fail("WAMR br_table LEA does not point immediately after the dispatch", .{});
        }
        const table_start: usize = dispatch_end;
        if (count == 0) continue;
        const table_end = table_start + count * 4;
        if (table_end > code.len) {
            return fail("WAMR br_table extends beyond the function", .{});
        }
        var offset: usize = table_start;
        while (offset < table_end) : (offset += 4) {
            const rel = std.mem.readInt(i32, code[offset..][0..4], .little);
            const target: i64 = @as(i64, @intCast(table_start)) + rel;
            if (target < 0 or target >= @as(i64, @intCast(code.len))) {
                return fail("WAMR br_table contains an out-of-function target", .{});
            }
        }
        try ranges.append(allocator, .{
            .start = table_start,
            .end = table_end,
            .entries = count,
        });
    }
    std.mem.sort(JumpTableRange, ranges.items, {}, struct {
        fn lessThan(_: void, a: JumpTableRange, b: JumpTableRange) bool {
            return a.start < b.start;
        }
    }.lessThan);
    if (ranges.items.len > 1) {
        for (1..ranges.items.len) |i| {
            if (ranges.items[i - 1].end > ranges.items[i].start) {
                return fail("overlapping WAMR inline jump tables", .{});
            }
        }
    }
    return ranges.toOwnedSlice(allocator);
}

pub const CaptureConfig = struct {
    benchmark_report_path: []const u8,
    benchmark_report_sha256: []const u8,
    core_wasm_path: []const u8,
    core_wasm_sha256: []const u8,
    wasm_index: ?usize,
    function_name: ?[]const u8,
    wamr_manifest_path: []const u8,
    wamr_cwasm_path: []const u8,
    wamr_local_func: usize,
    wasmtime_profile_path: ?[]const u8,
    wasmtime_profile_sha256: ?[]const u8,
};

fn resolvePathAndSha(
    allocator: std.mem.Allocator,
    io: std.Io,
    base: []const u8,
    value: ?Value,
    label: []const u8,
) !struct { path: []const u8, sha256: []const u8 } {
    const obj = try requireObject(value, label);
    const path = try resolvePath(allocator, io, base, obj.get("path"), try std.fmt.allocPrint(allocator, "{s}.path", .{label}));
    const expected_sha = try requireSha256(allocator, obj.get("sha256"), try std.fmt.allocPrint(allocator, "{s}.sha256", .{label}));
    const actual_sha = try sha256File(allocator, io, path);
    if (!std.mem.eql(u8, actual_sha, expected_sha)) {
        return fail("{s} SHA-256 mismatch for {s}: expected {s}, got {s}", .{ label, path, expected_sha, actual_sha });
    }
    return .{ .path = path, .sha256 = actual_sha };
}

pub fn loadCaptureConfig(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !CaptureConfig {
    const resolved_config = if (std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator)) |r|
        r
    else |_|
        try std.fs.path.resolve(allocator, &.{path});
    const base = std.fs.path.dirname(resolved_config) orelse ".";
    const raw = try loadJsonObject(allocator, io, resolved_config, "capture config");
    const sv = try requireInteger(raw.get("schema_version"), "schema_version", 1);
    if (sv != CONFIG_SCHEMA_VERSION) {
        return fail("capture config schema_version must be {d}", .{CONFIG_SCHEMA_VERSION});
    }
    const bench = try resolvePathAndSha(allocator, io, base, raw.get("benchmark_report"), "benchmark_report");
    const core = try resolvePathAndSha(allocator, io, base, raw.get("core_wasm"), "core_wasm");

    const fn_raw = try requireObject(raw.get("function"), "function");
    const wasm_index: ?usize = if (fn_raw.get("wasm_index")) |wi|
        if (wi != .null) @as(usize, @intCast(try requireInteger(wi, "function.wasm_index", 0))) else null
    else
        null;
    const function_name: ?[]const u8 = if (fn_raw.get("name")) |nm|
        if (nm != .null) try requireString(nm, "function.name") else null
    else
        null;
    if (wasm_index == null and function_name == null) {
        return fail("function must specify wasm_index, name, or both", .{});
    }

    const wamr_raw = try requireObject(raw.get("wamr"), "wamr");
    const wamr_manifest = try resolvePath(allocator, io, base, wamr_raw.get("manifest"), "wamr.manifest");
    const wamr_cwasm = try resolvePath(allocator, io, base, wamr_raw.get("cwasm"), "wamr.cwasm");
    const wamr_local_func: usize = @intCast(try requireInteger(wamr_raw.get("local_func"), "wamr.local_func", 0));

    const wt_raw = try requireObject(raw.get("wasmtime"), "wasmtime");
    var wt_profile_path: ?[]const u8 = null;
    var wt_profile_sha: ?[]const u8 = null;
    if (wt_raw.get("profile")) |prof| {
        if (prof != .null) {
            const p = try resolvePathAndSha(allocator, io, base, prof, "wasmtime.profile");
            wt_profile_path = p.path;
            wt_profile_sha = p.sha256;
        }
    }

    return .{
        .benchmark_report_path = bench.path,
        .benchmark_report_sha256 = bench.sha256,
        .core_wasm_path = core.path,
        .core_wasm_sha256 = core.sha256,
        .wasm_index = wasm_index,
        .function_name = function_name,
        .wamr_manifest_path = wamr_manifest,
        .wamr_cwasm_path = wamr_cwasm,
        .wamr_local_func = wamr_local_func,
        .wasmtime_profile_path = wt_profile_path,
        .wasmtime_profile_sha256 = wt_profile_sha,
    };
}

fn artifactHash(allocator: std.mem.Allocator, io: std.Io, report: ObjectMap, path: []const u8) ![]const u8 {
    const precompile = try requireObject(report.get("precompile"), "benchmark_report.precompile");
    const artifacts = try requireObject(precompile.get("artifacts"), "benchmark_report.precompile.artifacts");
    const wamr_art = try requireObject(artifacts.get("wamr"), "benchmark_report.precompile.artifacts.wamr");
    const files_val = wamr_art.get("files") orelse Value.null;
    if (files_val != .array) {
        return fail("benchmark_report.precompile.artifacts.wamr.files must be an array", .{});
    }
    const resolved_target = if (std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator)) |r|
        r
    else |_|
        try std.fs.path.resolve(allocator, &.{path});
    for (files_val.array.items) |entry_val| {
        if (entry_val != .object) continue;
        const ep = entry_val.object.get("path") orelse continue;
        if (ep != .string) continue;
        const resolved_ep = if (std.Io.Dir.cwd().realPathFileAlloc(io, ep.string, allocator)) |r|
            r
        else |_|
            try std.fs.path.resolve(allocator, &.{ep.string});
        if (std.mem.eql(u8, resolved_ep, resolved_target)) {
            return requireSha256(allocator, entry_val.object.get("sha256"), "benchmark_report artifact sha256");
        }
    }
    return fail("benchmark_report precompile artifact list does not include {s}", .{path});
}

fn jsonValueEql(a: Value, b: Value) bool {
    switch (a) {
        .null => return b == .null,
        .bool => |av| return b == .bool and av == b.bool,
        .integer => |av| return switch (b) {
            .integer => |bv| av == bv,
            .float => |bv| @as(f64, @floatFromInt(av)) == bv,
            else => false,
        },
        .float => |av| return switch (b) {
            .float => |bv| av == bv,
            .integer => |bv| av == @as(f64, @floatFromInt(bv)),
            else => false,
        },
        .string => |av| return b == .string and std.mem.eql(u8, av, b.string),
        .array => |av| {
            if (b != .array or av.items.len != b.array.items.len) return false;
            for (av.items, b.array.items) |ai, bi| {
                if (!jsonValueEql(ai, bi)) return false;
            }
            return true;
        },
        .object => |av| {
            if (b != .object or av.count() != b.object.count()) return false;
            var it = av.iterator();
            while (it.next()) |kv| {
                const bv = b.object.get(kv.key_ptr.*) orelse return false;
                if (!jsonValueEql(kv.value_ptr.*, bv)) return false;
            }
            return true;
        },
        .number_string => |av| return b == .number_string and std.mem.eql(u8, av, b.number_string),
    }
}

pub fn validateBenchmark(
    allocator: std.mem.Allocator,
    io: std.Io,
    config: CaptureConfig,
) !struct { report: ObjectMap, attribution: ObjectMap, component_sha: []const u8 } {
    const report = try loadJsonObject(allocator, io, config.benchmark_report_path, "benchmark_report");
    if ((try requireInteger(report.get("schema_version"), "benchmark_report.schema_version", 1)) != 1) {
        return fail("benchmark_report.schema_version must be 1", .{});
    }
    const kind = try requireString(report.get("kind"), "benchmark_report.kind");
    if (!std.mem.eql(u8, kind, "keyvault-tcgc-aot-benchmark")) {
        return fail("benchmark_report.kind must be 'keyvault-tcgc-aot-benchmark'", .{});
    }
    const validation = try requireObject(report.get("validation"), "benchmark_report.validation");
    const inputs = try requireObject(validation.get("inputs"), "benchmark_report.validation.inputs");
    const component = try requireObject(inputs.get("component"), "benchmark_report.validation.inputs.component");
    const component_sha = try requireSha256(allocator, component.get("sha256"), "benchmark_report.validation.inputs.component.sha256");

    const perf = try requireObject(report.get("perf"), "benchmark_report.perf");
    const selected = perf.get("selected") orelse Value.null;
    if (selected != .bool or !selected.bool) {
        return fail("benchmark_report.perf.selected must be true", .{});
    }
    const attribution = try requireObject(perf.get("attribution"), "benchmark_report.perf.attribution");
    const classified = try requireObject(attribution.get("classified_function"), "benchmark_report.perf.attribution.classified_function");

    if (classified.get("frame_attribution")) |fa| {
        if (fa != .null) {
            const precompile = try requireObject(report.get("precompile"), "benchmark_report.precompile");
            const perf_frame = try requireObject(perf.get("frame_attribution"), "benchmark_report.perf.frame_attribution");
            const precompile_frame = try requireObject(precompile.get("frame_attribution"), "benchmark_report.precompile.frame_attribution");
            if (!jsonValueEql(.{ .object = perf_frame }, .{ .object = precompile_frame })) {
                return fail("benchmark_report perf.frame_attribution does not match precompile.frame_attribution", .{});
            }
            const enabled = perf_frame.get("enabled") orelse Value.null;
            if (enabled != .bool or !enabled.bool) {
                return fail("benchmark_report frame_attribution.enabled must be true", .{});
            }
            const cwasm_path = try resolvePath(allocator, io, "/", perf_frame.get("cwasm"), "perf.frame_attribution.cwasm");
            if (!std.mem.eql(u8, cwasm_path, config.wamr_cwasm_path)) {
                return fail("perf.frame_attribution.cwasm does not match wamr.cwasm", .{});
            }
            const cwasm_sha = try requireSha256(allocator, perf_frame.get("cwasm_sha256"), "perf.frame_attribution.cwasm_sha256");
            if (!std.mem.eql(u8, try sha256File(allocator, io, config.wamr_cwasm_path), cwasm_sha)) {
                return fail("perf.frame_attribution.cwasm_sha256 does not match wamr.cwasm", .{});
            }
            const sidecar_path = try resolvePath(allocator, io, "/", perf_frame.get("sidecar"), "perf.frame_attribution.sidecar");
            const sidecar_sha = try requireSha256(allocator, perf_frame.get("sidecar_sha256"), "perf.frame_attribution.sidecar_sha256");
            if (!std.mem.eql(u8, try sha256File(allocator, io, sidecar_path), sidecar_sha)) {
                return fail("perf.frame_attribution.sidecar_sha256 does not match sidecar file", .{});
            }
            if (!std.mem.eql(u8, try artifactHash(allocator, io, report, sidecar_path), sidecar_sha)) {
                return fail("benchmark_report sidecar artifact SHA-256 mismatch", .{});
            }
        }
    }
    return .{ .report = report, .attribution = attribution, .component_sha = component_sha };
}

pub fn validateWamrIdentity(
    allocator: std.mem.Allocator,
    io: std.Io,
    config: CaptureConfig,
    report: ObjectMap,
    attribution: ObjectMap,
    component_sha: []const u8,
    core_identity: WasmModuleIdentity,
    wasm_index: usize,
) !struct { cwasm_sha256: []const u8, manifest_sha256: []const u8 } {
    const manifest = try loadJsonObject(allocator, io, config.wamr_manifest_path, "wamr.manifest");
    if ((try requireInteger(manifest.get("version"), "wamr.manifest.version", 1)) != 2) {
        return fail("wamr.manifest.version must be 2", .{});
    }
    const manifest_component = try requireSha256(allocator, manifest.get("component_sha256"), "wamr.manifest.component_sha256");
    if (!std.mem.eql(u8, manifest_component, component_sha)) {
        return fail("wamr.manifest component_sha256 does not match benchmark component", .{});
    }
    const modules_val = manifest.get("modules") orelse Value.null;
    if (modules_val != .array or modules_val.array.items.len == 0) {
        return fail("wamr.manifest.modules must be a non-empty array", .{});
    }
    var matched_entries: std.ArrayList(ObjectMap) = .empty;
    for (modules_val.array.items) |entry_val| {
        const entry = try requireObject(entry_val, "wamr.manifest.modules[]");
        const cwasm_entry = try resolvePath(allocator, io, std.fs.path.dirname(config.wamr_manifest_path) orelse ".", entry.get("cwasm"), "wamr.manifest.modules[].cwasm");
        if (std.mem.eql(u8, cwasm_entry, config.wamr_cwasm_path)) {
            try matched_entries.append(allocator, entry);
        }
    }
    if (matched_entries.items.len != 1) {
        return fail("wamr.manifest must contain a unique module entry for {s}", .{config.wamr_cwasm_path});
    }
    const entry = matched_entries.items[0];
    const core_sha = try requireSha256(allocator, entry.get("core_sha256"), "wamr.manifest.modules[].core_sha256");
    if (!std.mem.eql(u8, core_sha, core_identity.sha256)) {
        return fail("core_wasm SHA-256 ({s}) does not match WAMR manifest core_sha256 ({s})", .{ core_identity.sha256, core_sha });
    }
    const manifest_cwasm_sha = try requireSha256(allocator, entry.get("sha256"), "wamr.manifest.modules[].sha256");
    const actual_cwasm_sha = try sha256File(allocator, io, config.wamr_cwasm_path);
    if (!std.mem.eql(u8, actual_cwasm_sha, manifest_cwasm_sha)) {
        return fail("wamr.cwasm SHA-256 ({s}) does not match WAMR manifest sha256 ({s})", .{ actual_cwasm_sha, manifest_cwasm_sha });
    }
    if (!std.mem.eql(u8, try artifactHash(allocator, io, report, config.wamr_cwasm_path), actual_cwasm_sha)) {
        return fail("wamr.cwasm SHA-256 does not match benchmark_report precompile artifact", .{});
    }
    const manifest_sha = try sha256File(allocator, io, config.wamr_manifest_path);
    if (!std.mem.eql(u8, try artifactHash(allocator, io, report, config.wamr_manifest_path), manifest_sha)) {
        return fail("wamr.manifest SHA-256 does not match benchmark_report precompile artifact", .{});
    }

    const attr_cwasm = try resolvePath(allocator, io, "/", attribution.get("cwasm"), "perf.attribution.cwasm");
    if (!std.mem.eql(u8, attr_cwasm, config.wamr_cwasm_path)) {
        return fail("perf.attribution.cwasm ({s}) does not match wamr.cwasm ({s})", .{ attr_cwasm, config.wamr_cwasm_path });
    }
    const classified = try requireObject(attribution.get("classified_function"), "perf.attribution.classified_function");
    const classified_local: usize = @intCast(try requireInteger(classified.get("local_func"), "perf.attribution.classified_function.local_func", 0));
    if (classified_local != config.wamr_local_func) {
        return fail("perf.attribution classified_function.local_func ({d}) does not match wamr.local_func ({d})", .{ classified_local, config.wamr_local_func });
    }
    const expected_wasm_index = core_identity.imported_function_count + config.wamr_local_func;
    if (expected_wasm_index != wasm_index) {
        return fail("WAMR local_func {d} plus {d} imported functions maps to wasm function {d}, not configured wasm_index {d}", .{
            config.wamr_local_func,
            core_identity.imported_function_count,
            expected_wasm_index,
            wasm_index,
        });
    }
    return .{ .cwasm_sha256 = actual_cwasm_sha, .manifest_sha256 = manifest_sha };
}

fn runCommand(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8, label: []const u8) !struct { stdout: []const u8, stderr: []const u8 } {
    const res = std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .limited(32 * 1024 * 1024),
        .stderr_limit = .limited(4 * 1024 * 1024),
    }) catch |err| {
        return fail("{s} could not execute {s}: {s}", .{ label, argv[0], @errorName(err) });
    };
    switch (res.term) {
        .exited => |code| if (code != 0) {
            const detail = std.mem.trim(u8, if (res.stderr.len > 0) res.stderr else res.stdout, " \t\r\n");
            return fail("{s} failed (exit {d}): {s}", .{ label, code, detail });
        },
        else => return fail("{s} terminated abnormally", .{label}),
    }
    return .{ .stdout = res.stdout, .stderr = res.stderr };
}

pub fn disassembleWamr(
    allocator: std.mem.Allocator,
    io: std.Io,
    function: CwasmFunction,
    work_dir: []const u8,
) !struct { instructions: []Instruction, jump_tables: []JumpTableRange } {
    if (builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) {
        return fail("native hot-function capture currently requires Linux x86_64", .{});
    }
    const jump_tables = try findWamrJumpTables(allocator, function.code);
    const masked = try allocator.dupe(u8, function.code);
    for (jump_tables) |table| {
        @memset(masked[table.start..table.end], 0x90);
    }
    std.Io.Dir.cwd().createDirPath(io, work_dir) catch {};
    const scratch = try std.fs.path.join(allocator, &.{ work_dir, ".wamr-hot-function.bin" });
    defer std.Io.Dir.cwd().deleteFile(io, scratch) catch {};
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = scratch, .data = masked }) catch {
        return fail("cannot write WAMR function scratch binary: {s}", .{scratch});
    };
    const proc = try runCommand(allocator, io, &.{
        "objdump",
        "-D",
        "-b",
        "binary",
        "-m",
        "i386:x86-64",
        "-M",
        "intel",
        "--adjust-vma=0",
        scratch,
    }, "objdump WAMR function");
    const all_instructions = try parseDisassembly(allocator, proc.stdout, null);
    var covered: usize = 0;
    for (all_instructions) |inst| covered += inst.size;
    if (covered != function.code_size) {
        return fail("WAMR disassembly covered {d} bytes, expected {d}", .{ covered, function.code_size });
    }
    var instructions: std.ArrayList(Instruction) = .empty;
    for (all_instructions) |inst| {
        var in_table = false;
        for (jump_tables) |table| {
            if (inst.offset >= table.start and inst.offset < table.end) {
                in_table = true;
                break;
            }
        }
        if (!in_table) try instructions.append(allocator, inst);
    }
    if (instructions.items.len == 0) {
        return fail("WAMR disassembly contained only inline jump-table data", .{});
    }
    return .{
        .instructions = try instructions.toOwnedSlice(allocator),
        .jump_tables = jump_tables,
    };
}

fn deriveWasmtimeCompileCommand(
    allocator: std.mem.Allocator,
    io: std.Io,
    report: ObjectMap,
    core_wasm: []const u8,
    output_cwasm: []const u8,
) !struct { cmd: []const []const u8, tool_path: []const u8, tool_sha256: []const u8, expected_version: []const u8 } {
    const validation = try requireObject(report.get("validation"), "benchmark_report.validation");
    const tools = try requireObject(validation.get("tools"), "benchmark_report.validation.tools");
    const wt_tool = try requireObject(tools.get("wasmtime"), "benchmark_report.validation.tools.wasmtime");
    const tool_path = try resolvePath(allocator, io, "/", wt_tool.get("path"), "validation.tools.wasmtime.path");
    const expected_tool_sha = try requireSha256(allocator, wt_tool.get("sha256"), "validation.tools.wasmtime.sha256");
    const actual_tool_sha = try sha256File(allocator, io, tool_path);
    if (!std.mem.eql(u8, actual_tool_sha, expected_tool_sha)) {
        return fail("wasmtime binary SHA-256 mismatch: expected {s}, got {s}", .{ expected_tool_sha, actual_tool_sha });
    }
    const expected_version = try requireString(wt_tool.get("version"), "validation.tools.wasmtime.version");
    const ver_proc = try runCommand(allocator, io, &.{ tool_path, "--version" }, "wasmtime --version");
    const actual_version = std.mem.trim(u8, if (ver_proc.stdout.len > 0) ver_proc.stdout else ver_proc.stderr, " \t\r\n");
    if (!std.mem.eql(u8, actual_version, expected_version)) {
        return fail("wasmtime version mismatch: expected '{s}', got '{s}'", .{ expected_version, actual_version });
    }

    const inputs = try requireObject(validation.get("inputs"), "benchmark_report.validation.inputs");
    const component = try requireObject(inputs.get("component"), "benchmark_report.validation.inputs.component");
    const component_path = try requireString(component.get("path"), "validation.inputs.component.path");
    const resolved_component = if (std.Io.Dir.cwd().realPathFileAlloc(io, component_path, allocator)) |r|
        r
    else |_|
        try std.fs.path.resolve(allocator, &.{component_path});

    const precompile = try requireObject(report.get("precompile"), "benchmark_report.precompile");
    const commands = try requireObject(precompile.get("commands"), "benchmark_report.precompile.commands");
    const raw_cmd_val = commands.get("wasmtime") orelse Value.null;
    if (raw_cmd_val != .array or raw_cmd_val.array.items.len < 4) {
        return fail("benchmark_report.precompile.commands.wasmtime must be an command array", .{});
    }
    var cmd: std.ArrayList([]const u8) = .empty;
    for (raw_cmd_val.array.items) |item| {
        if (item != .string) return fail("benchmark_report.precompile.commands.wasmtime must be an command array", .{});
        try cmd.append(allocator, item.string);
    }
    const resolved_cmd0 = if (std.Io.Dir.cwd().realPathFileAlloc(io, cmd.items[0], allocator)) |r|
        r
    else |_|
        try std.fs.path.resolve(allocator, &.{cmd.items[0]});
    if (!std.mem.eql(u8, resolved_cmd0, tool_path) or !std.mem.eql(u8, cmd.items[1], "compile")) {
        return fail("unexpected Wasmtime precompile command shape in benchmark report", .{});
    }

    var out: std.ArrayList([]const u8) = .empty;
    try out.append(allocator, tool_path);
    try out.append(allocator, "compile");
    var replaced_output = false;
    var replaced_input = false;
    var idx: usize = 2;
    while (idx < cmd.items.len) : (idx += 1) {
        const token = cmd.items[idx];
        if (std.mem.eql(u8, token, "-o") or std.mem.eql(u8, token, "--output")) {
            if (idx + 1 >= cmd.items.len) return fail("Wasmtime precompile command has dangling -o", .{});
            try out.append(allocator, token);
            try out.append(allocator, output_cwasm);
            replaced_output = true;
            idx += 1;
            continue;
        }
        if (std.mem.startsWith(u8, token, "--output=")) {
            try out.append(allocator, try std.fmt.allocPrint(allocator, "--output={s}", .{output_cwasm}));
            replaced_output = true;
            continue;
        }
        const resolved_token = if (std.Io.Dir.cwd().realPathFileAlloc(io, token, allocator)) |r|
            r
        else |_|
            (std.fs.path.resolve(allocator, &.{token}) catch token);
        if (std.mem.eql(u8, token, component_path) or std.mem.eql(u8, resolved_token, resolved_component)) {
            try out.append(allocator, core_wasm);
            replaced_input = true;
            continue;
        }
        try out.append(allocator, token);
    }
    if (!replaced_output or !replaced_input) {
        return fail("could not derive core-wasm Wasmtime compile command from benchmark report", .{});
    }
    return .{
        .cmd = try out.toOwnedSlice(allocator),
        .tool_path = tool_path,
        .tool_sha256 = actual_tool_sha,
        .expected_version = expected_version,
    };
}

fn wasmtimeExploreCommand(allocator: std.mem.Allocator, compile_command: []const []const u8, explore_output: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    try out.append(allocator, compile_command[0]);
    try out.append(allocator, "explore");
    var idx: usize = 2;
    while (idx < compile_command.len) : (idx += 1) {
        const token = compile_command[idx];
        if (std.mem.eql(u8, token, "-o") or std.mem.eql(u8, token, "--output")) {
            try out.append(allocator, "-o");
            try out.append(allocator, explore_output);
            idx += 1;
            continue;
        }
        if (std.mem.startsWith(u8, token, "--output=")) {
            try out.append(allocator, try std.fmt.allocPrint(allocator, "--output={s}", .{explore_output}));
            continue;
        }
        try out.append(allocator, token);
    }
    return out.toOwnedSlice(allocator);
}

pub fn captureWasmtime(
    allocator: std.mem.Allocator,
    io: std.Io,
    report: ObjectMap,
    core_wasm: []const u8,
    wasm_index: usize,
    work_dir: []const u8,
) !struct { instructions: []Instruction, code_size: usize, provenance: ObjectMap } {
    std.Io.Dir.cwd().createDirPath(io, work_dir) catch {};
    const cwasm_path = try std.fs.path.join(allocator, &.{ work_dir, "wasmtime-hot-core.cwasm" });
    const explore_path = try std.fs.path.join(allocator, &.{ work_dir, "wasmtime-hot-core.explore.html" });
    const derived = try deriveWasmtimeCompileCommand(allocator, io, report, core_wasm, cwasm_path);
    _ = try runCommand(allocator, io, derived.cmd, "wasmtime compile core wasm");
    const cwasm_sha = try sha256File(allocator, io, cwasm_path);

    const filter_arg = try std.fmt.allocPrint(allocator, "wasm[0]::function[{d}]", .{wasm_index});
    const objdump_cmd = [_][]const u8{
        derived.tool_path,
        "objdump",
        cwasm_path,
        "--addresses",
        "--bytes",
        "--color",
        "never",
        "--addrmap=false",
        "--traps=false",
        "--stack-maps=false",
        "--exception-tables=false",
        "--frame-tables=false",
        "--funcs",
        "wasm",
        "--filter",
        filter_arg,
    };
    const objdump_proc = try runCommand(allocator, io, &objdump_cmd, "wasmtime objdump");
    const objdump_instructions = try parseDisassembly(allocator, objdump_proc.stdout, wasm_index);

    const explore_cmd = try wasmtimeExploreCommand(allocator, derived.cmd, explore_path);
    _ = try runCommand(allocator, io, explore_cmd, "wasmtime explore core wasm");
    const explore_sha = try sha256File(allocator, io, explore_path);
    const explore_res = try parseWasmtimeExplore(allocator, io, explore_path, wasm_index);
    const explore_body = explore_res.body;

    if (explore_body.len > objdump_instructions.len) {
        return fail("wasmtime explore body exceeds objdump instruction count", .{});
    }
    var objdump_bytes: std.ArrayList(u8) = .empty;
    for (objdump_instructions[0..explore_body.len]) |inst| try objdump_bytes.appendSlice(allocator, inst.raw_bytes);
    var explore_bytes: std.ArrayList(u8) = .empty;
    for (explore_body) |inst| try explore_bytes.appendSlice(allocator, inst.raw_bytes);
    if (!std.mem.eql(u8, objdump_bytes.items, explore_bytes.items)) {
        return fail("wasmtime objdump and explore instruction bytes do not match", .{});
    }

    const body_instructions = objdump_instructions[0..explore_body.len];
    var code_size: usize = 0;
    for (body_instructions) |inst| code_size += inst.size;
    var full_bytes: usize = 0;
    for (objdump_instructions) |inst| full_bytes += inst.size;
    const trailing_excluded_bytes = full_bytes - code_size;
    const trailing_excluded_records = objdump_instructions.len - body_instructions.len;

    var compile_arr = Array.init(allocator);
    for (derived.cmd) |tok| try compile_arr.append(.{ .string = tok });
    var objdump_arr = Array.init(allocator);
    for (objdump_cmd) |tok| try objdump_arr.append(.{ .string = tok });
    var explore_arr = Array.init(allocator);
    for (explore_cmd) |tok| try explore_arr.append(.{ .string = tok });

    var const_pool: ObjectMap = .empty;
    try const_pool.put(allocator, "method", .{ .string = "wasmtime explore wasm_offset body boundary through final terminator" });
    try const_pool.put(allocator, "full_symbol_bytes", .{ .integer = @intCast(full_bytes) });
    try const_pool.put(allocator, "executable_body_bytes", .{ .integer = @intCast(code_size) });
    try const_pool.put(allocator, "excluded_trailing_bytes", .{ .integer = @intCast(trailing_excluded_bytes) });
    try const_pool.put(allocator, "excluded_trailing_disassembly_records", .{ .integer = @intCast(trailing_excluded_records) });

    var provenance: ObjectMap = .empty;
    try provenance.put(allocator, "tool_path", .{ .string = derived.tool_path });
    try provenance.put(allocator, "tool_sha256", .{ .string = derived.tool_sha256 });
    try provenance.put(allocator, "tool_version", .{ .string = derived.expected_version });
    try provenance.put(allocator, "compile_command", .{ .array = compile_arr });
    try provenance.put(allocator, "objdump_command", .{ .array = objdump_arr });
    try provenance.put(allocator, "explore_command", .{ .array = explore_arr });
    try provenance.put(allocator, "cwasm_path", .{ .string = cwasm_path });
    try provenance.put(allocator, "cwasm_sha256", .{ .string = cwasm_sha });
    try provenance.put(allocator, "explore_path", .{ .string = explore_path });
    try provenance.put(allocator, "explore_sha256", .{ .string = explore_sha });
    try provenance.put(allocator, "constant_pool_exclusion", .{ .object = const_pool });

    return .{
        .instructions = body_instructions,
        .code_size = code_size,
        .provenance = provenance,
    };
}

pub fn capture(allocator: std.mem.Allocator, io: std.Io, config_path: []const u8, work_dir: []const u8) !ObjectMap {
    const config = try loadCaptureConfig(allocator, io, config_path);
    const bench = try validateBenchmark(allocator, io, config);
    const core_identity = try parseCoreWasm(allocator, io, config.core_wasm_path);
    if (!std.mem.eql(u8, core_identity.sha256, config.core_wasm_sha256)) {
        return fail("core_wasm SHA-256 changed during capture", .{});
    }
    const resolved_fn = try resolveWasmFunction(allocator, core_identity, config.wasm_index, config.function_name);
    const wasm_index = resolved_fn.wasm_index;
    const function_name = resolved_fn.name;
    const wamr_id = try validateWamrIdentity(
        allocator,
        io,
        config,
        bench.report,
        bench.attribution,
        bench.component_sha,
        core_identity,
        wasm_index,
    );

    const wamr_fn = try readWamrFunction(allocator, io, config.wamr_cwasm_path, config.wamr_local_func);
    if (wamr_fn.function_count != core_identity.local_function_count) {
        return fail("WAMR cwasm function_count ({d}) does not match core wasm local_function_count ({d})", .{
            wamr_fn.function_count,
            core_identity.local_function_count,
        });
    }
    const wamr_dis = try disassembleWamr(allocator, io, wamr_fn, work_dir);
    const wamr_static = try aggregateStaticMetrics(allocator, wamr_dis.instructions, wamr_fn.code_size);
    const wamr_dm = try wamrDynamicMetrics(allocator, bench.attribution);

    const wt_cap = try captureWasmtime(allocator, io, bench.report, config.core_wasm_path, wasm_index, work_dir);
    const wt_static = try aggregateStaticMetrics(allocator, wt_cap.instructions, wt_cap.code_size);

    var wt_dynamic: ObjectMap = .empty;
    var wt_dynamic_prov: ObjectMap = .empty;
    if (config.wasmtime_profile_path != null and config.wasmtime_profile_sha256 != null) {
        wt_dynamic = try loadExplicitProfile(
            allocator,
            io,
            config.wasmtime_profile_path.?,
            config.wasmtime_profile_sha256.?,
            "wasmtime",
            core_identity.sha256,
            wasm_index,
            function_name,
        );
        try wt_dynamic_prov.put(allocator, "status", .{ .string = "measured" });
        try wt_dynamic_prov.put(allocator, "profile_path", .{ .string = config.wasmtime_profile_path.? });
        try wt_dynamic_prov.put(allocator, "profile_sha256", .{ .string = config.wasmtime_profile_sha256.? });
    } else {
        for (METRIC_ORDER) |metric| {
            const reason = if (std.mem.endsWith(u8, metric, "_bytes"))
                "this metric is static metadata, not an instruction class"
            else
                "no verified Wasmtime function/instruction perf profile was supplied in capture config";
            try wt_dynamic.put(allocator, metric, try unavailableMetric(allocator, reason));
        }
        try wt_dynamic_prov.put(allocator, "status", .{ .string = "unavailable" });
        try wt_dynamic_prov.put(allocator, "reason", .{ .string = "no verified Wasmtime function/instruction perf profile was supplied in capture config" });
    }

    var jump_table_bytes: usize = 0;
    var jt_arr = Array.init(allocator);
    for (wamr_dis.jump_tables) |jt| {
        jump_table_bytes += jt.end - jt.start;
        var jt_obj: ObjectMap = .empty;
        try jt_obj.put(allocator, "start_offset", .{ .integer = @intCast(jt.start) });
        try jt_obj.put(allocator, "end_offset", .{ .integer = @intCast(jt.end) });
        try jt_obj.put(allocator, "entries", .{ .integer = @intCast(jt.entries) });
        try jt_arr.append(.{ .object = jt_obj });
    }

    const validation = bench.report.get("validation").?.object;
    const inputs = validation.get("inputs").?.object;
    const comp_input = inputs.get("component").?.object;
    const tools = validation.get("tools").?.object;
    const wamrc_tool = tools.get("wamrc").?.object;
    const precompile = bench.report.get("precompile").?.object;
    const precompile_cmds = precompile.get("commands").?.object;

    var comp_obj: ObjectMap = .empty;
    try comp_obj.put(allocator, "path", comp_input.get("path").?);
    try comp_obj.put(allocator, "sha256", .{ .string = bench.component_sha });

    var cw_obj: ObjectMap = .empty;
    try cw_obj.put(allocator, "path", .{ .string = config.core_wasm_path });
    try cw_obj.put(allocator, "sha256", .{ .string = core_identity.sha256 });
    try cw_obj.put(allocator, "imported_function_count", .{ .integer = @intCast(core_identity.imported_function_count) });
    try cw_obj.put(allocator, "local_function_count", .{ .integer = @intCast(core_identity.local_function_count) });
    try cw_obj.put(allocator, "total_function_count", .{ .integer = @intCast(core_identity.functionCount()) });

    var fn_obj: ObjectMap = .empty;
    try fn_obj.put(allocator, "wasm_index", .{ .integer = @intCast(wasm_index) });
    try fn_obj.put(allocator, "name", if (function_name) |n| .{ .string = n } else .null);
    try fn_obj.put(allocator, "wamr_local_func", .{ .integer = @intCast(config.wamr_local_func) });
    try fn_obj.put(allocator, "wasmtime_symbol", .{ .string = try std.fmt.allocPrint(allocator, "wasm[0]::function[{d}]", .{wasm_index}) });

    var id_obj: ObjectMap = .empty;
    try id_obj.put(allocator, "component", .{ .object = comp_obj });
    try id_obj.put(allocator, "core_wasm", .{ .object = cw_obj });
    try id_obj.put(allocator, "function", .{ .object = fn_obj });

    var br_obj: ObjectMap = .empty;
    try br_obj.put(allocator, "path", .{ .string = config.benchmark_report_path });
    try br_obj.put(allocator, "sha256", .{ .string = config.benchmark_report_sha256 });

    var inline_data: ObjectMap = .empty;
    try inline_data.put(allocator, "method", .{ .string = "x86_64 WAMR br_table dispatch pattern scan" });
    try inline_data.put(allocator, "jump_tables", .{ .array = jt_arr });
    try inline_data.put(allocator, "excluded_bytes", .{ .integer = @intCast(jump_table_bytes) });

    var wamr_prov: ObjectMap = .empty;
    try wamr_prov.put(allocator, "manifest_path", .{ .string = config.wamr_manifest_path });
    try wamr_prov.put(allocator, "manifest_sha256", .{ .string = wamr_id.manifest_sha256 });
    try wamr_prov.put(allocator, "cwasm_path", .{ .string = config.wamr_cwasm_path });
    try wamr_prov.put(allocator, "cwasm_sha256", .{ .string = wamr_id.cwasm_sha256 });
    try wamr_prov.put(allocator, "wamrc", wamrc_tool.get("path").?);
    try wamr_prov.put(allocator, "wamrc_sha256", wamrc_tool.get("sha256").?);
    try wamr_prov.put(allocator, "wamrc_version", wamrc_tool.get("version").?);
    try wamr_prov.put(allocator, "precompile_command", precompile_cmds.get("wamr").?);
    try wamr_prov.put(allocator, "text_offset", .{ .integer = @intCast(wamr_fn.text_offset) });
    try wamr_prov.put(allocator, "code_size_bytes", .{ .integer = @intCast(wamr_fn.code_size) });
    try wamr_prov.put(allocator, "inline_data_exclusion", .{ .object = inline_data });

    var wamr_engine: ObjectMap = .empty;
    try wamr_engine.put(allocator, "provenance", .{ .object = wamr_prov });
    try wamr_engine.put(allocator, "static_metrics", .{ .object = wamr_static });
    try wamr_engine.put(allocator, "dynamic_metrics", .{ .object = wamr_dm.dynamic });

    var wt_prov = wt_cap.provenance;
    try wt_prov.put(allocator, "dynamic_profile", .{ .object = wt_dynamic_prov });

    var wt_engine: ObjectMap = .empty;
    try wt_engine.put(allocator, "provenance", .{ .object = wt_prov });
    try wt_engine.put(allocator, "static_metrics", .{ .object = wt_static });
    try wt_engine.put(allocator, "dynamic_metrics", .{ .object = wt_dynamic });

    var engines_obj: ObjectMap = .empty;
    try engines_obj.put(allocator, "wamr", .{ .object = wamr_engine });
    try engines_obj.put(allocator, "wasmtime", .{ .object = wt_engine });

    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = CAPTURE_SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = "wamr-wasmtime-hot-function-capture" });
    try out.put(allocator, "benchmark_report", .{ .object = br_obj });
    try out.put(allocator, "identity", .{ .object = id_obj });
    try out.put(allocator, "hotness", .{ .object = wamr_dm.hotness });
    try out.put(allocator, "engines", .{ .object = engines_obj });
    return out;
}

fn validateMetricMap(
    allocator: std.mem.Allocator,
    metrics: ?Value,
    label: []const u8,
    kind: []const u8,
) !ObjectMap {
    const obj = try requireObject(metrics, label);
    for (METRIC_ORDER) |metric| {
        const entry = try requireObject(obj.get(metric), try std.fmt.allocPrint(allocator, "{s}.{s}", .{ label, metric }));
        const status = try requireString(entry.get("status"), try std.fmt.allocPrint(allocator, "{s}.{s}.status", .{ label, metric }));
        if (std.mem.eql(u8, status, "measured")) {
            if (std.mem.eql(u8, kind, "static")) {
                _ = try requireInteger(entry.get("value"), try std.fmt.allocPrint(allocator, "{s}.{s}.value", .{ label, metric }), 0);
                _ = try requireString(entry.get("unit"), try std.fmt.allocPrint(allocator, "{s}.{s}.unit", .{ label, metric }));
            } else {
                _ = try requireInteger(entry.get("samples"), try std.fmt.allocPrint(allocator, "{s}.{s}.samples", .{ label, metric }), 0);
                _ = try requireNumber(entry.get("percent_of_function_samples"), try std.fmt.allocPrint(allocator, "{s}.{s}.percent_of_function_samples", .{ label, metric }), 0.0);
                _ = try requireNumber(entry.get("percent_of_run_samples"), try std.fmt.allocPrint(allocator, "{s}.{s}.percent_of_run_samples", .{ label, metric }), 0.0);
            }
        } else if (std.mem.eql(u8, status, "unavailable")) {
            _ = try requireString(entry.get("reason"), try std.fmt.allocPrint(allocator, "{s}.{s}.reason", .{ label, metric }));
        } else {
            return fail("{s}.{s}.status must be 'measured' or 'unavailable'", .{ label, metric });
        }
    }
    return obj;
}

pub fn validateCapture(
    allocator: std.mem.Allocator,
    capture_doc: ObjectMap,
) !struct {
    identity: ObjectMap,
    hotness: ObjectMap,
    wamr_static: ObjectMap,
    wamr_dynamic: ObjectMap,
    wt_static: ObjectMap,
    wt_dynamic: ObjectMap,
} {
    if ((try requireInteger(capture_doc.get("schema_version"), "capture.schema_version", 1)) != CAPTURE_SCHEMA_VERSION) {
        return fail("capture.schema_version must be {d}", .{CAPTURE_SCHEMA_VERSION});
    }
    const kind = try requireString(capture_doc.get("kind"), "capture.kind");
    if (!std.mem.eql(u8, kind, "wamr-wasmtime-hot-function-capture")) {
        return fail("capture.kind must be 'wamr-wasmtime-hot-function-capture'", .{});
    }
    const bench = try requireObject(capture_doc.get("benchmark_report"), "capture.benchmark_report");
    _ = try requireSha256(allocator, bench.get("sha256"), "capture.benchmark_report.sha256");

    const identity = try requireObject(capture_doc.get("identity"), "capture.identity");
    const comp = try requireObject(identity.get("component"), "capture.identity.component");
    _ = try requireSha256(allocator, comp.get("sha256"), "capture.identity.component.sha256");
    const core = try requireObject(identity.get("core_wasm"), "capture.identity.core_wasm");
    _ = try requireSha256(allocator, core.get("sha256"), "capture.identity.core_wasm.sha256");
    const imported: usize = @intCast(try requireInteger(core.get("imported_function_count"), "capture.identity.core_wasm.imported_function_count", 0));
    const local_count: usize = @intCast(try requireInteger(core.get("local_function_count"), "capture.identity.core_wasm.local_function_count", 1));
    const fn_obj = try requireObject(identity.get("function"), "capture.identity.function");
    const wasm_index: usize = @intCast(try requireInteger(fn_obj.get("wasm_index"), "capture.identity.function.wasm_index", 0));
    const wamr_local: usize = @intCast(try requireInteger(fn_obj.get("wamr_local_func"), "capture.identity.function.wamr_local_func", 0));
    if (wamr_local >= local_count or imported + wamr_local != wasm_index) {
        return fail("capture.identity.function indices are inconsistent", .{});
    }
    const expected_symbol = try std.fmt.allocPrint(allocator, "wasm[0]::function[{d}]", .{wasm_index});
    const actual_symbol = try requireString(fn_obj.get("wasmtime_symbol"), "capture.identity.function.wasmtime_symbol");
    if (!std.mem.eql(u8, actual_symbol, expected_symbol)) {
        return fail("capture.identity.function.wasmtime_symbol does not match wasm_index", .{});
    }

    const hotness = try requireObject(capture_doc.get("hotness"), "capture.hotness");
    if (!std.mem.eql(u8, try requireString(hotness.get("status"), "capture.hotness.status"), "measured")) {
        return fail("capture.hotness.status must be 'measured'", .{});
    }
    _ = try requireNumber(hotness.get("percent_of_run"), "capture.hotness.percent_of_run", 0.0);

    const engines = try requireObject(capture_doc.get("engines"), "capture.engines");
    const wamr = try requireObject(engines.get("wamr"), "capture.engines.wamr");
    const wasmtime = try requireObject(engines.get("wasmtime"), "capture.engines.wasmtime");
    const wamr_prov = try requireObject(wamr.get("provenance"), "capture.engines.wamr.provenance");
    const wt_prov = try requireObject(wasmtime.get("provenance"), "capture.engines.wasmtime.provenance");
    _ = try requireSha256(allocator, wamr_prov.get("cwasm_sha256"), "capture.engines.wamr.provenance.cwasm_sha256");
    _ = try requireSha256(allocator, wt_prov.get("cwasm_sha256"), "capture.engines.wasmtime.provenance.cwasm_sha256");
    _ = try requireSha256(allocator, wt_prov.get("explore_sha256"), "capture.engines.wasmtime.provenance.explore_sha256");

    const wamr_static = try validateMetricMap(allocator, wamr.get("static_metrics"), "capture.engines.wamr.static_metrics", "static");
    const wamr_dynamic = try validateMetricMap(allocator, wamr.get("dynamic_metrics"), "capture.engines.wamr.dynamic_metrics", "dynamic");
    const wt_static = try validateMetricMap(allocator, wasmtime.get("static_metrics"), "capture.engines.wasmtime.static_metrics", "static");
    const wt_dynamic = try validateMetricMap(allocator, wasmtime.get("dynamic_metrics"), "capture.engines.wasmtime.dynamic_metrics", "dynamic");
    return .{
        .identity = identity,
        .hotness = hotness,
        .wamr_static = wamr_static,
        .wamr_dynamic = wamr_dynamic,
        .wt_static = wt_static,
        .wt_dynamic = wt_dynamic,
    };
}

pub fn compareCapture(
    allocator: std.mem.Allocator,
    capture_doc: ObjectMap,
    gate_headroom_pct: f64,
    gate_min_differences: usize,
) !ObjectMap {
    if (gate_headroom_pct <= 0.0 or !std.math.isFinite(gate_headroom_pct)) {
        return fail("gate_headroom_pct must be > 0", .{});
    }
    if (gate_min_differences < 1) {
        return fail("gate_min_differences must be >= 1", .{});
    }
    const v = try validateCapture(allocator, capture_doc);
    const fn_share = try requireNumber(v.hotness.get("percent_of_run"), "capture.hotness.percent_of_run", 0.0);

    var metric_rows = Array.init(allocator);
    var qualified: std.ArrayList(ObjectMap) = .empty;
    var ranked_candidates: std.ArrayList(ObjectMap) = .empty;

    for (METRIC_ORDER) |metric| {
        const ws = v.wamr_static.get(metric).?.object;
        const ts = v.wt_static.get(metric).?.object;
        const wd = v.wamr_dynamic.get(metric).?.object;
        const td = v.wt_dynamic.get(metric).?.object;

        var static_delta: ?i64 = null;
        var static_pct: ?f64 = null;
        var static_reduction_fraction: ?f64 = null;
        if (std.mem.eql(u8, ws.get("status").?.string, "measured") and std.mem.eql(u8, ts.get("status").?.string, "measured")) {
            const wv = ws.get("value").?.integer;
            const tv = ts.get("value").?.integer;
            static_delta = wv - tv;
            if (tv > 0) {
                static_pct = 100.0 * @as(f64, @floatFromInt(static_delta.?)) / @as(f64, @floatFromInt(tv));
            }
            if (wv > 0 and static_delta.? > 0) {
                static_reduction_fraction = @as(f64, @floatFromInt(static_delta.?)) / @as(f64, @floatFromInt(wv));
            } else if (wv >= 0) {
                static_reduction_fraction = 0.0;
            }
        }

        var dynamic_delta_fn: ?f64 = null;
        var dynamic_delta_run: ?f64 = null;
        if (std.mem.eql(u8, wd.get("status").?.string, "measured") and std.mem.eql(u8, td.get("status").?.string, "measured")) {
            dynamic_delta_fn = (try requireNumber(wd.get("percent_of_function_samples"), "", -1e9)) -
                (try requireNumber(td.get("percent_of_function_samples"), "", -1e9));
            dynamic_delta_run = (try requireNumber(wd.get("percent_of_run_samples"), "", -1e9)) -
                (try requireNumber(td.get("percent_of_run_samples"), "", -1e9));
        }

        var headroom_pct: ?f64 = null;
        var headroom_method: []const u8 = "not-gate-metric";
        var headroom_reason: ?[]const u8 = "summary size/count metric is context, not an isolated codegen lever";

        if (isGateMetric(metric)) {
            if (!std.mem.eql(u8, ws.get("status").?.string, "measured") or !std.mem.eql(u8, ts.get("status").?.string, "measured")) {
                headroom_method = "unavailable";
                headroom_reason = "static metric is unavailable for one or both engines";
            } else if (static_delta == null or static_delta.? <= 0) {
                headroom_pct = 0.0;
                headroom_method = "non-positive-static-delta";
                headroom_reason = "WAMR static count does not exceed Wasmtime";
            } else if (!std.mem.eql(u8, wd.get("status").?.string, "measured")) {
                headroom_method = "unavailable";
                headroom_reason = wd.get("reason").?.string;
            } else if (std.mem.eql(u8, td.get("status").?.string, "measured")) {
                const w_run = try requireNumber(wd.get("percent_of_run_samples"), "", 0.0);
                const t_run = try requireNumber(td.get("percent_of_run_samples"), "", 0.0);
                headroom_pct = @max(0.0, w_run - t_run);
                headroom_method = "measured-run-sample-share-delta";
                headroom_reason = null;
            } else {
                const w_run = try requireNumber(wd.get("percent_of_run_samples"), "", 0.0);
                headroom_pct = w_run * (static_reduction_fraction orelse 0.0);
                headroom_method = "wamr-dynamic-share-scaled-by-static-excess";
                headroom_reason = td.get("reason").?.string;
            }
        }

        const clears_gate = isGateMetric(metric) and static_delta != null and static_delta.? > 0 and
            headroom_pct != null and headroom_pct.? >= gate_headroom_pct;

        var st_obj: ObjectMap = .empty;
        try st_obj.put(allocator, "wamr", .{ .object = ws });
        try st_obj.put(allocator, "wasmtime", .{ .object = ts });
        try st_obj.put(allocator, "delta_wamr_minus_wasmtime", if (static_delta) |d| .{ .integer = d } else .null);
        try st_obj.put(allocator, "percent_vs_wasmtime", if (static_pct) |p| .{ .float = p } else .null);

        var dy_obj: ObjectMap = .empty;
        try dy_obj.put(allocator, "wamr", .{ .object = wd });
        try dy_obj.put(allocator, "wasmtime", .{ .object = td });
        try dy_obj.put(allocator, "delta_percent_of_function_samples", if (dynamic_delta_fn) |d| .{ .float = d } else .null);
        try dy_obj.put(allocator, "delta_percent_of_run_samples", if (dynamic_delta_run) |d| .{ .float = d } else .null);

        var hr_obj: ObjectMap = .empty;
        try hr_obj.put(allocator, "percent_of_run", if (headroom_pct) |p| .{ .float = p } else .null);
        try hr_obj.put(allocator, "percent_of_hot_function", if (headroom_pct != null and fn_share > 0.0) .{ .float = (headroom_pct.? / fn_share) * 100.0 } else .null);
        try hr_obj.put(allocator, "method", .{ .string = headroom_method });
        try hr_obj.put(allocator, "note", if (headroom_reason) |r| .{ .string = r } else .null);
        try hr_obj.put(allocator, "clears_gate", .{ .bool = clears_gate });

        var row: ObjectMap = .empty;
        try row.put(allocator, "metric", .{ .string = metric });
        try row.put(allocator, "label", .{ .string = metricLabel(metric) });
        try row.put(allocator, "gate_eligible", .{ .bool = isGateMetric(metric) });
        try row.put(allocator, "candidate_lever", if (isGateMetric(metric)) .{ .string = leverForMetric(metric) } else .null);
        try row.put(allocator, "static", .{ .object = st_obj });
        try row.put(allocator, "dynamic", .{ .object = dy_obj });
        try row.put(allocator, "conservative_headroom", .{ .object = hr_obj });

        try metric_rows.append(.{ .object = row });
        if (isGateMetric(metric)) {
            try ranked_candidates.append(allocator, row);
        }
        if (clears_gate) {
            try qualified.append(allocator, row);
        }
    }

    const SortCtx = struct {
        fn lessThan(_: void, a: ObjectMap, b: ObjectMap) bool {
            const ah = a.get("conservative_headroom").?.object;
            const bh = b.get("conservative_headroom").?.object;
            const ap = if (ah.get("percent_of_run")) |pv| (if (pv != .null) (requireNumber(pv, "", -1.0) catch -1.0) else -1.0) else -1.0;
            const bp = if (bh.get("percent_of_run")) |pv| (if (pv != .null) (requireNumber(pv, "", -1.0) catch -1.0) else -1.0) else -1.0;
            if (ap != bp) return ap > bp;
            const as = a.get("static").?.object;
            const bs = b.get("static").?.object;
            const ad = if (as.get("delta_wamr_minus_wasmtime")) |dv| (if (dv == .integer) dv.integer else std.math.minInt(i64)) else std.math.minInt(i64);
            const bd = if (bs.get("delta_wamr_minus_wasmtime")) |dv| (if (dv == .integer) dv.integer else std.math.minInt(i64)) else std.math.minInt(i64);
            return ad > bd;
        }
    };
    std.mem.sort(ObjectMap, ranked_candidates.items, {}, SortCtx.lessThan);
    std.mem.sort(ObjectMap, qualified.items, {}, SortCtx.lessThan);

    var ranked = Array.init(allocator);
    for (ranked_candidates.items[0..@min(3, ranked_candidates.items.len)], 1..) |row, idx| {
        var r_obj: ObjectMap = .empty;
        try r_obj.put(allocator, "rank", .{ .integer = @intCast(idx) });
        try r_obj.put(allocator, "metric", row.get("metric").?);
        try r_obj.put(allocator, "label", row.get("label").?);
        try r_obj.put(allocator, "candidate_lever", row.get("candidate_lever").?);
        try r_obj.put(allocator, "static_delta", row.get("static").?.object.get("delta_wamr_minus_wasmtime").?);
        try r_obj.put(allocator, "conservative_headroom_pct", row.get("conservative_headroom").?.object.get("percent_of_run").?);
        try r_obj.put(allocator, "headroom_method", row.get("conservative_headroom").?.object.get("method").?);
        try r_obj.put(allocator, "clears_gate", row.get("conservative_headroom").?.object.get("clears_gate").?);
        try ranked.append(.{ .object = r_obj });
    }

    var qual_names = Array.init(allocator);
    for (qualified.items) |row| try qual_names.append(row.get("metric").?);

    var gate_obj: ObjectMap = .empty;
    try gate_obj.put(allocator, "minimum_differences", .{ .integer = @intCast(gate_min_differences) });
    try gate_obj.put(allocator, "minimum_headroom_pct_each", .{ .float = gate_headroom_pct });
    try gate_obj.put(allocator, "qualified_differences", .{ .array = qual_names });

    var recommendation: ObjectMap = .empty;
    if (qualified.items.len >= gate_min_differences) {
        const top = qualified.items[0];
        const top_hr = try requireNumber(top.get("conservative_headroom").?.object.get("percent_of_run"), "", 0.0);
        try recommendation.put(allocator, "status", .{ .string = "recommend" });
        try recommendation.put(allocator, "selected_lever", top.get("candidate_lever").?);
        try recommendation.put(allocator, "primary_metric", top.get("metric").?);
        try recommendation.put(allocator, "gate", .{ .object = gate_obj });
        try recommendation.put(allocator, "rationale", .{ .string = try std.fmt.allocPrint(
            allocator,
            "{d} structural differences clear the >= {d:.1}% conservative run-level headroom gate; highest-headroom lever is '{s}' from {s} ({d:.2}% of run).",
            .{ qualified.items.len, gate_headroom_pct, top.get("candidate_lever").?.string, top.get("metric").?.string, top_hr },
        ) });
    } else {
        try recommendation.put(allocator, "status", .{ .string = "no-lever-clears-gate" });
        try recommendation.put(allocator, "selected_lever", .null);
        try recommendation.put(allocator, "primary_metric", .null);
        try recommendation.put(allocator, "gate", .{ .object = gate_obj });
        try recommendation.put(allocator, "rationale", .{ .string = try std.fmt.allocPrint(
            allocator,
            "Only {d} structural difference(s) clear the >= {d:.1}% conservative run-level headroom gate (minimum required: {d}); do not open a speculative codegen PR.",
            .{ qualified.items.len, gate_headroom_pct, gate_min_differences },
        ) });
    }

    const engines = capture_doc.get("engines").?.object;
    const wamr_prov = engines.get("wamr").?.object.get("provenance").?.object;
    const wt_prov = engines.get("wasmtime").?.object.get("provenance").?.object;

    var methodology: ObjectMap = .empty;
    try methodology.put(allocator, "function_mapping", .{ .string = "Core wasm imports and functions are parsed directly; WAMR local_func is verified against imported_function_count + local_func == wasm_index, and Wasmtime uses wasm[0]::function[wasm_index]." });
    try methodology.put(allocator, "wamr_data_exclusion", wamr_prov.get("inline_data_exclusion").?);
    try methodology.put(allocator, "wasmtime_data_exclusion", wt_prov.get("constant_pool_exclusion").?);
    try methodology.put(allocator, "headroom_formula", .{ .string = "For gate-eligible instruction classes with WAMR static excess, use max(0, wamr_run_share - wasmtime_run_share) when both dynamic shares are measured; otherwise scale WAMR's measured run-share in that class by (wamr_static - wasmtime_static) / wamr_static." });

    var limitations = Array.init(allocator);
    try limitations.append(.{ .string = "Static instruction counts do not prove dynamic execution weight on their own." });
    if (std.mem.eql(u8, wt_prov.get("dynamic_profile").?.object.get("status").?.string, "unavailable")) {
        try limitations.append(.{ .string = "Wasmtime dynamic per-instruction-class sample share is unavailable unless a verified Wasmtime perf profile is supplied." });
    }
    if (std.mem.eql(u8, v.wamr_dynamic.get("frame_loads").?.object.get("status").?.string, "unavailable")) {
        try limitations.append(v.wamr_dynamic.get("frame_loads").?.object.get("reason").?);
    }

    var cap_prov: ObjectMap = .empty;
    try cap_prov.put(allocator, "wamr", .{ .object = wamr_prov });
    try cap_prov.put(allocator, "wasmtime", .{ .object = wt_prov });

    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = REPORT_SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = "wamr-wasmtime-hot-function-comparison" });
    try out.put(allocator, "benchmark_report", capture_doc.get("benchmark_report").?);
    try out.put(allocator, "identity", .{ .object = v.identity });
    try out.put(allocator, "hotness", .{ .object = v.hotness });
    try out.put(allocator, "metrics", .{ .array = metric_rows });
    try out.put(allocator, "ranked_deltas", .{ .array = ranked });
    try out.put(allocator, "recommendation", .{ .object = recommendation });
    try out.put(allocator, "methodology", .{ .object = methodology });
    try out.put(allocator, "limitations", .{ .array = limitations });
    try out.put(allocator, "capture_provenance", .{ .object = cap_prov });
    return out;
}

fn fmtStatic(allocator: std.mem.Allocator, entry: ObjectMap) ![]const u8 {
    if (!std.mem.eql(u8, entry.get("status").?.string, "measured")) {
        return std.fmt.allocPrint(allocator, "unavailable ({s})", .{entry.get("reason").?.string});
    }
    return std.fmt.allocPrint(allocator, "{d} {s}", .{ entry.get("value").?.integer, entry.get("unit").?.string });
}

fn fmtDynamic(allocator: std.mem.Allocator, entry: ObjectMap) ![]const u8 {
    if (!std.mem.eql(u8, entry.get("status").?.string, "measured")) {
        return std.fmt.allocPrint(allocator, "unavailable ({s})", .{entry.get("reason").?.string});
    }
    const fn_pct = try requireNumber(entry.get("percent_of_function_samples"), "", 0.0);
    const run_pct = try requireNumber(entry.get("percent_of_run_samples"), "", 0.0);
    const samples = entry.get("samples").?.integer;
    return std.fmt.allocPrint(allocator, "{d:.2}% fn / {d:.2}% run ({d})", .{ fn_pct, run_pct, samples });
}

fn fmtDelta(allocator: std.mem.Allocator, static: ObjectMap) ![]const u8 {
    const delta_val = static.get("delta_wamr_minus_wasmtime");
    if (delta_val == null or delta_val.? == .null) return "n/a";
    const delta = delta_val.?.integer;
    const sign: []const u8 = if (delta >= 0) "+" else "";
    const pct_val = static.get("percent_vs_wasmtime");
    if (pct_val == null or pct_val.? == .null) {
        return std.fmt.allocPrint(allocator, "{s}{d}", .{ sign, delta });
    }
    const pct = try requireNumber(pct_val, "", 0.0);
    const psign: []const u8 = if (pct >= 0.0) "+" else "";
    return std.fmt.allocPrint(allocator, "{s}{d} ({s}{d:.1}%)", .{ sign, delta, psign, pct });
}

fn fmtHeadroom(allocator: std.mem.Allocator, headroom: ObjectMap) ![]const u8 {
    const pct_val = headroom.get("percent_of_run");
    if (pct_val == null or pct_val.? == .null) {
        const note_val = headroom.get("note");
        const note = if (note_val != null and note_val.? == .string) note_val.?.string else headroom.get("method").?.string;
        return std.fmt.allocPrint(allocator, "unavailable ({s})", .{note});
    }
    const pct = try requireNumber(pct_val, "", 0.0);
    const gate = if (headroom.get("clears_gate").?.bool) "clears gate" else "below gate";
    return std.fmt.allocPrint(allocator, "{d:.2}% run ({s})", .{ pct, gate });
}

pub fn renderMarkdown(allocator: std.mem.Allocator, report: ObjectMap) ![]const u8 {
    const identity = report.get("identity").?.object;
    const fn_obj = identity.get("function").?.object;
    const core = identity.get("core_wasm").?.object;
    const comp = identity.get("component").?.object;
    const hotness = report.get("hotness").?.object;
    const rec = report.get("recommendation").?.object;

    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(allocator, "# WAMR vs Wasmtime Hot-Function Codegen Comparison\n\n");
    try out.appendSlice(allocator, "## Target Identity\n\n");
    try out.print(allocator, "- Component: `{s}` (`{s}`)\n", .{ comp.get("path").?.string, comp.get("sha256").?.string });
    try out.print(allocator, "- Core module: `{s}` (`{s}`)\n", .{ core.get("path").?.string, core.get("sha256").?.string });
    const fn_name = if (fn_obj.get("name")) |n| (if (n == .string) n.string else "<unnamed>") else "<unnamed>";
    try out.print(allocator, "- Function: `{s}` (`wasm_index={d}`, `wamr local_func={d}`, Wasmtime `{s}`)\n", .{
        fn_name,
        fn_obj.get("wasm_index").?.integer,
        fn_obj.get("wamr_local_func").?.integer,
        fn_obj.get("wasmtime_symbol").?.string,
    });
    try out.print(allocator, "- Hotness: `{d:.2}%` of WAMR profile samples (`{d}` / `{d}` samples)\n\n", .{
        try requireNumber(hotness.get("percent_of_run"), "", 0.0),
        hotness.get("function_samples").?.integer,
        hotness.get("total_samples").?.integer,
    });
    try out.appendSlice(allocator, "## Metrics\n\n");
    try out.appendSlice(allocator, "| Metric | WAMR static | Wasmtime static | Static delta | WAMR dynamic | Wasmtime dynamic | Conservative headroom |\n");
    try out.appendSlice(allocator, "| --- | --- | --- | --- | --- | --- | --- |\n");

    for (report.get("metrics").?.array.items) |row_val| {
        const row = row_val.object;
        try out.print(allocator, "| {s} | {s} | {s} | {s} | {s} | {s} | {s} |\n", .{
            row.get("label").?.string,
            try fmtStatic(allocator, row.get("static").?.object.get("wamr").?.object),
            try fmtStatic(allocator, row.get("static").?.object.get("wasmtime").?.object),
            try fmtDelta(allocator, row.get("static").?.object),
            try fmtDynamic(allocator, row.get("dynamic").?.object.get("wamr").?.object),
            try fmtDynamic(allocator, row.get("dynamic").?.object.get("wasmtime").?.object),
            try fmtHeadroom(allocator, row.get("conservative_headroom").?.object),
        });
    }

    try out.appendSlice(allocator, "\n## Top Structural Deltas\n\n");
    for (report.get("ranked_deltas").?.array.items) |item_val| {
        const item = item_val.object;
        const sd_val = item.get("static_delta");
        const delta = if (sd_val == null or sd_val.? == .null)
            "n/a"
        else blk: {
            const dv = sd_val.?.integer;
            const ds: []const u8 = if (dv >= 0) "+" else "";
            break :blk try std.fmt.allocPrint(allocator, "{s}{d}", .{ ds, dv });
        };
        const ch_val = item.get("conservative_headroom_pct");
        const hr = if (ch_val == null or ch_val.? == .null)
            "unavailable"
        else
            try std.fmt.allocPrint(allocator, "{d:.2}% of run", .{try requireNumber(ch_val, "", 0.0)});
        const status = if (item.get("clears_gate").?.bool) "clears gate" else "below gate";
        try out.print(allocator, "{d}. **{s}** (`{s}`): static delta `{s}`, conservative headroom `{s}` ({s}); candidate lever: `{s}`.\n", .{
            item.get("rank").?.integer,
            item.get("label").?.string,
            item.get("metric").?.string,
            delta,
            hr,
            status,
            item.get("candidate_lever").?.string,
        });
    }

    try out.appendSlice(allocator, "\n## Recommendation\n\n");
    try out.print(allocator, "- Status: `{s}`\n", .{rec.get("status").?.string});
    const sel_lever = if (rec.get("selected_lever")) |sl| (if (sl == .string) sl.string else "none") else "none";
    try out.print(allocator, "- Selected lever: `{s}`\n", .{sel_lever});
    try out.print(allocator, "- Rationale: {s}\n", .{rec.get("rationale").?.string});

    const lims = report.get("limitations").?.array.items;
    if (lims.len > 0) {
        try out.appendSlice(allocator, "\n## Limitations\n\n");
        for (lims) |lim| {
            try out.print(allocator, "- {s}\n", .{lim.string});
        }
    }
    return out.toOwnedSlice(allocator);
}

pub fn writeJson(allocator: std.mem.Allocator, io: std.Io, path: []const u8, payload: Value) !void {
    if (std.fs.path.dirname(path)) |parent| {
        if (parent.len > 0) {
            std.Io.Dir.cwd().createDirPath(io, parent) catch {};
        }
    }
    const serialized = try std.fmt.allocPrint(allocator, "{f}\n", .{json.fmt(payload, .{ .whitespace = .indent_2 })});
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = serialized }) catch {
        return fail("cannot write JSON output: {s}", .{path});
    };
}

pub fn runCli(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !u8 {
    if (args.len == 0) {
        std.debug.print("error: subcommand required (capture or report)\n", .{});
        return 2;
    }
    const cmd = args[0];
    if (std.mem.eql(u8, cmd, "capture")) {
        var config_path: ?[]const u8 = null;
        var work_dir: ?[]const u8 = null;
        var output_path: ?[]const u8 = null;
        var i: usize = 1;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--config") and i + 1 < args.len) {
                i += 1;
                config_path = args[i];
            } else if (std.mem.eql(u8, args[i], "--work-dir") and i + 1 < args.len) {
                i += 1;
                work_dir = args[i];
            } else if (std.mem.eql(u8, args[i], "--output") and i + 1 < args.len) {
                i += 1;
                output_path = args[i];
            } else {
                std.debug.print("error: unknown or incomplete argument: {s}\n", .{args[i]});
                return 2;
            }
        }
        if (config_path == null or work_dir == null or output_path == null) {
            std.debug.print("error: capture requires --config, --work-dir, and --output\n", .{});
            return 2;
        }
        const cap = capture(allocator, io, config_path.?, work_dir.?) catch |err| {
            if (err == error.InvalidComparison) {
                std.debug.print("error: {s}\n", .{lastErrorMessage()});
                return 1;
            }
            return err;
        };
        writeJson(allocator, io, output_path.?, .{ .object = cap }) catch |err| {
            if (err == error.InvalidComparison) {
                std.debug.print("error: {s}\n", .{lastErrorMessage()});
                return 1;
            }
            return err;
        };
        return 0;
    } else if (std.mem.eql(u8, cmd, "report")) {
        var capture_path: ?[]const u8 = null;
        var report_json: ?[]const u8 = null;
        var report_markdown: ?[]const u8 = null;
        var gate_headroom_pct: f64 = 5.0;
        var gate_min_differences: usize = 2;
        var i: usize = 1;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--capture") and i + 1 < args.len) {
                i += 1;
                capture_path = args[i];
            } else if ((std.mem.eql(u8, args[i], "--report-json") or std.mem.eql(u8, args[i], "--json-output")) and i + 1 < args.len) {
                i += 1;
                report_json = args[i];
            } else if ((std.mem.eql(u8, args[i], "--report-markdown") or std.mem.eql(u8, args[i], "--markdown-output")) and i + 1 < args.len) {
                i += 1;
                report_markdown = args[i];
            } else if (std.mem.eql(u8, args[i], "--gate-headroom-pct") and i + 1 < args.len) {
                i += 1;
                gate_headroom_pct = std.fmt.parseFloat(f64, args[i]) catch {
                    std.debug.print("error: invalid --gate-headroom-pct\n", .{});
                    return 2;
                };
            } else if (std.mem.eql(u8, args[i], "--gate-min-differences") and i + 1 < args.len) {
                i += 1;
                gate_min_differences = std.fmt.parseInt(usize, args[i], 10) catch {
                    std.debug.print("error: invalid --gate-min-differences\n", .{});
                    return 2;
                };
            } else {
                std.debug.print("error: unknown or incomplete argument: {s}\n", .{args[i]});
                return 2;
            }
        }
        if (capture_path == null or report_json == null or report_markdown == null) {
            std.debug.print("error: report requires --capture, --report-json, and --report-markdown\n", .{});
            return 2;
        }
        const cap_obj = loadJsonObject(allocator, io, capture_path.?, "capture") catch |err| {
            if (err == error.InvalidComparison) {
                std.debug.print("error: {s}\n", .{lastErrorMessage()});
                return 1;
            }
            return err;
        };
        const rep = compareCapture(allocator, cap_obj, gate_headroom_pct, gate_min_differences) catch |err| {
            if (err == error.InvalidComparison) {
                std.debug.print("error: {s}\n", .{lastErrorMessage()});
                return 1;
            }
            return err;
        };
        writeJson(allocator, io, report_json.?, .{ .object = rep }) catch |err| {
            if (err == error.InvalidComparison) {
                std.debug.print("error: {s}\n", .{lastErrorMessage()});
                return 1;
            }
            return err;
        };
        const md = renderMarkdown(allocator, rep) catch |err| {
            if (err == error.InvalidComparison) {
                std.debug.print("error: {s}\n", .{lastErrorMessage()});
                return 1;
            }
            return err;
        };
        if (std.fs.path.dirname(report_markdown.?)) |parent| {
            if (parent.len > 0) std.Io.Dir.cwd().createDirPath(io, parent) catch {};
        }
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = report_markdown.?, .data = md }) catch {
            std.debug.print("error: cannot write markdown output: {s}\n", .{report_markdown.?});
            return 1;
        };
        return 0;
    } else {
        std.debug.print("error: unknown subcommand: {s}\n", .{cmd});
        return 2;
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    var args_list: std.ArrayList([]const u8) = .empty;
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();
    _ = args_it.next();
    while (args_it.next()) |arg| {
        try args_list.append(allocator, arg);
    }
    const code = try runCli(allocator, io, args_list.items);
    if (code != 0) std.process.exit(code);
}

// ============================================================================
// Unit Tests
// ============================================================================

const testing = std.testing;

fn testLebU32(allocator: std.mem.Allocator, value: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var remaining = value;
    while (true) {
        var byte: u8 = @intCast(remaining & 0x7F);
        remaining >>= 7;
        if (remaining != 0) byte |= 0x80;
        try out.append(allocator, byte);
        if (remaining == 0) break;
    }
    return out.toOwnedSlice(allocator);
}

fn testWasmName(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    return std.mem.concat(allocator, u8, &.{ try testLebU32(allocator, @intCast(value.len)), value });
}

fn testWasmSection(allocator: std.mem.Allocator, section_id: u8, payload: []const u8) ![]u8 {
    return std.mem.concat(allocator, u8, &.{ &[_]u8{section_id}, try testLebU32(allocator, @intCast(payload.len)), payload });
}

fn makeCoreWasm(
    allocator: std.mem.Allocator,
    imported_functions: usize,
    local_functions: usize,
    names: []const struct { index: u32, name: []const u8 },
) ![]u8 {
    const type_payload = [_]u8{ 0x01, 0x60, 0x00, 0x00 };
    var sections: std.ArrayList(u8) = .empty;
    try sections.appendSlice(allocator, try testWasmSection(allocator, 1, &type_payload));

    if (imported_functions > 0) {
        var import_entries: std.ArrayList(u8) = .empty;
        try import_entries.appendSlice(allocator, try testLebU32(allocator, @intCast(imported_functions)));
        for (0..imported_functions) |idx| {
            try import_entries.appendSlice(allocator, try testWasmName(allocator, "env"));
            try import_entries.appendSlice(allocator, try testWasmName(allocator, try std.fmt.allocPrint(allocator, "imp_{d}", .{idx})));
            try import_entries.appendSlice(allocator, &[_]u8{ 0x00, 0x00 });
        }
        try sections.appendSlice(allocator, try testWasmSection(allocator, 2, import_entries.items));
    }

    var func_payload: std.ArrayList(u8) = .empty;
    try func_payload.appendSlice(allocator, try testLebU32(allocator, @intCast(local_functions)));
    for (0..local_functions) |_| try func_payload.append(allocator, 0x00);
    try sections.appendSlice(allocator, try testWasmSection(allocator, 3, func_payload.items));

    var code_payload: std.ArrayList(u8) = .empty;
    try code_payload.appendSlice(allocator, try testLebU32(allocator, @intCast(local_functions)));
    for (0..local_functions) |_| try code_payload.appendSlice(allocator, &[_]u8{ 0x02, 0x00, 0x0B });
    try sections.appendSlice(allocator, try testWasmSection(allocator, 10, code_payload.items));

    var name_map: std.ArrayList(u8) = .empty;
    try name_map.appendSlice(allocator, try testLebU32(allocator, @intCast(names.len)));
    for (names) |entry| {
        try name_map.appendSlice(allocator, try testLebU32(allocator, entry.index));
        try name_map.appendSlice(allocator, try testWasmName(allocator, entry.name));
    }
    const name_subsection = try std.mem.concat(allocator, u8, &.{
        &[_]u8{0x01},
        try testLebU32(allocator, @intCast(name_map.items.len)),
        name_map.items,
    });
    const custom_payload = try std.mem.concat(allocator, u8, &.{
        try testWasmName(allocator, "name"),
        name_subsection,
    });
    try sections.appendSlice(allocator, try testWasmSection(allocator, 0, custom_payload));

    return std.mem.concat(allocator, u8, &.{ "\x00asm\x01\x00\x00\x00", sections.items });
}

fn makeWamrCwasm(allocator: std.mem.Allocator, functions: []const []const u8) ![]u8 {
    var text: std.ArrayList(u8) = .empty;
    var func_sec: std.ArrayList(u8) = .empty;
    var count_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &count_bytes, @intCast(functions.len), .little);
    try func_sec.appendSlice(allocator, &count_bytes);

    var offset: u32 = 0;
    for (functions) |code| {
        try text.appendSlice(allocator, code);
        var entry: [8]u8 = undefined;
        std.mem.writeInt(u32, entry[0..4], offset, .little);
        std.mem.writeInt(u32, entry[4..8], 0, .little);
        try func_sec.appendSlice(allocator, &entry);
        offset += @intCast(code.len);
    }

    var out: std.ArrayList(u8) = .empty;
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], AOT_MAGIC, .little);
    std.mem.writeInt(u32, hdr[4..8], AOT_VERSION, .little);
    try out.appendSlice(allocator, &hdr);

    var sec_hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, sec_hdr[0..4], AOT_TEXT_SECTION, .little);
    std.mem.writeInt(u32, sec_hdr[4..8], @intCast(text.items.len), .little);
    try out.appendSlice(allocator, &sec_hdr);
    try out.appendSlice(allocator, text.items);

    std.mem.writeInt(u32, sec_hdr[0..4], AOT_FUNCTION_SECTION, .little);
    std.mem.writeInt(u32, sec_hdr[4..8], @intCast(func_sec.items.len), .little);
    try out.appendSlice(allocator, &sec_hdr);
    try out.appendSlice(allocator, func_sec.items);
    return out.toOwnedSlice(allocator);
}

const TEST_WAMR_CODE = [_]u8{
    0x55, // push rbp
    0x48, 0x89, 0xE5, // mov rbp, rsp
    0x48, 0x83, 0xEC, 0x20, // sub rsp, 0x20
    0x48, 0x89, 0x7D, 0xF8, // mov [rbp-8], rdi
    0x48, 0x8B, 0x45, 0xF8, // mov rax, [rbp-8]
    0x48, 0x89, 0xC2, // mov rdx, rax
    0x48, 0x8D, 0x0C, 0x10, // lea rcx, [rax+rdx*1]
    0x49, 0x83, 0xFB, 0x01, // cmp r11, 1
    0x0F, 0x83, 0x11, 0x00, 0x00, 0x00, // jae +17
    0x4C, 0x8D, 0x15, 0x0A, 0x00, 0x00, 0x00, // lea r10, [rip+10]
    0x4F, 0x63, 0x1C, 0x9A, // movsxd r11, dword ptr [r10+r11*4]
    0x4D, 0x01, 0xDA, // add r10, r11
    0x41, 0xFF, 0xE2, // jmp r10
    0x04, 0x00, 0x00, 0x00, // inline jump-table entry (+4 -> leave)
    0xC9, // leave
    0xC3, // ret
};

fn makeTestDir(allocator: std.mem.Allocator, io: std.Io, label: []const u8) ![]const u8 {
    var rand_bytes: [8]u8 = undefined;
    io.random(&rand_bytes);
    const cwd_real = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/chf-test-{s}-{x}", .{
        cwd_real,
        label,
        std.mem.readInt(u64, &rand_bytes, .little),
    });
    try std.Io.Dir.cwd().createDirPath(io, dir);
    return dir;
}

const FakeWasmtimeOptions = struct {
    extra_pool: bool = true,
    duplicate_header: bool = false,
    duplicate_explore_func: bool = false,
};

fn writeFakeWasmtime(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, opts: FakeWasmtimeOptions) ![]const u8 {
    const script = try std.fs.path.join(allocator, &.{ dir, "fake-wasmtime" });
    const pool_dis = if (opts.extra_pool)
        "       20f: 00 00                   add    BYTE PTR [rax],al\\n"
    else
        "";
    const dup_hdr = if (opts.duplicate_header)
        "0000000000000300 <wasm[0]::function[3]::hot_loop>:\\n       300: c3                      ret\\n"
    else
        "";
    const pool_exp = if (opts.extra_pool)
        ",{\"address\":527,\"bytes\":[0,0],\"mnemonic\":\"addb\",\"operands\":\"%al, (%rax)\",\"wasm_offset\":null}"
    else
        "";
    const dup_exp = if (opts.duplicate_explore_func)
        ",{\"func_index\":3,\"name\":\"wasm[0]::function[3]::hot_loop\",\"instructions\":[{\"address\":768,\"bytes\":[195],\"mnemonic\":\"retq\",\"operands\":\"\",\"wasm_offset\":48}]}"
    else
        "";

    const content = try std.fmt.allocPrint(allocator,
        \\#!/bin/sh
        \\set -eu
        \\cmd="$1"
        \\shift
        \\if [ "$cmd" = "--version" ]; then
        \\  printf 'wasmtime 43.0.0 (deadbeef 2026-01-01)\n'
        \\  exit 0
        \\fi
        \\out=""
        \\prev=""
        \\for arg in "$@"; do
        \\  if [ "$prev" = "-o" ] || [ "$prev" = "--output" ]; then
        \\    out="$arg"
        \\    prev=""
        \\    continue
        \\  fi
        \\  case "$arg" in
        \\    -o|--output) prev="$arg" ;;
        \\    --output=*) out="${{arg#--output=}}" ;;
        \\  esac
        \\done
        \\if [ "$cmd" = "compile" ]; then
        \\  printf 'fake-wasmtime-cwasm' > "$out"
        \\  exit 0
        \\fi
        \\if [ "$cmd" = "objdump" ]; then
        \\  printf '0000000000000200 <wasm[0]::function[3]::hot_loop>:\n'
        \\  printf '       200: 55                      push   rbp\n'
        \\  printf '       201: 48 89 e5                mov    rbp,rsp\n'
        \\  printf '       204: 48 83 ec 10             sub    rsp,0x10\n'
        \\  printf '       208: 48 89 f8                mov    rax,rdi\n'
        \\  printf '       20b: 75 02                   jne    0x20f\n'
        \\  printf '       20d: c9                      leave\n'
        \\  printf '       20e: c3                      ret\n'
        \\  printf '{s}'
        \\  printf '{s}'
        \\  exit 0
        \\fi
        \\if [ "$cmd" = "explore" ]; then
        \\  cat > "$out" <<'EOF'
        \\<!doctype html><html><script>
        \\window.ASM = {{"functions":[{{"func_index":3,"name":"wasm[0]::function[3]::hot_loop","instructions":[
        \\  {{"address":512,"bytes":[85],"mnemonic":"pushq","operands":"%rbp","wasm_offset":null}},
        \\  {{"address":513,"bytes":[72,137,229],"mnemonic":"movq","operands":"%rsp, %rbp","wasm_offset":null}},
        \\  {{"address":516,"bytes":[72,131,236,16],"mnemonic":"subq","operands":"$0x10, %rsp","wasm_offset":32}},
        \\  {{"address":520,"bytes":[72,137,248],"mnemonic":"movq","operands":"%rdi, %rax","wasm_offset":36}},
        \\  {{"address":523,"bytes":[117,2],"mnemonic":"jne","operands":"0x20f","wasm_offset":40}},
        \\  {{"address":525,"bytes":[201],"mnemonic":"leave","operands":"","wasm_offset":44}},
        \\  {{"address":526,"bytes":[195],"mnemonic":"retq","operands":"","wasm_offset":null}}{s}
        \\]}}{s}]}};
        \\</script></html>
        \\EOF
        \\  exit 0
        \\fi
        \\echo "unexpected fake-wasmtime command: $cmd" >&2
        \\exit 2
        \\
    , .{ pool_dis, dup_hdr, pool_exp, dup_exp });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = script, .data = content });
    if (comptime builtin.os.tag != .windows) {
        var file = try std.Io.Dir.cwd().openFile(io, script, .{});
        defer file.close(io);
        try file.setPermissions(io, .executable_file);
    }
    return script;
}

const FixtureOptions = struct {
    wasmtime_profile: bool = false,
    wasmtime_profile_verified: bool = true,
    unknown_frame_samples: i64 = 0,
    unclassified_frame_load_samples: i64 = 0,
    wamr_code: []const u8 = &TEST_WAMR_CODE,
    fake_wasmtime_opts: FakeWasmtimeOptions = .{},
};

const FixturePaths = struct {
    config_path: []const u8,
    report_path: []const u8,
    manifest_path: []const u8,
    cwasm_path: []const u8,
    core_wasm_path: []const u8,
    sidecar_path: []const u8,
};

fn buildFixture(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, opts: FixtureOptions) !FixturePaths {
    const component = try std.fs.path.join(allocator, &.{ dir, "component.wasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = component, .data = "component-placeholder" });
    const core_wasm = try std.fs.path.join(allocator, &.{ dir, "core-module3.wasm" });
    const core_bytes = try makeCoreWasm(allocator, 2, 2, &.{
        .{ .index = 2, .name = "helper" },
        .{ .index = 3, .name = "hot_loop" },
    });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = core_wasm, .data = core_bytes });

    const cwasm = try std.fs.path.join(allocator, &.{ dir, "module3.cwasm" });
    const cwasm_bytes = try makeWamrCwasm(allocator, &.{ "\xc3", opts.wamr_code });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cwasm, .data = cwasm_bytes });

    const sidecar = try std.fs.path.join(allocator, &.{ dir, "module3.frame-origins.json" });
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = sidecar,
        .data = "{\"format\":\"wamr-aot-frame-origin-v1\",\"functions\":[{\"func_index\":1}]}\n",
    });

    const comp_sha = try sha256File(allocator, io, component);
    const core_sha = try sha256File(allocator, io, core_wasm);
    const cwasm_sha = try sha256File(allocator, io, cwasm);
    const sidecar_sha = try sha256File(allocator, io, sidecar);

    const manifest = try std.fs.path.join(allocator, &.{ dir, "component.cwasm.manifest.json" });
    const manifest_json = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "version": 2,
        \\  "component_sha256": "{s}",
        \\  "modules": [
        \\    {{
        \\      "index": 3,
        \\      "cwasm": "{s}",
        \\      "sha256": "{s}",
        \\      "core_sha256": "{s}",
        \\      "size_bytes": {d}
        \\    }}
        \\  ]
        \\}}
    , .{ comp_sha, cwasm, cwasm_sha, core_sha, cwasm_bytes.len });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = manifest, .data = manifest_json });
    const manifest_sha = try sha256File(allocator, io, manifest);

    const wamrc = try std.fs.path.join(allocator, &.{ dir, "fake-wamrc" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = wamrc, .data = "#!/bin/sh\nprintf 'wamrc 2.4.1\n'\n" });
    if (comptime builtin.os.tag != .windows) {
        var f = try std.Io.Dir.cwd().openFile(io, wamrc, .{});
        defer f.close(io);
        try f.setPermissions(io, .executable_file);
    }
    const wamrc_sha = try sha256File(allocator, io, wamrc);

    const wasmtime_bin = try writeFakeWasmtime(allocator, io, dir, opts.fake_wasmtime_opts);
    const wasmtime_sha = try sha256File(allocator, io, wasmtime_bin);
    const wt_comp_cwasm = try std.fs.path.join(allocator, &.{ dir, "component.wasmtime.cwasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = wt_comp_cwasm, .data = "component-wasmtime-cwasm" });
    const wt_comp_cwasm_sha = try sha256File(allocator, io, wt_comp_cwasm);

    const frame_meta = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "enabled": true,
        \\  "module_index": 3,
        \\  "cwasm": "{s}",
        \\  "cwasm_sha256": "{s}",
        \\  "sidecar": "{s}",
        \\  "sidecar_sha256": "{s}"
        \\}}
    , .{ cwasm, cwasm_sha, sidecar, sidecar_sha });

    const fl_extra = if (opts.unclassified_frame_load_samples > 0)
        try std.fmt.allocPrint(allocator, ",\n          \"frame_load\": {{ \"samples\": {d} }}", .{opts.unclassified_frame_load_samples})
    else
        "";
    const alu_samples: i64 = 200 - opts.unclassified_frame_load_samples;

    const origin_cov_pct: f64 = if (opts.unknown_frame_samples == 0) 100.0 else 90.0;
    const unknown_insts: i64 = if (opts.unknown_frame_samples == 0) 0 else 1;

    const report_path = try std.fs.path.join(allocator, &.{ dir, "bench-report.json" });
    const report_part1 = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "schema_version": 1,
        \\  "kind": "keyvault-tcgc-aot-benchmark",
        \\  "validation": {{
        \\    "inputs": {{
        \\      "component": {{ "path": "{s}", "sha256": "{s}", "size_bytes": 21 }}
        \\    }},
        \\    "tools": {{
        \\      "wamrc": {{ "path": "{s}", "sha256": "{s}", "version": "wamrc 2.4.1" }},
        \\      "wasmtime": {{ "path": "{s}", "sha256": "{s}", "version": "wasmtime 43.0.0 (deadbeef 2026-01-01)" }}
        \\    }}
        \\  }},
        \\  "precompile": {{
        \\    "commands": {{
        \\      "wamr": ["{s}", "--opt-level=3", "-o", "{s}", "{s}"],
        \\      "wasmtime": ["{s}", "compile", "-W", "component-model-async=y", "-o", "{s}", "{s}"]
        \\    }},
        \\    "frame_attribution": {s},
        \\    "artifacts": {{
        \\      "wamr": {{
        \\        "primary": "{s}",
        \\        "files": [
        \\          {{ "path": "{s}", "sha256": "{s}", "size_bytes": {d} }},
        \\          {{ "path": "{s}", "sha256": "{s}", "size_bytes": {d} }},
        \\          {{ "path": "{s}", "sha256": "{s}", "size_bytes": 75 }}
        \\        ]
        \\      }},
        \\      "wasmtime": {{
        \\        "primary": "{s}",
        \\        "files": [
        \\          {{ "path": "{s}", "sha256": "{s}", "size_bytes": 24 }}
        \\        ]
        \\      }}
        \\    }}
        \\  }},
        \\
    , .{
        component,
        comp_sha,
        wamrc,
        wamrc_sha,
        wasmtime_bin,
        wasmtime_sha,
        wamrc,
        manifest,
        component,
        wasmtime_bin,
        wt_comp_cwasm,
        component,
        frame_meta,
        manifest,
        manifest,
        manifest_sha,
        manifest_json.len,
        cwasm,
        cwasm_sha,
        cwasm_bytes.len,
        sidecar,
        sidecar_sha,
        wt_comp_cwasm,
        wt_comp_cwasm,
        wt_comp_cwasm_sha,
    });
    const report_part2 = try std.fmt.allocPrint(allocator,
        \\  "perf": {{
        \\    "selected": true,
        \\    "frame_attribution": {s},
        \\    "attribution": {{
        \\      "cwasm": "{s}",
        \\      "total_samples": 1000,
        \\      "function_count": 2,
        \\      "classified_function": {{
        \\        "local_func": 1,
        \\        "samples": 800,
        \\        "percent_of_run": 80.0,
        \\        "instruction_count": 15,
        \\        "frame_attribution": {{
        \\          "coverage": {{
        \\            "frame_instructions": 2,
        \\            "frame_samples": 300,
        \\            "unknown_frame_instructions": {d},
        \\            "unknown_frame_samples": {d},
        \\            "origin_coverage_pct": {d:.1},
        \\            "origin_sample_coverage_pct": {d:.1}
        \\          }},
        \\          "origins": {{
        \\            "allocator_spill": {{ "samples": 200 }},
        \\            "wasm_local_or_phi": {{ "samples": 100 }},
        \\            "explicit_frame_storage": {{ "samples": 0 }},
        \\            "fixed_runtime_frame_state": {{ "samples": 0 }},
        \\            "unknown": {{ "samples": {d} }}
        \\          }}
        \\        }},
        \\        "classes": {{
        \\          "allocator_spill_load": {{ "samples": 160 }},
        \\          "wasm_local_or_phi_load": {{ "samples": 40 }},
        \\          "allocator_spill_store": {{ "samples": 100 }},
        \\          "regmov": {{ "samples": 80 }},
        \\          "lea": {{ "samples": 70 }},
        \\          "branch": {{ "samples": 40 }},
        \\          "dispatch_jmp": {{ "samples": 90 }},
        \\          "call": {{ "samples": 20 }},
        \\          "alu": {{ "samples": {d} }}{s}
        \\        }}
        \\      }}
        \\    }}
        \\  }}
        \\}}
    , .{
        frame_meta,
        cwasm,
        unknown_insts,
        opts.unknown_frame_samples,
        origin_cov_pct,
        origin_cov_pct,
        opts.unknown_frame_samples,
        alu_samples,
        fl_extra,
    });
    const report_json = try std.mem.concat(allocator, u8, &.{ report_part1, report_part2 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = report_path, .data = report_json });
    const report_sha = try sha256File(allocator, io, report_path);

    var wt_profile_json: []const u8 = "null";
    if (opts.wasmtime_profile) {
        const ppath = try std.fs.path.join(allocator, &.{ dir, "wasmtime-profile.json" });
        const verified_str = if (opts.wasmtime_profile_verified) "true" else "false";
        const pcontent = try std.fmt.allocPrint(allocator,
            \\{{
            \\  "schema_version": 1,
            \\  "kind": "wasm-hot-function-profile",
            \\  "engine": "wasmtime",
            \\  "mapping_verified": {s},
            \\  "mapping_evidence": "perf jitdump mapped to wasm[0]::function[3]",
            \\  "module_sha256": "{s}",
            \\  "wasm_function_index": 3,
            \\  "function_name": "hot_loop",
            \\  "total_samples": 1000,
            \\  "function_samples": 700,
            \\  "metrics": {{
            \\    "frame_loads": {{ "status": "measured", "samples": 40 }},
            \\    "frame_stores": {{ "status": "measured", "samples": 20 }},
            \\    "reg_reg_moves": {{ "status": "measured", "samples": 60 }},
            \\    "address_generation": {{ "status": "measured", "samples": 10 }},
            \\    "branches": {{ "status": "measured", "samples": 40 }},
            \\    "indirect_dispatch": {{ "status": "measured", "samples": 10 }},
            \\    "calls": {{ "status": "measured", "samples": 20 }}
            \\  }}
            \\}}
        , .{ verified_str, core_sha });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ppath, .data = pcontent });
        const psha = try sha256File(allocator, io, ppath);
        wt_profile_json = try std.fmt.allocPrint(allocator, "{{ \"path\": \"{s}\", \"sha256\": \"{s}\" }}", .{ ppath, psha });
    }

    const config_path = try std.fs.path.join(allocator, &.{ dir, "config.json" });
    const cfg_json = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "schema_version": 1,
        \\  "benchmark_report": {{ "path": "{s}", "sha256": "{s}" }},
        \\  "core_wasm": {{ "path": "{s}", "sha256": "{s}" }},
        \\  "function": {{ "wasm_index": 3, "name": "hot_loop" }},
        \\  "wamr": {{
        \\    "manifest": "{s}",
        \\    "cwasm": "{s}",
        \\    "local_func": 1
        \\  }},
        \\  "wasmtime": {{ "profile": {s} }}
        \\}}
    , .{ report_path, report_sha, core_wasm, core_sha, manifest, cwasm, wt_profile_json });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = config_path, .data = cfg_json });

    return .{
        .config_path = config_path,
        .report_path = report_path,
        .manifest_path = manifest,
        .cwasm_path = cwasm,
        .core_wasm_path = core_wasm,
        .sidecar_path = sidecar,
    };
}

fn buildSyntheticCaptureDoc(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    opts: FixtureOptions,
) !ObjectMap {
    const fx = try buildFixture(allocator, io, dir, opts);
    const config = try loadCaptureConfig(allocator, io, fx.config_path);
    const bench = try validateBenchmark(allocator, io, config);
    const core_identity = try parseCoreWasm(allocator, io, config.core_wasm_path);
    const resolved_fn = try resolveWasmFunction(allocator, core_identity, config.wasm_index, config.function_name);
    const wamr_id = try validateWamrIdentity(
        allocator,
        io,
        config,
        bench.report,
        bench.attribution,
        bench.component_sha,
        core_identity,
        resolved_fn.wasm_index,
    );
    const wamr_fn = try readWamrFunction(allocator, io, config.wamr_cwasm_path, config.wamr_local_func);
    const jump_tables = try findWamrJumpTables(allocator, wamr_fn.code);
    const wamr_dis_text =
        \\   0: 55                      push   rbp
        \\   1: 48 89 e5                mov    rbp,rsp
        \\   4: 48 83 ec 20             sub    rsp,0x20
        \\   8: 48 89 7d f8             mov    QWORD PTR [rbp-0x8],rdi
        \\   c: 48 8b 45 f8             mov    rax,QWORD PTR [rbp-0x8]
        \\  10: 48 89 c2                mov    rdx,rax
        \\  13: 48 8d 0c 10             lea    rcx,[rax+rdx*1]
        \\  17: 49 83 fb 01             cmp    r11,0x1
        \\  1b: 0f 83 11 00 00 00       jae    0x32
        \\  21: 4c 8d 15 0a 00 00 00    lea    r10,[rip+0xa]
        \\  28: 4f 63 1c 9a             movsxd r11,DWORD PTR [r10+r11*4]
        \\  2c: 4d 01 da                add    r10,r11
        \\  2f: 41 ff e2                jmp    r10
        \\  36: c9                      leave
        \\  37: c3                      ret
        \\
    ;
    const wamr_insts = try parseDisassembly(allocator, wamr_dis_text, null);
    const wamr_static = try aggregateStaticMetrics(allocator, wamr_insts, wamr_fn.code_size);
    const wamr_dm = try wamrDynamicMetrics(allocator, bench.attribution);

    const work_dir = try std.fs.path.join(allocator, &.{ dir, "work" });
    const wt_cap = if (comptime builtin.os.tag == .windows) blk: {
        const wt_dis_text =
            \\0000000000000200 <wasm[0]::function[3]::hot_loop>:
            \\       200: 55                      push   rbp
            \\       201: 48 89 e5                mov    rbp,rsp
            \\       204: 48 83 ec 10             sub    rsp,0x10
            \\       208: 48 89 f8                mov    rax,rdi
            \\       20b: 75 02                   jne    0x20f
            \\       20d: c9                      leave
            \\       20e: c3                      ret
            \\
        ;
        const wt_insts = try parseDisassembly(allocator, wt_dis_text, resolved_fn.wasm_index);
        var wt_code_size: usize = 0;
        for (wt_insts) |inst| wt_code_size += inst.size;
        var wt_prov_inner: ObjectMap = .empty;
        try wt_prov_inner.put(allocator, "cwasm_sha256", .{ .string = "3333333333333333333333333333333333333333333333333333333333333333" });
        try wt_prov_inner.put(allocator, "explore_sha256", .{ .string = "4444444444444444444444444444444444444444444444444444444444444444" });
        break :blk .{ .instructions = wt_insts, .code_size = wt_code_size, .provenance = wt_prov_inner };
    } else try captureWasmtime(allocator, io, bench.report, config.core_wasm_path, resolved_fn.wasm_index, work_dir);
    const wt_static = try aggregateStaticMetrics(allocator, wt_cap.instructions, wt_cap.code_size);

    var wt_dynamic: ObjectMap = .empty;
    var wt_dynamic_prov: ObjectMap = .empty;
    if (config.wasmtime_profile_path != null and config.wasmtime_profile_sha256 != null) {
        wt_dynamic = try loadExplicitProfile(
            allocator,
            io,
            config.wasmtime_profile_path.?,
            config.wasmtime_profile_sha256.?,
            "wasmtime",
            core_identity.sha256,
            resolved_fn.wasm_index,
            resolved_fn.name,
        );
        try wt_dynamic_prov.put(allocator, "status", .{ .string = "measured" });
        try wt_dynamic_prov.put(allocator, "profile_path", .{ .string = config.wasmtime_profile_path.? });
        try wt_dynamic_prov.put(allocator, "profile_sha256", .{ .string = config.wasmtime_profile_sha256.? });
    } else {
        for (METRIC_ORDER) |metric| {
            const reason = if (std.mem.endsWith(u8, metric, "_bytes"))
                "this metric is static metadata, not an instruction class"
            else
                "no verified Wasmtime function/instruction perf profile was supplied in capture config";
            try wt_dynamic.put(allocator, metric, try unavailableMetric(allocator, reason));
        }
        try wt_dynamic_prov.put(allocator, "status", .{ .string = "unavailable" });
        try wt_dynamic_prov.put(allocator, "reason", .{ .string = "no verified Wasmtime function/instruction perf profile was supplied in capture config" });
    }

    var jump_table_bytes: usize = 0;
    var jt_arr = Array.init(allocator);
    for (jump_tables) |jt| {
        jump_table_bytes += jt.end - jt.start;
        var jt_obj: ObjectMap = .empty;
        try jt_obj.put(allocator, "start_offset", .{ .integer = @intCast(jt.start) });
        try jt_obj.put(allocator, "end_offset", .{ .integer = @intCast(jt.end) });
        try jt_obj.put(allocator, "entries", .{ .integer = @intCast(jt.entries) });
        try jt_arr.append(.{ .object = jt_obj });
    }

    const validation = bench.report.get("validation").?.object;
    const inputs = validation.get("inputs").?.object;
    const comp_input = inputs.get("component").?.object;
    const tools = validation.get("tools").?.object;
    const wamrc_tool = tools.get("wamrc").?.object;
    const precompile = bench.report.get("precompile").?.object;
    const precompile_cmds = precompile.get("commands").?.object;

    var comp_obj: ObjectMap = .empty;
    try comp_obj.put(allocator, "path", comp_input.get("path").?);
    try comp_obj.put(allocator, "sha256", .{ .string = bench.component_sha });

    var cw_obj: ObjectMap = .empty;
    try cw_obj.put(allocator, "path", .{ .string = config.core_wasm_path });
    try cw_obj.put(allocator, "sha256", .{ .string = core_identity.sha256 });
    try cw_obj.put(allocator, "imported_function_count", .{ .integer = @intCast(core_identity.imported_function_count) });
    try cw_obj.put(allocator, "local_function_count", .{ .integer = @intCast(core_identity.local_function_count) });
    try cw_obj.put(allocator, "total_function_count", .{ .integer = @intCast(core_identity.functionCount()) });

    var fn_obj: ObjectMap = .empty;
    try fn_obj.put(allocator, "wasm_index", .{ .integer = @intCast(resolved_fn.wasm_index) });
    try fn_obj.put(allocator, "name", if (resolved_fn.name) |n| .{ .string = n } else .null);
    try fn_obj.put(allocator, "wamr_local_func", .{ .integer = @intCast(config.wamr_local_func) });
    try fn_obj.put(allocator, "wasmtime_symbol", .{ .string = try std.fmt.allocPrint(allocator, "wasm[0]::function[{d}]", .{resolved_fn.wasm_index}) });

    var id_obj: ObjectMap = .empty;
    try id_obj.put(allocator, "component", .{ .object = comp_obj });
    try id_obj.put(allocator, "core_wasm", .{ .object = cw_obj });
    try id_obj.put(allocator, "function", .{ .object = fn_obj });

    var br_obj: ObjectMap = .empty;
    try br_obj.put(allocator, "path", .{ .string = config.benchmark_report_path });
    try br_obj.put(allocator, "sha256", .{ .string = config.benchmark_report_sha256 });

    var inline_data: ObjectMap = .empty;
    try inline_data.put(allocator, "method", .{ .string = "x86_64 WAMR br_table dispatch pattern scan" });
    try inline_data.put(allocator, "jump_tables", .{ .array = jt_arr });
    try inline_data.put(allocator, "excluded_bytes", .{ .integer = @intCast(jump_table_bytes) });

    var wamr_prov: ObjectMap = .empty;
    try wamr_prov.put(allocator, "manifest_path", .{ .string = config.wamr_manifest_path });
    try wamr_prov.put(allocator, "manifest_sha256", .{ .string = wamr_id.manifest_sha256 });
    try wamr_prov.put(allocator, "cwasm_path", .{ .string = config.wamr_cwasm_path });
    try wamr_prov.put(allocator, "cwasm_sha256", .{ .string = wamr_id.cwasm_sha256 });
    try wamr_prov.put(allocator, "wamrc", wamrc_tool.get("path").?);
    try wamr_prov.put(allocator, "wamrc_sha256", wamrc_tool.get("sha256").?);
    try wamr_prov.put(allocator, "wamrc_version", wamrc_tool.get("version").?);
    try wamr_prov.put(allocator, "precompile_command", precompile_cmds.get("wamr").?);
    try wamr_prov.put(allocator, "text_offset", .{ .integer = @intCast(wamr_fn.text_offset) });
    try wamr_prov.put(allocator, "code_size_bytes", .{ .integer = @intCast(wamr_fn.code_size) });
    try wamr_prov.put(allocator, "inline_data_exclusion", .{ .object = inline_data });

    var wamr_engine: ObjectMap = .empty;
    try wamr_engine.put(allocator, "provenance", .{ .object = wamr_prov });
    try wamr_engine.put(allocator, "static_metrics", .{ .object = wamr_static });
    try wamr_engine.put(allocator, "dynamic_metrics", .{ .object = wamr_dm.dynamic });

    var wt_prov = wt_cap.provenance;
    try wt_prov.put(allocator, "dynamic_profile", .{ .object = wt_dynamic_prov });

    var wt_engine: ObjectMap = .empty;
    try wt_engine.put(allocator, "provenance", .{ .object = wt_prov });
    try wt_engine.put(allocator, "static_metrics", .{ .object = wt_static });
    try wt_engine.put(allocator, "dynamic_metrics", .{ .object = wt_dynamic });

    var engines_obj: ObjectMap = .empty;
    try engines_obj.put(allocator, "wamr", .{ .object = wamr_engine });
    try engines_obj.put(allocator, "wasmtime", .{ .object = wt_engine });

    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = CAPTURE_SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = "wamr-wasmtime-hot-function-capture" });
    try out.put(allocator, "benchmark_report", .{ .object = br_obj });
    try out.put(allocator, "identity", .{ .object = id_obj });
    try out.put(allocator, "hotness", .{ .object = wamr_dm.hotness });
    try out.put(allocator, "engines", .{ .object = engines_obj });
    return out;
}

test "end to end capture and report recommend lever when gate clears" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "e2e");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const capture_path = try std.fs.path.join(allocator, &.{ dir, "capture.json" });
    const report_json = try std.fs.path.join(allocator, &.{ dir, "report.json" });
    const report_md = try std.fs.path.join(allocator, &.{ dir, "report.md" });

    if (builtin.os.tag == .linux and builtin.cpu.arch == .x86_64) {
        const fx = try buildFixture(allocator, io, dir, .{});
        const work_dir = try std.fs.path.join(allocator, &.{ dir, "work" });
        try testing.expectEqual(@as(u8, 0), try runCli(allocator, io, &.{ "capture", "--config", fx.config_path, "--work-dir", work_dir, "--output", capture_path }));
    } else {
        const cap_doc = try buildSyntheticCaptureDoc(allocator, io, dir, .{});
        try writeJson(allocator, io, capture_path, .{ .object = cap_doc });
    }
    try testing.expectEqual(@as(u8, 0), try runCli(allocator, io, &.{ "report", "--capture", capture_path, "--report-json", report_json, "--report-markdown", report_md }));

    const report = try loadJsonObject(allocator, io, report_json, "report");
    const rec = report.get("recommendation").?.object;
    try testing.expectEqualStrings("recommend", rec.get("status").?.string);
    try testing.expectEqualStrings("register-pressure / frame-access reduction", rec.get("selected_lever").?.string);
    try testing.expectEqualStrings("frame_loads", rec.get("primary_metric").?.string);

    var metrics_by_name: std.StringHashMapUnmanaged(ObjectMap) = .empty;
    for (report.get("metrics").?.array.items) |item| {
        try metrics_by_name.put(allocator, item.object.get("metric").?.string, item.object);
    }
    try testing.expectEqual(@as(i64, 15), metrics_by_name.get("native_instructions").?.get("static").?.object.get("wamr").?.object.get("value").?.integer);
    try testing.expectEqual(@as(i64, 7), metrics_by_name.get("native_instructions").?.get("static").?.object.get("wasmtime").?.object.get("value").?.integer);
    try testing.expectEqual(@as(i64, 40), metrics_by_name.get("frame_size_bytes").?.get("static").?.object.get("wamr").?.object.get("value").?.integer);
    try testing.expectEqual(@as(i64, 24), metrics_by_name.get("frame_size_bytes").?.get("static").?.object.get("wasmtime").?.object.get("value").?.integer);
    try testing.expectEqual(@as(i64, 15), metrics_by_name.get("code_size_bytes").?.get("static").?.object.get("wasmtime").?.object.get("value").?.integer);
    try testing.expectEqual(@as(i64, 1), metrics_by_name.get("indirect_dispatch").?.get("static").?.object.get("wamr").?.object.get("value").?.integer);

    const md = try std.Io.Dir.cwd().readFileAlloc(io, report_md, allocator, .limited(1024 * 1024));
    try testing.expect(std.mem.indexOf(u8, md, "Selected lever: `register-pressure / frame-access reduction`") != null);
    try testing.expect(std.mem.indexOf(u8, md, "| Frame-load instructions | 1 instructions | 0 instructions |") != null);
}

test "wasmtime explore continues to final terminator before trailing pool" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "explore-term");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const explore_path = try std.fs.path.join(allocator, &.{ dir, "explore.html" });
    const html =
        \\<script>window.ASM = {"functions": [{"func_index": 3, "name": "f", "instructions": [
        \\  {"address": 0, "bytes": [85], "mnemonic": "pushq", "operands": "%rbp", "wasm_offset": 1},
        \\  {"address": 1, "bytes": [201], "mnemonic": "leave", "operands": "", "wasm_offset": null},
        \\  {"address": 2, "bytes": [195], "mnemonic": "retq", "operands": "", "wasm_offset": null},
        \\  {"address": 3, "bytes": [144], "mnemonic": "nop", "operands": "", "wasm_offset": 2},
        \\  {"address": 4, "bytes": [15, 11], "mnemonic": "ud2", "operands": "", "wasm_offset": null},
        \\  {"address": 6, "bytes": [0, 0], "mnemonic": "addb", "operands": "%al, (%rax)", "wasm_offset": null}
        \\]}]};</script>
    ;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = explore_path, .data = html });
    const res = try parseWasmtimeExplore(allocator, io, explore_path, 3);
    try testing.expectEqual(@as(usize, 6), res.all.len);
    try testing.expectEqual(@as(usize, 5), res.body.len);
    try testing.expectEqualStrings("ud2", res.body[res.body.len - 1].mnemonic);
}

test "wasmtime explore rejects duplicate matching function indices" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "explore-dup");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const explore_path = try std.fs.path.join(allocator, &.{ dir, "explore.html" });
    const html =
        \\<script>window.ASM = {"functions": [
        \\  {"func_index": 3, "name": "f", "instructions": [{"address": 0, "bytes": [195], "mnemonic": "retq", "operands": "", "wasm_offset": 1}]},
        \\  {"func_index": 3, "name": "f", "instructions": [{"address": 0, "bytes": [195], "mnemonic": "retq", "operands": "", "wasm_offset": 1}]}
        \\]};</script>
    ;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = explore_path, .data = html });
    try testing.expectError(error.InvalidComparison, parseWasmtimeExplore(allocator, io, explore_path, 3));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "exactly one function") != null);
}

test "disassembly parser folds byte only continuation lines" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const text =
        \\0000000000000200 <wasm[0]::function[3]::hot_loop>:
        \\ 200: 48 b8 01 02 03 04 05    movabs rax,0x807060504030201
        \\ 207: 06 07 08
        \\ 20a: c3                      ret
        \\
    ;
    const instructions = try parseDisassembly(allocator, text, 3);
    try testing.expectEqual(@as(usize, 2), instructions.len);
    try testing.expectEqual(@as(usize, 10), instructions[0].size);
    try testing.expectEqualSlices(u8, &[_]u8{ 0x48, 0xB8, 1, 2, 3, 4, 5, 6, 7, 8 }, instructions[0].raw_bytes);
    try testing.expectEqual(@as(usize, 10), instructions[1].offset);
}

test "disassembly parser handles multiarch headers and mnemonics" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const text =
        \\0000000000000200 <wasm[0]::function[3]>::hot_loop:
        \\ 200: d10043ff        sub     sp, sp, #0x10
        \\ 204: 8b020020        add     x0, x1, x2
        \\ 208: 910003fd        mov     x29, sp
        \\ 20c: d65f03c0        ret
        \\
    ;
    const instructions = try parseDisassembly(allocator, text, 3);
    var mnems: std.ArrayList([]const u8) = .empty;
    var sizes: std.ArrayList(usize) = .empty;
    for (instructions) |inst| {
        try mnems.append(allocator, inst.mnemonic);
        try sizes.append(allocator, inst.size);
    }
    try testing.expectEqualStrings("sub", mnems.items[0]);
    try testing.expectEqualStrings("add", mnems.items[1]);
    try testing.expectEqualStrings("mov", mnems.items[2]);
    try testing.expectEqualStrings("ret", mnems.items[3]);
    try testing.expectEqualSlices(usize, &[_]usize{ 4, 4, 4, 4 }, sizes.items);
}

test "wasmtime disassembly rejects duplicate matching function headers" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const text =
        \\0000000000000200 <wasm[0]::function[3]::hot_loop>:
        \\ 200: c3                      ret
        \\0000000000000300 <wasm[0]::function[3]::hot_loop>:
        \\ 300: c3                      ret
        \\
    ;
    try testing.expectError(error.InvalidComparison, parseDisassembly(allocator, text, 3));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "duplicate matching function headers") != null);
}

test "wasm parser rejects duplicate sections and bad import kinds" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "wasm-dup");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const func_sec = try testWasmSection(allocator, 3, &[_]u8{ 0x01, 0x00 });
    const code_sec = try testWasmSection(allocator, 10, &[_]u8{ 0x01, 0x02, 0x00, 0x0B });
    const dup_funcs = try std.mem.concat(allocator, u8, &.{ "\x00asm\x01\x00\x00\x00", func_sec, func_sec, code_sec });
    const dup_path = try std.fs.path.join(allocator, &.{ dir, "dup-func.wasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dup_path, .data = dup_funcs });
    try testing.expectError(error.InvalidComparison, parseCoreWasm(allocator, io, dup_path));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "duplicate function section") != null);

    const imp_payload = try std.mem.concat(allocator, u8, &.{
        &[_]u8{0x01},
        try testWasmName(allocator, "env"),
        try testWasmName(allocator, "f"),
        &[_]u8{ 0x00, 0x00 },
    });
    const imp_sec = try testWasmSection(allocator, 2, imp_payload);
    const dup_imps = try std.mem.concat(allocator, u8, &.{ "\x00asm\x01\x00\x00\x00", imp_sec, imp_sec });
    const dup_imp_path = try std.fs.path.join(allocator, &.{ dir, "dup-imp.wasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dup_imp_path, .data = dup_imps });
    try testing.expectError(error.InvalidComparison, parseCoreWasm(allocator, io, dup_imp_path));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "duplicate import section") != null);

    const bad_imp_payload = try std.mem.concat(allocator, u8, &.{
        &[_]u8{0x01},
        try testWasmName(allocator, "env"),
        try testWasmName(allocator, "x"),
        &[_]u8{0x09},
    });
    const bad_imp = try std.mem.concat(allocator, u8, &.{ "\x00asm\x01\x00\x00\x00", try testWasmSection(allocator, 2, bad_imp_payload) });
    const bad_imp_path = try std.fs.path.join(allocator, &.{ dir, "bad-import.wasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = bad_imp_path, .data = bad_imp });
    try testing.expectError(error.InvalidComparison, parseCoreWasm(allocator, io, bad_imp_path));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "unsupported import kind") != null);
}

test "wasm parser handles typed ref tables and memory64" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "wasm-ref-m64");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const table_import = try std.mem.concat(allocator, u8, &.{
        try testWasmName(allocator, "env"),
        try testWasmName(allocator, "tbl"),
        &[_]u8{ 0x01, 0x63, 0x00, 0x00, 0x01 },
    });
    const mem64_import = try std.mem.concat(allocator, u8, &.{
        try testWasmName(allocator, "env"),
        try testWasmName(allocator, "mem64"),
        &[_]u8{ 0x02, 0x05 },
        try testLebU32(allocator, 1),
        try testLebU32(allocator, 65536),
    });
    const func_import = try std.mem.concat(allocator, u8, &.{
        try testWasmName(allocator, "env"),
        try testWasmName(allocator, "fn"),
        &[_]u8{ 0x00, 0x00 },
    });
    const imp_sec = try testWasmSection(allocator, 2, try std.mem.concat(allocator, u8, &.{
        &[_]u8{0x03},
        table_import,
        mem64_import,
        func_import,
    }));
    const func_sec = try testWasmSection(allocator, 3, &[_]u8{ 0x01, 0x00 });
    const code_sec = try testWasmSection(allocator, 10, &[_]u8{ 0x01, 0x02, 0x00, 0x0B });
    const wasm = try std.mem.concat(allocator, u8, &.{ "\x00asm\x01\x00\x00\x00", imp_sec, func_sec, code_sec });
    const wasm_path = try std.fs.path.join(allocator, &.{ dir, "ref-mem64.wasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = wasm_path, .data = wasm });

    const identity = try parseCoreWasm(allocator, io, wasm_path);
    try testing.expectEqual(@as(usize, 1), identity.imported_function_count);
    try testing.expectEqual(@as(usize, 1), identity.local_function_count);
}

test "wasm parser rejects duplicate custom name sections" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "wasm-dup-name");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const entry = try std.mem.concat(allocator, u8, &.{
        &[_]u8{0x01},
        try testLebU32(allocator, 0),
        try testWasmName(allocator, "first"),
    });
    const name_sub = try std.mem.concat(allocator, u8, &.{
        &[_]u8{0x01},
        try testLebU32(allocator, @intCast(entry.len)),
        entry,
    });
    const custom = try testWasmSection(allocator, 0, try std.mem.concat(allocator, u8, &.{
        try testWasmName(allocator, "name"),
        name_sub,
    }));
    const func_sec = try testWasmSection(allocator, 3, &[_]u8{ 0x01, 0x00 });
    const code_sec = try testWasmSection(allocator, 10, &[_]u8{ 0x01, 0x02, 0x00, 0x0B });
    const wasm = try std.mem.concat(allocator, u8, &.{ "\x00asm\x01\x00\x00\x00", func_sec, code_sec, custom, custom });
    const wasm_path = try std.fs.path.join(allocator, &.{ dir, "dup-name.wasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = wasm_path, .data = wasm });

    try testing.expectError(error.InvalidComparison, parseCoreWasm(allocator, io, wasm_path));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "duplicate function-name entry") != null);
}

test "frame size inference handles multiple pushes and rejects dynamic alignment" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const normal = try parseDisassembly(allocator,
        \\0: 55                   push   rbp
        \\1: 41 57                push   r15
        \\3: 48 89 e5             mov    rbp,rsp
        \\6: 48 83 ec 20          sub    rsp,0x20
        \\a: c3                   ret
        \\
    , null);
    const nf = try inferFrameSize(allocator, normal);
    try testing.expectEqual(@as(i64, 48), nf.object.get("value").?.integer);

    const aligned = try parseDisassembly(allocator,
        \\0: 55                   push   rbp
        \\1: 48 89 e5             mov    rbp,rsp
        \\4: 48 83 e4 f0          and    rsp,-16
        \\8: c3                   ret
        \\
    , null);
    const af = try inferFrameSize(allocator, aligned);
    try testing.expectEqualStrings("unavailable", af.object.get("status").?.string);
}

test "frame memory and register classification handles att and intel" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const instructions = try parseDisassembly(allocator,
        \\0: 48 8b 45 f8          movq   -8(%rbp), %rax
        \\4: 48 89 44 24 08       movq   %rax, 8(%rsp)
        \\9: c5 f8 28 c1          vmovaps %xmm1, %xmm0
        \\d: ff d0                callq  *%rax
        \\
    , null);
    const metrics = try aggregateStaticMetrics(allocator, instructions, 15);
    try testing.expectEqual(@as(i64, 1), metrics.get("frame_loads").?.object.get("value").?.integer);
    try testing.expectEqual(@as(i64, 1), metrics.get("frame_stores").?.object.get("value").?.integer);
    try testing.expectEqual(@as(i64, 1), metrics.get("reg_reg_moves").?.object.get("value").?.integer);
    try testing.expectEqual(@as(i64, 1), metrics.get("indirect_dispatch").?.object.get("value").?.integer);
    try testing.expectEqual(@as(i64, 1), metrics.get("calls").?.object.get("value").?.integer);
}

test "wamr jump table finder skips inline relative offsets" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const code = [_]u8{
        0x55,
        0x49,
        0x83,
        0xFB,
        0x02,
        0x0F,
        0x83,
        0x15,
        0x00,
        0x00,
        0x00,
        0x4C,
        0x8D,
        0x15,
        0x0A,
        0x00,
        0x00,
        0x00,
        0x4F,
        0x63,
        0x1C,
        0x9A,
        0x4D,
        0x01,
        0xDA,
        0x41,
        0xFF,
        0xE2,
        0x08,
        0x00,
        0x00,
        0x00,
        0x09,
        0x00,
        0x00,
        0x00,
        0x90,
        0xC3,
    };
    const tables = try findWamrJumpTables(allocator, &code);
    try testing.expectEqual(@as(usize, 1), tables.len);
    try testing.expectEqual(@as(usize, 28), tables[0].start);
    try testing.expectEqual(@as(usize, 36), tables[0].end);
    try testing.expectEqual(@as(usize, 2), tables[0].entries);
}

test "report returns no lever when fewer than two differences clear gate" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "no-lever");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const cap_doc = try buildSyntheticCaptureDoc(allocator, io, dir, .{});
    const report = try compareCapture(allocator, cap_doc, 15.0, 2);
    try testing.expectEqualStrings("no-lever-clears-gate", report.get("recommendation").?.object.get("status").?.string);
    try testing.expect(report.get("recommendation").?.object.get("selected_lever").? == .null);
}

test "incomplete frame origin coverage marks frame dynamic metrics unavailable" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "incomp-orig");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const cap_doc = try buildSyntheticCaptureDoc(allocator, io, dir, .{ .unknown_frame_samples = 10 });
    const wamr_dyn = cap_doc.get("engines").?.object.get("wamr").?.object.get("dynamic_metrics").?.object;
    try testing.expectEqualStrings("unavailable", wamr_dyn.get("frame_loads").?.object.get("status").?.string);
    try testing.expectEqualStrings("unavailable", wamr_dyn.get("frame_stores").?.object.get("status").?.string);
    const report = try compareCapture(allocator, cap_doc, 10.0, 2);
    try testing.expectEqualStrings("no-lever-clears-gate", report.get("recommendation").?.object.get("status").?.string);
}

test "unclassified frame load samples invalidate frame dynamic metrics" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "unclass-fl");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const cap_doc = try buildSyntheticCaptureDoc(allocator, io, dir, .{ .unclassified_frame_load_samples = 5 });
    const wamr_dyn = cap_doc.get("engines").?.object.get("wamr").?.object.get("dynamic_metrics").?.object;
    try testing.expectEqualStrings("unavailable", wamr_dyn.get("frame_loads").?.object.get("status").?.string);
    try testing.expectEqualStrings("measured", wamr_dyn.get("frame_stores").?.object.get("status").?.string);
}

test "mismatched function identity is rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "mismatch-id");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const fx = try buildFixture(allocator, io, dir, .{});
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, fx.config_path, allocator, .limited(1024 * 1024));
    const bad_cfg = try std.mem.replaceOwned(u8, allocator, cfg_text, "\"wasm_index\": 3", "\"wasm_index\": 2");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = fx.config_path, .data = bad_cfg });

    const work_dir = try std.fs.path.join(allocator, &.{ dir, "work" });
    try testing.expectError(error.InvalidComparison, capture(allocator, io, fx.config_path, work_dir));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "mismatch") != null or std.mem.indexOf(u8, lastErrorMessage(), "maps to wasm function") != null);
}

test "tampered sidecar hash is rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "tamper-sidecar");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const fx = try buildFixture(allocator, io, dir, .{});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = fx.sidecar_path, .data = "tampered\n" });

    const work_dir = try std.fs.path.join(allocator, &.{ dir, "work" });
    try testing.expectError(error.InvalidComparison, capture(allocator, io, fx.config_path, work_dir));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "sidecar_sha256") != null);
}

test "unverified wasmtime profile is rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "unver-prof");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    try testing.expectError(error.InvalidComparison, buildSyntheticCaptureDoc(allocator, io, dir, .{
        .wasmtime_profile = true,
        .wasmtime_profile_verified = false,
    }));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "mapping_verified=true") != null);
}

test "verified wasmtime profile uses measured dynamic share delta" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "ver-prof");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const cap_doc = try buildSyntheticCaptureDoc(allocator, io, dir, .{
        .wasmtime_profile = true,
        .wasmtime_profile_verified = true,
    });
    const report = try compareCapture(allocator, cap_doc, 5.0, 2);
    var metrics_by_name: std.StringHashMapUnmanaged(ObjectMap) = .empty;
    for (report.get("metrics").?.array.items) |item| {
        try metrics_by_name.put(allocator, item.object.get("metric").?.string, item.object);
    }
    const fl_hr = metrics_by_name.get("frame_loads").?.get("conservative_headroom").?.object;
    try testing.expectEqualStrings("measured-run-sample-share-delta", fl_hr.get("method").?.string);
    try testing.expectApproxEqAbs(@as(f64, 16.0), try requireNumber(fl_hr.get("percent_of_run"), "", 0.0), 1e-6);
}

test "example config matches schema shape" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;

    const schema = try loadJsonObject(allocator, io, "tests/benchmarks/keyvault/hot-comparison.schema.json", "schema");
    const example = try loadJsonObject(allocator, io, "tests/benchmarks/keyvault/hot-comparison.example.json", "example");
    const required = schema.get("required").?.array.items;
    for (required) |req_key| {
        try testing.expect(example.contains(req_key.string));
    }
    try testing.expectEqual(@as(i64, 1), example.get("schema_version").?.integer);
}
