//! Measure AOT function-entry cancel-poll cost in a leaf-call-heavy thread.
const std = @import("std");
const builtin = @import("builtin");
const benchmark_schema = @import("benchmark_schema.zig");
const benchmark_stats = @import("benchmark_stats.zig");
const compare_hot_function = @import("compare_hot_function.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const Array = std.json.Array;

pub const KIND = "leaf-cancel-cost";
pub const SCHEMA_VERSION: i64 = 1;
pub const AOT_MAGIC: u32 = 0x746F6100; // "\x00aot"
pub const AOT_VERSION: u32 = 11;
pub const AOT_TEXT_SECTION: u32 = 2;
pub const AOT_FUNCTION_SECTION: u32 = 3;
pub const AOT_EXPORT_SECTION: u32 = 4;
pub const AOT_IMPORT_SECTION: u32 = 8;
pub const LEAF_UNROLL: i64 = 64;
pub const LEAF_SEED: u64 = 0x243F6A8885A308D3;
pub const LEAF_STEP: u64 = 0x9E3779B97F4A7C15;
pub const MAX_LEAF_CALLS: i64 = 10_000_000_000;
pub const FIXTURE = "tests/benchmarks/leaf-cancel-cost/leaf_calls.wasm";

pub const SOURCE_PATHS = [_][]const u8{
    ".github/workflows/leaf-cancel-cost.yml",
    "tests/benchmarks/leaf-cancel-cost/README.md",
    "tests/benchmarks/leaf-cancel-cost/leaf_calls.c",
    "tests/benchmarks/leaf-cancel-cost/build-fixture.sh",
    "tests/benchmarks/leaf-cancel-cost/fixtures.sha256",
    "tests/benchmarks/leaf-cancel-cost/report.schema.json",
    "scripts/bench_leaf_cancel_cost.zig",
    "scripts/benchmark_schema.zig",
    "scripts/compare_hot_function.zig",
};

pub const CANCEL_POLL_X86_64 = [_]u8{
    0x83, 0xbb, 0xb0, 0x01, 0x00, 0x00, 0x00, 0x74, 0x0c, 0x48, 0x89,
    0xdf, 0x48, 0x8b, 0x87, 0xb8, 0x01, 0x00, 0x00, 0xff, 0xd0,
};

pub const CANCEL_POLL_AARCH64 = [_]u8{
    0x70, 0xb2, 0x41, 0xb9, // 0xB941B270
    0x90, 0x00, 0x00, 0x34, // 0x34000090
    0xe0, 0x03, 0x13, 0xaa, // 0xAA1303E0
    0x70, 0xde, 0x40, 0xf9, // 0xF940DE70
    0x00, 0x02, 0x3f, 0xd6, // 0xD63F0200
};

pub const CANCEL_POLL_ENTRY_PREFIX_X86_64 = [_]u8{
    0x48, 0x89, 0xfb, 0x4c, 0x8b, 0xbb, 0x00, 0x00, 0x00, 0x00,
};

pub const CANCEL_POLL_ENTRY_PREFIX_AARCH64 = [_]u8{
    0xf3, 0x03, 0x00, 0xaa, // 0xAA0003F3
    0x74, 0x02, 0x40, 0xf9, // 0xF9400274
    0xa1, 0x0f, 0x00, 0xf9, // 0xF9000FA1
};

pub fn cancelPollSequence(arch: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, arch, "x86_64")) return &CANCEL_POLL_X86_64;
    if (std.mem.eql(u8, arch, "aarch64")) return &CANCEL_POLL_AARCH64;
    return null;
}

pub fn cancelPollEntryPrefixSuffix(arch: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, arch, "x86_64")) return &CANCEL_POLL_ENTRY_PREFIX_X86_64;
    if (std.mem.eql(u8, arch, "aarch64")) return &CANCEL_POLL_ENTRY_PREFIX_AARCH64;
    return null;
}

pub const GUEST_KEYS = [_][]const u8{
    "kind",
    "leaf_calls",
    "leaf_unroll",
    "batches",
    "checksum",
    "clock_id",
    "raw_elapsed_ns",
    "timing_overhead_ns",
    "elapsed_ns",
    "timing_overhead_ppm",
};

pub const HarnessError = error{
    InvalidHarness,
    OutOfMemory,
};

var last_error_buf: [2048]u8 = undefined;
var last_error_len: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) HarnessError {
    const slice = std.fmt.bufPrint(&last_error_buf, fmt, args) catch blk: {
        const msg = "leaf cancel-cost benchmark error (message truncated)";
        @memcpy(last_error_buf[0..msg.len], msg);
        break :blk last_error_buf[0..msg.len];
    };
    last_error_len = slice.len;
    return error.InvalidHarness;
}

pub fn lastErrorMessage() []const u8 {
    return last_error_buf[0..last_error_len];
}

pub const Build = struct {
    name: []const u8,
    prefix: []const u8,
    wamr: []const u8,
    wamrc: ?[]const u8,
    command: []const []const u8,
    cache_key: []const u8,
};

pub const AotExport = struct {
    name: []const u8,
    kind: u8,
    index: u32,
};

pub const AotSection = struct {
    section_type: u32,
    payload: []const u8,
};

pub const AotImage = struct {
    path: []const u8,
    data: []const u8,
    sections: []const AotSection,
    text: []const u8,
    function_offsets: []const u32,
    function_type_indices: []const u32,
    exports: []const AotExport,
    imported_function_count: u32,

    pub fn functionCode(self: AotImage, local_index: usize) HarnessError![]const u8 {
        if (local_index >= self.function_offsets.len) {
            return fail("{s}: local function {d} is out of range", .{ self.path, local_index });
        }
        const start: usize = self.function_offsets[local_index];
        const end: usize = if (local_index + 1 < self.function_offsets.len)
            self.function_offsets[local_index + 1]
        else
            self.text.len;
        return self.text[start..end];
    }
};

pub const ExpectedResult = struct {
    kind: []const u8,
    leaf_calls: i64,
    leaf_unroll: i64,
    batches: i64,
    checksum: u64,
    clock_id: []const u8,

    pub fn toObjectMap(self: ExpectedResult, allocator: Allocator) !ObjectMap {
        var obj: ObjectMap = .empty;
        try obj.put(allocator, "kind", .{ .string = self.kind });
        try obj.put(allocator, "leaf_calls", .{ .integer = self.leaf_calls });
        try obj.put(allocator, "leaf_unroll", .{ .integer = self.leaf_unroll });
        try obj.put(allocator, "batches", .{ .integer = self.batches });
        const checksum_str = try std.fmt.allocPrint(allocator, "{d}", .{self.checksum});
        try obj.put(allocator, "checksum", .{ .number_string = checksum_str });
        try obj.put(allocator, "clock_id", .{ .string = self.clock_id });
        return obj;
    }
};

pub fn expectedResult(calls: i64) HarnessError!ExpectedResult {
    if (calls <= 0 or @rem(calls, LEAF_UNROLL) != 0) {
        return fail("leaf calls must be a positive multiple of {d}", .{LEAF_UNROLL});
    }
    const calls_u64: u64 = @intCast(calls);
    return .{
        .kind = "leaf-cancel-cost-result",
        .leaf_calls = calls,
        .leaf_unroll = LEAF_UNROLL,
        .batches = @divExact(calls, LEAF_UNROLL),
        .checksum = LEAF_SEED +% (LEAF_STEP *% calls_u64),
        .clock_id = "wasi-process-cputime",
    };
}

fn parseStrictJsonI64(val: Value, key: []const u8) HarnessError!i64 {
    return switch (val) {
        .integer => |i| i,
        .number_string => |s| std.fmt.parseInt(i64, s, 10) catch return fail("{s} must be an integer", .{key}),
        else => fail("{s} must be an integer", .{key}),
    };
}

fn parseStrictJsonU64(val: Value, key: []const u8) HarnessError!u64 {
    return switch (val) {
        .integer => |i| if (i >= 0) @intCast(i) else return fail("{s} must be a non-negative integer", .{key}),
        .number_string => |s| std.fmt.parseInt(u64, s, 10) catch return fail("{s} must be an integer", .{key}),
        else => fail("{s} must be an integer", .{key}),
    };
}

pub fn parseGuestResult(
    allocator: Allocator,
    stdout: []const u8,
    calls: i64,
    minimum_interval_ns: i64,
    enforce_quality: bool,
) !ObjectMap {
    var non_empty_line: ?[]const u8 = null;
    var line_count: usize = 0;
    var it = std.mem.splitScalar(u8, stdout, '\n');
    while (it.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len > 0) {
            non_empty_line = line;
            line_count += 1;
        }
    }
    if (line_count != 1) {
        return fail("expected one guest JSON line, got {d}", .{line_count});
    }
    const parsed = std.json.parseFromSlice(Value, allocator, non_empty_line.?, .{
        .duplicate_field_behavior = .@"error",
        .parse_numbers = false,
    }) catch |err| switch (err) {
        error.DuplicateField => return fail("duplicate guest key", .{}),
        else => return fail("guest output is not valid JSON", .{}),
    };
    if (parsed.value != .object) {
        return fail("guest output is not valid JSON", .{});
    }
    const raw_obj = parsed.value.object;
    if (raw_obj.count() != GUEST_KEYS.len) {
        return fail("guest fields mismatch", .{});
    }
    for (GUEST_KEYS) |k| {
        if (!raw_obj.contains(k)) {
            return fail("guest fields mismatch: missing {s}", .{k});
        }
    }

    const exp = try expectedResult(calls);
    const kind_val = raw_obj.get("kind").?;
    if (kind_val != .string or !std.mem.eql(u8, kind_val.string, exp.kind)) {
        return fail("guest result mismatch for kind", .{});
    }
    const clock_val = raw_obj.get("clock_id").?;
    if (clock_val != .string or !std.mem.eql(u8, clock_val.string, exp.clock_id)) {
        return fail("guest result mismatch for clock_id", .{});
    }
    const guest_calls = try parseStrictJsonI64(raw_obj.get("leaf_calls").?, "leaf_calls");
    if (guest_calls != exp.leaf_calls) {
        return fail("guest result mismatch for leaf_calls: expected {d}, got {d}", .{ exp.leaf_calls, guest_calls });
    }
    const guest_unroll = try parseStrictJsonI64(raw_obj.get("leaf_unroll").?, "leaf_unroll");
    if (guest_unroll != exp.leaf_unroll) {
        return fail("guest result mismatch for leaf_unroll: expected {d}, got {d}", .{ exp.leaf_unroll, guest_unroll });
    }
    const guest_batches = try parseStrictJsonI64(raw_obj.get("batches").?, "batches");
    if (guest_batches != exp.batches) {
        return fail("guest result mismatch for batches: expected {d}, got {d}", .{ exp.batches, guest_batches });
    }
    const guest_checksum = try parseStrictJsonU64(raw_obj.get("checksum").?, "checksum");
    if (guest_checksum != exp.checksum) {
        return fail("guest result mismatch for checksum: expected {d}, got {d}", .{ exp.checksum, guest_checksum });
    }

    const raw = try parseStrictJsonI64(raw_obj.get("raw_elapsed_ns").?, "raw_elapsed_ns");
    const overhead = try parseStrictJsonI64(raw_obj.get("timing_overhead_ns").?, "timing_overhead_ns");
    const elapsed = try parseStrictJsonI64(raw_obj.get("elapsed_ns").?, "elapsed_ns");
    const ppm = try parseStrictJsonI64(raw_obj.get("timing_overhead_ppm").?, "timing_overhead_ppm");

    if (raw <= 0 or overhead < 0 or elapsed <= 0 or raw - overhead != elapsed) {
        return fail("guest timing values are inconsistent", .{});
    }
    if (ppm != @divFloor(overhead * 1_000_000, raw)) {
        return fail("guest timing_overhead_ppm is inconsistent", .{});
    }
    if (enforce_quality and 99 * overhead >= elapsed) {
        return fail("guest timing overhead is not below one percent", .{});
    }
    if (enforce_quality and elapsed < minimum_interval_ns) {
        return fail("guest interval {d}ns is below required {d}ns", .{ elapsed, minimum_interval_ns });
    }

    var out = try exp.toObjectMap(allocator);
    try out.put(allocator, "raw_elapsed_ns", .{ .integer = raw });
    try out.put(allocator, "timing_overhead_ns", .{ .integer = overhead });
    try out.put(allocator, "elapsed_ns", .{ .integer = elapsed });
    try out.put(allocator, "timing_overhead_ppm", .{ .integer = ppm });
    return out;
}

pub fn roundCalls(value: i64) HarnessError!i64 {
    if (value <= 0) return fail("leaf call count must be positive", .{});
    return @divFloor(value + LEAF_UNROLL - 1, LEAF_UNROLL) * LEAF_UNROLL;
}

pub fn selectCalls(
    pilot_calls: i64,
    pilot_elapsed_ns: i64,
    target_interval_ns: i64,
) HarnessError!i64 {
    if (pilot_elapsed_ns <= 0 or target_interval_ns <= 0) {
        return fail("pilot and target intervals must be positive", .{});
    }
    const scaled_f = @ceil(@as(f64, @floatFromInt(pilot_calls)) * @as(f64, @floatFromInt(target_interval_ns)) / @as(f64, @floatFromInt(pilot_elapsed_ns)));
    if (scaled_f > @as(f64, @floatFromInt(MAX_LEAF_CALLS))) {
        return fail("selected leaf call count exceeds safety limit {d}", .{MAX_LEAF_CALLS});
    }
    const selected: i64 = @intFromFloat(scaled_f);
    if (selected > MAX_LEAF_CALLS) {
        return fail("selected leaf call count exceeds safety limit {d}", .{MAX_LEAF_CALLS});
    }
    return roundCalls(@max(selected, LEAF_UNROLL));
}

pub fn sha256Bytes(allocator: Allocator, data: []const u8) ![]const u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

pub fn sha256File(allocator: Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch {
        return fail("missing or unreadable file: {s}", .{path});
    };
    defer allocator.free(data);
    return sha256Bytes(allocator, data);
}

fn cacheKey(allocator: Allocator, value: Value) ![]const u8 {
    const key = try benchmark_schema.cacheKey(allocator, value);
    return allocator.dupe(u8, &key);
}

pub fn atomicWriteJson(allocator: Allocator, io: std.Io, path: []const u8, value: Value) !void {
    if (std.fs.path.dirname(path)) |parent| {
        if (parent.len > 0) std.Io.Dir.cwd().createDirPath(io, parent) catch {};
    }
    var rand_bytes: [8]u8 = undefined;
    io.random(&rand_bytes);
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.{x}.tmp", .{
        path,
        std.mem.readInt(u64, &rand_bytes, .little),
    });
    const canonical = try benchmark_schema.canonicalJson(allocator, value);
    const payload = try std.fmt.allocPrint(allocator, "{s}\n", .{canonical});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp_path, .data = payload });
    try std.Io.Dir.cwd().rename(tmp_path, std.Io.Dir.cwd(), path, io);
}

fn readAllPipe(allocator: Allocator, io: std.Io, file: std.Io.File) ![]u8 {
    var read_buf: [4096]u8 = undefined;
    var file_reader = file.readerStreaming(io, &read_buf);
    return file_reader.interface.allocRemaining(allocator, .limited(64 * 1024 * 1024));
}

pub fn runProcess(
    allocator: Allocator,
    io: std.Io,
    argv: []const []const u8,
    cwd: ?[]const u8,
    environ_map: ?*const std.process.Environ.Map,
) !struct { code: u8, stdout: []u8, stderr: []u8 } {
    var child = std.process.spawn(io, .{
        .argv = argv,
        .cwd = if (cwd) |c| .{ .path = c } else .inherit,
        .environ_map = environ_map,
        .stdout = .pipe,
        .stderr = .pipe,
    }) catch {
        return fail("failed to execute command: {s}", .{argv[0]});
    };
    const stdout = try readAllPipe(allocator, io, child.stdout.?);
    const stderr = try readAllPipe(allocator, io, child.stderr.?);
    const term = child.wait(io) catch {
        return fail("failed waiting for command: {s}", .{argv[0]});
    };
    const code: u8 = switch (term) {
        .exited => |c| c,
        else => 255,
    };
    return .{ .code = code, .stdout = stdout, .stderr = stderr };
}

pub fn commandIdentity(allocator: Allocator, io: std.Io, argv: []const []const u8) ![]const u8 {
    const res = runProcess(allocator, io, argv, null, null) catch {
        return allocator.dupe(u8, "unavailable");
    };
    const text = if (res.stdout.len > 0) res.stdout else res.stderr;
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) {
        return std.fmt.allocPrint(allocator, "exit {d}", .{res.code});
    }
    var it = std.mem.splitScalar(u8, trimmed, '\n');
    return allocator.dupe(u8, std.mem.trimEnd(u8, it.first(), "\r"));
}

fn readU32(payload: []const u8, offset: usize, label: []const u8) HarnessError!struct { val: u32, next: usize } {
    if (offset + 4 > payload.len) return fail("{s}: truncated u32", .{label});
    return .{
        .val = std.mem.readInt(u32, payload[offset..][0..4], .little),
        .next = offset + 4,
    };
}

fn readBytes(payload: []const u8, offset: usize, size: usize, label: []const u8) HarnessError!struct { val: []const u8, next: usize } {
    if (offset + size > payload.len) return fail("{s}: truncated byte range", .{label});
    return .{
        .val = payload[offset .. offset + size],
        .next = offset + size,
    };
}

fn readName(allocator: Allocator, payload: []const u8, offset: usize, label: []const u8) !struct { val: []const u8, next: usize } {
    const sz = try readU32(payload, offset, label);
    const raw = try readBytes(payload, sz.next, sz.val, label);
    if (!std.unicode.utf8ValidateSlice(raw.val)) {
        return fail("{s}: name is not UTF-8", .{label});
    }
    return .{
        .val = try allocator.dupe(u8, raw.val),
        .next = raw.next,
    };
}

fn parseFunctionSection(
    allocator: Allocator,
    payload: []const u8,
    label: []const u8,
) !struct { offsets: []u32, type_indices: []u32 } {
    const cnt = try readU32(payload, 0, label);
    const expected_size: usize = 4 + @as(usize, cnt.val) * 8;
    if (payload.len != expected_size) {
        return fail("{s}: function section size {d} != {d}", .{ label, payload.len, expected_size });
    }
    const offsets = try allocator.alloc(u32, cnt.val);
    const type_indices = try allocator.alloc(u32, cnt.val);
    var offset = cnt.next;
    for (0..cnt.val) |i| {
        const fo = try readU32(payload, offset, label);
        const ti = try readU32(payload, fo.next, label);
        offsets[i] = fo.val;
        type_indices[i] = ti.val;
        offset = ti.next;
    }
    return .{ .offsets = offsets, .type_indices = type_indices };
}

fn parseExports(allocator: Allocator, payload: []const u8, label: []const u8) ![]AotExport {
    const cnt = try readU32(payload, 0, label);
    var offset = cnt.next;
    var exports = try allocator.alloc(AotExport, cnt.val);
    for (0..cnt.val) |i| {
        const nm = try readName(allocator, payload, offset, label);
        const rk = try readBytes(payload, nm.next, 1, label);
        const idx = try readU32(payload, rk.next, label);
        exports[i] = .{ .name = nm.val, .kind = rk.val[0], .index = idx.val };
        offset = idx.next;
    }
    if (offset != payload.len) return fail("{s}: trailing export bytes", .{label});
    return exports;
}

fn parseImportedFunctionCount(allocator: Allocator, payload: []const u8, label: []const u8) !u32 {
    const cnt = try readU32(payload, 0, label);
    var offset = cnt.next;
    var functions: u32 = 0;
    for (0..cnt.val) |_| {
        const m = try readName(allocator, payload, offset, label);
        const f = try readName(allocator, payload, m.next, label);
        const rk = try readBytes(payload, f.next, 1, label);
        const kind = rk.val[0];
        offset = rk.next;
        if (kind == 0) {
            functions += 1;
            offset = (try readU32(payload, offset, label)).next;
        } else if (kind == 1) {
            offset = (try readBytes(payload, offset, 1, label)).next;
            offset = (try readU32(payload, offset, label)).next;
            const hm = try readBytes(payload, offset, 1, label);
            offset = hm.next;
            if (hm.val[0] != 0 and hm.val[0] != 1) return fail("{s}: invalid table maximum marker", .{label});
            if (hm.val[0] == 1) offset = (try readU32(payload, offset, label)).next;
        } else if (kind == 2) {
            offset = (try readU32(payload, offset, label)).next;
            const hm = try readBytes(payload, offset, 1, label);
            offset = hm.next;
            if (hm.val[0] != 0 and hm.val[0] != 1) return fail("{s}: invalid memory maximum marker", .{label});
            if (hm.val[0] == 1) offset = (try readU32(payload, offset, label)).next;
            offset = (try readBytes(payload, offset, 2, label)).next;
        } else if (kind == 3) {
            offset = (try readBytes(payload, offset, 2, label)).next;
        } else if (kind == 4) {
            offset = (try readU32(payload, offset, label)).next;
        } else {
            return fail("{s}: unsupported import kind {d}", .{ label, kind });
        }
    }
    if (offset != payload.len) return fail("{s}: trailing import bytes", .{label});
    return functions;
}

pub fn parseAot(allocator: Allocator, io: std.Io, path: []const u8) !AotImage {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch {
        return fail("not a WAMR AOT artifact: {s}", .{path});
    };
    if (data.len < 8 or std.mem.readInt(u32, data[0..4], .little) != AOT_MAGIC) {
        return fail("not a WAMR AOT artifact: {s}", .{path});
    }
    const version = std.mem.readInt(u32, data[4..8], .little);
    if (version != AOT_VERSION) {
        return fail("unsupported AOT version {d}", .{version});
    }
    var offset: usize = 8;
    var sections: std.ArrayList(AotSection) = .empty;
    var by_type: std.AutoHashMapUnmanaged(u32, []const u8) = .empty;
    while (offset + 8 <= data.len) {
        const section_type = std.mem.readInt(u32, data[offset..][0..4], .little);
        const size: usize = std.mem.readInt(u32, data[offset + 4 ..][0..4], .little);
        offset += 8;
        if (offset + size > data.len) {
            return fail("truncated AOT artifact: {s}", .{path});
        }
        if (by_type.contains(section_type)) {
            return fail("{s}: duplicate AOT section {d}", .{ path, section_type });
        }
        const payload = data[offset .. offset + size];
        try sections.append(allocator, .{ .section_type = section_type, .payload = payload });
        try by_type.put(allocator, section_type, payload);
        offset += size;
    }
    if (offset != data.len) {
        return fail("{s}: trailing truncated AOT header", .{path});
    }
    for ([_]u32{ AOT_TEXT_SECTION, AOT_FUNCTION_SECTION, AOT_EXPORT_SECTION, AOT_IMPORT_SECTION }) |st| {
        if (!by_type.contains(st)) {
            return fail("{s}: missing AOT section {d}", .{ path, st });
        }
    }
    const fn_label = try std.fmt.allocPrint(allocator, "{s} function section", .{path});
    const fn_sec = try parseFunctionSection(allocator, by_type.get(AOT_FUNCTION_SECTION).?, fn_label);
    const text = by_type.get(AOT_TEXT_SECTION).?;
    if (fn_sec.offsets.len == 0 or fn_sec.offsets[0] != 0 or fn_sec.offsets[fn_sec.offsets.len - 1] >= text.len) {
        return fail("{s}: invalid AOT function offsets", .{path});
    }
    for (1..fn_sec.offsets.len) |i| {
        if (fn_sec.offsets[i] <= fn_sec.offsets[i - 1]) {
            return fail("{s}: invalid AOT function offsets", .{path});
        }
    }
    const exp_label = try std.fmt.allocPrint(allocator, "{s} export section", .{path});
    const imp_label = try std.fmt.allocPrint(allocator, "{s} import section", .{path});
    return .{
        .path = try allocator.dupe(u8, path),
        .data = data,
        .sections = try sections.toOwnedSlice(allocator),
        .text = text,
        .function_offsets = fn_sec.offsets,
        .function_type_indices = fn_sec.type_indices,
        .exports = try parseExports(allocator, by_type.get(AOT_EXPORT_SECTION).?, exp_label),
        .imported_function_count = try parseImportedFunctionCount(allocator, by_type.get(AOT_IMPORT_SECTION).?, imp_label),
    };
}

pub const PollRange = struct {
    start: usize,
    end: usize,
};

fn countNonOverlapping(haystack: []const u8, needle: []const u8) usize {
    if (needle.len == 0) return 0;
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, pos, needle)) |idx| {
        count += 1;
        pos = idx + needle.len;
    }
    return count;
}

fn pollRanges(allocator: Allocator, code: []const u8, sequence: []const u8) ![]PollRange {
    var ranges: std.ArrayList(PollRange) = .empty;
    var offset: usize = 0;
    while (std.mem.indexOfPos(u8, code, offset, sequence)) |start| {
        try ranges.append(allocator, .{ .start = start, .end = start + sequence.len });
        offset = start + sequence.len;
    }
    return ranges.toOwnedSlice(allocator);
}

fn normalizeLocalOffset(offset: i64, ranges: []const PollRange, label: []const u8) HarnessError!i64 {
    var removed: i64 = 0;
    for (ranges) |r| {
        const start: i64 = @intCast(r.start);
        const end: i64 = @intCast(r.end);
        if (start < offset and offset < end) {
            return fail("{s}: control-flow target enters a cancel poll", .{label});
        }
        if (end <= offset) {
            removed += end - start;
        }
    }
    return offset - removed;
}

fn functionIndexForOffset(image: AotImage, offset: i64) HarnessError!usize {
    if (offset < 0) return fail("{s}: text target {d} is out of range", .{ image.path, offset });
    const uoff: usize = @intCast(offset);
    if (uoff == image.text.len) return image.function_offsets.len;
    for (image.function_offsets, 0..) |start_u32, index| {
        const start: usize = start_u32;
        const end: usize = if (index + 1 < image.function_offsets.len)
            image.function_offsets[index + 1]
        else
            image.text.len;
        if (start <= uoff and uoff < end) return index;
    }
    return fail("{s}: text target {d} is out of range", .{ image.path, offset });
}

fn normalizeOnTextOffset(
    allocator: Allocator,
    enabled: AotImage,
    disabled: AotImage,
    all_poll_ranges: []const []const PollRange,
    offset: i64,
) !i64 {
    const function_index = try functionIndexForOffset(enabled, offset);
    if (function_index == enabled.function_offsets.len) {
        return @intCast(disabled.text.len);
    }
    const local = offset - @as(i64, enabled.function_offsets[function_index]);
    const label = try std.fmt.allocPrint(allocator, "{s} function {d}", .{ enabled.path, function_index });
    const norm_local = try normalizeLocalOffset(local, all_poll_ranges[function_index], label);
    return @as(i64, disabled.function_offsets[function_index]) + norm_local;
}

fn x86RelativeInstruction(raw: []const u8) ?struct { opcode: []const u8, rel: i64 } {
    if (raw.len == 5 and (raw[0] == 0xE8 or raw[0] == 0xE9)) {
        return .{ .opcode = raw[0..1], .rel = std.mem.readInt(i32, raw[1..5], .little) };
    }
    if (raw.len == 2 and (raw[0] == 0xEB or (raw[0] >= 0x70 and raw[0] <= 0x7F))) {
        return .{ .opcode = raw[0..1], .rel = @as(i8, @bitCast(raw[1])) };
    }
    if (raw.len == 6 and raw[0] == 0x0F and (raw[1] >= 0x80 and raw[1] <= 0x8F)) {
        return .{ .opcode = raw[0..2], .rel = std.mem.readInt(i32, raw[2..6], .little) };
    }
    return null;
}

fn disassembleX86Function(
    allocator: Allocator,
    io: std.Io,
    code: []const u8,
    work_dir: []const u8,
    label: []const u8,
) !struct { instructions: []compare_hot_function.Instruction, tables: []compare_hot_function.JumpTableRange } {
    const tables = compare_hot_function.findWamrJumpTables(allocator, code) catch {
        return fail("{s}: invalid jump table", .{label});
    };
    const disassembly = try allocator.dupe(u8, code);
    for (tables) |t| {
        @memset(disassembly[t.start..t.end], 0x90);
    }
    std.Io.Dir.cwd().createDirPath(io, work_dir) catch {};
    const scratch_name = try std.fmt.allocPrint(allocator, "{s}.bin", .{label});
    const scratch = try std.fs.path.join(allocator, &.{ work_dir, scratch_name });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = scratch, .data = disassembly });
    defer std.Io.Dir.cwd().deleteFile(io, scratch) catch {};

    const command = [_][]const u8{
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
    };
    const res = try runProcess(allocator, io, &command, null, null);
    if (res.code != 0) {
        return fail("{s}: objdump failed with exit {d}: {s}", .{
            label,
            res.code,
            std.mem.trim(u8, res.stderr, " \t\r\n"),
        });
    }
    const instructions = compare_hot_function.parseDisassembly(allocator, res.stdout, null) catch {
        return fail("{s}: failed to parse disassembly", .{label});
    };
    var max_end: usize = 0;
    for (instructions) |inst| {
        if (inst.offset + inst.size > max_end) max_end = inst.offset + inst.size;
        if (std.mem.eql(u8, inst.mnemonic, "(bad)") or std.mem.eql(u8, inst.mnemonic, ".byte")) {
            return fail("{s}: undecodable x86 instruction", .{label});
        }
    }
    if (max_end != code.len) {
        return fail("{s}: disassembly did not cover the function", .{label});
    }
    var filtered: std.ArrayList(compare_hot_function.Instruction) = .empty;
    for (instructions) |inst| {
        var in_table = false;
        for (tables) |t| {
            if (t.start <= inst.offset and inst.offset < t.end) {
                in_table = true;
                break;
            }
        }
        if (!in_table) try filtered.append(allocator, inst);
    }
    return .{
        .instructions = try filtered.toOwnedSlice(allocator),
        .tables = tables,
    };
}

fn compareX86Function(
    allocator: Allocator,
    io: std.Io,
    function_index: usize,
    enabled: AotImage,
    disabled: AotImage,
    enabled_code: []const u8,
    disabled_code: []const u8,
    all_poll_ranges: []const []const PollRange,
    work_dir: []const u8,
) !struct { layout_fields: usize, table_entries: usize } {
    const on_label = try std.fmt.allocPrint(allocator, "on-{d}", .{function_index});
    const off_label = try std.fmt.allocPrint(allocator, "off-{d}", .{function_index});
    const on_dis = try disassembleX86Function(allocator, io, enabled_code, work_dir, on_label);
    const off_dis = try disassembleX86Function(allocator, io, disabled_code, work_dir, off_label);
    const function_polls = all_poll_ranges[function_index];
    var poll_instruction_offsets: std.AutoHashMapUnmanaged(usize, void) = .empty;
    for (function_polls) |pr| {
        var selected: std.ArrayList(compare_hot_function.Instruction) = .empty;
        var raw_concat: std.ArrayList(u8) = .empty;
        for (on_dis.instructions) |item| {
            if (pr.start <= item.offset and item.offset < pr.end) {
                try selected.append(allocator, item);
                try raw_concat.appendSlice(allocator, item.raw_bytes);
            }
        }
        if (selected.items.len == 0 or
            selected.items[0].offset != pr.start or
            selected.items[selected.items.len - 1].offset + selected.items[selected.items.len - 1].size != pr.end or
            !std.mem.eql(u8, raw_concat.items, &CANCEL_POLL_X86_64))
        {
            return fail("enabled function {d}: malformed cancel poll", .{function_index});
        }
        for (selected.items) |item| {
            try poll_instruction_offsets.put(allocator, item.offset, {});
        }
    }
    var on_retained: std.ArrayList(compare_hot_function.Instruction) = .empty;
    for (on_dis.instructions) |item| {
        if (!poll_instruction_offsets.contains(item.offset)) {
            try on_retained.append(allocator, item);
        }
    }
    if (on_retained.items.len != off_dis.instructions.len) {
        return fail("function {d}: normalized instruction count differs", .{function_index});
    }
    const fn_label = try std.fmt.allocPrint(allocator, "enabled function {d}", .{function_index});
    var layout_fields: usize = 0;
    for (on_retained.items, off_dis.instructions) |on_item, off_item| {
        const normalized_offset = try normalizeLocalOffset(@intCast(on_item.offset), function_polls, fn_label);
        if (normalized_offset != @as(i64, @intCast(off_item.offset)) or
            on_item.size != off_item.size or
            !std.mem.eql(u8, on_item.mnemonic, off_item.mnemonic))
        {
            return fail("function {d}: normalized instruction layout differs", .{function_index});
        }
        if (std.mem.eql(u8, on_item.raw_bytes, off_item.raw_bytes)) continue;
        const on_rel = x86RelativeInstruction(on_item.raw_bytes);
        const off_rel = x86RelativeInstruction(off_item.raw_bytes);
        if (on_rel == null or off_rel == null or !std.mem.eql(u8, on_rel.?.opcode, off_rel.?.opcode)) {
            return fail("function {d}: unrelated x86 instruction difference at normalized offset {d}", .{
                function_index,
                off_item.offset,
            });
        }
        const on_target = @as(i64, enabled.function_offsets[function_index]) +
            @as(i64, @intCast(on_item.offset)) +
            @as(i64, @intCast(on_item.size)) +
            on_rel.?.rel;
        const off_target = @as(i64, disabled.function_offsets[function_index]) +
            @as(i64, @intCast(off_item.offset)) +
            @as(i64, @intCast(off_item.size)) +
            off_rel.?.rel;
        if ((try normalizeOnTextOffset(allocator, enabled, disabled, all_poll_ranges, on_target)) != off_target) {
            return fail("function {d}: x86 control-flow target differs", .{function_index});
        }
        layout_fields += 1;
    }
    if (on_dis.tables.len != off_dis.tables.len) {
        return fail("function {d}: jump-table count differs", .{function_index});
    }
    var table_entries: usize = 0;
    for (on_dis.tables, off_dis.tables) |on_table, off_table| {
        const norm_table_start = try normalizeLocalOffset(@intCast(on_table.start), function_polls, fn_label);
        if (on_table.entries != off_table.entries or norm_table_start != @as(i64, @intCast(off_table.start))) {
            return fail("function {d}: jump-table layout differs", .{function_index});
        }
        for (0..on_table.entries) |entry_index| {
            const on_entry = on_table.start + entry_index * 4;
            const off_entry = off_table.start + entry_index * 4;
            const on_rel: i64 = std.mem.readInt(i32, enabled_code[on_entry..][0..4], .little);
            const off_rel: i64 = std.mem.readInt(i32, disabled_code[off_entry..][0..4], .little);
            const on_target = @as(i64, @intCast(on_table.start)) + on_rel;
            const off_target = @as(i64, @intCast(off_table.start)) + off_rel;
            if ((try normalizeLocalOffset(on_target, function_polls, fn_label)) != off_target) {
                return fail("function {d}: jump-table target differs", .{function_index});
            }
            table_entries += 1;
        }
    }
    return .{ .layout_fields = layout_fields, .table_entries = table_entries };
}

fn signExtend(value: u32, comptime bits: u5) i64 {
    const sign: i64 = @as(i64, 1) << (bits - 1);
    const v: i64 = @intCast(value);
    return (v ^ sign) - sign;
}

fn aarch64RelativeInstruction(word: u32) ?struct { masked: u32, rel: i64 } {
    if ((word & 0xFC000000) == 0x14000000 or (word & 0xFC000000) == 0x94000000) {
        return .{ .masked = word & 0xFC000000, .rel = signExtend(word & 0x03FFFFFF, 26) << 2 };
    }
    if ((word & 0xFF000010) == 0x54000000) {
        return .{ .masked = word & ~@as(u32, 0x00FFFFE0), .rel = signExtend((word >> 5) & 0x7FFFF, 19) << 2 };
    }
    if ((word & 0x7E000000) == 0x34000000) {
        return .{ .masked = word & ~@as(u32, 0x00FFFFE0), .rel = signExtend((word >> 5) & 0x7FFFF, 19) << 2 };
    }
    if ((word & 0x7E000000) == 0x36000000) {
        return .{ .masked = word & ~@as(u32, 0x0007FFE0), .rel = signExtend((word >> 5) & 0x3FFF, 14) << 2 };
    }
    return null;
}

fn compareAarch64Function(
    allocator: Allocator,
    function_index: usize,
    enabled: AotImage,
    disabled: AotImage,
    enabled_code: []const u8,
    disabled_code: []const u8,
    all_poll_ranges: []const []const PollRange,
) !usize {
    if (enabled_code.len % 4 != 0 or disabled_code.len % 4 != 0) {
        return fail("function {d}: unaligned AArch64 code", .{function_index});
    }
    const sequence = &CANCEL_POLL_AARCH64;
    const function_polls = all_poll_ranges[function_index];
    for (function_polls) |pr| {
        if (pr.start % 4 != 0 or !std.mem.eql(u8, enabled_code[pr.start..pr.end], sequence)) {
            return fail("enabled function {d}: malformed cancel poll", .{function_index});
        }
    }
    var stripped: std.ArrayList(u8) = .empty;
    var cursor: usize = 0;
    for (function_polls) |pr| {
        try stripped.appendSlice(allocator, enabled_code[cursor..pr.start]);
        cursor = pr.end;
    }
    try stripped.appendSlice(allocator, enabled_code[cursor..]);
    if (stripped.items.len != disabled_code.len) {
        return fail("function {d}: normalized AArch64 size differs", .{function_index});
    }
    var layout_fields: usize = 0;
    var offset: usize = 0;
    while (offset < disabled_code.len) : (offset += 4) {
        const on_word = std.mem.readInt(u32, stripped.items[offset..][0..4], .little);
        const off_word = std.mem.readInt(u32, disabled_code[offset..][0..4], .little);
        if (on_word == off_word) continue;
        const on_rel = aarch64RelativeInstruction(on_word);
        const off_rel = aarch64RelativeInstruction(off_word);
        if (on_rel == null or off_rel == null or on_rel.?.masked != off_rel.?.masked) {
            return fail("function {d}: unrelated AArch64 word difference at normalized offset {d}", .{
                function_index,
                offset,
            });
        }
        var on_original_offset: usize = offset;
        for (function_polls) |pr| {
            if (pr.start <= on_original_offset) {
                on_original_offset += pr.end - pr.start;
            }
        }
        const on_target = @as(i64, enabled.function_offsets[function_index]) +
            @as(i64, @intCast(on_original_offset)) +
            on_rel.?.rel;
        const off_target = @as(i64, disabled.function_offsets[function_index]) +
            @as(i64, @intCast(offset)) +
            off_rel.?.rel;
        if ((try normalizeOnTextOffset(allocator, enabled, disabled, all_poll_ranges, on_target)) != off_target) {
            return fail("function {d}: AArch64 control-flow target differs", .{function_index});
        }
        layout_fields += 1;
    }
    return layout_fields;
}

pub fn compileCommandIdentity(
    allocator: Allocator,
    off_path: []const u8,
    on_path: []const u8,
    off_cmd: []const []const u8,
    on_cmd: []const []const u8,
) !ObjectMap {
    const flag = "--benchmark-disable-cancel-points";
    const pairs = [_]struct {
        cond: []const u8,
        path: []const u8,
        cmd: []const []const u8,
        expected_flags: usize,
    }{
        .{ .cond = "cancel-points-off", .path = off_path, .cmd = off_cmd, .expected_flags = 1 },
        .{ .cond = "cancel-points-on", .path = on_path, .cmd = on_cmd, .expected_flags = 0 },
    };

    var norm_off: std.ArrayList([]const u8) = .empty;
    var norm_on: std.ArrayList([]const u8) = .empty;

    for (pairs, 0..) |p, p_idx| {
        if (p.cmd.len == 0) return fail("missing compile command for {s}", .{p.cond});
        var flag_count: usize = 0;
        var o_count: usize = 0;
        var o_idx: usize = 0;
        for (p.cmd, 0..) |item, idx| {
            if (std.mem.eql(u8, item, flag)) flag_count += 1;
            if (std.mem.eql(u8, item, "-o")) {
                o_count += 1;
                o_idx = idx;
            }
        }
        if (flag_count != p.expected_flags or o_count != 1) {
            return fail("invalid compile command for {s}", .{p.cond});
        }
        if (o_idx + 1 >= p.cmd.len or !std.mem.eql(u8, p.cmd[o_idx + 1], p.path)) {
            return fail("compile output path mismatch for {s}", .{p.cond});
        }
        var target_list = if (p_idx == 0) &norm_off else &norm_on;
        var index: usize = 0;
        while (index < p.cmd.len) {
            const item = p.cmd[index];
            if (std.mem.eql(u8, item, flag)) {
                index += 1;
                continue;
            }
            if (std.mem.eql(u8, item, "-o")) {
                try target_list.append(allocator, "-o");
                try target_list.append(allocator, "<OUTPUT>");
                index += 2;
                continue;
            }
            try target_list.append(allocator, item);
            index += 1;
        }
    }

    if (norm_off.items.len != norm_on.items.len) {
        return fail("AOT compile commands differ beyond the cancel-point flag and output", .{});
    }
    for (norm_off.items, norm_on.items) |a, b| {
        if (!std.mem.eql(u8, a, b)) {
            return fail("AOT compile commands differ beyond the cancel-point flag and output", .{});
        }
    }

    var norm_arr = Array.init(allocator);
    for (norm_on.items) |item| try norm_arr.append(.{ .string = item });
    const norm_json = try benchmark_schema.canonicalJson(allocator, .{ .array = norm_arr });
    const norm_sha = try sha256Bytes(allocator, norm_json);

    var out: ObjectMap = .empty;
    try out.put(allocator, "differing_flag", .{ .string = flag });
    try out.put(allocator, "output_paths_condition_bound", .{ .bool = true });
    try out.put(allocator, "normalized_command", .{ .array = norm_arr });
    try out.put(allocator, "normalized_command_sha256", .{ .string = norm_sha });
    return out;
}

pub fn artifactIdentity(
    allocator: Allocator,
    io: std.Io,
    off_path: []const u8,
    on_path: []const u8,
    off_cmd: []const []const u8,
    on_cmd: []const []const u8,
    arch: []const u8,
    work_dir: []const u8,
) !ObjectMap {
    const sequence = cancelPollSequence(arch) orelse return fail("unsupported AOT architecture: {s}", .{arch});
    const entry_prefix_suffix = cancelPollEntryPrefixSuffix(arch).?;
    const command_comparison = try compileCommandIdentity(allocator, off_path, on_path, off_cmd, on_cmd);
    const disabled_image = try parseAot(allocator, io, off_path);
    const enabled_image = try parseAot(allocator, io, on_path);

    if (enabled_image.sections.len != disabled_image.sections.len) {
        return fail("AOT section ordering differs", .{});
    }
    for (enabled_image.sections, disabled_image.sections) |on_sec, off_sec| {
        if (on_sec.section_type != off_sec.section_type) {
            return fail("AOT section ordering differs", .{});
        }
        if (on_sec.section_type != AOT_TEXT_SECTION and on_sec.section_type != AOT_FUNCTION_SECTION) {
            if (!std.mem.eql(u8, on_sec.payload, off_sec.payload)) {
                return fail("unrelated AOT section {d} differs", .{on_sec.section_type});
            }
        }
    }
    if (enabled_image.function_offsets.len != disabled_image.function_offsets.len or
        !std.mem.eql(u32, enabled_image.function_type_indices, disabled_image.function_type_indices))
    {
        return fail("AOT function identity metadata differs", .{});
    }

    const enabled_prefix = if (std.mem.eql(u8, arch, "x86_64")) sequence[0..9] else sequence[0..8];
    const enabled_count = countNonOverlapping(enabled_image.text, sequence);
    const disabled_count = countNonOverlapping(disabled_image.text, sequence);
    if (enabled_count == 0 or
        disabled_count != 0 or
        countNonOverlapping(enabled_image.text, enabled_prefix) != enabled_count or
        countNonOverlapping(disabled_image.text, enabled_prefix) != 0)
    {
        return fail("complete cancel-poll sequences invalid: enabled={d}, disabled={d}", .{ enabled_count, disabled_count });
    }

    const fn_count = enabled_image.function_offsets.len;
    const all_poll_ranges = try allocator.alloc([]const PollRange, fn_count);
    var sum_ranges: usize = 0;
    for (0..fn_count) |idx| {
        const r = try pollRanges(allocator, try enabled_image.functionCode(idx), sequence);
        all_poll_ranges[idx] = r;
        sum_ranges += r.len;
    }
    if (sum_ranges != enabled_count) {
        return fail("cancel-poll sequence crosses a function boundary", .{});
    }

    var cumulative_removed: usize = 0;
    var layout_fields: usize = 0;
    var jump_table_entries: usize = 0;
    for (all_poll_ranges, 0..) |function_polls, function_index| {
        if (@as(usize, enabled_image.function_offsets[function_index]) !=
            @as(usize, disabled_image.function_offsets[function_index]) + cumulative_removed)
        {
            return fail("AOT function offsets have unrelated differences", .{});
        }
        const enabled_code = try enabled_image.functionCode(function_index);
        const disabled_code = try disabled_image.functionCode(function_index);
        const removed = function_polls.len * sequence.len;
        if (enabled_code.len != disabled_code.len + removed) {
            return fail("function {d}: size delta is not cancel polls", .{function_index});
        }
        if (std.mem.eql(u8, arch, "x86_64")) {
            const x86_res = try compareX86Function(
                allocator,
                io,
                function_index,
                enabled_image,
                disabled_image,
                enabled_code,
                disabled_code,
                all_poll_ranges,
                work_dir,
            );
            layout_fields += x86_res.layout_fields;
            jump_table_entries += x86_res.table_entries;
        } else {
            layout_fields += try compareAarch64Function(
                allocator,
                function_index,
                enabled_image,
                disabled_image,
                enabled_code,
                disabled_code,
                all_poll_ranges,
            );
        }
        cumulative_removed += removed;
    }

    const delta = enabled_image.text.len - disabled_image.text.len;
    if (delta != enabled_count * sequence.len) {
        return fail("AOT text delta {d} is not exactly {d} complete polls", .{ delta, enabled_count });
    }

    var leaf_export: ?AotExport = null;
    var leaf_export_count: usize = 0;
    for (enabled_image.exports) |item| {
        if (std.mem.eql(u8, item.name, "leaf_step") and item.kind == 0) {
            leaf_export = item;
            leaf_export_count += 1;
        }
    }
    if (leaf_export_count != 1) {
        return fail("enabled AOT does not have one leaf_step function export", .{});
    }
    const le = leaf_export.?;
    if (le.index < enabled_image.imported_function_count or
        enabled_image.imported_function_count != disabled_image.imported_function_count or
        enabled_image.exports.len != disabled_image.exports.len)
    {
        return fail("leaf_step AOT function mapping is invalid", .{});
    }
    for (enabled_image.exports, disabled_image.exports) |ea, eb| {
        if (!std.mem.eql(u8, ea.name, eb.name) or ea.kind != eb.kind or ea.index != eb.index) {
            return fail("leaf_step AOT function mapping is invalid", .{});
        }
    }
    const leaf_local: usize = le.index - enabled_image.imported_function_count;
    if (leaf_local >= fn_count) {
        return fail("leaf_step AOT function mapping is invalid", .{});
    }
    const leaf_ranges = all_poll_ranges[leaf_local];
    const leaf_disabled_code = try disabled_image.functionCode(leaf_local);
    const leaf_enabled_code = try enabled_image.functionCode(leaf_local);
    if (leaf_ranges.len != 1 or
        std.mem.indexOf(u8, leaf_disabled_code, sequence) != null or
        !std.mem.endsWith(u8, leaf_enabled_code[0..leaf_ranges[0].start], entry_prefix_suffix))
    {
        return fail("leaf_step does not have exactly one enabled entry poll after the architecture ABI prefix and none disabled", .{});
    }

    var off_cmd_arr = Array.init(allocator);
    for (off_cmd) |s| try off_cmd_arr.append(.{ .string = s });
    var on_cmd_arr = Array.init(allocator);
    for (on_cmd) |s| try on_cmd_arr.append(.{ .string = s });
    var compile_cmds_obj: ObjectMap = .empty;
    try compile_cmds_obj.put(allocator, "cancel-points-off", .{ .array = off_cmd_arr });
    try compile_cmds_obj.put(allocator, "cancel-points-on", .{ .array = on_cmd_arr });

    var norm_comp: ObjectMap = .empty;
    try norm_comp.put(allocator, "method", .{ .string = "complete-poll-and-layout-normalization-v1" });
    try norm_comp.put(allocator, "other_sections_byte_identical", .{ .bool = true });
    try norm_comp.put(allocator, "function_count", .{ .integer = @intCast(fn_count) });
    try norm_comp.put(allocator, "function_type_indices_identical", .{ .bool = true });
    try norm_comp.put(allocator, "normalized_functions_identical", .{ .integer = @intCast(fn_count) });
    try norm_comp.put(allocator, "poll_sequences_removed", .{ .integer = @intCast(enabled_count) });
    try norm_comp.put(allocator, "relative_control_flow_fields_normalized", .{ .integer = @intCast(layout_fields) });
    try norm_comp.put(allocator, "jump_table_entries_normalized", .{ .integer = @intCast(jump_table_entries) });

    var prefix_hex_buf: [64]u8 = undefined;
    var prefix_hex_len: usize = 0;
    for (entry_prefix_suffix) |b| {
        const h = std.fmt.bytesToHex([1]u8{b}, .lower);
        prefix_hex_buf[prefix_hex_len] = h[0];
        prefix_hex_buf[prefix_hex_len + 1] = h[1];
        prefix_hex_len += 2;
    }

    var seq_hex_buf: [64]u8 = undefined;
    var seq_hex_len: usize = 0;
    for (sequence) |b| {
        const h = std.fmt.bytesToHex([1]u8{b}, .lower);
        seq_hex_buf[seq_hex_len] = h[0];
        seq_hex_buf[seq_hex_len + 1] = h[1];
        seq_hex_len += 2;
    }

    var leaf_step_obj: ObjectMap = .empty;
    try leaf_step_obj.put(allocator, "export_function_index", .{ .integer = le.index });
    try leaf_step_obj.put(allocator, "imported_function_count", .{ .integer = enabled_image.imported_function_count });
    try leaf_step_obj.put(allocator, "local_function_index", .{ .integer = @intCast(leaf_local) });
    try leaf_step_obj.put(allocator, "enabled_function_offset", .{ .integer = enabled_image.function_offsets[leaf_local] });
    try leaf_step_obj.put(allocator, "disabled_function_offset", .{ .integer = disabled_image.function_offsets[leaf_local] });
    try leaf_step_obj.put(allocator, "enabled_code_sha256", .{ .string = try sha256Bytes(allocator, leaf_enabled_code) });
    try leaf_step_obj.put(allocator, "disabled_code_sha256", .{ .string = try sha256Bytes(allocator, leaf_disabled_code) });
    try leaf_step_obj.put(allocator, "enabled_entry_poll_offset", .{ .integer = @intCast(leaf_ranges[0].start) });
    try leaf_step_obj.put(allocator, "entry_prefix_suffix_hex", .{ .string = try allocator.dupe(u8, prefix_hex_buf[0..prefix_hex_len]) });
    try leaf_step_obj.put(allocator, "enabled_poll_sequences", .{ .integer = 1 });
    try leaf_step_obj.put(allocator, "disabled_poll_sequences", .{ .integer = 0 });
    try leaf_step_obj.put(allocator, "normalized_body_identical", .{ .bool = true });

    var off_cond: ObjectMap = .empty;
    try off_cond.put(allocator, "path", .{ .string = off_path });
    try off_cond.put(allocator, "sha256", .{ .string = try sha256Bytes(allocator, disabled_image.data) });
    try off_cond.put(allocator, "file_bytes", .{ .integer = @intCast(disabled_image.data.len) });
    try off_cond.put(allocator, "text_bytes", .{ .integer = @intCast(disabled_image.text.len) });

    var on_cond: ObjectMap = .empty;
    try on_cond.put(allocator, "path", .{ .string = on_path });
    try on_cond.put(allocator, "sha256", .{ .string = try sha256Bytes(allocator, enabled_image.data) });
    try on_cond.put(allocator, "file_bytes", .{ .integer = @intCast(enabled_image.data.len) });
    try on_cond.put(allocator, "text_bytes", .{ .integer = @intCast(enabled_image.text.len) });

    var conditions_obj: ObjectMap = .empty;
    try conditions_obj.put(allocator, "cancel-points-off", .{ .object = off_cond });
    try conditions_obj.put(allocator, "cancel-points-on", .{ .object = on_cond });

    var out: ObjectMap = .empty;
    try out.put(allocator, "architecture", .{ .string = arch });
    try out.put(allocator, "cancel_poll_sequence_hex", .{ .string = try allocator.dupe(u8, seq_hex_buf[0..seq_hex_len]) });
    try out.put(allocator, "cancel_poll_sites_enabled", .{ .integer = @intCast(enabled_count) });
    try out.put(allocator, "cancel_poll_sites_disabled", .{ .integer = @intCast(disabled_count) });
    try out.put(allocator, "text_delta_bytes", .{ .integer = @intCast(delta) });
    try out.put(allocator, "bytes_per_poll_site", .{ .integer = @intCast(sequence.len) });
    try out.put(allocator, "compile_commands", .{ .object = compile_cmds_obj });
    try out.put(allocator, "compile_command_comparison", .{ .object = command_comparison });
    try out.put(allocator, "normalized_comparison", .{ .object = norm_comp });
    try out.put(allocator, "leaf_step", .{ .object = leaf_step_obj });
    try out.put(allocator, "conditions", .{ .object = conditions_obj });
    return out;
}

pub fn selectedCpuAffinity(allocator: Allocator) ![]i64 {
    if (builtin.os.tag == .linux) {
        var mask: [128]u8 = @splat(0);
        const rc = std.os.linux.syscall3(.sched_getaffinity, 0, mask.len, @intFromPtr(&mask));
        const signed_rc: isize = @bitCast(rc);
        if (signed_rc > 0) {
            const bytes: usize = @intCast(signed_rc);
            for (mask[0..@min(bytes, mask.len)], 0..) |b, byte_idx| {
                if (b != 0) {
                    for (0..8) |bit| {
                        if ((b & (@as(u8, 1) << @intCast(bit))) != 0) {
                            const out = try allocator.alloc(i64, 1);
                            out[0] = @intCast(byte_idx * 8 + bit);
                            return out;
                        }
                    }
                }
            }
        }
    }
    return &.{};
}

pub const MeasureOptions = struct {
    repo: []const u8,
    runner: []const []const u8,
    runtime: Build,
    artifact: []const u8,
    condition: []const u8,
    calls: i64,
    timeout: f64,
    minimum_interval_ns: i64,
    enforce_quality: bool,
    phase: []const u8,
    pair_index: i64,
    position: i64,
    leaf_entry_poll_proven: bool,
};

pub fn measure(
    allocator: Allocator,
    io: std.Io,
    opts: MeasureOptions,
) !ObjectMap {
    if (!opts.leaf_entry_poll_proven) {
        return fail("leaf-entry poll opportunities lack artifact proof", .{});
    }
    const affinity = try selectedCpuAffinity(allocator);
    var cmd_list: std.ArrayList([]const u8) = .empty;
    if (affinity.len > 0) {
        try cmd_list.append(allocator, "taskset");
        try cmd_list.append(allocator, "--cpu-list");
        try cmd_list.append(allocator, try std.fmt.allocPrint(allocator, "{d}", .{affinity[0]}));
    }
    for (opts.runner) |r| try cmd_list.append(allocator, r);
    try cmd_list.append(allocator, opts.runtime.wamr);
    try cmd_list.append(allocator, "run");
    try cmd_list.append(allocator, opts.artifact);
    try cmd_list.append(allocator, try std.fmt.allocPrint(allocator, "{d}", .{opts.calls}));

    const start_ts = std.Io.Clock.awake.now(io);
    const proc = try runProcess(allocator, io, cmd_list.items, opts.repo, null);
    const end_ts = std.Io.Clock.awake.now(io);
    const host_elapsed: i64 = @intCast(@max(0, start_ts.durationTo(end_ts).nanoseconds));

    if (proc.code != 0) {
        return fail("{s} guest exited {d}: {s}", .{ opts.condition, proc.code, proc.stderr });
    }
    const guest = try parseGuestResult(
        allocator,
        proc.stdout,
        opts.calls,
        opts.minimum_interval_ns,
        opts.enforce_quality,
    );
    const elapsed = guest.get("elapsed_ns").?.integer;
    var cmd_arr = Array.init(allocator);
    for (cmd_list.items) |arg| try cmd_arr.append(.{ .string = arg });

    var out: ObjectMap = .empty;
    try out.put(allocator, "phase", .{ .string = opts.phase });
    try out.put(allocator, "pair_index", .{ .integer = opts.pair_index });
    try out.put(allocator, "position", .{ .integer = opts.position });
    try out.put(allocator, "condition", .{ .string = opts.condition });
    try out.put(allocator, "command", .{ .array = cmd_arr });
    try out.put(allocator, "leaf_calls", .{ .integer = opts.calls });
    try out.put(allocator, "leaf_entry_poll_opportunities", .{
        .integer = if (std.mem.eql(u8, opts.condition, "cancel-points-on")) opts.calls else 0,
    });
    try out.put(allocator, "batches", .{ .integer = @divExact(opts.calls, LEAF_UNROLL) });
    try out.put(allocator, "guest_elapsed_ns", .{ .integer = elapsed });
    try out.put(allocator, "raw_guest_elapsed_ns", guest.get("raw_elapsed_ns").?);
    try out.put(allocator, "timing_overhead_ns", guest.get("timing_overhead_ns").?);
    try out.put(allocator, "timing_overhead_ppm", guest.get("timing_overhead_ppm").?);
    try out.put(allocator, "host_wall_elapsed_ns", .{ .integer = host_elapsed });
    try out.put(allocator, "leaf_calls_per_second", .{
        .float = @as(f64, @floatFromInt(opts.calls)) / (@as(f64, @floatFromInt(elapsed)) / 1e9),
    });
    try out.put(allocator, "guest", .{ .object = guest });
    try out.put(allocator, "correct", .{ .bool = true });
    return out;
}

pub fn conditionOrder(pair_index: usize) [2][]const u8 {
    return if (pair_index % 2 == 0)
        .{ "cancel-points-off", "cancel-points-on" }
    else
        .{ "cancel-points-on", "cancel-points-off" };
}

fn valueAsF64(val: Value) f64 {
    return switch (val) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |s| std.fmt.parseFloat(f64, s) catch 0.0,
        else => 0.0,
    };
}

fn buildSampleStatsObject(allocator: Allocator, values: []const Value, is_int: bool) !ObjectMap {
    const floats = try allocator.alloc(f64, values.len);
    for (values, 0..) |v, i| floats[i] = valueAsF64(v);
    const stats = try benchmark_stats.summarize(allocator, floats);

    var samples_arr = Array.init(allocator);
    for (values) |v| try samples_arr.append(v);

    var obj: ObjectMap = .empty;
    try obj.put(allocator, "samples", .{ .array = samples_arr });
    try obj.put(allocator, "runs", .{ .integer = @intCast(stats.runs) });
    try obj.put(allocator, "mean", .{ .float = stats.mean });
    if (is_int and values.len % 2 == 1) {
        try obj.put(allocator, "median", .{ .integer = @intFromFloat(stats.median) });
    } else {
        try obj.put(allocator, "median", .{ .float = stats.median });
    }
    if (is_int) {
        try obj.put(allocator, "min", .{ .integer = @intFromFloat(stats.min) });
        try obj.put(allocator, "max", .{ .integer = @intFromFloat(stats.max) });
        try obj.put(allocator, "range", .{ .integer = @intFromFloat(stats.range) });
    } else {
        try obj.put(allocator, "min", .{ .float = stats.min });
        try obj.put(allocator, "max", .{ .float = stats.max });
        try obj.put(allocator, "range", .{ .float = stats.range });
    }
    return obj;
}

pub fn summarize(
    allocator: Allocator,
    records: []const ObjectMap,
    calls: i64,
) !struct { summaries: Array, comparison: ObjectMap } {
    var summaries = Array.init(allocator);
    for ([_][]const u8{ "cancel-points-off", "cancel-points-on" }) |cond| {
        var guest_elapsed: std.ArrayList(Value) = .empty;
        var calls_per_sec: std.ArrayList(Value) = .empty;
        var host_wall: std.ArrayList(Value) = .empty;
        for (records) |item| {
            if (std.mem.eql(u8, item.get("condition").?.string, cond)) {
                try guest_elapsed.append(allocator, item.get("guest_elapsed_ns").?);
                try calls_per_sec.append(allocator, item.get("leaf_calls_per_second").?);
                try host_wall.append(allocator, item.get("host_wall_elapsed_ns").?);
            }
        }
        var sum_obj: ObjectMap = .empty;
        try sum_obj.put(allocator, "condition", .{ .string = cond });
        try sum_obj.put(allocator, "guest_elapsed_ns", .{ .object = try buildSampleStatsObject(allocator, guest_elapsed.items, true) });
        try sum_obj.put(allocator, "leaf_calls_per_second", .{ .object = try buildSampleStatsObject(allocator, calls_per_sec.items, false) });
        try sum_obj.put(allocator, "host_wall_elapsed_ns", .{ .object = try buildSampleStatsObject(allocator, host_wall.items, true) });
        try summaries.append(.{ .object = sum_obj });
    }

    var pair_indices: std.ArrayList(i64) = .empty;
    for (records) |item| {
        const pi = item.get("pair_index").?.integer;
        var seen = false;
        for (pair_indices.items) |existing| {
            if (existing == pi) seen = true;
        }
        if (!seen) try pair_indices.append(allocator, pi);
    }
    std.mem.sort(i64, pair_indices.items, {}, std.sort.asc(i64));

    var pair_rows = Array.init(allocator);
    var elapsed_ratios: std.ArrayList(f64) = .empty;
    var ns_per_call_list: std.ArrayList(f64) = .empty;
    var throughput_ratios: std.ArrayList(f64) = .empty;

    for (pair_indices.items) |pi| {
        var off_ns: i64 = 0;
        var on_ns: i64 = 0;
        for (records) |item| {
            if (item.get("pair_index").?.integer == pi) {
                const c = item.get("condition").?.string;
                if (std.mem.eql(u8, c, "cancel-points-off")) off_ns = item.get("guest_elapsed_ns").?.integer;
                if (std.mem.eql(u8, c, "cancel-points-on")) on_ns = item.get("guest_elapsed_ns").?.integer;
            }
        }
        const off_f: f64 = @floatFromInt(off_ns);
        const on_f: f64 = @floatFromInt(on_ns);
        const calls_f: f64 = @floatFromInt(calls);
        const elapsed_ratio = on_f / off_f;
        const delta_ns = on_ns - off_ns;
        const ns_per_call = @as(f64, @floatFromInt(delta_ns)) / calls_f;
        const throughput_ratio = off_f / on_f;

        try elapsed_ratios.append(allocator, elapsed_ratio);
        try ns_per_call_list.append(allocator, ns_per_call);
        try throughput_ratios.append(allocator, throughput_ratio);

        var row: ObjectMap = .empty;
        try row.put(allocator, "pair_index", .{ .integer = pi });
        try row.put(allocator, "on_over_off_elapsed_ratio", .{ .float = elapsed_ratio });
        try row.put(allocator, "on_minus_off_ns", .{ .integer = delta_ns });
        try row.put(allocator, "on_minus_off_ns_per_leaf_call", .{ .float = ns_per_call });
        try row.put(allocator, "on_over_off_throughput_ratio", .{ .float = throughput_ratio });
        try pair_rows.append(.{ .object = row });
    }

    const med_elapsed_ratio = (try benchmark_stats.summarize(allocator, elapsed_ratios.items)).median;
    const med_ns_per_call = (try benchmark_stats.summarize(allocator, ns_per_call_list.items)).median;
    const med_throughput_ratio = (try benchmark_stats.summarize(allocator, throughput_ratios.items)).median;

    var comparison: ObjectMap = .empty;
    try comparison.put(allocator, "pairs", .{ .array = pair_rows });
    try comparison.put(allocator, "median_on_over_off_elapsed_ratio", .{ .float = med_elapsed_ratio });
    try comparison.put(allocator, "median_elapsed_delta_percent", .{ .float = (med_elapsed_ratio - 1.0) * 100.0 });
    try comparison.put(allocator, "median_on_minus_off_ns_per_leaf_call", .{ .float = med_ns_per_call });
    try comparison.put(allocator, "median_on_over_off_throughput_ratio", .{ .float = med_throughput_ratio });

    return .{ .summaries = summaries, .comparison = comparison };
}

pub fn validateInvocations(
    invocations: []const ObjectMap,
    phase: []const u8,
    pairs: usize,
    calls: i64,
) !void {
    if (invocations.len != pairs * 2) {
        return fail("{s} invocation count mismatch", .{phase});
    }
    const exp = try expectedResult(calls);
    for (0..pairs) |pair_index| {
        const order = conditionOrder(pair_index);
        for (order, 0..) |cond, pos| {
            const item = invocations[pair_index * 2 + pos];
            const item_phase = item.get("phase") orelse return fail("{s} invocation ordering mismatch", .{phase});
            const item_pi = item.get("pair_index") orelse return fail("{s} invocation ordering mismatch", .{phase});
            const item_pos = item.get("position") orelse return fail("{s} invocation ordering mismatch", .{phase});
            const item_cond = item.get("condition") orelse return fail("{s} invocation ordering mismatch", .{phase});
            if (item_phase != .string or !std.mem.eql(u8, item_phase.string, phase) or
                item_pi != .integer or item_pi.integer != @as(i64, @intCast(pair_index)) or
                item_pos != .integer or item_pos.integer != @as(i64, @intCast(pos)) or
                item_cond != .string or !std.mem.eql(u8, item_cond.string, cond))
            {
                return fail("{s} invocation ordering mismatch", .{phase});
            }
            const lc = item.get("leaf_calls") orelse return fail("{s} invocation work mismatch", .{phase});
            const bt = item.get("batches") orelse return fail("{s} invocation work mismatch", .{phase});
            if (lc != .integer or lc.integer != calls or bt != .integer or bt.integer != @divExact(calls, LEAF_UNROLL)) {
                return fail("{s} invocation work mismatch", .{phase});
            }
            const expected_opp: i64 = if (std.mem.eql(u8, cond, "cancel-points-on")) calls else 0;
            const opp = item.get("leaf_entry_poll_opportunities") orelse return fail("{s} poll opportunity mismatch", .{phase});
            if (opp != .integer or opp.integer != expected_opp) {
                return fail("{s} poll opportunity mismatch", .{phase});
            }
            const guest_val = item.get("guest") orelse return fail("{s} guest work mismatch", .{phase});
            if (guest_val != .object) return fail("{s} guest work mismatch", .{phase});
            const g = guest_val.object;
            if ((try parseStrictJsonI64(g.get("leaf_calls") orelse .null, "leaf_calls")) != exp.leaf_calls or
                (try parseStrictJsonI64(g.get("leaf_unroll") orelse .null, "leaf_unroll")) != exp.leaf_unroll or
                (try parseStrictJsonI64(g.get("batches") orelse .null, "batches")) != exp.batches or
                (try parseStrictJsonU64(g.get("checksum") orelse .null, "checksum")) != exp.checksum)
            {
                return fail("{s} guest work mismatch", .{phase});
            }
            const correct_val = item.get("correct") orelse return fail("{s} correctness marker mismatch", .{phase});
            if (correct_val != .bool or !correct_val.bool) {
                return fail("{s} correctness marker mismatch", .{phase});
            }
        }
    }
}

fn sourceIdentity(allocator: Allocator, io: std.Io, repo: []const u8) !ObjectMap {
    const commit_proc = try runProcess(allocator, io, &.{ "git", "rev-parse", "HEAD" }, repo, null);
    if (commit_proc.code != 0) return fail("git rev-parse HEAD failed", .{});
    const commit = std.mem.trim(u8, commit_proc.stdout, " \t\r\n");

    const diff_proc = try runProcess(allocator, io, &.{ "git", "diff", "--binary", "HEAD" }, repo, null);
    if (diff_proc.code != 0) return fail("git diff HEAD failed", .{});
    const tracked_diff_sha = try sha256Bytes(allocator, diff_proc.stdout);

    var files_obj: ObjectMap = .empty;
    for (SOURCE_PATHS) |rel| {
        const full = try std.fs.path.join(allocator, &.{ repo, rel });
        const stat = std.Io.Dir.cwd().statFile(io, full, .{}) catch {
            return fail("missing harness source: {s}", .{rel});
        };
        var f_entry: ObjectMap = .empty;
        try f_entry.put(allocator, "sha256", .{ .string = try sha256File(allocator, io, full) });
        try f_entry.put(allocator, "bytes", .{ .integer = @intCast(stat.size) });
        try files_obj.put(allocator, rel, .{ .object = f_entry });
    }

    const ls_proc = try runProcess(allocator, io, &.{ "git", "ls-files", "-z", "--", "build.zig", "build.zig.zon", "src" }, repo, null);
    if (ls_proc.code != 0) return fail("git ls-files failed", .{});
    var build_source: std.ArrayList(u8) = .empty;
    var path_it = std.mem.splitScalar(u8, ls_proc.stdout, 0);
    while (path_it.next()) |raw| {
        if (raw.len == 0) continue;
        try build_source.appendSlice(allocator, raw);
        try build_source.append(allocator, 0);
        const full = try std.fs.path.join(allocator, &.{ repo, raw });
        const content = std.Io.Dir.cwd().readFileAlloc(io, full, allocator, .limited(64 * 1024 * 1024)) catch {
            return fail("cannot read build source file: {s}", .{raw});
        };
        defer allocator.free(content);
        try build_source.appendSlice(allocator, content);
        try build_source.append(allocator, 0);
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "commit", .{ .string = try allocator.dupe(u8, commit) });
    try out.put(allocator, "tracked_diff_sha256", .{ .string = tracked_diff_sha });
    try out.put(allocator, "build_source_sha256", .{ .string = try sha256Bytes(allocator, build_source.items) });
    try out.put(allocator, "harness_files", .{ .object = files_obj });
    return out;
}

fn fixtureIdentity(allocator: Allocator, io: std.Io, repo: []const u8) !ObjectMap {
    const fixture_path = try std.fs.path.join(allocator, &.{ repo, FIXTURE });
    const stat = std.Io.Dir.cwd().statFile(io, fixture_path, .{}) catch {
        return fail("missing fixture: {s}", .{fixture_path});
    };
    const manifest_path = try std.fs.path.join(allocator, &.{ repo, "tests/benchmarks/leaf-cancel-cost/fixtures.sha256" });
    const manifest_text = std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(64 * 1024)) catch {
        return fail("missing fixture checksum manifest", .{});
    };
    var tok_it = std.mem.tokenizeAny(u8, manifest_text, " \t\r\n");
    const expected_sha = tok_it.next() orelse return fail("invalid fixture checksum manifest", .{});
    const expected_name = tok_it.next() orelse return fail("invalid fixture checksum manifest", .{});
    if (tok_it.next() != null or !std.mem.eql(u8, expected_name, "leaf_calls.wasm")) {
        return fail("invalid fixture checksum manifest", .{});
    }
    const actual_sha = try sha256File(allocator, io, fixture_path);
    if (!std.mem.eql(u8, actual_sha, expected_sha)) {
        return fail("fixture checksum mismatch: expected {s}, got {s}", .{ expected_sha, actual_sha });
    }

    var toolchain: ObjectMap = .empty;
    try toolchain.put(allocator, "wasi_sdk_version", .{ .string = "25.0" });
    try toolchain.put(allocator, "clang", .{ .string = "clang version 19.1.5-wasi-sdk" });
    try toolchain.put(allocator, "archive_sha256", .{ .string = "52640dde13599bf127a95499e61d6d640256119456d1af8897ab6725bcf3d89c" });

    var out: ObjectMap = .empty;
    try out.put(allocator, "path", .{ .string = FIXTURE });
    try out.put(allocator, "sha256", .{ .string = actual_sha });
    try out.put(allocator, "bytes", .{ .integer = @intCast(stat.size) });
    try out.put(allocator, "toolchain", .{ .object = toolchain });
    return out;
}

fn controlledEnv(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    root: []const u8,
) !std.process.Environ.Map {
    var env = try base_env.clone(allocator);
    const dirs = [_]struct { key: []const u8, sub: []const u8 }{
        .{ .key = "ZIG_GLOBAL_CACHE_DIR", .sub = "zig-global-cache" },
        .{ .key = "ZIG_LOCAL_CACHE_DIR", .sub = "zig-local-cache" },
        .{ .key = "XDG_CACHE_HOME", .sub = "xdg-cache" },
        .{ .key = "TMPDIR", .sub = "tmp" },
    };
    for (dirs) |d| {
        const p = try std.fs.path.join(allocator, &.{ root, d.sub });
        try std.Io.Dir.cwd().createDirPath(io, p);
        try env.put(d.key, p);
    }
    try env.put("LANG", "C");
    try env.put("LC_ALL", "C");
    try env.put("TZ", "UTC");
    return env;
}

fn buildOne(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    repo: []const u8,
    output: []const u8,
    source: ObjectMap,
    name: []const u8,
    optimize: []const u8,
    target: ?[]const u8,
    compiler: bool,
    rebuild: bool,
) !Build {
    var parts: ObjectMap = .empty;
    try parts.put(allocator, "build_source_sha256", source.get("build_source_sha256").?);
    try parts.put(allocator, "optimize", .{ .string = optimize });
    try parts.put(allocator, "target", .{ .string = target orelse "native" });
    try parts.put(allocator, "compiler", .{ .bool = compiler });
    try parts.put(allocator, "benchmark_cancel_point_toggle", .{ .bool = true });
    try parts.put(allocator, "zig", .{ .string = try commandIdentity(allocator, io, &.{ "zig", "version" }) });

    const key = try cacheKey(allocator, .{ .object = parts });
    const build_sub = try std.fmt.allocPrint(allocator, "builds/{s}-{s}", .{ name, key[0..16] });
    const prefix = try std.fs.path.join(allocator, &.{ output, build_sub });
    const wamr = try std.fs.path.join(allocator, &.{ prefix, "bin/wamr" });
    const wamrc: ?[]const u8 = if (compiler) try std.fs.path.join(allocator, &.{ prefix, "bin/wamrc" }) else null;
    const marker = try std.fs.path.join(allocator, &.{ prefix, "leaf-cancel-build.json" });

    var cmd_list: std.ArrayList([]const u8) = .empty;
    try cmd_list.appendSlice(allocator, &.{ "zig", "build" });
    if (target) |t| {
        try cmd_list.append(allocator, try std.fmt.allocPrint(allocator, "-Dtarget={s}", .{t}));
    }
    try cmd_list.appendSlice(allocator, &.{
        try std.fmt.allocPrint(allocator, "-Doptimize={s}", .{optimize}),
        "-Dlib_wasi_threads=true",
        "-Dinterp=false",
        "-Daot=true",
        "-Dbenchmark-cancel-point-toggle=true",
        "--prefix",
        prefix,
    });
    const command = try cmd_list.toOwnedSlice(allocator);

    const wamr_exists = (std.Io.Dir.cwd().statFile(io, wamr, .{}) catch null) != null;
    const wamrc_exists = if (wamrc) |w| (std.Io.Dir.cwd().statFile(io, w, .{}) catch null) != null else true;
    if (!rebuild and wamr_exists and wamrc_exists) {
        if (std.Io.Dir.cwd().readFileAlloc(io, marker, allocator, .limited(64 * 1024))) |marker_bytes| {
            if (std.json.parseFromSlice(Value, allocator, marker_bytes, .{})) |parsed| {
                if (parsed.value == .object) {
                    if (parsed.value.object.get("cache_key")) |ck| {
                        if (ck == .string and std.mem.eql(u8, ck.string, key)) {
                            return .{
                                .name = name,
                                .prefix = prefix,
                                .wamr = wamr,
                                .wamrc = wamrc,
                                .command = command,
                                .cache_key = key,
                            };
                        }
                    }
                }
            } else |_| {}
        } else |_| {}
    }

    try std.Io.Dir.cwd().createDirPath(io, prefix);
    const cache_root = try std.fs.path.join(allocator, &.{ output, "cache", name });
    var env = try controlledEnv(allocator, io, base_env, cache_root);
    const res = try runProcess(allocator, io, command, repo, &env);
    if (res.code != 0) {
        return fail("{s} build failed with exit {d}: {s}", .{ name, res.code, res.stderr });
    }
    if ((std.Io.Dir.cwd().statFile(io, wamr, .{}) catch null) == null or
        (if (wamrc) |w| (std.Io.Dir.cwd().statFile(io, w, .{}) catch null) == null else false))
    {
        return fail("{s} build did not produce expected binaries", .{name});
    }

    var cmd_arr = Array.init(allocator);
    for (command) |s| try cmd_arr.append(.{ .string = s });
    var marker_obj: ObjectMap = .empty;
    try marker_obj.put(allocator, "cache_key", .{ .string = key });
    try marker_obj.put(allocator, "parts", .{ .object = parts });
    try marker_obj.put(allocator, "command", .{ .array = cmd_arr });
    try atomicWriteJson(allocator, io, marker, .{ .object = marker_obj });

    return .{
        .name = name,
        .prefix = prefix,
        .wamr = wamr,
        .wamrc = wamrc,
        .command = command,
        .cache_key = key,
    };
}

const BuiltTools = struct {
    compiler: Build,
    runtime: Build,
};

fn buildTools(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    repo: []const u8,
    output: []const u8,
    source: ObjectMap,
    optimize: []const u8,
    target: ?[]const u8,
    rebuild: bool,
) !BuiltTools {
    const compiler = try buildOne(
        allocator,
        io,
        base_env,
        repo,
        output,
        source,
        "host-compiler",
        optimize,
        null,
        true,
        rebuild,
    );
    if (target == null) {
        return .{ .compiler = compiler, .runtime = compiler };
    }
    const runtime = try buildOne(
        allocator,
        io,
        base_env,
        repo,
        output,
        source,
        "target-runtime",
        optimize,
        target,
        false,
        rebuild,
    );
    return .{ .compiler = compiler, .runtime = runtime };
}

const CompiledArtifacts = struct {
    off_path: []const u8,
    on_path: []const u8,
    off_cmd: []const []const u8,
    on_cmd: []const []const u8,
};

fn compileArtifacts(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    repo: []const u8,
    output: []const u8,
    compiler: Build,
    arch: []const u8,
) !CompiledArtifacts {
    const wamrc = compiler.wamrc orelse return fail("host compiler is missing wamrc", .{});
    const aot_dir = try std.fs.path.join(allocator, &.{ output, "aot" });
    try std.Io.Dir.cwd().createDirPath(io, aot_dir);
    const off_path = try std.fs.path.join(allocator, &.{ aot_dir, "leaf-calls-polls-off.cwasm" });
    const on_path = try std.fs.path.join(allocator, &.{ aot_dir, "leaf-calls-polls-on.cwasm" });
    const fixture_path = try std.fs.path.join(allocator, &.{ repo, FIXTURE });

    const off_cmd = try allocator.dupe([]const u8, &.{
        wamrc,
        "compile",
        "--target",
        arch,
        "--benchmark-disable-cancel-points",
        fixture_path,
        "-o",
        off_path,
    });
    const on_cmd = try allocator.dupe([]const u8, &.{
        wamrc,
        "compile",
        "--target",
        arch,
        fixture_path,
        "-o",
        on_path,
    });

    const cache_root = try std.fs.path.join(allocator, &.{ output, "cache/compile-aot" });
    var env = try controlledEnv(allocator, io, base_env, cache_root);

    const off_res = try runProcess(allocator, io, off_cmd, repo, &env);
    if (off_res.code != 0) return fail("AOT compilation failed for cancel-points-off: {s}", .{off_res.stderr});
    const on_res = try runProcess(allocator, io, on_cmd, repo, &env);
    if (on_res.code != 0) return fail("AOT compilation failed for cancel-points-on: {s}", .{on_res.stderr});

    return .{
        .off_path = off_path,
        .on_path = on_path,
        .off_cmd = off_cmd,
        .on_cmd = on_cmd,
    };
}

fn collectedAt(allocator: Allocator, io: std.Io) ![]const u8 {
    const now_ns: i128 = @intCast(std.Io.Clock.real.now(io).nanoseconds);
    const secs: u64 = @intCast(@max(@divFloor(now_ns, std.time.ns_per_s), 0));
    const epoch_secs = std.time.epoch.EpochSeconds{ .secs = secs };
    const epoch_day = epoch_secs.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = epoch_secs.getDaySeconds();
    return std.fmt.allocPrint(
        allocator,
        "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}+00:00",
        .{
            year_day.year,
            month_day.month.numeric(),
            month_day.day_index + 1,
            day_secs.getHoursIntoDay(),
            day_secs.getMinutesIntoHour(),
            day_secs.getSecondsIntoMinute(),
        },
    );
}

fn cpuModel(allocator: Allocator, io: std.Io, fallback: []const u8) ![]const u8 {
    const res = runProcess(allocator, io, &.{"lscpu"}, null, null) catch return fallback;
    if (res.code == 0) {
        var line_it = std.mem.splitScalar(u8, res.stdout, '\n');
        while (line_it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (std.ascii.startsWithIgnoreCase(trimmed, "model name:")) {
                if (std.mem.indexOfScalar(u8, trimmed, ':')) |colon| {
                    const val = std.mem.trim(u8, trimmed[colon + 1 ..], " \t\r");
                    if (val.len > 0) return try allocator.dupe(u8, val);
                }
            }
        }
    }
    return fallback;
}

fn hostMetadata(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    runner_environment_opt: ?[]const u8,
) !ObjectMap {
    const env_val = if (runner_environment_opt) |r|
        r
    else
        (environ_map.get("RUNNER_ENVIRONMENT") orelse "");
    const environment = if (env_val.len > 0)
        env_val
    else if (std.ascii.eqlIgnoreCase(environ_map.get("GITHUB_ACTIONS") orelse "", "true"))
        "github-actions-unspecified"
    else
        "local";

    const system = try commandIdentity(allocator, io, &.{ "uname", "-s" });
    const release = try commandIdentity(allocator, io, &.{ "uname", "-r" });
    const version = try commandIdentity(allocator, io, &.{ "uname", "-v" });
    const machine = try commandIdentity(allocator, io, &.{ "uname", "-m" });
    const processor = try commandIdentity(allocator, io, &.{ "uname", "-p" });
    const cpu = try cpuModel(allocator, io, machine);
    const logical_cpus: i64 = @intCast(std.Thread.getCpuCount() catch 1);

    const runner_name = environ_map.get("RUNNER_NAME") orelse "";
    const runner_image = environ_map.get("ImageOS") orelse "";
    const runner_os = environ_map.get("RUNNER_OS") orelse "";
    const runner_arch = environ_map.get("RUNNER_ARCH") orelse "";
    const github_run_id = environ_map.get("GITHUB_RUN_ID") orelse "";
    const github_run_attempt = environ_map.get("GITHUB_RUN_ATTEMPT") orelse "";
    const github_workflow = environ_map.get("GITHUB_WORKFLOW") orelse "";

    var fp_fields: ObjectMap = .empty;
    try fp_fields.put(allocator, "system", .{ .string = system });
    try fp_fields.put(allocator, "machine", .{ .string = machine });
    try fp_fields.put(allocator, "cpu", .{ .string = cpu });
    try fp_fields.put(allocator, "logical_cpus", .{ .integer = logical_cpus });
    try fp_fields.put(allocator, "runner_environment", .{ .string = environment });
    try fp_fields.put(allocator, "runner_image", .{ .string = runner_image });
    try fp_fields.put(allocator, "runner_os", .{ .string = runner_os });
    try fp_fields.put(allocator, "runner_arch", .{ .string = runner_arch });

    var fp_obj: ObjectMap = .empty;
    try fp_obj.put(allocator, "sha256", .{ .string = try cacheKey(allocator, .{ .object = fp_fields }) });
    try fp_obj.put(allocator, "fields", .{ .object = fp_fields });

    var meta: ObjectMap = .empty;
    try meta.put(allocator, "system", .{ .string = system });
    try meta.put(allocator, "release", .{ .string = release });
    try meta.put(allocator, "version", .{ .string = version });
    try meta.put(allocator, "machine", .{ .string = machine });
    try meta.put(allocator, "processor", .{ .string = processor });
    try meta.put(allocator, "cpu", .{ .string = cpu });
    try meta.put(allocator, "logical_cpus", .{ .integer = logical_cpus });
    try meta.put(allocator, "runner_name", .{ .string = runner_name });
    try meta.put(allocator, "runner_image", .{ .string = runner_image });
    try meta.put(allocator, "runner_os", .{ .string = runner_os });
    try meta.put(allocator, "runner_arch", .{ .string = runner_arch });
    try meta.put(allocator, "runner_environment", .{ .string = environment });
    try meta.put(allocator, "github_run_id", .{ .string = github_run_id });
    try meta.put(allocator, "github_run_attempt", .{ .string = github_run_attempt });
    try meta.put(allocator, "github_workflow", .{ .string = github_workflow });
    try meta.put(allocator, "host_fingerprint", .{ .object = fp_obj });
    return meta;
}

pub fn validateReport(report: ObjectMap) !void {
    const sv = report.get("schema_version") orelse return fail("report identity mismatch", .{});
    const kind = report.get("kind") orelse return fail("report identity mismatch", .{});
    if (sv != .integer or sv.integer != SCHEMA_VERSION or kind != .string or !std.mem.eql(u8, kind.string, KIND)) {
        return fail("report identity mismatch", .{});
    }
    const plan_val = report.get("plan") orelse return fail("report plan or records missing", .{});
    const records_val = report.get("records") orelse return fail("report plan or records missing", .{});
    if (plan_val != .object or records_val != .array or records_val.array.items.len == 0) {
        return fail("report plan or records missing", .{});
    }
    const plan = plan_val.object;
    const calls_val = plan.get("leaf_calls") orelse return fail("report leaf call count is invalid", .{});
    if (calls_val != .integer) return fail("report leaf call count is invalid", .{});
    const calls = calls_val.integer;
    _ = try expectedResult(calls);

    const meta_val = report.get("metadata") orelse return fail("report artifact proof missing", .{});
    if (meta_val != .object) return fail("report artifact proof missing", .{});
    const artifacts_val = meta_val.object.get("artifacts") orelse return fail("report artifact proof missing", .{});
    if (artifacts_val != .object) return fail("report artifact proof missing", .{});
    const artifacts = artifacts_val.object;
    const leaf_step_val = artifacts.get("leaf_step") orelse return fail("report artifact or leaf-entry proof mismatch", .{});
    const norm_val = artifacts.get("normalized_comparison") orelse return fail("report artifact or leaf-entry proof mismatch", .{});
    if (leaf_step_val != .object or norm_val != .object) return fail("report artifact or leaf-entry proof mismatch", .{});
    const leaf_proof = leaf_step_val.object;
    const normalized = norm_val.object;

    const sites_en = artifacts.get("cancel_poll_sites_enabled") orelse return fail("report artifact or leaf-entry proof mismatch", .{});
    const sites_dis = artifacts.get("cancel_poll_sites_disabled") orelse return fail("report artifact or leaf-entry proof mismatch", .{});
    if (sites_en != .integer or sites_en.integer <= 0 or sites_dis != .integer or sites_dis.integer != 0) {
        return fail("report artifact or leaf-entry proof mismatch", .{});
    }
    const en_seq = leaf_proof.get("enabled_poll_sequences") orelse return fail("report artifact or leaf-entry proof mismatch", .{});
    const dis_seq = leaf_proof.get("disabled_poll_sequences") orelse return fail("report artifact or leaf-entry proof mismatch", .{});
    const norm_id = leaf_proof.get("normalized_body_identical") orelse return fail("report artifact or leaf-entry proof mismatch", .{});
    if (en_seq != .integer or en_seq.integer != 1 or dis_seq != .integer or dis_seq.integer != 0 or norm_id != .bool or !norm_id.bool) {
        return fail("report artifact or leaf-entry proof mismatch", .{});
    }
    const method = normalized.get("method") orelse return fail("report artifact or leaf-entry proof mismatch", .{});
    if (method != .string or !std.mem.eql(u8, method.string, "complete-poll-and-layout-normalization-v1")) {
        return fail("report artifact or leaf-entry proof mismatch", .{});
    }
    const qual_val = report.get("quality") orelse return fail("report quality summary mismatch", .{});
    if (qual_val != .object) return fail("report quality summary mismatch", .{});
    for ([_][]const u8{
        "equivalent_guest_work",
        "equivalent_checksums",
        "all_intervals_passed",
        "all_timing_overhead_below_one_percent",
    }) |q_key| {
        const qv = qual_val.object.get(q_key) orelse return fail("report quality summary mismatch", .{});
        if (qv != .bool or !qv.bool) return fail("report quality summary mismatch", .{});
    }
}

fn appendShlexCommandEscaped(allocator: Allocator, out: *std.ArrayList(u8), command: []const Value) !void {
    for (command, 0..) |item, idx| {
        if (idx > 0) try out.append(allocator, ' ');
        const s = item.string;
        var needs_quote = s.len == 0;
        for (s) |c| {
            if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-' and c != '.' and c != '/' and c != ':' and c != '=') {
                needs_quote = true;
                break;
            }
        }
        if (!needs_quote) {
            for (s) |c| {
                if (c == '|') try out.appendSlice(allocator, "\\|") else try out.append(allocator, c);
            }
        } else {
            try out.append(allocator, '\'');
            for (s) |c| {
                if (c == '\'') {
                    try out.appendSlice(allocator, "'\"'\"'");
                } else if (c == '|') {
                    try out.appendSlice(allocator, "\\|");
                } else {
                    try out.append(allocator, c);
                }
            }
            try out.append(allocator, '\'');
        }
    }
}

pub fn renderMarkdown(allocator: Allocator, report: ObjectMap) ![]const u8 {
    const metadata = report.get("metadata").?.object;
    const comparison = report.get("comparison").?.object;
    const source = metadata.get("source").?.object;
    const host = metadata.get("host").?.object;
    const tools = metadata.get("tools").?.object;
    const artifacts = metadata.get("artifacts").?.object;
    const plan = report.get("plan").?.object;

    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(allocator, "# Leaf-call AOT cancel-poll cost\n\n## Immutable identities\n\n");
    try out.print(allocator, "- Source commit: `{s}`\n", .{source.get("commit").?.string});
    try out.print(allocator, "- Tracked diff SHA-256: `{s}`\n", .{source.get("tracked_diff_sha256").?.string});
    try out.print(allocator, "- Build-source SHA-256: `{s}`\n", .{source.get("build_source_sha256").?.string});
    try out.print(allocator, "- Platform: `{s}`\n", .{metadata.get("platform_id").?.string});
    try out.print(allocator, "- Host fingerprint: `{s}`\n", .{host.get("host_fingerprint").?.object.get("sha256").?.string});
    try out.print(allocator, "- Host: `{s} {s}` · `{s}` · `{s}` · {d} logical CPUs · `{s}`\n", .{
        host.get("system").?.string,
        host.get("release").?.string,
        host.get("machine").?.string,
        host.get("cpu").?.string,
        if (host.get("logical_cpus")) |lc| (if (lc == .integer) lc.integer else 1) else 1,
        host.get("runner_environment").?.string,
    });
    try out.print(allocator, "- AOT architecture: `{s}`\n", .{artifacts.get("architecture").?.string});
    try out.print(allocator, "- Fixture: `{s}`\n", .{metadata.get("fixture").?.object.get("sha256").?.string});
    try out.print(allocator, "- Zig: `{s}`\n", .{tools.get("zig").?.string});
    try out.print(allocator, "- Objdump: `{s}`\n", .{tools.get("objdump").?.string});
    try out.print(allocator, "- wamrc: `{s}` · `{s}`\n", .{
        tools.get("compiler").?.object.get("wamrc_sha256").?.string,
        tools.get("compiler").?.object.get("wamrc_version").?.string,
    });
    try out.print(allocator, "- wamr: `{s}` · `{s}`\n\n", .{
        tools.get("runtime").?.object.get("wamr_sha256").?.string,
        tools.get("runtime").?.object.get("wamr_version").?.string,
    });
    try out.appendSlice(allocator, "| AOT condition | SHA-256 | File bytes | Text bytes |\n|---|---|---:|---:|\n");
    for ([_][]const u8{ "cancel-points-off", "cancel-points-on" }) |cond| {
        const art = artifacts.get("conditions").?.object.get(cond).?.object;
        try out.print(allocator, "| `{s}` | `{s}` | {d} | {d} |\n", .{
            cond,
            art.get("sha256").?.string,
            art.get("file_bytes").?.integer,
            art.get("text_bytes").?.integer,
        });
    }
    const norm = artifacts.get("normalized_comparison").?.object;
    const cmd_comp = artifacts.get("compile_command_comparison").?.object;
    const leaf_step = artifacts.get("leaf_step").?.object;
    try out.print(allocator,
        \\
        \\## AOT single-variable proof
        \\
        \\- Normalization: `{s}`; {d} functions identical after removing {d} complete poll sequences and normalizing {d} control-flow layout fields and {d} jump-table entries.
        \\- Normalized compile command SHA-256: `{s}`; the only semantic option difference is `{s}`.
        \\- `leaf_step`: wasm function `{d}`, local function `{d}`, enabled entry-poll offset `{d}`; complete poll sequences on/off `1 / 0`; compiler ABI entry-prefix tail `{s}`; normalized body identical.
        \\
        \\| AOT condition | Exact compile command |
        \\|---|---|
        \\
    , .{
        norm.get("method").?.string,
        norm.get("normalized_functions_identical").?.integer,
        norm.get("poll_sequences_removed").?.integer,
        norm.get("relative_control_flow_fields_normalized").?.integer,
        norm.get("jump_table_entries_normalized").?.integer,
        cmd_comp.get("normalized_command_sha256").?.string,
        cmd_comp.get("differing_flag").?.string,
        leaf_step.get("export_function_index").?.integer,
        leaf_step.get("local_function_index").?.integer,
        leaf_step.get("enabled_entry_poll_offset").?.integer,
        leaf_step.get("entry_prefix_suffix_hex").?.string,
    });
    for ([_][]const u8{ "cancel-points-off", "cancel-points-on" }) |cond| {
        try out.print(allocator, "| `{s}` | `", .{cond});
        try appendShlexCommandEscaped(
            allocator,
            &out,
            artifacts.get("compile_commands").?.object.get(cond).?.array.items,
        );
        try out.appendSlice(allocator, "` |\n");
    }

    const sizing = plan.get("sizing").?.object;
    try out.print(allocator,
        \\
        \\## Measurement plan and validation
        \\
        \\- Work: `{d}` noinline leaf calls (`{d}` batches of `{d}`).
        \\- Invocations: `{d}` discarded balanced warmup pairs; `{d}` retained balanced sample pairs.
        \\- Order: `{s}`; raw rows below retain execution order.
        \\- Sizing: `{s}`; target `{d}` ns; minimum `{d}` ns.
        \\- Validation covers every warmup and retained sample; the optional sizing pilot intentionally does not enforce the minimum interval.
        \\- Static poll sites on/off: `{d}` / `{d}`; `{d}` bytes/site.
        \\
        \\| Validation | Result |
        \\|---|---|
        \\
    , .{
        plan.get("leaf_calls").?.integer,
        plan.get("batches").?.integer,
        plan.get("leaf_unroll").?.integer,
        plan.get("warmups").?.integer,
        plan.get("samples").?.integer,
        plan.get("pair_order").?.string,
        sizing.get("kind").?.string,
        plan.get("target_interval_ns").?.integer,
        plan.get("minimum_interval_ns").?.integer,
        artifacts.get("cancel_poll_sites_enabled").?.integer,
        artifacts.get("cancel_poll_sites_disabled").?.integer,
        artifacts.get("bytes_per_poll_site").?.integer,
    });

    const qual = report.get("quality").?.object;
    for ([_][]const u8{
        "equivalent_guest_work",
        "equivalent_checksums",
        "all_intervals_passed",
        "all_timing_overhead_below_one_percent",
    }) |q_key| {
        const b = qual.get(q_key).?.bool;
        try out.print(allocator, "| `{s}` | `{s}` |\n", .{ q_key, if (b) "true" else "false" });
    }

    try out.appendSlice(allocator,
        \\
        \\## Summary
        \\
        \\| Condition | Guest median ms | Median leaf calls/s | Host-wall median ms |
        \\|---|---:|---:|---:|
        \\
    );
    for (report.get("summaries").?.array.items) |sum_val| {
        const s = sum_val.object;
        const g_med = valueAsF64(s.get("guest_elapsed_ns").?.object.get("median").?) / 1e6;
        const c_med = valueAsF64(s.get("leaf_calls_per_second").?.object.get("median").?);
        const h_med = valueAsF64(s.get("host_wall_elapsed_ns").?.object.get("median").?) / 1e6;
        try out.print(allocator, "| `{s}` | {d:.3} | {d:.3} | {d:.3} |\n", .{
            s.get("condition").?.string,
            g_med,
            c_med,
            h_med,
        });
    }
    try out.print(allocator, "\n- Median polls-on / polls-off guest time: `{d:.9}` (`{d:+.4}%`).\n", .{
        valueAsF64(comparison.get("median_on_over_off_elapsed_ratio").?),
        valueAsF64(comparison.get("median_elapsed_delta_percent").?),
    });
    try out.print(allocator, "- Median polls-on minus polls-off cost per timed leaf call: `{d:.6} ns`.\n", .{
        valueAsF64(comparison.get("median_on_minus_off_ns_per_leaf_call").?),
    });
    try out.print(allocator, "- Median polls-on / polls-off throughput: `{d:.9}`.\n", .{
        valueAsF64(comparison.get("median_on_over_off_throughput_ratio").?),
    });
    try out.appendSlice(allocator,
        \\
        \\## Raw invocation order
        \\
        \\| Phase | Pair | Position | Condition | Calls | Leaf polls | Checksum | Raw guest ns | Timer overhead ns | Guest ns | Host-wall ns | Correct | Command |
        \\|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
        \\
    );

    var all_invocations: std.ArrayList(Value) = .empty;
    if (report.get("pilot")) |p| {
        if (p == .object) try all_invocations.append(allocator, p);
    }
    for (report.get("warmups").?.array.items) |w| try all_invocations.append(allocator, w);
    for (report.get("records").?.array.items) |r| try all_invocations.append(allocator, r);

    for (all_invocations.items) |inv_val| {
        const item = inv_val.object;
        const guest = item.get("guest").?.object;
        const checksum = try parseStrictJsonU64(guest.get("checksum").?, "checksum");
        try out.print(
            allocator,
            "| `{s}` | {d} | {d} | `{s}` | {d} | {d} | {d} | {d} | {d} | {d} | {d} | `{s}` | `",
            .{
                item.get("phase").?.string,
                item.get("pair_index").?.integer,
                item.get("position").?.integer,
                item.get("condition").?.string,
                item.get("leaf_calls").?.integer,
                item.get("leaf_entry_poll_opportunities").?.integer,
                checksum,
                item.get("raw_guest_elapsed_ns").?.integer,
                item.get("timing_overhead_ns").?.integer,
                item.get("guest_elapsed_ns").?.integer,
                item.get("host_wall_elapsed_ns").?.integer,
                if (item.get("correct").?.bool) "true" else "false",
            },
        );
        try appendShlexCommandEscaped(allocator, &out, item.get("command").?.array.items);
        try out.appendSlice(allocator, "` |\n");
    }

    return out.toOwnedSlice(allocator);
}

fn splitRunnerArgs(allocator: Allocator, text: []const u8) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    var tok_it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (tok_it.next()) |t| try list.append(allocator, t);
    return list.toOwnedSlice(allocator);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();
    _ = args_it.next();

    var repo_arg: []const u8 = ".";
    var output_dir_arg: []const u8 = "zig-out/leaf-cancel-cost";
    var calls_opt: ?i64 = null;
    var pilot_calls: i64 = 4_194_304;
    var target_interval_ms: f64 = 3000.0;
    var min_interval_ms: f64 = 1250.0;
    var warmups: usize = 2;
    var samples: usize = 12;
    var timeout: f64 = 60.0;
    var optimize: []const u8 = "ReleaseFast";
    var target: ?[]const u8 = null;
    var aot_target: ?[]const u8 = null;
    var runner_str: []const u8 = "";
    var platform_id: []const u8 = if (builtin.cpu.arch == .aarch64) "local-linux-aarch64" else "local-linux-x86_64";
    var runner_environment: ?[]const u8 = null;
    var rebuild: bool = false;

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--repo")) {
            repo_arg = args_it.next() orelse std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--output-dir")) {
            output_dir_arg = args_it.next() orelse std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--calls")) {
            calls_opt = std.fmt.parseInt(i64, args_it.next() orelse std.process.exit(2), 10) catch std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--pilot-calls")) {
            pilot_calls = std.fmt.parseInt(i64, args_it.next() orelse std.process.exit(2), 10) catch std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--target-interval-ms")) {
            target_interval_ms = std.fmt.parseFloat(f64, args_it.next() orelse std.process.exit(2)) catch std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--min-interval-ms")) {
            min_interval_ms = std.fmt.parseFloat(f64, args_it.next() orelse std.process.exit(2)) catch std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--warmups")) {
            warmups = std.fmt.parseInt(usize, args_it.next() orelse std.process.exit(2), 10) catch std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--samples")) {
            samples = std.fmt.parseInt(usize, args_it.next() orelse std.process.exit(2), 10) catch std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--timeout")) {
            timeout = std.fmt.parseFloat(f64, args_it.next() orelse std.process.exit(2)) catch std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--optimize")) {
            optimize = args_it.next() orelse std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--target")) {
            target = args_it.next() orelse std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--aot-target")) {
            aot_target = args_it.next() orelse std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--runner")) {
            runner_str = args_it.next() orelse std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--platform-id")) {
            platform_id = args_it.next() orelse std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--runner-environment")) {
            runner_environment = args_it.next() orelse std.process.exit(2);
        } else if (std.mem.eql(u8, arg, "--rebuild")) {
            rebuild = true;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            std.process.exit(2);
        }
    }

    if (calls_opt) |c| {
        if (c <= 0 or @rem(c, LEAF_UNROLL) != 0 or c > MAX_LEAF_CALLS) std.process.exit(2);
    }
    if (pilot_calls <= 0 or @rem(pilot_calls, LEAF_UNROLL) != 0 or pilot_calls > MAX_LEAF_CALLS) std.process.exit(2);
    if (target_interval_ms < min_interval_ms or min_interval_ms <= 0) std.process.exit(2);
    if (warmups > 2 or samples == 0 or samples > 12 or samples % 2 != 0 or timeout <= 0) std.process.exit(2);

    try std.Io.Dir.cwd().createDirPath(io, output_dir_arg);
    const repo = try std.Io.Dir.cwd().realPathFileAlloc(io, repo_arg, allocator);
    const output = try std.Io.Dir.cwd().realPathFileAlloc(io, output_dir_arg, allocator);

    const report_json_path = try std.fs.path.join(allocator, &.{ output, "report.json" });
    const report_md_path = try std.fs.path.join(allocator, &.{ output, "report.md" });
    std.Io.Dir.cwd().deleteFile(io, report_json_path) catch {};
    std.Io.Dir.cwd().deleteFile(io, report_md_path) catch {};

    const source = sourceIdentity(allocator, io, repo) catch std.process.exit(2);
    const fixture = fixtureIdentity(allocator, io, repo) catch std.process.exit(2);
    const built = buildTools(
        allocator,
        io,
        init.environ_map,
        repo,
        output,
        source,
        optimize,
        target,
        rebuild,
    ) catch std.process.exit(2);

    const arch = if (aot_target) |at|
        at
    else if (target != null and std.mem.startsWith(u8, target.?, "aarch64"))
        "aarch64"
    else if (builtin.cpu.arch == .aarch64)
        "aarch64"
    else
        "x86_64";

    const compiled = compileArtifacts(
        allocator,
        io,
        init.environ_map,
        repo,
        output,
        built.compiler,
        arch,
    ) catch std.process.exit(2);

    const norm_work_dir = try std.fs.path.join(allocator, &.{ output, "cache/artifact-normalization" });
    const artifact_report = artifactIdentity(
        allocator,
        io,
        compiled.off_path,
        compiled.on_path,
        compiled.off_cmd,
        compiled.on_cmd,
        arch,
        norm_work_dir,
    ) catch std.process.exit(2);

    const leaf_step = artifact_report.get("leaf_step").?.object;
    const leaf_entry_poll_proven = leaf_step.get("enabled_poll_sequences").?.integer == 1 and
        leaf_step.get("disabled_poll_sequences").?.integer == 0 and
        leaf_step.get("normalized_body_identical").?.bool;

    const runner = try splitRunnerArgs(allocator, runner_str);
    const minimum_interval_ns: i64 = @intFromFloat(min_interval_ms * 1_000_000.0);
    const target_interval_ns: i64 = @intFromFloat(target_interval_ms * 1_000_000.0);

    var pilot_val: Value = .null;
    var calls: i64 = calls_opt orelse 0;
    if (calls_opt == null) {
        const pilot_obj = measure(
            allocator,
            io,
            repo,
            runner,
            built.runtime,
            compiled.on_path,
            "cancel-points-on",
            pilot_calls,
            timeout,
            minimum_interval_ns,
            false,
            "pilot",
            0,
            0,
            leaf_entry_poll_proven,
        ) catch std.process.exit(2);
        calls = selectCalls(
            pilot_calls,
            pilot_obj.get("guest_elapsed_ns").?.integer,
            target_interval_ns,
        ) catch std.process.exit(2);
        pilot_val = .{ .object = pilot_obj };
    }

    var warmup_objs: std.ArrayList(ObjectMap) = .empty;
    var warmup_arr = Array.init(allocator);
    for (0..warmups) |pair_index| {
        const order = conditionOrder(pair_index);
        for (order, 0..) |cond, pos| {
            const art_path = if (std.mem.eql(u8, cond, "cancel-points-off")) compiled.off_path else compiled.on_path;
            const m = measure(
                allocator,
                io,
                repo,
                runner,
                built.runtime,
                art_path,
                cond,
                calls,
                timeout,
                minimum_interval_ns,
                true,
                "warmup",
                pair_index,
                pos,
                leaf_entry_poll_proven,
            ) catch std.process.exit(2);
            try warmup_objs.append(allocator, m);
            try warmup_arr.append(.{ .object = m });
        }
    }

    var record_objs: std.ArrayList(ObjectMap) = .empty;
    var record_arr = Array.init(allocator);
    for (0..samples) |pair_index| {
        const order = conditionOrder(pair_index);
        for (order, 0..) |cond, pos| {
            const art_path = if (std.mem.eql(u8, cond, "cancel-points-off")) compiled.off_path else compiled.on_path;
            const m = measure(
                allocator,
                io,
                repo,
                runner,
                built.runtime,
                art_path,
                cond,
                calls,
                timeout,
                minimum_interval_ns,
                true,
                "sample",
                pair_index,
                pos,
                leaf_entry_poll_proven,
            ) catch std.process.exit(2);
            try record_objs.append(allocator, m);
            try record_arr.append(.{ .object = m });
        }
    }

    const sum_res = summarize(allocator, record_objs.items, calls) catch std.process.exit(2);
    const host = hostMetadata(allocator, io, init.environ_map, runner_environment) catch std.process.exit(2);

    var comp_cmd_arr = Array.init(allocator);
    for (built.compiler.command) |s| try comp_cmd_arr.append(.{ .string = s });
    var comp_tool: ObjectMap = .empty;
    try comp_tool.put(allocator, "build_command", .{ .array = comp_cmd_arr });
    try comp_tool.put(allocator, "cache_key", .{ .string = built.compiler.cache_key });
    try comp_tool.put(allocator, "wamrc_path", .{ .string = built.compiler.wamrc.? });
    try comp_tool.put(allocator, "wamrc_sha256", .{ .string = try sha256File(allocator, io, built.compiler.wamrc.?) });
    try comp_tool.put(allocator, "wamrc_version", .{ .string = try commandIdentity(allocator, io, &.{ built.compiler.wamrc.?, "version" }) });

    var rt_cmd_arr = Array.init(allocator);
    for (built.runtime.command) |s| try rt_cmd_arr.append(.{ .string = s });
    var rt_ver_cmd: std.ArrayList([]const u8) = .empty;
    for (runner) |r| try rt_ver_cmd.append(allocator, r);
    try rt_ver_cmd.appendSlice(allocator, &.{ built.runtime.wamr, "version" });
    var rt_tool: ObjectMap = .empty;
    try rt_tool.put(allocator, "build_command", .{ .array = rt_cmd_arr });
    try rt_tool.put(allocator, "cache_key", .{ .string = built.runtime.cache_key });
    try rt_tool.put(allocator, "wamr_path", .{ .string = built.runtime.wamr });
    try rt_tool.put(allocator, "wamr_sha256", .{ .string = try sha256File(allocator, io, built.runtime.wamr) });
    try rt_tool.put(allocator, "wamr_version", .{ .string = try commandIdentity(allocator, io, rt_ver_cmd.items) });

    var tools_obj: ObjectMap = .empty;
    try tools_obj.put(allocator, "zig", .{ .string = try commandIdentity(allocator, io, &.{ "zig", "version" }) });
    try tools_obj.put(allocator, "objdump", .{ .string = try commandIdentity(allocator, io, &.{ "objdump", "--version" }) });
    try tools_obj.put(allocator, "compiler", .{ .object = comp_tool });
    try tools_obj.put(allocator, "runtime", .{ .object = rt_tool });

    var runner_arr = Array.init(allocator);
    for (runner) |r| try runner_arr.append(.{ .string = r });
    var aff_arr = Array.init(allocator);
    for (try selectedCpuAffinity(allocator, io)) |cpu_id| try aff_arr.append(.{ .integer = cpu_id });

    var exec_obj: ObjectMap = .empty;
    try exec_obj.put(allocator, "target", .{ .string = target orelse "native" });
    try exec_obj.put(allocator, "aot_target", .{ .string = arch });
    try exec_obj.put(allocator, "runner", .{ .array = runner_arr });
    try exec_obj.put(allocator, "host_cpu_affinity", .{ .array = aff_arr });
    try exec_obj.put(allocator, "optimize", .{ .string = optimize });

    var metadata: ObjectMap = .empty;
    try metadata.put(allocator, "collected_at", .{ .string = try collectedAt(allocator, io) });
    try metadata.put(allocator, "platform_id", .{ .string = platform_id });
    try metadata.put(allocator, "source", .{ .object = source });
    try metadata.put(allocator, "fixture", .{ .object = fixture });
    try metadata.put(allocator, "host", .{ .object = host });
    try metadata.put(allocator, "tools", .{ .object = tools_obj });
    try metadata.put(allocator, "artifacts", .{ .object = artifact_report });
    try metadata.put(allocator, "execution", .{ .object = exec_obj });

    var sizing_obj: ObjectMap = .empty;
    try sizing_obj.put(allocator, "kind", .{ .string = if (pilot_val == .object) "single-enabled-pilot-linear-scale" else "explicit-call-count" });
    try sizing_obj.put(allocator, "pilot_calls", if (pilot_val == .object) .{ .integer = pilot_calls } else .null);
    try sizing_obj.put(allocator, "selected_calls", .{ .integer = calls });

    const batches = @divExact(calls, LEAF_UNROLL);
    const non_leaf_bound = batches + 1;
    const calls_f: f64 = @floatFromInt(calls);
    const non_leaf_f: f64 = @floatFromInt(non_leaf_bound);

    var plan_obj: ObjectMap = .empty;
    try plan_obj.put(allocator, "identity", .{ .string = "issue-963-leaf-cancel-cost-v1" });
    try plan_obj.put(allocator, "scope", .{ .string = "isolated-from-wasi-thread-benchmark-966" });
    try plan_obj.put(allocator, "leaf_calls", .{ .integer = calls });
    try plan_obj.put(allocator, "leaf_unroll", .{ .integer = LEAF_UNROLL });
    try plan_obj.put(allocator, "batches", .{ .integer = batches });
    try plan_obj.put(allocator, "leaf_entry_poll_opportunities_per_enabled_sample", .{ .integer = calls });
    try plan_obj.put(allocator, "non_leaf_poll_opportunities_upper_bound_per_enabled_sample", .{ .integer = non_leaf_bound });
    try plan_obj.put(allocator, "minimum_leaf_entry_poll_share", .{ .float = calls_f / (calls_f + non_leaf_f) });
    try plan_obj.put(allocator, "warmups", .{ .integer = @intCast(warmups) });
    try plan_obj.put(allocator, "samples", .{ .integer = @intCast(samples) });
    try plan_obj.put(allocator, "minimum_interval_ns", .{ .integer = minimum_interval_ns });
    try plan_obj.put(allocator, "target_interval_ns", .{ .integer = target_interval_ns });
    try plan_obj.put(allocator, "pair_order", .{ .string = "alternating-off-on/on-off" });
    try plan_obj.put(allocator, "clock_id", .{ .string = "wasi-process-cputime" });
    try plan_obj.put(allocator, "sizing", .{ .object = sizing_obj });

    var quality_obj: ObjectMap = .empty;
    try quality_obj.put(allocator, "equivalent_guest_work", .{ .bool = true });
    try quality_obj.put(allocator, "equivalent_checksums", .{ .bool = true });
    try quality_obj.put(allocator, "all_intervals_passed", .{ .bool = true });
    try quality_obj.put(allocator, "all_timing_overhead_below_one_percent", .{ .bool = true });

    var report: ObjectMap = .empty;
    try report.put(allocator, "schema_version", .{ .integer = SCHEMA_VERSION });
    try report.put(allocator, "kind", .{ .string = KIND });
    try report.put(allocator, "metadata", .{ .object = metadata });
    try report.put(allocator, "plan", .{ .object = plan_obj });
    try report.put(allocator, "pilot", pilot_val);
    try report.put(allocator, "warmups", .{ .array = warmup_arr });
    try report.put(allocator, "records", .{ .array = record_arr });
    try report.put(allocator, "summaries", .{ .array = sum_res.summaries });
    try report.put(allocator, "comparison", .{ .object = sum_res.comparison });
    try report.put(allocator, "quality", .{ .object = quality_obj });

    validateReport(report) catch std.process.exit(2);
    atomicWriteJson(allocator, io, report_json_path, .{ .object = report }) catch std.process.exit(2);
    const md_text = renderMarkdown(allocator, report) catch std.process.exit(2);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = report_md_path, .data = md_text }) catch std.process.exit(2);
    var stdout_buf: [512]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    stdout_w.interface.print("{s}\n{s}\n", .{ report_json_path, report_md_path }) catch {};
    stdout_w.interface.flush() catch {};
}

// ============================================================================
// Unit Tests
// ============================================================================

const testing = std.testing;

fn makeTestDir(allocator: Allocator, io: std.Io, label: []const u8) ![]const u8 {
    var rand_bytes: [8]u8 = undefined;
    io.random(&rand_bytes);
    const cwd_real = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/lcc-test-{s}-{x}", .{
        cwd_real,
        label,
        std.mem.readInt(u64, &rand_bytes, .little),
    });
    try std.Io.Dir.cwd().createDirPath(io, dir);
    return dir;
}

fn testReturnInstruction(arch: []const u8) []const u8 {
    if (std.mem.eql(u8, arch, "x86_64")) return &[_]u8{0xC3};
    return &[_]u8{ 0xC0, 0x03, 0x5F, 0xD6 }; // 0xD65F03C0
}

fn testEntryBody(allocator: Allocator, arch: []const u8) ![]u8 {
    const prefix = cancelPollEntryPrefixSuffix(arch).?;
    const ret = testReturnInstruction(arch);
    return std.mem.concat(allocator, u8, &.{ prefix, ret });
}

fn testSection(allocator: Allocator, section_type: u32, payload: []const u8) ![]u8 {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], section_type, .little);
    std.mem.writeInt(u32, hdr[4..8], @intCast(payload.len), .little);
    return std.mem.concat(allocator, u8, &.{ &hdr, payload });
}

fn testBuildAot(allocator: Allocator, functions: []const []const u8, leaf_local: u32) ![]u8 {
    var text: std.ArrayList(u8) = .empty;
    var func_payload: std.ArrayList(u8) = .empty;
    var cnt_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &cnt_buf, @intCast(functions.len), .little);
    try func_payload.appendSlice(allocator, &cnt_buf);
    for (functions) |fn_bytes| {
        var entry: [8]u8 = undefined;
        std.mem.writeInt(u32, entry[0..4], @intCast(text.items.len), .little);
        std.mem.writeInt(u32, entry[4..8], 0, .little);
        try func_payload.appendSlice(allocator, &entry);
        try text.appendSlice(allocator, fn_bytes);
    }

    const name = "leaf_step";
    var exp_payload: std.ArrayList(u8) = .empty;
    var exp_hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, exp_hdr[0..4], 1, .little);
    std.mem.writeInt(u32, exp_hdr[4..8], @intCast(name.len), .little);
    try exp_payload.appendSlice(allocator, &exp_hdr);
    try exp_payload.appendSlice(allocator, name);
    try exp_payload.append(allocator, 0);
    var leaf_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &leaf_buf, leaf_local, .little);
    try exp_payload.appendSlice(allocator, &leaf_buf);

    var header: [8]u8 = undefined;
    std.mem.writeInt(u32, header[0..4], AOT_MAGIC, .little);
    std.mem.writeInt(u32, header[4..8], AOT_VERSION, .little);

    const sec0_zeros: [40]u8 = @splat(0);
    const imp_zeros: [4]u8 = @splat(0);

    return std.mem.concat(allocator, u8, &.{
        &header,
        try testSection(allocator, 0, &sec0_zeros),
        try testSection(allocator, 2, text.items),
        try testSection(allocator, 3, func_payload.items),
        try testSection(allocator, 4, exp_payload.items),
        try testSection(allocator, 8, &imp_zeros),
    });
}

fn objdumpSupportsX86(allocator: Allocator, io: std.Io) bool {
    const res = runProcess(allocator, io, &.{ "objdump", "-i" }, null, null) catch return false;
    return res.code == 0 and std.mem.indexOf(u8, res.stdout, "i386:x86-64") != null;
}

test "expected result is closed form and requires unroll" {
    const calls: i64 = 128;
    const res = try expectedResult(calls);
    try testing.expectEqual(@as(i64, 2), res.batches);
    try testing.expectEqual(LEAF_SEED +% (LEAF_STEP *% 128), res.checksum);
    try testing.expectError(error.InvalidHarness, expectedResult(65));
}

test "guest parser rejects work mismatch" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const exp = try expectedResult(64);
    const valid_json = try std.fmt.allocPrint(
        allocator,
        "{{\"kind\":\"leaf-cancel-cost-result\",\"leaf_calls\":64,\"leaf_unroll\":64,\"batches\":1,\"checksum\":{d},\"clock_id\":\"wasi-process-cputime\",\"raw_elapsed_ns\":2000100,\"timing_overhead_ns\":100,\"elapsed_ns\":2000000,\"timing_overhead_ppm\":49}}\n",
        .{exp.checksum},
    );
    const parsed = try parseGuestResult(allocator, valid_json, 64, 1_000_000, true);
    try testing.expectEqual(exp.checksum, try parseStrictJsonU64(parsed.get("checksum").?, "checksum"));

    const mismatch_json = try std.fmt.allocPrint(
        allocator,
        "{{\"kind\":\"leaf-cancel-cost-result\",\"leaf_calls\":128,\"leaf_unroll\":64,\"batches\":1,\"checksum\":{d},\"clock_id\":\"wasi-process-cputime\",\"raw_elapsed_ns\":2000100,\"timing_overhead_ns\":100,\"elapsed_ns\":2000000,\"timing_overhead_ppm\":49}}\n",
        .{exp.checksum},
    );
    try testing.expectError(error.InvalidHarness, parseGuestResult(allocator, mismatch_json, 64, 1_000_000, true));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "leaf_calls") != null);
}

test "sizing rounds to complete unrolled batches" {
    const selected = try selectCalls(64, 100, 201);
    try testing.expectEqual(@as(i64, 192), selected);
    try testing.expectEqual(@as(i64, 0), @rem(selected, LEAF_UNROLL));
    try testing.expectError(error.InvalidHarness, selectCalls(MAX_LEAF_CALLS, 1, 2));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "safety limit") != null);
}

test "balanced pair summary reports on minus off" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const makeRec = struct {
        fn build(alloc: Allocator, pi: i64, cond: []const u8, guest_ns: i64, cps: f64, wall_ns: i64) !ObjectMap {
            var m: ObjectMap = .empty;
            try m.put(alloc, "pair_index", .{ .integer = pi });
            try m.put(alloc, "condition", .{ .string = cond });
            try m.put(alloc, "guest_elapsed_ns", .{ .integer = guest_ns });
            try m.put(alloc, "leaf_calls_per_second", .{ .float = cps });
            try m.put(alloc, "host_wall_elapsed_ns", .{ .integer = wall_ns });
            return m;
        }
    }.build;

    const records = [_]ObjectMap{
        try makeRec(allocator, 0, "cancel-points-off", 100, 640_000_000.0, 120),
        try makeRec(allocator, 0, "cancel-points-on", 120, 533_333_333.0, 140),
        try makeRec(allocator, 1, "cancel-points-on", 132, 484_848_484.0, 150),
        try makeRec(allocator, 1, "cancel-points-off", 110, 581_818_181.0, 130),
    };
    const res = try summarize(allocator, &records, 64);
    try testing.expectApproxEqAbs(@as(f64, 1.2), res.comparison.get("median_on_over_off_elapsed_ratio").?.float, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 21.0 / 64.0), res.comparison.get("median_on_minus_off_ns_per_leaf_call").?.float, 1e-9);
}

test "invocation validation rejects reordered pair" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const exp = try (try expectedResult(64)).toObjectMap(allocator);
    var pair: [2]ObjectMap = undefined;
    for (conditionOrder(0), 0..) |cond, pos| {
        var m: ObjectMap = .empty;
        try m.put(allocator, "phase", .{ .string = "sample" });
        try m.put(allocator, "pair_index", .{ .integer = 0 });
        try m.put(allocator, "position", .{ .integer = @intCast(pos) });
        try m.put(allocator, "condition", .{ .string = cond });
        try m.put(allocator, "leaf_calls", .{ .integer = 64 });
        try m.put(allocator, "leaf_entry_poll_opportunities", .{ .integer = if (std.mem.eql(u8, cond, "cancel-points-on")) 64 else 0 });
        try m.put(allocator, "batches", .{ .integer = 1 });
        try m.put(allocator, "guest", .{ .object = exp });
        try m.put(allocator, "correct", .{ .bool = true });
        pair[pos] = m;
    }
    try validateInvocations(&pair, "sample", 1, 64);
    const reversed = [2]ObjectMap{ pair[1], pair[0] };
    try testing.expectError(error.InvalidHarness, validateInvocations(&reversed, "sample", 1, 64));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "ordering") != null);
}

test "measurement requires leaf entry artifact proof" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;

    const dummy_build = Build{
        .name = "dummy",
        .prefix = "zig-out/dummy",
        .wamr = "zig-out/dummy/wamr",
        .wamrc = null,
        .command = &.{},
        .cache_key = &(@as([64]u8, @splat('0'))),
    };
    try testing.expectError(error.InvalidHarness, measure(allocator, io, .{
        .repo = ".",
        .runner = &.{},
        .runtime = dummy_build,
        .artifact = "zig-out/dummy/leaf.cwasm",
        .condition = "cancel-points-on",
        .calls = 64,
        .timeout = 1.0,
        .minimum_interval_ns = 1,
        .enforce_quality = true,
        .phase = "sample",
        .pair_index = 0,
        .position = 0,
        .leaf_entry_poll_proven = false,
    }));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "artifact proof") != null);
}

test "artifact identity proves commands layout and leaf entry" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "art-id");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const has_x86 = objdumpSupportsX86(allocator, io);
    for ([_][]const u8{ "aarch64", "x86_64" }) |arch| {
        if (std.mem.eql(u8, arch, "x86_64") and !has_x86) continue;
        const sequence = cancelPollSequence(arch).?;
        const off = try std.fs.path.join(allocator, &.{ dir, try std.fmt.allocPrint(allocator, "{s}-off.cwasm", .{arch}) });
        const on = try std.fs.path.join(allocator, &.{ dir, try std.fmt.allocPrint(allocator, "{s}-on.cwasm", .{arch}) });
        const body = try testEntryBody(allocator, arch);
        const on_body = try std.mem.concat(allocator, u8, &.{
            cancelPollEntryPrefixSuffix(arch).?,
            sequence,
            testReturnInstruction(arch),
        });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = off, .data = try testBuildAot(allocator, &.{body}, 0) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = on, .data = try testBuildAot(allocator, &.{on_body}, 0) });

        const off_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, "--benchmark-disable-cancel-points", FIXTURE, "-o", off };
        const on_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, FIXTURE, "-o", on };
        const norm_dir = try std.fs.path.join(allocator, &.{ dir, try std.fmt.allocPrint(allocator, "{s}-norm", .{arch}) });
        const identity = try artifactIdentity(allocator, io, off, on, &off_cmd, &on_cmd, arch, norm_dir);

        try testing.expectEqual(@as(i64, 1), identity.get("cancel_poll_sites_enabled").?.integer);
        try testing.expectEqual(@as(i64, 0), identity.get("cancel_poll_sites_disabled").?.integer);
        try testing.expectEqual(@as(i64, @intCast(sequence.len)), identity.get("text_delta_bytes").?.integer);
        try testing.expectEqual(@as(i64, 0), identity.get("leaf_step").?.object.get("local_function_index").?.integer);
        try testing.expectEqual(true, identity.get("leaf_step").?.object.get("normalized_body_identical").?.bool);
    }
}

test "global poll sequence without leaf entry is rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "global-poll");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const arch = "aarch64";
    const sequence = cancelPollSequence(arch).?;
    const body = try testEntryBody(allocator, arch);
    const other = testReturnInstruction(arch);
    const on_other = try std.mem.concat(allocator, u8, &.{ sequence, other });
    const off = try std.fs.path.join(allocator, &.{ dir, "off.cwasm" });
    const on = try std.fs.path.join(allocator, &.{ dir, "on.cwasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = off, .data = try testBuildAot(allocator, &.{ body, other }, 0) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = on, .data = try testBuildAot(allocator, &.{ body, on_other }, 0) });

    const off_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, "--benchmark-disable-cancel-points", FIXTURE, "-o", off };
    const on_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, FIXTURE, "-o", on };
    const norm_dir = try std.fs.path.join(allocator, &.{ dir, "norm" });
    try testing.expectError(error.InvalidHarness, artifactIdentity(allocator, io, off, on, &off_cmd, &on_cmd, arch, norm_dir));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "leaf_step") != null);
}

test "unrelated function difference is rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "unrel-diff");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const has_x86 = objdumpSupportsX86(allocator, io);
    for ([_][]const u8{ "aarch64", "x86_64" }) |arch| {
        if (std.mem.eql(u8, arch, "x86_64") and !has_x86) continue;
        const sequence = cancelPollSequence(arch).?;
        const body = try testEntryBody(allocator, arch);
        const changed: []const u8 = if (std.mem.eql(u8, arch, "x86_64"))
            &[_]u8{0x90}
        else
            &[_]u8{ 0x1F, 0x20, 0x03, 0xD5 }; // 0xD503201F
        const on_body = try std.mem.concat(allocator, u8, &.{
            cancelPollEntryPrefixSuffix(arch).?,
            sequence,
            changed,
        });
        const off = try std.fs.path.join(allocator, &.{ dir, try std.fmt.allocPrint(allocator, "{s}-off.cwasm", .{arch}) });
        const on = try std.fs.path.join(allocator, &.{ dir, try std.fmt.allocPrint(allocator, "{s}-on.cwasm", .{arch}) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = off, .data = try testBuildAot(allocator, &.{body}, 0) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = on, .data = try testBuildAot(allocator, &.{on_body}, 0) });

        const off_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, "--benchmark-disable-cancel-points", FIXTURE, "-o", off };
        const on_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, FIXTURE, "-o", on };
        const norm_dir = try std.fs.path.join(allocator, &.{ dir, try std.fmt.allocPrint(allocator, "{s}-norm", .{arch}) });
        try testing.expectError(error.InvalidHarness, artifactIdentity(allocator, io, off, on, &off_cmd, &on_cmd, arch, norm_dir));
        try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "unrelated") != null or std.mem.indexOf(u8, lastErrorMessage(), "layout") != null);
    }
}

test "compile command extra option is rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "cmd-extra");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const arch = "aarch64";
    const sequence = cancelPollSequence(arch).?;
    const body = try testEntryBody(allocator, arch);
    const on_body = try std.mem.concat(allocator, u8, &.{
        cancelPollEntryPrefixSuffix(arch).?,
        sequence,
        testReturnInstruction(arch),
    });
    const off = try std.fs.path.join(allocator, &.{ dir, "off.cwasm" });
    const on = try std.fs.path.join(allocator, &.{ dir, "on.cwasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = off, .data = try testBuildAot(allocator, &.{body}, 0) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = on, .data = try testBuildAot(allocator, &.{on_body}, 0) });

    const off_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, "--benchmark-disable-cancel-points", FIXTURE, "-o", off };
    const on_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, "-O0", FIXTURE, "-o", on };
    const norm_dir = try std.fs.path.join(allocator, &.{ dir, "norm" });
    try testing.expectError(error.InvalidHarness, artifactIdentity(allocator, io, off, on, &off_cmd, &on_cmd, arch, norm_dir));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "commands differ") != null);
}

test "unrelated aot metadata difference is rejected" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;
    const dir = try makeTestDir(allocator, io, "meta-diff");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const arch = "aarch64";
    const sequence = cancelPollSequence(arch).?;
    const body = try testEntryBody(allocator, arch);
    const on_body = try std.mem.concat(allocator, u8, &.{
        cancelPollEntryPrefixSuffix(arch).?,
        sequence,
        testReturnInstruction(arch),
    });
    const off = try std.fs.path.join(allocator, &.{ dir, "off.cwasm" });
    const on = try std.fs.path.join(allocator, &.{ dir, "on.cwasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = off, .data = try testBuildAot(allocator, &.{body}, 0) });
    const enabled = try testBuildAot(allocator, &.{on_body}, 0);
    enabled[16] = 1;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = on, .data = enabled });

    const off_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, "--benchmark-disable-cancel-points", FIXTURE, "-o", off };
    const on_cmd = [_][]const u8{ "wamrc", "compile", "--target", arch, FIXTURE, "-o", on };
    const norm_dir = try std.fs.path.join(allocator, &.{ dir, "norm" });
    try testing.expectError(error.InvalidHarness, artifactIdentity(allocator, io, off, on, &off_cmd, &on_cmd, arch, norm_dir));
    try testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "section 0") != null);
}

test "schema requires retained evidence" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = testing.io;

    const schema_bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        "tests/benchmarks/leaf-cancel-cost/report.schema.json",
        allocator,
        .limited(1024 * 1024),
    );
    const parsed = try std.json.parseFromSlice(Value, allocator, schema_bytes, .{});
    const schema = parsed.value.object;
    try testing.expectEqual(@as(i64, 1), schema.get("properties").?.object.get("schema_version").?.object.get("const").?.integer);

    const defs = schema.get("$defs").?.object;
    const plan_req = defs.get("plan").?.object.get("required").?.array.items;
    var found_non_leaf = false;
    for (plan_req) |r| {
        if (std.mem.eql(u8, r.string, "non_leaf_poll_opportunities_upper_bound_per_enabled_sample")) found_non_leaf = true;
    }
    try testing.expect(found_non_leaf);
}
