const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const json = std.json;
const Sha256 = std.crypto.hash.sha2.Sha256;
const aot_jit_attr = @import("aot_jit_attr.zig");

pub const SCHEMA_VERSION: i64 = 1;
pub const MIN_MEASURED_RUNS: usize = 5;

pub const ObjectMap = json.ObjectMap;
pub const Array = json.Array;
pub const Value = json.Value;

pub const HarnessError = error{
    HarnessFailed,
    OutOfMemory,
};

pub const Diagnostic = struct {
    message: []const u8 = "",

    pub fn fail(
        self: *Diagnostic,
        allocator: Allocator,
        comptime fmt: []const u8,
        args: anytype,
    ) HarnessError {
        self.message = try std.fmt.allocPrint(allocator, fmt, args);
        return error.HarnessFailed;
    }
};

pub const Tool = struct {
    name: []const u8,
    path: []const u8,
    sha256: []const u8,
    version: []const u8,
};

pub const Mount = struct {
    host: []const u8,
    guest: []const u8,
    sha256_tree: []const u8,
};

pub const Source = struct {
    name: []const u8,
    path: []const u8,
    revision: []const u8,
};

pub const PerfConfig = struct {
    core_index: i64,
    hot_func: ?i64,
    min_samples: i64,
    min_attribution_coverage_pct: f64,
    base: ?[]const u8,
};

pub const EnvEntry = struct {
    key: []const u8,
    value: []const u8,
};

pub const Config = struct {
    manifest_path: []const u8,
    tools: std.StringArrayHashMapUnmanaged(Tool),
    sources: []Source,
    component: []const u8,
    component_sha256: []const u8,
    preopens_file: []const u8,
    preopens_sha256: []const u8,
    mounts: []Mount,
    package_name: []const u8,
    guest_env: []EnvEntry,
    expected_response_bytes: i64,
    perf: PerfConfig,
};

pub const CompletedProcess = struct {
    returncode: u8 = 0,
    stdout: []const u8 = "",
    stderr: []const u8 = "",
};

pub const Hooks = struct {
    ctx: ?*anyopaque = null,
    system_name: ?[]const u8 = null,
    machine_name: ?[]const u8 = null,
    paranoid_override: ??i64 = null,
    which_override: ?[]const u8 = null,
    access_override: ?bool = null,
    command_version_fn: ?*const fn (
        ctx: ?*anyopaque,
        allocator: Allocator,
        io: Io,
        tool: Tool,
        diag: *Diagnostic,
    ) HarnessError![]const u8 = null,
    run_checked_fn: ?*const fn (
        ctx: ?*anyopaque,
        allocator: Allocator,
        io: Io,
        command: []const []const u8,
        timeout_seconds: f64,
        label: []const u8,
        extra_env: ?[]const EnvEntry,
        diag: *Diagnostic,
    ) HarnessError!CompletedProcess = null,
};

fn isLowerHex(text: []const u8, expected_len: usize) bool {
    if (text.len != expected_len) return false;
    for (text) |ch| {
        const ok = (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f');
        if (!ok) return false;
    }
    return true;
}

fn isHexAddress(text: []const u8) bool {
    if (text.len < 3) return false;
    if (text[0] != '0' or text[1] != 'x') return false;
    for (text[2..]) |ch| {
        if (!std.ascii.isHex(ch)) return false;
    }
    return true;
}

fn asciiLowerDup(allocator: Allocator, text: []const u8) ![]u8 {
    const out = try allocator.dupe(u8, text);
    for (out) |*ch| ch.* = std.ascii.toLower(ch.*);
    return out;
}

fn requireObject(
    allocator: Allocator,
    value: ?Value,
    label: []const u8,
    diag: *Diagnostic,
) HarnessError!ObjectMap {
    const v = value orelse return diag.fail(allocator, "{s} must be a JSON object", .{label});
    if (v != .object) return diag.fail(allocator, "{s} must be a JSON object", .{label});
    return v.object;
}

fn requireList(
    allocator: Allocator,
    value: ?Value,
    label: []const u8,
    diag: *Diagnostic,
) HarnessError![]const Value {
    const v = value orelse return diag.fail(allocator, "{s} must be a non-empty JSON array", .{label});
    if (v != .array or v.array.items.len == 0) {
        return diag.fail(allocator, "{s} must be a non-empty JSON array", .{label});
    }
    return v.array.items;
}

fn requireString(
    allocator: Allocator,
    obj: ObjectMap,
    key: []const u8,
    label: []const u8,
    diag: *Diagnostic,
) HarnessError![]const u8 {
    const v = obj.get(key) orelse return diag.fail(allocator, "{s}.{s} must be a non-empty string", .{ label, key });
    if (v != .string or v.string.len == 0) {
        return diag.fail(allocator, "{s}.{s} must be a non-empty string", .{ label, key });
    }
    return try allocator.dupe(u8, v.string);
}

fn requireSha(
    allocator: Allocator,
    value: []const u8,
    label: []const u8,
    diag: *Diagnostic,
) HarnessError![]const u8 {
    const lower = try asciiLowerDup(allocator, value);
    if (!isLowerHex(lower, 64) or std.mem.allEqual(u8, lower, '0')) {
        return diag.fail(allocator, "{s} must be a non-zero, lowercase SHA-256", .{label});
    }
    return lower;
}

fn getEnvOwned(allocator: Allocator, key: [:0]const u8) ?[]const u8 {
    const ptr = std.c.getenv(key.ptr) orelse return null;
    return allocator.dupe(u8, std.mem.span(ptr)) catch null;
}

pub fn resolvePath(allocator: Allocator, io: Io, base: []const u8, value: []const u8) ![]const u8 {
    var expanded: []const u8 = value;
    if (std.mem.startsWith(u8, value, "~/") or std.mem.eql(u8, value, "~")) {
        if (getEnvOwned(allocator, "HOME")) |home| {
            expanded = if (value.len == 1)
                home
            else
                try std.fs.path.join(allocator, &.{ home, value[2..] });
        }
    }
    const joined = if (std.fs.path.isAbsolute(expanded))
        try std.fs.path.resolve(allocator, &.{expanded})
    else
        try std.fs.path.resolve(allocator, &.{ base, expanded });

    if (Io.Dir.cwd().realPathFileAlloc(io, joined, allocator)) |real| {
        return real;
    } else |_| {
        if (std.fs.path.dirname(joined)) |parent| {
            const base_name = std.fs.path.basename(joined);
            if (Io.Dir.cwd().realPathFileAlloc(io, parent, allocator)) |real_parent| {
                return std.fs.path.join(allocator, &.{ real_parent, base_name });
            } else |_| {}
        }
        return joined;
    }
}

pub fn loadManifest(
    allocator: Allocator,
    io: Io,
    path: []const u8,
    diag: *Diagnostic,
) HarnessError!Config {
    const raw_bytes = Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024)) catch {
        return diag.fail(allocator, "manifest not found: {s}", .{path});
    };
    const parsed = json.parseFromSliceLeaky(Value, allocator, raw_bytes, .{}) catch |err| {
        return diag.fail(allocator, "invalid JSON in {s}: {s}", .{ path, @errorName(err) });
    };
    const root = try requireObject(allocator, parsed, "manifest", diag);
    const schema_val = root.get("schema_version") orelse {
        return diag.fail(allocator, "schema_version must be {d}, got null", .{SCHEMA_VERSION});
    };
    if (schema_val != .integer or schema_val.integer != SCHEMA_VERSION) {
        return diag.fail(allocator, "schema_version must be {d}", .{SCHEMA_VERSION});
    }

    const cwd_real = Io.Dir.cwd().realPathFileAlloc(io, ".", allocator) catch ".";
    const resolved_manifest = try resolvePath(allocator, io, cwd_real, path);
    const base = std.fs.path.dirname(resolved_manifest) orelse ".";

    const tools_raw = try requireObject(allocator, root.get("tools"), "tools", diag);
    var tools: std.StringArrayHashMapUnmanaged(Tool) = .empty;
    for ([_][]const u8{ "wamr", "wamrc", "wasmtime" }) |tool_name| {
        const label = try std.fmt.allocPrint(allocator, "tools.{s}", .{tool_name});
        const sha_label = try std.fmt.allocPrint(allocator, "tools.{s}.sha256", .{tool_name});
        const item = try requireObject(allocator, tools_raw.get(tool_name), label, diag);
        const tool_path = try resolvePath(
            allocator,
            io,
            base,
            try requireString(allocator, item, "path", label, diag),
        );
        const tool_sha = try requireSha(
            allocator,
            try requireString(allocator, item, "sha256", label, diag),
            sha_label,
            diag,
        );
        const tool_ver = try requireString(allocator, item, "version", label, diag);
        const name_copy = try allocator.dupe(u8, tool_name);
        try tools.put(allocator, name_copy, .{
            .name = name_copy,
            .path = tool_path,
            .sha256 = tool_sha,
            .version = tool_ver,
        });
    }

    const sources_raw = try requireList(allocator, root.get("sources"), "sources", diag);
    var sources: std.ArrayList(Source) = .empty;
    for (sources_raw, 0..) |src_val, index| {
        const label = try std.fmt.allocPrint(allocator, "sources[{d}]", .{index});
        const item = try requireObject(allocator, src_val, label, diag);
        const revision = try asciiLowerDup(
            allocator,
            try requireString(allocator, item, "revision", label, diag),
        );
        if (!isLowerHex(revision, 40)) {
            return diag.fail(
                allocator,
                "{s}.revision must be a full 40-character commit SHA",
                .{label},
            );
        }
        try sources.append(allocator, .{
            .name = try requireString(allocator, item, "name", label, diag),
            .path = try resolvePath(
                allocator,
                io,
                base,
                try requireString(allocator, item, "path", label, diag),
            ),
            .revision = revision,
        });
    }

    const workload = try requireObject(allocator, root.get("workload"), "workload", diag);
    const component = try requireObject(allocator, workload.get("component"), "workload.component", diag);
    const preopens = try requireObject(allocator, workload.get("preopens_file"), "workload.preopens_file", diag);
    const mounts_raw = try requireList(allocator, workload.get("mounts"), "workload.mounts", diag);

    var mounts: std.ArrayList(Mount) = .empty;
    var guests: std.StringHashMapUnmanaged(void) = .empty;
    for (mounts_raw, 0..) |m_val, index| {
        const label = try std.fmt.allocPrint(allocator, "workload.mounts[{d}]", .{index});
        const sha_label = try std.fmt.allocPrint(allocator, "workload.mounts[{d}].sha256_tree", .{index});
        const item = try requireObject(allocator, m_val, label, diag);
        const guest = try requireString(allocator, item, "guest", label, diag);
        if (!std.mem.startsWith(u8, guest, "/") or guests.contains(guest)) {
            return diag.fail(
                allocator,
                "{s}.guest must be a unique absolute guest path",
                .{label},
            );
        }
        try guests.put(allocator, guest, {});
        try mounts.append(allocator, .{
            .host = try resolvePath(
                allocator,
                io,
                base,
                try requireString(allocator, item, "path", label, diag),
            ),
            .guest = guest,
            .sha256_tree = try requireSha(
                allocator,
                try requireString(allocator, item, "sha256_tree", label, diag),
                sha_label,
                diag,
            ),
        });
    }
    if (!guests.contains("/spec")) {
        return diag.fail(allocator, "workload.mounts must include the input spec tree at /spec", .{});
    }

    const expected_resp_val = workload.get("expected_tcgc_response_bytes") orelse {
        return diag.fail(allocator, "workload.expected_tcgc_response_bytes must be a positive integer", .{});
    };
    if (expected_resp_val != .integer or expected_resp_val.integer <= 0) {
        return diag.fail(allocator, "workload.expected_tcgc_response_bytes must be a positive integer", .{});
    }

    const guest_env_raw = try requireObject(allocator, workload.get("env"), "workload.env", diag);
    var guest_env: std.ArrayList(EnvEntry) = .empty;
    var env_it = guest_env_raw.iterator();
    while (env_it.next()) |entry| {
        const key = entry.key_ptr.*;
        const val = entry.value_ptr.*;
        if (key.len == 0 or std.mem.indexOfScalar(u8, key, '=') != null or val != .string) {
            return diag.fail(
                allocator,
                "workload.env must map non-empty names without '=' to string values",
                .{},
            );
        }
        if (std.mem.eql(u8, key, "WAMR_KEYVAULT_HARNESS")) {
            return diag.fail(allocator, "workload.env reserves WAMR_KEYVAULT_HARNESS", .{});
        }
        try guest_env.append(allocator, .{
            .key = try allocator.dupe(u8, key),
            .value = try allocator.dupe(u8, val.string),
        });
    }

    const perf = try requireObject(allocator, root.get("perf"), "perf", diag);
    const core_index_val = perf.get("core_index") orelse {
        return diag.fail(allocator, "perf.core_index must be a non-negative integer", .{});
    };
    if (core_index_val != .integer or core_index_val.integer < 0) {
        return diag.fail(allocator, "perf.core_index must be a non-negative integer", .{});
    }
    var hot_func: ?i64 = null;
    if (perf.get("hot_func")) |hf| {
        switch (hf) {
            .null => hot_func = null,
            .integer => |iv| {
                if (iv < 0) return diag.fail(allocator, "perf.hot_func must be null or a non-negative integer", .{});
                hot_func = iv;
            },
            else => return diag.fail(allocator, "perf.hot_func must be null or a non-negative integer", .{}),
        }
    }
    const min_samples_val = perf.get("min_samples") orelse {
        return diag.fail(allocator, "perf.min_samples must be an integer >= 1000", .{});
    };
    if (min_samples_val != .integer or min_samples_val.integer < 1000) {
        return diag.fail(allocator, "perf.min_samples must be an integer >= 1000", .{});
    }
    const min_cov_val = perf.get("min_attribution_coverage_pct") orelse {
        return diag.fail(allocator, "perf.min_attribution_coverage_pct must be in the range (0, 100]", .{});
    };
    const min_cov: f64 = switch (min_cov_val) {
        .integer => |iv| @floatFromInt(iv),
        .float => |fv| fv,
        else => return diag.fail(allocator, "perf.min_attribution_coverage_pct must be in the range (0, 100]", .{}),
    };
    if (!(min_cov > 0.0 and min_cov <= 100.0)) {
        return diag.fail(allocator, "perf.min_attribution_coverage_pct must be in the range (0, 100]", .{});
    }
    var perf_base: ?[]const u8 = null;
    if (perf.get("base")) |bv| {
        switch (bv) {
            .null => perf_base = null,
            .string => |sv| {
                if (!isHexAddress(sv)) {
                    return diag.fail(allocator, "perf.base must be null or a hexadecimal address", .{});
                }
                perf_base = try allocator.dupe(u8, sv);
            },
            else => return diag.fail(allocator, "perf.base must be null or a hexadecimal address", .{}),
        }
    }

    return .{
        .manifest_path = resolved_manifest,
        .tools = tools,
        .sources = try sources.toOwnedSlice(allocator),
        .component = try resolvePath(
            allocator,
            io,
            base,
            try requireString(allocator, component, "path", "workload.component", diag),
        ),
        .component_sha256 = try requireSha(
            allocator,
            try requireString(allocator, component, "sha256", "workload.component", diag),
            "workload.component.sha256",
            diag,
        ),
        .preopens_file = try resolvePath(
            allocator,
            io,
            base,
            try requireString(allocator, preopens, "path", "workload.preopens_file", diag),
        ),
        .preopens_sha256 = try requireSha(
            allocator,
            try requireString(allocator, preopens, "sha256", "workload.preopens_file", diag),
            "workload.preopens_file.sha256",
            diag,
        ),
        .mounts = try mounts.toOwnedSlice(allocator),
        .package_name = try requireString(allocator, workload, "package_name", "workload", diag),
        .guest_env = try guest_env.toOwnedSlice(allocator),
        .expected_response_bytes = expected_resp_val.integer,
        .perf = .{
            .core_index = core_index_val.integer,
            .hot_func = hot_func,
            .min_samples = min_samples_val.integer,
            .min_attribution_coverage_pct = min_cov,
            .base = perf_base,
        },
    };
}

pub fn sha256BytesHex(allocator: Allocator, bytes: []const u8) ![]const u8 {
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

pub fn sha256File(
    allocator: Allocator,
    io: Io,
    path: []const u8,
    diag: *Diagnostic,
) HarnessError![]const u8 {
    const data = Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(512 * 1024 * 1024)) catch {
        return diag.fail(allocator, "failed to read file for SHA-256: {s}", .{path});
    };
    return sha256BytesHex(allocator, data);
}

pub const TreeFileEntry = struct {
    path: []const u8,
    size: u64,
    sha256: []const u8,
};

pub const TreeSnapshot = struct {
    sha256: []const u8,
    file_count: usize,
    total_bytes: u64,
    files: []TreeFileEntry,

    pub fn toJsonValue(self: TreeSnapshot, allocator: Allocator) !Value {
        var files_arr = Array.init(allocator);
        for (self.files) |f| {
            var f_obj: ObjectMap = .empty;
            try f_obj.put(allocator, "path", .{ .string = f.path });
            try f_obj.put(allocator, "sha256", .{ .string = f.sha256 });
            try f_obj.put(allocator, "size", .{ .integer = @intCast(f.size) });
            try files_arr.append(.{ .object = f_obj });
        }
        var root: ObjectMap = .empty;
        try root.put(allocator, "file_count", .{ .integer = @intCast(self.file_count) });
        try root.put(allocator, "files", .{ .array = files_arr });
        try root.put(allocator, "sha256", .{ .string = self.sha256 });
        try root.put(allocator, "total_bytes", .{ .integer = @intCast(self.total_bytes) });
        return .{ .object = root };
    }

    pub fn toOutputJsonValue(self: TreeSnapshot, allocator: Allocator) !Value {
        var files_arr = Array.init(allocator);
        for (self.files) |f| {
            var f_obj: ObjectMap = .empty;
            try f_obj.put(allocator, "path", .{ .string = f.path });
            try f_obj.put(allocator, "sha256", .{ .string = f.sha256 });
            try f_obj.put(allocator, "size", .{ .integer = @intCast(f.size) });
            try files_arr.append(.{ .object = f_obj });
        }
        var root: ObjectMap = .empty;
        try root.put(allocator, "file_count", .{ .integer = @intCast(self.file_count) });
        try root.put(allocator, "files", .{ .array = files_arr });
        try root.put(allocator, "sha256_tree", .{ .string = self.sha256 });
        try root.put(allocator, "total_bytes", .{ .integer = @intCast(self.total_bytes) });
        return .{ .object = root };
    }
};

pub fn treeSnapshot(
    allocator: Allocator,
    io: Io,
    path: []const u8,
    diag: *Diagnostic,
) HarnessError!TreeSnapshot {
    var dir = Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch {
        return diag.fail(allocator, "tree input is not a directory: {s}", .{path});
    };
    defer dir.close(io);

    var entries: std.ArrayList(TreeFileEntry) = .empty;
    var walker = dir.walk(allocator) catch return error.OutOfMemory;
    defer walker.deinit();

    while (walker.next(io) catch |err| {
        return diag.fail(allocator, "failed walking directory {s}: {s}", .{ path, @errorName(err) });
    }) |entry| {
        const rel = try allocator.dupe(u8, entry.path);
        for (rel) |*ch| {
            if (ch.* == '\\') ch.* = '/';
        }
        const full_path = try std.fs.path.join(allocator, &.{ path, entry.path });
        switch (entry.kind) {
            .sym_link => return diag.fail(allocator, "symlinks are not allowed in hashed trees: {s}", .{full_path}),
            .directory => continue,
            .file => {
                const data = dir.readFileAlloc(io, entry.path, allocator, .limited(512 * 1024 * 1024)) catch {
                    return diag.fail(allocator, "failed reading tree file: {s}", .{full_path});
                };
                try entries.append(allocator, .{
                    .path = rel,
                    .size = data.len,
                    .sha256 = try sha256BytesHex(allocator, data),
                });
            },
            else => return diag.fail(allocator, "unsupported non-file tree entry: {s}", .{full_path}),
        }
    }

    std.mem.sort(TreeFileEntry, entries.items, {}, struct {
        fn lessThan(_: void, a: TreeFileEntry, b: TreeFileEntry) bool {
            return std.mem.order(u8, a.path, b.path) == .lt;
        }
    }.lessThan);

    var canonical_arr = Array.init(allocator);
    var total_bytes: u64 = 0;
    for (entries.items) |item| {
        total_bytes += item.size;
        var item_obj: ObjectMap = .empty;
        try item_obj.put(allocator, "path", .{ .string = item.path });
        try item_obj.put(allocator, "sha256", .{ .string = item.sha256 });
        try item_obj.put(allocator, "size", .{ .integer = @intCast(item.size) });
        try canonical_arr.append(.{ .object = item_obj });
    }
    var canonical_buf: Io.Writer.Allocating = .init(allocator);
    writeCanonicalJson(&canonical_buf.writer, .{ .array = canonical_arr }) catch return error.OutOfMemory;
    const canonical_bytes = try canonical_buf.toOwnedSlice();

    return .{
        .sha256 = try sha256BytesHex(allocator, canonical_bytes),
        .file_count = entries.items.len,
        .total_bytes = total_bytes,
        .files = try entries.toOwnedSlice(allocator),
    };
}

pub const PreopenMapping = struct {
    host: []const u8,
    guest: []const u8,
};

pub fn parsePreopens(
    allocator: Allocator,
    io: Io,
    path: []const u8,
    diag: *Diagnostic,
) HarnessError![]PreopenMapping {
    const content = Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(4 * 1024 * 1024)) catch {
        return diag.fail(allocator, "required preopens file is absent: {s}", .{path});
    };
    const parent = std.fs.path.dirname(path) orelse ".";
    var result: std.ArrayList(PreopenMapping) = .empty;
    var line_it = std.mem.splitScalar(u8, content, '\n');
    var line_number: usize = 0;
    while (line_it.next()) |raw_line| {
        line_number += 1;
        const line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (line.len == 0) continue;
        const eq_idx = std.mem.indexOfScalar(u8, line, '=') orelse {
            return diag.fail(allocator, "{s}:{d}: expected HOST=GUEST preopen mapping", .{ path, line_number });
        };
        const host = line[0..eq_idx];
        const guest = line[eq_idx + 1 ..];
        if (host.len == 0 or !std.mem.startsWith(u8, guest, "/")) {
            return diag.fail(
                allocator,
                "{s}:{d}: expected non-empty HOST and absolute GUEST",
                .{ path, line_number },
            );
        }
        const resolved_host = try resolvePath(allocator, io, parent, host);
        try result.append(allocator, .{
            .host = resolved_host,
            .guest = try allocator.dupe(u8, guest),
        });
    }
    if (result.items.len == 0) {
        return diag.fail(allocator, "preopens file has no mappings: {s}", .{path});
    }
    return result.toOwnedSlice(allocator);
}

pub fn controlledHostEnv(allocator: Allocator, extra_env: ?[]const EnvEntry) !std.process.Environ.Map {
    var env_map = std.process.Environ.Map.init(allocator);
    try env_map.put("PATH", getEnvOwned(allocator, "PATH") orelse "/usr/bin:/bin");
    try env_map.put("LANG", "C");
    try env_map.put("LC_ALL", "C");
    try env_map.put("TZ", "UTC");
    if (getEnvOwned(allocator, "HOME")) |home| {
        try env_map.put("HOME", home);
    }
    if (extra_env) |entries| {
        for (entries) |entry| {
            try env_map.put(entry.key, entry.value);
        }
    }
    return env_map;
}

pub fn defaultRunChecked(
    allocator: Allocator,
    io: Io,
    command: []const []const u8,
    timeout_seconds: f64,
    label: []const u8,
    extra_env: ?[]const EnvEntry,
    diag: *Diagnostic,
) HarnessError!CompletedProcess {
    _ = timeout_seconds;
    if (command.len > 0 and std.mem.eql(u8, command[0], "aot-jit-attr")) {
        var stdout_buf: Io.Writer.Allocating = .init(allocator);
        aot_jit_attr.runCli(
            allocator,
            io,
            command[1..],
            &stdout_buf.writer,
        ) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                const stdout = try stdout_buf.toOwnedSlice();
                const stderr = try std.fmt.allocPrint(allocator, "error: {s}\n", .{aot_jit_attr.lastError()});
                const joined = try std.mem.join(allocator, " ", command);
                return diag.fail(
                    allocator,
                    "{s} failed with exit 1\ncommand: {s}\nstdout:\n{s}\nstderr:\n{s}",
                    .{ label, joined, stdout, stderr },
                );
            },
        };
        const stdout = try stdout_buf.toOwnedSlice();
        return .{ .returncode = 0, .stdout = stdout, .stderr = "" };
    }

    var env_map = try controlledHostEnv(allocator, extra_env);
    const res = std.process.run(allocator, io, .{
        .argv = command,
        .environ_map = &env_map,
        .stdout_limit = .limited(32 * 1024 * 1024),
        .stderr_limit = .limited(32 * 1024 * 1024),
    }) catch |err| {
        const joined = try std.mem.join(allocator, " ", command);
        return diag.fail(
            allocator,
            "{s} failed to spawn ({s}): {s}",
            .{ label, @errorName(err), joined },
        );
    };
    const exit_code: u8 = switch (res.term) {
        .exited => |c| c,
        else => 255,
    };
    if (exit_code != 0) {
        const joined = try std.mem.join(allocator, " ", command);
        return diag.fail(
            allocator,
            "{s} failed with exit {d}\ncommand: {s}\nstdout:\n{s}\nstderr:\n{s}",
            .{ label, exit_code, joined, res.stdout, res.stderr },
        );
    }
    return .{
        .returncode = exit_code,
        .stdout = res.stdout,
        .stderr = res.stderr,
    };
}

pub fn runChecked(
    allocator: Allocator,
    io: Io,
    command: []const []const u8,
    timeout_seconds: f64,
    label: []const u8,
    extra_env: ?[]const EnvEntry,
    hooks: Hooks,
    diag: *Diagnostic,
) HarnessError!CompletedProcess {
    if (hooks.run_checked_fn) |func| {
        return func(hooks.ctx, allocator, io, command, timeout_seconds, label, extra_env, diag);
    }
    return defaultRunChecked(allocator, io, command, timeout_seconds, label, extra_env, diag);
}

pub fn commandVersion(
    allocator: Allocator,
    io: Io,
    tool: Tool,
    hooks: Hooks,
    diag: *Diagnostic,
) HarnessError![]const u8 {
    if (hooks.command_version_fn) |func| {
        return func(hooks.ctx, allocator, io, tool, diag);
    }
    const version_arg = if (std.mem.eql(u8, tool.name, "wasmtime")) "--version" else "version";
    var env_map = try controlledHostEnv(allocator, null);
    const res = std.process.run(allocator, io, .{
        .argv = &.{ tool.path, version_arg },
        .environ_map = &env_map,
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch |err| {
        return diag.fail(allocator, "{s} --version failed: {s}", .{ tool.name, @errorName(err) });
    };
    const combined = try std.fmt.allocPrint(allocator, "{s}{s}", .{ res.stdout, res.stderr });
    const output = std.mem.trim(u8, combined, " \t\r\n");
    const code: u8 = switch (res.term) {
        .exited => |c| c,
        else => 255,
    };
    if (code != 0) {
        return diag.fail(allocator, "{s} --version failed with exit {d}: {s}", .{ tool.name, code, output });
    }
    if (std.mem.indexOf(u8, output, tool.version) == null) {
        return diag.fail(
            allocator,
            "{s} version mismatch: expected output containing '{s}', got '{s}'",
            .{ tool.name, tool.version, output },
        );
    }
    return try allocator.dupe(u8, output);
}

fn isExecutableFile(io: Io, path: []const u8, hooks: Hooks) bool {
    if (hooks.access_override) |override| return override;
    const st = Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    if (st.kind != .file) return false;
    if (builtin.os.tag == .windows) return true;
    return (st.permissions.toMode() & 0o111) != 0;
}

fn isRegularFile(io: Io, path: []const u8) bool {
    const st = Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return st.kind == .file;
}

fn isDirectory(io: Io, path: []const u8) bool {
    var dir = Io.Dir.cwd().openDir(io, path, .{}) catch return false;
    dir.close(io);
    return true;
}

pub fn validateConfig(
    allocator: Allocator,
    io: Io,
    config: Config,
    hooks: Hooks,
    diag: *Diagnostic,
) HarnessError!Value {
    var tool_report: ObjectMap = .empty;
    var tool_it = config.tools.iterator();
    while (tool_it.next()) |entry| {
        const tool = entry.value_ptr.*;
        if (!isRegularFile(io, tool.path)) {
            return diag.fail(
                allocator,
                "required {s} binary is absent: {s}; install/build the pinned version and update the manifest path",
                .{ tool.name, tool.path },
            );
        }
        if (!isExecutableFile(io, tool.path, hooks)) {
            return diag.fail(allocator, "required {s} binary is not executable: {s}", .{ tool.name, tool.path });
        }
        const actual = try sha256File(allocator, io, tool.path, diag);
        if (!std.mem.eql(u8, actual, tool.sha256)) {
            return diag.fail(
                allocator,
                "{s} SHA-256 mismatch for {s}: expected {s}, got {s}",
                .{ tool.name, tool.path, tool.sha256, actual },
            );
        }
        const ver = try commandVersion(allocator, io, tool, hooks, diag);
        var t_obj: ObjectMap = .empty;
        try t_obj.put(allocator, "path", .{ .string = tool.path });
        try t_obj.put(allocator, "sha256", .{ .string = actual });
        try t_obj.put(allocator, "version", .{ .string = ver });
        try tool_report.put(allocator, tool.name, .{ .object = t_obj });
    }

    var source_report = Array.init(allocator);
    for (config.sources) |source| {
        const dot_git = try std.fs.path.join(allocator, &.{ source.path, ".git" });
        const has_dot_git = isDirectory(io, dot_git) or isRegularFile(io, dot_git);
        if (!has_dot_git) {
            const check = std.process.run(allocator, io, .{
                .argv = &.{ "git", "-C", source.path, "rev-parse", "--git-dir" },
            }) catch null;
            const ok = if (check) |res| (switch (res.term) {
                .exited => |c| c == 0,
                else => false,
            }) else false;
            if (!ok) {
                return diag.fail(allocator, "source checkout is absent or not a git tree: {s}", .{source.path});
            }
        }
        const rev_res = std.process.run(allocator, io, .{
            .argv = &.{ "git", "-C", source.path, "rev-parse", "HEAD" },
        }) catch {
            return diag.fail(allocator, "source checkout is absent or not a git tree: {s}", .{source.path});
        };
        const rev_ok = switch (rev_res.term) {
            .exited => |c| c == 0,
            else => false,
        };
        const actual_rev = try asciiLowerDup(allocator, std.mem.trim(u8, rev_res.stdout, " \t\r\n"));
        if (!rev_ok or !std.mem.eql(u8, actual_rev, source.revision)) {
            const detail = if (actual_rev.len > 0) actual_rev else std.mem.trim(u8, rev_res.stderr, " \t\r\n");
            return diag.fail(
                allocator,
                "source revision mismatch for {s} at {s}: expected {s}, got {s}",
                .{ source.name, source.path, source.revision, detail },
            );
        }

        const dirty_res = std.process.run(allocator, io, .{
            .argv = &.{ "git", "-C", source.path, "status", "--porcelain", "--untracked-files=no" },
        }) catch {
            return diag.fail(allocator, "source checkout has tracked modifications: {s} at {s}", .{ source.name, source.path });
        };
        const dirty_ok = switch (dirty_res.term) {
            .exited => |c| c == 0,
            else => false,
        };
        if (!dirty_ok or std.mem.trim(u8, dirty_res.stdout, " \t\r\n").len != 0) {
            return diag.fail(
                allocator,
                "source checkout has tracked modifications: {s} at {s}",
                .{ source.name, source.path },
            );
        }

        var s_obj: ObjectMap = .empty;
        try s_obj.put(allocator, "name", .{ .string = source.name });
        try s_obj.put(allocator, "path", .{ .string = source.path });
        try s_obj.put(allocator, "revision", .{ .string = actual_rev });
        try source_report.append(.{ .object = s_obj });
    }

    const FileCheck = struct { path: []const u8, expected: []const u8, label: []const u8 };
    for ([_]FileCheck{
        .{ .path = config.component, .expected = config.component_sha256, .label = "component" },
        .{ .path = config.preopens_file, .expected = config.preopens_sha256, .label = "preopens file" },
    }) |fc| {
        if (!isRegularFile(io, fc.path)) {
            return diag.fail(allocator, "required {s} is absent: {s}", .{ fc.label, fc.path });
        }
        const actual = try sha256File(allocator, io, fc.path, diag);
        if (!std.mem.eql(u8, actual, fc.expected)) {
            return diag.fail(
                allocator,
                "{s} SHA-256 mismatch for {s}: expected {s}, got {s}",
                .{ fc.label, fc.path, fc.expected, actual },
            );
        }
    }

    const file_mappings = try parsePreopens(allocator, io, config.preopens_file, diag);
    var configured_stdlib: std.StringHashMapUnmanaged(void) = .empty;
    for (config.mounts) |mount| {
        if (std.mem.eql(u8, mount.guest, "/spec") or std.mem.eql(u8, mount.guest, "/out")) continue;
        const pair_key = try std.fmt.allocPrint(allocator, "{s}::{s}", .{ mount.host, mount.guest });
        try configured_stdlib.put(allocator, pair_key, {});
    }
    var file_set: std.StringHashMapUnmanaged(void) = .empty;
    for (file_mappings) |mapping| {
        const pair_key = try std.fmt.allocPrint(allocator, "{s}::{s}", .{ mapping.host, mapping.guest });
        try file_set.put(allocator, pair_key, {});
    }
    var has_diff = file_set.count() != configured_stdlib.count();
    if (!has_diff) {
        var it = file_set.keyIterator();
        while (it.next()) |k| {
            if (!configured_stdlib.contains(k.*)) {
                has_diff = true;
                break;
            }
        }
    }
    if (has_diff) {
        return diag.fail(allocator, "preopens file and workload.mounts differ", .{});
    }

    var mount_report = Array.init(allocator);
    for (config.mounts) |mount| {
        const snapshot = try treeSnapshot(allocator, io, mount.host, diag);
        if (!std.mem.eql(u8, snapshot.sha256, mount.sha256_tree)) {
            return diag.fail(
                allocator,
                "tree SHA-256 mismatch for mount {s}::{s}: expected {s}, got {s}",
                .{ mount.host, mount.guest, mount.sha256_tree, snapshot.sha256 },
            );
        }
        var m_obj: ObjectMap = .empty;
        try m_obj.put(allocator, "file_count", .{ .integer = @intCast(snapshot.file_count) });
        try m_obj.put(allocator, "guest", .{ .string = mount.guest });
        try m_obj.put(allocator, "path", .{ .string = mount.host });
        try m_obj.put(allocator, "sha256_tree", .{ .string = snapshot.sha256 });
        try m_obj.put(allocator, "total_bytes", .{ .integer = @intCast(snapshot.total_bytes) });
        try mount_report.append(.{ .object = m_obj });
    }

    const comp_stat = Io.Dir.cwd().statFile(io, config.component, .{}) catch {
        return diag.fail(allocator, "required component is absent: {s}", .{config.component});
    };
    var comp_obj: ObjectMap = .empty;
    try comp_obj.put(allocator, "path", .{ .string = config.component });
    try comp_obj.put(allocator, "sha256", .{ .string = config.component_sha256 });
    try comp_obj.put(allocator, "size", .{ .integer = @intCast(comp_stat.size) });

    var preopens_obj: ObjectMap = .empty;
    try preopens_obj.put(allocator, "path", .{ .string = config.preopens_file });
    try preopens_obj.put(allocator, "sha256", .{ .string = config.preopens_sha256 });

    var guest_env_obj: ObjectMap = .empty;
    for (config.guest_env) |entry| {
        try guest_env_obj.put(allocator, entry.key, .{ .string = entry.value });
    }

    var inputs_obj: ObjectMap = .empty;
    try inputs_obj.put(allocator, "component", .{ .object = comp_obj });
    try inputs_obj.put(allocator, "expected_tcgc_response_bytes", .{ .integer = config.expected_response_bytes });
    try inputs_obj.put(allocator, "guest_env", .{ .object = guest_env_obj });
    try inputs_obj.put(allocator, "mounts", .{ .array = mount_report });
    try inputs_obj.put(allocator, "package_name", .{ .string = config.package_name });
    try inputs_obj.put(allocator, "preopens_file", .{ .object = preopens_obj });

    var root: ObjectMap = .empty;
    try root.put(allocator, "inputs", .{ .object = inputs_obj });
    try root.put(allocator, "sources", .{ .array = source_report });
    try root.put(allocator, "tools", .{ .object = tool_report });
    return .{ .object = root };
}

pub const PrecompileCommands = struct {
    wamr: []const []const u8,
    wasmtime: []const []const u8,
    wamr_manifest: []const u8,
    wasmtime_cwasm: []const u8,
};

pub fn precompileCommands(
    allocator: Allocator,
    config: Config,
    artifact_dir: []const u8,
) !PrecompileCommands {
    const wamr_manifest = try std.fs.path.join(allocator, &.{ artifact_dir, "keyvault.cwasm.json" });
    const wasmtime_cwasm = try std.fs.path.join(allocator, &.{ artifact_dir, "keyvault.wasmtime.cwasm" });
    const wamrc_path = config.tools.get("wamrc").?.path;
    const wasmtime_path = config.tools.get("wasmtime").?.path;

    const wamr = try allocator.dupe([]const u8, &.{
        wamrc_path,
        "compile-component",
        "--target=x86_64",
        config.component,
        "-o",
        wamr_manifest,
    });
    const wasmtime = try allocator.dupe([]const u8, &.{
        wasmtime_path,
        "compile",
        "-W",
        "max-memory-size=4294967296",
        "-O",
        "opt-level=2",
        config.component,
        "-o",
        wasmtime_cwasm,
    });
    return .{
        .wamr = wamr,
        .wasmtime = wasmtime,
        .wamr_manifest = wamr_manifest,
        .wasmtime_cwasm = wasmtime_cwasm,
    };
}

pub fn runtimeCommand(
    allocator: Allocator,
    io: Io,
    config: Config,
    runtime: []const u8,
    output_dir: []const u8,
    wamr_manifest: []const u8,
    wasmtime_cwasm: []const u8,
    diag: *Diagnostic,
) HarnessError![]const []const u8 {
    var env_entries: std.ArrayList(EnvEntry) = .empty;
    try env_entries.append(allocator, .{ .key = "WAMR_KEYVAULT_HARNESS", .value = "1" });
    for (config.guest_env) |entry| {
        try env_entries.append(allocator, entry);
    }
    std.mem.sort(EnvEntry, env_entries.items, {}, struct {
        fn lessThan(_: void, a: EnvEntry, b: EnvEntry) bool {
            return std.mem.order(u8, a.key, b.key) == .lt;
        }
    }.lessThan);

    const cwd_real = Io.Dir.cwd().realPathFileAlloc(io, ".", allocator) catch ".";
    const resolved_out = try resolvePath(allocator, io, cwd_real, output_dir);
    var command: std.ArrayList([]const u8) = .empty;

    if (std.mem.eql(u8, runtime, "wamr")) {
        const wamr_path = config.tools.get("wamr").?.path;
        try command.appendSlice(allocator, &.{
            wamr_path,
            "run",
            "--precompiled-manifest",
            wamr_manifest,
        });
        for (env_entries.items) |entry| {
            try command.appendSlice(allocator, &.{
                "--env",
                try std.fmt.allocPrint(allocator, "{s}={s}", .{ entry.key, entry.value }),
            });
        }
        for (config.mounts) |mount| {
            try command.appendSlice(allocator, &.{
                "--map-dir",
                try std.fmt.allocPrint(allocator, "{s}::{s}", .{ mount.host, mount.guest }),
            });
        }
        try command.appendSlice(allocator, &.{
            "--map-dir",
            try std.fmt.allocPrint(allocator, "{s}::/out", .{resolved_out}),
            config.component,
            "/spec",
            "/out",
            "--package-name",
            config.package_name,
        });
        return command.toOwnedSlice(allocator);
    }
    if (std.mem.eql(u8, runtime, "wasmtime")) {
        const wasmtime_path = config.tools.get("wasmtime").?.path;
        try command.appendSlice(allocator, &.{
            wasmtime_path,
            "run",
            "--allow-precompiled",
            "-S",
            "http",
            "-W",
            "max-memory-size=4294967296",
        });
        for (env_entries.items) |entry| {
            try command.appendSlice(allocator, &.{
                "--env",
                try std.fmt.allocPrint(allocator, "{s}={s}", .{ entry.key, entry.value }),
            });
        }
        for (config.mounts) |mount| {
            try command.appendSlice(allocator, &.{
                "--dir",
                try std.fmt.allocPrint(allocator, "{s}::{s}", .{ mount.host, mount.guest }),
            });
        }
        try command.appendSlice(allocator, &.{
            "--dir",
            try std.fmt.allocPrint(allocator, "{s}::/out", .{resolved_out}),
            wasmtime_cwasm,
            "/spec",
            "/out",
            "--package-name",
            config.package_name,
        });
        return command.toOwnedSlice(allocator);
    }
    return diag.fail(allocator, "unknown runtime: {s}", .{runtime});
}

pub fn parseTcgcResponseBytes(
    allocator: Allocator,
    output: []const u8,
    diag: *Diagnostic,
) HarnessError!i64 {
    var matches: std.ArrayList(i64) = .empty;
    var index: usize = 0;
    while (std.mem.indexOfPos(u8, output, index, "got")) |got_pos| {
        index = got_pos + 3;
        var pos = index;
        if (pos >= output.len or !std.ascii.isWhitespace(output[pos])) continue;
        while (pos < output.len and std.ascii.isWhitespace(output[pos])) : (pos += 1) {}
        const num_start = pos;
        while (pos < output.len and std.ascii.isDigit(output[pos])) : (pos += 1) {}
        if (pos == num_start) continue;
        const num_text = output[num_start..pos];
        if (pos >= output.len or !std.ascii.isWhitespace(output[pos])) continue;

        var word_matches = true;
        for ([_][]const u8{ "bytes", "back", "from", "tcgc.compile" }) |expected_word| {
            while (pos < output.len and std.ascii.isWhitespace(output[pos])) : (pos += 1) {}
            if (!std.mem.startsWith(u8, output[pos..], expected_word)) {
                word_matches = false;
                break;
            }
            pos += expected_word.len;
        }
        if (!word_matches) continue;
        const val = std.fmt.parseInt(i64, num_text, 10) catch continue;
        try matches.append(allocator, val);
        index = pos;
    }

    if (matches.items.len != 1) {
        return diag.fail(
            allocator,
            "expected exactly one 'got N bytes back from tcgc.compile' marker, found {d}",
            .{matches.items.len},
        );
    }
    return matches.items[0];
}

pub fn parseTimingSamples(
    allocator: Allocator,
    records: []const Value,
    runtime: []const u8,
    diag: *Diagnostic,
) HarnessError![]f64 {
    var samples: std.ArrayList(f64) = .empty;
    for (records) |rec_val| {
        if (rec_val != .object) continue;
        const rec = rec_val.object;
        const rt = rec.get("runtime") orelse continue;
        const ph = rec.get("phase") orelse continue;
        if (rt != .string or ph != .string) continue;
        if (!std.mem.eql(u8, rt.string, runtime) or !std.mem.eql(u8, ph.string, "measure")) continue;
        const elapsed_val = rec.get("elapsed_seconds") orelse {
            return diag.fail(allocator, "{s} has an invalid measured timing sample", .{runtime});
        };
        const elapsed: f64 = switch (elapsed_val) {
            .integer => |iv| @floatFromInt(iv),
            .float => |fv| fv,
            else => return diag.fail(allocator, "{s} has an invalid measured timing sample", .{runtime}),
        };
        if (elapsed <= 0.0) {
            return diag.fail(allocator, "{s} has an invalid measured timing sample", .{runtime});
        }
        try samples.append(allocator, elapsed);
    }
    if (samples.items.len < MIN_MEASURED_RUNS) {
        return diag.fail(
            allocator,
            "{s} has {d} measured runs; at least {d} are required",
            .{ runtime, samples.items.len, MIN_MEASURED_RUNS },
        );
    }
    return samples.toOwnedSlice(allocator);
}

pub fn frameMetadataPath(
    allocator: Allocator,
    config: Config,
    artifact_dir: []const u8,
) ![]const u8 {
    const filename = if (config.perf.hot_func) |hf|
        try std.fmt.allocPrint(allocator, "keyvault.mod{d}.func{d}.json", .{ config.perf.core_index, hf })
    else
        try std.fmt.allocPrint(allocator, "keyvault.mod{d}.funcNone.json", .{config.perf.core_index});
    return std.fs.path.join(allocator, &.{ artifact_dir, "frame", filename });
}

pub fn validateFrameArtifact(
    allocator: Allocator,
    io: Io,
    config: Config,
    artifact_dir: []const u8,
    timeout_seconds: f64,
    hooks: Hooks,
    diag: *Diagnostic,
) HarnessError!Value {
    const sidecar = try frameMetadataPath(allocator, config, artifact_dir);
    const core_name = try std.fmt.allocPrint(allocator, "keyvault.{d}.cwasm", .{config.perf.core_index});
    const core = try std.fs.path.join(allocator, &.{ artifact_dir, core_name });
    if (!isRegularFile(io, core) or !isRegularFile(io, sidecar)) {
        return diag.fail(
            allocator,
            "frame attribution requires the exact core and sidecar: {s}, {s}",
            .{ core, sidecar },
        );
    }
    const func_str = if (config.perf.hot_func) |hf|
        try std.fmt.allocPrint(allocator, "{d}", .{hf})
    else
        "None";
    const command = try allocator.dupe([]const u8, &.{
        "aot-jit-attr",
        "--cwasm",
        core,
        "--func",
        func_str,
        "--frame-metadata",
        sidecar,
        "--validate-frame-metadata",
    });
    _ = try runChecked(
        allocator,
        io,
        command,
        timeout_seconds,
        "exact-image frame metadata validation",
        null,
        hooks,
        diag,
    );

    var cmd_arr = Array.init(allocator);
    for (command) |arg| try cmd_arr.append(.{ .string = arg });

    var obj: ObjectMap = .empty;
    try obj.put(allocator, "cwasm", .{ .string = core });
    try obj.put(allocator, "cwasm_sha256", .{ .string = try sha256File(allocator, io, core, diag) });
    try obj.put(allocator, "sidecar", .{ .string = sidecar });
    try obj.put(allocator, "sidecar_sha256", .{ .string = try sha256File(allocator, io, sidecar, diag) });
    try obj.put(allocator, "validation_command", .{ .array = cmd_arr });
    return .{ .object = obj };
}

fn cleanDirectoryContents(allocator: Allocator, io: Io, dir_path: []const u8) !void {
    var dir = Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return;
    defer dir.close(io);

    var names: std.ArrayList(struct { name: []const u8, is_dir: bool }) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        try names.append(allocator, .{
            .name = try allocator.dupe(u8, entry.name),
            .is_dir = entry.kind == .directory,
        });
    }
    for (names.items) |item| {
        if (item.is_dir) {
            dir.deleteTree(io, item.name) catch {};
        } else {
            dir.deleteFile(io, item.name) catch {};
        }
    }
}

pub const PrecompileResult = struct {
    wamr_manifest: []const u8,
    wasmtime_cwasm: []const u8,
    report: Value,
};

pub fn precompile(
    allocator: Allocator,
    io: Io,
    config: Config,
    artifact_dir: []const u8,
    timeout_seconds: f64,
    frame_attribution: bool,
    hooks: Hooks,
    diag: *Diagnostic,
) HarnessError!PrecompileResult {
    Io.Dir.cwd().createDirPath(io, artifact_dir) catch {
        return diag.fail(allocator, "failed to create artifact directory: {s}", .{artifact_dir});
    };
    const cmds = try precompileCommands(allocator, config, artifact_dir);
    cleanDirectoryContents(allocator, io, artifact_dir) catch {};

    if (frame_attribution and config.perf.hot_func == null) {
        return diag.fail(allocator, "--profile requires perf.hot_func for exact frame attribution", .{});
    }
    var frame_env: ?[]const EnvEntry = null;
    if (frame_attribution) {
        const frame_dir = try std.fs.path.join(allocator, &.{ artifact_dir, "frame" });
        Io.Dir.cwd().createDirPath(io, frame_dir) catch {
            return diag.fail(allocator, "failed to create frame directory: {s}", .{frame_dir});
        };
        const prefix = try std.fs.path.join(allocator, &.{ frame_dir, "keyvault" });
        frame_env = try allocator.dupe(EnvEntry, &.{
            .{ .key = "WAMR_AOT_FRAME_ATTRIBUTION", .value = prefix },
            .{
                .key = "WAMR_AOT_FRAME_ATTRIBUTION_MODULE",
                .value = try std.fmt.allocPrint(allocator, "{d}", .{config.perf.core_index}),
            },
            .{
                .key = "WAMR_AOT_FRAME_ATTRIBUTION_FUNC",
                .value = try std.fmt.allocPrint(allocator, "{d}", .{config.perf.hot_func.?}),
            },
        });
    }

    const wamr_start = Io.Timestamp.now(io, .awake);
    _ = try runChecked(
        allocator,
        io,
        cmds.wamr,
        timeout_seconds,
        "WAMR precompile",
        frame_env,
        hooks,
        diag,
    );
    const wamr_ns = wamr_start.untilNow(io, .awake).toNanoseconds();
    const wamr_seconds: f64 = @as(f64, @floatFromInt(@max(wamr_ns, 1))) / 1e9;

    const wasmtime_start = Io.Timestamp.now(io, .awake);
    _ = try runChecked(
        allocator,
        io,
        cmds.wasmtime,
        timeout_seconds,
        "Wasmtime precompile",
        null,
        hooks,
        diag,
    );
    const wasmtime_ns = wasmtime_start.untilNow(io, .awake).toNanoseconds();
    const wasmtime_seconds: f64 = @as(f64, @floatFromInt(@max(wasmtime_ns, 1))) / 1e9;

    if (!isRegularFile(io, cmds.wamr_manifest) or !isRegularFile(io, cmds.wasmtime_cwasm)) {
        return diag.fail(allocator, "precompile completed without producing both artifacts", .{});
    }

    const frame_artifact: Value = if (frame_attribution)
        try validateFrameArtifact(allocator, io, config, artifact_dir, timeout_seconds, hooks, diag)
    else
        .null;

    var artifact_names: std.ArrayList([]const u8) = .empty;
    if (Io.Dir.cwd().openDir(io, artifact_dir, .{ .iterate = true })) |opened| {
        var dir = opened;
        defer dir.close(io);
        var it = dir.iterate();
        while (it.next(io) catch null) |entry| {
            if (entry.kind == .file) {
                try artifact_names.append(allocator, try allocator.dupe(u8, entry.name));
            }
        }
    } else |_| {}

    std.mem.sort([]const u8, artifact_names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    var artifacts_arr = Array.init(allocator);
    for (artifact_names.items) |name| {
        const item_path = try std.fs.path.join(allocator, &.{ artifact_dir, name });
        const st = Io.Dir.cwd().statFile(io, item_path, .{}) catch continue;
        var a_obj: ObjectMap = .empty;
        try a_obj.put(allocator, "path", .{ .string = item_path });
        try a_obj.put(allocator, "sha256", .{ .string = try sha256File(allocator, io, item_path, diag) });
        try a_obj.put(allocator, "size", .{ .integer = @intCast(st.size) });
        try artifacts_arr.append(.{ .object = a_obj });
    }

    var wamr_cmd_arr = Array.init(allocator);
    for (cmds.wamr) |arg| try wamr_cmd_arr.append(.{ .string = arg });
    var wasmtime_cmd_arr = Array.init(allocator);
    for (cmds.wasmtime) |arg| try wasmtime_cmd_arr.append(.{ .string = arg });
    var cmds_obj: ObjectMap = .empty;
    try cmds_obj.put(allocator, "wamr", .{ .array = wamr_cmd_arr });
    try cmds_obj.put(allocator, "wasmtime", .{ .array = wasmtime_cmd_arr });

    var rep: ObjectMap = .empty;
    try rep.put(allocator, "artifacts", .{ .array = artifacts_arr });
    try rep.put(allocator, "commands", .{ .object = cmds_obj });
    try rep.put(allocator, "excluded_from_measurement", .{ .bool = true });
    try rep.put(allocator, "frame_attribution", frame_artifact);
    try rep.put(allocator, "wamr_seconds", .{ .float = wamr_seconds });
    try rep.put(allocator, "wasmtime_seconds", .{ .float = wasmtime_seconds });

    return .{
        .wamr_manifest = cmds.wamr_manifest,
        .wasmtime_cwasm = cmds.wasmtime_cwasm,
        .report = .{ .object = rep },
    };
}

pub fn runOne(
    allocator: Allocator,
    io: Io,
    config: Config,
    runtime: []const u8,
    phase: []const u8,
    index: usize,
    output_root: []const u8,
    wamr_manifest: []const u8,
    wasmtime_cwasm: []const u8,
    timeout_seconds: f64,
    hooks: Hooks,
    diag: *Diagnostic,
) HarnessError!Value {
    const leaf = try std.fmt.allocPrint(allocator, "{s}-{d:0>2}", .{ phase, index });
    const output_dir = try std.fs.path.join(allocator, &.{ output_root, runtime, leaf });
    Io.Dir.cwd().deleteTree(io, output_dir) catch {};
    Io.Dir.cwd().createDirPath(io, output_dir) catch {
        return diag.fail(allocator, "failed to create output directory: {s}", .{output_dir});
    };
    const command = try runtimeCommand(
        allocator,
        io,
        config,
        runtime,
        output_dir,
        wamr_manifest,
        wasmtime_cwasm,
        diag,
    );
    const label = try std.fmt.allocPrint(allocator, "{s} {s} run {d}", .{ runtime, phase, index });
    const start = Io.Timestamp.now(io, .awake);
    const proc = try runChecked(allocator, io, command, timeout_seconds, label, null, hooks, diag);
    const elapsed_ns = start.untilNow(io, .awake).toNanoseconds();
    const elapsed: f64 = @as(f64, @floatFromInt(@max(elapsed_ns, 1))) / 1e9;

    const combined = try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ proc.stdout, proc.stderr });
    const response_bytes = try parseTcgcResponseBytes(allocator, combined, diag);
    if (response_bytes != config.expected_response_bytes) {
        return diag.fail(
            allocator,
            "{s} {s} run {d} returned {d} bytes from tcgc.compile; manifest requires {d}",
            .{ runtime, phase, index, response_bytes, config.expected_response_bytes },
        );
    }
    const snapshot = try treeSnapshot(allocator, io, output_dir, diag);
    var cmd_arr = Array.init(allocator);
    for (command) |arg| try cmd_arr.append(.{ .string = arg });

    var rec: ObjectMap = .empty;
    try rec.put(allocator, "command", .{ .array = cmd_arr });
    try rec.put(allocator, "elapsed_seconds", .{ .float = elapsed });
    try rec.put(allocator, "index", .{ .integer = @intCast(index) });
    try rec.put(allocator, "output", try snapshot.toOutputJsonValue(allocator));
    try rec.put(allocator, "phase", .{ .string = phase });
    try rec.put(allocator, "runtime", .{ .string = runtime });
    try rec.put(allocator, "tcgc_response_bytes", .{ .integer = response_bytes });
    return .{ .object = rec };
}

pub fn writeCanonicalJson(writer: *Io.Writer, val: Value) !void {
    switch (val) {
        .null => try writer.writeAll("null"),
        .bool => |b| try writer.writeAll(if (b) "true" else "false"),
        .integer => |n| try writer.print("{d}", .{n}),
        .float => |f| try writer.print("{d}", .{f}),
        .number_string => |s| try writer.writeAll(s),
        .string => |s| try json.Stringify.encodeJsonString(s, .{}, writer),
        .array => |arr| {
            try writer.writeByte('[');
            for (arr.items, 0..) |item, idx| {
                if (idx != 0) try writer.writeByte(',');
                try writeCanonicalJson(writer, item);
            }
            try writer.writeByte(']');
        },
        .object => |obj| {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const alloc = arena.allocator();
            const keys = try alloc.alloc([]const u8, obj.count());
            var it = obj.iterator();
            var idx: usize = 0;
            while (it.next()) |entry| : (idx += 1) {
                keys[idx] = entry.key_ptr.*;
            }
            std.mem.sort([]const u8, keys, {}, struct {
                fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.order(u8, a, b) == .lt;
                }
            }.lessThan);
            try writer.writeByte('{');
            for (keys, 0..) |k, k_idx| {
                if (k_idx != 0) try writer.writeByte(',');
                try json.Stringify.encodeJsonString(k, .{}, writer);
                try writer.writeByte(':');
                try writeCanonicalJson(writer, obj.get(k).?);
            }
            try writer.writeByte('}');
        },
    }
}

pub fn writePrettySortedJson(writer: *Io.Writer, val: Value, indent: usize) !void {
    switch (val) {
        .null => try writer.writeAll("null"),
        .bool => |b| try writer.writeAll(if (b) "true" else "false"),
        .integer => |n| try writer.print("{d}", .{n}),
        .float => |f| try writer.print("{d}", .{f}),
        .number_string => |s| try writer.writeAll(s),
        .string => |s| try json.Stringify.encodeJsonString(s, .{}, writer),
        .array => |arr| {
            if (arr.items.len == 0) {
                try writer.writeAll("[]");
                return;
            }
            try writer.writeAll("[\n");
            for (arr.items, 0..) |item, idx| {
                try writer.splatByteAll(' ', indent + 2);
                try writePrettySortedJson(writer, item, indent + 2);
                if (idx + 1 < arr.items.len) try writer.writeByte(',');
                try writer.writeByte('\n');
            }
            try writer.splatByteAll(' ', indent);
            try writer.writeByte(']');
        },
        .object => |obj| {
            if (obj.count() == 0) {
                try writer.writeAll("{}");
                return;
            }
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const alloc = arena.allocator();
            const keys = try alloc.alloc([]const u8, obj.count());
            var it = obj.iterator();
            var idx: usize = 0;
            while (it.next()) |entry| : (idx += 1) {
                keys[idx] = entry.key_ptr.*;
            }
            std.mem.sort([]const u8, keys, {}, struct {
                fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.order(u8, a, b) == .lt;
                }
            }.lessThan);
            try writer.writeAll("{\n");
            for (keys, 0..) |k, k_idx| {
                try writer.splatByteAll(' ', indent + 2);
                try json.Stringify.encodeJsonString(k, .{}, writer);
                try writer.writeAll(": ");
                try writePrettySortedJson(writer, obj.get(k).?, indent + 2);
                if (k_idx + 1 < keys.len) try writer.writeByte(',');
                try writer.writeByte('\n');
            }
            try writer.splatByteAll(' ', indent);
            try writer.writeByte('}');
        },
    }
}

fn canonicalJsonEquals(allocator: Allocator, a: Value, b: Value) error{OutOfMemory}!bool {
    var buf_a: Io.Writer.Allocating = .init(allocator);
    writeCanonicalJson(&buf_a.writer, a) catch return error.OutOfMemory;
    var buf_b: Io.Writer.Allocating = .init(allocator);
    writeCanonicalJson(&buf_b.writer, b) catch return error.OutOfMemory;
    return std.mem.eql(u8, buf_a.written(), buf_b.written());
}

pub fn verifyEquivalence(
    allocator: Allocator,
    records: []const Value,
    diag: *Diagnostic,
) HarnessError!Value {
    if (records.len == 0) {
        return diag.fail(allocator, "no runtime records were produced", .{});
    }
    const first = records[0].object;
    const expected_bytes = first.get("tcgc_response_bytes").?.integer;
    const expected_output = first.get("output").?;

    for (records[1..]) |rec_val| {
        const record = rec_val.object;
        const runtime = record.get("runtime").?.string;
        const phase = record.get("phase").?.string;
        const index = record.get("index").?.integer;
        const got_bytes = record.get("tcgc_response_bytes").?.integer;
        if (got_bytes != expected_bytes) {
            return diag.fail(
                allocator,
                "tcgc response-size drift: expected {d}, got {d} for {s} {s} {d}",
                .{ expected_bytes, got_bytes, runtime, phase, index },
            );
        }
        const got_output = record.get("output").?;
        if (!(try canonicalJsonEquals(allocator, got_output, expected_output))) {
            return diag.fail(
                allocator,
                "generated output mismatch: exact paths, sizes, or SHA-256 values differ for {s} {s} {d}",
                .{ runtime, phase, index },
            );
        }
    }

    const out_obj = expected_output.object;
    var res: ObjectMap = .empty;
    try res.put(allocator, "cross_runtime_exact_match", .{ .bool = true });
    try res.put(allocator, "deterministic_across_all_runs", .{ .bool = true });
    try res.put(allocator, "file_count", out_obj.get("file_count").?);
    try res.put(allocator, "output_sha256_tree", out_obj.get("sha256_tree").?);
    try res.put(allocator, "tcgc_response_bytes", .{ .integer = expected_bytes });
    try res.put(allocator, "total_bytes", out_obj.get("total_bytes").?);
    return .{ .object = res };
}

pub fn sampleStats(allocator: Allocator, values: []const f64) !Value {
    var sum: f64 = 0.0;
    var min_v: f64 = values[0];
    var max_v: f64 = values[0];
    var arr = Array.init(allocator);
    for (values) |v| {
        sum += v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        try arr.append(.{ .float = v });
    }
    const sorted = try allocator.dupe(f64, values);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const mid = sorted.len / 2;
    const median = if (sorted.len % 2 == 1)
        sorted[mid]
    else
        (sorted[mid - 1] + sorted[mid]) / 2.0;

    var obj: ObjectMap = .empty;
    try obj.put(allocator, "max_seconds", .{ .float = max_v });
    try obj.put(allocator, "mean_seconds", .{ .float = sum / @as(f64, @floatFromInt(values.len)) });
    try obj.put(allocator, "median_seconds", .{ .float = median });
    try obj.put(allocator, "min_seconds", .{ .float = min_v });
    try obj.put(allocator, "range_seconds", .{ .float = max_v - min_v });
    try obj.put(allocator, "runs", .{ .integer = @intCast(values.len) });
    try obj.put(allocator, "samples_seconds", .{ .array = arr });
    return .{ .object = obj };
}

fn runUnameField(allocator: Allocator, io: Io, flag: []const u8, fallback: []const u8) ![]const u8 {
    const res = std.process.run(allocator, io, .{ .argv = &.{ "uname", flag } }) catch {
        return allocator.dupe(u8, fallback);
    };
    const trimmed = std.mem.trim(u8, res.stdout, " \t\r\n");
    if (trimmed.len == 0) return allocator.dupe(u8, fallback);
    return allocator.dupe(u8, trimmed);
}

pub fn hostReport(allocator: Allocator, io: Io) !Value {
    var cpu: []const u8 = "unknown";
    if (std.process.run(allocator, io, .{ .argv = &.{"lscpu"} })) |res| {
        var line_it = std.mem.splitScalar(u8, res.stdout, '\n');
        while (line_it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r\n");
            if (std.ascii.startsWithIgnoreCase(trimmed, "model name:")) {
                if (std.mem.indexOfScalar(u8, trimmed, ':')) |colon| {
                    cpu = try allocator.dupe(u8, std.mem.trim(u8, trimmed[colon + 1 ..], " \t\r\n"));
                    break;
                }
            }
        }
    } else |_| {}

    const default_sys = switch (builtin.os.tag) {
        .linux => "Linux",
        .macos => "Darwin",
        .windows => "Windows",
        else => @tagName(builtin.os.tag),
    };
    const default_arch = @tagName(builtin.cpu.arch);

    var obj: ObjectMap = .empty;
    try obj.put(allocator, "cpu", .{ .string = cpu });
    try obj.put(allocator, "logical_cpus", .{ .integer = @intCast(std.Thread.getCpuCount() catch 1) });
    try obj.put(allocator, "machine", .{ .string = try runUnameField(allocator, io, "-m", default_arch) });
    try obj.put(allocator, "release", .{ .string = try runUnameField(allocator, io, "-r", "unknown") });
    try obj.put(allocator, "system", .{ .string = try runUnameField(allocator, io, "-s", default_sys) });
    try obj.put(allocator, "zig", .{ .string = builtin.zig_version_string });
    return .{ .object = obj };
}

fn findToolInPath(allocator: Allocator, io: Io, name: []const u8, hooks: Hooks) ?[]const u8 {
    if (hooks.which_override) |override| {
        if (override.len == 0) return null;
        return override;
    }
    const path_env = getEnvOwned(allocator, "PATH") orelse "/usr/bin:/bin";
    var it = std.mem.splitScalar(u8, path_env, ':');
    while (it.next()) |dir_entry| {
        if (dir_entry.len == 0) continue;
        const candidate = std.fs.path.join(allocator, &.{ dir_entry, name }) catch continue;
        if (isExecutableFile(io, candidate, .{})) return candidate;
    }
    return null;
}

pub fn runPerf(
    allocator: Allocator,
    io: Io,
    config: Config,
    work_dir: []const u8,
    wamr_manifest: []const u8,
    wasmtime_cwasm: []const u8,
    timeout_seconds: f64,
    reference_output: Value,
    hooks: Hooks,
    diag: *Diagnostic,
) HarnessError!Value {
    const sys_name = hooks.system_name orelse switch (builtin.os.tag) {
        .linux => "Linux",
        .macos => "Darwin",
        else => @tagName(builtin.os.tag),
    };
    const mach_name = hooks.machine_name orelse @tagName(builtin.cpu.arch);
    if (!std.mem.eql(u8, sys_name, "Linux") or !std.mem.eql(u8, mach_name, "x86_64")) {
        return diag.fail(allocator, "--profile requires Linux x86_64", .{});
    }
    for ([_][]const u8{ "perf", "objdump" }) |tool_name| {
        if (findToolInPath(allocator, io, tool_name, hooks) == null) {
            return diag.fail(
                allocator,
                "--profile explicitly selected, but required tool '{s}' is absent",
                .{tool_name},
            );
        }
    }

    var paranoid_val: ?i64 = null;
    if (hooks.paranoid_override) |po| {
        paranoid_val = po;
    } else if (Io.Dir.cwd().readFileAlloc(io, "/proc/sys/kernel/perf_event_paranoid", allocator, .limited(256))) |content| {
        paranoid_val = std.fmt.parseInt(i64, std.mem.trim(u8, content, " \t\r\n"), 10) catch null;
    } else |_| {}
    if (paranoid_val) |pv| {
        if (pv > 2) {
            return diag.fail(
                allocator,
                "--profile explicitly selected, but kernel.perf_event_paranoid is {d}; set it to <= 2",
                .{pv},
            );
        }
    }

    const perf_dir = try std.fs.path.join(allocator, &.{ work_dir, "perf" });
    const output_dir = try std.fs.path.join(allocator, &.{ perf_dir, "output" });
    Io.Dir.cwd().deleteTree(io, perf_dir) catch {};
    Io.Dir.cwd().createDirPath(io, output_dir) catch {
        return diag.fail(allocator, "failed to create perf output directory: {s}", .{output_dir});
    };

    const runtime_cmd = try runtimeCommand(
        allocator,
        io,
        config,
        "wamr",
        output_dir,
        wamr_manifest,
        wasmtime_cwasm,
        diag,
    );
    const perf_data = try std.fs.path.join(allocator, &.{ perf_dir, "wamr.perf.data" });
    var command: std.ArrayList([]const u8) = .empty;
    try command.appendSlice(allocator, &.{
        "perf",
        "record",
        "-g",
        "--call-graph=dwarf",
        "-F",
        "999",
        "-e",
        "cycles:u",
        "-o",
        perf_data,
        "--",
    });
    try command.appendSlice(allocator, runtime_cmd);

    const proc = try runChecked(
        allocator,
        io,
        command.items,
        timeout_seconds,
        "WAMR perf capture",
        null,
        hooks,
        diag,
    );
    const combined = try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ proc.stdout, proc.stderr });
    const response_bytes = try parseTcgcResponseBytes(allocator, combined, diag);
    const snapshot = try treeSnapshot(allocator, io, output_dir, diag);
    const current_output = try snapshot.toOutputJsonValue(allocator);
    if (response_bytes != config.expected_response_bytes or
        !(try canonicalJsonEquals(allocator, current_output, reference_output)))
    {
        return diag.fail(allocator, "profiled WAMR run was not semantically equivalent", .{});
    }

    const artifacts_dir = try std.fs.path.join(allocator, &.{ work_dir, "artifacts" });
    const core_name = try std.fmt.allocPrint(allocator, "keyvault.{d}.cwasm", .{config.perf.core_index});
    const core = try std.fs.path.join(allocator, &.{ artifacts_dir, core_name });
    if (!isRegularFile(io, core)) {
        return diag.fail(
            allocator,
            "configured perf core artifact is absent: {s}; check perf.core_index and the WAMR manifest",
            .{core},
        );
    }
    const attr_json = try std.fs.path.join(allocator, &.{ perf_dir, "attribution.json" });
    const frame_artifact = try validateFrameArtifact(
        allocator,
        io,
        config,
        artifacts_dir,
        timeout_seconds,
        hooks,
        diag,
    );
    const sidecar_path = frame_artifact.object.get("sidecar").?.string;

    var attr_command: std.ArrayList([]const u8) = .empty;
    try attr_command.appendSlice(allocator, &.{
        "aot-jit-attr",
        "--perf",
        perf_data,
        "--cwasm",
        core,
        "--json-out",
        attr_json,
        "--min-samples",
        try std.fmt.allocPrint(allocator, "{d}", .{config.perf.min_samples}),
    });
    if (config.perf.base) |base_str| {
        try attr_command.appendSlice(allocator, &.{ "--base", base_str });
    } else {
        try attr_command.append(allocator, "--require-size-match");
    }
    if (config.perf.hot_func) |hf| {
        try attr_command.appendSlice(allocator, &.{
            "--func",
            try std.fmt.allocPrint(allocator, "{d}", .{hf}),
            "--frame-metadata",
            sidecar_path,
        });
    }

    _ = try runChecked(
        allocator,
        io,
        attr_command.items,
        timeout_seconds,
        "perf attribution",
        null,
        hooks,
        diag,
    );

    const attr_bytes = Io.Dir.cwd().readFileAlloc(io, attr_json, allocator, .limited(32 * 1024 * 1024)) catch {
        return diag.fail(allocator, "failed reading attribution JSON: {s}", .{attr_json});
    };
    const attribution = json.parseFromSliceLeaky(Value, allocator, attr_bytes, .{}) catch {
        return diag.fail(allocator, "invalid attribution JSON: {s}", .{attr_json});
    };
    const sidecar_bytes = Io.Dir.cwd().readFileAlloc(io, sidecar_path, allocator, .limited(32 * 1024 * 1024)) catch {
        return diag.fail(allocator, "failed reading frame sidecar JSON: {s}", .{sidecar_path});
    };
    const sidecar_json = json.parseFromSliceLeaky(Value, allocator, sidecar_bytes, .{}) catch {
        return diag.fail(allocator, "invalid frame sidecar JSON: {s}", .{sidecar_path});
    };

    const attr_obj = if (attribution == .object) attribution.object else return diag.fail(allocator, "perf attribution does not report the exact selected frame sidecar/core", .{});
    const classified_val = attr_obj.get("classified_function") orelse .null;
    const classified = if (classified_val == .object) classified_val.object else return diag.fail(allocator, "perf attribution does not report the exact selected frame sidecar/core", .{});
    const identity_val = classified.get("frame_metadata_identity") orelse .null;
    const identity = if (identity_val == .object) identity_val.object else return diag.fail(allocator, "perf attribution does not report the exact selected frame sidecar/core", .{});
    const sidecar_obj = if (sidecar_json == .object) sidecar_json.object else return diag.fail(allocator, "perf attribution does not report the exact selected frame sidecar/core", .{});

    const cwd_real = Io.Dir.cwd().realPathFileAlloc(io, ".", allocator) catch ".";
    const attr_cwasm = if (attr_obj.get("cwasm")) |cv| (if (cv == .string) cv.string else "") else "";
    const resolved_attr_cwasm = try resolvePath(allocator, io, cwd_real, attr_cwasm);
    const resolved_core = try resolvePath(allocator, io, cwd_real, core);

    const local_func_ok = if (config.perf.hot_func) |hf|
        (if (classified.get("local_func")) |lf| (lf == .integer and lf.integer == hf) else false)
    else
        true;
    const fm_ok = if (classified.get("frame_metadata")) |fm|
        (fm == .string and std.mem.eql(u8, fm.string, sidecar_path))
    else
        false;
    const fmm_ok = if (classified.get("frame_metadata_module")) |fmm|
        (fmm == .integer and fmm.integer == config.perf.core_index)
    else
        false;
    const mod_sha_ok = if (identity.get("module_text_sha256")) |ms|
        (if (sidecar_obj.get("module_text_sha256")) |sms|
            (ms == .string and sms == .string and std.mem.eql(u8, ms.string, sms.string))
        else
            false)
    else
        false;
    const norm_sha_ok = if (identity.get("normalized_code_sha256")) |ns|
        (if (sidecar_obj.get("normalized_code_sha256")) |sns|
            (ns == .string and sns == .string and std.mem.eql(u8, ns.string, sns.string))
        else
            false)
    else
        false;
    const has_fa = classified.contains("frame_attribution");

    if (!std.mem.eql(u8, resolved_attr_cwasm, resolved_core) or
        !local_func_ok or !fm_ok or !fmm_ok or !mod_sha_ok or !norm_sha_ok or !has_fa)
    {
        return diag.fail(
            allocator,
            "perf attribution does not report the exact selected frame sidecar/core",
            .{},
        );
    }

    const cov_val = attr_obj.get("attribution_coverage_pct") orelse {
        return diag.fail(allocator, "perf attribution missing attribution_coverage_pct", .{});
    };
    const coverage: f64 = switch (cov_val) {
        .integer => |iv| @floatFromInt(iv),
        .float => |fv| fv,
        else => 0.0,
    };
    if (coverage < config.perf.min_attribution_coverage_pct) {
        return diag.fail(
            allocator,
            "perf attribution coverage {d:.2}% is below manifest minimum {d:.2}%",
            .{ coverage, config.perf.min_attribution_coverage_pct },
        );
    }

    var cmd_arr = Array.init(allocator);
    for (command.items) |arg| try cmd_arr.append(.{ .string = arg });
    var attr_cmd_arr = Array.init(allocator);
    for (attr_command.items) |arg| try attr_cmd_arr.append(.{ .string = arg });

    var rep: ObjectMap = .empty;
    try rep.put(allocator, "attribution", attribution);
    try rep.put(allocator, "attribution_command", .{ .array = attr_cmd_arr });
    try rep.put(allocator, "command", .{ .array = cmd_arr });
    try rep.put(allocator, "frame_attribution", frame_artifact);
    try rep.put(allocator, "perf_data", .{ .string = perf_data });
    try rep.put(allocator, "selected", .{ .bool = true });
    return .{ .object = rep };
}

fn jsonFloat(v: Value) f64 {
    return switch (v) {
        .float => |fv| fv,
        .integer => |iv| @floatFromInt(iv),
        else => 0.0,
    };
}

pub fn renderMarkdown(allocator: Allocator, report: Value) ![]const u8 {
    var out: Io.Writer.Allocating = .init(allocator);
    const w = &out.writer;

    const root = report.object;
    const timing = root.get("timing").?.object;
    const wamr = timing.get("wamr").?.object;
    const wasmtime = timing.get("wasmtime").?.object;
    const host = root.get("host").?.object;
    const plan = root.get("plan").?.object;
    const equivalence = root.get("equivalence").?.object;
    const validation = root.get("validation").?.object;

    try w.writeAll("# Keyvault SpiderMonkey/TCGC AOT benchmark\n\n");
    try w.print(
        "- Host: `{s} {s}` · `{s}` · {d} CPUs · `{s}`\n",
        .{
            host.get("system").?.string,
            host.get("release").?.string,
            host.get("machine").?.string,
            host.get("logical_cpus").?.integer,
            host.get("cpu").?.string,
        },
    );
    try w.print("- Manifest: `{s}`\n", .{root.get("manifest").?.string});
    try w.print("- Warmups: {d} per runtime\n", .{plan.get("warmups").?.integer});
    try w.print("- Authoritative measured runs: {d} per runtime\n", .{plan.get("runs").?.integer});
    try w.writeAll("- Compilation: precompiled once per runtime and excluded from measured execution\n\n");
    try w.writeAll("| Runtime | Samples (s) | Mean | Median | Min | Max | Range |\n");
    try w.writeAll("|---|---|---:|---:|---:|---:|---:|\n");

    inline for ([_]struct { label: []const u8, key: []const u8 }{
        .{ .label = "WAMR", .key = "wamr" },
        .{ .label = "Wasmtime", .key = "wasmtime" },
    }) |rt_info| {
        const rt_obj = if (std.mem.eql(u8, rt_info.key, "wamr")) wamr else wasmtime;
        var samples_buf: Io.Writer.Allocating = .init(allocator);
        for (rt_obj.get("samples_seconds").?.array.items, 0..) |sv, idx| {
            if (idx > 0) try samples_buf.writer.writeAll(", ");
            try samples_buf.writer.print("{d:.3}", .{jsonFloat(sv)});
        }
        try w.print(
            "| {s} | {s} | {d:.3} | {d:.3} | {d:.3} | {d:.3} | {d:.3} |\n",
            .{
                rt_info.label,
                samples_buf.written(),
                jsonFloat(rt_obj.get("mean_seconds").?),
                jsonFloat(rt_obj.get("median_seconds").?),
                jsonFloat(rt_obj.get("min_seconds").?),
                jsonFloat(rt_obj.get("max_seconds").?),
                jsonFloat(rt_obj.get("range_seconds").?),
            },
        );
    }

    try w.print(
        "\n**Wasmtime-time/WAMR-time ratio: {d:.4}×**\n\n",
        .{jsonFloat(timing.get("wasmtime_time_over_wamr_time").?)},
    );
    try w.writeAll("## Semantic equivalence\n\n");
    try w.print("- `tcgc.compile` bytes: {d}\n", .{equivalence.get("tcgc_response_bytes").?.integer});
    try w.print(
        "- Generated files: {d} ({d} bytes)\n",
        .{ equivalence.get("file_count").?.integer, equivalence.get("total_bytes").?.integer },
    );
    try w.print("- Output tree SHA-256: `{s}`\n", .{equivalence.get("output_sha256_tree").?.string});
    try w.writeAll("- Every warmup/measured output and both runtimes matched exactly by path, size, and SHA-256.\n\n");
    try w.writeAll("## Exact tools and inputs\n\n");

    const tools_obj = validation.get("tools").?.object;
    var tools_it = tools_obj.iterator();
    while (tools_it.next()) |entry| {
        const t = entry.value_ptr.*.object;
        try w.print(
            "- {s}: `{s}` · SHA-256 `{s}` · `{s}`\n",
            .{
                entry.key_ptr.*,
                t.get("version").?.string,
                t.get("sha256").?.string,
                t.get("path").?.string,
            },
        );
    }
    const inputs = validation.get("inputs").?.object;
    const comp = inputs.get("component").?.object;
    try w.print(
        "- component: SHA-256 `{s}` · {d} bytes\n",
        .{ comp.get("sha256").?.string, comp.get("size").?.integer },
    );
    const preopens = inputs.get("preopens_file").?.object;
    try w.print(
        "- preopens file: SHA-256 `{s}` · `{s}`\n",
        .{ preopens.get("sha256").?.string, preopens.get("path").?.string },
    );
    for (inputs.get("mounts").?.array.items) |m_val| {
        const m = m_val.object;
        try w.print(
            "- mount `{s}`: tree SHA-256 `{s}` · {d} files / {d} bytes · `{s}`\n",
            .{
                m.get("guest").?.string,
                m.get("sha256_tree").?.string,
                m.get("file_count").?.integer,
                m.get("total_bytes").?.integer,
                m.get("path").?.string,
            },
        );
    }
    for (validation.get("sources").?.array.items) |s_val| {
        const s = s_val.object;
        try w.print(
            "- source {s}: `{s}` · `{s}`\n",
            .{ s.get("name").?.string, s.get("revision").?.string, s.get("path").?.string },
        );
    }

    const perf = root.get("perf").?.object;
    if (perf.get("selected").?.bool) {
        const attr = perf.get("attribution").?.object;
        const fa = perf.get("frame_attribution").?.object;
        try w.writeAll("\n## Perf attribution\n\n");
        try w.print("- Total self samples: {d}\n", .{attr.get("total_samples").?.integer});
        try w.print("- Samples attributed to configured core: {d}\n", .{attr.get("attributed_samples").?.integer});
        try w.print("- Attribution coverage: {d:.2}%\n", .{jsonFloat(attr.get("attribution_coverage_pct").?)});
        try w.print("- `perf.data`: `{s}`\n", .{perf.get("perf_data").?.string});
        try w.print(
            "- Exact frame sidecar: `{s}` (SHA-256 `{s}`)\n",
            .{ fa.get("sidecar").?.string, fa.get("sidecar_sha256").?.string },
        );
        try w.print(
            "- Matched core image: `{s}` (SHA-256 `{s}`)\n",
            .{ fa.get("cwasm").?.string, fa.get("cwasm_sha256").?.string },
        );
        if (attr.get("classified_function")) |cf_val| {
            if (cf_val == .object) {
                if (cf_val.object.get("frame_attribution")) |frame_val| {
                    if (frame_val == .object) {
                        const frame = frame_val.object;
                        const cov = frame.get("coverage").?.object;
                        try w.print(
                            "- Proven frame origins: {d:.2}% of frame samples; unknown: {d} samples\n",
                            .{
                                jsonFloat(cov.get("origin_sample_coverage_pct").?),
                                cov.get("unknown_frame_samples").?.integer,
                            },
                        );
                        if (frame.get("origins")) |orig_val| {
                            if (orig_val == .object) {
                                var keys: std.ArrayList([]const u8) = .empty;
                                var o_it = orig_val.object.iterator();
                                while (o_it.next()) |entry| try keys.append(allocator, entry.key_ptr.*);
                                std.mem.sort([]const u8, keys.items, {}, struct {
                                    fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                                        return std.mem.order(u8, a, b) == .lt;
                                    }
                                }.lessThan);
                                for (keys.items) |origin| {
                                    const data = orig_val.object.get(origin).?.object;
                                    try w.print("  - {s}: {d} samples\n", .{ origin, data.get("samples").?.integer });
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        try w.writeAll(
            "\n## Perf attribution\n\n" ++
                "- Not selected. Re-run with `--profile`; missing/unsupported perf is a hard error in that mode.\n",
        );
    }

    return out.toOwnedSlice();
}

pub const ExecuteOptions = struct {
    manifest: []const u8,
    work_dir: []const u8,
    warmups: i64 = 1,
    runs: usize = 5,
    compile_timeout: f64 = 1800.0,
    run_timeout: f64 = 600.0,
    profile: bool = false,
    report_json: []const u8 = "keyvault-report.json",
    report_markdown: []const u8 = "keyvault-report.md",
};

pub fn execute(
    allocator: Allocator,
    io: Io,
    options: ExecuteOptions,
    hooks: Hooks,
    diag: *Diagnostic,
) HarnessError!Value {
    if (options.runs < MIN_MEASURED_RUNS) {
        return diag.fail(
            allocator,
            "--runs must be >= {d} for an authoritative result",
            .{MIN_MEASURED_RUNS},
        );
    }
    if (options.warmups < 0) {
        return diag.fail(allocator, "--warmups must be >= 0", .{});
    }
    const sys_name = hooks.system_name orelse switch (builtin.os.tag) {
        .linux => "Linux",
        .macos => "Darwin",
        else => @tagName(builtin.os.tag),
    };
    const mach_name = hooks.machine_name orelse @tagName(builtin.cpu.arch);
    if (!std.mem.eql(u8, sys_name, "Linux") or !std.mem.eql(u8, mach_name, "x86_64")) {
        return diag.fail(allocator, "the pinned #798 timing cohort requires Linux x86_64", .{});
    }

    const config = try loadManifest(allocator, io, options.manifest, diag);
    const validation = try validateConfig(allocator, io, config, hooks, diag);
    const cwd_real = Io.Dir.cwd().realPathFileAlloc(io, ".", allocator) catch ".";
    const work_dir = try resolvePath(allocator, io, cwd_real, options.work_dir);
    Io.Dir.cwd().createDirPath(io, work_dir) catch {
        return diag.fail(allocator, "failed to create work directory: {s}", .{work_dir});
    };

    const artifacts_dir = try std.fs.path.join(allocator, &.{ work_dir, "artifacts" });
    const precompiled = try precompile(
        allocator,
        io,
        config,
        artifacts_dir,
        options.compile_timeout,
        options.profile,
        hooks,
        diag,
    );

    const outputs_dir = try std.fs.path.join(allocator, &.{ work_dir, "outputs" });
    var records = Array.init(allocator);
    var w_idx: usize = 1;
    const warmup_count: usize = @intCast(options.warmups);
    while (w_idx <= warmup_count) : (w_idx += 1) {
        for ([_][]const u8{ "wamr", "wasmtime" }) |rt| {
            const rec = try runOne(
                allocator,
                io,
                config,
                rt,
                "warmup",
                w_idx,
                outputs_dir,
                precompiled.wamr_manifest,
                precompiled.wasmtime_cwasm,
                options.run_timeout,
                hooks,
                diag,
            );
            try records.append(rec);
        }
    }

    var m_idx: usize = 1;
    while (m_idx <= options.runs) : (m_idx += 1) {
        const order = if (m_idx % 2 == 1)
            [_][]const u8{ "wamr", "wasmtime" }
        else
            [_][]const u8{ "wasmtime", "wamr" };
        for (order) |rt| {
            const rec = try runOne(
                allocator,
                io,
                config,
                rt,
                "measure",
                m_idx,
                outputs_dir,
                precompiled.wamr_manifest,
                precompiled.wasmtime_cwasm,
                options.run_timeout,
                hooks,
                diag,
            );
            try records.append(rec);
        }
    }

    const equivalence = try verifyEquivalence(allocator, records.items, diag);
    const wamr_samples = try parseTimingSamples(allocator, records.items, "wamr", diag);
    const wasmtime_samples = try parseTimingSamples(allocator, records.items, "wasmtime", diag);
    const wamr_stats = try sampleStats(allocator, wamr_samples);
    const wasmtime_stats = try sampleStats(allocator, wasmtime_samples);
    const ratio = jsonFloat(wasmtime_stats.object.get("mean_seconds").?) /
        jsonFloat(wamr_stats.object.get("mean_seconds").?);

    var timing_obj: ObjectMap = .empty;
    try timing_obj.put(allocator, "wamr", wamr_stats);
    try timing_obj.put(allocator, "wasmtime", wasmtime_stats);
    try timing_obj.put(allocator, "wasmtime_time_over_wamr_time", .{ .float = ratio });

    var measured_reference: ?Value = null;
    for (records.items) |rec_val| {
        if (std.mem.eql(u8, rec_val.object.get("phase").?.string, "measure")) {
            measured_reference = rec_val.object.get("output").?;
            break;
        }
    }

    const perf_report: Value = if (options.profile)
        try runPerf(
            allocator,
            io,
            config,
            work_dir,
            precompiled.wamr_manifest,
            precompiled.wasmtime_cwasm,
            options.run_timeout,
            measured_reference.?,
            hooks,
            diag,
        )
    else blk: {
        var p_obj: ObjectMap = .empty;
        try p_obj.put(allocator, "selected", .{ .bool = false });
        break :blk .{ .object = p_obj };
    };

    var plan_obj: ObjectMap = .empty;
    try plan_obj.put(allocator, "compile_timeout_seconds", .{ .float = options.compile_timeout });
    try plan_obj.put(allocator, "run_timeout_seconds", .{ .float = options.run_timeout });
    try plan_obj.put(allocator, "runs", .{ .integer = @intCast(options.runs) });
    try plan_obj.put(allocator, "warmups", .{ .integer = options.warmups });

    var report_obj: ObjectMap = .empty;
    try report_obj.put(allocator, "equivalence", equivalence);
    try report_obj.put(allocator, "host", try hostReport(allocator, io));
    try report_obj.put(allocator, "kind", .{ .string = "keyvault-tcgc-aot-benchmark" });
    try report_obj.put(allocator, "manifest", .{ .string = config.manifest_path });
    try report_obj.put(allocator, "perf", perf_report);
    try report_obj.put(allocator, "plan", .{ .object = plan_obj });
    try report_obj.put(allocator, "precompile", precompiled.report);
    try report_obj.put(allocator, "records", .{ .array = records });
    try report_obj.put(allocator, "schema_version", .{ .integer = SCHEMA_VERSION });
    try report_obj.put(allocator, "timing", .{ .object = timing_obj });
    try report_obj.put(allocator, "validation", validation);

    const report: Value = .{ .object = report_obj };
    var json_buf: Io.Writer.Allocating = .init(allocator);
    writePrettySortedJson(&json_buf.writer, report, 0) catch return error.OutOfMemory;
    json_buf.writer.writeByte('\n') catch return error.OutOfMemory;

    if (std.fs.path.dirname(options.report_json)) |parent| {
        Io.Dir.cwd().createDirPath(io, parent) catch {};
    }
    if (std.fs.path.dirname(options.report_markdown)) |parent| {
        Io.Dir.cwd().createDirPath(io, parent) catch {};
    }
    Io.Dir.cwd().writeFile(io, .{ .sub_path = options.report_json, .data = json_buf.written() }) catch {
        return diag.fail(allocator, "failed writing report JSON: {s}", .{options.report_json});
    };
    const markdown = renderMarkdown(allocator, report) catch return error.OutOfMemory;
    Io.Dir.cwd().writeFile(io, .{ .sub_path = options.report_markdown, .data = markdown }) catch {
        return diag.fail(allocator, "failed writing report Markdown: {s}", .{options.report_markdown});
    };
    return report;
}

pub fn runCli(
    allocator: Allocator,
    io: Io,
    args: []const []const u8,
    stdout: *Io.Writer,
    stderr: *Io.Writer,
) !u8 {
    var manifest: ?[]const u8 = null;
    var work_dir: ?[]const u8 = null;
    var hash_path: ?[]const u8 = null;
    var warmups: i64 = 1;
    var runs: usize = 5;
    var compile_timeout: f64 = 1800.0;
    var run_timeout: f64 = 600.0;
    var profile = false;
    var validate_only = false;
    var report_json: []const u8 = "keyvault-report.json";
    var report_markdown: []const u8 = "keyvault-report.md";

    var diag: Diagnostic = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--manifest")) {
            if (i + 1 >= args.len) {
                try stderr.writeAll("error: --manifest requires a value\n");
                return 2;
            }
            i += 1;
            manifest = args[i];
        } else if (std.mem.eql(u8, arg, "--work-dir")) {
            if (i + 1 >= args.len) {
                try stderr.writeAll("error: --work-dir requires a value\n");
                return 2;
            }
            i += 1;
            work_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--hash")) {
            if (i + 1 >= args.len) {
                try stderr.writeAll("error: --hash requires a value\n");
                return 2;
            }
            i += 1;
            hash_path = args[i];
        } else if (std.mem.eql(u8, arg, "--warmups")) {
            if (i + 1 >= args.len) return 2;
            i += 1;
            warmups = std.fmt.parseInt(i64, args[i], 10) catch {
                try stderr.writeAll("error: invalid --warmups value\n");
                return 2;
            };
        } else if (std.mem.eql(u8, arg, "--runs")) {
            if (i + 1 >= args.len) return 2;
            i += 1;
            runs = std.fmt.parseInt(usize, args[i], 10) catch {
                try stderr.writeAll("error: invalid --runs value\n");
                return 2;
            };
        } else if (std.mem.eql(u8, arg, "--compile-timeout")) {
            if (i + 1 >= args.len) return 2;
            i += 1;
            compile_timeout = std.fmt.parseFloat(f64, args[i]) catch {
                try stderr.writeAll("error: invalid --compile-timeout value\n");
                return 2;
            };
        } else if (std.mem.eql(u8, arg, "--run-timeout")) {
            if (i + 1 >= args.len) return 2;
            i += 1;
            run_timeout = std.fmt.parseFloat(f64, args[i]) catch {
                try stderr.writeAll("error: invalid --run-timeout value\n");
                return 2;
            };
        } else if (std.mem.eql(u8, arg, "--profile")) {
            profile = true;
        } else if (std.mem.eql(u8, arg, "--validate-only")) {
            validate_only = true;
        } else if (std.mem.eql(u8, arg, "--report-json")) {
            if (i + 1 >= args.len) return 2;
            i += 1;
            report_json = args[i];
        } else if (std.mem.eql(u8, arg, "--report-markdown")) {
            if (i + 1 >= args.len) return 2;
            i += 1;
            report_markdown = args[i];
        } else {
            try stderr.print("error: unrecognized argument: {s}\n", .{arg});
            return 2;
        }
    }

    const cwd_real = Io.Dir.cwd().realPathFileAlloc(io, ".", allocator) catch ".";
    if (hash_path) |hp| {
        const resolved = try resolvePath(allocator, io, cwd_real, hp);
        if (isRegularFile(io, resolved)) {
            const sha = sha256File(allocator, io, resolved, &diag) catch {
                try stderr.print("error: {s}\n", .{diag.message});
                return 2;
            };
            var obj: ObjectMap = .empty;
            try obj.put(allocator, "path", .{ .string = resolved });
            try obj.put(allocator, "sha256", .{ .string = sha });
            try writeCanonicalJson(stdout, .{ .object = obj });
            try stdout.writeByte('\n');
            return 0;
        } else if (isDirectory(io, resolved)) {
            const snap = treeSnapshot(allocator, io, resolved, &diag) catch {
                try stderr.print("error: {s}\n", .{diag.message});
                return 2;
            };
            const snap_val = try snap.toJsonValue(allocator);
            try writePrettySortedJson(stdout, snap_val, 0);
            try stdout.writeByte('\n');
            return 0;
        } else {
            try stderr.print("error: path to hash is absent: {s}\n", .{resolved});
            return 2;
        }
    }

    const manifest_path = manifest orelse {
        try stderr.writeAll("error: --manifest is required unless --hash is used\n");
        return 2;
    };
    if (!validate_only and work_dir == null) {
        try stderr.writeAll("error: --work-dir is required for benchmark execution\n");
        return 2;
    }

    if (validate_only) {
        const config = loadManifest(allocator, io, manifest_path, &diag) catch {
            try stderr.print("error: {s}\n", .{diag.message});
            return 2;
        };
        const val = validateConfig(allocator, io, config, .{}, &diag) catch {
            try stderr.print("error: {s}\n", .{diag.message});
            return 2;
        };
        try writePrettySortedJson(stdout, val, 0);
        try stdout.writeByte('\n');
        return 0;
    }

    const resolved_json = try resolvePath(allocator, io, cwd_real, report_json);
    const resolved_md = try resolvePath(allocator, io, cwd_real, report_markdown);
    const report = execute(allocator, io, .{
        .manifest = manifest_path,
        .work_dir = work_dir.?,
        .warmups = warmups,
        .runs = runs,
        .compile_timeout = compile_timeout,
        .run_timeout = run_timeout,
        .profile = profile,
        .report_json = resolved_json,
        .report_markdown = resolved_md,
    }, .{}, &diag) catch {
        try stderr.print("error: {s}\n", .{diag.message});
        return 2;
    };
    const md = try renderMarkdown(allocator, report);
    try stdout.writeAll(md);
    return 0;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    var raw_args: std.ArrayList([]const u8) = .empty;
    var arg_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer arg_it.deinit();
    _ = arg_it.next();
    while (arg_it.next()) |arg| {
        try raw_args.append(allocator, try allocator.dupe(u8, arg));
    }

    var stdout_buf: [4096]u8 = undefined;
    var stderr_buf: [4096]u8 = undefined;
    var stdout_writer = Io.File.stdout().writer(io, &stdout_buf);
    var stderr_writer = Io.File.stderr().writer(io, &stderr_buf);

    const rc = try runCli(
        allocator,
        io,
        raw_args.items,
        &stdout_writer.interface,
        &stderr_writer.interface,
    );
    try stdout_writer.interface.flush();
    try stderr_writer.interface.flush();
    if (rc != 0) std.process.exit(rc);
}

const TestFixture = struct {
    scratch: []const u8,
    sdk: []const u8,
    spec_repo: []const u8,
    component: []const u8,
    preopens: []const u8,
    stdlib: []const u8,
    spec: []const u8,
    tool: []const u8,
    sdk_revision: []const u8,
    spec_revision: []const u8,
    manifest: []const u8,

    fn init(allocator: Allocator, io: Io, test_name: []const u8) !TestFixture {
        const cwd_real = try Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
        const scratch_name = try std.fmt.allocPrint(allocator, "keyvault-harness-{s}", .{test_name});
        const scratch = try std.fs.path.join(allocator, &.{ cwd_real, "zig-out", scratch_name });
        Io.Dir.cwd().deleteTree(io, scratch) catch {};
        try Io.Dir.cwd().createDirPath(io, scratch);

        const sdk = try std.fs.path.join(allocator, &.{ scratch, "sdk" });
        const spec_repo = try std.fs.path.join(allocator, &.{ scratch, "spec-repo" });
        const component = try std.fs.path.join(allocator, &.{ sdk, "component.wasm" });
        const preopens = try std.fs.path.join(allocator, &.{ sdk, "stdlib-preopens.txt" });
        const stdlib = try std.fs.path.join(allocator, &.{ sdk, "stdlib" });
        const spec = try std.fs.path.join(allocator, &.{ spec_repo, "Secrets" });
        try Io.Dir.cwd().createDirPath(io, stdlib);
        try Io.Dir.cwd().createDirPath(io, spec);

        try Io.Dir.cwd().writeFile(io, .{ .sub_path = component, .data = "\x00asm controlled component" });
        const lib_tsp = try std.fs.path.join(allocator, &.{ stdlib, "lib.tsp" });
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = lib_tsp, .data = "model Library {}\n" });
        const main_tsp = try std.fs.path.join(allocator, &.{ spec, "main.tsp" });
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = main_tsp, .data = "model Secret {}\n" });

        const resolved_stdlib = try Io.Dir.cwd().realPathFileAlloc(io, stdlib, allocator);
        const preopens_text = try std.fmt.allocPrint(allocator, "{s}=/stdlib\n", .{resolved_stdlib});
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = preopens, .data = preopens_text });

        const tool = try std.fs.path.join(allocator, &.{ scratch, "fake-tool" });
        try Io.Dir.cwd().writeFile(io, .{
            .sub_path = tool,
            .data = "#!/bin/sh\necho pinned-tool-1.0\n",
            .flags = .{ .permissions = .fromMode(0o755) },
        });

        const sdk_revision = try initGitRepo(allocator, io, sdk);
        const spec_revision = try initGitRepo(allocator, io, spec_repo);
        const manifest = try std.fs.path.join(allocator, &.{ scratch, "manifest.json" });

        var fx: TestFixture = .{
            .scratch = scratch,
            .sdk = sdk,
            .spec_repo = spec_repo,
            .component = component,
            .preopens = preopens,
            .stdlib = stdlib,
            .spec = spec,
            .tool = tool,
            .sdk_revision = sdk_revision,
            .spec_revision = spec_revision,
            .manifest = manifest,
        };
        const doc = try fx.manifestDocument(allocator, io);
        try fx.writeManifest(allocator, io, doc);
        return fx;
    }

    fn deinit(self: TestFixture, io: Io) void {
        Io.Dir.cwd().deleteTree(io, self.scratch) catch {};
    }

    fn initGitRepo(allocator: Allocator, io: Io, path: []const u8) ![]const u8 {
        _ = try std.process.run(allocator, io, .{ .argv = &.{ "git", "init", "-q", path } });
        _ = try std.process.run(allocator, io, .{ .argv = &.{ "git", "-C", path, "config", "user.email", "test@example.invalid" } });
        _ = try std.process.run(allocator, io, .{ .argv = &.{ "git", "-C", path, "config", "user.name", "Harness Test" } });
        _ = try std.process.run(allocator, io, .{ .argv = &.{ "git", "-C", path, "add", "." } });
        _ = try std.process.run(allocator, io, .{ .argv = &.{ "git", "-C", path, "commit", "-qm", "fixture" } });
        const rev_res = try std.process.run(allocator, io, .{ .argv = &.{ "git", "-C", path, "rev-parse", "HEAD" } });
        return allocator.dupe(u8, std.mem.trim(u8, rev_res.stdout, " \t\r\n"));
    }

    fn manifestDocument(self: TestFixture, allocator: Allocator, io: Io) !Value {
        var diag: Diagnostic = .{};
        const tool_sha = try sha256File(allocator, io, self.tool, &diag);

        var tools_obj: ObjectMap = .empty;
        for ([_][]const u8{ "wamr", "wamrc", "wasmtime" }) |name| {
            var t: ObjectMap = .empty;
            try t.put(allocator, "path", .{ .string = self.tool });
            try t.put(allocator, "sha256", .{ .string = tool_sha });
            try t.put(allocator, "version", .{ .string = "pinned-tool-1.0" });
            try tools_obj.put(allocator, name, .{ .object = t });
        }

        var sources_arr = Array.init(allocator);
        for ([_]struct { n: []const u8, p: []const u8, r: []const u8 }{
            .{ .n = "azure-sdk-for-zig", .p = self.sdk, .r = self.sdk_revision },
            .{ .n = "azure-rest-api-specs", .p = self.spec_repo, .r = self.spec_revision },
        }) |s_item| {
            var s: ObjectMap = .empty;
            try s.put(allocator, "name", .{ .string = s_item.n });
            try s.put(allocator, "path", .{ .string = s_item.p });
            try s.put(allocator, "revision", .{ .string = s_item.r });
            try sources_arr.append(.{ .object = s });
        }

        var comp_obj: ObjectMap = .empty;
        try comp_obj.put(allocator, "path", .{ .string = self.component });
        try comp_obj.put(allocator, "sha256", .{ .string = try sha256File(allocator, io, self.component, &diag) });

        var pre_obj: ObjectMap = .empty;
        try pre_obj.put(allocator, "path", .{ .string = self.preopens });
        try pre_obj.put(allocator, "sha256", .{ .string = try sha256File(allocator, io, self.preopens, &diag) });

        const spec_snap = try treeSnapshot(allocator, io, self.spec, &diag);
        const stdlib_snap = try treeSnapshot(allocator, io, self.stdlib, &diag);
        var mounts_arr = Array.init(allocator);
        for ([_]struct { p: []const u8, g: []const u8, s: []const u8 }{
            .{ .p = self.spec, .g = "/spec", .s = spec_snap.sha256 },
            .{ .p = self.stdlib, .g = "/stdlib", .s = stdlib_snap.sha256 },
        }) |m_item| {
            var m: ObjectMap = .empty;
            try m.put(allocator, "guest", .{ .string = m_item.g });
            try m.put(allocator, "path", .{ .string = m_item.p });
            try m.put(allocator, "sha256_tree", .{ .string = m_item.s });
            try mounts_arr.append(.{ .object = m });
        }

        const env_obj: ObjectMap = .empty;
        var workload_obj: ObjectMap = .empty;
        try workload_obj.put(allocator, "component", .{ .object = comp_obj });
        try workload_obj.put(allocator, "env", .{ .object = env_obj });
        try workload_obj.put(allocator, "expected_tcgc_response_bytes", .{ .integer = 58187 });
        try workload_obj.put(allocator, "mounts", .{ .array = mounts_arr });
        try workload_obj.put(allocator, "package_name", .{ .string = "keyvault-secrets" });
        try workload_obj.put(allocator, "preopens_file", .{ .object = pre_obj });

        var perf_obj: ObjectMap = .empty;
        try perf_obj.put(allocator, "base", .null);
        try perf_obj.put(allocator, "core_index", .{ .integer = 4 });
        try perf_obj.put(allocator, "hot_func", .{ .integer = 6145 });
        try perf_obj.put(allocator, "min_attribution_coverage_pct", .{ .integer = 50 });
        try perf_obj.put(allocator, "min_samples", .{ .integer = 5000 });

        var root: ObjectMap = .empty;
        try root.put(allocator, "perf", .{ .object = perf_obj });
        try root.put(allocator, "schema_version", .{ .integer = 1 });
        try root.put(allocator, "sources", .{ .array = sources_arr });
        try root.put(allocator, "tools", .{ .object = tools_obj });
        try root.put(allocator, "workload", .{ .object = workload_obj });
        return .{ .object = root };
    }

    fn writeManifest(self: TestFixture, allocator: Allocator, io: Io, doc: Value) !void {
        var buf: Io.Writer.Allocating = .init(allocator);
        try writePrettySortedJson(&buf.writer, doc, 0);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = self.manifest, .data = buf.written() });
    }
};

fn fakeVersionFn(
    _: ?*anyopaque,
    allocator: Allocator,
    _: Io,
    _: Tool,
    _: *Diagnostic,
) HarnessError![]const u8 {
    return allocator.dupe(u8, "pinned-tool-1.0");
}

test "manifest validation checks hashes revisions and preopens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "validation");
    defer fx.deinit(io);

    var diag: Diagnostic = .{};
    const config = try loadManifest(allocator, io, fx.manifest, &diag);
    const hooks: Hooks = .{ .access_override = true, .command_version_fn = fakeVersionFn };
    const report = try validateConfig(allocator, io, config, hooks, &diag);
    const inputs = report.object.get("inputs").?.object;
    try std.testing.expectEqual(@as(i64, "\x00asm controlled component".len), inputs.get("component").?.object.get("size").?.integer);
    try std.testing.expectEqual(@as(usize, 2), report.object.get("sources").?.array.items.len);
    try std.testing.expectEqual(@as(i64, 1), inputs.get("mounts").?.array.items[1].object.get("file_count").?.integer);

    const lib_tsp = try std.fs.path.join(allocator, &.{ fx.stdlib, "lib.tsp" });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = lib_tsp, .data = "changed\n" });
    try std.testing.expectError(error.HarnessFailed, validateConfig(allocator, io, config, hooks, &diag));
    try std.testing.expect(
        std.mem.indexOf(u8, diag.message, "tracked modifications") != null or
            std.mem.indexOf(u8, diag.message, "tree SHA-256 mismatch") != null,
    );
}

test "manifest rejects placeholders and short revisions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "placeholders");
    defer fx.deinit(io);

    var diag: Diagnostic = .{};
    var doc = try fx.manifestDocument(allocator, io);
    try doc.object.getPtr("tools").?.object.getPtr("wamr").?.object.put(
        allocator,
        "sha256",
        .{ .string = "0000000000000000000000000000000000000000000000000000000000000000" },
    );
    try doc.object.getPtr("sources").?.array.items[0].object.put(allocator, "revision", .{ .string = "19e06fc47" });
    try fx.writeManifest(allocator, io, doc);

    try std.testing.expectError(error.HarnessFailed, loadManifest(allocator, io, fx.manifest, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "non-zero") != null);

    try doc.object.getPtr("tools").?.object.getPtr("wamr").?.object.put(
        allocator,
        "sha256",
        .{ .string = try sha256File(allocator, io, fx.tool, &diag) },
    );
    try fx.writeManifest(allocator, io, doc);
    try std.testing.expectError(error.HarnessFailed, loadManifest(allocator, io, fx.manifest, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "40-character") != null);
}

test "relative paths resolve against manifest directory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "relative-paths");
    defer fx.deinit(io);

    var doc = try fx.manifestDocument(allocator, io);
    var workload = doc.object.getPtr("workload").?.object;
    try workload.getPtr("component").?.object.put(allocator, "path", .{ .string = "sdk/component.wasm" });
    try workload.getPtr("preopens_file").?.object.put(allocator, "path", .{ .string = "sdk/stdlib-preopens.txt" });
    try workload.getPtr("mounts").?.array.items[0].object.put(allocator, "path", .{ .string = "spec-repo/Secrets" });
    try doc.object.getPtr("perf").?.object.put(allocator, "base", .{ .string = "0x400000" });
    try fx.writeManifest(allocator, io, doc);

    var diag: Diagnostic = .{};
    const config = try loadManifest(allocator, io, fx.manifest, &diag);
    const real_comp = try Io.Dir.cwd().realPathFileAlloc(io, fx.component, allocator);
    const real_pre = try Io.Dir.cwd().realPathFileAlloc(io, fx.preopens, allocator);
    const real_spec = try Io.Dir.cwd().realPathFileAlloc(io, fx.spec, allocator);
    try std.testing.expectEqualStrings(real_comp, config.component);
    try std.testing.expectEqualStrings(real_pre, config.preopens_file);
    try std.testing.expectEqualStrings(real_spec, config.mounts[0].host);
    try std.testing.expectEqualStrings("0x400000", config.perf.base.?);
}

test "checked in schema and example track parser version" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const schema_bytes = try Io.Dir.cwd().readFileAlloc(
        io,
        "tests/benchmarks/keyvault/manifest.schema.json",
        allocator,
        .limited(1024 * 1024),
    );
    const example_bytes = try Io.Dir.cwd().readFileAlloc(
        io,
        "tests/benchmarks/keyvault/manifest.example.json",
        allocator,
        .limited(1024 * 1024),
    );
    const schema = try json.parseFromSliceLeaky(Value, allocator, schema_bytes, .{});
    const example = try json.parseFromSliceLeaky(Value, allocator, example_bytes, .{});
    const schema_const = schema.object.get("properties").?.object.get("schema_version").?.object.get("const").?.integer;
    try std.testing.expectEqual(@as(i64, 1), schema_const);
    try std.testing.expectEqual(SCHEMA_VERSION, example.object.get("schema_version").?.integer);
    try std.testing.expectEqualStrings(
        "fe08b02fa85209855850ba1ea72bc3ab554cec961fdee005cacb6348d9da8c92",
        example.object.get("workload").?.object.get("component").?.object.get("sha256").?.string,
    );
}

test "command construction precompiles then runs artifacts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "commands");
    defer fx.deinit(io);

    var diag: Diagnostic = .{};
    const config = try loadManifest(allocator, io, fx.manifest, &diag);
    const artifacts = try std.fs.path.join(allocator, &.{ fx.scratch, "artifacts" });
    const cmds = try precompileCommands(allocator, config, artifacts);

    try std.testing.expectEqualStrings("compile-component", cmds.wamr[1]);
    try std.testing.expectEqualStrings("--target=x86_64", cmds.wamr[2]);
    try std.testing.expectEqualStrings(config.component, cmds.wamr[3]);
    try std.testing.expectEqualStrings("-o", cmds.wasmtime[cmds.wasmtime.len - 2]);
    try std.testing.expectEqualStrings(cmds.wasmtime_cwasm, cmds.wasmtime[cmds.wasmtime.len - 1]);

    const out = try std.fs.path.join(allocator, &.{ fx.scratch, "out" });
    const wamr_run = try runtimeCommand(
        allocator,
        io,
        config,
        "wamr",
        out,
        cmds.wamr_manifest,
        cmds.wasmtime_cwasm,
        &diag,
    );
    const wasmtime_run = try runtimeCommand(
        allocator,
        io,
        config,
        "wasmtime",
        out,
        cmds.wamr_manifest,
        cmds.wasmtime_cwasm,
        &diag,
    );

    const wamr_joined = try std.mem.join(allocator, " ", wamr_run);
    const wasmtime_joined = try std.mem.join(allocator, " ", wasmtime_run);
    try std.testing.expect(std.mem.indexOf(u8, wamr_joined, "--precompiled-manifest") != null);
    try std.testing.expect(std.mem.indexOf(u8, wamr_joined, cmds.wamr_manifest) != null);
    try std.testing.expect(std.mem.indexOf(u8, wamr_joined, "WAMR_KEYVAULT_HARNESS=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, wamr_joined, "compile-component") == null);
    try std.testing.expect(std.mem.indexOf(u8, wasmtime_joined, "--allow-precompiled") != null);
    try std.testing.expect(std.mem.indexOf(u8, wasmtime_joined, cmds.wasmtime_cwasm) != null);
    try std.testing.expect(std.mem.indexOf(u8, wasmtime_joined, "WAMR_KEYVAULT_HARNESS=1") != null);
}

test "timing and tcgc output parsing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var diag: Diagnostic = .{};
    const resp = try parseTcgcResponseBytes(
        allocator,
        "calling tcgc.compile...\ngot 58187 bytes back from tcgc.compile\n",
        &diag,
    );
    try std.testing.expectEqual(@as(i64, 58187), resp);
    try std.testing.expectError(error.HarnessFailed, parseTcgcResponseBytes(allocator, "no response marker", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "exactly one") != null);

    var records = Array.init(allocator);
    for ([_]f64{ 5.1, 5.2, 5.3, 5.4, 5.5 }) |v| {
        var r: ObjectMap = .empty;
        try r.put(allocator, "runtime", .{ .string = "wamr" });
        try r.put(allocator, "phase", .{ .string = "measure" });
        try r.put(allocator, "elapsed_seconds", .{ .float = v });
        try records.append(.{ .object = r });
    }
    const samples = try parseTimingSamples(allocator, records.items, "wamr", &diag);
    try std.testing.expectApproxEqAbs(@as(f64, 5.3), samples[2], 1e-9);
    try std.testing.expectError(error.HarnessFailed, parseTimingSamples(allocator, records.items[0..4], "wamr", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "at least 5") != null);
}

test "equivalence detects 58187 58188 and file hash drift" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var f_obj: ObjectMap = .empty;
    try f_obj.put(allocator, "path", .{ .string = "a.zig" });
    try f_obj.put(allocator, "size", .{ .integer = 4 });
    try f_obj.put(allocator, "sha256", .{ .string = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" });
    var files_arr = Array.init(allocator);
    try files_arr.append(.{ .object = f_obj });

    var out_obj: ObjectMap = .empty;
    try out_obj.put(allocator, "sha256_tree", .{ .string = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" });
    try out_obj.put(allocator, "file_count", .{ .integer = 1 });
    try out_obj.put(allocator, "total_bytes", .{ .integer = 4 });
    try out_obj.put(allocator, "files", .{ .array = files_arr });

    var base: ObjectMap = .empty;
    try base.put(allocator, "runtime", .{ .string = "wamr" });
    try base.put(allocator, "phase", .{ .string = "measure" });
    try base.put(allocator, "index", .{ .integer = 1 });
    try base.put(allocator, "tcgc_response_bytes", .{ .integer = 58187 });
    try base.put(allocator, "output", .{ .object = out_obj });

    var drift_bytes = try base.clone(allocator);
    try drift_bytes.put(allocator, "runtime", .{ .string = "wasmtime" });
    try drift_bytes.put(allocator, "tcgc_response_bytes", .{ .integer = 58188 });

    var diag: Diagnostic = .{};
    try std.testing.expectError(
        error.HarnessFailed,
        verifyEquivalence(allocator, &.{ .{ .object = base }, .{ .object = drift_bytes } }, &diag),
    );
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "response-size drift") != null);

    var changed_out = try out_obj.clone(allocator);
    try changed_out.put(allocator, "total_bytes", .{ .integer = 5 });
    var drift_out = try base.clone(allocator);
    try drift_out.put(allocator, "runtime", .{ .string = "wasmtime" });
    try drift_out.put(allocator, "output", .{ .object = changed_out });
    try std.testing.expectError(
        error.HarnessFailed,
        verifyEquivalence(allocator, &.{ .{ .object = base }, .{ .object = drift_out } }, &diag),
    );
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "generated output mismatch") != null);
}

test "missing prerequisite is actionable" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "missing-prereq");
    defer fx.deinit(io);

    var diag: Diagnostic = .{};
    const config = try loadManifest(allocator, io, fx.manifest, &diag);
    try Io.Dir.cwd().deleteFile(io, fx.tool);
    try std.testing.expectError(error.HarnessFailed, validateConfig(allocator, io, config, .{}, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "required wamr binary is absent") != null);
}

test "profile mode never falls back when unsupported" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "profile-unsupported");
    defer fx.deinit(io);

    var diag: Diagnostic = .{};
    const config = try loadManifest(allocator, io, fx.manifest, &diag);
    const hooks: Hooks = .{ .system_name = "Darwin" };
    try std.testing.expectError(
        error.HarnessFailed,
        runPerf(
            allocator,
            io,
            config,
            fx.scratch,
            "manifest.cwasm.json",
            "wasmtime.cwasm",
            10.0,
            .null,
            hooks,
            &diag,
        ),
    );
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "requires Linux x86_64") != null);
}

fn buildSyntheticCoreAndSidecar(
    allocator: Allocator,
    code: []const u8,
    local_func: i64,
) !struct { cwasm_bytes: []const u8, sidecar_json: Value } {
    var target_payload: [40]u8 = @splat(0);
    std.mem.writeInt(u16, target_payload[0..2], 2, .little);
    std.mem.writeInt(u16, target_payload[4..6], 1, .little);
    std.mem.writeInt(u16, target_payload[6..8], 0x3E, .little);
    @memcpy(target_payload[16..22], "x86_64");

    var funcs_payload: [12]u8 = undefined;
    std.mem.writeInt(u32, funcs_payload[0..4], 1, .little);
    std.mem.writeInt(u32, funcs_payload[4..8], 0, .little);
    std.mem.writeInt(u32, funcs_payload[8..12], 0, .little);

    var cwasm: std.ArrayList(u8) = .empty;
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], aot_jit_attr.AOT_MAGIC, .little);
    std.mem.writeInt(u32, hdr[4..8], aot_jit_attr.AOT_VERSION, .little);
    try cwasm.appendSlice(allocator, &hdr);

    const Sec = struct { id: u32, payload: []const u8 };
    for ([_]Sec{
        .{ .id = aot_jit_attr.SEC_TARGET_INFO, .payload = &target_payload },
        .{ .id = aot_jit_attr.SEC_TEXT, .payload = code },
        .{ .id = aot_jit_attr.SEC_FUNCTION, .payload = &funcs_payload },
    }) |sec| {
        var sh: [8]u8 = undefined;
        std.mem.writeInt(u32, sh[0..4], sec.id, .little);
        std.mem.writeInt(u32, sh[4..8], @intCast(sec.payload.len), .little);
        try cwasm.appendSlice(allocator, &sh);
        try cwasm.appendSlice(allocator, sec.payload);
    }

    const code_sha = try sha256BytesHex(allocator, code);
    const parsed_meta = try json.parseFromSliceLeaky(
        Value,
        allocator,
        try std.fmt.allocPrint(
            allocator,
            \\{{
            \\  "schema": "wamr-aot-frame-attribution-v1",
            \\  "target": "x86_64",
            \\  "abi": "sysv",
            \\  "module": 4,
            \\  "local_func": {d},
            \\  "code_size": {d},
            \\  "module_text_size": {d},
            \\  "module_text_sha256": "{s}",
            \\  "normalized_code_sha256": "{s}",
            \\  "normalized_relocations": [],
            \\  "spill_metric": {{"spill_ld": 1, "spill_st": 1}},
            \\  "frame_regions": [{{"name": "spill_area", "base_reg": "rbp", "min_disp": -56, "max_disp": -49}}],
            \\  "slots": [{{
            \\    "slot_id": "spill:0", "origin": "allocator_spill", "region": "spill_area",
            \\    "base_reg": "rbp", "disp": -56, "size": 8,
            \\    "contributors": [{{"vreg": 3, "type": "i64", "source_kind": "ssa_value", "local_index": null, "ir_op": "add"}}]
            \\  }}],
            \\  "allocator_values": [{{
            \\    "vreg": 3, "slot_id": "spill:0", "disp": -56, "size": 8, "type": "i64",
            \\    "source_kind": "ssa_value", "local_index": null, "ir_op": "add",
            \\    "spill_loads": 1, "spill_stores": 1
            \\  }}],
            \\  "accesses": [
            \\    {{
            \\      "start_offset": 0, "end_offset": 4, "direction": "load", "width": 8,
            \\      "addressing_mode": "base_disp", "base_reg": "rbp", "disp": -56,
            \\      "slot_id": "spill:0", "origin": "allocator_spill",
            \\      "vreg": 3, "source_kind": "ssa_value", "local_index": null, "ir_op": "add"
            \\    }},
            \\    {{
            \\      "start_offset": 4, "end_offset": 8, "direction": "store", "width": 8,
            \\      "addressing_mode": "base_disp", "base_reg": "rbp", "disp": -56,
            \\      "slot_id": "spill:0", "origin": "allocator_spill",
            \\      "vreg": 3, "source_kind": "ssa_value", "local_index": null, "ir_op": "add"
            \\    }}
            \\  ]
            \\}}
        ,
            .{ local_func, code.len, code.len, code_sha, code_sha },
        ),
        .{},
    );
    return .{
        .cwasm_bytes = try cwasm.toOwnedSlice(allocator),
        .sidecar_json = parsed_meta,
    };
}

test "frame artifact binds exact core and fails closed" {
    if (builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "frame-artifact");
    defer fx.deinit(io);

    var diag: Diagnostic = .{};
    var config = try loadManifest(allocator, io, fx.manifest, &diag);
    const artifacts = try std.fs.path.join(allocator, &.{ fx.scratch, "artifacts" });
    try Io.Dir.cwd().createDirPath(io, artifacts);

    try std.testing.expectError(
        error.HarnessFailed,
        validateFrameArtifact(allocator, io, config, artifacts, 10.0, .{}, &diag),
    );
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "exact core and sidecar") != null);

    const code = [_]u8{ 0x48, 0x8b, 0x45, 0xc8, 0x48, 0x89, 0x45, 0xc8 };
    const core = try std.fs.path.join(allocator, &.{ artifacts, "keyvault.4.cwasm" });
    const built_bad = try buildSyntheticCoreAndSidecar(allocator, &code, config.perf.hot_func.?);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = core, .data = built_bad.cwasm_bytes });

    const sidecar = try frameMetadataPath(allocator, config, artifacts);
    try Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(sidecar).?);
    var bad_buf: Io.Writer.Allocating = .init(allocator);
    try aot_jit_attr.writeCanonicalJson(&bad_buf.writer, built_bad.sidecar_json);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = sidecar, .data = bad_buf.written() });

    try std.testing.expectError(
        error.HarnessFailed,
        validateFrameArtifact(allocator, io, config, artifacts, 10.0, .{}, &diag),
    );
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "exact-image frame metadata validation") != null);

    config.perf.hot_func = 0;
    const built_good = try buildSyntheticCoreAndSidecar(allocator, &code, 0);
    const new_sidecar = try frameMetadataPath(allocator, config, artifacts);
    var good_buf: Io.Writer.Allocating = .init(allocator);
    try aot_jit_attr.writeCanonicalJson(&good_buf.writer, built_good.sidecar_json);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = new_sidecar, .data = good_buf.written() });

    const validated = try validateFrameArtifact(allocator, io, config, artifacts, 10.0, .{}, &diag);
    try std.testing.expectEqualStrings(
        try sha256File(allocator, io, core, &diag),
        validated.object.get("cwasm_sha256").?.string,
    );
    try std.testing.expectEqualStrings(
        try sha256File(allocator, io, new_sidecar, &diag),
        validated.object.get("sidecar_sha256").?.string,
    );

    const nops = [_]u8{ 0x90, 0x90, 0x90, 0x90, 0x48, 0x89, 0x45, 0xc8 };
    const built_corrupt = try buildSyntheticCoreAndSidecar(allocator, &nops, 0);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = core, .data = built_corrupt.cwasm_bytes });
    try std.testing.expectError(
        error.HarnessFailed,
        validateFrameArtifact(allocator, io, config, artifacts, 10.0, .{}, &diag),
    );
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "exact-image frame metadata validation") != null);
}

test "profile precompile requires selected sidecar" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "profile-precompile");
    defer fx.deinit(io);

    var diag: Diagnostic = .{};
    var config = try loadManifest(allocator, io, fx.manifest, &diag);
    const artifacts = try std.fs.path.join(allocator, &.{ fx.scratch, "artifacts" });

    const FakeCompileCtx = struct {
        artifacts: []const u8,
        fn run(
            raw_ctx: ?*anyopaque,
            alloc: Allocator,
            call_io: Io,
            _: []const []const u8,
            _: f64,
            label: []const u8,
            extra_env: ?[]const EnvEntry,
            _: *Diagnostic,
        ) HarnessError!CompletedProcess {
            const self: *const @This() = @ptrCast(@alignCast(raw_ctx.?));
            if (std.mem.eql(u8, label, "WAMR precompile")) {
                const expected_prefix = std.fs.path.join(alloc, &.{ self.artifacts, "frame", "keyvault" }) catch return error.OutOfMemory;
                var found = false;
                if (extra_env) |entries| {
                    for (entries) |e| {
                        if (std.mem.eql(u8, e.key, "WAMR_AOT_FRAME_ATTRIBUTION") and std.mem.eql(u8, e.value, expected_prefix)) {
                            found = true;
                        }
                    }
                }
                if (!found) return error.HarnessFailed;
                const m_path = std.fs.path.join(alloc, &.{ self.artifacts, "keyvault.cwasm.json" }) catch return error.OutOfMemory;
                const c_path = std.fs.path.join(alloc, &.{ self.artifacts, "keyvault.4.cwasm" }) catch return error.OutOfMemory;
                Io.Dir.cwd().writeFile(call_io, .{ .sub_path = m_path, .data = "{}" }) catch return error.HarnessFailed;
                Io.Dir.cwd().writeFile(call_io, .{ .sub_path = c_path, .data = "core" }) catch return error.HarnessFailed;
            } else if (std.mem.eql(u8, label, "Wasmtime precompile")) {
                const w_path = std.fs.path.join(alloc, &.{ self.artifacts, "keyvault.wasmtime.cwasm" }) catch return error.OutOfMemory;
                Io.Dir.cwd().writeFile(call_io, .{ .sub_path = w_path, .data = "wasmtime" }) catch return error.HarnessFailed;
            }
            return .{};
        }
    };
    var fake_ctx: FakeCompileCtx = .{ .artifacts = artifacts };
    const hooks: Hooks = .{ .ctx = &fake_ctx, .run_checked_fn = FakeCompileCtx.run };
    try std.testing.expectError(
        error.HarnessFailed,
        precompile(allocator, io, config, artifacts, 10.0, true, hooks, &diag),
    );
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "exact core and sidecar") != null);

    config.perf.hot_func = null;
    try std.testing.expectError(
        error.HarnessFailed,
        precompile(allocator, io, config, artifacts, 10.0, true, hooks, &diag),
    );
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "perf.hot_func") != null);
}

test "profile reports matching sidecar and rejects wrong identity" {
    if (builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "profile-identity");
    defer fx.deinit(io);

    var diag: Diagnostic = .{};
    var config = try loadManifest(allocator, io, fx.manifest, &diag);
    config.perf.hot_func = 0;
    const artifacts = try std.fs.path.join(allocator, &.{ fx.scratch, "artifacts" });
    try Io.Dir.cwd().createDirPath(io, artifacts);

    const code = [_]u8{ 0x48, 0x8b, 0x45, 0xc8, 0x48, 0x89, 0x45, 0xc8 };
    const core = try std.fs.path.join(allocator, &.{ artifacts, "keyvault.4.cwasm" });
    const built = try buildSyntheticCoreAndSidecar(allocator, &code, 0);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = core, .data = built.cwasm_bytes });

    const sidecar = try frameMetadataPath(allocator, config, artifacts);
    try Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(sidecar).?);
    var s_buf: Io.Writer.Allocating = .init(allocator);
    try aot_jit_attr.writeCanonicalJson(&s_buf.writer, built.sidecar_json);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = sidecar, .data = s_buf.written() });

    const PerfMockCtx = struct {
        scratch: []const u8,
        core: []const u8,
        sidecar: []const u8,
        module_sha: []const u8,
        norm_sha: []const u8,
        wrong_identity: bool = false,

        fn run(
            raw_ctx: ?*anyopaque,
            alloc: Allocator,
            call_io: Io,
            command: []const []const u8,
            timeout_seconds: f64,
            label: []const u8,
            extra_env: ?[]const EnvEntry,
            call_diag: *Diagnostic,
        ) HarnessError!CompletedProcess {
            const self: *const @This() = @ptrCast(@alignCast(raw_ctx.?));
            if (std.mem.eql(u8, label, "exact-image frame metadata validation")) {
                return defaultRunChecked(alloc, call_io, command, timeout_seconds, label, extra_env, call_diag);
            }
            if (std.mem.eql(u8, label, "WAMR perf capture")) {
                const gen_file = std.fs.path.join(alloc, &.{ self.scratch, "perf", "output", "generated.zig" }) catch return error.OutOfMemory;
                Io.Dir.cwd().writeFile(call_io, .{ .sub_path = gen_file, .data = "same\n" }) catch return error.HarnessFailed;
                return .{ .stdout = "got 58187 bytes back from tcgc.compile" };
            }
            var json_out: ?[]const u8 = null;
            for (command, 0..) |arg, idx| {
                if (std.mem.eql(u8, arg, "--json-out") and idx + 1 < command.len) {
                    json_out = command[idx + 1];
                }
            }
            const norm = if (self.wrong_identity)
                "0000000000000000000000000000000000000000000000000000000000000000"
            else
                self.norm_sha;
            const rep_text = std.fmt.allocPrint(
                alloc,
                \\{{
                \\  "cwasm": "{s}",
                \\  "attribution_coverage_pct": 100,
                \\  "classified_function": {{
                \\    "local_func": 0,
                \\    "frame_metadata": "{s}",
                \\    "frame_metadata_module": 4,
                \\    "frame_metadata_identity": {{
                \\      "module_text_sha256": "{s}",
                \\      "normalized_code_sha256": "{s}"
                \\    }},
                \\    "frame_attribution": {{
                \\      "coverage": {{"origin_sample_coverage_pct": 100, "unknown_frame_samples": 0}},
                \\      "origins": {{"allocator_spill": {{"samples": 10}}}}
                \\    }}
                \\  }}
                \\}}
            ,
                .{ self.core, self.sidecar, self.module_sha, norm },
            ) catch return error.OutOfMemory;
            Io.Dir.cwd().writeFile(call_io, .{ .sub_path = json_out.?, .data = rep_text }) catch return error.HarnessFailed;
            return .{};
        }
    };

    const code_sha = try sha256BytesHex(allocator, &code);
    var mock_ctx: PerfMockCtx = .{
        .scratch = fx.scratch,
        .core = core,
        .sidecar = sidecar,
        .module_sha = code_sha,
        .norm_sha = code_sha,
    };

    const ref_dir = try std.fs.path.join(allocator, &.{ fx.scratch, "reference" });
    try Io.Dir.cwd().createDirPath(io, ref_dir);
    const ref_file = try std.fs.path.join(allocator, &.{ ref_dir, "generated.zig" });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = ref_file, .data = "same\n" });
    const ref_snap = try treeSnapshot(allocator, io, ref_dir, &diag);
    const reference = try ref_snap.toOutputJsonValue(allocator);

    const hooks: Hooks = .{
        .ctx = &mock_ctx,
        .paranoid_override = 2,
        .which_override = "/usr/bin/tool",
        .run_checked_fn = PerfMockCtx.run,
    };
    const report = try runPerf(
        allocator,
        io,
        config,
        fx.scratch,
        try std.fs.path.join(allocator, &.{ artifacts, "keyvault.cwasm.json" }),
        try std.fs.path.join(allocator, &.{ artifacts, "keyvault.wasmtime.cwasm" }),
        10.0,
        reference,
        hooks,
        &diag,
    );
    try std.testing.expectEqualStrings(
        try sha256File(allocator, io, sidecar, &diag),
        report.object.get("frame_attribution").?.object.get("sidecar_sha256").?.string,
    );
    try std.testing.expect(
        report.object.get("attribution").?.object.get("classified_function").?.object.contains("frame_attribution"),
    );

    mock_ctx.wrong_identity = true;
    try std.testing.expectError(
        error.HarnessFailed,
        runPerf(
            allocator,
            io,
            config,
            fx.scratch,
            try std.fs.path.join(allocator, &.{ artifacts, "keyvault.cwasm.json" }),
            try std.fs.path.join(allocator, &.{ artifacts, "keyvault.wasmtime.cwasm" }),
            10.0,
            reference,
            hooks,
            &diag,
        ),
    );
    try std.testing.expect(std.mem.indexOf(u8, diag.message, "exact selected frame sidecar") != null);
}

test "controlled end to end smoke" {
    if (builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const fx = try TestFixture.init(allocator, io, "e2e-smoke");
    defer fx.deinit(io);

    try Io.Dir.cwd().writeFile(io, .{
        .sub_path = fx.tool,
        .data =
        \\#!/bin/sh
        \\set -eu
        \\case "${1:-}" in
        \\  version|--version) echo pinned-tool-1.0; exit 0 ;;
        \\  compile-component)
        \\    while [ "$#" -gt 0 ]; do
        \\      if [ "$1" = -o ]; then shift; out="$1"; break; fi
        \\      shift
        \\    done
        \\    printf '{"modules":[]}' > "$out"
        \\    stem=${out%.cwasm.json}
        \\    printf core > "${stem}.4.cwasm"
        \\    exit 0 ;;
        \\  compile)
        \\    while [ "$#" -gt 0 ]; do
        \\      if [ "$1" = -o ]; then shift; out="$1"; break; fi
        \\      shift
        \\    done
        \\    printf wasmtime > "$out"
        \\    exit 0 ;;
        \\  run)
        \\    out=''
        \\    while [ "$#" -gt 0 ]; do
        \\      case "$1" in
        \\        *::/out) out=${1%::/out} ;;
        \\      esac
        \\      shift
        \\    done
        \\    mkdir -p "$out"
        \\    printf 'pub const generated = true;\n' > "$out/generated.zig"
        \\    echo 'got 58187 bytes back from tcgc.compile'
        \\    exit 0 ;;
        \\esac
        \\exit 3
        \\
        ,
        .flags = .{ .permissions = .fromMode(0o755) },
    });

    var diag: Diagnostic = .{};
    var doc = try fx.manifestDocument(allocator, io);
    const current_sha = try sha256File(allocator, io, fx.tool, &diag);
    for ([_][]const u8{ "wamr", "wamrc", "wasmtime" }) |t_name| {
        try doc.object.getPtr("tools").?.object.getPtr(t_name).?.object.put(allocator, "sha256", .{ .string = current_sha });
    }
    try fx.writeManifest(allocator, io, doc);

    const report_json = try std.fs.path.join(allocator, &.{ fx.scratch, "report.json" });
    const report_markdown = try std.fs.path.join(allocator, &.{ fx.scratch, "report.md" });
    const work_dir = try std.fs.path.join(allocator, &.{ fx.scratch, "work" });

    const report = try execute(allocator, io, .{
        .manifest = fx.manifest,
        .work_dir = work_dir,
        .warmups = 1,
        .runs = 5,
        .compile_timeout = 10.0,
        .run_timeout = 10.0,
        .profile = false,
        .report_json = report_json,
        .report_markdown = report_markdown,
    }, .{}, &diag);

    const timing = report.object.get("timing").?.object;
    try std.testing.expectEqual(@as(i64, 5), timing.get("wamr").?.object.get("runs").?.integer);
    try std.testing.expectEqual(@as(i64, 5), timing.get("wasmtime").?.object.get("runs").?.integer);
    try std.testing.expect(report.object.get("equivalence").?.object.get("cross_runtime_exact_match").?.bool);
    try std.testing.expect(isRegularFile(io, report_json));
    const md_text = try Io.Dir.cwd().readFileAlloc(io, report_markdown, allocator, .limited(1024 * 1024));
    try std.testing.expect(std.mem.indexOf(u8, md_text, "Wasmtime-time/WAMR-time ratio") != null);
}
