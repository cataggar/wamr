const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

const benchmark_optimize = @import("benchmark_optimize.zig");
const benchmark_process = @import("benchmark_process.zig");
const decimal_format = @import("decimal_format.zig");
const native_benchmark = @import("native_benchmark.zig");

pub const validation_text = "Correct operation validated.";
pub const performance_marker = "2K performance run parameters for coremark.";
pub const coremark_guest_args_default = [_][]const u8{ "0", "0", "0", "400000", "0" };
pub const expected_iterations: i64 = 400_000;
pub const default_fixture_rel = "tests/benchmarks/coremark/coremark_wasi.wasm";
pub const default_fixture_sha256 = "f4b7591296ead10264e0f101f355bdf848865c31329325594e66fbabefec235b";
pub const pinned_wasmtime_version = "44.0.1";
pub const report_schema_version: i64 = 2;
pub const report_kind = "coremark-authoritative-comparison";
pub const handoff_schema_version: i64 = 2;
pub const trap_retry_max: usize = 5;

pub const ProfileCounts = struct {
    warmups: usize,
    runs: usize,
};

pub fn profileCounts(profile: []const u8) ?ProfileCounts {
    if (std.mem.eql(u8, profile, "authoritative")) return .{ .warmups = 2, .runs = 10 };
    if (std.mem.eql(u8, profile, "ci")) return .{ .warmups = 0, .runs = 3 };
    return null;
}

pub fn profileIterations(profile: []const u8) ?i64 {
    if (std.mem.eql(u8, profile, "authoritative") or std.mem.eql(u8, profile, "ci")) {
        return expected_iterations;
    }
    return null;
}

pub const PinnedWasmtimeAsset = struct {
    os_name: []const u8,
    arch: []const u8,
    asset_name: []const u8,
    archive_sha256: []const u8,
    binary_sha256: []const u8,
};

pub const pinned_wasmtime_assets = [_]PinnedWasmtimeAsset{
    .{
        .os_name = "linux",
        .arch = "x86_64",
        .asset_name = "wasmtime-v44.0.1-x86_64-linux.tar.xz",
        .archive_sha256 = "afd58715f105e3a7f454169daed22168c5736ec5f225fb04c4ac62c54c9508a3",
        .binary_sha256 = "47e59a672f6a35ad071e3fbe7c0c541dcaea77b085d7aada4fd1a9ac77a0d601",
    },
    .{
        .os_name = "linux",
        .arch = "aarch64",
        .asset_name = "wasmtime-v44.0.1-aarch64-linux.tar.xz",
        .archive_sha256 = "9ec1e606c541099a7e8266865494e190a6f928abb3a2fb7bb4195fd2e8499d2b",
        .binary_sha256 = "ad27d96d24d193ff32665a59552699e3f5355baeaf7220fb4cd88f023694e93d",
    },
};

pub const ParsedCoreMark = struct {
    throughput: f64,
    iterations: i64,
};

pub const SampleRecord = struct {
    engine_key: []const u8,
    engine: []const u8,
    phase: []const u8,
    engine_ordinal: usize,
    schedule_position: usize,
    started_at: []const u8,
    completed_at: []const u8,
    elapsed_seconds: f64,
    iterations_per_second: f64,
    iterations: i64,

    pub fn toJsonValue(self: SampleRecord, allocator: Allocator) !Value {
        var obj: std.json.ObjectMap = .empty;
        try obj.put(allocator, "completed_at", .{ .string = self.completed_at });
        try obj.put(allocator, "elapsed_seconds", .{ .float = self.elapsed_seconds });
        try obj.put(allocator, "engine", .{ .string = self.engine });
        try obj.put(allocator, "engine_key", .{ .string = self.engine_key });
        try obj.put(allocator, "engine_ordinal", .{ .integer = @intCast(self.engine_ordinal) });
        try obj.put(allocator, "iterations", .{ .integer = self.iterations });
        try obj.put(allocator, "iterations_per_second", .{ .float = self.iterations_per_second });
        try obj.put(allocator, "phase", .{ .string = self.phase });
        try obj.put(allocator, "schedule_position", .{ .integer = @intCast(self.schedule_position) });
        try obj.put(allocator, "started_at", .{ .string = self.started_at });
        return .{ .object = obj };
    }
};

pub const EngineResult = struct {
    engine: []const u8,
    version: []const u8,
    optimize: []const u8,
    values: []const f64,
    samples: []const SampleRecord = &.{},
    identity: Value = .{ .object = .empty },
};

pub const PreparedEngine = struct {
    key: []const u8,
    engine: []const u8,
    version: []const u8,
    optimize: []const u8,
    cmd: []const []const u8,
    cwd: []const u8,
    env: ?*const std.process.Environ.Map = null,
    expected_iterations: i64 = expected_iterations,
    retry_wamr_flake: bool = false,
    identity: Value = .{ .object = .empty },
};

pub const AffinityInfo = struct {
    allowed_cpus: []const usize,
    selected_cpu: usize,
    taskset: []const u8,
};

pub const HostIdentity = struct {
    arch: []const u8,
    cpu_count: usize,
    cpu_model: []const u8,
    runner_name: []const u8,
    boot_id: []const u8,

    pub fn fingerprint(self: HostIdentity, allocator: Allocator) ![]u8 {
        const joined = try std.fmt.allocPrint(
            allocator,
            "{s}\x00{d}\x00{s}\x00{s}\x00{s}",
            .{ self.arch, self.cpu_count, self.cpu_model, self.runner_name, self.boot_id },
        );
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(joined, &digest, .{});
        const full_hex = std.fmt.bytesToHex(digest, .lower);
        return allocator.dupe(u8, full_hex[0..16]);
    }

    pub fn eql(self: HostIdentity, other: HostIdentity) bool {
        return std.mem.eql(u8, self.arch, other.arch) and
            self.cpu_count == other.cpu_count and
            std.mem.eql(u8, self.cpu_model, other.cpu_model) and
            std.mem.eql(u8, self.runner_name, other.runner_name) and
            std.mem.eql(u8, self.boot_id, other.boot_id);
    }
};

pub const CommandRunner = *const fn (
    ctx: ?*anyopaque,
    allocator: Allocator,
    io: std.Io,
    cmd: []const []const u8,
    cwd: ?[]const u8,
    env: ?*const std.process.Environ.Map,
) anyerror![]u8;

pub fn defaultRunCommand(
    _: ?*anyopaque,
    allocator: Allocator,
    io: std.Io,
    cmd: []const []const u8,
    cwd: ?[]const u8,
    env: ?*const std.process.Environ.Map,
) ![]u8 {
    return run(allocator, io, cmd, cwd, env);
}

var never_interrupted: std.atomic.Value(bool) = .init(false);

pub fn run(
    allocator: Allocator,
    io: std.Io,
    cmd: []const []const u8,
    cwd: ?[]const u8,
    env: ?*const std.process.Environ.Map,
) ![]u8 {
    return benchmark_process.run(allocator, io, cwd orelse ".", env, cmd, &never_interrupted);
}

pub fn sha256BytesAlloc(allocator: Allocator, bytes: []const u8) ![]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

pub fn sha256File(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch {
        return error.FileNotFound;
    };
    return sha256BytesAlloc(allocator, bytes);
}

pub fn isLowerHex(text: []const u8, expected_len: usize) bool {
    if (text.len != expected_len) return false;
    for (text) |ch| {
        const ok = (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f');
        if (!ok) return false;
    }
    return true;
}

pub fn requireDigestValue(val: ?Value, expected_len: usize) ![]const u8 {
    const v = val orelse return error.InvalidDigest;
    if (v != .string) return error.InvalidDigest;
    return requireDigest(v.string, expected_len);
}

pub fn requireDigest(text: []const u8, expected_len: usize) ![]const u8 {
    if (!isLowerHex(text, expected_len)) return error.InvalidDigest;
    return text;
}

pub fn makeWamrIdentity(
    allocator: Allocator,
    io: std.Io,
    opts: struct {
        ref: []const u8,
        source_sha: []const u8,
        optimize: []const u8,
        runtime_path: []const u8,
        compiler_path: []const u8,
        module_path: []const u8,
        simd_runner_path: ?[]const u8 = null,
    },
) !Value {
    _ = try requireDigest(opts.source_sha, 40);
    var source_obj: std.json.ObjectMap = .empty;
    try source_obj.put(allocator, "ref", .{ .string = try allocator.dupe(u8, opts.ref) });
    try source_obj.put(allocator, "sha", .{ .string = try allocator.dupe(u8, opts.source_sha) });

    var runtime_obj: std.json.ObjectMap = .empty;
    try runtime_obj.put(allocator, "name", .{ .string = "wamr" });
    try runtime_obj.put(allocator, "path", .{ .string = try allocator.dupe(u8, opts.runtime_path) });
    try runtime_obj.put(allocator, "sha256", .{ .string = try sha256File(allocator, io, opts.runtime_path) });

    var compiler_obj: std.json.ObjectMap = .empty;
    try compiler_obj.put(allocator, "name", .{ .string = "wamrc" });
    try compiler_obj.put(allocator, "path", .{ .string = try allocator.dupe(u8, opts.compiler_path) });
    try compiler_obj.put(allocator, "sha256", .{ .string = try sha256File(allocator, io, opts.compiler_path) });

    var module_obj: std.json.ObjectMap = .empty;
    try module_obj.put(allocator, "format", .{ .string = "cwasm" });
    try module_obj.put(allocator, "path", .{ .string = try allocator.dupe(u8, opts.module_path) });
    try module_obj.put(allocator, "sha256", .{ .string = try sha256File(allocator, io, opts.module_path) });

    var id_obj: std.json.ObjectMap = .empty;
    try id_obj.put(allocator, "type", .{ .string = "wamr" });
    try id_obj.put(allocator, "source", .{ .object = source_obj });
    try id_obj.put(allocator, "optimize", .{ .string = try allocator.dupe(u8, opts.optimize) });
    try id_obj.put(allocator, "runtime", .{ .object = runtime_obj });
    try id_obj.put(allocator, "compiler", .{ .object = compiler_obj });
    try id_obj.put(allocator, "module", .{ .object = module_obj });

    if (opts.simd_runner_path) |simd_path| {
        var simd_obj: std.json.ObjectMap = .empty;
        try simd_obj.put(allocator, "name", .{ .string = "simd-bench-runner" });
        try simd_obj.put(allocator, "path", .{ .string = try allocator.dupe(u8, simd_path) });
        try simd_obj.put(allocator, "sha256", .{ .string = try sha256File(allocator, io, simd_path) });
        try id_obj.put(allocator, "simd_runner", .{ .object = simd_obj });
    }
    return .{ .object = id_obj };
}

pub fn makeWasmtimeIdentity(
    allocator: Allocator,
    io: std.Io,
    opts: struct {
        version: []const u8,
        runtime_path: []const u8,
        channel: []const u8,
    },
) !Value {
    var runtime_obj: std.json.ObjectMap = .empty;
    try runtime_obj.put(allocator, "name", .{ .string = "wasmtime" });
    try runtime_obj.put(allocator, "path", .{ .string = try allocator.dupe(u8, opts.runtime_path) });
    try runtime_obj.put(allocator, "sha256", .{ .string = try sha256File(allocator, io, opts.runtime_path) });

    var id_obj: std.json.ObjectMap = .empty;
    try id_obj.put(allocator, "type", .{ .string = "wasmtime" });
    try id_obj.put(allocator, "channel", .{ .string = try allocator.dupe(u8, opts.channel) });
    try id_obj.put(allocator, "version", .{ .string = try allocator.dupe(u8, opts.version) });
    try id_obj.put(allocator, "runtime", .{ .object = runtime_obj });
    return .{ .object = id_obj };
}

pub fn validateEngineIdentity(identity_val: ?Value, role: []const u8) !std.json.ObjectMap {
    _ = role;
    const identity = identity_val orelse return error.MissingEngineIdentity;
    if (identity != .object) return error.MissingEngineIdentity;
    const obj = identity.object;
    const id_type_val = obj.get("type") orelse return error.UnsupportedEngineIdentityType;
    if (id_type_val != .string) return error.UnsupportedEngineIdentityType;
    const runtime_val = obj.get("runtime") orelse return error.MissingRuntimeIdentity;
    if (runtime_val != .object) return error.MissingRuntimeIdentity;
    _ = try requireDigestValue(runtime_val.object.get("sha256"), 64);

    if (std.mem.eql(u8, id_type_val.string, "wamr")) {
        const source_val = obj.get("source") orelse return error.MissingWamrSourceIdentity;
        if (source_val != .object) return error.MissingWamrSourceIdentity;
        const ref_val = source_val.object.get("ref") orelse return error.MissingWamrSourceIdentity;
        if (ref_val != .string or ref_val.string.len == 0) return error.MissingWamrSourceIdentity;
        _ = try requireDigestValue(source_val.object.get("sha"), 40);

        const compiler_val = obj.get("compiler") orelse return error.MissingCompilerIdentity;
        if (compiler_val != .object) return error.MissingCompilerIdentity;
        const module_val = obj.get("module") orelse return error.MissingModuleIdentity;
        if (module_val != .object) return error.MissingModuleIdentity;
        _ = try requireDigestValue(compiler_val.object.get("sha256"), 64);
        _ = try requireDigestValue(module_val.object.get("sha256"), 64);

        if (obj.get("simd_runner")) |simd_val| {
            if (simd_val != .object) return error.InvalidSimdRunnerIdentity;
            _ = try requireDigestValue(simd_val.object.get("sha256"), 64);
        }
        const opt_val = obj.get("optimize") orelse return error.UnsupportedWamrOptimizeMode;
        if (opt_val != .string or (!std.mem.eql(u8, opt_val.string, "ReleaseFast") and !std.mem.eql(u8, opt_val.string, "ReleaseSafe"))) {
            return error.UnsupportedWamrOptimizeMode;
        }
    } else if (std.mem.eql(u8, id_type_val.string, "wasmtime")) {
        const ver_val = obj.get("version") orelse return error.MissingWasmtimeReleaseIdentity;
        const chan_val = obj.get("channel") orelse return error.MissingWasmtimeReleaseIdentity;
        if (ver_val != .string or ver_val.string.len == 0 or chan_val != .string or chan_val.string.len == 0) {
            return error.MissingWasmtimeReleaseIdentity;
        }
    } else {
        return error.UnsupportedEngineIdentityType;
    }
    return obj;
}

pub fn cloneJsonValue(allocator: Allocator, val: Value) !Value {
    const encoded = try stringifyPrettySorted(allocator, val);
    const parsed = try std.json.parseFromSliceLeaky(Value, allocator, encoded, .{});
    return parsed;
}

pub fn jsonValuesEqual(a: Value, b: Value) bool {
    if (@as(std.meta.Tag(Value), a) != @as(std.meta.Tag(Value), b)) {
        const a_num: ?f64 = switch (a) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => null,
        };
        const b_num: ?f64 = switch (b) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => null,
        };
        if (a_num != null and b_num != null) return @abs(a_num.? - b_num.?) <= 1e-12;
        return false;
    }
    return switch (a) {
        .null => true,
        .bool => |v| v == b.bool,
        .integer => |v| v == b.integer,
        .float => |v| @abs(v - b.float) <= 1e-12,
        .number_string => |v| std.mem.eql(u8, v, b.number_string),
        .string => |v| std.mem.eql(u8, v, b.string),
        .array => |arr| blk: {
            if (arr.items.len != b.array.items.len) break :blk false;
            for (arr.items, b.array.items) |ai, bi| {
                if (!jsonValuesEqual(ai, bi)) break :blk false;
            }
            break :blk true;
        },
        .object => |obj| blk: {
            if (obj.count() != b.object.count()) break :blk false;
            var it = obj.iterator();
            while (it.next()) |entry| {
                const other_val = b.object.get(entry.key_ptr.*) orelse break :blk false;
                if (!jsonValuesEqual(entry.value_ptr.*, other_val)) break :blk false;
            }
            break :blk true;
        },
    };
}

pub fn retainWamrArtifactHandoff(
    allocator: Allocator,
    io: std.Io,
    prepared: PreparedEngine,
    artifact_dir_raw: []const u8,
    role: []const u8,
) !Value {
    if (!std.mem.eql(u8, role, "wamr-baseline") and !std.mem.eql(u8, role, "wamr-target")) {
        return error.UnsupportedWamrArtifactRole;
    }
    const identity = try validateEngineIdentity(prepared.identity, role);
    if (!std.mem.eql(u8, identity.get("type").?.string, "wamr")) {
        return error.OnlyWamrArtifactsCanBeRetained;
    }
    if (std.Io.Dir.cwd().statFile(io, artifact_dir_raw, .{})) |_| {
        return error.FileExists;
    } else |_| {}

    try std.Io.Dir.cwd().createDirPath(io, artifact_dir_raw);
    const artifact_dir = try std.Io.Dir.cwd().realPathFileAlloc(io, artifact_dir_raw, allocator);
    errdefer std.Io.Dir.cwd().deleteTree(io, artifact_dir) catch {};

    var retained_val = try cloneJsonValue(allocator, prepared.identity);
    const ArtifactPair = struct { key: []const u8, name: []const u8 };
    var pairs: std.ArrayList(ArtifactPair) = .empty;
    try pairs.appendSlice(allocator, &.{
        .{ .key = "runtime", .name = "wamr" },
        .{ .key = "compiler", .name = "wamrc" },
        .{ .key = "module", .name = "coremark.cwasm" },
    });
    if (identity.contains("simd_runner")) {
        try pairs.append(allocator, .{ .key = "simd_runner", .name = "simd-bench-runner" });
    }

    for (pairs.items) |pair| {
        const entry_obj = identity.get(pair.key).?.object;
        const source_path = entry_obj.get("path").?.string;
        const expected_sha = entry_obj.get("sha256").?.string;
        const destination = try std.fs.path.join(allocator, &.{ artifact_dir, pair.name });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, source_path, allocator, .limited(256 * 1024 * 1024));
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = destination, .data = bytes });
        const digest = try sha256File(allocator, io, destination);
        if (!std.mem.eql(u8, digest, expected_sha)) return error.RetainedHashChanged;
        try retained_val.object.getPtr(pair.key).?.object.put(allocator, "path", .{ .string = destination });
    }

    var manifest_obj: std.json.ObjectMap = .empty;
    try manifest_obj.put(allocator, "schema_version", .{ .integer = handoff_schema_version });
    try manifest_obj.put(allocator, "kind", .{ .string = "coremark-wamr-artifact-handoff" });
    try manifest_obj.put(allocator, "role", .{ .string = try allocator.dupe(u8, role) });
    try manifest_obj.put(allocator, "identity", retained_val);

    const manifest_path = try std.fs.path.join(allocator, &.{ artifact_dir, "manifest.json" });
    const manifest_bytes = try std.fmt.allocPrint(
        allocator,
        "{s}\n",
        .{try stringifyPrettySorted(allocator, .{ .object = manifest_obj })},
    );
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = manifest_path, .data = manifest_bytes });
    return retained_val;
}

pub fn loadWamrArtifactHandoff(
    allocator: Allocator,
    io: std.Io,
    artifact_dir_raw: []const u8,
    expected_identity: Value,
    role: []const u8,
) !std.StringHashMap([]const u8) {
    if (!std.mem.eql(u8, role, "wamr-baseline") and !std.mem.eql(u8, role, "wamr-target")) {
        return error.UnsupportedWamrArtifactRole;
    }
    const artifact_dir = std.Io.Dir.cwd().realPathFileAlloc(io, artifact_dir_raw, allocator) catch {
        return error.CannotReadArtifactHandoff;
    };
    const manifest_path = try std.fs.path.join(allocator, &.{ artifact_dir, "manifest.json" });
    const raw = std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(16 * 1024 * 1024)) catch {
        return error.CannotReadArtifactHandoff;
    };
    const manifest = std.json.parseFromSliceLeaky(Value, allocator, raw, .{}) catch {
        return error.CannotReadArtifactHandoff;
    };
    if (manifest != .object) return error.UnsupportedHandoffManifest;
    const m_obj = manifest.object;
    const schema_val = m_obj.get("schema_version") orelse return error.UnsupportedHandoffManifest;
    if (schema_val != .integer or (schema_val.integer != 1 and schema_val.integer != handoff_schema_version)) {
        return error.UnsupportedHandoffManifest;
    }
    const kind_val = m_obj.get("kind") orelse return error.UnsupportedHandoffManifest;
    if (kind_val != .string or !std.mem.eql(u8, kind_val.string, "coremark-wamr-artifact-handoff")) {
        return error.UnsupportedHandoffManifest;
    }
    if (schema_val.integer == 1) {
        const m_role = if (m_obj.get("role")) |rv| (if (rv == .string) rv.string else "") else role;
        if (!std.mem.eql(u8, role, "wamr-target") or !std.mem.eql(u8, m_role, role)) {
            return error.LegacyHandoffOnlySupportsWamrTarget;
        }
    } else {
        const m_role = m_obj.get("role") orelse return error.HandoffRoleMismatch;
        if (m_role != .string or !std.mem.eql(u8, m_role.string, role)) {
            return error.HandoffRoleMismatch;
        }
    }

    const identity = try validateEngineIdentity(m_obj.get("identity"), role);
    if (!std.mem.eql(u8, identity.get("type").?.string, "wamr")) {
        return error.HandoffMustIdentifyWamr;
    }
    if (!jsonValuesEqual(.{ .object = identity }, expected_identity)) {
        return error.HandoffIdentityMismatch;
    }

    const ArtifactPair = struct { key: []const u8, name: []const u8 };
    var pairs: std.ArrayList(ArtifactPair) = .empty;
    try pairs.appendSlice(allocator, &.{
        .{ .key = "runtime", .name = "wamr" },
        .{ .key = "compiler", .name = "wamrc" },
        .{ .key = "module", .name = "coremark.cwasm" },
    });
    if (identity.contains("simd_runner")) {
        try pairs.append(allocator, .{ .key = "simd_runner", .name = "simd-bench-runner" });
    }

    var paths: std.StringHashMap([]const u8) = .init(allocator);
    for (pairs.items) |pair| {
        const expected_path = try std.fs.path.join(allocator, &.{ artifact_dir, pair.name });
        const item_obj = identity.get(pair.key).?.object;
        const rec_val = item_obj.get("path") orelse return error.HandoffMissingArtifactPath;
        if (rec_val != .string or rec_val.string.len == 0) return error.HandoffMissingArtifactPath;
        const resolved_rec = std.Io.Dir.cwd().realPathFileAlloc(io, rec_val.string, allocator) catch {
            return error.HandoffMissingArtifactFile;
        };
        if (!std.mem.eql(u8, resolved_rec, expected_path)) return error.HandoffWrongArtifactPath;
        const stat = std.Io.Dir.cwd().statFile(io, expected_path, .{}) catch return error.HandoffMissingArtifactFile;
        if (stat.kind != .file) return error.HandoffMissingArtifactFile;
        const actual_sha = try sha256File(allocator, io, expected_path);
        if (!std.mem.eql(u8, actual_sha, item_obj.get("sha256").?.string)) {
            return error.HandoffArtifactHashMismatch;
        }
        try paths.put(pair.key, expected_path);
    }
    return paths;
}

pub const ResolvedFixture = struct {
    path: []const u8,
    sha256: []const u8,
};

pub fn resolveFixture(allocator: Allocator, io: std.Io, repo: []const u8, fixture_arg: []const u8) !ResolvedFixture {
    const joined = if (std.fs.path.isAbsolute(fixture_arg))
        try allocator.dupe(u8, fixture_arg)
    else
        try std.fs.path.join(allocator, &.{ repo, fixture_arg });
    const fixture = std.Io.Dir.cwd().realPathFileAlloc(io, joined, allocator) catch {
        return error.FixtureNotFound;
    };
    const stat = std.Io.Dir.cwd().statFile(io, fixture, .{}) catch return error.FixtureNotFound;
    if (stat.kind != .file) return error.FixtureNotFound;
    const digest = try sha256File(allocator, io, fixture);
    const default_joined = try std.fs.path.join(allocator, &.{ repo, default_fixture_rel });
    if (std.Io.Dir.cwd().realPathFileAlloc(io, default_joined, allocator)) |default_resolved| {
        if (std.mem.eql(u8, fixture, default_resolved) and !std.mem.eql(u8, digest, default_fixture_sha256)) {
            return error.TrackedFixtureChecksumMismatch;
        }
    } else |_| {}
    return .{ .path = fixture, .sha256 = digest };
}

fn countSubstring(haystack: []const u8, needle: []const u8) usize {
    if (needle.len == 0) return 0;
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, pos, needle)) |idx| {
        count += 1;
        pos = idx + needle.len;
    }
    return count;
}

pub fn parseCoremarkOutput(
    output: []const u8,
    engine: []const u8,
    expected_iters: i64,
) !ParsedCoreMark {
    _ = engine;
    if (countSubstring(output, validation_text) != 1 or std.mem.indexOf(u8, output, "ERROR!") != null) {
        return error.MissingCrcValidation;
    }
    if (countSubstring(output, performance_marker) != 1) {
        return error.MissingPerformanceMarker;
    }

    var throughput_lines: usize = 0;
    var throughput_val: f64 = 0.0;
    var iteration_lines: usize = 0;
    var iterations_val: i64 = 0;

    var line_it = std.mem.splitScalar(u8, output, '\n');
    while (line_it.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (std.mem.indexOf(u8, line, "Iterations/Sec") != null) {
            throughput_lines += 1;
            const trimmed = std.mem.trim(u8, line, " \t");
            if (!std.mem.startsWith(u8, trimmed, "Iterations/Sec")) return error.MalformedThroughputOutput;
            const after_key = std.mem.trimStart(u8, trimmed["Iterations/Sec".len..], " \t");
            if (after_key.len == 0 or after_key[0] != ':') return error.MalformedThroughputOutput;
            const num_str = std.mem.trim(u8, after_key[1..], " \t");
            if (num_str.len == 0) return error.MalformedThroughputOutput;
            for (num_str) |ch| {
                if ((ch < '0' or ch > '9') and ch != '.') return error.MalformedThroughputOutput;
            }
            throughput_val = std.fmt.parseFloat(f64, num_str) catch return error.MalformedThroughputOutput;
        }
        const trimmed = std.mem.trim(u8, line, " \t");
        if (std.mem.startsWith(u8, trimmed, "Iterations")) {
            const rest = std.mem.trimStart(u8, trimmed["Iterations".len..], " \t");
            if (rest.len != 0 and rest[0] == ':') {
                iteration_lines += 1;
                const num_str = std.mem.trim(u8, rest[1..], " \t");
                if (num_str.len == 0) return error.MalformedIterationsOutput;
                for (num_str) |ch| {
                    if (ch < '0' or ch > '9') return error.MalformedIterationsOutput;
                }
                iterations_val = std.fmt.parseInt(i64, num_str, 10) catch return error.MalformedIterationsOutput;
            }
        }
    }

    if (throughput_lines != 1) return error.AmbiguousThroughputOutput;
    if (iteration_lines != 1) return error.InvalidIterationsFieldCount;
    if (iterations_val != expected_iters) return error.FixedIterationsMismatch;

    return .{
        .throughput = throughput_val,
        .iterations = iterations_val,
    };
}

pub fn findExecutableInPath(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    name: []const u8,
) !?[]u8 {
    if (std.mem.indexOfScalar(u8, name, '/') != null) {
        if (std.Io.Dir.cwd().statFile(io, name, .{})) |st| {
            if (st.kind == .file) return try std.Io.Dir.cwd().realPathFileAlloc(io, name, allocator);
        } else |_| {}
        return null;
    }
    const path_env = environ_map.get("PATH") orelse "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
    var it = std.mem.splitScalar(u8, path_env, ':');
    while (it.next()) |dir| {
        if (dir.len == 0) continue;
        const candidate = try std.fs.path.join(allocator, &.{ dir, name });
        if (std.Io.Dir.cwd().statFile(io, candidate, .{})) |st| {
            if (st.kind == .file) return candidate;
        } else |_| {}
    }
    return null;
}

pub fn parseCpuList(allocator: Allocator, text: []const u8) ![]usize {
    var cpus: std.ArrayList(usize) = .empty;
    var it = std.mem.splitScalar(u8, std.mem.trim(u8, text, " \t\r\n"), ',');
    while (it.next()) |part_raw| {
        const part = std.mem.trim(u8, part_raw, " \t");
        if (part.len == 0) continue;
        if (std.mem.indexOfScalar(u8, part, '-')) |dash_idx| {
            const start = try std.fmt.parseInt(usize, part[0..dash_idx], 10);
            const end = try std.fmt.parseInt(usize, part[dash_idx + 1 ..], 10);
            if (end < start) return error.InvalidCpuList;
            var cpu = start;
            while (cpu <= end) : (cpu += 1) {
                try cpus.append(allocator, cpu);
            }
        } else {
            try cpus.append(allocator, try std.fmt.parseInt(usize, part, 10));
        }
    }
    std.mem.sort(usize, cpus.items, {}, std.sort.asc(usize));
    return cpus.toOwnedSlice(allocator);
}

pub fn detectAllowedCpus(allocator: Allocator, io: std.Io) ![]usize {
    const status = std.Io.Dir.cwd().readFileAlloc(io, "/proc/self/status", allocator, .limited(64 * 1024)) catch {
        return error.AffinityUnavailable;
    };
    var line_it = std.mem.splitScalar(u8, status, '\n');
    while (line_it.next()) |line| {
        if (std.mem.startsWith(u8, line, "Cpus_allowed_list:")) {
            const list_str = std.mem.trim(u8, line["Cpus_allowed_list:".len..], " \t\r");
            return parseCpuList(allocator, list_str);
        }
    }
    return error.AffinityUnavailable;
}

pub fn selectCpuAffinity(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
) !AffinityInfo {
    const allowed = try detectAllowedCpus(allocator, io);
    const taskset = (try findExecutableInPath(allocator, io, environ_map, "taskset")) orelse return error.TasksetNotFound;
    return selectCpuAffinityWith(allocator, io, environ_map, allowed, taskset, null, defaultRunCommand);
}

pub fn selectCpuAffinityWith(
    allocator: Allocator,
    io: std.Io,
    environ_map: ?*const std.process.Environ.Map,
    allowed_cpus: []const usize,
    taskset_opt: ?[]const u8,
    runner_ctx: ?*anyopaque,
    runner: CommandRunner,
) !AffinityInfo {
    if (allowed_cpus.len == 0) return error.EmptyCpuAffinitySet;
    const taskset = taskset_opt orelse return error.TasksetNotFound;
    const sorted = try allocator.dupe(usize, allowed_cpus);
    std.mem.sort(usize, sorted, {}, std.sort.asc(usize));
    const selected = sorted[0];
    const selected_str = try std.fmt.allocPrint(allocator, "{d}", .{selected});
    const output = try runner(
        runner_ctx,
        allocator,
        io,
        &.{
            taskset,
            "--cpu-list",
            selected_str,
            "sh",
            "-c",
            "sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status",
        },
        null,
        environ_map,
    );
    const trimmed = std.mem.trim(u8, output, " \t\r\n");
    if (!std.mem.eql(u8, trimmed, selected_str)) {
        return error.TasksetVerificationFailed;
    }
    return .{
        .allowed_cpus = sorted,
        .selected_cpu = selected,
        .taskset = try allocator.dupe(u8, taskset),
    };
}

pub fn applyAffinity(
    allocator: Allocator,
    cmd: []const []const u8,
    affinity_opt: ?AffinityInfo,
) ![]const []const u8 {
    const affinity = affinity_opt orelse return allocator.dupe([]const u8, cmd);
    var out: std.ArrayList([]const u8) = .empty;
    try out.append(allocator, affinity.taskset);
    try out.append(allocator, "--cpu-list");
    try out.append(allocator, try std.fmt.allocPrint(allocator, "{d}", .{affinity.selected_cpu}));
    try out.appendSlice(allocator, cmd);
    return out.toOwnedSlice(allocator);
}

pub fn counterbalancedOrder(
    allocator: Allocator,
    keys: []const []const u8,
    count_per_engine: isize,
) ![]const []const u8 {
    if (count_per_engine < 0) return error.NegativeCountPerEngine;
    if (keys.len == 0 and count_per_engine > 0) return error.EmptyEnginesSchedule;
    const count: usize = @intCast(count_per_engine);
    var order: std.ArrayList([]const u8) = .empty;
    for (0..count) |rep| {
        if (rep % 2 == 0) {
            for (keys) |k| try order.append(allocator, k);
        } else {
            var idx = keys.len;
            while (idx > 0) {
                idx -= 1;
                try order.append(allocator, keys[idx]);
            }
        }
    }
    return order.toOwnedSlice(allocator);
}

pub fn coremarkGuestArgs(allocator: Allocator, iterations: i64) ![]const []const u8 {
    if (iterations <= 0) return error.NonPositiveIterations;
    const out = try allocator.alloc([]const u8, 5);
    out[0] = "0";
    out[1] = "0";
    out[2] = "0";
    out[3] = try std.fmt.allocPrint(allocator, "{d}", .{iterations});
    out[4] = "0";
    return out;
}

pub fn resolveRefSha(
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    ref: []const u8,
) ![]u8 {
    const out = try run(allocator, io, &.{ "git", "rev-parse", ref }, repo, null);
    return allocator.dupe(u8, std.mem.trim(u8, out, " \t\r\n"));
}

pub fn captureExecutionIdentity(
    allocator: Allocator,
    environ_map: *const std.process.Environ.Map,
    local_run_id: ?[]const u8,
) !Value {
    if (environ_map.get("GITHUB_RUN_ID")) |gh_run_id| {
        if (gh_run_id.len != 0) {
            var obj: std.json.ObjectMap = .empty;
            try obj.put(allocator, "provider", .{ .string = "github-actions" });
            try obj.put(allocator, "repository", .{ .string = environ_map.get("GITHUB_REPOSITORY") orelse "" });
            try obj.put(allocator, "run_id", .{ .string = gh_run_id });
            try obj.put(allocator, "run_attempt", .{ .string = environ_map.get("GITHUB_RUN_ATTEMPT") orelse "" });
            try obj.put(allocator, "workflow", .{ .string = environ_map.get("GITHUB_WORKFLOW") orelse "" });
            try obj.put(allocator, "workflow_ref", .{ .string = environ_map.get("GITHUB_WORKFLOW_REF") orelse "" });
            try obj.put(allocator, "workflow_sha", .{ .string = environ_map.get("GITHUB_SHA") orelse "" });
            try obj.put(allocator, "job", .{ .string = environ_map.get("GITHUB_JOB") orelse "" });
            return .{ .object = obj };
        }
    }
    const raw_id = local_run_id orelse (environ_map.get("COREMARK_RUN_ID") orelse "");
    const run_id = std.mem.trim(u8, raw_id, " \t\r\n");
    if (run_id.len == 0) return error.MissingLocalExecutionIdentity;
    var obj: std.json.ObjectMap = .empty;
    try obj.put(allocator, "provider", .{ .string = "local" });
    try obj.put(allocator, "run_id", .{ .string = try allocator.dupe(u8, run_id) });
    return .{ .object = obj };
}

pub fn generateUuidV4(io: std.Io, allocator: Allocator) ![]u8 {
    var rand_bytes: [16]u8 = undefined;
    io.random(&rand_bytes);
    rand_bytes[6] = (rand_bytes[6] & 0x0f) | 0x40;
    rand_bytes[8] = (rand_bytes[8] & 0x3f) | 0x80;
    const h = std.fmt.bytesToHex(rand_bytes, .lower);
    return std.fmt.allocPrint(
        allocator,
        "{s}-{s}-{s}-{s}-{s}",
        .{ h[0..8], h[8..12], h[12..16], h[16..20], h[20..32] },
    );
}

pub fn formatUtcIsoTimestamp(allocator: Allocator, io: std.Io) ![]u8 {
    const now_ns: i128 = @intCast(std.Io.Clock.real.now(io).nanoseconds);
    return native_benchmark.formatUtcTimestamp(allocator, now_ns);
}

pub fn isValidUuid(text: []const u8) bool {
    if (text.len != 36) return false;
    for (text, 0..) |ch, idx| {
        if (idx == 8 or idx == 13 or idx == 18 or idx == 23) {
            if (ch != '-') return false;
        } else {
            const hex = (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f') or (ch >= 'A' and ch <= 'F');
            if (!hex) return false;
        }
    }
    return true;
}

pub fn captureReportProvenance(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    repo: []const u8,
    local_run_id_opt: ?[]const u8,
    source_sha_override: ?[]const u8,
) !Value {
    const raw_sha = if (source_sha_override) |s|
        try allocator.dupe(u8, s)
    else
        try resolveRefSha(allocator, io, repo, "HEAD");
    const source_sha = try requireDigest(raw_sha, 40);

    const script_rel = "scripts/bench_coremark.zig";
    const script_full = try std.fs.path.join(allocator, &.{ repo, script_rel });
    const script_sha = try sha256File(allocator, io, script_full);

    var effective_local_id = local_run_id_opt;
    const has_gh = if (environ_map.get("GITHUB_RUN_ID")) |v| v.len != 0 else false;
    if (effective_local_id == null and environ_map.get("COREMARK_RUN_ID") == null and !has_gh) {
        effective_local_id = try generateUuidV4(io, allocator);
    }
    const execution = try captureExecutionIdentity(allocator, environ_map, effective_local_id);

    var script_obj: std.json.ObjectMap = .empty;
    try script_obj.put(allocator, "path", .{ .string = script_rel });
    try script_obj.put(allocator, "sha256", .{ .string = script_sha });

    var producer_obj: std.json.ObjectMap = .empty;
    try producer_obj.put(allocator, "source_sha", .{ .string = source_sha });
    try producer_obj.put(allocator, "script", .{ .object = script_obj });

    var prov_obj: std.json.ObjectMap = .empty;
    try prov_obj.put(allocator, "report_id", .{ .string = try generateUuidV4(io, allocator) });
    try prov_obj.put(allocator, "generated_at", .{ .string = try formatUtcIsoTimestamp(allocator, io) });
    try prov_obj.put(allocator, "producer", .{ .object = producer_obj });
    try prov_obj.put(allocator, "execution", execution);
    return .{ .object = prov_obj };
}

pub const WorktreeInfo = struct {
    path: []const u8,
    sha: []const u8,
};

pub fn makeWorktree(
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    ref: []const u8,
    root: []const u8,
    label: []const u8,
) !WorktreeInfo {
    const sha = try resolveRefSha(allocator, io, repo, ref);
    const short_sha = sha[0..@min(sha.len, 12)];
    const wt = try std.fs.path.join(allocator, &.{ root, try std.fmt.allocPrint(allocator, "{s}-{s}", .{ label, short_sha }) });
    _ = try run(allocator, io, &.{ "git", "worktree", "add", "--detach", wt, sha }, repo, null);
    return .{ .path = wt, .sha = sha };
}

pub fn worktreeEnv(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    wt: []const u8,
) !std.process.Environ.Map {
    var env = try base_env.clone(allocator);
    _ = env.swapRemove("ZIG_LOCAL_CACHE_DIR");
    const global_cache = try std.fs.path.join(allocator, &.{ wt, ".zig-global-cache" });
    try std.Io.Dir.cwd().createDirPath(io, global_cache);
    try env.put("ZIG_GLOBAL_CACHE_DIR", global_cache);
    return env;
}

pub fn isTrapFlake(output: []const u8) bool {
    var line_it = std.mem.splitScalar(u8, output, '\n');
    while (line_it.next()) |line| {
        const lower = blk: {
            var buf: [512]u8 = undefined;
            const len = @min(line.len, buf.len);
            for (line[0..len], 0..) |ch, i| buf[i] = std.ascii.toLower(ch);
            break :blk buf[0..len];
        };
        const prefix_pos = std.mem.indexOf(u8, lower, "wasm trap: out of bounds memory access") orelse continue;
        const after_prefix = lower[prefix_pos..];
        const lf_pos = std.mem.indexOf(u8, after_prefix, "local_func[") orelse continue;
        var rest = after_prefix[lf_pos + "local_func[".len ..];
        if (rest.len != 0 and rest[0] == '-') rest = rest[1..];
        var digits: usize = 0;
        while (digits < rest.len and std.ascii.isDigit(rest[digits])) : (digits += 1) {}
        if (digits == 0 or digits >= rest.len or rest[digits] != ']') continue;
        rest = rest[digits + 1 ..];
        if (std.mem.startsWith(u8, rest, "+0x0")) return true;
        const trimmed = std.mem.trimStart(u8, rest, " \t");
        if (trimmed.len > 0 and trimmed[0] == '"') {
            if (std.mem.indexOfScalarPos(u8, trimmed, 1, '"')) |close_q| {
                if (std.mem.startsWith(u8, trimmed[close_q + 1 ..], "+0x0")) return true;
            }
        }
    }
    return false;
}

pub fn runWamrWithRetry(
    allocator: Allocator,
    io: std.Io,
    cmd: []const []const u8,
    cwd: []const u8,
    env: ?*const std.process.Environ.Map,
    sample_idx: usize,
    samples_total: usize,
    runner_ctx: ?*anyopaque,
    runner: CommandRunner,
) ![]u8 {
    for (0..(1 + trap_retry_max)) |attempt| {
        if (runner(runner_ctx, allocator, io, cmd, cwd, env)) |out| {
            return out;
        } else |err| {
            if (attempt == trap_retry_max) return err;
            if (!builtin.is_test) {
                std.debug.print(
                    "[harness]   sample {d}/{d}: command error ({s}) — retry {d}/{d}\n",
                    .{ sample_idx + 1, samples_total, @errorName(err), attempt + 1, trap_retry_max },
                );
            }
        }
    }
    return error.CommandFailed;
}

pub fn measureCommand(
    allocator: Allocator,
    io: std.Io,
    opts: struct {
        engine: []const u8,
        cmd: []const []const u8,
        cwd: []const u8,
        env: ?*const std.process.Environ.Map = null,
        warmups: usize,
        runs: usize,
        retry_wamr_flake: bool = false,
        affinity: ?AffinityInfo = null,
        expected_iterations: i64 = expected_iterations,
        runner_ctx: ?*anyopaque = null,
        runner: CommandRunner = defaultRunCommand,
    },
) ![]f64 {
    var values: std.ArrayList(f64) = .empty;
    const total = opts.warmups + opts.runs;
    const effective_cmd = try applyAffinity(allocator, opts.cmd, opts.affinity);
    for (0..total) |i| {
        const output = if (opts.retry_wamr_flake)
            try runWamrWithRetry(allocator, io, effective_cmd, opts.cwd, opts.env, i, total, opts.runner_ctx, opts.runner)
        else
            try opts.runner(opts.runner_ctx, allocator, io, effective_cmd, opts.cwd, opts.env);
        const parsed = try parseCoremarkOutput(output, opts.engine, opts.expected_iterations);
        if (i >= opts.warmups) {
            try values.append(allocator, parsed.throughput);
        }
    }
    return values.toOwnedSlice(allocator);
}

pub const MeasuredPreparedResult = struct {
    results: std.StringHashMap(EngineResult),
    records: []SampleRecord,
};

pub fn measurePreparedEngines(
    allocator: Allocator,
    io: std.Io,
    engines: []const PreparedEngine,
    opts: struct {
        warmups: usize,
        runs: usize,
        affinity: ?AffinityInfo = null,
        runner_ctx: ?*anyopaque = null,
        runner: CommandRunner = defaultRunCommand,
    },
) !MeasuredPreparedResult {
    var by_key: std.StringHashMap(PreparedEngine) = .init(allocator);
    var engine_records: std.StringHashMap(std.ArrayList(SampleRecord)) = .init(allocator);
    var warmup_ordinals: std.StringHashMap(usize) = .init(allocator);
    var measured_ordinals: std.StringHashMap(usize) = .init(allocator);
    var keys: std.ArrayList([]const u8) = .empty;

    for (engines) |engine| {
        if (by_key.contains(engine.key)) return error.DuplicatePreparedEngineKey;
        try by_key.put(engine.key, engine);
        try engine_records.put(engine.key, .empty);
        try warmup_ordinals.put(engine.key, 0);
        try measured_ordinals.put(engine.key, 0);
        try keys.append(allocator, engine.key);
    }

    var records: std.ArrayList(SampleRecord) = .empty;
    var schedule_position: usize = 0;

    const phases = [_]struct { name: []const u8, count: usize }{
        .{ .name = "warmup", .count = opts.warmups },
        .{ .name = "measured", .count = opts.runs },
    };
    for (phases) |phase_info| {
        const order = try counterbalancedOrder(allocator, keys.items, @intCast(phase_info.count));
        for (order) |key| {
            const engine = by_key.get(key).?;
            schedule_position += 1;
            const ord_map = if (std.mem.eql(u8, phase_info.name, "warmup")) &warmup_ordinals else &measured_ordinals;
            const ordinal = (ord_map.get(key) orelse 0) + 1;
            try ord_map.put(key, ordinal);

            const cmd = try applyAffinity(allocator, engine.cmd, opts.affinity);
            const started_at = try formatUtcIsoTimestamp(allocator, io);
            const start_ns = std.Io.Clock.awake.now(io).nanoseconds;
            const output = if (engine.retry_wamr_flake)
                try runWamrWithRetry(allocator, io, cmd, engine.cwd, engine.env, ordinal - 1, phase_info.count, opts.runner_ctx, opts.runner)
            else
                try opts.runner(opts.runner_ctx, allocator, io, cmd, engine.cwd, engine.env);
            const end_ns = std.Io.Clock.awake.now(io).nanoseconds;
            const raw_elapsed = @as(f64, @floatFromInt(@max(1, end_ns - start_ns))) / @as(f64, @floatFromInt(std.time.ns_per_s));
            const completed_at = try formatUtcIsoTimestamp(allocator, io);
            const parsed = try parseCoremarkOutput(output, engine.engine, engine.expected_iterations);

            const record: SampleRecord = .{
                .engine_key = engine.key,
                .engine = engine.engine,
                .phase = phase_info.name,
                .engine_ordinal = ordinal,
                .schedule_position = schedule_position,
                .started_at = started_at,
                .completed_at = completed_at,
                .elapsed_seconds = raw_elapsed,
                .iterations_per_second = parsed.throughput,
                .iterations = parsed.iterations,
            };
            try records.append(allocator, record);
            try engine_records.getPtr(key).?.append(allocator, record);
        }
    }

    var results: std.StringHashMap(EngineResult) = .init(allocator);
    for (engines) |engine| {
        const eng_recs = engine_records.get(engine.key).?.items;
        var measured_vals: std.ArrayList(f64) = .empty;
        for (eng_recs) |rec| {
            if (std.mem.eql(u8, rec.phase, "measured")) {
                try measured_vals.append(allocator, rec.iterations_per_second);
            }
        }
        if (measured_vals.items.len != opts.runs) return error.UnexpectedMeasuredSampleCount;
        try results.put(engine.key, .{
            .engine = engine.engine,
            .version = engine.version,
            .optimize = engine.optimize,
            .values = try measured_vals.toOwnedSlice(allocator),
            .samples = try allocator.dupe(SampleRecord, eng_recs),
            .identity = engine.identity,
        });
    }
    return .{
        .results = results,
        .records = try records.toOwnedSlice(allocator),
    };
}

pub fn zigBuildCommand(allocator: Allocator, io: std.Io, wt: []const u8, optimize: []const u8) ![]const []const u8 {
    if (benchmark_optimize.buildCommand(allocator, io, wt, optimize)) |arr| {
        return allocator.dupe([]const u8, &arr);
    } else |_| {
        const mode = if (std.mem.eql(u8, optimize, "Debug"))
            "debug"
        else if (std.mem.eql(u8, optimize, "ReleaseFast"))
            "fast"
        else if (std.mem.eql(u8, optimize, "ReleaseSafe"))
            "safe"
        else if (std.mem.eql(u8, optimize, "ReleaseSmall"))
            "small"
        else
            return error.InvalidOptimizeMode;
        return allocator.dupe([]const u8, &.{
            "zig",
            "build",
            try std.fmt.allocPrint(allocator, "-Doptimize={s}", .{mode}),
        });
    }
}

pub fn prepareWamr(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    wt: []const u8,
    ref: []const u8,
    sha: []const u8,
    fixture: []const u8,
    optimize: []const u8,
    guest_args: []const []const u8,
    expected_iters: i64,
    runner_ctx: ?*anyopaque,
    runner: CommandRunner,
) !PreparedEngine {
    const env_ptr = try allocator.create(std.process.Environ.Map);
    env_ptr.* = try worktreeEnv(allocator, io, base_env, wt);

    const build_cmd = try zigBuildCommand(allocator, io, wt, optimize);
    _ = try runner(runner_ctx, allocator, io, build_cmd, wt, env_ptr);

    const wamrc = try std.fs.path.join(allocator, &.{ wt, "zig-out/bin/wamrc" });
    const wamr = try std.fs.path.join(allocator, &.{ wt, "zig-out/bin/wamr" });
    const simd_runner = try std.fs.path.join(allocator, &.{ wt, "zig-out/bin/simd-bench-runner" });
    const cwasm = try std.fs.path.join(allocator, &.{ wt, ".bench-coremark.cwasm" });
    _ = try runner(runner_ctx, allocator, io, &.{ wamrc, "compile", fixture, "-o", cwasm }, wt, env_ptr);

    var cmd_list: std.ArrayList([]const u8) = .empty;
    try cmd_list.appendSlice(allocator, &.{ wamr, "run", cwasm });
    try cmd_list.appendSlice(allocator, guest_args);

    const has_simd = if (std.Io.Dir.cwd().statFile(io, simd_runner, .{})) |st| st.kind == .file else |_| false;
    const identity: Value = if (runner == defaultRunCommand)
        try makeWamrIdentity(allocator, io, .{
            .ref = ref,
            .source_sha = sha,
            .optimize = optimize,
            .runtime_path = wamr,
            .compiler_path = wamrc,
            .module_path = cwasm,
            .simd_runner_path = if (has_simd) simd_runner else null,
        })
    else
        .{ .object = .empty };

    return .{
        .key = try std.fmt.allocPrint(allocator, "wamr:{s}:{s}", .{ sha, optimize }),
        .engine = try std.fmt.allocPrint(allocator, "WAMR {s}", .{ref}),
        .version = try std.fmt.allocPrint(allocator, "{s} ({s})", .{ ref, sha }),
        .optimize = try allocator.dupe(u8, optimize),
        .cmd = try cmd_list.toOwnedSlice(allocator),
        .cwd = try allocator.dupe(u8, wt),
        .env = env_ptr,
        .expected_iterations = expected_iters,
        .retry_wamr_flake = true,
        .identity = identity,
    };
}

pub fn prepareWasmtime(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    path: []const u8,
    label: []const u8,
    version: []const u8,
    fixture: []const u8,
    guest_args: []const []const u8,
    expected_iters: i64,
) !PreparedEngine {
    const resolved = std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator) catch try allocator.dupe(u8, path);
    const bin_sha = sha256File(allocator, io, path) catch "0000000000000000000000000000000000000000000000000000000000000000";
    var cmd_list: std.ArrayList([]const u8) = .empty;
    try cmd_list.appendSlice(allocator, &.{ path, "run", fixture });
    try cmd_list.appendSlice(allocator, guest_args);

    const channel = if (std.mem.eql(u8, label, "Wasmtime historical pin")) "historical-pin" else "caller-selected";
    const identity: Value = makeWasmtimeIdentity(allocator, io, .{
        .version = version,
        .runtime_path = path,
        .channel = channel,
    }) catch .{ .object = .empty };

    return .{
        .key = try std.fmt.allocPrint(allocator, "wasmtime:{s}", .{resolved}),
        .engine = try allocator.dupe(u8, label),
        .version = try std.fmt.allocPrint(allocator, "{s} (sha256:{s}; {s})", .{ version, bin_sha, path }),
        .optimize = "default JIT",
        .cmd = try cmd_list.toOwnedSlice(allocator),
        .cwd = std.fs.path.dirname(fixture) orelse ".",
        .env = base_env,
        .expected_iterations = expected_iters,
        .identity = identity,
    };
}

pub fn normalizeArchName(arch: []const u8) []const u8 {
    if (std.ascii.eqlIgnoreCase(arch, "amd64") or std.ascii.eqlIgnoreCase(arch, "x64") or std.ascii.eqlIgnoreCase(arch, "x86_64")) {
        return "x86_64";
    }
    if (std.ascii.eqlIgnoreCase(arch, "arm64") or std.ascii.eqlIgnoreCase(arch, "aarch64")) {
        return "aarch64";
    }
    return arch;
}

pub fn normalizeArch() []const u8 {
    return normalizeArchName(@tagName(builtin.cpu.arch));
}

pub fn parseWasmtimeVersionOutput(allocator: Allocator, output: []const u8) ![]u8 {
    var tok = std.mem.tokenizeAny(u8, output, " \t\r\n");
    while (tok.next()) |word| {
        if (std.ascii.eqlIgnoreCase(word, "wasmtime")) {
            const next_tok = tok.next() orelse break;
            const ver_cand = if (next_tok.len > 0 and (next_tok[0] == 'v' or next_tok[0] == 'V'))
                next_tok[1..]
            else
                next_tok;
            var dot_count: usize = 0;
            var valid = ver_cand.len > 0;
            for (ver_cand) |ch| {
                if (ch == '.') {
                    dot_count += 1;
                } else if (!std.ascii.isDigit(ch)) {
                    valid = false;
                    break;
                }
            }
            if (valid and dot_count == 2) return allocator.dupe(u8, ver_cand);
        }
    }
    return error.CouldNotParseWasmtimeVersion;
}

pub fn wasmtimeVersionWith(
    allocator: Allocator,
    io: std.Io,
    path: []const u8,
    runner_ctx: ?*anyopaque,
    runner: CommandRunner,
) ![]u8 {
    const out = try runner(runner_ctx, allocator, io, &.{ path, "--version" }, null, null);
    return parseWasmtimeVersionOutput(allocator, std.mem.trim(u8, out, " \t\r\n"));
}

pub fn validatePinnedWasmtimeWith(
    allocator: Allocator,
    io: std.Io,
    path: []const u8,
    runner_ctx: ?*anyopaque,
    runner: CommandRunner,
) ![]u8 {
    const ver = try wasmtimeVersionWith(allocator, io, path, runner_ctx, runner);
    if (!std.mem.eql(u8, ver, pinned_wasmtime_version)) {
        return error.PinnedWasmtimeVersionMismatch;
    }
    return ver;
}

pub fn validatePinnedWasmtime(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    return validatePinnedWasmtimeWith(allocator, io, path, null, defaultRunCommand);
}

pub fn installPinnedWasmtime(
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    cache_arg: ?[]const u8,
) ![]u8 {
    const os_name = @tagName(builtin.os.tag);
    const arch = normalizeArch();
    var matched_asset: ?PinnedWasmtimeAsset = null;
    for (pinned_wasmtime_assets) |asset| {
        if (std.mem.eql(u8, asset.os_name, os_name) and std.mem.eql(u8, asset.arch, arch)) {
            matched_asset = asset;
            break;
        }
    }
    const asset = matched_asset orelse return error.UnsupportedPinnedWasmtimePlatform;
    const cache = if (cache_arg) |c|
        try allocator.dupe(u8, c)
    else
        try std.fs.path.join(allocator, &.{ repo, ".bench-coremark/tools" });
    const install_dir = try std.fs.path.join(
        allocator,
        &.{ cache, try std.fmt.allocPrint(allocator, "wasmtime-v{s}-{s}-{s}", .{ pinned_wasmtime_version, arch, os_name }) },
    );
    const binary = try std.fs.path.join(allocator, &.{ install_dir, "wasmtime" });
    if (std.Io.Dir.cwd().statFile(io, binary, .{})) |st| {
        if (st.kind == .file) {
            const actual_sha = try sha256File(allocator, io, binary);
            if (!std.mem.eql(u8, actual_sha, asset.binary_sha256)) {
                return error.CachedWasmtimeBinaryChecksumMismatch;
            }
            _ = try validatePinnedWasmtime(allocator, io, binary);
            return binary;
        }
    } else |_| {}

    try std.Io.Dir.cwd().createDirPath(io, cache);
    const archive = try std.fs.path.join(allocator, &.{ cache, asset.asset_name });
    const url = try std.fmt.allocPrint(
        allocator,
        "https://github.com/bytecodealliance/wasmtime/releases/download/v{s}/{s}",
        .{ pinned_wasmtime_version, asset.asset_name },
    );
    _ = try run(allocator, io, &.{ "curl", "-fsSL", "-o", archive, url }, null, null);
    const archive_sha = try sha256File(allocator, io, archive);
    if (!std.mem.eql(u8, archive_sha, asset.archive_sha256)) {
        std.Io.Dir.cwd().deleteFile(io, archive) catch {};
        return error.WasmtimeArchiveChecksumMismatch;
    }

    try std.Io.Dir.cwd().createDirPath(io, install_dir);
    _ = try run(
        allocator,
        io,
        &.{ "tar", "-xJf", archive, "--strip-components=1", "-C", install_dir },
        null,
        null,
    );
    std.Io.Dir.cwd().deleteFile(io, archive) catch {};
    const actual_bin_sha = try sha256File(allocator, io, binary);
    if (!std.mem.eql(u8, actual_bin_sha, asset.binary_sha256)) {
        std.Io.Dir.cwd().deleteFile(io, binary) catch {};
        return error.WasmtimeBinaryChecksumMismatch;
    }
    _ = try validatePinnedWasmtime(allocator, io, binary);
    return binary;
}

pub const StatsTuple = struct {
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
};

pub fn fmtStats(allocator: Allocator, values: []const f64) !StatsTuple {
    if (values.len == 0) return error.EmptySampleValues;
    var sum: f64 = 0.0;
    var min_v = values[0];
    var max_v = values[0];
    for (values) |v| {
        sum += v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    const mean = sum / @as(f64, @floatFromInt(values.len));
    const sorted = try allocator.dupe(f64, values);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const mid = sorted.len / 2;
    const median = if (sorted.len % 2 == 1)
        sorted[mid]
    else
        (sorted[mid - 1] + sorted[mid]) / 2.0;
    return .{ .mean = mean, .median = median, .min = min_v, .max = max_v };
}

pub fn computeDeltaPct(allocator: Allocator, baseline_vals: []const f64, target_vals: []const f64) !f64 {
    const b_stats = try fmtStats(allocator, baseline_vals);
    const t_stats = try fmtStats(allocator, target_vals);
    return (t_stats.mean / b_stats.mean - 1.0) * 100.0;
}

pub fn computeMedianDeltaPct(allocator: Allocator, baseline_vals: []const f64, target_vals: []const f64) !f64 {
    const b_stats = try fmtStats(allocator, baseline_vals);
    const t_stats = try fmtStats(allocator, target_vals);
    return (t_stats.median / b_stats.median - 1.0) * 100.0;
}

pub const RatioStats = struct {
    median_ratio: f64,
    mean_ratio: f64,
};

pub fn computeRatioStats(allocator: Allocator, numerator_vals: []const f64, denominator_vals: []const f64) !RatioStats {
    const n_stats = try fmtStats(allocator, numerator_vals);
    const d_stats = try fmtStats(allocator, denominator_vals);
    return .{
        .median_ratio = n_stats.median / d_stats.median,
        .mean_ratio = n_stats.mean / d_stats.mean,
    };
}

pub fn readOptionalText(allocator: Allocator, io: std.Io, path: []const u8) []const u8 {
    const raw = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024)) catch return "";
    return std.mem.trim(u8, raw, " \t\r\n");
}

pub fn hostCpuModel(allocator: Allocator, io: std.Io) []const u8 {
    if (run(allocator, io, &.{"lscpu"}, null, null)) |lscpu_out| {
        var it = std.mem.splitScalar(u8, lscpu_out, '\n');
        while (it.next()) |line| {
            if (std.ascii.startsWithIgnoreCase(line, "model name:")) {
                if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                    const val = std.mem.trim(u8, line[colon + 1 ..], " \t\r");
                    if (val.len != 0) return allocator.dupe(u8, val) catch "unknown CPU";
                }
            }
        }
    } else |_| {}

    const cpuinfo = readOptionalText(allocator, io, "/proc/cpuinfo");
    if (cpuinfo.len != 0) {
        var it = std.mem.splitScalar(u8, cpuinfo, '\n');
        while (it.next()) |line| {
            if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                const key = std.mem.trim(u8, line[0..colon], " \t\r");
                if (std.ascii.eqlIgnoreCase(key, "model name") or std.ascii.eqlIgnoreCase(key, "model")) {
                    const val = std.mem.trim(u8, line[colon + 1 ..], " \t\r");
                    if (val.len != 0) return allocator.dupe(u8, val) catch "unknown CPU";
                }
            }
        }
    }
    return @tagName(builtin.cpu.arch);
}

pub fn captureHostIdentity(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    require_boot_id: bool,
) !HostIdentity {
    const arch = normalizeArch();
    const cpu_count = std.Thread.getCpuCount() catch 0;
    const cpu_model = hostCpuModel(allocator, io);
    const runner_name = environ_map.get("RUNNER_NAME") orelse "";
    const boot_id = readOptionalText(allocator, io, "/proc/sys/kernel/random/boot_id");
    if (require_boot_id and boot_id.len == 0) {
        return error.MissingLinuxBootId;
    }
    return .{
        .arch = arch,
        .cpu_count = cpu_count,
        .cpu_model = cpu_model,
        .runner_name = runner_name,
        .boot_id = boot_id,
    };
}

pub fn hostEmulationEvidence(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
) ![]u8 {
    const lscpu_out = run(allocator, io, &.{"lscpu"}, null, null) catch "";
    return std.fmt.allocPrint(
        allocator,
        "{s}\n{s}\n{s}\n{s}\n{s}\n{s}",
        .{
            environ_map.get("QEMU_CPU") orelse "",
            environ_map.get("QEMU_LD_PREFIX") orelse "",
            readOptionalText(allocator, io, "/proc/cpuinfo"),
            readOptionalText(allocator, io, "/sys/class/dmi/id/product_name"),
            readOptionalText(allocator, io, "/sys/class/dmi/id/sys_vendor"),
            lscpu_out,
        },
    );
}

fn containsWordCaseInsensitive(haystack: []const u8, words: []const []const u8) bool {
    var i: usize = 0;
    while (i < haystack.len) {
        if (!std.ascii.isAlphanumeric(haystack[i]) and haystack[i] != '_') {
            i += 1;
            continue;
        }
        const start = i;
        while (i < haystack.len and (std.ascii.isAlphanumeric(haystack[i]) or haystack[i] == '_')) : (i += 1) {}
        const token = haystack[start..i];
        for (words) |w| {
            if (std.ascii.eqlIgnoreCase(token, w)) return true;
        }
    }
    return false;
}

pub fn validateNativeHostWith(
    expected_arch_raw: []const u8,
    identity: HostIdentity,
    runner_arch_opt: ?[]const u8,
    evidence: []const u8,
) !HostIdentity {
    const expected = normalizeArchName(expected_arch_raw);
    if (identity.boot_id.len == 0) return error.MissingLinuxBootId;
    if (!std.mem.eql(u8, identity.arch, expected)) return error.HostArchMismatch;
    if (runner_arch_opt) |ra| {
        if (ra.len != 0 and !std.mem.eql(u8, normalizeArchName(ra), expected)) {
            return error.RunnerArchMismatch;
        }
    }
    if (identity.cpu_count == 0) return error.UnavailableCpuCount;
    if (identity.cpu_model.len == 0 or std.mem.eql(u8, identity.cpu_model, "unknown CPU")) {
        return error.UnavailableCpuModel;
    }
    if (containsWordCaseInsensitive(evidence, &.{ "qemu", "tcg", "emulate", "emulated", "emulation", "emulator" })) {
        return error.EmulationDetected;
    }
    if (std.mem.eql(u8, expected, "aarch64") and
        containsWordCaseInsensitive(identity.cpu_model, &.{ "Intel", "AMD", "Xeon", "EPYC" }))
    {
        return error.EmulatedX86CpuModelOnAarch64;
    }
    return identity;
}

pub fn validateNativeHost(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    expected_arch: []const u8,
) !HostIdentity {
    const identity = try captureHostIdentity(allocator, io, environ_map, true);
    const evidence = try hostEmulationEvidence(allocator, io, environ_map);
    return validateNativeHostWith(expected_arch, identity, environ_map.get("RUNNER_ARCH"), evidence);
}

pub fn validateSameHostWith(expected: HostIdentity, actual: HostIdentity) !void {
    if (!expected.eql(actual)) return error.HostIdentityChanged;
}

pub fn validateSameHost(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    expected: HostIdentity,
) !void {
    const actual = try captureHostIdentity(allocator, io, environ_map, expected.boot_id.len != 0);
    return validateSameHostWith(expected, actual);
}

pub fn hostInfo(allocator: Allocator, identity: HostIdentity) ![]u8 {
    const cpu_str = if (identity.cpu_count != 0)
        try std.fmt.allocPrint(allocator, "{d}", .{identity.cpu_count})
    else
        "?";
    const fp = try identity.fingerprint(allocator);
    if (identity.runner_name.len != 0) {
        return std.fmt.allocPrint(
            allocator,
            "_Host: arch `{s}` · {s} vCPU · `{s}` · runner `{s}` · host fingerprint `{s}`_",
            .{ identity.arch, cpu_str, identity.cpu_model, identity.runner_name, fp },
        );
    }
    return std.fmt.allocPrint(
        allocator,
        "_Host: arch `{s}` · {s} vCPU · `{s}` · host fingerprint `{s}`_",
        .{ identity.arch, cpu_str, identity.cpu_model, fp },
    );
}

pub fn formatSamples(allocator: Allocator, values: []const f64) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    for (values, 0..) |val, idx| {
        if (idx != 0) try out.writer.writeAll(", ");
        const s = try decimal_format.fixed(allocator, val, 1);
        try out.writer.writeAll(s);
    }
    return out.toOwnedSlice();
}

pub fn scheduleLabel(allocator: Allocator, records: []const SampleRecord) ![]u8 {
    var names: std.StringHashMap(u8) = .init(allocator);
    var ordered_keys: std.ArrayList([]const u8) = .empty;
    var labels: std.ArrayList(u8) = .empty;
    for (records) |rec| {
        const label = if (names.get(rec.engine_key)) |existing|
            existing
        else blk: {
            const ch: u8 = @intCast('A' + ordered_keys.items.len);
            try names.put(rec.engine_key, ch);
            try ordered_keys.append(allocator, rec.engine_key);
            break :blk ch;
        };
        try labels.append(allocator, label);
    }
    if (ordered_keys.items.len == 0) return allocator.dupe(u8, "");
    var out: std.Io.Writer.Allocating = .init(allocator);
    try out.writer.writeAll(labels.items);
    try out.writer.writeAll(" (");
    for (ordered_keys.items, 0..) |key, idx| {
        if (idx != 0) try out.writer.writeAll(", ");
        const ch = names.get(key).?;
        var eng_name: []const u8 = key;
        for (records) |rec| {
            if (std.mem.eql(u8, rec.engine_key, key)) {
                eng_name = rec.engine;
                break;
            }
        }
        try out.writer.print("{c}={s}", .{ ch, eng_name });
    }
    try out.writer.writeAll(")");
    return out.toOwnedSlice();
}

pub fn renderTable(
    allocator: Allocator,
    results: []const EngineResult,
    opts: struct {
        profile: []const u8,
        warmups: usize,
        runs: usize,
        fixture: []const u8,
        fixture_sha: []const u8,
        host: HostIdentity,
        schedule_records: []const SampleRecord = &.{},
        affinity: ?AffinityInfo = null,
        guest_args: []const []const u8 = &coremark_guest_args_default,
        expected_iterations: i64 = expected_iterations,
    },
) ![]u8 {
    if (results.len < 2) return error.InsufficientEngines;
    for (results) |result| {
        if (result.values.len != opts.runs) return error.UnexpectedMeasuredSampleCount;
    }
    const baseline = results[0];
    const target = results[1];
    const delta_pct = try computeDeltaPct(allocator, baseline.values, target.values);
    const sign = if (delta_pct >= 0.0) "+" else "";
    const guest_joined = try std.mem.join(allocator, " ", opts.guest_args);

    var out: std.Io.Writer.Allocating = .init(allocator);
    const w = &out.writer;
    try w.writeAll("### CoreMark cross-engine comparison\n\n");
    try w.print(
        "- Measurement profile: `{s}` ({d} warmups discarded, {d} measured runs per engine)\n",
        .{ opts.profile, opts.warmups, opts.runs },
    );
    try w.print("- Fixture: `{s}` (`sha256:{s}`)\n", .{ opts.fixture, opts.fixture_sha });
    try w.print(
        "- Fixed guest args: `{s}`; required `Iterations: {d}` and `{s}`\n",
        .{ guest_joined, opts.expected_iterations, performance_marker },
    );
    try w.print("- WAMR optimize mode: `{s}`; Wasmtime mode: `default JIT`\n", .{target.optimize});
    try w.print(
        "- CRC validation: all {d} invocations per measured engine produced exactly one `{s}`, the fixed iteration count, and no CoreMark CRC error; warmups were discarded\n",
        .{ opts.warmups + opts.runs, validation_text },
    );
    try w.writeAll("- Methodology correction: supersedes non-authoritative run `33576430466`, which allowed independent self-calibration and separate unpinned engine blocks\n\n");
    try w.writeAll("| Engine | Version / ref | Optimize mode | Mean iter/s | Median | Range | Samples (iter/s) |\n");
    try w.writeAll("|---|---|---|---:|---:|---:|---|\n");

    for (results) |result| {
        const stats = try fmtStats(allocator, result.values);
        const mean_s = try decimal_format.fixed(allocator, stats.mean, 1);
        const med_s = try decimal_format.fixed(allocator, stats.median, 1);
        const min_s = try decimal_format.fixed(allocator, stats.min, 1);
        const max_s = try decimal_format.fixed(allocator, stats.max, 1);
        const samples_s = try formatSamples(allocator, result.values);
        try w.print(
            "| {s} | `{s}` | `{s}` | {s} | {s} | {s}–{s} | {s} |\n",
            .{ result.engine, result.version, result.optimize, mean_s, med_s, min_s, max_s, samples_s },
        );
    }

    const delta_s = try decimal_format.fixed(allocator, delta_pct, 2);
    try w.print("\n**WAMR target vs baseline:** {s}{s}%\n", .{ sign, delta_s });

    const wasmtime_results = results[2..];
    if (wasmtime_results.len != 0) {
        try w.writeAll("\n| Same-host ratio | Median iter/s ratio | Mean iter/s ratio |\n");
        try w.writeAll("|---|---:|---:|\n");
        for (wasmtime_results) |result| {
            const rstats = try computeRatioStats(allocator, target.values, result.values);
            const med_r = try decimal_format.fixed(allocator, rstats.median_ratio, 6);
            const mean_r = try decimal_format.fixed(allocator, rstats.mean_ratio, 6);
            try w.print(
                "| WAMR target / {s} `{s}` | {s}× | {s}× |\n",
                .{ result.engine, result.version, med_r, mean_r },
            );
        }
    }

    if (opts.affinity) |aff| {
        var allowed_str: std.Io.Writer.Allocating = .init(allocator);
        for (aff.allowed_cpus, 0..) |cpu, idx| {
            if (idx != 0) try allowed_str.writer.writeAll(",");
            try allowed_str.writer.print("{d}", .{cpu});
        }
        try w.print(
            "\n- CPU affinity: allowed `{s}`; selected and verified CPU `{d}` via `{s}`\n",
            .{ try allowed_str.toOwnedSlice(), aff.selected_cpu, aff.taskset },
        );
    }

    if (opts.schedule_records.len != 0) {
        const sched_lbl = try scheduleLabel(allocator, opts.schedule_records);
        try w.print(
            "- Counterbalanced order: `{s}`\n\n",
            .{sched_lbl},
        );
        try w.writeAll("| Position | Phase | Engine | Engine sample | Started (UTC) | Completed (UTC) | Iterations | Iter/s |\n");
        try w.writeAll("|---:|---|---|---:|---|---|---:|---:|\n");
        for (opts.schedule_records) |rec| {
            const iter_s = try decimal_format.fixed(allocator, rec.iterations_per_second, 1);
            try w.print(
                "| {d} | {s} | {s} | {d} | `{s}` | `{s}` | {d} | {s} |\n",
                .{
                    rec.schedule_position,
                    rec.phase,
                    rec.engine,
                    rec.engine_ordinal,
                    rec.started_at,
                    rec.completed_at,
                    rec.iterations,
                    iter_s,
                },
            );
        }
    }

    try w.print("\n{s}", .{try hostInfo(allocator, opts.host)});
    return out.toOwnedSlice();
}

pub fn buildJsonReport(
    allocator: Allocator,
    results: []const EngineResult,
    opts: struct {
        profile: []const u8,
        warmups: usize,
        runs: usize,
        fixture: []const u8,
        fixture_sha: []const u8,
        host: HostIdentity,
        schedule_records: []const SampleRecord,
        affinity: ?AffinityInfo,
        provenance: Value,
        guest_args: []const []const u8 = &coremark_guest_args_default,
        expected_iterations: i64 = expected_iterations,
    },
) !Value {
    var engines_arr: std.json.Array = .init(allocator);
    for (results, 0..) |result, index| {
        const stats = try fmtStats(allocator, result.values);
        const role = if (index == 0)
            "wamr-baseline"
        else if (index == 1)
            "wamr-target"
        else blk: {
            if (result.identity == .object) {
                if (result.identity.object.get("channel")) |ch| {
                    if (ch == .string and std.mem.eql(u8, ch.string, "historical-pin")) {
                        break :blk "wasmtime-baseline";
                    }
                }
            }
            break :blk "wasmtime-context";
        };

        var pos_arr: std.json.Array = .init(allocator);
        for (result.samples) |s| {
            try pos_arr.append(.{ .integer = @intCast(s.schedule_position) });
        }
        var vals_arr: std.json.Array = .init(allocator);
        for (result.values) |v| {
            try vals_arr.append(.{ .float = v });
        }

        var eng_obj: std.json.ObjectMap = .empty;
        try eng_obj.put(allocator, "role", .{ .string = role });
        try eng_obj.put(allocator, "engine", .{ .string = result.engine });
        try eng_obj.put(allocator, "version", .{ .string = result.version });
        try eng_obj.put(allocator, "optimize", .{ .string = result.optimize });
        try eng_obj.put(allocator, "identity", result.identity);
        try eng_obj.put(allocator, "sample_schedule_positions", .{ .array = pos_arr });
        try eng_obj.put(allocator, "mean_iterations_per_second", .{ .float = stats.mean });
        try eng_obj.put(allocator, "median_iterations_per_second", .{ .float = stats.median });
        try eng_obj.put(allocator, "minimum_iterations_per_second", .{ .float = stats.min });
        try eng_obj.put(allocator, "maximum_iterations_per_second", .{ .float = stats.max });
        try eng_obj.put(allocator, "values", .{ .array = vals_arr });
        try engines_arr.append(.{ .object = eng_obj });
    }

    var ratios_arr: std.json.Array = .init(allocator);
    const target = results[1];
    for (results[2..]) |result| {
        const rstats = try computeRatioStats(allocator, target.values, result.values);
        var r_obj: std.json.ObjectMap = .empty;
        try r_obj.put(allocator, "numerator", .{ .string = target.engine });
        try r_obj.put(allocator, "denominator", .{ .string = result.engine });
        try r_obj.put(allocator, "median_ratio", .{ .float = rstats.median_ratio });
        try r_obj.put(allocator, "mean_ratio", .{ .float = rstats.mean_ratio });
        try ratios_arr.append(.{ .object = r_obj });
    }

    var fixture_obj: std.json.ObjectMap = .empty;
    try fixture_obj.put(allocator, "path", .{ .string = opts.fixture });
    try fixture_obj.put(allocator, "sha256", .{ .string = opts.fixture_sha });

    var guest_arr: std.json.Array = .init(allocator);
    for (opts.guest_args) |arg| try guest_arr.append(.{ .string = arg });

    var host_obj: std.json.ObjectMap = .empty;
    try host_obj.put(allocator, "arch", .{ .string = opts.host.arch });
    try host_obj.put(allocator, "cpu_count", .{ .integer = @intCast(opts.host.cpu_count) });
    try host_obj.put(allocator, "cpu_model", .{ .string = opts.host.cpu_model });
    try host_obj.put(allocator, "runner_name", .{ .string = opts.host.runner_name });
    try host_obj.put(allocator, "fingerprint", .{ .string = try opts.host.fingerprint(allocator) });

    const affinity_val: Value = if (opts.affinity) |aff| blk: {
        var allowed_arr: std.json.Array = .init(allocator);
        for (aff.allowed_cpus) |cpu| try allowed_arr.append(.{ .integer = @intCast(cpu) });
        var aff_obj: std.json.ObjectMap = .empty;
        try aff_obj.put(allocator, "allowed_cpus", .{ .array = allowed_arr });
        try aff_obj.put(allocator, "selected_cpu", .{ .integer = @intCast(aff.selected_cpu) });
        try aff_obj.put(allocator, "taskset", .{ .string = aff.taskset });
        try aff_obj.put(allocator, "verified", .{ .bool = true });
        break :blk .{ .object = aff_obj };
    } else .null;

    var sched_arr: std.json.Array = .init(allocator);
    for (opts.schedule_records) |rec| {
        try sched_arr.append(try rec.toJsonValue(allocator));
    }

    var cmp_obj: std.json.ObjectMap = .empty;
    try cmp_obj.put(allocator, "baseline_role", .{ .string = "wamr-baseline" });
    try cmp_obj.put(allocator, "target_role", .{ .string = "wamr-target" });
    try cmp_obj.put(allocator, "mean_delta_pct", .{ .float = try computeDeltaPct(allocator, results[0].values, results[1].values) });
    try cmp_obj.put(allocator, "median_delta_pct", .{ .float = try computeMedianDeltaPct(allocator, results[0].values, results[1].values) });

    var report_obj: std.json.ObjectMap = .empty;
    try report_obj.put(allocator, "schema_version", .{ .integer = report_schema_version });
    try report_obj.put(allocator, "kind", .{ .string = report_kind });
    try report_obj.put(allocator, "provenance", opts.provenance);
    try report_obj.put(allocator, "profile", .{ .string = opts.profile });
    try report_obj.put(allocator, "warmups_per_engine", .{ .integer = @intCast(opts.warmups) });
    try report_obj.put(allocator, "measured_samples_per_engine", .{ .integer = @intCast(opts.runs) });
    try report_obj.put(allocator, "fixture", .{ .object = fixture_obj });
    try report_obj.put(allocator, "guest_args", .{ .array = guest_arr });
    try report_obj.put(allocator, "expected_iterations", .{ .integer = opts.expected_iterations });
    try report_obj.put(allocator, "required_performance_marker", .{ .string = performance_marker });
    try report_obj.put(allocator, "required_crc_marker", .{ .string = validation_text });
    try report_obj.put(allocator, "invalidated_run", .{ .integer = 33576430466 });
    try report_obj.put(allocator, "host", .{ .object = host_obj });
    try report_obj.put(allocator, "affinity", affinity_val);
    try report_obj.put(allocator, "schedule", .{ .array = sched_arr });
    try report_obj.put(allocator, "engines", .{ .array = engines_arr });
    try report_obj.put(allocator, "ratios", .{ .array = ratios_arr });
    try report_obj.put(allocator, "wamr_comparison", .{ .object = cmp_obj });

    const report_val: Value = .{ .object = report_obj };
    const auth_counts = profileCounts("authoritative").?;
    if (std.mem.eql(u8, opts.profile, "authoritative") and opts.warmups == auth_counts.warmups and opts.runs == auth_counts.runs) {
        try validateAuthoritativeBenchmarkReport(allocator, report_val);
    }
    return report_val;
}

pub const BenchmarkReportStatus = struct {
    status: []const u8,
    authoritative: bool,
    reason: ?[]const u8 = null,
};

pub fn benchmarkReportStatus(allocator: Allocator, report: Value) !BenchmarkReportStatus {
    if (report != .object) return error.InvalidBenchmarkReport;
    const obj = report.object;
    const kind_val = obj.get("kind") orelse return error.UnsupportedReportKind;
    if (kind_val != .string or !std.mem.eql(u8, kind_val.string, report_kind)) {
        return error.UnsupportedReportKind;
    }
    if (obj.get("schema_version")) |sv| {
        if (sv == .integer and sv.integer == 1) {
            return .{
                .status = "legacy-unverified",
                .authoritative = false,
                .reason = "schema version 1 predates explicit source, tool, run, and report provenance",
            };
        }
    }
    try validateAuthoritativeBenchmarkReport(allocator, report);
    return .{ .status = "current", .authoritative = true };
}

pub fn positiveFiniteNumber(val_opt: ?Value) ?f64 {
    const val = val_opt orelse return null;
    const num: f64 = switch (val) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return null,
    };
    if (!std.math.isFinite(num) or num <= 0.0 or num > std.math.floatMax(f64)) return null;
    return num;
}

pub fn validateAuthoritativeBenchmarkReport(allocator: Allocator, report_val: Value) !void {
    if (report_val != .object) return error.InvalidBenchmarkReport;
    const report = report_val.object;
    const sv = report.get("schema_version") orelse return error.UnsupportedSchemaVersion;
    if (sv != .integer) return error.UnsupportedSchemaVersion;
    if (sv.integer == 1) return error.LegacyUnverifiedSchemaVersion;
    if (sv.integer != report_schema_version) return error.UnsupportedSchemaVersion;

    const kind_val = report.get("kind") orelse return error.UnsupportedReportKind;
    if (kind_val != .string or !std.mem.eql(u8, kind_val.string, report_kind)) return error.UnsupportedReportKind;

    const prof_val = report.get("profile") orelse return error.NonAuthoritativeProfile;
    if (prof_val != .string or !std.mem.eql(u8, prof_val.string, "authoritative")) return error.NonAuthoritativeProfile;

    const auth_counts = profileCounts("authoritative").?;
    const warmups_val = report.get("warmups_per_engine") orelse return error.WrongAuthoritativeWarmupCount;
    if (warmups_val != .integer or warmups_val.integer != @as(i64, @intCast(auth_counts.warmups))) {
        return error.WrongAuthoritativeWarmupCount;
    }
    const runs_val = report.get("measured_samples_per_engine") orelse return error.WrongAuthoritativeSampleCount;
    if (runs_val != .integer or runs_val.integer != @as(i64, @intCast(auth_counts.runs))) {
        return error.WrongAuthoritativeSampleCount;
    }

    const fixture_val = report.get("fixture") orelse return error.MissingFixtureIdentity;
    if (fixture_val != .object) return error.MissingFixtureIdentity;
    _ = try requireDigestValue(fixture_val.object.get("sha256"), 64);

    const guest_val = report.get("guest_args") orelse return error.NonAuthoritativeGuestArgs;
    if (guest_val != .array or guest_val.array.items.len != coremark_guest_args_default.len) {
        return error.NonAuthoritativeGuestArgs;
    }
    for (guest_val.array.items, coremark_guest_args_default) |actual, expected| {
        if (actual != .string or !std.mem.eql(u8, actual.string, expected)) {
            return error.NonAuthoritativeGuestArgs;
        }
    }

    const iters_val = report.get("expected_iterations") orelse return error.NonAuthoritativeIterationCount;
    if (iters_val != .integer or iters_val.integer != expected_iterations) {
        return error.NonAuthoritativeIterationCount;
    }
    const perf_val = report.get("required_performance_marker") orelse return error.WrongPerformanceMarker;
    if (perf_val != .string or !std.mem.eql(u8, perf_val.string, performance_marker)) {
        return error.WrongPerformanceMarker;
    }
    const crc_val = report.get("required_crc_marker") orelse return error.WrongCrcMarker;
    if (crc_val != .string or !std.mem.eql(u8, crc_val.string, validation_text)) {
        return error.WrongCrcMarker;
    }

    const prov_val = report.get("provenance") orelse return error.MissingRequiredProvenance;
    if (prov_val != .object) return error.MissingRequiredProvenance;
    const prov = prov_val.object;
    const rep_id = prov.get("report_id") orelse return error.InvalidReportId;
    if (rep_id != .string or !isValidUuid(rep_id.string)) return error.InvalidReportId;
    const gen_at = prov.get("generated_at") orelse return error.MissingGenerationTimestamp;
    if (gen_at != .string or gen_at.string.len == 0) return error.MissingGenerationTimestamp;

    const prod_val = prov.get("producer") orelse return error.MissingProducerProvenance;
    if (prod_val != .object) return error.MissingProducerProvenance;
    _ = try requireDigestValue(prod_val.object.get("source_sha"), 40);
    const script_val = prod_val.object.get("script") orelse return error.MissingProducerScriptProvenance;
    if (script_val != .object) return error.MissingProducerScriptProvenance;
    const spath = script_val.object.get("path") orelse return error.MissingProducerScriptProvenance;
    if (spath != .string or spath.string.len == 0) return error.MissingProducerScriptProvenance;
    _ = try requireDigestValue(script_val.object.get("sha256"), 64);

    const exec_val = prov.get("execution") orelse return error.MissingExecutionProvenance;
    if (exec_val != .object) return error.MissingExecutionProvenance;
    const provider_val = exec_val.object.get("provider") orelse return error.MissingExecutionProvenance;
    const run_id_val = exec_val.object.get("run_id") orelse return error.MissingExecutionProvenance;
    if (provider_val != .string or
        (!std.mem.eql(u8, provider_val.string, "github-actions") and !std.mem.eql(u8, provider_val.string, "local")) or
        run_id_val != .string or run_id_val.string.len == 0)
    {
        return error.MissingExecutionProvenance;
    }

    const host_val = report.get("host") orelse return error.MissingHostIdentity;
    if (host_val != .object) return error.MissingHostIdentity;
    const host = host_val.object;
    const h_arch = host.get("arch") orelse return error.MissingHostIdentity;
    const h_model = host.get("cpu_model") orelse return error.MissingHostIdentity;
    const h_cpus = host.get("cpu_count") orelse return error.MissingHostIdentity;
    const h_fp = host.get("fingerprint") orelse return error.MissingHostIdentity;
    if (h_arch != .string or h_arch.string.len == 0 or
        h_model != .string or h_model.string.len == 0 or
        h_cpus != .integer or h_cpus.integer <= 0 or
        h_fp != .string or h_fp.string.len == 0)
    {
        return error.MissingHostIdentity;
    }

    const aff_val = report.get("affinity") orelse return error.MissingVerifiedCpuAffinity;
    if (aff_val != .object) return error.MissingVerifiedCpuAffinity;
    const ver_val = aff_val.object.get("verified") orelse return error.MissingVerifiedCpuAffinity;
    if (ver_val != .bool or !ver_val.bool) return error.MissingVerifiedCpuAffinity;

    const engines_val = report.get("engines") orelse return error.MissingBenchmarkEngines;
    if (engines_val != .array or engines_val.array.items.len < 2) return error.MissingBenchmarkEngines;

    var baseline_count: usize = 0;
    var target_count: usize = 0;
    var baseline_vals: []f64 = &.{};
    var target_vals: []f64 = &.{};

    for (engines_val.array.items) |eng_val| {
        if (eng_val != .object) return error.InvalidEngineEntry;
        const eng = eng_val.object;
        const role_val = eng.get("role") orelse return error.InvalidEngineEntry;
        if (role_val != .string) return error.InvalidEngineEntry;
        const role = role_val.string;
        if (std.mem.eql(u8, role, "wamr-baseline")) baseline_count += 1;
        if (std.mem.eql(u8, role, "wamr-target")) target_count += 1;

        const identity = try validateEngineIdentity(eng.get("identity"), role);
        const id_type = identity.get("type").?.string;
        if (std.mem.startsWith(u8, role, "wamr-") and !std.mem.eql(u8, id_type, "wamr")) {
            return error.InvalidWamrRoleIdentity;
        }
        if (std.mem.startsWith(u8, role, "wasmtime-") and !std.mem.eql(u8, id_type, "wasmtime")) {
            return error.InvalidWasmtimeRoleIdentity;
        }

        const values_val = eng.get("values") orelse return error.InvalidMeasuredSampleCount;
        if (values_val != .array or values_val.array.items.len != auth_counts.runs) {
            return error.InvalidMeasuredSampleCount;
        }
        const parsed_vals = try allocator.alloc(f64, values_val.array.items.len);
        for (values_val.array.items, 0..) |v, idx| {
            parsed_vals[idx] = positiveFiniteNumber(v) orelse return error.InvalidMeasuredSamples;
        }
        if (std.mem.eql(u8, role, "wamr-baseline")) baseline_vals = parsed_vals;
        if (std.mem.eql(u8, role, "wamr-target")) target_vals = parsed_vals;
    }

    if (baseline_count != 1 or target_count != 1) return error.MissingWamrBaselineOrTarget;

    if (report.get("wamr_comparison")) |cmp_val| {
        if (cmp_val != .null) {
            if (cmp_val != .object) return error.InvalidWamrComparison;
            const cmp = cmp_val.object;
            const b_role = cmp.get("baseline_role") orelse return error.InvalidWamrComparisonRoles;
            const t_role = cmp.get("target_role") orelse return error.InvalidWamrComparisonRoles;
            if (b_role != .string or !std.mem.eql(u8, b_role.string, "wamr-baseline") or
                t_role != .string or !std.mem.eql(u8, t_role.string, "wamr-target"))
            {
                return error.InvalidWamrComparisonRoles;
            }
            const mean_delta: f64 = switch (cmp.get("mean_delta_pct") orelse return error.WamrComparisonDisagreesWithSamples) {
                .float => |f| f,
                .integer => |i| @floatFromInt(i),
                else => return error.WamrComparisonDisagreesWithSamples,
            };
            const median_delta: f64 = switch (cmp.get("median_delta_pct") orelse return error.WamrComparisonDisagreesWithSamples) {
                .float => |f| f,
                .integer => |i| @floatFromInt(i),
                else => return error.WamrComparisonDisagreesWithSamples,
            };
            const exp_mean = try computeDeltaPct(allocator, baseline_vals, target_vals);
            const exp_median = try computeMedianDeltaPct(allocator, baseline_vals, target_vals);
            if (@abs(mean_delta - exp_mean) > 1e-12 or @abs(median_delta - exp_median) > 1e-12) {
                return error.WamrComparisonDisagreesWithSamples;
            }
        }
    }
}

pub fn engineWithRole(report_obj: std.json.ObjectMap, role: []const u8) !Value {
    const engines_val = report_obj.get("engines") orelse return error.MissingBenchmarkEngines;
    if (engines_val != .array) return error.MissingBenchmarkEngines;
    var found: ?Value = null;
    for (engines_val.array.items) |eng| {
        if (eng == .object) {
            if (eng.object.get("role")) |r| {
                if (r == .string and std.mem.eql(u8, r.string, role)) {
                    if (found != null) return error.DuplicateEngineRole;
                    found = eng;
                }
            }
        }
    }
    return found orelse error.MissingEngineRole;
}

pub fn validateExecutionMatch(benchmark_execution: Value, current_execution: Value) !void {
    if (benchmark_execution != .object or current_execution != .object) return error.ExecutionProviderMismatch;
    const b_obj = benchmark_execution.object;
    const c_obj = current_execution.object;
    const b_prov = b_obj.get("provider") orelse return error.ExecutionProviderMismatch;
    const c_prov = c_obj.get("provider") orelse return error.ExecutionProviderMismatch;
    if (b_prov != .string or c_prov != .string or !std.mem.eql(u8, b_prov.string, c_prov.string)) {
        return error.ExecutionProviderMismatch;
    }
    if (std.mem.eql(u8, b_prov.string, "github-actions")) {
        for ([_][]const u8{ "repository", "run_id", "run_attempt" }) |field| {
            const bv = b_obj.get(field) orelse return error.GithubActionsExecutionMismatch;
            const cv = c_obj.get(field) orelse return error.GithubActionsExecutionMismatch;
            if (bv != .string or cv != .string or !std.mem.eql(u8, bv.string, cv.string)) {
                return error.GithubActionsExecutionMismatch;
            }
        }
    } else {
        const b_id = b_obj.get("run_id") orelse return error.MissingLocalExecutionId;
        const c_id = c_obj.get("run_id") orelse return error.MissingLocalExecutionId;
        if (b_id != .string or c_id != .string or b_id.string.len == 0 or c_id.string.len == 0) {
            return error.MissingLocalExecutionId;
        }
        if (!std.mem.eql(u8, b_id.string, c_id.string)) {
            return error.LocalExecutionIdMismatch;
        }
    }
}

pub fn validateBenchmarkProfileMatch(
    allocator: Allocator,
    report_val: Value,
    opts: struct {
        expected_arch: []const u8,
        fixture_sha: []const u8,
        host: HostIdentity,
        affinity: AffinityInfo,
        wamr_source_sha: []const u8,
        wamr_optimize: []const u8,
        wamr_runtime_sha: []const u8,
        wamr_compiler_sha: []const u8,
        wamr_module_sha: []const u8,
        wasmtime_version_value: []const u8,
        wasmtime_runtime_sha: []const u8,
        producer_source_sha: []const u8,
        producer_script_sha: []const u8,
        current_execution: Value,
        benchmark_role: []const u8 = "wamr-target",
    },
) !Value {
    try validateAuthoritativeBenchmarkReport(allocator, report_val);
    const report = report_val.object;
    const host_obj = report.get("host").?.object;
    if (!std.mem.eql(u8, host_obj.get("arch").?.string, opts.expected_arch)) {
        return error.BenchmarkArchMismatch;
    }
    if (!std.mem.eql(u8, report.get("fixture").?.object.get("sha256").?.string, opts.fixture_sha)) {
        return error.BenchmarkFixtureHashMismatch;
    }
    const exp_fp = try opts.host.fingerprint(allocator);
    if (!std.mem.eql(u8, host_obj.get("arch").?.string, opts.host.arch) or
        host_obj.get("cpu_count").?.integer != @as(i64, @intCast(opts.host.cpu_count)) or
        !std.mem.eql(u8, host_obj.get("cpu_model").?.string, opts.host.cpu_model) or
        !std.mem.eql(u8, host_obj.get("runner_name").?.string, opts.host.runner_name) or
        !std.mem.eql(u8, host_obj.get("fingerprint").?.string, exp_fp))
    {
        return error.BenchmarkHostMismatch;
    }

    const aff_obj = report.get("affinity").?.object;
    const allowed_arr = aff_obj.get("allowed_cpus") orelse return error.BenchmarkAffinityMismatch;
    if (allowed_arr != .array or allowed_arr.array.items.len != opts.affinity.allowed_cpus.len) {
        return error.BenchmarkAffinityMismatch;
    }
    for (allowed_arr.array.items, opts.affinity.allowed_cpus) |actual_cpu, exp_cpu| {
        if (actual_cpu != .integer or actual_cpu.integer != @as(i64, @intCast(exp_cpu))) {
            return error.BenchmarkAffinityMismatch;
        }
    }
    if (aff_obj.get("selected_cpu").?.integer != @as(i64, @intCast(opts.affinity.selected_cpu)) or
        !std.mem.eql(u8, aff_obj.get("taskset").?.string, opts.affinity.taskset) or
        !aff_obj.get("verified").?.bool)
    {
        return error.BenchmarkAffinityMismatch;
    }

    if (!std.mem.eql(u8, opts.benchmark_role, "wamr-baseline") and !std.mem.eql(u8, opts.benchmark_role, "wamr-target")) {
        return error.UnsupportedBenchmarkRole;
    }
    const selected = try engineWithRole(report, opts.benchmark_role);
    const sel_id = selected.object.get("identity").?.object;
    if (!std.mem.eql(u8, sel_id.get("source").?.object.get("sha").?.string, opts.wamr_source_sha)) {
        return error.WamrSourceShaMismatch;
    }
    if (!std.mem.eql(u8, sel_id.get("runtime").?.object.get("sha256").?.string, opts.wamr_runtime_sha)) {
        return error.WamrRuntimeShaMismatch;
    }
    if (!std.mem.eql(u8, sel_id.get("compiler").?.object.get("sha256").?.string, opts.wamr_compiler_sha)) {
        return error.WamrCompilerShaMismatch;
    }
    if (!std.mem.eql(u8, sel_id.get("module").?.object.get("sha256").?.string, opts.wamr_module_sha)) {
        return error.WamrModuleShaMismatch;
    }
    if (!std.mem.eql(u8, sel_id.get("optimize").?.string, opts.wamr_optimize)) {
        return error.WamrOptimizeModeMismatch;
    }

    const wasmtime = try engineWithRole(report, "wasmtime-baseline");
    const wt_id = wasmtime.object.get("identity").?.object;
    if (!std.mem.eql(u8, wt_id.get("channel").?.string, "historical-pin")) {
        return error.WasmtimeNotHistoricalPin;
    }
    if (!std.mem.eql(u8, wt_id.get("version").?.string, opts.wasmtime_version_value)) {
        return error.WasmtimeVersionMismatch;
    }
    if (!std.mem.eql(u8, wt_id.get("runtime").?.object.get("sha256").?.string, opts.wasmtime_runtime_sha)) {
        return error.WasmtimeRuntimeShaMismatch;
    }

    const prov_obj = report.get("provenance").?.object;
    const prod_obj = prov_obj.get("producer").?.object;
    if (!std.mem.eql(u8, prod_obj.get("source_sha").?.string, opts.producer_source_sha)) {
        return error.ProducerSourceShaMismatch;
    }
    if (!std.mem.eql(u8, prod_obj.get("script").?.object.get("sha256").?.string, opts.producer_script_sha)) {
        return error.ProducerScriptShaMismatch;
    }

    const benchmark_execution = prov_obj.get("execution").?;
    try validateExecutionMatch(benchmark_execution, opts.current_execution);

    const schedule_val = report.get("schedule") orelse return error.IncompletePairedSchedule;
    if (schedule_val != .array) return error.IncompletePairedSchedule;
    const schedule = schedule_val.array.items;

    const baseline = try engineWithRole(report, "wamr-baseline");
    const target = try engineWithRole(report, "wamr-target");
    var participants: std.ArrayList(Value) = .empty;
    try participants.appendSlice(allocator, &.{ baseline, target, wasmtime });

    const b_pos = baseline.object.get("sample_schedule_positions") orelse .null;
    const t_pos = target.object.get("sample_schedule_positions") orelse .null;
    if (jsonValuesEqual(b_pos, t_pos)) {
        const left = baseline.object.get("identity").?.object;
        const right = target.object.get("identity").?.object;
        if (!std.mem.eql(u8, left.get("source").?.object.get("sha").?.string, right.get("source").?.object.get("sha").?.string) or
            !std.mem.eql(u8, left.get("optimize").?.string, right.get("optimize").?.string) or
            !std.mem.eql(u8, left.get("runtime").?.object.get("sha256").?.string, right.get("runtime").?.object.get("sha256").?.string) or
            !std.mem.eql(u8, left.get("compiler").?.object.get("sha256").?.string, right.get("compiler").?.object.get("sha256").?.string) or
            !std.mem.eql(u8, left.get("module").?.object.get("sha256").?.string, right.get("module").?.object.get("sha256").?.string) or
            !jsonValuesEqual(baseline.object.get("values").?, target.object.get("values").?))
        {
            return error.SharedSamplesMismatch;
        }
        _ = participants.orderedRemove(0);
    }

    const auth_counts = profileCounts("authoritative").?;
    const expected_samples = auth_counts.warmups + auth_counts.runs;
    const expected_schedule_len = participants.items.len * expected_samples;
    if (schedule.len != expected_schedule_len) return error.IncompletePairedSchedule;

    var position_to_engine: std.AutoHashMap(i64, []const u8) = .init(allocator);
    var participant_roles: std.ArrayList([]const u8) = .empty;
    for (participants.items) |eng| {
        const role = eng.object.get("role").?.string;
        try participant_roles.append(allocator, role);
        const pos_val = eng.object.get("sample_schedule_positions") orelse return error.IncompleteEngineSamples;
        if (pos_val != .array or pos_val.array.items.len != expected_samples) return error.IncompleteEngineSamples;
        for (pos_val.array.items) |p| {
            if (p != .integer) return error.InvalidSamplePosition;
            if (position_to_engine.contains(p.integer)) return error.DuplicateSamplePosition;
            try position_to_engine.put(p.integer, role);
        }
    }
    for (1..(expected_schedule_len + 1)) |pos| {
        if (!position_to_engine.contains(@intCast(pos))) return error.IncompleteSchedulePositions;
    }

    for (schedule) |rec_val| {
        if (rec_val != .object) return error.InvalidScheduleRecord;
        const rec = rec_val.object;
        const sp = rec.get("schedule_position") orelse return error.InvalidScheduleRecord;
        const iters = rec.get("iterations") orelse return error.InvalidScheduleRecord;
        const ph = rec.get("phase") orelse return error.InvalidScheduleRecord;
        const ek = rec.get("engine_key") orelse return error.InvalidScheduleRecord;
        if (sp != .integer or
            iters != .integer or iters.integer != expected_iterations or
            ph != .string or (!std.mem.eql(u8, ph.string, "warmup") and !std.mem.eql(u8, ph.string, "measured")) or
            ek != .string or ek.string.len == 0 or
            positiveFiniteNumber(rec.get("elapsed_seconds")) == null or
            positiveFiniteNumber(rec.get("iterations_per_second")) == null)
        {
            return error.InvalidScheduleRecord;
        }
    }

    const ordered_schedule = try allocator.dupe(Value, schedule);
    std.mem.sort(Value, ordered_schedule, {}, struct {
        fn lessThan(_: void, a: Value, b: Value) bool {
            return a.object.get("schedule_position").?.integer < b.object.get("schedule_position").?.integer;
        }
    }.lessThan);
    for (ordered_schedule, 1..) |rec_val, exp_pos| {
        if (rec_val.object.get("schedule_position").?.integer != @as(i64, @intCast(exp_pos))) {
            return error.IncompleteSchedulePositions;
        }
    }

    var participant_keys: std.StringHashMap(void) = .init(allocator);
    for (participants.items) |eng| {
        const role = eng.object.get("role").?.string;
        var single_key: ?[]const u8 = null;
        var measured_vals: std.json.Array = .init(allocator);
        for (ordered_schedule) |rec_val| {
            const rec = rec_val.object;
            const pos = rec.get("schedule_position").?.integer;
            if (!std.mem.eql(u8, position_to_engine.get(pos).?, role)) continue;
            const ek = rec.get("engine_key").?.string;
            if (single_key) |sk| {
                if (!std.mem.eql(u8, sk, ek)) return error.InconsistentScheduleEngineKeys;
            } else {
                if (participant_keys.contains(ek)) return error.InconsistentScheduleEngineKeys;
                single_key = ek;
            }
            if (std.mem.eql(u8, rec.get("phase").?.string, "measured")) {
                try measured_vals.append(rec.get("iterations_per_second").?);
            }
        }
        try participant_keys.put(single_key orelse return error.InconsistentScheduleEngineKeys, {});
        if (!jsonValuesEqual(.{ .array = measured_vals }, eng.object.get("values").?)) {
            return error.ValuesDisagreeWithMeasuredSchedule;
        }
    }

    const warmup_order = try counterbalancedOrder(allocator, participant_roles.items, @intCast(auth_counts.warmups));
    const measured_order = try counterbalancedOrder(allocator, participant_roles.items, @intCast(auth_counts.runs));
    const warmup_total = auth_counts.warmups * participants.items.len;
    for (ordered_schedule, 0..) |rec_val, idx| {
        const rec = rec_val.object;
        const exp_phase = if (idx < warmup_total) "warmup" else "measured";
        if (!std.mem.eql(u8, rec.get("phase").?.string, exp_phase)) {
            return error.UnorderedSchedulePhases;
        }
        const actual_role = position_to_engine.get(rec.get("schedule_position").?.integer).?;
        const exp_role = if (idx < warmup_total) warmup_order[idx] else measured_order[idx - warmup_total];
        if (!std.mem.eql(u8, actual_role, exp_role)) {
            return error.ScheduleOrderNotCounterbalanced;
        }
    }

    var out: std.json.ObjectMap = .empty;
    try out.put(allocator, "report_id", prov_obj.get("report_id").?);
    try out.put(allocator, "generated_at", prov_obj.get("generated_at").?);
    try out.put(allocator, "execution", benchmark_execution);
    try out.put(allocator, "producer", prov_obj.get("producer").?);
    try out.put(allocator, "selected_role", .{ .string = opts.benchmark_role });
    try out.put(allocator, "selected_wamr", selected);
    try out.put(allocator, "wasmtime_baseline", wasmtime);
    if (std.mem.eql(u8, opts.benchmark_role, "wamr-target")) {
        try out.put(allocator, "target", selected);
    }
    return .{ .object = out };
}

pub fn renderOptimizeTable(
    allocator: Allocator,
    target_ref: []const u8,
    fast_vals_opt: ?[]const f64,
    safe_vals_opt: ?[]const f64,
    opts: struct {
        profile: []const u8,
        warmups: usize,
        runs: usize,
        fixture: []const u8,
        fixture_sha: []const u8,
        host: HostIdentity,
    },
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    const w = &out.writer;
    try w.writeAll("### CoreMark AOT optimize-mode comparison\n\n");
    try w.print(
        "- Measurement profile: `{s}` ({d} warmups discarded, {d} measured runs per mode)\n",
        .{ opts.profile, opts.warmups, opts.runs },
    );
    try w.print("- Fixture: `{s}` (`sha256:{s}`)\n\n", .{ opts.fixture, opts.fixture_sha });
    try w.writeAll("| Ref | Mode | Mean iter/s | Median | Range | Samples (iter/s) |\n");
    try w.writeAll("|---|---|---:|---:|---:|---|\n");

    const modes = [_]struct { name: []const u8, vals: ?[]const f64 }{
        .{ .name = "ReleaseFast", .vals = fast_vals_opt },
        .{ .name = "ReleaseSafe", .vals = safe_vals_opt },
    };
    for (modes) |m| {
        if (m.vals) |vals| {
            const stats = try fmtStats(allocator, vals);
            const mean_s = try decimal_format.fixed(allocator, stats.mean, 1);
            const med_s = try decimal_format.fixed(allocator, stats.median, 1);
            const min_s = try decimal_format.fixed(allocator, stats.min, 1);
            const max_s = try decimal_format.fixed(allocator, stats.max, 1);
            const samples_s = try formatSamples(allocator, vals);
            try w.print(
                "| `{s}` | `{s}` | {s} | {s} | {s}–{s} | {s} |\n",
                .{ target_ref, m.name, mean_s, med_s, min_s, max_s, samples_s },
            );
        } else {
            try w.print("| `{s}` | `{s}` | failed | failed | failed | failed |\n", .{ target_ref, m.name });
        }
    }

    if (fast_vals_opt != null and safe_vals_opt != null) {
        const fast_stats = try fmtStats(allocator, fast_vals_opt.?);
        const safe_stats = try fmtStats(allocator, safe_vals_opt.?);
        const ratio_s = try benchmark_optimize.formatRatio(allocator, safe_stats.mean, fast_stats.mean);
        try w.print("\n**ReleaseSafe / ReleaseFast:** {s}\n", .{ratio_s});
    } else {
        try w.writeAll("\nAt least one optimize mode failed before producing complete timings.\n");
    }
    try w.print("\n{s}", .{try hostInfo(allocator, opts.host)});
    return out.toOwnedSlice();
}

pub fn resolveCounts(profile: []const u8, warmups_arg: ?isize, runs_arg: ?isize) !ProfileCounts {
    const defaults = profileCounts(profile) orelse return error.InvalidProfile;
    const warmups_i: isize = warmups_arg orelse @intCast(defaults.warmups);
    const runs_i: isize = runs_arg orelse @intCast(defaults.runs);
    if (warmups_i < 0) return error.NegativeWarmups;
    if (runs_i <= 0) return error.NonPositiveRuns;
    return .{
        .warmups = @intCast(warmups_i),
        .runs = @intCast(runs_i),
    };
}

pub fn profileLabel(allocator: Allocator, profile: []const u8, warmups_arg: ?isize, runs_arg: ?isize) ![]u8 {
    const defaults = profileCounts(profile) orelse return error.InvalidProfile;
    const resolved = try resolveCounts(profile, warmups_arg, runs_arg);
    if (resolved.warmups == defaults.warmups and resolved.runs == defaults.runs) {
        return allocator.dupe(u8, profile);
    }
    return std.fmt.allocPrint(allocator, "{s} (overridden)", .{profile});
}

pub fn stringifyPrettySorted(allocator: Allocator, value: Value) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    try writeJsonSorted(&out.writer, allocator, value, 0);
    return out.toOwnedSlice();
}

fn writeIndent(w: *std.Io.Writer, depth: usize) !void {
    for (0..depth) |_| try w.writeAll("  ");
}

fn writeJsonSorted(w: *std.Io.Writer, allocator: Allocator, value: Value, depth: usize) !void {
    switch (value) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .float => |f| try w.print("{d}", .{f}),
        .number_string => |s| try w.writeAll(s),
        .string => |s| try std.json.Stringify.encodeJsonString(s, .{}, w),
        .array => |arr| {
            if (arr.items.len == 0) {
                try w.writeAll("[]");
                return;
            }
            try w.writeAll("[\n");
            for (arr.items, 0..) |item, idx| {
                try writeIndent(w, depth + 1);
                try writeJsonSorted(w, allocator, item, depth + 1);
                if (idx + 1 < arr.items.len) try w.writeAll(",");
                try w.writeAll("\n");
            }
            try writeIndent(w, depth);
            try w.writeAll("]");
        },
        .object => |obj| {
            if (obj.count() == 0) {
                try w.writeAll("{}");
                return;
            }
            const keys = try allocator.alloc([]const u8, obj.count());
            var it = obj.iterator();
            var idx: usize = 0;
            while (it.next()) |entry| : (idx += 1) {
                keys[idx] = entry.key_ptr.*;
            }
            std.mem.sort([]const u8, keys, {}, struct {
                fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lessThan);
            try w.writeAll("{\n");
            for (keys, 0..) |k, k_idx| {
                try writeIndent(w, depth + 1);
                try std.json.Stringify.encodeJsonString(k, .{}, w);
                try w.writeAll(": ");
                try writeJsonSorted(w, allocator, obj.get(k).?, depth + 1);
                if (k_idx + 1 < keys.len) try w.writeAll(",");
                try w.writeAll("\n");
            }
            try writeIndent(w, depth);
            try w.writeAll("}");
        },
    }
}

pub fn runCli(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    argv: []const []const u8,
) !u8 {
    if (argv.len > 0 and std.mem.startsWith(u8, argv[0], "native-")) {
        return native_benchmark.runCli(allocator, io, environ_map, argv);
    }

    var baseline: []const u8 = "origin/main";
    var target: []const u8 = "HEAD";
    var profile: []const u8 = "authoritative";
    var warmups_arg: ?isize = null;
    var runs_arg: ?isize = null;
    var fixture_arg: []const u8 = default_fixture_rel;
    var repo_arg: []const u8 = ".";
    var out_path: ?[]const u8 = null;
    var json_out_path: ?[]const u8 = null;
    var execution_id: ?[]const u8 = null;
    var retain_baseline_artifacts: ?[]const u8 = null;
    var retain_target_artifacts: ?[]const u8 = null;
    var emit: []const u8 = "markdown";
    var optimize_arg: []const u8 = "ReleaseFast";
    var wasmtime_baseline: ?[]const u8 = null;
    var wasmtime_arg: ?[]const u8 = null;
    var wasmtime_cache: ?[]const u8 = null;
    var require_native_arch: ?[]const u8 = null;
    var min_delta_pct: ?f64 = null;

    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const raw_arg = argv[i];
        var flag = raw_arg;
        var inline_val: ?[]const u8 = null;
        if (std.mem.indexOfScalar(u8, raw_arg, '=')) |eq| {
            flag = raw_arg[0..eq];
            inline_val = raw_arg[eq + 1 ..];
        }
        const nextVal = struct {
            fn get(args: []const []const u8, idx: *usize, inl: ?[]const u8) ![]const u8 {
                if (inl) |v| return v;
                idx.* += 1;
                if (idx.* >= args.len) return error.InvalidCliArguments;
                return args[idx.*];
            }
        }.get;

        if (std.mem.eql(u8, flag, "--baseline")) {
            baseline = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--target")) {
            target = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--profile")) {
            profile = try nextVal(argv, &i, inline_val);
            if (profileCounts(profile) == null) return error.InvalidCliArguments;
        } else if (std.mem.eql(u8, flag, "--warmups")) {
            warmups_arg = std.fmt.parseInt(isize, try nextVal(argv, &i, inline_val), 10) catch return error.InvalidCliArguments;
        } else if (std.mem.eql(u8, flag, "--runs")) {
            runs_arg = std.fmt.parseInt(isize, try nextVal(argv, &i, inline_val), 10) catch return error.InvalidCliArguments;
        } else if (std.mem.eql(u8, flag, "--fixture")) {
            fixture_arg = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--repo")) {
            repo_arg = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--out")) {
            out_path = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--json-out")) {
            json_out_path = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--execution-id")) {
            execution_id = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--retain-baseline-artifacts")) {
            retain_baseline_artifacts = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--retain-target-artifacts")) {
            retain_target_artifacts = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--emit")) {
            emit = try nextVal(argv, &i, inline_val);
            if (!std.mem.eql(u8, emit, "markdown") and !std.mem.eql(u8, emit, "github")) return error.InvalidCliArguments;
        } else if (std.mem.eql(u8, flag, "--optimize")) {
            optimize_arg = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--wasmtime-baseline")) {
            wasmtime_baseline = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--wasmtime")) {
            wasmtime_arg = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--wasmtime-cache")) {
            wasmtime_cache = try nextVal(argv, &i, inline_val);
        } else if (std.mem.eql(u8, flag, "--require-native-arch")) {
            require_native_arch = try nextVal(argv, &i, inline_val);
            if (!std.mem.eql(u8, require_native_arch.?, "aarch64") and !std.mem.eql(u8, require_native_arch.?, "x86_64")) {
                return error.InvalidCliArguments;
            }
        } else if (std.mem.eql(u8, flag, "--min-delta-pct")) {
            min_delta_pct = std.fmt.parseFloat(f64, try nextVal(argv, &i, inline_val)) catch return error.InvalidCliArguments;
        } else {
            return error.InvalidCliArguments;
        }
    }

    const counts = resolveCounts(profile, warmups_arg, runs_arg) catch return error.InvalidCliArguments;
    const effective_profile = try profileLabel(allocator, profile, warmups_arg, runs_arg);
    const exp_iters = profileIterations(profile) orelse return error.InvalidCliArguments;
    const guest_args = try coremarkGuestArgs(allocator, exp_iters);
    const optimize_modes = benchmark_optimize.parseModes(optimize_arg) catch return error.InvalidCliArguments;

    if (std.mem.eql(u8, optimize_arg, "both") and (wasmtime_baseline != null or wasmtime_arg != null)) {
        return error.InvalidCliArguments;
    }
    if (std.mem.eql(u8, optimize_arg, "both") and json_out_path != null) {
        return error.InvalidCliArguments;
    }
    const has_retained = retain_baseline_artifacts != null or retain_target_artifacts != null;
    if (has_retained and json_out_path == null) return error.InvalidCliArguments;
    if (has_retained and wasmtime_baseline == null and wasmtime_arg == null) return error.InvalidCliArguments;
    if (has_retained and !std.mem.eql(u8, profile, "authoritative")) return error.InvalidCliArguments;
    if (retain_baseline_artifacts != null and retain_target_artifacts != null) {
        if (std.mem.eql(u8, retain_baseline_artifacts.?, retain_target_artifacts.?)) {
            return error.InvalidCliArguments;
        }
    }

    const repo = try std.Io.Dir.cwd().realPathFileAlloc(io, repo_arg, allocator);
    const resolved_fixture = try resolveFixture(allocator, io, repo, fixture_arg);
    const host = if (require_native_arch) |req_arch|
        try validateNativeHost(allocator, io, environ_map, req_arch)
    else
        try captureHostIdentity(allocator, io, environ_map, false);
    const report_prov = if (json_out_path != null)
        try captureReportProvenance(allocator, io, environ_map, repo, execution_id, null)
    else
        Value.null;

    const work_root = try std.fs.path.join(
        allocator,
        &.{ repo, ".bench-coremark", try std.fmt.allocPrint(allocator, "run-{s}", .{try generateUuidV4(io, allocator)}) },
    );
    try std.Io.Dir.cwd().createDirPath(io, work_root);
    defer {
        std.Io.Dir.cwd().deleteTree(io, work_root) catch {};
        _ = run(allocator, io, &.{ "git", "worktree", "prune" }, repo, null) catch {};
    }

    const affinity = if (std.mem.eql(u8, profile, "authoritative"))
        try selectCpuAffinity(allocator, io, environ_map)
    else
        null;

    var baseline_vals_opt: ?[]const f64 = null;
    var target_vals: []const f64 = &.{};
    var optimize_failed = false;
    var schedule_records: []SampleRecord = &.{};
    var report_results: ?[]EngineResult = null;
    var table: []u8 = "";

    if (std.mem.eql(u8, optimize_arg, "both")) {
        var fast_vals: ?[]const f64 = null;
        var safe_vals: ?[]const f64 = null;
        for (optimize_modes) |mode_name| {
            const mode_slug = try benchmark_optimize.slug(allocator, mode_name);
            const wt_info = try makeWorktree(
                allocator,
                io,
                repo,
                target,
                work_root,
                try std.fmt.allocPrint(allocator, "target-{s}", .{mode_slug}),
            );
            if (prepareWamr(allocator, io, environ_map, wt_info.path, target, wt_info.sha, resolved_fixture.path, mode_name, guest_args, exp_iters, null, defaultRunCommand)) |prep| {
                if (measureCommand(allocator, io, .{
                    .engine = prep.engine,
                    .cmd = prep.cmd,
                    .cwd = prep.cwd,
                    .env = prep.env,
                    .warmups = counts.warmups,
                    .runs = counts.runs,
                    .retry_wamr_flake = true,
                    .affinity = affinity,
                    .expected_iterations = exp_iters,
                })) |vals| {
                    if (std.mem.eql(u8, mode_name, "ReleaseFast")) fast_vals = vals else safe_vals = vals;
                } else |_| {
                    optimize_failed = true;
                }
            } else |_| {
                optimize_failed = true;
            }
        }
        table = try renderOptimizeTable(allocator, target, fast_vals, safe_vals, .{
            .profile = effective_profile,
            .warmups = counts.warmups,
            .runs = counts.runs,
            .fixture = resolved_fixture.path,
            .fixture_sha = resolved_fixture.sha256,
            .host = host,
        });
        target_vals = fast_vals orelse &.{};
    } else {
        const optimize_name = optimize_modes[0];
        const baseline_sha = try resolveRefSha(allocator, io, repo, baseline);
        const target_sha = try resolveRefSha(allocator, io, repo, target);

        if (wasmtime_baseline != null or wasmtime_arg != null) {
            var prepared: std.ArrayList(PreparedEngine) = .empty;
            var baseline_prepared: PreparedEngine = undefined;
            var target_prepared: PreparedEngine = undefined;
            if (std.mem.eql(u8, baseline_sha, target_sha)) {
                const target_wt = try makeWorktree(allocator, io, repo, target, work_root, "target");
                target_prepared = try prepareWamr(allocator, io, environ_map, target_wt.path, target, target_wt.sha, resolved_fixture.path, optimize_name, guest_args, exp_iters, null, defaultRunCommand);
                try prepared.append(allocator, target_prepared);
                baseline_prepared = target_prepared;
            } else {
                const baseline_wt = try makeWorktree(allocator, io, repo, baseline, work_root, "baseline");
                const target_wt = try makeWorktree(allocator, io, repo, target, work_root, "target");
                baseline_prepared = try prepareWamr(allocator, io, environ_map, baseline_wt.path, baseline, baseline_wt.sha, resolved_fixture.path, optimize_name, guest_args, exp_iters, null, defaultRunCommand);
                target_prepared = try prepareWamr(allocator, io, environ_map, target_wt.path, target, target_wt.sha, resolved_fixture.path, optimize_name, guest_args, exp_iters, null, defaultRunCommand);
                try prepared.appendSlice(allocator, &.{ baseline_prepared, target_prepared });
            }

            const WasmtimeSpec = struct { path: []const u8, label: []const u8, version: []const u8 };
            var wt_specs: std.ArrayList(WasmtimeSpec) = .empty;
            if (wasmtime_baseline) |wb| {
                const pinned_path = if (std.mem.eql(u8, wb, "auto"))
                    try installPinnedWasmtime(allocator, io, repo, wasmtime_cache)
                else
                    try std.Io.Dir.cwd().realPathFileAlloc(io, wb, allocator);
                const ver = try validatePinnedWasmtime(allocator, io, pinned_path);
                try wt_specs.append(allocator, .{ .path = pinned_path, .label = "Wasmtime historical pin", .version = ver });
            }
            if (wasmtime_arg) |wa| {
                const cur_path = try std.Io.Dir.cwd().realPathFileAlloc(io, wa, allocator);
                const ver = try wasmtimeVersionWith(allocator, io, cur_path, null, defaultRunCommand);
                try wt_specs.append(allocator, .{ .path = cur_path, .label = "Wasmtime caller-selected", .version = ver });
            }

            var prepared_wt_by_path: std.StringHashMap(PreparedEngine) = .init(allocator);
            for (wt_specs.items) |spec| {
                if (!prepared_wt_by_path.contains(spec.path)) {
                    const eng = try prepareWasmtime(allocator, io, environ_map, spec.path, spec.label, spec.version, resolved_fixture.path, guest_args, exp_iters);
                    try prepared_wt_by_path.put(spec.path, eng);
                    try prepared.append(allocator, eng);
                }
            }

            const measured_res = try measurePreparedEngines(allocator, io, prepared.items, .{
                .warmups = counts.warmups,
                .runs = counts.runs,
                .affinity = affinity,
            });
            schedule_records = measured_res.records;

            const target_measured = measured_res.results.get(target_prepared.key).?;
            var target_identity = target_prepared.identity;
            if (retain_target_artifacts) |rta| {
                target_identity = try retainWamrArtifactHandoff(allocator, io, target_prepared, rta, "wamr-target");
            }
            var baseline_identity = baseline_prepared.identity;
            if (retain_baseline_artifacts) |rba| {
                var base_prep = baseline_prepared;
                if (std.mem.eql(u8, baseline_sha, target_sha)) {
                    var id_copy = try cloneJsonValue(allocator, baseline_prepared.identity);
                    try id_copy.object.getPtr("source").?.object.put(allocator, "ref", .{ .string = baseline });
                    base_prep.identity = id_copy;
                }
                baseline_identity = try retainWamrArtifactHandoff(allocator, io, base_prep, rba, "wamr-baseline");
            }

            const target_result: EngineResult = .{
                .engine = "WAMR",
                .version = target_prepared.version,
                .optimize = optimize_name,
                .values = target_measured.values,
                .samples = target_measured.samples,
                .identity = target_identity,
            };
            const baseline_result: EngineResult = if (std.mem.eql(u8, baseline_sha, target_sha)) blk: {
                if (retain_baseline_artifacts == null) {
                    var id_copy = try cloneJsonValue(allocator, target_identity);
                    try id_copy.object.getPtr("source").?.object.put(allocator, "ref", .{ .string = baseline });
                    baseline_identity = id_copy;
                }
                break :blk .{
                    .engine = "WAMR",
                    .version = try std.fmt.allocPrint(allocator, "{s} ({s})", .{ baseline, baseline_sha }),
                    .optimize = optimize_name,
                    .values = try allocator.dupe(f64, target_result.values),
                    .samples = target_result.samples,
                    .identity = baseline_identity,
                };
            } else blk: {
                const base_measured = measured_res.results.get(baseline_prepared.key).?;
                break :blk .{
                    .engine = "WAMR",
                    .version = baseline_prepared.version,
                    .optimize = optimize_name,
                    .values = base_measured.values,
                    .samples = base_measured.samples,
                    .identity = baseline_identity,
                };
            };

            var res_list: std.ArrayList(EngineResult) = .empty;
            try res_list.appendSlice(allocator, &.{ baseline_result, target_result });
            for (wt_specs.items) |spec| {
                const prep_wt = prepared_wt_by_path.get(spec.path).?;
                const m_res = measured_res.results.get(prep_wt.key).?;
                try res_list.append(allocator, .{
                    .engine = spec.label,
                    .version = try std.fmt.allocPrint(allocator, "{s} (sha256:{s}; {s})", .{ spec.version, try sha256File(allocator, io, spec.path), spec.path }),
                    .optimize = "default JIT",
                    .values = m_res.values,
                    .samples = m_res.samples,
                    .identity = prep_wt.identity,
                });
            }
            report_results = try res_list.toOwnedSlice(allocator);
        } else {
            var res_list: std.ArrayList(EngineResult) = .empty;
            if (std.mem.eql(u8, baseline_sha, target_sha)) {
                const target_wt = try makeWorktree(allocator, io, repo, target, work_root, "target");
                const prep = try prepareWamr(allocator, io, environ_map, target_wt.path, target, target_wt.sha, resolved_fixture.path, optimize_name, guest_args, exp_iters, null, defaultRunCommand);
                const vals = try measureCommand(allocator, io, .{
                    .engine = prep.engine,
                    .cmd = prep.cmd,
                    .cwd = prep.cwd,
                    .env = prep.env,
                    .warmups = counts.warmups,
                    .runs = counts.runs,
                    .retry_wamr_flake = true,
                    .affinity = affinity,
                    .expected_iterations = exp_iters,
                });
                const target_result: EngineResult = .{
                    .engine = "WAMR",
                    .version = prep.version,
                    .optimize = optimize_name,
                    .values = vals,
                    .identity = prep.identity,
                };
                var base_id = try cloneJsonValue(allocator, prep.identity);
                try base_id.object.getPtr("source").?.object.put(allocator, "ref", .{ .string = baseline });
                const baseline_result: EngineResult = .{
                    .engine = "WAMR",
                    .version = try std.fmt.allocPrint(allocator, "{s} ({s})", .{ baseline, baseline_sha }),
                    .optimize = optimize_name,
                    .values = try allocator.dupe(f64, vals),
                    .identity = base_id,
                };
                try res_list.appendSlice(allocator, &.{ baseline_result, target_result });
            } else {
                const baseline_wt = try makeWorktree(allocator, io, repo, baseline, work_root, "baseline");
                const target_wt = try makeWorktree(allocator, io, repo, target, work_root, "target");
                const base_prep = try prepareWamr(allocator, io, environ_map, baseline_wt.path, baseline, baseline_wt.sha, resolved_fixture.path, optimize_name, guest_args, exp_iters, null, defaultRunCommand);
                const base_vals = try measureCommand(allocator, io, .{
                    .engine = base_prep.engine,
                    .cmd = base_prep.cmd,
                    .cwd = base_prep.cwd,
                    .env = base_prep.env,
                    .warmups = counts.warmups,
                    .runs = counts.runs,
                    .retry_wamr_flake = true,
                    .affinity = affinity,
                    .expected_iterations = exp_iters,
                });
                const targ_prep = try prepareWamr(allocator, io, environ_map, target_wt.path, target, target_wt.sha, resolved_fixture.path, optimize_name, guest_args, exp_iters, null, defaultRunCommand);
                const targ_vals = try measureCommand(allocator, io, .{
                    .engine = targ_prep.engine,
                    .cmd = targ_prep.cmd,
                    .cwd = targ_prep.cwd,
                    .env = targ_prep.env,
                    .warmups = counts.warmups,
                    .runs = counts.runs,
                    .retry_wamr_flake = true,
                    .affinity = affinity,
                    .expected_iterations = exp_iters,
                });
                try res_list.appendSlice(allocator, &.{
                    .{ .engine = "WAMR", .version = base_prep.version, .optimize = optimize_name, .values = base_vals, .identity = base_prep.identity },
                    .{ .engine = "WAMR", .version = targ_prep.version, .optimize = optimize_name, .values = targ_vals, .identity = targ_prep.identity },
                });
            }
            report_results = try res_list.toOwnedSlice(allocator);
        }

        baseline_vals_opt = report_results.?[0].values;
        target_vals = report_results.?[1].values;
        table = try renderTable(allocator, report_results.?, .{
            .profile = effective_profile,
            .warmups = counts.warmups,
            .runs = counts.runs,
            .fixture = resolved_fixture.path,
            .fixture_sha = resolved_fixture.sha256,
            .host = host,
            .schedule_records = schedule_records,
            .affinity = affinity,
            .guest_args = guest_args,
            .expected_iterations = exp_iters,
        });
    }

    try validateSameHost(allocator, io, environ_map, host);
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    try stdout_writer.interface.print("{s}\n", .{table});
    try stdout_writer.interface.flush();

    if (out_path) |op| {
        if (std.fs.path.dirname(op)) |parent| {
            if (parent.len != 0) try std.Io.Dir.cwd().createDirPath(io, parent);
        }
        try std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = op,
            .data = try std.fmt.allocPrint(allocator, "{s}\n", .{table}),
        });
    }
    if (json_out_path) |jp| {
        if (std.fs.path.dirname(jp)) |parent| {
            if (parent.len != 0) try std.Io.Dir.cwd().createDirPath(io, parent);
        }
        const report = try buildJsonReport(allocator, report_results.?, .{
            .profile = effective_profile,
            .warmups = counts.warmups,
            .runs = counts.runs,
            .fixture = resolved_fixture.path,
            .fixture_sha = resolved_fixture.sha256,
            .host = host,
            .schedule_records = schedule_records,
            .affinity = affinity,
            .provenance = report_prov,
            .guest_args = guest_args,
            .expected_iterations = exp_iters,
        });
        const encoded = try std.fmt.allocPrint(allocator, "{s}\n", .{try stringifyPrettySorted(allocator, report)});
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = jp, .data = encoded });
    }
    if (std.mem.eql(u8, emit, "github")) {
        if (environ_map.get("GITHUB_STEP_SUMMARY")) |summary_path| {
            if (summary_path.len != 0) {
                const prev = std.Io.Dir.cwd().readFileAlloc(io, summary_path, allocator, .limited(16 * 1024 * 1024)) catch "";
                const combined = try std.fmt.allocPrint(allocator, "{s}{s}\n", .{ prev, table });
                try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = summary_path, .data = combined });
            }
        }
    }

    if (optimize_failed) return 1;
    if (min_delta_pct) |min_d| {
        if (baseline_vals_opt) |b_vals| {
            const delta_pct = try computeDeltaPct(allocator, b_vals, target_vals);
            if (delta_pct < min_d) return 1;
        }
    }
    return 0;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const code = runCli(allocator, init.io, init.environ_map, if (args.len > 1) args[1..] else &.{}) catch |err| {
        std.debug.print("bench_coremark: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (code != 0) std.process.exit(code);
}

// ============================================================================
// Unit Tests (ported from BenchCoremarkTests in scripts/test_bench_coremark.py)
// ============================================================================

const test_valid_output =
    "2K performance run parameters for coremark.\n" ++
    "Iterations/Sec   : 12345.5\n" ++
    "Iterations       : 400000\n" ++
    "[0]crcfinal      : 0x33ff\n" ++
    "Correct operation validated. See README.md for run and reporting rules.\n";

fn testWamrIdentity(allocator: Allocator, ref: []const u8, source_sha: []const u8, digest: []const u8) !Value {
    const json_str = try std.fmt.allocPrint(
        allocator,
        "{{\"type\":\"wamr\",\"source\":{{\"ref\":\"{s}\",\"sha\":\"{s}\"}},\"optimize\":\"ReleaseFast\"," ++
            "\"runtime\":{{\"name\":\"wamr\",\"path\":\"/wamr\",\"sha256\":\"{s}\"}}," ++
            "\"compiler\":{{\"name\":\"wamrc\",\"path\":\"/wamrc\",\"sha256\":\"{s}\"}}," ++
            "\"module\":{{\"format\":\"cwasm\",\"path\":\"/coremark.cwasm\",\"sha256\":\"{s}\"}}}}",
        .{ ref, source_sha, digest, digest, digest },
    );
    return std.json.parseFromSliceLeaky(Value, allocator, json_str, .{});
}

fn testWasmtimeIdentity(allocator: Allocator, digest: []const u8) !Value {
    const json_str = try std.fmt.allocPrint(
        allocator,
        "{{\"type\":\"wasmtime\",\"channel\":\"historical-pin\",\"version\":\"{s}\"," ++
            "\"runtime\":{{\"name\":\"wasmtime\",\"path\":\"/wasmtime\",\"sha256\":\"{s}\"}}}}",
        .{ pinned_wasmtime_version, digest },
    );
    return std.json.parseFromSliceLeaky(Value, allocator, json_str, .{});
}

fn repeatByte(comptime ch: u8, comptime n: usize) *const [n]u8 {
    return &@as([n]u8, @splat(ch));
}

fn testReportProvenance(allocator: Allocator) !Value {
    const json_str =
        "{\"report_id\":\"12345678-1234-5678-1234-567812345678\"," ++
        "\"generated_at\":\"2026-09-08T00:00:00+00:00\"," ++
        "\"producer\":{\"source_sha\":\"" ++ repeatByte('c', 40) ++ "\"," ++
        "\"script\":{\"path\":\"scripts/bench_coremark.zig\",\"sha256\":\"" ++ repeatByte('d', 64) ++ "\"}}," ++
        "\"execution\":{\"provider\":\"local\",\"run_id\":\"test-run\"}}";
    return std.json.parseFromSliceLeaky(Value, allocator, json_str, .{});
}

const AuthoritativeReportFixture = struct {
    report: Value,
    host: HostIdentity,
    affinity: AffinityInfo,
    target_identity: Value,
};

fn makeAuthoritativeReportFixture(allocator: Allocator) !AuthoritativeReportFixture {
    const host: HostIdentity = .{
        .arch = "aarch64",
        .cpu_count = 4,
        .cpu_model = "Neoverse-N2",
        .runner_name = "runner",
        .boot_id = "boot-id",
    };
    const affinity: AffinityInfo = .{
        .allowed_cpus = &.{ 0, 1, 2, 3 },
        .selected_cpu = 0,
        .taskset = "/usr/bin/taskset",
    };
    const target_identity = try testWamrIdentity(allocator, "target", repeatByte('b', 40), repeatByte('2', 64));

    var records: std.ArrayList(SampleRecord) = .empty;
    var base_samples: std.ArrayList(SampleRecord) = .empty;
    var targ_samples: std.ArrayList(SampleRecord) = .empty;
    var wt_samples: std.ArrayList(SampleRecord) = .empty;
    var position: usize = 0;

    const phases = [_]struct { name: []const u8, count: isize }{
        .{ .name = "warmup", .count = 2 },
        .{ .name = "measured", .count = 10 },
    };
    for (phases) |ph| {
        var b_ord: usize = 0;
        var t_ord: usize = 0;
        var w_ord: usize = 0;
        const order = try counterbalancedOrder(allocator, &.{ "baseline", "target", "wasmtime" }, ph.count);
        for (order) |key| {
            position += 1;
            const ord = if (std.mem.eql(u8, key, "baseline")) blk: {
                b_ord += 1;
                break :blk b_ord;
            } else if (std.mem.eql(u8, key, "target")) blk: {
                t_ord += 1;
                break :blk t_ord;
            } else blk: {
                w_ord += 1;
                break :blk w_ord;
            };
            const rec: SampleRecord = .{
                .engine_key = key,
                .engine = if (std.mem.eql(u8, key, "wasmtime")) "Wasmtime" else try std.fmt.allocPrint(allocator, "WAMR {s}", .{key}),
                .phase = ph.name,
                .engine_ordinal = ord,
                .schedule_position = position,
                .started_at = "2026-09-08T00:00:00+00:00",
                .completed_at = "2026-09-08T00:00:01+00:00",
                .elapsed_seconds = 1.0,
                .iterations_per_second = if (std.mem.eql(u8, key, "baseline")) 100.0 else if (std.mem.eql(u8, key, "target")) 110.0 else 220.0,
                .iterations = expected_iterations,
            };
            try records.append(allocator, rec);
            if (std.mem.eql(u8, key, "baseline")) {
                try base_samples.append(allocator, rec);
            } else if (std.mem.eql(u8, key, "target")) {
                try targ_samples.append(allocator, rec);
            } else {
                try wt_samples.append(allocator, rec);
            }
        }
    }

    const results = [_]EngineResult{
        .{
            .engine = "WAMR",
            .version = "baseline",
            .optimize = "ReleaseFast",
            .values = &@as([10]f64, @splat(100.0)),
            .samples = base_samples.items,
            .identity = try testWamrIdentity(allocator, "baseline", repeatByte('a', 40), repeatByte('1', 64)),
        },
        .{
            .engine = "WAMR",
            .version = "target",
            .optimize = "ReleaseFast",
            .values = &@as([10]f64, @splat(110.0)),
            .samples = targ_samples.items,
            .identity = target_identity,
        },
        .{
            .engine = "Wasmtime historical pin",
            .version = "44.0.1",
            .optimize = "default JIT",
            .values = &@as([10]f64, @splat(220.0)),
            .samples = wt_samples.items,
            .identity = try testWasmtimeIdentity(allocator, repeatByte('3', 64)),
        },
    };
    const report = try buildJsonReport(allocator, &results, .{
        .profile = "authoritative",
        .warmups = 2,
        .runs = 10,
        .fixture = "coremark.wasm",
        .fixture_sha = repeatByte('f', 64),
        .host = host,
        .schedule_records = records.items,
        .affinity = affinity,
        .provenance = try testReportProvenance(allocator),
    });
    return .{
        .report = report,
        .host = host,
        .affinity = affinity,
        .target_identity = target_identity,
    };
}

test "tracked fixture checksum and output parsing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const resolved = try resolveFixture(allocator, io, ".", default_fixture_rel);
    const expected_path = try std.Io.Dir.cwd().realPathFileAlloc(io, default_fixture_rel, allocator);
    try std.testing.expectEqualStrings(expected_path, resolved.path);
    try std.testing.expectEqualStrings(default_fixture_sha256, resolved.sha256);

    const parsed = try parseCoremarkOutput(test_valid_output, "test", expected_iterations);
    try std.testing.expectEqual(@as(f64, 12345.5), parsed.throughput);
    try std.testing.expectEqual(@as(i64, 400000), parsed.iterations);

    try std.testing.expectError(
        error.MissingCrcValidation,
        parseCoremarkOutput("Iterations/Sec : 12345.5\nERROR! bad crc\n", "test", expected_iterations),
    );
    try std.testing.expectError(
        error.MissingCrcValidation,
        parseCoremarkOutput("Iterations/Sec : 12345.5\n", "test", expected_iterations),
    );
    try std.testing.expectError(
        error.AmbiguousThroughputOutput,
        parseCoremarkOutput(test_valid_output ++ "Iterations/Sec : 1\n", "test", expected_iterations),
    );

    const no_iters = try std.mem.replaceOwned(u8, allocator, test_valid_output, "Iterations       : 400000\n", "");
    try std.testing.expectError(error.InvalidIterationsFieldCount, parseCoremarkOutput(no_iters, "test", expected_iterations));
    try std.testing.expectError(
        error.InvalidIterationsFieldCount,
        parseCoremarkOutput(test_valid_output ++ "Iterations : 400000\n", "test", expected_iterations),
    );
    const auto_iters = try std.mem.replaceOwned(u8, allocator, test_valid_output, "Iterations       : 400000", "Iterations : auto");
    try std.testing.expectError(error.MalformedIterationsOutput, parseCoremarkOutput(auto_iters, "test", expected_iterations));
    const wrong_iters = try std.mem.replaceOwned(u8, allocator, test_valid_output, "400000", "300000");
    try std.testing.expectError(error.FixedIterationsMismatch, parseCoremarkOutput(wrong_iters, "test", expected_iterations));
    const no_perf = try std.mem.replaceOwned(u8, allocator, test_valid_output, "2K performance run parameters for coremark.\n", "");
    try std.testing.expectError(error.MissingPerformanceMarker, parseCoremarkOutput(no_perf, "test", expected_iterations));
    try std.testing.expectError(
        error.MissingCrcValidation,
        parseCoremarkOutput(test_valid_output ++ "Correct operation validated.\n", "test", expected_iterations),
    );
}

test "guest args affinity counterbalanced schedule and profile counts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const args = try coremarkGuestArgs(allocator, profileIterations("ci").?);
    try std.testing.expectEqual(@as(usize, 5), args.len);
    for (args, coremark_guest_args_default) |actual, exp| {
        try std.testing.expectEqualStrings(exp, actual);
    }

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    var base_env: std.process.Environ.Map = .init(allocator);
    const nopRunner = struct {
        fn run(_: ?*anyopaque, a: Allocator, _: std.Io, _: []const []const u8, _: ?[]const u8, _: ?*const std.process.Environ.Map) ![]u8 {
            return a.dupe(u8, "");
        }
    }.run;
    const wamr_prep = try prepareWamr(allocator, io, &base_env, root, "HEAD", repeatByte('a', 40), "/fixture.wasm", "ReleaseFast", &coremark_guest_args_default, expected_iterations, null, nopRunner);
    const wt_prep = try prepareWasmtime(allocator, io, &base_env, "/wasmtime", "Wasmtime", "44.0.1", "/fixture.wasm", &coremark_guest_args_default, expected_iterations);
    for (wamr_prep.cmd[wamr_prep.cmd.len - 5 ..], coremark_guest_args_default) |actual, exp| {
        try std.testing.expectEqualStrings(exp, actual);
    }
    for (wt_prep.cmd[wt_prep.cmd.len - 5 ..], coremark_guest_args_default) |actual, exp| {
        try std.testing.expectEqualStrings(exp, actual);
    }

    const affRunner = struct {
        fn run(_: ?*anyopaque, a: Allocator, _: std.Io, cmd: []const []const u8, _: ?[]const u8, _: ?*const std.process.Environ.Map) ![]u8 {
            try std.testing.expectEqualStrings("/usr/bin/taskset", cmd[0]);
            return a.dupe(u8, "4\n");
        }
    }.run;
    const aff = try selectCpuAffinityWith(allocator, io, null, &.{ 7, 4 }, "/usr/bin/taskset", null, affRunner);
    try std.testing.expectEqualSlices(usize, &.{ 4, 7 }, aff.allowed_cpus);
    try std.testing.expectEqual(@as(usize, 4), aff.selected_cpu);
    const applied = try applyAffinity(allocator, &.{"engine"}, aff);
    try std.testing.expectEqual(@as(usize, 4), applied.len);
    try std.testing.expectEqualStrings("/usr/bin/taskset", applied[0]);
    try std.testing.expectEqualStrings("--cpu-list", applied[1]);
    try std.testing.expectEqualStrings("4", applied[2]);
    try std.testing.expectEqualStrings("engine", applied[3]);

    const order2 = try counterbalancedOrder(allocator, &.{ "A", "B" }, 2);
    try std.testing.expectEqual(@as(usize, 4), order2.len);
    try std.testing.expectEqualStrings("A", order2[0]);
    try std.testing.expectEqualStrings("B", order2[1]);
    try std.testing.expectEqualStrings("B", order2[2]);
    try std.testing.expectEqualStrings("A", order2[3]);

    const c_auth = try resolveCounts("authoritative", null, null);
    try std.testing.expectEqual(@as(usize, 2), c_auth.warmups);
    try std.testing.expectEqual(@as(usize, 10), c_auth.runs);
    const c_ci = try resolveCounts("ci", null, null);
    try std.testing.expectEqual(@as(usize, 0), c_ci.warmups);
    try std.testing.expectEqual(@as(usize, 3), c_ci.runs);
    const c_ov = try resolveCounts("ci", 1, 4);
    try std.testing.expectEqual(@as(usize, 1), c_ov.warmups);
    try std.testing.expectEqual(@as(usize, 4), c_ov.runs);
    try std.testing.expectError(error.NegativeWarmups, resolveCounts("authoritative", -1, 10));
    try std.testing.expectError(error.NonPositiveRuns, resolveCounts("authoritative", 2, 0));
    try std.testing.expectEqualStrings("authoritative", try profileLabel(allocator, "authoritative", null, null));
    try std.testing.expectEqualStrings("authoritative (overridden)", try profileLabel(allocator, "authoritative", null, 3));
}

test "schedule groups samples and table rendering" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const MockState = struct {
        idx: usize = 0,
        vals: [4][]const u8 = .{ "10", "20", "22", "12" },
        fn run(ctx: ?*anyopaque, a: Allocator, _: std.Io, _: []const []const u8, _: ?[]const u8, _: ?*const std.process.Environ.Map) ![]u8 {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            const v = self.vals[self.idx];
            self.idx += 1;
            return std.mem.replaceOwned(u8, a, test_valid_output, "12345.5", v);
        }
    };
    var mock_state: MockState = .{};
    const engines = [_]PreparedEngine{
        .{ .key = "A", .engine = "WAMR", .version = "commit", .optimize = "ReleaseFast", .cmd = &.{"wamr"}, .cwd = "." },
        .{ .key = "B", .engine = "Wasmtime", .version = "44.0.1", .optimize = "default JIT", .cmd = &.{"wasmtime"}, .cwd = "." },
    };
    const measured = try measurePreparedEngines(allocator, io, &engines, .{
        .warmups = 0,
        .runs = 2,
        .runner_ctx = &mock_state,
        .runner = MockState.run,
    });
    const res_a = measured.results.get("A").?;
    const res_b = measured.results.get("B").?;
    try std.testing.expectEqualSlices(f64, &.{ 10.0, 12.0 }, res_a.values);
    try std.testing.expectEqualSlices(f64, &.{ 20.0, 22.0 }, res_b.values);
    try std.testing.expectEqualStrings("A", measured.records[0].engine_key);
    try std.testing.expectEqualStrings("B", measured.records[1].engine_key);
    try std.testing.expectEqualStrings("B", measured.records[2].engine_key);
    try std.testing.expectEqualStrings("A", measured.records[3].engine_key);

    const host: HostIdentity = .{
        .arch = "aarch64",
        .cpu_count = 4,
        .cpu_model = "Neoverse-N2",
        .runner_name = "runner",
        .boot_id = "boot-id",
    };
    const report = try buildJsonReport(allocator, &.{ res_a, res_a, res_b }, .{
        .profile = "authoritative",
        .warmups = 0,
        .runs = 2,
        .fixture = "fixture.wasm",
        .fixture_sha = "abc",
        .host = host,
        .schedule_records = measured.records,
        .affinity = null,
        .provenance = try testReportProvenance(allocator),
    });
    const ratio = report.object.get("ratios").?.array.items[0].object.get("median_ratio").?.float;
    try std.testing.expectApproxEqAbs(@as(f64, 11.0 / 21.0), ratio, 1e-12);

    const table_results = [_]EngineResult{
        .{ .engine = "WAMR", .version = "origin/main (aaaa)", .optimize = "ReleaseFast", .values = &.{ 50.0, 52.0 } },
        .{ .engine = "WAMR", .version = "HEAD (bbbb)", .optimize = "ReleaseFast", .values = &.{ 60.0, 62.0 } },
        .{ .engine = "Wasmtime historical pin", .version = "44.0.1 (sha256:abc; /pinned/wasmtime)", .optimize = "default JIT", .values = &.{ 100.0, 102.0 } },
        .{ .engine = "Wasmtime caller-selected", .version = "48.0.1 (sha256:def; /current/wasmtime)", .optimize = "default JIT", .values = &.{ 120.0, 122.0 } },
    };
    const table = try renderTable(allocator, &table_results, .{
        .profile = "authoritative",
        .warmups = 2,
        .runs = 2,
        .fixture = "coremark.wasm",
        .fixture_sha = "abc",
        .host = host,
    });
    try std.testing.expect(std.mem.indexOf(u8, table, "Median") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "50.0, 52.0") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "44.0.1 (sha256:abc; /pinned/wasmtime)") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "48.0.1 (sha256:def; /current/wasmtime)") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "WAMR target / Wasmtime historical pin") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "WAMR target / Wasmtime caller-selected") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "Median iter/s ratio") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "0.603960×") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "CRC validation") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "Neoverse-N2") != null);
    try std.testing.expect(std.mem.indexOf(u8, table, "host fingerprint") != null);

    // Full precision raw ratio check
    const raw_ratio: f64 = 0.48059510553690316;
    const rstats = try computeRatioStats(allocator, &.{11702.9216445}, &.{24350.89644});
    try std.testing.expectApproxEqAbs(raw_ratio, rstats.median_ratio, 1e-15);

    // Missing samples rejected
    try std.testing.expectError(
        error.UnexpectedMeasuredSampleCount,
        renderTable(allocator, &.{
            .{ .engine = "WAMR", .version = "origin/main (aaaa)", .optimize = "ReleaseFast", .values = &.{50.0} },
            .{ .engine = "WAMR", .version = "HEAD (bbbb)", .optimize = "ReleaseFast", .values = &.{ 60.0, 62.0 } },
        }, .{
            .profile = "authoritative",
            .warmups = 2,
            .runs = 2,
            .fixture = "coremark.wasm",
            .fixture_sha = "abc",
            .host = host,
        }),
    );
}

test "benchmark profile identity validation and fail-closed mismatches" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fix = try makeAuthoritativeReportFixture(allocator);
    const local_exec = try std.json.parseFromSliceLeaky(Value, allocator, "{\"provider\":\"local\",\"run_id\":\"test-run\"}", .{});
    const linkage = try validateBenchmarkProfileMatch(allocator, fix.report, .{
        .expected_arch = "aarch64",
        .fixture_sha = repeatByte('f', 64),
        .host = fix.host,
        .affinity = fix.affinity,
        .wamr_source_sha = repeatByte('b', 40),
        .wamr_optimize = "ReleaseFast",
        .wamr_runtime_sha = repeatByte('2', 64),
        .wamr_compiler_sha = repeatByte('2', 64),
        .wamr_module_sha = repeatByte('2', 64),
        .wasmtime_version_value = pinned_wasmtime_version,
        .wasmtime_runtime_sha = repeatByte('3', 64),
        .producer_source_sha = repeatByte('c', 40),
        .producer_script_sha = repeatByte('d', 64),
        .current_execution = local_exec,
    });
    try std.testing.expectEqualStrings(
        fix.report.object.get("provenance").?.object.get("report_id").?.string,
        linkage.object.get("report_id").?.string,
    );
    try std.testing.expectEqualStrings(
        repeatByte('b', 40),
        linkage.object.get("target").?.object.get("identity").?.object.get("source").?.object.get("sha").?.string,
    );

    const base_linkage = try validateBenchmarkProfileMatch(allocator, fix.report, .{
        .expected_arch = "aarch64",
        .fixture_sha = repeatByte('f', 64),
        .host = fix.host,
        .affinity = fix.affinity,
        .wamr_source_sha = repeatByte('a', 40),
        .wamr_optimize = "ReleaseFast",
        .wamr_runtime_sha = repeatByte('1', 64),
        .wamr_compiler_sha = repeatByte('1', 64),
        .wamr_module_sha = repeatByte('1', 64),
        .wasmtime_version_value = pinned_wasmtime_version,
        .wasmtime_runtime_sha = repeatByte('3', 64),
        .producer_source_sha = repeatByte('c', 40),
        .producer_script_sha = repeatByte('d', 64),
        .current_execution = local_exec,
        .benchmark_role = "wamr-baseline",
    });
    try std.testing.expectEqualStrings("wamr-baseline", base_linkage.object.get("selected_role").?.string);
    try std.testing.expect(!base_linkage.object.contains("target"));

    // Mismatches fail closed
    try std.testing.expectError(error.WamrSourceShaMismatch, validateBenchmarkProfileMatch(allocator, fix.report, .{
        .expected_arch = "aarch64",
        .fixture_sha = repeatByte('f', 64),
        .host = fix.host,
        .affinity = fix.affinity,
        .wamr_source_sha = repeatByte('9', 40),
        .wamr_optimize = "ReleaseFast",
        .wamr_runtime_sha = repeatByte('2', 64),
        .wamr_compiler_sha = repeatByte('2', 64),
        .wamr_module_sha = repeatByte('2', 64),
        .wasmtime_version_value = pinned_wasmtime_version,
        .wasmtime_runtime_sha = repeatByte('3', 64),
        .producer_source_sha = repeatByte('c', 40),
        .producer_script_sha = repeatByte('d', 64),
        .current_execution = local_exec,
    }));
    try std.testing.expectError(error.WamrRuntimeShaMismatch, validateBenchmarkProfileMatch(allocator, fix.report, .{
        .expected_arch = "aarch64",
        .fixture_sha = repeatByte('f', 64),
        .host = fix.host,
        .affinity = fix.affinity,
        .wamr_source_sha = repeatByte('b', 40),
        .wamr_optimize = "ReleaseFast",
        .wamr_runtime_sha = repeatByte('9', 64),
        .wamr_compiler_sha = repeatByte('2', 64),
        .wamr_module_sha = repeatByte('2', 64),
        .wasmtime_version_value = pinned_wasmtime_version,
        .wasmtime_runtime_sha = repeatByte('3', 64),
        .producer_source_sha = repeatByte('c', 40),
        .producer_script_sha = repeatByte('d', 64),
        .current_execution = local_exec,
    }));
    try std.testing.expectError(error.WamrCompilerShaMismatch, validateBenchmarkProfileMatch(allocator, fix.report, .{
        .expected_arch = "aarch64",
        .fixture_sha = repeatByte('f', 64),
        .host = fix.host,
        .affinity = fix.affinity,
        .wamr_source_sha = repeatByte('b', 40),
        .wamr_optimize = "ReleaseFast",
        .wamr_runtime_sha = repeatByte('2', 64),
        .wamr_compiler_sha = repeatByte('9', 64),
        .wamr_module_sha = repeatByte('2', 64),
        .wasmtime_version_value = pinned_wasmtime_version,
        .wasmtime_runtime_sha = repeatByte('3', 64),
        .producer_source_sha = repeatByte('c', 40),
        .producer_script_sha = repeatByte('d', 64),
        .current_execution = local_exec,
    }));
    try std.testing.expectError(error.WasmtimeRuntimeShaMismatch, validateBenchmarkProfileMatch(allocator, fix.report, .{
        .expected_arch = "aarch64",
        .fixture_sha = repeatByte('f', 64),
        .host = fix.host,
        .affinity = fix.affinity,
        .wamr_source_sha = repeatByte('b', 40),
        .wamr_optimize = "ReleaseFast",
        .wamr_runtime_sha = repeatByte('2', 64),
        .wamr_compiler_sha = repeatByte('2', 64),
        .wamr_module_sha = repeatByte('2', 64),
        .wasmtime_version_value = pinned_wasmtime_version,
        .wasmtime_runtime_sha = repeatByte('9', 64),
        .producer_source_sha = repeatByte('c', 40),
        .producer_script_sha = repeatByte('d', 64),
        .current_execution = local_exec,
    }));
    try std.testing.expectError(error.BenchmarkFixtureHashMismatch, validateBenchmarkProfileMatch(allocator, fix.report, .{
        .expected_arch = "aarch64",
        .fixture_sha = repeatByte('9', 64),
        .host = fix.host,
        .affinity = fix.affinity,
        .wamr_source_sha = repeatByte('b', 40),
        .wamr_optimize = "ReleaseFast",
        .wamr_runtime_sha = repeatByte('2', 64),
        .wamr_compiler_sha = repeatByte('2', 64),
        .wamr_module_sha = repeatByte('2', 64),
        .wasmtime_version_value = pinned_wasmtime_version,
        .wasmtime_runtime_sha = repeatByte('3', 64),
        .producer_source_sha = repeatByte('c', 40),
        .producer_script_sha = repeatByte('d', 64),
        .current_execution = local_exec,
    }));
    try std.testing.expectError(error.BenchmarkArchMismatch, validateBenchmarkProfileMatch(allocator, fix.report, .{
        .expected_arch = "x86_64",
        .fixture_sha = repeatByte('f', 64),
        .host = fix.host,
        .affinity = fix.affinity,
        .wamr_source_sha = repeatByte('b', 40),
        .wamr_optimize = "ReleaseFast",
        .wamr_runtime_sha = repeatByte('2', 64),
        .wamr_compiler_sha = repeatByte('2', 64),
        .wamr_module_sha = repeatByte('2', 64),
        .wasmtime_version_value = pinned_wasmtime_version,
        .wasmtime_runtime_sha = repeatByte('3', 64),
        .producer_source_sha = repeatByte('c', 40),
        .producer_script_sha = repeatByte('d', 64),
        .current_execution = local_exec,
    }));

    const legacy = try std.json.parseFromSliceLeaky(Value, allocator, "{\"schema_version\":1,\"kind\":\"coremark-authoritative-comparison\"}", .{});
    const st = try benchmarkReportStatus(allocator, legacy);
    try std.testing.expect(!st.authoritative);
    try std.testing.expectEqualStrings("legacy-unverified", st.status);
    try std.testing.expectError(error.LegacyUnverifiedSchemaVersion, validateAuthoritativeBenchmarkReport(allocator, legacy));
}

test "execution identity and artifact handoff retention" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    var empty_env: std.process.Environ.Map = .init(allocator);
    try std.testing.expectError(error.MissingLocalExecutionIdentity, captureExecutionIdentity(allocator, &empty_env, null));
    try std.testing.expectError(error.MissingLocalExecutionIdentity, captureExecutionIdentity(allocator, &empty_env, "   "));
    const exec_id = try captureExecutionIdentity(allocator, &empty_env, "shared-run");
    try std.testing.expectEqualStrings("local", exec_id.object.get("provider").?.string);
    try std.testing.expectEqualStrings("shared-run", exec_id.object.get("run_id").?.string);

    const prov = try captureReportProvenance(allocator, io, &empty_env, ".", null, repeatByte('b', 40));
    try std.testing.expectEqualStrings("local", prov.object.get("execution").?.object.get("provider").?.string);
    try std.testing.expect(prov.object.get("execution").?.object.get("run_id").?.string.len > 0);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    const source_dir = try std.fs.path.join(allocator, &.{ root, "temporary-benchmark-worktree/zig-out/bin" });
    const artifact_dir = try std.fs.path.join(allocator, &.{ root, "profile-artifacts" });
    const baseline_artifact_dir = try std.fs.path.join(allocator, &.{ root, "baseline-profile-artifacts" });
    try std.Io.Dir.cwd().createDirPath(io, source_dir);

    const wamr = try std.fs.path.join(allocator, &.{ source_dir, "wamr" });
    const wamrc = try std.fs.path.join(allocator, &.{ source_dir, "wamrc" });
    const simd_runner = try std.fs.path.join(allocator, &.{ source_dir, "simd-bench-runner" });
    const cwasm = try std.fs.path.join(allocator, &.{ root, "temporary-benchmark-worktree/.bench-coremark.cwasm" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = wamr, .data = "wamr-with-build-path-a" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = wamrc, .data = "wamrc-with-build-path-a" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = simd_runner, .data = "simd-runner-with-build-path-a" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cwasm, .data = "exact-cwasm" });

    const prepared: PreparedEngine = .{
        .key = "target",
        .engine = "WAMR target",
        .version = "target",
        .optimize = "ReleaseFast",
        .cmd = &.{ wamr, "run", cwasm },
        .cwd = source_dir,
        .identity = try makeWamrIdentity(allocator, io, .{
            .ref = "target",
            .source_sha = repeatByte('b', 40),
            .optimize = "ReleaseFast",
            .runtime_path = wamr,
            .compiler_path = wamrc,
            .module_path = cwasm,
            .simd_runner_path = simd_runner,
        }),
    };
    const retained = try retainWamrArtifactHandoff(allocator, io, prepared, artifact_dir, "wamr-target");
    try std.testing.expectError(error.FileExists, retainWamrArtifactHandoff(allocator, io, prepared, artifact_dir, "wamr-target"));

    try std.Io.Dir.cwd().deleteTree(io, try std.fs.path.join(allocator, &.{ root, "temporary-benchmark-worktree" }));
    const loaded = try loadWamrArtifactHandoff(allocator, io, artifact_dir, retained, "wamr-target");
    try std.testing.expectEqualStrings(
        "wamr-with-build-path-a",
        try std.Io.Dir.cwd().readFileAlloc(io, loaded.get("runtime").?, allocator, .limited(1024)),
    );
    try std.testing.expectEqualStrings(
        "simd-runner-with-build-path-a",
        try std.Io.Dir.cwd().readFileAlloc(io, loaded.get("simd_runner").?, allocator, .limited(1024)),
    );

    var base_id = try cloneJsonValue(allocator, retained);
    try base_id.object.getPtr("source").?.object.put(allocator, "ref", .{ .string = "baseline" });
    var base_prepared = prepared;
    base_prepared.identity = base_id;
    const base_retained = try retainWamrArtifactHandoff(allocator, io, base_prepared, baseline_artifact_dir, "wamr-baseline");
    const base_loaded = try loadWamrArtifactHandoff(allocator, io, baseline_artifact_dir, base_retained, "wamr-baseline");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = base_loaded.get("simd_runner").?, .data = "tampered" });
    try std.testing.expectError(
        error.HandoffArtifactHashMismatch,
        loadWamrArtifactHandoff(allocator, io, baseline_artifact_dir, base_retained, "wamr-baseline"),
    );
}

test "wasmtime version host validation and workflow contract" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const ver44 = struct {
        fn run(_: ?*anyopaque, a: Allocator, _: std.Io, _: []const []const u8, _: ?[]const u8, _: ?*const std.process.Environ.Map) ![]u8 {
            return a.dupe(u8, "wasmtime 44.0.1 (abcdef)");
        }
    }.run;
    const ver48 = struct {
        fn run(_: ?*anyopaque, a: Allocator, _: std.Io, _: []const []const u8, _: ?[]const u8, _: ?*const std.process.Environ.Map) ![]u8 {
            return a.dupe(u8, "wasmtime 48.0.1 (abcdef)");
        }
    }.run;
    try std.testing.expectEqualStrings("44.0.1", try validatePinnedWasmtimeWith(allocator, io, "/bin/wasmtime", null, ver44));
    try std.testing.expectError(error.PinnedWasmtimeVersionMismatch, validatePinnedWasmtimeWith(allocator, io, "/bin/wasmtime", null, ver48));

    const host: HostIdentity = .{
        .arch = "aarch64",
        .cpu_count = 4,
        .cpu_model = "Neoverse-N2",
        .runner_name = "runner",
        .boot_id = "boot-id",
    };
    const validated = try validateNativeHostWith("aarch64", host, "ARM64", "");
    try std.testing.expectEqualStrings("Neoverse-N2", validated.cpu_model);
    try std.testing.expectError(
        error.EmulationDetected,
        validateNativeHostWith("aarch64", host, "ARM64", "Hypervisor vendor: QEMU"),
    );
    try std.testing.expectError(
        error.HostIdentityChanged,
        validateSameHostWith(host, .{
            .arch = "aarch64",
            .cpu_count = 8,
            .cpu_model = "Neoverse-N2",
            .runner_name = "runner",
            .boot_id = "other-boot",
        }),
    );

    const workflow = try std.Io.Dir.cwd().readFileAlloc(io, ".github/workflows/coremark-aarch64.yml", allocator, .limited(4 * 1024 * 1024));
    try std.testing.expect(std.mem.indexOf(u8, workflow, "runs-on: ubuntu-24.04-arm") != null);
    try std.testing.expect(std.mem.indexOf(u8, workflow, "group: coremark-aarch64-${{ github.event_name }}-${{ github.ref }}") != null);
    try std.testing.expect(std.mem.indexOf(u8, workflow, "- profile") != null);
    try std.testing.expect(std.mem.indexOf(u8, workflow, "- paired-profile") != null);
    try std.testing.expect(std.mem.indexOf(u8, workflow, "19d046a5b23b9c39acf5f7062976f04c5ca8ca75") == null);
    try std.testing.expect(std.mem.indexOf(u8, workflow, "profile_sha=\"$(git rev-parse --verify --end-of-options \"${profile_ref}^{commit}\")\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, workflow, "--benchmark-role wamr-baseline") != null);
    try std.testing.expect(std.mem.indexOf(u8, workflow, "--benchmark-role wamr-target") != null);
}
