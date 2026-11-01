//! Collect reproducible WASI P3 phase timings and macOS sample profiles.
const std = @import("std");
const collect = @import("wasi_p3_profile_collect.zig");
const sample = @import("wasi_p3_profile_sample.zig");
const schema = @import("wasi_p3_profile_schema.zig");
const process_control = @import("process_control.zig");
const decimal_format = @import("decimal_format.zig");
const floatText = @import("float_text.zig").format;
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const min_sample_budget_s: f64 = 120.0;

pub const RunArgs = struct {
    mode: collect.Mode,
    platform_id: []const u8,
    output_dir: []const u8,
    cold_samples: usize = 5,
    warm_samples: usize = 10,
    sample_timeout_s: f64 = 1800.0,
    deadline_minutes: ?f64 = null,
};

pub const ProfileArgs = struct {
    mode: collect.Mode,
    selection: []const u8,
    output_dir: []const u8,
};

pub const Command = union(enum) {
    run: RunArgs,
    profile: ProfileArgs,
};

pub const Tools = struct {
    git: []const u8 = "git",
    zig: []const u8 = "zig",
    timeout_ns: u64 = 30 * std.time.ns_per_s,
    grace_ns: u64 = 10 * std.time.ns_per_s,
};

pub const CollectionOptions = struct {
    repo: []const u8,
    suite: []const u8,
    runner: []const []const u8,
    args: RunArgs,
    tools: Tools = .{},
    commit: ?[]const u8 = null,
    min_budget_s: f64 = min_sample_budget_s,
    interrupted: ?*const std.atomic.Value(bool) = null,
};

pub const ProfileOptions = struct {
    repo: []const u8,
    suite: []const u8,
    runner: []const []const u8,
    args: ProfileArgs,
    grace_ns: u64 = 10 * std.time.ns_per_s,
    interrupted: ?*const std.atomic.Value(bool) = null,
};

const JsonFloat = struct {
    allocator: Allocator,
    value: f64,

    pub fn jsonStringify(self: JsonFloat, writer: *std.json.Stringify) std.json.Stringify.Error!void {
        const text = floatText(self.allocator, self.value) catch return error.WriteFailed;
        defer self.allocator.free(text);
        try writer.beginWriteRaw();
        try writer.writer.writeAll(text);
        writer.endWriteRaw();
    }
};

pub const HostMetadata = struct {
    system: []const u8,
    release: []const u8,
    version: []const u8,
    machine: []const u8,
    processor: []const u8,
    runner_name: []const u8,
    runner_image: []const u8,
    runner_os: []const u8,
    runner_arch: []const u8,
};

pub const ToolVersions = struct {
    zig: []const u8,
    wamr: []const u8,
    wamrc: []const u8,
};

pub const CacheMetadata = struct {
    zig_global_cache_dir: []const u8,
    zig_local_cache_dir: []const u8,
    aot_sidecars: []const u8 = "mtime sibling cache",
    jit: []const u8 = "in-process; no persistent compiled-artifact cache",
};

pub const Metadata = struct {
    platform_id: []const u8,
    mode: []const u8,
    optimize: []const u8 = "ReleaseSafe",
    commit: []const u8,
    collected_at: []const u8,
    host: HostMetadata,
    tools: ToolVersions,
    cache: CacheMetadata,
};

fn parseMode(raw: []const u8) !collect.Mode {
    if (std.mem.eql(u8, raw, "aot")) return .aot;
    if (std.mem.eql(u8, raw, "jit")) return .jit;
    return error.InvalidProfileMode;
}

fn parsePositiveInt(raw: []const u8) !usize {
    if (raw.len == 0) return error.InvalidPositiveInteger;
    const val = std.fmt.parseInt(usize, raw, 10) catch return error.InvalidPositiveInteger;
    if (val == 0) return error.InvalidPositiveInteger;
    return val;
}

fn parsePositiveFloat(raw: []const u8) !f64 {
    if (raw.len == 0) return error.InvalidPositiveFloat;
    const val = std.fmt.parseFloat(f64, raw) catch return error.InvalidPositiveFloat;
    if (!std.math.isFinite(val) or val <= 0) return error.InvalidPositiveFloat;
    return val;
}

pub fn parseArgs(args: []const []const u8) !Command {
    if (args.len == 0) return error.MissingSubcommand;
    if (std.mem.eql(u8, args[0], "-h") or std.mem.eql(u8, args[0], "--help")) return error.HelpRequested;
    if (std.mem.eql(u8, args[0], "run")) {
        var mode: ?collect.Mode = null;
        var platform_id: ?[]const u8 = null;
        var output_dir: ?[]const u8 = null;
        var cold_samples: usize = 5;
        var warm_samples: usize = 10;
        var sample_timeout_s: f64 = 1800.0;
        var deadline_minutes: ?f64 = null;
        var index: usize = 1;
        while (index < args.len) : (index += 1) {
            const arg = args[index];
            if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) return error.HelpRequested;
            const split = std.mem.indexOfScalar(u8, arg, '=');
            const name = if (split) |pos| arg[0..pos] else arg;
            const value = if (split) |pos| arg[pos + 1 ..] else blk: {
                index += 1;
                if (index >= args.len) return error.MissingOptionValue;
                break :blk args[index];
            };
            if (std.mem.eql(u8, name, "--mode")) {
                mode = try parseMode(value);
            } else if (std.mem.eql(u8, name, "--platform-id")) {
                if (value.len == 0) return error.MissingPlatformId;
                platform_id = value;
            } else if (std.mem.eql(u8, name, "--output-dir")) {
                if (value.len == 0) return error.MissingOutputDirectory;
                output_dir = value;
            } else if (std.mem.eql(u8, name, "--cold-samples")) {
                cold_samples = try parsePositiveInt(value);
            } else if (std.mem.eql(u8, name, "--warm-samples")) {
                warm_samples = try parsePositiveInt(value);
            } else if (std.mem.eql(u8, name, "--sample-timeout")) {
                sample_timeout_s = try parsePositiveFloat(value);
            } else if (std.mem.eql(u8, name, "--deadline-minutes")) {
                deadline_minutes = try parsePositiveFloat(value);
            } else return error.UnknownOption;
        }
        return .{ .run = .{
            .mode = mode orelse return error.MissingProfileMode,
            .platform_id = platform_id orelse return error.MissingPlatformId,
            .output_dir = output_dir orelse return error.MissingOutputDirectory,
            .cold_samples = cold_samples,
            .warm_samples = warm_samples,
            .sample_timeout_s = sample_timeout_s,
            .deadline_minutes = deadline_minutes,
        } };
    }
    if (std.mem.eql(u8, args[0], "profile")) {
        var mode: ?collect.Mode = null;
        var selection: ?[]const u8 = null;
        var output_dir: ?[]const u8 = null;
        var index: usize = 1;
        while (index < args.len) : (index += 1) {
            const arg = args[index];
            if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) return error.HelpRequested;
            const split = std.mem.indexOfScalar(u8, arg, '=');
            const name = if (split) |pos| arg[0..pos] else arg;
            const value = if (split) |pos| arg[pos + 1 ..] else blk: {
                index += 1;
                if (index >= args.len) return error.MissingOptionValue;
                break :blk args[index];
            };
            if (std.mem.eql(u8, name, "--mode")) {
                mode = try parseMode(value);
            } else if (std.mem.eql(u8, name, "--selection")) {
                if (value.len == 0) return error.MissingSelectionPath;
                selection = value;
            } else if (std.mem.eql(u8, name, "--output-dir")) {
                if (value.len == 0) return error.MissingOutputDirectory;
                output_dir = value;
            } else return error.UnknownOption;
        }
        return .{ .profile = .{
            .mode = mode orelse return error.MissingProfileMode,
            .selection = selection orelse return error.MissingSelectionPath,
            .output_dir = output_dir orelse return error.MissingOutputDirectory,
        } };
    }
    return error.UnknownSubcommand;
}

pub fn splitCommand(allocator: Allocator, value: []const u8) ![]const []const u8 {
    var tokens: std.ArrayList([]const u8) = .empty;
    var token: std.ArrayList(u8) = .empty;
    var quote: enum { none, single, double } = .none;
    var started = false;
    var index: usize = 0;
    while (index < value.len) : (index += 1) {
        const byte = value[index];
        switch (quote) {
            .single => if (byte == '\'') {
                quote = .none;
            } else {
                try token.append(allocator, byte);
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
                ' ', '\t', '\r', '\n' => if (started) {
                    try tokens.append(allocator, try token.toOwnedSlice(allocator));
                    started = false;
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
    if (started) try tokens.append(allocator, try token.toOwnedSlice(allocator));
    if (tokens.items.len == 0) return error.MissingExecutable;
    return tokens.toOwnedSlice(allocator);
}

fn timestamp(allocator: Allocator, nanoseconds: i96) ![]const u8 {
    if (nanoseconds < 0 or nanoseconds >= 253_402_300_800 * std.time.ns_per_s) return error.InvalidTimestamp;
    const seconds: u64 = @intCast(@divTrunc(nanoseconds, std.time.ns_per_s));
    const micros: u32 = @intCast(@divTrunc(@rem(nanoseconds, std.time.ns_per_s), std.time.ns_per_us));
    const epoch: std.time.epoch.EpochSeconds = .{ .secs = seconds };
    const date = epoch.getEpochDay().calculateYearDay();
    const month_day = date.calculateMonthDay();
    const day = epoch.getDaySeconds();
    const prefix = try std.fmt.allocPrint(allocator, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}", .{
        date.year,             month_day.month.numeric(), @as(u8, month_day.day_index) + 1,
        day.getHoursIntoDay(), day.getMinutesIntoHour(),  day.getSecondsIntoMinute(),
    });
    return if (micros == 0)
        std.fmt.allocPrint(allocator, "{s}+00:00", .{prefix})
    else
        std.fmt.allocPrint(allocator, "{s}.{d:0>6}+00:00", .{ prefix, micros });
}

fn captureCommand(
    allocator: Allocator,
    io: std.Io,
    cwd: []const u8,
    env: *const std.process.Environ.Map,
    argv: []const []const u8,
    timeout_ns: u64,
    grace_ns: u64,
    interrupted: ?*const std.atomic.Value(bool),
) !struct { term: std.process.Child.Term, timed_out: bool, stdout: []u8, stderr: []u8 } {
    if (interrupted) |flag| if (flag.load(.acquire)) return error.Interrupted;
    const started = std.Io.Clock.awake.now(io);
    var child = try std.process.spawn(io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = env,
        .stdout = .pipe,
        .stderr = .pipe,
        .pgid = 0,
    });
    const pid = child.id.?;
    const stdout = child.stdout.?;
    const stderr = child.stderr.?;
    child.stdout = null;
    child.stderr = null;
    defer stdout.close(io);
    defer stderr.close(io);
    defer if (child.id != null) {
        sample.signalGroup(pid, .KILL) catch {};
        child.kill(io);
    };
    var storage: std.Io.File.MultiReader.Buffer(2) = undefined;
    var reader: std.Io.File.MultiReader = undefined;
    reader.init(allocator, io, storage.toStreams(), &.{ stdout, stderr });
    defer reader.deinit();
    var eof = false;
    var timed_out = false;
    var stopping = false;
    var killed = false;
    var stop_started: std.Io.Timestamp = undefined;
    while (!eof) {
        const now = std.Io.Clock.awake.now(io);
        const elapsed: u64 = @intCast(@max(0, started.durationTo(now).nanoseconds));
        const was_interrupted = if (interrupted) |flag| flag.load(.acquire) else false;
        if (!stopping and (was_interrupted or elapsed >= timeout_ns)) {
            timed_out = !was_interrupted;
            try sample.signalGroup(pid, .TERM);
            stopping = true;
            stop_started = now;
        }
        if (stopping and !killed and stop_started.durationTo(now).nanoseconds >= @as(i128, grace_ns)) {
            try sample.signalGroup(pid, .KILL);
            killed = true;
        }
        if (stopping and killed and stop_started.durationTo(now).nanoseconds >= @as(i128, grace_ns) + 10 * std.time.ns_per_s)
            return error.ProfileProcessDidNotTerminate;
        const deadline: std.Io.Timeout = .{ .deadline = .fromNow(io, .{ .clock = .awake, .raw = .fromMilliseconds(20) }) };
        reader.fill(4096, deadline) catch |err| switch (err) {
            error.Timeout => {},
            error.EndOfStream => eof = true,
            else => return err,
        };
        try reader.checkAnyError();
    }
    if (stopping and !killed) try sample.signalGroup(pid, .KILL);
    const term = (try sample.waitUntil(io, &child, 10 * std.time.ns_per_s)) orelse return error.ProfileProcessDidNotTerminate;
    if (interrupted) |flag| if (flag.load(.acquire)) return error.Interrupted;
    const out_bytes = try reader.toOwnedSlice(0);
    errdefer allocator.free(out_bytes);
    const err_bytes = try reader.toOwnedSlice(1);
    return .{ .term = term, .timed_out = timed_out, .stdout = out_bytes, .stderr = err_bytes };
}

pub fn commandVersion(
    allocator: Allocator,
    io: std.Io,
    cwd: []const u8,
    env: *const std.process.Environ.Map,
    argv: []const []const u8,
    timeout_ns: u64,
    grace_ns: u64,
    interrupted: ?*const std.atomic.Value(bool),
) ![]const u8 {
    const result = captureCommand(allocator, io, cwd, env, argv, timeout_ns, grace_ns, interrupted) catch |err| switch (err) {
        error.Interrupted, error.OutOfMemory => return err,
        else => return std.fmt.allocPrint(allocator, "unavailable: {s}", .{@errorName(err)}),
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (result.timed_out) return allocator.dupe(u8, "unavailable: TimeoutExpired");
    const raw = if (result.stdout.len != 0) result.stdout else result.stderr;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len != 0) {
        const first_line = std.mem.trimEnd(u8, std.mem.sliceTo(trimmed, '\n'), "\r");
        return allocator.dupe(u8, first_line);
    }
    return std.fmt.allocPrint(allocator, "exit {d}", .{try sample.exitCode(result.term)});
}

fn gitCommit(
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    env: *const std.process.Environ.Map,
    tools: Tools,
    interrupted: ?*const std.atomic.Value(bool),
) ![]const u8 {
    const res = try captureCommand(allocator, io, repo, env, &.{ tools.git, "rev-parse", "HEAD" }, tools.timeout_ns, tools.grace_ns, interrupted);
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    if (res.timed_out or !res.term.success()) return error.GitRevParseFailed;
    const sha = std.mem.trim(u8, res.stdout, " \t\r\n");
    if (sha.len == 0) return error.GitRevParseFailed;
    return allocator.dupe(u8, sha);
}

pub fn collectMetadata(
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    env: *const std.process.Environ.Map,
    mode: collect.Mode,
    platform_id: []const u8,
    tools: Tools,
    commit_override: ?[]const u8,
    interrupted: ?*const std.atomic.Value(bool),
) !Metadata {
    const uts = std.posix.uname();
    const system = try allocator.dupe(u8, std.mem.sliceTo(&uts.sysname, 0));
    const release = try allocator.dupe(u8, std.mem.sliceTo(&uts.release, 0));
    const version = try allocator.dupe(u8, std.mem.sliceTo(&uts.version, 0));
    const machine = try allocator.dupe(u8, std.mem.sliceTo(&uts.machine, 0));
    const default_wamr = try std.fs.path.join(allocator, &.{ repo, "zig-out/bin/wamr" });
    const default_wamrc = try std.fs.path.join(allocator, &.{ repo, "zig-out/bin/wamrc" });
    const wamr_raw = env.get("WAMR") orelse default_wamr;
    const wamrc_raw = env.get("WAMRC") orelse default_wamrc;
    const wamr_cmd = try splitCommand(allocator, if (wamr_raw.len != 0) wamr_raw else default_wamr);
    const wamrc_cmd = try splitCommand(allocator, if (wamrc_raw.len != 0) wamrc_raw else default_wamrc);
    const wamr_ver_argv = try std.mem.concat(allocator, []const u8, &.{ wamr_cmd, &.{"version"} });
    const wamrc_ver_argv = try std.mem.concat(allocator, []const u8, &.{ wamrc_cmd, &.{"version"} });
    const commit = if (commit_override) |sha|
        try allocator.dupe(u8, sha)
    else
        try gitCommit(allocator, io, repo, env, tools, interrupted);
    return .{
        .platform_id = platform_id,
        .mode = @tagName(mode),
        .commit = commit,
        .collected_at = try timestamp(allocator, std.Io.Clock.real.now(io).nanoseconds),
        .host = .{
            .system = system,
            .release = release,
            .version = version,
            .machine = machine,
            .processor = machine,
            .runner_name = env.get("RUNNER_NAME") orelse "",
            .runner_image = env.get("ImageOS") orelse "",
            .runner_os = env.get("RUNNER_OS") orelse "",
            .runner_arch = env.get("RUNNER_ARCH") orelse "",
        },
        .tools = .{
            .zig = try commandVersion(allocator, io, repo, env, &.{ tools.zig, "version" }, tools.timeout_ns, tools.grace_ns, interrupted),
            .wamr = try commandVersion(allocator, io, repo, env, wamr_ver_argv, tools.timeout_ns, tools.grace_ns, interrupted),
            .wamrc = try commandVersion(allocator, io, repo, env, wamrc_ver_argv, tools.timeout_ns, tools.grace_ns, interrupted),
        },
        .cache = .{
            .zig_global_cache_dir = env.get("ZIG_GLOBAL_CACHE_DIR") orelse "",
            .zig_local_cache_dir = env.get("ZIG_LOCAL_CACHE_DIR") orelse "",
        },
    };
}

fn persistDocument(
    allocator: Allocator,
    io: std.Io,
    destination: []const u8,
    metadata: Metadata,
    args: RunArgs,
    stopped_early: bool,
    stop_reason: []const u8,
    samples: []const sample.Sample,
) !void {
    const plan = .{
        .cold_samples = args.cold_samples,
        .warm_samples = args.warm_samples,
        .optimize = "ReleaseSafe",
        .sample_timeout_s = JsonFloat{ .allocator = allocator, .value = args.sample_timeout_s },
        .deadline_minutes = if (args.deadline_minutes) |dm| JsonFloat{ .allocator = allocator, .value = dm } else null,
    };
    const doc = .{
        .schema_version = @as(u32, 1),
        .kind = "wasi-p3-profile-runs",
        .metadata = metadata,
        .plan = plan,
        .planned_samples = args.cold_samples + args.warm_samples,
        .stopped_early = stopped_early,
        .stop_reason = stop_reason,
        .samples = samples,
    };
    const json_body = try std.json.Stringify.valueAlloc(allocator, doc, .{ .whitespace = .indent_2 });
    const text = try std.fmt.allocPrint(allocator, "{s}\n", .{json_body});
    const parsed = try std.json.parseFromSliceLeaky(Value, allocator, text, .{});
    try schema.validateRun(parsed);
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{destination});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp_path, .data = text });
    try std.Io.Dir.rename(std.Io.Dir.cwd(), tmp_path, std.Io.Dir.cwd(), destination, io);
}

pub fn runCollection(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    opts: CollectionOptions,
    out: *std.Io.Writer,
) !u8 {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.UnsupportedProfilePlatform;
    try std.Io.Dir.cwd().createDirPath(io, opts.args.output_dir);
    const output_dir = try std.Io.Dir.cwd().realPathFileAlloc(io, opts.args.output_dir, allocator);
    const destination = try std.fs.path.join(allocator, &.{ output_dir, "samples.json" });
    const deadline_s = if (opts.args.deadline_minutes) |dm| dm * 60.0 else null;
    const started = std.Io.Clock.awake.now(io);
    const metadata = try collectMetadata(
        allocator,
        io,
        opts.repo,
        base_env,
        opts.args.mode,
        opts.args.platform_id,
        opts.tools,
        opts.commit,
        opts.interrupted,
    );
    var samples: std.ArrayList(sample.Sample) = .empty;
    try persistDocument(allocator, io, destination, metadata, opts.args, false, "", samples.items);
    const phases = [_]struct { temp: collect.Temperature, count: usize }{
        .{ .temp = .cold, .count = opts.args.cold_samples },
        .{ .temp = .warm, .count = opts.args.warm_samples },
    };
    for (phases) |phase| {
        for (1..phase.count + 1) |index| {
            var timeout_s = opts.args.sample_timeout_s;
            if (deadline_s) |limit_s| {
                const elapsed_ns = @max(0, started.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
                const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);
                const remaining_s = limit_s - elapsed_s;
                if (remaining_s <= opts.min_budget_s) {
                    const dm_text = try floatText(allocator, opts.args.deadline_minutes.?);
                    const reason = try std.fmt.allocPrint(
                        allocator,
                        "collection deadline of {s}min reached after {d} of {d} samples",
                        .{ dm_text, samples.items.len, opts.args.cold_samples + opts.args.warm_samples },
                    );
                    try out.print("stopping early: {s}\n", .{reason});
                    try persistDocument(allocator, io, destination, metadata, opts.args, true, reason, samples.items);
                    return 1;
                }
                timeout_s = @min(timeout_s, remaining_s);
            }
            const timeout_ns: u64 = @intFromFloat(@max(1.0, @round(timeout_s * @as(f64, std.time.ns_per_s))));
            const timeout_text = try decimal_format.general(allocator, timeout_s);
            const item = try sample.run(allocator, io, base_env, .{
                .repo = opts.repo,
                .suite = opts.suite,
                .output_dir = output_dir,
                .runner = opts.runner,
                .mode = opts.args.mode,
                .temperature = phase.temp,
                .index = index,
                .timeout_ns = timeout_ns,
                .timeout_seconds_text = timeout_text,
                .grace_ns = opts.tools.grace_ns,
                .interrupted = opts.interrupted,
            });
            const elapsed_text = try decimal_format.fixed(
                allocator,
                @as(f64, @floatFromInt(item.suite_duration_ns)) / @as(f64, std.time.ns_per_s),
                3,
            );
            try out.print(
                "{s}: valid={s} passed={d}/{d} elapsed={s}s{s}\n",
                .{
                    item.id,
                    if (item.valid) "True" else "False",
                    item.counts.passed,
                    collect.expected_fixtures,
                    elapsed_text,
                    if (item.timed_out) " TIMEOUT" else "",
                },
            );
            try samples.append(allocator, item);
            try persistDocument(allocator, io, destination, metadata, opts.args, false, "", samples.items);
        }
    }
    for (samples.items) |item| if (!item.valid) return 1;
    return 0;
}

const ProfileTarget = struct {
    fixture: []const u8,
    phase: []const u8,
};

fn parseSelectionTargets(allocator: Allocator, root: Value, mode: collect.Mode) ![]const ProfileTarget {
    if (root != .object) return error.InvalidProfileSelection;
    const profiles = root.object.get("profiles") orelse return &.{};
    if (profiles != .array) return error.InvalidProfileSelection;
    var targets: std.ArrayList(ProfileTarget) = .empty;
    for (profiles.array.items) |entry| {
        if (entry != .object) return error.InvalidProfileSelection;
        const entry_mode = entry.object.get("mode") orelse continue;
        if (entry_mode != .string or !std.mem.eql(u8, entry_mode.string, @tagName(mode))) continue;
        const fixture = entry.object.get("fixture") orelse return error.InvalidProfileSelection;
        const phase = entry.object.get("phase") orelse return error.InvalidProfileSelection;
        if (fixture != .string or fixture.string.len == 0 or phase != .string) return error.InvalidProfileSelection;
        if (!std.mem.eql(u8, phase.string, "component_precompile") and
            !std.mem.eql(u8, phase.string, "core_precompile") and
            !std.mem.eql(u8, phase.string, "fixture_execution"))
            return error.InvalidProfileSelection;
        try targets.append(allocator, .{ .fixture = fixture.string, .phase = phase.string });
    }
    return targets.toOwnedSlice(allocator);
}

fn formatStringList(allocator: Allocator, items: []const []const u8) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeByte('[');
    for (items, 0..) |item, index| {
        if (index != 0) try w.writeAll(", ");
        try w.print("'{s}'", .{item});
    }
    try w.writeByte(']');
    return out.toOwnedSlice();
}

pub fn runProfile(
    allocator: Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    opts: ProfileOptions,
    out: *std.Io.Writer,
    err_out: *std.Io.Writer,
) !u8 {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.UnsupportedProfilePlatform;
    if (opts.runner.len == 0) return error.InvalidProfileRunner;
    const selection_bytes = try std.Io.Dir.cwd().readFileAlloc(io, opts.args.selection, allocator, .unlimited);
    const selection_json = try std.json.parseFromSliceLeaky(Value, allocator, selection_bytes, .{});
    const requested = try parseSelectionTargets(allocator, selection_json, opts.args.mode);
    if (requested.len == 0) {
        try out.print("No {s} profiles selected.\n", .{@tagName(opts.args.mode)});
        return 0;
    }
    const selection_abs = try std.Io.Dir.cwd().realPathFileAlloc(io, opts.args.selection, allocator);
    try std.Io.Dir.cwd().createDirPath(io, opts.args.output_dir);
    const output_dir = try std.Io.Dir.cwd().realPathFileAlloc(io, opts.args.output_dir, allocator);
    try sample.removeSidecars(io, opts.suite, opts.args.mode == .jit);

    const mode_name = @tagName(opts.args.mode);
    const run_id = try std.fmt.allocPrint(allocator, "profile-{s}", .{mode_name});
    const timing = try std.fs.path.join(allocator, &.{ output_dir, try std.fmt.allocPrint(allocator, "{s}.jsonl", .{run_id}) });
    const report = try std.fs.path.join(allocator, &.{ output_dir, try std.fmt.allocPrint(allocator, "{s}-report.json", .{run_id}) });
    const log_path = try std.fs.path.join(allocator, &.{ output_dir, try std.fmt.allocPrint(allocator, "{s}.log", .{run_id}) });
    try sample.removeIfPresent(io, timing);
    try sample.removeIfPresent(io, report);

    var env = try base_env.clone(allocator);
    defer env.deinit();
    try env.put("WAMR_PROFILE_TIMINGS", timing);
    try env.put("WAMR_PROFILE_RUN_ID", run_id);
    try env.put("WAMR_PROFILE_MODE", mode_name);
    try env.put("WAMR_PROFILE_SELECTION", selection_abs);
    try env.put("WAMR_PROFILE_OUTPUT_DIR", output_dir);
    try env.put("WAMR_P3_REPORT", report);
    try env.put("WAMR_TESTSUITE_TIMEOUT", "300");
    const launcher = env.get("WAMR_PROFILE_LAUNCHER") orelse "";
    if (launcher.len == 0) {
        try env.put("WAMR_PROFILE_LAUNCHER", try std.fs.path.join(allocator, &.{ opts.repo, "zig-out/bin/wasi-p3-sample-launch" }));
    }
    if (opts.args.mode == .jit) try env.put("WAMR_JIT_TESTSUITE", "1") else _ = env.swapRemove("WAMR_JIT_TESTSUITE");

    const argv = try allocator.alloc([]const u8, opts.runner.len + 2);
    @memcpy(argv[0..opts.runner.len], opts.runner);
    argv[opts.runner.len] = "--json-output-location";
    argv[opts.runner.len + 1] = report;

    const log_file = try std.Io.Dir.cwd().createFile(io, log_path, .{});
    defer log_file.close(io);
    const outcome = try sample.runCommand(allocator, io, argv, opts.repo, &env, log_file, null, opts.grace_ns, opts.interrupted);
    var returncode = try sample.exitCode(outcome.term);
    if (returncode == 0) {
        const report_bytes = std.Io.Dir.cwd().readFileAlloc(io, report, allocator, .unlimited) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        if (report_bytes) |bytes| {
            const counts = collect.reportCounts(try std.json.parseFromSliceLeaky(Value, allocator, bytes, .{ .duplicate_field_behavior = .use_last })) catch collect.Counts{
                .fixtures = 0,
                .executed = 0,
                .passed = 0,
                .failed = 0,
            };
            if (counts.fixtures != collect.expected_fixtures or
                counts.executed != collect.expected_fixtures or
                counts.passed != collect.expected_fixtures or
                counts.failed != 0)
            {
                returncode = 2;
            }
        } else {
            returncode = 2;
        }
    }

    var missing: std.ArrayList([]const u8) = .empty;
    for (requested) |item| {
        const name = try std.fmt.allocPrint(allocator, "{s}-{s}-{s}.sample.txt", .{ item.fixture, item.phase, mode_name });
        const path = try std.fs.path.join(allocator, &.{ output_dir, name });
        const ok = if (std.Io.Dir.cwd().statFile(io, path, .{})) |stat|
            stat.kind == .file and stat.size > 0
        else |err| switch (err) {
            error.FileNotFound, error.NotDir => false,
            else => return err,
        };
        if (!ok) try missing.append(allocator, name);
    }
    if (returncode != 0 or missing.items.len != 0) {
        try err_out.print("profile {s} failed: runner={d}, missing={s}\n", .{
            mode_name,
            returncode,
            try formatStringList(allocator, missing.items),
        });
        return 1;
    }
    try out.print("Captured {d} {s} sampling profile(s).\n", .{ requested.len, mode_name });
    return 0;
}

pub fn defaultRunner(allocator: Allocator, repo: []const u8, suite: []const u8) ![]const []const u8 {
    const exe = if (@import("builtin").os.tag == .windows) "zig-out/bin/wasi-suite-runner.exe" else "zig-out/bin/wasi-suite-runner";
    return allocator.dupe([]const u8, &.{
        try std.fs.path.join(allocator, &.{ repo, exe }),
        "-t",
        suite,
        "-r",
        "wamr-zig",
        "--disable-colors",
    });
}

pub fn ensureRunnerExists(io: std.Io, runner: []const []const u8) !void {
    if (runner.len == 0) return error.MissingSubprocessRunner;
    const stat = std.Io.Dir.cwd().statFile(io, runner[0], .{}) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return error.MissingSubprocessRunner,
        else => return err,
    };
    if (stat.kind != .file) return error.MissingSubprocessRunner;
}

var signal_interrupted: std.atomic.Value(bool) = .init(false);
var signal_number: std.atomic.Value(i32) = .init(0);

fn onSignal(sig: std.posix.SIG) callconv(.c) void {
    signal_number.store(@intCast(@backingInt(sig)), .release);
    signal_interrupted.store(true, .release);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const raw_args = try init.minimal.args.toSlice(allocator);
    const cmd = parseArgs(raw_args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            std.debug.print(
                \\usage: wasi-p3-profile <run|profile> [options]
                \\  run --mode <aot|jit> --platform-id ID --output-dir DIR [--cold-samples N] [--warm-samples N] [--sample-timeout SEC] [--deadline-minutes MIN]
                \\  profile --mode <aot|jit> --selection FILE --output-dir DIR
                \\
            , .{});
            return;
        }
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (@import("builtin").os.tag == .linux or @import("builtin").os.tag == .macos) {
        const action: std.posix.Sigaction = .{
            .handler = .{ .handler = onSignal },
            .mask = std.posix.sigemptyset(),
            .flags = 0,
        };
        std.posix.sigaction(.INT, &action, null);
        std.posix.sigaction(.TERM, &action, null);
    }
    const repo = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
    const suite = try std.fs.path.join(allocator, &.{ repo, "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3" });
    const runner = try defaultRunner(allocator, repo, suite);
    ensureRunnerExists(io, runner) catch |err| {
        if (err == error.MissingSubprocessRunner) {
            std.debug.print("error: native WASI suite runner is missing; run `zig build build-wasi-suite-runner` first\n", .{});
        } else {
            std.debug.print("error: {s}\n", .{@errorName(err)});
        }
        std.process.exit(2);
    };
    var out_buf: std.Io.Writer.Allocating = .init(allocator);
    var err_buf: std.Io.Writer.Allocating = .init(allocator);
    const status = switch (cmd) {
        .run => |run_args| runCollection(allocator, io, init.environ_map, .{
            .repo = repo,
            .suite = suite,
            .runner = runner,
            .args = run_args,
            .interrupted = &signal_interrupted,
        }, &out_buf.writer),
        .profile => |prof_args| runProfile(allocator, io, init.environ_map, .{
            .repo = repo,
            .suite = suite,
            .runner = runner,
            .args = prof_args,
            .interrupted = &signal_interrupted,
        }, &out_buf.writer, &err_buf.writer),
    } catch |err| {
        if (err == error.Interrupted) {
            const sig = signal_number.load(.acquire);
            std.process.exit(if (sig > 0) @intCast(128 + sig) else 130);
        }
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (out_buf.written().len != 0) try std.Io.File.stdout().writeStreamingAll(io, out_buf.written());
    if (err_buf.written().len != 0) try std.Io.File.stderr().writeStreamingAll(io, err_buf.written());
    if (status != 0) std.process.exit(status);
}

test "parseArgs validates run and profile subcommands and options" {
    const run_cmd = try parseArgs(&.{
        "run",                  "--mode",             "aot",
        "--platform-id=linux",  "--output-dir",       "out",
        "--cold-samples",       "2",                  "--warm-samples=3",
        "--sample-timeout=900", "--deadline-minutes", "12.5",
    });
    try std.testing.expect(run_cmd == .run);
    try std.testing.expectEqual(collect.Mode.aot, run_cmd.run.mode);
    try std.testing.expectEqualStrings("linux", run_cmd.run.platform_id);
    try std.testing.expectEqualStrings("out", run_cmd.run.output_dir);
    try std.testing.expectEqual(@as(usize, 2), run_cmd.run.cold_samples);
    try std.testing.expectEqual(@as(usize, 3), run_cmd.run.warm_samples);
    try std.testing.expectEqual(@as(f64, 900.0), run_cmd.run.sample_timeout_s);
    try std.testing.expectEqual(@as(?f64, 12.5), run_cmd.run.deadline_minutes);

    const prof_cmd = try parseArgs(&.{ "profile", "--mode=jit", "--selection", "sel.json", "--output-dir=prof" });
    try std.testing.expect(prof_cmd == .profile);
    try std.testing.expectEqual(collect.Mode.jit, prof_cmd.profile.mode);
    try std.testing.expectEqualStrings("sel.json", prof_cmd.profile.selection);
    try std.testing.expectEqualStrings("prof", prof_cmd.profile.output_dir);

    try std.testing.expectError(error.MissingSubcommand, parseArgs(&.{}));
    try std.testing.expectError(error.UnknownSubcommand, parseArgs(&.{"unknown"}));
    try std.testing.expectError(error.InvalidProfileMode, parseArgs(&.{ "run", "--mode", "interp", "--platform-id", "x", "--output-dir", "y" }));
    try std.testing.expectError(error.InvalidPositiveInteger, parseArgs(&.{ "run", "--mode", "aot", "--platform-id", "x", "--output-dir", "y", "--cold-samples", "0" }));
    try std.testing.expectError(error.InvalidPositiveFloat, parseArgs(&.{ "run", "--mode", "aot", "--platform-id", "x", "--output-dir", "y", "--sample-timeout", "-1" }));
    try std.testing.expectError(error.InvalidPositiveFloat, parseArgs(&.{ "run", "--mode", "aot", "--platform-id", "x", "--output-dir", "y", "--deadline-minutes", "0" }));
}

test "ensureRunnerExists rejects missing runner executable" {
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root);
    const missing = try std.fs.path.join(std.testing.allocator, &.{ root, "missing-runner" });
    defer std.testing.allocator.free(missing);
    try std.testing.expectError(error.MissingSubprocessRunner, ensureRunnerExists(io, &.{missing}));
}

test "runCollection persists valid cold and warm samples and supports early deadline stop" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    const suite = try std.fs.path.join(allocator, &.{ root, "suite" });
    const runner = try std.fs.path.join(allocator, &.{ root, "runner" });
    try temp.dir.createDir(io, "suite", .default_dir);
    try temp.dir.writeFile(io, .{ .sub_path = "runner", .data =
        \\#!/bin/sh
        \\[ "$1" = --json-output-location ] && [ "$2" = "$WAMR_P3_REPORT" ] || exit 8
        \\case "$WAMR_PROFILE_RUN_ID" in *-cold-*) cache=miss;; *-warm-*) cache=hit;; *) exit 9;; esac
        \\[ "$WAMR_PROFILE_MODE" != jit ] || cache=bypass
        \\printf '{"results":[{"tests":[' > "$WAMR_P3_REPORT"
        \\i=0
        \\while [ "$i" -lt 41 ]; do
        \\  [ "$i" -eq 0 ] || printf ',' >> "$WAMR_P3_REPORT"
        \\  printf '{"executed":true,"failures":[]}' >> "$WAMR_P3_REPORT"
        \\  for phase in fixture_execution component_precompile; do
        \\    c="$cache"; [ "$phase" != fixture_execution ] || c=n/a
        \\    printf '{"schema_version":1,"event":"phase_timing","run_id":"%s","mode":"%s","fixture":"f%s","phase":"%s","artifact_kind":"component","cache":"%s","duration_ns":10,"pid":1}\n' "$WAMR_PROFILE_RUN_ID" "$WAMR_PROFILE_MODE" "$i" "$phase" "$c" >> "$WAMR_PROFILE_TIMINGS"
        \\  done
        \\  i=$((i + 1))
        \\done
        \\printf ']}]}' >> "$WAMR_P3_REPORT"
        \\
    });
    try temp.dir.setFilePermissions(io, "runner", .executable_file, .{});

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("PATH", "/usr/bin:/bin");
    try env.put("WAMR", "/bin/echo wamr-test");
    try env.put("WAMRC", "/bin/echo wamrc-test");
    try env.put("ZIG_GLOBAL_CACHE_DIR", "/d/test-cache");

    const ok_dir = try std.fs.path.join(allocator, &.{ root, "ok" });
    var ok_out: std.Io.Writer.Allocating = .init(allocator);
    const ok_status = try runCollection(allocator, io, &env, .{
        .repo = ".",
        .suite = suite,
        .runner = &.{runner},
        .args = .{
            .mode = .aot,
            .platform_id = "linux-arm64",
            .output_dir = ok_dir,
            .cold_samples = 1,
            .warm_samples = 2,
            .sample_timeout_s = 900.0,
        },
        .tools = .{ .zig = "/bin/echo" },
    }, &ok_out.writer);
    try std.testing.expectEqual(@as(u8, 0), ok_status);
    try std.testing.expect(std.mem.indexOf(u8, ok_out.written(), "aot-cold-01: valid=True passed=41/41") != null);
    try std.testing.expect(std.mem.indexOf(u8, ok_out.written(), "aot-warm-02: valid=True passed=41/41") != null);

    const samples_bytes = try temp.dir.readFileAlloc(io, "ok/samples.json", allocator, .unlimited);
    const parsed_ok = try std.json.parseFromSliceLeaky(Value, allocator, samples_bytes, .{});
    try schema.validateRun(parsed_ok);
    try std.testing.expect(!parsed_ok.object.get("stopped_early").?.bool);
    try std.testing.expectEqual(@as(usize, 3), parsed_ok.object.get("samples").?.array.items.len);
    try std.testing.expectEqual(@as(usize, 40), parsed_ok.object.get("metadata").?.object.get("commit").?.string.len);

    const deadline_dir = try std.fs.path.join(allocator, &.{ root, "deadline" });
    var deadline_out: std.Io.Writer.Allocating = .init(allocator);
    const deadline_status = try runCollection(allocator, io, &env, .{
        .repo = ".",
        .suite = suite,
        .runner = &.{runner},
        .args = .{
            .mode = .jit,
            .platform_id = "test",
            .output_dir = deadline_dir,
            .cold_samples = 5,
            .warm_samples = 10,
            .sample_timeout_s = 900.0,
            .deadline_minutes = 0.001,
        },
        .tools = .{ .zig = "/bin/echo" },
        .commit = "0000000000000000000000000000000000000000",
    }, &deadline_out.writer);
    try std.testing.expectEqual(@as(u8, 1), deadline_status);
    try std.testing.expectEqualStrings(
        "stopping early: collection deadline of 0.001min reached after 0 of 15 samples\n",
        deadline_out.written(),
    );
    const deadline_bytes = try temp.dir.readFileAlloc(io, "deadline/samples.json", allocator, .unlimited);
    const parsed_deadline = try std.json.parseFromSliceLeaky(Value, allocator, deadline_bytes, .{});
    try schema.validateRun(parsed_deadline);
    try std.testing.expect(parsed_deadline.object.get("stopped_early").?.bool);
    try std.testing.expectEqualStrings(
        "collection deadline of 0.001min reached after 0 of 15 samples",
        parsed_deadline.object.get("stop_reason").?.string,
    );
    try std.testing.expectEqual(@as(i64, 15), parsed_deadline.object.get("planned_samples").?.integer);
    try std.testing.expectEqual(@as(usize, 0), parsed_deadline.object.get("samples").?.array.items.len);
}

test "runProfile captures selected profiles and reports missing or empty sample files" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    const suite = try std.fs.path.join(allocator, &.{ root, "suite" });
    const selection = try std.fs.path.join(allocator, &.{ root, "selection.json" });
    const runner = try std.fs.path.join(allocator, &.{ root, "runner" });
    try temp.dir.createDir(io, "suite", .default_dir);
    try temp.dir.writeFile(io, .{ .sub_path = "suite/stale.cwasm", .data = "stale" });
    try temp.dir.writeFile(io, .{ .sub_path = "selection.json", .data =
        \\{"profiles":[
        \\  {"mode":"aot","fixture":"http-fields","phase":"fixture_execution"},
        \\  {"mode":"aot","fixture":"cli-args","phase":"component_precompile"}
        \\]}
    });
    try temp.dir.writeFile(io, .{ .sub_path = "runner", .data =
        \\#!/bin/sh
        \\[ "$1" = --json-output-location ] && [ "$2" = "$WAMR_P3_REPORT" ] || exit 8
        \\[ "$WAMR_TESTSUITE_TIMEOUT" = 300 ] || exit 9
        \\[ -n "$WAMR_PROFILE_LAUNCHER" ] || exit 10
        \\printf '{"results":[{"tests":[' > "$WAMR_P3_REPORT"
        \\i=0
        \\while [ "$i" -lt 41 ]; do
        \\  [ "$i" -eq 0 ] || printf ',' >> "$WAMR_P3_REPORT"
        \\  printf '{"executed":true,"failures":[]}' >> "$WAMR_P3_REPORT"
        \\  i=$((i + 1))
        \\done
        \\printf ']}]}' >> "$WAMR_P3_REPORT"
        \\if [ "$PROFILE_TEST_MODE" = ok ]; then
        \\  printf sample1 > "$WAMR_PROFILE_OUTPUT_DIR/http-fields-fixture_execution-aot.sample.txt"
        \\  printf sample2 > "$WAMR_PROFILE_OUTPUT_DIR/cli-args-component_precompile-aot.sample.txt"
        \\elif [ "$PROFILE_TEST_MODE" = partial ]; then
        \\  printf sample1 > "$WAMR_PROFILE_OUTPUT_DIR/http-fields-fixture_execution-aot.sample.txt"
        \\  : > "$WAMR_PROFILE_OUTPUT_DIR/cli-args-component_precompile-aot.sample.txt"
        \\fi
        \\
    });
    try temp.dir.setFilePermissions(io, "runner", .executable_file, .{});

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("PATH", "/usr/bin:/bin");

    // 1. No profiles selected for mode=jit
    var jit_out: std.Io.Writer.Allocating = .init(allocator);
    var jit_err: std.Io.Writer.Allocating = .init(allocator);
    const jit_status = try runProfile(allocator, io, &env, .{
        .repo = root,
        .suite = suite,
        .runner = &.{runner},
        .args = .{ .mode = .jit, .selection = selection, .output_dir = try std.fs.path.join(allocator, &.{ root, "jit-prof" }) },
    }, &jit_out.writer, &jit_err.writer);
    try std.testing.expectEqual(@as(u8, 0), jit_status);
    try std.testing.expectEqualStrings("No jit profiles selected.\n", jit_out.written());

    // 2. Successful capture for mode=aot
    try env.put("PROFILE_TEST_MODE", "ok");
    var ok_out: std.Io.Writer.Allocating = .init(allocator);
    var ok_err: std.Io.Writer.Allocating = .init(allocator);
    const ok_status = try runProfile(allocator, io, &env, .{
        .repo = root,
        .suite = suite,
        .runner = &.{runner},
        .args = .{ .mode = .aot, .selection = selection, .output_dir = try std.fs.path.join(allocator, &.{ root, "aot-ok" }) },
    }, &ok_out.writer, &ok_err.writer);
    try std.testing.expectEqual(@as(u8, 0), ok_status);
    try std.testing.expectEqualStrings("Captured 2 aot sampling profile(s).\n", ok_out.written());
    try std.testing.expectError(error.FileNotFound, temp.dir.statFile(io, "suite/stale.cwasm", .{}));

    // 3. Empty/missing sample file fails with diagnostic
    try env.put("PROFILE_TEST_MODE", "partial");
    var fail_out: std.Io.Writer.Allocating = .init(allocator);
    var fail_err: std.Io.Writer.Allocating = .init(allocator);
    const fail_status = try runProfile(allocator, io, &env, .{
        .repo = root,
        .suite = suite,
        .runner = &.{runner},
        .args = .{ .mode = .aot, .selection = selection, .output_dir = try std.fs.path.join(allocator, &.{ root, "aot-fail" }) },
    }, &fail_out.writer, &fail_err.writer);
    try std.testing.expectEqual(@as(u8, 1), fail_status);
    try std.testing.expectEqualStrings(
        "profile aot failed: runner=0, missing=['cli-args-component_precompile-aot.sample.txt']\n",
        fail_err.written(),
    );
}
