//! `wamrc verify <wasm>` — differential testing harness (issue #757).
//!
//! Compiles the input wasm under wamr-AOT, runs it under both wamr
//! (the subject) and wasmtime (the oracle), and diffs stdout (by
//! default), stderr, and exit codes. Exits 0 on match, 1 on
//! divergence, 2 on setup error.
//!
//! Why wasmtime as the oracle (and not wamr's interpreter): wamr's
//! interp may be removed entirely in the future — building tooling
//! on top of it is wasted work. Black-box differential against an
//! external runtime also catches the bigger bug class: things both
//! wamr engines share (canon-lift quirks, adapter bugs, …) that an
//! interp-vs-AOT diff would miss. wasmtime is already on every
//! wamr dev box (`/home/g/.wasmtime/bin/wasmtime`) and CI runner.
//!
//! Why direct std.process subprocesses (and not re-using `wamrc run`
//! recursively for the wamr arm): keeps the precompile step inside
//! the same process so allocations and per-pass diagnostics stream
//! through one allocator instead of two, and avoids the
//! sibling-`.cwasm.json` discovery dance polluting the source tree
//! (the temp staging dir under `/tmp/wamrc-verify-<pid>/` is wiped
//! on exit unless `--keep-cwasm`).
//!
//! Motivating use case: #754 took ~6 hours to bisect. The actual fix
//! was 13 lines. The slow part was building probes to localize
//! divergence between wamr-AOT and wasmtime. A one-line
//! `wamrc verify` would have shortened that to ~30 minutes.

const std = @import("std");
const builtin = @import("builtin");
const wamr = @import("wamr");
const verify_args = @import("verify_args.zig");

pub const Options = verify_args.Options;
pub const ParseDiagnostic = verify_args.ParseDiagnostic;
pub const ParseError = verify_args.ParseError;
pub const parse = verify_args.parse;

/// Top-level outcome reported through both human and JSON renderers.
/// Exit codes:
///   `.match`         → 0
///   `.diverged`      → 1
///   `.setup_error`   → 2
pub const Outcome = enum { match, diverged, setup_error };

/// One side's captured run shape.
const RunCapture = struct {
    stdout: []u8,
    stderr: []u8,
    /// Exit code if the runtime exited normally. `null` covers
    /// signals, stops, unknown statuses, and timeout kills — every
    /// case where there is no clean numeric code to compare against.
    /// Pair with `term_tag` for the human-readable termination class.
    exit_code: ?u8,
    term_tag: TermTag,
    elapsed_ms: i64,

    pub fn deinit(self: *RunCapture, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }
};

const TermTag = enum {
    exited,
    signal,
    stopped,
    unknown,
    timeout,
    spawn_failed,

    fn name(self: TermTag) []const u8 {
        return @tagName(self);
    }
};

/// Public entry point. Returns the exit code `wamrc` should
/// propagate; never calls `std.process.exit` itself so the caller
/// (`main.zig:runVerify`) keeps responsibility for top-level
/// process termination semantics.
pub fn run(
    init: std.process.Init,
    allocator: std.mem.Allocator,
    options: Options,
) !u8 {
    // 1. Read the input wasm. `verify` only inspects bytes (component
    //    vs core wasm + AOT precompile); the source file is also
    //    handed verbatim to both runtimes' CLIs.
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const wasm_data = cwd.readFileAlloc(io, options.wasm_path, allocator, @enumFromInt(256 * 1024 * 1024)) catch |err| {
        std.debug.print("[verify] error: failed to read '{s}': {s}\n", .{ options.wasm_path, @errorName(err) });
        return 2;
    };
    defer allocator.free(wasm_data);

    const is_component = wamr.component_aot.isComponent(wasm_data) catch {
        std.debug.print("[verify] error: '{s}' is not a valid wasm module or component\n", .{options.wasm_path});
        return 2;
    };

    // 2. Resolve runtime binaries. Errors here are setup errors
    //    (exit 2) — a clear "install wasmtime ≥ 45 / set WASMTIME_BIN"
    //    is the action item, not a divergence report.
    const wasmtime_bin = try resolveWasmtimeBinary(allocator, init.environ_map, options.wasmtime_bin) orelse {
        std.debug.print(
            "[verify] error: wasmtime not found.\n" ++
                "  Install wasmtime ≥ 45 (https://wasmtime.dev/install/),\n" ++
                "  set $WASMTIME_BIN, or pass --wasmtime-bin <path>.\n",
            .{},
        );
        return 2;
    };
    defer allocator.free(wasmtime_bin);

    const wamr_bin = try resolveWamrBinary(allocator, io, init.environ_map, options.wamr_bin);
    defer allocator.free(wamr_bin);

    // 3. Stage AOT artifacts in a per-pid temp dir so the source
    //    tree stays clean. The dir holds the manifest sidecar +
    //    per-core .cwasm files for components, or the single .cwasm
    //    for a core module. Cleaned up on return unless --keep-cwasm.
    const stage = stageDir(allocator, io) catch |err| {
        std.debug.print("[verify] error: failed to create staging dir: {s}\n", .{@errorName(err)});
        return 2;
    };
    defer allocator.free(stage.dir);
    defer if (!options.keep_cwasm) {
        std.Io.Dir.cwd().deleteTree(io, stage.dir) catch |err| {
            std.debug.print("[verify] warning: failed to clean up '{s}': {s}\n", .{ stage.dir, @errorName(err) });
        };
    };
    if (options.keep_cwasm) {
        std.debug.print("[verify] --keep-cwasm: artifacts staying in {s}\n", .{stage.dir});
    }

    // 4. AOT compile. Component path writes a sibling manifest +
    //    .N.cwasm files inside `stage.dir`; core wasm writes one
    //    `program.cwasm`. The wamr arm consumes those via
    //    `--precompiled-manifest=` so it never auto-probes next to
    //    the source.
    std.debug.print("[verify] compiling AOT...\n", .{});
    var precompiled_manifest_path: ?[]u8 = null;
    defer if (precompiled_manifest_path) |p| allocator.free(p);
    var core_cwasm_path: ?[]u8 = null;
    defer if (core_cwasm_path) |p| allocator.free(p);

    if (is_component) {
        const mp = try std.fs.path.join(allocator, &.{ stage.dir, "manifest.cwasm.json" });
        var result = wamr.component_aot_compile.precompileComponent(allocator, wasm_data, mp, .{}) catch |err| {
            std.debug.print("[verify] error: AOT precompile failed: {s}\n", .{@errorName(err)});
            allocator.free(mp);
            return 2;
        };
        result.deinit();
        precompiled_manifest_path = mp;
    } else {
        const cp = try std.fs.path.join(allocator, &.{ stage.dir, "program.cwasm" });
        const cwasm_bytes = wamr.component_aot_compile.compileCoreWasm(allocator, wasm_data, .{}) catch |err| {
            std.debug.print("[verify] error: AOT compile failed: {s}\n", .{@errorName(err)});
            allocator.free(cp);
            return 2;
        };
        defer allocator.free(cwasm_bytes);
        cwd.writeFile(io, .{ .sub_path = cp, .data = cwasm_bytes }) catch |err| {
            std.debug.print("[verify] error: failed to write {s}: {s}\n", .{ cp, @errorName(err) });
            allocator.free(cp);
            return 2;
        };
        core_cwasm_path = cp;
    }
    std.debug.print("[verify] AOT compile done\n", .{});

    // 5. Run wasmtime (the oracle).
    var oracle = try runWasmtime(allocator, io, wasmtime_bin, options);
    defer oracle.deinit(allocator);
    reportSide("wasmtime (oracle)", &oracle);
    if (oracle.term_tag == .spawn_failed) return 2;

    // 6. Run wamr (the subject).
    var subject = try runWamr(
        allocator,
        io,
        wamr_bin,
        options,
        precompiled_manifest_path,
        core_cwasm_path,
        is_component,
    );
    defer subject.deinit(allocator);
    reportSide("wamr (subject)", &subject);
    if (subject.term_tag == .spawn_failed) return 2;

    // 7. Diff and report.
    return diffAndReport(allocator, options, oracle, subject);
}

// ── Binary resolution ───────────────────────────────────────────────────

fn resolveWasmtimeBinary(
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
    override: ?[]const u8,
) !?[]u8 {
    if (override) |p| {
        if (p.len > 0) return try allocator.dupe(u8, p);
    }
    if (env.get("WASMTIME_BIN")) |p| {
        if (p.len > 0) return try allocator.dupe(u8, p);
    }
    // Don't probe PATH ourselves — `spawn` does that. Returning the
    // bare name lets the OS resolve it and surface a clearer
    // FileNotFound if it really isn't installed.
    return try allocator.dupe(u8, "wasmtime");
}

fn resolveWamrBinary(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    override: ?[]const u8,
) ![]u8 {
    if (override) |p| {
        if (p.len > 0) return try allocator.dupe(u8, p);
    }
    if (env.get("WAMR_BIN")) |p| {
        if (p.len > 0) return try allocator.dupe(u8, p);
    }
    // Sibling-next-to-wamrc fallback mirrors `runRun`'s
    // `findWamrBinary` in main.zig — keeps behaviour consistent so
    // a `wamrc verify` works without explicit configuration on a
    // freshly-built tree.
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (std.process.executablePath(io, &exe_buf)) |n| {
        const exe_path = exe_buf[0..n];
        const dir = std.fs.path.dirname(exe_path) orelse "";
        if (dir.len > 0) {
            return try std.fs.path.join(allocator, &.{ dir, "wamr" });
        }
    } else |_| {}
    return try allocator.dupe(u8, "wamr");
}

// ── Subprocess invocations ──────────────────────────────────────────────

const StageInfo = struct { dir: []u8 };

fn stageDir(allocator: std.mem.Allocator, io: std.Io) !StageInfo {
    // `/tmp/wamrc-verify-<pid>` — the pid suffix prevents collisions
    // between parallel `wamrc verify` invocations on the same box.
    const pid: i64 = @intCast(std.os.linux.getpid());
    const dir = try std.fmt.allocPrint(allocator, "/tmp/wamrc-verify-{d}", .{pid});
    errdefer allocator.free(dir);
    std.Io.Dir.cwd().createDirPath(io, dir) catch |err| switch (err) {
        else => return err,
    };
    return .{ .dir = dir };
}

fn runWasmtime(
    allocator: std.mem.Allocator,
    io: std.Io,
    wasmtime_bin: []const u8,
    options: Options,
) !RunCapture {
    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(allocator);
    try verify_args.buildWasmtimeArgs(allocator, options, wasmtime_bin, &args);
    return spawnAndCapture(allocator, io, args.items, options.max_runtime_seconds);
}

fn runWamr(
    allocator: std.mem.Allocator,
    io: std.Io,
    wamr_bin: []const u8,
    options: Options,
    precompiled_manifest_path: ?[]const u8,
    core_cwasm_path: ?[]const u8,
    is_component: bool,
) !RunCapture {
    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(allocator);

    // `--precompiled-manifest=<staged path>` is the only argv slot
    // that needs an allocation past the parsed-Options slice
    // lifetimes — everything else either borrows from the caller's
    // argv or the staging-dir paths we already own. Capture it here
    // so its `defer free` lives in the same scope as the `spawn`.
    var manifest_arg: ?[]u8 = null;
    defer if (manifest_arg) |m| allocator.free(m);

    // wamr CLI: `wamr run [--precompiled-manifest=<p>] [--map-dir A::B]
    //                     [--env K=V] <input> [-- guest-args...]`.
    // Build it longhand here instead of calling verify_args.buildWamrArgs
    // because we also need to inject `--precompiled-manifest` (when the
    // input is a component) or swap the input path for the precompiled
    // `.cwasm` (when it's a core module) — those are verify-specific
    // wirings, not shared with anything else.
    try args.append(allocator, wamr_bin);
    try args.append(allocator, "run");
    if (is_component) {
        if (precompiled_manifest_path) |p| {
            manifest_arg = try std.fmt.allocPrint(allocator, "--precompiled-manifest={s}", .{p});
            try args.append(allocator, manifest_arg.?);
        }
    }
    for (options.map_dirs) |md| {
        try args.append(allocator, "--map-dir");
        try args.append(allocator, md);
    }
    for (options.env_vars) |ev| {
        try args.append(allocator, "--env");
        try args.append(allocator, ev);
    }
    if (is_component) {
        try args.append(allocator, options.wasm_path);
    } else if (core_cwasm_path) |p| {
        try args.append(allocator, p);
    } else {
        try args.append(allocator, options.wasm_path);
    }
    if (options.guest_args.len > 0) {
        try args.append(allocator, "--");
        for (options.guest_args) |ga| try args.append(allocator, ga);
    }

    return spawnAndCapture(allocator, io, args.items, options.max_runtime_seconds);
}

fn nowMs(io: std.Io) i64 {
    return std.Io.Clock.awake.now(io).toMilliseconds();
}

fn spawnAndCapture(
    allocator: std.mem.Allocator,
    io: std.Io,
    argv: []const []const u8,
    max_runtime_seconds: u32,
) !RunCapture {
    const start_ms = nowMs(io);
    const timeout: std.Io.Timeout = if (max_runtime_seconds == 0)
        .none
    else
        .{ .duration = .{ .raw = .fromSeconds(@intCast(max_runtime_seconds)), .clock = .awake } };
    const result = std.process.run(allocator, io, .{
        .argv = argv,
        .timeout = timeout,
    }) catch |err| {
        const elapsed_ms = nowMs(io) - start_ms;
        switch (err) {
            error.Timeout => {
                return .{
                    .stdout = try allocator.alloc(u8, 0),
                    .stderr = try allocator.alloc(u8, 0),
                    .exit_code = null,
                    .term_tag = .timeout,
                    .elapsed_ms = elapsed_ms,
                };
            },
            else => {
                std.debug.print("[verify] error: spawn '{s}' failed: {s}\n", .{ argv[0], @errorName(err) });
                return .{
                    .stdout = try allocator.alloc(u8, 0),
                    .stderr = try allocator.alloc(u8, 0),
                    .exit_code = null,
                    .term_tag = .spawn_failed,
                    .elapsed_ms = elapsed_ms,
                };
            },
        }
    };
    const elapsed_ms = nowMs(io) - start_ms;

    var capture: RunCapture = .{
        .stdout = result.stdout,
        .stderr = result.stderr,
        .exit_code = null,
        .term_tag = .unknown,
        .elapsed_ms = elapsed_ms,
    };
    switch (result.term) {
        .exited => |code| {
            capture.exit_code = code;
            capture.term_tag = .exited;
        },
        .signal => capture.term_tag = .signal,
        .stopped => capture.term_tag = .stopped,
        .unknown => capture.term_tag = .unknown,
    }
    return capture;
}

// ── Diff + reporting ────────────────────────────────────────────────────

fn diffAndReport(
    allocator: std.mem.Allocator,
    options: Options,
    oracle: RunCapture,
    subject: RunCapture,
) !u8 {
    const stdout_div: ?usize = firstDiff(oracle.stdout, subject.stdout);
    const stderr_div: ?usize = firstDiff(oracle.stderr, subject.stderr);

    const stdout_matches = stdout_div == null and oracle.stdout.len == subject.stdout.len;
    const stderr_matches = stderr_div == null and oracle.stderr.len == subject.stderr.len;

    const stream_matches = switch (options.diff_mode) {
        .stdout_only => stdout_matches,
        .stderr_only => stderr_matches,
        .both => stdout_matches and stderr_matches,
    };

    const exit_matches = exitMatches(oracle, subject);
    const overall_matches = stream_matches and (!options.strict_exit or exit_matches);

    if (options.json) {
        try emitJson(allocator, options, oracle, subject, stdout_div, stderr_div, overall_matches);
    } else {
        try emitHuman(allocator, options, oracle, subject, stdout_div, stderr_div, overall_matches, exit_matches);
    }

    return if (overall_matches) 0 else 1;
}

fn firstDiff(a: []const u8, b: []const u8) ?usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n) : (i += 1) if (a[i] != b[i]) return i;
    if (a.len != b.len) return n;
    return null;
}

fn exitMatches(o: RunCapture, s: RunCapture) bool {
    if (o.exit_code == null or s.exit_code == null) return false;
    return o.exit_code.? == s.exit_code.?;
}

fn reportSide(label: []const u8, capture: *const RunCapture) void {
    std.debug.print(
        "[verify] running {s}... ({d} ms, {d} B stdout, {d} B stderr, {s}",
        .{ label, capture.elapsed_ms, capture.stdout.len, capture.stderr.len, capture.term_tag.name() },
    );
    if (capture.exit_code) |c| std.debug.print(" {d}", .{c});
    std.debug.print(")\n", .{});
}

fn emitHuman(
    allocator: std.mem.Allocator,
    options: Options,
    oracle: RunCapture,
    subject: RunCapture,
    stdout_div: ?usize,
    stderr_div: ?usize,
    overall_matches: bool,
    exit_matches: bool,
) !void {
    _ = allocator;
    const want_stdout = options.diff_mode != .stderr_only;
    const want_stderr = options.diff_mode != .stdout_only;

    if (want_stdout) {
        if (stdout_div) |off| {
            try printDivergence("stdout", off, oracle.stdout, subject.stdout, options.hex_context);
        } else if (oracle.stdout.len != subject.stdout.len) {
            std.debug.print(
                "[verify] stdout length mismatch: wasmtime={d}B wamr={d}B\n",
                .{ oracle.stdout.len, subject.stdout.len },
            );
        }
    }
    if (want_stderr) {
        if (stderr_div) |off| {
            try printDivergence("stderr", off, oracle.stderr, subject.stderr, options.hex_context);
        } else if (oracle.stderr.len != subject.stderr.len) {
            std.debug.print(
                "[verify] stderr length mismatch: wasmtime={d}B wamr={d}B\n",
                .{ oracle.stderr.len, subject.stderr.len },
            );
        }
    }

    if (!exit_matches) {
        const oc = if (oracle.exit_code) |c| @as(i32, c) else -1;
        const sc = if (subject.exit_code) |c| @as(i32, c) else -1;
        const kind: []const u8 = if (options.strict_exit) "DIVERGENCE" else "warning";
        std.debug.print(
            "[verify] exit-code {s}: wasmtime={d} ({s}) wamr={d} ({s})\n",
            .{ kind, oc, oracle.term_tag.name(), sc, subject.term_tag.name() },
        );
    }

    if (overall_matches) {
        switch (options.diff_mode) {
            .stdout_only => std.debug.print("[verify] OK stdout {d}B match (stderr/exit ignored)\n", .{oracle.stdout.len}),
            .stderr_only => std.debug.print("[verify] OK stderr {d}B match (stdout/exit ignored)\n", .{oracle.stderr.len}),
            .both => std.debug.print("[verify] OK stdout {d}B / stderr {d}B match\n", .{ oracle.stdout.len, oracle.stderr.len }),
        }
        if (options.strict_exit) std.debug.print("[verify] exit codes match\n", .{});
    } else {
        std.debug.print("[verify] FAIL\n", .{});
        std.debug.print(
            "  hint: codegen-shaped bug? see .github/skills/aot-diff-debug/SKILL.md\n",
            .{},
        );
    }
}

fn printDivergence(
    stream: []const u8,
    offset: usize,
    a: []const u8,
    b: []const u8,
    hex_context: u32,
) !void {
    std.debug.print("[verify] DIVERGENCE at {s} offset 0x{x} ({d} bytes in)\n", .{ stream, offset, offset });
    try printHexLine("  wasmtime: ", a, offset, hex_context);
    try printHexLine("  wamr:     ", b, offset, hex_context);
}

fn printHexLine(
    label: []const u8,
    buf: []const u8,
    offset: usize,
    hex_context: u32,
) !void {
    const ctx = hex_context;
    const start = if (offset > ctx) offset - ctx else 0;
    const end = @min(buf.len, offset + ctx);
    std.debug.print("{s}", .{label});
    if (start >= buf.len) {
        std.debug.print("(stream ended before offset {d})\n", .{offset});
        return;
    }
    // Hex bytes.
    var i = start;
    while (i < end) : (i += 1) {
        std.debug.print("{x:0>2} ", .{buf[i]});
    }
    std.debug.print(" | ", .{});
    // ASCII gloss with `.` for non-printable / space and a `^` marker
    // immediately under the divergence byte.
    i = start;
    while (i < end) : (i += 1) {
        const c = buf[i];
        const printable: u8 = if (c >= 0x20 and c < 0x7f) c else '.';
        std.debug.print("{c}", .{printable});
    }
    std.debug.print("\n", .{});
}

fn emitJson(
    allocator: std.mem.Allocator,
    options: Options,
    oracle: RunCapture,
    subject: RunCapture,
    stdout_div: ?usize,
    stderr_div: ?usize,
    overall_matches: bool,
) !void {
    _ = allocator;
    var stdout_file = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);

    try w.print(
        "{{\"match\":{s},\"diff_mode\":\"{s}\",\"strict_exit\":{s}",
        .{
            if (overall_matches) "true" else "false",
            @tagName(options.diff_mode),
            if (options.strict_exit) "true" else "false",
        },
    );
    try w.print(
        ",\"wasmtime\":{{\"stdout_len\":{d},\"stderr_len\":{d},\"exit\":{d},\"term\":\"{s}\",\"elapsed_ms\":{d}}}",
        .{
            oracle.stdout.len,           oracle.stderr.len,
            if (oracle.exit_code) |c| @as(i32, c) else -1,
            oracle.term_tag.name(),      oracle.elapsed_ms,
        },
    );
    try w.print(
        ",\"wamr\":{{\"stdout_len\":{d},\"stderr_len\":{d},\"exit\":{d},\"term\":\"{s}\",\"elapsed_ms\":{d}}}",
        .{
            subject.stdout.len,           subject.stderr.len,
            if (subject.exit_code) |c| @as(i32, c) else -1,
            subject.term_tag.name(),      subject.elapsed_ms,
        },
    );
    if (stdout_div) |off| try w.print(",\"stdout_first_diff\":{d}", .{off});
    if (stderr_div) |off| try w.print(",\"stderr_first_diff\":{d}", .{off});
    try w.print("}}\n", .{});

    stdout_file.writeStreamingAll(std.Io.Threaded.global_single_threaded.io(), w.buffered()) catch {};
}

// ── Tests ───────────────────────────────────────────────────────────────

test "firstDiff: identical" {
    try std.testing.expectEqual(@as(?usize, null), firstDiff("abc", "abc"));
}

test "firstDiff: different at index 0" {
    try std.testing.expectEqual(@as(?usize, 0), firstDiff("xbc", "abc"));
}

test "firstDiff: different at middle index" {
    try std.testing.expectEqual(@as(?usize, 2), firstDiff("abcd", "abxd"));
}

test "firstDiff: prefix shorter" {
    try std.testing.expectEqual(@as(?usize, 3), firstDiff("abc", "abcd"));
}

test "firstDiff: prefix longer" {
    try std.testing.expectEqual(@as(?usize, 3), firstDiff("abcd", "abc"));
}

test "exitMatches: both exited same → true" {
    const o: RunCapture = .{ .stdout = &.{}, .stderr = &.{}, .exit_code = 0, .term_tag = .exited, .elapsed_ms = 0 };
    const s: RunCapture = .{ .stdout = &.{}, .stderr = &.{}, .exit_code = 0, .term_tag = .exited, .elapsed_ms = 0 };
    try std.testing.expect(exitMatches(o, s));
}

test "exitMatches: one missing → false" {
    const o: RunCapture = .{ .stdout = &.{}, .stderr = &.{}, .exit_code = 0, .term_tag = .exited, .elapsed_ms = 0 };
    const s: RunCapture = .{ .stdout = &.{}, .stderr = &.{}, .exit_code = null, .term_tag = .signal, .elapsed_ms = 0 };
    try std.testing.expect(!exitMatches(o, s));
}

test "exitMatches: different codes → false" {
    const o: RunCapture = .{ .stdout = &.{}, .stderr = &.{}, .exit_code = 0, .term_tag = .exited, .elapsed_ms = 0 };
    const s: RunCapture = .{ .stdout = &.{}, .stderr = &.{}, .exit_code = 1, .term_tag = .exited, .elapsed_ms = 0 };
    try std.testing.expect(!exitMatches(o, s));
}
