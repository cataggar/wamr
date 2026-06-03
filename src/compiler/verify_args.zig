//! Argument parsing + per-runtime translation for `wamrc verify`
//! (issue #757).
//!
//! Kept separate from `verify.zig` (the subprocess orchestrator) so
//! the flag-translation table is easy to unit-test in-process,
//! without spawning anything. Every public function here is pure on
//! its inputs and returns owned slices the caller must free.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const DiffMode = enum {
    /// Diff stdout only. The default — wamr emits more stderr noise
    /// (`[aot dispatch] …`) than wasmtime, so a strict-stderr default
    /// would surface false positives. Use `--strict-stderr` to opt in.
    stdout_only,
    /// Diff stderr only. Useful when the guest writes diagnostics to
    /// stderr and the user wants to differentialise *just* that channel.
    stderr_only,
    /// Diff both streams.
    both,
};

pub const Options = struct {
    /// Path to the input `.wasm` file. The orchestrator hands this to
    /// `wasmtime run` directly and to `wamrc compile-component` /
    /// `wamr run` after AOT compilation.
    wasm_path: []const u8,

    /// `--wasmtime-bin <path>` override. Resolution order in
    /// `verify.zig`: this field → `$WASMTIME_BIN` → `wasmtime` on `$PATH`.
    wasmtime_bin: ?[]const u8 = null,

    /// `--wamr-bin <path>` override. Resolution order in `verify.zig`:
    /// this field → `$WAMR_BIN` → sibling-to-wamrc → `wamr` on `$PATH`.
    wamr_bin: ?[]const u8 = null,

    /// Each `--map-dir HOST::GUEST` flag. Slices borrow from the
    /// caller's argv; lifetime is the caller's.
    map_dirs: []const []const u8 = &.{},

    /// Each `--env K=V` (or `--env K`) flag.
    env_vars: []const []const u8 = &.{},

    /// Guest argv after a literal `--` (`wamrc verify foo.wasm -- a b c`
    /// → `["a", "b", "c"]`).
    guest_args: []const []const u8 = &.{},

    /// `--max-runtime <sec>` watchdog for each runtime invocation
    /// (default 60). Zero disables the watchdog.
    max_runtime_seconds: u32 = 60,

    /// `--hex-context <N>` — bytes of hex+ASCII context to print on
    /// each side of the first-divergence offset (default 32).
    hex_context: u32 = 32,

    diff_mode: DiffMode = .stdout_only,

    /// `--strict-exit` — also diff the runtimes' exit codes. Without
    /// it, a code mismatch is only a warning when the chosen stream(s)
    /// match. Default off — wamr's "successful run then host SIGSEGV"
    /// failure class (separate latent bug; see #760 for one instance)
    /// would otherwise drown out genuine codegen regressions.
    strict_exit: bool = false,

    /// `--keep-cwasm` — leave the precompiled `<stem>.cwasm` and
    /// `<stem>.cwasm.json` next to the input for post-mortem
    /// inspection. Default: delete on exit.
    keep_cwasm: bool = false,

    /// `--json` — emit a single-line JSON report instead of the
    /// human-readable shape. Exit code semantics are unchanged.
    json: bool = false,
};

pub const ParseError = error{
    MissingValue,
    DuplicateInputPath,
    UnknownOption,
    MissingInputPath,
    InvalidIntegerArgument,
} || std.mem.Allocator.Error;

/// In-memory error context populated by `parse` when it returns a
/// `ParseError`. Callers print the surfaced `option` / `value` /
/// `detail` fields together with their own usage hint.
pub const ParseDiagnostic = struct {
    option: []const u8 = "",
    value: []const u8 = "",
    detail: []const u8 = "",
};

/// Owns the variable-length slices `map_dirs`, `env_vars`, and
/// `guest_args` allocated during parse. The string contents inside
/// each slice borrow from the original argv (caller-owned).
pub const ParsedOptions = struct {
    options: Options,
    map_dirs: std.ArrayList([]const u8),
    env_vars: std.ArrayList([]const u8),
    guest_args: std.ArrayList([]const u8),
    allocator: Allocator,

    pub fn deinit(self: *ParsedOptions) void {
        self.map_dirs.deinit(self.allocator);
        self.env_vars.deinit(self.allocator);
        self.guest_args.deinit(self.allocator);
    }
};

/// Parse `wamrc verify` subcommand arguments. `argv` excludes the
/// `verify` token itself (matches the convention used by every other
/// `wamrc` subcommand in `src/compiler/main.zig`).
///
/// On `ParseError.X` the `diag` is populated; the caller prints it +
/// a `try wamrc verify help` line. The returned `ParsedOptions` owns
/// its three slices and must be `.deinit()`'d on success.
pub fn parse(
    allocator: Allocator,
    argv: []const []const u8,
    diag: *ParseDiagnostic,
) ParseError!ParsedOptions {
    var parsed: ParsedOptions = .{
        .allocator = allocator,
        .options = .{ .wasm_path = "" },
        .map_dirs = .empty,
        .env_vars = .empty,
        .guest_args = .empty,
    };
    errdefer parsed.deinit();

    var input_path: ?[]const u8 = null;
    var saw_dashdash = false;

    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];

        if (saw_dashdash) {
            try parsed.guest_args.append(allocator, a);
            continue;
        }
        if (std.mem.eql(u8, a, "--")) {
            saw_dashdash = true;
            continue;
        }

        // ── Flags taking a value ────────────────────────────────────────
        if (std.mem.eql(u8, a, "--map-dir")) {
            const v = nextValue(argv, &i) catch {
                diag.option = "--map-dir";
                return ParseError.MissingValue;
            };
            try parsed.map_dirs.append(allocator, v);
            continue;
        }
        if (std.mem.startsWith(u8, a, "--map-dir=")) {
            try parsed.map_dirs.append(allocator, a["--map-dir=".len..]);
            continue;
        }
        if (std.mem.eql(u8, a, "--env")) {
            const v = nextValue(argv, &i) catch {
                diag.option = "--env";
                return ParseError.MissingValue;
            };
            try parsed.env_vars.append(allocator, v);
            continue;
        }
        if (std.mem.startsWith(u8, a, "--env=")) {
            try parsed.env_vars.append(allocator, a["--env=".len..]);
            continue;
        }
        if (std.mem.eql(u8, a, "--max-runtime")) {
            const v = nextValue(argv, &i) catch {
                diag.option = "--max-runtime";
                return ParseError.MissingValue;
            };
            parsed.options.max_runtime_seconds = parseU32(v) catch {
                diag.option = "--max-runtime";
                diag.value = v;
                return ParseError.InvalidIntegerArgument;
            };
            continue;
        }
        if (std.mem.startsWith(u8, a, "--max-runtime=")) {
            const v = a["--max-runtime=".len..];
            parsed.options.max_runtime_seconds = parseU32(v) catch {
                diag.option = "--max-runtime";
                diag.value = v;
                return ParseError.InvalidIntegerArgument;
            };
            continue;
        }
        if (std.mem.eql(u8, a, "--hex-context")) {
            const v = nextValue(argv, &i) catch {
                diag.option = "--hex-context";
                return ParseError.MissingValue;
            };
            parsed.options.hex_context = parseU32(v) catch {
                diag.option = "--hex-context";
                diag.value = v;
                return ParseError.InvalidIntegerArgument;
            };
            continue;
        }
        if (std.mem.startsWith(u8, a, "--hex-context=")) {
            const v = a["--hex-context=".len..];
            parsed.options.hex_context = parseU32(v) catch {
                diag.option = "--hex-context";
                diag.value = v;
                return ParseError.InvalidIntegerArgument;
            };
            continue;
        }
        if (std.mem.eql(u8, a, "--wasmtime-bin")) {
            parsed.options.wasmtime_bin = nextValue(argv, &i) catch {
                diag.option = "--wasmtime-bin";
                return ParseError.MissingValue;
            };
            continue;
        }
        if (std.mem.startsWith(u8, a, "--wasmtime-bin=")) {
            parsed.options.wasmtime_bin = a["--wasmtime-bin=".len..];
            continue;
        }
        if (std.mem.eql(u8, a, "--wamr-bin")) {
            parsed.options.wamr_bin = nextValue(argv, &i) catch {
                diag.option = "--wamr-bin";
                return ParseError.MissingValue;
            };
            continue;
        }
        if (std.mem.startsWith(u8, a, "--wamr-bin=")) {
            parsed.options.wamr_bin = a["--wamr-bin=".len..];
            continue;
        }

        // ── Boolean flags ───────────────────────────────────────────────
        if (std.mem.eql(u8, a, "--stdout-only")) {
            parsed.options.diff_mode = .stdout_only;
            continue;
        }
        if (std.mem.eql(u8, a, "--stderr-only")) {
            parsed.options.diff_mode = .stderr_only;
            continue;
        }
        if (std.mem.eql(u8, a, "--diff-everything")) {
            parsed.options.diff_mode = .both;
            continue;
        }
        if (std.mem.eql(u8, a, "--strict-exit")) {
            parsed.options.strict_exit = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--keep-cwasm")) {
            parsed.options.keep_cwasm = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--json")) {
            parsed.options.json = true;
            continue;
        }

        // ── Unknown options ─────────────────────────────────────────────
        if (a.len > 0 and a[0] == '-') {
            diag.option = a;
            return ParseError.UnknownOption;
        }

        // ── Positional: the input .wasm path ────────────────────────────
        if (input_path != null) {
            diag.value = a;
            return ParseError.DuplicateInputPath;
        }
        input_path = a;
    }

    parsed.options.wasm_path = input_path orelse return ParseError.MissingInputPath;
    parsed.options.map_dirs = parsed.map_dirs.items;
    parsed.options.env_vars = parsed.env_vars.items;
    parsed.options.guest_args = parsed.guest_args.items;
    return parsed;
}

fn nextValue(argv: []const []const u8, i: *usize) error{Missing}![]const u8 {
    if (i.* + 1 >= argv.len) return error.Missing;
    i.* += 1;
    return argv[i.*];
}

fn parseU32(s: []const u8) !u32 {
    return std.fmt.parseInt(u32, s, 10);
}

// ── Per-runtime argv builders ───────────────────────────────────────────
//
// `wasmtime run` and `wamr run` accept the same kinds of inputs
// (mounts, env, the wasm path, then guest argv after `--`) but the
// flag spelling differs:
//
//   | concept | wasmtime          | wamr               |
//   |---------|-------------------|--------------------|
//   | mount   | `--dir A::B`      | `--map-dir A::B`   |
//   | envvar  | `--env K=V`       | `--env K=V`        |
//
// The builders below are pure functions of an `Options` value plus
// the runtime's binary path; verify.zig calls them once per
// invocation and hands the result straight to `std.process.run`.
//
// Each builder appends to a caller-owned `std.ArrayList`. The list
// owns the spine; the strings inside are borrowed (they ultimately
// trace back to the parsed argv).

pub fn buildWasmtimeArgs(
    allocator: Allocator,
    options: Options,
    wasmtime_path: []const u8,
    list: *std.ArrayList([]const u8),
) !void {
    try list.append(allocator, wasmtime_path);
    try list.append(allocator, "run");
    for (options.map_dirs) |md| {
        try list.append(allocator, "--dir");
        try list.append(allocator, md);
    }
    for (options.env_vars) |ev| {
        try list.append(allocator, "--env");
        try list.append(allocator, ev);
    }
    try list.append(allocator, options.wasm_path);
    if (options.guest_args.len > 0) {
        try list.append(allocator, "--");
        for (options.guest_args) |ga| try list.append(allocator, ga);
    }
}

pub fn buildWamrArgs(
    allocator: Allocator,
    options: Options,
    wamr_path: []const u8,
    list: *std.ArrayList([]const u8),
) !void {
    try list.append(allocator, wamr_path);
    try list.append(allocator, "run");
    for (options.map_dirs) |md| {
        try list.append(allocator, "--map-dir");
        try list.append(allocator, md);
    }
    for (options.env_vars) |ev| {
        try list.append(allocator, "--env");
        try list.append(allocator, ev);
    }
    try list.append(allocator, options.wasm_path);
    if (options.guest_args.len > 0) {
        try list.append(allocator, "--");
        for (options.guest_args) |ga| try list.append(allocator, ga);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

test "parse: bare wasm path" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{"foo.wasm"};
    var parsed = try parse(std.testing.allocator, &argv, &diag);
    defer parsed.deinit();
    try std.testing.expectEqualStrings("foo.wasm", parsed.options.wasm_path);
    try std.testing.expectEqual(@as(usize, 0), parsed.options.map_dirs.len);
    try std.testing.expectEqual(@as(usize, 0), parsed.options.env_vars.len);
    try std.testing.expectEqual(@as(usize, 0), parsed.options.guest_args.len);
    try std.testing.expectEqual(@as(u32, 60), parsed.options.max_runtime_seconds);
    try std.testing.expectEqual(DiffMode.stdout_only, parsed.options.diff_mode);
    try std.testing.expect(!parsed.options.strict_exit);
    try std.testing.expect(!parsed.options.keep_cwasm);
    try std.testing.expect(!parsed.options.json);
}

test "parse: --map-dir repeatable, space and = forms" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--map-dir", "/a::/x", "--map-dir=/b::/y" };
    var parsed = try parse(std.testing.allocator, &argv, &diag);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 2), parsed.options.map_dirs.len);
    try std.testing.expectEqualStrings("/a::/x", parsed.options.map_dirs[0]);
    try std.testing.expectEqualStrings("/b::/y", parsed.options.map_dirs[1]);
}

test "parse: --env repeatable" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--env", "K=V", "--env=X=Y" };
    var parsed = try parse(std.testing.allocator, &argv, &diag);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 2), parsed.options.env_vars.len);
    try std.testing.expectEqualStrings("K=V", parsed.options.env_vars[0]);
    try std.testing.expectEqualStrings("X=Y", parsed.options.env_vars[1]);
}

test "parse: -- guest args" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--", "a", "b", "--bogus" };
    var parsed = try parse(std.testing.allocator, &argv, &diag);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 3), parsed.options.guest_args.len);
    try std.testing.expectEqualStrings("a", parsed.options.guest_args[0]);
    try std.testing.expectEqualStrings("b", parsed.options.guest_args[1]);
    try std.testing.expectEqualStrings("--bogus", parsed.options.guest_args[2]);
}

test "parse: bool flags" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--strict-exit", "--keep-cwasm", "--json", "--stderr-only" };
    var parsed = try parse(std.testing.allocator, &argv, &diag);
    defer parsed.deinit();
    try std.testing.expect(parsed.options.strict_exit);
    try std.testing.expect(parsed.options.keep_cwasm);
    try std.testing.expect(parsed.options.json);
    try std.testing.expectEqual(DiffMode.stderr_only, parsed.options.diff_mode);
}

test "parse: --diff-everything overrides default" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--diff-everything" };
    var parsed = try parse(std.testing.allocator, &argv, &diag);
    defer parsed.deinit();
    try std.testing.expectEqual(DiffMode.both, parsed.options.diff_mode);
}

test "parse: --max-runtime and --hex-context parse u32" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--max-runtime", "120", "--hex-context=8" };
    var parsed = try parse(std.testing.allocator, &argv, &diag);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u32, 120), parsed.options.max_runtime_seconds);
    try std.testing.expectEqual(@as(u32, 8), parsed.options.hex_context);
}

test "parse: missing input path" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{"--strict-exit"};
    try std.testing.expectError(ParseError.MissingInputPath, parse(std.testing.allocator, &argv, &diag));
}

test "parse: duplicate input path" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "a.wasm", "b.wasm" };
    try std.testing.expectError(ParseError.DuplicateInputPath, parse(std.testing.allocator, &argv, &diag));
    try std.testing.expectEqualStrings("b.wasm", diag.value);
}

test "parse: unknown option populates diag" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--unknown-flag" };
    try std.testing.expectError(ParseError.UnknownOption, parse(std.testing.allocator, &argv, &diag));
    try std.testing.expectEqualStrings("--unknown-flag", diag.option);
}

test "parse: missing value populates diag" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--map-dir" };
    try std.testing.expectError(ParseError.MissingValue, parse(std.testing.allocator, &argv, &diag));
    try std.testing.expectEqualStrings("--map-dir", diag.option);
}

test "parse: invalid integer for --max-runtime populates diag" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--max-runtime", "abc" };
    try std.testing.expectError(ParseError.InvalidIntegerArgument, parse(std.testing.allocator, &argv, &diag));
    try std.testing.expectEqualStrings("--max-runtime", diag.option);
    try std.testing.expectEqualStrings("abc", diag.value);
}

test "parse: --wasmtime-bin and --wamr-bin overrides" {
    var diag: ParseDiagnostic = .{};
    const argv = [_][]const u8{ "foo.wasm", "--wasmtime-bin", "/opt/wt", "--wamr-bin=/opt/wamr" };
    var parsed = try parse(std.testing.allocator, &argv, &diag);
    defer parsed.deinit();
    try std.testing.expectEqualStrings("/opt/wt", parsed.options.wasmtime_bin.?);
    try std.testing.expectEqualStrings("/opt/wamr", parsed.options.wamr_bin.?);
}

test "buildWasmtimeArgs: shape" {
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(std.testing.allocator);
    const opts: Options = .{
        .wasm_path = "foo.wasm",
        .map_dirs = &.{"/h::/g"},
        .env_vars = &.{"K=V"},
        .guest_args = &.{ "a", "b" },
    };
    try buildWasmtimeArgs(std.testing.allocator, opts, "/usr/bin/wasmtime", &list);
    try std.testing.expectEqualSlices([]const u8, &.{
        "/usr/bin/wasmtime", "run",
        "--dir",             "/h::/g",
        "--env",             "K=V",
        "foo.wasm",          "--",
        "a",                 "b",
    }, list.items);
}

test "buildWasmtimeArgs: no guest args → no trailing --" {
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(std.testing.allocator);
    const opts: Options = .{ .wasm_path = "foo.wasm" };
    try buildWasmtimeArgs(std.testing.allocator, opts, "wasmtime", &list);
    try std.testing.expectEqualSlices([]const u8, &.{ "wasmtime", "run", "foo.wasm" }, list.items);
}

test "buildWamrArgs: shape with --map-dir not --dir" {
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(std.testing.allocator);
    const opts: Options = .{
        .wasm_path = "foo.wasm",
        .map_dirs = &.{"/h::/g"},
        .env_vars = &.{"K=V"},
    };
    try buildWamrArgs(std.testing.allocator, opts, "/usr/bin/wamr", &list);
    try std.testing.expectEqualSlices([]const u8, &.{
        "/usr/bin/wamr", "run",
        "--map-dir",     "/h::/g",
        "--env",         "K=V",
        "foo.wasm",
    }, list.items);
}

test "buildWamrArgs: guest args after --" {
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(std.testing.allocator);
    const opts: Options = .{
        .wasm_path = "foo.wasm",
        .guest_args = &.{ "x", "y" },
    };
    try buildWamrArgs(std.testing.allocator, opts, "wamr", &list);
    try std.testing.expectEqualSlices([]const u8, &.{
        "wamr", "run", "foo.wasm", "--", "x", "y",
    }, list.items);
}
