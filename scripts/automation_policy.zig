const std = @import("std");

const Allocator = std.mem.Allocator;
const baseline_path = "scripts/automation-policy-baseline.txt";

fn isPythonFile(path: []const u8) bool {
    for ([_][]const u8{ ".py", ".pyw", ".pyi", ".ipynb" }) |ext| {
        if (std.ascii.endsWithIgnoreCase(path, ext)) return true;
    }
    return false;
}

fn isAutomation(path: []const u8) bool {
    if (std.mem.eql(u8, path, "scripts/automation_policy.zig")) return false;
    if (std.mem.eql(u8, path, "build.zig")) return true;
    if (std.mem.startsWith(u8, path, "build/") and std.mem.endsWith(u8, path, ".zig")) return true;
    if (std.mem.startsWith(u8, path, "scripts/") and std.mem.endsWith(u8, path, ".zig")) return true;
    if (std.mem.startsWith(u8, path, ".github/scripts/") and std.mem.endsWith(u8, path, ".zig")) return true;
    if ((std.mem.startsWith(u8, path, "src/tests/") or std.mem.startsWith(u8, path, "src/bench/")) and
        std.mem.endsWith(u8, path, ".zig")) return true;
    if (std.mem.startsWith(u8, path, "tests/") and std.mem.endsWith(u8, path, ".zig")) return true;
    if (std.mem.startsWith(u8, path, ".github/workflows/") or
        std.mem.startsWith(u8, path, ".github/actions/"))
    {
        if (std.mem.endsWith(u8, path, ".yml") or std.mem.endsWith(u8, path, ".yaml")) return true;
    }
    if (std.mem.startsWith(u8, path, ".github/actions/") or
        std.mem.startsWith(u8, path, ".github/scripts/") or
        std.mem.startsWith(u8, path, "scripts/"))
    {
        for ([_][]const u8{ ".js", ".mjs", ".cjs", ".ts" }) |ext| {
            if (std.ascii.endsWithIgnoreCase(path, ext)) return true;
        }
    }
    for ([_][]const u8{ ".sh", ".bash", ".ps1", ".bat", ".cmd", ".cmake" }) |ext| {
        if (std.ascii.endsWithIgnoreCase(path, ext)) return true;
    }
    const basename = std.fs.path.basename(path);
    return std.mem.eql(u8, basename, "Makefile") or
        std.mem.eql(u8, basename, "Dockerfile") or
        std.mem.eql(u8, basename, "CMakeLists.txt");
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

fn containsCommand(line: []const u8, command: []const u8) bool {
    if (command.len > line.len) return false;
    for (0..line.len - command.len + 1) |i| {
        if (!std.ascii.eqlIgnoreCase(line[i..][0..command.len], command)) continue;
        if (i > 0 and (std.ascii.isAlphanumeric(line[i - 1]) or line[i - 1] == '_')) continue;
        var end = i + command.len;
        while (end < line.len and (std.ascii.isDigit(line[end]) or line[end] == '.')) : (end += 1) {}
        if (end < line.len and (std.ascii.isAlphanumeric(line[end]) or line[end] == '_')) continue;
        return true;
    }
    return false;
}

fn isPythonReference(line: []const u8) bool {
    const trimmed = std.mem.trim(u8, line, " \t\r");
    if (trimmed.len == 0 or trimmed[0] == '#' or std.mem.startsWith(u8, trimmed, "//")) return false;
    return containsIgnoreCase(trimmed, "python") or
        containsCommand(trimmed, "pip") or
        containsCommand(trimmed, "pipx") or
        containsCommand(trimmed, "pipenv") or
        containsCommand(trimmed, "py") or
        containsCommand(trimmed, "pypy") or
        containsCommand(trimmed, "uvx") or
        containsCommand(trimmed, "rye") or
        containsCommand(trimmed, "pytest") or
        containsCommand(trimmed, "tox") or
        containsCommand(trimmed, "nox") or
        containsCommand(trimmed, "poetry") or
        containsIgnoreCase(trimmed, "uv run") or
        containsIgnoreCase(trimmed, "uv pip") or
        containsIgnoreCase(trimmed, "uv tool") or
        containsIgnoreCase(trimmed, ".py");
}

fn isPythonShebang(contents: []const u8) bool {
    const first = contents[0 .. std.mem.indexOfScalar(u8, contents, '\n') orelse contents.len];
    return std.mem.startsWith(u8, first, "#!") and
        (containsIgnoreCase(first, "python") or containsIgnoreCase(first, "pypy"));
}

fn readPrefix(io: std.Io, path: []const u8, buffer: []u8) !?[]const u8 {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound, error.IsDir => return null,
        else => return err,
    };
    defer file.close(io);
    const len = file.readPositionalAll(io, buffer, 0) catch |err| switch (err) {
        error.IsDir => return null,
        else => return err,
    };
    const prefix = buffer[0..len];
    if (std.mem.startsWith(u8, prefix, "#!") and len == buffer.len and
        std.mem.indexOfScalar(u8, prefix, '\n') == null) return error.ShebangTooLong;
    return prefix;
}

fn addFinding(allocator: Allocator, findings: *std.ArrayList([]const u8), kind: []const u8, path: []const u8, detail: []const u8) !void {
    const finding = if (detail.len == 0)
        try std.fmt.allocPrint(allocator, "{s}\t{s}", .{ kind, path })
    else
        try std.fmt.allocPrint(allocator, "{s}\t{s}\t{s}", .{ kind, path, detail });
    try findings.append(allocator, finding);
}

fn collect(allocator: Allocator, io: std.Io) !std.ArrayList([]const u8) {
    var findings: std.ArrayList([]const u8) = .empty;
    const files = try std.process.run(allocator, io, .{
        .argv = &.{ "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z" },
        .stdout_limit = .limited(16 * 1024 * 1024),
    });
    if (files.term != .exited or files.term.exited != 0) {
        std.debug.print("git ls-files failed: {s}\n", .{files.stderr});
        return error.GitFailed;
    }
    var paths = std.mem.splitScalar(u8, files.stdout, 0);
    while (paths.next()) |path| {
        if (path.len == 0) continue;
        const python_file = isPythonFile(path);
        var buffer: [4096]u8 = undefined;
        const prefix = (try readPrefix(io, path, &buffer)) orelse continue;
        const automation = isAutomation(path) or
            (!python_file and std.mem.startsWith(u8, prefix, "#!"));
        if (python_file) {
            try addFinding(allocator, &findings, "file", path, "");
        } else if (isPythonShebang(prefix)) {
            try addFinding(allocator, &findings, "shebang", path, "");
        }
        if (automation) {
            const contents = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(8 * 1024 * 1024));
            var lines = std.mem.splitScalar(u8, contents, '\n');
            while (lines.next()) |line| {
                if (isPythonReference(line)) {
                    try addFinding(allocator, &findings, "reference", path, std.mem.trim(u8, line, " \t\r"));
                }
            }
        }
    }
    std.mem.sort([]const u8, findings.items, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.less);
    return findings;
}

fn check(allocator: Allocator, findings: []const []const u8, baseline: []const u8) !bool {
    var counts: std.StringHashMap(usize) = .init(allocator);
    defer counts.deinit();
    var lines = std.mem.splitScalar(u8, baseline, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        const entry = try counts.getOrPut(line);
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }
    var ok = true;
    for (findings) |finding| {
        if (counts.getPtr(finding)) |count| {
            if (count.* != 0) {
                count.* -= 1;
                continue;
            }
        }
        std.debug.print("new Python usage: {s}\n", .{finding});
        ok = false;
    }
    var entries = counts.iterator();
    while (entries.next()) |entry| {
        if (entry.value_ptr.* == 0) continue;
        std.debug.print("stale policy baseline entry: {s} ({d} remaining)\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        ok = false;
    }
    return ok;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len > 2 or (args.len == 2 and !std.mem.eql(u8, args[1], "--write-baseline"))) {
        std.debug.print("usage: automation-policy [--write-baseline]\n", .{});
        return error.InvalidArguments;
    }
    const findings = try collect(allocator, init.io);
    if (args.len == 2) {
        var result: std.ArrayList(u8) = .empty;
        try result.appendSlice(allocator, "# Existing Python usage; remove entries as the migration proceeds.\n");
        for (findings.items) |finding| {
            try result.appendSlice(allocator, finding);
            try result.append(allocator, '\n');
        }
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = baseline_path, .data = result.items });
        return;
    }
    const baseline = try std.Io.Dir.cwd().readFileAlloc(init.io, baseline_path, allocator, .limited(4 * 1024 * 1024));
    if (!try check(allocator, findings.items, baseline)) std.process.exit(1);
}

test "recognize Python files and automation references" {
    try std.testing.expect(isPythonFile("scripts/example.PY"));
    try std.testing.expect(isPythonFile("example.ipynb"));
    try std.testing.expect(!isPythonFile("example.py.txt"));
    try std.testing.expect(isAutomation(".github/workflows/ci.yml"));
    try std.testing.expect(isAutomation("build/native_jit.zig"));
    try std.testing.expect(isAutomation("src/tests/runner.zig"));
    try std.testing.expect(isAutomation(".github/actions/action/index.js"));
    try std.testing.expect(!isAutomation("docs/example.md"));
    try std.testing.expect(isPythonReference("run: python3 scripts/example.py"));
    try std.testing.expect(isPythonReference("uses: actions/setup-python@v5"));
    try std.testing.expect(isPythonReference("pip install -r requirements.txt"));
    try std.testing.expect(isPythonReference("pip3 install -r requirements.txt"));
    try std.testing.expect(isPythonReference("pypy3 script"));
    try std.testing.expect(isPythonReference("py -3 script"));
    try std.testing.expect(isPythonReference("uv run tool"));
    try std.testing.expect(isPythonReference("uv tool install example"));
    try std.testing.expect(isPythonReference("pipenv run test"));
    try std.testing.expect(!isPythonReference("pipeline: enabled"));
    try std.testing.expect(!isPythonReference("# Python is mentioned in a comment"));
    try std.testing.expect(!isPythonReference("// Python is mentioned in a comment"));
    try std.testing.expect(isPythonShebang("#!/usr/bin/env python3\n"));
    try std.testing.expect(isPythonShebang("#!/usr/bin/pypy3\n"));
    try std.testing.expect(!isPythonShebang("#!/bin/sh\n# python\n"));
}

test "baseline is a multiset and does not permit new references" {
    const baseline = "reference\tci.yml\trun: python3 test.py\n";
    const one = [_][]const u8{"reference\tci.yml\trun: python3 test.py"};
    const two = [_][]const u8{ one[0], one[0] };
    try std.testing.expect(try check(std.testing.allocator, &one, baseline));
    try std.testing.expect(!try check(std.testing.allocator, &two, baseline));
    try std.testing.expect(!try check(std.testing.allocator, &.{}, baseline));
}
