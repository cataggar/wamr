//! Select a performance idea without re-filing an attempted investigation.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

const Idea = struct {
    id: []const u8,
    title: []const u8,
    area: []const u8,
    description: []const u8,
    files: []const []const u8,
    reference: []const u8,
    bench_hint: []const u8,
};

fn titles(allocator: Allocator, source: []const u8, only_perf: bool) ![]const []const u8 {
    const root = try std.json.parseFromSliceLeaky(Value, allocator, source, .{});
    if (root != .array) return error.InvalidGitHubResponse;
    var result: std.ArrayList([]const u8) = .empty;
    for (root.array.items) |item| {
        if (item != .object) return error.InvalidGitHubResponse;
        const title = item.object.get("title") orelse return error.InvalidGitHubResponse;
        if (title != .string) return error.InvalidGitHubResponse;
        if (!only_perf or std.mem.startsWith(u8, title.string, "perf:"))
            try result.append(allocator, title.string);
    }
    return result.toOwnedSlice(allocator);
}

fn githubTitles(allocator: Allocator, io: std.Io, args: []const []const u8, only_perf: bool) ![]const []const u8 {
    const result = std.process.run(allocator, io, .{
        .argv = args,
        .stdout_limit = .limited(16 * 1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch |err| {
        std.debug.print("error: unable to run gh: {s}\n", .{@errorName(err)});
        return err;
    };
    if (!result.term.success()) {
        std.debug.print("error: gh {s} failed ({f}): {s}\n", .{ args[1], result.term, std.mem.trim(u8, result.stderr, " \t\r\n") });
        return error.GitHubLookupFailed;
    }
    return titles(allocator, result.stdout, only_perf);
}

fn select(allocator: Allocator, ideas: []const Idea, issues: []const []const u8, prs: []const []const u8, forced: ?usize) !?Idea {
    if (forced) |index| if (index < ideas.len) return ideas[index];
    for (ideas) |idea| {
        const title = try std.fmt.allocPrint(allocator, "perf: {s}", .{idea.title});
        defer allocator.free(title);
        var tried = false;
        for (issues) |item| if (std.mem.eql(u8, title, item)) {
            tried = true;
            break;
        };
        for (prs) |item| if (std.mem.eql(u8, title, item) or
            (std.mem.startsWith(u8, item, title) and item.len > title.len and item[title.len] == ' '))
        {
            tried = true;
            break;
        };
        if (!tried) return idea;
    }
    return null;
}

fn isSpace(codepoint: u21) bool {
    return switch (codepoint) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn strip(source: []const u8) ![]const u8 {
    var iter = (try std.unicode.Utf8View.init(source)).iterator();
    var first: ?usize = null;
    var last: usize = 0;
    while (iter.nextCodepointSlice()) |slice| {
        if (!isSpace(try std.unicode.utf8Decode(slice))) {
            if (first == null) first = @intFromPtr(slice.ptr) - @intFromPtr(source.ptr);
            last = @intFromPtr(slice.ptr) - @intFromPtr(source.ptr) + slice.len;
        }
    }
    return if (first) |start| source[start..last] else "";
}

fn prefixCharacters(source: []const u8, limit: usize) ![]const u8 {
    var iter = (try std.unicode.Utf8View.init(source)).iterator();
    var end: usize = 0;
    var count: usize = 0;
    while (count < limit) : (count += 1) {
        const item = iter.nextCodepointSlice() orelse break;
        end = @intFromPtr(item.ptr) - @intFromPtr(source.ptr) + item.len;
    }
    return source[0..end];
}

fn normalizeNewlines(allocator: Allocator, source: []const u8) ![]const u8 {
    if (std.mem.indexOfScalar(u8, source, '\r') == null) return source;
    var output: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        if (source[i] == '\r') {
            try output.append(allocator, '\n');
            if (i + 1 < source.len and source[i + 1] == '\n') i += 1;
        } else {
            try output.append(allocator, source[i]);
        }
    }
    return output.toOwnedSlice(allocator);
}

fn baseline(allocator: Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const source = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return "(not produced)",
        else => return err,
    };
    return strip(try normalizeNewlines(allocator, source));
}

fn body(allocator: Allocator, idea: Idea, bench: []const u8, spec: []const u8) ![]const u8 {
    var file_list: std.Io.Writer.Allocating = .init(allocator);
    for (idea.files, 0..) |name, index| {
        if (index != 0) try file_list.writer.writeAll(", ");
        try file_list.writer.print("`{s}`", .{name});
    }
    return std.fmt.allocPrint(allocator,
        \\## Performance Investigation: {s}
        \\
        \\**Area:** {s}
        \\**Suggested files:** {s}
        \\
        \\### Current baseline
        \\
        \\<details><summary>Codegen bench (cycles/op, code bytes)</summary>
        \\
        \\```
        \\{s}
        \\```
        \\
        \\</details>
        \\
        \\<details><summary>Spec-tests AOT pass rate</summary>
        \\
        \\```
        \\{s}
        \\```
        \\
        \\</details>
        \\
        \\### What to do
        \\
        \\{s}
        \\
        \\### Benchmark tip
        \\
        \\{s}
        \\
        \\### Reference
        \\
        \\{s}
        \\
        \\### Acceptance criteria
        \\
        \\1. **`zig build test`** — all existing tests pass
        \\2. **`zig build bench`** — improvement or no regression in cycles/op and code size
        \\3. **`zig build spec-tests-aot`** — pass rate unchanged or improved
        \\4. Add a unit test or benchmark body for the new optimisation if applicable
        \\
        \\### How to verify
        \\
        \\```bash
        \\zig build -Doptimize=ReleaseFast
        \\zig build bench            # compare cycles/op and code size
        \\zig build spec-tests-aot   # verify pass rate
        \\zig build test             # ensure no regressions
        \\```
        \\
    , .{ idea.title, idea.area, file_list.written(), try prefixCharacters(bench, 4000), spec, idea.description, idea.bench_hint, idea.reference });
}

fn appendOutput(io: std.Io, path: ?[]const u8, contents: []const u8) !void {
    const output_path = path orelse return;
    if (output_path.len == 0) return;
    const file = try std.Io.Dir.cwd().createFile(io, output_path, .{ .truncate = false });
    defer file.close(io);
    try file.writePositionalAll(io, contents, (try file.stat(io)).size);
}

fn run(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const ideas = try std.json.parseFromSliceLeaky([]const Idea, allocator, @embedFile("perf_ideas.json"), .{});
    const issues = try githubTitles(allocator, io, &.{ "gh", "issue", "list", "--label", "perf-investigation", "--state", "all", "--json", "title", "--limit", "200" }, false);
    const prs = try githubTitles(allocator, io, &.{ "gh", "pr", "list", "--state", "merged", "--search", "perf in:title", "--json", "title", "--limit", "200" }, true);
    const raw = init.environ_map.get("IDEA_INDEX") orelse "auto";
    const index = std.fmt.parseInt(usize, std.mem.trim(u8, raw, " \t\r\n"), 10) catch null;
    const chosen = try select(allocator, ideas, issues, prs, index);
    const output = init.environ_map.get("GITHUB_OUTPUT");
    if (chosen) |idea| {
        const title = try std.fmt.allocPrint(allocator, "perf: {s}", .{idea.title});
        const report = try body(allocator, idea, try baseline(allocator, io, "bench-baseline.txt"), try baseline(allocator, io, "spec-baseline.txt"));
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "issue-body.md", .data = report });
        try appendOutput(io, output, try std.fmt.allocPrint(allocator, "skip=false\ntitle={s}\nidea_id={s}\n", .{ title, idea.id }));
        try std.Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(allocator, "Selected: {s} — {s}\n", .{ idea.id, title }));
    } else {
        try appendOutput(io, output, "skip=true\n");
        try std.Io.File.stdout().writeStreamingAll(io, "All curated ideas already investigated — skipping issue creation.\n");
    }
}

pub fn main(init: std.process.Init) !void {
    run(init) catch |err| {
        std.debug.print("error: performance investigation: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

test "selection rejects existing issues and exact or suffixed merged pull requests" {
    const allocator = std.testing.allocator;
    const ideas = [_]Idea{
        .{ .id = "a", .title = "first", .area = "", .description = "", .files = &.{}, .reference = "", .bench_hint = "" },
        .{ .id = "b", .title = "second", .area = "", .description = "", .files = &.{}, .reference = "", .bench_hint = "" },
    };
    try std.testing.expectEqualStrings("b", (try select(allocator, &ideas, &.{"perf: first"}, &.{}, null)).?.id);
    try std.testing.expectEqualStrings("a", (try select(allocator, &ideas, &.{"perf: first"}, &.{}, 0)).?.id);
    try std.testing.expect((try select(allocator, &ideas, &.{"perf: first"}, &.{"perf: second (#7)"}, 100)) == null);
    try std.testing.expectEqualStrings("b", (try select(allocator, &ideas, &.{"perf: first"}, &.{"perf: secondly"}, null)).?.id);
}

test "malformed GitHub responses are rejected" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    try std.testing.expectError(error.InvalidGitHubResponse, titles(allocator, "{}", false));
    try std.testing.expectError(error.InvalidGitHubResponse, titles(allocator, "[{}]", false));
    const result = try titles(allocator, "[{\"title\":\"other\"},{\"title\":\"perf: fixed\"}]", true);
    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqualStrings("perf: fixed", result[0]);
}

test "failed GitHub lookup never yields an empty tried set" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.GitHubLookupFailed, githubTitles(arena.allocator(), std.testing.io, &.{ "/bin/sh", "-c", "echo unavailable >&2; exit 9" }, false));
}

test "baseline text uses Unicode strip and character-count truncation" {
    try std.testing.expectEqualStrings("éx", try strip("\u{a0}éx\u{2003}"));
    try std.testing.expectEqualStrings("é", try prefixCharacters("éx", 1));
    const normalized = try normalizeNewlines(std.testing.allocator, "a\r\nb\rc\n");
    defer std.testing.allocator.free(normalized);
    try std.testing.expectEqualStrings("a\nb\nc\n", normalized);
}

test "workflow output appends without replacing earlier step values" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const path = try std.fs.path.join(allocator, &.{ root, "output" });
    defer allocator.free(path);
    try appendOutput(io, path, "existing=true\n");
    try appendOutput(io, path, "skip=false\ntitle=perf: example\nidea_id=test\n");
    const contents = try temp.dir.readFileAlloc(io, "output", allocator, .unlimited);
    defer allocator.free(contents);
    try std.testing.expectEqualStrings("existing=true\nskip=false\ntitle=perf: example\nidea_id=test\n", contents);
}
