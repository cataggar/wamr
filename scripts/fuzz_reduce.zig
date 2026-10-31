//! Reduce a reproducible fuzz input with the matching native harness.
const std = @import("std");
const temp_root = @import("temp_root.zig");
const Allocator = std.mem.Allocator;
const Dir = std.Io.Dir;
const predicate_environment = "WAMR_FUZZ_REDUCE_PREDICATE";

const Options = struct {
    root: []const u8 = ".",
    target: ?[]const u8 = null,
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    harness_dir: ?[]const u8 = null,
    duration: i64 = 2,
    max_passes: i64 = 4,
    extra: []const []const u8 = &.{},
    external: bool = true,
    predicate_mode: bool = false,
    harness: ?[]const u8 = null,
};

fn parseOptions(allocator: Allocator, args: []const []const u8) !Options {
    var options: Options = .{};
    var extras: std.ArrayList([]const u8) = .empty;
    var index: usize = 0;
    var positional_only = false;
    while (index < args.len) : (index += 1) {
        const raw = args[index];
        if (!positional_only and std.mem.eql(u8, raw, "--")) {
            positional_only = true;
            continue;
        }
        if (!positional_only and (std.mem.eql(u8, raw, "--help") or std.mem.eql(u8, raw, "-h"))) return error.HelpRequested;
        if (!positional_only and std.mem.eql(u8, raw, "--no-external-shrink")) {
            options.external = false;
            continue;
        }
        if (!positional_only and std.mem.eql(u8, raw, "--predicate-mode")) {
            options.predicate_mode = true;
            continue;
        }
        if (!positional_only and std.mem.startsWith(u8, raw, "--")) {
            const equals = std.mem.indexOfScalar(u8, raw, '=');
            const key = if (equals) |pos| raw[0..pos] else raw;
            const value = if (equals) |pos| raw[pos + 1 ..] else blk: {
                index += 1;
                if (index >= args.len) return error.MissingOptionValue;
                break :blk args[index];
            };
            if (std.mem.eql(u8, key, "--repo-root")) {
                options.root = value;
            } else if (std.mem.eql(u8, key, "--out")) {
                options.output = value;
            } else if (std.mem.eql(u8, key, "--harness-dir")) {
                options.harness_dir = value;
            } else if (std.mem.eql(u8, key, "--harness")) {
                options.harness = value;
            } else if (std.mem.eql(u8, key, "--duration")) {
                options.duration = try std.fmt.parseInt(i64, value, 10);
            } else if (std.mem.eql(u8, key, "--max-passes")) {
                options.max_passes = try std.fmt.parseInt(i64, value, 10);
            } else if (std.mem.eql(u8, key, "--extra")) {
                try extras.append(allocator, value);
            } else return error.UnknownOption;
        } else if (options.target == null) {
            options.target = raw;
        } else if (options.input == null) {
            options.input = raw;
        } else return error.UnexpectedArgument;
    }
    if (options.predicate_mode) {
        if (options.harness == null or options.target == null or options.input != null) return error.MissingPredicateInput;
    } else if (options.target == null or options.input == null) return error.MissingInput;
    options.extra = try extras.toOwnedSlice(allocator);
    return options;
}

const Predicate = struct {
    harness: []const u8,
    duration: i64,
    extra: []const []const u8,
    temp_root: []const u8,
};

const Scratch = struct {
    parent: Dir,
    dir: Dir,
    path: []const u8,
    name: []const u8,

    fn create(allocator: Allocator, io: std.Io, root: []const u8) !Scratch {
        const parent = try Dir.cwd().openDir(io, root, .{});
        errdefer parent.close(io);
        for (0..10) |_| {
            var random: [16]u8 = undefined;
            io.random(&random);
            const name = try std.fmt.allocPrint(allocator, "fuzz-reduce-{s}", .{std.fmt.bytesToHex(random, .lower)});
            parent.createDir(io, name, if (@import("builtin").os.tag == .windows) .default_dir else .fromMode(0o700)) catch |err| switch (err) {
                error.PathAlreadyExists => continue,
                else => return err,
            };
            errdefer parent.deleteTree(io, name) catch |err| std.debug.print("error removing temporary directory: {s}\n", .{@errorName(err)});
            const dir = try parent.openDir(io, name, .{ .iterate = true });
            errdefer dir.close(io);
            return .{ .parent = parent, .dir = dir, .name = name, .path = try std.fs.path.join(allocator, &.{ root, name }) };
        }
        return error.TemporaryNameCollisions;
    }

    fn destroy(self: Scratch, io: std.Io) !void {
        self.dir.close(io);
        defer self.parent.close(io);
        try self.parent.deleteTree(io, self.name);
    }
};

fn reproduces(allocator: Allocator, io: std.Io, predicate: Predicate, input: []const u8) !bool {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const scratch = try Scratch.create(gpa, io, predicate.temp_root);
    var pending = true;
    defer if (pending) scratch.destroy(io) catch |err| std.debug.print("error removing temporary directory {s}: {s}\n", .{ scratch.path, @errorName(err) });
    const result = try reproduceInScratch(gpa, io, predicate, input, scratch);
    pending = false;
    scratch.destroy(io) catch |err| {
        std.debug.print("error removing temporary directory {s}: {s}\n", .{ scratch.path, @errorName(err) });
        return err;
    };
    return result;
}

fn reproduceInScratch(gpa: Allocator, io: std.Io, predicate: Predicate, input: []const u8, scratch: Scratch) !bool {
    try scratch.dir.createDirPath(io, "corpus");
    try scratch.dir.createDirPath(io, "crashes");
    try scratch.dir.writeFile(io, .{ .sub_path = "corpus/candidate.wasm", .data = input });
    const corpus = try std.fs.path.join(gpa, &.{ scratch.path, "corpus" });
    const crashes = try std.fs.path.join(gpa, &.{ scratch.path, "crashes" });
    var args: std.ArrayList([]const u8) = .empty;
    try args.appendSlice(gpa, &.{ predicate.harness, "--corpus", corpus, "--crashes", crashes, "--duration", try std.fmt.allocPrint(gpa, "{d}", .{predicate.duration}) });
    try args.appendSlice(gpa, predicate.extra);
    var child = try std.process.spawn(io, .{ .argv = args.items, .stdout = .ignore, .stderr = .ignore });
    defer child.kill(io);
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return true;
    const crash_dir = try scratch.dir.openDir(io, "crashes", .{ .iterate = true });
    defer crash_dir.close(io);
    var iterator = crash_dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (std.mem.startsWith(u8, entry.name, "diff-mismatch-") and std.mem.endsWith(u8, entry.name, ".wasm"))
            return true;
    }
    return false;
}

fn shrink(allocator: Allocator, io: std.Io, predicate: Predicate, initial: []const u8, max_passes: i64, progress: bool) ![]u8 {
    var best = try allocator.dupe(u8, initial);
    errdefer allocator.free(best);
    var pass: i64 = 0;
    while (pass < max_passes) : (pass += 1) {
        var improved = false;
        var chunk = @max(1, best.len / 2);
        while (chunk > 0) : (chunk /= 2) {
            var offset: usize = 0;
            while (offset < best.len) {
                const end = @min(offset + chunk, best.len);
                const candidate = try allocator.alloc(u8, best.len - (end - offset));
                var retained = false;
                defer if (!retained) allocator.free(candidate);
                @memcpy(candidate[0..offset], best[0..offset]);
                @memcpy(candidate[offset..], best[end..]);
                const accepted = candidate.len != 0 and try reproduces(allocator, io, predicate, candidate);
                if (accepted) {
                    if (progress) std.debug.print("  shrink chunk={d} offset={d}: {d} -> {d} bytes\n", .{ chunk, offset, best.len, candidate.len });
                    allocator.free(best);
                    best = candidate;
                    retained = true;
                    improved = true;
                } else {
                    offset += chunk;
                }
            }
        }
        if (!improved) break;
    }
    return best;
}

fn suffixPath(allocator: Allocator, path: []const u8, suffix: []const u8) ![]const u8 {
    const basename = std.fs.path.basename(path);
    const stem = if (std.mem.endsWith(u8, basename, ".")) basename else std.fs.path.stem(basename);
    const file = try std.fmt.allocPrint(allocator, "{s}{s}", .{ stem, suffix });
    return if (std.fs.path.dirname(path)) |parent| std.fs.path.join(allocator, &.{ parent, file }) else file;
}

fn externalShrink(init: std.process.Init, predicate: Predicate, input: []const u8, output: []const u8) !bool {
    const allocator = init.arena.allocator();
    const io = init.io;
    const cwd = Dir.cwd();
    var environment = try init.environ_map.clone(allocator);
    defer environment.deinit();
    try environment.put(predicate_environment, try std.json.Stringify.valueAlloc(allocator, predicate, .{}));
    const executable = try std.process.executablePathAlloc(io, allocator);
    var child = std.process.spawn(io, .{
        .argv = &.{ "wabt", "module", "shrink", executable, input, "--output", output },
        .environ_map = &environment,
    }) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer child.kill(io);
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0)
        std.debug.print("wabt module shrink exited with status {f}\n", .{term});
    cwd.access(io, output, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    return true;
}

fn mainRun(init: std.process.Init, options: Options) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const cwd = Dir.cwd();
    const root = try cwd.realPathFileAlloc(io, options.root, allocator);
    var cwd_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_path = cwd_buffer[0..try cwd.realPath(io, &cwd_buffer)];
    if (options.predicate_mode) {
        const harness = try std.fs.path.resolve(allocator, &.{ cwd_path, options.harness.? });
        const predicate: Predicate = .{ .harness = harness, .duration = options.duration, .extra = options.extra, .temp_root = try temp_root.select(allocator, io, init.environ_map) };
        const candidate = try cwd.readFileAlloc(io, options.target.?, allocator, .unlimited);
        if (!try reproduces(init.gpa, io, predicate, candidate)) std.process.exit(1);
        return;
    }
    const harness_dir = options.harness_dir orelse try std.fs.path.join(allocator, &.{ root, "zig-out", "bin" });
    const harness_name = try std.fmt.allocPrint(allocator, "fuzz-{s}{s}", .{ options.target.?, if (@import("builtin").os.tag == .windows) ".exe" else "" });
    const harness = try std.fs.path.resolve(allocator, &.{ cwd_path, harness_dir, harness_name });
    const temporary = try temp_root.select(allocator, io, init.environ_map);
    const predicate: Predicate = .{ .harness = harness, .duration = options.duration, .extra = options.extra, .temp_root = temporary };
    const initial = try cwd.readFileAlloc(io, options.input.?, allocator, .unlimited);
    std.debug.print("verifying initial reproduction ({d} bytes)...\n", .{initial.len});
    if (!try reproduces(init.gpa, io, predicate, initial)) return error.InitialInputDoesNotReproduce;
    const output = options.output orelse try suffixPath(allocator, options.input.?, ".reduced.wasm");
    var best = initial;
    if (options.external) {
        const external_dir = try Scratch.create(allocator, io, temporary);
        var pending = true;
        defer if (pending) external_dir.destroy(io) catch |err| std.debug.print("error removing temporary directory {s}: {s}\n", .{ external_dir.path, @errorName(err) });
        const external_output = try std.fs.path.join(allocator, &.{ external_dir.path, "external.wasm" });
        if (try externalShrink(init, predicate, options.input.?, external_output)) {
            const external_bytes = try cwd.readFileAlloc(io, external_output, allocator, .unlimited);
            if (external_bytes.len < best.len and try reproduces(init.gpa, io, predicate, external_bytes)) best = external_bytes;
        }
        pending = false;
        external_dir.destroy(io) catch |err| {
            std.debug.print("error removing temporary directory {s}: {s}\n", .{ external_dir.path, @errorName(err) });
            return err;
        };
    }
    std.debug.print("running byte-level shrinker...\n", .{});
    const reduced = try shrink(init.gpa, io, predicate, best, options.max_passes, true);
    defer init.gpa.free(reduced);
    try cwd.writeFile(io, .{ .sub_path = output, .data = reduced });
    std.debug.print("done: {d} -> {d} bytes ({d}% of original)\n", .{ initial.len, reduced.len, reduced.len * 100 / @max(1, initial.len) });
    try std.Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(allocator, "{s}\n", .{output}));
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (init.environ_map.get(predicate_environment)) |raw| {
        if (args.len != 2) return error.InvalidPredicateArguments;
        const predicate = try std.json.parseFromSliceLeaky(Predicate, allocator, raw, .{});
        const candidate = try Dir.cwd().readFileAlloc(init.io, args[1], allocator, .unlimited);
        if (!try reproduces(init.gpa, init.io, predicate, candidate)) std.process.exit(1);
        return;
    }
    const options = parseOptions(allocator, args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            try std.Io.File.stdout().writeStreamingAll(init.io, "usage: fuzz-reduce TARGET INPUT [--out FILE] [--duration N] [--extra ARG] [--max-passes N] [--no-external-shrink]\n");
            return;
        }
        std.debug.print("fuzz-reduce: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    mainRun(init, options) catch |err| {
        std.debug.print("fuzz-reduce: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

test "options preserve extra arguments and reject missing inputs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const options = try parseOptions(arena.allocator(), &.{ "diff", "in.wasm", "--extra", "--fuel", "--extra=100", "--duration", "3", "--no-external-shrink" });
    try std.testing.expectEqualStrings("diff", options.target.?);
    try std.testing.expectEqualDeep(&[_][]const u8{ "--fuel", "100" }, options.extra);
    try std.testing.expectEqual(@as(i64, 3), options.duration);
    try std.testing.expect(!options.external);
    try std.testing.expectError(error.MissingInput, parseOptions(arena.allocator(), &.{"diff"}));
    try std.testing.expectError(error.MissingOptionValue, parseOptions(arena.allocator(), &.{ "diff", "in", "--out" }));
    const predicate = try parseOptions(arena.allocator(), &.{ "--predicate-mode", "--harness", "harness", "--", "-candidate" });
    try std.testing.expect(predicate.predicate_mode);
    try std.testing.expectEqualStrings("-candidate", predicate.target.?);
    try std.testing.expectEqualStrings("harness", predicate.harness.?);
    try std.testing.expectEqualStrings("input.reduced.wasm", try suffixPath(arena.allocator(), "input.wasm", ".reduced.wasm"));
    try std.testing.expectEqualStrings(".input.reduced.wasm", try suffixPath(arena.allocator(), ".input", ".reduced.wasm"));
    try std.testing.expectEqualStrings("input..reduced.wasm", try suffixPath(arena.allocator(), "input.", ".reduced.wasm"));
}

test "reduction handles crash exits and named crashers without accepting empty input" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{
        .sub_path = "fuzz-fake",
        .data =
        \\#!/bin/sh
        \\if grep -q X "$2/candidate.wasm"; then
        \\  if [ "$7" = "--named" ]; then
        \\    printf X > "$4/diff-mismatch-test.wasm"
        \\    exit 0
        \\  fi
        \\  exit 1
        \\fi
        \\exit 0
        \\
        ,
        .flags = .{ .permissions = if (@import("builtin").os.tag == .windows) .default_file else .fromMode(0o700) },
    });
    var buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = buffer[0..try tmp.dir.realPath(io, &buffer)];
    const harness = try std.fs.path.join(allocator, &.{ path, "fuzz-fake" });
    defer allocator.free(harness);
    var predicate: Predicate = .{ .harness = harness, .duration = 2, .extra = &.{}, .temp_root = path };
    try std.testing.expect(try reproduces(allocator, io, predicate, "abcXdef"));
    try std.testing.expect(!try reproduces(allocator, io, predicate, "abcdef"));
    const reduced = try shrink(allocator, io, predicate, "abcXdef", 4, false);
    defer allocator.free(reduced);
    try std.testing.expectEqualStrings("X", reduced);
    predicate.extra = &.{"--named"};
    try std.testing.expect(try reproduces(allocator, io, predicate, "X"));
    const unchanged = try shrink(allocator, io, predicate, "abcXdef", 0, false);
    defer allocator.free(unchanged);
    try std.testing.expectEqualStrings("abcXdef", unchanged);
}
