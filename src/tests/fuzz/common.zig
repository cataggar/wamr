//! Shared helpers for the fuzz harness CLIs.
//!
//! Each harness (loader, interp, aot, diff) links this module for
//! argument parsing, corpus iteration, and crash-artifact writing.
//!
//! Crash detection uses a sentinel-file pattern: before each
//! invocation we write the current input to `<crashes>/in-flight.wasm`.
//! If the process aborts (panic, SIGSEGV, stack overflow, etc.) the
//! file persists and the CI workflow picks it up as the offending
//! input. On clean exit we delete it.

const std = @import("std");

pub const Args = struct {
    corpus_dir: []const u8,
    crashes_dir: []const u8,
    duration_ms: u64,

    pub fn parse(argv: []const []const u8) !Args {
        var corpus: ?[]const u8 = null;
        var crashes: ?[]const u8 = null;
        var duration_s: u64 = 60;

        var i: usize = 1;
        while (i < argv.len) : (i += 1) {
            const a = argv[i];
            if (std.mem.eql(u8, a, "--corpus") and i + 1 < argv.len) {
                i += 1;
                corpus = argv[i];
            } else if (std.mem.eql(u8, a, "--crashes") and i + 1 < argv.len) {
                i += 1;
                crashes = argv[i];
            } else if (std.mem.eql(u8, a, "--duration") and i + 1 < argv.len) {
                i += 1;
                duration_s = try std.fmt.parseInt(u64, argv[i], 10);
            } else if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
                printUsage();
                std.process.exit(0);
            } else {
                std.log.err("unknown arg: {s}", .{a});
                printUsage();
                return error.BadArgs;
            }
        }

        return .{
            .corpus_dir = corpus orelse return error.MissingCorpus,
            .crashes_dir = crashes orelse return error.MissingCrashesDir,
            .duration_ms = duration_s * std.time.ms_per_s,
        };
    }

    fn printUsage() void {
        std.debug.print(
            \\usage: fuzz-<target> --corpus <dir> --crashes <dir> [--duration <seconds>]
            \\
            \\Replays every .wasm file under <corpus> through the target until
            \\<duration> seconds have elapsed. A crashing input is left at
            \\<crashes>/in-flight.wasm when the process aborts.
            \\
        , .{});
    }
};

pub const Corpus = struct {
    entries: std.ArrayList([]u8),
    arena: std.heap.ArenaAllocator,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Corpus {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();

        var list: std.ArrayList([]u8) = .empty;

        var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch |err| {
            std.log.err("cannot open corpus dir {s}: {}", .{ dir_path, err });
            return err;
        };
        defer dir.close(io);

        var it = dir.iterate();
        while (try it.next(io)) |e| {
            if (e.kind != .file) continue;
            if (!std.mem.endsWith(u8, e.name, ".wasm")) continue;
            const bytes = try dir.readFileAlloc(io, e.name, a, std.Io.Limit.limited(16 * 1024 * 1024));
            try list.append(a, bytes);
        }

        return .{ .entries = list, .arena = arena };
    }

    pub fn deinit(self: *Corpus) void {
        self.arena.deinit();
    }

    pub fn count(self: *const Corpus) usize {
        return self.entries.items.len;
    }

    pub fn get(self: *const Corpus, idx: usize) []const u8 {
        return self.entries.items[idx % self.entries.items.len];
    }
};

/// Writes the current in-flight input to `<crashes>/in-flight.wasm`.
/// Call before every target invocation so a subsequent process abort
/// leaves a reproducer behind.
pub fn markInFlight(io: std.Io, crashes_dir: []const u8, bytes: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    cwd.createDirPath(io, crashes_dir) catch {};
    var dir = try cwd.openDir(io, crashes_dir, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = "in-flight.wasm", .data = bytes });
}

/// Clear the in-flight marker after a clean invocation.
pub fn clearInFlight(io: std.Io, crashes_dir: []const u8) void {
    var dir = std.Io.Dir.cwd().openDir(io, crashes_dir, .{}) catch return;
    defer dir.close(io);
    dir.deleteFile(io, "in-flight.wasm") catch {};
}

/// Persist an input that returned a known error we want to escalate
/// (e.g. differential mismatch in fuzz-diff). Uses a SHA256 of the
/// bytes as the filename so duplicates collapse.
pub fn saveCrasher(io: std.Io, crashes_dir: []const u8, bytes: []const u8, tag: []const u8) !void {
    var hash: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    var name_buf: [128]u8 = undefined;
    var hex_buf: [16]u8 = undefined;
    _ = std.fmt.bufPrint(&hex_buf, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5], hash[6], hash[7],
    }) catch unreachable;
    const name = try std.fmt.bufPrint(&name_buf, "{s}-{s}.wasm", .{ tag, hex_buf[0..] });
    const cwd = std.Io.Dir.cwd();
    cwd.createDirPath(io, crashes_dir) catch {};
    var dir = try cwd.openDir(io, crashes_dir, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = name, .data = bytes });
}

/// Monotonic deadline built from a millisecond duration. Callers
/// create one before the loop and `expired(io)` in the condition.
pub const Deadline = struct {
    start: std.Io.Clock.Timestamp,
    duration_ms: u64,

    pub fn init(io: std.Io, duration_ms: u64) Deadline {
        return .{
            .start = std.Io.Clock.Timestamp.now(io, .awake),
            .duration_ms = duration_ms,
        };
    }

    pub fn expired(self: Deadline, io: std.Io) bool {
        const elapsed_ns = self.start.untilNow(io).raw.nanoseconds;
        const limit_ns: i96 = @as(i96, @intCast(self.duration_ms)) * std.time.ns_per_ms;
        return elapsed_ns >= limit_ns;
    }
};
