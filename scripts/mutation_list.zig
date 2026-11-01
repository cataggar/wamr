//! Experimental read-only mutation candidate listing.
const std = @import("std");
const candidates = @import("mutation_candidates.zig");
const shuffle = @import("mutation_shuffle.zig");
const report = @import("mutation_report.zig");
const Allocator = std.mem.Allocator;

pub const Options = struct {
    target: []const u8,
    seed: ?[]const u32,
    limit: ?Limit,
    ranges: []const candidates.Range,
};

pub const Limit = struct {
    label: []const u8,
    magnitude: usize,
    negative: bool,
};

fn parseLimit(allocator: Allocator, raw: []const u8) !Limit {
    var text = std.mem.trim(u8, raw, " \t\r\n\x0b\x0c");
    const negative = text.len != 0 and text[0] == '-';
    if (text.len != 0 and (text[0] == '-' or text[0] == '+')) text = text[1..];
    if (text.len == 0) return error.InvalidLimit;
    var digits: std.ArrayList(u8) = .empty;
    var underscore = false;
    var magnitude: usize = 0;
    for (text) |ch| {
        if (ch == '_') {
            if (digits.items.len == 0 or underscore) return error.InvalidLimit;
            underscore = true;
            continue;
        }
        if (!std.ascii.isDigit(ch)) return error.InvalidLimit;
        underscore = false;
        try digits.append(allocator, ch);
        magnitude = std.math.mul(usize, magnitude, 10) catch std.math.maxInt(usize);
        magnitude = std.math.add(usize, magnitude, ch - '0') catch std.math.maxInt(usize);
    }
    if (underscore) return error.InvalidLimit;
    var first: usize = 0;
    while (first + 1 < digits.items.len and digits.items[first] == '0') : (first += 1) {}
    const is_negative = negative and magnitude != 0;
    return .{
        .label = try std.fmt.allocPrint(allocator, "{s}{s}", .{ if (is_negative) "-" else "", digits.items[first..] }),
        .magnitude = magnitude,
        .negative = is_negative,
    };
}

fn seedLimbs(allocator: Allocator, raw: []const u8) ![]const u32 {
    var text = std.mem.trim(u8, raw, " \t\r\n\x0b\x0c");
    if (text.len > 0 and (text[0] == '+' or text[0] == '-')) text = text[1..];
    if (text.len == 0) return error.InvalidSeed;
    var limbs: std.ArrayList(u32) = .empty;
    try limbs.append(allocator, 0);
    var digit_seen = false;
    var underscore = false;
    for (text) |ch| {
        if (ch == '_') {
            if (!digit_seen or underscore) return error.InvalidSeed;
            underscore = true;
            continue;
        }
        if (!std.ascii.isDigit(ch)) return error.InvalidSeed;
        underscore = false;
        digit_seen = true;
        var carry: u64 = ch - '0';
        for (limbs.items) |*limb| {
            carry += @as(u64, limb.*) * 10;
            limb.* = @truncate(carry);
            carry >>= 32;
        }
        if (carry != 0) try limbs.append(allocator, @intCast(carry));
    }
    if (underscore) return error.InvalidSeed;
    return limbs.toOwnedSlice(allocator);
}

pub fn parseOptions(allocator: Allocator, args: []const []const u8) !Options {
    var target: ?[]const u8 = null;
    var seed: ?[]const u32 = null;
    var limit: ?Limit = null;
    var ranges: std.ArrayList(candidates.Range) = .empty;
    var listing = false;
    var positional = false;
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (!positional and std.mem.eql(u8, arg, "--")) {
            positional = true;
            continue;
        }
        if (!positional and std.mem.eql(u8, arg, "--list")) {
            listing = true;
            continue;
        }
        if (!positional and std.mem.startsWith(u8, arg, "--")) {
            const equals = std.mem.indexOfScalar(u8, arg, '=');
            const option = if (equals) |i| arg[0..i] else arg;
            const value = if (equals) |i| arg[i + 1 ..] else blk: {
                index += 1;
                if (index >= args.len) return error.MissingOptionValue;
                break :blk args[index];
            };
            if (std.mem.eql(u8, option, "--seed")) {
                seed = try seedLimbs(allocator, value);
            } else if (std.mem.eql(u8, option, "--limit")) {
                limit = try parseLimit(allocator, value);
            } else if (std.mem.eql(u8, option, "--line-range")) {
                const colon = std.mem.indexOfScalar(u8, value, ':') orelse return error.InvalidRange;
                const start = try std.fmt.parseInt(usize, value[0..colon], 10);
                const end = try std.fmt.parseInt(usize, value[colon + 1 ..], 10);
                if (start == 0 or end < start) return error.InvalidRange;
                try ranges.append(allocator, .{ .start = start, .end = end });
            } else return error.UnknownOption;
        } else if (target == null) {
            target = arg;
        } else return error.UnexpectedArgument;
    }
    if (!listing) return error.ListRequired;
    return .{ .target = target orelse return error.MissingTarget, .seed = seed, .limit = limit, .ranges = try ranges.toOwnedSlice(allocator) };
}

pub fn selected(all: []candidates.Mutant, options: Options) []const candidates.Mutant {
    if (options.seed) |seed| {
        var generator = shuffle.Generator.init(seed);
        generator.shuffle(candidates.Mutant, all);
    }
    const total = all.len;
    const length: usize = if (options.limit) |limit| if (!limit.negative)
        @min(total, limit.magnitude)
    else
        total - @min(total, limit.magnitude) else total;
    return all[0..length];
}

pub fn listingText(allocator: Allocator, source: []const u8, options: Options) ![]const u8 {
    const normalized = try candidates.normalizeNewlines(allocator, source);
    const all = try candidates.collect(allocator, normalized, options.ranges);
    const total = all.len;
    const chosen = selected(all, options);
    var result: std.Io.Writer.Allocating = .init(allocator);
    const writer = &result.writer;
    for (chosen, 1..) |mutant, index| {
        try writer.print("{d}: line {d}: {s}: {s}\n", .{ index, mutant.line_no, mutant.mutator, try report.strip(mutant.original_line) });
    }
    try writer.print("selected {d} of {d} mutants\n", .{ chosen.len, total });
    return result.toOwnedSlice();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const options = parseOptions(allocator, args[1..]) catch |err| {
        std.debug.print("mutation-list: {s}\nusage: mutation-list TARGET --list [--seed N] [--limit N] [--line-range START:END]\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (!std.mem.eql(u8, std.fs.path.extension(options.target), ".zig")) {
        std.debug.print("mutation-list: target is not a .zig file: {s}\n", .{options.target});
        std.process.exit(2);
    }
    const source = std.Io.Dir.cwd().readFileAlloc(init.io, options.target, allocator, .unlimited) catch |err| {
        std.debug.print("mutation-list: cannot read {s}: {s}\n", .{ options.target, @errorName(err) });
        std.process.exit(2);
    };
    const text = listingText(allocator, source, options) catch |err| {
        std.debug.print("mutation-list: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    try std.Io.File.stdout().writeStreamingAll(init.io, text);
}

test "read-only listing applies reference seed, ranges, and negative limits" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const opts = try parseOptions(allocator, &.{ "source.zig", "--list", "--seed=-737", "--limit=-1", "--line-range", "1:2" });
    try std.testing.expectEqualStrings(
        \\1: line 1: integer 1 -> 0 at column 5: x < 1;
        \\2: line 2: off-by-one + 1 -> - 1 at column 7: x = x + 1;
        \\3: line 2: integer 1 -> 0 at column 9: x = x + 1;
        \\selected 3 of 4 mutants
        \\
    , try listingText(allocator, "x < 1;\nx = x + 1;\n", opts));
    try std.testing.expectError(error.ListRequired, parseOptions(allocator, &.{"source.zig"}));
    try std.testing.expectError(error.InvalidSeed, seedLimbs(allocator, "1__2"));
    try std.testing.expectEqualSlices(u32, &.{ 12345, 0, 0, 16 }, try seedLimbs(allocator, "1267650600228229401496703217721"));
}

test "listing normalizes CRLF and CR source newlines" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const opts = try parseOptions(allocator, &.{ "source.zig", "--list" });
    const unix = try listingText(allocator, "x < 1;\nx = x + 1;\n", opts);
    const mixed = try listingText(allocator, "x < 1;\r\nx = x + 1;\r", opts);
    try std.testing.expectEqualStrings(unix, mixed);
}

test "limits preserve arbitrary integer labels and reference slice behavior" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const source = "x = 0;\nx = 1;\n";
    const huge = "999999999999999999999999999999999999999999";
    const positive = try parseOptions(allocator, &.{ "source.zig", "--list", "--limit", huge });
    try std.testing.expectEqualStrings(huge, positive.limit.?.label);
    try std.testing.expect(std.mem.endsWith(u8, try listingText(allocator, source, positive), "selected 2 of 2 mutants\n"));
    const negative = try parseOptions(allocator, &.{ "source.zig", "--list", "--limit", "-999999999999999999999999999999999999999999" });
    try std.testing.expectEqualStrings("selected 0 of 2 mutants\n", try listingText(allocator, source, negative));
    const negative_one = try parseOptions(allocator, &.{ "source.zig", "--list", "--limit", " -00_01 " });
    try std.testing.expectEqualStrings("-1", negative_one.limit.?.label);
    try std.testing.expect(std.mem.endsWith(u8, try listingText(allocator, source, negative_one), "selected 1 of 2 mutants\n"));
    const negative_zero = try parseLimit(allocator, "-0_0");
    try std.testing.expectEqualStrings("0", negative_zero.label);
    try std.testing.expect(!negative_zero.negative);
    try std.testing.expectError(error.InvalidLimit, parseLimit(allocator, "1__0"));
    try std.testing.expectError(error.InvalidLimit, parseLimit(allocator, "_1"));
    try std.testing.expectError(error.InvalidLimit, parseLimit(allocator, "1_"));
}
