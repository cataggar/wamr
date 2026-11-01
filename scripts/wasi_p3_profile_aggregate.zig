//! Validate and aggregate matched macOS/Linux ARM64 WASI P3 profiles.
const std = @import("std");
const aggregation = @import("wasi_p3_profile_aggregation.zig");
const fixed = @import("decimal_format.zig").fixed;
const floatText = @import("float_text.zig").format;
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

const Options = struct { input: []const u8, output: []const u8 };

fn options(args: []const []const u8) !Options {
    var input: ?[]const u8 = null;
    var output: ?[]const u8 = null;
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return error.HelpRequested;
        const split = std.mem.indexOfScalar(u8, arg, '=');
        const name = if (split) |pos| arg[0..pos] else arg;
        const value = if (split) |pos| arg[pos + 1 ..] else blk: {
            index += 1;
            if (index >= args.len) return error.MissingOptionValue;
            break :blk args[index];
        };
        if (std.mem.eql(u8, name, "--input-dir")) {
            input = value;
        } else if (std.mem.eql(u8, name, "--output-dir")) {
            output = value;
        } else return error.UnknownOption;
    }
    return .{ .input = input orelse return error.MissingInputDirectory, .output = output orelse return error.MissingOutputDirectory };
}

const Document = struct { path: []const u8, data: Value };

fn less(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn readJson(allocator: Allocator, io: std.Io, path: []const u8) !Value {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    return std.json.parseFromSliceLeaky(Value, allocator, bytes, .{});
}

fn load(allocator: Allocator, io: std.Io, root: []const u8) ![]const Document {
    const dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    var paths: std.ArrayList([]const u8) = .empty;
    while (try walker.next(io)) |entry| {
        if (std.mem.eql(u8, entry.basename, "samples.json"))
            try paths.append(allocator, try std.fs.path.join(allocator, &.{ root, entry.path }));
    }
    if (paths.items.len == 0) return error.NoProfileDocuments;
    std.mem.sort([]const u8, paths.items, {}, less);
    const documents = try allocator.alloc(Document, paths.items.len);
    for (paths.items, documents) |path, *document| document.* = .{ .path = path, .data = try readJson(allocator, io, path) };
    return documents;
}

fn microbench(allocator: Allocator, io: std.Io, documents: []const Document) !Value {
    var reports: std.json.ObjectMap = .empty;
    for (documents) |document| {
        const metadata = document.data.object.get("metadata").?.object;
        const platform = metadata.get("platform_id").?.string;
        if (reports.contains(platform)) continue;
        const parent = std.fs.path.dirname(document.path) orelse return error.InvalidProfilePath;
        const base = std.fs.path.dirname(parent) orelse return error.InvalidProfilePath;
        const path = try std.fs.path.join(allocator, &.{ base, "microbench", "report.json" });
        const report = try readJson(allocator, io, path);
        if (report != .object) return error.InvalidMicrobenchmark;
        const scenarios = report.object.get("scenarios") orelse return error.InvalidMicrobenchmark;
        if (scenarios != .array or scenarios.array.items.len != 4) return error.InvalidMicrobenchmark;
        for (scenarios.array.items) |item| {
            if (item != .object) return error.InvalidMicrobenchmark;
            const verdict = item.object.get("verdict") orelse return error.InvalidMicrobenchmark;
            if (verdict != .string or !std.mem.eql(u8, verdict.string, "no-budget")) return error.InvalidMicrobenchmark;
        }
        try reports.put(allocator, platform, report);
    }
    if (reports.count() != 2 or !reports.contains("linux-arm64") or !reports.contains("macos-arm64"))
        return error.InvalidMicrobenchmark;
    return .{ .object = reports };
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

fn milliseconds(allocator: Allocator, value: i128) ![]const u8 {
    const number = @as(f64, @floatFromInt(value)) / 1_000_000;
    const magnitude = try fixed(allocator, @abs(number), 3);
    return if (number < 0) std.fmt.allocPrint(allocator, "-{s}", .{magnitude}) else magnitude;
}

fn writeReprString(writer: *std.Io.Writer, text: []const u8) !void {
    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidMicrobenchmarkEncoding;
    const quote: u8 = if (std.mem.indexOfScalar(u8, text, '\'') != null and
        std.mem.indexOfScalar(u8, text, '"') == null) '"' else '\'';
    try writer.writeByte(quote);
    var index: usize = 0;
    while (index < text.len) {
        const width = try std.unicode.utf8ByteSequenceLength(text[index]);
        const point = try std.unicode.utf8Decode(text[index..][0..width]);
        const encoded = text[index..][0..width];
        index += width;
        switch (point) {
            '\t' => try writer.writeAll("\\t"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\\' => try writer.writeAll("\\\\"),
            else => {
                if (point == quote) {
                    try writer.writeByte('\\');
                    try writer.writeByte(quote);
                } else if (point < 0x20 or (point >= 0x7f and point <= 0xa0) or point == 0xad) {
                    try writer.print("\\x{x:0>2}", .{point});
                } else if (point <= 0xff) {
                    try writer.writeAll(encoded);
                } else {
                    return error.UnsupportedMicrobenchmarkRepr;
                }
            },
        }
    }
    try writer.writeByte(quote);
}

fn writeRepr(allocator: Allocator, writer: *std.Io.Writer, value: Value, depth: usize) anyerror!void {
    if (depth > 256) return error.UnsupportedMicrobenchmarkRepr;
    switch (value) {
        .string => |text| try writeReprString(writer, text),
        .array => |array| {
            try writer.writeByte('[');
            for (array.items, 0..) |item, index| {
                if (index != 0) try writer.writeAll(", ");
                try writeRepr(allocator, writer, item, depth + 1);
            }
            try writer.writeByte(']');
        },
        .object => |object| {
            try writer.writeByte('{');
            var iterator = object.iterator();
            var first = true;
            while (iterator.next()) |field| {
                if (!first) try writer.writeAll(", ");
                first = false;
                try writeReprString(writer, field.key_ptr.*);
                try writer.writeAll(": ");
                try writeRepr(allocator, writer, field.value_ptr.*, depth + 1);
            }
            try writer.writeByte('}');
        },
        else => {
            const text = try scalarText(allocator, value);
            defer switch (value) {
                .integer, .float => allocator.free(text),
                .number_string => |number| if (std.mem.indexOfAny(u8, number, ".eE") != null) {
                    allocator.free(text);
                },
                else => {},
            };
            try writer.writeAll(text);
        },
    }
}

fn scalarText(allocator: Allocator, value: Value) ![]const u8 {
    return switch (value) {
        .string => value.string,
        .integer => std.fmt.allocPrint(allocator, "{d}", .{value.integer}),
        .float => floatText(allocator, value.float),
        .number_string => if (std.mem.indexOfAny(u8, value.number_string, ".eE") != null)
            floatText(allocator, try std.fmt.parseFloat(f64, value.number_string))
        else
            value.number_string,
        .bool => if (value.bool) "True" else "False",
        .null => "None",
        .array, .object => blk: {
            var out: std.Io.Writer.Allocating = .init(allocator);
            errdefer out.deinit();
            try writeRepr(allocator, &out.writer, value, 0);
            break :blk try out.toOwnedSlice();
        },
    };
}

const ReportJson = struct {
    allocator: Allocator,
    value: Value,

    pub fn jsonStringify(self: ReportJson, writer: *std.json.Stringify) std.json.Stringify.Error!void {
        switch (self.value) {
            .object => |object| {
                try writer.beginObject();
                var fields = object.iterator();
                while (fields.next()) |field| {
                    try writer.objectField(field.key_ptr.*);
                    try writer.write(ReportJson{ .allocator = self.allocator, .value = field.value_ptr.* });
                }
                try writer.endObject();
            },
            .array => |array| {
                try writer.beginArray();
                for (array.items) |value|
                    try writer.write(ReportJson{ .allocator = self.allocator, .value = value });
                try writer.endArray();
            },
            .float => |value| {
                if (!std.math.isFinite(value)) {
                    try writer.beginWriteRaw();
                    try writer.writer.writeAll(if (std.math.isNan(value)) "NaN" else if (value < 0) "-Infinity" else "Infinity");
                    writer.endWriteRaw();
                    return;
                }
                const text = floatText(self.allocator, value) catch return error.WriteFailed;
                defer self.allocator.free(text);
                try writer.beginWriteRaw();
                try writer.writer.writeAll(text);
                writer.endWriteRaw();
            },
            else => try writer.write(self.value),
        }
    }
};

fn markdown(allocator: Allocator, result: aggregation.Analysis, micro: Value, date: []const u8) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    const w = &out.writer;
    try w.print(
        \\# WASI P3 macOS ARM64 performance investigation — {s}
        \\
        \\Revision: `{s}`.
        \\
        \\All 60 measured suite runs were valid unfiltered ReleaseSafe runs: 5 cold + 10 warm for AOT and JIT on each native ARM64 platform, with every sample passing 41/41.
        \\
        \\## Suite wall time
        \\
        \\| platform | mode | cache state | samples | median ms | p95 ms |
        \\|---|---:|---:|---:|---:|---:|
        \\
    , .{ date, result.commit });
    for (result.suite) |row| try w.print("| {s} | {s} | {s} | {d} | {s} | {s} |\n", .{
        row.platform_id,                            row.mode,                                row.temperature, row.samples,
        try milliseconds(allocator, row.median_ns), try milliseconds(allocator, row.p95_ns),
    });
    try w.writeAll(
        \\
        \\## Sampling-profile selection
        \\
        \\| mode | fixture | phase | reason |
        \\|---|---|---|---|
        \\
    );
    for (result.profiles) |profile| {
        try w.print("| {s} | {s} | {s} | ", .{ profile.mode, profile.fixture, profile.phase });
        for (profile.reasons, 0..) |reason, index| {
            if (index != 0) try w.writeAll("; ");
            try w.writeAll(reason);
        }
        try w.writeAll(" |\n");
    }
    try w.writeAll(
        \\
        \\## WASI microbenchmark (budget disabled)
        \\
        \\| platform | scenario | median ns | p95 ns |
        \\|---|---|---:|---:|
        \\
    );
    for ([_][]const u8{ "linux-arm64", "macos-arm64" }) |platform| {
        const scenarios = micro.object.get(platform).?.object.get("scenarios").?.array.items;
        for (scenarios) |scenario| {
            const name = scenario.object.get("name") orelse return error.InvalidMicrobenchmark;
            const median = scenario.object.get("median_ns") orelse return error.InvalidMicrobenchmark;
            const p95 = scenario.object.get("p95_ns") orelse return error.InvalidMicrobenchmark;
            try w.print("| {s} | {s} | {s} | {s} |\n", .{
                platform, try scalarText(allocator, name), try scalarText(allocator, median), try scalarText(allocator, p95),
            });
        }
    }
    try w.writeAll("\nThe existing Linux x86_64 budgets were not applied or changed. This manual workflow is non-blocking and does not alter normal conformance gates.\n");
    return out.toOwnedSlice();
}

fn writeJson(allocator: Allocator, io: std.Io, dir: []const u8, name: []const u8, value: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(allocator, value, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(allocator, &.{ dir, name }), .data = try std.fmt.allocPrint(allocator, "{s}\n", .{text}) });
}

pub const Summary = struct { commit: []const u8, profiles: usize };

pub fn aggregate(allocator: Allocator, io: std.Io, input: []const u8, output: []const u8) !Summary {
    const documents = try load(allocator, io, input);
    const values = try allocator.alloc(Value, documents.len);
    for (documents, values) |doc, *value| value.* = doc.data;
    const result = try aggregation.analyze(allocator, values);
    const micro = try microbench(allocator, io, documents);
    const now = try timestamp(allocator, std.Io.Clock.real.now(io).nanoseconds);
    const report_time = try timestamp(allocator, std.Io.Clock.real.now(io).nanoseconds);
    const report = try markdown(allocator, result, micro, report_time[0..10]);
    const selection = .{ .schema_version = 1, .kind = "wasi-p3-profile-selection", .profiles = result.profiles };
    const document = .{
        .schema_version = 1,
        .kind = "wasi-p3-profile-aggregate",
        .generated_at = now,
        .commit = result.commit,
        .valid = true,
        .contract = .{ .platforms = [_][]const u8{ "linux-arm64", "macos-arm64" }, .modes = [_][]const u8{ "aot", "jit" }, .cold_samples_per_platform_mode = 5, .warm_samples_per_platform_mode = 10, .fixtures_per_sample = 41 },
        .suite = result.suite,
        .phases = result.phases,
        .microbench = ReportJson{ .allocator = allocator, .value = micro },
    };
    try std.Io.Dir.cwd().createDirPath(io, output);
    try writeJson(allocator, io, output, "aggregate.json", document);
    try writeJson(allocator, io, output, "profile-selection.json", selection);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(allocator, &.{ output, "report.md" }), .data = report });
    return .{ .commit = result.commit, .profiles = result.profiles.len };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const usage = "usage: wasi-p3-profile-aggregate --input-dir INPUT_DIR --output-dir OUTPUT_DIR\n";
    const opts = options(args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            try std.Io.File.stdout().writeStreamingAll(init.io, usage);
            return;
        }
        std.debug.print("error: {s}\n{s}", .{ @errorName(err), usage });
        std.process.exit(2);
    };
    const summary = aggregate(allocator, init.io, opts.input, opts.output) catch |err| {
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };
    try std.Io.File.stdout().writeStreamingAll(init.io, try std.fmt.allocPrint(allocator, "Validated 60 41/41 suite samples at {s}; selected {d} profiles.\n", .{ summary.commit, summary.profiles }));
}

test "microbenchmark scalar text preserves unsigned integers and nulls" {
    const allocator = std.testing.allocator;
    const large = try std.json.parseFromSlice(Value, allocator, "18446744073709551615", .{});
    defer large.deinit();
    try std.testing.expectEqualStrings("18446744073709551615", try scalarText(allocator, large.value));
    try std.testing.expectEqualStrings("True", try scalarText(allocator, .{ .bool = true }));
    try std.testing.expectEqualStrings("None", try scalarText(allocator, .null));
    const infinity = try scalarText(allocator, .{ .number_string = "1e400" });
    defer allocator.free(infinity);
    try std.testing.expectEqualStrings("inf", infinity);
}

test "microbenchmark floats use reference decimal notation" {
    const allocator = std.testing.allocator;
    for ([_]struct { value: f64, expected: []const u8 }{
        .{ .value = 1.0, .expected = "1.0" },
        .{ .value = -0.0, .expected = "-0.0" },
        .{ .value = 0.0001, .expected = "0.0001" },
        .{ .value = 0.00001, .expected = "1e-05" },
        .{ .value = 1.25e-8, .expected = "1.25e-08" },
        .{ .value = 1e15, .expected = "1000000000000000.0" },
        .{ .value = 1e16, .expected = "1e+16" },
        .{ .value = 1e20, .expected = "1e+20" },
    }) |item| {
        const text = try floatText(allocator, item.value);
        defer allocator.free(text);
        try std.testing.expectEqualStrings(item.expected, text);
    }
}

test "microbenchmark nested values use reference representations" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Value, allocator,
        \\[{"can't":"a\"b\n\t\u0000","plain":[true,null,1.0,1e20]}, "both'\"\\", []]
    , .{});
    try std.testing.expectEqualStrings(
        \\[{"can't": 'a"b\n\t\x00', 'plain': [True, None, 1.0, 1e+20]}, 'both\'"\\', []]
    , try scalarText(allocator, parsed));
    const latin1 = try std.json.parseFromSliceLeaky(Value, allocator, "[{\"caf\xc3\xa9\":\"\\u00a0\\u00ad\"}]", .{});
    try std.testing.expectEqualStrings("[{'caf\xc3\xa9': '\\xa0\\xad'}]", try scalarText(allocator, latin1));
    const unsupported = try std.json.parseFromSliceLeaky(Value, allocator, "[\"\xe2\x98\x83\"]", .{});
    try std.testing.expectError(error.UnsupportedMicrobenchmarkRepr, scalarText(allocator, unsupported));
    try std.testing.expectEqualStrings("caf\xc3\xa9", try scalarText(allocator, .{ .string = "caf\xc3\xa9" }));
}

test "Latin-1 representations preserve printable characters and escape controls" {
    const allocator = std.testing.allocator;
    for (0x80..0x100) |point| {
        var bytes: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(@intCast(point), &bytes);
        var actual: std.Io.Writer.Allocating = .init(allocator);
        defer actual.deinit();
        try writeReprString(&actual.writer, bytes[0..len]);
        const expected = if (point <= 0xa0 or point == 0xad)
            try std.fmt.allocPrint(allocator, "'\\x{x:0>2}'", .{point})
        else
            try std.fmt.allocPrint(allocator, "'{s}'", .{bytes[0..len]});
        defer allocator.free(expected);
        try std.testing.expectEqualStrings(expected, actual.written());
    }
}

test "microbenchmark JSON preserves floating-point values" {
    const allocator = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Value, allocator, "{\"x\":[1.0,-0.0,1e20]}", .{});
    defer parsed.deinit();
    const output = try std.json.Stringify.valueAlloc(allocator, ReportJson{ .allocator = allocator, .value = parsed.value }, .{});
    defer allocator.free(output);
    try std.testing.expectEqualStrings("{\"x\":[1.0,-0.0,1e+20]}", output);
}

test "aggregate complete matched runs and reject budgeted microbenchmarks" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", allocator);
    const input = try std.fs.path.join(allocator, &.{ root, "input" });
    const output = try std.fs.path.join(allocator, &.{ root, "output" });
    for ([_][]const u8{ "linux-arm64", "macos-arm64" }) |platform| {
        for ([_][]const u8{ "aot", "jit" }) |mode| {
            const base = try std.fmt.allocPrint(allocator, "input/measure-{s}-{s}", .{ platform, mode });
            const run_dir = try std.fs.path.join(allocator, &.{ base, mode });
            const micro_dir = try std.fs.path.join(allocator, &.{ base, "microbench" });
            try temp.dir.createDirPath(io, run_dir);
            try temp.dir.createDirPath(io, micro_dir);
            const system = if (std.mem.eql(u8, platform, "linux-arm64")) "Linux" else "Darwin";
            const scale: u32 = if (std.mem.eql(u8, system, "Linux")) 1 else 3;
            const document_json = try std.fmt.allocPrint(allocator,
                \\{{"schema_version":1,"kind":"wasi-p3-profile-runs",
                \\ "metadata":{{"mode":"{s}","platform_id":"{s}","commit":"abc",
                \\ "optimize":"ReleaseSafe","host":{{"machine":"arm64","system":"{s}"}},
                \\ "tools":{{"zig":"z","wamr":"w","wamrc":"c"}},"cache":{{}}}},"samples":[]}}
            , .{ mode, platform, system });
            var document = try std.json.parseFromSliceLeaky(Value, allocator, document_json, .{});
            const samples = &document.object.getPtr("samples").?.array;
            for (0..15) |index| {
                const temperature = if (index < 5) "cold" else "warm";
                const sample_json = try std.fmt.allocPrint(allocator,
                    \\{{"id":"sample","temperature":"{s}","index":{d},"valid":true,"suite_duration_ns":{d},
                    \\ "counts":{{"fixtures":41,"executed":41,"passed":41,"failed":0}},"events":[]}}
                , .{ temperature, if (index < 5) index + 1 else index - 4, 1_000_000_000 * scale });
                var sample = try std.json.parseFromSliceLeaky(Value, allocator, sample_json, .{});
                const events = &sample.object.getPtr("events").?.array;
                for (0..41) |fixture_index| {
                    const fixture = if (fixture_index == 0) "http-fields" else try std.fmt.allocPrint(allocator, "fixture-{d}", .{fixture_index});
                    for ([_][]const u8{ "component_precompile", "fixture_execution" }) |phase| {
                        const duration: u32 = if (std.mem.eql(u8, phase, "fixture_execution"))
                            (if (fixture_index == 0) @as(u32, 300_000_000) else 1_000_000) * scale
                        else if (std.mem.eql(u8, mode, "jit")) 0 else 100 * scale;
                        const event_json = try std.fmt.allocPrint(allocator,
                            \\{{"schema_version":1,"event":"phase_timing","mode":"{s}",
                            \\ "fixture":"{s}","phase":"{s}","artifact_kind":"component",
                            \\ "cache":"n/a","duration_ns":{d}}}
                        , .{ mode, fixture, phase, duration });
                        try events.append(try std.json.parseFromSliceLeaky(Value, allocator, event_json, .{}));
                    }
                }
                try samples.append(sample);
            }
            const sample_path = try std.fs.path.join(allocator, &.{ run_dir, "samples.json" });
            try temp.dir.writeFile(io, .{ .sub_path = sample_path, .data = try std.json.Stringify.valueAlloc(allocator, document, .{}) });
            const micro_path = try std.fs.path.join(allocator, &.{ micro_dir, "report.json" });
            try temp.dir.writeFile(io, .{ .sub_path = micro_path, .data =
                \\{"scenarios":[{"name":"a","median_ns":10.0,"p95_ns":12.5,"verdict":"no-budget"},
                \\{"name":["caf\u00e9",true],"median_ns":{"value":10},"p95_ns":12,"verdict":"no-budget"},
                \\{"name":"c","median_ns":10,"p95_ns":12,"verdict":"no-budget"},
                \\{"name":"d","median_ns":10,"p95_ns":12,"verdict":"no-budget"}]}
            });
        }
    }
    const summary = try aggregate(allocator, io, input, output);
    try std.testing.expectEqualStrings("abc", summary.commit);
    const document = try readJson(allocator, io, try std.fs.path.join(allocator, &.{ output, "aggregate.json" }));
    try std.testing.expectEqual(@as(usize, 8), document.object.get("suite").?.array.items.len);
    try std.testing.expectEqual(@as(usize, 656), document.object.get("phases").?.array.items.len);
    const suite = document.object.get("suite").?.array.items[0].object;
    try std.testing.expectEqualStrings("linux-arm64", suite.get("platform_id").?.string);
    try std.testing.expectEqual(@as(i64, 1_000_000_000), suite.get("median_ns").?.integer);
    const scenario = document.object.get("microbench").?.object.get("linux-arm64").?.object.get("scenarios").?.array.items[0].object;
    try std.testing.expectEqual(@as(f64, 10.0), scenario.get("median_ns").?.float);
    const selection = try readJson(allocator, io, try std.fs.path.join(allocator, &.{ output, "profile-selection.json" }));
    try std.testing.expectEqual(summary.profiles, selection.object.get("profiles").?.array.items.len);
    try std.testing.expect(summary.profiles > 2);
    const report = try temp.dir.readFileAlloc(io, "output/report.md", allocator, .unlimited);
    try std.testing.expect(std.mem.indexOf(u8, report, "| linux-arm64 | aot | cold | 5 | 1000.000 | 1000.000 |") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "| linux-arm64 | a | 10.0 | 12.5 |") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "| linux-arm64 | ['caf\xc3\xa9', True] | {'value': 10} | 12 |") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "macOS/Linux median >=2x in cold and warm cells (100% paired samples >=2x)") != null);

    const sample_path = "input/measure-macos-arm64-aot/aot/samples.json";
    const sample_full = try std.fs.path.join(allocator, &.{ root, sample_path });
    const original = try temp.dir.readFileAlloc(io, sample_path, allocator, .unlimited);
    var changed = try readJson(allocator, io, sample_full);
    const sample = &changed.object.getPtr("samples").?.array.items[0];
    sample.object.getPtr("counts").?.object.getPtr("passed").?.* = .{ .integer = 40 };
    try temp.dir.writeFile(io, .{ .sub_path = sample_path, .data = try std.json.Stringify.valueAlloc(allocator, changed, .{}) });
    const failed = try std.fs.path.join(allocator, &.{ root, "failed" });
    try std.testing.expectError(error.InvalidPlanSample, aggregate(allocator, io, input, failed));
    try std.testing.expectError(error.FileNotFound, temp.dir.access(io, "failed/aggregate.json", .{}));
    try temp.dir.writeFile(io, .{ .sub_path = sample_path, .data = original });

    changed = try readJson(allocator, io, sample_full);
    changed.object.getPtr("metadata").?.object.getPtr("commit").?.* = .{ .string = "different" };
    try temp.dir.writeFile(io, .{ .sub_path = sample_path, .data = try std.json.Stringify.valueAlloc(allocator, changed, .{}) });
    try std.testing.expectError(error.PlanRevisionMismatch, aggregate(allocator, io, input, failed));
    try std.testing.expectError(error.FileNotFound, temp.dir.access(io, "failed/aggregate.json", .{}));
    try temp.dir.writeFile(io, .{ .sub_path = sample_path, .data = original });

    try temp.dir.deleteFile(io, sample_path);
    try std.testing.expectError(error.IncompletePlan, aggregate(allocator, io, input, failed));
    try std.testing.expectError(error.FileNotFound, temp.dir.access(io, "failed/aggregate.json", .{}));
    try temp.dir.writeFile(io, .{ .sub_path = sample_path, .data = original });
    try temp.dir.createDirPath(io, "input/duplicate/aot");
    try temp.dir.writeFile(io, .{ .sub_path = "input/duplicate/aot/samples.json", .data = original });
    try std.testing.expectError(error.DuplicatePlanCell, aggregate(allocator, io, input, failed));
    try std.testing.expectError(error.FileNotFound, temp.dir.access(io, "failed/aggregate.json", .{}));
    try temp.dir.deleteFile(io, "input/duplicate/aot/samples.json");

    const bad_micro = "input/measure-linux-arm64-aot/microbench/report.json";
    try temp.dir.writeFile(io, .{ .sub_path = bad_micro, .data =
        \\{"scenarios":[{"verdict":"no-budget"},{"verdict":"no-budget"},
        \\{"verdict":"no-budget"},{"verdict":"budgeted"}]}
    });
    try std.testing.expectError(error.InvalidMicrobenchmark, aggregate(allocator, io, input, failed));
    try std.testing.expectError(error.FileNotFound, temp.dir.access(io, "failed/aggregate.json", .{}));
}
