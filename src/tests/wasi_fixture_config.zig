//! Parsing and operation validation for the pinned wasi-testsuite sidecars.
const std = @import("std");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const Object = std.json.ObjectMap;

pub const World = enum { cli_command, http_service };
pub const Proposal = enum { http, sockets };
pub const Run = struct {
    args: []const []const u8 = &.{},
    env: []const Env = &.{},
    root: ?[]const u8 = null,
};
pub const Env = struct { name: []const u8, value: []const u8 };
pub const Stream = struct { id: []const u8, payload: []const u8 = "" };
pub const Request = struct {
    method: []const u8 = "GET",
    path: []const u8 = "/",
    status: i64 = 200,
    headers: []const Env = &.{},
    body: []const u8 = "",
};
pub const Operation = union(enum) {
    run: Run,
    wait: i64,
    read: Stream,
    write: Stream,
    connect: []const u8,
    send: Stream,
    recv: Stream,
    request: Request,
    kill: []const u8,
};
pub const Config = struct {
    operations: []const Operation,
    proposals: []const Proposal = &.{},
    world: World = .cli_command,
    debug: bool = false,
};

fn object(value: Value) !Object {
    return switch (value) {
        .object => |map| map,
        else => error.ExpectedObject,
    };
}

fn string(value: Value) ![]const u8 {
    return switch (value) {
        .string => |str| str,
        else => error.ExpectedString,
    };
}

fn text(map: Object, key: []const u8, fallback: []const u8) ![]const u8 {
    const value = map.get(key) orelse return fallback;
    return try string(value);
}

fn integer(map: Object, key: []const u8, fallback: i64) !i64 {
    const value = map.get(key) orelse return fallback;
    return switch (value) {
        .integer => |n| n,
        else => error.ExpectedInteger,
    };
}

fn array(value: Value) ![]const Value {
    return switch (value) {
        .array => |items| items.items,
        else => error.ExpectedArray,
    };
}

fn strings(allocator: Allocator, value: ?Value) ![]const []const u8 {
    const items = try array(value orelse return &.{});
    const result = try allocator.alloc([]const u8, items.len);
    for (items, 0..) |item, i| result[i] = try string(item);
    return result;
}

fn envEntries(allocator: Allocator, value: ?Value) ![]const Env {
    const map = try object(value orelse return &.{});
    const result = try allocator.alloc(Env, map.count());
    var it = map.iterator();
    var i: usize = 0;
    while (it.next()) |entry| : (i += 1) {
        result[i] = .{ .name = entry.key_ptr.*, .value = try string(entry.value_ptr.*) };
    }
    return result;
}

fn parseRun(allocator: Allocator, map: Object) !Run {
    const root_value = map.get("root");
    const root: ?[]const u8 = if (root_value) |value|
        if (value == .null) null else try string(value)
    else
        null;
    return .{
        .args = try strings(allocator, map.get("args")),
        .env = try envEntries(allocator, map.get("env")),
        .root = root,
    };
}

fn parseRequest(allocator: Allocator, map: Object) !Request {
    var result: Request = .{ .method = try text(map, "method", "GET"), .path = try text(map, "path", "/") };
    if (map.get("response")) |value| {
        const response = try object(value);
        result.status = try integer(response, "status", 200);
        result.headers = try envEntries(allocator, response.get("headers"));
        result.body = try text(response, "body", "");
    }
    return result;
}

pub fn validSignal(name: []const u8) bool {
    if (!std.mem.startsWith(u8, name, "SIG")) return false;
    const suffix = name[3..];
    if (std.meta.stringToEnum(std.c.SIG, suffix) != null) return true;
    inline for (@typeInfo(std.c.SIG).@"enum".decl_names) |decl| {
        if (@TypeOf(@field(std.c.SIG, decl)) == std.c.SIG and std.mem.eql(u8, suffix, decl)) return true;
    }
    return @import("builtin").os.tag == .linux and
        (std.mem.eql(u8, suffix, "RTMIN") or std.mem.eql(u8, suffix, "RTMAX"));
}

fn truthy(value: Value) bool {
    return switch (value) {
        .null => false,
        .bool => |v| v,
        .integer => |v| v != 0,
        .float => |v| v != 0,
        .number_string => |v| !std.mem.eql(u8, v, "0"),
        .string => |v| v.len != 0,
        .array => |v| v.items.len != 0,
        .object => |v| v.count() != 0,
    };
}

fn warnUnknown(map: Object, allowed: []const []const u8) void {
    var it = map.iterator();
    while (it.next()) |entry| {
        for (allowed) |key| {
            if (std.mem.eql(u8, key, entry.key_ptr.*)) break;
        } else {
            std.log.warn("unknown field in WASI fixture: {s}", .{entry.key_ptr.*});
        }
    }
}

fn parseOperation(allocator: Allocator, value: Value) !Operation {
    const map = try object(value);
    const kind = try text(map, "type", "");
    if (std.mem.eql(u8, kind, "run")) return .{ .run = try parseRun(allocator, map) };
    if (std.mem.eql(u8, kind, "wait")) return .{ .wait = try integer(map, "exit_code", 0) };
    if (std.mem.eql(u8, kind, "read")) return .{ .read = .{ .id = try text(map, "id", "stdout"), .payload = try text(map, "payload", "") } };
    if (std.mem.eql(u8, kind, "write")) return .{ .write = .{ .id = try text(map, "id", "write"), .payload = try text(map, "payload", "") } };
    if (std.mem.eql(u8, kind, "connect")) {
        if (!std.mem.eql(u8, try text(map, "protocol_type", "tcp"), "tcp")) return error.UnsupportedProtocol;
        return .{ .connect = try text(map, "id", "server") };
    }
    if (std.mem.eql(u8, kind, "send") or std.mem.eql(u8, kind, "recv")) {
        const id = try string(map.get("id") orelse return error.MissingStreamId);
        const stream: Stream = .{ .id = id, .payload = try text(map, "payload", "") };
        return if (std.mem.eql(u8, kind, "send")) .{ .send = stream } else .{ .recv = stream };
    }
    if (std.mem.eql(u8, kind, "request")) return .{ .request = try parseRequest(allocator, map) };
    if (std.mem.eql(u8, kind, "kill")) {
        const signal = try text(map, "signal", "SIGTERM");
        if (!validSignal(signal)) return error.InvalidSignal;
        return .{ .kill = signal };
    }
    return error.UnsupportedOperation;
}

const StreamType = enum { read, write, socket };

fn expectStream(streams: *std.StringHashMap(StreamType), id: []const u8, expected: StreamType) !void {
    if ((streams.get(id) orelse return error.UnknownStream) != expected) return error.WrongStreamType;
}

pub fn validate(allocator: Allocator, operations: []const Operation) !void {
    var streams: std.StringHashMap(StreamType) = .init(allocator);
    defer streams.deinit();
    var running = false;
    for (operations) |op| {
        switch (op) {
            .run => {
                if (running) return error.ProcessAlreadyRunning;
                running = true;
                try streams.put("stdin", .write);
                try streams.put("stdout", .read);
                try streams.put("stderr", .read);
            },
            .wait => {
                if (!running) return error.NoProcess;
                running = false;
                streams.clearRetainingCapacity();
            },
            .read => |v| {
                if (!running) return error.NoProcess;
                try expectStream(&streams, v.id, .read);
            },
            .write => |v| {
                if (!running) return error.NoProcess;
                try expectStream(&streams, v.id, .write);
            },
            .connect => |id| {
                if (!running) return error.NoProcess;
                if (streams.contains(id)) return error.DuplicateStream;
                try streams.put(id, .socket);
            },
            .send, .recv => |v| {
                if (!running) return error.NoProcess;
                try expectStream(&streams, v.id, .socket);
            },
            .request, .kill => if (!running) return error.NoProcess,
        }
    }
    if (running) return error.ProcessStillRunning;
}

/// The caller owns `allocator` and must retain it for the lifetime of the result.
pub fn parse(allocator: Allocator, source: []const u8) !Config {
    const value = try std.json.parseFromSliceLeaky(Value, allocator, source, .{ .allocate = .alloc_always });
    const map = try object(value);
    const modern = if (map.get("operations")) |v| v != .null else false;
    const proposals_present = if (map.get("proposals")) |v| v != .null else false;
    const has_modern_key = map.contains("operations") or map.contains("proposals") or map.contains("world");
    const legacy_keys = [_][]const u8{ "args", "root", "env", "exit_code", "stderr", "stdout" };
    if (has_modern_key) {
        for (legacy_keys) |key| if (map.contains(key)) return error.MixedConfigStyles;
    }
    var operations: std.ArrayList(Operation) = .empty;
    var proposals: std.ArrayList(Proposal) = .empty;
    var world: World = .cli_command;
    var debug = false;
    if (modern or proposals_present) {
        warnUnknown(map, &.{ "operations", "proposals", "world" });
        if (modern) {
            for (try array(map.get("operations").?)) |op| try operations.append(allocator, try parseOperation(allocator, op));
        }
        if (proposals_present) {
            for (try array(map.get("proposals").?)) |value_proposal| {
                const name = try string(value_proposal);
                const proposal: Proposal = if (std.mem.eql(u8, name, "http")) .http else if (std.mem.eql(u8, name, "sockets")) .sockets else return error.UnsupportedProposal;
                try proposals.append(allocator, proposal);
            }
        }
        const world_name = try text(map, "world", "wasi:cli/command");
        world = if (std.mem.eql(u8, world_name, "wasi:cli/command")) .cli_command else if (std.mem.eql(u8, world_name, "wasi:http/service")) .http_service else return error.UnsupportedWorld;
        if (map.get("debug")) |v| debug = truthy(v);
    } else {
        warnUnknown(map, &legacy_keys);
        try operations.append(allocator, .{ .run = try parseRun(allocator, map) });
        for ([_][]const u8{ "stdout", "stderr" }) |key| {
            if (map.get(key)) |v| {
                if (v != .null) try operations.append(allocator, .{ .read = .{ .id = key, .payload = try string(v) } });
            }
        }
        try operations.append(allocator, .{ .wait = try integer(map, "exit_code", 0) });
    }
    try validate(allocator, operations.items);
    return .{ .operations = try operations.toOwnedSlice(allocator), .proposals = try proposals.toOwnedSlice(allocator), .world = world, .debug = debug };
}

pub fn defaultConfig(allocator: Allocator) !Config {
    return parse(allocator, "{}");
}

test "legacy and modern sidecars produce equivalent operations" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const legacy = try parse(a, "{\"args\":[\"a\"],\"stdout\":\"ok\",\"exit_code\":3}");
    try std.testing.expectEqual(@as(usize, 3), legacy.operations.len);
    try std.testing.expectEqualStrings("a", legacy.operations[0].run.args[0]);
    try std.testing.expectEqualStrings("ok", legacy.operations[1].read.payload);
    try std.testing.expectEqual(@as(i64, 3), legacy.operations[2].wait);
    const modern = try parse(a, "{\"operations\":[{\"type\":\"run\"},{\"type\":\"write\",\"id\":\"stdin\",\"payload\":\"x\"},{\"type\":\"read\",\"payload\":\"x\"},{\"type\":\"wait\"}]}");
    try std.testing.expectEqual(@as(usize, 4), modern.operations.len);
    try std.testing.expectEqualStrings("stdout", modern.operations[2].read.id);
}

test "invalid operation streams and mixed sidecars fail" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.MixedConfigStyles, parse(a, "{\"args\":[],\"operations\":[]}"));
    try std.testing.expectError(error.WrongStreamType, parse(a, "{\"operations\":[{\"type\":\"run\"},{\"type\":\"read\",\"id\":\"stdin\"},{\"type\":\"wait\"}]}"));
    try std.testing.expectError(error.ProcessStillRunning, parse(a, "{\"operations\":[{\"type\":\"run\"}]}"));
    try std.testing.expectError(error.UnsupportedProtocol, parse(a, "{\"operations\":[{\"type\":\"run\"},{\"type\":\"connect\",\"protocol_type\":\"udp\"},{\"type\":\"wait\"}]}"));
    try std.testing.expectError(error.InvalidSignal, parse(a, "{\"operations\":[{\"type\":\"run\"},{\"type\":\"kill\",\"signal\":\"SIGBOGUS\"},{\"type\":\"wait\"}]}"));
    try std.testing.expectError(error.DuplicateStream, parse(a, "{\"operations\":[{\"type\":\"run\"},{\"type\":\"connect\",\"id\":\"stdout\"},{\"type\":\"wait\"}]}"));
    const socket = try parse(a, "{\"proposals\":[\"sockets\"],\"operations\":[{\"type\":\"run\"},{\"type\":\"connect\",\"id\":\"socket\"},{\"type\":\"send\",\"id\":\"socket\",\"payload\":\"hi\"},{\"type\":\"recv\",\"id\":\"socket\"},{\"type\":\"kill\",\"signal\":\"SIGINT\"},{\"type\":\"wait\"}]}");
    try std.testing.expectEqual(Proposal.sockets, socket.proposals[0]);
    try std.testing.expectEqualStrings("hi", socket.operations[2].send.payload);
}

test "HTTP requests, default configuration, and unsupported proposals" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const defaults = try defaultConfig(a);
    try std.testing.expectEqual(@as(usize, 2), defaults.operations.len);
    try std.testing.expectEqual(World.cli_command, defaults.world);
    const http = try parse(a,
        \\{"world":"wasi:http/service","proposals":["http"],"operations":[
        \\{"type":"run"},{"type":"request","method":"POST","path":"/echo",
        \\ "response":{"status":201,"headers":{"x-name":"value"},"body":"ok"}},
        \\{"type":"kill"},{"type":"wait"}]}
    );
    try std.testing.expectEqual(World.http_service, http.world);
    try std.testing.expectEqual(Proposal.http, http.proposals[0]);
    try std.testing.expectEqualStrings("POST", http.operations[1].request.method);
    try std.testing.expectEqualStrings("/echo", http.operations[1].request.path);
    try std.testing.expectEqual(@as(i64, 201), http.operations[1].request.status);
    try std.testing.expectEqualStrings("x-name", http.operations[1].request.headers[0].name);
    try std.testing.expectEqualStrings("value", http.operations[1].request.headers[0].value);
    try std.testing.expectEqualStrings("ok", http.operations[1].request.body);
    try std.testing.expectEqualStrings("SIGTERM", http.operations[2].kill);
    try std.testing.expectError(error.UnsupportedWorld, parse(a, "{\"world\":\"unknown\",\"proposals\":[]}"));
    try std.testing.expectError(error.UnsupportedProposal, parse(a, "{\"proposals\":[\"unknown\"]}"));
    try std.testing.expectError(error.ExpectedString, parse(a, "{\"operations\":[{\"type\":\"run\"},{\"type\":\"request\",\"response\":{\"body\":1}},{\"type\":\"wait\"}]}"));
}

test "pinned WASI sidecar corpus parses and validates" {
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    var upstream = cwd.openDir(io, "tests/wasi-testsuite/test-runner", .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    upstream.close(io);
    const roots = [_][]const u8{
        "tests/wasi-testsuite/tests/c/testsuite/wasm32-wasip1",
        "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip1",
        "tests/wasi-testsuite/tests/assemblyscript/testsuite/wasm32-wasip1",
        "tests/wasi-testsuite/tests/rust/testsuite/wasm32-wasip3",
    };
    for (roots) |root| {
        var dir = try cwd.openDir(io, root, .{ .iterate = true });
        defer dir.close(io);
        var count: usize = 0;
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (!std.mem.endsWith(u8, entry.name, ".json") or std.mem.eql(u8, entry.name, "manifest.json")) continue;
            var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer arena.deinit();
            const wasm = try std.fmt.allocPrint(arena.allocator(), "{s}.wasm", .{entry.name[0 .. entry.name.len - ".json".len]});
            dir.access(io, wasm, .{}) catch |err| switch (err) {
                error.FileNotFound => continue,
                else => return err,
            };
            const bytes = try dir.readFileAlloc(io, entry.name, arena.allocator(), .limited(1024 * 1024));
            _ = parse(arena.allocator(), bytes) catch |err| {
                std.debug.print("invalid fixture {s}/{s}: {s}\n", .{ root, entry.name, @errorName(err) });
                return err;
            };
            count += 1;
        }
        try std.testing.expect(count > 0);
    }
}
