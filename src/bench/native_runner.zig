//! Reusable guest-side execution/capture logic. No Linux, filesystem, compiler,
//! interpreter, hosted harness, or campaign/receipt generation belongs here.
const std = @import("std");
pub const aot = @import("../api/aot.zig");
const wasi = @import("minimal-wasi");
const adapter = @import("native-wasi").Adapter(aot);

/// CRC-only gate for the pinned 2K fixtures. Full timing/output qualification
/// remains in native_benchmark.py; sub-ten-second checks are not measurements.
pub fn coremarkCrc(stdout: []const u8) ?u16 {
    const markers = [_][]const u8{ "seedcrc", "[0]crclist", "[0]crcmatrix", "[0]crcstate", "[0]crcfinal" };
    const expected = [_]?u16{ 0xe9f5, 0xe714, 0x1fd7, 0x8e3a, null };
    var found: [markers.len]?u16 = @splat(null);
    var lines = std.mem.splitScalar(u8, stdout, '\n');
    while (lines.next()) |line| {
        for (markers, 0..) |marker, index| {
            if (!std.mem.startsWith(u8, line, marker)) continue;
            if (found[index] != null) return null;
            const suffix = std.mem.trim(u8, line[marker.len..], " \t\r");
            if (!std.mem.startsWith(u8, suffix, ":")) return null;
            const value = std.mem.trim(u8, suffix[1..], " \t");
            if (value.len != 6 or !std.mem.startsWith(u8, value, "0x")) return null;
            found[index] = std.fmt.parseInt(u16, value[2..], 16) catch return null;
            if (expected[index]) |crc| if (found[index].? != crc) return null;
        }
    }
    for (found) |crc| if (crc == null) return null;
    return found[4];
}

pub const Output = struct {
    allocator: std.mem.Allocator,
    stdout: std.ArrayList(u8) = .empty,
    stderr: std.ArrayList(u8) = .empty,
    failure: ?wasi.Errno = null,
    limit: usize = 1024 * 1024,
    /// Test injection still consumes a genuine prefix and preserves the errno.
    fail_after: ?usize = null,

    pub fn deinit(self: *Output) void {
        self.stdout.deinit(self.allocator);
        self.stderr.deinit(self.allocator);
    }

    pub fn clear(self: *Output) void {
        self.stdout.clearRetainingCapacity();
        self.stderr.clearRetainingCapacity();
        self.failure = null;
    }

    pub fn write(raw: ?*anyopaque, fd: u32, bytes: []const u8) wasi.WriteResult {
        const self: *Output = @ptrCast(@alignCast(raw.?));
        const buffer = switch (fd) {
            1 => &self.stdout,
            2 => &self.stderr,
            else => return .{ .errno = .badf },
        };
        const length = self.stdout.items.len + self.stderr.items.len;
        const remaining = @min(self.limit, self.fail_after orelse self.limit) -| length;
        const count = @min(remaining, bytes.len);
        buffer.appendSlice(self.allocator, bytes[0..count]) catch {
            self.failure = .io;
            return .{ .errno = .io };
        };
        if (count != bytes.len) {
            self.failure = .nospc;
            return .{ .written = count, .errno = .nospc };
        }
        return .{ .written = count };
    }
};

pub const Invocation = struct {
    ticks: ?u64,
    outcome: []const u8,
    exit_code: ?u32,
    stdout: []const u8,
    stderr: []const u8,
    diagnostic: ?[]const u8 = null,
    output_failure: bool = false,
    timing_error: ?[]const u8 = null,

    pub fn succeeded(self: Invocation) bool {
        return self.ticks != null and !self.output_failure and (std.mem.eql(u8, self.outcome, "returned") or
            (std.mem.eql(u8, self.outcome, "proc_exit") and self.exit_code == 0));
    }
};

/// Allocation-free private evidence, written before any report serialization.
/// Stdout/stderr are lossless even when they are not valid UTF-8.
pub fn writeInvocationEvidence(writer: *std.Io.Writer, phase: []const u8, index: usize, invocation: Invocation) !void {
    try writer.writeAll("WAMR_NATIVE_INVOCATION={\"schema_version\":1,\"kind\":\"wamr-native-invocation-evidence\",\"phase\":");
    try std.json.Stringify.value(phase, .{}, writer);
    try writer.writeAll(",\"index\":");
    try std.json.Stringify.value(index, .{}, writer);
    try writer.writeAll(",\"outcome\":");
    try std.json.Stringify.value(invocation.outcome, .{}, writer);
    try writer.writeAll(",\"exit_code\":");
    try std.json.Stringify.value(invocation.exit_code, .{}, writer);
    try writer.writeAll(",\"ticks\":");
    try std.json.Stringify.value(invocation.ticks, .{}, writer);
    try writer.writeAll(",\"timing_error\":");
    try std.json.Stringify.value(invocation.timing_error, .{}, writer);
    try writer.writeAll(",\"diagnostic\":");
    try std.json.Stringify.value(invocation.diagnostic, .{}, writer);
    try writer.writeAll(",\"output_failure\":");
    try std.json.Stringify.value(invocation.output_failure, .{}, writer);
    try writer.writeAll(",\"stdout_base64\":");
    try writeBase64(writer, invocation.stdout);
    try writer.writeAll(",\"stderr_base64\":");
    try writeBase64(writer, invocation.stderr);
    try writer.writeAll("}\n");
}

fn writeBase64(writer: *std.Io.Writer, bytes: []const u8) !void {
    try writer.writeByte('"');
    var at: usize = 0;
    var encoded: [64]u8 = undefined;
    while (at < bytes.len) {
        const count = @min(bytes.len - at, 48);
        try writer.writeAll(std.base64.standard.Encoder.encode(&encoded, bytes[at..][0..count]));
        at += count;
    }
    try writer.writeByte('"');
}

pub const Session = struct {
    allocator: std.mem.Allocator,
    native: aot.Platform,
    output: Output,
    context: wasi.Context,
    instance: ?*aot.Instance = null,
    wasi_options: wasi.Options,
    snapshot: ?Snapshot = null,
    load_ticks: u64 = 0,
    instantiate_ticks: u64 = 0,

    pub const Progress = struct {
        load_ticks: ?u64 = null,
        instantiate_ticks: ?u64 = null,
    };

    pub fn create(allocator: std.mem.Allocator, native: aot.Platform, bytes: []const u8, args: []const []const u8, environment: []const []const u8, clock: wasi.Clock) !*Session {
        var progress: Progress = .{};
        return createTimed(allocator, native, bytes, args, environment, clock, &progress);
    }

    pub fn createTimed(allocator: std.mem.Allocator, native: aot.Platform, bytes: []const u8, args: []const []const u8, environment: []const []const u8, clock: wasi.Clock, progress: *Progress) !*Session {
        progress.* = .{};
        const self = try allocator.create(Session);
        self.* = .{
            .allocator = allocator,
            .native = native,
            .output = .{ .allocator = allocator },
            .context = undefined,
            .wasi_options = .{ .args = args, .environment = environment, .output = null, .clock = clock },
        };
        errdefer self.deinit();
        const load_begin = try native.monotonicNs();
        const loaded = aot.Instance.loadModule(allocator, native, bytes, .{});
        if (loaded) |instance| self.instance = instance else |_| {}
        const load_end = try native.monotonicNs();
        if (load_end < load_begin) return error.ClockFailed;
        progress.load_ticks = load_end - load_begin;
        self.instance = try loaded;
        self.load_ticks = progress.load_ticks.?;
        const instantiate_begin = try native.monotonicNs();
        const initialized = self.initialize();
        const instantiate_end = try native.monotonicNs();
        if (instantiate_end < instantiate_begin) return error.ClockFailed;
        progress.instantiate_ticks = instantiate_end - instantiate_begin;
        try initialized;
        self.instantiate_ticks = progress.instantiate_ticks.?;
        const snapshot = try Snapshot.capture(allocator, self.instance.?);
        self.snapshot = snapshot;
        return self;
    }

    fn initialize(self: *Session) !void {
        self.wasi_options.output = .{ .userdata = &self.output, .write = Output.write };
        self.context = try wasi.Context.init(self.wasi_options);
        const bindings = adapter.imports(&self.context);
        try self.instance.?.instantiate(&bindings, .{});
        const started = try self.instance.?.start();
        if (started != .returned) return error.StartFailed;
    }

    pub fn deinit(self: *Session) void {
        if (self.snapshot) |*snapshot| snapshot.arena.deinit();
        if (self.instance) |instance| instance.deinit();
        self.output.deinit();
        self.allocator.destroy(self);
    }

    /// Restore this same instance's initialized guest state, never just clear
    /// proc_exit and assume a libc command is reentrant. Linear memory growth
    /// is revoked through the runtime; table growth remains unsupported here.
    /// Reset/capture cost is outside invocation timing and must be identified
    /// as snapshot-reset semantics, not warm libc/process reuse.
    pub fn reset(self: *Session) !void {
        try self.snapshot.?.restore(self.instance.?);
        self.context = try wasi.Context.init(self.wasi_options);
        self.output.clear();
    }

    /// Output is borrowed until reset, another invocation or deinit. No
    /// post-call allocation may erase the actual terminal/output evidence.
    pub fn invoke(self: *Session, entry: []const u8) !Invocation {
        const begin = try self.native.monotonicNs();
        const outcome = self.instance.?.call(entry, &.{}, &.{});
        var result: Invocation = .{
            .ticks = null,
            .outcome = "error",
            .exit_code = null,
            .stdout = self.output.stdout.items,
            .stderr = self.output.stderr.items,
        };
        if (outcome) |terminal| {
            switch (terminal) {
                .returned => {
                    result.outcome = "returned";
                },
                .exit => |code| {
                    result.outcome = "proc_exit";
                    result.exit_code = code;
                },
                .trap => |trap| {
                    result.outcome = "trap";
                    result.diagnostic = @tagName(trap);
                },
                .host_error => |failure| result.diagnostic = @errorName(failure),
            }
        } else |failure| result.diagnostic = @errorName(failure);
        if (self.output.failure != null or self.context.pendingWriteError(1) != null or self.context.pendingWriteError(2) != null) {
            result.output_failure = true;
            if (result.diagnostic == null) result.diagnostic = "output-callback-failure";
        }
        const end = self.native.monotonicNs() catch |failure| {
            result.timing_error = @errorName(failure);
            return result;
        };
        if (end < begin) {
            result.timing_error = "ClockWentBackwards";
            return result;
        }
        result.ticks = end - begin;
        return result;
    }
};

const Snapshot = struct {
    const Table = struct { pointers: []usize, signatures: []u32, size: u32 };
    arena: std.heap.ArenaAllocator,
    memory: []u8,
    globals: []u64,
    tables: []Table,
    dropped_data: []bool,
    dropped_elements: []bool,

    fn capture(allocator: std.mem.Allocator, instance: *aot.Instance) !Snapshot {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        const memory = try a.dupe(u8, instance.memory());
        const globals = try a.dupe(u64, instance.globals);
        const tables = try a.alloc(Table, instance.tables.len);
        for (instance.tables, tables) |source, *target| target.* = .{
            .pointers = try a.dupe(usize, source.pointers),
            .signatures = try a.dupe(u32, source.signatures),
            .size = source.size,
        };
        const dropped_data = try a.dupe(bool, instance.dropped_data);
        const dropped_elements = try a.dupe(bool, instance.dropped_elements);
        return .{
            .arena = arena,
            .memory = memory,
            .globals = globals,
            .tables = tables,
            .dropped_data = dropped_data,
            .dropped_elements = dropped_elements,
        };
    }

    fn restore(self: *const Snapshot, instance: *aot.Instance) !void {
        if (instance.active) return error.Busy;
        for (instance.tables, self.tables) |table, saved|
            if (table.size != saved.size) return error.UnsupportedReset;
        try instance.restoreMemory(self.memory);
        @memcpy(instance.globals, self.globals);
        for (instance.tables, self.tables) |table, saved| {
            @memcpy(table.pointers, saved.pointers);
            @memcpy(table.signatures, saved.signatures);
        }
        @memcpy(instance.dropped_data, self.dropped_data);
        @memcpy(instance.dropped_elements, self.dropped_elements);
    }
};
