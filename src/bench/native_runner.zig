//! Reusable guest-side execution/capture logic. No Linux, filesystem, compiler,
//! interpreter, hosted harness, or campaign/receipt generation belongs here.
const std = @import("std");
pub const aot = @import("../api/aot.zig");
const wasi = @import("minimal-wasi");
const adapter = @import("native-wasi").Adapter(aot);

pub const execution_lifecycle = .{
    .mode = "snapshot-replay",
    .reset_policy = "restore-post-start-snapshot",
    .reset_before = "each-steady-invocation",
    .reset_timing = "excluded-from-invocation",
    .reset_scope = &[_][]const u8{
        "globals",                         "invocation-output",
        "linear-memory-access-protection", "linear-memory-contents",
        "linear-memory-logical-size",      "passive-segment-drop-state",
        "table-entries-signatures",        "wasi-context",
    },
};

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
    stdout_complete: bool = true,
    stderr_complete: bool = true,
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
        self.stdout_complete = true;
        self.stderr_complete = true;
    }

    pub fn write(raw: ?*anyopaque, fd: u32, bytes: []const u8) wasi.WriteResult {
        const self: *Output = @ptrCast(@alignCast(raw.?));
        const buffer = switch (fd) {
            1 => &self.stdout,
            2 => &self.stderr,
            else => return .{ .errno = .badf },
        };
        const complete = if (fd == 1) &self.stdout_complete else &self.stderr_complete;
        const length = self.stdout.items.len + self.stderr.items.len;
        const remaining = @min(self.limit, self.fail_after orelse self.limit) -| length;
        const count = @min(remaining, bytes.len);
        buffer.appendSlice(self.allocator, bytes[0..count]) catch {
            self.failure = .io;
            complete.* = false;
            return .{ .errno = .io };
        };
        if (count != bytes.len) {
            self.failure = .nospc;
            complete.* = false;
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
    stdout_complete: bool = true,
    stderr_complete: bool = true,

    pub fn succeeded(self: Invocation) bool {
        return self.ticks != null and self.stdout_complete and self.stderr_complete and !self.output_failure and (std.mem.eql(u8, self.outcome, "returned") or
            (std.mem.eql(u8, self.outcome, "proc_exit") and self.exit_code == 0));
    }

    pub fn measurementErrors(self: Invocation, storage: *[3][]const u8) []const []const u8 {
        var count: usize = 0;
        if (self.output_failure) {
            storage[count] = "pending-output";
            count += 1;
        }
        if (self.timing_error != null) {
            storage[count] = "post-call-clock";
            count += 1;
        }
        if (!self.stdout_complete) {
            storage[count] = "stdout-copy";
            count += 1;
        }
        return storage[0..count];
    }
};

pub const Reset = struct {
    outcome: enum { completed, @"error" } = .@"error",
    elapsed_ticks: ?u64 = null,
    diagnostic: ?[]const u8 = null,
    timing_error: ?[]const u8 = null,
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
    try writer.writeAll(",\"stdout_complete\":");
    try std.json.Stringify.value(invocation.stdout_complete, .{}, writer);
    try writer.writeAll(",\"stderr_complete\":");
    try std.json.Stringify.value(invocation.stderr_complete, .{}, writer);
    try writer.writeAll(",\"measurement_errors\":");
    var errors: [3][]const u8 = undefined;
    try std.json.Stringify.value(invocation.measurementErrors(&errors), .{}, writer);
    try writer.writeAll(",\"stdout_base64\":");
    try writeBase64(writer, invocation.stdout);
    try writer.writeAll(",\"stderr_base64\":");
    try writeBase64(writer, invocation.stderr);
    try writer.writeAll("}\n");
}

pub fn writeBase64(writer: *std.Io.Writer, bytes: []const u8) !void {
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
    lifecycle_setup_ticks: u64 = 0,
    start_invocation: ?Invocation = null,

    pub const Capture = struct {
        output_limit: usize = 1024 * 1024,
        /// Persist module-start output/terminal before cleanup or buffer reuse.
        /// The writer must not allocate using the Session allocator.
        setup_evidence: ?*std.Io.Writer = null,
    };

    pub const Progress = struct {
        load_ticks: ?u64 = null,
        instantiate_ticks: ?u64 = null,
        lifecycle_setup_ticks: ?u64 = null,
    };

    pub fn create(allocator: std.mem.Allocator, native: aot.Platform, bytes: []const u8, args: []const []const u8, environment: []const []const u8, clock: wasi.Clock) !*Session {
        var progress: Progress = .{};
        return createTimed(allocator, native, bytes, args, environment, clock, &progress);
    }

    pub fn createTimed(allocator: std.mem.Allocator, native: aot.Platform, bytes: []const u8, args: []const []const u8, environment: []const []const u8, clock: wasi.Clock, progress: *Progress) !*Session {
        return createCaptured(allocator, native, bytes, args, environment, clock, progress, .{});
    }

    pub fn createCaptured(allocator: std.mem.Allocator, native: aot.Platform, bytes: []const u8, args: []const []const u8, environment: []const []const u8, clock: wasi.Clock, progress: *Progress, capture: Capture) !*Session {
        progress.* = .{};
        const self = try allocator.create(Session);
        self.* = .{
            .allocator = allocator,
            .native = native,
            .output = .{ .allocator = allocator, .limit = capture.output_limit },
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
        const instantiate_end_reading = native.monotonicNs();
        if (capture.setup_evidence) |writer| {
            if (self.start_invocation) |invocation| {
                try writeInvocationEvidence(writer, "module-start", 0, invocation);
                try writer.flush();
            }
        }
        const instantiate_end = try instantiate_end_reading;
        if (instantiate_end < instantiate_begin) return error.ClockFailed;
        progress.instantiate_ticks = instantiate_end - instantiate_begin;
        try initialized;
        self.instantiate_ticks = progress.instantiate_ticks.?;
        if (capture.setup_evidence != null) {
            if (self.output.failure != null or self.context.pendingWriteError(1) != null or self.context.pendingWriteError(2) != null)
                return error.OutputFailurePending;
            // The native guest's baseline includes module-start descriptor
            // changes, not merely the pre-start WASI options.
            self.wasi_options.descriptors = self.context.descriptors;
            self.output.clear();
            self.start_invocation = null;
        }
        const snapshot_begin = try native.monotonicNs();
        const snapshot = Snapshot.capture(allocator, self.instance.?);
        if (snapshot) |saved| self.snapshot = saved else |_| {}
        const snapshot_end = try native.monotonicNs();
        if (snapshot_end < snapshot_begin) return error.ClockFailed;
        progress.lifecycle_setup_ticks = snapshot_end - snapshot_begin;
        _ = try snapshot;
        self.lifecycle_setup_ticks = progress.lifecycle_setup_ticks.?;
        return self;
    }

    fn initialize(self: *Session) !void {
        self.wasi_options.output = .{ .userdata = &self.output, .write = Output.write };
        self.context = try wasi.Context.init(self.wasi_options);
        const bindings = adapter.imports(&self.context);
        try self.instance.?.instantiate(&bindings, .{});
        const started_result = self.instance.?.start();
        self.start_invocation = self.captureOutcome(started_result);
        const started = try started_result;
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
        const instance = self.instance.?;
        if (!instance.instantiated) return error.NotInstantiated;
        if (instance.active) return error.Busy;
        if (!instance.started or instance.start_failed) return error.StartFailed;
        if (self.output.failure != null or self.context.pendingWriteError(1) != null or self.context.pendingWriteError(2) != null)
            return error.OutputFailurePending;
        const context = try wasi.Context.init(self.wasi_options);
        try self.snapshot.?.restore(self.instance.?);
        self.context = context;
        self.output.clear();
    }

    /// Every attempted reset has its own outcome and optional measured time,
    /// including attempts that fail before any subsequent guest call.
    pub fn resetTimed(self: *Session) Reset {
        var result: Reset = .{};
        const begin = self.native.monotonicNs() catch |failure| {
            result.timing_error = @errorName(failure);
            return result;
        };
        const restored = self.reset();
        const end_reading = self.native.monotonicNs();
        if (restored) |_| {} else |failure| result.diagnostic = @errorName(failure);
        const end = end_reading catch |failure| {
            result.timing_error = @errorName(failure);
            return result;
        };
        if (end < begin) {
            result.timing_error = "ClockWentBackwards";
            return result;
        }
        result.elapsed_ticks = end - begin;
        if (result.diagnostic == null) result.outcome = .completed;
        return result;
    }

    /// Output is borrowed until reset, another invocation or deinit. No
    /// post-call allocation may erase the actual terminal/output evidence.
    pub fn invoke(self: *Session, entry: []const u8) !Invocation {
        const begin = try self.native.monotonicNs();
        const outcome = self.instance.?.call(entry, &.{}, &.{});
        const end_reading = self.native.monotonicNs();
        var result = self.captureOutcome(outcome);
        const end = end_reading catch |failure| {
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

    fn captureOutcome(self: *Session, outcome: anytype) Invocation {
        var result: Invocation = .{
            .ticks = null,
            .outcome = "error",
            .exit_code = null,
            .stdout = self.output.stdout.items,
            .stderr = self.output.stderr.items,
            .stdout_complete = self.output.stdout_complete,
            .stderr_complete = self.output.stderr_complete,
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
