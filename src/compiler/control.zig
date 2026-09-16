//! Caller-owned cooperative compilation budget, independent of OS facilities.
const std = @import("std");

pub const Error = error{ CompileCancelled, CompileLimitExceeded, CodeLimitExceeded, ClockFailed };

pub const Control = struct {
    context: ?*anyopaque = null,
    cancelled: ?*const fn (?*anyopaque) bool = null,
    monotonic_ns: ?*const fn (?*anyopaque) error{ClockFailed}!u64 = null,
    deadline_ns: ?u64 = null,
    remaining_polls: u64,
    max_blocks_per_function: usize = 2048,
    max_instructions_per_function: usize = 32768,
    max_functions: usize = 1024,
    max_function_bytes: usize = 65536,
    max_locals: usize = 4096,
    max_code_bytes: usize = 4 * 1024 * 1024,
    polls: u64 = 0,
    last_timestamp: ?u64 = null,

    pub fn poll(self: *Control) Error!void {
        if (self.remaining_polls == 0) return error.CompileLimitExceeded;
        self.remaining_polls -= 1;
        self.polls += 1;
        if (self.cancelled) |callback| if (callback(self.context)) return error.CompileCancelled;
        if (self.deadline_ns) |deadline| {
            const timestamp = try self.now();
            if (timestamp >= deadline) return error.CompileCancelled;
        }
    }

    pub fn now(self: *Control) Error!u64 {
        const clock = self.monotonic_ns orelse return error.ClockFailed;
        const timestamp = try clock(self.context);
        if (timestamp == std.math.maxInt(u64)) return error.ClockFailed;
        if (self.last_timestamp) |previous| if (timestamp < previous) return error.ClockFailed;
        self.last_timestamp = timestamp;
        return timestamp;
    }

    pub fn function(self: *Control, func: anytype) Error!void {
        try self.poll();
        if (func.blocks.items.len > self.max_blocks_per_function) return error.CompileLimitExceeded;
        var count: usize = 0;
        for (func.blocks.items) |block| {
            count = std.math.add(usize, count, block.instructions.items.len) catch return error.CompileLimitExceeded;
            if (count > self.max_instructions_per_function) return error.CompileLimitExceeded;
        }
    }
};

pub const Metrics = struct {
    /// Null means no clock supplied, not a fabricated zero-duration sample.
    parse_ns: ?u64 = null,
    lower_ns: ?u64 = null,
    optimize_ns: ?u64 = null,
    codegen_ns: ?u64 = null,
    emit_ns: ?u64 = null,
    code_bytes: usize = 0,
};
