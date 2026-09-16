//! Platform-neutral sampler ownership, accounting and bounded serial framing.
const std = @import("std");
pub const sample = @import("native_jit_sample.zig");
pub const aot = @import("../api/aot.zig");
const allocations = @import("native_allocations.zig");

pub const max_record_bytes = 8192;
pub const Mode = enum { aot, fast, full };

/// Initialize and consume in place. Report slices point into this object; do not
/// copy or move it after initialize. No caller input or allocation is retained.
pub const Capture = struct {
    report: sample.Report = undefined,
    request_hash: [64]u8 = undefined,
    wasm_hash: [64]u8 = undefined,
    code_hash: [64]u8 = undefined,

    pub fn initialize(self: *Capture, request: []const u8, mode: Mode, wasm: []const u8, resolution_ns: u64) !void {
        self.request_hash = if (request.len == 64) request[0..64].* else @splat('0');
        self.wasm_hash = @splat('0');
        self.report = .{
            .request_sha256 = &self.request_hash,
            .mode = @tagName(mode),
            .compiler_embedded = mode != .aot,
            .wasm_sha256 = &self.wasm_hash,
            .wasm_bytes = wasm.len,
            .clock_resolution_ns = resolution_ns,
            .fuel_per_invocation = if (mode == .aot) null else 100_000,
            .failure_stage = "request",
            .failure = "InvalidArguments",
        };
        if (request.len != 64 or resolution_ns == 0) return error.InvalidArguments;
        for (self.request_hash) |c| if (!std.ascii.isHex(c)) return error.InvalidArguments;
        identify(wasm, &self.wasm_hash);
        self.report.failure_stage = null;
        self.report.failure = null;
    }

    pub fn identifyCode(self: *Capture, bytes: []const u8) void {
        identify(bytes, &self.code_hash);
        self.report.cwasm_sha256 = &self.code_hash;
        self.report.cwasm_bytes = bytes.len;
    }

    /// Write only after run returns (including failure). The caller owns serial
    /// I/O and must propagate incomplete writes; there is no console or boot ABI.
    pub fn writeRecord(self: *const Capture, writer: *std.Io.Writer) !void {
        var buffer: [max_record_bytes]u8 = undefined;
        var record = std.Io.Writer.fixed(&buffer);
        try record.writeAll("WAMR_JIT_SAMPLE=");
        try std.json.Stringify.value(self.report, .{}, &record);
        try record.writeByte('\n');
        try writer.writeAll(record.buffered());
    }
};

fn identify(bytes: []const u8, output: *[64]u8) void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    output.* = std.fmt.bytesToHex(digest, .lower);
}

pub const Accounting = struct {
    counter: allocations.Counter,
    pages: Pages,

    pub fn init(allocator: std.mem.Allocator, native: aot.Platform) Accounting {
        return .{ .counter = .{ .child = allocator }, .pages = .{ .native = native } };
    }

    pub fn finish(self: *Accounting, capture: *Capture) void {
        capture.report.caller_peak_bytes = self.counter.peak;
        capture.report.caller_live_after_teardown = self.counter.live;
        capture.report.reserved_after_teardown = self.pages.reserved;
        if (self.counter.live != 0 or self.pages.reserved != 0) {
            capture.report.failure_stage = "teardown";
            capture.report.failure = "IncompleteTeardown";
        }
    }
};

/// Counts only reservations requested through this invocation's Platform.
/// Successful unmap is the adapter's infallible release contract, not a physical
/// memory observation. Independent native memory observers remain necessary.
const Pages = struct {
    native: aot.Platform,
    reserved: usize = 0,

    pub fn platform(self: *Pages) aot.Platform {
        return .{
            .context = self,
            .page_size = self.native.page_size,
            .reserve = reserve,
            .commit = commit,
            .protect = protect,
            .unmap = unmap,
            .monotonic_ns = clock,
        };
    }
    fn cast(ctx: *anyopaque) *Pages {
        return @ptrCast(@alignCast(ctx));
    }
    fn reserve(ctx: *anyopaque, size: usize) aot.PlatformError![*]align(4096) u8 {
        const self = cast(ctx);
        const total = std.math.add(usize, self.reserved, size) catch return error.OutOfMemory;
        const address = try self.native.reserve(self.native.context, size);
        self.reserved = total;
        return address;
    }
    fn commit(ctx: *anyopaque, address: [*]align(4096) u8, size: usize) aot.PlatformError!void {
        const p = cast(ctx).native;
        return p.commit(p.context, address, size);
    }
    fn protect(ctx: *anyopaque, address: [*]align(4096) u8, size: usize, protection: aot.platform.Protection) aot.PlatformError!void {
        const p = cast(ctx).native;
        return p.protect(p.context, address, size, protection);
    }
    fn unmap(ctx: *anyopaque, address: [*]align(4096) u8, size: usize) void {
        const self = cast(ctx);
        self.native.unmap(self.native.context, address, size);
        self.reserved -= size;
    }
    fn clock(ctx: *anyopaque) aot.PlatformError!u64 {
        return cast(ctx).native.monotonicNs();
    }
};
