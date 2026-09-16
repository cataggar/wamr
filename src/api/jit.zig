//! Explicit in-memory native compilation. No loader fallback or hosted services.
const std = @import("std");
const pipeline = @import("../component/aot_compile.zig");
const control = @import("../compiler/control.zig");
pub const aot = @import("aot.zig");
pub const PassPreset = @import("../compiler/ir/passes.zig").PassPreset;
pub const VerifyMode = @import("../compiler/ir/verifier.zig").VerifyMode;
pub const Metrics = control.Metrics;
/// The native loader requires all of these caps for a fuel-profile artifact.
pub const runtime_options: aot.Options = .{
    .max_run_fuel = 100_000,
    .max_heap_bytes = 16 * 1024 * 1024,
    .max_reserved_bytes = 32 * 1024 * 1024,
    .max_code_bytes = 4 * 1024 * 1024,
};
pub const Error = pipeline.PrecompileError || error{ InvalidLimits, CompilerMemoryLimitExceeded, InputLimitExceeded, CodeLimitExceeded };

pub const Options = struct {
    preset: PassPreset = .fast,
    verify: VerifyMode = .after_each_pass,
    max_input_bytes: usize = 1024 * 1024,
    max_compiler_bytes: usize = 64 * 1024 * 1024,
    max_code_bytes: usize = 4 * 1024 * 1024,
    max_polls: u64 = 100_000,
    max_functions: usize = 1024,
    max_function_bytes: usize = 65536,
    max_locals: usize = 4096,
    max_blocks_per_function: usize = 2048,
    max_instructions_per_function: usize = 32768,
    context: ?*anyopaque = null,
    cancelled: ?*const fn (?*anyopaque) bool = null,
    monotonic_ns: ?*const fn (?*anyopaque) error{ClockFailed}!u64 = null,
    /// Cooperative deadline, checked at documented poll sites, NOT preemption.
    deadline_ns: ?u64 = null,
};

pub const Artifact = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    metrics: Metrics,
    compiler_peak_bytes: usize,
    compiler_retained_bytes: usize,
    polls: u64,

    pub fn deinit(self: *Artifact) void {
        self.allocator.free(self.bytes);
        self.* = undefined;
    }
};

pub fn compile(allocator: std.mem.Allocator, wasm: []const u8, options: Options) Error!Artifact {
    if (!@import("../config.zig").unikraft_jit) return error.UnsupportedOptions;
    if (options.max_input_bytes == 0 or options.max_compiler_bytes == 0 or
        options.max_code_bytes == 0 or options.max_polls == 0 or options.max_functions == 0 or
        options.max_function_bytes == 0 or options.max_locals == 0 or
        options.max_blocks_per_function == 0 or options.max_instructions_per_function == 0 or
        (options.deadline_ns != null and options.monotonic_ns == null))
        return error.InvalidLimits;
    if (wasm.len > options.max_input_bytes) return error.InputLimitExceeded;
    var budget: control.Control = .{
        .context = options.context,
        .cancelled = options.cancelled,
        .monotonic_ns = options.monotonic_ns,
        .deadline_ns = options.deadline_ns,
        .remaining_polls = options.max_polls,
        .max_blocks_per_function = options.max_blocks_per_function,
        .max_instructions_per_function = options.max_instructions_per_function,
        .max_functions = options.max_functions,
        .max_function_bytes = options.max_function_bytes,
        .max_locals = options.max_locals,
        .max_code_bytes = options.max_code_bytes,
    };
    var heap: CappedAllocator = .{ .parent = allocator, .limit = options.max_compiler_bytes, .control = &budget };
    var metrics: Metrics = .{};
    // Bounded scratch ownership also reclaims partially-mutated IR on an
    // interrupted pass. The cap counts retained arena chunks, not logical frees.
    const bytes = blk: {
        var scratch = std.heap.ArenaAllocator.init(heap.allocator());
        defer scratch.deinit();
        const generated = pipeline.compileCoreWasm(scratch.allocator(), wasm, .{
            .target_arch = .x86_64,
            .pass_preset = options.preset,
            .verify_mode = options.verify,
            .control = &budget,
            .metrics = &metrics,
        }) catch |err| return heap.failure orelse err;
        if (heap.failure) |err| return err;
        break :blk heap.allocator().dupe(u8, generated) catch |err| return heap.failure orelse err;
    };
    errdefer allocator.free(bytes);
    if (heap.failure) |err| return err;
    if (metrics.code_bytes > options.max_code_bytes) return error.CodeLimitExceeded;
    std.debug.assert(heap.live == bytes.len);
    return .{
        .allocator = allocator,
        .bytes = bytes,
        .metrics = metrics,
        .compiler_peak_bytes = heap.peak,
        .compiler_retained_bytes = heap.live,
        .polls = budget.polls,
    };
}

const CappedAllocator = struct {
    parent: std.mem.Allocator,
    limit: usize,
    control: *control.Control,
    live: usize = 0,
    peak: usize = 0,
    failure: ?Error = null,

    fn allocator(self: *CappedAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn admit(self: *CappedAllocator, growth: usize) bool {
        if (self.failure != null) return false;
        self.control.poll() catch |err| {
            self.failure = err;
            return false;
        };
        if (growth > self.limit - self.live) {
            self.failure = error.CompilerMemoryLimitExceeded;
            return false;
        }
        return true;
    }
    fn add(self: *CappedAllocator, amount: usize) void {
        self.live += amount;
        self.peak = @max(self.peak, self.live);
    }
    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *CappedAllocator = @ptrCast(@alignCast(ctx));
        if (!self.admit(len)) return null;
        const result = self.parent.rawAlloc(len, alignment, ra) orelse {
            self.failure = error.OutOfMemory;
            return null;
        };
        self.add(len);
        return result;
    }
    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *CappedAllocator = @ptrCast(@alignCast(ctx));
        if (new_len > memory.len and !self.admit(new_len - memory.len)) return false;
        if (!self.parent.rawResize(memory, alignment, new_len, ra)) return false;
        self.live -= memory.len;
        self.add(new_len);
        return true;
    }
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *CappedAllocator = @ptrCast(@alignCast(ctx));
        if (new_len > memory.len and !self.admit(new_len - memory.len)) return null;
        const result = self.parent.rawRemap(memory, alignment, new_len, ra) orelse return null;
        self.live -= memory.len;
        self.add(new_len);
        return result;
    }
    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *CappedAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(memory, alignment, ra);
        self.live -= memory.len;
    }
};
