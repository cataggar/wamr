//! Bounded, compiler-free, freestanding producer of the existing host v2 wire
//! format. The image integrator owns capabilities and independent attestations.
const std = @import("std");
const json = @import("native_json.zig");
const allocations = @import("native_allocations.zig");
pub const runner = @import("native_runner.zig");
pub const aot = runner.aot;
pub const wasi = @import("minimal-wasi");
pub const Writer = std.Io.Writer;

pub const Identity = struct {
    sha256: []const u8,
    bytes: u64,

    /// Storage must outlive the returned identity. Suitable for pre-image
    /// runtime archives, external compiler bytes, and canonical source bundles.
    pub fn fromBytes(bytes: []const u8, storage: *[64]u8) Identity {
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
        storage.* = std.fmt.bytesToHex(hash, .lower);
        return .{ .sha256 = storage, .bytes = bytes.len };
    }
};

pub const Source = struct {
    commit: []const u8,
    tree_sha256: []const u8,
    tracked_diff_sha256: []const u8,
};

pub const MemoryPolicy = struct {
    allocator_by_phase: struct { load: []const u8, instantiate: []const u8, first: []const u8, steady: []const u8 },
    page_policy: struct { reservation: []const u8, commitment: []const u8, release: []const u8 },
};

/// Independently generated from the actual build inputs, never the request.
/// Runtime identifies the actual linked pre-image consumer archive/object,
/// including producer code. Supply this declaration separately from the hashed
/// object; it must not contain its own hash. Source digests cover actual source
/// inputs (including dirty/untracked inputs), with the digest recipe in evidence.
pub const Build = struct {
    source: Source,
    runtime: Identity,
    compiler: struct { binary: Identity, source: Source, version: []const u8 },
    target_abi: []const u8,
    memory_policy: MemoryPolicy,
};

pub const Workload = struct {
    name: []const u8,
    wasm: []const u8,
    aot_bytes: []const u8,
};

/// Supplied over a trusted, independent deployment channel after image assembly.
/// No self-image hash is embedded, and a "hardware" label alone is not proof.
pub const Deployment = struct {
    receipt_json: []const u8,
    image: Identity,
    azure_sku: []const u8,
    azure_region: []const u8,
    configured_vm_ram_bytes: u64,
    qualification: enum { correctness_only, independently_qualified_hardware } = .correctness_only,
};

/// Actual guest observations, not configuration/SKU capacity or request fields.
pub const Cpu = struct { model: []const u8, active_count: u32 };

pub const Clock = struct {
    source: []const u8,
    /// Actual resolution of Platform.monotonic_ns; null means unavailable.
    /// Nanosecond units do not imply one-nanosecond resolution.
    resolution_ns: ?u64,
    wasi_clock: wasi.Clock,
};

pub const Footprint = struct {
    reserved_address_bytes: u64,
    /// Retained committed backing, including PROT_NONE snapshot-reset suffixes.
    /// Neither logical memory.size nor requested allocator bytes belong here.
    retained_committed_bytes: u64,
};

/// Counts the live code/linear-memory mappings served by this native Platform.
/// This intentionally does not claim whole-image or whole-guest RAM accounting.
pub const Pages = struct {
    context: ?*anyopaque,
    method: []const u8,
    sample: *const fn (?*anyopaque) anyerror!Footprint,
};

pub const Options = struct {
    request_json: []const u8,
    /// Digest delivered by the host control channel, separate from request JSON.
    config_sha256: []const u8,
    build: Build,
    deployment: Deployment,
    workloads: []const Workload,
    cpu: Cpu,
    native: aot.Platform,
    clock: Clock,
    pages: Pages,
    allocator: std.mem.Allocator,
    /// Fixed budget for parsing, identity reconciliation, and retained reports.
    /// Must not overlap inputs, output/evidence writer buffers, or guest memory.
    workspace: []u8,
    result: *Writer,
    /// Allocation-free transport; flushed before borrowed output can be reused.
    /// May be the same serial writer as result. Failure aborts the attempt.
    evidence: *Writer,
    max_request_bytes: usize = 256 * 1024,
    max_output_bytes: usize = 1024 * 1024,
    max_steady_invocations: usize = 10000,
};

const runtime_options = .{
    .optimize = @tagName(@import("builtin").mode),
    .bounds_checks = true,
    .wx_enforced = true,
    .import_checks = true,
    .trap_isolation = true,
    .stack_checks = true,
    .simd = false,
    .threads = false,
    .memory64 = false,
};
const pinned = [_]struct { name: []const u8, path: []const u8, sha256: []const u8 }{
    .{ .name = "coremark", .path = "tests/benchmarks/coremark/coremark_wasi.wasm", .sha256 = "f4b7591296ead10264e0f101f355bdf848865c31329325594e66fbabefec235b" },
    .{ .name = "coremark-nofp", .path = "tests/benchmarks/coremark/coremark_wasi_nofp.wasm", .sha256 = "24c0cc1bd52b641cf9e8ae74d1be188cba38d74cdb7ac18378de47382aab9541" },
    .{ .name = "compute", .path = "tests/benchmarks/loop-passes/unroll4.wasm", .sha256 = "6870b3373e4098117c82b6736d0ca7cbcc7d8d747fe87ca5b7d1ebf0e4d12890" },
    .{ .name = "memory", .path = "tests/benchmarks/loop-passes/iv_store.wasm", .sha256 = "b1979dd330c14d5f898b8a7c8c313e58f6521ad6eb7db9a6e7d7e78788ac6e72" },
};
const coremark_args: []const []const u8 = &.{ "0", "0", "0", "400000", "0" };

fn exact(value: json.Value, names: []const []const u8) !void {
    try json.require(value == .object and value.object.count() == names.len);
    for (names) |name| _ = try json.field(value, name);
}

fn equivalent(a: std.mem.Allocator, actual: anytype, expected: json.Value) !void {
    const encoded = try std.json.Stringify.valueAlloc(a, actual, .{});
    const parsed = try std.json.parseFromSlice(json.Value, a, encoded, .{});
    try json.require(try json.equal(a, parsed.value, expected));
}

fn token(value: []const u8) !void {
    try json.require(value.len > 0 and value.len <= 256);
    for (value) |c| try json.require(std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, "_.:+-", c) != null);
}

fn hex(value: []const u8, length: usize) !void {
    try json.require(value.len == length);
    for (value) |c| try json.require(std.ascii.isDigit(c) or (c >= 'a' and c <= 'f'));
}

fn identity(value: Identity) !void {
    try hex(value.sha256, 64);
    try json.require(value.bytes > 0);
}

fn source(value: Source) !void {
    try hex(value.commit, 40);
    try hex(value.tree_sha256, 64);
    try hex(value.tracked_diff_sha256, 64);
}

/// Bound nesting before the recursive JSON canonicalizer/stringifier is used.
fn parse(a: std.mem.Allocator, bytes: []const u8, limit: usize) !json.Value {
    if (bytes.len > limit) return error.InputTooLarge;
    var depth: usize = 0;
    var quoted = false;
    var escape = false;
    for (bytes) |c| {
        if (quoted) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') quoted = false;
        } else switch (c) {
            '"' => quoted = true,
            '{', '[' => {
                depth += 1;
                if (depth > 24) return error.InputTooDeep;
            },
            '}', ']' => depth -|= 1,
            else => {},
        }
    }
    return (try std.json.parseFromSlice(json.Value, a, bytes, .{})).value;
}

const Admitted = struct {
    config: json.Value,
    config_hash: []const u8,
    receipt_hash: []const u8,
    workload: Workload,
    aot_hash: []const u8,
    wasm_hash: []const u8,
    steady: usize,
    args: []const []const u8,
};

fn admit(o: Options, a: std.mem.Allocator) !Admitted {
    const request = try parse(a, o.request_json, o.max_request_bytes);
    try exact(request, &.{ "config", "config_sha256" });
    const config = try json.field(request, "config");
    try exact(config, &.{ "campaign_id", "run", "target", "workload", "steady_invocations", "execution_lifecycle", "phase_contract" });
    const config_hash = try json.digest(a, try json.canonical(a, config));
    try hex(o.config_sha256, 64);
    try json.require(std.mem.eql(u8, config_hash, o.config_sha256) and json.matches(try json.field(request, "config_sha256"), config_hash));
    try token(try json.string(config, "campaign_id"));
    try json.require(json.matches(try json.field(config, "phase_contract"), "wamr-embedding-v2"));
    const run_info = try json.field(config, "run");
    try exact(run_info, &.{ "run_id", "target", "workload", "phase", "attempt", "position" });
    try token(try json.string(run_info, "run_id"));
    try json.require(json.matches(try json.field(run_info, "target"), "unikraft"));
    try json.require(json.matches(try json.field(run_info, "phase"), "warmup") or json.matches(try json.field(run_info, "phase"), "measured"));
    try json.require(try json.number(run_info, "attempt") == 1 and try json.number(run_info, "position") > 0);
    const steady = try json.number(config, "steady_invocations");
    try json.require(steady > 0 and steady <= @min(o.max_steady_invocations, 10000));
    const target = try json.field(config, "target");
    try exact(target, &.{
        "os",            "runtime",              "compiler",    "source", "target_abi", "platform",        "options",       "image",
        "image_receipt", "image_receipt_sha256", "aot_modules", "mode",   "jit_preset", "compile_profile", "memory_policy", "execution_lifecycle",
    });
    try json.require(json.matches(try json.field(target, "mode"), "aot") and (try json.field(target, "jit_preset")) == .null);
    const receipt = try parse(a, o.deployment.receipt_json, o.max_request_bytes);
    try exact(receipt, &.{
        "schema_version",  "kind",        "evidence_kind", "os",                  "image",      "runtime",  "source",                  "compiler",
        "compile_profile", "options",     "memory_policy", "execution_lifecycle", "target_abi", "platform", "configured_vm_ram_bytes", "compiler_embedded",
        "runtime_linkage", "aot_modules",
    });
    try json.require(try json.number(receipt, "schema_version") == 2);
    try json.require(json.matches(try json.field(receipt, "kind"), "wamr-native-image-receipt"));
    try json.require(json.matches(try json.field(receipt, "evidence_kind"), "measurement"));
    try json.require(json.matches(try json.field(receipt, "os"), "unikraft"));
    try json.require(json.matches(try json.field(receipt, "compile_profile"), "unikraft-x86_64"));
    try equivalent(a, false, try json.field(receipt, "compiler_embedded"));
    try equivalent(a, "static", try json.field(receipt, "runtime_linkage"));
    const receipt_hash = try json.digest(a, o.deployment.receipt_json);
    try json.require(json.matches(try json.field(target, "image_receipt_sha256"), receipt_hash));
    try json.require(try json.equal(a, receipt, try json.field(target, "image_receipt")));
    for ([_][]const u8{ "os", "image", "runtime", "source", "compiler", "compile_profile", "options", "memory_policy", "execution_lifecycle", "target_abi", "platform", "aot_modules" }) |name|
        try json.require(try json.equal(a, try json.field(receipt, name), try json.field(target, name)));
    try source(o.build.source);
    try source(o.build.compiler.source);
    try identity(o.build.runtime);
    try identity(o.build.compiler.binary);
    try identity(o.deployment.image);
    try token(o.build.compiler.version);
    try token(o.build.target_abi);
    try equivalent(a, o.build.source, try json.field(receipt, "source"));
    try equivalent(a, o.build.source, try json.field(try json.field(receipt, "compiler"), "source"));
    try equivalent(a, o.build.compiler, try json.field(receipt, "compiler"));
    try equivalent(a, o.build.runtime, try json.field(receipt, "runtime"));
    try equivalent(a, o.deployment.image, try json.field(receipt, "image"));
    try equivalent(a, o.build.target_abi, try json.field(receipt, "target_abi"));
    try equivalent(a, runtime_options, try json.field(receipt, "options"));
    try equivalent(a, o.build.memory_policy, try json.field(receipt, "memory_policy"));
    inline for (std.meta.fields(@TypeOf(o.build.memory_policy.allocator_by_phase))) |f|
        try token(@field(o.build.memory_policy.allocator_by_phase, f.name));
    inline for (std.meta.fields(@TypeOf(o.build.memory_policy.page_policy))) |f|
        try token(@field(o.build.memory_policy.page_policy, f.name));
    for ([_]json.Value{ config, target, receipt }) |value|
        try equivalent(a, runner.execution_lifecycle, try json.field(value, "execution_lifecycle"));
    try token(o.cpu.model);
    try token(o.deployment.azure_sku);
    try token(o.deployment.azure_region);
    try json.require(o.cpu.active_count > 0 and o.deployment.configured_vm_ram_bytes > 0);
    try equivalent(a, platform(o), try json.field(receipt, "platform"));
    try equivalent(a, o.deployment.configured_vm_ram_bytes, try json.field(receipt, "configured_vm_ram_bytes"));
    try token(o.clock.source);
    try token(o.pages.method);
    const name = try json.string(run_info, "workload");
    const requested_workload = try json.field(config, "workload");
    var selected: ?Workload = null;
    for (o.workloads) |workload| {
        if (!std.mem.eql(u8, name, workload.name)) continue;
        try json.require(selected == null);
        selected = workload;
    }
    const workload = selected orelse return error.WorkloadNotEmbedded;
    const wasm_hash = try json.digest(a, workload.wasm);
    const aot_hash = try json.digest(a, workload.aot_bytes);
    try json.require(workload.aot_bytes.len > 0);
    try equivalent(a, Identity{ .sha256 = aot_hash, .bytes = workload.aot_bytes.len }, try json.field(try json.field(receipt, "aot_modules"), name));
    var args: ?[]const []const u8 = null;
    for (pinned) |fixture| {
        if (!std.mem.eql(u8, fixture.name, name)) continue;
        try json.require(std.mem.eql(u8, fixture.sha256, wasm_hash));
        args = if (std.mem.startsWith(u8, name, "coremark")) coremark_args else &.{};
        try equivalent(a, .{ .path = fixture.path, .sha256 = wasm_hash, .@"export" = "_start", .args = args.? }, requested_workload);
    }
    const guest_args = args orelse return error.UnsupportedWorkload;
    const argv = try a.alloc([]const u8, guest_args.len + 1);
    argv[0] = workload.name;
    @memcpy(argv[1..], guest_args);
    return .{
        .config = config,
        .config_hash = config_hash,
        .receipt_hash = receipt_hash,
        .workload = workload,
        .aot_hash = aot_hash,
        .wasm_hash = wasm_hash,
        .steady = @intCast(steady),
        .args = argv,
    };
}

fn platform(o: Options) struct { arch: []const u8, cpu_model: []const u8, active_cpu_count: u32, azure_sku: []const u8, azure_region: []const u8 } {
    return .{
        .arch = "x86_64",
        .cpu_model = o.cpu.model,
        .active_cpu_count = o.cpu.active_count,
        .azure_sku = o.deployment.azure_sku,
        .azure_region = o.deployment.azure_region,
    };
}

const Record = struct {
    invocation: runner.Invocation,
    phase: []const u8,

    pub fn jsonStringify(self: Record, jw: anytype) !void {
        var errors: [3][]const u8 = undefined;
        try jw.write(.{
            .phase = self.phase,
            .outcome = self.invocation.outcome,
            .exit_code = self.invocation.exit_code,
            .stdout_base64 = Base64{ .bytes = self.invocation.stdout },
            .stdout_complete = self.invocation.stdout_complete,
            .measurement_errors = self.invocation.measurementErrors(&errors),
        });
    }
};

const Base64 = struct {
    bytes: []const u8,
    pub fn jsonStringify(self: Base64, jw: anytype) !void {
        try jw.beginWriteRaw();
        try runner.writeBase64(jw.writer, self.bytes);
        jw.endWriteRaw();
    }
};
const ResetRecord = struct { before_invocation: usize, outcome: []const u8, elapsed_ticks: ?u64 };
const Sample = struct { stage: []const u8, reserved_address_bytes: u64, committed_bytes: u64 };

fn sample(pages: Pages, stage: []const u8) !Sample {
    const value = try pages.sample(pages.context);
    if (value.retained_committed_bytes == 0 or value.reserved_address_bytes < value.retained_committed_bytes)
        return error.InvalidFootprint;
    return .{ .stage = stage, .reserved_address_bytes = value.reserved_address_bytes, .committed_bytes = value.retained_committed_bytes };
}

fn allocationEvidence(o: Options, counter: *const allocations.Counter, stage: []const u8) !void {
    try o.evidence.print("WAMR_NATIVE_ALLOCATOR stage={s} method=caller-requested-bytes live={d} peak={d}\n", .{ stage, counter.live, counter.peak });
    try o.evidence.flush();
}

/// Exactly one request per call. No environment, files, subprocesses, compiler,
/// global allocator or scheduler. Errors never imply a successful result.
/// Runtime failures produce failed v2 records where possible. If admission,
/// report allocation or transport fails, private evidence already flushed is
/// authoritative and the caller must mark capture failed, not retry silently.
pub fn run(o: Options) !void {
    return runImpl(o) catch |failure| {
        // Best effort only: a broken transport cannot be repaired by inventing
        // a result. Preserve the original error for the enclosing capture.
        o.evidence.print("WAMR_NATIVE_FAILURE stage=producer error={s}\n", .{@errorName(failure)}) catch {};
        o.evidence.flush() catch {};
        return failure;
    };
}

fn runImpl(o: Options) !void {
    var workspace = std.heap.FixedBufferAllocator.init(o.workspace);
    const a = workspace.allocator();
    const input = try admit(o, a);
    const records = try a.alloc(Record, input.steady + 1);
    const ticks = try a.alloc(?u64, input.steady);
    const resets = try a.alloc(ResetRecord, input.steady);
    var record_count: usize = 0;
    var reset_count: usize = 0;
    var samples: [3]Sample = undefined;
    var sample_count: usize = 0;
    var memory_reliable = true;
    var progress: runner.Session.Progress = .{};
    var outcome: []const u8 = "error";
    var first_ticks: ?u64 = null;
    var exit_code: ?u32 = null;
    var counter: allocations.Counter = .{ .child = o.allocator };
    const resolution = if (o.clock.resolution_ns) |ns| (if (ns > 0 and ns <= 1_000_000_000) ns else null) else null;
    if (resolution != null) {
        if (runner.Session.createCaptured(counter.allocator(), o.native, input.workload.aot_bytes, input.args, &.{}, o.clock.wasi_clock, &progress, .{
            .output_limit = o.max_output_bytes,
            .setup_evidence = o.evidence,
        })) |session| {
            defer session.deinit();
            try allocationEvidence(o, &counter, "after_instantiation");
            samples[0] = try sample(o.pages, "after_instantiation");
            sample_count = 1;
            outcome = "success";
            var first_crc: ?u16 = null;
            for (0..input.steady + 1) |index| {
                if (index != 0) {
                    const event = session.resetTimed();
                    try o.evidence.writeAll("WAMR_NATIVE_RESET=");
                    try std.json.Stringify.value(.{
                        .before_invocation = index + 1,
                        .outcome = @tagName(event.outcome),
                        .elapsed_ticks = event.elapsed_ticks,
                        .diagnostic = event.diagnostic,
                        .timing_error = event.timing_error,
                    }, .{}, o.evidence);
                    try o.evidence.writeByte('\n');
                    try o.evidence.flush();
                    resets[reset_count] = .{ .before_invocation = index + 1, .outcome = @tagName(event.outcome), .elapsed_ticks = event.elapsed_ticks };
                    reset_count += 1;
                    if (event.outcome != .completed) {
                        outcome = "error";
                        memory_reliable = false;
                        break;
                    }
                }
                const invocation = session.invoke("_start") catch |failure| {
                    try o.evidence.print("WAMR_NATIVE_FAILURE stage=invocation-open error={s}\n", .{@errorName(failure)});
                    try o.evidence.flush();
                    outcome = "error";
                    break;
                };
                const phase = if (index == 0) "first" else "steady";
                try runner.writeInvocationEvidence(o.evidence, phase, index, invocation);
                try o.evidence.flush();
                try allocationEvidence(o, &counter, if (index == 0) "after_first" else "after_steady");
                var saved = invocation;
                saved.stdout = try a.dupe(u8, invocation.stdout);
                // Stderr is already losslessly persisted privately; v2 has no
                // public stderr field. Do not retain a dangling borrowed slice.
                saved.stderr = &.{};
                records[record_count] = .{ .invocation = saved, .phase = phase };
                record_count += 1;
                exit_code = invocation.exit_code;
                if (index == 0) first_ticks = invocation.ticks else ticks[index - 1] = invocation.ticks;
                if (index == 0) {
                    samples[1] = try sample(o.pages, "after_first");
                    sample_count = 2;
                } else {
                    samples[2] = try sample(o.pages, "after_steady");
                    sample_count = 3;
                }
                const crc = runner.coremarkCrc(invocation.stdout);
                if (index == 0) first_crc = crc;
                const valid = if (std.mem.startsWith(u8, input.workload.name, "coremark"))
                    crc != null and crc == first_crc and
                        std.mem.indexOf(u8, invocation.stdout, "Correct operation validated") != null and
                        std.mem.indexOf(u8, invocation.stdout, "ERROR!") == null
                else
                    std.mem.eql(u8, invocation.outcome, "returned") and invocation.stdout.len == 0;
                if (!invocation.succeeded() or !valid) {
                    outcome = if (std.mem.eql(u8, invocation.outcome, "trap")) "trap" else "error";
                    break;
                }
            }
        } else |failure| {
            try o.evidence.print("WAMR_NATIVE_FAILURE stage=setup error={s}\n", .{@errorName(failure)});
            try o.evidence.flush();
        }
    }
    try allocationEvidence(o, &counter, "after_release");
    if (counter.live != 0) return error.ResourceLeak;
    const memory = .{
        .image_sha256 = o.deployment.image.sha256,
        .configured_vm_ram_bytes = o.deployment.configured_vm_ram_bytes,
        .coverage = "partial-guest",
        .method = o.pages.method,
        .policy = o.build.memory_policy,
        .covered_regions = &[_][]const u8{ "native-code", "wasm-linear-memory" },
        .omitted_regions = &[_][]const u8{
            "runtime-allocator", "snapshot-allocator", "wasm-tables-globals", "producer-buffers",
            "guest-stacks",      "kernel",             "boot-image",          "other-allocations",
        },
        .samples = samples[0..sample_count],
    };
    const clock = .{ .source = o.clock.source, .unit = "ns", .ticks_per_second = @as(u64, 1_000_000_000), .resolution_ticks = resolution };
    // Correctness-only transport is deliberately unimportable as measurement.
    try o.result.writeAll(if (o.deployment.qualification == .independently_qualified_hardware) "WAMR_BENCH_RESULT=" else "WAMR_NATIVE_CHECK_RESULT=");
    try std.json.Stringify.value(.{
        .schema_version = 2,
        .kind = "wamr-native-benchmark-result",
        .evidence_kind = "measurement",
        .campaign_id = try json.field(input.config, "campaign_id"),
        .run_id = try json.field(try json.field(input.config, "run"), "run_id"),
        .config_sha256 = input.config_hash,
        .image_receipt_sha256 = input.receipt_hash,
        .observed = .{
            .image_sha256 = o.deployment.image.sha256,
            .runtime_sha256 = o.build.runtime.sha256,
            .aot_sha256 = input.aot_hash,
            .wasm_sha256 = input.wasm_hash,
            .platform = platform(o),
            .options = runtime_options,
            .mode = "aot",
            .jit_preset = @as(?u8, null),
            .compile_profile = "unikraft-x86_64",
        },
        .outcome = outcome,
        .exit_code = exit_code,
        .clock = if (resolution != null) clock else null,
        .phase_contract = "wamr-embedding-v2",
        .execution_lifecycle = runner.execution_lifecycle,
        .phases = .{
            .compile_ticks = @as(?u64, null),
            .load_ticks = progress.load_ticks,
            .instantiate_ticks = progress.instantiate_ticks,
            .lifecycle_setup_ticks = progress.lifecycle_setup_ticks,
            .first_invocation_ticks = first_ticks,
            .steady_state_ticks = ticks[0..record_count -| 1],
        },
        .invocations = records[0..record_count],
        .reset_events = resets[0..reset_count],
        .memory = if (memory_reliable and sample_count != 0) memory else null,
    }, .{}, o.result);
    try o.result.writeByte('\n');
    try o.result.flush();
}
