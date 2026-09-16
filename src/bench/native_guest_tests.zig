//! Test-only, explicitly synthetic build/deployment envelopes around real
//! compiler output. These never qualify an image or produce measurement receipts.
const std = @import("std");
const guest = @import("native_guest.zig");
const json = @import("native_json.zig");
const linux = @import("native_linux.zig");
const fixtures = @import("fixtures");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

const Envelope = struct {
    arena: std.heap.ArenaAllocator,
    options: guest.Options,

    fn init(name: []const u8, wasm: []const u8, native_bytes: []const u8, pages: *linux.Pages, workspace: []u8, result: *guest.Writer, evidence: *guest.Writer) !Envelope {
        return initWithAllocator(std.testing.allocator, name, wasm, native_bytes, pages, workspace, result, evidence);
    }

    fn initWithAllocator(allocator: std.mem.Allocator, name: []const u8, wasm: []const u8, native_bytes: []const u8, pages: *linux.Pages, workspace: []u8, result: *guest.Writer, evidence: *guest.Writer) !Envelope {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        const build: guest.Build = .{
            .source = .{
                .commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                .tree_sha256 = try json.digest(a, @embedFile("native_guest.zig")),
                .tracked_diff_sha256 = try json.digest(a, ""),
            },
            .runtime = .{ .sha256 = try json.digest(a, "synthetic-runtime"), .bytes = 17 },
            .compiler = .{
                .binary = .{ .sha256 = try json.digest(a, "synthetic-compiler"), .bytes = 18 },
                .source = undefined,
                .version = "synthetic-test-only",
            },
            .target_abi = "synthetic-sysv-test-only",
            .memory_policy = .{
                .allocator_by_phase = .{
                    .load = "test-counted-allocator",
                    .instantiate = "test-counted-allocator",
                    .first = "test-counted-allocator",
                    .steady = "test-counted-allocator",
                },
                .page_policy = .{ .reservation = "test-mmap", .commitment = "test-mprotect", .release = "test-munmap" },
            },
        };
        var actual_build = build;
        actual_build.compiler.source = actual_build.source;
        const image: guest.Identity = .{ .sha256 = try json.digest(a, "synthetic-image"), .bytes = 15 };
        const platform = .{
            .arch = "x86_64",
            .cpu_model = "synthetic-test-cpu",
            .active_cpu_count = 1,
            .azure_sku = "synthetic-no-deployment",
            .azure_region = "synthetic-no-deployment",
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
        var modules = json.object(a);
        try json.put(a, &modules, name, try value(a, guest.Identity{ .sha256 = try json.digest(a, native_bytes), .bytes = native_bytes.len }));
        const receipt = try value(a, .{
            .schema_version = 2,
            .kind = "wamr-native-image-receipt",
            .evidence_kind = "measurement",
            .os = "unikraft",
            .image = image,
            .runtime = actual_build.runtime,
            .source = actual_build.source,
            .compiler = actual_build.compiler,
            .compile_profile = "unikraft-x86_64",
            .options = runtime_options,
            .memory_policy = actual_build.memory_policy,
            .execution_lifecycle = guest.runner.execution_lifecycle,
            .target_abi = actual_build.target_abi,
            .platform = platform,
            .configured_vm_ram_bytes = 1024 * 1024 * 1024,
            .compiler_embedded = false,
            .runtime_linkage = "static",
            .aot_modules = modules,
        });
        const receipt_bytes = try std.json.Stringify.valueAlloc(a, receipt, .{});
        var target = json.object(a);
        for ([_][]const u8{
            "os",            "image",               "runtime",    "source",   "compiler",    "compile_profile", "options",
            "memory_policy", "execution_lifecycle", "target_abi", "platform", "aot_modules",
        }) |field| try json.put(a, &target, field, try json.field(receipt, field));
        try json.put(a, &target, "image_receipt", receipt);
        try json.put(a, &target, "image_receipt_sha256", json.str(try json.digest(a, receipt_bytes)));
        try json.put(a, &target, "mode", json.str("aot"));
        try json.put(a, &target, "jit_preset", .null);
        const path = if (std.mem.eql(u8, name, "compute"))
            "tests/benchmarks/loop-passes/unroll4.wasm"
        else if (std.mem.eql(u8, name, "memory"))
            "tests/benchmarks/loop-passes/iv_store.wasm"
        else if (std.mem.eql(u8, name, "coremark"))
            "tests/benchmarks/coremark/coremark_wasi.wasm"
        else
            "tests/benchmarks/coremark/coremark_wasi_nofp.wasm";
        const args: []const []const u8 = if (std.mem.startsWith(u8, name, "coremark")) &.{ "0", "0", "0", "400000", "0" } else &.{};
        const config = try value(a, .{
            .campaign_id = "synthetic-test-only",
            .run = .{ .run_id = "run-0001", .target = "unikraft", .workload = name, .phase = "measured", .attempt = 1, .position = 1 },
            .target = target,
            .workload = .{ .path = path, .sha256 = try json.digest(a, wasm), .@"export" = "_start", .args = args },
            .steady_invocations = 2,
            .execution_lifecycle = guest.runner.execution_lifecycle,
            .phase_contract = "wamr-embedding-v2",
        });
        const config_hash = try json.digest(a, try json.canonical(a, config));
        const request = try std.json.Stringify.valueAlloc(a, .{ .config = config, .config_sha256 = config_hash }, .{});
        const workloads = try a.alloc(guest.Workload, 1);
        workloads[0] = .{ .name = name, .wasm = wasm, .aot_bytes = native_bytes };
        return .{ .arena = arena, .options = .{
            .request_json = request,
            .config_sha256 = config_hash,
            .build = actual_build,
            .deployment = .{
                .receipt_json = receipt_bytes,
                .image = image,
                .azure_sku = platform.azure_sku,
                .azure_region = platform.azure_region,
                .configured_vm_ram_bytes = 1024 * 1024 * 1024,
            },
            .workloads = workloads,
            .cpu = .{ .model = platform.cpu_model, .active_count = 1 },
            .native = pages.platform(),
            .clock = .{ .source = "test-linux-clock-monotonic", .resolution_ns = try linux.resolution(.MONOTONIC), .wasi_clock = try linux.wasiClock() },
            .pages = .{ .context = pages, .method = "test-mmap-retained-commit", .sample = footprint },
            .allocator = allocator,
            .workspace = workspace,
            .result = result,
            .evidence = evidence,
        } };
    }

    fn deinit(self: *Envelope) void {
        self.arena.deinit();
    }
};

fn value(a: std.mem.Allocator, data: anytype) !json.Value {
    return (try std.json.parseFromSlice(json.Value, a, try std.json.Stringify.valueAlloc(a, data, .{}), .{})).value;
}

fn footprint(raw: ?*anyopaque) !guest.Footprint {
    const pages: *linux.Pages = @ptrCast(@alignCast(raw.?));
    return .{ .reserved_address_bytes = pages.reserved(), .retained_committed_bytes = pages.committed() };
}

fn parsedResult(writer: *const guest.Writer) !std.json.Parsed(json.Value) {
    const prefix = "WAMR_NATIVE_CHECK_RESULT=";
    try expect(std.mem.startsWith(u8, writer.buffered(), prefix));
    return std.json.parseFromSlice(json.Value, std.testing.allocator, writer.buffered()[prefix.len..], .{});
}

test "native guest real pinned compute and memory v2 snapshot replay" {
    for ([_][]const u8{ "compute", "memory" }, [_][]const u8{ fixtures.compute_wasm, fixtures.memory_wasm }, [_][]const u8{ fixtures.compute, fixtures.memory }) |name, wasm, native_bytes| {
        var pages: linux.Pages = .{};
        var workspace: [256 * 1024]u8 = undefined;
        var result_storage: [32768]u8 = undefined;
        var evidence_storage: [32768]u8 = undefined;
        var result = guest.Writer.fixed(&result_storage);
        var evidence = guest.Writer.fixed(&evidence_storage);
        var envelope = try Envelope.init(name, wasm, native_bytes, &pages, &workspace, &result, &evidence);
        defer envelope.deinit();
        try guest.run(envelope.options);
        const parsed = try parsedResult(&result);
        defer parsed.deinit();
        try expect(json.matches(try json.field(parsed.value, "outcome"), "success"));
        try equal(@as(usize, 3), (try json.field(parsed.value, "invocations")).array.items.len);
        try equal(@as(usize, 2), (try json.field(parsed.value, "reset_events")).array.items.len);
        const phases = try json.field(parsed.value, "phases");
        try expect((try json.field(phases, "compile_ticks")) == .null);
        for ([_][]const u8{ "load_ticks", "instantiate_ticks", "lifecycle_setup_ticks", "first_invocation_ticks" }) |field|
            try expect(try json.number(phases, field) > 0);
        const memory = try json.field(parsed.value, "memory");
        try expect(json.matches(try json.field(memory, "coverage"), "partial-guest"));
        try equal(@as(usize, 3), (try json.field(memory, "samples")).array.items.len);
        try expect(std.mem.indexOf(u8, evidence.buffered(), "after_release method=caller-requested-bytes live=0") != null);
        try equal(@as(usize, 0), pages.reserved());
    }
}

test "native guest clocks preserve uncalled resets and executed untimed calls" {
    for ([_]usize{ 1, 2, 4, 6, 7, 8, 9, 10, 11, 12 }) |failed_read| {
        for ([_]bool{ false, true }) |backwards| {
            var pages: linux.Pages = .{};
            if (backwards) pages.backwards_clock_at = failed_read else pages.fail_clock_at = failed_read;
            var workspace: [256 * 1024]u8 = undefined;
            var result_storage: [32768]u8 = undefined;
            var evidence_storage: [32768]u8 = undefined;
            var result = guest.Writer.fixed(&result_storage);
            var evidence = guest.Writer.fixed(&evidence_storage);
            var envelope = try Envelope.init("compute", fixtures.compute_wasm, fixtures.compute, &pages, &workspace, &result, &evidence);
            defer envelope.deinit();
            try guest.run(envelope.options);
            const parsed = try parsedResult(&result);
            defer parsed.deinit();
            const invocations = (try json.field(parsed.value, "invocations")).array.items;
            // A zero opening reading is not a backwards interval; only closing
            // reversal is an error. Returned timestamps remain actual readings.
            if (backwards and failed_read % 2 != 0) continue;
            try expect(json.matches(try json.field(parsed.value, "outcome"), "error"));
            const count: usize = if (failed_read <= 7) 0 else if (failed_read <= 11) 1 else 2;
            try equal(count, invocations.len);
            if (failed_read == 8 or failed_read == 12) {
                const last = invocations[invocations.len - 1];
                try expect(json.matches(try json.field(last, "outcome"), "returned"));
                try equal(@as(usize, 1), (try json.field(last, "measurement_errors")).array.items.len);
                try expect(std.mem.indexOf(u8, evidence.buffered(), "\"timing_error\":null") != null);
                const phases = try json.field(parsed.value, "phases");
                if (failed_read == 8) try expect((try json.field(phases, "first_invocation_ticks")) == .null);
            }
            if (failed_read >= 9) try equal(@as(usize, 1), (try json.field(parsed.value, "reset_events")).array.items.len);
            try equal(@as(usize, 0), pages.reserved());
        }
    }
}

test "native guest unavailable resolution never starts or invents durations" {
    var pages: linux.Pages = .{};
    var workspace: [256 * 1024]u8 = undefined;
    var result_storage: [32768]u8 = undefined;
    var evidence_storage: [32768]u8 = undefined;
    var result = guest.Writer.fixed(&result_storage);
    var evidence = guest.Writer.fixed(&evidence_storage);
    var envelope = try Envelope.init("compute", fixtures.compute_wasm, fixtures.compute, &pages, &workspace, &result, &evidence);
    defer envelope.deinit();
    envelope.options.clock.resolution_ns = null;
    try guest.run(envelope.options);
    const parsed = try parsedResult(&result);
    defer parsed.deinit();
    try expect((try json.field(parsed.value, "clock")) == .null);
    try expect((try json.field(parsed.value, "memory")) == .null);
    try equal(@as(usize, 0), pages.clock_reads);
    try equal(@as(usize, 0), (try json.field(parsed.value, "invocations")).array.items.len);
}

test "native guest independent identities reject request echo before mapping" {
    inline for (.{ "runtime", "source", "compiler", "image", "cpu", "ram", "policy", "digest", "wasm", "aot" }) |changed| {
        var pages: linux.Pages = .{};
        var workspace: [256 * 1024]u8 = undefined;
        var result_storage: [32768]u8 = undefined;
        var evidence_storage: [32768]u8 = undefined;
        var result = guest.Writer.fixed(&result_storage);
        var evidence = guest.Writer.fixed(&evidence_storage);
        var envelope = try Envelope.init("compute", fixtures.compute_wasm, fixtures.compute, &pages, &workspace, &result, &evidence);
        defer envelope.deinit();
        const o = &envelope.options;
        if (comptime std.mem.eql(u8, changed, "runtime")) o.build.runtime.bytes += 1;
        if (comptime std.mem.eql(u8, changed, "source")) o.build.source.commit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        if (comptime std.mem.eql(u8, changed, "compiler")) o.build.compiler.version = "different-compiler";
        if (comptime std.mem.eql(u8, changed, "image")) o.deployment.image.bytes += 1;
        if (comptime std.mem.eql(u8, changed, "cpu")) o.cpu.active_count += 1;
        if (comptime std.mem.eql(u8, changed, "ram")) o.deployment.configured_vm_ram_bytes += 1;
        if (comptime std.mem.eql(u8, changed, "policy")) o.build.memory_policy.page_policy.commitment = "wrong-policy";
        if (comptime std.mem.eql(u8, changed, "digest")) o.config_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        if (comptime std.mem.eql(u8, changed, "wasm")) o.workloads = &.{.{ .name = "compute", .wasm = fixtures.memory_wasm, .aot_bytes = fixtures.compute }};
        if (comptime std.mem.eql(u8, changed, "aot")) o.workloads = &.{.{ .name = "compute", .wasm = fixtures.compute_wasm, .aot_bytes = fixtures.memory }};
        try std.testing.expectError(error.InvalidConfiguration, guest.run(o.*));
        try equal(@as(usize, 0), pages.clock_reads);
        try equal(@as(usize, 0), result.end);
    }
}

test "native guest bounded request workspace and output preserve failure evidence" {
    var pages: linux.Pages = .{};
    var workspace: [256 * 1024]u8 = undefined;
    var result_storage: [32768]u8 = undefined;
    var evidence_storage: [32768]u8 = undefined;
    var result = guest.Writer.fixed(&result_storage);
    var evidence = guest.Writer.fixed(&evidence_storage);
    var envelope = try Envelope.init("compute", fixtures.compute_wasm, fixtures.guest, &pages, &workspace, &result, &evidence);
    defer envelope.deinit();
    var o = envelope.options;
    o.max_request_bytes = 1;
    try std.testing.expectError(error.InputTooLarge, guest.run(o));
    o = envelope.options;
    o.workspace = workspace[0..1];
    try std.testing.expectError(error.OutOfMemory, guest.run(o));
    o = envelope.options;
    o.request_json = "[" ** 25;
    try std.testing.expectError(error.InputTooDeep, guest.run(o));
    o = envelope.options;
    o.max_output_bytes = 3;
    try guest.run(o);
    const parsed = try parsedResult(&result);
    defer parsed.deinit();
    try expect(json.matches(try json.field(parsed.value, "outcome"), "error"));
    try equal(@as(usize, 0), (try json.field(parsed.value, "invocations")).array.items.len);
    try expect(std.mem.indexOf(u8, evidence.buffered(), "\"phase\":\"module-start\"") != null);
    try expect(std.mem.indexOf(u8, evidence.buffered(), "\"stdout_base64\":\"c3Rh\"") != null);
    try expect(std.mem.indexOf(u8, evidence.buffered(), "\"stdout_complete\":false") != null);
    try expect(std.mem.indexOf(u8, evidence.buffered(), "OutputFailurePending") != null);
    try equal(@as(usize, 0), pages.reserved());
}

test "native guest start output and full u32 terminal remain exact across reset" {
    var pages: linux.Pages = .{};
    var storage: [32768]u8 = undefined;
    var writer = guest.Writer.fixed(&storage);
    var progress: guest.runner.Session.Progress = .{};
    const session = try guest.runner.Session.createCaptured(std.testing.allocator, pages.platform(), fixtures.guest, &.{}, &.{}, try linux.wasiClock(), &progress, .{ .setup_evidence = &writer });
    defer session.deinit();
    try expect(std.mem.indexOf(u8, writer.buffered(), "\"stdout_base64\":\"c3RhcnQK\"") != null);
    const first = try session.invoke("returned");
    try std.testing.expectEqualSlices(u8, "\x00\xff\xc3(\n\x00", first.stdout);
    try expect(first.succeeded());
    try session.reset();
    const repeated = try session.invoke("returned");
    try expect(repeated.succeeded());
    try std.testing.expectEqualSlices(u8, "\x00\xff\xc3(\n\x00", repeated.stdout);
    try session.reset();
    const exited = try session.invoke("exit_full_u32");
    try equal(@as(?u32, 0xffffffff), exited.exit_code);
    try std.testing.expectEqualStrings("proc_exit", exited.outcome);
    try guest.runner.writeInvocationEvidence(&writer, "steady", 2, exited);
    try expect(std.mem.indexOf(u8, writer.buffered(), "\"exit_code\":4294967295") != null);
    try session.reset();
    const trapped = try session.invoke("trap");
    try std.testing.expectEqualStrings("trap", trapped.outcome);
    try equal(@as(?u32, null), trapped.exit_code);
    try session.reset();
    session.output.limit = 3;
    const truncated = try session.invoke("returned");
    try std.testing.expectEqualSlices(u8, "\x00\xff\xc3", truncated.stdout);
    try expect(truncated.output_failure and !truncated.stdout_complete);
    try std.testing.expectError(error.OutputFailurePending, session.reset());
}

test "native guest module start evidence survives failed closing setup clock" {
    var pages: linux.Pages = .{ .fail_clock_at = 4 };
    var storage: [32768]u8 = undefined;
    var writer = guest.Writer.fixed(&storage);
    var progress: guest.runner.Session.Progress = .{};
    try std.testing.expectError(error.ClockFailed, guest.runner.Session.createCaptured(std.testing.allocator, pages.platform(), fixtures.guest, &.{}, &.{}, try linux.wasiClock(), &progress, .{ .setup_evidence = &writer }));
    try expect(std.mem.indexOf(u8, writer.buffered(), "\"stdout_base64\":\"c3RhcnQK\"") != null);
    try equal(@as(?u64, null), progress.instantiate_ticks);
    try equal(@as(usize, 0), pages.reserved());
}

test "native guest report OOM after execution cannot erase output or create success" {
    var pages: linux.Pages = .{};
    var workspace: [256 * 1024]u8 = undefined;
    var result_storage: [32768]u8 = undefined;
    var evidence_storage: [32768]u8 = undefined;
    var result = guest.Writer.fixed(&result_storage);
    var evidence = guest.Writer.fixed(&evidence_storage);
    var envelope = try Envelope.init("compute", fixtures.compute_wasm, fixtures.guest, &pages, &workspace, &result, &evidence);
    defer envelope.deinit();
    var low: usize = 1;
    var high: usize = workspace.len;
    while (low < high) {
        const midpoint = low + (high - low) / 2;
        var o = envelope.options;
        o.workspace = workspace[0..midpoint];
        result.end = 0;
        evidence.end = 0;
        if (guest.run(o)) |_| {
            high = midpoint;
        } else |failure| {
            // The JSON canonicalizer's allocating writer exposes budget
            // exhaustion as WriteFailed before native execution.
            try expect(failure == error.OutOfMemory or failure == error.WriteFailed);
            low = midpoint + 1;
        }
        try equal(@as(usize, 0), pages.reserved());
    }
    result.end = 0;
    evidence.end = 0;
    envelope.options.workspace = workspace[0 .. low - 1];
    try std.testing.expectError(error.OutOfMemory, guest.run(envelope.options));
    try equal(@as(usize, 0), result.end);
    try expect(std.mem.indexOf(u8, evidence.buffered(), "\"phase\":\"first\"") != null);
    try expect(std.mem.indexOf(u8, evidence.buffered(), "\"stdout_base64\":\"AP/DKAoA\"") != null);
    try expect(std.mem.indexOf(u8, evidence.buffered(), "stage=producer error=OutOfMemory") != null);
}

fn allocationCase(allocator: std.mem.Allocator) !void {
    var pages: linux.Pages = .{};
    var workspace: [256 * 1024]u8 = undefined;
    var result_storage: [32768]u8 = undefined;
    var evidence_storage: [32768]u8 = undefined;
    var result = guest.Writer.fixed(&result_storage);
    var evidence = guest.Writer.fixed(&evidence_storage);
    var envelope = try Envelope.init("compute", fixtures.compute_wasm, fixtures.compute, &pages, &workspace, &result, &evidence);
    defer envelope.deinit();
    envelope.options.allocator = allocator;
    try guest.run(envelope.options);
    try equal(@as(usize, 0), pages.reserved());
    const parsed = try parsedResult(&result);
    defer parsed.deinit();
    if (!json.matches(try json.field(parsed.value, "outcome"), "success")) return error.OutOfMemory;
}

test "native guest every runtime allocation failure produces failed evidence and releases mappings" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}

fn saturatedClosingClock(raw: *anyopaque) guest.aot.PlatformError!u64 {
    const pages: *linux.Pages = @ptrCast(@alignCast(raw));
    pages.clock_reads += 1;
    return if (pages.clock_reads == 8) std.math.maxInt(u64) else linux.now();
}

test "native guest saturated closing clock and failed evidence transport fail closed" {
    var pages: linux.Pages = .{};
    var workspace: [256 * 1024]u8 = undefined;
    var result_storage: [32768]u8 = undefined;
    var evidence_storage: [32768]u8 = undefined;
    var result = guest.Writer.fixed(&result_storage);
    var evidence = guest.Writer.fixed(&evidence_storage);
    var envelope = try Envelope.init("compute", fixtures.compute_wasm, fixtures.compute, &pages, &workspace, &result, &evidence);
    defer envelope.deinit();
    envelope.options.native.monotonic_ns = saturatedClosingClock;
    try guest.run(envelope.options);
    const parsed = try parsedResult(&result);
    defer parsed.deinit();
    try expect(json.matches(try json.field(parsed.value, "outcome"), "error"));
    try expect((try json.field(try json.field(parsed.value, "phases"), "first_invocation_ticks")) == .null);
    try equal(@as(usize, 1), (try json.field(parsed.value, "invocations")).array.items.len);
    try expect(std.mem.indexOf(u8, evidence.buffered(), "\"timing_error\":\"ClockFailed\"") != null);
    var short = guest.Writer.fixed(evidence_storage[0..1]);
    envelope.options.evidence = &short;
    result.end = 0;
    try std.testing.expectError(error.WriteFailed, guest.run(envelope.options));
    try equal(@as(usize, 0), result.end);
    try equal(@as(usize, 0), pages.reserved());
}

test "native guest CoreMark measurement argv is pinned with no short-run override" {
    for ([_][]const u8{ "coremark", "coremark-nofp" }, [_][]const u8{ fixtures.coremark_wasm, fixtures.nofp_wasm }, [_][]const u8{ fixtures.coremark, fixtures.nofp }) |name, wasm, native_bytes| {
        var pages: linux.Pages = .{};
        var workspace: [256 * 1024]u8 = undefined;
        var result_storage: [32768]u8 = undefined;
        var evidence_storage: [32768]u8 = undefined;
        var result = guest.Writer.fixed(&result_storage);
        var evidence = guest.Writer.fixed(&evidence_storage);
        var envelope = try Envelope.init(name, wasm, native_bytes, &pages, &workspace, &result, &evidence);
        defer envelope.deinit();
        envelope.options.clock.resolution_ns = null;
        try guest.run(envelope.options);
        const a = envelope.arena.allocator();
        const request = (try std.json.parseFromSlice(json.Value, a, envelope.options.request_json, .{})).value;
        const config = try json.field(request, "config");
        const args = try json.field(try json.field(config, "workload"), "args");
        args.array.items[3] = json.str("100");
        const hash = try json.digest(a, try json.canonical(a, config));
        envelope.options.config_sha256 = hash;
        envelope.options.request_json = try std.json.Stringify.valueAlloc(a, .{ .config = config, .config_sha256 = hash }, .{});
        result.end = 0;
        try std.testing.expectError(error.InvalidConfiguration, guest.run(envelope.options));
        try equal(@as(usize, 0), result.end);
        try equal(@as(usize, 0), pages.clock_reads);
    }
}

/// Hosted test executable only: sends synthetic envelopes and real native
/// payloads to the existing Python v2 validator, never a deployment artifact.
pub fn protocolFixture(init: std.process.Init) !void {
    for ([_]?usize{ null, 8, 9, 11, 12, null }, 0..) |failure, index| {
        var pages: linux.Pages = .{ .fail_clock_at = failure };
        var workspace: [256 * 1024]u8 = undefined;
        var out_storage: [4096]u8 = undefined;
        var err_storage: [4096]u8 = undefined;
        var out = std.Io.File.stdout().writer(init.io, &out_storage);
        var err = std.Io.File.stderr().writer(init.io, &err_storage);
        var envelope = try Envelope.initWithAllocator(init.gpa, "compute", fixtures.compute_wasm, if (index == 5) fixtures.guest else fixtures.compute, &pages, &workspace, &out.interface, &err.interface);
        defer envelope.deinit();
        const a = envelope.arena.allocator();
        const config = try json.field((try std.json.parseFromSlice(json.Value, a, envelope.options.request_json, .{})).value, "config");
        var workloads = json.object(a);
        try json.put(a, &workloads, "compute", try json.field(config, "workload"));
        try out.interface.writeAll("WAMR_NATIVE_TEST_MANIFEST=");
        try std.json.Stringify.value(.{
            .campaign_id = try json.field(config, "campaign_id"),
            .evidence_kind = "measurement",
            .schedule = &[_]json.Value{try json.field(config, "run")},
            .targets = .{ .unikraft = try json.field(config, "target") },
            .workloads = workloads,
            .steady_invocations = 2,
            .created_at = "2026-01-01T00:00:00Z",
            .expires_at = "2026-01-01T01:00:00Z",
        }, .{}, &out.interface);
        try out.interface.writeByte('\n');
        try out.interface.flush();
        try guest.run(envelope.options);
    }
}
