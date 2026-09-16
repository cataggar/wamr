//! Linux embedding consumer. Only trusted, externally precompiled native
//! artifacts enter this binary; no hosted runtime/compiler is linked.
const std = @import("std");
const linux = @import("bench/native_linux.zig");
const runner = @import("bench/native_runner.zig");
const allocations = @import("bench/native_allocations.zig");
const json = @import("bench/native_json.zig");
const Value = json.Value;
// Enable only with the host-owned request/receipt/result lifecycle binding.
const lifecycle_protocol_supported = false;

const Files = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    fn read(self: Files, path: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAlloc(self.io, path, self.allocator, .limited(256 * 1024 * 1024));
    }

    fn parse(self: Files, path: []const u8) !Value {
        return (try std.json.parseFromSlice(Value, self.allocator, try self.read(path), .{ .allocate = .alloc_always })).value;
    }

    fn hash(self: Files, path: []const u8) ![]const u8 {
        return (try self.identity(path)).sha256;
    }

    fn identity(self: Files, path: []const u8) !struct { sha256: []const u8, bytes: u64 } {
        const file = try std.Io.Dir.cwd().openFile(self.io, path, .{});
        defer file.close(self.io);
        var buffer: [65536]u8 = undefined;
        var file_reader = file.reader(self.io, &buffer);
        var block: [65536]u8 = undefined;
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        var count: u64 = 0;
        while (true) {
            const length = try file_reader.interface.readSliceShort(&block);
            if (length == 0) break;
            hasher.update(block[0..length]);
            count = try std.math.add(u64, count, length);
        }
        var hash_bytes: [32]u8 = undefined;
        hasher.final(&hash_bytes);
        return .{ .sha256 = try self.allocator.dupe(u8, &std.fmt.bytesToHex(hash_bytes, .lower)), .bytes = count };
    }

    fn verifyFile(self: Files, path: []const u8, expected: Value) !void {
        const actual = try self.identity(path);
        try json.require(actual.bytes != 0 and actual.bytes == try json.number(expected, "bytes"));
        try json.require(std.mem.eql(u8, actual.sha256, try json.string(expected, "sha256")));
    }

    fn verify(self: Files, path: []const u8, expected: Value) ![]const u8 {
        const bytes = try self.read(path);
        try json.require(bytes.len == try json.number(expected, "bytes"));
        try json.require(std.mem.eql(u8, try json.digest(self.allocator, bytes), try json.string(expected, "sha256")));
        return bytes;
    }
};

fn set(a: std.mem.Allocator, obj: *Value, key: []const u8, value: anytype) !void {
    const encoded = try std.json.Stringify.valueAlloc(a, value, .{});
    try json.put(a, obj, key, (try std.json.parseFromSlice(Value, a, encoded, .{})).value);
}

fn emit(io: std.Io, a: std.mem.Allocator, prefix: []const u8, value: Value) !void {
    const bytes = try std.json.Stringify.valueAlloc(a, value, .{});
    var buffer: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &buffer);
    try writer.interface.writeAll(prefix);
    try writer.interface.writeAll(bytes);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

fn stringArray(a: std.mem.Allocator, value: Value) ![]const []const u8 {
    if (value != .array) return error.InvalidConfiguration;
    const items = try a.alloc([]const u8, value.array.items.len);
    for (value.array.items, items) |v, *item| item.* = try json.text(v);
    return items;
}

fn invocationJson(a: std.mem.Allocator, invocation: runner.Invocation, phase: []const u8) !Value {
    var obj = json.object(a);
    try set(a, &obj, "phase", phase);
    try set(a, &obj, "outcome", invocation.outcome);
    try set(a, &obj, "exit_code", invocation.exit_code);
    try set(a, &obj, "stdout_base64", try base64(a, invocation.stdout));
    return obj;
}

fn base64(a: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(bytes.len));
    return std.base64.standard.Encoder.encode(encoded, bytes);
}

fn memoryPolicy(a: std.mem.Allocator) !Value {
    const encoded = try std.json.Stringify.valueAlloc(a, .{
        .allocator_by_phase = .{
            .load = "counted-zig016-debug-and-arena",
            .instantiate = "counted-zig016-debug-and-arena",
            .first = "counted-zig016-debug-output",
            .steady = "counted-zig016-debug-output",
        },
        .page_policy = .{
            .reservation = "linux-mmap-prot-none",
            .commitment = "linux-mprotect-rw-zero",
            .release = "linux-munmap-full",
        },
    }, .{});
    return (try std.json.parseFromSlice(Value, a, encoded, .{})).value;
}

fn snapshotJson(a: std.mem.Allocator, pages: *const linux.Pages, stage: []const u8) !Value {
    var obj = json.object(a);
    try set(a, &obj, "stage", stage);
    try set(a, &obj, "reserved_address_bytes", pages.reserved());
    try set(a, &obj, "committed_bytes", pages.committed());
    return obj;
}

fn allocationDiagnostic(io: std.Io, counter: *const allocations.Counter, stage: []const u8) !void {
    var buffer: [512]u8 = undefined;
    var output = std.Io.File.stderr().writer(io, &buffer);
    try output.interface.print(
        "WAMR_NATIVE_ALLOCATOR stage={s} method=caller-requested-bytes live={d} peak={d}\n",
        .{ stage, counter.live, counter.peak },
    );
    try output.interface.flush();
}

fn diagnostics(io: std.Io, phase: []const u8, index: usize, invocation: runner.Invocation) !void {
    var buffer: [4096]u8 = undefined;
    var output = std.Io.File.stderr().writer(io, &buffer);
    try runner.writeInvocationEvidence(&output.interface, phase, index, invocation);
    try output.interface.flush();
}

fn check(init: std.process.Init, args: []const []const u8) !void {
    if (args.len != 6 and args.len != 7) return error.Usage;
    const Fault = enum { @"closing-clock", @"backwards-clock", @"report-oom" };
    const fault: ?Fault = if (args.len == 7) std.meta.stringToEnum(Fault, args[6]) orelse return error.Usage else null;
    const a = init.arena.allocator();
    const iterations = try std.fmt.parseInt(u32, args[3], 10);
    const repeats = try std.fmt.parseInt(u32, args[4], 10);
    if (iterations == 0 or repeats == 0 or repeats > 10000) return error.InvalidConfiguration;
    const files: Files = .{ .allocator = a, .io = init.io };
    const bytes = try files.read(args[2]);
    var pages: linux.Pages = .{};
    var counter: allocations.Counter = .{ .child = init.gpa };
    defer allocationDiagnostic(init.io, &counter, "after_release") catch {};
    const caller_allocator = counter.allocator();
    const argv: []const []const u8 = if (std.mem.startsWith(u8, args[5], "coremark"))
        &.{ args[5], "0", "0", "0", args[3], "0" }
    else
        &.{args[5]};
    var records = json.array(a);
    var steady_ticks = json.array(a);
    var samples = json.array(a);
    var phases = json.object(a);
    var good = true;
    {
        const session = try runner.Session.create(caller_allocator, pages.platform(), bytes, argv, &.{"WAMR_NATIVE_CHECK=1"}, try linux.wasiClock());
        defer session.deinit();
        try allocationDiagnostic(init.io, &counter, "after_instantiation");
        try set(a, &phases, "compile_ticks", @as(?u64, null));
        try set(a, &phases, "load_ticks", session.load_ticks);
        try set(a, &phases, "instantiate_ticks", session.instantiate_ticks);
        try samples.array.append(try snapshotJson(a, &pages, "after_instantiation"));
        var first_crc: ?u16 = null;
        for (0..repeats) |index| {
            if (index != 0) try session.reset();
            if (fault == .@"closing-clock") pages.fail_clock_at = pages.clock_reads + 2;
            if (fault == .@"backwards-clock") pages.backwards_clock_at = pages.clock_reads + 2;
            const result = try session.invoke("_start");
            const phase = if (index == 0) "first" else "steady";
            try diagnostics(init.io, phase, index, result);
            var report_allocations: allocations.Counter = .{ .child = a, .fail_allocations = fault == .@"report-oom" };
            const report_allocator = if (fault == .@"report-oom") report_allocations.allocator() else a;
            var record = try invocationJson(report_allocator, result, phase);
            try set(a, &record, "elapsed_ns", result.ticks);
            try set(a, &record, "stderr_base64", try base64(a, result.stderr));
            try set(a, &record, "diagnostic", result.diagnostic);
            try set(a, &record, "timing_error", result.timing_error);
            if (std.mem.startsWith(u8, args[5], "coremark")) {
                const crc = runner.coremarkCrc(result.stdout);
                if (index == 0) first_crc = crc;
                const valid = crc != null and crc == first_crc;
                try set(a, &record, "crc_valid", valid);
                good = good and valid;
            }
            good = good and result.succeeded();
            try allocationDiagnostic(init.io, &counter, if (index == 0) "after_first" else "after_steady");
            try records.array.append(record);
            if (index == 0) {
                try set(a, &phases, "first_invocation_ticks", result.ticks);
                try samples.array.append(try snapshotJson(a, &pages, "after_first"));
            } else if (result.ticks) |ticks| try steady_ticks.array.append(json.num(ticks));
            if (!good) break;
        }
        if (steady_ticks.array.items.len != 0) try samples.array.append(try snapshotJson(a, &pages, "after_steady"));
    }
    try json.put(a, &phases, "steady_state_ticks", steady_ticks);
    var result = json.object(a);
    try set(a, &result, "kind", "wamr-native-correctness-check");
    try set(a, &result, "qualification", "correctness-only-not-performance");
    try set(a, &result, "reset_semantics", "same-instance-full-snapshot-reset");
    try set(a, &result, "injected_fault", if (fault) |f| @as(?[]const u8, @tagName(f)) else null);
    try set(a, &result, "outcome", if (good) @as([]const u8, "success") else "error");
    try set(a, &result, "artifact_sha256", try json.digest(a, bytes));
    try set(a, &result, "workload", args[5]);
    try set(a, &result, "iterations", iterations);
    try set(a, &result, "clock_resolution_ns", try linux.resolution(.MONOTONIC));
    try json.put(a, &result, "invocations", records);
    try json.put(a, &result, "phases", phases);
    try json.put(a, &result, "memory_samples", samples);
    try emit(init.io, a, "WAMR_NATIVE_CHECK_RESULT=", result);
    if (pages.reserved() != 0 or counter.live != 0) return error.ResourceLeak;
    if (!good) return error.GuestFailed;
}

/// CPU identity comes from Linux, never from the incoming benchmark request.
/// The token conversion is fixed and reproducible by the image integrator.
fn observedPlatform(files: Files, receipt_platform: Value) !Value {
    const cpuinfo = try files.read("/proc/cpuinfo");
    var model: ?[]const u8 = null;
    var lines = std.mem.tokenizeScalar(u8, cpuinfo, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "model name")) continue;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        model = std.mem.trim(u8, line[colon + 1 ..], " \t");
        break;
    }
    const description = model orelse return error.UnsupportedPlatform;
    const token = try files.allocator.dupe(u8, description);
    for (token) |*c| {
        if (!std.ascii.isAlphanumeric(c.*) and std.mem.indexOfScalar(u8, "_.:+-", c.*) == null) c.* = '_';
    }
    const online = std.mem.trim(u8, try files.read("/sys/devices/system/cpu/online"), " \t\r\n");
    var count: u64 = 0;
    var ranges = std.mem.splitScalar(u8, online, ',');
    while (ranges.next()) |range| {
        var ends = std.mem.splitScalar(u8, range, '-');
        const first = try std.fmt.parseInt(u32, ends.next() orelse return error.UnsupportedPlatform, 10);
        const last = if (ends.next()) |end| try std.fmt.parseInt(u32, end, 10) else first;
        if (last < first or ends.next() != null) return error.UnsupportedPlatform;
        count += @as(u64, last) - first + 1;
    }
    var platform = json.object(files.allocator);
    try set(files.allocator, &platform, "arch", "x86_64");
    try set(files.allocator, &platform, "cpu_model", token);
    try set(files.allocator, &platform, "active_cpu_count", count);
    for ([_][]const u8{ "azure_sku", "azure_region" }) |name|
        try json.put(files.allocator, &platform, name, try json.field(receipt_platform, name));
    try json.require(try json.equal(files.allocator, platform, receipt_platform));
    return platform;
}

fn measure(init: std.process.Init, deployment_path: []const u8) !void {
    if (!lifecycle_protocol_supported) return error.LifecycleProtocolPending;
    const a = init.arena.allocator();
    const files: Files = .{ .allocator = a, .io = init.io };
    const deployment = try files.parse(deployment_path);
    try json.require(json.matches(try json.field(deployment, "kind"), "wamr-linux-producer-deployment"));
    try json.require(try json.number(deployment, "schema_version") == 1);
    try json.require(json.matches(try json.field(deployment, "qualification"), "hardware"));
    const request_path = init.environ_map.get("WAMR_BENCH_REQUEST") orelse return error.MissingRequest;
    const expected_hash = init.environ_map.get("WAMR_BENCH_CONFIG_SHA256") orelse return error.MissingRequest;
    const request = try files.parse(request_path);
    const config = try json.field(request, "config");
    const config_hash = try json.digest(a, try json.canonical(a, config));
    try json.require(std.mem.eql(u8, config_hash, expected_hash));
    try json.require(std.mem.eql(u8, config_hash, try json.string(request, "config_sha256")));
    const run = try json.field(config, "run");
    try json.require(json.matches(try json.field(run, "target"), "linux"));
    try json.require(json.matches(try json.field(config, "phase_contract"), "wamr-embedding-v1"));
    const target = try json.field(config, "target");
    const receipt_path = try json.string(deployment, "receipt_path");
    const receipt = try files.parse(receipt_path);
    const receipt_hash = try files.hash(receipt_path);
    try json.require(std.mem.eql(u8, receipt_hash, try json.string(target, "image_receipt_sha256")));
    try json.require(try json.equal(a, receipt, try json.field(target, "image_receipt")));
    try json.require(json.matches(try json.field(receipt, "kind"), "wamr-native-image-receipt"));
    try json.require(json.matches(try json.field(receipt, "evidence_kind"), "measurement"));
    try json.require(json.matches(try json.field(receipt, "os"), "linux"));
    try json.require(json.matches(try json.field(receipt, "compile_profile"), "unikraft-x86_64"));
    try json.require((try json.field(receipt, "compiler_embedded")) == .bool and !(try json.field(receipt, "compiler_embedded")).bool);
    try json.require(json.matches(try json.field(receipt, "runtime_linkage"), "static"));
    for ([_][]const u8{ "bounds_checks", "wx_enforced", "import_checks", "trap_isolation" }) |name| {
        const enabled = try json.field(try json.field(receipt, "options"), name);
        try json.require(enabled == .bool and enabled.bool);
    }
    const policy = try memoryPolicy(a);
    try json.require(try json.equal(a, policy, try json.field(receipt, "memory_policy")));
    for ([_][]const u8{ "os", "image", "runtime", "source", "compiler", "compile_profile", "target_abi", "platform", "options", "aot_modules", "memory_policy" }) |name|
        try json.require(try json.equal(a, try json.field(receipt, name), try json.field(target, name)));
    const self_hash = try files.hash("/proc/self/exe");
    try json.require(std.mem.eql(u8, self_hash, try json.string(deployment, "producer_sha256")));
    try json.require(std.mem.eql(u8, self_hash, try json.string(try json.field(receipt, "runtime"), "sha256")));
    try json.require(json.matches(try json.field(try json.field(receipt, "options"), "optimize"), @tagName(@import("builtin").mode)));
    try files.verifyFile(try json.string(deployment, "image_path"), try json.field(receipt, "image"));
    try files.verifyFile(try json.string(deployment, "runtime_path"), try json.field(receipt, "runtime"));
    try files.verifyFile(try json.string(deployment, "compiler_path"), try json.field(try json.field(receipt, "compiler"), "binary"));
    const workload_name = try json.string(run, "workload");
    const artifact_identity = try json.field(try json.field(receipt, "aot_modules"), workload_name);
    const bytes = try files.verify(try json.string(try json.field(deployment, "aot_paths"), workload_name), artifact_identity);
    const workload = try json.field(config, "workload");
    const wasm_hash = try files.hash(try json.string(try json.field(deployment, "wasm_paths"), workload_name));
    try json.require(std.mem.eql(u8, wasm_hash, try json.string(workload, "sha256")));
    const platform = try observedPlatform(files, try json.field(receipt, "platform"));
    const requested_args = try stringArray(a, try json.field(workload, "args"));
    const argv = try a.alloc([]const u8, requested_args.len + 1);
    argv[0] = workload_name;
    @memcpy(argv[1..], requested_args);
    const environment = try stringArray(a, try json.field(deployment, "environment"));
    try json.require(environment.len == 0);
    const steady = try json.number(config, "steady_invocations");
    try json.require(steady > 0 and steady <= 10000);
    var observed = json.object(a);
    try json.put(a, &observed, "image_sha256", try json.field(try json.field(receipt, "image"), "sha256"));
    try json.put(a, &observed, "runtime_sha256", try json.field(try json.field(receipt, "runtime"), "sha256"));
    try json.put(a, &observed, "aot_sha256", try json.field(artifact_identity, "sha256"));
    try set(a, &observed, "wasm_sha256", wasm_hash);
    try json.put(a, &observed, "platform", platform);
    try json.put(a, &observed, "options", try json.field(receipt, "options"));
    try set(a, &observed, "mode", "aot");
    try set(a, &observed, "jit_preset", @as(?u8, null));
    try set(a, &observed, "compile_profile", "unikraft-x86_64");
    var result = json.object(a);
    try set(a, &result, "schema_version", 1);
    try set(a, &result, "kind", "wamr-native-benchmark-result");
    try set(a, &result, "evidence_kind", "measurement");
    try json.put(a, &result, "campaign_id", try json.field(config, "campaign_id"));
    try json.put(a, &result, "run_id", try json.field(run, "run_id"));
    try set(a, &result, "config_sha256", config_hash);
    try set(a, &result, "image_receipt_sha256", receipt_hash);
    try json.put(a, &result, "observed", observed);
    try set(a, &result, "phase_contract", "wamr-embedding-v1");
    try set(a, &result, "clock", .{
        .source = "linux-clock-monotonic",
        .unit = "ns",
        .ticks_per_second = @as(u64, 1_000_000_000),
        .resolution_ticks = try linux.resolution(.MONOTONIC),
    });
    var phases = json.object(a);
    try set(a, &phases, "compile_ticks", @as(?u64, null));
    var invocations = json.array(a);
    var steady_ticks = json.array(a);
    var samples = json.array(a);
    var first_ticks: ?u64 = null;
    var outcome: []const u8 = "error";
    var exit_code: ?u32 = null;
    var progress: runner.Session.Progress = .{};
    var memory_reliable = true;
    var pages: linux.Pages = .{};
    var counter: allocations.Counter = .{ .child = init.gpa };
    defer allocationDiagnostic(init.io, &counter, "after_release") catch {};
    const caller_allocator = counter.allocator();
    if (runner.Session.createTimed(caller_allocator, pages.platform(), bytes, argv, environment, try linux.wasiClock(), &progress)) |session| {
        defer session.deinit();
        try allocationDiagnostic(init.io, &counter, "after_instantiation");
        try samples.array.append(try snapshotJson(a, &pages, "after_instantiation"));
        outcome = "success";
        var first_crc: ?u16 = null;
        for (0..@intCast(steady + 1)) |index| {
            if (index != 0) session.reset() catch |failure| {
                std.debug.print("native reset failed: {s}\n", .{@errorName(failure)});
                outcome = "error";
                memory_reliable = false;
                break;
            };
            const invocation = session.invoke(try json.string(workload, "export")) catch |failure| {
                std.debug.print("native invocation capture failed: {s}\n", .{@errorName(failure)});
                outcome = "error";
                break;
            };
            const phase = if (index == 0) "first" else "steady";
            try diagnostics(init.io, phase, index, invocation);
            const ticks = invocation.ticks orelse return error.UntimedInvocation;
            try allocationDiagnostic(init.io, &counter, if (index == 0) "after_first" else "after_steady");
            exit_code = invocation.exit_code;
            if (index == 0) first_ticks = ticks else try steady_ticks.array.append(json.num(ticks));
            try invocations.array.append(try invocationJson(a, invocation, phase));
            if (index == 0) try samples.array.append(try snapshotJson(a, &pages, "after_first"));
            const crc = runner.coremarkCrc(invocation.stdout);
            if (index == 0) first_crc = crc;
            const valid_output = if (std.mem.startsWith(u8, workload_name, "coremark"))
                crc != null and crc == first_crc and
                    std.mem.indexOf(u8, invocation.stdout, "Correct operation validated") != null and
                    std.mem.indexOf(u8, invocation.stdout, "ERROR!") == null
            else
                std.mem.eql(u8, invocation.outcome, "returned") and invocation.stdout.len == 0;
            if (!invocation.succeeded() or !valid_output) {
                outcome = if (std.mem.eql(u8, invocation.outcome, "trap")) "trap" else "error";
                exit_code = invocation.exit_code;
                break;
            }
        }
        if (steady_ticks.array.items.len != 0) try samples.array.append(try snapshotJson(a, &pages, "after_steady"));
    } else |failure| {
        std.debug.print("native load/instantiate failed: {s}\n", .{@errorName(failure)});
    }
    try set(a, &phases, "load_ticks", progress.load_ticks);
    try set(a, &phases, "instantiate_ticks", progress.instantiate_ticks);
    try set(a, &phases, "first_invocation_ticks", first_ticks);
    try json.put(a, &phases, "steady_state_ticks", steady_ticks);
    try json.put(a, &result, "phases", phases);
    try json.put(a, &result, "invocations", invocations);
    try set(a, &result, "outcome", outcome);
    try set(a, &result, "exit_code", exit_code);
    if (!memory_reliable or samples.array.items.len == 0) {
        try json.put(a, &result, "memory", .null);
    } else {
        var memory = json.object(a);
        try json.put(a, &memory, "image_sha256", try json.field(observed, "image_sha256"));
        try json.put(a, &memory, "configured_vm_ram_bytes", try json.field(receipt, "configured_vm_ram_bytes"));
        try set(a, &memory, "coverage", "partial-guest");
        try set(a, &memory, "method", "linux-mmap-retained-commit-high-water");
        try json.put(a, &memory, "policy", policy);
        try set(a, &memory, "covered_regions", &[_][]const u8{ "native-code", "wasm-linear-memory" });
        try set(a, &memory, "omitted_regions", &[_][]const u8{
            "runtime-allocator", "snapshot-allocator", "wasm-tables-globals",
            "producer-buffers",  "process-stacks",     "kernel",
            "other-processes",   "boot-image",
        });
        try json.put(a, &memory, "samples", samples);
        try json.put(a, &result, "memory", memory);
    }
    try emit(init.io, a, "WAMR_BENCH_RESULT=", result);
    if (pages.reserved() != 0 or counter.live != 0) return error.ResourceLeak;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--check")) return check(init, args);
    if (args.len == 3 and std.mem.eql(u8, args[1], "--deployment")) return measure(init, args[2]);
    std.debug.print(
        \\usage: wamr-native-bench --check FILE.cwasm ITERATIONS REPEATS WORKLOAD [closing-clock|backwards-clock|report-oom]
        \\       wamr-native-bench --deployment DEPLOYMENT.json
        \\--check is correctness-only, including on QEMU; it is never benchmark evidence.
        \\
    , .{});
    return error.Usage;
}
