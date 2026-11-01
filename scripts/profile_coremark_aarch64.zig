const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const json = std.json;
const Value = json.Value;

const benchmark_process = @import("benchmark_process.zig");
const benchmark_optimize = @import("benchmark_optimize.zig");
const decimal_format = @import("decimal_format.zig");
pub const bench_coremark = @import("bench_coremark.zig");
pub const compare_hot_function = @import("compare_hot_function.zig");
pub const aot_jit_attr = @import("aot_jit_attr.zig");
pub const provenance = @import("aarch64_instruction_provenance.zig");

pub const REPORT_SCHEMA_VERSION: i64 = 2;
pub const REPORT_KIND = "coremark-aarch64-matched-profile";
pub const HISTORICAL_REPORT_SCHEMA_VERSION: i64 = 1;
pub const HISTORICAL_BASELINE_RUN: i64 = 33631050708;
pub const DEFAULT_MIN_SAMPLES: i64 = 1000;
pub const DEFAULT_TOP_FUNCTIONS: usize = 10;
pub const DEFAULT_CLASSIFY_FUNCTIONS: usize = 3;
pub const DEFAULT_MAX_PERF_BYTES: u64 = 25 * 1024 * 1024;
pub const PROFILE_CAPTURES_PER_ENGINE: usize = 2;
pub const MIN_ATTRIBUTION_COVERAGE_PCT: f64 = 99.0;

pub const ALL_ALU_WORDING = "all ALU-class instructions: add/sub, logical operations, mul/div, shifts, compares, csel, and address-generation instructions";
pub const NARROW_ALU_WORDING = "architecture-only CFG and reaching-definition paths preserve the legacy all_alu partition and separately classify a complete common gating universe as address_generation, structural_address_guard, algorithmic_alu, mixed, or unknown without engine-register heuristics";

pub const REQUIRED_ANALYSIS_SOURCES = [_][]const u8{
    "scripts/profile_coremark_aarch64.zig",
    "scripts/aarch64_instruction_provenance.zig",
    "scripts/compare_hot_function.zig",
    "scripts/bench_coremark.zig",
    "scripts/benchmark_optimize.zig",
    "scripts/aot_jit_attr.zig",
};

pub const ClassGroup = struct {
    group: []const u8,
    raw_classes: []const []const u8,
};

pub const CLASS_GROUPS = [_]ClassGroup{
    .{
        .group = "frame_traffic",
        .raw_classes = &.{
            "frame_load_unattributed",
            "frame_store_unattributed",
            "unknown_frame_load",
            "unknown_frame_store",
        },
    },
    .{ .group = "reg_moves", .raw_classes = &.{"regmov"} },
    .{ .group = "all_alu", .raw_classes = &.{"alu"} },
    .{ .group = "bounds_checks", .raw_classes = &.{ "bounds_cmp", "bounds_branch" } },
    .{ .group = "linear_memory", .raw_classes = &.{ "linear_memory", "mem_access" } },
    .{ .group = "calls", .raw_classes = &.{"call"} },
    .{ .group = "conditional_branches", .raw_classes = &.{"cond_branch"} },
    .{ .group = "direct_branches", .raw_classes = &.{ "direct_branch", "jmp" } },
    .{ .group = "indirect_dispatch", .raw_classes = &.{ "indirect_dispatch", "dispatch_jmp" } },
};

pub const ProfileError = error{
    ProfileFailed,
    OutOfMemory,
    WriteFailed,
};

threadlocal var last_error_buf: [4096]u8 = undefined;
threadlocal var last_error_len: usize = 0;

pub fn lastErrorMessage() []const u8 {
    return last_error_buf[0..last_error_len];
}

pub fn clearErrorMessage() void {
    last_error_len = 0;
}

pub fn fail(comptime fmt: []const u8, args: anytype) ProfileError {
    const written = std.fmt.bufPrint(&last_error_buf, fmt, args) catch blk: {
        const fallback = "profile error";
        @memcpy(last_error_buf[0..fallback.len], fallback);
        break :blk last_error_buf[0..fallback.len];
    };
    last_error_len = written.len;
    return error.ProfileFailed;
}

fn jsonNumber(val: ?Value) ?f64 {
    const v = val orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn jsonInt(val: ?Value) ?i64 {
    const v = val orelse return null;
    return switch (v) {
        .integer => |i| i,
        else => null,
    };
}

fn jsonString(val: ?Value) ?[]const u8 {
    const v = val orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn jsonStringOrEmpty(val: ?Value) []const u8 {
    return jsonString(val) orelse "";
}

fn jsonBool(val: ?Value) ?bool {
    const v = val orelse return null;
    return switch (v) {
        .bool => |b| b,
        else => null,
    };
}

pub fn jsonValueEql(a: Value, b: Value) bool {
    return switch (a) {
        .null => b == .null,
        .bool => |ab| b == .bool and ab == b.bool,
        .integer => |ai| switch (b) {
            .integer => |bi| ai == bi,
            .float => |bf| @as(f64, @floatFromInt(ai)) == bf,
            else => false,
        },
        .float => |af| switch (b) {
            .float => |bf| af == bf,
            .integer => |bi| af == @as(f64, @floatFromInt(bi)),
            else => false,
        },
        .number_string => |as| b == .number_string and std.mem.eql(u8, as, b.number_string),
        .string => |as| b == .string and std.mem.eql(u8, as, b.string),
        .array => |aa| blk: {
            if (b != .array) break :blk false;
            const ba = b.array;
            if (aa.items.len != ba.items.len) break :blk false;
            for (aa.items, ba.items) |ai_item, bi_item| {
                if (!jsonValueEql(ai_item, bi_item)) break :blk false;
            }
            break :blk true;
        },
        .object => |ao| blk: {
            if (b != .object) break :blk false;
            const bo = b.object;
            if (ao.count() != bo.count()) break :blk false;
            var it = ao.iterator();
            while (it.next()) |kv| {
                const bval = bo.get(kv.key_ptr.*) orelse break :blk false;
                if (!jsonValueEql(kv.value_ptr.*, bval)) break :blk false;
            }
            break :blk true;
        },
    };
}

pub fn cloneJsonValue(allocator: Allocator, val: Value) !Value {
    return switch (val) {
        .null => .null,
        .bool => |b| .{ .bool = b },
        .integer => |i| .{ .integer = i },
        .float => |f| .{ .float = f },
        .number_string => |s| .{ .number_string = try allocator.dupe(u8, s) },
        .string => |s| .{ .string = try allocator.dupe(u8, s) },
        .array => |arr| blk: {
            var out = json.Array.init(allocator);
            for (arr.items) |item| {
                try out.append(try cloneJsonValue(allocator, item));
            }
            break :blk .{ .array = out };
        },
        .object => |obj| blk: {
            var out: json.ObjectMap = .empty;
            var it = obj.iterator();
            while (it.next()) |kv| {
                try out.put(
                    allocator,
                    try allocator.dupe(u8, kv.key_ptr.*),
                    try cloneJsonValue(allocator, kv.value_ptr.*),
                );
            }
            break :blk .{ .object = out };
        },
    };
}

pub fn parseValidatedCoremark(
    output: []const u8,
    engine: []const u8,
    expected_runs: usize,
) ProfileError!bench_coremark.ParsedCoreMark {
    if (expected_runs != 1) {
        return fail("{s}: profiled CoreMark run count must be one", .{engine});
    }
    return bench_coremark.parseCoremarkOutput(output, engine, bench_coremark.expected_iterations) catch |err| {
        return fail("{s}: failed to parse CRC-validated CoreMark output: {s}", .{ engine, @errorName(err) });
    };
}

pub fn aggregateWamrRankings(allocator: Allocator, reports: []const Value) ProfileError!Value {
    if (reports.len == 0) {
        return fail("no WAMR attribution reports to aggregate", .{});
    }
    if (reports[0] != .object) return fail("WAMR attribution report must be an object", .{});
    const text_size = jsonInt(reports[0].object.get("text_size")) orelse return fail("WAMR attribution report missing text_size", .{});
    const function_count = jsonInt(reports[0].object.get("function_count")) orelse return fail("WAMR attribution report missing function_count", .{});

    var samples: std.AutoHashMap(i64, i64) = .init(allocator);
    var code_bytes: std.AutoHashMap(i64, i64) = .init(allocator);
    var order_funcs: std.ArrayList(i64) = .empty;
    var total: i64 = 0;
    var attributed: i64 = 0;

    for (reports) |rep_val| {
        if (rep_val != .object) return fail("WAMR attribution report must be an object", .{});
        const rep = rep_val.object;
        if ((jsonInt(rep.get("text_size")) orelse -1) != text_size or
            (jsonInt(rep.get("function_count")) orelse -1) != function_count)
        {
            return fail("WAMR attribution captures disagree on cwasm layout", .{});
        }
        total += jsonInt(rep.get("total_samples")) orelse 0;
        attributed += jsonInt(rep.get("attributed_samples")) orelse 0;
        const top_val = rep.get("top_functions") orelse continue;
        if (top_val != .array) return fail("WAMR top_functions must be a list", .{});
        for (top_val.array.items) |item_val| {
            if (item_val != .object) return fail("WAMR top_functions item must be an object", .{});
            const item = item_val.object;
            const lf = jsonInt(item.get("local_func")) orelse return fail("WAMR top_functions item missing local_func", .{});
            const cb = jsonInt(item.get("code_bytes")) orelse return fail("WAMR top_functions item missing code_bytes", .{});
            const s = jsonInt(item.get("samples")) orelse return fail("WAMR top_functions item missing samples", .{});
            if (code_bytes.get(lf)) |existing_cb| {
                if (existing_cb != cb) {
                    return fail("WAMR local_func={d} code size changed across captures", .{lf});
                }
            } else {
                try code_bytes.put(lf, cb);
                try order_funcs.append(allocator, lf);
            }
            const prev = samples.get(lf) orelse 0;
            try samples.put(lf, prev + s);
        }
    }

    const Pair = struct { lf: i64, cnt: i64, ord: usize };
    var pairs: std.ArrayList(Pair) = .empty;
    for (order_funcs.items, 0..) |lf, ord| {
        try pairs.append(allocator, .{ .lf = lf, .cnt = samples.get(lf) orelse 0, .ord = ord });
    }
    std.mem.sort(Pair, pairs.items, {}, struct {
        fn lessThan(_: void, a: Pair, b: Pair) bool {
            if (a.cnt != b.cnt) return a.cnt > b.cnt;
            return a.ord < b.ord;
        }
    }.lessThan);

    var top_arr = json.Array.init(allocator);
    for (pairs.items) |p| {
        var obj: json.ObjectMap = .empty;
        try obj.put(allocator, "local_func", .{ .integer = p.lf });
        try obj.put(allocator, "samples", .{ .integer = p.cnt });
        try obj.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(p.cnt)) / @as(f64, @floatFromInt(total)) },
        );
        try obj.put(allocator, "code_bytes", .{ .integer = code_bytes.get(p.lf).? });
        try top_arr.append(.{ .object = obj });
    }

    var out: json.ObjectMap = .empty;
    try out.put(allocator, "total_samples", .{ .integer = total });
    try out.put(allocator, "attributed_samples", .{ .integer = attributed });
    try out.put(
        allocator,
        "attribution_coverage_pct",
        .{ .float = 100.0 * @as(f64, @floatFromInt(attributed)) / @as(f64, @floatFromInt(total)) },
    );
    try out.put(allocator, "text_size", .{ .integer = text_size });
    try out.put(allocator, "function_count", .{ .integer = function_count });
    try out.put(allocator, "top_functions", .{ .array = top_arr });
    return .{ .object = out };
}

pub fn validateWamrCapture(
    allocator: Allocator,
    report_val: Value,
    minimum_samples: i64,
    minimum_coverage_pct: f64,
) ProfileError!Value {
    if (report_val != .object) return fail("WAMR capture must be an object", .{});
    const report = report_val.object;
    const total = jsonInt(report.get("total_samples")) orelse {
        return fail("WAMR capture has null total samples; requires at least {d}", .{minimum_samples});
    };
    if (total < minimum_samples) {
        return fail("WAMR capture has {d} total samples; requires at least {d}", .{ total, minimum_samples });
    }
    const attributed = jsonInt(report.get("attributed_samples")) orelse {
        return fail("WAMR capture has invalid attributed sample count", .{});
    };
    if (attributed <= 0 or attributed > total) {
        return fail("WAMR capture has invalid attributed sample count", .{});
    }
    const coverage = 100.0 * @as(f64, @floatFromInt(attributed)) / @as(f64, @floatFromInt(total));
    if (coverage < minimum_coverage_pct) {
        return fail("WAMR capture attribution coverage {d:.4}% is below {d:.4}%", .{ coverage, minimum_coverage_pct });
    }
    const mapping_val = report.get("mapping") orelse {
        return fail("WAMR capture did not use automatic exact-size mmap attribution", .{});
    };
    if (mapping_val != .object) {
        return fail("WAMR capture did not use automatic exact-size mmap attribution", .{});
    }
    const mapping = mapping_val.object;
    const auth = jsonBool(mapping.get("authoritative")) orelse false;
    const has_override = if (mapping.get("override")) |ov| ov != .null else false;
    const sz = jsonInt(mapping.get("size"));
    const exp_sz = jsonInt(mapping.get("expected_size"));
    if (!auth or has_override or sz == null or exp_sz == null or sz.? != exp_sz.?) {
        return fail("WAMR capture did not use automatic exact-size mmap attribution", .{});
    }

    var out: json.ObjectMap = .empty;
    try out.put(allocator, "total_samples", .{ .integer = total });
    try out.put(allocator, "attributed_samples", .{ .integer = attributed });
    try out.put(allocator, "coverage_pct", .{ .float = coverage });
    try out.put(allocator, "mapping", try cloneJsonValue(allocator, mapping_val));
    return .{ .object = out };
}

pub const WasmtimeFuncSampleEntry = struct {
    samples: i64 = 0,
    names: std.StringHashMapUnmanaged(void) = .empty,
    offsets: std.AutoHashMapUnmanaged(usize, i64) = .empty,
    mapping_methods: std.StringHashMapUnmanaged(void) = .empty,
};

pub const ParsedWasmtimeSamples = struct {
    total_samples: i64 = 0,
    functions: std.AutoHashMapUnmanaged(usize, WasmtimeFuncSampleEntry) = .empty,
};

pub fn validateWasmtimeCapture(
    allocator: Allocator,
    capture: *const ParsedWasmtimeSamples,
    minimum_samples: i64,
    minimum_coverage_pct: f64,
) ProfileError!Value {
    const total = capture.total_samples;
    var attributed: i64 = 0;
    var it = capture.functions.iterator();
    while (it.next()) |kv| {
        attributed += kv.value_ptr.samples;
    }
    if (total < minimum_samples) {
        return fail("Wasmtime capture has {d} samples; requires at least {d}", .{ total, minimum_samples });
    }
    const coverage = 100.0 * @as(f64, @floatFromInt(attributed)) / @as(f64, @floatFromInt(total));
    if (coverage < minimum_coverage_pct) {
        return fail("Wasmtime capture attribution coverage {d:.4}% is below {d:.4}%", .{ coverage, minimum_coverage_pct });
    }
    var out: json.ObjectMap = .empty;
    try out.put(allocator, "total_samples", .{ .integer = total });
    try out.put(allocator, "attributed_samples", .{ .integer = attributed });
    try out.put(allocator, "coverage_pct", .{ .float = coverage });
    return .{ .object = out };
}

pub fn aggregateWamrFunction(
    allocator: Allocator,
    reports: []const Value,
    total_samples: i64,
    function_start: i64,
) ProfileError!Value {
    if (reports.len == 0) {
        return fail("no WAMR function reports to aggregate", .{});
    }
    if (reports[0] != .object) return fail("WAMR report must be an object", .{});
    const first_cf = reports[0].object.get("classified_function") orelse return fail("WAMR report missing classified_function", .{});
    if (first_cf != .object) return fail("WAMR classified_function must be an object", .{});
    const local_func = jsonInt(first_cf.object.get("local_func")) orelse return fail("WAMR classified_function missing local_func", .{});
    const inst_count = jsonInt(first_cf.object.get("instruction_count")) orelse 0;

    var class_samples: std.StringHashMap(i64) = .init(allocator);
    var static_counts: std.StringHashMap(i64) = .init(allocator);
    var all_class_names: std.StringHashMap(void) = .init(allocator);

    const HotKey = struct {
        offset: i64,
        instruction: []const u8,
    };
    const HotEntry = struct {
        key: HotKey,
        samples: i64,
        ord: usize,
    };
    var hot_entries: std.ArrayList(HotEntry) = .empty;
    var function_samples: i64 = 0;

    for (reports) |rep_val| {
        if (rep_val != .object) return fail("WAMR report must be an object", .{});
        const rep = rep_val.object;
        const cf_val = rep.get("classified_function") orelse return fail("WAMR report missing classified_function", .{});
        if (cf_val != .object) return fail("WAMR classified_function must be an object", .{});
        const cf = cf_val.object;
        if ((jsonInt(cf.get("local_func")) orelse -1) != local_func) {
            return fail("WAMR function captures disagree on local_func", .{});
        }
        function_samples += jsonInt(cf.get("samples")) orelse 0;
        const classes_val = cf.get("classes") orelse return fail("WAMR classified_function missing classes", .{});
        if (classes_val != .object) return fail("WAMR classified_function classes must be an object", .{});
        var cit = classes_val.object.iterator();
        while (cit.next()) |kv| {
            const cname = kv.key_ptr.*;
            if (kv.value_ptr.* != .object) continue;
            const cobj = kv.value_ptr.object;
            const static_cnt = jsonInt(cobj.get("static_instructions")) orelse 0;
            const samp = jsonInt(cobj.get("samples")) orelse 0;
            if (static_counts.get(cname)) |existing_st| {
                if (existing_st != static_cnt) {
                    return fail("WAMR {s} static count changed across captures", .{cname});
                }
            } else {
                const duped = try allocator.dupe(u8, cname);
                try static_counts.put(duped, static_cnt);
                try all_class_names.put(duped, {});
            }
            const prev_s = class_samples.get(cname) orelse 0;
            if (!class_samples.contains(cname)) {
                const duped = try allocator.dupe(u8, cname);
                try class_samples.put(duped, samp);
                try all_class_names.put(duped, {});
            } else {
                try class_samples.put(cname, prev_s + samp);
            }
        }

        const text_base = jsonInt(rep.get("text_base")) orelse 0;
        const function_base = text_base + function_start;
        if (cf.get("hottest_instructions")) |hot_val| {
            if (hot_val == .array) {
                for (hot_val.array.items) |hitem| {
                    if (hitem != .object) continue;
                    const hobj = hitem.object;
                    const addr = jsonInt(hobj.get("address")) orelse 0;
                    const offset = addr - function_base;
                    const inst_txt = jsonStringOrEmpty(hobj.get("instruction"));
                    const cnt = jsonInt(hobj.get("samples")) orelse 0;
                    var found: ?usize = null;
                    for (hot_entries.items, 0..) |existing, idx| {
                        if (existing.key.offset == offset and std.mem.eql(u8, existing.key.instruction, inst_txt)) {
                            found = idx;
                            break;
                        }
                    }
                    if (found) |idx| {
                        hot_entries.items[idx].samples += cnt;
                    } else {
                        try hot_entries.append(allocator, .{
                            .key = .{
                                .offset = offset,
                                .instruction = try allocator.dupe(u8, inst_txt),
                            },
                            .samples = cnt,
                            .ord = hot_entries.items.len,
                        });
                    }
                }
            }
        }
    }

    var sorted_names: std.ArrayList([]const u8) = .empty;
    var nit = all_class_names.keyIterator();
    while (nit.next()) |k| try sorted_names.append(allocator, k.*);
    std.mem.sort([]const u8, sorted_names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    var classes_out: json.ObjectMap = .empty;
    for (sorted_names.items) |cname| {
        const samp = class_samples.get(cname) orelse 0;
        const st = static_counts.get(cname) orelse 0;
        var cobj: json.ObjectMap = .empty;
        try cobj.put(allocator, "samples", .{ .integer = samp });
        try cobj.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(samp)) / @as(f64, @floatFromInt(total_samples)) },
        );
        try cobj.put(allocator, "static_instructions", .{ .integer = st });
        try classes_out.put(allocator, cname, .{ .object = cobj });
    }

    std.mem.sort(HotEntry, hot_entries.items, {}, struct {
        fn lessThan(_: void, a: HotEntry, b: HotEntry) bool {
            if (a.samples != b.samples) return a.samples > b.samples;
            return a.ord < b.ord;
        }
    }.lessThan);

    var hot_arr = json.Array.init(allocator);
    const limit = @min(hot_entries.items.len, 20);
    for (hot_entries.items[0..limit]) |he| {
        var hobj: json.ObjectMap = .empty;
        try hobj.put(allocator, "offset", .{ .integer = he.key.offset });
        try hobj.put(allocator, "instruction", .{ .string = he.key.instruction });
        try hobj.put(allocator, "samples", .{ .integer = he.samples });
        try hobj.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(he.samples)) / @as(f64, @floatFromInt(total_samples)) },
        );
        try hot_arr.append(.{ .object = hobj });
    }

    var out: json.ObjectMap = .empty;
    try out.put(allocator, "local_func", .{ .integer = local_func });
    try out.put(allocator, "samples", .{ .integer = function_samples });
    try out.put(
        allocator,
        "percent_of_run",
        .{ .float = 100.0 * @as(f64, @floatFromInt(function_samples)) / @as(f64, @floatFromInt(total_samples)) },
    );
    try out.put(allocator, "instruction_count", .{ .integer = inst_count });
    try out.put(allocator, "classes", .{ .object = classes_out });
    try out.put(allocator, "hottest_instructions", .{ .array = hot_arr });
    return .{ .object = out };
}

pub fn aggregateWasmtimeSamples(
    allocator: Allocator,
    captures: []const ParsedWasmtimeSamples,
) ProfileError!ParsedWasmtimeSamples {
    if (captures.len == 0) {
        return fail("no Wasmtime samples to aggregate", .{});
    }
    var out = ParsedWasmtimeSamples{};
    for (captures) |cap| {
        out.total_samples += cap.total_samples;
        var it = cap.functions.iterator();
        while (it.next()) |kv| {
            const gop = try out.functions.getOrPut(allocator, kv.key_ptr.*);
            if (!gop.found_existing) {
                gop.value_ptr.* = .{};
            }
            gop.value_ptr.samples += kv.value_ptr.samples;
            var nit = kv.value_ptr.names.keyIterator();
            while (nit.next()) |nm| {
                try gop.value_ptr.names.put(allocator, try allocator.dupe(u8, nm.*), {});
            }
            var oit = kv.value_ptr.offsets.iterator();
            while (oit.next()) |okv| {
                const prev = gop.value_ptr.offsets.get(okv.key_ptr.*) orelse 0;
                try gop.value_ptr.offsets.put(allocator, okv.key_ptr.*, prev + okv.value_ptr.*);
            }
            var mit = kv.value_ptr.mapping_methods.keyIterator();
            while (mit.next()) |mm| {
                try gop.value_ptr.mapping_methods.put(allocator, try allocator.dupe(u8, mm.*), {});
            }
        }
    }
    return out;
}

pub fn parseSpillMetrics(allocator: Allocator, text: []const u8) ProfileError!std.AutoHashMap(usize, Value) {
    const integer_fields = [_][]const u8{
        "local_func",
        "mod",
        "insts",
        "clobbers",
        "slots",
        "spilled_vregs",
        "scalar",
        "v128",
        "slots_scalar",
        "slots_v128",
        "spill_ld",
        "spill_st",
        "remat",
        "callee_saved",
    };

    var metrics: std.AutoHashMap(usize, Value) = .init(allocator);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (std.mem.indexOf(u8, line, "[aot-spill-metric]") == null) continue;
        var fields: json.ObjectMap = .empty;
        var tok_it = std.mem.tokenizeAny(u8, line, " \t");
        while (tok_it.next()) |token| {
            const eq = std.mem.indexOfScalar(u8, token, '=') orelse continue;
            const key = token[0..eq];
            const val_str = token[eq + 1 ..];
            var is_int = false;
            for (integer_fields) |int_f| {
                if (std.mem.eql(u8, key, int_f)) {
                    is_int = true;
                    break;
                }
            }
            if (is_int) {
                const parsed_int = std.fmt.parseInt(i64, val_str, 10) catch {
                    return fail("malformed spill metric line: {s}", .{line});
                };
                try fields.put(allocator, try allocator.dupe(u8, key), .{ .integer = parsed_int });
            } else {
                try fields.put(allocator, try allocator.dupe(u8, key), .{ .string = try allocator.dupe(u8, val_str) });
            }
        }
        const lf_val = jsonInt(fields.get("local_func")) orelse {
            return fail("malformed spill metric line: {s}", .{line});
        };
        if (lf_val < 0) return fail("malformed spill metric line: {s}", .{line});
        const lf: usize = @intCast(lf_val);
        if (metrics.contains(lf)) {
            return fail("duplicate spill metric for local_func={d}", .{lf});
        }
        try metrics.put(lf, .{ .object = fields });
    }
    return metrics;
}

const WasmtimeSymbolMatch = struct {
    module_idx: usize,
    func_idx: usize,
    name: ?[]const u8,
    end_pos: usize,
};

fn findWasmtimeSymbols(line: []const u8, allocator: Allocator) ![]WasmtimeSymbolMatch {
    var list: std.ArrayList(WasmtimeSymbolMatch) = .empty;
    var cursor: usize = 0;
    while (std.mem.indexOfPos(u8, line, cursor, "wasm[")) |start| {
        const after_wasm = start + "wasm[".len;
        const close_mod = std.mem.indexOfScalarPos(u8, line, after_wasm, ']') orelse {
            cursor = after_wasm;
            continue;
        };
        const mod_idx = std.fmt.parseInt(usize, line[after_wasm..close_mod], 10) catch {
            cursor = after_wasm;
            continue;
        };
        const rest1 = line[close_mod + 1 ..];
        if (!std.mem.startsWith(u8, rest1, "::function[")) {
            cursor = close_mod + 1;
            continue;
        }
        const after_fn = close_mod + 1 + "::function[".len;
        const close_fn = std.mem.indexOfScalarPos(u8, line, after_fn, ']') orelse {
            cursor = after_fn;
            continue;
        };
        const fn_idx = std.fmt.parseInt(usize, line[after_fn..close_fn], 10) catch {
            cursor = after_fn;
            continue;
        };
        var end_pos = close_fn + 1;
        var name_opt: ?[]const u8 = null;
        if (end_pos + 2 <= line.len and std.mem.eql(u8, line[end_pos .. end_pos + 2], "::")) {
            const name_start = end_pos + 2;
            var name_end = name_start;
            while (name_end < line.len and
                line[name_end] != '+' and
                line[name_end] != '(' and
                !std.ascii.isWhitespace(line[name_end])) : (name_end += 1)
            {}
            if (name_end > name_start) {
                name_opt = line[name_start..name_end];
                end_pos = name_end;
            }
        }
        try list.append(allocator, .{
            .module_idx = mod_idx,
            .func_idx = fn_idx,
            .name = name_opt,
            .end_pos = end_pos,
        });
        cursor = end_pos;
    }
    return list.toOwnedSlice(allocator);
}

fn findHexOffsetInTail(tail: []const u8) ?usize {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, tail, pos, "0x")) |ox| {
        const valid_prefix = (ox == 0) or
            tail[ox - 1] == '+' or
            std.ascii.isWhitespace(tail[ox - 1]);
        if (valid_prefix) {
            var hex_end = ox + 2;
            while (hex_end < tail.len and std.ascii.isHex(tail[hex_end])) : (hex_end += 1) {}
            if (hex_end > ox + 2) {
                const valid_suffix = (hex_end == tail.len) or
                    tail[hex_end] == '(' or
                    std.ascii.isWhitespace(tail[hex_end]);
                if (valid_suffix) {
                    if (std.fmt.parseInt(usize, tail[ox + 2 .. hex_end], 16)) |val| {
                        return val;
                    } else |_| {}
                }
            }
        }
        pos = ox + 2;
    }
    return null;
}

const PlainJittedMatch = struct {
    name: []const u8,
    offset: ?usize,
    dso: []const u8,
    local_idx: usize,
};

fn matchPlainJittedLine(stripped: []const u8) ?PlainJittedMatch {
    // Matches `^(?:0x)?[0-9a-fA-F]+\s+(?P<name>[^+\s(]+)(?:\+0x(?P<offset>[0-9a-fA-F]+))?\s+\((?P<dso>[^)]+/jitted-\d+-(?P<local>\d+)\.so)\)$`
    var i: usize = 0;
    if (std.mem.startsWith(u8, stripped, "0x") or std.mem.startsWith(u8, stripped, "0X")) {
        i = 2;
    }
    const hex_start = i;
    while (i < stripped.len and std.ascii.isHex(stripped[i])) : (i += 1) {}
    if (i == hex_start) return null;
    if (i >= stripped.len or !std.ascii.isWhitespace(stripped[i])) return null;
    while (i < stripped.len and std.ascii.isWhitespace(stripped[i])) : (i += 1) {}

    const name_start = i;
    while (i < stripped.len and stripped[i] != '+' and stripped[i] != '(' and !std.ascii.isWhitespace(stripped[i])) : (i += 1) {}
    if (i == name_start) return null;
    const name = stripped[name_start..i];

    var offset: ?usize = null;
    if (i + 3 < stripped.len and std.mem.eql(u8, stripped[i .. i + 3], "+0x")) {
        i += 3;
        const off_start = i;
        while (i < stripped.len and std.ascii.isHex(stripped[i])) : (i += 1) {}
        if (i == off_start) return null;
        offset = std.fmt.parseInt(usize, stripped[off_start..i], 16) catch return null;
    }
    if (i >= stripped.len or !std.ascii.isWhitespace(stripped[i])) return null;
    while (i < stripped.len and std.ascii.isWhitespace(stripped[i])) : (i += 1) {}
    if (i >= stripped.len or stripped[i] != '(' or stripped[stripped.len - 1] != ')') return null;
    const dso = stripped[i + 1 .. stripped.len - 1];
    if (!std.mem.endsWith(u8, dso, ".so")) return null;
    const slash_jit = std.mem.lastIndexOf(u8, dso, "/jitted-") orelse return null;
    const after_jit = dso[slash_jit + "/jitted-".len .. dso.len - 3];
    const dash = std.mem.indexOfScalar(u8, after_jit, '-') orelse return null;
    const pid_str = after_jit[0..dash];
    const local_str = after_jit[dash + 1 ..];
    if (pid_str.len == 0 or local_str.len == 0) return null;
    for (pid_str) |c| if (!std.ascii.isDigit(c)) return null;
    const local_idx = std.fmt.parseInt(usize, local_str, 10) catch return null;
    return .{
        .name = name,
        .offset = offset,
        .dso = dso,
        .local_idx = local_idx,
    };
}

pub fn parseWasmtimeSamples(
    allocator: Allocator,
    text: []const u8,
    identity: ?*const compare_hot_function.WasmModuleIdentity,
) ProfileError!ParsedWasmtimeSamples {
    var out = ParsedWasmtimeSamples{};

    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        const stripped = std.mem.trim(u8, line, " \t");
        if (stripped.len == 0 or stripped[0] == '#') continue;
        out.total_samples += 1;

        const matches = try findWasmtimeSymbols(stripped, allocator);
        if (matches.len > 1) {
            return fail("ambiguous Wasmtime sample mapping: {s}", .{line});
        }

        var wasm_index: usize = 0;
        var name_opt: ?[]const u8 = null;
        var offset_opt: ?usize = null;
        var mapping_method: []const u8 = "";

        if (matches.len == 1) {
            const m = matches[0];
            if (m.module_idx != 0) continue;
            wasm_index = m.func_idx;
            name_opt = m.name;
            offset_opt = findHexOffsetInTail(stripped[m.end_pos..]);
            mapping_method = "full_wasm_symbol";
        } else {
            const plain = matchPlainJittedLine(stripped) orelse continue;
            const id = identity orelse continue;
            var matched_indices: std.ArrayList(usize) = .empty;
            var fn_it = id.function_names.iterator();
            while (fn_it.next()) |kv| {
                if (std.mem.eql(u8, kv.value_ptr.*, plain.name)) {
                    try matched_indices.append(allocator, kv.key_ptr.*);
                }
            }
            if (matched_indices.items.len != 1) {
                if (matched_indices.items.len > 1) {
                    return fail("Wasmtime JIT symbol '{s}' is ambiguous", .{plain.name});
                }
                continue;
            }
            wasm_index = matched_indices.items[0];
            const expected_local = wasm_index - id.imported_function_count;
            if (expected_local != plain.local_idx) {
                return fail(
                    "Wasmtime JIT DSO {s} implies local function {d}, but name-section mapping for '{s}' implies {d}",
                    .{ plain.dso, plain.local_idx, plain.name, expected_local },
                );
            }
            name_opt = plain.name;
            offset_opt = plain.offset;
            mapping_method = "name_section_and_jitted_dso_local_index";
        }

        const gop = try out.functions.getOrPut(allocator, wasm_index);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{};
        }
        gop.value_ptr.samples += 1;
        try gop.value_ptr.mapping_methods.put(allocator, mapping_method, {});
        if (name_opt) |nm| {
            if (nm.len > 0) {
                try gop.value_ptr.names.put(allocator, try allocator.dupe(u8, nm), {});
            }
        }
        if (offset_opt) |off| {
            const prev = gop.value_ptr.offsets.get(off) orelse 0;
            try gop.value_ptr.offsets.put(allocator, off, prev + 1);
        }
    }

    if (out.total_samples == 0) {
        return fail("perf script produced no Wasmtime samples", .{});
    }
    return out;
}

pub fn validateWasmtimeMapping(
    parsed: *const ParsedWasmtimeSamples,
    identity: *const compare_hot_function.WasmModuleIdentity,
) ProfileError!void {
    var it = parsed.functions.iterator();
    while (it.next()) |kv| {
        const wasm_index = kv.key_ptr.*;
        const entry = kv.value_ptr;
        if (wasm_index < identity.imported_function_count) {
            return fail("Wasmtime mapped generated code to imported function {d}", .{wasm_index});
        }
        if (wasm_index >= identity.functionCount()) {
            return fail("Wasmtime function index {d} exceeds module function count", .{wasm_index});
        }
        const expected_name = identity.function_names.get(wasm_index);
        if (entry.names.count() > 1) {
            return fail("Wasmtime function {d} has ambiguous names", .{wasm_index});
        }
        if (entry.names.count() == 1 and expected_name != null) {
            var nit = entry.names.keyIterator();
            const actual_name = nit.next().?.*;
            if (!std.mem.eql(u8, actual_name, expected_name.?)) {
                return fail("Wasmtime function {d} name mismatch: {s} != {s}", .{ wasm_index, actual_name, expected_name.? });
            }
        }
    }
}

pub fn classSamples(classes_val: Value, names: []const []const u8) i64 {
    if (classes_val != .object) return 0;
    var sum: i64 = 0;
    for (names) |name| {
        if (classes_val.object.get(name)) |v| {
            if (v == .object) {
                sum += jsonInt(v.object.get("samples")) orelse 0;
            }
        }
    }
    return sum;
}

pub fn classifyWasmtimeFunctionWithParsed(
    allocator: Allocator,
    parsed: []const compare_hot_function.Instruction,
    wasm_index: usize,
    offsets: *const std.AutoHashMapUnmanaged(usize, i64),
    total_samples: i64,
) ProfileError!Value {
    var instructions = try allocator.alloc(aot_jit_attr.Instruction, parsed.len);
    for (parsed, 0..) |item, idx| {
        instructions[idx] = .{
            .address = item.offset,
            .offset = item.offset,
            .size = item.size,
            .text = item.text,
        };
    }
    const classes = aot_jit_attr.classifyInstructionStream(allocator, instructions, null, "aarch64") catch {
        return fail("failed to classify Wasmtime instructions: {s}", .{aot_jit_attr.lastError()});
    };

    var by_class: std.StringHashMap(i64) = .init(allocator);
    var static_counts: std.StringHashMap(i64) = .init(allocator);
    var all_names: std.StringHashMap(void) = .init(allocator);
    for (classes) |cname| {
        const prev = static_counts.get(cname) orelse 0;
        try static_counts.put(cname, prev + 1);
        try all_names.put(cname, {});
    }

    var by_instruction: std.AutoHashMap(usize, i64) = .init(allocator);
    var unresolved: i64 = 0;
    var function_samples: i64 = 0;

    var oit = offsets.iterator();
    while (oit.next()) |kv| {
        const offset = kv.key_ptr.*;
        const samples = kv.value_ptr.*;
        function_samples += samples;

        // bisect_right(starts, offset) - 1
        var matched_idx: ?usize = null;
        for (parsed, 0..) |inst, idx| {
            if (inst.offset <= offset) {
                matched_idx = idx;
            } else {
                break;
            }
        }
        if (matched_idx == null or offset >= parsed[matched_idx.?].offset + parsed[matched_idx.?].size) {
            unresolved += samples;
            continue;
        }
        const idx = matched_idx.?;
        const cname = classes[idx];
        const prev_c = by_class.get(cname) orelse 0;
        try by_class.put(cname, prev_c + samples);
        try all_names.put(cname, {});
        const prev_i = by_instruction.get(idx) orelse 0;
        try by_instruction.put(idx, prev_i + samples);
    }

    const mapped = function_samples - unresolved;
    if (function_samples > 0 and mapped * 100 < function_samples * 90) {
        return fail(
            "Wasmtime function {d} instruction mapping covered only {d}/{d} samples",
            .{ wasm_index, mapped, function_samples },
        );
    }

    const HotInst = struct { idx: usize, samples: i64 };
    var hot_list: std.ArrayList(HotInst) = .empty;
    var bit = by_instruction.iterator();
    while (bit.next()) |kv| {
        try hot_list.append(allocator, .{ .idx = kv.key_ptr.*, .samples = kv.value_ptr.* });
    }
    std.mem.sort(HotInst, hot_list.items, {}, struct {
        fn lessThan(_: void, a: HotInst, b: HotInst) bool {
            if (a.samples != b.samples) return a.samples > b.samples;
            return a.idx < b.idx;
        }
    }.lessThan);

    var hot_arr = json.Array.init(allocator);
    const hot_limit = @min(hot_list.items.len, 20);
    for (hot_list.items[0..hot_limit]) |hi| {
        const item = parsed[hi.idx];
        var hobj: json.ObjectMap = .empty;
        try hobj.put(allocator, "offset", .{ .integer = @intCast(item.offset) });
        try hobj.put(allocator, "samples", .{ .integer = hi.samples });
        try hobj.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(hi.samples)) / @as(f64, @floatFromInt(total_samples)) },
        );
        try hobj.put(allocator, "instruction", .{ .string = try allocator.dupe(u8, item.text) });
        try hot_arr.append(.{ .object = hobj });
    }

    var sorted_names: std.ArrayList([]const u8) = .empty;
    var nit = all_names.keyIterator();
    while (nit.next()) |k| try sorted_names.append(allocator, k.*);
    std.mem.sort([]const u8, sorted_names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    var classes_out: json.ObjectMap = .empty;
    for (sorted_names.items) |cname| {
        const samp = by_class.get(cname) orelse 0;
        const st = static_counts.get(cname) orelse 0;
        var cobj: json.ObjectMap = .empty;
        try cobj.put(allocator, "samples", .{ .integer = samp });
        try cobj.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(samp)) / @as(f64, @floatFromInt(total_samples)) },
        );
        try cobj.put(allocator, "static_instructions", .{ .integer = st });
        try classes_out.put(allocator, try allocator.dupe(u8, cname), .{ .object = cobj });
    }

    var out: json.ObjectMap = .empty;
    try out.put(allocator, "instruction_count", .{ .integer = @intCast(parsed.len) });
    try out.put(allocator, "mapped_instruction_samples", .{ .integer = mapped });
    try out.put(allocator, "unresolved_instruction_samples", .{ .integer = unresolved });
    try out.put(
        allocator,
        "mapping_coverage_pct",
        .{ .float = if (function_samples > 0)
            100.0 * @as(f64, @floatFromInt(mapped)) / @as(f64, @floatFromInt(function_samples))
        else
            0.0 },
    );
    try out.put(allocator, "classes", .{ .object = classes_out });
    try out.put(allocator, "hottest_instructions", .{ .array = hot_arr });
    return .{ .object = out };
}

pub fn classifyWasmtimeFunction(
    allocator: Allocator,
    objdump_text: []const u8,
    wasm_index: usize,
    offsets: *const std.AutoHashMapUnmanaged(usize, i64),
    total_samples: i64,
) ProfileError!Value {
    const parsed = compare_hot_function.parseDisassembly(allocator, objdump_text, wasm_index) catch |err| {
        return fail("failed to parse Wasmtime disassembly: {s}", .{@errorName(err)});
    };
    return classifyWasmtimeFunctionWithParsed(allocator, parsed, wasm_index, offsets, total_samples);
}

pub fn wasmtimeFunctionAddressBase(objdump_text: []const u8, wasm_index: usize) ProfileError!i64 {
    var lines = std.mem.splitScalar(u8, objdump_text, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (compare_hot_function.parseWasmtimeHeader(line)) |hdr| {
            if (hdr.module_index == 0 and hdr.function_index == wasm_index) {
                if (hdr.address) |addr| {
                    return @intCast(addr);
                }
                return fail("Wasmtime function {d} header lacks an address", .{wasm_index});
            }
        }
    }
    return fail("Wasmtime objdump lacks function header for wasm index {d}", .{wasm_index});
}

pub const MappedInstructionSamples = struct {
    mapped: std.AutoHashMapUnmanaged(usize, i64) = .empty,
    unresolved: i64 = 0,
};

pub fn mapSamplesToInstructionStarts(
    allocator: Allocator,
    instructions: []const compare_hot_function.Instruction,
    offsets: *const std.AutoHashMapUnmanaged(usize, i64),
) !MappedInstructionSamples {
    var out = MappedInstructionSamples{};
    var it = offsets.iterator();
    while (it.next()) |kv| {
        const offset = kv.key_ptr.*;
        const samples = kv.value_ptr.*;
        var matched_idx: ?usize = null;
        for (instructions, 0..) |inst, idx| {
            if (inst.offset <= offset) {
                matched_idx = idx;
            } else {
                break;
            }
        }
        if (matched_idx == null or offset >= instructions[matched_idx.?].offset + instructions[matched_idx.?].size) {
            out.unresolved += samples;
            continue;
        }
        const start_off = instructions[matched_idx.?].offset;
        const prev = out.mapped.get(start_off) orelse 0;
        try out.mapped.put(allocator, start_off, prev + samples);
    }
    return out;
}

pub fn analyzeWasmtimeAluProvenance(
    allocator: Allocator,
    objdump_text: []const u8,
    wasm_index: usize,
    offsets: *const std.AutoHashMapUnmanaged(usize, i64),
    total_samples: i64,
    global_attributed_samples: ?i64,
) ProfileError!Value {
    const instructions = compare_hot_function.parseDisassembly(allocator, objdump_text, wasm_index) catch |err| {
        return fail("failed to parse Wasmtime disassembly: {s}", .{@errorName(err)});
    };
    var adapter = try allocator.alloc(aot_jit_attr.Instruction, instructions.len);
    var prov_inputs = try allocator.alloc(provenance.InstructionInput, instructions.len);
    for (instructions, 0..) |item, idx| {
        adapter[idx] = .{
            .address = item.offset,
            .offset = item.offset,
            .size = item.size,
            .text = item.text,
        };
        prov_inputs[idx] = .{
            .offset = @intCast(item.offset),
            .size = @intCast(item.size),
            .mnemonic = item.mnemonic,
            .operands = item.operands,
            .text = item.text,
        };
    }
    const broad_classes = aot_jit_attr.classifyInstructionStream(allocator, adapter, null, "aarch64") catch {
        return fail("failed to classify Wasmtime instruction stream: {s}", .{aot_jit_attr.lastError()});
    };
    const mapped_res = try mapSamplesToInstructionStarts(allocator, instructions, offsets);

    var sample_entries: std.ArrayList(provenance.SampleEntry) = .empty;
    var mapped_sum: i64 = 0;
    var mit = mapped_res.mapped.iterator();
    while (mit.next()) |kv| {
        mapped_sum += kv.value_ptr.*;
        try sample_entries.append(allocator, .{
            .offset = @intCast(kv.key_ptr.*),
            .samples = kv.value_ptr.*,
        });
    }

    const addr_base = try wasmtimeFunctionAddressBase(objdump_text, wasm_index);
    var diag = provenance.Diagnostic{};
    var result = provenance.analyzeInstructionStream(
        allocator,
        prov_inputs,
        .{
            .broad_classes = broad_classes,
            .samples_by_offset = sample_entries.items,
            .total_run_samples = total_samples,
            .address_base = addr_base,
            .global_attributed_samples = global_attributed_samples,
        },
        &diag,
    ) catch {
        return fail("{s}", .{diag.message});
    };

    var sm_obj: json.ObjectMap = .empty;
    try sm_obj.put(allocator, "mapped_function_samples", .{ .integer = mapped_sum });
    try sm_obj.put(allocator, "unresolved_function_samples", .{ .integer = mapped_res.unresolved });
    try result.object.put(allocator, "sample_mapping", .{ .object = sm_obj });
    return result;
}

pub const CaptureCountsEntry = struct {
    counts: std.AutoHashMapUnmanaged(u64, u64),
    text_base: u64,
};

pub fn analyzeWamrAluProvenance(
    allocator: Allocator,
    io: std.Io,
    cwasm_info: aot_jit_attr.CwasmInfo,
    local_func: i64,
    capture_counts: []const CaptureCountsEntry,
    total_samples: i64,
    scratch_dir: []const u8,
    global_attributed_samples: ?i64,
) ProfileError!Value {
    const bounds = aot_jit_attr.functionBounds(cwasm_info, local_func) catch {
        return fail("{s}", .{aot_jit_attr.lastError()});
    };
    const function_code = cwasm_info.data[cwasm_info.text_file_offset + bounds.start .. cwasm_info.text_file_offset + bounds.end];
    const label = try std.fmt.allocPrint(allocator, "alu-provenance-{d}", .{local_func});
    const instructions = aot_jit_attr.disassembleBlob(
        allocator,
        io,
        function_code,
        0,
        scratch_dir,
        label,
        0,
        "aarch64",
    ) catch {
        return fail("{s}", .{aot_jit_attr.lastError()});
    };
    const broad_classes = aot_jit_attr.classifyInstructionStream(allocator, instructions, null, "aarch64") catch {
        return fail("{s}", .{aot_jit_attr.lastError()});
    };

    var inst_offsets: std.AutoHashMap(usize, void) = .init(allocator);
    var prov_inputs = try allocator.alloc(provenance.InstructionInput, instructions.len);
    for (instructions, 0..) |inst, idx| {
        try inst_offsets.put(inst.offset, {});
        prov_inputs[idx] = .{
            .offset = @intCast(inst.offset),
            .size = @intCast(inst.size),
            .text = inst.text,
            .address = @intCast(inst.address),
        };
    }

    var samples_map: std.AutoHashMap(usize, i64) = .init(allocator);
    var unresolved: i64 = 0;
    for (capture_counts) |cc| {
        const function_base = cc.text_base + bounds.start;
        var cit = cc.counts.iterator();
        while (cit.next()) |kv| {
            const addr = kv.key_ptr.*;
            const cnt: i64 = @intCast(kv.value_ptr.*);
            if (addr < function_base or addr >= function_base + function_code.len) continue;
            const offset: usize = @intCast(addr - function_base);
            if (inst_offsets.contains(offset)) {
                const prev = samples_map.get(offset) orelse 0;
                try samples_map.put(offset, prev + cnt);
            } else {
                unresolved += cnt;
            }
        }
    }

    var sample_entries: std.ArrayList(provenance.SampleEntry) = .empty;
    var mapped_sum: i64 = 0;
    var sit = samples_map.iterator();
    while (sit.next()) |kv| {
        mapped_sum += kv.value_ptr.*;
        try sample_entries.append(allocator, .{
            .offset = @intCast(kv.key_ptr.*),
            .samples = kv.value_ptr.*,
        });
    }

    var diag = provenance.Diagnostic{};
    var result = provenance.analyzeInstructionStream(
        allocator,
        prov_inputs,
        .{
            .broad_classes = broad_classes,
            .samples_by_offset = sample_entries.items,
            .total_run_samples = total_samples,
            .global_attributed_samples = global_attributed_samples,
        },
        &diag,
    ) catch {
        return fail("{s}", .{diag.message});
    };

    var sm_obj: json.ObjectMap = .empty;
    try sm_obj.put(allocator, "mapped_function_samples", .{ .integer = mapped_sum });
    try sm_obj.put(allocator, "unresolved_function_samples", .{ .integer = unresolved });
    try result.object.put(allocator, "sample_mapping", .{ .object = sm_obj });
    return result;
}

pub const FrameProvenanceHooks = struct {
    bounds: ?aot_jit_attr.FunctionBounds = null,
    metadata: ?aot_jit_attr.FrameMetadata = null,
    instructions: ?[]const aot_jit_attr.Instruction = null,
    summary_json: ?Value = null,
};

pub fn analyzeWamrFrameProvenance(
    allocator: Allocator,
    io: std.Io,
    cwasm_info: aot_jit_attr.CwasmInfo,
    metadata_path: []const u8,
    local_func: i64,
    capture_counts: []const CaptureCountsEntry,
    ranking_reports: []const Value,
    total_samples: i64,
    spill_metric: Value,
    scratch_dir: []const u8,
    hooks: FrameProvenanceHooks,
) ProfileError!Value {
    const bounds = hooks.bounds orelse (aot_jit_attr.functionBounds(cwasm_info, local_func) catch {
        return fail("{s}", .{aot_jit_attr.lastError()});
    });
    const text = cwasm_info.data[cwasm_info.text_file_offset .. cwasm_info.text_file_offset + cwasm_info.text_size];
    const metadata = hooks.metadata orelse (aot_jit_attr.loadFrameMetadata(
        allocator,
        io,
        metadata_path,
        local_func,
        text[bounds.start..bounds.end],
        cwasm_info.version,
        text,
        bounds.start,
        cwasm_info,
    ) catch {
        return fail("{s}", .{aot_jit_attr.lastError()});
    });
    if (metadata.module != 0) {
        return fail("CoreMark frame metadata must describe module 0", .{});
    }
    const instructions = hooks.instructions orelse blk: {
        const label = try std.fmt.allocPrint(allocator, "frame-provenance-{d}", .{local_func});
        const dis = aot_jit_attr.disassembleFunction(
            allocator,
            io,
            text[bounds.start..bounds.end],
            bounds.start,
            scratch_dir,
            label,
            metadata.inline_data_ranges,
            metadata.architecture,
        ) catch {
            return fail("{s}", .{aot_jit_attr.lastError()});
        };
        aot_jit_attr.validateMetadataDisassembly(allocator, &metadata, dis) catch {
            return fail("{s}", .{aot_jit_attr.lastError()});
        };
        break :blk dis;
    };

    if (capture_counts.len != ranking_reports.len) {
        return fail("frame counts and ranking captures disagree", .{});
    }

    var combined: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    var captures_arr = json.Array.init(allocator);
    var combined_sum: i64 = 0;

    for (capture_counts, ranking_reports, 1..) |cc, ranking_val, ordinal| {
        if (ranking_val != .object) return fail("ranking report must be an object", .{});
        const ranking = ranking_val.object;

        const OffPair = struct { addr_rel: u64, count: u64 };
        var norm_pairs: std.ArrayList(OffPair) = .empty;
        var norm_sum: i64 = 0;

        var cit = cc.counts.iterator();
        while (cit.next()) |kv| {
            const addr = kv.key_ptr.*;
            const cnt = kv.value_ptr.*;
            if (addr >= cc.text_base + bounds.start and addr < cc.text_base + bounds.end) {
                const rel = addr - cc.text_base;
                try norm_pairs.append(allocator, .{ .addr_rel = rel, .count = cnt });
                norm_sum += @intCast(cnt);
                const prev = combined.get(rel) orelse 0;
                try combined.put(allocator, rel, prev + cnt);
            }
        }
        std.mem.sort(OffPair, norm_pairs.items, {}, struct {
            fn lessThan(_: void, a: OffPair, b: OffPair) bool {
                return a.addr_rel < b.addr_rel;
            }
        }.lessThan);

        var expected: i64 = 0;
        if (ranking.get("top_functions")) |tf_val| {
            if (tf_val == .array) {
                for (tf_val.array.items) |item| {
                    if (item == .object) {
                        if ((jsonInt(item.object.get("local_func")) orelse -1) == local_func) {
                            expected += jsonInt(item.object.get("samples")) orelse 0;
                        }
                    }
                }
            }
        }
        if (norm_sum != expected or expected <= 0) {
            return fail("frame function samples do not match the captured function ranking", .{});
        }
        combined_sum += norm_sum;

        var sbo_arr = json.Array.init(allocator);
        for (norm_pairs.items) |np| {
            var pair_arr = json.Array.init(allocator);
            try pair_arr.append(.{ .integer = @intCast(np.addr_rel - bounds.start) });
            try pair_arr.append(.{ .integer = @intCast(np.count) });
            try sbo_arr.append(.{ .array = pair_arr });
        }

        var cap_obj: json.ObjectMap = .empty;
        try cap_obj.put(allocator, "ordinal", .{ .integer = @intCast(ordinal) });
        try cap_obj.put(allocator, "total_samples", .{ .integer = jsonInt(ranking.get("total_samples")) orelse 0 });
        try cap_obj.put(allocator, "function_samples", .{ .integer = expected });
        try cap_obj.put(allocator, "samples_by_offset", .{ .array = sbo_arr });
        try captures_arr.append(.{ .object = cap_obj });
    }

    var summary_val = if (hooks.summary_json) |sj|
        try cloneJsonValue(allocator, sj)
    else blk: {
        const summary_struct = aot_jit_attr.buildFrameSummary(allocator, instructions, &combined, &metadata) catch {
            return fail("{s}", .{aot_jit_attr.lastError()});
        };
        var aw: std.Io.Writer.Allocating = .init(allocator);
        aot_jit_attr.writeFrameSummaryJson(&aw.writer, &summary_struct) catch return error.OutOfMemory;
        const json_bytes = try aw.toOwnedSlice();
        const parsed_json = json.parseFromSlice(Value, allocator, json_bytes, .{}) catch return error.OutOfMemory;
        break :blk parsed_json.value;
    };

    const rec = summary_val.object.get("reconciliation").?.object;
    const spill_ld = if (spill_metric == .object) (jsonInt(spill_metric.object.get("spill_ld")) orelse 0) else 0;
    const spill_st = if (spill_metric == .object) (jsonInt(spill_metric.object.get("spill_st")) orelse 0) else 0;
    if (!(jsonBool(rec.get("matches")) orelse false) or
        (jsonInt(rec.get("emitted_allocator_loads")) orelse -1) != spill_ld or
        (jsonInt(rec.get("emitted_allocator_stores")) orelse -1) != spill_st)
    {
        return fail("frame origins disagree with the emitted spill metric log", .{});
    }

    var origins_it = summary_val.object.getPtr("origins").?.object.iterator();
    while (origins_it.next()) |kv| {
        const s = jsonInt(kv.value_ptr.object.get("samples")) orelse 0;
        try kv.value_ptr.object.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(s)) / @as(f64, @floatFromInt(total_samples)) },
        );
    }

    for (summary_val.object.getPtr("allocator_contributors").?.array.items) |*contrib| {
        const s = jsonInt(contrib.object.get("samples")) orelse 0;
        try contrib.object.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(s)) / @as(f64, @floatFromInt(total_samples)) },
        );
    }

    const meta_sha = bench_coremark.sha256File(allocator, io, metadata_path) catch {
        return fail("cannot hash frame metadata {s}", .{metadata_path});
    };

    var meta_obj: json.ObjectMap = .empty;
    try meta_obj.put(allocator, "path", .{ .string = try allocator.dupe(u8, std.fs.path.basename(metadata_path)) });
    try meta_obj.put(allocator, "sha256", .{ .string = meta_sha });
    try meta_obj.put(allocator, "schema", .{ .string = try allocator.dupe(u8, metadata.schema) });
    try meta_obj.put(allocator, "schema_version", .{ .integer = metadata.schema_version });
    try meta_obj.put(allocator, "architecture", .{ .string = try allocator.dupe(u8, metadata.architecture) });
    try meta_obj.put(allocator, "abi", .{ .string = try allocator.dupe(u8, metadata.abi) });
    try meta_obj.put(allocator, "compiler_build_id", .{ .string = try allocator.dupe(u8, metadata.compiler_build_id) });
    try meta_obj.put(allocator, "module_text_sha256", .{ .string = try allocator.dupe(u8, metadata.module_text_sha256) });
    try meta_obj.put(allocator, "normalized_code_sha256", .{ .string = try allocator.dupe(u8, metadata.normalized_code_sha256) });

    var out: json.ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = 1 });
    try out.put(allocator, "module", .{ .integer = 0 });
    try out.put(allocator, "local_func", .{ .integer = local_func });
    try out.put(allocator, "code_size", .{ .integer = @intCast(bounds.end - bounds.start) });
    try out.put(allocator, "function_samples", .{ .integer = combined_sum });
    try out.put(allocator, "total_samples", .{ .integer = total_samples });
    try out.put(allocator, "sample_coordinates", .{ .string = "function-relative native byte offsets" });
    try out.put(allocator, "metadata", .{ .object = meta_obj });
    try out.put(allocator, "captures", .{ .array = captures_arr });
    try out.put(allocator, "summary", summary_val);
    return .{ .object = out };
}

pub fn assembleAluProvenance(
    allocator: Allocator,
    wamr_val: Value,
    wasmtime_val: Value,
    expected_wamr_all_alu_samples: i64,
    expected_wasmtime_all_alu_samples: i64,
) ProfileError!Value {
    if (wamr_val != .object or wasmtime_val != .object) {
        return fail("ALU provenance inputs must be JSON objects", .{});
    }
    if (std.mem.eql(u8, jsonStringOrEmpty(wamr_val.object.get("status")), "measured")) {
        if ((jsonInt(wamr_val.object.get("broad_alu_samples")) orelse -1) != expected_wamr_all_alu_samples) {
            return fail("WAMR narrow ALU partitions do not match existing all_alu samples", .{});
        }
    }
    if (std.mem.eql(u8, jsonStringOrEmpty(wasmtime_val.object.get("status")), "measured")) {
        if ((jsonInt(wasmtime_val.object.get("broad_alu_samples")) orelse -1) != expected_wasmtime_all_alu_samples) {
            return fail("Wasmtime narrow ALU partitions do not match existing all_alu samples", .{});
        }
    }

    var diag = provenance.Diagnostic{};
    const gate = provenance.compareEngineAnalyses(allocator, wamr_val, wasmtime_val, 5.0, &diag) catch {
        return fail("{s}", .{diag.message});
    };

    var out: json.ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = provenance.SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = provenance.ANALYSIS_KIND });
    try out.put(allocator, "wording", .{ .string = NARROW_ALU_WORDING });
    try out.put(allocator, "wamr", try cloneJsonValue(allocator, wamr_val));
    try out.put(allocator, "wasmtime", try cloneJsonValue(allocator, wasmtime_val));
    try out.put(allocator, "gate", gate);
    return .{ .object = out };
}

pub fn profileReportStatus(allocator: Allocator, report_val: Value) ProfileError!Value {
    if (report_val != .object) return fail("profile report root must be a JSON object", .{});
    const report = report_val.object;
    if (!std.mem.eql(u8, jsonStringOrEmpty(report.get("kind")), REPORT_KIND)) {
        return fail("profile report has an unsupported kind", .{});
    }
    if ((jsonInt(report.get("schema_version")) orelse -1) == HISTORICAL_REPORT_SCHEMA_VERSION) {
        const baseline_run = report.get("authoritative_baseline_run") orelse .null;
        const is_known = (jsonInt(baseline_run) orelse -1) == HISTORICAL_BASELINE_RUN;
        var out: json.ObjectMap = .empty;
        try out.put(allocator, "status", .{ .string = "historical-unverified" });
        try out.put(allocator, "authoritative", .{ .bool = false });
        try out.put(allocator, "baseline_run", baseline_run);
        try out.put(allocator, "known_historical_baseline", .{ .bool = is_known });
        try out.put(
            allocator,
            "reason",
            .{ .string = "schema version 1 named a historical run without embedding validated benchmark/report provenance" },
        );
        return .{ .object = out };
    }
    try validateReport(allocator, report_val);
    var out: json.ObjectMap = .empty;
    try out.put(allocator, "status", .{ .string = "current" });
    try out.put(allocator, "authoritative", .{ .bool = true });
    return .{ .object = out };
}

fn requireSha(val: ?Value, label: []const u8, length: usize) ProfileError![]const u8 {
    const s = jsonString(val) orelse return fail("{s} must be a {d}-character lowercase hex digest", .{ label, length });
    if (s.len != length) return fail("{s} must be a {d}-character lowercase hex digest", .{ label, length });
    for (s) |c| {
        if (!((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'))) {
            return fail("{s} must be a {d}-character lowercase hex digest", .{ label, length });
        }
    }
    return s;
}

fn validateNarrowAluAnalysis(
    allocator: Allocator,
    analysis_val: Value,
    engine: []const u8,
    expected_samples: i64,
    function_samples: i64,
    total_samples: i64,
    global_attributed_samples: i64,
) ProfileError!void {
    var diag = provenance.Diagnostic{};
    provenance.validateAnalysisSamples(allocator, analysis_val, engine, &diag) catch {
        return fail("{s}", .{diag.message});
    };
    const analysis = analysis_val.object;
    if ((jsonInt(analysis.get("total_run_samples")) orelse -1) != total_samples or
        (jsonInt(analysis.get("broad_alu_samples")) orelse -1) != expected_samples)
    {
        return fail("{s} narrow ALU samples do not reconcile to all_alu", .{engine});
    }
    const gsm = analysis.get("global_sample_mapping") orelse return fail("{s} global sample mapping is invalid", .{engine});
    if (gsm != .object or (jsonInt(gsm.object.get("attributed_samples")) orelse -1) != global_attributed_samples) {
        return fail("{s} global sample mapping is invalid", .{engine});
    }
    const isa = analysis.get("instruction_sample_accounting") orelse return fail("{s} instruction sample accounting is invalid", .{engine});
    const mapped = if (isa == .object) (jsonInt(isa.object.get("mapped_instruction_samples")) orelse -1) else -1;
    const unresolved = if (analysis.get("sample_mapping")) |sm|
        if (sm == .object) (jsonInt(sm.object.get("unresolved_function_samples")) orelse 0) else 0
    else
        0;
    if (mapped + unresolved != function_samples) {
        return fail("{s} instruction samples do not reconcile to its matched function", .{engine});
    }
}

fn frameCount(val: ?Value, name: []const u8) ProfileError!i64 {
    const v = val orelse return fail("invalid frame {s} count", .{name});
    if (v != .integer or v.integer < 0) {
        return fail("invalid frame {s} count", .{name});
    }
    return v.integer;
}

pub fn validateFrameProvenance(frame_val: Value, report_val: Value) ProfileError!void {
    if (frame_val != .object or report_val != .object) {
        return fail("frame provenance disagrees with the measured artifact", .{});
    }
    const frame = frame_val.object;
    const report = report_val.object;

    const local_func = try frameCount(frame.get("local_func"), "local function");
    const engines = (report.get("engines") orelse return fail("frame provenance disagrees with the measured artifact", .{})).object;
    const wamr_eng = (engines.get("wamr") orelse return fail("frame provenance disagrees with the measured artifact", .{})).object;
    const total = jsonInt(wamr_eng.get("total_samples")) orelse return fail("frame provenance disagrees with the measured artifact", .{});
    const wasm_obj = (report.get("wasm") orelse return fail("frame provenance disagrees with the measured artifact", .{})).object;
    const local_fn_cnt = jsonInt(wasm_obj.get("local_function_count")) orelse 0;
    const imported_fn_cnt = jsonInt(wasm_obj.get("imported_function_count")) orelse 0;
    const wamr_obj = (report.get("wamr") orelse return fail("frame provenance disagrees with the measured artifact", .{})).object;

    if ((jsonInt(frame.get("schema_version")) orelse -1) != 1 or
        (jsonInt(frame.get("module")) orelse -1) != 0 or
        local_func >= local_fn_cnt or
        (jsonInt(frame.get("wasm_function_index")) orelse -1) != local_func + imported_fn_cnt or
        (jsonInt(frame.get("total_samples")) orelse -1) != total or
        !std.mem.eql(u8, jsonStringOrEmpty(frame.get("compiler_sha256")), jsonStringOrEmpty(wamr_obj.get("compiler_sha256"))) or
        !std.mem.eql(u8, jsonStringOrEmpty(frame.get("cwasm_sha256")), jsonStringOrEmpty(wamr_obj.get("cwasm_sha256"))) or
        !std.mem.eql(u8, jsonStringOrEmpty(frame.get("sample_coordinates")), "function-relative native byte offsets"))
    {
        return fail("frame provenance disagrees with the measured artifact", .{});
    }

    const meta_val = frame.get("metadata") orelse return fail("frame metadata identity is incompatible", .{});
    if (meta_val != .object) return fail("frame metadata identity is incompatible", .{});
    const metadata = meta_val.object;
    var exp_path_buf: [128]u8 = undefined;
    const exp_path = std.fmt.bufPrint(&exp_path_buf, "wamr-frame.mod0.func{d}.json", .{local_func}) catch return fail("frame metadata identity is incompatible", .{});
    if (!std.mem.eql(u8, jsonStringOrEmpty(metadata.get("schema")), "wamr-aot-frame-attribution") or
        (jsonInt(metadata.get("schema_version")) orelse -1) != 2 or
        !std.mem.eql(u8, jsonStringOrEmpty(metadata.get("architecture")), "aarch64") or
        !std.mem.eql(u8, jsonStringOrEmpty(metadata.get("abi")), "aapcs64") or
        !std.mem.eql(u8, jsonStringOrEmpty(metadata.get("path")), exp_path))
    {
        return fail("frame metadata identity is incompatible", .{});
    }
    _ = try requireSha(metadata.get("sha256"), "frame metadata sha256", 64);
    _ = try requireSha(metadata.get("module_text_sha256"), "frame metadata module_text_sha256", 64);
    _ = try requireSha(metadata.get("normalized_code_sha256"), "frame metadata normalized_code_sha256", 64);

    const code_size = try frameCount(frame.get("code_size"), "code size");
    if (code_size == 0 or @mod(code_size, 4) != 0) {
        return fail("frame code size is not AArch64 instruction-aligned", .{});
    }

    const captures_val = frame.get("captures") orelse return fail("frame provenance must contain both WAMR captures", .{});
    if (captures_val != .array or captures_val.array.items.len != PROFILE_CAPTURES_PER_ENGINE) {
        return fail("frame provenance must contain both WAMR captures", .{});
    }
    const wamr_captures_val = report.get("wamr_captures") orelse return fail("frame captures do not match the measured schedule", .{});
    if (wamr_captures_val != .array or wamr_captures_val.array.items.len != PROFILE_CAPTURES_PER_ENGINE) {
        return fail("frame captures do not match the measured schedule", .{});
    }

    var function_samples: i64 = 0;
    for (captures_val.array.items, wamr_captures_val.array.items, 1..) |cap_v, meas_v, ordinal| {
        if (cap_v != .object or meas_v != .object) return fail("frame captures do not match the measured schedule", .{});
        const cap = cap_v.object;
        const meas = meas_v.object;
        if ((jsonInt(cap.get("ordinal")) orelse -1) != @as(i64, @intCast(ordinal)) or
            (jsonInt(cap.get("total_samples")) orelse -1) != (jsonInt(meas.get("total_samples")) orelse -2))
        {
            return fail("frame captures do not match the measured schedule", .{});
        }
        var samples: i64 = 0;
        var previous: i64 = -1;
        const sbo_val = cap.get("samples_by_offset") orelse return fail("frame instruction samples have invalid offsets/counts", .{});
        if (sbo_val != .array) return fail("frame instruction samples have invalid offsets/counts", .{});
        for (sbo_val.array.items) |pair_v| {
            if (pair_v != .array or pair_v.array.items.len != 2) {
                return fail("frame instruction samples have invalid offsets/counts", .{});
            }
            const offset = try frameCount(pair_v.array.items[0], "instruction offset");
            const count = try frameCount(pair_v.array.items[1], "instruction samples");
            if (offset <= previous or offset >= code_size or @mod(offset, 4) != 0 or count == 0) {
                return fail("frame instruction samples have invalid offsets/counts", .{});
            }
            previous = offset;
            samples += count;
        }
        const cap_fn_samples = jsonInt(cap.get("function_samples")) orelse -1;
        const meas_attr = jsonInt(meas.get("attributed_samples")) orelse 0;
        if (samples != cap_fn_samples or samples <= 0 or samples > meas_attr) {
            return fail("frame function samples exceed their capture budget", .{});
        }
        function_samples += samples;
    }

    var expected: i64 = 0;
    if (wamr_eng.get("top_functions")) |tf_v| {
        if (tf_v == .array) {
            for (tf_v.array.items) |item| {
                if (item == .object and (jsonInt(item.object.get("local_func")) orelse -1) == local_func) {
                    expected += jsonInt(item.object.get("samples")) orelse 0;
                }
            }
        }
    }
    if ((jsonInt(frame.get("function_samples")) orelse -1) != function_samples or expected != function_samples) {
        return fail("frame function samples do not reconcile to the ranking", .{});
    }

    const summary = (frame.get("summary") orelse return fail("frame coverage does not conserve counts", .{})).object;
    const coverage = (summary.get("coverage") orelse return fail("frame coverage does not conserve counts", .{})).object;

    const whole_inst = try frameCount(coverage.get("frame_instructions"), "instructions");
    const mapped_inst = try frameCount(coverage.get("attributed_frame_instructions"), "mapped instructions");
    const proven_inst = try frameCount(coverage.get("proven_origin_frame_instructions"), "proven instructions");
    const unknown_inst = try frameCount(coverage.get("unknown_frame_instructions"), "unknown instructions");
    if (!(proven_inst <= mapped_inst and mapped_inst <= whole_inst) or unknown_inst != whole_inst - proven_inst) {
        return fail("frame coverage does not conserve counts", .{});
    }

    const whole_samp = try frameCount(coverage.get("frame_samples"), "samples");
    const mapped_samp = try frameCount(coverage.get("attributed_frame_samples"), "mapped samples");
    const proven_samp = try frameCount(coverage.get("proven_origin_frame_samples"), "proven samples");
    const unknown_samp = try frameCount(coverage.get("unknown_frame_samples"), "unknown samples");
    if (!(proven_samp <= mapped_samp and mapped_samp <= whole_samp) or unknown_samp != whole_samp - proven_samp) {
        return fail("frame coverage does not conserve counts", .{});
    }

    if (whole_samp > function_samples or whole_inst > @divTrunc(code_size, 4)) {
        return fail("frame coverage exceeds the selected function", .{});
    }

    const origins = (summary.get("origins") orelse return fail("frame report has an unsupported origin", .{})).object;
    var sum_orig_samples: i64 = 0;
    var sum_proven_orig_samples: i64 = 0;
    var sum_orig_insts: i64 = 0;
    var sum_proven_orig_insts: i64 = 0;
    var alloc_spill_samples: i64 = 0;

    var oit = origins.iterator();
    while (oit.next()) |kv| {
        const origin = kv.key_ptr.*;
        var supported = false;
        for (aot_jit_attr.FRAME_ORIGINS) |allowed| {
            if (std.mem.eql(u8, origin, allowed)) {
                supported = true;
                break;
            }
        }
        if (!supported) {
            return fail("frame report has an unsupported origin", .{});
        }
        if (kv.value_ptr.* != .object) return fail("frame origins do not conserve coverage counts", .{});
        const vobj = kv.value_ptr.object;
        const osamp = try frameCount(vobj.get("samples"), "origin samples");
        const oinst = try frameCount(vobj.get("static_instructions"), "origin static_instructions");
        sum_orig_samples += osamp;
        sum_orig_insts += oinst;
        if (!std.mem.eql(u8, origin, "unknown")) {
            sum_proven_orig_samples += osamp;
            sum_proven_orig_insts += oinst;
        }
        if (std.mem.eql(u8, origin, "allocator_spill")) {
            alloc_spill_samples = osamp;
        }
        const pct_run = jsonNumber(vobj.get("percent_of_run")) orelse -1.0;
        const expected_pct = 100.0 * @as(f64, @floatFromInt(osamp)) / @as(f64, @floatFromInt(total));
        if (pct_run != expected_pct) {
            return fail("frame origin percentage disagrees with its samples", .{});
        }
    }

    if (sum_orig_samples != whole_samp or
        sum_proven_orig_samples != proven_samp or
        sum_orig_insts != whole_inst or
        sum_proven_orig_insts != proven_inst)
    {
        return fail("frame origins do not conserve coverage counts", .{});
    }

    const contributors = (summary.get("allocator_contributors") orelse return fail("allocator contributor samples are duplicated or missing", .{})).array;
    var contributor_samples: i64 = 0;
    var ranked_loads: i64 = 0;
    var ranked_stores: i64 = 0;
    for (contributors.items) |item_v| {
        if (item_v != .object) return fail("allocator contributor samples are duplicated or missing", .{});
        const item = item_v.object;
        const csamp = try frameCount(item.get("samples"), "allocator contributor");
        contributor_samples += csamp;
        const pct_run = jsonNumber(item.get("percent_of_run")) orelse -1.0;
        const expected_pct = 100.0 * @as(f64, @floatFromInt(csamp)) / @as(f64, @floatFromInt(total));
        if (pct_run != expected_pct) {
            return fail("allocator percentage disagrees with its samples", .{});
        }
        ranked_loads += try frameCount(item.get("static_loads"), "component loads");
        ranked_stores += try frameCount(item.get("static_stores"), "component stores");
    }
    if (contributor_samples != alloc_spill_samples) {
        return fail("allocator contributor samples are duplicated or missing", .{});
    }

    const reconciliation = (summary.get("reconciliation") orelse return fail("frame emitted counts did not reconcile", .{})).object;
    if (!(jsonBool(reconciliation.get("matches")) orelse false)) {
        return fail("frame emitted counts did not reconcile", .{});
    }
    const comp_counts = (summary.get("allocator_component_counts") orelse return fail("frame static component counts were duplicated or lost", .{})).object;

    const emitted_loads = try frameCount(reconciliation.get("emitted_allocator_loads"), "emitted loads");
    const metric_loads = try frameCount(reconciliation.get("spill_metric_loads"), "spill metric loads");
    const total_comp_loads = try frameCount(comp_counts.get("total_loads"), "total component loads");
    const unranked_loads = try frameCount(comp_counts.get("unranked_loads"), "unranked component loads");
    if (!(emitted_loads == metric_loads and metric_loads == total_comp_loads and total_comp_loads == ranked_loads + unranked_loads)) {
        return fail("frame static component counts were duplicated or lost", .{});
    }

    const emitted_stores = try frameCount(reconciliation.get("emitted_allocator_stores"), "emitted stores");
    const metric_stores = try frameCount(reconciliation.get("spill_metric_stores"), "spill metric stores");
    const total_comp_stores = try frameCount(comp_counts.get("total_stores"), "total component stores");
    const unranked_stores = try frameCount(comp_counts.get("unranked_stores"), "unranked component stores");
    if (!(emitted_stores == metric_stores and metric_stores == total_comp_stores and total_comp_stores == ranked_stores + unranked_stores)) {
        return fail("frame static component counts were duplicated or lost", .{});
    }
}

pub fn validateReport(allocator: Allocator, report_val: Value) ProfileError!void {
    if (report_val != .object) return fail("profile report root must be a JSON object", .{});
    const report = report_val.object;
    if ((jsonInt(report.get("schema_version")) orelse -1) != REPORT_SCHEMA_VERSION) {
        return fail("profile report has an unsupported schema_version", .{});
    }
    if (!std.mem.eql(u8, jsonStringOrEmpty(report.get("kind")), REPORT_KIND)) {
        return fail("profile report has an unsupported kind", .{});
    }
    if (!std.mem.eql(u8, jsonStringOrEmpty(report.get("architecture")), "aarch64")) {
        return fail("profile report architecture must be aarch64", .{});
    }
    const bench_val = report.get("benchmark") orelse return fail("profile report lacks benchmark provenance", .{});
    if (bench_val != .object) return fail("profile report lacks benchmark provenance", .{});
    const benchmark = bench_val.object;
    if (jsonStringOrEmpty(benchmark.get("report_id")).len == 0 or jsonStringOrEmpty(benchmark.get("generated_at")).len == 0) {
        return fail("profile report has incomplete benchmark linkage", .{});
    }
    _ = try requireSha(benchmark.get("report_sha256"), "benchmark report sha256", 64);
    const handoff_val = benchmark.get("artifact_handoff") orelse return fail("profile report lacks benchmark artifact handoff", .{});
    if (handoff_val != .object or jsonStringOrEmpty(handoff_val.object.get("directory")).len == 0) {
        return fail("profile report lacks benchmark artifact handoff", .{});
    }
    _ = try requireSha(handoff_val.object.get("manifest_sha256"), "benchmark handoff manifest sha256", 64);

    const selected_role = jsonString(benchmark.get("selected_role")) orelse "wamr-target";
    var selected_opt = benchmark.get("selected_wamr");
    const legacy_target = (selected_opt == null or selected_opt.? == .null) and std.mem.eql(u8, selected_role, "wamr-target");
    if (legacy_target) {
        selected_opt = benchmark.get("target");
    }
    if ((!std.mem.eql(u8, selected_role, "wamr-baseline") and !std.mem.eql(u8, selected_role, "wamr-target")) or
        selected_opt == null or selected_opt.? != .object)
    {
        return fail("profile report lacks its selected benchmark WAMR identity", .{});
    }
    const selected = selected_opt.?.object;
    const role_in_selected = jsonString(selected.get("role")) orelse (if (legacy_target) selected_role else "");
    if (!std.mem.eql(u8, role_in_selected, selected_role)) {
        return fail("profile report lacks its selected benchmark WAMR identity", .{});
    }
    if (benchmark.get("target")) |target_val| {
        if (!std.mem.eql(u8, selected_role, "wamr-target") or !jsonValueEql(target_val, selected_opt.?)) {
            return fail("profile benchmark target alias contradicts its selected role", .{});
        }
    }

    const wamr_val = report.get("wamr") orelse return fail("profile and benchmark WAMR roles differ", .{});
    if (wamr_val != .object) return fail("profile and benchmark WAMR roles differ", .{});
    const wamr = wamr_val.object;
    const wamr_role = jsonString(wamr.get("benchmark_role")) orelse "wamr-target";
    if (!std.mem.eql(u8, wamr_role, selected_role)) {
        return fail("profile and benchmark WAMR roles differ", .{});
    }

    const selected_identity = if (selected.get("identity")) |idv| (if (idv == .object) idv.object else return fail("profile and benchmark WAMR source SHAs differ", .{})) else return fail("profile and benchmark WAMR source SHAs differ", .{});
    const selected_source = if (selected_identity.get("source")) |sv| (if (sv == .object) sv.object else return fail("profile and benchmark WAMR source SHAs differ", .{})) else return fail("profile and benchmark WAMR source SHAs differ", .{});
    if (!std.mem.eql(u8, jsonStringOrEmpty(selected_source.get("sha")), jsonStringOrEmpty(wamr.get("commit")))) {
        return fail("profile and benchmark WAMR source SHAs differ", .{});
    }
    const sel_rt = if (selected_identity.get("runtime")) |v| (if (v == .object) jsonStringOrEmpty(v.object.get("sha256")) else "") else "";
    const sel_comp = if (selected_identity.get("compiler")) |v| (if (v == .object) jsonStringOrEmpty(v.object.get("sha256")) else "") else "";
    const sel_mod = if (selected_identity.get("module")) |v| (if (v == .object) jsonStringOrEmpty(v.object.get("sha256")) else "") else "";
    if (!std.mem.eql(u8, sel_rt, jsonStringOrEmpty(wamr.get("runtime_sha256"))) or
        !std.mem.eql(u8, sel_comp, jsonStringOrEmpty(wamr.get("compiler_sha256"))) or
        !std.mem.eql(u8, sel_mod, jsonStringOrEmpty(wamr.get("cwasm_sha256"))))
    {
        return fail("profile and benchmark WAMR tool identities differ", .{});
    }

    const baseline_val = benchmark.get("wasmtime_baseline") orelse return fail("profile report lacks benchmark Wasmtime identity", .{});
    if (baseline_val != .object) return fail("profile report lacks benchmark Wasmtime identity", .{});
    const baseline_id = if (baseline_val.object.get("identity")) |v| (if (v == .object) v.object else return fail("profile and benchmark Wasmtime identities differ", .{})) else return fail("profile and benchmark Wasmtime identities differ", .{});
    const wt_report_obj = if (report.get("wasmtime")) |v| (if (v == .object) v.object else return fail("profile and benchmark Wasmtime identities differ", .{})) else return fail("profile and benchmark Wasmtime identities differ", .{});
    const base_rt_sha = if (baseline_id.get("runtime")) |v| (if (v == .object) jsonStringOrEmpty(v.object.get("sha256")) else "") else "";
    if (!std.mem.eql(u8, jsonStringOrEmpty(baseline_id.get("version")), jsonStringOrEmpty(wt_report_obj.get("version"))) or
        !std.mem.eql(u8, base_rt_sha, jsonStringOrEmpty(wt_report_obj.get("sha256"))))
    {
        return fail("profile and benchmark Wasmtime identities differ", .{});
    }

    const prov_val = report.get("provenance") orelse return fail("profile report lacks producer provenance", .{});
    if (prov_val != .object) return fail("profile report lacks producer provenance", .{});
    const prov_obj = prov_val.object;
    const prod_source_sha = try requireSha(prov_obj.get("producer_source_sha"), "profile producer source SHA", 40);
    const script_sha = try requireSha(prov_obj.get("script_sha256"), "profile producer script sha256", 64);
    const bench_prod_sha = if (benchmark.get("producer")) |v| (if (v == .object) jsonStringOrEmpty(v.object.get("source_sha")) else "") else "";
    if (!std.mem.eql(u8, prod_source_sha, bench_prod_sha)) {
        return fail("profile and benchmark tooling source SHAs differ", .{});
    }
    const prov_exec = prov_obj.get("execution") orelse .null;
    const bench_exec = benchmark.get("execution") orelse .null;
    if (!jsonValueEql(prov_exec, bench_exec)) {
        return fail("profile and benchmark execution provenance differ", .{});
    }

    const ga_val = report.get("guest_args") orelse return fail("profile report guest args are not authoritative", .{});
    if (ga_val != .array or ga_val.array.items.len != bench_coremark.coremark_guest_args_default.len) {
        return fail("profile report guest args are not authoritative", .{});
    }
    for (ga_val.array.items, bench_coremark.coremark_guest_args_default) |actual_arg, exp_arg| {
        if (!std.mem.eql(u8, jsonStringOrEmpty(actual_arg), exp_arg)) {
            return fail("profile report guest args are not authoritative", .{});
        }
    }
    if ((jsonInt(report.get("expected_iterations")) orelse -1) != bench_coremark.expected_iterations) {
        return fail("profile report iteration count is not authoritative", .{});
    }

    const cw_val = report.get("classifier_wording") orelse return fail("profile report has ambiguous ALU classifier wording", .{});
    if (cw_val != .object or !std.mem.eql(u8, jsonStringOrEmpty(cw_val.object.get("all_alu")), ALL_ALU_WORDING)) {
        return fail("profile report has ambiguous ALU classifier wording", .{});
    }

    const narrow_summary_opt = report.get("alu_provenance");
    const has_narrow = narrow_summary_opt != null and narrow_summary_opt.? != .null;
    if (has_narrow) {
        const narrow_summary = if (narrow_summary_opt.? == .object) narrow_summary_opt.?.object else return fail("profile report has invalid narrow ALU summary", .{});
        if (!std.mem.eql(u8, jsonStringOrEmpty(cw_val.object.get("narrow_alu_provenance")), NARROW_ALU_WORDING)) {
            return fail("profile report has ambiguous narrow ALU classifier wording", .{});
        }
        const am_val = prov_obj.get("analysis_module") orelse return fail("profile report lacks ALU analysis module identity", .{});
        if (am_val != .object) return fail("profile report lacks ALU analysis module identity", .{});
        const am_sha = try requireSha(am_val.object.get("sha256"), "ALU analysis module sha256", 64);

        const sources_val = prov_obj.get("analysis_sources") orelse return fail("profile report lacks complete committed analysis sources", .{});
        if (sources_val != .object) return fail("profile report lacks complete committed analysis sources", .{});
        const sources = sources_val.object;
        const files_val = sources.get("files") orelse return fail("profile report lacks complete committed analysis sources", .{});
        if (!std.mem.eql(u8, jsonStringOrEmpty(sources.get("source_mode")), "commit-verified") or
            !std.mem.eql(u8, jsonStringOrEmpty(sources.get("commit")), prod_source_sha) or
            files_val != .object)
        {
            return fail("profile report lacks complete committed analysis sources", .{});
        }
        const files_obj = files_val.object;

        const script_path = jsonStringOrEmpty(prov_obj.get("script_path"));
        for (&REQUIRED_ANALYSIS_SOURCES) |req_path| {
            if (!files_obj.contains(req_path)) {
                return fail("profile report lacks complete committed analysis sources", .{});
            }
        }

        var fit = files_obj.iterator();
        while (fit.next()) |kv| {
            const p = kv.key_ptr.*;
            if (p.len == 0 or p[0] == '/' or std.mem.indexOf(u8, p, "..") != null) {
                return fail("profile analysis source path is invalid", .{});
            }
            _ = try requireSha(kv.value_ptr.*, "analysis source sha256", 64);
        }

        const expected_script_path = "scripts/profile_coremark_aarch64.zig";
        const expected_am_path = "scripts/aarch64_instruction_provenance.zig";
        const expected_bench_path = "scripts/bench_coremark.zig";

        const bench_script_sha = if (benchmark.get("producer")) |bp|
            if (bp == .object and bp.object.get("script") != null and bp.object.get("script").? == .object)
                jsonStringOrEmpty(bp.object.get("script").?.object.get("sha256"))
            else
                ""
        else
            "";

        if (!std.mem.eql(u8, script_path, expected_script_path) or
            !std.mem.eql(u8, jsonStringOrEmpty(files_obj.get(expected_script_path)), script_sha) or
            !std.mem.eql(u8, jsonStringOrEmpty(am_val.object.get("path")), expected_am_path) or
            !std.mem.eql(u8, jsonStringOrEmpty(files_obj.get(expected_am_path)), am_sha) or
            !std.mem.eql(u8, jsonStringOrEmpty(files_obj.get(expected_bench_path)), bench_script_sha))
        {
            return fail("profile analysis source identities disagree", .{});
        }

        if ((jsonInt(narrow_summary.get("schema_version")) orelse -1) != provenance.SCHEMA_VERSION or
            !std.mem.eql(u8, jsonStringOrEmpty(narrow_summary.get("kind")), provenance.ANALYSIS_KIND))
        {
            return fail("profile report has invalid narrow ALU summary", .{});
        }
    }

    const aff_val = report.get("affinity") orelse return fail("profile report lacks verified CPU affinity", .{});
    if (aff_val != .object or !(jsonBool(aff_val.object.get("verified")) orelse false)) {
        return fail("profile report lacks verified CPU affinity", .{});
    }

    const sched_val = report.get("profile_schedule") orelse return fail("profile report must contain four warmups and four captures", .{});
    if (sched_val != .array or sched_val.array.items.len != 8) {
        return fail("profile report must contain four warmups and four captures", .{});
    }
    const expected_engines = [_][]const u8{ "wamr", "wasmtime", "wasmtime", "wamr", "wamr", "wasmtime", "wasmtime", "wamr" };
    const expected_phases = [_][]const u8{ "warmup", "warmup", "warmup", "warmup", "profile", "profile", "profile", "profile" };
    for (sched_val.array.items, 0..) |item_v, idx| {
        if (item_v != .object) return fail("profile report execution order is not ABBA/ABBA", .{});
        if (!std.mem.eql(u8, jsonStringOrEmpty(item_v.object.get("engine")), expected_engines[idx])) {
            return fail("profile report execution order is not ABBA/ABBA", .{});
        }
        if (!std.mem.eql(u8, jsonStringOrEmpty(item_v.object.get("phase")), expected_phases[idx])) {
            return fail("profile report phases are not balanced", .{});
        }
    }

    const min_cov = jsonNumber(report.get("minimum_attribution_coverage_pct")) orelse -1.0;
    if (min_cov != MIN_ATTRIBUTION_COVERAGE_PCT) {
        return fail("profile report has the wrong coverage threshold", .{});
    }

    const wamr_caps = report.get("wamr_captures") orelse return fail("profile report must validate two captures per engine", .{});
    const wt_caps = report.get("wasmtime_captures") orelse return fail("profile report must validate two captures per engine", .{});
    if (wamr_caps != .array or wamr_caps.array.items.len != PROFILE_CAPTURES_PER_ENGINE or
        wt_caps != .array or wt_caps.array.items.len != PROFILE_CAPTURES_PER_ENGINE)
    {
        return fail("profile report must validate two captures per engine", .{});
    }
    for (wamr_caps.array.items) |cap_v| {
        if (cap_v != .object) return fail("WAMR capture failed authoritative attribution", .{});
        const cov = jsonNumber(cap_v.object.get("coverage_pct")) orelse 0.0;
        const auth = if (cap_v.object.get("mapping")) |mv|
            if (mv == .object) (jsonBool(mv.object.get("authoritative")) orelse false) else false
        else
            false;
        if (cov < MIN_ATTRIBUTION_COVERAGE_PCT or !auth) {
            return fail("WAMR capture failed authoritative attribution", .{});
        }
    }
    for (wt_caps.array.items) |cap_v| {
        if (cap_v != .object) return fail("Wasmtime capture failed attribution coverage", .{});
        const cov = jsonNumber(cap_v.object.get("coverage_pct")) orelse 0.0;
        if (cov < MIN_ATTRIBUTION_COVERAGE_PCT) {
            return fail("Wasmtime capture failed attribution coverage", .{});
        }
    }

    const engines_val = report.get("engines") orelse return fail("profile report must contain WAMR and Wasmtime engines", .{});
    if (engines_val != .object or engines_val.object.count() != 2 or
        !engines_val.object.contains("wamr") or !engines_val.object.contains("wasmtime"))
    {
        return fail("profile report must contain WAMR and Wasmtime engines", .{});
    }
    for ([_][]const u8{ "wamr", "wasmtime" }) |ename| {
        const eng_v = engines_val.object.get(ename).?;
        if (eng_v != .object) return fail("invalid {s} sample totals", .{ename});
        const tot = jsonInt(eng_v.object.get("total_samples")) orelse -1;
        const attr = jsonInt(eng_v.object.get("attributed_samples")) orelse -1;
        if (tot <= 0 or attr <= 0 or attr > tot) {
            return fail("invalid {s} sample totals", .{ename});
        }
    }

    const wamr_eng = engines_val.object.get("wamr").?.object;
    const wt_eng = engines_val.object.get("wasmtime").?.object;
    const matched_val = report.get("matched_functions") orelse return fail("profile report has no matched functions", .{});
    if (matched_val != .array or matched_val.array.items.len == 0) {
        return fail("profile report has no matched functions", .{});
    }
    const imported_fn_cnt = if (report.get("wasm")) |wv|
        if (wv == .object) (jsonInt(wv.object.get("imported_function_count")) orelse 0) else 0
    else
        0;

    for (matched_val.array.items) |item_v| {
        if (item_v != .object) return fail("profile report has no matched functions", .{});
        const item = item_v.object;
        const lf = jsonInt(item.get("local_func")) orelse -1;
        const wf = jsonInt(item.get("wasm_function_index")) orelse -1;
        if (wf != lf + imported_fn_cnt) {
            return fail("local_func/wasm function mapping is inconsistent", .{});
        }
        const wamr_fn_s = if (item.get("wamr")) |v| (if (v == .object) (jsonInt(v.object.get("samples")) orelse 0) else 0) else 0;
        const wt_fn_s = if (item.get("wasmtime")) |v| (if (v == .object) (jsonInt(v.object.get("samples")) orelse 0) else 0) else 0;
        if (wamr_fn_s <= 0 or wt_fn_s <= 0) {
            return fail("matched functions must have samples in both engines", .{});
        }
        if (has_narrow) {
            const narrow_v = item.get("alu_provenance") orelse return fail("matched function lacks narrow ALU provenance", .{});
            if (narrow_v != .object or !std.mem.eql(u8, jsonStringOrEmpty(narrow_v.object.get("wording")), NARROW_ALU_WORDING)) {
                return fail("matched function lacks narrow ALU provenance", .{});
            }
            const narrow = narrow_v.object;
            const all_alu_group = if (item.get("class_groups")) |cg|
                if (cg == .object and cg.object.get("all_alu") != null and cg.object.get("all_alu").? == .object)
                    cg.object.get("all_alu").?.object
                else
                    return fail("matched function lacks all_alu class group", .{})
            else
                return fail("matched function lacks all_alu class group", .{});

            try validateNarrowAluAnalysis(
                allocator,
                narrow.get("wamr") orelse .null,
                "WAMR",
                jsonInt(all_alu_group.get("wamr_samples")) orelse -1,
                wamr_fn_s,
                jsonInt(wamr_eng.get("total_samples")).?,
                jsonInt(wamr_eng.get("attributed_samples")).?,
            );
            try validateNarrowAluAnalysis(
                allocator,
                narrow.get("wasmtime") orelse .null,
                "Wasmtime",
                jsonInt(all_alu_group.get("wasmtime_samples")) orelse -1,
                wt_fn_s,
                jsonInt(wt_eng.get("total_samples")).?,
                jsonInt(wt_eng.get("attributed_samples")).?,
            );

            const gate_val = narrow.get("gate") orelse return fail("narrow ALU gate does not match its validated sample evidence", .{});
            if (gate_val != .object) return fail("narrow ALU gate does not match its validated sample evidence", .{});
            if (gate_val.object.get("categories")) |cats_v| {
                if (cats_v == .object and cats_v.object.contains("all_alu")) {
                    return fail("narrow ALU gate does not match its validated sample evidence", .{});
                }
            }
            var diag = provenance.Diagnostic{};
            const recomputed_gate = provenance.compareEngineAnalyses(
                allocator,
                narrow.get("wamr").?,
                narrow.get("wasmtime").?,
                5.0,
                &diag,
            ) catch {
                return fail("narrow ALU gate does not match its validated sample evidence", .{});
            };
            if (!jsonValueEql(gate_val, recomputed_gate)) {
                return fail("narrow ALU gate does not match its validated sample evidence", .{});
            }
        }
    }

    if (has_narrow) {
        const narrow_summary = narrow_summary_opt.?.object;
        const retained_opt = if (report.get("retained_analysis_artifacts")) |ra|
            if (ra == .object) ra.object.get("wamr_cwasm") else null
        else
            null;
        if (retained_opt == null or retained_opt.? != .object or
            !(jsonBool(retained_opt.?.object.get("retained")) orelse false) or
            !std.mem.eql(u8, jsonStringOrEmpty(retained_opt.?.object.get("source_sha256")), jsonStringOrEmpty(wamr.get("cwasm_sha256"))))
        {
            return fail("profile report did not retain the exact analyzed WAMR cwasm", .{});
        }
        const target_lf = jsonInt(narrow_summary.get("target_local_func")) orelse -1;
        const target_wf = jsonInt(narrow_summary.get("target_wasm_function_index")) orelse -1;
        var matched_target: ?Value = null;
        for (matched_val.array.items) |item_v| {
            const item = item_v.object;
            if ((jsonInt(item.get("local_func")) orelse -2) == target_lf and
                (jsonInt(item.get("wasm_function_index")) orelse -2) == target_wf)
            {
                matched_target = item_v;
                break;
            }
        }
        if (matched_target) |mt| {
            const top_gate = narrow_summary.get("gate") orelse .null;
            const fn_gate = mt.object.get("alu_provenance").?.object.get("gate") orelse .null;
            if (!jsonValueEql(top_gate, fn_gate)) {
                return fail("top-level narrow ALU gate does not match its target function", .{});
            }
        } else {
            if (narrow_summary.get("gate")) |g| {
                if (g == .object and (jsonBool(g.object.get("optimization_authorized")) orelse false)) {
                    return fail("unmatched target function cannot authorize optimization", .{});
                }
            }
        }
    }

    if (report.get("frame_attribution")) |fa| {
        if (fa != .null) {
            if (!has_narrow) {
                return fail("frame profiles require complete committed analysis sources", .{});
            }
            try validateFrameProvenance(fa, report_val);
        }
    }
}

pub const RunCommandResult = struct {
    stdout: []const u8 = "",
    stderr: []const u8 = "",
};

pub const CommandRunner = struct {
    ctx: *anyopaque,
    runFn: *const fn (
        ctx: *anyopaque,
        allocator: Allocator,
        command: []const []const u8,
        log_name: []const u8,
        cwd: []const u8,
        env_map: *const std.StringHashMap([]const u8),
    ) ProfileError!RunCommandResult,
};

fn fileExists(io: std.Io, path: []const u8) bool {
    _ = std.Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return true;
}

pub const DiagnosticsCompileResult = struct {
    metrics: std.AutoHashMap(usize, Value),
    metadata_path: ?[]const u8,
};

pub fn compileWamrDiagnostics(
    allocator: Allocator,
    io: std.Io,
    runner: CommandRunner,
    wamrc: []const u8,
    fixture: []const u8,
    cwasm: []const u8,
    out_dir: []const u8,
    build_repo: []const u8,
    helper: []const u8,
    frame_func: ?usize,
    base_env: *const std.StringHashMap([]const u8),
) ProfileError!DiagnosticsCompileResult {
    const metadata_path: ?[]const u8 = if (frame_func) |ff|
        try std.fmt.allocPrint(allocator, "{s}/wamr-frame.mod0.func{d}.json", .{ out_dir, ff })
    else
        null;

    if (metadata_path) |mp| {
        if (fileExists(io, mp)) {
            return fail("refusing to reuse existing frame metadata: {s}", .{mp});
        }
    }

    const diag_cwasm = try std.fmt.allocPrint(allocator, "{s}/coremark.diagnostic.cwasm", .{out_dir});
    defer std.Io.Dir.cwd().deleteFile(io, diag_cwasm) catch {};

    var env: std.StringHashMap([]const u8) = .init(allocator);
    var eit = base_env.iterator();
    while (eit.next()) |kv| {
        if (std.mem.startsWith(u8, kv.key_ptr.*, "WAMR_AOT_FRAME_ATTRIBUTION")) continue;
        try env.put(kv.key_ptr.*, kv.value_ptr.*);
    }
    try env.put("WAMR_AOT_SPILL_METRIC", "1");
    try env.put("WAMR_AOT_SPILL_METRIC_MIN_SPILLS", "1");
    try env.put("WAMR_AOT_CODEGEN_TIMING", "1");
    try env.put("WAMR_AOT_CODEGEN_TIMING_THRESHOLD_MS", "0");
    if (frame_func) |ff| {
        try env.put("WAMR_AOT_FRAME_ATTRIBUTION", try std.fmt.allocPrint(allocator, "{s}/wamr-frame", .{out_dir}));
        try env.put("WAMR_AOT_FRAME_ATTRIBUTION_MODULE", "0");
        try env.put("WAMR_AOT_FRAME_ATTRIBUTION_FUNC", try std.fmt.allocPrint(allocator, "{d}", .{ff}));
    }

    const compile_cmd = [_][]const u8{
        wamrc,
        "compile",
        fixture,
        "-o",
        diag_cwasm,
    };
    const res = try runner.runFn(runner.ctx, allocator, &compile_cmd, "wamr-compile.log", build_repo, &env);

    const expected_bytes = std.Io.Dir.cwd().readFileAlloc(io, cwasm, allocator, .limited(256 * 1024 * 1024)) catch {
        return fail("cannot read benchmark cwasm: {s}", .{cwasm});
    };
    const diag_bytes = std.Io.Dir.cwd().readFileAlloc(io, diag_cwasm, allocator, .limited(256 * 1024 * 1024)) catch {
        return fail("diagnostic compile did not produce {s}", .{diag_cwasm});
    };
    if (!std.mem.eql(u8, expected_bytes, diag_bytes)) {
        return fail("diagnostic cwasm does not match the exact benchmark-built module", .{});
    }
    std.Io.Dir.cwd().deleteFile(io, diag_cwasm) catch {};

    if (metadata_path) |mp| {
        if (!fileExists(io, mp)) {
            return fail(
                "the retained benchmark compiler did not emit requested frame metadata; benchmark a candidate with AArch64 frame attribution",
                .{},
            );
        }
        const val_cmd = [_][]const u8{
            "zig-aot-jit-attr",
            helper,
            "--cwasm",
            cwasm,
            "--arch",
            "aarch64",
            "--func",
            try std.fmt.allocPrint(allocator, "{d}", .{frame_func.?}),
            "--frame-metadata",
            mp,
            "--validate-frame-metadata",
        };
        _ = try runner.runFn(runner.ctx, allocator, &val_cmd, "wamr-frame-validation.log", build_repo, &env);
    }

    const combined_out = try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ res.stdout, res.stderr });
    const metrics = try parseSpillMetrics(allocator, combined_out);
    return .{
        .metrics = metrics,
        .metadata_path = metadata_path,
    };
}

pub fn gzipIfSmall(
    allocator: Allocator,
    io: std.Io,
    path: []const u8,
    max_bytes: usize,
) ProfileError!Value {
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch {
        return fail("cannot stat perf artifact: {s}", .{path});
    };
    const size: usize = @intCast(st.size);
    const base_name = std.fs.path.basename(path);
    if (size > max_bytes) {
        var obj: json.ObjectMap = .empty;
        try obj.put(allocator, "path", .{ .string = try allocator.dupe(u8, base_name) });
        try obj.put(allocator, "size_bytes", .{ .integer = @intCast(size) });
        try obj.put(allocator, "retained", .{ .bool = false });
        return .{ .object = obj };
    }
    const target = try std.fmt.allocPrint(allocator, "{s}.gz", .{path});
    const gz_res = std.process.run(allocator, io, .{
        .argv = &.{ "gzip", "-9", "-c", path },
        .stdout_limit = .limited(256 * 1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch return fail("gzip failed for {s}", .{path});
    switch (gz_res.term) {
        .exited => |c| if (c != 0) return fail("gzip failed for {s}: {s}", .{ path, gz_res.stderr }),
        else => return fail("gzip failed for {s}", .{path}),
    }
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = target, .data = gz_res.stdout }) catch {
        return fail("cannot write compressed artifact: {s}", .{target});
    };
    var obj: json.ObjectMap = .empty;
    try obj.put(allocator, "path", .{ .string = try std.fmt.allocPrint(allocator, "{s}.gz", .{base_name}) });
    try obj.put(allocator, "source_size_bytes", .{ .integer = @intCast(size) });
    try obj.put(allocator, "size_bytes", .{ .integer = @intCast(gz_res.stdout.len) });
    try obj.put(allocator, "retained", .{ .bool = true });
    return .{ .object = obj };
}

pub fn retainExactBinary(
    allocator: Allocator,
    io: std.Io,
    source: []const u8,
    target: []const u8,
    max_bytes: usize,
) ProfileError!Value {
    const st = std.Io.Dir.cwd().statFile(io, source, .{}) catch {
        return fail("cannot stat binary: {s}", .{source});
    };
    const size: usize = @intCast(st.size);
    const source_sha = bench_coremark.sha256File(allocator, io, source) catch {
        return fail("cannot hash binary: {s}", .{source});
    };
    const target_name = std.fs.path.basename(target);
    if (size > max_bytes) {
        var obj: json.ObjectMap = .empty;
        try obj.put(allocator, "path", .{ .string = try allocator.dupe(u8, target_name) });
        try obj.put(allocator, "source_size_bytes", .{ .integer = @intCast(size) });
        try obj.put(allocator, "source_sha256", .{ .string = source_sha });
        try obj.put(allocator, "retained", .{ .bool = false });
        try obj.put(allocator, "reason", .{ .string = try std.fmt.allocPrint(allocator, "source exceeds retention limit {d}", .{max_bytes}) });
        return .{ .object = obj };
    }
    const gz_res = std.process.run(allocator, io, .{
        .argv = &.{ "gzip", "-9", "-c", source },
        .stdout_limit = .limited(256 * 1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch return fail("gzip failed for {s}", .{source});
    switch (gz_res.term) {
        .exited => |c| if (c != 0) return fail("gzip failed for {s}: {s}", .{ source, gz_res.stderr }),
        else => return fail("gzip failed for {s}", .{source}),
    }
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = target, .data = gz_res.stdout }) catch {
        return fail("cannot write compressed binary: {s}", .{target});
    };
    var obj: json.ObjectMap = .empty;
    try obj.put(allocator, "path", .{ .string = try allocator.dupe(u8, target_name) });
    try obj.put(allocator, "source_size_bytes", .{ .integer = @intCast(size) });
    try obj.put(allocator, "source_sha256", .{ .string = source_sha });
    try obj.put(allocator, "size_bytes", .{ .integer = @intCast(gz_res.stdout.len) });
    try obj.put(allocator, "retained", .{ .bool = true });
    return .{ .object = obj };
}

pub const LoadedBenchmarkReport = struct {
    report: Value,
    sha256: []const u8,
};

pub fn loadBenchmarkReport(allocator: Allocator, io: std.Io, path: []const u8) ProfileError!LoadedBenchmarkReport {
    const raw = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch |err| {
        return fail("cannot read benchmark report {s}: {s}", .{ path, @errorName(err) });
    };
    const parsed = json.parseFromSliceLeaky(Value, allocator, raw, .{}) catch |err| {
        return fail("cannot read benchmark report {s}: {s}", .{ path, @errorName(err) });
    };
    if (parsed != .object) return fail("benchmark report root must be a JSON object", .{});
    bench_coremark.validateAuthoritativeBenchmarkReport(allocator, parsed) catch |err| {
        return fail("invalid authoritative benchmark report: {s}", .{@errorName(err)});
    };
    const sha = try bench_coremark.sha256BytesAlloc(allocator, raw);
    return .{ .report = parsed, .sha256 = sha };
}

pub const CommandRecorder = struct {
    allocator: Allocator,
    io: std.Io,
    out_dir: []const u8,
    base_environ: *const std.process.Environ.Map,
    commands: std.ArrayList(Value) = .empty,

    pub fn runner(self: *CommandRecorder) CommandRunner {
        return .{
            .ctx = self,
            .runFn = runCallback,
        };
    }

    fn runCallback(
        ctx: *anyopaque,
        allocator: Allocator,
        command: []const []const u8,
        log_name: []const u8,
        cwd: []const u8,
        env_map: *const std.StringHashMap([]const u8),
    ) ProfileError!RunCommandResult {
        const self: *CommandRecorder = @ptrCast(@alignCast(ctx));
        return self.run(allocator, command, log_name, cwd, env_map);
    }

    pub fn run(
        self: *CommandRecorder,
        allocator: Allocator,
        command: []const []const u8,
        log_name: []const u8,
        cwd: []const u8,
        env_overrides: ?*const std.StringHashMap([]const u8),
    ) ProfileError!RunCommandResult {
        var rendered_buf: std.ArrayList(u8) = .empty;
        for (command, 0..) |part, idx| {
            if (idx > 0) try rendered_buf.append(allocator, ' ');
            try rendered_buf.appendSlice(allocator, part);
        }
        const rendered = try rendered_buf.toOwnedSlice(allocator);
        const start_ns = std.Io.Clock.awake.now(self.io).nanoseconds;

        var stdout_text: []const u8 = "";
        var stderr_text: []const u8 = "";
        var exit_code: i64 = 0;

        if (command.len >= 2 and std.mem.eql(u8, command[0], "zig-aot-jit-attr")) {
            var out_aw: std.Io.Writer.Allocating = .init(allocator);
            aot_jit_attr.runCli(allocator, self.io, command[2..], &out_aw.writer) catch {
                stderr_text = try allocator.dupe(u8, aot_jit_attr.lastError());
                exit_code = 1;
            };
            if (exit_code == 0) {
                stdout_text = try allocator.dupe(u8, out_aw.written());
            }
        } else {
            var merged_env = try self.base_environ.clone(allocator);
            defer merged_env.deinit();
            if (env_overrides) |ov| {
                var oit = ov.iterator();
                while (oit.next()) |kv| {
                    try merged_env.put(kv.key_ptr.*, kv.value_ptr.*);
                }
            }
            const proc = std.process.run(allocator, self.io, .{
                .argv = command,
                .cwd = .{ .path = cwd },
                .environ_map = &merged_env,
                .stdout_limit = .limited(256 * 1024 * 1024),
                .stderr_limit = .limited(64 * 1024 * 1024),
            }) catch |err| {
                return fail("command failed ({s}): {s}", .{ rendered, @errorName(err) });
            };
            stdout_text = proc.stdout;
            stderr_text = proc.stderr;
            exit_code = switch (proc.term) {
                .exited => |c| c,
                else => -1,
            };
        }

        const end_ns = std.Io.Clock.awake.now(self.io).nanoseconds;
        const elapsed_sec: f64 = @as(f64, @floatFromInt(@max(@as(i128, 0), end_ns - start_ns))) / 1_000_000_000.0;
        const rounded_elapsed = @round(elapsed_sec * 1000.0) / 1000.0;

        const log_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ self.out_dir, log_name });
        const stdout_nl = if (stdout_text.len == 0 or std.mem.endsWith(u8, stdout_text, "\n")) "" else "\n";
        const stderr_nl = if (stderr_text.len == 0 or std.mem.endsWith(u8, stderr_text, "\n")) "" else "\n";
        const log_content = try std.fmt.allocPrint(
            allocator,
            "$ {s}\n[exit={d} elapsed_seconds={d:.3}]\n--- stdout ---\n{s}{s}--- stderr ---\n{s}{s}",
            .{ rendered, exit_code, elapsed_sec, stdout_text, stdout_nl, stderr_text, stderr_nl },
        );
        std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = log_path, .data = log_content }) catch {
            return fail("cannot write command log: {s}", .{log_path});
        };

        var cmd_obj: json.ObjectMap = .empty;
        try cmd_obj.put(self.allocator, "command", .{ .string = rendered });
        try cmd_obj.put(self.allocator, "cwd", .{ .string = try self.allocator.dupe(u8, cwd) });
        try cmd_obj.put(self.allocator, "exit_code", .{ .integer = exit_code });
        try cmd_obj.put(self.allocator, "elapsed_seconds", .{ .float = rounded_elapsed });
        try cmd_obj.put(self.allocator, "log", .{ .string = try self.allocator.dupe(u8, std.fs.path.basename(log_name)) });
        try self.commands.append(self.allocator, .{ .object = cmd_obj });

        if (exit_code != 0) {
            const detail = std.mem.trim(u8, if (stderr_text.len != 0) stderr_text else stdout_text, " \t\r\n");
            if (detail.len != 0) {
                return fail("command failed ({s}): {s}", .{ rendered, detail });
            }
            return fail("command failed ({s}): exit {d}", .{ rendered, exit_code });
        }

        return .{ .stdout = stdout_text, .stderr = stderr_text };
    }
};

pub fn captureAnalysisSources(
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    commit: []const u8,
    rel_paths: []const []const u8,
    git_cat_fn: ?*const fn (allocator: Allocator, repo: []const u8, spec: []const u8) ?[]const u8,
) ProfileError!Value {
    var files_obj: json.ObjectMap = .empty;
    for (rel_paths) |rel| {
        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ repo, rel });
        const worktree_bytes = std.Io.Dir.cwd().readFileAlloc(io, full_path, allocator, .limited(64 * 1024 * 1024)) catch {
            return fail("cannot read analysis source {s}", .{rel});
        };
        const spec = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ commit, rel });
        const committed_bytes = if (git_cat_fn) |fn_ptr|
            fn_ptr(allocator, repo, spec) orelse return fail("analysis source {s} is not tracked at {s}", .{ rel, commit })
        else blk: {
            const argv = [_][]const u8{ "git", "-C", repo, "cat-file", "blob", spec };
            const res = std.process.run(allocator, io, .{
                .argv = &argv,
                .stdout_limit = .limited(64 * 1024 * 1024),
                .stderr_limit = .limited(1024 * 1024),
            }) catch return fail("analysis source {s} is not tracked at {s}", .{ rel, commit });
            switch (res.term) {
                .exited => |code| if (code != 0) return fail("analysis source {s} is not tracked at {s}", .{ rel, commit }),
                else => return fail("analysis source {s} is not tracked at {s}", .{ rel, commit }),
            }
            break :blk res.stdout;
        };
        if (!std.mem.eql(u8, worktree_bytes, committed_bytes)) {
            return fail("analysis source {s} differs from committed {s}", .{ rel, commit });
        }
        const digest = try bench_coremark.sha256BytesAlloc(allocator, worktree_bytes);
        try files_obj.put(allocator, try allocator.dupe(u8, rel), .{ .string = digest });
    }

    var out: json.ObjectMap = .empty;
    try out.put(allocator, "source_mode", .{ .string = "commit-verified" });
    try out.put(allocator, "commit", .{ .string = try allocator.dupe(u8, commit) });
    try out.put(allocator, "files", .{ .object = files_obj });
    return .{ .object = out };
}

pub fn validateAnalysisSourcesUnchanged(
    allocator: Allocator,
    io: std.Io,
    repo: []const u8,
    sources_val: Value,
) ProfileError!void {
    if (sources_val != .object) return fail("invalid analysis_sources", .{});
    const files_val = sources_val.object.get("files") orelse return fail("invalid analysis_sources", .{});
    if (files_val != .object) return fail("invalid analysis_sources", .{});
    var it = files_val.object.iterator();
    while (it.next()) |kv| {
        const rel = kv.key_ptr.*;
        const expected = jsonStringOrEmpty(kv.value_ptr.*);
        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ repo, rel });
        const actual = bench_coremark.sha256File(allocator, io, full_path) catch {
            return fail("analysis source {s} changed during profiling", .{rel});
        };
        if (!std.mem.eql(u8, actual, expected)) {
            return fail("analysis source {s} changed during profiling", .{rel});
        }
    }
}

pub fn renderMarkdown(allocator: Allocator, report_val: Value) ProfileError![]u8 {
    if (report_val != .object) return fail("report must be an object", .{});
    const report = report_val.object;
    const host = report.get("host").?.object;
    const engines = report.get("engines").?.object;
    const wamr = engines.get("wamr").?.object;
    const wasmtime = engines.get("wasmtime").?.object;
    const benchmark = report.get("benchmark").?.object;
    const execution = benchmark.get("execution").?.object;

    const benchmark_link = if (std.mem.eql(u8, jsonStringOrEmpty(execution.get("provider")), "github-actions"))
        try std.fmt.allocPrint(
            allocator,
            "[{s}](https://github.com/{s}/actions/runs/{s})",
            .{
                jsonStringOrEmpty(execution.get("run_id")),
                jsonStringOrEmpty(execution.get("repository")),
                jsonStringOrEmpty(execution.get("run_id")),
            },
        )
    else
        try std.fmt.allocPrint(allocator, "`{s}`", .{jsonStringOrEmpty(execution.get("run_id"))});

    var order_buf: std.ArrayList(u8) = .empty;
    for (report.get("profile_schedule").?.array.items) |item| {
        const eng = jsonStringOrEmpty(item.object.get("engine"));
        try order_buf.append(allocator, if (std.mem.eql(u8, eng, "wamr")) 'A' else 'B');
    }

    var wamr_cov_buf: std.ArrayList(u8) = .empty;
    for (report.get("wamr_captures").?.array.items, 0..) |item, idx| {
        if (idx > 0) try wamr_cov_buf.appendSlice(allocator, ", ");
        try wamr_cov_buf.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d:.4}%", .{jsonNumber(item.object.get("coverage_pct")) orelse 0.0}));
    }
    var wt_cov_buf: std.ArrayList(u8) = .empty;
    for (report.get("wasmtime_captures").?.array.items, 0..) |item, idx| {
        if (idx > 0) try wt_cov_buf.appendSlice(allocator, ", ");
        try wt_cov_buf.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d:.4}%", .{jsonNumber(item.object.get("coverage_pct")) orelse 0.0}));
    }

    var guest_args_buf: std.ArrayList(u8) = .empty;
    if (report.get("guest_args")) |ga_val| {
        if (ga_val == .array) {
            for (ga_val.array.items, 0..) |arg_v, idx| {
                if (idx > 0) try guest_args_buf.append(allocator, ' ');
                try guest_args_buf.appendSlice(allocator, jsonStringOrEmpty(arg_v));
            }
        }
    }

    var allowed_cpus_buf: std.ArrayList(u8) = .empty;
    const affinity_obj = report.get("affinity").?.object;
    if (affinity_obj.get("allowed_cpus")) |ac_val| {
        if (ac_val == .array) {
            for (ac_val.array.items, 0..) |cpu_v, idx| {
                if (idx > 0) try allowed_cpus_buf.append(allocator, ',');
                try allowed_cpus_buf.appendSlice(allocator, try std.fmt.allocPrint(allocator, "{d}", .{jsonInt(cpu_v) orelse 0}));
            }
        }
    }

    var out: std.ArrayList(u8) = .empty;
    var w: std.Io.Writer.Allocating = .fromArrayList(allocator, &out);
    const writer = &w.writer;

    const wamr_info = report.get("wamr").?.object;
    const handoff = benchmark.get("artifact_handoff").?.object;
    const selected_role = jsonString(benchmark.get("selected_role")) orelse "wamr-target";
    const selected_wamr = (benchmark.get("selected_wamr") orelse benchmark.get("target").?).object;
    const selected_sha = selected_wamr.get("identity").?.object.get("source").?.object.get("sha").?.string;
    const prod_sha = benchmark.get("producer").?.object.get("source_sha").?.string;
    const fixture = report.get("fixture").?.object;

    try writer.writeAll("### Matched-host AArch64 CoreMark profiles\n\n");
    try writer.print("- Commit: `{s}` (`ReleaseFast`)\n", .{jsonStringOrEmpty(wamr_info.get("commit"))});
    try writer.print(
        "- Authoritative benchmark: report `{s}` (`sha256:{s}`), run {s}\n",
        .{ jsonStringOrEmpty(benchmark.get("report_id")), jsonStringOrEmpty(benchmark.get("report_sha256")), benchmark_link },
    );
    try writer.print("- Exact WAMR artifact handoff: manifest `sha256:{s}`\n", .{jsonStringOrEmpty(handoff.get("manifest_sha256"))});
    try writer.print("- Benchmarked WAMR role: `{s}` at `{s}`; producer tooling `{s}`\n", .{ selected_role, selected_sha, prod_sha });
    try writer.print("- Fixture: `{s}` (`sha256:{s}`)\n", .{ jsonStringOrEmpty(fixture.get("path")), jsonStringOrEmpty(fixture.get("sha256")) });
    try writer.print("- Fixed guest args: `{s}`; every run required `Iterations: {d}`\n", .{ guest_args_buf.items, jsonInt(report.get("expected_iterations")) orelse 0 });
    try writer.print(
        "- CPU affinity: allowed `{s}`; selected/verified CPU `{d}` via `{s}`\n",
        .{ allowed_cpus_buf.items, jsonInt(affinity_obj.get("selected_cpu")) orelse 0, jsonStringOrEmpty(affinity_obj.get("taskset")) },
    );
    try writer.print("- Counterbalanced execution order: `{s}` (A=WAMR, B=Wasmtime; warmups then profile captures)\n", .{order_buf.items});
    try writer.print(
        "- Per-capture attribution gate: `≥{d:.2}%`; WAMR `{s}`, Wasmtime `{s}`\n",
        .{ jsonNumber(report.get("minimum_attribution_coverage_pct")) orelse 99.0, wamr_cov_buf.items, wt_cov_buf.items },
    );
    try writer.print(
        "- Host: `{s}` · {d} vCPU · `{s}` · kernel `{s}` · fingerprint `{s}`\n",
        .{
            jsonStringOrEmpty(host.get("architecture")),
            jsonInt(host.get("cpu_count")) orelse 0,
            jsonStringOrEmpty(host.get("cpu_model")),
            jsonStringOrEmpty(host.get("kernel")),
            jsonStringOrEmpty(host.get("fingerprint")),
        },
    );
    if (report.get("perf")) |perf_val| {
        if (perf_val == .object) {
            const perf = perf_val.object;
            try writer.print(
                "- perf: `{s}` · package `{s}` · paranoid `{s} → {s}` · native sampling verified\n",
                .{
                    jsonStringOrEmpty(perf.get("version")),
                    jsonStringOrEmpty(perf.get("package")),
                    jsonStringOrEmpty(perf.get("paranoid_initial")),
                    jsonStringOrEmpty(perf.get("paranoid_effective")),
                },
            );
        }
    }
    if (report.get("wasmtime")) |wt_info_val| {
        if (wt_info_val == .object) {
            const wt_info = wt_info_val.object;
            try writer.print(
                "- Wasmtime: `{s}` (`sha256:{s}`), `--profile=jitdump`\n",
                .{ jsonStringOrEmpty(wt_info.get("version")), jsonStringOrEmpty(wt_info.get("sha256")) },
            );
        }
    }
    try writer.print(
        "- Validated runs: WAMR {d:.2}s at {d:.1} iter/s; Wasmtime {d:.2}s at {d:.1} iter/s\n\n",
        .{
            jsonNumber(wamr.get("wall_seconds")) orelse 0.0,
            jsonNumber(wamr.get("iterations_per_second")) orelse 0.0,
            jsonNumber(wasmtime.get("wall_seconds")) orelse 0.0,
            jsonNumber(wasmtime.get("iterations_per_second")) orelse 0.0,
        },
    );

    try writer.writeAll("| Engine | Total self samples | Generated-wasm samples | Engine share |\n");
    try writer.writeAll("|---|---:|---:|---:|\n");
    try writer.print(
        "| WAMR | {d} | {d} | {d:.2}% |\n",
        .{
            jsonInt(wamr.get("total_samples")) orelse 0,
            jsonInt(wamr.get("attributed_samples")) orelse 0,
            jsonNumber(wamr.get("attribution_coverage_pct")) orelse 0.0,
        },
    );
    try writer.print(
        "| Wasmtime | {d} | {d} | {d:.2}% |\n",
        .{
            jsonInt(wasmtime.get("total_samples")) orelse 0,
            jsonInt(wasmtime.get("attributed_samples")) orelse 0,
            jsonNumber(wasmtime.get("attribution_coverage_pct")) orelse 0.0,
        },
    );

    if (report.get("matched_functions")) |mf_val| {
        if (mf_val == .array) {
            try writer.writeAll("\n#### Top functions\n\n");
            try writer.writeAll("| WAMR local | Wasm index | Name | WAMR run share | Wasmtime run share |\n");
            try writer.writeAll("|---:|---:|---|---:|---:|\n");
            for (mf_val.array.items) |item_v| {
                const item = item_v.object;
                try writer.print(
                    "| {d} | {d} | `{s}` | {d:.2}% | {d:.2}% |\n",
                    .{
                        jsonInt(item.get("local_func")) orelse 0,
                        jsonInt(item.get("wasm_function_index")) orelse 0,
                        jsonStringOrEmpty(item.get("name")),
                        jsonNumber(item.get("wamr").?.object.get("percent_of_run")) orelse 0.0,
                        jsonNumber(item.get("wasmtime").?.object.get("percent_of_run")) orelse 0.0,
                    },
                );
            }

            if (report.get("alu_provenance")) |ap_top| {
                if (ap_top != .null) {
                    try writer.writeAll("\n#### Conservative common-universe ALU provenance\n\n");
                    try writer.writeAll("| Function | Category | WAMR samples / run share | Wasmtime samples / run share | Conservative headroom |\n");
                    try writer.writeAll("|---|---|---:|---:|---:|\n");
                    for (mf_val.array.items) |item_v| {
                        const item = item_v.object;
                        const fn_name = jsonStringOrEmpty(item.get("name"));
                        const narrow = item.get("alu_provenance").?.object;
                        const gate = narrow.get("gate").?.object;
                        const gate_cats_opt = gate.get("categories");
                        for (provenance.CATEGORIES) |cat| {
                            const w_cat = narrow.get("wamr").?.object.get("common_gating_universe").?.object.get("categories").?.object.get(cat).?.object;
                            const wt_cat = narrow.get("wasmtime").?.object.get("common_gating_universe").?.object.get("categories").?.object.get(cat).?.object;
                            var headroom: []const u8 = "not gate-eligible";
                            if (gate_cats_opt) |gc| {
                                if (gc == .object) {
                                    if (gc.object.get(cat)) |gv| {
                                        if (gv == .object) {
                                            const h_val = jsonNumber(gv.object.get("conservative_headroom_percentage_points")) orelse 0.0;
                                            const sign: []const u8 = if (h_val >= 0.0) "+" else "";
                                            headroom = try std.fmt.allocPrint(allocator, "{s}{d:.2} pp", .{ sign, h_val });
                                        }
                                    }
                                }
                            }
                            try writer.print(
                                "| `{s}` | `{s}` | {d} / {d:.2}% | {d} / {d:.2}% | {s} |\n",
                                .{
                                    fn_name,
                                    cat,
                                    jsonInt(w_cat.get("samples")) orelse 0,
                                    jsonNumber(w_cat.get("percent_of_run")) orelse 0.0,
                                    jsonInt(wt_cat.get("samples")) orelse 0,
                                    jsonNumber(wt_cat.get("percent_of_run")) orelse 0.0,
                                    headroom,
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    if (report.get("frame_attribution")) |fa_val| {
        if (fa_val == .object) {
            const fa = fa_val.object;
            const summary = fa.get("summary").?.object;
            const coverage = summary.get("coverage").?.object;
            const meta = fa.get("metadata").?.object;
            try writer.writeAll("\n#### Exact WAMR frame origins\n\n");
            try writer.print(
                "`{s}`: module 0, local_func {d}, wasm index {d}. Sidecar `{s}` (`sha256:{s}`).\n\n",
                .{
                    jsonStringOrEmpty(fa.get("name")),
                    jsonInt(fa.get("local_func")) orelse 0,
                    jsonInt(fa.get("wasm_function_index")) orelse 0,
                    jsonStringOrEmpty(meta.get("path")),
                    jsonStringOrEmpty(meta.get("sha256")),
                },
            );
            try writer.print(
                "Proven origins: {d}/{d} static native frame instructions; {d}/{d} frame samples. Unknown: {d} instructions / {d} samples.\n",
                .{
                    jsonInt(coverage.get("proven_origin_frame_instructions")) orelse 0,
                    jsonInt(coverage.get("frame_instructions")) orelse 0,
                    jsonInt(coverage.get("proven_origin_frame_samples")) orelse 0,
                    jsonInt(coverage.get("frame_samples")) orelse 0,
                    jsonInt(coverage.get("unknown_frame_instructions")) orelse 0,
                    jsonInt(coverage.get("unknown_frame_samples")) orelse 0,
                },
            );
        }
    }

    out = w.toArrayList();
    return out.toOwnedSlice(allocator);
}

pub const ProfileCliOptions = struct {
    repo: ?[]const u8 = null,
    out_dir: []const u8,
    benchmark_report: []const u8,
    wamr_ref: ?[]const u8 = null,
    benchmark_role: []const u8 = "wamr-target",
    benchmark_artifacts: []const u8,
    execution_id: ?[]const u8 = null,
    wasmtime_cache: ?[]const u8 = null,
    frequency: usize = 999,
    min_samples: i64 = DEFAULT_MIN_SAMPLES,
    top: usize = DEFAULT_TOP_FUNCTIONS,
    classify: usize = DEFAULT_CLASSIFY_FUNCTIONS,
    frame_func: ?usize = null,
    max_perf_bytes: usize = DEFAULT_MAX_PERF_BYTES,
};

fn listJitDumps(allocator: Allocator, io: std.Io, dir_path: []const u8) !std.StringArrayHashMapUnmanaged(void) {
    var set: std.StringArrayHashMapUnmanaged(void) = .empty;
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return set;
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind == .file and std.mem.startsWith(u8, entry.name, "jit-") and std.mem.endsWith(u8, entry.name, ".dump")) {
            const full = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
            try set.put(allocator, full, {});
        }
    }
    return set;
}

pub fn runProfile(
    allocator: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
    opts: ProfileCliOptions,
) anyerror!Value {
    const repo = std.Io.Dir.cwd().realPathFileAlloc(io, opts.repo orelse ".", allocator) catch {
        return fail("cannot resolve repo path", .{});
    };
    std.Io.Dir.cwd().createDirPath(io, opts.out_dir) catch {
        return fail("cannot create out_dir: {s}", .{opts.out_dir});
    };
    const out_dir = std.Io.Dir.cwd().realPathFileAlloc(io, opts.out_dir, allocator) catch {
        return fail("cannot resolve out_dir: {s}", .{opts.out_dir});
    };
    const benchmark_report_resolved = std.Io.Dir.cwd().realPathFileAlloc(io, opts.benchmark_report, allocator) catch {
        return fail("cannot resolve benchmark report: {s}", .{opts.benchmark_report});
    };
    const benchmark_artifacts_resolved = std.Io.Dir.cwd().realPathFileAlloc(io, opts.benchmark_artifacts, allocator) catch {
        return fail("cannot resolve benchmark artifacts: {s}", .{opts.benchmark_artifacts});
    };

    var recorder = CommandRecorder{
        .allocator = allocator,
        .io = io,
        .out_dir = out_dir,
        .base_environ = environ_map,
    };

    const loaded_bench = try loadBenchmarkReport(allocator, io, benchmark_report_resolved);
    const benchmark_report = loaded_bench.report;
    const benchmark_report_sha = loaded_bench.sha256;

    const benchmark_wamr = bench_coremark.engineWithRole(benchmark_report.object, opts.benchmark_role) catch |err| {
        return fail("benchmark role lookup failed: {s}", .{@errorName(err)});
    };
    const benchmark_wamr_sha = benchmark_wamr.object.get("identity").?.object.get("source").?.object.get("sha").?.string;

    const current_execution = bench_coremark.captureExecutionIdentity(allocator, environ_map, opts.execution_id) catch |err| {
        return fail("captureExecutionIdentity failed: {s}", .{@errorName(err)});
    };
    bench_coremark.validateExecutionMatch(
        benchmark_report.object.get("provenance").?.object.get("execution").?,
        current_execution,
    ) catch |err| {
        return fail("validateExecutionMatch failed: {s}", .{@errorName(err)});
    };
    const handoff = bench_coremark.loadWamrArtifactHandoff(
        allocator,
        io,
        benchmark_artifacts_resolved,
        benchmark_wamr.object.get("identity").?,
        opts.benchmark_role,
    ) catch |err| {
        return fail("loadWamrArtifactHandoff failed: {s}", .{@errorName(err)});
    };
    const manifest_path = try std.fs.path.join(allocator, &.{ benchmark_artifacts_resolved, "manifest.json" });
    const handoff_manifest_sha = bench_coremark.sha256File(allocator, io, manifest_path) catch {
        return fail("cannot hash handoff manifest", .{});
    };

    const host_identity = bench_coremark.validateNativeHost(allocator, io, environ_map, "aarch64") catch |err| {
        return fail("validateNativeHost failed: {s}", .{@errorName(err)});
    };
    const resolved_fixture = bench_coremark.resolveFixture(allocator, io, repo, bench_coremark.default_fixture_rel) catch |err| {
        return fail("resolveFixture failed: {s}", .{@errorName(err)});
    };
    const fixture = resolved_fixture.path;
    const fixture_sha = resolved_fixture.sha256;

    const wasm_identity = compare_hot_function.parseCoreWasm(allocator, io, fixture) catch |err| {
        return fail("parseCoreWasm failed: {s}", .{@errorName(err)});
    };
    if (opts.frame_func) |ff| {
        if (ff >= wasm_identity.local_function_count) {
            return fail("frame function is outside the benchmark module", .{});
        }
    }

    const wamr_ref = opts.wamr_ref orelse benchmark_wamr_sha;
    const commit_res = try recorder.run(allocator, &.{ "git", "rev-parse", wamr_ref }, "git-identity.log", repo, null);
    const commit = std.mem.trim(u8, commit_res.stdout, " \t\r\n");
    if (!std.mem.eql(u8, commit, benchmark_wamr_sha)) {
        return fail("--wamr-ref resolves to {s}, but benchmark role {s} is {s}", .{ commit, opts.benchmark_role, benchmark_wamr_sha });
    }
    const checkout_commit = bench_coremark.resolveRefSha(allocator, io, repo, "HEAD") catch {
        return fail("cannot resolve HEAD SHA", .{});
    };
    var analysis_sources = try captureAnalysisSources(allocator, io, repo, checkout_commit, &REQUIRED_ANALYSIS_SOURCES, null);

    const build_repo = benchmark_artifacts_resolved;
    const kernel = std.mem.trim(u8, (try recorder.run(allocator, &.{ "uname", "-r" }, "kernel.log", repo, null)).stdout, " \t\r\n");
    const uname = std.mem.trim(u8, (try recorder.run(allocator, &.{ "uname", "-a" }, "uname.log", repo, null)).stdout, " \t\r\n");
    const perf_bin = environ_map.get("PERF") orelse "perf";
    const perf_version = std.mem.trim(u8, (try recorder.run(allocator, &.{ perf_bin, "--version" }, "perf-version.log", repo, null)).stdout, " \t\r\n");

    const wamr = handoff.get("runtime").?;
    const wamrc = handoff.get("compiler").?;
    const cwasm = handoff.get("module").?;
    const wamr_version = std.mem.trim(u8, (try recorder.run(allocator, &.{ wamr, "version" }, "wamr-version.log", build_repo, null)).stdout, " \t\r\n");
    const wamrc_version = std.mem.trim(u8, (try recorder.run(allocator, &.{ wamrc, "version" }, "wamrc-version.log", build_repo, null)).stdout, " \t\r\n");

    var base_env_map: std.StringHashMap([]const u8) = .init(allocator);
    var env_it = environ_map.iterator();
    while (env_it.next()) |kv| {
        try base_env_map.put(kv.key_ptr.*, kv.value_ptr.*);
    }
    const helper_path = try std.fs.path.join(allocator, &.{ repo, "scripts/aot_jit_attr.zig" });
    const diag_res = try compileWamrDiagnostics(
        allocator,
        io,
        recorder.runner(),
        wamrc,
        fixture,
        cwasm,
        out_dir,
        build_repo,
        helper_path,
        opts.frame_func,
        &base_env_map,
    );
    const spill_metrics = diag_res.metrics;
    const frame_metadata = diag_res.metadata_path;
    const frame_metadata_sha = if (frame_metadata) |fm|
        (bench_coremark.sha256File(allocator, io, fm) catch return fail("cannot hash frame metadata", .{}))
    else
        null;

    const cwasm_info = aot_jit_attr.parseCwasm(allocator, io, cwasm) catch {
        return fail("cannot parse WAMR cwasm: {s}", .{aot_jit_attr.lastError()});
    };
    var supported_ver = false;
    for (aot_jit_attr.SUPPORTED_AOT_VERSIONS) |sv| {
        if (cwasm_info.version == sv) {
            supported_ver = true;
            break;
        }
    }
    if (!supported_ver) return fail("WAMR cwasm version is unsupported by attribution", .{});

    const profile_wamr_identity = bench_coremark.makeWamrIdentity(
        allocator,
        io,
        .{
            .ref = wamr_ref,
            .source_sha = commit,
            .optimize = "ReleaseFast",
            .runtime_path = wamr,
            .compiler_path = wamrc,
            .module_path = cwasm,
        },
    ) catch |err| return fail("makeWamrIdentity failed: {s}", .{@errorName(err)});

    const retained_cwasm_target = try std.fs.path.join(allocator, &.{ out_dir, "wamr-profiled.cwasm.gz" });
    const retained_wamr_binary = try retainExactBinary(allocator, io, cwasm, retained_cwasm_target, opts.max_perf_bytes);
    if (!std.mem.eql(
        u8,
        retained_wamr_binary.object.get("source_sha256").?.string,
        profile_wamr_identity.object.get("module").?.object.get("sha256").?.string,
    )) {
        return fail("retained WAMR cwasm identity changed during handoff", .{});
    }

    const wasmtime = bench_coremark.installPinnedWasmtime(allocator, io, repo, opts.wasmtime_cache) catch |err| {
        return fail("installPinnedWasmtime failed: {s}", .{@errorName(err)});
    };
    const wasmtime_version = bench_coremark.validatePinnedWasmtime(allocator, io, wasmtime) catch |err| {
        return fail("validatePinnedWasmtime failed: {s}", .{@errorName(err)});
    };
    const profile_wasmtime_identity = bench_coremark.makeWasmtimeIdentity(
        allocator,
        io,
        .{
            .version = wasmtime_version,
            .runtime_path = wasmtime,
            .channel = "historical-pin",
        },
    ) catch |err| return fail("makeWasmtimeIdentity failed: {s}", .{@errorName(err)});

    const wasmtime_cwasm = try std.fs.path.join(allocator, &.{ out_dir, "coremark.wasmtime.cwasm" });
    _ = try recorder.run(
        allocator,
        &.{ wasmtime, "compile", "-O", "opt-level=2", fixture, "-o", wasmtime_cwasm },
        "wasmtime-compile.log",
        out_dir,
        null,
    );
    const objdump = (try recorder.run(
        allocator,
        &.{ wasmtime, "objdump", wasmtime_cwasm, "--addresses", "--bytes" },
        "wasmtime-objdump-command.log",
        out_dir,
        null,
    )).stdout;
    const objdump_file = try std.fs.path.join(allocator, &.{ out_dir, "wasmtime-objdump.txt" });
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = objdump_file, .data = objdump }) catch {
        return fail("cannot write wasmtime-objdump.txt", .{});
    };

    const affinity = bench_coremark.selectCpuAffinity(allocator, io, environ_map) catch |err| {
        return fail("selectCpuAffinity failed: {s}", .{@errorName(err)});
    };
    const bench_script_path = try std.fs.path.join(allocator, &.{ repo, "scripts/bench_coremark.zig" });
    const producer_script_sha = bench_coremark.sha256File(allocator, io, bench_script_path) catch {
        return fail("cannot hash scripts/bench_coremark.zig", .{});
    };
    const benchmark_identity = bench_coremark.validateBenchmarkProfileMatch(
        allocator,
        benchmark_report,
        .{
            .expected_arch = "aarch64",
            .fixture_sha = fixture_sha,
            .host = host_identity,
            .affinity = affinity,
            .wamr_source_sha = commit,
            .wamr_optimize = "ReleaseFast",
            .wamr_runtime_sha = profile_wamr_identity.object.get("runtime").?.object.get("sha256").?.string,
            .wamr_compiler_sha = profile_wamr_identity.object.get("compiler").?.object.get("sha256").?.string,
            .wamr_module_sha = profile_wamr_identity.object.get("module").?.object.get("sha256").?.string,
            .wasmtime_version_value = wasmtime_version,
            .wasmtime_runtime_sha = profile_wasmtime_identity.object.get("runtime").?.object.get("sha256").?.string,
            .producer_source_sha = checkout_commit,
            .producer_script_sha = producer_script_sha,
            .current_execution = current_execution,
            .benchmark_role = opts.benchmark_role,
        },
    ) catch |err| return fail("validateBenchmarkProfileMatch failed: {s}", .{@errorName(err)});

    const guest_args = bench_coremark.coremarkGuestArgs(allocator, bench_coremark.expected_iterations) catch unreachable;
    var wamr_base_cmd: std.ArrayList([]const u8) = .empty;
    try wamr_base_cmd.appendSlice(allocator, &.{ wamr, "run", cwasm });
    try wamr_base_cmd.appendSlice(allocator, guest_args);
    const wamr_command = bench_coremark.applyAffinity(allocator, wamr_base_cmd.items, affinity) catch unreachable;

    var wt_base_cmd: std.ArrayList([]const u8) = .empty;
    try wt_base_cmd.appendSlice(allocator, &.{ wasmtime, "run", "--allow-precompiled", wasmtime_cwasm });
    try wt_base_cmd.appendSlice(allocator, guest_args);
    const wasmtime_command = bench_coremark.applyAffinity(allocator, wt_base_cmd.items, affinity) catch unreachable;

    var wt_prof_base_cmd: std.ArrayList([]const u8) = .empty;
    try wt_prof_base_cmd.appendSlice(allocator, &.{ wasmtime, "run", "--allow-precompiled", "--profile=jitdump", wasmtime_cwasm });
    try wt_prof_base_cmd.appendSlice(allocator, guest_args);
    const wasmtime_profile_command = bench_coremark.applyAffinity(allocator, wt_prof_base_cmd.items, affinity) catch unreachable;

    var profile_schedule: std.ArrayList(Value) = .empty;
    var wamr_warmup_ord: i64 = 0;
    var wt_warmup_ord: i64 = 0;
    var schedule_position: i64 = 0;
    const warmup_order = bench_coremark.counterbalancedOrder(allocator, &.{ "wamr", "wasmtime" }, 2) catch unreachable;
    for (warmup_order) |engine| {
        schedule_position += 1;
        const is_wamr = std.mem.eql(u8, engine, "wamr");
        const ord = if (is_wamr) blk: {
            wamr_warmup_ord += 1;
            break :blk wamr_warmup_ord;
        } else blk: {
            wt_warmup_ord += 1;
            break :blk wt_warmup_ord;
        };
        const command = if (is_wamr) wamr_command else wasmtime_command;
        const cwd = if (is_wamr) build_repo else out_dir;
        const started_at = bench_coremark.formatUtcIsoTimestamp(allocator, io) catch unreachable;
        const log_name = try std.fmt.allocPrint(allocator, "{s}-warmup-{d}.log", .{ engine, ord });
        const result = try recorder.run(allocator, command, log_name, cwd, null);
        const completed_at = bench_coremark.formatUtcIsoTimestamp(allocator, io) catch unreachable;
        const combined = try std.fmt.allocPrint(allocator, "{s}{s}", .{ result.stdout, result.stderr });
        const parsed_cm = try parseValidatedCoremark(combined, engine, 1);

        var sched_item: json.ObjectMap = .empty;
        try sched_item.put(allocator, "position", .{ .integer = schedule_position });
        try sched_item.put(allocator, "phase", .{ .string = "warmup" });
        try sched_item.put(allocator, "engine", .{ .string = engine });
        try sched_item.put(allocator, "engine_ordinal", .{ .integer = ord });
        try sched_item.put(allocator, "started_at", .{ .string = started_at });
        try sched_item.put(allocator, "completed_at", .{ .string = completed_at });
        try sched_item.put(allocator, "iterations", .{ .integer = bench_coremark.expected_iterations });
        try sched_item.put(allocator, "iterations_per_second", .{ .float = parsed_cm.throughput });
        try profile_schedule.append(allocator, .{ .object = sched_item });
    }

    var wamr_perfs: std.ArrayList([]const u8) = .empty;
    var wasmtime_perfs: std.ArrayList([]const u8) = .empty;
    var jitdumps: std.ArrayList([]const u8) = .empty;
    var wamr_values: std.ArrayList(f64) = .empty;
    var wasmtime_values: std.ArrayList(f64) = .empty;
    var wasmtime_captures: std.ArrayList(ParsedWasmtimeSamples) = .empty;
    var wamr_prof_ord: i64 = 0;
    var wt_prof_ord: i64 = 0;

    const prof_order = bench_coremark.counterbalancedOrder(allocator, &.{ "wamr", "wasmtime" }, PROFILE_CAPTURES_PER_ENGINE) catch unreachable;
    for (prof_order) |engine| {
        schedule_position += 1;
        const is_wamr = std.mem.eql(u8, engine, "wamr");
        const ordinal = if (is_wamr) blk: {
            wamr_prof_ord += 1;
            break :blk wamr_prof_ord;
        } else blk: {
            wt_prof_ord += 1;
            break :blk wt_prof_ord;
        };
        const started_at = bench_coremark.formatUtcIsoTimestamp(allocator, io) catch unreachable;
        const freq_str = try std.fmt.allocPrint(allocator, "{d}", .{opts.frequency});
        var perf_path: []const u8 = "";
        var value: f64 = 0.0;

        if (is_wamr) {
            perf_path = try std.fs.path.join(allocator, &.{ out_dir, try std.fmt.allocPrint(allocator, "wamr-{d}.perf.data", .{ordinal}) });
            var rec_cmd: std.ArrayList([]const u8) = .empty;
            try rec_cmd.appendSlice(allocator, &.{ perf_bin, "record", "-k", "mono", "-F", freq_str, "-e", "cycles:u", "-o", perf_path, "--" });
            try rec_cmd.appendSlice(allocator, wamr_command);
            const result = try recorder.run(allocator, rec_cmd.items, try std.fmt.allocPrint(allocator, "wamr-run-{d}.log", .{ordinal}), build_repo, null);
            const combined = try std.fmt.allocPrint(allocator, "{s}{s}", .{ result.stdout, result.stderr });
            value = (try parseValidatedCoremark(combined, "WAMR", 1)).throughput;
            try wamr_values.append(allocator, value);
            try wamr_perfs.append(allocator, perf_path);
        } else {
            perf_path = try std.fs.path.join(allocator, &.{ out_dir, try std.fmt.allocPrint(allocator, "wasmtime-{d}.perf.data", .{ordinal}) });
            const before = try listJitDumps(allocator, io, out_dir);
            var rec_cmd: std.ArrayList([]const u8) = .empty;
            try rec_cmd.appendSlice(allocator, &.{ perf_bin, "record", "-k", "mono", "-F", freq_str, "-e", "cycles:u", "-o", perf_path, "--" });
            try rec_cmd.appendSlice(allocator, wasmtime_profile_command);
            const result = try recorder.run(allocator, rec_cmd.items, try std.fmt.allocPrint(allocator, "wasmtime-run-{d}.log", .{ordinal}), out_dir, null);
            const combined = try std.fmt.allocPrint(allocator, "{s}{s}", .{ result.stdout, result.stderr });
            value = (try parseValidatedCoremark(combined, "Wasmtime", 1)).throughput;
            try wasmtime_values.append(allocator, value);
            try wasmtime_perfs.append(allocator, perf_path);

            const after = try listJitDumps(allocator, io, out_dir);
            var new_dumps: std.ArrayList([]const u8) = .empty;
            for (after.keys()) |k| {
                if (!before.contains(k)) try new_dumps.append(allocator, k);
            }
            if (new_dumps.items.len != 1) {
                return fail("Wasmtime capture {d} produced {d} new jitdump files; expected exactly one", .{ ordinal, new_dumps.items.len });
            }
            try jitdumps.appendSlice(allocator, new_dumps.items);
            const injected = try std.fs.path.join(allocator, &.{ out_dir, try std.fmt.allocPrint(allocator, "wasmtime-{d}.perf.jit.data", .{ordinal}) });
            _ = try recorder.run(
                allocator,
                &.{ perf_bin, "inject", "--jit", "-i", perf_path, "-o", injected },
                try std.fmt.allocPrint(allocator, "wasmtime-perf-inject-{d}.log", .{ordinal}),
                out_dir,
                null,
            );
            const perf_script = (try recorder.run(
                allocator,
                &.{ perf_bin, "script", "-i", injected, "-F", "ip,sym,symoff,dso" },
                try std.fmt.allocPrint(allocator, "wasmtime-perf-script-{d}.log", .{ordinal}),
                out_dir,
                null,
            )).stdout;
            const samples_txt = try std.fs.path.join(allocator, &.{ out_dir, try std.fmt.allocPrint(allocator, "wasmtime-samples-{d}.txt", .{ordinal}) });
            std.Io.Dir.cwd().writeFile(io, .{ .sub_path = samples_txt, .data = perf_script }) catch {
                return fail("cannot write wasmtime-samples-{d}.txt", .{ordinal});
            };
            const parsed_wt = try parseWasmtimeSamples(allocator, perf_script, &wasm_identity);
            try validateWasmtimeMapping(&parsed_wt, &wasm_identity);
            _ = try validateWasmtimeCapture(allocator, &parsed_wt, opts.min_samples, MIN_ATTRIBUTION_COVERAGE_PCT);
            try wasmtime_captures.append(allocator, parsed_wt);
        }

        const completed_at = bench_coremark.formatUtcIsoTimestamp(allocator, io) catch unreachable;
        var sched_item: json.ObjectMap = .empty;
        try sched_item.put(allocator, "position", .{ .integer = schedule_position });
        try sched_item.put(allocator, "phase", .{ .string = "profile" });
        try sched_item.put(allocator, "engine", .{ .string = engine });
        try sched_item.put(allocator, "engine_ordinal", .{ .integer = ordinal });
        try sched_item.put(allocator, "started_at", .{ .string = started_at });
        try sched_item.put(allocator, "completed_at", .{ .string = completed_at });
        try sched_item.put(allocator, "iterations", .{ .integer = bench_coremark.expected_iterations });
        try sched_item.put(allocator, "iterations_per_second", .{ .float = value });
        try sched_item.put(allocator, "perf_file", .{ .string = try allocator.dupe(u8, std.fs.path.basename(perf_path)) });
        try profile_schedule.append(allocator, .{ .object = sched_item });
    }

    var ranking_reports: std.ArrayList(Value) = .empty;
    var wamr_capture_validations: std.ArrayList(Value) = .empty;
    for (wamr_perfs.items, 1..) |wamr_perf, ordinal| {
        const ranking_json = try std.fs.path.join(allocator, &.{ out_dir, try std.fmt.allocPrint(allocator, "wamr-attribution-{d}.json", .{ordinal}) });
        _ = try recorder.run(
            allocator,
            &.{
                "zig-aot-jit-attr",
                helper_path,
                "--perf",
                wamr_perf,
                "--cwasm",
                cwasm,
                "--arch",
                "aarch64",
                "--top",
                try std.fmt.allocPrint(allocator, "{d}", .{cwasm_info.func_offsets.len}),
                "--min-samples",
                try std.fmt.allocPrint(allocator, "{d}", .{opts.min_samples}),
                "--min-attribution-pct",
                try std.fmt.allocPrint(allocator, "{d:.1}", .{MIN_ATTRIBUTION_COVERAGE_PCT}),
                "--authoritative",
                "--json-out",
                ranking_json,
            },
            try std.fmt.allocPrint(allocator, "wamr-attribution-{d}.log", .{ordinal}),
            repo,
            null,
        );
        const r_bytes = std.Io.Dir.cwd().readFileAlloc(io, ranking_json, allocator, .limited(64 * 1024 * 1024)) catch {
            return fail("cannot read {s}", .{ranking_json});
        };
        const r_val = json.parseFromSliceLeaky(Value, allocator, r_bytes, .{}) catch {
            return fail("invalid JSON in {s}", .{ranking_json});
        };
        try wamr_capture_validations.append(allocator, try validateWamrCapture(allocator, r_val, opts.min_samples, MIN_ATTRIBUTION_COVERAGE_PCT));
        try ranking_reports.append(allocator, r_val);
    }

    const wamr_attribution = try aggregateWamrRankings(allocator, ranking_reports.items);
    const all_top = wamr_attribution.object.get("top_functions").?.array.items;
    if (all_top.len < opts.classify) {
        return fail("WAMR attribution produced only {d} hot functions", .{all_top.len});
    }
    const top_functions = all_top[0..opts.classify];

    var classified_wamr: std.AutoHashMap(usize, Value) = .init(allocator);
    for (top_functions) |item_v| {
        const local_func: usize = @intCast(jsonInt(item_v.object.get("local_func")).?);
        var fn_reports: std.ArrayList(Value) = .empty;
        for (wamr_perfs.items, 1..) |wamr_perf, ordinal| {
            const fn_json = try std.fs.path.join(allocator, &.{ out_dir, try std.fmt.allocPrint(allocator, "wamr-func-{d}-{d}.json", .{ local_func, ordinal }) });
            _ = try recorder.run(
                allocator,
                &.{
                    "zig-aot-jit-attr",
                    helper_path,
                    "--perf",
                    wamr_perf,
                    "--cwasm",
                    cwasm,
                    "--arch",
                    "aarch64",
                    "--func",
                    try std.fmt.allocPrint(allocator, "{d}", .{local_func}),
                    "--top",
                    try std.fmt.allocPrint(allocator, "{d}", .{opts.top}),
                    "--min-samples",
                    try std.fmt.allocPrint(allocator, "{d}", .{opts.min_samples}),
                    "--min-attribution-pct",
                    try std.fmt.allocPrint(allocator, "{d:.1}", .{MIN_ATTRIBUTION_COVERAGE_PCT}),
                    "--authoritative",
                    "--json-out",
                    fn_json,
                },
                try std.fmt.allocPrint(allocator, "wamr-func-{d}-{d}.log", .{ local_func, ordinal }),
                repo,
                null,
            );
            const fb = std.Io.Dir.cwd().readFileAlloc(io, fn_json, allocator, .limited(64 * 1024 * 1024)) catch {
                return fail("cannot read {s}", .{fn_json});
            };
            const fv = json.parseFromSliceLeaky(Value, allocator, fb, .{}) catch {
                return fail("invalid JSON in {s}", .{fn_json});
            };
            try fn_reports.append(allocator, fv);
        }
        const bounds = aot_jit_attr.functionBounds(cwasm_info, @intCast(local_func)) catch {
            return fail("functionBounds failed: {s}", .{aot_jit_attr.lastError()});
        };
        const agg_fn = try aggregateWamrFunction(
            allocator,
            fn_reports.items,
            jsonInt(wamr_attribution.object.get("total_samples")).?,
            @intCast(bounds.start),
        );
        try classified_wamr.put(local_func, agg_fn);
    }

    var wamr_capture_counts: std.ArrayList(CaptureCountsEntry) = .empty;
    for (wamr_perfs.items, ranking_reports.items) |wamr_perf, ranking_report| {
        const ac = aot_jit_attr.addrCounts(allocator, io, wamr_perf) catch {
            return fail("addrCounts failed: {s}", .{aot_jit_attr.lastError()});
        };
        if (@as(i64, @intCast(ac.total)) != jsonInt(ranking_report.object.get("total_samples")).?) {
            return fail("WAMR perf sample totals changed while building ALU provenance", .{});
        }
        try wamr_capture_counts.append(allocator, .{
            .counts = ac.counts,
            .text_base = @intCast(jsonInt(ranking_report.object.get("text_base")).?),
        });
    }

    var frame_attribution_opt: ?Value = null;
    if (frame_metadata) |fm_path| {
        const ff = opts.frame_func.?;
        const spill_val = spill_metrics.get(ff) orelse .null;
        var fa = try analyzeWamrFrameProvenance(
            allocator,
            io,
            cwasm_info,
            fm_path,
            @intCast(ff),
            wamr_capture_counts.items,
            ranking_reports.items,
            jsonInt(wamr_attribution.object.get("total_samples")).?,
            spill_val,
            out_dir,
            .{},
        );
        const wasm_idx = ff + wasm_identity.imported_function_count;
        try fa.object.put(allocator, "wasm_function_index", .{ .integer = @intCast(wasm_idx) });
        if (wasm_identity.function_names.get(wasm_idx)) |fn_name| {
            try fa.object.put(allocator, "name", .{ .string = try allocator.dupe(u8, fn_name) });
        } else {
            try fa.object.put(allocator, "name", .null);
        }
        try fa.object.put(allocator, "compiler_sha256", .{ .string = profile_wamr_identity.object.get("compiler").?.object.get("sha256").?.string });
        try fa.object.put(allocator, "cwasm_sha256", .{ .string = profile_wamr_identity.object.get("module").?.object.get("sha256").?.string });
        if (!std.mem.eql(u8, fa.object.get("metadata").?.object.get("sha256").?.string, frame_metadata_sha.?)) {
            return fail("frame metadata changed during capture", .{});
        }
        frame_attribution_opt = fa;
    }

    const parsed_wasmtime = try aggregateWasmtimeSamples(allocator, wasmtime_captures.items);
    try validateWasmtimeMapping(&parsed_wasmtime, &wasm_identity);
    if (parsed_wasmtime.total_samples < opts.min_samples) {
        return fail("Wasmtime perf data has only {d} samples", .{parsed_wasmtime.total_samples});
    }
    var wasmtime_attributed: i64 = 0;
    var wt_fn_it = parsed_wasmtime.functions.iterator();
    while (wt_fn_it.next()) |kv| {
        wasmtime_attributed += kv.value_ptr.samples;
    }

    const wamr_total_samples = jsonInt(wamr_attribution.object.get("total_samples")).?;
    const wamr_attributed_samples = jsonInt(wamr_attribution.object.get("attributed_samples")).?;

    var matched: std.ArrayList(Value) = .empty;
    for (top_functions) |top_v| {
        const local_func: usize = @intCast(jsonInt(top_v.object.get("local_func")).?);
        const wasm_index = local_func + wasm_identity.imported_function_count;
        const name = wasm_identity.function_names.get(wasm_index) orelse try std.fmt.allocPrint(allocator, "func_{d}", .{wasm_index});
        const wasmtime_entry = parsed_wasmtime.functions.getPtr(wasm_index) orelse {
            return fail("Wasmtime has no samples for WAMR local_func={d}, wasm function {d} ({s})", .{ local_func, wasm_index, name });
        };
        if (wasmtime_entry.samples <= 0) {
            return fail("Wasmtime has no samples for WAMR local_func={d}, wasm function {d} ({s})", .{ local_func, wasm_index, name });
        }
        var sum_offsets: i64 = 0;
        var oit = wasmtime_entry.offsets.iterator();
        while (oit.next()) |okv| sum_offsets += okv.value_ptr.*;
        if (sum_offsets != wasmtime_entry.samples) {
            return fail("Wasmtime function {d} has {d} symbol samples but only {d} symbol offsets", .{ wasm_index, wasmtime_entry.samples, sum_offsets });
        }

        const wamr_function = classified_wamr.get(local_func).?.object;
        const wasmtime_instruction = try classifyWasmtimeFunction(
            allocator,
            objdump,
            wasm_index,
            &wasmtime_entry.offsets,
            parsed_wasmtime.total_samples,
        );
        const wamr_classes = wamr_function.get("classes").?;
        const wasmtime_classes = wasmtime_instruction.object.get("classes").?;

        var class_groups: json.ObjectMap = .empty;
        for (CLASS_GROUPS) |cg| {
            const wamr_s = classSamples(wamr_classes, cg.raw_classes);
            const wt_s = classSamples(wasmtime_classes, cg.raw_classes);
            const wamr_pct = 100.0 * @as(f64, @floatFromInt(wamr_s)) / @as(f64, @floatFromInt(wamr_total_samples));
            const wt_pct = 100.0 * @as(f64, @floatFromInt(wt_s)) / @as(f64, @floatFromInt(parsed_wasmtime.total_samples));
            var cg_obj: json.ObjectMap = .empty;
            try cg_obj.put(allocator, "wamr_samples", .{ .integer = wamr_s });
            try cg_obj.put(allocator, "wasmtime_samples", .{ .integer = wt_s });
            try cg_obj.put(allocator, "wamr_percent_of_run", .{ .float = wamr_pct });
            try cg_obj.put(allocator, "wasmtime_percent_of_run", .{ .float = wt_pct });
            try cg_obj.put(allocator, "delta_percentage_points", .{ .float = wamr_pct - wt_pct });
            try class_groups.put(allocator, cg.group, .{ .object = cg_obj });
        }

        var static_frame_loads: i64 = 0;
        var static_frame_stores: i64 = 0;
        var wc_it = wamr_classes.object.iterator();
        while (wc_it.next()) |kv| {
            const k = kv.key_ptr.*;
            const st_insts = if (kv.value_ptr.* == .object) (jsonInt(kv.value_ptr.object.get("static_instructions")) orelse 0) else 0;
            if (std.mem.eql(u8, k, "frame_load_unattributed") or std.mem.eql(u8, k, "unknown_frame_load")) {
                static_frame_loads += st_insts;
            } else if (std.mem.eql(u8, k, "frame_store_unattributed") or std.mem.eql(u8, k, "unknown_frame_store")) {
                static_frame_stores += st_insts;
            }
        }
        const frame_samples = class_groups.get("frame_traffic").?.object.get("wamr_samples").?.integer;
        const wamr_alu_prov = try analyzeWamrAluProvenance(
            allocator,
            io,
            cwasm_info,
            @intCast(local_func),
            wamr_capture_counts.items,
            wamr_total_samples,
            out_dir,
            wamr_attributed_samples,
        );
        const wt_alu_prov = try analyzeWasmtimeAluProvenance(
            allocator,
            objdump,
            wasm_index,
            &wasmtime_entry.offsets,
            parsed_wasmtime.total_samples,
            wasmtime_attributed,
        );
        const narrow_alu = try assembleAluProvenance(
            allocator,
            wamr_alu_prov,
            wt_alu_prov,
            class_groups.get("all_alu").?.object.get("wamr_samples").?.integer,
            class_groups.get("all_alu").?.object.get("wasmtime_samples").?.integer,
        );

        var spill_obj: json.ObjectMap = .empty;
        const sp_opt = spill_metrics.get(local_func);
        for ([_][]const u8{ "spill_ld", "spill_st", "spilled_vregs", "slots" }) |sk| {
            const sv = if (sp_opt) |sp| (if (sp == .object) (jsonInt(sp.object.get(sk)) orelse 0) else 0) else 0;
            try spill_obj.put(allocator, sk, .{ .integer = sv });
        }

        var fcc_obj: json.ObjectMap = .empty;
        try fcc_obj.put(allocator, "static_frame_loads", .{ .integer = static_frame_loads });
        try fcc_obj.put(allocator, "static_frame_stores", .{ .integer = static_frame_stores });
        try fcc_obj.put(allocator, "samples", .{ .integer = frame_samples });
        try fcc_obj.put(allocator, "percent_of_run", .{ .float = 100.0 * @as(f64, @floatFromInt(frame_samples)) / @as(f64, @floatFromInt(wamr_total_samples)) });

        var wamr_m_obj: json.ObjectMap = .empty;
        try wamr_m_obj.put(allocator, "samples", wamr_function.get("samples").?);
        try wamr_m_obj.put(allocator, "percent_of_run", wamr_function.get("percent_of_run").?);
        try wamr_m_obj.put(allocator, "classes", wamr_classes);
        try wamr_m_obj.put(allocator, "hottest_instructions", wamr_function.get("hottest_instructions").?);
        try wamr_m_obj.put(allocator, "frame_cross_check", .{ .object = fcc_obj });

        var wt_m_obj = try cloneJsonValue(allocator, wasmtime_instruction);
        try wt_m_obj.object.put(allocator, "samples", .{ .integer = wasmtime_entry.samples });
        try wt_m_obj.object.put(allocator, "percent_of_run", .{ .float = 100.0 * @as(f64, @floatFromInt(wasmtime_entry.samples)) / @as(f64, @floatFromInt(parsed_wasmtime.total_samples)) });

        var m_item: json.ObjectMap = .empty;
        try m_item.put(allocator, "local_func", .{ .integer = @intCast(local_func) });
        try m_item.put(allocator, "wasm_function_index", .{ .integer = @intCast(wasm_index) });
        try m_item.put(allocator, "name", .{ .string = try allocator.dupe(u8, name) });
        try m_item.put(allocator, "spill_metric", .{ .object = spill_obj });
        try m_item.put(allocator, "spill_metric_basis", .{ .string = if (opts.frame_func != null and opts.frame_func.? == local_func) "emitter-traced" else "pre-emission IR estimate" });
        try m_item.put(allocator, "wamr", .{ .object = wamr_m_obj });
        try m_item.put(allocator, "wasmtime", wt_m_obj);
        try m_item.put(allocator, "class_groups", .{ .object = class_groups });
        try m_item.put(allocator, "alu_provenance", narrow_alu);
        try matched.append(allocator, .{ .object = m_item });
    }

    var perf_artifacts: std.ArrayList(Value) = .empty;
    for (wamr_perfs.items) |p| try perf_artifacts.append(allocator, try gzipIfSmall(allocator, io, p, opts.max_perf_bytes));
    for (wasmtime_perfs.items) |p| try perf_artifacts.append(allocator, try gzipIfSmall(allocator, io, p, opts.max_perf_bytes));
    for (jitdumps.items) |p| try perf_artifacts.append(allocator, try gzipIfSmall(allocator, io, p, opts.max_perf_bytes));

    var target_alu: ?Value = null;
    for (matched.items) |item_v| {
        const item = item_v.object;
        if ((jsonInt(item.get("local_func")) orelse -1) == 3 and
            (jsonInt(item.get("wasm_function_index")) orelse -1) == @as(i64, @intCast(3 + wasm_identity.imported_function_count)))
        {
            target_alu = item.get("alu_provenance");
            break;
        }
    }

    try validateAnalysisSourcesUnchanged(allocator, io, repo, analysis_sources);
    analysis_sources = try captureAnalysisSources(allocator, io, repo, checkout_commit, &REQUIRED_ANALYSIS_SOURCES, null);

    var bench_section = try cloneJsonValue(allocator, benchmark_identity);
    try bench_section.object.put(allocator, "report_path", .{ .string = benchmark_report_resolved });
    try bench_section.object.put(allocator, "report_sha256", .{ .string = benchmark_report_sha });
    try bench_section.object.put(allocator, "schema_version", benchmark_report.object.get("schema_version").?);
    try bench_section.object.put(allocator, "kind", benchmark_report.object.get("kind").?);
    var handoff_obj: json.ObjectMap = .empty;
    try handoff_obj.put(allocator, "directory", .{ .string = benchmark_artifacts_resolved });
    try handoff_obj.put(allocator, "manifest_sha256", .{ .string = handoff_manifest_sha });
    try bench_section.object.put(allocator, "artifact_handoff", .{ .object = handoff_obj });

    const script_full = try std.fs.path.join(allocator, &.{ repo, "scripts/profile_coremark_aarch64.zig" });
    const script_sha = bench_coremark.sha256File(allocator, io, script_full) catch return fail("cannot hash script", .{});
    const analysis_mod_full = try std.fs.path.join(allocator, &.{ repo, "scripts/aarch64_instruction_provenance.zig" });
    const analysis_mod_sha = bench_coremark.sha256File(allocator, io, analysis_mod_full) catch return fail("cannot hash analysis module", .{});

    var analysis_mod_obj: json.ObjectMap = .empty;
    try analysis_mod_obj.put(allocator, "path", .{ .string = "scripts/aarch64_instruction_provenance.zig" });
    try analysis_mod_obj.put(allocator, "sha256", .{ .string = analysis_mod_sha });

    var prov_obj: json.ObjectMap = .empty;
    try prov_obj.put(allocator, "producer_source_sha", .{ .string = checkout_commit });
    try prov_obj.put(allocator, "script_path", .{ .string = "scripts/profile_coremark_aarch64.zig" });
    try prov_obj.put(allocator, "script_sha256", .{ .string = script_sha });
    try prov_obj.put(allocator, "analysis_module", .{ .object = analysis_mod_obj });
    try prov_obj.put(allocator, "analysis_sources", analysis_sources);
    try prov_obj.put(allocator, "generated_at", .{ .string = bench_coremark.formatUtcIsoTimestamp(allocator, io) catch unreachable });
    try prov_obj.put(allocator, "execution", current_execution);

    var guest_args_arr: json.Array = .init(allocator);
    for (guest_args) |ga| try guest_args_arr.append(.{ .string = ga });

    var allowed_cpus_arr: json.Array = .init(allocator);
    for (affinity.allowed_cpus) |c| try allowed_cpus_arr.append(.{ .integer = @intCast(c) });
    var aff_obj: json.ObjectMap = .empty;
    try aff_obj.put(allocator, "allowed_cpus", .{ .array = allowed_cpus_arr });
    try aff_obj.put(allocator, "selected_cpu", .{ .integer = @intCast(affinity.selected_cpu) });
    try aff_obj.put(allocator, "taskset", .{ .string = affinity.taskset });
    try aff_obj.put(allocator, "verified", .{ .bool = true });

    var cw_obj: json.ObjectMap = .empty;
    try cw_obj.put(allocator, "all_alu", .{ .string = ALL_ALU_WORDING });
    try cw_obj.put(allocator, "narrow_alu_provenance", .{ .string = NARROW_ALU_WORDING });

    var wt_cap_vals: json.Array = .init(allocator);
    for (wasmtime_captures.items) |*cap| {
        try wt_cap_vals.append(try validateWasmtimeCapture(allocator, cap, opts.min_samples, MIN_ATTRIBUTION_COVERAGE_PCT));
    }

    var fix_obj: json.ObjectMap = .empty;
    try fix_obj.put(allocator, "path", .{ .string = fixture });
    try fix_obj.put(allocator, "sha256", .{ .string = fixture_sha });

    var wasm_obj: json.ObjectMap = .empty;
    try wasm_obj.put(allocator, "imported_function_count", .{ .integer = @intCast(wasm_identity.imported_function_count) });
    try wasm_obj.put(allocator, "local_function_count", .{ .integer = @intCast(wasm_identity.local_function_count) });

    var host_obj: json.ObjectMap = .empty;
    try host_obj.put(allocator, "architecture", .{ .string = host_identity.arch });
    try host_obj.put(allocator, "cpu_count", .{ .integer = @intCast(host_identity.cpu_count) });
    try host_obj.put(allocator, "cpu_model", .{ .string = host_identity.cpu_model });
    try host_obj.put(allocator, "runner_name", .{ .string = host_identity.runner_name });
    try host_obj.put(allocator, "fingerprint", .{ .string = host_identity.fingerprint(allocator) catch unreachable });
    try host_obj.put(allocator, "kernel", .{ .string = kernel });
    try host_obj.put(allocator, "uname", .{ .string = uname });

    const paranoid_eff_raw = std.Io.Dir.cwd().readFileAlloc(io, "/proc/sys/kernel/perf_event_paranoid", allocator, .limited(1024)) catch "unknown";
    const paranoid_eff = std.mem.trim(u8, paranoid_eff_raw, " \t\r\n");
    var perf_obj: json.ObjectMap = .empty;
    try perf_obj.put(allocator, "version", .{ .string = perf_version });
    try perf_obj.put(allocator, "package", .{ .string = environ_map.get("COREMARK_PERF_PACKAGE") orelse "unknown" });
    try perf_obj.put(allocator, "package_version", .{ .string = environ_map.get("COREMARK_PERF_PACKAGE_VERSION") orelse "unknown" });
    try perf_obj.put(allocator, "paranoid_initial", .{ .string = environ_map.get("COREMARK_PERF_PARANOID_INITIAL") orelse "unknown" });
    try perf_obj.put(allocator, "paranoid_effective", .{ .string = paranoid_eff });
    try perf_obj.put(allocator, "event", .{ .string = "cycles:u" });
    try perf_obj.put(allocator, "frequency", .{ .integer = @intCast(opts.frequency) });
    try perf_obj.put(allocator, "sampling_permitted", .{ .bool = true });

    var wamr_meta_obj: json.ObjectMap = .empty;
    try wamr_meta_obj.put(allocator, "benchmark_role", .{ .string = opts.benchmark_role });
    try wamr_meta_obj.put(allocator, "commit", .{ .string = commit });
    try wamr_meta_obj.put(allocator, "version", .{ .string = wamr_version });
    try wamr_meta_obj.put(allocator, "wamrc_version", .{ .string = wamrc_version });
    try wamr_meta_obj.put(allocator, "cwasm_aot_version", .{ .integer = cwasm_info.version });
    try wamr_meta_obj.put(allocator, "runtime_sha256", .{ .string = profile_wamr_identity.object.get("runtime").?.object.get("sha256").?.string });
    try wamr_meta_obj.put(allocator, "compiler_sha256", .{ .string = profile_wamr_identity.object.get("compiler").?.object.get("sha256").?.string });
    try wamr_meta_obj.put(allocator, "cwasm_sha256", .{ .string = profile_wamr_identity.object.get("module").?.object.get("sha256").?.string });

    const wt_cwasm_sha = bench_coremark.sha256File(allocator, io, wasmtime_cwasm) catch return fail("cannot hash wasmtime cwasm", .{});
    var wt_meta_obj: json.ObjectMap = .empty;
    try wt_meta_obj.put(allocator, "version", .{ .string = wasmtime_version });
    try wt_meta_obj.put(allocator, "sha256", .{ .string = profile_wasmtime_identity.object.get("runtime").?.object.get("sha256").?.string });
    try wt_meta_obj.put(allocator, "profile_strategy", .{ .string = "jitdump" });
    try wt_meta_obj.put(
        allocator,
        "function_mapping",
        .{ .string = "Wasmtime v44 perf inject emitted name-only symbols in jitted-<pid>-<defined-func-index>.so. Each symbol name was resolved uniquely through the wasm name section and the DSO suffix was required to equal wasm_index - import_count." },
    );
    try wt_meta_obj.put(allocator, "cwasm_sha256", .{ .string = wt_cwasm_sha });

    var wamr_vals_arr: json.Array = .init(allocator);
    var wamr_sum: f64 = 0.0;
    for (wamr_values.items) |v| {
        try wamr_vals_arr.append(.{ .float = v });
        wamr_sum += v;
    }
    var wt_vals_arr: json.Array = .init(allocator);
    var wt_sum: f64 = 0.0;
    for (wasmtime_values.items) |v| {
        try wt_vals_arr.append(.{ .float = v });
        wt_sum += v;
    }

    var wamr_wall: f64 = 0.0;
    var wt_wall: f64 = 0.0;
    for (recorder.commands.items) |cmd_v| {
        const log_n = jsonStringOrEmpty(cmd_v.object.get("log"));
        const el = jsonNumber(cmd_v.object.get("elapsed_seconds")) orelse 0.0;
        if (std.mem.startsWith(u8, log_n, "wamr-run-")) {
            wamr_wall += el;
        } else if (std.mem.startsWith(u8, log_n, "wasmtime-run-")) {
            wt_wall += el;
        }
    }

    var wamr_eng_obj: json.ObjectMap = .empty;
    try wamr_eng_obj.put(allocator, "total_samples", wamr_attribution.object.get("total_samples").?);
    try wamr_eng_obj.put(allocator, "attributed_samples", wamr_attribution.object.get("attributed_samples").?);
    try wamr_eng_obj.put(allocator, "attribution_coverage_pct", wamr_attribution.object.get("attribution_coverage_pct").?);
    try wamr_eng_obj.put(allocator, "iterations_per_second_samples", .{ .array = wamr_vals_arr });
    try wamr_eng_obj.put(allocator, "iterations_per_second", .{ .float = wamr_sum / @as(f64, @floatFromInt(wamr_values.items.len)) });
    try wamr_eng_obj.put(allocator, "wall_seconds", .{ .float = wamr_wall });
    try wamr_eng_obj.put(allocator, "top_functions", wamr_attribution.object.get("top_functions").?);

    const WtTopSort = struct {
        idx: usize,
        samples: i64,
        name: ?[]const u8,
    };
    var wt_top_sort: std.ArrayList(WtTopSort) = .empty;
    var wt_it2 = parsed_wasmtime.functions.iterator();
    while (wt_it2.next()) |kv| {
        var nm_it = kv.value_ptr.names.keyIterator();
        const nm = if (nm_it.next()) |k| k.* else wasm_identity.function_names.get(kv.key_ptr.*);
        try wt_top_sort.append(allocator, .{ .idx = kv.key_ptr.*, .samples = kv.value_ptr.samples, .name = nm });
    }
    std.mem.sort(WtTopSort, wt_top_sort.items, {}, struct {
        fn lessThan(_: void, a: WtTopSort, b: WtTopSort) bool {
            if (a.samples != b.samples) return a.samples > b.samples;
            return a.idx < b.idx;
        }
    }.lessThan);

    var wt_top_arr: json.Array = .init(allocator);
    const wt_top_limit = @min(opts.top, wt_top_sort.items.len);
    for (wt_top_sort.items[0..wt_top_limit]) |item| {
        var t_obj: json.ObjectMap = .empty;
        try t_obj.put(allocator, "wasm_function_index", .{ .integer = @intCast(item.idx) });
        if (item.name) |nm| {
            try t_obj.put(allocator, "name", .{ .string = try allocator.dupe(u8, nm) });
        } else {
            try t_obj.put(allocator, "name", .null);
        }
        try t_obj.put(allocator, "samples", .{ .integer = item.samples });
        try t_obj.put(allocator, "percent_of_run", .{ .float = 100.0 * @as(f64, @floatFromInt(item.samples)) / @as(f64, @floatFromInt(parsed_wasmtime.total_samples)) });
        try wt_top_arr.append(.{ .object = t_obj });
    }

    var wt_eng_obj: json.ObjectMap = .empty;
    try wt_eng_obj.put(allocator, "total_samples", .{ .integer = parsed_wasmtime.total_samples });
    try wt_eng_obj.put(allocator, "attributed_samples", .{ .integer = wasmtime_attributed });
    try wt_eng_obj.put(allocator, "attribution_coverage_pct", .{ .float = 100.0 * @as(f64, @floatFromInt(wasmtime_attributed)) / @as(f64, @floatFromInt(parsed_wasmtime.total_samples)) });
    try wt_eng_obj.put(allocator, "iterations_per_second_samples", .{ .array = wt_vals_arr });
    try wt_eng_obj.put(allocator, "iterations_per_second", .{ .float = wt_sum / @as(f64, @floatFromInt(wasmtime_values.items.len)) });
    try wt_eng_obj.put(allocator, "wall_seconds", .{ .float = wt_wall });
    try wt_eng_obj.put(allocator, "top_functions", .{ .array = wt_top_arr });

    var engines_obj: json.ObjectMap = .empty;
    try engines_obj.put(allocator, "wamr", .{ .object = wamr_eng_obj });
    try engines_obj.put(allocator, "wasmtime", .{ .object = wt_eng_obj });

    var alu_top_obj: json.ObjectMap = .empty;
    try alu_top_obj.put(allocator, "schema_version", .{ .integer = provenance.SCHEMA_VERSION });
    try alu_top_obj.put(allocator, "kind", .{ .string = provenance.ANALYSIS_KIND });
    try alu_top_obj.put(allocator, "target_local_func", .{ .integer = 3 });
    try alu_top_obj.put(allocator, "target_wasm_function_index", .{ .integer = @intCast(3 + wasm_identity.imported_function_count) });
    if (target_alu) |ta| {
        try alu_top_obj.put(allocator, "gate", try cloneJsonValue(allocator, ta.object.get("gate").?));
    } else {
        var blocked_gate: json.ObjectMap = .empty;
        try blocked_gate.put(allocator, "status", .{ .string = "blocked" });
        try blocked_gate.put(allocator, "optimization_authorized", .{ .bool = false });
        try blocked_gate.put(allocator, "reason", .{ .string = "core_bench_list was not among the classified functions" });
        try alu_top_obj.put(allocator, "gate", .{ .object = blocked_gate });
    }

    var retained_analysis_obj: json.ObjectMap = .empty;
    try retained_analysis_obj.put(allocator, "wamr_cwasm", retained_wamr_binary);

    var caveats_arr: json.Array = .init(allocator);
    try caveats_arr.append(.{ .string = "Self samples only; WAMR generated code has no unwind CFI." });
    try caveats_arr.append(.{ .string = "Unselected AArch64 spill metrics are pre-emission estimates; selected frame sidecars reconcile emitted component counts." });
    try caveats_arr.append(.{ .string = "Wasmtime wasm symbols use full function indices including imports." });
    try caveats_arr.append(.{ .string = "Two captures per engine were collected in ABBA order after ABBA warmups." });
    try caveats_arr.append(.{ .string = ALL_ALU_WORDING ++ "; the all-ALU differential is not address/check headroom." });
    try caveats_arr.append(.{ .string = "Narrow ALU provenance uses an identical complete architecture-only gating universe in both engines; common unknown/mixed, instruction-unresolved, and globally unattributed reference samples cannot clear the optimization gate. Structural guards are diagnostic only." });

    var sched_arr: json.Array = .init(allocator);
    try sched_arr.appendSlice(profile_schedule.items);
    var wamr_cap_arr: json.Array = .init(allocator);
    try wamr_cap_arr.appendSlice(wamr_capture_validations.items);
    var matched_arr: json.Array = .init(allocator);
    try matched_arr.appendSlice(matched.items);
    var cmds_arr: json.Array = .init(allocator);
    try cmds_arr.appendSlice(recorder.commands.items);
    var perf_art_arr: json.Array = .init(allocator);
    try perf_art_arr.appendSlice(perf_artifacts.items);

    var report_obj: json.ObjectMap = .empty;
    try report_obj.put(allocator, "schema_version", .{ .integer = REPORT_SCHEMA_VERSION });
    try report_obj.put(allocator, "kind", .{ .string = REPORT_KIND });
    try report_obj.put(allocator, "architecture", .{ .string = "aarch64" });
    try report_obj.put(allocator, "benchmark", bench_section);
    try report_obj.put(allocator, "provenance", .{ .object = prov_obj });
    try report_obj.put(allocator, "guest_args", .{ .array = guest_args_arr });
    try report_obj.put(allocator, "expected_iterations", .{ .integer = bench_coremark.expected_iterations });
    try report_obj.put(allocator, "affinity", .{ .object = aff_obj });
    try report_obj.put(allocator, "profile_schedule", .{ .array = sched_arr });
    try report_obj.put(allocator, "classifier_wording", .{ .object = cw_obj });
    try report_obj.put(allocator, "minimum_attribution_coverage_pct", .{ .float = MIN_ATTRIBUTION_COVERAGE_PCT });
    try report_obj.put(allocator, "wamr_captures", .{ .array = wamr_cap_arr });
    try report_obj.put(allocator, "wasmtime_captures", .{ .array = wt_cap_vals });
    try report_obj.put(allocator, "fixture", .{ .object = fix_obj });
    try report_obj.put(allocator, "wasm", .{ .object = wasm_obj });
    try report_obj.put(allocator, "host", .{ .object = host_obj });
    try report_obj.put(allocator, "perf", .{ .object = perf_obj });
    try report_obj.put(allocator, "wamr", .{ .object = wamr_meta_obj });
    try report_obj.put(allocator, "wasmtime", .{ .object = wt_meta_obj });
    try report_obj.put(allocator, "engines", .{ .object = engines_obj });
    try report_obj.put(allocator, "matched_functions", .{ .array = matched_arr });
    try report_obj.put(allocator, "alu_provenance", .{ .object = alu_top_obj });
    try report_obj.put(allocator, "commands", .{ .array = cmds_arr });
    try report_obj.put(allocator, "retained_perf_artifacts", .{ .array = perf_art_arr });
    try report_obj.put(allocator, "retained_analysis_artifacts", .{ .object = retained_analysis_obj });
    try report_obj.put(allocator, "caveats", .{ .array = caveats_arr });

    bench_coremark.validateSameHost(allocator, io, environ_map, host_identity) catch |err| {
        return fail("validateSameHost failed: {s}", .{@errorName(err)});
    };
    try validateAnalysisSourcesUnchanged(allocator, io, repo, analysis_sources);
    if (frame_attribution_opt) |fa| {
        const cur_sha = bench_coremark.sha256File(allocator, io, frame_metadata.?) catch return fail("cannot hash frame metadata", .{});
        if (!std.mem.eql(u8, cur_sha, frame_metadata_sha.?)) {
            return fail("frame metadata changed during capture", .{});
        }
        try report_obj.put(allocator, "frame_attribution", fa);
    }

    const final_report: Value = .{ .object = report_obj };
    try validateReport(allocator, final_report);
    return final_report;
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = init.io;
    const environ_map = init.environ_map;

    var args_list: std.ArrayList([]const u8) = .empty;
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();
    _ = args_it.next();
    while (args_it.next()) |arg| {
        try args_list.append(allocator, arg);
    }

    var validate_report_path: ?[]const u8 = null;
    var report_status_path: ?[]const u8 = null;
    var render_md_from: ?[]const u8 = null;
    var md_out_path: ?[]const u8 = null;

    var repo_arg: ?[]const u8 = null;
    var out_dir_arg: ?[]const u8 = null;
    var benchmark_report_arg: ?[]const u8 = null;
    var wamr_ref_arg: ?[]const u8 = null;
    var benchmark_role_arg: []const u8 = "wamr-target";
    var benchmark_artifacts_arg: ?[]const u8 = null;
    var execution_id_arg: ?[]const u8 = null;
    var wasmtime_cache_arg: ?[]const u8 = null;
    var frequency_arg: usize = 999;
    var min_samples_arg: i64 = DEFAULT_MIN_SAMPLES;
    var top_arg: usize = DEFAULT_TOP_FUNCTIONS;
    var classify_arg: usize = DEFAULT_CLASSIFY_FUNCTIONS;
    var frame_func_arg: ?usize = null;
    var max_perf_bytes_arg: usize = DEFAULT_MAX_PERF_BYTES;

    var i: usize = 0;
    while (i < args_list.items.len) : (i += 1) {
        const arg = args_list.items[i];
        if (std.mem.eql(u8, arg, "--validate-report")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            validate_report_path = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--report-status")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            report_status_path = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--render-markdown")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            render_md_from = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--markdown-out")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            md_out_path = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--repo")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            repo_arg = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--out-dir")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            out_dir_arg = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--benchmark-report")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            benchmark_report_arg = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--wamr-ref")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            wamr_ref_arg = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--benchmark-role")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            benchmark_role_arg = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--benchmark-artifacts")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            benchmark_artifacts_arg = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--execution-id")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            execution_id_arg = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--wasmtime-cache")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            wasmtime_cache_arg = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--frequency")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            frequency_arg = try std.fmt.parseInt(usize, args_list.items[i], 10);
        } else if (std.mem.eql(u8, arg, "--min-samples")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            min_samples_arg = try std.fmt.parseInt(i64, args_list.items[i], 10);
        } else if (std.mem.eql(u8, arg, "--top")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            top_arg = try std.fmt.parseInt(usize, args_list.items[i], 10);
        } else if (std.mem.eql(u8, arg, "--classify")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            classify_arg = try std.fmt.parseInt(usize, args_list.items[i], 10);
        } else if (std.mem.eql(u8, arg, "--frame-func")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            frame_func_arg = try std.fmt.parseInt(usize, args_list.items[i], 10);
        } else if (std.mem.eql(u8, arg, "--max-perf-bytes")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            max_perf_bytes_arg = try std.fmt.parseInt(usize, args_list.items[i], 10);
        }
    }

    if (report_status_path) |p| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, allocator, .limited(64 * 1024 * 1024));
        const parsed = try json.parseFromSlice(Value, allocator, bytes, .{});
        const status = profileReportStatus(allocator, parsed.value) catch {
            std.debug.print("error: {s}\n", .{lastErrorMessage()});
            std.process.exit(1);
        };
        var aw: std.Io.Writer.Allocating = .init(allocator);
        try json.Stringify.value(status, .{ .whitespace = .indent_2 }, &aw.writer);
        try aw.writer.writeAll("\n");
        var out_buf: [4096]u8 = undefined;
        var out_w = std.Io.File.stdout().writer(io, &out_buf);
        try out_w.interface.writeAll(aw.written());
        try out_w.interface.flush();
        return;
    }

    if (validate_report_path) |p| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, allocator, .limited(64 * 1024 * 1024));
        const parsed = try json.parseFromSlice(Value, allocator, bytes, .{});
        validateReport(allocator, parsed.value) catch {
            std.debug.print("error: {s}\n", .{lastErrorMessage()});
            std.process.exit(1);
        };
        if (md_out_path) |mp| {
            const md = try renderMarkdown(allocator, parsed.value);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = mp, .data = md });
        }
        return;
    }

    if (render_md_from) |p| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, allocator, .limited(64 * 1024 * 1024));
        const parsed = try json.parseFromSlice(Value, allocator, bytes, .{});
        const md = renderMarkdown(allocator, parsed.value) catch {
            std.debug.print("error: {s}\n", .{lastErrorMessage()});
            std.process.exit(1);
        };
        if (md_out_path) |mp| {
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = mp, .data = md });
        } else {
            var out_buf: [4096]u8 = undefined;
            var out_w = std.Io.File.stdout().writer(io, &out_buf);
            try out_w.interface.writeAll(md);
            try out_w.interface.flush();
        }
        return;
    }

    if (out_dir_arg == null or benchmark_report_arg == null or benchmark_artifacts_arg == null) {
        std.debug.print("error: --out-dir, --benchmark-report, and --benchmark-artifacts are required\n", .{});
        std.process.exit(2);
    }
    if (frequency_arg == 0 or min_samples_arg <= 0) {
        std.debug.print("error: --frequency and --min-samples must be positive\n", .{});
        std.process.exit(2);
    }
    if (top_arg == 0 or classify_arg == 0 or classify_arg > top_arg) {
        std.debug.print("error: --classify must be positive and no greater than --top\n", .{});
        std.process.exit(2);
    }

    const out_dir = out_dir_arg.?;
    try std.Io.Dir.cwd().createDirPath(io, out_dir);
    const report = runProfile(allocator, io, environ_map, .{
        .repo = repo_arg,
        .out_dir = out_dir,
        .benchmark_report = benchmark_report_arg.?,
        .wamr_ref = wamr_ref_arg,
        .benchmark_role = benchmark_role_arg,
        .benchmark_artifacts = benchmark_artifacts_arg.?,
        .execution_id = execution_id_arg,
        .wasmtime_cache = wasmtime_cache_arg,
        .frequency = frequency_arg,
        .min_samples = min_samples_arg,
        .top = top_arg,
        .classify = classify_arg,
        .frame_func = frame_func_arg,
        .max_perf_bytes = max_perf_bytes_arg,
    }) catch {
        var retained: json.Array = .init(allocator);
        if (std.Io.Dir.cwd().openDir(io, out_dir, .{ .iterate = true })) |dir_val| {
            var dir = dir_val;
            defer dir.close(io);
            var it = dir.iterate();
            while (it.next(io) catch null) |entry| {
                if (entry.kind == .file and
                    (std.mem.endsWith(u8, entry.name, ".perf.data") or
                        (std.mem.startsWith(u8, entry.name, "jit-") and std.mem.endsWith(u8, entry.name, ".dump"))))
                {
                    const fp = try std.fs.path.join(allocator, &.{ out_dir, entry.name });
                    if (gzipIfSmall(allocator, io, fp, max_perf_bytes_arg)) |gz_v| {
                        try retained.append(gz_v);
                    } else |_| {}
                }
            }
        } else |_| {}

        var fail_obj: json.ObjectMap = .empty;
        try fail_obj.put(allocator, "schema_version", .{ .integer = REPORT_SCHEMA_VERSION });
        try fail_obj.put(allocator, "kind", .{ .string = REPORT_KIND });
        try fail_obj.put(allocator, "status", .{ .string = "failed" });
        try fail_obj.put(allocator, "error", .{ .string = lastErrorMessage() });
        try fail_obj.put(allocator, "retained_perf_artifacts", .{ .array = retained });
        const fail_json = try bench_coremark.stringifyPrettySorted(allocator, .{ .object = fail_obj });
        const fail_path = try std.fs.path.join(allocator, &.{ out_dir, "failure.json" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = fail_path, .data = fail_json });
        std.debug.print("error: {s}\n", .{lastErrorMessage()});
        std.process.exit(1);
    };

    const profile_json = try bench_coremark.stringifyPrettySorted(allocator, report);
    const profile_json_path = try std.fs.path.join(allocator, &.{ out_dir, "profile.json" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = profile_json_path, .data = profile_json });

    const markdown = try renderMarkdown(allocator, report);
    const profile_md_path = try std.fs.path.join(allocator, &.{ out_dir, "profile.md" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = profile_md_path, .data = markdown });

    var out_buf: [4096]u8 = undefined;
    var out_w = std.Io.File.stdout().writer(io, &out_buf);
    try out_w.interface.writeAll(markdown);
    try out_w.interface.flush();

    if (environ_map.get("GITHUB_STEP_SUMMARY")) |summary_path| {
        if (summary_path.len != 0) {
            const existing = std.Io.Dir.cwd().readFileAlloc(io, summary_path, allocator, .limited(16 * 1024 * 1024)) catch "";
            const combined = try std.fmt.allocPrint(allocator, "{s}{s}", .{ existing, markdown });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = summary_path, .data = combined });
        }
    }
}

// ============================================================================
// Unit Tests (port of scripts/test_profile_coremark_aarch64.py)
// ============================================================================

test "validated coremark run counts fail closed" {
    const output =
        "2K performance run parameters for coremark.\n" ++
        "Iterations/Sec : 12000.0\n" ++
        "Iterations : 400000\n" ++
        "Correct operation validated.\n";
    const parsed = try parseValidatedCoremark(output, "engine", 1);
    try std.testing.expectEqual(@as(i64, 400000), parsed.iterations);
    try std.testing.expectApproxEqAbs(@as(f64, 12000.0), parsed.throughput, 1e-6);

    try std.testing.expectError(error.ProfileFailed, parseValidatedCoremark(output, "engine", 2));
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "must be one") != null);

    try std.testing.expectError(
        error.ProfileFailed,
        parseValidatedCoremark(output ++ "ERROR! bad crc\n", "engine", 1),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "CRC-validated") != null);
}

test "spill metrics are keyed by local function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const text =
        "[aot-spill-metric] local_func=3 mod=0 name=core_bench_list " ++
        "insts=10 clobbers=2 slots=4 spilled_vregs=3 scalar=3 v128=0 " ++
        "slots_scalar=4 slots_v128=0 spill_ld=7 spill_st=2 remat=1 " ++
        "callee_saved=4\n";
    const metrics = try parseSpillMetrics(allocator, text);
    const m3 = metrics.get(3).?.object;
    try std.testing.expectEqual(@as(i64, 7), m3.get("spill_ld").?.integer);
    try std.testing.expectEqualStrings("core_bench_list", m3.get("name").?.string);

    try std.testing.expectError(error.ProfileFailed, parseSpillMetrics(allocator, text ++ text));
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "duplicate") != null);
}

test "wasmtime function index and offsets are parsed and validated" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const text =
        "ffff0000 wasm[0]::function[15]::core_bench_list+0x4 (jitted-1.so)\n" ++
        "ffff0004 wasm[0]::function[15]::core_bench_list +0x8 (jitted-1.so)\n" ++
        "aaaa0000 wasmtime::runtime+0x10 (/bin/wasmtime)\n";
    const parsed = try parseWasmtimeSamples(allocator, text, null);
    try std.testing.expectEqual(@as(i64, 3), parsed.total_samples);
    const f15 = parsed.functions.get(15).?;
    try std.testing.expectEqual(@as(i64, 2), f15.samples);
    try std.testing.expectEqual(@as(i64, 1), f15.offsets.get(4).?);
    try std.testing.expectEqual(@as(i64, 1), f15.offsets.get(8).?);

    var fn_names: std.AutoArrayHashMapUnmanaged(usize, []const u8) = .empty;
    try fn_names.put(allocator, 15, "core_bench_list");
    const identity = compare_hot_function.WasmModuleIdentity{
        .sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        .imported_function_count = 12,
        .local_function_count = 73,
        .function_names = fn_names,
    };
    try validateWasmtimeMapping(&parsed, &identity);

    const bad = try parseWasmtimeSamples(allocator, "ffff wasm[0]::function[15]::wrong+0x4 (jitted.so)\n", null);
    try std.testing.expectError(error.ProfileFailed, validateWasmtimeMapping(&bad, &identity));
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "name mismatch") != null);

    const plain = try parseWasmtimeSamples(
        allocator,
        "ffff core_bench_list+0x14 (/work/jitted-99-3.so)\n",
        &identity,
    );
    try std.testing.expectEqual(@as(i64, 1), plain.functions.get(15).?.samples);
    try std.testing.expectEqual(@as(i64, 1), plain.functions.get(15).?.offsets.get(0x14).?);

    try std.testing.expectError(
        error.ProfileFailed,
        parseWasmtimeSamples(allocator, "ffff core_bench_list+0x14 (/work/jitted-99-4.so)\n", &identity),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "implies local function") != null);

    try std.testing.expectError(
        error.ProfileFailed,
        parseWasmtimeSamples(allocator, "ffff wasm[0]::function[15]+0x4 wasm[0]::function[16]+0x8\n", null),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "ambiguous") != null);
}

test "wasmtime hot instruction mapping and narrow alu provenance" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const parsed_insts = [_]compare_hot_function.Instruction{
        .{ .offset = 0, .size = 4, .mnemonic = "mov", .operands = "w0, w1", .text = "mov w0, w1", .raw_bytes = &.{ 0, 0, 0, 0 } },
        .{ .offset = 4, .size = 4, .mnemonic = "add", .operands = "w0, w0, #1", .text = "add w0, w0, #1", .raw_bytes = &.{ 0, 0, 0, 0 } },
    };
    var offsets: std.AutoHashMapUnmanaged(usize, i64) = .empty;
    try offsets.put(allocator, 0, 6);
    try offsets.put(allocator, 4, 4);
    const classified = try classifyWasmtimeFunctionWithParsed(allocator, &parsed_insts, 15, &offsets, 20);
    try std.testing.expectEqual(@as(i64, 10), classified.object.get("mapped_instruction_samples").?.integer);
    try std.testing.expectEqual(
        @as(i64, 6),
        classified.object.get("classes").?.object.get("regmov").?.object.get("samples").?.integer,
    );
    try std.testing.expectEqual(
        @as(i64, 4),
        classified.object.get("classes").?.object.get("alu").?.object.get("samples").?.integer,
    );

    const objdump =
        "00000100 wasm[0]::function[15]::core_bench_list:\n" ++
        "       100: 29 04 00 91                  add     x9, x1, #1\n" ++
        "       104: 40 68 69 b8                  ldr     w0, [x2, x9]\n" ++
        "       108: c0 03 5f d6                  ret\n";
    var narrow_offsets: std.AutoHashMapUnmanaged(usize, i64) = .empty;
    try narrow_offsets.put(allocator, 0, 9);
    try narrow_offsets.put(allocator, 4, 3);
    try narrow_offsets.put(allocator, 8, 1);

    const result = try analyzeWasmtimeAluProvenance(allocator, objdump, 15, &narrow_offsets, 20, 18);
    try std.testing.expectEqual(@as(i64, 9), result.object.get("broad_alu_samples").?.integer);
    try std.testing.expectEqual(
        @as(i64, 9),
        result.object.get("categories").?.object.get("address_generation").?.object.get("samples").?.integer,
    );
    try std.testing.expectEqual(
        @as(i64, 13),
        result.object.get("sample_mapping").?.object.get("mapped_function_samples").?.integer,
    );
    try std.testing.expectEqual(
        @as(i64, 0),
        result.object.get("sample_mapping").?.object.get("unresolved_function_samples").?.integer,
    );
    try std.testing.expectEqual(
        @as(i64, 2),
        result.object.get("global_sample_mapping").?.object.get("unattributed_samples").?.integer,
    );
}

test "narrow report assembly and report validation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const insts = [_]provenance.InstructionInput{
        .{ .offset = 0, .size = 4, .mnemonic = "add", .operands = "x9, x1, #4", .text = "add x9, x1, #4", .address = 0 },
        .{ .offset = 4, .size = 4, .mnemonic = "ldr", .operands = "w0, [x2, x9]", .text = "ldr w0, [x2, x9]", .address = 4 },
        .{ .offset = 8, .size = 4, .mnemonic = "ret", .operands = "", .text = "ret", .address = 8 },
    };
    const broad = [_][]const u8{ "alu", "other", "other" };
    var diag = provenance.Diagnostic{};
    const wamr_alu = try provenance.analyzeInstructionStream(
        allocator,
        &insts,
        .{
            .broad_classes = &broad,
            .samples_by_offset = &.{
                .{ .offset = 0, .samples = 1 },
                .{ .offset = 4, .samples = 29 },
            },
            .total_run_samples = 100,
            .global_attributed_samples = 99,
        },
        &diag,
    );
    const wt_alu = try provenance.analyzeInstructionStream(
        allocator,
        &insts,
        .{
            .broad_classes = &broad,
            .samples_by_offset = &.{
                .{ .offset = 0, .samples = 1 },
                .{ .offset = 4, .samples = 27 },
            },
            .total_run_samples = 100,
            .global_attributed_samples = 98,
        },
        &diag,
    );
    const assembled = try assembleAluProvenance(allocator, wamr_alu, wt_alu, 1, 1);
    try std.testing.expectError(error.ProfileFailed, assembleAluProvenance(allocator, wamr_alu, wt_alu, 7, 1));

    const report_json =
        \\{
        \\  "schema_version": 2,
        \\  "kind": "coremark-aarch64-matched-profile",
        \\  "architecture": "aarch64",
        \\  "benchmark": {
        \\    "report_id": "12345678-1234-5678-1234-567812345678",
        \\    "generated_at": "2026-09-08T00:00:00+00:00",
        \\    "report_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        \\    "artifact_handoff": {
        \\      "directory": "/artifacts",
        \\      "manifest_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        \\    },
        \\    "execution": {"provider": "local", "run_id": "test-run"},
        \\    "producer": {
        \\      "source_sha": "cccccccccccccccccccccccccccccccccccccccc",
        \\      "script": {
        \\        "path": "scripts/bench_coremark.zig",
        \\        "sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        \\      }
        \\    },
        \\    "target": {
        \\      "identity": {
        \\        "source": {"sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
        \\        "runtime": {"sha256": "1111111111111111111111111111111111111111111111111111111111111111"},
        \\        "compiler": {"sha256": "1111111111111111111111111111111111111111111111111111111111111111"},
        \\        "module": {"sha256": "1111111111111111111111111111111111111111111111111111111111111111"}
        \\      }
        \\    },
        \\    "wasmtime_baseline": {
        \\      "identity": {
        \\        "version": "43.0.0",
        \\        "runtime": {"sha256": "3333333333333333333333333333333333333333333333333333333333333333"}
        \\      }
        \\    }
        \\  },
        \\  "provenance": {
        \\    "producer_source_sha": "cccccccccccccccccccccccccccccccccccccccc",
        \\    "script_path": "scripts/profile_coremark_aarch64.zig",
        \\    "script_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        \\    "analysis_module": {
        \\      "path": "scripts/aarch64_instruction_provenance.zig",
        \\      "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        \\    },
        \\    "analysis_sources": {
        \\      "source_mode": "commit-verified",
        \\      "commit": "cccccccccccccccccccccccccccccccccccccccc",
        \\      "files": {
        \\        "scripts/profile_coremark_aarch64.zig": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        \\        "scripts/aarch64_instruction_provenance.zig": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        \\        "scripts/compare_hot_function.zig": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        \\        "scripts/bench_coremark.zig": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        \\        "scripts/benchmark_optimize.zig": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        \\        "scripts/aot_jit_attr.zig": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        \\      }
        \\    },
        \\    "execution": {"provider": "local", "run_id": "test-run"}
        \\  },
        \\  "guest_args": ["0", "0", "0", "400000", "0"],
        \\  "expected_iterations": 400000,
        \\  "classifier_wording": {
        \\    "all_alu": "all ALU-class instructions: add/sub, logical operations, mul/div, shifts, compares, csel, and address-generation instructions",
        \\    "narrow_alu_provenance": "architecture-only CFG and reaching-definition paths preserve the legacy all_alu partition and separately classify a complete common gating universe as address_generation, structural_address_guard, algorithmic_alu, mixed, or unknown without engine-register heuristics"
        \\  },
        \\  "affinity": {"verified": true},
        \\  "profile_schedule": [
        \\    {"engine": "wamr", "phase": "warmup"},
        \\    {"engine": "wasmtime", "phase": "warmup"},
        \\    {"engine": "wasmtime", "phase": "warmup"},
        \\    {"engine": "wamr", "phase": "warmup"},
        \\    {"engine": "wamr", "phase": "profile"},
        \\    {"engine": "wasmtime", "phase": "profile"},
        \\    {"engine": "wasmtime", "phase": "profile"},
        \\    {"engine": "wamr", "phase": "profile"}
        \\  ],
        \\  "minimum_attribution_coverage_pct": 99.0,
        \\  "wamr_captures": [
        \\    {"coverage_pct": 99.96, "mapping": {"authoritative": true}},
        \\    {"coverage_pct": 99.95, "mapping": {"authoritative": true}}
        \\  ],
        \\  "wasmtime_captures": [
        \\    {"coverage_pct": 99.9},
        \\    {"coverage_pct": 99.9}
        \\  ],
        \\  "wasm": {"imported_function_count": 12, "local_function_count": 73},
        \\  "wamr": {
        \\    "commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        \\    "runtime_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        \\    "compiler_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        \\    "cwasm_sha256": "1111111111111111111111111111111111111111111111111111111111111111"
        \\  },
        \\  "wasmtime": {
        \\    "version": "43.0.0",
        \\    "sha256": "3333333333333333333333333333333333333333333333333333333333333333"
        \\  },
        \\  "engines": {
        \\    "wamr": {"total_samples": 100, "attributed_samples": 99},
        \\    "wasmtime": {"total_samples": 100, "attributed_samples": 98}
        \\  },
        \\  "matched_functions": [
        \\    {
        \\      "local_func": 3,
        \\      "wasm_function_index": 15,
        \\      "wamr": {"samples": 30},
        \\      "wasmtime": {"samples": 28},
        \\      "class_groups": {
        \\        "all_alu": {"wamr_samples": 1, "wasmtime_samples": 1}
        \\      }
        \\    }
        \\  ],
        \\  "retained_analysis_artifacts": {
        \\    "wamr_cwasm": {
        \\      "retained": true,
        \\      "source_sha256": "1111111111111111111111111111111111111111111111111111111111111111"
        \\    }
        \\  }
        \\}
    ;
    var parsed_rep = try json.parseFromSlice(Value, allocator, report_json, .{});
    var mf0 = parsed_rep.value.object.getPtr("matched_functions").?.array.items[0].object;
    try mf0.put(allocator, "alu_provenance", assembled);
    parsed_rep.value.object.getPtr("matched_functions").?.array.items[0] = .{ .object = mf0 };

    var top_alu: json.ObjectMap = .empty;
    try top_alu.put(allocator, "schema_version", .{ .integer = provenance.SCHEMA_VERSION });
    try top_alu.put(allocator, "kind", .{ .string = provenance.ANALYSIS_KIND });
    try top_alu.put(allocator, "target_local_func", .{ .integer = 3 });
    try top_alu.put(allocator, "target_wasm_function_index", .{ .integer = 15 });
    try top_alu.put(allocator, "gate", try cloneJsonValue(allocator, assembled.object.get("gate").?));
    try parsed_rep.value.object.put(allocator, "alu_provenance", .{ .object = top_alu });

    try validateReport(allocator, parsed_rep.value);

    var bad_rep = try cloneJsonValue(allocator, parsed_rep.value);
    try bad_rep.object.getPtr("matched_functions").?.array.items[0].object.put(allocator, "wasm_function_index", .{ .integer = 14 });
    try std.testing.expectError(error.ProfileFailed, validateReport(allocator, bad_rep));
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "inconsistent") != null);

    const hist_json =
        \\{
        \\  "schema_version": 1,
        \\  "kind": "coremark-aarch64-matched-profile",
        \\  "authoritative_baseline_run": 33631050708
        \\}
    ;
    const hist_parsed = try json.parseFromSlice(Value, allocator, hist_json, .{});
    const status = try profileReportStatus(allocator, hist_parsed.value);
    try std.testing.expectEqualStrings("historical-unverified", status.object.get("status").?.string);
    try std.testing.expect(!status.object.get("authoritative").?.bool);
    try std.testing.expect(status.object.get("known_historical_baseline").?.bool);
}

test "frame diagnostics use exact compiler and artifact" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);

    const modes = [_][]const u8{ "enabled", "disabled", "unsupported", "changed-code", "stale" };
    for (modes) |mode| {
        const mode_dir = try std.fs.path.join(allocator, &.{ root, mode });
        try std.Io.Dir.cwd().createDirPath(io, mode_dir);
        const cwasm = try std.fs.path.join(allocator, &.{ mode_dir, "benchmark.cwasm" });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cwasm, .data = "exact benchmark code" });
        const compiler = try std.fs.path.join(allocator, &.{ mode_dir, "retained-wamrc" });
        const fixture = try std.fs.path.join(allocator, &.{ mode_dir, "fixture.wasm" });
        const helper = try std.fs.path.join(allocator, &.{ mode_dir, "aot_jit_attr.zig" });
        const metadata = try std.fs.path.join(allocator, &.{ mode_dir, "wamr-frame.mod0.func10.json" });
        if (std.mem.eql(u8, mode, "stale")) {
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = metadata, .data = "old metadata" });
        }

        const MockCtx = struct {
            io: std.Io,
            mode: []const u8,
            mode_dir: []const u8,
            compiler: []const u8,
            fixture: []const u8,
            cwasm: []const u8,
            metadata: []const u8,
            helper: []const u8,
            call_count: usize = 0,
        };
        var ctx = MockCtx{
            .io = io,
            .mode = mode,
            .mode_dir = mode_dir,
            .compiler = compiler,
            .fixture = fixture,
            .cwasm = cwasm,
            .metadata = metadata,
            .helper = helper,
        };

        const runner = CommandRunner{
            .ctx = &ctx,
            .runFn = struct {
                fn run(
                    raw_ctx: *anyopaque,
                    alloc: Allocator,
                    command: []const []const u8,
                    _: []const u8,
                    cwd: []const u8,
                    env_map: *const std.StringHashMap([]const u8),
                ) ProfileError!RunCommandResult {
                    const c: *MockCtx = @ptrCast(@alignCast(raw_ctx));
                    c.call_count += 1;
                    if (std.mem.eql(u8, command[0], c.compiler)) {
                        if (!std.mem.eql(u8, cwd, c.mode_dir)) return fail("wrong cwd", .{});
                        if (!std.mem.eql(u8, command[2], c.fixture)) return fail("wrong fixture", .{});
                        if (std.mem.eql(u8, c.mode, "disabled")) {
                            if (env_map.contains("WAMR_AOT_FRAME_ATTRIBUTION")) return fail("unexpected frame env", .{});
                        } else {
                            const exp_prefix = std.fmt.allocPrint(alloc, "{s}/wamr-frame", .{c.mode_dir}) catch unreachable;
                            if (!std.mem.eql(u8, env_map.get("WAMR_AOT_FRAME_ATTRIBUTION") orelse "", exp_prefix)) return fail("wrong prefix", .{});
                            if (!std.mem.eql(u8, env_map.get("WAMR_AOT_FRAME_ATTRIBUTION_MODULE") orelse "", "0")) return fail("wrong mod", .{});
                            if (!std.mem.eql(u8, env_map.get("WAMR_AOT_FRAME_ATTRIBUTION_FUNC") orelse "", "10")) return fail("wrong func", .{});
                        }
                        const out_file = command[command.len - 1];
                        const data = if (std.mem.eql(u8, c.mode, "changed-code")) "changed code" else "exact benchmark code";
                        std.Io.Dir.cwd().writeFile(c.io, .{ .sub_path = out_file, .data = data }) catch unreachable;
                        if (!std.mem.eql(u8, c.mode, "disabled") and !std.mem.eql(u8, c.mode, "unsupported")) {
                            std.Io.Dir.cwd().writeFile(c.io, .{ .sub_path = c.metadata, .data = "{}" }) catch unreachable;
                        }
                    } else {
                        if (!std.mem.eql(u8, command[1], c.helper)) return fail("wrong helper", .{});
                    }
                    return .{};
                }
            }.run,
        };

        var base_env: std.StringHashMap([]const u8) = .init(allocator);
        try base_env.put("WAMR_AOT_FRAME_ATTRIBUTION", "/unrecorded");
        try base_env.put("WAMR_AOT_FRAME_ATTRIBUTION_MODULE", "9");
        try base_env.put("WAMR_AOT_FRAME_ATTRIBUTION_FUNC", "99");

        const ff: ?usize = if (std.mem.eql(u8, mode, "disabled")) null else 10;
        if (std.mem.eql(u8, mode, "unsupported")) {
            try std.testing.expectError(
                error.ProfileFailed,
                compileWamrDiagnostics(allocator, io, runner, compiler, fixture, cwasm, mode_dir, mode_dir, helper, ff, &base_env),
            );
            try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "benchmark a candidate") != null);
            try std.testing.expectEqual(@as(usize, 1), ctx.call_count);
        } else if (std.mem.eql(u8, mode, "changed-code")) {
            try std.testing.expectError(
                error.ProfileFailed,
                compileWamrDiagnostics(allocator, io, runner, compiler, fixture, cwasm, mode_dir, mode_dir, helper, ff, &base_env),
            );
            try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "exact benchmark") != null);
            try std.testing.expectEqual(@as(usize, 1), ctx.call_count);
        } else if (std.mem.eql(u8, mode, "stale")) {
            try std.testing.expectError(
                error.ProfileFailed,
                compileWamrDiagnostics(allocator, io, runner, compiler, fixture, cwasm, mode_dir, mode_dir, helper, ff, &base_env),
            );
            try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "existing frame metadata") != null);
            try std.testing.expectEqual(@as(usize, 0), ctx.call_count);
        } else {
            const res = try compileWamrDiagnostics(allocator, io, runner, compiler, fixture, cwasm, mode_dir, mode_dir, helper, ff, &base_env);
            try std.testing.expectEqual(@as(usize, 0), res.metrics.count());
            if (std.mem.eql(u8, mode, "enabled")) {
                try std.testing.expectEqualStrings(metadata, res.metadata_path.?);
                try std.testing.expectEqual(@as(usize, 2), ctx.call_count);
            } else {
                try std.testing.expect(res.metadata_path == null);
                try std.testing.expectEqual(@as(usize, 1), ctx.call_count);
            }
            const diag_cwasm = try std.fmt.allocPrint(allocator, "{s}/coremark.diagnostic.cwasm", .{mode_dir});
            try std.testing.expect(!fileExists(io, diag_cwasm));
        }
    }
}

test "each wamr capture must pass exact mapping and coverage and aggregates preserve balanced samples" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const base_json =
        \\{
        \\  "total_samples": 34410,
        \\  "attributed_samples": 34398,
        \\  "mapping": {
        \\    "authoritative": true,
        \\    "override": null,
        \\    "size": 122880,
        \\    "expected_size": 122880
        \\  }
        \\}
    ;
    const base_parsed = try json.parseFromSlice(Value, allocator, base_json, .{});
    const first = try validateWamrCapture(allocator, base_parsed.value, 1000, MIN_ATTRIBUTION_COVERAGE_PCT);
    try std.testing.expect(first.object.get("coverage_pct").?.float > 99.9);

    var low_cov = try cloneJsonValue(allocator, base_parsed.value);
    try low_cov.object.put(allocator, "total_samples", .{ .integer = 1000 });
    try low_cov.object.put(allocator, "attributed_samples", .{ .integer = 980 });
    try std.testing.expectError(error.ProfileFailed, validateWamrCapture(allocator, low_cov, 1000, MIN_ATTRIBUTION_COVERAGE_PCT));
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "below") != null);

    var low_samples = try cloneJsonValue(allocator, base_parsed.value);
    try low_samples.object.put(allocator, "total_samples", .{ .integer = 999 });
    try low_samples.object.put(allocator, "attributed_samples", .{ .integer = 998 });
    try std.testing.expectError(error.ProfileFailed, validateWamrCapture(allocator, low_samples, 1000, MIN_ATTRIBUTION_COVERAGE_PCT));
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "requires at least") != null);

    const rankings_json =
        \\[
        \\  {
        \\    "text_size": 100,
        \\    "function_count": 2,
        \\    "total_samples": 100,
        \\    "attributed_samples": 99,
        \\    "top_functions": [
        \\      {"local_func": 0, "samples": 60, "percent_of_run": 60.0, "code_bytes": 40}
        \\    ]
        \\  },
        \\  {
        \\    "text_size": 100,
        \\    "function_count": 2,
        \\    "total_samples": 120,
        \\    "attributed_samples": 118,
        \\    "top_functions": [
        \\      {"local_func": 0, "samples": 70, "percent_of_run": 58.3, "code_bytes": 40}
        \\    ]
        \\  }
        \\]
    ;
    const rankings_parsed = try json.parseFromSlice(Value, allocator, rankings_json, .{});
    const merged = try aggregateWamrRankings(allocator, rankings_parsed.value.array.items);
    try std.testing.expectEqual(@as(i64, 220), merged.object.get("total_samples").?.integer);
    try std.testing.expectEqual(@as(i64, 130), merged.object.get("top_functions").?.array.items[0].object.get("samples").?.integer);
}

test "analysis dependencies must match committed source" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const repo = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    const scripts_dir = try std.fs.path.join(allocator, &.{ repo, "scripts" });
    try std.Io.Dir.cwd().createDirPath(io, scripts_dir);
    const rel_paths = [_][]const u8{
        "scripts/compare_hot_function.zig",
        "scripts/aot_jit_attr.zig",
    };
    for (rel_paths) |rel| {
        const fp = try std.fs.path.join(allocator, &.{ repo, rel });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = fp, .data = "VALUE = 1\n" });
    }

    const good_cat = struct {
        fn cat(_: Allocator, _: []const u8, _: []const u8) ?[]const u8 {
            return "VALUE = 1\n";
        }
    }.cat;
    const sources = try captureAnalysisSources(allocator, io, repo, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", &rel_paths, good_cat);
    try std.testing.expectEqualStrings("commit-verified", sources.object.get("source_mode").?.string);
    try validateAnalysisSourcesUnchanged(allocator, io, repo, sources);

    const fp0 = try std.fs.path.join(allocator, &.{ repo, rel_paths[0] });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = fp0, .data = "VALUE = 2\n" });
    try std.testing.expectError(
        error.ProfileFailed,
        captureAnalysisSources(allocator, io, repo, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", &rel_paths, good_cat),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "differs from") != null);
    try std.testing.expectError(
        error.ProfileFailed,
        validateAnalysisSourcesUnchanged(allocator, io, repo, sources),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "changed during") != null);
}
