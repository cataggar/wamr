const std = @import("std");
const Allocator = std.mem.Allocator;
const json = std.json;
const Value = json.Value;

const provenance = @import("aarch64_instruction_provenance.zig");
const profile = @import("profile_coremark_aarch64.zig");
const compare_hot_function = @import("compare_hot_function.zig");
const aot_jit_attr = @import("aot_jit_attr.zig");
const bench_coremark = @import("bench_coremark.zig");

pub const ReanalyzeError = error{
    ReanalyzeFailed,
    OutOfMemory,
    WriteFailed,
};

threadlocal var last_error_buf: [4096]u8 = undefined;
threadlocal var last_error_len: usize = 0;

pub fn lastErrorMessage() []const u8 {
    return last_error_buf[0..last_error_len];
}

pub fn fail(comptime fmt: []const u8, args: anytype) ReanalyzeError {
    const written = std.fmt.bufPrint(&last_error_buf, fmt, args) catch blk: {
        const fallback = "reanalyze error";
        @memcpy(last_error_buf[0..fallback.len], fallback);
        break :blk last_error_buf[0..fallback.len];
    };
    last_error_len = written.len;
    return error.ReanalyzeFailed;
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

pub fn auditRows(allocator: Allocator, analysis_val: Value, engine: []const u8) ReanalyzeError![]Value {
    var diag = provenance.Diagnostic{};
    provenance.validateAnalysisSamples(allocator, analysis_val, engine, &diag) catch {
        return fail("{s}", .{diag.message});
    };
    const universe = analysis_val.object.get("common_gating_universe").?.object;
    var counts: std.StringHashMap(i64) = .init(allocator);
    for (provenance.CATEGORIES) |cat| {
        try counts.put(cat, 0);
    }
    var seen: std.AutoHashMap(i64, void) = .init(allocator);
    var uncertain: std.ArrayList(Value) = .empty;

    const sampled_arr = universe.get("sampled_instructions").?.array;
    for (sampled_arr.items) |row_val| {
        if (row_val != .object) {
            return fail("{s}: malformed or duplicate sampled instruction", .{engine});
        }
        const row = row_val.object;
        const offset = jsonInt(row.get("offset")) orelse return fail("{s}: malformed or duplicate sampled instruction", .{engine});
        const samples = jsonInt(row.get("samples")) orelse return fail("{s}: malformed or duplicate sampled instruction", .{engine});
        const category = jsonString(row.get("category")) orelse return fail("{s}: malformed or duplicate sampled instruction", .{engine});
        const reason = jsonString(row.get("reason"));
        const path_ev = row.get("path_evidence");
        if (offset < 0 or seen.contains(offset) or samples <= 0 or !counts.contains(category) or
            reason == null or path_ev == null or path_ev.? != .array)
        {
            return fail("{s}: malformed or duplicate sampled instruction", .{engine});
        }
        try seen.put(offset, {});
        try counts.put(category, counts.get(category).? + samples);
        if (std.mem.eql(u8, category, "unknown") or std.mem.eql(u8, category, "mixed")) {
            try uncertain.append(allocator, try profile.cloneJsonValue(allocator, row_val));
        }
    }

    const cats_obj = universe.get("categories").?.object;
    for (provenance.CATEGORIES) |cat| {
        const expected = cats_obj.get(cat).?.object.get("samples").?.integer;
        if (counts.get(cat).? != expected) {
            return fail("{s}: {s} sampled rows do not reconcile", .{ engine, cat });
        }
    }

    const items = try uncertain.toOwnedSlice(allocator);
    std.mem.sort(Value, items, {}, struct {
        fn lessThan(_: void, a: Value, b: Value) bool {
            const as = a.object.get("samples").?.integer;
            const bs = b.object.get("samples").?.integer;
            if (as != bs) return as > bs;
            return a.object.get("offset").?.integer < b.object.get("offset").?.integer;
        }
    }.lessThan);
    return items;
}

fn reanalyzeWasmtimeNativeExhaustion(
    allocator: Allocator,
    analyses_val: Value,
    wasmtime_objdump_text: []const u8,
) ReanalyzeError!Value {
    const analyses = analyses_val.object;
    const wasmtime_saved = analyses.get("wasmtime").?;
    const insts = compare_hot_function.parseDisassembly(allocator, wasmtime_objdump_text, 15) catch |err| {
        return fail("failed to parse Wasmtime disassembly: {s}", .{@errorName(err)});
    };
    const address_base = profile.wasmtimeFunctionAddressBase(wasmtime_objdump_text, 15) catch {
        return fail("{s}", .{profile.lastErrorMessage()});
    };

    var adapter = try allocator.alloc(aot_jit_attr.Instruction, insts.len);
    var prov_inputs = try allocator.alloc(provenance.InstructionInput, insts.len);
    var inst_by_offset: std.AutoHashMap(i64, compare_hot_function.Instruction) = .init(allocator);
    for (insts, 0..) |item, idx| {
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
        try inst_by_offset.put(@intCast(item.offset), item);
    }

    const broad_classes_list = aot_jit_attr.classifyInstructionStream(allocator, adapter, null, "aarch64") catch {
        return fail("{s}", .{aot_jit_attr.lastError()});
    };

    var sample_entries: std.ArrayList(provenance.SampleEntry) = .empty;
    const sampled_rows = wasmtime_saved.object
        .get("common_gating_universe").?.object
        .get("sampled_instructions").?.array.items;

    for (sampled_rows) |row_val| {
        const row = row_val.object;
        const offset = row.get("offset").?.integer;
        const inst = inst_by_offset.get(offset) orelse {
            return fail("wasmtime sampled offset {d} is absent from native disassembly", .{offset});
        };
        const expected_text = std.mem.trim(u8, inst.text, " \t\r\n");
        const row_inst = row.get("instruction").?.string;
        if (!std.mem.eql(u8, row_inst, expected_text)) {
            return fail(
                "wasmtime sampled instruction at offset {d} ('{s}') differs from native disassembly ('{s}')",
                .{ offset, row_inst, expected_text },
            );
        }
        try sample_entries.append(allocator, .{
            .offset = offset,
            .samples = row.get("samples").?.integer,
        });
    }

    const total_run_samples = wasmtime_saved.object.get("total_run_samples").?.integer;
    const global_attributed = wasmtime_saved.object
        .get("global_sample_mapping").?.object
        .get("attributed_samples").?.integer;

    var diag = provenance.Diagnostic{};
    const rerun = provenance.analyzeInstructionStream(
        allocator,
        prov_inputs,
        .{
            .broad_classes = broad_classes_list,
            .samples_by_offset = sample_entries.items,
            .total_run_samples = total_run_samples,
            .address_base = address_base,
            .global_attributed_samples = global_attributed,
        },
        &diag,
    ) catch {
        return fail("{s}", .{diag.message});
    };

    const check_keys = [_][]const u8{
        "categories",
        "coverage",
        "common_gating_universe",
        "cfg",
        "sampled_instructions",
        "reason_counts",
    };
    for (check_keys) |key| {
        if (!profile.jsonValueEql(rerun.object.get(key).?, wasmtime_saved.object.get(key).?)) {
            return fail("retained Wasmtime {s} differs from native disassembly rerun", .{key});
        }
    }

    const exhaustion = provenance.analyzeTerminalSinkExhaustion(
        allocator,
        prov_inputs,
        .{
            .broad_classes = broad_classes_list,
            .samples_by_offset = sample_entries.items,
            .total_run_samples = total_run_samples,
            .address_base = address_base,
            .global_attributed_samples = global_attributed,
        },
        &diag,
    ) catch {
        return fail("{s}", .{diag.message});
    };

    const exhaustion_gate = provenance.compareTerminalSinkExhaustion(
        allocator,
        analyses.get("wamr").?,
        wasmtime_saved,
        exhaustion,
        null,
        5.0,
        &diag,
    ) catch {
        return fail("{s}", .{diag.message});
    };

    var sink_summary: json.ObjectMap = .empty;
    var sit = exhaustion.object.get("sink_classes").?.object.iterator();
    while (sit.next()) |kv| {
        const sink = kv.key_ptr.*;
        const data = kv.value_ptr.object;
        const sampled = data.get("sampled_instructions").?.array.items;
        const top_len = @min(sampled.len, 10);
        var top_arr = json.Array.init(allocator);
        for (sampled[0..top_len]) |s_item| {
            try top_arr.append(try profile.cloneJsonValue(allocator, s_item));
        }

        var sobj: json.ObjectMap = .empty;
        try sobj.put(allocator, "samples", data.get("samples").?);
        try sobj.put(allocator, "percent_of_run", data.get("percent_of_run").?);
        try sobj.put(allocator, "percent_of_common_universe", data.get("percent_of_common_universe").?);
        try sobj.put(allocator, "static_instructions", data.get("static_instructions").?);
        try sobj.put(allocator, "top_sampled_instructions", .{ .array = top_arr });
        try sink_summary.put(allocator, try allocator.dupe(u8, sink), .{ .object = sobj });
    }

    var out: json.ObjectMap = .empty;
    try out.put(allocator, "verified_retained_provenance", .{ .bool = true });
    try out.put(allocator, "address_base", .{ .integer = address_base });
    try out.put(allocator, "static_instructions", .{ .integer = @intCast(insts.len) });
    try out.put(allocator, "common_static_instructions", exhaustion.object.get("common_static_instructions").?);
    try out.put(allocator, "sink_classes", .{ .object = sink_summary });
    try out.put(allocator, "reclassified_from_unknown", try profile.cloneJsonValue(allocator, exhaustion.object.get("reclassified_from_unknown").?));
    try out.put(allocator, "exhaustion_gate", exhaustion_gate);
    return .{ .object = out };
}

pub fn reanalyze(
    allocator: Allocator,
    report_val: Value,
    cwasm_bytes: []const u8,
    benchmark_bytes: []const u8,
    wasmtime_objdump_text: ?[]const u8,
) ReanalyzeError!Value {
    profile.validateReport(allocator, report_val) catch {
        return fail("{s}", .{profile.lastErrorMessage()});
    };
    const report = report_val.object;
    const artifact = report.get("retained_analysis_artifacts").?.object.get("wamr_cwasm").?.object;
    if (!std.mem.eql(u8, jsonStringOrEmpty(artifact.get("path")), "wamr-profiled.cwasm.gz") or
        @as(i64, @intCast(cwasm_bytes.len)) != (jsonInt(artifact.get("source_size_bytes")) orelse -1))
    {
        return fail("retained WAMR binary identity or size differs from artifact manifest", .{});
    }

    const bench_sha = try bench_coremark.sha256BytesAlloc(allocator, benchmark_bytes);
    const report_bench = report.get("benchmark").?.object;
    if (!std.mem.eql(u8, bench_sha, jsonStringOrEmpty(report_bench.get("report_sha256")))) {
        return fail("benchmark report hash differs from retained profile", .{});
    }

    const bench_parsed = json.parseFromSlice(Value, allocator, benchmark_bytes, .{}) catch {
        return fail("benchmark report is not valid JSON", .{});
    };
    const bench_id = if (bench_parsed.value == .object and bench_parsed.value.object.get("provenance") != null and bench_parsed.value.object.get("provenance").? == .object)
        jsonStringOrEmpty(bench_parsed.value.object.get("provenance").?.object.get("report_id"))
    else
        "";
    if (!std.mem.eql(u8, bench_id, jsonStringOrEmpty(report_bench.get("report_id")))) {
        return fail("benchmark report identity differs from retained profile", .{});
    }

    const cwasm_sha = try bench_coremark.sha256BytesAlloc(allocator, cwasm_bytes);
    const report_wamr = report.get("wamr").?.object;
    if (!std.mem.eql(u8, cwasm_sha, jsonStringOrEmpty(report_wamr.get("cwasm_sha256")))) {
        return fail("retained WAMR binary hash differs from profile", .{});
    }
    if (!std.mem.eql(u8, cwasm_sha, jsonStringOrEmpty(artifact.get("source_sha256")))) {
        return fail("retained WAMR binary hash differs from artifact manifest", .{});
    }

    var target_opt: ?Value = null;
    var target_count: usize = 0;
    for (report.get("matched_functions").?.array.items) |item_v| {
        if ((jsonInt(item_v.object.get("local_func")) orelse -1) == 3) {
            target_opt = item_v;
            target_count += 1;
        }
    }
    if (target_count != 1 or (jsonInt(target_opt.?.object.get("wasm_function_index")) orelse -1) != 15) {
        return fail("missing or ambiguous core_bench_list function identity", .{});
    }
    const target = target_opt.?.object;
    const analyses_val = target.get("alu_provenance").?;
    const analyses = analyses_val.object;

    var diag = provenance.Diagnostic{};
    const gate = provenance.compareEngineAnalyses(
        allocator,
        analyses.get("wamr").?,
        analyses.get("wasmtime").?,
        5.0,
        &diag,
    ) catch {
        return fail("{s}", .{diag.message});
    };
    if (!profile.jsonValueEql(gate, analyses.get("gate").?) or
        !profile.jsonValueEql(gate, report.get("alu_provenance").?.object.get("gate").?))
    {
        return fail("retained conservative gate differs from recomputed gate", .{});
    }

    var total_run_samples_obj: json.ObjectMap = .empty;
    var categories_obj: json.ObjectMap = .empty;
    var unknown_clusters_obj: json.ObjectMap = .empty;
    var top_uncertain_obj: json.ObjectMap = .empty;

    for ([_][]const u8{ "wamr", "wasmtime" }) |engine| {
        const eng_analysis = analyses.get(engine).?;
        const ranked = try auditRows(allocator, eng_analysis, engine);
        const clusters_val = provenance.classifyUnknownReasonClusters(allocator, eng_analysis, engine, &diag) catch {
            return fail("{s}", .{diag.message});
        };
        const clusters = clusters_val.object;

        try total_run_samples_obj.put(allocator, engine, eng_analysis.object.get("total_run_samples").?);
        try categories_obj.put(
            allocator,
            engine,
            try profile.cloneJsonValue(allocator, eng_analysis.object.get("common_gating_universe").?.object.get("categories").?),
        );

        var inner_clusters: json.ObjectMap = .empty;
        var cit = clusters.get("clusters").?.object.iterator();
        while (cit.next()) |kv| {
            const cname = kv.key_ptr.*;
            const item = kv.value_ptr.object;
            const si_items = item.get("sampled_instructions").?.array.items;
            const top5_len = @min(si_items.len, 5);
            var top5_arr = json.Array.init(allocator);
            for (si_items[0..top5_len]) |si| {
                try top5_arr.append(try profile.cloneJsonValue(allocator, si));
            }
            var c_entry: json.ObjectMap = .empty;
            try c_entry.put(allocator, "samples", item.get("samples").?);
            try c_entry.put(allocator, "percent_of_run", item.get("percent_of_run").?);
            try c_entry.put(allocator, "percent_of_uncertain", item.get("percent_of_uncertain").?);
            try c_entry.put(allocator, "sampled_instruction_count", item.get("sampled_instruction_count").?);
            try c_entry.put(allocator, "top_sampled_instructions", .{ .array = top5_arr });
            try inner_clusters.put(allocator, try allocator.dupe(u8, cname), .{ .object = c_entry });
        }

        var eng_clusters_out: json.ObjectMap = .empty;
        try eng_clusters_out.put(allocator, "uncertain_samples", clusters.get("uncertain_samples").?);
        try eng_clusters_out.put(allocator, "uncertain_percent_of_run", clusters.get("uncertain_percent_of_run").?);
        try eng_clusters_out.put(allocator, "clusters", .{ .object = inner_clusters });
        try unknown_clusters_obj.put(allocator, engine, .{ .object = eng_clusters_out });

        const top20_len = @min(ranked.len, 20);
        var top20_arr = json.Array.init(allocator);
        for (ranked[0..top20_len]) |r_item| {
            try top20_arr.append(r_item);
        }
        try top_uncertain_obj.put(allocator, engine, .{ .array = top20_arr });
    }

    const source_commit = report.get("provenance").?.object
        .get("analysis_sources").?.object
        .get("commit").?.string;

    var out: json.ObjectMap = .empty;
    try out.put(allocator, "source_commit", .{ .string = try allocator.dupe(u8, source_commit) });
    try out.put(allocator, "cwasm_sha256", .{ .string = try allocator.dupe(u8, report_wamr.get("cwasm_sha256").?.string) });
    try out.put(allocator, "function", .{ .string = "core_bench_list" });
    try out.put(allocator, "local_func", .{ .integer = 3 });
    try out.put(allocator, "wasm_index", .{ .integer = 15 });
    try out.put(allocator, "total_run_samples", .{ .object = total_run_samples_obj });
    try out.put(allocator, "categories", .{ .object = categories_obj });
    try out.put(allocator, "unknown_reason_clusters", .{ .object = unknown_clusters_obj });
    try out.put(allocator, "top_uncertain_paths", .{ .object = top_uncertain_obj });
    try out.put(allocator, "gate", gate);
    try out.put(
        allocator,
        "scope",
        .{ .string = "Audits retained sampled instruction partitions, groups uncertain samples by reason cluster, recomputes the conservative gate, and when native Wasmtime disassembly is supplied, verifies terminal-sink exhaustion across CFG joins and cycles." },
    );

    if (wasmtime_objdump_text) |objdump_txt| {
        const wt_ex = try reanalyzeWasmtimeNativeExhaustion(allocator, analyses_val, objdump_txt);
        try out.put(allocator, "wasmtime_native_exhaustion", wt_ex);
    }

    return .{ .object = out };
}

fn readGzipOrRaw(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024));
    if (std.mem.endsWith(u8, path, ".gz") and raw.len >= 2 and raw[0] == 0x1f and raw[1] == 0x8b) {
        const res = try std.process.run(allocator, io, .{
            .argv = &.{ "gzip", "-dc", path },
            .stdout_limit = .limited(256 * 1024 * 1024),
            .stderr_limit = .limited(4 * 1024 * 1024),
        });
        allocator.free(res.stderr);
        switch (res.term) {
            .exited => |code| if (code == 0) return res.stdout,
            else => {},
        }
        return error.DecompressFailed;
    }
    return raw;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args_list: std.ArrayList([]const u8) = .empty;
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();
    _ = args_it.next();
    while (args_it.next()) |arg| {
        try args_list.append(allocator, arg);
    }

    var report_path: ?[]const u8 = null;
    var cwasm_gz_path: ?[]const u8 = null;
    var benchmark_report_path: ?[]const u8 = null;
    var wasmtime_objdump_path: ?[]const u8 = null;
    var json_out_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args_list.items.len) : (i += 1) {
        const arg = args_list.items[i];
        if (std.mem.eql(u8, arg, "--report")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            report_path = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--cwasm-gz")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            cwasm_gz_path = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--benchmark-report")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            benchmark_report_path = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--wasmtime-objdump")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            wasmtime_objdump_path = args_list.items[i];
        } else if (std.mem.eql(u8, arg, "--json-out")) {
            i += 1;
            if (i >= args_list.items.len) return error.InvalidArguments;
            json_out_path = args_list.items[i];
        }
    }

    const rp = report_path orelse return error.InvalidArguments;
    const cp = cwasm_gz_path orelse return error.InvalidArguments;
    const bp = benchmark_report_path orelse return error.InvalidArguments;

    const report_bytes = try std.Io.Dir.cwd().readFileAlloc(io, rp, allocator, .limited(64 * 1024 * 1024));
    const report_parsed = try json.parseFromSlice(Value, allocator, report_bytes, .{});
    const cwasm_bytes = try readGzipOrRaw(allocator, io, cp);
    const benchmark_bytes = try std.Io.Dir.cwd().readFileAlloc(io, bp, allocator, .limited(64 * 1024 * 1024));
    const objdump_text = if (wasmtime_objdump_path) |wp|
        try std.Io.Dir.cwd().readFileAlloc(io, wp, allocator, .limited(64 * 1024 * 1024))
    else
        null;

    const result = reanalyze(allocator, report_parsed.value, cwasm_bytes, benchmark_bytes, objdump_text) catch {
        std.debug.print("error: {s}\n", .{lastErrorMessage()});
        std.process.exit(1);
    };

    var aw: std.Io.Writer.Allocating = .init(allocator);
    try json.Stringify.value(result, .{ .whitespace = .indent_2 }, &aw.writer);
    try aw.writer.writeAll("\n");
    const rendered = aw.written();

    if (json_out_path) |jp| {
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = jp, .data = rendered });
    } else {
        var out_buf: [4096]u8 = undefined;
        var out_w = std.Io.File.stdout().writer(io, &out_buf);
        try out_w.interface.writeAll(rendered);
        try out_w.interface.flush();
    }
}

// ============================================================================
// Unit Tests (port of scripts/test_reanalyze_coremark_986.py)
// ============================================================================

const FixtureBundle = struct {
    report: Value,
    cwasm_bytes: []const u8,
    benchmark_bytes: []const u8,
    wasmtime_objdump: []const u8,
};

fn makeFixture(allocator: Allocator) !FixtureBundle {
    const cwasm_bytes = "cwasm-fixture";
    const cwasm_sha = try bench_coremark.sha256BytesAlloc(allocator, cwasm_bytes);
    const benchmark_bytes = "{\"provenance\":{\"report_id\":\"12345678-1234-5678-1234-567812345678\"}}";
    const benchmark_sha = try bench_coremark.sha256BytesAlloc(allocator, benchmark_bytes);

    const wamr_insts = [_]provenance.InstructionInput{
        .{ .offset = 0, .size = 4, .mnemonic = "add", .operands = "x9, x1, #4", .text = "add x9, x1, #4", .address = 0 },
        .{ .offset = 4, .size = 4, .mnemonic = "ldr", .operands = "w0, [x2, x9]", .text = "ldr w0, [x2, x9]", .address = 4 },
        .{ .offset = 8, .size = 4, .mnemonic = "add", .operands = "w10, w0, #1", .text = "add w10, w0, #1", .address = 8 },
        .{ .offset = 12, .size = 4, .mnemonic = "ret", .operands = "", .text = "ret", .address = 12 },
    };
    const wamr_broad = [_][]const u8{ "alu", "other", "alu", "other" };

    const wasmtime_objdump =
        "00001000 <wasm[0]::function[15]::core_bench_list>:\n" ++
        "    1000:\tb4000060 \tcbz\tx0, 0x100c\n" ++
        "    1004:\t91001028 \tadd\tx8, x1, #0x4\n" ++
        "    1008:\t14000002 \tb\t0x1010\n" ++
        "    100c:\t91002028 \tadd\tx8, x1, #0x8\n" ++
        "    1010:\tb8686840 \tldr\tw0, [x2, x8]\n" ++
        "    1014:\t1100040a \tadd\tw10, w0, #0x1\n" ++
        "    1018:\td65f03c0 \tret\n";

    const wt_parsed = try compare_hot_function.parseDisassembly(allocator, wasmtime_objdump, 15);
    var wt_adapter = try allocator.alloc(aot_jit_attr.Instruction, wt_parsed.len);
    var wt_inputs = try allocator.alloc(provenance.InstructionInput, wt_parsed.len);
    for (wt_parsed, 0..) |item, idx| {
        wt_adapter[idx] = .{ .address = item.offset, .offset = item.offset, .size = item.size, .text = item.text };
        wt_inputs[idx] = .{
            .offset = @intCast(item.offset),
            .size = @intCast(item.size),
            .mnemonic = item.mnemonic,
            .operands = item.operands,
            .text = item.text,
        };
    }
    const wt_broad = try aot_jit_attr.classifyInstructionStream(allocator, wt_adapter, null, "aarch64");

    var diag = provenance.Diagnostic{};
    var wamr_alu = try provenance.analyzeInstructionStream(
        allocator,
        &wamr_insts,
        .{
            .broad_classes = &wamr_broad,
            .samples_by_offset = &.{
                .{ .offset = 0, .samples = 2 },
                .{ .offset = 4, .samples = 25 },
                .{ .offset = 8, .samples = 3 },
            },
            .total_run_samples = 100,
            .global_attributed_samples = 99,
        },
        &diag,
    );
    var sm_wamr: json.ObjectMap = .empty;
    try sm_wamr.put(allocator, "mapped_function_samples", .{ .integer = 30 });
    try sm_wamr.put(allocator, "unresolved_function_samples", .{ .integer = 0 });
    try wamr_alu.object.put(allocator, "sample_mapping", .{ .object = sm_wamr });

    var wt_alu = try provenance.analyzeInstructionStream(
        allocator,
        wt_inputs,
        .{
            .broad_classes = wt_broad,
            .samples_by_offset = &.{
                .{ .offset = 0x4, .samples = 3 },
                .{ .offset = 0x10, .samples = 24 },
                .{ .offset = 0x14, .samples = 1 },
            },
            .total_run_samples = 100,
            .address_base = 0x1000,
            .global_attributed_samples = 98,
        },
        &diag,
    );
    var sm_wt: json.ObjectMap = .empty;
    try sm_wt.put(allocator, "mapped_function_samples", .{ .integer = 28 });
    try sm_wt.put(allocator, "unresolved_function_samples", .{ .integer = 0 });
    try wt_alu.object.put(allocator, "sample_mapping", .{ .object = sm_wt });

    const assembled = try profile.assembleAluProvenance(allocator, wamr_alu, wt_alu, 5, 4);

    const template = try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "schema_version": 2,
        \\  "kind": "coremark-aarch64-matched-profile",
        \\  "architecture": "aarch64",
        \\  "benchmark": {{
        \\    "report_id": "12345678-1234-5678-1234-567812345678",
        \\    "generated_at": "2026-09-08T00:00:00+00:00",
        \\    "report_sha256": "{s}",
        \\    "artifact_handoff": {{
        \\      "directory": "/artifacts",
        \\      "manifest_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        \\    }},
        \\    "execution": {{"provider": "local", "run_id": "test-run"}},
        \\    "producer": {{
        \\      "source_sha": "cccccccccccccccccccccccccccccccccccccccc",
        \\      "script": {{
        \\        "path": "scripts/bench_coremark.zig",
        \\        "sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        \\      }}
        \\    }},
        \\    "target": {{
        \\      "identity": {{
        \\        "source": {{"sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}},
        \\        "runtime": {{"sha256": "1111111111111111111111111111111111111111111111111111111111111111"}},
        \\        "compiler": {{"sha256": "1111111111111111111111111111111111111111111111111111111111111111"}},
        \\        "module": {{"sha256": "{s}"}}
        \\      }}
        \\    }},
        \\    "wasmtime_baseline": {{
        \\      "identity": {{
        \\        "version": "43.0.0",
        \\        "runtime": {{"sha256": "3333333333333333333333333333333333333333333333333333333333333333"}}
        \\      }}
        \\    }}
        \\  }},
        \\  "provenance": {{
        \\    "producer_source_sha": "cccccccccccccccccccccccccccccccccccccccc",
        \\    "script_path": "scripts/profile_coremark_aarch64.zig",
        \\    "script_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        \\    "analysis_module": {{
        \\      "path": "scripts/aarch64_instruction_provenance.zig",
        \\      "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        \\    }},
        \\    "analysis_sources": {{
        \\      "source_mode": "commit-verified",
        \\      "commit": "cccccccccccccccccccccccccccccccccccccccc",
        \\      "files": {{
        \\        "scripts/profile_coremark_aarch64.zig": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        \\        "scripts/aarch64_instruction_provenance.zig": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        \\        "scripts/compare_hot_function.zig": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        \\        "scripts/bench_coremark.zig": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        \\        "scripts/benchmark_optimize.zig": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        \\        "scripts/aot_jit_attr.zig": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        \\      }}
        \\    }},
        \\    "execution": {{"provider": "local", "run_id": "test-run"}}
        \\  }},
        \\  "guest_args": ["0", "0", "0", "400000", "0"],
        \\  "expected_iterations": 400000,
        \\  "classifier_wording": {{
        \\    "all_alu": "all ALU-class instructions: add/sub, logical operations, mul/div, shifts, compares, csel, and address-generation instructions",
        \\    "narrow_alu_provenance": "architecture-only CFG and reaching-definition paths preserve the legacy all_alu partition and separately classify a complete common gating universe as address_generation, structural_address_guard, algorithmic_alu, mixed, or unknown without engine-register heuristics"
        \\  }},
        \\  "affinity": {{"verified": true}},
        \\  "profile_schedule": [
        \\    {{"engine": "wamr", "phase": "warmup"}},
        \\    {{"engine": "wasmtime", "phase": "warmup"}},
        \\    {{"engine": "wasmtime", "phase": "warmup"}},
        \\    {{"engine": "wamr", "phase": "warmup"}},
        \\    {{"engine": "wamr", "phase": "profile"}},
        \\    {{"engine": "wasmtime", "phase": "profile"}},
        \\    {{"engine": "wasmtime", "phase": "profile"}},
        \\    {{"engine": "wamr", "phase": "profile"}}
        \\  ],
        \\  "minimum_attribution_coverage_pct": 99.0,
        \\  "wamr_captures": [
        \\    {{"coverage_pct": 99.96, "mapping": {{"authoritative": true}}}},
        \\    {{"coverage_pct": 99.95, "mapping": {{"authoritative": true}}}}
        \\  ],
        \\  "wasmtime_captures": [
        \\    {{"coverage_pct": 99.9}},
        \\    {{"coverage_pct": 99.9}}
        \\  ],
        \\  "wasm": {{"imported_function_count": 12}},
        \\  "wamr": {{
        \\    "commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        \\    "runtime_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        \\    "compiler_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        \\    "cwasm_sha256": "{s}"
        \\  }},
        \\  "wasmtime": {{
        \\    "version": "43.0.0",
        \\    "sha256": "3333333333333333333333333333333333333333333333333333333333333333"
        \\  }},
        \\  "engines": {{
        \\    "wamr": {{"total_samples": 100, "attributed_samples": 99}},
        \\    "wasmtime": {{"total_samples": 100, "attributed_samples": 98}}
        \\  }},
        \\  "matched_functions": [
        \\    {{
        \\      "local_func": 3,
        \\      "wasm_function_index": 15,
        \\      "wamr": {{"samples": 30}},
        \\      "wasmtime": {{"samples": 28}},
        \\      "class_groups": {{
        \\        "all_alu": {{"wamr_samples": 5, "wasmtime_samples": 4}}
        \\      }}
        \\    }}
        \\  ],
        \\  "retained_analysis_artifacts": {{
        \\    "wamr_cwasm": {{
        \\      "retained": true,
        \\      "path": "wamr-profiled.cwasm.gz",
        \\      "source_sha256": "{s}",
        \\      "source_size_bytes": {d}
        \\    }}
        \\  }}
        \\}}
    ,
        .{ benchmark_sha, cwasm_sha, cwasm_sha, cwasm_sha, cwasm_bytes.len },
    );

    var parsed_rep = try json.parseFromSlice(Value, allocator, template, .{});
    var mf0 = parsed_rep.value.object.getPtr("matched_functions").?.array.items[0].object;
    try mf0.put(allocator, "alu_provenance", assembled);
    parsed_rep.value.object.getPtr("matched_functions").?.array.items[0] = .{ .object = mf0 };

    var top_alu: json.ObjectMap = .empty;
    try top_alu.put(allocator, "schema_version", .{ .integer = provenance.SCHEMA_VERSION });
    try top_alu.put(allocator, "kind", .{ .string = provenance.ANALYSIS_KIND });
    try top_alu.put(allocator, "target_local_func", .{ .integer = 3 });
    try top_alu.put(allocator, "target_wasm_function_index", .{ .integer = 15 });
    try top_alu.put(allocator, "gate", try profile.cloneJsonValue(allocator, assembled.object.get("gate").?));
    try parsed_rep.value.object.put(allocator, "alu_provenance", .{ .object = top_alu });

    return .{
        .report = parsed_rep.value,
        .cwasm_bytes = cwasm_bytes,
        .benchmark_bytes = benchmark_bytes,
        .wasmtime_objdump = wasmtime_objdump,
    };
}

test "reanalyze validates report and artifact hashes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fx = try makeFixture(allocator);
    const summary = try reanalyze(allocator, fx.report, fx.cwasm_bytes, fx.benchmark_bytes, null);
    try std.testing.expectEqual(@as(i64, 3), summary.object.get("local_func").?.integer);
    try std.testing.expectEqual(
        @as(i64, 3),
        summary.object.get("top_uncertain_paths").?.object.get("wamr").?.array.items[0].object.get("samples").?.integer,
    );
    try std.testing.expectEqual(
        @as(i64, 3),
        summary.object.get("unknown_reason_clusters").?.object.get("wamr").?.object.get("uncertain_samples").?.integer,
    );
    try std.testing.expectEqual(
        @as(i64, 3),
        summary.object.get("unknown_reason_clusters").?.object.get("wasmtime").?.object
            .get("clusters").?.object.get("ambiguous_cfg_join").?.object.get("samples").?.integer,
    );

    const summary_with_disasm = try reanalyze(allocator, fx.report, fx.cwasm_bytes, fx.benchmark_bytes, fx.wasmtime_objdump);
    const native = summary_with_disasm.object.get("wasmtime_native_exhaustion").?.object;
    try std.testing.expect(native.get("verified_retained_provenance").?.bool);
    try std.testing.expectEqual(
        @as(i64, 3),
        native.get("reclassified_from_unknown").?.object.get("pure_address").?.object.get("samples").?.integer,
    );
    try std.testing.expectEqualStrings(
        "exhausted",
        native.get("exhaustion_gate").?.object.get("status").?.string,
    );
}

test "reanalyze rejects mismatched binary benchmark or corrupted rows" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fx = try makeFixture(allocator);
    try std.testing.expectError(error.ReanalyzeFailed, reanalyze(allocator, fx.report, "wrong", fx.benchmark_bytes, null));
    try std.testing.expectError(error.ReanalyzeFailed, reanalyze(allocator, fx.report, fx.cwasm_bytes, "{}", null));

    const bad_objdump = std.mem.replaceOwned(
        u8,
        allocator,
        fx.wasmtime_objdump,
        "add\tx8, x1, #0x4",
        "sub\tx8, x1, #0x4",
    ) catch unreachable;
    try std.testing.expectError(
        error.ReanalyzeFailed,
        reanalyze(allocator, fx.report, fx.cwasm_bytes, fx.benchmark_bytes, bad_objdump),
    );
    try std.testing.expect(std.mem.indexOf(u8, lastErrorMessage(), "differs from native disassembly") != null);

    var bad_rep = try profile.cloneJsonValue(allocator, fx.report);
    var mf0 = bad_rep.object.getPtr("matched_functions").?.array.items[0].object;
    var alu_p = mf0.getPtr("alu_provenance").?.object;
    var wamr_p = alu_p.getPtr("wamr").?.object;
    var cgu = wamr_p.getPtr("common_gating_universe").?.object;
    try cgu.getPtr("sampled_instructions").?.array.items[0].object.put(allocator, "samples", .{ .integer = 99 });
    wamr_p.put(allocator, "common_gating_universe", .{ .object = cgu }) catch unreachable;
    alu_p.put(allocator, "wamr", .{ .object = wamr_p }) catch unreachable;
    mf0.put(allocator, "alu_provenance", .{ .object = alu_p }) catch unreachable;
    bad_rep.object.getPtr("matched_functions").?.array.items[0] = .{ .object = mf0 };

    try std.testing.expectError(error.ReanalyzeFailed, reanalyze(allocator, bad_rep, fx.cwasm_bytes, fx.benchmark_bytes, null));
}
