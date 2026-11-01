//! Summarization, report validation, budget evaluation, and markdown rendering for WASI thread benchmarks.
const std = @import("std");
const benchmark_schema = @import("benchmark_schema.zig");
const benchmark_stats = @import("benchmark_stats.zig");
const spec = @import("wasi_thread_bench_spec.zig");
const sizing = @import("wasi_thread_bench_sizing.zig");
const runner = @import("wasi_thread_bench_runner.zig");

pub const Allocator = spec.Allocator;
pub const Value = spec.Value;
pub const ObjectMap = spec.ObjectMap;
pub const Array = spec.Array;
pub const HarnessError = spec.HarnessError;
const fail = spec.fail;
const failData = spec.failData;
const require = spec.require;

const SummaryGroupKey = struct {
    revision: []const u8,
    pair_kind: []const u8,
    pair_key: []const u8,
    condition: []const u8,

    fn eq(a: SummaryGroupKey, b: SummaryGroupKey) bool {
        return std.mem.eql(u8, a.revision, b.revision) and
            std.mem.eql(u8, a.pair_kind, b.pair_kind) and
            std.mem.eql(u8, a.pair_key, b.pair_key) and
            std.mem.eql(u8, a.condition, b.condition);
    }

    fn lessThan(_: void, a: SummaryGroupKey, b: SummaryGroupKey) bool {
        const c1 = std.mem.order(u8, a.revision, b.revision);
        if (c1 != .eq) return c1 == .lt;
        const c2 = std.mem.order(u8, a.pair_kind, b.pair_kind);
        if (c2 != .eq) return c2 == .lt;
        const c3 = std.mem.order(u8, a.pair_key, b.pair_key);
        if (c3 != .eq) return c3 == .lt;
        return std.mem.order(u8, a.condition, b.condition) == .lt;
    }
};

pub fn summarize(allocator: Allocator, records: []const ObjectMap) HarnessError![]ObjectMap {
    var keys: std.ArrayList(SummaryGroupKey) = .empty;
    for (records) |rec| {
        if (!std.mem.eql(u8, rec.get("phase").?.string, "measure")) continue;
        const k: SummaryGroupKey = .{
            .revision = rec.get("revision").?.string,
            .pair_kind = rec.get("pair_kind").?.string,
            .pair_key = rec.get("pair_key").?.string,
            .condition = rec.get("condition").?.string,
        };
        var found = false;
        for (keys.items) |ek| {
            if (SummaryGroupKey.eq(k, ek)) {
                found = true;
                break;
            }
        }
        if (!found) try keys.append(allocator, k);
    }
    std.mem.sort(SummaryGroupKey, keys.items, {}, SummaryGroupKey.lessThan);

    var summaries: std.ArrayList(ObjectMap) = .empty;
    for (keys.items) |k| {
        var elapsed: std.ArrayList(i64) = .empty;
        var host_wall: std.ArrayList(i64) = .empty;
        var throughput: std.ArrayList(f64) = .empty;
        var per_thread: std.ArrayList(f64) = .empty;
        var metric_kind: []const u8 = "";

        for (records) |rec| {
            if (!std.mem.eql(u8, rec.get("phase").?.string, "measure")) continue;
            const rk: SummaryGroupKey = .{
                .revision = rec.get("revision").?.string,
                .pair_kind = rec.get("pair_kind").?.string,
                .pair_key = rec.get("pair_key").?.string,
                .condition = rec.get("condition").?.string,
            };
            if (SummaryGroupKey.eq(k, rk)) {
                metric_kind = rec.get("metric_kind").?.string;
                try elapsed.append(allocator, rec.get("elapsed_ns").?.integer);
                try host_wall.append(allocator, rec.get("host_wall_elapsed_ns").?.integer);
                try throughput.append(allocator, spec.valueAsF64(rec.get("throughput_ops_per_second").?).?);
                try per_thread.append(allocator, spec.valueAsF64(rec.get("per_thread_ops_per_second").?).?);
            }
        }

        var item: ObjectMap = .empty;
        try item.put(allocator, "revision", .{ .string = k.revision });
        try item.put(allocator, "pair_kind", .{ .string = k.pair_kind });
        try item.put(allocator, "pair_key", .{ .string = k.pair_key });
        try item.put(allocator, "condition", .{ .string = k.condition });
        try item.put(allocator, "metric_kind", .{ .string = metric_kind });
        try item.put(allocator, "elapsed", try spec.sampleStatsI64Value(allocator, elapsed.items, "samples_ns"));
        try item.put(allocator, "host_wall", try spec.sampleStatsI64Value(allocator, host_wall.items, "samples_ns"));
        try item.put(allocator, "throughput", try spec.sampleStatsF64Value(allocator, throughput.items, "samples_ops_per_second"));
        try item.put(allocator, "per_thread_throughput", try spec.sampleStatsF64Value(allocator, per_thread.items, "samples_ops_per_second"));
        try summaries.append(allocator, item);
    }
    return summaries.toOwnedSlice(allocator);
}

const PairedCellKey = struct {
    revision: []const u8,
    pair_kind: []const u8,
    pair_key: []const u8,
    pair_index: i64,

    fn eq(a: PairedCellKey, b: PairedCellKey) bool {
        return std.mem.eql(u8, a.revision, b.revision) and
            std.mem.eql(u8, a.pair_kind, b.pair_kind) and
            std.mem.eql(u8, a.pair_key, b.pair_key) and
            a.pair_index == b.pair_index;
    }
};

const PairedGroupKey = struct {
    revision: []const u8,
    pair_kind: []const u8,
    pair_key: []const u8,
    left: []const u8,
    right: []const u8,

    fn eq(a: PairedGroupKey, b: PairedGroupKey) bool {
        return std.mem.eql(u8, a.revision, b.revision) and
            std.mem.eql(u8, a.pair_kind, b.pair_kind) and
            std.mem.eql(u8, a.pair_key, b.pair_key) and
            std.mem.eql(u8, a.left, b.left) and
            std.mem.eql(u8, a.right, b.right);
    }

    fn lessThan(_: void, a: PairedGroupKey, b: PairedGroupKey) bool {
        const c1 = std.mem.order(u8, a.revision, b.revision);
        if (c1 != .eq) return c1 == .lt;
        const c2 = std.mem.order(u8, a.pair_kind, b.pair_kind);
        if (c2 != .eq) return c2 == .lt;
        const c3 = std.mem.order(u8, a.pair_key, b.pair_key);
        if (c3 != .eq) return c3 == .lt;
        const c4 = std.mem.order(u8, a.left, b.left);
        if (c4 != .eq) return c4 == .lt;
        return std.mem.order(u8, a.right, b.right) == .lt;
    }
};

pub fn pairedSummaries(allocator: Allocator, records: []const ObjectMap) HarnessError![]ObjectMap {
    const Cell = struct {
        key: PairedCellKey,
        conds: ObjectMap,
    };
    var cells: std.ArrayList(Cell) = .empty;
    for (records) |rec| {
        if (!std.mem.eql(u8, rec.get("phase").?.string, "measure")) continue;
        const ck: PairedCellKey = .{
            .revision = rec.get("revision").?.string,
            .pair_kind = rec.get("pair_kind").?.string,
            .pair_key = rec.get("pair_key").?.string,
            .pair_index = rec.get("pair_index").?.integer,
        };
        const cond = rec.get("condition").?.string;
        var found = false;
        for (cells.items) |*c| {
            if (PairedCellKey.eq(c.key, ck)) {
                try c.conds.put(allocator, cond, .{ .object = rec });
                found = true;
                break;
            }
        }
        if (!found) {
            var m: ObjectMap = .empty;
            try m.put(allocator, cond, .{ .object = rec });
            try cells.append(allocator, .{ .key = ck, .conds = m });
        }
    }

    const Group = struct {
        key: PairedGroupKey,
        elapsed: std.ArrayList(f64),
        throughput: std.ArrayList(f64),
    };
    var groups: std.ArrayList(Group) = .empty;

    for (cells.items) |c| {
        if (c.conds.count() != 2) {
            return fail("incomplete measured pair: {s}/{s}", .{ c.key.revision, c.key.pair_key });
        }
        var it = c.conds.iterator();
        const first_rec = it.next().?.value_ptr.object;
        const second_rec = it.next().?.value_ptr.object;
        const left = first_rec.get("pair_left").?.string;
        const right = first_rec.get("pair_right").?.string;
        if (!std.mem.eql(u8, second_rec.get("pair_left").?.string, left) or
            !std.mem.eql(u8, second_rec.get("pair_right").?.string, right))
        {
            return fail("inconsistent pair direction: {s}/{s}", .{ c.key.revision, c.key.pair_key });
        }
        const left_val = c.conds.get(left) orelse return fail("pair conditions do not match direction: {s}/{s}", .{ c.key.revision, c.key.pair_key });
        const right_val = c.conds.get(right) orelse return fail("pair conditions do not match direction: {s}/{s}", .{ c.key.revision, c.key.pair_key });
        const left_rec = left_val.object;
        const right_rec = right_val.object;

        const el_ratio = @as(f64, @floatFromInt(right_rec.get("elapsed_ns").?.integer)) /
            @as(f64, @floatFromInt(left_rec.get("elapsed_ns").?.integer));
        const th_ratio = spec.valueAsF64(right_rec.get("throughput_ops_per_second").?).? /
            spec.valueAsF64(left_rec.get("throughput_ops_per_second").?).?;

        const gk: PairedGroupKey = .{
            .revision = c.key.revision,
            .pair_kind = c.key.pair_kind,
            .pair_key = c.key.pair_key,
            .left = left,
            .right = right,
        };
        var found = false;
        for (groups.items) |*g| {
            if (PairedGroupKey.eq(g.key, gk)) {
                try g.elapsed.append(allocator, el_ratio);
                try g.throughput.append(allocator, th_ratio);
                found = true;
                break;
            }
        }
        if (!found) {
            var el: std.ArrayList(f64) = .empty;
            var th: std.ArrayList(f64) = .empty;
            try el.append(allocator, el_ratio);
            try th.append(allocator, th_ratio);
            try groups.append(allocator, .{ .key = gk, .elapsed = el, .throughput = th });
        }
    }

    std.mem.sort(Group, groups.items, {}, struct {
        fn lt(_: void, a: Group, b: Group) bool {
            return PairedGroupKey.lessThan({}, a.key, b.key);
        }
    }.lt);

    var result: std.ArrayList(ObjectMap) = .empty;
    for (groups.items) |g| {
        const el_stats = try spec.sampleStatsF64Value(allocator, g.elapsed.items, "samples");
        const th_stats = try spec.sampleStatsF64Value(allocator, g.throughput.items, "samples");
        const el_med = spec.valueAsF64(el_stats.object.get("median").?).?;
        const th_med = spec.valueAsF64(th_stats.object.get("median").?).?;

        var item: ObjectMap = .empty;
        try item.put(allocator, "revision", .{ .string = g.key.revision });
        try item.put(allocator, "pair_kind", .{ .string = g.key.pair_kind });
        try item.put(allocator, "pair_key", .{ .string = g.key.pair_key });
        try item.put(allocator, "left", .{ .string = g.key.left });
        try item.put(allocator, "right", .{ .string = g.key.right });
        try item.put(allocator, "elapsed_right_over_left", el_stats);
        try item.put(allocator, "throughput_right_over_left", th_stats);
        try item.put(allocator, "median_elapsed_delta_pct", .{ .float = (el_med - 1.0) * 100.0 });
        try item.put(allocator, "median_throughput_delta_pct", .{ .float = (th_med - 1.0) * 100.0 });
        try result.append(allocator, item);
    }
    return result.toOwnedSlice(allocator);
}

pub fn comparisonSummaries(allocator: Allocator, records: []const ObjectMap) HarnessError![]ObjectMap {
    var has_baseline = false;
    var has_candidate = false;
    var has_other = false;
    for (records) |rec| {
        const rev = rec.get("revision").?.string;
        if (std.mem.eql(u8, rev, "baseline")) has_baseline = true else if (std.mem.eql(u8, rev, "candidate")) has_candidate = true else has_other = true;
    }
    if (!has_baseline and has_candidate and !has_other) return &.{};
    if (!has_baseline or !has_candidate or has_other) {
        return fail("comparison records have incomplete revision roles", .{});
    }

    const CellKey = struct {
        pair_kind: []const u8,
        pair_key: []const u8,
        condition: []const u8,
        pair_index: i64,

        fn eq(a: @This(), b: @This()) bool {
            return std.mem.eql(u8, a.pair_kind, b.pair_kind) and
                std.mem.eql(u8, a.pair_key, b.pair_key) and
                std.mem.eql(u8, a.condition, b.condition) and
                a.pair_index == b.pair_index;
        }
    };
    const Cell = struct {
        key: CellKey,
        revs: ObjectMap,
    };
    var cells: std.ArrayList(Cell) = .empty;
    for (records) |rec| {
        if (!std.mem.eql(u8, rec.get("phase").?.string, "measure")) continue;
        const ck: CellKey = .{
            .pair_kind = rec.get("pair_kind").?.string,
            .pair_key = rec.get("pair_key").?.string,
            .condition = rec.get("condition").?.string,
            .pair_index = rec.get("pair_index").?.integer,
        };
        const rev = rec.get("revision").?.string;
        var found = false;
        for (cells.items) |*c| {
            if (CellKey.eq(c.key, ck)) {
                if (c.revs.contains(rev)) {
                    return fail("duplicate revision sample: {s}/{s}/{s}/{d}", .{ rev, ck.pair_key, ck.condition, ck.pair_index });
                }
                try c.revs.put(allocator, rev, .{ .object = rec });
                found = true;
                break;
            }
        }
        if (!found) {
            var m: ObjectMap = .empty;
            try m.put(allocator, rev, .{ .object = rec });
            try cells.append(allocator, .{ .key = ck, .revs = m });
        }
    }

    const GroupKey = struct {
        pair_kind: []const u8,
        pair_key: []const u8,
        condition: []const u8,
        metric_kind: []const u8,

        fn eq(a: @This(), b: @This()) bool {
            return std.mem.eql(u8, a.pair_kind, b.pair_kind) and
                std.mem.eql(u8, a.pair_key, b.pair_key) and
                std.mem.eql(u8, a.condition, b.condition) and
                std.mem.eql(u8, a.metric_kind, b.metric_kind);
        }

        fn lessThan(_: void, a: @This(), b: @This()) bool {
            const c1 = std.mem.order(u8, a.pair_kind, b.pair_kind);
            if (c1 != .eq) return c1 == .lt;
            const c2 = std.mem.order(u8, a.pair_key, b.pair_key);
            if (c2 != .eq) return c2 == .lt;
            const c3 = std.mem.order(u8, a.condition, b.condition);
            if (c3 != .eq) return c3 == .lt;
            return std.mem.order(u8, a.metric_kind, b.metric_kind) == .lt;
        }
    };
    const Group = struct {
        key: GroupKey,
        elapsed: std.ArrayList(f64),
        throughput: std.ArrayList(f64),
    };
    var groups: std.ArrayList(Group) = .empty;

    for (cells.items) |c| {
        const base_v = c.revs.get("baseline") orelse return fail("incomplete revision pair: {s}/{s}", .{ c.key.pair_key, c.key.condition });
        const cand_v = c.revs.get("candidate") orelse return fail("incomplete revision pair: {s}/{s}", .{ c.key.pair_key, c.key.condition });
        const baseline = base_v.object;
        const candidate = cand_v.object;
        const b_mk = baseline.get("metric_kind").?.string;
        const c_mk = candidate.get("metric_kind").?.string;
        if (!std.mem.eql(u8, b_mk, c_mk)) {
            return fail("mixed metric kind: {s}/{s}", .{ c.key.pair_key, c.key.condition });
        }
        const el_ratio = @as(f64, @floatFromInt(candidate.get("elapsed_ns").?.integer)) /
            @as(f64, @floatFromInt(baseline.get("elapsed_ns").?.integer));
        const th_ratio = spec.valueAsF64(candidate.get("throughput_ops_per_second").?).? /
            spec.valueAsF64(baseline.get("throughput_ops_per_second").?).?;

        const gk: GroupKey = .{
            .pair_kind = c.key.pair_kind,
            .pair_key = c.key.pair_key,
            .condition = c.key.condition,
            .metric_kind = b_mk,
        };
        var found = false;
        for (groups.items) |*g| {
            if (GroupKey.eq(g.key, gk)) {
                try g.elapsed.append(allocator, el_ratio);
                try g.throughput.append(allocator, th_ratio);
                found = true;
                break;
            }
        }
        if (!found) {
            var el: std.ArrayList(f64) = .empty;
            var th: std.ArrayList(f64) = .empty;
            try el.append(allocator, el_ratio);
            try th.append(allocator, th_ratio);
            try groups.append(allocator, .{ .key = gk, .elapsed = el, .throughput = th });
        }
    }

    std.mem.sort(Group, groups.items, {}, struct {
        fn lt(_: void, a: Group, b: Group) bool {
            return GroupKey.lessThan({}, a.key, b.key);
        }
    }.lt);

    var result: std.ArrayList(ObjectMap) = .empty;
    for (groups.items) |g| {
        const el_stats = try spec.positionBalancedRatioStats(allocator, g.elapsed.items, "samples");
        const th_stats = try spec.positionBalancedRatioStats(allocator, g.throughput.items, "samples");
        const el_med = spec.valueAsF64(el_stats.get("median").?).?;
        const th_med = spec.valueAsF64(th_stats.get("median").?).?;

        var item: ObjectMap = .empty;
        try item.put(allocator, "pair_kind", .{ .string = g.key.pair_kind });
        try item.put(allocator, "pair_key", .{ .string = g.key.pair_key });
        try item.put(allocator, "condition", .{ .string = g.key.condition });
        try item.put(allocator, "metric_kind", .{ .string = g.key.metric_kind });
        try item.put(allocator, "baseline", .{ .string = "baseline" });
        try item.put(allocator, "candidate", .{ .string = "candidate" });
        try item.put(allocator, "elapsed_candidate_over_baseline", .{ .object = el_stats });
        try item.put(allocator, "throughput_candidate_over_baseline", .{ .object = th_stats });
        try item.put(allocator, "median_elapsed_delta_pct", .{ .float = (el_med - 1.0) * 100.0 });
        try item.put(allocator, "median_throughput_delta_pct", .{ .float = (th_med - 1.0) * 100.0 });
        try result.append(allocator, item);
    }
    return result.toOwnedSlice(allocator);
}

pub fn ratioOfRatiosSummaries(allocator: Allocator, records: []const ObjectMap) HarnessError![]ObjectMap {
    var has_baseline = false;
    var has_candidate = false;
    var has_other = false;
    for (records) |rec| {
        const rev = rec.get("revision").?.string;
        if (std.mem.eql(u8, rev, "baseline")) has_baseline = true else if (std.mem.eql(u8, rev, "candidate")) has_candidate = true else has_other = true;
    }
    if (!has_baseline and has_candidate and !has_other) return &.{};
    if (!has_baseline or !has_candidate or has_other) {
        return fail("ratio-of-ratios records have incomplete revision roles", .{});
    }

    const CellKey = struct {
        pair_kind: []const u8,
        pair_key: []const u8,
        pair_index: i64,

        fn eq(a: @This(), b: @This()) bool {
            return std.mem.eql(u8, a.pair_kind, b.pair_kind) and
                std.mem.eql(u8, a.pair_key, b.pair_key) and
                a.pair_index == b.pair_index;
        }
    };
    const Cell = struct {
        key: CellKey,
        revs: ObjectMap, // revision -> ObjectMap(condition -> record)
    };
    var cells: std.ArrayList(Cell) = .empty;
    for (records) |rec| {
        if (!std.mem.eql(u8, rec.get("phase").?.string, "measure")) continue;
        const ck: CellKey = .{
            .pair_kind = rec.get("pair_kind").?.string,
            .pair_key = rec.get("pair_key").?.string,
            .pair_index = rec.get("pair_index").?.integer,
        };
        const rev = rec.get("revision").?.string;
        const cond = rec.get("condition").?.string;
        var found = false;
        for (cells.items) |*c| {
            if (CellKey.eq(c.key, ck)) {
                if (c.revs.getPtr(rev)) |conds_v| {
                    if (conds_v.object.contains(cond)) {
                        return fail("duplicate ratio-of-ratios sample: {s}/{s}/{s}/{d}", .{ rev, ck.pair_key, cond, ck.pair_index });
                    }
                    try conds_v.object.put(allocator, cond, .{ .object = rec });
                } else {
                    var cm: ObjectMap = .empty;
                    try cm.put(allocator, cond, .{ .object = rec });
                    try c.revs.put(allocator, rev, .{ .object = cm });
                }
                found = true;
                break;
            }
        }
        if (!found) {
            var cm: ObjectMap = .empty;
            try cm.put(allocator, cond, .{ .object = rec });
            var rm: ObjectMap = .empty;
            try rm.put(allocator, rev, .{ .object = cm });
            try cells.append(allocator, .{ .key = ck, .revs = rm });
        }
    }

    const GroupKey = struct {
        pair_kind: []const u8,
        pair_key: []const u8,
        left: []const u8,
        right: []const u8,

        fn eq(a: @This(), b: @This()) bool {
            return std.mem.eql(u8, a.pair_kind, b.pair_kind) and
                std.mem.eql(u8, a.pair_key, b.pair_key) and
                std.mem.eql(u8, a.left, b.left) and
                std.mem.eql(u8, a.right, b.right);
        }

        fn lessThan(_: void, a: @This(), b: @This()) bool {
            const c1 = std.mem.order(u8, a.pair_kind, b.pair_kind);
            if (c1 != .eq) return c1 == .lt;
            const c2 = std.mem.order(u8, a.pair_key, b.pair_key);
            if (c2 != .eq) return c2 == .lt;
            const c3 = std.mem.order(u8, a.left, b.left);
            if (c3 != .eq) return c3 == .lt;
            return std.mem.order(u8, a.right, b.right) == .lt;
        }
    };
    const Group = struct {
        key: GroupKey,
        elapsed: std.ArrayList(f64),
        throughput: std.ArrayList(f64),
    };
    var groups: std.ArrayList(Group) = .empty;

    for (cells.items) |c| {
        const base_v = c.revs.get("baseline") orelse return fail("incomplete revisions for pair {s}", .{c.key.pair_key});
        const cand_v = c.revs.get("candidate") orelse return fail("incomplete revisions for pair {s}", .{c.key.pair_key});
        const baseline = base_v.object;
        const candidate = cand_v.object;
        if (baseline.count() == 0) return fail("incomplete internal pair for {s}", .{c.key.pair_key});
        var b_it = baseline.iterator();
        const first = b_it.next().?.value_ptr.object;
        const left = first.get("pair_left").?.string;
        const right = first.get("pair_right").?.string;
        if (baseline.count() != 2 or !baseline.contains(left) or !baseline.contains(right) or
            candidate.count() != 2 or !candidate.contains(left) or !candidate.contains(right))
        {
            return fail("incomplete internal pair for {s}", .{c.key.pair_key});
        }
        const b_left = baseline.get(left).?.object;
        const b_right = baseline.get(right).?.object;
        const c_left = candidate.get(left).?.object;
        const c_right = candidate.get(right).?.object;

        const base_el = @as(f64, @floatFromInt(b_right.get("elapsed_ns").?.integer)) /
            @as(f64, @floatFromInt(b_left.get("elapsed_ns").?.integer));
        const cand_el = @as(f64, @floatFromInt(c_right.get("elapsed_ns").?.integer)) /
            @as(f64, @floatFromInt(c_left.get("elapsed_ns").?.integer));
        const base_th = spec.valueAsF64(b_right.get("throughput_ops_per_second").?).? /
            spec.valueAsF64(b_left.get("throughput_ops_per_second").?).?;
        const cand_th = spec.valueAsF64(c_right.get("throughput_ops_per_second").?).? /
            spec.valueAsF64(c_left.get("throughput_ops_per_second").?).?;

        const gk: GroupKey = .{
            .pair_kind = c.key.pair_kind,
            .pair_key = c.key.pair_key,
            .left = left,
            .right = right,
        };
        var found = false;
        for (groups.items) |*g| {
            if (GroupKey.eq(g.key, gk)) {
                try g.elapsed.append(allocator, cand_el / base_el);
                try g.throughput.append(allocator, cand_th / base_th);
                found = true;
                break;
            }
        }
        if (!found) {
            var el: std.ArrayList(f64) = .empty;
            var th: std.ArrayList(f64) = .empty;
            try el.append(allocator, cand_el / base_el);
            try th.append(allocator, cand_th / base_th);
            try groups.append(allocator, .{ .key = gk, .elapsed = el, .throughput = th });
        }
    }

    std.mem.sort(Group, groups.items, {}, struct {
        fn lt(_: void, a: Group, b: Group) bool {
            return GroupKey.lessThan({}, a.key, b.key);
        }
    }.lt);

    var result: std.ArrayList(ObjectMap) = .empty;
    for (groups.items) |g| {
        const el_stats = try spec.positionBalancedRatioStats(allocator, g.elapsed.items, "samples");
        const th_stats = try spec.positionBalancedRatioStats(allocator, g.throughput.items, "samples");
        const el_med = spec.valueAsF64(el_stats.get("median").?).?;
        const th_med = spec.valueAsF64(th_stats.get("median").?).?;

        var item: ObjectMap = .empty;
        try item.put(allocator, "pair_kind", .{ .string = g.key.pair_kind });
        try item.put(allocator, "pair_key", .{ .string = g.key.pair_key });
        try item.put(allocator, "left", .{ .string = g.key.left });
        try item.put(allocator, "right", .{ .string = g.key.right });
        try item.put(allocator, "baseline", .{ .string = "baseline" });
        try item.put(allocator, "candidate", .{ .string = "candidate" });
        try item.put(allocator, "elapsed_ratio_of_ratios", .{ .object = el_stats });
        try item.put(allocator, "throughput_ratio_of_ratios", .{ .object = th_stats });
        try item.put(allocator, "median_elapsed_delta_pct", .{ .float = (el_med - 1.0) * 100.0 });
        try item.put(allocator, "median_throughput_delta_pct", .{ .float = (th_med - 1.0) * 100.0 });
        try result.append(allocator, item);
    }
    return result.toOwnedSlice(allocator);
}

pub fn validateReport(allocator: Allocator, document: Value) HarnessError!void {
    try require(document == .object, "report must be a JSON object", .{});
    const doc = document.object;
    const sv = doc.get("schema_version") orelse return failData("schema_version", .{});
    const kind = doc.get("kind") orelse return failData("kind", .{});
    try require(
        sv == .integer and sv.integer == spec.REPORT_SCHEMA_VERSION and
            kind == .string and std.mem.eql(u8, kind.string, spec.KIND),
        "schema_version/kind",
        .{},
    );
    for ([_][]const u8{ "metadata", "plan", "records", "summaries" }) |k| {
        try require(doc.contains(k), "report missing {s}", .{k});
    }
    const meta_v = doc.get("metadata").?;
    const plan_v = doc.get("plan").?;
    const records_v = doc.get("records").?;
    const summaries_v = doc.get("summaries").?;
    try require(meta_v == .object and plan_v == .object and records_v == .array and summaries_v == .array, "report top-level types", .{});

    const metadata = meta_v.object;
    for ([_][]const u8{ "commit", "tracked_diff_sha256", "build_source_sha256", "collected_at", "host", "tools" }) |k| {
        try require(metadata.contains(k), "metadata.{s}", .{k});
    }
    try require(metadata.get("commit").? == .string and spec.isHexCommit(metadata.get("commit").?.string), "metadata.commit", .{});
    for ([_][]const u8{ "tracked_diff_sha256", "build_source_sha256", "fixture_set_sha256", "plan_sha256", "measurement_plan_sha256" }) |k| {
        const v = metadata.get(k) orelse return failData("metadata.{s}", .{k});
        try require(v == .string and spec.isHexSha256(v.string), "metadata.{s}", .{k});
    }
    const pid_v = metadata.get("platform_id") orelse return failData("metadata.platform_id", .{});
    try require(pid_v == .string and pid_v.string.len > 0, "metadata.platform_id", .{});
    const mpv = metadata.get("measurement_plan_version") orelse return failData("metadata.measurement_plan_version", .{});
    try require(mpv == .integer and mpv.integer == spec.MEASUREMENT_PLAN_IDENTITY_VERSION, "metadata.measurement_plan_version", .{});

    const plan = plan_v.object;
    const warmups_v = plan.get("warmups") orelse return failData("plan.warmups", .{});
    const samples_v = plan.get("samples") orelse return failData("plan.samples", .{});
    const min_int_v = plan.get("minimum_timed_interval_ns") orelse return failData("plan.minimum_timed_interval_ns", .{});
    try require(warmups_v == .integer and warmups_v.integer >= 0, "plan.warmups", .{});
    try require(samples_v == .integer and samples_v.integer > 0, "plan.samples", .{});
    try require(
        min_int_v == .integer and min_int_v.integer >= @as(i64, @intFromFloat(spec.MIN_TIMED_INTERVAL_MS * 1_000_000.0)),
        "plan.minimum_timed_interval_ns",
        .{},
    );
    const warmups = warmups_v.integer;
    const samples = samples_v.integer;
    const min_interval_ns = min_int_v.integer;

    const preflight_plan_v = plan.get("scheduler_barrier_preflight") orelse return failData("plan.scheduler_barrier_preflight", .{});
    try require(preflight_plan_v == .object, "plan.scheduler_barrier_preflight", .{});
    const preflight_plan = preflight_plan_v.object;

    try require(
        try spec.jsonEquals(allocator, plan.get("cpu_placement") orelse .null, try spec.cpuPlacementPolicyValue(allocator)),
        "plan.cpu_placement",
        .{},
    );
    try require(
        try spec.jsonEquals(allocator, plan.get("pair_execution") orelse .null, try spec.pairExecutionPolicyValue(allocator)),
        "plan.pair_execution",
        .{},
    );
    const rap_v = plan.get("revision_artifact_policy") orelse return failData("plan.revision_artifact_policy", .{});
    try require(rap_v == .string and std.mem.eql(u8, rap_v.string, spec.REVISION_ARTIFACT_POLICY), "plan.revision_artifact_policy", .{});
    const fsp_v = plan.get("fixture_source_policy") orelse return failData("plan.fixture_source_policy", .{});
    try require(fsp_v == .string and std.mem.eql(u8, fsp_v.string, spec.FIXTURE_SOURCE_POLICY), "plan.fixture_source_policy", .{});
    try require(
        try spec.jsonEquals(allocator, plan.get("guest_clock_policy") orelse .null, try spec.guestClockPolicyValue(allocator)),
        "plan.guest_clock_policy",
        .{},
    );

    const pf_enabled_v = preflight_plan.get("enabled") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_rule_v = preflight_plan.get("acceptance_rule") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_mode_v = preflight_plan.get("mode") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_wl_v = preflight_plan.get("workload") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_cid_v = preflight_plan.get("clock_id") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_mult_v = preflight_plan.get("iteration_multiplier") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_ppt_v = preflight_plan.get("probes_per_thread") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_lim_v = preflight_plan.get("timing_overhead_ratio_limit") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_tb_v = preflight_plan.get("target_barrier_ns") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_tri_v = preflight_plan.get("target_required_interval_ns") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_mih_v = preflight_plan.get("minimum_interval_headroom_ns") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const pf_mab_v = preflight_plan.get("maximum_accepted_barrier_ns") orelse return failData("plan.scheduler_barrier_preflight policy", .{});
    const max_acc_barrier = runner.maximumPreflightBarrierNs(min_interval_ns) catch return failData("plan.scheduler_barrier_preflight policy", .{});
    try require(
        pf_enabled_v == .bool and
            pf_rule_v == .string and pf_rule_v.string.len > 0 and
            pf_mode_v == .string and std.mem.eql(u8, pf_mode_v.string, "aot") and
            pf_wl_v == .string and std.mem.eql(u8, pf_wl_v.string, "hot") and
            pf_cid_v == .string and std.mem.eql(u8, pf_cid_v.string, spec.WASI_MONOTONIC_CLOCK_ID) and
            pf_mult_v == .string and std.mem.eql(u8, pf_mult_v.string, spec.TRUSTED_BARRIER_PREFLIGHT_ITERATION_MULTIPLIER) and
            pf_ppt_v == .integer and pf_ppt_v.integer == spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD and
            (spec.valueAsF64(pf_lim_v) orelse -1.0) == spec.TIMING_OVERHEAD_RATIO_LIMIT and
            pf_tb_v == .integer and pf_tb_v.integer == spec.TARGET_BARRIER_NS and
            pf_tri_v == .integer and pf_tri_v.integer == spec.TARGET_BARRIER_REQUIRED_INTERVAL_NS and
            pf_mih_v == .integer and pf_mih_v.integer == min_interval_ns - spec.TARGET_BARRIER_REQUIRED_INTERVAL_NS and
            pf_mab_v == .integer and pf_mab_v.integer == max_acc_barrier,
        "plan.scheduler_barrier_preflight policy",
        .{},
    );

    const modes_v = plan.get("modes") orelse return failData("plan.modes", .{});
    const tc_v = plan.get("thread_counts") orelse return failData("plan.thread_counts", .{});
    const iters_v = plan.get("iterations") orelse return failData("plan.iterations modes", .{});
    try require(modes_v == .array and tc_v == .array and iters_v == .object, "plan.iterations modes", .{});

    var modes: std.ArrayList([]const u8) = .empty;
    for (modes_v.array.items) |mv| {
        try require(mv == .string, "plan.iterations modes", .{});
        try modes.append(allocator, mv.string);
    }
    try require(iters_v.object.count() == modes.items.len, "plan.iterations modes", .{});
    for (modes.items) |m| try require(iters_v.object.contains(m), "plan.iterations modes", .{});

    var thread_counts: std.ArrayList(i64) = .empty;
    for (tc_v.array.items) |tv| {
        try require(tv == .integer, "plan.thread_counts", .{});
        try thread_counts.append(allocator, tv.integer);
    }

    for (modes.items) |mode| {
        const values_v = iters_v.object.get(mode).?;
        try require(values_v == .object, "plan.iterations.{s} workloads", .{mode});
        const values = values_v.object;
        const expected_wl_count: usize = if (std.mem.eql(u8, mode, "aot")) 6 else 5;
        try require(values.count() == expected_wl_count, "plan.iterations.{s} workloads", .{mode});
        const sh_v = values.get("single-hot") orelse return failData("plan.iterations.{s} workloads", .{mode});
        const sh_u = spec.valueAsU64(sh_v) orelse return failData("plan.iterations.{s}.single-hot", .{mode});
        try require(sh_u > 0, "plan.iterations.{s}.single-hot", .{mode});
        const wls = [_][]const u8{ "hot", "atomic", "wait-notify", "spawn-join" };
        for (wls) |wl| {
            const counts_v = values.get(wl) orelse return failData("plan.iterations.{s} workloads", .{mode});
            try require(counts_v == .object and counts_v.object.count() == thread_counts.items.len, "plan.iterations.{s}.{s} threads", .{ mode, wl });
            for (thread_counts.items) |tc| {
                const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                const cv = counts_v.object.get(tc_str) orelse return failData("plan.iterations.{s}.{s} threads", .{ mode, wl });
                const cu = spec.valueAsU64(cv) orelse return failData("plan.iterations.{s}.{s} counts", .{ mode, wl });
                try require(cu > 0, "plan.iterations.{s}.{s} counts", .{ mode, wl });
            }
        }
        if (std.mem.eql(u8, mode, "aot")) {
            const counts_v = values.get("cancel-hot") orelse return failData("plan.iterations.aot workloads", .{});
            try require(counts_v == .object and counts_v.object.count() == thread_counts.items.len, "plan.iterations.aot.cancel-hot threads", .{});
            for (thread_counts.items) |tc| {
                const tc_str = try std.fmt.allocPrint(allocator, "{d}", .{tc});
                const cv = counts_v.object.get(tc_str) orelse return failData("plan.iterations.aot.cancel-hot threads", .{});
                const cu = spec.valueAsU64(cv) orelse return failData("plan.iterations.aot.cancel-hot counts", .{});
                try require(cu > 0, "plan.iterations.aot.cancel-hot counts", .{});
            }
        }
    }

    const rev_mode_v = plan.get("revision_mode") orelse return failData("plan.revision_mode", .{});
    const comp_purpose_v = plan.get("comparison_purpose") orelse return failData("plan.comparison_purpose", .{});
    try require(rev_mode_v == .string and comp_purpose_v == .string, "plan.revision_mode", .{});
    const revision_mode = rev_mode_v.string;
    const comparison_purpose = comp_purpose_v.string;

    const revision_roles: []const []const u8 = if (std.mem.eql(u8, revision_mode, "paired-revisions")) blk: {
        try require(
            std.mem.eql(u8, comparison_purpose, "candidate-evaluation") or
                std.mem.eql(u8, comparison_purpose, "noise-calibration"),
            "plan.comparison_purpose",
            .{},
        );
        try require(@rem(samples, 4) == 0, "paired revision samples must be divisible by 4", .{});
        break :blk &spec.REVISION_ROLES;
    } else blk: {
        try require(std.mem.eql(u8, revision_mode, "single-revision-compatibility"), "plan.revision_mode", .{});
        try require(std.mem.eql(u8, comparison_purpose, "single-revision-compatibility"), "plan.comparison_purpose", .{});
        break :blk &spec.SINGLE_REVISION_ROLES;
    };

    const roles_v = plan.get("revision_roles") orelse return failData("plan.revision_roles", .{});
    try require(roles_v == .array and roles_v.array.items.len == revision_roles.len, "plan.revision_roles", .{});
    for (revision_roles, 0..) |r, idx| {
        const item = roles_v.array.items[idx];
        try require(item == .string and std.mem.eql(u8, item.string, r), "plan.revision_roles", .{});
    }

    const expected_pf_count: i64 = if (pf_enabled_v.bool)
        spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD * @as(i64, @intCast(thread_counts.items.len))
    else
        0;
    const pf_pc_v = preflight_plan.get("probe_count") orelse return failData("plan.scheduler_barrier_preflight probe count", .{});
    try require(pf_pc_v == .integer and pf_pc_v.integer == expected_pf_count, "plan.scheduler_barrier_preflight probe count", .{});
    if (pf_enabled_v.bool) {
        var has_aot = false;
        for (modes.items) |m| if (std.mem.eql(u8, m, "aot")) {
            has_aot = true;
        };
        try require(
            std.mem.eql(u8, revision_mode, "paired-revisions") and
                std.mem.eql(u8, comparison_purpose, "noise-calibration") and
                has_aot,
            "trusted quality preflight activation",
            .{},
        );
    }

    const expected_plan_sha = try spec.cacheKey(allocator, plan_v);
    try require(std.mem.eql(u8, metadata.get("plan_sha256").?.string, expected_plan_sha), "metadata.plan_sha256", .{});
    const expected_mplan_sha = try spec.measurementPlanSha256(allocator, plan_v);
    try require(std.mem.eql(u8, metadata.get("measurement_plan_sha256").?.string, expected_mplan_sha), "metadata.measurement_plan_sha256", .{});

    const cpu_placement_v = metadata.get("cpu_placement") orelse return failData("metadata.cpu_placement policy", .{});
    try spec.validateCpuPlacement(allocator, cpu_placement_v, thread_counts.items);
    const cpu_placement = cpu_placement_v.object;

    const revisions_v = metadata.get("revisions") orelse return failData("metadata.revisions", .{});
    try require(revisions_v == .object and revisions_v.object.count() == revision_roles.len, "metadata.revisions", .{});
    const revisions = revisions_v.object;
    for (revision_roles) |r| try require(revisions.contains(r), "metadata.revisions", .{});

    const checkouts_v = metadata.get("revision_checkouts") orelse return failData("metadata.revision_checkouts", .{});
    try require(checkouts_v == .object and checkouts_v.object.count() == revision_roles.len, "metadata.revision_checkouts", .{});
    for (revision_roles) |r| {
        const cv = checkouts_v.object.get(r) orelse return failData("metadata.revision_checkouts", .{});
        try require(cv == .string and cv.string.len > 0, "metadata.revision_checkouts", .{});
    }
    if (std.mem.eql(u8, revision_mode, "paired-revisions")) {
        const b_path = checkouts_v.object.get("baseline").?.string;
        const c_path = checkouts_v.object.get("candidate").?.string;
        try require(!std.mem.eql(u8, b_path, c_path), "paired revisions use the same checkout path", .{});
    }

    const fs_v = metadata.get("fixture_source") orelse return failData("metadata.fixture_source", .{});
    var expected_fs: ObjectMap = .empty;
    try expected_fs.put(allocator, "policy", .{ .string = spec.FIXTURE_SOURCE_POLICY });
    try expected_fs.put(allocator, "role", .{ .string = "candidate" });
    try expected_fs.put(allocator, "checkout", checkouts_v.object.get("candidate").?);
    try require(try spec.jsonEquals(allocator, fs_v, .{ .object = expected_fs }), "metadata.fixture_source", .{});

    for (revision_roles) |role| {
        const rv = revisions.get(role).?;
        try require(rv == .object and rv.object.count() == 7, "metadata.revisions.{s} fields", .{role});
        const robj = rv.object;
        for ([_][]const u8{ "commit", "tracked_diff_sha256", "build_source_sha256", "fixture_set_sha256", "plan_sha256", "host_pair_id", "host_fingerprint_sha256" }) |rk| {
            try require(robj.contains(rk), "metadata.revisions.{s} fields", .{role});
        }
        try require(robj.get("commit").? == .string and spec.isHexCommit(robj.get("commit").?.string), "metadata.revisions.{s}.commit", .{role});
        for ([_][]const u8{ "tracked_diff_sha256", "build_source_sha256", "fixture_set_sha256", "plan_sha256", "host_fingerprint_sha256" }) |sk| {
            try require(robj.get(sk).? == .string and spec.isHexSha256(robj.get(sk).?.string), "metadata.revisions.{s}.{s}", .{ role, sk });
        }
        try require(
            std.mem.eql(u8, robj.get("fixture_set_sha256").?.string, metadata.get("fixture_set_sha256").?.string),
            "mixed fixture identity for {s}",
            .{role},
        );
        try require(
            std.mem.eql(u8, robj.get("plan_sha256").?.string, metadata.get("plan_sha256").?.string),
            "mixed plan identity for {s}",
            .{role},
        );
    }

    const cand_rev = revisions.get("candidate").?.object;
    for ([_][]const u8{ "commit", "tracked_diff_sha256", "build_source_sha256" }) |k| {
        try require(
            std.mem.eql(u8, metadata.get(k).?.string, cand_rev.get(k).?.string),
            "metadata.{s} candidate compatibility alias",
            .{k},
        );
    }

    const host_v = metadata.get("host").?;
    try require(host_v == .object, "metadata.host", .{});
    const host = host_v.object;
    const fp_v = host.get("host_fingerprint") orelse return failData("metadata.host.host_fingerprint", .{});
    try require(fp_v == .object, "metadata.host.host_fingerprint", .{});
    const fp_fields_v = fp_v.object.get("fields") orelse return failData("metadata.host.host_fingerprint.fields", .{});
    try require(fp_fields_v == .object and fp_fields_v.object.count() == benchmark_schema.HOST_FINGERPRINT_FIELDS.len, "host fingerprint fields", .{});
    for (benchmark_schema.HOST_FINGERPRINT_FIELDS) |f| {
        try require(fp_fields_v.object.contains(f), "host fingerprint fields", .{});
    }
    const expected_fp_sha = try spec.cacheKey(allocator, fp_fields_v);
    const actual_fp_sha_v = fp_v.object.get("sha256") orelse return failData("metadata.host.host_fingerprint", .{});
    try require(actual_fp_sha_v == .string and std.mem.eql(u8, actual_fp_sha_v.string, expected_fp_sha), "metadata.host.host_fingerprint", .{});

    const hp_v = metadata.get("host_pair") orelse return failData("metadata.host_pair", .{});
    try require(hp_v == .object and hp_v.object.count() == 3, "metadata.host_pair", .{});
    const hp = hp_v.object;
    const hp_id = hp.get("id") orelse return failData("metadata.host_pair", .{});
    const hp_re = hp.get("runner_environment") orelse return failData("metadata.host_pair", .{});
    const hp_fp = hp.get("host_fingerprint_sha256") orelse return failData("metadata.host_pair", .{});
    try require(hp_id == .string and hp_id.string.len > 0, "metadata.host_pair.id", .{});
    const host_re = host.get("runner_environment") orelse return failData("metadata.host_pair runner environment", .{});
    try require(hp_re == .string and hp_re.string.len > 0 and host_re == .string and std.mem.eql(u8, hp_re.string, host_re.string), "metadata.host_pair runner environment", .{});
    try require(hp_fp == .string and std.mem.eql(u8, hp_fp.string, actual_fp_sha_v.string), "metadata.host_pair fingerprint", .{});

    for (revision_roles) |role| {
        const robj = revisions.get(role).?.object;
        try require(std.mem.eql(u8, robj.get("host_pair_id").?.string, hp_id.string), "mixed host pair for {s}", .{role});
        try require(std.mem.eql(u8, robj.get("host_fingerprint_sha256").?.string, hp_fp.string), "mixed host fingerprint for {s}", .{role});
    }

    if (std.mem.eql(u8, comparison_purpose, "noise-calibration")) {
        const b_rev = revisions.get("baseline").?.object;
        for ([_][]const u8{ "commit", "tracked_diff_sha256", "build_source_sha256" }) |k| {
            try require(std.mem.eql(u8, b_rev.get(k).?.string, cand_rev.get(k).?.string), "noise calibration revision identity", .{});
        }
    }

    const ra_v = metadata.get("revision_artifacts") orelse return failData("metadata.revision_artifacts policy", .{});
    try require(ra_v == .object and ra_v.object.count() == 3, "metadata.revision_artifacts policy", .{});
    const ra = ra_v.object;
    const ra_pol = ra.get("policy") orelse return failData("metadata.revision_artifacts policy", .{});
    const ra_strat = ra.get("strategy") orelse return failData("metadata.revision_artifacts policy", .{});
    const ra_src = ra.get("source_role") orelse return failData("metadata.revision_artifacts policy", .{});
    try require(ra_pol == .string and std.mem.eql(u8, ra_pol.string, spec.REVISION_ARTIFACT_POLICY), "metadata.revision_artifacts policy", .{});
    const expected_strat = if (std.mem.eql(u8, comparison_purpose, "noise-calibration"))
        "shared-exact-build"
    else if (std.mem.eql(u8, revision_mode, "paired-revisions"))
        "independent-builds"
    else
        "single-revision-build";
    try require(ra_strat == .string and std.mem.eql(u8, ra_strat.string, expected_strat), "metadata.revision_artifacts strategy", .{});
    if (std.mem.eql(u8, expected_strat, "shared-exact-build")) {
        try require(ra_src == .string and std.mem.eql(u8, ra_src.string, "baseline"), "metadata.revision_artifacts strategy", .{});
    } else {
        try require(ra_src == .null, "metadata.revision_artifacts strategy", .{});
    }

    const tools_v = metadata.get("tools") orelse return failData("metadata revision artifact roles", .{});
    const aot_art_v = metadata.get("aot_artifacts") orelse return failData("metadata revision artifact roles", .{});
    try require(
        tools_v == .object and tools_v.object.count() == revision_roles.len and
            aot_art_v == .object and aot_art_v.object.count() == revision_roles.len,
        "metadata revision artifact roles",
        .{},
    );
    for (revision_roles) |r| {
        try require(tools_v.object.contains(r) and aot_art_v.object.contains(r), "metadata revision artifact roles", .{});
    }
    if (std.mem.eql(u8, comparison_purpose, "noise-calibration")) {
        try require(
            (try spec.jsonEquals(allocator, tools_v.object.get("baseline").?, tools_v.object.get("candidate").?)) and
                (try spec.jsonEquals(allocator, aot_art_v.object.get("baseline").?, aot_art_v.object.get("candidate").?)),
            "noise calibration exact artifact reuse",
            .{},
        );
    }

    const budget_v = doc.get("budget") orelse return failData("budget", .{});
    try require(budget_v == .object, "budget", .{});
    const budget = budget_v.object;
    const b_status_v = budget.get("status") orelse return failData("budget.status", .{});
    try require(
        b_status_v == .string and
            (std.mem.eql(u8, b_status_v.string, "disabled") or
                std.mem.eql(u8, b_status_v.string, "not-selected") or
                std.mem.eql(u8, b_status_v.string, "passed") or
                std.mem.eql(u8, b_status_v.string, "failed")),
        "budget.status",
        .{},
    );
    if (std.mem.eql(u8, comparison_purpose, "noise-calibration")) {
        const b_path = budget.get("path") orelse return failData("noise calibration must be non-enforcing", .{});
        const b_fails = budget.get("failures") orelse return failData("noise calibration must be non-enforcing", .{});
        try require(
            std.mem.eql(u8, b_status_v.string, "disabled") and
                b_path == .null and
                b_fails == .array and b_fails.array.items.len == 0,
            "noise calibration must be non-enforcing",
            .{},
        );
    }
    if (std.mem.eql(u8, b_status_v.string, "passed") or std.mem.eql(u8, b_status_v.string, "failed")) {
        try require(std.mem.eql(u8, comparison_purpose, "candidate-evaluation"), "budget enforcement requires candidate evaluation", .{});
        const b_rev = revisions.get("baseline").?.object;
        try require(!std.mem.eql(u8, b_rev.get("commit").?.string, cand_rev.get("commit").?.string), "budget enforcement requires distinct revision commits", .{});
        try require(!std.mem.eql(u8, b_rev.get("build_source_sha256").?.string, cand_rev.get("build_source_sha256").?.string), "budget enforcement requires distinct build identities", .{});
    }

    const qp_v = doc.get("quality_preflight") orelse return failData("quality_preflight", .{});
    try require(qp_v == .object, "quality_preflight", .{});
    const qp = qp_v.object;
    try require((qp.get("enabled") orelse .null) == .bool and qp.get("enabled").?.bool == pf_enabled_v.bool, "quality_preflight enabled", .{});
    try require(
        (qp.get("probe_count") orelse .null) == .integer and qp.get("probe_count").?.integer == expected_pf_count and
            (qp.get("probes_per_thread") orelse .null) == .integer and qp.get("probes_per_thread").?.integer == spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD and
            (qp.get("minimum_timed_interval_ns") orelse .null) == .integer and qp.get("minimum_timed_interval_ns").?.integer == min_interval_ns and
            (spec.valueAsF64(qp.get("timing_overhead_ratio_limit") orelse .null) orelse -1.0) == spec.TIMING_OVERHEAD_RATIO_LIMIT and
            (qp.get("maximum_accepted_barrier_ns") orelse .null) == .integer and qp.get("maximum_accepted_barrier_ns").?.integer == max_acc_barrier and
            (qp.get("mode") orelse .null) == .string and std.mem.eql(u8, qp.get("mode").?.string, pf_mode_v.string) and
            (qp.get("workload") orelse .null) == .string and std.mem.eql(u8, qp.get("workload").?.string, pf_wl_v.string) and
            (qp.get("clock_id") orelse .null) == .string and std.mem.eql(u8, qp.get("clock_id").?.string, pf_cid_v.string) and
            (qp.get("iteration_multiplier") orelse .null) == .string and std.mem.eql(u8, qp.get("iteration_multiplier").?.string, pf_mult_v.string) and
            (try spec.jsonEquals(allocator, qp.get("thread_counts") orelse .null, tc_v)) and
            (qp.get("acceptance_rule") orelse .null) == .string and std.mem.eql(u8, qp.get("acceptance_rule").?.string, pf_rule_v.string) and
            (qp.get("host_quiescence_at_start") orelse .null) == .object,
        "quality_preflight policy",
        .{},
    );
    const qp_samples_v = qp.get("samples") orelse return failData("quality_preflight samples", .{});
    try require(qp_samples_v == .array and @as(i64, @intCast(qp_samples_v.array.items.len)) == expected_pf_count, "quality_preflight sample count", .{});
    if (pf_enabled_v.bool) {
        try require((qp.get("status") orelse .null) == .string and std.mem.eql(u8, qp.get("status").?.string, "passed"), "quality_preflight status", .{});
        var overheads: std.ArrayList(f64) = .empty;
        var sample_idx: usize = 0;
        for (thread_counts.items) |tc| {
            var p_idx: i64 = 0;
            while (p_idx < spec.TRUSTED_BARRIER_PREFLIGHT_PROBES_PER_THREAD) : ({
                p_idx += 1;
                sample_idx += 1;
            }) {
                const sv_item = qp_samples_v.array.items[sample_idx];
                try require(sv_item == .object, "quality_preflight sample", .{});
                const sobj = sv_item.object;
                try require(
                    (sobj.get("threads") orelse .null) == .integer and sobj.get("threads").?.integer == tc and
                        (sobj.get("probe_index") orelse .null) == .integer and sobj.get("probe_index").?.integer == p_idx,
                    "quality_preflight probe order",
                    .{},
                );
                const ov = (sobj.get("timing_overhead_ns") orelse return failData("quality_preflight acceptance", .{})).integer;
                const tm = (sobj.get("timed_interval_ns") orelse return failData("quality_preflight acceptance", .{})).integer;
                const rw = (sobj.get("raw_elapsed_ns") orelse return failData("quality_preflight sample", .{})).integer;
                const acc_v = sobj.get("accepted") orelse return failData("quality_preflight acceptance", .{});
                try require(
                    acc_v == .bool and acc_v.bool and runner.preflightSampleAccepted(ov, tm, min_interval_ns),
                    "quality_preflight acceptance",
                    .{},
                );
                const exp_aff = spec.cpuAffinityFor(allocator, cpu_placement, "hot", tc) catch return failData("quality_preflight sample", .{});
                var exp_aff_arr = Array.init(allocator);
                for (exp_aff) |c| try exp_aff_arr.append(.{ .integer = c });
                const base_hot_iters = sizing.iterationCount(iters_v.object, "aot", "hot", tc) catch return failData("quality_preflight sample", .{});
                const exp_iters = base_hot_iters * @as(u64, @intCast(exp_aff.len));
                const s_iters = spec.valueAsU64(sobj.get("iterations") orelse .null) orelse return failData("quality_preflight sample", .{});
                const s_ppm = (sobj.get("timing_overhead_ppm") orelse return failData("quality_preflight sample", .{})).integer;
                const s_ratio = spec.valueAsF64(sobj.get("timing_overhead_ratio") orelse .null) orelse return failData("quality_preflight sample", .{});
                const s_rmin = spec.valueAsF64(sobj.get("ratio_at_minimum_timed_interval") orelse .null) orelse return failData("quality_preflight sample", .{});
                const exp_ppm: i64 = @intCast(@divTrunc(@as(i128, ov) * 1_000_000, @as(i128, rw)));
                try require(
                    (sobj.get("mode") orelse .null) == .string and std.mem.eql(u8, sobj.get("mode").?.string, "aot") and
                        (sobj.get("workload") orelse .null) == .string and std.mem.eql(u8, sobj.get("workload").?.string, "hot") and
                        (sobj.get("clock_id") orelse .null) == .string and std.mem.eql(u8, sobj.get("clock_id").?.string, spec.WASI_MONOTONIC_CLOCK_ID) and
                        (try spec.jsonEquals(allocator, sobj.get("cpu_affinity") orelse .null, .{ .array = exp_aff_arr })) and
                        s_iters == exp_iters and
                        rw == tm + ov and
                        s_ppm == exp_ppm and
                        std.math.approxEqRel(f64, s_ratio, @as(f64, @floatFromInt(ov)) / @as(f64, @floatFromInt(rw)), 1e-9) and
                        std.math.approxEqRel(f64, s_rmin, @as(f64, @floatFromInt(ov)) / @as(f64, @floatFromInt(min_interval_ns + ov)), 1e-9),
                    "quality_preflight sample",
                    .{},
                );
                try overheads.append(allocator, @floatFromInt(ov));
            }
        }
        const ov_stats = benchmark_stats.summarize(allocator, overheads.items) catch return failData("quality_preflight summary", .{});
        var exp_sum: ObjectMap = .empty;
        try exp_sum.put(allocator, "minimum_barrier_ns", .{ .integer = @intFromFloat(ov_stats.min) });
        if (@floor(ov_stats.median) == ov_stats.median) {
            try exp_sum.put(allocator, "median_barrier_ns", .{ .integer = @intFromFloat(ov_stats.median) });
        } else {
            try exp_sum.put(allocator, "median_barrier_ns", .{ .float = ov_stats.median });
        }
        try exp_sum.put(allocator, "maximum_barrier_ns", .{ .integer = @intFromFloat(ov_stats.max) });
        try require(try spec.jsonEquals(allocator, qp.get("summary") orelse .null, .{ .object = exp_sum }), "quality_preflight summary", .{});
    } else {
        try require(
            (qp.get("status") orelse .null) == .string and
                std.mem.eql(u8, qp.get("status").?.string, "not-requested") and
                qp_samples_v.array.items.len == 0,
            "disabled quality_preflight",
            .{},
        );
    }

    const pair_plan_v = plan.get("pairs") orelse return failData("plan.pairs", .{});
    try require(pair_plan_v == .array and pair_plan_v.array.items.len > 0, "plan.pairs", .{});
    const expected_pairs = try sizing.expectedPairSpecsForPlan(allocator, plan);
    var expected_pairs_arr = Array.init(allocator);
    for (expected_pairs) |ep| try expected_pairs_arr.append(.{ .object = ep });
    try require(try spec.jsonEquals(allocator, pair_plan_v, .{ .array = expected_pairs_arr }), "plan.pairs is incomplete or reordered", .{});
    try sizing.validateSizingPlan(allocator, plan);

    var pair_by_key: ObjectMap = .empty;
    for (pair_plan_v.array.items) |pv| {
        try require(pv == .object, "plan pair object", .{});
        const pk_v = pv.object.get("pair_key") orelse return failData("plan pair key", .{});
        try require(pk_v == .string and pk_v.string.len > 0, "plan pair key", .{});
        try require(!pair_by_key.contains(pk_v.string), "duplicate plan pair {s}", .{pk_v.string});
        const l = pv.object.get("left") orelse return failData("pair direction {s}", .{pk_v.string});
        const r = pv.object.get("right") orelse return failData("pair direction {s}", .{pk_v.string});
        try require(l == .string and r == .string and !std.mem.eql(u8, l.string, r.string), "pair direction {s}", .{pk_v.string});
        try pair_by_key.put(allocator, pk_v.string, pv);
    }

    var record_objs: std.ArrayList(ObjectMap) = .empty;
    var seen_records: std.StringHashMapUnmanaged(void) = .empty;
    const CellTrack = struct {
        pair_key: []const u8,
        phase: []const u8,
        pair_index: i64,
        order: std.ArrayList(spec.InvocationPair),
    };
    var cell_tracks: std.ArrayList(CellTrack) = .empty;
    const ConcurrentTrack = struct {
        pair_key: []const u8,
        phase: []const u8,
        pair_index: i64,
        revision: []const u8,
        records: std.ArrayList(ObjectMap),
    };
    var concurrent_tracks: std.ArrayList(ConcurrentTrack) = .empty;

    for (records_v.array.items) |rv| {
        try require(rv == .object, "record object", .{});
        const record = rv.object;
        try record_objs.append(allocator, record);

        const phase_v = record.get("phase") orelse return failData("record phase", .{});
        try require(
            phase_v == .string and (std.mem.eql(u8, phase_v.string, "warmup") or std.mem.eql(u8, phase_v.string, "measure")),
            "record phase",
            .{},
        );
        const rev_role_v = record.get("revision") orelse return failData("record revision", .{});
        try require(rev_role_v == .string and revisions.contains(rev_role_v.string), "record revision", .{});
        const revision_role = rev_role_v.string;
        const revision = revisions.get(revision_role).?.object;

        try require((record.get("correct") orelse .null) == .bool and record.get("correct").?.bool, "record correctness", .{});
        const guest_elapsed_v = record.get("guest_elapsed_ns") orelse return failData("record guest elapsed", .{});
        try require(guest_elapsed_v == .integer and guest_elapsed_v.integer > 0, "record guest elapsed", .{});
        const guest_elapsed_ns = guest_elapsed_v.integer;

        const hs_v = record.get("host_started_ns") orelse return failData("record host interval", .{});
        const hf_v = record.get("host_finished_ns") orelse return failData("record host interval", .{});
        const hw_v = record.get("host_wall_elapsed_ns") orelse return failData("record host interval", .{});
        try require(
            hs_v == .integer and hf_v == .integer and hw_v == .integer and
                hs_v.integer >= 0 and hf_v.integer > hs_v.integer and
                hf_v.integer - hs_v.integer == hw_v.integer,
            "record host interval",
            .{},
        );

        const wl_v = record.get("workload") orelse return failData("record workload", .{});
        try require(wl_v == .string, "record workload", .{});
        const record_clock_id = spec.expectedGuestClockId(wl_v.string) catch return failData("record workload", .{});
        try require(
            std.mem.eql(u8, record_clock_id, spec.WASI_PROCESS_CPU_CLOCK_ID) or hw_v.integer >= guest_elapsed_ns,
            "record host wall diagnostic",
            .{},
        );
        const ov_v = record.get("timing_overhead_ns") orelse return failData("timing overhead", .{});
        try require(ov_v == .integer and ov_v.integer >= 0 and 99 * @as(i128, ov_v.integer) < @as(i128, guest_elapsed_ns), "timing overhead", .{});
        try require(guest_elapsed_ns >= min_interval_ns, "record minimum timed interval", .{});

        const pk_v = record.get("pair_key") orelse return failData("unknown record pair", .{});
        try require(pk_v == .string and pair_by_key.contains(pk_v.string), "unknown record pair", .{});
        const pair_key = pk_v.string;
        const pair = pair_by_key.get(pair_key).?.object;
        const p_kind = pair.get("pair_kind").?.string;
        const p_left = pair.get("left").?.string;
        const p_right = pair.get("right").?.string;

        try require((record.get("pair_kind") orelse .null) == .string and std.mem.eql(u8, record.get("pair_kind").?.string, p_kind), "record pair kind", .{});
        try require((record.get("pair_left") orelse .null) == .string and std.mem.eql(u8, record.get("pair_left").?.string, p_left), "record pair left", .{});
        try require((record.get("pair_right") orelse .null) == .string and std.mem.eql(u8, record.get("pair_right").?.string, p_right), "record pair right", .{});
        const cond_v = record.get("condition") orelse return failData("record pair condition", .{});
        try require(
            cond_v == .string and (std.mem.eql(u8, cond_v.string, p_left) or std.mem.eql(u8, cond_v.string, p_right)),
            "record pair condition",
            .{},
        );
        const condition = cond_v.string;
        const pair_index = (record.get("pair_index") orelse return failData("record pair_index", .{})).integer;
        const global_index: usize = @intCast(if (std.mem.eql(u8, phase_v.string, "warmup")) pair_index else warmups + pair_index);

        const cmd_v = record.get("command") orelse return failData("record command", .{});
        try require(cmd_v == .array and cmd_v.array.items.len >= 3, "record command", .{});

        var expected_cpu_affinity: []i64 = &.{};
        var expected_pair_exec: []const u8 = "";
        var expected_mgr_enabled: bool = true;
        if (std.mem.eql(u8, p_kind, "single-infrastructure")) {
            expected_cpu_affinity = spec.directPairCpuAffinity(allocator, cpu_placement, global_index, condition, p_left, p_right) catch return failData("record CPU affinity", .{});
            expected_pair_exec = "concurrent-topology-paired-cpus";
            expected_mgr_enabled = std.mem.eql(u8, condition, "threads-enabled");
            const mgr_opt = if (expected_mgr_enabled) "--benchmark-wasi-thread-manager=enabled" else "--benchmark-wasi-thread-manager=disabled";
            var opt_count: usize = 0;
            for (cmd_v.array.items) |arg_v| {
                if (arg_v == .string and std.mem.eql(u8, arg_v.string, mgr_opt)) opt_count += 1;
            }
            try require(opt_count == 1, "record thread-manager command", .{});

            var found_ct = false;
            for (concurrent_tracks.items) |*ct| {
                if (std.mem.eql(u8, ct.pair_key, pair_key) and
                    std.mem.eql(u8, ct.phase, phase_v.string) and
                    ct.pair_index == pair_index and
                    std.mem.eql(u8, ct.revision, revision_role))
                {
                    try ct.records.append(allocator, record);
                    found_ct = true;
                    break;
                }
            }
            if (!found_ct) {
                var rl: std.ArrayList(ObjectMap) = .empty;
                try rl.append(allocator, record);
                try concurrent_tracks.append(allocator, .{
                    .pair_key = pair_key,
                    .phase = phase_v.string,
                    .pair_index = pair_index,
                    .revision = revision_role,
                    .records = rl,
                });
            }
        } else {
            const r_threads = (record.get("threads") orelse return failData("record threads", .{})).integer;
            expected_cpu_affinity = spec.cpuAffinityFor(allocator, cpu_placement, wl_v.string, r_threads) catch return failData("record CPU affinity", .{});
            expected_pair_exec = "sequential-position-balanced";
            expected_mgr_enabled = true;
            for (cmd_v.array.items) |arg_v| {
                if (arg_v == .string and std.mem.startsWith(u8, arg_v.string, "--benchmark-wasi-thread-manager=")) {
                    return failData("non-direct record thread-manager command", .{});
                }
            }
        }

        var exp_aff_arr = Array.init(allocator);
        var aff_strs: std.ArrayList([]const u8) = .empty;
        for (expected_cpu_affinity) |c| {
            try exp_aff_arr.append(.{ .integer = c });
            try aff_strs.append(allocator, try std.fmt.allocPrint(allocator, "{d}", .{c}));
        }
        try require(try spec.jsonEquals(allocator, record.get("cpu_affinity") orelse .null, .{ .array = exp_aff_arr }), "record CPU affinity", .{});
        try require((record.get("pair_execution") orelse .null) == .string and std.mem.eql(u8, record.get("pair_execution").?.string, expected_pair_exec), "record pair execution", .{});
        try require(
            (record.get("threads_enabled") orelse .null) == .bool and record.get("threads_enabled").?.bool and
                (record.get("thread_manager_enabled") orelse .null) == .bool and record.get("thread_manager_enabled").?.bool == expected_mgr_enabled,
            "record thread-manager state",
            .{},
        );

        const exp_clock_mode = if (std.mem.eql(u8, record_clock_id, spec.WASI_PROCESS_CPU_CLOCK_ID))
            spec.WASI_PROCESS_CPU_CLOCK_MODE
        else
            spec.WASI_MONOTONIC_CLOCK_MODE;
        if (std.mem.eql(u8, wl_v.string, "hot") or std.mem.eql(u8, wl_v.string, "atomic")) {
            var mode_cnt: usize = 0;
            for (cmd_v.array.items) |arg_v| {
                if (arg_v == .string and std.mem.eql(u8, arg_v.string, exp_clock_mode)) mode_cnt += 1;
            }
            const last_arg = cmd_v.array.items[cmd_v.array.items.len - 1];
            try require(mode_cnt == 1 and last_arg == .string and std.mem.eql(u8, last_arg.string, exp_clock_mode), "record guest clock command", .{});
        } else {
            for (cmd_v.array.items) |arg_v| {
                if (arg_v == .string and (std.mem.eql(u8, arg_v.string, spec.WASI_PROCESS_CPU_CLOCK_MODE) or std.mem.eql(u8, arg_v.string, spec.WASI_MONOTONIC_CLOCK_MODE))) {
                    return failData("record guest clock command", .{});
                }
            }
        }
        const guest_v = record.get("guest") orelse return failData("record guest clock", .{});
        try require(
            guest_v == .object and (guest_v.object.get("clock_id") orelse .null) == .string and
                std.mem.eql(u8, guest_v.object.get("clock_id").?.string, record_clock_id),
            "record guest clock",
            .{},
        );
        const exp_cpu_list = try std.mem.join(allocator, ",", aff_strs.items);
        try require(
            cmd_v.array.items[0] == .string and std.mem.eql(u8, cmd_v.array.items[0].string, "taskset") and
                cmd_v.array.items[1] == .string and std.mem.eql(u8, cmd_v.array.items[1].string, "--cpu-list") and
                cmd_v.array.items[2] == .string and std.mem.eql(u8, cmd_v.array.items[2].string, exp_cpu_list),
            "record CPU affinity command",
            .{},
        );

        const rec_mode = (record.get("mode") orelse return failData("record mode", .{})).string;
        const rec_threads = (record.get("threads") orelse return failData("record threads", .{})).integer;
        const exp_iters = if (std.mem.eql(u8, p_kind, "single-infrastructure"))
            sizing.iterationCount(iters_v.object, rec_mode, "single-hot", 1) catch return failData("record plan iterations", .{})
        else if (std.mem.eql(u8, p_kind, "cancel-point-cost"))
            sizing.iterationCount(iters_v.object, "aot", "cancel-hot", rec_threads) catch return failData("record plan iterations", .{})
        else
            sizing.iterationCount(iters_v.object, rec_mode, wl_v.string, rec_threads) catch return failData("record plan iterations", .{});
        const rec_iters = spec.valueAsU64(record.get("iterations") orelse .null) orelse return failData("record plan iterations", .{});
        try require(rec_iters == exp_iters, "record plan iterations", .{});

        const id_pairs = [_]struct { rev_k: []const u8, rec_k: []const u8 }{
            .{ .rev_k = "commit", .rec_k = "revision_commit" },
            .{ .rev_k = "build_source_sha256", .rec_k = "revision_build_source_sha256" },
            .{ .rev_k = "fixture_set_sha256", .rec_k = "fixture_set_sha256" },
            .{ .rev_k = "plan_sha256", .rec_k = "plan_sha256" },
            .{ .rev_k = "host_pair_id", .rec_k = "host_pair_id" },
            .{ .rev_k = "host_fingerprint_sha256", .rec_k = "host_fingerprint_sha256" },
        };
        for (id_pairs) |ip| {
            const rv_id = record.get(ip.rec_k) orelse return failData("record mixed {s}", .{ip.rev_k});
            try require(rv_id == .string and std.mem.eql(u8, rv_id.string, revision.get(ip.rev_k).?.string), "record mixed {s}", .{ip.rev_k});
        }

        const expected_revs = if (revision_roles.len == 2)
            benchmark_schema.alternatingPairOrder(global_index, "baseline", "candidate")
        else
            [2][]const u8{ "candidate", "" };
        const exp_rev_idx: i64 = if (std.mem.eql(u8, expected_revs[0], revision_role)) 0 else 1;
        const expected_conds = spec.pairedConditionOrder(global_index, p_kind, p_left, p_right);
        const exp_cond_idx: i64 = if (std.mem.eql(u8, expected_conds[0], condition)) 0 else 1;
        try require((record.get("revision_order") orelse .null) == .integer and record.get("revision_order").?.integer == exp_rev_idx, "record revision order", .{});
        try require((record.get("order") orelse .null) == .integer and record.get("order").?.integer == exp_cond_idx, "record condition order", .{});

        const dup_key = try std.fmt.allocPrint(allocator, "{s}|{s}|{s}|{d}|{s}", .{ pair_key, revision_role, condition, pair_index, phase_v.string });
        try require(!seen_records.contains(dup_key), "duplicate record {s}", .{dup_key});
        try seen_records.put(allocator, dup_key, {});

        var found_cell = false;
        for (cell_tracks.items) |*ct| {
            if (std.mem.eql(u8, ct.pair_key, pair_key) and std.mem.eql(u8, ct.phase, phase_v.string) and ct.pair_index == pair_index) {
                try ct.order.append(allocator, .{ .revision = revision_role, .condition = condition });
                found_cell = true;
                break;
            }
        }
        if (!found_cell) {
            var ord: std.ArrayList(spec.InvocationPair) = .empty;
            try ord.append(allocator, .{ .revision = revision_role, .condition = condition });
            try cell_tracks.append(allocator, .{
                .pair_key = pair_key,
                .phase = phase_v.string,
                .pair_index = pair_index,
                .order = ord,
            });
        }
    }

    for (concurrent_tracks.items) |ct| {
        try require(ct.records.items.len == 2, "incomplete concurrent pair", .{});
        const r0 = ct.records.items[0];
        const r1 = ct.records.items[1];
        const c0 = r0.get("cpu_affinity").?.array.items[0].integer;
        const c1 = r1.get("cpu_affinity").?.array.items[0].integer;
        const shp = cpu_placement.get("assignments").?.object.get("single-hot-pair").?.object.get("logical_cpus").?.array.items;
        const p0 = shp[0].integer;
        const p1 = shp[1].integer;
        try require((c0 == p0 and c1 == p1) or (c0 == p1 and c1 == p0), "concurrent pair does not use its topology CPU pair", .{});
        const max_s = @max(r0.get("host_started_ns").?.integer, r1.get("host_started_ns").?.integer);
        const min_f = @min(r0.get("host_finished_ns").?.integer, r1.get("host_finished_ns").?.integer);
        try require(max_s < min_f, "concurrent pair host intervals do not overlap", .{});

        var stripped0: std.ArrayList([]const u8) = .empty;
        var stripped1: std.ArrayList([]const u8) = .empty;
        for (r0.get("command").?.array.items, 0..) |av, idx| {
            if (idx == 2) {
                try stripped0.append(allocator, "<balanced-cpu>");
            } else if (!std.mem.startsWith(u8, av.string, "--benchmark-wasi-thread-manager=")) {
                try stripped0.append(allocator, av.string);
            }
        }
        for (r1.get("command").?.array.items, 0..) |av, idx| {
            if (idx == 2) {
                try stripped1.append(allocator, "<balanced-cpu>");
            } else if (!std.mem.startsWith(u8, av.string, "--benchmark-wasi-thread-manager=")) {
                try stripped1.append(allocator, av.string);
            }
        }
        try require(stripped0.items.len == stripped1.items.len, "concurrent pair does not use one runtime artifact", .{});
        for (stripped0.items, stripped1.items) |s0, s1| {
            try require(std.mem.eql(u8, s0, s1), "concurrent pair does not use one runtime artifact", .{});
        }
    }

    const expected_records_per_pair: usize = revision_roles.len * 2 * @as(usize, @intCast(warmups + samples));
    var pk_it = pair_by_key.iterator();
    while (pk_it.next()) |entry| {
        const pair_key = entry.key_ptr.*;
        const pair = entry.value_ptr.object;
        var count_for_pair: usize = 0;
        for (record_objs.items) |r| {
            if (std.mem.eql(u8, r.get("pair_key").?.string, pair_key)) count_for_pair += 1;
        }
        try require(count_for_pair > 0, "report is missing planned pairs", .{});
        try require(count_for_pair == expected_records_per_pair, "incomplete sample pairing", .{});

        const phases = [_]struct { name: []const u8, cnt: i64 }{
            .{ .name = "warmup", .cnt = warmups },
            .{ .name = "measure", .cnt = samples },
        };
        for (phases) |ph| {
            var idx: i64 = 0;
            while (idx < ph.cnt) : (idx += 1) {
                var found_ct: ?CellTrack = null;
                for (cell_tracks.items) |ct| {
                    if (std.mem.eql(u8, ct.pair_key, pair_key) and std.mem.eql(u8, ct.phase, ph.name) and ct.pair_index == idx) {
                        found_ct = ct;
                        break;
                    }
                }
                const ct = found_ct orelse return failData("incomplete pair cell {s}/{s}/{d}", .{ pair_key, ph.name, idx });
                const global_index: usize = @intCast(if (std.mem.eql(u8, ph.name, "warmup")) idx else warmups + idx);
                const exp_order = try spec.pairedInvocationOrder(
                    allocator,
                    global_index,
                    pair.get("pair_kind").?.string,
                    pair.get("left").?.string,
                    pair.get("right").?.string,
                    revision_roles,
                );
                try require(ct.order.items.len == exp_order.len, "incomplete pair cell {s}/{s}/{d}", .{ pair_key, ph.name, idx });
                for (ct.order.items, exp_order) |actual_ip, exp_ip| {
                    try require(
                        std.mem.eql(u8, actual_ip.revision, exp_ip.revision) and
                            std.mem.eql(u8, actual_ip.condition, exp_ip.condition),
                        "inverted pair order {s}/{s}/{d}",
                        .{ pair_key, ph.name, idx },
                    );
                }
            }
        }
    }

    const recomputed_summaries = summarize(allocator, record_objs.items) catch return failData("revision summaries do not match records", .{});
    var rs_arr = Array.init(allocator);
    for (recomputed_summaries) |s| try rs_arr.append(.{ .object = s });
    try require(summaries_v.array.items.len == rs_arr.items.len, "summaries do not cover every planned condition", .{});
    try require(try spec.jsonEquals(allocator, summaries_v, .{ .array = rs_arr }), "revision summaries do not match records", .{});

    const ps_v = doc.get("paired_summaries") orelse return failData("paired_summaries", .{});
    try require(ps_v == .array, "paired_summaries", .{});
    for (ps_v.array.items) |item_v| {
        try require(item_v == .object, "paired summary object", .{});
        const pk = (item_v.object.get("pair_key") orelse return failData("paired summaries do not cover every planned pair", .{})).string;
        const pair = (pair_by_key.get(pk) orelse return failData("paired summaries do not cover every planned pair", .{})).object;
        try require(
            (item_v.object.get("left") orelse .null) == .string and
                std.mem.eql(u8, item_v.object.get("left").?.string, pair.get("left").?.string),
            "paired summary left",
            .{},
        );
        try require(
            (item_v.object.get("right") orelse .null) == .string and
                std.mem.eql(u8, item_v.object.get("right").?.string, pair.get("right").?.string),
            "paired summary right",
            .{},
        );
    }
    const recomputed_paired = pairedSummaries(allocator, record_objs.items) catch return failData("paired summaries do not match records", .{});
    var rp_arr = Array.init(allocator);
    for (recomputed_paired) |s| try rp_arr.append(.{ .object = s });
    try require(ps_v.array.items.len == rp_arr.items.len, "paired summaries do not cover every planned pair", .{});
    try require(try spec.jsonEquals(allocator, ps_v, .{ .array = rp_arr }), "paired summaries do not match records", .{});

    const cs_v = doc.get("comparison_summaries") orelse return failData("comparison_summaries", .{});
    try require(cs_v == .array, "comparison_summaries", .{});
    const recomputed_comp = comparisonSummaries(allocator, record_objs.items) catch return failData("comparison summaries do not match records", .{});
    var rc_arr = Array.init(allocator);
    for (recomputed_comp) |s| try rc_arr.append(.{ .object = s });
    try require(cs_v.array.items.len == rc_arr.items.len, "comparison summaries do not cover every planned condition", .{});
    try require(try spec.jsonEquals(allocator, cs_v, .{ .array = rc_arr }), "comparison summaries do not match records", .{});

    const rr_v = doc.get("ratio_of_ratios_summaries") orelse return failData("ratio_of_ratios_summaries", .{});
    try require(rr_v == .array, "ratio_of_ratios_summaries", .{});
    const recomputed_rr = ratioOfRatiosSummaries(allocator, record_objs.items) catch return failData("ratio-of-ratios summaries do not match records", .{});
    var rrr_arr = Array.init(allocator);
    for (recomputed_rr) |s| try rrr_arr.append(.{ .object = s });
    try require(rr_v.array.items.len == rrr_arr.items.len, "ratio-of-ratios summaries do not cover every planned pair", .{});
    try require(try spec.jsonEquals(allocator, rr_v, .{ .array = rrr_arr }), "ratio-of-ratios summaries do not match records", .{});
}

fn requireExactKeys(obj: ObjectMap, expected: []const []const u8, label: []const u8) HarnessError!void {
    if (obj.count() != expected.len) {
        return fail("{s} keys mismatch", .{label});
    }
    for (expected) |k| {
        if (!obj.contains(k)) {
            return fail("{s} keys mismatch: missing {s}", .{ label, k });
        }
    }
}

fn positiveNumber(v: Value) bool {
    const f = spec.valueAsF64(v) orelse return false;
    return std.math.isFinite(f) and f > 0.0;
}

pub fn requireBudgetEligibleReport(allocator: Allocator, report: Value) HarnessError!void {
    validateReport(allocator, report) catch {
        const msg = try allocator.dupe(u8, spec.lastErrorMessage());
        return fail("invalid report for budget enforcement: {s}", .{msg});
    };
    const doc = report.object;
    const plan = doc.get("plan").?.object;
    if (!std.mem.eql(u8, plan.get("revision_mode").?.string, "paired-revisions") or
        !std.mem.eql(u8, plan.get("comparison_purpose").?.string, "candidate-evaluation"))
    {
        return fail("budget enforcement requires a paired candidate-evaluation report", .{});
    }
    const revs = doc.get("metadata").?.object.get("revisions").?.object;
    const b_rev = revs.get("baseline").?.object;
    const c_rev = revs.get("candidate").?.object;
    if (std.mem.eql(u8, b_rev.get("commit").?.string, c_rev.get("commit").?.string)) {
        return fail("budget enforcement requires distinct baseline/candidate commits", .{});
    }
    if (std.mem.eql(u8, b_rev.get("build_source_sha256").?.string, c_rev.get("build_source_sha256").?.string)) {
        return fail("budget enforcement requires distinct baseline/candidate build identities", .{});
    }
}

pub fn loadBudget(
    allocator: Allocator,
    io: std.Io,
    path: []const u8,
    report: Value,
) HarnessError!ObjectMap {
    try requireBudgetEligibleReport(allocator, report);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024)) catch {
        return fail("invalid budget {s}: cannot read file", .{path});
    };
    const parsed = std.json.parseFromSlice(Value, allocator, bytes, .{
        .duplicate_field_behavior = .@"error",
    }) catch |err| switch (err) {
        error.DuplicateField => return fail("budget contains duplicate key", .{}),
        else => return fail("invalid budget {s}: parse error", .{path}),
    };
    if (parsed.value != .object) return fail("budget must be an object", .{});
    const budget = parsed.value.object;
    try requireExactKeys(budget, &.{
        "schema_version",
        "kind",
        "calibrated",
        "enforcement",
        "calibration_requirements",
        "calibration_provenance",
        "platforms",
    }, "budget");

    const sv = budget.get("schema_version").?;
    const kind = budget.get("kind").?;
    if (sv != .integer or sv.integer != spec.REPORT_SCHEMA_VERSION or
        kind != .string or !std.mem.eql(u8, kind.string, "wasi-thread-benchmark-budget"))
    {
        return fail("budget schema/kind mismatch", .{});
    }
    const cal_b = budget.get("calibrated").?;
    if (cal_b != .bool or !cal_b.bool) {
        return fail("budget is not calibrated; collect retained hosted reports and run with --no-budget", .{});
    }
    const enf_b = budget.get("enforcement").?;
    if (enf_b != .bool or !enf_b.bool) {
        return fail("budget enforcement is disabled; run with --no-budget", .{});
    }

    const reqs_v = budget.get("calibration_requirements").?;
    const cal_v = budget.get("calibration_provenance").?;
    const plats_v = budget.get("platforms").?;
    if (reqs_v != .object or cal_v != .object or plats_v != .object) {
        return fail("budget requirements/calibration/platforms must be objects", .{});
    }
    const requirements = reqs_v.object;
    const calibration = cal_v.object;
    const platforms = plats_v.object;

    try requireExactKeys(requirements, &.{
        "minimum_reports_per_platform",
        "minimum_reports_per_cpu_class",
        "minimum_training_reports_per_cpu_class",
        "minimum_holdout_reports_per_cpu_class",
        "required_profile",
        "required_platforms",
    }, "calibration_requirements");

    const min_rep_v = requirements.get("minimum_reports_per_platform").?;
    const min_cls_v = requirements.get("minimum_reports_per_cpu_class").?;
    const min_trn_v = requirements.get("minimum_training_reports_per_cpu_class").?;
    const min_hld_v = requirements.get("minimum_holdout_reports_per_cpu_class").?;
    const req_prof_v = requirements.get("required_profile").?;
    const req_plats_v = requirements.get("required_platforms").?;

    if (min_rep_v != .integer or min_rep_v.integer < 20) {
        return fail("budget minimum_reports_per_platform must be >= 20", .{});
    }
    if (min_cls_v != .integer or min_trn_v != .integer or min_hld_v != .integer or
        min_cls_v.integer != min_rep_v.integer or
        min_trn_v.integer < 16 or min_hld_v.integer < 4 or
        min_trn_v.integer + min_hld_v.integer > min_cls_v.integer)
    {
        return fail("budget CPU-class calibration minima are invalid", .{});
    }
    if (req_prof_v != .string or !std.mem.eql(u8, req_prof_v.string, "authoritative")) {
        return fail("budget required_profile must be authoritative", .{});
    }
    if (req_plats_v != .array or req_plats_v.array.items.len != 2) {
        return fail("budget required_platforms must be exactly the canonical hosted platforms", .{});
    }
    var has_x86 = false;
    var has_arm = false;
    for (req_plats_v.array.items) |pv| {
        if (pv != .string) return fail("budget required_platforms must be exactly the canonical hosted platforms", .{});
        if (std.mem.eql(u8, pv.string, "ubuntu-22.04-x86_64")) has_x86 = true;
        if (std.mem.eql(u8, pv.string, "ubuntu-24.04-aarch64")) has_arm = true;
    }
    if (!has_x86 or !has_arm) {
        return fail("budget required_platforms must be exactly the canonical hosted platforms", .{});
    }
    const required_platforms = [_][]const u8{ "ubuntu-22.04-x86_64", "ubuntu-24.04-aarch64" };

    try requireExactKeys(calibration, &.{
        "baseline_revision",
        "candidate_revision",
        "comparison_purpose",
        "fixture_set_sha256",
        "measurement_plan_version",
        "measurement_plan_sha256",
        "profile",
        "report_count_by_platform",
        "accepted_cpu_classes",
        "report_count_by_cpu_class",
        "partition_report_count_by_cpu_class",
    }, "calibration_provenance");

    const cp_v = calibration.get("comparison_purpose").?;
    if (cp_v != .string or !std.mem.eql(u8, cp_v.string, "noise-calibration")) {
        return fail("budget calibration provenance must be noise calibration", .{});
    }
    for (spec.REVISION_ROLES) |role| {
        const key = try std.fmt.allocPrint(allocator, "{s}_revision", .{role});
        const id_v = calibration.get(key).?;
        if (id_v != .object) return fail("budget calibration {s} revision must be an object", .{role});
        try requireExactKeys(id_v.object, &.{ "commit", "build_source_sha256" }, key);
        const c_v = id_v.object.get("commit").?;
        const b_v = id_v.object.get("build_source_sha256").?;
        if (c_v != .string or !spec.isHexCommit(c_v.string)) {
            return fail("budget calibration {s}.commit has invalid identity", .{role});
        }
        if (b_v != .string or !spec.isHexSha256(b_v.string)) {
            return fail("budget calibration {s}.build_source_sha256 has invalid identity", .{role});
        }
    }
    if (!(try spec.jsonEquals(allocator, calibration.get("baseline_revision").?, calibration.get("candidate_revision").?))) {
        return fail("budget noise-calibration revisions must have identical identities", .{});
    }
    for ([_][]const u8{ "fixture_set_sha256", "measurement_plan_sha256" }) |k| {
        const v = calibration.get(k).?;
        if (v != .string or !spec.isHexSha256(v.string)) {
            return fail("budget calibration_provenance.{s} has invalid identity", .{k});
        }
    }
    const cal_mpv = calibration.get("measurement_plan_version").?;
    if (cal_mpv != .integer or cal_mpv.integer != spec.MEASUREMENT_PLAN_IDENTITY_VERSION) {
        return fail("budget measurement plan identity version mismatch", .{});
    }
    const cal_prof = calibration.get("profile").?;
    if (cal_prof != .string or !std.mem.eql(u8, cal_prof.string, "authoritative")) {
        return fail("budget calibration profile mismatch", .{});
    }

    const counts_v = calibration.get("report_count_by_platform").?;
    if (counts_v != .object or counts_v.object.count() != 2) {
        return fail("budget calibration report counts do not cover platforms", .{});
    }
    for (required_platforms) |plat| {
        const cv = counts_v.object.get(plat) orelse return fail("budget calibration report counts do not cover platforms", .{});
        if (cv != .integer or cv.integer < min_rep_v.integer) {
            return fail("budget calibration has insufficient retained reports", .{});
        }
    }

    const acc_cls_v = calibration.get("accepted_cpu_classes").?;
    const cls_cnt_v = calibration.get("report_count_by_cpu_class").?;
    const part_cnt_v = calibration.get("partition_report_count_by_cpu_class").?;
    if (acc_cls_v != .object or acc_cls_v.object.count() != 2 or
        cls_cnt_v != .object or cls_cnt_v.object.count() != 2 or
        part_cnt_v != .object or part_cnt_v.object.count() != 2)
    {
        return fail("budget CPU-class provenance does not cover platforms", .{});
    }

    for (required_platforms) |plat| {
        const classes_v = acc_cls_v.object.get(plat) orelse return fail("budget CPU-class provenance does not cover platforms", .{});
        const cc_v = cls_cnt_v.object.get(plat) orelse return fail("budget CPU-class provenance does not cover platforms", .{});
        const pc_v = part_cnt_v.object.get(plat) orelse return fail("budget CPU-class provenance does not cover platforms", .{});
        if (classes_v != .array or classes_v.array.items.len == 0 or cc_v != .object or pc_v != .object or
            cc_v.object.count() != classes_v.array.items.len or pc_v.object.count() != classes_v.array.items.len)
        {
            return fail("budget CPU-class provenance is invalid for {s}", .{plat});
        }
        var total: i64 = 0;
        for (classes_v.array.items, 0..) |c_val, idx| {
            if (c_val != .string) return fail("budget CPU-class provenance is invalid for {s}", .{plat});
            const cpu_class = c_val.string;
            const trimmed = std.mem.trim(u8, cpu_class, " \t\r\n");
            if (cpu_class.len == 0 or !std.mem.eql(u8, cpu_class, trimmed)) {
                return fail("budget CPU-class provenance is invalid for {s}", .{plat});
            }
            if (idx > 0) {
                const prev = classes_v.array.items[idx - 1].string;
                if (std.mem.order(u8, prev, cpu_class) != .lt) {
                    return fail("budget CPU-class provenance is invalid for {s}", .{plat});
                }
            }
            const cnt_val = cc_v.object.get(cpu_class) orelse return fail("budget CPU-class provenance is invalid for {s}", .{plat});
            const split_val = pc_v.object.get(cpu_class) orelse return fail("budget CPU-class provenance is invalid for {s}", .{plat});
            if (cnt_val != .integer or cnt_val.integer < min_cls_v.integer or
                split_val != .object or split_val.object.count() != 2)
            {
                return fail("budget CPU class '{s}' for {s} is undersampled", .{ cpu_class, plat });
            }
            const tr_v = split_val.object.get("training") orelse return fail("budget CPU class '{s}' for {s} is undersampled", .{ cpu_class, plat });
            const hl_v = split_val.object.get("holdout") orelse return fail("budget CPU class '{s}' for {s} is undersampled", .{ cpu_class, plat });
            if (tr_v != .integer or tr_v.integer < min_trn_v.integer or
                hl_v != .integer or hl_v.integer < min_hld_v.integer or
                tr_v.integer + hl_v.integer != cnt_val.integer)
            {
                return fail("budget CPU class '{s}' for {s} is undersampled", .{ cpu_class, plat });
            }
            total += cnt_val.integer;
        }
        if (total != counts_v.object.get(plat).?.integer) {
            return fail("budget CPU-class counts do not sum for {s}", .{plat});
        }
    }

    if (platforms.count() != 2 or !platforms.contains("ubuntu-22.04-x86_64") or !platforms.contains("ubuntu-24.04-aarch64")) {
        return fail("budget platforms are incomplete or unknown", .{});
    }

    const doc = report.object;
    const metadata = doc.get("metadata").?.object;
    const baseline = metadata.get("revisions").?.object.get("baseline").?.object;
    const cal_base = calibration.get("baseline_revision").?.object;

    if (!std.mem.eql(u8, cal_base.get("commit").?.string, baseline.get("commit").?.string)) {
        return fail("budget/report calibration mismatch for baseline_revision.commit", .{});
    }
    if (!std.mem.eql(u8, cal_base.get("build_source_sha256").?.string, baseline.get("build_source_sha256").?.string)) {
        return fail("budget/report calibration mismatch for baseline_revision.build_source_sha256", .{});
    }
    if (!std.mem.eql(u8, calibration.get("fixture_set_sha256").?.string, metadata.get("fixture_set_sha256").?.string)) {
        return fail("budget/report calibration mismatch for fixture_set_sha256", .{});
    }
    if (calibration.get("measurement_plan_version").?.integer != metadata.get("measurement_plan_version").?.integer) {
        return fail("budget/report calibration mismatch for measurement_plan_version", .{});
    }
    if (!(try spec.isCalibrationCompatibleMeasurementPlanSha256(
        allocator,
        doc.get("plan").?,
        calibration.get("measurement_plan_sha256").?.string,
    ))) {
        return fail("budget/report calibration mismatch for measurement_plan_sha256", .{});
    }
    if (!std.mem.eql(u8, calibration.get("profile").?.string, doc.get("plan").?.object.get("profile").?.string)) {
        return fail("budget/report calibration mismatch for profile", .{});
    }

    const platform_id = metadata.get("platform_id").?.string;
    const plat_budget_v = platforms.get(platform_id) orelse return fail("budget has no platform '{s}'", .{platform_id});
    if (plat_budget_v != .object) return fail("platform budget must be an object", .{});
    const platform_budget = plat_budget_v.object;
    try requireExactKeys(platform_budget, &.{
        "host_system",
        "host_machine",
        "runner_environment",
        "accepted_cpu_classes",
        "comparisons",
        "ratio_of_ratios",
    }, "platforms");

    const host = metadata.get("host").?.object;
    const exp_host = spec.canonicalPlatformHost(platform_id).?;
    if (!std.mem.eql(u8, host.get("system").?.string, exp_host.system) or
        !std.mem.eql(u8, host.get("machine").?.string, exp_host.machine))
    {
        return fail("report platform ID does not match its canonical host identity", .{});
    }
    if (platform_budget.get("host_system").? != .string or
        !std.mem.eql(u8, platform_budget.get("host_system").?.string, host.get("system").?.string) or
        platform_budget.get("host_machine").? != .string or
        !std.mem.eql(u8, platform_budget.get("host_machine").?.string, host.get("machine").?.string))
    {
        return fail("budget/report host platform mismatch", .{});
    }
    if (platform_budget.get("runner_environment").? != .string or
        !std.mem.eql(u8, platform_budget.get("runner_environment").?.string, host.get("runner_environment").?.string))
    {
        return fail("budget/report runner environment mismatch", .{});
    }
    if (!(try spec.jsonEquals(allocator, platform_budget.get("accepted_cpu_classes").?, acc_cls_v.object.get(platform_id).?))) {
        return fail("budget platform CPU classes differ from provenance", .{});
    }
    const report_cpu_class = host.get("host_fingerprint").?.object.get("fields").?.object.get("cpu").?.string;
    var cpu_accepted = false;
    for (platform_budget.get("accepted_cpu_classes").?.array.items) |cv| {
        if (std.mem.eql(u8, cv.string, report_cpu_class)) cpu_accepted = true;
    }
    if (!cpu_accepted) {
        return fail("report CPU class '{s}' is not calibrated for {s}", .{ report_cpu_class, platform_id });
    }

    const rep_comps = doc.get("comparison_summaries").?.array.items;
    const comp_thresholds_v = platform_budget.get("comparisons").?;
    if (comp_thresholds_v != .array) return fail("platform comparisons must be an array", .{});
    var seen_comps: std.StringHashMapUnmanaged(void) = .empty;
    for (comp_thresholds_v.array.items) |item_v| {
        if (item_v != .object) return fail("comparison threshold must be an object", .{});
        const item = item_v.object;
        try requireExactKeys(item, &.{
            "pair_key",
            "condition",
            "metric_kind",
            "min_candidate_over_baseline_throughput_ratio",
            "max_candidate_over_baseline_elapsed_ratio",
        }, "comparison threshold");
        const pk_v = item.get("pair_key").?;
        const cd_v = item.get("condition").?;
        const mk_v = item.get("metric_kind").?;
        if (pk_v != .string or cd_v != .string or mk_v != .string) return fail("unknown comparison threshold", .{});
        const k_str = try std.fmt.allocPrint(allocator, "{s}::{s}", .{ pk_v.string, cd_v.string });
        if (seen_comps.contains(k_str)) return fail("duplicate comparison threshold {s}", .{k_str});
        try seen_comps.put(allocator, k_str, {});

        var matched_mk: ?[]const u8 = null;
        for (rep_comps) |rc| {
            if (std.mem.eql(u8, rc.object.get("pair_key").?.string, pk_v.string) and
                std.mem.eql(u8, rc.object.get("condition").?.string, cd_v.string))
            {
                matched_mk = rc.object.get("metric_kind").?.string;
                break;
            }
        }
        if (matched_mk == null) return fail("unknown comparison threshold {s}", .{k_str});
        if (!std.mem.eql(u8, mk_v.string, matched_mk.?)) return fail("comparison metric mismatch for {s}", .{k_str});
        if (!positiveNumber(item.get("min_candidate_over_baseline_throughput_ratio").?) or
            !positiveNumber(item.get("max_candidate_over_baseline_elapsed_ratio").?))
        {
            return fail("comparison ratio must be positive for {s}", .{k_str});
        }
    }
    if (seen_comps.count() != rep_comps.len) {
        return fail("comparison thresholds do not cover every report comparison", .{});
    }

    const rep_pairs = doc.get("plan").?.object.get("pairs").?.array.items;
    const ratio_thresholds_v = platform_budget.get("ratio_of_ratios").?;
    if (ratio_thresholds_v != .array) return fail("platform ratio_of_ratios must be an array", .{});
    var seen_pairs: std.StringHashMapUnmanaged(void) = .empty;
    for (ratio_thresholds_v.array.items) |item_v| {
        if (item_v != .object) return fail("ratio-of-ratios threshold must be an object", .{});
        const item = item_v.object;
        try requireExactKeys(item, &.{
            "pair_key",
            "left",
            "right",
            "min_candidate_over_baseline_throughput_ratio_of_ratios",
            "max_candidate_over_baseline_elapsed_ratio_of_ratios",
        }, "ratio-of-ratios threshold");
        const pk_v = item.get("pair_key").?;
        const l_v = item.get("left").?;
        const r_v = item.get("right").?;
        if (pk_v != .string or l_v != .string or r_v != .string) return fail("unknown ratio-of-ratios threshold", .{});
        if (seen_pairs.contains(pk_v.string)) return fail("duplicate ratio-of-ratios threshold {s}", .{pk_v.string});
        try seen_pairs.put(allocator, pk_v.string, {});

        var matched_pair: ?ObjectMap = null;
        for (rep_pairs) |rp| {
            if (std.mem.eql(u8, rp.object.get("pair_key").?.string, pk_v.string)) {
                matched_pair = rp.object;
                break;
            }
        }
        if (matched_pair == null) return fail("unknown ratio-of-ratios threshold {s}", .{pk_v.string});
        if (!std.mem.eql(u8, l_v.string, matched_pair.?.get("left").?.string) or
            !std.mem.eql(u8, r_v.string, matched_pair.?.get("right").?.string))
        {
            return fail("ratio-of-ratios direction mismatch for {s}", .{pk_v.string});
        }
        if (!positiveNumber(item.get("min_candidate_over_baseline_throughput_ratio_of_ratios").?) or
            !positiveNumber(item.get("max_candidate_over_baseline_elapsed_ratio_of_ratios").?))
        {
            return fail("ratio-of-ratios limit must be positive for {s}", .{pk_v.string});
        }
    }
    if (seen_pairs.count() != rep_pairs.len) {
        return fail("ratio-of-ratios thresholds do not cover the complete report plan", .{});
    }
    return platform_budget;
}

pub fn evaluateBudget(
    allocator: Allocator,
    platform_budget: ObjectMap,
    report: Value,
) HarnessError![]const []const u8 {
    try requireBudgetEligibleReport(allocator, report);
    const doc = report.object;
    const comparisons = doc.get("comparison_summaries").?.array.items;
    const ratio_of_ratios = doc.get("ratio_of_ratios_summaries").?.array.items;
    var failures: std.ArrayList([]const u8) = .empty;

    for (platform_budget.get("comparisons").?.array.items) |thresh_v| {
        const thresh = thresh_v.object;
        const pk = thresh.get("pair_key").?.string;
        const cond = thresh.get("condition").?.string;
        const min_th = spec.valueAsF64(thresh.get("min_candidate_over_baseline_throughput_ratio").?).?;
        const max_el = spec.valueAsF64(thresh.get("max_candidate_over_baseline_elapsed_ratio").?).?;

        for (comparisons) |comp_v| {
            const comp = comp_v.object;
            if (std.mem.eql(u8, comp.get("pair_key").?.string, pk) and
                std.mem.eql(u8, comp.get("condition").?.string, cond))
            {
                const th = spec.valueAsF64(comp.get("throughput_candidate_over_baseline").?.object.get("median").?).?;
                const el = spec.valueAsF64(comp.get("elapsed_candidate_over_baseline").?.object.get("median").?).?;
                if (th < min_th) {
                    try failures.append(allocator, try std.fmt.allocPrint(
                        allocator,
                        "{s}::{s} candidate/baseline throughput: {d:.4} < {d:.4}",
                        .{ pk, cond, th, min_th },
                    ));
                }
                if (el > max_el) {
                    try failures.append(allocator, try std.fmt.allocPrint(
                        allocator,
                        "{s}::{s} candidate/baseline elapsed: {d:.4} > {d:.4}",
                        .{ pk, cond, el, max_el },
                    ));
                }
                break;
            }
        }
    }

    for (platform_budget.get("ratio_of_ratios").?.array.items) |thresh_v| {
        const thresh = thresh_v.object;
        const pk = thresh.get("pair_key").?.string;
        const min_th = spec.valueAsF64(thresh.get("min_candidate_over_baseline_throughput_ratio_of_ratios").?).?;
        const max_el = spec.valueAsF64(thresh.get("max_candidate_over_baseline_elapsed_ratio_of_ratios").?).?;

        for (ratio_of_ratios) |rr_v| {
            const rr = rr_v.object;
            if (std.mem.eql(u8, rr.get("pair_key").?.string, pk)) {
                const th = spec.valueAsF64(rr.get("throughput_ratio_of_ratios").?.object.get("median").?).?;
                const el = spec.valueAsF64(rr.get("elapsed_ratio_of_ratios").?.object.get("median").?).?;
                if (th < min_th) {
                    try failures.append(allocator, try std.fmt.allocPrint(
                        allocator,
                        "{s} candidate/baseline throughput ratio-of-ratios: {d:.4} < {d:.4}",
                        .{ pk, th, min_th },
                    ));
                }
                if (el > max_el) {
                    try failures.append(allocator, try std.fmt.allocPrint(
                        allocator,
                        "{s} candidate/baseline elapsed ratio-of-ratios: {d:.4} > {d:.4}",
                        .{ pk, el, max_el },
                    ));
                }
                break;
            }
        }
    }
    return failures.toOwnedSlice(allocator);
}

fn formatIntListBracketed(allocator: Allocator, arr: Array) ![]const u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    for (arr.items) |item| {
        try parts.append(allocator, try std.fmt.allocPrint(allocator, "{d}", .{item.integer}));
    }
    const inner = try std.mem.join(allocator, ", ", parts.items);
    return std.fmt.allocPrint(allocator, "[{s}]", .{inner});
}

fn fmtSignedPct(allocator: Allocator, val: f64) ![]const u8 {
    if (val >= 0.0) {
        return std.fmt.allocPrint(allocator, "+{d:.2}%", .{val});
    }
    return std.fmt.allocPrint(allocator, "{d:.2}%", .{val});
}

pub fn renderMarkdown(allocator: Allocator, document: ObjectMap) ![]const u8 {
    const metadata = document.get("metadata").?.object;
    const host = metadata.get("host").?.object;
    const revisions = metadata.get("revisions").?.object;
    const plan = document.get("plan").?.object;
    const sizing_obj = plan.get("sizing").?.object;
    const alg_obj = sizing_obj.get("algorithm").?.object;
    const resolved_obj = sizing_obj.get("resolved").?.object;
    const checksum_prep = metadata.get("checksum_preparation").?.object;
    const ordered_cpus_str = try formatIntListBracketed(
        allocator,
        metadata.get("cpu_placement").?.object.get("ordered_logical_cpus").?.array,
    );

    var lines: std.ArrayList([]const u8) = .empty;
    try lines.append(allocator, "# WASI threaded benchmark");
    try lines.append(allocator, "");
    if (std.mem.eql(u8, plan.get("revision_mode").?.string, "paired-revisions")) {
        try lines.append(allocator, try std.fmt.allocPrint(
            allocator,
            "- Baseline: `{s}`",
            .{revisions.get("baseline").?.object.get("commit").?.string},
        ));
    }
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Candidate: `{s}`",
        .{revisions.get("candidate").?.object.get("commit").?.string},
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Comparison purpose: `{s}`",
        .{plan.get("comparison_purpose").?.string},
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Platform identity: `{s}`",
        .{metadata.get("platform_id").?.string},
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Host pair: `{s}` · fingerprint `{s}`",
        .{
            metadata.get("host_pair").?.object.get("id").?.string,
            metadata.get("host_pair").?.object.get("host_fingerprint_sha256").?.string,
        },
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Runner environment: `{s}`",
        .{host.get("runner_environment").?.string},
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- CPU placement: `{s}` · ordered logical CPUs `{s}`",
        .{
            plan.get("cpu_placement").?.object.get("kind").?.string,
            ordered_cpus_str,
        },
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Revision artifacts: `{s}`",
        .{metadata.get("revision_artifacts").?.object.get("strategy").?.string},
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Host: `{s} {s}` · `{s}` · {d} CPUs · `{s}`",
        .{
            host.get("system").?.string,
            host.get("release").?.string,
            host.get("machine").?.string,
            host.get("logical_cpus").?.integer,
            host.get("cpu").?.string,
        },
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Profile: `{s}` ({d} warmups, {d} samples)",
        .{
            plan.get("profile").?.string,
            plan.get("warmups").?.integer,
            plan.get("samples").?.integer,
        },
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- One-shot sizing: `{s}` v{d}; {d} retained pilots, {d} frozen cells, {d} declarative cell envelope; projected benchmark {d:.1}s",
        .{
            alg_obj.get("kind").?.string,
            alg_obj.get("version").?.integer,
            resolved_obj.get("pilots").?.array.items.len,
            resolved_obj.get("cells").?.array.items.len,
            alg_obj.get("cell_envelopes").?.array.items.len,
            @as(f64, @floatFromInt(resolved_obj.get("projected_benchmark_ns").?.integer)) / 1e9,
        },
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Budget: `{s}`",
        .{document.get("budget").?.object.get("status").?.string},
    ));
    try lines.append(allocator, try std.fmt.allocPrint(
        allocator,
        "- Checksum preparation: `{s}`; worst {d} ns, total {d} ns",
        .{
            checksum_prep.get("algorithm").?.string,
            checksum_prep.get("worst_ns").?.integer,
            checksum_prep.get("total_ns").?.integer,
        },
    ));

    const qp = document.get("quality_preflight").?.object;
    if (qp.get("enabled").?.bool) {
        try lines.append(allocator, try std.fmt.allocPrint(
            allocator,
            "- Scheduler/barrier preflight: `{s}`; {d} fixed probes; maximum {d} ns (limit {d} ns)",
            .{
                qp.get("status").?.string,
                qp.get("probe_count").?.integer,
                qp.get("summary").?.object.get("maximum_barrier_ns").?.integer,
                qp.get("maximum_accepted_barrier_ns").?.integer,
            },
        ));
    }

    if (metadata.get("aot_artifacts")) |aa_v| {
        if (aa_v == .object) {
            if (aa_v.object.get("candidate")) |cand_aa| {
                if (cand_aa == .object) {
                    if (cand_aa.object.get("cancel_poll_static")) |cps_v| {
                        if (cps_v == .object) {
                            try lines.append(allocator, try std.fmt.allocPrint(
                                allocator,
                                "- AOT cancel polls: {d} static sites, {d} bytes/site; timed hot kernel executes 1 poll opportunity per operation when enabled",
                                .{
                                    cps_v.object.get("sites_enabled").?.integer,
                                    cps_v.object.get("bytes_per_site").?.integer,
                                },
                            ));
                        }
                    }
                }
            }
        }
    }

    try lines.append(allocator, "");
    try lines.append(allocator, "| Revision | Pair | Condition | Metric | Guest median ms | Host-wall median ms | Guest range ms | Median aggregate ops/s | Median per-thread ops/s |");
    try lines.append(allocator, "|---|---|---|---|---:|---:|---:|---:|---:|");
    for (document.get("summaries").?.array.items) |item_v| {
        const item = item_v.object;
        const el = item.get("elapsed").?.object;
        const hw = item.get("host_wall").?.object;
        const th = item.get("throughput").?.object;
        const pt = item.get("per_thread_throughput").?.object;
        try lines.append(allocator, try std.fmt.allocPrint(
            allocator,
            "| `{s}` | `{s}` | `{s}` | `{s}` | {d:.3} | {d:.3} | {d:.3}–{d:.3} | {d:.3} | {d:.3} |",
            .{
                item.get("revision").?.string,
                item.get("pair_key").?.string,
                item.get("condition").?.string,
                item.get("metric_kind").?.string,
                (spec.valueAsF64(el.get("median").?).?) / 1e6,
                (spec.valueAsF64(hw.get("median").?).?) / 1e6,
                (spec.valueAsF64(el.get("min").?).?) / 1e6,
                (spec.valueAsF64(el.get("max").?).?) / 1e6,
                spec.valueAsF64(th.get("median").?).?,
                spec.valueAsF64(pt.get("median").?).?,
            },
        ));
    }

    try lines.append(allocator, "");
    try lines.append(allocator, "| Revision | Internal pair | Right / left guest-time median delta | Right / left throughput median delta |");
    try lines.append(allocator, "|---|---|---:|---:|");
    for (document.get("paired_summaries").?.array.items) |item_v| {
        const item = item_v.object;
        const el_pct = try fmtSignedPct(allocator, spec.valueAsF64(item.get("median_elapsed_delta_pct").?).?);
        const th_pct = try fmtSignedPct(allocator, spec.valueAsF64(item.get("median_throughput_delta_pct").?).?);
        try lines.append(allocator, try std.fmt.allocPrint(
            allocator,
            "| `{s}` | `{s}`: `{s}` / `{s}` | {s} | {s} |",
            .{
                item.get("revision").?.string,
                item.get("pair_key").?.string,
                item.get("right").?.string,
                item.get("left").?.string,
                el_pct,
                th_pct,
            },
        ));
    }

    const comps = document.get("comparison_summaries").?.array.items;
    if (comps.len > 0) {
        try lines.append(allocator, "");
        try lines.append(allocator, "| Matched revision comparison | Candidate / baseline elapsed | Candidate / baseline throughput |");
        try lines.append(allocator, "|---|---:|---:|");
        for (comps) |item_v| {
            const item = item_v.object;
            try lines.append(allocator, try std.fmt.allocPrint(
                allocator,
                "| `{s}` / `{s}` | {d:.4} | {d:.4} |",
                .{
                    item.get("pair_key").?.string,
                    item.get("condition").?.string,
                    spec.valueAsF64(item.get("elapsed_candidate_over_baseline").?.object.get("median").?).?,
                    spec.valueAsF64(item.get("throughput_candidate_over_baseline").?.object.get("median").?).?,
                },
            ));
        }

        try lines.append(allocator, "");
        try lines.append(allocator, "| Internal-pair ratio-of-ratios | Candidate / baseline elapsed ratio | Candidate / baseline throughput ratio |");
        try lines.append(allocator, "|---|---:|---:|");
        for (document.get("ratio_of_ratios_summaries").?.array.items) |item_v| {
            const item = item_v.object;
            try lines.append(allocator, try std.fmt.allocPrint(
                allocator,
                "| `{s}`: `{s}` / `{s}` | {d:.4} | {d:.4} |",
                .{
                    item.get("pair_key").?.string,
                    item.get("right").?.string,
                    item.get("left").?.string,
                    spec.valueAsF64(item.get("elapsed_ratio_of_ratios").?.object.get("median").?).?,
                    spec.valueAsF64(item.get("throughput_ratio_of_ratios").?.object.get("median").?).?,
                },
            ));
        }
    }

    try lines.append(allocator, "");
    try lines.append(allocator, "Kernel throughput uses guest monotonic time corrected by a same-process barrier/timer calibration below 1%; host wall time is diagnostic only. Spawn/join is reported separately as a guest-timed lifecycle metric. Absolute values remain diagnostic; budget enforcement uses only matched candidate/baseline ratios.");
    try lines.append(allocator, "");
    try lines.append(allocator, "Every warmup and measured invocation passed its deterministic timing, checksum, operation count, workload, thread-count, and iteration assertions.");
    try lines.append(allocator, "");
    return std.mem.join(allocator, "\n", lines.items);
}
