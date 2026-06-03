//! AOT codegen bisection knobs (#761 / #743).
//!
//! `wamrc` exposes three env vars to let bisects narrow a suspected
//! IR-optimisation miscompile down to a single (pass, function) pair
//! without recompiling the rest of the module with a partial pipeline:
//!
//! ```
//! WAMR_AOT_SKIP_PASS=15                      # skip pass 15 in every func
//! WAMR_AOT_SKIP_PASS=15:fn=11040             # skip pass 15 only in fn 11040
//! WAMR_AOT_SKIP_PASS=15-19:fn=11040          # skip a pass range
//! WAMR_AOT_SKIP_PASS=15:mod=4                # skip pass 15 only in core module 4
//! WAMR_AOT_SKIP_PASS=15:mod=4:fn=0-2079      # only module 4 + fn 0-2079
//! WAMR_AOT_SKIP_PASSES=15;17:fn=11040        # multiple specs, ';'-joined
//! WAMR_AOT_PASSES_LIMIT=30                   # cap pipeline at 30 passes
//! WAMR_AOT_PASSES_LIMIT=30:fn=11040,11041    # cap only listed funcs
//! WAMR_AOT_PASSES_LIMIT=30:mod=4             # cap only in module 4
//! ```
//!
//! Grammar (single env var):
//!   spec        := skip_spec | limit_spec
//!   skip_spec   := <pass_idx_or_range> { ':' filter }
//!   limit_spec  := <usize>             { ':' filter }
//!   filter      := 'fn=' <fn_list> | 'mod=' <fn_list>
//!   pass_idx_or_range := <usize> [ '-' <usize> ]
//!   fn_list     := <fn_item> { ',' <fn_item> }
//!   fn_item     := <usize> [ '-' <usize> ]
//!
//! Both `fn=` and `mod=` accept the same range/list syntax. Each may
//! appear at most once per spec; their order is interchangeable. A
//! filter that's omitted means "match all" (all functions / all
//! modules respectively). Module indices are the per-core indices
//! assigned by `wamrc compile-component`; the single-module
//! `wamrc compile` path treats every spec as module 0, so a
//! `:mod=N` filter with `N != 0` never matches there.
//!
//! Multiple `Skip` entries may be supplied via `WAMR_AOT_SKIP_PASSES`
//! by joining specs with `;`. The single-spec form
//! (`WAMR_AOT_SKIP_PASS`) is an alias accepting at most one spec.
//!
//! All parsing is pure (`parseSkipSpec` / `parseLimitSpec` /
//! `parseFnList`) and unit-tested without env-var I/O. `parseFromEnv`
//! is a thin wrapper that pulls the three keys and stamps the result
//! into the process-global `global` for `runPassesWithOptions` to
//! consult via `passes.RunOptions.bisect`.

const std = @import("std");
const passes = @import("ir/passes.zig");

pub const Spec = passes.PassBisectSpec;
pub const Skip = passes.PassBisectSpec.Skip;
pub const Limit = passes.PassBisectSpec.Limit;

pub const ParseError = error{
    EmptyInput,
    InvalidNumber,
    InvalidRange,
    UnknownTrailing,
    OutOfMemory,
};

/// Process-global bisect spec. Populated by `parseFromEnv` at wamrc
/// startup; consumed by both single-module (`runCompile`) and
/// component (`compileCoreWasm`) AOT paths. Default `.{}` means "no
/// filtering, full pipeline".
///
/// Lifetime: the env parser leaks the backing slices into the GPA
/// passed to `parseFromEnv` for the process lifetime — `wamrc` is a
/// short-lived CLI tool and these are tens of bytes.
pub var global: Spec = .{};

/// Pull the three env vars out of `env` and stamp `global` with the
/// resulting spec. Logs `[#761 bisect] …` warnings for each active
/// knob and `[#761 bisect] WARN: …` on parse errors (env-var typos
/// silently degrading to "full pipeline" is the failure mode users
/// have repeatedly hit with the global form on the 743b branch).
///
/// Allocations live in `gpa` and are never freed.
pub fn parseFromEnv(env: *const std.process.Environ.Map, gpa: std.mem.Allocator) void {
    // Warn if both forms are set — we silently prefer the strictly-
    // more-expressive `_PASSES`, but a contradiction is worth
    // surfacing so users aren't confused by which spec is active.
    if (env.get("WAMR_AOT_SKIP_PASS") != null and env.get("WAMR_AOT_SKIP_PASSES") != null) {
        std.log.warn("[#761 bisect] both WAMR_AOT_SKIP_PASS and WAMR_AOT_SKIP_PASSES set — using WAMR_AOT_SKIP_PASSES", .{});
    }

    // Parse SKIP first. `appendSkipsFromMulti` is *atomic*: any parse
    // error leaves `skips` empty so a single typo in
    // `WAMR_AOT_SKIP_PASSES="3;bad"` does NOT silently apply the
    // pass-3 skip while warning that the env var was ignored.
    var skips: std.ArrayList(Skip) = .empty;
    if (env.get("WAMR_AOT_SKIP_PASSES")) |v| {
        appendSkipsFromMulti(&skips, v, gpa) catch |e| {
            std.log.warn("[#761 bisect] WAMR_AOT_SKIP_PASSES={s}: {s} — ignoring entire spec", .{ v, @errorName(e) });
        };
    } else if (env.get("WAMR_AOT_SKIP_PASS")) |v| {
        if (parseSkipSpec(v, gpa)) |s| {
            skips.append(gpa, s) catch {};
        } else |e| {
            std.log.warn("[#761 bisect] WAMR_AOT_SKIP_PASS={s}: {s} — ignoring", .{ v, @errorName(e) });
        }
    }
    if (skips.items.len > 0) {
        global.skip = skips.toOwnedSlice(gpa) catch &.{};
        for (global.skip) |s| {
            logSkip(s);
        }
    }

    if (env.get("WAMR_AOT_PASSES_LIMIT")) |v| {
        if (parseLimitSpec(v, gpa)) |lim| {
            global.limit = lim;
            logLimit(lim);
        } else |e| {
            std.log.warn("[#761 bisect] WAMR_AOT_PASSES_LIMIT={s}: {s} — ignoring", .{ v, @errorName(e) });
        }
    }
}

/// Parse a `;`-joined sequence of skip specs into `out`. Atomic: on
/// any per-spec parse error the staging buffer is discarded and `out`
/// is left untouched, so callers don't see partially-applied filters.
fn appendSkipsFromMulti(out: *std.ArrayList(Skip), text: []const u8, gpa: std.mem.Allocator) ParseError!void {
    var staging: std.ArrayList(Skip) = .empty;
    errdefer {
        for (staging.items) |s| {
            if (s.func_filter) |f| gpa.free(f);
            if (s.mod_filter) |m| gpa.free(m);
        }
        staging.deinit(gpa);
    }
    var it = std.mem.splitScalar(u8, text, ';');
    while (it.next()) |raw| {
        const t = std.mem.trim(u8, raw, " \t");
        if (t.len == 0) continue;
        const s = try parseSkipSpec(t, gpa);
        try staging.append(gpa, s);
    }
    if (staging.items.len == 0) return error.EmptyInput;
    // Commit to the caller's list.
    try out.appendSlice(gpa, staging.items);
    staging.deinit(gpa);
}

/// Parse a single skip spec, e.g. `"15"`, `"15-19"`, `"15:fn=11040"`,
/// `"15-19:fn=11040,11050-11060"`, `"15:mod=4:fn=0-2079"`. The `:fn=`
/// and `:mod=` clauses may appear in either order.
pub fn parseSkipSpec(text: []const u8, gpa: std.mem.Allocator) ParseError!Skip {
    const trimmed = std.mem.trim(u8, text, " \t");
    if (trimmed.len == 0) return error.EmptyInput;
    const head, const rest = splitOnce(trimmed, ':');
    const lo, const hi = try parsePassRange(head);
    var fn_filter: ?[]const u32 = null;
    var mod_filter: ?[]const u32 = null;
    errdefer {
        if (fn_filter) |f| gpa.free(f);
        if (mod_filter) |m| gpa.free(m);
    }
    if (rest) |r| try parseFnAndModFilters(r, gpa, &fn_filter, &mod_filter);
    return .{
        .pass_idx_lo = lo,
        .pass_idx_hi = hi,
        .func_filter = fn_filter,
        .mod_filter = mod_filter,
    };
}

/// Parse a single limit spec, e.g. `"30"`, `"30:fn=11040"`,
/// `"30:mod=4:fn=0-2079"`. The `:fn=` and `:mod=` clauses may appear
/// in either order.
pub fn parseLimitSpec(text: []const u8, gpa: std.mem.Allocator) ParseError!Limit {
    const trimmed = std.mem.trim(u8, text, " \t");
    if (trimmed.len == 0) return error.EmptyInput;
    const head, const rest = splitOnce(trimmed, ':');
    const n = parseU32(head) catch return error.InvalidNumber;
    var fn_filter: ?[]const u32 = null;
    var mod_filter: ?[]const u32 = null;
    errdefer {
        if (fn_filter) |f| gpa.free(f);
        if (mod_filter) |m| gpa.free(m);
    }
    if (rest) |r| try parseFnAndModFilters(r, gpa, &fn_filter, &mod_filter);
    return .{ .n = n, .func_filter = fn_filter, .mod_filter = mod_filter };
}

/// Parse the trailing `fn=...[:mod=...]` or `mod=...[:fn=...]` after
/// the head of a skip/limit spec. Each clause may appear at most once;
/// a duplicate `fn=` or `mod=` returns `error.UnknownTrailing` (same
/// shape as an unrecognised prefix to keep error semantics uniform).
fn parseFnAndModFilters(
    rest: []const u8,
    gpa: std.mem.Allocator,
    fn_filter: *?[]const u32,
    mod_filter: *?[]const u32,
) ParseError!void {
    var remaining: ?[]const u8 = rest;
    while (remaining) |r| {
        const t = std.mem.trim(u8, r, " \t");
        if (t.len == 0) break;
        // splitOnce on ':' splits at the FIRST colon. So for
        // "fn=0-2079:mod=4" head="fn=0-2079" and tail="mod=4".
        const head, const tail = splitOnce(t, ':');
        if (stripFnPrefix(head)) |fn_text| {
            if (fn_filter.* != null) return error.UnknownTrailing;
            fn_filter.* = try parseFnList(fn_text, gpa);
        } else if (stripModPrefix(head)) |mod_text| {
            if (mod_filter.* != null) return error.UnknownTrailing;
            mod_filter.* = try parseFnList(mod_text, gpa);
        } else {
            return error.UnknownTrailing;
        }
        remaining = tail;
    }
}

/// Parse a comma-separated function list with optional inclusive
/// ranges, e.g. `"11040"`, `"11040,11050-11055,11100"`. Returns a
/// sorted, de-duplicated slice owned by `gpa`.
pub fn parseFnList(text: []const u8, gpa: std.mem.Allocator) ParseError![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |raw| {
        const t = std.mem.trim(u8, raw, " \t");
        if (t.len == 0) continue;
        const lo, const hi = try parseFnRange(t);
        // Defensive cap: a typo like `fn=0-4000000000` would otherwise
        // allocate ~16 GB. The keyvault component has ~12 k functions;
        // 1 M is a generous ceiling that still rejects the typo case.
        if (hi - lo > 1_000_000) return error.InvalidRange;
        var v = lo;
        while (true) : (v += 1) {
            try out.append(gpa, v);
            if (v == hi) break;
        }
    }
    if (out.items.len == 0) return error.EmptyInput;
    return try out.toOwnedSlice(gpa);
}

fn parsePassRange(text: []const u8) ParseError!struct { u32, u32 } {
    const lo, const hi = try parseRangeBody(text);
    return .{ lo, hi };
}

fn parseFnRange(text: []const u8) ParseError!struct { u32, u32 } {
    return parseRangeBody(text);
}

fn parseRangeBody(text: []const u8) ParseError!struct { u32, u32 } {
    if (std.mem.indexOfScalar(u8, text, '-')) |i| {
        if (i == 0 or i == text.len - 1) return error.InvalidRange;
        const lo = parseU32(text[0..i]) catch return error.InvalidNumber;
        const hi = parseU32(text[i + 1 ..]) catch return error.InvalidNumber;
        if (hi < lo) return error.InvalidRange;
        return .{ lo, hi };
    }
    const v = parseU32(text) catch return error.InvalidNumber;
    return .{ v, v };
}

fn parseU32(text: []const u8) !u32 {
    const trimmed = std.mem.trim(u8, text, " \t");
    return std.fmt.parseInt(u32, trimmed, 10);
}

fn stripFnPrefix(text: []const u8) ?[]const u8 {
    const t = std.mem.trim(u8, text, " \t");
    const prefix = "fn=";
    if (!std.mem.startsWith(u8, t, prefix)) return null;
    return t[prefix.len..];
}

fn stripModPrefix(text: []const u8) ?[]const u8 {
    const t = std.mem.trim(u8, text, " \t");
    const prefix = "mod=";
    if (!std.mem.startsWith(u8, t, prefix)) return null;
    return t[prefix.len..];
}

fn splitOnce(text: []const u8, sep: u8) struct { []const u8, ?[]const u8 } {
    if (std.mem.indexOfScalar(u8, text, sep)) |i| {
        return .{ text[0..i], text[i + 1 ..] };
    }
    return .{ text, null };
}

fn logSkip(s: Skip) void {
    var fn_buf: [64]u8 = undefined;
    var mod_buf: [64]u8 = undefined;
    const fn_part = if (s.func_filter) |list|
        std.fmt.bufPrint(&fn_buf, " for {d} func(s)", .{list.len}) catch ""
    else
        "";
    const mod_part = if (s.mod_filter) |list|
        std.fmt.bufPrint(&mod_buf, " in {d} module(s)", .{list.len}) catch ""
    else
        "";
    const scope_suffix = if (fn_part.len == 0 and mod_part.len == 0) " (all funcs, all modules)" else "";
    if (s.pass_idx_lo == s.pass_idx_hi) {
        std.log.warn(
            "[#761 bisect] SKIP pass {d}{s}{s}{s}",
            .{ s.pass_idx_lo, fn_part, mod_part, scope_suffix },
        );
    } else {
        std.log.warn(
            "[#761 bisect] SKIP passes {d}-{d}{s}{s}{s}",
            .{ s.pass_idx_lo, s.pass_idx_hi, fn_part, mod_part, scope_suffix },
        );
    }
}

fn logLimit(lim: Limit) void {
    var fn_buf: [64]u8 = undefined;
    var mod_buf: [64]u8 = undefined;
    const fn_part = if (lim.func_filter) |list|
        std.fmt.bufPrint(&fn_buf, " for {d} func(s)", .{list.len}) catch ""
    else
        "";
    const mod_part = if (lim.mod_filter) |list|
        std.fmt.bufPrint(&mod_buf, " in {d} module(s)", .{list.len}) catch ""
    else
        "";
    const scope_suffix = if (fn_part.len == 0 and mod_part.len == 0) " (all funcs, all modules)" else "";
    std.log.warn(
        "[#761 bisect] LIMIT pipeline to first {d} passes{s}{s}{s}",
        .{ lim.n, fn_part, mod_part, scope_suffix },
    );
}

// ---------------------------------------------------------------------
// Tests — pure, no env-var or filesystem I/O.
// ---------------------------------------------------------------------

const testing = std.testing;

test "parseSkipSpec: bare pass index" {
    const s = try parseSkipSpec("15", testing.allocator);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_lo);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_hi);
    try testing.expectEqual(@as(?[]const u32, null), s.func_filter);
}

test "parseSkipSpec: pass range" {
    const s = try parseSkipSpec("15-19", testing.allocator);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_lo);
    try testing.expectEqual(@as(u32, 19), s.pass_idx_hi);
    try testing.expectEqual(@as(?[]const u32, null), s.func_filter);
}

test "parseSkipSpec: single pass + fn filter" {
    const s = try parseSkipSpec("15:fn=11040", testing.allocator);
    defer testing.allocator.free(s.func_filter.?);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_lo);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_hi);
    try testing.expectEqualSlices(u32, &.{11040}, s.func_filter.?);
}

test "parseSkipSpec: range + multi-fn filter with sub-range" {
    const s = try parseSkipSpec("15-19:fn=11040,11050-11052", testing.allocator);
    defer testing.allocator.free(s.func_filter.?);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_lo);
    try testing.expectEqual(@as(u32, 19), s.pass_idx_hi);
    try testing.expectEqualSlices(u32, &.{ 11040, 11050, 11051, 11052 }, s.func_filter.?);
}

test "parseSkipSpec: rejects malformed range" {
    try testing.expectError(error.InvalidRange, parseSkipSpec("15-", testing.allocator));
    try testing.expectError(error.InvalidRange, parseSkipSpec("19-15", testing.allocator));
    try testing.expectError(error.InvalidNumber, parseSkipSpec("abc", testing.allocator));
    try testing.expectError(error.UnknownTrailing, parseSkipSpec("15:foo=1", testing.allocator));
    try testing.expectError(error.EmptyInput, parseSkipSpec("", testing.allocator));
}

test "parseSkipSpec: rejects oversize fn range (typo guard)" {
    try testing.expectError(error.InvalidRange, parseSkipSpec("15:fn=0-4000000000", testing.allocator));
}

test "parseLimitSpec: bare + with fn filter" {
    {
        const lim = try parseLimitSpec("30", testing.allocator);
        try testing.expectEqual(@as(u32, 30), lim.n);
        try testing.expectEqual(@as(?[]const u32, null), lim.func_filter);
        try testing.expectEqual(@as(?[]const u32, null), lim.mod_filter);
    }
    {
        const lim = try parseLimitSpec("30:fn=11040,11041", testing.allocator);
        defer testing.allocator.free(lim.func_filter.?);
        try testing.expectEqual(@as(u32, 30), lim.n);
        try testing.expectEqualSlices(u32, &.{ 11040, 11041 }, lim.func_filter.?);
        try testing.expectEqual(@as(?[]const u32, null), lim.mod_filter);
    }
}

test "parseSkipSpec: pass + mod filter" {
    const s = try parseSkipSpec("15:mod=4", testing.allocator);
    defer testing.allocator.free(s.mod_filter.?);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_lo);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_hi);
    try testing.expectEqual(@as(?[]const u32, null), s.func_filter);
    try testing.expectEqualSlices(u32, &.{4}, s.mod_filter.?);
}

test "parseSkipSpec: pass + mod range + fn range" {
    // The keyvault bisect use case: skip pass 15 only in module 4's
    // fns 0-2079 (leaves module 0 unaffected even though it shares
    // that fn-index range).
    const s = try parseSkipSpec("15:mod=4:fn=0-2079", testing.allocator);
    defer testing.allocator.free(s.mod_filter.?);
    defer testing.allocator.free(s.func_filter.?);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_lo);
    try testing.expectEqual(@as(u32, 15), s.pass_idx_hi);
    try testing.expectEqualSlices(u32, &.{4}, s.mod_filter.?);
    try testing.expectEqual(@as(usize, 2080), s.func_filter.?.len);
    try testing.expectEqual(@as(u32, 0), s.func_filter.?[0]);
    try testing.expectEqual(@as(u32, 2079), s.func_filter.?[2079]);
}

test "parseSkipSpec: fn= and mod= order is interchangeable" {
    const a = try parseSkipSpec("15:fn=11040:mod=4", testing.allocator);
    defer testing.allocator.free(a.func_filter.?);
    defer testing.allocator.free(a.mod_filter.?);
    const b = try parseSkipSpec("15:mod=4:fn=11040", testing.allocator);
    defer testing.allocator.free(b.func_filter.?);
    defer testing.allocator.free(b.mod_filter.?);
    try testing.expectEqualSlices(u32, a.func_filter.?, b.func_filter.?);
    try testing.expectEqualSlices(u32, a.mod_filter.?, b.mod_filter.?);
}

test "parseSkipSpec: duplicate fn= clause is rejected" {
    try testing.expectError(error.UnknownTrailing, parseSkipSpec("15:fn=1:fn=2", testing.allocator));
}

test "parseSkipSpec: duplicate mod= clause is rejected" {
    try testing.expectError(error.UnknownTrailing, parseSkipSpec("15:mod=1:mod=2", testing.allocator));
}

test "parseLimitSpec: limit + mod filter" {
    const lim = try parseLimitSpec("30:mod=4", testing.allocator);
    defer testing.allocator.free(lim.mod_filter.?);
    try testing.expectEqual(@as(u32, 30), lim.n);
    try testing.expectEqual(@as(?[]const u32, null), lim.func_filter);
    try testing.expectEqualSlices(u32, &.{4}, lim.mod_filter.?);
}

test "Spec.shouldSkip + effectiveLimit + affectsFunction" {
    const funcs = [_]u32{11040};
    const spec: Spec = .{
        .skip = &.{
            .{ .pass_idx_lo = 15, .pass_idx_hi = 15, .func_filter = &funcs },
            .{ .pass_idx_lo = 17, .pass_idx_hi = 19, .func_filter = null },
        },
        .limit = .{ .n = 30, .func_filter = &funcs },
    };
    // pass 15 skipped only for func 11040, module unconstrained
    try testing.expect(spec.shouldSkip(0, 11040, 15));
    try testing.expect(spec.shouldSkip(4, 11040, 15)); // any module
    try testing.expect(!spec.shouldSkip(0, 11041, 15));
    // pass 17 / 18 / 19 skipped for everyone
    try testing.expect(spec.shouldSkip(0, 0, 17));
    try testing.expect(spec.shouldSkip(0, 99999, 19));
    try testing.expect(!spec.shouldSkip(0, 0, 20));
    // limit applies only to func 11040
    try testing.expectEqual(@as(?u32, 30), spec.effectiveLimit(0, 11040));
    try testing.expectEqual(@as(?u32, null), spec.effectiveLimit(0, 11041));
    // affectsFunction: any of skip/limit
    try testing.expect(spec.affectsFunction(0, 11040));
    try testing.expect(spec.affectsFunction(0, 0)); // hit by pass-17 all-funcs skip
    try testing.expect(spec.affectsFunction(0, 99999)); // same
}

test "Spec.shouldSkip honours mod_filter" {
    const mods = [_]u32{4};
    const spec: Spec = .{
        .skip = &.{
            // Skip pass 15 only in module 4
            .{ .pass_idx_lo = 15, .pass_idx_hi = 15, .mod_filter = &mods },
        },
    };
    // pass 15 in module 4: skipped, any func
    try testing.expect(spec.shouldSkip(4, 0, 15));
    try testing.expect(spec.shouldSkip(4, 11040, 15));
    // pass 15 in other modules: not skipped
    try testing.expect(!spec.shouldSkip(0, 0, 15));
    try testing.expect(!spec.shouldSkip(1, 11040, 15));
    // affectsFunction matches the module
    try testing.expect(spec.affectsFunction(4, 0));
    try testing.expect(!spec.affectsFunction(0, 0));
}

test "Spec.shouldSkip with both mod_filter AND func_filter" {
    const mods = [_]u32{4};
    const fns = [_]u32{ 0, 1, 2, 3 };
    const spec: Spec = .{
        .skip = &.{
            .{
                .pass_idx_lo = 15,
                .pass_idx_hi = 15,
                .mod_filter = &mods,
                .func_filter = &fns,
            },
        },
    };
    // Only matches in module 4 AND fn ∈ {0,1,2,3}
    try testing.expect(spec.shouldSkip(4, 0, 15));
    try testing.expect(spec.shouldSkip(4, 3, 15));
    try testing.expect(!spec.shouldSkip(4, 4, 15)); // wrong fn
    try testing.expect(!spec.shouldSkip(0, 0, 15)); // wrong module
}

test "Spec.effectiveLimit honours mod_filter" {
    const mods = [_]u32{4};
    const spec: Spec = .{
        .limit = .{ .n = 5, .mod_filter = &mods },
    };
    try testing.expectEqual(@as(?u32, 5), spec.effectiveLimit(4, 0));
    try testing.expectEqual(@as(?u32, 5), spec.effectiveLimit(4, 9999));
    try testing.expectEqual(@as(?u32, null), spec.effectiveLimit(0, 0));
    try testing.expectEqual(@as(?u32, null), spec.effectiveLimit(1, 0));
}

test "Spec.isEmpty default" {
    const spec: Spec = .{};
    try testing.expect(spec.isEmpty());
    try testing.expect(!spec.affectsFunction(0, 0));
    try testing.expect(!spec.shouldSkip(0, 0, 0));
    try testing.expectEqual(@as(?u32, null), spec.effectiveLimit(0, 0));
}

test "appendSkipsFromMulti: ';' joined" {
    var skips: std.ArrayList(Skip) = .empty;
    defer {
        for (skips.items) |s| {
            if (s.func_filter) |f| testing.allocator.free(f);
            if (s.mod_filter) |m| testing.allocator.free(m);
        }
        skips.deinit(testing.allocator);
    }
    try appendSkipsFromMulti(&skips, "15;17:fn=11040", testing.allocator);
    try testing.expectEqual(@as(usize, 2), skips.items.len);
    try testing.expectEqual(@as(u32, 15), skips.items[0].pass_idx_lo);
    try testing.expectEqual(@as(u32, 17), skips.items[1].pass_idx_lo);
    try testing.expectEqualSlices(u32, &.{11040}, skips.items[1].func_filter.?);
}

test "appendSkipsFromMulti: atomic on per-spec parse error" {
    // Regression: previously `appendSkipsFromMulti` mutated `out` as
    // it parsed, so `3;abc` would commit a skip for pass 3 while
    // logging that the env var was ignored. Atomic semantics now
    // discard everything on first error.
    var skips: std.ArrayList(Skip) = .empty;
    defer {
        for (skips.items) |s| {
            if (s.func_filter) |f| testing.allocator.free(f);
            if (s.mod_filter) |m| testing.allocator.free(m);
        }
        skips.deinit(testing.allocator);
    }
    try testing.expectError(error.InvalidNumber, appendSkipsFromMulti(&skips, "3;abc", testing.allocator));
    try testing.expectEqual(@as(usize, 0), skips.items.len);
}

test "appendSkipsFromMulti: tolerates trailing or doubled ';'" {
    var skips: std.ArrayList(Skip) = .empty;
    defer {
        for (skips.items) |s| {
            if (s.func_filter) |f| testing.allocator.free(f);
            if (s.mod_filter) |m| testing.allocator.free(m);
        }
        skips.deinit(testing.allocator);
    }
    try appendSkipsFromMulti(&skips, "3;;7;", testing.allocator);
    try testing.expectEqual(@as(usize, 2), skips.items.len);
    try testing.expectEqual(@as(u32, 3), skips.items[0].pass_idx_lo);
    try testing.expectEqual(@as(u32, 7), skips.items[1].pass_idx_lo);
}

test "parseSkipSpec: rejects ';' embedded in single-spec form" {
    // `WAMR_AOT_SKIP_PASS` accepts at most one spec; use _PASSES for
    // multi. We make sure `parseSkipSpec` (not the multi wrapper)
    // rejects.
    try testing.expectError(error.InvalidNumber, parseSkipSpec("3;7", testing.allocator));
}

test "parseLimitSpec: zero pipeline length is legal (skip all passes for fn)" {
    const lim = try parseLimitSpec("0:fn=11040", testing.allocator);
    defer testing.allocator.free(lim.func_filter.?);
    try testing.expectEqual(@as(u32, 0), lim.n);
    try testing.expectEqualSlices(u32, &.{11040}, lim.func_filter.?);
}
