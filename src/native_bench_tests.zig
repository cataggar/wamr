const std = @import("std");
const linux = @import("bench/native_linux.zig");
const runner = @import("bench/native_runner.zig");
const aot = runner.aot;
const fixtures = @import("fixtures");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

test {
    _ = @import("bench/native_json.zig");
    _ = @import("bench/native_allocations.zig");
    _ = linux;
}

test "native benchmark deterministic real compute and memory repeated exports" {
    var pages: linux.Pages = .{};
    const session = try runner.Session.create(std.testing.allocator, pages.platform(), fixtures.deterministic, &.{}, &.{}, try linux.wasiClock());
    const instance = session.instance.?;
    var values: [1]aot.Value = undefined;
    for (0..4) |i| {
        const seed: i32 = @intCast(42 + i);
        try equal(@as(usize, 1), (try instance.call("compute", &.{.{ .i32 = seed }}, &values)).returned);
        var expected: u32 = @intCast(seed);
        for (0..4096) |j| expected = (expected *% 1664525 +% 1013904223) ^ @as(u32, @intCast(j));
        try equal(expected, @as(u32, @bitCast(values[0].i32)));
        try equal(@as(usize, 1), (try instance.call("memory_checksum", &.{.{ .i32 = seed }}, &values)).returned);
        try equal(@as(i32, 570026) + @as(i32, @intCast(i * 257)), values[0].i32);
        try equal(@as(usize, 1), (try instance.call("memory_base", &.{}, &values)).returned);
        const base: usize = @intCast(values[0].i32);
        for ([_]usize{ 0, 128, 256 }) |cell| {
            const stored = std.mem.readInt(u32, instance.memory()[base + cell * 4 ..][0..4], .little);
            try equal(@as(u32, @intCast(seed)) + @as(u32, @intCast(cell * 17)), stored);
        }
        const invocation = try session.invoke("_start");
        try expect(invocation.succeeded());
        try equal(@as(usize, 0), invocation.stdout.len);
    }
    session.deinit();
    try equal(@as(usize, 0), pages.reserved());
}

test "native benchmark phase-staged load does not instantiate or call the guest" {
    var pages: linux.Pages = .{};
    const instance = try aot.Instance.loadModule(std.testing.allocator, pages.platform(), fixtures.deterministic, .{});
    defer instance.deinit();
    try equal(@as(usize, 0), pages.reserved());
    try std.testing.expectError(error.NotInstantiated, instance.start());
    try std.testing.expectError(error.NotInstantiated, instance.call("_start", &.{}, &.{}));
    try equal(@as(?u32, null), instance.grow(1));
    try instance.instantiate(&.{}, .{});
    try expect(pages.reserved() > 0);
    try std.testing.expectError(error.Busy, instance.instantiate(&.{}, .{}));
    try equal(@as(usize, 0), (try instance.start()).returned);
    try equal(@as(usize, 0), (try instance.call("_start", &.{}, &.{})).returned);
}

test "native benchmark phase convenience load preserves optional instrumentation" {
    var pages: linux.Pages = .{};
    var timings: aot.LoadTimings = .{};
    const instance = try aot.Instance.load(std.testing.allocator, pages.platform(), fixtures.deterministic, &.{}, .{ .timings = &timings });
    instance.deinit();
    try equal(@as(u32, 3), timings.completed);
    try equal(@as(usize, 3), pages.clock_reads);
    try expect(timings.load_ns > 0 and timings.instantiate_ns > 0);
    try equal(@as(usize, 0), pages.reserved());
    pages = .{ .fail_clock_at = 3 };
    try std.testing.expectError(error.ClockFailed, aot.Instance.load(
        std.testing.allocator,
        pages.platform(),
        fixtures.deterministic,
        &.{},
        .{ .timings = &timings },
    ));
    try equal(@as(u32, 1), timings.completed);
    try equal(@as(usize, 0), pages.reserved());
}

test "native benchmark phase-staged admission freezes every non-timing option" {
    var pages: linux.Pages = .{};
    const admitted: aot.Options = .{ .max_memory_pages = 8, .max_table_elements = 128 };
    inline for (.{ "max_memory_pages", "max_table_elements", "cpu_feature_mask" }) |field| {
        const instance = try aot.Instance.loadModule(std.testing.allocator, pages.platform(), fixtures.deterministic, admitted);
        defer instance.deinit();
        var changed = admitted;
        @field(changed, field) ^= 1;
        try std.testing.expectError(error.OptionsMismatch, instance.instantiate(&.{}, changed));
        try equal(@as(usize, 0), pages.reserved());
        try std.testing.expectError(error.NotInstantiated, instance.call("_start", &.{}, &.{}));
        try std.testing.expectError(error.Busy, instance.instantiate(&.{}, admitted));
    }
    const instance = try aot.Instance.loadModule(std.testing.allocator, pages.platform(), fixtures.deterministic, admitted);
    defer instance.deinit();
    var timings: aot.LoadTimings = .{};
    var with_timings = admitted;
    with_timings.timings = &timings;
    try instance.instantiate(&.{}, with_timings);
    try expect(instance.instantiated);
}

test "native benchmark both tracked CoreMark variants real WASI clocks and CRC" {
    for ([_][]const u8{ fixtures.coremark, fixtures.nofp }) |bytes| {
        var pages: linux.Pages = .{};
        const session = try runner.Session.create(std.testing.allocator, pages.platform(), bytes, &.{ "coremark", "0", "0", "0", "100", "0" }, &.{"BENCH_NATIVE=1"}, try linux.wasiClock());
        errdefer session.deinit();
        const invocation = try session.invoke("_start");
        if (!invocation.succeeded()) std.debug.print("CoreMark {s}: {?s}\n{s}\n{s}\n", .{ invocation.outcome, invocation.diagnostic, invocation.stdout, invocation.stderr });
        try expect(invocation.succeeded());
        for ([_][]const u8{ "seedcrc", "0xe9f5", "0xe714", "0x1fd7", "0x8e3a", "Total ticks", "Total time (secs)", "Must execute for at least 10 secs" }) |marker| {
            if (std.mem.indexOf(u8, invocation.stdout, marker) == null) {
                std.debug.print("Missing {s} from:\n{s}\n", .{ marker, invocation.stdout });
                return error.CoreMarkMarkerMissing;
            }
        }
        try expect(invocation.ticks.? > 0);
        const instance = session.instance.?;
        for (0..2) |_| {
            try session.reset();
            try expect(session.instance.? == instance);
            const repeated = try session.invoke("_start");
            try expect(repeated.succeeded());
            for ([_][]const u8{ "0xe9f5", "0xe714", "0x1fd7", "0x8e3a", "0x988c" }) |crc|
                try expect(std.mem.indexOf(u8, repeated.stdout, crc) != null);
        }
        session.deinit();
        try equal(@as(usize, 0), pages.reserved());
    }
}

test "native benchmark snapshot reset revokes grown pages and restores logical bounds" {
    var pages: linux.Pages = .{};
    const session = try runner.Session.create(std.testing.allocator, pages.platform(), fixtures.deterministic, &.{}, &.{}, try linux.wasiClock());
    defer session.deinit();
    const size = session.instance.?.memory().len;
    const committed = pages.committed();
    try expect(session.instance.?.grow(1) != null);
    session.instance.?.memory()[size] = 0xaa;
    try session.reset();
    try equal(size, session.instance.?.memory().len);
    try equal(committed, pages.committed());
    try expect(session.instance.?.grow(1) != null);
    try equal(@as(u8, 0), session.instance.?.memory()[size]);
}

test "native benchmark snapshot reset protection failure poisons even a partially changed mapping" {
    for ([_]bool{ false, true }) |partial| {
        var pages: linux.Pages = .{};
        const session = try runner.Session.create(std.testing.allocator, pages.platform(), fixtures.deterministic, &.{}, &.{}, try linux.wasiClock());
        errdefer session.deinit();
        try expect(session.instance.?.grow(1) != null);
        pages.fail_protect = !partial;
        pages.fail_protect_after_transition = partial;
        pages.probe_instance = session.instance.?;
        try std.testing.expectError(error.ProtectionFailed, session.reset());
        try equal(@as(?bool, false), pages.revocation_observed_callable);
        try equal(partial, pages.partial_protection_applied);
        try std.testing.expectError(error.NotInstantiated, session.instance.?.call("_start", &.{}, &.{}));
        try std.testing.expectError(error.NotInstantiated, session.instance.?.start());
        try equal(@as(?u32, null), session.instance.?.grow(1));
        try std.testing.expectError(error.NotInstantiated, session.reset());
        session.deinit();
        try equal(@as(usize, 0), pages.reserved());
    }
}

test "native benchmark evidence retains terminal bytes after closing clock failure or reversal" {
    for ([_]bool{ false, true }) |backwards| {
        for ([_][]const u8{ "args_environment_output", "output_exit_nonzero", "output_trap" }, 0..) |entry, index| {
            var pages: linux.Pages = .{};
            const session = try runner.Session.create(std.testing.allocator, pages.platform(), fixtures.wasi, &.{ "fixture", "arg1" }, &.{"A=B"}, try linux.wasiClock());
            defer session.deinit();
            if (backwards) pages.backwards_clock_at = pages.clock_reads + 2 else pages.fail_clock_at = pages.clock_reads + 2;
            const result = try session.invoke(entry);
            try equal(@as(?u64, null), result.ticks);
            try std.testing.expectEqualStrings(if (backwards) "ClockWentBackwards" else "ClockFailed", result.timing_error.?);
            try std.testing.expectEqualStrings(([_][]const u8{ "returned", "proc_exit", "trap" })[index], result.outcome);
            try equal(@as(?u32, if (index == 1) 23 else null), result.exit_code);
            try std.testing.expectEqualStrings("onetwo", result.stdout);
            try std.testing.expectEqualStrings("onetwo", result.stderr);
            try expect(!result.succeeded());
            var bytes: [2048]u8 = undefined;
            var writer = std.Io.Writer.fixed(&bytes);
            try runner.writeInvocationEvidence(&writer, "first", 0, result);
            const prefix = "WAMR_NATIVE_INVOCATION=";
            const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, writer.buffered()[prefix.len..], .{});
            defer parsed.deinit();
            try expect(parsed.value.object.get("ticks").? == .null);
            try std.testing.expectEqualStrings("b25ldHdv", parsed.value.object.get("stdout_base64").?.string);
            try std.testing.expectEqualStrings(result.outcome, parsed.value.object.get("outcome").?.string);
        }
    }
}

test "native benchmark evidence needs no post-call allocation and preserves binary stdout" {
    const Counter = @import("bench/native_allocations.zig").Counter;
    var counter: Counter = .{ .child = std.testing.allocator };
    const allocator = counter.allocator();
    var pages: linux.Pages = .{};
    const session = try runner.Session.create(allocator, pages.platform(), fixtures.wasi, &.{ "fixture", "arg1" }, &.{"A=B"}, try linux.wasiClock());
    errdefer session.deinit();
    try session.output.stdout.ensureTotalCapacity(allocator, 64);
    try session.output.stderr.ensureTotalCapacity(allocator, 64);
    counter.fail_allocations = true;
    try std.testing.expectError(error.OutOfMemory, allocator.alloc(u8, 1));
    const result = try session.invoke("binary_output");
    try expect(result.succeeded());
    try std.testing.expectEqualSlices(u8, &.{ 0, 255, 195, 40, 10, 0 }, result.stdout);
    var bytes: [2048]u8 = undefined;
    var writer = std.Io.Writer.fixed(&bytes);
    try runner.writeInvocationEvidence(&writer, "first", 0, result);
    try expect(std.mem.indexOf(u8, writer.buffered(), "\"stdout_base64\":\"AP/DKAoA\"") != null);
    session.deinit();
    try equal(@as(usize, 0), counter.live);
    try equal(@as(usize, 0), pages.reserved());
}

test "native benchmark evidence separates trap output failure and closing clock failure" {
    var pages: linux.Pages = .{};
    const session = try runner.Session.create(std.testing.allocator, pages.platform(), fixtures.wasi, &.{}, &.{}, try linux.wasiClock());
    defer session.deinit();
    session.output.limit = 0;
    pages.fail_clock_at = pages.clock_reads + 2;
    const result = try session.invoke("binary_output");
    try std.testing.expectEqualStrings("trap", result.outcome);
    try std.testing.expectEqualStrings("unreachable_instruction", result.diagnostic.?);
    try std.testing.expectEqualStrings("ClockFailed", result.timing_error.?);
    try expect(result.output_failure);
    try equal(@as(?u64, null), result.ticks);
    try equal(@as(?u32, null), result.exit_code);
}

test "native benchmark failed phase clocks release loaded and mapped resources" {
    for (1..5) |index| {
        var pages: linux.Pages = .{ .fail_clock_at = index };
        try std.testing.expectError(error.ClockFailed, runner.Session.create(
            std.testing.allocator,
            pages.platform(),
            fixtures.deterministic,
            &.{},
            &.{},
            try linux.wasiClock(),
        ));
        try equal(@as(usize, 0), pages.reserved());
    }
}

test "native benchmark empty artifacts and actual backend resource failure cleanup" {
    var pages: linux.Pages = .{};
    if (runner.Session.create(std.testing.allocator, pages.platform(), &.{}, &.{}, &.{}, try linux.wasiClock())) |session| {
        session.deinit();
        return error.EmptyArtifactAccepted;
    } else |_| {}
    try equal(@as(usize, 0), pages.reserved());
    for (0..3) |index| {
        pages.fail_reserve = index == 0;
        pages.fail_commit = index == 1;
        pages.fail_protect = index == 2;
        if (runner.Session.create(std.testing.allocator, pages.platform(), fixtures.deterministic, &.{}, &.{}, try linux.wasiClock())) |session| {
            session.deinit();
            return error.InjectedFailureIgnored;
        } else |_| {}
        try equal(@as(usize, 0), pages.reserved());
    }
}

test "native benchmark partial output failure preserves consumed bytes and fails attempt" {
    var pages: linux.Pages = .{};
    const session = try runner.Session.create(std.testing.allocator, pages.platform(), fixtures.coremark, &.{ "coremark", "0", "0", "0", "10", "0" }, &.{}, try linux.wasiClock());
    defer session.deinit();
    session.output.fail_after = 7;
    const invocation = try session.invoke("_start");
    try std.testing.expectEqualStrings("returned", invocation.outcome);
    try expect(invocation.output_failure);
    try equal(@as(usize, 7), invocation.stdout.len + invocation.stderr.len);
    try expect(!invocation.succeeded());
}

test "native benchmark real argv environ stdout stderr exits and traps" {
    for ([_][]const u8{ "args_environment_output", "exit_zero", "exit_nonzero", "trap", "missing" }, 0..) |entry, index| {
        var pages: linux.Pages = .{};
        const session = try runner.Session.create(std.testing.allocator, pages.platform(), fixtures.wasi, &.{ "fixture", "arg1" }, &.{"A=B"}, try linux.wasiClock());
        errdefer session.deinit();
        const invocation = try session.invoke(entry);
        switch (index) {
            0 => {
                try std.testing.expectEqualStrings("returned", invocation.outcome);
                try std.testing.expectEqualStrings("onetwo", invocation.stdout);
                try std.testing.expectEqualStrings("onetwo", invocation.stderr);
            },
            1, 2 => {
                try std.testing.expectEqualStrings("proc_exit", invocation.outcome);
                try equal(@as(?u32, if (index == 1) 0 else 23), invocation.exit_code);
                if (index == 1) {
                    try equal(@as(?u32, 0), session.context.exit_code);
                    try session.reset();
                    try equal(@as(?u32, null), session.context.exit_code);
                    const repeated = try session.invoke("args_environment_output");
                    try expect(repeated.succeeded());
                    try std.testing.expectEqualStrings("onetwo", repeated.stdout);
                    try std.testing.expectEqualStrings("onetwo", repeated.stderr);
                }
            },
            3 => {
                try std.testing.expectEqualStrings("trap", invocation.outcome);
                try equal(@as(?u32, null), invocation.exit_code);
            },
            else => {
                try std.testing.expectEqualStrings("error", invocation.outcome);
                try std.testing.expectEqualStrings("FunctionNotFound", invocation.diagnostic.?);
            },
        }
        session.deinit();
        try equal(@as(usize, 0), pages.reserved());
    }
}
