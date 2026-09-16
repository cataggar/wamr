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
        const invocation = try session.invoke("_start");
        defer invocation.deinit(std.testing.allocator);
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

test "native benchmark both tracked CoreMark variants real WASI clocks and CRC" {
    for ([_][]const u8{ fixtures.coremark, fixtures.nofp }) |bytes| {
        var pages: linux.Pages = .{};
        const session = try runner.Session.create(std.testing.allocator, pages.platform(), bytes, &.{ "coremark", "0", "0", "0", "100", "0" }, &.{"BENCH_NATIVE=1"}, try linux.wasiClock());
        errdefer session.deinit();
        const invocation = try session.invoke("_start");
        defer invocation.deinit(std.testing.allocator);
        if (!invocation.succeeded()) std.debug.print("CoreMark {s}: {?s}\n{s}\n{s}\n", .{ invocation.outcome, invocation.diagnostic, invocation.stdout, invocation.stderr });
        try expect(invocation.succeeded());
        for ([_][]const u8{ "seedcrc", "0xe9f5", "0xe714", "0x1fd7", "0x8e3a", "Total ticks", "Total time (secs)", "Must execute for at least 10 secs" }) |marker| {
            if (std.mem.indexOf(u8, invocation.stdout, marker) == null) {
                std.debug.print("Missing {s} from:\n{s}\n", .{ marker, invocation.stdout });
                return error.CoreMarkMarkerMissing;
            }
        }
        try expect(invocation.ticks > 0);
        const instance = session.instance.?;
        for (0..2) |_| {
            try session.reset();
            try expect(session.instance.? == instance);
            const repeated = try session.invoke("_start");
            defer repeated.deinit(std.testing.allocator);
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
    pages.fail_protect = true;
    try std.testing.expectError(error.ProtectionFailed, session.reset());
    try equal(size + 65536, session.instance.?.memory().len);
    try equal(@as(u8, 0xaa), session.instance.?.memory()[size]);
    pages.fail_protect = false;
    try session.reset();
    try equal(size, session.instance.?.memory().len);
    try equal(committed, pages.committed());
    try expect(session.instance.?.grow(1) != null);
    try equal(@as(u8, 0), session.instance.?.memory()[size]);
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
    defer invocation.deinit(std.testing.allocator);
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
        defer invocation.deinit(std.testing.allocator);
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
                    defer repeated.deinit(std.testing.allocator);
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
