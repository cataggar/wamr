//! Fixed-point rendering with round-to-nearest, ties-to-even.
const std = @import("std");

pub fn fixed(allocator: std.mem.Allocator, value: f64, comptime precision: u8) ![]u8 {
    if (!std.math.isFinite(value) or value < 0) return error.InvalidMeasurement;
    const bits: u64 = @bitCast(value);
    const exponent: u16 = @intCast((bits >> 52) & 0x7ff);
    const fraction = bits & ((@as(u64, 1) << 52) - 1);
    const mantissa: u128 = if (exponent == 0) fraction else fraction | (@as(u64, 1) << 52);
    const scale = comptime std.math.pow(u128, 10, precision);
    const numerator = try std.math.mul(u128, mantissa, scale);
    const power: i32 = (if (exponent == 0) @as(i32, -1022) else @as(i32, exponent) - 1023) - 52;
    var rounded: u128 = 0;
    if (power >= 0) {
        if (power >= 128 or numerator > @as(u128, std.math.maxInt(u128)) >> @as(u7, @intCast(power)))
            return error.MeasurementTooLarge;
        rounded = numerator << @as(u7, @intCast(power));
    } else {
        const shift: u32 = @intCast(-power);
        if (shift < 128) {
            const amount: u7 = @intCast(shift);
            rounded = numerator >> amount;
            const mask = (@as(u128, 1) << amount) - 1;
            const remainder = numerator & mask;
            const half = @as(u128, 1) << @as(u7, @intCast(shift - 1));
            if (remainder > half or (remainder == half and rounded % 2 != 0))
                rounded = try std.math.add(u128, rounded, 1);
        }
    }
    return switch (precision) {
        0 => std.fmt.allocPrint(allocator, "{d}", .{rounded}),
        1 => std.fmt.allocPrint(allocator, "{d}.{d:0>1}", .{ rounded / scale, rounded % scale }),
        2 => std.fmt.allocPrint(allocator, "{d}.{d:0>2}", .{ rounded / scale, rounded % scale }),
        3 => std.fmt.allocPrint(allocator, "{d}.{d:0>3}", .{ rounded / scale, rounded % scale }),
        6 => std.fmt.allocPrint(allocator, "{d}.{d:0>6}", .{ rounded / scale, rounded % scale }),
        else => @compileError("unsupported fixed-point precision"),
    };
}

fn mulPow10(big: *std.math.big.int.Managed, allocator: std.mem.Allocator, exp: u32) !void {
    var ten = try std.math.big.int.Managed.initSet(allocator, 10);
    defer ten.deinit();
    var factor = try std.math.big.int.Managed.initSet(allocator, 1);
    defer factor.deinit();
    var remaining = exp;
    while (remaining > 0) : (remaining >>= 1) {
        if (remaining & 1 != 0) try factor.mul(&factor, &ten);
        if (remaining > 1) try ten.mul(&ten, &ten);
    }
    try big.mul(big, &factor);
}

fn buildRationalAt10(
    allocator: std.mem.Allocator,
    mantissa: u64,
    power2: i32,
    power10: i32,
) !struct { num: std.math.big.int.Managed, den: std.math.big.int.Managed } {
    var num = try std.math.big.int.Managed.initSet(allocator, mantissa);
    errdefer num.deinit();
    var den = try std.math.big.int.Managed.initSet(allocator, 1);
    errdefer den.deinit();
    if (power10 < 0) {
        try mulPow10(&num, allocator, @intCast(-power10));
    } else if (power10 > 0) {
        try mulPow10(&den, allocator, @intCast(power10));
    }
    if (power2 > 0) {
        try num.shiftLeft(&num, @intCast(power2));
    } else if (power2 < 0) {
        try den.shiftLeft(&den, @intCast(-power2));
    }
    return .{ .num = num, .den = den };
}

pub fn general(allocator: std.mem.Allocator, value: f64) ![]u8 {
    if (std.math.isNan(value)) return allocator.dupe(u8, "nan");
    const bits: u64 = @bitCast(value);
    const negative = (bits >> 63) != 0;
    if (std.math.isInf(value)) return allocator.dupe(u8, if (negative) "-inf" else "inf");
    const abs_val = @abs(value);
    if (abs_val == 0) return allocator.dupe(u8, if (negative) "-0" else "0");

    const exponent: u16 = @intCast((bits >> 52) & 0x7ff);
    const fraction = bits & ((@as(u64, 1) << 52) - 1);
    const mantissa: u64 = if (exponent == 0) fraction else fraction | (@as(u64, 1) << 52);
    const power2: i32 = (if (exponent == 0) @as(i32, -1022) else @as(i32, exponent) - 1023) - 52;

    var e10: i32 = @as(i32, @intFromFloat(@floor(@log10(abs_val)))) - 1;
    while (true) {
        var rat = try buildRationalAt10(allocator, mantissa, power2, e10 + 1);
        defer rat.num.deinit();
        defer rat.den.deinit();
        if (rat.num.order(rat.den) == .lt) break;
        e10 += 1;
    }

    var rat = try buildRationalAt10(allocator, mantissa, power2, e10 - 5);
    defer rat.num.deinit();
    defer rat.den.deinit();
    var q_big = try std.math.big.int.Managed.init(allocator);
    defer q_big.deinit();
    var rem = try std.math.big.int.Managed.init(allocator);
    defer rem.deinit();
    try q_big.divTrunc(&rem, &rat.num, &rat.den);
    try rem.shiftLeft(&rem, 1);
    var q = try q_big.toInt(u32);
    const cmp = rem.order(rat.den);
    if (cmp == .gt or (cmp == .eq and q % 2 == 1)) q += 1;
    if (q == 1_000_000) {
        q = 100_000;
        e10 += 1;
    }

    var digits_buf: [6]u8 = undefined;
    _ = std.fmt.bufPrint(&digits_buf, "{d:0>6}", .{q}) catch unreachable;
    var k: usize = 6;
    while (k > 1 and digits_buf[k - 1] == '0') : (k -= 1) {}
    const digits = digits_buf[0..k];

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    if (negative) try w.writeByte('-');
    if (e10 >= -4 and e10 < 6) {
        if (e10 >= 0) {
            const int_digits: usize = @intCast(e10 + 1);
            const head = @min(k, int_digits);
            try w.writeAll(digits[0..head]);
            for (0..int_digits - head) |_| try w.writeByte('0');
            if (k > int_digits) {
                try w.writeByte('.');
                try w.writeAll(digits[int_digits..k]);
            }
        } else {
            try w.writeAll("0.");
            for (0..@intCast(-e10 - 1)) |_| try w.writeByte('0');
            try w.writeAll(digits);
        }
    } else {
        try w.writeByte(digits[0]);
        if (k > 1) {
            try w.writeByte('.');
            try w.writeAll(digits[1..k]);
        }
        try w.print("e{s}{d:0>2}", .{ if (e10 >= 0) "+" else "-", @abs(e10) });
    }
    return out.toOwnedSlice();
}

test "general matches reference :g formatting" {
    const Case = struct { val: f64, expected: []const u8 };
    const cases = [_]Case{
        .{ .val = 0.0, .expected = "0" },
        .{ .val = -0.0, .expected = "-0" },
        .{ .val = 0.5, .expected = "0.5" },
        .{ .val = 1.0, .expected = "1" },
        .{ .val = 120.0, .expected = "120" },
        .{ .val = 900.0, .expected = "900" },
        .{ .val = 1800.0, .expected = "1800" },
        .{ .val = 123.4567, .expected = "123.457" },
        .{ .val = 123456.0, .expected = "123456" },
        .{ .val = 123456.7, .expected = "123457" },
        .{ .val = 999999.4, .expected = "999999" },
        .{ .val = 999999.5, .expected = "1e+06" },
        .{ .val = 1e6, .expected = "1e+06" },
        .{ .val = 1e-4, .expected = "0.0001" },
        .{ .val = 9.999995e-5, .expected = "0.0001" },
        .{ .val = 1e-5, .expected = "1e-05" },
        .{ .val = 1.234567e-5, .expected = "1.23457e-05" },
        .{ .val = 1.2345e-5, .expected = "1.2345e-05" },
        .{ .val = 1e100, .expected = "1e+100" },
        .{ .val = 125.0005, .expected = "125.001" },
        .{ .val = 125000.5, .expected = "125000" },
        .{ .val = 125001.5, .expected = "125002" },
    };
    for (cases) |case| {
        const text = try general(std.testing.allocator, case.val);
        defer std.testing.allocator.free(text);
        try std.testing.expectEqualStrings(case.expected, text);
    }
}
