//! Floating-point averages of two i128 values without overflowing their sum.
const std = @import("std");

pub fn average(first: i128, second: i128) f64 {
    const sum, const overflow = @addWithOverflow(first, second);
    if (overflow == 0) return @as(f64, @floatFromInt(sum)) / 2;
    const negative = first < 0;
    const wrapped: u128 = @bitCast(sum);
    const magnitude = if (negative) 0 -% wrapped else wrapped;
    if (magnitude == 0) return @floatFromInt(std.math.minInt(i128));
    const result = @as(f64, @floatFromInt(magnitude)) / 2;
    return if (negative) -result else result;
}
