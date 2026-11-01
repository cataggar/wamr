//! Accurately rounded summation for benchmark measurements.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ExactSum = struct {
    partials: std.ArrayList(f64) = .empty,

    pub fn add(self: *ExactSum, allocator: Allocator, value: f64) !void {
        if (!std.math.isFinite(value)) return error.InvalidMeasurement;
        var x = value;
        var count: usize = 0;
        for (self.partials.items) |partial| {
            var y = partial;
            if (@abs(x) < @abs(y)) std.mem.swap(f64, &x, &y);
            const hi = x + y;
            if (!std.math.isFinite(hi)) return error.MeasurementTooLarge;
            const lo = y - (hi - x);
            if (lo != 0) {
                self.partials.items[count] = lo;
                count += 1;
            }
            x = hi;
        }
        self.partials.shrinkRetainingCapacity(count);
        try self.partials.append(allocator, x);
    }

    pub fn result(self: *const ExactSum) f64 {
        var count = self.partials.items.len;
        if (count == 0) return 0;
        count -= 1;
        var hi = self.partials.items[count];
        var lo: f64 = 0;
        while (count > 0) {
            count -= 1;
            const x = hi;
            const y = self.partials.items[count];
            hi = x + y;
            lo = y - (hi - x);
            if (lo != 0) break;
        }
        // Account for remaining partials when the first discarded bit is a tie.
        if (count > 0 and ((lo < 0 and self.partials.items[count - 1] < 0) or
            (lo > 0 and self.partials.items[count - 1] > 0)))
        {
            const y = lo * 2;
            const x = hi + y;
            if (x - hi == y) hi = x;
        }
        return hi;
    }

    pub fn deinit(self: *ExactSum, allocator: Allocator) void {
        self.partials.deinit(allocator);
    }
};
