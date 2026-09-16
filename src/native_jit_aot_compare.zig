//! Separately linked AOT comparator. The comptime-false driver has no compiler.
pub fn main(init: @import("std").process.Init) !void {
    return @import("bench/native_jit_driver.zig").main(false, init);
}
