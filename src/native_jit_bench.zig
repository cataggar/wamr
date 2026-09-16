pub fn main(init: @import("std").process.Init) !void {
    return @import("bench/native_jit_driver.zig").main(true, init);
}
