//! Built to wasm32 by the installed matching Zig, then compiled by this
//! checkout's host-side wamrc with --profile=unikraft-x86_64.
extern "env" fn host_add(i32, i32) i32;
extern "env" fn host_exit(u32) void;
extern "env" fn host_fail() void;

export fn add(a: i32, b: i32) i32 {
    return a +% b;
}
export fn add64(a: i64, b: i64) i64 {
    return a +% b;
}
export fn add_float(a: f64, b: f64) f64 {
    return a + b;
}
fn twice(value: i32) callconv(.c) i32 {
    return value *% 2;
}
fn thrice(value: i32) callconv(.c) i32 {
    return value *% 3;
}
var callbacks = [_]*const fn (i32) callconv(.c) i32{ twice, thrice };
export fn indirect(index: u32, value: i32) i32 {
    const pointer: *volatile *const fn (i32) callconv(.c) i32 = &callbacks[index & 1];
    return pointer.*(value);
}
export fn noop() void {}
export fn load(address: u32) u32 {
    return @as(*allowzero align(1) const u32, @ptrFromInt(address)).*;
}
export fn store(address: u32, value: u32) void {
    @as(*allowzero align(1) u32, @ptrFromInt(address)).* = value;
}
export fn grow(delta: u32) i32 {
    return @intCast(@wasmMemoryGrow(0, delta));
}
export fn size() i32 {
    return @intCast(@wasmMemorySize(0));
}
export fn trap() void {
    @trap();
}
export fn divide(a: i32, b: i32) i32 {
    return @divTrunc(a, b);
}
export fn call_host(a: i32, b: i32) i32 {
    return host_add(a, b);
}
export fn exit(code: u32) void {
    host_exit(code);
    // Must never execute after the host terminal result.
    @as(*allowzero align(1) u32, @ptrFromInt(64)).* = 0xbad;
}
export fn fail() void {
    host_fail();
    @as(*allowzero align(1) u32, @ptrFromInt(64)).* = 0xbad;
}
