extern "wasi_unstable" fn args_sizes_get(*u32, *u32) u32;
extern "wasi_unstable" fn args_get([*]u32, [*]u8) u32;
extern "wasi_unstable" fn environ_sizes_get(*u32, *u32) u32;
extern "wasi_unstable" fn environ_get([*]u32, [*]u8) u32;
extern "wasi_unstable" fn fd_write(u32, [*]const Iovec, u32, *u32) u32;
extern "wasi_unstable" fn proc_exit(u32) void;
const Iovec = extern struct { pointer: [*]const u8, length: u32 };

export fn args_environment_output() void {
    var count: u32 = 0;
    var size: u32 = 0;
    var pointers: [3]u32 = undefined;
    var bytes: [64]u8 = undefined;
    if (args_sizes_get(&count, &size) != 0 or count != 2 or size != 13) @trap();
    if (args_get(&pointers, &bytes) != 0) @trap();
    if (!equal(bytes[0..13], "fixture\x00arg1\x00")) @trap();
    if (environ_sizes_get(&count, &size) != 0 or count != 1 or size != 4) @trap();
    if (environ_get(&pointers, &bytes) != 0) @trap();
    if (!equal(bytes[0..4], "A=B\x00")) @trap();
    const iov = [_]Iovec{ .{ .pointer = "one", .length = 3 }, .{ .pointer = "two", .length = 3 } };
    var written: u32 = 0;
    if (fd_write(1, &iov, 2, &written) != 0 or written != 6) @trap();
    if (fd_write(2, &iov, 2, &written) != 0 or written != 6) @trap();
}

fn equal(left: []const u8, right: []const u8) bool {
    for (left, right) |a, b| if (a != b) return false;
    return true;
}

export fn exit_zero() void {
    proc_exit(0);
    @trap();
}

export fn exit_nonzero() void {
    proc_exit(23);
    @trap();
}

export fn trap() void {
    @trap();
}

export fn output_exit_nonzero() void {
    args_environment_output();
    proc_exit(23);
    @trap();
}

export fn output_trap() void {
    args_environment_output();
    @trap();
}

export fn binary_output() void {
    const bytes = [_]u8{ 0, 255, 195, 40, 10, 0 };
    const iov = [_]Iovec{.{ .pointer = &bytes, .length = bytes.len }};
    var written: u32 = 0;
    if (fd_write(1, &iov, 1, &written) != 0 or written != bytes.len) @trap();
}
