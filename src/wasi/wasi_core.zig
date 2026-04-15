//! Pure WASI core logic decoupled from the interpreter's ExecEnv.
//!
//! Each function takes raw arguments and a linear-memory slice, making the
//! logic reusable by both the interpreter path (via host_functions.zig) and
//! AOT/JIT paths that provide memory directly.

const std = @import("std");
const platform = @import("../platform/platform.zig");

// ── WASI errno constants ──────────────────────────────────────────────

pub const WASI_ESUCCESS: i32 = 0;
pub const WASI_EBADF: i32 = 8;
pub const WASI_EINVAL: i32 = 28;
pub const WASI_ENOSYS: i32 = 52;

// ── WASI clock IDs ────────────────────────────────────────────────────

pub const WASI_CLOCK_REALTIME: i32 = 0;
pub const WASI_CLOCK_MONOTONIC: i32 = 1;

// ── Memory helpers ────────────────────────────────────────────────────

/// Read a little-endian u32 from linear memory at the given offset.
pub fn memReadU32(mem: []const u8, offset: u32) ?u32 {
    if (offset + 4 > mem.len) return null;
    return std.mem.readInt(u32, mem[offset..][0..4], .little);
}

/// Write a little-endian u32 to linear memory at the given offset.
pub fn memWriteU32(mem: []u8, offset: u32, val: u32) bool {
    if (offset + 4 > mem.len) return false;
    std.mem.writeInt(u32, mem[offset..][0..4], val, .little);
    return true;
}

/// Write a little-endian u64 to linear memory at the given offset.
pub fn memWriteU64(mem: []u8, offset: u32, val: u64) bool {
    if (offset + 8 > mem.len) return false;
    std.mem.writeInt(u64, mem[offset..][0..8], val, .little);
    return true;
}

// ── Core WASI functions ───────────────────────────────────────────────

/// Core logic for `fd_write`.  Walks the iov array for stdout/stderr,
/// writes the data via `std.debug.print`, and stores the byte count at
/// `nwritten_ptr`.
pub fn fdWriteCore(mem: []u8, fd: i32, iovs_ptr: u32, iovs_len: u32, nwritten_ptr: u32) i32 {
    // Only support stdout (1) and stderr (2)
    if (fd != 1 and fd != 2) return WASI_EBADF;

    var total_written: u32 = 0;
    for (0..iovs_len) |i| {
        const iov_offset = iovs_ptr + @as(u32, @intCast(i)) * 8;
        const buf_ptr = memReadU32(mem, iov_offset) orelse break;
        const buf_len = memReadU32(mem, iov_offset + 4) orelse break;
        if (buf_ptr + buf_len > mem.len) break;
        const data = mem[buf_ptr .. buf_ptr + buf_len];
        std.debug.print("{s}", .{data});
        total_written += buf_len;
    }

    _ = memWriteU32(mem, nwritten_ptr, total_written);
    return WASI_ESUCCESS;
}

/// Core logic for `fd_seek` — not supported, returns `ENOSYS`.
pub fn fdSeekCore() i32 {
    return WASI_ENOSYS;
}

/// Core logic for `fd_close`.  Accepts stdin/stdout/stderr (0-2);
/// returns `EBADF` for everything else.
pub fn fdCloseCore(fd: i32) i32 {
    return if (fd >= 0 and fd <= 2) WASI_ESUCCESS else WASI_EBADF;
}

/// Core logic for `fd_fdstat_get`.  Fills the 24-byte fdstat struct in
/// linear memory for the standard file descriptors.
pub fn fdFdstatGetCore(mem: []u8, fd: i32, buf_ptr: u32) i32 {
    if (fd < 0 or fd > 2) return WASI_EBADF;

    // fdstat struct: fs_filetype(u8) + fs_flags(u16) + padding +
    //               fs_rights_base(u64) + fs_rights_inheriting(u64) = 24 bytes
    if (buf_ptr + 24 > mem.len) return WASI_EINVAL;

    @memset(mem[buf_ptr .. buf_ptr + 24], 0);
    // fs_filetype: 2 = character device (for stdin/stdout/stderr)
    mem[buf_ptr] = 2;
    // fs_rights_base: allow fd_write
    _ = memWriteU64(mem, buf_ptr + 8, 0x1FFFFFFF);

    return WASI_ESUCCESS;
}

/// Core logic for `fd_prestat_get` — no preopened dirs, returns `EBADF`.
pub fn fdPrestatGetCore() i32 {
    return WASI_EBADF;
}

/// Core logic for `fd_prestat_dir_name` — no preopened dirs, returns `EBADF`.
pub fn fdPrestatDirNameCore() i32 {
    return WASI_EBADF;
}

/// Core logic for `clock_time_get`.  Writes the current time in
/// nanoseconds to `time_ptr` for the supported clock IDs.
pub fn clockTimeGetCore(mem: []u8, clock_id: i32, time_ptr: u32) i32 {
    const nanos: u64 = switch (clock_id) {
        WASI_CLOCK_REALTIME, WASI_CLOCK_MONOTONIC => platform.timeGetBootUs() * std.time.ns_per_us,
        else => return WASI_EINVAL,
    };

    _ = memWriteU64(mem, time_ptr, nanos);
    return WASI_ESUCCESS;
}

/// Core logic for `environ_sizes_get`.  Reports zero environment
/// variables by writing 0 to both pointers.
pub fn environSizesGetCore(mem: []u8, count_ptr: u32, buf_size_ptr: u32) i32 {
    _ = memWriteU32(mem, count_ptr, 0);
    _ = memWriteU32(mem, buf_size_ptr, 0);
    return WASI_ESUCCESS;
}

/// Core logic for `environ_get` — no environment variables to populate.
pub fn environGetCore() i32 {
    return WASI_ESUCCESS;
}

/// Core logic for `args_sizes_get`.  Reports zero arguments by writing
/// 0 to both pointers.
pub fn argsSizesGetCore(mem: []u8, count_ptr: u32, buf_size_ptr: u32) i32 {
    _ = memWriteU32(mem, count_ptr, 0);
    _ = memWriteU32(mem, buf_size_ptr, 0);
    return WASI_ESUCCESS;
}

/// Core logic for `args_get` — no arguments to populate.
pub fn argsGetCore() i32 {
    return WASI_ESUCCESS;
}

/// Marker for `proc_exit`.  The actual trap signalling is
/// caller-specific (ExecEnv vs AOT), so this function only represents
/// the intent.  The caller is responsible for triggering the trap.
pub fn procExitCore() void {}

// ── Tests ─────────────────────────────────────────────────────────────

test "memReadU32: reads little-endian value" {
    const data = [_]u8{ 0x78, 0x56, 0x34, 0x12 };
    try std.testing.expectEqual(@as(u32, 0x12345678), memReadU32(&data, 0).?);
}

test "memReadU32: out-of-bounds returns null" {
    const data = [_]u8{ 0x01, 0x02 };
    try std.testing.expect(memReadU32(&data, 0) == null);
}

test "memWriteU32: writes little-endian value" {
    var buf: [4]u8 = undefined;
    try std.testing.expect(memWriteU32(&buf, 0, 0xDEADBEEF));
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), memReadU32(&buf, 0).?);
}

test "memWriteU32: out-of-bounds returns false" {
    var buf: [2]u8 = undefined;
    try std.testing.expect(!memWriteU32(&buf, 0, 0));
}

test "memWriteU64: writes little-endian value" {
    var buf: [8]u8 = undefined;
    try std.testing.expect(memWriteU64(&buf, 0, 0x0102030405060708));
    const lo = memReadU32(&buf, 0).?;
    const hi = memReadU32(&buf, 4).?;
    try std.testing.expectEqual(@as(u32, 0x05060708), lo);
    try std.testing.expectEqual(@as(u32, 0x01020304), hi);
}

test "memWriteU64: out-of-bounds returns false" {
    var buf: [4]u8 = undefined;
    try std.testing.expect(!memWriteU64(&buf, 0, 0));
}

test "fdWriteCore: invalid fd returns EBADF" {
    var mem: [64]u8 = undefined;
    try std.testing.expectEqual(WASI_EBADF, fdWriteCore(&mem, 0, 0, 0, 0));
    try std.testing.expectEqual(WASI_EBADF, fdWriteCore(&mem, 3, 0, 0, 0));
}

test "fdWriteCore: stdout with zero iovs succeeds" {
    var mem = [_]u8{0} ** 64;
    const result = fdWriteCore(&mem, 1, 0, 0, 60);
    try std.testing.expectEqual(WASI_ESUCCESS, result);
    try std.testing.expectEqual(@as(u32, 0), memReadU32(&mem, 60).?);
}

test "fdWriteCore: stdout writes correct byte count" {
    var mem = [_]u8{0} ** 64;
    // Set up one iov: buf_ptr=16, buf_len=5
    std.mem.writeInt(u32, mem[0..4], 16, .little); // iov[0].buf_ptr
    std.mem.writeInt(u32, mem[4..8], 5, .little); // iov[0].buf_len
    @memcpy(mem[16..21], "hello");

    const result = fdWriteCore(&mem, 1, 0, 1, 60);
    try std.testing.expectEqual(WASI_ESUCCESS, result);
    try std.testing.expectEqual(@as(u32, 5), memReadU32(&mem, 60).?);
}

test "fdSeekCore: returns ENOSYS" {
    try std.testing.expectEqual(WASI_ENOSYS, fdSeekCore());
}

test "fdCloseCore: stdin/stdout/stderr succeed" {
    try std.testing.expectEqual(WASI_ESUCCESS, fdCloseCore(0));
    try std.testing.expectEqual(WASI_ESUCCESS, fdCloseCore(1));
    try std.testing.expectEqual(WASI_ESUCCESS, fdCloseCore(2));
}

test "fdCloseCore: other fds return EBADF" {
    try std.testing.expectEqual(WASI_EBADF, fdCloseCore(3));
    try std.testing.expectEqual(WASI_EBADF, fdCloseCore(-1));
}

test "fdFdstatGetCore: invalid fd returns EBADF" {
    var mem = [_]u8{0} ** 64;
    try std.testing.expectEqual(WASI_EBADF, fdFdstatGetCore(&mem, 3, 0));
    try std.testing.expectEqual(WASI_EBADF, fdFdstatGetCore(&mem, -1, 0));
}

test "fdFdstatGetCore: out-of-bounds returns EINVAL" {
    var mem = [_]u8{0} ** 16;
    try std.testing.expectEqual(WASI_EINVAL, fdFdstatGetCore(&mem, 1, 0));
}

test "fdFdstatGetCore: stdout fills struct correctly" {
    var mem = [_]u8{0} ** 64;
    const result = fdFdstatGetCore(&mem, 1, 0);
    try std.testing.expectEqual(WASI_ESUCCESS, result);
    // fs_filetype = 2 (character device)
    try std.testing.expectEqual(@as(u8, 2), mem[0]);
    // fs_rights_base at offset 8
    const rights = std.mem.readInt(u64, mem[8..16], .little);
    try std.testing.expectEqual(@as(u64, 0x1FFFFFFF), rights);
}

test "fdPrestatGetCore: returns EBADF" {
    try std.testing.expectEqual(WASI_EBADF, fdPrestatGetCore());
}

test "fdPrestatDirNameCore: returns EBADF" {
    try std.testing.expectEqual(WASI_EBADF, fdPrestatDirNameCore());
}

test "clockTimeGetCore: realtime succeeds" {
    var mem = [_]u8{0} ** 16;
    const result = clockTimeGetCore(&mem, WASI_CLOCK_REALTIME, 0);
    try std.testing.expectEqual(WASI_ESUCCESS, result);
    const nanos = std.mem.readInt(u64, mem[0..8], .little);
    try std.testing.expect(nanos > 0);
}

test "clockTimeGetCore: monotonic succeeds" {
    var mem = [_]u8{0} ** 16;
    const result = clockTimeGetCore(&mem, WASI_CLOCK_MONOTONIC, 0);
    try std.testing.expectEqual(WASI_ESUCCESS, result);
}

test "clockTimeGetCore: invalid clock returns EINVAL" {
    var mem = [_]u8{0} ** 16;
    try std.testing.expectEqual(WASI_EINVAL, clockTimeGetCore(&mem, 99, 0));
}

test "environSizesGetCore: writes zeroes" {
    var mem = [_]u8{0xFF} ** 16;
    const result = environSizesGetCore(&mem, 0, 4);
    try std.testing.expectEqual(WASI_ESUCCESS, result);
    try std.testing.expectEqual(@as(u32, 0), memReadU32(&mem, 0).?);
    try std.testing.expectEqual(@as(u32, 0), memReadU32(&mem, 4).?);
}

test "environGetCore: returns ESUCCESS" {
    try std.testing.expectEqual(WASI_ESUCCESS, environGetCore());
}

test "argsSizesGetCore: writes zeroes" {
    var mem = [_]u8{0xFF} ** 16;
    const result = argsSizesGetCore(&mem, 0, 4);
    try std.testing.expectEqual(WASI_ESUCCESS, result);
    try std.testing.expectEqual(@as(u32, 0), memReadU32(&mem, 0).?);
    try std.testing.expectEqual(@as(u32, 0), memReadU32(&mem, 4).?);
}

test "argsGetCore: returns ESUCCESS" {
    try std.testing.expectEqual(WASI_ESUCCESS, argsGetCore());
}
