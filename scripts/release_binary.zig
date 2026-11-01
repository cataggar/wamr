//! Platform identities and executable-header checks for release archives.
const std = @import("std");

pub const System = enum { linux, macos, windows };
pub const Arch = enum { x86_64, aarch64, riscv64 };
pub const Target = struct {
    name: []const u8,
    system: System,
    arch: Arch,
    required_native: bool = false,
};

pub const targets = [_]Target{
    .{ .name = "linux-x64", .system = .linux, .arch = .x86_64, .required_native = true },
    .{ .name = "linux-arm64", .system = .linux, .arch = .aarch64, .required_native = true },
    .{ .name = "macos-arm64", .system = .macos, .arch = .aarch64, .required_native = true },
    .{ .name = "windows-x64", .system = .windows, .arch = .x86_64, .required_native = true },
    .{ .name = "macos-x64", .system = .macos, .arch = .x86_64 },
    .{ .name = "linux-musl-x64", .system = .linux, .arch = .x86_64 },
    .{ .name = "linux-musl-arm64", .system = .linux, .arch = .aarch64 },
    .{ .name = "linux-riscv64", .system = .linux, .arch = .riscv64 },
    .{ .name = "windows-arm64", .system = .windows, .arch = .aarch64 },
};

pub fn target(name: []const u8) ?Target {
    for (targets) |value| if (std.mem.eql(u8, value.name, name)) return value;
    return null;
}

/// Check the same ELF64, Mach-O64, and PE32+ identity fields as the release verifier.
pub fn checkBinary(data: []const u8, system: System, arch: Arch) !void {
    const valid = switch (system) {
        .linux => blk: {
            if (data.len < 64 or !std.mem.eql(u8, data[0..6], "\x7fELF\x02\x01")) break :blk false;
            const kind = std.mem.readInt(u16, data[16..18], .little);
            const machine = std.mem.readInt(u16, data[18..20], .little);
            const expected: u16 = switch (arch) {
                .x86_64 => 62,
                .aarch64 => 183,
                .riscv64 => 243,
            };
            break :blk (kind == 2 or kind == 3) and machine == expected;
        },
        .macos => blk: {
            if (data.len < 32 or !std.mem.eql(u8, data[0..4], "\xcf\xfa\xed\xfe")) break :blk false;
            const expected: u32 = switch (arch) {
                .x86_64 => 0x1000007,
                .aarch64 => 0x100000c,
                .riscv64 => break :blk false,
            };
            break :blk std.mem.readInt(u32, data[4..8], .little) == expected and
                std.mem.readInt(u32, data[12..16], .little) == 2;
        },
        .windows => blk: {
            if (data.len < 64 or !std.mem.eql(u8, data[0..2], "MZ")) break :blk false;
            const offset: usize = std.mem.readInt(u32, data[60..64], .little);
            if (offset < 64 or offset > data.len or data.len - offset < 26) break :blk false;
            const expected: u16 = switch (arch) {
                .x86_64 => 0x8664,
                .aarch64 => 0xaa64,
                .riscv64 => break :blk false,
            };
            break :blk std.mem.eql(u8, data[offset..][0..4], "PE\x00\x00") and
                std.mem.readInt(u16, data[offset + 4 ..][0..2], .little) == expected and
                std.mem.readInt(u16, data[offset + 24 ..][0..2], .little) == 0x20b;
        },
    };
    if (!valid) return error.InvalidExecutableIdentity;
}

test "ELF identity checks class, byte order, executable type, and machine" {
    var data: [64]u8 = @splat(0);
    @memcpy(data[0..6], "\x7fELF\x02\x01");
    std.mem.writeInt(u16, data[16..18], 3, .little);
    std.mem.writeInt(u16, data[18..20], 183, .little);
    try checkBinary(&data, .linux, .aarch64);
    try std.testing.expectError(error.InvalidExecutableIdentity, checkBinary(&data, .linux, .x86_64));
    std.mem.writeInt(u16, data[16..18], 1, .little);
    try std.testing.expectError(error.InvalidExecutableIdentity, checkBinary(&data, .linux, .aarch64));
    try std.testing.expectError(error.InvalidExecutableIdentity, checkBinary(data[0..63], .linux, .aarch64));
}

test "Mach-O and PE identities reject incorrect machines and truncated offsets" {
    var macho: [32]u8 = @splat(0);
    @memcpy(macho[0..4], "\xcf\xfa\xed\xfe");
    std.mem.writeInt(u32, macho[4..8], 0x1000007, .little);
    std.mem.writeInt(u32, macho[12..16], 2, .little);
    try checkBinary(&macho, .macos, .x86_64);
    try std.testing.expectError(error.InvalidExecutableIdentity, checkBinary(&macho, .macos, .aarch64));

    var pe: [90]u8 = @splat(0);
    @memcpy(pe[0..2], "MZ");
    std.mem.writeInt(u32, pe[60..64], 64, .little);
    @memcpy(pe[64..68], "PE\x00\x00");
    std.mem.writeInt(u16, pe[68..70], 0xaa64, .little);
    std.mem.writeInt(u16, pe[88..90], 0x20b, .little);
    try checkBinary(&pe, .windows, .aarch64);
    try std.testing.expectError(error.InvalidExecutableIdentity, checkBinary(&pe, .windows, .x86_64));
    std.mem.writeInt(u32, pe[60..64], std.math.maxInt(u32), .little);
    try std.testing.expectError(error.InvalidExecutableIdentity, checkBinary(&pe, .windows, .aarch64));
}
