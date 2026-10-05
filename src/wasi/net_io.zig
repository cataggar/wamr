const std = @import("std");

pub fn read(io: std.Io, handle: std.Io.net.Socket.Handle, data: [][]u8) std.Io.net.Stream.Reader.Error!usize {
    const result = try (try io.operate(.{ .net_read = .{
        .socket_handle = handle,
        .data = data,
    } })).net_read;
    return result.data_len;
}

pub fn write(io: std.Io, handle: std.Io.net.Socket.Handle, header: []const u8, data: []const []const u8, splat: usize) std.Io.net.Stream.Writer.Error!usize {
    return (try io.operate(.{ .net_write = .{
        .socket_handle = handle,
        .header = header,
        .data = data,
        .splat = splat,
    } })).net_write;
}
