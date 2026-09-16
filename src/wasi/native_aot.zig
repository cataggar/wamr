//! Typed import adapter for the compiler-free native AOT embedding API.
//! Pass the native API module to Adapter; no hosted runtime is imported here.
const minimal = @import("minimal-wasi");

pub const Context = minimal.Context;
pub const Options = minimal.Options;

pub fn Adapter(comptime aot: type) type {
    return struct {
        /// The native loader copies these descriptors. The supplied context and
        /// its borrowed options/callback state must outlive the native instance.
        pub fn imports(context: *Context) [minimal.imports.len]aot.HostImport {
            var bindings: [minimal.imports.len]aot.HostImport = undefined;
            inline for (minimal.imports, 0..) |entry, index| {
                const thunk = Thunk(@enumFromInt(index));
                bindings[index] = .{
                    .module = entry.namespace,
                    .name = entry.name,
                    .params = &thunk.params,
                    .results = &thunk.results,
                    .context = context,
                    .callback = thunk.call,
                };
            }
            return bindings;
        }

        fn types(comptime bytes: []const u8) [bytes.len]aot.ValType {
            var values: [bytes.len]aot.ValType = undefined;
            for (bytes, 0..) |byte, index| {
                values[index] = switch (byte) {
                    0x7f => .i32,
                    0x7e => .i64,
                    else => @compileError("unsupported minimal WASI value type"),
                };
            }
            return values;
        }

        fn Thunk(comptime function: minimal.Function) type {
            const spec = minimal.imports[@intFromEnum(function)];
            return struct {
                const params = types(spec.params);
                const results = types(spec.results);

                fn call(raw: ?*anyopaque, host: *aot.HostContext, arguments: []const aot.Value, result: []aot.Value) aot.HostError!void {
                    if (raw == null or arguments.len != params.len or result.len != results.len)
                        return error.InvalidArgument;
                    const context: *Context = @ptrCast(@alignCast(raw.?));
                    var bits: [params.len]u64 = undefined;
                    inline for (params, 0..) |kind, index| {
                        bits[index] = switch (kind) {
                            .i32 => switch (arguments[index]) {
                                .i32 => |value| @as(u32, @bitCast(value)),
                                else => return error.InvalidArgument,
                            },
                            .i64 => switch (arguments[index]) {
                                .i64 => |value| @bitCast(value),
                                else => return error.InvalidArgument,
                            },
                            else => unreachable,
                        };
                    }
                    const outcome = context.dispatch(host.memory(), function, &bits) catch return error.InvalidArgument;
                    switch (outcome) {
                        .returned => |errno| {
                            if (comptime results.len != 1) return error.InvalidArgument;
                            result[0] = .{ .i32 = @intFromEnum(errno) };
                        },
                        .exited => |code| {
                            // Native terminate records a pending exit. Return to
                            // the dispatcher so host defers run before it unwinds
                            // guest frames; never return an errno for proc_exit.
                            host.terminate(code);
                            return;
                        },
                    }
                }
            };
        }
    };
}

test {
    _ = @import("native_aot_test.zig");
}
