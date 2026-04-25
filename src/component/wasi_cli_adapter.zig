//! Minimal host adapter for the `wasi:cli/run`-style component interfaces.
//!
//! Phase 2B-adapter scope: a captured-buffer "stdout" sink reachable from a
//! component's canon-lowered host imports. This is the substrate the
//! end-to-end `println!` flow will plug into; it deliberately omits the
//! pieces that need ABI work outside of this slice (real `wasi:io/streams`
//! resource handles, `result<_, error>` return values, `wasi:cli/exit`).
//!
//! What this slice delivers:
//!   - A `WasiCliAdapter` that owns a writable buffer.
//!   - A reusable `writeBytes` host callback that materializes a guest
//!     `list<u8>` argument via `ComponentInstance.readGuestBytes` and
//!     appends it to the adapter's stdout buffer.
//!   - A `populateProviders` helper that registers a `HostInstance` with
//!     a single `write` member, suitable for hand-authored test fixtures
//!     that import an instance with a `(list<u8>) -> ()` member.
//!
//! Out of scope (deferred to subsequent slices):
//!   - Resource-handle plumbing for `output-stream` (tracked by issue #142
//!     phase "2B-hello"). Real WASI invocations route bytes through a
//!     handle returned by `wasi:cli/stdout.get-stdout`; for now we expose
//!     a single bytewise `write(list<u8>) -> ()` member.
//!   - Flat lowering of compound results (`result<_, _>`); current
//!     trampoline cannot push a `.result_val`. Hand-authored fixtures
//!     model the host call as `() -> ()` for now.
//!   - `wasi:cli/exit` with `WasiExit(code)` trap variant.

const std = @import("std");
const Allocator = std.mem.Allocator;

const instance_mod = @import("instance.zig");
const ComponentInstance = instance_mod.ComponentInstance;
const HostFunc = instance_mod.HostFunc;
const HostInstance = instance_mod.HostInstance;
const HostInstanceMember = instance_mod.HostInstanceMember;
const ImportBinding = instance_mod.ImportBinding;
const InterfaceValue = instance_mod.InterfaceValue;

const streams = @import("../wasi/preview2/streams.zig");

/// Captured stdout adapter. Owns its `OutputStream` and the `HostInstance`
/// objects exposed to the runtime via `populateProviders`.
///
/// Lifetime: caller `init`s, registers via `populateProviders`, then
/// retains the adapter until any `ComponentInstance` whose imports point
/// at the registered `HostInstance` is destroyed. `deinit` releases the
/// buffer and member maps.
pub const WasiCliAdapter = struct {
    allocator: Allocator,
    stdout: streams.OutputStream,
    /// `HostInstance` registered for the test interface name (default
    /// `"wasi:hello/world"`). Stored inline so the adapter owns its
    /// lifetime; callers pass a stable pointer to `linkImports`.
    write_iface: HostInstance = .{},

    /// Initialize with a buffer-backed stdout sink. Use `getStdoutBytes`
    /// after the component runs to inspect captured output.
    pub fn init(allocator: Allocator) WasiCliAdapter {
        return .{
            .allocator = allocator,
            .stdout = streams.OutputStream.toBuffer(),
        };
    }

    pub fn deinit(self: *WasiCliAdapter) void {
        self.stdout.deinit(self.allocator);
        self.write_iface.deinit(self.allocator);
    }

    /// Captured stdout bytes (valid for buffer-backed sinks).
    pub fn getStdoutBytes(self: *const WasiCliAdapter) []const u8 {
        return self.stdout.getBufferContents();
    }

    /// Register this adapter's `write_iface` as a provider for the named
    /// instance import (e.g. `"wasi:hello/world"`). The interface exposes
    /// a single member `member_name` (e.g. `"write"`) implementing
    /// `(list<u8>) -> ()`: each call appends the bytes to the captured
    /// stdout buffer.
    ///
    /// The caller-owned `providers` map is updated with a `host_instance`
    /// binding whose lifetime is tied to `self` — keep the adapter alive
    /// for as long as any component instance depends on it.
    pub fn populateProviders(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        instance_name: []const u8,
        member_name: []const u8,
    ) !void {
        try self.write_iface.members.put(self.allocator, member_name, .{
            .func = .{
                .context = self,
                .call = &writeBytes,
            },
        });
        try providers.put(self.allocator, instance_name, .{
            .host_instance = &self.write_iface,
        });
    }

    /// HostFunc callback for `(list<u8>) -> ()`. Pulls the (ptr, len)
    /// list arg out of guest memory via `ComponentInstance.readGuestBytes`
    /// and appends to `self.stdout`.
    fn writeBytes(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        _ = results; // `() -> ()` — no flat results to lower.
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len == 0) return error.InvalidArgs;
        const list = switch (args[0]) {
            .list => |pl| pl,
            else => return error.InvalidArgs,
        };
        const bytes = ci.readGuestBytes(list.ptr, list.len) orelse
            return error.OutOfBoundsMemory;
        switch (self.stdout.write(bytes, self.allocator)) {
            .ok => {},
            .err, .closed => return error.IoError,
        }
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "WasiCliAdapter: init/deinit captures empty buffer" {
    var adapter = WasiCliAdapter.init(std.testing.allocator);
    defer adapter.deinit();
    try std.testing.expectEqualStrings("", adapter.getStdoutBytes());
}

test "WasiCliAdapter: end-to-end via instance import + alias + canon.lower" {
    const ctypes = @import("types.zig");
    const testing = std.testing;

    // Hand-authored core module:
    //   (type 0 (func (param i32 i32)))
    //   (type 1 (func))
    //   (import "host" "write" (func (type 0)))
    //   (memory 1)
    //   (func (type 1)
    //     i32.const 8 i32.const 6 call 0)            ;; write(ptr=8, len=6)
    //   (export "run" (func 1))
    //   (data (i32.const 8) "hello\n")
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: 2 types (5+3 bytes content + count = 9)
        0x01, 0x09, 0x02,
        0x60, 0x02, 0x7f, 0x7f, 0x00, // (i32, i32) -> ()
        0x60, 0x00, 0x00, // () -> ()
        // import section: host.write : type 0 (14 bytes content)
        0x02, 0x0e, 0x01,
        0x04, 'h', 'o', 's', 't',
        0x05, 'w', 'r', 'i', 't', 'e',
        0x00, 0x00,
        // function section: 1 local fn of type 1
        0x03, 0x02, 0x01, 0x01,
        // memory section: 1 mem, min=1
        0x05, 0x03, 0x01, 0x00, 0x01,
        // export section: "run" -> func 1
        0x07, 0x07, 0x01,
        0x03, 'r', 'u', 'n',
        0x00, 0x01,
        // code section: 1 body, 8 bytes
        0x0a, 0x0a, 0x01,
        0x08, 0x00,
        0x41, 0x08, // i32.const 8
        0x41, 0x06, // i32.const 6
        0x10, 0x00, // call 0 (host.write)
        0x0b, // end
        // data section: 1 segment, mem 0, offset i32.const 8, "hello\n"
        0x0b, 0x0c, 0x01,
        0x00, // active mem 0
        0x41, 0x08, 0x0b, // offset = i32.const 8
        0x06, 'h', 'e', 'l', 'l', 'o', '\n',
    };

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};

    // Component types:
    //   type 0 = func (list<u8>) -> ()  for the canon-lowered host call
    //   type 1 = list<u8>  (list-element u8)
    //   type 2 = instance { export "write" (func (type 0)) }
    const list_u8 = ctypes.TypeDef{ .list = .{ .element = .u8 } };
    const params = [_]ctypes.NamedValType{
        .{ .name = "data", .type = .{ .list = 1 } },
    };
    const inst_decls = [_]ctypes.Decl{
        .{ .@"export" = .{ .name = "write", .desc = .{ .func = 0 } } },
    };
    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &params, .results = .none } },
        list_u8,
        .{ .instance = .{ .decls = &inst_decls } },
    };

    // imports: (instance (type 2)) — name "wasi:hello/world"
    const imports_decl = [_]ctypes.ImportDecl{
        .{ .name = "wasi:hello/world", .desc = .{ .instance = 2 } },
    };

    // alias instance_export sort=func instance=0 name="write" → component func 0
    const aliases_decl = [_]ctypes.Alias{
        .{ .instance_export = .{
            .sort = .func,
            .instance_idx = 0,
            .name = "write",
        } },
    };

    // canon.lower of component func 0 → core func 0 (the host import slot)
    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &.{} } },
    };

    // Wire core instance: inline exports {write: func 0 (the lowered slot)},
    // then instantiate the core module passing inline as "host".
    const inline_exports = [_]ctypes.CoreInlineExport{
        .{ .name = "write", .sort_idx = .{ .sort = .func, .idx = 0 } },
    };
    const inst_args = [_]ctypes.CoreInstantiateArg{
        .{ .name = "host", .instance_idx = 0 },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .exports = &inline_exports },
        .{ .instantiate = .{ .module_idx = 0, .args = &inst_args } },
    };

    const component = ctypes.Component{
        .core_modules = &core_modules,
        .core_instances = &core_insts,
        .core_types = &.{},
        .components = &.{},
        .instances = &.{},
        .aliases = &aliases_decl,
        .types = &type_defs,
        .canons = &canons,
        .imports = &imports_decl,
        .exports = &.{},
    };

    const inst = try instance_mod.instantiate(&component, testing.allocator);
    defer inst.deinit();

    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(testing.allocator);
    try adapter.populateProviders(&providers, "wasi:hello/world", "write");

    try inst.linkImports(providers);

    // Trampoline must have been re-bound to the adapter's host fn.
    try testing.expect(inst.trampoline_ctxs.items.len == 1);
    try testing.expect(inst.trampoline_ctxs.items[0].host_func.call != null);

    // Invoke "run", which calls write(8, 6) and pulls "hello\n" out of memory.
    const mi = inst.firstModuleInst() orelse return error.TestFailed;
    const run_idx = mi.getExportFunc("run") orelse return error.TestFailed;
    const env = try @import("../runtime/common/exec_env.zig").ExecEnv.create(mi, 512, testing.allocator);
    defer env.destroy();
    try @import("../runtime/interpreter/interp.zig").executeFunction(env, run_idx);

    try testing.expectEqualStrings("hello\n", adapter.getStdoutBytes());
}
