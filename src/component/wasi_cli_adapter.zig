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
    /// `HostInstance` registered for the simple test interface name
    /// (e.g. `"wasi:hello/world"`). Stored inline so the adapter owns its
    /// lifetime; callers pass a stable pointer to `linkImports`.
    write_iface: HostInstance = .{},
    /// `HostInstance` for `wasi:cli/stdout` (member: `get-stdout`).
    cli_stdout_iface: HostInstance = .{},
    /// `HostInstance` for `wasi:io/streams` (members:
    /// `[method]output-stream.blocking-write-and-flush`,
    /// `[resource-drop]output-stream`).
    io_streams_iface: HostInstance = .{},

    /// Adapter-internal output-stream resource handles. Deliberately
    /// independent of `ComponentInstance.resource_tables` — these handles
    /// only flow through host calls and never need to round-trip into
    /// guest-side `resource.{new,drop,rep}`. `0` is reserved and means
    /// "stdout"; future handle types (stderr, files) will get distinct
    /// non-zero values.
    stream_table: std.ArrayListUnmanaged(?*streams.OutputStream) = .empty,

    /// Initialize with a buffer-backed stdout sink. Use `getStdoutBytes`
    /// after the component runs to inspect captured output.
    pub fn init(allocator: Allocator) WasiCliAdapter {
        return .{
            .allocator = allocator,
            .stdout = streams.OutputStream.toBuffer(),
        };
    }

    /// Initialize with stdout backed by a host file descriptor (typically
    /// `std.posix.STDOUT_FILENO`). For the production CLI path; tests
    /// continue to use `init` with the captured-buffer sink.
    pub fn initWithStdoutFd(allocator: Allocator, fd: std.posix.fd_t) WasiCliAdapter {
        return .{
            .allocator = allocator,
            .stdout = streams.OutputStream.toFd(fd),
        };
    }

    pub fn deinit(self: *WasiCliAdapter) void {
        self.stdout.deinit(self.allocator);
        self.write_iface.deinit(self.allocator);
        self.cli_stdout_iface.deinit(self.allocator);
        self.io_streams_iface.deinit(self.allocator);
        self.stream_table.deinit(self.allocator);
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

    /// Register the WASI cli/run "println" surface used by the Phase 2B-hello
    /// fixture: `wasi:cli/stdout` (with `get-stdout`) and `wasi:io/streams`
    /// (with the output-stream method + resource-drop). The full WIT names
    /// of WASI 0.2 are passed verbatim in `instance_*_name` so callers can
    /// pin a specific interface version (`wasi:cli/stdout@0.2.6` etc.).
    pub fn populateWasiCliRun(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        cli_stdout_name: []const u8,
        io_streams_name: []const u8,
    ) !void {
        try self.cli_stdout_iface.members.put(self.allocator, "get-stdout", .{
            .func = .{ .context = self, .call = &getStdoutHandle },
        });
        try providers.put(self.allocator, cli_stdout_name, .{
            .host_instance = &self.cli_stdout_iface,
        });

        try self.io_streams_iface.members.put(
            self.allocator,
            "[method]output-stream.blocking-write-and-flush",
            .{ .func = .{ .context = self, .call = &blockingWriteAndFlush } },
        );
        try self.io_streams_iface.members.put(
            self.allocator,
            "[resource-drop]output-stream",
            .{ .func = .{ .context = self, .call = &dropOutputStream } },
        );
        try providers.put(self.allocator, io_streams_name, .{
            .host_instance = &self.io_streams_iface,
        });
    }

    fn allocStreamHandle(self: *WasiCliAdapter, stream: *streams.OutputStream) !u32 {
        // Linear scan for a free slot before extending; output streams are
        // few in number for cli/run.
        for (self.stream_table.items, 0..) |slot, i| {
            if (slot == null) {
                self.stream_table.items[i] = stream;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.stream_table.items.len);
        try self.stream_table.append(self.allocator, stream);
        return idx;
    }

    fn lookupStream(self: *WasiCliAdapter, handle: u32) ?*streams.OutputStream {
        if (handle >= self.stream_table.items.len) return null;
        return self.stream_table.items[handle];
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
        _ = results;
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

    /// `wasi:cli/stdout.get-stdout: () -> own<output-stream>`. Returns a
    /// fresh handle into the adapter's stream table, pointing at the
    /// captured stdout buffer.
    fn getStdoutHandle(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const handle = try self.allocStreamHandle(&self.stdout);
        results[0] = .{ .handle = handle };
    }

    /// `wasi:io/streams.[method]output-stream.blocking-write-and-flush:
    ///   (own<output-stream>, list<u8>) -> result<_, stream-error>`.
    /// Looks up the handle, writes the guest bytes through, and returns
    /// the canonical "ok" discriminant. (The error arm is unreachable in
    /// captured-buffer mode and is omitted from the synthetic fixture's
    /// `result<_, _>` type — this keeps the flat lower path on the
    /// no-payload fast path.)
    fn blockingWriteAndFlush(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 2 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const list = switch (args[1]) {
            .list => |pl| pl,
            else => return error.InvalidArgs,
        };
        const stream = self.lookupStream(handle) orelse return error.InvalidHandle;
        const bytes = ci.readGuestBytes(list.ptr, list.len) orelse
            return error.OutOfBoundsMemory;
        switch (stream.write(bytes, self.allocator)) {
            .ok => {},
            .err, .closed => return error.IoError,
        }
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `wasi:io/streams.[resource-drop]output-stream: (own<output-stream>) -> ()`.
    /// Marks the handle's slot inactive. Stdout itself is not closed —
    /// the same buffer can be reopened with another `get-stdout`.
    fn dropOutputStream(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        _: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        if (handle < self.stream_table.items.len) {
            self.stream_table.items[handle] = null;
        }
    }
};

// ── Top-level CLI dispatch ─────────────────────────────────────────────────

const ctypes_root = @import("types.zig");
const component_loader = @import("loader.zig");
const executor_root = @import("executor.zig");
const abi_root = @import("canonical_abi.zig");

pub const RunComponentError = error{
    LoadFailed,
    InstantiateFailed,
    LinkFailed,
    NoRunExport,
    Trap,
    OutOfMemory,
};

pub const RunOutcome = struct {
    /// The component exited normally. `is_ok=true` means run returned ok;
    /// false means the component returned `result::error(_)`.
    is_ok: bool,
};

/// Whether a component import name matches a WASI interface prefix,
/// allowing a trailing `@<version>` (e.g. `@0.2.6`).
fn matchesWasiPrefix(import_name: []const u8, prefix: []const u8) bool {
    if (!std.mem.startsWith(u8, import_name, prefix)) return false;
    const rest = import_name[prefix.len..];
    return rest.len == 0 or rest[0] == '@';
}

/// Bind WASI cli/run-style imports for `component` against `adapter`.
///
/// Walks every top-level instance import and, for each name that
/// matches a known WASI interface (with or without a `@<version>`
/// suffix), binds the corresponding `HostInstance` from the adapter.
/// Imports that don't match are left for the caller to wire (and will
/// cause `linkImports` to fail with `MissingImport` — by design).
pub fn populateWasiProviders(
    adapter: *WasiCliAdapter,
    component: *const ctypes_root.Component,
    providers: *std.StringHashMapUnmanaged(ImportBinding),
) !void {
    var matched_stdout: ?[]const u8 = null;
    var matched_streams: ?[]const u8 = null;
    for (component.imports) |imp| {
        if (imp.desc != .instance) continue;
        if (matched_stdout == null and matchesWasiPrefix(imp.name, "wasi:cli/stdout"))
            matched_stdout = imp.name;
        if (matched_streams == null and matchesWasiPrefix(imp.name, "wasi:io/streams"))
            matched_streams = imp.name;
    }
    // Always populate both interfaces' members; only register
    // providers for the names the component actually imports so
    // `linkImports` strict-checking surfaces unrelated misses.
    try adapter.populateWasiCliRun(
        providers,
        matched_stdout orelse "wasi:cli/stdout",
        matched_streams orelse "wasi:io/streams",
    );
    if (matched_stdout == null) _ = providers.remove("wasi:cli/stdout");
    if (matched_streams == null) _ = providers.remove("wasi:io/streams");
}

/// Run an already-loaded component. See `runComponentBytes` for the
/// byte-level entry point and the policy notes that apply equally here.
pub fn runLoadedComponent(
    component: *const ctypes_root.Component,
    allocator: Allocator,
    adapter: *WasiCliAdapter,
) RunComponentError!RunOutcome {
    const inst = instance_mod.instantiate(component, allocator) catch return error.InstantiateFailed;
    defer inst.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(allocator);

    populateWasiProviders(adapter, component, &providers) catch return error.OutOfMemory;
    inst.linkImports(providers) catch return error.LinkFailed;

    if (inst.getExport("run") == null) return error.NoRunExport;

    var results: [1]abi_root.InterfaceValue = undefined;
    executor_root.callComponentFunc(inst, "run", &.{}, &results, allocator) catch return error.Trap;

    return .{ .is_ok = switch (results[0]) {
        .result_val => |rv| rv.is_ok,
        else => true,
    } };
}

/// Load + instantiate + run a component binary, mapping its `run` export
/// to a normalized outcome the CLI can turn into an exit code.
///
/// Phase 3 narrow scope:
///   - Recognizes the run entrypoint as a top-level component export
///     named exactly `"run"` (matching the hand-authored 2B-hello fixture).
///     Real WASI components nest `run` inside an exported
///     `wasi:cli/run` *instance*; lifting that into our `exported_funcs`
///     map requires the indexspace work tracked under `142-1b-indexspaces`
///     and is deferred. Until then, real Rust components are rejected
///     with `error.NoRunExport` rather than silently mis-exiting.
///   - Binds WASI imports against `adapter`'s `wasi:cli/stdout` +
///     `wasi:io/streams` interfaces; any other instance import will
///     surface as `error.LinkFailed` from `linkImports`.
///   - The `run` export must return `result<_,_>`. Trapping inside `run`
///     surfaces as `error.Trap` from this function.
pub fn runComponentBytes(
    data: []const u8,
    allocator: Allocator,
    adapter: *WasiCliAdapter,
) RunComponentError!RunOutcome {
    const component_storage = allocator.create(ctypes_root.Component) catch return error.OutOfMemory;
    defer allocator.destroy(component_storage);
    component_storage.* = component_loader.load(data, allocator) catch return error.LoadFailed;
    return runLoadedComponent(component_storage, allocator, adapter);
}

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

test "WasiCliAdapter: hello-world fixture (cli/stdout + io/streams + run)" {
    const ctypes = @import("types.zig");
    const executor = @import("executor.zig");
    const abi_mod = @import("canonical_abi.zig");
    const testing = std.testing;

    // Hand-authored core module:
    //   (type 0 (func (result i32)))                    ;; () -> i32
    //   (type 1 (func (param i32 i32 i32) (result i32))) ;; write_flush
    //   (type 2 (func (param i32)))                      ;; drop_stream
    //   (import "host" "get_stdout"  (func (type 0)))   ;; func 0
    //   (import "host" "write_flush" (func (type 1)))   ;; func 1
    //   (import "host" "drop_stream" (func (type 2)))   ;; func 2
    //   (memory 1)
    //   (func (type 0)                                   ;; func 3 = "run"
    //     (local i32)
    //     call $get_stdout         ;; handle on stack
    //     local.tee 0              ;; save handle, leave on stack
    //     i32.const 16 i32.const 14
    //     call $write_flush        ;; result discriminant on stack
    //     drop                     ;; ignore (test: should be 0)
    //     local.get 0
    //     call $drop_stream
    //     i32.const 0)             ;; return ok
    //   (export "run" (func 3))
    //   (data (i32.const 16) "hello, world!\n")
    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type section: 3 types, content = 1 + 4 + 7 + 4 = 16 bytes
        0x01, 0x10, 0x03,
        0x60, 0x00, 0x01, 0x7f, // type 0
        0x60, 0x03, 0x7f, 0x7f, 0x7f, 0x01, 0x7f, // type 1
        0x60, 0x01, 0x7f, 0x00, // type 2
        // import section: 3 imports, content = 1 + 18 + 19 + 19 = 57 bytes
        0x02, 0x39, 0x03,
        0x04, 'h', 'o', 's', 't', 0x0a, 'g', 'e', 't', '_', 's', 't', 'd', 'o', 'u', 't', 0x00, 0x00,
        0x04, 'h', 'o', 's', 't', 0x0b, 'w', 'r', 'i', 't', 'e', '_', 'f', 'l', 'u', 's', 'h', 0x00, 0x01,
        0x04, 'h', 'o', 's', 't', 0x0b, 'd', 'r', 'o', 'p', '_', 's', 't', 'r', 'e', 'a', 'm', 0x00, 0x02,
        // function section: 1 local fn of type 0
        0x03, 0x02, 0x01, 0x00,
        // memory section: 1 mem, min=1
        0x05, 0x03, 0x01, 0x00, 0x01,
        // export section: "run" -> func 3
        0x07, 0x07, 0x01,
        0x03, 'r', 'u', 'n',
        0x00, 0x03,
        // code section: 1 body, body_size=21, count=1 + 22 body = 23
        0x0a, 0x17, 0x01,
        0x15, // body size
        0x01, 0x01, 0x7f, // 1 local of i32
        0x10, 0x00, // call 0 (get_stdout)
        0x22, 0x00, // local.tee 0
        0x41, 0x10, // i32.const 16
        0x41, 0x0e, // i32.const 14
        0x10, 0x01, // call 1 (write_flush)
        0x1a, // drop
        0x20, 0x00, // local.get 0
        0x10, 0x02, // call 2 (drop_stream)
        0x41, 0x00, // i32.const 0
        0x0b, // end
        // data section: 1 segment, mem 0, offset i32.const 16, "hello, world!\n"
        0x0b, 0x14, 0x01,
        0x00,
        0x41, 0x10, 0x0b,
        0x0e,
        'h', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\n',
    };

    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};

    // Component types:
    //   type 0 = list element u8 (used as element of list<u8>)  -- not a TypeDef; lists carry their element inline
    //   type 0 = func () -> own<output-stream(=resource type idx 99 sentinel)>
    // For simplicity use `.handle = ...` via own/borrow with a stable resource
    // type idx — we're not exercising guest-side resource tables, just the
    // i32 round-trip.
    const RES_TYPE_IDX: u32 = 100; // sentinel; never actually looked up.

    // type 0: get-stdout : () -> own<output-stream>
    const t0_results: ctypes.FuncType.ResultList = .{ .unnamed = .{ .own = RES_TYPE_IDX } };
    // type 1: write-and-flush : (own<output-stream>, list<u8>) -> result<_,_>
    const t1_params = [_]ctypes.NamedValType{
        .{ .name = "this", .type = .{ .own = RES_TYPE_IDX } },
        .{ .name = "data", .type = .{ .list = 4 } }, // list type at type idx 4
    };
    // type 2: drop : (own<output-stream>) -> ()
    const t2_params = [_]ctypes.NamedValType{
        .{ .name = "this", .type = .{ .own = RES_TYPE_IDX } },
    };
    // type 3: run : () -> result<_,_>  (idx of result type below)
    const t3_results: ctypes.FuncType.ResultList = .{ .unnamed = .{ .result = 5 } };

    const inst_stdout_decls = [_]ctypes.Decl{
        .{ .@"export" = .{ .name = "get-stdout", .desc = .{ .func = 0 } } },
    };
    const inst_streams_decls = [_]ctypes.Decl{
        .{ .@"export" = .{ .name = "[method]output-stream.blocking-write-and-flush", .desc = .{ .func = 1 } } },
        .{ .@"export" = .{ .name = "[resource-drop]output-stream", .desc = .{ .func = 2 } } },
    };

    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &.{}, .results = t0_results } }, // 0
        .{ .func = .{ .params = &t1_params, .results = .{ .unnamed = .{ .result = 5 } } } }, // 1
        .{ .func = .{ .params = &t2_params, .results = .none } }, // 2
        .{ .func = .{ .params = &.{}, .results = t3_results } }, // 3
        .{ .list = .{ .element = .u8 } }, // 4
        .{ .result = .{ .ok = null, .err = null } }, // 5
        .{ .instance = .{ .decls = &inst_stdout_decls } }, // 6
        .{ .instance = .{ .decls = &inst_streams_decls } }, // 7
    };

    const imports_decl = [_]ctypes.ImportDecl{
        .{ .name = "wasi:cli/stdout", .desc = .{ .instance = 6 } },
        .{ .name = "wasi:io/streams", .desc = .{ .instance = 7 } },
    };

    // Aliases: 3 component-func + 1 core-func (for the lifted run export).
    const aliases_decl = [_]ctypes.Alias{
        .{ .instance_export = .{ .sort = .func, .instance_idx = 0, .name = "get-stdout" } }, // → comp func 0
        .{ .instance_export = .{ .sort = .func, .instance_idx = 1, .name = "[method]output-stream.blocking-write-and-flush" } }, // → comp func 1
        .{ .instance_export = .{ .sort = .func, .instance_idx = 1, .name = "[resource-drop]output-stream" } }, // → comp func 2
        .{ .instance_export = .{ .sort = .{ .core = .func }, .instance_idx = 1, .name = "run" } }, // core-func alias
    };

    // Canons: 3 lowers (core funcs 0..2), then 1 lift (comp func 3).
    // The lift's core_func_idx points into the component-level core-func
    // index space: 0..2 = lowers, 3 = the "run" core alias above.
    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &.{} } },
        .{ .lower = .{ .func_idx = 1, .opts = &.{} } },
        .{ .lower = .{ .func_idx = 2, .opts = &.{} } },
        .{ .lift = .{ .core_func_idx = 3, .type_idx = 3, .opts = &.{} } },
    };

    // Inline core instance exposes the 3 lowered funcs as host.{...} imports
    // for the actual core module instantiation.
    const inline_exports = [_]ctypes.CoreInlineExport{
        .{ .name = "get_stdout", .sort_idx = .{ .sort = .func, .idx = 0 } },
        .{ .name = "write_flush", .sort_idx = .{ .sort = .func, .idx = 1 } },
        .{ .name = "drop_stream", .sort_idx = .{ .sort = .func, .idx = 2 } },
    };
    const inst_args = [_]ctypes.CoreInstantiateArg{
        .{ .name = "host", .instance_idx = 0 },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .exports = &inline_exports },
        .{ .instantiate = .{ .module_idx = 0, .args = &inst_args } },
    };

    const exports_decl = [_]ctypes.ExportDecl{
        .{ .name = "run", .desc = .{ .func = 3 } },
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
        .exports = &exports_decl,
    };

    const inst = try instance_mod.instantiate(&component, testing.allocator);
    defer inst.deinit();

    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(testing.allocator);
    try adapter.populateWasiCliRun(&providers, "wasi:cli/stdout", "wasi:io/streams");

    try inst.linkImports(providers);

    // All three trampolines should now have their host_func bound.
    try testing.expect(inst.trampoline_ctxs.items.len == 3);
    for (inst.trampoline_ctxs.items) |ctx| {
        try testing.expect(ctx.host_func.call != null);
    }

    // Invoke the lifted "run" component export. It returns
    // result<_,_> which lifts to a `.result_val { is_ok = true }`.
    var results: [1]abi_mod.InterfaceValue = undefined;
    try executor.callComponentFunc(inst, "run", &.{}, &results, testing.allocator);
    try testing.expect(results[0] == .result_val);
    try testing.expect(results[0].result_val.is_ok);

    try testing.expectEqualStrings("hello, world!\n", adapter.getStdoutBytes());

    // Stream handle should now be dropped (slot is null).
    try testing.expect(adapter.stream_table.items.len >= 1);
    try testing.expect(adapter.stream_table.items[0] == null);
}

test "matchesWasiPrefix: exact and version-suffixed names" {
    const testing = std.testing;
    try testing.expect(matchesWasiPrefix("wasi:cli/stdout", "wasi:cli/stdout"));
    try testing.expect(matchesWasiPrefix("wasi:cli/stdout@0.2.6", "wasi:cli/stdout"));
    try testing.expect(matchesWasiPrefix("wasi:io/streams@0.2", "wasi:io/streams"));
    try testing.expect(!matchesWasiPrefix("wasi:cli/stdout-extra", "wasi:cli/stdout"));
    try testing.expect(!matchesWasiPrefix("other:cli/stdout", "wasi:cli/stdout"));
    try testing.expect(!matchesWasiPrefix("wasi:cli/std", "wasi:cli/stdout"));
}

test "runLoadedComponent: matches versioned WASI import names" {
    // Same hand-authored fixture as the "hello-world fixture" test, but
    // with versioned import names (`@0.2.6`) and dispatched through the
    // CLI's `runLoadedComponent` helper. This exercises the full
    // populateWasiProviders → linkImports → callComponentFunc path.
    const ctypes = @import("types.zig");
    const testing = std.testing;

    const core_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x10, 0x03,
        0x60, 0x00, 0x01, 0x7f,
        0x60, 0x03, 0x7f, 0x7f, 0x7f, 0x01, 0x7f,
        0x60, 0x01, 0x7f, 0x00,
        0x02, 0x39, 0x03,
        0x04, 'h', 'o', 's', 't', 0x0a, 'g', 'e', 't', '_', 's', 't', 'd', 'o', 'u', 't', 0x00, 0x00,
        0x04, 'h', 'o', 's', 't', 0x0b, 'w', 'r', 'i', 't', 'e', '_', 'f', 'l', 'u', 's', 'h', 0x00, 0x01,
        0x04, 'h', 'o', 's', 't', 0x0b, 'd', 'r', 'o', 'p', '_', 's', 't', 'r', 'e', 'a', 'm', 0x00, 0x02,
        0x03, 0x02, 0x01, 0x00,
        0x05, 0x03, 0x01, 0x00, 0x01,
        0x07, 0x07, 0x01,
        0x03, 'r', 'u', 'n',
        0x00, 0x03,
        0x0a, 0x17, 0x01,
        0x15,
        0x01, 0x01, 0x7f,
        0x10, 0x00,
        0x22, 0x00,
        0x41, 0x10,
        0x41, 0x0e,
        0x10, 0x01,
        0x1a,
        0x20, 0x00,
        0x10, 0x02,
        0x41, 0x00,
        0x0b,
        0x0b, 0x14, 0x01,
        0x00,
        0x41, 0x10, 0x0b,
        0x0e,
        'h', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\n',
    };
    const core_modules = [_]ctypes.CoreModule{.{ .data = &core_wasm }};

    const RES_TYPE_IDX: u32 = 100;
    const t1_params = [_]ctypes.NamedValType{
        .{ .name = "this", .type = .{ .own = RES_TYPE_IDX } },
        .{ .name = "data", .type = .{ .list = 4 } },
    };
    const t2_params = [_]ctypes.NamedValType{
        .{ .name = "this", .type = .{ .own = RES_TYPE_IDX } },
    };

    const inst_stdout_decls = [_]ctypes.Decl{
        .{ .@"export" = .{ .name = "get-stdout", .desc = .{ .func = 0 } } },
    };
    const inst_streams_decls = [_]ctypes.Decl{
        .{ .@"export" = .{ .name = "[method]output-stream.blocking-write-and-flush", .desc = .{ .func = 1 } } },
        .{ .@"export" = .{ .name = "[resource-drop]output-stream", .desc = .{ .func = 2 } } },
    };

    const type_defs = [_]ctypes.TypeDef{
        .{ .func = .{ .params = &.{}, .results = .{ .unnamed = .{ .own = RES_TYPE_IDX } } } },
        .{ .func = .{ .params = &t1_params, .results = .{ .unnamed = .{ .result = 5 } } } },
        .{ .func = .{ .params = &t2_params, .results = .none } },
        .{ .func = .{ .params = &.{}, .results = .{ .unnamed = .{ .result = 5 } } } },
        .{ .list = .{ .element = .u8 } },
        .{ .result = .{ .ok = null, .err = null } },
        .{ .instance = .{ .decls = &inst_stdout_decls } },
        .{ .instance = .{ .decls = &inst_streams_decls } },
    };

    // Versioned import names — the CLI must still match these.
    const imports_decl = [_]ctypes.ImportDecl{
        .{ .name = "wasi:cli/stdout@0.2.6", .desc = .{ .instance = 6 } },
        .{ .name = "wasi:io/streams@0.2.6", .desc = .{ .instance = 7 } },
    };

    const aliases_decl = [_]ctypes.Alias{
        .{ .instance_export = .{ .sort = .func, .instance_idx = 0, .name = "get-stdout" } },
        .{ .instance_export = .{ .sort = .func, .instance_idx = 1, .name = "[method]output-stream.blocking-write-and-flush" } },
        .{ .instance_export = .{ .sort = .func, .instance_idx = 1, .name = "[resource-drop]output-stream" } },
        .{ .instance_export = .{ .sort = .{ .core = .func }, .instance_idx = 1, .name = "run" } },
    };

    const canons = [_]ctypes.Canon{
        .{ .lower = .{ .func_idx = 0, .opts = &.{} } },
        .{ .lower = .{ .func_idx = 1, .opts = &.{} } },
        .{ .lower = .{ .func_idx = 2, .opts = &.{} } },
        .{ .lift = .{ .core_func_idx = 3, .type_idx = 3, .opts = &.{} } },
    };

    const inline_exports = [_]ctypes.CoreInlineExport{
        .{ .name = "get_stdout", .sort_idx = .{ .sort = .func, .idx = 0 } },
        .{ .name = "write_flush", .sort_idx = .{ .sort = .func, .idx = 1 } },
        .{ .name = "drop_stream", .sort_idx = .{ .sort = .func, .idx = 2 } },
    };
    const inst_args = [_]ctypes.CoreInstantiateArg{
        .{ .name = "host", .instance_idx = 0 },
    };
    const core_insts = [_]ctypes.CoreInstanceExpr{
        .{ .exports = &inline_exports },
        .{ .instantiate = .{ .module_idx = 0, .args = &inst_args } },
    };

    const exports_decl = [_]ctypes.ExportDecl{
        .{ .name = "run", .desc = .{ .func = 3 } },
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
        .exports = &exports_decl,
    };

    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const outcome = try runLoadedComponent(&component, testing.allocator, &adapter);
    try testing.expect(outcome.is_ok);
    try testing.expectEqualStrings("hello, world!\n", adapter.getStdoutBytes());
}
