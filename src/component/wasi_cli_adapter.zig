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
    stderr: streams.OutputStream,
    /// Captured stdin buffer. Defaults to empty; callers populate via
    /// `setStdinBytes` before running the component. The slice is
    /// borrowed — keep it alive for as long as the component runs.
    stdin: streams.InputStream = .{ .source = .closed },

    /// Set by `wasi:cli/exit.exit` (or `exit-with-code`). The host fn
    /// also returns `error.Trap`; `runLoadedComponent` checks this
    /// field to translate the trap into a normal `RunOutcome`.
    exit_code: ?u32 = null,

    /// `HostInstance` registered for the simple test interface name
    /// (e.g. `"wasi:hello/world"`). Stored inline so the adapter owns its
    /// lifetime; callers pass a stable pointer to `linkImports`.
    write_iface: HostInstance = .{},
    cli_stdout_iface: HostInstance = .{},
    cli_stderr_iface: HostInstance = .{},
    cli_stdin_iface: HostInstance = .{},
    cli_exit_iface: HostInstance = .{},
    cli_environment_iface: HostInstance = .{},
    cli_terminal_stdin_iface: HostInstance = .{},
    cli_terminal_stdout_iface: HostInstance = .{},
    cli_terminal_stderr_iface: HostInstance = .{},
    cli_terminal_input_iface: HostInstance = .{},
    cli_terminal_output_iface: HostInstance = .{},
    io_streams_iface: HostInstance = .{},
    io_poll_iface: HostInstance = .{},
    io_error_iface: HostInstance = .{},

    stream_table: std.ArrayListUnmanaged(?*streams.OutputStream) = .empty,
    input_stream_table: std.ArrayListUnmanaged(?*streams.InputStream) = .empty,

    /// Initialize with a buffer-backed stdout sink. Use `getStdoutBytes`
    /// after the component runs to inspect captured output.
    pub fn init(allocator: Allocator) WasiCliAdapter {
        return .{
            .allocator = allocator,
            .stdout = streams.OutputStream.toBuffer(),
            .stderr = streams.OutputStream.toBuffer(),
        };
    }

    pub fn deinit(self: *WasiCliAdapter) void {
        self.stdout.deinit(self.allocator);
        self.stderr.deinit(self.allocator);
        self.write_iface.deinit(self.allocator);
        self.cli_stdout_iface.deinit(self.allocator);
        self.cli_stderr_iface.deinit(self.allocator);
        self.cli_stdin_iface.deinit(self.allocator);
        self.cli_exit_iface.deinit(self.allocator);
        self.cli_environment_iface.deinit(self.allocator);
        self.cli_terminal_stdin_iface.deinit(self.allocator);
        self.cli_terminal_stdout_iface.deinit(self.allocator);
        self.cli_terminal_stderr_iface.deinit(self.allocator);
        self.cli_terminal_input_iface.deinit(self.allocator);
        self.cli_terminal_output_iface.deinit(self.allocator);
        self.io_streams_iface.deinit(self.allocator);
        self.io_poll_iface.deinit(self.allocator);
        self.io_error_iface.deinit(self.allocator);
        self.stream_table.deinit(self.allocator);
        self.input_stream_table.deinit(self.allocator);
    }

    /// Captured stderr bytes (separate buffer from stdout).
    pub fn getStderrBytes(self: *const WasiCliAdapter) []const u8 {
        return self.stderr.getBufferContents();
    }

    /// Set the captured stdin to read from `bytes`. The slice is
    /// borrowed — caller must keep it alive for the run.
    pub fn setStdinBytes(self: *WasiCliAdapter, bytes: []const u8) void {
        self.stdin = streams.InputStream.fromBuffer(bytes);
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
            "[method]output-stream.write",
            .{ .func = .{ .context = self, .call = &outputStreamWrite } },
        );
        try self.io_streams_iface.members.put(
            self.allocator,
            "[method]output-stream.check-write",
            .{ .func = .{ .context = self, .call = &outputStreamCheckWrite } },
        );
        try self.io_streams_iface.members.put(
            self.allocator,
            "[method]output-stream.blocking-flush",
            .{ .func = .{ .context = self, .call = &outputStreamBlockingFlush } },
        );
        try self.io_streams_iface.members.put(
            self.allocator,
            "[method]output-stream.flush",
            .{ .func = .{ .context = self, .call = &outputStreamBlockingFlush } },
        );
        try self.io_streams_iface.members.put(
            self.allocator,
            "[method]output-stream.subscribe",
            .{ .func = .{ .context = self, .call = &outputStreamSubscribe } },
        );
        try self.io_streams_iface.members.put(
            self.allocator,
            "[method]input-stream.subscribe",
            .{ .func = .{ .context = self, .call = &inputStreamSubscribe } },
        );
        try self.io_streams_iface.members.put(
            self.allocator,
            "[resource-drop]output-stream",
            .{ .func = .{ .context = self, .call = &dropOutputStream } },
        );
        // Input-stream member surface used by stdio-echo via stdin reads.
        try self.io_streams_iface.members.put(
            self.allocator,
            "[method]input-stream.blocking-read",
            .{ .func = .{ .context = self, .call = &blockingRead } },
        );
        try self.io_streams_iface.members.put(
            self.allocator,
            "[method]input-stream.read",
            .{ .func = .{ .context = self, .call = &blockingRead } },
        );
        try self.io_streams_iface.members.put(
            self.allocator,
            "[resource-drop]input-stream",
            .{ .func = .{ .context = self, .call = &dropInputStream } },
        );
        try providers.put(self.allocator, io_streams_name, .{
            .host_instance = &self.io_streams_iface,
        });
    }

    /// Register `wasi:cli/stderr` (mirror of stdout but writing to the
    /// adapter's separate stderr buffer).
    pub fn populateWasiCliStderr(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        cli_stderr_name: []const u8,
    ) !void {
        try self.cli_stderr_iface.members.put(self.allocator, "get-stderr", .{
            .func = .{ .context = self, .call = &getStderrHandle },
        });
        try providers.put(self.allocator, cli_stderr_name, .{
            .host_instance = &self.cli_stderr_iface,
        });
    }

    /// Register `wasi:cli/exit`. `exit` and `exit-with-code` set
    /// `self.exit_code` and surface as `error.Trap`; `runLoadedComponent`
    /// translates that into a normal `RunOutcome` carrying the code.
    pub fn populateWasiCliExit(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        cli_exit_name: []const u8,
    ) !void {
        try self.cli_exit_iface.members.put(self.allocator, "exit", .{
            .func = .{ .context = self, .call = &cliExit },
        });
        try self.cli_exit_iface.members.put(self.allocator, "exit-with-code", .{
            .func = .{ .context = self, .call = &cliExitWithCode },
        });
        try providers.put(self.allocator, cli_exit_name, .{
            .host_instance = &self.cli_exit_iface,
        });
    }

    /// Register `wasi:cli/environment`. All three accessors return
    /// empty/none — sufficient for stdio-style components that don't
    /// inspect env or args.
    pub fn populateWasiCliEnvironment(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        cli_env_name: []const u8,
    ) !void {
        try self.cli_environment_iface.members.put(self.allocator, "get-environment", .{
            .func = .{ .context = self, .call = &getEnvironment },
        });
        try self.cli_environment_iface.members.put(self.allocator, "get-arguments", .{
            .func = .{ .context = self, .call = &getArguments },
        });
        try self.cli_environment_iface.members.put(self.allocator, "initial-cwd", .{
            .func = .{ .context = self, .call = &initialCwd },
        });
        try providers.put(self.allocator, cli_env_name, .{
            .host_instance = &self.cli_environment_iface,
        });
    }

    /// Register `wasi:cli/terminal-{stdin,stdout,stderr,input,output}`.
    /// In captured-buffer mode there is no real TTY, so each
    /// `get-terminal-*` returns `none`. Resource-drop members on
    /// `terminal-input` / `terminal-output` are wired as no-ops in
    /// case the guest's runtime drops a freshly-pulled handle.
    pub fn populateWasiCliTerminal(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        terminal_stdin_name: []const u8,
        terminal_stdout_name: []const u8,
        terminal_stderr_name: []const u8,
        terminal_input_name: []const u8,
        terminal_output_name: []const u8,
    ) !void {
        try self.cli_terminal_stdin_iface.members.put(self.allocator, "get-terminal-stdin", .{
            .func = .{ .context = self, .call = &getTerminalNone },
        });
        try providers.put(self.allocator, terminal_stdin_name, .{
            .host_instance = &self.cli_terminal_stdin_iface,
        });
        try self.cli_terminal_stdout_iface.members.put(self.allocator, "get-terminal-stdout", .{
            .func = .{ .context = self, .call = &getTerminalNone },
        });
        try providers.put(self.allocator, terminal_stdout_name, .{
            .host_instance = &self.cli_terminal_stdout_iface,
        });
        try self.cli_terminal_stderr_iface.members.put(self.allocator, "get-terminal-stderr", .{
            .func = .{ .context = self, .call = &getTerminalNone },
        });
        try providers.put(self.allocator, terminal_stderr_name, .{
            .host_instance = &self.cli_terminal_stderr_iface,
        });
        try self.cli_terminal_input_iface.members.put(self.allocator, "[resource-drop]terminal-input", .{
            .func = .{ .context = self, .call = &noopResourceDrop },
        });
        try providers.put(self.allocator, terminal_input_name, .{
            .host_instance = &self.cli_terminal_input_iface,
        });
        try self.cli_terminal_output_iface.members.put(self.allocator, "[resource-drop]terminal-output", .{
            .func = .{ .context = self, .call = &noopResourceDrop },
        });
        try providers.put(self.allocator, terminal_output_name, .{
            .host_instance = &self.cli_terminal_output_iface,
        });
    }

    /// Register `wasi:cli/stdin` host binding. Members:
    ///   - `get-stdin: () -> own<input-stream>` returns a handle into
    ///     `self.input_stream_table` pointing at the captured stdin.
    pub fn populateWasiCliStdin(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        cli_stdin_name: []const u8,
    ) !void {
        try self.cli_stdin_iface.members.put(self.allocator, "get-stdin", .{
            .func = .{ .context = self, .call = &getStdinHandle },
        });
        try providers.put(self.allocator, cli_stdin_name, .{
            .host_instance = &self.cli_stdin_iface,
        });
    }

    /// Register `wasi:io/poll` and `wasi:io/error` host bindings.
    ///
    /// Both interfaces only need `[resource-drop]<resource>` wired for the
    /// stdio-echo happy path — the guest never blocks on a pollable nor
    /// inspects an error to-debug-string when stdout writes succeed. Any
    /// other member call will surface as an unresolved-import trap.
    pub fn populateWasiIoPollError(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        io_poll_name: []const u8,
        io_error_name: []const u8,
    ) !void {
        try self.io_poll_iface.members.put(self.allocator, "[resource-drop]pollable", .{
            .func = .{ .context = self, .call = &noopResourceDrop },
        });
        try providers.put(self.allocator, io_poll_name, .{
            .host_instance = &self.io_poll_iface,
        });

        try self.io_error_iface.members.put(self.allocator, "[resource-drop]error", .{
            .func = .{ .context = self, .call = &noopResourceDrop },
        });
        try providers.put(self.allocator, io_error_name, .{
            .host_instance = &self.io_error_iface,
        });
    }

    /// `(own<R>) -> ()` no-op — the host never produces non-stream
    /// resources on the happy path, so dropping one is purely guest-side
    /// bookkeeping that we can swallow.
    fn noopResourceDrop(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        _: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {}

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

    /// `wasi:cli/stderr.get-stderr: () -> own<output-stream>`. Mirror of
    /// `getStdoutHandle` for the captured stderr buffer.
    fn getStderrHandle(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const handle = try self.allocStreamHandle(&self.stderr);
        results[0] = .{ .handle = handle };
    }

    /// `wasi:cli/exit.exit: (result<_, _>) -> ()`. The arg is a
    /// `result<_, _>` discriminant where `0` (ok) → exit 0, `1` (err) →
    /// exit 1. Sets `self.exit_code` then traps so the run unwinds.
    fn cliExit(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        _: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        const code: u32 = if (args.len > 0) switch (args[0]) {
            .result_val => |rv| if (rv.is_ok) @as(u32, 0) else 1,
            else => 0,
        } else 0;
        self.exit_code = code;
        return error.WasiExit;
    }

    /// `wasi:cli/exit.exit-with-code: (u8) -> ()`. Sets the exit code
    /// and traps. Mirrors `cliExit` for the typed-code form added in
    /// later WASI 0.2 drafts.
    fn cliExitWithCode(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        _: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        const code: u32 = if (args.len > 0) switch (args[0]) {
            .u8 => |v| @intCast(v),
            .u32 => |v| v,
            .u64 => |v| @truncate(v),
            else => 0,
        } else 0;
        self.exit_code = code;
        return error.WasiExit;
    }

    /// `wasi:cli/environment.get-environment: () -> list<tuple<string,string>>`.
    /// Returns an empty list (canonical pair `ptr=0, len=0`).
    fn getEnvironment(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .list = .{ .ptr = 0, .len = 0 } };
    }

    /// `wasi:cli/environment.get-arguments: () -> list<string>`. Empty.
    fn getArguments(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .list = .{ .ptr = 0, .len = 0 } };
    }

    /// `wasi:cli/environment.initial-cwd: () -> option<string>`. None.
    fn initialCwd(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
    }

    /// `wasi:cli/terminal-*.get-terminal-*: () -> option<terminal-*>`.
    /// Captured-buffer mode has no TTY → return `none`.
    fn getTerminalNone(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
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

    /// `[method]output-stream.write: (borrow<output-stream>, list<u8>)
    ///   -> result<_, stream-error>`. Same write semantics as
    /// `blocking-write-and-flush`; the captured buffer never blocks.
    fn outputStreamWrite(
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

    /// `[method]output-stream.check-write: (borrow<output-stream>)
    ///   -> result<u64, stream-error>`. The captured buffer can always
    /// accept; report a generous chunk so the guest writes in one call.
    fn outputStreamCheckWrite(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        _ = ctx_opaque;
        if (args.len == 0 or results.len == 0) return error.InvalidArgs;
        const payload = try allocator.create(InterfaceValue);
        payload.* = .{ .u64 = 64 * 1024 };
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = payload } };
    }

    /// `[method]output-stream.blocking-flush: (borrow<output-stream>)
    ///   -> result<_, stream-error>`. Captured buffer is unbuffered; ok.
    fn outputStreamBlockingFlush(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `[method]output-stream.subscribe: (borrow<output-stream>)
    ///   -> own<pollable>`. Captured-buffer streams are always ready;
    /// return a sentinel handle that drop-pollable will swallow.
    fn outputStreamSubscribe(
        _: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (args.len == 0 or results.len == 0) return error.InvalidArgs;
        results[0] = .{ .handle = 0 };
    }

    /// `[method]input-stream.subscribe: (borrow<input-stream>)
    ///   -> own<pollable>`. Same sentinel as the output side — the
    /// captured stdin buffer is always ready.
    fn inputStreamSubscribe(
        _: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (args.len == 0 or results.len == 0) return error.InvalidArgs;
        results[0] = .{ .handle = 0 };
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

    fn allocInputStreamHandle(self: *WasiCliAdapter, stream: *streams.InputStream) !u32 {
        for (self.input_stream_table.items, 0..) |slot, i| {
            if (slot == null) {
                self.input_stream_table.items[i] = stream;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.input_stream_table.items.len);
        try self.input_stream_table.append(self.allocator, stream);
        return idx;
    }

    fn lookupInputStream(self: *WasiCliAdapter, handle: u32) ?*streams.InputStream {
        if (handle >= self.input_stream_table.items.len) return null;
        return self.input_stream_table.items[handle];
    }

    /// `wasi:cli/stdin.get-stdin: () -> own<input-stream>`.
    fn getStdinHandle(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const handle = try self.allocInputStreamHandle(&self.stdin);
        results[0] = .{ .handle = handle };
    }

    /// `wasi:io/streams.[method]input-stream.blocking-read:
    ///   (borrow<input-stream>, u64) -> result<list<u8>, stream-error>`.
    /// Reads up to `len` bytes from the captured stdin into a freshly
    /// guest-allocated buffer (via `cabi_realloc`) and returns the list
    /// in the ok arm. End-of-stream surfaces as the closed variant in
    /// the err arm with payload omitted (caller's `result_val.payload`
    /// is `null` — the canonical-ABI store path zero-fills the payload
    /// slots that aren't populated).
    fn blockingRead(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 2 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const want_u64 = switch (args[1]) {
            .u64 => |v| v,
            else => return error.InvalidArgs,
        };
        const stream = self.lookupInputStream(handle) orelse return error.InvalidHandle;

        const want: usize = @min(want_u64, std.math.maxInt(usize));
        // Cap at a sane upper bound so an unbounded request from the guest
        // (Rust's read_line passes 0xFFFF_FFFF_FFFF_FFFF) doesn't allocate
        // a multi-exabyte host buffer. 64 KiB matches the canonical-ABI
        // chunk used by wit-bindgen.
        const capped: usize = @min(want, 64 * 1024);

        const buf = try allocator.alloc(u8, capped);
        defer allocator.free(buf);

        switch (stream.read(buf)) {
            .ok => |n| {
                const guest_ptr = ci.hostAllocAndWrite(buf[0..n]) orelse return error.IoError;
                const list_val = try allocator.create(InterfaceValue);
                list_val.* = .{ .list = .{ .ptr = guest_ptr, .len = @intCast(n) } };
                results[0] = .{ .result_val = .{ .is_ok = true, .payload = list_val } };
            },
            .closed => {
                // err arm; payload (stream-error variant) zero-fills.
                results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            },
            .err => return error.IoError,
        }
    }

    /// `wasi:io/streams.[resource-drop]input-stream: (own<input-stream>) -> ()`.
    fn dropInputStream(
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
        if (handle < self.input_stream_table.items.len) {
            self.input_stream_table.items[handle] = null;
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
    var matched_stderr: ?[]const u8 = null;
    var matched_stdin: ?[]const u8 = null;
    var matched_exit: ?[]const u8 = null;
    var matched_environment: ?[]const u8 = null;
    var matched_terminal_stdin: ?[]const u8 = null;
    var matched_terminal_stdout: ?[]const u8 = null;
    var matched_terminal_stderr: ?[]const u8 = null;
    var matched_terminal_input: ?[]const u8 = null;
    var matched_terminal_output: ?[]const u8 = null;
    var matched_streams: ?[]const u8 = null;
    var matched_poll: ?[]const u8 = null;
    var matched_error: ?[]const u8 = null;
    for (component.imports) |imp| {
        if (imp.desc != .instance) continue;
        if (matched_stdout == null and matchesWasiPrefix(imp.name, "wasi:cli/stdout"))
            matched_stdout = imp.name;
        if (matched_stderr == null and matchesWasiPrefix(imp.name, "wasi:cli/stderr"))
            matched_stderr = imp.name;
        if (matched_stdin == null and matchesWasiPrefix(imp.name, "wasi:cli/stdin"))
            matched_stdin = imp.name;
        if (matched_exit == null and matchesWasiPrefix(imp.name, "wasi:cli/exit"))
            matched_exit = imp.name;
        if (matched_environment == null and matchesWasiPrefix(imp.name, "wasi:cli/environment"))
            matched_environment = imp.name;
        if (matched_terminal_stdin == null and matchesWasiPrefix(imp.name, "wasi:cli/terminal-stdin"))
            matched_terminal_stdin = imp.name;
        if (matched_terminal_stdout == null and matchesWasiPrefix(imp.name, "wasi:cli/terminal-stdout"))
            matched_terminal_stdout = imp.name;
        if (matched_terminal_stderr == null and matchesWasiPrefix(imp.name, "wasi:cli/terminal-stderr"))
            matched_terminal_stderr = imp.name;
        if (matched_terminal_input == null and matchesWasiPrefix(imp.name, "wasi:cli/terminal-input"))
            matched_terminal_input = imp.name;
        if (matched_terminal_output == null and matchesWasiPrefix(imp.name, "wasi:cli/terminal-output"))
            matched_terminal_output = imp.name;
        if (matched_streams == null and matchesWasiPrefix(imp.name, "wasi:io/streams"))
            matched_streams = imp.name;
        if (matched_poll == null and matchesWasiPrefix(imp.name, "wasi:io/poll"))
            matched_poll = imp.name;
        if (matched_error == null and matchesWasiPrefix(imp.name, "wasi:io/error"))
            matched_error = imp.name;
    }
    // Always populate every interface's members so the adapter's
    // HostInstance maps are well-formed; only register providers for
    // the names the component actually imports so `linkImports`
    // strict-checking surfaces unrelated misses.
    try adapter.populateWasiCliRun(
        providers,
        matched_stdout orelse "wasi:cli/stdout",
        matched_streams orelse "wasi:io/streams",
    );
    if (matched_stdout == null) _ = providers.remove("wasi:cli/stdout");
    if (matched_streams == null) _ = providers.remove("wasi:io/streams");

    try adapter.populateWasiCliStderr(
        providers,
        matched_stderr orelse "wasi:cli/stderr",
    );
    if (matched_stderr == null) _ = providers.remove("wasi:cli/stderr");

    try adapter.populateWasiCliStdin(
        providers,
        matched_stdin orelse "wasi:cli/stdin",
    );
    if (matched_stdin == null) _ = providers.remove("wasi:cli/stdin");

    try adapter.populateWasiCliExit(
        providers,
        matched_exit orelse "wasi:cli/exit",
    );
    if (matched_exit == null) _ = providers.remove("wasi:cli/exit");

    try adapter.populateWasiCliEnvironment(
        providers,
        matched_environment orelse "wasi:cli/environment",
    );
    if (matched_environment == null) _ = providers.remove("wasi:cli/environment");

    try adapter.populateWasiCliTerminal(
        providers,
        matched_terminal_stdin orelse "wasi:cli/terminal-stdin",
        matched_terminal_stdout orelse "wasi:cli/terminal-stdout",
        matched_terminal_stderr orelse "wasi:cli/terminal-stderr",
        matched_terminal_input orelse "wasi:cli/terminal-input",
        matched_terminal_output orelse "wasi:cli/terminal-output",
    );
    if (matched_terminal_stdin == null) _ = providers.remove("wasi:cli/terminal-stdin");
    if (matched_terminal_stdout == null) _ = providers.remove("wasi:cli/terminal-stdout");
    if (matched_terminal_stderr == null) _ = providers.remove("wasi:cli/terminal-stderr");
    if (matched_terminal_input == null) _ = providers.remove("wasi:cli/terminal-input");
    if (matched_terminal_output == null) _ = providers.remove("wasi:cli/terminal-output");

    try adapter.populateWasiIoPollError(
        providers,
        matched_poll orelse "wasi:io/poll",
        matched_error orelse "wasi:io/error",
    );
    if (matched_poll == null) _ = providers.remove("wasi:io/poll");
    if (matched_error == null) _ = providers.remove("wasi:io/error");
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
    if (executor_root.callComponentFunc(inst, "run", &.{}, &results, allocator)) |_| {
        return .{ .is_ok = switch (results[0]) {
            .result_val => |rv| rv.is_ok,
            else => true,
        } };
    } else |_| {
        // `wasi:cli/exit.{exit, exit-with-code}` traps after stashing
        // a code on the adapter; translate that into a normal outcome.
        if (adapter.exit_code) |code| return .{ .is_ok = code == 0 };
        return error.Trap;
    }
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
        .{ .name = "run", .desc = .{ .func = 3 }, .sort_idx = .{ .sort = .func, .idx = 3 } },
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
        .{ .name = "run", .desc = .{ .func = 3 }, .sort_idx = .{ .sort = .func, .idx = 3 } },
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

test "populateWasiProviders: binds wasi:io/poll and wasi:io/error (#154)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    // Hand-build a component with versioned poll + error instance imports.
    const imports = [_]ctypes_root.ImportDecl{
        .{ .name = "wasi:io/poll@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:io/error@0.2.6", .desc = .{ .instance = 0 } },
    };
    const component = ctypes_root.Component{
        .core_modules = &.{}, .core_instances = &.{}, .core_types = &.{},
        .components = &.{},   .instances = &.{},      .aliases = &.{},
        .types = &.{},        .canons = &.{},
        .imports = &imports,  .exports = &.{},
    };

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(testing.allocator);

    try populateWasiProviders(&adapter, &component, &providers);

    try testing.expect(providers.contains("wasi:io/poll@0.2.6"));
    try testing.expect(providers.contains("wasi:io/error@0.2.6"));
    // Bare names (no version suffix) must NOT be registered when the
    // component imports the versioned form.
    try testing.expect(!providers.contains("wasi:io/poll"));
    try testing.expect(!providers.contains("wasi:io/error"));

    // Resource-drop members are wired so the guest can drop pollables/errors.
    try testing.expect(adapter.io_poll_iface.members.contains("[resource-drop]pollable"));
    try testing.expect(adapter.io_error_iface.members.contains("[resource-drop]error"));
}

test "populateWasiProviders: binds wasi:cli/stdin (#152)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    adapter.setStdinBytes("hello\n");

    const imports = [_]ctypes_root.ImportDecl{
        .{ .name = "wasi:cli/stdin@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:io/streams@0.2.6", .desc = .{ .instance = 0 } },
    };
    const component = ctypes_root.Component{
        .core_modules = &.{}, .core_instances = &.{}, .core_types = &.{},
        .components = &.{},   .instances = &.{},      .aliases = &.{},
        .types = &.{},        .canons = &.{},
        .imports = &imports,  .exports = &.{},
    };

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(testing.allocator);

    try populateWasiProviders(&adapter, &component, &providers);

    try testing.expect(providers.contains("wasi:cli/stdin@0.2.6"));
    try testing.expect(!providers.contains("wasi:cli/stdin"));
    try testing.expect(adapter.cli_stdin_iface.members.contains("get-stdin"));
    try testing.expect(adapter.io_streams_iface.members.contains("[method]input-stream.blocking-read"));
    try testing.expect(adapter.io_streams_iface.members.contains("[resource-drop]input-stream"));

    // The captured stdin buffer is reachable as an InputStream.
    var buf: [16]u8 = undefined;
    const r = adapter.stdin.read(&buf);
    switch (r) {
        .ok => |n| try testing.expectEqualStrings("hello\n", buf[0..n]),
        else => try testing.expect(false),
    }
}

test "populateWasiProviders: binds full cli surface (#153)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const imports = [_]ctypes_root.ImportDecl{
        .{ .name = "wasi:cli/stderr@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:cli/exit@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:cli/environment@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:cli/terminal-stdin@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:cli/terminal-stdout@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:cli/terminal-stderr@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:cli/terminal-input@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:cli/terminal-output@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:io/streams@0.2.6", .desc = .{ .instance = 0 } },
    };
    const component = ctypes_root.Component{
        .core_modules = &.{}, .core_instances = &.{}, .core_types = &.{},
        .components = &.{},   .instances = &.{},      .aliases = &.{},
        .types = &.{},        .canons = &.{},
        .imports = &imports,  .exports = &.{},
    };

    var providers: std.StringHashMapUnmanaged(ImportBinding) = .empty;
    defer providers.deinit(testing.allocator);
    try populateWasiProviders(&adapter, &component, &providers);

    try testing.expect(providers.contains("wasi:cli/stderr@0.2.6"));
    try testing.expect(providers.contains("wasi:cli/exit@0.2.6"));
    try testing.expect(providers.contains("wasi:cli/environment@0.2.6"));
    try testing.expect(providers.contains("wasi:cli/terminal-stdin@0.2.6"));
    try testing.expect(providers.contains("wasi:cli/terminal-output@0.2.6"));

    try testing.expect(adapter.cli_stderr_iface.members.contains("get-stderr"));
    try testing.expect(adapter.cli_exit_iface.members.contains("exit"));
    try testing.expect(adapter.cli_exit_iface.members.contains("exit-with-code"));
    try testing.expect(adapter.cli_environment_iface.members.contains("get-environment"));
    try testing.expect(adapter.cli_environment_iface.members.contains("get-arguments"));
    try testing.expect(adapter.cli_environment_iface.members.contains("initial-cwd"));
    try testing.expect(adapter.cli_terminal_stdin_iface.members.contains("get-terminal-stdin"));
    try testing.expect(adapter.cli_terminal_input_iface.members.contains("[resource-drop]terminal-input"));
    try testing.expect(adapter.cli_terminal_output_iface.members.contains("[resource-drop]terminal-output"));

    // Output-stream method trio used by Rust's stdlib (instead of
    // `blocking-write-and-flush`).
    try testing.expect(adapter.io_streams_iface.members.contains("[method]output-stream.write"));
    try testing.expect(adapter.io_streams_iface.members.contains("[method]output-stream.check-write"));
    try testing.expect(adapter.io_streams_iface.members.contains("[method]output-stream.blocking-flush"));
    try testing.expect(adapter.io_streams_iface.members.contains("[method]output-stream.subscribe"));
    try testing.expect(adapter.io_streams_iface.members.contains("[method]input-stream.subscribe"));
}

test "stdio-echo: end-to-end real wasi-p2 component (#156)" {
    // The stdio-echo fixture (Rust → wasm32-wasip2 via wit-component) uses
    // composition machinery this runtime doesn't yet implement:
    //   * three core modules (`$main`, `$wit-component-shim-module`,
    //     `$wit-component-fixup`) wired via `(core instance (instantiate
    //     $m (with ...)))` with cross-instance export aliases;
    //   * an indirect-call trampoline table patched by `$fixup` after
    //     `$main` is instantiated;
    //   * `(canon resource.drop ...)` core funcs and full resource
    //     indexspace plumbing;
    //   * `(canon lower ...)` of host-instance methods sourced from
    //     aliased imported instance exports rather than direct imports.
    //
    // The host-side WASI surface (#150–#155) is in place, and the loader
    // / TypeRegistry now handle the type indexspace correctly so the run
    // export resolves through the wit-bindgen 0.2.0 shim. Lighting up
    // execution requires a follow-up slice that extends the core-side
    // resolver; tracked alongside #156. Once that lands, drop the skip
    // and the body below becomes the regression gate.
    return error.SkipZigTest;
}

test "stdio-echo: end-to-end real wasi-p2 component (#156, disabled body)" {
    if (true) return error.SkipZigTest;
    const testing = std.testing;
    const data = @embedFile("fixtures/stdio-echo.wasm");

    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    adapter.setStdinBytes("hello\n");

    const outcome = runComponentBytes(data, testing.allocator, &adapter) catch |err| {
        std.debug.print("stdio-echo run failed: {s}\n", .{@errorName(err)});
        std.debug.print("stdout so far: {s}\n", .{adapter.getStdoutBytes()});
        std.debug.print("stderr so far: {s}\n", .{adapter.getStderrBytes()});
        return err;
    };

    try testing.expect(outcome.is_ok);
    try testing.expectEqualStrings("echo: hello\n", adapter.getStdoutBytes());
}
