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
const wasi_p2_core = @import("../wasi/preview2/core.zig");

/// Captured stdout adapter. Owns its `OutputStream` and the `HostInstance`
/// objects exposed to the runtime via `populateProviders`.
///
/// Lifetime: caller `init`s, registers via `populateProviders`, then
/// retains the adapter until any `ComponentInstance` whose imports point
/// at the registered `HostInstance` is destroyed. `deinit` releases the
/// buffer and member maps.
/// `wasi:filesystem/types.descriptor` resource — slot in the descriptor
/// table. Indices in `WasiCliAdapter.fs_descriptor_table` are guest
/// handles. Slots are nulled on `[resource-drop]descriptor`, except
/// `.preopen` slots which are persistent (a guest dropping a preopen
/// handle is treated as a no-op so subsequent calls to
/// `preopens.get-directories` and `descriptor.open-at` keep working).
pub const FsDescriptor = union(enum) {
    /// Regular file or any other non-directory descriptor opened via
    /// `descriptor.open-at`. Owned — closed on resource-drop.
    file: std.Io.File,
    /// Directory descriptor opened via `descriptor.open-at` with the
    /// `directory` open-flag. Owned — closed on resource-drop.
    dir: std.Io.Dir,
    /// Preopen sandbox root — adapter-owned. Resource-drop is a no-op
    /// for preopens; the adapter closes them in `deinit`.
    preopen: std.Io.Dir,
};

pub const FsPreopen = struct {
    name: []const u8,
    dir_handle: u32,
};

/// Close a descriptor's underlying handle. Used by both
/// `[resource-drop]descriptor` (for `.file` / `.dir`) and adapter
/// `deinit` (which also closes preopens).
fn closeFsDescriptor(d: FsDescriptor) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    switch (d) {
        .file => |f| f.close(io),
        .dir => |dir| dir.close(io),
        .preopen => |dir| dir.close(io),
    }
}

/// `wasi:filesystem/types.error-code` discriminant indices, in the order
/// the WIT spec declares them. Used as `.variant_val.discriminant` for
/// the `result<_, error-code>` err arm.
const FsErrorCode = enum(u32) {
    access = 0,
    would_block = 1,
    already = 2,
    bad_descriptor = 3,
    busy = 4,
    deadlock = 5,
    quota = 6,
    exist = 7,
    file_too_large = 8,
    illegal_byte_sequence = 9,
    in_progress = 10,
    interrupted = 11,
    invalid = 12,
    io = 13,
    is_directory = 14,
    loop = 15,
    too_many_links = 16,
    message_size = 17,
    name_too_long = 18,
    no_device = 19,
    no_entry = 20,
    no_lock = 21,
    insufficient_memory = 22,
    insufficient_space = 23,
    not_directory = 24,
    not_empty = 25,
    not_recoverable = 26,
    unsupported = 27,
    no_tty = 28,
    no_such_device = 29,
    overflow = 30,
    not_permitted = 31,
    pipe = 32,
    read_only = 33,
    invalid_seek = 34,
    text_file_busy = 35,
    cross_device = 36,
};

/// `wasi:sockets/network.error-code` discriminants in WIT-declaration order
/// (37 variants). Used as `.variant_val.discriminant` for the err arm of
/// `result<_, error-code>` returns from sockets host fns.
const SocketErrorCode = enum(u32) {
    unknown = 0,
    access_denied = 1,
    not_supported = 2,
    invalid_argument = 3,
    out_of_memory = 4,
    timeout = 5,
    concurrency_conflict = 6,
    not_in_progress = 7,
    would_block = 8,
    invalid_state = 9,
    new_socket_limit = 10,
    address_not_bindable = 11,
    address_in_use = 12,
    remote_unreachable = 13,
    connection_refused = 14,
    connection_reset = 15,
    connection_aborted = 16,
    datagram_too_large = 17,
    name_unresolvable = 18,
    temporary_resolver_failure = 19,
    permanent_resolver_failure = 20,
};

/// IPv4 vs IPv6 — matches the discriminant order of `ip-address-family`.
const IpAddressFamily = enum(u32) {
    ipv4 = 0,
    ipv6 = 1,
};

const SocketKind = enum { tcp, udp };

const SocketState = enum { unbound, bound, listening, connected, closed };

/// Adapter-side `tcp-socket` / `udp-socket` rep. Default-deny: the adapter
/// never actually binds a host socket; methods that would touch the network
/// return `error-code.access-denied`. The slot exists so that
/// `address-family` and the other pure getters have something to read.
///
/// TODO(#148): replace with a real `std.Io.net.Socket` once the capability
/// allow-list lands. The shape is intentionally minimal.
pub const Socket = struct {
    kind: SocketKind,
    family: IpAddressFamily,
    state: SocketState = .unbound,
};

/// Adapter-side `network` rep. Currently empty — the WIT declares zero
/// methods on `network`, and the default-deny capability model means the
/// rep does not need to carry an allow-list yet.
pub const Network = struct {};

/// Adapter-side `resolve-address-stream` rep. `pos` advances on each
/// `resolve-next-address` until `pos == results.len`, after which the
/// stream returns `option<ip-address>::none`. The slice and `self`
/// itself are owned by the adapter and freed in `deinit` /
/// `[resource-drop]resolve-address-stream`.
pub const ResolveAddressStream = struct {
    results: []std.Io.net.IpAddress,
    pos: usize,
    allocator: Allocator,
};

/// Map a Zig std.fs / std.posix error to the closest `error-code` variant.
/// Errors not represented map to `.io` so the guest still sees a
/// well-typed result rather than a host trap.
fn mapFsError(err: anyerror) FsErrorCode {
    return switch (err) {
        error.AccessDenied, error.PermissionDenied => .access,
        error.FileNotFound => .no_entry,
        error.NotDir => .not_directory,
        error.PathAlreadyExists => .exist,
        error.IsDir => .is_directory,
        error.NameTooLong => .name_too_long,
        error.SymLinkLoop => .loop,
        error.FileTooBig => .file_too_large,
        error.NoSpaceLeft => .insufficient_space,
        error.OutOfMemory => .insufficient_memory,
        error.DeviceBusy => .busy,
        error.WouldBlock => .would_block,
        error.BrokenPipe => .pipe,
        error.ReadOnlyFileSystem => .read_only,
        error.InvalidUtf8 => .illegal_byte_sequence,
        else => .io,
    };
}

/// `wasi:http/types.error-code` discriminant indices in WIT-declaration
/// order. Used as `.variant_val.discriminant` for `result<_, error-code>`
/// returns. Several variants carry payloads in the WIT (option<u64>,
/// option<string>, etc.); the adapter only ever returns no-payload
/// codes from default-deny stubs, so payload handling is deferred.
const HttpErrorCode = enum(u32) {
    DNS_timeout = 0,
    DNS_error = 1,
    destination_not_found = 2,
    destination_unavailable = 3,
    destination_IP_prohibited = 4,
    destination_IP_unroutable = 5,
    connection_refused = 6,
    connection_terminated = 7,
    connection_timeout = 8,
    connection_read_timeout = 9,
    connection_write_timeout = 10,
    connection_limit_reached = 11,
    TLS_protocol_error = 12,
    TLS_certificate_error = 13,
    TLS_alert_received = 14,
    HTTP_request_denied = 15,
    HTTP_request_length_required = 16,
    HTTP_request_body_size = 17,
    HTTP_request_method_invalid = 18,
    HTTP_request_URI_invalid = 19,
    HTTP_request_URI_too_long = 20,
    HTTP_request_header_section_size = 21,
    HTTP_request_header_size = 22,
    HTTP_request_trailer_section_size = 23,
    HTTP_request_trailer_size = 24,
    HTTP_response_incomplete = 25,
    HTTP_response_header_section_size = 26,
    HTTP_response_header_size = 27,
    HTTP_response_body_size = 28,
    HTTP_response_trailer_section_size = 29,
    HTTP_response_trailer_size = 30,
    HTTP_response_transfer_coding = 31,
    HTTP_response_content_coding = 32,
    HTTP_response_timeout = 33,
    HTTP_upgrade_failed = 34,
    HTTP_protocol_error = 35,
    loop_detected = 36,
    configuration_error = 37,
    internal_error = 38,
};

/// `wasi:http/types.fields`: a multi-map of header name -> value bytes.
/// Each entry's name and value are owned host-allocated slices, freed
/// when the slot is dropped or the adapter deinits.
pub const HttpFieldEntry = struct {
    name: []u8,
    value: []u8,
};

pub const HttpFields = struct {
    entries: std.ArrayListUnmanaged(HttpFieldEntry) = .empty,
    /// `[static]fields.from-list` returns mutable fields per WIT; the
    /// flag exists so a future implementation of header-immutability
    /// (e.g. fields owned by an incoming-request) has somewhere to
    /// record it. Default-deny stubs ignore it.
    immutable: bool = false,

    pub fn deinit(self: *HttpFields, allocator: Allocator) void {
        for (self.entries.items) |e| {
            allocator.free(e.name);
            allocator.free(e.value);
        }
        self.entries.deinit(allocator);
    }
};

/// `wasi:http/types.outgoing-request`. The constructor borrows a
/// `fields` handle for headers; mutation goes through `set-*` methods
/// which the default-deny adapter accepts as ok no-ops (real header
/// validation is deferred). String fields are owned host slices.
pub const OutgoingRequest = struct {
    method_disc: u32 = 0, // .get
    method_other: ?[]u8 = null,
    path_with_query: ?[]u8 = null,
    scheme_disc: ?u32 = null,
    scheme_other: ?[]u8 = null,
    authority: ?[]u8 = null,
    headers_handle: u32,
    body_consumed: bool = false,

    pub fn deinit(self: *OutgoingRequest, allocator: Allocator) void {
        if (self.method_other) |s| allocator.free(s);
        if (self.path_with_query) |s| allocator.free(s);
        if (self.scheme_other) |s| allocator.free(s);
        if (self.authority) |s| allocator.free(s);
    }
};

/// `wasi:http/types.incoming-request`. The adapter never produces one
/// on its own (it is only constructed by an inbound HTTP server, which
/// is out of scope); the rep exists so resource-drop and the getter
/// stubs link cleanly.
pub const IncomingRequest = struct {
    headers_handle: u32 = 0,
    body_consumed: bool = false,
};

/// `wasi:http/types.incoming-response`. Used by the
/// `future-incoming-response.get` happy path once #149 follow-up wires
/// real outbound HTTP. For default-deny, no incoming-response is ever
/// produced.
pub const IncomingResponse = struct {
    status: u16 = 0,
    headers_handle: u32 = 0,
    body_consumed: bool = false,
};

/// `wasi:http/types.outgoing-response`.
pub const OutgoingResponse = struct {
    status: u16 = 200,
    headers_handle: u32,
    body_consumed: bool = false,
};

/// `wasi:http/types.incoming-body`. Placeholder rep — the adapter
/// returns `error` from `incoming-body.stream` on the default-deny
/// path, so the only operation that touches a slot is resource-drop.
pub const IncomingBody = struct {};

/// `wasi:http/types.outgoing-body`.
pub const OutgoingBody = struct {
    /// Set by `[method]outgoing-body.write` to ensure a second call
    /// returns err per WIT contract. Default-deny errs on first call
    /// too, but the field still tracks state for symmetry.
    stream_taken: bool = false,
};

/// `wasi:http/types.future-incoming-response`. Default-deny resolves
/// every future immediately to `error-code.HTTP-request-denied`.
///
/// TODO(#149 follow-up): replace `state` with an actual async result
/// once `std.http.Client` (with TLS) is wired through a capability
/// allow-list.
pub const FutureIncomingResponse = struct {
    pub const State = union(enum) {
        pending,
        ready_ok: u32, // incoming-response handle
        ready_err: u32, // HttpErrorCode discriminant
    };

    state: State,

    /// True on the second `.get()` call — the WIT says a future must
    /// return `err(())` (the outer result-of-result) once it has been
    /// consumed.
    polled: bool = false,
};

/// `wasi:http/types.future-trailers`. Default-deny resolves to
/// `some(ok(ok(none)))` (no trailers).
pub const FutureTrailers = struct {
    polled: bool = false,
};

/// `wasi:http/types.request-options`. Pure record of optional
/// timeouts; the constructor returns a fresh slot, getters return
/// option-none, setters store values.
pub const RequestOptions = struct {
    connect_timeout_ns: ?u64 = null,
    first_byte_timeout_ns: ?u64 = null,
    between_bytes_timeout_ns: ?u64 = null,
};

/// `wasi:http/types.response-outparam`. Server-side handoff slot;
/// `[static]response-outparam.set` records that the guest produced a
/// response. The host never inspects it on the default-deny path.
pub const ResponseOutparam = struct {
    set: bool = false,
};

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
    clocks_wall_iface: HostInstance = .{},
    clocks_monotonic_iface: HostInstance = .{},
    random_iface: HostInstance = .{},
    random_insecure_iface: HostInstance = .{},
    random_insecure_seed_iface: HostInstance = .{},
    fs_types_iface: HostInstance = .{},
    fs_preopens_iface: HostInstance = .{},
    sockets_network_iface: HostInstance = .{},
    sockets_instance_network_iface: HostInstance = .{},
    sockets_tcp_iface: HostInstance = .{},
    sockets_tcp_create_iface: HostInstance = .{},
    sockets_udp_iface: HostInstance = .{},
    sockets_udp_create_iface: HostInstance = .{},
    sockets_ip_name_lookup_iface: HostInstance = .{},
    http_types_iface: HostInstance = .{},
    http_outgoing_handler_iface: HostInstance = .{},
    http_incoming_handler_iface: HostInstance = .{},

    stream_table: std.ArrayListUnmanaged(?*streams.OutputStream) = .empty,
    input_stream_table: std.ArrayListUnmanaged(?*streams.InputStream) = .empty,
    /// Heap-allocated input streams created by `descriptor.read-via-stream`.
    /// Owned by the adapter; freed in `deinit`. The indices in this list
    /// are unrelated to guest handles — guest handles live in
    /// `input_stream_table`. Same arrangement for `owned_output_streams`.
    owned_input_streams: std.ArrayListUnmanaged(*streams.InputStream) = .empty,
    owned_output_streams: std.ArrayListUnmanaged(*streams.OutputStream) = .empty,

    /// `wasi:filesystem` descriptor table. Slot index = guest handle.
    /// Slots are nulled on `[resource-drop]descriptor` (except `.preopen`
    /// slots, which are persistent — see `dropFsDescriptor`).
    fs_descriptor_table: std.ArrayListUnmanaged(?FsDescriptor) = .empty,
    /// Names of the preopen slots, in `get-directories` order. Each entry's
    /// `dir_handle` indexes into `fs_descriptor_table`. The string is owned
    /// by the adapter and freed in `deinit`.
    fs_preopens: std.ArrayListUnmanaged(FsPreopen) = .empty,

    /// `wasi:sockets/network` resource table. Slot index = guest handle.
    /// Slots are nulled on `[resource-drop]network`.
    network_table: std.ArrayListUnmanaged(?Network) = .empty,
    /// `wasi:sockets/{tcp,udp}` socket resource table. Slot index = guest
    /// handle (shared across both kinds — `Socket.kind` discriminates).
    /// Slots are nulled on `[resource-drop]tcp-socket` /
    /// `[resource-drop]udp-socket`.
    socket_table: std.ArrayListUnmanaged(?Socket) = .empty,
    /// `wasi:sockets/ip-name-lookup.resolve-address-stream` table. Slots
    /// own the heap-allocated stream struct; nulled on resource-drop.
    resolve_streams: std.ArrayListUnmanaged(?*ResolveAddressStream) = .empty,

    /// `wasi:http/types` resource tables (#149). Each slot owns a
    /// heap-allocated rep struct; resource-drop nulls the slot and
    /// frees the underlying allocation. `deinit` mops up any slots
    /// the guest leaked.
    http_fields_table: std.ArrayListUnmanaged(?*HttpFields) = .empty,
    http_outgoing_requests: std.ArrayListUnmanaged(?*OutgoingRequest) = .empty,
    http_incoming_requests: std.ArrayListUnmanaged(?*IncomingRequest) = .empty,
    http_outgoing_responses: std.ArrayListUnmanaged(?*OutgoingResponse) = .empty,
    http_incoming_responses: std.ArrayListUnmanaged(?*IncomingResponse) = .empty,
    http_request_options: std.ArrayListUnmanaged(?*RequestOptions) = .empty,
    http_response_outparams: std.ArrayListUnmanaged(?*ResponseOutparam) = .empty,
    http_incoming_bodies: std.ArrayListUnmanaged(?*IncomingBody) = .empty,
    http_outgoing_bodies: std.ArrayListUnmanaged(?*OutgoingBody) = .empty,
    http_future_responses: std.ArrayListUnmanaged(?*FutureIncomingResponse) = .empty,
    http_future_trailers: std.ArrayListUnmanaged(?*FutureTrailers) = .empty,

    /// Optional deterministic-clock injection. When set, `wasi:clocks/wall-clock.now`
    /// returns this datetime instead of reading the host wall clock — used by
    /// tests that assert on the lifted `record { seconds, nanoseconds }`.
    wall_clock_override: ?Datetime = null,
    /// Optional deterministic-monotonic-clock injection. When set,
    /// `wasi:clocks/monotonic-clock.now` and `subscribe-*` use this value
    /// (`subscribe-instant` clamps the deadline; `subscribe-duration` adds
    /// to it). Defaults to live `std.time.Instant`.
    monotonic_clock_override: ?u64 = null,
    /// Counter that mints unique synthetic `pollable` handles for
    /// `subscribe-instant` / `subscribe-duration`. The handles are opaque
    /// from the runtime's perspective — every pollable produced by the
    /// clocks adapter is treated as immediately ready (same simplification
    /// as the existing `wasi:io/poll` stub). Resource-drop is a no-op.
    next_pollable_handle: u32 = 1,

    /// State for the insecure PRNG. When `null`, init-time auto-seed runs
    /// on first use. Tests can overwrite this before invoking the component
    /// to get deterministic output.
    insecure_prng: ?std.Random.DefaultPrng = null,

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
        self.clocks_wall_iface.deinit(self.allocator);
        self.clocks_monotonic_iface.deinit(self.allocator);
        self.random_iface.deinit(self.allocator);
        self.random_insecure_iface.deinit(self.allocator);
        self.random_insecure_seed_iface.deinit(self.allocator);
        self.fs_types_iface.deinit(self.allocator);
        self.fs_preopens_iface.deinit(self.allocator);
        self.sockets_network_iface.deinit(self.allocator);
        self.sockets_instance_network_iface.deinit(self.allocator);
        self.sockets_tcp_iface.deinit(self.allocator);
        self.sockets_tcp_create_iface.deinit(self.allocator);
        self.sockets_udp_iface.deinit(self.allocator);
        self.sockets_udp_create_iface.deinit(self.allocator);
        self.sockets_ip_name_lookup_iface.deinit(self.allocator);
        self.http_types_iface.deinit(self.allocator);
        self.http_outgoing_handler_iface.deinit(self.allocator);
        self.http_incoming_handler_iface.deinit(self.allocator);
        self.stream_table.deinit(self.allocator);
        self.input_stream_table.deinit(self.allocator);

        // Own-streams created by `read-via-stream`/`write-via-stream` —
        // close any underlying borrowed file refs and free the heap stream
        // structs. The streams' `host_file` variant only borrows the
        // descriptor's File handle, so we don't close it here (the
        // descriptor's slot owns it).
        for (self.owned_input_streams.items) |s| {
            s.* = undefined;
            self.allocator.destroy(s);
        }
        self.owned_input_streams.deinit(self.allocator);
        for (self.owned_output_streams.items) |s| {
            s.deinit(self.allocator);
            self.allocator.destroy(s);
        }
        self.owned_output_streams.deinit(self.allocator);

        // Close every owning descriptor in the table (including preopens).
        for (self.fs_descriptor_table.items) |slot| {
            if (slot) |d| closeFsDescriptor(d);
        }
        self.fs_descriptor_table.deinit(self.allocator);
        for (self.fs_preopens.items) |p| self.allocator.free(p.name);
        self.fs_preopens.deinit(self.allocator);

        // wasi:sockets resource tables. Sockets and networks are POD, so
        // dropping a slot simply discards it. Resolve streams own their
        // result slice + the heap struct itself.
        self.network_table.deinit(self.allocator);
        self.socket_table.deinit(self.allocator);
        for (self.resolve_streams.items) |maybe| {
            if (maybe) |s| {
                self.allocator.free(s.results);
                self.allocator.destroy(s);
            }
        }
        self.resolve_streams.deinit(self.allocator);

        // wasi:http resource tables (#149). Each slot owns its heap
        // rep; HttpFields and OutgoingRequest also own inner string
        // slices that need recursive cleanup.
        for (self.http_fields_table.items) |maybe| {
            if (maybe) |f| {
                f.deinit(self.allocator);
                self.allocator.destroy(f);
            }
        }
        self.http_fields_table.deinit(self.allocator);
        for (self.http_outgoing_requests.items) |maybe| {
            if (maybe) |r| {
                r.deinit(self.allocator);
                self.allocator.destroy(r);
            }
        }
        self.http_outgoing_requests.deinit(self.allocator);
        for (self.http_incoming_requests.items) |maybe| {
            if (maybe) |r| self.allocator.destroy(r);
        }
        self.http_incoming_requests.deinit(self.allocator);
        for (self.http_outgoing_responses.items) |maybe| {
            if (maybe) |r| self.allocator.destroy(r);
        }
        self.http_outgoing_responses.deinit(self.allocator);
        for (self.http_incoming_responses.items) |maybe| {
            if (maybe) |r| self.allocator.destroy(r);
        }
        self.http_incoming_responses.deinit(self.allocator);
        for (self.http_request_options.items) |maybe| {
            if (maybe) |r| self.allocator.destroy(r);
        }
        self.http_request_options.deinit(self.allocator);
        for (self.http_response_outparams.items) |maybe| {
            if (maybe) |r| self.allocator.destroy(r);
        }
        self.http_response_outparams.deinit(self.allocator);
        for (self.http_incoming_bodies.items) |maybe| {
            if (maybe) |b| self.allocator.destroy(b);
        }
        self.http_incoming_bodies.deinit(self.allocator);
        for (self.http_outgoing_bodies.items) |maybe| {
            if (maybe) |b| self.allocator.destroy(b);
        }
        self.http_outgoing_bodies.deinit(self.allocator);
        for (self.http_future_responses.items) |maybe| {
            if (maybe) |f| self.allocator.destroy(f);
        }
        self.http_future_responses.deinit(self.allocator);
        for (self.http_future_trailers.items) |maybe| {
            if (maybe) |f| self.allocator.destroy(f);
        }
        self.http_future_trailers.deinit(self.allocator);
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

    /// Register `wasi:clocks/wall-clock` (#146).
    ///
    /// Members:
    ///   - `now: () -> datetime`        (real-time clock; nanos since UNIX epoch)
    ///   - `resolution: () -> datetime` (clock granularity, conservatively `1ns`)
    ///
    /// `datetime` is `record { seconds: u64, nanoseconds: u32 }`. Lift uses
    /// `std.time.nanoTimestamp` unless `wall_clock_override` is set (test mode).
    pub fn populateWasiClocksWallClock(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        wall_clock_name: []const u8,
    ) !void {
        try self.clocks_wall_iface.members.put(self.allocator, "now", .{
            .func = .{ .context = self, .call = &wallClockNow },
        });
        try self.clocks_wall_iface.members.put(self.allocator, "resolution", .{
            .func = .{ .context = self, .call = &wallClockResolution },
        });
        try providers.put(self.allocator, wall_clock_name, .{
            .host_instance = &self.clocks_wall_iface,
        });
    }

    /// Register `wasi:clocks/monotonic-clock` (#146).
    ///
    /// Members:
    ///   - `now: () -> instant`                          (`u64`, ns since arbitrary epoch)
    ///   - `resolution: () -> duration`                  (`u64`, ns)
    ///   - `subscribe-instant: (instant) -> own<pollable>`
    ///   - `subscribe-duration: (duration) -> own<pollable>`
    ///
    /// The `subscribe-*` calls mint synthetic always-ready pollable handles,
    /// matching the `wasi:io/poll` stub: stdio-style components never
    /// actually multiplex on a deadline, but they do drop the returned
    /// pollables, so `[resource-drop]pollable` (already wired by
    /// `populateWasiIoPollError`) handles cleanup.
    pub fn populateWasiClocksMonotonicClock(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        monotonic_clock_name: []const u8,
    ) !void {
        try self.clocks_monotonic_iface.members.put(self.allocator, "now", .{
            .func = .{ .context = self, .call = &monotonicClockNow },
        });
        try self.clocks_monotonic_iface.members.put(self.allocator, "resolution", .{
            .func = .{ .context = self, .call = &monotonicClockResolution },
        });
        try self.clocks_monotonic_iface.members.put(self.allocator, "subscribe-instant", .{
            .func = .{ .context = self, .call = &monotonicSubscribe },
        });
        try self.clocks_monotonic_iface.members.put(self.allocator, "subscribe-duration", .{
            .func = .{ .context = self, .call = &monotonicSubscribe },
        });
        try providers.put(self.allocator, monotonic_clock_name, .{
            .host_instance = &self.clocks_monotonic_iface,
        });
    }

    /// Register `wasi:random/random` (#147).
    ///
    /// Members:
    ///   - `get-random-bytes: (len: u64) -> list<u8>` — secure random bytes
    ///     from the OS (linux: `getrandom(2)`; windows: `ProcessPrng`;
    ///     other: `arc4random_buf`). The returned `list<u8>` is allocated
    ///     into guest linear memory via `cabi_realloc` and lifted as the
    ///     canonical `(ptr, len)` pair.
    ///   - `get-random-u64: () -> u64`.
    pub fn populateWasiRandomRandom(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        random_name: []const u8,
    ) !void {
        try self.random_iface.members.put(self.allocator, "get-random-bytes", .{
            .func = .{ .context = self, .call = &getRandomBytes },
        });
        try self.random_iface.members.put(self.allocator, "get-random-u64", .{
            .func = .{ .context = self, .call = &getRandomU64 },
        });
        try providers.put(self.allocator, random_name, .{
            .host_instance = &self.random_iface,
        });
    }

    /// Register `wasi:random/insecure` (#147). Backed by
    /// `std.Random.DefaultPrng` (Xoshiro256), lazily auto-seeded from the
    /// secure source. Tests can overwrite `self.insecure_prng` for
    /// determinism.
    pub fn populateWasiRandomInsecure(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        insecure_name: []const u8,
    ) !void {
        try self.random_insecure_iface.members.put(self.allocator, "get-insecure-random-bytes", .{
            .func = .{ .context = self, .call = &getInsecureRandomBytes },
        });
        try self.random_insecure_iface.members.put(self.allocator, "get-insecure-random-u64", .{
            .func = .{ .context = self, .call = &getInsecureRandomU64 },
        });
        try providers.put(self.allocator, insecure_name, .{
            .host_instance = &self.random_insecure_iface,
        });
    }

    /// Register `wasi:random/insecure-seed` (#147). `insecure-seed: () ->
    /// tuple<u64, u64>` returns 128 bits of *secure* entropy that the
    /// guest can use to seed its own PRNG. The "insecure" label refers to
    /// the *guest's* responsibility, not the host source quality.
    pub fn populateWasiRandomInsecureSeed(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        insecure_seed_name: []const u8,
    ) !void {
        try self.random_insecure_seed_iface.members.put(self.allocator, "insecure-seed", .{
            .func = .{ .context = self, .call = &insecureSeed },
        });
        try providers.put(self.allocator, insecure_seed_name, .{
            .host_instance = &self.random_insecure_seed_iface,
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

    /// `wasi:random/random.get-random-bytes: (u64) -> list<u8>` (#147).
    fn getRandomBytes(
        _: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const want_u64 = switch (args[0]) {
            .u64 => |v| v,
            else => return error.InvalidArgs,
        };
        const want: usize = @min(want_u64, std.math.maxInt(usize));
        // Cap at 64 KiB — same cap `blockingRead` uses, prevents a hostile
        // guest from forcing a multi-exabyte host allocation.
        const capped: usize = @min(want, 64 * 1024);
        const buf = try allocator.alloc(u8, capped);
        defer allocator.free(buf);
        wasi_p2_core.Random.getRandomBytes(buf);
        const guest_ptr = ci.hostAllocAndWrite(buf) orelse return error.IoError;
        results[0] = .{ .list = .{ .ptr = guest_ptr, .len = @intCast(capped) } };
    }

    /// `wasi:random/random.get-random-u64: () -> u64` (#147).
    fn getRandomU64(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .u64 = wasi_p2_core.Random.getRandomU64() };
    }

    /// `wasi:random/insecure.get-insecure-random-bytes: (u64) -> list<u8>` (#147).
    fn getInsecureRandomBytes(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const want_u64 = switch (args[0]) {
            .u64 => |v| v,
            else => return error.InvalidArgs,
        };
        const want: usize = @min(want_u64, std.math.maxInt(usize));
        const capped: usize = @min(want, 64 * 1024);
        const buf = try allocator.alloc(u8, capped);
        defer allocator.free(buf);
        self.ensureInsecurePrng().bytes(buf);
        const guest_ptr = ci.hostAllocAndWrite(buf) orelse return error.IoError;
        results[0] = .{ .list = .{ .ptr = guest_ptr, .len = @intCast(capped) } };
    }

    /// `wasi:random/insecure.get-insecure-random-u64: () -> u64` (#147).
    fn getInsecureRandomU64(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .u64 = self.ensureInsecurePrng().int(u64) };
    }

    /// `wasi:random/insecure-seed.insecure-seed: () -> tuple<u64, u64>` (#147).
    fn insecureSeed(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        const fields = try allocator.alloc(InterfaceValue, 2);
        fields[0] = .{ .u64 = wasi_p2_core.Random.getRandomU64() };
        fields[1] = .{ .u64 = wasi_p2_core.Random.getRandomU64() };
        results[0] = .{ .tuple_val = fields };
    }

    /// Lazy-init view onto the adapter's insecure PRNG. Auto-seeds from
    /// the OS on first call; tests can pin a deterministic seed by
    /// writing `self.insecure_prng` directly first.
    fn ensureInsecurePrng(self: *WasiCliAdapter) std.Random {
        if (self.insecure_prng == null) {
            var seed_bytes: [8]u8 = undefined;
            wasi_p2_core.Random.getRandomBytes(&seed_bytes);
            const seed = std.mem.readInt(u64, &seed_bytes, .little);
            self.insecure_prng = std.Random.DefaultPrng.init(seed);
        }
        return self.insecure_prng.?.random();
    }

    /// `wasi:clocks/wall-clock.now: () -> datetime` (#146).
    /// Lifts UNIX-epoch nanos into `record { seconds: u64, nanoseconds: u32 }`.
    /// Honors `self.wall_clock_override` for deterministic tests.
    fn wallClockNow(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const dt = self.wall_clock_override orelse readClockDatetime(.REALTIME);
        results[0] = try buildDatetimeRecord(allocator, dt);
    }

    /// `wasi:clocks/wall-clock.resolution: () -> datetime` (#146).
    /// Reports a 1-nanosecond resolution; conservative for any host that
    /// can read `nanoTimestamp`.
    fn wallClockResolution(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = try buildDatetimeRecord(allocator, .{ .seconds = 0, .nanoseconds = 1 });
    }

    /// `wasi:clocks/monotonic-clock.now: () -> instant` (#146).
    /// `instant` lifts as a flat `u64`. Honors `monotonic_clock_override`.
    fn monotonicClockNow(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .u64 = self.monotonicNs() };
    }

    /// `wasi:clocks/monotonic-clock.resolution: () -> duration` (#146).
    /// `duration` lifts as `u64`; we report 1ns.
    fn monotonicClockResolution(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .u64 = 1 };
    }

    /// `wasi:clocks/monotonic-clock.subscribe-instant: (instant) -> own<pollable>`
    /// **and** `subscribe-duration: (duration) -> own<pollable>` (#146).
    ///
    /// Both shapes are identical at the canonical-ABI level (`(u64) -> i32`),
    /// so a single host fn services both members. We mint a synthetic
    /// always-ready pollable handle. Real `poll.poll` integration arrives
    /// when (and if) the runtime grows a cooperative scheduler.
    fn monotonicSubscribe(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        // The deadline argument is ignored by the always-ready stub, but
        // accept any flat-u64-like representation produced by the lift.
        switch (args[0]) {
            .u64, .s64, .u32, .s32 => {},
            else => return error.InvalidArgs,
        }
        const handle = self.next_pollable_handle;
        self.next_pollable_handle +%= 1;
        results[0] = .{ .handle = handle };
    }

    /// Read the monotonic clock as nanoseconds. Honors
    /// `monotonic_clock_override` for tests.
    fn monotonicNs(self: *const WasiCliAdapter) u64 {
        if (self.monotonic_clock_override) |v| return v;
        return readClockNs(.MONOTONIC);
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

    // ── wasi:filesystem (#145) ─────────────────────────────────────────────

    /// Append `dir` to the descriptor table as a `.preopen` slot and
    /// register it in `fs_preopens` under `name`. Returns the descriptor
    /// handle (slot index) so callers can pre-bake guest expectations.
    /// The adapter takes ownership of `dir` (closed in `deinit`); `name`
    /// is duplicated.
    pub fn addPreopen(self: *WasiCliAdapter, name: []const u8, dir: std.Io.Dir) !u32 {
        const slot_idx: u32 = @intCast(self.fs_descriptor_table.items.len);
        try self.fs_descriptor_table.append(self.allocator, .{ .preopen = dir });
        const dup_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(dup_name);
        try self.fs_preopens.append(self.allocator, .{ .name = dup_name, .dir_handle = slot_idx });
        return slot_idx;
    }

    /// Register `wasi:filesystem/preopens` (#145).
    ///
    /// Single member `get-directories: () -> list<tuple<own<descriptor>, string>>`.
    /// Each tuple element is laid out in guest memory as 12 bytes:
    ///   +0  handle: u32
    ///   +4  string-ptr: u32
    ///   +8  string-len: u32
    /// (`record max-align = 4`, see `canonical_abi.zig:336-343`.)
    pub fn populateWasiFilesystemPreopens(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        try self.fs_preopens_iface.members.put(self.allocator, "get-directories", .{
            .func = .{ .context = self, .call = &fsGetDirectories },
        });
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.fs_preopens_iface,
        });
    }

    /// Register `wasi:filesystem/types` (#145).
    pub fn populateWasiFilesystemTypes(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        const M = struct { name: []const u8, call: *const fn (?*anyopaque, *ComponentInstance, []const InterfaceValue, []InterfaceValue, Allocator) anyerror!void };
        const members = [_]M{
            .{ .name = "[method]descriptor.get-type", .call = &fsDescriptorGetType },
            .{ .name = "[method]descriptor.stat", .call = &fsDescriptorStat },
            .{ .name = "[method]descriptor.open-at", .call = &fsDescriptorOpenAt },
            .{ .name = "[method]descriptor.read-via-stream", .call = &fsDescriptorReadViaStream },
            .{ .name = "[method]descriptor.write-via-stream", .call = &fsDescriptorWriteViaStream },
            .{ .name = "[method]descriptor.append-via-stream", .call = &fsDescriptorAppendViaStream },
            .{ .name = "[resource-drop]descriptor", .call = &fsDescriptorDrop },
        };
        for (members) |m| {
            try self.fs_types_iface.members.put(self.allocator, m.name, .{
                .func = .{ .context = self, .call = m.call },
            });
        }
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.fs_types_iface,
        });
    }

    /// Reject paths that would escape the preopen sandbox: any `..` path
    /// component, any `\\` separator, any `:` (Windows drive prefix), or
    /// a leading `/` (absolute). Returns `.access` on rejection.
    fn validateSandboxPath(path: []const u8) ?FsErrorCode {
        if (path.len == 0) return null;
        if (path[0] == '/') return .access;
        for (path) |c| {
            if (c == '\\' or c == ':') return .access;
        }
        var it = std.mem.splitScalar(u8, path, '/');
        while (it.next()) |comp| {
            if (std.mem.eql(u8, comp, "..")) return .access;
        }
        return null;
    }

    fn lookupFsDescriptor(self: *WasiCliAdapter, handle: u32) ?*FsDescriptor {
        if (handle >= self.fs_descriptor_table.items.len) return null;
        if (self.fs_descriptor_table.items[handle]) |*d| return d;
        return null;
    }

    /// Append a new descriptor slot, returning the handle. Reuses null
    /// slots if any.
    fn pushFsDescriptor(self: *WasiCliAdapter, d: FsDescriptor) !u32 {
        for (self.fs_descriptor_table.items, 0..) |slot, i| {
            if (slot == null) {
                self.fs_descriptor_table.items[i] = d;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.fs_descriptor_table.items.len);
        try self.fs_descriptor_table.append(self.allocator, d);
        return idx;
    }

    /// Build a `result<X, error-code>` lift where the err arm carries the
    /// `error-code` variant `code`. Caller-owned via `allocator`.
    fn fsResultErr(allocator: Allocator, code: FsErrorCode) !InterfaceValue {
        const payload = try allocator.create(InterfaceValue);
        payload.* = .{ .variant_val = .{ .discriminant = @intFromEnum(code), .payload = null } };
        return .{ .result_val = .{ .is_ok = false, .payload = payload } };
    }

    /// Build a `result<X, error-code>` ok lift. `value` is moved into a
    /// fresh `*InterfaceValue` payload.
    fn fsResultOk(allocator: Allocator, value: InterfaceValue) !InterfaceValue {
        const payload = try allocator.create(InterfaceValue);
        payload.* = value;
        return .{ .result_val = .{ .is_ok = true, .payload = payload } };
    }

    /// `wasi:filesystem/preopens.get-directories: () -> list<tuple<own<descriptor>, string>>`.
    fn fsGetDirectories(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;

        const n = self.fs_preopens.items.len;
        if (n == 0) {
            results[0] = .{ .list = .{ .ptr = 0, .len = 0 } };
            return;
        }

        // Each tuple<own<descriptor>, string>: handle@0:u32, ptr@4:u32, len@8:u32.
        const stride: usize = 12;
        const scratch = try allocator.alloc(u8, n * stride);
        defer allocator.free(scratch);

        for (self.fs_preopens.items, 0..) |p, i| {
            const name_ptr = ci.hostAllocAndWrite(p.name) orelse return error.IoError;
            const off = i * stride;
            std.mem.writeInt(u32, scratch[off..][0..4], p.dir_handle, .little);
            std.mem.writeInt(u32, scratch[off + 4 ..][0..4], name_ptr, .little);
            std.mem.writeInt(u32, scratch[off + 8 ..][0..4], @intCast(p.name.len), .little);
        }

        const list_ptr = ci.hostAllocAndWrite(scratch) orelse return error.IoError;
        results[0] = .{ .list = .{ .ptr = list_ptr, .len = @intCast(n) } };
    }

    /// `wasi:filesystem/types.descriptor-type` discriminants in WIT order.
    const DescType = enum(u32) {
        unknown = 0,
        block_device = 1,
        character_device = 2,
        directory = 3,
        fifo = 4,
        symbolic_link = 5,
        regular_file = 6,
        socket = 7,
    };

    /// `[method]descriptor.get-type: (borrow<descriptor>) -> result<descriptor-type, error-code>`.
    fn fsDescriptorGetType(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const d = self.lookupFsDescriptor(handle) orelse {
            results[0] = try fsResultErr(allocator, .bad_descriptor);
            return;
        };
        const dt: DescType = switch (d.*) {
            .preopen, .dir => .directory,
            .file => .regular_file,
        };
        const variant = InterfaceValue{ .variant_val = .{
            .discriminant = @intFromEnum(dt),
            .payload = null,
        } };
        results[0] = try fsResultOk(allocator, variant);
    }

    /// `[method]descriptor.stat: (borrow<descriptor>) -> result<descriptor-stat, error-code>`.
    /// The record fields are `(type, link-count, size, atime?, mtime?, ctime?)`.
    /// Timestamps are reported as `none` to keep the implementation portable
    /// across the std.Io vtable (whose `Timestamp` shape changed between
    /// 0.15 and 0.16); a dedicated atime/mtime adapter is deferred.
    fn fsDescriptorStat(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const d = self.lookupFsDescriptor(handle) orelse {
            results[0] = try fsResultErr(allocator, .bad_descriptor);
            return;
        };

        const io = std.Io.Threaded.global_single_threaded.io();
        var dt: DescType = .unknown;
        var size: u64 = 0;
        var nlink: u64 = 1;

        switch (d.*) {
            .preopen, .dir => {
                dt = .directory;
            },
            .file => |f| {
                dt = .regular_file;
                if (f.stat(io)) |st| {
                    size = st.size;
                    nlink = @intCast(st.nlink);
                    dt = switch (st.kind) {
                        .directory => .directory,
                        .file => .regular_file,
                        .block_device => .block_device,
                        .character_device => .character_device,
                        .named_pipe => .fifo,
                        .sym_link => .symbolic_link,
                        .unix_domain_socket => .socket,
                        else => .unknown,
                    };
                } else |err| {
                    results[0] = try fsResultErr(allocator, mapFsError(err));
                    return;
                }
            },
        }

        const fields = try allocator.alloc(InterfaceValue, 6);
        fields[0] = .{ .variant_val = .{ .discriminant = @intFromEnum(dt), .payload = null } };
        fields[1] = .{ .u64 = nlink };
        fields[2] = .{ .u64 = size };
        fields[3] = .{ .option_val = .{ .is_some = false, .payload = null } };
        fields[4] = .{ .option_val = .{ .is_some = false, .payload = null } };
        fields[5] = .{ .option_val = .{ .is_some = false, .payload = null } };
        results[0] = try fsResultOk(allocator, .{ .record_val = fields });
    }

    /// `[method]descriptor.open-at: (borrow<descriptor>, path-flags, string,
    ///   open-flags, descriptor-flags) -> result<own<descriptor>, error-code>`.
    fn fsDescriptorOpenAt(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 5 or results.len == 0) return error.InvalidArgs;

        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        // path-flags (args[1]): currently advisory; we always follow symlinks.
        const path_pl = switch (args[2]) {
            .string => |pl| pl,
            else => return error.InvalidArgs,
        };
        const open_flags: u32 = switch (args[3]) {
            .flags_val => |w| if (w.len == 0) 0 else w[0],
            .u32 => |v| v,
            else => 0,
        };
        const desc_flags: u32 = switch (args[4]) {
            .flags_val => |w| if (w.len == 0) 0 else w[0],
            .u32 => |v| v,
            else => 0,
        };

        const d = self.lookupFsDescriptor(handle) orelse {
            results[0] = try fsResultErr(allocator, .bad_descriptor);
            return;
        };
        const base_dir: std.Io.Dir = switch (d.*) {
            .preopen => |dir| dir,
            .dir => |dir| dir,
            .file => {
                results[0] = try fsResultErr(allocator, .not_directory);
                return;
            },
        };

        const path_bytes = ci.readGuestBytes(path_pl.ptr, path_pl.len) orelse
            return error.OutOfBoundsMemory;

        if (validateSandboxPath(path_bytes)) |code| {
            results[0] = try fsResultErr(allocator, code);
            return;
        }

        // open-flags bits per WIT order: 0=create, 1=directory, 2=exclusive, 3=truncate.
        const want_create = (open_flags & 0b0001) != 0;
        const want_directory = (open_flags & 0b0010) != 0;
        const want_exclusive = (open_flags & 0b0100) != 0;
        const want_truncate = (open_flags & 0b1000) != 0;
        // descriptor-flags bits: 0=read, 1=write.
        const want_read = (desc_flags & 0b01) != 0;
        const want_write = (desc_flags & 0b10) != 0;

        const io = std.Io.Threaded.global_single_threaded.io();

        if (want_directory) {
            const new_dir = base_dir.openDir(io, path_bytes, .{}) catch |err| {
                results[0] = try fsResultErr(allocator, mapFsError(err));
                return;
            };
            const new_handle = self.pushFsDescriptor(.{ .dir = new_dir }) catch {
                new_dir.close(io);
                results[0] = try fsResultErr(allocator, .insufficient_memory);
                return;
            };
            results[0] = try fsResultOk(allocator, .{ .handle = new_handle });
            return;
        }

        const new_file = if (want_create) base_dir.createFile(io, path_bytes, .{
            .read = want_read,
            .truncate = want_truncate,
            .exclusive = want_exclusive,
        }) catch |err| {
            results[0] = try fsResultErr(allocator, mapFsError(err));
            return;
        } else base_dir.openFile(io, path_bytes, .{
            .mode = if (want_write and want_read)
                .read_write
            else if (want_write)
                .write_only
            else
                .read_only,
        }) catch |err| {
            results[0] = try fsResultErr(allocator, mapFsError(err));
            return;
        };

        const new_handle = self.pushFsDescriptor(.{ .file = new_file }) catch {
            new_file.close(io);
            results[0] = try fsResultErr(allocator, .insufficient_memory);
            return;
        };
        results[0] = try fsResultOk(allocator, .{ .handle = new_handle });
    }

    /// `[method]descriptor.read-via-stream: (borrow<descriptor>, u64) -> result<own<input-stream>, error-code>`.
    fn fsDescriptorReadViaStream(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
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
        const offset = switch (args[1]) {
            .u64 => |v| v,
            else => return error.InvalidArgs,
        };
        const d = self.lookupFsDescriptor(handle) orelse {
            results[0] = try fsResultErr(allocator, .bad_descriptor);
            return;
        };
        const file = switch (d.*) {
            .file => |f| f,
            .dir, .preopen => {
                results[0] = try fsResultErr(allocator, .is_directory);
                return;
            },
        };
        const stream = self.allocator.create(streams.InputStream) catch {
            results[0] = try fsResultErr(allocator, .insufficient_memory);
            return;
        };
        stream.* = streams.InputStream.fromHostFile(file, offset);
        try self.owned_input_streams.append(self.allocator, stream);
        const stream_handle = try self.allocInputStreamHandle(stream);
        results[0] = try fsResultOk(allocator, .{ .handle = stream_handle });
    }

    /// `[method]descriptor.write-via-stream: (borrow<descriptor>, u64) -> result<own<output-stream>, error-code>`.
    fn fsDescriptorWriteViaStream(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
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
        const offset = switch (args[1]) {
            .u64 => |v| v,
            else => return error.InvalidArgs,
        };
        const stream_handle = try self.fsAllocOutputFileStream(handle, offset, false, allocator, results);
        if (stream_handle) |h| {
            results[0] = try fsResultOk(allocator, .{ .handle = h });
        }
    }

    /// `[method]descriptor.append-via-stream: (borrow<descriptor>) -> result<own<output-stream>, error-code>`.
    fn fsDescriptorAppendViaStream(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const stream_handle = try self.fsAllocOutputFileStream(handle, 0, true, allocator, results);
        if (stream_handle) |h| {
            results[0] = try fsResultOk(allocator, .{ .handle = h });
        }
    }

    /// Common helper for `write-via-stream` / `append-via-stream`.
    /// On error, writes the err-arm `result_val` into `results[0]` and
    /// returns `null`. On success, returns the new stream handle and
    /// leaves `results[0]` untouched (caller wraps).
    fn fsAllocOutputFileStream(
        self: *WasiCliAdapter,
        handle: u32,
        offset: u64,
        append: bool,
        allocator: Allocator,
        results: []InterfaceValue,
    ) !?u32 {
        const d = self.lookupFsDescriptor(handle) orelse {
            results[0] = try fsResultErr(allocator, .bad_descriptor);
            return null;
        };
        const file = switch (d.*) {
            .file => |f| f,
            .dir, .preopen => {
                results[0] = try fsResultErr(allocator, .is_directory);
                return null;
            },
        };
        const stream = self.allocator.create(streams.OutputStream) catch {
            results[0] = try fsResultErr(allocator, .insufficient_memory);
            return null;
        };
        stream.* = streams.OutputStream.toHostFile(file, offset, append);
        try self.owned_output_streams.append(self.allocator, stream);
        return try self.allocStreamHandle(stream);
    }

    /// `wasi:filesystem/types.[resource-drop]descriptor: (own<descriptor>) -> ()`.
    /// Preopen slots are persistent — drop is a no-op so subsequent calls
    /// to `preopens.get-directories` and `descriptor.open-at` continue
    /// to work.
    fn fsDescriptorDrop(
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
        if (handle >= self.fs_descriptor_table.items.len) return;
        const slot = self.fs_descriptor_table.items[handle] orelse return;
        switch (slot) {
            .preopen => {}, // persistent — keep the slot live.
            else => {
                closeFsDescriptor(slot);
                self.fs_descriptor_table.items[handle] = null;
            },
        }
    }

    // ── wasi:sockets/* (#148) ──────────────────────────────────────────────
    //
    // Default-deny capability model: every method that would touch the
    // network returns `error-code.access-denied` (or, for resolution,
    // `name-unresolvable`). Pure getters (e.g. `address-family`,
    // `is-listening`) read the adapter-side `Socket` rep so they can
    // answer without IO. The TODOs throughout flag the code paths the
    // future allow-list capability work will fill in.

    /// Build a `result<X, error-code>` err InterfaceValue. Caller-owned via
    /// `allocator`. Mirrors `fsResultErr` but specialized to sockets'
    /// error-code variant indices.
    fn socketResultErr(allocator: Allocator, code: SocketErrorCode) !InterfaceValue {
        const payload = try allocator.create(InterfaceValue);
        payload.* = .{ .variant_val = .{ .discriminant = @intFromEnum(code), .payload = null } };
        return .{ .result_val = .{ .is_ok = false, .payload = payload } };
    }

    /// Build a `result<X, error-code>` ok InterfaceValue.
    fn socketResultOk(allocator: Allocator, value: InterfaceValue) !InterfaceValue {
        const payload = try allocator.create(InterfaceValue);
        payload.* = value;
        return .{ .result_val = .{ .is_ok = true, .payload = payload } };
    }

    fn lookupSocket(self: *WasiCliAdapter, handle: u32) ?*Socket {
        if (handle >= self.socket_table.items.len) return null;
        if (self.socket_table.items[handle]) |*s| return s;
        return null;
    }

    fn pushSocket(self: *WasiCliAdapter, s: Socket) !u32 {
        for (self.socket_table.items, 0..) |slot, i| {
            if (slot == null) {
                self.socket_table.items[i] = s;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.socket_table.items.len);
        try self.socket_table.append(self.allocator, s);
        return idx;
    }

    fn pushNetwork(self: *WasiCliAdapter, n: Network) !u32 {
        for (self.network_table.items, 0..) |slot, i| {
            if (slot == null) {
                self.network_table.items[i] = n;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.network_table.items.len);
        try self.network_table.append(self.allocator, n);
        return idx;
    }

    fn pushResolveStream(self: *WasiCliAdapter, s: *ResolveAddressStream) !u32 {
        for (self.resolve_streams.items, 0..) |slot, i| {
            if (slot == null) {
                self.resolve_streams.items[i] = s;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.resolve_streams.items.len);
        try self.resolve_streams.append(self.allocator, s);
        return idx;
    }

    /// Generic `(...) -> result<_, error-code>` access-denied stub. Used as
    /// the body of every default-deny socket method.
    fn socketDenyAccess(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = try socketResultErr(allocator, .access_denied);
    }

    /// `(own<R>) -> ()` no-op resource drop that nulls the matching slot in
    /// the socket table.
    fn socketResourceDrop(
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
        if (handle < self.socket_table.items.len) {
            self.socket_table.items[handle] = null;
        }
    }

    fn networkResourceDrop(
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
        if (handle < self.network_table.items.len) {
            self.network_table.items[handle] = null;
        }
    }

    /// `wasi:sockets/instance-network.instance-network: () -> own<network>`.
    fn instanceNetwork(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const h = try self.pushNetwork(.{});
        results[0] = .{ .handle = h };
    }

    /// `wasi:sockets/tcp-create-socket.create-tcp-socket:
    ///   (ip-address-family) -> result<own<tcp-socket>, error-code>`.
    fn createTcpSocket(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const fam_disc: u32 = switch (args[0]) {
            .enum_val => |d| d,
            .u32 => |d| d,
            .variant_val => |v| v.discriminant,
            else => return error.InvalidArgs,
        };
        const fam: IpAddressFamily = if (fam_disc == 0) .ipv4 else .ipv6;
        const h = try self.pushSocket(.{ .kind = .tcp, .family = fam });
        results[0] = try socketResultOk(allocator, .{ .handle = h });
    }

    /// `wasi:sockets/udp-create-socket.create-udp-socket:
    ///   (ip-address-family) -> result<own<udp-socket>, error-code>`.
    fn createUdpSocket(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const fam_disc: u32 = switch (args[0]) {
            .enum_val => |d| d,
            .u32 => |d| d,
            .variant_val => |v| v.discriminant,
            else => return error.InvalidArgs,
        };
        const fam: IpAddressFamily = if (fam_disc == 0) .ipv4 else .ipv6;
        const h = try self.pushSocket(.{ .kind = .udp, .family = fam });
        results[0] = try socketResultOk(allocator, .{ .handle = h });
    }

    /// `[method]tcp-socket.address-family / [method]udp-socket.address-family:
    ///   (borrow<X>) -> ip-address-family`. Pure getter — reads the rep.
    fn socketAddressFamily(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const s = self.lookupSocket(handle) orelse {
            // Unknown rep — default to ipv4 to keep the call total.
            results[0] = .{ .enum_val = @intFromEnum(IpAddressFamily.ipv4) };
            return;
        };
        results[0] = .{ .enum_val = @intFromEnum(s.family) };
    }

    /// `[method]tcp-socket.is-listening: (borrow<tcp-socket>) -> bool`.
    fn tcpIsListening(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const s = self.lookupSocket(handle) orelse {
            results[0] = .{ .bool = false };
            return;
        };
        results[0] = .{ .bool = s.state == .listening };
    }

    /// `[method]X.subscribe: (borrow<X>) -> own<pollable>`. Mints a stub
    /// always-ready pollable — same simplification as the clocks adapter.
    fn socketSubscribe(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const h = self.next_pollable_handle;
        self.next_pollable_handle += 1;
        results[0] = .{ .handle = h };
    }

    /// `wasi:sockets/ip-name-lookup.resolve-addresses:
    ///   (borrow<network>, string) -> result<own<resolve-address-stream>,
    ///                                        error-code>`.
    ///
    /// Default-deny: always returns `error-code.name-unresolvable`. A
    /// real `std.Io.net.HostName.lookup` integration is deferred until
    /// the allow-list capability lands (see #148 follow-up).
    fn resolveAddresses(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = try socketResultErr(allocator, .name_unresolvable);
    }

    /// `[method]resolve-address-stream.resolve-next-address:
    ///   (borrow<resolve-address-stream>) -> result<option<ip-address>, error-code>`.
    fn resolveNextAddress(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        if (handle >= self.resolve_streams.items.len) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        const stream = self.resolve_streams.items[handle] orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        // Always exhausted on the default-deny path — return ok(none).
        _ = stream;
        const none_val = InterfaceValue{ .option_val = .{ .is_some = false, .payload = null } };
        results[0] = try socketResultOk(allocator, none_val);
    }

    /// `[resource-drop]resolve-address-stream`.
    fn resolveStreamDrop(
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
        if (handle >= self.resolve_streams.items.len) return;
        if (self.resolve_streams.items[handle]) |s| {
            self.allocator.free(s.results);
            self.allocator.destroy(s);
            self.resolve_streams.items[handle] = null;
        }
    }

    /// Register `wasi:sockets/network` (#148). The WIT declares zero
    /// methods; only the resource-drop is bound.
    pub fn populateWasiSocketsNetwork(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        try self.sockets_network_iface.members.put(self.allocator, "[resource-drop]network", .{
            .func = .{ .context = self, .call = &networkResourceDrop },
        });
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.sockets_network_iface,
        });
    }

    /// Register `wasi:sockets/instance-network` (#148).
    pub fn populateWasiSocketsInstanceNetwork(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        try self.sockets_instance_network_iface.members.put(self.allocator, "instance-network", .{
            .func = .{ .context = self, .call = &instanceNetwork },
        });
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.sockets_instance_network_iface,
        });
    }

    /// Register `wasi:sockets/tcp` (#148). Default-deny: every method that
    /// would touch the network returns `error-code.access-denied`. Pure
    /// getters (`address-family`, `is-listening`) read the rep.
    pub fn populateWasiSocketsTcp(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        const M = struct { name: []const u8, call: *const fn (?*anyopaque, *ComponentInstance, []const InterfaceValue, []InterfaceValue, Allocator) anyerror!void };
        // Per-method routing: most methods go to the default-deny stub;
        // a few have specific handlers.
        const members = [_]M{
            // network IO — all access-denied.
            .{ .name = "[method]tcp-socket.start-bind", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.finish-bind", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.start-connect", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.finish-connect", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.start-listen", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.finish-listen", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.accept", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.local-address", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.remote-address", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.shutdown", .call = &socketDenyAccess },
            // setters that surface socket options — also access-denied.
            .{ .name = "[method]tcp-socket.set-listen-backlog-size", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.set-keep-alive-enabled", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.set-keep-alive-idle-time", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.set-keep-alive-interval", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.set-keep-alive-count", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.set-hop-limit", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.set-receive-buffer-size", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.set-send-buffer-size", .call = &socketDenyAccess },
            // getters that surface socket options. They return
            // `result<X, error-code>` per WIT, so default-deny is well-typed.
            .{ .name = "[method]tcp-socket.keep-alive-enabled", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.keep-alive-idle-time", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.keep-alive-interval", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.keep-alive-count", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.hop-limit", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.receive-buffer-size", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.send-buffer-size", .call = &socketDenyAccess },
            // pure getters answerable from the rep.
            .{ .name = "[method]tcp-socket.address-family", .call = &socketAddressFamily },
            .{ .name = "[method]tcp-socket.is-listening", .call = &tcpIsListening },
            // pollable + drop.
            .{ .name = "[method]tcp-socket.subscribe", .call = &socketSubscribe },
            .{ .name = "[resource-drop]tcp-socket", .call = &socketResourceDrop },
        };
        for (members) |m| {
            try self.sockets_tcp_iface.members.put(self.allocator, m.name, .{
                .func = .{ .context = self, .call = m.call },
            });
        }
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.sockets_tcp_iface,
        });
    }

    /// Register `wasi:sockets/tcp-create-socket` (#148).
    pub fn populateWasiSocketsTcpCreateSocket(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        try self.sockets_tcp_create_iface.members.put(self.allocator, "create-tcp-socket", .{
            .func = .{ .context = self, .call = &createTcpSocket },
        });
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.sockets_tcp_create_iface,
        });
    }

    /// Register `wasi:sockets/udp` (#148). Default-deny across all IO
    /// methods. The two sub-resources `incoming-datagram-stream` and
    /// `outgoing-datagram-stream` only have their resource-drops bound;
    /// the host never produces handles for them on the default-deny path.
    pub fn populateWasiSocketsUdp(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        const M = struct { name: []const u8, call: *const fn (?*anyopaque, *ComponentInstance, []const InterfaceValue, []InterfaceValue, Allocator) anyerror!void };
        const members = [_]M{
            .{ .name = "[method]udp-socket.start-bind", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.finish-bind", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.stream", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.local-address", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.remote-address", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.unicast-hop-limit", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.set-unicast-hop-limit", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.receive-buffer-size", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.set-receive-buffer-size", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.send-buffer-size", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.set-send-buffer-size", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.address-family", .call = &socketAddressFamily },
            .{ .name = "[method]udp-socket.subscribe", .call = &socketSubscribe },
            .{ .name = "[resource-drop]udp-socket", .call = &socketResourceDrop },
            // Datagram-stream sub-resources. The host never returns a
            // handle to one (because `udp-socket.stream` is access-denied),
            // but a guest may still receive a guest-side wrapper that
            // calls drop on dispose, so the drop must be linkable. The
            // method bodies are wired to default-deny too in case a guest
            // somehow constructs a synthetic handle.
            .{ .name = "[resource-drop]incoming-datagram-stream", .call = &noopResourceDrop },
            .{ .name = "[method]incoming-datagram-stream.receive", .call = &socketDenyAccess },
            .{ .name = "[method]incoming-datagram-stream.subscribe", .call = &socketSubscribe },
            .{ .name = "[resource-drop]outgoing-datagram-stream", .call = &noopResourceDrop },
            .{ .name = "[method]outgoing-datagram-stream.check-send", .call = &socketDenyAccess },
            .{ .name = "[method]outgoing-datagram-stream.send", .call = &socketDenyAccess },
            .{ .name = "[method]outgoing-datagram-stream.subscribe", .call = &socketSubscribe },
        };
        for (members) |m| {
            try self.sockets_udp_iface.members.put(self.allocator, m.name, .{
                .func = .{ .context = self, .call = m.call },
            });
        }
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.sockets_udp_iface,
        });
    }

    /// Register `wasi:sockets/udp-create-socket` (#148).
    pub fn populateWasiSocketsUdpCreateSocket(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        try self.sockets_udp_create_iface.members.put(self.allocator, "create-udp-socket", .{
            .func = .{ .context = self, .call = &createUdpSocket },
        });
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.sockets_udp_create_iface,
        });
    }

    /// Register `wasi:sockets/ip-name-lookup` (#148). DNS resolution is
    /// stubbed to `error-code.name-unresolvable`; the
    /// `resolve-address-stream` resource is fully wired so a real
    /// implementation can drop in later without changing the contract.
    pub fn populateWasiSocketsIpNameLookup(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        try self.sockets_ip_name_lookup_iface.members.put(self.allocator, "resolve-addresses", .{
            .func = .{ .context = self, .call = &resolveAddresses },
        });
        try self.sockets_ip_name_lookup_iface.members.put(self.allocator, "[method]resolve-address-stream.resolve-next-address", .{
            .func = .{ .context = self, .call = &resolveNextAddress },
        });
        try self.sockets_ip_name_lookup_iface.members.put(self.allocator, "[method]resolve-address-stream.subscribe", .{
            .func = .{ .context = self, .call = &socketSubscribe },
        });
        try self.sockets_ip_name_lookup_iface.members.put(self.allocator, "[resource-drop]resolve-address-stream", .{
            .func = .{ .context = self, .call = &resolveStreamDrop },
        });
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.sockets_ip_name_lookup_iface,
        });
    }

    // ----- wasi:http (#149) -----

    /// Build a `result<X, http error-code>` err InterfaceValue.
    fn httpResultErr(allocator: Allocator, code: HttpErrorCode) !InterfaceValue {
        const payload = try allocator.create(InterfaceValue);
        payload.* = .{ .variant_val = .{ .discriminant = @intFromEnum(code), .payload = null } };
        return .{ .result_val = .{ .is_ok = false, .payload = payload } };
    }

    fn httpResultOk(allocator: Allocator, value: InterfaceValue) !InterfaceValue {
        const payload = try allocator.create(InterfaceValue);
        payload.* = value;
        return .{ .result_val = .{ .is_ok = true, .payload = payload } };
    }

    // Generic table push/lookup helpers for http resources. We reuse
    // the slot-reuse pattern from sockets so the table doesn't grow
    // unbounded as guests churn handles.
    fn pushHttpFields(self: *WasiCliAdapter, f: *HttpFields) !u32 {
        for (self.http_fields_table.items, 0..) |slot, i| {
            if (slot == null) {
                self.http_fields_table.items[i] = f;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.http_fields_table.items.len);
        try self.http_fields_table.append(self.allocator, f);
        return idx;
    }
    fn lookupHttpFields(self: *WasiCliAdapter, h: u32) ?*HttpFields {
        if (h >= self.http_fields_table.items.len) return null;
        return self.http_fields_table.items[h];
    }
    fn pushOutgoingRequest(self: *WasiCliAdapter, r: *OutgoingRequest) !u32 {
        for (self.http_outgoing_requests.items, 0..) |slot, i| {
            if (slot == null) {
                self.http_outgoing_requests.items[i] = r;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.http_outgoing_requests.items.len);
        try self.http_outgoing_requests.append(self.allocator, r);
        return idx;
    }
    fn lookupOutgoingRequest(self: *WasiCliAdapter, h: u32) ?*OutgoingRequest {
        if (h >= self.http_outgoing_requests.items.len) return null;
        return self.http_outgoing_requests.items[h];
    }
    fn pushOutgoingResponse(self: *WasiCliAdapter, r: *OutgoingResponse) !u32 {
        for (self.http_outgoing_responses.items, 0..) |slot, i| {
            if (slot == null) {
                self.http_outgoing_responses.items[i] = r;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.http_outgoing_responses.items.len);
        try self.http_outgoing_responses.append(self.allocator, r);
        return idx;
    }
    fn pushOutgoingBody(self: *WasiCliAdapter, b: *OutgoingBody) !u32 {
        for (self.http_outgoing_bodies.items, 0..) |slot, i| {
            if (slot == null) {
                self.http_outgoing_bodies.items[i] = b;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.http_outgoing_bodies.items.len);
        try self.http_outgoing_bodies.append(self.allocator, b);
        return idx;
    }
    fn pushFutureResponse(self: *WasiCliAdapter, f: *FutureIncomingResponse) !u32 {
        for (self.http_future_responses.items, 0..) |slot, i| {
            if (slot == null) {
                self.http_future_responses.items[i] = f;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.http_future_responses.items.len);
        try self.http_future_responses.append(self.allocator, f);
        return idx;
    }
    fn lookupFutureResponse(self: *WasiCliAdapter, h: u32) ?*FutureIncomingResponse {
        if (h >= self.http_future_responses.items.len) return null;
        return self.http_future_responses.items[h];
    }
    fn pushFutureTrailers(self: *WasiCliAdapter, f: *FutureTrailers) !u32 {
        for (self.http_future_trailers.items, 0..) |slot, i| {
            if (slot == null) {
                self.http_future_trailers.items[i] = f;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.http_future_trailers.items.len);
        try self.http_future_trailers.append(self.allocator, f);
        return idx;
    }
    fn pushRequestOptions(self: *WasiCliAdapter, r: *RequestOptions) !u32 {
        for (self.http_request_options.items, 0..) |slot, i| {
            if (slot == null) {
                self.http_request_options.items[i] = r;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.http_request_options.items.len);
        try self.http_request_options.append(self.allocator, r);
        return idx;
    }

    // --- fields ---

    /// `[constructor]fields() -> own<fields>`.
    fn httpFieldsConstructor(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const f = try self.allocator.create(HttpFields);
        f.* = .{};
        const h = try self.pushHttpFields(f);
        results[0] = .{ .handle = h };
    }

    /// `[static]fields.from-list(list<tuple<field-name, field-value>>)
    ///   -> result<own<fields>, header-error>`.
    ///
    /// The argument is a guest-memory list of name/value tuples;
    /// reading and copying each entry would require recursive lifting
    /// that the canonical-ABI layer doesn't yet hand to host fns in a
    /// uniform shape. The default-deny adapter therefore allocates an
    /// empty fields rep and returns ok — guests that rely on
    /// `from-list` populating headers should instead call
    /// `[constructor]fields` + `append` once that path is fully wired.
    /// TODO(#149 follow-up): copy entries when ABI exposes lifted
    /// list<tuple<string,string>> values.
    fn httpFieldsFromList(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const f = try self.allocator.create(HttpFields);
        f.* = .{};
        const h = try self.pushHttpFields(f);
        results[0] = try httpResultOk(allocator, .{ .handle = h });
    }

    /// `[method]fields.entries(borrow<fields>)
    ///   -> list<tuple<field-name, field-value>>`. Returns an empty
    /// list on the default-deny path; allocator-owned 0-len slice so
    /// the result deinit shape is correct.
    fn httpFieldsEntries(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        const empty = try allocator.alloc(InterfaceValue, 0);
        results[0] = .{ .list_val = empty };
    }

    /// `[method]fields.get(borrow<fields>, field-name)
    ///   -> list<field-value>`.
    fn httpFieldsGet(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        const empty = try allocator.alloc(InterfaceValue, 0);
        results[0] = .{ .list_val = empty };
    }

    /// `[method]fields.has(borrow<fields>, field-name) -> bool`.
    fn httpFieldsHas(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .bool = false };
    }

    /// `[method]fields.set/append/delete -> result<_, header-error>`.
    /// Default-deny accepts the operation as ok (no entries actually
    /// stored — see `httpFieldsFromList` rationale).
    fn httpFieldsSetlikeOk(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        // result<_, header-error> ok arm has no payload.
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `[method]fields.clone(borrow<fields>) -> own<fields>`.
    fn httpFieldsClone(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const f = try self.allocator.create(HttpFields);
        f.* = .{};
        const h = try self.pushHttpFields(f);
        results[0] = .{ .handle = h };
    }

    /// `[resource-drop]fields`.
    fn httpFieldsDrop(
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
        if (handle >= self.http_fields_table.items.len) return;
        if (self.http_fields_table.items[handle]) |f| {
            f.deinit(self.allocator);
            self.allocator.destroy(f);
            self.http_fields_table.items[handle] = null;
        }
    }

    // --- outgoing-request ---

    /// `[constructor]outgoing-request(headers: own<fields>)
    ///   -> outgoing-request`.
    fn httpOutgoingRequestConstructor(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const headers_handle = switch (args[0]) {
            .handle => |h| h,
            else => 0,
        };
        const r = try self.allocator.create(OutgoingRequest);
        r.* = .{ .headers_handle = headers_handle };
        const h = try self.pushOutgoingRequest(r);
        results[0] = .{ .handle = h };
    }

    /// `[method]outgoing-request.method(borrow) -> method`. Returns a
    /// variant value; the variant has 9 cases, the last (`other`)
    /// carries a string payload. Default rep is `.get` (no payload).
    fn httpOutgoingRequestMethod(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const r = self.lookupOutgoingRequest(handle) orelse {
            results[0] = .{ .variant_val = .{ .discriminant = 0, .payload = null } };
            return;
        };
        results[0] = .{ .variant_val = .{ .discriminant = r.method_disc, .payload = null } };
    }

    /// `[method]outgoing-request.set-method(borrow, method)
    ///   -> result<_, _>`.
    fn httpOutgoingRequestSetMethod(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// Generic `(borrow) -> option<string>` getter that always returns
    /// `none` on the default-deny path.
    fn httpReturnOptionNone(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
    }

    /// Generic `(borrow, ...) -> result<_, _>` setter that returns ok.
    fn httpReturnResultOkUnit(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `[method]outgoing-request.headers(borrow) -> own<fields>`.
    /// The WIT contract is that this returns a mutable child handle
    /// that aliases the request's headers; we hand back the stored
    /// handle. Real implementations would clone or refcount; the
    /// default-deny adapter trusts the guest not to drop the alias
    /// before the request itself.
    fn httpOutgoingRequestHeaders(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const r = self.lookupOutgoingRequest(handle) orelse {
            results[0] = .{ .handle = 0 };
            return;
        };
        results[0] = .{ .handle = r.headers_handle };
    }

    /// `[method]outgoing-request.body(borrow)
    ///   -> result<own<outgoing-body>, _>`. First call allocates a
    /// body slot; second call errs.
    fn httpOutgoingRequestBody(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const r = self.lookupOutgoingRequest(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        };
        if (r.body_consumed) {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        }
        r.body_consumed = true;
        const body = try self.allocator.create(OutgoingBody);
        body.* = .{};
        const bh = try self.pushOutgoingBody(body);
        results[0] = try httpResultOk(allocator, .{ .handle = bh });
    }

    /// `[resource-drop]outgoing-request`.
    fn httpOutgoingRequestDrop(
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
        if (handle >= self.http_outgoing_requests.items.len) return;
        if (self.http_outgoing_requests.items[handle]) |r| {
            r.deinit(self.allocator);
            self.allocator.destroy(r);
            self.http_outgoing_requests.items[handle] = null;
        }
    }

    // --- outgoing-response ---

    /// `[constructor]outgoing-response(headers: own<fields>)
    ///   -> outgoing-response`.
    fn httpOutgoingResponseConstructor(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const headers_handle = switch (args[0]) {
            .handle => |h| h,
            else => 0,
        };
        const r = try self.allocator.create(OutgoingResponse);
        r.* = .{ .headers_handle = headers_handle };
        const h = try self.pushOutgoingResponse(r);
        results[0] = .{ .handle = h };
    }

    /// `[method]outgoing-response.status-code(borrow) -> status-code`
    /// (status-code is an alias for u16).
    fn httpOutgoingResponseStatusCode(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        if (handle >= self.http_outgoing_responses.items.len or
            self.http_outgoing_responses.items[handle] == null)
        {
            results[0] = .{ .u16 = 0 };
            return;
        }
        const r = self.http_outgoing_responses.items[handle].?;
        results[0] = .{ .u16 = r.status };
    }

    /// `[method]outgoing-response.set-status-code(borrow, status-code)
    ///   -> result<_, _>`.
    fn httpOutgoingResponseSetStatusCode(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
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
        const status: u16 = switch (args[1]) {
            .u16 => |v| v,
            .u32 => |v| @intCast(v & 0xFFFF),
            else => 0,
        };
        if (handle < self.http_outgoing_responses.items.len) {
            if (self.http_outgoing_responses.items[handle]) |r| r.status = status;
        }
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `[method]outgoing-response.headers(borrow) -> own<fields>`.
    fn httpOutgoingResponseHeaders(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        if (handle >= self.http_outgoing_responses.items.len or
            self.http_outgoing_responses.items[handle] == null)
        {
            results[0] = .{ .handle = 0 };
            return;
        }
        results[0] = .{ .handle = self.http_outgoing_responses.items[handle].?.headers_handle };
    }

    /// `[method]outgoing-response.body(borrow)
    ///   -> result<own<outgoing-body>, _>`.
    fn httpOutgoingResponseBody(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        if (handle >= self.http_outgoing_responses.items.len or
            self.http_outgoing_responses.items[handle] == null)
        {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        }
        const r = self.http_outgoing_responses.items[handle].?;
        if (r.body_consumed) {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        }
        r.body_consumed = true;
        const body = try self.allocator.create(OutgoingBody);
        body.* = .{};
        const bh = try self.pushOutgoingBody(body);
        results[0] = try httpResultOk(allocator, .{ .handle = bh });
    }

    /// `[resource-drop]outgoing-response`.
    fn httpOutgoingResponseDrop(
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
        if (handle >= self.http_outgoing_responses.items.len) return;
        if (self.http_outgoing_responses.items[handle]) |r| {
            self.allocator.destroy(r);
            self.http_outgoing_responses.items[handle] = null;
        }
    }

    // --- incoming-request / incoming-response (stubs) ---

    /// `[method]incoming-request.method/path-with-query/scheme/authority
    ///   /headers/consume`. The default-deny adapter never produces an
    /// incoming-request handle, so any call hits a missing slot and we
    /// return safe defaults.
    fn httpIncomingRequestMethod(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .variant_val = .{ .discriminant = 0, .payload = null } };
    }

    fn httpIncomingRequestHeaders(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        // Mint an empty fields slot so the borrow points somewhere live.
        const f = try self.allocator.create(HttpFields);
        f.* = .{};
        const h = try self.pushHttpFields(f);
        results[0] = .{ .handle = h };
    }

    fn httpIncomingRequestConsume(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
    }

    fn httpIncomingRequestDrop(
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
        if (handle >= self.http_incoming_requests.items.len) return;
        if (self.http_incoming_requests.items[handle]) |r| {
            self.allocator.destroy(r);
            self.http_incoming_requests.items[handle] = null;
        }
    }

    /// `[method]incoming-response.status(borrow) -> status-code`.
    fn httpIncomingResponseStatus(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        if (handle >= self.http_incoming_responses.items.len or
            self.http_incoming_responses.items[handle] == null)
        {
            results[0] = .{ .u16 = 0 };
            return;
        }
        results[0] = .{ .u16 = self.http_incoming_responses.items[handle].?.status };
    }

    fn httpIncomingResponseHeaders(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        if (handle >= self.http_incoming_responses.items.len or
            self.http_incoming_responses.items[handle] == null)
        {
            // Mint an empty fields slot to keep the call total.
            const f = try self.allocator.create(HttpFields);
            f.* = .{};
            const fh = try self.pushHttpFields(f);
            results[0] = .{ .handle = fh };
            return;
        }
        results[0] = .{ .handle = self.http_incoming_responses.items[handle].?.headers_handle };
    }

    fn httpIncomingResponseConsume(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
    }

    fn httpIncomingResponseDrop(
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
        if (handle >= self.http_incoming_responses.items.len) return;
        if (self.http_incoming_responses.items[handle]) |r| {
            self.allocator.destroy(r);
            self.http_incoming_responses.items[handle] = null;
        }
    }

    // --- bodies ---

    /// `[method]incoming-body.stream(borrow)
    ///   -> result<own<input-stream>, _>`. Default-deny errs because
    /// no real wire-stream exists yet.
    fn httpIncomingBodyStream(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
    }

    /// `[static]incoming-body.finish(own<incoming-body>)
    ///   -> own<future-trailers>`.
    fn httpIncomingBodyFinish(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const ft = try self.allocator.create(FutureTrailers);
        ft.* = .{};
        const h = try self.pushFutureTrailers(ft);
        results[0] = .{ .handle = h };
    }

    fn httpIncomingBodyDrop(
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
        if (handle >= self.http_incoming_bodies.items.len) return;
        if (self.http_incoming_bodies.items[handle]) |b| {
            self.allocator.destroy(b);
            self.http_incoming_bodies.items[handle] = null;
        }
    }

    /// `[method]outgoing-body.write(borrow)
    ///   -> result<own<output-stream>, _>`. Default-deny errs.
    fn httpOutgoingBodyWrite(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
    }

    /// `[static]outgoing-body.finish(own<outgoing-body>,
    ///   option<own<trailers>>) -> result<_, error-code>`.
    fn httpOutgoingBodyFinish(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        if (results.len == 0) return error.InvalidArgs;
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    fn httpOutgoingBodyDrop(
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
        if (handle >= self.http_outgoing_bodies.items.len) return;
        if (self.http_outgoing_bodies.items[handle]) |b| {
            self.allocator.destroy(b);
            self.http_outgoing_bodies.items[handle] = null;
        }
    }

    // --- futures ---

    /// `[method]future-incoming-response.subscribe(borrow)
    ///   -> own<pollable>`. Stub always-ready pollable, same handle
    /// minting as the sockets adapter.
    fn httpFutureSubscribe(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const h = self.next_pollable_handle;
        self.next_pollable_handle += 1;
        results[0] = .{ .handle = h };
    }

    /// `[method]future-incoming-response.get(borrow)
    ///   -> option<result<result<own<incoming-response>, error-code>,
    ///                    _>>`.
    ///
    /// Decoding the triple-nest:
    /// - outer `option`: `none` while pending; `some(...)` once ready.
    /// - middle `result<_, _>` outer arm: `err(())` if `.get` was
    ///   already polled (per WIT — a future is single-shot).
    /// - inner `result<incoming-response, error-code>`: actual outcome.
    ///
    /// Default-deny path: ready_err — `some(ok(err(HTTP_request_denied)))`.
    fn httpFutureGet(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const fut = self.lookupFutureResponse(handle) orelse {
            results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
            return;
        };
        if (fut.polled) {
            // some(err(())) — already consumed.
            const inner = try allocator.create(InterfaceValue);
            inner.* = .{ .result_val = .{ .is_ok = false, .payload = null } };
            results[0] = .{ .option_val = .{ .is_some = true, .payload = inner } };
            return;
        }
        switch (fut.state) {
            .pending => {
                results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
                return;
            },
            .ready_ok => |ir_handle| {
                fut.polled = true;
                const inner_ok = try allocator.create(InterfaceValue);
                inner_ok.* = .{ .handle = ir_handle };
                const middle = try allocator.create(InterfaceValue);
                middle.* = .{ .result_val = .{ .is_ok = true, .payload = inner_ok } };
                const outer = try allocator.create(InterfaceValue);
                outer.* = .{ .result_val = .{ .is_ok = true, .payload = middle } };
                results[0] = .{ .option_val = .{ .is_some = true, .payload = outer } };
            },
            .ready_err => |err_disc| {
                fut.polled = true;
                const err_payload = try allocator.create(InterfaceValue);
                err_payload.* = .{ .variant_val = .{ .discriminant = err_disc, .payload = null } };
                const middle = try allocator.create(InterfaceValue);
                middle.* = .{ .result_val = .{ .is_ok = false, .payload = err_payload } };
                const outer = try allocator.create(InterfaceValue);
                outer.* = .{ .result_val = .{ .is_ok = true, .payload = middle } };
                results[0] = .{ .option_val = .{ .is_some = true, .payload = outer } };
            },
        }
    }

    fn httpFutureDrop(
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
        if (handle >= self.http_future_responses.items.len) return;
        if (self.http_future_responses.items[handle]) |f| {
            self.allocator.destroy(f);
            self.http_future_responses.items[handle] = null;
        }
    }

    /// `[method]future-trailers.get(borrow)
    ///   -> option<result<result<option<own<trailers>>, error-code>, _>>`.
    /// Default-deny: `some(ok(ok(none)))` (no trailers).
    fn httpFutureTrailersGet(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        if (handle >= self.http_future_trailers.items.len or
            self.http_future_trailers.items[handle] == null)
        {
            results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
            return;
        }
        const ft = self.http_future_trailers.items[handle].?;
        if (ft.polled) {
            const inner = try allocator.create(InterfaceValue);
            inner.* = .{ .result_val = .{ .is_ok = false, .payload = null } };
            results[0] = .{ .option_val = .{ .is_some = true, .payload = inner } };
            return;
        }
        ft.polled = true;
        // inner: option<own<trailers>> = none
        const inner_opt = try allocator.create(InterfaceValue);
        inner_opt.* = .{ .option_val = .{ .is_some = false, .payload = null } };
        // middle: result<option<own<trailers>>, error-code> = ok(none)
        const middle = try allocator.create(InterfaceValue);
        middle.* = .{ .result_val = .{ .is_ok = true, .payload = inner_opt } };
        // outer: result<middle, _> = ok(middle)
        const outer = try allocator.create(InterfaceValue);
        outer.* = .{ .result_val = .{ .is_ok = true, .payload = middle } };
        results[0] = .{ .option_val = .{ .is_some = true, .payload = outer } };
    }

    fn httpFutureTrailersDrop(
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
        if (handle >= self.http_future_trailers.items.len) return;
        if (self.http_future_trailers.items[handle]) |f| {
            self.allocator.destroy(f);
            self.http_future_trailers.items[handle] = null;
        }
    }

    // --- request-options / response-outparam ---

    /// `[constructor]request-options() -> request-options`.
    fn httpRequestOptionsConstructor(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const r = try self.allocator.create(RequestOptions);
        r.* = .{};
        const h = try self.pushRequestOptions(r);
        results[0] = .{ .handle = h };
    }

    fn httpRequestOptionsDrop(
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
        if (handle >= self.http_request_options.items.len) return;
        if (self.http_request_options.items[handle]) |r| {
            self.allocator.destroy(r);
            self.http_request_options.items[handle] = null;
        }
    }

    /// `[static]response-outparam.set(own<response-outparam>,
    ///   result<own<outgoing-response>, error-code>)`. No-op: the
    /// host never inspects the outparam on the default-deny path.
    fn httpResponseOutparamSet(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        _: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {}

    fn httpResponseOutparamDrop(
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
        if (handle >= self.http_response_outparams.items.len) return;
        if (self.http_response_outparams.items[handle]) |r| {
            self.allocator.destroy(r);
            self.http_response_outparams.items[handle] = null;
        }
    }

    // --- outgoing-handler ---

    /// `wasi:http/outgoing-handler.handle(own<outgoing-request>,
    ///   option<own<request-options>>)
    ///   -> result<own<future-incoming-response>, error-code>`.
    ///
    /// Default-deny: allocate a future-incoming-response in
    /// `ready_err(HTTP_request_denied)` and return ok wrapping the
    /// handle. The guest gets back a well-typed future that, when
    /// polled, yields the denial error — so neither the synchronous
    /// outer nor the eventual inner nesting traps.
    ///
    /// TODO(#149 follow-up): replace with a real `std.http.Client`
    /// integration gated by an outbound-allow capability list.
    /// Until then we never open a host socket, never speak TLS, and
    /// never dispatch a request; the deny path keeps wasi:http
    /// usable for components that gracefully handle denial.
    fn httpOutgoingHandlerHandle(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const fut = try self.allocator.create(FutureIncomingResponse);
        fut.* = .{ .state = .{ .ready_err = @intFromEnum(HttpErrorCode.HTTP_request_denied) } };
        const h = try self.pushFutureResponse(fut);
        results[0] = try httpResultOk(allocator, .{ .handle = h });
    }

    // --- incoming-handler ---

    /// `wasi:http/incoming-handler.handle(own<incoming-request>,
    ///   own<response-outparam>)`. Servers normally export this; the
    /// host providing it as an import is unusual but legal. No-op.
    fn httpIncomingHandlerHandle(
        _: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        _: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {}

    /// Register `wasi:http/types` (#149).
    pub fn populateWasiHttpTypes(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        const M = struct { name: []const u8, call: *const fn (?*anyopaque, *ComponentInstance, []const InterfaceValue, []InterfaceValue, Allocator) anyerror!void };
        const members = [_]M{
            // fields
            .{ .name = "[constructor]fields", .call = &httpFieldsConstructor },
            .{ .name = "[static]fields.from-list", .call = &httpFieldsFromList },
            .{ .name = "[method]fields.entries", .call = &httpFieldsEntries },
            .{ .name = "[method]fields.get", .call = &httpFieldsGet },
            .{ .name = "[method]fields.has", .call = &httpFieldsHas },
            .{ .name = "[method]fields.set", .call = &httpFieldsSetlikeOk },
            .{ .name = "[method]fields.append", .call = &httpFieldsSetlikeOk },
            .{ .name = "[method]fields.delete", .call = &httpFieldsSetlikeOk },
            .{ .name = "[method]fields.clone", .call = &httpFieldsClone },
            .{ .name = "[resource-drop]fields", .call = &httpFieldsDrop },
            // outgoing-request
            .{ .name = "[constructor]outgoing-request", .call = &httpOutgoingRequestConstructor },
            .{ .name = "[method]outgoing-request.method", .call = &httpOutgoingRequestMethod },
            .{ .name = "[method]outgoing-request.set-method", .call = &httpOutgoingRequestSetMethod },
            .{ .name = "[method]outgoing-request.path-with-query", .call = &httpReturnOptionNone },
            .{ .name = "[method]outgoing-request.set-path-with-query", .call = &httpReturnResultOkUnit },
            .{ .name = "[method]outgoing-request.scheme", .call = &httpReturnOptionNone },
            .{ .name = "[method]outgoing-request.set-scheme", .call = &httpReturnResultOkUnit },
            .{ .name = "[method]outgoing-request.authority", .call = &httpReturnOptionNone },
            .{ .name = "[method]outgoing-request.set-authority", .call = &httpReturnResultOkUnit },
            .{ .name = "[method]outgoing-request.headers", .call = &httpOutgoingRequestHeaders },
            .{ .name = "[method]outgoing-request.body", .call = &httpOutgoingRequestBody },
            .{ .name = "[resource-drop]outgoing-request", .call = &httpOutgoingRequestDrop },
            // outgoing-response
            .{ .name = "[constructor]outgoing-response", .call = &httpOutgoingResponseConstructor },
            .{ .name = "[method]outgoing-response.status-code", .call = &httpOutgoingResponseStatusCode },
            .{ .name = "[method]outgoing-response.set-status-code", .call = &httpOutgoingResponseSetStatusCode },
            .{ .name = "[method]outgoing-response.headers", .call = &httpOutgoingResponseHeaders },
            .{ .name = "[method]outgoing-response.body", .call = &httpOutgoingResponseBody },
            .{ .name = "[resource-drop]outgoing-response", .call = &httpOutgoingResponseDrop },
            // incoming-request
            .{ .name = "[method]incoming-request.method", .call = &httpIncomingRequestMethod },
            .{ .name = "[method]incoming-request.path-with-query", .call = &httpReturnOptionNone },
            .{ .name = "[method]incoming-request.scheme", .call = &httpReturnOptionNone },
            .{ .name = "[method]incoming-request.authority", .call = &httpReturnOptionNone },
            .{ .name = "[method]incoming-request.headers", .call = &httpIncomingRequestHeaders },
            .{ .name = "[method]incoming-request.consume", .call = &httpIncomingRequestConsume },
            .{ .name = "[resource-drop]incoming-request", .call = &httpIncomingRequestDrop },
            // incoming-response
            .{ .name = "[method]incoming-response.status", .call = &httpIncomingResponseStatus },
            .{ .name = "[method]incoming-response.headers", .call = &httpIncomingResponseHeaders },
            .{ .name = "[method]incoming-response.consume", .call = &httpIncomingResponseConsume },
            .{ .name = "[resource-drop]incoming-response", .call = &httpIncomingResponseDrop },
            // bodies
            .{ .name = "[method]incoming-body.stream", .call = &httpIncomingBodyStream },
            .{ .name = "[static]incoming-body.finish", .call = &httpIncomingBodyFinish },
            .{ .name = "[resource-drop]incoming-body", .call = &httpIncomingBodyDrop },
            .{ .name = "[method]outgoing-body.write", .call = &httpOutgoingBodyWrite },
            .{ .name = "[static]outgoing-body.finish", .call = &httpOutgoingBodyFinish },
            .{ .name = "[resource-drop]outgoing-body", .call = &httpOutgoingBodyDrop },
            // futures
            .{ .name = "[method]future-incoming-response.subscribe", .call = &httpFutureSubscribe },
            .{ .name = "[method]future-incoming-response.get", .call = &httpFutureGet },
            .{ .name = "[resource-drop]future-incoming-response", .call = &httpFutureDrop },
            .{ .name = "[method]future-trailers.subscribe", .call = &httpFutureSubscribe },
            .{ .name = "[method]future-trailers.get", .call = &httpFutureTrailersGet },
            .{ .name = "[resource-drop]future-trailers", .call = &httpFutureTrailersDrop },
            // request-options / response-outparam
            .{ .name = "[constructor]request-options", .call = &httpRequestOptionsConstructor },
            .{ .name = "[resource-drop]request-options", .call = &httpRequestOptionsDrop },
            .{ .name = "[static]response-outparam.set", .call = &httpResponseOutparamSet },
            .{ .name = "[resource-drop]response-outparam", .call = &httpResponseOutparamDrop },
        };
        for (members) |m| {
            try self.http_types_iface.members.put(self.allocator, m.name, .{
                .func = .{ .context = self, .call = m.call },
            });
        }
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.http_types_iface,
        });
    }

    /// Register `wasi:http/outgoing-handler` (#149). Default-deny:
    /// `handle` resolves the future to `HTTP-request-denied`.
    pub fn populateWasiHttpOutgoingHandler(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        try self.http_outgoing_handler_iface.members.put(self.allocator, "handle", .{
            .func = .{ .context = self, .call = &httpOutgoingHandlerHandle },
        });
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.http_outgoing_handler_iface,
        });
    }

    /// Register `wasi:http/incoming-handler` (#149). The host
    /// implementation is a no-op — components are expected to *export*
    /// this interface (they are HTTP servers); a host-side import
    /// stub exists only so that components which incidentally import
    /// it (e.g. for adapter-style composition) link cleanly.
    pub fn populateWasiHttpIncomingHandler(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        try self.http_incoming_handler_iface.members.put(self.allocator, "handle", .{
            .func = .{ .context = self, .call = &httpIncomingHandlerHandle },
        });
        try providers.put(self.allocator, interface_name, .{
            .host_instance = &self.http_incoming_handler_iface,
        });
    }
};

/// `wasi:clocks/wall-clock.datetime`: a record of `(seconds: u64, nanoseconds: u32)`.
pub const Datetime = struct {
    seconds: u64,
    nanoseconds: u32,
};

/// Read a POSIX clock and lift it into a `Datetime`. On failure (or where
/// `clock_gettime` is unavailable, e.g. Windows / freestanding), returns
/// the zero datetime — sufficient for the spec contract on those targets
/// (host clocks are never observed by the unit tests, which use the
/// adapter's deterministic-clock injection paths). Both fields are
/// clamped to valid ranges (`seconds >= 0`, `nanoseconds < 1e9`).
fn readClockDatetime(clock_id: ClockId) Datetime {
    if (!have_posix_clock) return .{ .seconds = 0, .nanoseconds = 0 };
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(clock_id, &ts);
    if (std.posix.errno(rc) != .SUCCESS) return .{ .seconds = 0, .nanoseconds = 0 };
    const secs: u64 = if (ts.sec < 0) 0 else @intCast(ts.sec);
    const nsec_raw = ts.nsec;
    const nsec: u32 = if (nsec_raw < 0)
        0
    else if (nsec_raw >= 1_000_000_000)
        999_999_999
    else
        @intCast(nsec_raw);
    return .{ .seconds = secs, .nanoseconds = nsec };
}

/// Read a POSIX clock and flatten to nanoseconds since its epoch. Used for
/// `monotonic-clock.now`, where preview-2's `instant` is a flat `u64`.
fn readClockNs(clock_id: ClockId) u64 {
    const dt = readClockDatetime(clock_id);
    return dt.seconds *% 1_000_000_000 +% @as(u64, dt.nanoseconds);
}

/// Cross-platform alias for the host clock-id type. On POSIX targets it
/// is `std.posix.clockid_t`; on Windows `std.posix.clockid_t` is `void`
/// and `clock_gettime` is unusable, so we substitute a plain enum so
/// `.REALTIME` / `.MONOTONIC` enum literals at the call sites still
/// compile. `readClockDatetime` short-circuits before consulting the
/// value on those targets, so the substitute never reaches the syscall.
const have_posix_clock = @import("builtin").os.tag != .windows and
    @hasDecl(std.posix.system, "clock_gettime");
const ClockId = if (have_posix_clock)
    std.posix.clockid_t
else
    enum { REALTIME, MONOTONIC };

/// Allocate an `InterfaceValue` carrying a `record { seconds: u64, nanoseconds: u32 }`
/// shape. Caller (the trampoline) `deinit`s the result via the same allocator.
fn buildDatetimeRecord(allocator: Allocator, dt: Datetime) !InterfaceValue {
    const fields = try allocator.alloc(InterfaceValue, 2);
    fields[0] = .{ .u64 = dt.seconds };
    fields[1] = .{ .u32 = dt.nanoseconds };
    return .{ .record_val = fields };
}

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
    var matched_wall_clock: ?[]const u8 = null;
    var matched_monotonic_clock: ?[]const u8 = null;
    var matched_random: ?[]const u8 = null;
    var matched_random_insecure: ?[]const u8 = null;
    var matched_random_insecure_seed: ?[]const u8 = null;
    var matched_fs_types: ?[]const u8 = null;
    var matched_fs_preopens: ?[]const u8 = null;
    var matched_sockets_network: ?[]const u8 = null;
    var matched_sockets_instance_network: ?[]const u8 = null;
    var matched_sockets_tcp: ?[]const u8 = null;
    var matched_sockets_tcp_create: ?[]const u8 = null;
    var matched_sockets_udp: ?[]const u8 = null;
    var matched_sockets_udp_create: ?[]const u8 = null;
    var matched_sockets_ip_name_lookup: ?[]const u8 = null;
    var matched_http_types: ?[]const u8 = null;
    var matched_http_outgoing_handler: ?[]const u8 = null;
    var matched_http_incoming_handler: ?[]const u8 = null;
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
        if (matched_wall_clock == null and matchesWasiPrefix(imp.name, "wasi:clocks/wall-clock"))
            matched_wall_clock = imp.name;
        if (matched_monotonic_clock == null and matchesWasiPrefix(imp.name, "wasi:clocks/monotonic-clock"))
            matched_monotonic_clock = imp.name;
        // `wasi:random/insecure-seed` shares the `wasi:random/insecure`
        // prefix, so test the more-specific name first.
        if (matched_random_insecure_seed == null and matchesWasiPrefix(imp.name, "wasi:random/insecure-seed"))
            matched_random_insecure_seed = imp.name;
        if (matched_random_insecure == null and
            !(matched_random_insecure_seed != null and
                std.mem.eql(u8, matched_random_insecure_seed.?, imp.name)) and
            matchesWasiPrefix(imp.name, "wasi:random/insecure"))
            matched_random_insecure = imp.name;
        if (matched_random == null and matchesWasiPrefix(imp.name, "wasi:random/random"))
            matched_random = imp.name;
        if (matched_fs_types == null and matchesWasiPrefix(imp.name, "wasi:filesystem/types"))
            matched_fs_types = imp.name;
        if (matched_fs_preopens == null and matchesWasiPrefix(imp.name, "wasi:filesystem/preopens"))
            matched_fs_preopens = imp.name;
        // wasi:sockets/* (#148). `instance-network` shares the
        // `wasi:sockets/` prefix with `network`, so probe more-specific
        // names first to avoid an aliasing match.
        if (matched_sockets_instance_network == null and matchesWasiPrefix(imp.name, "wasi:sockets/instance-network"))
            matched_sockets_instance_network = imp.name;
        if (matched_sockets_tcp_create == null and matchesWasiPrefix(imp.name, "wasi:sockets/tcp-create-socket"))
            matched_sockets_tcp_create = imp.name;
        if (matched_sockets_udp_create == null and matchesWasiPrefix(imp.name, "wasi:sockets/udp-create-socket"))
            matched_sockets_udp_create = imp.name;
        if (matched_sockets_ip_name_lookup == null and matchesWasiPrefix(imp.name, "wasi:sockets/ip-name-lookup"))
            matched_sockets_ip_name_lookup = imp.name;
        if (matched_sockets_tcp == null and matchesWasiPrefix(imp.name, "wasi:sockets/tcp"))
            matched_sockets_tcp = imp.name;
        if (matched_sockets_udp == null and matchesWasiPrefix(imp.name, "wasi:sockets/udp"))
            matched_sockets_udp = imp.name;
        if (matched_sockets_network == null and matchesWasiPrefix(imp.name, "wasi:sockets/network"))
            matched_sockets_network = imp.name;
        // wasi:http/* (#149). `outgoing-handler` and `incoming-handler`
        // share the `wasi:http/` prefix with `types`, but `matchesWasiPrefix`
        // matches at interface-name granularity so order doesn't matter.
        if (matched_http_outgoing_handler == null and matchesWasiPrefix(imp.name, "wasi:http/outgoing-handler"))
            matched_http_outgoing_handler = imp.name;
        if (matched_http_incoming_handler == null and matchesWasiPrefix(imp.name, "wasi:http/incoming-handler"))
            matched_http_incoming_handler = imp.name;
        if (matched_http_types == null and matchesWasiPrefix(imp.name, "wasi:http/types"))
            matched_http_types = imp.name;
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

    try adapter.populateWasiClocksWallClock(
        providers,
        matched_wall_clock orelse "wasi:clocks/wall-clock",
    );
    if (matched_wall_clock == null) _ = providers.remove("wasi:clocks/wall-clock");

    try adapter.populateWasiClocksMonotonicClock(
        providers,
        matched_monotonic_clock orelse "wasi:clocks/monotonic-clock",
    );
    if (matched_monotonic_clock == null) _ = providers.remove("wasi:clocks/monotonic-clock");

    try adapter.populateWasiRandomRandom(
        providers,
        matched_random orelse "wasi:random/random",
    );
    if (matched_random == null) _ = providers.remove("wasi:random/random");

    try adapter.populateWasiRandomInsecure(
        providers,
        matched_random_insecure orelse "wasi:random/insecure",
    );
    if (matched_random_insecure == null) _ = providers.remove("wasi:random/insecure");

    try adapter.populateWasiRandomInsecureSeed(
        providers,
        matched_random_insecure_seed orelse "wasi:random/insecure-seed",
    );
    if (matched_random_insecure_seed == null) _ = providers.remove("wasi:random/insecure-seed");

    try adapter.populateWasiFilesystemTypes(
        providers,
        matched_fs_types orelse "wasi:filesystem/types",
    );
    if (matched_fs_types == null) _ = providers.remove("wasi:filesystem/types");

    try adapter.populateWasiFilesystemPreopens(
        providers,
        matched_fs_preopens orelse "wasi:filesystem/preopens",
    );
    if (matched_fs_preopens == null) _ = providers.remove("wasi:filesystem/preopens");

    try adapter.populateWasiSocketsNetwork(
        providers,
        matched_sockets_network orelse "wasi:sockets/network",
    );
    if (matched_sockets_network == null) _ = providers.remove("wasi:sockets/network");

    try adapter.populateWasiSocketsInstanceNetwork(
        providers,
        matched_sockets_instance_network orelse "wasi:sockets/instance-network",
    );
    if (matched_sockets_instance_network == null) _ = providers.remove("wasi:sockets/instance-network");

    try adapter.populateWasiSocketsTcp(
        providers,
        matched_sockets_tcp orelse "wasi:sockets/tcp",
    );
    if (matched_sockets_tcp == null) _ = providers.remove("wasi:sockets/tcp");

    try adapter.populateWasiSocketsTcpCreateSocket(
        providers,
        matched_sockets_tcp_create orelse "wasi:sockets/tcp-create-socket",
    );
    if (matched_sockets_tcp_create == null) _ = providers.remove("wasi:sockets/tcp-create-socket");

    try adapter.populateWasiSocketsUdp(
        providers,
        matched_sockets_udp orelse "wasi:sockets/udp",
    );
    if (matched_sockets_udp == null) _ = providers.remove("wasi:sockets/udp");

    try adapter.populateWasiSocketsUdpCreateSocket(
        providers,
        matched_sockets_udp_create orelse "wasi:sockets/udp-create-socket",
    );
    if (matched_sockets_udp_create == null) _ = providers.remove("wasi:sockets/udp-create-socket");

    try adapter.populateWasiSocketsIpNameLookup(
        providers,
        matched_sockets_ip_name_lookup orelse "wasi:sockets/ip-name-lookup",
    );
    if (matched_sockets_ip_name_lookup == null) _ = providers.remove("wasi:sockets/ip-name-lookup");

    try adapter.populateWasiHttpTypes(
        providers,
        matched_http_types orelse "wasi:http/types",
    );
    if (matched_http_types == null) _ = providers.remove("wasi:http/types");

    try adapter.populateWasiHttpOutgoingHandler(
        providers,
        matched_http_outgoing_handler orelse "wasi:http/outgoing-handler",
    );
    if (matched_http_outgoing_handler == null) _ = providers.remove("wasi:http/outgoing-handler");

    try adapter.populateWasiHttpIncomingHandler(
        providers,
        matched_http_incoming_handler orelse "wasi:http/incoming-handler",
    );
    if (matched_http_incoming_handler == null) _ = providers.remove("wasi:http/incoming-handler");
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
    // The provider populators allocate hashmap entries via `self.allocator`
    // (the adapter's allocator) — keep the deinit consistent with that so
    // hand-rolled callers like `runComponentBytes` (which pass an arena)
    // don't leak the underlying hashmap storage.
    defer providers.deinit(adapter.allocator);

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
    const testing = std.testing;
    const data = @embedFile("fixtures/stdio-echo.wasm");

    // The loader has no `Component.deinit` yet (#142 Phase 1B); use an
    // arena so the test doesn't leak. Mirrors the pattern used by the
    // other end-to-end component tests in this codebase.
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    adapter.setStdinBytes("hello\n");

    const outcome = runComponentBytes(data, arena_alloc, &adapter) catch |err| {
        std.debug.print("stdio-echo run failed: {s}\n", .{@errorName(err)});
        std.debug.print("stdout so far: {s}\n", .{adapter.getStdoutBytes()});
        std.debug.print("stderr so far: {s}\n", .{adapter.getStderrBytes()});
        return err;
    };

    try testing.expect(outcome.is_ok);
    try testing.expectEqualStrings("echo: hello\n", adapter.getStdoutBytes());
}

test "populateWasiProviders: binds wasi:clocks/wall-clock + monotonic-clock (#146)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const imports = [_]ctypes_root.ImportDecl{
        .{ .name = "wasi:clocks/wall-clock@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:clocks/monotonic-clock@0.2.6", .desc = .{ .instance = 0 } },
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

    try testing.expect(providers.contains("wasi:clocks/wall-clock@0.2.6"));
    try testing.expect(providers.contains("wasi:clocks/monotonic-clock@0.2.6"));
    // Bare names must NOT be registered when the component imports the versioned form.
    try testing.expect(!providers.contains("wasi:clocks/wall-clock"));
    try testing.expect(!providers.contains("wasi:clocks/monotonic-clock"));

    try testing.expect(adapter.clocks_wall_iface.members.contains("now"));
    try testing.expect(adapter.clocks_wall_iface.members.contains("resolution"));
    try testing.expect(adapter.clocks_monotonic_iface.members.contains("now"));
    try testing.expect(adapter.clocks_monotonic_iface.members.contains("resolution"));
    try testing.expect(adapter.clocks_monotonic_iface.members.contains("subscribe-instant"));
    try testing.expect(adapter.clocks_monotonic_iface.members.contains("subscribe-duration"));
}

test "wasi:clocks/wall-clock.now lifts injected datetime through record_val (#146)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    adapter.wall_clock_override = .{ .seconds = 1_700_000_000, .nanoseconds = 123_456_789 };

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.wallClockNow(&adapter, &ci, &.{}, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .record_val);
    try testing.expectEqual(@as(usize, 2), results[0].record_val.len);
    try testing.expectEqual(@as(u64, 1_700_000_000), results[0].record_val[0].u64);
    try testing.expectEqual(@as(u32, 123_456_789), results[0].record_val[1].u32);
}

test "wasi:clocks/wall-clock.resolution returns 1ns datetime (#146)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.wallClockResolution(null, &ci, &.{}, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .record_val);
    try testing.expectEqual(@as(u64, 0), results[0].record_val[0].u64);
    try testing.expectEqual(@as(u32, 1), results[0].record_val[1].u32);
}

test "wasi:clocks/monotonic-clock.now lifts injected u64 instant (#146)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    adapter.monotonic_clock_override = 42_000_000_000;

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.monotonicClockNow(&adapter, &ci, &.{}, &results, testing.allocator);
    try testing.expectEqual(@as(u64, 42_000_000_000), results[0].u64);
}

test "wasi:clocks/monotonic-clock.subscribe-instant mints unique pollable handles (#146)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    const args: [1]InterfaceValue = .{.{ .u64 = 0 }};
    var r1: [1]InterfaceValue = .{.{ .u32 = 0 }};
    var r2: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.monotonicSubscribe(&adapter, &ci, &args, &r1, testing.allocator);
    try WasiCliAdapter.monotonicSubscribe(&adapter, &ci, &args, &r2, testing.allocator);

    try testing.expect(r1[0].handle != r2[0].handle);
}

test "populateWasiProviders: binds wasi:random/random + insecure + insecure-seed (#147)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const imports = [_]ctypes_root.ImportDecl{
        .{ .name = "wasi:random/random@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:random/insecure@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:random/insecure-seed@0.2.6", .desc = .{ .instance = 0 } },
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

    try testing.expect(providers.contains("wasi:random/random@0.2.6"));
    try testing.expect(providers.contains("wasi:random/insecure@0.2.6"));
    try testing.expect(providers.contains("wasi:random/insecure-seed@0.2.6"));
    try testing.expect(!providers.contains("wasi:random/random"));
    try testing.expect(!providers.contains("wasi:random/insecure"));
    try testing.expect(!providers.contains("wasi:random/insecure-seed"));
    try testing.expect(adapter.random_iface.members.contains("get-random-bytes"));
    try testing.expect(adapter.random_iface.members.contains("get-random-u64"));
    try testing.expect(adapter.random_insecure_iface.members.contains("get-insecure-random-bytes"));
    try testing.expect(adapter.random_insecure_iface.members.contains("get-insecure-random-u64"));
    try testing.expect(adapter.random_insecure_seed_iface.members.contains("insecure-seed"));
}

test "wasi:random/random.get-random-u64 returns a u64 lift (#147)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.getRandomU64(null, &ci, &.{}, &results, testing.allocator);

    try testing.expect(results[0] == .u64);
}

test "wasi:random/insecure-seed lifts tuple<u64, u64> via tuple_val (#147)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.insecureSeed(null, &ci, &.{}, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .tuple_val);
    try testing.expectEqual(@as(usize, 2), results[0].tuple_val.len);
    try testing.expect(results[0].tuple_val[0] == .u64);
    try testing.expect(results[0].tuple_val[1] == .u64);
}

test "wasi:random/insecure: deterministic seed produces reproducible output (#147)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    adapter.insecure_prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const a = adapter.ensureInsecurePrng().int(u64);
    const b = adapter.ensureInsecurePrng().int(u64);

    adapter.insecure_prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const a2 = adapter.ensureInsecurePrng().int(u64);
    const b2 = adapter.ensureInsecurePrng().int(u64);

    try testing.expectEqual(a, a2);
    try testing.expectEqual(b, b2);
    try testing.expect(a != b);
}

test "wasi:random secure helpers fill a host buffer (#147)" {
    const testing = std.testing;
    var buf: [32]u8 = @splat(0);
    wasi_p2_core.Random.getRandomBytes(&buf);

    var any_nonzero = false;
    for (buf) |b| if (b != 0) { any_nonzero = true; break; };
    try testing.expect(any_nonzero);
}

test "populateWasiProviders: binds wasi:filesystem/types + preopens (#145)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const imports = [_]ctypes_root.ImportDecl{
        .{ .name = "wasi:filesystem/types@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:filesystem/preopens@0.2.6", .desc = .{ .instance = 0 } },
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

    try testing.expect(providers.contains("wasi:filesystem/types@0.2.6"));
    try testing.expect(providers.contains("wasi:filesystem/preopens@0.2.6"));
    try testing.expect(!providers.contains("wasi:filesystem/types"));
    try testing.expect(!providers.contains("wasi:filesystem/preopens"));
    try testing.expect(adapter.fs_types_iface.members.contains("[method]descriptor.get-type"));
    try testing.expect(adapter.fs_types_iface.members.contains("[method]descriptor.stat"));
    try testing.expect(adapter.fs_types_iface.members.contains("[method]descriptor.open-at"));
    try testing.expect(adapter.fs_types_iface.members.contains("[method]descriptor.read-via-stream"));
    try testing.expect(adapter.fs_types_iface.members.contains("[method]descriptor.write-via-stream"));
    try testing.expect(adapter.fs_types_iface.members.contains("[method]descriptor.append-via-stream"));
    try testing.expect(adapter.fs_types_iface.members.contains("[resource-drop]descriptor"));
    try testing.expect(adapter.fs_preopens_iface.members.contains("get-directories"));
}

test "filesystem: get-directories returns empty list when no preopens configured (#145)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.fsGetDirectories(&adapter, &ci, &.{}, &results, testing.allocator);

    try testing.expect(results[0] == .list);
    try testing.expectEqual(@as(u32, 0), results[0].list.len);
}

test "filesystem: addPreopen registers descriptor + name (#145)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const handle = try adapter.addPreopen("/tmp", tmp.dir);
    // tmp.cleanup() will close tmp.dir, which is the same handle the
    // adapter now owns. Replace the slot with a null so adapter.deinit
    // doesn't double-close.
    adapter.fs_descriptor_table.items[handle] = null;

    try testing.expectEqual(@as(u32, 0), handle);
    try testing.expectEqual(@as(usize, 1), adapter.fs_preopens.items.len);
    try testing.expectEqualStrings("/tmp", adapter.fs_preopens.items[0].name);
    try testing.expectEqual(@as(u32, 0), adapter.fs_preopens.items[0].dir_handle);
}

test "filesystem: open-at sandbox rejects .. and absolute paths (#145)" {
    const testing = std.testing;

    try testing.expectEqual(@as(?FsErrorCode, .access), WasiCliAdapter.validateSandboxPath("/etc/passwd"));
    try testing.expectEqual(@as(?FsErrorCode, .access), WasiCliAdapter.validateSandboxPath("../escape"));
    try testing.expectEqual(@as(?FsErrorCode, .access), WasiCliAdapter.validateSandboxPath("a/../b"));
    try testing.expectEqual(@as(?FsErrorCode, .access), WasiCliAdapter.validateSandboxPath("a\\b"));
    try testing.expectEqual(@as(?FsErrorCode, .access), WasiCliAdapter.validateSandboxPath("C:foo"));
    try testing.expectEqual(@as(?FsErrorCode, null), WasiCliAdapter.validateSandboxPath("a/b/c.txt"));
    try testing.expectEqual(@as(?FsErrorCode, null), WasiCliAdapter.validateSandboxPath(""));
    try testing.expectEqual(@as(?FsErrorCode, null), WasiCliAdapter.validateSandboxPath("..foo"));
}

test "populateWasiProviders: binds wasi:sockets/* (#148)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const imports = [_]ctypes_root.ImportDecl{
        .{ .name = "wasi:sockets/network@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:sockets/instance-network@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:sockets/tcp@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:sockets/tcp-create-socket@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:sockets/udp@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:sockets/udp-create-socket@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:sockets/ip-name-lookup@0.2.6", .desc = .{ .instance = 0 } },
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

    try testing.expect(providers.contains("wasi:sockets/network@0.2.6"));
    try testing.expect(providers.contains("wasi:sockets/instance-network@0.2.6"));
    try testing.expect(providers.contains("wasi:sockets/tcp@0.2.6"));
    try testing.expect(providers.contains("wasi:sockets/tcp-create-socket@0.2.6"));
    try testing.expect(providers.contains("wasi:sockets/udp@0.2.6"));
    try testing.expect(providers.contains("wasi:sockets/udp-create-socket@0.2.6"));
    try testing.expect(providers.contains("wasi:sockets/ip-name-lookup@0.2.6"));
    // Bare names (no @-suffix) must NOT linger when only versioned imports
    // were declared.
    try testing.expect(!providers.contains("wasi:sockets/network"));
    try testing.expect(!providers.contains("wasi:sockets/tcp"));

    // Spot-check that key methods are wired.
    try testing.expect(adapter.sockets_network_iface.members.contains("[resource-drop]network"));
    try testing.expect(adapter.sockets_instance_network_iface.members.contains("instance-network"));
    try testing.expect(adapter.sockets_tcp_create_iface.members.contains("create-tcp-socket"));
    try testing.expect(adapter.sockets_udp_create_iface.members.contains("create-udp-socket"));
    try testing.expect(adapter.sockets_tcp_iface.members.contains("[method]tcp-socket.start-bind"));
    try testing.expect(adapter.sockets_tcp_iface.members.contains("[method]tcp-socket.address-family"));
    try testing.expect(adapter.sockets_tcp_iface.members.contains("[method]tcp-socket.subscribe"));
    try testing.expect(adapter.sockets_tcp_iface.members.contains("[resource-drop]tcp-socket"));
    try testing.expect(adapter.sockets_udp_iface.members.contains("[method]udp-socket.start-bind"));
    try testing.expect(adapter.sockets_udp_iface.members.contains("[resource-drop]udp-socket"));
    try testing.expect(adapter.sockets_udp_iface.members.contains("[resource-drop]incoming-datagram-stream"));
    try testing.expect(adapter.sockets_udp_iface.members.contains("[resource-drop]outgoing-datagram-stream"));
    try testing.expect(adapter.sockets_ip_name_lookup_iface.members.contains("resolve-addresses"));
    try testing.expect(adapter.sockets_ip_name_lookup_iface.members.contains("[method]resolve-address-stream.resolve-next-address"));
    try testing.expect(adapter.sockets_ip_name_lookup_iface.members.contains("[resource-drop]resolve-address-stream"));
}

test "sockets: create-tcp-socket allocates slot (#148)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{.{ .enum_val = 1 }}; // ipv6
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);

    try testing.expect(results[0] == .result_val);
    try testing.expect(results[0].result_val.is_ok);
    try testing.expectEqual(@as(usize, 1), adapter.socket_table.items.len);
    const slot = adapter.socket_table.items[0].?;
    try testing.expectEqual(SocketKind.tcp, slot.kind);
    try testing.expectEqual(IpAddressFamily.ipv6, slot.family);
    try testing.expectEqual(SocketState.unbound, slot.state);
}

test "sockets: create-udp-socket allocates slot (#148)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{.{ .enum_val = 0 }}; // ipv4
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createUdpSocket(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);

    try testing.expect(results[0].result_val.is_ok);
    try testing.expectEqual(@as(usize, 1), adapter.socket_table.items.len);
    const slot = adapter.socket_table.items[0].?;
    try testing.expectEqual(SocketKind.udp, slot.kind);
    try testing.expectEqual(IpAddressFamily.ipv4, slot.family);
}

test "sockets: instance-network returns network handle (#148)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(&adapter, &ci, &.{}, &results, testing.allocator);

    try testing.expect(results[0] == .handle);
    try testing.expectEqual(@as(usize, 1), adapter.network_table.items.len);
    try testing.expect(adapter.network_table.items[0] != null);
}

test "sockets: tcp start-bind returns access-denied by default (#148)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    // Allocate a socket the call refers to.
    var ci: ComponentInstance = undefined;
    const create_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var create_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &create_args, &create_results, testing.allocator);
    defer testing.allocator.destroy(create_results[0].result_val.payload.?);

    // start-bind is a default-deny stub.
    const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, .{ .u32 = 0 } };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.socketDenyAccess(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);

    try testing.expect(results[0] == .result_val);
    try testing.expect(!results[0].result_val.is_ok);
    const err = results[0].result_val.payload.?.*;
    try testing.expectEqual(@as(u32, @intFromEnum(SocketErrorCode.access_denied)), err.variant_val.discriminant);
}

test "sockets: tcp address-family reads rep family (#148)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    const create_args = [_]InterfaceValue{.{ .enum_val = 1 }}; // ipv6
    var create_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &create_args, &create_results, testing.allocator);
    defer testing.allocator.destroy(create_results[0].result_val.payload.?);

    const args = [_]InterfaceValue{.{ .handle = 0 }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.socketAddressFamily(&adapter, &ci, &args, &results, testing.allocator);

    try testing.expect(results[0] == .enum_val);
    try testing.expectEqual(@as(u32, @intFromEnum(IpAddressFamily.ipv6)), results[0].enum_val);
}

test "sockets: ip-name-lookup smoke (#148)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    // resolve-addresses is default-deny: it must return name-unresolvable
    // cleanly without panicking.
    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .string = .{ .ptr = 0, .len = 0 } } };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveAddresses(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);

    try testing.expect(results[0] == .result_val);
    try testing.expect(!results[0].result_val.is_ok);
    const err = results[0].result_val.payload.?.*;
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.name_unresolvable)),
        err.variant_val.discriminant,
    );
}

test "populateWasiProviders: binds wasi:http/* (#149)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const imports = [_]ctypes_root.ImportDecl{
        .{ .name = "wasi:http/types@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:http/outgoing-handler@0.2.6", .desc = .{ .instance = 0 } },
        .{ .name = "wasi:http/incoming-handler@0.2.6", .desc = .{ .instance = 0 } },
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

    try testing.expect(providers.contains("wasi:http/types@0.2.6"));
    try testing.expect(providers.contains("wasi:http/outgoing-handler@0.2.6"));
    try testing.expect(providers.contains("wasi:http/incoming-handler@0.2.6"));
    // Bare names (no @-suffix) must NOT linger when only versioned
    // imports were declared.
    try testing.expect(!providers.contains("wasi:http/types"));
    try testing.expect(!providers.contains("wasi:http/outgoing-handler"));

    // Spot-check key methods.
    try testing.expect(adapter.http_types_iface.members.contains("[constructor]fields"));
    try testing.expect(adapter.http_types_iface.members.contains("[static]fields.from-list"));
    try testing.expect(adapter.http_types_iface.members.contains("[constructor]outgoing-request"));
    try testing.expect(adapter.http_types_iface.members.contains("[constructor]outgoing-response"));
    try testing.expect(adapter.http_types_iface.members.contains("[constructor]request-options"));
    try testing.expect(adapter.http_types_iface.members.contains("[static]response-outparam.set"));
    try testing.expect(adapter.http_types_iface.members.contains("[method]future-incoming-response.get"));
    try testing.expect(adapter.http_types_iface.members.contains("[resource-drop]fields"));
    try testing.expect(adapter.http_types_iface.members.contains("[resource-drop]outgoing-request"));
    try testing.expect(adapter.http_types_iface.members.contains("[resource-drop]future-incoming-response"));
    try testing.expect(adapter.http_outgoing_handler_iface.members.contains("handle"));
    try testing.expect(adapter.http_incoming_handler_iface.members.contains("handle"));
}

test "http: fields constructor + drop roundtrip (#149)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsConstructor(&adapter, &ci, &.{}, &results, testing.allocator);

    try testing.expect(results[0] == .handle);
    const handle = results[0].handle;
    try testing.expectEqual(@as(usize, 1), adapter.http_fields_table.items.len);
    try testing.expect(adapter.http_fields_table.items[handle] != null);

    // entries returns an empty list_val.
    var entries_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    const entries_args = [_]InterfaceValue{.{ .handle = handle }};
    try WasiCliAdapter.httpFieldsEntries(&adapter, &ci, &entries_args, &entries_results, testing.allocator);
    try testing.expect(entries_results[0] == .list_val);
    try testing.expectEqual(@as(usize, 0), entries_results[0].list_val.len);
    entries_results[0].deinit(testing.allocator);

    // Drop frees the slot.
    const drop_args = [_]InterfaceValue{.{ .handle = handle }};
    try WasiCliAdapter.httpFieldsDrop(&adapter, &ci, &drop_args, &.{}, testing.allocator);
    try testing.expect(adapter.http_fields_table.items[handle] == null);
}

test "http: fields.from-list returns ok with fresh handle (#149)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    // Pass an empty list-of-tuples — the adapter ignores the contents
    // anyway on the default-deny path.
    var ci: ComponentInstance = undefined;
    const empty_list = try testing.allocator.alloc(InterfaceValue, 0);
    defer testing.allocator.free(empty_list);
    const args = [_]InterfaceValue{.{ .list_val = empty_list }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsFromList(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .result_val);
    try testing.expect(results[0].result_val.is_ok);
    try testing.expect(results[0].result_val.payload.?.* == .handle);
    try testing.expectEqual(@as(usize, 1), adapter.http_fields_table.items.len);
}

test "http: outgoing-request constructor allocates slot (#149)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    // First make a fields slot; its handle becomes the request's
    // headers_handle.
    var f_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsConstructor(&adapter, &ci, &.{}, &f_results, testing.allocator);
    const fh = f_results[0].handle;

    const args = [_]InterfaceValue{.{ .handle = fh }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestConstructor(&adapter, &ci, &args, &results, testing.allocator);

    try testing.expect(results[0] == .handle);
    try testing.expectEqual(@as(usize, 1), adapter.http_outgoing_requests.items.len);
    const slot = adapter.http_outgoing_requests.items[0].?;
    try testing.expectEqual(fh, slot.headers_handle);

    // headers() should round-trip the same handle.
    const hdr_args = [_]InterfaceValue{.{ .handle = results[0].handle }};
    var hdr_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestHeaders(&adapter, &ci, &hdr_args, &hdr_results, testing.allocator);
    try testing.expectEqual(fh, hdr_results[0].handle);
}

test "http: outgoing-handler.handle returns ready future with denied (#149)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    // Outer call: result<own<future-incoming-response>, error-code>.
    const args = [_]InterfaceValue{
        .{ .handle = 0 }, // own<outgoing-request>
        .{ .option_val = .{ .is_some = false, .payload = null } }, // option<own<request-options>>
    };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingHandlerHandle(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .result_val);
    try testing.expect(results[0].result_val.is_ok);
    const fut_handle = results[0].result_val.payload.?.handle;

    // Future must be in ready_err state pre-poll.
    const fut = adapter.http_future_responses.items[fut_handle].?;
    try testing.expect(fut.state == .ready_err);
    try testing.expectEqual(
        @as(u32, @intFromEnum(HttpErrorCode.HTTP_request_denied)),
        fut.state.ready_err,
    );

    // Polling .get() yields some(ok(err(HTTP_request_denied))).
    const get_args = [_]InterfaceValue{.{ .handle = fut_handle }};
    var get_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFutureGet(&adapter, &ci, &get_args, &get_results, testing.allocator);
    defer get_results[0].deinit(testing.allocator);

    try testing.expect(get_results[0] == .option_val);
    try testing.expect(get_results[0].option_val.is_some);
    const outer = get_results[0].option_val.payload.?.*;
    try testing.expect(outer == .result_val);
    try testing.expect(outer.result_val.is_ok); // outer ok = future was not yet polled
    const middle = outer.result_val.payload.?.*;
    try testing.expect(middle == .result_val);
    try testing.expect(!middle.result_val.is_ok); // inner err = the actual error
    const err_variant = middle.result_val.payload.?.*;
    try testing.expectEqual(
        @as(u32, @intFromEnum(HttpErrorCode.HTTP_request_denied)),
        err_variant.variant_val.discriminant,
    );

    // Future is now polled — second .get() must return some(err(())).
    var get2_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFutureGet(&adapter, &ci, &get_args, &get2_results, testing.allocator);
    defer get2_results[0].deinit(testing.allocator);

    try testing.expect(get2_results[0].option_val.is_some);
    const outer2 = get2_results[0].option_val.payload.?.*;
    try testing.expect(!outer2.result_val.is_ok);
}
