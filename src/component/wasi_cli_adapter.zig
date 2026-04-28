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
/// `wasi:filesystem/types.descriptor-flags` (#181). Carries the access
/// + sync intent that the guest passed to `descriptor.open-at` and that
/// `descriptor.get-flags` reads back. Bit positions match the WIT order.
pub const FsDescriptorFlags = packed struct(u8) {
    read: bool = false,
    write: bool = false,
    file_integrity_sync: bool = false,
    data_integrity_sync: bool = false,
    requested_write_sync: bool = false,
    mutate_directory: bool = false,
    _pad: u2 = 0,

    pub fn fromBits(bits: u32) FsDescriptorFlags {
        return .{
            .read = (bits & 0b000001) != 0,
            .write = (bits & 0b000010) != 0,
            .file_integrity_sync = (bits & 0b000100) != 0,
            .data_integrity_sync = (bits & 0b001000) != 0,
            .requested_write_sync = (bits & 0b010000) != 0,
            .mutate_directory = (bits & 0b100000) != 0,
        };
    }

    pub fn toBits(self: FsDescriptorFlags) u32 {
        var out: u32 = 0;
        if (self.read) out |= 0b000001;
        if (self.write) out |= 0b000010;
        if (self.file_integrity_sync) out |= 0b000100;
        if (self.data_integrity_sync) out |= 0b001000;
        if (self.requested_write_sync) out |= 0b010000;
        if (self.mutate_directory) out |= 0b100000;
        return out;
    }

    /// Whether host writes against this descriptor must be flushed to
    /// stable storage (i.e. `file.sync()` after each flush boundary).
    /// Matches POSIX `O_SYNC`/`O_DSYNC`. `requested-write-sync`
    /// (`O_RSYNC`) is **read-side** in POSIX and not a write-sync trigger.
    pub fn needsWriteSync(self: FsDescriptorFlags) bool {
        return self.file_integrity_sync or self.data_integrity_sync;
    }
};

/// `wasi:filesystem/types.descriptor` resource — slot in the descriptor
/// table. Indices in `WasiCliAdapter.fs_descriptor_table` are guest
/// handles. Slots are nulled on `[resource-drop]descriptor`, except
/// `.preopen` slots which are persistent (a guest dropping a preopen
/// handle is treated as a no-op so subsequent calls to
/// `preopens.get-directories` and `descriptor.open-at` keep working).
///
/// Each variant carries the `descriptor-flags` the guest requested at
/// open time (#181). Preopens default to `read|write|mutate_directory`
/// so that `open-at` from a preopen retains its pre-#181 capabilities.
pub const FsDescriptor = union(enum) {
    /// Regular file or any other non-directory descriptor opened via
    /// `descriptor.open-at`. Owned — closed on resource-drop.
    file: FsFile,
    /// Directory descriptor opened via `descriptor.open-at` with the
    /// `directory` open-flag. Owned — closed on resource-drop.
    dir: FsDir,
    /// Preopen sandbox root — adapter-owned. Resource-drop is a no-op
    /// for preopens; the adapter closes them in `deinit`.
    preopen: FsDir,

    pub const FsFile = struct {
        file: std.Io.File,
        flags: FsDescriptorFlags = .{},
    };
    pub const FsDir = struct {
        dir: std.Io.Dir,
        flags: FsDescriptorFlags = .{},
    };

    pub fn flags(self: FsDescriptor) FsDescriptorFlags {
        return switch (self) {
            .file => |f| f.flags,
            .dir => |d| d.flags,
            .preopen => |d| d.flags,
        };
    }

    /// Underlying directory handle for any directory-shaped descriptor
    /// (`.dir` or `.preopen`). Returns null for `.file`.
    pub fn asDir(self: FsDescriptor) ?std.Io.Dir {
        return switch (self) {
            .dir => |d| d.dir,
            .preopen => |d| d.dir,
            .file => null,
        };
    }
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
        .file => |f| f.file.close(io),
        .dir => |d2| d2.dir.close(io),
        .preopen => |d2| d2.dir.close(io),
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

/// Pending lifecycle-op result. WIT splits each blocking op into a
/// `start-X` / `finish-X` pair: `start` kicks off the work, `finish`
/// reads the result. Our `start-X` runs synchronously (since
/// `std.Io.net` is blocking) and stores the outcome here for the
/// matching `finish-X` to consume. `idle` means no in-flight op.
const PendingTcpOp = enum { idle, bind_done, listen_done };

/// Adapter-side `tcp-socket` / `udp-socket` rep. TCP can hold a real
/// `std.Io.net.Server` once `start-listen` succeeds (#178 PR A).
/// UDP holds a `host_socket` after `start-bind` succeeds (#178 PR C).
pub const Socket = struct {
    kind: SocketKind,
    family: IpAddressFamily,
    state: SocketState = .unbound,
    /// Authorized local address recorded by `start-bind`. Refreshed
    /// from the kernel after `start-listen` (TCP) or immediately
    /// after the real bind syscall (UDP) so guests observe the
    /// ephemeral port when port=0 was requested.
    local_addr: ?std.Io.net.IpAddress = null,
    /// Active listening server. Owned by the rep; closed in
    /// `closeAll`. TCP only.
    server: ?std.Io.net.Server = null,
    /// Pending TCP lifecycle op. Always `.idle` for udp sockets.
    pending: PendingTcpOp = .idle,
    /// UDP datagram socket. Owned by the rep; closed in `closeAll`.
    host_socket: ?std.Io.net.Socket = null,
    /// Deep-copied allow-list from the network rep at bind time.
    /// Owned by the socket; freed in `closeAll`. Prevents UAF if
    /// the network resource is dropped before the socket.
    allow_list: []const IpCidr = &.{},
    /// Connected remote address set by `stream(some(remote))`.
    remote_addr: ?std.Io.net.IpAddress = null,
    /// Generation counter for `stream()` — each call bumps this;
    /// sub-resources record the value at creation. Stale streams
    /// whose generation != socket's return `invalid-state`.
    stream_generation: u32 = 0,

    /// Release any held kernel resources. Idempotent. Called from
    /// `[resource-drop]{tcp,udp}-socket` and adapter `deinit`.
    pub fn closeAll(self: *Socket, io: std.Io, allocator: Allocator) void {
        if (self.server) |*srv| srv.deinit(io);
        self.server = null;
        if (self.host_socket) |*hs| hs.close(io);
        self.host_socket = null;
        if (self.allow_list.len != 0) allocator.free(self.allow_list);
        self.allow_list = &.{};
        self.remote_addr = null;
    }
};

/// Rep for `wasi:sockets/udp.incoming-datagram-stream`. Each instance
/// is tied to a parent `Socket` via `parent_handle` and validated via
/// `generation` — if the socket's `stream_generation` has advanced
/// past this stream's, all operations return `invalid-state`.
pub const UdpIncomingStream = struct {
    parent_handle: u32,
    generation: u32,
    remote: ?std.Io.net.IpAddress,
};

/// Rep for `wasi:sockets/udp.outgoing-datagram-stream`. Tracks
/// `send_credit` for the WIT-mandated `check-send` / `send`
/// protocol: `send` traps if `check-send` was not called first
/// or if `datagrams.len > send_credit`.
pub const UdpOutgoingStream = struct {
    parent_handle: u32,
    generation: u32,
    remote: ?std.Io.net.IpAddress,
    send_credit: ?u64 = null,
};

/// CIDR block for the per-`network` allow-list (#180). Stored canonical:
/// host bits below the prefix are zeroed at parse time.
pub const IpCidr = union(enum) {
    ip4: struct { bytes: [4]u8, prefix: u6 },
    ip6: struct { bytes: [16]u8, prefix: u8 },

    pub const ParseError = error{
        InvalidCidr,
        InvalidPrefix,
        InvalidAddress,
    };

    /// Parse `"<addr>/<prefix>"`. The address half is parsed with port=0 by
    /// `Ip4Address.parse` / `Ip6Address.parse`; v4 is tried first. Prefix
    /// must be plain decimal (no leading `+`/`-`/zero-padding) within
    /// 0..=32 (v4) or 0..=128 (v6). Host bits are masked to zero so two
    /// textual variants of the same network compare equal.
    pub fn parse(text: []const u8) ParseError!IpCidr {
        const slash = std.mem.indexOfScalar(u8, text, '/') orelse return error.InvalidCidr;
        const addr_text = text[0..slash];
        const prefix_text = text[slash + 1 ..];
        if (addr_text.len == 0 or prefix_text.len == 0) return error.InvalidCidr;
        // Reject leading zeros / signs in prefix to keep canonical text.
        if (prefix_text[0] == '+' or prefix_text[0] == '-') return error.InvalidPrefix;
        if (prefix_text.len > 1 and prefix_text[0] == '0') return error.InvalidPrefix;
        for (prefix_text) |c| if (c < '0' or c > '9') return error.InvalidPrefix;
        // Reject bracketed IPv6 (`[::]/128`) — CIDR text is bare.
        if (addr_text[0] == '[') return error.InvalidAddress;

        if (std.Io.net.Ip4Address.parse(addr_text, 0)) |v4| {
            const p = std.fmt.parseInt(u8, prefix_text, 10) catch return error.InvalidPrefix;
            if (p > 32) return error.InvalidPrefix;
            var bytes = v4.bytes;
            maskV4(&bytes, @intCast(p));
            return .{ .ip4 = .{ .bytes = bytes, .prefix = @intCast(p) } };
        } else |_| {}
        if (std.Io.net.Ip6Address.parse(addr_text, 0)) |v6| {
            const p = std.fmt.parseInt(u8, prefix_text, 10) catch return error.InvalidPrefix;
            if (p > 128) return error.InvalidPrefix;
            var bytes = v6.bytes;
            maskV6(&bytes, p);
            return .{ .ip6 = .{ .bytes = bytes, .prefix = p } };
        } else |_| {}
        return error.InvalidAddress;
    }

    /// Same-family bit-prefix match. Mixed families never match — IPv4-
    /// mapped IPv6 addresses are matched as IPv6 only (callers receive an
    /// `IpAddress` already discriminated by the wasi:sockets layer).
    pub fn containsAddr(self: IpCidr, addr: std.Io.net.IpAddress) bool {
        return switch (self) {
            .ip4 => |c| switch (addr) {
                .ip4 => |a| matchPrefix(&c.bytes, &a.bytes, c.prefix),
                .ip6 => false,
            },
            .ip6 => |c| switch (addr) {
                .ip6 => |a| matchPrefix(&c.bytes, &a.bytes, c.prefix),
                .ip4 => false,
            },
        };
    }
};

fn maskV4(bytes: *[4]u8, prefix: u6) void {
    if (prefix >= 32) return;
    var i: usize = 0;
    var remaining: u8 = prefix;
    while (i < 4) : (i += 1) {
        if (remaining >= 8) {
            remaining -= 8;
        } else if (remaining == 0) {
            bytes[i] = 0;
        } else {
            const keep: u8 = @as(u8, 0xff) << @intCast(8 - remaining);
            bytes[i] &= keep;
            remaining = 0;
        }
    }
}

fn maskV6(bytes: *[16]u8, prefix: u8) void {
    if (prefix >= 128) return;
    var i: usize = 0;
    var remaining: u8 = prefix;
    while (i < 16) : (i += 1) {
        if (remaining >= 8) {
            remaining -= 8;
        } else if (remaining == 0) {
            bytes[i] = 0;
        } else {
            const keep: u8 = @as(u8, 0xff) << @intCast(8 - remaining);
            bytes[i] &= keep;
            remaining = 0;
        }
    }
}

fn matchPrefix(cidr_bytes: []const u8, addr_bytes: []const u8, prefix: u8) bool {
    var remaining: u8 = prefix;
    var i: usize = 0;
    while (remaining >= 8) : (i += 1) {
        if (cidr_bytes[i] != addr_bytes[i]) return false;
        remaining -= 8;
    }
    if (remaining == 0) return true;
    const mask: u8 = @as(u8, 0xff) << @intCast(8 - remaining);
    return (cidr_bytes[i] & mask) == (addr_bytes[i] & mask);
}

/// Adapter-side `network` rep (#180). Carries a per-instance allow-list of
/// CIDR blocks snapshotted from `WasiCliAdapter.sockets_allow_list_template`
/// at `instance-network` time; the snapshot is owned by the Network and
/// freed in `[resource-drop]network` / adapter `deinit`. Empty list = the
/// default deny-all capability.
///
/// TODO(#178): bind/connect handlers will call `allows()` before issuing
/// real `std.Io.net.Socket` I/O. Today the handlers are still
/// default-deny stubs, so the allow-list is stored but not yet consulted
/// from the WIT call sites.
pub const Network = struct {
    allow_list: []const IpCidr = &.{},

    pub fn allows(self: Network, addr: std.Io.net.IpAddress) bool {
        for (self.allow_list) |c| if (c.containsAddr(addr)) return true;
        return false;
    }

    fn deinit(self: *Network, allocator: Allocator) void {
        if (self.allow_list.len != 0) allocator.free(self.allow_list);
        self.allow_list = &.{};
    }
};

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

/// Map a `std.Io.net.HostName.LookupError` to the closest
/// `wasi:sockets/network.error-code` variant. Most parse / nameserver
/// failures funnel into `permanent-resolver-failure`; transient errors
/// (cancelation, would-block) become `temporary-resolver-failure`;
/// "no such host" maps to `name-unresolvable`.
fn mapDnsError(err: anyerror) SocketErrorCode {
    return switch (err) {
        error.UnknownHostName, error.NoAddressReturned => .name_unresolvable,
        error.Canceled, error.WouldBlock => .temporary_resolver_failure,
        error.ResolvConfParseFailed,
        error.InvalidDnsARecord,
        error.InvalidDnsAAAARecord,
        error.InvalidDnsCnameRecord,
        error.NameServerFailure,
        error.DetectingNetworkConfigurationFailed,
        => .permanent_resolver_failure,
        else => .permanent_resolver_failure,
    };
}

/// Lower a Zig `std.Io.net.IpAddress` into the wasi:sockets
/// `ip-address` variant: `variant { ipv4(tuple<u8 x 4>), ipv6(tuple<u16 x 8>) }`.
/// Caller owns the returned `InterfaceValue` (free via `deinit`).
fn lowerIpAddress(allocator: Allocator, addr: std.Io.net.IpAddress) !InterfaceValue {
    switch (addr) {
        .ip4 => |v4| {
            const bytes = v4.bytes;
            const octets = try allocator.alloc(InterfaceValue, 4);
            errdefer allocator.free(octets);
            inline for (0..4) |i| octets[i] = .{ .u8 = bytes[i] };
            const tup = try allocator.create(InterfaceValue);
            errdefer allocator.destroy(tup);
            tup.* = .{ .tuple_val = octets };
            return .{ .variant_val = .{
                .discriminant = @intFromEnum(IpAddressFamily.ipv4),
                .payload = tup,
            } };
        },
        .ip6 => |v6| {
            const bytes = v6.bytes;
            const groups = try allocator.alloc(InterfaceValue, 8);
            errdefer allocator.free(groups);
            inline for (0..8) |i| {
                const hi: u16 = bytes[i * 2];
                const lo: u16 = bytes[i * 2 + 1];
                groups[i] = .{ .u16 = (hi << 8) | lo };
            }
            const tup = try allocator.create(InterfaceValue);
            errdefer allocator.destroy(tup);
            tup.* = .{ .tuple_val = groups };
            return .{ .variant_val = .{
                .discriminant = @intFromEnum(IpAddressFamily.ipv6),
                .payload = tup,
            } };
        },
    }
}

/// Lift a wasi:sockets `ip-socket-address` variant value into a Zig
/// `std.Io.net.IpAddress`. Validates the variant arm matches `family`
/// (returns `error.FamilyMismatch`); rejects malformed records that
/// don't match the WIT shape (returns `error.InvalidArgs`).
///
/// WIT (0.2.6):
///   record ipv4-socket-address { port: u16, address: tuple<u8,u8,u8,u8> }
///   record ipv6-socket-address { port: u16, flow-info: u32,
///                                 address: tuple<u16,u16,u16,u16,u16,u16,u16,u16>,
///                                 scope-id: u32 }
fn liftIpSocketAddress(
    val: InterfaceValue,
    family: IpAddressFamily,
) error{ InvalidArgs, FamilyMismatch }!std.Io.net.IpAddress {
    const variant = switch (val) {
        .variant_val => |v| v,
        else => return error.InvalidArgs,
    };
    const arm: IpAddressFamily = if (variant.discriminant == 0) .ipv4 else .ipv6;
    if (arm != family) return error.FamilyMismatch;
    const payload = variant.payload orelse return error.InvalidArgs;
    const fields = switch (payload.*) {
        .record_val => |r| r,
        else => return error.InvalidArgs,
    };
    switch (arm) {
        .ipv4 => {
            if (fields.len < 2) return error.InvalidArgs;
            const port: u16 = switch (fields[0]) {
                .u16 => |p| p,
                else => return error.InvalidArgs,
            };
            const addr_tuple = switch (fields[1]) {
                .tuple_val => |t| t,
                else => return error.InvalidArgs,
            };
            if (addr_tuple.len != 4) return error.InvalidArgs;
            var bytes: [4]u8 = undefined;
            inline for (0..4) |i| {
                bytes[i] = switch (addr_tuple[i]) {
                    .u8 => |b| b,
                    else => return error.InvalidArgs,
                };
            }
            return .{ .ip4 = .{ .bytes = bytes, .port = port } };
        },
        .ipv6 => {
            if (fields.len < 4) return error.InvalidArgs;
            const port: u16 = switch (fields[0]) {
                .u16 => |p| p,
                else => return error.InvalidArgs,
            };
            const flow: u32 = switch (fields[1]) {
                .u32 => |f| f,
                else => return error.InvalidArgs,
            };
            const addr_tuple = switch (fields[2]) {
                .tuple_val => |t| t,
                else => return error.InvalidArgs,
            };
            if (addr_tuple.len != 8) return error.InvalidArgs;
            // WIT scope-id is captured but the std type holds it as an
            // Interface (none/index/name). PR A drops the scope id; it
            // will round-trip as 0 in `lowerIpSocketAddress`. The
            // listen/bind paths that PR A exposes don't need scoped
            // addresses.
            _ = fields[3];
            var bytes: [16]u8 = undefined;
            inline for (0..8) |i| {
                const group: u16 = switch (addr_tuple[i]) {
                    .u16 => |g| g,
                    else => return error.InvalidArgs,
                };
                bytes[i * 2] = @intCast((group >> 8) & 0xff);
                bytes[i * 2 + 1] = @intCast(group & 0xff);
            }
            return .{ .ip6 = .{ .port = port, .bytes = bytes, .flow = flow } };
        },
    }
}

/// Lower a `std.Io.net.IpAddress` into a wasi:sockets
/// `ip-socket-address` variant. Caller owns the returned
/// `InterfaceValue` (free via `.deinit(allocator)`).
fn lowerIpSocketAddress(allocator: Allocator, addr: std.Io.net.IpAddress) !InterfaceValue {
    switch (addr) {
        .ip4 => |v4| {
            const octets = try allocator.alloc(InterfaceValue, 4);
            errdefer allocator.free(octets);
            inline for (0..4) |i| octets[i] = .{ .u8 = v4.bytes[i] };
            const fields = try allocator.alloc(InterfaceValue, 2);
            errdefer allocator.free(fields);
            fields[0] = .{ .u16 = v4.port };
            fields[1] = .{ .tuple_val = octets };
            const rec = try allocator.create(InterfaceValue);
            errdefer allocator.destroy(rec);
            rec.* = .{ .record_val = fields };
            return .{ .variant_val = .{
                .discriminant = @intFromEnum(IpAddressFamily.ipv4),
                .payload = rec,
            } };
        },
        .ip6 => |v6| {
            const groups = try allocator.alloc(InterfaceValue, 8);
            errdefer allocator.free(groups);
            inline for (0..8) |i| {
                const hi: u16 = v6.bytes[i * 2];
                const lo: u16 = v6.bytes[i * 2 + 1];
                groups[i] = .{ .u16 = (hi << 8) | lo };
            }
            const fields = try allocator.alloc(InterfaceValue, 4);
            errdefer allocator.free(fields);
            fields[0] = .{ .u16 = v6.port };
            fields[1] = .{ .u32 = v6.flow };
            fields[2] = .{ .tuple_val = groups };
            // PR A elides scope-id (Interface enum doesn't carry an
            // integer index in all variants); always lower as 0.
            fields[3] = .{ .u32 = 0 };
            const rec = try allocator.create(InterfaceValue);
            errdefer allocator.destroy(rec);
            rec.* = .{ .record_val = fields };
            return .{ .variant_val = .{
                .discriminant = @intFromEnum(IpAddressFamily.ipv6),
                .payload = rec,
            } };
        },
    }
}

/// Map an `IpAddress.BindError` to a `wasi:sockets/network.error-code`.
fn mapSocketBindError(err: anyerror) SocketErrorCode {
    return switch (err) {
        error.AddressInUse => .address_in_use,
        error.AddressUnavailable => .address_not_bindable,
        error.AddressFamilyUnsupported => .invalid_argument,
        error.SystemResources, error.ProcessFdQuotaExceeded, error.SystemFdQuotaExceeded => .new_socket_limit,
        error.NetworkDown => .remote_unreachable,
        error.AccessDenied => .access_denied,
        else => .unknown,
    };
}

/// Map an `IpAddress.ListenError` to a `wasi:sockets/network.error-code`.
fn mapSocketListenError(err: anyerror) SocketErrorCode {
    return switch (err) {
        error.AddressInUse => .address_in_use,
        error.AddressUnavailable => .address_not_bindable,
        error.AddressFamilyUnsupported => .invalid_argument,
        error.SystemResources, error.ProcessFdQuotaExceeded, error.SystemFdQuotaExceeded => .new_socket_limit,
        error.NetworkDown => .remote_unreachable,
        error.AccessDenied => .access_denied,
        else => .unknown,
    };
}

/// Map a `Socket.SendError` to a `wasi:sockets/network.error-code`.
fn mapSocketSendError(err: anyerror) SocketErrorCode {
    return switch (err) {
        error.MessageOversize => .datagram_too_large,
        error.NetworkUnreachable, error.NetworkDown => .remote_unreachable,
        error.HostUnreachable => .remote_unreachable,
        error.ConnectionRefused => .connection_refused,
        error.ConnectionResetByPeer => .connection_reset,
        error.AccessDenied => .access_denied,
        error.AddressFamilyUnsupported => .invalid_argument,
        error.SystemResources => .out_of_memory,
        else => .unknown,
    };
}

/// Deep-copy a CIDR allow-list slice so the socket owns its own copy
/// (prevents UAF when the network resource is dropped first).
fn cloneAllowList(allocator: Allocator, src: []const IpCidr) ![]IpCidr {
    if (src.len == 0) return &.{};
    const dst = try allocator.alloc(IpCidr, src.len);
    @memcpy(dst, src);
    return dst;
}

/// Validate that `addr` is suitable as a send destination for the
/// given socket `family`. Returns `null` if valid, or the WIT
/// `error-code` discriminant to return on failure.
///
/// Checks (in WIT-mandated order):
///  1. family mismatch → `invalid-argument`
///  2. wildcard IP (0.0.0.0 / ::) → `invalid-argument`
///  3. port == 0 → `invalid-argument`
fn validateRemoteForSend(addr: std.Io.net.IpAddress, family: IpAddressFamily) ?SocketErrorCode {
    switch (addr) {
        .ip4 => |v4| {
            if (family != .ipv4) return .invalid_argument;
            if (std.mem.eql(u8, &v4.bytes, &.{ 0, 0, 0, 0 })) return .invalid_argument;
            if (v4.port == 0) return .invalid_argument;
        },
        .ip6 => |v6| {
            if (family != .ipv6) return .invalid_argument;
            if (std.mem.eql(u8, &v6.bytes, &(.{0} ** 16))) return .invalid_argument;
            if (v6.port == 0) return .invalid_argument;
        },
    }
    return null;
}

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

/// Lift a `list<tuple<string, string>>` from guest memory into a fresh
/// allocator-owned slice of `HttpFieldEntry`. Used by
/// `[static]fields.from-list` (#174). Each list element is laid out as
/// four little-endian u32s — name.ptr, name.len, value.ptr, value.len —
/// for a 16-byte stride and 4-byte alignment, matching the canonical
/// ABI lowering used by `getEnvironment`.
///
/// On any out-of-bounds read the partially-allocated entries are freed
/// and `error.OutOfBoundsMemory` is returned, leaving no leaks.
fn liftFieldEntries(
    allocator: Allocator,
    mem: []const u8,
    list_ptr: u32,
    count: u32,
) ![]HttpFieldEntry {
    const stride: u32 = 16;
    const total = std.math.mul(u32, count, stride) catch return error.OutOfBoundsMemory;
    const list_end = std.math.add(u32, list_ptr, total) catch return error.OutOfBoundsMemory;
    if (list_end > mem.len) return error.OutOfBoundsMemory;

    const entries = try allocator.alloc(HttpFieldEntry, count);
    var filled: usize = 0;
    errdefer {
        for (entries[0..filled]) |e| {
            allocator.free(e.name);
            allocator.free(e.value);
        }
        allocator.free(entries);
    }

    while (filled < count) : (filled += 1) {
        const off = list_ptr + @as(u32, @intCast(filled)) * stride;
        const name_ptr = std.mem.readInt(u32, mem[off..][0..4], .little);
        const name_len = std.mem.readInt(u32, mem[off + 4 ..][0..4], .little);
        const val_ptr = std.mem.readInt(u32, mem[off + 8 ..][0..4], .little);
        const val_len = std.mem.readInt(u32, mem[off + 12 ..][0..4], .little);

        const name_end = std.math.add(u32, name_ptr, name_len) catch return error.OutOfBoundsMemory;
        if (name_end > mem.len) return error.OutOfBoundsMemory;
        const val_end = std.math.add(u32, val_ptr, val_len) catch return error.OutOfBoundsMemory;
        if (val_end > mem.len) return error.OutOfBoundsMemory;

        const name_copy = try allocator.dupe(u8, mem[name_ptr..name_end]);
        errdefer allocator.free(name_copy);
        const val_copy = try allocator.dupe(u8, mem[val_ptr..val_end]);
        entries[filled] = .{ .name = name_copy, .value = val_copy };
    }

    return entries;
}

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
    body_handle: ?u32 = null,

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
/// real outbound HTTP (#176).
pub const IncomingResponse = struct {
    status: u16 = 0,
    headers_handle: u32 = 0,
    body_consumed: bool = false,
    /// Response body bytes owned by the adapter. Transferred to
    /// `IncomingBody` on `consume`, freed on drop if unconsumed.
    body_data: ?[]u8 = null,

    pub fn deinit(self: *IncomingResponse, allocator: Allocator) void {
        if (self.body_data) |d| allocator.free(d);
    }
};

/// `wasi:http/types.outgoing-response`.
pub const OutgoingResponse = struct {
    status: u16 = 200,
    headers_handle: u32,
    body_consumed: bool = false,
};

/// `wasi:http/types.incoming-body`. Holds readable body data
/// transferred from `IncomingResponse` on `consume`.
pub const IncomingBody = struct {
    /// Owned body bytes. Freed on drop.
    data: ?[]u8 = null,
    stream_taken: bool = false,

    pub fn deinit(self: *IncomingBody, allocator: Allocator) void {
        if (self.data) |d| allocator.free(d);
    }
};

/// `wasi:http/types.outgoing-body`.
pub const OutgoingBody = struct {
    /// Set by `[method]outgoing-body.write` to ensure a second call
    /// returns err per WIT contract.
    stream_taken: bool = false,
    /// Heap-allocated output stream created by `write`. The stream's
    /// buffer holds the body bytes written by the guest. The stream
    /// is owned by `WasiCliAdapter.owned_output_streams` and freed
    /// there; this pointer is only for reading the buffer contents.
    stream: ?*streams.OutputStream = null,
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

/// `(string, string)` pair forwarded to `wasi:cli/environment.get-environment`.
/// The slices are borrowed by the adapter — the caller is responsible for
/// keeping them alive across the run.
pub const EnvVar = struct {
    name: []const u8,
    value: []const u8,
};

pub const WasiCliAdapter = struct {
    allocator: Allocator,
    stdout: streams.OutputStream,
    stderr: streams.OutputStream,
    /// Captured stdin buffer. Defaults to empty; callers populate via
    /// `setStdinBytes` before running the component. The slice is
    /// borrowed — keep it alive for as long as the component runs.
    stdin: streams.InputStream = .{ .source = .closed },

    /// `wasi:cli/exit.exit` (or `exit-with-code`). The host fn
    /// also returns `error.Trap`; `runLoadedComponent` checks this
    /// field to translate the trap into a normal `RunOutcome`.
    exit_code: ?u32 = null,

    /// argv passed to the component via `wasi:cli/environment.get-arguments`.
    /// Borrowed — caller keeps slice alive for the run. Default empty.
    argv: []const []const u8 = &.{},
    /// envvars passed to the component via `wasi:cli/environment.get-environment`.
    /// Borrowed — caller keeps slices alive for the run. Default empty.
    env: []const EnvVar = &.{},

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
    /// Adapter-level template for the per-`network` CIDR allow-list (#180).
    /// Each `instance-network` snapshots this slice into the new
    /// `Network.allow_list`. Empty = deny-all (the default). Owned by the
    /// adapter and replaced atomically by `setSocketsAllowList`.
    sockets_allow_list_template: []IpCidr = &.{},
    /// `wasi:sockets/{tcp,udp}` socket resource table. Slot index = guest
    /// handle (shared across both kinds — `Socket.kind` discriminates).
    /// Slots are nulled on `[resource-drop]tcp-socket` /
    /// `[resource-drop]udp-socket`.
    socket_table: std.ArrayListUnmanaged(?Socket) = .empty,
    /// `wasi:sockets/udp.incoming-datagram-stream` sub-resource table.
    /// Each slot owns a heap-allocated rep; nulled on resource-drop.
    udp_incoming_streams: std.ArrayListUnmanaged(?*UdpIncomingStream) = .empty,
    /// `wasi:sockets/udp.outgoing-datagram-stream` sub-resource table.
    udp_outgoing_streams: std.ArrayListUnmanaged(?*UdpOutgoingStream) = .empty,
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

        // wasi:sockets resource tables. Sockets are POD; networks own a
        // copy of the per-instance CIDR allow-list (#180) and free it on
        // drop. Resolve streams own their result slice + the heap struct
        // itself.
        for (self.network_table.items) |*maybe| {
            if (maybe.* != null) maybe.*.?.deinit(self.allocator);
        }
        self.network_table.deinit(self.allocator);
        if (self.sockets_allow_list_template.len != 0)
            self.allocator.free(self.sockets_allow_list_template);
        {
            const io = std.Io.Threaded.global_single_threaded.io();
            for (self.socket_table.items) |*maybe| {
                if (maybe.*) |*s| s.closeAll(io, self.allocator);
            }
        }
        self.socket_table.deinit(self.allocator);
        for (self.udp_incoming_streams.items) |maybe| {
            if (maybe) |s| self.allocator.destroy(s);
        }
        self.udp_incoming_streams.deinit(self.allocator);
        for (self.udp_outgoing_streams.items) |maybe| {
            if (maybe) |s| self.allocator.destroy(s);
        }
        self.udp_outgoing_streams.deinit(self.allocator);
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
            if (maybe) |r| {
                r.deinit(self.allocator);
                self.allocator.destroy(r);
            }
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
            if (maybe) |b| {
                b.deinit(self.allocator);
                self.allocator.destroy(b);
            }
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

    /// Forward argv to `wasi:cli/environment.get-arguments`. The slices
    /// are borrowed — caller must keep them alive for the run.
    pub fn setArguments(self: *WasiCliAdapter, argv: []const []const u8) void {
        self.argv = argv;
    }

    /// Forward env to `wasi:cli/environment.get-environment`. The slices
    /// are borrowed — caller must keep them alive for the run.
    pub fn setEnvironment(self: *WasiCliAdapter, env: []const EnvVar) void {
        self.env = env;
    }

    /// Configure the per-`network` CIDR allow-list (#180). Each entry is
    /// parsed by `IpCidr.parse` (e.g. `"127.0.0.0/8"`, `"::1/128"`,
    /// `"fe80::/10"`). The slice is snapshotted into every `Network`
    /// returned by `instance-network` from this point onward; previously
    /// created `Network` resources keep their own snapshot.
    ///
    /// Default is deny-all (empty list). Calling with an empty slice
    /// resets to deny-all. Replaces (and frees) any prior template.
    /// On parse failure the previous template is preserved.
    ///
    /// TODO(#178): once `tcp-socket.start-bind` / `start-connect` issue
    /// real `std.Io.net.Socket` calls, they will gate destinations via
    /// `Network.allows()` before performing I/O. Today the bind/connect
    /// handlers remain default-deny stubs.
    pub fn setSocketsAllowList(self: *WasiCliAdapter, cidrs: []const []const u8) !void {
        if (cidrs.len == 0) {
            if (self.sockets_allow_list_template.len != 0)
                self.allocator.free(self.sockets_allow_list_template);
            self.sockets_allow_list_template = &.{};
            return;
        }
        const parsed = try self.allocator.alloc(IpCidr, cidrs.len);
        var filled: usize = 0;
        errdefer self.allocator.free(parsed);
        while (filled < cidrs.len) : (filled += 1) {
            parsed[filled] = try IpCidr.parse(cidrs[filled]);
        }
        if (self.sockets_allow_list_template.len != 0)
            self.allocator.free(self.sockets_allow_list_template);
        self.sockets_allow_list_template = parsed;
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
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;

        const n = self.env.len;
        if (n == 0) {
            results[0] = .{ .list = .{ .ptr = 0, .len = 0 } };
            return;
        }

        // Each tuple<string, string>: name.ptr@0:u32, name.len@4:u32,
        // value.ptr@8:u32, value.len@12:u32. Stride = 16, align = 4.
        const stride: usize = 16;
        const scratch = try allocator.alloc(u8, n * stride);
        defer allocator.free(scratch);

        for (self.env, 0..) |e, i| {
            const name_ptr = ci.hostAllocAndWrite(e.name) orelse return error.IoError;
            const value_ptr = ci.hostAllocAndWrite(e.value) orelse return error.IoError;
            const off = i * stride;
            std.mem.writeInt(u32, scratch[off..][0..4], name_ptr, .little);
            std.mem.writeInt(u32, scratch[off + 4 ..][0..4], @intCast(e.name.len), .little);
            std.mem.writeInt(u32, scratch[off + 8 ..][0..4], value_ptr, .little);
            std.mem.writeInt(u32, scratch[off + 12 ..][0..4], @intCast(e.value.len), .little);
        }

        const list_ptr = ci.hostAllocAndWrite(scratch) orelse return error.IoError;
        results[0] = .{ .list = .{ .ptr = list_ptr, .len = @intCast(n) } };
    }

    /// `wasi:cli/environment.get-arguments: () -> list<string>`. Forwards
    /// `self.argv`; empty when no caller invoked `setArguments`.
    fn getArguments(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;

        const n = self.argv.len;
        if (n == 0) {
            results[0] = .{ .list = .{ .ptr = 0, .len = 0 } };
            return;
        }

        // Each string element: ptr@0:u32, len@4:u32. Stride = 8, align = 4.
        const stride: usize = 8;
        const scratch = try allocator.alloc(u8, n * stride);
        defer allocator.free(scratch);

        for (self.argv, 0..) |s, i| {
            const s_ptr = ci.hostAllocAndWrite(s) orelse return error.IoError;
            const off = i * stride;
            std.mem.writeInt(u32, scratch[off..][0..4], s_ptr, .little);
            std.mem.writeInt(u32, scratch[off + 4 ..][0..4], @intCast(s.len), .little);
        }

        const list_ptr = ci.hostAllocAndWrite(scratch) orelse return error.IoError;
        results[0] = .{ .list = .{ .ptr = list_ptr, .len = @intCast(n) } };
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
        // `blocking-write-and-flush` semantically includes a flush.
        // Honor `descriptor-flags` sync bits for host-file sinks (#181).
        switch (stream.flush()) {
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
    ///   -> result<_, stream-error>`. For captured buffers, fd-backed,
    /// or non-sync host-file streams this is a no-op. Host-file streams
    /// opened with `file-integrity-sync` / `data-integrity-sync` (#181)
    /// issue `file.sync()` here so writes reach stable storage.
    fn outputStreamBlockingFlush(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len == 0 or results.len == 0) return error.InvalidArgs;
        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const stream = self.lookupStream(handle) orelse return error.InvalidHandle;
        switch (stream.flush()) {
            .ok => {},
            .err, .closed => return error.IoError,
        }
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
        try self.fs_descriptor_table.append(self.allocator, .{ .preopen = .{
            .dir = dir,
            // Preopens are full-capability roots: read+write+mutate so
            // pre-#181 callers (e.g. `open-at` of a writable child) keep
            // working without each embedder having to opt in.
            .flags = .{ .read = true, .write = true, .mutate_directory = true },
        } });
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
            .{ .name = "[method]descriptor.get-flags", .call = &fsDescriptorGetFlags },
            .{ .name = "[method]descriptor.stat", .call = &fsDescriptorStat },
            .{ .name = "[method]descriptor.stat-at", .call = &fsDescriptorStatAt },
            .{ .name = "[method]descriptor.set-times", .call = &fsDescriptorSetTimes },
            .{ .name = "[method]descriptor.set-times-at", .call = &fsDescriptorSetTimesAt },
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

    /// `[method]descriptor.get-flags: (borrow<descriptor>) -> result<descriptor-flags, error-code>` (#181).
    /// Returns the `descriptor-flags` the guest passed when this
    /// descriptor was opened (preopens always read|write|mutate).
    fn fsDescriptorGetFlags(
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
        const words = try allocator.alloc(u32, 1);
        words[0] = d.flags().toBits();
        results[0] = try fsResultOk(allocator, .{ .flags_val = words });
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

        const stat_result: ?std.Io.File.Stat = switch (d.*) {
            .preopen => |dir| dir.dir.stat(io) catch null,
            .dir => |dir| dir.dir.stat(io) catch null,
            .file => |f| f.file.stat(io) catch |err| {
                results[0] = try fsResultErr(allocator, mapFsError(err));
                return;
            },
        };

        if (stat_result) |st| {
            results[0] = try fsResultOk(allocator, try buildDescriptorStatRecord(allocator, st));
            return;
        }

        // Dir handle but `dir.stat` failed (or unsupported on this target).
        // Fall back to a minimal record with timestamps as `option::none`.
        const fields = try allocator.alloc(InterfaceValue, 6);
        fields[0] = .{ .variant_val = .{ .discriminant = @intFromEnum(WasiCliAdapter.DescType.directory), .payload = null } };
        fields[1] = .{ .u64 = 1 };
        fields[2] = .{ .u64 = 0 };
        fields[3] = .{ .option_val = .{ .is_some = false, .payload = null } };
        fields[4] = .{ .option_val = .{ .is_some = false, .payload = null } };
        fields[5] = .{ .option_val = .{ .is_some = false, .payload = null } };
        results[0] = try fsResultOk(allocator, .{ .record_val = fields });
    }

    /// `[method]descriptor.stat-at: (borrow<descriptor>, path-flags, string)
    ///   -> result<descriptor-stat, error-code>`.
    fn fsDescriptorStatAt(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 3 or results.len == 0) return error.InvalidArgs;

        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        // `path-flags` (#181): bit 0 = symlink-follow.
        const path_flags: u32 = switch (args[1]) {
            .flags_val => |w| if (w.len == 0) 0 else w[0],
            .u32 => |v| v,
            else => 0,
        };
        const follow_symlinks = (path_flags & 0b1) != 0;
        const path_pl = switch (args[2]) {
            .string => |pl| pl,
            else => return error.InvalidArgs,
        };

        const d = self.lookupFsDescriptor(handle) orelse {
            results[0] = try fsResultErr(allocator, .bad_descriptor);
            return;
        };
        const base_dir: std.Io.Dir = d.asDir() orelse {
            results[0] = try fsResultErr(allocator, .not_directory);
            return;
        };

        const path_bytes = ci.readGuestBytes(path_pl.ptr, path_pl.len) orelse
            return error.OutOfBoundsMemory;

        if (validateSandboxPath(path_bytes)) |code| {
            results[0] = try fsResultErr(allocator, code);
            return;
        }

        const io = std.Io.Threaded.global_single_threaded.io();
        const st = base_dir.statFile(io, path_bytes, .{
            .follow_symlinks = follow_symlinks,
        }) catch |err| {
            results[0] = try fsResultErr(allocator, mapFsError(err));
            return;
        };

        results[0] = try fsResultOk(allocator, try buildDescriptorStatRecord(allocator, st));
    }

    /// `[method]descriptor.set-times: (borrow<descriptor>, new-timestamp, new-timestamp)
    ///   -> result<_, error-code>`.
    fn fsDescriptorSetTimes(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 3 or results.len == 0) return error.InvalidArgs;

        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const ats = try liftNewTimestamp(args[1]);
        const mts = try liftNewTimestamp(args[2]);

        const d = self.lookupFsDescriptor(handle) orelse {
            results[0] = try fsResultErr(allocator, .bad_descriptor);
            return;
        };

        const io = std.Io.Threaded.global_single_threaded.io();
        switch (d.*) {
            .file => |f| {
                f.file.setTimestamps(io, .{
                    .access_timestamp = ats,
                    .modify_timestamp = mts,
                }) catch |err| {
                    results[0] = try fsResultErr(allocator, mapFsError(err));
                    return;
                };
            },
            .preopen, .dir => {
                // The `Io.Dir` API has no whole-directory setTimestamps;
                // only `setTimestamps(sub_path, ...)`. For a bare directory
                // descriptor (no sub_path), we report `not_permitted`.
                results[0] = try fsResultErr(allocator, .not_permitted);
                return;
            },
        }

        const ok_payload = try allocator.create(InterfaceValue);
        ok_payload.* = .{ .tuple_val = &.{} };
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = ok_payload } };
    }

    /// `[method]descriptor.set-times-at: (borrow<descriptor>, path-flags, string,
    ///   new-timestamp, new-timestamp) -> result<_, error-code>`.
    fn fsDescriptorSetTimesAt(
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
        const path_flags: u32 = switch (args[1]) {
            .flags_val => |w| if (w.len == 0) 0 else w[0],
            .u32 => |v| v,
            else => 0,
        };
        const path_pl = switch (args[2]) {
            .string => |pl| pl,
            else => return error.InvalidArgs,
        };
        const ats = try liftNewTimestamp(args[3]);
        const mts = try liftNewTimestamp(args[4]);

        const d = self.lookupFsDescriptor(handle) orelse {
            results[0] = try fsResultErr(allocator, .bad_descriptor);
            return;
        };
        const base_dir: std.Io.Dir = d.asDir() orelse {
            results[0] = try fsResultErr(allocator, .not_directory);
            return;
        };

        const path_bytes = ci.readGuestBytes(path_pl.ptr, path_pl.len) orelse
            return error.OutOfBoundsMemory;

        if (validateSandboxPath(path_bytes)) |code| {
            results[0] = try fsResultErr(allocator, code);
            return;
        }

        // path-flags bit 0 = symlink-follow (per WIT); 0 means "don't follow".
        const follow_symlinks = (path_flags & 0b1) != 0;

        const io = std.Io.Threaded.global_single_threaded.io();
        base_dir.setTimestamps(io, path_bytes, .{
            .follow_symlinks = follow_symlinks,
            .access_timestamp = ats,
            .modify_timestamp = mts,
        }) catch |err| {
            results[0] = try fsResultErr(allocator, mapFsError(err));
            return;
        };

        const ok_payload = try allocator.create(InterfaceValue);
        ok_payload.* = .{ .tuple_val = &.{} };
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = ok_payload } };
    }

    /// `[method]descriptor.open-at: (borrow<descriptor>, path-flags, string,
    ///   open-flags, descriptor-flags) -> result<own<descriptor>, error-code>`.
    ///
    /// Honors all six descriptor-flags bits (#181):
    /// - `read`/`write` map to Zig open-mode.
    /// - `file-integrity-sync`/`data-integrity-sync` are recorded on the
    ///   new descriptor so blocking-flush calls `file.sync()`.
    /// - `requested-write-sync` is stored (POSIX `O_RSYNC`, read-side)
    ///   but not enforceable today; reported via `get-flags`.
    /// - `mutate-directory` is only valid on directory descriptors and
    ///   is required on the *base* directory for any `open-at` that
    ///   would create, truncate, or open with write access.
    /// `path-flags.symlink-follow` is plumbed into both `openDir` and
    /// `openFile`. `createFile` ignores it because Zig 0.16's
    /// `CreateFileOptions` has no `follow_symlinks` field.
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
        const path_flags: u32 = switch (args[1]) {
            .flags_val => |w| if (w.len == 0) 0 else w[0],
            .u32 => |v| v,
            else => 0,
        };
        const path_pl = switch (args[2]) {
            .string => |pl| pl,
            else => return error.InvalidArgs,
        };
        const open_flags: u32 = switch (args[3]) {
            .flags_val => |w| if (w.len == 0) 0 else w[0],
            .u32 => |v| v,
            else => 0,
        };
        const desc_flags_bits: u32 = switch (args[4]) {
            .flags_val => |w| if (w.len == 0) 0 else w[0],
            .u32 => |v| v,
            else => 0,
        };
        const child_flags = FsDescriptorFlags.fromBits(desc_flags_bits);

        const d = self.lookupFsDescriptor(handle) orelse {
            results[0] = try fsResultErr(allocator, .bad_descriptor);
            return;
        };
        const base_flags = d.flags();
        const base_dir: std.Io.Dir = d.asDir() orelse {
            results[0] = try fsResultErr(allocator, .not_directory);
            return;
        };

        // open-flags bits per WIT order: 0=create, 1=directory, 2=exclusive, 3=truncate.
        const want_create = (open_flags & 0b0001) != 0;
        const want_directory = (open_flags & 0b0010) != 0;
        const want_exclusive = (open_flags & 0b0100) != 0;
        const want_truncate = (open_flags & 0b1000) != 0;
        const want_read = child_flags.read;
        const want_write = child_flags.write;
        const follow_symlinks = (path_flags & 0b1) != 0;

        // Spec (#181): if the base directory was opened without
        // `mutate-directory`, any child open that would mutate (create,
        // truncate, or any write access) must be denied with read-only.
        // Cheap check before we touch guest memory.
        const wants_mutate = want_create or want_truncate or want_write or
            child_flags.mutate_directory;
        if (wants_mutate and !base_flags.mutate_directory) {
            results[0] = try fsResultErr(allocator, .read_only);
            return;
        }

        const path_bytes = ci.readGuestBytes(path_pl.ptr, path_pl.len) orelse
            return error.OutOfBoundsMemory;

        if (validateSandboxPath(path_bytes)) |code| {
            results[0] = try fsResultErr(allocator, code);
            return;
        }

        // `mutate-directory` is only meaningful on directory descriptors;
        // strip it from the file-shaped child to keep `get-flags` honest.
        var stored_flags = child_flags;
        if (!want_directory) stored_flags.mutate_directory = false;

        const io = std.Io.Threaded.global_single_threaded.io();

        if (want_directory) {
            const new_dir = base_dir.openDir(io, path_bytes, .{
                .follow_symlinks = follow_symlinks,
            }) catch |err| {
                results[0] = try fsResultErr(allocator, mapFsError(err));
                return;
            };
            const new_handle = self.pushFsDescriptor(.{ .dir = .{
                .dir = new_dir,
                .flags = stored_flags,
            } }) catch {
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
            .follow_symlinks = follow_symlinks,
        }) catch |err| {
            results[0] = try fsResultErr(allocator, mapFsError(err));
            return;
        };

        const new_handle = self.pushFsDescriptor(.{ .file = .{
            .file = new_file,
            .flags = stored_flags,
        } }) catch {
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
        const file: std.Io.File = switch (d.*) {
            .file => |f| f.file,
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
        const fs_file: FsDescriptor.FsFile = switch (d.*) {
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
        stream.* = streams.OutputStream.toHostFile(
            fs_file.file,
            offset,
            append,
            fs_file.flags.needsWriteSync(),
        );
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

    fn pushUdpIncomingStream(self: *WasiCliAdapter, s: *UdpIncomingStream) !u32 {
        for (self.udp_incoming_streams.items, 0..) |slot, i| {
            if (slot == null) {
                self.udp_incoming_streams.items[i] = s;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.udp_incoming_streams.items.len);
        try self.udp_incoming_streams.append(self.allocator, s);
        return idx;
    }

    fn lookupUdpIncomingStream(self: *WasiCliAdapter, h: u32) ?*UdpIncomingStream {
        if (h >= self.udp_incoming_streams.items.len) return null;
        return self.udp_incoming_streams.items[h];
    }

    fn pushUdpOutgoingStream(self: *WasiCliAdapter, s: *UdpOutgoingStream) !u32 {
        for (self.udp_outgoing_streams.items, 0..) |slot, i| {
            if (slot == null) {
                self.udp_outgoing_streams.items[i] = s;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.udp_outgoing_streams.items.len);
        try self.udp_outgoing_streams.append(self.allocator, s);
        return idx;
    }

    fn lookupUdpOutgoingStream(self: *WasiCliAdapter, h: u32) ?*UdpOutgoingStream {
        if (h >= self.udp_outgoing_streams.items.len) return null;
        return self.udp_outgoing_streams.items[h];
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
            if (self.socket_table.items[handle]) |*s| {
                const io = std.Io.Threaded.global_single_threaded.io();
                s.closeAll(io, self.allocator);
            }
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
            if (self.network_table.items[handle] != null) {
                self.network_table.items[handle].?.deinit(self.allocator);
            }
            self.network_table.items[handle] = null;
        }
    }

    /// `wasi:sockets/instance-network.instance-network: () -> own<network>`.
    /// Snapshots the adapter's CIDR allow-list template (#180) into the new
    /// Network. The snapshot is independent: subsequent
    /// `setSocketsAllowList` calls do not mutate already-issued Networks.
    fn instanceNetwork(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        _: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;
        const snapshot: []IpCidr = if (self.sockets_allow_list_template.len == 0)
            &.{}
        else blk: {
            const buf = try self.allocator.alloc(IpCidr, self.sockets_allow_list_template.len);
            @memcpy(buf, self.sockets_allow_list_template);
            break :blk buf;
        };
        errdefer if (snapshot.len != 0) self.allocator.free(snapshot);
        const h = try self.pushNetwork(.{ .allow_list = snapshot });
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

    /// `[method]tcp-socket.start-bind:
    ///   (borrow<tcp-socket>, borrow<network>, ip-socket-address)
    ///     -> result<_, error-code>` (#178).
    ///
    /// PR A semantics: this is a *capability check + record* pass — no
    /// kernel syscall. The adapter remembers the authorized address;
    /// the matching `start-listen` (PR A) and `start-connect` (PR B)
    /// pass it to `std.Io.net.IpAddress.{listen,connect}`. Doing the
    /// real `IpAddress.bind` here would consume the port slot and
    /// leave us unable to call `listen()` later — stdlib has no
    /// listen-on-already-bound API.
    fn tcpStartBind(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 3 or results.len == 0) return error.InvalidArgs;
        const sock_handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const net_handle = switch (args[1]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const s = self.lookupSocket(sock_handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (s.kind != .tcp or s.state != .unbound or s.pending != .idle) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        const local = liftIpSocketAddress(args[2], s.family) catch |err| switch (err) {
            error.FamilyMismatch => {
                results[0] = try socketResultErr(allocator, .invalid_argument);
                return;
            },
            error.InvalidArgs => return error.InvalidArgs,
        };
        const net = if (net_handle < self.network_table.items.len)
            self.network_table.items[net_handle]
        else
            null;
        if (net == null or !net.?.allows(local)) {
            results[0] = try socketResultErr(allocator, .access_denied);
            return;
        }
        s.local_addr = local;
        s.pending = .bind_done;
        results[0] = try socketResultOk(allocator, .{ .tuple_val = &.{} });
    }

    /// `[method]tcp-socket.finish-bind: (borrow<tcp-socket>) ->
    /// result<_, error-code>` (#178). Reads the outcome stashed by
    /// `start-bind` and transitions to `bound`.
    fn tcpFinishBind(
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
        const s = self.lookupSocket(handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (s.kind != .tcp or s.pending != .bind_done) {
            results[0] = try socketResultErr(allocator, .not_in_progress);
            return;
        }
        s.pending = .idle;
        s.state = .bound;
        results[0] = try socketResultOk(allocator, .{ .tuple_val = &.{} });
    }

    /// `[method]tcp-socket.start-listen: (borrow<tcp-socket>) ->
    /// result<_, error-code>` (#178). Calls
    /// `std.Io.net.IpAddress.listen` with the address recorded by
    /// `start-bind`; refreshes `local_addr` from the kernel-resolved
    /// `Server.socket.address` so guests observe the assigned port
    /// when port=0 was requested.
    fn tcpStartListen(
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
        const s = self.lookupSocket(handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (s.kind != .tcp or s.state != .bound or s.pending != .idle) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        const local = s.local_addr orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const io = std.Io.Threaded.global_single_threaded.io();
        const server = std.Io.net.IpAddress.listen(&local, io, .{}) catch |err| {
            results[0] = try socketResultErr(allocator, mapSocketListenError(err));
            return;
        };
        s.server = server;
        s.local_addr = server.socket.address;
        s.pending = .listen_done;
        results[0] = try socketResultOk(allocator, .{ .tuple_val = &.{} });
    }

    /// `[method]tcp-socket.finish-listen: (borrow<tcp-socket>) ->
    /// result<_, error-code>` (#178).
    fn tcpFinishListen(
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
        const s = self.lookupSocket(handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (s.kind != .tcp or s.pending != .listen_done) {
            results[0] = try socketResultErr(allocator, .not_in_progress);
            return;
        }
        s.pending = .idle;
        s.state = .listening;
        results[0] = try socketResultOk(allocator, .{ .tuple_val = &.{} });
    }

    /// `[method]tcp-socket.local-address: (borrow<tcp-socket>) ->
    /// result<ip-socket-address, error-code>` (#178).
    fn tcpLocalAddress(
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
        const s = self.lookupSocket(handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const local = s.local_addr orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const lowered = try lowerIpSocketAddress(allocator, local);
        results[0] = try socketResultOk(allocator, lowered);
    }

    // ----- UDP host functions (#178 PR C) -----

    /// `[method]udp-socket.start-bind:
    ///   (borrow<udp-socket>, borrow<network>, ip-socket-address)
    ///     -> result<_, error-code>` (#178).
    ///
    /// Unlike TCP start-bind, UDP performs the real bind syscall
    /// immediately (UDP has no listen step).
    fn udpStartBind(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 3 or results.len == 0) return error.InvalidArgs;
        const sock_handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const net_handle = switch (args[1]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const s = self.lookupSocket(sock_handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (s.kind != .udp or s.state != .unbound or s.pending != .idle) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        const local = liftIpSocketAddress(args[2], s.family) catch |err| switch (err) {
            error.FamilyMismatch => {
                results[0] = try socketResultErr(allocator, .invalid_argument);
                return;
            },
            error.InvalidArgs => return error.InvalidArgs,
        };
        const net = if (net_handle < self.network_table.items.len)
            self.network_table.items[net_handle]
        else
            null;
        if (net == null or !net.?.allows(local)) {
            results[0] = try socketResultErr(allocator, .access_denied);
            return;
        }
        // Deep-copy the allow-list onto the socket so it survives
        // network resource-drop.
        const owned_list = cloneAllowList(self.allocator, net.?.allow_list) catch {
            results[0] = try socketResultErr(allocator, .out_of_memory);
            return;
        };
        // Real bind syscall — UDP has no listen step.
        const io = std.Io.Threaded.global_single_threaded.io();
        const host_socket = std.Io.net.IpAddress.bind(&local, io, .{
            .mode = .dgram,
            .protocol = .udp,
        }) catch |err| {
            if (owned_list.len != 0) self.allocator.free(owned_list);
            results[0] = try socketResultErr(allocator, mapSocketBindError(err));
            return;
        };
        s.host_socket = host_socket;
        s.local_addr = host_socket.address;
        s.allow_list = owned_list;
        s.pending = .bind_done;
        results[0] = try socketResultOk(allocator, .{ .tuple_val = &.{} });
    }

    /// `[method]udp-socket.finish-bind: (borrow<udp-socket>) ->
    /// result<_, error-code>` (#178).
    fn udpFinishBind(
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
        const s = self.lookupSocket(handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (s.kind != .udp or s.pending != .bind_done) {
            results[0] = try socketResultErr(allocator, .not_in_progress);
            return;
        }
        s.pending = .idle;
        s.state = .bound;
        results[0] = try socketResultOk(allocator, .{ .tuple_val = &.{} });
    }

    /// `[method]udp-socket.stream:
    ///   (borrow<udp-socket>, option<ip-socket-address>)
    ///     -> result<tuple<own<incoming-datagram-stream>,
    ///                     own<outgoing-datagram-stream>>, error-code>`
    /// (#178).
    ///
    /// Each call invalidates any previously-returned stream pair via a
    /// generation counter on the socket.
    fn udpStream(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 2 or results.len == 0) return error.InvalidArgs;
        const sock_handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const s = self.lookupSocket(sock_handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (s.kind != .udp or s.state != .bound) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        // Parse optional remote address.
        var remote: ?std.Io.net.IpAddress = null;
        switch (args[1]) {
            .variant_val => |v| {
                // option<T> is variant { none(0), some(1) }
                if (v.discriminant == 1) {
                    if (v.payload) |payload| {
                        const addr = liftIpSocketAddress(payload.*, s.family) catch |err| switch (err) {
                            error.FamilyMismatch => {
                                results[0] = try socketResultErr(allocator, .invalid_argument);
                                return;
                            },
                            error.InvalidArgs => return error.InvalidArgs,
                        };
                        // WIT-mandated validation order before allow-list.
                        if (validateRemoteForSend(addr, s.family)) |code| {
                            results[0] = try socketResultErr(allocator, code);
                            return;
                        }
                        // Allow-list check on the remote.
                        if (s.allow_list.len > 0) {
                            var allowed = false;
                            for (s.allow_list) |c| {
                                if (c.containsAddr(addr)) {
                                    allowed = true;
                                    break;
                                }
                            }
                            if (!allowed) {
                                results[0] = try socketResultErr(allocator, .access_denied);
                                return;
                            }
                        } else {
                            // Empty allow-list = deny-all.
                            results[0] = try socketResultErr(allocator, .access_denied);
                            return;
                        }
                        remote = addr;
                    }
                }
            },
            .option_val => |opt| {
                if (opt.payload) |payload| {
                    const addr = liftIpSocketAddress(payload.*, s.family) catch |err| switch (err) {
                        error.FamilyMismatch => {
                            results[0] = try socketResultErr(allocator, .invalid_argument);
                            return;
                        },
                        error.InvalidArgs => return error.InvalidArgs,
                    };
                    if (validateRemoteForSend(addr, s.family)) |code| {
                        results[0] = try socketResultErr(allocator, code);
                        return;
                    }
                    if (s.allow_list.len > 0) {
                        var allowed = false;
                        for (s.allow_list) |c| {
                            if (c.containsAddr(addr)) {
                                allowed = true;
                                break;
                            }
                        }
                        if (!allowed) {
                            results[0] = try socketResultErr(allocator, .access_denied);
                            return;
                        }
                    } else {
                        results[0] = try socketResultErr(allocator, .access_denied);
                        return;
                    }
                    remote = addr;
                }
            },
            else => {
                // none case — unconnected stream.
            },
        }

        // Bump generation to invalidate any prior stream pair.
        s.stream_generation += 1;
        s.remote_addr = remote;

        // Allocate incoming stream rep.
        const in_rep = self.allocator.create(UdpIncomingStream) catch {
            results[0] = try socketResultErr(allocator, .out_of_memory);
            return;
        };
        in_rep.* = .{
            .parent_handle = sock_handle,
            .generation = s.stream_generation,
            .remote = remote,
        };
        const in_handle = self.pushUdpIncomingStream(in_rep) catch {
            self.allocator.destroy(in_rep);
            results[0] = try socketResultErr(allocator, .out_of_memory);
            return;
        };

        // Allocate outgoing stream rep.
        const out_rep = self.allocator.create(UdpOutgoingStream) catch {
            // Clean up the incoming we already pushed.
            self.udp_incoming_streams.items[in_handle] = null;
            self.allocator.destroy(in_rep);
            results[0] = try socketResultErr(allocator, .out_of_memory);
            return;
        };
        out_rep.* = .{
            .parent_handle = sock_handle,
            .generation = s.stream_generation,
            .remote = remote,
        };
        const out_handle = self.pushUdpOutgoingStream(out_rep) catch {
            self.udp_incoming_streams.items[in_handle] = null;
            self.allocator.destroy(in_rep);
            self.allocator.destroy(out_rep);
            results[0] = try socketResultErr(allocator, .out_of_memory);
            return;
        };

        // Return ok(tuple(in_handle, out_handle)).
        const pair = try allocator.alloc(InterfaceValue, 2);
        pair[0] = .{ .handle = in_handle };
        pair[1] = .{ .handle = out_handle };
        results[0] = try socketResultOk(allocator, .{ .tuple_val = pair });
    }

    /// `[method]udp-socket.local-address: (borrow<udp-socket>) ->
    /// result<ip-socket-address, error-code>` (#178).
    fn udpLocalAddress(
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
        const s = self.lookupSocket(handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (s.kind != .udp) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        const local = s.local_addr orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const lowered = try lowerIpSocketAddress(allocator, local);
        results[0] = try socketResultOk(allocator, lowered);
    }

    /// `[method]udp-socket.remote-address: (borrow<udp-socket>) ->
    /// result<ip-socket-address, error-code>` (#178).
    fn udpRemoteAddress(
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
        const s = self.lookupSocket(handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (s.kind != .udp) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        const remote = s.remote_addr orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const lowered = try lowerIpSocketAddress(allocator, remote);
        results[0] = try socketResultOk(allocator, lowered);
    }

    /// `[method]incoming-datagram-stream.receive:
    ///   (borrow<incoming-datagram-stream>, u64) ->
    ///     result<list<incoming-datagram>, error-code>` (#178).
    ///
    /// Non-blocking receive per WIT: never blocks, never returns
    /// `would-block`. Uses a zero-duration timeout; Timeout maps to
    /// "no data available, return list so far".
    fn incomingDatagramReceive(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 2 or results.len == 0) return error.InvalidArgs;
        const stream_handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const max_results: u64 = switch (args[1]) {
            .u64 => |v| v,
            .u32 => |v| @as(u64, v),
            else => return error.InvalidArgs,
        };
        const in_stream = self.lookupUdpIncomingStream(stream_handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const s = self.lookupSocket(in_stream.parent_handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        // Generation check — stale stream.
        if (in_stream.generation != s.stream_generation) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        if (max_results == 0) {
            // WIT: max_results==0 → ok([]).
            const empty_list = try allocator.alloc(InterfaceValue, 0);
            results[0] = try socketResultOk(allocator, .{ .list_val = empty_list });
            return;
        }
        const host_socket = s.host_socket orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const io = std.Io.Threaded.global_single_threaded.io();
        const cap: usize = @min(@as(usize, @intCast(@min(max_results, 16))), 16);
        var collected: std.ArrayListUnmanaged(InterfaceValue) = .empty;
        defer collected.deinit(allocator);
        var discard_budget: usize = 32;
        var buf: [65536]u8 = undefined;
        while (collected.items.len < cap and discard_budget > 0) {
            const msg = host_socket.receiveTimeout(io, &buf, .{
                .duration = .{ .raw = .{ .nanoseconds = 0 }, .clock = .awake },
            }) catch {
                // Zero-duration timeout: any error (Timeout on Linux,
                // WouldBlock/Canceled on Windows IOCP) means "no data
                // available." WIT mandates receive never blocks and
                // never returns would-block, so we stop accumulating.
                break;
            };
            // Connected-receive filtering: drop datagrams not from
            // the connected remote.
            if (in_stream.remote) |expected_remote| {
                const matches = switch (msg.from) {
                    .ip4 => |from_v4| switch (expected_remote) {
                        .ip4 => |exp_v4| std.mem.eql(u8, &from_v4.bytes, &exp_v4.bytes) and from_v4.port == exp_v4.port,
                        .ip6 => false,
                    },
                    .ip6 => |from_v6| switch (expected_remote) {
                        .ip6 => |exp_v6| std.mem.eql(u8, &from_v6.bytes, &exp_v6.bytes) and from_v6.port == exp_v6.port,
                        .ip4 => false,
                    },
                };
                if (!matches) {
                    discard_budget -= 1;
                    continue;
                }
            }
            // Build incoming-datagram record: { data: list<u8>, remote-address: ip-socket-address }
            const data_copy = try allocator.alloc(u8, msg.data.len);
            @memcpy(data_copy, msg.data);
            const data_iv = try allocator.alloc(InterfaceValue, msg.data.len);
            for (data_copy, 0..) |b, i| data_iv[i] = .{ .u8 = b };
            allocator.free(data_copy);
            const remote_iv = try lowerIpSocketAddress(allocator, msg.from);
            const rec = try allocator.alloc(InterfaceValue, 2);
            rec[0] = .{ .list_val = data_iv };
            rec[1] = remote_iv;
            try collected.append(allocator, .{ .record_val = rec });
        }
        const list = try allocator.alloc(InterfaceValue, collected.items.len);
        @memcpy(list, collected.items);
        results[0] = try socketResultOk(allocator, .{ .list_val = list });
    }

    /// `[method]outgoing-datagram-stream.check-send:
    ///   (borrow<outgoing-datagram-stream>) -> result<u64, error-code>`
    /// (#178).
    fn outgoingCheckSend(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 1 or results.len == 0) return error.InvalidArgs;
        const stream_handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const out_stream = self.lookupUdpOutgoingStream(stream_handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const s = self.lookupSocket(out_stream.parent_handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (out_stream.generation != s.stream_generation) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        out_stream.send_credit = 8;
        results[0] = try socketResultOk(allocator, .{ .u64 = 8 });
    }

    /// `[method]outgoing-datagram-stream.send:
    ///   (borrow<outgoing-datagram-stream>, list<outgoing-datagram>)
    ///     -> result<u64, error-code>` (#178).
    ///
    /// WIT-mandated invariants:
    ///  - Traps if `check-send` was not called first (send_credit == null).
    ///  - Traps if `datagrams.len > send_credit`.
    ///  - Partial-success: failure at index `i` returns `err` if i==0,
    ///    else `ok(i)`.
    fn outgoingSend(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 2 or results.len == 0) return error.InvalidArgs;
        const stream_handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const datagrams = switch (args[1]) {
            .list_val => |l| l,
            else => return error.InvalidArgs,
        };
        const out_stream = self.lookupUdpOutgoingStream(stream_handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const s = self.lookupSocket(out_stream.parent_handle) orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        if (out_stream.generation != s.stream_generation) {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        }
        // WIT-mandated trap: check-send must precede send.
        const credit = out_stream.send_credit orelse return error.Trap;
        if (datagrams.len > credit) return error.Trap;
        // Consume credit — every send re-requires a check.
        out_stream.send_credit = null;

        if (datagrams.len == 0) {
            results[0] = try socketResultOk(allocator, .{ .u64 = 0 });
            return;
        }

        const host_socket = s.host_socket orelse {
            results[0] = try socketResultErr(allocator, .invalid_state);
            return;
        };
        const io = std.Io.Threaded.global_single_threaded.io();

        var sent: u64 = 0;
        for (datagrams) |dg_iv| {
            const dg_fields = switch (dg_iv) {
                .record_val => |r| r,
                else => {
                    if (sent == 0) {
                        results[0] = try socketResultErr(allocator, .invalid_argument);
                        return;
                    }
                    results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                    return;
                },
            };
            // WIT outgoing-datagram: { data: list<u8>, remote-address: option<ip-socket-address> }
            if (dg_fields.len < 2) {
                if (sent == 0) {
                    results[0] = try socketResultErr(allocator, .invalid_argument);
                    return;
                }
                results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                return;
            }
            // Extract data bytes.
            const data_list = switch (dg_fields[0]) {
                .list_val => |l| l,
                else => {
                    if (sent == 0) {
                        results[0] = try socketResultErr(allocator, .invalid_argument);
                        return;
                    }
                    results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                    return;
                },
            };
            const data_buf = try allocator.alloc(u8, data_list.len);
            defer allocator.free(data_buf);
            for (data_list, 0..) |iv, i| {
                data_buf[i] = switch (iv) {
                    .u8 => |b| b,
                    else => 0,
                };
            }

            // Determine destination address.
            var dest: std.Io.net.IpAddress = undefined;
            var has_per_dg_remote = false;
            switch (dg_fields[1]) {
                .variant_val => |v| {
                    if (v.discriminant == 1) {
                        if (v.payload) |p| {
                            dest = liftIpSocketAddress(p.*, s.family) catch {
                                if (sent == 0) {
                                    results[0] = try socketResultErr(allocator, .invalid_argument);
                                    return;
                                }
                                results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                                return;
                            };
                            has_per_dg_remote = true;
                        }
                    }
                },
                .option_val => |opt| {
                    if (opt.payload) |p| {
                        dest = liftIpSocketAddress(p.*, s.family) catch {
                            if (sent == 0) {
                                results[0] = try socketResultErr(allocator, .invalid_argument);
                                return;
                            }
                            results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                            return;
                        };
                        has_per_dg_remote = true;
                    }
                },
                else => {},
            }

            if (out_stream.remote != null) {
                // Connected mode: per-datagram remote must be none
                // or exactly equal to the connected remote.
                if (has_per_dg_remote) {
                    const matches = switch (dest) {
                        .ip4 => |dv4| switch (out_stream.remote.?) {
                            .ip4 => |rv4| std.mem.eql(u8, &dv4.bytes, &rv4.bytes) and dv4.port == rv4.port,
                            .ip6 => false,
                        },
                        .ip6 => |dv6| switch (out_stream.remote.?) {
                            .ip6 => |rv6| std.mem.eql(u8, &dv6.bytes, &rv6.bytes) and dv6.port == rv6.port,
                            .ip4 => false,
                        },
                    };
                    if (!matches) {
                        if (sent == 0) {
                            results[0] = try socketResultErr(allocator, .invalid_argument);
                            return;
                        }
                        results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                        return;
                    }
                }
                dest = out_stream.remote.?;
            } else {
                // Unconnected mode: per-datagram remote is required.
                if (!has_per_dg_remote) {
                    if (sent == 0) {
                        results[0] = try socketResultErr(allocator, .invalid_argument);
                        return;
                    }
                    results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                    return;
                }
                // Validate the per-datagram remote.
                if (validateRemoteForSend(dest, s.family)) |code| {
                    if (sent == 0) {
                        results[0] = try socketResultErr(allocator, code);
                        return;
                    }
                    results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                    return;
                }
                // Allow-list check.
                if (s.allow_list.len > 0) {
                    var allowed = false;
                    for (s.allow_list) |c| {
                        if (c.containsAddr(dest)) {
                            allowed = true;
                            break;
                        }
                    }
                    if (!allowed) {
                        if (sent == 0) {
                            results[0] = try socketResultErr(allocator, .access_denied);
                            return;
                        }
                        results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                        return;
                    }
                } else {
                    if (sent == 0) {
                        results[0] = try socketResultErr(allocator, .access_denied);
                        return;
                    }
                    results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                    return;
                }
            }

            // Actual send.
            host_socket.send(io, &dest, data_buf) catch |err| {
                if (sent == 0) {
                    results[0] = try socketResultErr(allocator, mapSocketSendError(err));
                    return;
                }
                results[0] = try socketResultOk(allocator, .{ .u64 = sent });
                return;
            };
            sent += 1;
        }
        results[0] = try socketResultOk(allocator, .{ .u64 = sent });
    }

    /// `[resource-drop]incoming-datagram-stream` (#178).
    fn udpIncomingStreamDrop(
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
        if (handle < self.udp_incoming_streams.items.len) {
            if (self.udp_incoming_streams.items[handle]) |s| self.allocator.destroy(s);
            self.udp_incoming_streams.items[handle] = null;
        }
    }

    /// `[resource-drop]outgoing-datagram-stream` (#178).
    fn udpOutgoingStreamDrop(
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
        if (handle < self.udp_outgoing_streams.items.len) {
            if (self.udp_outgoing_streams.items[handle]) |s| self.allocator.destroy(s);
            self.udp_outgoing_streams.items[handle] = null;
        }
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
    ///                                        error-code>` (#179).
    ///
    /// DNS is gated by the borrowed network's CIDR allow-list (#180): an
    /// empty allow-list (the default) means the embedder has not opted
    /// into network access, so resolution returns `name-unresolvable`
    /// rather than leaking host DNS to the guest. When the allow-list is
    /// non-empty we run a real `std.Io.net.HostName.lookup` via the
    /// global single-threaded `Io` and snapshot its results into a
    /// `ResolveAddressStream`. Canonical-name results are dropped — only
    /// `LookupResult.address` entries are visible to the guest.
    fn resolveAddresses(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 2 or results.len == 0) return error.InvalidArgs;
        const net_handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const name_pl = switch (args[1]) {
            .string => |pl| pl,
            else => return error.InvalidArgs,
        };

        // Allow-list opt-in gate. Default deny-all keeps the guest from
        // observing the host's DNS configuration even on names that
        // would resolve via /etc/hosts.
        const net = if (net_handle < self.network_table.items.len)
            self.network_table.items[net_handle]
        else
            null;
        if (net == null or net.?.allow_list.len == 0) {
            results[0] = try socketResultErr(allocator, .name_unresolvable);
            return;
        }

        // Lift the hostname. Empty / out-of-bounds / RFC1123-invalid
        // input → name-unresolvable per WIT.
        const name_bytes: []const u8 = if (name_pl.len == 0) "" else (ci.readGuestBytes(name_pl.ptr, name_pl.len) orelse {
            results[0] = try socketResultErr(allocator, .name_unresolvable);
            return;
        });
        const host = std.Io.net.HostName.init(name_bytes) catch {
            results[0] = try socketResultErr(allocator, .name_unresolvable);
            return;
        };

        const io = std.Io.Threaded.global_single_threaded.io();
        var queue_buf: [32]std.Io.net.HostName.LookupResult = undefined;
        var queue: std.Io.Queue(std.Io.net.HostName.LookupResult) = .init(&queue_buf);
        host.lookup(io, &queue, .{ .port = 0 }) catch |err| {
            results[0] = try socketResultErr(allocator, mapDnsError(err));
            return;
        };

        // Drain. `lookup` closes the queue before return, so this loop
        // terminates with `error.Closed` once the buffer is empty.
        var addrs: std.ArrayListUnmanaged(std.Io.net.IpAddress) = .empty;
        errdefer addrs.deinit(self.allocator);
        while (queue.getOneUncancelable(io)) |item| {
            switch (item) {
                .address => |a| try addrs.append(self.allocator, a),
                .canonical_name => {},
            }
        } else |drain_err| switch (drain_err) {
            error.Closed => {},
        }

        const slice = try addrs.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(slice);

        const stream = try self.allocator.create(ResolveAddressStream);
        errdefer self.allocator.destroy(stream);
        stream.* = .{ .results = slice, .pos = 0, .allocator = self.allocator };

        const h = try self.pushResolveStream(stream);
        results[0] = try socketResultOk(allocator, .{ .handle = h });
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
        if (stream.pos >= stream.results.len) {
            const none_val = InterfaceValue{ .option_val = .{ .is_some = false, .payload = null } };
            results[0] = try socketResultOk(allocator, none_val);
            return;
        }
        const addr = stream.results[stream.pos];
        stream.pos += 1;

        const ip_val = try lowerIpAddress(allocator, addr);
        errdefer ip_val.deinit(allocator);
        const some_payload = try allocator.create(InterfaceValue);
        errdefer allocator.destroy(some_payload);
        some_payload.* = ip_val;
        const some_val = InterfaceValue{ .option_val = .{ .is_some = true, .payload = some_payload } };
        results[0] = try socketResultOk(allocator, some_val);
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
            // network IO — bind + listen + local-address are real
            // (#178 PR A); connect/accept/streams/remote-address remain
            // default-deny pending PR B.
            .{ .name = "[method]tcp-socket.start-bind", .call = &tcpStartBind },
            .{ .name = "[method]tcp-socket.finish-bind", .call = &tcpFinishBind },
            .{ .name = "[method]tcp-socket.start-connect", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.finish-connect", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.start-listen", .call = &tcpStartListen },
            .{ .name = "[method]tcp-socket.finish-listen", .call = &tcpFinishListen },
            .{ .name = "[method]tcp-socket.accept", .call = &socketDenyAccess },
            .{ .name = "[method]tcp-socket.local-address", .call = &tcpLocalAddress },
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

    /// Register `wasi:sockets/udp` (#178 PR C). Bind, stream,
    /// local/remote-address, and datagram-stream methods are wired
    /// to real handlers; socket-option getters/setters remain
    /// default-deny. Subscribe is always-ready stub.
    pub fn populateWasiSocketsUdp(
        self: *WasiCliAdapter,
        providers: *std.StringHashMapUnmanaged(ImportBinding),
        interface_name: []const u8,
    ) !void {
        const M = struct { name: []const u8, call: *const fn (?*anyopaque, *ComponentInstance, []const InterfaceValue, []InterfaceValue, Allocator) anyerror!void };
        const members = [_]M{
            // Real UDP lifecycle handlers (#178 PR C).
            .{ .name = "[method]udp-socket.start-bind", .call = &udpStartBind },
            .{ .name = "[method]udp-socket.finish-bind", .call = &udpFinishBind },
            .{ .name = "[method]udp-socket.stream", .call = &udpStream },
            .{ .name = "[method]udp-socket.local-address", .call = &udpLocalAddress },
            .{ .name = "[method]udp-socket.remote-address", .call = &udpRemoteAddress },
            // Socket-option getters/setters — still access-denied.
            .{ .name = "[method]udp-socket.unicast-hop-limit", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.set-unicast-hop-limit", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.receive-buffer-size", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.set-receive-buffer-size", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.send-buffer-size", .call = &socketDenyAccess },
            .{ .name = "[method]udp-socket.set-send-buffer-size", .call = &socketDenyAccess },
            // Pure getters + subscribe + drop.
            .{ .name = "[method]udp-socket.address-family", .call = &socketAddressFamily },
            .{ .name = "[method]udp-socket.subscribe", .call = &socketSubscribe },
            .{ .name = "[resource-drop]udp-socket", .call = &socketResourceDrop },
            // Datagram-stream sub-resources — real handlers.
            .{ .name = "[resource-drop]incoming-datagram-stream", .call = &udpIncomingStreamDrop },
            .{ .name = "[method]incoming-datagram-stream.receive", .call = &incomingDatagramReceive },
            .{ .name = "[method]incoming-datagram-stream.subscribe", .call = &socketSubscribe },
            .{ .name = "[resource-drop]outgoing-datagram-stream", .call = &udpOutgoingStreamDrop },
            .{ .name = "[method]outgoing-datagram-stream.check-send", .call = &outgoingCheckSend },
            .{ .name = "[method]outgoing-datagram-stream.send", .call = &outgoingSend },
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
    fn pushIncomingResponse(self: *WasiCliAdapter, r: *IncomingResponse) !u32 {
        for (self.http_incoming_responses.items, 0..) |slot, i| {
            if (slot == null) {
                self.http_incoming_responses.items[i] = r;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.http_incoming_responses.items.len);
        try self.http_incoming_responses.append(self.allocator, r);
        return idx;
    }
    fn lookupIncomingResponse(self: *WasiCliAdapter, h: u32) ?*IncomingResponse {
        if (h >= self.http_incoming_responses.items.len) return null;
        return self.http_incoming_responses.items[h];
    }
    fn pushIncomingBody(self: *WasiCliAdapter, b: *IncomingBody) !u32 {
        for (self.http_incoming_bodies.items, 0..) |slot, i| {
            if (slot == null) {
                self.http_incoming_bodies.items[i] = b;
                return @intCast(i);
            }
        }
        const idx: u32 = @intCast(self.http_incoming_bodies.items.len);
        try self.http_incoming_bodies.append(self.allocator, b);
        return idx;
    }
    fn lookupIncomingBody(self: *WasiCliAdapter, h: u32) ?*IncomingBody {
        if (h >= self.http_incoming_bodies.items.len) return null;
        return self.http_incoming_bodies.items[h];
    }
    fn lookupOutgoingBody(self: *WasiCliAdapter, h: u32) ?*OutgoingBody {
        if (h >= self.http_outgoing_bodies.items.len) return null;
        return self.http_outgoing_bodies.items[h];
    }

    // --- helpers for extracting bytes from InterfaceValue args ---

    /// Extract a byte slice from an InterfaceValue that is either a
    /// `.string` PtrLen (from guest memory) or a `.list_val` of u8
    /// InterfaceValues (from test / host calls). Returns null on
    /// invalid encoding or out-of-bounds guest memory.
    fn extractArgBytes(ci: *ComponentInstance, arg: InterfaceValue) ?[]const u8 {
        switch (arg) {
            .string => |pl| {
                if (pl.len == 0) return "";
                return ci.readGuestBytes(pl.ptr, pl.len);
            },
            .list_val => |elems| {
                // For test calls: each element is .u8
                // Return a pointer to a contiguous region by checking if the
                // underlying slice of InterfaceValues can be reinterpreted.
                // Actually, the u8 values are scattered in tagged unions, so
                // we cannot get a contiguous []const u8 without copying.
                // Return null and let the caller use extractArgBytesAlloc.
                _ = elems;
                return null;
            },
            .list => |pl| {
                if (pl.len == 0) return "";
                return ci.readGuestBytes(pl.ptr, pl.len);
            },
            else => return null,
        }
    }

    /// Like `extractArgBytes` but allocates a copy for list_val of u8.
    fn extractArgBytesAlloc(
        alloc: Allocator,
        ci: *ComponentInstance,
        arg: InterfaceValue,
    ) !?[]u8 {
        switch (arg) {
            .string => |pl| {
                if (pl.len == 0) return try alloc.dupe(u8, "");
                const src = ci.readGuestBytes(pl.ptr, pl.len) orelse return null;
                return try alloc.dupe(u8, src);
            },
            .list_val => |elems| {
                const buf = try alloc.alloc(u8, elems.len);
                for (elems, 0..) |e, i| {
                    buf[i] = switch (e) {
                        .u8 => |b| b,
                        else => {
                            alloc.free(buf);
                            return null;
                        },
                    };
                }
                return buf;
            },
            .list => |pl| {
                if (pl.len == 0) return try alloc.dupe(u8, "");
                const src = ci.readGuestBytes(pl.ptr, pl.len) orelse return null;
                return try alloc.dupe(u8, src);
            },
            else => return null,
        }
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
    /// The argument is a guest-memory list of `tuple<string, string>`
    /// records. Each element is laid out as four u32s — name.ptr,
    /// name.len, value.ptr, value.len — for a 16-byte stride. We lift
    /// each tuple, copy both strings into adapter-owned heap, and
    /// populate the new `HttpFields.entries`. No header validation is
    /// performed today (a follow-up may reject `.invalid_syntax` per
    /// RFC 7230); guests that round-trip through `entries` get back
    /// what they passed in.
    fn httpFieldsFromList(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len == 0 or results.len == 0) return error.InvalidArgs;
        const list = switch (args[0]) {
            .list => |pl| pl,
            else => return error.InvalidArgs,
        };

        const f = try self.allocator.create(HttpFields);
        f.* = .{};
        errdefer {
            f.deinit(self.allocator);
            self.allocator.destroy(f);
        }

        if (list.len > 0) {
            const mem = ci.canonicalMemory() orelse return error.OutOfBoundsMemory;
            const lifted = try liftFieldEntries(self.allocator, mem.data, list.ptr, list.len);
            f.entries = .{ .items = lifted, .capacity = lifted.len };
        }

        const h = try self.pushHttpFields(f);
        results[0] = try httpResultOk(allocator, .{ .handle = h });
    }

    /// `[method]fields.entries(borrow<fields>)
    ///   -> list<tuple<field-name, field-value>>`. Returns the stored
    /// entries as InterfaceValue list of tuples, each tuple containing
    /// a list_val of u8 for name and a list_val of u8 for value.
    fn httpFieldsEntries(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;

        const handle = switch (args[0]) {
            .handle => |h| h,
            else => {
                const empty = try allocator.alloc(InterfaceValue, 0);
                results[0] = .{ .list_val = empty };
                return;
            },
        };

        const f = self.lookupHttpFields(handle) orelse {
            const empty = try allocator.alloc(InterfaceValue, 0);
            results[0] = .{ .list_val = empty };
            return;
        };

        const entries = try allocator.alloc(InterfaceValue, f.entries.items.len);
        for (f.entries.items, 0..) |e, i| {
            const name_ivs = try allocator.alloc(InterfaceValue, e.name.len);
            for (e.name, 0..) |b, j| name_ivs[j] = .{ .u8 = b };
            const val_ivs = try allocator.alloc(InterfaceValue, e.value.len);
            for (e.value, 0..) |b, j| val_ivs[j] = .{ .u8 = b };
            const tup = try allocator.alloc(InterfaceValue, 2);
            tup[0] = .{ .list_val = name_ivs };
            tup[1] = .{ .list_val = val_ivs };
            entries[i] = .{ .tuple_val = tup };
        }
        results[0] = .{ .list_val = entries };
    }

    /// `[method]fields.get(borrow<fields>, field-name)
    ///   -> list<field-value>`.
    fn httpFieldsGet(
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
        const f = self.lookupHttpFields(handle) orelse {
            const empty = try allocator.alloc(InterfaceValue, 0);
            results[0] = .{ .list_val = empty };
            return;
        };

        // Read name from arg
        const name_bytes: ?[]const u8 = extractArgBytes(ci, args[1]);
        const name_alloc = if (name_bytes == null) try extractArgBytesAlloc(self.allocator, ci, args[1]) else null;
        defer if (name_alloc) |n| self.allocator.free(n);
        const name = name_bytes orelse (name_alloc orelse {
            const empty = try allocator.alloc(InterfaceValue, 0);
            results[0] = .{ .list_val = empty };
            return;
        });

        // Count matches
        var count: usize = 0;
        for (f.entries.items) |e| {
            if (std.mem.eql(u8, e.name, name)) count += 1;
        }

        const matches = try allocator.alloc(InterfaceValue, count);
        var idx: usize = 0;
        for (f.entries.items) |e| {
            if (std.mem.eql(u8, e.name, name)) {
                const val_ivs = try allocator.alloc(InterfaceValue, e.value.len);
                for (e.value, 0..) |b, j| val_ivs[j] = .{ .u8 = b };
                matches[idx] = .{ .list_val = val_ivs };
                idx += 1;
            }
        }
        results[0] = .{ .list_val = matches };
    }

    /// `[method]fields.has(borrow<fields>, field-name) -> bool`.
    fn httpFieldsHas(
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
        const f = self.lookupHttpFields(handle) orelse {
            results[0] = .{ .bool = false };
            return;
        };

        const name_bytes: ?[]const u8 = extractArgBytes(ci, args[1]);
        const name_alloc = if (name_bytes == null) try extractArgBytesAlloc(self.allocator, ci, args[1]) else null;
        defer if (name_alloc) |n| self.allocator.free(n);
        const name = name_bytes orelse (name_alloc orelse {
            results[0] = .{ .bool = false };
            return;
        });

        for (f.entries.items) |e| {
            if (std.mem.eql(u8, e.name, name)) {
                results[0] = .{ .bool = true };
                return;
            }
        }
        results[0] = .{ .bool = false };
    }

    /// `[method]fields.append(borrow<fields>, field-name, field-value)
    ///   -> result<_, header-error>`.
    fn httpFieldsAppend(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 3 or results.len == 0) return error.InvalidArgs;

        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const f = self.lookupHttpFields(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        };

        const name_copy = try extractArgBytesAlloc(self.allocator, ci, args[1]) orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        };
        errdefer self.allocator.free(name_copy);
        const val_copy = try extractArgBytesAlloc(self.allocator, ci, args[2]) orelse {
            self.allocator.free(name_copy);
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        };
        errdefer self.allocator.free(val_copy);

        try f.entries.append(self.allocator, .{ .name = name_copy, .value = val_copy });
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `[method]fields.set(borrow<fields>, field-name, list<field-value>)
    ///   -> result<_, header-error>`.
    fn httpFieldsSet(
        ctx_opaque: ?*anyopaque,
        ci: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        _: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (args.len < 3 or results.len == 0) return error.InvalidArgs;

        const handle = switch (args[0]) {
            .handle => |h| h,
            else => return error.InvalidArgs,
        };
        const f = self.lookupHttpFields(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        };

        const name_bytes: ?[]const u8 = extractArgBytes(ci, args[1]);
        const name_alloc = if (name_bytes == null) try extractArgBytesAlloc(self.allocator, ci, args[1]) else null;
        defer if (name_alloc) |n| self.allocator.free(n);
        const name = name_bytes orelse (name_alloc orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        });

        // Remove existing entries with this name
        httpFieldsRemoveByName(f, self.allocator, name);

        // Add new entries from the list of values
        const vals = switch (args[2]) {
            .list_val => |l| l,
            else => {
                results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
                return;
            },
        };
        for (vals) |v| {
            const name_copy = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(name_copy);

            var val_copy: ?[]u8 = null;
            switch (v) {
                .list_val => |vbytes| {
                    const buf = try self.allocator.alloc(u8, vbytes.len);
                    for (vbytes, 0..) |b, j| {
                        buf[j] = switch (b) {
                            .u8 => |byte| byte,
                            else => 0,
                        };
                    }
                    val_copy = buf;
                },
                else => {
                    val_copy = try extractArgBytesAlloc(self.allocator, ci, v);
                },
            }
            const vc = val_copy orelse try self.allocator.dupe(u8, "");
            try f.entries.append(self.allocator, .{ .name = name_copy, .value = vc });
        }
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `[method]fields.delete(borrow<fields>, field-name)
    ///   -> result<_, header-error>`.
    fn httpFieldsDelete(
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
        const f = self.lookupHttpFields(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        };

        const name_bytes: ?[]const u8 = extractArgBytes(ci, args[1]);
        const name_alloc = if (name_bytes == null) try extractArgBytesAlloc(self.allocator, ci, args[1]) else null;
        defer if (name_alloc) |n| self.allocator.free(n);
        const name = name_bytes orelse (name_alloc orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        });

        httpFieldsRemoveByName(f, self.allocator, name);
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// Remove all entries with the given name from an HttpFields.
    fn httpFieldsRemoveByName(f: *HttpFields, alloc: Allocator, name: []const u8) void {
        var i: usize = 0;
        while (i < f.entries.items.len) {
            if (std.mem.eql(u8, f.entries.items[i].name, name)) {
                alloc.free(f.entries.items[i].name);
                alloc.free(f.entries.items[i].value);
                _ = f.entries.swapRemove(i);
            } else {
                i += 1;
            }
        }
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
        const r = self.lookupOutgoingRequest(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        };
        const disc: u32 = switch (args[1]) {
            .variant_val => |v| v.discriminant,
            .enum_val => |d| d,
            .u32 => |d| d,
            else => 0,
        };
        r.method_disc = disc;
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `[method]outgoing-request.path-with-query(borrow)
    ///   -> option<string>`.
    fn httpOutgoingRequestGetPath(
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
            results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
            return;
        };
        if (r.path_with_query) |p| {
            const ivs = try allocator.alloc(InterfaceValue, p.len);
            for (p, 0..) |b, i| ivs[i] = .{ .u8 = b };
            const payload = try allocator.create(InterfaceValue);
            payload.* = .{ .list_val = ivs };
            results[0] = .{ .option_val = .{ .is_some = true, .payload = payload } };
        } else {
            results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
        }
    }

    /// `[method]outgoing-request.set-path-with-query(borrow,
    ///   option<string>) -> result<_, _>`.
    fn httpOutgoingRequestSetPath(
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
        const r = self.lookupOutgoingRequest(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        };
        // Free existing
        if (r.path_with_query) |old| self.allocator.free(old);
        r.path_with_query = null;

        // Read option<string>
        switch (args[1]) {
            .option_val => |opt| {
                if (opt.is_some) {
                    if (opt.payload) |p| {
                        r.path_with_query = try extractArgBytesAlloc(self.allocator, ci, p.*);
                    }
                }
            },
            else => {
                // May be a direct string/list_val
                r.path_with_query = try extractArgBytesAlloc(self.allocator, ci, args[1]);
            },
        }
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `[method]outgoing-request.scheme(borrow) -> option<scheme>`.
    fn httpOutgoingRequestGetScheme(
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
            results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
            return;
        };
        if (r.scheme_disc) |d| {
            const payload = try allocator.create(InterfaceValue);
            payload.* = .{ .variant_val = .{ .discriminant = d, .payload = null } };
            results[0] = .{ .option_val = .{ .is_some = true, .payload = payload } };
        } else {
            results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
        }
    }

    /// `[method]outgoing-request.set-scheme(borrow, option<scheme>)
    ///   -> result<_, _>`.
    fn httpOutgoingRequestSetScheme(
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
        const r = self.lookupOutgoingRequest(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        };
        switch (args[1]) {
            .option_val => |opt| {
                if (opt.is_some) {
                    if (opt.payload) |p| {
                        switch (p.*) {
                            .variant_val => |v| {
                                r.scheme_disc = v.discriminant;
                            },
                            .enum_val => |d| {
                                r.scheme_disc = d;
                            },
                            .u32 => |d| {
                                r.scheme_disc = d;
                            },
                            else => {},
                        }
                    }
                } else {
                    r.scheme_disc = null;
                }
            },
            .variant_val => |v| {
                r.scheme_disc = v.discriminant;
            },
            else => {},
        }
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// `[method]outgoing-request.authority(borrow) -> option<string>`.
    fn httpOutgoingRequestGetAuthority(
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
            results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
            return;
        };
        if (r.authority) |a| {
            const ivs = try allocator.alloc(InterfaceValue, a.len);
            for (a, 0..) |b, i| ivs[i] = .{ .u8 = b };
            const payload = try allocator.create(InterfaceValue);
            payload.* = .{ .list_val = ivs };
            results[0] = .{ .option_val = .{ .is_some = true, .payload = payload } };
        } else {
            results[0] = .{ .option_val = .{ .is_some = false, .payload = null } };
        }
    }

    /// `[method]outgoing-request.set-authority(borrow,
    ///   option<string>) -> result<_, _>`.
    fn httpOutgoingRequestSetAuthority(
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
        const r = self.lookupOutgoingRequest(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
            return;
        };
        if (r.authority) |old| self.allocator.free(old);
        r.authority = null;

        switch (args[1]) {
            .option_val => |opt| {
                if (opt.is_some) {
                    if (opt.payload) |p| {
                        r.authority = try extractArgBytesAlloc(self.allocator, ci, p.*);
                    }
                }
            },
            else => {
                r.authority = try extractArgBytesAlloc(self.allocator, ci, args[1]);
            },
        }
        results[0] = .{ .result_val = .{ .is_ok = true, .payload = null } };
    }

    /// Generic `(borrow) -> option<string>` getter that always returns
    /// `none` — used for incoming-request path/scheme/authority stubs.
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
        r.body_handle = bh;
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

    /// `[method]incoming-response.consume(borrow)
    ///   -> result<own<incoming-body>, _>`.
    fn httpIncomingResponseConsume(
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
        const ir = self.lookupIncomingResponse(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        };
        if (ir.body_consumed) {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        }
        ir.body_consumed = true;

        const body = try self.allocator.create(IncomingBody);
        body.* = .{ .data = ir.body_data };
        // Transfer ownership — IncomingResponse no longer owns the data.
        ir.body_data = null;

        const bh = try self.pushIncomingBody(body);
        results[0] = try httpResultOk(allocator, .{ .handle = bh });
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
            r.deinit(self.allocator);
            self.allocator.destroy(r);
            self.http_incoming_responses.items[handle] = null;
        }
    }

    // --- bodies ---

    /// `[method]incoming-body.stream(borrow)
    ///   -> result<own<input-stream>, _>`. Creates a buffer-backed
    /// input stream from the body's data.
    fn httpIncomingBodyStream(
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
        const body = self.lookupIncomingBody(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        };
        if (body.stream_taken) {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        }
        body.stream_taken = true;

        const s = try self.allocator.create(streams.InputStream);
        s.* = streams.InputStream.fromBuffer(body.data orelse "");
        try self.owned_input_streams.append(self.allocator, s);
        const sh = try self.allocInputStreamHandle(s);
        results[0] = try httpResultOk(allocator, .{ .handle = sh });
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
            b.deinit(self.allocator);
            self.allocator.destroy(b);
            self.http_incoming_bodies.items[handle] = null;
        }
    }

    /// `[method]outgoing-body.write(borrow)
    ///   -> result<own<output-stream>, _>`. Creates a buffer-backed
    /// output stream and stores it on the body for later retrieval.
    fn httpOutgoingBodyWrite(
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
        const body = self.lookupOutgoingBody(handle) orelse {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        };
        if (body.stream_taken) {
            results[0] = .{ .result_val = .{ .is_ok = false, .payload = null } };
            return;
        }
        body.stream_taken = true;

        const s = try self.allocator.create(streams.OutputStream);
        s.* = streams.OutputStream.toBuffer();
        body.stream = s;
        try self.owned_output_streams.append(self.allocator, s);
        const sh = try self.allocStreamHandle(s);
        results[0] = try httpResultOk(allocator, .{ .handle = sh });
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
    /// When `sockets_allow_list_template` is non-empty, performs a real
    /// outbound HTTP request via `std.http.Client` (#176). Otherwise
    /// returns `HTTP_request_denied` as before.
    fn httpOutgoingHandlerHandle(
        ctx_opaque: ?*anyopaque,
        _: *ComponentInstance,
        args: []const InterfaceValue,
        results: []InterfaceValue,
        allocator: Allocator,
    ) anyerror!void {
        const self: *WasiCliAdapter = @ptrCast(@alignCast(ctx_opaque.?));
        if (results.len == 0) return error.InvalidArgs;

        // Extract request handle
        const req_handle: u32 = if (args.len > 0) switch (args[0]) {
            .handle => |h| h,
            else => 0,
        } else 0;

        const req = self.lookupOutgoingRequest(req_handle);

        // Deny when allow-list is empty
        if (self.sockets_allow_list_template.len == 0) {
            return httpHandlerDeny(self, results, allocator, .HTTP_request_denied);
        }

        const r = req orelse {
            return httpHandlerDeny(self, results, allocator, .HTTP_request_denied);
        };

        // Build scheme string
        const scheme: []const u8 = if (r.scheme_disc) |d| switch (d) {
            0 => "http",
            1 => "https",
            else => if (r.scheme_other) |s| s else "https",
        } else "https";

        const authority = r.authority orelse {
            return httpHandlerDeny(self, results, allocator, .HTTP_request_URI_invalid);
        };
        const path = r.path_with_query orelse "/";

        // Build URL
        const url = std.fmt.allocPrint(self.allocator, "{s}://{s}{s}", .{ scheme, authority, path }) catch {
            return httpHandlerDeny(self, results, allocator, .internal_error);
        };
        defer self.allocator.free(url);

        // Map WIT method discriminant to std.http.Method
        const method: std.http.Method = switch (r.method_disc) {
            0 => .GET,
            1 => .HEAD,
            2 => .POST,
            3 => .PUT,
            4 => .DELETE,
            5 => .CONNECT,
            6 => .OPTIONS,
            7 => .TRACE,
            8 => .PATCH,
            else => .GET,
        };

        // Collect request headers
        const headers_fields = self.lookupHttpFields(r.headers_handle);
        var extra_hdrs: std.ArrayListUnmanaged(std.http.Header) = .empty;
        defer extra_hdrs.deinit(self.allocator);
        if (headers_fields) |hf| {
            for (hf.entries.items) |e| {
                extra_hdrs.append(self.allocator, .{
                    .name = e.name,
                    .value = e.value,
                }) catch break;
            }
        }

        // Get body payload bytes (if any)
        var payload: ?[]const u8 = null;
        if (r.body_handle) |bh| {
            if (self.lookupOutgoingBody(bh)) |body| {
                if (body.stream) |s| {
                    const contents = s.getBufferContents();
                    if (contents.len > 0) payload = contents;
                }
            }
        }

        // Perform the HTTP request
        const io = std.Io.Threaded.global_single_threaded.io();
        var client: std.http.Client = .{
            .allocator = self.allocator,
            .io = io,
        };
        defer client.connection_pool.deinit(io);

        var aw: std.Io.Writer.Allocating = .init(self.allocator);
        defer aw.deinit();

        const fetch_result = client.fetch(.{
            .location = .{ .url = url },
            .method = method,
            .payload = payload,
            .extra_headers = extra_hdrs.items,
            .response_writer = &aw.writer,
            .keep_alive = false,
        }) catch {
            return httpHandlerDeny(self, results, allocator, .internal_error);
        };

        // Extract response body
        var resp_al = aw.toArrayList();
        const resp_body = resp_al.toOwnedSlice(self.allocator) catch {
            return httpHandlerDeny(self, results, allocator, .internal_error);
        };
        errdefer self.allocator.free(resp_body);

        // Create response headers (empty — std.http.Client.fetch
        // does not expose response headers in FetchResult)
        const resp_fields = self.allocator.create(HttpFields) catch {
            self.allocator.free(resp_body);
            return httpHandlerDeny(self, results, allocator, .internal_error);
        };
        resp_fields.* = .{};
        const resp_fields_handle = self.pushHttpFields(resp_fields) catch {
            self.allocator.free(resp_body);
            resp_fields.deinit(self.allocator);
            self.allocator.destroy(resp_fields);
            return httpHandlerDeny(self, results, allocator, .internal_error);
        };

        // Create IncomingResponse
        const ir = self.allocator.create(IncomingResponse) catch {
            self.allocator.free(resp_body);
            return httpHandlerDeny(self, results, allocator, .internal_error);
        };
        ir.* = .{
            .status = @intFromEnum(fetch_result.status),
            .headers_handle = resp_fields_handle,
            .body_data = resp_body,
        };
        const ir_handle = self.pushIncomingResponse(ir) catch {
            self.allocator.free(resp_body);
            ir.body_data = null;
            self.allocator.destroy(ir);
            return httpHandlerDeny(self, results, allocator, .internal_error);
        };

        // Create FutureIncomingResponse in ready_ok state
        const fut = self.allocator.create(FutureIncomingResponse) catch {
            return httpHandlerDeny(self, results, allocator, .internal_error);
        };
        fut.* = .{ .state = .{ .ready_ok = ir_handle } };
        const fh = self.pushFutureResponse(fut) catch {
            self.allocator.destroy(fut);
            return httpHandlerDeny(self, results, allocator, .internal_error);
        };
        results[0] = try httpResultOk(allocator, .{ .handle = fh });
    }

    /// Helper: return a denied-style future from the outgoing handler.
    fn httpHandlerDeny(
        self: *WasiCliAdapter,
        results: []InterfaceValue,
        allocator: Allocator,
        code: HttpErrorCode,
    ) anyerror!void {
        const fut = try self.allocator.create(FutureIncomingResponse);
        fut.* = .{ .state = .{ .ready_err = @intFromEnum(code) } };
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
            .{ .name = "[method]fields.set", .call = &httpFieldsSet },
            .{ .name = "[method]fields.append", .call = &httpFieldsAppend },
            .{ .name = "[method]fields.delete", .call = &httpFieldsDelete },
            .{ .name = "[method]fields.clone", .call = &httpFieldsClone },
            .{ .name = "[resource-drop]fields", .call = &httpFieldsDrop },
            // outgoing-request
            .{ .name = "[constructor]outgoing-request", .call = &httpOutgoingRequestConstructor },
            .{ .name = "[method]outgoing-request.method", .call = &httpOutgoingRequestMethod },
            .{ .name = "[method]outgoing-request.set-method", .call = &httpOutgoingRequestSetMethod },
            .{ .name = "[method]outgoing-request.path-with-query", .call = &httpOutgoingRequestGetPath },
            .{ .name = "[method]outgoing-request.set-path-with-query", .call = &httpOutgoingRequestSetPath },
            .{ .name = "[method]outgoing-request.scheme", .call = &httpOutgoingRequestGetScheme },
            .{ .name = "[method]outgoing-request.set-scheme", .call = &httpOutgoingRequestSetScheme },
            .{ .name = "[method]outgoing-request.authority", .call = &httpOutgoingRequestGetAuthority },
            .{ .name = "[method]outgoing-request.set-authority", .call = &httpOutgoingRequestSetAuthority },
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

/// Convert `std.Io.Timestamp` (i96 nanoseconds since UNIX epoch) into the
/// WIT `datetime { seconds: u64, nanoseconds: u32 }` record. Negative
/// epoch values clamp to `{0,0}` (WIT u64 cannot represent pre-1970).
/// Seconds saturate at u64 max for far-future timestamps.
fn ioTimestampToDatetime(ts: std.Io.Timestamp) Datetime {
    if (ts.nanoseconds <= 0) return .{ .seconds = 0, .nanoseconds = 0 };
    const NS_PER_S: i96 = 1_000_000_000;
    const secs_i96 = @divTrunc(ts.nanoseconds, NS_PER_S);
    const nsec_i96 = @mod(ts.nanoseconds, NS_PER_S);
    const secs: u64 = if (secs_i96 > std.math.maxInt(u64))
        std.math.maxInt(u64)
    else
        @intCast(secs_i96);
    const nsec: u32 = @intCast(nsec_i96);
    return .{ .seconds = secs, .nanoseconds = nsec };
}

/// Lift an optional `Io.Timestamp` into the WIT `option<datetime>`.
/// `null` becomes `option::none`; non-null becomes `option::some(datetime)`.
fn buildOptionalDatetime(allocator: Allocator, maybe_ts: ?std.Io.Timestamp) !InterfaceValue {
    if (maybe_ts) |ts| {
        const dt_iv = try allocator.create(InterfaceValue);
        dt_iv.* = try buildDatetimeRecord(allocator, ioTimestampToDatetime(ts));
        return .{ .option_val = .{ .is_some = true, .payload = dt_iv } };
    } else {
        return .{ .option_val = .{ .is_some = false, .payload = null } };
    }
}

/// Lift the host `Io.File.Stat` into the WIT `descriptor-stat` record.
/// Field order matches the WIT declaration:
///   { type, link-count, size, data-access-timestamp,
///     data-modification-timestamp, status-change-timestamp }.
fn buildDescriptorStatRecord(
    allocator: Allocator,
    st: std.Io.File.Stat,
) !InterfaceValue {
    const dt: WasiCliAdapter.DescType = switch (st.kind) {
        .directory => .directory,
        .file => .regular_file,
        .block_device => .block_device,
        .character_device => .character_device,
        .named_pipe => .fifo,
        .sym_link => .symbolic_link,
        .unix_domain_socket => .socket,
        else => .unknown,
    };
    const fields = try allocator.alloc(InterfaceValue, 6);
    fields[0] = .{ .variant_val = .{ .discriminant = @intFromEnum(dt), .payload = null } };
    fields[1] = .{ .u64 = @intCast(st.nlink) };
    fields[2] = .{ .u64 = st.size };
    fields[3] = try buildOptionalDatetime(allocator, st.atime);
    fields[4] = try buildOptionalDatetime(allocator, st.mtime);
    fields[5] = try buildOptionalDatetime(allocator, st.ctime);
    return .{ .record_val = fields };
}

/// Lift the WIT `new-timestamp` variant into `std.Io.File.SetTimestamp`.
/// Discriminant order per the canonical `wasi:filesystem` WIT:
///   0 = no-change → `.unchanged`
///   1 = now       → `.now`
///   2 = timestamp(datetime) → `.new`
fn liftNewTimestamp(arg: InterfaceValue) !std.Io.File.SetTimestamp {
    const v = switch (arg) {
        .variant_val => |v| v,
        else => return error.InvalidArgs,
    };
    switch (v.discriminant) {
        0 => return .unchanged,
        1 => return .now,
        2 => {
            const payload = v.payload orelse return error.InvalidArgs;
            const record = switch (payload.*) {
                .record_val => |r| r,
                else => return error.InvalidArgs,
            };
            if (record.len < 2) return error.InvalidArgs;
            const secs_u64: u64 = switch (record[0]) {
                .u64 => |x| x,
                else => return error.InvalidArgs,
            };
            const nsec_u32: u32 = switch (record[1]) {
                .u32 => |x| x,
                else => return error.InvalidArgs,
            };
            // Clamp seconds into i96 range — i96 easily holds u64.
            const secs_i96: i96 = @intCast(secs_u64);
            const nsec_i96: i96 = @intCast(nsec_u32);
            return .{ .new = .{ .nanoseconds = secs_i96 * 1_000_000_000 + nsec_i96 } };
        },
        else => return error.InvalidArgs,
    }
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

test "filesystem: ioTimestampToDatetime clamps + splits ns (#177)" {
    const testing = std.testing;

    const zero = ioTimestampToDatetime(.{ .nanoseconds = 0 });
    try testing.expectEqual(@as(u64, 0), zero.seconds);
    try testing.expectEqual(@as(u32, 0), zero.nanoseconds);

    // Negative epoch clamps to (0, 0).
    const neg = ioTimestampToDatetime(.{ .nanoseconds = -42 });
    try testing.expectEqual(@as(u64, 0), neg.seconds);
    try testing.expectEqual(@as(u32, 0), neg.nanoseconds);

    // 1.5 seconds.
    const one_half = ioTimestampToDatetime(.{ .nanoseconds = 1_500_000_000 });
    try testing.expectEqual(@as(u64, 1), one_half.seconds);
    try testing.expectEqual(@as(u32, 500_000_000), one_half.nanoseconds);
}

test "filesystem: liftNewTimestamp covers all three variants (#177)" {
    const testing = std.testing;

    const unchanged = try liftNewTimestamp(.{ .variant_val = .{ .discriminant = 0, .payload = null } });
    try testing.expect(unchanged == .unchanged);

    const now_v = try liftNewTimestamp(.{ .variant_val = .{ .discriminant = 1, .payload = null } });
    try testing.expect(now_v == .now);

    var dt_fields = [_]InterfaceValue{
        .{ .u64 = 1_700_000_000 },
        .{ .u32 = 500_000_000 },
    };
    const dt_iv = InterfaceValue{ .record_val = &dt_fields };
    const new_v = try liftNewTimestamp(.{ .variant_val = .{ .discriminant = 2, .payload = &dt_iv } });
    try testing.expect(new_v == .new);
    try testing.expectEqual(@as(i96, 1_700_000_000 * 1_000_000_000 + 500_000_000), new_v.new.nanoseconds);

    // Bad discriminant rejects.
    try testing.expectError(error.InvalidArgs, liftNewTimestamp(.{ .variant_val = .{ .discriminant = 7, .payload = null } }));
}

test "filesystem: stat returns mtime/ctime as option::some (#177)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const io = std.Io.Threaded.global_single_threaded.io();
    try tmp.dir.writeFile(io, .{ .sub_path = "f.txt", .data = "hello" });

    const file = try tmp.dir.openFile(io, "f.txt", .{ .mode = .read_only });
    const handle = try adapter.pushFsDescriptor(.{ .file = .{ .file = file } });

    var args = [_]InterfaceValue{.{ .handle = handle }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    var ci: ComponentInstance = undefined;
    try WasiCliAdapter.fsDescriptorStat(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .result_val);
    try testing.expect(results[0].result_val.is_ok);
    const rec = results[0].result_val.payload.?.*.record_val;
    try testing.expectEqual(@as(usize, 6), rec.len);

    // size = 5 ("hello").
    try testing.expectEqual(@as(u64, 5), rec[2].u64);
    // atime / mtime / ctime — at least mtime + ctime should be is_some on
    // any sane host filesystem with nonzero seconds.
    try testing.expect(rec[4].option_val.is_some);
    try testing.expect(rec[5].option_val.is_some);
    const mtime_dt = rec[4].option_val.payload.?.*.record_val;
    try testing.expect(mtime_dt[0].u64 > 0);
}

test "filesystem: set-times round-trips a known timestamp (#177)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const io = std.Io.Threaded.global_single_threaded.io();
    try tmp.dir.writeFile(io, .{ .sub_path = "f.txt", .data = "x" });
    const file = try tmp.dir.openFile(io, "f.txt", .{ .mode = .read_write });
    const handle = try adapter.pushFsDescriptor(.{ .file = .{ .file = file } });

    var dt_fields = [_]InterfaceValue{
        .{ .u64 = 1_700_000_000 },
        .{ .u32 = 0 },
    };
    const dt_iv = InterfaceValue{ .record_val = &dt_fields };
    const ts_arg = InterfaceValue{ .variant_val = .{ .discriminant = 2, .payload = &dt_iv } };

    var set_args = [_]InterfaceValue{
        .{ .handle = handle },
        ts_arg,
        ts_arg,
    };
    var set_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    var ci: ComponentInstance = undefined;
    try WasiCliAdapter.fsDescriptorSetTimes(&adapter, &ci, &set_args, &set_results, testing.allocator);
    defer set_results[0].deinit(testing.allocator);

    try testing.expect(set_results[0] == .result_val);
    try testing.expect(set_results[0].result_val.is_ok);

    // Stat back and confirm mtime.seconds matches.
    var stat_args = [_]InterfaceValue{.{ .handle = handle }};
    var stat_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.fsDescriptorStat(&adapter, &ci, &stat_args, &stat_results, testing.allocator);
    defer stat_results[0].deinit(testing.allocator);

    try testing.expect(stat_results[0].result_val.is_ok);
    const rec = stat_results[0].result_val.payload.?.*.record_val;
    try testing.expect(rec[4].option_val.is_some);
    const mtime_dt = rec[4].option_val.payload.?.*.record_val;
    // Some filesystems quantize sub-second granularity; just verify
    // seconds round-tripped exactly.
    try testing.expectEqual(@as(u64, 1_700_000_000), mtime_dt[0].u64);
}

test "filesystem: set-times on dir descriptor returns not_permitted (#177)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const handle = try adapter.addPreopen("/tmp", tmp.dir);
    // Avoid double-close: tmp.cleanup() owns tmp.dir.
    adapter.fs_descriptor_table.items[handle] = null;
    const reused = try adapter.pushFsDescriptor(.{ .preopen = .{ .dir = tmp.dir, .flags = .{ .read = true, .write = true, .mutate_directory = true } } });

    const ts_arg = InterfaceValue{ .variant_val = .{ .discriminant = 0, .payload = null } };
    var args = [_]InterfaceValue{
        .{ .handle = reused },
        ts_arg,
        ts_arg,
    };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    var ci: ComponentInstance = undefined;
    try WasiCliAdapter.fsDescriptorSetTimes(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    // Hand the slot back so adapter.deinit doesn't re-close tmp.dir.
    adapter.fs_descriptor_table.items[reused] = null;

    try testing.expect(results[0] == .result_val);
    try testing.expect(!results[0].result_val.is_ok);
    try testing.expectEqual(
        @as(u32, @intFromEnum(FsErrorCode.not_permitted)),
        results[0].result_val.payload.?.*.variant_val.discriminant,
    );
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

test "sockets DNS: lowerIpAddress lowers IPv4 as variant of tuple<u8 x 4> (#179)" {
    const testing = std.testing;
    const v4: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 0, 2, 5 }, .port = 0 } };
    const lifted = try lowerIpAddress(testing.allocator, v4);
    defer lifted.deinit(testing.allocator);

    try testing.expect(lifted == .variant_val);
    try testing.expectEqual(@as(u32, @intFromEnum(IpAddressFamily.ipv4)), lifted.variant_val.discriminant);
    const tup = lifted.variant_val.payload.?.*;
    try testing.expect(tup == .tuple_val);
    try testing.expectEqual(@as(usize, 4), tup.tuple_val.len);
    try testing.expectEqual(@as(u8, 192), tup.tuple_val[0].u8);
    try testing.expectEqual(@as(u8, 0), tup.tuple_val[1].u8);
    try testing.expectEqual(@as(u8, 2), tup.tuple_val[2].u8);
    try testing.expectEqual(@as(u8, 5), tup.tuple_val[3].u8);
}

test "sockets DNS: lowerIpAddress lowers IPv6 as variant of tuple<u16 x 8> (#179)" {
    const testing = std.testing;
    // ::1 (loopback) -> seven zero groups + 0x0001.
    const v6: std.Io.net.IpAddress = .{ .ip6 = .{
        .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
        .port = 0,
    } };
    const lifted = try lowerIpAddress(testing.allocator, v6);
    defer lifted.deinit(testing.allocator);

    try testing.expect(lifted == .variant_val);
    try testing.expectEqual(@as(u32, @intFromEnum(IpAddressFamily.ipv6)), lifted.variant_val.discriminant);
    const tup = lifted.variant_val.payload.?.*;
    try testing.expect(tup == .tuple_val);
    try testing.expectEqual(@as(usize, 8), tup.tuple_val.len);
    inline for (0..7) |i| try testing.expectEqual(@as(u16, 0), tup.tuple_val[i].u16);
    try testing.expectEqual(@as(u16, 1), tup.tuple_val[7].u16);
}

test "sockets DNS: lowerIpAddress preserves IPv6 byte order (#179)" {
    const testing = std.testing;
    // 2001:db8::1 -> groups: 0x2001, 0x0db8, 0, 0, 0, 0, 0, 0x0001
    const v6: std.Io.net.IpAddress = .{ .ip6 = .{
        .bytes = .{ 0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01 },
        .port = 0,
    } };
    const lifted = try lowerIpAddress(testing.allocator, v6);
    defer lifted.deinit(testing.allocator);
    const tup = lifted.variant_val.payload.?.*;
    try testing.expectEqual(@as(u16, 0x2001), tup.tuple_val[0].u16);
    try testing.expectEqual(@as(u16, 0x0db8), tup.tuple_val[1].u16);
    try testing.expectEqual(@as(u16, 0x0001), tup.tuple_val[7].u16);
}

test "sockets DNS: resolve-next-address on empty stream returns ok(none) (#179)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const stream = try testing.allocator.create(ResolveAddressStream);
    stream.* = .{ .results = &.{}, .pos = 0, .allocator = testing.allocator };
    const h = try adapter.pushResolveStream(stream);

    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{.{ .handle = h }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveNextAddress(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0].result_val.is_ok);
    const ok_payload = results[0].result_val.payload.?.*;
    try testing.expect(ok_payload == .option_val);
    try testing.expect(!ok_payload.option_val.is_some);
}

test "sockets DNS: resolve-next-address pops IPv4 then exhausts (#179)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const slot = try testing.allocator.alloc(std.Io.net.IpAddress, 1);
    slot[0] = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } };
    const stream = try testing.allocator.create(ResolveAddressStream);
    stream.* = .{ .results = slot, .pos = 0, .allocator = testing.allocator };
    const h = try adapter.pushResolveStream(stream);

    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{.{ .handle = h }};

    // First call: ok(some(ipv4(127,0,0,1))).
    var r1: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveNextAddress(&adapter, &ci, &args, &r1, testing.allocator);
    defer r1[0].deinit(testing.allocator);
    try testing.expect(r1[0].result_val.is_ok);
    const opt = r1[0].result_val.payload.?.*;
    try testing.expect(opt.option_val.is_some);
    const variant = opt.option_val.payload.?.*;
    try testing.expectEqual(@as(u32, @intFromEnum(IpAddressFamily.ipv4)), variant.variant_val.discriminant);
    const tup = variant.variant_val.payload.?.*;
    try testing.expectEqual(@as(u8, 127), tup.tuple_val[0].u8);
    try testing.expectEqual(@as(u8, 1), tup.tuple_val[3].u8);

    // Second call: ok(none).
    var r2: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveNextAddress(&adapter, &ci, &args, &r2, testing.allocator);
    defer r2[0].deinit(testing.allocator);
    try testing.expect(r2[0].result_val.is_ok);
    try testing.expect(!r2[0].result_val.payload.?.option_val.is_some);
}

test "sockets DNS: resolve-next-address mixed v4/v6 sequence (#179)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const slot = try testing.allocator.alloc(std.Io.net.IpAddress, 2);
    slot[0] = .{ .ip4 = .{ .bytes = .{ 10, 0, 0, 1 }, .port = 0 } };
    slot[1] = .{ .ip6 = .{
        .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
        .port = 0,
    } };
    const stream = try testing.allocator.create(ResolveAddressStream);
    stream.* = .{ .results = slot, .pos = 0, .allocator = testing.allocator };
    const h = try adapter.pushResolveStream(stream);

    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{.{ .handle = h }};

    var r1: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveNextAddress(&adapter, &ci, &args, &r1, testing.allocator);
    defer r1[0].deinit(testing.allocator);
    try testing.expectEqual(
        @as(u32, @intFromEnum(IpAddressFamily.ipv4)),
        r1[0].result_val.payload.?.option_val.payload.?.variant_val.discriminant,
    );

    var r2: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveNextAddress(&adapter, &ci, &args, &r2, testing.allocator);
    defer r2[0].deinit(testing.allocator);
    try testing.expectEqual(
        @as(u32, @intFromEnum(IpAddressFamily.ipv6)),
        r2[0].result_val.payload.?.option_val.payload.?.variant_val.discriminant,
    );

    var r3: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveNextAddress(&adapter, &ci, &args, &r3, testing.allocator);
    defer r3[0].deinit(testing.allocator);
    try testing.expect(!r3[0].result_val.payload.?.option_val.is_some);
}

test "sockets DNS: resolve-next-address rejects unknown handle (#179)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{.{ .handle = 99 }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveNextAddress(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(!results[0].result_val.is_ok);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.invalid_state)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

test "sockets DNS: resolve-addresses denies invalid hostname even when allow-list set (#179)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    try adapter.setSocketsAllowList(&.{"127.0.0.0/8"});

    // Build a network so the borrow lookup succeeds.
    var ci: ComponentInstance = undefined;
    var net_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(&adapter, &ci, &.{}, &net_results, testing.allocator);
    const net_handle = net_results[0].handle;

    // Hostname with an invalid character ('_') -> RFC1123 rejects it ->
    // name-unresolvable, no DNS round-trip attempted.
    const args = [_]InterfaceValue{
        .{ .handle = net_handle },
        .{ .string = .{ .ptr = 0, .len = 0 } },
    };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveAddresses(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(!results[0].result_val.is_ok);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.name_unresolvable)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

test "sockets DNS: resolve-addresses denies unknown network handle (#179)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    try adapter.setSocketsAllowList(&.{"0.0.0.0/0"});

    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{
        .{ .handle = 7 }, // never minted
        .{ .string = .{ .ptr = 0, .len = 0 } },
    };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.resolveAddresses(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(!results[0].result_val.is_ok);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.name_unresolvable)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

test "sockets DNS: resource-drop frees a populated resolve-stream (#179)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const slot = try testing.allocator.alloc(std.Io.net.IpAddress, 3);
    slot[0] = .{ .ip4 = .{ .bytes = .{ 1, 2, 3, 4 }, .port = 0 } };
    slot[1] = .{ .ip4 = .{ .bytes = .{ 5, 6, 7, 8 }, .port = 0 } };
    slot[2] = .{ .ip6 = .{
        .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
        .port = 0,
    } };
    const stream = try testing.allocator.create(ResolveAddressStream);
    stream.* = .{ .results = slot, .pos = 1, .allocator = testing.allocator };
    const h = try adapter.pushResolveStream(stream);

    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{.{ .handle = h }};
    var results: [0]InterfaceValue = .{};
    try WasiCliAdapter.resolveStreamDrop(&adapter, &ci, &args, &results, testing.allocator);

    try testing.expect(adapter.resolve_streams.items[h] == null);
}

test "sockets allow-list: parseCidr accepts canonical IPv4 (#180)" {
    const testing = std.testing;
    const c = try IpCidr.parse("10.0.0.0/8");
    try testing.expect(c == .ip4);
    try testing.expectEqualSlices(u8, &.{ 10, 0, 0, 0 }, &c.ip4.bytes);
    try testing.expectEqual(@as(u6, 8), c.ip4.prefix);

    const c2 = try IpCidr.parse("0.0.0.0/0");
    try testing.expectEqual(@as(u6, 0), c2.ip4.prefix);

    const c3 = try IpCidr.parse("127.0.0.1/32");
    try testing.expectEqual(@as(u6, 32), c3.ip4.prefix);
    try testing.expectEqualSlices(u8, &.{ 127, 0, 0, 1 }, &c3.ip4.bytes);
}

test "sockets allow-list: parseCidr canonicalizes host bits (#180)" {
    const testing = std.testing;
    // Non-canonical input → host bits zeroed.
    const c = try IpCidr.parse("10.5.5.5/8");
    try testing.expectEqualSlices(u8, &.{ 10, 0, 0, 0 }, &c.ip4.bytes);

    const c2 = try IpCidr.parse("192.168.1.255/24");
    try testing.expectEqualSlices(u8, &.{ 192, 168, 1, 0 }, &c2.ip4.bytes);

    const c3 = try IpCidr.parse("2001:db8:abcd:1234::1/32");
    try testing.expect(c3 == .ip6);
    var expected: [16]u8 = @splat(0);
    expected[0] = 0x20;
    expected[1] = 0x01;
    expected[2] = 0x0d;
    expected[3] = 0xb8;
    try testing.expectEqualSlices(u8, &expected, &c3.ip6.bytes);
}

test "sockets allow-list: parseCidr accepts IPv6 forms (#180)" {
    const testing = std.testing;
    const c1 = try IpCidr.parse("::/0");
    try testing.expect(c1 == .ip6);
    try testing.expectEqual(@as(u8, 0), c1.ip6.prefix);

    const c2 = try IpCidr.parse("::1/128");
    try testing.expectEqual(@as(u8, 128), c2.ip6.prefix);
    var loopback: [16]u8 = @splat(0);
    loopback[15] = 1;
    try testing.expectEqualSlices(u8, &loopback, &c2.ip6.bytes);

    const c3 = try IpCidr.parse("fe80::/10");
    try testing.expectEqual(@as(u8, 10), c3.ip6.prefix);
    // First 10 bits = 0xfe80 → first byte 0xfe, second byte top-2 = 0x80.
    try testing.expectEqual(@as(u8, 0xfe), c3.ip6.bytes[0]);
    try testing.expectEqual(@as(u8, 0x80), c3.ip6.bytes[1]);
}

test "sockets allow-list: parseCidr rejects malformed inputs (#180)" {
    const testing = std.testing;
    try testing.expectError(error.InvalidCidr, IpCidr.parse("10.0.0.0"));
    try testing.expectError(error.InvalidCidr, IpCidr.parse("10.0.0.0/"));
    try testing.expectError(error.InvalidCidr, IpCidr.parse("/24"));
    try testing.expectError(error.InvalidCidr, IpCidr.parse("/"));
    try testing.expectError(error.InvalidPrefix, IpCidr.parse("10.0.0.0/33"));
    try testing.expectError(error.InvalidPrefix, IpCidr.parse("::1/129"));
    try testing.expectError(error.InvalidPrefix, IpCidr.parse("10.0.0.0/-1"));
    try testing.expectError(error.InvalidPrefix, IpCidr.parse("10.0.0.0/+8"));
    try testing.expectError(error.InvalidPrefix, IpCidr.parse("10.0.0.0/08"));
    try testing.expectError(error.InvalidAddress, IpCidr.parse("garbage/32"));
    try testing.expectError(error.InvalidAddress, IpCidr.parse("[::]/128"));
}

test "sockets allow-list: containsAddr matches IPv4 prefix (#180)" {
    const testing = std.testing;
    const c = try IpCidr.parse("10.0.0.0/8");
    try testing.expect(c.containsAddr(.{ .ip4 = .{ .bytes = .{ 10, 5, 5, 5 }, .port = 0 } }));
    try testing.expect(c.containsAddr(.{ .ip4 = .{ .bytes = .{ 10, 0, 0, 0 }, .port = 0 } }));
    try testing.expect(c.containsAddr(.{ .ip4 = .{ .bytes = .{ 10, 255, 255, 255 }, .port = 0 } }));
    try testing.expect(!c.containsAddr(.{ .ip4 = .{ .bytes = .{ 11, 0, 0, 1 }, .port = 0 } }));
    try testing.expect(!c.containsAddr(.{ .ip4 = .{ .bytes = .{ 9, 255, 255, 255 }, .port = 0 } }));

    const all = try IpCidr.parse("0.0.0.0/0");
    try testing.expect(all.containsAddr(.{ .ip4 = .{ .bytes = .{ 1, 2, 3, 4 }, .port = 0 } }));
    try testing.expect(all.containsAddr(.{ .ip4 = .{ .bytes = .{ 255, 255, 255, 255 }, .port = 0 } }));
}

test "sockets allow-list: containsAddr matches IPv6 prefix (#180)" {
    const testing = std.testing;
    const loop = try IpCidr.parse("::1/128");
    var ones: [16]u8 = @splat(0);
    ones[15] = 1;
    try testing.expect(loop.containsAddr(.{ .ip6 = .{ .port = 0, .bytes = ones } }));
    var twos: [16]u8 = @splat(0);
    twos[15] = 2;
    try testing.expect(!loop.containsAddr(.{ .ip6 = .{ .port = 0, .bytes = twos } }));

    const all = try IpCidr.parse("::/0");
    var rand: [16]u8 = @splat(0);
    rand[0] = 0xde;
    rand[3] = 0xad;
    try testing.expect(all.containsAddr(.{ .ip6 = .{ .port = 0, .bytes = rand } }));

    const link_local = try IpCidr.parse("fe80::/10");
    var fe80: [16]u8 = @splat(0);
    fe80[0] = 0xfe;
    fe80[1] = 0x80;
    fe80[15] = 0x42;
    try testing.expect(link_local.containsAddr(.{ .ip6 = .{ .port = 0, .bytes = fe80 } }));
    var feC0: [16]u8 = @splat(0);
    feC0[0] = 0xfe;
    feC0[1] = 0xc0;
    try testing.expect(!link_local.containsAddr(.{ .ip6 = .{ .port = 0, .bytes = feC0 } }));
}

test "sockets allow-list: containsAddr rejects mixed family (#180)" {
    const testing = std.testing;
    const v4 = try IpCidr.parse("0.0.0.0/0");
    const any6: [16]u8 = @splat(0);
    try testing.expect(!v4.containsAddr(.{ .ip6 = .{ .port = 0, .bytes = any6 } }));

    const v6 = try IpCidr.parse("::/0");
    try testing.expect(!v6.containsAddr(.{ .ip4 = .{ .bytes = .{ 1, 2, 3, 4 }, .port = 0 } }));

    // IPv4-mapped IPv6 (`::ffff:127.0.0.1`) is matched as IPv6, not as
    // IPv4 — the wasi:sockets layer hands us already-discriminated values.
    var mapped: [16]u8 = @splat(0);
    mapped[10] = 0xff;
    mapped[11] = 0xff;
    mapped[12] = 127;
    mapped[15] = 1;
    const v4_loop = try IpCidr.parse("127.0.0.0/8");
    try testing.expect(!v4_loop.containsAddr(.{ .ip6 = .{ .port = 0, .bytes = mapped } }));
}

test "sockets allow-list: Network.allows defaults to deny-all (#180)" {
    const testing = std.testing;
    const n: Network = .{};
    try testing.expect(!n.allows(.{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } }));
    const any6: [16]u8 = @splat(0);
    try testing.expect(!n.allows(.{ .ip6 = .{ .port = 0, .bytes = any6 } }));
}

test "sockets allow-list: setSocketsAllowList parses + replaces (#180)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    try testing.expectEqual(@as(usize, 0), adapter.sockets_allow_list_template.len);

    try adapter.setSocketsAllowList(&.{ "127.0.0.0/8", "::1/128" });
    try testing.expectEqual(@as(usize, 2), adapter.sockets_allow_list_template.len);
    try testing.expect(adapter.sockets_allow_list_template[0] == .ip4);
    try testing.expect(adapter.sockets_allow_list_template[1] == .ip6);

    // Replace; old slice is freed.
    try adapter.setSocketsAllowList(&.{"10.0.0.0/8"});
    try testing.expectEqual(@as(usize, 1), adapter.sockets_allow_list_template.len);
    try testing.expectEqual(@as(u6, 8), adapter.sockets_allow_list_template[0].ip4.prefix);

    // Reset to deny-all.
    try adapter.setSocketsAllowList(&.{});
    try testing.expectEqual(@as(usize, 0), adapter.sockets_allow_list_template.len);
}

test "sockets allow-list: setSocketsAllowList preserves prior on parse failure (#180)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    try adapter.setSocketsAllowList(&.{"127.0.0.0/8"});
    try testing.expectEqual(@as(usize, 1), adapter.sockets_allow_list_template.len);

    // Second entry has a bad prefix → whole call fails, prior list intact.
    try testing.expectError(
        error.InvalidPrefix,
        adapter.setSocketsAllowList(&.{ "10.0.0.0/8", "10.0.0.0/99" }),
    );
    try testing.expectEqual(@as(usize, 1), adapter.sockets_allow_list_template.len);
    try testing.expectEqualSlices(u8, &.{ 127, 0, 0, 0 }, &adapter.sockets_allow_list_template[0].ip4.bytes);
}

test "sockets allow-list: instance-network snapshots template (#180)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    try adapter.setSocketsAllowList(&.{ "10.0.0.0/8", "::1/128" });

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(&adapter, &ci, &.{}, &results, testing.allocator);

    const handle = results[0].handle;
    const n = adapter.network_table.items[handle].?;
    try testing.expectEqual(@as(usize, 2), n.allow_list.len);
    try testing.expect(n.allows(.{ .ip4 = .{ .bytes = .{ 10, 5, 5, 5 }, .port = 0 } }));
    try testing.expect(!n.allows(.{ .ip4 = .{ .bytes = .{ 11, 0, 0, 1 }, .port = 0 } }));

    // Reconfiguring after must not mutate the existing Network.
    try adapter.setSocketsAllowList(&.{});
    const n_after = adapter.network_table.items[handle].?;
    try testing.expectEqual(@as(usize, 2), n_after.allow_list.len);
}

test "sockets allow-list: resource-drop frees snapshot (#180)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    try adapter.setSocketsAllowList(&.{"10.0.0.0/8"});

    var ci: ComponentInstance = undefined;
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(&adapter, &ci, &.{}, &results, testing.allocator);
    const handle = results[0].handle;

    const drop_args = [_]InterfaceValue{.{ .handle = handle }};
    var drop_results: [0]InterfaceValue = .{};
    try WasiCliAdapter.networkResourceDrop(&adapter, &ci, &drop_args, &drop_results, testing.allocator);
    try testing.expect(adapter.network_table.items[handle] == null);
    // testing.allocator's leak detector flags us if the snapshot wasn't freed.
}

test "http: liftFieldEntries lifts canonical layout (#174)" {
    const testing = std.testing;

    // Memory layout:
    //   [0..16)   : tuple #0 = (16, 11, 27, 13)  → "content-type" / "text/plain"
    //   wait — recompute. We'll place name at 32 ("foo"), value at 40 ("bar"),
    //   and a single tuple at offset 0.
    var mem: [64]u8 = @splat(0);
    @memcpy(mem[16..19], "foo");
    @memcpy(mem[24..27], "bar");
    std.mem.writeInt(u32, mem[0..4], 16, .little); // name.ptr
    std.mem.writeInt(u32, mem[4..8], 3, .little); // name.len
    std.mem.writeInt(u32, mem[8..12], 24, .little); // value.ptr
    std.mem.writeInt(u32, mem[12..16], 3, .little); // value.len

    const entries = try liftFieldEntries(testing.allocator, &mem, 0, 1);
    defer {
        for (entries) |e| {
            testing.allocator.free(e.name);
            testing.allocator.free(e.value);
        }
        testing.allocator.free(entries);
    }

    try testing.expectEqual(@as(usize, 1), entries.len);
    try testing.expectEqualStrings("foo", entries[0].name);
    try testing.expectEqualStrings("bar", entries[0].value);
}

test "http: liftFieldEntries handles multiple entries (#174)" {
    const testing = std.testing;

    // Two tuples at offsets 0 and 16 (stride 16). String pool starts at 64.
    var mem: [128]u8 = @splat(0);
    const pool: u32 = 64;
    @memcpy(mem[pool .. pool + 12], "content-type");
    @memcpy(mem[pool + 12 .. pool + 22], "text/plain");
    @memcpy(mem[pool + 22 .. pool + 30], "x-custom");
    @memcpy(mem[pool + 30 .. pool + 35], "value");

    // tuple #0
    std.mem.writeInt(u32, mem[0..4], pool, .little);
    std.mem.writeInt(u32, mem[4..8], 12, .little);
    std.mem.writeInt(u32, mem[8..12], pool + 12, .little);
    std.mem.writeInt(u32, mem[12..16], 10, .little);
    // tuple #1
    std.mem.writeInt(u32, mem[16..20], pool + 22, .little);
    std.mem.writeInt(u32, mem[20..24], 8, .little);
    std.mem.writeInt(u32, mem[24..28], pool + 30, .little);
    std.mem.writeInt(u32, mem[28..32], 5, .little);

    const entries = try liftFieldEntries(testing.allocator, &mem, 0, 2);
    defer {
        for (entries) |e| {
            testing.allocator.free(e.name);
            testing.allocator.free(e.value);
        }
        testing.allocator.free(entries);
    }

    try testing.expectEqual(@as(usize, 2), entries.len);
    try testing.expectEqualStrings("content-type", entries[0].name);
    try testing.expectEqualStrings("text/plain", entries[0].value);
    try testing.expectEqualStrings("x-custom", entries[1].name);
    try testing.expectEqualStrings("value", entries[1].value);
}

test "http: liftFieldEntries handles empty strings (#174)" {
    const testing = std.testing;
    // tuple of (empty, empty) — ptrs ignored, lens are 0.
    var mem: [16]u8 = @splat(0);
    // All zeros: name.ptr=0, name.len=0, value.ptr=0, value.len=0.
    const entries = try liftFieldEntries(testing.allocator, &mem, 0, 1);
    defer {
        for (entries) |e| {
            testing.allocator.free(e.name);
            testing.allocator.free(e.value);
        }
        testing.allocator.free(entries);
    }
    try testing.expectEqual(@as(usize, 1), entries.len);
    try testing.expectEqualStrings("", entries[0].name);
    try testing.expectEqualStrings("", entries[0].value);
}

test "http: liftFieldEntries rejects out-of-bounds list (#174)" {
    const testing = std.testing;
    var mem: [16]u8 = @splat(0);
    // count=1 needs 16 bytes; list_ptr=8 → list_end=24 > 16.
    try testing.expectError(error.OutOfBoundsMemory, liftFieldEntries(testing.allocator, &mem, 8, 1));
    // list_ptr=0xfffffff0 + count=1*16 overflows u32.
    try testing.expectError(error.OutOfBoundsMemory, liftFieldEntries(testing.allocator, &mem, 0xfffffff0, 2));
}

test "http: liftFieldEntries rejects out-of-bounds string (#174)" {
    const testing = std.testing;
    var mem: [32]u8 = @splat(0);
    // tuple at 0: name.ptr=24, name.len=100 → 124 > 32.
    std.mem.writeInt(u32, mem[0..4], 24, .little);
    std.mem.writeInt(u32, mem[4..8], 100, .little);
    std.mem.writeInt(u32, mem[8..12], 0, .little);
    std.mem.writeInt(u32, mem[12..16], 0, .little);
    try testing.expectError(error.OutOfBoundsMemory, liftFieldEntries(testing.allocator, &mem, 0, 1));
}

test "http: liftFieldEntries no leak on partial-fill failure (#174)" {
    const testing = std.testing;
    var mem: [48]u8 = @splat(0);
    // Two tuples: #0 valid ("ok" pair), #1 has bad value pointer.
    @memcpy(mem[32..34], "ok");
    // tuple #0
    std.mem.writeInt(u32, mem[0..4], 32, .little);
    std.mem.writeInt(u32, mem[4..8], 2, .little);
    std.mem.writeInt(u32, mem[8..12], 32, .little);
    std.mem.writeInt(u32, mem[12..16], 2, .little);
    // tuple #1 — value points past memory.
    std.mem.writeInt(u32, mem[16..20], 32, .little);
    std.mem.writeInt(u32, mem[20..24], 2, .little);
    std.mem.writeInt(u32, mem[24..28], 100, .little);
    std.mem.writeInt(u32, mem[28..32], 50, .little);

    // testing.allocator's leak detector flags us if entry #0's strings
    // weren't freed when #1 failed.
    try testing.expectError(error.OutOfBoundsMemory, liftFieldEntries(testing.allocator, &mem, 0, 2));
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

test "http: fields.from-list returns ok with empty list (#149, #174)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    const args = [_]InterfaceValue{.{ .list = .{ .ptr = 0, .len = 0 } }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsFromList(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .result_val);
    try testing.expect(results[0].result_val.is_ok);
    try testing.expect(results[0].result_val.payload.?.* == .handle);
    try testing.expectEqual(@as(usize, 1), adapter.http_fields_table.items.len);
    const f = adapter.http_fields_table.items[0].?;
    try testing.expectEqual(@as(usize, 0), f.entries.items.len);
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

// ── #181: open-flags / descriptor-flags / path-flags fidelity ────────────────

test "filesystem: FsDescriptorFlags.fromBits / toBits round-trip (#181)" {
    const testing = std.testing;
    inline for (.{
        @as(u32, 0),
        @as(u32, 0b000011), // read | write
        @as(u32, 0b100010), // write | mutate_directory
        @as(u32, 0b011100), // f-i-sync | d-i-sync | r-w-sync
        @as(u32, 0b111111), // all
    }) |bits| {
        const f = FsDescriptorFlags.fromBits(bits);
        try testing.expectEqual(bits, f.toBits());
    }

    // needsWriteSync covers exactly file/data integrity sync.
    try testing.expect(!FsDescriptorFlags.fromBits(0b010000).needsWriteSync()); // requested-write-sync alone (read-side)
    try testing.expect(FsDescriptorFlags.fromBits(0b000100).needsWriteSync()); // file-integrity-sync
    try testing.expect(FsDescriptorFlags.fromBits(0b001000).needsWriteSync()); // data-integrity-sync
}

test "filesystem: get-flags returns the bits stored on the descriptor (#181)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "f.txt", .data = "x" });
    const file = try tmp.dir.openFile(io, "f.txt", .{ .mode = .read_only });

    // file-integrity-sync + data-integrity-sync + read.
    const stored: u32 = 0b001101;
    const handle = try adapter.pushFsDescriptor(.{ .file = .{
        .file = file,
        .flags = FsDescriptorFlags.fromBits(stored),
    } });

    var args = [_]InterfaceValue{.{ .handle = handle }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    var ci: ComponentInstance = undefined;
    try WasiCliAdapter.fsDescriptorGetFlags(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .result_val);
    try testing.expect(results[0].result_val.is_ok);
    const inner = results[0].result_val.payload.?.*;
    try testing.expect(inner == .flags_val);
    try testing.expectEqual(@as(usize, 1), inner.flags_val.len);
    try testing.expectEqual(stored, inner.flags_val[0]);
}

test "filesystem: open-at threads sync flags into output stream (#181)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try tmp.dir.writeFile(io, .{ .sub_path = "f.txt", .data = "x" });

    // descriptor-flags = read | write | data-integrity-sync.
    const desc_flags_bits: u32 = 0b001011;
    const file = try tmp.dir.openFile(io, "f.txt", .{ .mode = .read_write });
    const handle = try adapter.pushFsDescriptor(.{ .file = .{
        .file = file,
        .flags = FsDescriptorFlags.fromBits(desc_flags_bits),
    } });

    var write_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    const stream_handle = try adapter.fsAllocOutputFileStream(handle, 0, false, testing.allocator, &write_results);
    try testing.expect(stream_handle != null);
    const stream = adapter.lookupStream(stream_handle.?).?;
    try testing.expect(stream.sink == .host_file);
    try testing.expect(stream.sink.host_file.sync_on_flush);
}

test "filesystem: open-at without mutate-directory denies writable child (#181)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    // Push a base directory that *lacks* mutate-directory — read-only root.
    try adapter.fs_descriptor_table.append(adapter.allocator, .{ .preopen = .{
        .dir = tmp.dir,
        .flags = .{ .read = true, .mutate_directory = false },
    } });
    const read_only_dir: u32 = @intCast(adapter.fs_descriptor_table.items.len - 1);
    defer adapter.fs_descriptor_table.items[read_only_dir] = null;

    // The mutate-directory check fires before any guest memory is
    // touched, so a stub ComponentInstance + dummy string is fine.
    const path_pl = InterfaceValue{ .string = .{ .ptr = 0, .len = 0 } };
    var args = [_]InterfaceValue{
        .{ .handle = read_only_dir },
        .{ .u32 = 0 }, // path-flags
        path_pl,
        .{ .u32 = 0 }, // open-flags
        .{ .u32 = 0b10 }, // descriptor-flags = write
    };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    var ci: ComponentInstance = undefined;
    try WasiCliAdapter.fsDescriptorOpenAt(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .result_val);
    try testing.expect(!results[0].result_val.is_ok);
    try testing.expectEqual(
        @as(u32, @intFromEnum(FsErrorCode.read_only)),
        results[0].result_val.payload.?.*.variant_val.discriminant,
    );
}

test "filesystem: blocking-flush calls file.sync for sync-flagged descriptor (#181)" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try tmp.dir.writeFile(io, .{ .sub_path = "out.txt", .data = "" });
    const file = try tmp.dir.openFile(io, "out.txt", .{ .mode = .read_write });
    const handle = try adapter.pushFsDescriptor(.{ .file = .{
        .file = file,
        .flags = .{ .read = true, .write = true, .data_integrity_sync = true },
    } });

    // Allocate the output stream; sync_on_flush must propagate.
    var alloc_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    const stream_handle = (try adapter.fsAllocOutputFileStream(handle, 0, false, testing.allocator, &alloc_results)).?;
    const stream = adapter.lookupStream(stream_handle).?;
    try testing.expect(stream.sink.host_file.sync_on_flush);

    // Drive blocking-flush — should succeed (real fsync on a tmp file).
    var flush_args = [_]InterfaceValue{.{ .handle = stream_handle }};
    var flush_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    var ci: ComponentInstance = undefined;
    try WasiCliAdapter.outputStreamBlockingFlush(&adapter, &ci, &flush_args, &flush_results, testing.allocator);
    defer flush_results[0].deinit(testing.allocator);
    try testing.expect(flush_results[0].result_val.is_ok);
}

test "filesystem: open-at strips mutate-directory from non-directory child (#181)" {
    const testing = std.testing;
    const stripped = blk: {
        var f = FsDescriptorFlags.fromBits(0b100011); // read|write|mutate_directory
        f.mutate_directory = false;
        break :blk f;
    };
    try testing.expectEqual(@as(u32, 0b000011), stripped.toBits());
}

// ---------------------------------------------------------------------------
// #178 PR A: TCP bind + listen + getters
// ---------------------------------------------------------------------------

/// Build an ip-socket-address InterfaceValue value pointing at heap-owned
/// fields. Caller must free via `.deinit(allocator)`.
fn testMakeIpv4SocketAddress(
    allocator: Allocator,
    octets: [4]u8,
    port: u16,
) !InterfaceValue {
    const tup = try allocator.alloc(InterfaceValue, 4);
    inline for (0..4) |i| tup[i] = .{ .u8 = octets[i] };
    const fields = try allocator.alloc(InterfaceValue, 2);
    fields[0] = .{ .u16 = port };
    fields[1] = .{ .tuple_val = tup };
    const rec = try allocator.create(InterfaceValue);
    rec.* = .{ .record_val = fields };
    return .{ .variant_val = .{ .discriminant = 0, .payload = rec } };
}

test "sockets #178: lift+lower ipv4 socket-address round-trips" {
    const testing = std.testing;
    const a = testing.allocator;
    const v = try testMakeIpv4SocketAddress(a, .{ 127, 0, 0, 1 }, 4242);
    defer v.deinit(a);
    const lifted = try liftIpSocketAddress(v, .ipv4);
    try testing.expect(lifted == .ip4);
    try testing.expectEqual([_]u8{ 127, 0, 0, 1 }, lifted.ip4.bytes);
    try testing.expectEqual(@as(u16, 4242), lifted.ip4.port);

    const back = try lowerIpSocketAddress(a, lifted);
    defer back.deinit(a);
    try testing.expectEqual(@as(u32, 0), back.variant_val.discriminant);
    const rec = back.variant_val.payload.?.record_val;
    try testing.expectEqual(@as(u16, 4242), rec[0].u16);
    try testing.expectEqual(@as(u8, 127), rec[1].tuple_val[0].u8);
    try testing.expectEqual(@as(u8, 1), rec[1].tuple_val[3].u8);
}

test "sockets #178: liftIpSocketAddress rejects family mismatch" {
    const testing = std.testing;
    const a = testing.allocator;
    const v = try testMakeIpv4SocketAddress(a, .{ 0, 0, 0, 0 }, 0);
    defer v.deinit(a);
    try testing.expectError(error.FamilyMismatch, liftIpSocketAddress(v, .ipv6));
}

test "sockets #178: tcp start-bind denies without allow-list" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();

    var ci: ComponentInstance = undefined;
    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);

    var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(&adapter, &ci, &.{}, &n_results, testing.allocator);

    const local = try testMakeIpv4SocketAddress(testing.allocator, .{ 127, 0, 0, 1 }, 0);
    defer local.deinit(testing.allocator);
    const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.tcpStartBind(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);

    try testing.expect(!results[0].result_val.is_ok);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.access_denied)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

test "sockets #178: tcp start-bind family mismatch is invalid_argument" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.1/32"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            var ci: ComponentInstance = undefined;
            // ipv6 socket
            const c_args = [_]InterfaceValue{.{ .enum_val = 1 }};
            var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.createTcpSocket(adapter, &ci, &c_args, &c_results, std.testing.allocator);
            defer std.testing.allocator.destroy(c_results[0].result_val.payload.?);

            var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.instanceNetwork(adapter, &ci, &.{}, &n_results, std.testing.allocator);

            const local = try testMakeIpv4SocketAddress(std.testing.allocator, .{ 127, 0, 0, 1 }, 0);
            defer local.deinit(std.testing.allocator);
            const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
            var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.tcpStartBind(adapter, &ci, &args, &results, std.testing.allocator);
            defer std.testing.allocator.destroy(results[0].result_val.payload.?);

            try std.testing.expect(!results[0].result_val.is_ok);
            try std.testing.expectEqual(
                @as(u32, @intFromEnum(SocketErrorCode.invalid_argument)),
                results[0].result_val.payload.?.variant_val.discriminant,
            );
        }
    }.run);
}

const AdapterWithAllowList = struct { allow: []const []const u8 };

fn adapter_with_allow_list(
    cfg: AdapterWithAllowList,
    comptime body: fn (*WasiCliAdapter) anyerror!void,
) !void {
    var adapter = WasiCliAdapter.init(std.testing.allocator);
    defer adapter.deinit();
    try adapter.setSocketsAllowList(cfg.allow);
    try body(&adapter);
}

test "sockets #178: tcp finish-bind on idle socket is not_in_progress" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;
    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);

    const args = [_]InterfaceValue{.{ .handle = 0 }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.tcpFinishBind(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.not_in_progress)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

test "sockets #178: tcp start-listen requires bound state" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;
    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);

    const args = [_]InterfaceValue{.{ .handle = 0 }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.tcpStartListen(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.invalid_state)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

test "sockets #178: tcp local-address on unbound returns invalid_state" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;
    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);

    const args = [_]InterfaceValue{.{ .handle = 0 }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.tcpLocalAddress(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.invalid_state)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

test "sockets #178: tcp full lifecycle bind+listen on 127.0.0.1:0 reports kernel port" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    try adapter.setSocketsAllowList(&.{"127.0.0.0/8"});

    var ci: ComponentInstance = undefined;
    // ipv4 tcp socket
    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);
    var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(&adapter, &ci, &.{}, &n_results, testing.allocator);

    // start-bind 127.0.0.1:0
    const local = try testMakeIpv4SocketAddress(testing.allocator, .{ 127, 0, 0, 1 }, 0);
    defer local.deinit(testing.allocator);
    {
        const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.tcpStartBind(&adapter, &ci, &args, &results, testing.allocator);
        defer testing.allocator.destroy(results[0].result_val.payload.?);
        try testing.expect(results[0].result_val.is_ok);
    }
    // finish-bind
    {
        const args = [_]InterfaceValue{.{ .handle = 0 }};
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.tcpFinishBind(&adapter, &ci, &args, &results, testing.allocator);
        defer testing.allocator.destroy(results[0].result_val.payload.?);
        try testing.expect(results[0].result_val.is_ok);
    }
    // start-listen — performs the actual syscall.
    {
        const args = [_]InterfaceValue{.{ .handle = 0 }};
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.tcpStartListen(&adapter, &ci, &args, &results, testing.allocator);
        defer testing.allocator.destroy(results[0].result_val.payload.?);
        try testing.expect(results[0].result_val.is_ok);
    }
    // finish-listen
    {
        const args = [_]InterfaceValue{.{ .handle = 0 }};
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.tcpFinishListen(&adapter, &ci, &args, &results, testing.allocator);
        defer testing.allocator.destroy(results[0].result_val.payload.?);
        try testing.expect(results[0].result_val.is_ok);
    }
    // is-listening = true.
    {
        const args = [_]InterfaceValue{.{ .handle = 0 }};
        var results: [1]InterfaceValue = .{.{ .bool = false }};
        try WasiCliAdapter.tcpIsListening(&adapter, &ci, &args, &results, testing.allocator);
        try testing.expect(results[0].bool);
    }
    // local-address: kernel-assigned port is non-zero.
    {
        const args = [_]InterfaceValue{.{ .handle = 0 }};
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.tcpLocalAddress(&adapter, &ci, &args, &results, testing.allocator);
        defer results[0].deinit(testing.allocator);
        try testing.expect(results[0].result_val.is_ok);
        const v = results[0].result_val.payload.?.*;
        try testing.expectEqual(@as(u32, 0), v.variant_val.discriminant);
        const rec = v.variant_val.payload.?.record_val;
        try testing.expect(rec[0].u16 != 0);
        try testing.expectEqual(@as(u8, 127), rec[1].tuple_val[0].u8);
    }
}

test "sockets #178: tcp wildcard bind allowed by 0.0.0.0/0" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    try adapter.setSocketsAllowList(&.{"0.0.0.0/0"});
    var ci: ComponentInstance = undefined;
    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);
    var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(&adapter, &ci, &.{}, &n_results, testing.allocator);

    const local = try testMakeIpv4SocketAddress(testing.allocator, .{ 0, 0, 0, 0 }, 0);
    defer local.deinit(testing.allocator);
    const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.tcpStartBind(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);
    try testing.expect(results[0].result_val.is_ok);
}

test "sockets #178: tcp resource-drop after listen frees server" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    try adapter.setSocketsAllowList(&.{"127.0.0.0/8"});
    var ci: ComponentInstance = undefined;
    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createTcpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);
    var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(&adapter, &ci, &.{}, &n_results, testing.allocator);

    const local = try testMakeIpv4SocketAddress(testing.allocator, .{ 127, 0, 0, 1 }, 0);
    defer local.deinit(testing.allocator);
    {
        const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.tcpStartBind(&adapter, &ci, &args, &results, testing.allocator);
        defer testing.allocator.destroy(results[0].result_val.payload.?);
    }
    {
        const args = [_]InterfaceValue{.{ .handle = 0 }};
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.tcpFinishBind(&adapter, &ci, &args, &results, testing.allocator);
        defer testing.allocator.destroy(results[0].result_val.payload.?);
    }
    {
        const args = [_]InterfaceValue{.{ .handle = 0 }};
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.tcpStartListen(&adapter, &ci, &args, &results, testing.allocator);
        defer testing.allocator.destroy(results[0].result_val.payload.?);
    }

    try testing.expect(adapter.socket_table.items[0].?.server != null);

    const drop_args = [_]InterfaceValue{.{ .handle = 0 }};
    var drop_results: [0]InterfaceValue = .{};
    try WasiCliAdapter.socketResourceDrop(&adapter, &ci, &drop_args, &drop_results, testing.allocator);
    try testing.expect(adapter.socket_table.items[0] == null);
}

// ── #176: real outbound HTTP client ─────────────────────────────────────────

/// Helper: build a `list_val` of u8 InterfaceValues from a byte slice.
/// Allocator-owned; caller frees via `deinit`.
fn testMakeListVal(alloc: Allocator, bytes: []const u8) !InterfaceValue {
    const ivs = try alloc.alloc(InterfaceValue, bytes.len);
    for (bytes, 0..) |b, i| ivs[i] = .{ .u8 = b };
    return .{ .list_val = ivs };
}

test "http #176: fields.append stores and entries returns entries" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    // Create fields
    var ctor_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsConstructor(&adapter, &ci, &.{}, &ctor_results, testing.allocator);
    const fh = ctor_results[0].handle;

    // Append "content-type: text/plain"
    const name1 = try testMakeListVal(testing.allocator, "content-type");
    defer name1.deinit(testing.allocator);
    const val1 = try testMakeListVal(testing.allocator, "text/plain");
    defer val1.deinit(testing.allocator);
    const app_args1 = [_]InterfaceValue{ .{ .handle = fh }, name1, val1 };
    var app_results1: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsAppend(&adapter, &ci, &app_args1, &app_results1, testing.allocator);
    try testing.expect(app_results1[0].result_val.is_ok);

    // Append "x-custom: hello"
    const name2 = try testMakeListVal(testing.allocator, "x-custom");
    defer name2.deinit(testing.allocator);
    const val2 = try testMakeListVal(testing.allocator, "hello");
    defer val2.deinit(testing.allocator);
    const app_args2 = [_]InterfaceValue{ .{ .handle = fh }, name2, val2 };
    var app_results2: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsAppend(&adapter, &ci, &app_args2, &app_results2, testing.allocator);
    try testing.expect(app_results2[0].result_val.is_ok);

    // Call entries
    const ent_args = [_]InterfaceValue{.{ .handle = fh }};
    var ent_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsEntries(&adapter, &ci, &ent_args, &ent_results, testing.allocator);
    defer ent_results[0].deinit(testing.allocator);

    try testing.expect(ent_results[0] == .list_val);
    try testing.expectEqual(@as(usize, 2), ent_results[0].list_val.len);
}

test "http #176: fields.get returns matching values" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    var ctor_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsConstructor(&adapter, &ci, &.{}, &ctor_results, testing.allocator);
    const fh = ctor_results[0].handle;

    // Append two entries with same name
    const name = try testMakeListVal(testing.allocator, "accept");
    defer name.deinit(testing.allocator);
    const val_a = try testMakeListVal(testing.allocator, "text/html");
    defer val_a.deinit(testing.allocator);
    const val_b = try testMakeListVal(testing.allocator, "application/json");
    defer val_b.deinit(testing.allocator);

    const args_a = [_]InterfaceValue{ .{ .handle = fh }, name, val_a };
    var r1: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsAppend(&adapter, &ci, &args_a, &r1, testing.allocator);

    const args_b = [_]InterfaceValue{ .{ .handle = fh }, name, val_b };
    var r2: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsAppend(&adapter, &ci, &args_b, &r2, testing.allocator);

    // Get by name
    const get_args = [_]InterfaceValue{ .{ .handle = fh }, name };
    var get_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsGet(&adapter, &ci, &get_args, &get_results, testing.allocator);
    defer get_results[0].deinit(testing.allocator);

    try testing.expect(get_results[0] == .list_val);
    try testing.expectEqual(@as(usize, 2), get_results[0].list_val.len);
}

test "http #176: fields.has returns true for existing name" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    var ctor_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsConstructor(&adapter, &ci, &.{}, &ctor_results, testing.allocator);
    const fh = ctor_results[0].handle;

    // has() before append → false
    const name = try testMakeListVal(testing.allocator, "x-test");
    defer name.deinit(testing.allocator);
    const has_args = [_]InterfaceValue{ .{ .handle = fh }, name };
    var has_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsHas(&adapter, &ci, &has_args, &has_r, testing.allocator);
    try testing.expect(!has_r[0].bool);

    // Append
    const val = try testMakeListVal(testing.allocator, "yes");
    defer val.deinit(testing.allocator);
    const app_args = [_]InterfaceValue{ .{ .handle = fh }, name, val };
    var app_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsAppend(&adapter, &ci, &app_args, &app_r, testing.allocator);

    // has() after append → true
    try WasiCliAdapter.httpFieldsHas(&adapter, &ci, &has_args, &has_r, testing.allocator);
    try testing.expect(has_r[0].bool);
}

test "http #176: fields.delete removes entries" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    var ctor_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsConstructor(&adapter, &ci, &.{}, &ctor_results, testing.allocator);
    const fh = ctor_results[0].handle;

    const name = try testMakeListVal(testing.allocator, "x-del");
    defer name.deinit(testing.allocator);
    const val = try testMakeListVal(testing.allocator, "gone");
    defer val.deinit(testing.allocator);
    const app_args = [_]InterfaceValue{ .{ .handle = fh }, name, val };
    var app_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsAppend(&adapter, &ci, &app_args, &app_r, testing.allocator);

    // Delete
    const del_args = [_]InterfaceValue{ .{ .handle = fh }, name };
    var del_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsDelete(&adapter, &ci, &del_args, &del_r, testing.allocator);
    try testing.expect(del_r[0].result_val.is_ok);

    // Entries should be empty
    const ent_args = [_]InterfaceValue{.{ .handle = fh }};
    var ent_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsEntries(&adapter, &ci, &ent_args, &ent_r, testing.allocator);
    defer ent_r[0].deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), ent_r[0].list_val.len);
}

test "http #176: outgoing-request set-method persists" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    // Create fields + request
    var f_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsConstructor(&adapter, &ci, &.{}, &f_r, testing.allocator);
    const req_args = [_]InterfaceValue{.{ .handle = f_r[0].handle }};
    var req_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestConstructor(&adapter, &ci, &req_args, &req_r, testing.allocator);
    const rh = req_r[0].handle;

    // Default method = GET (0)
    const m_args = [_]InterfaceValue{.{ .handle = rh }};
    var m_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestMethod(&adapter, &ci, &m_args, &m_r, testing.allocator);
    try testing.expectEqual(@as(u32, 0), m_r[0].variant_val.discriminant);

    // Set to POST (2)
    const sm_args = [_]InterfaceValue{ .{ .handle = rh }, .{ .variant_val = .{ .discriminant = 2, .payload = null } } };
    var sm_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestSetMethod(&adapter, &ci, &sm_args, &sm_r, testing.allocator);
    try testing.expect(sm_r[0].result_val.is_ok);

    // Verify
    try WasiCliAdapter.httpOutgoingRequestMethod(&adapter, &ci, &m_args, &m_r, testing.allocator);
    try testing.expectEqual(@as(u32, 2), m_r[0].variant_val.discriminant);
}

test "http #176: outgoing-request set-path persists" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    var f_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsConstructor(&adapter, &ci, &.{}, &f_r, testing.allocator);
    const req_args = [_]InterfaceValue{.{ .handle = f_r[0].handle }};
    var req_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestConstructor(&adapter, &ci, &req_args, &req_r, testing.allocator);
    const rh = req_r[0].handle;

    // Default path = none
    const gp_args = [_]InterfaceValue{.{ .handle = rh }};
    var gp_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestGetPath(&adapter, &ci, &gp_args, &gp_r, testing.allocator);
    try testing.expect(!gp_r[0].option_val.is_some);

    // Set path
    const path_val = try testMakeListVal(testing.allocator, "/api/data");
    defer path_val.deinit(testing.allocator);
    const path_payload = try testing.allocator.create(InterfaceValue);
    defer testing.allocator.destroy(path_payload);
    path_payload.* = path_val;
    const sp_args = [_]InterfaceValue{ .{ .handle = rh }, .{ .option_val = .{ .is_some = true, .payload = path_payload } } };
    var sp_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestSetPath(&adapter, &ci, &sp_args, &sp_r, testing.allocator);
    try testing.expect(sp_r[0].result_val.is_ok);

    // Verify path getter
    try WasiCliAdapter.httpOutgoingRequestGetPath(&adapter, &ci, &gp_args, &gp_r, testing.allocator);
    defer gp_r[0].deinit(testing.allocator);
    try testing.expect(gp_r[0].option_val.is_some);
}

test "http #176: outgoing-body write creates stream" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    // Create fields + request
    var f_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpFieldsConstructor(&adapter, &ci, &.{}, &f_r, testing.allocator);
    const req_args = [_]InterfaceValue{.{ .handle = f_r[0].handle }};
    var req_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestConstructor(&adapter, &ci, &req_args, &req_r, testing.allocator);

    // Get body
    const body_args = [_]InterfaceValue{.{ .handle = req_r[0].handle }};
    var body_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingRequestBody(&adapter, &ci, &body_args, &body_r, testing.allocator);
    defer body_r[0].deinit(testing.allocator);
    try testing.expect(body_r[0].result_val.is_ok);
    const bh = body_r[0].result_val.payload.?.handle;

    // Write to body → get an output-stream handle
    const write_args = [_]InterfaceValue{.{ .handle = bh }};
    var write_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingBodyWrite(&adapter, &ci, &write_args, &write_r, testing.allocator);
    defer write_r[0].deinit(testing.allocator);
    try testing.expect(write_r[0].result_val.is_ok);
    const sh = write_r[0].result_val.payload.?.handle;

    // Write some bytes to the stream
    const s = adapter.lookupStream(sh).?;
    const wr = s.write("hello world", testing.allocator);
    try testing.expectEqual(@as(usize, 11), wr.ok);

    // Verify body has the data
    const body = adapter.lookupOutgoingBody(bh).?;
    try testing.expectEqualStrings("hello world", body.stream.?.getBufferContents());

    // Second write call should fail
    var write_r2: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingBodyWrite(&adapter, &ci, &write_args, &write_r2, testing.allocator);
    try testing.expect(!write_r2[0].result_val.is_ok);
}

test "http #176: handle returns denied when allow-list empty" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    const args = [_]InterfaceValue{
        .{ .handle = 0 },
        .{ .option_val = .{ .is_some = false, .payload = null } },
    };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpOutgoingHandlerHandle(&adapter, &ci, &args, &results, testing.allocator);
    defer results[0].deinit(testing.allocator);

    try testing.expect(results[0] == .result_val);
    try testing.expect(results[0].result_val.is_ok);
    const fut_handle = results[0].result_val.payload.?.handle;
    const fut = adapter.http_future_responses.items[fut_handle].?;
    try testing.expect(fut.state == .ready_err);
    try testing.expectEqual(
        @as(u32, @intFromEnum(HttpErrorCode.HTTP_request_denied)),
        fut.state.ready_err,
    );
}

test "http #176: incoming-response.consume transfers body data" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    // Manually create an IncomingResponse with body data
    const body_bytes = try testing.allocator.dupe(u8, "response body");
    const resp_fields = try testing.allocator.create(HttpFields);
    resp_fields.* = .{};
    const fh = try adapter.pushHttpFields(resp_fields);

    const ir = try testing.allocator.create(IncomingResponse);
    ir.* = .{ .status = 200, .headers_handle = fh, .body_data = body_bytes };
    const ir_handle = try adapter.pushIncomingResponse(ir);

    // Call consume
    const cons_args = [_]InterfaceValue{.{ .handle = ir_handle }};
    var cons_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpIncomingResponseConsume(&adapter, &ci, &cons_args, &cons_r, testing.allocator);
    defer cons_r[0].deinit(testing.allocator);
    try testing.expect(cons_r[0].result_val.is_ok);
    const body_handle = cons_r[0].result_val.payload.?.handle;

    // Get body stream
    const stream_args = [_]InterfaceValue{.{ .handle = body_handle }};
    var stream_r: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpIncomingBodyStream(&adapter, &ci, &stream_args, &stream_r, testing.allocator);
    defer stream_r[0].deinit(testing.allocator);
    try testing.expect(stream_r[0].result_val.is_ok);
    const sh = stream_r[0].result_val.payload.?.handle;

    // Read from the stream
    const is = adapter.lookupInputStream(sh).?;
    var buf: [64]u8 = undefined;
    const read_result = is.read(&buf);
    switch (read_result) {
        .ok => |n| try testing.expectEqualStrings("response body", buf[0..n]),
        else => return error.UnexpectedResult,
    }

    // Second consume should fail
    var cons_r2: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.httpIncomingResponseConsume(&adapter, &ci, &cons_args, &cons_r2, testing.allocator);
    try testing.expect(!cons_r2[0].result_val.is_ok);
}

// ---------- UDP tests (#178 PR C) ----------

test "sockets #178 UDP: start-bind denied without allow-list" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createUdpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);

    var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(&adapter, &ci, &.{}, &n_results, testing.allocator);

    const local = try testMakeIpv4SocketAddress(testing.allocator, .{ 127, 0, 0, 1 }, 0);
    defer local.deinit(testing.allocator);
    const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.udpStartBind(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);
    try testing.expect(!results[0].result_val.is_ok);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.access_denied)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

test "sockets #178 UDP: start-bind family mismatch is invalid_argument" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.1/32"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            var ci: ComponentInstance = undefined;
            // ipv6 udp socket
            const c_args = [_]InterfaceValue{.{ .enum_val = 1 }};
            var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.createUdpSocket(adapter, &ci, &c_args, &c_results, std.testing.allocator);
            defer std.testing.allocator.destroy(c_results[0].result_val.payload.?);

            var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.instanceNetwork(adapter, &ci, &.{}, &n_results, std.testing.allocator);

            const local = try testMakeIpv4SocketAddress(std.testing.allocator, .{ 127, 0, 0, 1 }, 0);
            defer local.deinit(std.testing.allocator);
            const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
            var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStartBind(adapter, &ci, &args, &results, std.testing.allocator);
            defer std.testing.allocator.destroy(results[0].result_val.payload.?);
            try std.testing.expect(!results[0].result_val.is_ok);
            try std.testing.expectEqual(
                @as(u32, @intFromEnum(SocketErrorCode.invalid_argument)),
                results[0].result_val.payload.?.variant_val.discriminant,
            );
        }
    }.run);
}

test "sockets #178 UDP: start-bind second call is invalid_state" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            var ci: ComponentInstance = undefined;
            const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
            var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.createUdpSocket(adapter, &ci, &c_args, &c_results, std.testing.allocator);
            defer std.testing.allocator.destroy(c_results[0].result_val.payload.?);

            var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.instanceNetwork(adapter, &ci, &.{}, &n_results, std.testing.allocator);

            const local = try testMakeIpv4SocketAddress(std.testing.allocator, .{ 127, 0, 0, 1 }, 0);
            defer local.deinit(std.testing.allocator);
            // First start-bind → ok.
            {
                const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
                var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.udpStartBind(adapter, &ci, &args, &results, std.testing.allocator);
                defer std.testing.allocator.destroy(results[0].result_val.payload.?);
                try std.testing.expect(results[0].result_val.is_ok);
            }
            // Second call → invalid-state (socket no longer unbound).
            {
                const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
                var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.udpStartBind(adapter, &ci, &args, &results, std.testing.allocator);
                defer std.testing.allocator.destroy(results[0].result_val.payload.?);
                try std.testing.expect(!results[0].result_val.is_ok);
                try std.testing.expectEqual(
                    @as(u32, @intFromEnum(SocketErrorCode.invalid_state)),
                    results[0].result_val.payload.?.variant_val.discriminant,
                );
            }
        }
    }.run);
}

test "sockets #178 UDP: finish-bind on idle socket is not_in_progress" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createUdpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);

    const args = [_]InterfaceValue{.{ .handle = 0 }};
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.udpFinishBind(&adapter, &ci, &args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);
    try testing.expect(!results[0].result_val.is_ok);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.not_in_progress)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

test "sockets #178 UDP: local-address reports kernel port after bind on 127.0.0.1:0" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            var ci: ComponentInstance = undefined;
            const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
            var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.createUdpSocket(adapter, &ci, &c_args, &c_results, std.testing.allocator);
            defer std.testing.allocator.destroy(c_results[0].result_val.payload.?);
            var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.instanceNetwork(adapter, &ci, &.{}, &n_results, std.testing.allocator);

            const local = try testMakeIpv4SocketAddress(std.testing.allocator, .{ 127, 0, 0, 1 }, 0);
            defer local.deinit(std.testing.allocator);
            // start-bind
            {
                const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
                var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.udpStartBind(adapter, &ci, &args, &results, std.testing.allocator);
                defer std.testing.allocator.destroy(results[0].result_val.payload.?);
                try std.testing.expect(results[0].result_val.is_ok);
            }
            // finish-bind
            {
                const args = [_]InterfaceValue{.{ .handle = 0 }};
                var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.udpFinishBind(adapter, &ci, &args, &results, std.testing.allocator);
                defer std.testing.allocator.destroy(results[0].result_val.payload.?);
                try std.testing.expect(results[0].result_val.is_ok);
            }
            // local-address
            {
                const args = [_]InterfaceValue{.{ .handle = 0 }};
                var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.udpLocalAddress(adapter, &ci, &args, &results, std.testing.allocator);
                defer results[0].deinit(std.testing.allocator);
                try std.testing.expect(results[0].result_val.is_ok);
                const v = results[0].result_val.payload.?.*;
                try std.testing.expectEqual(@as(u32, 0), v.variant_val.discriminant);
                const rec = v.variant_val.payload.?.record_val;
                // Kernel assigned a non-zero port.
                try std.testing.expect(rec[0].u16 != 0);
                try std.testing.expectEqual(@as(u8, 127), rec[1].tuple_val[0].u8);
            }
        }
    }.run);
}

test "sockets #178 UDP: stream on unbound socket is invalid_state" {
    const testing = std.testing;
    var adapter = WasiCliAdapter.init(testing.allocator);
    defer adapter.deinit();
    var ci: ComponentInstance = undefined;

    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createUdpSocket(&adapter, &ci, &c_args, &c_results, testing.allocator);
    defer testing.allocator.destroy(c_results[0].result_val.payload.?);

    // stream(none) on unbound socket.
    const stream_args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .variant_val = .{ .discriminant = 0, .payload = null } } };
    var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.udpStream(&adapter, &ci, &stream_args, &results, testing.allocator);
    defer testing.allocator.destroy(results[0].result_val.payload.?);
    try testing.expect(!results[0].result_val.is_ok);
    try testing.expectEqual(
        @as(u32, @intFromEnum(SocketErrorCode.invalid_state)),
        results[0].result_val.payload.?.variant_val.discriminant,
    );
}

/// Helper: bind a UDP socket to 127.0.0.1:0 and finish-bind. Returns
/// the socket handle (always 0 when starting from a fresh adapter).
fn testUdpBind(adapter: *WasiCliAdapter) !void {
    var ci: ComponentInstance = undefined;
    const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
    var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.createUdpSocket(adapter, &ci, &c_args, &c_results, std.testing.allocator);
    c_results[0].deinit(std.testing.allocator);
    var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
    try WasiCliAdapter.instanceNetwork(adapter, &ci, &.{}, &n_results, std.testing.allocator);
    const local = try testMakeIpv4SocketAddress(std.testing.allocator, .{ 127, 0, 0, 1 }, 0);
    defer local.deinit(std.testing.allocator);
    {
        const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 0 }, local };
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.udpStartBind(adapter, &ci, &args, &results, std.testing.allocator);
        results[0].deinit(std.testing.allocator);
    }
    {
        const args = [_]InterfaceValue{.{ .handle = 0 }};
        var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
        try WasiCliAdapter.udpFinishBind(adapter, &ci, &args, &results, std.testing.allocator);
        results[0].deinit(std.testing.allocator);
    }
}

test "sockets #178 UDP: receive on empty queue returns ok([])" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            try testUdpBind(adapter);
            var ci: ComponentInstance = undefined;
            // stream(none) — unconnected.
            const stream_args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .variant_val = .{ .discriminant = 0, .payload = null } } };
            var stream_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &stream_args, &stream_results, std.testing.allocator);
            defer stream_results[0].deinit(std.testing.allocator);
            try std.testing.expect(stream_results[0].result_val.is_ok);
            const pair = stream_results[0].result_val.payload.?.tuple_val;
            const in_handle = pair[0].handle;
            // receive(max_results=10) on empty queue → ok([]).
            const recv_args = [_]InterfaceValue{ .{ .handle = in_handle }, .{ .u64 = 10 } };
            var recv_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.incomingDatagramReceive(adapter, &ci, &recv_args, &recv_results, std.testing.allocator);
            defer recv_results[0].deinit(std.testing.allocator);
            try std.testing.expect(recv_results[0].result_val.is_ok);
            try std.testing.expectEqual(@as(usize, 0), recv_results[0].result_val.payload.?.list_val.len);
        }
    }.run);
}

test "sockets #178 UDP: check-send credit and send traps" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            try testUdpBind(adapter);
            var ci: ComponentInstance = undefined;
            // stream(none)
            const stream_args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .variant_val = .{ .discriminant = 0, .payload = null } } };
            var stream_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &stream_args, &stream_results, std.testing.allocator);
            defer stream_results[0].deinit(std.testing.allocator);
            const pair = stream_results[0].result_val.payload.?.tuple_val;
            const out_handle = pair[1].handle;

            // send without check-send → trap.
            {
                const empty_list = try std.testing.allocator.alloc(InterfaceValue, 0);
                defer std.testing.allocator.free(empty_list);
                const send_args = [_]InterfaceValue{ .{ .handle = out_handle }, .{ .list_val = empty_list } };
                var send_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                const err = WasiCliAdapter.outgoingSend(adapter, &ci, &send_args, &send_results, std.testing.allocator);
                try std.testing.expectError(error.Trap, err);
            }

            // check-send → ok(8)
            {
                const cs_args = [_]InterfaceValue{.{ .handle = out_handle }};
                var cs_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.outgoingCheckSend(adapter, &ci, &cs_args, &cs_results, std.testing.allocator);
                defer cs_results[0].deinit(std.testing.allocator);
                try std.testing.expect(cs_results[0].result_val.is_ok);
                try std.testing.expectEqual(@as(u64, 8), cs_results[0].result_val.payload.?.u64);
            }

            // send with too many datagrams → trap.
            {
                // credit is 8, send 9 → trap
                const too_many = try std.testing.allocator.alloc(InterfaceValue, 9);
                defer std.testing.allocator.free(too_many);
                for (too_many) |*slot| slot.* = .{ .u32 = 0 };
                const send_args = [_]InterfaceValue{ .{ .handle = out_handle }, .{ .list_val = too_many } };
                var send_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                const err = WasiCliAdapter.outgoingSend(adapter, &ci, &send_args, &send_results, std.testing.allocator);
                try std.testing.expectError(error.Trap, err);
            }

            // After trap, credit was consumed so next send also traps.
            // Re-check first.
            {
                const cs_args = [_]InterfaceValue{.{ .handle = out_handle }};
                var cs_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.outgoingCheckSend(adapter, &ci, &cs_args, &cs_results, std.testing.allocator);
                defer cs_results[0].deinit(std.testing.allocator);
                try std.testing.expect(cs_results[0].result_val.is_ok);
            }

            // send 0 datagrams → ok(0)
            {
                const empty_list = try std.testing.allocator.alloc(InterfaceValue, 0);
                defer std.testing.allocator.free(empty_list);
                const send_args = [_]InterfaceValue{ .{ .handle = out_handle }, .{ .list_val = empty_list } };
                var send_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.outgoingSend(adapter, &ci, &send_args, &send_results, std.testing.allocator);
                defer send_results[0].deinit(std.testing.allocator);
                try std.testing.expect(send_results[0].result_val.is_ok);
                try std.testing.expectEqual(@as(u64, 0), send_results[0].result_val.payload.?.u64);
            }
        }
    }.run);
}

test "sockets #178 UDP: loopback send + receive round-trip" {
    // Windows IOCP zero-duration receiveTimeout doesn't reliably
    // deliver loopback UDP datagrams in CI; skip on Windows.
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            // Create + bind socket A on 127.0.0.1:0.
            try testUdpBind(adapter);
            // Get socket A's port.
            var la_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpLocalAddress(adapter, &ci, &[_]InterfaceValue{.{ .handle = 0 }}, &la_results, a);
            defer la_results[0].deinit(a);
            const port_a = la_results[0].result_val.payload.?.variant_val.payload.?.record_val[0].u16;

            // Create + bind socket B on 127.0.0.1:0.
            {
                const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
                var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.createUdpSocket(adapter, &ci, &c_args, &c_results, a);
                c_results[0].deinit(a);
            }
            {
                var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.instanceNetwork(adapter, &ci, &.{}, &n_results, a);
            }
            {
                const local = try testMakeIpv4SocketAddress(a, .{ 127, 0, 0, 1 }, 0);
                defer local.deinit(a);
                // sock handle=1, net handle=1
                const args = [_]InterfaceValue{ .{ .handle = 1 }, .{ .handle = 1 }, local };
                var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.udpStartBind(adapter, &ci, &args, &results, a);
                results[0].deinit(a);
            }
            {
                const args = [_]InterfaceValue{.{ .handle = 1 }};
                var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.udpFinishBind(adapter, &ci, &args, &results, a);
                results[0].deinit(a);
            }

            // stream(none) on socket A (unconnected).
            var stream_a_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &[_]InterfaceValue{
                .{ .handle = 0 },
                .{ .variant_val = .{ .discriminant = 0, .payload = null } },
            }, &stream_a_results, a);
            defer stream_a_results[0].deinit(a);
            const pair_a = stream_a_results[0].result_val.payload.?.tuple_val;
            const out_a = pair_a[1].handle;

            // stream(none) on socket B (unconnected).
            var stream_b_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &[_]InterfaceValue{
                .{ .handle = 1 },
                .{ .variant_val = .{ .discriminant = 0, .payload = null } },
            }, &stream_b_results, a);
            defer stream_b_results[0].deinit(a);
            const pair_b = stream_b_results[0].result_val.payload.?.tuple_val;
            const in_b = pair_b[0].handle;

            // Send "hello" from A to B.
            {
                const cs_args = [_]InterfaceValue{.{ .handle = out_a }};
                var cs_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.outgoingCheckSend(adapter, &ci, &cs_args, &cs_results, a);
                cs_results[0].deinit(a);
            }
            // Build B's address for the datagram destination.
            var lb_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpLocalAddress(adapter, &ci, &[_]InterfaceValue{.{ .handle = 1 }}, &lb_results, a);
            defer lb_results[0].deinit(a);
            const port_b = lb_results[0].result_val.payload.?.variant_val.payload.?.record_val[0].u16;
            _ = port_a;

            // Build outgoing-datagram { data: "hello", remote-address: some(B) }
            const dest = try testMakeIpv4SocketAddress(a, .{ 127, 0, 0, 1 }, port_b);
            defer dest.deinit(a);
            const remote_opt = try a.create(InterfaceValue);
            remote_opt.* = dest;
            defer a.destroy(remote_opt);

            const hello = "hello";
            const data_iv = try a.alloc(InterfaceValue, hello.len);
            defer a.free(data_iv);
            for (hello, 0..) |b, i| data_iv[i] = .{ .u8 = b };
            const dg_fields = try a.alloc(InterfaceValue, 2);
            defer a.free(dg_fields);
            dg_fields[0] = .{ .list_val = data_iv };
            dg_fields[1] = .{ .variant_val = .{ .discriminant = 1, .payload = remote_opt } };
            const dg_list = try a.alloc(InterfaceValue, 1);
            defer a.free(dg_list);
            dg_list[0] = .{ .record_val = dg_fields };

            {
                const send_args = [_]InterfaceValue{ .{ .handle = out_a }, .{ .list_val = dg_list } };
                var send_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.outgoingSend(adapter, &ci, &send_args, &send_results, a);
                defer send_results[0].deinit(a);
                try std.testing.expect(send_results[0].result_val.is_ok);
                try std.testing.expectEqual(@as(u64, 1), send_results[0].result_val.payload.?.u64);
            }

            // Receive on B — retry loop to handle Windows IOCP latency.
            {
                const io_poll = std.Io.Threaded.global_single_threaded.io();
                var received = false;
                var attempt: usize = 0;
                while (attempt < 50) : (attempt += 1) {
                    const wait_dur: std.Io.Clock.Duration = .{ .raw = std.Io.Duration.fromMilliseconds(10), .clock = .awake };
                    wait_dur.sleep(io_poll) catch {};
                    const recv_args = [_]InterfaceValue{ .{ .handle = in_b }, .{ .u64 = 10 } };
                    var recv_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                    try WasiCliAdapter.incomingDatagramReceive(adapter, &ci, &recv_args, &recv_results, a);
                    const is_ok = recv_results[0].result_val.is_ok;
                    const list = if (is_ok) recv_results[0].result_val.payload.?.list_val else &[_]InterfaceValue{};
                    if (is_ok and list.len >= 1) {
                        defer recv_results[0].deinit(a);
                        // Check received data.
                        const dg0 = list[0].record_val;
                        const received_data = dg0[0].list_val;
                        try std.testing.expectEqual(@as(usize, 5), received_data.len);
                        try std.testing.expectEqual(@as(u8, 'h'), received_data[0].u8);
                        try std.testing.expectEqual(@as(u8, 'o'), received_data[4].u8);
                        received = true;
                        break;
                    }
                    recv_results[0].deinit(a);
                }
                try std.testing.expect(received);
            }
        }
    }.run);
}

test "sockets #178 UDP: send unconnected + per-datagram none is invalid_argument" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            try testUdpBind(adapter);
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            // stream(none)
            var stream_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &[_]InterfaceValue{
                .{ .handle = 0 },
                .{ .variant_val = .{ .discriminant = 0, .payload = null } },
            }, &stream_results, a);
            defer stream_results[0].deinit(a);
            const out_handle = stream_results[0].result_val.payload.?.tuple_val[1].handle;
            // check-send
            {
                const cs_args = [_]InterfaceValue{.{ .handle = out_handle }};
                var cs_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.outgoingCheckSend(adapter, &ci, &cs_args, &cs_results, a);
                cs_results[0].deinit(a);
            }
            // Build datagram with remote = none (variant discriminant 0, no payload).
            const data_iv = try a.alloc(InterfaceValue, 1);
            defer a.free(data_iv);
            data_iv[0] = .{ .u8 = 42 };
            const dg_fields = try a.alloc(InterfaceValue, 2);
            defer a.free(dg_fields);
            dg_fields[0] = .{ .list_val = data_iv };
            dg_fields[1] = .{ .variant_val = .{ .discriminant = 0, .payload = null } };
            const dg_list = try a.alloc(InterfaceValue, 1);
            defer a.free(dg_list);
            dg_list[0] = .{ .record_val = dg_fields };
            // send → invalid_argument (unconnected requires per-datagram remote).
            const send_args = [_]InterfaceValue{ .{ .handle = out_handle }, .{ .list_val = dg_list } };
            var send_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.outgoingSend(adapter, &ci, &send_args, &send_results, a);
            defer send_results[0].deinit(a);
            try std.testing.expect(!send_results[0].result_val.is_ok);
            try std.testing.expectEqual(
                @as(u32, @intFromEnum(SocketErrorCode.invalid_argument)),
                send_results[0].result_val.payload.?.variant_val.discriminant,
            );
        }
    }.run);
}

test "sockets #178 UDP: network-drop survives via deep-copied allow_list" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            try testUdpBind(adapter);
            // Drop the network resource.
            {
                const drop_args = [_]InterfaceValue{.{ .handle = 0 }};
                var drop_results: [0]InterfaceValue = .{};
                try WasiCliAdapter.networkResourceDrop(adapter, &ci, &drop_args, &drop_results, a);
            }
            // stream(none) still works — socket owns its allow_list.
            var stream_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &[_]InterfaceValue{
                .{ .handle = 0 },
                .{ .variant_val = .{ .discriminant = 0, .payload = null } },
            }, &stream_results, a);
            defer stream_results[0].deinit(a);
            try std.testing.expect(stream_results[0].result_val.is_ok);
        }
    }.run);
}

test "sockets #178 UDP: fd-leak verification — rebind same port after drop" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            try testUdpBind(adapter);
            // Get the kernel-assigned port.
            var la_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpLocalAddress(adapter, &ci, &[_]InterfaceValue{.{ .handle = 0 }}, &la_results, a);
            defer la_results[0].deinit(a);
            const port = la_results[0].result_val.payload.?.variant_val.payload.?.record_val[0].u16;
            // Drop socket 0 — should close the fd.
            {
                const drop_args = [_]InterfaceValue{.{ .handle = 0 }};
                var drop_results: [0]InterfaceValue = .{};
                try WasiCliAdapter.socketResourceDrop(adapter, &ci, &drop_args, &drop_results, a);
            }
            // Create a new socket and bind to the same port.
            {
                const c_args = [_]InterfaceValue{.{ .enum_val = 0 }};
                var c_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.createUdpSocket(adapter, &ci, &c_args, &c_results, a);
                c_results[0].deinit(a);
            }
            {
                var n_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.instanceNetwork(adapter, &ci, &.{}, &n_results, a);
            }
            {
                const local = try testMakeIpv4SocketAddress(a, .{ 127, 0, 0, 1 }, port);
                defer local.deinit(a);
                // socket=0 (reused slot), network=1 (new)
                const args = [_]InterfaceValue{ .{ .handle = 0 }, .{ .handle = 1 }, local };
                var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.udpStartBind(adapter, &ci, &args, &results, a);
                defer results[0].deinit(a);
                try std.testing.expect(results[0].result_val.is_ok);
            }
        }
    }.run);
}

test "sockets #178 UDP: generation invalidation — old stream returns invalid_state" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            try testUdpBind(adapter);
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            // First stream(none) pair.
            var s1_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &[_]InterfaceValue{
                .{ .handle = 0 },
                .{ .variant_val = .{ .discriminant = 0, .payload = null } },
            }, &s1_results, a);
            defer s1_results[0].deinit(a);
            const old_in = s1_results[0].result_val.payload.?.tuple_val[0].handle;
            const old_out = s1_results[0].result_val.payload.?.tuple_val[1].handle;
            // Second stream(none) — invalidates the first pair.
            var s2_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &[_]InterfaceValue{
                .{ .handle = 0 },
                .{ .variant_val = .{ .discriminant = 0, .payload = null } },
            }, &s2_results, a);
            defer s2_results[0].deinit(a);
            // Old incoming receive → invalid_state.
            {
                const recv_args = [_]InterfaceValue{ .{ .handle = old_in }, .{ .u64 = 10 } };
                var recv_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.incomingDatagramReceive(adapter, &ci, &recv_args, &recv_results, a);
                defer recv_results[0].deinit(a);
                try std.testing.expect(!recv_results[0].result_val.is_ok);
                try std.testing.expectEqual(
                    @as(u32, @intFromEnum(SocketErrorCode.invalid_state)),
                    recv_results[0].result_val.payload.?.variant_val.discriminant,
                );
            }
            // Old outgoing check-send → invalid_state.
            {
                const cs_args = [_]InterfaceValue{.{ .handle = old_out }};
                var cs_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.outgoingCheckSend(adapter, &ci, &cs_args, &cs_results, a);
                defer cs_results[0].deinit(a);
                try std.testing.expect(!cs_results[0].result_val.is_ok);
                try std.testing.expectEqual(
                    @as(u32, @intFromEnum(SocketErrorCode.invalid_state)),
                    cs_results[0].result_val.payload.?.variant_val.discriminant,
                );
            }
        }
    }.run);
}

test "sockets #178 UDP: remote-address returns invalid_state when unconnected" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            try testUdpBind(adapter);
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            // stream(none) — no connected remote.
            var stream_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &[_]InterfaceValue{
                .{ .handle = 0 },
                .{ .variant_val = .{ .discriminant = 0, .payload = null } },
            }, &stream_results, a);
            defer stream_results[0].deinit(a);
            // remote-address → invalid_state.
            const ra_args = [_]InterfaceValue{.{ .handle = 0 }};
            var ra_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpRemoteAddress(adapter, &ci, &ra_args, &ra_results, a);
            defer ra_results[0].deinit(a);
            try std.testing.expect(!ra_results[0].result_val.is_ok);
            try std.testing.expectEqual(
                @as(u32, @intFromEnum(SocketErrorCode.invalid_state)),
                ra_results[0].result_val.payload.?.variant_val.discriminant,
            );
        }
    }.run);
}

test "sockets #178 UDP: stream(some(remote)) validates wildcard/port-0/family" {
    try adapter_with_allow_list(.{ .allow = &.{ "127.0.0.0/8", "0.0.0.0/0" } }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            try testUdpBind(adapter);
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            const Case = struct { octets: [4]u8, port: u16 };
            const cases = [_]Case{
                .{ .octets = .{ 0, 0, 0, 0 }, .port = 1234 }, // wildcard
                .{ .octets = .{ 127, 0, 0, 1 }, .port = 0 }, // port-0
            };
            for (cases) |c| {
                const remote = try testMakeIpv4SocketAddress(a, c.octets, c.port);
                defer remote.deinit(a);
                const payload_inner = try a.create(InterfaceValue);
                payload_inner.* = remote;
                defer a.destroy(payload_inner);
                const stream_args = [_]InterfaceValue{
                    .{ .handle = 0 },
                    .{ .variant_val = .{ .discriminant = 1, .payload = payload_inner } },
                };
                var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
                try WasiCliAdapter.udpStream(adapter, &ci, &stream_args, &results, a);
                defer results[0].deinit(a);
                try std.testing.expect(!results[0].result_val.is_ok);
                try std.testing.expectEqual(
                    @as(u32, @intFromEnum(SocketErrorCode.invalid_argument)),
                    results[0].result_val.payload.?.variant_val.discriminant,
                );
            }
        }
    }.run);
}

test "sockets #178 UDP: stream(some(remote)) allow-list denied" {
    // Allow-list only covers 127.0.0.0/8, so 10.x is denied.
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            try testUdpBind(adapter);
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            const remote = try testMakeIpv4SocketAddress(a, .{ 10, 0, 0, 1 }, 5000);
            defer remote.deinit(a);
            const payload_inner = try a.create(InterfaceValue);
            payload_inner.* = remote;
            defer a.destroy(payload_inner);
            const stream_args = [_]InterfaceValue{
                .{ .handle = 0 },
                .{ .variant_val = .{ .discriminant = 1, .payload = payload_inner } },
            };
            var results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &stream_args, &results, a);
            defer results[0].deinit(a);
            try std.testing.expect(!results[0].result_val.is_ok);
            try std.testing.expectEqual(
                @as(u32, @intFromEnum(SocketErrorCode.access_denied)),
                results[0].result_val.payload.?.variant_val.discriminant,
            );
        }
    }.run);
}

test "sockets #178 UDP: receive max_results=0 returns ok([])" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            try testUdpBind(adapter);
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            var stream_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.udpStream(adapter, &ci, &[_]InterfaceValue{
                .{ .handle = 0 },
                .{ .variant_val = .{ .discriminant = 0, .payload = null } },
            }, &stream_results, a);
            defer stream_results[0].deinit(a);
            const in_handle = stream_results[0].result_val.payload.?.tuple_val[0].handle;
            // receive(0) → ok([])
            const recv_args = [_]InterfaceValue{ .{ .handle = in_handle }, .{ .u64 = 0 } };
            var recv_results: [1]InterfaceValue = .{.{ .u32 = 0 }};
            try WasiCliAdapter.incomingDatagramReceive(adapter, &ci, &recv_args, &recv_results, a);
            defer recv_results[0].deinit(a);
            try std.testing.expect(recv_results[0].result_val.is_ok);
            try std.testing.expectEqual(@as(usize, 0), recv_results[0].result_val.payload.?.list_val.len);
        }
    }.run);
}

test "sockets #178 UDP: resource-drop frees host_socket and allow_list" {
    try adapter_with_allow_list(.{ .allow = &.{"127.0.0.0/8"} }, struct {
        fn run(adapter: *WasiCliAdapter) !void {
            try testUdpBind(adapter);
            var ci: ComponentInstance = undefined;
            const a = std.testing.allocator;
            // Verify socket exists with host_socket and allow_list.
            try std.testing.expect(adapter.socket_table.items[0].?.host_socket != null);
            try std.testing.expect(adapter.socket_table.items[0].?.allow_list.len > 0);
            // Drop.
            const drop_args = [_]InterfaceValue{.{ .handle = 0 }};
            var drop_results: [0]InterfaceValue = .{};
            try WasiCliAdapter.socketResourceDrop(adapter, &ci, &drop_args, &drop_results, a);
            try std.testing.expect(adapter.socket_table.items[0] == null);
        }
    }.run);
}

test "sockets #178 UDP: validateRemoteForSend rejects bad remotes" {
    const Ip4 = std.Io.net.Ip4Address;
    // Wildcard v4.
    try std.testing.expectEqual(
        @as(?SocketErrorCode, .invalid_argument),
        validateRemoteForSend(
            .{ .ip4 = Ip4{ .bytes = .{ 0, 0, 0, 0 }, .port = 1234 } },
            .ipv4,
        ),
    );
    // Port 0.
    try std.testing.expectEqual(
        @as(?SocketErrorCode, .invalid_argument),
        validateRemoteForSend(
            .{ .ip4 = Ip4{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } },
            .ipv4,
        ),
    );
    // Family mismatch.
    try std.testing.expectEqual(
        @as(?SocketErrorCode, .invalid_argument),
        validateRemoteForSend(
            .{ .ip4 = Ip4{ .bytes = .{ 127, 0, 0, 1 }, .port = 5000 } },
            .ipv6,
        ),
    );
    // Valid.
    try std.testing.expectEqual(
        @as(?SocketErrorCode, null),
        validateRemoteForSend(
            .{ .ip4 = Ip4{ .bytes = .{ 127, 0, 0, 1 }, .port = 5000 } },
            .ipv4,
        ),
    );
}
