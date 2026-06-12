//! `wit_http` — guest-side helper for writing
//! `wasi:http/incoming-handler@0.2.6` server components in pure Zig.
//!
//! ## Why this exists
//!
//! A component guest must speak the **canonical ABI**: host functions are
//! imported with lowered core-wasm signatures (every WIT type flattened to
//! `i32` / `i64` slots), and any result wider than one core value is
//! returned through a guest-allocated "ret-area" pointer passed as the
//! last parameter. There is no Zig `wit-bindgen` backend that generates
//! these bindings, so historically each example hand-wrote ~12 `extern`
//! declarations plus the ret-area bookkeeping inline.
//!
//! This module writes those `extern`s **once** and exposes a small typed
//! Zig API on top. Two language features make the result ergonomic:
//!
//!   * **Dead-code elimination.** A guest only imports the host functions
//!     it transitively calls — an `extern` that is never referenced is not
//!     emitted as a wasm import. So a minimal handler that only reads the
//!     request path and writes a body links against exactly that subset,
//!     even though this module declares the full surface. Each example can
//!     therefore keep a *minimal* WIT world; sharing the helper does not
//!     force a superset of imports onto everyone.
//!
//!   * **`comptime` export wiring.** `exportIncomingHandler` takes your
//!     handler as a `comptime` function value and emits the canonical
//!     `wasi:http/incoming-handler@0.2.6#handle` export for you (along with
//!     a `cabi_realloc` backed by a scratch arena), so the verbose export
//!     name and the per-request setup live here instead of in every guest.
//!
//! ## Usage
//!
//! ```zig
//! const wit = @import("wit_http");
//!
//! comptime {
//!     wit.exportIncomingHandler(handle);
//! }
//!
//! fn handle(req: wit.Request, res: *wit.Responder) void {
//!     const path = req.path() orelse "/";
//!     if (std.mem.eql(u8, path, "/")) {
//!         res.respond(200, "Hello, world!\n");
//!     } else {
//!         res.respond(404, "");
//!     }
//! }
//! ```

const std = @import("std");

// ── cabi_realloc scratch arena ─────────────────────────────────────
//
// Canonical-ABI lifts of host-side `string` / `list` values into guest
// memory call `cabi_realloc`. The host uses it to materialize the
// request path, the request method's `other(string)` payload, and each
// request-body read chunk. `exportIncomingHandler` resets the arena at
// the top of every request, so each invocation gets a fresh 64 KiB
// scratch surface.
//
// `cabi_realloc` is `export`ed from this module; because exported
// symbols are linker roots, it appears in the final component's core
// wasm even though it is defined in a dependency module (not the guest's
// root source file).

var arena_buf: [65536]u8 align(16) = undefined;
var arena_top: usize = 0;

inline fn alignUp(x: usize, a: usize) usize {
    return (x + a - 1) & ~(a - 1);
}

export fn cabi_realloc(
    _: usize, // old_ptr — we never free
    _: usize, // old_size
    alignment: usize,
    new_size: usize,
) usize {
    if (new_size == 0) return 0;
    const a = if (alignment == 0) 1 else alignment;
    const start = alignUp(arena_top, a);
    if (start + new_size > arena_buf.len) return 0;
    arena_top = start + new_size;
    return @intFromPtr(&arena_buf[start]);
}

/// Reset the scratch arena. Called automatically by the
/// `exportIncomingHandler` wrapper at the start of each request.
pub fn resetScratch() void {
    arena_top = 0;
}

// ── Ret-area for spilled results ───────────────────────────────────
//
// Every imported method whose flat result exceeds one core value writes
// its result words here; the helpers below read them back. 32 bytes (8
// words) comfortably covers the widest result we decode
// (`outgoing-body.finish` → 5 words).

var ret_area: [32]u8 align(8) = undefined;

inline fn retPtr() i32 {
    return @intCast(@intFromPtr(&ret_area));
}

inline fn retWords() [*]u32 {
    return @ptrCast(@alignCast(&ret_area));
}

/// Decode the ret-area as `result<own<handle>>` → the handle on the ok
/// arm (word layout `[disc, handle]`), or null on the err arm.
inline fn readResultHandle() ?i32 {
    const w = retWords();
    return if (w[0] == 0) @bitCast(w[1]) else null;
}

/// Decode the ret-area as `option<string>` (`[disc, ptr, len]`) into a
/// slice borrowing from the scratch arena, or null for `none`.
inline fn readOptionString() ?[]const u8 {
    const w = retWords();
    if (w[0] != 1) return null;
    const p: [*]const u8 = @ptrFromInt(w[1]);
    return p[0..w[2]];
}

// ── Host imports (canonical-ABI lowered signatures) ────────────────
//
// `wasi:http/types@0.2.6`. Results wider than one core value spill
// through the trailing `retptr`; bare `result` lowers to a single i32
// discriminant returned directly.

/// `[constructor]fields() -> own<fields>`.
extern "wasi:http/types@0.2.6" fn @"[constructor]fields"() i32;

/// `[method]fields.append(borrow, field-name (string), field-value (list<u8>))
///   -> result<_, header-error>`. retptr → [disc, header-err-disc].
extern "wasi:http/types@0.2.6" fn @"[method]fields.append"(
    self: i32,
    name_ptr: i32,
    name_len: i32,
    value_ptr: i32,
    value_len: i32,
    retptr: i32,
) void;

/// `[constructor]outgoing-response(own<fields>) -> own<outgoing-response>`.
extern "wasi:http/types@0.2.6" fn @"[constructor]outgoing-response"(headers: i32) i32;

/// `[method]outgoing-response.set-status-code(borrow, u16) -> result`.
extern "wasi:http/types@0.2.6" fn @"[method]outgoing-response.set-status-code"(self: i32, status: i32) i32;

/// `[method]outgoing-response.body(borrow) -> result<own<outgoing-body>>`.
/// retptr → [disc, body_handle].
extern "wasi:http/types@0.2.6" fn @"[method]outgoing-response.body"(self: i32, retptr: i32) void;

/// `[method]outgoing-body.write(borrow) -> result<own<output-stream>>`.
/// retptr → [disc, stream_handle].
extern "wasi:http/types@0.2.6" fn @"[method]outgoing-body.write"(self: i32, retptr: i32) void;

/// `[static]outgoing-body.finish(own<outgoing-body>, option<own<fields>>)
///   -> result<_, error-code>`. `option<own<fields>>` lowers to
/// (disc, value); result is up to 5 i32s → retptr.
extern "wasi:http/types@0.2.6" fn @"[static]outgoing-body.finish"(this: i32, trailers_disc: i32, trailers_val: i32, retptr: i32) void;

/// `[method]incoming-request.method(borrow) -> method`. The `method`
/// variant's widest arm is `other(string)`, so the flat result is 3
/// i32s → retptr [disc, ptr, len]. For the named methods the disc alone
/// is meaningful.
extern "wasi:http/types@0.2.6" fn @"[method]incoming-request.method"(self: i32, retptr: i32) void;

/// `[method]incoming-request.path-with-query(borrow) -> option<string>`.
/// retptr → [disc, ptr, len].
extern "wasi:http/types@0.2.6" fn @"[method]incoming-request.path-with-query"(self: i32, retptr: i32) void;

/// `[method]incoming-request.consume(borrow) -> result<own<incoming-body>>`.
/// retptr → [disc, body_handle].
extern "wasi:http/types@0.2.6" fn @"[method]incoming-request.consume"(self: i32, retptr: i32) void;

/// `[method]incoming-body.stream(borrow) -> result<own<input-stream>>`.
/// retptr → [disc, stream_handle].
extern "wasi:http/types@0.2.6" fn @"[method]incoming-body.stream"(self: i32, retptr: i32) void;

/// `[static]response-outparam.set(own<response-outparam>,
///   result<own<outgoing-response>, error-code>) -> ()`. Result lowers
/// to (disc, payload[0..3]); the ok arm packs payload[0]=own<outgoing-response>.
extern "wasi:http/types@0.2.6" fn @"[static]response-outparam.set"(
    outparam: i32,
    resp_disc: i32,
    resp_p0: i32,
    resp_p1: i32,
    resp_p2: i32,
    resp_p3: i32,
) void;

// `wasi:io/streams@0.2.6`.

/// `[method]input-stream.blocking-read(borrow, u64) -> result<list<u8>, stream-error>`.
/// `u64` lowers to i64; result is 3 i32s → retptr [disc, ptr, len].
/// End-of-stream surfaces as the err arm (disc=1).
extern "wasi:io/streams@0.2.6" fn @"[method]input-stream.blocking-read"(self: i32, len: i64, retptr: i32) void;

/// `[method]output-stream.blocking-write-and-flush(borrow, list<u8>) -> result<_, stream-error>`.
/// retptr → [disc, stream_err_disc].
extern "wasi:io/streams@0.2.6" fn @"[method]output-stream.blocking-write-and-flush"(
    self: i32,
    contents_ptr: i32,
    contents_len: i32,
    retptr: i32,
) void;

// ── Public types ───────────────────────────────────────────────────

/// HTTP request method. Discriminants match the `wasi:http/types`
/// `method` variant lowering (get=0 … patch=8); any extension method
/// the host could not classify lifts as `.other`.
pub const Method = enum(u8) {
    get = 0,
    head = 1,
    post = 2,
    put = 3,
    delete = 4,
    connect = 5,
    options = 6,
    trace = 7,
    patch = 8,
    other = 9,
};

/// A borrowed incoming request. Methods read host-owned fields lazily;
/// returned slices borrow from the scratch arena and stay valid until
/// the next `resetScratch` (i.e. for the duration of the handler).
pub const Request = struct {
    handle: i32,

    pub fn method(self: Request) Method {
        @"[method]incoming-request.method"(self.handle, retPtr());
        const disc = retWords()[0];
        return if (disc <= 9) @enumFromInt(@as(u8, @intCast(disc))) else .other;
    }

    /// `path-with-query`, e.g. `/pets/1?nameFilter=Ball`. Null only if
    /// the host reports `none` (rare for a real request).
    pub fn path(self: Request) ?[]const u8 {
        @"[method]incoming-request.path-with-query"(self.handle, retPtr());
        return readOptionString();
    }

    /// Read the full request body into `buf`, returning the filled
    /// prefix (possibly empty), or null if the body could not be
    /// consumed. Reads in ≤4 KiB chunks until end-of-stream or `buf`
    /// fills.
    pub fn readBody(self: Request, buf: []u8) ?[]const u8 {
        @"[method]incoming-request.consume"(self.handle, retPtr());
        const body_handle = readResultHandle() orelse return null;

        @"[method]incoming-body.stream"(body_handle, retPtr());
        const stream_handle = readResultHandle() orelse return null;

        var len: usize = 0;
        while (len < buf.len) {
            @"[method]input-stream.blocking-read"(stream_handle, 4096, retPtr());
            const w = retWords();
            if (w[0] != 0) break; // err arm = end-of-stream (closed)
            const chunk_len: usize = w[2];
            if (chunk_len == 0) break;
            const src: [*]const u8 = @ptrFromInt(w[1]);
            const take = @min(chunk_len, buf.len - len);
            @memcpy(buf[len..][0..take], src[0..take]);
            len += take;
            if (take < chunk_len) break;
        }
        return buf[0..len];
    }
};

/// Sink for the single response a handler produces. Call exactly one of
/// the `respond*` methods; the `exportIncomingHandler` wrapper delivers
/// a `500` automatically if the handler returns without responding.
pub const Responder = struct {
    outp: i32,
    sent: bool = false,

    /// Send `status` + `body` with no response headers.
    pub fn respond(self: *Responder, status: u16, body: []const u8) void {
        if (self.sent) return;
        self.sent = true;
        deliver(self.outp, status, @"[constructor]fields"(), body);
    }

    /// Send `status` + `body` with a single `content-type` header. This
    /// is the only path that references `fields.append`, so handlers
    /// that never call it do not import that host function.
    pub fn respondWithContentType(
        self: *Responder,
        status: u16,
        content_type: []const u8,
        body: []const u8,
    ) void {
        if (self.sent) return;
        self.sent = true;
        const headers = @"[constructor]fields"();
        const name = "content-type";
        @"[method]fields.append"(
            headers,
            @intCast(@intFromPtr(name.ptr)),
            @intCast(name.len),
            @intCast(@intFromPtr(content_type.ptr)),
            @intCast(content_type.len),
            retPtr(),
        );
        deliver(self.outp, status, headers, body);
    }
};

// ── Response delivery (shared, header-agnostic) ────────────────────

fn deliver(outp: i32, status: u16, headers: i32, body: []const u8) void {
    const response = @"[constructor]outgoing-response"(headers);

    if (status != 200) {
        _ = @"[method]outgoing-response.set-status-code"(response, @as(i32, status));
    }

    @"[method]outgoing-response.body"(response, retPtr());
    const body_handle = readResultHandle() orelse return deliverErr(outp);

    @"[method]outgoing-body.write"(body_handle, retPtr());
    const stream_handle = readResultHandle() orelse return deliverErr(outp);

    if (body.len > 0) {
        @"[method]output-stream.blocking-write-and-flush"(
            stream_handle,
            @intCast(@intFromPtr(body.ptr)),
            @intCast(body.len),
            retPtr(),
        );
    }

    // option<own<trailers>> = none → (disc=0, val=0).
    @"[static]outgoing-body.finish"(body_handle, 0, 0, retPtr());
    // ok(response) = (disc=0, payload[0]=response, rest=0).
    @"[static]response-outparam.set"(outp, 0, response, 0, 0, 0);
}

/// Deliver `err(internal-error(none))` if response construction tripped.
/// disc=1 (err), payload[0]=error-code-disc=0 (internal-error),
/// payload[1]=opt-disc=0 (none).
fn deliverErr(outp: i32) void {
    @"[static]response-outparam.set"(outp, 1, 0, 0, 0, 0);
}

// ── comptime export wiring ─────────────────────────────────────────

/// Emit the canonical `wasi:http/incoming-handler@0.2.6#handle` export
/// that dispatches to `handler`. Call once at file scope:
///
/// ```zig
/// comptime { wit.exportIncomingHandler(handle); }
/// fn handle(req: wit.Request, res: *wit.Responder) void { ... }
/// ```
///
/// The wrapper resets the scratch arena before each call and, as a
/// safety net, delivers a `500` if `handler` returns without responding.
pub fn exportIncomingHandler(comptime handler: fn (req: Request, res: *Responder) void) void {
    const Wrapper = struct {
        fn entry(req_handle: i32, outp: i32) callconv(.c) void {
            resetScratch();
            var res = Responder{ .outp = outp };
            handler(.{ .handle = req_handle }, &res);
            if (!res.sent) res.respond(500, "");
        }
    };
    @export(&Wrapper.entry, .{ .name = "wasi:http/incoming-handler@0.2.6#handle" });
}
