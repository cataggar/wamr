//! Zig WASI HTTP component — mirrors the bytecodealliance
//! [Rust HTTP-in-components tutorial][rust-http]:
//!
//!   GET /       -> 200 "Hello, world!\n"
//!   anything    -> 404
//!
//! Runs end-to-end through wamr:
//!
//!   wamr run --listen=127.0.0.1:8080 zig-http.component.wasm
//!   curl -i http://127.0.0.1:8080/
//!
//! The handler is hand-written canonical-ABI: each call into
//! `wasi:http/types@0.2.6` / `wasi:io/streams@0.2.6` declares the
//! lowered core signature directly (no Zig wit-bindgen equivalent
//! exists). Lowered sigs follow `MAX_FLAT_PARAMS=16`,
//! `MAX_FLAT_RESULTS=1` — any result whose flat representation
//! exceeds 1 core value is returned via a guest-allocated ret-area
//! pointer passed as the last param.
//!
//! [rust-http]: https://component-model.bytecodealliance.org/language-support/using-http-in-components/rust.html
//!
//! See `../README.md` for the WIT layout and build pipeline.

const std = @import("std");

// ── cabi_realloc bump arena ────────────────────────────────────────
//
// Canonical-ABI lifts of host-side `string` / `list` values into
// guest memory call `cabi_realloc`; we also use the same arena for
// our own ret-area buffers (see `retArea`). The arena resets at
// the top of every `handle` call so each request gets a fresh
// 64 KiB scratch surface — comfortably bigger than the
// Rust-tutorial-shaped responses + any path-with-query payload.

var arena_buf: [65536]u8 align(16) = undefined;
var arena_top: usize = 0;

inline fn alignUp(x: usize, a: usize) usize {
    return (x + a - 1) & ~(a - 1);
}

fn arenaAlloc(comptime T: type, n: usize, alignment: usize) [*]T {
    const a = if (alignment == 0) @alignOf(T) else alignment;
    const start = alignUp(arena_top, a);
    const bytes = n * @sizeOf(T);
    arena_top = start + bytes;
    return @ptrCast(@alignCast(&arena_buf[start]));
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

// ── Host imports (canonical-ABI lowered signatures) ────────────────

/// `[constructor]fields() -> own<fields>` — empty headers map.
extern "wasi:http/types@0.2.6" fn @"[constructor]fields"() i32;

/// `[constructor]outgoing-response(own<fields>) -> own<outgoing-response>`.
extern "wasi:http/types@0.2.6" fn @"[constructor]outgoing-response"(headers: i32) i32;

/// `[method]outgoing-response.set-status-code(borrow, u16) -> result`.
/// Bare `result` lowers to a single i32 discriminant (0 = ok, 1 = err).
extern "wasi:http/types@0.2.6" fn @"[method]outgoing-response.set-status-code"(self: i32, status: i32) i32;

/// `[method]outgoing-response.body(borrow) -> result<own<outgoing-body>>`.
/// Two-i32 flat result → spilled via the retptr last param.
/// Layout @ retptr: [disc:u32, body_handle:u32].
extern "wasi:http/types@0.2.6" fn @"[method]outgoing-response.body"(self: i32, retptr: i32) void;

/// `[method]outgoing-body.write(borrow) -> result<own<output-stream>>`.
/// Same shape as `outgoing-response.body`: retptr → [disc, stream_handle].
extern "wasi:http/types@0.2.6" fn @"[method]outgoing-body.write"(self: i32, retptr: i32) void;

/// `[static]outgoing-body.finish(own<outgoing-body>, option<own<fields>>) -> result<_, error-code>`.
/// `option<own<fields>>` lowers to (disc:i32, value:i32). The result is
/// `result<_, error-code>` where `error-code` is a variant whose largest
/// case is `internal-error(option<string>)` → flat (disc, opt_disc, ptr, len)
/// = 4 i32s. Total result flat = 5 i32s → retptr.
extern "wasi:http/types@0.2.6" fn @"[static]outgoing-body.finish"(this: i32, trailers_disc: i32, trailers_val: i32, retptr: i32) void;

/// `[method]incoming-request.path-with-query(borrow) -> option<string>`.
/// Result flat = (disc:i32, ptr:i32, len:i32) = 3 i32s → retptr.
extern "wasi:http/types@0.2.6" fn @"[method]incoming-request.path-with-query"(self: i32, retptr: i32) void;

/// `[static]response-outparam.set(own<response-outparam>, result<own<outgoing-response>, error-code>) -> ()`.
///
/// The result lowers to (disc:i32, payload[0..3]:i32) where the payload slots
/// are wide enough to hold the err-side `error-code` (4 i32s). For the ok arm
/// we pack only `payload[0] = own<outgoing-response>` and leave the rest zero.
/// Total flat params = 1 (outparam) + 5 (result) = 6 — within MAX_FLAT_PARAMS,
/// so no spill.
extern "wasi:http/types@0.2.6" fn @"[static]response-outparam.set"(
    outparam: i32,
    resp_disc: i32,
    resp_p0: i32,
    resp_p1: i32,
    resp_p2: i32,
    resp_p3: i32,
) void;

/// `[method]output-stream.blocking-write-and-flush(borrow, list<u8>) -> result<_, stream-error>`.
/// `list<u8>` lowers to (ptr:i32, len:i32). Result flat = (disc:i32, stream_err_disc:i32)
/// = 2 i32s → retptr.
extern "wasi:io/streams@0.2.6" fn @"[method]output-stream.blocking-write-and-flush"(
    self: i32,
    contents_ptr: i32,
    contents_len: i32,
    retptr: i32,
) void;

// ── Handler ────────────────────────────────────────────────────────

const HELLO_BODY = "Hello, world!\n";

export fn @"wasi:http/incoming-handler@0.2.6#handle"(req: i32, outp: i32) void {
    arena_top = 0;

    // Allocate a fixed 20-byte ret-area, large enough for every
    // canon return we make (path-with-query needs 12; finish needs 20).
    const ret = arenaAlloc(u8, 20, 4);
    const ret_i32: i32 = @intCast(@intFromPtr(ret));

    // 1. Read the request path. Result layout: [disc, ptr, len] at `ret`.
    @"[method]incoming-request.path-with-query"(req, ret_i32);
    const ret_words: [*]u32 = @ptrCast(@alignCast(ret));
    const path_present = ret_words[0] == 1;
    const path_ptr = ret_words[1];
    const path_len = ret_words[2];

    const is_root = blk: {
        if (!path_present) break :blk false;
        if (path_len != 1) break :blk false;
        const p: [*]const u8 = @ptrFromInt(path_ptr);
        break :blk p[0] == '/';
    };

    const status_code: u16 = if (is_root) 200 else 404;
    const body: []const u8 = if (is_root) HELLO_BODY else "";

    // 2. Build response.
    const headers = @"[constructor]fields"();
    const resp = @"[constructor]outgoing-response"(headers);

    // 3. Bump status if not 200.
    if (status_code != 200) {
        _ = @"[method]outgoing-response.set-status-code"(resp, @as(i32, status_code));
    }

    // 4. Acquire the outgoing-body. retptr → [disc, body_handle].
    @"[method]outgoing-response.body"(resp, ret_i32);
    if (ret_words[0] != 0) {
        // body() failed — fall back to delivering an err result.
        deliverErr(outp);
        return;
    }
    const body_handle: i32 = @bitCast(ret_words[1]);

    // 5. Acquire the output-stream from the body. Same retptr layout.
    @"[method]outgoing-body.write"(body_handle, ret_i32);
    if (ret_words[0] != 0) {
        deliverErr(outp);
        return;
    }
    const stream_handle: i32 = @bitCast(ret_words[1]);

    // 6. Push body bytes. result<_, stream-error> → retptr [disc, _].
    if (body.len > 0) {
        @"[method]output-stream.blocking-write-and-flush"(
            stream_handle,
            @intCast(@intFromPtr(body.ptr)),
            @intCast(body.len),
            ret_i32,
        );
    }

    // 7. Finish the body (option<own<trailers>> = none → disc=0, val=0).
    //    Per WIT, dropping the output-stream first is recommended, but
    //    wamr's host doesn't enforce it (httpOutgoingBodyFinish always
    //    returns ok). Skipping the drop keeps the handler shorter.
    @"[static]outgoing-body.finish"(body_handle, 0, 0, ret_i32);

    // 8. Deliver the response. ok(resp) = (disc=0, payload[0]=resp,
    //    payload[1..3]=0).
    @"[static]response-outparam.set"(outp, 0, resp, 0, 0, 0);
}

/// Best-effort error delivery if we trip during response construction.
/// Sends `err(internal-error(none))` through `response-outparam.set`.
/// disc=1 (err), payload[0]=error-code-disc=0 (internal-error),
/// payload[1]=opt-disc=0 (none), payload[2]=ptr (unused), payload[3]=len.
fn deliverErr(outp: i32) void {
    @"[static]response-outparam.set"(outp, 1, 0, 0, 0, 0);
}
