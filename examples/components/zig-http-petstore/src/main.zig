//! Zig WASI HTTP component implementing the Microsoft TypeSpec
//! **petstore** sample API:
//!
//!   https://github.com/microsoft/typespec/blob/main/packages/samples/specs/petstore/petstore.tsp
//!
//! Routes (all bodies are `application/json`):
//!
//!   GET    /pets                 -> 200 ResponsePage<Pet>  {"items":[…]}
//!   POST   /pets        (Pet)    -> 200 Pet                (echoes created pet)
//!   GET    /pets/{petId}         -> 200 Pet | 404 Error
//!   DELETE /pets/{petId}         -> 200 (empty) | 404 Error
//!   GET    /pets/{petId}/toys    -> 200 ResponsePage<Toy>  (?nameFilter= filters)
//!   anything else                -> 404 Error
//!
//! Runs end-to-end through wamr:
//!
//!   wamr run --listen=127.0.0.1:8080 zig-http-petstore.component.wasm
//!   curl -i http://127.0.0.1:8080/pets
//!
//! Like the sibling `zig-http` example, the handler is hand-written
//! canonical ABI: each call into `wasi:http/types@0.2.6` /
//! `wasi:io/streams@0.2.6` declares the lowered core signature directly
//! (no Zig wit-bindgen equivalent exists). Lowered sigs follow
//! `MAX_FLAT_PARAMS=16`, `MAX_FLAT_RESULTS=1` — any result whose flat
//! representation exceeds 1 core value is returned via a guest-allocated
//! ret-area pointer passed as the last param.
//!
//! See `../README.md` for the WIT layout and build pipeline.

const std = @import("std");

// ── cabi_realloc bump arena ────────────────────────────────────────
//
// Canonical-ABI lifts of host-side `string` / `list` values into guest
// memory call `cabi_realloc`; the host materializes the request path,
// the request method's `other(string)` payload, and each request-body
// read chunk here. The arena resets at the top of every `handle` call.

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

/// `[constructor]fields() -> own<fields>`.
extern "wasi:http/types@0.2.6" fn @"[constructor]fields"() i32;

/// `[method]fields.append(borrow, field-name (string), field-value (list<u8>))
///   -> result<_, header-error>`. `string` + `list<u8>` each lower to
/// (ptr, len); the `result<_, header-error>` is 2 i32s → retptr
/// [disc, header-err-disc]. The handler ignores the result.
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
/// i32s → retptr [disc, ptr, len]. For get/post/delete the disc alone
/// (0 / 2 / 4) is meaningful.
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
/// to (disc, payload[0..3]); ok packs payload[0]=own<outgoing-response>.
extern "wasi:http/types@0.2.6" fn @"[static]response-outparam.set"(
    outparam: i32,
    resp_disc: i32,
    resp_p0: i32,
    resp_p1: i32,
    resp_p2: i32,
    resp_p3: i32,
) void;

/// `[method]input-stream.blocking-read(borrow, u64) -> result<list<u8>, stream-error>`.
/// `u64` lowers to i64; result is 3 i32s → retptr [disc, ptr, len].
/// End-of-body surfaces as the err arm (disc=1).
extern "wasi:io/streams@0.2.6" fn @"[method]input-stream.blocking-read"(self: i32, len: i64, retptr: i32) void;

/// `[method]output-stream.blocking-write-and-flush(borrow, list<u8>) -> result<_, stream-error>`.
/// retptr → [disc, stream_err_disc].
extern "wasi:io/streams@0.2.6" fn @"[method]output-stream.blocking-write-and-flush"(
    self: i32,
    contents_ptr: i32,
    contents_len: i32,
    retptr: i32,
) void;

// ── In-memory pet store ────────────────────────────────────────────
//
// wamr instantiates the component once and reuses the instance across
// every request in the serve loop (`serveOneHttpConnection` keeps the
// same `inst`), so module-level state persists request-to-request. The
// store is seeded lazily on the first request. Created-pet strings are
// copied into `store_buf` (never reset) so they outlive the per-request
// arena.

const Pet = struct {
    id: i32,
    name: []const u8,
    tag: ?[]const u8,
    age: i32,
    present: bool,
};

const Toy = struct {
    id: i64,
    pet_id: i64,
    name: []const u8,
};

const MAX_PETS = 64;
var pets: [MAX_PETS]Pet = undefined;
var pets_len: usize = 0;
var next_id: i32 = 1;
var seeded = false;

var store_buf: [8192]u8 = undefined;
var store_top: usize = 0;

// Seeded toys are static (read-only) — the TypeSpec sample only exposes
// `list` for toys, never create/delete.
const toys = [_]Toy{
    .{ .id = 1, .pet_id = 1, .name = "Ball" },
    .{ .id = 2, .pet_id = 1, .name = "Mouse" },
    .{ .id = 3, .pet_id = 2, .name = "Bone" },
};

fn ensureSeeded() void {
    if (seeded) return;
    seeded = true;
    pets_len = 0;
    addPet(1, "Fluffy", "cat", 3);
    addPet(2, "Rex", null, 5);
    next_id = 3;
}

fn addPet(id: i32, name: []const u8, tag: ?[]const u8, age: i32) void {
    if (pets_len >= MAX_PETS) return;
    pets[pets_len] = .{ .id = id, .name = name, .tag = tag, .age = age, .present = true };
    pets_len += 1;
}

/// Copy `s` into the persistent store buffer. Returns the copy, or null
/// if the store is full.
fn storeDup(s: []const u8) ?[]const u8 {
    if (store_top + s.len > store_buf.len) return null;
    const dst = store_buf[store_top..][0..s.len];
    @memcpy(dst, s);
    store_top += s.len;
    return dst;
}

fn findPet(id: i32) ?*Pet {
    var i: usize = 0;
    while (i < pets_len) : (i += 1) {
        if (pets[i].present and pets[i].id == id) return &pets[i];
    }
    return null;
}

// ── JSON (de)serialization via std.json ─────────────────────────────
//
// Wire models mirror the TypeSpec petstore `.tsp`. They are distinct
// from the internal `Pet` / `Toy` storage structs above: the wire `Pet`
// carries no `id` (the spec's `Pet` model has none — `petId` is a path
// parameter), and field names use the exact casing the API emits
// (`petId`). The optional `tag` is omitted when absent via
// `emit_null_optional_fields = false`, matching `tag?` in the spec.

const PetWire = struct { name: []const u8, tag: ?[]const u8 = null, age: i32 };
const ToyWire = struct { id: i64, petId: i64, name: []const u8 };
const ErrorBody = struct { code: i32, message: []const u8 };

fn ResponsePage(comptime T: type) type {
    return struct { items: []const T };
}

// Response bodies are serialized into this fixed buffer; the returned
// `Response.body` slice borrows from it (one response per request).
var resp_buf: [16384]u8 = undefined;

/// Serialize `value` to JSON in `resp_buf` and return the written bytes.
/// On overflow the `std.Io.Writer.fixed` write fails and we return an
/// empty body.
fn toJson(value: anytype) []const u8 {
    var w = std.Io.Writer.fixed(&resp_buf);
    var s: std.json.Stringify = .{
        .writer = &w,
        .options = .{ .emit_null_optional_fields = false },
    };
    s.write(value) catch return "";
    return resp_buf[0..w.end];
}

/// Extract a query parameter value (`?key=value&…`). Returns a slice
/// borrowing from `query`. No percent-decoding (out of scope).
fn queryParam(query: []const u8, key: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, query, '&');
    while (it.next()) |pair| {
        if (std.mem.indexOfScalar(u8, pair, '=')) |eq| {
            if (std.mem.eql(u8, pair[0..eq], key)) return pair[eq + 1 ..];
        } else if (std.mem.eql(u8, pair, key)) {
            return "";
        }
    }
    return null;
}

// ── Routing ────────────────────────────────────────────────────────

const Response = struct {
    status: u16,
    body: []const u8,
};

// WIT `method` discriminants (host lowering): GET=0, POST=2, DELETE=4.
const METHOD_GET: u32 = 0;
const METHOD_POST: u32 = 2;
const METHOD_DELETE: u32 = 4;

const ret_words_t = [*]u32;

fn route(req: i32, method_disc: u32, path: []const u8, query: []const u8, ret_i32: i32, w: ret_words_t) Response {
    if (std.mem.eql(u8, path, "/pets")) {
        return switch (method_disc) {
            METHOD_GET => listPets(),
            METHOD_POST => createPet(req, ret_i32, w),
            else => errorResponse(405, "Method not allowed"),
        };
    }

    if (std.mem.startsWith(u8, path, "/pets/")) {
        const rest = path["/pets/".len..];
        if (std.mem.endsWith(u8, rest, "/toys")) {
            const id_str = rest[0 .. rest.len - "/toys".len];
            const id = std.fmt.parseInt(i64, id_str, 10) catch
                return errorResponse(404, "Pet not found");
            if (method_disc != METHOD_GET) return errorResponse(405, "Method not allowed");
            return listToys(id, query);
        }
        // A pure `/pets/{petId}` segment — reject any deeper path.
        if (std.mem.indexOfScalar(u8, rest, '/') != null) {
            return errorResponse(404, "Not found");
        }
        const id = std.fmt.parseInt(i32, rest, 10) catch
            return errorResponse(404, "Pet not found");
        return switch (method_disc) {
            METHOD_GET => readPet(id),
            METHOD_DELETE => deletePet(id),
            else => errorResponse(405, "Method not allowed"),
        };
    }

    return errorResponse(404, "Not found");
}

fn listPets() Response {
    var items: [MAX_PETS]PetWire = undefined;
    var n: usize = 0;
    var i: usize = 0;
    while (i < pets_len) : (i += 1) {
        if (!pets[i].present) continue;
        items[n] = .{ .name = pets[i].name, .tag = pets[i].tag, .age = pets[i].age };
        n += 1;
    }
    return .{ .status = 200, .body = toJson(ResponsePage(PetWire){ .items = items[0..n] }) };
}

fn readPet(id: i32) Response {
    const pet = findPet(id) orelse return errorResponse(404, "Pet not found");
    return .{ .status = 200, .body = toJson(PetWire{ .name = pet.name, .tag = pet.tag, .age = pet.age }) };
}

fn deletePet(id: i32) Response {
    const pet = findPet(id) orelse return errorResponse(404, "Pet not found");
    pet.present = false;
    return .{ .status = 200, .body = "" };
}

fn listToys(pet_id: i64, query: []const u8) Response {
    const filter = queryParam(query, "nameFilter");
    var items: [toys.len]ToyWire = undefined;
    var n: usize = 0;
    for (toys) |toy| {
        if (toy.pet_id != pet_id) continue;
        if (filter) |f| {
            if (f.len != 0 and std.mem.indexOf(u8, toy.name, f) == null) continue;
        }
        items[n] = .{ .id = toy.id, .petId = toy.pet_id, .name = toy.name };
        n += 1;
    }
    return .{ .status = 200, .body = toJson(ResponsePage(ToyWire){ .items = items[0..n] }) };
}

// Scratch arena for parsing the POST body. Reset (re-`init`ed) on every
// `createPet` call, so the parsed values are valid only until the next
// request — we copy the strings we keep into `store_buf` immediately.
var json_scratch: [8192]u8 = undefined;

fn createPet(req: i32, ret_i32: i32, w: ret_words_t) Response {
    const body = readRequestBody(req, ret_i32, w) orelse
        return errorResponse(400, "Missing or unreadable request body");

    var fba = std.heap.FixedBufferAllocator.init(&json_scratch);
    const parsed = std.json.parseFromSlice(PetWire, fba.allocator(), body, .{
        .ignore_unknown_fields = true,
    }) catch return errorResponse(400, "Invalid pet JSON");
    const incoming = parsed.value;
    if (incoming.name.len == 0)
        return errorResponse(400, "Invalid pet: 'name' is required");

    // Persist the strings (parsed values live in the per-request scratch).
    const name = storeDup(incoming.name) orelse return errorResponse(507, "Store full");
    const tag: ?[]const u8 = if (incoming.tag) |t|
        (storeDup(t) orelse return errorResponse(507, "Store full"))
    else
        null;
    const age = clampAge(incoming.age);

    const id = next_id;
    next_id += 1;
    addPet(id, name, tag, age);

    return .{ .status = 200, .body = toJson(PetWire{ .name = name, .tag = tag, .age = age }) };
}

fn clampAge(age: i32) i32 {
    // The TypeSpec model annotates `@minValue(0) @maxValue(20)`.
    if (age < 0) return 0;
    if (age > 20) return 20;
    return age;
}

/// Drive `consume` → `stream` → repeated `blocking-read` to pull the
/// full request body into a fixed buffer. Returns the body bytes, or
/// null if the body could not be consumed.
var body_buf: [8192]u8 = undefined;

fn readRequestBody(req: i32, ret_i32: i32, w: ret_words_t) ?[]const u8 {
    @"[method]incoming-request.consume"(req, ret_i32);
    if (w[0] != 0) return null;
    const body_handle: i32 = @bitCast(w[1]);

    @"[method]incoming-body.stream"(body_handle, ret_i32);
    if (w[0] != 0) return null;
    const stream_handle: i32 = @bitCast(w[1]);

    var len: usize = 0;
    while (len < body_buf.len) {
        @"[method]input-stream.blocking-read"(stream_handle, 4096, ret_i32);
        if (w[0] != 0) break; // err arm = end-of-stream (closed)
        const chunk_len: usize = w[2];
        if (chunk_len == 0) break;
        const src: [*]const u8 = @ptrFromInt(w[1]);
        const take = @min(chunk_len, body_buf.len - len);
        @memcpy(body_buf[len..][0..take], src[0..take]);
        len += take;
        if (take < chunk_len) break;
    }
    return body_buf[0..len];
}

fn errorResponse(status: u16, message: []const u8) Response {
    return .{ .status = status, .body = toJson(ErrorBody{ .code = status, .message = message }) };
}

// ── Handler entry point ────────────────────────────────────────────

export fn @"wasi:http/incoming-handler@0.2.6#handle"(req: i32, outp: i32) void {
    arena_top = 0;
    ensureSeeded();

    // 24-byte ret-area, big enough for every canon return we make
    // (method / blocking-read need 12; finish needs 20).
    const ret = arenaAlloc(u8, 24, 8);
    const ret_i32: i32 = @intCast(@intFromPtr(ret));
    const w: ret_words_t = @ptrCast(@alignCast(ret));

    // Read the method discriminant.
    @"[method]incoming-request.method"(req, ret_i32);
    const method_disc: u32 = w[0];

    // Read the request path (option<string>).
    @"[method]incoming-request.path-with-query"(req, ret_i32);
    var full_path: []const u8 = "/";
    if (w[0] == 1) {
        const p: [*]const u8 = @ptrFromInt(w[1]);
        full_path = p[0..w[2]];
    }

    // Split off the query string.
    var path = full_path;
    var query: []const u8 = "";
    if (std.mem.indexOfScalar(u8, full_path, '?')) |q| {
        path = full_path[0..q];
        query = full_path[q + 1 ..];
    }

    const resp = route(req, method_disc, path, query, ret_i32, w);
    deliver(outp, ret_i32, w, resp);
}

/// Build and deliver the `outgoing-response` (with a JSON content-type
/// header) back to the host via `response-outparam.set`.
fn deliver(outp: i32, ret_i32: i32, w: ret_words_t, resp: Response) void {
    const headers = @"[constructor]fields"();

    // Best-effort `content-type: application/json` (ignore the result).
    const ct_name = "content-type";
    const ct_value = "application/json";
    @"[method]fields.append"(
        headers,
        @intCast(@intFromPtr(ct_name.ptr)),
        @intCast(ct_name.len),
        @intCast(@intFromPtr(ct_value.ptr)),
        @intCast(ct_value.len),
        ret_i32,
    );

    const response = @"[constructor]outgoing-response"(headers);

    if (resp.status != 200) {
        _ = @"[method]outgoing-response.set-status-code"(response, @as(i32, resp.status));
    }

    @"[method]outgoing-response.body"(response, ret_i32);
    if (w[0] != 0) {
        deliverErr(outp);
        return;
    }
    const body_handle: i32 = @bitCast(w[1]);

    @"[method]outgoing-body.write"(body_handle, ret_i32);
    if (w[0] != 0) {
        deliverErr(outp);
        return;
    }
    const stream_handle: i32 = @bitCast(w[1]);

    if (resp.body.len > 0) {
        @"[method]output-stream.blocking-write-and-flush"(
            stream_handle,
            @intCast(@intFromPtr(resp.body.ptr)),
            @intCast(resp.body.len),
            ret_i32,
        );
    }

    @"[static]outgoing-body.finish"(body_handle, 0, 0, ret_i32);
    @"[static]response-outparam.set"(outp, 0, response, 0, 0, 0);
}

/// Best-effort error delivery: `err(internal-error(none))`.
fn deliverErr(outp: i32) void {
    @"[static]response-outparam.set"(outp, 1, 0, 0, 0, 0);
}
