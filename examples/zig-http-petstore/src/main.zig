//! Zig WASI HTTP component implementing the Microsoft TypeSpec
//! **petstore** sample API, with state persisted in `wasi:keyvalue`:
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
//! Why keyvalue: pets live in a host-side `wasi:keyvalue` bucket, not in
//! guest globals. A component's linear memory does not necessarily
//! survive across requests — `wasmtime serve` instantiates a fresh
//! instance per request — so cross-request state must live in the host.
//! This makes the example behave identically under wamr's serial serve
//! loop and under `wasmtime serve`.
//!
//! Run under wamr (in-memory keyvalue store for the server lifetime):
//!
//!   wamrc serve --addr 127.0.0.1:8080 zig-http-petstore.wasm
//!   curl -i http://127.0.0.1:8080/pets
//!
//! Run under wasmtime:
//!
//!   wasmtime serve -S keyvalue zig-http-petstore.wasm
//!
//! The canonical-ABI plumbing lives in the shared guest helper modules
//! (`@import("wasi_http")` + `@import("wasi_keyvalue")`, both over the
//! `abi` module; sources under `src/guest/`). This file is just the
//! petstore routing + JSON + storage logic.
//!
//! See `../README.md` for the WIT layout and build pipeline.

const std = @import("std");
const http = @import("wasi_http");
const kv = @import("wasi_keyvalue");

comptime {
    http.exportIncomingHandler(handle);
}

// ── Storage layout (keyvalue bucket) ───────────────────────────────
//
// One bucket named "petstore" holds:
//   * `next_id`  → ASCII decimal of the next id to assign.
//   * `ids`      → comma-separated decimal ids of all live pets, in
//                  insertion order (used to enumerate for GET /pets).
//   * `pet:<id>` → the pet's canonical JSON (`{"name":…,"age":…}`),
//                  stored exactly as it is served.
// The bucket is seeded with Fluffy + Rex on first use.

const BUCKET = "petstore";

// Toys are static read-only seed data — the TypeSpec sample only
// exposes `list` for toys, never create/delete, so they don't need the
// keyvalue store.
const toys = [_]ToyWire{
    .{ .id = 1, .petId = 1, .name = "Ball" },
    .{ .id = 2, .petId = 1, .name = "Mouse" },
    .{ .id = 3, .petId = 2, .name = "Bone" },
};

// ── JSON wire models (mirror the TypeSpec petstore `.tsp`) ─────────
//
// The wire `Pet` carries no `id` (the spec's `Pet` has none — `petId`
// is a path parameter); field names use the API's casing (`petId`). The
// optional `tag` is omitted when absent via `emit_null_optional_fields`,
// matching `tag?` in the spec.

const PetWire = struct { name: []const u8, tag: ?[]const u8 = null, age: i32 };
const ToyWire = struct { id: i64, petId: i64, name: []const u8 };
const ErrorBody = struct { code: i32, message: []const u8 };

fn ResponsePage(comptime T: type) type {
    return struct { items: []const T };
}

// ── Fixed buffers ──────────────────────────────────────────────────
//
// One response per request, so a single response buffer is enough.
// `pet_buf` holds a single pet's canonical JSON while it is being stored
// (kept distinct from `resp_buf` so storing during a GET /pets loop
// doesn't clobber the response being assembled).

var resp_buf: [16384]u8 = undefined;
var pet_buf: [1024]u8 = undefined;
var ids_buf: [4096]u8 = undefined;
var json_parse_buf: [8192]u8 = undefined;

/// Serialize `value` to JSON in `buf`, returning the written bytes (or
/// empty on overflow).
fn toJsonBuf(buf: []u8, value: anytype) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    var s: std.json.Stringify = .{
        .writer = &w,
        .options = .{ .emit_null_optional_fields = false },
    };
    s.write(value) catch return "";
    return buf[0..w.end];
}

fn toJson(value: anytype) []const u8 {
    return toJsonBuf(&resp_buf, value);
}

// ── Handler entry point ────────────────────────────────────────────

/// Dispatched by `http.exportIncomingHandler`. The wrapper has already
/// reset the scratch arena.
fn handle(req: http.Request, res: *http.Responder) void {
    const bucket = store() orelse {
        res.respondWithContentType(500, "application/json", toJson(ErrorBody{ .code = 500, .message = "key-value store unavailable" }));
        return;
    };
    ensureSeeded(bucket);

    const full_path = req.path() orelse "/";
    var path = full_path;
    var query: []const u8 = "";
    if (std.mem.indexOfScalar(u8, full_path, '?')) |q| {
        path = full_path[0..q];
        query = full_path[q + 1 ..];
    }

    const resp = route(bucket, req, req.method(), path, query);
    res.respondWithContentType(resp.status, "application/json", resp.body);
}

// The bucket handle is opened once per component instance and cached.
// Handles are not explicitly dropped (resource-drop is a canonical
// built-in, not a portable host import); caching avoids accumulating one
// handle per request on runtimes that reuse a single instance (wamr),
// and is harmless on runtimes that re-instantiate per request (wasmtime
// serve), where the global simply re-opens once per fresh instance.
var cached_bucket: ?kv.Bucket = null;

fn store() ?kv.Bucket {
    if (cached_bucket) |b| return b;
    const b = kv.open(BUCKET) orelse return null;
    cached_bucket = b;
    return b;
}

const Response = struct {
    status: u16,
    body: []const u8,
};

fn route(bucket: kv.Bucket, req: http.Request, method: http.Method, path: []const u8, query: []const u8) Response {
    if (std.mem.eql(u8, path, "/pets")) {
        return switch (method) {
            .get => listPets(bucket),
            .post => createPet(bucket, req),
            else => errorResponse(405, "Method not allowed"),
        };
    }

    if (std.mem.startsWith(u8, path, "/pets/")) {
        const rest = path["/pets/".len..];
        if (std.mem.endsWith(u8, rest, "/toys")) {
            const id_str = rest[0 .. rest.len - "/toys".len];
            const id = std.fmt.parseInt(i64, id_str, 10) catch
                return errorResponse(404, "Pet not found");
            if (method != .get) return errorResponse(405, "Method not allowed");
            return listToys(id, query);
        }
        // A pure `/pets/{petId}` segment — reject any deeper path.
        if (std.mem.indexOfScalar(u8, rest, '/') != null) {
            return errorResponse(404, "Not found");
        }
        const id = std.fmt.parseInt(i32, rest, 10) catch
            return errorResponse(404, "Pet not found");
        return switch (method) {
            .get => readPet(bucket, id),
            .delete => deletePet(bucket, id),
            else => errorResponse(405, "Method not allowed"),
        };
    }

    return errorResponse(404, "Not found");
}

// ── Routes ─────────────────────────────────────────────────────────

fn listPets(bucket: kv.Bucket) Response {
    const ids = bucket.get("ids") orelse "";
    var w = std.Io.Writer.fixed(&resp_buf);
    w.writeAll("{\"items\":[") catch return errorResponse(500, "response too large");
    var first = true;
    var it = std.mem.splitScalar(u8, ids, ',');
    while (it.next()) |id_str| {
        if (id_str.len == 0) continue;
        const pet_json = petJson(bucket, id_str) orelse continue;
        if (!first) w.writeByte(',') catch return errorResponse(500, "response too large");
        first = false;
        w.writeAll(pet_json) catch return errorResponse(500, "response too large");
    }
    w.writeAll("]}") catch return errorResponse(500, "response too large");
    return .{ .status = 200, .body = resp_buf[0..w.end] };
}

fn readPet(bucket: kv.Bucket, id: i32) Response {
    var key_buf: [32]u8 = undefined;
    const key = petKey(&key_buf, id);
    const pet_json = bucket.get(key) orelse return errorResponse(404, "Pet not found");
    // The stored value is already the pet's canonical JSON. Copy it into
    // `resp_buf` so the returned slice doesn't alias the scratch arena
    // (which a later helper call could bump past).
    if (pet_json.len > resp_buf.len) return errorResponse(500, "response too large");
    @memcpy(resp_buf[0..pet_json.len], pet_json);
    return .{ .status = 200, .body = resp_buf[0..pet_json.len] };
}

fn deletePet(bucket: kv.Bucket, id: i32) Response {
    var key_buf: [32]u8 = undefined;
    const key = petKey(&key_buf, id);
    if (!bucket.exists(key)) return errorResponse(404, "Pet not found");
    _ = bucket.delete(key);
    removeId(bucket, id);
    return .{ .status = 200, .body = "" };
}

fn createPet(bucket: kv.Bucket, req: http.Request) Response {
    var body_buf: [8192]u8 = undefined;
    const body = req.readBody(&body_buf) orelse
        return errorResponse(400, "Missing or unreadable request body");

    var fba = std.heap.FixedBufferAllocator.init(&json_parse_buf);
    const parsed = std.json.parseFromSlice(PetWire, fba.allocator(), body, .{
        .ignore_unknown_fields = true,
    }) catch return errorResponse(400, "Invalid pet JSON");
    const incoming = parsed.value;
    if (incoming.name.len == 0)
        return errorResponse(400, "Invalid pet: 'name' is required");

    const id = nextId(bucket);
    const pet = PetWire{ .name = incoming.name, .tag = incoming.tag, .age = clampAge(incoming.age) };
    const pet_json = toJsonBuf(&pet_buf, pet);

    var key_buf: [32]u8 = undefined;
    const key = petKey(&key_buf, id);
    if (!bucket.set(key, pet_json)) return errorResponse(507, "store write failed");
    appendId(bucket, id);
    bumpNextId(bucket, id + 1);

    return .{ .status = 200, .body = pet_json };
}

fn listToys(pet_id: i64, query: []const u8) Response {
    const filter = queryParam(query, "nameFilter");
    var items: [toys.len]ToyWire = undefined;
    var n: usize = 0;
    for (toys) |toy| {
        if (toy.petId != pet_id) continue;
        if (filter) |f| {
            if (f.len != 0 and std.mem.indexOf(u8, toy.name, f) == null) continue;
        }
        items[n] = toy;
        n += 1;
    }
    return .{ .status = 200, .body = toJson(ResponsePage(ToyWire){ .items = items[0..n] }) };
}

fn errorResponse(status: u16, message: []const u8) Response {
    return .{ .status = status, .body = toJson(ErrorBody{ .code = status, .message = message }) };
}

// ── Storage helpers ────────────────────────────────────────────────

/// Seed Fluffy + Rex on first use (detected by the absence of `next_id`).
fn ensureSeeded(bucket: kv.Bucket) void {
    if (bucket.exists("next_id")) return;
    var key_buf: [32]u8 = undefined;
    _ = bucket.set(petKey(&key_buf, 1), toJsonBuf(&pet_buf, PetWire{ .name = "Fluffy", .tag = "cat", .age = 3 }));
    _ = bucket.set(petKey(&key_buf, 2), toJsonBuf(&pet_buf, PetWire{ .name = "Rex", .tag = null, .age = 5 }));
    _ = bucket.set("ids", "1,2");
    _ = bucket.set("next_id", "3");
}

/// Fetch `pet:<id_str>`'s stored JSON (already canonical; no decode).
fn petJson(bucket: kv.Bucket, id_str: []const u8) ?[]const u8 {
    var key_buf: [40]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "pet:{s}", .{id_str}) catch return null;
    return bucket.get(key);
}

fn petKey(buf: []u8, id: i32) []const u8 {
    return std.fmt.bufPrint(buf, "pet:{d}", .{id}) catch unreachable;
}

fn nextId(bucket: kv.Bucket) i32 {
    const v = bucket.get("next_id") orelse return 1;
    return std.fmt.parseInt(i32, v, 10) catch 1;
}

fn bumpNextId(bucket: kv.Bucket, value: i32) void {
    var buf: [16]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{d}", .{value}) catch return;
    _ = bucket.set("next_id", s);
}

/// Append `id` to the comma-separated `ids` index.
fn appendId(bucket: kv.Bucket, id: i32) void {
    const cur = bucket.get("ids") orelse "";
    var w = std.Io.Writer.fixed(&ids_buf);
    w.writeAll(cur) catch return;
    if (cur.len != 0) w.writeByte(',') catch return;
    w.print("{d}", .{id}) catch return;
    _ = bucket.set("ids", ids_buf[0..w.end]);
}

/// Remove `id` from the comma-separated `ids` index.
fn removeId(bucket: kv.Bucket, id: i32) void {
    const cur = bucket.get("ids") orelse return;
    var w = std.Io.Writer.fixed(&ids_buf);
    var first = true;
    var it = std.mem.splitScalar(u8, cur, ',');
    while (it.next()) |s| {
        if (s.len == 0) continue;
        const v = std.fmt.parseInt(i32, s, 10) catch continue;
        if (v == id) continue;
        if (!first) w.writeByte(',') catch return;
        first = false;
        w.writeAll(s) catch return;
    }
    _ = bucket.set("ids", ids_buf[0..w.end]);
}

fn clampAge(age: i32) i32 {
    // The TypeSpec model annotates `@minValue(0) @maxValue(20)`.
    if (age < 0) return 0;
    if (age > 20) return 20;
    return age;
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
