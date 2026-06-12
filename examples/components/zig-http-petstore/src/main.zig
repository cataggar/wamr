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
//! The canonical-ABI plumbing (host imports, ret-area decoding, the
//! `cabi_realloc` arena, and the `wasi:http/incoming-handler` export
//! wiring) lives in the shared `wit_http` helper module (imported as
//! `@import("wit_http")`; source at `src/guest/wit_http.zig`). This
//! file is just the petstore routing + JSON logic. See that module's
//! doc comment for how the canonical ABI is bridged.
//!
//! See `../README.md` for the WIT layout and build pipeline.

const std = @import("std");
const wit = @import("wit_http");

comptime {
    wit.exportIncomingHandler(handle);
}

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

fn route(req: wit.Request, method: wit.Method, path: []const u8, query: []const u8) Response {
    if (std.mem.eql(u8, path, "/pets")) {
        return switch (method) {
            .get => listPets(),
            .post => createPet(req),
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
            .get => readPet(id),
            .delete => deletePet(id),
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

fn createPet(req: wit.Request) Response {
    const body = req.readBody(&body_buf) orelse
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

// Fixed buffer the request body is read into (see `wit.Request.readBody`).
var body_buf: [8192]u8 = undefined;

fn errorResponse(status: u16, message: []const u8) Response {
    return .{ .status = status, .body = toJson(ErrorBody{ .code = status, .message = message }) };
}

// ── Handler entry point ────────────────────────────────────────────

/// Dispatched by `wit.exportIncomingHandler` (see the `comptime` block
/// at the top). The wrapper has already reset the scratch arena.
fn handle(req: wit.Request, res: *wit.Responder) void {
    ensureSeeded();

    const full_path = req.path() orelse "/";

    // Split off the query string.
    var path = full_path;
    var query: []const u8 = "";
    if (std.mem.indexOfScalar(u8, full_path, '?')) |q| {
        path = full_path[0..q];
        query = full_path[q + 1 ..];
    }

    const resp = route(req, req.method(), path, query);
    res.respondWithContentType(resp.status, "application/json", resp.body);
}
