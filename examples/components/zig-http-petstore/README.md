# zig-http-petstore

Pure-Zig WebAssembly component implementing the Microsoft TypeSpec
[petstore sample API][tsp] over `wasi:http/incoming-handler@0.2.6`.

[tsp]: https://github.com/microsoft/typespec/blob/main/packages/samples/specs/petstore/petstore.tsp

It is the REST counterpart to the sibling [`zig-http`](../zig-http/)
"hello world" example — a strict superset that additionally reads the
request **method** and **body**, routes on the path, and returns JSON
with a `content-type: application/json` header.

## API

All request/response bodies are `application/json`.

| Method   | Route                 | Behaviour                                              |
| -------- | --------------------- | ----------------------------------------------------- |
| `GET`    | `/pets`               | `200` `ResponsePage<Pet>` — `{"items":[…]}`           |
| `POST`   | `/pets`               | `200` `Pet` — parses the JSON body, echoes the pet    |
| `GET`    | `/pets/{petId}`       | `200` `Pet`, or `404` `Error` if unknown              |
| `DELETE` | `/pets/{petId}`       | `200` (empty), or `404` `Error` if unknown            |
| `GET`    | `/pets/{petId}/toys`  | `200` `ResponsePage<Toy>` — `?nameFilter=` substring  |
| *(other)*| *(any)*               | `404` `Error`                                         |

Models mirror the TypeSpec sample:

```jsonc
// Pet
{ "name": "Fluffy", "tag": "cat", "age": 3 }   // tag is optional; age ∈ [0, 20]
// Toy
{ "id": 1, "petId": 1, "name": "Ball" }
// Error
{ "code": 404, "message": "Pet not found" }
```

The store is seeded with two pets (`Fluffy` / `Rex`) and three toys, and
is held in module-level state. wamr instantiates the component once and
reuses the instance across requests, so `POST` / `DELETE` mutations
persist for the lifetime of the server process.

## Run

```console
$ ./zig-out/bin/wamr run --listen=127.0.0.1:8080 \
        zig-out/component-examples/zig-http-petstore.component.wasm &

$ curl -s http://127.0.0.1:8080/pets
{"items":[{"name":"Fluffy","tag":"cat","age":3},{"name":"Rex","age":5}]}

$ curl -s http://127.0.0.1:8080/pets/1
{"name":"Fluffy","tag":"cat","age":3}

$ curl -s -X POST -H 'content-type: application/json' \
        -d '{"name":"Buddy","tag":"dog","age":2}' http://127.0.0.1:8080/pets
{"name":"Buddy","tag":"dog","age":2}

$ curl -s 'http://127.0.0.1:8080/pets/1/toys?nameFilter=Mouse'
{"items":[{"id":2,"petId":1,"name":"Mouse"}]}

$ curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/pets/99
404
```

The repo's root `build.zig` automates everything:

```sh
zig build component-examples       # build + validate
zig build component-examples-run   # build + spin-up + smoke (Linux)
```

## WIT layout

```
wit/
├── world.wit                              # package example:zig-http-petstore
└── deps/
    ├── wasi-http/
    │   ├── incoming-handler.wit           # package wasi:http@0.2.6 (sorts first → carries the package decl)
    │   └── types.wit                      # no package decl (concatenated after)
    └── wasi-io/
        └── streams.wit                    # package wasi:io@0.2.6
```

Each dep is the **minimum subset** of the canonical upstream WIT that
the handler links against. Relative to `zig-http`, the petstore world
adds:

* `wasi:io/streams.input-stream` with `blocking-read` — reading the
  `POST /pets` request body.
* `wasi:http/types.incoming-request.method` / `.consume`,
  `wasi:http/types.incoming-body` with `%stream` — note `stream` is a
  WIT keyword, so it is written `%stream` (matching upstream wasi:http).
* `wasi:http/types.fields.append` — setting the `content-type` header.

World-import order matters: `wasi:io/streams` is imported **before**
`wasi:http/types` because the latter's `use wasi:io/streams.{…}` clause
must see those resources already in the world's type-indexspace. See the
`zig-http` README for the full rationale on encoder ordering, multi-file
`package` placement, and the cataggar/wabt feature stack this relies on.

## Source walkthrough

The handler is hand-written canonical ABI — no Zig `wit-bindgen`
equivalent exists, so each call into `wasi:http/types@0.2.6` /
`wasi:io/streams@0.2.6` declares the lowered core signature directly at
the `extern "wasi:…"` declarations in [`src/main.zig`](src/main.zig).
The lowering rules (`MAX_FLAT_PARAMS=16`, `MAX_FLAT_RESULTS=1`) mean any
result wider than one core value is returned through a guest-allocated
ret-area pointer (the last param).

`cabi_realloc` is a bump arena (64 KiB, reset per request) the host uses
to materialize the request path, the request method's `other(string)`
payload, and each `POST` body read chunk. Response JSON is built into a
separate fixed buffer, and created-pet strings are copied into a
persistent store buffer so they outlive the per-request arena.

Request flow:

1. `incoming-request.method` → route on `GET` / `POST` / `DELETE`.
2. `incoming-request.path-with-query` → split path + query, match route.
3. For `POST /pets`: `incoming-request.consume` →
   `incoming-body.stream` → repeated `input-stream.blocking-read` until
   the `closed` (err) arm, then `std.json.parseFromSlice` (over a
   `FixedBufferAllocator`) decodes the body into a `Pet` wire struct.
4. Build the JSON response with `std.json.Stringify`, `fields.append`
   `content-type`, construct the `outgoing-response`, optionally bump
   `set-status-code`, write the body via `outgoing-body` →
   `output-stream.blocking-write-and-flush`, `outgoing-body.finish`, and
   deliver via `response-outparam.set`.

## Build pipeline

```sh
# 1. Core wasm with the canonical-ABI export + cabi_realloc.
zig build-exe -target wasm32-freestanding -O ReleaseSmall -fno-entry \
    --export="wasi:http/incoming-handler@0.2.6#handle" \
    --export=cabi_realloc \
    src/main.zig

# 2. Embed the WIT subset.
wabt component embed --world petstore wit main.wasm -o main.embed.wasm

# 3. Wrap into a component.
wabt component new main.embed.wasm -o zig-http-petstore.component.wasm
```

## Notes / gotchas

The same constraints documented in the `zig-http` README apply here
(empty-resource `{}` form, single `package` per WIT directory, explicit
world imports for every `use` target, `result<T>` one-arm form, and
host-side resource cleanup after each request). Two petstore-specific
points:

* `stream` is a reserved WIT keyword (the async `stream<T>` type), so
  `incoming-body`'s method is written `%stream` — the `%` escape encodes
  the plain name `stream`, which is what the host binds
  (`[method]incoming-body.stream`).
* JSON is handled by the Zig standard library (`std.json`): request
  bodies are decoded with `std.json.parseFromSlice` over a
  `FixedBufferAllocator`, and responses are produced with
  `std.json.Stringify` (with `emit_null_optional_fields = false` so the
  optional `tag` is omitted when absent, matching `tag?` in the spec).
  `std.json` compiles cleanly for `wasm32-freestanding` — it needs only
  an allocator, no OS surface.
