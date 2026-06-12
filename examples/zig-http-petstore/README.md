# zig-http-petstore

Pure-Zig WebAssembly component implementing the Microsoft TypeSpec
[petstore sample API][tsp] over `wasi:http/incoming-handler@0.2.6`.

[tsp]: https://github.com/microsoft/typespec/blob/main/packages/samples/specs/petstore/petstore.tsp

It is the REST counterpart to the sibling [`zig-http`](../zig-http/)
"hello world" example — it reads the request **method** and **body**,
routes on the path, returns JSON, and **persists pets in a host-backed
`wasi:keyvalue` store**. Unlike `zig-http` (which embeds a trimmed,
wamr-targeting WIT), this example embeds the **canonical** `wasi:http` /
`wasi:io` / `wasi:clocks` WIT, so the same binary is intended to run on
both wamr and `wasmtime serve` (see [Runtime status](#runtime-status)).

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

The store is seeded with two pets (`Fluffy` / `Rex`) and three toys.
Pets live in a `wasi:keyvalue` bucket (keys `next_id`, `ids`, `pet:<id>`),
not in guest globals — so `POST` / `DELETE` mutations persist across
requests **even on runtimes that re-instantiate the component per
request** (`wasmtime serve` does; wamr's serial loop reuses one
instance). Toys are static read-only seed data.

## Run

### wamr

Keyvalue is in-memory by default (persists for the server process); add
`--keyvalue-store=<file>` to persist to a JSON file across restarts.

```console
$ ./zig-out/bin/wamr run --listen=127.0.0.1:8080 \
        zig-out/examples/zig-http-petstore.wasm &

$ curl -s http://127.0.0.1:8080/pets
{"items":[{"name":"Fluffy","tag":"cat","age":3},{"name":"Rex","age":5}]}

$ curl -s -X POST -H 'content-type: application/json' \
        -d '{"name":"Buddy","tag":"dog","age":2}' http://127.0.0.1:8080/pets
{"name":"Buddy","tag":"dog","age":2}

$ curl -s 'http://127.0.0.1:8080/pets/1/toys?nameFilter=Mouse'
{"items":[{"id":2,"petId":1,"name":"Mouse"}]}
```

### wasmtime

```sh
wasmtime serve -S keyvalue zig-out/examples/zig-http-petstore.wasm
```

The repo's root `build.zig` automates the wamr build:

```sh
zig build examples       # build + validate
zig build examples-run   # build + spin-up + smoke (Linux)
```

## Runtime status

| Runtime | Status |
| --- | --- |
| wamr (Linux) | HTTP + keyvalue expected to work (CI serve-smoke). |
| wamr (Windows, AOT) | HTTP works; **keyvalue is not bridged under the Windows AOT host path** (`aot-broken-components` class) — pets read back empty. |
| `wasmtime serve` | **Currently blocked** by a cataggar/wabt encoder discrepancy: wabt flattens the `wasi:http` `error-code` variant with an `i64` (from `HTTP-request-body-size(option<u64>)`) at one `response-outparam.set` result slot, where wasmtime expects `i32`. The component otherwise targets the canonical world wasmtime provides. |

The keyvalue ABI and canonical world are validated structurally (the
component links past keyvalue under wasmtime and runs the HTTP path on
wamr); the blockers above are in the toolchain/host, not this example.

## WIT layout

```
wit/
├── world.wit                  # package example:zig-http-petstore
└── deps/
    ├── wasi-http/             # canonical wasi:http@0.2.6 (handler + types)
    ├── wasi-io/               # canonical wasi:io@0.2.6 (streams, error, poll)
    ├── wasi-clocks/           # canonical wasi:clocks@0.2.6 (monotonic-clock)
    └── wasi-keyvalue/         # trimmed wasi:keyvalue@0.2.0-draft2 (store)
```

The `wasi:http` / `wasi:io` / `wasi:clocks` deps are the **canonical,
unmodified** upstream WIT (vendored from the in-repo `wasi-canon` tree),
so the component's types match exactly what `wasmtime serve` provides.
This differs from `zig-http`, which trims those interfaces down for a
wamr-only, minimal-import build. The keyvalue dep is trimmed to the
`open` + `bucket.{get,set,delete,exists}` subset the handler uses.

World-import order matters: `wasi:io/streams` is imported **before**
`wasi:http/types` because the latter's `use wasi:io/streams.{…}` clause
must see those resources already in the world's type-indexspace. Per
cataggar/wabt, exactly one file per `deps/<pkg>/` directory may carry the
`package` declaration (the alphabetically-first one).

## Source walkthrough

The canonical-ABI bridge lives in the shared guest helper modules under
[`src/guest/`](../../../src/guest/): [`wasi_http`](../../../src/guest/wasi_http.zig)
(`@import("wasi_http")`, the same one `zig-http` uses) and
[`wasi_keyvalue`](../../../src/guest/wasi_keyvalue.zig)
(`@import("wasi_keyvalue")`), both built on the shared
[`abi`](../../../src/guest/abi.zig) module (the single `cabi_realloc`
arena + ret-area). There is no Zig `wit-bindgen` backend, so these
hand-write the host imports + ret-area decoding once, behind typed APIs
(`http.Request` / `http.Responder` / `http.Method`; `kv.Bucket`). This
file is just the petstore routing + JSON + storage logic.

The example reads the request through `http.Request`
(`req.method()`, `req.path()`, `req.readBody()`), persists pets via
`kv.Bucket` (`open` / `get` / `set` / `delete` / `exists`), and replies
through `http.Responder` (`res.respondWithContentType(...)`), which hides
the whole `fields → outgoing-response → outgoing-body → output-stream →
finish → response-outparam.set` sequence. Response JSON is built with
`std.json.Stringify`; the `POST` body is decoded with
`std.json.parseFromSlice`.

Request flow:

1. `req.method()` → route on `GET` / `POST` / `DELETE`.
2. `req.path()` → split path + query, match route.
3. For `POST /pets`: `req.readBody()` pulls the body, `std.json`
   decodes it into a `Pet`, and the pet's canonical JSON is stored at
   `pet:<id>` in the bucket (with the `ids` index + `next_id` updated).
4. `GET` reads back from the bucket; `res.respondWithContentType(...)`
   delivers the response.

The bucket handle is opened once per instance and cached (handles aren't
explicitly dropped — `resource-drop` is a canonical built-in, not a
portable host import).

## Build pipeline

```sh
# 1. Core wasm with the canonical-ABI export + cabi_realloc.
zig build-exe -target wasm32-freestanding -O ReleaseSmall -fno-entry \
    --export="wasi:http/incoming-handler@0.2.6#handle" \
    --export=cabi_realloc \
    -femit-bin=zig-http-petstore.core.wasm src/main.zig

# 2. Embed the WIT (from wit/) and wrap into a component in one step
#    (wabt ≥ v3.0.0-dev.13); a `<name>.core.wasm` input yields `<name>.wasm`.
wabt component new --world petstore --wit wit zig-http-petstore.core.wasm
# -> zig-http-petstore.wasm
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
