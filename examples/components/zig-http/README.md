# zig-http (WIP — blocked on [cataggar/wabt#191][wabt-191])

Pure-Zig WebAssembly component mirroring the bytecodealliance
[Rust HTTP-in-components tutorial][bca-http]:

* `GET /` → `200 "Hello, world!\n"`
* anything else → `404`

Designed to run end-to-end through wamr via
`wamr run --listen=127.0.0.1:8080 zig-http.component.wasm` and
`curl -i http://127.0.0.1:8080/`.

[bca-http]: https://component-model.bytecodealliance.org/language-support/using-http-in-components/rust.html
[wabt-191]: https://github.com/cataggar/wabt/issues/191

## Current state

| Step                                | Status                                |
|-------------------------------------|---------------------------------------|
| `wabt component embed --world …`    | ✓ encodes (verified)                  |
| `wabt component new`                | ✗ `error: building component: UnsupportedShape` ([wabt#191][wabt-191]) |
| Zig source (canonical-ABI handler)  | placeholder; full impl gated on the above |
| `build.zig` wiring                  | unwired; see plan.md                  |
| `wamr run --listen=…` smoke         | gated                                 |

Two follow-up wabt issues filed during the audit:

* [cataggar/wabt#191][wabt-191] — `metadata_decode` rejects
  encoder-emitted alias decls. This is the gate for
  `wabt component new`.
* [cataggar/wabt#192][wabt-192] — leading doc comment before
  `package` makes the parser eat the `package` keyword. Working
  around in the WIT files by using plain `//` comments at the
  file head.

The WIT layout, world shape, and handler skeleton are in this tree so
that the rest of the example can land in a single follow-up the moment
[wabt#191][wabt-191] ships and `cataggar/wamr` bumps its `build.zig.zon`
wabt pin.

## WIT layout

```
wit/
├── world.wit                              # package example:zig-http
└── deps/
    ├── wasi-http/
    │   ├── incoming-handler.wit           # package wasi:http@0.2.6 (sorts first → carries the package decl)
    │   └── types.wit                      # no package decl (concatenated after)
    └── wasi-io/
        └── streams.wit                    # package wasi:io@0.2.6
```

Each dep is the **minimum subset** of the canonical upstream WIT that
the handler links against. We deliberately don't ship the canonical
`wasi:http/proxy` world (it uses `include wasi:clocks/imports;`, which
the cataggar/wabt encoder doesn't support yet) and don't ship a full
`wasi:http/types` (we'd pull in `wasi:io/poll` for pollables, the
30-case `error-code` variant, futures, outgoing-handler, etc.).

Constraints surfaced by the audit ([cataggar/wabt#191][wabt-191],
[cataggar/wabt#192][wabt-192]) and already encoded in the layout
above:

* Empty resource bodies use `resource foo {}` — the wabt parser
  rejects the bare `resource foo;` form (`parser.zig:parseResource`
  requires `{`).
* No trailing commas in func param lists.
* In a multi-`*.wit` directory, only the file sorting first
  alphabetically may carry the `package` declaration. wabt's
  `readWitDir` concatenates files in alpha order, and the parser
  chokes on a second `kw_package` mid-document.
* `world http-hello` must explicitly `import` every interface a
  used interface references. The encoder's pre-pass populates
  `alias_requests` keyed by source-qname; the main loop only emits
  matching alias decls when the world visits the source as
  `import` / `export`, so without the explicit import you get
  `error: encoding world: InvalidWit`.
* World-declaration order matters when one imported interface
  `use`s another. Imports are emitted (and their `use`-target
  alias slots populated) in textual order, so the *consumer* must
  appear after its *source*. We import `wasi:io/streams` before
  `wasi:http/types` (which has `use wasi:io/streams.{output-stream};`)
  to keep the alias-slot reference forward-only.
* Use `result<T>` (no err arm) rather than `result<T, _>` in
  return-type positions — the parser only special-cases the `_`
  placeholder in the **ok** slot (`parser.zig:kw_result`).
* File-header comments use plain `//`, not `///` — a `///` block
  before `package <id>;` triggers [cataggar/wabt#192][wabt-192].

[wabt-192]: https://github.com/cataggar/wabt/issues/192

## Build pipeline (planned — pseudocode until [wabt#191][wabt-191])

```sh
# 1. core wasm with canonical-ABI export. Freestanding (not -wasi)
#    because reactor-shape adapters have no scratch-memory source.
zig build-exe -target wasm32-freestanding -O ReleaseSmall \
    -fno-entry \
    --export="wasi:http/incoming-handler@0.2.6#handle" \
    src/main.zig

# 2. embed the trimmed WIT.
wabt component embed --world http-hello wit main.wasm \
    -o main.embed.wasm

# 3. wrap into a component. The embed has no `_start`, so wabt's
#    auto-selection picks the reactor-shape preview1 adapter.
wabt component new main.embed.wasm -o zig-http.component.wasm
```

## Run (planned)

```console
$ ./zig-out/bin/wamr run --listen=127.0.0.1:8080 \
        zig-out/component-examples/zig-http.component.wasm &
$ curl -i http://127.0.0.1:8080/
HTTP/1.1 200 OK
content-length: 14

Hello, world!
$ curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/missing
404
```

## What this exercises (once unblocked)

* The first real-world end-to-end exercise of wamr's
  `runHttpComponent` / `serveHttpComponentBytes` path — today only
  one synthetic unit test (`http: discovers versioned
  incoming-handler export (#201)`) drives the export scanner.
* The first real fixture for wabt's **reactor-shape** wasi-preview1
  adapter ([cataggar/wabt#167][wabt-167]); its README explicitly
  notes the deferred validation gap.

[wabt-167]: https://github.com/cataggar/wabt/issues/167

## Notes / open follow-ups

* The handler will need a small (~30-line) hand-written
  `cabi_realloc` bump arena so the canonical-ABI lowering of
  `[method]incoming-request.path-with-query`'s `option<string>`
  return value lands somewhere — wamr's host writes into guest
  memory through the standard realloc protocol.
* Resource lifecycle: wamr's `serveOneHttpConnection` calls
  `cleanupHttpResources` after every request, so the handler is
  not required to call `[resource-drop]…` itself. We still emit a
  defensive drop on `output-stream` because the host registers
  one and the canon stage synthesizes the import.
