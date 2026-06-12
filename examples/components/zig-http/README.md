# zig-http

Pure-Zig WebAssembly component mirroring the bytecodealliance
[Rust HTTP-in-components tutorial][bca-http]:

* `GET /`        → `200 "Hello, world!\n"`
* anything else  → `404`

Runs end-to-end through wamr today:

```console
$ ./zig-out/bin/wamr run --listen=127.0.0.1:8080 \
        zig-out/component-examples/zig-http.component.wasm &
$ curl -i http://127.0.0.1:8080/
HTTP/1.1 200 OK
Content-Length: 14
Connection: close

Hello, world!
$ curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/missing
404
```

The repo's root `build.zig` automates everything:

```sh
zig build component-examples       # build + validate
zig build component-examples-run   # build + spin-up + smoke (curl-equivalent in Zig)
```

[bca-http]: https://component-model.bytecodealliance.org/language-support/using-http-in-components/rust.html

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
`wasi:http/proxy` world — it pulls in clocks / random / cli / etc., none
of which our handler uses, and a leaner WIT keeps the example tight.
The Zig source is shorter than the WIT it consumes.

```wit
package example:zig-http;

world http-hello {
    import wasi:io/streams@0.2.6;   // imported BEFORE wasi:http/types
    import wasi:http/types@0.2.6;   // (which has `use wasi:io/streams.{output-stream}`)
    export wasi:http/incoming-handler@0.2.6;
}
```

World-declaration order matters: when one imported interface `use`s
another, the source must be imported BEFORE the consumer (else the
encoder's alias-emit pass references a slot that hasn't been
populated yet). Worth noting because the order is the opposite of
how a reader might intuitively write it ("the entry point first").

## Source walkthrough

The canonical-ABI bridge lives in the shared
[`wasi_http`](../../../src/guest/wasi_http.zig) helper module
(`@import("wasi_http")`), not in this example. There is no Zig
`wit-bindgen` backend, so `wasi_http` hand-writes the host imports
(`extern "wasi:…"` declarations), the ret-area decoding for results
wider than one core value, the `cabi_realloc` scratch arena, and the
`wasi:http/incoming-handler@0.2.6#handle` export — once, behind a small
typed API. See that module's doc comment for the details and for why a
guest still imports only the host functions it actually calls (Zig drops
unreferenced `extern`s, so this minimal handler keeps a minimal WIT
world even though the helper declares the full surface).

The example itself is just the routing logic:

```zig
const http = @import("wasi_http");

comptime { http.exportIncomingHandler(handle); }

fn handle(req: http.Request, res: *http.Responder) void {
    const path = req.path() orelse "/";
    if (std.mem.eql(u8, path, "/")) {
        res.respond(200, "Hello, world!\n");
    } else {
        res.respond(404, "");
    }
}
```

`exportIncomingHandler` takes the handler as a `comptime` function value
and emits the canonical export plus the per-request arena reset, so the
verbose export name never appears in the example. `Responder.respond`
hides the whole `fields → outgoing-response → outgoing-body →
output-stream → finish → response-outparam.set` sequence.

## Build pipeline

```sh
# 1. Core wasm with the canonical-ABI export + cabi_realloc.
zig build-exe -target wasm32-freestanding -O ReleaseSmall \
    -fno-entry \
    --export="wasi:http/incoming-handler@0.2.6#handle" \
    --export=cabi_realloc \
    src/main.zig

# 2. Embed the WIT subset.
wabt component embed --world http-hello wit main.wasm \
    -o main.embed.wasm

# 3. Wrap into a component. No preview1 imports → wabt's plain
#    (non-adapter) component_new wraps it directly; canon.lower
#    trampolines for every imported method are auto-emitted
#    (cataggar/wabt#202/#205/#207), and `cabi_realloc` is used
#    as the canon-options realloc.
wabt component new main.embed.wasm -o zig-http.component.wasm
```

## Implementation notes / gotchas

A handful of constraints fell out of bringing this up; each is
already encoded in the example tree:

* Empty resource bodies use `resource foo {}` — the wabt parser
  rejects the bare `resource foo;` form (`parser.zig:parseResource`
  requires `{`).
* Multi-`*.wit` directory: only the file sorting first
  alphabetically may carry the `package <id>;` declaration. wabt's
  `readWitDir` concatenates files in alpha order, and the parser
  chokes on a second `kw_package` mid-document.
* `world` must explicitly `import` every interface a `use` clause
  references. The encoder's pre-pass populates `alias_requests`
  keyed by source-qname; the main loop only emits matching alias
  decls when the world visits the source as `import` / `export`,
  so without the explicit import you get
  `error: encoding world: InvalidWit`.
* World-import order matters when one imported interface `use`s
  another — see the WIT layout note above.
* `result<T, _>` placeholder in the err arm isn't supported —
  the parser only special-cases `_` in the ok slot
  (`parser.zig:kw_result`). Use `result<T>` (one-arm) instead.
* Resource lifecycle: wamr's `serveOneHttpConnection` calls
  `cleanupHttpResources` after every request, so the handler is
  not required to call `[resource-drop]…` itself — and currently
  doesn't. Real production code on other hosts should drop the
  output-stream before `outgoing-body.finish` per WIT semantics.
