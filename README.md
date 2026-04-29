# WAMR: WebAssembly Micro Runtime

A fork of [bytecodealliance/wasm-micro-runtime](https://github.com/bytecodealliance/wasm-micro-runtime) ported from C to Zig and maintained with AI assistance.

**100% spec conformance** — 20,901/20,901 tests passing.

## Install

Install pre-built binaries from [GitHub Releases](https://github.com/cataggar/wamr/releases) with [ghr](https://github.com/cataggar/ghr):

```console
$ ghr install cataggar/wamr
```

See [INSTALL.md](INSTALL.md) for alternative installation methods (uv, pip, dist) and detailed instructions.

## Tools

 - **wamr**: decode and run a WebAssembly binary file using a stack-based interpreter
 - **wamrc**: AOT compiler — compile a .wasm module to native code

## Building

Requires [Zig](https://ziglang.org/) 0.16.x. No other dependencies.

```console
$ git clone https://github.com/cataggar/wamr
$ cd wamr
$ zig build
```

For release builds:

```console
$ zig build -Doptimize=ReleaseSafe
```

Cross-compilation works out of the box:

```console
$ zig build -Dtarget=aarch64-linux -Doptimize=ReleaseSafe
$ zig build -Dtarget=aarch64-macos -Doptimize=ReleaseSafe
$ zig build -Dtarget=x86_64-windows -Doptimize=ReleaseSafe
```

## Running tests

Unit tests:

```console
$ zig build test
```

Spec tests:

```console
$ zig build
$ ./zig-out/bin/spec-test-runner tests/spec-json
```

## Component Model (preview)

Experimental support for the [WebAssembly Component Model][cm] and
[WASI Preview 2][p2]. Run a component with `wamr component.wasm`; it is
auto-detected by the binary's version word. Components that export
`wasi:http/incoming-handler.handle` can be served with
`wamr --listen=127.0.0.1:8080 component.wasm`.

[cm]: https://component-model.bytecodealliance.org/
[p2]: https://github.com/WebAssembly/WASI/blob/main/wasip2/README.md

What works today:

- Loading + instantiating a single-module component.
- Canonical ABI lift/lower for primitive numerics, strings (`list<u8>`),
  and `result<_, _>` returns.
- Top-level `run` export → process exit code (0 = ok, 1 = err / trap),
  including `run` nested inside an exported `wasi:cli/run` instance.
- WASI Preview 2 host adapters covering the full proxy + cli surface.
  Versioned and unversioned interface names are both accepted (e.g.
  `wasi:cli/stdout@0.2.6`):
  - `wasi:cli/{stdin,stdout,stderr,exit,environment,terminal-*}`
  - `wasi:io/{streams,poll,error}`
  - `wasi:clocks/{wall-clock,monotonic-clock}`
  - `wasi:random/{random,insecure,insecure-seed}`
  - `wasi:filesystem/{types,preopens}` — `open-at`, `read-via-stream`,
    `write-via-stream`, `append-via-stream`, `stat`, `get-type`,
    sandboxed path validation, configurable preopens
  - `wasi:sockets/{network,instance-network,tcp,tcp-create-socket,udp,
    udp-create-socket,ip-name-lookup}` — resource binding + handle
    allocation; outbound I/O default-deny pending capability allow-list
  - `wasi:http/{types,outgoing-handler,incoming-handler}` — resource
    tables, outbound requests via `std.http.Client` when network access is
    allowed, and an opt-in incoming HTTP/1.1 server entry point via
    `--listen=<ip:port>`. The incoming server currently handles one request
    per connection, closes after each response, and supports bounded
    `Content-Length` request bodies (no chunked request decoding yet).

Deferred follow-ups (not blocking preview-2 component loading) are
tracked individually: HTTP TLS/chunked/server concurrency, sockets
capability allow-list refinements, DNS, and filesystem timestamp fidelity.

## License

[Apache 2.0](LICENSE)
