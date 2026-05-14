# WAMR: WebAssembly Micro Runtime

A fork of [bytecodealliance/wasm-micro-runtime](https://github.com/bytecodealliance/wasm-micro-runtime) ported from C to Zig and maintained with AI assistance. It passes the [WebAssembly/spec](https://github.com/WebAssembly/spec) test suite of 20k+ tests. It supports the [Component Model](https://github.com/webassembly/component-model). It has a very fast cold start, small engine binary size with no dependencies, and is easy to build & fork.

[Wasmtime](https://github.com/bytecodealliance/wasmtime) is currently about 3x faster in CoreMark benchmarks. It has years of production usage and use with a proven track record and security audits.

## Install

Install pre-built binaries from GitHub Releases with [ghr](https://github.com/cataggar/ghr):

```console
$ ghr install cataggar/wamr
```

See [INSTALL.md](INSTALL.md) for alternative installation methods (winget, uv, pip) and detailed instructions.

## Tools

 - **wamrc**: AOT compiler — compile a `.wasm` module to a native `.cwasm` binary (`wamrc compile foo.wasm`)
 - **wamr**: run a WebAssembly module — either a `.wasm` file via the stack-based interpreter, or a precompiled `.cwasm` file produced by `wamrc` (`wamr run foo.wasm`)

## Building

Requires [Zig](https://ziglang.org/) 0.16. No other dependencies.

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

WASI conformance ([WebAssembly/wasi-testsuite][wts]):

```console
$ git submodule update --init tests/wasi-testsuite
$ pip install -r tests/wasi-testsuite/test-runner/requirements.txt
$ zig build wasi-testsuite
```

The suite drives the freshly-built `wamr` CLI through the in-tree adapter at
[`tests/wasi-testsuite-adapter/wamr-zig.py`](tests/wasi-testsuite-adapter/wamr-zig.py)
and applies the curated skiplist at
[`tests/wasi-testsuite-skip.json`](tests/wasi-testsuite-skip.json). Every entry
in the skiplist must carry a one-line rationale and a follow-up issue number.
When a previously-skipped test starts passing, delete the entry — the suite is
the gate against regressions in already-shipped WASI host functions.

[wts]: https://github.com/WebAssembly/wasi-testsuite

## WASI limitations

### `wasi:http` — outbound HTTP and HTTPS

`wamr`'s `wasi:http/outgoing-handler.handle` (Preview 2) and
`wasi:http/client.send` (Preview 3) issue real outbound HTTP and **HTTPS**
requests via `std.http.Client.fetch`. TLS is delegated to Zig 0.16's
[`std.crypto.tls`](https://ziglang.org/documentation/master/std/#std.crypto.tls)
client — system root certificates are loaded via
`std.crypto.Certificate.Bundle.rescan` (see
[#521](https://github.com/cataggar/wamr/issues/521)). Both `http://` and
`https://` schemes are accepted when `sockets_allow_list_template` is
non-empty; the empty allow-list still surfaces
`error-code::HTTP_request_denied`.

Transport, TLS handshake, and parse failures collapse into
`error-code::internal_error` — finer-grained diagnostic mapping is left to
follow-up work. Response headers are not yet surfaced (a `std.http.Client`
limitation: `FetchResult` only exposes the status line) and incoming-handler
server semantics remain a stub.

The earlier `error-code::HTTP_protocol_error` short-circuit on `https://`
(introduced in [#477](https://github.com/cataggar/wamr/issues/477) /
[#501](https://github.com/cataggar/wamr/pull/501)) was removed once we
confirmed Zig 0.16 ships a working TLS client in `std.crypto.tls`.

To exercise the real outbound HTTPS path in unit tests (off by default
so CI stays hermetic):

```console
$ zig build test -Dnetwork_tests=true
```

## License

[Apache 2.0](LICENSE)
