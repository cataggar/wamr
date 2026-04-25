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
auto-detected by the binary's version word.

[cm]: https://component-model.bytecodealliance.org/
[p2]: https://github.com/WebAssembly/WASI/blob/main/wasip2/README.md

What works today:

- Loading + instantiating a single-module component.
- Canonical ABI lift/lower for primitive numerics, strings (`list<u8>`),
  and `result<_, _>` returns.
- WASI host adapter for **`wasi:cli/stdout`** + **`wasi:io/streams`**
  (`get-stdout`, `[method]output-stream.blocking-write-and-flush`,
  `[resource-drop]output-stream`). Versioned and unversioned interface
  names are both accepted (e.g. `wasi:cli/stdout@0.2.6`).
- Top-level `run` export → process exit code (0 = ok, 1 = err / trap).

Not yet supported (tracked in [#142]):

- `run` nested inside an exported `wasi:cli/run` instance — real Rust
  components hit this. Needs the indexspace resolver.
- `wasi:cli/stdin`, `wasi:cli/stderr`, `wasi:cli/exit`, args, env.
- `wasi:filesystem`, `wasi:clocks`, `wasi:random`, `wasi:sockets`,
  `wasi:http`.
- Multi-memory components.
- Compound `result<T, E>` payload encoding beyond the empty arms.

[#142]: https://github.com/cataggar/wamr/issues/142

## License

[Apache 2.0](LICENSE)
