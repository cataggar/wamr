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

## License

[Apache 2.0](LICENSE)


<!-- Fix #123 -->
