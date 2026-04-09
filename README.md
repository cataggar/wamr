# WAMR: WebAssembly Micro Runtime

A fork of [bytecodealliance/wasm-micro-runtime](https://github.com/bytecodealliance/wasm-micro-runtime) ported from C to Zig and maintained with AI assistance.

**99.9% spec conformance** — 20,878/20,901 tests passing.

## Install

Pre-built binaries are published to [GitHub Releases](https://github.com/cataggar/wamr/releases) and [PyPI](https://pypi.org/project/wamr-bin/). See [installation details](https://github.com/cataggar/wamr/issues/69).

```console
$ dist install cataggar/wamr
```

```console
$ uv tool install wamr-bin
```

## Tools

 - **iwasm**: decode and run a WebAssembly binary file using a stack-based interpreter
 - **wamrc**: AOT compiler — compile a .wasm module to native code

## Building

Requires [Zig](https://ziglang.org/) 0.15.x. No other dependencies.

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
