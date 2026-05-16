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
$ zig build wasi-testsuite      # WASI Preview 1 + Preview 2 — passing
$ zig build wasi-p3-testsuite   # WASI Preview 3 (wasm32-wasip3) — 40 / 40 passing
$ zig build wasi-p3-parity      # Same fixtures via wamr + wasmtime, diff the reports
```

The upstream `do_wait` timeout is 5s; that matches GitHub Actions runner
timings but is tight on slow developer VMs (e.g. `http-fields` takes
~11s on the project Azure dev VM). Set `WAMR_TESTSUITE_TIMEOUT=<seconds>`
to override it — see
[`tests/wasi-testsuite-runner-patch/`](tests/wasi-testsuite-runner-patch/wasi_test_runner.py)
([#583](https://github.com/cataggar/wamr/issues/583) A7).

The suite drives the freshly-built `wamr` CLI through the in-tree adapter at
[`tests/wasi-testsuite-adapter/wamr-zig.py`](tests/wasi-testsuite-adapter/wamr-zig.py)
and applies the curated skiplists at
[`tests/wasi-testsuite-skip.json`](tests/wasi-testsuite-skip.json) (Preview 1 / 2)
and [`tests/wasi-p3-testsuite-skip.json`](tests/wasi-p3-testsuite-skip.json)
(Preview 3 — currently empty: every `wasm32-wasip3` fixture passes). Every entry
in either skiplist must carry a one-line rationale and a follow-up issue number.
When a previously-skipped test starts passing, delete the entry — the suite is
the gate against regressions in already-shipped WASI host functions.

`zig build wasi-p3-parity` is the cross-runtime gate ([#583
C1](https://github.com/cataggar/wamr/issues/583)): it runs the same
`wasm32-wasip3` corpus through wamr **and** upstream
[Wasmtime](https://wasmtime.dev/) (CI pin `v44.0.1`, the first stable
release with `-Sp3` support) and diffs the JSON reports via
[`scripts/diff-testsuite-reports.py`](scripts/diff-testsuite-reports.py).
The classifier exits non-zero only on *regressions* (wamr fails a
fixture that Wasmtime still passes); deltas in the other direction
(Wasmtime fails a fixture wamr passes) are downgraded to fixture /
runtime-bug warnings so a wamr regression that Wasmtime also exhibits
doesn't masquerade as a wamr bug. The CI workflow at
[`.github/workflows/wasi-p3-parity.yml`](.github/workflows/wasi-p3-parity.yml)
runs the gate on push to `main` and nightly.

[wts]: https://github.com/WebAssembly/wasi-testsuite

## WASI

`wamr` ships the WASI 0.2.x **and** 0.3.0 interface surface (`wasi:cli`,
`wasi:clocks`, `wasi:filesystem`, `wasi:http`, `wasi:io`, `wasi:random`,
`wasi:sockets`); both gates are green (`zig build wasi-testsuite` —
**72 / 72** Preview 1 fixtures; `zig build wasi-p3-testsuite` —
**40 / 40** Preview 3 fixtures). Outbound HTTP and HTTPS issue real
requests via `std.http.Client` and Zig 0.16's `std.crypto.tls`.

See **[`docs/wasi.md`](docs/wasi.md)** for the full feature matrix —
interface → version → method count → fixture pass-rate → known
limitations — and [#583](https://github.com/cataggar/wamr/issues/583)
for post-Preview-3 hardening items.

To exercise the real outbound HTTPS path in unit tests (off by default
so CI stays hermetic):

```console
$ zig build test -Dnetwork_tests=true
```

## License

[Apache 2.0](LICENSE)
