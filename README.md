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

`ReleaseSafe` is the primary source-build recommendation. The 2026-05-29 optimize-mode comparison shows cold-start `noop.cwasm` at ×0.54 Safe/Fast and SIMD interpreter rows at ×0.99 median on the project VM, while also showing `ReleaseSafe` catching a CoreMark AOT compiler invariant before timing; see [`docs/bench/optimize-mode-comparison-2026-05-29.md`](docs/bench/optimize-mode-comparison-2026-05-29.md).

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
The classifier exits non-zero on:

* *Regressions* — wamr fails a fixture that Wasmtime still passes
  (a true wamr regression).
* *Stale skip-list entries* — a fixture listed in
  [`tests/wasi-p3-parity-skip.json`](tests/wasi-p3-parity-skip.json)
  is no longer in the wamr-pass / Wasmtime-fail shape (e.g. the
  upstream Wasmtime / wasi-testsuite fix has landed and the entry
  must be retired).
* *Undocumented Wasmtime-side bugs* under `--strict` — a fixture
  wamr passes but Wasmtime fails for which the skip-list has no
  tracking entry.

The skip-list at
[`tests/wasi-p3-parity-skip.json`](tests/wasi-p3-parity-skip.json)
maps each known Wasmtime-side or fixture-side bug to an upstream
tracking issue. Entries listed there are *documented* deltas — they
are reported on stderr but never fail the parity gate, so a Wasmtime
v44.0.1 fixture failure (currently `http-service`,
`sockets-tcp-connect`, `sockets-tcp-listen`, `sockets-udp-send`)
does not masquerade as a wamr bug. To document a new Wasmtime delta,
file an upstream issue at `bytecodealliance/wasmtime` (or
`WebAssembly/wasi-testsuite` for fixture bugs) and add an entry
keyed by fixture name with the tracking URL as the value. The CI
workflow at
[`.github/workflows/wasi-p3-parity.yml`](.github/workflows/wasi-p3-parity.yml)
runs the gate on push to `main` and nightly and is *required* — a
new wamr regression or new undocumented Wasmtime delta blocks the
merge queue.

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

### wasi:http

The `wasi:http/handler@0.3.0` incoming-handler server (PRs #580 + #595)
accepts plaintext HTTP/1.1 — keep-alive, chunked Transfer-Encoding,
response trailers, and 431 / 413 oversize limits are all in. **HTTPS
termination** ([#609](https://github.com/cataggar/wamr/issues/609)) is
live:

```console
$ wamr run --listen=<addr> --tls-cert=<cert.pem> --tls-key=<key.pem> app.wasm
$ wamr run --listen=<addr> --tls-pem=<combined.pem> app.wasm   # cert + key in one file
```

The cert chain (`std.crypto.Certificate.Bundle`) and PEM-encoded private
key (PKCS#8 / RSA / EC) are parsed + validated at startup, so a missing
file, malformed PEM, or key/cert mismatch surfaces before `bind(2)`.
Each accepted connection is upgraded with a TLS 1.3 server-side handshake
(RSA and ECDSA server certificates) before HTTP/1.1 is served over the
encrypted stream. Server-side TLS comes from the pure-Zig
[`cataggar/tls.zig`](https://github.com/cataggar/tls.zig) dependency —
Zig 0.16's `std.crypto.tls` ships only a client. When `--tls-cert` /
`--tls-pem` is omitted the listener serves plaintext HTTP/1.1.

## Configuration

`wasi:config@0.2.0-rc.1` (`store.get` / `store.get-all`) is wired
through a layered host adapter (#583 B6). Guest components import
`wasi:config/store@0.2.0-rc.1` and read `string → string` pairs from
two sources:

1. **Environment variables** prefixed with `WAMR_CONFIG_`. The
   prefix is stripped and the remainder lower-cased ASCII
   (`WAMR_CONFIG_API_KEY=secret` → `api_key=secret`).
2. **JSON file** passed via `--config-store=PATH`. The file must
   be a flat object of string values (`{"key":"value", …}`); nested
   objects, arrays, or non-string scalars are rejected at startup.

**Precedence: file overrides env.** When the same lower-cased key
appears in both layers, the env entry is dropped and the file value
wins. Surviving env-only entries are appended after the file entries.

```console
$ cat config.json
{ "host": "api.example.com", "timeout_ms": "5000" }

$ WAMR_CONFIG_HOST=localhost WAMR_CONFIG_DEBUG=1 \
    wamr run --config-store=config.json my-component.wasm
# Guest sees: host=api.example.com (file wins), timeout_ms=5000 (file),
#             debug=1 (env-only).
```

The in-memory store never surfaces the `error` arms (`upstream` /
`io`) defined by `wasi:config@0.2.0-rc.1` — every lookup returns
`Ok(Some)` or `Ok(None)`. Those arms are reserved for future Vault /
Kubernetes ConfigMaps / etc. back-ends.

## License

[Apache 2.0](LICENSE)
