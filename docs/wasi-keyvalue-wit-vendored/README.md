# `wasi:keyvalue@0.2.0-draft2` WIT pin

Vendored copy of the WIT for the `wasi:keyvalue@0.2.0-draft2` host
adapter shipped by this repo (`#583 B4`). The interfaces:

* [`store.wit`](store.wit) — `open`, `bucket.{get,set,delete,exists,list-keys}`.
* [`atomic.wit`](atomic.wit) — `atomics.increment`. The `cas` resource
  + `swap` are stubbed out (return `error::other("compare-and-swap not
  implemented in the memory-store adapter")`) so guests that import
  them link cleanly.
* [`batch.wit`](batch.wit) — `batch.{get-many, set-many, delete-many}`.
* [`world.wit`](world.wit) — assembles the three interfaces into the
  `wasi:keyvalue/imports` world the host satisfies.

## Pinned commit

These files mirror the upstream
[`WebAssembly/wasi-keyvalue`](https://github.com/WebAssembly/wasi-keyvalue)
repo at commit
[`fb6e23d11d41d0704b41cdd6362536c5750e0329`](https://github.com/WebAssembly/wasi-keyvalue/tree/fb6e23d11d41d0704b41cdd6362536c5750e0329/wit)
(the tip of `main` as of the PR that landed `#583 B4`). The package
declaration is `wasi:keyvalue@0.2.0-draft2`; the host adapter matches
import names prefixed `wasi:keyvalue/{store,atomics,batch}` at the
0.2.x version band (`matchesWasiPrefix` + `wasiVersion == .p2` /
`.unspecified`).

## Why vendor?

The `tests/wasi-testsuite` submodule does **not** ship any keyvalue
fixtures (verified at commit `40c1f7d`), so the host adapter is
exercised exclusively by the in-tree unit tests at the tail of
`src/component/wasi_cli_adapter.zig`. Vendoring the WIT here keeps the
canonical schema next to the host implementation for review purposes;
the runtime itself doesn't parse these files (host imports are
matched by interface name at the canon-ABI level).

## Limitations

The host implementation is a single-process **memory store** — each
adapter instance owns its own `std.StringHashMapUnmanaged([]const u8)`
per bucket. No disk persistence, no cross-process sharing, and no
upstream upstream / replicated consistency. That's intentional scope
for `#583 B4` — see the PR description; persistence and disk-backed
stores are tracked separately.
