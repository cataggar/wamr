# WASI feature matrix

Canonical reference for the WASI surface shipped by `wamr` — interface,
supported versions, WIT-method count, fixture pass-rate, and known
limitations. This document is the embedder-facing companion to the
[`README.md`](../README.md) "WASI conformance" section; the README
keeps the build / test invocation block, this file owns the detail.

The full Preview-3 host adapter lives in
[`src/component/wasi_cli_adapter.zig`](../src/component/wasi_cli_adapter.zig);
the `populateWasiProviders` function (≈ line 18823) is the registration
seam where each interface name is version-multiplexed onto the matching
`HostInstance`.

* Adapter source: ~16 k LOC, single Zig file (`wasi_cli_adapter.zig`).
* P1 conformance gate: `zig build wasi-testsuite` — **72 / 72** passing
  (C + Rust + AssemblyScript suites).
* P3 conformance gate: `zig build wasi-p3-testsuite` — **40 / 40** passing
  (`wasm32-wasip3` Rust suite).
* Curated component gate: `zig build wasi-p2-testsuite` — 5 / 5 passing
  (`zig-hello`, `zig-exit`, `zig-calculator-cmd`, `mixed-zig-rust-calc`,
  `zig-http`).

## Table of contents

- [Status overview](#status-overview)
- [Detail table](#detail-table)
- [Preview 1 / 2 / 3 milestones](#preview-1--2--3-milestones)
- [Known limitations](#known-limitations)
- [Build & test](#build--test)
- [Roadmap](#roadmap)
- [See also](#see-also)

## Status overview

| Interface family | 0.2.x | 0.3.0 | One-line status |
| ---------------- | :---: | :---: | --------------- |
| `wasi:cli`         | ✅ | ✅ | Real stdio capture, env / args / exit / terminal stubs. |
| `wasi:clocks`      | ✅ | ✅ | Monotonic + wall / system clock; `wait-for` / `wait-until` host-driven. |
| `wasi:filesystem`  | ✅ | ✅ | Full descriptor + preopens surface; sandboxed by `--preopen`. |
| `wasi:http`        | ✅ | ✅ | Real outbound HTTP/HTTPS via `std.http.Client`; incoming-handler HTTP/1.1 (`Connection: close`). |
| `wasi:io`          | ✅ | ✅ | `poll` / `error` / `streams`; P3 stream/future plumbing lives in the canonical ABI. |
| `wasi:random`      | ✅ | ✅ | OS CSPRNG (`std.crypto.random`) + insecure variants + 128-bit seed. |
| `wasi:sockets`     | ✅ | ✅ | TCP + UDP + DNS; allow-list gated; SO_REUSEADDR; Windows + POSIX parity. |
| `wasi:keyvalue`    | ✅ | — | Memory-store host adapter — `store` + `atomics` + `batch` (#583 B4). |
| `wasi:logging`     | ✅ | — | `wasi:logging@0.1.0-draft`: routes guest log calls to host stderr + `std.log.scoped(.wasi_guest)`. Level filter via `--log-level` / `WAMR_LOG_LEVEL`. |
| `wasi:config`      | ✅ (rc.1) | — | Layered env (`WAMR_CONFIG_*`) + `--config-store=PATH.json` host adapter (#583 B6). |
| `wasi:blobstore`   | — | — | Not implemented (#583 B7). |
| `wasi:threads`     | — | — | Not implemented (#583 B3). |

✅ = shipped + gated by the conformance suite. — = not in scope today.

## Detail table

Method counts are the registered host-side members (each row in the
`members.put(...)` block of the matching `populateWasiXxx` helper). They
correspond 1:1 with the WIT functions / methods / `[constructor]` /
`[static]` / `[resource-drop]` entries.

### WASI Preview 2 (0.2.x)

| Interface                          | Methods | Fixture coverage | Limitations / tracking |
| ---------------------------------- | ------: | ---------------- | ---------------------- |
| `wasi:cli/stdin`                   |  1 | `wasi-testsuite` Rust + C suites | — |
| `wasi:cli/stdout`                  | 12 (run + streams) | `wasi-testsuite` | `populateWasiCliRun` packs `stdout` + `streams` together. |
| `wasi:cli/stderr`                  |  1 | `wasi-testsuite` | — |
| `wasi:cli/exit`                    |  2 | `wasi-testsuite` | `exit`, `exit-with-code`. |
| `wasi:cli/environment`             |  3 | `wasi-testsuite` | `get-environment`, `get-arguments`, `initial-cwd`. |
| `wasi:cli/terminal-{std{in,out,err},input,output}` | 5 | `wasi-testsuite` | Captured-buffer mode returns `none` — no real TTY. |
| `wasi:io/poll`                     |  ┐ | `wasi-testsuite` | — |
| `wasi:io/error`                    |  ┘5 (combined) | `wasi-testsuite` | — |
| `wasi:io/streams`                  | (covered by `cli/stdout`) | `wasi-testsuite` | — |
| `wasi:clocks/wall-clock`           |  2 | `wasi-testsuite` | — |
| `wasi:clocks/monotonic-clock`      |  4 | `wasi-testsuite` | — |
| `wasi:random/random`               |  2 | `wasi-testsuite` | `std.crypto.random`. |
| `wasi:random/insecure`             |  2 | `wasi-testsuite` | — |
| `wasi:random/insecure-seed`        |  1 | `wasi-testsuite` | 128-bit seed. |
| `wasi:filesystem/preopens`         |  1 | `wasi-testsuite` Rust + C | Surfaces `--preopen` mappings. |
| `wasi:filesystem/types`            | 32 | `wasi-testsuite` Rust + C | Full descriptor surface (#475 / #476). |
| `wasi:sockets/network`             |  1 | `wasi-p2-testsuite` | — |
| `wasi:sockets/instance-network`    |  1 | `wasi-p2-testsuite` | — |
| `wasi:sockets/ip-name-lookup`      |  4 | `wasi-p2-testsuite` | DNS via `std.net.getAddressList`. |
| `wasi:sockets/tcp`                 | 30 | `wasi-p2-testsuite` | Real `bind/connect/listen/accept`; getters/setters cover `keep-alive`, `hop-limit`, buffer sizes. |
| `wasi:sockets/tcp-create-socket`   |  1 | `wasi-p2-testsuite` | — |
| `wasi:sockets/udp`                 | 22 | `wasi-p2-testsuite` | Real `bind` + `recvfrom` / `sendto`. |
| `wasi:sockets/udp-create-socket`   |  1 | `wasi-p2-testsuite` | — |
| `wasi:http/types`                  | 56 | `wasi-p2-testsuite` (`zig-http`) | Fields + outgoing/incoming request/response + bodies + futures. |
| `wasi:http/outgoing-handler`       |  1 | `wasi-p2-testsuite` (`zig-http`) | Real `std.http.Client.fetch` for `http://` + `https://`. |
| `wasi:http/incoming-handler`       |  1 | `wasi-p2-testsuite` (`zig-http`) | Real TCP-listener-backed dispatch (#580). |
| `wasi:keyvalue/store@0.2.0-draft2`     |  7 | unit tests (`#583 B4`) | Memory-store `bucket`: `open`, `get`, `set`, `delete`, `exists`, `list-keys`, `[resource-drop]`. |
| `wasi:keyvalue/atomics@0.2.0-draft2`   |  5 | unit tests (`#583 B4`) | `increment` (real); `cas` resource + `swap` registered as `error::other` stubs. |
| `wasi:keyvalue/batch@0.2.0-draft2`     |  3 | unit tests (`#583 B4`) | `get-many`, `set-many`, `delete-many` over the same bucket table. |
| `wasi:logging/logging@0.1.0-draft` |  1 | unit tests | Host stderr + `std.log.scoped(.wasi_guest)`; level filter via `--log-level` / `WAMR_LOG_LEVEL`. No structured-logging backends yet (#583 B5). |
| `wasi:config/store@0.2.0-rc.1`     |  2 | Adapter unit tests (#583 B6) | `get` / `get-all`. Layered backing: env vars matching `WAMR_CONFIG_<KEY>=<value>` (prefix stripped, key lower-cased ASCII) plus an optional `--config-store=PATH.json` flat object. **File overrides env** on duplicate keys. In-memory store never surfaces the `error` arms (`upstream` / `io`) — reserved for future Vault / Kubernetes / etc. backends. Pinned to upstream `wasi:config@0.2.0-rc.1` ([WebAssembly/wasi-config](https://github.com/WebAssembly/wasi-config)); the version-multiplex in `populateWasiProviders` accepts any `wasi:config/store@…` import so future revisions that keep the method shape work without code changes. |

### WASI Preview 3 (0.3.0)

| Interface                                | Methods | Fixture coverage | Limitations / tracking |
| ---------------------------------------- | ------: | ---------------- | ---------------------- |
| `wasi:cli/{stdin,stdout,stderr}@0.3.0`   | 3 × 1 | `wasi-p3-testsuite` | Async stdio over `stream<u8>` (#482 / #548). |
| `wasi:cli/exit@0.3.0`                    |  2 | `wasi-p3-testsuite` | — |
| `wasi:cli/environment@0.3.0`             |  3 | `wasi-p3-testsuite` | — |
| `wasi:cli/types@0.3.0`                   |  0 | `wasi-p3-testsuite` | Type-only instance. |
| `wasi:cli/terminal-*@0.3.0`              |  5 | `wasi-p3-testsuite` | Same captured-buffer stubs as 0.2. |
| `wasi:io/streams@0.3.0`, `error@0.3.0`   |  0 | `wasi-p3-testsuite` | Stub registrations; P3 stream/future engine in the canon ABI (#478 / #505). |
| `wasi:clocks/monotonic-clock@0.3.0`      |  4 | `wasi-p3-testsuite` | `wait-for` / `wait-until` host-driven (#558). |
| `wasi:clocks/system-clock@0.3.0`         |  2 | `wasi-p3-testsuite` | Renamed from `wall-clock` (#483). |
| `wasi:clocks/types@0.3.0`                |  0 | `wasi-p3-testsuite` | Type-only instance (#534). |
| `wasi:random/random@0.3.0`               |  2 | `wasi-p3-testsuite` | — |
| `wasi:random/insecure@0.3.0`             |  2 | `wasi-p3-testsuite` | — |
| `wasi:random/insecure-seed@0.3.0`        |  1 | `wasi-p3-testsuite` | — |
| `wasi:filesystem/preopens@0.3.0`         |  1 | `wasi-p3-testsuite` | — |
| `wasi:filesystem/types@0.3.0`            | 27 | `wasi-p3-testsuite` (`filesystem-*`) | `read-via-stream` / `write-via-stream` host-drivers (#577 / #579). |
| `wasi:sockets/types@0.3.0`               | 43 | `wasi-p3-testsuite` (`sockets-*`) | Unified TCP + UDP resource surface (#486 / #544 / #565). |
| `wasi:sockets/ip-name-lookup@0.3.0`      |  1 | `wasi-p3-testsuite` | — |
| `wasi:http/types@0.3.0`                  | 40 | `wasi-p3-testsuite` (`http-*`) | Unified `request` / `response` resource (#487 / #568). |
| `wasi:http/handler@0.3.0`                |  1 | `wasi-p3-testsuite` (`http-service`) | Incoming-handler trampoline (#549 / #580). |
| `wasi:http/client@0.3.0`                 |  1 | `wasi-p3-testsuite` (`http-request`, `http-fields`) | Outbound; async state machine (#583 A2 / #590). |

Both `wasi-testsuite-skip.json` and `wasi-p3-testsuite-skip.json` are
intentionally **empty** at the time of writing — every vendored fixture
passes. New entries must carry a one-line rationale and a tracking issue.

## Preview 1 / 2 / 3 milestones

Chronology of the WASI rollout in this fork. Each milestone is a PR that
either added a new interface family or closed a tracker issue.

### Preview 1 → Preview 2 (umbrella: [#451](https://github.com/cataggar/wamr/issues/451))

| Milestone                                                                                 | PR    | Notes |
| ----------------------------------------------------------------------------------------- | ----- | ----- |
| `proc_exit → host rc` (preview-1 exit propagation)                                        | [#447](https://github.com/cataggar/wamr/pull/447), [#455](https://github.com/cataggar/wamr/pull/455) | Closes #436. |
| `wasi:filesystem/types` — descriptor file-I/O batch A                                    | [#500](https://github.com/cataggar/wamr/pull/500) | Closes #475. |
| `wasi:filesystem/types` — directory ops batch B                                          | [#503](https://github.com/cataggar/wamr/pull/503) | Closes #476. |
| `wasi-p2-testsuite` conformance gate (curated component fixtures)                         | [#507](https://github.com/cataggar/wamr/pull/507) | Closes #479. |
| HTTPS / TLS — drop `HTTP_protocol_error` short-circuit, use `std.crypto.tls`              | [#526](https://github.com/cataggar/wamr/pull/526) | Closes #521. |

### Preview 3 (umbrella: [#451](https://github.com/cataggar/wamr/issues/451), [#520](https://github.com/cataggar/wamr/issues/520))

| Milestone                                                              | PR    | Notes |
| ---------------------------------------------------------------------- | ----- | ----- |
| `wasi:io@0.3.0` + version-multiplex `populateWasiProviders`           | [#510](https://github.com/cataggar/wamr/pull/510) | Closes #481. P3 foundation. |
| `wasi:random@0.3.0`                                                    | [#512](https://github.com/cataggar/wamr/pull/512) | Closes #485. |
| `wasi:clocks@0.3.0` (`wait-for` / `wait-until` + 0.2 polyfill)        | [#513](https://github.com/cataggar/wamr/pull/513) | Closes #483. |
| `wasi:cli@0.3.0` (async run + `stream<u8>` stdio + exit)              | [#514](https://github.com/cataggar/wamr/pull/514) | Closes #482. |
| `wasi:sockets@0.3.0` (stream/future-based TCP + UDP)                  | [#515](https://github.com/cataggar/wamr/pull/515) | Closes #486. |
| `wasi:filesystem@0.3.0` (stream<u8> read/write + future sync/set-size) | [#516](https://github.com/cataggar/wamr/pull/516) | Closes #484. |
| `wasi:http@0.3.0` (async handlers + stream<u8> bodies + future trailers) | [#517](https://github.com/cataggar/wamr/pull/517) | Closes #487. |
| `wasi-p3-testsuite` conformance gate                                  | [#518](https://github.com/cataggar/wamr/pull/518) | Closes #489. |
| End-to-end load wasm32-wasip3 fixtures via `wamr run`                  | [#532](https://github.com/cataggar/wamr/pull/532) | Closes #520. |
| `wasi:sockets@0.3.0` — real async wiring                              | [#531](https://github.com/cataggar/wamr/pull/531) | Closes #519. |
| `wasi:filesystem@0.3.0` — executor-side async-result lift             | [#528](https://github.com/cataggar/wamr/pull/528) | Closes #522. |
| Complete `wasi:cli@0.3.0` adapter (env / exit / stdio / run-with-err) | [#548](https://github.com/cataggar/wamr/pull/548) | Closes #537. |
| `wasi:http@0.3.0` incoming-handler trampoline                         | [#549](https://github.com/cataggar/wamr/pull/549) | Closes #538. |
| `clocks@0.3.0` canon-lower-of-async-func + `task.cancel`              | [#558](https://github.com/cataggar/wamr/pull/558) | Closes #551. |
| `wasi:http@0.3.0` `http-fields` CI fit + byte-validation              | [#559](https://github.com/cataggar/wamr/pull/559) | Closes #552. |
| `wasi:http@0.3.0` host-handle wire offset                              | [#568](https://github.com/cataggar/wamr/pull/568) | Closes #562. |
| `wasi:sockets@0.3.0` tcp-bind kernel-port + listen reuse              | [#565](https://github.com/cataggar/wamr/pull/565) | Closes #563. |
| `wasi:sockets@0.3.0` tcp-socket property rep                          | [#566](https://github.com/cataggar/wamr/pull/566) | Closes #561. |
| Hybrid 0.2 / 0.3 routing — `wasi:clocks/types@0.3` + type aliases     | [#546](https://github.com/cataggar/wamr/pull/546) | — |
| Generic async-lower trampoline (any `FuncType.is_async`)              | [#567](https://github.com/cataggar/wamr/pull/567) | Closes #564. |
| `wasi:filesystem@0.3.0` fixture completeness (flags, TRUNCATE, BADF)  | [#573](https://github.com/cataggar/wamr/pull/573) | Closes #571 (partial). |
| `wasi:sockets@0.3.0` 0.3 async-future encoding + UDP implicit-bind    | [#574](https://github.com/cataggar/wamr/pull/574) | Closes #569. |
| `wasi:filesystem@0.3.0` `write-via-stream` host driver                | [#577](https://github.com/cataggar/wamr/pull/577) | Closes #571 (residual). |
| `wasi:sockets@0.3.0` `udp-receive` deferred host-driver                | [#578](https://github.com/cataggar/wamr/pull/578) | Closes #576. |
| `wasi:filesystem@0.3.0` `filesystem-stat` fixture detail               | [#579](https://github.com/cataggar/wamr/pull/579) | Closes #571 (residual). |
| `wasi:http@0.3.0` `http-service` end-to-end dispatch                  | [#580](https://github.com/cataggar/wamr/pull/580) | Closes #570. |
| `wasi:sockets@0.3.0` `tcp-bind` SO_REUSEADDR                          | [#581](https://github.com/cataggar/wamr/pull/581) | Closes #575. |
| **Preview 3 gate flip — 40 / 40 `wasm32-wasip3` fixtures pass**       | [#582](https://github.com/cataggar/wamr/pull/582) | Closes #451 / #520. |

### Post-Preview-3 hardening (umbrella: [#583](https://github.com/cataggar/wamr/issues/583))

| Milestone                                                       | PR    | Tracker |
| --------------------------------------------------------------- | ----- | ------- |
| WASI Preview 1 + 2 conformance audit                            | [#584](https://github.com/cataggar/wamr/pull/584) | #583 C2 |
| Configurable `wasi-testsuite` timeout + README refresh          | [#585](https://github.com/cataggar/wamr/pull/585) | #583 A7 + D1 |
| Outbound HTTP / HTTPS finer-grained error mapping               | [#586](https://github.com/cataggar/wamr/pull/586) | #583 A3 |
| `wasi:sockets` Windows `bindAndGetsockname` parity              | [#587](https://github.com/cataggar/wamr/pull/587) | #583 A6 |
| Sockets allow-list consultation at kernel-I/O                   | [#588](https://github.com/cataggar/wamr/pull/588) | #583 A1 |
| Outbound HTTP client async state machine                        | [#590](https://github.com/cataggar/wamr/pull/590) | #583 A2 |
| `wasi:logging@0.1.x` host adapter                               | [#598](https://github.com/cataggar/wamr/pull/598) | #583 B5 |
| `wasi:keyvalue@0.2.0-draft2` memory-store host adapter          | (this PR) | #583 B4 |

## Known limitations

Tracked under the post-Preview-3 umbrella
[#583](https://github.com/cataggar/wamr/issues/583). Each bullet is a
PR-sized child item; cross-references below match the section letters
in that tracker.

### Already-shipped 0.3.0 surfaces (section A)

* **Outbound HTTP response headers not surfaced.** `std.http.Client.FetchResult`
  only exposes the status line; headers are dropped on the way back to the
  guest. Either patch upstream or switch to `std.http.Client.Request`.
  ([#583 A4](https://github.com/cataggar/wamr/issues/583))
* **Incoming-handler HTTP/1.1 robustness.** `Connection: close` +
  `Content-Length` only. Missing: HTTP keep-alive (multiple round-trips
  per accepted fd), chunked transfer-encoding (request and response),
  guest-supplied trailers, `max-header-bytes` / `max-body-bytes` limits,
  graceful `503` on dispatch errors. ([#583 A5](https://github.com/cataggar/wamr/issues/583))

### Roadmap items (section B)

* **Broader `task.cancel` propagation.** The cancel protocol is wired
  for `wasi:clocks/wait-for`; extend to `wasi:sockets/tcp-socket.start-connect`,
  `udp-socket.receive`, `wasi:filesystem` async ops, and
  `wasi:http/handler.handle` host-side. ([#583 B1](https://github.com/cataggar/wamr/issues/583))
* **Caller-supplied-buffer / zero-copy `stream` specialisations.** Today
  every `stream.read` / `stream.write` goes through host scratch buffers.
  Spec allows the lifted call to peek at the guest's destination linmem
  and write directly when alignment + length permit ("borrowed mode" in
  wasmtime). ([#583 B2](https://github.com/cataggar/wamr/issues/583))
* **`wasi:threads@0.3.x`** (preemptive threads) — not implemented;
  upstream WIT still draft. ([#583 B3](https://github.com/cataggar/wamr/issues/583))
* **`wasi:keyvalue@0.2.x`** — memory-store host adapter shipped.
  Limitations: in-process `std.StringHashMapUnmanaged` only; no disk
  persistence and no cross-process / replicated consistency. The
  `cas` resource (`atomics.cas.new` / `cas.current` / `atomics.swap`)
  is registered as `error::other("…")` stubs — guests link cleanly
  but a real CAS round-trip is rejected with a typed error. Disk-
  backed stores and CAS are intentionally out of scope for
  [#583 B4](https://github.com/cataggar/wamr/issues/583); upstream
  WIT pinned at `wasi:keyvalue@0.2.0-draft2`
  ([commit `fb6e23d`](https://github.com/WebAssembly/wasi-keyvalue/tree/fb6e23d11d41d0704b41cdd6362536c5750e0329)
  — vendored under [`docs/wasi-keyvalue-wit-vendored/`](wasi-keyvalue-wit-vendored/)).
* **`wasi:logging@0.1.x`** — host adapter shipped. Routes guest
  `log(level, context, message)` calls to host stderr + Zig's
  `std.log.scoped(.wasi_guest)`. Level filter: `--log-level=<name>` CLI
  flag or `WAMR_LOG_LEVEL` env var. No structured-logging backend
  integration yet — future work.
* **`wasi:config@0.2.x`** — not implemented; would read from env / CLI /
  config-store file. ([#583 B6](https://github.com/cataggar/wamr/issues/583))
* **`wasi:blobstore`, `wasi:cli/run-with-server`** — pending upstream WIT
  pin. ([#583 B7](https://github.com/cataggar/wamr/issues/583))

### Conformance & CI (section C)

* **Wasmtime parity matrix.** The wamr-side P3 gate lands in PR #518;
  the original #489 proposal also called for running the same fixtures
  through Wasmtime in CI and diffing the report so a regression that
  Wasmtime also exhibits is flagged as a fixture bug, not a wamr bug.
  ([#583 C1](https://github.com/cataggar/wamr/issues/583))

## Build & test

The conformance gates exercise the freshly-built `wamr` CLI against
vendored upstream fixtures.

### Submodule + dependencies

```console
$ git submodule update --init tests/wasi-testsuite
$ pip install -r tests/wasi-testsuite/test-runner/requirements.txt
```

### Run the gates

```console
$ zig build wasi-testsuite      # WASI Preview 1 (C + Rust + AssemblyScript) — 72 / 72
$ zig build wasi-p2-testsuite   # Curated component fixtures               —  5 /  5
$ zig build wasi-p3-testsuite   # WASI Preview 3 (wasm32-wasip3)            — 40 / 40
```

### `WAMR_TESTSUITE_TIMEOUT`

The upstream `do_wait` timeout is hard-coded to 5 s in
[`tests/wasi-testsuite/test-runner/wasi_test_runner/test_suite_runner.py:241`](../tests/wasi-testsuite/test-runner/wasi_test_runner/test_suite_runner.py).
That matches GitHub Actions runner timings but is borderline on slow
developer VMs (`http-fields` reaches ~11 s on the project Azure dev VM).
Set `WAMR_TESTSUITE_TIMEOUT=<seconds>` to override — see
[`tests/wasi-testsuite-runner-patch/`](../tests/wasi-testsuite-runner-patch/wasi_test_runner.py)
([#583 A7](https://github.com/cataggar/wamr/issues/583),
PR [#585](https://github.com/cataggar/wamr/pull/585)).

```console
$ WAMR_TESTSUITE_TIMEOUT=30 zig build wasi-testsuite
```

### Outbound HTTPS in unit tests

Off by default so CI stays hermetic:

```console
$ zig build test -Dnetwork_tests=true
```

### Skip-lists

Both skiplists are intentionally **empty** today — every fixture passes
on every supported platform.

| Skip-list                                                      | Suite                                              | Source-of-truth |
| -------------------------------------------------------------- | -------------------------------------------------- | --------------- |
| [`tests/wasi-testsuite-skip.json`](../tests/wasi-testsuite-skip.json)    | `wasi-testsuite` (Preview 1 — C, Rust, AssemblyScript) | filters out fixtures by name; empty ⇒ gate enforces 100 %. |
| [`tests/wasi-p3-testsuite-skip.json`](../tests/wasi-p3-testsuite-skip.json) | `wasi-p3-testsuite` (Preview 3 — Rust)                  | same. |
| [`tests/wasi-p2-testsuite-skip.json`](../tests/wasi-p2-testsuite-skip.json) | `wasi-p2-testsuite` (curated components)               | basename-keyed. |

Re-adding an entry requires a one-line rationale + tracking issue.

## Roadmap

The post-Preview-3 hardening tracker is
[#583](https://github.com/cataggar/wamr/issues/583). Pending interface
additions (one PR per interface, gated behind the existing
`populateWasiProviders` version-multiplex):

* **`wasi:threads@0.3.x`** — [#583 B3](https://github.com/cataggar/wamr/issues/583).
* **`wasi:keyvalue@0.2.x`** — [#583 B4](https://github.com/cataggar/wamr/issues/583).
* ~~**`wasi:logging@0.1.x`**~~ — host adapter shipped (#583 B5); see the
  Preview-2 detail table for the registered methods.
* **`wasi:config@0.2.x`** — [#583 B6](https://github.com/cataggar/wamr/issues/583).
* **`wasi:blobstore`, `wasi:cli/run-with-server`** — [#583 B7](https://github.com/cataggar/wamr/issues/583).

Performance work is tracked separately under
[#393](https://github.com/cataggar/wamr/issues/393); TLS in Zig std and
Linux/macOS-specific async I/O (`io_uring`, `kqueue`) are explicitly out
of scope.

## See also

* [`README.md`](../README.md) — quick-start build / test invocations.
* [`tests/wasi-conformance-audit.md`](../tests/wasi-conformance-audit.md)
  — date-stamped audit log behind the empty `tests/wasi-testsuite-skip.json`
  (Preview 1 + 2; #583 C2 / PR #584).
* [`src/component/wasi_cli_adapter.zig`](../src/component/wasi_cli_adapter.zig)
  — the production host adapter; `populateWasiProviders` is the
  registration seam.
* [`build.zig`](../build.zig) — `wasi-testsuite` / `wasi-p2-testsuite` /
  `wasi-p3-testsuite` build steps (lines ~214–275 and ~713–740).
* [WASI roadmap](https://wasi.dev/roadmap) — upstream interface
  stability tracker.
* [Component Model](https://github.com/webassembly/component-model)
  — the binding spec the adapter targets.
