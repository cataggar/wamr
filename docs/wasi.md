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
* P1 conformance gate: `zig build wasi-testsuite` — **70 / 72** passing
  (C + Rust + AssemblyScript suites; the 2 skips are a narrow
  environ-inheritance behavioral mismatch and a flaky upstream fixture
  that compares clock reads without accounting for second rollover,
  not crashes).
* P3 conformance and parity: all 41 `wasm32-wasip3` fixtures pass in
  manifest AOT, no-sidecar JIT, and Wasmtime 46.0.1 after
  [PR #908](https://github.com/cataggar/wamr/pull/908). The unfiltered
  WAMR gates execute the full corpus in both runtime modes.
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
- [wasi:blobstore — deferred (#583 B7)](#wasiblobstore--deferred-583-b7)
- [See also](#see-also)

## Status overview

| Interface family | 0.2.x | 0.3.0 | One-line status |
| ---------------- | :---: | :---: | --------------- |
| `wasi:cli`         | ✅ | ✅ | Real stdio capture, env / args / exit / terminal stubs. |
| `wasi:clocks`      | ✅ | ✅ | Monotonic + wall / system clock; `wait-for` / `wait-until` host-driven. |
| `wasi:filesystem`  | ✅ | ✅ | Full descriptor + preopens surface; sandboxed by `--preopen`. |
| `wasi:http`        | ✅ | ✅ | Real outbound HTTP/HTTPS via `std.http.Client`; incoming-handler HTTP/1.1 (keep-alive / chunked / trailers / limits). HTTPS termination live (`--tls-cert` / `--tls-key` / `--tls-pem`): TLS 1.3 server handshake (RSA + ECDSA certs) via the `cataggar/tls.zig` dependency ([#799](https://github.com/cataggar/wamr/pull/799)). |
| `wasi:io`          | ✅ | ✅ | `poll` / `error` / `streams`; P3 stream/future plumbing lives in the canonical ABI. |
| `wasi:random`      | ✅ | ✅ | OS CSPRNG (`std.crypto.random`) + insecure variants + 128-bit seed. |
| `wasi:sockets`     | ✅ | ✅ | TCP + UDP + DNS; allow-list gated; SO_REUSEADDR; Windows + POSIX parity. |
| `wasi:keyvalue`    | ✅ | — | Memory-store host adapter — `store` + `atomics` (real CAS) + `batch`; optional file-backed persistence via `--keyvalue-store=<path>` (#583 B4). |
| `wasi:logging`     | ✅ | — | `wasi:logging@0.1.0-draft`: routes guest log calls to host stderr + `std.log.scoped(.wasi_guest)`. Level filter via `--log-level` / `WAMR_LOG_LEVEL`. |
| `wasi:config`      | ✅ (rc.1) | — | Layered env (`WAMR_CONFIG_*`) + `--config-store=PATH.json` host adapter (#583 B6). |
| `wasi:blobstore`   | — | — | Not implemented (#583 B7). |
| `wasi:threads`     | — | — | Legacy Preview-1 `wasi.thread-spawn` is production-wired for interpreter-only builds behind `-Dlib_wasi_threads=true -Daot=false`; AOT and Component Model threads remain out of scope ([#616 B1](https://github.com/cataggar/wamr/issues/616)). |

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
| `wasi:keyvalue/store@0.2.0-draft2`     |  7 | unit tests (`#583 B4`) | Memory-store `bucket`: `open`, `get`, `set`, `delete`, `exists`, `list-keys`, `[resource-drop]`. Hydrated from `--keyvalue-store=PATH.json` on startup when set. |
| `wasi:keyvalue/atomics@0.2.0-draft2`   |  5 | unit tests (`#583 B4`) | `increment` + real CAS: `[static]cas.new` snapshots `(bucket, key)`, `[method]cas.current` lifts the snapshot, `swap` is the atomic test-and-set (`cas-error::cas-failed(cas)` on mismatch, re-snapshotted handle for retry). |
| `wasi:keyvalue/batch@0.2.0-draft2`     |  3 | unit tests (`#583 B4`) | `get-many`, `set-many`, `delete-many` over the same bucket table; mutations flush through to `--keyvalue-store` when set. |
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
| `wasi:sockets/types@0.3.0`               | 43 | `wasi-p3-testsuite` (`sockets-*`) | Unified TCP + UDP resource surface (#486 / #544 / #565); callback-driven async `run` resumption (#905). |
| `wasi:sockets/ip-name-lookup@0.3.0`      |  1 | `wasi-p3-testsuite` | — |
| `wasi:http/types@0.3.0`                  | 40 | `wasi-p3-testsuite` (`http-*`) | Unified `request` / `response` resource (#487 / #568). |
| `wasi:http/handler@0.3.0`                |  1 | `wasi-p3-testsuite` (`http-service`) | Incoming-handler trampoline (#549 / #580); HTTP/1.1 keep-alive + chunked + trailers + 431/413 limits (#595); HTTPS termination live — TLS 1.3 server handshake (RSA + ECDSA certs) via the `cataggar/tls.zig` dependency (#799). |
| `wasi:http/client@0.3.0`                 |  1 | `wasi-p3-testsuite` (`http-request`, `http-fields`) | Outbound; async state machine (#583 A2 / #590). |

`wasi-testsuite-expectations.toml` carries the narrow Preview-1 exceptions.
`wasi-p3-testsuite-expectations.toml` currently carries no exclusions.
The unfiltered P3 contract asserts that all 41 fixtures execute and pass, so
either expectation file cannot make the P3 result appear green by excluding
its corpus.

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
| Preview 3 gate — 41 / 41 pass, including callback-driven `sockets-echo` | [#582](https://github.com/cataggar/wamr/pull/582), [#881](https://github.com/cataggar/wamr/issues/881), [#905](https://github.com/cataggar/wamr/issues/905) | #881 restores shared AOT/JIT dispatch; #905 adds callback resumption. |

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
| Incoming-handler HTTPS termination                              | [#799](https://github.com/cataggar/wamr/pull/799) | Closes #609. |

### Hardening cycle 2 (umbrella: [#616](https://github.com/cataggar/wamr/issues/616))

| Milestone | PR | Tracker status |
| --------- | -- | -------------- |
| P3 AOT/JIT/Wasmtime parity, 41 / 41 in each mode | [#908](https://github.com/cataggar/wamr/pull/908) | Parity matrix complete. |
| Initial DNS/TCP connect deadlines; Linux `splice(2)`; calibrated blocking microbenchmark job; stable host-import audit | [#909](https://github.com/cataggar/wamr/pull/909) | Completes A2, A5, and C1. Only the initial DNS/TCP portion of A1 is complete; TLS handshakes, redirect reconnects, and first-/between-byte deadlines remain open. |
| Per-hop redirect connect deadlines, `first-byte-timeout`, `between-bytes-timeout` | [#616 A1](https://github.com/cataggar/wamr/issues/616) | Completes A1. All three `request-options` budgets are enforced. |

## Known limitations

Tracked under the post-Preview-3 umbrella
[#583](https://github.com/cataggar/wamr/issues/583). Each bullet is a
PR-sized child item; cross-references below match the section letters
in that tracker.

The current follow-up tracker is
[#616](https://github.com/cataggar/wamr/issues/616). PR #909 completed
A2, A5, and C1.

#### Outbound HTTP `request-options` deadlines (A1)

All three `wasi:http/types.request-options` budgets are enforced on the
outbound path, for both Preview 2 and Preview 3:

* **`connect-timeout`** bounds acquiring one hop's connection: DNS, TCP,
  and — for `https` — the TLS handshake. The handshake is included
  because `std.http.Client.connectTcpOptions` builds the TLS connection
  through `std.crypto.tls.Client.init`, which completes the wire
  handshake inline rather than deferring it to the first request write.
* **`first-byte-timeout`** bounds request transmission plus response-head
  arrival, armed freshly for *each* redirect hop.
* **`between-bytes-timeout`** bounds each response body chunk. A fresh
  timer is armed per chunk, so arriving bytes re-arm the window: the
  budget measures time without progress, not a whole-body allowance.

Redirects are followed by the adapter rather than by
`std.http.Client.Request.receiveHead`, because the stdlib path reconnects
through the untimed `Client.connect` and puts an entire redirect chain
under a single response budget. The guest-visible policy is unchanged:
payload-free requests follow up to three hops and then fail with
`loop-detected`; requests carrying a payload are never redirected and
receive the 3xx response verbatim.

Expiry of a response-phase budget is reported as `HTTP-response-timeout`.
Only `connect-timeout` expiry maps to `connection-timeout` — a slow
response arrives on a connection that was established successfully, so
attributing it to the connection would be misleading.

An unset budget runs its operation inline, so guests that configure no
`request-options` pay nothing for this enforcement.

#### Fine-grained outbound HTTP cancellation (A7)

Guest cancellation interrupts an outbound request **while it is
blocked**, rather than being noticed only when the current operation
happens to return.

Previously the per-fetch cancel flag was advisory: it was polled at
phase boundaries (before connect, before send, before the response
head, before the body, and every 64 KiB of body). A worker already
parked in `getaddrinfo`, `connect`, `read` or `write` therefore kept
running to completion — for an unreachable host, that is the full OS
SYN-retry period — holding a thread, a socket and a TLS session for a
result no one would read.

Every blocking step on the outbound path now races the operation
against a cancel lane inside `std.Io.Select`. Because the outbound path
runs on Zig's threaded I/O backend, the losing lane is cancelled for
real: the in-flight syscall is interrupted. The lane is armed over:

* connection acquisition (DNS, TCP, and the inline TLS handshake),
* request transmission — head, body and flush; the body is written in
  64 KiB slices so cancellation is observed between chunks rather than
  only after the whole payload,
* response-head arrival, and
* each response body chunk.

The lane itself is a bounded-backoff sleep (1 ms, doubling to 25 ms)
rather than a condition-variable wait, because it must be cancellable
*through* `std.Io` — otherwise a request that completes normally would
leave its cancel lane parked forever.

Two things raise the flag:

* **`task.cancel`**, via the existing `async_cancel_driver` hook, and
* **dropping the readable end of the response future**, via the
  `async_future_drop_driver` hook added for this work. A Preview 3
  guest that abandons a response issues no `task.cancel` at all — it
  simply drops the future — so without this the abandoned request ran
  to completion. The Preview 2 path already flipped the same flag from
  its own future-drop handling.

Cancellation is reported to the guest as `HTTP-request-denied`,
matching the disposition the drainer already used for a cancel
observed after the worker finished.

Requests with neither a `request-options` budget nor a cancel flag
still run inline, so the zero-configuration path is unchanged.

#### Client-disconnect handling on the serve path (A8)

`wamr serve` now distinguishes a client hanging up from a server-side
I/O fault, and reacts to it.

Previously the TCP and TLS output sinks folded *every* write failure
into a generic I/O error, so a disconnect mid-response was
indistinguishable from a real fault. The serve loops swallowed the
error outright, then looped back to read another request that would
never arrive, and the guest was never told its response had not
landed.

Now:

* `ECONNRESET` and `EPIPE`/`ENOTCONN` from the socket write — and the
  equivalent conditions from the TLS record layer — surface as a
  *closed* stream rather than an error, which the response writers
  translate into a distinct disconnect condition.
* Response bodies are written to the socket in 64 KiB slices, so a
  disconnect is detected within one chunk instead of after the entire
  transfer has been handed to the socket layer. This also bounds how
  much of a single guest response the host can queue at once. The
  bytes on the wire and their framing are unchanged; only the number
  of `write` calls differs.
* On disconnect the connection is torn down instead of being kept
  alive, and any host operations the guest still owns on that
  instance — an outbound fetch it never awaited, a timer, a socket
  read — are cancelled, since their results are no longer observable.
* For Preview 3, the response's transmission future is closed without
  a ready payload, which is how the guest learns its response was not
  delivered. The inbound path previously never wrote to that future at
  all, so a response abandoned mid-flight looked exactly like one
  delivered in full.

The remaining live-streaming half is also implemented. Preview 2
`outgoing-body` writes and Preview 3 `stream<u8>` writes now feed a
per-request 64 KiB ring while a dedicated network writer drains it to the
client. The guest producer blocks when the ring is full, so response memory
does not grow with total body size and a slow client provides real
backpressure. Preview 3 stream FIFOs created during an inbound handler use
the same high-water mark even before `response.new` associates the stream
with the response; a parked producer is woken once association moves the
buffer into the live ring.

Status and immutable response headers are snapshotted before the first wire
byte. Explicit `Content-Length` is preserved and verified against the bytes
produced; unknown-length bodies use HTTP/1.1 chunked framing, with the
terminating chunk and trailers emitted only after the body closes. Empty
bodies retain `Content-Length: 0`.

The network writer never accesses component or adapter resource tables.
Guest-side setup takes stable leases only long enough to snapshot handles
and metadata, then releases them before queue waits, socket writes, callback
execution, cancellation, or thread joins. Disconnect, socket failure,
handler failure, shutdown, and normal completion race through one atomic
terminal state. That state wakes blocked producers/writers, interrupts the
accepted socket when cancellation must break a write, settles the Preview 3
transmission future exactly once, and cancels remaining host async
operations.

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
* **Caller-supplied-buffer / zero-copy `stream` specialisations.**
  Done for the hot-spot rendezvous paths:
  * `stream.read` (TCP receive) — opt-in `host_driver.on_read_into`
    callback skips the `stream.buffer` FIFO scratch alloc + the
    second `@memcpy` into guest linmem. 60–73% wall-clock saved at
    16/64 KiB chunks (`zig build wasi-streams-bench`).
  * `stream.write` (TCP send + FS write-via-stream) — opt-in
    `host_driver.on_write_from` callback with a thinner signature
    (drops the unused `*AsyncStream` / `Allocator` params). The
    write path was already memcpy-free via
    `comp_inst.readGuestBytes` — drivers operate on a borrowed
    guest linmem slice — so the win here is API hygiene, not
    memcpy elimination; the microbench confirms parity (±0.3%).
  Both paths validate `ptr + len ≤ memory.size` synchronously
  before the host call, and the executor never yields to a
  `memory.grow` between the bounds check and the driver return.
  ([#583 B2](https://github.com/cataggar/wamr/issues/583))
* **Guest threads** — Preview-1 interpreter spawning, shared resource
  lifetimes, atomics/wait-notify/fence, per-thread context isolation, and
  end-to-end gates are implemented behind `-Dlib_wasi_threads=true
  -Daot=false`. AOT spawning and final cross-thread cancellation remain open.
  ([#616 B1](https://github.com/cataggar/wamr/issues/616))
* **`wasi:keyvalue@0.2.x`** — memory-store host adapter shipped, with
  optional file-backed persistence and real compare-and-swap.
  Limitations: in-process `std.StringHashMapUnmanaged` only; no
  cross-process / replicated consistency beyond the file-backed
  snapshot. Persistence is opt-in via `--keyvalue-store=<path>` —
  the file holds a JSON object whose top-level keys are bucket
  identifiers and whose inner objects map keys to base64-encoded
  value bytes (e.g. `{"primary":{"alpha":"dmFsdWU="}}`). Reads on
  startup, rewrites synchronously on every mutation (`set` /
  `delete` / successful `swap` / `increment` / batch.{set,delete}-many).
  The flush is a plain `writeFile`; the metadata-preserving,
  crash-safe replacement protocol is specified in
  [`docs/design/metadata-safe-json-persistence.md`](design/metadata-safe-json-persistence.md)
  for follow-up implementation. The `cas`
  resource is now a real atomic test-and-set (`[static]cas.new` /
  `[method]cas.current` / `swap` / `[resource-drop]cas`); mismatches
  surface `cas-error::cas-failed(cas)` with a re-snapshotted handle
  so guests can retry without re-issuing `cas.new`. Upstream WIT
  pinned at `wasi:keyvalue@0.2.0-draft2`
  ([commit `fb6e23d`](https://github.com/WebAssembly/wasi-keyvalue/tree/fb6e23d11d41d0704b41cdd6362536c5750e0329)
  — vendored under [`docs/wasi-keyvalue-wit-vendored/`](wasi-keyvalue-wit-vendored/)).
* **`wasi:logging@0.1.x`** — host adapter shipped. Routes guest
  `log(level, context, message)` calls to host stderr + Zig's
  `std.log.scoped(.wasi_guest)`. Level filter: `--log-level=<name>` CLI
  flag or `WAMR_LOG_LEVEL` env var. No structured-logging backend
  integration yet — future work.
* **`wasi:config@0.2.x`** — not implemented; would read from env / CLI /
  config-store file. ([#583 B6](https://github.com/cataggar/wamr/issues/583))
* **`wasi:blobstore`, `wasi:cli/run-with-server`** — deferred. The
  upstream WIT is still WASI Phase 1 (Feasibility) with active rename
  proposals and no wasmtime baseline; `run-with-server` has no upstream
  WIT at all. See [`wasi:blobstore — deferred`](#wasiblobstore--deferred-583-b7)
  below for the full status check, concrete blockers, and acceptance
  criteria. ([#583 B7](https://github.com/cataggar/wamr/issues/583))

### Conformance & CI (section C)

* **Wasmtime parity matrix.** Completed by
  [PR #908](https://github.com/cataggar/wamr/pull/908): CI runs the same
  41-fixture Preview 3 corpus through manifest AOT, no-sidecar JIT, and
  Wasmtime 46.0.1. All three modes pass 41 / 41 with zero parity deltas.
  ([#583 C1](https://github.com/cataggar/wamr/issues/583))
* **Stable host-import audit.** Refreshed by
  [PR #909](https://github.com/cataggar/wamr/pull/909): 351 / 352
  methods are implemented (99.7 %), with no missing stable imports.
  The only stub is P2
  `wasi:filesystem/types.filesystem-error-code`, which ignores its
  borrowed error and returns canned `option::none`.

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
$ zig build wasi-testsuite                            # WASI Preview 1 — 70 / 72
$ zig build wasi-p2-testsuite                         # Curated components — 5 / 5
$ zig build wasi-p3-testsuite                         # Preview 3 AOT — 41 / 41
$ zig build wasi-p3-testsuite-jit -Djit=true          # Preview 3 JIT — 41 / 41
$ zig build wasi-p3-testsuite-wasmtime                # Wasmtime 46.0.1 — 41 / 41
$ zig build test-wasi-threads -Daot=false             # Feature-disabled ABI
$ zig build test-wasi-threads -Dlib_wasi_threads=true -Daot=false
                                                       # Preview-1 interpreter threads
```

### `WAMR_TESTSUITE_TIMEOUT`

The upstream runner obtains its wait timeout from the runtime adapter,
defaulting to 5 s. That matches GitHub Actions runner timings but is borderline on slow
developer VMs (`http-fields` reaches ~11 s on the project Azure dev VM).
Set `WAMR_TESTSUITE_TIMEOUT=<seconds>` to override — see
[`tests/wasi-testsuite-runner-patch/`](../tests/wasi-testsuite-runner-patch/wasi_test_runner.py)
([#583 A7](https://github.com/cataggar/wamr/issues/583),
PR [#585](https://github.com/cataggar/wamr/pull/585)).

```console
$ WAMR_TESTSUITE_TIMEOUT=30 zig build wasi-testsuite
```

### `WAMR_COMPILE_TIMEOUT`

`WAMR_TESTSUITE_TIMEOUT` bounds only the guest process wait. Adapter-side
`wamrc` precompilation runs *before* that process exists, so a codegen hang
there is otherwise unbounded — it stalls until the CI job timeout cancels the
job, which skips the `if: always()` artifact upload and destroys the
diagnostics along with it. `WAMR_COMPILE_TIMEOUT=<seconds>` bounds a single
`wamrc` invocation and defaults to 600 s
([#616](https://github.com/cataggar/wamr/issues/616) D3).

```console
$ WAMR_COMPILE_TIMEOUT=120 zig build wasi-p3-testsuite
```

### `WAMR_TESTSUITE_OP_TIMEOUT`

`WAMR_TESTSUITE_TIMEOUT` bounds only `do_wait`'s `Popen.wait`. Fixtures that
declare an `operations` list (`sockets-echo`, `cli-stdio-roundtrip`, …) also
perform blocking work on the *harness* side, none of which upstream bounds:
`do_connect` does an unbounded `readline()` on the guest's stdout to discover
`host:port`, and `do_read` / `do_send` / `do_recv` block on a pipe or socket
with no timeout.

A guest that starts but never produces its expected output therefore parks the
harness forever, so **one wedged fixture hangs the entire suite** instead of
failing a single test. That is exactly how five AArch64 AOT fixtures consumed
900 s profiling samples while reporting nothing at all
([#929](https://github.com/cataggar/wamr/issues/929)).

Each blocking operation now runs under a watchdog that kills the guest process
when the deadline expires. Killing the guest is what actually unblocks the
harness: its stdout hits EOF and its sockets are closed by the kernel, so
upstream's existing error paths turn the stall into an ordinary failure with a
precise message. `WAMR_TESTSUITE_OP_TIMEOUT=<seconds>` sets the budget;
unset, it inherits `WAMR_TESTSUITE_TIMEOUT`, falling back to 60 s.

```console
$ WAMR_TESTSUITE_OP_TIMEOUT=30 zig build wasi-p3-testsuite
```

### Outbound HTTPS in unit tests

Off by default so CI stays hermetic:

```console
$ zig build test -Dnetwork_tests=true
```

### Expectations

The filtered convenience gates carry three narrow exceptions; the
unfiltered P3 gates still execute every Preview 3 fixture.

| Expectations / skip-list | Suite | Source-of-truth |
| --- | --- | --- |
| [`tests/wasi-testsuite-expectations.toml`](../tests/wasi-testsuite-expectations.toml) | `wasi-testsuite` (Preview 1 — C, Rust, AssemblyScript) | Upstream TOML expectation format. |
| [`tests/wasi-p3-testsuite-expectations.toml`](../tests/wasi-p3-testsuite-expectations.toml) | `wasi-p3-testsuite` (Preview 3 — Rust) | Upstream TOML expectation format. |
| [`tests/wasi-p2-testsuite-skip.json`](../tests/wasi-p2-testsuite-skip.json) | `wasi-p2-testsuite` (curated components) | Basename-keyed local format. |

Re-adding an entry requires a one-line rationale + tracking issue.

### Updating the `wasi-microbench` budget

`zig build wasi-microbench` is a host-path regression detector
(parallel to the CoreMark + cold-start benches) that drives
`executor.dispatchCanonBuiltin` → `AsyncStream` → host_driver across
four hot Preview-3 paths — HTTP keep-alive RT, UDP receive, fs
read/write-via-stream — and fails when the median sample exceeds the
per-scenario budget in
[`tests/benchmarks/wasi-microbench/budget.json`](../tests/benchmarks/wasi-microbench/budget.json)
by more than the regression threshold (default +10 %). CI lives at
[`.github/workflows/wasi-microbench.yml`](../.github/workflows/wasi-microbench.yml)
and the benchmark is blocking within that job whenever the
path-filtered workflow runs. Its report artifact is uploaded even when
the benchmark fails. Branch protection does not currently require this
workflow context, so the failure does not by itself prevent every PR
from merging.

The current baseline was calibrated on 2026-07-15 from all 98 retained
successful `ubuntu-22.04` x86_64 workflow artifact reports in the latest
100 runs (the other two were cancelled), spanning 2026-06-06 through
2026-07-14. Every report used 10 samples per scenario. Each run's
`median_ns` was treated as one observation; the arithmetic mean and
sample standard deviation (σ) of those 98 run medians were:

| Scenario | Mean | σ | Effective gate | Observed pass rate |
| --- | ---: | ---: | ---: | ---: |
| HTTP keep-alive 100 RT | 28.653 µs | 4.132 µs | 38.5 µs | 98/98 |
| UDP receive 1 MiB | 28.163 µs | 6.034 µs | 40.7 µs | 95/98 |
| fs write-via-stream 1 MiB | 170.439 µs | 14.788 µs | 200.2 µs | 98/98 |
| fs read-via-stream 1 MiB | 23.612 µs | 2.371 µs | 28.6 µs | 98/98 |

The target effective gate is approximately `mean + 2σ`. Because the
harness already fails at `median_ns_budget × 1.10`, `budget.json`
stores the target gate divided by 1.10, rounded up to whole
microseconds; the threshold is the noise margin, not a second margin.
The HTTP base was kept one additional microsecond above the direct
rounding (2.38σ effective headroom) so the observed whole-workflow pass
rate is 95/98 (96.9 %), meeting the ≥95 % gate target.

To recalibrate after an intentional performance change, collect a
representative set of successful GitHub-hosted artifact reports. For a
local diagnostic run:

```console
$ zig build wasi-microbench -- --no-budget --samples 20 --warmup 5
```

Compute the distribution above from per-run medians, choose an
effective gate near `mean + 2σ` with an observed whole-workflow pass
rate of at least 95 %, then divide by the workflow's threshold
multiplier before updating `budget.json`. Record the sample, dates,
statistics, threshold, and any additional rounding so future changes
can distinguish intentional regressions from platform drift.

The bench is intentionally synthetic: it stubs out real sockets /
`pwrite(2)` to isolate the canonical-ABI lowering + host_driver
dispatch cost (i.e. the WAMR-side surface), so kernel jitter doesn't
swamp the signal. Real-network coverage stays under the conformance
gates (`wasi-testsuite`, `wasi-p3-testsuite`).

## Roadmap

The post-Preview-3 hardening tracker is
[#583](https://github.com/cataggar/wamr/issues/583). Pending interface
additions (one PR per interface, gated behind the existing
`populateWasiProviders` version-multiplex):

* **Guest threads** — [#616 B1](https://github.com/cataggar/wamr/issues/616).
  Design doc: [`docs/design/wasi-threads.md`](design/wasi-threads.md)
  (prototype inventory, multi-threaded interpreter state isolation,
  and production-gating plan).
* **`wasi:keyvalue@0.2.x`** — [#583 B4](https://github.com/cataggar/wamr/issues/583).
* ~~**`wasi:logging@0.1.x`**~~ — host adapter shipped (#583 B5); see the
  Preview-2 detail table for the registered methods.
* **`wasi:config@0.2.x`** — [#583 B6](https://github.com/cataggar/wamr/issues/583).
* **`wasi:blobstore`, `wasi:cli/run-with-server`** — deferred. See
  [`wasi:blobstore — deferred (#583 B7)`](#wasiblobstore--deferred-583-b7)
  for the upstream status check and acceptance criteria
  ([#583 B7](https://github.com/cataggar/wamr/issues/583)).

Performance work is tracked separately under
[#393](https://github.com/cataggar/wamr/issues/393); TLS in Zig std and
Linux/macOS-specific async I/O (`io_uring`, `kqueue`) are explicitly out
of scope.

## wasi:blobstore — deferred (#583 B7)

[`#583 B7`](https://github.com/cataggar/wamr/issues/583) groups two
interfaces whose host adapters are explicitly _not_ being landed in this
wave: `wasi:blobstore` and the speculative `wasi:cli/run-with-server`
world. Status as of wave 10 (Nov 2025):

### Upstream snapshot

| Repo / interface | Latest pin | Wasmtime crate | Fixtures | Phase |
| ---------------- | ---------- | -------------- | -------- | ----- |
| [`WebAssembly/wasi-blobstore`](https://github.com/WebAssembly/wasi-blobstore) | `v0.2.0-draft` ([commit `005c425`](https://github.com/WebAssembly/wasi-blobstore/tree/v0.2.0-draft), Feb 2024 — re-tagged of the same commit Feb 2024; only chore commits since) | _none_ | _none_ | Phase 1 (Feasibility) |
| `wasi:cli/run-with-server` (per [#583](https://github.com/cataggar/wamr/issues/583)) | _does not exist upstream_ — no interface in [`WebAssembly/wasi-cli`](https://github.com/WebAssembly/wasi-cli) (`wit/` or `wit-0.3.0-draft/`), no tracking issue, no draft PR | n/a | n/a | n/a |

The `wasi-blobstore` interface surface (summarised here rather than
vendored under `docs/wasi-blobstore-wit-vendored/`, because we are
intentionally not shipping host code yet) covers:

* `wasi:blobstore/blobstore` — `create-container`, `get-container`,
  `delete-container`, `container-exists`, `copy-object`, `move-object`.
* `wasi:blobstore/container` — `container` and `stream-object-names`
  resources (`name`, `info`, `get-data`, `write-data`, `list-objects`,
  `delete-object`, `delete-objects`, `has-object`, `object-info`,
  `clear`; plus `read-stream-object-names` / `skip-stream-object-names`).
* `wasi:blobstore/types` — `container-metadata`, `object-metadata`,
  `object-id`, plus `outgoing-value` / `incoming-value` resources
  bridging to `wasi:io/streams@0.2.1`.

### Concrete blockers

1. **No wasmtime baseline.** `bytecodealliance/wasmtime` has zero
   references to `blobstore` in its source tree — there is no
   `crates/wasi-blobstore`, no host trait, and no commit history
   suggesting one is imminent. Without a reference host we cannot align
   `populateWasiProviders` registration, behavioural semantics
   (`copy-object` cross-container, `delete-objects` partial-failure,
   `get-data` start/end inclusivity past end-of-blob), or canonicalised
   `error` strings.
2. **No conformance fixtures.** The proposal repo ships an empty
   `test/` directory, and `WebAssembly/wasi-testsuite` contains zero
   blobstore fixtures. We would have to author the entire conformance
   suite ourselves, which both burns engineering budget and risks
   shipping semantics that diverge from whatever the upstream test
   suite eventually mandates.
3. **Active design churn — likely breaking changes pending.** Open PRs
   and issues on the proposal repo (as of Nov 2025):
   * [PR #26](https://github.com/WebAssembly/wasi-blobstore/pull/26) —
     rename `wasi:blobstore/blobstore` → `wasi:blob/store` (package
     rename; would force a re-vendor + every adapter import path
     change).
   * [PR #23](https://github.com/WebAssembly/wasi-blobstore/pull/23) —
     "Reorganizes the blobstore interface" (still open since 2024).
   * [Issue #32](https://github.com/WebAssembly/wasi-blobstore/issues/32) —
     rename `container` → `bucket` (resource name).
   * [Issue #33](https://github.com/WebAssembly/wasi-blobstore/issues/33) —
     consistency-guarantee semantics still undefined.
   * [Issue #30](https://github.com/WebAssembly/wasi-blobstore/issues/30) —
     `container.delete-objects` semantics undefined.
   * [Issue #29](https://github.com/WebAssembly/wasi-blobstore/issues/29) —
     `stream-object-names.skip` required-behaviour gap.
   * [Issue #27](https://github.com/WebAssembly/wasi-blobstore/issues/27) —
     intended multi-store usage pattern (no concept of store handle).
   * [Issue #7](https://github.com/WebAssembly/wasi-blobstore/issues/7) —
     `timestamp` should be a richer type than `u64`.
   Any one of these landing would invalidate a shipped wamr adapter.
4. **No Preview-3 async story.** The current WIT targets
   `wasi:io/streams@0.2.1` synchronous streams. The proposal repo has
   no `wit-0.3.0-draft/` tree and PR #38 ("Make selected methods
   async") was closed unmerged. Anything we ship today would have to be
   re-plumbed onto the P3 `stream<u8>` / `future<...>` lifted forms
   when the proposal eventually adopts them, mirroring the
   `wasi:io@0.2 → @0.3` rework that just landed across our other
   adapters.
5. **Phase 4 advancement criteria explicitly unmet.** The proposal's
   own README lists, as preconditions for Phase 4: "At least two
   independent production implementations", "At least two cloud
   provider implementations", a cross-platform test suite. None of
   these exist yet.
6. **`wasi:cli/run-with-server` does not exist upstream.** The name
   appears in our [#583](https://github.com/cataggar/wamr/issues/583)
   tracker as a placeholder for a hypothetical world that fuses
   `wasi:cli/run` with `wasi:http/incoming-handler` for long-lived
   server components, but no such world is drafted in
   [`WebAssembly/wasi-cli`](https://github.com/WebAssembly/wasi-cli),
   including its `wit-0.3.0-draft/` tree at the
   [`v0.3.0-rc-2025-09-16`](https://github.com/WebAssembly/wasi-cli/tree/v0.3.0-rc-2025-09-16)
   tag, and there is no tracking issue. There is nothing to implement
   against until the upstream world is drafted.

### Acceptance criteria for wamr to implement

We will re-evaluate B7 once **all** of the following are true:

1. **Tagged, non-draft WIT** at `wasi:blobstore@0.2.x` (or successor
   package name) with rename churn settled — i.e. PRs #26 and #23 on
   the proposal repo are either merged or formally closed/withdrawn,
   and the proposal has advanced to WASI Phase 2 (Proposed Spec Text)
   or higher.
2. **Upstream host baseline.** Either a `wasi-blobstore` crate landed
   in `bytecodealliance/wasmtime` (with a tagged release we can target,
   mirroring our `wasi:keyvalue` / `wasi:config` cadence), _or_ another
   recognised runtime (jco / wasm-tools) ships a maintained host
   adapter we can spec-align against.
3. **Conformance fixtures.** Blobstore fixtures present in
   `WebAssembly/wasi-testsuite` (or the proposal repo's own
   `test/` directory), so the wamr adapter can be gated under
   `zig build wasi-testsuite` / `wasi-p3-testsuite` like every other
   shipped interface.
4. **P3 async story decided.** Either a `wit-0.3.0-draft/` tree exists
   in the proposal repo binding to `wasi:io@0.3` `stream` / `future`
   primitives, or the WASI subgroup explicitly resolves that
   `wasi:blobstore` stays on `wasi:io@0.2.1` semantics for the
   foreseeable future.
5. **`wasi:cli/run-with-server`**: a draft world (or issue with
   accepted API sketch) exists in `WebAssembly/wasi-cli`. Until then
   we treat the `run-with-server` half of B7 as a documentation entry
   only.

When all five hold, the implementation plan mirrors
[PR #601](https://github.com/cataggar/wamr/pull/601) (wasi:keyvalue
memory-store host adapter): vendor the pinned WIT under
`docs/wasi-blobstore-wit-vendored/`, ship a memory-backed
`StringHashMapUnmanaged(StringHashMapUnmanaged([]const u8))` adapter in
`src/component/wasi_cli_adapter.zig`, register through
`populateWasiProviders`, add unit tests for container/object lifecycle
+ `list-objects`, and bump the Detail-table / Status-overview rows
above.

## See also

* [`README.md`](../README.md) — quick-start build / test invocations.
* [`tests/wasi-conformance-audit.md`](../tests/wasi-conformance-audit.md)
  — date-stamped audit log behind `tests/wasi-testsuite-expectations.toml`
  (Preview 1 + 2; #583 C2 / PR #584).
* [`wasi-impl-audit.md`](wasi-impl-audit.md) — per-interface,
  per-method audit of every registered WIT method vs. upstream
  (#583 D / PR follows this one).
* [`src/component/wasi_cli_adapter.zig`](../src/component/wasi_cli_adapter.zig)
  — the production host adapter; `populateWasiProviders` is the
  registration seam.
* [`build.zig`](../build.zig) — `wasi-testsuite` / `wasi-p2-testsuite` /
  `wasi-p3-testsuite` build steps (lines ~214–275 and ~713–740).
* [WASI roadmap](https://wasi.dev/roadmap) — upstream interface
  stability tracker.
* [Component Model](https://github.com/webassembly/component-model)
  — the binding spec the adapter targets.
