# WASI WIT-vs-impl audit

| Field         | Value                                                  |
| ------------- | ------------------------------------------------------ |
| Origin commit | [`fa850da4`](https://github.com/cataggar/wamr/commit/fa850da41df12e39b9cac461a92d81243958ce15) |
| Audit date    | 2026-05-16                                             |
| Adapter file  | [`src/component/wasi_cli_adapter.zig`](../src/component/wasi_cli_adapter.zig) |
| Tracker       | [#583 D](https://github.com/cataggar/wamr/issues/583) (post-Preview-3 hardening) |
| Companion     | [`tests/wasi-conformance-audit.md`](../tests/wasi-conformance-audit.md) — fixture-pass-rate audit; this doc is the method-level companion. |

This audit walks every WIT method on every WASI interface that wamr
registers, and marks each one as **✅ Implemented**, **⚠️ Stub**,
**❌ Missing**, or **N/A** (deliberately out of scope or type-only).
The companion `docs/wasi.md` gives the embedder-facing matrix; this
file is the per-method ground-truth so a future contributor can see
exactly which arms are wired, which are canned errors, and which
upstream WIT methods aren't yet bound.

## Table of contents

- [Methodology](#methodology)
- [Status legend](#status-legend)
- [Summary](#summary)
- [Findings: ❌ Missing arms](#findings-❌-missing-arms)
- [Findings: ⚠️ Stubbed arms](#findings-⚠️-stubbed-arms)
- [Preview 2 (0.2.x) interfaces](#preview-2-02x-interfaces)
- [Preview 3 (0.3.0) interfaces](#preview-3-030-interfaces)
- [Cross-version (no version-multiplex split)](#cross-version-no-version-multiplex-split)

## Methodology

The audit is **reproducible** — every column can be regenerated from
`main` with the steps below. Future audits should re-run the same
grep / cross-reference and update this file.

### Enumerate registered interfaces

```console
$ grep -n 'populateWasi' src/component/wasi_cli_adapter.zig \
    | grep -E 'try adapter\.populateWasi'
```

Each call inside the top-level `populateWasiProviders`
([`wasi_cli_adapter.zig:21853`](../src/component/wasi_cli_adapter.zig))
is the registration seam for one WIT interface (or one
version-multiplexed group of interfaces, e.g. `wasi:cli/exit` 0.2 +
0.3).

### Enumerate registered methods per interface

Each `populateWasi*` body holds either an inline list of
`.members.put(...)` calls or a `[_]M{...}` table whose entries are
`.{ .name = "<wit-member-name>", .call = &<host-fn> }`. The
following Python one-liner scans both patterns and emits one
`L<line>: <member-name> -> <host-fn>` row per registration:

```python
import re
src = open("src/component/wasi_cli_adapter.zig").read()
fn_re   = re.compile(r'^\s*pub fn (populateWasi\w+)\s*\(', re.MULTILINE)
put_re  = re.compile(
    r'members\.put\(\s*[^,]+,\s*"([^"]+)"\s*,\s*\.\{\s*\.func\s*='
    r'\s*\.\{[^}]*?\.call\s*=\s*&([^,}\s]+)',
    re.MULTILINE | re.DOTALL)
tbl_re  = re.compile(
    r'\.\{\s*\.name\s*=\s*"([^"]+)"\s*,\s*\.call\s*=\s*&([^\n]+?)\s*\}',
    re.MULTILINE)
# … iterate populates(start_line) and slice; dedupe by member name
```

### Cross-reference against upstream WIT

Each interface section below links to the upstream
`github.com/WebAssembly/<repo>/wit/<file>.wit` source. The audit
compares the registered names against the WIT's `func` /
`static func` / `[constructor]` / `[method]` declarations.
Resource-drop entries are required by the canonical ABI for every
`resource <X>` and are counted as one method per resource.

### Classify each method

* **✅ Implemented** — the host function performs real work
  (syscalls, library calls, in-memory state mutation).
* **⚠️ Stub** — registered, links cleanly, but returns a canned
  error (`error::other`, `access-denied`) or a no-op outcome for
  the captured-buffer / default-deny profile. The guest's call
  succeeds at the ABI layer but the host has no real backing.
* **❌ Missing** — declared in upstream WIT but **not registered**
  in wamr. A guest that imports the method will fail at
  `linkImports`. Counts toward the gap percentage.
* **N/A** — explicitly out of scope: a type-only interface (no
  WIT functions, e.g. `wasi:io/streams@0.3.0`), a deliberately-
  unimplemented feature gate, or an upstream WIT enum that is
  lifted as a discriminant byte without host dispatch.

## Status legend

| Mark | Meaning |
| :--- | :------ |
| ✅   | Implemented (real syscall / library call). |
| ⚠️   | Stub — registered, canned error or no-op return. |
| ❌   | Missing — upstream WIT declares it, wamr does not register it. |
| N/A | Type-only interface or deliberately out-of-scope. |

## Summary

| Surface       | Total methods | ✅ Implemented | ⚠️ Stubbed | ❌ Missing | Implemented % |
| ------------- | ------------: | -------------: | ---------: | --------: | ------------: |
| Preview 2 (0.2.x) — `wasi:cli` + `wasi:io` + `wasi:clocks` + `wasi:random` + `wasi:filesystem` + `wasi:sockets` + `wasi:http` | 188 | 188 | — | 0 | 100.0 % |
| Preview 3 (0.3.0) | 135 | 135 | — | 0 | 100.0 % |
| `wasi:keyvalue@0.2.0-draft2` | 15 | 12 | 3 | 0 | 80.0 % |
| `wasi:logging@0.1.0-draft` | 1 | 1 | — | — | 100.0 % |
| `wasi:config/store@0.2.0-rc.1` | 2 | 2 | — | — | 100.0 % |
| **Total** | **341** | **338** | **3** | **0** | **99.1 %** |

Stubbed %: 0.9 %. Missing %: 0.0 %. (Type-only WIT instances —
`wasi:io/streams@0.3.0`, `wasi:io/error@0.3.0`,
`wasi:cli/types@0.3.0`, `wasi:clocks/types@0.3.0` — declare no host
methods and so contribute zero rows to either denominator.)

**Audit-driven follow-ups (post-origin-commit):**

* **PR #604 follow-up** — 8 missing 0.2 arms flipped from ❌ to ✅:
  the 6 `wasi:io/streams` slow-path / splice methods, plus
  `wasi:io/error.to-debug-string` and `wasi:sockets/network.network-error-code`.
  Preview-2 coverage rose from 92.0 % → 96.3 %; overall coverage
  rose from 94.7 % → 97.1 %.

## Findings: ❌ Missing arms

All 15 P2 ❌ rows are now ✅. The Preview-2 surface is 100 %. The
only remaining non-implemented WIT methods are the 3 ⚠️ stubbed
`wasi:keyvalue/atomics` CAS variants (kept as documented stubs in
the memory-store backend per PR #608).

### `wasi:http/types@0.2.x` — closed in W11-2 (PR #612)

[`http-0.2-types.wit`](https://github.com/WebAssembly/wasi-http/blob/main/wit/types.wit)

All seven previously-missing arms (1 free fn + 6 `request-options`
timeout getters/setters) are now bound. The 0.2 spec uses the
unprefixed getter names (`connect-timeout` etc.) while 0.3 uses
`get-connect-timeout` etc.; both surfaces share the `RequestOptions`
rep struct and the `HttpErrorCode` enum (the 0.2 and 0.3
`error-code` variants happen to share WIT-declaration order).

### `wasi:io@0.2.x` + `wasi:sockets/network@0.2.x` — closed in W11-1 (PR #615)

[`io-0.2-streams.wit`](https://github.com/WebAssembly/wasi-io/blob/main/wit/streams.wit),
[`io-0.2-error.wit`](https://github.com/WebAssembly/wasi-io/blob/main/wit/error.wit),
[`sockets-0.2-network.wit`](https://github.com/WebAssembly/wasi-sockets/blob/main/wit/network.wit)

All eight previously-missing arms (`input-stream.skip` /
`blocking-skip`, `output-stream.write-zeroes` /
`blocking-write-zeroes-and-flush` / `splice` / `blocking-splice`,
`error.to-debug-string`, `network-error-code`) are now bound.
Linux descriptor-backed streams use a `splice(2)` fast path; other
platforms and unsupported endpoint combinations retain the buffer-through
implementation.

`[method]response-outparam.send-informational` remains the only
unbound member: it is `@unstable(feature = informational-outbound-responses)`
upstream and wamr unconditionally omits it (tracked under
[#583 A5](https://github.com/cataggar/wamr/issues/583)).

Timeout values are advisory in this PR — the worker reads them off
the rep struct but `std.http.Client.fetch` in
`httpOutgoingHandlerHandle` does not yet thread them through to
the underlying TCP/TLS handshake. Future work under #583 A5 wires
them into `std.http.Client.Request`'s timeout options.

### Audit fill-ins (resolved in PR #604 follow-up)

The following eight rows used to be ❌; PR #604's follow-up (this
patch series, agent W11-1) flipped them all to ✅. The detailed
per-interface tables below reflect the post-fill-in state.

* `wasi:io/streams@0.2.x`:
  - `[method]input-stream.skip`
  - `[method]input-stream.blocking-skip`
  - `[method]output-stream.write-zeroes`
  - `[method]output-stream.blocking-write-zeroes-and-flush`
  - `[method]output-stream.splice`
  - `[method]output-stream.blocking-splice`
* `wasi:io/error@0.2.x`:
  - `[method]error.to-debug-string`
* `wasi:sockets/network@0.2.x`:
  - `network-error-code` (free fn)

## Findings: ⚠️ Stubbed arms

Three WIT methods register as host bindings but return canned
errors. Each is intentional — the in-memory keyvalue backend does
not implement CAS round-trips. Guests can compile against the WIT
unchanged; a real CAS request is rejected with a typed error.

| WIT method | Stub behaviour | Source |
| --- | --- | --- |
| `swap` (`wasi:keyvalue/atomics`) | Returns `cas-error::store-error(error::other("compare-and-swap not implemented in memory store"))`. | [`wasi_cli_adapter.zig:21092`](../src/component/wasi_cli_adapter.zig) |
| `[static]cas.new` (`wasi:keyvalue/atomics`) | Returns `error::other("…")`. No `cas` handle is ever produced. | [`wasi_cli_adapter.zig:21123`](../src/component/wasi_cli_adapter.zig) |
| `[method]cas.current` (`wasi:keyvalue/atomics`) | Returns `error::other("…")`. Defensive — `cas.new` never succeeds. | [`wasi_cli_adapter.zig:21140`](../src/component/wasi_cli_adapter.zig) |

`[resource-drop]cas` (line 21158) is a real no-op — semantically
correct because no live CAS handle ever exists.

The captured-buffer profile makes every `get-terminal-*` (both 0.2
and 0.3) return `none`. That is the spec-conformant answer when
stdio is not a TTY, so it is classified ✅ Implemented, not ⚠️ Stub.

## Preview 2 (0.2.x) interfaces

### `wasi:cli/stdout` & `wasi:io/streams`

Upstream WIT: [`cli-0.2-stdio.wit`](https://github.com/WebAssembly/wasi-cli/blob/main/wit/stdio.wit),
[`io-0.2-streams.wit`](https://github.com/WebAssembly/wasi-io/blob/main/wit/streams.wit)

Registered together by
[`populateWasiCliRun`](../src/component/wasi_cli_adapter.zig).

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `get-stdout` | ✅ | wasi_cli_adapter.zig:4607 | Returns captured stdout sink handle. |
| `[method]output-stream.blocking-write-and-flush` | ✅ | :4614 | — |
| `[method]output-stream.write` | ✅ | :4619 | — |
| `[method]output-stream.check-write` | ✅ | :4624 | — |
| `[method]output-stream.blocking-flush` | ✅ | :4629 | — |
| `[method]output-stream.flush` | ✅ | :4634 | Aliased to `blocking-flush`. |
| `[method]output-stream.subscribe` | ✅ | :4639 | — |
| `[method]output-stream.write-zeroes` | ✅ | :4811 / impl :8281 | Audit fill-in (#583, PR #604). |
| `[method]output-stream.blocking-write-zeroes-and-flush` | ✅ | :4816 / impl :8323 | Audit fill-in (#583, PR #604). |
| `[method]output-stream.splice` | ✅ | :4821 / impl :8377 | Nonblocking Linux pipe-to-pipe `splice(2)`; buffer fallback otherwise. |
| `[method]output-stream.blocking-splice` | ✅ | :4826 / impl :8377 | Blocking Linux `splice(2)` with `EAGAIN` readiness wait/retry; buffer fallback otherwise. |
| `[resource-drop]output-stream` | ✅ | :4649 | — |
| `[method]input-stream.subscribe` | ✅ | :4644 | — |
| `[method]input-stream.read` | ✅ | :4660 | Aliased to `blocking-read`. |
| `[method]input-stream.blocking-read` | ✅ | :4655 | — |
| `[method]input-stream.skip` | ✅ | :4801 / impl :8231 | Audit fill-in (#583, PR #604). |
| `[method]input-stream.blocking-skip` | ✅ | :4806 / impl :8231 | Same host helper as `skip`. |
| `[resource-drop]input-stream` | ✅ | :4665 | — |

Coverage: `wasi:cli/stdout` 1/1 (100 %); `wasi:io/streams` 17/17
(100 %). Six audit-arm methods (`skip`, `blocking-skip`,
`write-zeroes`, `blocking-write-zeroes-and-flush`, `splice`,
`blocking-splice`) flipped from ❌ to ✅ in PR #604's follow-up
(audit-driven). Linux descriptor-backed streams now use `splice(2)`;
ordinary splice restricts the fast path to pipe pairs and uses
`SPLICE_F_NONBLOCK`, while blocking-splice waits and retries `EAGAIN`.
Unsupported endpoint pairs and non-Linux targets still read and write
through the bounded host scratch buffer.

### `wasi:cli/stderr`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-stderr` | ✅ | wasi_cli_adapter.zig:4682 |

1/1 (100 %).

### `wasi:cli/stdin`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-stdin` | ✅ | wasi_cli_adapter.zig:4785 |

1/1 (100 %).

### `wasi:cli/exit`

Upstream WIT: [`exit.wit`](https://github.com/WebAssembly/wasi-cli/blob/main/wit/exit.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `exit` | ✅ | wasi_cli_adapter.zig:4698 | Sets adapter exit code; raises `error.Trap`. |
| `exit-with-code` (`@unstable`) | ✅ | :4701 | Unconditionally registered — divergence from wasmtime documented at :4901. |

2/2 (100 %).

### `wasi:cli/environment`

Upstream WIT: [`environment.wit`](https://github.com/WebAssembly/wasi-cli/blob/main/wit/environment.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-environment` | ✅ | wasi_cli_adapter.zig:4717 |
| `get-arguments` | ✅ | :4720 |
| `initial-cwd` | ✅ | :4723 |

3/3 (100 %).

### `wasi:cli/terminal-{stdin,stdout,stderr,input,output}`

Upstream WIT: [`terminal.wit`](https://github.com/WebAssembly/wasi-cli/blob/main/wit/terminal.wit)

Captured-buffer mode — `get-terminal-*` returns `none` because
stdio is not a TTY (spec-conformant).

| Interface | WIT method | Status | Adapter location |
| --- | --- | :-: | --- |
| `terminal-stdin` | `get-terminal-stdin` | ✅ | :4745 |
| `terminal-stdout` | `get-terminal-stdout` | ✅ | :4751 |
| `terminal-stderr` | `get-terminal-stderr` | ✅ | :4757 |
| `terminal-input` | `[resource-drop]terminal-input` | ✅ | :4763 |
| `terminal-output` | `[resource-drop]terminal-output` | ✅ | :4769 |

5/5 (100 %). `terminal-input` / `terminal-output` are resources
with **no** WIT methods upstream (just the bare resource), so only
the canonical-ABI-required `[resource-drop]` is registered.

### `wasi:io/poll`

Upstream WIT: [`poll.wit`](https://github.com/WebAssembly/wasi-io/blob/main/wit/poll.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[method]pollable.ready` | ✅ | wasi_cli_adapter.zig:5644 |
| `[method]pollable.block` | ✅ | :5647 |
| `poll` (free fn) | ✅ | :5650 |
| `[resource-drop]pollable` | ✅ | :5653 |

4/4 (100 %).

### `wasi:io/error`

Upstream WIT: [`error.wit`](https://github.com/WebAssembly/wasi-io/blob/main/wit/error.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `[method]error.to-debug-string` | ✅ | :5831 / impl :8434 | Audit fill-in (#583, PR #604). wamr does not track io-error provenance — returns an opaque `"wasi:io error (opaque host handle #N)"` description. |
| `[resource-drop]error` | ✅ | :5660 | — |

2/2 (100 %). The `to-debug-string` method was flipped from ❌ to ✅
in PR #604's follow-up — the host returns a best-effort opaque
description (handle-suffixed) since wamr does not currently keep an
io-error table.

### `wasi:clocks/wall-clock`

Upstream WIT: [`wall-clock.wit`](https://github.com/WebAssembly/wasi-clocks/blob/main/wit/wall-clock.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `now` | ✅ | wasi_cli_adapter.zig:5703 |
| `resolution` | ✅ | :5706 |

2/2 (100 %).

### `wasi:clocks/monotonic-clock`

Upstream WIT: [`monotonic-clock.wit`](https://github.com/WebAssembly/wasi-clocks/blob/main/wit/monotonic-clock.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `now` | ✅ | wasi_cli_adapter.zig:5729 |
| `resolution` | ✅ | :5732 |
| `subscribe-instant` | ✅ | :5735 |
| `subscribe-duration` | ✅ | :5738 |

4/4 (100 %).

### `wasi:random/random`

Upstream WIT: [`random.wit`](https://github.com/WebAssembly/wasi-random/blob/main/wit/random.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-random-bytes` | ✅ | wasi_cli_adapter.zig:5843 |
| `get-random-u64` | ✅ | :5846 |

2/2 (100 %).

### `wasi:random/insecure`

Upstream WIT: [`insecure.wit`](https://github.com/WebAssembly/wasi-random/blob/main/wit/insecure.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-insecure-random-bytes` | ✅ | wasi_cli_adapter.zig:5863 |
| `get-insecure-random-u64` | ✅ | :5866 |

2/2 (100 %).

### `wasi:random/insecure-seed`

Upstream WIT: [`insecure-seed.wit`](https://github.com/WebAssembly/wasi-random/blob/main/wit/insecure-seed.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `insecure-seed` | ✅ | wasi_cli_adapter.zig:5883 |

1/1 (100 %).

### `wasi:filesystem/preopens`

Upstream WIT: [`preopens.wit`](https://github.com/WebAssembly/wasi-filesystem/blob/main/wit/preopens.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-directories` | ✅ | wasi_cli_adapter.zig:8078 |

1/1 (100 %).

### `wasi:filesystem/types`

Upstream WIT: [`types.wit`](https://github.com/WebAssembly/wasi-filesystem/blob/main/wit/types.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[method]descriptor.get-type` | ✅ | wasi_cli_adapter.zig:8094 |
| `[method]descriptor.get-flags` | ✅ | :8095 |
| `[method]descriptor.stat` | ✅ | :8096 |
| `[method]descriptor.stat-at` | ✅ | :8097 |
| `[method]descriptor.set-times` | ✅ | :8098 |
| `[method]descriptor.set-times-at` | ✅ | :8099 |
| `[method]descriptor.open-at` | ✅ | :8100 |
| `[method]descriptor.read-via-stream` | ✅ | :8101 |
| `[method]descriptor.write-via-stream` | ✅ | :8102 |
| `[method]descriptor.append-via-stream` | ✅ | :8103 |
| `[method]descriptor.read` | ✅ | :8104 |
| `[method]descriptor.write` | ✅ | :8105 |
| `[method]descriptor.sync` | ✅ | :8106 |
| `[method]descriptor.sync-data` | ✅ | :8107 |
| `[method]descriptor.set-size` | ✅ | :8108 |
| `[method]descriptor.advise` | ✅ | :8109 |
| `[method]descriptor.is-same-object` | ✅ | :8110 |
| `[method]descriptor.metadata-hash` | ✅ | :8111 |
| `[method]descriptor.metadata-hash-at` | ✅ | :8112 |
| `[method]descriptor.create-directory-at` | ✅ | :8114 |
| `[method]descriptor.unlink-file-at` | ✅ | :8115 |
| `[method]descriptor.remove-directory-at` | ✅ | :8116 |
| `[method]descriptor.rename-at` | ✅ | :8117 |
| `[method]descriptor.link-at` | ✅ | :8118 |
| `[method]descriptor.symlink-at` | ✅ | :8119 |
| `[method]descriptor.readlink-at` | ✅ | :8120 |
| `[method]descriptor.read-directory` | ✅ | :8122 |
| `[method]directory-entry-stream.read-directory-entry` | ✅ | :8123 |
| `[resource-drop]directory-entry-stream` | ✅ | :8124 |
| `filesystem-error-code` (free fn) | ✅ | :8125 |
| `[resource-drop]descriptor` | ✅ | :8126 |

31/31 (100 %). All 27 descriptor methods, both `directory-entry-stream`
members, both resource-drops, and the free `filesystem-error-code`
downcast are wired.

### `wasi:sockets/network`

Upstream WIT: [`network.wit`](https://github.com/WebAssembly/wasi-sockets/blob/main/wit/network.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `network-error-code` (free fn) | ✅ | :14279 / impl :8473 | Audit fill-in (#583, PR #604). Always returns `option::none` — wamr's sockets paths return typed `error-code` payloads directly, so the io-error indirection carries no sockets provenance to downcast. |
| `[resource-drop]network` | ✅ | :13687 | — |

2/2 (100 %). The free function `network-error-code` was flipped
from ❌ to ✅ in PR #604's follow-up.

### `wasi:sockets/instance-network`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `instance-network` | ✅ | wasi_cli_adapter.zig:13701 |

1/1 (100 %).

### `wasi:sockets/tcp`

Upstream WIT: [`tcp.wit`](https://github.com/WebAssembly/wasi-sockets/blob/main/wit/tcp.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[method]tcp-socket.start-bind` | ✅ | wasi_cli_adapter.zig:13723 |
| `[method]tcp-socket.finish-bind` | ✅ | :13724 |
| `[method]tcp-socket.start-connect` | ✅ | :13725 |
| `[method]tcp-socket.finish-connect` | ✅ | :13726 |
| `[method]tcp-socket.start-listen` | ✅ | :13727 |
| `[method]tcp-socket.finish-listen` | ✅ | :13728 |
| `[method]tcp-socket.accept` | ✅ | :13729 |
| `[method]tcp-socket.local-address` | ✅ | :13730 |
| `[method]tcp-socket.remote-address` | ✅ | :13731 |
| `[method]tcp-socket.shutdown` | ✅ | :13732 |
| `[method]tcp-socket.set-listen-backlog-size` | ✅ | :13734 |
| `[method]tcp-socket.set-keep-alive-enabled` | ✅ | :13735 |
| `[method]tcp-socket.set-keep-alive-idle-time` | ✅ | :13736 |
| `[method]tcp-socket.set-keep-alive-interval` | ✅ | :13737 |
| `[method]tcp-socket.set-keep-alive-count` | ✅ | :13738 |
| `[method]tcp-socket.set-hop-limit` | ✅ | :13739 |
| `[method]tcp-socket.set-receive-buffer-size` | ✅ | :13740 |
| `[method]tcp-socket.set-send-buffer-size` | ✅ | :13741 |
| `[method]tcp-socket.keep-alive-enabled` | ✅ | :13743 |
| `[method]tcp-socket.keep-alive-idle-time` | ✅ | :13744 |
| `[method]tcp-socket.keep-alive-interval` | ✅ | :13745 |
| `[method]tcp-socket.keep-alive-count` | ✅ | :13746 |
| `[method]tcp-socket.hop-limit` | ✅ | :13747 |
| `[method]tcp-socket.receive-buffer-size` | ✅ | :13748 |
| `[method]tcp-socket.send-buffer-size` | ✅ | :13749 |
| `[method]tcp-socket.address-family` | ✅ | :13751 |
| `[method]tcp-socket.is-listening` | ✅ | :13752 |
| `[method]tcp-socket.subscribe` | ✅ | :13754 |
| `[resource-drop]tcp-socket` | ✅ | :13755 |

29/29 (100 %). Allow-list-gated; real
`bind` / `connect` / `listen` / `accept`; SO_REUSEADDR; POSIX +
Windows parity (`bindAndGetsockname` shim, [PR #587](https://github.com/cataggar/wamr/pull/587)).

### `wasi:sockets/tcp-create-socket`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `create-tcp-socket` | ✅ | wasi_cli_adapter.zig:13773 |

1/1 (100 %).

### `wasi:sockets/udp`

Upstream WIT: [`udp.wit`](https://github.com/WebAssembly/wasi-sockets/blob/main/wit/udp.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[method]udp-socket.start-bind` | ✅ | wasi_cli_adapter.zig:13793 |
| `[method]udp-socket.finish-bind` | ✅ | :13794 |
| `[method]udp-socket.stream` | ✅ | :13795 |
| `[method]udp-socket.local-address` | ✅ | :13796 |
| `[method]udp-socket.remote-address` | ✅ | :13797 |
| `[method]udp-socket.unicast-hop-limit` | ✅ | :13799 |
| `[method]udp-socket.set-unicast-hop-limit` | ✅ | :13800 |
| `[method]udp-socket.receive-buffer-size` | ✅ | :13801 |
| `[method]udp-socket.set-receive-buffer-size` | ✅ | :13802 |
| `[method]udp-socket.send-buffer-size` | ✅ | :13803 |
| `[method]udp-socket.set-send-buffer-size` | ✅ | :13804 |
| `[method]udp-socket.address-family` | ✅ | :13806 |
| `[method]udp-socket.subscribe` | ✅ | :13807 |
| `[resource-drop]udp-socket` | ✅ | :13808 |
| `[resource-drop]incoming-datagram-stream` | ✅ | :13810 |
| `[method]incoming-datagram-stream.receive` | ✅ | :13811 |
| `[method]incoming-datagram-stream.subscribe` | ✅ | :13812 |
| `[resource-drop]outgoing-datagram-stream` | ✅ | :13813 |
| `[method]outgoing-datagram-stream.check-send` | ✅ | :13814 |
| `[method]outgoing-datagram-stream.send` | ✅ | :13815 |
| `[method]outgoing-datagram-stream.subscribe` | ✅ | :13816 |

21/21 (100 %).

### `wasi:sockets/udp-create-socket`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `create-udp-socket` | ✅ | wasi_cli_adapter.zig:13834 |

1/1 (100 %).

### `wasi:sockets/ip-name-lookup`

Upstream WIT: [`ip-name-lookup.wit`](https://github.com/WebAssembly/wasi-sockets/blob/main/wit/ip-name-lookup.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `resolve-addresses` (free fn) | ✅ | wasi_cli_adapter.zig:13851 | `std.net.getAddressList`-backed; allow-list-gated. |
| `[method]resolve-address-stream.resolve-next-address` | ✅ | :13854 | — |
| `[method]resolve-address-stream.subscribe` | ✅ | :13857 | — |
| `[resource-drop]resolve-address-stream` | ✅ | :13860 | — |

4/4 (100 %).

### `wasi:http/types`

Upstream WIT: [`types.wit`](https://github.com/WebAssembly/wasi-http/blob/main/wit/types.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `http-error-code` (free fn) | ✅ | wasi_cli_adapter.zig:20919 | Cross-references the borrowed `io-error` handle against `WasiCliAdapter.http_io_errors`; returns `some(error-code)` for HTTP-origin errors, `none` otherwise. |
| `[constructor]fields` | ✅ | wasi_cli_adapter.zig:20370 | — |
| `[static]fields.from-list` | ✅ | :20371 | — |
| `[method]fields.entries` | ✅ | :20372 | — |
| `[method]fields.get` | ✅ | :20373 | — |
| `[method]fields.has` | ✅ | :20374 | — |
| `[method]fields.set` | ✅ | :20375 | — |
| `[method]fields.append` | ✅ | :20376 | — |
| `[method]fields.delete` | ✅ | :20377 | — |
| `[method]fields.clone` | ✅ | :20378 | — |
| `[resource-drop]fields` | ✅ | :20379 | — |
| `[constructor]outgoing-request` | ✅ | :20381 | — |
| `[method]outgoing-request.method` | ✅ | :20382 | — |
| `[method]outgoing-request.set-method` | ✅ | :20383 | — |
| `[method]outgoing-request.path-with-query` | ✅ | :20384 | — |
| `[method]outgoing-request.set-path-with-query` | ✅ | :20385 | — |
| `[method]outgoing-request.scheme` | ✅ | :20386 | — |
| `[method]outgoing-request.set-scheme` | ✅ | :20387 | — |
| `[method]outgoing-request.authority` | ✅ | :20388 | — |
| `[method]outgoing-request.set-authority` | ✅ | :20389 | — |
| `[method]outgoing-request.headers` | ✅ | :20390 | — |
| `[method]outgoing-request.body` | ✅ | :20391 | — |
| `[resource-drop]outgoing-request` | ✅ | :20392 | — |
| `[constructor]outgoing-response` | ✅ | :20394 | — |
| `[method]outgoing-response.status-code` | ✅ | :20395 | — |
| `[method]outgoing-response.set-status-code` | ✅ | :20396 | — |
| `[method]outgoing-response.headers` | ✅ | :20397 | — |
| `[method]outgoing-response.body` | ✅ | :20398 | — |
| `[resource-drop]outgoing-response` | ✅ | :20399 | — |
| `[method]incoming-request.method` | ✅ | :20401 | — |
| `[method]incoming-request.path-with-query` | ✅ | :20402 | — |
| `[method]incoming-request.scheme` | ✅ | :20403 | — |
| `[method]incoming-request.authority` | ✅ | :20404 | — |
| `[method]incoming-request.headers` | ✅ | :20405 | — |
| `[method]incoming-request.consume` | ✅ | :20406 | — |
| `[resource-drop]incoming-request` | ✅ | :20407 | — |
| `[method]incoming-response.status` | ✅ | :20409 | — |
| `[method]incoming-response.headers` | ✅ | :20410 | — |
| `[method]incoming-response.consume` | ✅ | :20411 | — |
| `[resource-drop]incoming-response` | ✅ | :20412 | — |
| `[method]incoming-body.stream` | ✅ | :20414 | — |
| `[static]incoming-body.finish` | ✅ | :20415 | — |
| `[resource-drop]incoming-body` | ✅ | :20416 | — |
| `[method]outgoing-body.write` | ✅ | :20417 | — |
| `[static]outgoing-body.finish` | ✅ | :20418 | — |
| `[resource-drop]outgoing-body` | ✅ | :20419 | — |
| `[method]future-incoming-response.subscribe` | ✅ | :20421 | — |
| `[method]future-incoming-response.get` | ✅ | :20422 | — |
| `[resource-drop]future-incoming-response` | ✅ | :20423 | — |
| `[method]future-trailers.subscribe` | ✅ | :20424 | — |
| `[method]future-trailers.get` | ✅ | :20425 | — |
| `[resource-drop]future-trailers` | ✅ | :20426 | — |
| `[constructor]request-options` | ✅ | :20906 | — |
| `[method]request-options.connect-timeout` | ✅ | :20909 | 0.2-style unprefixed getter name (`get-` prefix only in 0.3). Stored in nanoseconds, copied into worker-owned state, and applied to initial DNS/TCP acquisition. Zig's lazy TLS handshake and automatic redirect reconnects are not covered by this deadline. |
| `[method]request-options.set-connect-timeout` | ✅ | :20910 | Stores `option<duration>` (nanoseconds). Returns `result` ok unconditionally. |
| `[method]request-options.first-byte-timeout` | ✅ | :20911 | Stored for round-tripping only; enforcing it requires a deadline-aware HTTP/TLS reader and remains #616 A1b/A7 work. |
| `[method]request-options.set-first-byte-timeout` | ✅ | :20912 | Same as above; field `first_byte_timeout_ns`. |
| `[method]request-options.between-bytes-timeout` | ✅ | :20913 | Stored for round-tripping only; enforcing it requires a deadline-aware HTTP/TLS reader and remains #616 A1b/A7 work. |
| `[method]request-options.set-between-bytes-timeout` | ✅ | :20914 | Same as above; field `between_bytes_timeout_ns`. |
| `[resource-drop]request-options` | ✅ | :20915 | — |
| `[method]response-outparam.send-informational` (`@unstable`) | N/A | — | Unstable feature; not yet registered. Tracked under [#583 A5](https://github.com/cataggar/wamr/issues/583). |
| `[static]response-outparam.set` | ✅ | :20916 | — |
| `[resource-drop]response-outparam` | ✅ | :20917 | — |

62/62 stable (100 %). `send-informational` is the only
`@unstable` feature and is excluded from the implemented %.

### `wasi:http/outgoing-handler`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `handle` | ✅ | wasi_cli_adapter.zig:20450 |

1/1 (100 %). `std.http.Client.fetch` real outbound, http + https.

### `wasi:http/incoming-handler`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `handle` | ✅ | wasi_cli_adapter.zig:20468 |

1/1 (100 %). Real TCP-listener-backed dispatch
([PR #580](https://github.com/cataggar/wamr/pull/580)).

## Preview 3 (0.3.0) interfaces

### `wasi:cli/stdin@0.3.0`, `stdout@0.3.0`, `stderr@0.3.0`

| Interface | WIT method | Status | Adapter location |
| --- | --- | :-: | --- |
| `stdin@0.3.0` | `read-via-stream` | ✅ | wasi_cli_adapter.zig:4858 |
| `stdout@0.3.0` | `write-via-stream` | ✅ | :4872 |
| `stderr@0.3.0` | `write-via-stream` | ✅ | :4886 |

3/3 (100 %). Host-attached `stream<u8>` + ready
`future<result<_,error-code>>`
([PR #514](https://github.com/cataggar/wamr/pull/514) / #548).

### `wasi:cli/exit@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `exit` | ✅ | wasi_cli_adapter.zig:4908 |
| `exit-with-code` (`@unstable`) | ✅ | :4911 |

2/2 (100 %).

### `wasi:cli/environment@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `get-environment` | ✅ | wasi_cli_adapter.zig:4928 | — |
| `get-arguments` | ✅ | :4931 | — |
| `get-initial-cwd` | ✅ | :4935 | 0.3 rename of `initial-cwd`. |

3/3 (100 %).

### `wasi:cli/types@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| _(none — type-only)_ | N/A | wasi_cli_adapter.zig:4947 |

0/0 (N/A). Holds the `error-code { io, illegal-byte-sequence, pipe }`
enum only. Lifted as a u8 discriminant in canonical-ABI.

### `wasi:cli/terminal-*@0.3.0`

Identical surface to 0.2.

| Interface | WIT method | Status | Adapter location |
| --- | --- | :-: | --- |
| `terminal-stdin@0.3.0` | `get-terminal-stdin` | ✅ | :4969 |
| `terminal-stdout@0.3.0` | `get-terminal-stdout` | ✅ | :4975 |
| `terminal-stderr@0.3.0` | `get-terminal-stderr` | ✅ | :4981 |
| `terminal-input@0.3.0` | `[resource-drop]terminal-input` | ✅ | :4987 |
| `terminal-output@0.3.0` | `[resource-drop]terminal-output` | ✅ | :4993 |

5/5 (100 %).

### `wasi:io/streams@0.3.0` / `wasi:io/error@0.3.0`

| Interface | WIT method | Status | Adapter location |
| --- | --- | :-: | --- |
| `streams@0.3.0` | _(none — type-only)_ | N/A | wasi_cli_adapter.zig:5676 |
| `error@0.3.0` | _(none — type-only)_ | N/A | :5676 |

0/0 (N/A). The P3 `stream<u8>` / `future<…>` engine lives in the
canonical-ABI (see `comp_inst.streams` and the executor in
`stream_canon.zig` / `async_canon.zig`), not on the adapter
([PR #510](https://github.com/cataggar/wamr/pull/510) / #481).

### `wasi:clocks/monotonic-clock@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `now` | ✅ | wasi_cli_adapter.zig:5763 | — |
| `get-resolution` | ✅ | :5766 | 0.3 rename of `resolution`. |
| `wait-for` | ✅ | :5769 | Host-driven `task.cancel`-aware ([PR #558](https://github.com/cataggar/wamr/pull/558)). |
| `wait-until` | ✅ | :5772 | Same. |

4/4 (100 %).

### `wasi:clocks/system-clock@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `now` | ✅ | wasi_cli_adapter.zig:5792 | 0.3 rename `wall-clock` → `system-clock`. |
| `get-resolution` | ✅ | :5795 | — |

2/2 (100 %).

### `wasi:clocks/types@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| _(none — type-only)_ | N/A | wasi_cli_adapter.zig:5819 |

0/0 (N/A). Holds `duration` / `instant` aliases. Needed only so the
P3 `populateWasiProviders` matcher binds the type-only import.

### `wasi:random/random@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-random-bytes` | ✅ | wasi_cli_adapter.zig:5901 |
| `get-random-u64` | ✅ | :5904 |

2/2 (100 %).

### `wasi:random/insecure@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-insecure-random-bytes` | ✅ | wasi_cli_adapter.zig:5920 |
| `get-insecure-random-u64` | ✅ | :5923 |

2/2 (100 %).

### `wasi:random/insecure-seed@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `get-insecure-seed` | ✅ | wasi_cli_adapter.zig:5941 | 0.3 rename of `insecure-seed`. |

1/1 (100 %).

### `wasi:filesystem/preopens@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-directories` | ✅ | wasi_cli_adapter.zig:8148 |

1/1 (100 %).

### `wasi:filesystem/types@0.3.0`

P3 reshapes the 0.2 surface: `read`/`write` are dropped (the
`stream<u8>` variants subsume them); `read-via-stream` /
`write-via-stream` / `append-via-stream` return
`tuple<stream<u8>, future<result<_,error-code>>>`; every other
descriptor method becomes `async func` returning
`future<result<…>>`; `filesystem-error-code` and the
`directory-entry-stream` resource are replaced by
`error-context` (canon-ABI).

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `[method]descriptor.read-via-stream` | ✅ | wasi_cli_adapter.zig:8177 | Host-driver eager-lower; see [PR #577](https://github.com/cataggar/wamr/pull/577). |
| `[method]descriptor.write-via-stream` | ✅ | :8178 | Host-driver pwrite ([PR #577](https://github.com/cataggar/wamr/pull/577)). |
| `[method]descriptor.append-via-stream` | ✅ | :8179 | — |
| `[method]descriptor.read-directory` | ✅ | :8181 | `stream<directory-entry>` driver. |
| `[method]descriptor.advise` | ✅ | :8183 | Async wrapper over 0.2 body. |
| `[method]descriptor.create-directory-at` | ✅ | :8184 | — |
| `[method]descriptor.get-flags` | ✅ | :8185 | — |
| `[method]descriptor.get-type` | ✅ | :8186 | — |
| `[method]descriptor.is-same-object` | ✅ | :8187 | — |
| `[method]descriptor.link-at` | ✅ | :8188 | — |
| `[method]descriptor.metadata-hash` | ✅ | :8189 | — |
| `[method]descriptor.metadata-hash-at` | ✅ | :8190 | — |
| `[method]descriptor.open-at` | ✅ | :8191 | — |
| `[method]descriptor.readlink-at` | ✅ | :8192 | — |
| `[method]descriptor.remove-directory-at` | ✅ | :8193 | — |
| `[method]descriptor.rename-at` | ✅ | :8194 | — |
| `[method]descriptor.set-size` | ✅ | :8195 | — |
| `[method]descriptor.set-times` | ✅ | :8196 | — |
| `[method]descriptor.set-times-at` | ✅ | :8197 | — |
| `[method]descriptor.stat` | ✅ | :8198 | — |
| `[method]descriptor.stat-at` | ✅ | :8199 | — |
| `[method]descriptor.symlink-at` | ✅ | :8200 | — |
| `[method]descriptor.sync` | ✅ | :8201 | — |
| `[method]descriptor.sync-data` | ✅ | :8202 | — |
| `[method]descriptor.unlink-file-at` | ✅ | :8203 | — |
| `[resource-drop]descriptor` | ✅ | :8205 | Carry-over from 0.2 (`fsDescriptorDrop`). |

26/26 (100 %).

### `wasi:sockets/types@0.3.0`

Unified TCP + UDP resource surface ([PR #486](https://github.com/cataggar/wamr/pull/486)
/ [#544](https://github.com/cataggar/wamr/pull/544) / [#565](https://github.com/cataggar/wamr/pull/565)).
Getters/setters are wrapped via `p3SocketWrapper(...)` so the 0.2
bodies serve both surfaces.

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[static]tcp-socket.create` | ✅ | wasi_cli_adapter.zig:15426 |
| `[method]tcp-socket.bind` | ✅ | :15427 |
| `[method]tcp-socket.connect` | ✅ | :15428 |
| `[method]tcp-socket.listen` | ✅ | :15429 |
| `[method]tcp-socket.send` | ✅ | :15430 |
| `[method]tcp-socket.receive` | ✅ | :15431 |
| `[method]tcp-socket.get-local-address` | ✅ | :15433 |
| `[method]tcp-socket.get-remote-address` | ✅ | :15434 |
| `[method]tcp-socket.get-is-listening` | ✅ | :15435 |
| `[method]tcp-socket.get-address-family` | ✅ | :15436 |
| `[method]tcp-socket.set-listen-backlog-size` | ✅ | :15437 |
| `[method]tcp-socket.get-keep-alive-enabled` | ✅ | :15438 |
| `[method]tcp-socket.set-keep-alive-enabled` | ✅ | :15439 |
| `[method]tcp-socket.get-keep-alive-idle-time` | ✅ | :15440 |
| `[method]tcp-socket.set-keep-alive-idle-time` | ✅ | :15441 |
| `[method]tcp-socket.get-keep-alive-interval` | ✅ | :15442 |
| `[method]tcp-socket.set-keep-alive-interval` | ✅ | :15443 |
| `[method]tcp-socket.get-keep-alive-count` | ✅ | :15444 |
| `[method]tcp-socket.set-keep-alive-count` | ✅ | :15445 |
| `[method]tcp-socket.get-hop-limit` | ✅ | :15446 |
| `[method]tcp-socket.set-hop-limit` | ✅ | :15447 |
| `[method]tcp-socket.get-receive-buffer-size` | ✅ | :15448 |
| `[method]tcp-socket.set-receive-buffer-size` | ✅ | :15449 |
| `[method]tcp-socket.get-send-buffer-size` | ✅ | :15450 |
| `[method]tcp-socket.set-send-buffer-size` | ✅ | :15451 |
| `[resource-drop]tcp-socket` | ✅ | :15452 |
| `[static]udp-socket.create` | ✅ | :15454 |
| `[method]udp-socket.bind` | ✅ | :15455 |
| `[method]udp-socket.connect` | ✅ | :15456 |
| `[method]udp-socket.disconnect` | ✅ | :15457 |
| `[method]udp-socket.send` | ✅ | :15458 |
| `[method]udp-socket.receive` | ✅ | :15459 |
| `[method]udp-socket.get-local-address` | ✅ | :15461 |
| `[method]udp-socket.get-remote-address` | ✅ | :15462 |
| `[method]udp-socket.get-address-family` | ✅ | :15463 |
| `[method]udp-socket.get-unicast-hop-limit` | ✅ | :15464 |
| `[method]udp-socket.set-unicast-hop-limit` | ✅ | :15465 |
| `[method]udp-socket.get-receive-buffer-size` | ✅ | :15466 |
| `[method]udp-socket.set-receive-buffer-size` | ✅ | :15467 |
| `[method]udp-socket.get-send-buffer-size` | ✅ | :15468 |
| `[method]udp-socket.set-send-buffer-size` | ✅ | :15469 |
| `[resource-drop]udp-socket` | ✅ | :15470 |

42/42 (100 %). Upstream `wasi-sockets` does not yet publish an
`@0.3.0` package; wamr's 0.3 surface tracks the in-progress draft
(matches the layout used by `wasm32-wasip3` Rust bindgen).

### `wasi:sockets/ip-name-lookup@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `resolve-addresses` (free fn) | ✅ | wasi_cli_adapter.zig:15493 | `std.net.HostName.lookup`-backed, allow-list-gated; settles a `future<result<list<ip-address>, error-code>>` via `socketReadyResultFuture` / `spawnReadyFutureBytes`. |

1/1 (100 %).

### `wasi:http/types@0.3.0`

Unified `request` / `response` resource ([PR #487](https://github.com/cataggar/wamr/pull/487)
/ [#568](https://github.com/cataggar/wamr/pull/568)).

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[constructor]fields` | ✅ | wasi_cli_adapter.zig:20280 |
| `[static]fields.from-list` | ✅ | :20281 |
| `[method]fields.get` | ✅ | :20282 |
| `[method]fields.has` | ✅ | :20283 |
| `[method]fields.set` | ✅ | :20284 |
| `[method]fields.delete` | ✅ | :20285 |
| `[method]fields.get-and-delete` | ✅ | :20286 |
| `[method]fields.append` | ✅ | :20287 |
| `[method]fields.copy-all` | ✅ | :20288 |
| `[method]fields.clone` | ✅ | :20289 |
| `[resource-drop]fields` | ✅ | :20290 |
| `[static]request.new` | ✅ | :20292 |
| `[method]request.get-method` | ✅ | :20293 |
| `[method]request.set-method` | ✅ | :20294 |
| `[method]request.get-path-with-query` | ✅ | :20295 |
| `[method]request.set-path-with-query` | ✅ | :20296 |
| `[method]request.get-scheme` | ✅ | :20297 |
| `[method]request.set-scheme` | ✅ | :20298 |
| `[method]request.get-authority` | ✅ | :20299 |
| `[method]request.set-authority` | ✅ | :20300 |
| `[method]request.get-options` | ✅ | :20301 |
| `[method]request.get-headers` | ✅ | :20302 |
| `[static]request.consume-body` | ✅ | :20303 |
| `[resource-drop]request` | ✅ | :20304 |
| `[constructor]request-options` | ✅ | :20306 |
| `[method]request-options.get-connect-timeout` | ✅ | :20307 | Stored in nanoseconds, snapshotted when the request is constructed, and applied to initial DNS/TCP acquisition. The child options resource may be dropped before send. Zig's lazy TLS handshake and automatic redirect reconnects are not deadline-covered. |
| `[method]request-options.set-connect-timeout` | ✅ | :20308 | — |
| `[method]request-options.get-first-byte-timeout` | ✅ | :20309 | Round-trip only; deadline-aware HTTP/TLS reader support remains #616 A1b/A7 work. |
| `[method]request-options.set-first-byte-timeout` | ✅ | :20310 | — |
| `[method]request-options.get-between-bytes-timeout` | ✅ | :20311 | Round-trip only; deadline-aware HTTP/TLS reader support remains #616 A1b/A7 work. |
| `[method]request-options.set-between-bytes-timeout` | ✅ | :20312 | — |
| `[method]request-options.clone` | ✅ | :20313 |
| `[resource-drop]request-options` | ✅ | :20314 |
| `[static]response.new` | ✅ | :20316 |
| `[method]response.get-status-code` | ✅ | :20317 |
| `[method]response.set-status-code` | ✅ | :20318 |
| `[method]response.get-headers` | ✅ | :20319 |
| `[static]response.consume-body` | ✅ | :20320 |
| `[resource-drop]response` | ✅ | :20321 |

39/39 (100 %). All six `request-options` timeout accessors plus
`clone` are bound — the 0.2 gap above is closed in 0.3.

### `wasi:http/handler@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `handle` | ✅ | wasi_cli_adapter.zig:20339 |

1/1 (100 %). Incoming-handler trampoline; delegates to
`httpClientSendP3` ([PR #549](https://github.com/cataggar/wamr/pull/549)
/ [#580](https://github.com/cataggar/wamr/pull/580)).

### `wasi:http/client@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `send` | ✅ | wasi_cli_adapter.zig:20353 |

1/1 (100 %). Outbound async state machine; HTTP + HTTPS via
`std.http.Client` + `std.crypto.tls` ([PR #583 A2](https://github.com/cataggar/wamr/issues/583)
/ [#590](https://github.com/cataggar/wamr/pull/590)).

## Cross-version (no version-multiplex split)

### `wasi:keyvalue/store@0.2.0-draft2`

Upstream WIT:
[`store.wit`](../docs/wasi-keyvalue-wit-vendored/store.wit) (vendored).

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `open` (free fn) | ✅ | wasi_cli_adapter.zig:20508 |
| `[method]bucket.get` | ✅ | :20509 |
| `[method]bucket.set` | ✅ | :20510 |
| `[method]bucket.delete` | ✅ | :20511 |
| `[method]bucket.exists` | ✅ | :20512 |
| `[method]bucket.list-keys` | ✅ | :20513 |
| `[resource-drop]bucket` | ✅ | :20514 |

7/7 (100 %). In-memory `std.StringHashMapUnmanaged`-backed; no
disk persistence, no replication.

### `wasi:keyvalue/atomics@0.2.0-draft2`

Upstream WIT:
[`atomic.wit`](../docs/wasi-keyvalue-wit-vendored/atomic.wit) (vendored).

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `increment` (free fn) | ✅ | wasi_cli_adapter.zig:20545 | Real arithmetic over the bucket map ([PR #583 B4](https://github.com/cataggar/wamr/issues/583)). |
| `swap` (free fn) | ⚠️ | :20546 | Returns `cas-error::store-error(error::other("…"))` — stub. |
| `[static]cas.new` | ⚠️ | :20547 | Returns `error::other("…")` — stub. |
| `[method]cas.current` | ⚠️ | :20548 | Returns `error::other("…")` — stub. Unreachable in practice because `cas.new` never produces a handle. |
| `[resource-drop]cas` | ✅ | :20549 | No-op (no live CAS handle ever exists). |

5/5 registered (3 stubs). Implemented-or-N/A = 2/5 (40 %); stubs
explicitly tracked under [#583 B4 follow-up #602](https://github.com/cataggar/wamr/issues/583).

### `wasi:keyvalue/batch@0.2.0-draft2`

Upstream WIT:
[`batch.wit`](../docs/wasi-keyvalue-wit-vendored/batch.wit) (vendored).

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-many` (free fn) | ✅ | wasi_cli_adapter.zig:20575 |
| `set-many` (free fn) | ✅ | :20576 |
| `delete-many` (free fn) | ✅ | :20577 |

3/3 (100 %).

### `wasi:logging/logging@0.1.0-draft`

Upstream WIT: [`logging.wit`](https://github.com/WebAssembly/wasi-logging/blob/main/wit/logging.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `log` (free fn) | ✅ | wasi_cli_adapter.zig:6109 | Routes to host stderr + `std.log.scoped(.wasi_guest)`; level filter via `--log-level` / `WAMR_LOG_LEVEL`. |

1/1 (100 %). The `_p2` slot (`populateWasiLoggingP2`, line 6122)
is reserved for a future `@0.2.x` revision and currently mirrors
the same `wasiLog` callback.

### `wasi:config/store@0.2.0-rc.1`

Upstream WIT: [`store.wit`](https://github.com/WebAssembly/wasi-config/blob/main/wit/store.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `get` (free fn) | ✅ | wasi_cli_adapter.zig:21395 | Layered: `--config-store=PATH.json` overrides `WAMR_CONFIG_<KEY>` env. |
| `get-all` (free fn) | ✅ | :21398 | Same layering. |

2/2 (100 %). The `error::upstream` / `error::io` arms are
reserved for future Vault / Kubernetes / etc. backends; the
in-memory store never surfaces them.

## Footer

Re-run this audit against any future `main` SHA by:

1. Updating the SHA + date at the top of this file.
2. Running the [methodology](#methodology) extractor against
   `wasi_cli_adapter.zig`.
3. Diffing the extracted `(L<line>, member-name)` rows against the
   tables here; add/remove rows as needed.
4. Re-fetching the upstream WIT files (links in each section) and
   diffing `func` / `[method]` / `[static]` declarations against
   the registered rows. Any upstream addition with no matching
   row → ❌ Missing; any new wamr stub → ⚠️ Stub.

The previous audit lived at `fa850da4` (this document).
