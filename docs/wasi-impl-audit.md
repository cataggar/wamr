# WASI WIT-vs-impl audit

| Field         | Value                                                  |
| ------------- | ------------------------------------------------------ |
| Origin commit | [`b8e6e566`](https://github.com/cataggar/wamr/commit/b8e6e566526c26deeebd12b9cfe103ae6142b0ed) |
| Audit date    | 2026-07-15                                             |
| Adapter file  | [`src/component/wasi_cli_adapter.zig`](../src/component/wasi_cli_adapter.zig) |
| WIT/test pin  | [`wasi-testsuite@7d0a7811`](https://github.com/WebAssembly/wasi-testsuite/commit/7d0a78116fa9955dd2d113beb8583bc9648b6116) |
| Tracker       | [#616 C1](https://github.com/cataggar/wamr/issues/616) (quarterly WIT-vs-implementation audit) |
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
- [Preview 2 (0.2.6) interfaces](#preview-2-026-interfaces)
- [Preview 3 (0.3.0) interfaces](#preview-3-030-interfaces)
- [Cross-version (no version-multiplex split)](#cross-version-no-version-multiplex-split)

## Methodology

The audit is **reproducible** — every column can be regenerated from
`main` with the steps below. Future audits should re-run the same
grep / cross-reference and update this file.

### Enumerate registered interfaces

```console
$ awk '
    /^pub fn populateWasiProviders\(/ { in_fn = 1 }
    in_fn && /try adapter\.populateWasi/ { print NR ":" $0 }
    in_fn && /^}$/ { exit }
  ' src/component/wasi_cli_adapter.zig
```

Each call inside the top-level `populateWasiProviders`
([`wasi_cli_adapter.zig:23994`](../src/component/wasi_cli_adapter.zig#L23994))
is the registration seam for one WIT interface (or one
version-multiplexed group of interfaces, e.g. `wasi:cli/exit` 0.2 +
0.3). The command prints 45 top-level populator calls at this commit.

### Enumerate registered methods per interface

Each `populateWasi*` body holds either an inline list of
`.members.put(...)` calls or a `[_]M{...}` table whose entries are
`.{ .name = "<wit-member-name>", .call = &<host-fn> }`. The
following script scans balanced function bodies and counts both
patterns (including registrations wrapped by `tracedAdapterCall`):

```python
import re
from pathlib import Path

src = Path("src/component/wasi_cli_adapter.zig").read_text()
total = populators = 0
for match in re.finditer(r"^\s*pub fn (populateWasi\w+)\s*\(", src, re.M):
    begin = src.find("{", match.end())
    depth, end = 1, begin + 1
    while depth:
        depth += (src[end] == "{") - (src[end] == "}")
        end += 1
    body, names = src[begin + 1:end - 1], []
    for pattern in (
        r'members\.put\(\s*[^,]+,\s*"([^"]+)"',
        r'\.name\s*=\s*"([^"]+)"',
    ):
        for member in re.finditer(pattern, body, re.S):
            if member.group(1) not in names:
                names.append(member.group(1))
    if names:
        print(f"{match.group(1)}\t{len(names)}")
        total += len(names)
        populators += 1
print(f"populators={populators} registered-member-rows={total}")
```

At this audit: `populators=48 registered-member-rows=356`. The stable
host-import denominator is 352. Four registered rows do not enter it:
the reserved unpublished `wasi:logging@0.2.x` duplicate, the unstable
P2 `cli.exit-with-code` and `sockets.network-error-code` methods, and
the synthetic P2 `http/incoming-handler.handle` provider for an
interface the pinned world exports. The source registers 30 P2 named
instances, of which 29 contain stable host imports, and 26 P3
instances (135 stable host methods plus four zero-method/type-only
instances).

### Cross-reference against the pinned WIT/build inputs

The denominator is version-pinned; links to a moving upstream `main`
branch are not inputs:

* **P3:** the six exact `package.wit` files under
  [`tests/wasi-testsuite/tests/rust/wasm32-wasip3/wit/deps`](../tests/wasi-testsuite/tests/rust/wasm32-wasip3/wit/deps)
  at submodule commit `7d0a7811`. They contain 139 declarations:
  135 host-import methods, the guest-exported `wasi:cli/run.run`, and
  three `@unstable(feature = clocks-timezone)` methods.
* **P2:** main's `wasip2` build dependency is pinned in
  [`build.zig.zon`](../build.zig.zon) to commit `de1b26f7`; its
  generated bindings import `@0.2.6`. The complete denominator is
  checked against the immutable `v0.2.6` WIT commits for
  [`wasi-cli`](https://github.com/WebAssembly/wasi-cli/tree/939bd6d492c11bbd7d3c349b91096061022bc3d7/wit),
  [`wasi-io`](https://github.com/WebAssembly/wasi-io/tree/176892a2b6abfb63c2608aefb7cf92558dee530d/wit),
  [`wasi-clocks`](https://github.com/WebAssembly/wasi-clocks/tree/13d1c82efd6287bb497bca55b8ef994e0f2c338c/wit),
  [`wasi-random`](https://github.com/WebAssembly/wasi-random/tree/4e946631f1665364202c67a04b11544d6a9bfe60/wit),
  [`wasi-filesystem`](https://github.com/WebAssembly/wasi-filesystem/tree/e2a2ddc6cdcd9093d4bb5db0c44eb93d4c9c013b/wit),
  [`wasi-sockets`](https://github.com/WebAssembly/wasi-sockets/tree/bb247e2827e3207d82520f737769265048d17111/wit),
  and [`wasi-http`](https://github.com/WebAssembly/wasi-http/tree/d97efe470e4d1e3f959162d81eef9666335bc186/wit).
  The final P3 fixture binaries also embed reduced `@0.2.4` adapter
  worlds; those are fixture dependencies, not a complete P2 method
  inventory. No complete P2 WIT package is vendored locally, so this
  immutable 0.2.6 provenance is stated explicitly rather than
  presenting the moving upstream links as vendored inputs. Unstable
  declarations are excluded consistently, and `proxy.wit` world
  direction determines whether HTTP handlers are host imports or
  guest exports. Across those pinned packages, 206 relevant method
  declarations split into 199 stable host imports, five unstable
  methods, and two guest exports.
* **Other registered packages:** `wasi:keyvalue@0.2.0-draft2` uses the
  in-tree [`docs/wasi-keyvalue-wit-vendored`](wasi-keyvalue-wit-vendored)
  copy. Logging and config have no vendored WIT or fixture corpus in
  this tree; their one- and two-method denominators are therefore
  checked against their published package shapes and called out as a
  provenance limitation.

The comparison counts every WIT `func`, `static func`,
`constructor`, resource method, and canonical resource drop.
Resource-drop entries are required by the canonical ABI for every
`resource <X>` and are counted as one method per resource.

### Classify each method

* **✅ Implemented** — the host function performs real work
  (syscalls, library calls, in-memory state mutation).
* **⚠️ Stub** — registered and links cleanly, but unconditionally
  returns a canned result, performs no work, or substitutes behavior
  that does not implement the WIT contract. Capability-gated methods
  with a real enabled path are not stubs merely because the default
  profile denies access.
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
| Preview 2 (0.2.6) — `wasi:cli` + `wasi:io` + `wasi:clocks` + `wasi:random` + `wasi:filesystem` + `wasi:sockets` + `wasi:http` | 199 | 198 | 1 | 0 | 99.5 % |
| Preview 3 (0.3.0) | 135 | 135 | — | 0 | 100.0 % |
| `wasi:keyvalue@0.2.0-draft2` | 15 | 15 | — | 0 | 100.0 % |
| `wasi:logging@0.1.0-draft` | 1 | 1 | — | — | 100.0 % |
| `wasi:config/store@0.2.0-rc.1` | 2 | 2 | — | — | 100.0 % |
| **Total** | **352** | **351** | **1** | **0** | **99.7 %** |

Stubbed: 0.3 %. Missing: 0.0 %. Type-only P3 instances
(`wasi:cli/types`, `wasi:clocks/types`, plus the adapter's synthetic
`wasi:io/streams` / `wasi:io/error` instances) contribute zero rows.
The unstable P2 `cli.exit-with-code`,
`sockets.network-error-code`,
`response-outparam.send-informational`, and
`clocks/timezone.{display,utc-offset}` methods are excluded, as are
the three unstable P3 timezone methods. Guest exports
P2 `cli/run.run`, P2 `http/incoming-handler.handle`, and P3
`cli/run.run` are also outside the host-import denominator.

The final testsuite pin passes **41/41** under P3 AOT, **41/41** under
P3 JIT, and **41/41** under Wasmtime parity. Those fixture results
exercise only methods imported by those 41 binaries; they do **not**
establish 351/352 method implementation coverage.

**Changes found by this refresh:**

* The old 188-method P2 summary undercounted the stable host-import
  surface by 11. The first refresh then overcorrected to 202 by
  including two unstable methods and one guest export; the consistent
  denominator is 199.
* The three `wasi:keyvalue/atomics` CAS arms are real implementations
  since PR #608 and move from ⚠️ to ✅.
* P2 `wasi:http/incoming-handler.handle` is a guest export in the
  pinned proxy world, so its synthetic no-op provider is N/A.
* P3 `wasi:http/handler.handle` is a stable import in the middleware
  world. Forwarding through the real outbound HTTP(S) client is
  permitted by its WIT contract, so it remains ✅.
* P2 `wasi:filesystem/types.filesystem-error-code` ignores its
  borrowed error and always returns `option::none`, so it is ⚠️.

## Findings: ❌ Missing arms

None. The immutable 0.2.6 comparison reports 199/199 stable
host-import names registered; the vendored 0.3.0 comparison reports
135/135 host-import names registered. The only unregistered
declarations encountered are explicitly excluded above (unstable
features or a guest export).

## Findings: ⚠️ Stubbed arms

| WIT method | Stub behaviour | Source |
| --- | --- | --- |
| `wasi:filesystem/types@0.2.6.filesystem-error-code` | Ignores the borrowed `io/error` and unconditionally returns `option::none`; no filesystem error provenance is retained to downcast. | [`wasi_cli_adapter.zig:11271`](../src/component/wasi_cli_adapter.zig#L11271) |

The CAS resource is no longer stubbed: `cas.new` snapshots,
`cas.current` returns the snapshot, and `swap` performs a real
conditional write with retry-handle semantics.

The captured-buffer profile makes every `get-terminal-*` (both 0.2
and 0.3) return `none`. That is the spec-conformant answer when
stdio is not a TTY, so it is classified ✅ Implemented, not ⚠️ Stub.

## Preview 2 (0.2.6) interfaces

### `wasi:cli/stdout` & `wasi:io/streams`

Pinned WIT: [`stdio.wit`](https://github.com/WebAssembly/wasi-cli/blob/939bd6d492c11bbd7d3c349b91096061022bc3d7/wit/stdio.wit),
[`streams.wit`](https://github.com/WebAssembly/wasi-io/blob/176892a2b6abfb63c2608aefb7cf92558dee530d/wit/streams.wit)

Registered together by
[`populateWasiCliRun`](../src/component/wasi_cli_adapter.zig).

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `get-stdout` | ✅ | wasi_cli_adapter.zig:5166 | Returns captured stdout sink handle. |
| `[method]output-stream.blocking-write-and-flush` | ✅ | :5173 | — |
| `[method]output-stream.write` | ✅ | :5178 | — |
| `[method]output-stream.check-write` | ✅ | :5183 | — |
| `[method]output-stream.blocking-flush` | ✅ | :5188 | — |
| `[method]output-stream.flush` | ✅ | :5193 | Aliased to `blocking-flush`. |
| `[method]output-stream.subscribe` | ✅ | :5198 | — |
| `[method]output-stream.write-zeroes` | ✅ | :5235 | Audit fill-in (#583, PR #604). |
| `[method]output-stream.blocking-write-zeroes-and-flush` | ✅ | :5240 | Audit fill-in (#583, PR #604). |
| `[method]output-stream.splice` | ✅ | :5245 | Nonblocking Linux pipe-to-pipe `splice(2)`; buffer fallback otherwise. |
| `[method]output-stream.blocking-splice` | ✅ | :5250 | Blocking Linux `splice(2)` with `EAGAIN` readiness wait/retry; buffer fallback otherwise. |
| `[resource-drop]output-stream` | ✅ | :5208 | — |
| `[method]input-stream.subscribe` | ✅ | :5203 | — |
| `[method]input-stream.read` | ✅ | :5219 | Aliased to `blocking-read`. |
| `[method]input-stream.blocking-read` | ✅ | :5214 | — |
| `[method]input-stream.skip` | ✅ | :5225 | Audit fill-in (#583, PR #604). |
| `[method]input-stream.blocking-skip` | ✅ | :5230 | Same host helper as `skip`. |
| `[resource-drop]input-stream` | ✅ | :5255 | — |

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
| `get-stderr` | ✅ | wasi_cli_adapter.zig:5272 |

1/1 (100 %).

### `wasi:cli/stdin`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-stdin` | ✅ | wasi_cli_adapter.zig:5375 |

1/1 (100 %).

### `wasi:cli/exit`

Pinned WIT: [`exit.wit`](https://github.com/WebAssembly/wasi-cli/blob/939bd6d492c11bbd7d3c349b91096061022bc3d7/wit/exit.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `exit` | ✅ | wasi_cli_adapter.zig:5288 | Sets adapter exit code; raises `error.Trap`. |
| `exit-with-code` (`@unstable`) | N/A | :5291 | Registered without a feature gate, but excluded from the stable denominator. |

1/1 stable host method implemented (100 %); the unstable method is
registered but excluded.

### `wasi:cli/environment`

Pinned WIT: [`environment.wit`](https://github.com/WebAssembly/wasi-cli/blob/939bd6d492c11bbd7d3c349b91096061022bc3d7/wit/environment.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-environment` | ✅ | wasi_cli_adapter.zig:5307 |
| `get-arguments` | ✅ | :5310 |
| `initial-cwd` | ✅ | :5313 |

3/3 (100 %).

### `wasi:cli/run`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `run` (guest export) | N/A | — |

Pinned [`command.wit`](https://github.com/WebAssembly/wasi-cli/blob/939bd6d492c11bbd7d3c349b91096061022bc3d7/wit/command.wit)
exports `run`; it is not a host import and does not enter the
implementation denominator.

### `wasi:cli/terminal-{stdin,stdout,stderr,input,output}`

Pinned WIT: [`terminal.wit`](https://github.com/WebAssembly/wasi-cli/blob/939bd6d492c11bbd7d3c349b91096061022bc3d7/wit/terminal.wit)

Captured-buffer mode — `get-terminal-*` returns `none` because
stdio is not a TTY (spec-conformant).

| Interface | WIT method | Status | Adapter location |
| --- | --- | :-: | --- |
| `terminal-stdin` | `get-terminal-stdin` | ✅ | :5335 |
| `terminal-stdout` | `get-terminal-stdout` | ✅ | :5341 |
| `terminal-stderr` | `get-terminal-stderr` | ✅ | :5347 |
| `terminal-input` | `[resource-drop]terminal-input` | ✅ | :5353 |
| `terminal-output` | `[resource-drop]terminal-output` | ✅ | :5359 |

5/5 (100 %). `terminal-input` / `terminal-output` are resources
with **no** WIT methods upstream (just the bare resource), so only
the canonical-ABI-required `[resource-drop]` is registered.

### `wasi:io/poll`

Pinned WIT: [`poll.wit`](https://github.com/WebAssembly/wasi-io/blob/176892a2b6abfb63c2608aefb7cf92558dee530d/wit/poll.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[method]pollable.ready` | ✅ | wasi_cli_adapter.zig:6255 |
| `[method]pollable.block` | ✅ | :6258 |
| `poll` (free fn) | ✅ | :6261 |
| `[resource-drop]pollable` | ✅ | :6264 |

4/4 (100 %).

### `wasi:io/error`

Pinned WIT: [`error.wit`](https://github.com/WebAssembly/wasi-io/blob/176892a2b6abfb63c2608aefb7cf92558dee530d/wit/error.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `[method]error.to-debug-string` | ✅ | :6278 | Audit fill-in (#583, PR #604). wamr does not track io-error provenance — returns an opaque `"wasi:io error (opaque host handle #N)"` description. |
| `[resource-drop]error` | ✅ | :6271 | — |

2/2 (100 %). The `to-debug-string` method was flipped from ❌ to ✅
in PR #604's follow-up — the host returns a best-effort opaque
description (handle-suffixed) since wamr does not currently keep an
io-error table.

### `wasi:clocks/wall-clock`

Pinned WIT: [`wall-clock.wit`](https://github.com/WebAssembly/wasi-clocks/blob/13d1c82efd6287bb497bca55b8ef994e0f2c338c/wit/wall-clock.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `now` | ✅ | wasi_cli_adapter.zig:6321 |
| `resolution` | ✅ | :6324 |

2/2 (100 %).

### `wasi:clocks/monotonic-clock`

Pinned WIT: [`monotonic-clock.wit`](https://github.com/WebAssembly/wasi-clocks/blob/13d1c82efd6287bb497bca55b8ef994e0f2c338c/wit/monotonic-clock.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `now` | ✅ | wasi_cli_adapter.zig:6347 |
| `resolution` | ✅ | :6350 |
| `subscribe-instant` | ✅ | :6353 |
| `subscribe-duration` | ✅ | :6356 |

4/4 (100 %).

### `wasi:clocks/timezone`

Pinned WIT: [`timezone.wit`](https://github.com/WebAssembly/wasi-clocks/blob/13d1c82efd6287bb497bca55b8ef994e0f2c338c/wit/timezone.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `display` (`@unstable`) | N/A | — | The entire interface and method are gated by `clocks-timezone`; not registered. |
| `utc-offset` (`@unstable`) | N/A | — | Same exclusion. |

0/0 stable host methods. Both unstable methods are excluded.

### `wasi:random/random`

Pinned WIT: [`random.wit`](https://github.com/WebAssembly/wasi-random/blob/4e946631f1665364202c67a04b11544d6a9bfe60/wit/random.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-random-bytes` | ✅ | wasi_cli_adapter.zig:6461 |
| `get-random-u64` | ✅ | :6464 |

2/2 (100 %).

### `wasi:random/insecure`

Pinned WIT: [`insecure.wit`](https://github.com/WebAssembly/wasi-random/blob/4e946631f1665364202c67a04b11544d6a9bfe60/wit/insecure.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-insecure-random-bytes` | ✅ | wasi_cli_adapter.zig:6481 |
| `get-insecure-random-u64` | ✅ | :6484 |

2/2 (100 %).

### `wasi:random/insecure-seed`

Pinned WIT: [`insecure-seed.wit`](https://github.com/WebAssembly/wasi-random/blob/4e946631f1665364202c67a04b11544d6a9bfe60/wit/insecure-seed.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `insecure-seed` | ✅ | wasi_cli_adapter.zig:6501 |

1/1 (100 %).

### `wasi:filesystem/preopens`

Pinned WIT: [`preopens.wit`](https://github.com/WebAssembly/wasi-filesystem/blob/e2a2ddc6cdcd9093d4bb5db0c44eb93d4c9c013b/wit/preopens.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-directories` | ✅ | wasi_cli_adapter.zig:9096 |

1/1 (100 %).

### `wasi:filesystem/types`

Pinned WIT: [`types.wit`](https://github.com/WebAssembly/wasi-filesystem/blob/e2a2ddc6cdcd9093d4bb5db0c44eb93d4c9c013b/wit/types.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[method]descriptor.get-type` | ✅ | wasi_cli_adapter.zig:9115 |
| `[method]descriptor.get-flags` | ✅ | :9116 |
| `[method]descriptor.stat` | ✅ | :9117 |
| `[method]descriptor.stat-at` | ✅ | :9118 |
| `[method]descriptor.set-times` | ✅ | :9119 |
| `[method]descriptor.set-times-at` | ✅ | :9120 |
| `[method]descriptor.open-at` | ✅ | :9121 |
| `[method]descriptor.read-via-stream` | ✅ | :9122 |
| `[method]descriptor.write-via-stream` | ✅ | :9123 |
| `[method]descriptor.append-via-stream` | ✅ | :9124 |
| `[method]descriptor.read` | ✅ | :9125 |
| `[method]descriptor.write` | ✅ | :9126 |
| `[method]descriptor.sync` | ✅ | :9127 |
| `[method]descriptor.sync-data` | ✅ | :9128 |
| `[method]descriptor.set-size` | ✅ | :9129 |
| `[method]descriptor.advise` | ✅ | :9130 |
| `[method]descriptor.is-same-object` | ✅ | :9131 |
| `[method]descriptor.metadata-hash` | ✅ | :9132 |
| `[method]descriptor.metadata-hash-at` | ✅ | :9133 |
| `[method]descriptor.create-directory-at` | ✅ | :9135 |
| `[method]descriptor.unlink-file-at` | ✅ | :9136 |
| `[method]descriptor.remove-directory-at` | ✅ | :9137 |
| `[method]descriptor.rename-at` | ✅ | :9138 |
| `[method]descriptor.link-at` | ✅ | :9139 |
| `[method]descriptor.symlink-at` | ✅ | :9140 |
| `[method]descriptor.readlink-at` | ✅ | :9141 |
| `[method]descriptor.read-directory` | ✅ | :9143 |
| `[method]directory-entry-stream.read-directory-entry` | ✅ | :9144 |
| `[resource-drop]directory-entry-stream` | ✅ | :9145 |
| `filesystem-error-code` (free fn) | ⚠️ | :9146 |
| `[resource-drop]descriptor` | ✅ | :9147 |

30/31 implemented (96.8 %); 1/31 stubbed. All 27 descriptor methods,
both `directory-entry-stream` members, and both resource-drops are
implemented. The free `filesystem-error-code` downcast is registered
but always returns `option::none`.

### `wasi:sockets/network`

Pinned WIT: [`network.wit`](https://github.com/WebAssembly/wasi-sockets/blob/bb247e2827e3207d82520f737769265048d17111/wit/network.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `network-error-code` (`@unstable`, free fn) | N/A | :14879 | Registered, but excluded from the stable denominator. It always returns `option::none` because wamr's typed sockets errors do not retain `io-error` provenance. |
| `[resource-drop]network` | ✅ | :14876 | — |

1/1 stable host method implemented (100 %). The unstable free
function is registered but excluded.

### `wasi:sockets/instance-network`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `instance-network` | ✅ | wasi_cli_adapter.zig:14893 |

1/1 (100 %).

### `wasi:sockets/tcp`

Pinned WIT: [`tcp.wit`](https://github.com/WebAssembly/wasi-sockets/blob/bb247e2827e3207d82520f737769265048d17111/wit/tcp.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[method]tcp-socket.start-bind` | ✅ | wasi_cli_adapter.zig:14915 |
| `[method]tcp-socket.finish-bind` | ✅ | :14916 |
| `[method]tcp-socket.start-connect` | ✅ | :14917 |
| `[method]tcp-socket.finish-connect` | ✅ | :14918 |
| `[method]tcp-socket.start-listen` | ✅ | :14919 |
| `[method]tcp-socket.finish-listen` | ✅ | :14920 |
| `[method]tcp-socket.accept` | ✅ | :14921 |
| `[method]tcp-socket.local-address` | ✅ | :14922 |
| `[method]tcp-socket.remote-address` | ✅ | :14923 |
| `[method]tcp-socket.shutdown` | ✅ | :14924 |
| `[method]tcp-socket.set-listen-backlog-size` | ✅ | :14926 |
| `[method]tcp-socket.set-keep-alive-enabled` | ✅ | :14927 |
| `[method]tcp-socket.set-keep-alive-idle-time` | ✅ | :14928 |
| `[method]tcp-socket.set-keep-alive-interval` | ✅ | :14929 |
| `[method]tcp-socket.set-keep-alive-count` | ✅ | :14930 |
| `[method]tcp-socket.set-hop-limit` | ✅ | :14931 |
| `[method]tcp-socket.set-receive-buffer-size` | ✅ | :14932 |
| `[method]tcp-socket.set-send-buffer-size` | ✅ | :14933 |
| `[method]tcp-socket.keep-alive-enabled` | ✅ | :14935 |
| `[method]tcp-socket.keep-alive-idle-time` | ✅ | :14936 |
| `[method]tcp-socket.keep-alive-interval` | ✅ | :14937 |
| `[method]tcp-socket.keep-alive-count` | ✅ | :14938 |
| `[method]tcp-socket.hop-limit` | ✅ | :14939 |
| `[method]tcp-socket.receive-buffer-size` | ✅ | :14940 |
| `[method]tcp-socket.send-buffer-size` | ✅ | :14941 |
| `[method]tcp-socket.address-family` | ✅ | :14943 |
| `[method]tcp-socket.is-listening` | ✅ | :14944 |
| `[method]tcp-socket.subscribe` | ✅ | :14946 |
| `[resource-drop]tcp-socket` | ✅ | :14947 |

29/29 (100 %). Allow-list-gated; real
`bind` / `connect` / `listen` / `accept`; SO_REUSEADDR; POSIX +
Windows parity (`bindAndGetsockname` shim, [PR #587](https://github.com/cataggar/wamr/pull/587)).

### `wasi:sockets/tcp-create-socket`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `create-tcp-socket` | ✅ | wasi_cli_adapter.zig:14965 |

1/1 (100 %).

### `wasi:sockets/udp`

Pinned WIT: [`udp.wit`](https://github.com/WebAssembly/wasi-sockets/blob/bb247e2827e3207d82520f737769265048d17111/wit/udp.wit)

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[method]udp-socket.start-bind` | ✅ | wasi_cli_adapter.zig:14985 |
| `[method]udp-socket.finish-bind` | ✅ | :14986 |
| `[method]udp-socket.stream` | ✅ | :14987 |
| `[method]udp-socket.local-address` | ✅ | :14988 |
| `[method]udp-socket.remote-address` | ✅ | :14989 |
| `[method]udp-socket.unicast-hop-limit` | ✅ | :14991 |
| `[method]udp-socket.set-unicast-hop-limit` | ✅ | :14992 |
| `[method]udp-socket.receive-buffer-size` | ✅ | :14993 |
| `[method]udp-socket.set-receive-buffer-size` | ✅ | :14994 |
| `[method]udp-socket.send-buffer-size` | ✅ | :14995 |
| `[method]udp-socket.set-send-buffer-size` | ✅ | :14996 |
| `[method]udp-socket.address-family` | ✅ | :14998 |
| `[method]udp-socket.subscribe` | ✅ | :14999 |
| `[resource-drop]udp-socket` | ✅ | :15000 |
| `[resource-drop]incoming-datagram-stream` | ✅ | :15002 |
| `[method]incoming-datagram-stream.receive` | ✅ | :15003 |
| `[method]incoming-datagram-stream.subscribe` | ✅ | :15004 |
| `[resource-drop]outgoing-datagram-stream` | ✅ | :15005 |
| `[method]outgoing-datagram-stream.check-send` | ✅ | :15006 |
| `[method]outgoing-datagram-stream.send` | ✅ | :15007 |
| `[method]outgoing-datagram-stream.subscribe` | ✅ | :15008 |

21/21 (100 %).

### `wasi:sockets/udp-create-socket`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `create-udp-socket` | ✅ | wasi_cli_adapter.zig:15026 |

1/1 (100 %).

### `wasi:sockets/ip-name-lookup`

Pinned WIT: [`ip-name-lookup.wit`](https://github.com/WebAssembly/wasi-sockets/blob/bb247e2827e3207d82520f737769265048d17111/wit/ip-name-lookup.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `resolve-addresses` (free fn) | ✅ | wasi_cli_adapter.zig:15043 | `std.net.getAddressList`-backed; allow-list-gated. |
| `[method]resolve-address-stream.resolve-next-address` | ✅ | :15046 | — |
| `[method]resolve-address-stream.subscribe` | ✅ | :15049 | — |
| `[resource-drop]resolve-address-stream` | ✅ | :15052 | — |

4/4 (100 %).

### `wasi:http/types`

Pinned WIT: [`types.wit`](https://github.com/WebAssembly/wasi-http/blob/d97efe470e4d1e3f959162d81eef9666335bc186/wit/types.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `http-error-code` (free fn) | ✅ | wasi_cli_adapter.zig:22083 | Cross-references the borrowed `io-error` handle against `WasiCliAdapter.http_io_errors`; returns `some(error-code)` for HTTP-origin errors, `none` otherwise. |
| `[constructor]fields` | ✅ | wasi_cli_adapter.zig:22012 | — |
| `[static]fields.from-list` | ✅ | :22013 | — |
| `[method]fields.entries` | ✅ | :22014 | — |
| `[method]fields.get` | ✅ | :22015 | — |
| `[method]fields.has` | ✅ | :22016 | — |
| `[method]fields.set` | ✅ | :22017 | — |
| `[method]fields.append` | ✅ | :22018 | — |
| `[method]fields.delete` | ✅ | :22019 | — |
| `[method]fields.clone` | ✅ | :22020 | — |
| `[resource-drop]fields` | ✅ | :22021 | — |
| `[constructor]outgoing-request` | ✅ | :22023 | — |
| `[method]outgoing-request.method` | ✅ | :22024 | — |
| `[method]outgoing-request.set-method` | ✅ | :22025 | — |
| `[method]outgoing-request.path-with-query` | ✅ | :22026 | — |
| `[method]outgoing-request.set-path-with-query` | ✅ | :22027 | — |
| `[method]outgoing-request.scheme` | ✅ | :22028 | — |
| `[method]outgoing-request.set-scheme` | ✅ | :22029 | — |
| `[method]outgoing-request.authority` | ✅ | :22030 | — |
| `[method]outgoing-request.set-authority` | ✅ | :22031 | — |
| `[method]outgoing-request.headers` | ✅ | :22032 | — |
| `[method]outgoing-request.body` | ✅ | :22033 | — |
| `[resource-drop]outgoing-request` | ✅ | :22034 | — |
| `[constructor]outgoing-response` | ✅ | :22036 | — |
| `[method]outgoing-response.status-code` | ✅ | :22037 | — |
| `[method]outgoing-response.set-status-code` | ✅ | :22038 | — |
| `[method]outgoing-response.headers` | ✅ | :22039 | — |
| `[method]outgoing-response.body` | ✅ | :22040 | — |
| `[resource-drop]outgoing-response` | ✅ | :22041 | — |
| `[method]incoming-request.method` | ✅ | :22043 | — |
| `[method]incoming-request.path-with-query` | ✅ | :22044 | — |
| `[method]incoming-request.scheme` | ✅ | :22045 | — |
| `[method]incoming-request.authority` | ✅ | :22046 | — |
| `[method]incoming-request.headers` | ✅ | :22047 | — |
| `[method]incoming-request.consume` | ✅ | :22048 | — |
| `[resource-drop]incoming-request` | ✅ | :22049 | — |
| `[method]incoming-response.status` | ✅ | :22051 | — |
| `[method]incoming-response.headers` | ✅ | :22052 | — |
| `[method]incoming-response.consume` | ✅ | :22053 | — |
| `[resource-drop]incoming-response` | ✅ | :22054 | — |
| `[method]incoming-body.stream` | ✅ | :22056 | — |
| `[static]incoming-body.finish` | ✅ | :22057 | — |
| `[resource-drop]incoming-body` | ✅ | :22058 | — |
| `[method]outgoing-body.write` | ✅ | :22059 | — |
| `[static]outgoing-body.finish` | ✅ | :22060 | — |
| `[resource-drop]outgoing-body` | ✅ | :22061 | — |
| `[method]future-incoming-response.subscribe` | ✅ | :22063 | — |
| `[method]future-incoming-response.get` | ✅ | :22064 | — |
| `[resource-drop]future-incoming-response` | ✅ | :22065 | — |
| `[method]future-trailers.subscribe` | ✅ | :22066 | — |
| `[method]future-trailers.get` | ✅ | :22067 | — |
| `[resource-drop]future-trailers` | ✅ | :22068 | — |
| `[constructor]request-options` | ✅ | :22070 | — |
| `[method]request-options.connect-timeout` | ✅ | :22073 | 0.2-style unprefixed getter name (`get-` prefix only in 0.3). Stored in nanoseconds, copied into worker-owned state, and applied to initial DNS/TCP acquisition. Zig's lazy TLS handshake and automatic redirect reconnects are not covered by this deadline. |
| `[method]request-options.set-connect-timeout` | ✅ | :22074 | Stores `option<duration>` (nanoseconds). Returns `result` ok unconditionally. |
| `[method]request-options.first-byte-timeout` | ✅ | :22075 | Stored for round-tripping only; enforcing it requires a deadline-aware HTTP/TLS reader and remains #616 A1b/A7 work. |
| `[method]request-options.set-first-byte-timeout` | ✅ | :22076 | Same as above; field `first_byte_timeout_ns`. |
| `[method]request-options.between-bytes-timeout` | ✅ | :22077 | Stored for round-tripping only; enforcing it requires a deadline-aware HTTP/TLS reader and remains #616 A1b/A7 work. |
| `[method]request-options.set-between-bytes-timeout` | ✅ | :22078 | Same as above; field `between_bytes_timeout_ns`. |
| `[resource-drop]request-options` | ✅ | :22079 | — |
| `[method]response-outparam.send-informational` (`@unstable`) | N/A | — | Unstable feature; intentionally outside the stable denominator. No current follow-up is identified. |
| `[static]response-outparam.set` | ✅ | :22080 | — |
| `[resource-drop]response-outparam` | ✅ | :22081 | — |

62/62 stable (100 %). `send-informational` is the only
`@unstable` feature and is excluded from the implemented %.

### `wasi:http/outgoing-handler`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `handle` | ✅ | wasi_cli_adapter.zig:22102 |

1/1 (100 %). `std.http.Client.fetch` real outbound, http + https.

### `wasi:http/incoming-handler`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `handle` (guest export) | N/A | wasi_cli_adapter.zig:22120 |

0/0 stable host imports. Pinned
[`proxy.wit`](https://github.com/WebAssembly/wasi-http/blob/d97efe470e4d1e3f959162d81eef9666335bc186/wit/proxy.wit)
exports `incoming-handler`; its WIT documentation likewise says
components should export it. The adapter's no-op provider is a
synthetic compatibility registration and is not method implementation
coverage.

## Preview 3 (0.3.0) interfaces

Published P3 method rows below are compared to the six vendored
`package.wit` files at testsuite pin `7d0a7811`, not to moving
upstream branches. The synthetic P3 `wasi:io` compatibility instances
are called out separately as zero-method adapter registrations.

### `wasi:cli/stdin@0.3.0`, `stdout@0.3.0`, `stderr@0.3.0`

| Interface | WIT method | Status | Adapter location |
| --- | --- | :-: | --- |
| `stdin@0.3.0` | `read-via-stream` | ✅ | wasi_cli_adapter.zig:5448 |
| `stdout@0.3.0` | `write-via-stream` | ✅ | :5462 |
| `stderr@0.3.0` | `write-via-stream` | ✅ | :5476 |

3/3 (100 %). Host-attached `stream<u8>` + ready
`future<result<_,error-code>>`
([PR #514](https://github.com/cataggar/wamr/pull/514) / #548).

### `wasi:cli/exit@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `exit` | ✅ | wasi_cli_adapter.zig:5498 |
| `exit-with-code` | ✅ | :5501 |

2/2 (100 %).

### `wasi:cli/environment@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `get-environment` | ✅ | wasi_cli_adapter.zig:5518 | — |
| `get-arguments` | ✅ | :5521 | — |
| `get-initial-cwd` | ✅ | :5525 | 0.3 rename of `initial-cwd`. |

3/3 (100 %).

### `wasi:cli/types@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| _(none — type-only)_ | N/A | wasi_cli_adapter.zig:5534 |

0/0 (N/A). Holds the `error-code { io, illegal-byte-sequence, pipe }`
enum only. Lifted as a u8 discriminant in canonical-ABI.

### `wasi:cli/run@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `run` (guest export) | N/A | — |

`run` is exported by the vendored command world, not a host import
registered by `populateWasiProviders`.

### `wasi:cli/terminal-*@0.3.0`

Identical surface to 0.2.

| Interface | WIT method | Status | Adapter location |
| --- | --- | :-: | --- |
| `terminal-stdin@0.3.0` | `get-terminal-stdin` | ✅ | :5559 |
| `terminal-stdout@0.3.0` | `get-terminal-stdout` | ✅ | :5565 |
| `terminal-stderr@0.3.0` | `get-terminal-stderr` | ✅ | :5571 |
| `terminal-input@0.3.0` | `[resource-drop]terminal-input` | ✅ | :5577 |
| `terminal-output@0.3.0` | `[resource-drop]terminal-output` | ✅ | :5583 |

5/5 (100 %).

### `wasi:io/streams@0.3.0` / `wasi:io/error@0.3.0`

| Interface | WIT method | Status | Adapter location |
| --- | --- | :-: | --- |
| `streams@0.3.0` | _(none — type-only)_ | N/A | wasi_cli_adapter.zig:6287 |
| `error@0.3.0` | _(none — type-only)_ | N/A | :6287 |

0/0 (N/A). The P3 `stream<u8>` / `future<…>` engine lives in the
canonical-ABI (see `comp_inst.streams` and the executor in
`stream_canon.zig` / `async_canon.zig`), not on the adapter
([PR #510](https://github.com/cataggar/wamr/pull/510) / #481).

### `wasi:clocks/monotonic-clock@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `now` | ✅ | wasi_cli_adapter.zig:6381 | — |
| `get-resolution` | ✅ | :6384 | 0.3 rename of `resolution`. |
| `wait-for` | ✅ | :6387 | Host-driven `task.cancel`-aware ([PR #558](https://github.com/cataggar/wamr/pull/558)). |
| `wait-until` | ✅ | :6390 | Same. |

4/4 (100 %).

### `wasi:clocks/system-clock@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `now` | ✅ | wasi_cli_adapter.zig:6410 | 0.3 rename `wall-clock` → `system-clock`. |
| `get-resolution` | ✅ | :6413 | — |

2/2 (100 %).

### `wasi:clocks/types@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| _(none — type-only)_ | N/A | wasi_cli_adapter.zig:6435 |

0/0 (N/A). Holds `duration` / `instant` aliases. Needed only so the
P3 `populateWasiProviders` matcher binds the type-only import.

### `wasi:clocks/timezone@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `iana-id` (`@unstable`) | N/A | — |
| `utc-offset` (`@unstable`) | N/A | — |
| `to-debug-string` (`@unstable`) | N/A | — |

The vendored WIT gates all three behind
`@unstable(feature = clocks-timezone)`. wamr does not register the
interface; all are excluded from the stable denominator.

### `wasi:random/random@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-random-bytes` | ✅ | wasi_cli_adapter.zig:6519 |
| `get-random-u64` | ✅ | :6522 |

2/2 (100 %).

### `wasi:random/insecure@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-insecure-random-bytes` | ✅ | wasi_cli_adapter.zig:6538 |
| `get-insecure-random-u64` | ✅ | :6541 |

2/2 (100 %).

### `wasi:random/insecure-seed@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `get-insecure-seed` | ✅ | wasi_cli_adapter.zig:6559 | 0.3 rename of `insecure-seed`. |

1/1 (100 %).

### `wasi:filesystem/preopens@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-directories` | ✅ | wasi_cli_adapter.zig:9172 |

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
| `[method]descriptor.read-via-stream` | ✅ | wasi_cli_adapter.zig:9204 | Host-driver eager-lower; see [PR #577](https://github.com/cataggar/wamr/pull/577). |
| `[method]descriptor.write-via-stream` | ✅ | :9205 | Host-driver pwrite ([PR #577](https://github.com/cataggar/wamr/pull/577)). |
| `[method]descriptor.append-via-stream` | ✅ | :9206 | — |
| `[method]descriptor.read-directory` | ✅ | :9208 | `stream<directory-entry>` driver. |
| `[method]descriptor.advise` | ✅ | :9210 | Async wrapper over 0.2 body. |
| `[method]descriptor.create-directory-at` | ✅ | :9211 | — |
| `[method]descriptor.get-flags` | ✅ | :9212 | — |
| `[method]descriptor.get-type` | ✅ | :9213 | — |
| `[method]descriptor.is-same-object` | ✅ | :9214 | — |
| `[method]descriptor.link-at` | ✅ | :9215 | — |
| `[method]descriptor.metadata-hash` | ✅ | :9216 | — |
| `[method]descriptor.metadata-hash-at` | ✅ | :9217 | — |
| `[method]descriptor.open-at` | ✅ | :9218 | — |
| `[method]descriptor.readlink-at` | ✅ | :9219 | — |
| `[method]descriptor.remove-directory-at` | ✅ | :9220 | — |
| `[method]descriptor.rename-at` | ✅ | :9221 | — |
| `[method]descriptor.set-size` | ✅ | :9222 | — |
| `[method]descriptor.set-times` | ✅ | :9223 | — |
| `[method]descriptor.set-times-at` | ✅ | :9224 | — |
| `[method]descriptor.stat` | ✅ | :9225 | — |
| `[method]descriptor.stat-at` | ✅ | :9226 | — |
| `[method]descriptor.symlink-at` | ✅ | :9227 | — |
| `[method]descriptor.sync` | ✅ | :9228 | — |
| `[method]descriptor.sync-data` | ✅ | :9229 | — |
| `[method]descriptor.unlink-file-at` | ✅ | :9230 | — |
| `[resource-drop]descriptor` | ✅ | :9232 | Carry-over from 0.2 (`fsDescriptorDrop`). |

26/26 (100 %).

### `wasi:sockets/types@0.3.0`

Unified TCP + UDP resource surface ([PR #486](https://github.com/cataggar/wamr/pull/486)
/ [#544](https://github.com/cataggar/wamr/pull/544) / [#565](https://github.com/cataggar/wamr/pull/565)).
Getters/setters are wrapped via `p3SocketWrapper(...)` so the 0.2
bodies serve both surfaces.

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[static]tcp-socket.create` | ✅ | wasi_cli_adapter.zig:16778 |
| `[method]tcp-socket.bind` | ✅ | :16779 |
| `[method]tcp-socket.connect` | ✅ | :16780 |
| `[method]tcp-socket.listen` | ✅ | :16781 |
| `[method]tcp-socket.send` | ✅ | :16782 |
| `[method]tcp-socket.receive` | ✅ | :16783 |
| `[method]tcp-socket.get-local-address` | ✅ | :16785 |
| `[method]tcp-socket.get-remote-address` | ✅ | :16786 |
| `[method]tcp-socket.get-is-listening` | ✅ | :16787 |
| `[method]tcp-socket.get-address-family` | ✅ | :16788 |
| `[method]tcp-socket.set-listen-backlog-size` | ✅ | :16789 |
| `[method]tcp-socket.get-keep-alive-enabled` | ✅ | :16790 |
| `[method]tcp-socket.set-keep-alive-enabled` | ✅ | :16791 |
| `[method]tcp-socket.get-keep-alive-idle-time` | ✅ | :16792 |
| `[method]tcp-socket.set-keep-alive-idle-time` | ✅ | :16793 |
| `[method]tcp-socket.get-keep-alive-interval` | ✅ | :16794 |
| `[method]tcp-socket.set-keep-alive-interval` | ✅ | :16795 |
| `[method]tcp-socket.get-keep-alive-count` | ✅ | :16796 |
| `[method]tcp-socket.set-keep-alive-count` | ✅ | :16797 |
| `[method]tcp-socket.get-hop-limit` | ✅ | :16798 |
| `[method]tcp-socket.set-hop-limit` | ✅ | :16799 |
| `[method]tcp-socket.get-receive-buffer-size` | ✅ | :16800 |
| `[method]tcp-socket.set-receive-buffer-size` | ✅ | :16801 |
| `[method]tcp-socket.get-send-buffer-size` | ✅ | :16802 |
| `[method]tcp-socket.set-send-buffer-size` | ✅ | :16803 |
| `[resource-drop]tcp-socket` | ✅ | :16804 |
| `[static]udp-socket.create` | ✅ | :16806 |
| `[method]udp-socket.bind` | ✅ | :16807 |
| `[method]udp-socket.connect` | ✅ | :16808 |
| `[method]udp-socket.disconnect` | ✅ | :16809 |
| `[method]udp-socket.send` | ✅ | :16810 |
| `[method]udp-socket.receive` | ✅ | :16811 |
| `[method]udp-socket.get-local-address` | ✅ | :16813 |
| `[method]udp-socket.get-remote-address` | ✅ | :16814 |
| `[method]udp-socket.get-address-family` | ✅ | :16815 |
| `[method]udp-socket.get-unicast-hop-limit` | ✅ | :16816 |
| `[method]udp-socket.set-unicast-hop-limit` | ✅ | :16817 |
| `[method]udp-socket.get-receive-buffer-size` | ✅ | :16818 |
| `[method]udp-socket.set-receive-buffer-size` | ✅ | :16819 |
| `[method]udp-socket.get-send-buffer-size` | ✅ | :16820 |
| `[method]udp-socket.set-send-buffer-size` | ✅ | :16821 |
| `[resource-drop]udp-socket` | ✅ | :16822 |

42/42 (100 %), compared directly with the vendored
`wasi-sockets-0.3.0/package.wit` at testsuite pin `7d0a7811`.

### `wasi:sockets/ip-name-lookup@0.3.0`

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `resolve-addresses` (free fn) | ✅ | wasi_cli_adapter.zig:16845 | `std.net.HostName.lookup`-backed, allow-list-gated; settles a `future<result<list<ip-address>, error-code>>` via `socketReadyResultFuture` / `spawnReadyFutureBytes`. |

1/1 (100 %).

### `wasi:http/types@0.3.0`

Unified `request` / `response` resource ([PR #487](https://github.com/cataggar/wamr/pull/487)
/ [#568](https://github.com/cataggar/wamr/pull/568)).

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `[constructor]fields` | ✅ | wasi_cli_adapter.zig:21922 |
| `[static]fields.from-list` | ✅ | :21923 |
| `[method]fields.get` | ✅ | :21924 |
| `[method]fields.has` | ✅ | :21925 |
| `[method]fields.set` | ✅ | :21926 |
| `[method]fields.delete` | ✅ | :21927 |
| `[method]fields.get-and-delete` | ✅ | :21928 |
| `[method]fields.append` | ✅ | :21929 |
| `[method]fields.copy-all` | ✅ | :21930 |
| `[method]fields.clone` | ✅ | :21931 |
| `[resource-drop]fields` | ✅ | :21932 |
| `[static]request.new` | ✅ | :21934 |
| `[method]request.get-method` | ✅ | :21935 |
| `[method]request.set-method` | ✅ | :21936 |
| `[method]request.get-path-with-query` | ✅ | :21937 |
| `[method]request.set-path-with-query` | ✅ | :21938 |
| `[method]request.get-scheme` | ✅ | :21939 |
| `[method]request.set-scheme` | ✅ | :21940 |
| `[method]request.get-authority` | ✅ | :21941 |
| `[method]request.set-authority` | ✅ | :21942 |
| `[method]request.get-options` | ✅ | :21943 |
| `[method]request.get-headers` | ✅ | :21944 |
| `[static]request.consume-body` | ✅ | :21945 |
| `[resource-drop]request` | ✅ | :21946 |
| `[constructor]request-options` | ✅ | :21948 |
| `[method]request-options.get-connect-timeout` | ✅ | :21949 | Stored in nanoseconds, snapshotted when the request is constructed, and applied to initial DNS/TCP acquisition. The child options resource may be dropped before send. Zig's lazy TLS handshake and automatic redirect reconnects are not deadline-covered. |
| `[method]request-options.set-connect-timeout` | ✅ | :21950 |
| `[method]request-options.get-first-byte-timeout` | ✅ | :21951 | Round-trip only; deadline-aware HTTP/TLS reader support remains #616 A1b/A7 work. |
| `[method]request-options.set-first-byte-timeout` | ✅ | :21952 |
| `[method]request-options.get-between-bytes-timeout` | ✅ | :21953 | Round-trip only; deadline-aware HTTP/TLS reader support remains #616 A1b/A7 work. |
| `[method]request-options.set-between-bytes-timeout` | ✅ | :21954 |
| `[method]request-options.clone` | ✅ | :21955 |
| `[resource-drop]request-options` | ✅ | :21956 |
| `[static]response.new` | ✅ | :21958 |
| `[method]response.get-status-code` | ✅ | :21959 |
| `[method]response.set-status-code` | ✅ | :21960 |
| `[method]response.get-headers` | ✅ | :21961 |
| `[static]response.consume-body` | ✅ | :21962 |
| `[resource-drop]response` | ✅ | :21963 |

39/39 (100 %). All six `request-options` timeout accessors plus
`clone` are bound — the 0.2 gap above is closed in 0.3.

### `wasi:http/handler@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `handle` | ✅ | wasi_cli_adapter.zig:21981 |

1/1 (100 %). The middleware world imports this interface, and the
pinned WIT permits requests read from the network, synthesized, or
forwarded by another component. Delegating to `httpClientSendP3`
therefore implements a valid real HTTP(S)-forwarding path.

### `wasi:http/client@0.3.0`

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `send` | ✅ | wasi_cli_adapter.zig:21995 |

1/1 (100 %). Outbound async state machine; HTTP + HTTPS via
`std.http.Client` + `std.crypto.tls` ([PR #583 A2](https://github.com/cataggar/wamr/issues/583)
/ [#590](https://github.com/cataggar/wamr/pull/590)).

## Cross-version (no version-multiplex split)

### `wasi:keyvalue/store@0.2.0-draft2`

Vendored WIT:
[`store.wit`](../docs/wasi-keyvalue-wit-vendored/store.wit) (vendored).

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `open` (free fn) | ✅ | wasi_cli_adapter.zig:22163 |
| `[method]bucket.get` | ✅ | :22164 |
| `[method]bucket.set` | ✅ | :22165 |
| `[method]bucket.delete` | ✅ | :22166 |
| `[method]bucket.exists` | ✅ | :22167 |
| `[method]bucket.list-keys` | ✅ | :22168 |
| `[resource-drop]bucket` | ✅ | :22169 |

7/7 (100 %). `std.StringHashMapUnmanaged`-backed, with optional
JSON-file persistence via `--keyvalue-store`; no replication.

### `wasi:keyvalue/atomics@0.2.0-draft2`

Vendored WIT:
[`atomic.wit`](../docs/wasi-keyvalue-wit-vendored/atomic.wit) (vendored).

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `increment` (free fn) | ✅ | wasi_cli_adapter.zig:22202 | Real arithmetic over the bucket map ([PR #583 B4](https://github.com/cataggar/wamr/issues/583)). |
| `swap` (free fn) | ✅ | :22203 | Atomic compare-and-swap; writes on match and returns a refreshed retry handle on mismatch. |
| `[static]cas.new` | ✅ | :22204 | Captures the current bucket/key value in a live CAS resource. |
| `[method]cas.current` | ✅ | :22205 | Returns the captured optional byte value. |
| `[resource-drop]cas` | ✅ | :22206 | Frees the live CAS resource and owned snapshot. |

5/5 implemented (100 %). The three formerly stubbed CAS operations
were completed by [PR #608](https://github.com/cataggar/wamr/pull/608).

### `wasi:keyvalue/batch@0.2.0-draft2`

Vendored WIT:
[`batch.wit`](../docs/wasi-keyvalue-wit-vendored/batch.wit) (vendored).

| WIT method | Status | Adapter location |
| --- | :-: | --- |
| `get-many` (free fn) | ✅ | wasi_cli_adapter.zig:22232 |
| `set-many` (free fn) | ✅ | :22233 |
| `delete-many` (free fn) | ✅ | :22234 |

3/3 (100 %).

### `wasi:logging/logging@0.1.0-draft`

Pinned reference: [`logging.wit`](https://github.com/WebAssembly/wasi-logging/blob/d31c41d0d9eed81aabe02333d0025d42acf3fb75/wit/logging.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `log` (free fn) | ✅ | wasi_cli_adapter.zig:6727 | Routes to host stderr + `std.log.scoped(.wasi_guest)`; level filter via `--log-level` / `WAMR_LOG_LEVEL`. |

1/1 (100 %). The `_p2` slot (`populateWasiLoggingP2`, line 6740)
is reserved for a future `@0.2.x` revision and currently mirrors
the same `wasiLog` callback. It is a registration compatibility slot,
not a second published method in the denominator.

### `wasi:config/store@0.2.0-rc.1`

Pinned WIT: [`store.wit`](https://github.com/WebAssembly/wasi-config/blob/f5bf419bf12ef6e4f942438d36d59136a10c781b/wit/store.wit)

| WIT method | Status | Adapter location | Notes |
| --- | :-: | --- | --- |
| `get` (free fn) | ✅ | wasi_cli_adapter.zig:23528 | Layered: `--config-store=PATH.json` overrides `WAMR_CONFIG_<KEY>` env. |
| `get-all` (free fn) | ✅ | :23531 | Same layering. |

2/2 (100 %). The `error::upstream` / `error::io` arms are
reserved for future Vault / Kubernetes / etc. backends; the
in-memory store never surfaces them.

## Footer

Re-run this audit against any future `main` SHA by:

1. Updating the SHA + date at the top of this file.
2. Running the [methodology](#methodology) extractor against
   `wasi_cli_adapter.zig`.
3. Verifying the testsuite submodule and `build.zig.zon` dependency
   pins, then diffing the extracted member names against those exact
   WIT/build inputs.
4. Reading every registered callback body; registration alone proves
   only link coverage. Any absent stable name is ❌, and any canned,
   no-op, or placeholder callback is ⚠️.
5. Recomputing the table totals rather than carrying forward the prior
   summary.

Previous refresh: `bf7ab7ef` (2026-07-13). Quarterly cadence:
next review in mid-October 2026, tracked by
[#616 C1](https://github.com/cataggar/wamr/issues/616).
