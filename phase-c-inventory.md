# Phase C — AOT cross-instance fn-import inventory (#662)

Tool: `WAMR_AOT_DEBUG=1 ./zig-out/bin/wamr run …` with a temporary debug
hook in `firstUnsupportedAotImport` (see `src/component/instance.zig`).
The hook prints every non-WASI/non-spectest function import (and every
table/memory/global/tag import) for each core that the AOT-only policy
walks. Output below is collated per fixture and grouped by what wiring
each import actually needs.

The inventory captures the four representative fixtures called out in
the task: `zig-hello`, `zig-exit`, `zig-http`, and one p3 fixture
(`cli-stdio`). The p3 surface as a whole follows the same shape as
`cli-stdio`.

## Findings summary

* `__main_module__.{_start, cabi_realloc}` is the **only true
  cross-instance core-to-core import** in the component-examples set —
  the wabt-bundled wasi-preview1→preview2 adapter imports these from
  the main core via `(instance (instantiate $adapter (with
  "__main_module__" (instance $main))))`.
* Every other rejected import on the four fixtures is **either a
  canon.lower trampoline** (interface name like
  `wasi:cli/stdout@0.2.6.get-stdout`) **or a canon-builtin** import
  (`[task-return]`, `[context-get-0]`, `[waitable-*]`,
  `[stream-*]`, `[future-*]`, …). Neither is "cross-instance" in the
  core-to-core sense; both need host-bridge plumbing that lives in
  Phase D / #520 / #533 territory.
* The wabt adapter cores in zig-hello / zig-exit / zig-http carry a
  large `wasi:*@0.2.6` canon.lower surface, but at runtime the
  preview1 imports of the *main* core are satisfied directly by the
  AOT host bridge (`wasi_snapshot_preview1`), so the adapter body is
  effectively dead code — its exports are never reached on the
  call paths these examples exercise. We can therefore make the
  adapter merely *instantiate* (install best-effort thunks) without
  having to make every adapter import actually do useful work.
* p3 fixtures are dominated by **canon-builtin** imports
  (`[task-return]run`, `[task-cancel]`, `[context-get-0]`,
  `[waitable-join]`, `[waitable-set-{new,poll,drop}]`,
  `[stream-{new,read,write,cancel-read,cancel-write,drop-readable,drop-writable}-*]`,
  `[future-{new,read,write,cancel-read,cancel-write,drop-readable,drop-writable}-*]`)
  plus host canon.lower over `wasi:cli/* @ 0.2.0`,
  `wasi:io/* @ 0.2.0`. The p3 surface cannot be unblocked by
  cross-instance fn-import wiring alone; those imports must reach
  `dispatchCanonBuiltin` / `componentTrampoline` from the AOT side,
  which is out of scope for Phase C.

## zig-hello.component.wasm

* Core 0 (main): only `env.memory` + `wasi_snapshot_preview1.*`
  imports → already AOT-clean.
* Core 1 (wabt preview1→preview2 adapter):

### cross-instance core-to-core (in scope for Phase C)
- module=`__main_module__`, field=`_start`,             sig=`() -> ()`
- module=`__main_module__`, field=`cabi_realloc`,       sig=`(i32,i32,i32,i32) -> i32`

### canon.lower host-trampoline imports (out of scope; install trap stub)
- `wasi:cli/stdout@0.2.6.get-stdout` `() -> i32`
- `wasi:cli/stderr@0.2.6.get-stderr` `() -> i32`
- `wasi:io/streams@0.2.6.[method]output-stream.blocking-write-and-flush` `(i32,i32,i32,i32) -> ()`
- `wasi:io/streams@0.2.6.[resource-drop]output-stream` `(i32) -> ()`
- `wasi:filesystem/preopens@0.2.6.get-directories` `(i32) -> ()`
- `wasi:filesystem/types@0.2.6.[method]descriptor.write-via-stream` `(i32,i64,i32) -> ()`

## zig-exit.component.wasm

Same shape as zig-hello (same wabt adapter on top of a slightly
different main core).

### cross-instance core-to-core (in scope)
- `__main_module__._start` `() -> ()`
- `__main_module__.cabi_realloc` `(i32,i32,i32,i32) -> i32`

### canon.lower host-trampoline imports (out of scope; trap stub OK)
- `wasi:cli/exit@0.2.6.exit` `(i32) -> ()`
- `wasi:cli/exit@0.2.6.exit-with-code` `(i32) -> ()`
- `wasi:cli/stdout@0.2.6.get-stdout` `() -> i32`
- `wasi:cli/stderr@0.2.6.get-stderr` `() -> i32`
- `wasi:io/streams@0.2.6.[method]output-stream.blocking-write-and-flush` `(i32,i32,i32,i32) -> ()`
- `wasi:io/streams@0.2.6.[resource-drop]output-stream` `(i32) -> ()`
- `wasi:filesystem/preopens@0.2.6.get-directories` `(i32) -> ()`
- `wasi:filesystem/types@0.2.6.[method]descriptor.write-via-stream` `(i32,i64,i32) -> ()`

## zig-http.component.wasm

Not separately runnable today (the smoke-driver gate is currently
behind `aot_broken_components`); shape mirrors zig-hello with an
extra `wasi:http@0.2.6` canon.lower surface on the adapter.

## cli-stdio.wasm (one representative p3 fixture)

The first rejected import is the **canon-builtin** `task-return`:

```
[export]wasi:cli/run@0.3.0-rc-2026-03-15 . [task-return]run   (i32) -> ()
```

The full p3 reject set on `cli-stdio` (canon-builtins + canon.lower
of preview2 interfaces):

### canon-builtins (Phase D / #520 territory)
- `[export]wasi:cli/run@0.3.0-rc-2026-03-15.[task-return]run` `(i32) -> ()`
- `[export]$root.[task-cancel]` `() -> ()`
- `$root.[context-get-0]` `() -> i32`
- `$root.[context-set-0]` `(i32) -> ()`
- `$root.[waitable-join]` `(i32,i32) -> ()`
- `$root.[waitable-set-new]` `() -> i32`
- `$root.[waitable-set-poll]` `(i32,i32) -> i32`
- `$root.[waitable-set-drop]` `(i32) -> ()`
- (cli-stdio adds the stream-*/future-* set on `wasi:cli/std{in,out,err}@0.3.0-rc`):
  `[stream-new-0]`, `[stream-{cancel,drop}-{read,write}-0]`,
  `[async-lower][stream-{read,write}-0]`, plus the matching `future-*`
  surface — all sig `(i32,…) -> i32 / -> ()` / `() -> i64`.

### canon.lower over preview2 interfaces (host trampolines)
- `wasi:cli/std{in,out,err}@0.3.0-rc-2026-03-15.{read-via-stream,write-via-stream}`
- `wasi:cli/environment@0.3.0-rc-2026-03-15.{get-arguments,get-environment,get-initial-cwd}`
- The full `wasi:io/{error,poll,streams}@0.2.0` resource-drop set +
  pollable.block + output-stream.{check-write,write,blocking-flush,subscribe}
- The full `wasi:cli/terminal-*@0.2.0` resource-drop + `get-terminal-*` set
- `wasi:cli/environment@0.2.0.get-environment`
- `wasi:cli/exit@0.2.0.exit`
- `wasi:cli/stdin@0.2.0.get-stdin`,  `wasi:cli/stdout@0.2.0.get-stdout`, `wasi:cli/stderr@0.2.0.get-stderr`

## scope conclusion for Phase C

The only set of imports this PR can legitimately *wire* is the
cross-instance core-to-core pair (`__main_module__._start`,
`__main_module__.cabi_realloc`). Everything else collapses to one of:

1. **Canon-builtin** dispatch (`dispatchCanonBuiltin`) — out of scope
   here, blocks the entire p3 surface, tracked in #520 / Phase D.
2. **Canon.lower into a host import** (`componentTrampoline` /
   `dispatchAotComponentTrampoline`) — also out of scope here, blocks
   any p2/p3 fixture that actually calls into preview2 interfaces.

For the three component-examples (zig-hello, zig-exit, zig-http) the
adapter's canon.lower imports are **dead code** at runtime — the main
core's preview1 imports are satisfied directly by the AOT host bridge
via `host_bridge.isWasiModule`. Installing a `trap-on-call` stub for
those adapter-side canon.lower imports is enough to let the component
instantiate; the actual `dispatchCanonBuiltin` /
`componentTrampoline` plumbing remains follow-up work.

For the 38 p3 fixtures, the canon-builtin imports are on the **hot
path**; trap-on-call stubs would just push the failure from
instantiation time to call time. The p3 skip entries in
`tests/wasi-p3-testsuite-skip.json` therefore stay in place until
canon-builtin + canon.lower bridging lands.
