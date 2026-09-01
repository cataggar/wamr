# Preview-1 WASI thread fixtures

These deterministic core-Wasm fixtures exercise the production interpreter
and AOT bindings enabled by `-Dlib_wasi_threads=true`.

- `pthread-contract`: 300 retained completed threads, generation-safe TIDs,
  guest-side atomic joins, a shared counter, start arguments, guest TLS/global
  isolation, distinct runtime auxiliary stacks, and nested spawning.
- `imported-memory`: host allocation and child retention of an imported shared
  memory, matching the strict proposal shape.
- `shared-descriptors`: a child uses the parent's preopen and stdout table.
- `parent-teardown`: the CLI waits for a child after the parent entry returns.
- `child-trap` / `child-proc-exit`: child terminal outcomes reach the process.
- `terminate-futex-waiter` / `terminate-poll-waiter`: a sibling parked in
  `memory.atomic.wait32` or in `poll_oneoff` is woken by group termination.
  Both print only if the block outlived termination, so the fixtures' empty
  stdout is the wakeup assertion.
- `trap-beats-late-exit` / `exit-beats-late-trap`: first-wins ordering — a
  losing `proc_exit(0)` cannot mask a trap (status 1) and a losing trap cannot
  mask `proc_exit(6)`.
- `missing-thread-start` / `wrong-thread-start-signature`: spawn fails
  synchronously with a negative ABI result.
- `disabled-rejection`: the disabled host import continues returning `-1`.

The adjacent `.wasm` files are tracked so tests need no external SDK. Regenerate
them with:

```sh
zig build update-wasi-thread-fixtures -Dlib_wasi_threads=true -Daot=false
```

Run them through either backend with:

```sh
zig build test-wasi-threads -Dlib_wasi_threads=true -Daot=false
zig build test-aot-threads -Dlib_wasi_threads=true -Dinterp=false
```

The pinned `cataggar/wabt` parses atomic mnemonics but not the WAT `shared`
limits keyword yet. `generate.zig` substitutes type-equivalent marker
instructions, marks memories shared in WABT's IR, validates the ordinary WAT,
then lowers the markers to atomic opcodes before writing the binaries.
`memory.atomic.wait32` uses the same trick (`drop drop` has its stack effect)
and is lowered with `align=4, offset=0`, so waits address memory directly.
