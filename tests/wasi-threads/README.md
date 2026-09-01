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
- `terminate-spinning-child` / `parent-trap-spinning-child`: a child spinning
  in a bare guest loop with no host calls is stopped by `proc_exit` and by a
  parent trap. These gate the AOT loop-header cancel poll; without it the run
  only ends on the teardown deadline, which the fixtures reject by requiring
  empty stderr.
- `trap-beats-late-exit` / `exit-beats-late-trap`: first-wins ordering — a
  losing `proc_exit(0)` cannot mask a trap (status 1) and a losing trap cannot
  mask `proc_exit(6)`.

Every group-termination fixture runs under `tests/wasi-threads/run_bounded.zig`,
which kills the run after 30 s and exits 124, so a regression that hangs fails
the build instead of stalling CI.
- `missing-thread-start` / `wrong-thread-start-signature`: spawn fails
  synchronously with a negative ABI result.
- `unaligned-atomic-*` / `oob-atomic-*`: load, store, RMW, compare-exchange,
  wait32, wait64, and notify produce catchable alignment/bounds traps in the
  interpreter and both AOT backends. `unaligned-oob-atomic-rmw` pins alignment
  as the first trap.
- `concurrent-table-grow`: a maxless table becomes address-stable only when a
  child clone is requested, then survives 2,047 parent growth operations while
  the child loops through `call_indirect`.
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
The wait32/wait64/notify markers use equivalent `drop` sequences and are
lowered with their natural alignment and offset 0.
