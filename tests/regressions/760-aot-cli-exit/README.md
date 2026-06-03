# `wasi:cli/exit.exit` AOT SIGSEGV (#760)

Regression fixtures for [issue #760](https://github.com/cataggar/wamr/issues/760).

## Symptom

Under `wamr run` (AOT-only path, post-#644 / #680), a component that
calls `wasi:cli/exit.exit` for a clean guest shutdown SIGSEGV'd the
host (exit 139) instead of exiting with the requested code. All guest
output was already flushed by the time the crash happened, so the
failure mode looked like "correct work, then crash".

## Root cause (one-line)

`wamrAotDispatchComponentTrampoline{,Aot}` in `src/component/executor.zig`
translated the `error.WasiExit` raised by `cliExit` / `cliExitWithCode`
into a generic `status=1` failure → `genericDispatcher` wrote the
post-#714 `0xdeaddeaddeaddead` sentinel into the guest's return slot →
the wit-bindgen WASIp1 → preview2 adapter's
`unreachable executed at adapter line 2335: host exit implementation
didn't exit!` defensive assertion crashed the host.

## Fix

`cliExit` / `cliExitWithCode` now also pin the requested exit code in
`wasi_cli_adapter.pending_wasi_exit_code` before raising
`error.WasiExit`. The AOT dispatcher catch arms read (and clear) it
via `takePendingWasiExitCode()` and call `std.process.exit` directly,
mirroring the existing `aotProcExit` precedent at
`src/runtime/aot/host_bridge.zig:464` for raw-WASIp1 `proc_exit`.

When the TLS is *not* set (a contract violation by a future raiser of
`error.WasiExit`) the catch arm falls through to the historical
sentinel/warn-once path so the bug surfaces rather than silently
exiting with code 0.

## Fixtures

- `exit-with-code-7.wat` / `.wasm` — calls `wasi:cli/exit.exit` with
  `result<_, _>.err` (discriminant 1) → host process exits **1**.
- `exit-ok.wat` / `.wasm` — calls `wasi:cli/exit.exit` with
  `result<_, _>.ok` (discriminant 0) → host process exits **0**. This
  is the case from the original issue report (a successful run that
  used to crash).

Both fixtures import `wasi:cli/exit@0.2.0` *directly* (no preview1 →
preview2 adapter) to side-step the unrelated #662 cross-instance
issue that blocks the `zig-exit` testsuite fixture from exercising
this same code path under AOT.

The `.wasm` files are checked in (each is 339 bytes) so the test
harness doesn't need a `wasm-tools` build dependency. Rebuild after
editing the WAT:

```sh
wasm-tools parse -o exit-with-code-7.wasm exit-with-code-7.wat
wasm-tools parse -o exit-ok.wasm exit-ok.wat
```

## Automated test wiring

`build.zig`'s `addComponentExamples` flow registers both fixtures as
`zig build component-examples-run` arms that compile through `wamrc`
and assert the host process exit code matches the requested
discriminant.

## Manual repro (matches the original issue's invocation shape)

```sh
unset ZIG_LOCAL_CACHE_DIR
export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
zig build -Doptimize=ReleaseSafe

./zig-out/bin/wamrc compile-component tests/regressions/760-aot-cli-exit/exit-with-code-7.wasm
./zig-out/bin/wamr run tests/regressions/760-aot-cli-exit/exit-with-code-7.wasm
echo "exit=$?"   # expects 1, used to be 139 (SIGSEGV)

./zig-out/bin/wamrc compile-component tests/regressions/760-aot-cli-exit/exit-ok.wasm
./zig-out/bin/wamr run tests/regressions/760-aot-cli-exit/exit-ok.wasm
echo "exit=$?"   # expects 0, used to be 139 (SIGSEGV)
```
