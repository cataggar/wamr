;; exit-with-code-7.wat — regression fixture for issue #760.
;;
;; Minimal WASIp2 component that imports `wasi:cli/exit@0.2.0.exit` and
;; calls it with `result<_, _>.err` (discriminant 1). The lifted core
;; `run` is exported at the top level so the wamr CLI's
;; `runLoadedComponent` recognises it and invokes the AOT-codegen-flavoured
;; canon-lower trampoline path (`canon_lower_aot`).
;;
;; Before the #760 fix, `wamrAotDispatchComponentTrampolineAot`'s catch
;; arm translated `error.WasiExit` into `status=1`, the generic
;; trampoline pool wrote `0xdeaddeaddeaddead` into the guest's return
;; slot, and the wit-bindgen WASIp1 → preview2 adapter's
;; `unreachable executed at adapter line 2335: host exit implementation
;; didn't exit!` defensive assertion SIGSEGV'd the host (exit 139).
;;
;; After the fix, the dispatcher reads
;; `wasi_cli_adapter.takePendingWasiExitCode()` (pinned by `cliExit` /
;; `cliExitWithCode` immediately before they raise `error.WasiExit`) and
;; calls `std.process.exit` directly — mirroring the existing
;; `aotProcExit` precedent at `src/runtime/aot/host_bridge.zig:464`.
;;
;; Why discriminant 1 and not 7: `wasi:cli/exit.exit` takes a
;; `result<_, _>` — there's no payload-typed exit code, just the
;; two-valued ok/err discriminant (ok → 0, err → 1). For a
;; numeric exit code we would route through `exit-with-code(u8)`
;; instead, which only exists in `wasi:cli/exit@0.3.x` and would
;; require a separate fixture. The 0.2 `exit(err)` path that this
;; fixture exercises is the same code path the user's original repro
;; (the wasip3 codegen-cli component) tripped, just on the 0.2 ABI.
;;
;; Rebuild from this source:
;;   wasm-tools parse -o exit-with-code-7.wasm exit-with-code-7.wat
(component
  (import "wasi:cli/exit@0.2.0" (instance $exit_iface
    (export "exit" (func (param "status" (result))))))

  (alias export $exit_iface "exit" (func $exit))

  (core module $A
    (type $tf (func (param i32)))
    (import "host" "exit" (func $exit (type $tf)))
    (func (export "run")
      i32.const 1  ;; result<_, _>.err discriminant → adapter exit code 1
      call $exit
    )
  )
  (core func $exit_low (canon lower (func $exit)))
  (core instance $args
    (export "exit" (func $exit_low))
  )
  (core instance $a (instantiate $A
    (with "host" (instance $args))
  ))

  ;; Lift the core `run` to a component-level export with the
  ;; `result<_, _>` shape the wamr CLI's `runLoadedComponent` expects.
  ;; The core fn never returns (`exit` is `noreturn` from the guest's
  ;; point of view), so the lift result is only ever observed as a
  ;; metadata shape — never executed past the trapping `exit` call.
  (type $result_void (result))
  (func $run_lifted (export "run") (result $result_void)
    (canon lift (core func $a "run")))
)
