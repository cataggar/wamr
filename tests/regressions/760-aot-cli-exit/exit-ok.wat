;; exit-ok.wat — companion regression fixture for issue #760.
;;
;; Same shape as `exit-with-code-7.wat`, but calls `exit` with
;; `result<_, _>.ok` (discriminant 0). Asserts the clean-exit path —
;; the original failure mode in the issue: a guest that finished its
;; work and called `wasi:cli/exit.exit(ok)` for a normal shutdown
;; SIGSEGV'd the host instead of exiting 0.
;;
;; Rebuild:
;;   wasm-tools parse -o exit-ok.wasm exit-ok.wat
(component
  (import "wasi:cli/exit@0.2.0" (instance $exit_iface
    (export "exit" (func (param "status" (result))))))

  (alias export $exit_iface "exit" (func $exit))

  (core module $A
    (type $tf (func (param i32)))
    (import "host" "exit" (func $exit (type $tf)))
    (func (export "run")
      i32.const 0  ;; result<_, _>.ok discriminant → adapter exit code 0
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

  (type $result_void (result))
  (func $run_lifted (export "run") (result $result_void)
    (canon lift (core func $a "run")))
)
