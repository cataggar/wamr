;; h3-start-host-call.wat — regression fixture for issue #308.
;;
;; A core module's `(start ...)` directive calls a canon-lowered host
;; import. Before #308 was fixed, instantiation crashed because the
;; trampoline `host_func.call` is bound by `linkImports` and that step
;; runs after the core (start) directive — so the start would dispatch
;; into an unbound trampoline and trap with `HostFuncNotBound`.
;;
;; The post-fix flow defers core (start ...) execution until after
;; `linkImports` binds trampoline `host_func`s, so the host receives the
;; expected `42` argument exactly once during `linkImports`.
;;
;; To rebuild from this source:
;;   wasm-tools parse -o h3-start-host-call.wasm h3-start-host-call.wat
(component
  (import "host:nop/run" (func $run (param "x" u32)))
  (core module $A
    (type $tf (func (param i32)))
    (import "host" "f" (func $f (type $tf)))
    (start $start)
    (func $start
      i32.const 42
      call $f
    )
  )
  (core func $f_low (canon lower (func $run)))
  (core instance $args
    (export "f" (func $f_low))
  )
  (core instance $a (instantiate $A
      (with "host" (instance $args))
    )
  )
)
