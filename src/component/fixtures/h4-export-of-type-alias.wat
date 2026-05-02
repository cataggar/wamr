;; h4-export-of-type-alias.wat — regression fixture for issue #310.
;;
;; Reproduces the canon-lower trampoline bug where an instance-type body's
;; `(export "X" (type (eq N)))` slot resolved to `null` in the per-trampoline
;; TypeRegistry extension, even though it should alias local slot N.
;;
;; Trampoline result type is `.type_idx = base + 1` (the export-of-type slot),
;; which pre-fix had `registry.get(idx) == null` and tripped
;; `CompoundNeedsRegistry`. Post-fix, the slot resolves to the underlying
;; `(type u64)` at slot 0 and the round-trip works.
;;
;; Mirrors the shape of `wasi:clocks/wall-clock` and `monotonic-clock`
;; instance bodies that TinyGo's `_initialize` calls during component
;; startup once issue #309 lets it run.
;;
;; Build with:
;;   wasm-tools parse -o h4-export-of-type-alias.wasm h4-export-of-type-alias.wat
(component
  (type (;0;) (instance
    (type (;0;) u64)
    (export (;1;) "instant" (type (eq 0)))
    (type (;2;) (func (result 1)))
    (export (;0;) "now" (func (type 2)))
  ))
  (import "host:test/clock" (instance (;0;) (type 0)))
  (alias export 0 "now" (func (;0;)))
  (core func (;0;) (canon lower (func 0)))
  (core module (;0;)
    (type (;0;) (func (result i64)))
    (import "host" "now" (func $now (;0;) (type 0)))
    (func (export "call_now") (type 0)
      call $now
    )
  )
  (core instance (;0;)
    (export "now" (func 0))
  )
  (core instance (;1;) (instantiate 0
      (with "host" (instance 0))
    )
  )
  (alias core export 1 "call_now" (core func (;1;)))
  (type (;1;) (func (result u64)))
  (func (;1;) (type 1) (canon lift (core func 1)))
  (export (;2;) "call-now" (func 1))
)
