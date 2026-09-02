;; Regression for #979: an active global.get segment before a passive
;; segment must not compact the AOT segment index space.
(module
  (import "env" "base" (global $base i32))
  (memory (export "memory") 1 2)
  (data (global.get $base) "\2a")
  (data $payload "\2b")

  (func $assert (param $condition i32)
    local.get $condition
    i32.eqz
    if
      unreachable
    end)

  (func (export "_start")
    i32.const 0
    i32.load8_u
    i32.const 42
    i32.eq
    call $assert

    i32.const 1
    i32.const 0
    i32.const 1
    memory.init $payload
    data.drop $payload

    i32.const 1
    i32.load8_u
    i32.const 43
    i32.eq
    call $assert))
