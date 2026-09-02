;; AArch64 regression for #979: a live SIMD value must survive the C-ABI
;; memory.init and data.drop helper calls.
(module
  (memory (export "memory") 1 2)
  (data (i32.const 0)
    "\01\00\00\00\00\00\00\00"
    "\02\00\00\00\00\00\00\00")
  (data $payload "x")

  (func $assert (param $condition i32)
    local.get $condition
    i32.eqz
    if
      unreachable
    end)

  (func (export "_start") (local $vec v128)
    i32.const 0
    v128.load
    local.set $vec

    i32.const 128
    i32.const 0
    i32.const 1
    memory.init $payload
    data.drop $payload

    i32.const 256
    local.get $vec
    v128.store
    i32.const 256
    i64.load
    i64.const 1
    i64.eq
    call $assert
    i32.const 264
    i64.load
    i64.const 2
    i64.eq
    call $assert))
