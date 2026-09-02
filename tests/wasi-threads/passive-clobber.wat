;; Regression for #979: memory.init/data.drop are host-helper calls in AOT.
;; Scalar values deliberately stay live across both calls.
(module
  (memory (export "memory") 1 2)
  (data (i32.const 0)
    "\01\00\00\00\00\00\00\00"
    "\02\00\00\00\00\00\00\00"
    "\03\00\00\00\00\00\00\00"
    "\04\00\00\00\00\00\00\00"
    "\05\00\00\00\00\00\00\00"
    "\06\00\00\00\00\00\00\00"
    "\07\00\00\00\00\00\00\00"
    "\08\00\00\00\00\00\00\00"
    "\09\00\00\00\00\00\00\00"
    "\0a\00\00\00\00\00\00\00")
  (data $payload "abcdefghijklmnop")

  (func $assert (param $condition i32)
    local.get $condition
    i32.eqz
    if
      unreachable
    end)

  (func (export "_start")
    (local $a i64) (local $b i64) (local $c i64) (local $d i64)
    (local $e i64) (local $f i64) (local $g i64) (local $h i64)
    (local $i i64) (local $j i64)

    i32.const 0
    i64.load
    local.set $a
    i32.const 8
    i64.load
    local.set $b
    i32.const 16
    i64.load
    local.set $c
    i32.const 24
    i64.load
    local.set $d
    i32.const 32
    i64.load
    local.set $e
    i32.const 40
    i64.load
    local.set $f
    i32.const 48
    i64.load
    local.set $g
    i32.const 56
    i64.load
    local.set $h
    i32.const 64
    i64.load
    local.set $i
    i32.const 72
    i64.load
    local.set $j

    i32.const 128
    i32.const 0
    i32.const 16
    memory.init $payload
    data.drop $payload

    local.get $a
    local.get $b
    i64.add
    local.get $c
    i64.add
    local.get $d
    i64.add
    local.get $e
    i64.add
    local.get $f
    i64.add
    local.get $g
    i64.add
    local.get $h
    i64.add
    local.get $i
    i64.add
    local.get $j
    i64.add
    i64.const 55
    i64.eq
    call $assert

    i32.const 128
    i64.load
    i64.const 7523094288207667809
    i64.eq
    call $assert
    i32.const 136
    i64.load
    i64.const 8101815670912281193
    i64.eq
    call $assert))
