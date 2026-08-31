(module
  (func $tiny (param $seed i32) (result i32)
    (local $i i32)
    (local $acc i32)

    local.get $seed
    local.set $acc
    i32.const 0
    local.set $i

    (loop $loop
      local.get $acc
      local.get $i
      i32.add
      i32.const 1
      i32.add
      local.set $acc

      local.get $i
      i32.const 1
      i32.add
      local.set $i
      local.get $i
      i32.const 4
      i32.lt_s
      br_if $loop)

    local.get $acc)

  (func (export "_start")
    (local $round i32)
    (local $acc i32)

    i32.const 0
    local.set $round
    i32.const 0
    local.set $acc

    (loop $outer
      local.get $acc
      call $tiny
      local.set $acc

      local.get $round
      i32.const 1
      i32.add
      local.set $round
      local.get $round
      i32.const 100000000
      i32.lt_s
      br_if $outer)

    local.get $acc
    i32.const 1000000000
    i32.ne
    if unreachable end)
)
