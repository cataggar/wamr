(module
  (memory 2)

  (func (export "_start")
    (local $base i32)
    (local $round i32)
    (local $i i32)

    i32.const 1024
    local.set $base
    i32.const 0
    local.set $round

    (loop $outer
      i32.const 0
      local.set $i

      (loop $inner
        local.get $base
        local.get $i
        i32.add
        local.get $i
        i32.store8

        local.get $i
        i32.const 1
        i32.add
        local.set $i
        local.get $i
        i32.const 65536
        i32.lt_s
        br_if $inner)

      local.get $round
      i32.const 1
      i32.add
      local.set $round
      local.get $round
      i32.const 20000
      i32.lt_s
      br_if $outer)

    i32.const 1025
    i32.load8_u
    i32.const 1
    i32.ne
    if unreachable end

    i32.const 1279
    i32.load8_u
    i32.const 255
    i32.ne
    if unreachable end

    i32.const 66559
    i32.load8_u
    i32.const 255
    i32.ne
    if unreachable end)
)
