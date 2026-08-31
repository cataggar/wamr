(module
  (import "wasi_snapshot_preview1" "random_get"
    (func $barrier (param i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (local i32)
    i32.const 3
    local.set 0
    block
      loop
        local.get 0
        i32.const 1
        i32.sub
        local.tee 0
        br_if 0
      end
    end

    i32.const 256
    i32.const 0
    i64.load

    i32.const 8
    i64.load
    i32.const 16
    i64.load
    i32.const 24
    i64.load
    i32.const 32
    i64.load
    i32.const 40
    i64.load
    i32.const 48
    i64.load
    i32.const 56
    i64.load
    i32.const 64
    i64.load
    i32.const 72
    i64.load
    i32.const 80
    i64.load
    i32.const 88
    i64.load
    i32.const 96
    i64.load
    i32.const 104
    i64.load
    i32.const 112
    i64.load
    i32.const 120
    i64.load
    i32.const 128
    i64.load
    i32.const 136
    i64.load
    i32.const 144
    i64.load
    i32.const 152
    i64.load

    i32.const 0
    i32.const 0
    call $barrier
    drop
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.add
    i64.store))
