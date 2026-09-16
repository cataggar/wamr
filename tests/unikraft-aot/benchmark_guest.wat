(module
  (import "wasi_unstable" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_unstable" "proc_exit" (func $exit (param i32)))
  (import "wasi_unstable" "fd_close" (func $close (param i32) (result i32)))
  (memory (export "memory") 1 3)
  (global $starts (mut i32) (i32.const 0))
  (global $calls (mut i32) (i32.const 0))
  (data (i32.const 32) "start\0a")
  (data (i32.const 48) "\00\ff\c3(\0a\00")
  (func $output (param $at i32)
    i32.const 0 local.get $at i32.store
    i32.const 4 i32.const 6 i32.store
    i32.const 1 i32.const 0 i32.const 1 i32.const 8 call $write drop)
  (func $initialize
    global.get $starts i32.const 1 i32.add global.set $starts
    i32.const 32 call $output
    i32.const 2 call $close if unreachable end)
  (start $initialize)
  (func $returned (export "returned")
    global.get $starts i32.const 1 i32.ne if unreachable end
    global.get $calls if unreachable end
    i32.const 2 call $close i32.const 8 i32.ne if unreachable end
    i32.const 1 global.set $calls
    i32.const 48 call $output)
  (func (export "_start")
    call $returned
    i32.const -1 call $exit)
  (func (export "exit_full_u32")
    i32.const 48 call $output
    i32.const -1 call $exit)
  (func (export "trap")
    i32.const 48 call $output
    unreachable))
