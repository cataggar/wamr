(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (memory (export "memory") 1 8)
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))
  (func (export "_start")
    i32.const 0
    call $thread_spawn
    i32.const 0
    i32.ge_s
    if
      unreachable
    end))
