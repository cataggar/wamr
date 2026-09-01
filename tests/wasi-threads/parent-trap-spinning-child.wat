;; The parent traps while a child spins forever in a bare guest loop. The
;; group trap must reach the child on both backends, so the process reports
;; the trap (status 1) instead of waiting for a thread that never finishes.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (memory (export "memory") 1 8)
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))
  (func (export "wasi_thread_start") (param i32 i32)
    i32.const 0
    i32.const 1
    i32.atomic.rmw.add
    drop
    (loop $spin
      br $spin))
  (func (export "_start")
    i32.const 0
    call $thread_spawn
    i32.const 0
    i32.le_s
    if
      unreachable
    end
    (block $ready
      (loop $wait
        i32.const 0
        i32.atomic.load align=4
        i32.const 1
        i32.ge_u
        br_if $ready
        br $wait))
    unreachable))
