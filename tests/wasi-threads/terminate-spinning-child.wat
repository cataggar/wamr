;; A child spins forever in a bare guest loop that never calls a host
;; function while the parent calls `proc_exit(7)`. The interpreter's dispatch
;; loop polls the group interrupt; AOT code reaches the same interruption
;; point through the loop-header `VmCtx.cancel_flag` poll (#616). Without one
;; the child cannot be stopped and teardown can only expire on its deadline.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
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
    (local $i i32)
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
    i32.const 7
    call $proc_exit))
