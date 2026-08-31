;; The host allocates an imported memory once and every child clone retains the
;; same shared MemoryInstance.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "sched_yield" (func $sched_yield (result i32)))
  (import "env" "memory" (memory 1 8))
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))
  (func (export "wasi_thread_start") (param i32 i32)
    i32.const 0
    i32.const 1
    i32.atomic.store align=4)
  (func (export "_start") (local $spins i32)
    i32.const 0
    call $thread_spawn
    i32.const 0
    i32.le_s
    if
      unreachable
    end
    (block $done
      (loop $wait
        i32.const 0
        i32.atomic.load align=4
        br_if $done
        call $sched_yield
        drop
        local.get $spins
        i32.const 1
        i32.add
        local.tee $spins
        i32.const 100000
        i32.lt_u
        br_if $wait
        unreachable))))
