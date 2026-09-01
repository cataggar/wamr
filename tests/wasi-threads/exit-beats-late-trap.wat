;; First-wins ordering, the other way round: the parent's `proc_exit(6)` wins
;; and a sibling that traps afterwards must not turn the result into status 1.
;; The child waits for the parent's release flag, spins briefly so the exit
;; claim is published first, then traps.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (memory (export "memory") 1 8)
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))
  (func (export "wasi_thread_start") (param i32 i32)
    (local $i i32)
    i32.const 0
    i32.const 1
    i32.atomic.rmw.add
    drop
    (block $go
      (loop $spin
        i32.const 4
        i32.atomic.load align=4
        i32.const 1
        i32.ge_u
        br_if $go
        br $spin))
    i32.const 0
    local.set $i
    (block $done
      (loop $delay
        local.get $i
        i32.const 1
        i32.add
        local.tee $i
        i32.const 2000000
        i32.lt_u
        br_if $delay
        br $done))
    unreachable)
  (func (export "_start")
    i32.const 0
    call $thread_spawn
    i32.const 0
    i32.le_s
    if
      unreachable
    end
    (block $ready
      (loop $spin
        i32.const 0
        i32.atomic.load align=4
        i32.const 1
        i32.ge_u
        br_if $ready
        br $spin))
    i32.const 4
    i32.const 1
    i32.atomic.store align=4
    i32.const 6
    call $proc_exit))
