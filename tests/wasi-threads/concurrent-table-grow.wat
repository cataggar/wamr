;; A child repeatedly dispatches through slot 0 while the parent grows the
;; shared table one entry at a time. AOT clones must never retain a relocated
;; native backing pointer or observe a published length before initialization.
(module
  (type $callee_type (func (result i32)))
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "sched_yield" (func $sched_yield (result i32)))
  (memory (export "memory") 1 8)
  (table (export "__indirect_function_table") 1 funcref)
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))
  (elem (i32.const 0) $callee)

  (func $callee (result i32)
    i32.const 42)

  (func (export "wasi_thread_start") (param i32 i32)
    (block $done
      (loop $dispatch
        i32.const 0
        call_indirect (type $callee_type)
        i32.const 42
        i32.ne
        if
          unreachable
        end
        i32.const 0
        i32.atomic.load align=4
        br_if $done
        call $sched_yield
        drop
        br $dispatch))
    i32.const 4
    i32.const 1
    i32.atomic.store align=4)

  (func (export "_start") (local $grown i32) (local $spins i32)
    i32.const 8
    i32.const 0
    i32.const 1
    i32.atomic.rmw.cmpxchg align=4
    i32.const 0
    i32.ne
    if
      unreachable
    end

    i32.const 0
    call $thread_spawn
    i32.const 0
    i32.le_s
    if
      unreachable
    end

    (block $grown_all
      (loop $grow
        ref.null func
        i32.const 1
        table.grow 0
        i32.const -1
        i32.eq
        if
          unreachable
        end
        call $sched_yield
        drop
        local.get $grown
        i32.const 1
        i32.add
        local.tee $grown
        i32.const 2047
        i32.lt_u
        br_if $grow))

    i32.const 0
    i32.const 1
    i32.atomic.store align=4

    (block $done
      (loop $wait
        i32.const 4
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
