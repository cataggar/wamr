;; A child inherits the parent's Preview-1 descriptor/preopen table and can
;; both inspect fd 3 and write through fd 1.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "sched_yield" (func $sched_yield (result i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_prestat_get"
    (func $fd_prestat_get (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_prestat_dir_name"
    (func $fd_prestat_dir_name (param i32 i32 i32) (result i32)))
  (memory (export "memory") 1 8)
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))
  (data (i32.const 256) "child-shared-fd\0a")

  (func $assert (param $condition i32)
    local.get $condition
    i32.eqz
    if
      unreachable
    end)

  (func (export "wasi_thread_start") (param i32 i32)
    i32.const 320
    i32.const 3
    i32.const 384
    call $fd_prestat_get
    i32.store
    i32.const 324
    i32.const 3
    i32.const 392
    i32.const 7
    call $fd_prestat_dir_name
    i32.store

    i32.const 336
    i32.const 256
    i32.store
    i32.const 340
    i32.const 16
    i32.store
    i32.const 328
    i32.const 1
    i32.const 336
    i32.const 1
    i32.const 344
    call $fd_write
    i32.store
    i32.const 332
    i32.const 1
    i32.atomic.store align=4)

  (func (export "_start") (local $spins i32)
    i32.const 0
    call $thread_spawn
    i32.const 0
    i32.gt_s
    call $assert
    (block $done
      (loop $wait
        i32.const 332
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
        unreachable))
    i32.const 320
    i32.load
    i32.eqz
    call $assert
    i32.const 324
    i32.load
    i32.eqz
    call $assert
    i32.const 328
    i32.load
    i32.eqz
    call $assert
    i32.const 344
    i32.load
    i32.const 16
    i32.eq
    call $assert
    i32.const 392
    i32.load
    i32.const 1954048358
    i32.eq
    call $assert
    i32.const 396
    i32.load8_u
    i32.const 117
    i32.eq
    call $assert
    i32.const 398
    i32.load8_u
    i32.const 101
    i32.eq
    call $assert))
