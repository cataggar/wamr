;; A child blocks in `poll_oneoff` on a one-second monotonic clock while the
;; parent calls `proc_exit(5)`. Group termination must interrupt the blocking
;; host wait at a slice boundary; a missed wakeup lets the clock fire and the
;; child prints, breaking the empty-stdout expectation on both backends.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (import "wasi_snapshot_preview1" "poll_oneoff"
    (func $poll_oneoff (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1 8)
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))
  (data (i32.const 128) "poll-not-interrupted\0a")
  (func (export "wasi_thread_start") (param i32 i32)
    ;; subscription record at 256: userdata=0, tag=0 (clock),
    ;; clock_id=1 (monotonic) at +16, timeout=1s at +24, flags=0 (relative).
    i32.const 272
    i32.const 1
    i32.store
    i32.const 280
    i64.const 1000000000
    i64.store
    i32.const 0
    i32.const 1
    i32.atomic.rmw.add
    drop
    i32.const 256
    i32.const 512
    i32.const 1
    i32.const 600
    call $poll_oneoff
    drop
    ;; Only reachable when the poll was not interrupted by termination.
    i32.const 96
    i32.const 128
    i32.store
    i32.const 100
    i32.const 21
    i32.store
    i32.const 1
    i32.const 96
    i32.const 1
    i32.const 104
    call $fd_write
    drop)
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
      (loop $spin
        i32.const 0
        i32.atomic.load align=4
        i32.const 1
        i32.ge_u
        br_if $ready
        br $spin))
    ;; Let the child reach its blocking wait before termination starts, so
    ;; the fixture exercises the wakeup and not the pre-park guard.
    i32.const 0
    local.set $i
    (block $settled
      (loop $delay
        local.get $i
        i32.const 1
        i32.add
        local.tee $i
        i32.const 100000
        i32.lt_u
        br_if $delay
        br $settled))
    i32.const 5
    call $proc_exit))
