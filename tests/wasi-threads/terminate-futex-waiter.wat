;; A child parks in the guest futex with an indefinite-looking (1s)
;; timeout. The parent calls `proc_exit(5)`; group termination must cancel the
;; guest futex wait so the child unwinds immediately. If the wakeup is missed
;; the wait times out first and the child prints, which fails the fixture's
;; empty-stdout expectation on both the interpreter and AOT backends.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1 8)
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))
  (data (i32.const 128) "futex-not-interrupted\0a")
  (func (export "wasi_thread_start") (param i32 i32)
    ;; Publish readiness at address 0, then park on address 16.
    i32.const 0
    i32.const 1
    i32.atomic.rmw.add
    drop
    i32.const 16
    i32.const 0
    i64.const 1000000000
    memory.atomic.wait32
    drop
    ;; Only reachable when the wait was not cancelled by termination.
    i32.const 96
    i32.const 128
    i32.store
    i32.const 100
    i32.const 22
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
