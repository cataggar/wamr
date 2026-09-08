;; A child publishes readiness, then makes forward progress only through a
;; true tail-recursive call. There is no loop/back-edge or host call after
;; readiness, so AOT can interrupt it promptly only at function entry (#963).
;; A sibling waits for readiness and claims the first terminal outcome with
;; proc_exit(7). The bounded runner also requires completion within 1500 ms,
;; comfortably before ThreadManager's 2-second teardown fallback.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (memory (export "memory") 1 8)
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))

  (func $recurse (result i32)
    return_call $recurse)

  (func (export "wasi_thread_start") (param i32 i32)
    local.get 1
    i32.eqz
    if
      i32.const 0
      i32.const 1
      i32.atomic.store align=4
      call $recurse
      drop
      return
    end

    (block $ready
      (loop $wait
        i32.const 0
        i32.atomic.load align=4
        br_if $ready
        br $wait))
    i32.const 7
    call $proc_exit)

  (func (export "_start")
    i32.const 0
    call $thread_spawn
    i32.const 0
    i32.le_s
    if
      unreachable
    end
    i32.const 1
    call $thread_spawn
    i32.const 0
    i32.le_s
    if
      unreachable
    end))
