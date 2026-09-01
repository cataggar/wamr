;; First-wins ordering: the child publishes readiness and traps, so the trap
;; claims the group's terminal outcome. The parent then spins long enough for
;; that claim to be published before calling `proc_exit(0)`, which must be
;; ignored — the process reports the trap (status 1), never a success exit.
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
    unreachable)
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
    ;; Give the child's trap time to be recorded as the terminal outcome.
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
    i32.const 0
    call $proc_exit))
