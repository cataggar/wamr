;; The parent returns immediately. The CLI-owned ThreadManager must keep the
;; process state, stdout descriptor, clone, ExecEnv, and native child alive
;; until the child finishes and group shutdown joins it.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1 8)
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))
  (data (i32.const 128) "child-after-parent\0a")
  (func (export "wasi_thread_start") (param i32 i32)
    (local $i i32)
    i32.const 0
    local.set $i
    (block $done
      (loop $delay
        local.get $i
        i32.const 1
        i32.add
        local.tee $i
        i32.const 1000000
        i32.lt_u
        br_if $delay
        br $done))
    i32.const 96
    i32.const 128
    i32.store
    i32.const 100
    i32.const 19
    i32.store
    i32.const 1
    i32.const 96
    i32.const 1
    i32.const 104
    call $fd_write
    drop)
  (func (export "_start")
    i32.const 256
    call $thread_spawn
    i32.const 0
    i32.le_s
    if
      unreachable
    end))
