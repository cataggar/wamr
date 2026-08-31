;; Deterministic Preview-1 pthread-style contract fixture.
;; `zig build update-wasi-thread-fixtures` marks the memory shared before
;; validating and emitting the adjacent wasm; cataggar/wabt does not yet parse
;; the text-format `shared` limits keyword.
(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (import "wasi_snapshot_preview1" "sched_yield" (func $sched_yield (result i32)))

  (memory (export "memory") 1 256)
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 4096))
  (global $__tls_base (export "__tls_base") (mut i32) (i32.const 777))
  (global $__heap_base (export "__heap_base") (mut i32) (i32.const 8192))

  (func $assert (param $condition i32)
    local.get $condition
    i32.eqz
    if
      unreachable
    end)

  (func $spawn_checked (param $arg i32) (result i32)
    (local $tid i32)
    local.get $arg
    call $thread_spawn
    local.tee $tid
    i32.const 0
    i32.gt_s
    call $assert
    local.get $tid
    i32.const 536870912
    i32.lt_u
    call $assert
    local.get $tid)

  (func $wait_done (param $arg i32) (local $spins i32)
    (block $done
      (loop $wait
        local.get $arg
        i32.atomic.load offset=20 align=4
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
        unreachable)))

  (func $finish (param $arg i32)
    local.get $arg
    i32.const 1
    i32.atomic.store offset=20 align=4)

  (func (export "wasi_thread_start") (param $tid i32) (param $arg i32)
    ;; Record the runtime-provided auxiliary stack and inherited TLS state.
    local.get $arg
    global.get $__stack_pointer
    i32.store offset=8
    local.get $arg
    global.get $__tls_base
    i32.store offset=12

    ;; Mirror wasi-libc's trampoline: start_arg[0] is the guest stack pointer,
    ;; start_arg[4] is the TLS base, and start_arg remains otherwise opaque.
    local.get $arg
    i32.load
    global.set $__stack_pointer
    local.get $arg
    i32.load offset=4
    global.set $__tls_base
    local.get $arg
    global.get $__stack_pointer
    i32.store offset=44
    local.get $arg
    global.get $__tls_base
    i32.store offset=48
    local.get $arg
    local.get $tid
    i32.store offset=24

    i32.const 0
    i32.const 1
    i32.atomic.rmw.add
    drop

    ;; command 1: nested spawn using the same process/thread group.
    local.get $arg
    i32.load offset=16
    i32.const 1
    i32.eq
    if
      local.get $arg
      local.get $arg
      i32.load offset=32
      call $thread_spawn
      i32.store offset=28
    end

    local.get $arg
    call $finish)

  (func (export "_start")
    (local $i i32)
    (local $tid i32)

    global.get $__heap_base
    i32.const 8192
    i32.gt_u
    call $assert

    ;; Retain 300 immediately completed native handles. Guest-side atomic
    ;; polling is the pthread-style join; host ownership remains until group
    ;; shutdown, proving there is no 256-record cap.
    i32.const 0
    local.set $i
    (block $lifecycle_done
      (loop $lifecycle
        i32.const 1024
        i32.const 6000
        i32.store
        i32.const 1024
        i32.const 10000
        local.get $i
        i32.add
        i32.store offset=4
        i32.const 1024
        i32.const 0
        i32.store offset=16
        i32.const 1024
        i32.const 0
        i32.atomic.store offset=20 align=4

        i32.const 1024
        call $spawn_checked
        local.tee $tid
        i32.const 72
        i32.load
        i32.ne
        call $assert
        i32.const 72
        local.get $tid
        i32.store

        i32.const 1024
        call $wait_done
        i32.const 1024
        i32.load offset=44
        i32.const 6000
        i32.eq
        call $assert
        i32.const 1024
        i32.load offset=48
        i32.const 10000
        local.get $i
        i32.add
        i32.eq
        call $assert

        local.get $i
        i32.eqz
        if
          i32.const 64
          i32.const 1024
          i32.load offset=8
          i32.store
        end
        i32.const 68
        i32.const 1024
        i32.load offset=8
        i32.store

        local.get $i
        i32.const 1
        i32.add
        local.tee $i
        i32.const 300
        i32.lt_u
        br_if $lifecycle
        br $lifecycle_done))

    i32.const 0
    i32.atomic.load align=4
    i32.const 300
    i32.eq
    call $assert
    i32.const 64
    i32.load
    i32.const 68
    i32.load
    i32.ne
    call $assert

    ;; Two concurrent children receive isolated cloned globals, distinct
    ;; auxiliary stacks, and bit-preserved start_arg stack/TLS payloads.
    i32.const 2048
    i32.const 6100
    i32.store
    i32.const 2048
    i32.const 11000
    i32.store offset=4
    i32.const 2048
    i32.const 0
    i32.store offset=16
    i32.const 2048
    i32.const 0
    i32.atomic.store offset=20 align=4

    i32.const 2112
    i32.const 6200
    i32.store
    i32.const 2112
    i32.const 12000
    i32.store offset=4
    i32.const 2112
    i32.const 0
    i32.store offset=16
    i32.const 2112
    i32.const 0
    i32.atomic.store offset=20 align=4

    i32.const 2048
    call $spawn_checked
    drop
    i32.const 2112
    call $spawn_checked
    drop
    i32.const 2048
    call $wait_done
    i32.const 2112
    call $wait_done

    i32.const 2048
    i32.load offset=8
    i32.const 2112
    i32.load offset=8
    i32.ne
    call $assert
    i32.const 2048
    i32.load offset=12
    i32.const 777
    i32.eq
    call $assert
    i32.const 2112
    i32.load offset=12
    i32.const 777
    i32.eq
    call $assert
    i32.const 2048
    i32.load offset=44
    i32.const 6100
    i32.eq
    call $assert
    i32.const 2048
    i32.load offset=48
    i32.const 11000
    i32.eq
    call $assert
    i32.const 2112
    i32.load offset=44
    i32.const 6200
    i32.eq
    call $assert
    i32.const 2112
    i32.load offset=48
    i32.const 12000
    i32.eq
    call $assert
    global.get $__stack_pointer
    i32.const 4096
    i32.eq
    call $assert
    global.get $__tls_base
    i32.const 777
    i32.eq
    call $assert

    ;; Nested spawning inherits the same manager and shared process state.
    i32.const 2240
    i32.const 6400
    i32.store
    i32.const 2240
    i32.const 14000
    i32.store offset=4
    i32.const 2240
    i32.const 0
    i32.store offset=16
    i32.const 2240
    i32.const 0
    i32.atomic.store offset=20 align=4

    i32.const 2176
    i32.const 6300
    i32.store
    i32.const 2176
    i32.const 13000
    i32.store offset=4
    i32.const 2176
    i32.const 1
    i32.store offset=16
    i32.const 2176
    i32.const 0
    i32.atomic.store offset=20 align=4
    i32.const 2176
    i32.const 2240
    i32.store offset=32

    i32.const 2176
    call $spawn_checked
    local.set $tid
    i32.const 2176
    call $wait_done
    i32.const 2240
    call $wait_done
    i32.const 2176
    i32.load offset=28
    i32.const 0
    i32.gt_s
    call $assert
    i32.const 2176
    i32.load offset=28
    local.get $tid
    i32.ne
    call $assert

    i32.const 0
    i32.atomic.load align=4
    i32.const 304
    i32.eq
    call $assert))
