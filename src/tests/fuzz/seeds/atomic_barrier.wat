(module
    ;; Atomics form memory barriers for load forwarding.
    ;; The function stores an initial tag, performs an atomic RMW, fences,
    ;; then uses cmpxchg to publish the final tag.
    ;; Forwarding the initial load across any atomic operation would return
    ;; a value that no sound execution can observe at the final reload.
    (memory 1 1 shared)

    (func $atomic_barrier (result i32)
        (local $before i32)

        (i32.atomic.store
            (i32.const 112)
            (i32.const 0x73900710))
        (local.set $before
            (i32.load
                (i32.const 112)))

        (drop
            (i32.atomic.rmw.add
                (i32.const 112)
                (i32.const 0x10)))
        (atomic.fence)
        (drop
            (i32.atomic.rmw.cmpxchg
                (i32.const 112)
                (i32.const 0x73900720)
                (i32.const 0x73900730)))

        (i32.xor
            (i32.load
                (i32.const 112))
            (i32.and
                (local.get $before)
                (i32.const 0))))

    (export "atomic_barrier" (func $atomic_barrier))
)
