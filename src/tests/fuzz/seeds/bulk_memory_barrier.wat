(module
    ;; Bulk-memory operations act as barriers for ordinary loads.
    ;; A passive segment initializes bytes at a scratch address, memory.copy
    ;; moves the tagged word over the original slot, and memory.fill touches
    ;; a neighboring slot.
    ;; Forwarding the first load through init/copy/fill would miss the
    ;; copied tag at the merge reload.
    (memory 1)
    (data $payload "\20\08\90\73")

    (func $bulk_memory_barrier (result i32)
        (local $before i32)

        (i32.store
            (i32.const 128)
            (i32.const 0x73900810))
        (local.set $before
            (i32.load
                (i32.const 128)))

        (memory.init $payload
            (i32.const 132)
            (i32.const 0)
            (i32.const 4))
        (memory.copy
            (i32.const 128)
            (i32.const 132)
            (i32.const 4))
        (memory.fill
            (i32.const 136)
            (i32.const 0x55)
            (i32.const 4))

        (i32.xor
            (i32.load
                (i32.const 128))
            (i32.and
                (local.get $before)
                (i32.const 0))))

    (export "bulk_memory_barrier" (func $bulk_memory_barrier))
)
