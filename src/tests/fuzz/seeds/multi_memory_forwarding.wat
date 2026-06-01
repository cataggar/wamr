(module
    ;; Multi-memory seed keeps two memories live while forwarding from one.
    ;; Loads from memory $a bracket stores to memory $b and then memory $a,
    ;; so aliases must be tracked per memory index.
    ;; Forwarding the first $a load across the later $a store, or confusing
    ;; it with $b, would return the wrong tag.
    (memory $a 1)
    (memory $b 1)

    (func $multi_memory_forwarding (result i32)
        (local $first_a i32)

        (i32.store $a
            (i32.const 0)
            (i32.const 0x73901010))
        (i32.store $b
            (i32.const 0)
            (i32.const 0x73901020))
        (local.set $first_a
            (i32.load $a
                (i32.const 0)))

        (i32.store $b
            (i32.const 0)
            (i32.const 0x73901030))
        (i32.store $a
            (i32.const 0)
            (i32.const 0x73901040))

        (i32.xor
            (i32.load $a
                (i32.const 0))
            (i32.and
                (local.get $first_a)
                (i32.const 0))))

    (export "multi_memory_forwarding" (func $multi_memory_forwarding))
)
