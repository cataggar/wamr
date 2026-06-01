(module
    ;; Nested ifs mix loads, stores, and a call-side effect.
    ;; The outer arm performs a load, an inner arm calls a mutator, and a
    ;; sibling nested arm contains a dead store shape for CFG pressure.
    ;; Forwarding any dominating load through the nested call/store mix
    ;; would return a stale tag at the final reload.
    (memory 1)

    (func $write_nested
        (i32.store
            (i32.const 80)
            (i32.const 0x73900530)))

    (func $nested_ifs_mixed (result i32)
        (local $seen i32)

        (i32.store
            (i32.const 80)
            (i32.const 0x73900510))

        (if (i32.const 1)
            (then
                (local.set $seen
                    (i32.load
                        (i32.const 80)))
                (if (i32.const 1)
                    (then
                        (call $write_nested))
                    (else
                        (i32.store
                            (i32.const 80)
                            (i32.const 0x73900520))))
                (if (i32.const 0)
                    (then
                        (i32.store
                            (i32.const 84)
                            (local.get $seen))))))

        (i32.load
            (i32.const 80)))

    (export "nested_ifs_mixed" (func $nested_ifs_mixed))
)
