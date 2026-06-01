(module
    ;; Triangle CFG with a store only in the taken arm.
    ;; The head load captures the initial tag, the single arm overwrites
    ;; memory, and the merge block reloads the same address.
    ;; A stale forwarded head load would ignore the arm-local store.
    (memory 1)

    (func $triangle_store (result i32)
        (local $head i32)

        (i32.store
            (i32.const 64)
            (i32.const 0x73900410))
        (local.set $head
            (i32.load
                (i32.const 64)))

        (if (i32.const 1)
            (then
                (i32.store
                    (i32.const 64)
                    (i32.const 0x73900420))))

        (i32.xor
            (i32.load
                (i32.const 64))
            (i32.and
                (local.get $head)
                (i32.const 0))))

    (export "triangle_store" (func $triangle_store))
)
