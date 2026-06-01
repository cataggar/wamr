(module
    ;; Sibling branches place an indirect call and a direct call in the
    ;; same barrier position before the merge reload.
    ;; Both callees write the final tag so either branch has a memory side
    ;; effect that must block forwarding.
    ;; Treating only direct calls as barriers would miscompile the
    ;; indirect-call sibling.
    (type $barrier (func))
    (memory 1)
    (table 1 funcref)
    (elem (i32.const 0) $write_indirect)

    (func $write_direct
        (i32.store
            (i32.const 96)
            (i32.const 0x73900620)))

    (func $write_indirect
        (i32.store
            (i32.const 96)
            (i32.const 0x73900620)))

    (func $indirect_vs_direct_call (result i32)
        (i32.store
            (i32.const 96)
            (i32.const 0x73900610))
        (drop
            (i32.load
                (i32.const 96)))

        (if (i32.const 1)
            (then
                (call_indirect (type $barrier)
                    (i32.const 0)))
            (else
                (call $write_direct)))

        (i32.load
            (i32.const 96)))

    (export "indirect_vs_direct_call" (func $indirect_vs_direct_call))
)
