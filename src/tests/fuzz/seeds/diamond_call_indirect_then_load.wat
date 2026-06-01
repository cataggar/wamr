(module
    ;; Diamond with an indirect-call barrier on one branch.
    ;; The head writes a tagged value, the taken arm mutates it through a
    ;; table dispatch, and the merge block reloads memory.
    ;; Forwarding the head value across the indirect call would return the
    ;; stale tag instead of the branch-written tag.
    (type $barrier (func))
    (memory 1)
    (table 1 funcref)
    (elem (i32.const 0) $write_indirect)

    (func $write_indirect
        (i32.store
            (i32.const 16)
            (i32.const 0x73900122)))

    (func $diamond_call_indirect_then_load (result i32)
        (i32.store
            (i32.const 16)
            (i32.const 0x73900111))

        (if (i32.const 1)
            (then
                (call_indirect (type $barrier)
                    (i32.const 0)))
            (else
                (i32.store
                    (i32.const 16)
                    (i32.const 0x73900133))))

        (i32.load
            (i32.const 16)))

    (export "diamond_call_indirect_then_load" (func $diamond_call_indirect_then_load))
)
