(module
    ;; Tail calls are control-flow barriers for forwarding.
    ;; The exported function loads the initial tag, overwrites it, then
    ;; return_call transfers directly to a callee that reloads memory.
    ;; Reusing the pre-tail load in the callee would return the stale tag
    ;; instead of the value stored immediately before the tail call.
    (memory 1)

    (func $tail_target (result i32)
        (local $tag i32)

        (local.set $tag
            (i32.load
                (i32.const 144)))
        (i32.xor
            (local.get $tag)
            (i32.const 0)))

    (func $tail_call_barrier (result i32)
        (i32.store
            (i32.const 144)
            (i32.const 0x73900910))
        (drop
            (i32.load
                (i32.const 144)))
        (i32.store
            (i32.const 144)
            (i32.const 0x73900920))
        (return_call $tail_target))

    (export "tail_call_barrier" (func $tail_call_barrier))
)
