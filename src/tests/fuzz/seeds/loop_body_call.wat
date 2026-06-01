(module
    ;; Loop body contains a direct-call barrier between two loads.
    ;; The first load in the preheader sees the initial tag, then the
    ;; callee stores the loop tag on every trip.
    ;; Reusing the preheader load after the loop would miss the call's
    ;; memory side effect.
    (memory 1)

    (func $write_from_body
        (i32.store
            (i32.const 48)
            (i32.const 0x73900320)))

    (func $loop_body_call (result i32)
        (local $i i32)
        (local $before i32)

        (i32.store
            (i32.const 48)
            (i32.const 0x73900310))
        (local.set $before
            (i32.load
                (i32.const 48)))
        (local.set $i
            (i32.const 2))

        (block $exit
            (loop $again
                (call $write_from_body)
                (br_if $again
                    (local.tee $i
                        (i32.sub
                            (local.get $i)
                            (i32.const 1))))))

        (i32.xor
            (i32.load
                (i32.const 48))
            (i32.and
                (local.get $before)
                (i32.const 0))))

    (export "loop_body_call" (func $loop_body_call))
)
