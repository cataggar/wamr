(module
    ;; Loop backedge where the latch reloads after a body store.
    ;; A preheader load observes the initial tag, but each trip writes a
    ;; new tag before the latch load and conditional backedge.
    ;; Forwarding the preheader value around the loop would hide the body
    ;; store and return the stale tag.
    (memory 1)

    (func $loop_backedge_barrier (result i32)
        (local $i i32)
        (local $seen i32)

        (i32.store
            (i32.const 32)
            (i32.const 0x73900210))
        (local.set $seen
            (i32.load
                (i32.const 32)))
        (local.set $i
            (i32.const 2))

        (block $exit
            (loop $again
                (i32.store
                    (i32.const 32)
                    (i32.const 0x73900220))
                (local.set $seen
                    (i32.load
                        (i32.const 32)))
                (br_if $again
                    (local.tee $i
                        (i32.sub
                            (local.get $i)
                            (i32.const 1))))))

        (local.get $seen))

    (export "loop_backedge_barrier" (func $loop_backedge_barrier))
)
