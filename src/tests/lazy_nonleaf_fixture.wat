(module
  (type (func (param i32) (result i32)))

  (func $eager_entry (export "eager_entry") (param i32) (result i32)
    (drop (ref.func $eager_entry))
    (local.get 0)
    (call $lazy_add))

  (func $lazy_add (param i32) (result i32)
    (local.get 0)
    (i32.const 1)
    (i32.add))

  (func $eager_callee (param i32) (result i32)
    (drop (ref.func $eager_callee))
    (local.get 0)
    (i32.const 2)
    (i32.mul))

  (func $lazy_to_eager (export "lazy_to_eager") (param i32) (result i32)
    (local.get 0)
    (call $eager_callee))

  (func $nested_callee (param i32) (result i32)
    (local.get 0)
    (i32.const 5)
    (i32.add))

  (func $nested_tail (export "nested_tail") (param i32) (result i32)
    (local.get 0)
    (return_call $nested_callee))

  (func $unused_leaf (param i32) (result i32)
    (local.get 0)
    (i32.const 7)
    (i32.add))

  (func $unused_nonleaf (export "unused_nonleaf") (param i32) (result i32)
    (local.get 0)
    (call $unused_leaf)))
