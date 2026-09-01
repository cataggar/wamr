(module
  (memory (export "memory") 1 1)
  (func (export "_start")
    i32.const 0
    i32.const 5
    i32.atomic.store offset=1 align=4))
