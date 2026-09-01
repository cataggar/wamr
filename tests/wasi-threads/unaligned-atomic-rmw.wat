(module
  (memory (export "memory") 1 1)
  (func (export "_start")
    i32.const 1
    i32.const 5
    i32.atomic.rmw.add align=4
    drop))
