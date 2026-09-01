(module
  (memory (export "memory") 1 1)
  (func (export "_start")
    i32.const 65536
    i32.const 1
    memory.atomic.notify
    drop))
