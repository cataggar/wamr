(module
  (import "wasi" "thread-spawn" (func $thread_spawn (param i32) (result i32)))
  (func (export "_start")
    i32.const 305419896
    call $thread_spawn
    i32.const -1
    i32.ne
    if
      unreachable
    end))
