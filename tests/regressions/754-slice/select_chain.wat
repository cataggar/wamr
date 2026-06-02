;; Reproduces the chained-select pattern from js::math_min and
;; prints the result via wasi fd_write.
(module
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)

  ;; Mirrors js::math_min inner loop for (cur, val) → returns next min.
  (func $minlike (param $val f64) (param $cur f64) (result f64)
    local.get $val
    local.get $val
    local.get $val
    local.get $cur
    local.get $cur
    local.get $val
    f64.eq
    select
    local.get $cur
    local.get $val
    i64.reinterpret_f64
    i64.const -9223372036854775808
    i64.eq
    select
    local.get $cur
    local.get $val
    f64.gt
    select
    local.get $val
    local.get $val
    f64.ne
    select
  )

  ;; Convert an f64 to its bit pattern as hex (16-char fixed length).
  (func $hex64 (param $bits i64) (param $out i32)
    (local $i i32)
    (local $nib i32)
    i32.const 15
    local.set $i
    loop $L
      local.get $out
      local.get $i
      i32.add
      local.get $bits
      i32.wrap_i64
      i32.const 15
      i32.and
      local.tee $nib
      i32.const 10
      i32.lt_s
      if (result i32)
        i32.const 48
        local.get $nib
        i32.add
      else
        i32.const 87
        local.get $nib
        i32.add
      end
      i32.store8
      local.get $bits
      i64.const 4
      i64.shr_u
      local.set $bits
      local.get $i
      i32.const 1
      i32.sub
      local.tee $i
      i32.const -1
      i32.ne
      br_if $L
    end
  )

  ;; Write "minlike(3,11)=<hex16>\nminlike(11,3)=<hex16>\n" to stderr.
  (func $_start (export "_start")
    ;; layout: 0x100..0x110 hex buf #1, 0x110..0x120 hex buf #2,
    ;;         0x200 iovec[0].buf=0x300 len=L, iovec[1] etc, 0x400 nwritten
    ;; Use a simpler format: just print the two hex values.
    ;; Prefix: "min(3,11)=0x"
    i32.const 0x300
    i32.const 0x6d  ;; 'm'
    i32.store8
    i32.const 0x301
    i32.const 0x69  ;; 'i'
    i32.store8
    i32.const 0x302
    i32.const 0x6e  ;; 'n'
    i32.store8
    i32.const 0x303
    i32.const 0x3d  ;; '='
    i32.store8

    ;; Compute minlike(3, 11) and write its bits at 0x304..0x314
    f64.const 3.0
    f64.const 11.0
    call $minlike
    i64.reinterpret_f64
    i32.const 0x304
    call $hex64

    ;; Newline at 0x314
    i32.const 0x314
    i32.const 0x0a
    i32.store8

    ;; Prefix #2: "min2="
    i32.const 0x315
    i32.const 0x6d
    i32.store8
    i32.const 0x316
    i32.const 0x69
    i32.store8
    i32.const 0x317
    i32.const 0x6e
    i32.store8
    i32.const 0x318
    i32.const 0x32
    i32.store8
    i32.const 0x319
    i32.const 0x3d
    i32.store8

    ;; Compute minlike(11, 3) and write its bits at 0x31a..0x32a
    f64.const 11.0
    f64.const 3.0
    call $minlike
    i64.reinterpret_f64
    i32.const 0x31a
    call $hex64

    ;; Newline at 0x32a
    i32.const 0x32a
    i32.const 0x0a
    i32.store8

    ;; iovec: {buf=0x300, len=0x2b}
    i32.const 0x200
    i32.const 0x300
    i32.store
    i32.const 0x204
    i32.const 0x2b
    i32.store

    ;; fd_write(stderr=2, iovs=0x200, iovs_len=1, nwritten=0x400)
    i32.const 2
    i32.const 0x200
    i32.const 1
    i32.const 0x400
    call $fd_write
    drop
  )
)
