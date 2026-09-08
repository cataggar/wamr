# AOT unsigned reciprocal regression

`printf-decimal.wasm` exercises wasi-libc's decimal formatter and direct
unsigned quotient/remainder by 10. The old `computeMagicU32` accepted
`429496730 >> 32`; for `1385676899`, its quotient was one too large and the
formatter emitted byte `ff` instead of the two digits `99`.

Rebuild the fixture with:

```sh
zig cc --target=wasm32-wasi -Oz -g0 -Wl,--strip-all \
  printf-decimal.c -o printf-decimal.wasm
```

The AOT regression requires this exact stdout:

```text
{"checksum":13856768990818897060,"cases":[[1385676899,138567689,9],[2147483648,214748364,8],[3000000008,300000000,8]]}
```
