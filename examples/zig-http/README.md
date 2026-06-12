# zig-http

A minimal `wasi:http` component written in Zig. It exports
`wasi:http/incoming-handler@0.2.6` and mirrors the bytecodealliance
[Rust HTTP-in-components tutorial](https://component-model.bytecodealliance.org/language-support/using-http-in-components/rust.html):

* `GET /`       → `200 "Hello, world!\n"`
* anything else → `404`

The canonical-ABI plumbing lives in the shared `wasi_http` helper
(`src/guest/wasi_http.zig`).

## Build

```sh
zig build examples
```

Produces `zig-out/examples/zig-http.wasm`.

## Run

```sh
# wamr
wamrc run zig-out/examples/zig-http.wasm -- --listen=127.0.0.1:8080
```

```console
$ curl -i http://127.0.0.1:8080/
HTTP/1.1 200 OK
Content-Length: 14
Connection: close

Hello, world!
```

```
$ curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/missing
404
```
