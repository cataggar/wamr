# Malformed Test Cases
This folder contains malformed test cases collected from github and peach fuzzer. More cases should be added to here when found.

# Note
Malformed tests are currently for testing running mode of interpreter including classic interpreter and fast interpreter.

# Test
Use the provided Zig tool to test a wasm runtime from this directory:
``` shell
zig run malformed_test.zig -- -r "path/to/runtime"
```

For example you can test wasmtime with:
``` shell
zig run malformed_test.zig -- -r "/usr/bin/wasmtime"
```
If no args are passed to the tool, the default runtime `iwasm` will be used.
