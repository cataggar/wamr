//! Rust half of the `mixed-zig-rust-calc` example: a `wasm32-wasip1`
//! binary that imports `docs:adder/add@0.1.0` via `wit-bindgen` and
//! prints two sample sums to stdout. After `wasm-tools component
//! embed/new` (with the wasi-preview1 adapter) it becomes a Component
//! Model command component whose only non-WASI import is the adder
//! interface — letting `wasm-tools compose` link it against the Zig
//! adder in `../../zig-adder`.

wit_bindgen::generate!({
    path: "wit",
    world: "app",
    generate_all,
});

use docs::adder::add::add;

#[no_mangle]
pub extern "C" fn _start() {
    println!("40 + 2 = {}", add(40, 2));
    println!("100 + 200 = {}", add(100, 200));
}
