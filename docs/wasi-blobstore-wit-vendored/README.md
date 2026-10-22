# `wasi:blobstore@0.2.0-draft` WIT pin

Vendored copy of the WIT for the `wasi:blobstore@0.2.0-draft` host
adapter shipped by this repo (`#972 E2`). The interfaces:

* [`blobstore.wit`](blobstore.wit) — `create-container`, `get-container`,
  `delete-container`, `container-exists`, `copy-object`, `move-object`.
* [`container.wit`](container.wit) — `container.{name,info,get-data,write-data,list-objects,delete-object,delete-objects,has-object,object-info,clear}`
  and `stream-object-names.{read-stream-object-names,skip-stream-object-names}`.
* [`types.wit`](types.wit) — `container-metadata`, `object-metadata`,
  `object-id`, `outgoing-value`, and `incoming-value` bridging to
  `wasi:io/streams@0.2.0.{input-stream,output-stream}`.
* [`world.wit`](world.wit) — assembles the three interfaces into the
  `wasi:blobstore/imports` world the host satisfies.

## Pinned commit

These files mirror the upstream
[`WebAssembly/wasi-blobstore`](https://github.com/WebAssembly/wasi-blobstore)
repo at tag `v0.2.0-draft` (commit
[`005c4252bf6d9d9a487f817675db91a7eb942f01`](https://github.com/WebAssembly/wasi-blobstore/tree/v0.2.0-draft/wit)).
The package declaration is `wasi:blobstore@0.2.0-draft`; the host adapter
matches import names prefixed `wasi:blobstore/{blobstore,container,types}`
for both unversioned and `@0.2.*` / `@0.2.0-draft` imports.
