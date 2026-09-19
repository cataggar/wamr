# Source-pinned Unikraft images

This document describes the implemented, compiler-free tiny-AOT image path for
[WAMR #1060](https://github.com/cataggar/wamr/issues/1060): build an EFI
executable and raw disk, finalize a standalone QCOW2, accept that exact QCOW2
twice in QEMU, derive a fixed VHD from the accepted QCOW2, and accept the
complete VHD twice through QEMU's `vpc` path. It documents the merged
[cataggar/unikraft#177](https://github.com/cataggar/unikraft/pull/177)
interface, not a proposed `wamr image` command.

This is narrower than arbitrary Wasm image production. It does not qualify new
modules, CoreMark, JIT, snapshots, performance, or a reusable Azure deployment.
The lower-level WAMR library boundary remains documented in
[unikraft-aot.md](unikraft-aot.md).

## Repository ownership and supported profile

The implementation intentionally spans three repositories:

* **WAMR** owns the workload and runtime contract: the compiler-free AOT
  library, `unikraft-x86_64` host compiler profile, embedded tiny fixture,
  answer/growth/trap expectations, and runtime identities.
* **Unikraft** owns the `appwamraot` application, native image build,
  packaging, QEMU/KVM execution, six-mode CI, immutable handoff/public bundle,
  validators, candidate plan, and the separately guarded direct-Azure adapter.
* **Miz** owns raw/QCOW2/fixed-VHD formats, conversion, and structural image
  inspection. Unikraft calls the pinned native Miz library; it does not
  implement a second converter.

The only supported image profile here is:

| Layer | Closed selection |
| --- | --- |
| Workload | `tiny`: `answer()` returns 42, `memory.grow(1)` returns the previous two-page size, and `trap()` produces the expected unreachable terminal result |
| WAMR build | `-Dprofile=unikraft-aot`, `ReleaseSafe`, x86_64 freestanding, compiler-free |
| AOT compilation | `wamrc compile --target=x86_64 --profile=unikraft-x86_64` |
| Unikraft application/profile | `appwamraot` / `hyperv-x86_64-efi-wamr` |
| Image-chain profile | `qcow2-derived-vhd` |
| Guest features | no guest compiler, JIT, interpreter, Component Model, threads, filesystem, network, minimal WASI, or CoreMark |

The implementation-selected WAMR and native profiles are invoked by
`support/apps/wamr-aot/prepare.py` and `build-image.py`; they are not extra
user-selectable image CLI options. Optional sampler, snapshot, JIT, CoreMark,
and benchmark paths in the wider source tree are outside this contract. Their
tracking issues remain
[#1044](https://github.com/cataggar/wamr/issues/1044),
[#1045](https://github.com/cataggar/wamr/issues/1045), and
[#1046](https://github.com/cataggar/wamr/issues/1046).

## Exact pins and accepted reference

Do not replace the supported WAMR SDK with the current WAMR branch head.

| Input | Exact identity |
| --- | --- |
| WAMR SDK | `a53205d77be3b880eb8f8b96679512ba58e2331a` |
| Unikraft merged implementation | `2ae4ce9edf1be141fc3c44c6ff6d2651cd76be20` ([PR #177](https://github.com/cataggar/unikraft/pull/177), completing [issue #170](https://github.com/cataggar/unikraft/issues/170)) |
| Unikraft implementation tree | `c374caf47a9cee306aca53bef1c51ac76cc0d4ff` |
| Miz | `669a27982b376311f558e820b69e9a692735b0cd` |
| Miz Zig package hash | `miz-0.2.0-Z3lHlD--2gAdGiguNwbjjdjBmv2f8QlAcwHYRw1De0Sx` |
| Zig | `0.16.0` |
| LLVM tools in the hosted lane | `llvm-tools-22.1.8-x86_64-linux.tar.xz@llvm-zig-22.1.8` |
| QEMU in the hosted lane | `v11.0.50-z.7`, archive SHA-256 `f8b9cc818959f95326010c95dad644177ebb0cbb0feef3db9528c4434855e397` |

The accepted public reference is
[run 35472031857, attempt 1](https://github.com/cataggar/unikraft/actions/runs/35472031857/attempts/1),
job `wamr-native-compute`, artifact ID `10594140223`. The archive records the
tested pull-request merge checkout rather than silently substituting either
the PR head or later merged commit:

```text
source revision: c201182f0ee7455695496e3e88f1128c129ca9e2
source tree:     c374caf47a9cee306aca53bef1c51ac76cc0d4ff
inner ZIP:       8417778e4cf83ca7d37c6106c5cbd9347115b45326665116f00359570c5878a9
container:       70533ffcaaa4b68e6552874c05cedade1b5f4b0b74c3b17cd6c0a21b858448d1
```

The PR head `6127c9fd2d66ed592d912341a82de067d88350c5`, tested merge
`c201182f0ee7455695496e3e88f1128c129ca9e2`, and merged commit
`2ae4ce9edf1be141fc3c44c6ff6d2651cd76be20` all resolve to the recorded
tree. Import still requires the exact recorded source revision and tree; tree
equivalence does not permit replacing the expected revision argument.
The Actions container digest is transport metadata and is not the trusted
inner-ZIP digest.

The command lines and record names below are taken from the exact merged
[workflow](https://github.com/cataggar/unikraft/blob/2ae4ce9edf1be141fc3c44c6ff6d2651cd76be20/.github/workflows/wamr-native-compute.yaml),
[adapter documentation](https://github.com/cataggar/unikraft/blob/2ae4ce9edf1be141fc3c44c6ff6d2651cd76be20/support/build/wamr-native-ci/README.md),
[`run.py`](https://github.com/cataggar/unikraft/blob/2ae4ce9edf1be141fc3c44c6ff6d2651cd76be20/support/build/wamr-native-ci/run.py),
[`handoff.py`](https://github.com/cataggar/unikraft/blob/2ae4ce9edf1be141fc3c44c6ff6d2651cd76be20/support/build/wamr-native-ci/handoff.py),
and
[direct-compute operator boundary](https://github.com/cataggar/unikraft/blob/2ae4ce9edf1be141fc3c44c6ff6d2651cd76be20/support/azure/WAMR-DIRECT-COMPUTE.md).

## Host requirements

Image construction needs Linux, Zig 0.16.0, Python 3, Make, Bash, Bison and its
`yacc` name, Flex and its `lex` name, M4, LLVM `nm`/`objcopy`/`objdump`/
`readelf`/`strip`, and OVMF 4M code/variables files. The hosted workflow also
uses `getfacl`, authenticated package data, and records the exact firmware,
tool, dynamic-library, and package identities it consumed.

Real acceptance needs an **x86_64 host with readable and writable `/dev/kvm`**
and the pinned VPC-capable QEMU. Missing x86, KVM, OVMF, or required tools is a
failure; there is no successful skip or TCG fallback. Native build-only tests
can run on Linux aarch64, but the current Linux aarch64 operator host cannot
replace the hosted x86 KVM/QEMU acceptance with TCG. Do not run the hosted
wrapper's privilege-sensitive setup on a shared development host.

Keep build caches, downloads, private state, and output under an owned work
root. Runtime and handoff directories are mode 0700; retained files are mode
0600. The commands below use uppercase variables for caller-selected,
absolute, canonical paths. `$PRIVATE_PARENT`, `$VALIDATOR`, and `$SUPERVISOR`
are private operator-owned paths, not public artifact members.

## Implemented build and six-mode entry points

### Build the exact tiny image

At the exact Unikraft source, with the pinned WAMR commit available in the
local WAMR repository:

```sh
python3 support/apps/wamr-aot/prepare.py prepare --source "$WAMR_SOURCE"
python3 support/apps/wamr-aot/build-image.py olddefconfig
python3 support/apps/wamr-aot/build-image.py native-images
```

`prepare.py` exports only
`a53205d77be3b880eb8f8b96679512ba58e2331a`, then builds the compiler-free
runtime archive and the matching host `wamrc`. `native-images` selects
`hyperv-x86_64-efi-wamr` internally and retains the EFI, debug ELF, bootinfo,
solved configuration, compiler/runtime, wasm/cwasm, and identity records.

The stricter hosted production adapter wraps those steps in source/tool
custody and native command supervision:

```sh
python3 support/build/wamr-native-ci/run.py build \
  --runtime "$RUNTIME" \
  --wamr-source "$WAMR_SOURCE"
bash .github/scripts/hyperv-qemu-candidate-runtime.sh "$RUNTIME" compute
```

`$RUNTIME` must be a fresh absolute private root. The candidate wrapper is the
actual `wamr-native-compute` GitHub Actions entry: it requires the named job,
recorded GitHub source identity, an ordinary user, x86_64, and real KVM. It
executes `run.py boot --runtime "$RUNTIME"` inside the restricted process
tree. Do not call conversion leaves by hand to bypass sequencing.

### Private handoff and non-authorizing plan

After all six boots and final custody checks succeed:

```sh
python3 support/build/wamr-native-ci/handoff.py export \
  --runtime "$RUNTIME" \
  --output "$PRIVATE_PARENT/FRESH-image-handoff"
"$VALIDATOR" handoff \
  "$PRIVATE_PARENT/FRESH-image-handoff/bundle.json"
python3 support/build/wamr-native-ci/handoff.py plan \
  --bundle "$PRIVATE_PARENT/FRESH-image-handoff/bundle.json" \
  --output "$PRIVATE_PARENT/FRESH-unapproved-plan.json"
```

`export` and `plan` never invoke Azure. The bundle and plan remain
`authority=not_admitted`; failed or partial export state is not resumable
success.

### Public-source bundle

The public export has no runtime, output, or arbitrary operator-tree options:

```sh
python3 support/build/wamr-native-ci/handoff.py public-source-bundle
```

That command is intentionally valid only in the public
`cataggar/unikraft` `wamr-native-compute` job with its fixed recorded runtime
and GitHub context. It performs the private export and native handoff check,
copies the closed public allowlist, creates `tiny-aot-public-source.zip`,
reopens it, and reports the inner digest and source tree. It is not a general
local publication command.

### Download, verify, import, and plan the accepted public artifact

First make the recorded source commit available in the exact Unikraft
repository and check its tree:

```sh
SOURCE_SHA=c201182f0ee7455695496e3e88f1128c129ca9e2
SOURCE_TREE=c374caf47a9cee306aca53bef1c51ac76cc0d4ff
git cat-file -e "${SOURCE_SHA}^{commit}"
test "$(git rev-parse "${SOURCE_SHA}^{tree}")" = "$SOURCE_TREE"
```

Download the exact artifact ID, not an artifact selected only by name. The
following public download paths are examples; choose an owned work directory:

```sh
ARTIFACT_ID=10594140223
RUN_ID=35472031857
RUN_ATTEMPT=1
ARCHIVE_SHA256=8417778e4cf83ca7d37c6106c5cbd9347115b45326665116f00359570c5878a9
CONTAINER_DIGEST=70533ffcaaa4b68e6552874c05cedade1b5f4b0b74c3b17cd6c0a21b858448d1
DOWNLOAD_ROOT="$PWD/.d/wamr-download"

umask 077
mkdir -p "$DOWNLOAD_ROOT"
gh api -H "Accept: application/vnd.github+json" \
  "/repos/cataggar/unikraft/actions/artifacts/${ARTIFACT_ID}/zip" \
  > "$DOWNLOAD_ROOT/artifact-container.zip"
python3 - "$DOWNLOAD_ROOT/artifact-container.zip" \
  "$DOWNLOAD_ROOT/tiny-aot-public-source.zip" <<'PY'
from pathlib import Path
import os
import shutil
import sys
import zipfile

container, output = map(Path, sys.argv[1:])
with zipfile.ZipFile(container) as zipped:
    entries = zipped.infolist()
    if (
        len(entries) != 1
        or entries[0].filename != "tiny-aot-public-source.zip"
        or entries[0].is_dir()
        or not 0 < entries[0].file_size <= 512 * 1024 * 1024
    ):
        raise SystemExit("unexpected exact-artifact members")
    with zipped.open(entries[0]) as source, output.open("xb") as target:
        os.fchmod(target.fileno(), 0o600)
        shutil.copyfileobj(source, target, 65536)
        target.flush()
        os.fsync(target.fileno())
PY
printf '%s  %s\n' "$ARCHIVE_SHA256" \
  "$DOWNLOAD_ROOT/tiny-aot-public-source.zip" | sha256sum -c -
```

Import requires explicit reviewed native validator and supervisor executables.
There is no `PATH`, sibling-file, or ambient `WAMR_CI_SUPERVISOR` fallback:

```sh
python3 support/build/wamr-native-ci/handoff.py \
  import-public-source-bundle \
  --archive "$DOWNLOAD_ROOT/tiny-aot-public-source.zip" \
  --output "$PRIVATE_PARENT/FRESH-imported-image" \
  --expected-source "$SOURCE_SHA" \
  --expected-tree "$SOURCE_TREE" \
  --expected-archive-sha256 "$ARCHIVE_SHA256" \
  --run-id "$RUN_ID" \
  --run-attempt "$RUN_ATTEMPT" \
  --validator "$VALIDATOR" \
  --supervisor "$SUPERVISOR" \
  --artifact-id "$ARTIFACT_ID" \
  --container-digest "$CONTAINER_DIGEST"
"$VALIDATOR" handoff \
  "$PRIVATE_PARENT/FRESH-imported-image/bundle.json"
python3 support/build/wamr-native-ci/handoff.py plan \
  --bundle "$PRIVATE_PARENT/FRESH-imported-image/bundle.json" \
  --output "$PRIVATE_PARENT/FRESH-unapproved-plan.json"
"$VALIDATOR" candidate "$PRIVATE_PARENT/FRESH-unapproved-plan.json"
```

The import attaches the exact-ID transport record, rewrites only local file
references, and republishes `bundle.json` only after native revalidation. The
candidate command validates the version-2 admission wrapper but does not grant
authority.

## Immutable image lineage and records

The orchestrator, not a caller assertion, fixes the order:

1. Build and inspect the EFI executable.
2. Package the 66-MiB raw disk and retain both raw QEMU modes.
3. Finalize `unikraft.qcow2` as standalone QCOW2 v3, zstd-compressed, with
   64-KiB clusters and no backing file, external data file, snapshots, or
   encryption.
4. Boot that exact QCOW2 with normal x2APIC and masked x2APIC
   (`legacy-apic`), then emit `qcow2-acceptance.json`.
5. Confirm that no derived VHD exists, bind the accepted QCOW2 digest and
   capacity in the derivation intent/gate, and create
   `unikraft-derived.vhd`.
6. Boot the complete fixed VHD twice through the read-only QEMU `vpc` node.
7. Reopen the package and all inputs, bind all six boots in final inspection,
   then export/bundle/import. The raw two-mode regression remains part of
   version 2.

Within `$RUNTIME/compute`, inspect:

| Path | Meaning |
| --- | --- |
| `package/unikraft.raw` | packaged raw disk |
| `package/unikraft.qcow2` | finalized accepted-QCOW2 candidate |
| `package/qcow2-finalization.json` | native worker's finalization result |
| `package/unikraft-derived.vhd` | fixed VHD derived from the accepted QCOW2 |
| `package/fixed-vhd-derivation.json` | native worker's derivation result and footer/provenance |
| `evidence/qcow2-finalization-intent.json` | expected raw digest, size, workload identity, limits, and deadline |
| `evidence/qcow2-finalization.json` | canonical `uk.wamr.compute-qcow2-finalization` version 1 copy |
| `evidence/qcow2-acceptance.json` | exact QCOW2 identity plus the retained raw/QCOW2 boot hashes |
| `evidence/fixed-vhd-derivation-intent.json` | accepted QCOW2 digest, file bytes, and virtual capacity |
| `evidence/fixed-vhd-derivation-gate.json` | proof that acceptance exists and derived output was absent |
| `evidence/fixed-vhd-derivation.json` | canonical `uk.wamr.compute-fixed-vhd-derivation` version 1 copy |
| `evidence/final-inspection.json` | `uk.wamr.compute-image-chain-inspection` version 1; final artifacts, six modes, boot hashes, and prior record hashes |
| `evidence/result.json` | local result schema version 2, profile `qcow2-derived-vhd`, `hardware_acceptance=not_established`, `cloud_authority=not_admitted`, and `benchmark=not_measured` |
| `boot-MODE/{request.json,report.json,hyperv-efi-boot.log}` | original bounded request, report, and serial for one mode |
| `evidence/MODE-compute.json` | parsed compute result, input pins, report hashes, unchanged-input and cleanup result |

Each supervised stage also has `evidence/command-STAGE.json`; these records
bind executable roles, argv/environment/cwd, inputs, deadlines, output
digests, native supervisor identity, and cleanup/poison status.

The private `bundle.json` is `uk.wamr.local-image-handoff` version 2. The
public ZIP adds `uk.wamr.public-source-bundle` version 2 in
`public-source.json`. Import creates `uk.wamr.public-source-transport` version
2; planning creates `uk.wamr.direct-compute-admission` version 2 and a
`uk.wamr.direct-compute` version-2 candidate. Every one remains
`not_admitted`.

Version 2 is exactly 85 regular ZIP members: 26 artifact roles, 24 members
for six boot sets, 33 evidence records, and two manifests. Version 1 remains
the old 55-member contract: 17 artifact roles, four boot sets, 20 evidence
records, and two manifests. Version 1 is not reinterpreted as the
`qcow2-derived-vhd` profile.

## Accepted reference measurements

The exact public reference produced:

| Artifact | File bytes | Filesystem allocated bytes | Virtual capacity | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| EFI executable | 1,435,456 | 1,437,696 | not a disk | `08fd526ccaaca68d9138ce0738e8c109fc249264bbc4fad49591f663ee4a3f83` |
| Raw disk | 69,206,016 | 2,543,616 | 69,206,016 | `a191921fdb45f26e4055b8673a5e060541d7bb081d709f1e13d9575d1e3a6f07` |
| Standalone QCOW2 | 589,824 | 479,232 | 69,206,016 | `a6e8e53d3b49cb7e07160a81bf93ac95b553180004bd084cedbf31d21f2978fd` |
| Derived fixed VHD | 69,206,528 | 6,295,552 | 69,206,016 | `72ed33a5e09b8ee08b498fba590e3c20a59da13ddec2d8a3893b406bf6d85fb1` |

The fixed-VHD footer SHA-256 is
`588ceb1d2a8a3c8b7847a025217fab860bd7b14745d2712683b2bb37d6f27b74`.
The derivation record binds Miz revision
`669a27982b376311f558e820b69e9a692735b0cd`, parent kind `qcow2`, and parent
SHA-256
`a6e8e53d3b49cb7e07160a81bf93ac95b553180004bd084cedbf31d21f2978fd`.

All six modes passed with answer 42, two checks, the expected unreachable
terminal/detail, platform status 0, exit 0, complete process cleanup, and
unchanged inputs:

```text
raw-x2apic
raw-legacy-apic
qcow2-x2apic
qcow2-legacy-apic
vpc-x2apic
vpc-legacy-apic
```

## Interpreting size and memory fields

These quantities are deliberately separate:

* **File bytes** are the exact named file length and are covered by its
  SHA-256. For the VHD, this includes the 512-byte footer.
* **Filesystem allocated bytes** are `st_blocks * 512` when the producing
  filesystem exposes that value. They vary with filesystem, sparse-file
  handling, copy/import stage, and allocation timing. A record may instead
  say `state=unavailable` with no byte value; never infer an allocation value.
* **Virtual capacity** is the logical disk capacity presented by raw/QCOW2/VHD,
  not the host space occupied by the file.
* **Executable size** is the EFI executable's file length. Disk capacity,
  partition padding, debug ELF size, and compressed-container size are not
  executable size.
* **Guest memory** requires explicit guest counters with declared coverage.
  The six local records show the four caller-owned counters at teardown, not
  peak memory, allocator backing/overhead, or whole-guest RAM.
* **Host memory** would require a separate host-process measurement. File
  allocation and virtual disk capacity do not measure QEMU or build-process
  RAM.

The smaller QCOW2 file and allocated-byte values establish storage-format
observations only. They do not establish lower guest RAM, faster execution,
lower Azure cost, or any other performance benefit.

## Public, private, and Azure boundaries

The authorized public bundle contains only the fixed public-source image,
compiler/runtime/config identities, portable manifests, image-lineage
records, six local request/report/serial/compute sets, and bounded command
evidence. It contains no credentials, subscription or resource identifiers,
SAS values, approval document, private campaign ledger, Azure serial/capture
records, or arbitrary operator diagnostics.

Local six-mode correctness, successful public import, and candidate
validation do **not** establish Azure acceptance. Azure still consumes the
exact derived fixed VHD, and a live run requires fresh human approval for
that image, subscription, names, region/SKU, time/cost bounds, evidence
policy, and cleanup plan. The private persistent ledger, approval, attempt
directory, resource observations, cleanup proof, and failure records remain
private. The completed initial tiny-AOT run in
[cataggar/unikraft#156](https://github.com/cataggar/unikraft/issues/156)
supplies baseline image and memory observations, but its consumed authority is
not reusable. No Azure execution has been performed for WAMR #1060.

Implementation issue
[cataggar/unikraft#170](https://github.com/cataggar/unikraft/issues/170) and
merged [cataggar/unikraft#177](https://github.com/cataggar/unikraft/pull/177)
provide the image interface documented here. Original CoreMark correctness,
matched Linux/Unikraft timing and peak-memory comparison, and optional JIT
qualification remain separate in WAMR
[#1045](https://github.com/cataggar/wamr/issues/1045),
[#1046](https://github.com/cataggar/wamr/issues/1046), and
[#1044](https://github.com/cataggar/wamr/issues/1044).
