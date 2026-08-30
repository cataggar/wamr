# Metadata-safe, crash-safe JSON replacement

Status: **DESIGN — implementation-ready; no persistence behavior changed**

Tracking: [#616 A6](https://github.com/cataggar/wamr/issues/616).

## Purpose and acceptance contract

The optional `--keyvalue-store=PATH` backend currently rewrites its complete
JSON snapshot with one `writeFile` call. A process or machine failure during
that write can leave truncated JSON, and replacing the inode naively would
discard metadata that controls access to the store. This design replaces that
write path without changing the JSON format or the `wasi:keyvalue` interface.

For every successful mutation, after any process crash and, on platforms whose
directory-sync primitive is supported, after a machine crash or power loss, a
restart must observe exactly one of:

1. the complete old JSON document with the old file's required metadata; or
2. the complete new JSON document with the old file's required metadata.

It must never observe a partial document, mixed old/new bytes, or silently
weakened permissions. "Required metadata" means:

- POSIX: permission and special mode bits, UID, GID, access ACL, and every
  extended attribute the caller can enumerate and read;
- Windows: owner SID and DACL, including whether the DACL is protected from
  inheritance. `ReplaceFileW`-preserved attributes and streams are desirable
  additional coverage, but not substitutes for explicit owner/DACL checks.

If the target does not yet exist, the new-file policy is mode `0600`, effective
UID/GID, and no ACL entry that grants effective access to another principal on
POSIX. On Windows it is the current user as owner and a protected DACL granting
full control only to that owner and `SYSTEM`. A backend that cannot establish
and verify that policy must fail rather than use inherited, potentially broader
permissions.

Content-derived metadata is intentionally not preserved: size, modification
time, change time, and file identity must change with the new content. Creation
time, DOS attributes, compression, encryption, and named streams may be
preserved by a platform replacement API, but are outside the minimum contract
until separately tested.

## Current code and scope

The current path is:

1. [`main.zig`](../../src/main.zig) parses `--keyvalue-store` and calls
   `WasiCliAdapter.setKeyvalueStorePath`.
2. [`setKeyvalueStorePath`](../../src/component/wasi_cli_adapter.zig) reads up
   to 8 MiB and parses the bucket-to-key-to-base64 JSON snapshot.
3. `bucketStoreEntry` mutates the live bucket and persisted snapshot, then
   `persistedStoreEntry` calls `flushKeyvalueStore`.
4. `bucketRemoveEntry` mutates the live bucket, but currently discards every
   `persistedRemoveEntry`/flush error with `catch {}`.
5. [`flushKeyvalueStore`](../../src/component/wasi_cli_adapter.zig) serializes
   the whole map in memory and calls `std.Io.Dir.cwd().writeFile`, which opens
   and truncates the destination directly.

The same file also contains successful restart/read-back tests, but no
crash-point, metadata, symlink, or concurrent-process coverage. The limitation
is documented in [`docs/wasi.md`](../wasi.md).

`--config-store=PATH` is related JSON input but is read-only today:
`loadComponentConfigStore` in [`main.zig`](../../src/main.zig) reads and parses
a flat string map before the adapter receives it. No config write path changes
under this design. Its loader should eventually reuse the secure
descriptor/handle-relative open helper, and a future mutable config backend may
reuse the replacement engine.

This PR changes documentation only; it does not alter the current persistence
path.

## Required architecture

Split persistence into a portable transaction layer and platform backends:

```text
keyvalue operation
    -> PersistenceTxn.apply(logical mutation)
       -> acquire process mutex and stable cross-process lock
       -> securely open parent and current target
       -> read/parse current disk snapshot
       -> apply one logical mutation to that snapshot
       -> serialize and validate complete JSON in memory
       -> AtomicFileBackend.replace(bytes, metadata policy, fault hook)
       -> publish the committed snapshot to the live adapter
```

The transaction takes a logical operation (`set`, `delete`, `increment`, CAS
swap, or an entire batch), not a precomputed stale document. It reloads the
latest snapshot after acquiring the cross-process lock. This prevents two WAMR
processes from serially overwriting each other's unrelated changes and keeps
`increment`, CAS, and batch operations atomic across cooperating writers.

The in-memory bucket and `keyvalue_persisted` maps are updated only after a
durable commit. An implementation may instead mutate copies and swap them into
place. It must not retain today's "mutate first, then possibly fail flush"
behavior, because rollback is incomplete after allocation or I/O errors.

### Backend interface

The platform-independent layer should depend on an interface equivalent to:

```zig
const ReplaceOutcome = union(enum) {
    durable,
    committed_not_durable: ReplaceFailure,
};

const ReplaceFailure = struct {
    stage: Stage,
    os_code: u32,
};

fn replace(
    parent: SecureDirHandle,
    leaf_name: []const u8,
    bytes: []const u8,
    existing: ?SecureFileHandle,
    fault: ?*FaultInjector,
) ReplaceError!ReplaceOutcome;
```

`ReplaceError` means replacement definitely did not occur. Once atomic rename
or replacement succeeds, later failures are returned as
`committed_not_durable`; callers must not assume that retrying a non-idempotent
operation is safe. The adapter reloads the visible target before returning a
`wasi:keyvalue/store.error::other` message containing the failing stage and
"commit visible; durability unknown". It must remove the current `catch {}` on
delete and propagate failures for every mutation and every item-independent
batch as one batch error. No persistence failure may be converted to guest
success.

Serialization/allocation failures occur before filesystem mutation and are
ordinary `not committed` errors. Error strings must not include JSON values or
other secret content.

## Exact replacement protocol

All names below are leaf names relative to an already-open parent directory.
No step after secure parent resolution uses the original multi-component path.

1. **Acquire locks.**
   - Take an adapter-local mutex keyed by canonical store identity.
   - Open or create the stable sidecar `.<leaf>.wamr-lock` with owner-only
     permissions, rejecting symlinks/reparse points and non-regular files.
   - Hold an exclusive OS lock for the complete read-modify-replace sequence:
     `flock(LOCK_EX)`/whole-file `fcntl` lock on POSIX and `LockFileEx` on
     Windows. Never unlink the sidecar; unlinking creates split lock domains.
   - If reliable locking is unsupported, return `UnsupportedFilesystem`.

2. **Resolve the parent securely.**
   - Reject empty leaf names, `.`/`..`, embedded separators, and paths whose
     final component is a symlink, junction, mount-point reparse point, or
     other reparse object.
   - Open every directory component without following symlinks/reparse points.
     Retain the final parent handle through cleanup and directory sync.

3. **Open and identify the target.**
   - Open an existing target without following the final link and require a
     regular disk file.
   - Record stable identity (`st_dev`/`st_ino` on POSIX,
     volume serial/file ID on Windows), size, and required metadata from that
     handle.
   - Read and parse the current JSON from the same handle. Do not stat by path
     and later reopen by path.
   - A target that appears, disappears, or changes identity despite the
     cooperative lock produces `ConcurrentModification`; retry the whole
     transaction a bounded number of times. Writers that ignore the sidecar
     lock are outside the lost-update guarantee, but the atomic replacement
     still prevents WAMR from publishing partial bytes.

4. **Apply the logical mutation and serialize.**
   - Apply exactly one guest operation (or one batch operation) to the freshly
     loaded snapshot.
   - Serialize the complete JSON into memory, enforce the existing 8 MiB load
     limit symmetrically on output, and parse the generated bytes in a test or
     validation build before opening a temp file.

5. **Create a same-directory temp file.**
   - Name it `.<leaf>.wamr-tmp.<random-128-bit-hex>`. If this exceeds the
     filesystem component limit, truncate the escaped leaf and append a
     SHA-256-derived leaf digest before the random suffix.
   - Use cryptographic OS randomness, exclusive create, owner-only access,
     no-follow/open-reparse-point flags, and a parent-relative API. Retry a
     bounded number of random collisions.
   - Same-directory placement is mandatory: it gives same-filesystem atomic
     replacement and makes the parent directory sync cover both names.

6. **Write all bytes.**
   - Loop over short writes and `EINTR`; a zero-byte write before completion is
     an error. Do not expose or rename the temp file on failure.
   - Flush file data (`fdatasync` where meaningful; `FlushFileBuffers` on
     Windows). This first barrier isolates data-write failures from metadata
     failures and is a fault-injection point.

7. **Apply required metadata to the temp handle.**
   - Existing target: apply the captured owner/group, mode, ACL, and xattrs, in
     the platform order below. Any inaccessible, unsupported, truncated, or
     unrepresentable required item is fatal.
   - New target: apply the secure new-file policy above.
   - Re-read metadata from the temp handle and compare semantically with the
     capture. Byte comparison is appropriate for opaque xattr values; ACLs and
     Windows security descriptors require canonical semantic comparison.
   - Flush the temp handle again with the full metadata-sync primitive
     (`fsync`, `F_FULLFSYNC` where selected, or `FlushFileBuffers`).

8. **Revalidate immediately before commit.**
   - For an existing target, resolve the leaf without following links and
     require the recorded identity, content digest, and required metadata. For
     a missing target, require it still to be absent. Also require the lock
     sidecar name still to identify the locked file. A mismatch restarts from
     step 3 after deleting the temp.
   - Verify the temp is still a regular file with link count one and the
     expected identity, length, content digest, and required metadata.

9. **Atomically replace.**
   - Rename/replace only through the held parent handle and only on the same
     filesystem/volume. Readers must see the old complete file until this
     single operation takes effect, then the new complete file.
   - Do not implement a two-rename "target to backup, temp to target" sequence:
     it creates a missing-target window and additional recovery states.
   - Do not fall back to truncate-and-write under any error.

10. **Sync the directory entry.**
    - Sync the held parent directory handle after replacement. A successful
      file sync before rename is not sufficient: the rename changes directory
      metadata.
    - If directory sync fails, return `committed_not_durable`, reconcile the
      adapter from the visible target, retain no claim that a retry is safe,
      and log the stage without content. Success is returned only after this
      barrier.

11. **Cleanup and unlock.**
    - Close the temp handle before or after rename as the platform requires.
      If commit did not occur, delete the temp by parent handle after verifying
      its recorded identity and regular-file/link-count properties.
    - Release the sidecar lock last. Cleanup failure is reported when commit
      did not occur; after commit it is logged without changing a durable
      success into failure unless it compromises lock correctness.

## Metadata rules and platform APIs

### Linux and other POSIX systems

Use parent-relative descriptors throughout. On Linux, prefer
[`openat2`](https://man7.org/linux/man-pages/man2/openat2.2.html) with
`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS`; fall back to
walking one component at a time with `openat(O_DIRECTORY | O_NOFOLLOW)` on
kernels without `openat2`. Use `fstat`/`fstatat(AT_SYMLINK_NOFOLLOW)`,
`openat(O_CREAT | O_EXCL | O_NOFOLLOW)`, and `renameat`.

Capture `st_mode & 07777`, `st_uid`, and `st_gid` from the open target. Enumerate
xattrs with `flistxattr` and read each value with `fgetxattr`, retrying if the
reported size changes. On Linux this includes opaque
`system.posix_acl_access`, `security.capability`, and security-label values when
the process is allowed to see them. Never silently skip a namespace or an
`EACCES`/`EPERM` result.

Apply in this order:

1. `fchown` (because ownership changes may clear set-ID bits);
2. `fchmod` with the captured mode;
3. `fsetxattr` for every captured xattr, including the ACL xattr where exposed;
4. platform ACL API when the ACL is not represented as an enumerable xattr;
5. re-read mode/owner/ACL/xattrs and require equality;
6. `fsync` the temp file.

The verify step catches ACL-mask changes induced by mode operations. Linux may
use the opaque ACL xattr for exact transfer; a libacl-based implementation may
instead use `acl_get_fd`/`acl_set_fd`, but must not apply the same ACL twice and
must compare the normalized ACL afterward.

Use `fdatasync` after data and `fsync` after metadata. After `renameat`, call
`fsync(parent_fd)`. The Linux
[`fsync(2)` documentation](https://man7.org/linux/man-pages/man2/fsync.2.html)
explicitly states that syncing a file does not sync its directory entry.

On macOS, prefer `fcopyfile(source_fd, temp_fd, ..., COPYFILE_SECURITY |
COPYFILE_XATTR)` for metadata-only copying, followed by explicit
`fchown`/`fchmod` and verification with `fstat`, `acl_get_fd_np`, and fd-based
xattr calls. Use `fcntl(F_FULLFSYNC)` for the temp file when available; failure
must be surfaced or the backend must explicitly advertise only ordinary
`fsync` durability. Rename with `renameat` and sync the parent directory.
Apple's [`copyfile(3)`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/copyfile.3.html)
API is platform-specific and therefore belongs in the Darwin backend, not the
portable layer.

`ENOTSUP`/`EOPNOTSUPP` from xattr or ACL enumeration is acceptable only when
the filesystem cannot store that metadata class at all. Failure to read or
write one item on a filesystem that supports the class is fatal. UID/GID
preservation failures (common when replacing a file owned by another user) are
fatal. Do not special-case them into a mode-only replacement.

### Windows

Open directories and files with `CreateFileW`, share delete access, and use
`FILE_FLAG_OPEN_REPARSE_POINT`; query `FileAttributeTagInfo` and reject every
reparse point. Because `CreateFileW` only protects the final component, the
Windows backend must either walk components by handles using `NtCreateFile`
with a `RootDirectory`, or provide an equivalently reviewed handle-relative
helper. Path-only validation followed by `ReplaceFileW` is not sufficient
against a parent junction swap.

Open the target with `READ_CONTROL`, `FILE_READ_ATTRIBUTES`, and stable sharing.
Capture owner and DACL from the handle with
[`GetSecurityInfo`](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-getsecurityinfo)
using `OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION`, and inspect the
returned descriptor's control bits to retain protected-versus-inherited DACL
semantics. Apply them to the temp handle with
[`SetSecurityInfo`](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-setsecurityinfo).
A null DACL grants everyone full access, so it must be preserved only when the
existing target actually has one; it must never arise from an error or missing
pointer. Compare owner SID, DACL ACEs/canonical meaning, and protected status
before commit.

For an existing target, `ReplaceFileW` with flags `0` is the documented
metadata-preserving primitive: Microsoft documents that it preserves DACLs and
other attributes and warns that either `REPLACEFILE_IGNORE_MERGE_ERRORS` or
`REPLACEFILE_IGNORE_ACL_ERRORS` may succeed without preserving ACLs. Those
flags are prohibited. See
[`ReplaceFileW`](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-replacefilew).
However, it is path-based and its documented failure codes include partially
moved states. The hardened backend should therefore prefer a handle-relative
`SetFileInformationByHandle(FileRenameInfoEx)`/native
`FileRenameInformationEx` replacement after explicitly copying and verifying
owner/DACL. `ReplaceFileW` may be enabled only after the reparse-race and every
documented partial-failure state have dedicated tests and recovery handling.

Call `FlushFileBuffers` on the temp before replacement. Windows has no
documented Win32 equivalent of POSIX `fsync(parent_directory)` with the same
power-loss contract. The Windows backend may claim atomic visibility and
owner/DACL preservation once implemented, but must remain
`durable_directory_sync = false`—and return an unsupported-durability error
when strict crash durability is requested—until a reviewed, tested NT API or
filesystem-specific contract supplies the final directory barrier. It must not
claim the full power-loss acceptance criterion merely because rename returned
success.

Require a local volume with persistent ACL support (`FILE_PERSISTENT_ACLS`) for
the metadata-preserving backend. FAT-like filesystems and remote providers
without equivalent locking, security-descriptor, replacement, and flush
semantics return `UnsupportedFilesystem`.

## Symlink and reparse-point policy

- The store target, lock file, temp file, and every parent component must be
  non-link objects opened with no-follow semantics.
- A target that is a symlink/reparse point is rejected; WAMR never replaces the
  link itself and never follows it to another object.
- Temp cleanup and orphan recovery operate by parent handle and verify file
  identity immediately before deletion.
- Hard-linked POSIX targets (`st_nlink != 1`) are rejected. Inode replacement
  would update only one name and produce surprising security/content behavior
  for the other links.
- Mount points, device files, FIFOs, sockets, directories, and Windows reparse
  tags are rejected.

## State machine

```text
UNLOCKED
  | acquire stable sidecar lock
  v
LOCKED
  | securely resolve parent + target, capture metadata, read JSON
  v
SNAPSHOT
  | apply logical mutation + serialize/validate
  v
SERIALIZED
  | exclusive same-directory temp create
  v
TEMP_CREATED
  | write all bytes
  v
DATA_WRITTEN
  | data flush
  v
DATA_SYNCED
  | apply + verify metadata
  v
METADATA_READY
  | full temp sync
  v
TEMP_DURABLE
  | revalidate target identity
  v
COMMIT_READY
  | atomic replace                 (commit point)
  v
VISIBLE_NEW
  | parent-directory sync
  v
DURABLE_NEW
  | cleanup + unlock
  v
DONE
```

Any failure before `VISIBLE_NEW` transitions to `ABORTING`: close and
identity-check/delete the temp, leave the old target untouched, then unlock.
A crash in those states leaves the old target plus at most an unreferenced temp
file. A crash at or after the commit point leaves either the old or new complete
name binding according to the filesystem's atomic-rename contract. A directory
sync success makes the new binding durable.

A directory-sync failure transitions to `COMMIT_UNKNOWN`, not `ABORTING`; the
new file may already be visible and must not be renamed back. Reopen/reconcile,
report `committed_not_durable`, and unlock.

## Failure-injection test matrix

The replacement layer must expose a test-only hook immediately before and
after every stage. Each crash test performs a real child-process write, kills
or `_exit`s at the hook without cleanup, starts a fresh process, parses the
store, and checks content plus metadata. Ordinary error tests inject the
platform error return and verify the reported commit classification.

| Injection point | Expected call result / recovery | Restart-visible content | Metadata requirement |
| --- | --- | --- | --- |
| lock open/create | not committed; no temp | old | old unchanged |
| lock acquisition | not committed; no temp | old | old unchanged |
| parent component open | not committed | old | old unchanged; link target untouched |
| target open/identity check | not committed or bounded retry | old or independently replaced complete file | WAMR changes none |
| metadata capture: stat/owner | not committed | old | old unchanged |
| metadata capture: ACL enumerate/read | not committed | old | old unchanged |
| metadata capture: each xattr enumerate/read | not committed | old | old unchanged |
| current JSON read/parse | not committed | old (possibly externally invalid) | unchanged |
| mutation/serialization/output-size check | not committed | old | unchanged |
| temp create | not committed; no usable temp | old | unchanged |
| after temp create, before write | orphan may exist | old | unchanged |
| first/middle/final data write | short/failed write aborts; orphan cleanup/recovery | old | unchanged |
| data flush (`fdatasync`/`FlushFileBuffers`) | not committed | old | unchanged |
| owner/group application | not committed | old | unchanged |
| mode application | not committed | old | unchanged |
| ACL application | not committed | old | unchanged |
| each xattr application | not committed | old | unchanged |
| metadata verification mismatch | not committed | old | unchanged; never rename weakened temp |
| full temp metadata sync | not committed | old | unchanged |
| pre-commit target revalidation | bounded retry or conflict | old or independent complete file | WAMR changes none |
| immediately before atomic replace | not committed | old | unchanged |
| atomic replace returns definite failure | not committed; inspect documented platform state | old | unchanged |
| crash inside/just after atomic replace | no call result | complete old or complete new | whichever content is named has required metadata |
| immediately after replace, before directory sync | `committed_not_durable` on injected error | complete old or complete new after machine restart; new after process-only restart | unchanged required metadata |
| directory sync call | success only on zero/supported result; otherwise `committed_not_durable` | complete old or new | unchanged required metadata |
| after successful directory sync | durable success | new | required metadata equals old |
| temp close/delete cleanup | pre-commit failure if cleanup safety is uncertain; post-commit log only | old before commit, new after durable commit | target unchanged |
| unlock/lock-handle close | report lock subsystem failure; never redo mutation | old or committed new according to commit state | unchanged |

Run the matrix for:

- replacement of a plain file;
- mode bits including group/other denial and set-ID bits where permitted;
- a nontrivial POSIX ACL;
- multiple user/trusted/security xattrs where permitted, including zero-length
  and binary values;
- a Windows owner different from the writer where privileges permit, a
  protected DACL, deny and allow ACEs, and a null DACL fixture isolated to a
  test account;
- missing-target secure creation;
- symlink in each path component, final symlink, hard link, and Windows
  junction/reparse fixtures;
- two cooperating processes writing unrelated keys, same-key last-lock-winner,
  concurrent increments, CAS conflict, and whole-batch atomicity;
- an uncooperative external replacement between snapshot and commit, which
  must be detected at revalidation when it occurs before the check;
- disk full/quota, read-only filesystem, permission denial, unsupported
  xattr/ACL, interrupted and short writes, rename sharing violation, and
  unsupported directory sync;
- orphan temp cleanup with malicious lookalike names, symlinks/reparse points,
  wrong owner, multiple links, and a currently open writer.

The minimum CI assertion for every supported backend is:

```text
restart JSON ∈ {complete_old, complete_new}
AND parse(restart JSON) succeeds
AND required_metadata(restart target) == required_metadata(original target)
```

No test may weaken the assertion to "an error was returned".

## Cleanup and recovery

Atomic replacement needs no journal or backup for content recovery. Startup,
after taking the stable sidecar lock, may remove orphan names matching the
exact generated prefix only when all of these hold:

- opened relative to the trusted parent without following links;
- regular file/reparse-free, link count one, and not the target or lock
  identity;
- owned by the store owner (or current user for a never-created store);
- older than a conservative grace period; and
- successfully locked/opened exclusively so no live writer is using it.

Anything ambiguous is left in place and logged by name only. Recovery never
promotes a temp file: without a completed atomic replacement there is no proof
that its data and metadata passed every barrier. The deterministic sidecar lock
is retained permanently and is not considered an orphan.

## Unsupported filesystems and guarantees

Capability is established by successful operations, not only a filesystem-name
allowlist. The backend returns `UnsupportedFilesystem` before commit when it
cannot provide same-directory atomic replacement, reliable cross-process
locking, required metadata enumeration/application, or the requested
durability barrier. It never falls back to direct overwrite.

Some network, FUSE, overlay, removable, and userspace filesystems may accept an
API while providing weaker server/device semantics. Initial implementation
support should be limited to filesystems exercised by the full crash matrix;
other filesystems require an explicit opt-in support entry and evidence. A
successful `fsync` can only mean the guarantee documented by the mounted
filesystem and storage stack.

Windows without a proven directory-entry durability primitive is explicitly
not a full power-loss-safe backend. It can still implement and test atomic
visibility and owner/DACL preservation, but strict mode must reject it rather
than overstate the guarantee.

## Implementation units

### Portable work that can land first

1. Extract deterministic JSON parsing/serialization and represent every guest
   mutation as one replayable transaction.
2. Add the adapter-local lock, cross-process-lock abstraction, size limit, and
   publish-after-commit/rollback-safe state handling.
3. Add `Stage`, `ReplaceOutcome`, sanitized error mapping, and remove all
   swallowed persistence errors.
4. Add a fake backend and the complete pre/post-stage failure matrix, including
   `committed_not_durable`.
5. Add common temp-name generation, orphan selection rules, and metadata
   comparison data structures.
6. Reuse the secure read helper for `setKeyvalueStorePath`; later reuse it for
   read-only `--config-store`.

These units do not claim crash safety until a real backend passes the platform
matrix. They can be reviewed independently of syscall bindings.

### POSIX-specific work

1. Secure parent walk (`openat2` on Linux plus portable `openat` fallback),
   stable lock, identity checks, temp create, `renameat`, and directory `fsync`.
2. Linux mode/UID/GID/xattr/ACL capture, application, and verification.
3. Darwin `copyfile`/ACL/xattr implementation and `F_FULLFSYNC` policy.
4. Crash-runner tests on each supported filesystem and CI operating system.

### Windows-specific work and decisions

1. Add reviewed Zig bindings for handle-relative directory traversal,
   reparse-tag queries, security descriptors, locking, temp creation, and
   handle-relative replace.
2. Decide and validate the final atomic primitive:
   `FileRenameInformationEx` after explicit security copying is preferred;
   `ReplaceFileW` requires exhaustive partial-state and parent-reparse tests.
3. Establish whether a supported NT/volume flush sequence supplies a
   documented directory-entry durability guarantee. Until then, keep strict
   crash-safe persistence unsupported on Windows.
4. Run owner/DACL and crash tests on NTFS and ReFS before advertising support.

## Non-goals

- Changing the JSON schema or adding a database/WAL.
- Providing lock-free multi-process writes.
- Preserving metadata that is supposed to describe the new content, such as
  size or modification time.
- Treating writers that ignore the sidecar protocol as transactionally
  coordinated.
- Changing persistence behavior in the design PR.
