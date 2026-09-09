# Release preflight

The `Release` workflow supports two paths through the same build, signing,
packaging, smoke, SBOM, checksum, and provenance jobs:

| Trigger | Result |
|---|---|
| Manual `workflow_dispatch` with a version | Retained artifacts and evidence only; no tag or GitHub/PyPI release |
| Push of a `v*` tag | The verified artifacts are published through the existing release step |

After a successful local release-mode build, dispatch the preflight from the
intended candidate branch:

```sh
gh workflow run release.yml --ref <candidate-branch> -f version=3.0.0-dev.14
```

Record the run's exact `headSha`, not just the branch name. Each job checks out
that event revision, and the inventory binds the archives, verification reports,
source tarball, and release notes to it. A later candidate commit requires a new
preflight; a preflight attestation is not a substitute for tag-release provenance.

## Artifact gates

The release matrix retains all nine existing targets. Every binary archive must
contain exactly `wamr`, `wamrc`, `LICENSE`, and `README.md` in the expected layout.
The verifier checks ELF64, Mach-O64, or PE32+ machine identities, executable
permissions where applicable, and documentation contents. CRLF/LF differences in
packaged documentation are allowed; binary hashes always describe the exact bytes.
Windows ZIP separators are normalized before applying the same member whitelist
on every host; traversal and duplicate normalized names remain errors.
Machine-header checks do not establish a libc deployment floor.

The verifier extracts only the expected regular files into a fresh directory.
On a matching supported native host it runs the **packaged** tools, checks both
version commands, compiles and executes the no-op fixture, and requires the exact
119-byte output of the unsigned division/remainder/decimal regression.
Linux x64/ARM64, macOS ARM64, and Windows x64 must execute these paths. Other
host/target mismatches are explicitly recorded as not executed; successful
cross-compilation is not runtime acceptance. Linux-musl x64 also executes when
built on a matching Linux x64 host.

Both Windows archives are signed through the existing `release` environment and
Azure Trusted Signing configuration. Their extracted executables must have valid
Authenticode signatures and timestamp certificates, including the cross-built
ARM64 executables. Signing verification does not imply ARM64 execution on x64.

SBOM generation scans the staged package, not unrelated installed benchmark/test
binaries. The inventory checks SPDX document identity and hashes its actual bytes;
it does not claim that an SBOM scanner discovers every statically linked dependency.
The source tarball's Git PAX commit and selected source files must agree with the
candidate checkout.

## Retained evidence and publication

Platform/source bundles are retained for seven days. The smaller `release-evidence`
artifact is retained for thirty days and includes per-platform reports, SBOMs,
`release-manifest.json`, `SHA256SUMS`, release notes, and the attestation bundle.
The manifest records actual archive/member hashes and native/signing coverage.
Checksums cover all release payloads and the manifest; provenance also attests the
checksums themselves. Download only the evidence unless binary replay is needed.

Version-specific notes are read from `docs/releases/v<version>.md` when present;
the source SHA and installation command are appended automatically.
`3.0.0-dev.14` maps to Python `3.0.0.dev14` through the same helper used by the wheel
builders. Existing PyPI workflows remain stable-only and cannot publish from a
manual preflight. This preflight does not build or validate Python wheels.

Review the exact-candidate CI, artifact evidence, and any unexecuted platforms
before authorizing publication. A preflight never creates a tag or release.
Dev-tag publication is explicitly prerelease and not latest; the final release
metadata still needs inspection after an authorized tag push.
