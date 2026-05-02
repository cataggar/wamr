# Security Review and Advisory Process

This is a maintainer checklist for lightweight security handling in this
repository. It complements the public reporting instructions in
[SECURITY.md](SECURITY.md) and the sandbox-critical review checklist in
[SECURITY_AUDIT.md](SECURITY_AUDIT.md).

This project is experimental, independently maintained, and not
production-supported. The process below is best-effort guidance; it is not a
formal security response SLA, long-term support policy, embargo guarantee, or
CVE commitment.

## Private vulnerability report triage

Use GitHub Private Vulnerability Reporting for sensitive reports. Keep report
details, payloads, and reproducers private until a public fix or advisory plan is
chosen.

1. Acknowledge the report in the private thread and ask for any missing
   non-sensitive reproduction context.
2. Reproduce or narrow the issue without moving exploit details into public
   issues, public PRs, logs, or docs.
3. Classify the affected boundary:
   - loader or validation
   - interpreter runtime
   - AOT compiler, codegen, or runtime
   - component canonical ABI
   - WASI or component host resources
   - build, release, dependency, or supply chain
   - documentation or process only
4. Assess impact in repository terms:
   - host memory access or sandbox escape
   - guest isolation break
   - WASI/component capability bypass
   - denial of service
   - incorrect trap behavior or AOT/interpreter mismatch
   - build, release, or supply-chain exposure
   - non-security correctness bug
5. Decide whether the fix should happen in a private security fork/branch or in
   a normal public PR with minimal disclosure.
6. Keep reproducers attached to the private report when needed. Public
   regression tests should be minimized or redacted if the full reproducer would
   reveal an unfixed vulnerability.
7. After a fix is public, cross-link the private report, public issue or PR, and
   any advisory record that applies.

## Advisory and CVE decisions

Use a GitHub Security Advisory when a report is a confirmed vulnerability in this
repository and private coordination is useful before public disclosure.

Consider requesting a CVE only when all of the following are true:

- the issue has a confirmed security impact in this repository;
- released artifacts or real users are plausibly affected;
- ecosystem-wide tracking would help downstream users or distributors.

Use a normal public issue or PR for:

- non-sensitive hardening ideas;
- already-public bugs;
- test or checklist gaps;
- correctness bugs without a plausible security impact;
- issues that only affect unsupported local experiments.

If an issue only affects upstream C/C++ WAMR, Wasmtime, or another project, link
to the upstream advisory and record this Zig implementation as unaffected or
under review instead of creating a misleading advisory here.

## Upstream advisory tracking

Periodically review upstream advisories and releases that could overlap this
runtime's threat model:

- upstream `bytecodealliance/wasm-micro-runtime` security advisories, releases,
  and issues;
- relevant `bytecodealliance/wasmtime` advisories and releases;
- GitHub Security Advisories for WebAssembly runtimes or dependencies used by
  this repository.

For each relevant advisory, record a short public note in the tracking issue or
follow-up PR. Use this format:

| Field | Expected content |
| --- | --- |
| Source | Project and advisory or release link |
| Reviewed | Date reviewed |
| Area | Loader, interpreter, AOT, component, WASI, release, dependency, etc. |
| Status | `unaffected`, `affected`, `under-review`, or `not-applicable` |
| Rationale | Short reason without exploit details |
| Follow-up | Issue or PR link, if any |

Do not copy exploit payloads, embargoed details, or sensitive reproducer material
into public tracking notes.

## PR review expectations

Use [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for the detailed sandbox-critical
checklist. For everyday PR review, first decide whether a change touches one of
these boundaries:

- WebAssembly loader or validation;
- interpreter dispatch, memory, table, or trap handling;
- compiler frontend, IR passes, register allocation, AOT codegen, or AOT
  runtime;
- component canonical ABI lifting/lowering or guest-memory access;
- WASI/component host capabilities, resource tables, or resource lifetimes;
- build, release, dependency, or workflow behavior.

If a PR touches a boundary, reviewers should look for:

- overflow-safe `ptr + len`, `addr + offset + width`, and element-count
  calculations before slicing host memory;
- trap semantics preserved across interpreter and AOT paths;
- explicit table, function, signature, null-reference, and dropped-segment
  checks before dispatch;
- host-resource ownership and drop behavior that cannot double-free, leak
  capabilities, or reuse stale handles incorrectly;
- tests or documented rationale for boundary cases, especially when the change
  intentionally shifts behavior.

## CI and validation expectations

Existing CI provides these security-relevant signals:

- `zig build test` is the baseline for normal code changes.
- Debug builds keep Zig safety checks visible, which helps catch bounds and
  overflow mistakes.
- ReleaseSafe builds and cross-target jobs provide platform coverage.
- Spec tests exercise WebAssembly semantic compatibility.
- The wasm32-wasi smoke job checks the self-hosted WASI build path.
- The fuzz workflow runs core-loader, component-loader, interpreter, AOT, and
  differential harness paths on schedule, on demand, and on PRs that touch
  runtime, component, compiler, or fuzz files.

Reviewer guidance:

- For interpreter, compiler, runtime, or component boundary changes, expect
  `zig build test` and consider targeted spec, differential, AOT, or fuzz
  coverage.
- For loader/runtime/component/compiler/fuzz harness changes, consider
  `zig build fuzz` or the GitHub fuzz workflow when the change affects input
  parsing or execution boundaries. See [tests/fuzz/README.md](tests/fuzz/README.md)
  for harness-specific oracles and
  [tests/fuzz/OSS_FUZZ.md](tests/fuzz/OSS_FUZZ.md) for the OSS-Fuzz integration
  decision and prerequisites.
- For documentation-only changes, a full Zig build is usually unnecessary; check
  links, wording, and whether the text avoids unsupported support promises.

## Public communication

Keep public issues and PRs factual and minimal until any sensitive fix is
available. Public notes should describe impact, affected area, and fixed version
or commit when appropriate, but should not include exploit instructions or
unredacted payloads for an unresolved vulnerability.
