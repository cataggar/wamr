---
name: Bug report
about: Create a report to help us improve
title: "Add a placeholder for issue title. ex: [BUG]"
labels: bug
assignees: ""
---

**Is it a security vulnerability?**
If this may expose host memory, bypass a sandbox or WASI/component capability,
mis-handle a private vulnerability report, or includes sensitive proof-of-concept
details, do not file it publicly. Use this repository's private vulnerability
reporting instead:
https://github.com/cataggar/wamr/security/advisories/new

Public issues are fine for non-sensitive bugs, crashes without exploit details,
and hardening ideas.

**Describe the bug**
A clear and concise description of what the bug is.

**Version**
Information like tags, release version, commits.

**To Reproduce**
Steps to reproduce the behavior:

1. Build `wamr` with flags like '...'
2. (Optional) Build `wamrc` with flags like '....'
3. (Optional) Run `wamrc` with CLI options like '...' to generate AOT output
4. Run `wamr` with CLI options like '...'
5. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Actual Result**
What you've got.

**Desktop (please complete the following information):**

- Arch [e.g. x86_64, arm64, 32bit]
- Board [e.g. STM32F407]
- OS [e.g. Linux, Windows, macOS, FreeRTOS]
- Version [e.g. 22]

**Additional context**
Add any other context about the problem here.
