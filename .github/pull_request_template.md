## Summary

-

## Security review

- [ ] This PR does not touch sandbox-critical code or security process docs.
- [ ] This PR touches a sandbox-critical boundary; boundary:
- [ ] Guest-memory range checks use overflow-safe arithmetic before slicing or
      native memory access.
- [ ] Interpreter/AOT trap semantics and differential or spec coverage were
      considered.
- [ ] WASI/component resource capability and ownership/lifetime rules were
      considered.
- [ ] Security reporting/process wording keeps the project's experimental,
      independent, no-SLA status clear.

## Validation

-
