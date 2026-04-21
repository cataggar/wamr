---
name: aot-diff-debug
description: Debug AOT codegen mismatches by comparing interpreter vs AOT output on real wasm binaries. Use when AOT produces wrong results, CRC errors, garbled output, or any computation mismatch versus the interpreter.
---
# AOT Differential Debugging

Systematically find AOT codegen bugs by comparing interpreter and AOT execution of the same wasm binary. This skill encodes the lessons from debugging the CoreMark CRC bug (memory_fill r11 clobber, issue #109).

## When to Use

- AOT produces wrong numeric results, garbled text, or CRC errors
- A wasm program works in the interpreter but fails in AOT
- You suspect a codegen bug but don't know which instruction is wrong

## Core Principle

**Use the interpreter as an oracle. Diff real outputs first, hypothesize second.**

Do NOT start by:
- Reading codegen source code for suspicious patterns
- Building synthetic differential tests for hypothesized instructions
- Auditing register allocation or spill/reload logic

DO start by:
- Running the same binary through both paths and comparing output

## Workflow

### Phase 1: Reproduce and Diff (do this FIRST, every time)

1. **Identify the wasm binary** that produces wrong results in AOT.

2. **Run through both paths** and capture output:
   ```powershell
   # Interpreter (the oracle — always correct)
   .\zig-out\bin\wamr.exe program.wasm > interp-out.txt 2>&1

   # AOT
   .\zig-out\bin\wamrc.exe -o program.aot program.wasm
   .\zig-out\bin\wamr.exe program.aot > aot-out.txt 2>&1
   ```

3. **Diff the outputs** byte-by-byte. Find the FIRST divergence:
   ```powershell
   $interp = Get-Content interp-out.txt
   $aot = Get-Content aot-out.txt
   for ($i = 0; $i -lt [Math]::Min($interp.Count, $aot.Count); $i++) {
       if ($interp[$i] -ne $aot[$i]) {
           Write-Host "FIRST DIFF at line $($i+1):"
           Write-Host "  INTERP: $($interp[$i])"
           Write-Host "  AOT:    $($aot[$i])"
           break
       }
   }
   ```

4. **Characterize the divergence**:
   - Is it a wrong number? (codegen arithmetic bug)
   - Is it garbled characters? (memory corruption or wrong pointer)
   - Is it extra/missing output? (control flow bug)
   - Is it a specific character that's always wrong? (lookup table or constant bug)

### Phase 2: Build a Minimal Reproducer

Once you know WHAT is wrong from Phase 1, build the smallest C program that reproduces it.

**Key technique**: compile C to wasm32-wasi, run through both paths:
```powershell
$zig = "path\to\zig.exe"
& $zig build-exe test.c -target wasm32-wasi -lc -O ReleaseFast --name test
.\zig-out\bin\wamr.exe test.wasm        # interpreter
.\zig-out\bin\wamrc.exe -o test.aot test.wasm
.\zig-out\bin\wamr.exe test.aot         # AOT
```

**Minimization strategy** (in order):
1. Remove unrelated functionality from the C source
2. Replace complex formatting with `putchar()` to isolate whether values or formatting is wrong
3. Add both `%d` (decimal) and `%x` (hex) output to distinguish value errors from formatting errors
4. Test edge cases: value 0, negative values, max values, boundary values

**Example**: if hex output is garbled but decimal output is correct, the BUG is in how the hex formatter works (memory_fill, lookup table, etc.), not in the value computation.

### Phase 3: Identify the Guilty Instruction Handler

Once you have a minimal reproducer (ideally <10 lines of C):

1. **Enable IR dump** in `src/compiler/codegen/x86_64/compile.zig`:
   ```zig
   const dump_all: bool = true;  // line ~1300
   ```
   Rebuild `wamrc` and compile the test wasm. The IR dump shows every function's instructions and register allocations.

2. **Search the IR dump** for the operation that produces wrong results:
   - `memory_fill` — REP STOSB fill operations
   - `memory_copy` — REP MOVSB copy operations
   - `div_u` / `rem_u` — division and remainder
   - `call_indirect` — indirect function calls
   - `br_table` — switch/jump tables
   - `select` — conditional select

3. **Check the instruction handler** in `compile.zig` for scratch register clobbers:
   - Does it stash a value in `r10` or `r11` before calling a helper?
   - Does the helper (`emitMemBoundsCheck`, `emitMemBoundsCheckDynamic`, `emitOobCmpAndTrap`, `emitCallIndirectSigCheck`) clobber that register?
   - **Known clobber sets**:
     - `emitMemBoundsCheck` / `emitMemBoundsCheckDynamic`: clobbers **r11**, r10, rax, param_regs[0]
     - `emitOobCmpAndTrap`: clobbers r10→param_regs[0], rax
     - `emitCallIndirectSigCheck`: clobbers **rcx**, **r11**

### Phase 4: Fix and Verify

1. **Fix pattern**: prefer "load late" over "stash and restore":
   ```zig
   // BAD: stash in r11, call helper that clobbers r11, restore from r11
   try code.movRegReg(.r11, .rax);   // save
   try emitMemBoundsCheckDynamic();   // CLOBBERS r11!
   try code.movRegReg(.rax, .r11);   // "restore" — WRONG

   // GOOD: load the value AFTER the helper call
   try emitMemBoundsCheckDynamic();   // do bounds check first
   const val_reg = try useVReg(...);  // load value after (survives check)
   ```

2. **Run unit tests**: `zig build test`
3. **Run the minimal reproducer** through both interpreter and AOT — outputs must match
4. **Run the original failing program** (e.g., CoreMark) — CRCs/output must match
5. **Run spec tests**: `zig build spec-tests-aot` (check for regressions)

## Anti-Patterns (things that waste time)

| Anti-Pattern | Why It Wastes Time | Do Instead |
|---|---|---|
| Reading codegen handlers looking for bugs | Too many handlers, bug could be anywhere | Diff real output first |
| Building synthetic wasm tests for hypothesized opcodes | Tests pass because they don't exercise the real pattern | Build minimal C reproducer of the actual failure |
| Testing with different register counts | Only eliminates allocatable-reg bugs, not scratch-reg bugs | Check scratch register clobbers in handlers |
| Assuming "spec tests pass ∴ all opcodes correct" | Spec tests don't test libc internals or complex interactions | Test with real programs |
| Hypothesizing before observing | Leads to confirmation bias | Always observe the divergence first |

## Historical Bugs Found with This Approach

### CoreMark CRC bug (issue #109)
- **Symptom**: list CRC=0x5531 (expected 0xe714), state CRC=0x344b (expected 0x8e3a)
- **Phase 1 finding**: hex digit '0' replaced with garbage characters (garbled printf output)
- **Phase 2 reproducer**: `printf("%d\n", 0)` — 1 line of C
- **Root cause**: `memory_fill` handler stashed fill value in r11, then called `emitMemBoundsCheckDynamic` which clobbers r11 via `lea r11, [rax+rcx]`
- **Fix**: load fill value AFTER bounds check (commit cfa702b5)
- **Time wasted before this approach**: ~8 hours of hypothesis-driven debugging
- **Time with this approach**: <1 hour

## References

- `src/compiler/codegen/x86_64/compile.zig` — x86-64 instruction handlers
- `src/compiler/codegen/x86_64/compile.zig:1298` — `dump_func_idx` / `dump_all` debug knobs
- `src/compiler/frontend.zig` — wasm-to-IR translation
- `src/compiler/codegen/x86_64/compile.zig:277` — `emitMemBoundsCheck` (clobbers r11)
- `src/compiler/codegen/x86_64/compile.zig:313` — `emitMemBoundsCheckDynamic` (clobbers r11)
