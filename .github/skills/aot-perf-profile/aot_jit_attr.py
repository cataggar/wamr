#!/usr/bin/env python3
"""De-anonymize a WAMR AOT `perf` profile and attribute cycles per function
and per instruction class.

WAMR maps generated AOT code as an anonymous RX mmap, so `perf` collapses
~98% of a precompiled run into one unsymbolized `[JIT]` bucket. This tool
maps sample IPs back to wasm `local_func` indices using the `.cwasm`
function section (`func_offsets[]`) plus the text mmap base recorded in the
perf data, then (optionally) classifies the hot instructions of one
function (spill reloads / frame stores / bounds checks / dispatch / ...).

Usage:
  aot_jit_attr.py --perf wamr.perf --cwasm core4.cwasm            # per-function top
  aot_jit_attr.py --perf wamr.perf --cwasm core4.cwasm --func 6145  # + instr mix

Requires: perf, objdump, python3. x86_64 only for --func classification.
"""
import argparse, bisect, json, re, struct, subprocess, sys, os
from pathlib import Path

AOT_MAGIC = 0x746F6100  # "\0aot"
AOT_VERSION = 7
SEC_TEXT = 2
SEC_FUNCTION = 3


def parse_cwasm(path):
    """Return (func_offsets:list[int], text_size:int, text_fileoff:int)."""
    f = Path(path).read_bytes()
    magic, ver = struct.unpack_from("<II", f, 0)
    if magic != AOT_MAGIC:
        sys.exit(f"{path}: bad magic {magic:#x} (not a .cwasm)")
    if ver != AOT_VERSION:
        print(f"warning: {path} aot_version={ver}, expected {AOT_VERSION}", file=sys.stderr)
    pos = 8
    offs = text_size = text_fileoff = None
    while pos + 8 <= len(f):
        st, sz = struct.unpack_from("<II", f, pos)
        pos += 8
        if st == SEC_TEXT:
            text_fileoff, text_size = pos, sz
        elif st == SEC_FUNCTION:
            cnt, = struct.unpack_from("<I", f, pos)
            arr = struct.unpack_from("<%dI" % (cnt * 2), f, pos + 4)
            offs = list(arr[0::2])  # interleaved (offset, type_idx)
        pos += sz
    if offs is None or text_size is None:
        sys.exit(f"{path}: missing function/text section")
    return offs, text_size, text_fileoff, f


def jit_exec_mmaps(perf):
    """[(base,size)] of anonymous executable mappings, largest first."""
    out = subprocess.run(
        ["perf", "script", "-i", perf, "--show-mmap-events"],
        capture_output=True, text=True).stdout
    maps = []
    pat = re.compile(r"\[(0x[0-9a-f]+)\((0x[0-9a-f]+)\).*?\]: r[w-]xp //anon")
    for line in out.splitlines():
        m = pat.search(line)
        if m:
            maps.append((int(m.group(1), 16), int(m.group(2), 16)))
    return sorted(set(maps), key=lambda x: -x[1])


def addr_counts(perf):
    """{ip:self_sample_count} for [JIT] addresses; also total self samples."""
    out = subprocess.run(
        ["perf", "report", "-i", perf, "--stdio", "-g", "none", "-n",
         "--sort=dso,symbol", "--percent-limit", "0"],
        capture_output=True, text=True).stdout
    row = re.compile(r"^\s*[\d.]+%\s+[\d.]+%\s+(\d+)\s+(.*)$")
    cnt, total = {}, 0
    for line in out.splitlines():
        m = row.match(line)
        if not m:
            continue
        c = int(m.group(1)); total += c
        rest = m.group(2)
        if "[JIT]" in rest:
            sm = re.search(r"\[[.]\]\s+(0x[0-9a-fA-F]+)", rest)
            if sm:
                cnt[int(sm.group(1), 16)] = cnt.get(int(sm.group(1), 16), 0) + c
    return cnt, total


def classify(t):
    mn = t.split()[0] if t else ""
    frame = ("[rbp-" in t) or ("[rbp+" in t) or ("[rsp" in t)
    is_mov = mn in ("mov", "movzx", "movsx", "movsxd", "movsd", "movss",
                    "movdqu", "movdqa", "movaps", "movups", "movq", "movd")
    if frame and is_mov:
        dest = t[len(mn):].strip().split(",")[0]
        return "spill_store" if (("[rbp-" in dest) or ("[rbp+" in dest) or ("[rsp" in dest)) else "spill_load"
    if mn == "cmp" and re.search(r"\[(rbx|r10|r11)\+0x8\]", t):
        return "bounds_cmp"
    if mn in ("ja", "jae", "jb", "jbe"):
        return "bounds_branch"
    if mn == "call":
        return "call"
    if mn == "jmp":
        return "dispatch_jmp" if re.search(r"jmp\s+r(ax|10|11)", t) else "jmp"
    if mn in ("je", "jne", "jl", "jle", "jg", "jge", "js", "jns", "jp", "jnp", "jo", "jno"):
        return "cond_branch"
    if is_mov and "[" in t and not frame and "rip" not in t:
        return "mem_access"
    if is_mov:
        return "regmov"
    if mn in ("add", "sub", "and", "or", "xor", "shl", "shr", "sar", "imul",
              "mul", "inc", "dec", "neg", "not", "test", "lea", "sete", "setne",
              "seta", "cdqe", "cqo"):
        return "alu"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", required=True)
    ap.add_argument("--cwasm", required=True, help="the core .cwasm that owns the hot code")
    ap.add_argument("--func", type=int, default=None, help="classify this local_func's instr mix")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--base", default=None, help="override text mmap base (hex)")
    ap.add_argument("--json-out", default=None,
                    help="write machine-readable attribution summary")
    ap.add_argument("--min-samples", type=int, default=0,
                    help="fail unless perf contains at least this many self samples")
    ap.add_argument("--require-size-match", action="store_true",
                    help="fail instead of guessing when the text mmap is not unique")
    a = ap.parse_args()

    offs, tsize, tfileoff, fbytes = parse_cwasm(a.cwasm)
    cnt, total = addr_counts(a.perf)
    if total == 0:
        sys.exit("no samples in perf data")
    if total < a.min_samples:
        sys.exit(f"perf data has only {total} self samples; "
                 f"--min-samples requires at least {a.min_samples}")

    if a.base:
        base = int(a.base, 16)
    else:
        # pick the anon exec mapping whose size matches this core's text (page-rounded)
        maps = jit_exec_mmaps(a.perf)
        matches = [(b, s) for b, s in maps if abs(s - tsize) <= 0x2000]
        if a.require_size_match and len(matches) != 1:
            sys.exit(
                f"expected one anonymous executable mmap matching text_size={tsize}, "
                f"found {len(matches)}; pass --base explicitly"
            )
        base = matches[0][0] if matches else None
        if base is None and maps:
            base = maps[0][0]
            print(f"warning: no mmap size match for text_size={tsize}; using largest "
                  f"anon exec base {base:#x} (size {maps[0][1]})", file=sys.stderr)
        if base is None:
            sys.exit("could not find an anonymous executable mapping in perf data")
    end = base + tsize
    print(f"core text base={base:#x} size={tsize} ({tsize/1048576:.1f} MB), "
          f"func_count={len(offs)}, total self samples={total}")

    perfunc, in_core = {}, 0
    for ip, c in cnt.items():
        if base <= ip < end:
            in_core += c
            fi = bisect.bisect_right(offs, ip - base) - 1
            perfunc[fi] = perfunc.get(fi, 0) + c
    print(f"samples in this core: {in_core} ({100*in_core/total:.1f}% of run)\n")
    print(f"=== top {a.top} functions by self samples ===")
    top_functions = []
    for fi, c in sorted(perfunc.items(), key=lambda x: -x[1])[:a.top]:
        fend = offs[fi + 1] if fi + 1 < len(offs) else tsize
        top_functions.append({
            "local_func": fi,
            "samples": c,
            "percent_of_run": 100 * c / total,
            "code_bytes": fend - offs[fi],
        })
        print(f"  local_func={fi:<6} samples={c:<6} ({100*c/total:.2f}% of run)  "
              f"code_bytes={fend-offs[fi]}")

    report = {
        "schema_version": 1,
        "perf": str(Path(a.perf).resolve()),
        "cwasm": str(Path(a.cwasm).resolve()),
        "text_base": base,
        "text_size": tsize,
        "function_count": len(offs),
        "total_samples": total,
        "attributed_samples": in_core,
        "attribution_coverage_pct": 100 * in_core / total,
        "top_functions": top_functions,
    }

    if a.func is None:
        if a.json_out:
            Path(a.json_out).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="UTF-8",
            )
        return
    fi = a.func
    if fi < 0 or fi >= len(offs):
        sys.exit(f"--func {fi} out of range (0..{len(offs)-1})")
    s = offs[fi]; e = offs[fi + 1] if fi + 1 < len(offs) else tsize
    fnbase = base + s
    scratch = Path(a.perf).resolve().parent
    tmp = scratch / f".aot-jit-attr-{os.getpid()}-{fi}.bin"
    tmp.write_bytes(fbytes[tfileoff + s:tfileoff + e])
    try:
        asm = subprocess.run(
            ["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
             "--adjust-vma=0x%x" % fnbase, str(tmp)],
            capture_output=True, text=True).stdout
    finally:
        tmp.unlink(missing_ok=True)
    ipat = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*(.*)$")
    fn_total = 0
    byclass = {}
    hot = []
    for line in asm.splitlines():
        m = ipat.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16); txt = m.group(2).strip()
        c = cnt.get(addr, 0)
        fn_total += c
        cl = classify(txt)
        byclass[cl] = byclass.get(cl, 0) + c
        if c:
            hot.append((c, addr, txt))
    print(f"\n=== local_func={fi} instruction-class mix "
          f"(self={fn_total}, {100*fn_total/total:.1f}% of run) ===")
    groups = {
        "spill_load (reloads)": ["spill_load"],
        "frame stores": ["spill_store"],
        "reg-reg mov": ["regmov"],
        "ALU": ["alu"],
        "bounds-check": ["bounds_cmp", "bounds_branch"],
        "linear-mem access": ["mem_access"],
        "dispatch (computed-goto)": ["dispatch_jmp"],
        "call": ["call"],
        "other branches": ["cond_branch", "jmp"],
        "other": ["other"],
    }
    for g, ks in sorted(groups.items(), key=lambda kv: -sum(byclass.get(k, 0) for k in kv[1])):
        v = sum(byclass.get(k, 0) for k in ks)
        print(f"  {g:<28} {v:>7}  ({100*v/total:.1f}% of run)")
    sp = byclass.get("spill_load", 0) + byclass.get("spill_store", 0)
    print(f"  >> stack/spill traffic = {100*sp/total:.1f}% of total run; "
          f"reloads alone = {100*byclass.get('spill_load',0)/total:.1f}%")
    print(f"\n=== top 20 hottest instructions in local_func={fi} ===")
    for c, addr, txt in sorted(hot, reverse=True)[:20]:
        print(f"  {c:>5} ({100*c/total:.2f}%)  {addr:x}: {txt}")
    report["classified_function"] = {
        "local_func": fi,
        "samples": fn_total,
        "percent_of_run": 100 * fn_total / total,
        "classes": {
            group: {
                "samples": sum(byclass.get(key, 0) for key in keys),
                "percent_of_run": (
                    100 * sum(byclass.get(key, 0) for key in keys) / total
                ),
            }
            for group, keys in groups.items()
        },
        "hottest_instructions": [
            {
                "samples": count,
                "percent_of_run": 100 * count / total,
                "address": address,
                "instruction": text,
            }
            for count, address, text in sorted(hot, reverse=True)[:20]
        ],
    }
    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="UTF-8",
        )


if __name__ == "__main__":
    main()
