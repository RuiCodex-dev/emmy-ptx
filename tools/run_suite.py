#!/usr/bin/env python3
"""Run compile.py on every problem folder under one or more directories through
the task repository's submit.py (compile with this compiler, assemble with
ptxas on the runner, launch on its GPU, compare against expected.npz).

    python tools/run_suite.py --submit ../emmy-hackathon-2026/submit.py \
        ../emmy-hackathon-2026/problems gen_extra

Prints one line per problem and a summary; exit code 0 only if all pass.
Writes the full per-problem log to out/suite.log.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="directories holding problem folders")
    ap.add_argument("--submit", required=True, help="path to the task repository's submit.py")
    ap.add_argument("--only", default=None, help="regex filter on problem ids")
    args = ap.parse_args()

    problems = []
    for d in args.dirs:
        for p in sorted(Path(d).iterdir()):
            if (p / "meta.json").exists() and (not args.only or re.search(args.only, p.name)):
                problems.append(p)

    (ROOT / "out").mkdir(exist_ok=True)
    log = open(ROOT / "out" / "suite.log", "w", encoding="utf-8")
    passed = 0
    for p in problems:
        t0 = time.time()
        r = subprocess.run([sys.executable, args.submit, str(p), "--compiler", str(ROOT / "compile.py")],
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        log.write(f"==== {p.name} (exit {r.returncode})\n{out}\n")
        ok = r.returncode == 0
        passed += ok
        us = re.findall(r"\(([\d.]+) us\)", out)
        regs = re.findall(r"(\d+) registers", out)
        detail = f"{'+'.join(us)} us, {'/'.join(regs)} regs" if ok else out.splitlines()[-1][:150]
        print(f"{'PASS' if ok else 'FAIL'}  {p.name:<28} {detail}  [{time.time() - t0:.1f}s]", flush=True)
    print(f"\n{passed}/{len(problems)} passed")
    return 0 if passed == len(problems) else 1


if __name__ == "__main__":
    sys.exit(main())
