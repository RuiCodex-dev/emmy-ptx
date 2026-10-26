#!/usr/bin/env python3
"""Benchmark: compile each problem, run it on the organisers' runner with
warm-up and timed repetitions, check the result, and print a Markdown table.

    python tools/bench.py ../emmy-hackathon-2026/problems practice/extra practice/hard > out/bench.md

Timings are the runner's median over ``--iters`` launches (no clock locking,
so treat them as indicative).  Single-kernel problems only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "http://runner.riftstack.ai:48701"


def post(path: str, body: dict) -> dict:
    for _ in range(60):
        r = requests.post(RUNNER + path, json=body, timeout=(10, 300))
        r.raise_for_status()
        out = r.json()
        if out.get("stage") != "busy":
            return out
        time.sleep(2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    rows = []
    for d in args.dirs:
        for p in sorted(Path(d).iterdir()):
            if not (p / "meta.json").exists() or not (p / "expected.npz").exists():
                continue
            meta = json.loads((p / "meta.json").read_text())
            if len(meta["kernels"]) != 1:
                continue
            k = meta["kernels"][0]
            t0 = time.perf_counter()
            c = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(p / f"{k['entry']}.ir.json")],
                               capture_output=True, text=True)
            t_compile = time.perf_counter() - t0
            if c.returncode != 0:
                rows.append((p.name, "compile error", "", "", "", ""))
                continue
            comp = post("/compile", {"ptx": c.stdout})
            if not comp.get("ok"):
                rows.append((p.name, "ptxas error", "", "", "", ""))
                continue
            need = max(int(np.prod(a["shape"])) for a in k["args"] if a.get("output"))
            run = post("/run", {"cubin_id": comp["cubin_id"], "entry": k["entry"], "grid": k["grid"],
                                "block": k["block"], "smem_bytes": k["smem_bytes"],
                                "zero_outputs": k["zero_outputs"], "args": k["args"],
                                "warmup": 3, "iters": args.iters, "max_return": max(4096, need)})
            expected = dict(np.load(p / "expected.npz"))
            ok = run.get("ok", False)
            err = 0.0
            for name, out in run.get("outputs", {}).items():
                if name in expected:
                    got = np.array(out["values"], dtype=np.float32).reshape(expected[name].shape)
                    ok &= bool(np.allclose(got, expected[name], rtol=meta["rtol"], atol=meta["atol"]))
                    err = max(err, float(np.max(np.abs(got - expected[name]))))
            # note: with zero_outputs, timed repetitions re-zero atomic outputs, so the check holds
            rows.append((p.name, "pass" if ok else "FAIL", f"{run.get('median_us', 0):.1f}",
                         str(comp["registers"].get(k["entry"], "")), f"{err:.1e}", f"{t_compile:.2f}"))
            print(f"{p.name}: {rows[-1][1:]}", file=sys.stderr, flush=True)
    print("| problem | result | median us | registers | max abs err | compile s |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
