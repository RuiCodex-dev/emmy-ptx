"""Offline tests: compile every problem we have and check the PTX contract.

No GPU and no network: these check what can be checked without ptxas --
the CLI contract, the entry name, the parameter order from meta.json, pure
ASCII, declared registers, balanced labels.  tests/test_runner.py (opt-in)
does the real assemble-run-compare on the organisers' GPU.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COMPILE = ROOT / "compile.py"


def problem_dirs():
    roots = [ROOT / "practice" / "extra", ROOT / "practice" / "hard"]
    task = os.environ.get("EMMY_TASK_PROBLEMS")
    if task:
        roots.insert(0, Path(task))
    out = []
    for r in roots:
        if r.exists():
            out += [p for p in sorted(r.iterdir()) if (p / "meta.json").exists()]
    return out


def kernels():
    for p in problem_dirs():
        meta = json.loads((p / "meta.json").read_text())
        for k in meta["kernels"]:
            yield pytest.param(p, k, id=f"{p.name}:{k['entry']}")


def run(ir: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(COMPILE), str(ir)], capture_output=True, text=True)


@pytest.mark.parametrize("pdir,kernel", list(kernels()))
def test_contract(pdir, kernel):
    r = run(pdir / f"{kernel['entry']}.ir.json")
    assert r.returncode == 0, r.stderr
    ptx = r.stdout
    ptx.encode("ascii")                                       # PTX rejects non-ASCII
    first = next(l for l in ptx.splitlines() if l.strip() and not l.startswith("//"))
    assert first.startswith(".version")                      # stdout is PTX only
    assert f".visible .entry {kernel['entry']}(" in ptx       # entry named as the IR names it
    params = re.findall(r"\.param \.\w+ p_(\w+)", ptx)
    assert params == [re.sub(r"[^A-Za-z0-9_]", "_", a) for a in kernel["arg_order"]]


@pytest.mark.parametrize("pdir,kernel", list(kernels()))
def test_registers_declared_and_labels_resolve(pdir, kernel):
    ptx = run(pdir / f"{kernel['entry']}.ir.json").stdout
    for prefix in ("%p", "%r", "%f", "%rd"):
        decl = re.search(rf"\.reg \.\w+\s+{re.escape(prefix)}<(\d+)>;", ptx)
        used = [int(n) for n in re.findall(rf"{re.escape(prefix)}(\d+)\b", ptx)]
        if used:
            assert decl, f"{prefix} registers used but not declared"
            assert max(used) < int(decl.group(1))
    defined = set(re.findall(r"^(\$L_\w+):", ptx, re.M))
    targets = set(re.findall(r"bra(?:\.uni)?\s+(\$L_\w+);", ptx))
    assert targets <= defined


def test_cli_usage_error_goes_to_stderr():
    r = subprocess.run([sys.executable, str(COMPILE)], capture_output=True, text=True)
    assert r.returncode == 2 and r.stdout == "" and "usage" in r.stderr
