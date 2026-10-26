#!/usr/bin/env python3
"""Emmy Kernel IR -> PTX.

    python compile.py path/to/kernel.ir.json > out.ptx

PTX goes to stdout and nothing else does; diagnostics go to stderr and a
kernel this compiler cannot lower exits with status 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ptxgen.kernel import compile_file  # noqa: E402
from ptxgen.lower import CompileError  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        ptx = compile_file(argv[1])
    except (CompileError, NotImplementedError) as exc:
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
