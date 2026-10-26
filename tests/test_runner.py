"""Opt-in end-to-end test on the organisers' GPU runner.

    EMMY_SUBMIT=/path/to/emmy-hackathon-2026/submit.py python -m pytest tests/test_runner.py

Each problem is compiled by compile.py, assembled by ptxas on the runner,
launched with the geometry in meta.json and compared with expected.npz.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from test_compile import COMPILE, problem_dirs

SUBMIT = os.environ.get("EMMY_SUBMIT")

pytestmark = pytest.mark.skipif(not SUBMIT, reason="set EMMY_SUBMIT to the task repository's submit.py")


@pytest.mark.parametrize("pdir", [d for d in problem_dirs() if (d / "expected.npz").exists()],
                         ids=lambda p: Path(p).name)
def test_passes_on_gpu(pdir):
    r = subprocess.run([sys.executable, SUBMIT, str(pdir), "--compiler", str(COMPILE)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
