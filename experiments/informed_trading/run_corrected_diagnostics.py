from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_corrected_state_diagnostics.py"),
            "--config",
            str(root / "configs/tick/corrected_candidate/state_diagnostics.yaml"),
        ],
        cwd=root,
        check=False,
    ).returncode
