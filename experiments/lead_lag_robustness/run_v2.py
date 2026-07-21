from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "scripts/run_tick_lead_lag_robustness_v2.py"),
        "--config",
        str(root / "configs/tick/multi_asset_5y/lead_lag_robustness_v2.yaml"),
    ]
    return subprocess.run(command, cwd=root, check=False).returncode
