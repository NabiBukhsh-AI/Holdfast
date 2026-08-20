"""Grade a completed run against the acceptance thresholds. TASK-018, spec 15.10.

Exits non zero when the reproduction does not meet the bands. Missing thresholds on more than
15 percent of cells is a FINDING to escalate, not a number to nudge.

    python scripts/verify_reproduction.py --results artifacts/run_0001/results.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compint.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "verify", *sys.argv[1:]]
    app()
