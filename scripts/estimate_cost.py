"""Pre flight cost projection. TASK-017, execution contract rule 15.

Thin wrapper over `compint cost`. Kept as a separate entry point because the reproduction
documentation refers to it by path, and because a budget gate that is awkward to run is a
budget gate nobody runs.

    python scripts/estimate_cost.py --config configs/research/rq1_baseline.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compint.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "cost", *sys.argv[1:]]
    app()
