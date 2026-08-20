"""Regenerate golden fixtures. Requires an explicit confirm.

Spec 20.1: changing a golden file requires an explicit reviewer sign off in the PR
description. This script makes regeneration deliberate rather than accidental, and it prints
a diff summary so a reviewer can see exactly what moved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compint.core.catalog import load_catalog
from compint.core.framing import TEMPLATE_VERSION, all_framings
from shared.prompts import PromptRegistry

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"


def build_framing_golden() -> dict[str, object]:
    catalog = load_catalog(ROOT / "data" / "sc_catalog" / "v1.yaml")
    strings: dict[str, str] = {}
    for sc in catalog.constraints:
        for framed in all_framings(sc):
            key = f"{sc.id:02d}_{framed.strength.value}_{framed.explicitness.value}"
            strings[key] = framed.rendered_text
    return {
        "_comment": (
            "15 SCs x 4 framings = 60 byte exact strings. Spec 6.7 and TASK-004. "
            "Any change here invalidates every downstream result and requires reviewer sign off."
        ),
        "template_version": TEMPLATE_VERSION,
        "catalog_version": catalog.version,
        "count": len(strings),
        "strings": strings,
    }


def build_prompt_hashes() -> dict[str, object]:
    registry = PromptRegistry(ROOT / "prompts")
    missing = [req.prompt_id for req in registry.missing_required()]
    return {
        "_comment": (
            "Prompt content hashes. Spec 11.4: a one character prompt change silently "
            "invalidates cross run comparisons, so changing a hash must be deliberate."
        ),
        "hashes": registry.hashes(),
        "unfetched_required_prompts": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="write the files")
    parser.add_argument("--target", choices=["framing", "prompts", "all"], default="all")
    args = parser.parse_args()

    targets: dict[str, tuple[Path, dict[str, object]]] = {}
    if args.target in ("framing", "all"):
        targets["framing"] = (GOLDEN / "framing_60_strings.json", build_framing_golden())
    if args.target in ("prompts", "all"):
        targets["prompts"] = (GOLDEN / "prompt_hashes.json", build_prompt_hashes())

    for name, (path, payload) in targets.items():
        rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        if path.exists():
            current = path.read_text(encoding="utf-8")
            if current == rendered:
                print(f"{name}: unchanged")
                continue
            print(f"{name}: WOULD CHANGE {path}")
        else:
            print(f"{name}: WOULD CREATE {path}")
        if not args.confirm:
            print("  (dry run; pass --confirm to write)")
            continue
        path.write_text(rendered, encoding="utf-8")
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
