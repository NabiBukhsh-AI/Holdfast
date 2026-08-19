"""Registry block delimiters and the strip function that keeps INV-7 true.

Spec 6.13 and 14.8. `ENGINEERING RECOMMENDATION`: the paper's Equation 10 is bare
concatenation. In production, compaction fires repeatedly. Without a delimiter the registry
block from compaction event n becomes input history for event n+1, gets summarized, and is
subject to exactly the compression loss the registry exists to prevent.

This module is imported by both arms. It has no dependencies beyond the standard library so
that the delimiter strings cannot drift between research and production.
"""

from __future__ import annotations

import re

REGISTRY_OPEN = "<session_constraints>"
REGISTRY_CLOSE = "</session_constraints>"

# PAPER SPECIFICATION spec 6.13 shows this preamble inside the delimited block.
REGISTRY_PREAMBLE = (
    "The following constraints were issued by the user earlier in this session "
    "and remain in effect:"
)

_BLOCK_PATTERN = re.compile(
    re.escape(REGISTRY_OPEN) + r".*?" + re.escape(REGISTRY_CLOSE),
    flags=re.DOTALL,
)


def contains_registry_block(text: str) -> bool:
    """True when `text` carries at least one delimited registry block."""
    return _BLOCK_PATTERN.search(text) is not None


def count_registry_blocks(text: str) -> int:
    """Number of delimited registry blocks. Exactly one is the post assembly invariant."""
    return len(_BLOCK_PATTERN.findall(text))


def strip_registry_blocks(text: str) -> str:
    """Remove every prior registry block from a text.

    Spec 14.8 step 3. This is what prevents the previous registry block from being fed back
    into the compactor and re-summarized. Without it the registry decays across successive
    compactions, which is the paper's failure mode with extra steps.

    Trailing whitespace left by the removal is collapsed so that repeated strip and assemble
    cycles converge instead of accumulating blank lines.
    """
    stripped = _BLOCK_PATTERN.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", stripped).rstrip()
