"""Research mode registry: a flat append only list. TASK-020.

PAPER SPECIFICATION: S^t is a running append only list of SCs extracted from x^0_U .. x^t_U.
The paper's registry has no tombstones, no conflict detection, no budget, and no revocation.
This module reproduces exactly that, so a research run measures the paper's mechanism rather
than the hardened production one.

The production registry (`scguard.registry`) adds tombstoning, conflict detection, and budget
enforcement. It is a different object on purpose. `registry.mode: paper_flat_list` selects
this one; anything else is an engineering recommendation and must never silently apply to a
reproduction run.

Entries satisfy `shared.assembly.RegistryEntryLike` structurally, which is what lets the same
assemble() serve both this and the production store without either arm importing the other.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from pydantic import BaseModel, ConfigDict

from compint.core.models import SCCategoryId
from compint.extractor.parser import ExtractedSC


class SimEntry(BaseModel):
    """One registry row in research mode."""

    model_config = ConfigDict(frozen=True)

    seq: int
    canonical_text: str
    evidence_span: str
    category: SCCategoryId
    source_turn_index: int

    @property
    def is_active(self) -> bool:
        """Always true: the paper's registry has no way to deactivate an entry."""
        return True


def normalize_text(text: str) -> str:
    """Casefolded, whitespace collapsed form used for exact duplicate suppression."""
    return " ".join(text.split()).strip().casefold()


class SimRegistry:
    """S^t as the paper defines it: append only, flat, unbounded."""

    def __init__(self) -> None:
        self._entries: list[SimEntry] = []
        self._normalized: set[str] = set()
        self.duplicates_suppressed = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[SimEntry]:
        return iter(self._entries)

    @property
    def entries(self) -> tuple[SimEntry, ...]:
        return tuple(self._entries)

    def texts(self) -> tuple[str, ...]:
        """The registry as passed into the extraction prompt for deduplication only."""
        return tuple(entry.canonical_text for entry in self._entries)

    def add_all(
        self, candidates: Sequence[ExtractedSC], *, turn_index: int
    ) -> tuple[SimEntry, ...]:
        """Append every non duplicate candidate. Returns what was actually added.

        Exact duplicate suppression happens here as a backstop. The paper's primary mechanism
        is passing the registry into the extractor prompt, which is why `texts()` exists; this
        catches the case where the model restates something anyway.
        """
        added: list[SimEntry] = []
        for candidate in candidates:
            normalized = normalize_text(candidate.canonical_text)
            if normalized in self._normalized:
                self.duplicates_suppressed += 1
                continue
            entry = SimEntry(
                seq=len(self._entries),
                canonical_text=candidate.canonical_text,
                evidence_span=candidate.evidence_span,
                category=candidate.category,
                source_turn_index=turn_index,
            )
            self._entries.append(entry)
            self._normalized.add(normalized)
            added.append(entry)
        return tuple(added)

    def by_category(self, category: SCCategoryId) -> tuple[SimEntry, ...]:
        return tuple(entry for entry in self._entries if entry.category is category)

    def token_count(self, chars_per_token: float = 4.0) -> int:
        """Approximate registry size, reported so unbounded growth is at least visible.

        Research mode does NOT enforce a budget, because the paper does not. Reporting the
        number is how the reproduction shows what an unbounded registry would cost against the
        301 to 857 token compactor output band.
        """
        total_chars = sum(len(entry.canonical_text) + 2 for entry in self._entries)
        return int(total_chars / chars_per_token)
