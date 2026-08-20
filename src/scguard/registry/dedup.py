"""Registry update: deduplication, conflict resolution, and budget. TASK-023, Algorithm 14.6.

Tiered, cheapest first:

    Tier 1  exact text match after normalization        -> DUPLICATE
    Tier 2  embedding cosine similarity above tau_dup   -> SEMANTIC DUPLICATE
    Tier 3  same action class, opposing polarity        -> escalate to tier 4
    Tier 4  adjudication: DUPLICATE, CONFLICT, INDEPENDENT

`UNKNOWN` tau_dup is not a value that can be guessed. `RegistryConfig.require_tau_dup()` raises
if it is unset, and the resolution path is a labelled pair set plus a reported ROC.

`registry.mode: paper_flat_list` disables tiers 3 and 4 and disables tombstoning, reproducing
the source research's append only list behavior exactly. Production behavior must never leak
into a reproduction run.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from scguard.audit.emitter import AuditEmitter, AuditEventType
from scguard.registry.conflicts import (
    Adjudication,
    Adjudicator,
    find_conflict_candidates,
)
from scguard.registry.store import (
    RegistryStore,
    SCCategory,
    SCStatus,
    SessionConstraint,
    build_constraint,
    normalize_text,
)


class CandidateOutcome(StrEnum):
    ADDED = "ADDED"
    DUPLICATE_EXACT = "DUPLICATE_EXACT"
    DUPLICATE_SEMANTIC = "DUPLICATE_SEMANTIC"
    SUPERSEDED_EXISTING = "SUPERSEDED_EXISTING"


class CandidateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    canonical_text: str
    outcome: CandidateOutcome
    constraint_id: str | None = None
    superseded_ids: tuple[str, ...] = ()
    matched_id: str | None = None
    similarity: float | None = None
    detail: str = ""


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Both inputs are L2 normalized by the embedder, so this is a dot product."""
    return float(np.dot(a.ravel(), b.ravel()))


class RegistryUpdater:
    """Algorithm 14.6. Maintains a correct, bounded, auditable constraint registry."""

    def __init__(
        self,
        store: RegistryStore,
        audit: AuditEmitter,
        *,
        mode: str = "production",
        tau_dup: float | None = None,
        embedder: Embedder | None = None,
        adjudicator: Adjudicator | None = None,
        extractor_model: str = "",
        prompt_hash: str = "",
    ) -> None:
        self._store = store
        self._audit = audit
        self._mode = mode
        self._tau_dup = tau_dup
        self._embedder = embedder
        self._adjudicator = adjudicator
        self._extractor_model = extractor_model
        self._prompt_hash = prompt_hash

    @property
    def store(self) -> RegistryStore:
        """The backing store, so callers can read the active registry without reaching in."""
        return self._store

    @property
    def conflict_detection_enabled(self) -> bool:
        """Tiers 3 and 4 are production only. paper_flat_list reproduces the source exactly."""
        return self._mode != "paper_flat_list" and self._adjudicator is not None

    async def _semantic_duplicate(
        self, text: str, active: Sequence[SessionConstraint]
    ) -> tuple[SessionConstraint | None, float | None]:
        """Tier 2. Returns the best match above tau_dup, if any."""
        if self._embedder is None or self._tau_dup is None or not active:
            return None, None
        vectors = self._embedder.encode([text] + [row.canonical_text for row in active])
        query = vectors[0]
        best: tuple[SessionConstraint | None, float] = (None, -1.0)
        for row, vector in zip(active, vectors[1:], strict=True):
            score = cosine_similarity(query, vector)
            if score > best[1]:
                best = (row, score)
        if best[0] is not None and best[1] >= self._tau_dup:
            return best[0], best[1]
        return None, best[1] if best[0] is not None else None

    async def add_candidate(
        self,
        session_id: str,
        tenant_id: str,
        *,
        canonical_text: str,
        category: SCCategory,
        turn_index: int,
        token_count: int,
        evidence_span: str | None = None,
        pinned: bool = False,
    ) -> CandidateResult:
        """Steps 1 through 11 for one candidate."""
        active = await self._store.active(session_id)
        normalized = normalize_text(canonical_text)

        # Tier 1: exact duplicate after normalization.
        for row in active:
            if row.normalized_text == normalized:
                self._audit.emit(
                    session_id,
                    tenant_id,
                    AuditEventType.CONSTRAINT_DUPLICATE_SUPPRESSED,
                    constraint_id=row.constraint_id,
                    turn_index=turn_index,
                    tier=1,
                    canonical_text=canonical_text,
                )
                return CandidateResult(
                    canonical_text=canonical_text,
                    outcome=CandidateOutcome.DUPLICATE_EXACT,
                    matched_id=row.constraint_id,
                    detail="normalized text already present",
                )

        # Tier 2: semantic duplicate.
        match, similarity = await self._semantic_duplicate(canonical_text, active)
        if match is not None:
            self._audit.emit(
                session_id,
                tenant_id,
                AuditEventType.CONSTRAINT_DUPLICATE_SUPPRESSED,
                constraint_id=match.constraint_id,
                turn_index=turn_index,
                tier=2,
                similarity=similarity,
                canonical_text=canonical_text,
            )
            return CandidateResult(
                canonical_text=canonical_text,
                outcome=CandidateOutcome.DUPLICATE_SEMANTIC,
                matched_id=match.constraint_id,
                similarity=similarity,
                detail=f"cosine {similarity:.3f} at or above tau_dup {self._tau_dup}",
            )

        # Tiers 3 and 4: conflict detection, production only.
        superseded: list[str] = []
        if self.conflict_detection_enabled:
            assert self._adjudicator is not None
            for flagged in find_conflict_candidates(canonical_text, active):
                verdict = await self._adjudicator.adjudicate(
                    flagged.existing.canonical_text, canonical_text
                )
                if verdict is Adjudication.DUPLICATE:
                    self._audit.emit(
                        session_id,
                        tenant_id,
                        AuditEventType.CONSTRAINT_DUPLICATE_SUPPRESSED,
                        constraint_id=flagged.existing.constraint_id,
                        turn_index=turn_index,
                        tier=4,
                        canonical_text=canonical_text,
                    )
                    return CandidateResult(
                        canonical_text=canonical_text,
                        outcome=CandidateOutcome.DUPLICATE_SEMANTIC,
                        matched_id=flagged.existing.constraint_id,
                        detail="tier 4 adjudicated DUPLICATE",
                    )
                if verdict is Adjudication.CONFLICT:
                    superseded.append(flagged.existing.constraint_id)
                # INDEPENDENT: a refinement. Both stay active.

        seq = await self._store.next_seq(session_id)
        constraint = build_constraint(
            session_id=session_id,
            tenant_id=tenant_id,
            seq=seq,
            canonical_text=canonical_text,
            category=category,
            source_turn_index=turn_index,
            token_count=token_count,
            evidence_span=evidence_span,
            pinned=pinned,
            extractor_model=self._extractor_model,
            prompt_hash=self._prompt_hash,
        )
        stored = await self._store.append(constraint)

        # Newest wins; the older constraint is TOMBSTONED, never deleted, and the supersession
        # is surfaced so a user can be told which of their instructions was replaced.
        for old_id in superseded:
            await self._store.set_status(
                session_id, old_id, SCStatus.SUPERSEDED, stored.constraint_id
            )
            self._audit.emit(
                session_id,
                tenant_id,
                AuditEventType.CONSTRAINT_SUPERSEDED,
                constraint_id=old_id,
                turn_index=turn_index,
                superseded_by=stored.constraint_id,
                superseding_text=canonical_text,
            )

        self._audit.emit(
            session_id,
            tenant_id,
            AuditEventType.CONSTRAINT_ADDED,
            constraint_id=stored.constraint_id,
            turn_index=turn_index,
            canonical_text=stored.canonical_text,
            category=stored.category.value,
            token_count=stored.token_count,
        )
        return CandidateResult(
            canonical_text=canonical_text,
            outcome=(
                CandidateOutcome.SUPERSEDED_EXISTING if superseded else CandidateOutcome.ADDED
            ),
            constraint_id=stored.constraint_id,
            superseded_ids=tuple(superseded),
        )
