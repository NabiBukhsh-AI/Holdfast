"""Audit event emission and point in time reconstruction. TASK-028, FR-082.

Every registry mutation produces an immutable audit record sufficient to reconstruct registry
state at any prior turn. That requirement is not bookkeeping: when a user asks why the agent
did something, the answer has to be recoverable from the log, and when a constraint was
evicted the log is the only place that fact survives.

`GET /constraints?as_of_turn=N` is served entirely from this stream, so
`reconstruct_at_turn()` is the function that requirement lives or dies on.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scguard.registry.store import SCCategory, SCStatus


class AuditEventType(StrEnum):
    """Spec 18.6 enumerates the stream. Each maps to one registry state transition."""

    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_SUPERSEDED = "constraint_superseded"
    CONSTRAINT_REVOKED = "constraint_revoked"
    CONSTRAINT_DUPLICATE_SUPPRESSED = "constraint_duplicate_suppressed"
    REGISTRY_EVICTED = "registry_evicted"
    ASSEMBLY_PERFORMED = "assembly_performed"
    EXTRACTION_FAILED = "extraction_failed"
    HALLUCINATED_EVIDENCE_REJECTED = "hallucinated_evidence_rejected"


# Spec 14.7: eviction is the paper's failure mode reintroduced, so it must be the single
# loudest event in the system. These are never debug level, whatever the log config says.
LOUD_EVENTS: frozenset[AuditEventType] = frozenset(
    {
        AuditEventType.REGISTRY_EVICTED,
        AuditEventType.EXTRACTION_FAILED,
        AuditEventType.CONSTRAINT_SUPERSEDED,
        AuditEventType.HALLUCINATED_EVIDENCE_REJECTED,
    }
)


class AuditEvent(BaseModel):
    """One immutable record. Append only, never updated, never deleted except by DSR."""

    model_config = ConfigDict(frozen=True)

    event_id: int
    session_id: str
    tenant_id: str
    event_type: AuditEventType
    constraint_id: str | None = None
    turn_index: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_loud(self) -> bool:
        return self.event_type in LOUD_EVENTS


class ReconstructedConstraint(BaseModel):
    """Registry state for one constraint as of a given turn."""

    model_config = ConfigDict(frozen=True)

    constraint_id: str
    canonical_text: str
    category: SCCategory
    status: SCStatus
    source_turn_index: int
    superseded_by: str | None = None


class AuditEmitter:
    """In-process audit sink. The Postgres backed implementation shares this interface."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._next_id = 1

    def emit(
        self,
        session_id: str,
        tenant_id: str,
        event_type: AuditEventType,
        *,
        constraint_id: str | None = None,
        turn_index: int | None = None,
        **payload: Any,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=self._next_id,
            session_id=session_id,
            tenant_id=tenant_id,
            event_type=event_type,
            constraint_id=constraint_id,
            turn_index=turn_index,
            payload=payload,
        )
        self._next_id += 1
        self._events.append(event)
        return event

    def events(
        self, session_id: str | None = None, event_type: AuditEventType | None = None
    ) -> tuple[AuditEvent, ...]:
        selected = [
            event
            for event in self._events
            if (session_id is None or event.session_id == session_id)
            and (event_type is None or event.event_type is event_type)
        ]
        return tuple(selected)

    def loud_events(self, session_id: str | None = None) -> tuple[AuditEvent, ...]:
        return tuple(event for event in self.events(session_id) if event.is_loud)

    def __len__(self) -> int:
        return len(self._events)


def reconstruct_at_turn(
    events: Sequence[AuditEvent], as_of_turn: int
) -> tuple[ReconstructedConstraint, ...]:
    """Rebuild registry state at turn N from the audit stream ALONE. FR-082.

    The current table is deliberately not consulted. If reconstruction needed it, the audit
    log would not actually be sufficient, and the requirement would be satisfied only in
    appearance.
    """
    state: dict[str, ReconstructedConstraint] = {}
    ordered = sorted(events, key=lambda event: event.event_id)

    for event in ordered:
        # A mutation at a later turn has not happened yet, from the perspective of turn N.
        if event.turn_index is not None and event.turn_index > as_of_turn:
            continue

        if event.event_type is AuditEventType.CONSTRAINT_ADDED:
            if event.constraint_id is None:
                raise ValueError(f"event {event.event_id} adds a constraint with no id")
            state[event.constraint_id] = ReconstructedConstraint(
                constraint_id=event.constraint_id,
                canonical_text=str(event.payload["canonical_text"]),
                category=SCCategory(event.payload.get("category", "other")),
                status=SCStatus.ACTIVE,
                source_turn_index=event.turn_index if event.turn_index is not None else 0,
            )
        elif event.event_type is AuditEventType.CONSTRAINT_SUPERSEDED:
            existing = state.get(event.constraint_id or "")
            if existing is not None:
                state[existing.constraint_id] = existing.model_copy(
                    update={
                        "status": SCStatus.SUPERSEDED,
                        "superseded_by": event.payload.get("superseded_by"),
                    }
                )
        elif event.event_type is AuditEventType.CONSTRAINT_REVOKED:
            existing = state.get(event.constraint_id or "")
            if existing is not None:
                state[existing.constraint_id] = existing.model_copy(
                    update={"status": SCStatus.REVOKED}
                )
        elif event.event_type is AuditEventType.REGISTRY_EVICTED:
            existing = state.get(event.constraint_id or "")
            if existing is not None:
                state[existing.constraint_id] = existing.model_copy(
                    update={"status": SCStatus.EVICTED}
                )

    return tuple(sorted(state.values(), key=lambda row: row.source_turn_index))


def active_at_turn(
    events: Sequence[AuditEvent], as_of_turn: int
) -> tuple[ReconstructedConstraint, ...]:
    """The subset that would have been injected at turn N."""
    return tuple(
        row for row in reconstruct_at_turn(events, as_of_turn) if row.status is SCStatus.ACTIVE
    )
