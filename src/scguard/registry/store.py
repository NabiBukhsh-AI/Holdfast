"""Append only registry store. TASK-022, FR-080, FR-082, spec 19.1.

`ENGINEERING RECOMMENDATION` spec 1.6 judgment 1: the registry is append only with tombstoned
revocation rather than mutable, because silent constraint deletion reintroduces the exact
failure the source research identifies. A constraint that vanishes without a trace is
indistinguishable, from the user's side, from a constraint the compactor dropped.

The in-memory backend here mirrors the SQL constraints in `migrations/001_scguard_schema.sql`
exactly, including the unique normalized text constraint and the supersession check, so the
dev and CI path cannot accept a write that Postgres would reject.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from shared.errors import HoldFastError


class SCStatus(StrEnum):
    """Lifecycle states. Every terminal state is a tombstone, never a deletion."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"
    EVICTED = "evicted"


class SCCategory(StrEnum):
    """Open enumeration in production (FR-042). `other` is a monitored metric."""

    ACTION = "action"
    INFORMATION = "information"
    PROCESS = "process"
    PREFERENCE = "preference"
    OUTPUT = "output"
    OTHER = "other"


# ENGINEERING RECOMMENDATION spec 14.7 rule 1, assumption A-10. Lower is more severe.
# Action and Information govern side effects and disclosure: losing an Output constraint
# produces a formatting error, losing an Action constraint produces an unauthorized tool call.
CATEGORY_SEVERITY: dict[SCCategory, int] = {
    SCCategory.ACTION: 1,
    SCCategory.INFORMATION: 2,
    SCCategory.PROCESS: 3,
    SCCategory.PREFERENCE: 4,
    SCCategory.OUTPUT: 5,
    SCCategory.OTHER: 6,
}


class AppendOnlyViolationError(HoldFastError):
    """An attempt to mutate immutable constraint text or provenance."""


class DuplicateConstraintError(HoldFastError):
    """The unique (session_id, normalized_text) constraint rejected this write."""


class SessionNotFoundError(HoldFastError):
    """The session does not exist. Never auto-created on a read path."""


class RegistryUnavailableError(HoldFastError):
    """The store is unreachable.

    Surfaced as HTTP 503. Spec 18.2: never degrade to an empty registry on store failure,
    because that is exactly the silent failure this system exists to prevent.
    """


def normalize_text(text: str) -> str:
    """Casefolded, whitespace collapsed form backing the unique constraint."""
    return " ".join(text.split()).strip().casefold()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class SessionConstraint(BaseModel):
    """One registry row. Frozen: rows are replaced by new rows, never edited."""

    model_config = ConfigDict(frozen=True)

    constraint_id: str
    session_id: str
    tenant_id: str
    seq: int = Field(ge=0)
    canonical_text: str = Field(min_length=1)
    evidence_span: str | None = None
    category: SCCategory = SCCategory.OTHER
    status: SCStatus = SCStatus.ACTIVE
    superseded_by: str | None = None
    source_turn_index: int = Field(ge=0)
    token_count: int = Field(gt=0)
    pinned: bool = False
    extractor_model: str = ""
    prompt_hash: str = ""
    normalized_text: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status_changed_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        """Satisfies shared.assembly.RegistryEntryLike, so assemble() can render this row."""
        return self.status is SCStatus.ACTIVE

    @property
    def severity(self) -> int:
        return CATEGORY_SEVERITY[self.category]


class Session(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    tenant_id: str
    extractor_model: str
    prompt_hash: str
    registry_version: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    schema_version: int = 1


class RegistryStore(Protocol):
    """The persistence surface the assembly service and API depend on."""

    async def create_session(self, session: Session) -> Session: ...

    async def get_session(self, session_id: str) -> Session: ...

    async def append(self, constraint: SessionConstraint) -> SessionConstraint: ...

    async def active(self, session_id: str) -> tuple[SessionConstraint, ...]: ...

    async def all_constraints(self, session_id: str) -> tuple[SessionConstraint, ...]: ...

    async def set_status(
        self, session_id: str, constraint_id: str, status: SCStatus, superseded_by: str | None
    ) -> SessionConstraint: ...

    async def next_seq(self, session_id: str) -> int: ...


class InMemoryRegistryStore:
    """Reference implementation, used by dev and CI.

    Mirrors every database level constraint so a write that would fail in Postgres also fails
    here. Where the two could drift, the SQL is authoritative and this class is the thing that
    must be corrected.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._constraints: dict[str, dict[str, SessionConstraint]] = {}
        self._normalized: dict[str, dict[str, str]] = {}
        self.available = True

    def _check_available(self) -> None:
        if not self.available:
            raise RegistryUnavailableError(
                "registry store is unavailable. Return 503; do NOT proceed with an empty "
                "registry (spec 18.2)."
            )

    async def create_session(self, session: Session) -> Session:
        self._check_available()
        existing = self._sessions.get(session.session_id)
        if existing is not None:
            return existing
        self._sessions[session.session_id] = session
        self._constraints[session.session_id] = {}
        self._normalized[session.session_id] = {}
        return session

    async def get_session(self, session_id: str) -> Session:
        self._check_available()
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"session {session_id} does not exist")
        return session

    async def next_seq(self, session_id: str) -> int:
        self._check_available()
        rows = self._constraints.get(session_id)
        if rows is None:
            raise SessionNotFoundError(f"session {session_id} does not exist")
        return len(rows)

    async def append(self, constraint: SessionConstraint) -> SessionConstraint:
        """Insert one row. Enforces uq_session_normalized and ck_supersede_status."""
        self._check_available()
        rows = self._constraints.get(constraint.session_id)
        if rows is None:
            raise SessionNotFoundError(f"session {constraint.session_id} does not exist")

        normalized = constraint.normalized_text or normalize_text(constraint.canonical_text)
        stored = constraint.model_copy(update={"normalized_text": normalized})
        _assert_supersede_invariant(stored)

        index = self._normalized[constraint.session_id]
        if normalized in index:
            raise DuplicateConstraintError(
                f"session {constraint.session_id} already holds a constraint with normalized "
                f"text {normalized!r} (constraint {index[normalized]})"
            )
        if any(row.seq == stored.seq for row in rows.values()):
            raise DuplicateConstraintError(
                f"session {constraint.session_id} already holds seq {stored.seq}"
            )

        rows[stored.constraint_id] = stored
        index[normalized] = stored.constraint_id
        session = self._sessions[constraint.session_id]
        self._sessions[constraint.session_id] = session.model_copy(
            update={
                "registry_version": session.registry_version + 1,
                "last_activity_at": datetime.now(UTC),
            }
        )
        return stored

    async def active(self, session_id: str) -> tuple[SessionConstraint, ...]:
        self._check_available()
        rows = self._constraints.get(session_id)
        if rows is None:
            raise SessionNotFoundError(f"session {session_id} does not exist")
        return tuple(
            sorted(
                (row for row in rows.values() if row.status is SCStatus.ACTIVE),
                key=lambda row: row.seq,
            )
        )

    async def all_constraints(self, session_id: str) -> tuple[SessionConstraint, ...]:
        self._check_available()
        rows = self._constraints.get(session_id)
        if rows is None:
            raise SessionNotFoundError(f"session {session_id} does not exist")
        return tuple(sorted(rows.values(), key=lambda row: row.seq))

    async def set_status(
        self,
        session_id: str,
        constraint_id: str,
        status: SCStatus,
        superseded_by: str | None = None,
    ) -> SessionConstraint:
        """Transition a row's status. The only legal mutation: text is never rewritten."""
        self._check_available()
        rows = self._constraints.get(session_id)
        if rows is None:
            raise SessionNotFoundError(f"session {session_id} does not exist")
        current = rows.get(constraint_id)
        if current is None:
            raise SessionNotFoundError(
                f"constraint {constraint_id} does not exist in session {session_id}"
            )
        updated = current.model_copy(
            update={
                "status": status,
                "superseded_by": superseded_by,
                "status_changed_at": datetime.now(UTC),
            }
        )
        _assert_supersede_invariant(updated)
        _assert_append_only(current, updated)
        rows[constraint_id] = updated
        session = self._sessions[session_id]
        self._sessions[session_id] = session.model_copy(
            update={"registry_version": session.registry_version + 1}
        )
        return updated

    async def replace_text(
        self,
        session_id: str,  # noqa: ARG002 - present so a text rewrite cannot even be expressed
        constraint_id: str,
        text: str,  # noqa: ARG002 - see above
    ) -> None:
        """Exists only to fail. TASK-022 acceptance: updating text via the repository raises.

        The parameters are unused on purpose. They exist so that the shape of a text rewrite is
        representable in the type system and always terminates in a raise, rather than being
        absent and therefore quietly added later by someone who needs it.
        """
        raise AppendOnlyViolationError(
            f"session_constraints is append only (FR-080). Constraint {constraint_id} text is "
            "immutable; tombstone it and append a replacement instead."
        )

    def set_available(self, available: bool) -> None:
        """Test hook for the store-down path, which must surface as 503, never as empty."""
        self.available = available


def _assert_supersede_invariant(row: SessionConstraint) -> None:
    """ck_supersede_status, mirrored in application code."""
    if row.status is SCStatus.SUPERSEDED and row.superseded_by is None:
        raise AppendOnlyViolationError(
            f"constraint {row.constraint_id} is superseded but names nothing that superseded "
            "it; the supersession pointer is what makes the history reconstructible"
        )
    if row.status is not SCStatus.SUPERSEDED and row.superseded_by is not None:
        raise AppendOnlyViolationError(
            f"constraint {row.constraint_id} carries a supersession pointer but its status is "
            f"{row.status.value}"
        )


def _assert_append_only(before: SessionConstraint, after: SessionConstraint) -> None:
    """Mirror of the database trigger: text and provenance never change."""
    immutable = (
        "canonical_text",
        "evidence_span",
        "normalized_text",
        "source_turn_index",
        "created_at",
        "session_id",
        "constraint_id",
        "seq",
    )
    for field in immutable:
        if getattr(before, field) != getattr(after, field):
            raise AppendOnlyViolationError(
                f"constraint {before.constraint_id} field {field} is immutable (FR-080)"
            )


def build_constraint(
    *,
    session_id: str,
    tenant_id: str,
    seq: int,
    canonical_text: str,
    category: SCCategory,
    source_turn_index: int,
    token_count: int,
    evidence_span: str | None = None,
    pinned: bool = False,
    extractor_model: str = "",
    prompt_hash: str = "",
) -> SessionConstraint:
    """Construct a row with its id and normalized text derived, not passed in."""
    return SessionConstraint(
        constraint_id=new_id("sc"),
        session_id=session_id,
        tenant_id=tenant_id,
        seq=seq,
        canonical_text=canonical_text,
        evidence_span=evidence_span,
        category=category,
        source_turn_index=source_turn_index,
        token_count=token_count,
        pinned=pinned,
        extractor_model=extractor_model,
        prompt_hash=prompt_hash,
        normalized_text=normalize_text(canonical_text),
    )


def registry_tokens(constraints: Sequence[SessionConstraint]) -> int:
    return sum(row.token_count for row in constraints)
