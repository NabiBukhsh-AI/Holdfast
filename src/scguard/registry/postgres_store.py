"""Postgres backed registry store. TASK-022, spec 19.1.

Implements the same `RegistryStore` protocol as `InMemoryRegistryStore`, against the schema in
`migrations/001_scguard_schema.sql`. The in-memory store exists to mirror this one, so the two
are held to a single shared contract test suite rather than to two sets of expectations.

Three things this module gets right that a naive repository would not:

1. **The two unique constraints mean different things.** A `uq_session_seq` violation is two
   workers racing for the same sequence number, which is transient and should be retried. A
   `uq_session_normalized` violation is a genuine duplicate constraint, which must surface as
   `DuplicateConstraintError` and must NOT be retried. Collapsing them would either lose real
   duplicates or spin forever on a race.

2. **Sequence numbers are allocated inside the INSERT.** Reading MAX(seq) and then inserting is
   a read-modify-write race. The subquery makes allocation atomic per statement, and the unique
   constraint plus a bounded retry covers the remaining window.

3. **Status transitions are the only UPDATE.** The database trigger rejects any UPDATE touching
   constraint text or provenance, so the repository never even constructs one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from scguard.registry.store import (
    AppendOnlyViolationError,
    DuplicateConstraintError,
    RegistryUnavailableError,
    SCCategory,
    SCStatus,
    Session,
    SessionConstraint,
    SessionNotFoundError,
    normalize_text,
)
from shared.errors import ConfigError

# Spec 19.1 column order, used by every SELECT so row mapping stays in one place.
CONSTRAINT_COLUMNS = """
    constraint_id, session_id, tenant_id, seq, canonical_text, evidence_span, category,
    status, superseded_by, source_turn_index, token_count, pinned, extractor_model,
    prompt_hash, normalized_text, created_at, status_changed_at
"""

SESSION_COLUMNS = """
    session_id, tenant_id, created_at, last_activity_at, registry_version, extractor_model,
    prompt_hash, schema_version, expires_at
"""

# Bounded because a seq race resolves in one or two attempts. An unbounded retry would turn a
# genuine constraint violation into a hang.
MAX_SEQ_RETRIES = 5


def _row_to_constraint(row: Any) -> SessionConstraint:
    return SessionConstraint(
        constraint_id=row["constraint_id"],
        session_id=row["session_id"],
        tenant_id=row["tenant_id"],
        seq=row["seq"],
        canonical_text=row["canonical_text"],
        evidence_span=row["evidence_span"],
        category=SCCategory(row["category"]),
        status=SCStatus(row["status"]),
        superseded_by=row["superseded_by"],
        source_turn_index=row["source_turn_index"],
        token_count=row["token_count"],
        pinned=row["pinned"],
        extractor_model=row["extractor_model"],
        prompt_hash=row["prompt_hash"],
        normalized_text=row["normalized_text"],
        created_at=row["created_at"],
        status_changed_at=row["status_changed_at"],
    )


def _row_to_session(row: Any) -> Session:
    return Session(
        session_id=row["session_id"],
        tenant_id=row["tenant_id"],
        created_at=row["created_at"],
        last_activity_at=row["last_activity_at"],
        registry_version=row["registry_version"],
        extractor_model=row["extractor_model"],
        prompt_hash=row["prompt_hash"],
        schema_version=row["schema_version"],
        expires_at=row["expires_at"],
    )


def _constraint_name(exc: Exception) -> str:
    """asyncpg exposes the violated constraint; fall back to parsing the message."""
    name = getattr(exc, "constraint_name", None)
    if isinstance(name, str) and name:
        return name
    return str(exc)


class PostgresRegistryStore:
    """Append only registry over asyncpg. Same protocol as the in-memory store."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    @classmethod
    async def connect(
        cls, dsn: str, *, min_size: int = 2, max_size: int = 10
    ) -> PostgresRegistryStore:
        try:
            import asyncpg
        except ImportError as exc:
            raise ConfigError(
                "database.backend=postgres requires asyncpg. Install the service extra, or "
                "use backend=memory for dev and CI."
            ) from exc
        pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
        if pool is None:
            raise RegistryUnavailableError(f"could not create a connection pool for {dsn}")
        return cls(pool)

    async def aclose(self) -> None:
        await self._pool.close()

    async def create_session(self, session: Session) -> Session:
        """Idempotent: an existing session is returned unchanged rather than overwritten."""
        query = f"""
            INSERT INTO sessions (
                session_id, tenant_id, created_at, last_activity_at, registry_version,
                extractor_model, prompt_hash, schema_version, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (session_id) DO NOTHING
            RETURNING {SESSION_COLUMNS}
        """
        expires = session.expires_at or datetime.now(UTC)
        row = await self._pool.fetchrow(
            query,
            session.session_id,
            session.tenant_id,
            session.created_at,
            session.last_activity_at,
            session.registry_version,
            session.extractor_model,
            session.prompt_hash,
            session.schema_version,
            expires,
        )
        if row is None:
            return await self.get_session(session.session_id)
        return _row_to_session(row)

    async def get_session(self, session_id: str) -> Session:
        row = await self._pool.fetchrow(
            f"SELECT {SESSION_COLUMNS} FROM sessions WHERE session_id = $1", session_id
        )
        if row is None:
            raise SessionNotFoundError(f"session {session_id} does not exist")
        return _row_to_session(row)

    async def next_seq(self, session_id: str) -> int:
        """Advisory only. `append` allocates its own seq atomically."""
        await self.get_session(session_id)
        value = await self._pool.fetchval(
            "SELECT COALESCE(MAX(seq) + 1, 0) FROM session_constraints WHERE session_id = $1",
            session_id,
        )
        return int(value or 0)

    async def append(self, constraint: SessionConstraint) -> SessionConstraint:
        """Insert one row, allocating `seq` inside the statement.

        Retries only on a sequence collision. A normalized text collision is a real duplicate
        and is raised immediately.
        """
        await self.get_session(constraint.session_id)
        normalized = constraint.normalized_text or normalize_text(constraint.canonical_text)
        query = f"""
            INSERT INTO session_constraints (
                constraint_id, session_id, tenant_id, seq, canonical_text, evidence_span,
                category, status, superseded_by, source_turn_index, token_count, pinned,
                extractor_model, prompt_hash, normalized_text, created_at
            )
            VALUES (
                $1, $2, $3,
                (SELECT COALESCE(MAX(seq) + 1, 0) FROM session_constraints WHERE session_id = $2),
                $4, $5, $6::sc_category, $7::sc_status, $8, $9, $10, $11, $12, $13, $14, $15
            )
            RETURNING {CONSTRAINT_COLUMNS}
        """
        last_error: Exception | None = None
        for _attempt in range(MAX_SEQ_RETRIES):
            try:
                row = await self._pool.fetchrow(
                    query,
                    constraint.constraint_id,
                    constraint.session_id,
                    constraint.tenant_id,
                    constraint.canonical_text,
                    constraint.evidence_span,
                    constraint.category.value,
                    constraint.status.value,
                    constraint.superseded_by,
                    constraint.source_turn_index,
                    constraint.token_count,
                    constraint.pinned,
                    constraint.extractor_model,
                    constraint.prompt_hash,
                    normalized,
                    constraint.created_at,
                )
            except Exception as exc:  # asyncpg raises typed subclasses; classify by constraint
                name = _constraint_name(exc)
                if "uq_session_normalized" in name:
                    raise DuplicateConstraintError(
                        f"session {constraint.session_id} already holds a constraint with "
                        f"normalized text {normalized!r}"
                    ) from exc
                if "ck_supersede_status" in name:
                    raise AppendOnlyViolationError(
                        f"constraint {constraint.constraint_id} violates the supersession "
                        "invariant: a superseded row must name what superseded it"
                    ) from exc
                if "uq_session_seq" in name:
                    # Two workers raced for the same sequence number. Transient.
                    last_error = exc
                    continue
                raise
            if row is None:
                raise RegistryUnavailableError("insert returned no row")
            await self._bump_version(constraint.session_id)
            return _row_to_constraint(row)

        raise RegistryUnavailableError(
            f"could not allocate a sequence number for session {constraint.session_id} after "
            f"{MAX_SEQ_RETRIES} attempts: {last_error}"
        )

    async def active(self, session_id: str) -> tuple[SessionConstraint, ...]:
        await self.get_session(session_id)
        rows = await self._pool.fetch(
            f"""SELECT {CONSTRAINT_COLUMNS} FROM session_constraints
                WHERE session_id = $1 AND status = 'active' ORDER BY seq""",
            session_id,
        )
        return tuple(_row_to_constraint(row) for row in rows)

    async def all_constraints(self, session_id: str) -> tuple[SessionConstraint, ...]:
        await self.get_session(session_id)
        rows = await self._pool.fetch(
            f"SELECT {CONSTRAINT_COLUMNS} FROM session_constraints WHERE session_id = $1 ORDER BY seq",
            session_id,
        )
        return tuple(_row_to_constraint(row) for row in rows)

    async def set_status(
        self,
        session_id: str,
        constraint_id: str,
        status: SCStatus,
        superseded_by: str | None = None,
    ) -> SessionConstraint:
        """The only legal UPDATE. Text and provenance are immutable, enforced by a trigger."""
        if status is SCStatus.SUPERSEDED and superseded_by is None:
            raise AppendOnlyViolationError(
                f"constraint {constraint_id} is superseded but names nothing that superseded "
                "it; the supersession pointer is what makes the history reconstructible"
            )
        if status is not SCStatus.SUPERSEDED and superseded_by is not None:
            raise AppendOnlyViolationError(
                f"constraint {constraint_id} carries a supersession pointer but its status is "
                f"{status.value}"
            )
        row = await self._pool.fetchrow(
            f"""UPDATE session_constraints
                SET status = $3::sc_status, superseded_by = $4, status_changed_at = $5
                WHERE session_id = $1 AND constraint_id = $2
                RETURNING {CONSTRAINT_COLUMNS}""",
            session_id,
            constraint_id,
            status.value,
            superseded_by,
            datetime.now(UTC),
        )
        if row is None:
            raise SessionNotFoundError(
                f"constraint {constraint_id} does not exist in session {session_id}"
            )
        await self._bump_version(session_id)
        return _row_to_constraint(row)

    async def replace_text(
        self,
        session_id: str,  # noqa: ARG002 - present so a text rewrite cannot be expressed
        constraint_id: str,
        text: str,  # noqa: ARG002 - see above
    ) -> None:
        """Exists only to fail, mirroring the in-memory store and the database trigger."""
        raise AppendOnlyViolationError(
            f"session_constraints is append only (FR-080). Constraint {constraint_id} text is "
            "immutable; tombstone it and append a replacement instead."
        )

    async def _bump_version(self, session_id: str) -> None:
        await self._pool.execute(
            """UPDATE sessions SET registry_version = registry_version + 1,
               last_activity_at = $2 WHERE session_id = $1""",
            session_id,
            datetime.now(UTC),
        )

    async def record_audit(
        self,
        session_id: str,
        tenant_id: str,
        event_type: str,
        *,
        constraint_id: str | None = None,
        turn_index: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist one audit event. FR-082: the stream must survive process restart."""
        await self._pool.execute(
            """INSERT INTO audit_events
               (session_id, tenant_id, event_type, constraint_id, turn_index, payload)
               VALUES ($1, $2, $3, $4, $5, $6::jsonb)""",
            session_id,
            tenant_id,
            event_type,
            constraint_id,
            turn_index,
            json.dumps(payload or {}),
        )

    async def apply_migrations(self, sql: str) -> None:
        """Run the schema DDL. Used by the integration test fixture and by first deploy."""
        await self._pool.execute(sql)
