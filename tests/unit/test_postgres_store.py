"""Postgres store logic that does not need a server. TASK-022.

The behavioural contract lives in `tests/contract/test_registry_store_contract.py` and runs
against a real database when `HOLDFAST_TEST_PG_DSN` is set. What is tested HERE is the part a
real database would exercise only under concurrency, and therefore flakily: how the repository
classifies constraint violations.

That classification is the subtle bit. Two unique constraints on the same table mean completely
different things:

    uq_session_seq          two workers raced for a sequence number  -> transient, retry
    uq_session_normalized   the same constraint text was submitted   -> real duplicate, raise

Collapsing them would either spin forever on a genuine duplicate or silently drop a real one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from scguard.registry.postgres_store import MAX_SEQ_RETRIES, PostgresRegistryStore
from scguard.registry.store import (
    AppendOnlyViolationError,
    DuplicateConstraintError,
    RegistryUnavailableError,
    SCCategory,
    SCStatus,
    SessionNotFoundError,
    build_constraint,
)

SESSION = "sess_pg"
TENANT = "tenant_pg"


class UniqueViolationError(Exception):
    """Stands in for asyncpg.exceptions.UniqueViolationError, which exposes constraint_name."""

    def __init__(self, constraint_name: str) -> None:
        super().__init__(f"duplicate key value violates unique constraint {constraint_name!r}")
        self.constraint_name = constraint_name


class CheckViolationError(Exception):
    def __init__(self, constraint_name: str) -> None:
        super().__init__(f"new row violates check constraint {constraint_name!r}")
        self.constraint_name = constraint_name


# NOT NULL columns in the schema, so a faithful fake never returns None for them.
NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)

SESSION_ROW = {
    "session_id": SESSION,
    "tenant_id": TENANT,
    "created_at": NOW,
    "last_activity_at": NOW,
    "registry_version": 0,
    "extractor_model": "qwen3.5-9b",
    "prompt_hash": "sha256:abc",
    "schema_version": 1,
    "expires_at": NOW,
}


def constraint_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "constraint_id": "sc_1",
        "session_id": SESSION,
        "tenant_id": TENANT,
        "seq": 0,
        "canonical_text": "Never send email.",
        "evidence_span": None,
        "category": "action",
        "status": "active",
        "superseded_by": None,
        "source_turn_index": 0,
        "token_count": 10,
        "pinned": False,
        "extractor_model": "qwen3.5-9b",
        "prompt_hash": "sha256:abc",
        "normalized_text": "never send email.",
        "created_at": NOW,
        "status_changed_at": None,
    }
    row.update(overrides)
    return row


class FakePool:
    """Records queries and replays scripted results or errors."""

    def __init__(self, *, session_exists: bool = True) -> None:
        self.session_exists = session_exists
        self.queries: list[str] = []
        self.insert_errors: list[Exception] = []
        self.insert_attempts = 0
        self.executed: list[str] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.queries.append(query)
        if "FROM sessions" in query:
            return dict(SESSION_ROW) if self.session_exists else None
        if query.strip().startswith("INSERT INTO sessions"):
            return dict(SESSION_ROW)
        if "INSERT INTO session_constraints" in query:
            self.insert_attempts += 1
            if self.insert_errors:
                raise self.insert_errors.pop(0)
            return constraint_row()
        if query.strip().startswith("UPDATE session_constraints"):
            return constraint_row(status=args[2], superseded_by=args[3])
        return None

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.queries.append(query)
        return [constraint_row()]

    async def fetchval(self, query: str, *args: Any) -> int:
        self.queries.append(query)
        return 0

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append(query)

    async def close(self) -> None:
        return None


def make_constraint(text: str = "Never send email."):
    return build_constraint(
        session_id=SESSION,
        tenant_id=TENANT,
        seq=0,
        canonical_text=text,
        category=SCCategory.ACTION,
        source_turn_index=0,
        token_count=10,
    )


# ---------------------------------------------------------------- violation classification


async def test_duplicate_text_raises_immediately_and_does_not_retry() -> None:
    """A real duplicate must surface, not spin."""
    pool = FakePool()
    pool.insert_errors = [UniqueViolationError("uq_session_normalized")]
    store = PostgresRegistryStore(pool)

    with pytest.raises(DuplicateConstraintError, match="normalized text"):
        await store.append(make_constraint())
    assert pool.insert_attempts == 1, "a duplicate must not be retried"


async def test_seq_collision_is_retried_and_then_succeeds() -> None:
    """Two workers racing for a sequence number is transient."""
    pool = FakePool()
    pool.insert_errors = [UniqueViolationError("uq_session_seq")]
    store = PostgresRegistryStore(pool)

    stored = await store.append(make_constraint())
    assert stored.constraint_id == "sc_1"
    assert pool.insert_attempts == 2, "the first attempt should have been retried"


async def test_persistent_seq_collision_gives_up_loudly() -> None:
    """Retrying forever would turn a schema problem into a hang."""
    pool = FakePool()
    pool.insert_errors = [
        UniqueViolationError("uq_session_seq") for _ in range(MAX_SEQ_RETRIES + 2)
    ]
    store = PostgresRegistryStore(pool)

    with pytest.raises(RegistryUnavailableError, match="could not allocate a sequence number"):
        await store.append(make_constraint())
    assert pool.insert_attempts == MAX_SEQ_RETRIES


async def test_supersede_check_violation_maps_to_append_only_error() -> None:
    pool = FakePool()
    pool.insert_errors = [CheckViolationError("ck_supersede_status")]
    store = PostgresRegistryStore(pool)

    with pytest.raises(AppendOnlyViolationError, match="supersession invariant"):
        await store.append(make_constraint())


async def test_an_unrecognized_error_is_not_swallowed() -> None:
    """Rule 13: an error path that degrades quietly is a bug, however convenient."""
    pool = FakePool()
    pool.insert_errors = [RuntimeError("disk full")]
    store = PostgresRegistryStore(pool)

    with pytest.raises(RuntimeError, match="disk full"):
        await store.append(make_constraint())


# ---------------------------------------------------------------- statement shape


async def test_seq_is_allocated_inside_the_insert() -> None:
    """Reading MAX(seq) and then inserting is a read-modify-write race."""
    pool = FakePool()
    store = PostgresRegistryStore(pool)
    await store.append(make_constraint())
    insert = next(q for q in pool.queries if "INSERT INTO session_constraints" in q)
    assert "SELECT COALESCE(MAX(seq) + 1, 0)" in insert, "seq must be computed in the statement"


async def test_active_filters_and_orders_in_sql() -> None:
    pool = FakePool()
    store = PostgresRegistryStore(pool)
    await store.active(SESSION)
    query = next(q for q in pool.queries if "FROM session_constraints" in q and "status" in q)
    assert "status = 'active'" in query
    assert "ORDER BY seq" in query


async def test_status_update_touches_only_status_columns() -> None:
    """The database trigger rejects any UPDATE touching text, so we never construct one.

    Only the SET clause is inspected. RETURNING legitimately reads the immutable columns back:
    reading them is how the caller gets the updated row, and it is writing them that the
    trigger forbids.
    """
    pool = FakePool()
    store = PostgresRegistryStore(pool)
    await store.set_status(SESSION, "sc_1", SCStatus.REVOKED, None)
    update = next(q for q in pool.queries if q.strip().startswith("UPDATE session_constraints"))
    set_clause = update.split("RETURNING")[0]
    for forbidden in ("canonical_text", "normalized_text", "source_turn_index", "created_at"):
        assert forbidden not in set_clause, f"{forbidden} must never be assigned in an UPDATE"


async def test_mutations_bump_the_registry_version() -> None:
    pool = FakePool()
    store = PostgresRegistryStore(pool)
    await store.append(make_constraint())
    assert any("registry_version = registry_version + 1" in q for q in pool.executed)


# ---------------------------------------------------------------- guards


async def test_supersede_without_a_pointer_is_rejected_before_sql() -> None:
    """Caught in application code so the round trip is not wasted on an invalid write."""
    pool = FakePool()
    store = PostgresRegistryStore(pool)
    with pytest.raises(AppendOnlyViolationError, match="names nothing that superseded it"):
        await store.set_status(SESSION, "sc_1", SCStatus.SUPERSEDED, None)
    assert not any("UPDATE session_constraints" in q for q in pool.queries)


async def test_pointer_on_a_non_superseded_status_is_rejected() -> None:
    pool = FakePool()
    store = PostgresRegistryStore(pool)
    with pytest.raises(AppendOnlyViolationError, match="carries a supersession pointer"):
        await store.set_status(SESSION, "sc_1", SCStatus.REVOKED, "sc_other")


async def test_operations_on_an_unknown_session_raise() -> None:
    pool = FakePool(session_exists=False)
    store = PostgresRegistryStore(pool)
    with pytest.raises(SessionNotFoundError):
        await store.get_session(SESSION)
    with pytest.raises(SessionNotFoundError):
        await store.append(make_constraint())


async def test_replace_text_exists_only_to_fail() -> None:
    store = PostgresRegistryStore(FakePool())
    with pytest.raises(AppendOnlyViolationError, match="append only"):
        await store.replace_text(SESSION, "sc_1", "rewritten")


async def test_connect_requires_the_driver() -> None:
    import sys

    from shared.errors import ConfigError

    previous = sys.modules.pop("asyncpg", None)
    sys.modules["asyncpg"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigError, match="requires asyncpg"):
            await PostgresRegistryStore.connect("postgresql://localhost/holdfast")
    finally:
        sys.modules.pop("asyncpg", None)
        if previous is not None:
            sys.modules["asyncpg"] = previous
