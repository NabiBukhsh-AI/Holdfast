"""One contract, every RegistryStore implementation. TASK-022.

The in-memory store exists to mirror Postgres. That claim is only worth anything if both are
held to the SAME tests, so the behaviour lives here once and is parametrized over backends
rather than written twice with two sets of expectations.

The in-memory backend always runs. The Postgres backend runs when `HOLDFAST_TEST_PG_DSN` is
set, and is skipped otherwise, so CI stays fast and offline while a developer or a nightly job
with a database gets real coverage of the same assertions.

    HOLDFAST_TEST_PG_DSN=postgresql://localhost/holdfast_test pytest tests/contract -q
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from scguard.registry.cache import CachedRegistryStore, InMemoryRegistryCache
from scguard.registry.store import (
    AppendOnlyViolationError,
    DuplicateConstraintError,
    InMemoryRegistryStore,
    RegistryStore,
    SCCategory,
    SCStatus,
    Session,
    SessionNotFoundError,
    build_constraint,
)

REPO = Path(__file__).resolve().parents[2]
PG_DSN = os.environ.get("HOLDFAST_TEST_PG_DSN")

SESSION = "sess_contract"
TENANT = "tenant_contract"


async def make_memory_store() -> AsyncIterator[RegistryStore]:
    yield InMemoryRegistryStore()


async def make_cached_memory_store() -> AsyncIterator[RegistryStore]:
    """The cache wrapper must satisfy the same contract as a bare store.

    If wrapping a store in a cache changed its observable behaviour, the cache would be a
    correctness risk rather than a latency optimisation.
    """
    yield CachedRegistryStore(InMemoryRegistryStore(), InMemoryRegistryCache())


async def make_postgres_store() -> AsyncIterator[RegistryStore]:
    from scguard.registry.postgres_store import PostgresRegistryStore

    assert PG_DSN is not None
    store = await PostgresRegistryStore.connect(PG_DSN)
    schema = (REPO / "migrations" / "001_scguard_schema.sql").read_text(encoding="utf-8")
    await store.apply_migrations(schema)
    # Each test gets a clean session namespace; the schema itself is shared.
    await store._pool.execute(
        "DELETE FROM session_constraints WHERE session_id LIKE 'sess_contract%'"
    )
    await store._pool.execute("DELETE FROM sessions WHERE session_id LIKE 'sess_contract%'")
    try:
        yield store
    finally:
        await store.aclose()


BACKENDS = [
    pytest.param(make_memory_store, id="memory"),
    pytest.param(make_cached_memory_store, id="memory+cache"),
    pytest.param(
        make_postgres_store,
        id="postgres",
        marks=pytest.mark.skipif(
            PG_DSN is None,
            reason="set HOLDFAST_TEST_PG_DSN to run the contract against Postgres",
        ),
    ),
]


@pytest.fixture(params=BACKENDS)
async def store(request) -> AsyncIterator[RegistryStore]:  # type: ignore[no-untyped-def]
    async for built in request.param():
        await built.create_session(
            Session(
                session_id=SESSION,
                tenant_id=TENANT,
                extractor_model="qwen3.5-9b",
                prompt_hash="sha256:abc",
            )
        )
        yield built


def constraint(text: str, category: SCCategory = SCCategory.ACTION, seq: int = 0, tokens: int = 12):
    return build_constraint(
        session_id=SESSION,
        tenant_id=TENANT,
        seq=seq,
        canonical_text=text,
        category=category,
        source_turn_index=seq,
        token_count=tokens,
    )


# ---------------------------------------------------------------- sessions


async def test_create_session_is_idempotent(store: RegistryStore) -> None:
    again = await store.create_session(
        Session(
            session_id=SESSION,
            tenant_id=TENANT,
            extractor_model="different-model",
            prompt_hash="sha256:different",
        )
    )
    assert again.session_id == SESSION
    # The original wins; a re-create must not silently rewrite provenance.
    assert again.extractor_model == "qwen3.5-9b"


async def test_unknown_session_raises(store: RegistryStore) -> None:
    with pytest.raises(SessionNotFoundError):
        await store.get_session("sess_contract_missing")
    with pytest.raises(SessionNotFoundError):
        await store.active("sess_contract_missing")


# ---------------------------------------------------------------- append


async def test_append_and_read_active(store: RegistryStore) -> None:
    await store.append(constraint("Draft emails, never send them.", seq=0))
    await store.append(constraint("Use metric units.", SCCategory.PREFERENCE, seq=1))
    active = await store.active(SESSION)
    assert [row.canonical_text for row in active] == [
        "Draft emails, never send them.",
        "Use metric units.",
    ]
    assert all(row.is_active for row in active)


async def test_seq_is_monotonic_within_a_session(store: RegistryStore) -> None:
    for index in range(4):
        await store.append(constraint(f"constraint number {index}", seq=index))
    seqs = [row.seq for row in await store.all_constraints(SESSION)]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


async def test_duplicate_normalized_text_is_rejected(store: RegistryStore) -> None:
    """uq_session_normalized: two workers racing on the same text produce one row."""
    await store.append(constraint("Draft emails, never send them.", seq=0))
    with pytest.raises(DuplicateConstraintError):
        await store.append(constraint("  DRAFT   emails,  never send them. ", seq=1))
    assert len(await store.all_constraints(SESSION)) == 1


async def test_registry_version_advances_on_every_mutation(store: RegistryStore) -> None:
    start = (await store.get_session(SESSION)).registry_version
    row = await store.append(constraint("Never send email.", seq=0))
    after_append = (await store.get_session(SESSION)).registry_version
    assert after_append > start

    await store.set_status(SESSION, row.constraint_id, SCStatus.REVOKED, None)
    assert (await store.get_session(SESSION)).registry_version > after_append


# ---------------------------------------------------------------- append only


async def test_text_cannot_be_rewritten(store: RegistryStore) -> None:
    """FR-080. A registry that can lie about what the user asked for is unauditable."""
    row = await store.append(constraint("Never send email.", seq=0))
    with pytest.raises(AppendOnlyViolationError):
        await store.replace_text(SESSION, row.constraint_id, "Always send email.")  # type: ignore[attr-defined]


async def test_revocation_tombstones_rather_than_deletes(store: RegistryStore) -> None:
    row = await store.append(constraint("Never send email.", seq=0))
    await store.set_status(SESSION, row.constraint_id, SCStatus.REVOKED, None)

    assert await store.active(SESSION) == ()
    everything = await store.all_constraints(SESSION)
    assert len(everything) == 1
    assert everything[0].status is SCStatus.REVOKED
    assert everything[0].canonical_text == "Never send email.", "text survives revocation"


async def test_supersession_requires_a_pointer(store: RegistryStore) -> None:
    """ck_supersede_status, enforced identically in both backends."""
    row = await store.append(constraint("Confirm before running commands.", seq=0))
    with pytest.raises(AppendOnlyViolationError):
        await store.set_status(SESSION, row.constraint_id, SCStatus.SUPERSEDED, None)


async def test_non_superseded_status_rejects_a_pointer(store: RegistryStore) -> None:
    row = await store.append(constraint("Never send email.", seq=0))
    with pytest.raises(AppendOnlyViolationError):
        await store.set_status(SESSION, row.constraint_id, SCStatus.REVOKED, "sc_other")


async def test_supersession_records_the_pointer(store: RegistryStore) -> None:
    old = await store.append(constraint("Confirm before running commands.", seq=0))
    new = await store.append(constraint("Never confirm, just run commands.", seq=1))
    tombstoned = await store.set_status(
        SESSION, old.constraint_id, SCStatus.SUPERSEDED, new.constraint_id
    )
    assert tombstoned.status is SCStatus.SUPERSEDED
    assert tombstoned.superseded_by == new.constraint_id
    assert tombstoned.status_changed_at is not None
    assert [row.constraint_id for row in await store.active(SESSION)] == [new.constraint_id]


async def test_status_change_preserves_immutable_fields(store: RegistryStore) -> None:
    row = await store.append(constraint("Never send email.", seq=0))
    updated = await store.set_status(SESSION, row.constraint_id, SCStatus.EVICTED, None)
    for field in ("canonical_text", "normalized_text", "source_turn_index", "seq", "created_at"):
        assert getattr(updated, field) == getattr(row, field), f"{field} must be immutable"


async def test_setting_status_on_an_unknown_constraint_raises(store: RegistryStore) -> None:
    with pytest.raises(SessionNotFoundError):
        await store.set_status(SESSION, "sc_does_not_exist", SCStatus.REVOKED, None)


# ---------------------------------------------------------------- ordering and filtering


async def test_active_excludes_every_tombstone_state(store: RegistryStore) -> None:
    rows = [await store.append(constraint(f"constraint {i}", seq=i)) for i in range(4)]
    await store.set_status(SESSION, rows[1].constraint_id, SCStatus.REVOKED, None)
    await store.set_status(SESSION, rows[2].constraint_id, SCStatus.EVICTED, None)
    await store.set_status(
        SESSION, rows[3].constraint_id, SCStatus.SUPERSEDED, rows[0].constraint_id
    )
    active = await store.active(SESSION)
    assert [row.constraint_id for row in active] == [rows[0].constraint_id]
    assert len(await store.all_constraints(SESSION)) == 4, "nothing is deleted"


async def test_active_is_ordered_by_seq(store: RegistryStore) -> None:
    """The rendered block reads in the order the user issued the constraints."""
    for index in range(5):
        await store.append(constraint(f"constraint number {index}", seq=index))
    active = await store.active(SESSION)
    assert [row.seq for row in active] == sorted(row.seq for row in active)
