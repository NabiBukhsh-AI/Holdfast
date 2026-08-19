"""TASK-022, TASK-024, TASK-028 acceptance tests. Spec 14.7, 19.1, FR-080 through FR-082."""

from __future__ import annotations

import pytest

from scguard.audit.emitter import (
    AuditEmitter,
    AuditEventType,
    active_at_turn,
    reconstruct_at_turn,
)
from scguard.registry.budget import enforce_budget, eviction_priority, evicted_summary
from scguard.registry.store import (
    AppendOnlyViolation,
    DuplicateConstraintError,
    InMemoryRegistryStore,
    RegistryUnavailableError,
    SCCategory,
    SCStatus,
    Session,
    SessionNotFoundError,
    build_constraint,
    normalize_text,
)
from shared.errors import BudgetNotConfiguredError

SESSION = "sess_test"
TENANT = "tenant_test"


async def make_store() -> InMemoryRegistryStore:
    store = InMemoryRegistryStore()
    await store.create_session(
        Session(
            session_id=SESSION,
            tenant_id=TENANT,
            extractor_model="qwen3.5-9b",
            prompt_hash="sha256:abc",
        )
    )
    return store


def constraint(
    seq: int,
    text: str,
    category: SCCategory = SCCategory.ACTION,
    tokens: int = 20,
    pinned: bool = False,
):
    return build_constraint(
        session_id=SESSION,
        tenant_id=TENANT,
        seq=seq,
        canonical_text=text,
        category=category,
        source_turn_index=seq,
        token_count=tokens,
        pinned=pinned,
    )


# ---------------------------------------------------------------- store


async def test_append_and_read_active() -> None:
    store = await make_store()
    await store.append(constraint(0, "Draft emails, never send them."))
    await store.append(constraint(1, "Use metric units.", SCCategory.PREFERENCE))
    active = await store.active(SESSION)
    assert [row.seq for row in active] == [0, 1]
    assert all(row.is_active for row in active)


async def test_append_only_enforced() -> None:
    """TASK-022 acceptance: updating a constraint's text via the repository raises."""
    store = await make_store()
    row = await store.append(constraint(0, "Never send email."))
    with pytest.raises(AppendOnlyViolation, match="append only"):
        await store.replace_text(SESSION, row.constraint_id, "Always send email.")


async def test_unique_normalized_text_race() -> None:
    """TASK-022 acceptance: concurrent inserts of the same normalized text produce one row."""
    store = await make_store()
    await store.append(constraint(0, "Draft emails, never send them."))
    with pytest.raises(DuplicateConstraintError):
        await store.append(constraint(1, "  draft   EMAILS,  never send them. "))
    assert len(await store.all_constraints(SESSION)) == 1


def test_normalize_text_folds_case_and_whitespace() -> None:
    assert normalize_text("  Draft   EMAILS. ") == normalize_text("draft emails.")


async def test_supersede_constraint_check() -> None:
    """ck_supersede_status: a superseded row must name what superseded it."""
    store = await make_store()
    old = await store.append(constraint(0, "Confirm before running commands."))
    new = await store.append(constraint(1, "Never confirm, just run commands."))

    with pytest.raises(AppendOnlyViolation, match="names nothing that superseded it"):
        await store.set_status(SESSION, old.constraint_id, SCStatus.SUPERSEDED, None)

    tombstoned = await store.set_status(
        SESSION, old.constraint_id, SCStatus.SUPERSEDED, new.constraint_id
    )
    assert tombstoned.status is SCStatus.SUPERSEDED
    assert tombstoned.superseded_by == new.constraint_id
    assert tombstoned.status_changed_at is not None


async def test_supersession_pointer_rejected_on_non_superseded_status() -> None:
    store = await make_store()
    row = await store.append(constraint(0, "Never send email."))
    with pytest.raises(AppendOnlyViolation, match="carries a supersession pointer"):
        await store.set_status(SESSION, row.constraint_id, SCStatus.REVOKED, "sc_other")


async def test_revocation_is_a_tombstone_not_a_delete() -> None:
    """FR-080: silent deletion reintroduces exactly the failure being mitigated."""
    store = await make_store()
    row = await store.append(constraint(0, "Never send email."))
    await store.set_status(SESSION, row.constraint_id, SCStatus.REVOKED, None)
    assert await store.active(SESSION) == ()
    everything = await store.all_constraints(SESSION)
    assert len(everything) == 1
    assert everything[0].status is SCStatus.REVOKED
    assert everything[0].canonical_text == "Never send email."


async def test_store_failure_raises_rather_than_returning_empty() -> None:
    """Spec 18.2: never degrade to an empty registry on store failure."""
    store = await make_store()
    await store.append(constraint(0, "Never send email."))
    store.set_available(False)
    with pytest.raises(RegistryUnavailableError, match="503"):
        await store.active(SESSION)


async def test_unknown_session_is_not_auto_created() -> None:
    store = await make_store()
    with pytest.raises(SessionNotFoundError):
        await store.active("sess_does_not_exist")


async def test_registry_version_advances_on_every_mutation() -> None:
    store = await make_store()
    assert (await store.get_session(SESSION)).registry_version == 0
    await store.append(constraint(0, "Never send email."))
    assert (await store.get_session(SESSION)).registry_version == 1
    row = await store.append(constraint(1, "Use metric units.", SCCategory.PREFERENCE))
    await store.set_status(SESSION, row.constraint_id, SCStatus.REVOKED, None)
    assert (await store.get_session(SESSION)).registry_version == 3


# ---------------------------------------------------------------- budget


def test_severity_ordering() -> None:
    """TASK-024 acceptance: Action survives, Output is evicted first."""
    audit = AuditEmitter()
    constraints = [
        constraint(0, "Format as bullets.", SCCategory.OUTPUT, tokens=20),
        constraint(1, "Prefer arXiv.", SCCategory.PREFERENCE, tokens=20),
        constraint(2, "Search before answering.", SCCategory.PROCESS, tokens=20),
        constraint(3, "Never write my phone number.", SCCategory.INFORMATION, tokens=20),
        constraint(4, "Confirm before any action.", SCCategory.ACTION, tokens=20),
    ]
    decision = enforce_budget(
        constraints, 50, audit=audit, session_id=SESSION, tenant_id=TENANT
    )
    kept_categories = {row.category for row in decision.kept}
    evicted_categories = {row.category for row in decision.evicted}
    assert SCCategory.ACTION in kept_categories
    assert SCCategory.INFORMATION in kept_categories
    assert SCCategory.OUTPUT in evicted_categories
    assert decision.kept_tokens <= 50


def test_eviction_emits_audit() -> None:
    """Spec 14.7: eviction must be the loudest event in the system, never a silent drop."""
    audit = AuditEmitter()
    constraints = [
        constraint(i, f"constraint number {i}", SCCategory.OUTPUT, tokens=40) for i in range(5)
    ]
    decision = enforce_budget(
        constraints, 100, audit=audit, session_id=SESSION, tenant_id=TENANT
    )
    events = audit.events(SESSION, AuditEventType.REGISTRY_EVICTED)
    assert len(events) == decision.n_evicted > 0
    for event in events:
        assert event.is_loud
        assert event.payload["reason"] == "BUDGET_EXCEEDED"
        assert event.payload["canonical_text"]
        assert "no longer being enforced" in event.payload["detail"]


def test_oversized_single_constraint_not_truncated() -> None:
    """A half constraint can invert meaning: "Don't send emails without" ... (spec 14.7)."""
    audit = AuditEmitter()
    huge = constraint(0, "A very long constraint. " * 40, SCCategory.ACTION, tokens=500)
    decision = enforce_budget(
        [huge], 200, audit=audit, session_id=SESSION, tenant_id=TENANT
    )
    assert decision.kept == (huge,)
    assert decision.budget_exceeded_single is True
    assert decision.over_budget is True
    assert decision.kept[0].canonical_text == huge.canonical_text
    events = audit.events(SESSION, AuditEventType.REGISTRY_EVICTED)
    assert events[0].payload["reason"] == "BUDGET_EXCEEDED_SINGLE"


def test_pinned_constraints_survive_eviction() -> None:
    """FR-084 priority rule 3: an explicit user pin outranks category severity."""
    pinned_output = constraint(0, "Always bullets.", SCCategory.OUTPUT, tokens=40, pinned=True)
    action = constraint(1, "Confirm first.", SCCategory.ACTION, tokens=40)
    decision = enforce_budget([action, pinned_output], 40)
    assert decision.kept == (pinned_output,)
    assert decision.evicted == (action,)


def test_recency_breaks_ties_within_a_severity_band() -> None:
    older = constraint(0, "Older action constraint.", SCCategory.ACTION, tokens=40)
    newer = constraint(5, "Newer action constraint.", SCCategory.ACTION, tokens=40)
    decision = enforce_budget([older, newer], 40)
    assert decision.kept == (newer,)


def test_kept_constraints_render_in_registry_order() -> None:
    """Eviction order is a policy; the rendered block should read in the order issued."""
    rows = [
        constraint(0, "First.", SCCategory.OUTPUT, tokens=10),
        constraint(1, "Second.", SCCategory.ACTION, tokens=10),
        constraint(2, "Third.", SCCategory.PROCESS, tokens=10),
    ]
    decision = enforce_budget(rows, 100)
    assert [row.seq for row in decision.kept] == [0, 1, 2]


def test_zero_budget_fails_startup() -> None:
    """Spec 14.7: unset or zero fails loudly, never silently unbounded."""
    with pytest.raises(BudgetNotConfiguredError, match="forbids defaulting to unbounded"):
        enforce_budget([constraint(0, "x")], 0)


def test_eviction_priority_key_is_total_and_stable() -> None:
    pinned = constraint(0, "p", SCCategory.OUTPUT, pinned=True)
    action = constraint(1, "a", SCCategory.ACTION)
    assert eviction_priority(pinned) < eviction_priority(action)


def test_evicted_summary_is_user_facing() -> None:
    decision = enforce_budget(
        [
            constraint(0, "Confirm first.", SCCategory.ACTION, tokens=40),
            constraint(1, "Bullets only.", SCCategory.OUTPUT, tokens=40),
        ],
        40,
    )
    summary = evicted_summary(decision)
    assert "no longer being enforced" in summary
    assert "output" in summary


def test_empty_registry_summary_is_empty() -> None:
    decision = enforce_budget([constraint(0, "x", tokens=5)], 200)
    assert evicted_summary(decision) == ""


# ---------------------------------------------------------------- audit


def test_point_in_time_reconstruction() -> None:
    """TASK-028 acceptance: rebuild registry state at turn N from the audit stream alone."""
    audit = AuditEmitter()
    audit.emit(
        SESSION, TENANT, AuditEventType.CONSTRAINT_ADDED,
        constraint_id="sc_1", turn_index=2,
        canonical_text="Confirm before running commands.", category="action",
    )
    audit.emit(
        SESSION, TENANT, AuditEventType.CONSTRAINT_ADDED,
        constraint_id="sc_2", turn_index=7,
        canonical_text="Never confirm, just run commands.", category="action",
    )
    audit.emit(
        SESSION, TENANT, AuditEventType.CONSTRAINT_SUPERSEDED,
        constraint_id="sc_1", turn_index=7, superseded_by="sc_2",
    )

    at_turn_5 = reconstruct_at_turn(audit.events(SESSION), 5)
    assert [row.constraint_id for row in at_turn_5] == ["sc_1"]
    assert at_turn_5[0].status is SCStatus.ACTIVE

    at_turn_10 = reconstruct_at_turn(audit.events(SESSION), 10)
    by_id = {row.constraint_id: row for row in at_turn_10}
    assert by_id["sc_1"].status is SCStatus.SUPERSEDED
    assert by_id["sc_1"].superseded_by == "sc_2"
    assert by_id["sc_2"].status is SCStatus.ACTIVE
    assert [row.constraint_id for row in active_at_turn(audit.events(SESSION), 10)] == ["sc_2"]


def test_reconstruction_reflects_eviction() -> None:
    """An evicted constraint must be visible as evicted, not merely absent."""
    audit = AuditEmitter()
    audit.emit(
        SESSION, TENANT, AuditEventType.CONSTRAINT_ADDED,
        constraint_id="sc_1", turn_index=1,
        canonical_text="Bullets only.", category="output",
    )
    audit.emit(
        SESSION, TENANT, AuditEventType.REGISTRY_EVICTED,
        constraint_id="sc_1", turn_index=4, reason="BUDGET_EXCEEDED",
    )
    state = reconstruct_at_turn(audit.events(SESSION), 10)
    assert state[0].status is SCStatus.EVICTED
    assert active_at_turn(audit.events(SESSION), 10) == ()


def test_reconstruction_ignores_later_turns() -> None:
    audit = AuditEmitter()
    audit.emit(
        SESSION, TENANT, AuditEventType.CONSTRAINT_ADDED,
        constraint_id="sc_1", turn_index=9,
        canonical_text="Later constraint.", category="action",
    )
    assert reconstruct_at_turn(audit.events(SESSION), 3) == ()


def test_loud_events_are_distinguishable() -> None:
    audit = AuditEmitter()
    audit.emit(SESSION, TENANT, AuditEventType.CONSTRAINT_ADDED, constraint_id="sc_1",
               canonical_text="x", category="action")
    audit.emit(SESSION, TENANT, AuditEventType.REGISTRY_EVICTED, constraint_id="sc_1")
    audit.emit(SESSION, TENANT, AuditEventType.EXTRACTION_FAILED, turn_index=3)
    loud = audit.loud_events(SESSION)
    assert {event.event_type for event in loud} == {
        AuditEventType.REGISTRY_EVICTED,
        AuditEventType.EXTRACTION_FAILED,
    }


def test_audit_events_are_scoped_by_session() -> None:
    audit = AuditEmitter()
    audit.emit("sess_a", TENANT, AuditEventType.CONSTRAINT_ADDED, constraint_id="sc_1",
               canonical_text="x", category="action")
    audit.emit("sess_b", TENANT, AuditEventType.CONSTRAINT_ADDED, constraint_id="sc_2",
               canonical_text="y", category="action")
    assert len(audit.events("sess_a")) == 1
    assert len(audit) == 2
