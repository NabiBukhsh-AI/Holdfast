"""TASK-023 acceptance tests. Algorithm 14.6."""

from __future__ import annotations

import numpy as np

from compint.core.catalog import SCCatalog
from scguard.audit.emitter import AuditEmitter, AuditEventType
from scguard.registry.conflicts import (
    Adjudication,
    HeuristicAdjudicator,
    action_classes,
    class_polarity,
    conflicting_classes,
    find_conflict_candidates,
    opposing_polarity,
    polarity,
)
from scguard.registry.dedup import CandidateOutcome, RegistryUpdater
from scguard.registry.store import (
    InMemoryRegistryStore,
    SCCategory,
    SCStatus,
    Session,
)

SESSION = "sess_dedup"
TENANT = "tenant_dedup"


async def make_updater(
    *, mode: str = "production", tau_dup: float | None = None, embedder: object | None = None
) -> tuple[RegistryUpdater, InMemoryRegistryStore, AuditEmitter]:
    store = InMemoryRegistryStore()
    await store.create_session(
        Session(
            session_id=SESSION,
            tenant_id=TENANT,
            extractor_model="qwen3.5-9b",
            prompt_hash="sha256:abc",
        )
    )
    audit = AuditEmitter()
    updater = RegistryUpdater(
        store,
        audit,
        mode=mode,
        tau_dup=tau_dup,
        embedder=embedder,  # type: ignore[arg-type]
        adjudicator=HeuristicAdjudicator() if mode != "paper_flat_list" else None,
    )
    return updater, store, audit


async def add(
    updater: RegistryUpdater, text: str, category: SCCategory, turn: int, tokens: int = 15
):
    return await updater.add_candidate(
        SESSION,
        TENANT,
        canonical_text=text,
        category=category,
        turn_index=turn,
        token_count=tokens,
    )


# ---------------------------------------------------------------- polarity heuristics


def test_polarity_detection() -> None:
    assert polarity("Never send emails on my behalf.") == -1
    assert polarity("Always wait for my approval before sending.") == 1


def test_polarity_is_class_local_not_sentence_wide(catalog: SCCatalog) -> None:
    """SC 2 requires confirmation but says nothing directive about sending.

    A sentence level reading would give SC 2 a single +1 and pit it against SC 3's -1 over the
    shared `sending` class. Both constraints actually keep a human in the loop.
    """
    sc2 = catalog.by_id(2).body
    assert class_polarity(sc2, "confirmation") == 1
    assert class_polarity(sc2, "sending") == 0
    assert class_polarity(catalog.by_id(3).body, "sending") == -1


def test_compatible_restrictive_pair_is_not_a_conflict(catalog: SCCatalog) -> None:
    """SC 2 and SC 3 both keep a human in the loop and must both stay active.

    Auto-tombstoning one of them would silently drop a user constraint, which is the exact
    harm this system exists to prevent, so a false CONFLICT here is worse than a missed one.
    """
    assert conflicting_classes(catalog.by_id(2).body, catalog.by_id(3).body) == frozenset()
    assert not opposing_polarity(catalog.by_id(2).body, catalog.by_id(3).body)


async def test_compatible_pair_leaves_both_active(catalog: SCCatalog) -> None:
    updater, store, audit = await make_updater()
    await add(updater, catalog.by_id(2).body, SCCategory.ACTION, 1)
    result = await add(updater, catalog.by_id(3).body, SCCategory.ACTION, 4)
    assert result.outcome is CandidateOutcome.ADDED
    assert len(await store.active(SESSION)) == 2
    assert audit.events(SESSION, AuditEventType.CONSTRAINT_SUPERSEDED) == ()


def test_action_class_detection(catalog: SCCatalog) -> None:
    """SC 1 and SC 2 must land in the same action class or tier 3 never fires."""
    sc1 = action_classes(catalog.by_id(1).body)
    sc2 = action_classes(catalog.by_id(2).body)
    assert sc1 & sc2, f"SC1 {sc1} and SC2 {sc2} share no action class"
    assert "confirmation" in sc1 & sc2


def test_opposing_polarity_on_the_catalog_pair(catalog: SCCatalog) -> None:
    assert opposing_polarity(catalog.by_id(1).body, catalog.by_id(2).body)


def test_unrelated_constraints_do_not_oppose(catalog: SCCatalog) -> None:
    """ "Never send email" and "always use metric" are not a conflict."""
    assert not (action_classes(catalog.by_id(3).body) & action_classes(catalog.by_id(11).body))


# ---------------------------------------------------------------- tiers 1 and 2


async def test_exact_duplicate_is_suppressed() -> None:
    updater, store, audit = await make_updater()
    await add(updater, "Draft emails, never send them.", SCCategory.ACTION, 1)
    result = await add(updater, "  DRAFT   emails,  never send them. ", SCCategory.ACTION, 4)
    assert result.outcome is CandidateOutcome.DUPLICATE_EXACT
    assert len(await store.active(SESSION)) == 1
    events = audit.events(SESSION, AuditEventType.CONSTRAINT_DUPLICATE_SUPPRESSED)
    assert events[0].payload["tier"] == 1


class FakeEmbedder:
    """Controlled vectors so tier 2 is testable without a real encoder."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def encode(self, texts):  # type: ignore[no-untyped-def]
        rows = []
        for text in texts:
            vector = np.array(self._mapping[text], dtype=np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)


async def test_semantic_duplicate_is_suppressed_above_tau() -> None:
    """Tier 2 catches paraphrase drift that tier 1 cannot see."""
    original = "Draft emails instead of sending them."
    paraphrase = "Write emails as drafts rather than sending."
    embedder = FakeEmbedder({original: [1.0, 0.0], paraphrase: [0.99, 0.14]})
    updater, store, audit = await make_updater(tau_dup=0.9, embedder=embedder)

    await add(updater, original, SCCategory.ACTION, 1)
    result = await add(updater, paraphrase, SCCategory.ACTION, 5)
    assert result.outcome is CandidateOutcome.DUPLICATE_SEMANTIC
    assert result.similarity is not None and result.similarity >= 0.9
    assert len(await store.active(SESSION)) == 1
    assert (
        audit.events(SESSION, AuditEventType.CONSTRAINT_DUPLICATE_SUPPRESSED)[0].payload["tier"]
        == 2
    )


async def test_below_tau_is_not_a_duplicate() -> None:
    first = "Draft emails instead of sending them."
    second = "Use metric units for measurements."
    embedder = FakeEmbedder({first: [1.0, 0.0], second: [0.0, 1.0]})
    updater, store, _ = await make_updater(tau_dup=0.9, embedder=embedder)
    await add(updater, first, SCCategory.ACTION, 1)
    result = await add(updater, second, SCCategory.PREFERENCE, 2)
    assert result.outcome is CandidateOutcome.ADDED
    assert len(await store.active(SESSION)) == 2


async def test_tier_two_is_skipped_when_tau_is_unset() -> None:
    """UNKNOWN tau_dup: never hardcoded, so with no value tier 2 simply does not run."""
    updater, _store, _ = await make_updater(tau_dup=None, embedder=None)
    await add(updater, "Draft emails instead of sending.", SCCategory.ACTION, 1)
    result = await add(updater, "Write emails as drafts, do not send.", SCCategory.ACTION, 2)
    assert result.outcome is CandidateOutcome.ADDED


# ---------------------------------------------------------------- tiers 3 and 4


async def test_sc1_sc2_detected_as_conflict(catalog: SCCatalog) -> None:
    """TASK-023 acceptance: the catalog's contradictory pair is a CONFLICT, not a duplicate."""
    updater, store, audit = await make_updater()
    first = await add(updater, catalog.by_id(2).body, SCCategory.ACTION, 1)
    second = await add(updater, catalog.by_id(1).body, SCCategory.ACTION, 6)

    assert second.outcome is CandidateOutcome.SUPERSEDED_EXISTING
    assert first.constraint_id in second.superseded_ids

    active = await store.active(SESSION)
    assert len(active) == 1, "newest wins"
    assert active[0].constraint_id == second.constraint_id

    everything = await store.all_constraints(SESSION)
    tombstoned = [row for row in everything if row.status is SCStatus.SUPERSEDED]
    assert len(tombstoned) == 1
    assert tombstoned[0].superseded_by == second.constraint_id
    assert tombstoned[0].canonical_text == catalog.by_id(2).body, "tombstoned, not deleted"

    events = audit.events(SESSION, AuditEventType.CONSTRAINT_SUPERSEDED)
    assert len(events) == 1
    assert events[0].is_loud


async def test_refinement_is_independent() -> None:
    """TASK-023 acceptance: a narrowing refinement leaves both constraints active."""
    updater, store, _ = await make_updater()
    await add(updater, "Never send messages on my behalf.", SCCategory.ACTION, 1)
    result = await add(
        updater, "That restriction applies only to emails, not to files.", SCCategory.ACTION, 3
    )
    assert result.outcome is CandidateOutcome.ADDED
    assert result.superseded_ids == ()
    assert len(await store.active(SESSION)) == 2


async def test_paper_mode_disables_conflict_detection(catalog: SCCatalog) -> None:
    """registry.mode paper_flat_list reproduces the append only list exactly."""
    updater, store, audit = await make_updater(mode="paper_flat_list")
    assert updater.conflict_detection_enabled is False
    await add(updater, catalog.by_id(2).body, SCCategory.ACTION, 1)
    result = await add(updater, catalog.by_id(1).body, SCCategory.ACTION, 6)
    assert result.outcome is CandidateOutcome.ADDED
    assert len(await store.active(SESSION)) == 2, "the paper stacks both contradictory rules"
    assert audit.events(SESSION, AuditEventType.CONSTRAINT_SUPERSEDED) == ()


async def test_added_candidate_emits_an_audit_event() -> None:
    updater, _, audit = await make_updater()
    result = await add(updater, "Reply in bullet points only.", SCCategory.OUTPUT, 2)
    events = audit.events(SESSION, AuditEventType.CONSTRAINT_ADDED)
    assert len(events) == 1
    assert events[0].constraint_id == result.constraint_id
    assert events[0].turn_index == 2
    assert events[0].payload["category"] == "output"


# ---------------------------------------------------------------- adjudicator


async def test_heuristic_adjudicator_verdicts(catalog: SCCatalog) -> None:
    adjudicator = HeuristicAdjudicator()
    assert (
        await adjudicator.adjudicate("Never send email.", "Never send email.")
        is Adjudication.DUPLICATE
    )
    assert (
        await adjudicator.adjudicate(catalog.by_id(1).body, catalog.by_id(2).body)
        is Adjudication.CONFLICT
    )
    assert (
        await adjudicator.adjudicate("Use metric units.", "Cite primary sources.")
        is Adjudication.INDEPENDENT
    )


async def test_llm_adjudicator_defaults_to_independent_on_unparseable() -> None:
    """An unparseable verdict must not silently tombstone a constraint."""
    from scguard.registry.conflicts import LLMAdjudicator
    from shared.llm_client import StubLLMClient

    adjudicator = LLMAdjudicator(StubLLMClient(default_factory=lambda _r: "unclear"), "model")
    assert await adjudicator.adjudicate("a", "b") is Adjudication.INDEPENDENT


def test_find_conflict_candidates_reports_its_reason(catalog: SCCatalog) -> None:
    from scguard.registry.store import build_constraint

    existing = build_constraint(
        session_id=SESSION,
        tenant_id=TENANT,
        seq=0,
        canonical_text=catalog.by_id(2).body,
        category=SCCategory.ACTION,
        source_turn_index=0,
        token_count=20,
    )
    flagged = find_conflict_candidates(catalog.by_id(1).body, [existing])
    assert len(flagged) == 1
    assert "confirmation" in flagged[0].shared_classes
    assert "opposing polarity" in flagged[0].reason
