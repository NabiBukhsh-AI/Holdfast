"""TASK-025 and TASK-026 acceptance tests. Algorithm 14.8, spec 16.3, NFR-008, INV-7."""

from __future__ import annotations

import json

import pytest

from compint.extractor.client import SCExtractor
from scguard.assembly.service import AssemblyService, assert_single_registry_block
from scguard.audit.emitter import AuditEmitter, AuditEventType
from scguard.extractor.queue import ExtractionQueue, JobStatus, QueueFullError
from scguard.extractor.worker import ExtractionWorker
from scguard.registry.conflicts import HeuristicAdjudicator
from scguard.registry.dedup import RegistryUpdater
from scguard.registry.store import (
    InMemoryRegistryStore,
    RegistryUnavailableError,
    SCCategory,
    Session,
    build_constraint,
)
from shared.delimiters import REGISTRY_OPEN, count_registry_blocks
from shared.llm_client import StubLLMClient
from shared.prompts import Prompt

SESSION = "sess_pipeline"
TENANT = "tenant_pipeline"

USER_TURN = "Also, from now on don't send any emails on my behalf, just draft them."


def extraction_prompt() -> Prompt:
    return Prompt(
        id="sc_extractor",
        version="v1",
        provenance="fetched",
        source_url="https://example.invalid/repo",
        fetched_at="2026-08-19T00:00:00Z",
        user="EXTRACTION INSTRUCTIONS\n{inputs}",
    )


def extraction_payload(*items: dict[str, str]) -> str:
    return json.dumps(list(items))


async def build_pipeline(
    *,
    extractor_response: str = "[]",
    shadow_mode: bool = False,
    budget_tokens: int = 200,
    capacity: int = 100,
):
    store = InMemoryRegistryStore()
    await store.create_session(
        Session(
            session_id=SESSION, tenant_id=TENANT,
            extractor_model="qwen3.5-9b", prompt_hash="sha256:abc",
        )
    )
    audit = AuditEmitter()
    queue = ExtractionQueue(capacity=capacity, lease_seconds=0.05)
    client = StubLLMClient(default_factory=lambda r: extractor_response)
    extractor = SCExtractor(client, extraction_prompt(), "qwen3.5-9b", max_retries=0, retry_backoff_s=0.0)
    updater = RegistryUpdater(store, audit, mode="production", adjudicator=HeuristicAdjudicator())
    worker = ExtractionWorker(queue, extractor, updater, audit)
    service = AssemblyService(
        store, queue, audit,
        assembly_mode="delimited", budget_tokens=budget_tokens,
        drain_timeout_ms=50, shadow_mode=shadow_mode,
    )
    return store, queue, audit, worker, service


# ---------------------------------------------------------------- queue


async def test_idempotent_enqueue() -> None:
    """TASK-025 acceptance: duplicate submissions produce one extraction."""
    _, queue, _, _, _ = await build_pipeline()
    first = await queue.enqueue(SESSION, TENANT, 4, USER_TURN)
    second = await queue.enqueue(SESSION, TENANT, 4, USER_TURN)
    assert first.job_id == second.job_id
    assert queue.depth == 1


async def test_same_index_different_content_is_a_separate_job() -> None:
    _, queue, _, _, _ = await build_pipeline()
    first = await queue.enqueue(SESSION, TENANT, 4, "one thing")
    second = await queue.enqueue(SESSION, TENANT, 4, "a different thing")
    assert first.job_id != second.job_id


async def test_backpressure_raises_rather_than_dropping() -> None:
    """A dropped turn is a constraint nobody knows was missed, so the queue refuses instead."""
    _, queue, _, _, _ = await build_pipeline(capacity=2)
    await queue.enqueue(SESSION, TENANT, 0, "turn zero")
    await queue.enqueue(SESSION, TENANT, 1, "turn one")
    with pytest.raises(QueueFullError) as excinfo:
        await queue.enqueue(SESSION, TENANT, 2, "turn two")
    assert excinfo.value.capacity == 2
    assert "429" in str(excinfo.value) or "dropping" in str(excinfo.value)


async def test_crashed_job_reclaimed() -> None:
    """A worker that dies mid job must not leave that turn permanently unextracted."""
    import asyncio

    _, queue, _, _, _ = await build_pipeline()
    await queue.enqueue(SESSION, TENANT, 0, USER_TURN)
    claimed = queue.claim()
    assert claimed is not None and claimed.status is JobStatus.RUNNING
    assert queue.claim() is None, "a leased job is not handed to a second worker"

    await asyncio.sleep(0.06)  # lease_seconds is 0.05 in the fixture
    reclaimed = queue.reclaim_expired()
    assert len(reclaimed) == 1
    again = queue.claim()
    assert again is not None
    assert again.attempts == 2


async def test_retry_then_terminal_failure() -> None:
    _, queue, _, _, _ = await build_pipeline()
    job = await queue.enqueue(SESSION, TENANT, 0, USER_TURN)
    for _ in range(3):
        claimed = queue.claim()
        assert claimed is not None
        queue.retry_or_fail(claimed.job_id, "extractor unreachable")
    final = queue.get(job.job_id)
    assert final is not None and final.status is JobStatus.FAILED
    assert final.error_detail == "extractor unreachable"


# ---------------------------------------------------------------- worker


async def test_worker_adds_extracted_constraints() -> None:
    store, queue, _, worker, _ = await build_pipeline(
        extractor_response=extraction_payload(
            {
                "canonical_text": "Draft emails instead of sending them.",
                "evidence_span": "don't send any emails on my behalf, just draft them",
                "category": "action",
            }
        )
    )
    await queue.enqueue(SESSION, TENANT, 3, USER_TURN)
    await worker.drain_all()
    active = await store.active(SESSION)
    assert len(active) == 1
    assert active[0].source_turn_index == 3
    assert active[0].category is SCCategory.ACTION


async def test_extractor_outage_is_loud_and_not_an_empty_registry() -> None:
    """NFR-008: an outage must never look like a turn that declared nothing."""
    store, queue, audit, worker, _ = await build_pipeline(extractor_response="__TIMEOUT__")
    job = await queue.enqueue(SESSION, TENANT, 1, USER_TURN)
    await worker.drain_all()
    finished = queue.get(job.job_id)
    assert finished is not None and finished.status is JobStatus.FAILED
    events = audit.events(SESSION, AuditEventType.EXTRACTION_FAILED)
    assert len(events) == 1 and events[0].is_loud
    assert await store.active(SESSION) == ()


async def test_hallucinated_evidence_is_audited() -> None:
    store, queue, audit, worker, _ = await build_pipeline(
        extractor_response=extraction_payload(
            {
                "canonical_text": "Invented constraint.",
                "evidence_span": "text the user never wrote",
                "category": "action",
            }
        )
    )
    await queue.enqueue(SESSION, TENANT, 1, USER_TURN)
    await worker.drain_all()
    assert await store.active(SESSION) == ()
    assert len(audit.events(SESSION, AuditEventType.HALLUCINATED_EVIDENCE_REJECTED)) == 1


# ---------------------------------------------------------------- assembly


async def test_assembly_attaches_the_registry() -> None:
    store, _, _, _, service = await build_pipeline()
    await store.append(
        build_constraint(
            session_id=SESSION, tenant_id=TENANT, seq=0,
            canonical_text="Draft emails instead of sending them.",
            category=SCCategory.ACTION, source_turn_index=1, token_count=12,
        )
    )
    result = await service.compact(
        SESSION, TENANT, compaction_index=0, compacted_summary="<summary>work so far</summary>"
    )
    assert REGISTRY_OPEN in result.augmented_context
    assert "Draft emails instead of sending them." in result.augmented_context
    assert result.registry.injected_count == 1
    assert result.registry_incomplete is False


async def test_empty_registry_returns_the_bare_summary() -> None:
    _, _, _, _, service = await build_pipeline()
    result = await service.compact(
        SESSION, TENANT, compaction_index=0, compacted_summary="just the summary"
    )
    assert result.augmented_context == "just the summary"
    assert REGISTRY_OPEN not in result.augmented_context
    assert result.registry.injected_count == 0


async def test_drain_timeout_sets_incomplete() -> None:
    """Bounded wait, then surface. Never block the user, never proceed silently."""
    _, queue, _, _, service = await build_pipeline()
    await queue.enqueue(SESSION, TENANT, 5, USER_TURN)  # never processed
    result = await service.compact(
        SESSION, TENANT, compaction_index=0, compacted_summary="summary", drain_timeout_ms=20
    )
    assert result.registry_incomplete is True
    codes = {warning.code for warning in result.warnings}
    assert "REGISTRY_INCOMPLETE" in codes
    detail = next(w.detail for w in result.warnings if w.code == "REGISTRY_INCOMPLETE")
    assert "NOT in this context" in detail


async def test_drain_completes_when_nothing_is_pending() -> None:
    store, queue, _, worker, service = await build_pipeline()
    await queue.enqueue(SESSION, TENANT, 0, USER_TURN)
    await worker.drain_all()
    result = await service.compact(
        SESSION, TENANT, compaction_index=0, compacted_summary="summary"
    )
    assert result.registry_incomplete is False


async def test_double_compaction_single_block() -> None:
    """TASK-026 acceptance and INV-7: exactly one registry block survives."""
    store, _, _, _, service = await build_pipeline()
    await store.append(
        build_constraint(
            session_id=SESSION, tenant_id=TENANT, seq=0,
            canonical_text="Confirm before acting.", category=SCCategory.ACTION,
            source_turn_index=1, token_count=8,
        )
    )
    first = await service.compact(
        SESSION, TENANT, compaction_index=0, compacted_summary="<summary>first</summary>"
    )
    assert count_registry_blocks(first.augmented_context) == 1

    # The harness hands the previously augmented context back as the next compaction input.
    second = await service.compact(
        SESSION, TENANT, compaction_index=1,
        compacted_summary=first.augmented_context + "\n\nmore work happened",
    )
    assert count_registry_blocks(second.augmented_context) == 1
    assert_single_registry_block(second.augmented_context)
    assert "Confirm before acting." in second.augmented_context


async def test_store_failure_returns_503_not_empty_registry() -> None:
    """Spec 18.2: degrading to an empty registry is the failure this system prevents."""
    store, _, _, _, service = await build_pipeline()
    await store.append(
        build_constraint(
            session_id=SESSION, tenant_id=TENANT, seq=0,
            canonical_text="Never send email.", category=SCCategory.ACTION,
            source_turn_index=0, token_count=6,
        )
    )
    store.set_available(False)
    with pytest.raises(RegistryUnavailableError):
        await service.compact(
            SESSION, TENANT, compaction_index=0, compacted_summary="summary"
        )


async def test_eviction_surfaces_as_a_warning() -> None:
    store, _, audit, _, service = await build_pipeline(budget_tokens=20)
    for seq, (text, category, tokens) in enumerate(
        [
            ("Confirm before any action.", SCCategory.ACTION, 15),
            ("Reply in bullet points only.", SCCategory.OUTPUT, 15),
        ]
    ):
        await store.append(
            build_constraint(
                session_id=SESSION, tenant_id=TENANT, seq=seq,
                canonical_text=text, category=category,
                source_turn_index=seq, token_count=tokens,
            )
        )
    result = await service.compact(
        SESSION, TENANT, compaction_index=0, compacted_summary="summary"
    )
    assert result.registry.evicted_count == 1
    codes = {warning.code for warning in result.warnings}
    assert "REGISTRY_EVICTED" in codes
    assert audit.events(SESSION, AuditEventType.REGISTRY_EVICTED)
    assert "Confirm before any action." in result.augmented_context, "Action outranks Output"


async def test_shadow_mode_records_but_does_not_inject() -> None:
    """FR-086: safe rollout measurement. Everything is recorded, nothing is attached."""
    store, _, audit, _, service = await build_pipeline(shadow_mode=True)
    await store.append(
        build_constraint(
            session_id=SESSION, tenant_id=TENANT, seq=0,
            canonical_text="Never send email.", category=SCCategory.ACTION,
            source_turn_index=0, token_count=6,
        )
    )
    result = await service.compact(
        SESSION, TENANT, compaction_index=0, compacted_summary="summary only"
    )
    assert result.augmented_context == "summary only"
    assert result.registry.active_count == 1
    assert result.registry.injected_count == 0
    assert "SHADOW_MODE" in {warning.code for warning in result.warnings}
    assert audit.events(SESSION, AuditEventType.ASSEMBLY_PERFORMED)[0].payload["shadow_mode"] is True


async def test_assembly_emits_an_audit_record() -> None:
    _, _, audit, _, service = await build_pipeline()
    await service.compact(SESSION, TENANT, compaction_index=2, compacted_summary="summary")
    events = audit.events(SESSION, AuditEventType.ASSEMBLY_PERFORMED)
    assert len(events) == 1
    assert events[0].payload["compaction_index"] == 2
    assert events[0].payload["assembly_mode"] == "delimited"


async def test_end_to_end_turn_to_augmented_context() -> None:
    """The whole path: submit a turn, extract it, compact, and see the constraint attached."""
    store, queue, _, worker, service = await build_pipeline(
        extractor_response=extraction_payload(
            {
                "canonical_text": "Draft emails instead of sending them.",
                "evidence_span": "don't send any emails on my behalf, just draft them",
                "category": "action",
            }
        )
    )
    await queue.enqueue(SESSION, TENANT, 7, USER_TURN)
    await worker.drain_all()
    result = await service.compact(
        SESSION, TENANT, compaction_index=0,
        compacted_summary="<summary>The user reorganized their calendar.</summary>",
    )
    assert result.registry_incomplete is False
    assert "Draft emails instead of sending them." in result.augmented_context
    assert result.augmented_context.index("calendar") < result.augmented_context.index("Draft emails")
