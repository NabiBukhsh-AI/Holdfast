"""TASK-014 and TASK-015 acceptance tests. Spec 6.8, 6.9, 6.10, 14.9, 14.10."""

from __future__ import annotations

import pytest

from compint.compactors.recent_n import RecentNCompactor
from compint.core.catalog import SCCatalog
from compint.core.framing import frame
from compint.core.injection import inject
from compint.core.models import CompactedContext, CompactionStatus, History
from compint.core.random_source import RandomSource
from compint.core.tokenization import Tokenizer
from compint.eval.compliance import (
    ComplianceHarness,
    ContextCache,
    parse_answer,
    render_mcq,
)
from compint.eval.records import ProbeStatus, RetentionStatus
from compint.eval.retention_judge import (
    RetentionJudge,
    parse_verdict_normalized,
    parse_verdict_strict,
    parser_leniency_delta,
)
from shared.llm_client import StubLLMClient
from shared.prompts import PromptRegistry


def make_compacted(text: str, status: CompactionStatus = CompactionStatus.OK) -> CompactedContext:
    return CompactedContext(
        text=text,
        compactor_id="recent_5",
        model_id="none",
        input_tokens=1000,
        output_tokens=max(1, len(text) // 4),
        status=status,
    )


# ---------------------------------------------------------------- judge parsing


@pytest.mark.parametrize("raw,expected", [("YES", "YES"), ("NO", "NO"), (" yes\n", "YES")])
def test_judge_strict_parsing_accepts_only_bare_verdicts(raw: str, expected: str) -> None:
    assert parse_verdict_strict(raw) == expected


@pytest.mark.parametrize("raw", ["YES.", "Yes, the constraint is present.", "MAYBE", "", "A"])
def test_judge_strict_parsing_rejects_everything_else(raw: str) -> None:
    """Coercing these to NO would inflate the headline finding."""
    assert parse_verdict_strict(raw) is None


def test_normalized_parser_recovers_punctuated_verdicts() -> None:
    assert parse_verdict_normalized("YES.") == "YES"
    assert parse_verdict_normalized("Yes, the constraint is present.") == "YES"
    assert parse_verdict_normalized("MAYBE") is None


async def test_judge_records_both_verdicts_and_never_coerces(
    prompts: PromptRegistry, catalog: SCCatalog
) -> None:
    client = StubLLMClient(default_factory=lambda r: "Yes, it is present.")
    judge = RetentionJudge(client, prompts.get("retention_judge"), "gpt-5.4")
    record = await judge.judge(
        frame(catalog.by_id(1)), make_compacted("summary"), instance_id="i1"
    )
    assert record.status is RetentionStatus.UNPARSEABLE
    assert record.verdict is None, "unparseable must never become 0 or 1"
    assert record.normalized_verdict == "YES"
    assert record.retained is None
    assert record.raw_response == "Yes, it is present."


async def test_judge_ok_verdict(prompts: PromptRegistry, catalog: SCCatalog) -> None:
    client = StubLLMClient(default_factory=lambda r: "NO")
    judge = RetentionJudge(client, prompts.get("retention_judge"), "gpt-5.4")
    record = await judge.judge(
        frame(catalog.by_id(2)), make_compacted("a summary with no constraint"), instance_id="i"
    )
    assert record.status is RetentionStatus.OK
    assert record.verdict == "NO"
    assert record.retained == 0


async def test_judge_blocked_status(prompts: PromptRegistry, catalog: SCCatalog) -> None:
    """Spec 6.8: a reproducible WildChat hazard, counted rather than treated as an error."""
    client = StubLLMClient(default_factory=lambda r: "__CONTENT_FILTER__")
    judge = RetentionJudge(client, prompts.get("retention_judge"), "gpt-5.4")
    record = await judge.judge(frame(catalog.by_id(5)), make_compacted("s"), instance_id="i")
    assert record.status is RetentionStatus.BLOCKED
    assert record.retained is None


async def test_judge_not_called_on_failed_compaction(
    prompts: PromptRegistry, catalog: SCCatalog
) -> None:
    """Spec 14.9: empty or wrapper only output is COMPACTION_FAILED, do not judge it."""
    client = StubLLMClient(default_factory=lambda r: "YES")
    judge = RetentionJudge(client, prompts.get("retention_judge"), "gpt-5.4")
    record = await judge.judge(
        frame(catalog.by_id(1)),
        make_compacted("", CompactionStatus.COMPACTION_FAILED),
        instance_id="i",
    )
    assert record.status is RetentionStatus.COMPACTION_FAILED
    assert client.call_count == 0


async def test_judge_cache_hit(prompts: PromptRegistry, catalog: SCCatalog) -> None:
    """Spec 14.9: cached on (prompt_hash, sc_hash, context_hash) because reruns are common."""
    client = StubLLMClient(default_factory=lambda r: "YES")
    judge = RetentionJudge(client, prompts.get("retention_judge"), "gpt-5.4")
    framed = frame(catalog.by_id(3))
    compacted = make_compacted("identical summary")
    await judge.judge(framed, compacted, instance_id="i1")
    await judge.judge(framed, compacted, instance_id="i2")
    assert client.call_count == 1
    assert judge.cache_hits == 1


async def test_judge_cache_is_keyed_on_context(
    prompts: PromptRegistry, catalog: SCCatalog
) -> None:
    client = StubLLMClient(default_factory=lambda r: "YES")
    judge = RetentionJudge(client, prompts.get("retention_judge"), "gpt-5.4")
    framed = frame(catalog.by_id(3))
    await judge.judge(framed, make_compacted("summary one"), instance_id="i1")
    await judge.judge(framed, make_compacted("summary two"), instance_id="i2")
    assert client.call_count == 2


def test_judge_cannot_receive_uncompacted_context(catalog: SCCatalog) -> None:
    """INV-4 is enforced at the type level: judge() accepts CompactedContext only.

    A History and an InjectedHistory are distinct types and neither is accepted, so judging
    the injected context (which would be trivially 1 by construction) cannot happen by
    accident.
    """
    import inspect

    signature = inspect.signature(RetentionJudge.judge)
    assert signature.parameters["compacted"].annotation == "CompactedContext"


def test_parser_leniency_delta_is_measurable(catalog: SCCatalog) -> None:
    from compint.eval.records import RetentionRecord

    records = [
        RetentionRecord(
            instance_id="a",
            sc_id=1,
            category=catalog.by_id(1).category,
            compactor_id="c",
            compacted_hash="h",
            status=RetentionStatus.UNPARSEABLE,
            normalized_verdict="YES",
        ),
        RetentionRecord(
            instance_id="b",
            sc_id=2,
            category=catalog.by_id(2).category,
            compactor_id="c",
            compacted_hash="h",
            verdict="NO",
            status=RetentionStatus.OK,
        ),
    ]
    delta = parser_leniency_delta(records)
    assert delta["unparseable_strict"] == 1
    assert delta["recovered_by_normalization_yes"] == 1


# ---------------------------------------------------------------- MCQ


def test_mcq_records_option_order_and_gold(prompts: PromptRegistry, catalog: SCCatalog) -> None:
    """Spec 6.9: without the mapping, A becomes a positional prior."""
    framed = frame(catalog.by_id(2))
    ab = render_mcq(prompts.get("mcq_probe"), framed, option_order="AB")
    ba = render_mcq(prompts.get("mcq_probe"), framed, option_order="BA")
    assert ab.gold == "A" and ba.gold == "B"
    assert framed.sc.option_compliant in ab.text
    assert ab.text != ba.text
    assert framed.sc.probe_query in ab.text


@pytest.mark.parametrize("raw,expected", [("A", "A"), (" b ", "B"), ("Answer: B", "B")])
def test_parse_answer(raw: str, expected: str) -> None:
    assert parse_answer(raw) == expected


@pytest.mark.parametrize("raw", ["C", "", "I cannot say"])
def test_parse_answer_rejects_non_letters(raw: str) -> None:
    assert parse_answer(raw) is None


# ---------------------------------------------------------------- conditions


def _harness(prompts: PromptRegistry, client: StubLLMClient, **kwargs: object) -> ComplianceHarness:
    return ComplianceHarness(
        client, prompts.get("mcq_probe"), "gpt-oss-120b", rng=RandomSource(7), **kwargs  # type: ignore[arg-type]
    )


async def test_klctx_computed_once_per_context(
    prompts: PromptRegistry,
    catalog: SCCatalog,
    hermes_history: History,
    tokenizer: Tokenizer,
) -> None:
    """TASK-015 acceptance: a 15 SC run issues exactly one K_lctx build and one C(H^t)."""
    client = StubLLMClient(default_factory=lambda r: "A")
    harness = _harness(prompts, client)
    compactor = RecentNCompactor(5, tokenizer)
    cache = ContextCache()

    for sc in catalog.constraints:
        framed = frame(sc)
        injected = inject(hermes_history, framed, {0})
        await harness.run_conditions(
            ("lctx", "ub"),
            history=hermes_history,
            injected=injected,
            framed_sc=framed,
            compactor=compactor,
            cache=cache,
            instance_id=f"i{sc.id}",
        )

    assert cache.lctx_builds == 1, "K_lctx is constant across the 15 SCs"
    assert cache.compaction_calls == 1, "C(H^t) for K_ub is constant across the 15 SCs"


async def test_comp_condition_compacts_per_sc(
    prompts: PromptRegistry,
    catalog: SCCatalog,
    hermes_history: History,
    tokenizer: Tokenizer,
) -> None:
    """K_comp genuinely differs per SC, so it cannot be cached the way K_ub can."""
    client = StubLLMClient(default_factory=lambda r: "A")
    harness = _harness(prompts, client)
    compactor = RecentNCompactor(5, tokenizer)
    cache = ContextCache()
    for sc in catalog.constraints[:3]:
        framed = frame(sc)
        await harness.run_conditions(
            ("comp",),
            history=hermes_history,
            injected=inject(hermes_history, framed, {0}),
            framed_sc=framed,
            compactor=compactor,
            cache=cache,
            instance_id=f"i{sc.id}",
        )
    assert cache.compaction_calls == 3


async def test_kub_uses_shared_assemble(
    prompts: PromptRegistry,
    catalog: SCCatalog,
    wildchat_history: History,
    tokenizer: Tokenizer,
) -> None:
    """INV-5: K_ub is built by the same assemble() production uses."""
    from shared.assembly import assemble

    client = StubLLMClient(default_factory=lambda r: "A")
    harness = _harness(prompts, client, assembly_mode="delimited")
    compactor = RecentNCompactor(5, tokenizer)
    cache = ContextCache()
    framed = frame(catalog.by_id(1))

    context, status = await harness.build_condition_context(
        "ub",
        history=wildchat_history,
        injected=inject(wildchat_history, framed, {0}),
        framed_sc=framed,
        compactor=compactor,
        cache=cache,
    )
    assert status is ProbeStatus.OK
    assert context is not None

    compacted = cache.uninjected_compaction
    assert compacted is not None

    class _Entry:
        canonical_text = framed.rendered_text
        is_active = True

    expected = assemble(compacted.text, [_Entry()], mode="delimited").text
    assert context == expected


async def test_lctx_carries_no_sc_text(
    prompts: PromptRegistry,
    catalog: SCCatalog,
    wildchat_history: History,
    tokenizer: Tokenizer,
) -> None:
    """K_lctx is the baseline correction term: it must contain no constraint at all."""
    client = StubLLMClient(default_factory=lambda r: "A")
    harness = _harness(prompts, client)
    framed = frame(catalog.by_id(1))
    context, _ = await harness.build_condition_context(
        "lctx",
        history=wildchat_history,
        injected=inject(wildchat_history, framed, {0}),
        framed_sc=framed,
        compactor=RecentNCompactor(5, tokenizer),
        cache=ContextCache(),
    )
    assert context is not None
    assert framed.rendered_text not in context


async def test_option_order_recorded(
    prompts: PromptRegistry, catalog: SCCatalog
) -> None:
    client = StubLLMClient(default_factory=lambda r: "A")
    harness = _harness(prompts, client, option_order="randomized")
    record = await harness.probe("comp", "ctx", frame(catalog.by_id(1)), instance_id="i")
    assert record.option_order in ("AB", "BA")
    assert record.gold == ("A" if record.option_order == "AB" else "B")


async def test_overflow_is_a_distinct_status_not_an_error(
    prompts: PromptRegistry, catalog: SCCatalog
) -> None:
    """Spec 14.10: at 220K the probe window is exceeded on the long context conditions."""
    client = StubLLMClient(default_factory=lambda r: "A")
    harness = _harness(prompts, client, probe_context_limit=10)
    record = await harness.probe(
        "lctx", "x" * 400, frame(catalog.by_id(1)), instance_id="i", context_tokens=100
    )
    assert record.status is ProbeStatus.OVERFLOW
    assert record.answer is None
    assert client.call_count == 0


async def test_refusal_is_recorded_not_swallowed(
    prompts: PromptRegistry, catalog: SCCatalog
) -> None:
    client = StubLLMClient(default_factory=lambda r: "__REFUSAL__")
    harness = _harness(prompts, client)
    record = await harness.probe("comp", "ctx", frame(catalog.by_id(6)), instance_id="i")
    assert record.status is ProbeStatus.REFUSED
    assert record.compliant is None


async def test_failed_compaction_yields_a_record_not_an_exception(
    prompts: PromptRegistry, catalog: SCCatalog, tokenizer: Tokenizer
) -> None:
    """A compaction that produced nothing must surface as a status, never as a silent skip."""
    from compint.core.models import Message, Role

    client = StubLLMClient(default_factory=lambda r: "A")
    harness = _harness(prompts, client)
    empty = History(
        messages=(Message(index=0, role=Role.USER, content="only turn", token_count=3),)
    )
    framed = frame(catalog.by_id(1))

    class _FailingCompactor:
        id = "failing"

        async def compact(self, history: History) -> CompactedContext:
            return make_compacted("", CompactionStatus.REFUSED)

    records = await harness.run_conditions(
        ("comp",),
        history=empty,
        injected=inject(empty, framed, {0}),
        framed_sc=framed,
        compactor=_FailingCompactor(),  # type: ignore[arg-type]
        cache=ContextCache(),
        instance_id="i",
    )
    assert records[0].status is ProbeStatus.REFUSED
