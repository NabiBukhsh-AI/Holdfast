"""TASK-006 through TASK-010 acceptance tests. Spec 11.1, 11.2, 12.2, 14.1, 14.2."""

from __future__ import annotations

import numpy as np
import pytest

from compint.core.models import Conversation, Message, Role
from compint.core.tokenization import HeuristicTokenizer, Tokenizer, assert_reportable
from compint.data.context_builder import (
    ContextBuilder,
    assert_no_source_leakage,
    summarize,
)
from compint.data.contexts import ContextStatus
from compint.data.embedding import (
    StubEmbeddingModel,
    embed_conversations,
    l2_normalize,
    serialize_conversation,
)
from compint.data.ingest.base import QuarantineReason
from compint.data.ingest.registry import build_adapter
from compint.data.knn_index import ExactKNNIndex, NeighborPoolExhaustedError
from compint.data.splits import Split, assert_reportable_split, partition_pool
from compint.data.stitching import InsufficientDataError, topic_cohesive_stitch
from compint.data.truncation import (
    build_openresearcher_contexts,
    concatenate_for_220k,
    truncate_at_turn_boundary,
)
from shared.config import AppConfig
from shared.errors import ConfigError

# ---------------------------------------------------------------- ingestion


def test_wildchat_no_system_messages(tokenizer: Tokenizer) -> None:
    """PAPER SPECIFICATION spec 11.1: WildChat carries no system prompt and no tools."""
    adapter = build_adapter("wildchat", tokenizer)
    result = adapter.to_conversations(
        [
            {
                "conversation": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            },
            {
                "conversation": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ]
            },
            {
                "conversation": [
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "content": "res"},
                ]
            },
        ]
    )
    assert len(result.conversations) == 1
    assert result.reasons() == {"UNEXPECTED_SYSTEM_PROMPT": 1, "SCHEMA_MISMATCH": 1}


def test_hermes_has_system_at_zero(tokenizer: Tokenizer) -> None:
    adapter = build_adapter("hermes_agent", tokenizer)
    result = adapter.to_conversations(
        [
            {
                "conversation": [
                    {"role": "system", "content": "you are an agent"},
                    {"role": "user", "content": "do the thing"},
                    {"role": "tool", "content": "result", "tool_name": "search"},
                ]
            },
            {
                "conversation": [
                    {"role": "user", "content": "do the thing"},
                    {"role": "system", "content": "misplaced"},
                ]
            },
        ]
    )
    assert len(result.conversations) == 1
    assert result.conversations[0].messages[0].role is Role.SYSTEM
    assert result.reasons() == {"SYSTEM_PROMPT_NOT_FIRST": 1}


def test_openresearcher_single_user_turn(tokenizer: Tokenizer) -> None:
    """Table 1 reports exactly 1.00 user turns; a second one is a schema violation."""
    adapter = build_adapter("openresearcher", tokenizer)
    result = adapter.to_conversations(
        [
            {
                "conversation": [
                    {"role": "user", "content": "research x"},
                    {"role": "tool", "content": "r"},
                ]
            },
            {"conversation": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]},
        ]
    )
    assert len(result.conversations) == 1
    assert result.conversations[0].n_user_turns == 1
    assert result.reasons() == {"MULTIPLE_USER_TURNS": 1}


def test_quarantine_records_reason(tokenizer: Tokenizer) -> None:
    """Malformed rows are counted with a reason, never dropped silently."""
    adapter = build_adapter("wildchat", tokenizer)
    result = adapter.to_conversations(
        [
            {"conversation": [{"role": "user", "content": "ok"}]},
            {"no_turns_here": True},
            {"conversation": [{"role": "wizard", "content": "x"}]},
            {"conversation": [{"role": "user", "content": "   "}]},
            {"conversation": []},
        ]
    )
    assert result.n_seen == 5
    assert set(result.reasons()) == {
        QuarantineReason.SCHEMA_MISMATCH.value,
        QuarantineReason.UNKNOWN_ROLE.value,
        QuarantineReason.MISSING_CONTENT.value,
        QuarantineReason.EMPTY_CONVERSATION.value,
    }
    for record in result.quarantined:
        assert record.detail, "every quarantined row must carry a reason detail"


def test_duplicate_conversations_are_quarantined(tokenizer: Tokenizer) -> None:
    adapter = build_adapter("wildchat", tokenizer)
    row = {
        "conversation": [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "text"},
        ]
    }
    result = adapter.to_conversations([row, dict(row)])
    assert len(result.conversations) == 1
    assert result.reasons() == {"DUPLICATE_CONTENT": 1}


def test_unknown_role_is_not_guessed(tokenizer: Tokenizer) -> None:
    """Mapping an unrecognized role onto ASSISTANT would silently change |U^t|."""
    adapter = build_adapter("wildchat", tokenizer)
    result = adapter.to_conversations([{"conversation": [{"role": "narrator", "content": "x"}]}])
    assert not result.conversations
    assert result.quarantined[0].reason is QuarantineReason.UNKNOWN_ROLE


def test_ingestion_records_the_tokenizer(tokenizer: Tokenizer) -> None:
    """Spec 11.1: token counts computed with the wrong tokenizer shift every context length."""
    adapter = build_adapter("wildchat", tokenizer)
    result = adapter.to_conversations([{"conversation": [{"role": "user", "content": "hello"}]}])
    assert result.tokenizer_id == tokenizer.id


# ---------------------------------------------------------------- embedding


def test_embeddings_are_l2_normalized(wildchat_corpus: list[Conversation]) -> None:
    model = StubEmbeddingModel()
    vectors = embed_conversations(wildchat_corpus[:20], model, "role_prefixed_newline")
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedding_is_deterministic_on_content(wildchat_corpus: list[Conversation]) -> None:
    model = StubEmbeddingModel()
    first = embed_conversations(wildchat_corpus[:10], model, "role_prefixed_newline")
    second = embed_conversations(wildchat_corpus[:10], model, "role_prefixed_newline")
    assert np.array_equal(first, second)


def test_l2_normalize_refuses_zero_vectors() -> None:
    """A zero embedding means the encoder failed; fabricating a direction would corrupt ranking."""
    with pytest.raises(ValueError, match="zero vectors"):
        l2_normalize(np.zeros((2, 4), dtype=np.float32))


def test_serialization_strategies_differ(wildchat_corpus: list[Conversation]) -> None:
    """UNKNOWN U-06: the strategy changes the embedded text, so it is config driven."""
    conversation = wildchat_corpus[0]
    role_prefixed = serialize_conversation(conversation, "role_prefixed_newline")
    content_only = serialize_conversation(conversation, "content_only")
    assert role_prefixed != content_only
    assert role_prefixed.startswith("user: ")
    with pytest.raises(ConfigError, match="U-06"):
        serialize_conversation(conversation, "invented_strategy")


# ---------------------------------------------------------------- kNN index


def test_index_returns_ranked_neighbors() -> None:
    vectors = l2_normalize(np.eye(5, dtype=np.float32) + 0.01)
    index = ExactKNNIndex(vectors)
    ranked = index.search(vectors[2], 5)
    assert ranked[0] == 2, "a vector must be its own nearest neighbor"
    assert len(ranked) == 5


def test_neighbor_lookup_doubles_window() -> None:
    """FR-017: when the whole top-w window is used, w doubles and the query repeats."""
    rng = np.random.default_rng(7)
    vectors = l2_normalize(rng.standard_normal((64, 8)).astype(np.float32))
    index = ExactKNNIndex(vectors)
    query = vectors[0]
    ranked = index.search(query, 64)
    # Everything except the very last ranked candidate is unavailable, so a small initial
    # window cannot succeed and the doubling path must run.
    available = {ranked[-1]}
    assert index.nearest_available(query, available, initial_w=2) == ranked[-1]


def test_neighbor_lookup_raises_at_cap() -> None:
    """A silent fallback here would quietly change which conversations were stitched."""
    rng = np.random.default_rng(11)
    vectors = l2_normalize(rng.standard_normal((16, 4)).astype(np.float32))
    index = ExactKNNIndex(vectors)
    with pytest.raises(NeighborPoolExhaustedError):
        index.nearest_available(vectors[0], set(), initial_w=4)


def test_index_ties_break_on_lower_index() -> None:
    """Deterministic tie breaking is what lets the numpy and FAISS backends agree."""
    vectors = l2_normalize(np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1)))
    index = ExactKNNIndex(vectors, backend="numpy")
    assert index.search(vectors[0], 4) == [0, 1, 2, 3]


# ---------------------------------------------------------------- splits


def test_no_source_leakage_between_splits(wildchat_corpus: list[Conversation]) -> None:
    """TASK-009: the pool is partitioned BEFORE stitching, so dev cannot leak into eval."""
    pool = partition_pool(wildchat_corpus, "wildchat")
    pool.assert_disjoint()
    assert pool.dev and pool.eval
    assert len(pool.dev) + len(pool.eval) == len(wildchat_corpus)


def test_partition_is_content_addressed_not_positional(
    wildchat_corpus: list[Conversation],
) -> None:
    """Adding rows must not reshuffle the partition and move a conversation from eval to dev."""
    first = partition_pool(wildchat_corpus[:100], "wildchat")
    second = partition_pool(wildchat_corpus, "wildchat")
    first_dev = {c.conversation_id for c in first.dev}
    second_dev = {c.conversation_id for c in second.dev}
    original = {c.conversation_id for c in wildchat_corpus[:100]}
    assert first_dev == (second_dev & original)


def test_report_refuses_dev_as_headline() -> None:
    assert_reportable_split(Split.EVAL)
    with pytest.raises(ConfigError, match="never be reported as headline"):
        assert_reportable_split(Split.DEV)


# ---------------------------------------------------------------- stitching


def _stitch(
    corpus: list[Conversation],
    dataset: str,
    tokenizer: Tokenizer,
    *,
    target_tokens: int = 2000,
    n_contexts: int = 3,
):
    model = StubEmbeddingModel()
    embeddings = embed_conversations(corpus, model, "role_prefixed_newline")
    index = ExactKNNIndex(embeddings)
    return topic_cohesive_stitch(
        corpus,
        embeddings,
        index,
        tokenizer,
        dataset=dataset,
        split="eval",
        target_tokens=target_tokens,
        n_contexts=n_contexts,
    )


def test_stitching_deterministic(wildchat_corpus: list[Conversation], tokenizer: Tokenizer) -> None:
    """TASK-008 acceptance: two runs produce byte identical context sets."""
    first = _stitch(wildchat_corpus, "wildchat", tokenizer)
    second = _stitch(wildchat_corpus, "wildchat", tokenizer)
    assert [c.history.content_hash() for c in first] == [c.history.content_hash() for c in second]
    assert [c.source_ids for c in first] == [c.source_ids for c in second]


def test_seed_selection_is_lowest_index(
    wildchat_corpus: list[Conversation], tokenizer: Tokenizer
) -> None:
    """FR-014: the seed is the lowest indexed remaining sample. Never randomized."""
    contexts = _stitch(wildchat_corpus, "wildchat", tokenizer)
    assert contexts[0].source_ids[0] == wildchat_corpus[0].conversation_id


def test_length_bounds(wildchat_corpus: list[Conversation], tokenizer: Tokenizer) -> None:
    """FR-013: output length lands in [l_t, 1.25 * l_t] unless the pool ran out."""
    target = 2000
    for context in _stitch(wildchat_corpus, "wildchat", tokenizer, target_tokens=target):
        if context.status is ContextStatus.SHORT_POOL_EXHAUSTED:
            continue
        assert context.actual_tokens >= target
        assert context.actual_tokens <= int(target * 1.25)


def test_pool_shrinks_by_used_set(
    wildchat_corpus: list[Conversation], tokenizer: Tokenizer
) -> None:
    """Line 18 removes the WHOLE used set, so contexts draw from disjoint pools."""
    contexts = _stitch(wildchat_corpus, "wildchat", tokenizer, n_contexts=4)
    seen: set[str] = set()
    for context in contexts:
        overlap = seen & set(context.source_ids)
        assert not overlap, f"source reused across contexts: {overlap}"
        seen.update(context.source_ids)


def test_no_source_used_twice_within_one_context(
    wildchat_corpus: list[Conversation], tokenizer: Tokenizer
) -> None:
    for context in _stitch(wildchat_corpus, "wildchat", tokenizer):
        assert len(set(context.source_ids)) == len(context.source_ids)


def test_centroid_is_unnormalized_sum(tokenizer: Tokenizer) -> None:
    """Accumulating normalized centroids instead gives different, wrong neighbor choices.

    Two vectors are near duplicates and a third is far away. The unnormalized sum keeps the
    running centroid pinned near the duplicate pair, which drives the third append. A
    normalized-per-step accumulator drifts and would pick differently.
    """
    vectors = l2_normalize(
        np.array(
            [[1.0, 0.02, 0.0], [1.0, 0.0, 0.02], [0.0, 1.0, 0.0], [0.9, 0.1, 0.05]],
            dtype=np.float32,
        )
    )
    corpus = [
        Conversation(
            conversation_id=f"c{i}",
            dataset="wildchat",
            messages=(Message(index=0, role=Role.USER, content=f"text {i}", token_count=300),),
            token_count=300,
            source_index=i,
        )
        for i in range(4)
    ]
    index = ExactKNNIndex(vectors)
    contexts = topic_cohesive_stitch(
        corpus,
        vectors,
        index,
        tokenizer,
        dataset="wildchat",
        split="eval",
        target_tokens=900,
        n_contexts=1,
    )
    # Seed c0, nearest c1 (near duplicate), then the centroid of {c0, c1} still favours c3
    # over the orthogonal c2.
    assert contexts[0].source_ids == ("c0", "c1", "c3")


def test_hermes_single_system_prompt(
    hermes_corpus: list[Conversation], tokenizer: Tokenizer
) -> None:
    """FR-015: keep the seed's system prompt, strip every subsequently appended one."""
    contexts = _stitch(hermes_corpus, "hermes_agent", tokenizer, target_tokens=3000, n_contexts=2)
    for context in contexts:
        system_messages = [m for m in context.history.messages if m.role is Role.SYSTEM]
        assert len(system_messages) == 1
        assert system_messages[0].index == 0
        assert context.n_stitched > 1, "the test is vacuous unless stitching actually appended"


def test_wildchat_keeps_zero_system_messages(
    wildchat_corpus: list[Conversation], tokenizer: Tokenizer
) -> None:
    """FR-015: WildChat does NOT receive the system prompt treatment."""
    for context in _stitch(wildchat_corpus, "wildchat", tokenizer):
        assert not [m for m in context.history.messages if m.role is Role.SYSTEM]


def test_stitching_raises_when_pool_cannot_fill_n_contexts(
    tokenizer: Tokenizer,
    make_conversation,
) -> None:
    tiny = [make_conversation("wildchat", i, n_user_turns=1, words=5) for i in range(2)]
    with pytest.raises(InsufficientDataError):
        _stitch(tiny, "wildchat", tokenizer, target_tokens=50_000, n_contexts=5)


# ---------------------------------------------------------------- truncation


def test_truncate_keeps_smallest_turn_aligned_prefix_at_or_above_target(
    openresearcher_corpus: list[Conversation],
) -> None:
    """Spec 14.2: the SMALLEST turn aligned prefix that is at least l_t, never exactly l_t."""
    messages = openresearcher_corpus[0].messages
    target = 200
    kept = truncate_at_turn_boundary(messages, target)
    assert sum(m.token_count for m in kept) >= target
    assert sum(m.token_count for m in kept[:-1]) < target
    assert kept == messages[: len(kept)], "truncation keeps a prefix, in order"


def test_openresearcher_contexts_are_degenerate(
    openresearcher_corpus: list[Conversation],
) -> None:
    """FR-023: exactly one user turn, so all four injection conditions collapse."""
    contexts = build_openresearcher_contexts(
        openresearcher_corpus, split="eval", target_tokens=300, n_contexts=5
    )
    assert len(contexts) == 5
    for context in contexts:
        assert context.n_user_turns == 1
        assert context.is_degenerate_for_injection
        assert context.n_stitched == 1
        assert context.status is ContextStatus.TRUNCATED_AT_TURN_BOUNDARY


def test_openresearcher_raises_when_too_few_long_trajectories(
    openresearcher_corpus: list[Conversation],
) -> None:
    with pytest.raises(InsufficientDataError):
        build_openresearcher_contexts(
            openresearcher_corpus, split="eval", target_tokens=10**9, n_contexts=5
        )


def test_220k_pairing_yields_two_user_turns(
    openresearcher_corpus: list[Conversation],
) -> None:
    """FR-019: two 110K entries concatenated, which partially un-collapses the conditions."""
    contexts = build_openresearcher_contexts(
        openresearcher_corpus, split="eval", target_tokens=300, n_contexts=4
    )
    paired = concatenate_for_220k(contexts, target_tokens=600)
    assert len(paired) == 2
    for context in paired:
        assert context.n_user_turns == 2
        assert not context.is_degenerate_for_injection
        assert len(context.source_ids) == 2


# ---------------------------------------------------------------- builder


def test_builder_produces_disjoint_dev_and_eval_contexts(
    wildchat_corpus: list[Conversation], base_config: AppConfig, tokenizer: Tokenizer
) -> None:
    config = base_config.model_copy(
        update={
            "context": base_config.context.model_copy(
                update={"target_tokens": 1500, "dev_contexts": 2, "eval_contexts": 3}
            )
        }
    )
    builder = ContextBuilder(config, tokenizer, StubEmbeddingModel())
    built = builder.build_both_splits(wildchat_corpus, dataset="wildchat")
    assert len(built["dev"]) == 2
    assert len(built["eval"]) == 3
    assert_no_source_leakage(built["dev"], built["eval"])


def test_builder_refuses_stub_models_for_a_reported_run(
    base_config: AppConfig, tokenizer: Tokenizer
) -> None:
    """A reported run may not define context lengths with the approximation tokenizer."""
    with pytest.raises(ConfigError):
        ContextBuilder(base_config, tokenizer, StubEmbeddingModel(), require_reportable=True)


def test_assert_reportable_rejects_the_heuristic_tokenizer() -> None:
    with pytest.raises(ConfigError, match="approximation"):
        assert_reportable(HeuristicTokenizer())


def test_summary_reports_table1_columns(
    openresearcher_corpus: list[Conversation],
) -> None:
    """Table 1 reports # Turns and # User Turns as separate columns for a reason."""
    contexts = build_openresearcher_contexts(
        openresearcher_corpus, split="eval", target_tokens=300, n_contexts=5
    )
    stats = summarize(contexts)
    assert stats.mean_user_turns == 1.0
    assert stats.n_degenerate == 5
    comparison = stats.compare_to_table1()
    assert comparison["n_user_turns"]["within_tolerance"] is True


def test_table1_mismatch_raises_rather_than_warning(
    openresearcher_corpus: list[Conversation],
) -> None:
    """Escalation trigger 2: wrong contexts poison every downstream number."""
    contexts = build_openresearcher_contexts(
        openresearcher_corpus, split="eval", target_tokens=300, n_contexts=5
    )
    stats = summarize(contexts).model_copy(update={"mean_user_turns": 9.0})
    with pytest.raises(ConfigError, match="does not reproduce Table 1"):
        stats.assert_within_tolerance()
