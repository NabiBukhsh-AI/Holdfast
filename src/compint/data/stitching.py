"""Topic cohesive stitching. TASK-008, Algorithm 1 (spec 14.1), FR-012 through FR-015.

Construct N synthetic long context conversations of target length l_t by iteratively appending
topically similar conversations, so the result is coherent rather than an arbitrary
concatenation of unrelated chats.

Four details in this algorithm are easy to get wrong and each changes every context:

1. **Seed selection is DETERMINISTIC**: the lowest indexed remaining sample under the original
   dataset ordering. Explicitly for reproducibility. Do NOT randomize (FR-014).
2. **The centroid accumulator holds the UNNORMALIZED SUM** of member embeddings. Normalization
   happens only when computing c(H~) at line 11. Accumulating normalized centroids instead
   gives different, wrong results.
3. **Line 18 removes the whole used set U from R**, not just the seed. Successive contexts
   therefore draw from a strictly shrinking, disjoint pool.
4. **Hermes Agent only**: keep the seed conversation's system prompt and STRIP the leading
   system prompt from every subsequently appended conversation, yielding exactly one system
   message at position zero. WildChat gets no such treatment (FR-015).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from compint.core.models import Conversation, History, Message, Role
from compint.core.tokenization import Tokenizer
from compint.data.contexts import ContextStatus, FillerContext
from compint.data.knn_index import ExactKNNIndex
from shared.errors import HoldFastError

logger = logging.getLogger(__name__)

# PAPER SPECIFICATION A.1: contexts whose system prompt handling differs from the default.
SYSTEM_PROMPT_DATASETS = frozenset({"hermes_agent"})


class InsufficientDataError(HoldFastError):
    """Fewer than N contexts are constructible from the pool."""


def _strip_leading_system(messages: Sequence[Message]) -> list[Message]:
    """Drop a leading system message. FR-015, applied to appended conversations only."""
    out = list(messages)
    while out and out[0].role is Role.SYSTEM:
        out.pop(0)
    return out


def _reindex(messages: Sequence[Message]) -> tuple[Message, ...]:
    """Renumber message indices from zero, preserving order."""
    return tuple(m.model_copy(update={"index": i}) for i, m in enumerate(messages))


def _crop_to_cap(
    messages: Sequence[Message],
    n_from_last_sample: int,
    cap_tokens: int,
    granularity: str,
    tokenizer: Tokenizer,
) -> tuple[list[Message], bool]:
    """Crop the final appended sample so the total falls under the soft cap.

    Spec 14.1: cropping must only remove overshoot, and must respect message boundaries where
    possible. UNKNOWN U-14 covers message versus token granularity; the choice is config driven.
    Messages belonging to earlier samples are never touched, because cropping into them would
    silently alter conversations that were already accepted.
    """
    kept = list(messages)
    croppable_start = len(kept) - n_from_last_sample
    changed = False

    while sum(m.token_count for m in kept) > cap_tokens and len(kept) > croppable_start + 1:
        kept.pop()
        changed = True

    if granularity == "token" and sum(m.token_count for m in kept) > cap_tokens:
        overshoot = sum(m.token_count for m in kept) - cap_tokens
        last = kept[-1]
        keep_tokens = max(1, last.token_count - overshoot)
        truncated_text = tokenizer.truncate(last.content, keep_tokens)
        if truncated_text:
            kept[-1] = last.model_copy(
                update={"content": truncated_text, "token_count": tokenizer.count(truncated_text)}
            )
            changed = True

    return kept, changed


def topic_cohesive_stitch(
    conversations: Sequence[Conversation],
    embeddings: np.ndarray,
    index: ExactKNNIndex,
    tokenizer: Tokenizer,
    *,
    dataset: str,
    split: str,
    target_tokens: int,
    n_contexts: int,
    knn_k: int = 32,
    soft_cap_multiplier: float = 1.25,
    crop_granularity: str = "message",
    context_set_version: str = "v1",
) -> tuple[FillerContext, ...]:
    """Algorithm 1. Deterministic given the corpus, the embeddings, and the ordering."""
    if len(conversations) != embeddings.shape[0]:
        raise ValueError(
            f"{len(conversations)} conversations against {embeddings.shape[0]} embeddings"
        )
    if n_contexts < 1:
        raise ValueError(f"n_contexts must be at least 1, got {n_contexts}")

    cap_tokens = int(target_tokens * soft_cap_multiplier)
    strip_system = dataset in SYSTEM_PROMPT_DATASETS
    remaining: set[int] = set(range(len(conversations)))
    built: list[FillerContext] = []

    while len(built) < n_contexts and remaining:
        # Line 6: select_seed(R) is the LOWEST INDEXED remaining sample. Never randomized.
        seed_index = min(remaining)
        seed = conversations[seed_index]

        messages: list[Message] = list(seed.messages)
        source_ids: list[str] = [seed.conversation_id]
        used: set[int] = {seed_index}
        # Line 9: the accumulator is the UNNORMALIZED SUM.
        centroid_sum = embeddings[seed_index].astype(np.float64).copy()
        status = ContextStatus.OK
        detail = ""
        n_from_last_sample = len(messages)

        while sum(m.token_count for m in messages) < target_tokens:
            available = remaining - used
            if not available:
                # Spec 14.1 edge case: pool exhaustion mid context. Emit SHORT, flagged.
                status = ContextStatus.SHORT_POOL_EXHAUSTED
                detail = (
                    f"pool exhausted after {len(used)} samples at "
                    f"{sum(m.token_count for m in messages)} tokens, target {target_tokens}"
                )
                logger.warning("stitch_pool_exhausted", extra={"dataset": dataset, "detail": detail})
                break

            # Line 11: normalize ONLY at query time.
            mean_vector = centroid_sum / len(used)
            norm = float(np.linalg.norm(mean_vector))
            if norm == 0.0:
                raise HoldFastError(
                    "centroid collapsed to the zero vector; embeddings are degenerate"
                )
            centroid = (mean_vector / norm).astype(np.float32)

            # FR-017: initial w = k + |H~|, doubling on exhaustion, raising at the cap.
            candidate = index.nearest_available(
                centroid, available, initial_w=knn_k + len(used)
            )

            appended = conversations[candidate]
            appended_messages = (
                _strip_leading_system(appended.messages) if strip_system else list(appended.messages)
            )
            messages.extend(appended_messages)
            n_from_last_sample = len(appended_messages)
            source_ids.append(appended.conversation_id)
            used.add(candidate)
            # Line 15: accumulate the raw embedding, not a normalized centroid.
            centroid_sum += embeddings[candidate].astype(np.float64)

        if sum(m.token_count for m in messages) > cap_tokens:
            messages, cropped = _crop_to_cap(
                messages, n_from_last_sample, cap_tokens, crop_granularity, tokenizer
            )
            if cropped and status is ContextStatus.OK:
                status = ContextStatus.CROPPED_TO_SOFT_CAP
                detail = f"cropped the final sample to fall under {cap_tokens} tokens"

        history = History(messages=_reindex(messages))
        if strip_system:
            system_count = sum(1 for m in history.messages if m.role is Role.SYSTEM)
            if system_count > 1:
                raise HoldFastError(
                    f"{dataset} context retained {system_count} system messages; FR-015 requires "
                    "exactly one, at position zero"
                )

        built.append(
            FillerContext(
                context_id=f"{dataset}_{split}_{target_tokens}_{len(built):04d}",
                dataset=dataset,
                split=split,
                target_tokens=target_tokens,
                actual_tokens=history.token_count,
                history=history,
                source_ids=tuple(source_ids),
                n_stitched=len(source_ids),
                status=status,
                context_set_version=context_set_version,
                detail=detail,
            )
        )
        # Line 18: remove the WHOLE used set from the pool, not just the seed.
        remaining -= used

    if len(built) < n_contexts:
        raise InsufficientDataError(
            f"{dataset}/{split}: built {len(built)} contexts from a pool of "
            f"{len(conversations)} conversations, needed {n_contexts}"
        )
    return tuple(built)
