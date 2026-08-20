"""OpenResearcher context construction. TASK-010, Algorithm 14.2, FR-016 and FR-019.

PAPER SPECIFICATION spec 4.2: build filler contexts from natively long trajectories WITHOUT
stitching. Select trajectories longer than l_t, then truncate each at a TURN BOUNDARY while
retaining at least l_t tokens.

`EDGE CASE` spec 14.2: "truncate at a turn boundary while retaining at least l_t" means keeping
the SMALLEST prefix that is turn aligned AND at least l_t tokens. Truncating to exactly l_t
mid turn would violate the stated procedure and would also risk cutting a user turn in half,
which would change |U^t|.

The result has exactly ONE user turn, so all four injection conditions collapse and the cells
are marked DEGENERATE (FR-023).
"""

from __future__ import annotations

from collections.abc import Sequence

from compint.core.models import Conversation, History, Message
from compint.data.contexts import ContextStatus, FillerContext
from compint.data.stitching import InsufficientDataError


def truncate_at_turn_boundary(
    messages: Sequence[Message], target_tokens: int
) -> tuple[Message, ...]:
    """Smallest turn aligned prefix carrying at least `target_tokens` tokens."""
    total = 0
    for position, message in enumerate(messages):
        total += message.token_count
        if total >= target_tokens:
            return tuple(messages[: position + 1])
    raise ValueError(
        f"trajectory holds {total} tokens, below the target of {target_tokens}; "
        "candidates must be filtered before truncation"
    )


def build_openresearcher_contexts(
    conversations: Sequence[Conversation],
    *,
    split: str,
    target_tokens: int,
    n_contexts: int,
    dataset: str = "openresearcher",
    context_set_version: str = "v1",
) -> tuple[FillerContext, ...]:
    """Algorithm 14.2. Deterministic ordering, no stitching, no randomness."""
    candidates = [c for c in conversations if c.token_count > target_tokens]
    if len(candidates) < n_contexts:
        raise InsufficientDataError(
            f"{dataset}/{split}: {len(candidates)} trajectories exceed {target_tokens} tokens, "
            f"needed {n_contexts}"
        )
    # Step 3: first N candidates under deterministic ordering.
    selected = sorted(candidates, key=lambda c: c.source_index)[:n_contexts]

    contexts: list[FillerContext] = []
    for position, conversation in enumerate(selected):
        kept = truncate_at_turn_boundary(conversation.messages, target_tokens)
        history = History(
            messages=tuple(m.model_copy(update={"index": i}) for i, m in enumerate(kept))
        )
        contexts.append(
            FillerContext(
                context_id=f"{dataset}_{split}_{target_tokens}_{position:04d}",
                dataset=dataset,
                split=split,
                target_tokens=target_tokens,
                actual_tokens=history.token_count,
                history=history,
                source_ids=(conversation.conversation_id,),
                n_stitched=1,
                status=ContextStatus.TRUNCATED_AT_TURN_BOUNDARY,
                context_set_version=context_set_version,
                detail=(
                    f"kept {len(kept)} of {len(conversation.messages)} messages, the smallest "
                    f"turn aligned prefix at or above {target_tokens} tokens"
                ),
            )
        )
    return tuple(contexts)


def concatenate_for_220k(
    contexts: Sequence[FillerContext], *, target_tokens: int = 220000
) -> tuple[FillerContext, ...]:
    """FR-019: each 220K entry is built by CONCATENATING TWO 110K entries.

    OpenResearcher contains no native 220K rows. `IMPLEMENTATION NOTE` spec 14.2: this yields
    TWO user turns, so the injection conditions partially un-collapse for that condition alone.
    That asymmetry is recorded on the context rather than left for a reader to infer.
    """
    if len(contexts) % 2 != 0:
        raise ValueError(f"pairing requires an even number of source contexts, got {len(contexts)}")
    paired: list[FillerContext] = []
    for position in range(0, len(contexts), 2):
        first, second = contexts[position], contexts[position + 1]
        merged = list(first.history.messages) + list(second.history.messages)
        history = History(
            messages=tuple(m.model_copy(update={"index": i}) for i, m in enumerate(merged))
        )
        paired.append(
            FillerContext(
                context_id=f"{first.dataset}_{first.split}_{target_tokens}_{position // 2:04d}",
                dataset=first.dataset,
                split=first.split,
                target_tokens=target_tokens,
                actual_tokens=history.token_count,
                history=history,
                source_ids=first.source_ids + second.source_ids,
                n_stitched=first.n_stitched + second.n_stitched,
                status=ContextStatus.TRUNCATED_AT_TURN_BOUNDARY,
                context_set_version=first.context_set_version,
                detail=(
                    "FR-019: two entries concatenated for the 220K condition, which yields two "
                    "user turns and partially un-collapses the injection conditions"
                ),
            )
        )
    return tuple(paired)
