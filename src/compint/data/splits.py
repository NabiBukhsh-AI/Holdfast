"""dev and eval pool partitioning. TASK-009, spec 12.2 stage 8.

`PAPER SPECIFICATION`: there is no train/validation/test split, because no model is trained
anywhere in this work. What replaces it is a held out PROMPT DEVELOPMENT split, and it must be
created because it does not exist in the paper.

`CRITICAL DATA LEAKAGE CHECK` spec 12.2: the stitching pool is partitioned BEFORE stitching,
not after. Algorithm 1 removes used samples from the pool within a run, so partitioning up
front is a small change with large correctness value: without it, any extractor prompt
iteration on `dev` is fitting on the reported test set.

This does not resolve UNKNOWN U-16, which asks whether the PAPER's own extraction prompt was
developed on its reported contexts. It prevents the same contamination here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from compint.core.models import Conversation
from shared.errors import ConfigError


class Split(StrEnum):
    DEV = "dev"
    EVAL = "eval"


class PartitionedPool(BaseModel):
    """Two disjoint source pools. Disjointness is a property, not a hope."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    dev: tuple[Conversation, ...]
    eval: tuple[Conversation, ...]
    dev_fraction: float
    salt: str

    def pool(self, split: Split) -> tuple[Conversation, ...]:
        return self.dev if split is Split.DEV else self.eval

    def assert_disjoint(self) -> None:
        """No source conversation may appear in both partitions."""
        dev_hashes = {c.content_hash() for c in self.dev}
        eval_hashes = {c.content_hash() for c in self.eval}
        overlap = dev_hashes & eval_hashes
        if overlap:
            raise ConfigError(
                f"{len(overlap)} source conversations appear in both dev and eval for "
                f"{self.dataset}. Prompt tuning on dev would leak into reported results."
            )


def _bucket(content_hash: str, salt: str) -> float:
    """Map a conversation to [0, 1) deterministically by content, never by position.

    Content addressed rather than index addressed so that adding rows to the corpus does not
    reshuffle the partition and silently move a conversation from eval to dev.
    """
    digest = hashlib.sha256(f"{salt}:{content_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def partition_pool(
    conversations: Sequence[Conversation],
    dataset: str,
    *,
    dev_fraction: float = 0.2,
    salt: str = "holdfast-split-v1",
) -> PartitionedPool:
    """Split the source pool deterministically by content hash, before any stitching."""
    if not 0.0 < dev_fraction < 1.0:
        raise ConfigError(f"dev_fraction must be in (0, 1), got {dev_fraction}")
    dev: list[Conversation] = []
    evaluation: list[Conversation] = []
    for conversation in conversations:
        target = dev if _bucket(conversation.content_hash(), salt) < dev_fraction else evaluation
        target.append(conversation)
    pool = PartitionedPool(
        dataset=dataset,
        dev=tuple(dev),
        eval=tuple(evaluation),
        dev_fraction=dev_fraction,
        salt=salt,
    )
    pool.assert_disjoint()
    return pool


def assert_reportable_split(split: Split) -> None:
    """A report generator refuses to emit dev results as headline numbers (TASK-009)."""
    if split is not Split.EVAL:
        raise ConfigError(
            f"results from the {split.value} split are for prompt development only and must "
            "never be reported as headline numbers (spec 12.2 stage 8)"
        )
