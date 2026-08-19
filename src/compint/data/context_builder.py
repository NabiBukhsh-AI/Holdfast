"""Context builder orchestration and Table 1 validation. TASK-010.

Produces 50 evaluation contexts and 10 development contexts per dataset at each target length,
plus the statistics report that gates the whole downstream pipeline.

`GATE` spec 32.2: if the Table 1 statistics cannot be reproduced within tolerance, the contexts
are wrong, so every downstream number is wrong. That is escalation trigger 2, not a bug to
grind through.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from compint.core.models import Conversation
from compint.core.tokenization import Tokenizer, assert_reportable
from compint.data.contexts import FillerContext
from compint.data.embedding import EmbeddingModel, embed_conversations
from compint.data.knn_index import ExactKNNIndex
from compint.data.splits import PartitionedPool, Split, partition_pool
from compint.data.stitching import topic_cohesive_stitch
from compint.data.truncation import build_openresearcher_contexts
from shared.config import AppConfig
from shared.errors import ConfigError

# PAPER SPECIFICATION Table 1, as reported in the engineering spec at a 100K target.
# Used to gate context construction, not to shape it.
TABLE1_REFERENCE: dict[str, dict[str, float]] = {
    "wildchat": {"n_user_turns": 257.22, "n_stitched": 128.4},
    "hermes_agent": {"n_turns": 119.66, "n_user_turns": 9.04, "n_stitched": 6.98},
    "openresearcher": {"n_turns": 310.46, "n_user_turns": 1.00},
}

# ENGINEERING RECOMMENDATION: the paper reports no variance, so the reproduction defines its
# own bands. User turn counts are structural and must land close; stitch counts depend on the
# corpus revision (U-13) and are given more room.
TABLE1_TOLERANCE: dict[str, float] = {
    "n_turns": 0.15,
    "n_user_turns": 0.15,
    "n_stitched": 0.25,
}


class DatasetStatistics(BaseModel):
    """One row of the Table 1 equivalent report."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    split: str
    target_tokens: int
    n_contexts: int
    mean_tokens: float
    mean_turns: float
    mean_user_turns: float
    mean_stitched: float
    n_degenerate: int
    statuses: dict[str, int]

    def compare_to_table1(self) -> dict[str, dict[str, float | bool]]:
        """Relative deviation from the published figures, per available column."""
        reference = TABLE1_REFERENCE.get(self.dataset, {})
        observed = {
            "n_turns": self.mean_turns,
            "n_user_turns": self.mean_user_turns,
            "n_stitched": self.mean_stitched,
        }
        report: dict[str, dict[str, float | bool]] = {}
        for column, expected in reference.items():
            actual = observed[column]
            deviation = abs(actual - expected) / expected if expected else 0.0
            report[column] = {
                "expected": expected,
                "actual": actual,
                "relative_deviation": deviation,
                "within_tolerance": deviation <= TABLE1_TOLERANCE[column],
            }
        return report

    def assert_within_tolerance(self) -> None:
        """Escalation trigger 2. Wrong contexts poison everything downstream."""
        failures = {
            column: values
            for column, values in self.compare_to_table1().items()
            if not values["within_tolerance"]
        }
        if failures:
            raise ConfigError(
                f"{self.dataset}/{self.split} at {self.target_tokens} tokens does not reproduce "
                f"Table 1 within tolerance: {failures}. The contexts are wrong, so every "
                "downstream number would be wrong. Escalate rather than proceeding."
            )


def summarize(contexts: Sequence[FillerContext]) -> DatasetStatistics:
    """Aggregate a context set into its Table 1 row."""
    if not contexts:
        raise ConfigError("cannot summarize an empty context set")
    first = contexts[0]
    statuses: dict[str, int] = {}
    for context in contexts:
        statuses[context.status.value] = statuses.get(context.status.value, 0) + 1
    return DatasetStatistics(
        dataset=first.dataset,
        split=first.split,
        target_tokens=first.target_tokens,
        n_contexts=len(contexts),
        mean_tokens=statistics.fmean(c.actual_tokens for c in contexts),
        mean_turns=statistics.fmean(c.n_turns for c in contexts),
        mean_user_turns=statistics.fmean(c.n_user_turns for c in contexts),
        mean_stitched=statistics.fmean(c.n_stitched for c in contexts),
        n_degenerate=sum(1 for c in contexts if c.is_degenerate_for_injection),
        statuses=statuses,
    )


class ContextBuilder:
    """Ingested conversations to filler contexts, per dataset and split."""

    def __init__(
        self,
        config: AppConfig,
        tokenizer: Tokenizer,
        embedding_model: EmbeddingModel,
        *,
        require_reportable: bool = False,
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._embedding_model = embedding_model
        if require_reportable:
            # A reported run may not define its context lengths with the approximation
            # tokenizer, nor rank neighbors with the stub encoder.
            assert_reportable(tokenizer)
            if not getattr(embedding_model, "reportable", False):
                raise ConfigError(
                    f"embedding model {embedding_model.id} is a stub and must not construct "
                    "contexts for a reported run"
                )
            config.context.require_embedding_revision()

    def partition(self, conversations: Sequence[Conversation], dataset: str) -> PartitionedPool:
        """Partition BEFORE stitching, so prompt development cannot leak into results."""
        return partition_pool(conversations, dataset)

    def build(
        self,
        conversations: Sequence[Conversation],
        *,
        dataset: str,
        split: Split,
        target_tokens: int | None = None,
        n_contexts: int | None = None,
    ) -> tuple[FillerContext, ...]:
        """Construct one context set. Dispatches on the corpus's construction method."""
        target = target_tokens or self._config.context.target_tokens
        count = n_contexts or (
            self._config.context.eval_contexts
            if split is Split.EVAL
            else self._config.context.dev_contexts
        )
        if dataset == "openresearcher":
            # FR-016: native long trajectories, truncated at a turn boundary. No stitching.
            return build_openresearcher_contexts(
                conversations,
                split=split.value,
                target_tokens=target,
                n_contexts=count,
            )
        embeddings = embed_conversations(
            conversations, self._embedding_model, self._config.context.serialization
        )
        index = ExactKNNIndex(embeddings)
        return topic_cohesive_stitch(
            conversations,
            embeddings,
            index,
            self._tokenizer,
            dataset=dataset,
            split=split.value,
            target_tokens=target,
            n_contexts=count,
            knn_k=self._config.context.knn_k,
            soft_cap_multiplier=self._config.context.soft_cap_multiplier,
            crop_granularity=self._config.context.crop_granularity,
        )

    def build_both_splits(
        self, conversations: Sequence[Conversation], *, dataset: str
    ) -> dict[str, tuple[FillerContext, ...]]:
        """Build dev and eval context sets from disjoint source pools."""
        pool = self.partition(conversations, dataset)
        return {
            Split.DEV.value: self.build(pool.dev, dataset=dataset, split=Split.DEV),
            Split.EVAL.value: self.build(pool.eval, dataset=dataset, split=Split.EVAL),
        }


def assert_no_source_leakage(
    dev: Sequence[FillerContext], evaluation: Sequence[FillerContext]
) -> None:
    """TASK-009 acceptance: no source conversation appears in both a dev and an eval context."""
    dev_sources = {source for context in dev for source in context.source_ids}
    eval_sources = {source for context in evaluation for source in context.source_ids}
    overlap = dev_sources & eval_sources
    if overlap:
        raise ConfigError(
            f"{len(overlap)} source conversations appear in both dev and eval contexts, for "
            f"example {sorted(overlap)[:5]}. Prompt tuning on dev would leak into results."
        )
