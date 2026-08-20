"""RQ2: how do compaction system factors affect SC retention? TASK-031, TASK-032, spec 15.7.

Two sweeps:

- **Context length** (10K, 50K, 100K). The expected result is a monotonic decline. Spec 6.14
  gives the mechanism: compactor output length is fixed by the compaction prompt, not by input
  length, so as input grows the compression ratio grows proportionally and the probability that
  any specific non task span survives falls roughly as 1/|H^t|.

- **Compaction rate** (the Table 11 equivalent). Output length is expected to grow only 0.84x
  to 1.28x while input grows 10x. That near invariance is the whole mechanism behind the
  failure, which is why it is measured directly rather than assumed.
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from compint.core.framing import FramingSpec
from compint.core.models import InjectionCondition
from compint.data.contexts import FillerContext
from compint.eval.metrics import RetentionResult, retention_rate
from compint.eval.records import RetentionRecord
from compint.experiments.base import GridCell, build_grid
from compint.experiments.runner import RunResult
from shared.errors import EmptyEvaluationSetError


class LengthPoint(BaseModel):
    """One (dataset, compactor, target length) measurement."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    compactor_id: str
    target_tokens: int
    retention: RetentionResult
    mean_output_tokens: float
    mean_compaction_ratio: float


class LengthSweep(BaseModel):
    model_config = ConfigDict(frozen=True)

    points: tuple[LengthPoint, ...]

    def series(self, dataset: str, compactor_id: str) -> tuple[LengthPoint, ...]:
        return tuple(
            sorted(
                (p for p in self.points if p.dataset == dataset and p.compactor_id == compactor_id),
                key=lambda p: p.target_tokens,
            )
        )

    def declines_monotonically(self, dataset: str, compactor_id: str) -> bool:
        """TASK-031 acceptance: retention declines as context length grows."""
        rates = [point.retention.rate for point in self.series(dataset, compactor_id)]
        return all(earlier >= later for earlier, later in itertools.pairwise(rates))

    def output_length_growth(self, dataset: str, compactor_id: str) -> float | None:
        """Output tokens at the longest length divided by output tokens at the shortest.

        Spec 5.2 reports 0.84x to 1.28x across a 10x input increase. A value near 1 confirms
        the invariance; a value near 10 would mean the compactor is scaling with input and the
        mechanism does not apply.
        """
        series = self.series(dataset, compactor_id)
        if len(series) < 2 or series[0].mean_output_tokens == 0:
            return None
        return series[-1].mean_output_tokens / series[0].mean_output_tokens


def build_length_grid(
    contexts_by_length: dict[int, Sequence[FillerContext]],
    sc_ids: Sequence[int],
    compactor_ids: Sequence[str],
    *,
    framing: FramingSpec,
    injection_seed: int,
    prompt_hashes: dict[str, str],
) -> tuple[GridCell, ...]:
    cells: list[GridCell] = []
    for _length, contexts in sorted(contexts_by_length.items()):
        cells.extend(
            build_grid(
                contexts,
                sc_ids,
                compactor_ids,
                framing=framing,
                condition=InjectionCondition.TOP,
                injection_seed=injection_seed,
                prompt_hashes=prompt_hashes,
            )
        )
    return tuple(cells)


def summarize_length_sweep(result: RunResult, target_by_context: dict[str, int]) -> LengthSweep:
    """Group a completed run by (dataset, compactor, target length)."""
    buckets: dict[tuple[str, str, int], list[RetentionRecord]] = {}
    outputs: dict[tuple[str, str, int], list[int]] = {}
    ratios: dict[tuple[str, str, int], list[float]] = {}

    for outcome in result.outcomes:
        target = target_by_context.get(outcome.instance_id)
        if target is None:
            continue
        key = (outcome.dataset, outcome.compactor_id, target)
        if outcome.retention is not None:
            buckets.setdefault(key, []).append(outcome.retention)
        if outcome.compacted_tokens > 0:
            outputs.setdefault(key, []).append(outcome.compacted_tokens)
            ratios.setdefault(key, []).append(outcome.input_tokens / outcome.compacted_tokens)

    points: list[LengthPoint] = []
    for key, records in sorted(buckets.items()):
        dataset, compactor_id, target = key
        try:
            retention = retention_rate(records)
        except EmptyEvaluationSetError:
            continue
        points.append(
            LengthPoint(
                dataset=dataset,
                compactor_id=compactor_id,
                target_tokens=target,
                retention=retention,
                mean_output_tokens=statistics.fmean(outputs.get(key, [0])),
                mean_compaction_ratio=statistics.fmean(ratios.get(key, [0.0])),
            )
        )
    return LengthSweep(points=tuple(points))


# PAPER SPECIFICATION Table 11: mean post compaction context length in tokens. Used to check
# that a reproduction's compactors land in the same band, never to shape their output.
TABLE11_OUTPUT_TOKENS: dict[tuple[str, str, int], int] = {
    ("hermes_agent", "gpt_oss_120b__anthropic", 10000): 672,
    ("hermes_agent", "gpt_oss_120b__anthropic", 50000): 844,
    ("hermes_agent", "gpt_oss_120b__anthropic", 100000): 857,
    ("hermes_agent", "gpt_oss_120b__pi_mono", 10000): 533,
    ("hermes_agent", "gpt_oss_120b__pi_mono", 50000): 616,
    ("hermes_agent", "gpt_oss_120b__pi_mono", 100000): 631,
    ("hermes_agent", "qwen3_30b_a3b__anthropic", 10000): 301,
    ("hermes_agent", "qwen3_30b_a3b__anthropic", 50000): 337,
    ("hermes_agent", "qwen3_30b_a3b__anthropic", 100000): 349,
    ("wildchat", "gpt_oss_120b__anthropic", 10000): 511,
    ("wildchat", "gpt_oss_120b__anthropic", 50000): 587,
    ("wildchat", "gpt_oss_120b__anthropic", 100000): 614,
    ("wildchat", "gpt_oss_120b__pi_mono", 10000): 443,
    ("wildchat", "gpt_oss_120b__pi_mono", 50000): 484,
    ("wildchat", "gpt_oss_120b__pi_mono", 100000): 478,
    ("wildchat", "qwen3_30b_a3b__anthropic", 10000): 451,
    ("wildchat", "qwen3_30b_a3b__anthropic", 50000): 369,
    ("wildchat", "qwen3_30b_a3b__anthropic", 100000): 377,
}

# Spec 11.3 and Table 11: LLM compactor output spans 301 to 857 tokens.
OUTPUT_BAND = (301, 857)


def output_within_band(mean_output_tokens: float) -> bool:
    """TASK-013 acceptance: output lengths fall in the Table 11 band."""
    return OUTPUT_BAND[0] <= mean_output_tokens <= OUTPUT_BAND[1]
