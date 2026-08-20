"""Ablations. TASK-038, TASK-039, spec 15.8.

Two questions, both of which exist to rule out an easy explanation of the headline finding.

**SC targeted prompt (TASK-038).** If simply asking the compactor to preserve constraints fixed
the problem, the architectural argument would collapse. It does not: the source reports
gpt-oss moving 1.3 to 24.8 and Qwen3 3.3 to 37.6. Large relative gains, still far below what a
registry outside the compression path achieves. The ablation measures the LIFT and checks it
lands in that shape, because "the prompt helps a lot and is still not enough" is the actual
claim being reproduced.

**Source and compactor matching (TASK-039).** A skeptic can argue the compactors do badly
because the trajectories came from a different model family. Matching them (gpt-oss compacting
gpt-oss generated context) gives 8.4 percent, so mismatch is not the explanation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from compint.eval.metrics import RetentionResult, retention_rate
from compint.experiments.runner import RunResult
from shared.errors import EmptyEvaluationSetError

# PAPER SPECIFICATION spec 15.8 TASK-038 acceptance figures, as (baseline, sc_targeted).
PAPER_SC_TARGETED_LIFT: dict[str, tuple[float, float]] = {
    "gpt_oss_120b": (1.3, 24.8),
    "qwen3_30b_a3b": (3.3, 37.6),
}

# PAPER SPECIFICATION TASK-039: matched gpt-oss retention.
PAPER_MATCHED_RETENTION_PCT = 8.4

# A prompt that closed most of the gap would undercut the architectural argument, so the
# ablation records where the lift lands rather than only that it is positive.
REGISTRY_REFERENCE_PCT = 93.7


class PromptAblation(BaseModel):
    """Baseline prompt against its SC targeted variant, for one model."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    baseline: RetentionResult
    sc_targeted: RetentionResult

    @property
    def lift_pp(self) -> float:
        """Absolute gain in percentage points."""
        return self.sc_targeted.percent - self.baseline.percent

    @property
    def relative_lift(self) -> float | None:
        """Multiplicative gain. None when the baseline is zero, where a ratio is meaningless."""
        if self.baseline.percent == 0:
            return None
        return self.sc_targeted.percent / self.baseline.percent

    @property
    def remaining_gap_pp(self) -> float:
        """How far the improved prompt still sits below the registry approach.

        This is the number that carries the argument. A large lift that still leaves a large
        gap says prompt engineering helps and does not solve it.
        """
        return REGISTRY_REFERENCE_PCT - self.sc_targeted.percent

    def matches_paper_shape(self) -> bool:
        """Substantial lift, still far short of the registry."""
        return self.lift_pp > 0 and self.remaining_gap_pp > 30.0

    def format(self) -> str:
        relative = f"{self.relative_lift:.1f}x" if self.relative_lift is not None else "n/a"
        return (
            f"{self.model_id}: {self.baseline.percent:.1f}% -> "
            f"{self.sc_targeted.percent:.1f}% (+{self.lift_pp:.1f} pp, {relative}), "
            f"still {self.remaining_gap_pp:.1f} pp below the registry approach"
        )


def summarize_prompt_ablation(
    result: RunResult,
    *,
    model_id: str,
    baseline_compactor: str,
    sc_targeted_compactor: str,
    dataset: str | None = None,
) -> PromptAblation | None:
    """Pair a baseline compactor with its SC targeted variant."""
    try:
        baseline = retention_rate(result.retention_records(dataset, baseline_compactor))
        targeted = retention_rate(result.retention_records(dataset, sc_targeted_compactor))
    except EmptyEvaluationSetError:
        return None
    return PromptAblation(model_id=model_id, baseline=baseline, sc_targeted=targeted)


class MatchingAblation(BaseModel):
    """Does compacting a model's own trajectories rescue retention? Spec 15.8 says no."""

    model_config = ConfigDict(frozen=True)

    compactor_id: str
    matched: RetentionResult
    mismatched: RetentionResult | None = None

    @property
    def matched_percent(self) -> float:
        return self.matched.percent

    def mismatch_explains_the_failure(self) -> bool:
        """True only if matching actually rescues retention.

        The expected answer is False. Matched retention lands near 8.4 percent, which is the
        same failure regime, so provenance mismatch is not the cause.
        """
        if self.mismatched is None:
            return self.matched.percent > 50.0
        return self.matched.percent - self.mismatched.percent > 25.0

    def format(self) -> str:
        against = (
            f" against {self.mismatched.percent:.1f}% mismatched"
            if self.mismatched is not None
            else ""
        )
        verdict = (
            "mismatch DOES explain the failure"
            if self.mismatch_explains_the_failure()
            else "mismatch does not explain the failure"
        )
        return f"{self.compactor_id}: {self.matched.percent:.1f}% matched{against} ({verdict})"


def summarize_matching_ablation(
    result: RunResult,
    *,
    compactor_id: str,
    matched_dataset: str,
    mismatched_dataset: str | None = None,
) -> MatchingAblation | None:
    try:
        matched = retention_rate(result.retention_records(matched_dataset, compactor_id))
    except EmptyEvaluationSetError:
        return None
    mismatched = None
    if mismatched_dataset is not None:
        try:
            mismatched = retention_rate(result.retention_records(mismatched_dataset, compactor_id))
        except EmptyEvaluationSetError:
            mismatched = None
    return MatchingAblation(compactor_id=compactor_id, matched=matched, mismatched=mismatched)
