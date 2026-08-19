"""Equations 6, 8, and 9 plus Wilson intervals. TASK-016.

Spec 6.8, 6.11, 6.12, and Algorithm 14.11.

INV-6 is enforced structurally: no type in this module exposes an accessor that returns a
bare rate. Every rate arrives with its denominator, its exclusion counts, the reasons for
those exclusions, and a Wilson interval. A reader who copies a number out of this system
cannot accidentally drop the information needed to interpret it.

Effect Retention NEVER clips. A negative ER means compaction made compliance worse than
having no constraint at all, which is a real and important signal, not an error to smooth
away. An ER above 1 is anomalous and is flagged for investigation, never silently truncated.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from compint.eval.records import ProbeRecord, ProbeStatus, RetentionRecord, RetentionStatus
from shared.errors import EmptyEvaluationSetError

# ENGINEERING RECOMMENDATION spec 6.12. UNKNOWN in the paper: below this the ER denominator
# is too small for the ratio to mean anything, so ER is reported as undefined rather than as
# a large number produced by division by near zero.
MIN_ER_DENOMINATOR = 0.05


class WilsonInterval(BaseModel):
    """95 percent Wilson score interval.

    ENGINEERING RECOMMENDATION spec 6.11. The paper reports no uncertainty on any figure. At
    N=750 a rate of 17 percent carries roughly plus or minus 2.7 points; at N=300 roughly
    plus or minus 4.3. Several of the paper's comparative claims sit inside those intervals.
    Wilson is used rather than the normal approximation because several cells sit at or near
    0.0 percent, where the normal interval is invalid.
    """

    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    confidence: float = 0.95

    @property
    def half_width_pp(self) -> float:
        return (self.upper - self.lower) * 50.0

    def format_pp(self) -> str:
        return f"[{self.lower * 100:.1f}, {self.upper * 100:.1f}]"


def _z_for(confidence: float) -> float:
    """Two sided normal quantile. Table lookup for the values actually used."""
    table = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    if confidence in table:
        return table[confidence]
    # Acklam style inverse normal is overkill here; refuse rather than approximate silently.
    raise ValueError(
        f"unsupported confidence {confidence}; supported: {sorted(table)}. "
        "Add the exact quantile rather than approximating it."
    )


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> WilsonInterval:
    """Wilson score interval for a binomial proportion. Valid at 0 and at 1."""
    if n < 0 or successes < 0:
        raise ValueError(f"counts must be non negative, got successes={successes} n={n}")
    if successes > n:
        raise ValueError(f"successes {successes} exceeds n {n}")
    if n == 0:
        return WilsonInterval(lower=0.0, upper=1.0, confidence=confidence)
    z = _z_for(confidence)
    p = successes / n
    denominator = 1.0 + (z * z) / n
    centre = p + (z * z) / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
    lower = (centre - spread) / denominator
    upper = (centre + spread) / denominator
    return WilsonInterval(
        lower=max(0.0, lower), upper=min(1.0, upper), confidence=confidence
    )


class RateResult(BaseModel):
    """A rate that cannot be read without its denominator (INV-6)."""

    model_config = ConfigDict(frozen=True)

    n_correct: int = Field(ge=0)
    n_valid: int = Field(ge=0)
    n_excluded: int = Field(ge=0)
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)
    wilson_ci: WilsonInterval

    @property
    def rate(self) -> float:
        """Only reachable when n_valid > 0; construction refuses an empty denominator."""
        return self.n_correct / self.n_valid

    @property
    def percent(self) -> float:
        return self.rate * 100.0

    def format(self) -> str:
        """The canonical rendering: rate, interval, denominator, exclusions."""
        excluded = (
            f", excluded {self.n_excluded} ({self.exclusion_reasons})"
            if self.n_excluded
            else ""
        )
        return (
            f"{self.percent:.1f}% {self.wilson_ci.format_pp()} "
            f"n={self.n_valid}{excluded}"
        )


class RetentionResult(RateResult):
    """Retention Rate. Equation 6 aggregated, spec 24.1."""

    n_blocked: int = Field(default=0, ge=0)
    n_unparseable: int = Field(default=0, ge=0)
    n_degenerate: int = Field(default=0, ge=0)


class ComplianceResult(RateResult):
    """Compliance rate c_bar_g. Equation 8."""

    condition: str = ""


class ERStatus(StrEnum):
    """Effect Retention outcome classes. Spec 6.12 numerical hazards."""

    OK = "OK"
    # c_ub is too close to c_lctx: the probe model already picks the compliant option without
    # any constraint, so the normalization has no headroom and ER is undefined.
    DEGENERATE_DENOMINATOR = "DEGENERATE_DENOMINATOR"
    # c_comp < c_lctx: compaction made compliance WORSE than having no constraint at all.
    NEGATIVE = "NEGATIVE"
    # c_comp > c_ub: anomalous, investigate. Never clipped.
    ABOVE_UPPER_BOUND = "ABOVE_UPPER_BOUND"


class EffectRetentionResult(BaseModel):
    """Equation 9 with honest failure reporting."""

    model_config = ConfigDict(frozen=True)

    value: float | None
    status: ERStatus
    detail: str = ""
    c_comp: float | None = None
    c_lctx: float | None = None
    c_ub: float | None = None
    components: dict[str, ComplianceResult] = Field(default_factory=dict)

    @property
    def percent(self) -> float | None:
        return None if self.value is None else self.value * 100.0

    def format(self) -> str:
        if self.value is None:
            return f"undefined ({self.status.value}: {self.detail})"
        suffix = "" if self.status is ERStatus.OK else f" [{self.status.value}]"
        return f"{self.value * 100:.1f}%{suffix}"


def retention_rate(
    records: Sequence[RetentionRecord], confidence: float = 0.95
) -> RetentionResult:
    """Aggregate Equation 6 over an evaluation set.

    FR-041 and FR-043: UNPARSEABLE and BLOCKED are excluded from the denominator and counted,
    never coerced to 0. The counts travel with the rate so that a reader can see how much of
    the grid the number actually rests on.
    """
    valid = [r for r in records if r.status is RetentionStatus.OK and r.verdict is not None]
    if not valid:
        raise EmptyEvaluationSetError(
            f"no valid retention records among {len(records)}; "
            f"statuses={dict(Counter(r.status.value for r in records))}"
        )
    n_correct = sum(1 for r in valid if r.verdict == "YES")
    excluded = [r for r in records if r not in valid]
    reasons = Counter(r.status.value for r in excluded)
    return RetentionResult(
        n_correct=n_correct,
        n_valid=len(valid),
        n_excluded=len(excluded),
        exclusion_reasons=dict(reasons),
        wilson_ci=wilson_interval(n_correct, len(valid), confidence),
        n_blocked=reasons.get(RetentionStatus.BLOCKED.value, 0),
        n_unparseable=reasons.get(RetentionStatus.UNPARSEABLE.value, 0),
        n_degenerate=sum(1 for r in records if r.degenerate),
    )


def compliance_rate(
    records: Sequence[ProbeRecord], confidence: float = 0.95
) -> ComplianceResult:
    """Equation 8. Returns the rate plus the diagnostic counts the paper omits."""
    valid = [r for r in records if r.status is ProbeStatus.OK and r.answer is not None]
    if not valid:
        raise EmptyEvaluationSetError(
            f"no valid probe records among {len(records)}; "
            f"statuses={dict(Counter(r.status.value for r in records))}"
        )
    n_correct = sum(1 for r in valid if r.answer == r.gold)
    excluded = [r for r in records if r.status is not ProbeStatus.OK or r.answer is None]
    conditions = {r.condition for r in records}
    return ComplianceResult(
        n_correct=n_correct,
        n_valid=len(valid),
        n_excluded=len(excluded),
        exclusion_reasons=dict(Counter(r.status.value for r in excluded)),
        wilson_ci=wilson_interval(n_correct, len(valid), confidence),
        condition=conditions.pop() if len(conditions) == 1 else "mixed",
    )


def effect_retention(
    c_comp: float,
    c_lctx: float,
    c_ub: float,
    *,
    min_denominator: float = MIN_ER_DENOMINATOR,
) -> EffectRetentionResult:
    """Equation 9:  ER = (c_comp - c_lctx) / (c_ub - c_lctx). Never clips. Flags instead."""
    for name, value in (("c_comp", c_comp), ("c_lctx", c_lctx), ("c_ub", c_ub)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a rate in [0, 1], got {value}")
    denominator = c_ub - c_lctx
    if abs(denominator) < min_denominator:
        return EffectRetentionResult(
            value=None,
            status=ERStatus.DEGENERATE_DENOMINATOR,
            detail=(
                f"c_ub ({c_ub:.4f}) too close to c_lctx ({c_lctx:.4f}); "
                f"ER undefined below a denominator of {min_denominator}"
            ),
            c_comp=c_comp,
            c_lctx=c_lctx,
            c_ub=c_ub,
        )
    value = (c_comp - c_lctx) / denominator
    if value < 0:
        status = ERStatus.NEGATIVE
        detail = "compaction produced worse compliance than having no constraint at all"
    elif value > 1:
        status = ERStatus.ABOVE_UPPER_BOUND
        detail = "c_comp exceeds the post compaction upper bound; investigate"
    else:
        status = ERStatus.OK
        detail = ""
    return EffectRetentionResult(
        value=value,
        status=status,
        detail=detail,
        c_comp=c_comp,
        c_lctx=c_lctx,
        c_ub=c_ub,
    )


def aggregate_effect_retention(
    records: Sequence[ProbeRecord],
    *,
    min_denominator: float = MIN_ER_DENOMINATOR,
    confidence: float = 0.95,
) -> EffectRetentionResult:
    """Algorithm 14.11. Compute all four condition rates, then Equation 9 over them."""
    by_condition: dict[str, list[ProbeRecord]] = {}
    for record in records:
        by_condition.setdefault(record.condition, []).append(record)
    required = ("lctx", "comp", "ub")
    missing = [c for c in required if c not in by_condition]
    if missing:
        raise EmptyEvaluationSetError(
            f"Effect Retention needs conditions {required}, missing {missing}"
        )
    components = {
        condition: compliance_rate(rows, confidence)
        for condition, rows in sorted(by_condition.items())
    }
    result = effect_retention(
        components["comp"].rate,
        components["lctx"].rate,
        components["ub"].rate,
        min_denominator=min_denominator,
    )
    return result.model_copy(update={"components": components})


def compaction_ratio(input_tokens: int, output_tokens: int) -> float | None:
    """Spec 6.14:  |H^t| / |C(H^t)|. None when the compactor produced nothing."""
    if output_tokens <= 0:
        return None
    return input_tokens / output_tokens


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa over two binary label sequences. Spec 6.15.

    The paper uses sklearn. This implementation exists so the metric is available in CI
    without the optional research dependency set, and `judge_agreement.py` cross checks it
    against sklearn when that package is installed.
    """
    if len(a) != len(b):
        raise ValueError(f"label sequences differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise EmptyEvaluationSetError("cannot compute kappa over zero pairs")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    labels = set(a) | set(b)
    expected = sum(
        (sum(1 for x in a if x == label) / n) * (sum(1 for y in b if y == label) / n)
        for label in labels
    )
    if math.isclose(expected, 1.0):
        # Both raters used a single label for everything: kappa is undefined, not 1.0.
        raise ValueError(
            "expected agreement is 1.0, so kappa is undefined; report raw agreement and the "
            "confusion matrix instead"
        )
    return (observed - expected) / (1.0 - expected)
