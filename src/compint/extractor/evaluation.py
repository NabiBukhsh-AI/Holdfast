"""Extractor precision and injection resistance measurement. TASK-019, TASK-029.

The paper reports extractor RETENTION (recall) only. `EXPERIMENT E-02` spec 30.4 ranks
precision as one of the two highest priority missing experiments, and the reason is blunt: an
extractor that fires on every turn scores 100 percent on the paper's metric. False positive
constraints are a direct user harm, because the registry is an instruction channel that the
downstream agent obeys.

This module measures both halves against the committed fixture suites, and it is the same
code path the research experiment and the security test use, so the numbers reported in a run
manifest are the numbers CI checks.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from compint.eval.metrics import WilsonInterval, wilson_interval
from compint.extractor.client import SCExtractor
from compint.extractor.parser import ExtractionStatus


class SuiteCase(BaseModel):
    """One fixture row."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    user_turn: str
    expect_extraction: bool
    why: str = ""
    attack_class: str | None = None
    generic_clause: str | None = None
    category: str | None = None


class CaseOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    expected: bool
    observed: bool
    status: ExtractionStatus
    extracted_texts: tuple[str, ...] = ()
    attack_class: str | None = None
    passed: bool = False


class SuiteResult(BaseModel):
    """Outcome counts plus the interval, because a rate without its denominator is unreadable."""

    model_config = ConfigDict(frozen=True)

    suite: str
    n_cases: int = Field(ge=0)
    n_passed: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    n_errored: int = Field(ge=0)
    outcomes: tuple[CaseOutcome, ...] = ()
    wilson_ci: WilsonInterval

    @property
    def pass_rate(self) -> float:
        scored = self.n_passed + self.n_failed
        if scored == 0:
            raise ValueError(f"{self.suite}: no case produced a scoreable outcome")
        return self.n_passed / scored

    def by_attack_class(self) -> dict[str, dict[str, int]]:
        """Failures grouped by mechanism, so a defense can be aimed rather than guessed at."""
        grouped: dict[str, dict[str, int]] = {}
        for outcome in self.outcomes:
            if outcome.attack_class is None:
                continue
            bucket = grouped.setdefault(outcome.attack_class, {"passed": 0, "failed": 0})
            bucket["passed" if outcome.passed else "failed"] += 1
        return grouped

    def failures(self) -> tuple[CaseOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.passed)

    def format(self) -> str:
        return (
            f"{self.suite}: {self.pass_rate * 100:.1f}% {self.wilson_ci.format_pp()} "
            f"passed {self.n_passed}/{self.n_passed + self.n_failed}"
            + (f", errored {self.n_errored}" if self.n_errored else "")
        )


def load_suite(path: str | Path) -> tuple[SuiteCase, ...]:
    """Read a committed JSONL fixture suite."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"fixture suite not found: {resolved}")
    cases: list[SuiteCase] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(SuiteCase.model_validate(json.loads(line)))
    return tuple(cases)


async def run_suite(
    extractor: SCExtractor,
    cases: Sequence[SuiteCase],
    *,
    suite_name: str,
    confidence: float = 0.95,
) -> SuiteResult:
    """Run every case and score it against its expectation.

    An EXTRACTION_FAILED case is counted as ERRORED, never as a pass. An outage that produced
    no constraints is not evidence that the extractor correctly declined to extract.
    """
    outcomes: list[CaseOutcome] = []
    passed = failed = errored = 0

    for case in cases:
        call = await extractor.extract(case.user_turn)
        result = call.result
        if result.status is ExtractionStatus.EXTRACTION_FAILED:
            errored += 1
            outcomes.append(
                CaseOutcome(
                    case_id=case.id,
                    expected=case.expect_extraction,
                    observed=False,
                    status=result.status,
                    attack_class=case.attack_class,
                    passed=False,
                )
            )
            continue

        observed = len(result.extracted) > 0
        # A parse error is a failure of the extractor, not a correct empty output.
        case_passed = observed == case.expect_extraction and result.status is ExtractionStatus.OK
        if case_passed:
            passed += 1
        else:
            failed += 1
        outcomes.append(
            CaseOutcome(
                case_id=case.id,
                expected=case.expect_extraction,
                observed=observed,
                status=result.status,
                extracted_texts=tuple(sc.canonical_text for sc in result.extracted),
                attack_class=case.attack_class,
                passed=case_passed,
            )
        )

    scored = passed + failed
    return SuiteResult(
        suite=suite_name,
        n_cases=len(cases),
        n_passed=passed,
        n_failed=failed,
        n_errored=errored,
        outcomes=tuple(outcomes),
        wilson_ci=wilson_interval(passed, scored, confidence),
    )


class PrecisionRecallReport(BaseModel):
    """Both halves reported together. Recall alone is not a claim about this system."""

    model_config = ConfigDict(frozen=True)

    true_positives: int
    false_positives: int
    false_negatives: int
    precision_ci: WilsonInterval
    recall_ci: WilsonInterval

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            raise ValueError("precision is undefined: the extractor produced no positives")
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            raise ValueError("recall is undefined: the suite contains no positive cases")
        return self.true_positives / denominator

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    def format(self) -> str:
        return (
            f"precision {self.precision * 100:.1f}% {self.precision_ci.format_pp()}, "
            f"recall {self.recall * 100:.1f}% {self.recall_ci.format_pp()}, "
            f"F1 {self.f1 * 100:.1f}%"
        )


def precision_recall(
    positive: SuiteResult, negative: SuiteResult, *, confidence: float = 0.95
) -> PrecisionRecallReport:
    """Combine a positive suite (should extract) and a negative suite (should not).

    E-02. The negative suite is what turns the paper's recall number into a claim about a
    shippable system.
    """
    true_positives = positive.n_passed
    false_negatives = positive.n_failed
    false_positives = negative.n_failed
    return PrecisionRecallReport(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision_ci=wilson_interval(true_positives, true_positives + false_positives, confidence),
        recall_ci=wilson_interval(true_positives, true_positives + false_negatives, confidence),
    )
