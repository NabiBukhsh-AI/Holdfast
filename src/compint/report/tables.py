"""Result tables and the reproduction verdict. TASK-018, spec 15.10, 24.1.

Two responsibilities:

1. Render the Table 2 equivalent: retention, the four compliance rates, and Effect Retention
   per (dataset, compactor), every one of them carrying its denominator and interval (INV-6).

2. Decide whether a reproduction SUCCEEDED, against the acceptance thresholds in spec 15.10.
   The verdict is computed, not asserted: a run that misses on more than 15 percent of cells is
   reported as a finding rather than quietly rounded into agreement.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from compint.eval.metrics import (
    ComplianceResult,
    EffectRetentionResult,
    ERStatus,
    RetentionResult,
)

# PAPER SPECIFICATION Table 2, as published. Used to grade a reproduction, never to shape it.
# Values are retention percentages by (dataset, compactor).
PAPER_TABLE2_RETENTION: dict[tuple[str, str], float] = {
    ("hermes_agent", "recent_5"): 0.4,
    ("wildchat", "recent_5"): 0.0,
    ("openresearcher", "recent_5"): 0.0,
    ("hermes_agent", "llmlingua2_t500"): 0.0,
    ("wildchat", "llmlingua2_t500"): 0.1,
    ("openresearcher", "llmlingua2_t500"): 0.0,
    ("hermes_agent", "gpt_oss_120b__pi_mono"): 36.3,
    ("wildchat", "gpt_oss_120b__pi_mono"): 6.7,
}

# ENGINEERING RECOMMENDATION spec 15.10. The paper reports no uncertainty, so the reproduction
# defines its own bands.
FLOOR_CELL_THRESHOLD_PCT = 1.0
FLOOR_TOLERANCE_PP = 2.0
RELATIVE_TOLERANCE = 0.25
ABSOLUTE_TOLERANCE_PP = 5.0
COMPLIANCE_TOLERANCE_PP = 5.0
ER_VS_PAPER_TOLERANCE_PP = 8.0
EXTRACTOR_TOLERANCE_PP = 5.0
EXTRACTOR_FLOOR_PCT = 85.0
NON_LLM_CEILING_PCT = 2.0
MAX_OUT_OF_TOLERANCE_FRACTION = 0.15


class CellVerdict(StrEnum):
    WITHIN_TOLERANCE = "WITHIN_TOLERANCE"
    OUT_OF_TOLERANCE = "OUT_OF_TOLERANCE"
    NO_REFERENCE = "NO_REFERENCE"
    DEGENERATE = "DEGENERATE"


class ResultCell(BaseModel):
    """One (dataset, compactor) row of the Table 2 equivalent."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    compactor_id: str
    retention: RetentionResult
    compliance: dict[str, ComplianceResult] = Field(default_factory=dict)
    effect_retention: EffectRetentionResult | None = None
    degenerate: bool = False

    @property
    def retention_percent(self) -> float:
        return self.retention.percent

    def reference_retention(self) -> float | None:
        return PAPER_TABLE2_RETENTION.get((self.dataset, self.compactor_id))

    def verdict(self) -> CellVerdict:
        """Grade this cell against spec 15.10."""
        if self.degenerate:
            return CellVerdict.DEGENERATE
        expected = self.reference_retention()
        if expected is None:
            return CellVerdict.NO_REFERENCE
        observed = self.retention_percent
        if expected <= FLOOR_CELL_THRESHOLD_PCT:
            # Floor effects: at N=750 the Wilson lower bound is essentially zero, so an
            # absolute band is the only meaningful comparison.
            within = abs(observed - expected) <= FLOOR_TOLERANCE_PP
        else:
            relative_ok = abs(observed - expected) <= expected * RELATIVE_TOLERANCE
            absolute_ok = abs(observed - expected) <= ABSOLUTE_TOLERANCE_PP
            within = relative_ok or absolute_ok
        return CellVerdict.WITHIN_TOLERANCE if within else CellVerdict.OUT_OF_TOLERANCE

    def format_row(self) -> str:
        er = self.effect_retention.format() if self.effect_retention else "n/a"
        marker = " [DEGENERATE]" if self.degenerate else ""
        return (
            f"{self.dataset:<16} {self.compactor_id:<32} "
            f"retention {self.retention.format():<34} ER {er}{marker}"
        )


class Table2(BaseModel):
    """The headline table plus its verdict."""

    model_config = ConfigDict(frozen=True)

    cells: tuple[ResultCell, ...]
    split: str = "eval"
    run_id: str = ""

    def out_of_tolerance(self) -> tuple[ResultCell, ...]:
        return tuple(c for c in self.cells if c.verdict() is CellVerdict.OUT_OF_TOLERANCE)

    def graded_cells(self) -> tuple[ResultCell, ...]:
        """Cells that have a published reference to grade against."""
        return tuple(
            c
            for c in self.cells
            if c.verdict() in (CellVerdict.WITHIN_TOLERANCE, CellVerdict.OUT_OF_TOLERANCE)
        )

    def mean_retention_percent(self) -> float:
        """The paper's headline: 17 percent mean retention across tested compactors."""
        if not self.cells:
            raise ValueError("cannot average over an empty table")
        return sum(c.retention_percent for c in self.cells) / len(self.cells)

    def render(self) -> str:
        lines = [f"Table 2 equivalent ({self.split} split, run {self.run_id or 'unnamed'})", ""]
        lines.extend(cell.format_row() for cell in self.cells)
        lines.append("")
        lines.append(f"mean retention across cells: {self.mean_retention_percent():.1f}%")
        return "\n".join(lines)


class QualitativeOrdering(BaseModel):
    """An ordering claim. Spec 15.10: these must match EXACTLY.

    The orderings are the paper's actual argument. A reproduction that matches every number
    within tolerance but inverts an ordering has not reproduced the finding.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    holds: bool
    detail: str = ""


def check_orderings(table: Table2) -> tuple[QualitativeOrdering, ...]:
    """Evaluate the ordering claims the reproduction must preserve."""
    by_key = {(c.dataset, c.compactor_id): c for c in table.cells}
    orderings: list[QualitativeOrdering] = []

    non_llm = [c for c in table.cells if c.compactor_id in ("recent_5", "llmlingua2_t500")]
    if non_llm:
        worst = max(c.retention_percent for c in non_llm)
        orderings.append(
            QualitativeOrdering(
                name="non_llm_compactors_near_zero",
                description=(
                    "Truncation and extractive compression fail STRUCTURALLY, not for want of "
                    "prompt engineering, so both sit below 2 percent everywhere"
                ),
                holds=worst <= NON_LLM_CEILING_PCT,
                detail=f"highest non LLM retention observed: {worst:.1f}%",
            )
        )

    llm_cells = [c for c in table.cells if c.compactor_id not in ("recent_5", "llmlingua2_t500")]
    if llm_cells and non_llm:
        best_llm = max(c.retention_percent for c in llm_cells)
        best_non_llm = max(c.retention_percent for c in non_llm)
        orderings.append(
            QualitativeOrdering(
                name="llm_compactors_beat_non_llm",
                description="Prompt based summarization retains more than truncation",
                holds=best_llm > best_non_llm,
                detail=f"best LLM {best_llm:.1f}% against best non LLM {best_non_llm:.1f}%",
            )
        )

    hermes = by_key.get(("hermes_agent", "gpt_oss_120b__pi_mono"))
    wildchat = by_key.get(("wildchat", "gpt_oss_120b__pi_mono"))
    if hermes and wildchat:
        orderings.append(
            QualitativeOrdering(
                name="pi_mono_hermes_above_wildchat",
                description=(
                    "The same compactor retains more on Hermes Agent than on WildChat, where "
                    "user turns are far denser and the compression ratio is harsher"
                ),
                holds=hermes.retention_percent > wildchat.retention_percent,
                detail=(
                    f"Hermes {hermes.retention_percent:.1f}% against "
                    f"WildChat {wildchat.retention_percent:.1f}%"
                ),
            )
        )

    return tuple(orderings)


class ReproductionVerdict(BaseModel):
    """The spec 15.10 decision, computed rather than asserted."""

    model_config = ConfigDict(frozen=True)

    succeeded: bool
    n_cells: int
    n_graded: int
    n_out_of_tolerance: int
    out_of_tolerance_fraction: float
    orderings: tuple[QualitativeOrdering, ...]
    extractor_retention: dict[str, float] = Field(default_factory=dict)
    failures: tuple[str, ...] = ()

    def render(self) -> str:
        status = "PASS" if self.succeeded else "FAIL"
        lines = [
            f"Reproduction verdict: {status}",
            f"  cells graded: {self.n_graded} of {self.n_cells}",
            f"  out of tolerance: {self.n_out_of_tolerance} "
            f"({self.out_of_tolerance_fraction * 100:.1f}%)",
        ]
        for ordering in self.orderings:
            mark = "ok" if ordering.holds else "FAILED"
            lines.append(f"  ordering {ordering.name}: {mark} ({ordering.detail})")
        for dataset, value in sorted(self.extractor_retention.items()):
            lines.append(f"  extractor {dataset}: {value:.1f}%")
        for failure in self.failures:
            lines.append(f"  FAILURE: {failure}")
        if not self.succeeded:
            lines.append(
                "  Missing thresholds on more than 15 percent of cells is a FINDING, not a bug "
                "to grind through (spec 32.4). Every out of tolerance cell must be explained "
                "individually in DEVIATIONS.md."
            )
        return "\n".join(lines)


def verify_reproduction(
    table: Table2,
    *,
    extractor_retention: dict[str, float] | None = None,
) -> ReproductionVerdict:
    """Spec 15.10 acceptance decision.

    A reproduction is declared successful when all qualitative orderings hold, the extractor
    exceeds 85 percent on all three datasets, non LLM compactors are below 2 percent
    everywhere, and no more than 15 percent of quantitative cells fall outside tolerance.
    """
    graded = table.graded_cells()
    out = table.out_of_tolerance()
    fraction = len(out) / len(graded) if graded else 0.0
    orderings = check_orderings(table)
    extractor = dict(extractor_retention or {})

    failures: list[str] = []
    if fraction > MAX_OUT_OF_TOLERANCE_FRACTION:
        failures.append(
            f"{fraction * 100:.1f}% of graded cells are out of tolerance, above the "
            f"{MAX_OUT_OF_TOLERANCE_FRACTION * 100:.0f}% limit"
        )
    for ordering in orderings:
        if not ordering.holds:
            failures.append(
                f"qualitative ordering {ordering.name} does not hold: {ordering.detail}"
            )
    for dataset, value in extractor.items():
        if value < EXTRACTOR_FLOOR_PCT:
            failures.append(
                f"extractor retention on {dataset} is {value:.1f}%, below the "
                f"{EXTRACTOR_FLOOR_PCT:.0f}% floor; the mitigation claim is the load bearing one"
            )
    if table.split != "eval":
        failures.append(
            f"results come from the {table.split} split, which is never reportable as a "
            "headline number (spec 12.2 stage 8)"
        )

    return ReproductionVerdict(
        succeeded=not failures,
        n_cells=len(table.cells),
        n_graded=len(graded),
        n_out_of_tolerance=len(out),
        out_of_tolerance_fraction=fraction,
        orderings=orderings,
        extractor_retention=extractor,
        failures=tuple(failures),
    )


def er_matches_paper(observed: EffectRetentionResult, expected_percent: float) -> bool:
    """Spec 15.10: ER against the paper is compared within 8 points, since it compounds four
    noisy inputs."""
    if observed.value is None or observed.status is ERStatus.DEGENERATE_DENOMINATOR:
        return False
    assert observed.percent is not None
    return abs(observed.percent - expected_percent) <= ER_VS_PAPER_TOLERANCE_PP


def compliance_matches_paper(observed: ComplianceResult, expected_percent: float) -> bool:
    """Spec 15.10: compliance rates within 5 percentage points absolute."""
    return abs(observed.percent - expected_percent) <= COMPLIANCE_TOLERANCE_PP


def render_per_category(
    rows: Sequence[tuple[str, RetentionResult]], title: str = "Per category retention"
) -> str:
    """The Table 14 equivalent: retention by SC category.

    Action is the hardest category and also the one whose loss is most severe, which is the
    asymmetry the category weighted alerting exists for.
    """
    lines = [title, ""]
    for category, result in rows:
        lines.append(f"  {category:<14} {result.format()}")
    return "\n".join(lines)
