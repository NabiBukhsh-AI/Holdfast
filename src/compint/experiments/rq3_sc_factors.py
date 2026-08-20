"""RQ3: how do SC side factors affect retention? TASK-033 to TASK-037, spec 15.8.

Four sweeps over properties of the constraint itself rather than of the compaction system:

- **Injection location** (Top, Middle, Bottom, Multi). The expected result is a proximity
  gradient: constraints declared closer to the end of the history retain better, because the
  compaction instruction immediately follows the context (spec 14.4) and later positions sit
  nearer to it. OpenResearcher is excluded, not because it is uninteresting but because
  |U^t| = 1 makes all four conditions the same cell (FR-023).

- **Framing** (the full 2x2). The full grid is computed and stored, and the marginals are
  DERIVED from it. Spec 6.7 flags that the published framing table reports four columns for
  what the method section defines as a 2x2, and that the most coherent reading is two marginal
  sweeps. Storing the full grid makes both readings recoverable instead of committing to one.

- **Repetition** (r in 1 to 30). Expected: front loaded gains, converging by roughly 30.

- **SC type**: retention by category, which is where the Action asymmetry shows up.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from compint.core.framing import FramingSpec
from compint.core.models import Explicitness, InjectionCondition, SCCategoryId, Strength
from compint.data.contexts import FillerContext
from compint.eval.metrics import RetentionResult, retention_rate
from compint.eval.records import RetentionRecord
from compint.experiments.base import GridCell, build_grid
from compint.experiments.runner import GridRunner, RunResult
from shared.errors import EmptyEvaluationSetError

# Spec 6.6 and FR-023: with a single user turn every condition collapses onto index 0.
DEGENERATE_DATASETS = frozenset({"openresearcher"})

# PAPER SPECIFICATION FR-025, Figure 7.
REPETITION_SWEEP = (1, 5, 10, 15, 20, 25, 30)


class LocationPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    compactor_id: str
    condition: InjectionCondition
    retention: RetentionResult


class LocationSweep(BaseModel):
    """Figures 5 and 8 equivalent."""

    model_config = ConfigDict(frozen=True)

    points: tuple[LocationPoint, ...]
    excluded_datasets: tuple[str, ...] = ()

    def rate(self, dataset: str, compactor_id: str, condition: InjectionCondition) -> float | None:
        for point in self.points:
            if (
                point.dataset == dataset
                and point.compactor_id == compactor_id
                and point.condition is condition
            ):
                return point.retention.rate
        return None

    def proximity_gradient(self, dataset: str, compactor_id: str) -> tuple[float, float] | None:
        """(Middle minus Top, Bottom minus Middle) in percentage points.

        Spec 15.8 expects this gradient to reproduce in SIGN and ORDERING rather than in
        magnitude, so it is returned as deltas for the caller to check rather than compared to
        a stored constant here.
        """
        top = self.rate(dataset, compactor_id, InjectionCondition.TOP)
        middle = self.rate(dataset, compactor_id, InjectionCondition.MIDDLE)
        bottom = self.rate(dataset, compactor_id, InjectionCondition.BOTTOM)
        if top is None or middle is None or bottom is None:
            return None
        return ((middle - top) * 100.0, (bottom - middle) * 100.0)

    def gradient_holds(self, dataset: str, compactor_id: str) -> bool:
        """Later positions retain at least as well as earlier ones."""
        deltas = self.proximity_gradient(dataset, compactor_id)
        return deltas is not None and deltas[0] >= 0 and deltas[1] >= 0


def build_location_grid(
    contexts_by_dataset: dict[str, Sequence[FillerContext]],
    sc_ids: Sequence[int],
    compactor_ids: Sequence[str],
    *,
    framing: FramingSpec,
    injection_seed: int,
    prompt_hashes: dict[str, str],
    repetition_r: int,
    include_degenerate: bool = False,
) -> tuple[tuple[GridCell, ...], tuple[str, ...]]:
    """Build the location sweep, skipping datasets where the conditions collapse."""
    cells: list[GridCell] = []
    excluded: list[str] = []
    for dataset, contexts in sorted(contexts_by_dataset.items()):
        if dataset in DEGENERATE_DATASETS and not include_degenerate:
            excluded.append(dataset)
            continue
        for condition in InjectionCondition:
            cells.extend(
                build_grid(
                    contexts,
                    sc_ids,
                    compactor_ids,
                    framing=framing,
                    condition=condition,
                    injection_seed=injection_seed,
                    prompt_hashes=prompt_hashes,
                    repetition_r=repetition_r if condition is InjectionCondition.MULTI else None,
                )
            )
    return tuple(cells), tuple(excluded)


def summarize_locations(
    result: RunResult, condition_by_instance: dict[str, InjectionCondition]
) -> LocationSweep:
    buckets: dict[tuple[str, str, InjectionCondition], list[RetentionRecord]] = {}
    for outcome in result.outcomes:
        condition = condition_by_instance.get(outcome.instance_id)
        if condition is None or outcome.retention is None:
            continue
        buckets.setdefault((outcome.dataset, outcome.compactor_id, condition), []).append(
            outcome.retention
        )

    points: list[LocationPoint] = []
    for (dataset, compactor_id, condition), records in sorted(
        buckets.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value)
    ):
        try:
            points.append(
                LocationPoint(
                    dataset=dataset,
                    compactor_id=compactor_id,
                    condition=condition,
                    retention=retention_rate(records),
                )
            )
        except EmptyEvaluationSetError:
            continue
    return LocationSweep(points=tuple(points))


class FramingCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    strength: Strength
    explicitness: Explicitness
    retention: RetentionResult


class FramingGrid(BaseModel):
    """The full 2x2, with the published table's marginals derived from it."""

    model_config = ConfigDict(frozen=True)

    cells: tuple[FramingCell, ...]

    def cell(self, strength: Strength, explicitness: Explicitness) -> FramingCell | None:
        for entry in self.cells:
            if entry.strength is strength and entry.explicitness is explicitness:
                return entry
        return None

    def strength_marginal(self, explicitness: Explicitness) -> dict[str, float]:
        """Strength swept with explicitness held fixed."""
        return {
            strength.value: cell.retention.rate
            for strength in Strength
            if (cell := self.cell(strength, explicitness)) is not None
        }

    def explicitness_marginal(self, strength: Strength) -> dict[str, float]:
        """Explicitness swept with strength held fixed."""
        return {
            explicitness.value: cell.retention.rate
            for explicitness in Explicitness
            if (cell := self.cell(strength, explicitness)) is not None
        }

    def is_complete(self) -> bool:
        return len(self.cells) == 4


def build_framing_grid(
    contexts: Sequence[FillerContext],
    sc_ids: Sequence[int],
    compactor_ids: Sequence[str],
    *,
    injection_seed: int,
    prompt_hashes: dict[str, str],
) -> tuple[GridCell, ...]:
    """All four framings. The marginals are derived later, never measured separately."""
    cells: list[GridCell] = []
    for strength in Strength:
        for explicitness in Explicitness:
            cells.extend(
                build_grid(
                    contexts,
                    sc_ids,
                    compactor_ids,
                    framing=FramingSpec(strength=strength, explicitness=explicitness),
                    condition=InjectionCondition.TOP,
                    injection_seed=injection_seed,
                    prompt_hashes=prompt_hashes,
                )
            )
    return tuple(cells)


def summarize_framings(
    result: RunResult, framing_by_instance: dict[str, FramingSpec]
) -> FramingGrid:
    buckets: dict[tuple[Strength, Explicitness], list[RetentionRecord]] = {}
    for outcome in result.outcomes:
        framing = framing_by_instance.get(outcome.instance_id)
        if framing is None or outcome.retention is None:
            continue
        buckets.setdefault((framing.strength, framing.explicitness), []).append(outcome.retention)

    cells: list[FramingCell] = []
    for (strength, explicitness), records in sorted(
        buckets.items(), key=lambda item: (item[0][0].value, item[0][1].value)
    ):
        try:
            cells.append(
                FramingCell(
                    strength=strength,
                    explicitness=explicitness,
                    retention=retention_rate(records),
                )
            )
        except EmptyEvaluationSetError:
            continue
    return FramingGrid(cells=tuple(cells))


class RepetitionPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    r: int
    retention: RetentionResult


class RepetitionSweep(BaseModel):
    model_config = ConfigDict(frozen=True)

    points: tuple[RepetitionPoint, ...]

    def is_front_loaded(self) -> bool:
        """Gains concentrate at low r and flatten by the top of the sweep.

        Checked as a shape rather than as a threshold: the first half of the sweep should
        deliver more of the total gain than the second half.
        """
        ordered = sorted(self.points, key=lambda p: p.r)
        if len(ordered) < 3:
            return False
        total = ordered[-1].retention.rate - ordered[0].retention.rate
        if total <= 0:
            return False
        midpoint = ordered[len(ordered) // 2]
        early = midpoint.retention.rate - ordered[0].retention.rate
        return early >= total / 2


def summarize_repetition(result: RunResult, r_by_instance: dict[str, int]) -> RepetitionSweep:
    buckets: dict[int, list[RetentionRecord]] = {}
    for outcome in result.outcomes:
        r = r_by_instance.get(outcome.instance_id)
        if r is None or outcome.retention is None:
            continue
        buckets.setdefault(r, []).append(outcome.retention)

    points: list[RepetitionPoint] = []
    for r, records in sorted(buckets.items()):
        try:
            points.append(RepetitionPoint(r=r, retention=retention_rate(records)))
        except EmptyEvaluationSetError:
            continue
    return RepetitionSweep(points=tuple(points))


def summarize_by_category(
    result: RunResult, dataset: str | None = None
) -> dict[SCCategoryId, RetentionResult]:
    """Figures 6 and 9 equivalent: retention by SC category."""
    grouped: dict[SCCategoryId, list[RetentionRecord]] = {}
    for record in result.retention_records(dataset):
        grouped.setdefault(record.category, []).append(record)

    summary: dict[SCCategoryId, RetentionResult] = {}
    for category, records in grouped.items():
        try:
            summary[category] = retention_rate(records)
        except EmptyEvaluationSetError:
            continue
    return summary


async def run_location_sweep(
    runner: GridRunner,
    contexts_by_dataset: dict[str, Sequence[FillerContext]],
    sc_ids: Sequence[int],
    compactor_ids: Sequence[str],
    *,
    framing: FramingSpec,
    injection_seed: int,
    prompt_hashes: dict[str, str],
    repetition_r: int,
) -> tuple[RunResult, LocationSweep]:
    cells, excluded = build_location_grid(
        contexts_by_dataset,
        sc_ids,
        compactor_ids,
        framing=framing,
        injection_seed=injection_seed,
        prompt_hashes=prompt_hashes,
        repetition_r=repetition_r,
    )
    condition_by_instance = {cell.instance_id: cell.condition for cell in cells}
    contexts = [c for contexts in contexts_by_dataset.values() for c in contexts]
    result = await runner.run(cells, contexts, repetition_r=repetition_r)
    sweep = summarize_locations(result, condition_by_instance)
    return result, sweep.model_copy(update={"excluded_datasets": excluded})
