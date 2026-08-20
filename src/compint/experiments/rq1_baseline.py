"""RQ1: to what extent do current compaction methods preserve SCs? TASK-018, spec 15.6.

The headline experiment. Grid: 50 contexts x 15 SCs = 750 instances per compaction condition
per dataset, at the default framing (preferential, direct) and the default injection (Top,
once).

Produces the Table 2 equivalent and the reproduction verdict. Nothing here decides whether the
reproduction succeeded: that judgment lives in `compint.report.tables.verify_reproduction`, so
the thresholds are applied by one function whatever produced the numbers.
"""

from __future__ import annotations

from collections.abc import Sequence

from compint.core.framing import FramingSpec
from compint.core.models import Explicitness, InjectionCondition, SCCategoryId, Strength
from compint.data.contexts import FillerContext
from compint.eval.metrics import (
    ComplianceResult,
    RetentionResult,
    aggregate_effect_retention,
    compliance_rate,
    retention_rate,
)
from compint.eval.records import ProbeRecord, RetentionRecord
from compint.experiments.base import GridCell, build_grid
from compint.experiments.runner import GridRunner, RunResult
from compint.report.tables import ResultCell, Table2
from shared.errors import EmptyEvaluationSetError


def build_rq1_grid(
    contexts_by_dataset: dict[str, Sequence[FillerContext]],
    sc_ids: Sequence[int],
    compactor_ids: Sequence[str],
    *,
    injection_seed: int,
    prompt_hashes: dict[str, str],
) -> tuple[GridCell, ...]:
    """The RQ1 grid: default framing, Top injection, every compactor on every dataset."""
    framing = FramingSpec(strength=Strength.PREFERENTIAL, explicitness=Explicitness.DIRECT)
    cells: list[GridCell] = []
    for _dataset, contexts in sorted(contexts_by_dataset.items()):
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


def build_table2(result: RunResult, *, split: str = "eval", run_id: str = "") -> Table2:
    """Aggregate a completed run into the Table 2 equivalent.

    A (dataset, compactor) pair whose every instance failed produces no cell rather than a
    zero: zero retention and no measurement are different claims, and only one of them is
    evidence about a compactor.
    """
    cells: list[ResultCell] = []
    for dataset, compactor_id in result.pairs():
        retention_records = result.retention_records(dataset, compactor_id)
        if not retention_records:
            continue
        try:
            retention = retention_rate(retention_records)
        except EmptyEvaluationSetError:
            # Every judgment was BLOCKED, UNPARSEABLE or on a failed compaction. Reporting a
            # rate here would invent a denominator that does not exist.
            continue

        probes = result.probe_records(dataset, compactor_id)
        by_condition: dict[str, list[ProbeRecord]] = {}
        for probe in probes:
            by_condition.setdefault(probe.condition, []).append(probe)
        compliance: dict[str, ComplianceResult] = {}
        for condition, rows in sorted(by_condition.items()):
            try:
                compliance[condition] = compliance_rate(rows)
            except EmptyEvaluationSetError:
                continue

        effect = None
        if {"lctx", "comp", "ub"} <= set(compliance):
            try:
                effect = aggregate_effect_retention(probes)
            except EmptyEvaluationSetError:
                effect = None

        degenerate = all(
            outcome.degenerate
            for outcome in result.outcomes
            if outcome.dataset == dataset and outcome.compactor_id == compactor_id
        )
        cells.append(
            ResultCell(
                dataset=dataset,
                compactor_id=compactor_id,
                retention=retention,
                compliance=compliance,
                effect_retention=effect,
                degenerate=degenerate,
            )
        )
    return Table2(cells=tuple(cells), split=split, run_id=run_id)


def per_category_retention(
    result: RunResult, dataset: str | None = None
) -> list[tuple[str, RetentionResult]]:
    """The Table 14 equivalent, grouped by SC category.

    Action is the hardest category and also the one whose loss is most severe. Reporting only
    the dataset average hides exactly that asymmetry.
    """
    grouped: dict[SCCategoryId, list[RetentionRecord]] = {}
    for record in result.retention_records(dataset):
        grouped.setdefault(record.category, []).append(record)

    rows: list[tuple[str, RetentionResult]] = []
    for category in SCCategoryId:
        records = grouped.get(category)
        if not records:
            continue
        try:
            rows.append((category.value, retention_rate(records)))
        except EmptyEvaluationSetError:
            continue
    return rows


async def run_rq1(
    runner: GridRunner,
    contexts_by_dataset: dict[str, Sequence[FillerContext]],
    sc_ids: Sequence[int],
    compactor_ids: Sequence[str],
    *,
    injection_seed: int,
    prompt_hashes: dict[str, str],
    split: str = "eval",
    run_id: str = "",
) -> tuple[RunResult, Table2]:
    """Build the grid, execute it, and aggregate."""
    cells = build_rq1_grid(
        contexts_by_dataset,
        sc_ids,
        compactor_ids,
        injection_seed=injection_seed,
        prompt_hashes=prompt_hashes,
    )
    all_contexts = [c for contexts in contexts_by_dataset.values() for c in contexts]
    result = await runner.run(cells, all_contexts)
    return result, build_table2(result, split=split, run_id=run_id)
