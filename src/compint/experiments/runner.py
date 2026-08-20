"""Grid execution. TASK-017, TASK-018.

One place where a grid cell actually becomes numbers: inject, compact, judge, probe, record.
Every experiment suite in this package is a thin layer that decides WHICH cells to build and
how to group the results; none of them re-implements the per cell work, because a second copy
of this loop would be a second place for the condition caching to be got wrong.

Cost control is structural rather than advisory. Cells are executed grouped by
(context, compactor) so that `ContextCache` can hold K_lctx and the un-injected C(H^t) across
all 15 SCs sharing that pair. Running the same cells in a different order would produce the
same numbers at roughly fifteen times the price.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from compint.compactors.base import Compactor
from compint.core.catalog import SCCatalog
from compint.core.framing import frame
from compint.core.injection import inject, injection_locations
from compint.core.models import CompactionStatus, InjectionCondition
from compint.core.random_source import RandomSource
from compint.data.contexts import FillerContext
from compint.eval.compliance import ComplianceHarness, ContextCache
from compint.eval.records import ComplianceCondition, ProbeRecord, RetentionRecord
from compint.eval.retention_judge import RetentionJudge
from compint.experiments.base import Checkpoint, GridCell
from shared.errors import ConfigError

logger = logging.getLogger(__name__)


class InstanceOutcome(BaseModel):
    """Everything one grid cell produced, including how it failed."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    dataset: str
    compactor_id: str
    sc_id: int
    degenerate: bool
    compaction_status: CompactionStatus
    compacted_tokens: int = 0
    input_tokens: int = 0
    retention: RetentionRecord | None = None
    probes: tuple[ProbeRecord, ...] = ()


class RunResult(BaseModel):
    """Aggregate of a whole grid, plus the call counts that prove the caching worked."""

    model_config = ConfigDict(frozen=True)

    outcomes: tuple[InstanceOutcome, ...] = ()
    skipped_resumed: int = 0
    compaction_calls: int = 0
    lctx_builds: int = 0
    judge_calls: int = 0
    probe_calls: int = 0
    degenerate_cells: int = 0

    def retention_records(
        self, dataset: str | None = None, compactor_id: str | None = None
    ) -> tuple[RetentionRecord, ...]:
        return tuple(
            outcome.retention
            for outcome in self.outcomes
            if outcome.retention is not None
            and (dataset is None or outcome.dataset == dataset)
            and (compactor_id is None or outcome.compactor_id == compactor_id)
        )

    def probe_records(
        self, dataset: str | None = None, compactor_id: str | None = None
    ) -> tuple[ProbeRecord, ...]:
        return tuple(
            probe
            for outcome in self.outcomes
            for probe in outcome.probes
            if (dataset is None or outcome.dataset == dataset)
            and (compactor_id is None or outcome.compactor_id == compactor_id)
        )

    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Every (dataset, compactor) pair present, in a stable order."""
        return tuple(sorted({(o.dataset, o.compactor_id) for o in self.outcomes}))

    def compaction_ratios(self) -> tuple[float, ...]:
        """Spec 6.14, for the Table 11 equivalent."""
        return tuple(
            outcome.input_tokens / outcome.compacted_tokens
            for outcome in self.outcomes
            if outcome.compacted_tokens > 0
        )


class GridRunner:
    """Executes cells against a set of compactors, a judge, and a probe harness."""

    def __init__(
        self,
        catalog: SCCatalog,
        compactors: dict[str, Compactor],
        judge: RetentionJudge,
        compliance: ComplianceHarness,
        *,
        rng: RandomSource,
        conditions: Sequence[ComplianceCondition] = ("lctx", "lctx_sc", "comp", "ub"),
        separator: str = " ",
        direction: str = "append",
        checkpoint: Checkpoint | None = None,
    ) -> None:
        self._catalog = catalog
        self._compactors = compactors
        self._judge = judge
        self._compliance = compliance
        self._rng = rng
        self._conditions = tuple(conditions)
        self._separator = separator
        self._direction = direction
        self._checkpoint = checkpoint

    async def run(
        self,
        cells: Sequence[GridCell],
        contexts: Sequence[FillerContext],
        *,
        repetition_r: int | None = None,
    ) -> RunResult:
        """Execute a grid, grouped so the per context caching actually applies."""
        by_id = {context.context_id: context for context in contexts}
        missing = {cell.context_id for cell in cells} - set(by_id)
        if missing:
            raise ConfigError(f"grid references contexts that were not supplied: {sorted(missing)}")

        # Group by (context, compactor): that is exactly the scope over which K_lctx and the
        # un-injected compaction are constant.
        groups: dict[tuple[str, str], list[GridCell]] = {}
        for cell in cells:
            groups.setdefault((cell.context_id, cell.compactor_id), []).append(cell)

        outcomes: list[InstanceOutcome] = []
        skipped = 0
        compaction_calls = 0
        lctx_builds = 0
        judge_calls = 0
        probe_calls = 0

        for (context_id, compactor_id), group in sorted(groups.items()):
            context = by_id[context_id]
            compactor = self._compactors.get(compactor_id)
            if compactor is None:
                raise ConfigError(f"no compactor registered for id {compactor_id}")
            cache = ContextCache()

            for cell in group:
                if self._checkpoint is not None and cell.instance_id in self._checkpoint:
                    skipped += 1
                    continue
                outcome = await self._run_cell(cell, context, compactor, cache, repetition_r)
                outcomes.append(outcome)
                judge_calls += 1 if outcome.retention is not None else 0
                probe_calls += len(outcome.probes)
                if self._checkpoint is not None:
                    self._checkpoint.mark(
                        cell.instance_id,
                        dataset=cell.dataset,
                        compactor_id=cell.compactor_id,
                        sc_id=cell.sc_id,
                        status=outcome.compaction_status.value,
                    )

            compaction_calls += cache.compaction_calls
            lctx_builds += cache.lctx_builds

        return RunResult(
            outcomes=tuple(outcomes),
            skipped_resumed=skipped,
            compaction_calls=compaction_calls,
            lctx_builds=lctx_builds,
            judge_calls=judge_calls,
            probe_calls=probe_calls,
            degenerate_cells=sum(1 for outcome in outcomes if outcome.degenerate),
        )

    async def _run_cell(
        self,
        cell: GridCell,
        context: FillerContext,
        compactor: Compactor,
        cache: ContextCache,
        repetition_r: int | None,
    ) -> InstanceOutcome:
        framed = frame(
            self._catalog.by_id(cell.sc_id), cell.framing.strength, cell.framing.explicitness
        )
        history = context.history

        locations = injection_locations(
            cell.condition,
            history.n_user_turns,
            self._rng.derive(f"{cell.instance_id}:injection"),
            r=repetition_r,
        )
        injected = inject(
            history,
            framed,
            locations,
            condition=cell.condition,
            separator=self._separator,
            direction=self._direction,
            repetition_r=repetition_r,
        )

        # K_comp: the setting under test. Compact the INJECTED history.
        compacted = await compactor.compact(injected.history)
        cache.compaction_calls += 1

        # Judge retention on the compacted context. INV-4 is enforced by the parameter type.
        retention = await self._judge.judge(
            framed, compacted, instance_id=cell.instance_id, degenerate=injected.degenerate
        )

        probes: list[ProbeRecord] = []
        for condition in self._conditions:
            if condition == "comp":
                # Reuse the compaction just performed rather than paying for it twice.
                if compacted.status is not CompactionStatus.OK:
                    probes.append(
                        self._compliance.failure_record(
                            condition,
                            framed,
                            instance_id=cell.instance_id,
                            status=self._compliance.probe_status_for(compacted.status),
                            detail="compaction did not produce a probeable context",
                        )
                    )
                    continue
                probes.append(
                    await self._compliance.probe(
                        condition,
                        compacted.text,
                        framed,
                        instance_id=cell.instance_id,
                        context_tokens=compacted.output_tokens,
                    )
                )
                continue

            built = await self._compliance.run_conditions(
                (condition,),
                history=history,
                injected=injected,
                framed_sc=framed,
                compactor=compactor,
                cache=cache,
                instance_id=cell.instance_id,
            )
            probes.extend(built)

        if injected.degenerate:
            logger.warning(
                "degenerate_cell",
                extra={"instance_id": cell.instance_id, "dataset": cell.dataset},
            )

        return InstanceOutcome(
            instance_id=cell.instance_id,
            dataset=cell.dataset,
            compactor_id=cell.compactor_id,
            sc_id=cell.sc_id,
            degenerate=injected.degenerate,
            compaction_status=compacted.status,
            compacted_tokens=compacted.output_tokens,
            input_tokens=compacted.input_tokens,
            retention=retention,
            probes=tuple(probes),
        )


class DatasetGroup(BaseModel):
    """One dataset's contexts and the cells built over them."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dataset: str
    contexts: tuple[FillerContext, ...]
    cells: tuple[GridCell, ...] = Field(default_factory=tuple)


def default_conditions() -> tuple[ComplianceCondition, ...]:
    return ("lctx", "lctx_sc", "comp", "ub")


def default_injection_condition() -> InjectionCondition:
    """FR-024: Top, once, for the main experiments."""
    return InjectionCondition.TOP
