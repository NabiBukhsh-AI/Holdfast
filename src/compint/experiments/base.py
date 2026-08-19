"""Experiment runner: grid, checkpointing, and the cost gate. TASK-017.

Three requirements shape this module.

1. **Resumability.** Spec 17.4: a 90 GPU hour run that dies at hour 80 with no resumability is
   a real and expensive failure. Every instance carries an idempotency key
   (spec 12.2 stage 4 to 6) and finished instances are skipped on restart.

2. **The cost gate.** Execution contract rule 15: do not spend money or GPU time without an
   explicit gate. The source research spent roughly 800 USD on one 220K experiment. A run
   refuses to start above the configured ceiling without an explicit confirmation.

3. **Manifest first.** The manifest is written BEFORE the first model call, so a run that dies
   halfway still records what it was and which UNKNOWN values it used.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from compint.core.framing import FramingSpec
from compint.core.models import InjectionCondition
from compint.data.contexts import FillerContext
from compint.eval.records import InstanceKey
from compint.report.manifest import RunManifest
from shared.config import AppConfig
from shared.errors import CostCeilingExceededError


class GridCell(BaseModel):
    """One unit of work: a context, an SC, a framing, an injection, and a compactor."""

    model_config = ConfigDict(frozen=True)

    key: InstanceKey
    context_id: str
    dataset: str
    sc_id: int
    framing: FramingSpec
    condition: InjectionCondition
    compactor_id: str
    degenerate: bool = False

    @property
    def instance_id(self) -> str:
        return self.key.instance_id()


def build_grid(
    contexts: Sequence[FillerContext],
    sc_ids: Sequence[int],
    compactor_ids: Sequence[str],
    *,
    framing: FramingSpec,
    condition: InjectionCondition,
    injection_seed: int,
    prompt_hashes: dict[str, str],
    repetition_r: int | None = None,
) -> tuple[GridCell, ...]:
    """Cartesian product in a deterministic order.

    Grid cardinality for the main experiments is 50 contexts x 15 SCs = 750 per compaction
    condition per dataset (spec 12.2 stage 4 to 6).
    """
    cells: list[GridCell] = []
    for context in contexts:
        for sc_id in sc_ids:
            for compactor_id in compactor_ids:
                key = InstanceKey(
                    context_id=context.context_id,
                    sc_id=sc_id,
                    strength=framing.strength.value,
                    explicitness=framing.explicitness.value,
                    injection_condition=condition,
                    injection_seed=injection_seed,
                    compactor_id=compactor_id,
                    prompt_hash=prompt_hashes.get(compactor_id, "none"),
                    repetition_r=repetition_r,
                    target_tokens=context.target_tokens,
                )
                cells.append(
                    GridCell(
                        key=key,
                        context_id=context.context_id,
                        dataset=context.dataset,
                        sc_id=sc_id,
                        framing=framing,
                        condition=condition,
                        compactor_id=compactor_id,
                        degenerate=context.is_degenerate_for_injection,
                    )
                )
    return tuple(cells)


class CostEstimate(BaseModel):
    """Pre-flight projection. Spec 12.2: the instance table is where budget is enforced."""

    model_config = ConfigDict(frozen=True)

    n_instances: int
    compaction_calls: int
    judge_calls: int
    probe_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_usd: float
    assumptions: dict[str, float | int | str] = Field(default_factory=dict)

    def format(self) -> str:
        return (
            f"{self.n_instances} instances: {self.compaction_calls} compactions, "
            f"{self.judge_calls} judge calls, {self.probe_calls} probes, "
            f"approximately {self.estimated_usd:.2f} USD"
        )


def estimate_cost(
    cells: Sequence[GridCell],
    config: AppConfig,
    *,
    n_contexts: int,
    conditions: Sequence[str] = ("lctx", "lctx_sc", "comp", "ub"),
    mean_context_tokens: int | None = None,
) -> CostEstimate:
    """Project the call counts and spend for a grid, accounting for the condition caching.

    The caching is not a rounding detail: K_lctx and C(H^t) are computed once per context
    rather than once per (context, SC), which removes 14 of every 15 calls in those two arms.
    An estimate that ignored it would overstate cost by roughly the SC count.
    """
    n_instances = len(cells)
    n_scs = len({cell.sc_id for cell in cells}) or 1
    n_compactors = len({cell.compactor_id for cell in cells}) or 1
    context_tokens = mean_context_tokens or config.context.target_tokens

    # One compaction per instance for K_comp, plus one un-injected compaction per
    # (context, compactor) for K_ub.
    compaction_calls = 0
    if "comp" in conditions:
        compaction_calls += n_instances
    if "ub" in conditions:
        compaction_calls += n_contexts * n_compactors

    judge_calls = n_instances  # one retention judgment per instance

    probe_calls = 0
    for condition in conditions:
        if condition in ("lctx",):
            probe_calls += n_contexts  # constant across the SCs
        elif condition == "ub":
            probe_calls += n_instances
        else:
            probe_calls += n_instances

    summary_tokens = 600  # Table 11 band midpoint, used for output side estimates.
    estimated_input = (
        compaction_calls * context_tokens
        + judge_calls * summary_tokens
        + probe_calls * context_tokens // 2
    )
    estimated_output = compaction_calls * summary_tokens + judge_calls * 2 + probe_calls * 2

    input_prices = config.cost.price_per_1k_input_usd
    output_prices = config.cost.price_per_1k_output_usd
    # UNKNOWN: per model prices are deployment specific and are not in the paper. With no
    # price table configured the projection reports zero dollars and says so, rather than
    # inventing a rate that would make the gate meaningless.
    mean_input_price = (
        sum(input_prices.values()) / len(input_prices) if input_prices else 0.0
    )
    mean_output_price = (
        sum(output_prices.values()) / len(output_prices) if output_prices else 0.0
    )
    estimated_usd = (
        estimated_input / 1000 * mean_input_price + estimated_output / 1000 * mean_output_price
    )

    return CostEstimate(
        n_instances=n_instances,
        compaction_calls=compaction_calls,
        judge_calls=judge_calls,
        probe_calls=probe_calls,
        estimated_input_tokens=estimated_input,
        estimated_output_tokens=estimated_output,
        estimated_usd=estimated_usd,
        assumptions={
            "mean_context_tokens": context_tokens,
            "assumed_summary_tokens": summary_tokens,
            "n_scs": n_scs,
            "n_contexts": n_contexts,
            "n_compactors": n_compactors,
            "price_table": "configured" if input_prices else "EMPTY, so USD is reported as 0",
        },
    )


def enforce_cost_gate(estimate: CostEstimate, config: AppConfig, *, confirm: bool) -> None:
    """Refuse to start above the ceiling without an explicit confirmation."""
    if estimate.estimated_usd > config.cost.ceiling_usd and not confirm:
        raise CostCeilingExceededError(
            f"projected cost {estimate.estimated_usd:.2f} USD exceeds the ceiling "
            f"{config.cost.ceiling_usd:.2f} USD. Re-run with --confirm to proceed. "
            f"({estimate.format()})"
        )
    if config.cost.require_confirm and not confirm:
        raise CostCeilingExceededError(
            "this run spends money or GPU time and cost.require_confirm is set. "
            f"Re-run with --confirm. ({estimate.format()})"
        )


class Checkpoint:
    """Append only ledger of completed instance ids.

    JSONL rather than a database so that a run on a laptop, in CI, and on a cluster all
    resume the same way, and so that a partially written line from a hard kill costs one
    instance rather than the whole ledger.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._done: set[str] = set()
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line from a hard kill. Skip it and rerun that instance,
                        # which is cheaper and safer than trusting a truncated record.
                        continue
                    instance_id = record.get("instance_id")
                    if isinstance(instance_id, str):
                        self._done.add(instance_id)

    def __contains__(self, instance_id: str) -> bool:
        return instance_id in self._done

    def __len__(self) -> int:
        return len(self._done)

    @property
    def completed_ids(self) -> frozenset[str]:
        return frozenset(self._done)

    def mark(self, instance_id: str, **fields: object) -> None:
        self._done.add(instance_id)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "instance_id": instance_id,
            "at": datetime.now(UTC).isoformat(),
            **fields,
        }
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload) + "\n")

    def pending(self, cells: Sequence[GridCell]) -> Iterator[GridCell]:
        """Yield only the cells this run still has to do."""
        for cell in cells:
            if cell.instance_id not in self._done:
                yield cell


class Experiment(Protocol):
    """Every experiment suite implements this. Spec 20 lists them under experiments/."""

    name: str

    async def run(self, confirm: bool) -> RunManifest: ...
