"""End to end experiment suite tests. TASK-018, TASK-020, TASK-031 to TASK-039.

These run COMPLETE grids against deterministic stubs. That proves the orchestration, the
condition caching, the aggregation and the acceptance logic all work together, which is
everything except the model outputs themselves. When the fetch gate opens, the only thing that
changes is which client is wired in.

The stub judge is driven by a rule the tests control (which SCs "survive"), so the aggregated
retention rates are known in advance and the aggregation can be checked against them rather
than merely observed.
"""

from __future__ import annotations

import pytest

from compint.compactors.recent_n import RecentNCompactor
from compint.core.catalog import SCCatalog
from compint.core.framing import FramingSpec
from compint.core.models import (
    Explicitness,
    History,
    InjectionCondition,
    Message,
    Role,
    SCCategoryId,
    Strength,
)
from compint.core.random_source import RandomSource
from compint.core.tokenization import HeuristicTokenizer
from compint.data.contexts import ContextStatus, FillerContext
from compint.eval.compliance import ComplianceHarness
from compint.eval.retention_judge import RetentionJudge
from compint.experiments.ablations import (
    summarize_matching_ablation,
    summarize_prompt_ablation,
)
from compint.experiments.base import Checkpoint
from compint.experiments.rq1_baseline import build_table2, per_category_retention, run_rq1
from compint.experiments.rq2_system_factors import (
    build_length_grid,
    output_within_band,
    summarize_length_sweep,
)
from compint.experiments.rq3_sc_factors import (
    build_framing_grid,
    build_location_grid,
    summarize_framings,
    summarize_locations,
)
from compint.experiments.runner import GridRunner
from compint.report.tables import verify_reproduction
from shared.llm_client import LLMRequest, StubLLMClient
from shared.prompts import PromptRegistry

SC_IDS = (1, 2, 4, 7, 13)


def make_context(
    dataset: str, index: int, user_turns: int = 4, target: int = 100000
) -> FillerContext:
    messages: list[Message] = []
    position = 0
    for turn in range(user_turns):
        messages.append(
            Message(
                index=position,
                role=Role.USER,
                content=f"{dataset} user turn {turn} of context {index}",
                token_count=25,
            )
        )
        position += 1
        messages.append(
            Message(
                index=position,
                role=Role.ASSISTANT,
                content=f"{dataset} assistant reply {turn}",
                token_count=25,
            )
        )
        position += 1
    return FillerContext(
        context_id=f"{dataset}_eval_{target}_{index:04d}",
        dataset=dataset,
        split="eval",
        target_tokens=target,
        actual_tokens=sum(m.token_count for m in messages),
        history=History(messages=tuple(messages)),
        source_ids=(f"{dataset}_src_{index}",),
        n_stitched=1,
        status=ContextStatus.OK,
    )


def judge_client(surviving_sc_ids: set[int]) -> StubLLMClient:
    """A judge that says YES only for SCs the test declares survive."""

    def verdict(request: LLMRequest) -> str:
        for sc_id in sorted(surviving_sc_ids):
            marker = f"SC{sc_id}_MARKER"
            if marker in request.user:
                return "YES"
        return "NO"

    return StubLLMClient(default_factory=verdict)


def build_runner(
    catalog: SCCatalog,
    prompts: PromptRegistry,
    *,
    surviving: set[int],
    checkpoint: Checkpoint | None = None,
    conditions: tuple[str, ...] = ("lctx", "lctx_sc", "comp", "ub"),
) -> GridRunner:
    tokenizer = HeuristicTokenizer()
    # Recent-N keeps the last messages verbatim, so an injected SC marker survives into the
    # compacted text and the judge stub can see it. That makes the pipeline observable end to
    # end without a real compactor.
    compactors = {
        "recent_20": RecentNCompactor(20, tokenizer, compactor_id="recent_20"),
        "recent_2": RecentNCompactor(2, tokenizer, compactor_id="recent_2"),
    }
    judge = RetentionJudge(judge_client(surviving), prompts.get("retention_judge"), "stub-judge")
    compliance = ComplianceHarness(
        StubLLMClient(default_factory=lambda _r: "A"),
        prompts.get("mcq_probe"),
        "stub-probe",
        rng=RandomSource(1),
    )
    return GridRunner(
        catalog,
        compactors,
        judge,
        compliance,
        rng=RandomSource(20260731),
        conditions=conditions,  # type: ignore[arg-type]
        checkpoint=checkpoint,
    )


FRAMING = FramingSpec(strength=Strength.PREFERENTIAL, explicitness=Explicitness.DIRECT)


# ---------------------------------------------------------------- RQ1


async def test_rq1_runs_a_full_grid_and_aggregates(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    contexts = {"wildchat": [make_context("wildchat", i) for i in range(3)]}
    runner = build_runner(catalog, prompts, surviving=set())
    result, table = await run_rq1(
        runner,
        contexts,
        SC_IDS,
        ("recent_20",),
        injection_seed=20260731,
        prompt_hashes={"recent_20": "none"},
        run_id="run_rq1",
    )
    assert len(result.outcomes) == 3 * len(SC_IDS)
    assert result.judge_calls == 15
    assert len(table.cells) == 1
    assert table.cells[0].dataset == "wildchat"
    assert table.cells[0].retention.n_valid == 15


async def test_condition_caching_holds_across_a_real_grid(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    """The cost property, checked on an actual run rather than on the estimator.

    K_lctx is built once per (context, compactor). C(H^t) for K_ub is compacted once per
    (context, compactor). Only K_comp pays per instance.
    """
    contexts = {"wildchat": [make_context("wildchat", i) for i in range(3)]}
    runner = build_runner(catalog, prompts, surviving=set())
    result, _ = await run_rq1(
        runner,
        contexts,
        SC_IDS,
        ("recent_20",),
        injection_seed=1,
        prompt_hashes={"recent_20": "none"},
    )
    n_contexts, n_scs = 3, len(SC_IDS)
    assert result.lctx_builds == n_contexts, "K_lctx must be built once per context"
    # One injected compaction per instance, plus one un-injected compaction per context.
    assert result.compaction_calls == n_contexts * n_scs + n_contexts


async def test_rq1_reports_per_category_retention(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    contexts = {"wildchat": [make_context("wildchat", i) for i in range(2)]}
    runner = build_runner(catalog, prompts, surviving=set(), conditions=("comp",))
    result, _ = await run_rq1(
        runner, contexts, SC_IDS, ("recent_20",), injection_seed=1, prompt_hashes={}
    )
    rows = dict(per_category_retention(result))
    assert SCCategoryId.ACTION.value in rows
    assert SCCategoryId.OUTPUT.value in rows


async def test_a_pair_whose_every_judgment_failed_produces_no_cell(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    """Zero retention and no measurement are different claims (INV-6)."""
    contexts = {"wildchat": [make_context("wildchat", 0)]}
    blocked = StubLLMClient(default_factory=lambda _r: "__CONTENT_FILTER__")
    runner = GridRunner(
        catalog,
        {"recent_20": RecentNCompactor(20, HeuristicTokenizer(), compactor_id="recent_20")},
        RetentionJudge(blocked, prompts.get("retention_judge"), "stub-judge"),
        ComplianceHarness(
            StubLLMClient(default_factory=lambda _r: "A"),
            prompts.get("mcq_probe"),
            "stub-probe",
            rng=RandomSource(1),
        ),
        rng=RandomSource(1),
        conditions=("comp",),
    )
    result, table = await run_rq1(
        runner, contexts, SC_IDS, ("recent_20",), injection_seed=1, prompt_hashes={}
    )
    assert result.outcomes, "instances ran"
    assert table.cells == (), "no cell may be emitted when nothing was measurable"


async def test_verdict_runs_over_a_produced_table(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    """The reproduction verdict consumes what the suite actually produces."""
    contexts = {"wildchat": [make_context("wildchat", i) for i in range(2)]}
    runner = build_runner(catalog, prompts, surviving=set(), conditions=("comp",))
    _result, table = await run_rq1(
        runner, contexts, SC_IDS, ("recent_2",), injection_seed=1, prompt_hashes={}
    )
    verdict = verify_reproduction(table)
    assert verdict.n_cells == 1
    assert isinstance(verdict.succeeded, bool)
    assert "Reproduction verdict" in verdict.render()


async def test_checkpoint_resumes_a_partially_completed_grid(
    catalog: SCCatalog, prompts: PromptRegistry, tmp_path
) -> None:
    """Spec 17.4: a run that dies partway resumes rather than restarting."""
    contexts = [make_context("wildchat", i) for i in range(2)]
    path = tmp_path / "checkpoint.jsonl"

    first = build_runner(
        catalog, prompts, surviving=set(), checkpoint=Checkpoint(path), conditions=("comp",)
    )
    result_one, _ = await run_rq1(
        first,
        {"wildchat": contexts},
        SC_IDS,
        ("recent_20",),
        injection_seed=1,
        prompt_hashes={},
    )
    assert len(result_one.outcomes) == 2 * len(SC_IDS)

    resumed = build_runner(
        catalog, prompts, surviving=set(), checkpoint=Checkpoint(path), conditions=("comp",)
    )
    result_two, _ = await run_rq1(
        resumed,
        {"wildchat": contexts},
        SC_IDS,
        ("recent_20",),
        injection_seed=1,
        prompt_hashes={},
    )
    assert result_two.outcomes == (), "every instance was already done"
    assert result_two.skipped_resumed == 2 * len(SC_IDS)


async def test_runner_refuses_a_grid_referencing_unknown_contexts(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    from compint.experiments.rq1_baseline import build_rq1_grid
    from shared.errors import ConfigError

    contexts = [make_context("wildchat", 0)]
    cells = build_rq1_grid(
        {"wildchat": contexts}, SC_IDS, ("recent_20",), injection_seed=1, prompt_hashes={}
    )
    runner = build_runner(catalog, prompts, surviving=set())
    with pytest.raises(ConfigError, match="contexts that were not supplied"):
        await runner.run(cells, [])


# ---------------------------------------------------------------- RQ2


async def test_length_sweep_groups_by_target_and_reports_output_length(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    by_length = {
        10000: [make_context("wildchat", i, target=10000) for i in range(2)],
        100000: [make_context("wildchat", i + 10, target=100000) for i in range(2)],
    }
    cells = build_length_grid(
        by_length,
        SC_IDS,
        ("recent_20",),
        framing=FRAMING,
        injection_seed=1,
        prompt_hashes={},
    )
    contexts = [c for group in by_length.values() for c in group]
    target_by_context = {
        cell.instance_id: next(c.target_tokens for c in contexts if c.context_id == cell.context_id)
        for cell in cells
    }
    runner = build_runner(catalog, prompts, surviving=set(), conditions=("comp",))
    result = await runner.run(cells, contexts)
    sweep = summarize_length_sweep(result, target_by_context)

    series = sweep.series("wildchat", "recent_20")
    assert [point.target_tokens for point in series] == [10000, 100000]
    assert all(point.mean_output_tokens > 0 for point in series)
    growth = sweep.output_length_growth("wildchat", "recent_20")
    assert growth is not None


def test_output_band_check_matches_the_published_range() -> None:
    """Spec 11.3 and Table 11: LLM compactor output spans 301 to 857 tokens."""
    assert output_within_band(301)
    assert output_within_band(857)
    assert not output_within_band(120)
    assert not output_within_band(10_000)


# ---------------------------------------------------------------- RQ3


async def test_location_sweep_covers_four_conditions_and_excludes_degenerate_datasets(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    """FR-023: OpenResearcher is excluded because all four conditions are the same cell."""
    by_dataset = {
        "wildchat": [make_context("wildchat", i) for i in range(2)],
        "openresearcher": [make_context("openresearcher", i, user_turns=1) for i in range(2)],
    }
    cells, excluded = build_location_grid(
        by_dataset,
        SC_IDS,
        ("recent_20",),
        framing=FRAMING,
        injection_seed=1,
        prompt_hashes={},
        repetition_r=2,
    )
    assert excluded == ("openresearcher",)
    assert {cell.dataset for cell in cells} == {"wildchat"}
    assert {cell.condition for cell in cells} == set(InjectionCondition)

    condition_by_instance = {cell.instance_id: cell.condition for cell in cells}
    runner = build_runner(catalog, prompts, surviving=set(), conditions=("comp",))
    sweep_result = await runner.run(cells, by_dataset["wildchat"], repetition_r=2)
    sweep = summarize_locations(sweep_result, condition_by_instance)
    assert len(sweep.points) == 4
    assert sweep.proximity_gradient("wildchat", "recent_20") is not None


async def test_framing_grid_stores_the_full_2x2_and_derives_marginals(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    """Spec 6.7: store the full grid so both readings of the published table are recoverable."""
    contexts = [make_context("wildchat", i) for i in range(2)]
    cells = build_framing_grid(contexts, SC_IDS, ("recent_20",), injection_seed=1, prompt_hashes={})
    framing_by_instance = {cell.instance_id: cell.framing for cell in cells}
    runner = build_runner(catalog, prompts, surviving=set(), conditions=("comp",))
    result = await runner.run(cells, contexts)
    grid = summarize_framings(result, framing_by_instance)

    assert grid.is_complete(), "all four framing cells must be measured"
    strength_marginal = grid.strength_marginal(Explicitness.CONTEXTUALIZED)
    explicitness_marginal = grid.explicitness_marginal(Strength.PREFERENTIAL)
    assert set(strength_marginal) == {"preferential", "strict"}
    assert set(explicitness_marginal) == {"contextualized", "direct"}


# ---------------------------------------------------------------- ablations


async def test_prompt_ablation_reports_lift_and_remaining_gap(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    """The claim is "the prompt helps a lot and is still not enough", so both are measured."""
    contexts = [make_context("wildchat", i) for i in range(2)]
    from compint.experiments.base import build_grid

    baseline_cells = build_grid(
        contexts,
        SC_IDS,
        ("recent_2",),
        framing=FRAMING,
        condition=InjectionCondition.TOP,
        injection_seed=1,
        prompt_hashes={},
    )
    targeted_cells = build_grid(
        contexts,
        SC_IDS,
        ("recent_20",),
        framing=FRAMING,
        condition=InjectionCondition.TOP,
        injection_seed=1,
        prompt_hashes={},
    )
    runner = build_runner(catalog, prompts, surviving=set(SC_IDS), conditions=("comp",))
    result = await runner.run(baseline_cells + targeted_cells, contexts)

    ablation = summarize_prompt_ablation(
        result,
        model_id="stub-model",
        baseline_compactor="recent_2",
        sc_targeted_compactor="recent_20",
    )
    assert ablation is not None
    assert ablation.lift_pp >= 0
    assert "below the registry approach" in ablation.format()


async def test_matching_ablation_reports_whether_mismatch_explains_the_failure(
    catalog: SCCatalog, prompts: PromptRegistry
) -> None:
    contexts = [make_context("hermes_agent", i) for i in range(2)]
    runner = build_runner(catalog, prompts, surviving=set(), conditions=("comp",))
    from compint.experiments.base import build_grid

    cells = build_grid(
        contexts,
        SC_IDS,
        ("recent_2",),
        framing=FRAMING,
        condition=InjectionCondition.TOP,
        injection_seed=1,
        prompt_hashes={},
    )
    result = await runner.run(cells, contexts)
    ablation = summarize_matching_ablation(
        result, compactor_id="recent_2", matched_dataset="hermes_agent"
    )
    assert ablation is not None
    assert ablation.mismatch_explains_the_failure() is False
    assert "does not explain the failure" in ablation.format()


def test_table2_from_an_empty_run_is_empty() -> None:
    from compint.experiments.runner import RunResult

    assert build_table2(RunResult()).cells == ()
