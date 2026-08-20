"""TASK-018 acceptance tests. Spec 15.10, 24.1."""

from __future__ import annotations

import pytest

from compint.core.models import SCCategoryId
from compint.eval.metrics import (
    compliance_rate,
    effect_retention,
    retention_rate,
    wilson_interval,
)
from compint.eval.records import ProbeRecord, RetentionRecord, RetentionStatus
from compint.report.tables import (
    MAX_OUT_OF_TOLERANCE_FRACTION,
    CellVerdict,
    ResultCell,
    Table2,
    check_orderings,
    compliance_matches_paper,
    er_matches_paper,
    render_per_category,
    verify_reproduction,
)


def make_retention(percent: float, n: int = 750) -> object:
    correct = round(n * percent / 100)
    records = [
        RetentionRecord(
            instance_id=f"i{i}",
            sc_id=(i % 15) + 1,
            category=SCCategoryId.ACTION,
            compactor_id="c",
            compacted_hash="h",
            verdict="YES" if i < correct else "NO",
            status=RetentionStatus.OK,
        )
        for i in range(n)
    ]
    return retention_rate(records)


def make_cell(dataset: str, compactor: str, percent: float, degenerate: bool = False) -> ResultCell:
    return ResultCell(
        dataset=dataset,
        compactor_id=compactor,
        retention=make_retention(percent),  # type: ignore[arg-type]
        degenerate=degenerate,
    )


def test_floor_cells_use_an_absolute_band() -> None:
    """Spec 15.10: cells at 0.0 to 1.0 percent are compared within 2 points absolute."""
    assert make_cell("wildchat", "recent_5", 0.0).verdict() is CellVerdict.WITHIN_TOLERANCE
    assert make_cell("wildchat", "recent_5", 1.5).verdict() is CellVerdict.WITHIN_TOLERANCE
    assert make_cell("wildchat", "recent_5", 8.0).verdict() is CellVerdict.OUT_OF_TOLERANCE


def test_higher_cells_use_the_looser_of_relative_and_absolute() -> None:
    """36.3 percent reference: within 25 percent relative OR 5 points absolute."""
    assert (
        make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 36.3).verdict()
        is CellVerdict.WITHIN_TOLERANCE
    )
    assert (
        make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 30.0).verdict()
        is CellVerdict.WITHIN_TOLERANCE
    )
    assert (
        make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 5.0).verdict()
        is CellVerdict.OUT_OF_TOLERANCE
    )


def test_cells_without_a_published_reference_are_not_graded() -> None:
    assert (
        make_cell("wildchat", "gemma_4_e4b__anthropic", 12.0).verdict() is CellVerdict.NO_REFERENCE
    )


def test_degenerate_cells_are_not_graded() -> None:
    """FR-023: an OpenResearcher location cell is degenerate by construction."""
    cell = make_cell("openresearcher", "recent_5", 0.0, degenerate=True)
    assert cell.verdict() is CellVerdict.DEGENERATE


def test_table_reports_the_headline_mean() -> None:
    table = Table2(
        cells=(
            make_cell("wildchat", "recent_5", 0.0),
            make_cell("hermes_agent", "recent_5", 0.4),
            make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 36.3),
            make_cell("wildchat", "gpt_oss_120b__pi_mono", 6.7),
        ),
        run_id="run_test",
    )
    assert table.mean_retention_percent() == pytest.approx(10.85, abs=0.2)
    rendered = table.render()
    assert "mean retention across cells" in rendered
    assert "run_test" in rendered


def test_orderings_hold_on_a_faithful_reproduction() -> None:
    table = Table2(
        cells=(
            make_cell("wildchat", "recent_5", 0.0),
            make_cell("hermes_agent", "recent_5", 0.4),
            make_cell("wildchat", "llmlingua2_t500", 0.1),
            make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 36.3),
            make_cell("wildchat", "gpt_oss_120b__pi_mono", 6.7),
        )
    )
    orderings = {o.name: o for o in check_orderings(table)}
    assert orderings["non_llm_compactors_near_zero"].holds
    assert orderings["llm_compactors_beat_non_llm"].holds
    assert orderings["pi_mono_hermes_above_wildchat"].holds


def test_ordering_failure_is_detected() -> None:
    """An inverted ordering is a reproduction failure even if every number is in band."""
    table = Table2(
        cells=(
            make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 6.7),
            make_cell("wildchat", "gpt_oss_120b__pi_mono", 36.3),
        )
    )
    orderings = {o.name: o for o in check_orderings(table)}
    assert not orderings["pi_mono_hermes_above_wildchat"].holds


def test_non_llm_ceiling_is_enforced() -> None:
    table = Table2(cells=(make_cell("wildchat", "recent_5", 15.0),))
    orderings = {o.name: o for o in check_orderings(table)}
    assert not orderings["non_llm_compactors_near_zero"].holds


def test_verdict_passes_a_faithful_reproduction() -> None:
    table = Table2(
        cells=(
            make_cell("wildchat", "recent_5", 0.0),
            make_cell("hermes_agent", "recent_5", 0.4),
            make_cell("wildchat", "llmlingua2_t500", 0.1),
            make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 36.3),
            make_cell("wildchat", "gpt_oss_120b__pi_mono", 6.7),
        )
    )
    verdict = verify_reproduction(
        table,
        extractor_retention={"wildchat": 90.3, "hermes_agent": 95.6, "openresearcher": 95.1},
    )
    assert verdict.succeeded, verdict.render()
    assert verdict.n_out_of_tolerance == 0
    assert "PASS" in verdict.render()


def test_verdict_fails_when_too_many_cells_miss() -> None:
    table = Table2(
        cells=(
            make_cell("wildchat", "recent_5", 20.0),
            make_cell("hermes_agent", "recent_5", 0.4),
            make_cell("wildchat", "llmlingua2_t500", 0.1),
            make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 36.3),
        )
    )
    verdict = verify_reproduction(table)
    assert not verdict.succeeded
    assert verdict.out_of_tolerance_fraction > MAX_OUT_OF_TOLERANCE_FRACTION
    assert "FINDING" in verdict.render()


def test_verdict_fails_when_the_extractor_misses_its_floor() -> None:
    """The mitigation claim is the load bearing one, so 85 percent is a hard floor."""
    table = Table2(cells=(make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 36.3),))
    verdict = verify_reproduction(table, extractor_retention={"wildchat": 71.0})
    assert not verdict.succeeded
    assert any("below the 85% floor" in f for f in verdict.failures)


def test_dev_split_results_are_never_reportable() -> None:
    table = Table2(cells=(make_cell("hermes_agent", "gpt_oss_120b__pi_mono", 36.3),), split="dev")
    verdict = verify_reproduction(table)
    assert not verdict.succeeded
    assert any("never reportable" in f for f in verdict.failures)


def test_er_comparison_uses_the_eight_point_band() -> None:
    """Spec 15.10: ER against the paper compounds four noisy inputs."""
    observed = effect_retention(0.585, 0.461, 0.983)
    assert er_matches_paper(observed, 23.8)
    assert er_matches_paper(observed, 30.0)
    assert not er_matches_paper(observed, 45.0)


def test_degenerate_er_never_matches() -> None:
    degenerate = effect_retention(0.55, 0.52, 0.54)
    assert not er_matches_paper(degenerate, 0.0)


def test_compliance_comparison_uses_five_points() -> None:
    records = [
        ProbeRecord(
            instance_id=f"i{i}",
            sc_id=1,
            category=SCCategoryId.ACTION,
            condition="comp",
            gold="A",
            answer="A" if i < 500 else "B",
        )
        for i in range(1000)
    ]
    observed = compliance_rate(records)
    assert compliance_matches_paper(observed, 50.0)
    assert compliance_matches_paper(observed, 54.0)
    assert not compliance_matches_paper(observed, 60.0)


def test_per_category_rendering_lists_every_category() -> None:
    rows = [(category.value, make_retention(90.0)) for category in SCCategoryId]
    rendered = render_per_category(rows)  # type: ignore[arg-type]
    for category in SCCategoryId:
        assert category.value in rendered


def test_wilson_interval_travels_with_every_cell() -> None:
    """INV-6 at the report layer: a rate is never printed without its denominator."""
    cell = make_cell("wildchat", "recent_5", 0.0)
    row = cell.format_row()
    assert "n=750" in row
    assert wilson_interval(0, 750).format_pp() in row
