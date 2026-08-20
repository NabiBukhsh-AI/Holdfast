"""TASK-016 acceptance tests. Spec 6.11, 6.12, Algorithm 14.11.

The ER paper fixtures are the cheapest correctness gate in the project (execution contract
rule 18) and they run in CI on every commit with no GPU and no paid API call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compint.core.models import SCCategoryId
from compint.eval.metrics import (
    ERStatus,
    aggregate_effect_retention,
    cohens_kappa,
    compaction_ratio,
    compliance_rate,
    effect_retention,
    retention_rate,
    wilson_interval,
)
from compint.eval.records import (
    ProbeRecord,
    ProbeStatus,
    RetentionRecord,
    RetentionStatus,
)
from shared.errors import EmptyEvaluationSetError

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[1] / "golden" / "paper_table2_fixtures.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["id"])
def test_er_paper_fixtures(case: dict[str, object]) -> None:
    """Reproduce Table 2's ER column from its own compliance columns, within 0.15 pp."""
    result = effect_retention(float(case["c_comp"]), float(case["c_lctx"]), float(case["c_ub"]))
    assert result.status.value == case["expected_status"]
    assert result.percent is not None
    assert result.percent == pytest.approx(
        float(case["expected_er_percent"]), abs=FIXTURES["tolerance_pp"]
    )


@pytest.mark.parametrize("case", FIXTURES["hazard_cases"], ids=lambda c: c["id"])
def test_er_numerical_hazards(case: dict[str, object]) -> None:
    """Spec 6.12: degenerate denominator, negative ER, and ER above 1 are all flagged."""
    result = effect_retention(float(case["c_comp"]), float(case["c_lctx"]), float(case["c_ub"]))
    assert result.status.value == case["expected_status"]
    if case["expected_status"] == "DEGENERATE_DENOMINATOR":
        assert result.value is None
        assert "undefined" in result.detail
    else:
        assert result.percent is not None
        assert result.percent == pytest.approx(float(case["expected_er_percent"]), abs=0.15)


def test_er_negative_flagged_not_clipped() -> None:
    result = effect_retention(0.40, 0.50, 0.95)
    assert result.status is ERStatus.NEGATIVE
    assert result.value is not None
    assert result.value < 0, "a negative ER must survive as a negative number"


def test_er_above_one_not_clipped() -> None:
    result = effect_retention(0.99, 0.50, 0.95)
    assert result.status is ERStatus.ABOVE_UPPER_BOUND
    assert result.value is not None
    assert result.value > 1.0


def test_er_degenerate_denominator() -> None:
    result = effect_retention(0.55, 0.52, 0.54)
    assert result.status is ERStatus.DEGENERATE_DENOMINATOR
    assert result.value is None


def test_er_rejects_out_of_range_rates() -> None:
    with pytest.raises(ValueError, match="must be a rate"):
        effect_retention(1.4, 0.5, 0.9)


def test_wilson_interval_at_zero_rate() -> None:
    """Several Table 2 cells sit at 0.0 percent, where the normal interval is invalid."""
    interval = wilson_interval(0, 750)
    assert interval.lower == 0.0
    assert 0.0 < interval.upper < 0.01


def test_wilson_interval_at_full_rate() -> None:
    interval = wilson_interval(750, 750)
    assert interval.upper == 1.0
    assert 0.99 < interval.lower < 1.0


def test_wilson_half_width_matches_spec_estimate() -> None:
    """Spec 6.11: at N=750 a 17 percent rate carries roughly plus or minus 2.7 points."""
    interval = wilson_interval(128, 750)
    assert interval.half_width_pp == pytest.approx(2.7, abs=0.4)


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        wilson_interval(10, 5)


def _probe(
    condition: str, answer: str | None, gold: str = "A", status: ProbeStatus = ProbeStatus.OK
) -> ProbeRecord:
    return ProbeRecord(
        instance_id="i",
        sc_id=1,
        category=SCCategoryId.ACTION,
        condition=condition,  # type: ignore[arg-type]
        gold=gold,  # type: ignore[arg-type]
        answer=answer,  # type: ignore[arg-type]
        status=status,
    )


def test_compliance_rate_reports_denominator_and_exclusions() -> None:
    """INV-6: no rate is readable without its denominator and exclusion counts."""
    records = [
        _probe("comp", "A"),
        _probe("comp", "B"),
        _probe("comp", None, status=ProbeStatus.OVERFLOW),
        _probe("comp", None, status=ProbeStatus.REFUSED),
    ]
    result = compliance_rate(records)
    assert result.n_valid == 2
    assert result.n_correct == 1
    assert result.rate == 0.5
    assert result.n_excluded == 2
    assert result.exclusion_reasons == {"OVERFLOW": 1, "REFUSED": 1}
    assert "n=2" in result.format()


def test_compliance_rate_refuses_empty_denominator() -> None:
    """Never silently return 0.0 for a rate over zero valid records."""
    with pytest.raises(EmptyEvaluationSetError):
        compliance_rate([_probe("comp", None, status=ProbeStatus.ERROR)])


def _retention(
    verdict: str | None, status: RetentionStatus = RetentionStatus.OK
) -> RetentionRecord:
    return RetentionRecord(
        instance_id="i",
        sc_id=1,
        category=SCCategoryId.ACTION,
        compactor_id="recent_5",
        compacted_hash="sha256:x",
        verdict=verdict,  # type: ignore[arg-type]
        status=status,
    )


def test_unparseable_and_blocked_excluded_never_coerced() -> None:
    """FR-041 and FR-043: coercing these to NO would inflate the headline finding."""
    records = [
        _retention("YES"),
        _retention("NO"),
        _retention(None, RetentionStatus.UNPARSEABLE),
        _retention(None, RetentionStatus.BLOCKED),
    ]
    result = retention_rate(records)
    assert result.n_valid == 2
    assert result.rate == 0.5, "excluded records must not land in the denominator"
    assert result.n_unparseable == 1
    assert result.n_blocked == 1


def test_aggregate_effect_retention_over_records() -> None:
    records: list[ProbeRecord] = []
    # 60 percent under comp, 40 percent under lctx, 100 percent under ub.
    for i in range(10):
        records.append(_probe("comp", "A" if i < 6 else "B"))
        records.append(_probe("lctx", "A" if i < 4 else "B"))
        records.append(_probe("ub", "A"))
    result = aggregate_effect_retention(records)
    assert result.status is ERStatus.OK
    assert result.percent == pytest.approx(33.33, abs=0.01)
    assert set(result.components) == {"comp", "lctx", "ub"}
    assert result.components["ub"].n_valid == 10


def test_aggregate_effect_retention_requires_all_conditions() -> None:
    with pytest.raises(EmptyEvaluationSetError, match="missing"):
        aggregate_effect_retention([_probe("comp", "A")])


def test_compaction_ratio() -> None:
    """Spec 6.14: roughly 182x at 100K."""
    assert compaction_ratio(100000, 550) == pytest.approx(181.8, abs=0.1)
    assert compaction_ratio(100000, 0) is None


def test_cohens_kappa_perfect_agreement_on_balanced_sample() -> None:
    """Spec 6.15: kappa 1.000 on the paper's deliberately balanced N=50 sample."""
    a = [1] * 25 + [0] * 25
    assert cohens_kappa(a, list(a)) == pytest.approx(1.0)


def test_cohens_kappa_undefined_when_one_label_used() -> None:
    with pytest.raises(ValueError, match="undefined"):
        cohens_kappa([1] * 10, [1] * 10)
