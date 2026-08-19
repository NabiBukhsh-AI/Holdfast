"""Fixture suite integrity and the precision harness. TASK-019, experiment E-02.

The suites themselves are checked in CI. The measured precision and recall numbers require a
fetched extraction prompt and a real model, so those runs are marked and skipped here rather
than faked with a stub, which would produce a number that means nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compint.extractor.client import SCExtractor
from compint.extractor.evaluation import load_suite, precision_recall, run_suite
from compint.extractor.parser import ExtractionStatus
from shared.llm_client import StubLLMClient
from shared.prompts import Prompt

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NEGATIVE = FIXTURES / "negative_turns.jsonl"
MIXED = FIXTURES / "mixed_clause_turns.jsonl"

MINIMUM_NEGATIVE_CASES = 200  # TASK-019 acceptance.
MINIMUM_MIXED_CASES = 40  # Spec 23.10 test data management.


def _prompt() -> Prompt:
    return Prompt(
        id="sc_extractor",
        version="v1",
        provenance="fetched",
        source_url="https://example.invalid/repo",
        fetched_at="2026-08-19T00:00:00Z",
        user="EXTRACTION INSTRUCTIONS\n{inputs}",
    )


def test_negative_suite_meets_required_size() -> None:
    assert len(load_suite(NEGATIVE)) >= MINIMUM_NEGATIVE_CASES


def test_mixed_clause_suite_meets_required_size() -> None:
    assert len(load_suite(MIXED)) >= MINIMUM_MIXED_CASES


def test_negative_suite_expects_no_extraction_and_explains_why() -> None:
    for case in load_suite(NEGATIVE):
        assert case.expect_extraction is False
        assert case.why, f"{case.id} has no rationale, so a failure would not be diagnosable"


def test_negative_suite_covers_every_documented_exclusion_rule() -> None:
    """FR-067 names the categories the extractor must not fire on."""
    reasons = {case.why for case in load_suite(NEGATIVE)}
    for expected in (
        "current task instruction",
        "one off correction",
        "local formatting request",
        "politeness or filler",
        "background fact",
    ):
        assert expected in reasons, f"the negative suite does not exercise: {expected}"


def test_mixed_clause_suite_carries_the_generic_clause() -> None:
    """Spec 14.5 edge case: only the generic clause is a session constraint."""
    for case in load_suite(MIXED):
        assert case.expect_extraction is True
        assert case.generic_clause
        assert case.generic_clause.lower() in case.user_turn.lower()
        assert case.category


async def test_run_suite_scores_a_clean_negative_run() -> None:
    """A stub that always returns an empty list passes the negative suite by construction.

    This checks the harness arithmetic, not the extractor. The real measurement needs the
    fetched prompt and is marked below.
    """
    client = StubLLMClient(default_factory=lambda r: "[]")
    extractor = SCExtractor(client, _prompt(), "qwen3.5-9b")
    result = await run_suite(extractor, load_suite(NEGATIVE)[:20], suite_name="negative")
    assert result.n_passed == 20
    assert result.pass_rate == 1.0
    assert result.format().startswith("negative: 100.0%")
    assert "passed 20/20" in result.format()


async def test_extraction_failure_is_never_scored_as_a_pass() -> None:
    """An outage that produced no constraints is not evidence of correct restraint."""
    client = StubLLMClient(default_factory=lambda r: "__TIMEOUT__")
    extractor = SCExtractor(client, _prompt(), "qwen3.5-9b", max_retries=0, retry_backoff_s=0.0)
    result = await run_suite(extractor, load_suite(NEGATIVE)[:5], suite_name="negative")
    assert result.n_errored == 5
    assert result.n_passed == 0
    with pytest.raises(ValueError, match="no case produced a scoreable outcome"):
        _ = result.pass_rate


async def test_parse_error_counts_as_a_failure_not_a_pass() -> None:
    client = StubLLMClient(default_factory=lambda r: "I did not find anything.")
    extractor = SCExtractor(client, _prompt(), "qwen3.5-9b", max_retries=0, retry_backoff_s=0.0)
    result = await run_suite(extractor, load_suite(NEGATIVE)[:3], suite_name="negative")
    assert result.n_failed == 3
    assert all(o.status is ExtractionStatus.EXTRACTION_PARSE_ERROR for o in result.outcomes)


def test_precision_recall_reports_both_halves() -> None:
    """E-02: recall alone is not a claim about this system."""
    from compint.eval.metrics import wilson_interval
    from compint.extractor.evaluation import SuiteResult

    positive = SuiteResult(
        suite="mixed",
        n_cases=50,
        n_passed=45,
        n_failed=5,
        n_errored=0,
        wilson_ci=wilson_interval(45, 50),
    )
    negative = SuiteResult(
        suite="negative",
        n_cases=200,
        n_passed=190,
        n_failed=10,
        n_errored=0,
        wilson_ci=wilson_interval(190, 200),
    )
    report = precision_recall(positive, negative)
    assert report.recall == pytest.approx(0.90)
    assert report.precision == pytest.approx(45 / 55)
    assert 0.0 < report.f1 < 1.0
    assert "precision" in report.format() and "recall" in report.format()


def test_precision_is_undefined_rather_than_zero_with_no_positives() -> None:
    from compint.eval.metrics import wilson_interval
    from compint.extractor.evaluation import SuiteResult

    empty = SuiteResult(
        suite="mixed", n_cases=0, n_passed=0, n_failed=0, n_errored=0, wilson_ci=wilson_interval(0, 0)
    )
    report = precision_recall(empty, empty)
    with pytest.raises(ValueError, match="precision is undefined"):
        _ = report.precision


@pytest.mark.gpu
@pytest.mark.skip(
    reason=(
        "BLOCKING GATE TASK-001 / U-03: the extraction prompt has not been fetched. "
        "TASK-019 acceptance (empty output on the negative suite above 95 percent, per "
        "category recall above 80 percent) cannot be measured against a stub."
    )
)
async def test_negative_suite_returns_empty_above_threshold() -> None:
    """TASK-019 acceptance, measured against the real extractor once the gate is open."""
    from shared.config import load_config
    from shared.llm_client import OpenAICompatibleClient
    from shared.prompts import get_registry

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "research" / "rq4_extractor.yaml")
    prompts = get_registry(str(root / "prompts"))
    prompts.assert_fetch_gate_open()
    assert config.llm.base_url is not None

    extractor = SCExtractor(
        OpenAICompatibleClient(config.llm.base_url),
        prompts.get("sc_extractor"),
        config.extractor.model,
    )
    negative = await run_suite(extractor, load_suite(NEGATIVE), suite_name="negative")
    positive = await run_suite(extractor, load_suite(MIXED), suite_name="mixed_clause")
    assert negative.pass_rate > 0.95, negative.format()
    report = precision_recall(positive, negative)
    assert report.precision > 0.80, report.format()
