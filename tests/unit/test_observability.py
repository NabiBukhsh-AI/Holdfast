"""TASK-030 acceptance tests. Spec 27.2, 27.4, definition of done in 32.3."""

from __future__ import annotations

from pathlib import Path

import pytest

from scguard.observability.metrics import (
    ALERTS,
    METRICS,
    MetricsRegistry,
    Severity,
    assert_alerts_reference_known_metrics,
    assert_every_alert_has_a_runbook,
    metric_names,
)

REPO = Path(__file__).resolve().parents[2]


def test_every_alert_has_a_runbook() -> None:
    """Spec 32.3: an alert without a runbook is a page nobody can action."""
    assert_every_alert_has_a_runbook()
    for alert in ALERTS:
        path = REPO / alert.runbook
        assert path.is_file(), f"{alert.name} points at a missing runbook: {alert.runbook}"
        assert len(path.read_text(encoding="utf-8")) > 500, f"{alert.runbook} is a stub"


def test_every_alert_references_a_metric_something_emits() -> None:
    """An alert on a metric nothing emits is an alert that will never fire."""
    assert_alerts_reference_known_metrics()


def test_every_alert_states_why_it_exists() -> None:
    for alert in ALERTS:
        assert alert.rationale, f"{alert.name} has no rationale"


def test_action_eviction_pages_on_a_single_occurrence() -> None:
    """Losing an Action constraint produces an unauthorized tool call, not a typo."""
    action_alert = next(a for a in ALERTS if a.name == "ActionConstraintEvicted")
    assert action_alert.severity is Severity.PAGE
    assert 'category="action"' in action_alert.expression
    assert "> 0" in action_alert.expression


def test_the_two_correctness_alerts_page() -> None:
    """Latency and throughput say the service is up; these two say it is working."""
    by_name = {alert.name: alert for alert in ALERTS}
    assert by_name["ExtractorDown"].severity is Severity.PAGE
    assert by_name["RegistryIncompleteSpike"].severity is Severity.PAGE


def test_required_metrics_are_declared() -> None:
    """Spec 27.2 and NFR-015: every compaction event reports these."""
    names = metric_names()
    for required in (
        "scguard_registry_evicted_total",
        "scguard_registry_incomplete_total",
        "scguard_registry_tokens",
        "scguard_extraction_failed_total",
        "scguard_extraction_latency_ms",
        "scguard_hallucinated_evidence_total",
        "scguard_category_other_ratio",
    ):
        assert required in names, f"{required} is not declared"


def test_constraint_metrics_are_labelled_by_category() -> None:
    """Losing an Action constraint and losing an Output constraint are different events."""
    by_name = {spec.name: spec for spec in METRICS}
    for name in (
        "scguard_registry_evicted_total",
        "scguard_constraints_added_total",
        "scguard_hallucinated_evidence_total",
    ):
        assert "category" in by_name[name].labels, f"{name} is not labelled by category"


def test_counters_accumulate_by_label_set() -> None:
    registry = MetricsRegistry()
    registry.increment(
        "scguard_registry_evicted_total", category="action", reason="BUDGET_EXCEEDED"
    )
    registry.increment(
        "scguard_registry_evicted_total", category="action", reason="BUDGET_EXCEEDED"
    )
    registry.increment(
        "scguard_registry_evicted_total", category="output", reason="BUDGET_EXCEEDED"
    )
    assert (
        registry.counter_value(
            "scguard_registry_evicted_total", category="action", reason="BUDGET_EXCEEDED"
        )
        == 2
    )
    assert (
        registry.counter_value(
            "scguard_registry_evicted_total", category="output", reason="BUDGET_EXCEEDED"
        )
        == 1
    )


def test_histograms_and_quantiles() -> None:
    registry = MetricsRegistry()
    for value in range(1, 101):
        registry.observe("scguard_extraction_latency_ms", float(value), model="qwen3.5-9b")
    assert registry.quantile("scguard_extraction_latency_ms", 0.5, model="qwen3.5-9b") == 51.0
    assert registry.quantile("scguard_extraction_latency_ms", 0.99, model="qwen3.5-9b") == 100.0


def test_quantile_of_an_unobserved_metric_is_none_not_zero() -> None:
    """Zero would read as an excellent latency rather than as no data."""
    assert MetricsRegistry().quantile("scguard_extraction_latency_ms", 0.5) is None


def test_quantile_rejects_an_out_of_range_q() -> None:
    registry = MetricsRegistry()
    registry.observe("scguard_assembly_latency_ms", 1.0)
    with pytest.raises(ValueError, match="quantile must be in"):
        registry.quantile("scguard_assembly_latency_ms", 1.5)


def test_timed_context_records_a_duration() -> None:
    registry = MetricsRegistry()
    with registry.timed("scguard_assembly_latency_ms"):
        pass
    values = registry.histogram_values("scguard_assembly_latency_ms")
    assert len(values) == 1
    assert values[0] >= 0.0


def test_gauges_replace_rather_than_accumulate() -> None:
    registry = MetricsRegistry()
    registry.set_gauge("scguard_queue_depth", 5)
    registry.set_gauge("scguard_queue_depth", 2)
    assert registry.gauge_value("scguard_queue_depth") == 2


def test_unset_gauge_is_none() -> None:
    assert MetricsRegistry().gauge_value("scguard_queue_depth") is None
