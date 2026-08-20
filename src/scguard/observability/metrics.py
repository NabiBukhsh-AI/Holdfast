"""Metric definitions and the alert rules they back. TASK-030, spec 27.2 and 27.4.

Two design points specific to this system:

1. **Eviction and registry_incomplete are the metrics that matter.** Ordinary service metrics
   (latency, throughput, error rate) say whether SC-GUARD is up. These two say whether it is
   doing its job. A service that is fast, healthy, and quietly dropping constraints is the
   exact failure being mitigated, so those counters carry severity labels and page.

2. **Category labels are load bearing.** Action constraints govern side effects; Output
   constraints govern formatting. Losing one of each is not the same event, so every
   constraint scoped metric is labelled by category and the Action alert is stricter.

The registry is deliberately dependency free: it records into plain counters and histograms so
CI can assert on them, and a Prometheus or OpenTelemetry exporter reads the same objects.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    PAGE = "page"
    TICKET = "ticket"
    INFO = "info"


@dataclass(frozen=True)
class MetricSpec:
    """One metric, its meaning, and why it exists."""

    name: str
    kind: str  # counter | histogram | gauge
    labels: tuple[str, ...]
    description: str


# Spec 27.2. Every metric here is emitted by the service paths in this package.
METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "scguard_turns_submitted_total",
        "counter",
        ("tenant", "outcome"),
        "User turns accepted for extraction. `outcome` separates queued from rejected.",
    ),
    MetricSpec(
        "scguard_extraction_latency_ms",
        "histogram",
        ("model",),
        "Per user turn extraction latency. NFR-001 p50 under 150 ms, NFR-002 p99 under 500 ms.",
    ),
    MetricSpec(
        "scguard_extraction_failed_total",
        "counter",
        ("reason",),
        "Extractions that did NOT produce a verdict. Never conflated with an empty result.",
    ),
    MetricSpec(
        "scguard_hallucinated_evidence_total",
        "counter",
        ("category",),
        "Candidates rejected because their evidence span was not in the user turn. A rising "
        "rate means the extractor is inventing support for constraints it claims to find.",
    ),
    MetricSpec(
        "scguard_constraints_added_total",
        "counter",
        ("category",),
        "Constraints appended to a registry, by category.",
    ),
    MetricSpec(
        "scguard_constraints_superseded_total",
        "counter",
        ("category",),
        "Constraints tombstoned by a later conflicting one.",
    ),
    MetricSpec(
        "scguard_registry_evicted_total",
        "counter",
        ("category", "reason"),
        "Constraints dropped for budget. THIS IS THE PAPER'S FAILURE MODE REINTRODUCED and is "
        "the single loudest signal in the system.",
    ),
    MetricSpec(
        "scguard_registry_incomplete_total",
        "counter",
        ("tenant",),
        "Assemblies that ran while extractions were still pending. NFR-008: never silent.",
    ),
    MetricSpec(
        "scguard_assembly_latency_ms",
        "histogram",
        (),
        "Assembly time excluding compactor time. NFR-003 target under 20 ms.",
    ),
    MetricSpec(
        "scguard_registry_tokens",
        "histogram",
        (),
        "Injected registry size. NFR-012: under 25 percent of mean compactor output.",
    ),
    MetricSpec(
        "scguard_queue_depth",
        "gauge",
        (),
        "Queued plus running extraction jobs. Drives autoscaling and backpressure.",
    ),
    MetricSpec(
        "scguard_category_other_ratio",
        "gauge",
        (),
        "FR-042: share of constraints landing in the open `other` bucket. A rising value means "
        "the five category taxonomy no longer fits what users actually ask for.",
    ),
)


@dataclass(frozen=True)
class AlertRule:
    """An alert and the runbook that answers it. Spec 27.4: every alert has a runbook."""

    name: str
    expression: str
    severity: Severity
    runbook: str
    rationale: str


ALERTS: tuple[AlertRule, ...] = (
    AlertRule(
        name="ExtractorDown",
        expression="rate(scguard_extraction_failed_total[5m]) > 0.1",
        severity=Severity.PAGE,
        runbook="docs/runbooks/extractor_down.md",
        rationale=(
            "Every failed extraction is a turn whose constraints were never read. The registry "
            "keeps reporting itself complete unless this is surfaced."
        ),
    ),
    AlertRule(
        name="RegistryIncompleteSpike",
        expression="rate(scguard_registry_incomplete_total[10m]) > 0.05",
        severity=Severity.PAGE,
        runbook="docs/runbooks/registry_incomplete_spike.md",
        rationale=(
            "Assemblies are proceeding without the constraints from recent turns. The agent is "
            "acting on an incomplete instruction set."
        ),
    ),
    AlertRule(
        name="ActionConstraintEvicted",
        expression='increase(scguard_registry_evicted_total{category="action"}[15m]) > 0',
        severity=Severity.PAGE,
        runbook="docs/runbooks/eviction_spike.md",
        rationale=(
            "Losing an Output constraint produces a formatting error. Losing an Action "
            "constraint produces an unauthorized tool call, so any Action eviction pages."
        ),
    ),
    AlertRule(
        name="EvictionSpike",
        expression="rate(scguard_registry_evicted_total[15m]) > 0.02",
        severity=Severity.TICKET,
        runbook="docs/runbooks/eviction_spike.md",
        rationale="Sustained eviction means the token budget is too small for real sessions.",
    ),
    AlertRule(
        name="ExtractorLatencyHigh",
        expression="histogram_quantile(0.99, scguard_extraction_latency_ms) > 500",
        severity=Severity.TICKET,
        runbook="docs/runbooks/extractor_down.md",
        rationale="NFR-002. Sustained breach pushes drain timeouts into registry_incomplete.",
    ),
    AlertRule(
        name="HallucinatedEvidenceRising",
        expression="rate(scguard_hallucinated_evidence_total[1h]) > 0.01",
        severity=Severity.TICKET,
        runbook="docs/runbooks/extractor_down.md",
        rationale=(
            "The extractor is inventing evidence. Precision, not recall, is what makes false "
            "constraints a user harm."
        ),
    ),
    AlertRule(
        name="CategoryOtherRising",
        expression="scguard_category_other_ratio > 0.25",
        severity=Severity.INFO,
        runbook="docs/runbooks/extractor_down.md",
        rationale="FR-042: the taxonomy needs revision when real constraints stop fitting it.",
    ),
)


@dataclass
class MetricsRegistry:
    """In-process metric sink. An exporter reads these; tests assert on them."""

    counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (name, tuple(sorted((labels or {}).items())))

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        self.counters[self._key(name, labels)] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        self.histograms[self._key(name, labels)].append(value)

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        self.gauges[self._key(name, labels)] = value

    def counter_value(self, name: str, **labels: str) -> float:
        return self.counters.get(self._key(name, labels), 0.0)

    def histogram_values(self, name: str, **labels: str) -> list[float]:
        return list(self.histograms.get(self._key(name, labels), []))

    def gauge_value(self, name: str, **labels: str) -> float | None:
        return self.gauges.get(self._key(name, labels))

    def quantile(self, name: str, q: float, **labels: str) -> float | None:
        values = sorted(self.histogram_values(name, **labels))
        if not values:
            return None
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {q}")
        index = min(len(values) - 1, int(q * len(values)))
        return values[index]

    @contextmanager
    def timed(self, name: str, **labels: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - started) * 1000.0, **labels)


def metric_names() -> frozenset[str]:
    return frozenset(spec.name for spec in METRICS)


def assert_every_alert_has_a_runbook() -> None:
    """Spec 32.3 definition of done: all alerts have runbooks."""
    missing = [alert.name for alert in ALERTS if not alert.runbook]
    if missing:
        raise ValueError(f"alerts without a runbook: {missing}")


def assert_alerts_reference_known_metrics() -> None:
    """An alert on a metric nothing emits is an alert that will never fire."""
    known = metric_names()
    orphans = [
        alert.name for alert in ALERTS if not any(name in alert.expression for name in known)
    ]
    if orphans:
        raise ValueError(f"alerts referencing no known metric: {orphans}")
