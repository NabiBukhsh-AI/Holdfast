"""Registry token budget enforcement. TASK-024, Algorithm 14.7.

`ENGINEERING RECOMMENDATION` The source research leaves S^t unbounded. That cannot ship.
Compactor output is 301 to 857 tokens and is nearly INVARIANT to input length (spec 6.14), so
an unbounded registry would eventually dominate the compacted context and defeat the very
mechanism it implements.

`EVICTION IS THE FAILURE MODE REINTRODUCED.` Spec 14.7 is emphatic about this and so is the
implementation: every eviction emits a loud audit event and a metric. It is never a debug log
and never a silent drop. A constraint the system quietly stopped enforcing is precisely the
harm this whole system exists to prevent, so if it must happen, it happens noisily.

Priority ordering (assumption A-10, to be validated by experiment E-04):
    1. Pinned constraints, if the FR-084 management endpoint is enabled
    2. Category severity: Action > Information > Process > Preference > Output
    3. Recency: newer before older within the same severity band
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from scguard.audit.emitter import AuditEmitter, AuditEventType
from scguard.registry.store import SessionConstraint
from shared.errors import BudgetNotConfiguredError


class BudgetDecision(BaseModel):
    """What survived, what did not, and why. Both halves are always reported."""

    model_config = ConfigDict(frozen=True)

    kept: tuple[SessionConstraint, ...]
    evicted: tuple[SessionConstraint, ...]
    budget_tokens: int
    kept_tokens: int
    # A single constraint larger than the whole budget is kept whole and flagged. Truncating a
    # constraint mid sentence is worse than exceeding the budget, because a half constraint can
    # invert meaning: "Don't send emails without" ... (spec 14.7 edge case).
    budget_exceeded_single: bool = False

    @property
    def n_evicted(self) -> int:
        return len(self.evicted)

    @property
    def over_budget(self) -> bool:
        return self.kept_tokens > self.budget_tokens


def eviction_priority(constraint: SessionConstraint) -> tuple[int, int, int]:
    """Sort key: lower sorts first and is kept first.

    Pinned wins outright, then category severity, then recency (higher seq is newer, so it is
    negated to sort first).
    """
    return (0 if constraint.pinned else 1, constraint.severity, -constraint.seq)


def enforce_budget(
    constraints: Sequence[SessionConstraint],
    budget_tokens: int,
    *,
    audit: AuditEmitter | None = None,
    session_id: str = "",
    tenant_id: str = "",
) -> BudgetDecision:
    """Algorithm 14.7. Returns the injected set and the eviction ledger.

    Raises when the budget is zero or unset: spec 14.7 forbids defaulting to unbounded, since
    an unbounded registry is how this mechanism quietly stops working.
    """
    if budget_tokens <= 0:
        raise BudgetNotConfiguredError(
            f"registry budget must be positive, got {budget_tokens}. Spec 14.7 forbids "
            "defaulting to unbounded: that is how the registry silently grows to dominate "
            "the compacted context."
        )

    ordered = sorted(constraints, key=eviction_priority)
    kept: list[SessionConstraint] = []
    evicted: list[SessionConstraint] = []
    total = 0
    exceeded_single = False

    for constraint in ordered:
        if total + constraint.token_count <= budget_tokens:
            kept.append(constraint)
            total += constraint.token_count
            continue
        if not kept and constraint.token_count > budget_tokens:
            # Keep it whole and exceed the budget rather than truncate a constraint.
            kept.append(constraint)
            total += constraint.token_count
            exceeded_single = True
            if audit is not None:
                audit.emit(
                    session_id,
                    tenant_id,
                    AuditEventType.REGISTRY_EVICTED,
                    constraint_id=constraint.constraint_id,
                    reason="BUDGET_EXCEEDED_SINGLE",
                    detail=(
                        f"constraint is {constraint.token_count} tokens against a budget of "
                        f"{budget_tokens}; kept whole because a truncated constraint can "
                        "invert its own meaning"
                    ),
                    canonical_text=constraint.canonical_text,
                    category=constraint.category.value,
                )
            continue
        evicted.append(constraint)
        if audit is not None:
            # LOUD. Spec 14.7: audit record and metric, never a debug log, and in production
            # surfaced to the agent so it can tell the user what it can no longer guarantee.
            audit.emit(
                session_id,
                tenant_id,
                AuditEventType.REGISTRY_EVICTED,
                constraint_id=constraint.constraint_id,
                reason="BUDGET_EXCEEDED",
                detail=(
                    f"evicted at {total}/{budget_tokens} tokens; this constraint is no longer "
                    "being enforced"
                ),
                canonical_text=constraint.canonical_text,
                category=constraint.category.value,
                severity=constraint.severity,
                token_count=constraint.token_count,
            )

    # Restore registry order for rendering: the user issued them in a sequence and the block
    # reads better in that sequence than in eviction priority order.
    kept.sort(key=lambda row: row.seq)

    return BudgetDecision(
        kept=tuple(kept),
        evicted=tuple(evicted),
        budget_tokens=budget_tokens,
        kept_tokens=total,
        budget_exceeded_single=exceeded_single,
    )


def evicted_summary(decision: BudgetDecision) -> str:
    """Human readable line for the agent facing warning when constraints were dropped."""
    if not decision.evicted:
        return ""
    by_category: dict[str, int] = {}
    for constraint in decision.evicted:
        by_category[constraint.category.value] = by_category.get(constraint.category.value, 0) + 1
    listed = ", ".join(f"{count} {category}" for category, count in sorted(by_category.items()))
    return (
        f"{decision.n_evicted} constraint(s) exceeded the {decision.budget_tokens} token "
        f"registry budget and are no longer being enforced ({listed})"
    )
