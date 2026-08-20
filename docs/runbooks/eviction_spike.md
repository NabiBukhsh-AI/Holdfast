# Runbook: EvictionSpike and ActionConstraintEvicted

**Alerts:**
- `increase(scguard_registry_evicted_total{category="action"}[15m]) > 0` (page)
- `rate(scguard_registry_evicted_total[15m]) > 0.02` (ticket)

## What is actually wrong

Constraints are being dropped from the registry because the session exceeded its token budget.
An evicted constraint is no longer being enforced. This is the source research's failure mode
reintroduced by our own bounding, which is why it is the loudest event in the system rather
than a debug line.

The Action variant pages on a SINGLE occurrence. Losing an Output constraint produces a
formatting error. Losing an Action constraint produces an unauthorized tool call, and those are
not the same incident.

## Triage

1. Pull the eviction events. Each carries the constraint text, category, severity and the
   budget state at the moment it was dropped.

       GET /v1/sessions/{session_id}/audit?event_type=registry_evicted

2. Check whether the session is unusually long or whether one constraint is unusually large.
   A `BUDGET_EXCEEDED_SINGLE` reason means one constraint exceeded the entire budget and was
   kept whole rather than truncated, which is intended: a half constraint can invert its own
   meaning.
3. Look at the distribution of `scguard_registry_tokens`. If the median is near the budget,
   the budget is too small for real traffic rather than this session being pathological.

## Likely causes

| Pattern | Cause | Action |
| --- | --- | --- |
| Action constraints evicted | Budget too small, or too many high severity constraints | Raise `registry.budget_tokens`; this is the case the severity ordering exists to prevent, so it should be rare |
| Many low severity evictions | Long session accumulating Output and Preference constraints | Expected; confirm the ordering kept the important ones |
| Sudden spike across sessions | A config change lowered the budget | Check recent deploys |
| Duplicates filling the budget | Deduplication is not catching paraphrases | Check `tau_dup`; an unset value disables tier 2 entirely |

## Budget sizing

The 200 token default is an assumption derived from compactor output being 301 to 857 tokens
and roughly invariant to input length: a registry much larger than that changes the character
of the compacted context. It has not been validated against real sessions. That validation is
open experiment E-04, and this alert firing regularly is evidence it should be run.

## Tell the user

When constraints are evicted the assembly response carries a `REGISTRY_EVICTED` warning naming
how many and in which categories. Surface it. A user who is told the assistant can no longer
guarantee a constraint can re-state it; a user who is not told cannot.
