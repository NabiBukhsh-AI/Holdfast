# Runbook: RegistryIncompleteSpike

**Alert:** `rate(scguard_registry_incomplete_total[10m]) > 0.05`
**Severity:** page

## What is actually wrong

Compaction events are completing while extractions for recent turns are still pending. The
augmented context those events produced is missing constraints the user has already issued.
The agent is now acting on an incomplete instruction set and does not know it.

This is the designed-visible version of the failure. The alternative, waiting indefinitely for
the drain, blocks the user, so the service bounds the wait and tells you instead.

## Triage

1. Find the affected assemblies: they are indexed for exactly this query.

       SELECT session_id, compaction_index, drain_wait_ms, created_at
       FROM assemblies WHERE registry_incomplete = TRUE ORDER BY created_at DESC LIMIT 100;

2. Compare `drain_wait_ms` against the configured `service.drain_timeout_ms` (default 200).
   Values pinned at the timeout mean the drain is genuinely running out of time.
3. Check extraction latency (`scguard_extraction_latency_ms` p99) and queue depth.

## Likely causes

| Pattern | Cause | Action |
| --- | --- | --- |
| p99 latency above 500 ms | Extractor saturated or a slow model revision | Scale replicas; verify the pinned model revision |
| Queue depth climbing | Turn arrival exceeds extraction throughput | Scale workers; check for a hot tenant |
| Bursty, correlated with long turns | WildChat style dense-user-turn traffic | Expected under load; consider raising the drain timeout for that tenant |
| Steady low rate | Turn submitted microseconds before compaction | Benign and inherent; watch the rate, not individual events |

## Tuning the drain timeout

The default of 200 ms is roughly 4x the expected p50 extraction latency, and it is an
assumption, not a measurement from the source research. Raising it trades user-visible
compaction latency for registry completeness. Do not raise it above the point where compaction
becomes perceptible; prefer fixing extraction throughput.

## What NOT to do

Do not suppress the flag or stop returning `X-SC-Registry-Incomplete`. The flag is the only
mechanism by which a caller can know its context is incomplete.
