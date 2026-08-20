# Runbook: ExtractorDown

**Alert:** `rate(scguard_extraction_failed_total[5m]) > 0.1`
**Severity:** page

## What is actually wrong

Turns are arriving and their constraints are never being read. This is not a availability
problem with a cosmetic symptom: every failed extraction is a user instruction that the agent
will not receive at the next compaction, and the user has no way to tell.

SC-GUARD does not paper over this. A failed extraction is recorded as `EXTRACTION_FAILED`, and
any assembly that runs while jobs are outstanding sets `registry_incomplete`. If you see this
alert without a matching `RegistryIncompleteSpike`, extraction is failing fast enough that the
drain finds nothing pending, which is worse, not better.

## Triage

1. `GET /v1/ready` on a few replicas. A 503 with `checks.extractor = unreachable` confirms the
   SLM endpoint, not the service.
2. Check the extraction queue depth (`scguard_queue_depth`). Rising depth with failures means
   jobs are being retried and re-queued; flat depth means they are terminating.
3. Read recent `extraction_failed` audit events. `detail` carries the provider error and
   `attempts` shows whether retries were exhausted (NFR-010 allows two).

## Likely causes

| Symptom | Cause | Action |
| --- | --- | --- |
| Connection refused or DNS failure | vLLM pool down or scaled to zero | Restore the pool; jobs re-queue automatically once leases expire |
| Timeouts under load | Extractor saturated | Scale replicas; check whether one tenant is driving turn volume |
| Parse errors, not transport errors | Model or prompt changed | Compare `prompt_hash` on recent constraints against the pinned value |
| 4xx from the provider | Credential or quota | Rotate or raise quota |

## What NOT to do

Do not disable extraction to clear the alert. A quiet system with an empty registry is the
failure this service exists to prevent, and it will look healthy on every other metric.

If you must reduce load, prefer shadow mode (`service.shadow_mode: true`), which keeps
extracting and recording while injecting nothing. That degrades to measurably-nothing rather
than invisibly-nothing.

## Recovery

Jobs whose leases expire are reclaimed and retried automatically. Turns that terminated as
`EXTRACTION_FAILED` are NOT retried on their own: they were already given their retry budget.
To recover those, re-submit the affected turns, which is idempotent on
`(session_id, turn_index, content_hash)`.
