"""Bounded extraction queue. TASK-025, FR-085, spec 16.3.

`ENGINEERING RECOMMENDATION` spec 1.6 judgment 2: extraction is asynchronous and off the
request critical path, because the source research's own WildChat latency (12.93 s per 100K
context) is fatal to interactive latency budgets if run inline at compaction time.

Three properties:

1. **Bounded.** A full queue surfaces as HTTP 429 with Retry-After, never as a silent drop. A
   dropped turn is a constraint that was never extracted, and nothing downstream would know.
2. **Idempotent** on (session_id, turn_index, content_hash). A client retry produces one
   extraction, not two, and returns the original job.
3. **Reclaimable.** A worker that dies mid job leaves the job visible and re-claimable after a
   lease expiry rather than stuck in `running` forever.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from scguard.registry.store import content_hash, new_id
from shared.errors import HoldFastError


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARSE_ERROR = "parse_error"


class QueueFullError(HoldFastError):
    """Backpressure. Surfaced as 429 with Retry-After, never as a dropped turn."""

    def __init__(self, depth: int, capacity: int) -> None:
        super().__init__(
            f"extraction queue is full ({depth}/{capacity}). Returning 429 rather than "
            "dropping the turn: a dropped turn is a constraint nobody knows was missed."
        )
        self.depth = depth
        self.capacity = capacity


class ExtractionJob(BaseModel):
    """One unit of extraction work."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    session_id: str
    tenant_id: str
    turn_index: int = Field(ge=0)
    content: str
    previous_assistant_content: str | None = None
    content_hash: str
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    n_extracted: int | None = None
    raw_response: str | None = None
    error_detail: str | None = None
    enqueued_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    claimed_at: datetime | None = None
    completed_at: datetime | None = None
    latency_ms: int | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.PARSE_ERROR)

    @property
    def idempotency_key(self) -> tuple[str, int, str]:
        return (self.session_id, self.turn_index, self.content_hash)


class ExtractionQueue:
    """In-process bounded queue with leases. Redis Streams backed impl shares this interface."""

    def __init__(
        self,
        *,
        capacity: int = 10000,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"queue capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._lease = timedelta(seconds=lease_seconds)
        self._max_attempts = max_attempts
        self._pending: deque[str] = deque()
        self._jobs: dict[str, ExtractionJob] = {}
        self._by_key: dict[tuple[str, int, str], str] = {}
        self._condition = asyncio.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def depth(self) -> int:
        """Queued plus running. This is what the HPA scales on and the alert fires on."""
        return sum(1 for job in self._jobs.values() if not job.is_terminal)

    def pending_count(self, session_id: str | None = None) -> int:
        return sum(
            1
            for job in self._jobs.values()
            if not job.is_terminal and (session_id is None or job.session_id == session_id)
        )

    def get(self, job_id: str) -> ExtractionJob | None:
        return self._jobs.get(job_id)

    def jobs_for(self, session_id: str) -> tuple[ExtractionJob, ...]:
        return tuple(
            sorted(
                (job for job in self._jobs.values() if job.session_id == session_id),
                key=lambda job: job.turn_index,
            )
        )

    async def enqueue(
        self,
        session_id: str,
        tenant_id: str,
        turn_index: int,
        content: str,
        previous_assistant_content: str | None = None,
    ) -> ExtractionJob:
        """Idempotent on (session_id, turn_index, content_hash)."""
        digest = content_hash(content)
        key = (session_id, turn_index, digest)
        existing_id = self._by_key.get(key)
        if existing_id is not None:
            # A repeat submission returns the ORIGINAL job. Spec 18.1.
            return self._jobs[existing_id]

        if self.depth >= self._capacity:
            raise QueueFullError(self.depth, self._capacity)

        job = ExtractionJob(
            job_id=new_id("job"),
            session_id=session_id,
            tenant_id=tenant_id,
            turn_index=turn_index,
            content=content,
            previous_assistant_content=previous_assistant_content,
            content_hash=digest,
        )
        self._jobs[job.job_id] = job
        self._by_key[key] = job.job_id
        self._pending.append(job.job_id)
        async with self._condition:
            self._condition.notify_all()
        return job

    def reclaim_expired(self) -> tuple[ExtractionJob, ...]:
        """Return leases that expired back to the queue.

        A worker that crashed mid job must not leave that turn permanently unextracted while
        the registry reports itself complete.
        """
        now = datetime.now(UTC)
        reclaimed: list[ExtractionJob] = []
        for job_id, job in self._jobs.items():
            if job.status is not JobStatus.RUNNING or job.claimed_at is None:
                continue
            if now - job.claimed_at < self._lease:
                continue
            requeued = job.model_copy(update={"status": JobStatus.QUEUED, "claimed_at": None})
            self._jobs[job_id] = requeued
            self._pending.append(job_id)
            reclaimed.append(requeued)
        return tuple(reclaimed)

    def claim(self) -> ExtractionJob | None:
        """Take the next queued job and start its lease."""
        self.reclaim_expired()
        while self._pending:
            job_id = self._pending.popleft()
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal or job.status is JobStatus.RUNNING:
                continue
            claimed = job.model_copy(
                update={
                    "status": JobStatus.RUNNING,
                    "claimed_at": datetime.now(UTC),
                    "attempts": job.attempts + 1,
                }
            )
            self._jobs[job_id] = claimed
            return claimed
        return None

    def complete(
        self,
        job_id: str,
        *,
        status: JobStatus,
        n_extracted: int | None = None,
        raw_response: str | None = None,
        error_detail: str | None = None,
        latency_ms: int | None = None,
    ) -> ExtractionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise HoldFastError(f"job {job_id} does not exist")
        finished = job.model_copy(
            update={
                "status": status,
                "n_extracted": n_extracted,
                # Retained for audit and reparse: a parser bug found next month can be fixed
                # by reparsing rather than re-invoking the model (spec 19.1).
                "raw_response": raw_response,
                "error_detail": error_detail,
                "completed_at": datetime.now(UTC),
                "latency_ms": latency_ms,
            }
        )
        self._jobs[job_id] = finished
        return finished

    def retry_or_fail(self, job_id: str, error_detail: str) -> ExtractionJob:
        """NFR-010: bounded retries, then a recorded terminal failure."""
        job = self._jobs.get(job_id)
        if job is None:
            raise HoldFastError(f"job {job_id} does not exist")
        if job.attempts >= self._max_attempts:
            return self.complete(job_id, status=JobStatus.FAILED, error_detail=error_detail)
        requeued = job.model_copy(
            update={"status": JobStatus.QUEUED, "claimed_at": None, "error_detail": error_detail}
        )
        self._jobs[job_id] = requeued
        self._pending.append(job_id)
        return requeued

    async def drain(self, session_id: str, timeout_ms: int) -> tuple[bool, int, float]:
        """Wait, bounded, for this session's pending extractions to finish.

        Algorithm 14.8 steps 1 and 2. Returns (complete, still_pending, waited_ms).

        NEVER waits unboundedly, which would block the user, and NEVER proceeds silently,
        which would recreate the failure this system exists to prevent. On timeout the caller
        sets registry_incomplete and surfaces it.
        """
        deadline = asyncio.get_running_loop().time() + timeout_ms / 1000.0
        started = asyncio.get_running_loop().time()
        while True:
            outstanding = self.pending_count(session_id)
            if outstanding == 0:
                waited = (asyncio.get_running_loop().time() - started) * 1000.0
                return True, 0, waited
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                waited = (asyncio.get_running_loop().time() - started) * 1000.0
                return False, outstanding, waited
            try:
                async with self._condition:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
            except TimeoutError:
                waited = (asyncio.get_running_loop().time() - started) * 1000.0
                return False, self.pending_count(session_id), waited

    async def notify_progress(self) -> None:
        """Wake any drain waiting on this session."""
        async with self._condition:
            self._condition.notify_all()
