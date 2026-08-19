"""Extraction worker. TASK-025.

Pulls jobs, runs the extractor over the three inputs, and feeds surviving candidates into the
registry updater.

`NFR-008` An extractor outage becomes `EXTRACTION_FAILED` on the job and a loud audit event.
It never becomes "this turn declared no constraints". The distinction is the whole reason the
extractor returns a status rather than a list.
"""

from __future__ import annotations

import asyncio
import logging

from compint.extractor.client import SCExtractor
from compint.extractor.parser import ExtractionStatus, RejectionReason
from scguard.audit.emitter import AuditEmitter, AuditEventType
from scguard.extractor.queue import ExtractionJob, ExtractionQueue, JobStatus
from scguard.registry.dedup import RegistryUpdater
from scguard.registry.store import SCCategory

logger = logging.getLogger(__name__)


class ExtractionWorker:
    """One worker. Stateless and horizontally scalable (NFR-006)."""

    def __init__(
        self,
        queue: ExtractionQueue,
        extractor: SCExtractor,
        updater: RegistryUpdater,
        audit: AuditEmitter,
        *,
        chars_per_token: float = 4.0,
    ) -> None:
        self._queue = queue
        self._extractor = extractor
        self._updater = updater
        self._audit = audit
        self._chars_per_token = chars_per_token
        self.processed = 0

    def _token_count(self, text: str) -> int:
        return max(1, int(len(text) / self._chars_per_token))

    async def process_one(self) -> ExtractionJob | None:
        """Claim and run a single job. Returns None when the queue is empty."""
        job = self._queue.claim()
        if job is None:
            return None
        try:
            finished = await self._run(job)
        finally:
            # A drain waiting on this session must wake whether the job succeeded or not.
            await self._queue.notify_progress()
        self.processed += 1
        return finished

    async def _run(self, job: ExtractionJob) -> ExtractionJob:
        registry = await self._updater.store.active(job.session_id)
        call = await self._extractor.extract(
            job.content,
            job.previous_assistant_content,
            [row.canonical_text for row in registry],
        )
        result = call.result

        if result.status is ExtractionStatus.EXTRACTION_FAILED:
            # LOUD. The turn was not examined, and the registry is therefore incomplete.
            self._audit.emit(
                job.session_id,
                job.tenant_id,
                AuditEventType.EXTRACTION_FAILED,
                turn_index=job.turn_index,
                detail=result.detail,
                attempts=call.attempts,
            )
            logger.error(
                "extraction_failed",
                extra={
                    "session_id": job.session_id,
                    "turn_index": job.turn_index,
                    "detail": result.detail,
                },
            )
            return self._queue.complete(
                job.job_id,
                status=JobStatus.FAILED,
                error_detail=result.detail,
                latency_ms=int(call.latency_ms),
            )

        if result.status is ExtractionStatus.EXTRACTION_PARSE_ERROR:
            self._audit.emit(
                job.session_id,
                job.tenant_id,
                AuditEventType.EXTRACTION_FAILED,
                turn_index=job.turn_index,
                detail=f"parse error: {result.detail}",
                attempts=call.attempts,
            )
            return self._queue.complete(
                job.job_id,
                status=JobStatus.PARSE_ERROR,
                raw_response=result.raw_response,
                error_detail=result.detail,
                latency_ms=int(call.latency_ms),
            )

        for rejected in result.rejected:
            if rejected.reason is RejectionReason.HALLUCINATED_EVIDENCE:
                self._audit.emit(
                    job.session_id,
                    job.tenant_id,
                    AuditEventType.HALLUCINATED_EVIDENCE_REJECTED,
                    turn_index=job.turn_index,
                    detail=rejected.detail,
                    **rejected.payload,
                )

        for candidate in result.extracted:
            await self._updater.add_candidate(
                job.session_id,
                job.tenant_id,
                canonical_text=candidate.canonical_text,
                category=SCCategory(candidate.category.value),
                turn_index=job.turn_index,
                token_count=self._token_count(candidate.canonical_text),
                evidence_span=candidate.evidence_span,
            )

        return self._queue.complete(
            job.job_id,
            status=JobStatus.SUCCEEDED,
            n_extracted=len(result.extracted),
            raw_response=result.raw_response,
            latency_ms=int(call.latency_ms),
        )

    async def run_forever(self, poll_interval_s: float = 0.05) -> None:
        """Worker loop. Cancelled by the service on shutdown."""
        while True:
            job = await self.process_one()
            if job is None:
                await asyncio.sleep(poll_interval_s)

    async def drain_all(self, limit: int = 1000) -> int:
        """Process every currently queued job. Used by tests and by graceful shutdown."""
        processed = 0
        while processed < limit:
            job = await self.process_one()
            if job is None:
                return processed
            processed += 1
        return processed
