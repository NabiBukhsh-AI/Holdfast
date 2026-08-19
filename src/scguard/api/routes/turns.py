"""POST /v1/sessions/{session_id}/turns. TASK-027, spec 18.1.

`INV-3 AT THE API BOUNDARY` The endpoint rejects any role other than `user`. That is the
outermost enforcement of "the extractor never receives assistant turns as extraction sources":
a non user turn cannot even enter the system, let alone reach the extractor.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from scguard.api.app import ServiceContext, get_context, require_tenant
from scguard.api.errors import APIError, ErrorCode
from scguard.extractor.queue import QueueFullError
from scguard.registry.store import content_hash

router = APIRouter(tags=["turns"])


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=0)
    role: str
    content: str = Field(min_length=1)
    previous_assistant_content: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class TurnResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    turn_index: int
    extraction_job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "parse_error"]
    registry_version: int


@router.post(
    "/sessions/{session_id}/turns",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TurnResponse,
)
async def submit_turn(
    session_id: str, payload: TurnRequest, request: Request, response: Response
) -> TurnResponse:
    tenant_id = require_tenant(request)
    context: ServiceContext = get_context(request)

    if payload.role != "user":
        raise APIError(
            ErrorCode.INVALID_ROLE,
            f"role must be 'user', got {payload.role!r}. The extractor reads user turns only "
            "(INV-3, FR-060), so non user turns are rejected at the boundary.",
        )
    if len(payload.content.encode("utf-8")) > context.config.service.max_content_bytes:
        raise APIError(
            ErrorCode.CONTENT_TOO_LARGE,
            f"content exceeds {context.config.service.max_content_bytes} bytes",
        )

    session = await context.ensure_session(session_id, tenant_id)

    # Idempotency is on (session_id, turn_index, content_hash). A repeat of the SAME content
    # returns the original job; the same index with DIFFERENT content is a conflict, because
    # silently replacing it would discard whichever extraction was already under way.
    digest = content_hash(payload.content)
    for job in context.queue.jobs_for(session_id):
        if job.turn_index == payload.turn_index and job.content_hash != digest:
            raise APIError(
                ErrorCode.TURN_CONFLICT,
                f"turn {payload.turn_index} was already submitted with different content",
                submitted_hash=digest,
                existing_hash=job.content_hash,
            )

    try:
        job = await context.queue.enqueue(
            session_id,
            tenant_id,
            payload.turn_index,
            payload.content,
            payload.previous_assistant_content,
        )
    except QueueFullError as exc:
        # Backpressure surfaces as 429, never as a silent drop (TASK-025).
        raise APIError(
            ErrorCode.RATE_LIMITED,
            str(exc),
            queue_depth=exc.depth,
            queue_capacity=exc.capacity,
        ) from exc

    response.headers["Location"] = f"/v1/sessions/{session_id}/turns/{payload.turn_index}"
    return TurnResponse(
        session_id=session_id,
        turn_index=payload.turn_index,
        extraction_job_id=job.job_id,
        status=job.status.value,  # type: ignore[arg-type]
        registry_version=session.registry_version,
    )
