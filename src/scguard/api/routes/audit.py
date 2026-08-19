"""GET /v1/sessions/{session_id}/audit. TASK-027, spec 18.6.

Requires the `audit:read` scope, which is separate from ordinary session scopes: the audit
stream carries verbatim user evidence spans and is therefore more sensitive than the registry
itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict

from scguard.api.app import ServiceContext, get_context, require_scope, require_tenant
from scguard.audit.emitter import AuditEventType

router = APIRouter(tags=["audit"])

AUDIT_SCOPE = "audit:read"


class AuditEventView(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: int
    event_type: AuditEventType
    constraint_id: str | None
    turn_index: int | None
    payload: dict[str, Any]
    occurred_at: datetime
    loud: bool


class AuditListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    events: tuple[AuditEventView, ...]
    total: int
    next_offset: int | None = None


@router.get("/sessions/{session_id}/audit", response_model=AuditListResponse)
async def list_audit(
    session_id: str,
    request: Request,
    event_type: AuditEventType | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> AuditListResponse:
    require_tenant(request)
    require_scope(request, AUDIT_SCOPE)
    context: ServiceContext = get_context(request)

    events = context.audit.events(session_id, event_type)
    page = events[offset : offset + limit]
    return AuditListResponse(
        session_id=session_id,
        events=tuple(
            AuditEventView(
                event_id=event.event_id,
                event_type=event.event_type,
                constraint_id=event.constraint_id,
                turn_index=event.turn_index,
                payload=event.payload,
                occurred_at=event.occurred_at,
                loud=event.is_loud,
            )
            for event in page
        ),
        total=len(events),
        next_offset=offset + limit if offset + limit < len(events) else None,
    )
