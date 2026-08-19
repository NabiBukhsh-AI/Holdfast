"""Constraint management endpoints. TASK-027, spec 18.3, 18.4, 18.5, FR-084.

These power user facing transparency: list what the assistant believes it was told, add a
constraint explicitly, and revoke one. Revocation TOMBSTONES; it never deletes (FR-080).
"""

from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from scguard.api.app import ServiceContext, get_context, require_tenant
from scguard.api.errors import APIError, ErrorCode
from scguard.audit.emitter import AuditEventType, reconstruct_at_turn
from scguard.registry.store import (
    DuplicateConstraintError,
    SCCategory,
    SCStatus,
    SessionConstraint,
    SessionNotFoundError,
)

router = APIRouter(tags=["constraints"])


class ConstraintView(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    canonical_text: str
    category: SCCategory
    evidence_span: str | None
    source_turn_index: int
    status: SCStatus
    created_at: datetime
    tokens: int
    pinned: bool = False
    superseded_by: str | None = None


class ConstraintListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    version: int
    constraints: tuple[ConstraintView, ...]
    tombstoned: tuple[ConstraintView, ...] = ()
    as_of_turn: int | None = None


class AddConstraintRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_text: str = Field(min_length=1)
    category: SCCategory = SCCategory.OTHER
    # Exempts the constraint from budget eviction (Algorithm 14.7 priority rule 3).
    pinned: bool = False


def to_view(row: SessionConstraint) -> ConstraintView:
    return ConstraintView(
        id=row.constraint_id,
        canonical_text=row.canonical_text,
        category=row.category,
        evidence_span=row.evidence_span,
        source_turn_index=row.source_turn_index,
        status=row.status,
        created_at=row.created_at,
        tokens=row.token_count,
        pinned=row.pinned,
        superseded_by=row.superseded_by,
    )


@router.get("/sessions/{session_id}/constraints", response_model=ConstraintListResponse)
async def list_constraints(
    session_id: str,
    request: Request,
    include_tombstoned: bool = Query(default=True),
    as_of_turn: int | None = Query(default=None, ge=0),
) -> ConstraintListResponse:
    require_tenant(request)
    context: ServiceContext = get_context(request)
    session = await context.store.get_session(session_id)
    everything = await context.store.all_constraints(session_id)

    if as_of_turn is not None:
        # FR-082: point in time reconstruction, served from the audit stream alone. The stored
        # rows supply only the immutable fields; the STATUS comes from the replayed events.
        rebuilt = reconstruct_at_turn(context.audit.events(session_id), as_of_turn)
        by_id = {row.constraint_id: row for row in everything}
        views: list[ConstraintView] = []
        for entry in rebuilt:
            stored = by_id.get(entry.constraint_id)
            if stored is None:
                continue
            views.append(to_view(stored).model_copy(update={"status": entry.status}))
        return ConstraintListResponse(
            session_id=session_id,
            version=session.registry_version,
            constraints=tuple(v for v in views if v.status is SCStatus.ACTIVE),
            tombstoned=(
                tuple(v for v in views if v.status is not SCStatus.ACTIVE)
                if include_tombstoned
                else ()
            ),
            as_of_turn=as_of_turn,
        )

    return ConstraintListResponse(
        session_id=session_id,
        version=session.registry_version,
        constraints=tuple(to_view(row) for row in everything if row.status is SCStatus.ACTIVE),
        tombstoned=(
            tuple(to_view(row) for row in everything if row.status is not SCStatus.ACTIVE)
            if include_tombstoned
            else ()
        ),
    )


@router.post("/sessions/{session_id}/constraints", response_model=ConstraintView)
async def add_constraint(
    session_id: str, payload: AddConstraintRequest, request: Request
) -> ConstraintView:
    """Manual declaration, for tenants who prefer it over extraction (spec 18.4)."""
    tenant_id = require_tenant(request)
    context: ServiceContext = get_context(request)
    await context.ensure_session(session_id, tenant_id)

    try:
        result = await context.updater.add_candidate(
            session_id,
            tenant_id,
            canonical_text=payload.canonical_text,
            category=payload.category,
            turn_index=0,
            token_count=max(1, len(payload.canonical_text) // 4),
            pinned=payload.pinned,
        )
    except DuplicateConstraintError as exc:
        raise APIError(ErrorCode.VALIDATION_ERROR, str(exc)) from exc

    # Idempotent on normalized text: a duplicate submission returns the existing row.
    target_id = result.constraint_id or result.matched_id
    rows = await context.store.all_constraints(session_id)
    for row in rows:
        if row.constraint_id == target_id:
            return to_view(row)
    raise APIError(ErrorCode.CONSTRAINT_NOT_FOUND, "the constraint was not persisted")


@router.delete(
    "/sessions/{session_id}/constraints/{constraint_id}", response_model=ConstraintView
)
async def revoke_constraint(
    session_id: str, constraint_id: str, request: Request
) -> ConstraintView:
    """Tombstone, not delete. Idempotent: revoking an already revoked constraint returns 200."""
    tenant_id = require_tenant(request)
    context: ServiceContext = get_context(request)

    try:
        rows = await context.store.all_constraints(session_id)
    except SessionNotFoundError as exc:
        raise APIError(ErrorCode.SESSION_NOT_FOUND, str(exc)) from exc

    existing = next((row for row in rows if row.constraint_id == constraint_id), None)
    if existing is None:
        raise APIError(
            ErrorCode.CONSTRAINT_NOT_FOUND,
            f"constraint {constraint_id} does not exist in session {session_id}",
        )
    if existing.status is SCStatus.REVOKED:
        return to_view(existing)

    updated = await context.store.set_status(session_id, constraint_id, SCStatus.REVOKED, None)
    context.audit.emit(
        session_id,
        tenant_id,
        AuditEventType.CONSTRAINT_REVOKED,
        constraint_id=constraint_id,
        turn_index=existing.source_turn_index,
        canonical_text=existing.canonical_text,
    )
    return to_view(updated)
