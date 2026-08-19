"""POST /v1/sessions/{session_id}/compact. TASK-027, spec 18.2, Equation 10.

The caller supplies the compactor output; SC-GUARD does not run the compactor. What it does is
drain pending extractions, read the registry, enforce the budget, and attach.

`registry_incomplete` is returned in the body AND as `X-SC-Registry-Incomplete`, so a caller
that ignores the body still has a chance to notice. NFR-008 requires this never be silent.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from scguard.api.app import ServiceContext, get_context, require_tenant
from scguard.api.errors import APIError, ErrorCode
from scguard.assembly.service import AssemblyWarning, RegistrySummary

router = APIRouter(tags=["compact"])


class CompactOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assembly_mode: str | None = None
    budget_tokens: int | None = None


class CompactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compaction_index: int = Field(ge=0)
    compacted_summary: str = Field(min_length=1)
    drain_timeout_ms: int | None = Field(default=None, ge=0, le=60_000)
    options: CompactOptions = CompactOptions()


class CompactResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    augmented_context: str
    registry: RegistrySummary
    warnings: tuple[AssemblyWarning, ...]
    registry_incomplete: bool
    metadata: dict[str, str]


@router.post("/sessions/{session_id}/compact", response_model=CompactResponse)
async def compact(
    session_id: str, payload: CompactRequest, request: Request, response: Response
) -> CompactResponse:
    tenant_id = require_tenant(request)
    context: ServiceContext = get_context(request)

    if payload.options.budget_tokens is not None and payload.options.budget_tokens <= 0:
        raise APIError(
            ErrorCode.BUDGET_INVALID,
            f"budget_tokens must be positive, got {payload.options.budget_tokens}",
        )
    if payload.options.assembly_mode not in (None, "bare", "delimited"):
        raise APIError(
            ErrorCode.VALIDATION_ERROR,
            f"assembly_mode must be bare or delimited, got {payload.options.assembly_mode!r}",
        )

    await context.ensure_session(session_id, tenant_id)

    result = await context.assembly.compact(
        session_id,
        tenant_id,
        compaction_index=payload.compaction_index,
        compacted_summary=payload.compacted_summary,
        drain_timeout_ms=payload.drain_timeout_ms,
        assembly_mode=payload.options.assembly_mode,  # type: ignore[arg-type]
        budget_tokens=payload.options.budget_tokens,
    )

    # Body and header, so a caller ignoring one still sees the other.
    response.headers["X-SC-Registry-Incomplete"] = str(result.registry_incomplete).lower()
    metadata = dict(result.metadata)
    metadata["drain_wait_ms"] = str(result.drain_wait_ms)
    metadata["assembly_ms"] = str(result.assembly_ms)

    return CompactResponse(
        augmented_context=result.augmented_context,
        registry=result.registry,
        warnings=result.warnings,
        registry_incomplete=result.registry_incomplete,
        metadata=metadata,
    )
