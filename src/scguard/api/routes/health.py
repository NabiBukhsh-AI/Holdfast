"""Liveness and readiness. TASK-027, spec 18.7.

`/health` is a liveness check with no dependencies. `/ready` verifies the registry store, the
queue, and the SLM endpoint, and returns 503 when the extractor is unreachable, because a
ready but non extracting replica silently produces empty registries, which is exactly the
failure this system exists to prevent.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, ConfigDict

from scguard.api.app import ServiceContext, get_context

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool
    checks: dict[str, str]
    detail: str = ""


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """No dependencies. A failing dependency must not take a live pod out of rotation."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request, response: Response) -> ReadyResponse:
    context: ServiceContext = get_context(request)
    checks: dict[str, str] = {}

    store_ok = bool(getattr(context.store, "available", True))
    checks["registry_store"] = "ok" if store_ok else "unavailable"

    depth = context.queue.depth
    queue_ok = depth < context.queue.capacity
    checks["queue"] = "ok" if queue_ok else f"full at {depth}/{context.queue.capacity}"

    extractor_ok = context.extractor_reachable
    checks["extractor"] = "ok" if extractor_ok else "unreachable"

    ready_now = store_ok and queue_ok and extractor_ok
    if not ready_now:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        ready=ready_now,
        checks=checks,
        detail=(
            ""
            if ready_now
            else "a replica that is ready but not extracting silently produces empty registries"
        ),
    )
