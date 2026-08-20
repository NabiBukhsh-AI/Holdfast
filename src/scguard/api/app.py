"""SC-GUARD HTTP surface. TASK-027, spec 18.

Base path `/v1`. Bearer token plus `X-Tenant-Id`. Every response carries `X-Request-Id`.
Errors are RFC 9457 problem documents.

The app is assembled from an explicit `ServiceContext` rather than module level globals, so a
test can build a fully wired service with in-memory backends and no network in one call.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from compint.extractor.client import SCExtractor
from scguard.api.errors import PROBLEM_CONTENT_TYPE, APIError, ErrorCode
from scguard.assembly.service import AssemblyService
from scguard.audit.emitter import AuditEmitter
from scguard.extractor.queue import ExtractionQueue
from scguard.extractor.worker import ExtractionWorker
from scguard.registry.conflicts import Adjudicator, HeuristicAdjudicator
from scguard.registry.dedup import RegistryUpdater
from scguard.registry.store import (
    InMemoryRegistryStore,
    RegistryStore,
    RegistryUnavailableError,
    Session,
    SessionNotFoundError,
)
from shared.config import AppConfig
from shared.errors import BudgetNotConfiguredError, ConfigError
from shared.llm_client import LLMClient, StubLLMClient
from shared.prompts import Prompt

API_PREFIX = "/v1"


@dataclass
class ServiceContext:
    """Everything a request handler needs. Constructed once at startup."""

    config: AppConfig
    store: RegistryStore
    queue: ExtractionQueue
    audit: AuditEmitter
    updater: RegistryUpdater
    worker: ExtractionWorker
    assembly: AssemblyService
    extractor_reachable: bool = True

    async def ensure_session(self, session_id: str, tenant_id: str) -> Session:
        """Sessions are created on first turn submission, never on a read path."""
        try:
            return await self.store.get_session(session_id)
        except SessionNotFoundError:
            return await self.store.create_session(
                Session(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    extractor_model=self.config.extractor.model,
                    prompt_hash="unfetched",
                    expires_at=datetime.now(UTC)
                    + timedelta(days=self.config.service.session_ttl_days),
                )
            )


def build_context(
    config: AppConfig,
    *,
    llm_client: LLMClient | None = None,
    extraction_prompt: Prompt | None = None,
    adjudicator: Adjudicator | None = None,
    store: RegistryStore | None = None,
) -> ServiceContext:
    """Wire the service. Defaults are the in-memory, no-network backends used by dev and CI.

    A caller that has already connected a Postgres store and a Redis cache passes the wrapped
    store in; `build_context_async` does exactly that. Keeping this function synchronous means
    the dev and CI path stays a single call with no event loop required.
    """
    # Fails at startup rather than at the first compaction (spec 14.7).
    budget = config.registry.require_budget_tokens()

    store = store if store is not None else InMemoryRegistryStore()
    audit = AuditEmitter()
    queue = ExtractionQueue(capacity=config.service.queue_max_depth)
    client = llm_client or StubLLMClient(default_factory=lambda _request: "[]")
    extractor = SCExtractor(
        client,
        extraction_prompt,
        config.extractor.model,
        temperature=config.extractor.temperature,
        timeout_s=config.extractor.timeout_s,
        guided_json=config.extractor.guided_json,
        max_retries=config.extractor.max_retries,
        allow_other_category=config.catalog.allow_other_category,
    )
    updater = RegistryUpdater(
        store,
        audit,
        mode=config.registry.mode,
        tau_dup=config.registry.tau_dup,
        adjudicator=(
            adjudicator
            if adjudicator is not None
            else (HeuristicAdjudicator() if config.registry.conflict_detection else None)
        ),
        extractor_model=config.extractor.model,
    )
    worker = ExtractionWorker(queue, extractor, updater, audit)
    assembly = AssemblyService(
        store,
        queue,
        audit,
        assembly_mode=config.assembly.mode,
        budget_tokens=budget,
        drain_timeout_ms=config.service.drain_timeout_ms,
        shadow_mode=config.service.shadow_mode,
        extractor_model=config.extractor.model,
    )
    return ServiceContext(
        config=config,
        store=store,
        queue=queue,
        audit=audit,
        updater=updater,
        worker=worker,
        assembly=assembly,
    )


async def build_context_async(
    config: AppConfig,
    *,
    llm_client: LLMClient | None = None,
    extraction_prompt: Prompt | None = None,
    adjudicator: Adjudicator | None = None,
) -> ServiceContext:
    """Connect the configured backends, then wire the service.

    Backend selection is config, never inference. `memory` is the dev and CI path and needs no
    services running; `postgres` and `redis` are the deployable path. A configured backend with
    a missing DSN fails at startup rather than at the first request, because a service that
    accepts traffic and then cannot persist a constraint is worse than one that refuses to
    start.
    """
    import os

    from scguard.registry.cache import CachedRegistryStore, build_cache

    store: RegistryStore
    if config.database.backend == "postgres":
        from scguard.registry.postgres_store import PostgresRegistryStore

        dsn = os.environ.get(config.database.dsn_env)
        if not dsn:
            raise ConfigError(
                f"database.backend is postgres but {config.database.dsn_env} is unset. "
                "Set it, or use backend=memory for dev and CI."
            )
        store = await PostgresRegistryStore.connect(dsn)
    else:
        store = InMemoryRegistryStore()

    if config.redis.backend == "redis":
        redis_dsn = os.environ.get(config.redis.dsn_env)
        if not redis_dsn:
            raise ConfigError(
                f"redis.backend is redis but {config.redis.dsn_env} is unset. "
                "Set it, or use backend=memory for dev and CI."
            )
        store = CachedRegistryStore(store, build_cache("redis", redis_dsn))
    elif config.redis.backend == "memory" and config.database.backend == "postgres":
        # A process local cache in front of a shared database is still correct, because every
        # mutation goes through this process and invalidates it. It is simply not shared.
        store = CachedRegistryStore(store, build_cache("memory"))

    return build_context(
        config,
        llm_client=llm_client,
        extraction_prompt=extraction_prompt,
        adjudicator=adjudicator,
        store=store,
    )


def require_tenant(request: Request) -> str:
    """Bearer token plus X-Tenant-Id. Spec 18 conventions."""
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer ") or not authorization[7:].strip():
        raise APIError(ErrorCode.UNAUTHORIZED, "a bearer token is required")
    tenant = request.headers.get("X-Tenant-Id", "").strip()
    if not tenant:
        raise APIError(ErrorCode.UNAUTHORIZED, "X-Tenant-Id is required")
    return tenant


def require_scope(request: Request, scope: str) -> None:
    """Audit reads need a scope separate from ordinary session scopes (spec 18.6)."""
    granted = {s.strip() for s in request.headers.get("X-Scopes", "").split(",") if s.strip()}
    if scope not in granted:
        raise APIError(
            ErrorCode.FORBIDDEN,
            f"this endpoint requires the {scope} scope, which is separate from session scopes",
        )


def create_app(context: ServiceContext) -> FastAPI:
    """Build the ASGI app around an already wired context."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.context = context
        yield

    app = FastAPI(
        title="SC-GUARD",
        version="0.1.0",
        summary="Session constraint registry and compaction-time assembly",
        lifespan=lifespan,
    )
    app.state.context = context

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        request_id = request.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    def problem_response(error: APIError, request: Request) -> JSONResponse:
        problem = error.to_problem().model_copy(
            update={"instance": error.instance or str(request.url.path)}
        )
        headers = {"X-Request-Id": getattr(request.state, "request_id", "")}
        if error.code is ErrorCode.RATE_LIMITED:
            headers["Retry-After"] = "1"
        return JSONResponse(
            status_code=error.status,
            content=problem.model_dump(mode="json"),
            media_type=PROBLEM_CONTENT_TYPE,
            headers=headers,
        )

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        return problem_response(exc, request)

    @app.exception_handler(RegistryUnavailableError)
    async def handle_registry_unavailable(
        request: Request, exc: RegistryUnavailableError
    ) -> JSONResponse:
        # Spec 18.2: 503, never an empty registry.
        return problem_response(APIError(ErrorCode.REGISTRY_UNAVAILABLE, str(exc)), request)

    @app.exception_handler(SessionNotFoundError)
    async def handle_session_not_found(request: Request, exc: SessionNotFoundError) -> JSONResponse:
        return problem_response(APIError(ErrorCode.SESSION_NOT_FOUND, str(exc)), request)

    @app.exception_handler(BudgetNotConfiguredError)
    async def handle_budget_invalid(
        request: Request, exc: BudgetNotConfiguredError
    ) -> JSONResponse:
        return problem_response(APIError(ErrorCode.BUDGET_INVALID, str(exc)), request)

    from scguard.api.routes import audit as audit_routes
    from scguard.api.routes import compact as compact_routes
    from scguard.api.routes import constraints as constraint_routes
    from scguard.api.routes import health as health_routes
    from scguard.api.routes import turns as turn_routes

    app.include_router(turn_routes.router, prefix=API_PREFIX)
    app.include_router(compact_routes.router, prefix=API_PREFIX)
    app.include_router(constraint_routes.router, prefix=API_PREFIX)
    app.include_router(audit_routes.router, prefix=API_PREFIX)
    app.include_router(health_routes.router, prefix=API_PREFIX)
    return app


def get_context(request: Request) -> ServiceContext:
    context = getattr(request.app.state, "context", None)
    if not isinstance(context, ServiceContext):
        raise APIError(ErrorCode.REGISTRY_UNAVAILABLE, "service context is not initialized")
    return context
