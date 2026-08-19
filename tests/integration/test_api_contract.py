"""TASK-027 API contract tests. Spec 18.

Every endpoint and every error code in the taxonomy is exercised. The service runs on the
in-memory backends with a stub extractor, so these tests need no database, no Redis, no GPU,
and no network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from compint.extractor.client import SCExtractor
from scguard.api.errors import ERROR_TAXONOMY, PROBLEM_CONTENT_TYPE, ErrorCode
from scguard.api.app import ServiceContext, build_context, create_app
from scguard.extractor.worker import ExtractionWorker
from shared.config import AppConfig, load_config
from shared.llm_client import StubLLMClient
from shared.prompts import Prompt

SESSION = "sess_api"
TENANT = "tenant_api"
AUTH = {"Authorization": "Bearer test-token", "X-Tenant-Id": TENANT}
AUDIT_AUTH = {**AUTH, "X-Scopes": "audit:read"}

USER_TURN = "Also, from now on don't send any emails on my behalf, just draft them."


def extraction_prompt() -> Prompt:
    return Prompt(
        id="sc_extractor",
        version="v1",
        provenance="fetched",
        source_url="https://example.invalid/repo",
        fetched_at="2026-08-19T00:00:00Z",
        user="EXTRACTION INSTRUCTIONS\n{inputs}",
    )


def extraction_payload(*items: dict[str, str]) -> str:
    return json.dumps(list(items))


DRAFT_CONSTRAINT = extraction_payload(
    {
        "canonical_text": "Draft emails instead of sending them.",
        "evidence_span": "don't send any emails on my behalf, just draft them",
        "category": "action",
    }
)


@pytest.fixture
def config(repo_root) -> AppConfig:  # type: ignore[no-untyped-def]
    return load_config(repo_root / "configs" / "production" / "dev.yaml")


@pytest.fixture
def context(config: AppConfig) -> ServiceContext:
    return build_context(
        config,
        llm_client=StubLLMClient(default_factory=lambda request: DRAFT_CONSTRAINT),
        extraction_prompt=extraction_prompt(),
    )


@pytest.fixture
def client(context: ServiceContext) -> Iterator[TestClient]:
    with TestClient(create_app(context)) as test_client:
        yield test_client


def submit_turn(client: TestClient, *, turn_index: int = 0, content: str = USER_TURN, role: str = "user"):
    return client.post(
        f"/v1/sessions/{SESSION}/turns",
        headers=AUTH,
        json={"turn_index": turn_index, "role": role, "content": content},
    )


# ---------------------------------------------------------------- conventions


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.headers["X-Request-Id"]


def test_supplied_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/v1/health", headers={"X-Request-Id": "req_supplied"})
    assert response.headers["X-Request-Id"] == "req_supplied"


def test_missing_bearer_token_is_401(client: TestClient) -> None:
    response = client.post(
        f"/v1/sessions/{SESSION}/turns",
        headers={"X-Tenant-Id": TENANT},
        json={"turn_index": 0, "role": "user", "content": "hello"},
    )
    assert response.status_code == 401
    assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
    assert response.json()["code"] == ErrorCode.UNAUTHORIZED.value


def test_missing_tenant_header_is_401(client: TestClient) -> None:
    response = client.post(
        f"/v1/sessions/{SESSION}/turns",
        headers={"Authorization": "Bearer t"},
        json={"turn_index": 0, "role": "user", "content": "hello"},
    )
    assert response.status_code == 401


def test_problem_documents_follow_rfc9457(client: TestClient) -> None:
    response = submit_turn(client, role="assistant")
    body = response.json()
    for field in ("type", "title", "status", "detail", "code"):
        assert field in body, f"problem document is missing {field}"
    assert body["type"].startswith("https://")
    assert body["status"] == response.status_code
    assert body["instance"] == f"/v1/sessions/{SESSION}/turns"


# ---------------------------------------------------------------- turns


def test_turns_rejects_assistant_role(client: TestClient) -> None:
    """INV-3 at the API boundary: a non user turn cannot enter the system at all."""
    response = submit_turn(client, role="assistant")
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == ErrorCode.INVALID_ROLE.value
    assert "INV-3" in body["detail"]


@pytest.mark.parametrize("role", ["system", "tool", "thinking", "ASSISTANT", ""])
def test_turns_rejects_every_non_user_role(client: TestClient, role: str) -> None:
    assert submit_turn(client, role=role).status_code == 400


def test_turn_submission_is_accepted_and_queued(client: TestClient) -> None:
    response = submit_turn(client)
    assert response.status_code == 202
    body = response.json()
    assert body["session_id"] == SESSION
    assert body["status"] == "queued"
    assert body["extraction_job_id"].startswith("job_")


def test_turn_submission_is_idempotent(client: TestClient) -> None:
    first = submit_turn(client).json()
    second = submit_turn(client).json()
    assert first["extraction_job_id"] == second["extraction_job_id"]


def test_same_index_different_content_is_409(client: TestClient) -> None:
    submit_turn(client, turn_index=3, content="first content")
    response = submit_turn(client, turn_index=3, content="different content")
    assert response.status_code == 409
    assert response.json()["code"] == ErrorCode.TURN_CONFLICT.value


def test_oversized_content_is_413(client: TestClient, config: AppConfig) -> None:
    response = submit_turn(client, content="x" * (config.service.max_content_bytes + 1))
    assert response.status_code == 413
    assert response.json()["code"] == ErrorCode.CONTENT_TOO_LARGE.value


def test_queue_backpressure_is_429_with_retry_after(context: ServiceContext) -> None:
    """Backpressure surfaces as 429, never as a silently dropped turn."""
    small = build_context(
        context.config.model_copy(
            update={"service": context.config.service.model_copy(update={"queue_max_depth": 1})}
        ),
        llm_client=StubLLMClient(default_factory=lambda request: "[]"),
        extraction_prompt=extraction_prompt(),
    )
    with TestClient(create_app(small)) as client:
        assert submit_turn(client, turn_index=0, content="one").status_code == 202
        response = submit_turn(client, turn_index=1, content="two")
        assert response.status_code == 429
        assert response.json()["code"] == ErrorCode.RATE_LIMITED.value
        assert response.json()["retryable"] is True
        assert response.headers["Retry-After"]


# ---------------------------------------------------------------- compact


async def process_queue(context: ServiceContext) -> None:
    worker: ExtractionWorker = context.worker
    await worker.drain_all()


def test_compact_attaches_the_registry(client: TestClient, context: ServiceContext) -> None:
    import asyncio

    submit_turn(client, turn_index=1)
    # TestClient runs the app in its own loop, so the worker is driven explicitly here rather
    # than by a background task, which keeps the assertion deterministic.
    asyncio.run(process_queue(context))

    response = client.post(
        f"/v1/sessions/{SESSION}/compact",
        headers=AUTH,
        json={"compaction_index": 0, "compacted_summary": "<summary>work so far</summary>"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "Draft emails instead of sending them." in body["augmented_context"]
    assert body["registry"]["injected_count"] == 1
    assert body["registry_incomplete"] is False


def test_incomplete_flag_in_header_and_body(client: TestClient) -> None:
    """NFR-008: a caller ignoring the body still has a chance to notice the header."""
    submit_turn(client, turn_index=2)  # left unprocessed on purpose
    response = client.post(
        f"/v1/sessions/{SESSION}/compact",
        headers=AUTH,
        json={
            "compaction_index": 0,
            "compacted_summary": "summary",
            "drain_timeout_ms": 10,
        },
    )
    assert response.status_code == 200
    assert response.json()["registry_incomplete"] is True
    assert response.headers["X-SC-Registry-Incomplete"] == "true"
    codes = {warning["code"] for warning in response.json()["warnings"]}
    assert "REGISTRY_INCOMPLETE" in codes


def test_compact_with_empty_registry_returns_bare_summary(client: TestClient) -> None:
    response = client.post(
        f"/v1/sessions/{SESSION}/compact",
        headers=AUTH,
        json={"compaction_index": 0, "compacted_summary": "just the summary"},
    )
    assert response.json()["augmented_context"] == "just the summary"


def test_invalid_budget_is_422(client: TestClient) -> None:
    response = client.post(
        f"/v1/sessions/{SESSION}/compact",
        headers=AUTH,
        json={
            "compaction_index": 0,
            "compacted_summary": "summary",
            "options": {"budget_tokens": 0},
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == ErrorCode.BUDGET_INVALID.value


def test_store_failure_is_503_not_an_empty_registry(
    client: TestClient, context: ServiceContext
) -> None:
    """Spec 18.2: the one error that must never degrade gracefully."""
    submit_turn(client)
    context.store.set_available(False)  # type: ignore[attr-defined]
    response = client.post(
        f"/v1/sessions/{SESSION}/compact",
        headers=AUTH,
        json={"compaction_index": 0, "compacted_summary": "summary"},
    )
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == ErrorCode.REGISTRY_UNAVAILABLE.value
    assert body["retryable"] is True


# ---------------------------------------------------------------- constraints


def test_add_list_and_revoke_constraint(client: TestClient) -> None:
    created = client.post(
        f"/v1/sessions/{SESSION}/constraints",
        headers=AUTH,
        json={"canonical_text": "Never delete files without confirmation.", "category": "action", "pinned": True},
    )
    assert created.status_code == 200
    constraint_id = created.json()["id"]
    assert created.json()["pinned"] is True

    listed = client.get(f"/v1/sessions/{SESSION}/constraints", headers=AUTH)
    assert listed.status_code == 200
    assert [c["id"] for c in listed.json()["constraints"]] == [constraint_id]

    revoked = client.request(
        "DELETE", f"/v1/sessions/{SESSION}/constraints/{constraint_id}", headers=AUTH
    )
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "revoked"

    after = client.get(f"/v1/sessions/{SESSION}/constraints", headers=AUTH).json()
    assert after["constraints"] == []
    assert [c["id"] for c in after["tombstoned"]] == [constraint_id], "tombstoned, not deleted"


def test_revocation_is_idempotent(client: TestClient) -> None:
    created = client.post(
        f"/v1/sessions/{SESSION}/constraints",
        headers=AUTH,
        json={"canonical_text": "Use metric units.", "category": "preference"},
    ).json()
    path = f"/v1/sessions/{SESSION}/constraints/{created['id']}"
    assert client.request("DELETE", path, headers=AUTH).status_code == 200
    assert client.request("DELETE", path, headers=AUTH).status_code == 200


def test_revoking_unknown_constraint_is_404(client: TestClient) -> None:
    client.post(
        f"/v1/sessions/{SESSION}/constraints",
        headers=AUTH,
        json={"canonical_text": "anything", "category": "other"},
    )
    response = client.request(
        "DELETE", f"/v1/sessions/{SESSION}/constraints/sc_missing", headers=AUTH
    )
    assert response.status_code == 404
    assert response.json()["code"] == ErrorCode.CONSTRAINT_NOT_FOUND.value


def test_listing_unknown_session_is_404(client: TestClient) -> None:
    response = client.get("/v1/sessions/sess_unknown/constraints", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["code"] == ErrorCode.SESSION_NOT_FOUND.value


def test_exclude_tombstoned(client: TestClient) -> None:
    created = client.post(
        f"/v1/sessions/{SESSION}/constraints",
        headers=AUTH,
        json={"canonical_text": "Cite primary sources.", "category": "preference"},
    ).json()
    client.request("DELETE", f"/v1/sessions/{SESSION}/constraints/{created['id']}", headers=AUTH)
    response = client.get(
        f"/v1/sessions/{SESSION}/constraints?include_tombstoned=false", headers=AUTH
    )
    assert response.json()["tombstoned"] == []


def test_point_in_time_reconstruction_endpoint(client: TestClient) -> None:
    """FR-082 through the API: GET /constraints?as_of_turn=N."""
    client.post(
        f"/v1/sessions/{SESSION}/constraints",
        headers=AUTH,
        json={"canonical_text": "Confirm before running commands.", "category": "action"},
    )
    response = client.get(f"/v1/sessions/{SESSION}/constraints?as_of_turn=0", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["as_of_turn"] == 0
    assert len(response.json()["constraints"]) == 1


# ---------------------------------------------------------------- audit


def test_audit_requires_a_separate_scope(client: TestClient) -> None:
    client.post(
        f"/v1/sessions/{SESSION}/constraints",
        headers=AUTH,
        json={"canonical_text": "Never send email.", "category": "action"},
    )
    denied = client.get(f"/v1/sessions/{SESSION}/audit", headers=AUTH)
    assert denied.status_code == 403
    assert denied.json()["code"] == ErrorCode.FORBIDDEN.value

    allowed = client.get(f"/v1/sessions/{SESSION}/audit", headers=AUDIT_AUTH)
    assert allowed.status_code == 200
    assert allowed.json()["total"] >= 1


def test_audit_filters_by_event_type(client: TestClient) -> None:
    client.post(
        f"/v1/sessions/{SESSION}/constraints",
        headers=AUTH,
        json={"canonical_text": "Never send email.", "category": "action"},
    )
    response = client.get(
        f"/v1/sessions/{SESSION}/audit?event_type=constraint_added", headers=AUDIT_AUTH
    )
    assert response.status_code == 200
    assert all(e["event_type"] == "constraint_added" for e in response.json()["events"])


# ---------------------------------------------------------------- health


def test_health_has_no_dependencies(client: TestClient, context: ServiceContext) -> None:
    context.store.set_available(False)  # type: ignore[attr-defined]
    context.extractor_reachable = False
    assert client.get("/v1/health").status_code == 200


def test_ready_returns_503_when_the_extractor_is_unreachable(
    client: TestClient, context: ServiceContext
) -> None:
    """A ready but non extracting replica silently produces empty registries."""
    assert client.get("/v1/ready").status_code == 200
    context.extractor_reachable = False
    response = client.get("/v1/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["checks"]["extractor"] == "unreachable"


def test_ready_reports_store_failure(client: TestClient, context: ServiceContext) -> None:
    context.store.set_available(False)  # type: ignore[attr-defined]
    response = client.get("/v1/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["registry_store"] == "unavailable"


# ---------------------------------------------------------------- taxonomy


def test_error_taxonomy_complete() -> None:
    """Spec 18.9: every declared code has a status and a retryability, and nothing is orphaned."""
    for code in ErrorCode:
        assert code in ERROR_TAXONOMY, f"{code} has no taxonomy entry"
        status, retryable, title = ERROR_TAXONOMY[code]
        assert 400 <= status <= 599
        assert isinstance(retryable, bool)
        assert title
    assert ERROR_TAXONOMY[ErrorCode.RATE_LIMITED][1] is True
    assert ERROR_TAXONOMY[ErrorCode.REGISTRY_UNAVAILABLE][1] is True
    assert ERROR_TAXONOMY[ErrorCode.INVALID_ROLE][1] is False


def test_openapi_schema_generates(client: TestClient) -> None:
    """TASK-027 acceptance: the OpenAPI schema is generated."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]
    for expected in (
        "/v1/sessions/{session_id}/turns",
        "/v1/sessions/{session_id}/compact",
        "/v1/sessions/{session_id}/constraints",
        "/v1/sessions/{session_id}/constraints/{constraint_id}",
        "/v1/sessions/{session_id}/audit",
        "/v1/health",
        "/v1/ready",
    ):
        assert expected in paths, f"{expected} is missing from the OpenAPI schema"
