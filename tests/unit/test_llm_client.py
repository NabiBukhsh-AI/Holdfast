"""LLM client tests. Spec 17.2, NFR-018, style rule 5.

The OpenAI compatible client is exercised against an httpx MockTransport rather than a live
endpoint, so every provider failure mode this system distinguishes gets a real test without a
network call. That distinction matters: refusal, content filter, overflow and timeout are four
different terminal states here, and collapsing them would hide exactly the information the
evaluation needs to exclude a sample honestly.
"""

from __future__ import annotations

import httpx
import pytest

from shared.errors import (
    ContentFilterError,
    ContextOverflowError,
    ProviderError,
    ProviderRefusalError,
    ProviderTimeoutError,
)
from shared.llm_client import (
    CachingLLMClient,
    LLMRequest,
    OpenAICompatibleClient,
    StubLLMClient,
)


def build_client(handler, **kwargs) -> OpenAICompatibleClient:  # type: ignore[no-untyped-def]
    client = OpenAICompatibleClient("https://models.invalid", api_key="test-key", **kwargs)
    client._client = httpx.AsyncClient(
        base_url="https://models.invalid",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return client


def chat_response(content: str, finish_reason: str = "stop", **extra: object) -> httpx.Response:
    message: dict[str, object] = {"role": "assistant", "content": content}
    message.update(extra)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 8},
        },
    )


REQUEST = LLMRequest(model="gpt-oss-120b", user="summarize this", system="you help")


async def test_successful_completion_carries_usage_and_latency() -> None:
    client = build_client(lambda request: chat_response("a summary"))
    response = await client.complete(REQUEST)
    assert response.text == "a summary"
    assert response.input_tokens == 120
    assert response.output_tokens == 8
    assert response.finish_reason == "stop"
    assert response.raw == "a summary"
    assert response.latency_ms >= 0.0
    await client.aclose()


async def test_payload_carries_system_and_user_in_order() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return chat_response("ok")

    client = build_client(handler)
    await client.complete(REQUEST)
    messages = seen["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]  # type: ignore[index,union-attr]
    assert seen["model"] == "gpt-oss-120b"
    assert seen["temperature"] == 0.0
    await client.aclose()


async def test_system_is_omitted_when_absent() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return chat_response("ok")

    client = build_client(handler)
    await client.complete(LLMRequest(model="m", user="just a user turn"))
    assert [m["role"] for m in seen["messages"]] == ["user"]  # type: ignore[index,union-attr]
    await client.aclose()


async def test_guided_json_and_thinking_reach_the_provider() -> None:
    """Guided decoding is the production extractor path; thinking disabled is FR-068."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return chat_response("[]")

    client = build_client(handler)
    await client.complete(
        LLMRequest(model="qwen3.5-9b", user="turn", guided_json={"type": "array"}, thinking=False)
    )
    assert seen["guided_json"] == {"type": "array"}
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    await client.aclose()


async def test_truncated_generation_is_visible_as_finish_reason_length() -> None:
    """U-17: truncation would confound results, so it must not look like a clean stop."""
    client = build_client(lambda request: chat_response("half a summ", finish_reason="length"))
    response = await client.complete(REQUEST)
    assert response.finish_reason == "length"
    await client.aclose()


async def test_context_overflow_is_its_own_error() -> None:
    client = build_client(
        lambda request: httpx.Response(400, text="context length exceeded for this model")
    )
    with pytest.raises(ContextOverflowError):
        await client.complete(REQUEST)
    await client.aclose()


@pytest.mark.parametrize("status", [403, 451])
async def test_content_filter_status_codes(status: int) -> None:
    """Spec 6.8: a filtered sample is excluded with a count, not treated as a transport error."""
    client = build_client(lambda request: httpx.Response(status, text="blocked"))
    with pytest.raises(ContentFilterError):
        await client.complete(REQUEST)
    await client.aclose()


async def test_content_filter_finish_reason() -> None:
    client = build_client(lambda request: chat_response("", finish_reason="content_filter"))
    with pytest.raises(ContentFilterError):
        await client.complete(REQUEST)
    await client.aclose()


async def test_explicit_refusal_field_is_a_refusal_not_an_empty_answer() -> None:
    client = build_client(lambda request: chat_response("", refusal="I cannot help with that"))
    with pytest.raises(ProviderRefusalError, match="cannot help"):
        await client.complete(REQUEST)
    await client.aclose()


async def test_server_error_surfaces_with_its_body() -> None:
    client = build_client(lambda request: httpx.Response(500, text="upstream exploded"))
    with pytest.raises(ProviderError, match="HTTP 500"):
        await client.complete(REQUEST)
    await client.aclose()


async def test_empty_choices_is_an_error_not_an_empty_string() -> None:
    """An empty string would flow onward looking like a successful compaction of nothing."""
    client = build_client(lambda request: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ProviderError, match="no choices"):
        await client.complete(REQUEST)
    await client.aclose()


async def test_null_content_is_an_error() -> None:
    client = build_client(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
        )
    )
    with pytest.raises(ProviderError, match="null content"):
        await client.complete(REQUEST)
    await client.aclose()


async def test_timeout_is_distinguished_from_transport_failure() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = build_client(timeout)
    with pytest.raises(ProviderTimeoutError, match="exceeded"):
        await client.complete(REQUEST)
    await client.aclose()


async def test_transport_failure_is_a_provider_error() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = build_client(broken)
    with pytest.raises(ProviderError, match="transport failure"):
        await client.complete(REQUEST)
    await client.aclose()


async def test_concurrency_is_bounded_by_the_semaphore() -> None:
    """A 750 instance grid must not open 750 sockets against one GPU server."""
    import asyncio

    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return chat_response("ok")

    client = build_client(handler, max_concurrency=2)
    await asyncio.gather(*(client.complete(REQUEST) for _ in range(8)))
    assert peak <= 2, f"semaphore did not bound concurrency, peak was {peak}"
    await client.aclose()


def test_client_requires_a_base_url() -> None:
    with pytest.raises(ProviderError, match="base_url is required"):
        OpenAICompatibleClient("")


# ---------------------------------------------------------------- cache and stub


async def test_caching_client_collapses_identical_requests() -> None:
    inner = StubLLMClient(default_factory=lambda _r: "verdict")
    cached = CachingLLMClient(inner)
    await cached.complete(REQUEST)
    await cached.complete(REQUEST)
    assert inner.call_count == 1
    assert cached.hits == 1
    assert cached.misses == 1
    await cached.aclose()


async def test_caching_client_separates_different_requests() -> None:
    inner = StubLLMClient(default_factory=lambda _r: "verdict")
    cached = CachingLLMClient(inner)
    await cached.complete(REQUEST)
    await cached.complete(REQUEST.model_copy(update={"user": "a different context"}))
    assert inner.call_count == 2
    assert cached.hits == 0


def test_cache_key_separates_every_response_changing_field() -> None:
    """Two requests that could produce different answers must not share a cache entry."""
    base = REQUEST
    variants = {
        "model": base.model_copy(update={"model": "other-model"}),
        "system": base.model_copy(update={"system": "different system"}),
        "user": base.model_copy(update={"user": "different user"}),
        "temperature": base.model_copy(update={"temperature": 0.7}),
        "top_p": base.model_copy(update={"top_p": 0.9}),
        "max_tokens": base.model_copy(update={"max_tokens": 4096}),
        "guided": base.model_copy(update={"guided_json": {"type": "array"}}),
        "thinking": base.model_copy(update={"thinking": True}),
    }
    for label, variant in variants.items():
        assert variant.cache_key() != base.cache_key(), f"{label} does not change the cache key"


def test_stub_registers_responses_by_cache_key() -> None:
    stub = StubLLMClient()
    stub.register(REQUEST.cache_key(), "registered answer")
    assert stub._fixtures[REQUEST.cache_key()] == "registered answer"


async def test_stub_reset_clears_the_call_log() -> None:
    stub = StubLLMClient(default_factory=lambda _r: "ok")
    await stub.complete(REQUEST)
    assert stub.call_count == 1
    stub.reset()
    assert stub.call_count == 0


async def test_stub_default_response_is_stable_and_labelled() -> None:
    """The synthetic fallback must never be mistaken for a real model answer."""
    stub = StubLLMClient()
    first = await stub.complete(REQUEST)
    second = await stub.complete(REQUEST)
    assert first.text == second.text
    assert first.text.startswith("STUB_RESPONSE:")


@pytest.mark.parametrize(
    "sentinel,expected",
    [
        ("__CONTENT_FILTER__", ContentFilterError),
        ("__REFUSAL__", ProviderRefusalError),
        ("__TIMEOUT__", ProviderTimeoutError),
        ("__OVERFLOW__", ContextOverflowError),
    ],
)
async def test_stub_sentinels_raise_the_matching_error(sentinel: str, expected: type) -> None:
    stub = StubLLMClient(default_factory=lambda _r: sentinel)
    with pytest.raises(expected):
        await stub.complete(REQUEST)
