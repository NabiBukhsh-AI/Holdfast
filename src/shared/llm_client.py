"""Async OpenAI compatible model client, plus the deterministic stub CI runs against.

Spec 17.2 and NFR-018: model serving is abstracted so vLLM, a hosted API, or a stub can be
swapped by config. Style rule 5: every function that touches an LLM is async and takes an
explicit timeout. Rule 13: no error path degrades quietly, so refusal, content filter,
overflow, and timeout are four distinct outcomes and none of them is an empty string.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from shared.errors import (
    ContentFilterError,
    ContextOverflowError,
    ProviderError,
    ProviderRefusalError,
    ProviderTimeoutError,
)

FinishReason = Literal["stop", "length", "content_filter", "refusal", "error"]


class LLMRequest(BaseModel):
    """One model call. Frozen so a request cannot be mutated between retry attempts."""

    model_config = ConfigDict(frozen=True)

    model: str
    user: str
    system: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout_s: float = 120.0
    # vLLM guided decoding. PRODUCTION IMPLEMENTATION for the extractor (spec 17.2).
    guided_json: dict[str, Any] | None = None
    # Qwen3.5 thinking toggle. PAPER SPECIFICATION FR-068: thinking disabled.
    thinking: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def cache_key(self) -> str:
        """Stable key over everything that can change the response."""
        payload = "\x00".join(
            [
                self.model,
                self.system or "",
                self.user,
                f"{self.temperature}",
                f"{self.top_p}",
                f"{self.max_tokens}",
                "guided" if self.guided_json else "free",
                f"thinking={self.thinking}",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    finish_reason: FinishReason = "stop"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    # Always retained. Spec 12.2: without the raw text, a parser bug found later cannot be
    # corrected without rerunning and re-paying for every call.
    raw: str = ""


class LLMClient(Protocol):
    """Every model role in the system (compactor, extractor, judge, probe) uses this."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    async def aclose(self) -> None: ...


class StubLLMClient:
    """Deterministic client for CI and unit tests. No network, no GPU, no spend.

    Spec 23.9: the LLM client is replaced by a deterministic stub keyed on input hash,
    returning fixtures. Registered fixtures win; otherwise `default_factory` is consulted;
    otherwise a stable synthetic string derived from the request hash is returned, which is
    enough for cache and call count assertions but is never mistaken for a real result.
    """

    def __init__(
        self,
        fixtures: Mapping[str, str] | None = None,
        default_factory: Callable[[LLMRequest], str] | None = None,
        *,
        model_label: str = "stub",
        latency_ms: float = 0.0,
    ) -> None:
        self._fixtures = dict(fixtures or {})
        self._default_factory = default_factory
        self._model_label = model_label
        self._latency_ms = latency_ms
        self.calls: list[LLMRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def register(self, key: str, response: str) -> None:
        """Bind a response to a request cache key or to a substring trigger."""
        self._fixtures[key] = response

    def reset(self) -> None:
        self.calls.clear()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        key = request.cache_key()
        if key in self._fixtures:
            text = self._fixtures[key]
        elif self._default_factory is not None:
            text = self._default_factory(request)
        else:
            digest = key[:16]
            text = f"STUB_RESPONSE:{digest}"
        if text == "__CONTENT_FILTER__":
            raise ContentFilterError(f"stub content filter for {key[:12]}")
        if text == "__REFUSAL__":
            raise ProviderRefusalError(f"stub refusal for {key[:12]}")
        if text == "__TIMEOUT__":
            raise ProviderTimeoutError(f"stub timeout for {key[:12]}")
        if text == "__OVERFLOW__":
            raise ContextOverflowError(f"stub overflow for {key[:12]}")
        return LLMResponse(
            text=text,
            model=f"{self._model_label}:{request.model}",
            input_tokens=len(request.user) // 4,
            output_tokens=len(text) // 4,
            latency_ms=self._latency_ms,
            raw=text,
        )

    async def aclose(self) -> None:
        return None


class OpenAICompatibleClient:
    """Talks to vLLM or any OpenAI compatible endpoint.

    One client per role, because the roles have different models, timeouts, and concurrency
    limits. Concurrency is bounded by a semaphore rather than left to the event loop, so a
    750 instance grid cannot open 750 sockets against a single GPU server.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        max_concurrency: int = 8,
        connect_timeout_s: float = 10.0,
    ) -> None:
        if not base_url:
            raise ProviderError("base_url is required for the OpenAI compatible client")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(connect=connect_timeout_s, read=None, write=None, pool=None),
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.user})
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if request.guided_json is not None:
            payload["guided_json"] = request.guided_json
        if request.thinking is not None:
            payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = request.thinking
        payload.update(request.extra)
        return payload

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        async with self._semaphore:
            try:
                response = await self._client.post(
                    "/v1/chat/completions",
                    json=self._payload(request),
                    timeout=httpx.Timeout(request.timeout_s),
                )
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(
                    f"{request.model} exceeded {request.timeout_s}s"
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderError(f"transport failure calling {request.model}: {exc}") from exc

        if response.status_code == 400 and "context" in response.text.lower():
            raise ContextOverflowError(
                f"{request.model} rejected the context: {response.text[:400]}"
            )
        if response.status_code in (403, 451):
            raise ContentFilterError(f"{request.model} content filter: {response.text[:400]}")
        if response.status_code >= 400:
            raise ProviderError(
                f"{request.model} returned HTTP {response.status_code}: {response.text[:400]}"
            )

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(f"{request.model} returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise ProviderRefusalError(f"{request.model} refused: {message['refusal'][:400]}")
        text = message.get("content")
        if text is None:
            raise ProviderError(f"{request.model} returned a null content field")
        finish = choice.get("finish_reason") or "stop"
        if finish == "content_filter":
            raise ContentFilterError(f"{request.model} finish_reason=content_filter")
        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            model=request.model,
            finish_reason=finish if finish in ("stop", "length") else "stop",
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            raw=text,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class CachingLLMClient:
    """Wraps a client with an in-process cache keyed on the full request.

    Spec 14.9: judge calls are cacheable on (prompt_hash, sc_hash, context_hash) and reruns
    are common. The cache is explicit rather than hidden inside a call site so that a test
    can assert exactly how many underlying calls a grid issued.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._cache: dict[str, LLMResponse] = {}
        self.hits = 0
        self.misses = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.cache_key()
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        response = await self._inner.complete(request)
        self._cache[key] = response
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()
