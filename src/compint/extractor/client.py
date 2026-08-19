"""SC extractor client. TASK-019, Algorithm 14.5, FR-060 through FR-068.

Qwen3.5-9B with thinking disabled, training free, prompting only.

`INV-3` is enforced at the call site: `extract()` takes a user message string plus an optional
previous assistant message that is passed under a separate, differently labelled envelope tag.
There is no parameter through which a history could be handed to this class, so the extractor
cannot be accidentally fed assistant turns as extraction sources.

`NFR-008` If the SLM is unavailable, the result is EXTRACTION_FAILED. It is never an empty
list. Treating an outage as "no constraints found" would silently recreate the paper's exact
failure mode inside the mitigation.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from compint.extractor.parser import ExtractionResult, ExtractionStatus, parse_extraction
from compint.extractor.prompt_builder import build_messages
from shared.errors import ProviderError, ProviderTimeoutError
from shared.llm_client import LLMClient, LLMRequest
from shared.prompts import Prompt

# vLLM guided decoding schema. PRODUCTION IMPLEMENTATION per spec 17.2: unconstrained JSON
# from a 9B model fails to parse at a non trivial rate, and extraction reliability directly
# determines the headline retention number. The research arm runs unconstrained so the
# difference between the two is measured rather than assumed.
EXTRACTION_JSON_SCHEMA: dict[str, object] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "canonical_text": {"type": "string"},
            "evidence_span": {"type": "string"},
            "category": {
                "type": "string",
                "enum": ["action", "information", "process", "preference", "output", "other"],
            },
        },
        "required": ["canonical_text", "evidence_span", "category"],
        "additionalProperties": False,
    },
}


class ExtractionCall(BaseModel):
    """One extraction attempt plus its cost and timing, for NFR-001 and NFR-011."""

    model_config = ConfigDict(frozen=True)

    result: ExtractionResult
    latency_ms: float
    attempts: int
    model: str
    prompt_hash: str
    guided_json: bool


class SCExtractor:
    """Per user turn constraint detection."""

    def __init__(
        self,
        client: LLMClient,
        prompt: Prompt | None,
        model: str,
        *,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        max_tokens: int = 512,
        guided_json: bool = False,
        max_retries: int = 2,
        allow_other_category: bool = True,
        retry_backoff_s: float = 0.05,
    ) -> None:
        self._client = client
        self._prompt = prompt
        self._model = model
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._guided_json = guided_json
        self._max_retries = max_retries
        self._allow_other_category = allow_other_category
        self._retry_backoff_s = retry_backoff_s

    @property
    def prompt_hash(self) -> str:
        return self._prompt.content_hash if self._prompt is not None else "unfetched"

    async def extract(
        self,
        current_user_message: str,
        previous_assistant_message: str | None = None,
        registry_texts: Sequence[str] = (),
    ) -> ExtractionCall:
        """Algorithm 14.5. Exactly three inputs. Never the whole history."""
        system, user = build_messages(
            self._prompt, current_user_message, previous_assistant_message, registry_texts
        )
        request = LLMRequest(
            model=self._model,
            system=system,
            user=user,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout_s=self._timeout_s,
            guided_json=EXTRACTION_JSON_SCHEMA if self._guided_json else None,
            # PAPER SPECIFICATION FR-068: thinking disabled.
            thinking=False,
        )

        started = time.perf_counter()
        attempts = 0
        last_error = ""
        # NFR-010: bounded retries with jittered backoff, then a recorded terminal failure.
        while attempts <= self._max_retries:
            attempts += 1
            try:
                response = await self._client.complete(request)
            except (ProviderTimeoutError, ProviderError) as exc:
                last_error = str(exc)
                if attempts > self._max_retries:
                    break
                await asyncio.sleep(self._retry_backoff_s * attempts)
                continue

            result = parse_extraction(
                response.raw or response.text,
                current_user_message,
                allow_other_category=self._allow_other_category,
            )
            if result.status is ExtractionStatus.EXTRACTION_PARSE_ERROR and attempts <= self._max_retries:
                last_error = result.detail
                await asyncio.sleep(self._retry_backoff_s * attempts)
                continue
            return ExtractionCall(
                result=result,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                attempts=attempts,
                model=self._model,
                prompt_hash=self.prompt_hash,
                guided_json=self._guided_json,
            )

        return ExtractionCall(
            result=ExtractionResult(
                status=ExtractionStatus.EXTRACTION_FAILED,
                raw_response="",
                detail=(
                    f"extractor unavailable after {attempts} attempts: {last_error}. "
                    "This is NOT an empty constraint list (NFR-008)."
                ),
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            attempts=attempts,
            model=self._model,
            prompt_hash=self.prompt_hash,
            guided_json=self._guided_json,
        )
