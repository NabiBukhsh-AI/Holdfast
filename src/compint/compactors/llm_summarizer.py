"""Prompt based LLM summarization compactor. TASK-013, FR-033.

Parameterized by (model, prompt). Three things here are load bearing:

1. `CRITICAL PLACEMENT FACT` spec 14.4 and PAPER SPECIFICATION 5.3: the compaction
   instruction IMMEDIATELY FOLLOWS the context to be compacted. That ordering is the paper's
   own explanation for why later injection positions retain better, because they sit closer
   to the compaction instruction. Reversing it would break the injection location finding, so
   a test asserts the instruction appears after the context.

2. The prompt is FETCHED, never written (TASK-001, unknowns U-01 and U-02). Constructing this
   compactor with an unfetched prompt raises PromptNotFetchedError from the registry. There is
   no fallback prompt and there must not be one.

3. Refusal, truncation, and overflow are distinct terminal statuses, never an empty summary.
"""

from __future__ import annotations

import re
import time

from compint.compactors.base import CompactionResult, failed_result
from compint.core.models import CompactionStatus, History
from compint.core.tokenization import Tokenizer
from shared.errors import (
    ContentFilterError,
    ContextOverflowError,
    ProviderError,
    ProviderRefusalError,
    ProviderTimeoutError,
)
from shared.llm_client import LLMClient, LLMRequest
from shared.prompts import Prompt

# PAPER SPECIFICATION Table 5 (spec 11.4): output wrapper by prompt.
DEFAULT_WRAPPERS: dict[str, str] = {
    "anthropic": "summary_tag",
    "anthropic_sc_targeted": "summary_tag",
    "pi_mono": "markdown",
}

_SUMMARY_TAG = re.compile(r"<summary>(.*?)</summary>", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", flags=re.DOTALL)


def strip_wrapper(text: str, wrapper: str) -> str:
    """Remove the compactor's output wrapper.

    `summary_tag` extracts the content of the first <summary> element. `markdown` unwraps a
    fenced block if the whole output is one. `none` returns the text unchanged. An output that
    is only a wrapper with nothing inside yields an empty string, which the caller must treat
    as COMPACTION_FAILED rather than as a summary of a context with no content.
    """
    if wrapper == "summary_tag":
        match = _SUMMARY_TAG.search(text)
        return match.group(1).strip() if match else text.strip()
    if wrapper == "markdown":
        match = _CODE_FENCE.match(text)
        return match.group(1).strip() if match else text.strip()
    if wrapper == "none":
        return text.strip()
    raise ValueError(f"unknown output wrapper {wrapper}")


def render_compaction_input(history: History, instruction: str) -> str:
    """Context first, instruction second. Spec 14.4, PAPER SPECIFICATION 5.3."""
    return f"{history.render()}\n\n{instruction}"


class LLMSummarizerCompactor:
    """(model, prompt) parameterized compactor over an OpenAI compatible endpoint."""

    def __init__(
        self,
        client: LLMClient,
        prompt: Prompt,
        model_id: str,
        tokenizer: Tokenizer,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 2048,
        timeout_s: float = 600.0,
        compactor_id: str | None = None,
        wrapper: str | None = None,
    ) -> None:
        self._client = client
        self._prompt = prompt
        self.model_id = model_id
        self.prompt_id = prompt.id
        self.prompt_hash = prompt.content_hash
        self.id = compactor_id or f"{model_id}__{prompt.id}"
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._top_p = top_p
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s
        self._wrapper = wrapper or str(
            getattr(prompt, "output_wrapper", None) or DEFAULT_WRAPPERS.get(prompt.id, "none")
        )

    @property
    def instruction(self) -> str:
        """The compaction instruction text, exactly as fetched."""
        body = self._prompt.text if self._prompt.text is not None else self._prompt.user
        if body is None:
            raise ProviderError(f"compaction prompt {self._prompt.id} carries no instruction")
        return body

    async def compact(self, history: History) -> CompactionResult:
        started = time.perf_counter()
        user = render_compaction_input(history, self.instruction)
        request = LLMRequest(
            model=self.model_id,
            system=self._prompt.system,
            user=user,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_output_tokens,
            timeout_s=self._timeout_s,
        )
        try:
            response = await self._client.complete(request)
        except ContextOverflowError as exc:
            return failed_result(
                self.id, self.model_id, CompactionStatus.OVERFLOW, history.token_count, str(exc)
            )
        except ProviderRefusalError as exc:
            # Spec 14.4: model refusal on WildChat unsafe content is a distinct terminal state.
            return failed_result(
                self.id, self.model_id, CompactionStatus.REFUSED, history.token_count, str(exc)
            )
        except ContentFilterError as exc:
            return failed_result(
                self.id, self.model_id, CompactionStatus.BLOCKED, history.token_count, str(exc)
            )
        except ProviderTimeoutError as exc:
            return failed_result(
                self.id, self.model_id, CompactionStatus.ERROR, history.token_count, str(exc)
            )

        summary = strip_wrapper(response.text, self._wrapper)
        if not summary.strip():
            # Spec 14.4: empty or wrapper only output is COMPACTION_FAILED, do not judge it.
            return failed_result(
                self.id,
                self.model_id,
                CompactionStatus.COMPACTION_FAILED,
                history.token_count,
                response.text,
            )
        status = (
            CompactionStatus.TRUNCATED
            if response.finish_reason == "length"
            else CompactionStatus.OK
        )
        return CompactionResult(
            text=summary,
            compactor_id=self.id,
            model_id=self.model_id,
            prompt_id=self.prompt_id,
            prompt_hash=self.prompt_hash,
            input_tokens=history.token_count,
            output_tokens=self._tokenizer.count(summary),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            status=status,
            raw=response.raw or response.text,
        )
