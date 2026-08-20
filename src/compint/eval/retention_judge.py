"""Retention judge. TASK-014, Equation 6, Algorithm 14.9.

    Retain(s, C(H^t_{s,I})) in {0, 1}                                       (6)

Three properties are load bearing:

1. **INV-4 at the type level.** `judge()` accepts `CompactedContext` and nothing else. An
   `InjectedHistory` cannot reach it. `CRITICAL NOTATION NOTE` spec 6.8: Equation 6 as printed
   applies Retain to the INJECTED context, but the surrounding prose and the judge prompt make
   clear the judged object is the COMPACTED context. Judging the injected context would be
   trivially 1 by construction, since injection is what put the SC there.

2. **Strict parsing.** The prompt says "Output only YES or NO." Anything else is a parse
   FAILURE recorded as UNPARSEABLE, never coerced to 0 or 1. Silently coercing unparseable
   verdicts to 0 would inflate the headline finding.

3. **Content filter rejections are a first class terminal state.** Spec 6.8: this is a
   reproducible operational hazard of WildChat, not an incidental error. BLOCKED records are
   counted, excluded from denominators, and printed alongside every affected metric.
"""

from __future__ import annotations

import hashlib
import time
from typing import Literal

from compint.core.models import CompactedContext, CompactionStatus, FramedSC, SCCategoryId
from compint.eval.records import RetentionRecord, RetentionStatus
from shared.errors import ContentFilterError, ProviderError, ProviderTimeoutError
from shared.llm_client import LLMClient, LLMRequest
from shared.prompts import Prompt

Verdict = Literal["YES", "NO"]


def parse_verdict_strict(raw: str) -> Verdict | None:
    """Exactly YES or NO after stripping surrounding whitespace. Nothing else parses."""
    candidate = raw.strip().upper()
    if candidate == "YES":
        return "YES"
    if candidate == "NO":
        return "NO"
    return None


def parse_verdict_normalized(raw: str) -> Verdict | None:
    """Lenient parse: strip punctuation and take the first token.

    ENGINEERING RECOMMENDATION spec 14.9. Recorded ALONGSIDE the strict verdict, never instead
    of it, so the effect of parser leniency on the headline number is measurable rather than
    assumed. Only the strict verdict feeds a reported rate (see DEVIATIONS.md D-02).
    """
    candidate = raw.strip().upper().lstrip("*_ \t\n")
    if not candidate:
        return None
    first = candidate.replace(",", " ").replace(".", " ").replace(":", " ").split()
    if not first:
        return None
    token = first[0]
    if token == "YES":
        return "YES"
    if token == "NO":
        return "NO"
    return None


def judge_cache_key(prompt_hash: str, sc_text: str, compacted: CompactedContext) -> str:
    """Spec 14.9: cacheable on (prompt_hash, sc_hash, context_hash). Reruns are common."""
    sc_hash = hashlib.sha256(sc_text.encode("utf-8")).hexdigest()
    payload = f"{prompt_hash}\x00{sc_hash}\x00{compacted.context_hash()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RetentionJudge:
    """GPT-5.4 as judge, over the compacted context only."""

    def __init__(
        self,
        client: LLMClient,
        prompt: Prompt,
        model: str,
        *,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        max_tokens: int = 8,
        record_normalized: bool = True,
    ) -> None:
        self._client = client
        self._prompt = prompt
        self._model = model
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._record_normalized = record_normalized
        self._cache: dict[str, RetentionRecord] = {}
        self.cache_hits = 0

    @property
    def prompt_hash(self) -> str:
        return self._prompt.content_hash

    async def judge(
        self,
        framed_sc: FramedSC,
        compacted: CompactedContext,
        *,
        instance_id: str,
        degenerate: bool = False,
    ) -> RetentionRecord:
        """One binary retention judgment on C(H^t_{s,I}).

        The parameter type is `CompactedContext`, which is how INV-4 is enforced: there is no
        overload accepting an injected or uncompacted history.
        """
        if compacted.status is not CompactionStatus.OK or not compacted.text.strip():
            # Spec 14.9 edge case: do not call the judge on a failed compaction.
            return RetentionRecord(
                instance_id=instance_id,
                sc_id=framed_sc.sc_id,
                category=framed_sc.category,
                compactor_id=compacted.compactor_id,
                compacted_hash=compacted.context_hash(),
                status=RetentionStatus.COMPACTION_FAILED,
                judge_model=self._model,
                judge_prompt_hash=self.prompt_hash,
                raw_response=compacted.raw[:2000],
                degenerate=degenerate,
            )

        key = judge_cache_key(self.prompt_hash, framed_sc.rendered_text, compacted)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached.model_copy(update={"instance_id": instance_id, "degenerate": degenerate})

        system, user = self._prompt.render(
            injected_sc=framed_sc.rendered_text, compacted_context=compacted.text
        )
        request = LLMRequest(
            model=self._model,
            system=system,
            user=user,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout_s=self._timeout_s,
        )

        started = time.perf_counter()
        try:
            response = await self._client.complete(request)
        except ContentFilterError as exc:
            record = self._terminal(
                framed_sc, compacted, instance_id, RetentionStatus.BLOCKED, str(exc), degenerate
            )
            self._cache[key] = record
            return record
        except (ProviderTimeoutError, ProviderError) as exc:
            # Not cached: a transport failure is not a property of the input and must be
            # retryable on the next run.
            return self._terminal(
                framed_sc, compacted, instance_id, RetentionStatus.ERROR, str(exc), degenerate
            )

        strict = parse_verdict_strict(response.text)
        normalized = parse_verdict_normalized(response.text) if self._record_normalized else None
        record = RetentionRecord(
            instance_id=instance_id,
            sc_id=framed_sc.sc_id,
            category=framed_sc.category,
            compactor_id=compacted.compactor_id,
            compacted_hash=compacted.context_hash(),
            verdict=strict,
            status=RetentionStatus.OK if strict is not None else RetentionStatus.UNPARSEABLE,
            judge_model=self._model,
            judge_prompt_hash=self.prompt_hash,
            # Always retained. Spec 12.2: a parser bug found later cannot be corrected without
            # rerunning and re-paying for every judge call.
            raw_response=response.raw or response.text,
            normalized_verdict=normalized,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            degenerate=degenerate,
        )
        self._cache[key] = record
        return record

    def _terminal(
        self,
        framed_sc: FramedSC,
        compacted: CompactedContext,
        instance_id: str,
        status: RetentionStatus,
        detail: str,
        degenerate: bool,
    ) -> RetentionRecord:
        return RetentionRecord(
            instance_id=instance_id,
            sc_id=framed_sc.sc_id,
            category=framed_sc.category,
            compactor_id=compacted.compactor_id,
            compacted_hash=compacted.context_hash(),
            status=status,
            judge_model=self._model,
            judge_prompt_hash=self.prompt_hash,
            raw_response=detail[:2000],
            degenerate=degenerate,
        )


def parser_leniency_delta(records: list[RetentionRecord]) -> dict[str, int]:
    """How many verdicts the lenient parser would have recovered. DEVIATIONS.md D-02.

    Reported as a diagnostic so the cost of strict parsing is a measured number rather than an
    argument.
    """
    recovered_yes = 0
    recovered_no = 0
    for record in records:
        if record.status is RetentionStatus.UNPARSEABLE and record.normalized_verdict is not None:
            if record.normalized_verdict == "YES":
                recovered_yes += 1
            else:
                recovered_no += 1
    return {
        "unparseable_strict": sum(1 for r in records if r.status is RetentionStatus.UNPARSEABLE),
        "recovered_by_normalization_yes": recovered_yes,
        "recovered_by_normalization_no": recovered_no,
    }


def category_breakdown(records: list[RetentionRecord]) -> dict[SCCategoryId, list[RetentionRecord]]:
    """Group by SC category for the Table 14 equivalent per type analysis."""
    grouped: dict[SCCategoryId, list[RetentionRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)
    return grouped
