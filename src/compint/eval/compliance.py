"""Compliance probe harness. TASK-015, Equations 7 and 8, spec 6.10, Algorithm 14.10.

Four conditions that vary ONLY in the context K supplied to the probe model:

    K_lctx     = H^t                     full history, no SC. The baseline correction term.
    K_lctx_sc  = H^t_{s,P}               full history with the SC injected.
    K_comp     = C(H^t_{s,P})            SC injected BEFORE compaction. The setting under test.
    K_ub       = assemble(C(H^t), [s])   SC appended AFTER compaction. The ceiling.

`ARCHITECTURAL SIGNIFICANCE` spec 6.10: K_ub is not merely a control, it IS the mechanism the
extractor implements. Equation 10 is structurally identical to K_ub with a single SC replaced
by the extracted registry. So K_ub must be built by the SHARED assemble() (INV-5), which
guarantees the measured upper bound is the mechanism actually shipped.

`COST` spec 14.10: K_lctx is constant across all 15 SCs for a given context, and K_ub uses
C(H^t), the UN-injected compaction, which is also constant across the 15. Both are computed
once per context. A naive implementation wastes 14 of every 15 of those calls, and compliance
probing is the dominant cost in the whole evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from compint.compactors.base import Compactor
from compint.core.models import (
    CompactedContext,
    CompactionStatus,
    FramedSC,
    History,
    InjectedHistory,
)
from compint.core.random_source import RandomSource
from compint.eval.records import ComplianceCondition, OptionOrder, ProbeRecord, ProbeStatus
from shared.assembly import AssemblyMode, assemble
from shared.errors import (
    ContentFilterError,
    ContextOverflowError,
    ProviderError,
    ProviderRefusalError,
    ProviderTimeoutError,
)
from shared.llm_client import LLMClient, LLMRequest
from shared.prompts import Prompt

Answer = Literal["A", "B"]


@dataclass(frozen=True)
class RenderedMCQ:
    """One rendered probe plus the option mapping, which is ALWAYS recorded (spec 6.9)."""

    text: str
    option_order: OptionOrder
    gold: Answer


def render_mcq(
    prompt: Prompt,
    framed_sc: FramedSC,
    *,
    option_order: OptionOrder,
) -> RenderedMCQ:
    """Render the MCQ under a known option order.

    `CRITICAL IMPLEMENTATION REQUIREMENT` spec 6.9: option order must be recorded, or A becomes
    a positional prior and the compliance metric measures answer position bias. UNKNOWN U-11:
    the paper does not state whether it randomized. Both orders are supported and the mapping
    is always stored on the record.
    """
    compliant = framed_sc.sc.option_compliant
    violating = framed_sc.sc.option_violating
    if option_order == "AB":
        option_a, option_b, gold = compliant, violating, "A"
    else:
        option_a, option_b, gold = violating, compliant, "B"
    _, user = prompt.render(
        probe_query=framed_sc.sc.probe_query, option_a=option_a, option_b=option_b
    )
    return RenderedMCQ(text=user, option_order=option_order, gold=gold)  # type: ignore[arg-type]


def parse_answer(raw: str) -> Answer | None:
    """Strict single letter parse. Anything else is UNPARSEABLE, never guessed."""
    candidate = raw.strip().upper().lstrip("*_( \t\n")
    if not candidate:
        return None
    first = candidate[0]
    if first in ("A", "B"):
        # Guard against "ANSWER: B" style prefixes being read as "A".
        if candidate.startswith("ANSWER"):
            tail = candidate[len("ANSWER") :].lstrip(": \t")
            return tail[0] if tail[:1] in ("A", "B") else None  # type: ignore[return-value]
        return first  # type: ignore[return-value]
    return None


@dataclass
class ContextCache:
    """Per context artifacts that are constant across all 15 SCs.

    Holding these on an explicit object (rather than memoizing inside a call site) is what
    lets a test assert exactly how many compactions a 15 SC run issued.
    """

    uninjected_compaction: CompactedContext | None = None
    lctx_text: str | None = None
    compaction_calls: int = 0
    lctx_builds: int = 0
    _ub_by_sc: dict[int, str] = field(default_factory=dict)

    async def get_uninjected_compaction(
        self, history: History, compactor: Compactor
    ) -> CompactedContext:
        """C(H^t). Computed once per context, reused by every K_ub in that context."""
        if self.uninjected_compaction is None:
            self.uninjected_compaction = await compactor.compact(history)
            self.compaction_calls += 1
        return self.uninjected_compaction

    def get_lctx(self, history: History) -> str:
        """K_lctx. Constant across the 15 SCs: there is no SC in it."""
        if self.lctx_text is None:
            self.lctx_text = history.render()
            self.lctx_builds += 1
        return self.lctx_text


class ComplianceHarness:
    """Builds the four condition contexts and probes each one."""

    def __init__(
        self,
        client: LLMClient,
        mcq_prompt: Prompt,
        model: str,
        *,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        max_tokens: int = 8,
        option_order: str = "fixed",
        assembly_mode: AssemblyMode = "bare",
        probe_context_limit: int | None = None,
        rng: RandomSource | None = None,
    ) -> None:
        self._client = client
        self._prompt = mcq_prompt
        self._model = model
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._option_order_mode = option_order
        self._assembly_mode: AssemblyMode = assembly_mode
        self._probe_context_limit = probe_context_limit
        self._rng = rng or RandomSource(0)

    def _choose_option_order(self, framed_sc: FramedSC) -> OptionOrder:
        if self._option_order_mode == "fixed":
            # Research default: matches the paper's Table 12 layout, compliant option first.
            return "AB"
        # Derived from the SC id so the draw is reproducible per instance, not call ordered.
        return "AB" if self._rng.derive(f"mcq:{framed_sc.sc_id}").random() < 0.5 else "BA"

    async def build_condition_context(
        self,
        condition: ComplianceCondition,
        *,
        history: History,
        injected: InjectedHistory,
        framed_sc: FramedSC,
        compactor: Compactor,
        cache: ContextCache,
    ) -> tuple[str | None, ProbeStatus]:
        """Build K_g for one condition. Returns (context, status)."""
        if condition == "lctx":
            return cache.get_lctx(history), ProbeStatus.OK

        if condition == "lctx_sc":
            return injected.history.render(), ProbeStatus.OK

        if condition == "comp":
            compacted = await compactor.compact(injected.history)
            cache.compaction_calls += 1
            if compacted.status is not CompactionStatus.OK:
                return None, _status_for(compacted.status)
            return compacted.text, ProbeStatus.OK

        if condition == "ub":
            compacted = await cache.get_uninjected_compaction(history, compactor)
            if compacted.status is not CompactionStatus.OK:
                return None, _status_for(compacted.status)
            # INV-5: the SHARED assemble(), the same function production uses.
            output = assemble(
                compacted.text,
                [_SingleSCEntry(framed_sc.rendered_text)],
                mode=self._assembly_mode,
            )
            return output.text, ProbeStatus.OK

        raise ValueError(f"unknown compliance condition {condition}")

    async def probe(
        self,
        condition: ComplianceCondition,
        context_text: str,
        framed_sc: FramedSC,
        *,
        instance_id: str,
        context_tokens: int = 0,
    ) -> ProbeRecord:
        """Equation 7: append the probe as the next user turn and read back A or B."""
        order = self._choose_option_order(framed_sc)
        mcq = render_mcq(self._prompt, framed_sc, option_order=order)

        if self._probe_context_limit is not None and context_tokens > self._probe_context_limit:
            # Spec 14.10: at 220K the probe window is exceeded on the long context conditions.
            # A distinct status, not an error, and not a silently truncated context.
            return self._record(
                condition,
                framed_sc,
                mcq,
                instance_id,
                None,
                ProbeStatus.OVERFLOW,
                context_tokens,
                f"context {context_tokens} exceeds probe window {self._probe_context_limit}",
            )

        request = LLMRequest(
            model=self._model,
            user=f"{context_text}\n\n{mcq.text}",
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout_s=self._timeout_s,
        )
        try:
            response = await self._client.complete(request)
        except ContextOverflowError as exc:
            return self._record(
                condition,
                framed_sc,
                mcq,
                instance_id,
                None,
                ProbeStatus.OVERFLOW,
                context_tokens,
                str(exc),
            )
        except ProviderRefusalError as exc:
            return self._record(
                condition,
                framed_sc,
                mcq,
                instance_id,
                None,
                ProbeStatus.REFUSED,
                context_tokens,
                str(exc),
            )
        except ContentFilterError as exc:
            return self._record(
                condition,
                framed_sc,
                mcq,
                instance_id,
                None,
                ProbeStatus.BLOCKED,
                context_tokens,
                str(exc),
            )
        except (ProviderTimeoutError, ProviderError) as exc:
            return self._record(
                condition,
                framed_sc,
                mcq,
                instance_id,
                None,
                ProbeStatus.ERROR,
                context_tokens,
                str(exc),
            )

        answer = parse_answer(response.text)
        return self._record(
            condition,
            framed_sc,
            mcq,
            instance_id,
            answer,
            ProbeStatus.OK if answer is not None else ProbeStatus.UNPARSEABLE,
            context_tokens,
            response.raw or response.text,
        )

    def failure_record(
        self,
        condition: ComplianceCondition,
        framed_sc: FramedSC,
        *,
        instance_id: str,
        status: ProbeStatus,
        detail: str,
    ) -> ProbeRecord:
        """Build a probe record for a condition that could not be probed at all.

        Public because the grid runner needs it too: a cell whose compaction failed still has
        to produce a record carrying the reason, or the instance would silently vanish from the
        denominator instead of being counted as an exclusion (INV-6).
        """
        if status is ProbeStatus.OK:
            raise ValueError("failure_record requires a non OK status")
        return self._record(
            condition,
            framed_sc,
            render_mcq(self._prompt, framed_sc, option_order=self._choose_option_order(framed_sc)),
            instance_id,
            None,
            status,
            0,
            detail,
        )

    @staticmethod
    def probe_status_for(compaction_status: CompactionStatus) -> ProbeStatus:
        """Map a compaction failure onto the probe status that records why nothing was probed."""
        return _status_for(compaction_status)

    def _record(
        self,
        condition: ComplianceCondition,
        framed_sc: FramedSC,
        mcq: RenderedMCQ,
        instance_id: str,
        answer: Answer | None,
        status: ProbeStatus,
        context_tokens: int,
        raw: str,
    ) -> ProbeRecord:
        return ProbeRecord(
            instance_id=instance_id,
            sc_id=framed_sc.sc_id,
            category=framed_sc.category,
            condition=condition,
            option_order=mcq.option_order,
            gold=mcq.gold,
            answer=answer,
            status=status,
            probe_model=self._model,
            context_tokens=context_tokens,
            raw_response=raw[:2000],
        )

    async def run_conditions(
        self,
        conditions: tuple[ComplianceCondition, ...],
        *,
        history: History,
        injected: InjectedHistory,
        framed_sc: FramedSC,
        compactor: Compactor,
        cache: ContextCache,
        instance_id: str,
    ) -> list[ProbeRecord]:
        """Probe every requested condition for one (context, SC) pair."""
        records: list[ProbeRecord] = []
        for condition in conditions:
            context_text, status = await self.build_condition_context(
                condition,
                history=history,
                injected=injected,
                framed_sc=framed_sc,
                compactor=compactor,
                cache=cache,
            )
            if context_text is None:
                mcq = render_mcq(
                    self._prompt, framed_sc, option_order=self._choose_option_order(framed_sc)
                )
                records.append(
                    self._record(
                        condition,
                        framed_sc,
                        mcq,
                        instance_id,
                        None,
                        status,
                        0,
                        "compaction did not produce a probeable context",
                    )
                )
                continue
            records.append(
                await self.probe(
                    condition,
                    context_text,
                    framed_sc,
                    instance_id=instance_id,
                    context_tokens=len(context_text) // 4,
                )
            )
        return records


@dataclass(frozen=True)
class _SingleSCEntry:
    """Adapts one framed SC to the registry entry shape assemble() consumes."""

    canonical_text: str

    @property
    def is_active(self) -> bool:
        return True


def _status_for(compaction_status: CompactionStatus) -> ProbeStatus:
    """Map a compaction failure onto the probe status that records why nothing was probed."""
    mapping = {
        CompactionStatus.OVERFLOW: ProbeStatus.OVERFLOW,
        CompactionStatus.REFUSED: ProbeStatus.REFUSED,
        CompactionStatus.BLOCKED: ProbeStatus.BLOCKED,
    }
    return mapping.get(compaction_status, ProbeStatus.ERROR)
