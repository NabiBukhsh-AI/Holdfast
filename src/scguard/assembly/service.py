"""Assembly service. TASK-026, Algorithm 14.8, Equation 10.

    H~^t = C(H^t) (+) S^t

The caller supplies the compactor output; SC-GUARD does not run the compactor (spec 18.2).
What SC-GUARD does is drain, read, budget, and attach.

Three edge cases carry the design:

1. **Drain timeout.** A user turn that arrived microseconds before compaction may still be in
   extraction. The wait is bounded, then `registry_incomplete` is set. Never wait unboundedly
   (that blocks the user) and never proceed silently (that recreates the failure being fixed).

2. **Second compaction event.** Stripping the previous registry block is what stops it being
   fed back into the compactor and re-summarized. Without it the registry decays across
   successive compactions, which is the original failure mode with extra steps (INV-7).

3. **Store failure.** Returns 503. It does NOT return an empty registry, because an empty
   registry is indistinguishable from a session with no constraints.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from scguard.audit.emitter import AuditEmitter, AuditEventType
from scguard.extractor.queue import ExtractionQueue
from scguard.registry.budget import BudgetDecision, enforce_budget, evicted_summary
from scguard.registry.store import RegistryStore, new_id
from shared.assembly import AssemblyMode, assemble
from shared.delimiters import count_registry_blocks, strip_registry_blocks


class AssemblyWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    detail: str


class RegistrySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    active_count: int
    injected_count: int
    evicted_count: int
    tokens: int


class AssemblyResult(BaseModel):
    """What `POST /compact` returns, plus what the assemblies table records."""

    model_config = ConfigDict(frozen=True)

    assembly_id: str
    augmented_context: str
    registry: RegistrySummary
    warnings: tuple[AssemblyWarning, ...] = ()
    # NFR-008: never silent. Returned in the body AND as X-SC-Registry-Incomplete.
    registry_incomplete: bool = False
    drain_wait_ms: int = 0
    assembly_ms: int = 0
    summary_tokens: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)


class AssemblyService:
    """Algorithm 14.8."""

    def __init__(
        self,
        store: RegistryStore,
        queue: ExtractionQueue,
        audit: AuditEmitter,
        *,
        assembly_mode: AssemblyMode = "delimited",
        budget_tokens: int = 200,
        drain_timeout_ms: int = 200,
        shadow_mode: bool = False,
        chars_per_token: float = 4.0,
        extractor_model: str = "",
        prompt_hash: str = "",
    ) -> None:
        self._store = store
        self._queue = queue
        self._audit = audit
        self._assembly_mode: AssemblyMode = assembly_mode
        self._budget_tokens = budget_tokens
        self._drain_timeout_ms = drain_timeout_ms
        # FR-086: extract and record but do not inject, for safe rollout measurement.
        self._shadow_mode = shadow_mode
        self._chars_per_token = chars_per_token
        self._extractor_model = extractor_model
        self._prompt_hash = prompt_hash

    def _tokens(self, text: str) -> int:
        return max(0, int(len(text) / self._chars_per_token))

    async def compact(
        self,
        session_id: str,
        tenant_id: str,
        *,
        compaction_index: int,
        compacted_summary: str,
        drain_timeout_ms: int | None = None,
        assembly_mode: AssemblyMode | None = None,
        budget_tokens: int | None = None,
    ) -> AssemblyResult:
        """Drain, read, budget, attach. Returns the augmented context and its report."""
        started = time.perf_counter()
        warnings: list[AssemblyWarning] = []
        timeout = drain_timeout_ms if drain_timeout_ms is not None else self._drain_timeout_ms
        mode: AssemblyMode = assembly_mode or self._assembly_mode
        budget = budget_tokens if budget_tokens is not None else self._budget_tokens

        # Steps 1 and 2: bounded drain.
        complete, outstanding, waited_ms = await self._queue.drain(session_id, timeout)
        registry_incomplete = not complete
        if registry_incomplete:
            warnings.append(
                AssemblyWarning(
                    code="REGISTRY_INCOMPLETE",
                    detail=(
                        f"{outstanding} extraction job(s) still pending after {timeout}ms drain "
                        "timeout; constraints from those turns are NOT in this context"
                    ),
                )
            )

        # Step 3: strip any prior registry block. The caller may hand back a previously
        # augmented context; nesting blocks would double-inject and, worse, let the older block
        # be summarized on the next pass.
        stripped_summary = strip_registry_blocks(compacted_summary)

        # Step 5: read the registry. A store failure propagates as RegistryUnavailableError,
        # which the API renders as 503. It must never become an empty registry here.
        active = await self._store.active(session_id)
        session = await self._store.get_session(session_id)

        # Step 6: budget.
        decision: BudgetDecision = enforce_budget(
            active,
            budget,
            audit=self._audit,
            session_id=session_id,
            tenant_id=tenant_id,
        )
        if decision.evicted:
            warnings.append(
                AssemblyWarning(code="REGISTRY_EVICTED", detail=evicted_summary(decision))
            )
        if decision.budget_exceeded_single:
            warnings.append(
                AssemblyWarning(
                    code="BUDGET_EXCEEDED_SINGLE",
                    detail=(
                        "a single constraint exceeds the whole registry budget and was kept "
                        "whole rather than truncated"
                    ),
                )
            )

        # Steps 7 to 9: attach. Shadow mode records everything and injects nothing.
        if self._shadow_mode:
            augmented = stripped_summary
            injected_count = 0
            warnings.append(
                AssemblyWarning(
                    code="SHADOW_MODE",
                    detail=(
                        f"{len(decision.kept)} constraint(s) were extracted and recorded but "
                        "NOT injected, because the service is in shadow mode"
                    ),
                )
            )
        else:
            output = assemble(stripped_summary, decision.kept, mode=mode)
            augmented = output.text
            injected_count = output.report.injected_count

        assembly_ms = int((time.perf_counter() - started) * 1000.0)
        assembly_id = new_id("asm")

        self._audit.emit(
            session_id,
            tenant_id,
            AuditEventType.ASSEMBLY_PERFORMED,
            assembly_id=assembly_id,
            compaction_index=compaction_index,
            registry_version=session.registry_version,
            active_count=len(active),
            injected_count=injected_count,
            evicted_count=decision.n_evicted,
            registry_incomplete=registry_incomplete,
            drain_wait_ms=int(waited_ms),
            assembly_mode=mode,
            shadow_mode=self._shadow_mode,
        )

        return AssemblyResult(
            assembly_id=assembly_id,
            augmented_context=augmented,
            registry=RegistrySummary(
                version=session.registry_version,
                active_count=len(active),
                injected_count=injected_count,
                evicted_count=decision.n_evicted,
                tokens=sum(row.token_count for row in decision.kept),
            ),
            warnings=tuple(warnings),
            registry_incomplete=registry_incomplete,
            drain_wait_ms=int(waited_ms),
            assembly_ms=assembly_ms,
            summary_tokens=self._tokens(stripped_summary),
            metadata={
                "assembly_mode": mode,
                "extractor_model": self._extractor_model,
                "prompt_hash": self._prompt_hash,
                "shadow_mode": str(self._shadow_mode).lower(),
            },
        )


def assert_single_registry_block(text: str) -> None:
    """INV-7 check, used by tests and by the double compaction integration path."""
    count = count_registry_blocks(text)
    if count > 1:
        raise AssertionError(
            f"augmented context carries {count} registry blocks; exactly one is the invariant "
            "(INV-7). A prior block was not stripped before reassembly."
        )
