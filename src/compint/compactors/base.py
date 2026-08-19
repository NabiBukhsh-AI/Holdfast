"""The Compactor protocol. TASK-011, FR-030, spec 11.3.

INV-2 is enforced at the type level: `compact()` accepts a `History` and nothing else. There
is no parameter through which the registry S^t could reach a compactor, which is the whole
point of the architectural separation the paper argues for. The registry is never compressed
because the compactor cannot see it.

`CompactionResult` is `CompactedContext`, the type the retention judge accepts (INV-4). Using
one type here means a compactor's output can be judged and an injected history cannot.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from compint.core.models import CompactedContext, CompactionStatus, History

# Spec 11.3 names this type CompactionResult. It is CompactedContext so that the type level
# guarantees for INV-2 and INV-4 are the same object rather than two that can drift.
CompactionResult = CompactedContext


@runtime_checkable
class Compactor(Protocol):
    """C: List[Message] -> str, plus the metadata every result row needs."""

    id: str

    async def compact(self, history: History) -> CompactionResult: ...


def should_compact(history_tokens: int, l_max: int, alpha_l: float) -> bool:
    """Equation 3 trigger:  |H^t| >= alpha_l * l_max.

    Pure so it is unit testable at the boundary. The paper uses `>=`, so equality TRIGGERS.
    UNKNOWN U-10: alpha_l is read from config and must never be hardcoded (spec 6.3).
    """
    if not 0.0 < alpha_l <= 1.0:
        raise ValueError(f"alpha_l must be in (0, 1], got {alpha_l}")
    if l_max <= 0:
        raise ValueError(f"l_max must be positive, got {l_max}")
    return history_tokens >= alpha_l * l_max


def failed_result(
    compactor_id: str,
    model_id: str,
    status: CompactionStatus,
    input_tokens: int,
    detail: str,
) -> CompactionResult:
    """Build a terminal failure result.

    Spec 14.4: empty or wrapper only output is COMPACTION_FAILED and must NOT be judged.
    Rule 13: the failure is a value the caller must handle, never an empty summary string
    that flows onward looking like a successful compaction of a context with no constraints.
    """
    if status is CompactionStatus.OK:
        raise ValueError("failed_result requires a non OK status")
    return CompactionResult(
        text="",
        compactor_id=compactor_id,
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=0,
        status=status,
        raw=detail,
    )
