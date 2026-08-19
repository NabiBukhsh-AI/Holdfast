"""FillerContext: one constructed long context instance. Spec 12.2 stage 3.

A filler context is the H^t that SCs are injected into. It carries full provenance (which
source conversations, in what order) because a context whose construction cannot be traced
cannot be debugged when its downstream numbers look wrong.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from compint.core.models import History


class ContextStatus(StrEnum):
    """How construction terminated. Every non OK status is reported, never hidden."""

    OK = "OK"
    # The pool ran out before the target length was reached. Algorithm 1 emits the context
    # SHORT of l_t; the spec requires this be flagged, not silently accepted.
    SHORT_POOL_EXHAUSTED = "SHORT_POOL_EXHAUSTED"
    # The final append pushed past 1.25 * l_t and the last sample was cropped.
    CROPPED_TO_SOFT_CAP = "CROPPED_TO_SOFT_CAP"
    # OpenResearcher: truncated at a turn boundary while retaining at least l_t.
    TRUNCATED_AT_TURN_BOUNDARY = "TRUNCATED_AT_TURN_BOUNDARY"


class FillerContext(BaseModel):
    """One long context instance plus its provenance."""

    model_config = ConfigDict(frozen=True)

    context_id: str
    dataset: str
    split: str
    target_tokens: int = Field(ge=1)
    actual_tokens: int = Field(ge=0)
    history: History
    # Ordered provenance: which source conversations, in the order they were appended.
    source_ids: tuple[str, ...]
    n_stitched: int = Field(ge=1)
    status: ContextStatus = ContextStatus.OK
    context_set_version: str = "v1"
    detail: str = ""

    @property
    def n_turns(self) -> int:
        return self.history.n_messages

    @property
    def n_user_turns(self) -> int:
        """Table 1's `# User Turns` column, reported separately from `# Turns`."""
        return self.history.n_user_turns

    @property
    def is_degenerate_for_injection(self) -> bool:
        return self.history.is_degenerate_for_injection

    def statistics(self) -> dict[str, float | int | str]:
        """The per context row behind the Table 1 statistics report."""
        return {
            "context_id": self.context_id,
            "dataset": self.dataset,
            "split": self.split,
            "target_tokens": self.target_tokens,
            "actual_tokens": self.actual_tokens,
            "n_turns": self.n_turns,
            "n_user_turns": self.n_user_turns,
            "n_stitched": self.n_stitched,
            "status": self.status.value,
        }
