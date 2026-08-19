"""OpenResearcher adapter. TASK-006, FR-010, FR-016.

PAPER SPECIFICATION spec 11.1 and 14.2: one user turn, then long autonomous tool cycles
(search, open, find). Table 1 reports 310.46 turns and exactly 1.00 user turns.

`CRITICAL EDGE CASE` spec 6.6: with |U^t| = 1 all four injection conditions collapse onto the
same location. Contexts built from this corpus are marked DEGENERATE downstream (FR-023),
which is why the paper's injection location figures omit OpenResearcher entirely.
"""

from __future__ import annotations

from compint.core.models import Message, Role
from compint.data.ingest.base import BaseAdapter, QuarantineReason


class OpenResearcherAdapter(BaseAdapter):
    dataset = "openresearcher"

    def __init__(self, tokenizer, *, allow_multiple_user_turns: bool = False) -> None:  # type: ignore[no-untyped-def]
        super().__init__(tokenizer)
        # The 220K condition concatenates two 110K entries and therefore has TWO user turns
        # (spec 14.2). That asymmetry is allowed explicitly, never by accident.
        self._allow_multiple_user_turns = allow_multiple_user_turns

    def validate(self, messages: list[Message]) -> tuple[QuarantineReason | None, str]:
        reason, detail = super().validate(messages)
        if reason is not None:
            return reason, detail
        user_turns = [m for m in messages if m.role is Role.USER]
        if len(user_turns) > 1 and not self._allow_multiple_user_turns:
            return (
                QuarantineReason.MULTIPLE_USER_TURNS,
                f"OpenResearcher trajectories carry exactly one user turn, saw {len(user_turns)}",
            )
        if user_turns and user_turns[0].index != 0:
            return (
                QuarantineReason.SCHEMA_MISMATCH,
                f"the user turn must open the trajectory, saw index {user_turns[0].index}",
            )
        return None, ""
