"""Hermes Agent adapter. TASK-006, FR-010.

PAPER SPECIFICATION spec 11.1: system prompt at position 0, tool calls, tool results, and
thinking turns preserved. Table 1: 119.66 total turns against 9.04 user turns per 100K
context, roughly 12 non user messages per user message.

That ratio is why the injection operator indexes U^t rather than the raw message list
(INFERENCE spec 6.1). Stripping tool traffic here to make the trajectory look alternating
would destroy the experiment.
"""

from __future__ import annotations

from compint.core.models import Message, Role
from compint.data.ingest.base import BaseAdapter, QuarantineReason


class HermesAgentAdapter(BaseAdapter):
    dataset = "hermes_agent"

    def validate(self, messages: list[Message]) -> tuple[QuarantineReason | None, str]:
        reason, detail = super().validate(messages)
        if reason is not None:
            return reason, detail
        system_positions = [m.index for m in messages if m.role is Role.SYSTEM]
        if len(system_positions) > 1:
            return (
                QuarantineReason.SYSTEM_PROMPT_NOT_FIRST,
                f"expected at most one system prompt, saw {len(system_positions)}",
            )
        if system_positions and system_positions[0] != 0:
            return (
                QuarantineReason.SYSTEM_PROMPT_NOT_FIRST,
                f"system prompt at index {system_positions[0]}, expected 0",
            )
        return None, ""
