"""WildChat adapter. TASK-006, FR-010.

PAPER SPECIFICATION spec 11.1: user and assistant only. No system prompt, no tools. Table 1
reports 257.22 user turns per 100K context, so user turns are DENSE here, which is why
Recent-5 with bottom injection scores 100 percent on this corpus and 0.4 percent on Hermes.

`OPERATIONAL HAZARD` spec 6.8: WildChat carries unsafe user content. Provider content filters
reject some samples (the paper hit this on 15 of 2,000 Gemini calls). That is handled
downstream as a first class BLOCKED status, not by filtering the corpus here, because
filtering would change the contexts the paper measured.
"""

from __future__ import annotations

from compint.core.models import Message, Role
from compint.data.ingest.base import BaseAdapter, QuarantineReason


class WildChatAdapter(BaseAdapter):
    dataset = "wildchat"

    def validate(self, messages: list[Message]) -> tuple[QuarantineReason | None, str]:
        reason, detail = super().validate(messages)
        if reason is not None:
            return reason, detail
        for message in messages:
            if message.role is Role.SYSTEM:
                return (
                    QuarantineReason.UNEXPECTED_SYSTEM_PROMPT,
                    f"WildChat conversations carry no system prompt, saw one at index {message.index}",
                )
            if message.role in (Role.TOOL, Role.THINKING):
                return (
                    QuarantineReason.SCHEMA_MISMATCH,
                    f"WildChat carries no tool or thinking turns, saw {message.role.value}",
                )
        return None, ""
