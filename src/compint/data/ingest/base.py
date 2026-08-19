"""Corpus ingestion contract. TASK-006, spec 11.1.

Three heterogeneous public corpora are normalized into one internal `Conversation` schema.
Malformed rows are QUARANTINED with a reason and counted; they are never dropped silently,
because a silent 3 percent drop rate would shift every downstream context statistic without
appearing anywhere in the results.

`CRITICAL` spec 11.1: token counting uses the tokenizer of the compactor model under test.
"100K tokens" is not tokenizer invariant and the paper does not say which tokenizer defines
the target length (UNKNOWN U-07). The tokenizer id is recorded in the ingestion manifest.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from compint.core.models import Conversation, Message, Role
from compint.core.tokenization import Tokenizer


class QuarantineReason(StrEnum):
    """Why a source row did not become a Conversation. Every reason is counted and reported."""

    EMPTY_CONVERSATION = "EMPTY_CONVERSATION"
    NO_USER_TURN = "NO_USER_TURN"
    UNKNOWN_ROLE = "UNKNOWN_ROLE"
    MISSING_CONTENT = "MISSING_CONTENT"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    UNEXPECTED_SYSTEM_PROMPT = "UNEXPECTED_SYSTEM_PROMPT"
    SYSTEM_PROMPT_NOT_FIRST = "SYSTEM_PROMPT_NOT_FIRST"
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"
    MULTIPLE_USER_TURNS = "MULTIPLE_USER_TURNS"


class QuarantinedRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    source_index: int
    reason: QuarantineReason
    detail: str = ""


class IngestionResult(BaseModel):
    """Conversations plus the quarantine ledger. Both halves are always reported together."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    tokenizer_id: str
    conversations: tuple[Conversation, ...]
    quarantined: tuple[QuarantinedRecord, ...] = ()

    @property
    def n_seen(self) -> int:
        return len(self.conversations) + len(self.quarantined)

    @property
    def quarantine_rate(self) -> float:
        """TASK-006 acceptance criterion: below 0.5 percent."""
        if self.n_seen == 0:
            return 0.0
        return len(self.quarantined) / self.n_seen

    def reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.quarantined:
            counts[record.reason.value] = counts.get(record.reason.value, 0) + 1
        return counts


class SourceAdapter(Protocol):
    """Normalizes one corpus. Implementations must be pure functions of their input rows."""

    dataset: str

    def to_conversations(self, rows: Iterable[dict[str, Any]]) -> IngestionResult: ...


ROLE_ALIASES: dict[str, Role] = {
    "system": Role.SYSTEM,
    "user": Role.USER,
    "human": Role.USER,
    "assistant": Role.ASSISTANT,
    "gpt": Role.ASSISTANT,
    "model": Role.ASSISTANT,
    "tool": Role.TOOL,
    "function": Role.TOOL,
    "tool_response": Role.TOOL,
    "observation": Role.TOOL,
    "thinking": Role.THINKING,
    "thought": Role.THINKING,
    "reasoning": Role.THINKING,
}


def map_role(raw: str) -> Role | None:
    """Map a source role label onto the internal enum, or None when it is unrecognized.

    Returning None rather than guessing is deliberate: an unrecognized role silently mapped
    to ASSISTANT would change |U^t| and therefore change every injection location.
    """
    return ROLE_ALIASES.get(raw.strip().lower())


class BaseAdapter:
    """Shared row to Conversation machinery. Subclasses supply per corpus validation."""

    dataset: str = "unknown"

    def __init__(self, tokenizer: Tokenizer, *, turns_key: str = "conversation") -> None:
        self._tokenizer = tokenizer
        self._turns_key = turns_key

    def _extract_turns(self, row: dict[str, Any]) -> list[dict[str, Any]] | None:
        for key in (self._turns_key, "conversation", "conversations", "messages", "turns"):
            value = row.get(key)
            if isinstance(value, list):
                return value
        return None

    def _build_messages(
        self, turns: list[dict[str, Any]]
    ) -> tuple[list[Message], QuarantineReason | None, str]:
        messages: list[Message] = []
        for position, turn in enumerate(turns):
            if not isinstance(turn, dict):
                return [], QuarantineReason.SCHEMA_MISMATCH, f"turn {position} is not a mapping"
            raw_role = turn.get("role") or turn.get("from") or ""
            role = map_role(str(raw_role))
            if role is None:
                return [], QuarantineReason.UNKNOWN_ROLE, f"turn {position} role {raw_role!r}"
            content = turn.get("content")
            if content is None:
                content = turn.get("value")
            if content is None or not str(content).strip():
                return [], QuarantineReason.MISSING_CONTENT, f"turn {position} has no content"
            text = str(content)
            messages.append(
                Message(
                    index=len(messages),
                    role=role,
                    content=text,
                    tool_name=turn.get("tool_name") or turn.get("name"),
                    tool_call_id=turn.get("tool_call_id") or turn.get("id"),
                    token_count=self._tokenizer.count(text),
                )
            )
        return messages, None, ""

    def validate(self, messages: list[Message]) -> tuple[QuarantineReason | None, str]:
        """Per corpus structural checks. Default: at least one user turn."""
        if not messages:
            return QuarantineReason.EMPTY_CONVERSATION, "no messages"
        if not any(m.role is Role.USER for m in messages):
            return QuarantineReason.NO_USER_TURN, "no user turn"
        return None, ""

    def to_conversations(self, rows: Iterable[dict[str, Any]]) -> IngestionResult:
        conversations: list[Conversation] = []
        quarantined: list[QuarantinedRecord] = []
        seen_hashes: set[str] = set()

        for source_index, row in enumerate(rows):
            turns = self._extract_turns(row)
            if turns is None:
                quarantined.append(
                    QuarantinedRecord(
                        dataset=self.dataset,
                        source_index=source_index,
                        reason=QuarantineReason.SCHEMA_MISMATCH,
                        detail=f"no turn list under keys {sorted(row)[:6]}",
                    )
                )
                continue
            messages, reason, detail = self._build_messages(turns)
            if reason is None:
                reason, detail = self.validate(messages)
            if reason is not None:
                quarantined.append(
                    QuarantinedRecord(
                        dataset=self.dataset,
                        source_index=source_index,
                        reason=reason,
                        detail=detail,
                    )
                )
                continue
            conversation = Conversation(
                conversation_id=str(
                    row.get("conversation_id") or row.get("id") or f"{self.dataset}_{source_index}"
                ),
                dataset=self.dataset,
                messages=tuple(messages),
                token_count=sum(m.token_count for m in messages),
                source_index=source_index,
                metadata={
                    key: str(row[key])
                    for key in ("language", "model", "toxic", "source")
                    if key in row
                },
            )
            content_hash = conversation.content_hash()
            if content_hash in seen_hashes:
                quarantined.append(
                    QuarantinedRecord(
                        dataset=self.dataset,
                        source_index=source_index,
                        reason=QuarantineReason.DUPLICATE_CONTENT,
                        detail=content_hash,
                    )
                )
                continue
            seen_hashes.add(content_hash)
            conversations.append(conversation)

        return IngestionResult(
            dataset=self.dataset,
            tokenizer_id=self._tokenizer.id,
            conversations=tuple(conversations),
            quarantined=tuple(quarantined),
        )


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Local fixture loader. The HuggingFace path lives in scripts/build_contexts.py."""
    import json
    from pathlib import Path

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
