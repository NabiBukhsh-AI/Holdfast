"""Core domain models. TASK-002.

Equation 1 (spec 6.1) writes the history as a strictly alternating user/agent pair list.
`CRITICAL IMPLEMENTATION NOTE`: real trajectories do not alternate. Hermes Agent averages
119.66 total messages but only 9.04 user turns per 100K context; OpenResearcher averages
310.46 messages with exactly 1.00 user turn. History is therefore modelled as an ordered
message list, and `user_turn_indices` exposes U^t, which is the index space the injection
operator works in. Implementing injection over raw message indices would produce a completely
different experiment (INFERENCE, spec 6.1, assumption A-04).
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    THINKING = "thinking"


class Message(BaseModel):
    """One message in H^t. Frozen: INV-1 says H^t is never mutated in place anywhere."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="position in H^t, 0 based, strictly increasing")
    role: Role
    content: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    # Computed at ingestion and cached. Never recomputed implicitly: a silent recount with a
    # different tokenizer would shift every context length (spec 11.1).
    token_count: int = Field(ge=0)


class History(BaseModel):
    """H^t. An ordered message list, not an alternating pair list."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...]

    @model_validator(mode="after")
    def _indices_strictly_increasing(self) -> Self:
        previous = -1
        for message in self.messages:
            if message.index <= previous:
                raise ValueError(
                    f"message indices must be strictly increasing, saw {message.index} "
                    f"after {previous}"
                )
            previous = message.index
        return self

    @property
    def user_turn_indices(self) -> tuple[int, ...]:
        """U^t. THIS is the index space used by the injection operator.

        Returns the RAW message indices of user role messages, in order. Tool, thinking,
        assistant, and system messages are excluded (spec 6.1).
        """
        return tuple(m.index for m in self.messages if m.role is Role.USER)

    @property
    def n_user_turns(self) -> int:
        return len(self.user_turn_indices)

    @property
    def n_messages(self) -> int:
        return len(self.messages)

    @property
    def token_count(self) -> int:
        """Sum of cached per message counts. |H^t| in Equation 3 is token length."""
        return sum(m.token_count for m in self.messages)

    @property
    def is_degenerate_for_injection(self) -> bool:
        """True when |U^t| == 1, so Top, Middle, Bottom, and Multi all collapse.

        Spec 6.6 and FR-023. Not a bug: it is a property of the single user turn
        long horizon setting, and it is why the paper's injection location figures omit
        OpenResearcher. Cells built on such a history are marked DEGENERATE, never reported
        as four independent numbers.
        """
        return self.n_user_turns == 1

    def render(self) -> str:
        """Flatten to the text form a compactor or probe model consumes."""
        parts: list[str] = []
        for message in self.messages:
            label = message.role.value.upper()
            if message.tool_name:
                label = f"{label}({message.tool_name})"
            parts.append(f"[{label}]\n{message.content}")
        return "\n\n".join(parts)

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        for message in self.messages:
            digest.update(f"{message.index}\x00{message.role.value}\x00".encode())
            digest.update(message.content.encode("utf-8"))
            digest.update(b"\x01")
        return "sha256:" + digest.hexdigest()


class Conversation(BaseModel):
    """One normalized source conversation, before stitching."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    dataset: str
    messages: tuple[Message, ...]
    token_count: int = Field(ge=0)
    source_index: int = Field(ge=0, description="position in the original dataset ordering")
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def n_user_turns(self) -> int:
        return sum(1 for m in self.messages if m.role is Role.USER)

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.dataset.encode("utf-8"))
        for message in self.messages:
            digest.update(f"\x00{message.role.value}\x00".encode())
            digest.update(message.content.encode("utf-8"))
        return "sha256:" + digest.hexdigest()

    def to_history(self) -> History:
        """Reindex messages from zero and present them as a History."""
        return History(
            messages=tuple(
                m.model_copy(update={"index": i}) for i, m in enumerate(self.messages)
            )
        )


class SCCategoryId(StrEnum):
    """The five research categories plus the production only fallback (FR-001, FR-042)."""

    ACTION = "action"
    INFORMATION = "information"
    PROCESS = "process"
    PREFERENCE = "preference"
    OUTPUT = "output"
    OTHER = "other"


class Strength(StrEnum):
    """Constraint Strength axis, spec 6.7."""

    STRICT = "strict"
    PREFERENTIAL = "preferential"


class Explicitness(StrEnum):
    """Explicitness axis, spec 6.7."""

    DIRECT = "direct"
    CONTEXTUALIZED = "contextualized"


class InjectionCondition(StrEnum):
    """The four injection conditions, spec 6.6."""

    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"
    MULTI = "multi"


class SideConstraint(BaseModel):
    """One catalog SC with its probe. PAPER SPECIFICATION Appendix Table 12 (FR-003)."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(ge=1)
    category: SCCategoryId
    body: str
    probe_query: str
    option_compliant: str
    option_violating: str
    citation: str = "N/A"

    def sc_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.body.encode("utf-8")).hexdigest()


class FramedSC(BaseModel):
    """A catalog SC rendered under one (Strength, Explicitness) framing."""

    model_config = ConfigDict(frozen=True)

    sc: SideConstraint
    strength: Strength
    explicitness: Explicitness
    rendered_text: str
    template_version: str

    @property
    def sc_id(self) -> int:
        return self.sc.id

    @property
    def category(self) -> SCCategoryId:
        return self.sc.category


class InjectedHistory(BaseModel):
    """H^t_{s,I}: a history with an SC injected at known user turn positions.

    A distinct type from History, and from CompactedContext, so that INV-4 is enforced at the
    type level: the retention judge cannot be handed an uncompacted context by accident.
    """

    model_config = ConfigDict(frozen=True)

    history: History
    framed_sc: FramedSC
    condition: InjectionCondition
    # Indices into U^t, not raw message indices.
    locations: tuple[int, ...]
    repetition_r: int | None = None
    degenerate: bool = False
    separator: str = " "

    @property
    def token_count(self) -> int:
        return self.history.token_count


class CompactionStatus(StrEnum):
    """Terminal states of a compaction call. Spec 14.4 edge cases."""

    OK = "OK"
    COMPACTION_FAILED = "COMPACTION_FAILED"
    REFUSED = "REFUSED"
    OVERFLOW = "OVERFLOW"
    TRUNCATED = "TRUNCATED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


class CompactedContext(BaseModel):
    """C(H^t) or C(H^t_{s,I}). The only type the retention judge accepts (INV-4)."""

    model_config = ConfigDict(frozen=True)

    text: str
    compactor_id: str
    model_id: str
    prompt_id: str | None = None
    prompt_hash: str | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = 0.0
    status: CompactionStatus = CompactionStatus.OK
    raw: str = ""

    @property
    def compaction_ratio(self) -> float | None:
        """|H^t| / |C(H^t)|, spec 6.14. None when the denominator is zero."""
        if self.output_tokens <= 0:
            return None
        return self.input_tokens / self.output_tokens

    def context_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()
