"""Evaluation record types. Spec 12.2 stage 7.

`DESIGN NOTE` spec 12.2: `raw_response` retention is non negotiable. Every aggregate in this
system rests on a parsed model output. Without the raw text, a parser bug discovered later
cannot be corrected without rerunning and re-paying for every judge call.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from compint.core.models import InjectionCondition, SCCategoryId


class RetentionStatus(StrEnum):
    """Terminal states of a retention judgment. Spec 14.9, FR-041, FR-043."""

    OK = "OK"
    # Judge output was neither exactly YES nor exactly NO. NEVER coerced to 0 or 1:
    # silently coercing unparseable verdicts to 0 would inflate the headline finding.
    UNPARSEABLE = "UNPARSEABLE"
    # Provider content filter rejected the sample. The paper hit this on 15 of 2,000 Gemini
    # samples. Counted and reported alongside every affected metric, excluded from denominators.
    BLOCKED = "BLOCKED"
    # The compaction that should have produced the judged text failed, so nothing was judged.
    COMPACTION_FAILED = "COMPACTION_FAILED"
    ERROR = "ERROR"


class ProbeStatus(StrEnum):
    """Terminal states of a compliance probe. Spec 14.10."""

    OK = "OK"
    UNPARSEABLE = "UNPARSEABLE"
    # K_lctx and K_lctx_sc carry the full context; at 220K the probe window is exceeded.
    # A distinct status, not an error (spec 14.10).
    OVERFLOW = "OVERFLOW"
    REFUSED = "REFUSED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


ComplianceCondition = Literal["lctx", "lctx_sc", "comp", "ub"]
OptionOrder = Literal["AB", "BA"]


class RetentionRecord(BaseModel):
    """One Retain(s, C(H^t_{s,I})) judgment. Equation 6."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    sc_id: int
    category: SCCategoryId
    compactor_id: str
    compacted_hash: str
    verdict: Literal["YES", "NO"] | None = None
    status: RetentionStatus = RetentionStatus.OK
    judge_model: str = ""
    judge_prompt_hash: str = ""
    raw_response: str = ""
    # ENGINEERING RECOMMENDATION spec 14.9: record BOTH the strict and the leniently
    # normalized verdict so the effect of parser leniency is measurable rather than assumed.
    normalized_verdict: Literal["YES", "NO"] | None = None
    latency_ms: float = 0.0
    degenerate: bool = False

    @property
    def retained(self) -> int | None:
        """1, 0, or None. None means this record contributes to no numerator or denominator."""
        if self.status is not RetentionStatus.OK or self.verdict is None:
            return None
        return 1 if self.verdict == "YES" else 0


class ProbeRecord(BaseModel):
    """One compliance probe under one of the four conditions. Equations 7 and 8."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    sc_id: int
    category: SCCategoryId
    condition: ComplianceCondition
    # UNKNOWN U-11: always recorded, whichever order was used (spec 6.9).
    option_order: OptionOrder = "AB"
    gold: Literal["A", "B"]
    answer: Literal["A", "B"] | None = None
    status: ProbeStatus = ProbeStatus.OK
    probe_model: str = ""
    context_tokens: int = 0
    raw_response: str = ""

    @property
    def compliant(self) -> bool | None:
        if self.status is not ProbeStatus.OK or self.answer is None:
            return None
        return self.answer == self.gold


class FreeGenerationLabel(StrEnum):
    """FR-048. NEI is a first class label, not a missing value."""

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEI = "NEI"
    UNPARSEABLE = "UNPARSEABLE"
    BLOCKED = "BLOCKED"


class FreeGenerationRecord(BaseModel):
    """One free generation compliance judgment. FR-047, FR-048."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    sc_id: int
    category: SCCategoryId
    condition: ComplianceCondition
    label: FreeGenerationLabel
    transcript: str = ""
    tool_calls: tuple[str, ...] = ()
    raw_response: str = ""
    judge_model: str = ""


class InstanceKey(BaseModel):
    """Identity of one grid cell. Spec 12.2 stage 4 to 6 idempotency key."""

    model_config = ConfigDict(frozen=True)

    context_id: str
    sc_id: int
    strength: str
    explicitness: str
    injection_condition: InjectionCondition
    injection_seed: int
    compactor_id: str
    prompt_hash: str
    repetition_r: int | None = None
    target_tokens: int = Field(default=100000, ge=1)

    def instance_id(self) -> str:
        import hashlib

        payload = "\x00".join(
            [
                self.context_id,
                str(self.sc_id),
                self.strength,
                self.explicitness,
                self.injection_condition.value,
                str(self.injection_seed),
                self.compactor_id,
                self.prompt_hash,
                str(self.repetition_r),
                str(self.target_tokens),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
