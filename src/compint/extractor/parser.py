"""Strict extraction output parsing with evidence span validation. TASK-019.

Algorithm 14.5 steps 3 and 4.

Two rules govern everything here:

1. **A parse failure is EXTRACTION_PARSE_ERROR, never an empty list.** An empty list is a
   meaningful extractor output: it means "this turn declares no constraint", which is the
   common case. Collapsing a parse failure into that same value would make a broken extractor
   indistinguishable from a working one on a turn with no constraint, which is precisely the
   silent failure this system exists to prevent (NFR-008).

2. **Evidence spans must appear verbatim in the CURRENT USER TURN.** `ENGINEERING
   RECOMMENDATION` spec 14.5 step 4: this guards against hallucinated evidence. A candidate
   with an unfindable span is rejected, its siblings are kept, and a metric is emitted.
"""

from __future__ import annotations

import json
import re
import unicodedata
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from compint.core.models import SCCategoryId


class ExtractionStatus(StrEnum):
    """Terminal states of one extraction call. Spec 14.5 failure conditions."""

    OK = "OK"
    # The SLM produced output that is not the agreed JSON shape. Never an empty list.
    EXTRACTION_PARSE_ERROR = "EXTRACTION_PARSE_ERROR"
    # The SLM was unreachable or errored. MUST be surfaced, never treated as "no constraints
    # found" (NFR-008).
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


class RejectionReason(StrEnum):
    HALLUCINATED_EVIDENCE = "HALLUCINATED_EVIDENCE"
    DUPLICATE_OF_REGISTRY = "DUPLICATE_OF_REGISTRY"
    EMPTY_CANONICAL_TEXT = "EMPTY_CANONICAL_TEXT"
    UNKNOWN_CATEGORY = "UNKNOWN_CATEGORY"


class ExtractedSC(BaseModel):
    """One constraint the extractor claims the current user turn declares."""

    model_config = ConfigDict(frozen=True)

    canonical_text: str = Field(min_length=1)
    # A short span drawn from the CURRENT USER TURN, validated as a substring of it.
    evidence_span: str = Field(min_length=1)
    # INFERENCE spec 14.5, from Table 14: category tagging enables per type analysis (FR-073).
    category: SCCategoryId = SCCategoryId.OTHER

    @field_validator("canonical_text", "evidence_span")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class RejectedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: RejectionReason
    detail: str
    payload: dict[str, str] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Everything one extraction call produced, including what it got wrong."""

    model_config = ConfigDict(frozen=True)

    status: ExtractionStatus
    extracted: tuple[ExtractedSC, ...] = ()
    rejected: tuple[RejectedCandidate, ...] = ()
    raw_response: str = ""
    detail: str = ""

    @property
    def is_ok(self) -> bool:
        return self.status is ExtractionStatus.OK

    @property
    def n_hallucinated(self) -> int:
        """Emitted as a metric. A rising rate means the SLM is inventing evidence."""
        return sum(1 for r in self.rejected if r.reason is RejectionReason.HALLUCINATED_EVIDENCE)


# The extractor is instructed to emit a bare JSON list. Some models wrap it in a fence or
# in a single key object anyway; both are recovered structurally, which is different from
# guessing at prose.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL)


def _normalize_for_match(text: str) -> str:
    """Fold whitespace and unicode punctuation for substring comparison.

    A model that reproduces a span with a curly apostrophe where the user typed a straight one
    has not hallucinated it. Normalizing here keeps the check honest without making it so
    strict that it rejects faithful spans.
    """
    # The noqa directives below are justified: this function exists to FOLD typographic
    # punctuation, so it has to name the characters it folds.
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("’", "'").replace("‘", "'")  # noqa: RUF001
    folded = folded.replace("“", '"').replace("”", '"')
    folded = folded.replace("–", "-").replace("—", "-")  # noqa: RUF001  # allow-dash
    return re.sub(r"\s+", " ", folded).strip().lower()


def evidence_span_is_present(span: str, user_message: str) -> bool:
    """Step 4: the span must be findable in the current user turn."""
    return _normalize_for_match(span) in _normalize_for_match(user_message)


def _extract_json_payload(raw: str) -> object:
    """Recover the JSON list from the response, or raise ValueError."""
    candidate = raw.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    if not candidate:
        raise ValueError("empty response")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        # A bare list may be embedded in a sentence. Recover the outermost brackets rather
        # than giving up, but never invent structure that is not there.
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"response is not JSON: {exc}") from exc
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as inner:
            raise ValueError(f"response is not JSON: {inner}") from inner


def parse_extraction(
    raw: str,
    current_user_message: str,
    *,
    allow_other_category: bool = True,
) -> ExtractionResult:
    """Algorithm 14.5 steps 3 and 4. Strict JSON, then evidence span validation."""
    try:
        payload = _extract_json_payload(raw)
    except ValueError as exc:
        return ExtractionResult(
            status=ExtractionStatus.EXTRACTION_PARSE_ERROR,
            raw_response=raw,
            detail=str(exc),
        )

    # The contract is a list. A single object is accepted as a one element list because that
    # is a shape variation, not a semantic guess. Anything else is a parse error.
    if isinstance(payload, dict):
        for key in ("constraints", "session_constraints", "extracted"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        return ExtractionResult(
            status=ExtractionStatus.EXTRACTION_PARSE_ERROR,
            raw_response=raw,
            detail=f"expected a JSON list, got {type(payload).__name__}",
        )

    extracted: list[ExtractedSC] = []
    rejected: list[RejectedCandidate] = []

    for position, item in enumerate(payload):
        if not isinstance(item, dict):
            return ExtractionResult(
                status=ExtractionStatus.EXTRACTION_PARSE_ERROR,
                raw_response=raw,
                detail=f"element {position} is {type(item).__name__}, expected an object",
            )
        canonical = str(item.get("canonical_text") or item.get("constraint") or "").strip()
        span = str(item.get("evidence_span") or item.get("evidence") or "").strip()
        raw_category = str(item.get("category") or "other").strip().lower()

        if not canonical:
            rejected.append(
                RejectedCandidate(
                    reason=RejectionReason.EMPTY_CANONICAL_TEXT,
                    detail=f"element {position} carries no canonical_text",
                    payload={"raw": json.dumps(item)[:500]},
                )
            )
            continue

        try:
            category = SCCategoryId(raw_category)
        except ValueError:
            rejected.append(
                RejectedCandidate(
                    reason=RejectionReason.UNKNOWN_CATEGORY,
                    detail=f"element {position} category {raw_category!r} is not in the taxonomy",
                    payload={"canonical_text": canonical},
                )
            )
            continue
        if category is SCCategoryId.OTHER and not allow_other_category:
            rejected.append(
                RejectedCandidate(
                    reason=RejectionReason.UNKNOWN_CATEGORY,
                    detail="the research taxonomy is closed and does not admit `other`",
                    payload={"canonical_text": canonical},
                )
            )
            continue

        if not span or not evidence_span_is_present(span, current_user_message):
            # Reject this candidate, KEEP its valid siblings, emit a metric.
            rejected.append(
                RejectedCandidate(
                    reason=RejectionReason.HALLUCINATED_EVIDENCE,
                    detail=(
                        f"evidence span {span[:120]!r} does not appear in the current user turn"
                    ),
                    payload={"canonical_text": canonical, "evidence_span": span[:500]},
                )
            )
            continue

        extracted.append(
            ExtractedSC(canonical_text=canonical, evidence_span=span, category=category)
        )

    return ExtractionResult(
        status=ExtractionStatus.OK,
        extracted=tuple(extracted),
        rejected=tuple(rejected),
        raw_response=raw,
    )
