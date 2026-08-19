"""Conflict detection between session constraints. TASK-023, Algorithm 14.6 tiers 3 and 4.

`ENGINEERING RECOMMENDATION` The source research never discusses conflicts or revocation. Its
registry is a flat append only list, so "Confirm before running commands" and "Never ask me to
confirm, just run them" would both sit in it and both be injected, leaving the agent to guess.

Two constraints CONFLICT when they bind the same action class with incompatible directives.
The canonical example is in the catalog itself: SC 1 and SC 2 are direct negations over the
same action class, and the catalog contains both on purpose.

A REFINEMENT is neither a duplicate nor a conflict: "actually, only for emails, not files"
narrows an existing constraint and both should remain active. Tier 4 must return INDEPENDENT
for it, which is why tier 3 escalates rather than deciding.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from scguard.registry.store import SessionConstraint
from shared.llm_client import LLMClient, LLMRequest


class Adjudication(StrEnum):
    """Tier 4 verdicts. Spec 14.6."""

    DUPLICATE = "DUPLICATE"
    CONFLICT = "CONFLICT"
    INDEPENDENT = "INDEPENDENT"


# Polarity markers. A constraint that forbids an action and one that requires it are the
# shape tier 3 is looking for. This is a cheap prefilter, not a decision: it escalates.
#
# Sequencing words ("before", "first", "then") are deliberately NOT markers. They appear in
# both directives with equal frequency, as the catalog's own contradictory pair shows:
#   SC 1  "Don't ask me to confirm BEFORE running commands, just do them."   forbidding
#   SC 2  "BEFORE you run a command ... wait for my go-ahead."               requiring
# Counting "before" as a requiring marker cancels SC 1's negation and the pair stops looking
# opposed, which is the one case this detector exists to catch.
NEGATIVE_MARKERS = (
    r"\bdon'?t\b", r"\bdo not\b", r"\bnever\b", r"\bno longer\b", r"\bstop\b",
    r"\bavoid\b", r"\brefrain\b", r"\bwithout\b", r"\bskip\b",
)
# Only modal words that express requirement belong here. An ACTION phrase such as "ask me"
# must not: it sits inside the negation's scope in "Don't ask me to confirm", so counting it
# as a requiring marker cancels the negation. Action phrases belong to ACTION_CLASS_PATTERNS,
# which is what the two constraints are compared ON, not what decides their direction.
POSITIVE_MARKERS = (
    r"\balways\b", r"\bmust\b", r"\bevery time\b", r"\bbe sure to\b", r"\brequire\b",
    r"\bmake sure\b", r"\bwait for\b",
)

# Action classes the polarity attaches to. Two constraints must be about the same thing before
# opposing polarity means anything: "never send email" and "always use metric" do not conflict.
ACTION_CLASS_PATTERNS: dict[str, tuple[str, ...]] = {
    "confirmation": (r"\bconfirm\b", r"\bapprov", r"\bgo-?ahead\b", r"\bpermission\b", r"\bask me\b"),
    "sending": (r"\bsend\b", r"\bemail\b", r"\bmessage\b", r"\bdraft\b"),
    "file_access": (r"\bfile\b", r"\bfolder\b", r"\bread\b", r"\bopen\b", r"\bdelete\b"),
    "search": (r"\bsearch\b", r"\bweb\b", r"\blook up\b"),
    "identity": (r"\bname\b", r"\bphone\b", r"\baddress\b", r"\bpersonal\b"),
    "formatting": (r"\bbullet\b", r"\bparagraph\b", r"\bformat\b", r"\bdigits?\b", r"\bword\b"),
    "units": (r"\bmetric\b", r"\bimperial\b", r"\bmeasurement\b", r"\bunits?\b"),
    "citation": (r"\bcite\b", r"\bsource\b", r"\barxiv\b", r"\bpaper\b"),
}


def action_classes(text: str) -> frozenset[str]:
    """Which action classes a constraint appears to bind."""
    lowered = text.lower()
    return frozenset(
        name
        for name, patterns in ACTION_CLASS_PATTERNS.items()
        if any(re.search(pattern, lowered) for pattern in patterns)
    )


# A polarity marker binds the action nearest to it, not the whole sentence. Beyond this
# window the two are unrelated clauses that happen to share a sentence.
POLARITY_WINDOW_CHARS = 40


def polarity(text: str) -> int:
    """Whole sentence polarity: -1 forbidding, +1 requiring, 0 neither or mixed.

    Reporting only. Conflict detection uses `class_polarity`, because a sentence can forbid
    one action while requiring another.
    """
    lowered = text.lower()
    negative = sum(1 for pattern in NEGATIVE_MARKERS if re.search(pattern, lowered))
    positive = sum(1 for pattern in POSITIVE_MARKERS if re.search(pattern, lowered))
    if negative > positive:
        return -1
    if positive > negative:
        return 1
    return 0


def _positions(text: str, patterns: tuple[str, ...]) -> list[int]:
    return [match.start() for pattern in patterns for match in re.finditer(pattern, text)]


def class_polarity(text: str, action_class: str) -> int:
    """Polarity of the directive that binds ONE action class. -1, +1, or 0.

    Polarity is a property of a (constraint, action class) pair rather than of the sentence.
    Without this distinction the catalog's compatible pair looks contradictory:

      SC 2  "... show me what you're about to do and WAIT FOR my go-ahead"   requires confirmation
      SC 3  "DON'T send any messages on my behalf, draft them ..."           forbids sending

    Both keep a human in the loop, but a sentence level reading sees +1 against -1 over the
    shared `sending` class and calls it a conflict. Tombstoning one of them would silently drop
    a user constraint, which is precisely the harm this system exists to prevent. Attaching
    each marker to its nearest action, within a bounded window, keeps them independent.
    """
    lowered = text.lower()
    class_positions = _positions(lowered, ACTION_CLASS_PATTERNS.get(action_class, ()))
    if not class_positions:
        return 0
    negatives = _positions(lowered, NEGATIVE_MARKERS)
    positives = _positions(lowered, POSITIVE_MARKERS)

    score = 0
    for anchor in class_positions:
        nearest_negative = min((abs(anchor - p) for p in negatives), default=None)
        nearest_positive = min((abs(anchor - p) for p in positives), default=None)
        candidates = [
            (distance, sign)
            for distance, sign in ((nearest_negative, -1), (nearest_positive, 1))
            if distance is not None and distance <= POLARITY_WINDOW_CHARS
        ]
        if not candidates:
            continue
        score += min(candidates)[1]
    return 0 if score == 0 else (1 if score > 0 else -1)


def conflicting_classes(first: str, second: str) -> frozenset[str]:
    """Action classes both constraints bind with opposing, class-local polarity."""
    shared = action_classes(first) & action_classes(second)
    return frozenset(
        name
        for name in shared
        if class_polarity(first, name) != 0
        and class_polarity(second, name) != 0
        and class_polarity(first, name) != class_polarity(second, name)
    )


def opposing_polarity(first: str, second: str) -> bool:
    """True when the two oppose over at least one shared action class."""
    return bool(conflicting_classes(first, second))


def shares_action_class(first: str, second: str) -> bool:
    return bool(action_classes(first) & action_classes(second))


class ConflictCandidate(BaseModel):
    """A pair tier 3 flagged for tier 4 adjudication."""

    model_config = ConfigDict(frozen=True)

    existing: SessionConstraint
    candidate_text: str
    shared_classes: tuple[str, ...]
    similarity: float | None = None
    reason: str = ""


class Adjudicator(Protocol):
    """Tier 4. Returns one of DUPLICATE, CONFLICT, INDEPENDENT."""

    async def adjudicate(self, first: str, second: str) -> Adjudication: ...


class HeuristicAdjudicator:
    """Deterministic tier 4 stand-in for CI and for deployments without an adjudication model.

    It is deliberately conservative: it returns CONFLICT only for a clear polarity inversion
    over a shared action class, and INDEPENDENT otherwise. Being wrong toward INDEPENDENT
    leaves both constraints active, which is the safer error: the agent sees both instructions
    rather than silently losing one.
    """

    async def adjudicate(self, first: str, second: str) -> Adjudication:
        if first.strip().casefold() == second.strip().casefold():
            return Adjudication.DUPLICATE
        if shares_action_class(first, second) and opposing_polarity(first, second):
            return Adjudication.CONFLICT
        return Adjudication.INDEPENDENT


ADJUDICATION_INSTRUCTION = (
    "You are comparing two instructions a user gave an assistant during one session.\n"
    "Answer with exactly one word:\n"
    "DUPLICATE if they express the same requirement.\n"
    "CONFLICT if following one necessarily violates the other.\n"
    "INDEPENDENT if both can be followed at once, including when one narrows the other.\n"
)


class LLMAdjudicator:
    """Tier 4 backed by a model. Bounded by the registry budget, so call volume is bounded."""

    def __init__(self, client: LLMClient, model: str, *, timeout_s: float = 15.0) -> None:
        self._client = client
        self._model = model
        self._timeout_s = timeout_s

    async def adjudicate(self, first: str, second: str) -> Adjudication:
        response = await self._client.complete(
            LLMRequest(
                model=self._model,
                system=ADJUDICATION_INSTRUCTION,
                user=f"[A]\n{first}\n\n[B]\n{second}",
                temperature=0.0,
                max_tokens=8,
                timeout_s=self._timeout_s,
            )
        )
        verdict = response.text.strip().upper()
        for candidate in Adjudication:
            if verdict.startswith(candidate.value):
                return candidate
        # An unparseable adjudication must not silently tombstone a constraint. Treating it as
        # INDEPENDENT keeps both active, which is the recoverable error.
        return Adjudication.INDEPENDENT


def find_conflict_candidates(
    candidate_text: str,
    active: Sequence[SessionConstraint],
    *,
    similarity: dict[str, float] | None = None,
) -> tuple[ConflictCandidate, ...]:
    """Tier 3: same action class AND opposing polarity, flagged for adjudication."""
    flagged: list[ConflictCandidate] = []
    for existing in active:
        opposed = conflicting_classes(existing.canonical_text, candidate_text)
        if not opposed:
            continue
        flagged.append(
            ConflictCandidate(
                existing=existing,
                candidate_text=candidate_text,
                shared_classes=tuple(sorted(opposed)),
                similarity=(similarity or {}).get(existing.constraint_id),
                reason=(
                    f"shared action class {sorted(opposed)} with opposing polarity "
                    f"({class_polarity(existing.canonical_text, sorted(opposed)[0])} vs "
                    f"{class_polarity(candidate_text, sorted(opposed)[0])})"
                ),
            )
        )
    return tuple(flagged)
