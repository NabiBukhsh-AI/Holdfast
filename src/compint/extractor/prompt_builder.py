"""Three input extraction prompt construction and fetched prompt validation. TASK-019, TASK-029.

Algorithm 14.5 step 1, FR-060 through FR-067.

`IMPLEMENTATION NOTE` spec 14.5: the three input structure is load bearing and MUST NOT be
simplified. Feeding the whole history destroys both the efficiency argument (cost is O(user
turns), not O(context tokens)) and the "user turns only" invariant, INV-3.

    current_user_message     the ONLY extraction source
    prev_assistant_message   reference resolution ONLY ("do that going forward")
    registry                 deduplication ONLY, never a source of new constraints

`UNKNOWN` U-03: the full extraction prompt including few shot examples is not printed by the
paper. It must be FETCHED. What this module supplies is (a) machine checkable requirements
derived from the paper's structured summary, used to VALIDATE a fetched prompt, and (b) the
data envelope that carries the three inputs. Neither is a substitute for the prompt text, and
`build_messages()` refuses to run without it.

`SECURITY` spec 26.4 and TASK-029: the registry is an instruction channel. Hostile content
inside a user turn (a pasted email, a fetched page, quoted tool output) may try to write
attacker controlled instructions into it. The data envelope below is structurally separated
and delimited so that instruction and data are distinguishable. Evidence span validation is
only a PARTIAL defense: an injected instruction genuinely present in the user turn will
validate. The behavioral defense is measured by the injection suite, not asserted here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from shared.errors import ConfigError, PromptNotFetchedError
from shared.prompts import Prompt

# Delimiters for the data envelope. Chosen to be implausible in ordinary prose so that user
# content cannot close the block by accident, and so an attempt to close it is visible.
CURRENT_TURN_OPEN = "<current_user_turn>"
CURRENT_TURN_CLOSE = "</current_user_turn>"
PREV_ASSISTANT_OPEN = "<previous_assistant_turn_for_reference_only>"
PREV_ASSISTANT_CLOSE = "</previous_assistant_turn_for_reference_only>"
REGISTRY_OPEN = "<existing_registry_for_deduplication_only>"
REGISTRY_CLOSE = "</existing_registry_for_deduplication_only>"

_ENVELOPE_TAGS = (
    CURRENT_TURN_OPEN,
    CURRENT_TURN_CLOSE,
    PREV_ASSISTANT_OPEN,
    PREV_ASSISTANT_CLOSE,
    REGISTRY_OPEN,
    REGISTRY_CLOSE,
)


class PromptRequirement(BaseModel):
    """One element the paper says the extraction prompt contains."""

    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    # Any one of these patterns satisfying the requirement is enough. They are deliberately
    # loose: this checks that a fetched prompt covers the documented ground, it does not
    # reconstruct wording.
    patterns: tuple[str, ...]
    spec_reference: str

    def satisfied_by(self, text: str) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in self.patterns)


# PAPER SPECIFICATION spec 14.5 step 1 and FR-065 through FR-067, as machine checkable
# requirements. These VALIDATE a fetched prompt. They do not stand in for one.
REQUIRED_PROMPT_ELEMENTS: tuple[PromptRequirement, ...] = (
    PromptRequirement(
        id="sc_definition",
        description="defines a session constraint as persisting across future turns",
        patterns=(r"persist", r"across .{0,20}turns", r"rest of (the|this) session"),
        spec_reference="FR-066, spec 14.5 step 1",
    ),
    PromptRequirement(
        id="empty_is_default",
        description="states that most turns contain no SC and empty is the default output",
        patterns=(r"most turns", r"empty (list|array)", r"\[\s*\]"),
        spec_reference="FR-065",
    ),
    PromptRequirement(
        id="persistence_criterion",
        description="asks whether the instruction would still apply to an unrelated later question",
        patterns=(r"unrelated", r"several turns later", r"still apply"),
        spec_reference="FR-066",
    ),
    PromptRequirement(
        id="exclusion_rules",
        description="excludes current task instructions, one off corrections, and filler",
        patterns=(r"one[- ]off", r"current task", r"do not extract", r"politeness"),
        spec_reference="FR-067",
    ),
    PromptRequirement(
        id="assistant_turn_is_reference_only",
        description="states the previous assistant turn is for reference resolution only",
        patterns=(r"reference", r"not .{0,30}extract .{0,30}assistant", r"resolve"),
        spec_reference="FR-062, INV-3",
    ),
    PromptRequirement(
        id="registry_is_dedup_only",
        description="states the registry is for suppressing duplicates only",
        patterns=(r"duplicat", r"paraphras", r"already"),
        spec_reference="FR-063",
    ),
    PromptRequirement(
        id="json_output_contract",
        description="specifies a JSON list with canonical text and an evidence span",
        patterns=(r"json", r"canonical", r"evidence"),
        spec_reference="FR-064",
    ),
)


class PromptValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_id: str
    prompt_hash: str
    satisfied: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing


def validate_fetched_prompt(prompt: Prompt) -> PromptValidationReport:
    """Check a FETCHED extraction prompt against the paper's structured summary.

    A missing element does not mean the fetch was wrong; the paper's summary may simply be
    looser than its own prompt. It means a human should look before the numbers are trusted,
    so the report is emitted into the run manifest rather than raising.
    """
    text = "\n".join(part for part in (prompt.system, prompt.user, prompt.text) if part)
    satisfied: list[str] = []
    missing: list[str] = []
    for requirement in REQUIRED_PROMPT_ELEMENTS:
        (satisfied if requirement.satisfied_by(text) else missing).append(requirement.id)
    return PromptValidationReport(
        prompt_id=prompt.id,
        prompt_hash=prompt.content_hash,
        satisfied=tuple(satisfied),
        missing=tuple(missing),
    )


def sanitize_envelope_markup(text: str) -> str:
    """Neutralize envelope delimiters appearing inside user supplied content.

    ENGINEERING RECOMMENDATION spec 26.4. A user turn containing `</current_user_turn>` would
    otherwise close the data block early and let the remainder read as instructions. Escaping
    the tags keeps the content intact and visible while removing its structural power.
    """
    out = text
    for tag in _ENVELOPE_TAGS:
        out = out.replace(tag, tag.replace("<", "&lt;").replace(">", "&gt;"))
    return out


def build_data_envelope(
    current_user_message: str,
    previous_assistant_message: str | None,
    registry_texts: Sequence[str],
) -> str:
    """The three inputs, structurally separated and labelled by their permitted use."""
    if not current_user_message.strip():
        raise ConfigError("the current user message is the only extraction source and is empty")

    parts = [
        f"{CURRENT_TURN_OPEN}\n{sanitize_envelope_markup(current_user_message)}\n{CURRENT_TURN_CLOSE}"
    ]
    if previous_assistant_message is not None and previous_assistant_message.strip():
        parts.append(
            f"{PREV_ASSISTANT_OPEN}\n"
            f"{sanitize_envelope_markup(previous_assistant_message)}\n"
            f"{PREV_ASSISTANT_CLOSE}"
        )
    if registry_texts:
        listed = "\n".join(f"- {sanitize_envelope_markup(text)}" for text in registry_texts)
        parts.append(f"{REGISTRY_OPEN}\n{listed}\n{REGISTRY_CLOSE}")
    return "\n\n".join(parts)


def build_messages(
    prompt: Prompt | None,
    current_user_message: str,
    previous_assistant_message: str | None = None,
    registry_texts: Sequence[str] = (),
) -> tuple[str | None, str]:
    """Render (system, user) for one extraction call.

    Raises PromptNotFetchedError when the extraction prompt has not been fetched. There is no
    fallback prompt, because a reconstructed one would produce an extractor whose measured
    retention is not the paper's extractor's retention.
    """
    if prompt is None:
        raise PromptNotFetchedError(
            "sc_extractor",
            "U-03",
            "the released code of the paper; the paper prints only a structured summary",
        )
    envelope = build_data_envelope(
        current_user_message, previous_assistant_message, registry_texts
    )
    body = prompt.user if prompt.user is not None else prompt.text
    if body is None:
        raise ConfigError(f"extraction prompt {prompt.id} carries no user body")
    # The fetched prompt may or may not declare a placeholder for the inputs. When it does,
    # bind it; when it does not, the envelope follows the instruction text.
    if "{inputs}" in body:
        user = body.replace("{inputs}", envelope)
    else:
        user = f"{body}\n\n{envelope}"
    return prompt.system, user
