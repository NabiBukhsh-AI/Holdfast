"""The 2x2 framing renderer. TASK-004, FR-005.

PAPER SPECIFICATION spec 6.7. Two binary attributes control the SC surface form:

    Constraint Strength   strict        prepend "This is an important constraint:"
                          preferential  omit
    Explicitness          direct        prepend "For the rest of this session,"
                          contextualized omit

Figure 2 renders all four on "Reply only in bullet points", and those four strings are the
anchor for this implementation.

`UNKNOWN / REQUIRES VALIDATION` U-15: the exact composition order and punctuation when both
prefixes apply. The paper's literal prefix strings end with a colon and a period respectively,
but Figure 2 shows them merged with a comma and the body lowercased. The template table below
IS that decision, made once, versioned, and recorded in every run manifest. Sixty golden
strings (15 SCs x 4 framings) pin it; changing a template fails CI until the golden file is
deliberately updated, because any change invalidates every downstream result.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from compint.core.models import Explicitness, FramedSC, SideConstraint, Strength

TEMPLATE_VERSION = "v1"

STRICT_PREFIX = "This is an important constraint:"
SESSION_SCOPE_PHRASE = "For the rest of this session,"

# ENGINEERING RECOMMENDATION spec 6.7: an explicit, versioned, testable template table
# rather than string concatenation scattered through the codebase.
FRAMING_TEMPLATES: dict[tuple[Strength, Explicitness], str] = {
    (Strength.PREFERENTIAL, Explicitness.CONTEXTUALIZED): "{body_sentence}",
    (Strength.PREFERENTIAL, Explicitness.DIRECT): "For the rest of this session, {body_lower}",
    (Strength.STRICT, Explicitness.CONTEXTUALIZED): "This is an important constraint: {body_lower}",
    (Strength.STRICT, Explicitness.DIRECT): (
        "This is an important constraint: for the rest of this session, {body_lower}"
    ),
}

# PAPER SPECIFICATION spec 6.7 and Appendix A: the default for all main experiments.
DEFAULT_STRENGTH = Strength.PREFERENTIAL
DEFAULT_EXPLICITNESS = Explicitness.DIRECT


class FramingSpec(BaseModel):
    """A framing choice, carried through the grid and stored on every instance row."""

    model_config = ConfigDict(frozen=True)

    strength: Strength = DEFAULT_STRENGTH
    explicitness: Explicitness = DEFAULT_EXPLICITNESS

    @property
    def label(self) -> str:
        return f"{self.strength.value}_{self.explicitness.value}"


def _lowercase_first(body: str) -> str:
    """Lowercase the leading character so the body can follow a prefix.

    Only the first character is touched, and only when the first token is not an acronym.
    Lowercasing an acronym ("URL", "TCP") would corrupt the constraint text, and the catalog
    contains constraint bodies where that matters.
    """
    if not body:
        return body
    first_token = body.split(" ", 1)[0].strip(".,:;")
    if first_token.isupper() and len(first_token) > 1:
        return body
    return body[0].lower() + body[1:]


def render_framing(body: str, strength: Strength, explicitness: Explicitness) -> str:
    """Render one SC body under one framing. Pure, deterministic, template driven."""
    template = FRAMING_TEMPLATES[(strength, explicitness)]
    return template.format(body_sentence=body, body_lower=_lowercase_first(body))


def frame(
    sc: SideConstraint,
    strength: Strength = DEFAULT_STRENGTH,
    explicitness: Explicitness = DEFAULT_EXPLICITNESS,
) -> FramedSC:
    """Frame a catalog SC, stamping the template version onto the result."""
    return FramedSC(
        sc=sc,
        strength=strength,
        explicitness=explicitness,
        rendered_text=render_framing(sc.body, strength, explicitness),
        template_version=TEMPLATE_VERSION,
    )


def all_framings(sc: SideConstraint) -> tuple[FramedSC, ...]:
    """All four framings of one SC, in a stable order: the unit of the golden file."""
    return tuple(
        frame(sc, strength, explicitness)
        for strength in (Strength.PREFERENTIAL, Strength.STRICT)
        for explicitness in (Explicitness.CONTEXTUALIZED, Explicitness.DIRECT)
    )
