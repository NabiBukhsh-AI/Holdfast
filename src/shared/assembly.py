"""Equation 10: the single assemble() used by BOTH arms.

    H~^t = C(H^t) (+) S^t                                                   (10)

Spec 6.13, 14.8, TASK-021. INV-5 requires that the `K_ub` evaluation condition and the
production assembly path share one concatenation code path, so that the measured upper bound
is the mechanism actually shipped. That guarantee is structural: there is exactly one
implementation and it lives here, in `src/shared/`, which both `compint` and `scguard` import
and neither can shadow.

Two modes:
  bare       PAPER SPECIFICATION. Literal concatenation, for exact research reproduction.
  delimited  ENGINEERING RECOMMENDATION. Marked block a later compaction can strip (INV-7).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from shared.delimiters import REGISTRY_CLOSE, REGISTRY_OPEN, REGISTRY_PREAMBLE

AssemblyMode = Literal["bare", "delimited"]

# UNKNOWN: the paper writes (+) as textual concatenation without specifying the separator
# between C(H^t) and S^t. Recorded as U-20 in OPEN_QUESTIONS.md and stamped into manifests.
DEFAULT_JOIN = "\n\n"


@runtime_checkable
class RegistryEntryLike(Protocol):
    """Structural type of anything the assembler will render.

    Both `compint.extractor.registry_sim.SimEntry` and
    `scguard.registry.store.SessionConstraint` satisfy this. Keeping it structural is what
    lets one assemble() serve both arms without either arm importing the other.
    """

    @property
    def canonical_text(self) -> str: ...

    @property
    def is_active(self) -> bool: ...


class AssemblyReport(BaseModel):
    """What the assembly did, for the audit trail and for NFR-015 metrics."""

    model_config = ConfigDict(frozen=True)

    mode: AssemblyMode
    active_count: int
    injected_count: int
    registry_rendered: bool
    summary_chars: int
    block_chars: int


class AssemblyOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    report: AssemblyReport


def render_registry_lines(registry: Sequence[RegistryEntryLike]) -> str:
    """Bulleted canonical texts of the active entries, in registry order."""
    return "\n".join(f"- {entry.canonical_text}" for entry in registry if entry.is_active)


def assemble(
    compacted: str,
    registry: Sequence[RegistryEntryLike],
    *,
    mode: AssemblyMode = "bare",
    join: str = DEFAULT_JOIN,
) -> AssemblyOutput:
    """Produce H~^t = C(H^t) (+) S^t.

    Spec 14.8 step 7 and the edge case note: an empty registry returns the bare summary. An
    empty <session_constraints> block would read as "there are no constraints", which is a
    stronger and potentially false claim than silence.
    """
    active = [entry for entry in registry if entry.is_active]
    if not active:
        return AssemblyOutput(
            text=compacted,
            report=AssemblyReport(
                mode=mode,
                active_count=0,
                injected_count=0,
                registry_rendered=False,
                summary_chars=len(compacted),
                block_chars=0,
            ),
        )

    lines = render_registry_lines(active)
    if mode == "bare":
        # PAPER SPECIFICATION: Equation 10 is textual concatenation with no markup.
        block = lines
    else:
        block = f"{REGISTRY_OPEN}\n{REGISTRY_PREAMBLE}\n{lines}\n{REGISTRY_CLOSE}"

    text = f"{compacted}{join}{block}"
    return AssemblyOutput(
        text=text,
        report=AssemblyReport(
            mode=mode,
            active_count=len(active),
            injected_count=len(active),
            registry_rendered=True,
            summary_chars=len(compacted),
            block_chars=len(block),
        ),
    )
