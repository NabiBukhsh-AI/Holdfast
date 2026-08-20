"""TASK-021 acceptance tests. Spec 6.13, 14.8, INV-5 and INV-7."""

from __future__ import annotations

from pydantic import BaseModel

from shared.assembly import assemble, render_registry_lines
from shared.delimiters import (
    REGISTRY_CLOSE,
    REGISTRY_OPEN,
    contains_registry_block,
    count_registry_blocks,
    strip_registry_blocks,
)


class Entry(BaseModel):
    """Minimal structural stand in for both arms' registry record types."""

    canonical_text: str
    is_active: bool = True


def test_empty_registry_returns_bare_summary() -> None:
    """Spec 14.8: an empty block would read as "there are no constraints", a false claim."""
    out = assemble("summary text", [])
    assert out.text == "summary text"
    assert out.report.registry_rendered is False
    assert REGISTRY_OPEN not in out.text


def test_registry_of_only_inactive_entries_returns_bare_summary() -> None:
    out = assemble("summary text", [Entry(canonical_text="revoked", is_active=False)])
    assert out.text == "summary text"
    assert out.report.active_count == 0


def test_bare_mode_is_literal_concatenation() -> None:
    """PAPER SPECIFICATION Equation 10 carries no markup."""
    out = assemble("summary", [Entry(canonical_text="Draft, never send.")], mode="bare")
    assert out.text == "summary\n\n- Draft, never send."
    assert REGISTRY_OPEN not in out.text


def test_delimited_mode_emits_the_marked_block() -> None:
    out = assemble("summary", [Entry(canonical_text="Draft, never send.")], mode="delimited")
    assert REGISTRY_OPEN in out.text
    assert REGISTRY_CLOSE in out.text
    assert "remain in effect" in out.text
    assert "- Draft, never send." in out.text


def test_registry_follows_the_summary() -> None:
    """Spec 6.13: S^t follows C(H^t), analogous to K_ub where the SC sits before the probe."""
    out = assemble("SUMMARY_MARKER", [Entry(canonical_text="CONSTRAINT_MARKER")])
    assert out.text.index("SUMMARY_MARKER") < out.text.index("CONSTRAINT_MARKER")


def test_render_registry_lines_skips_inactive() -> None:
    lines = render_registry_lines(
        [
            Entry(canonical_text="keep me"),
            Entry(canonical_text="drop me", is_active=False),
        ]
    )
    assert lines == "- keep me"


def test_strip_removes_prior_block() -> None:
    """Spec 14.8 step 3: without this the registry decays across successive compactions."""
    assembled = assemble(
        "summary", [Entry(canonical_text="Never send email.")], mode="delimited"
    ).text
    stripped = strip_registry_blocks(assembled)
    assert stripped == "summary"
    assert not contains_registry_block(stripped)


def test_double_compaction_leaves_exactly_one_block() -> None:
    """INV-7 and TASK-026: strip then reassemble must not stack registry blocks."""
    registry = [Entry(canonical_text="Confirm before acting.")]
    first = assemble("summary one", registry, mode="delimited").text
    # A second compaction event: the previous augmented context becomes input history.
    stripped = strip_registry_blocks(first + "\n\nmore conversation happened here")
    second = assemble(stripped, registry, mode="delimited").text
    assert count_registry_blocks(second) == 1


def test_strip_is_idempotent() -> None:
    text = assemble("summary", [Entry(canonical_text="x")], mode="delimited").text
    once = strip_registry_blocks(text)
    assert strip_registry_blocks(once) == once


def test_report_counts_are_reported() -> None:
    out = assemble(
        "summary",
        [
            Entry(canonical_text="a"),
            Entry(canonical_text="b"),
            Entry(canonical_text="c", is_active=False),
        ],
        mode="delimited",
    )
    assert out.report.active_count == 2
    assert out.report.injected_count == 2
    assert out.report.block_chars > 0
    assert out.report.summary_chars == len("summary")
