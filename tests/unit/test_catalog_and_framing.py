"""TASK-003 and TASK-004 acceptance tests. Spec 3.4, 6.7, 11.5."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from compint.core.catalog import SCCatalog, load_catalog
from compint.core.framing import (
    DEFAULT_EXPLICITNESS,
    DEFAULT_STRENGTH,
    FRAMING_TEMPLATES,
    TEMPLATE_VERSION,
    all_framings,
    frame,
    render_framing,
)
from compint.core.models import Explicitness, SCCategoryId, Strength
from compint.core.taxonomy import RESEARCH_CATEGORIES, Taxonomy


def test_catalog_has_15_scs_3_per_category(catalog: SCCatalog) -> None:
    """FR-002: exactly 15 SCs, exactly 3 per category, asserted at load time."""
    assert len(catalog) == 15
    for category in RESEARCH_CATEGORIES:
        assert len(catalog.by_category(category)) == 3


def test_catalog_holds_the_deliberately_contradictory_pair(catalog: SCCatalog) -> None:
    """SC 1 forbids confirmation prompts; SC 2 requires them. Spec 11.5 engineering note."""
    sc1 = catalog.by_id(1)
    sc2 = catalog.by_id(2)
    assert sc1.category is SCCategoryId.ACTION
    assert sc2.category is SCCategoryId.ACTION
    assert "Don't ask me to confirm" in sc1.body
    assert "wait for my go-ahead" in sc2.body


def test_catalog_immutable(catalog: SCCatalog) -> None:
    """FR-004: immutable per version. Any edit produces a new version file."""
    with pytest.raises(ValidationError):
        catalog.version = "v2"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        catalog.constraints[0].body = "rewritten"  # type: ignore[misc]


def test_catalog_records_every_table12_column(catalog: SCCatalog) -> None:
    """FR-003: id, category, body, probe query, both options, and provenance citation."""
    for sc in catalog.constraints:
        assert sc.body.strip()
        assert sc.probe_query.strip()
        assert sc.option_compliant.strip()
        assert sc.option_violating.strip()
        assert sc.citation.strip()


def test_free_generation_subset_matches_paper(catalog: SCCatalog) -> None:
    """PAPER SPECIFICATION spec 11.5: SC ids 1, 2, 3, 6, 7, 10."""
    assert catalog.free_generation_subset == (1, 2, 3, 6, 7, 10)
    assert len(catalog.free_generation_scs()) == 6


def test_catalog_rejects_wrong_shape(tmp_path: Path, repo_root: Path) -> None:
    import yaml

    raw = yaml.safe_load(
        (repo_root / "data" / "sc_catalog" / "v1.yaml").read_text(encoding="utf-8")
    )
    raw["constraints"] = raw["constraints"][:14]
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="exactly 15"):
        load_catalog(broken)


def test_taxonomy_is_closed_in_research_mode(taxonomy: Taxonomy) -> None:
    """FR-001: exactly the five research categories, with `other` production only."""
    assert tuple(c.id for c in taxonomy.research_categories()) == RESEARCH_CATEGORIES
    other = taxonomy.definition(SCCategoryId.OTHER)
    assert other.production_only is True


def test_taxonomy_severity_order_matches_assumption_a10(taxonomy: Taxonomy) -> None:
    """Action > Information > Process > Preference > Output (spec 14.7, A-10)."""
    ranks = [taxonomy.severity_rank(c) for c in RESEARCH_CATEGORIES]
    assert ranks == sorted(ranks)
    assert taxonomy.severity_rank(SCCategoryId.ACTION) < taxonomy.severity_rank(SCCategoryId.OUTPUT)


def test_framing_golden_60(catalog: SCCatalog, repo_root: Path) -> None:
    """TASK-004: all 60 renderings byte exact. Changing a template fails CI on purpose."""
    golden = json.loads(
        (repo_root / "tests" / "golden" / "framing_60_strings.json").read_text(encoding="utf-8")
    )
    assert golden["count"] == 60
    assert golden["template_version"] == TEMPLATE_VERSION
    produced: dict[str, str] = {}
    for sc in catalog.constraints:
        for framed in all_framings(sc):
            produced[f"{sc.id:02d}_{framed.strength.value}_{framed.explicitness.value}"] = (
                framed.rendered_text
            )
    assert produced == golden["strings"]


def test_framing_matches_paper_figure_2() -> None:
    """PAPER SPECIFICATION spec 6.7 Figure 2, rendered on "Reply only in bullet points"."""
    body = "Reply only in bullet points."
    assert (
        render_framing(body, Strength.PREFERENTIAL, Explicitness.CONTEXTUALIZED)
        == "Reply only in bullet points."
    )
    assert (
        render_framing(body, Strength.PREFERENTIAL, Explicitness.DIRECT)
        == "For the rest of this session, reply only in bullet points."
    )
    assert (
        render_framing(body, Strength.STRICT, Explicitness.CONTEXTUALIZED)
        == "This is an important constraint: reply only in bullet points."
    )
    assert (
        render_framing(body, Strength.STRICT, Explicitness.DIRECT)
        == "This is an important constraint: for the rest of this session, reply only in bullet points."
    )


def test_default_framing_is_direct_preferential() -> None:
    """PAPER SPECIFICATION spec 6.7: fixed across all main experiments."""
    assert DEFAULT_STRENGTH is Strength.PREFERENTIAL
    assert DEFAULT_EXPLICITNESS is Explicitness.DIRECT


def test_framing_table_covers_the_full_2x2() -> None:
    assert len(FRAMING_TEMPLATES) == 4
    for strength in Strength:
        for explicitness in Explicitness:
            assert (strength, explicitness) in FRAMING_TEMPLATES


def test_framing_does_not_lowercase_an_acronym() -> None:
    """Lowercasing a leading acronym would corrupt the constraint text."""
    assert render_framing(
        "URL shorteners are banned.", Strength.STRICT, Explicitness.CONTEXTUALIZED
    ) == ("This is an important constraint: URL shorteners are banned.")


def test_frame_stamps_template_version(catalog: SCCatalog) -> None:
    framed = frame(catalog.by_id(1))
    assert framed.template_version == TEMPLATE_VERSION
    assert framed.category is SCCategoryId.ACTION
    assert framed.sc_id == 1
