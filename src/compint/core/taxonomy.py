"""The five category SC taxonomy. TASK-003, FR-001.

PAPER SPECIFICATION Table 13 (spec 3.4). The taxonomy is organized by WHAT the constraint
binds. The enumeration is CLOSED in research mode and OPEN in production mode, where an
`other` bucket exists and its rate is a monitored metric (FR-042).

Severity ordering is `ENGINEERING RECOMMENDATION` (assumption A-10, spec 14.7): losing an
Output constraint produces a formatting error, losing an Action constraint produces an
unauthorized tool call. The paper does not weight categories; this system argues it must.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from compint.core.models import SCCategoryId
from shared.errors import ConfigError

RESEARCH_CATEGORIES: tuple[SCCategoryId, ...] = (
    SCCategoryId.ACTION,
    SCCategoryId.INFORMATION,
    SCCategoryId.PROCESS,
    SCCategoryId.PREFERENCE,
    SCCategoryId.OUTPUT,
)


class CategoryDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: SCCategoryId
    name: str
    binds: str
    definition: str
    severity_rank: int
    production_only: bool = False


class Taxonomy(BaseModel):
    """Immutable per version (FR-004). Run manifests record the version."""

    model_config = ConfigDict(frozen=True)

    version: str
    source: str
    categories: tuple[CategoryDefinition, ...]
    severity_order: tuple[SCCategoryId, ...]

    def definition(self, category: SCCategoryId) -> CategoryDefinition:
        for entry in self.categories:
            if entry.id is category:
                return entry
        raise ConfigError(f"category {category} is not in taxonomy {self.version}")

    def severity_rank(self, category: SCCategoryId) -> int:
        """Lower is more severe. Drives budget eviction priority (spec 14.7 rule 1)."""
        return self.definition(category).severity_rank

    def research_categories(self) -> tuple[CategoryDefinition, ...]:
        return tuple(c for c in self.categories if not c.production_only)

    def assert_research_closed(self) -> None:
        """FR-001: research mode defines exactly five categories, no fallback."""
        ids = tuple(c.id for c in self.research_categories())
        if ids != RESEARCH_CATEGORIES:
            raise ConfigError(
                f"research taxonomy must be exactly {[c.value for c in RESEARCH_CATEGORIES]}, "
                f"got {[c.value for c in ids]}"
            )


def load_taxonomy(path: str | Path) -> Taxonomy:
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigError(f"taxonomy file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    taxonomy = Taxonomy.model_validate(raw)
    taxonomy.assert_research_closed()
    return taxonomy
