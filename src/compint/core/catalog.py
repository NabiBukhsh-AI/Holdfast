"""The 15 SC catalog. TASK-003, FR-002.

PAPER SPECIFICATION Appendix Table 12 (spec 11.5). Exactly 15 SCs, exactly 3 per category,
enforced with a load time assertion. Immutable after load: any edit produces a new version
file and the run manifest records which version produced which numbers (FR-004).

`ENGINEERING NOTE` SC 1 and SC 2 are logically contradictory (one forbids confirmation
prompts, the other requires them). That is deliberate and useful: it proves the extractor is
doing constraint extraction rather than topic detection, and it is the fixture the production
conflict detector is tested against (TASK-023).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from compint.core.models import SCCategoryId, SideConstraint
from compint.core.taxonomy import RESEARCH_CATEGORIES
from shared.errors import ConfigError

EXPECTED_SC_COUNT = 15
EXPECTED_PER_CATEGORY = 3


class SCCatalog(BaseModel):
    """Frozen catalog. Lookup by id or category, never mutated."""

    model_config = ConfigDict(frozen=True)

    version: str
    source: str
    constraints: tuple[SideConstraint, ...]
    # PAPER SPECIFICATION spec 11.5: SCs whose compliance is directly determinable from
    # generated behavior, used by the free generation robustness arm (FR-047).
    free_generation_subset: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> SCCatalog:
        if len(self.constraints) != EXPECTED_SC_COUNT:
            raise ValueError(
                f"catalog must hold exactly {EXPECTED_SC_COUNT} SCs (FR-002), "
                f"got {len(self.constraints)}"
            )
        ids = [sc.id for sc in self.constraints]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate SC ids in catalog: {ids}")
        for category in RESEARCH_CATEGORIES:
            count = sum(1 for sc in self.constraints if sc.category is category)
            if count != EXPECTED_PER_CATEGORY:
                raise ValueError(
                    f"category {category.value} must hold exactly {EXPECTED_PER_CATEGORY} "
                    f"SCs (FR-002), got {count}"
                )
        if any(sc.category is SCCategoryId.OTHER for sc in self.constraints):
            raise ValueError("the research catalog must not use the production only category")
        missing = set(self.free_generation_subset) - set(ids)
        if missing:
            raise ValueError(f"free_generation_subset references unknown SC ids: {sorted(missing)}")
        return self

    def by_id(self, sc_id: int) -> SideConstraint:
        for sc in self.constraints:
            if sc.id == sc_id:
                return sc
        raise ConfigError(f"SC {sc_id} is not in catalog {self.version}")

    def by_category(self, category: SCCategoryId) -> tuple[SideConstraint, ...]:
        return tuple(sc for sc in self.constraints if sc.category is category)

    def free_generation_scs(self) -> tuple[SideConstraint, ...]:
        return tuple(self.by_id(sc_id) for sc_id in self.free_generation_subset)

    def __len__(self) -> int:
        return len(self.constraints)

    def __iter__(self):  # type: ignore[override]
        return iter(self.constraints)


def load_catalog(path: str | Path) -> SCCatalog:
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigError(f"SC catalog not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return SCCatalog.model_validate(raw)
