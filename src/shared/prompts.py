"""Prompt registry: the single versioned home for every prompt in the system.

Spec section 11.4. Nothing outside `prompts/` may contain a prompt string; a CI check greps
for triple quoted strings elsewhere. Every prompt is hashed at load and the hash is recorded
in every result row, because a one character prompt change silently invalidates cross run
comparisons and that is the single largest reproducibility risk in the project.

Three prompts CANNOT be written here and must be fetched (TASK-001, unknowns U-01, U-02,
U-03). Requesting one before it has been fetched raises PromptNotFetchedError. That is the
blocking gate from spec 32.2, deliberately not a warning.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from shared.errors import ConfigError, PromptIntegrityError, PromptNotFetchedError

Provenance = Literal["paper_verbatim", "fetched", "engineering_recommendation"]


class FetchRequirement(BaseModel):
    """A prompt that must come from outside this repository."""

    model_config = ConfigDict(frozen=True)

    prompt_id: str
    unknown_id: str
    relative_path: str
    source_hint: str


# TASK-001 and spec 30.2. These four artifacts block the reproduction.
REQUIRED_FETCHED_PROMPTS: tuple[FetchRequirement, ...] = (
    FetchRequirement(
        prompt_id="anthropic",
        unknown_id="U-01",
        relative_path="compaction/anthropic.v1.yaml",
        source_hint=(
            "Anthropic platform documentation context compaction prompt, cited by the paper "
            "(last accessed 20 May 2026), or the reference repository "
            "https://github.com/ZhiqiEliWang/compaction-integrity"
        ),
    ),
    FetchRequirement(
        prompt_id="pi_mono",
        unknown_id="U-02",
        relative_path="compaction/pi_mono.v1.yaml",
        source_hint=(
            "pi-mono compaction prompt cited by the paper, or the reference repository "
            "https://github.com/ZhiqiEliWang/compaction-integrity"
        ),
    ),
    FetchRequirement(
        prompt_id="anthropic_sc_targeted",
        unknown_id="U-01",
        relative_path="compaction/anthropic_sc_targeted.v1.yaml",
        source_hint=(
            "the fetched Anthropic prompt plus the verbatim SC targeted addendum in "
            "prompts/compaction/sc_targeted_addendum.v1.yaml"
        ),
    ),
    FetchRequirement(
        prompt_id="sc_extractor",
        unknown_id="U-03",
        relative_path="extraction/sc_extractor.v1.yaml",
        source_hint=(
            "the released code of the paper; the paper prints only a structured summary of "
            "this prompt, not its text"
        ),
    ),
)

_REQUIRED_BY_ID = {req.prompt_id: req for req in REQUIRED_FETCHED_PROMPTS}


class Prompt(BaseModel):
    """One versioned prompt plus its provenance and content hash."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    version: str
    provenance: Provenance
    source_url: str | None = None
    fetched_at: datetime | None = None
    model_role: str | None = None
    placeholders: tuple[str, ...] = ()
    system: str | None = None
    user: str | None = None
    text: str | None = None
    sha256: str | None = None

    @model_validator(mode="after")
    def _has_content_and_provenance(self) -> Prompt:
        if self.system is None and self.user is None and self.text is None:
            raise ValueError(f"prompt {self.id} carries no text")
        # TASK-001 acceptance criterion: a prompt file without provenance is a hard failure.
        if self.provenance == "fetched" and not self.source_url:
            raise ValueError(f"fetched prompt {self.id} has no source_url")
        if self.provenance == "fetched" and self.fetched_at is None:
            raise ValueError(f"fetched prompt {self.id} has no fetched_at")
        return self

    @property
    def content_hash(self) -> str:
        """SHA-256 over the prompt's text fields, in a fixed order."""
        parts = [self.system or "", self.user or "", self.text or ""]
        digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def render(self, **values: str) -> tuple[str | None, str]:
        """Render the system and user halves, checking every declared placeholder is bound.

        Returns (system, user). Unknown or missing placeholders raise rather than producing
        a prompt with a literal brace left in it.
        """
        missing = set(self.placeholders) - set(values)
        if missing:
            raise ConfigError(f"prompt {self.id} missing placeholders: {sorted(missing)}")
        extra = set(values) - set(self.placeholders)
        if extra:
            raise ConfigError(f"prompt {self.id} given unknown placeholders: {sorted(extra)}")
        body = self.user if self.user is not None else self.text
        if body is None:
            raise ConfigError(f"prompt {self.id} has no user body to render")
        system = self.system.format(**values) if self.system else self.system
        return system, body.format(**values)


class PromptRegistry:
    """Loads, hashes, and serves every prompt under `prompts/`."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._prompts: dict[str, Prompt] = {}
        self._load()

    def _load(self) -> None:
        if not self._root.is_dir():
            raise ConfigError(f"prompts directory not found: {self._root}")
        for path in sorted(self._root.rglob("*.yaml")):
            with path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
            if not isinstance(raw, dict):
                raise ConfigError(f"prompt file is not a mapping: {path}")
            prompt = Prompt.model_validate(raw)
            # Integrity: a stored hash that no longer matches means the file was edited in
            # place after fetching. That silently invalidates every result row citing it.
            if prompt.sha256 is not None:
                stored = prompt.sha256 if prompt.sha256.startswith("sha256:") else f"sha256:{prompt.sha256}"
                if stored != prompt.content_hash:
                    raise PromptIntegrityError(
                        f"prompt {prompt.id} at {path} has stored hash {stored} but its text "
                        f"hashes to {prompt.content_hash}. The file was modified after fetch."
                    )
            if prompt.id in self._prompts:
                raise ConfigError(f"duplicate prompt id {prompt.id} at {path}")
            self._prompts[prompt.id] = prompt

    def get(self, prompt_id: str) -> Prompt:
        """Return a prompt, or fail loudly with the reason it is absent."""
        prompt = self._prompts.get(prompt_id)
        if prompt is not None:
            return prompt
        requirement = _REQUIRED_BY_ID.get(prompt_id)
        if requirement is not None:
            raise PromptNotFetchedError(
                prompt_id, requirement.unknown_id, requirement.source_hint
            )
        raise ConfigError(
            f"unknown prompt id {prompt_id}. Known ids: {sorted(self._prompts)}"
        )

    def has(self, prompt_id: str) -> bool:
        return prompt_id in self._prompts

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._prompts))

    def hashes(self) -> dict[str, str]:
        """Prompt id to content hash, the payload of tests/golden/prompt_hashes.json."""
        return {pid: p.content_hash for pid, p in sorted(self._prompts.items())}

    def missing_required(self) -> tuple[FetchRequirement, ...]:
        """Fetch requirements not yet satisfied. Empty tuple means the gate is open."""
        return tuple(req for req in REQUIRED_FETCHED_PROMPTS if req.prompt_id not in self._prompts)

    def assert_fetch_gate_open(self) -> None:
        """Raise unless every externally sourced prompt is present (TASK-001 gate)."""
        missing = self.missing_required()
        if missing:
            first = missing[0]
            raise PromptNotFetchedError(first.prompt_id, first.unknown_id, first.source_hint)


@lru_cache(maxsize=8)
def get_registry(root: str = "prompts") -> PromptRegistry:
    """Process wide prompt registry, loaded and hashed once."""
    return PromptRegistry(Path(root))
