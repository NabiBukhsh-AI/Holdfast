"""Run manifest: the reproducibility anchor. NFR-017, spec 19.2.

Every research run emits one of these. It captures the config hash, catalog version, context
set version, model identifiers, prompt hashes, seed, git SHA, and **every UNKNOWN parameter
with the value that run chose for it**.

Execution contract rule 17: when a decision depends on missing information, state the missing
information in the code, in OPEN_QUESTIONS.md, and in the run manifest. This module is the
third of those three. Without it, a number produced by this system cannot be traced back to
the choices that produced it, and several of those choices are values the paper never supplied.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shared.config import AppConfig


def git_sha(repo_root: Path | None = None) -> str:
    """Current commit, or an explicit marker. Never a silent empty string."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    if result.returncode != 0:
        return f"unavailable: {result.stderr.strip() or 'git rev-parse failed'}"
    sha = result.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    # A dirty tree means the committed SHA does not describe what actually ran. Say so.
    return f"{sha}-dirty" if dirty.stdout.strip() else sha


class ModelRecord(BaseModel):
    """One (role, model, revision, prompt) binding. Spec 17.6's configuration registry."""

    model_config = ConfigDict(frozen=True)

    role: str
    model_id: str
    revision: str | None = None
    prompt_id: str | None = None
    prompt_hash: str | None = None
    serving_config: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """One row per experiment invocation."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    experiment: str
    status: str = "running"
    config_hash: str
    catalog_version: str
    taxonomy_version: str
    context_set_version: str
    framing_template_version: str
    seed: int
    git_sha: str
    # Every UNKNOWN parameter and the value this run chose for it.
    unknowns: dict[str, Any] = Field(default_factory=dict)
    models: tuple[ModelRecord, ...] = ()
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    # Fetch requirements still unmet when the run started. A non empty list means the run
    # cannot produce headline numbers, and that fact travels with the results.
    unfetched_prompts: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    compactors: tuple[str, ...] = ()
    split: str = "eval"
    grid_size: int = 0
    estimated_cost_usd: float | None = None
    started_at: datetime
    completed_at: datetime | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def is_reportable(self) -> bool:
        """A run with unfetched prompts or a dev split produces no headline number."""
        return not self.unfetched_prompts and self.split == "eval"

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.manifest.json"
        path.write_text(self.to_json() + "\n", encoding="utf-8", newline="\n")
        return path

    def completed(self, status: str = "completed") -> RunManifest:
        return self.model_copy(update={"status": status, "completed_at": datetime.now(UTC)})


def build_manifest(
    run_id: str,
    config: AppConfig,
    *,
    catalog_version: str,
    taxonomy_version: str,
    prompt_hashes: dict[str, str],
    unfetched_prompts: tuple[str, ...] = (),
    models: tuple[ModelRecord, ...] = (),
    context_set_version: str = "v1",
    grid_size: int = 0,
    estimated_cost_usd: float | None = None,
    split: str = "eval",
    repo_root: Path | None = None,
    notes: tuple[str, ...] = (),
) -> RunManifest:
    """Assemble the manifest for one run, before any model is invoked."""
    return RunManifest(
        run_id=run_id,
        experiment=config.experiment or "unnamed",
        config_hash=config.config_hash(),
        catalog_version=catalog_version,
        taxonomy_version=taxonomy_version,
        context_set_version=context_set_version,
        framing_template_version=config.framing.template_version,
        seed=config.random.seed,
        git_sha=git_sha(repo_root),
        unknowns=config.unknowns(),
        models=models,
        prompt_hashes=prompt_hashes,
        unfetched_prompts=unfetched_prompts,
        datasets=config.datasets,
        compactors=config.compactors,
        split=split,
        grid_size=grid_size,
        estimated_cost_usd=estimated_cost_usd,
        started_at=datetime.now(UTC),
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tokenizer": config.tokenization.tokenizer_id,
            "tokenizer_backend": config.tokenization.backend,
            "embedding_backend": config.context.embedding_backend,
        },
        notes=notes,
    )
