"""TASK-001 gate tests and configuration tests. Spec 11.4, 30.2, execution contract rules 3 and 4."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from shared.config import load_config
from shared.errors import (
    BudgetNotConfiguredError,
    ConfigError,
    PromptIntegrityError,
    PromptNotFetchedError,
    UnresolvedUnknownError,
)
from shared.prompts import REQUIRED_FETCHED_PROMPTS, Prompt, PromptRegistry


def test_all_prompts_have_provenance(prompts: PromptRegistry) -> None:
    """TASK-001 acceptance: a prompt file without provenance fails the build."""
    for prompt_id in prompts.ids():
        prompt = prompts.get(prompt_id)
        assert prompt.provenance in ("paper_verbatim", "fetched", "engineering_recommendation")
        if prompt.provenance != "paper_verbatim":
            assert prompt.source_url, f"{prompt_id} has no source_url"


def test_prompt_hash_stability(prompts: PromptRegistry, repo_root: Path) -> None:
    """Spec 11.4: a one character prompt change must fail CI, not pass silently."""
    golden = json.loads(
        (repo_root / "tests" / "golden" / "prompt_hashes.json").read_text(encoding="utf-8")
    )
    assert prompts.hashes() == golden["hashes"]


def test_retention_judge_prompt_is_verbatim(prompts: PromptRegistry) -> None:
    """PAPER SPECIFICATION Appendix C.1. The paper prints this one in full."""
    prompt = prompts.get("retention_judge")
    assert prompt.provenance == "paper_verbatim"
    assert prompt.system is not None
    assert "Output only YES or NO." in prompt.system
    assert "Do NOT judge whether the assistant followed or acknowledged the SC" in prompt.system
    assert set(prompt.placeholders) == {"injected_sc", "compacted_context"}


def test_unfetched_compaction_prompt_raises_the_blocking_gate(tmp_path: Path) -> None:
    """Execution contract rule 4: fetch it or block. There is no fallback prompt.

    Exercised against an empty registry so the assertion holds whether or not this checkout
    has run the fetch. The gate is a mechanism, not a fact about the current working tree.
    """
    (tmp_path / "judging").mkdir()
    empty = PromptRegistry(tmp_path)
    with pytest.raises(PromptNotFetchedError) as excinfo:
        empty.get("anthropic")
    assert "BLOCKING GATE" in str(excinfo.value)
    assert "U-01" in str(excinfo.value)
    assert "fetch_prompts.py" in str(excinfo.value)


def test_every_required_prompt_reports_its_unknown_id(prompts: PromptRegistry) -> None:
    for requirement in REQUIRED_FETCHED_PROMPTS:
        if prompts.has(requirement.prompt_id):
            continue
        with pytest.raises(PromptNotFetchedError, match=requirement.unknown_id):
            prompts.get(requirement.prompt_id)


def test_fetch_gate_reports_all_missing(tmp_path: Path) -> None:
    """Every externally sourced prompt is accounted for, and the gate refuses to open."""
    (tmp_path / "judging").mkdir()
    empty = PromptRegistry(tmp_path)
    missing = {req.prompt_id for req in empty.missing_required()}
    assert missing == {"anthropic", "pi_mono", "anthropic_sc_targeted", "sc_extractor"}
    with pytest.raises(PromptNotFetchedError):
        empty.assert_fetch_gate_open()


def test_fetch_gate_opens_once_every_prompt_is_present(tmp_path: Path) -> None:
    """The complement: with all four present the gate opens and get() stops raising."""
    (tmp_path / "compaction").mkdir()
    (tmp_path / "extraction").mkdir()
    for prompt_id, subdir in (
        ("anthropic", "compaction"),
        ("pi_mono", "compaction"),
        ("anthropic_sc_targeted", "compaction"),
        ("sc_extractor", "extraction"),
    ):
        payload = {
            "id": prompt_id,
            "version": "v1",
            "provenance": "fetched",
            "source_url": "https://example.invalid/repo",
            "fetched_at": "2026-08-19T00:00:00Z",
            "text": f"stand in body for {prompt_id}",
        }
        (tmp_path / subdir / f"{prompt_id}.v1.yaml").write_text(
            yaml.safe_dump(payload), encoding="utf-8"
        )
    registry = PromptRegistry(tmp_path)
    assert registry.missing_required() == ()
    registry.assert_fetch_gate_open()
    assert registry.get("anthropic").provenance == "fetched"


def test_working_tree_gate_state_is_reported_not_asserted(prompts: PromptRegistry) -> None:
    """Informational: whichever state this checkout is in, it must be self consistent."""
    for requirement in prompts.missing_required():
        assert not prompts.has(requirement.prompt_id)
        with pytest.raises(PromptNotFetchedError):
            prompts.get(requirement.prompt_id)


def test_sc_targeted_addendum_is_stored_verbatim(prompts: PromptRegistry) -> None:
    """Spec 11.4 prints this sentence; it is an addendum, never a compaction prompt."""
    addendum = prompts.get("anthropic_sc_targeted_addendum")
    assert addendum.text is not None
    assert "preserve every user-provided session-level constraint" in addendum.text
    assert getattr(addendum, "composes_onto", None) == "anthropic"


def test_prompt_integrity_check_detects_post_fetch_edits(tmp_path: Path) -> None:
    """A stored hash that no longer matches means the file was edited after fetching."""
    (tmp_path / "j").mkdir()
    payload = {
        "id": "tampered",
        "version": "v1",
        "provenance": "fetched",
        "source_url": "https://example.invalid/prompt",
        "fetched_at": "2026-08-19T00:00:00Z",
        "text": "edited after fetch",
        "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    }
    (tmp_path / "j" / "p.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(PromptIntegrityError, match="modified after fetch"):
        PromptRegistry(tmp_path)


def test_fetched_prompt_without_source_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="no source_url"):
        Prompt.model_validate(
            {"id": "x", "version": "v1", "provenance": "fetched", "text": "body"}
        )


def test_render_rejects_missing_and_unknown_placeholders(prompts: PromptRegistry) -> None:
    judge = prompts.get("retention_judge")
    with pytest.raises(ConfigError, match="missing placeholders"):
        judge.render(injected_sc="s")
    with pytest.raises(ConfigError, match="unknown placeholders"):
        judge.render(injected_sc="s", compacted_context="c", bogus="b")


def test_render_binds_both_halves(prompts: PromptRegistry) -> None:
    system, user = prompts.get("retention_judge").render(
        injected_sc="SC_MARKER", compacted_context="CONTEXT_MARKER"
    )
    assert system is not None and "Output only YES or NO." in system
    assert "SC_MARKER" in user and "CONTEXT_MARKER" in user


def test_research_config_defaults_match_the_paper(repo_root: Path) -> None:
    config = load_config(repo_root / "configs" / "research" / "rq1_baseline.yaml")
    assert config.mode == "research"
    assert config.framing.default_strength == "preferential"
    assert config.framing.default_explicitness == "direct"
    assert config.injection.default_condition == "top"
    assert config.assembly.mode == "bare"
    assert config.compaction.recent_n == 5
    assert config.compaction.llmlingua_target_tokens == 500
    assert config.context.n_contexts == 50
    assert config.context.target_tokens == 100000


def test_production_config_enables_the_engineering_recommendations(repo_root: Path) -> None:
    config = load_config(repo_root / "configs" / "production" / "dev.yaml")
    assert config.mode == "production"
    assert config.assembly.mode == "delimited"
    assert config.registry.mode == "production"
    assert config.registry.conflict_detection is True
    assert config.registry.tombstoning is True
    assert config.catalog.allow_other_category is True


def test_unresolved_unknown_raises_with_its_id(repo_root: Path) -> None:
    """Execution contract rule 3: fail loudly, never invent a value."""
    config = load_config(repo_root / "configs" / "base.yaml")
    with pytest.raises(UnresolvedUnknownError, match="U-09"):
        config.injection.require_repetition_r()
    with pytest.raises(UnresolvedUnknownError, match="U-13"):
        config.context.require_embedding_revision()
    with pytest.raises(UnresolvedUnknownError, match="U-18"):
        config.registry.require_tau_dup()


def test_zero_budget_fails_loudly(repo_root: Path, tmp_path: Path) -> None:
    """Spec 14.7: budget set to 0 or unset fails at startup, never defaults to unbounded."""
    config = load_config(repo_root / "configs" / "base.yaml")
    broken = config.registry.model_copy(update={"budget_tokens": 0})
    with pytest.raises(BudgetNotConfiguredError, match="unbounded"):
        broken.require_budget_tokens()


def test_manifest_unknowns_cover_the_spec_list(repo_root: Path) -> None:
    """NFR-017: every UNKNOWN and its chosen value lands in the run manifest."""
    config = load_config(repo_root / "configs" / "research" / "rq1_baseline.yaml")
    unknowns = config.unknowns()
    for uid in ("U-04", "U-05", "U-06", "U-07", "U-08", "U-09", "U-10", "U-11", "U-12", "U-13", "U-14", "U-17"):
        assert any(key.startswith(uid) for key in unknowns), f"{uid} missing from the manifest"


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    """A typo that silently reverted an experiment to a default is the failure class this
    system exists to prevent."""
    path = tmp_path / "typo.yaml"
    path.write_text("mode: research\ninjection:\n  seperator: ' '\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_config_alpha_l_must_be_a_valid_fraction(tmp_path: Path) -> None:
    path = tmp_path / "bad_alpha.yaml"
    path.write_text("compaction:\n  alpha_l: 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
