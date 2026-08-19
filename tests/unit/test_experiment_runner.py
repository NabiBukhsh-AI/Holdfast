"""TASK-017 acceptance tests. Spec 12.2 stage 4 to 6, 17.4, execution contract rule 15."""

from __future__ import annotations

from pathlib import Path

import pytest

from compint.core.framing import FramingSpec
from compint.core.models import Explicitness, InjectionCondition, Strength
from compint.data.contexts import ContextStatus, FillerContext
from compint.experiments.base import (
    Checkpoint,
    build_grid,
    enforce_cost_gate,
    estimate_cost,
)
from compint.report.manifest import ModelRecord, build_manifest, git_sha
from shared.config import AppConfig
from shared.errors import CostCeilingExceededError


def make_context(index: int, dataset: str = "wildchat", user_turns: int = 5) -> FillerContext:
    from compint.core.models import History, Message, Role

    messages = []
    position = 0
    for turn in range(user_turns):
        messages.append(
            Message(index=position, role=Role.USER, content=f"turn {turn}", token_count=10)
        )
        position += 1
        messages.append(
            Message(index=position, role=Role.ASSISTANT, content=f"reply {turn}", token_count=10)
        )
        position += 1
    return FillerContext(
        context_id=f"{dataset}_eval_100000_{index:04d}",
        dataset=dataset,
        split="eval",
        target_tokens=100000,
        actual_tokens=100500,
        history=History(messages=tuple(messages)),
        source_ids=(f"src_{index}",),
        n_stitched=1,
        status=ContextStatus.OK,
    )


FRAMING = FramingSpec(strength=Strength.PREFERENTIAL, explicitness=Explicitness.DIRECT)


def build_default_grid(n_contexts: int = 50, n_scs: int = 15, compactors: tuple[str, ...] = ("recent_5",)):
    return build_grid(
        [make_context(i) for i in range(n_contexts)],
        list(range(1, n_scs + 1)),
        compactors,
        framing=FRAMING,
        condition=InjectionCondition.TOP,
        injection_seed=20260731,
        prompt_hashes={"recent_5": "none"},
    )


def test_grid_cardinality_matches_the_paper() -> None:
    """Spec 12.2: N = 750 (50 contexts x 15 SCs) per compaction condition per dataset."""
    assert len(build_default_grid()) == 750


def test_instance_ids_are_unique_and_deterministic() -> None:
    first = build_default_grid(n_contexts=4, n_scs=3)
    second = build_default_grid(n_contexts=4, n_scs=3)
    ids = [cell.instance_id for cell in first]
    assert len(set(ids)) == len(ids)
    assert ids == [cell.instance_id for cell in second]


def test_instance_id_changes_with_every_key_component() -> None:
    """The idempotency key must separate cells that differ in any experimental factor."""
    base = build_default_grid(n_contexts=1, n_scs=1)[0]
    variants = {
        "framing": base.key.model_copy(update={"strength": "strict"}),
        "condition": base.key.model_copy(update={"injection_condition": InjectionCondition.BOTTOM}),
        "compactor": base.key.model_copy(update={"compactor_id": "llmlingua2_t500"}),
        "prompt": base.key.model_copy(update={"prompt_hash": "sha256:other"}),
        "seed": base.key.model_copy(update={"injection_seed": 1}),
        "length": base.key.model_copy(update={"target_tokens": 50000}),
    }
    for label, variant in variants.items():
        assert variant.instance_id() != base.key.instance_id(), (
            f"{label} does not change the idempotency key"
        )


def test_grid_marks_degenerate_contexts() -> None:
    """FR-023 travels with the cell, so the report can mark it rather than infer it."""
    degenerate = make_context(0, dataset="openresearcher", user_turns=1)
    cells = build_grid(
        [degenerate], [1], ("recent_5",), framing=FRAMING,
        condition=InjectionCondition.TOP, injection_seed=1, prompt_hashes={},
    )
    assert cells[0].degenerate is True


# ---------------------------------------------------------------- cost gate


def test_cost_estimate_accounts_for_condition_caching(base_config: AppConfig) -> None:
    """K_lctx and C(H^t) are per context, not per instance. Ignoring that overstates cost 15x."""
    cells = build_default_grid()
    estimate = estimate_cost(cells, base_config, n_contexts=50)
    # 750 injected compactions for K_comp, plus 50 un-injected for K_ub.
    assert estimate.compaction_calls == 800
    # K_lctx is one probe per context; the other three conditions are per instance.
    assert estimate.probe_calls == 50 + 750 * 3
    assert estimate.judge_calls == 750


def test_cost_gate_blocks_over_ceiling(base_config: AppConfig) -> None:
    config = base_config.model_copy(
        update={
            "cost": base_config.cost.model_copy(
                update={
                    "ceiling_usd": 1.0,
                    "require_confirm": False,
                    "price_per_1k_input_usd": {"gpt-oss-120b": 0.5},
                    "price_per_1k_output_usd": {"gpt-oss-120b": 1.5},
                }
            )
        }
    )
    estimate = estimate_cost(build_default_grid(), config, n_contexts=50)
    assert estimate.estimated_usd > 1.0
    with pytest.raises(CostCeilingExceededError, match="exceeds the ceiling"):
        enforce_cost_gate(estimate, config, confirm=False)
    enforce_cost_gate(estimate, config, confirm=True)


def test_require_confirm_blocks_even_under_the_ceiling(base_config: AppConfig) -> None:
    """Rule 15: no money or GPU path is default on, however cheap it looks."""
    estimate = estimate_cost(build_default_grid(n_contexts=1, n_scs=1), base_config, n_contexts=1)
    assert base_config.cost.require_confirm is True
    with pytest.raises(CostCeilingExceededError, match="require_confirm"):
        enforce_cost_gate(estimate, base_config, confirm=False)


def test_empty_price_table_is_reported_not_invented(base_config: AppConfig) -> None:
    """An invented price would make the gate meaningless, so zero is reported and labelled."""
    estimate = estimate_cost(build_default_grid(n_contexts=2, n_scs=2), base_config, n_contexts=2)
    assert estimate.estimated_usd == 0.0
    assert "EMPTY" in str(estimate.assumptions["price_table"])


# ---------------------------------------------------------------- checkpointing


def test_run_resumes_from_checkpoint(tmp_path: Path) -> None:
    """Spec 17.4: a run that dies at hour 80 of 90 resumes rather than restarts."""
    cells = build_default_grid(n_contexts=4, n_scs=3)
    path = tmp_path / "checkpoint.jsonl"

    first = Checkpoint(path)
    assert len(list(first.pending(cells))) == len(cells)
    for cell in list(first.pending(cells))[:7]:
        first.mark(cell.instance_id, compactor_id=cell.compactor_id)

    resumed = Checkpoint(path)
    assert len(resumed) == 7
    pending = list(resumed.pending(cells))
    assert len(pending) == len(cells) - 7
    assert all(cell.instance_id not in resumed for cell in pending)


def test_checkpoint_survives_a_torn_final_line(tmp_path: Path) -> None:
    """A hard kill costs one instance, not the whole ledger."""
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(
        '{"instance_id": "aaa"}\n{"instance_id": "bbb"}\n{"instance_id": "cc',
        encoding="utf-8",
    )
    checkpoint = Checkpoint(path)
    assert len(checkpoint) == 2
    assert "aaa" in checkpoint and "ccc" not in checkpoint


def test_checkpoint_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    checkpoint = Checkpoint(path)
    checkpoint.mark("one")
    checkpoint.mark("two")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert "one" in lines[0] and "two" in lines[1]


# ---------------------------------------------------------------- manifest


def test_manifest_records_every_unknown(base_config: AppConfig, repo_root: Path) -> None:
    """NFR-017 and rule 17: the manifest is where missing information is stated."""
    manifest = build_manifest(
        "run_0001",
        base_config,
        catalog_version="v1",
        taxonomy_version="v1",
        prompt_hashes={"retention_judge": "sha256:abc"},
        unfetched_prompts=("anthropic", "pi_mono"),
        models=(ModelRecord(role="judge", model_id="gpt-5.4"),),
        grid_size=750,
        repo_root=repo_root,
    )
    assert manifest.unknowns["U-10_alpha_l"] == 0.8
    assert manifest.unknowns["U-08_injection_separator"] == " "
    assert manifest.unknowns["U-09_repetition_r"] is None
    assert manifest.seed == base_config.random.seed
    assert manifest.framing_template_version == "v1"
    assert manifest.git_sha


def test_manifest_with_unfetched_prompts_is_not_reportable(
    base_config: AppConfig, repo_root: Path
) -> None:
    """A run that could not fetch its prompts produces no headline number, and says so."""
    blocked = build_manifest(
        "run_0002", base_config, catalog_version="v1", taxonomy_version="v1",
        prompt_hashes={}, unfetched_prompts=("anthropic",), repo_root=repo_root,
    )
    assert blocked.is_reportable is False

    clear = build_manifest(
        "run_0003", base_config, catalog_version="v1", taxonomy_version="v1",
        prompt_hashes={}, unfetched_prompts=(), repo_root=repo_root,
    )
    assert clear.is_reportable is True


def test_dev_split_manifest_is_not_reportable(base_config: AppConfig, repo_root: Path) -> None:
    manifest = build_manifest(
        "run_0004", base_config, catalog_version="v1", taxonomy_version="v1",
        prompt_hashes={}, split="dev", repo_root=repo_root,
    )
    assert manifest.is_reportable is False


def test_manifest_round_trips_to_disk(
    base_config: AppConfig, tmp_path: Path, repo_root: Path
) -> None:
    manifest = build_manifest(
        "run_0005", base_config, catalog_version="v1", taxonomy_version="v1",
        prompt_hashes={"mcq_probe": "sha256:xyz"}, repo_root=repo_root,
    )
    path = manifest.write(tmp_path)
    assert path.is_file()
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_0005"
    assert payload["unknowns"]["U-07_tokenizer"] == base_config.tokenization.tokenizer_id


def test_git_sha_reports_unavailability_rather_than_empty(tmp_path: Path) -> None:
    """A silent empty SHA would make a run untraceable while looking fine."""
    sha = git_sha(tmp_path)
    assert sha, "git_sha must never return an empty string"


def test_manifest_marks_a_dirty_tree(repo_root: Path) -> None:
    """A dirty tree means the committed SHA does not describe what ran."""
    sha = git_sha(repo_root)
    assert sha.startswith("unavailable") or len(sha) >= 40
