"""Run one experiment suite end to end. TASK-017, TASK-018.

    python scripts/run_experiment.py --config configs/research/rq1_baseline.yaml --confirm

The order of operations is deliberate and is the order the gates need to fire in:

    1. resolve config, and refuse any UNKNOWN the suite actually depends on
    2. open the prompt fetch gate, or stop
    3. load the context sets built by scripts/build_contexts.py
    4. project cost and enforce the ceiling
    5. write the run manifest BEFORE the first model call
    6. execute, checkpointing every instance
    7. aggregate, grade, and write results

Steps 2 and 4 come before step 5 on purpose: a run that cannot legally proceed should not leave
a manifest implying it started.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compint.compactors.registry import COMPACTOR_SPECS, build_compactor  # noqa: E402
from compint.core.catalog import load_catalog  # noqa: E402
from compint.core.random_source import RandomSource  # noqa: E402
from compint.core.taxonomy import load_taxonomy  # noqa: E402
from compint.core.tokenization import build_tokenizer  # noqa: E402
from compint.data.contexts import FillerContext  # noqa: E402
from compint.eval.compliance import ComplianceHarness  # noqa: E402
from compint.eval.retention_judge import RetentionJudge  # noqa: E402
from compint.experiments.base import (  # noqa: E402
    Checkpoint,
    enforce_cost_gate,
    estimate_cost_from_counts,
)
from compint.experiments.rq1_baseline import (  # noqa: E402
    build_rq1_grid,
    build_table2,
    per_category_retention,
)
from compint.experiments.runner import GridRunner  # noqa: E402
from compint.report.manifest import ModelRecord, build_manifest  # noqa: E402
from compint.report.tables import render_per_category, verify_reproduction  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.errors import CostCeilingExceededError, HoldFastError  # noqa: E402
from shared.llm_client import LLMClient, OpenAICompatibleClient, StubLLMClient  # noqa: E402
from shared.prompts import PromptRegistry  # noqa: E402


def display_path(path: Path) -> str:
    """Repo relative when possible, absolute otherwise. --out-dir may point anywhere."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_contexts(directory: Path, dataset: str, split: str) -> list[FillerContext]:
    path = directory / dataset / split
    if not path.is_dir():
        raise HoldFastError(
            f"{path} not found. Build context sets first:\n"
            "  python scripts/build_contexts.py --dataset all --confirm"
        )
    contexts = [
        FillerContext.model_validate_json(file.read_text(encoding="utf-8"))
        for file in sorted(path.glob("*.json"))
        if file.name != "statistics.json"
    ]
    if not contexts:
        raise HoldFastError(f"{path} holds no contexts")
    return contexts


def build_client(config, role: str) -> LLMClient:  # type: ignore[no-untyped-def]
    """Real client when a base_url is configured, deterministic stub otherwise."""
    backend = getattr(config, role).backend
    if backend == "stub":
        return StubLLMClient(default_factory=lambda _request: "NO")
    if config.llm.base_url is None:
        raise HoldFastError(
            f"{role}.backend is {backend} but llm.base_url is unset. A real run needs an "
            "endpoint; set llm.base_url or switch the backend to stub."
        )
    import os

    return OpenAICompatibleClient(
        config.llm.base_url,
        api_key=os.environ.get(config.llm.api_key_env),
        max_concurrency=config.llm.max_concurrency,
        connect_timeout_s=config.llm.connect_timeout_s,
    )


async def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    catalog = load_catalog(ROOT / config.paths.catalog)
    taxonomy = load_taxonomy(ROOT / config.paths.taxonomy)
    prompts = PromptRegistry(ROOT / config.paths.prompts_dir)

    compactor_ids = list(config.compactors) or ["recent_5"]
    needs_prompt = [
        cid
        for cid in compactor_ids
        if COMPACTOR_SPECS.get(cid, None) and COMPACTOR_SPECS[cid].prompt_id
    ]

    # GATE 1. A reconstructed compaction prompt produces numbers that look like results and
    # are not, so a suite that needs one stops here rather than substituting anything.
    if needs_prompt:
        missing = {req.prompt_id for req in prompts.missing_required()}
        blocked = [cid for cid in needs_prompt if COMPACTOR_SPECS[cid].prompt_id in missing]
        if blocked:
            print(
                "BLOCKING GATE CLOSED. These compactors need prompts that have not been "
                f"fetched: {blocked}\n"
                "  Run: python scripts/fetch_prompts.py --confirm --source-dir <checkout>\n"
                "Nothing was run and no manifest was written.",
                file=sys.stderr,
            )
            return 2

    sc_ids = list(config.sc_subset) or [sc.id for sc in catalog.constraints]
    datasets = list(config.datasets) or ["wildchat"]
    split = "dev" if args.dev else "eval"

    contexts_by_dataset = {
        dataset: load_contexts(args.contexts_dir, dataset, split) for dataset in datasets
    }
    n_contexts = sum(len(v) for v in contexts_by_dataset.values())

    # GATE 2. Cost, before anything is spent and before a manifest exists.
    estimate = estimate_cost_from_counts(
        config,
        n_instances=n_contexts * len(sc_ids) * len(compactor_ids),
        n_contexts=n_contexts,
        n_scs=len(sc_ids),
        n_compactors=len(compactor_ids),
        conditions=list(config.conditions) or ["lctx", "lctx_sc", "comp", "ub"],
    )
    print(estimate.format())
    try:
        enforce_cost_gate(estimate, config, confirm=args.confirm)
    except CostCeilingExceededError as exc:
        print(f"\nCOST GATE: {exc}", file=sys.stderr)
        return 3

    run_id = args.run_id or f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(
        config.tokenization.backend,
        config.tokenization.tokenizer_id,
        chars_per_token=config.tokenization.heuristic_chars_per_token,
    )
    compactor_client = build_client(config, "compaction") if needs_prompt else None
    compactors = {
        cid: build_compactor(cid, config, tokenizer, client=compactor_client, prompts=prompts)
        for cid in compactor_ids
    }
    judge = RetentionJudge(
        build_client(config, "judge"),
        prompts.get("retention_judge"),
        config.judge.model,
        temperature=config.judge.temperature,
        timeout_s=config.judge.timeout_s,
        record_normalized=config.judge.record_normalized_verdict,
    )
    compliance = ComplianceHarness(
        build_client(config, "probe"),
        prompts.get("mcq_probe"),
        config.probe.model,
        temperature=config.probe.temperature,
        timeout_s=config.probe.timeout_s,
        option_order=config.probe.option_order,
        assembly_mode=config.assembly.mode,
        rng=RandomSource(config.random.seed).derive("mcq"),
    )

    # The manifest is written BEFORE the first model call, so a run that dies halfway still
    # records what it was and which UNKNOWN values it used.
    manifest = build_manifest(
        run_id,
        config,
        catalog_version=catalog.version,
        taxonomy_version=taxonomy.version,
        prompt_hashes=prompts.hashes(),
        unfetched_prompts=tuple(req.prompt_id for req in prompts.missing_required()),
        models=(
            ModelRecord(role="judge", model_id=config.judge.model),
            ModelRecord(role="probe", model_id=config.probe.model),
        ),
        grid_size=estimate.n_instances,
        estimated_cost_usd=estimate.estimated_usd,
        split=split,
        repo_root=ROOT,
    )
    manifest_path = manifest.write(out_dir)
    print(f"manifest: {display_path(manifest_path)}")
    if not manifest.is_reportable:
        print(
            "  NOTE: this run is not reportable as a headline number "
            f"(split={manifest.split}, unfetched={list(manifest.unfetched_prompts)})"
        )

    checkpoint = Checkpoint(out_dir / "checkpoint.jsonl")
    runner = GridRunner(
        catalog,
        compactors,
        judge,
        compliance,
        rng=RandomSource(config.random.seed),
        conditions=tuple(config.conditions) or ("lctx", "lctx_sc", "comp", "ub"),
        separator=config.injection.separator,
        direction=config.injection.direction,
        checkpoint=checkpoint,
    )

    cells = build_rq1_grid(
        contexts_by_dataset,
        sc_ids,
        compactor_ids,
        injection_seed=config.random.seed,
        prompt_hashes=prompts.hashes(),
    )
    all_contexts = [c for group in contexts_by_dataset.values() for c in group]
    print(f"executing {len(cells)} cells ({len(checkpoint)} already done)")
    result = await runner.run(cells, all_contexts)

    table = build_table2(result, split=split, run_id=run_id)
    verdict = verify_reproduction(table)
    print("\n" + table.render())
    print("\n" + render_per_category(per_category_retention(result)))  # type: ignore[arg-type]
    print("\n" + verdict.render())

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "table": table.model_dump(mode="json"),
                "verdict": verdict.model_dump(mode="json"),
                "call_counts": {
                    "compaction": result.compaction_calls,
                    "lctx_builds": result.lctx_builds,
                    "judge": result.judge_calls,
                    "probe": result.probe_calls,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest.completed("completed" if verdict.succeeded else "completed_with_failures").write(
        out_dir
    )
    print(f"\nresults: {display_path(out_dir / 'results.json')}")
    return 0 if verdict.succeeded else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contexts-dir", type=Path, default=ROOT / "artifacts" / "contexts")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dev", action="store_true", help="use the dev split; never reportable")
    parser.add_argument("--confirm", action="store_true", help="authorize spend")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except HoldFastError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
