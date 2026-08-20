"""Build the long context environments. TASK-010, FR-011 through FR-019.

Ingests the three corpora, partitions each pool into dev and eval BEFORE stitching, builds the
context sets, and prints the Table 1 statistics report that gates everything downstream.

`GATE` If the Table 1 statistics do not reproduce within tolerance, the contexts are wrong, so
every downstream number is wrong. This script exits non zero in that case rather than letting
a run proceed on bad contexts.

    python scripts/build_contexts.py --dataset all --target-tokens 100000 --confirm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compint.core.tokenization import build_tokenizer  # noqa: E402
from compint.data.context_builder import (  # noqa: E402
    ContextBuilder,
    assert_no_source_leakage,
    summarize,
)
from compint.data.embedding import build_embedding_model  # noqa: E402
from compint.data.ingest.base import read_jsonl  # noqa: E402
from compint.data.ingest.registry import DATASETS, build_adapter  # noqa: E402
from compint.data.splits import Split  # noqa: E402
from compint.data.stitching import InsufficientDataError  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.errors import ConfigError  # noqa: E402


def display_path(path: Path) -> str:
    """Repo relative when possible, absolute otherwise.

    --out-dir may legitimately point outside the repository, and a path helper is not a
    reason for the whole build to crash after the contexts were already written.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_rows(dataset: str, source_dir: Path) -> list[dict[str, object]]:
    """Read normalized JSONL for one corpus.

    The HuggingFace download path is deliberately NOT here. Corpora are fetched separately,
    pinned by revision (U-13), and staged as JSONL, so that context construction is a pure
    function of files on disk and can be re-run without touching the network.
    """
    path = source_dir / f"{dataset}.jsonl"
    if not path.is_file():
        raise ConfigError(
            f"{path} not found. Stage the corpus as JSONL first, pinned by revision (U-13). "
            "Context construction is deliberately offline so it is reproducible."
        )
    return list(read_jsonl(str(path)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--dataset", default="all", choices=["all", *DATASETS])
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "corpora")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "contexts")
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument(
        "--reportable",
        action="store_true",
        help="refuse stub models and require a pinned embedding revision",
    )
    parser.add_argument("--confirm", action="store_true", help="write the context sets")
    args = parser.parse_args()

    config = load_config(args.config)
    target = args.target_tokens or config.context.target_tokens
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    tokenizer = build_tokenizer(
        config.tokenization.backend,
        config.tokenization.tokenizer_id,
        chars_per_token=config.tokenization.heuristic_chars_per_token,
    )
    embedder = build_embedding_model(
        config.context.embedding_backend,
        config.context.embedding_model,
        config.context.embedding_revision,
    )
    builder = ContextBuilder(config, tokenizer, embedder, require_reportable=args.reportable)

    failures: list[str] = []
    for dataset in datasets:
        try:
            rows = load_rows(dataset, args.source_dir)
        except ConfigError as exc:
            # A missing corpus is an operator error with an obvious fix, so it gets a message
            # rather than a traceback.
            print(f"{dataset}: {exc}", file=sys.stderr)
            failures.append(f"{dataset}: corpus not staged")
            continue
        ingestion = build_adapter(dataset, tokenizer).to_conversations(rows)
        print(f"\n{dataset}: {len(ingestion.conversations)} conversations")
        if ingestion.quarantined:
            print(f"  quarantined {len(ingestion.quarantined)} ({ingestion.quarantine_rate:.2%})")
            for reason, count in sorted(ingestion.reasons().items()):
                print(f"    {reason}: {count}")
        if ingestion.quarantine_rate > 0.005:
            failures.append(
                f"{dataset} quarantine rate {ingestion.quarantine_rate:.2%} exceeds the 0.5% "
                "acceptance threshold (TASK-006)"
            )

        pool = builder.partition(ingestion.conversations, dataset)
        try:
            built = {
                Split.DEV.value: builder.build(
                    pool.dev, dataset=dataset, split=Split.DEV, target_tokens=target
                ),
                Split.EVAL.value: builder.build(
                    pool.eval, dataset=dataset, split=Split.EVAL, target_tokens=target
                ),
            }
        except InsufficientDataError as exc:
            # Emitting fewer contexts than asked for would silently shrink the grid, so the
            # builder refuses. Report it as a sizing problem rather than a crash.
            print(f"{dataset}: {exc}", file=sys.stderr)
            failures.append(f"{dataset}: {exc}")
            continue
        assert_no_source_leakage(built["dev"], built["eval"])

        for split, contexts in built.items():
            stats = summarize(contexts)
            print(
                f"  {split}: {stats.n_contexts} contexts, mean {stats.mean_tokens:.0f} tokens, "
                f"{stats.mean_turns:.2f} turns, {stats.mean_user_turns:.2f} user turns"
            )
            for column, values in stats.compare_to_table1().items():
                mark = "ok" if values["within_tolerance"] else "OUT OF TOLERANCE"
                print(
                    f"    Table 1 {column}: expected {values['expected']}, "
                    f"got {values['actual']:.2f} ({mark})"
                )
            if split == Split.EVAL.value:
                try:
                    stats.assert_within_tolerance()
                except ConfigError as exc:
                    failures.append(str(exc))
            if args.confirm:
                out = args.out_dir / dataset / split
                out.mkdir(parents=True, exist_ok=True)
                for context in contexts:
                    (out / f"{context.context_id}.json").write_text(
                        context.model_dump_json(indent=2), encoding="utf-8", newline="\n"
                    )
                (out / "statistics.json").write_text(
                    json.dumps(stats.model_dump(mode="json"), indent=2),
                    encoding="utf-8",
                    newline="\n",
                )
                print(f"    wrote {len(contexts)} contexts to {display_path(out)}")

    if failures:
        print("\nCONTEXT CONSTRUCTION FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print(
            "\nWrong contexts mean every downstream number is wrong. This is escalation "
            "trigger 2 in the execution contract, not a tolerance to widen.",
            file=sys.stderr,
        )
        return 1

    if not args.confirm:
        print("\n(dry run; pass --confirm to write the context sets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
