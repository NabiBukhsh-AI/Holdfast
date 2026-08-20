"""COMPINT command line. Spec 20, TASK-017, TASK-018.

Every subcommand that spends money or GPU time requires `--confirm`. Every subcommand that
depends on an unfetched prompt fails loudly with the unknown id rather than proceeding.

    compint status                      what is built, what is gated, what is unknown
    compint catalog                     the 15 SCs and their framings
    compint framings --sc-id 15         all four renderings of one SC
    compint gate                        prompt fetch gate state, exit 1 when closed
    compint cost --config ...           pre flight projection
    compint verify --results ...        grade a completed run against spec 15.10
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from compint.core.catalog import load_catalog
from compint.core.framing import all_framings
from compint.core.taxonomy import load_taxonomy
from shared.config import load_config
from shared.prompts import PromptRegistry

app = typer.Typer(
    add_completion=False,
    help="COMPINT: measure session constraint loss under context compaction.",
    no_args_is_help=True,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry() -> PromptRegistry:
    return PromptRegistry(REPO_ROOT / "prompts")


@app.command()
def status() -> None:
    """Report what is built, what is gated, and what remains unknown."""
    catalog = load_catalog(REPO_ROOT / "data" / "sc_catalog" / "v1.yaml")
    taxonomy = load_taxonomy(REPO_ROOT / "data" / "taxonomy" / "v1.yaml")
    registry = _registry()
    missing = registry.missing_required()

    typer.echo("HoldFast / COMPINT")
    typer.echo(f"  catalog       {catalog.version}: {len(catalog)} SCs")
    typer.echo(
        f"  taxonomy      {taxonomy.version}: {len(taxonomy.research_categories())} research categories"
    )
    typer.echo(f"  prompts       loaded: {', '.join(registry.ids())}")
    if missing:
        typer.echo("  fetch gate    CLOSED")
        for requirement in missing:
            typer.echo(f"                {requirement.prompt_id} ({requirement.unknown_id})")
        typer.echo("  No Table 2 or Table 4 number is producible until the gate opens.")
        typer.echo("  Run: python scripts/fetch_prompts.py --confirm --source-dir <checkout>")
    else:
        typer.echo("  fetch gate    OPEN")


@app.command()
def gate() -> None:
    """Exit non zero while any externally sourced prompt is missing."""
    registry = _registry()
    missing = registry.missing_required()
    if not missing:
        typer.echo("BLOCKING GATE OPEN: every required prompt has been fetched.")
        raise typer.Exit(0)
    typer.echo("BLOCKING GATE CLOSED.", err=True)
    for requirement in missing:
        typer.echo(f"  {requirement.prompt_id} ({requirement.unknown_id})", err=True)
        typer.echo(f"    source: {requirement.source_hint}", err=True)
    typer.echo(
        "\nReconstructing these prompts is a spec violation, not a workaround: a reconstructed "
        "compaction prompt produces numbers that look like results and are not.",
        err=True,
    )
    raise typer.Exit(1)


@app.command()
def catalog(
    category: str = typer.Option("", help="filter to one category"),
    as_json: bool = typer.Option(False, "--json", help="emit JSON"),
) -> None:
    """List the SC catalog."""
    loaded = load_catalog(REPO_ROOT / "data" / "sc_catalog" / "v1.yaml")
    rows = [sc for sc in loaded.constraints if not category or sc.category.value == category]
    if as_json:
        typer.echo(json.dumps([sc.model_dump(mode="json") for sc in rows], indent=2))
        return
    for sc in rows:
        typer.echo(f"{sc.id:2d} [{sc.category.value:<11}] {sc.body}")
        typer.echo(f"     probe: {sc.probe_query}")


@app.command()
def framings(sc_id: int = typer.Option(..., help="catalog SC id, 1 to 15")) -> None:
    """Render all four framings of one SC. Spec 6.7."""
    loaded = load_catalog(REPO_ROOT / "data" / "sc_catalog" / "v1.yaml")
    for framed in all_framings(loaded.by_id(sc_id)):
        typer.echo(
            f"{framed.strength.value:<13} {framed.explicitness.value:<15} {framed.rendered_text}"
        )


@app.command()
def unknowns(config_path: Path = typer.Option(..., "--config", exists=True)) -> None:
    """Print every UNKNOWN and the value this config resolves it to."""
    config = load_config(config_path)
    for key, value in sorted(config.unknowns().items()):
        marker = "UNRESOLVED" if value is None else str(value)
        typer.echo(f"  {key:<36} {marker}")


@app.command()
def cost(
    config_path: Path = typer.Option(..., "--config", exists=True),
    n_contexts: int = typer.Option(50),
    n_scs: int = typer.Option(15),
) -> None:
    """Pre flight cost projection. Spec 12.2, execution contract rule 15."""
    from compint.experiments.base import estimate_cost_from_counts

    config = load_config(config_path)
    n_compactors = max(1, len(config.compactors))
    n_instances = n_contexts * n_scs * n_compactors

    estimate = estimate_cost_from_counts(
        config,
        n_instances=n_instances,
        n_contexts=n_contexts,
        n_scs=n_scs,
        n_compactors=n_compactors,
    )
    typer.echo(estimate.format())
    typer.echo(f"  instances:        {n_instances}")
    typer.echo(f"  compaction calls: {estimate.compaction_calls}")
    typer.echo(f"  judge calls:      {estimate.judge_calls}")
    typer.echo(f"  probe calls:      {estimate.probe_calls}")
    for key, value in estimate.assumptions.items():
        typer.echo(f"  assumption {key}: {value}")
    if estimate.estimated_usd == 0.0:
        typer.echo(
            "\nNo price table is configured, so the dollar figure is 0 by construction rather "
            "than by measurement. Populate cost.price_per_1k_input_usd before relying on the "
            "gate."
        )


@app.command()
def verify(results: Path = typer.Option(..., "--results", exists=True)) -> None:
    """Grade a completed run against the spec 15.10 acceptance thresholds."""
    from compint.report.tables import Table2, verify_reproduction

    payload = json.loads(results.read_text(encoding="utf-8"))
    table = Table2.model_validate(payload["table"])
    verdict = verify_reproduction(table, extractor_retention=payload.get("extractor_retention"))
    typer.echo(table.render())
    typer.echo("")
    typer.echo(verdict.render())
    raise typer.Exit(0 if verdict.succeeded else 1)


if __name__ == "__main__":
    app()
