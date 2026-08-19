"""Fetch and pin the external prompts. TASK-001, the BLOCKING GATE.

The Anthropic compaction prompt, the pi-mono compaction prompt, and the full SC extraction
prompt are cited by the source research but not reprinted. They must be fetched from their
cited sources or from the reference repository.

`EXECUTION CONTRACT RULE 4` Never generate, paraphrase, or reconstruct prompt text. If a fetch
fails, this script exits non-zero with a clear message and writes nothing. Spec 32.4 makes an
unfetchable prompt a project level escalation, not an engineering problem to work around,
because a reconstructed compaction prompt produces numbers that look like Table 2 results and
are not.

Usage:

    python scripts/fetch_prompts.py --confirm
    python scripts/fetch_prompts.py --confirm --source-dir /path/to/compaction-integrity
    python scripts/fetch_prompts.py --status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shared.prompts import REQUIRED_FETCHED_PROMPTS, PromptRegistry  # noqa: E402

REFERENCE_REPO = "https://github.com/ZhiqiEliWang/compaction-integrity"


@dataclass(frozen=True)
class FetchTarget:
    """Where one prompt comes from and where it lands."""

    prompt_id: str
    unknown_id: str
    destination: Path
    # Candidate paths inside a local checkout of the reference repository.
    repo_paths: tuple[str, ...]
    # The primary citation, for the provenance record.
    source_url: str
    model_role: str
    output_wrapper: str | None = None
    composes_onto: str | None = None


TARGETS: tuple[FetchTarget, ...] = (
    FetchTarget(
        prompt_id="anthropic",
        unknown_id="U-01",
        destination=ROOT / "prompts" / "compaction" / "anthropic.v1.yaml",
        repo_paths=(
            "prompts/compaction/anthropic.txt",
            "prompts/anthropic_compaction.txt",
            "compaction_prompts/anthropic.txt",
        ),
        source_url="https://docs.anthropic.com/ (context compaction prompt, cited by the paper)",
        model_role="compactor",
        output_wrapper="summary_tag",
    ),
    FetchTarget(
        prompt_id="pi_mono",
        unknown_id="U-02",
        destination=ROOT / "prompts" / "compaction" / "pi_mono.v1.yaml",
        repo_paths=(
            "prompts/compaction/pi_mono.txt",
            "prompts/pi_mono_compaction.txt",
            "compaction_prompts/pi_mono.txt",
        ),
        source_url="pi-mono compaction prompt, cited by the paper",
        model_role="compactor",
        output_wrapper="markdown",
    ),
    FetchTarget(
        prompt_id="sc_extractor",
        unknown_id="U-03",
        destination=ROOT / "prompts" / "extraction" / "sc_extractor.v1.yaml",
        repo_paths=(
            "prompts/extraction/sc_extractor.txt",
            "prompts/extractor.txt",
            "extraction/prompt.txt",
        ),
        source_url=f"{REFERENCE_REPO} (the paper prints only a structured summary)",
        model_role="extractor",
    ),
)


def content_hash(system: str | None, user: str | None, text: str | None) -> str:
    """Must match shared.prompts.Prompt.content_hash exactly."""
    parts = [system or "", user or "", text or ""]
    return "sha256:" + hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def yaml_quote(value: str) -> str:
    return json.dumps(value)


def render_prompt_yaml(
    target: FetchTarget, text: str, fetched_at: str, origin: str
) -> str:
    """Write the prompt verbatim in a block scalar, with full provenance."""
    indented = "\n".join(f"  {line}" if line else "" for line in text.split("\n"))
    lines = [
        f"# FETCHED by scripts/fetch_prompts.py. UNKNOWN {target.unknown_id}.",
        "# Verbatim third party text. Do not edit: the stored sha256 is verified at import and",
        "# an edit here silently invalidates every result row that cites this hash.",
        f"id: {target.prompt_id}",
        "version: v1",
        "provenance: fetched",
        f"source_url: {yaml_quote(target.source_url)}",
        f"fetched_from: {yaml_quote(origin)}",
        f"fetched_at: {fetched_at}",
        f"model_role: {target.model_role}",
        f"sha256: {content_hash(None, None, text)}",
    ]
    if target.output_wrapper:
        lines.append(f"output_wrapper: {target.output_wrapper}")
    if target.composes_onto:
        lines.append(f"composes_onto: {target.composes_onto}")
    lines.append("text: |-")
    lines.append(indented)
    return "\n".join(lines) + "\n"


def read_from_source_dir(target: FetchTarget, source_dir: Path) -> tuple[str, str] | None:
    """Look for the prompt in a local checkout of the reference repository."""
    for relative in target.repo_paths:
        candidate = source_dir / relative
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text, str(candidate)
    return None


def fetch_over_http(target: FetchTarget, base_url: str) -> tuple[str, str] | None:
    """Fetch from a raw content base URL, for example a raw.githubusercontent.com prefix."""
    try:
        import httpx
    except ImportError:
        print("  httpx is not installed; cannot fetch over HTTP", file=sys.stderr)
        return None
    for relative in target.repo_paths:
        url = f"{base_url.rstrip('/')}/{relative}"
        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
        except httpx.HTTPError as exc:
            print(f"  {url}: {exc}", file=sys.stderr)
            continue
        if response.status_code == 200 and response.text.strip():
            return response.text.strip(), url
        print(f"  {url}: HTTP {response.status_code}", file=sys.stderr)
    return None


def build_sc_targeted(anthropic_text: str) -> str:
    """Compose the SC targeted variant from the FETCHED base plus the verbatim addendum.

    The addendum sentence IS printed by the paper and lives in
    prompts/compaction/sc_targeted_addendum.v1.yaml. The base is not, which is why this
    variant cannot exist until the Anthropic prompt is fetched.
    """
    import yaml

    addendum_path = ROOT / "prompts" / "compaction" / "sc_targeted_addendum.v1.yaml"
    addendum = yaml.safe_load(addendum_path.read_text(encoding="utf-8"))
    return f"{anthropic_text.rstrip()}\n\n{str(addendum['text']).strip()}"


def report_status() -> int:
    registry = PromptRegistry(ROOT / "prompts")
    missing = registry.missing_required()
    print("Prompt registry status")
    print(f"  loaded: {', '.join(registry.ids())}")
    if not missing:
        print("  BLOCKING GATE OPEN: every required prompt has been fetched.")
        return 0
    print("  BLOCKING GATE CLOSED. Missing:")
    for requirement in missing:
        print(f"    {requirement.prompt_id} ({requirement.unknown_id}) -> {requirement.relative_path}")
        print(f"      source: {requirement.source_hint}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm", action="store_true", help="write the fetched prompts to prompts/"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="local checkout of the reference repository to copy prompts from",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="raw content base URL to fetch from, for example a raw.githubusercontent.com prefix",
    )
    parser.add_argument("--status", action="store_true", help="report the gate state and exit")
    args = parser.parse_args()

    if args.status:
        return report_status()

    if args.source_dir is None and args.base_url is None:
        print(
            "ERROR: no source given. Pass --source-dir with a local checkout of\n"
            f"  {REFERENCE_REPO}\n"
            "or --base-url with a raw content prefix.\n\n"
            "This script will not invent prompt text. Execution contract rule 4: a\n"
            "reconstructed compaction prompt produces numbers that look like results and\n"
            "are not. If the prompts cannot be obtained, that is escalation trigger 1 in\n"
            "spec 32.4, and the correct action is to stop, not to work around it.",
            file=sys.stderr,
        )
        return 2

    fetched_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved: dict[str, str] = {}
    failures: list[str] = []

    for target in TARGETS:
        print(f"{target.prompt_id} ({target.unknown_id}):")
        found: tuple[str, str] | None = None
        if args.source_dir is not None:
            found = read_from_source_dir(target, args.source_dir)
        if found is None and args.base_url is not None:
            found = fetch_over_http(target, args.base_url)
        if found is None:
            failures.append(target.prompt_id)
            print("  NOT FOUND", file=sys.stderr)
            continue
        text, origin = found
        resolved[target.prompt_id] = text
        print(f"  found at {origin} ({len(text)} chars)")
        if args.confirm:
            target.destination.parent.mkdir(parents=True, exist_ok=True)
            target.destination.write_text(
                render_prompt_yaml(target, text, fetched_at, origin), encoding="utf-8", newline="\n"
            )
            print(f"  wrote {target.destination.relative_to(ROOT)}")

    # The SC targeted variant is derived, not fetched: base prompt plus the verbatim addendum.
    if "anthropic" in resolved:
        composed = build_sc_targeted(resolved["anthropic"])
        derived = FetchTarget(
            prompt_id="anthropic_sc_targeted",
            unknown_id="U-01",
            destination=ROOT / "prompts" / "compaction" / "anthropic_sc_targeted.v1.yaml",
            repo_paths=(),
            source_url="fetched Anthropic prompt plus the paper's verbatim SC targeted addendum",
            model_role="compactor",
            output_wrapper="summary_tag",
            composes_onto="anthropic",
        )
        print("anthropic_sc_targeted: composed from the fetched base plus the paper addendum")
        if args.confirm:
            derived.destination.write_text(
                render_prompt_yaml(derived, composed, fetched_at, "derived"),
                encoding="utf-8",
                newline="\n",
            )
            print(f"  wrote {derived.destination.relative_to(ROOT)}")

    if failures:
        print(
            f"\nERROR: could not fetch {failures}. Nothing was reconstructed and the gate\n"
            "stays closed. Do not write these prompts by hand.",
            file=sys.stderr,
        )
        return 1

    if not args.confirm:
        print("\n(dry run; pass --confirm to write the files)")
        return 0

    # Re-import to verify every stored hash matches its text, then refresh the golden fixture.
    registry = PromptRegistry(ROOT / "prompts")
    registry.assert_fetch_gate_open()
    golden = ROOT / "tests" / "golden" / "prompt_hashes.json"
    payload = {
        "_comment": (
            "Prompt content hashes. Spec 11.4: a one character prompt change silently "
            "invalidates cross run comparisons, so changing a hash must be deliberate."
        ),
        "hashes": registry.hashes(),
        "unfetched_required_prompts": [r.prompt_id for r in registry.missing_required()],
    }
    golden.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nBLOCKING GATE OPEN. Updated {golden.relative_to(ROOT)}.")
    print("Review the hash changes deliberately before committing them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
