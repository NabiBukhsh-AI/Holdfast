"""Structural rules enforced in CI. Spec 20.1, execution contract rules 5 and 14.

These are the rules that keep the architecture load bearing rather than aspirational. Each one
failed silently in some other project before it became a test somewhere.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = Path(__file__).resolve().parents[2]

PYTHON_FILES = sorted(SRC.rglob("*.py"))


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_compint_does_not_import_scguard() -> None:
    """ENFORCED IMPORT RULE spec 20.1.

    Research code drifting into production only behavior would silently invalidate the
    reproduction, because the reproduction would then be measuring the hardened system rather
    than the one the source research describes.
    """
    offenders: list[str] = []
    for path in PYTHON_FILES:
        if "compint" not in path.parts:
            continue
        for module in imported_modules(path):
            if module == "scguard" or module.startswith("scguard."):
                offenders.append(f"{path.relative_to(REPO)} imports {module}")
    assert not offenders, "compint must not import scguard: " + "; ".join(offenders)


def test_scguard_does_not_import_compint_except_the_extractor() -> None:
    """The service reuses the extractor client and parser, which are shared research artifacts.

    Everything else in `compint` is offline benchmark machinery and must not reach production.
    """
    allowed = {
        "compint.extractor.client",
        "compint.extractor.parser",
        "compint.extractor.prompt_builder",
    }
    offenders: list[str] = []
    for path in PYTHON_FILES:
        if "scguard" not in path.parts:
            continue
        for module in imported_modules(path):
            if (module == "compint" or module.startswith("compint.")) and module not in allowed:
                offenders.append(f"{path.relative_to(REPO)} imports {module}")
    assert not offenders, "scguard may only import the extractor from compint: " + "; ".join(
        offenders
    )


def test_shared_imports_neither_arm() -> None:
    """`src/shared/` is the code that must be identical between the two arms."""
    offenders: list[str] = []
    for path in PYTHON_FILES:
        if "shared" not in path.parts:
            continue
        for module in imported_modules(path):
            if module.startswith(("compint", "scguard")):
                offenders.append(f"{path.relative_to(REPO)} imports {module}")
    assert not offenders, "shared must depend on neither arm: " + "; ".join(offenders)


def test_there_is_exactly_one_assemble_implementation() -> None:
    """INV-5 is structural: one assemble(), in shared, used by K_ub and by production."""
    definitions = [
        path.relative_to(REPO)
        for path in PYTHON_FILES
        if re.search(r"^def assemble\(", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    ]
    assert definitions == [Path("src/shared/assembly.py")], (
        f"assemble() must be defined once, in shared. Found: {definitions}"
    )


def test_no_prompt_strings_outside_the_prompts_directory() -> None:
    """Spec 20.1: nothing outside `prompts/` may contain a prompt string.

    The heuristic looks for long triple quoted strings that read like model instructions.
    Docstrings are excluded by construction, since this only inspects string literals that are
    not the first statement of a module, class, or function.
    """
    instruction_markers = re.compile(
        r"\b(you are an? (assistant|agent|model)|your task is|output only|respond with|"
        r"you will be given)\b",
        flags=re.IGNORECASE,
    )
    offenders: list[str] = []
    for path in PYTHON_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        docstrings = {
            ast.get_docstring(node, clean=False)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            text = node.value
            if text in docstrings or len(text) < 120:
                continue
            if instruction_markers.search(text):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, "prompt text must live under prompts/, not in source: " + "; ".join(
        offenders
    )


# Written as escapes so this file does not trip its own check.
EM_DASH = "—"
EN_DASH = "–"  # noqa: RUF001

DOC_FILES = sorted(
    path
    for path in REPO.rglob("*.md")
    if ".venv" not in path.parts and "node_modules" not in path.parts
)


def test_no_em_dashes_in_documentation() -> None:
    """Execution contract rule 14. Commas, colons, parentheses, or separate sentences.

    Code that MANIPULATES these characters has to name them: the evidence span normalizer
    folds typographic punctuation so a faithful span is not read as hallucinated. Such a line
    may opt out with an explicit `allow-dash` marker. Prose cannot.
    """
    offenders: list[str] = []
    for path in DOC_FILES + PYTHON_FILES:
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.split("\n"), start=1):
            if "allow-dash" in line:
                continue
            if EM_DASH in line or EN_DASH in line:
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert not offenders, "em or en dashes found: " + "; ".join(offenders[:20])


def test_no_bare_except_clauses() -> None:
    """Style rule 3 and execution contract rule 13: no bare except, no silent pass."""
    offenders: list[str] = []
    for path in PYTHON_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno} bare except")
                elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno} silent pass")
    assert not offenders, "; ".join(offenders)


def test_no_module_level_random_seed() -> None:
    """Style rule 4: all randomness routes through the injected RandomSource."""
    offenders: list[str] = []
    for path in PYTHON_FILES:
        if path.name == "random_source.py":
            continue
        source = path.read_text(encoding="utf-8")
        if re.search(r"^\s*random\.seed\(", source, flags=re.MULTILINE):
            offenders.append(str(path.relative_to(REPO)))
        if re.search(r"^\s*np\.random\.seed\(", source, flags=re.MULTILINE):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, "module level seeding found: " + "; ".join(offenders)


@pytest.mark.parametrize(
    "document",
    ["README.md", "REPRODUCTION.md", "OPEN_QUESTIONS.md", "DEVIATIONS.md"],
)
def test_required_documents_exist(document: str) -> None:
    """Spec 20: these four are part of the deliverable, not optional extras."""
    path = REPO / document
    assert path.is_file(), f"{document} is missing"
    assert len(path.read_text(encoding="utf-8")) > 500, f"{document} is a stub"


def test_open_questions_covers_every_blocking_unknown() -> None:
    text = (REPO / "OPEN_QUESTIONS.md").read_text(encoding="utf-8")
    for unknown in ("U-01", "U-02", "U-03", "U-09", "U-13", "U-16"):
        assert unknown in text, f"{unknown} is not documented in OPEN_QUESTIONS.md"
