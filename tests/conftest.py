"""Shared fixtures. No GPU, no network, no spend: spec 23.9."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compint.core.catalog import SCCatalog, load_catalog  # noqa: E402
from compint.core.models import History, Message, Role  # noqa: E402
from compint.core.random_source import RandomSource  # noqa: E402
from compint.core.taxonomy import Taxonomy, load_taxonomy  # noqa: E402
from compint.core.tokenization import HeuristicTokenizer, Tokenizer  # noqa: E402
from shared.config import AppConfig, load_config  # noqa: E402
from shared.prompts import PromptRegistry  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def catalog() -> SCCatalog:
    return load_catalog(ROOT / "data" / "sc_catalog" / "v1.yaml")


@pytest.fixture(scope="session")
def taxonomy() -> Taxonomy:
    return load_taxonomy(ROOT / "data" / "taxonomy" / "v1.yaml")


@pytest.fixture(scope="session")
def base_config() -> AppConfig:
    return load_config(ROOT / "configs" / "base.yaml")


@pytest.fixture(scope="session")
def production_config() -> AppConfig:
    return load_config(ROOT / "configs" / "production" / "dev.yaml")


@pytest.fixture(scope="session")
def prompts() -> PromptRegistry:
    return PromptRegistry(ROOT / "prompts")


@pytest.fixture
def tokenizer() -> Tokenizer:
    return HeuristicTokenizer()


@pytest.fixture
def rng() -> RandomSource:
    return RandomSource(20260731)


def build_message(index: int, role: Role, content: str) -> Message:
    return Message(
        index=index, role=role, content=content, token_count=max(1, len(content) // 4)
    )


@pytest.fixture
def wildchat_history() -> History:
    """Dense user turns, no system prompt, no tools. Five user turns."""
    messages = []
    index = 0
    for turn in range(5):
        messages.append(build_message(index, Role.USER, f"user question number {turn}"))
        index += 1
        messages.append(build_message(index, Role.ASSISTANT, f"assistant answer number {turn}"))
        index += 1
    return History(messages=tuple(messages))


@pytest.fixture
def hermes_history() -> History:
    """One system prompt at index 0, nine user turns, many tool and thinking messages.

    Mirrors Table 1: roughly 120 messages against roughly 9 user turns, which is the shape
    that makes the user turn index space distinct from the message index space.
    """
    messages = [build_message(0, Role.SYSTEM, "you are an agent with tools")]
    index = 1
    for turn in range(9):
        messages.append(build_message(index, Role.USER, f"do task {turn} please"))
        index += 1
        messages.append(build_message(index, Role.THINKING, f"planning task {turn}"))
        index += 1
        for call in range(5):
            messages.append(
                Message(
                    index=index,
                    role=Role.ASSISTANT,
                    content=f"calling tool {call} for task {turn}",
                    tool_name="search",
                    tool_call_id=f"c{turn}_{call}",
                    token_count=8,
                )
            )
            index += 1
            messages.append(
                Message(
                    index=index,
                    role=Role.TOOL,
                    content=f"tool result {call} for task {turn}",
                    tool_name="search",
                    tool_call_id=f"c{turn}_{call}",
                    token_count=8,
                )
            )
            index += 1
        messages.append(build_message(index, Role.ASSISTANT, f"task {turn} complete"))
        index += 1
    return History(messages=tuple(messages))


@pytest.fixture
def openresearcher_history() -> History:
    """Exactly one user turn, then long autonomous tool cycles. The DEGENERATE case."""
    messages = [build_message(0, Role.USER, "research the state of context compaction")]
    index = 1
    for step in range(40):
        messages.append(
            Message(
                index=index,
                role=Role.ASSISTANT,
                content=f"search step {step}",
                tool_name="search",
                token_count=6,
            )
        )
        index += 1
        messages.append(
            Message(
                index=index,
                role=Role.TOOL,
                content=f"result {step}",
                tool_name="search",
                token_count=6,
            )
        )
        index += 1
    return History(messages=tuple(messages))
