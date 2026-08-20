"""Tokenizer and RandomSource tests. Spec 11.1 (U-07), style rule 4.

The tiktoken and HuggingFace backends are exercised against fake modules injected into
`sys.modules`. That tests OUR wrapper logic (count, truncation slicing, reportability) without
pulling multi-hundred-megabyte optional dependencies into CI, and it also lets the
missing-dependency error paths be tested, which is the behaviour an operator actually hits
first.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from compint.core.random_source import RandomSource
from compint.core.tokenization import (
    HeuristicTokenizer,
    HuggingFaceTokenizer,
    TiktokenTokenizer,
    assert_reportable,
    build_tokenizer,
)
from shared.errors import ConfigError


@contextmanager
def fake_module(name: str, module: types.ModuleType) -> Iterator[None]:
    """Install a stand-in module for the duration of one test."""
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class FakeEncoding:
    """Whitespace tokenizer standing in for a BPE encoding."""

    def encode(self, text: str, disallowed_special: object = ()) -> list[int]:
        return [len(word) for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join("x" * i for i in ids)


def tiktoken_module() -> types.ModuleType:
    module = types.ModuleType("tiktoken")
    module.get_encoding = lambda name: FakeEncoding()  # type: ignore[attr-defined]
    return module


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model_id: str, revision: str | None = None) -> FakeAutoTokenizer:
        return FakeAutoTokenizer()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(word) for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join("y" * i for i in ids)


def transformers_module() -> types.ModuleType:
    module = types.ModuleType("transformers")
    module.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    return module


# ---------------------------------------------------------------- heuristic


def test_heuristic_counts_and_truncates() -> None:
    tokenizer = HeuristicTokenizer(chars_per_token=4.0)
    assert tokenizer.count("") == 0
    assert tokenizer.count("abcd") == 1
    assert tokenizer.count("abcde") == 2
    assert tokenizer.truncate("abcdefgh", 1) == "abcd"
    assert tokenizer.truncate("abcdefgh", 0) == ""


def test_heuristic_rejects_a_non_positive_ratio() -> None:
    with pytest.raises(ConfigError, match="must be positive"):
        HeuristicTokenizer(chars_per_token=0)


def test_heuristic_is_never_reportable() -> None:
    """U-07: an approximation must not define context lengths for a published number."""
    with pytest.raises(ConfigError, match="approximation"):
        assert_reportable(HeuristicTokenizer())


# ---------------------------------------------------------------- tiktoken


def test_tiktoken_backend_counts_and_truncates() -> None:
    with fake_module("tiktoken", tiktoken_module()):
        tokenizer = TiktokenTokenizer("cl100k_base")
        assert tokenizer.id == "cl100k_base"
        assert tokenizer.count("one two three") == 3
        assert tokenizer.truncate("one two three", 2) == "xxx xxx"
        assert tokenizer.truncate("one two three", 0) == ""
        assert_reportable(tokenizer)


def test_tiktoken_missing_dependency_names_the_fix() -> None:
    previous = sys.modules.pop("tiktoken", None)
    sys.modules["tiktoken"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigError, match="requires the tiktoken package"):
            TiktokenTokenizer("cl100k_base")
    finally:
        sys.modules.pop("tiktoken", None)
        if previous is not None:
            sys.modules["tiktoken"] = previous


# ---------------------------------------------------------------- huggingface


def test_huggingface_backend_counts_and_truncates() -> None:
    with fake_module("transformers", transformers_module()):
        tokenizer = HuggingFaceTokenizer("Qwen/Qwen3-Embedding-0.6B", revision="abc123")
        assert tokenizer.id == "Qwen/Qwen3-Embedding-0.6B"
        assert tokenizer.count("one two three") == 3
        assert tokenizer.truncate("one two three", 2) == "yyy yyy"
        assert tokenizer.truncate("one two three", 0) == ""
        assert_reportable(tokenizer)


def test_huggingface_missing_dependency_names_the_fix() -> None:
    previous = sys.modules.pop("transformers", None)
    sys.modules["transformers"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigError, match="requires transformers"):
            HuggingFaceTokenizer("some/model")
    finally:
        sys.modules.pop("transformers", None)
        if previous is not None:
            sys.modules["transformers"] = previous


# ---------------------------------------------------------------- factory


def test_build_tokenizer_dispatches_every_backend() -> None:
    assert isinstance(build_tokenizer("heuristic", "h"), HeuristicTokenizer)
    with fake_module("tiktoken", tiktoken_module()):
        assert isinstance(build_tokenizer("tiktoken", "cl100k_base"), TiktokenTokenizer)
    with fake_module("transformers", transformers_module()):
        assert isinstance(build_tokenizer("huggingface", "some/model"), HuggingFaceTokenizer)


def test_build_tokenizer_rejects_an_unknown_backend() -> None:
    """Backend selection is config, never inference."""
    with pytest.raises(ConfigError, match="unknown tokenization backend"):
        build_tokenizer("guess", "whatever")


# ---------------------------------------------------------------- RandomSource


def test_derive_depends_on_the_label_not_on_call_order() -> None:
    """Adding a compactor must not shift the Multi injection draw of an unrelated cell."""
    root = RandomSource(20260731)
    first = root.derive("cell:a").sample(range(100), 5)
    _ = root.derive("cell:unrelated").sample(range(100), 5)
    again = RandomSource(20260731).derive("cell:a").sample(range(100), 5)
    assert first == again


def test_derive_produces_different_streams_for_different_labels() -> None:
    root = RandomSource(1)
    assert root.derive("a").sample(range(1000), 10) != root.derive("b").sample(range(1000), 10)


def test_same_seed_reproduces_the_same_draws() -> None:
    assert RandomSource(7).sample(range(50), 10) == RandomSource(7).sample(range(50), 10)


def test_sample_is_without_replacement() -> None:
    drawn = RandomSource(3).sample(range(20), 12)
    assert len(drawn) == len(set(drawn)) == 12


def test_sample_rejects_impossible_sizes() -> None:
    source = RandomSource(3)
    with pytest.raises(ValueError, match="non negative"):
        source.sample(range(10), -1)
    with pytest.raises(ValueError, match="cannot sample"):
        source.sample(range(3), 5)


def test_shuffled_returns_a_new_list_and_preserves_membership() -> None:
    original = list(range(20))
    shuffled = RandomSource(11).shuffled(original)
    assert shuffled is not original
    assert original == list(range(20)), "the input must not be mutated"
    assert sorted(shuffled) == original


def test_random_choice_and_randrange_are_seeded() -> None:
    first = RandomSource(5)
    second = RandomSource(5)
    assert first.random() == second.random()
    assert first.choice("abcdef") == second.choice("abcdef")
    assert first.randrange(1000) == second.randrange(1000)


def test_choice_rejects_an_empty_population() -> None:
    with pytest.raises(ValueError, match="empty population"):
        RandomSource(1).choice([])


def test_seed_is_inspectable() -> None:
    """NFR-017: the seed lands in the manifest, so it has to be readable back off the source."""
    assert RandomSource(20260731).seed == 20260731
