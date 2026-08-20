"""Tokenizer registry and token counting.

`CRITICAL` spec 11.1: token counting must use the tokenizer of the compactor model under
test, not one global tokenizer. "100K tokens" is not tokenizer invariant, and the paper does
not state which tokenizer defines the target length. That is UNKNOWN U-07. The choice is
recorded in the run manifest and held fixed across a comparison set, otherwise compactor
comparisons are confounded by context length differences.

The `heuristic` backend exists so that unit tests and CI need no model download. It is
deterministic and documented, and it is never valid for a reported run: `assert_reportable()`
refuses it.
"""

from __future__ import annotations

import math
from typing import Protocol

from shared.errors import ConfigError


class Tokenizer(Protocol):
    """Minimal surface. Nothing in the pipeline needs token ids, only counts and truncation."""

    id: str

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class HeuristicTokenizer:
    """Character ratio approximation. CI and unit tests only.

    UNKNOWN: spec 30.2 U-07. This backend is NOT valid for a reported run. It exists so that
    the 900 or so pure-function tests in this repository run in seconds with no model files.
    """

    reportable = False

    def __init__(self, chars_per_token: float = 4.0, tokenizer_id: str = "heuristic") -> None:
        if chars_per_token <= 0:
            raise ConfigError("chars_per_token must be positive")
        self.id = tokenizer_id
        self._ratio = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self._ratio))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        return text[: int(max_tokens * self._ratio)]


class TiktokenTokenizer:
    """Real BPE counts via tiktoken. Requires the optional dependency."""

    reportable = True

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ConfigError(
                "tokenization.backend=tiktoken requires the tiktoken package. "
                "Install it or switch to backend=heuristic for tests."
            ) from exc
        self.id = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        ids = self._encoding.encode(text, disallowed_special=())[:max_tokens]
        return str(self._encoding.decode(ids))


class HuggingFaceTokenizer:
    """The compactor model's own tokenizer. The only correct choice for a reported run."""

    reportable = True

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ConfigError(
                "tokenization.backend=huggingface requires transformers. "
                "Install it or switch to backend=heuristic for tests."
            ) from exc
        self.id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        ids = self._tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return str(self._tokenizer.decode(ids))


def build_tokenizer(
    backend: str,
    tokenizer_id: str,
    *,
    chars_per_token: float = 4.0,
    revision: str | None = None,
) -> Tokenizer:
    """Construct the configured tokenizer. Backend selection is config, never inference."""
    if backend == "heuristic":
        return HeuristicTokenizer(chars_per_token, tokenizer_id)
    if backend == "tiktoken":
        return TiktokenTokenizer(tokenizer_id)
    if backend == "huggingface":
        return HuggingFaceTokenizer(tokenizer_id, revision)
    raise ConfigError(f"unknown tokenization backend {backend}")


def assert_reportable(tokenizer: Tokenizer) -> None:
    """Refuse to publish numbers computed with the approximation backend."""
    if not getattr(tokenizer, "reportable", False):
        raise ConfigError(
            f"tokenizer {tokenizer.id} is an approximation and must not define context "
            "lengths for a reported run. Set tokenization.backend to tiktoken or huggingface."
        )
