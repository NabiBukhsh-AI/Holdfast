"""Conversation embedding for topic cohesive stitching. TASK-007, FR-012.

PAPER SPECIFICATION: `Qwen/Qwen3-Embedding-0.6B`, L2 normalized.

`UNKNOWN` U-06: the conversation to string serialization strategy is not stated and it changes
neighbor ranking, therefore it changes every stitched context. The strategy is read from config
and recorded in the manifest rather than chosen implicitly.

`REPRODUCIBILITY HAZARD` spec 11.2: stitching is deterministic given the corpus, the embedding
model, the index, and the ordering, but NOT across embedding model revisions. Pin the model by
commit hash before any reported run; `ContextConfig.require_embedding_revision()` enforces it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from compint.core.models import Conversation
from shared.errors import ConfigError


def serialize_conversation(conversation: Conversation, strategy: str) -> str:
    """Render a conversation to the string the embedding model sees.

    UNKNOWN U-06. Two strategies are implemented so the sensitivity check the spec asks for
    (neighbor overlap across serializations) is runnable rather than hypothetical.
    """
    if strategy == "role_prefixed_newline":
        return "\n".join(f"{m.role.value}: {m.content}" for m in conversation.messages)
    if strategy == "content_only":
        return "\n".join(m.content for m in conversation.messages)
    if strategy == "user_turns_only":
        return "\n".join(m.content for m in conversation.messages if m.role.value == "user")
    raise ConfigError(
        f"unknown serialization strategy {strategy}. "
        "This is UNKNOWN U-06 and must be an explicit config value."
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row wise L2 normalization. PAPER SPECIFICATION A.1.

    A zero vector cannot be normalized. Rather than dividing by zero or silently substituting
    an arbitrary unit vector, this raises: a zero embedding means the encoder failed on that
    input, and a fabricated direction would corrupt every neighbor ranking that touches it.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        zero_rows = np.flatnonzero(norms.ravel() == 0).tolist()
        raise ValueError(f"cannot L2 normalize zero vectors at rows {zero_rows[:10]}")
    normalized: np.ndarray = matrix / norms
    return normalized


class EmbeddingModel(Protocol):
    id: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class StubEmbeddingModel:
    """Deterministic pseudo random unit vectors seeded on content hash.

    Spec 23.9: CI replaces the embedding model with exactly this. It is deterministic, so
    stitching determinism tests are meaningful, and it never touches the network. It is NOT
    valid for a reported run: `reportable` is False and the context builder refuses it.
    """

    reportable = False

    def __init__(self, dimension: int = 64, model_id: str = "stub-embedding") -> None:
        self.id = model_id
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.empty((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False) % (2**32)
            vectors[row] = np.random.default_rng(seed).standard_normal(self.dimension)
        return l2_normalize(vectors)


class HuggingFaceEmbeddingModel:
    """The real encoder. PAPER SPECIFICATION Qwen/Qwen3-Embedding-0.6B."""

    reportable = True

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        revision: str | None = None,
        *,
        batch_size: int = 16,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigError(
                "context.embedding_backend=huggingface requires sentence-transformers. "
                "Install the research extra, or use backend=stub for tests."
            ) from exc
        self.id = model_id
        self._revision = revision
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_id, revision=revision)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raw = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(raw, dtype=np.float32))


def build_embedding_model(
    backend: str, model_id: str, revision: str | None = None, *, dimension: int = 64
) -> EmbeddingModel:
    if backend == "stub":
        return StubEmbeddingModel(dimension=dimension)
    if backend == "huggingface":
        return HuggingFaceEmbeddingModel(model_id, revision)
    raise ConfigError(f"unknown embedding backend {backend}")


def embed_conversations(
    conversations: Sequence[Conversation], model: EmbeddingModel, strategy: str
) -> np.ndarray:
    """Embed a corpus in dataset order. Returns an L2 normalized matrix."""
    texts = [serialize_conversation(c, strategy) for c in conversations]
    vectors = model.encode(texts)
    if vectors.shape[0] != len(conversations):
        raise ValueError(
            f"embedding model returned {vectors.shape[0]} vectors for {len(conversations)} inputs"
        )
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("embedding model did not return L2 normalized vectors")
    return vectors
