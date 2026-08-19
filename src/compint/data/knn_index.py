"""Exact k-NN index with the doubling window neighbor lookup. TASK-007, FR-017.

PAPER SPECIFICATION Algorithm 1: a global k-NN index over the normalized embeddings, queried
for the highest ranked neighbor of the running centroid that still lies in the remaining pool.

`EXACT SEARCH ONLY` spec 17.3: approximate indexes introduce non determinism into context
construction, which directly harms reproducibility. FAISS is used when available and a numpy
brute force path is used otherwise; both are exact and both tie break identically (lowest
index wins), so the two backends produce the same context sets.

The doubling window exists because a k-NN backend returns a fixed size top-w list, not a
stream. When every returned neighbor has already been used, w doubles and the query repeats,
capped at the index size. A query at w = |I| that still finds no valid candidate RAISES. The
paper states this never triggered in their runs; the failure path exists anyway, because a
silent fallback here would quietly change which conversations were stitched together.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet

import numpy as np

from shared.errors import HoldFastError


class NeighborPoolExhaustedError(HoldFastError):
    """No valid candidate exists even after querying the entire index."""


class ExactKNNIndex:
    """Inner product search over L2 normalized vectors, which is cosine similarity."""

    def __init__(self, vectors: np.ndarray, *, backend: str = "auto") -> None:
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2D matrix, got shape {vectors.shape}")
        if vectors.shape[0] == 0:
            raise ValueError("cannot build an index over zero vectors")
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.size = int(self._vectors.shape[0])
        self.dimension = int(self._vectors.shape[1])
        self._faiss_index = None
        self.backend = "numpy"
        if backend in ("auto", "faiss"):
            self._faiss_index = self._try_faiss(backend)
            if self._faiss_index is not None:
                self.backend = "faiss"

    def _try_faiss(self, backend: str):  # type: ignore[no-untyped-def]
        try:
            import faiss
        except ImportError:
            if backend == "faiss":
                raise
            return None
        index = faiss.IndexFlatIP(self.dimension)
        index.add(self._vectors)
        return index

    def search(self, query: np.ndarray, w: int) -> list[int]:
        """Return the top-w indices by cosine similarity, highest first.

        Ties break on the lower index so that the FAISS and numpy backends agree and so that
        two runs over the same corpus produce identical results.
        """
        if w < 1:
            raise ValueError(f"w must be at least 1, got {w}")
        w = min(w, self.size)
        vector = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(vector, w)
            return [int(i) for i in indices[0] if i >= 0]
        scores = (self._vectors @ vector.ravel()).astype(np.float64)
        # Negate for ascending sort so equal scores order by ascending index.
        order = np.lexsort((np.arange(self.size), -scores))
        return [int(i) for i in order[:w]]

    def nearest_available(
        self,
        query: np.ndarray,
        available: AbstractSet[int],
        *,
        initial_w: int,
    ) -> int:
        """Highest ranked neighbor lying in `available`, doubling w until one is found.

        FR-017. Raises NeighborPoolExhaustedError at the cap rather than returning an
        arbitrary member of the pool.
        """
        if not available:
            raise NeighborPoolExhaustedError("the remaining pool is empty")
        w = max(1, min(initial_w, self.size))
        while True:
            for candidate in self.search(query, w):
                if candidate in available:
                    return candidate
            if w >= self.size:
                raise NeighborPoolExhaustedError(
                    f"scanned the entire index of {self.size} vectors and found no candidate "
                    f"in the remaining pool of {len(available)}"
                )
            w = min(self.size, w * 2)
