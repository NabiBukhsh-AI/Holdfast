"""The ONLY source of randomness in the system.

Style rule 4 (spec 2.2): all randomness routed through a single injected `RandomSource`
object; no module level `random.seed()`. This matters because the paper supplies no seeds
(U-04), so the reproduction must at minimum be internally reproducible and must record which
seed produced which numbers.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


class RandomSource:
    """A seeded, inspectable, derivable random source.

    Derivation matters for the grid: every instance needs its own stream so that adding a
    compactor to a run does not shift the Multi injection draw of an unrelated cell. Streams
    are derived by name from the root seed rather than pulled sequentially from one generator.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def seed(self) -> int:
        return self._seed

    def derive(self, label: str) -> RandomSource:
        """A child source whose stream depends on this seed and the label, not on call order."""
        child_seed = random.Random(f"{self._seed}:{label}").randrange(2**63)
        return RandomSource(child_seed)

    def sample(self, population: Sequence[T], k: int) -> list[T]:
        """Uniform sample WITHOUT replacement. Spec 6.6, the Multi injection condition."""
        if k < 0:
            raise ValueError(f"sample size must be non negative, got {k}")
        if k > len(population):
            raise ValueError(f"cannot sample {k} items from a population of {len(population)}")
        return self._rng.sample(list(population), k)

    def shuffled(self, items: Sequence[T]) -> list[T]:
        out = list(items)
        self._rng.shuffle(out)
        return out

    def random(self) -> float:
        return self._rng.random()

    def choice(self, population: Sequence[T]) -> T:
        if not population:
            raise ValueError("cannot choose from an empty population")
        return self._rng.choice(list(population))

    def randrange(self, stop: int) -> int:
        return self._rng.randrange(stop)
