"""Random search baseline — uniform sampling from the search space."""
from __future__ import annotations

import random
from typing import Any

from tba.types import EvalResult
from tba.optimizer.base import BaseOptimizer
from tba.optimizer.search_space import SearchSpace


class RandomSearchOptimizer(BaseOptimizer):
    """Uniformly random sampling. The minimum bar TBA must crush."""

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str,
        budget: int,
        seed: int = 42,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)
        self.rng = random.Random(seed)

    def ask(self) -> dict[str, Any]:
        return self.search_space.sample_random(self.rng)
