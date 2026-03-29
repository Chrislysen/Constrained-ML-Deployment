"""Feasible-region TPE sampler.

After TBA finds feasibility, this module takes over the optimization phase.
Splits feasible history into good (top 30%) and bad (bottom 70%) by objective,
fits per-variable KDEs on each group, and samples from the good distribution
weighted by l(x)/g(x) — the core TPE idea, but restricted to configs that
the RF surrogate predicts will be feasible and non-crashing.
"""
from __future__ import annotations

import math
import random as stdlib_random
from typing import Any

import numpy as np
from scipy.stats import gaussian_kde

from tba.types import EvalResult, VariableDef
from tba.optimizer.search_space import SearchSpace
from tba.optimizer.surrogate import RFSurrogate


class FeasibleTPESampler:
    """KDE-based sampler that operates only within the feasible region.

    Splits feasible observations into good/bad by objective, fits marginal
    KDEs per variable for each group, and samples from l(x)/g(x).
    Uses the RF surrogate as a feasibility pre-filter to avoid wasting evals.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        surrogate: RFSurrogate | None,
        seed: int = 42,
        gamma: float = 0.3,          # top 30% = "good"
        n_candidates: int = 24,      # draw this many, pick best EI ratio
        bandwidth_factor: float = 1.0,
    ):
        self.space = search_space
        self.surrogate = surrogate
        self.rng = stdlib_random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.bw_factor = bandwidth_factor

        # Variable metadata
        self._var_names = list(search_space._all_names)
        self._var_defs = [search_space.variables[n] for n in self._var_names]

        # Fitted state
        self._good_data: dict[str, np.ndarray] = {}
        self._bad_data: dict[str, np.ndarray] = {}
        self._good_kdes: dict[str, gaussian_kde | None] = {}
        self._bad_kdes: dict[str, gaussian_kde | None] = {}
        self._fitted = False

    def fit(self, feasible_history: list[tuple[dict[str, Any], EvalResult]]) -> bool:
        """Fit KDEs on feasible history split into good/bad.

        Returns True if enough data to sample, False otherwise.
        """
        if len(feasible_history) < 4:
            self._fitted = False
            return False

        # Sort by objective descending
        sorted_hist = sorted(feasible_history, key=lambda cr: cr[1].objective_value, reverse=True)
        n_good = max(2, int(len(sorted_hist) * self.gamma))
        good = sorted_hist[:n_good]
        bad = sorted_hist[n_good:]
        if len(bad) < 2:
            bad = sorted_hist[n_good - 1:]  # ensure at least 2

        # Extract per-variable values for each group
        self._good_data = {}
        self._bad_data = {}
        self._good_kdes = {}
        self._bad_kdes = {}

        for name, vdef in zip(self._var_names, self._var_defs):
            g_vals = self._extract_values(good, name, vdef)
            b_vals = self._extract_values(bad, name, vdef)

            self._good_data[name] = g_vals
            self._bad_data[name] = b_vals

            self._good_kdes[name] = self._fit_kde(g_vals, vdef)
            self._bad_kdes[name] = self._fit_kde(b_vals, vdef)

        self._fitted = True
        return True

    def sample(self) -> dict[str, Any]:
        """Sample a config from the good distribution, filtered by surrogate feasibility.

        Generates n_candidates samples from l(x), scores by l(x)/g(x),
        optionally filters by RF feasibility, returns the best.
        """
        if not self._fitted:
            return self.space.sample_random(self.rng)

        candidates = []
        for _ in range(self.n_candidates):
            config = self._sample_from_good()
            score = self._tpe_score(config)
            candidates.append((config, score))

        # If surrogate available, boost score for predicted-feasible configs
        if self.surrogate is not None and self.surrogate.is_ready:
            scored = []
            for config, tpe_score in candidates:
                _, p_feas = self.surrogate.predict(config)
                # Combined score: TPE ratio * feasibility probability
                combined = tpe_score * (0.3 + 0.7 * p_feas)
                scored.append((config, combined))
            candidates = scored

        # Pick highest scoring
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    # ------------------------------------------------------------------
    # Internal: extraction, KDE fitting, sampling
    # ------------------------------------------------------------------

    def _extract_values(
        self,
        history: list[tuple[dict[str, Any], EvalResult]],
        name: str,
        vdef: VariableDef,
    ) -> np.ndarray:
        """Extract numeric values for a variable from history.
        Categoricals → ordinal index. Missing (inactive) → NaN filtered out.
        """
        vals = []
        for config, _ in history:
            v = config.get(name)
            if v is None:
                continue
            if vdef.var_type == "categorical":
                if v in vdef.choices:
                    vals.append(float(vdef.choices.index(v)))
            else:
                vals.append(float(v))
        return np.array(vals) if vals else np.array([])

    def _fit_kde(self, vals: np.ndarray, vdef: VariableDef) -> gaussian_kde | None:
        """Fit a 1D KDE on values. Returns None if not enough data."""
        if len(vals) < 2:
            return None
        # Add tiny jitter to avoid singular matrix with identical values
        if np.std(vals) < 1e-10:
            vals = vals + self.np_rng.normal(0, 1e-6, size=len(vals))
        try:
            bw = "scott" if len(vals) >= 5 else "silverman"
            kde = gaussian_kde(vals, bw_method=bw)
            kde.set_bandwidth(kde.factor * self.bw_factor)
            return kde
        except Exception:
            return None

    def _sample_from_good(self) -> dict[str, Any]:
        """Sample one config from the good distribution, respecting hierarchy."""
        config: dict[str, Any] = {}

        # Sample variables in order, respecting conditions
        for name, vdef in zip(self._var_names, self._var_defs):
            if vdef.condition is not None:
                if not self.space._eval_condition(vdef.condition, config):
                    continue

            kde = self._good_kdes.get(name)
            if kde is not None and len(self._good_data.get(name, [])) >= 2:
                val = self._sample_var_from_kde(kde, vdef)
            else:
                # Fallback: uniform sample
                val = self._sample_var_uniform(vdef)

            config[name] = val

        return config

    def _sample_var_from_kde(self, kde: gaussian_kde, vdef: VariableDef) -> Any:
        """Sample a single value from a KDE, clipped to variable bounds."""
        for _ in range(20):  # retry if out of bounds
            raw = float(kde.resample(1, seed=self.np_rng.randint(0, 2**31))[0, 0])

            if vdef.var_type == "categorical":
                idx = max(0, min(len(vdef.choices) - 1, int(round(raw))))
                return vdef.choices[idx]
            elif vdef.var_type == "integer":
                val = int(round(raw))
                if vdef.low <= val <= vdef.high:
                    return val
            else:  # continuous
                if vdef.low <= raw <= vdef.high:
                    return raw

        # Fallback after retries
        return self._sample_var_uniform(vdef)

    def _sample_var_uniform(self, vdef: VariableDef) -> Any:
        if vdef.var_type == "categorical":
            return self.rng.choice(vdef.choices)
        elif vdef.var_type == "integer":
            return self.rng.randint(int(vdef.low), int(vdef.high))
        else:
            return self.rng.uniform(vdef.low, vdef.high)

    def _tpe_score(self, config: dict[str, Any]) -> float:
        """Compute l(x) / g(x) — the TPE acquisition score.
        Higher means x is more likely under 'good' than 'bad'.
        """
        log_l = 0.0  # log P(x | good)
        log_g = 0.0  # log P(x | bad)

        for name, vdef in zip(self._var_names, self._var_defs):
            v = config.get(name)
            if v is None:
                continue

            if vdef.var_type == "categorical":
                num_val = float(vdef.choices.index(v)) if v in vdef.choices else 0.0
            else:
                num_val = float(v)

            l_kde = self._good_kdes.get(name)
            g_kde = self._bad_kdes.get(name)

            # Evaluate densities with floor to avoid log(0)
            l_density = float(l_kde.evaluate([num_val])[0]) if l_kde is not None else 1e-10
            g_density = float(g_kde.evaluate([num_val])[0]) if g_kde is not None else 1e-10

            log_l += math.log(max(l_density, 1e-10))
            log_g += math.log(max(g_density, 1e-10))

        # l(x) / g(x) in log space
        return log_l - log_g
