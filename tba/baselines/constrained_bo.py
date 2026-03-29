"""Constrained Bayesian Optimization baseline using scikit-learn GPs.

Uses a GP surrogate with constrained Expected Improvement.
Categoricals are ordinal-encoded, hierarchy is handled by fixing inactive
dimensions to the midpoint.

This is technically the strongest baseline for continuous spaces but
struggles with mixed hierarchical structure — that's part of the argument.
"""
from __future__ import annotations

import math
import random as stdlib_random
import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

from tba.types import EvalResult, VariableDef
from tba.optimizer.base import BaseOptimizer
from tba.optimizer.search_space import SearchSpace


class ConstrainedBOOptimizer(BaseOptimizer):
    """Constrained Bayesian optimization with sklearn GP surrogate.

    Uses GP + constrained EI (EI * P(feasible)) when enough data exists.
    Falls back to random sampling for the first n_initial trials.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str,
        budget: int,
        seed: int = 42,
        n_initial: int = 5,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)
        self.rng = stdlib_random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.n_initial = n_initial

        # Build encoding map
        self._var_names: list[str] = list(search_space._all_names)
        self._dim_info: list[tuple[str, VariableDef]] = [
            (name, search_space.variables[name]) for name in self._var_names
        ]
        self._n_dims = len(self._var_names)

        # Bounds for each encoded dimension
        self._lower = np.zeros(self._n_dims)
        self._upper = np.ones(self._n_dims)
        for i, (name, v) in enumerate(self._dim_info):
            if v.var_type == "categorical":
                self._lower[i] = 0.0
                self._upper[i] = float(len(v.choices) - 1)
            elif v.var_type in ("integer", "continuous"):
                self._lower[i] = float(v.low)
                self._upper[i] = float(v.high)

        # Normalize to [0, 1]
        self._range = self._upper - self._lower
        self._range[self._range < 1e-8] = 1.0

        # Observation storage
        self._X: list[np.ndarray] = []
        self._Y_obj: list[float] = []
        self._Y_violation: list[float] = []

    def ask(self) -> dict[str, Any]:
        if self.trial_count < self.n_initial:
            return self.search_space.sample_random(self.rng)

        try:
            x_next = self._bo_suggest()
            return self._decode(x_next)
        except Exception:
            return self.search_space.sample_random(self.rng)

    def tell(self, config: dict[str, Any], result: EvalResult) -> None:
        super().tell(config, result)

        x = self._encode(config)
        self._X.append(x)

        if result.crashed:
            self._Y_obj.append(-1e4)
            self._Y_violation.append(100.0)
        else:
            self._Y_obj.append(result.objective_value)
            violation = sum(
                max(0.0, result.constraints.get(name, 0.0) - cap)
                for name, cap in self.constraints.items()
            )
            self._Y_violation.append(violation)

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _encode(self, config: dict[str, Any]) -> np.ndarray:
        x = np.zeros(self._n_dims)
        for i, (name, v) in enumerate(self._dim_info):
            val = config.get(name)
            if val is None:
                x[i] = (self._lower[i] + self._upper[i]) / 2
            elif v.var_type == "categorical":
                x[i] = float(v.choices.index(val))
            else:
                x[i] = float(val)
        # Normalize to [0, 1]
        return (x - self._lower) / self._range

    def _decode(self, x_norm: np.ndarray) -> dict[str, Any]:
        x = x_norm * self._range + self._lower
        space = self.search_space
        raw: dict[str, Any] = {}
        for i, (name, v) in enumerate(self._dim_info):
            val = x[i]
            if v.var_type == "categorical":
                idx = max(0, min(len(v.choices) - 1, int(round(val))))
                raw[name] = v.choices[idx]
            elif v.var_type == "integer":
                raw[name] = max(int(v.low), min(int(v.high), int(round(val))))
            else:
                raw[name] = max(v.low, min(v.high, val))

        # Only include active variables
        config: dict[str, Any] = {}
        for name, v in self._dim_info:
            if v.condition is None or space._eval_condition(v.condition, raw):
                config[name] = raw[name]
        return config

    # ------------------------------------------------------------------
    # BO suggest
    # ------------------------------------------------------------------

    def _bo_suggest(self) -> np.ndarray:
        X = np.array(self._X)
        Y_obj = np.array(self._Y_obj)
        Y_viol = np.array(self._Y_violation)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Fit objective GP
            kernel_obj = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(self._n_dims))
            gp_obj = GaussianProcessRegressor(
                kernel=kernel_obj, alpha=1e-4, n_restarts_optimizer=3,
                random_state=self.np_rng.randint(0, 2**31),
            )
            gp_obj.fit(X, Y_obj)

            # Fit constraint GP
            kernel_con = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(self._n_dims))
            gp_con = GaussianProcessRegressor(
                kernel=kernel_con, alpha=1e-4, n_restarts_optimizer=3,
                random_state=self.np_rng.randint(0, 2**31),
            )
            gp_con.fit(X, Y_viol)

        # Best feasible objective so far
        feasible_mask = Y_viol <= 0
        if feasible_mask.any():
            best_f = Y_obj[feasible_mask].max()
        else:
            best_f = Y_obj.max()

        # Optimize constrained EI via multi-start L-BFGS-B
        def neg_constrained_ei(x_flat):
            x = x_flat.reshape(1, -1)
            mu_obj, sigma_obj = gp_obj.predict(x, return_std=True)
            mu_con, sigma_con = gp_con.predict(x, return_std=True)

            sigma_obj = max(sigma_obj[0], 1e-8)
            sigma_con = max(sigma_con[0], 1e-8)

            # EI
            z = (mu_obj[0] - best_f) / sigma_obj
            ei = (mu_obj[0] - best_f) * norm.cdf(z) + sigma_obj * norm.pdf(z)

            # P(feasible) = P(violation <= 0)
            p_feas = norm.cdf(-mu_con[0] / sigma_con)

            return -(ei * p_feas)

        bounds_01 = [(0.0, 1.0)] * self._n_dims
        best_x = None
        best_val = float("inf")

        # Multi-start optimization
        n_restarts = 10
        x_starts = self.np_rng.rand(n_restarts, self._n_dims)

        for x0 in x_starts:
            try:
                res = scipy_minimize(
                    neg_constrained_ei, x0, method="L-BFGS-B",
                    bounds=bounds_01, options={"maxiter": 50},
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                continue

        if best_x is None:
            # Fallback: random point
            best_x = self.np_rng.rand(self._n_dims)

        return best_x
