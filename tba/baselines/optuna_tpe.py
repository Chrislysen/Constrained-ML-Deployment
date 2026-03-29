"""Optuna TPE baseline — Tree-structured Parzen Estimator.

Uses Optuna's native conditional sampling so TPE gets full advantage of
its tree-structured hierarchy handling. This is a STRONG baseline.
"""
from __future__ import annotations

import math
from typing import Any

import optuna

from tba.types import EvalResult, VariableDef
from tba.optimizer.base import BaseOptimizer
from tba.optimizer.search_space import SearchSpace

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTPEOptimizer(BaseOptimizer):
    """TPE baseline using Optuna with proper conditional sampling.

    Constraints are handled via Optuna's constraint interface: infeasible
    trials are deprioritized but still inform the model.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str,
        budget: int,
        seed: int = 42,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)
        self._constraints = constraints

        # Build Optuna study with constraint support
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=max(3, budget // 5),
            constraints_func=self._constraints_func,
        )
        self._study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        # Queue of configs: ask() enqueues a trial, tell() completes it
        self._pending_trial: optuna.trial.Trial | None = None
        self._pending_config: dict[str, Any] | None = None

    def ask(self) -> dict[str, Any]:
        trial = self._study.ask()
        config = self._sample_from_trial(trial)
        self._pending_trial = trial
        self._pending_config = config
        return config

    def tell(self, config: dict[str, Any], result: EvalResult) -> None:
        super().tell(config, result)

        if self._pending_trial is None:
            return

        # Store constraint violations as user attributes for the constraint func
        total_violation = 0.0
        for name, cap in self._constraints.items():
            measured = result.constraints.get(name, 0.0)
            violation = max(0.0, measured - cap)
            total_violation += violation
        self._pending_trial.set_user_attr("constraint_violation", total_violation)
        self._pending_trial.set_user_attr("crashed", result.crashed)

        if result.crashed:
            # Report as failed with very bad value
            self._study.tell(self._pending_trial, float("-inf"))
        else:
            self._study.tell(self._pending_trial, result.objective_value)

        self._pending_trial = None
        self._pending_config = None

    def _sample_from_trial(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Sample a config from an Optuna trial using proper conditional suggest_* calls.

        This is the key: we translate our SearchSpace hierarchy into Optuna's
        native conditional sampling so TPE can exploit the tree structure.
        """
        config: dict[str, Any] = {}
        space = self.search_space

        # Sample variables in order, respecting conditions
        for name in space._all_names:
            v = space.variables[name]

            # Check if this variable should be active given already-sampled values
            if v.condition is not None:
                if not space._eval_condition(v.condition, config):
                    continue

            # Sample using Optuna's native suggest_* methods
            if v.var_type == "categorical":
                config[name] = trial.suggest_categorical(name, v.choices)
            elif v.var_type == "integer":
                if v.log_scale:
                    config[name] = trial.suggest_int(name, int(v.low), int(v.high), log=True)
                else:
                    config[name] = trial.suggest_int(name, int(v.low), int(v.high))
            elif v.var_type == "continuous":
                if v.log_scale:
                    config[name] = trial.suggest_float(name, v.low, v.high, log=True)
                else:
                    config[name] = trial.suggest_float(name, v.low, v.high)

        return config

    @staticmethod
    def _constraints_func(frozen_trial: optuna.trial.FrozenTrial) -> list[float]:
        """Optuna constraint function: return list of constraint values.
        Values <= 0 mean satisfied. Positive means violated."""
        violation = frozen_trial.user_attrs.get("constraint_violation", 0.0)
        crashed = frozen_trial.user_attrs.get("crashed", False)
        if crashed:
            return [100.0]  # Heavily penalize crashed configs
        return [violation]
