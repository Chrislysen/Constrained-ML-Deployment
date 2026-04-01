"""Subspace blacklisting for SA phases.

Tracks consecutive failures per categorical value and temporarily removes
values from the mutation pool after repeated crashes/infeasible results.
Prevents SA from burning budget on known-dead subspaces (e.g. int8_dynamic
on slow GPUs).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


class SubspaceTracker:
    """Tracks consecutive failures per categorical value and manages blacklisting.

    Args:
        categorical_names: Which variable names to track (e.g. ["model_name", "backend", "quantization"]).
        max_consecutive_failures: Blacklist a value after this many consecutive bad results.
        cooldown_trials: Un-blacklist after this many trials (in case conditions changed).
    """

    def __init__(
        self,
        categorical_names: list[str] | None = None,
        max_consecutive_failures: int = 3,
        cooldown_trials: int = 8,
    ):
        self.tracked_names = categorical_names or ["model_name", "backend", "quantization"]
        self.max_failures = max_consecutive_failures
        self.cooldown = cooldown_trials
        # {("quantization", "int8_dynamic"): 3, ...}
        self.consecutive_failures: dict[tuple[str, Any], int] = defaultdict(int)
        # {("quantization", "int8_dynamic"): trial_number_when_blacklisted, ...}
        self.blacklisted_at: dict[tuple[str, Any], int] = {}

    def record_result(
        self, config: dict[str, Any], status: str, trial_number: int,
    ) -> None:
        """Call after each trial. status is 'ok', 'crash', or 'infeasible'."""
        for var_name in self.tracked_names:
            if var_name not in config:
                continue
            key = (var_name, config[var_name])
            if status in ("crash", "infeasible"):
                self.consecutive_failures[key] += 1
                if self.consecutive_failures[key] >= self.max_failures:
                    self.blacklisted_at[key] = trial_number
            else:
                # Success resets counter and un-blacklists
                self.consecutive_failures[key] = 0
                self.blacklisted_at.pop(key, None)

    def get_allowed_values(
        self, var_name: str, all_values: list[Any], current_trial: int,
    ) -> list[Any]:
        """Returns the subset of values NOT currently blacklisted."""
        allowed = []
        for val in all_values:
            key = (var_name, val)
            if key in self.blacklisted_at:
                if current_trial - self.blacklisted_at[key] >= self.cooldown:
                    # Cooldown expired — un-blacklist
                    del self.blacklisted_at[key]
                    self.consecutive_failures[key] = 0
                    allowed.append(val)
                # else: still blacklisted, skip
            else:
                allowed.append(val)

        # Safety valve: never blacklist ALL values
        if not allowed:
            return list(all_values)
        return allowed

    @property
    def blacklisted(self) -> list[tuple[str, Any]]:
        """Return currently blacklisted (var_name, value) pairs."""
        return list(self.blacklisted_at.keys())
