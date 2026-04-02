"""Subspace blacklisting for SA phases.

Tracks consecutive failures per categorical value and temporarily removes
values from the mutation pool after repeated crashes/infeasible results.
Prevents SA from burning budget on known-dead subspaces (e.g. int8_dynamic
on slow GPUs).

v3 adds combination-level blacklisting: tracks 2-way categorical pairs
(e.g. int8_dynamic + onnxruntime) so fine-grained patterns are caught
without over-blacklisting individual values.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any


class SubspaceTracker:
    """Tracks consecutive failures per categorical value and manages blacklisting.

    Args:
        categorical_names: Which variable names to track (e.g. ["model_name", "backend", "quantization"]).
        max_consecutive_failures: Blacklist a value after this many consecutive bad results.
        cooldown_trials: Un-blacklist after this many trials (in case conditions changed).
        enable_combo_blacklisting: Track 2-way categorical combination failures (v3).
        combo_pairs: Which variable pairs to track. Defaults to all 2-way combos of categorical_names.
    """

    def __init__(
        self,
        categorical_names: list[str] | None = None,
        max_consecutive_failures: int = 3,
        cooldown_trials: int = 8,
        enable_combo_blacklisting: bool = True,
        combo_pairs: list[tuple[str, str]] | None = None,
    ):
        self.tracked_names = categorical_names or ["model_name", "backend", "quantization"]
        self.max_failures = max_consecutive_failures
        self.cooldown = cooldown_trials
        # {("quantization", "int8_dynamic"): 3, ...}
        self.consecutive_failures: dict[tuple[str, Any], int] = defaultdict(int)
        # {("quantization", "int8_dynamic"): trial_number_when_blacklisted, ...}
        self.blacklisted_at: dict[tuple[str, Any], int] = {}

        # v3: combination-level blacklisting
        self.enable_combo_blacklisting = enable_combo_blacklisting
        self.combo_pairs: list[tuple[str, str]] = (
            combo_pairs if combo_pairs is not None
            else list(combinations(self.tracked_names, 2))
        )
        # key: (("var1", "val1"), ("var2", "val2")) -> failure count
        self.combo_consecutive_failures: dict[tuple[tuple[str, Any], tuple[str, Any]], int] = defaultdict(int)
        # key -> trial_number when blacklisted
        self.combo_blacklisted_at: dict[tuple[tuple[str, Any], tuple[str, Any]], int] = {}

    def _combo_keys(self, config: dict[str, Any]) -> list[tuple[tuple[str, Any], tuple[str, Any]]]:
        """Extract all tracked 2-way combination keys from a config."""
        keys = []
        for var_a, var_b in self.combo_pairs:
            if var_a in config and var_b in config:
                keys.append(((var_a, config[var_a]), (var_b, config[var_b])))
        return keys

    def record_result(
        self, config: dict[str, Any], status: str, trial_number: int,
    ) -> None:
        """Call after each trial. status is 'ok', 'crash', or 'infeasible'."""
        # --- per-value tracking (v2) ---
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

        # --- per-combination tracking (v3) ---
        if self.enable_combo_blacklisting:
            for combo_key in self._combo_keys(config):
                if status in ("crash", "infeasible"):
                    self.combo_consecutive_failures[combo_key] += 1
                    if self.combo_consecutive_failures[combo_key] >= self.max_failures:
                        self.combo_blacklisted_at[combo_key] = trial_number
                else:
                    self.combo_consecutive_failures[combo_key] = 0
                    self.combo_blacklisted_at.pop(combo_key, None)

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

    def is_combo_blacklisted(
        self, config: dict[str, Any], current_trial: int,
    ) -> bool:
        """Check if a config contains any blacklisted 2-way combination.

        Also handles cooldown expiry for combinations.
        """
        if not self.enable_combo_blacklisting:
            return False
        for combo_key in self._combo_keys(config):
            if combo_key in self.combo_blacklisted_at:
                if current_trial - self.combo_blacklisted_at[combo_key] >= self.cooldown:
                    # Cooldown expired
                    del self.combo_blacklisted_at[combo_key]
                    self.combo_consecutive_failures[combo_key] = 0
                else:
                    return True
        return False

    @property
    def blacklisted(self) -> list[tuple[str, Any]]:
        """Return currently blacklisted (var_name, value) pairs."""
        return list(self.blacklisted_at.keys())

    @property
    def combo_blacklisted(self) -> list[tuple[tuple[str, Any], tuple[str, Any]]]:
        """Return currently blacklisted 2-way combinations."""
        return list(self.combo_blacklisted_at.keys())
