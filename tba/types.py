"""Core data types for TBA — dataclasses for type safety."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result of evaluating a single configuration."""

    objective_value: float = float("-inf")
    constraints: dict[str, float] = field(default_factory=dict)
    feasible: bool = False
    crashed: bool = False
    eval_time_s: float = 0.0
    error_msg: str | None = None


@dataclass
class TrialRecord:
    """One complete trial: config + result + metadata."""

    trial_id: int
    config: dict[str, Any]
    result: EvalResult
    best_feasible_so_far: float | None
    cumulative_crashes: int
    cumulative_infeasible: int
    wall_clock_s: float


@dataclass
class VariableDef:
    """Definition of a single variable in the search space."""

    name: str
    var_type: str  # "categorical", "integer", "continuous"
    choices: list[Any] | None = None  # for categorical
    low: float | None = None  # for integer/continuous
    high: float | None = None  # for integer/continuous
    log_scale: bool = False  # search in log space
    condition: str | None = None  # python expression, e.g. "backend == 'tensorrt'"

    def __post_init__(self):
        if self.var_type == "categorical" and not self.choices:
            raise ValueError(f"Categorical variable '{self.name}' needs choices")
        if self.var_type in ("integer", "continuous") and (self.low is None or self.high is None):
            raise ValueError(f"Numeric variable '{self.name}' needs low and high")
