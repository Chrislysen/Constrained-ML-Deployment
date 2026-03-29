"""Synthetic benchmarks with crash zones and constraints.

These are fast to evaluate and let us validate TBA before touching real models.
"""
from __future__ import annotations

import math

from tba.types import EvalResult, VariableDef
from tba.optimizer.search_space import SearchSpace


# ------------------------------------------------------------------
# Standard Branin function
# ------------------------------------------------------------------

def _branin(x1: float, x2: float) -> float:
    """Standard Branin-Hoo. Global min ≈ 0.397887."""
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


# ------------------------------------------------------------------
# Problem 1: Crashy Constrained Branin
# ------------------------------------------------------------------

def crashy_branin_space() -> SearchSpace:
    """Search space for the crashy Branin benchmark.

    - 1 categorical (mode: A/B/C — shifts landscape)
    - 2 continuous (x1, x2)
    - 1 integer (resolution: 1-10)
    - 2 inequality constraints
    - ~35% of random configs crash or are infeasible
    """
    return SearchSpace([
        VariableDef(name="mode", var_type="categorical", choices=["A", "B", "C"]),
        VariableDef(name="x1", var_type="continuous", low=0.0, high=15.0),
        VariableDef(name="x2", var_type="continuous", low=0.0, high=15.0),
        VariableDef(name="resolution", var_type="integer", low=1, high=10),
    ])


def evaluate_crashy_branin(config: dict) -> EvalResult:
    """Evaluate the crashy Branin benchmark.

    Crash zones (~15% of space):
      - mode C and x1 > 8
      - resolution > 7 and x2 > 12

    Constraint violations (~20% of remaining):
      - g1: x1 + x2 <= 14
      - g2: -x1 + 2*x2 <= 8

    Combined: ~35% of random configs are invalid.
    """
    x1 = config["x1"]
    x2 = config["x2"]
    mode = config["mode"]
    resolution = config["resolution"]

    # Crash zones — simulate OOM / invalid param combos
    if mode == "C" and x1 > 8:
        return EvalResult(
            crashed=True,
            error_msg="Simulated crash: mode C with x1 > 8",
        )
    if resolution > 7 and x2 > 12:
        return EvalResult(
            crashed=True,
            error_msg="Simulated crash: high resolution with x2 > 12",
        )

    # Compute objective (shifted Branin per mode, lower is better → negate to maximize)
    shift = {"A": 0.0, "B": 2.0, "C": -2.0}[mode]
    raw_obj = _branin(x1 + shift, x2) + resolution * 0.1
    objective_value = -raw_obj  # maximize negative branin

    # Constraints: g1 <= 0, g2 <= 0
    # Tighter constraints — feasible region ~25-30% of non-crash space
    g1 = x1 + x2 - 16.0
    g2 = -x1 + 2.0 * x2 - 10.0
    feasible = g1 <= 0 and g2 <= 0

    return EvalResult(
        objective_value=objective_value,
        constraints={"g1": g1, "g2": g2},
        feasible=feasible,
        crashed=False,
    )


# ------------------------------------------------------------------
# Problem 2: Hierarchical Rosenbrock
# ------------------------------------------------------------------

def hierarchical_rosenbrock_space() -> SearchSpace:
    """Search space for the hierarchical Rosenbrock benchmark.

    A categorical 'mode' variable activates different subsets of continuous vars.
    Tests hierarchical handling.
    """
    return SearchSpace([
        VariableDef(name="mode", var_type="categorical", choices=["simple", "extended", "full"]),
        # Always active
        VariableDef(name="x1", var_type="continuous", low=-5.0, high=5.0),
        VariableDef(name="x2", var_type="continuous", low=-5.0, high=5.0),
        # Only in extended or full mode
        VariableDef(
            name="x3", var_type="continuous", low=-5.0, high=5.0,
            condition="mode in ['extended', 'full']",
        ),
        VariableDef(
            name="x4", var_type="continuous", low=-5.0, high=5.0,
            condition="mode in ['extended', 'full']",
        ),
        # Only in full mode
        VariableDef(
            name="x5", var_type="continuous", low=-5.0, high=5.0,
            condition="mode == 'full'",
        ),
        # Integer param — number of iterations (affects noise)
        VariableDef(name="n_iters", var_type="integer", low=1, high=20),
    ])


def evaluate_hierarchical_rosenbrock(config: dict) -> EvalResult:
    """Evaluate the hierarchical Rosenbrock.

    Crash zone: full mode with x1 > 3 and x5 > 3.
    Constraint: sum of all active x_i must be <= 5.
    """
    mode = config["mode"]
    x1, x2 = config["x1"], config["x2"]
    n_iters = config["n_iters"]

    xs = [x1, x2]
    if mode in ("extended", "full"):
        xs.extend([config["x3"], config["x4"]])
    if mode == "full":
        xs.append(config["x5"])

    # Crash zone
    if mode == "full" and x1 > 3 and config.get("x5", 0) > 3:
        return EvalResult(crashed=True, error_msg="Simulated crash: full mode extreme values")

    # Rosenbrock (sum of pairwise terms)
    obj = 0.0
    for i in range(len(xs) - 1):
        obj += 100 * (xs[i + 1] - xs[i] ** 2) ** 2 + (1 - xs[i]) ** 2

    # Add noise proportional to n_iters (fewer iters = more noise)
    noise_scale = 1.0 / n_iters
    obj += noise_scale * (hash(tuple(xs)) % 100) * 0.01

    objective_value = -obj  # maximize negative rosenbrock

    # Constraint: sum of active vars <= 5
    var_sum = sum(xs) - 5.0
    feasible = var_sum <= 0

    return EvalResult(
        objective_value=objective_value,
        constraints={"var_sum": var_sum},
        feasible=feasible,
        crashed=False,
    )


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

BENCHMARKS = {
    "crashy_branin": {
        "space_fn": crashy_branin_space,
        "eval_fn": evaluate_crashy_branin,
        "constraints": {"g1": 0.0, "g2": 0.0},  # both <= 0
        "objective": "maximize",
    },
    "hierarchical_rosenbrock": {
        "space_fn": hierarchical_rosenbrock_space,
        "eval_fn": evaluate_hierarchical_rosenbrock,
        "constraints": {"var_sum": 0.0},
        "objective": "maximize",
    },
}
