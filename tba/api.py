"""High-level API for TBA deployment optimization.

Usage:
    from tba import optimize_deployment

    best_config, best_result = optimize_deployment(
        data_dir="path/to/imagenette2-320",
        constraints={"latency_p95_ms": 20, "memory_peak_mb": 512},
        budget=25,
    )
"""
from __future__ import annotations

import warnings
from typing import Any

warnings.filterwarnings("ignore", message=".*does not have constraint values.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from tba.types import EvalResult
from tba.optimizer.tba_tpe_hybrid import TBATPEHybrid
from tba.spaces.image_classification import image_classification_space
from tba.benchmarks.profiler import evaluate_config


def optimize_deployment(
    data_dir: str,
    constraints: dict[str, float] | None = None,
    budget: int = 25,
    objective: str = "maximize_accuracy",
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> tuple[dict[str, Any], EvalResult]:
    """Find the best deployment config under constraints in *budget* evaluations.

    Args:
        data_dir: Path to ImageNette (or ImageNet) validation data.
        constraints: Hard constraint caps, e.g. {"latency_p95_ms": 20, "memory_peak_mb": 512}.
            Defaults to edge deployment constraints.
        budget: Maximum number of expensive evaluations.
        objective: "maximize_accuracy" or "maximize_throughput".
        device: "cuda" or "cpu".
        seed: Random seed for reproducibility.
        verbose: Print progress per trial.

    Returns:
        (best_config, best_result) — the best feasible configuration found
        and its evaluation result, or (None, None) if no feasible config found.

    Example:
        >>> from tba import optimize_deployment
        >>> config, result = optimize_deployment("data/imagenette2-320", budget=25)
        >>> print(f"Best: {config['model_name']}, accuracy={result.objective_value:.3f}")
    """
    if constraints is None:
        constraints = {"latency_p95_ms": 20.0, "memory_peak_mb": 512.0}

    space = image_classification_space()
    optimizer = TBATPEHybrid(
        search_space=space,
        constraints=constraints,
        objective=objective,
        budget=budget,
        seed=seed,
    )

    for trial_id in range(budget):
        config = optimizer.ask()

        result = evaluate_config(
            config, data_dir, device=device, constraint_caps=constraints,
            has_feasible=(optimizer.best_feasible() is not None),
        )

        # Override objective for throughput
        if objective == "maximize_throughput" and not result.crashed:
            result.objective_value = result.constraints.get("throughput_qps", 0.0)

        optimizer.tell(config, result)

        if verbose:
            status = "CRASH" if result.crashed else ("INFEAS" if not result.feasible else "OK")
            m = config.get("model_name", "?")
            b = config.get("backend", "?")
            q = config.get("quantization", "?")
            bs = config.get("batch_size", "?")
            bf = optimizer.best_feasible()
            best_str = f"{bf[1].objective_value:.4f}" if bf else "N/A"
            print(f"  [{trial_id+1:>2}/{budget}] {m}/{b}/{q}/bs={bs}  "
                  f"{status}  best={best_str}  [{result.eval_time_s:.1f}s]")

    bf = optimizer.best_feasible()
    if bf is None:
        if verbose:
            print("No feasible configuration found.")
        return None, None

    best_config, best_result = bf
    if verbose:
        m = best_config["model_name"]
        b = best_config["backend"]
        q = best_config["quantization"]
        bs = best_config["batch_size"]
        print(f"\nBest config: {m}/{b}/{q}/bs={bs}")
        print(f"  Objective: {best_result.objective_value:.4f}")
        for k, v in best_result.constraints.items():
            print(f"  {k}: {v:.2f}")

    return best_config, best_result
