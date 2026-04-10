"""Adaptive Confidence-Gated Feasibility (ACGF) evaluator.

Instead of making a binary feasible/infeasible decision from a single
evaluation, ACGF allocates re-evaluation budget adaptively based on how
close the result is to the constraint boundary.

Configs clearly inside or outside the boundary get a single evaluation.
Borderline configs (within band_width_ms of the threshold) get up to
max_reeval additional evaluations, with the count proportional to
closeness. The final feasibility decision uses the empirical probability
across all evaluations.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tba.types import EvalResult
from tba.benchmarks.profiler import evaluate_config


@dataclass
class ACGFParams:
    """Parameters controlling the ACGF algorithm."""

    band_width_ms: float = 5.0
    max_reeval: int = 5
    confidence_threshold: float = 0.7
    latency_constraint_key: str = "latency_p95_ms"
    # If set, use fixed K instead of adaptive (for ablation A)
    fixed_k: int | None = None


@dataclass
class ACGFResult:
    """Result of an ACGF-gated evaluation."""

    final_result: EvalResult
    evaluations_used: int
    is_borderline: bool
    feasibility_probability: float | None  # None if not borderline
    was_rescued: bool  # initially infeasible, re-eval showed feasible
    was_caught: bool  # initially feasible, re-eval showed infeasible
    all_p95_latencies: list[float] = field(default_factory=list)
    gpu_temps: list[float] = field(default_factory=list)
    individual_results: list[EvalResult] = field(default_factory=list)


def get_gpu_temperature(gpu_id: int = 0) -> float | None:
    """Query GPU temperature in Celsius via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return float(out.strip())
    except Exception:
        return None


def compute_adaptive_k(
    distance_ms: float,
    band_width_ms: float,
    max_reeval: int,
) -> int:
    """Compute adaptive re-evaluation count based on distance to threshold.

    Configs right on the boundary (distance=0) get max_reeval.
    Configs at the edge of the band get 1.
    Configs outside the band get 0.
    """
    if band_width_ms <= 0 or max_reeval <= 0:
        return 0
    if abs(distance_ms) >= band_width_ms:
        return 0
    closeness = 1.0 - abs(distance_ms) / band_width_ms
    return max(1, math.ceil(max_reeval * closeness))


def evaluate_with_acgf(
    config: dict[str, Any],
    data_dir: str,
    constraint_caps: dict[str, float],
    device: str = "cuda",
    params: ACGFParams | None = None,
    budget_remaining: int | None = None,
    has_feasible: bool = False,
    gpu_id: int = 0,
) -> ACGFResult:
    """Evaluate a config with adaptive confidence-gated feasibility.

    Steps:
      1. Run standard evaluate_config() once
      2. If crashed → return immediately
      3. Compute distance from observed p95 to latency threshold
      4. If clearly feasible/infeasible (outside band) → return single result
      5. If borderline → compute K re-evaluations (adaptive or fixed)
      6. Feasibility = empirical probability >= confidence_threshold
    """
    if params is None:
        params = ACGFParams()

    threshold_ms = constraint_caps.get(params.latency_constraint_key)
    if threshold_ms is None:
        # No latency constraint — single evaluation, no ACGF logic
        result = evaluate_config(
            config, data_dir, device=device,
            constraint_caps=constraint_caps, has_feasible=has_feasible,
        )
        return ACGFResult(
            final_result=result, evaluations_used=1, is_borderline=False,
            feasibility_probability=None, was_rescued=False, was_caught=False,
            individual_results=[result],
        )

    # --- Step 1: Initial evaluation ---
    gpu_temps: list[float] = []
    temp = get_gpu_temperature(gpu_id)
    if temp is not None:
        gpu_temps.append(temp)

    result = evaluate_config(
        config, data_dir, device=device,
        constraint_caps=constraint_caps, has_feasible=has_feasible,
    )
    all_results = [result]

    # --- Step 2: Crashed configs are not borderline ---
    if result.crashed:
        return ACGFResult(
            final_result=result, evaluations_used=1, is_borderline=False,
            feasibility_probability=0.0, was_rescued=False, was_caught=False,
            all_p95_latencies=[], gpu_temps=gpu_temps,
            individual_results=all_results,
        )

    # --- Step 3: Distance to threshold ---
    observed_p95 = result.constraints.get(params.latency_constraint_key, 0.0)
    distance = threshold_ms - observed_p95  # positive = feasible side
    all_p95 = [observed_p95]
    initial_feasible = result.feasible

    # --- Step 4: Check if borderline ---
    if abs(distance) > params.band_width_ms:
        return ACGFResult(
            final_result=result, evaluations_used=1, is_borderline=False,
            feasibility_probability=1.0 if result.feasible else 0.0,
            was_rescued=False, was_caught=False,
            all_p95_latencies=all_p95, gpu_temps=gpu_temps,
            individual_results=all_results,
        )

    # --- Step 5: Borderline — compute K and re-evaluate ---
    if params.fixed_k is not None:
        K = params.fixed_k
    else:
        K = compute_adaptive_k(distance, params.band_width_ms, params.max_reeval)

    # Cap by remaining budget (we already used 1)
    if budget_remaining is not None:
        K = min(K, budget_remaining - 1)
    K = max(0, K)

    for _ in range(K):
        temp = get_gpu_temperature(gpu_id)
        if temp is not None:
            gpu_temps.append(temp)

        r = evaluate_config(
            config, data_dir, device=device,
            constraint_caps=constraint_caps, has_feasible=has_feasible,
        )
        all_results.append(r)
        if not r.crashed:
            all_p95.append(r.constraints.get(params.latency_constraint_key, 0.0))

    # --- Step 6: Feasibility probability ---
    feasible_count = sum(1 for r in all_results if r.feasible and not r.crashed)
    non_crash_count = sum(1 for r in all_results if not r.crashed)
    feas_prob = feasible_count / non_crash_count if non_crash_count > 0 else 0.0

    final_feasible = feas_prob >= params.confidence_threshold

    # Build merged result: median latency, best accuracy among non-crashed
    best_accuracy = max(
        (r.objective_value for r in all_results if not r.crashed),
        default=result.objective_value,
    )
    median_p95 = float(np.median(all_p95)) if all_p95 else observed_p95

    final_result = EvalResult(
        objective_value=best_accuracy,
        constraints=result.constraints.copy(),
        feasible=final_feasible,
        crashed=False,
        eval_time_s=sum(r.eval_time_s for r in all_results),
        error_msg=None,
        early_stopped=False,
    )
    final_result.constraints[params.latency_constraint_key] = median_p95

    return ACGFResult(
        final_result=final_result,
        evaluations_used=1 + K,
        is_borderline=True,
        feasibility_probability=feas_prob,
        was_rescued=not initial_feasible and final_feasible,
        was_caught=initial_feasible and not final_feasible,
        all_p95_latencies=all_p95,
        gpu_temps=gpu_temps,
        individual_results=all_results,
    )
