"""Optimizer variants for the ACGF comparison experiment.

Five strategies sharing the same evaluation budget:
  1. Random Search — uniform random sampling, binary feasibility
  2. Optuna TPE — model-based optimization, binary feasibility
  3. TBA Hybrid — existing TBA->TPE, binary feasibility
  4. TBA + Fixed Margin — TBA with tightened constraint
  5. TBA + ACGF — TBA with adaptive re-evaluation

All variants use the same ask/tell interface and the same evaluate_config()
from the existing profiler. ACGF re-evaluations count against budget.
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from tba.types import EvalResult
from tba.optimizer.base import BaseOptimizer
from tba.optimizer.search_space import SearchSpace
from tba.optimizer.tba_tpe_hybrid import TBATPEHybrid
from tba.spaces.image_classification import image_classification_space
from tba.benchmarks.profiler import evaluate_config

from experiments.acgf.acgf_evaluator import (
    ACGFParams,
    ACGFResult,
    evaluate_with_acgf,
    get_gpu_temperature,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Lightweight optimizers ────────────────────────────────────────────


class RandomOptimizer(BaseOptimizer):
    """Uniform random sampling — the simplest baseline."""

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str,
        budget: int,
        seed: int = 42,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)
        self._rng = random.Random(seed)

    def ask(self) -> dict[str, Any]:
        return self.search_space.sample_random(self._rng)


class TPEOptimizer(BaseOptimizer):
    """Optuna TPE without any TBA exploration phase."""

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str,
        budget: int,
        seed: int = 42,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)
        self._distributions = self._build_distributions()
        sampler = optuna.samplers.TPESampler(
            seed=seed, constraints_func=self._constraints_func,
        )
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._pending_trial: optuna.trial.Trial | None = None

    def _build_distributions(self) -> dict[str, Any]:
        dists: dict[str, Any] = {}
        for name, v in self.search_space.variables.items():
            if v.var_type == "categorical":
                dists[name] = CategoricalDistribution(v.choices)
            elif v.var_type == "integer":
                dists[name] = IntDistribution(int(v.low), int(v.high), log=v.log_scale)
            elif v.var_type == "continuous":
                dists[name] = FloatDistribution(v.low, v.high, log=v.log_scale)
        return dists

    def ask(self) -> dict[str, Any]:
        trial = self._study.ask()
        config: dict[str, Any] = {}
        for name in self.search_space._all_names:
            v = self.search_space.variables[name]
            if v.condition is not None:
                if not self.search_space._eval_condition(v.condition, config):
                    continue
            if v.var_type == "categorical":
                config[name] = trial.suggest_categorical(name, v.choices)
            elif v.var_type == "integer":
                config[name] = trial.suggest_int(
                    name, int(v.low), int(v.high), log=v.log_scale,
                )
            elif v.var_type == "continuous":
                config[name] = trial.suggest_float(
                    name, v.low, v.high, log=v.log_scale,
                )
        self._pending_trial = trial
        return config

    def tell(self, config: dict[str, Any], result: EvalResult) -> None:
        super().tell(config, result)
        if self._pending_trial is None:
            return
        violation = 0.0
        if result.crashed:
            violation = 100.0
        else:
            for name, cap in self.constraints.items():
                measured = result.constraints.get(name, 0.0)
                violation += max(0.0, measured - cap)
        self._pending_trial.set_user_attr("constraint_violation", violation)
        self._pending_trial.set_user_attr("crashed", result.crashed)
        value = float("-inf") if result.crashed else result.objective_value
        self._study.tell(self._pending_trial, value)
        self._pending_trial = None

    @staticmethod
    def _constraints_func(
        frozen_trial: optuna.trial.FrozenTrial,
    ) -> list[float]:
        violation = frozen_trial.user_attrs.get("constraint_violation", 0.0)
        crashed = frozen_trial.user_attrs.get("crashed", False)
        return [100.0 if crashed else violation]


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class TrialOutcome:
    """Per-trial record for one config evaluation (or ACGF sequence)."""

    trial_id: int
    config: dict[str, Any]
    evaluations_used: int
    feasibility_probability: float | None
    is_borderline: bool
    was_rescued: bool
    was_caught: bool
    result_feasible: bool
    result_crashed: bool
    result_objective: float
    result_latency_p95_ms: float
    result_memory_peak_mb: float
    result_eval_time_s: float
    all_p95_latencies: list[float]
    gpu_temps: list[float]
    wall_clock_s: float
    best_feasible_so_far: float | None


@dataclass
class RunResult:
    """Aggregate result for one variant + seed."""

    variant_name: str
    seed: int
    budget: int
    scenario: str
    outcomes: list[TrialOutcome]
    best_accuracy: float | None
    total_evaluations: int
    unique_configs: int
    n_feasible: int
    n_infeasible: int
    n_crashed: int
    n_borderline: int
    n_reeval: int
    n_rescued: int
    n_caught: int
    wall_clock_s: float


# ── Factory ───────────────────────────────────────────────────────────


def create_optimizer(
    variant_name: str,
    constraints: dict[str, float],
    budget: int,
    seed: int,
    margin_ms: float = 0.0,
) -> BaseOptimizer:
    """Create an optimizer instance for the given variant name.

    For margin variants, the optimizer sees a tightened constraint so its
    internal feasibility tracking is conservative.
    """
    space = image_classification_space()
    objective = "maximize_accuracy"

    effective = constraints.copy()
    if margin_ms > 0:
        for k in list(effective.keys()):
            if "latency" in k:
                effective[k] -= margin_ms

    if variant_name == "random":
        return RandomOptimizer(space, effective, objective, budget, seed)
    elif variant_name == "tpe":
        return TPEOptimizer(space, effective, objective, budget, seed)
    elif variant_name.startswith("tba"):
        return TBATPEHybrid(space, effective, objective, budget, seed)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")


# ── Experiment loop ───────────────────────────────────────────────────


def run_variant(
    variant_name: str,
    budget: int,
    data_dir: str,
    device: str,
    constraint_caps: dict[str, float],
    seed: int,
    scenario: str,
    acgf_params: ACGFParams | None = None,
    margin_ms: float = 0.0,
    gpu_id: int = 0,
    verbose: bool = True,
) -> RunResult:
    """Run a single optimizer variant for a given budget and seed.

    For margin variants the optimizer's internal constraint is tightened,
    but evaluate_config always uses the REAL caps for honest measurement.
    For ACGF, re-evaluations count against the budget.
    """
    optimizer = create_optimizer(
        variant_name, constraint_caps, budget, seed, margin_ms=margin_ms,
    )

    # For margin variants, the optimizer uses tightened caps internally,
    # but we evaluate against real caps and let the optimizer's own
    # feasibility check (against tightened caps) drive its decisions.
    eval_caps = constraint_caps.copy()
    if margin_ms > 0:
        for k in list(eval_caps.keys()):
            if "latency" in k:
                eval_caps[k] -= margin_ms

    outcomes: list[TrialOutcome] = []
    budget_used = 0
    trial_id = 0
    run_start = time.time()

    while budget_used < budget:
        config = optimizer.ask()
        trial_start = time.time()

        if acgf_params is not None:
            # ACGF uses real caps — it makes its own feasibility decision
            acgf_result = evaluate_with_acgf(
                config, data_dir, constraint_caps=constraint_caps,
                device=device, params=acgf_params,
                budget_remaining=budget - budget_used,
                has_feasible=optimizer.best_feasible() is not None,
                gpu_id=gpu_id,
            )
            budget_used += acgf_result.evaluations_used
            result = acgf_result.final_result
            evals_used = acgf_result.evaluations_used
            is_borderline = acgf_result.is_borderline
            feas_prob = acgf_result.feasibility_probability
            was_rescued = acgf_result.was_rescued
            was_caught = acgf_result.was_caught
            all_p95 = acgf_result.all_p95_latencies
            gpu_temps = acgf_result.gpu_temps
        else:
            result = evaluate_config(
                config, data_dir, device=device,
                constraint_caps=eval_caps,
                has_feasible=optimizer.best_feasible() is not None,
            )
            budget_used += 1
            evals_used = 1
            is_borderline = False
            feas_prob = None
            was_rescued = False
            was_caught = False
            p95 = result.constraints.get("latency_p95_ms", 0.0)
            all_p95 = [p95] if not result.crashed else []
            temp = get_gpu_temperature(gpu_id)
            gpu_temps = [temp] if temp is not None else []

        optimizer.tell(config, result)

        bf = optimizer.best_feasible()
        best_so_far = bf[1].objective_value if bf else None

        outcome = TrialOutcome(
            trial_id=trial_id,
            config=config,
            evaluations_used=evals_used,
            feasibility_probability=feas_prob,
            is_borderline=is_borderline,
            was_rescued=was_rescued,
            was_caught=was_caught,
            result_feasible=result.feasible,
            result_crashed=result.crashed,
            result_objective=result.objective_value,
            result_latency_p95_ms=result.constraints.get("latency_p95_ms", 0.0),
            result_memory_peak_mb=result.constraints.get("memory_peak_mb", 0.0),
            result_eval_time_s=result.eval_time_s,
            all_p95_latencies=all_p95,
            gpu_temps=gpu_temps,
            wall_clock_s=time.time() - trial_start,
            best_feasible_so_far=best_so_far,
        )
        outcomes.append(outcome)

        if verbose:
            status = "CRASH" if result.crashed else (
                "INFEAS" if not result.feasible else "OK"
            )
            tag = ""
            if is_borderline:
                tag += " [BORDER]"
            if was_rescued:
                tag += " RESCUED"
            if was_caught:
                tag += " CAUGHT"
            m = config.get("model_name", "?")
            b = config.get("backend", "?")[:6]
            q = config.get("quantization", "?")[:5]
            bs = config.get("batch_size", "?")
            best_str = f"{best_so_far:.4f}" if best_so_far is not None else "N/A"
            print(
                f"  [{budget_used:>2}/{budget}] {m}/{b}/{q}/bs={bs}  "
                f"{status}{tag}  best={best_str}  [{result.eval_time_s:.1f}s]",
                flush=True,
            )

        trial_id += 1

    bf = optimizer.best_feasible()
    return RunResult(
        variant_name=variant_name,
        seed=seed,
        budget=budget,
        scenario=scenario,
        outcomes=outcomes,
        best_accuracy=bf[1].objective_value if bf else None,
        total_evaluations=budget_used,
        unique_configs=len(outcomes),
        n_feasible=sum(1 for o in outcomes if o.result_feasible and not o.result_crashed),
        n_infeasible=sum(
            1 for o in outcomes if not o.result_feasible and not o.result_crashed
        ),
        n_crashed=sum(1 for o in outcomes if o.result_crashed),
        n_borderline=sum(1 for o in outcomes if o.is_borderline),
        n_reeval=sum(o.evaluations_used - 1 for o in outcomes),
        n_rescued=sum(1 for o in outcomes if o.was_rescued),
        n_caught=sum(1 for o in outcomes if o.was_caught),
        wall_clock_s=time.time() - run_start,
    )


# ── Serialization ─────────────────────────────────────────────────────


def run_result_to_dict(rr: RunResult) -> dict[str, Any]:
    """Serialize a RunResult to a JSON-compatible dict."""
    d = {
        "variant": rr.variant_name,
        "seed": rr.seed,
        "budget": rr.budget,
        "scenario": rr.scenario,
        "best_accuracy": rr.best_accuracy,
        "total_evaluations": rr.total_evaluations,
        "unique_configs": rr.unique_configs,
        "n_feasible": rr.n_feasible,
        "n_infeasible": rr.n_infeasible,
        "n_crashed": rr.n_crashed,
        "n_borderline": rr.n_borderline,
        "n_reeval": rr.n_reeval,
        "n_rescued": rr.n_rescued,
        "n_caught": rr.n_caught,
        "wall_clock_s": rr.wall_clock_s,
        "outcomes": [],
    }
    for o in rr.outcomes:
        od = {
            "trial_id": o.trial_id,
            "config": o.config,
            "evaluations_used": o.evaluations_used,
            "feasibility_probability": o.feasibility_probability,
            "is_borderline": o.is_borderline,
            "was_rescued": o.was_rescued,
            "was_caught": o.was_caught,
            "result_feasible": o.result_feasible,
            "result_crashed": o.result_crashed,
            "result_objective": o.result_objective,
            "result_latency_p95_ms": o.result_latency_p95_ms,
            "result_memory_peak_mb": o.result_memory_peak_mb,
            "result_eval_time_s": o.result_eval_time_s,
            "all_p95_latencies": o.all_p95_latencies,
            "gpu_temps": o.gpu_temps,
            "wall_clock_s": o.wall_clock_s,
            "best_feasible_so_far": o.best_feasible_so_far,
        }
        d["outcomes"].append(od)
    return d


def append_result_jsonl(path: Path, rr: RunResult) -> None:
    """Append a single RunResult as one JSONL line, flushing immediately."""
    d = run_result_to_dict(rr)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, default=str) + "\n")
        f.flush()
