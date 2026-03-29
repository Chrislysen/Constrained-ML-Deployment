"""Budget sweep: all optimizers across budget = 10,15,20,25,30,40,50 on crashy Branin.

Generates:
  1. Line chart: best feasible objective vs budget (with error bands)
  2. Line chart: wasted budget fraction vs budget
  3. Full summary table
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tba.types import EvalResult
from tba.optimizer.search_space import SearchSpace
from tba.optimizer.tba_optimizer import TBAOptimizer
from tba.optimizer.tba_tpe_hybrid import TBATPEHybrid
from tba.baselines.random_search import RandomSearchOptimizer
from tba.baselines.optuna_tpe import OptunaTPEOptimizer
from tba.baselines.constrained_bo import ConstrainedBOOptimizer
from tba.benchmarks.synthetic_bench import BENCHMARKS


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


OPTIMIZERS = {
    "tba_tpe": TBATPEHybrid,
    "tba_nosurr": TBAOptimizer,
    "random": RandomSearchOptimizer,
    "tpe": OptunaTPEOptimizer,
    "bo": ConstrainedBOOptimizer,
}

OPT_COLORS = {
    "tba_tpe": "#e74c3c",
    "tba_nosurr": "#c0392b",
    "random": "#95a5a6",
    "tpe": "#2ecc71",
    "bo": "#3498db",
}

OPT_LABELS = {
    "tba_tpe": "TBA>TPE Hybrid",
    "tba_nosurr": "TBA (pure SA)",
    "random": "Random Search",
    "tpe": "Optuna TPE",
    "bo": "Constrained BO",
}

OPT_LINESTYLES = {
    "tba_tpe": "-",
    "tba_nosurr": "--",
    "random": ":",
    "tpe": "-",
    "bo": "-.",
}

RESULTS_DIR = Path("results_budget_sweep_v6")
N_SEEDS = 10
BUDGETS = [10, 15, 20, 25, 30, 40, 50]
BENCHMARK = "crashy_branin"
OPT_NAMES = ["tba_tpe", "tba_nosurr", "random", "tpe", "bo"]


def make_optimizer(name, space, constraints, budget, seed):
    cls = OPTIMIZERS[name]
    kwargs = dict(
        search_space=space, constraints=constraints,
        objective="maximize", budget=budget, seed=seed,
    )
    if name == "tba_nosurr":
        kwargs["surrogate"] = False
    return cls(**kwargs)


import warnings
warnings.filterwarnings("ignore", message=".*does not have constraint values.*")


def run_single(opt_name, budget, seed):
    bench = BENCHMARKS[BENCHMARK]
    space = bench["space_fn"]()
    constraints = bench["constraints"]
    eval_fn = bench["eval_fn"]

    opt = make_optimizer(opt_name, space, constraints, budget, seed)
    cumulative_crashes = 0
    cumulative_infeasible = 0
    best_feasible_so_far = None

    for trial_id in range(budget):
        config = opt.ask()
        try:
            result = eval_fn(config)
        except Exception as e:
            result = EvalResult(crashed=True, error_msg=str(e)[:200])
        opt.tell(config, result)

        if result.crashed:
            cumulative_crashes += 1
        elif not result.feasible:
            cumulative_infeasible += 1

        bf = opt.best_feasible()
        if bf is not None:
            best_feasible_so_far = bf[1].objective_value

    n_feasible = budget - cumulative_crashes - cumulative_infeasible
    ttff = None
    for i, (_, r) in enumerate(opt.history):
        if r.feasible and not r.crashed:
            ttff = i + 1
            break

    return {
        "optimizer": opt_name,
        "seed": seed,
        "budget": budget,
        "n_crashes": cumulative_crashes,
        "n_infeasible": cumulative_infeasible,
        "n_feasible": n_feasible,
        "wasted_fraction": (cumulative_crashes + cumulative_infeasible) / budget,
        "best_feasible_obj": best_feasible_so_far,
        "time_to_first_feasible": ttff,
    }


def main():
    # master_results[opt_name][budget] = list of summary dicts (one per seed)
    master_results: dict[str, dict[int, list[dict]]] = {
        name: {b: [] for b in BUDGETS} for name in OPT_NAMES
    }

    total = len(OPT_NAMES) * len(BUDGETS) * N_SEEDS
    done = 0

    for opt_name in OPT_NAMES:
        t0 = time.time()
        for budget in BUDGETS:
            for seed in range(N_SEEDS):
                summary = run_single(opt_name, budget, seed)
                master_results[opt_name][budget].append(summary)
                done += 1
        elapsed = time.time() - t0
        print(f"  {OPT_LABELS[opt_name]:<16} done  [{elapsed:.1f}s]  ({done}/{total})")

    # --- Plot 1: Best feasible objective vs budget ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    for opt_name in OPT_NAMES:
        means, stds = [], []
        for b in BUDGETS:
            objs = [s["best_feasible_obj"] for s in master_results[opt_name][b]
                    if s["best_feasible_obj"] is not None]
            if objs:
                means.append(np.mean(objs))
                stds.append(np.std(objs))
            else:
                means.append(np.nan)
                stds.append(0)
        means = np.array(means)
        stds = np.array(stds)

        color = OPT_COLORS[opt_name]
        ls = OPT_LINESTYLES[opt_name]
        ax.plot(BUDGETS, means, label=OPT_LABELS[opt_name], color=color,
                linestyle=ls, linewidth=2.5, marker="o", markersize=6)
        ax.fill_between(BUDGETS, means - stds, means + stds, alpha=0.12, color=color)

    ax.set_xlabel("Trial Budget", fontsize=13)
    ax.set_ylabel("Best Feasible Objective (higher is better)", fontsize=13)
    ax.set_title("Budget Sweep — Crashy Branin", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(BUDGETS)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "objective_vs_budget.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {RESULTS_DIR / 'objective_vs_budget.png'}")

    # --- Plot 2: Wasted budget fraction vs budget ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for opt_name in OPT_NAMES:
        wasted_means = []
        for b in BUDGETS:
            fracs = [s["wasted_fraction"] for s in master_results[opt_name][b]]
            wasted_means.append(np.mean(fracs))

        color = OPT_COLORS[opt_name]
        ls = OPT_LINESTYLES[opt_name]
        ax.plot(BUDGETS, wasted_means, label=OPT_LABELS[opt_name], color=color,
                linestyle=ls, linewidth=2.5, marker="s", markersize=6)

    ax.set_xlabel("Trial Budget", fontsize=13)
    ax.set_ylabel("Fraction of Budget Wasted", fontsize=13)
    ax.set_title("Wasted Budget vs Trial Budget — Crashy Branin", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(BUDGETS)
    ax.set_ylim(0, 0.85)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "wasted_vs_budget.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {RESULTS_DIR / 'wasted_vs_budget.png'}")

    # --- Full summary table ---
    print("\n" + "=" * 130)
    print("BUDGET SWEEP — CRASHY BRANIN — FULL TABLE")
    print("=" * 130)
    header = f"{'Budget':>6}  {'Optimizer':<16}  {'Best Obj (mean±std)':>22}  {'Wasted %':>10}  {'TTFF':>8}  {'Feas Seeds':>12}"
    print(header)
    print("-" * 130)

    for b in BUDGETS:
        for opt_name in OPT_NAMES:
            sums = master_results[opt_name][b]
            objs = [s["best_feasible_obj"] for s in sums if s["best_feasible_obj"] is not None]
            wasted = [s["wasted_fraction"] for s in sums]
            ttffs = [s["time_to_first_feasible"] for s in sums if s["time_to_first_feasible"] is not None]

            if objs:
                obj_str = f"{np.mean(objs):>8.2f} ± {np.std(objs):<8.2f}"
            else:
                obj_str = "    N/A          "
            wasted_str = f"{np.mean(wasted):>8.1%}"
            ttff_str = f"{np.mean(ttffs):>6.1f}" if ttffs else "   N/A"
            seeds_str = f"{len(objs):>5}/{len(sums)}"

            print(f"{b:>6}  {OPT_LABELS[opt_name]:<16}  {obj_str:>22}  {wasted_str:>10}  {ttff_str:>8}  {seeds_str:>12}")
        print()

    print("=" * 130)

    # Save raw
    with open(RESULTS_DIR / "sweep_summary.json", "w") as f:
        json.dump(master_results, f, indent=2, cls=_NumpyEncoder)
    print(f"\nRaw results saved to {RESULTS_DIR / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
