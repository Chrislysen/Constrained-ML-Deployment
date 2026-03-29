"""Phase 2: Run TBA + Random + TPE + BO on both synthetic benchmarks
at budget=15 and budget=30 with 10 seeds each.

Generates overlaid regret curves and a unified summary table.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tba.types import EvalResult
from tba.optimizer.search_space import SearchSpace
from tba.optimizer.tba_optimizer import TBAOptimizer
from tba.baselines.random_search import RandomSearchOptimizer
from tba.baselines.optuna_tpe import OptunaTPEOptimizer
from tba.baselines.constrained_bo import ConstrainedBOOptimizer
from tba.benchmarks.synthetic_bench import BENCHMARKS

OPTIMIZERS = {
    "tba": TBAOptimizer,
    "tba_nosurr": TBAOptimizer,
    "random": RandomSearchOptimizer,
    "tpe": OptunaTPEOptimizer,
    "bo": ConstrainedBOOptimizer,
}

OPT_COLORS = {
    "tba": "#e74c3c",
    "tba_nosurr": "#c0392b",
    "random": "#95a5a6",
    "tpe": "#2ecc71",
    "bo": "#3498db",
}

OPT_LABELS = {
    "tba": "TBA+Surrogate",
    "tba_nosurr": "TBA (no surr)",
    "random": "Random Search",
    "tpe": "Optuna TPE",
    "bo": "Constrained BO",
}

RESULTS_DIR = Path("results_phase2_ablation")
N_SEEDS = 10
BUDGETS = [15, 30]
BENCHMARK_NAMES = ["crashy_branin", "hierarchical_rosenbrock"]
OPT_NAMES = ["tba", "tba_nosurr", "random", "tpe", "bo"]


def make_optimizer(name, space, constraints, budget, seed):
    cls = OPTIMIZERS[name]
    kwargs = dict(
        search_space=space, constraints=constraints,
        objective="maximize", budget=budget, seed=seed,
    )
    if name == "tba_nosurr":
        kwargs["surrogate"] = False
    return cls(**kwargs)


def run_single(opt_name, bench_name, budget, seed):
    bench = BENCHMARKS[bench_name]
    space = bench["space_fn"]()
    constraints = bench["constraints"]
    eval_fn = bench["eval_fn"]

    opt = make_optimizer(opt_name, space, constraints, budget, seed)
    records = []
    cumulative_crashes = 0
    cumulative_infeasible = 0
    best_feasible_so_far = None
    wall_start = time.time()

    for trial_id in range(budget):
        config = opt.ask()
        t0 = time.time()
        try:
            result = eval_fn(config)
        except Exception as e:
            result = EvalResult(crashed=True, error_msg=f"{type(e).__name__}: {str(e)[:200]}")
        result.eval_time_s = time.time() - t0
        opt.tell(config, result)

        if result.crashed:
            cumulative_crashes += 1
        elif not result.feasible:
            cumulative_infeasible += 1

        bf = opt.best_feasible()
        if bf is not None:
            best_feasible_so_far = bf[1].objective_value

        records.append({
            "trial_id": trial_id,
            "config": config,
            "result": asdict(result),
            "best_feasible_so_far": best_feasible_so_far,
            "cumulative_crashes": cumulative_crashes,
            "cumulative_infeasible": cumulative_infeasible,
            "wall_clock_s": time.time() - wall_start,
        })

    # Save JSONL
    out_dir = RESULTS_DIR / bench_name / f"budget_{budget}" / opt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"seed_{seed}.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, cls=_NumpyEncoder) + "\n")

    return records


def summarize_run(records, opt_name, seed, budget):
    n_total = len(records)
    last = records[-1]
    n_crashes = last["cumulative_crashes"]
    n_infeasible = last["cumulative_infeasible"]
    n_feasible = n_total - n_crashes - n_infeasible
    best = last["best_feasible_so_far"]

    ttff = None
    for r in records:
        if r["best_feasible_so_far"] is not None:
            ttff = r["trial_id"] + 1
            break

    return {
        "optimizer": opt_name,
        "seed": seed,
        "budget": budget,
        "n_crashes": n_crashes,
        "n_infeasible": n_infeasible,
        "n_feasible": n_feasible,
        "wasted_fraction": (n_crashes + n_infeasible) / n_total,
        "best_feasible_obj": best,
        "time_to_first_feasible": ttff,
        "wall_clock_s": last["wall_clock_s"],
    }


def load_curves(bench_name, budget, opt_name):
    """Load per-trial best-feasible curves for all seeds."""
    curves = []
    for seed in range(N_SEEDS):
        path = RESULTS_DIR / bench_name / f"budget_{budget}" / opt_name / f"seed_{seed}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            records = [json.loads(line) for line in f]
        raw = [r["best_feasible_so_far"] for r in records]
        # Forward-fill Nones
        filled = []
        last = None
        for v in raw:
            if v is not None:
                last = v
            filled.append(last)
        curves.append(filled)
    return curves


def plot_regret_curves(bench_name, budget, all_summaries):
    """Regret curves with all optimizers overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for opt_name in OPT_NAMES:
        curves = load_curves(bench_name, budget, opt_name)
        if not curves:
            continue

        # Filter to curves that found feasible
        valid = [c for c in curves if c[-1] is not None]
        if not valid:
            continue

        # For curves with leading Nones, replace with a very bad value
        first_vals = [next((v for v in c if v is not None), None) for c in valid]
        bad_val = min(v for v in first_vals if v is not None) - 50
        padded = []
        for c in valid:
            padded.append([v if v is not None else bad_val for v in c])

        arr = np.array(padded)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        trials = np.arange(1, len(mean) + 1)

        color = OPT_COLORS[opt_name]
        label = OPT_LABELS[opt_name]
        ax.plot(trials, mean, label=label, color=color, linewidth=2.5)
        ax.fill_between(trials, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Trial Number", fontsize=13)
    ax.set_ylabel("Best Feasible Objective", fontsize=13)
    ax.set_title(f"Regret Curves — {bench_name} (budget={budget})", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_dir = RESULTS_DIR / bench_name / f"budget_{budget}" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "regret_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'regret_curves.png'}")


def plot_wasted_budget(bench_name, budget, all_summaries):
    """Stacked bar chart of wasted budget."""
    fig, ax = plt.subplots(figsize=(9, 5))

    crash_means, infeas_means = [], []
    labels = []
    for opt_name in OPT_NAMES:
        sums = all_summaries.get(opt_name, [])
        if not sums:
            continue
        labels.append(OPT_LABELS[opt_name])
        crash_means.append(np.mean([s["n_crashes"] / s["budget"] for s in sums]))
        infeas_means.append(np.mean([s["n_infeasible"] / s["budget"] for s in sums]))

    x = np.arange(len(labels))
    width = 0.5
    ax.bar(x, crash_means, width, label="Crashes", color="#e74c3c", alpha=0.8)
    ax.bar(x, infeas_means, width, bottom=crash_means, label="Infeasible", color="#f39c12", alpha=0.8)

    ax.set_ylabel("Fraction of Budget Wasted", fontsize=13)
    ax.set_title(f"Wasted Budget — {bench_name} (budget={budget})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    plot_dir = RESULTS_DIR / bench_name / f"budget_{budget}" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "wasted_budget.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'wasted_budget.png'}")


def print_unified_table(master_results):
    """Print a single summary table across all benchmarks, budgets, and optimizers."""
    print("\n" + "=" * 120)
    print("UNIFIED SUMMARY TABLE")
    print("=" * 120)

    header = f"{'Benchmark':<25} {'Budget':>6} {'Optimizer':<16} {'Best Obj (mean±std)':>22} {'Wasted %':>10} {'TTFF':>8} {'Feasible Seeds':>15}"
    print(header)
    print("-" * 120)

    for bench_name in BENCHMARK_NAMES:
        for budget in BUDGETS:
            key = (bench_name, budget)
            if key not in master_results:
                continue
            for opt_name in OPT_NAMES:
                sums = master_results[key].get(opt_name, [])
                if not sums:
                    continue

                best_objs = [s["best_feasible_obj"] for s in sums if s["best_feasible_obj"] is not None]
                wasted = [s["wasted_fraction"] for s in sums]
                ttffs = [s["time_to_first_feasible"] for s in sums if s["time_to_first_feasible"] is not None]
                n_found = len(best_objs)

                if best_objs:
                    mean_obj = np.mean(best_objs)
                    std_obj = np.std(best_objs)
                    obj_str = f"{mean_obj:>8.2f} ± {std_obj:<8.2f}"
                else:
                    obj_str = "    N/A          "

                wasted_str = f"{np.mean(wasted):>8.1%}"
                ttff_str = f"{np.mean(ttffs):>6.1f}" if ttffs else "   N/A"
                seeds_str = f"{n_found:>6}/{len(sums)}"

                label = OPT_LABELS[opt_name]
                print(f"{bench_name:<25} {budget:>6} {label:<16} {obj_str:>22} {wasted_str:>10} {ttff_str:>8} {seeds_str:>15}")
            print()  # blank line between budget groups

    print("=" * 120)


def main():
    master_results = {}
    total_runs = len(BENCHMARK_NAMES) * len(BUDGETS) * len(OPT_NAMES) * N_SEEDS
    completed = 0

    for bench_name in BENCHMARK_NAMES:
        for budget in BUDGETS:
            key = (bench_name, budget)
            master_results[key] = {}

            print(f"\n{'='*60}")
            print(f"  {bench_name} | budget={budget}")
            print(f"{'='*60}")

            for opt_name in OPT_NAMES:
                summaries = []
                t0 = time.time()

                for seed in range(N_SEEDS):
                    records = run_single(opt_name, bench_name, budget, seed)
                    summary = summarize_run(records, opt_name, seed, budget)
                    summaries.append(summary)
                    completed += 1

                elapsed = time.time() - t0
                master_results[key][opt_name] = summaries

                # Quick inline summary
                best_objs = [s["best_feasible_obj"] for s in summaries if s["best_feasible_obj"] is not None]
                wasted = np.mean([s["wasted_fraction"] for s in summaries])
                if best_objs:
                    print(f"  {OPT_LABELS[opt_name]:<16} obj={np.mean(best_objs):>8.2f}±{np.std(best_objs):.2f}  wasted={wasted:.1%}  [{elapsed:.1f}s]  ({completed}/{total_runs})")
                else:
                    print(f"  {OPT_LABELS[opt_name]:<16} NO FEASIBLE  wasted={wasted:.1%}  [{elapsed:.1f}s]  ({completed}/{total_runs})")

            # Generate plots
            plot_regret_curves(bench_name, budget, master_results[key])
            plot_wasted_budget(bench_name, budget, master_results[key])

    # Unified summary table
    print_unified_table(master_results)

    # Save raw summary
    summary_path = RESULTS_DIR / "phase2_summary.json"
    serializable = {}
    for (bench, bud), opt_sums in master_results.items():
        serializable[f"{bench}_budget{bud}"] = opt_sums
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2, cls=_NumpyEncoder)
    print(f"\nRaw results saved to {summary_path}")


if __name__ == "__main__":
    main()
