"""Run TBA vs baselines on synthetic benchmarks.

Usage:
    python experiments/run_synthetic.py
    python experiments/run_synthetic.py --benchmark crashy_branin --budget 30 --seeds 10
    python experiments/run_synthetic.py --optimizers tba random
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from tba.types import EvalResult
from tba.optimizer.search_space import SearchSpace
from tba.optimizer.tba_optimizer import TBAOptimizer
from tba.baselines.random_search import RandomSearchOptimizer
from tba.baselines.optuna_tpe import OptunaTPEOptimizer
from tba.baselines.constrained_bo import ConstrainedBOOptimizer
from tba.benchmarks.synthetic_bench import BENCHMARKS


# ------------------------------------------------------------------
# Optimizer factory
# ------------------------------------------------------------------

OPTIMIZERS = {
    "tba": TBAOptimizer,
    "random": RandomSearchOptimizer,
    "tpe": OptunaTPEOptimizer,
    "bo": ConstrainedBOOptimizer,
}


def make_optimizer(name: str, space: SearchSpace, constraints: dict, budget: int, seed: int):
    cls = OPTIMIZERS[name]
    return cls(
        search_space=space,
        constraints=constraints,
        objective="maximize",
        budget=budget,
        seed=seed,
    )


# ------------------------------------------------------------------
# Single run
# ------------------------------------------------------------------

def run_single(
    optimizer_name: str,
    benchmark_name: str,
    budget: int,
    seed: int,
    results_dir: Path,
) -> list[dict]:
    bench = BENCHMARKS[benchmark_name]
    space = bench["space_fn"]()
    constraints = bench["constraints"]
    eval_fn = bench["eval_fn"]

    opt = make_optimizer(optimizer_name, space, constraints, budget, seed)

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

        record = {
            "trial_id": trial_id,
            "config": config,
            "result": asdict(result),
            "best_feasible_so_far": best_feasible_so_far,
            "cumulative_crashes": cumulative_crashes,
            "cumulative_infeasible": cumulative_infeasible,
            "wall_clock_s": time.time() - wall_start,
        }
        records.append(record)

    # Save results
    out_dir = results_dir / benchmark_name / optimizer_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed_{seed}.jsonl"
    with open(out_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return records


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def summarize_run(records: list[dict], optimizer_name: str, seed: int) -> dict:
    n_total = len(records)
    last = records[-1]
    n_crashes = last["cumulative_crashes"]
    n_infeasible = last["cumulative_infeasible"]
    n_feasible = n_total - n_crashes - n_infeasible
    best = last["best_feasible_so_far"]

    # Time to first feasible
    ttff = None
    for r in records:
        if r["best_feasible_so_far"] is not None:
            ttff = r["trial_id"] + 1
            break

    return {
        "optimizer": optimizer_name,
        "seed": seed,
        "budget": n_total,
        "n_crashes": n_crashes,
        "n_infeasible": n_infeasible,
        "n_feasible": n_feasible,
        "wasted_fraction": (n_crashes + n_infeasible) / n_total,
        "best_feasible_obj": best,
        "time_to_first_feasible": ttff,
        "wall_clock_s": last["wall_clock_s"],
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run synthetic benchmarks")
    parser.add_argument("--benchmark", default="crashy_branin", choices=list(BENCHMARKS.keys()))
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--optimizers", nargs="+", default=["tba", "random"])
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_summaries: dict[str, list[dict]] = {name: [] for name in args.optimizers}

    print(f"Benchmark: {args.benchmark}")
    print(f"Budget: {args.budget} trials")
    print(f"Seeds: {args.seeds}")
    print(f"Optimizers: {args.optimizers}")
    print("=" * 60)

    for opt_name in args.optimizers:
        for seed in range(args.seeds):
            records = run_single(opt_name, args.benchmark, args.budget, seed, results_dir)
            summary = summarize_run(records, opt_name, seed)
            all_summaries[opt_name].append(summary)

    # Print aggregate results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for opt_name in args.optimizers:
        summaries = all_summaries[opt_name]
        best_objs = [s["best_feasible_obj"] for s in summaries if s["best_feasible_obj"] is not None]
        wasted = [s["wasted_fraction"] for s in summaries]
        ttffs = [s["time_to_first_feasible"] for s in summaries if s["time_to_first_feasible"] is not None]

        print(f"\n--- {opt_name.upper()} ---")
        if best_objs:
            mean_obj = sum(best_objs) / len(best_objs)
            std_obj = (sum((x - mean_obj) ** 2 for x in best_objs) / len(best_objs)) ** 0.5
            print(f"  Best feasible obj: {mean_obj:.4f} ± {std_obj:.4f}")
        else:
            print("  Best feasible obj: NO FEASIBLE CONFIG FOUND")

        mean_wasted = sum(wasted) / len(wasted)
        print(f"  Wasted budget:     {mean_wasted:.1%}")

        if ttffs:
            mean_ttff = sum(ttffs) / len(ttffs)
            print(f"  Time to 1st feas:  {mean_ttff:.1f} trials")
        else:
            print(f"  Time to 1st feas:  NEVER")

        n_found = len(best_objs)
        print(f"  Seeds w/ feasible: {n_found}/{len(summaries)}")

    # Save aggregate summary
    summary_file = results_dir / args.benchmark / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nDetailed results saved to {results_dir}/{args.benchmark}/")

    # Generate plots
    try:
        generate_plots(all_summaries, args.benchmark, args.budget, results_dir)
    except ImportError as e:
        print(f"\nSkipping plots (missing dependency): {e}")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def generate_plots(all_summaries: dict, benchmark: str, budget: int, results_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = results_dir / benchmark / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Regret curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"tba": "#e74c3c", "random": "#3498db", "optuna_tpe": "#2ecc71", "constrained_bo": "#9b59b6"}

    for opt_name, summaries in all_summaries.items():
        # Load JSONL files to get per-trial data
        all_best_curves = []
        for s in summaries:
            seed = s["seed"]
            jsonl_path = results_dir / benchmark / opt_name / f"seed_{seed}.jsonl"
            if jsonl_path.exists():
                with open(jsonl_path) as f:
                    records = [json.loads(line) for line in f]
                curve = [r["best_feasible_so_far"] for r in records]
                all_best_curves.append(curve)

        if not all_best_curves:
            continue

        # Forward-fill None values for plotting
        filled_curves = []
        for curve in all_best_curves:
            filled = []
            last_val = None
            for v in curve:
                if v is not None:
                    last_val = v
                filled.append(last_val)
            filled_curves.append(filled)

        # Compute mean and std (only over seeds that found feasible)
        import numpy as np
        # Filter out curves that never found feasible
        valid_curves = [c for c in filled_curves if c[-1] is not None]
        if not valid_curves:
            continue

        # Pad shorter curves
        max_len = max(len(c) for c in valid_curves)
        padded = []
        for c in valid_curves:
            # Replace leading Nones with a very bad value for plotting
            first_val = next((v for v in c if v is not None), None)
            if first_val is None:
                continue
            padded_c = [v if v is not None else first_val - 50 for v in c]
            padded.append(padded_c)

        if not padded:
            continue

        arr = np.array(padded)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        trials = np.arange(1, len(mean) + 1)

        color = colors.get(opt_name, "#333333")
        ax.plot(trials, mean, label=opt_name.upper(), color=color, linewidth=2)
        ax.fill_between(trials, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Best Feasible Objective", fontsize=12)
    ax.set_title(f"Regret Curves — {benchmark}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "regret_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'regret_curves.png'}")

    # --- Plot 2: Wasted budget bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    opt_names = list(all_summaries.keys())
    crash_means, infeas_means = [], []
    for name in opt_names:
        summaries = all_summaries[name]
        crash_fracs = [s["n_crashes"] / s["budget"] for s in summaries]
        infeas_fracs = [s["n_infeasible"] / s["budget"] for s in summaries]
        crash_means.append(sum(crash_fracs) / len(crash_fracs))
        infeas_means.append(sum(infeas_fracs) / len(infeas_fracs))

    import numpy as np
    x = np.arange(len(opt_names))
    width = 0.4
    bars1 = ax.bar(x, crash_means, width, label="Crashes", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x, infeas_means, width, bottom=crash_means, label="Infeasible", color="#f39c12", alpha=0.8)

    ax.set_xlabel("Optimizer", fontsize=12)
    ax.set_ylabel("Fraction of Budget Wasted", fontsize=12)
    ax.set_title(f"Wasted Budget — {benchmark}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in opt_names])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "wasted_budget.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'wasted_budget.png'}")

    # --- Plot 3: Time to first feasible ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ttff_data = []
    labels = []
    for name in opt_names:
        summaries = all_summaries[name]
        ttffs = [s["time_to_first_feasible"] for s in summaries if s["time_to_first_feasible"] is not None]
        if ttffs:
            ttff_data.append(ttffs)
            labels.append(name.upper())

    if ttff_data:
        bp = ax.boxplot(ttff_data, labels=labels, patch_artist=True)
        color_list = [colors.get(n, "#333") for n in opt_names if all_summaries[n][0].get("time_to_first_feasible") is not None or any(s["time_to_first_feasible"] is not None for s in all_summaries[n])]
        for patch, color in zip(bp["boxes"], color_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel("Trials to First Feasible Config", fontsize=12)
    ax.set_title(f"Time to First Feasible — {benchmark}", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "time_to_first_feasible.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'time_to_first_feasible.png'}")


if __name__ == "__main__":
    main()
