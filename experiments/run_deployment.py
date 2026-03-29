"""Phase 3: Run TBA-TPE Hybrid vs baselines on real deployment benchmarks.

Supports both edge_tight (maximize accuracy) and server_throughput (maximize throughput).
Prints best config found by each optimizer per seed.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*does not have constraint values.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ExperimentalWarning.*")
warnings.filterwarnings("ignore", message=".*Missing annotation.*")

from tba.types import EvalResult
from tba.optimizer.tba_tpe_hybrid import TBATPEHybrid
from tba.optimizer.tba_optimizer import TBAOptimizer
from tba.baselines.random_search import RandomSearchOptimizer
from tba.baselines.optuna_tpe import OptunaTPEOptimizer
from tba.benchmarks.profiler import evaluate_config
from tba.spaces.image_classification import image_classification_space, SCENARIOS


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


OPTIMIZERS = {
    "tba_tpe": TBATPEHybrid,
    "tba_sa": TBAOptimizer,
    "random": RandomSearchOptimizer,
    "tpe": OptunaTPEOptimizer,
}

OPT_LABELS = {
    "tba_tpe": "TBA>TPE Hybrid",
    "tba_sa": "TBA (pure SA)",
    "random": "Random Search",
    "tpe": "Optuna TPE",
}

DATA_DIR = "data/imagenette2-320"
DEVICE = "cuda"
RESULTS_DIR = Path("results_deployment_v2")


def make_optimizer(name, space, constraints, budget, seed):
    cls = OPTIMIZERS[name]
    kwargs = dict(
        search_space=space, constraints=constraints,
        objective="maximize", budget=budget, seed=seed,
    )
    if name == "tba_sa":
        kwargs["surrogate"] = False
    return cls(**kwargs)


def config_str(config):
    m = config.get("model_name", "?")
    b = config.get("backend", "?")
    q = config.get("quantization", "?")
    bs = config.get("batch_size", "?")
    return f"{m}/{b}/{q}/bs={bs}"


def run_single(opt_name, scenario_name, seed, objective_type):
    scenario = SCENARIOS[scenario_name]
    space = image_classification_space()
    constraints = scenario["constraints"]
    budget = scenario["budget"]

    opt = make_optimizer(opt_name, space, constraints, budget, seed)

    records = []
    cumulative_crashes = 0
    cumulative_infeasible = 0
    best_feasible_so_far = None
    best_config_so_far = None
    wall_start = time.time()

    for trial_id in range(budget):
        config = opt.ask()

        result = evaluate_config(config, DATA_DIR, device=DEVICE, constraint_caps=constraints)

        # For throughput objective, override objective_value
        if objective_type == "maximize_throughput" and not result.crashed:
            result.objective_value = result.constraints.get("throughput_qps", 0.0)

        opt.tell(config, result)

        if result.crashed:
            cumulative_crashes += 1
            status = "CRASH"
        elif not result.feasible:
            cumulative_infeasible += 1
            status = "INFEAS"
        else:
            status = "OK"

        bf = opt.best_feasible()
        if bf is not None:
            best_feasible_so_far = bf[1].objective_value
            best_config_so_far = bf[0]

        wall = time.time() - wall_start
        obj_val = result.objective_value if not result.crashed else 0
        lat95 = result.constraints.get("latency_p95_ms", 0)
        lat99 = result.constraints.get("latency_p99_ms", 0)
        mem = result.constraints.get("memory_peak_mb", 0)
        tput = result.constraints.get("throughput_qps", 0)

        cs = config_str(config)
        best_str = f"{best_feasible_so_far:.4f}" if best_feasible_so_far else "N/A"
        print(f"    [{opt_name:>7}] t{trial_id:2d} {cs:<50} {status:<6} "
              f"obj={obj_val:.3f} p95={lat95:.1f} p99={lat99:.1f} mem={mem:.0f} "
              f"tput={tput:.0f}  best={best_str} [{result.eval_time_s:.1f}s]")

        records.append({
            "trial_id": trial_id,
            "config": config,
            "result": asdict(result),
            "best_feasible_so_far": best_feasible_so_far,
            "cumulative_crashes": cumulative_crashes,
            "cumulative_infeasible": cumulative_infeasible,
            "wall_clock_s": wall,
        })

    # Save JSONL
    out_dir = RESULTS_DIR / scenario_name / opt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"seed_{seed}.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, cls=_NumpyEncoder) + "\n")

    return records, best_config_so_far


def summarize_run(records, opt_name, seed, budget):
    last = records[-1]
    n_total = len(records)
    n_crashes = last["cumulative_crashes"]
    n_infeasible = last["cumulative_infeasible"]
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
        "n_feasible": n_total - n_crashes - n_infeasible,
        "wasted_fraction": (n_crashes + n_infeasible) / n_total,
        "best_feasible_obj": best,
        "time_to_first_feasible": ttff,
        "wall_clock_s": last["wall_clock_s"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+",
                        default=["edge_tight", "server_throughput"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--optimizers", nargs="+",
                        default=["tba_tpe", "tba_sa", "random", "tpe"])
    args = parser.parse_args()

    n_seeds = args.seeds
    opt_names = args.optimizers

    for scenario_name in args.scenarios:
        scenario = SCENARIOS[scenario_name]
        budget = scenario["budget"]
        objective_type = scenario["objective"]

        print(f"\n{'='*110}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"  Constraints: {scenario['constraints']}")
        print(f"  Objective: {objective_type}")
        print(f"  Budget: {budget}  Seeds: {n_seeds}")
        print(f"{'='*110}")

        all_summaries: dict[str, list[dict]] = {n: [] for n in opt_names}
        all_best_configs: dict[str, list[dict | None]] = {n: [] for n in opt_names}

        for opt_name in opt_names:
            for seed in range(n_seeds):
                print(f"\n--- {OPT_LABELS[opt_name]} | seed={seed} ---")
                records, best_cfg = run_single(opt_name, scenario_name, seed, objective_type)
                summary = summarize_run(records, opt_name, seed, budget)
                all_summaries[opt_name].append(summary)
                all_best_configs[opt_name].append(best_cfg)

                s = summary
                obj_str = f"{s['best_feasible_obj']:.4f}" if s['best_feasible_obj'] else "N/A"
                cfg_str = config_str(best_cfg) if best_cfg else "NONE"
                print(f"  => Best={obj_str} Wasted={s['wasted_fraction']:.0%} "
                      f"TTFF={s['time_to_first_feasible']} Config={cfg_str}")

        # --- Per-seed best config table ---
        print(f"\n{'='*110}")
        print(f"  BEST CONFIG PER SEED — {scenario_name}")
        print(f"{'='*110}")
        header = f"{'Seed':>4}  " + "  ".join(f"{OPT_LABELS[n]:>20}" for n in opt_names)
        print(header)
        print("-" * 110)
        for seed in range(n_seeds):
            parts = []
            for opt_name in opt_names:
                cfg = all_best_configs[opt_name][seed]
                if cfg:
                    m = cfg.get("model_name", "?")[:10]
                    b = cfg.get("backend", "?")[:8]
                    q = cfg.get("quantization", "?")[:5]
                    bs = cfg.get("batch_size", "?")
                    parts.append(f"{m}/{q}/bs{bs}")
                else:
                    parts.append("NONE")
            print(f"  {seed:>2}  " + "  ".join(f"{p:>20}" for p in parts))

        # --- Summary table ---
        print(f"\n{'='*110}")
        print(f"  SUMMARY — {scenario_name} ({objective_type})")
        print(f"{'='*110}")
        obj_label = "Accuracy" if "accuracy" in objective_type else "Throughput"
        header = f"{'Optimizer':<18} {'Best '+obj_label+' (mean+-std)':>28} {'Wasted %':>10} {'TTFF':>8} {'Seeds w/ feas':>14}"
        print(header)
        print("-" * 110)

        for opt_name in opt_names:
            sums = all_summaries[opt_name]
            objs = [s["best_feasible_obj"] for s in sums if s["best_feasible_obj"] is not None]
            wasted = [s["wasted_fraction"] for s in sums]
            ttffs = [s["time_to_first_feasible"] for s in sums if s["time_to_first_feasible"] is not None]

            if objs:
                obj_str = f"{np.mean(objs):.4f} +- {np.std(objs):.4f}"
            else:
                obj_str = "N/A"
            wasted_str = f"{np.mean(wasted):.0%}"
            ttff_str = f"{np.mean(ttffs):.1f}" if ttffs else "N/A"
            seeds_str = f"{len(objs)}/{len(sums)}"

            print(f"{OPT_LABELS[opt_name]:<18} {obj_str:>28} {wasted_str:>10} {ttff_str:>8} {seeds_str:>14}")

        print("=" * 110)

        # Save
        summary_path = RESULTS_DIR / scenario_name / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({"summaries": all_summaries,
                        "best_configs": {k: [c for c in v] for k, v in all_best_configs.items()}},
                      f, indent=2, cls=_NumpyEncoder)


if __name__ == "__main__":
    main()
