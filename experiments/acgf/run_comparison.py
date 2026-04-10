#!/usr/bin/env python3
"""Run the ACGF comparison experiment.

Runs all optimizer variants across multiple seeds on the same scenario
and budget, saving results incrementally as JSONL.

Usage:
    python -u experiments/acgf/run_comparison.py \
        --scenario edge_tight \
        --budget 25 \
        --seeds 10 \
        --output-dir results/acgf_comparison/ \
        --dataset-path data \
        --gpu-id 0 \
        --variants random tpe tba tba_margin_1 tba_margin_2 tba_margin_3 tba_margin_5 tba_acgf
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tba.spaces.image_classification import SCENARIOS
from experiments.acgf.acgf_evaluator import ACGFParams
from experiments.acgf.optimizer_variants import (
    run_variant,
    append_result_jsonl,
)


# ── Variant definitions ──────────────────────────────────────────────

VARIANT_DEFS: dict[str, dict] = {
    "random":        {"margin_ms": 0.0, "acgf": False},
    "tpe":           {"margin_ms": 0.0, "acgf": False},
    "tba":           {"margin_ms": 0.0, "acgf": False},
    "tba_margin_1":  {"margin_ms": 1.0, "acgf": False},
    "tba_margin_2":  {"margin_ms": 2.0, "acgf": False},
    "tba_margin_3":  {"margin_ms": 3.0, "acgf": False},
    "tba_margin_5":  {"margin_ms": 5.0, "acgf": False},
    "tba_acgf":      {"margin_ms": 0.0, "acgf": True},
}

ALL_VARIANTS = list(VARIANT_DEFS.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACGF comparison: run optimizer variants head-to-head",
    )
    p.add_argument(
        "--scenario", default="edge_tight",
        choices=list(SCENARIOS.keys()),
        help="Constraint scenario (default: edge_tight)",
    )
    p.add_argument("--budget", type=int, default=25, help="Evaluation budget per run")
    p.add_argument("--seeds", type=int, default=10, help="Number of seeds (0..seeds-1)")
    p.add_argument(
        "--output-dir", type=str, default="results/acgf_comparison",
        help="Directory for result JSONL",
    )
    p.add_argument("--dataset-path", type=str, default="data", help="Path to dataset")
    p.add_argument("--gpu-id", type=int, default=0, help="GPU index")
    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument(
        "--variants", nargs="+", default=ALL_VARIANTS,
        choices=ALL_VARIANTS,
        help="Variants to run (default: all)",
    )
    # ACGF parameters
    p.add_argument("--acgf-band-width", type=float, default=5.0)
    p.add_argument("--acgf-max-reeval", type=int, default=5)
    p.add_argument("--acgf-confidence", type=float, default=0.7)
    # Resume support
    p.add_argument(
        "--resume", action="store_true",
        help="Skip variant+seed combos that already appear in the output file",
    )
    return p.parse_args()


def load_completed(output_path: Path) -> set[tuple[str, int]]:
    """Load (variant, seed) pairs already in the JSONL file."""
    completed: set[tuple[str, int]] = set()
    if not output_path.exists():
        return completed
    import json
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                completed.add((d["variant"], d["seed"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def main() -> None:
    args = parse_args()

    scenario_def = SCENARIOS[args.scenario]
    constraint_caps = scenario_def["constraints"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison_results.jsonl"

    acgf_params = ACGFParams(
        band_width_ms=args.acgf_band_width,
        max_reeval=args.acgf_max_reeval,
        confidence_threshold=args.acgf_confidence,
    )

    # Build work list
    work: list[tuple[str, int]] = []
    for variant in args.variants:
        for seed in range(args.seeds):
            work.append((variant, seed))

    # Resume support
    if args.resume:
        completed = load_completed(output_path)
        work = [(v, s) for v, s in work if (v, s) not in completed]
        if completed:
            print(f"Resuming: {len(completed)} runs already done, {len(work)} remaining")

    total_runs = len(work)
    if total_runs == 0:
        print("All runs already completed.")
        return

    print(f"ACGF Comparison Experiment")
    print(f"  Scenario:   {args.scenario}")
    print(f"  Budget:     {args.budget} evaluations")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Variants:   {', '.join(args.variants)}")
    print(f"  Total runs: {total_runs}")
    print(f"  Output:     {output_path}")
    print(f"  ACGF:       band={acgf_params.band_width_ms}ms  "
          f"max_reeval={acgf_params.max_reeval}  "
          f"conf={acgf_params.confidence_threshold}")
    print()

    experiment_start = time.time()

    for run_idx, (variant, seed) in enumerate(work):
        vdef = VARIANT_DEFS[variant]
        use_acgf = vdef["acgf"]
        margin = vdef["margin_ms"]

        elapsed = time.time() - experiment_start
        if run_idx > 0:
            avg_per_run = elapsed / run_idx
            remaining = avg_per_run * (total_runs - run_idx)
            eta_min = remaining / 60
            eta_str = f"  ETA: {eta_min:.0f}min"
        else:
            eta_str = ""

        print(
            f"{'='*60}\n"
            f"Run {run_idx+1}/{total_runs}: {variant} seed={seed}  "
            f"[margin={margin}ms acgf={use_acgf}]{eta_str}\n"
            f"{'='*60}",
            flush=True,
        )

        rr = run_variant(
            variant_name=variant,
            budget=args.budget,
            data_dir=args.dataset_path,
            device=args.device,
            constraint_caps=constraint_caps,
            seed=seed,
            scenario=args.scenario,
            acgf_params=acgf_params if use_acgf else None,
            margin_ms=margin,
            gpu_id=args.gpu_id,
            verbose=True,
        )

        append_result_jsonl(output_path, rr)

        best_str = f"{rr.best_accuracy:.4f}" if rr.best_accuracy is not None else "NONE"
        print(
            f"  >> {variant} seed={seed}: best={best_str}  "
            f"feasible={rr.n_feasible} infeas={rr.n_infeasible} "
            f"crash={rr.n_crashed} border={rr.n_borderline} "
            f"reeval={rr.n_reeval} rescued={rr.n_rescued} caught={rr.n_caught}  "
            f"[{rr.wall_clock_s:.0f}s]\n",
            flush=True,
        )

    total_time = time.time() - experiment_start
    print(f"\nExperiment complete in {total_time/60:.1f} minutes.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
