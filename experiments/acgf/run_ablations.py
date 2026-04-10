#!/usr/bin/env python3
"""Run ACGF ablation experiments.

Four ablations that interrogate the ACGF algorithm:
  A. Fixed-K vs Adaptive-K  — is proportional allocation better than fixed?
  B. Band width sensitivity — how sensitive is ACGF to band_width_ms?
  C. Confidence threshold   — conservative vs permissive feasibility gate
  D. Thermal conditioning   — widen band when GPU is hot (optional bonus)

Usage:
    python -u experiments/acgf/run_ablations.py \
        --ablation A \
        --budget 25 --seeds 10 \
        --output-dir results/acgf_comparison/ablations/ \
        --dataset-path data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tba.spaces.image_classification import SCENARIOS
from experiments.acgf.acgf_evaluator import ACGFParams
from experiments.acgf.optimizer_variants import (
    run_variant,
    append_result_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACGF ablation experiments")
    p.add_argument(
        "--ablation", required=True, choices=["A", "B", "C", "D", "all"],
        help="Which ablation to run (A/B/C/D/all)",
    )
    p.add_argument("--scenario", default="edge_tight")
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="results/acgf_comparison/ablations")
    p.add_argument("--dataset-path", type=str, default="data")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_completed(path: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    if not path.exists():
        return completed
    with open(path, encoding="utf-8") as f:
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


def run_ablation_set(
    name: str,
    configs: list[tuple[str, ACGFParams]],
    args: argparse.Namespace,
    constraint_caps: dict[str, float],
) -> None:
    """Run a set of ACGF parameter configs across seeds."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ablation_{name}.jsonl"

    work: list[tuple[str, ACGFParams, int]] = []
    for variant_label, params in configs:
        for seed in range(args.seeds):
            work.append((variant_label, params, seed))

    if args.resume:
        completed = load_completed(output_path)
        work = [(v, p, s) for v, p, s in work if (v, s) not in completed]
        if completed:
            print(f"  Resuming: {len(completed)} done, {len(work)} remaining")

    total = len(work)
    if total == 0:
        print(f"  Ablation {name}: all runs completed.")
        return

    print(f"\nAblation {name}: {total} runs")
    print(f"  Output: {output_path}")
    start = time.time()

    for idx, (variant_label, params, seed) in enumerate(work):
        elapsed = time.time() - start
        if idx > 0:
            eta = (elapsed / idx) * (total - idx) / 60
            eta_str = f"  ETA: {eta:.0f}min"
        else:
            eta_str = ""

        print(
            f"  [{idx+1}/{total}] {variant_label} seed={seed}{eta_str}",
            flush=True,
        )

        rr = run_variant(
            variant_name=variant_label,
            budget=args.budget,
            data_dir=args.dataset_path,
            device=args.device,
            constraint_caps=constraint_caps,
            seed=seed,
            scenario=args.scenario,
            acgf_params=params,
            margin_ms=0.0,
            gpu_id=args.gpu_id,
            verbose=False,
        )

        append_result_jsonl(output_path, rr)

        best_str = f"{rr.best_accuracy:.4f}" if rr.best_accuracy is not None else "NONE"
        print(
            f"    best={best_str} border={rr.n_borderline} "
            f"reeval={rr.n_reeval} rescued={rr.n_rescued}  "
            f"[{rr.wall_clock_s:.0f}s]",
            flush=True,
        )

    total_time = time.time() - start
    print(f"  Ablation {name} done in {total_time/60:.1f} min\n")


def main() -> None:
    args = parse_args()
    scenario_def = SCENARIOS[args.scenario]
    constraint_caps = scenario_def["constraints"]

    ablations_to_run = (
        ["A", "B", "C", "D"] if args.ablation == "all" else [args.ablation]
    )

    for abl in ablations_to_run:
        if abl == "A":
            # Fixed-K vs Adaptive-K
            configs = [
                ("acgf_adaptive", ACGFParams(
                    band_width_ms=3.0, max_reeval=5,
                    confidence_threshold=0.7, fixed_k=None,
                )),
                ("acgf_fixed_k1", ACGFParams(
                    band_width_ms=3.0, max_reeval=5,
                    confidence_threshold=0.7, fixed_k=1,
                )),
                ("acgf_fixed_k2", ACGFParams(
                    band_width_ms=3.0, max_reeval=5,
                    confidence_threshold=0.7, fixed_k=2,
                )),
                ("acgf_fixed_k3", ACGFParams(
                    band_width_ms=3.0, max_reeval=5,
                    confidence_threshold=0.7, fixed_k=3,
                )),
                ("acgf_fixed_k5", ACGFParams(
                    band_width_ms=3.0, max_reeval=5,
                    confidence_threshold=0.7, fixed_k=5,
                )),
            ]
            run_ablation_set("A_fixed_vs_adaptive_k", configs, args, constraint_caps)

        elif abl == "B":
            # Band width sensitivity
            configs = [
                (f"acgf_band_{bw}ms", ACGFParams(
                    band_width_ms=bw, max_reeval=5,
                    confidence_threshold=0.7,
                ))
                for bw in [2.0, 3.0, 4.0, 5.0, 7.0]
            ]
            run_ablation_set("B_band_width", configs, args, constraint_caps)

        elif abl == "C":
            # Confidence threshold sensitivity
            configs = [
                (f"acgf_conf_{ct}", ACGFParams(
                    band_width_ms=3.0, max_reeval=5,
                    confidence_threshold=ct,
                ))
                for ct in [0.5, 0.6, 0.7, 0.8, 0.9]
            ]
            run_ablation_set("C_confidence_threshold", configs, args, constraint_caps)

        elif abl == "D":
            # Thermal conditioning: wider band when GPU > 70°C
            # Implemented by using a wider band (proxy — actual thermal
            # conditioning would require modifying ACGF at eval time)
            configs = [
                ("acgf_no_thermal", ACGFParams(
                    band_width_ms=3.0, max_reeval=5,
                    confidence_threshold=0.7,
                )),
                ("acgf_wide_band_thermal", ACGFParams(
                    band_width_ms=4.5, max_reeval=5,
                    confidence_threshold=0.7,
                )),
                ("acgf_extra_reeval_thermal", ACGFParams(
                    band_width_ms=3.0, max_reeval=7,
                    confidence_threshold=0.7,
                )),
            ]
            run_ablation_set("D_thermal_proxy", configs, args, constraint_caps)

    print("All requested ablations complete.")


if __name__ == "__main__":
    main()
