#!/usr/bin/env python3
"""Analyze ACGF comparison results and generate paper figures.

Reads JSONL output from run_comparison.py and run_ablations.py,
computes summary statistics, runs statistical tests, and produces:
  Table 1: Main results (accuracy, waste rate, flippers, budget)
  Figure 1: Convergence curves
  Figure 2: Budget allocation breakdown
  Figure 3: Flipper false-feasible rates
  Figure 4: Fixed margin sweep with ACGF baseline
  Figure 5: Ablation results (2x2 grid)

Usage:
    python -u experiments/acgf/analyze_comparison.py \
        --results results/acgf_comparison/comparison_results.jsonl \
        --ablation-dir results/acgf_comparison/ablations/ \
        --output-dir results/acgf_comparison/figures/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── Data loading ──────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_variant(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        v = r["variant"]
        groups.setdefault(v, []).append(r)
    return groups


# ── Summary statistics ────────────────────────────────────────────────


def compute_variant_stats(runs: list[dict]) -> dict[str, Any]:
    """Aggregate stats across seeds for one variant."""
    accs = [r["best_accuracy"] for r in runs if r["best_accuracy"] is not None]
    return {
        "n_seeds": len(runs),
        "accuracy_mean": float(np.mean(accs)) if accs else None,
        "accuracy_std": float(np.std(accs)) if accs else None,
        "accuracy_median": float(np.median(accs)) if accs else None,
        "accuracy_values": accs,
        "unique_configs_mean": float(np.mean([r["unique_configs"] for r in runs])),
        "n_feasible_mean": float(np.mean([r["n_feasible"] for r in runs])),
        "n_infeasible_mean": float(np.mean([r["n_infeasible"] for r in runs])),
        "n_crashed_mean": float(np.mean([r["n_crashed"] for r in runs])),
        "n_borderline_mean": float(np.mean([r["n_borderline"] for r in runs])),
        "n_reeval_mean": float(np.mean([r["n_reeval"] for r in runs])),
        "n_rescued_mean": float(np.mean([r["n_rescued"] for r in runs])),
        "n_caught_mean": float(np.mean([r["n_caught"] for r in runs])),
        "wall_clock_mean_s": float(np.mean([r["wall_clock_s"] for r in runs])),
        "waste_rate": float(np.mean([
            r["n_infeasible"] / max(r["unique_configs"], 1) for r in runs
        ])),
        "no_feasible_rate": sum(
            1 for r in runs if r["best_accuracy"] is None
        ) / len(runs),
    }


# ── Statistical tests ────────────────────────────────────────────────


def paired_test(accs_a: list[float], accs_b: list[float]) -> dict[str, float]:
    """Wilcoxon signed-rank test + Cohen's d between paired seed results."""
    from scipy import stats

    a = np.array(accs_a)
    b = np.array(accs_b)

    # Cohen's d
    diff = a - b
    d_mean = float(np.mean(diff))
    d_std = float(np.std(diff, ddof=1)) if len(diff) > 1 else 1e-10
    cohens_d = d_mean / d_std if d_std > 0 else 0.0

    # Wilcoxon signed-rank (need at least 6 non-zero diffs)
    nonzero = diff[diff != 0]
    if len(nonzero) >= 6:
        stat, p_value = stats.wilcoxon(nonzero)
    else:
        stat, p_value = float("nan"), float("nan")

    return {
        "mean_diff": d_mean,
        "cohens_d": cohens_d,
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p_value),
        "n_pairs": len(a),
    }


# ── Convergence curve data ────────────────────────────────────────────


def convergence_curves(
    runs: list[dict], budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_curve, std_curve) of best-so-far across seeds.

    Each curve has `budget` entries: best feasible accuracy after each
    evaluation unit.
    """
    curves = []
    for r in runs:
        curve = []
        evals_so_far = 0
        best = None
        for o in r["outcomes"]:
            evals_so_far += o["evaluations_used"]
            if o["result_feasible"] and not o["result_crashed"]:
                obj = o["result_objective"]
                if best is None or obj > best:
                    best = obj
            # Fill the curve for each eval unit consumed
            while len(curve) < evals_so_far and len(curve) < budget:
                curve.append(best if best is not None else 0.0)
        # Pad to budget length
        while len(curve) < budget:
            curve.append(best if best is not None else 0.0)
        curves.append(curve[:budget])
    arr = np.array(curves)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


# ── Figures ───────────────────────────────────────────────────────────


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


VARIANT_COLORS = {
    "random": "#999999",
    "tpe": "#2196F3",
    "tba": "#4CAF50",
    "tba_margin_1": "#FFC107",
    "tba_margin_2": "#FF9800",
    "tba_margin_3": "#FF5722",
    "tba_margin_5": "#9C27B0",
    "tba_acgf": "#E91E63",
}

VARIANT_LABELS = {
    "random": "Random",
    "tpe": "TPE",
    "tba": "TBA",
    "tba_margin_1": "TBA +1ms",
    "tba_margin_2": "TBA +2ms",
    "tba_margin_3": "TBA +3ms",
    "tba_margin_5": "TBA +5ms",
    "tba_acgf": "TBA+ACGF",
}


def figure1_convergence(groups: dict, budget: int, output_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots()

    for variant in ["random", "tpe", "tba", "tba_acgf"]:
        if variant not in groups:
            continue
        mean_c, std_c = convergence_curves(groups[variant], budget)
        x = np.arange(1, budget + 1)
        color = VARIANT_COLORS.get(variant, "#000")
        label = VARIANT_LABELS.get(variant, variant)
        ax.plot(x, mean_c, color=color, label=label, linewidth=2)
        ax.fill_between(x, mean_c - std_c, mean_c + std_c, alpha=0.15, color=color)

    ax.set_xlabel("Evaluation Budget Used")
    ax.set_ylabel("Best Feasible Accuracy")
    ax.set_title("Convergence: Best Accuracy vs Budget")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_convergence.png")
    plt.close(fig)
    print("  Saved fig1_convergence.png")


def figure2_budget_breakdown(
    groups: dict, all_stats: dict, output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    variants = [v for v in VARIANT_LABELS if v in all_stats]
    labels = [VARIANT_LABELS.get(v, v) for v in variants]

    feasible = [all_stats[v]["n_feasible_mean"] for v in variants]
    infeasible = [all_stats[v]["n_infeasible_mean"] for v in variants]
    crashed = [all_stats[v]["n_crashed_mean"] for v in variants]
    reeval = [all_stats[v]["n_reeval_mean"] for v in variants]

    x = np.arange(len(variants))
    width = 0.6

    fig, ax = plt.subplots()
    bottom = np.zeros(len(variants))

    ax.bar(x, feasible, width, label="Feasible", color="#4CAF50", bottom=bottom)
    bottom += feasible
    ax.bar(x, infeasible, width, label="Infeasible", color="#FF9800", bottom=bottom)
    bottom += infeasible
    ax.bar(x, crashed, width, label="Crashed", color="#F44336", bottom=bottom)
    bottom += crashed
    ax.bar(x, reeval, width, label="Re-evaluations", color="#9C27B0", bottom=bottom)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Evaluations")
    ax.set_title("Budget Allocation Breakdown")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_budget_breakdown.png")
    plt.close(fig)
    print("  Saved fig2_budget_breakdown.png")


def figure3_flipper_rates(groups: dict, all_stats: dict, output_dir: Path) -> None:
    """Bar chart of false-feasible rate (borderline configs accepted)."""
    plt = setup_matplotlib()
    variants = [v for v in VARIANT_LABELS if v in all_stats]
    labels = [VARIANT_LABELS.get(v, v) for v in variants]

    # For non-ACGF: all borderline configs accepted (no re-eval)
    # For ACGF: caught configs = ones that would have been false-feasible
    rates = []
    for v in variants:
        runs = groups[v]
        total_borderline_feasible = 0
        total_borderline = 0
        for r in runs:
            for o in r["outcomes"]:
                if not o["result_crashed"]:
                    p95 = o.get("result_latency_p95_ms", 0)
                    threshold = 20.0
                    if abs(threshold - p95) <= 3.0:
                        total_borderline += 1
                        if o["result_feasible"]:
                            total_borderline_feasible += 1
        rate = total_borderline_feasible / max(total_borderline, 1)
        rates.append(rate)

    fig, ax = plt.subplots()
    colors = [VARIANT_COLORS.get(v, "#666") for v in variants]
    ax.bar(range(len(variants)), rates, color=colors)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Borderline Acceptance Rate")
    ax.set_title("Flipper Analysis: Borderline Configs Accepted as Feasible")
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_flipper_rates.png")
    plt.close(fig)
    print("  Saved fig3_flipper_rates.png")


def figure4_margin_sweep(all_stats: dict, output_dir: Path) -> None:
    """Fixed margin sweep with ACGF as horizontal reference line."""
    plt = setup_matplotlib()

    margins = [0, 1, 2, 3, 5]
    margin_variants = ["tba", "tba_margin_1", "tba_margin_2", "tba_margin_3", "tba_margin_5"]
    margin_labels = ["0ms", "1ms", "2ms", "3ms", "5ms"]

    accs = []
    wastes = []
    for v in margin_variants:
        if v in all_stats:
            s = all_stats[v]
            accs.append(s["accuracy_mean"] if s["accuracy_mean"] else 0)
            wastes.append(s["waste_rate"])
        else:
            accs.append(0)
            wastes.append(0)

    fig, ax1 = plt.subplots()
    x = np.arange(len(margins))

    ax1.bar(x, accs, 0.5, color="#4CAF50", alpha=0.8, label="Accuracy")
    ax1.set_ylabel("Best Accuracy (mean)", color="#4CAF50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(margin_labels)
    ax1.set_xlabel("Fixed Margin")

    # ACGF reference line
    if "tba_acgf" in all_stats and all_stats["tba_acgf"]["accuracy_mean"]:
        acgf_acc = all_stats["tba_acgf"]["accuracy_mean"]
        ax1.axhline(y=acgf_acc, color="#E91E63", linestyle="--", linewidth=2,
                     label=f"ACGF ({acgf_acc:.4f})")

    ax2 = ax1.twinx()
    ax2.plot(x, wastes, "o-", color="#FF5722", linewidth=2, label="Waste Rate")
    ax2.set_ylabel("Waste Rate", color="#FF5722")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Fixed Margin Sweep: Accuracy vs Waste Rate")
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_margin_sweep.png")
    plt.close(fig)
    print("  Saved fig4_margin_sweep.png")


def figure5_ablations(ablation_dir: Path, output_dir: Path) -> None:
    """2x2 subplot grid for ablation results."""
    plt = setup_matplotlib()

    abl_files = {
        "A": ablation_dir / "ablation_A_fixed_vs_adaptive_k.jsonl",
        "B": ablation_dir / "ablation_B_band_width.jsonl",
        "C": ablation_dir / "ablation_C_confidence_threshold.jsonl",
        "D": ablation_dir / "ablation_D_thermal_proxy.jsonl",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_map = {"A": axes[0, 0], "B": axes[0, 1], "C": axes[1, 0], "D": axes[1, 1]}

    for abl_name, ax in ax_map.items():
        path = abl_files[abl_name]
        if not path.exists():
            ax.text(0.5, 0.5, f"Ablation {abl_name}\n(no data)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Ablation {abl_name}")
            continue

        records = load_jsonl(path)
        groups = group_by_variant(records)
        stats = {v: compute_variant_stats(runs) for v, runs in groups.items()}

        labels = list(stats.keys())
        means = [
            s["accuracy_mean"] if s["accuracy_mean"] else 0
            for s in stats.values()
        ]
        stds = [
            s["accuracy_std"] if s["accuracy_std"] else 0
            for s in stats.values()
        ]

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color="#2196F3", alpha=0.8)
        ax.set_xticks(x)
        short_labels = [l.replace("acgf_", "") for l in labels]
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy (mean ± std)")

        titles = {
            "A": "A: Fixed-K vs Adaptive-K",
            "B": "B: Band Width Sensitivity",
            "C": "C: Confidence Threshold",
            "D": "D: Thermal Conditioning",
        }
        ax.set_title(titles[abl_name])

    fig.tight_layout()
    fig.savefig(output_dir / "fig5_ablations.png")
    plt.close(fig)
    print("  Saved fig5_ablations.png")


# ── Main ──────────────────────────────────────────────────────────────


def parse_main_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze ACGF comparison results")
    p.add_argument(
        "--results", type=str,
        default="results/acgf_comparison/comparison_results.jsonl",
    )
    p.add_argument(
        "--ablation-dir", type=str,
        default="results/acgf_comparison/ablations",
    )
    p.add_argument(
        "--output-dir", type=str,
        default="results/acgf_comparison/figures",
    )
    return p.parse_args()


def main() -> None:
    args = parse_main_args()
    results_path = Path(args.results)
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run run_comparison.py first.")
        sys.exit(1)

    records = load_jsonl(results_path)
    print(f"Loaded {len(records)} run records from {results_path}")

    groups = group_by_variant(records)
    budget = records[0]["budget"] if records else 25

    # ── Compute per-variant stats ──
    all_stats: dict[str, dict] = {}
    for v, runs in groups.items():
        all_stats[v] = compute_variant_stats(runs)

    # ── Table 1: Main Results ──
    print("\n" + "=" * 90)
    print("TABLE 1: MAIN RESULTS")
    print("=" * 90)
    header = (
        f"{'Optimizer':<16} {'Accuracy':>15} {'Waste':>7} "
        f"{'Flippers':>8} {'Configs':>8} {'Budget':>7} "
        f"{'Rescue':>7} {'Caught':>7}"
    )
    print(header)
    print("-" * 90)

    for v in VARIANT_LABELS:
        if v not in all_stats:
            continue
        s = all_stats[v]
        label = VARIANT_LABELS[v]
        if s["accuracy_mean"] is not None:
            acc_str = f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}"
        else:
            acc_str = "N/A"
        print(
            f"{label:<16} {acc_str:>15} {s['waste_rate']:>6.1%} "
            f"{s['n_borderline_mean']:>8.1f} {s['unique_configs_mean']:>8.1f} "
            f"{s['n_reeval_mean']+s['unique_configs_mean']:>7.1f} "
            f"{s['n_rescued_mean']:>7.1f} {s['n_caught_mean']:>7.1f}"
        )
    print()

    # ── Statistical tests ──
    print("=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    test_pairs = [
        ("tba", "tba_acgf", "TBA vs TBA+ACGF"),
        ("tba_margin_3", "tba_acgf", "TBA+3ms vs TBA+ACGF"),
        ("tpe", "tba_acgf", "TPE vs TBA+ACGF"),
    ]

    for va, vb, desc in test_pairs:
        if va not in all_stats or vb not in all_stats:
            continue
        accs_a = all_stats[va]["accuracy_values"]
        accs_b = all_stats[vb]["accuracy_values"]
        n = min(len(accs_a), len(accs_b))
        if n < 3:
            print(f"  {desc}: insufficient paired data (n={n})")
            continue
        try:
            result = paired_test(accs_a[:n], accs_b[:n])
            sig = "***" if result["wilcoxon_p"] < 0.001 else (
                "**" if result["wilcoxon_p"] < 0.01 else (
                    "*" if result["wilcoxon_p"] < 0.05 else "ns"
                )
            )
            print(
                f"  {desc}: diff={result['mean_diff']:+.4f}  "
                f"d={result['cohens_d']:.3f}  "
                f"p={result['wilcoxon_p']:.4f} {sig}"
            )
        except Exception as e:
            print(f"  {desc}: test failed ({e})")
    print()

    # ── Figures ──
    print("Generating figures...")
    figure1_convergence(groups, budget, output_dir)
    figure2_budget_breakdown(groups, all_stats, output_dir)
    figure3_flipper_rates(groups, all_stats, output_dir)
    figure4_margin_sweep(all_stats, output_dir)
    if ablation_dir.exists():
        figure5_ablations(ablation_dir, output_dir)
    else:
        print("  Skipping fig5 (no ablation data)")

    # ── Save summary JSON ──
    summary = {v: {k: val for k, val in s.items() if k != "accuracy_values"}
               for v, s in all_stats.items()}
    summary_path = output_dir / "summary_stats.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {summary_path}")

    # ── Verdict ──
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    tba_acc = all_stats.get("tba", {}).get("accuracy_mean")
    acgf_acc = all_stats.get("tba_acgf", {}).get("accuracy_mean")

    if tba_acc is not None and acgf_acc is not None:
        diff = acgf_acc - tba_acc
        direction = "BETTER" if diff > 0 else "WORSE" if diff < 0 else "TIED"
        print(f"  TBA+ACGF vs TBA: {direction} by {abs(diff):.4f} accuracy")

        # Best margin
        best_margin_v, best_margin_acc = None, -1.0
        for mv in ["tba_margin_1", "tba_margin_2", "tba_margin_3", "tba_margin_5"]:
            if mv in all_stats and all_stats[mv]["accuracy_mean"] is not None:
                if all_stats[mv]["accuracy_mean"] > best_margin_acc:
                    best_margin_acc = all_stats[mv]["accuracy_mean"]
                    best_margin_v = mv

        if best_margin_v:
            margin_diff = acgf_acc - best_margin_acc
            margin_dir = "BETTER" if margin_diff > 0 else "WORSE"
            print(
                f"  TBA+ACGF vs best margin ({VARIANT_LABELS[best_margin_v]}): "
                f"{margin_dir} by {abs(margin_diff):.4f}"
            )

        acgf_configs = all_stats["tba_acgf"]["unique_configs_mean"]
        tba_configs = all_stats["tba"]["unique_configs_mean"]
        print(
            f"  Config exploration: ACGF={acgf_configs:.1f} vs TBA={tba_configs:.1f} "
            f"({acgf_configs - tba_configs:+.1f} unique configs)"
        )

        rescued = all_stats["tba_acgf"]["n_rescued_mean"]
        caught = all_stats["tba_acgf"]["n_caught_mean"]
        print(f"  ACGF rescued {rescued:.1f} / caught {caught:.1f} borderline configs (mean)")
    else:
        print("  Insufficient data for verdict.")

    print("=" * 60)


if __name__ == "__main__":
    main()
