#!/usr/bin/env python3
"""Analyze boundary probe results: produce figures and go/no-go verdict.

Reads probe JSONL output and generates:
  1. Feasibility probability vs distance to threshold
  2. Same, grouped by model family
  3. Flip rate heatmap
  4. Cold vs warm comparison
  5. GPU temperature vs latency
  + Summary statistics JSON
  + Go/no-go assessment
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


LATENCY_THRESHOLD_MS = 20.0

# Zone classification (same as selector)
ZONES = {
    "DEEP_FEASIBLE":   (0.0,  15.0),
    "NEAR_FEASIBLE":   (15.0, 19.0),
    "GRAY_ZONE":       (19.0, 22.0),
    "NEAR_INFEASIBLE": (22.0, 26.0),
    "DEEP_INFEASIBLE": (26.0, float("inf")),
}

ZONE_COLORS = {
    "DEEP_FEASIBLE":   "#2ecc71",
    "NEAR_FEASIBLE":   "#82e0aa",
    "GRAY_ZONE":       "#f39c12",
    "NEAR_INFEASIBLE": "#e74c3c",
    "DEEP_INFEASIBLE": "#922b21",
}

MODEL_MARKERS = {
    "resnet18":       "o",
    "resnet50":       "s",
    "mobilenet_v2":   "^",
    "efficientnet_b0": "D",
    "vit_tiny":       "P",
}

MODEL_COLORS = {
    "resnet18":       "#3498db",
    "resnet50":       "#2ecc71",
    "mobilenet_v2":   "#e74c3c",
    "efficientnet_b0": "#9b59b6",
    "vit_tiny":       "#f39c12",
}


def load_probe_results(result_paths: list[Path]) -> list[dict]:
    """Load all probe result records from JSONL files."""
    records = []
    for path in result_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def group_by_config(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by config key."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        cfg = rec["config"]
        key = f"{cfg['model']}|{cfg['backend']}|{cfg['quantization']}|{cfg['batch_size']}"
        groups[key].append(rec)
    return groups


def classify_zone(latency_ms: float) -> str:
    """Classify latency into a zone."""
    for zone_name, (lo, hi) in ZONES.items():
        if lo <= latency_ms < hi:
            return zone_name
    return "DEEP_INFEASIBLE"


def compute_config_stats(groups: dict[str, list[dict]]) -> list[dict]:
    """Compute per-config statistics from grouped probe results."""
    config_stats = []
    for key, records in groups.items():
        model, backend, quant, batch_size = key.split("|")

        # Filter non-crashed records with valid summaries
        valid = [r for r in records if not r["crash"] and r["summary"]]
        if not valid:
            continue

        p95_values = [r["summary"]["p95"] for r in valid]
        feasible_count = sum(1 for r in valid if r["feasible"])
        total = len(valid)

        cold_records = [r for r in valid if r["condition"] == "cold"]
        warm_records = [r for r in valid if r["condition"] == "warm"]

        cold_p95 = [r["summary"]["p95"] for r in cold_records] if cold_records else []
        warm_p95 = [r["summary"]["p95"] for r in warm_records] if warm_records else []

        cold_feasible = sum(1 for r in cold_records if r["feasible"])
        warm_feasible = sum(1 for r in warm_records if r["feasible"])

        median_p95 = float(np.median(p95_values))
        distance = median_p95 - LATENCY_THRESHOLD_MS
        feasibility_rate = feasible_count / total if total > 0 else 0.0
        flip_rate = 1.0 - max(feasible_count, total - feasible_count) / total if total > 0 else 0.0

        zone = classify_zone(median_p95)

        # GPU temperatures
        temps = [
            r["gpu_state"]["temperature_c"]
            for r in valid
            if r["gpu_state"].get("temperature_c") is not None
        ]

        stat = {
            "key": key,
            "model": model,
            "backend": backend,
            "quantization": quant,
            "batch_size": int(batch_size),
            "zone": zone,
            "median_p95_ms": median_p95,
            "mean_p95_ms": float(np.mean(p95_values)),
            "std_p95_ms": float(np.std(p95_values)),
            "min_p95_ms": float(np.min(p95_values)),
            "max_p95_ms": float(np.max(p95_values)),
            "distance_to_threshold_ms": distance,
            "feasibility_rate": feasibility_rate,
            "flip_rate": flip_rate,
            "total_repeats": total,
            "feasible_count": feasible_count,
            "cold_p95_values": cold_p95,
            "warm_p95_values": warm_p95,
            "cold_feasible": cold_feasible,
            "cold_total": len(cold_records),
            "warm_feasible": warm_feasible,
            "warm_total": len(warm_records),
            "temperatures": temps,
            "p95_values": p95_values,
        }
        config_stats.append(stat)

    return config_stats


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure1_feasibility_vs_distance(config_stats: list[dict], output_dir: Path):
    """Figure 1: Feasibility probability vs distance to threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for stat in config_stats:
        color = ZONE_COLORS.get(stat["zone"], "#999999")
        ax.scatter(
            stat["distance_to_threshold_ms"],
            stat["feasibility_rate"],
            c=color,
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )

    # Reference lines
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="20ms threshold")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)

    # Zone bands
    ax.axvspan(-5, -1, alpha=0.05, color="green", label="Near feasible")
    ax.axvspan(-1, 2, alpha=0.1, color="orange", label="Gray zone")
    ax.axvspan(2, 6, alpha=0.05, color="red", label="Near infeasible")

    ax.set_xlabel("Distance to 20ms threshold (ms)\n(negative = under threshold)", fontsize=12)
    ax.set_ylabel("Feasibility rate (fraction of repeats with p95 ≤ 20ms)", fontsize=12)
    ax.set_title("Stochastic Feasibility Boundary:\nFeasibility Probability vs Distance to Threshold", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add zone color legend
    from matplotlib.patches import Patch
    zone_patches = [
        Patch(facecolor=ZONE_COLORS[z], label=z.replace("_", " ").title())
        for z in ZONES
    ]
    legend2 = ax.legend(handles=zone_patches, loc="center right", fontsize=8, title="Zone")
    ax.add_artist(legend2)

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_feasibility_vs_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_feasibility_vs_distance.png")


def figure2_by_model_family(config_stats: list[dict], output_dir: Path):
    """Figure 2: Feasibility vs distance, grouped by model family."""
    models = sorted(set(s["model"] for s in config_stats))
    n_models = len(models)

    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        model_stats = [s for s in config_stats if s["model"] == model_name]

        for stat in model_stats:
            color = ZONE_COLORS.get(stat["zone"], "#999999")
            ax.scatter(
                stat["distance_to_threshold_ms"],
                stat["feasibility_rate"],
                c=color,
                s=50,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Distance to threshold (ms)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Feasibility rate", fontsize=10)

    fig.suptitle("Feasibility Transition by Model Family", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_by_model_family.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_by_model_family.png")


def figure3_flip_rate_heatmap(config_stats: list[dict], output_dir: Path):
    """Figure 3: Flip rate heatmap (model x backend vs quant x batch_size)."""
    # Build row/col labels
    row_labels = sorted(set(f"{s['model']}|{s['backend']}" for s in config_stats))
    col_labels = sorted(set(f"{s['quantization']}|bs{s['batch_size']}" for s in config_stats))

    if not row_labels or not col_labels:
        return

    data = np.full((len(row_labels), len(col_labels)), np.nan)
    annot = [[" " for _ in col_labels] for _ in row_labels]

    row_map = {r: i for i, r in enumerate(row_labels)}
    col_map = {c: i for i, c in enumerate(col_labels)}

    for stat in config_stats:
        row_key = f"{stat['model']}|{stat['backend']}"
        col_key = f"{stat['quantization']}|bs{stat['batch_size']}"
        if row_key in row_map and col_key in col_map:
            ri = row_map[row_key]
            ci = col_map[col_key]
            data[ri, ci] = stat["flip_rate"]
            annot[ri][ci] = f"{stat['flip_rate']:.0%}" if stat["flip_rate"] > 0 else "0%"

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.2), max(6, len(row_labels) * 0.5)))

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)

    # Labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([c.replace("|", "\n") for c in col_labels], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([r.replace("|", " / ") for r in row_labels], fontsize=7)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=6,
                        color="white" if data[i, j] > 0.25 else "black")

    ax.set_title("Flip Rate Heatmap\n(fraction of repeats disagreeing with majority feasibility)", fontsize=11)
    fig.colorbar(im, ax=ax, label="Flip rate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_flip_rate_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_flip_rate_heatmap.png")


def figure4_cold_vs_warm(config_stats: list[dict], output_dir: Path):
    """Figure 4: Cold vs warm p95 latency distributions for gray-zone configs."""
    # Select gray zone and near-boundary configs
    gray_stats = [
        s for s in config_stats
        if s["zone"] in ("GRAY_ZONE", "NEAR_FEASIBLE", "NEAR_INFEASIBLE")
        and s["cold_p95_values"] and s["warm_p95_values"]
    ]

    if not gray_stats:
        print("  Skipping fig4: no gray-zone configs with cold/warm data")
        return

    # Limit to most interesting (highest flip rate)
    gray_stats.sort(key=lambda s: s["flip_rate"], reverse=True)
    gray_stats = gray_stats[:12]

    n = len(gray_stats)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for idx, stat in enumerate(gray_stats):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        positions = [0, 1]
        data_to_plot = [stat["cold_p95_values"], stat["warm_p95_values"]]
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)

        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][1].set_facecolor("#e74c3c")

        ax.axhline(LATENCY_THRESHOLD_MS, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(["Cold", "Warm"], fontsize=9)

        label = f"{stat['model']}\n{stat['backend']}/{stat['quantization']}\nbs{stat['batch_size']}"
        ax.set_title(label, fontsize=8)
        ax.set_ylabel("p95 latency (ms)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Cold vs Warm p95 Latency (Near-Boundary Configs)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_cold_vs_warm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_cold_vs_warm.png")


def figure5_temp_vs_latency(config_stats: list[dict], output_dir: Path):
    """Figure 5: GPU temperature vs p95 latency."""
    fig, ax = plt.subplots(figsize=(10, 6))

    all_temps = []
    all_latencies = []

    for stat in config_stats:
        temps = stat["temperatures"]
        p95s = stat["p95_values"]

        if len(temps) != len(p95s):
            # Align by taking min length
            n = min(len(temps), len(p95s))
            temps = temps[:n]
            p95s = p95s[:n]

        if not temps:
            continue

        color = MODEL_COLORS.get(stat["model"], "#999999")
        marker = MODEL_MARKERS.get(stat["model"], "o")

        ax.scatter(
            temps, p95s,
            c=color, marker=marker, s=20, alpha=0.4,
            label=stat["model"],
        )

        all_temps.extend(temps)
        all_latencies.extend(p95s)

    if all_temps and all_latencies:
        # Compute correlation
        r, p = stats.pearsonr(all_temps, all_latencies)
        ax.text(
            0.02, 0.98,
            f"Pearson r = {r:.3f}, p = {p:.2e}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.axhline(LATENCY_THRESHOLD_MS, color="red", linestyle="--", alpha=0.5, label="20ms threshold")
    ax.set_xlabel("GPU Temperature (C)", fontsize=12)
    ax.set_ylabel("p95 Latency (ms)", fontsize=12)
    ax.set_title("GPU Temperature vs Observed p95 Latency", fontsize=13)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fig5_temp_vs_latency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_temp_vs_latency.png")


# ---------------------------------------------------------------------------
# Summary statistics and go/no-go
# ---------------------------------------------------------------------------

def compute_summary_stats(
    config_stats: list[dict], records: list[dict]
) -> dict:
    """Compute all summary statistics for the go/no-go assessment."""
    total_configs = len(config_stats)
    total_repeats = sum(s["total_repeats"] for s in config_stats)

    any_instability = sum(1 for s in config_stats if s["flip_rate"] > 0)
    meaningful_instability = sum(1 for s in config_stats if s["flip_rate"] > 0.10)
    strong_instability = sum(1 for s in config_stats if s["flip_rate"] > 0.30)

    # Gray zone width: range of distance where feasibility rate is 10%-90%
    transition_stats = [
        s for s in config_stats
        if 0.10 <= s["feasibility_rate"] <= 0.90
    ]
    if transition_stats:
        distances = [s["distance_to_threshold_ms"] for s in transition_stats]
        gray_zone_width = max(distances) - min(distances)
        gray_zone_min = min(distances)
        gray_zone_max = max(distances)
    else:
        gray_zone_width = 0.0
        gray_zone_min = 0.0
        gray_zone_max = 0.0

    # Per-model gray zone width
    per_model_width = {}
    models = sorted(set(s["model"] for s in config_stats))
    for model_name in models:
        model_transition = [
            s for s in transition_stats if s["model"] == model_name
        ]
        if model_transition:
            dists = [s["distance_to_threshold_ms"] for s in model_transition]
            per_model_width[model_name] = round(max(dists) - min(dists), 3)
        else:
            per_model_width[model_name] = 0.0

    # Temperature correlation
    all_temps = []
    all_lats = []
    for stat in config_stats:
        n = min(len(stat["temperatures"]), len(stat["p95_values"]))
        all_temps.extend(stat["temperatures"][:n])
        all_lats.extend(stat["p95_values"][:n])

    temp_corr_r = 0.0
    temp_corr_p = 1.0
    if len(all_temps) > 2:
        temp_corr_r, temp_corr_p = stats.pearsonr(all_temps, all_lats)

    # Cold vs warm
    cold_lats = []
    warm_lats = []
    cold_flips = 0
    warm_flips = 0
    cold_total = 0
    warm_total = 0
    for s in config_stats:
        cold_lats.extend(s["cold_p95_values"])
        warm_lats.extend(s["warm_p95_values"])
        cold_total += s["cold_total"]
        warm_total += s["warm_total"]
        cold_flips += s["cold_total"] - s["cold_feasible"] if s["feasibility_rate"] >= 0.5 else s["cold_feasible"]
        warm_flips += s["warm_total"] - s["warm_feasible"] if s["feasibility_rate"] >= 0.5 else s["warm_feasible"]

    cold_mean = float(np.mean(cold_lats)) if cold_lats else 0.0
    warm_mean = float(np.mean(warm_lats)) if warm_lats else 0.0
    cold_flip_rate = cold_flips / cold_total if cold_total > 0 else 0.0
    warm_flip_rate = warm_flips / warm_total if warm_total > 0 else 0.0

    # Near-threshold analysis
    near_threshold = [
        s for s in config_stats
        if abs(s["distance_to_threshold_ms"]) <= 5.0
    ]
    near_with_flips = sum(1 for s in near_threshold if s["flip_rate"] > 0.10)
    near_fraction_flipping = near_with_flips / len(near_threshold) if near_threshold else 0.0

    # Heterogeneity: do different models have different transition widths?
    nonzero_widths = {m: w for m, w in per_model_width.items() if w > 0}
    heterogeneity = len(set(nonzero_widths.values())) > 1 if len(nonzero_widths) > 1 else False

    return {
        "total_configs": total_configs,
        "total_repeats": total_repeats,
        "configs_any_instability": any_instability,
        "configs_meaningful_instability_gt10pct": meaningful_instability,
        "configs_strong_instability_gt30pct": strong_instability,
        "gray_zone_width_ms": round(gray_zone_width, 3),
        "gray_zone_range_ms": [round(gray_zone_min, 3), round(gray_zone_max, 3)],
        "per_model_gray_zone_width_ms": per_model_width,
        "temp_correlation_pearson_r": round(temp_corr_r, 4),
        "temp_correlation_p_value": round(temp_corr_p, 6),
        "cold_mean_p95_ms": round(cold_mean, 3),
        "warm_mean_p95_ms": round(warm_mean, 3),
        "cold_warm_diff_ms": round(cold_mean - warm_mean, 3),
        "cold_flip_rate": round(cold_flip_rate, 4),
        "warm_flip_rate": round(warm_flip_rate, 4),
        "near_threshold_configs": len(near_threshold),
        "near_threshold_with_flips_gt10pct": near_with_flips,
        "near_threshold_fraction_flipping": round(near_fraction_flipping, 4),
        "heterogeneity_detected": heterogeneity,
    }


def go_no_go(summary: dict) -> str:
    """Produce the go/no-go verdict."""
    lines = ["\n" + "=" * 70, "GO / NO-GO ASSESSMENT", "=" * 70]

    near_frac = summary["near_threshold_fraction_flipping"]
    meaningful = summary["configs_meaningful_instability_gt10pct"]
    strong = summary["configs_strong_instability_gt30pct"]
    heterogeneity = summary["heterogeneity_detected"]
    total = summary["total_configs"]

    lines.append(f"\nConfigs probed: {total}")
    lines.append(f"Configs with any instability: {summary['configs_any_instability']}")
    lines.append(f"Configs with >10% flip rate: {meaningful}")
    lines.append(f"Configs with >30% flip rate: {strong}")
    lines.append(f"Gray zone width: {summary['gray_zone_width_ms']:.1f}ms")
    lines.append(f"Near-threshold configs flipping >10%: {near_frac:.0%}")
    lines.append(f"Cold-warm latency diff: {summary['cold_warm_diff_ms']:.2f}ms")
    lines.append(f"Temp-latency correlation: r={summary['temp_correlation_pearson_r']:.3f}")
    lines.append(f"Heterogeneity across models: {'YES' if heterogeneity else 'NO'}")

    verdict_lines = []
    if near_frac > 0.10:
        verdict_lines.append(
            "GO -- Stochastic boundary is wide enough for a paper. "
            f"{near_frac:.0%} of near-threshold configs show flip rates >10%."
        )
    elif meaningful > 0 and near_frac <= 0.10:
        verdict_lines.append(
            "MARGINAL -- Boundary is real but narrow. "
            f"Only {near_frac:.0%} of near-threshold configs flip significantly. "
            "Consider simpler framing."
        )
    else:
        verdict_lines.append(
            "NO-GO -- Boundary is deterministic. "
            "No configs show meaningful feasibility flips. "
            "Pivot to standard crash prediction."
        )

    if heterogeneity:
        verdict_lines.append(
            "HETEROGENEITY CONFIRMED -- Different model families have different "
            "transition widths. Uncertainty modeling is justified over fixed margins."
        )

    lines.append("")
    for v in verdict_lines:
        lines.append(f">>> {v}")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze boundary probe results and produce figures + go/no-go verdict"
    )
    parser.add_argument(
        "--result-files",
        nargs="+",
        type=Path,
        required=True,
        help="JSONL probe result files from probe_runner.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/boundary_probe/figures"),
        help="Output directory for figures and stats (default: results/boundary_probe/figures/)",
    )
    args = parser.parse_args()

    # Validate inputs
    for f in args.result_files:
        if not f.exists():
            print(f"ERROR: Result file not found: {f}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading {len(args.result_files)} result files...")
    records = load_probe_results(args.result_files)
    print(f"Loaded {len(records)} records")

    groups = group_by_config(records)
    print(f"Unique configs: {len(groups)}")

    config_stats = compute_config_stats(groups)
    print(f"Configs with valid data: {len(config_stats)}")

    if not config_stats:
        print("ERROR: No valid config stats to analyze", file=sys.stderr)
        sys.exit(1)

    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\nGenerating figures...")
    figure1_feasibility_vs_distance(config_stats, args.output_dir)
    figure2_by_model_family(config_stats, args.output_dir)
    figure3_flip_rate_heatmap(config_stats, args.output_dir)
    figure4_cold_vs_warm(config_stats, args.output_dir)
    figure5_temp_vs_latency(config_stats, args.output_dir)

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary = compute_summary_stats(config_stats, records)

    # Save stats
    stats_file = args.output_dir.parent / "summary_stats.json"
    with open(stats_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {stats_file}")

    # Go/no-go
    verdict = go_no_go(summary)
    print(verdict)


if __name__ == "__main__":
    main()
