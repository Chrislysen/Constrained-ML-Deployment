"""Generate publication-quality figures from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "tba_tpe": "#e74c3c",
    "tba_nosurr": "#c0392b",
    "random": "#7f8c8d",
    "tpe": "#27ae60",
    "bo": "#2980b9",
}
LABELS = {
    "tba_tpe": "TBA-TPE Hybrid (Ours)",
    "tba_nosurr": "TBA (pure SA)",
    "random": "Random Search",
    "tpe": "Optuna TPE",
    "bo": "Constrained BO",
}
LINESTYLES = {
    "tba_tpe": "-",
    "tba_nosurr": "--",
    "random": ":",
    "tpe": "-",
    "bo": "-.",
}
MARKERS = {
    "tba_tpe": "o",
    "tba_nosurr": "s",
    "random": "^",
    "tpe": "D",
    "bo": "v",
}


# =====================================================================
# Figure 1: Budget Sweep — Objective vs Budget
# =====================================================================
def fig_budget_sweep_objective():
    path = Path("results_synthetic/sweep_summary.json")
    if not path.exists():
        print(f"  Skipping budget sweep: {path} not found")
        return
    with open(path) as f:
        data = json.load(f)

    budgets = [10, 15, 20, 25, 30, 40, 50]
    opt_names = ["tba_tpe", "tba_nosurr", "random", "tpe", "bo"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for opt in opt_names:
        if opt not in data:
            continue
        means, stds = [], []
        for b in budgets:
            sums = data[opt].get(str(b), [])
            objs = [s["best_feasible_obj"] for s in sums if s["best_feasible_obj"] is not None]
            means.append(np.mean(objs) if objs else np.nan)
            stds.append(np.std(objs) if objs else 0)
        means, stds = np.array(means), np.array(stds)
        ax.plot(budgets, means, label=LABELS[opt], color=COLORS[opt],
                linestyle=LINESTYLES[opt], linewidth=2.5, marker=MARKERS[opt], markersize=7)
        ax.fill_between(budgets, means - stds, means + stds, alpha=0.1, color=COLORS[opt])

    ax.set_xlabel("Trial Budget")
    ax.set_ylabel("Best Feasible Objective (higher is better)")
    ax.set_title("Budget Sweep on Crashy Branin Benchmark")
    ax.set_xticks(budgets)
    ax.legend(loc="lower right")
    fig.savefig(FIGURES_DIR / "fig1_budget_sweep_objective.png")
    plt.close(fig)
    print(f"  Saved fig1_budget_sweep_objective.png")


# =====================================================================
# Figure 2: Budget Sweep — Wasted Budget vs Budget
# =====================================================================
def fig_budget_sweep_wasted():
    path = Path("results_synthetic/sweep_summary.json")
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)

    budgets = [10, 15, 20, 25, 30, 40, 50]
    opt_names = ["tba_tpe", "tba_nosurr", "random", "tpe", "bo"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for opt in opt_names:
        if opt not in data:
            continue
        wasted_means = []
        for b in budgets:
            sums = data[opt].get(str(b), [])
            fracs = [s["wasted_fraction"] for s in sums]
            wasted_means.append(np.mean(fracs) if fracs else np.nan)
        ax.plot(budgets, wasted_means, label=LABELS[opt], color=COLORS[opt],
                linestyle=LINESTYLES[opt], linewidth=2.5, marker=MARKERS[opt], markersize=7)

    ax.set_xlabel("Trial Budget")
    ax.set_ylabel("Fraction of Budget Wasted (crashes + infeasible)")
    ax.set_title("Budget Efficiency — Crashy Branin Benchmark")
    ax.set_xticks(budgets)
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper right")
    fig.savefig(FIGURES_DIR / "fig2_budget_sweep_wasted.png")
    plt.close(fig)
    print(f"  Saved fig2_budget_sweep_wasted.png")


# =====================================================================
# Figure 3: Deployment Results — Edge Tight (bar chart)
# =====================================================================
def fig_deployment_edge_tight():
    path = Path("results_deployment/edge_tight/summary.json")
    if not path.exists():
        print(f"  Skipping deployment: {path} not found")
        return
    with open(path) as f:
        data = json.load(f)

    opt_names = ["tba_tpe", "tba_sa", "random", "tpe"]
    deploy_labels = {
        "tba_tpe": "TBA-TPE\nHybrid",
        "tba_sa": "TBA\n(pure SA)",
        "random": "Random\nSearch",
        "tpe": "Optuna\nTPE",
    }
    deploy_colors = {
        "tba_tpe": "#e74c3c",
        "tba_sa": "#c0392b",
        "random": "#7f8c8d",
        "tpe": "#27ae60",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Accuracy
    ax = axes[0]
    means, stds, labels_list, colors = [], [], [], []
    for opt in opt_names:
        sums = data["summaries"][opt]
        objs = [s["best_feasible_obj"] for s in sums if s["best_feasible_obj"] is not None]
        means.append(np.mean(objs))
        stds.append(np.std(objs))
        labels_list.append(deploy_labels[opt])
        colors.append(deploy_colors[opt])

    x = np.arange(len(opt_names))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, width=0.6)
    ax.set_ylabel("Best Feasible Accuracy")
    ax.set_title("Edge Deployment (p95 < 20ms, mem < 512MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list)
    ax.set_ylim(0.72, 0.78)

    # Right: Wasted budget
    ax = axes[1]
    waste_means = []
    for opt in opt_names:
        sums = data["summaries"][opt]
        waste_means.append(np.mean([s["wasted_fraction"] for s in sums]))

    bars = ax.bar(x, waste_means, color=colors, alpha=0.85, width=0.6)
    ax.set_ylabel("Fraction of Budget Wasted")
    ax.set_title("Budget Efficiency — Edge Deployment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_deployment_edge_tight.png")
    plt.close(fig)
    print(f"  Saved fig3_deployment_edge_tight.png")


# =====================================================================
# Figure 4: Deployment Results — Server Throughput
# =====================================================================
def fig_deployment_server():
    path = Path("results_deployment/server_throughput/summary.json")
    if not path.exists():
        print(f"  Skipping server throughput: {path} not found")
        return
    with open(path) as f:
        data = json.load(f)

    opt_names = ["tba_tpe", "tba_sa", "random", "tpe"]
    deploy_labels = {
        "tba_tpe": "TBA-TPE\nHybrid",
        "tba_sa": "TBA\n(pure SA)",
        "random": "Random\nSearch",
        "tpe": "Optuna\nTPE",
    }
    deploy_colors = {
        "tba_tpe": "#e74c3c",
        "tba_sa": "#c0392b",
        "random": "#7f8c8d",
        "tpe": "#27ae60",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Throughput
    ax = axes[0]
    means, stds, labels_list, colors = [], [], [], []
    for opt in opt_names:
        sums = data["summaries"][opt]
        objs = [s["best_feasible_obj"] for s in sums if s["best_feasible_obj"] is not None]
        means.append(np.mean(objs))
        stds.append(np.std(objs))
        labels_list.append(deploy_labels[opt])
        colors.append(deploy_colors[opt])

    x = np.arange(len(opt_names))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, width=0.6)
    ax.set_ylabel("Best Feasible Throughput (QPS)")
    ax.set_title("Server Deployment (p99 < 100ms, mem < 4GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list)

    # Right: Wasted budget
    ax = axes[1]
    waste_means = []
    for opt in opt_names:
        sums = data["summaries"][opt]
        waste_means.append(np.mean([s["wasted_fraction"] for s in sums]))

    bars = ax.bar(x, waste_means, color=colors, alpha=0.85, width=0.6)
    ax.set_ylabel("Fraction of Budget Wasted")
    ax.set_title("Budget Efficiency — Server Deployment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_deployment_server_throughput.png")
    plt.close(fig)
    print(f"  Saved fig4_deployment_server_throughput.png")


# =====================================================================
# Figure 5: Config convergence heatmap (edge_tight)
# =====================================================================
def fig_config_convergence():
    path = Path("results_deployment/edge_tight/summary.json")
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)

    opt_names = ["tba_tpe", "tba_sa", "random", "tpe"]
    opt_labels = ["TBA-TPE Hybrid", "TBA (pure SA)", "Random Search", "Optuna TPE"]
    model_names = ["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0", "vit_tiny"]

    # Count how many seeds each optimizer converged to each model
    counts = np.zeros((len(opt_names), len(model_names)))
    for i, opt in enumerate(opt_names):
        configs = data["best_configs"][opt]
        for cfg in configs:
            if cfg and cfg.get("model_name") in model_names:
                j = model_names.index(cfg["model_name"])
                counts[i, j] += 1

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(counts, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_yticks(range(len(opt_names)))
    ax.set_yticklabels(opt_labels)
    ax.set_title("Config Convergence: Model Selected per Optimizer (10 seeds, edge_tight)")

    # Annotate cells
    for i in range(len(opt_names)):
        for j in range(len(model_names)):
            val = int(counts[i, j])
            color = "white" if val > 5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Number of seeds (out of 10)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_config_convergence.png")
    plt.close(fig)
    print(f"  Saved fig5_config_convergence.png")


if __name__ == "__main__":
    print("Generating publication figures...")
    fig_budget_sweep_objective()
    fig_budget_sweep_wasted()
    fig_deployment_edge_tight()
    fig_deployment_server()
    fig_config_convergence()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
