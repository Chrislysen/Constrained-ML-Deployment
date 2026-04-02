# Thermal Budget Annealing (TBA)

**Crash-aware deployment optimization for ML models under hard constraints and tiny budgets.**

![Edge Deployment Results](figures/fig3_deployment_edge_tight.png)

Standard optimizers (TPE, Bayesian optimization, random search) waste 40-80% of their trial budget on configurations that crash, OOM, or violate deployment constraints. TBA-TPE Hybrid uses a two-phase strategy: Phase 1 runs crash-aware simulated annealing to map the feasible region in 5-15 trials, then Phase 2 warm-starts Optuna's TPE with all crash data so it optimizes within the safe region. On real ML deployment benchmarks (5 models x 3 backends x 3 quantization modes x 6 batch sizes), TBA-TPE finds the best feasible configuration in 25 trials while wasting 41% of budget, compared to TPE's 35% waste but lower accuracy (0.754 vs 0.761) because TPE gets stuck on suboptimal models it discovered early.

## Key Results

| Scenario | TBA-TPE Hybrid | Optuna TPE | Random Search | TBA (pure SA) |
|----------|---------------|------------|---------------|---------------|
| **Edge (acc)** | **0.761 +- 0.010** | 0.754 +- 0.010 | 0.762 +- 0.008 | 0.755 +- 0.015 |
| Edge (waste) | 41% | 35% | **77%** | **33%** |
| **Server (QPS)** | 5087 +- 1787 | **5423 +- 1325** | 4758 +- 1399 | 4131 +- 2036 |
| Server (waste) | 23% | 24% | 54% | **18%** |

![Budget Efficiency](figures/fig2_budget_sweep_wasted.png)

![Config Convergence](figures/fig5_config_convergence.png)

## Install

```bash
pip install -e ".[gpu]"
```

Requires Python >= 3.10. GPU dependencies (torch, torchvision, onnxruntime) are in the `[gpu]` extra.

## Quick Start

```python
from tba import optimize_deployment

best_config, result = optimize_deployment(
    data_dir="data/imagenette2-320",
    constraints={"latency_p95_ms": 20, "memory_peak_mb": 512},
    budget=25,
)
print(f"Best: {best_config['model_name']}, accuracy={result.objective_value:.3f}")
```

## How It Works

```
Phase 1: TBA Feasible-First SA (5-15 trials)
  - Adaptive simulated annealing explores the crash-heavy space
  - Finds feasible configs fast, maps crash zones
  - Hands off when 5+ feasible AND 3+ crashed/infeasible AND 3+ model families explored

Phase 2: Optuna TPE Warm-Started (remaining budget)
  - ALL Phase 1 history injected into Optuna study
  - TPE starts with a model trained on crash data
  - Optimizes within the feasible region
```

## Project Structure

```
tba/
  api.py                    # optimize_deployment() one-liner
  types.py                  # EvalResult, VariableDef dataclasses
  optimizer/
    tba_tpe_hybrid.py       # Primary optimizer (TBA -> TPE handoff)
    tba_optimizer.py        # TBA pure SA (ablation baseline)
    search_space.py         # Hierarchical mixed-variable search space
    subspace_tracker.py     # Subspace + combination blacklisting (v2/v3)
    surrogate.py            # RF surrogate (OOB-gated)
    feasible_tpe.py         # KDE-based feasible-region sampler
    base.py                 # BaseOptimizer ask/tell interface
  baselines/
    random_search.py        # Random search baseline
    optuna_tpe.py           # Optuna TPE baseline
    constrained_bo.py       # Constrained BO baseline (sklearn GP)
  benchmarks/
    synthetic_bench.py      # Crashy Branin + Hierarchical Rosenbrock
    profiler.py             # Real deployment profiler (latency/memory/accuracy)
    model_zoo.py            # Pre-trained ImageNet models
  spaces/
    image_classification.py # Deployment search space + scenario configs
experiments/
  run_synthetic.py          # Synthetic benchmark runner
  run_deployment.py         # Real deployment benchmark runner
  run_budget_sweep.py       # Budget sweep experiment
  generate_figures.py       # Publication figure generator
  test_v2_features.py       # v2 feature tests (subspace blacklisting, early stop)
  test_v3_features.py       # v3 feature tests (diversity, adaptive timeout, combos)
paper/
  README.md                 # Paper PDF placeholder
```

## Running Experiments

```bash
# Synthetic benchmarks (fast, no GPU needed)
PYTHONPATH=. python experiments/run_synthetic.py --benchmark crashy_branin --budget 30 --seeds 10

# Budget sweep
PYTHONPATH=. python experiments/run_budget_sweep.py

# Real deployment (needs GPU + ImageNette)
PYTHONPATH=. python experiments/run_deployment.py --scenarios edge_tight server_throughput --seeds 10

# Generate figures
PYTHONPATH=. python experiments/generate_figures.py
```

## Tested Platforms

| GPU | Environment | Notes |
|-----|-------------|-------|
| NVIDIA H100 | Google Colab | |
| NVIDIA A100 | Google Colab | |
| NVIDIA RTX 5080 | Windows, CUDA 12.8 | |
| NVIDIA L4 | Google Colab | |
| NVIDIA T4 | Google Colab | |
| Apple M1 Max | macOS, CPU-only | 64 GB RAM |

## Version History

### v1 (March 2026)
- Core TBA optimizer with feasible-first SA exploration
- Adaptive temperature control, elite restart, decaying snap-back
- TPE warm-start handoff
- DeployBench benchmark suite
- Experiments on RTX 5080 (10 seeds)
- **Paper:** [v1 PDF](paper/tba_paper_v1.pdf) — single-GPU results

### v2 (April 2026)
- Trial timeouts: abort evaluations when latency > 5x constraint after warmup
- Subspace blacklisting: temporarily suppress categorical values after 3 consecutive failures
- Multi-GPU experiments: H100, A100, RTX 5080, L4, T4
- 31/31 tests passing
- **Paper:** [v2 PDF](paper/tba_paper_v2.pdf) — multi-GPU results, softened claims

### v3 (April 2026)
- Diversity-gated handoff: Phase 1 must explore >=3 model families before handing off to TPE
- Adaptive timeout threshold: tightens from 5x to 3x after first feasible solution found
- Combination-level blacklisting: tracks 2-way categorical failure patterns instead of just individual values
- All existing tests passing + new v3 tests (46 total)

## Paper

**Feasible-First Exploration for Constrained ML Deployment Optimization in Crash-Prone Hierarchical Search Spaces**

Christian Lysenstoen, UC Berkeley, April 2026

- [v1 paper (single-GPU results)](paper/tba_paper_v1.pdf)
- [v2 paper (multi-GPU results)](paper/tba_paper_v2.pdf)
- arXiv link: *coming soon*

The paper studies budget-constrained deployment optimization in partially invalid hierarchical search spaces. The main finding is that explicit feasible-first exploration improves model-family discovery under tight constraints compared to cold-start TPE, with the hybrid discovering vit_tiny in 8/10 seeds versus TPE's 3/10 on RTX 5080.

## Citation

If you use TBA in your research, please cite:

```
@software{tba2026,
  author={Christian Lysenstoen},
  title={Thermal Budget Annealing: Crash-Aware Deployment Optimization for ML Models},
  year={2026},
  url={https://github.com/Chrislysen/Constrained-ML-Deployment}
}
```

## License

MIT
