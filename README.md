# DeployBench: Constrained ML Deployment Optimization

**Christian Lysenstøen** — UC Berkeley / INN (Inland Norway University of Applied Sciences)

This repository contains two related research projects sharing the DeployBench infrastructure for benchmarking ML deployment under hard constraints.

| Paper | Status | PDF |
|-------|--------|-----|
| **Feasible-First Exploration for Constrained ML Deployment Optimization** (TBA) | Awaiting arXiv endorsement | [PDF](paper/ConstrainedML_deployment_v2.pdf) |
| **Hidden Device Heterogeneity in INT8 Inference** | Ready for arXiv submission | [PDF](paper/lysenstoen2026_hidden_device_heterogeneity.pdf) |

---

## Overview

### Paper 1: Thermal Budget Annealing (TBA)

A feasible-first exploration optimizer for constrained ML deployment. Standard optimizers (TPE, Bayesian optimization, random search) waste 40–80% of their trial budget on configurations that crash, OOM, or violate deployment constraints. TBA uses a two-phase strategy:

1. **Phase 1 — Crash-aware simulated annealing** maps the feasible region in 5–15 trials
2. **Phase 2 — Warm-started TPE** optimizes within the safe region using all Phase 1 history

On RTX 5080 with `edge_tight` constraints (p95 ≤ 20ms, memory ≤ 512MB), TBA-TPE achieves **0.761 accuracy** and discovers `vit_tiny` in **8/10 seeds** vs. TPE's 3/10.

### Paper 2: Hidden Device Heterogeneity

An empirical study showing that PyTorch's dynamic INT8 quantization (`torch.ao.quantization`) silently routes inference from GPU to CPU. This creates **stochastic feasibility boundaries**: identical configurations produce different latency distributions depending on which device actually executes them. GPU inference is deterministic; CPU inference under INT8 flips feasibility up to **39% of the time** for MobileNetV2.

This finding explains why deployment optimizers see non-reproducible results near constraint boundaries — the search space has a hidden device dimension that existing frameworks do not account for.

---

## Key Findings (Paper 2)

### GPU vs CPU Execution

| Metric | GPU configs (14 tested) | CPU/INT8 configs (6 tested) |
|--------|------------------------|-----------------------------|
| Max flip rate | 2% (noise-level) | **39%** |
| Configs with >10% flips | 0 | 2 |
| Execution device | Deterministic (CUDA) | Stochastic (CPU scheduler) |

### Worst-Case Configuration

**MobileNetV2 / INT8 / batch size 3:**
- p95 latency: 19.70 ± 3.46 ms (threshold: 20 ms)
- Flip rate: **39%** — feasibility changes between runs
- Root cause: CPU thread scheduling variance under `torch.ao.quantization.quantize_dynamic`

### Cross-GPU Comparison

- RTX 5080 → T4: **5.8× latency shift** for identical configurations
- Fixed safety margins (e.g., 15ms instead of 20ms): all statistically identical to baseline — margins cannot solve device heterogeneity

### ACGF (Negative Result)

Adaptive Constraint Gradient Following was attempted as a learned alternative to fixed margins. It did not outperform the baseline. This negative result is documented in the paper and the code is preserved in `experiments/acgf/`.

---

## Repository Structure

```
tba/                              # TBA optimizer (Paper 1)
  api.py                          #   optimize_deployment() entry point
  types.py                        #   EvalResult, VariableDef dataclasses
  optimizer/
    tba_tpe_hybrid.py             #   TBA → TPE handoff optimizer
    tba_optimizer.py              #   Pure SA (ablation baseline)
    search_space.py               #   Hierarchical mixed-variable space
    subspace_tracker.py           #   Subspace + combo blacklisting
    surrogate.py                  #   RF surrogate (OOB-gated)
    feasible_tpe.py               #   KDE-based feasible-region sampler
  baselines/
    random_search.py              #   Random search baseline
    optuna_tpe.py                 #   Optuna TPE baseline
    constrained_bo.py             #   Constrained BO baseline
  benchmarks/
    profiler.py                   #   Deployment profiler (latency/memory/accuracy)
    model_zoo.py                  #   Pre-trained ImageNet models
    synthetic_bench.py            #   Crashy Branin + Hierarchical Rosenbrock
  spaces/
    image_classification.py       #   Search space definition + scenarios

experiments/
  boundary_probe/                 # Boundary probe experiment (Paper 2)
    select_boundary_configs.py    #   Config selector (zone-based)
    probe_runner.py               #   Subprocess-isolated probe runner
    analyze_boundary.py           #   Analyzer + figure generator
    boundary_configs.json         #   Selected configs (25 configs)
    flipper_configs.json          #   Follow-up flipper experiment configs
    int8_cross_model_configs.json #   Cross-model INT8 configs
  acgf/                           # ACGF method (negative result, Paper 2)
    acgf_evaluator.py             #   ACGF evaluator
    run_comparison.py             #   Comparison runner
    analyze_comparison.py         #   Comparison analyzer
  run_deployment.py               # TBA deployment experiment runner
  run_synthetic.py                # Synthetic benchmark runner
  run_budget_sweep.py             # Budget sweep experiment
  generate_figures.py             # TBA paper figure generator

results/
  boundary_probe/                 # Probe JSONL data (2,858 records)
    probe_*.jsonl                 #   Raw probe results (7 runs)
    summary_stats.json            #   Computed summary statistics
    figures/                      #   Generated figures (5 plots)
  acgf_comparison/                # ACGF comparison results
  
paper/                            # Paper PDFs
  ConstrainedML_deployment.pdf    #   TBA paper v1 (single-GPU)
  ConstrainedML_deployment_v2.pdf #   TBA paper v2 (multi-GPU)
  lysenstoen2026_hidden_device_heterogeneity.pdf  # Paper 2

data/imagenette2-320/             # ImageNette dataset (10-class ImageNet subset)
figures/                          # TBA paper figures
```

---

## DeployBench Configuration Space

| Dimension | Values |
|-----------|--------|
| Models | `resnet18`, `resnet50`, `mobilenet_v2`, `efficientnet_b0`, `vit_tiny` |
| Backends | `pytorch_eager`, `torch_compile`, `onnxruntime` |
| Quantization | `fp32`, `fp16`, `int8_dynamic` |
| Batch size | 1–32 (log scale) |
| num_threads | 1–8 (conditional on backend) |

**Constraint scenario (`edge_tight`):** p95 latency ≤ 20ms, peak memory ≤ 512MB

**Dataset:** ImageNette — a 10-class subset of ImageNet (Tench, English Springer, Cassette Player, Chain Saw, Church, French Horn, Garbage Truck, Gas Pump, Golf Ball, Parachute).

---

## Reproducing Results

### Prerequisites

```bash
# Python >= 3.10 required
pip install -e ".[gpu]"

# Additional dependency for boundary probe
pip install scipy psutil
```

### Paper 1: TBA Experiments

```bash
# Synthetic benchmarks (no GPU needed)
PYTHONPATH=. python experiments/run_synthetic.py --benchmark crashy_branin --budget 30 --seeds 10

# Budget sweep
PYTHONPATH=. python experiments/run_budget_sweep.py

# Real deployment (GPU + ImageNette required)
PYTHONPATH=. python experiments/run_deployment.py --scenarios edge_tight server_throughput --seeds 10

# Generate figures
PYTHONPATH=. python experiments/generate_figures.py
```

### Paper 2: Boundary Probe

```bash
# Step 1: Select boundary configs from existing TBA results
PYTHONPATH=. python experiments/boundary_probe/select_boundary_configs.py \
    --result-dirs results_deployment/edge_tight results_deployment_v2/edge_tight \
    --output experiments/boundary_probe/boundary_configs.json

# Step 2: Run the probe (50 repeats × 200 measurement passes per config)
PYTHONPATH=. python experiments/boundary_probe/probe_runner.py \
    --config-file experiments/boundary_probe/boundary_configs.json \
    --output-dir results/boundary_probe \
    --repeats 50 \
    --warmup-passes 20 \
    --measurement-passes 200 \
    --dataset-path data

# Step 3: Analyze results and generate figures
PYTHONPATH=. python experiments/boundary_probe/analyze_boundary.py \
    --input-dir results/boundary_probe \
    --output-dir results/boundary_probe/figures
```

**Expected output** from the analyzer:
- `summary_stats.json` with per-config flip rates
- 5 figures: feasibility vs distance, by-model breakdown, flip rate heatmap, cold/warm comparison, temperature correlation
- Go/no-go verdict for the stochastic feasibility claim

### Paper 2: ACGF Comparison (Negative Result)

```bash
PYTHONPATH=. python experiments/acgf/run_comparison.py
PYTHONPATH=. python experiments/acgf/analyze_comparison.py
```

---

## Hardware

| Machine | GPU | OS | Python | Role |
|---------|-----|----|--------|------|
| Primary | NVIDIA RTX 5080 (Laptop) | Windows 11 | 3.14 | All experiments |
| Secondary | NVIDIA Tesla T4 | Linux (Colab) | 3.10 | Cross-GPU validation |

Additional GPUs tested for Paper 1: H100, A100, L4 (all Google Colab).

---

## Citation

```bibtex
@article{lysenstoen2026tba,
  author  = {Christian Lysenstøen},
  title   = {Feasible-First Exploration for Constrained {ML} Deployment 
             Optimization in Crash-Prone Hierarchical Search Spaces},
  year    = {2026},
  note    = {Awaiting arXiv endorsement},
  url     = {https://github.com/Chrislysen/Constrained-ML-Deployment}
}

@article{lysenstoen2026heterogeneity,
  author  = {Christian Lysenstøen},
  title   = {Hidden Device Heterogeneity: How Dynamic {INT8} Quantization
             Creates Stochastic Feasibility Boundaries in {ML} Deployment},
  year    = {2026},
  note    = {Submitted to arXiv},
  url     = {https://github.com/Chrislysen/Constrained-ML-Deployment}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
