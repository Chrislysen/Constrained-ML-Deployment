# Thermal Budget Annealing: Complete Build Instructions for Claude Code

## PROJECT OVERVIEW

**Project Name:** Thermal Budget Annealing (TBA) — Feasible-First Optimization for Real-World ML Deployment

**One-Line Pitch:** A constraint-aware, budget-conscious optimizer that finds near-optimal ML deployment configurations (model variant × quantization × batch size × runtime backend × compiler flags) in 10–50 expensive trials, outperforming standard BO/TPE/random search in mixed hierarchical search spaces with hard latency/memory/power constraints.

**Research Question:** Can a feasible-first annealing-based meta-optimization method outperform TPE, Bayesian optimization, and evolutionary search in mixed hierarchical deployment spaces when only 20–50 expensive evaluations are allowed and hard constraints (latency cap, memory cap, thermal/power cap) must be satisfied?

**Success Metrics (what counts as winning):**
1. Best feasible objective found (accuracy or throughput) after N trials
2. Regret curve over trial budget (should drop faster than baselines)
3. Time-to-first-feasible-config (TBA should find a valid config faster)
4. Pareto frontier quality (hypervolume indicator for multi-objective)
5. Constraint violation rate (TBA should have fewer infeasible evaluations)

---

## WHAT WE'RE BUILDING — THE FULL SYSTEM

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   TBA Optimizer                      │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Search Space  │  │  Annealing   │  │ Constraint │ │
│  │   Manager     │→ │   Engine     │→ │  Handler   │ │
│  │ (hierarchical │  │ (adaptive    │  │ (feasible- │ │
│  │  mixed vars)  │  │  temperature)│  │  first)    │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│           │                │                │        │
│           ▼                ▼                ▼        │
│  ┌──────────────────────────────────────────────────┐│
│  │           Surrogate Model (optional)             ││
│  │    (Random Forest or GP for cheap predictions)   ││
│  └──────────────────────────────────────────────────┘│
│                        │                             │
│                        ▼                             │
│  ┌──────────────────────────────────────────────────┐│
│  │              Trial Cost Estimator                ││
│  │   (predicts wall-clock cost of next evaluation)  ││
│  └──────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Deployment Benchmark Suite              │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐ │
│  │ Model Zoo  │  │ Quantizer  │  │ Runtime/Backend│ │
│  │ (ResNet,   │  │ (INT8,FP16 │  │ (ONNX, TRT,   │ │
│  │  MobileNet │  │  INT4,     │  │  TVM, Torch)   │ │
│  │  EfficientN│  │  mixed)    │  │                │ │
│  │  ViT, YOLO)│  │            │  │                │ │
│  └────────────┘  └────────────┘  └────────────────┘ │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐ │
│  │ Batch Size │  │ Hardware   │  │  Profiler      │ │
│  │ (1,2,4,8,  │  │ Targets    │  │  (latency, mem │ │
│  │  16,32,64) │  │ (CPU, GPU, │  │   throughput,  │ │
│  │            │  │  edge)     │  │   power)       │ │
│  └────────────┘  └────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Evaluation & Reporting                  │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐ │
│  │ Regret     │  │ Pareto     │  │ Dashboard      │ │
│  │ Curves     │  │ Frontiers  │  │ (Streamlit)    │ │
│  └────────────┘  └────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## DIRECTORY STRUCTURE

```
thermal-budget-annealing/
├── README.md                          # Project overview, results, how to run
├── pyproject.toml                     # Package config
├── setup.py
├── requirements.txt
│
├── tba/                               # Main package
│   ├── __init__.py
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── tba_optimizer.py           # CORE: The TBA optimizer
│   │   ├── annealing_engine.py        # Adaptive simulated annealing with restarts
│   │   ├── search_space.py            # Hierarchical mixed-variable search space
│   │   ├── constraint_handler.py      # Feasible-first constraint management
│   │   ├── surrogate.py               # Optional surrogate model (RF/GP)
│   │   ├── cost_estimator.py          # Trial cost prediction
│   │   └── temperature_schedule.py    # Adaptive cooling schedules
│   │
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── deployment_bench.py        # ML deployment benchmark suite
│   │   ├── synthetic_bench.py         # Synthetic test functions with constraints
│   │   ├── model_zoo.py               # Pre-configured models for benchmarking
│   │   └── profiler.py                # Latency/memory/throughput profiling
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── random_search.py           # Random search baseline
│   │   ├── optuna_tpe.py              # Optuna TPE baseline
│   │   ├── optuna_cmaes.py            # Optuna CMA-ES baseline
│   │   ├── constrained_bo.py          # Constrained Bayesian optimization
│   │   └── evolutionary.py            # Evolutionary strategy baseline
│   │
│   ├── spaces/
│   │   ├── __init__.py
│   │   ├── image_classification.py    # ResNet/EfficientNet/MobileNet space
│   │   ├── object_detection.py        # YOLO/SSD space
│   │   └── transformer_inference.py   # ViT/BERT space
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── plotting.py                # Regret curves, Pareto fronts, heatmaps
│       ├── metrics.py                 # Hypervolume, regret, feasibility rate
│       └── reproducibility.py         # Seed management, result serialization
│
├── experiments/
│   ├── run_benchmark.py               # Main experiment runner
│   ├── run_synthetic.py               # Synthetic benchmark experiments
│   ├── run_deployment.py              # Real deployment experiments
│   ├── analyze_results.py             # Result analysis and plot generation
│   └── configs/
│       ├── synthetic_experiments.yaml
│       ├── deployment_experiments.yaml
│       └── ablation_experiments.yaml
│
├── dashboard/
│   ├── app.py                         # Streamlit dashboard
│   └── components/
│       ├── pareto_viewer.py
│       ├── regret_plot.py
│       └── config_explorer.py
│
├── tests/
│   ├── test_optimizer.py
│   ├── test_search_space.py
│   ├── test_constraints.py
│   ├── test_benchmarks.py
│   └── test_baselines.py
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_results_analysis.ipynb
│   └── 03_paper_figures.ipynb
│
└── docs/
    ├── RESEARCH_CLAIM.md              # Exact novelty statement
    ├── BENCHMARK_PROTOCOL.md          # How experiments are run
    └── RESULTS.md                     # Summary of findings
```

---

## STEP 1: ENVIRONMENT SETUP

```bash
# Create project
mkdir thermal-budget-annealing
cd thermal-budget-annealing
git init

# Create conda environment
conda create -n tba python=3.10 -y
conda activate tba

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu          # or onnxruntime for CPU-only
pip install optuna                    # Baseline optimizer (TPE, CMA-ES)
pip install botorch gpytorch          # Bayesian optimization baseline
pip install scikit-learn              # Random forest surrogate
pip install numpy scipy pandas
pip install matplotlib seaborn plotly # Plotting
pip install streamlit                 # Dashboard
pip install pyyaml                    # Config files
pip install rich                      # Nice console output
pip install pytest                    # Testing
pip install hydra-core                # Experiment configuration (optional but recommended)

# For real deployment benchmarking (install what's available on your hardware)
pip install onnx onnxruntime
pip install tensorrt                  # If NVIDIA GPU available (may need separate install)
# pip install apache-tvm             # Optional, harder to install

# For profiling
pip install psutil gputil py3nvml    # System monitoring

# Quantization tools
pip install neural-compressor        # Intel Neural Compressor (broad quantization support)
pip install optimum                  # HuggingFace Optimum (ONNX + quantization)
pip install torch-pruning            # Structured pruning
```

---

## STEP 2: THE SEARCH SPACE DEFINITION

This is critical. The search space must be **hierarchical** and **mixed** (discrete + continuous + categorical + conditional).

### File: `tba/optimizer/search_space.py`

The search space should support these variable types:
- **Categorical**: model_name, backend, quantization_mode
- **Integer**: batch_size, num_threads
- **Continuous**: learning_rate (if fine-tuning), quantization_percentile thresholds
- **Conditional/Hierarchical**: some variables only exist when others take certain values
  - Example: `tensorrt_precision` only relevant when `backend == "tensorrt"`
  - Example: `int8_calibration_method` only relevant when `quantization == "int8"`

### Example Search Space for Image Classification Deployment:

```python
DEPLOYMENT_SEARCH_SPACE = {
    "model_name": {
        "type": "categorical",
        "choices": [
            "resnet18", "resnet50", "resnet101",
            "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
            "efficientnet_b0", "efficientnet_b2", "efficientnet_b4",
            "vit_tiny_patch16_224", "vit_small_patch16_224"
        ]
    },
    "quantization": {
        "type": "categorical",
        "choices": ["fp32", "fp16", "int8_dynamic", "int8_static", "int4_weight_only"]
    },
    "batch_size": {
        "type": "integer",
        "low": 1,
        "high": 64,
        "log": True  # Search in log space: 1, 2, 4, 8, 16, 32, 64
    },
    "backend": {
        "type": "categorical",
        "choices": ["pytorch_eager", "torch_compile", "onnxruntime", "tensorrt"]
    },
    "num_threads": {
        "type": "integer",
        "low": 1,
        "high": 16,
        "condition": "backend in ['pytorch_eager', 'onnxruntime']"
    },
    "torch_compile_backend": {
        "type": "categorical",
        "choices": ["inductor", "cudagraphs", "onnxrt"],
        "condition": "backend == 'torch_compile'"
    },
    "torch_compile_mode": {
        "type": "categorical",
        "choices": ["default", "reduce-overhead", "max-autotune"],
        "condition": "backend == 'torch_compile'"
    },
    "trt_workspace_mb": {
        "type": "integer",
        "low": 256,
        "high": 4096,
        "condition": "backend == 'tensorrt'"
    },
    "int8_calibration_method": {
        "type": "categorical",
        "choices": ["minmax", "entropy", "percentile"],
        "condition": "quantization in ['int8_static']"
    },
    "int8_calibration_samples": {
        "type": "integer",
        "low": 50,
        "high": 500,
        "condition": "quantization in ['int8_static']"
    }
}
```

### Hierarchy Rules (important):
- If `backend == "tensorrt"`, then `quantization` must be `"fp32"`, `"fp16"`, or `"int8_static"` (TRT doesn't support all modes)
- If `backend == "pytorch_eager"`, then `torch_compile_*` variables don't exist
- If `quantization == "fp32"`, then `int8_calibration_*` variables don't exist
- Some (model, quantization) combos may be invalid (e.g., INT4 may not work on all architectures)

The search space manager must:
1. Know which variables are active given current values of other variables
2. Sample only valid configurations
3. Propose neighbors that respect hierarchy (e.g., changing `backend` may activate/deactivate other variables)
4. Compute distance between configs accounting for mixed types and conditional structure

---

## STEP 3: THE CORE TBA OPTIMIZER

### File: `tba/optimizer/tba_optimizer.py`

**Key algorithmic ideas that make TBA different from vanilla simulated annealing:**

### 3.1 Feasible-First Strategy
Instead of optimizing objective and then checking constraints, TBA operates in two phases:

**Phase 1 — Feasibility Search (high temperature):**
- Start with high temperature
- Accept any move that brings you closer to the feasible region
- Objective: minimize total constraint violation (sum of max(0, latency - cap) + max(0, memory - cap) + ...)
- Only once a feasible config is found, switch to Phase 2

**Phase 2 — Objective Optimization (cooling):**
- Standard annealing on the actual objective (e.g., maximize throughput or accuracy)
- But with a hard feasibility filter: never accept a move into infeasible space unless temperature is very high (exploration)
- Track best feasible solution separately from current solution

### 3.2 Adaptive Temperature Schedule
Don't use a fixed cooling schedule. Instead:

```python
# Pseudocode for adaptive temperature
class AdaptiveTemperature:
    def __init__(self, T_init, T_min, patience=5):
        self.T = T_init
        self.T_min = T_min
        self.no_improve_count = 0
        self.patience = patience

    def update(self, improved: bool, phase: str):
        if improved:
            self.no_improve_count = 0
            if phase == "feasibility":
                self.T *= 0.95  # Cool slowly while making progress
            else:
                self.T *= 0.90  # Cool faster during optimization
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.patience:
                # Reheat — we're stuck
                self.T = min(self.T * 2.0, T_init * 0.5)
                self.no_improve_count = 0
```

### 3.3 Hierarchical Neighborhood
When proposing a neighbor config, the optimizer should:

1. With probability `p_structural` (e.g., 0.3): change a categorical/structural variable (model, backend, quantization mode). This may activate/deactivate other variables.
2. With probability `1 - p_structural`: change a numerical variable within current structure (batch_size, num_threads, etc.)
3. Step sizes for numerical variables should be proportional to temperature
4. When a structural change happens, randomly initialize newly activated variables

### 3.4 Optional Surrogate Acceleration
After accumulating 10+ evaluations, optionally train a Random Forest surrogate to:
- Pre-filter clearly bad configurations before expensive evaluation
- Estimate constraint satisfaction cheaply
- Guide the annealing toward promising regions

This is optional but can significantly reduce wasted evaluations.

### 3.5 Trial-Cost-Aware Search
Each evaluation has a different cost (some configs take 2s to profile, others take 60s). TBA should:
- Track wall-clock cost of each trial
- When budget is measured in time (not just trial count), prefer cheaper exploratory moves early
- Optionally train a cost model to predict evaluation time

### Core Optimizer Interface:

```python
class TBAOptimizer:
    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict,          # {"latency_ms": 50, "memory_mb": 2048, "power_w": 150}
        objective: str,             # "maximize_throughput" or "maximize_accuracy"
        budget: int,                # Max number of evaluations
        seed: int = 42,
        surrogate: bool = True,
        cost_aware: bool = True,
        T_init: float = 1.0,
        T_min: float = 0.001,
    ):
        ...

    def ask(self) -> dict:
        """Return the next configuration to evaluate."""
        ...

    def tell(self, config: dict, result: dict):
        """
        Report evaluation result.
        result = {
            "accuracy": 0.76,
            "throughput_qps": 1250.0,
            "latency_p50_ms": 12.3,
            "latency_p95_ms": 28.7,
            "latency_p99_ms": 45.2,
            "memory_mb": 1847.0,
            "power_w": 95.0,
            "eval_time_s": 34.5,
            "feasible": True
        }
        """
        ...

    def best(self) -> tuple[dict, dict]:
        """Return best feasible (config, result) found so far."""
        ...

    def get_pareto_front(self) -> list[tuple[dict, dict]]:
        """Return Pareto-optimal feasible configs for multi-objective."""
        ...

    def get_history(self) -> pd.DataFrame:
        """Return full optimization history."""
        ...
```

---

## STEP 4: BASELINES TO BEAT

You MUST implement these baselines using the exact same interface, search space, and evaluation function. No baseline should get an unfair disadvantage.

### Baseline 1: Random Search
```python
# tba/baselines/random_search.py
# Uniformly random sampling from the search space
# Filter: only report best FEASIBLE result
# This is the minimum bar. You must crush this.
```

### Baseline 2: Optuna TPE (Tree-structured Parzen Estimator)
```python
# tba/baselines/optuna_tpe.py
import optuna

# Use Optuna's built-in TPE sampler
# Handle constraints via Optuna's constraint interface:
#   study = optuna.create_study(
#       sampler=optuna.samplers.TPESampler(constraints_func=...)
#   )
# This is a STRONG baseline. Beating TPE meaningfully is hard.
```

### Baseline 3: Optuna CMA-ES
```python
# tba/baselines/optuna_cmaes.py
# CMA-ES via Optuna
# NOTE: CMA-ES struggles with categorical/discrete variables
# This baseline may be weak on mixed spaces — that's part of your argument
```

### Baseline 4: Constrained Bayesian Optimization
```python
# tba/baselines/constrained_bo.py
# Use BoTorch with constrained expected improvement
# This is technically the strongest baseline for continuous spaces
# It may struggle with the mixed hierarchical structure — document this
```

### Baseline 5: Regularized Evolutionary Search
```python
# tba/baselines/evolutionary.py
# Simple (μ + λ) evolutionary strategy
# Tournament selection with feasibility preference
# Mutation respects variable types
# Population size: 10-20 (appropriate for low budget)
```

### How to run baselines fairly:
- Same search space object
- Same evaluation function
- Same constraint thresholds
- Same random seeds (run each method with seeds 0-19, report mean ± std)
- Same budget (number of evaluations)
- Log every single evaluation for every method

---

## STEP 5: BENCHMARKS

### 5.1 Synthetic Benchmarks (fast iteration, develop algorithms here first)

Create constrained optimization problems where you know the ground truth:

```python
# tba/benchmarks/synthetic_bench.py

# Problem 1: Constrained Branin with mixed variables
# - 2 continuous vars, 1 categorical (3 choices), 1 integer
# - 2 inequality constraints
# - Known global optimum

# Problem 2: Hierarchical Rosenbrock
# - Categorical "mode" variable that activates different subsets of continuous vars
# - This tests hierarchical handling

# Problem 3: Noisy constrained multi-objective
# - 2 objectives (simulating accuracy vs latency)
# - 3 constraints
# - Gaussian noise on evaluations
# - Tests robustness

# Problem 4: High-dimensional mixed space
# - 5 categorical, 5 integer, 5 continuous
# - Conditional dependencies between categoricals
# - Tests scalability
```

### 5.2 Real Deployment Benchmarks

#### Dataset: ImageNet validation set (or a subset)
```bash
# Download ImageNet validation set (you need this for accuracy measurement)
# Option A: Use torchvision's built-in ImageNet loading
# Option B: Use a smaller proxy — CIFAR-100 or ImageNette (ImageNet subset)
# For FAST iteration, use ImageNette:
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz

# For full benchmarks, use ImageNet val set if you have access via Berkeley resources
```

#### Pre-trained Models (no training needed):
```python
# All models come from torchvision.models or timm — no training required
import torchvision.models as models
import timm

# These are your model choices in the search space:
model_zoo = {
    "resnet18": models.resnet18(weights="IMAGENET1K_V1"),
    "resnet50": models.resnet50(weights="IMAGENET1K_V2"),
    "resnet101": models.resnet101(weights="IMAGENET1K_V2"),
    "mobilenet_v2": models.mobilenet_v2(weights="IMAGENET1K_V2"),
    "mobilenet_v3_small": models.mobilenet_v3_small(weights="IMAGENET1K_V1"),
    "mobilenet_v3_large": models.mobilenet_v3_large(weights="IMAGENET1K_V2"),
    "efficientnet_b0": models.efficientnet_b0(weights="IMAGENET1K_V1"),
    "efficientnet_b2": models.efficientnet_b2(weights="IMAGENET1K_V1"),
    "efficientnet_b4": models.efficientnet_b4(weights="IMAGENET1K_V1"),
    "vit_tiny_patch16_224": timm.create_model("vit_tiny_patch16_224", pretrained=True),
    "vit_small_patch16_224": timm.create_model("vit_small_patch16_224", pretrained=True),
}
```

#### Profiling Function (the objective evaluator):
```python
# tba/benchmarks/profiler.py

def evaluate_config(config: dict, dataset, device="cuda") -> dict:
    """
    Given a deployment config, measure:
    - accuracy (top-1 on validation set or subset)
    - throughput (queries per second)
    - latency (p50, p95, p99 in milliseconds)
    - memory (peak GPU/CPU memory in MB)
    - power (if measurable, or estimate)
    - eval_time (wall-clock seconds for this evaluation)

    Steps:
    1. Load model from model_zoo[config["model_name"]]
    2. Apply quantization per config["quantization"]
    3. Export/compile to backend per config["backend"]
    4. Run warmup (50 batches)
    5. Run benchmark (200 batches), measure latency for each
    6. Run accuracy evaluation on validation subset (500-1000 images)
    7. Record peak memory
    8. Return result dict
    """

    result = {
        "accuracy": ...,
        "throughput_qps": ...,
        "latency_p50_ms": ...,
        "latency_p95_ms": ...,
        "latency_p99_ms": ...,
        "memory_peak_mb": ...,
        "power_w": ...,         # Optional
        "eval_time_s": ...,
        "feasible": ...,        # True if all constraints satisfied
        "error": None,          # If config crashed, store error message
    }
    return result
```

**IMPORTANT:** Some configs WILL crash (e.g., INT4 on a model that doesn't support it, or OOM). The evaluation function must catch all exceptions and return a result with `feasible=False` and the error message. The optimizer must handle crashed configs gracefully.

### 5.3 Constraint Scenarios (run experiments under each)

```yaml
# experiments/configs/deployment_experiments.yaml

scenarios:
  # Scenario 1: Edge deployment (tight constraints)
  edge_tight:
    constraints:
      latency_p95_ms: 20      # Must be under 20ms
      memory_peak_mb: 512     # Must fit in 512MB
    objective: maximize_accuracy
    budget: 30                 # Only 30 trials allowed

  # Scenario 2: Server deployment (relaxed latency, optimize throughput)
  server_throughput:
    constraints:
      latency_p99_ms: 100
      memory_peak_mb: 4096
    objective: maximize_throughput
    budget: 50

  # Scenario 3: Balanced multi-objective
  balanced:
    constraints:
      latency_p95_ms: 50
      memory_peak_mb: 2048
    objectives:
      - maximize_accuracy
      - maximize_throughput
    budget: 40

  # Scenario 4: Ultra-low budget
  ultra_low_budget:
    constraints:
      latency_p95_ms: 30
      memory_peak_mb: 1024
    objective: maximize_accuracy
    budget: 15                 # Only 15 trials — this is where TBA should shine

  # Scenario 5: Many constraints
  multi_constraint:
    constraints:
      latency_p50_ms: 15
      latency_p99_ms: 50
      memory_peak_mb: 1024
      power_w: 75              # If measurable
    objective: maximize_accuracy
    budget: 40
```

---

## STEP 6: EXPERIMENT RUNNER

### File: `experiments/run_benchmark.py`

```python
"""
Main experiment runner.
Runs all optimizers on all scenarios with multiple seeds.
Saves complete history for analysis.

Usage:
    python experiments/run_benchmark.py --config experiments/configs/deployment_experiments.yaml
    python experiments/run_benchmark.py --config experiments/configs/synthetic_experiments.yaml
    python experiments/run_benchmark.py --scenario edge_tight --optimizer tba --seeds 0-19
"""

# For each (scenario, optimizer, seed):
#   1. Initialize search space
#   2. Initialize optimizer with constraints and budget
#   3. Run ask/tell loop for budget iterations
#   4. Save full history to results/{scenario}/{optimizer}/seed_{seed}.json
#   5. Save best config and result

# After all runs:
#   - Compute mean ± std of best feasible objective across seeds
#   - Generate regret curves
#   - Generate Pareto fronts
#   - Generate feasibility rate plots
#   - Save summary table
```

---

## STEP 7: EVALUATION METRICS AND PLOTS

### File: `tba/utils/metrics.py`

```python
def compute_regret(history, optimal_value):
    """Simple regret: optimal - best_feasible_so_far at each step."""

def compute_feasibility_rate(history):
    """Fraction of evaluations that were feasible at each step."""

def compute_time_to_first_feasible(history):
    """Number of trials before first feasible config found."""

def compute_hypervolume(pareto_front, reference_point):
    """Hypervolume indicator for multi-objective comparison."""

def compute_anytime_performance(history, budget_fractions=[0.25, 0.5, 0.75, 1.0]):
    """Best feasible value at 25%, 50%, 75%, 100% of budget."""
```

### File: `tba/utils/plotting.py`

Generate these plots (these are what go in your README and writeup):

1. **Regret curves**: x = trial number, y = simple regret, one line per optimizer, shaded ± std across seeds
2. **Feasibility rate**: x = trial number, y = cumulative feasible fraction
3. **Time to first feasible**: bar chart per optimizer
4. **Pareto front comparison**: 2D scatter (accuracy vs latency or throughput vs memory) with Pareto fronts overlaid
5. **Budget efficiency**: x = wall-clock time spent, y = best feasible objective (accounts for variable trial costs)
6. **Ablation heatmap**: which TBA components matter (surrogate on/off, feasible-first on/off, reheating on/off)
7. **Search trajectory**: 2D projection of configs explored, colored by feasibility and objective value
8. **Constraint violation over time**: shows how TBA learns to avoid infeasible regions

---

## STEP 8: ABLATION STUDY

Run these variants of TBA to show which components matter:

```yaml
# experiments/configs/ablation_experiments.yaml

ablations:
  tba_full:             # Full system
    surrogate: true
    feasible_first: true
    adaptive_temp: true
    cost_aware: true
    reheating: true

  tba_no_surrogate:     # Remove surrogate
    surrogate: false
    feasible_first: true
    adaptive_temp: true
    cost_aware: true
    reheating: true

  tba_no_feasible_first: # Remove feasible-first (treat as penalty instead)
    surrogate: true
    feasible_first: false
    adaptive_temp: true
    cost_aware: true
    reheating: true

  tba_fixed_temp:       # Fixed cooling schedule instead of adaptive
    surrogate: true
    feasible_first: true
    adaptive_temp: false
    cost_aware: true
    reheating: false

  tba_no_cost_aware:    # Ignore trial costs
    surrogate: true
    feasible_first: true
    adaptive_temp: true
    cost_aware: false
    reheating: true

  vanilla_sa:           # Plain simulated annealing (no bells and whistles)
    surrogate: false
    feasible_first: false
    adaptive_temp: false
    cost_aware: false
    reheating: false
```

---

## STEP 9: DASHBOARD

### File: `dashboard/app.py`

Build a Streamlit dashboard that shows:

1. **Live experiment tracking**: progress bar, current best, recent evaluations
2. **Interactive Pareto explorer**: click configs on the Pareto front to see their details
3. **Regret curve comparison**: select which optimizers to compare
4. **Config detail view**: for any evaluated config, show model, quantization, backend, all metrics
5. **Search space visualization**: show which regions of the space have been explored
6. **One-click run**: button to launch a new optimization run with selected scenario

```bash
# Run with:
streamlit run dashboard/app.py
```

---

## STEP 10: README AND PACKAGING

### README.md structure:

```markdown
# Thermal Budget Annealing (TBA)

**Feasible-first optimization for ML deployment under hard constraints and tiny budgets.**

> Given a model zoo, quantization options, and runtime backends, TBA finds the
> best deployment configuration that satisfies your latency, memory, and power
> constraints — in as few as 15-30 expensive evaluations.

## Key Results

[Insert regret curve plot showing TBA beating baselines]
[Insert Pareto front comparison]
[Insert time-to-first-feasible bar chart]

## Why TBA?

Standard optimizers (TPE, BO, evolutionary) treat constraints as penalties or
filters. TBA uses a feasible-first annealing strategy that:
- Finds valid configs 2-3x faster
- Wastes fewer evaluations on infeasible configs
- Handles mixed hierarchical search spaces natively
- Adapts exploration based on remaining budget

## Quick Start
...

## Benchmark Results
...

## Citation
...
```

---

## STEP 11: IMPLEMENTATION ORDER (RECOMMENDED BUILD SEQUENCE)

Build in this order to get results as fast as possible:

### Week 1: Foundation
1. Set up repo structure and environment
2. Implement `SearchSpace` class with hierarchical mixed-variable support
3. Implement basic `TBAOptimizer` with feasible-first and adaptive temperature (NO surrogate yet)
4. Implement `RandomSearch` baseline
5. Test on synthetic benchmark (constrained Branin)

### Week 2: Baselines + Synthetic
6. Implement Optuna TPE baseline
7. Implement Optuna CMA-ES baseline
8. Implement all synthetic benchmarks
9. Run TBA vs baselines on synthetic problems
10. First results: does TBA beat random search and match/beat TPE on synthetics?

### Week 3: Real Deployment
11. Implement model zoo and profiler
12. Implement deployment benchmark (start with image classification)
13. Handle crashes and edge cases in profiling
14. Run first real deployment experiments (edge_tight scenario)

### Week 4: Polish + Extend
15. Add surrogate model
16. Add cost-aware search
17. Implement constrained BO baseline
18. Implement evolutionary baseline
19. Run all scenarios with all baselines (20 seeds each)

### Week 5: Analysis + Dashboard
20. Generate all plots
21. Run ablation study
22. Build Streamlit dashboard
23. Write RESEARCH_CLAIM.md
24. Polish README with actual results

### Week 6+: Stretch Goals
25. Add object detection search space (YOLO)
26. Add transformer inference search space (ViT/BERT)
27. Test cross-hardware transfer (if second device available)
28. Write up as workshop paper or technical report

---

## CRITICAL IMPLEMENTATION DETAILS

### Handling Crashes
Many configs WILL fail (OOM, unsupported quantization, compilation error). The system MUST:
- Wrap every evaluation in try/except
- Return `feasible=False` with error info on crash
- Never count a crash as worse than a successful infeasible config
- Log crashes for debugging

### Reproducibility
- Set `torch.manual_seed()`, `np.random.seed()`, `random.seed()` at start of each run
- Use `torch.backends.cudnn.deterministic = True`
- Log exact package versions
- Save complete configs and results as JSON
- Every experiment must be reproducible from the saved config

### Profiling Accuracy
- Always do warmup runs before timing (at least 50 iterations)
- Use CUDA synchronization for GPU timing (`torch.cuda.synchronize()`)
- Report p50, p95, p99 latency (not just mean)
- Measure peak memory, not just allocated memory
- Run accuracy on a FIXED subset (not random) for consistency

### Statistical Rigor
- Run every (optimizer, scenario) pair with 20 different random seeds
- Report mean ± standard deviation
- Use Wilcoxon signed-rank test for pairwise comparisons
- Show individual seed results in supplementary plots, not just means

---

## WHAT "WINNING" LOOKS LIKE

You have a strong result if you can show ANY of these:

1. **TBA finds a feasible config in 3-5 trials while TPE takes 10-15** on the constrained scenarios
2. **TBA's best feasible accuracy/throughput is 2-5% better** than TPE/BO at budget=30
3. **TBA wastes 30-50% fewer evaluations on infeasible configs** compared to baselines
4. **TBA's Pareto front has better hypervolume** on the multi-objective scenario
5. **TBA is especially strong at ultra-low budgets (15-20 trials)** where BO can't build a good model and random search is too sparse

You have a **great** result if TBA wins on multiple scenarios AND the ablation shows each component contributes.

---

## FILES TO DOWNLOAD / EXTERNAL DEPENDENCIES

### Required Downloads:
```bash
# ImageNette dataset (fast proxy for ImageNet, ~1.5GB)
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz -C data/

# Pre-trained models download automatically via torchvision/timm on first use
# No manual download needed
```

### Python Package Versions (pin these):
```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12
optuna>=3.4.0
botorch>=0.9.0
gpytorch>=1.11
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
streamlit>=1.28.0
onnxruntime>=1.16.0
neural-compressor>=2.3.0
optimum>=1.13.0
pyyaml>=6.0
rich>=13.0.0
psutil>=5.9.0
pytest>=7.4.0
```

### Hardware Recommendations:
- **Minimum**: Any machine with a GPU (even a single consumer GPU like RTX 3060)
- **Ideal**: Access to a Berkeley cluster with mixed hardware (GPU + CPU targets)
- **CPU-only works too**: Just limit backends to pytorch_eager/onnxruntime and skip TensorRT
- The whole point is that TBA finds good configs WITH FEWER TRIALS, so you don't need massive compute

---

## NOVELTY STATEMENT (put this in docs/RESEARCH_CLAIM.md)

**What is NOT novel:** Simulated annealing, Bayesian optimization, hyperparameter tuning, ML deployment optimization, constrained optimization. All of these exist independently.

**What IS novel:** The combination of:
1. Feasible-first phase transition (not just penalty-based constraint handling)
2. Hierarchical-aware neighborhood for mixed deployment spaces
3. Adaptive reheating tied to constraint satisfaction progress
4. Cost-aware trial selection for unequal evaluation budgets
5. Applied specifically to the joint model×quantization×backend×runtime optimization problem

**The claim:** In the regime of 15-50 trials with hard constraints on mixed hierarchical deployment spaces, TBA finds better feasible configurations faster than standard TPE, BO, CMA-ES, and evolutionary search.

---

## COMMON PITFALLS TO AVOID

1. **Don't benchmark unfairly.** Give every baseline the same search space, budget, and constraint info. If TPE can use pruning, let it. If BO can use constraint modeling, let it.
2. **Don't cherry-pick seeds.** Run 20 seeds minimum and report aggregate statistics.
3. **Don't ignore failed configs.** Crashes are part of the problem. If TBA crashes less, that's a legitimate advantage — but log it explicitly.
4. **Don't over-tune TBA's hyperparameters.** Use the same T_init, cooling rate, etc. across all scenarios. If you tune per-scenario, you're overfitting.
5. **Don't claim generality you can't prove.** This is for deployment optimization, not all optimization. Be precise.
6. **Don't skip the ablation.** Without it, reviewers will ask "isn't this just simulated annealing?" The ablation is your answer.
