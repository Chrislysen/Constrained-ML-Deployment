# TBA Follow-Up Instructions — READ BEFORE BUILDING

## SCOPE CHANGE: Build smaller, sharper

Ignore the multi-objective / Pareto / hypervolume stuff for now. The core project is:

**Single-objective optimization (maximize accuracy OR maximize throughput) under hard constraints (latency cap, memory cap) in a crash-heavy, hierarchical deployment space with a tiny trial budget (10-50 evals).**

That's it. No multi-objective until the single-objective story is solid.

Cut to:
- 1 deployment domain: image classification only
- 3 scenarios: edge_tight (budget=20), server_throughput (budget=40), ultra_low (budget=15)
- 3 baselines first: Random Search, Optuna TPE, Constrained BO (BoTorch)
- Add CMA-ES and evolutionary ONLY if time permits
- No Streamlit dashboard until results exist — just matplotlib/plotly plots
- 10 seeds minimum, not 20 (scale up later)

## THE REAL NOVELTY: CRASH-AWARE FEASIBLE-FIRST SEARCH

The project identity is NOT "annealing optimizer." It is:

**"Existing optimizers waste most of their budget on configs that crash, OOM, or violate constraints. TBA finds feasible configs first, then optimizes within the feasible region."**

This means the benchmark MUST include lots of invalid configs. The search space should be designed so that ~30-50% of random configs either crash or violate constraints. That's realistic and it's where TBA wins.

Track and report these crash-specific metrics:
- crash_rate: fraction of evaluations that errored/OOM'd per optimizer
- wasted_budget: fraction of budget spent on infeasible OR crashed configs
- time_to_first_feasible: how many trials before finding ANY valid config

## BUILD ORDER: Start with this exact sequence

### PHASE 1: Prove the algorithm works (synthetic, no ML models)

Build these files first:
1. `tba/optimizer/search_space.py` — hierarchical mixed-variable space
2. `tba/optimizer/tba_optimizer.py` — core optimizer with feasible-first + adaptive temp
3. `tba/baselines/random_search.py` — random baseline
4. `tba/benchmarks/synthetic_bench.py` — constrained Branin + hierarchical Rosenbrock
5. `experiments/run_synthetic.py` — run TBA vs random on synthetics

**STOP HERE AND TEST.** If TBA doesn't beat random search on synthetics, debug before moving on. Do not touch real models until synthetics work.

### PHASE 2: Add real baselines, still synthetic

6. `tba/baselines/optuna_tpe.py`
7. `tba/baselines/constrained_bo.py`
8. Run all 3 optimizers on synthetics with 10 seeds
9. Generate regret curves

**CHECKPOINT:** TBA should match or beat TPE on constrained synthetics. If not, tune the algorithm.

### PHASE 3: Real deployment benchmark

10. `tba/benchmarks/model_zoo.py` — load pretrained models
11. `tba/benchmarks/profiler.py` — measure latency/memory/accuracy
12. `tba/spaces/image_classification.py` — the real search space
13. `experiments/run_deployment.py` — run on real models
14. Generate all plots

### PHASE 4: Ablation + polish

15. Ablation: TBA_full vs TBA_no_feasible_first vs TBA_no_reheating vs vanilla_SA
16. Final plots and README
17. Dashboard (if time)

## CRITICAL IMPLEMENTATION DETAILS CLAUDE CODE NEEDS

### The ask/tell interface pattern

Every optimizer (TBA and all baselines) MUST use the same interface:

```python
class BaseOptimizer:
    def __init__(self, search_space, constraints, objective, budget, seed):
        ...
    
    def ask(self) -> dict:
        """Return next config to evaluate. Must be a valid point in the space."""
        ...
    
    def tell(self, config: dict, result: dict) -> None:
        """
        Report result. Result dict always has these keys:
        - objective_value: float (the thing we're optimizing)
        - constraints: dict[str, float] (measured values for each constraint)
        - feasible: bool (did it satisfy ALL constraints?)
        - crashed: bool (did evaluation error out?)
        - eval_time_s: float (wall-clock cost of this evaluation)
        - error_msg: str | None
        """
        ...
    
    def best_feasible(self) -> tuple[dict, dict] | None:
        """Return best (config, result) that was feasible, or None."""
        ...
```

### Optuna TPE baseline — handle hierarchy properly

```python
# The Optuna baseline MUST use conditional sampling to be fair:
def objective(trial):
    backend = trial.suggest_categorical("backend", ["pytorch_eager", "torch_compile", "onnxruntime"])
    
    # Conditional params — only suggest when relevant
    if backend == "torch_compile":
        compile_mode = trial.suggest_categorical("compile_mode", ["default", "reduce-overhead", "max-autotune"])
    
    if backend in ["pytorch_eager", "onnxruntime"]:
        num_threads = trial.suggest_int("num_threads", 1, 16)
    
    # This is how Optuna handles hierarchy natively. Give it this advantage.
```

### Constrained BO baseline — use BoTorch properly

```python
# Use BoTorch's constrained expected improvement
# Key: encode categoricals as one-hot, handle hierarchy by fixing inactive dims
# This baseline is strong on continuous spaces but struggles with hierarchy
# That's part of your argument — document it, don't handicap it
```

### The profiler MUST handle crashes gracefully

```python
import signal, traceback

class TimeoutError(Exception):
    pass

def evaluate_config(config, dataset, device, timeout_seconds=120):
    """
    CRITICAL: This function must NEVER crash the experiment runner.
    Every possible failure returns a valid result dict with crashed=True.
    """
    result = {
        "objective_value": float("-inf"),
        "constraints": {},
        "feasible": False,
        "crashed": False,
        "eval_time_s": 0.0,
        "error_msg": None,
    }
    
    start = time.time()
    try:
        # Set timeout
        # ... load model, quantize, compile, benchmark ...
        
        # Measure everything
        result["objective_value"] = accuracy  # or throughput
        result["constraints"] = {
            "latency_p95_ms": measured_latency,
            "memory_peak_mb": measured_memory,
        }
        result["feasible"] = all(
            result["constraints"][k] <= v 
            for k, v in constraint_caps.items()
        )
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result["crashed"] = True
        result["error_msg"] = "CUDA OOM"
    except Exception as e:
        result["crashed"] = True
        result["error_msg"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        result["eval_time_s"] = time.time() - start
        # ALWAYS clear GPU memory between evaluations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result
```

### Search space — make invalid configs common on purpose

The search space should be designed so random sampling hits ~30-50% invalid/crash configs. Ways to achieve this:

```python
# These combos will crash or fail in practice:
KNOWN_INVALID_COMBOS = [
    # INT4 weight-only doesn't work on all architectures
    ("vit_tiny_patch16_224", "int4_weight_only"),
    ("vit_small_patch16_224", "int4_weight_only"),
    
    # TensorRT doesn't support all quantization modes
    ("tensorrt", "int8_dynamic"),
    ("tensorrt", "int4_weight_only"),
    
    # Large models + large batch sizes = OOM on consumer GPUs
    ("resnet101", 64),  # batch_size=64 may OOM
    ("efficientnet_b4", 32),
    ("vit_small_patch16_224", 32),
]

# Don't filter these out of the space. Let the optimizer discover them.
# That's the whole point — TBA should learn to avoid them faster.
```

### Synthetic benchmark — constrained Branin with crash zones

```python
def synthetic_crashy_branin(config):
    """
    Modified Branin with:
    - 1 categorical variable (3 choices, each shifts landscape)
    - 2 continuous variables
    - 1 integer variable
    - 2 inequality constraints
    - A "crash zone" where evaluation raises an error
    
    ~35% of random configs crash or are infeasible.
    """
    x1, x2 = config["x1"], config["x2"]
    mode = config["mode"]  # categorical: "A", "B", "C"
    resolution = config["resolution"]  # integer: 1-10
    
    # Crash zone: simulate OOM or invalid config
    if mode == "C" and x1 > 8:
        raise RuntimeError("Simulated crash: invalid parameter combination")
    if resolution > 7 and x2 > 12:
        raise RuntimeError("Simulated crash: resource exceeded")
    
    # Shifted Branin per mode
    shift = {"A": 0, "B": 2, "C": -2}[mode]
    obj = branin(x1 + shift, x2) + resolution * 0.1
    
    # Constraints
    g1 = x1 + x2 - 14    # g1 <= 0
    g2 = -x1 + 2*x2 - 8  # g2 <= 0
    
    return {
        "objective_value": -obj,  # minimize branin = maximize negative
        "constraints": {"g1": g1, "g2": g2},
        "feasible": g1 <= 0 and g2 <= 0,
        "crashed": False,
    }
```

## RESULT SERIALIZATION

Every single evaluation must be saved. Use this format:

```python
# Save after every evaluation, not just at the end
# results/{scenario}/{optimizer}/seed_{seed}.jsonl

# Each line is one evaluation:
{
    "trial_id": 7,
    "config": {"model_name": "resnet50", "quantization": "int8_static", ...},
    "result": {"objective_value": 0.762, "feasible": true, "crashed": false, ...},
    "best_feasible_so_far": 0.758,
    "cumulative_crashes": 2,
    "cumulative_infeasible": 4,
    "wall_clock_s": 187.3,
    "timestamp": "2026-03-28T14:22:01Z"
}
```

## PLOTS TO GENERATE (in priority order)

1. **Regret curve**: x=trial number, y=best feasible objective so far, one line per optimizer, shaded ±std. THIS IS THE MOST IMPORTANT PLOT.
2. **Wasted budget bar chart**: fraction of budget wasted on crashes+infeasible per optimizer. THIS IS THE SECOND MOST IMPORTANT — it shows TBA's core advantage.
3. **Time to first feasible**: bar chart, lower is better.
4. **Final best feasible objective**: table with mean ± std per optimizer per scenario.
5. **Ablation table**: TBA variants showing which components matter.

Plots 1 and 2 are the whole story. Everything else is supporting evidence.

## ONE MORE THING: Testing

Write tests FIRST for the search space:
- test that sampling produces valid configs
- test that hierarchical conditions are respected
- test that neighbor proposals respect hierarchy
- test that distance computation handles mixed types
- test that the optimizer runs for N steps without crashing on the synthetic benchmark

If the search space has bugs, everything downstream is garbage.
