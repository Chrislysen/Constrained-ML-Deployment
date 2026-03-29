"""Deployment search space for image classification.

Variables:
  - model_name: categorical (5 models)
  - backend: categorical (pytorch_eager, torch_compile, onnxruntime)
  - quantization: categorical (fp32, fp16, int8_dynamic)
  - batch_size: integer (1-32, log scale)
  - num_threads: integer (1-8), conditional on backend != torch_compile

Known invalid combos exist — the optimizer must discover them:
  - int8_dynamic doesn't work on all backends (onnxruntime needs specific setup)
  - Large models + large batch sizes = OOM on 16GB GPU
  - torch_compile may fail on some models
  - vit_tiny + int8_dynamic may not work
"""
from __future__ import annotations

from tba.types import VariableDef
from tba.optimizer.search_space import SearchSpace


def image_classification_space() -> SearchSpace:
    """Build the deployment search space for image classification."""
    return SearchSpace([
        VariableDef(
            name="model_name",
            var_type="categorical",
            choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0", "vit_tiny"],
        ),
        VariableDef(
            name="backend",
            var_type="categorical",
            choices=["pytorch_eager", "torch_compile", "onnxruntime"],
        ),
        VariableDef(
            name="quantization",
            var_type="categorical",
            choices=["fp32", "fp16", "int8_dynamic"],
        ),
        VariableDef(
            name="batch_size",
            var_type="integer",
            low=1,
            high=32,
            log_scale=True,
        ),
        VariableDef(
            name="num_threads",
            var_type="integer",
            low=1,
            high=8,
            condition="backend in ['pytorch_eager', 'onnxruntime']",
        ),
    ])


# Constraint thresholds for different scenarios
SCENARIOS = {
    "edge_tight": {
        "constraints": {"latency_p95_ms": 20.0, "memory_peak_mb": 512.0},
        "objective": "maximize_accuracy",
        "budget": 25,
    },
    "server_throughput": {
        "constraints": {"latency_p99_ms": 100.0, "memory_peak_mb": 4096.0},
        "objective": "maximize_throughput",
        "budget": 25,
    },
    "ultra_low_budget": {
        "constraints": {"latency_p95_ms": 30.0, "memory_peak_mb": 1024.0},
        "objective": "maximize_accuracy",
        "budget": 15,
    },
}
