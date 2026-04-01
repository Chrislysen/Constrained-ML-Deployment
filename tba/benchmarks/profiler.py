"""Deployment profiler: measures latency, memory, accuracy, throughput.

CRITICAL: This function must NEVER crash the experiment runner.
Every possible failure returns a valid EvalResult with crashed=True.
Always cleans up GPU memory after every evaluation.
"""
from __future__ import annotations

import gc
import io
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets

from tba.types import EvalResult
from tba.benchmarks.model_zoo import load_model, get_input_size


# ImageNette class count (10 classes, mapped to ImageNet class indices)
IMAGENETTE_CLASSES = 10

# Fixed validation subset for consistent accuracy measurement
_VAL_SUBSET_SIZE = 500
_WARMUP_ITERS = 30
_BENCH_ITERS = 100

# Early-stop defaults: abort warmup if latency exceeds multiplier * constraint
EARLY_STOP_MULTIPLIER = 5
EARLY_STOP_WARMUP_ITERS = 3


def _get_val_loader(data_dir: str, batch_size: int, input_size: int = 224):
    """Load ImageNette validation set."""
    val_dir = Path(data_dir) / "val"
    if not val_dir.exists():
        # Try imagenette2-320 layout
        val_dir = Path(data_dir) / "imagenette2-320" / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation dir not found at {val_dir}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(val_dir), transform=transform)

    # Use a fixed subset for reproducibility
    n = min(_VAL_SUBSET_SIZE, len(dataset))
    indices = list(range(n))
    subset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return loader


def _apply_quantization(model: nn.Module, quant: str, device: str) -> nn.Module:
    """Apply quantization to a model."""
    if quant == "fp32":
        return model
    elif quant == "fp16":
        return model.half()
    elif quant == "int8_dynamic":
        # Dynamic quantization — CPU only
        model = model.cpu()
        model = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8,
        )
        return model
    else:
        raise ValueError(f"Unknown quantization: {quant}")


def _prepare_backend(
    model: nn.Module, config: dict[str, Any], device: str, input_size: int
) -> Any:
    """Prepare model for the specified backend. Returns (model_or_session, is_onnx)."""
    backend = config["backend"]
    batch_size = config["batch_size"]

    if backend == "pytorch_eager":
        if "num_threads" in config:
            torch.set_num_threads(config["num_threads"])
        return model.to(device), False

    elif backend == "torch_compile":
        model = model.to(device)
        compiled = torch.compile(model, mode="reduce-overhead")
        # Trigger compilation with a dummy forward
        dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
        if config["quantization"] == "fp16":
            dummy = dummy.half()
        with torch.no_grad():
            compiled(dummy)
        return compiled, False

    elif backend == "onnxruntime":
        import onnxruntime as ort

        # Export to ONNX in memory
        dummy = torch.randn(batch_size, 3, input_size, input_size)
        if config["quantization"] == "fp16":
            dummy = dummy.half()

        model_cpu = model.cpu()
        buf = io.BytesIO()
        torch.onnx.export(
            model_cpu, dummy, buf,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
        )
        buf.seek(0)

        # Create ORT session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        if "num_threads" in config:
            sess_opts.intra_op_num_threads = config["num_threads"]
        session = ort.InferenceSession(buf.read(), sess_opts, providers=providers)
        return session, True

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _benchmark_latency(
    model_or_session, is_onnx: bool, batch_size: int,
    input_size: int, device: str, quant: str,
    latency_constraint_ms: float | None = None,
    early_stop_multiplier: float = EARLY_STOP_MULTIPLIER,
    early_stop_warmup_iters: int = EARLY_STOP_WARMUP_ITERS,
) -> tuple[dict[str, float], bool]:
    """Measure latency over multiple iterations.

    Returns (latency_dict, early_stopped). If early_stopped is True,
    the latency dict contains partial warmup estimates.
    """
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
    if quant == "fp16":
        dummy = dummy.half()
    if quant == "int8_dynamic":
        dummy = dummy.cpu()
        device = "cpu"

    latencies = []
    early_stop_threshold = None
    if latency_constraint_ms is not None:
        early_stop_threshold = early_stop_multiplier * latency_constraint_ms

    if is_onnx:
        import onnxruntime as ort
        dummy_np = dummy.cpu().numpy()
        input_name = model_or_session.get_inputs()[0].name

        # Warmup with early-stop check
        warmup_start = time.perf_counter()
        for i in range(_WARMUP_ITERS):
            model_or_session.run(None, {input_name: dummy_np})
            if (early_stop_threshold is not None
                    and i == early_stop_warmup_iters - 1):
                elapsed_per_iter = (time.perf_counter() - warmup_start) / (i + 1) * 1000
                if elapsed_per_iter > early_stop_threshold:
                    return {
                        "latency_p50_ms": elapsed_per_iter,
                        "latency_p95_ms": elapsed_per_iter,
                        "latency_p99_ms": elapsed_per_iter,
                        "throughput_qps": float(batch_size * 1000.0 / elapsed_per_iter),
                    }, True

        # Benchmark
        for _ in range(_BENCH_ITERS):
            t0 = time.perf_counter()
            model_or_session.run(None, {input_name: dummy_np})
            latencies.append((time.perf_counter() - t0) * 1000)
    else:
        # Warmup with early-stop check
        warmup_start = time.perf_counter()
        with torch.no_grad():
            for i in range(_WARMUP_ITERS):
                model_or_session(dummy)
                if device == "cuda":
                    torch.cuda.synchronize()
                if (early_stop_threshold is not None
                        and i == early_stop_warmup_iters - 1):
                    elapsed_per_iter = (time.perf_counter() - warmup_start) / (i + 1) * 1000
                    if elapsed_per_iter > early_stop_threshold:
                        return {
                            "latency_p50_ms": elapsed_per_iter,
                            "latency_p95_ms": elapsed_per_iter,
                            "latency_p99_ms": elapsed_per_iter,
                            "throughput_qps": float(batch_size * 1000.0 / elapsed_per_iter),
                        }, True

        # Benchmark
        with torch.no_grad():
            for _ in range(_BENCH_ITERS):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model_or_session(dummy)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
        "throughput_qps": float(batch_size * 1000.0 / np.mean(arr)),
    }, False


def _measure_accuracy(
    model_or_session, is_onnx: bool, data_dir: str,
    batch_size: int, input_size: int, device: str, quant: str,
) -> float:
    """Measure top-1 accuracy on ImageNette validation subset."""
    loader = _get_val_loader(data_dir, batch_size, input_size)
    correct = 0
    total = 0

    for images, labels in loader:
        if is_onnx:
            if quant == "fp16":
                images = images.half()
            outputs_np = model_or_session.run(
                None, {model_or_session.get_inputs()[0].name: images.numpy()}
            )[0]
            preds = np.argmax(outputs_np, axis=1)
            correct += (preds == labels.numpy()).sum()
        else:
            images = images.to(device)
            if quant == "fp16":
                images = images.half()
            if quant == "int8_dynamic":
                images = images.cpu()
            with torch.no_grad():
                outputs = model_or_session(images)
            preds = outputs.argmax(dim=1)
            if quant == "int8_dynamic":
                correct += (preds == labels).sum().item()
            else:
                correct += (preds == labels.to(preds.device)).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def _measure_peak_memory(device: str) -> float:
    """Return peak GPU memory in MB, or 0 for CPU."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _cleanup(device: str) -> None:
    """Aggressively clean up GPU memory."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def evaluate_config(
    config: dict[str, Any],
    data_dir: str,
    device: str = "cuda",
    constraint_caps: dict[str, float] | None = None,
    enable_early_stop: bool = True,
    early_stop_multiplier: float = EARLY_STOP_MULTIPLIER,
    early_stop_warmup_iters: int = EARLY_STOP_WARMUP_ITERS,
) -> EvalResult:
    """Evaluate a deployment config. NEVER crashes the experiment runner.

    Steps:
      1. Load model
      2. Apply quantization
      3. Prepare backend (compile, export to ONNX, etc.)
      4. Warmup + benchmark latency (with optional early stop)
      5. Measure accuracy on validation subset
      6. Record peak memory
      7. Return result

    All wrapped in try/except. GPU memory cleaned after every eval.
    """
    start = time.time()
    result = EvalResult()

    try:
        # Reset GPU memory tracking
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        model_name = config["model_name"]
        quant = config["quantization"]
        backend = config["backend"]
        batch_size = config["batch_size"]
        input_size = get_input_size(model_name)

        # Determine actual device
        actual_device = device
        if quant == "int8_dynamic":
            actual_device = "cpu"  # int8 dynamic is CPU-only

        # Extract latency constraint for early stopping
        latency_constraint_ms = None
        if enable_early_stop and constraint_caps:
            # Use the tightest latency constraint available
            for key in ("latency_p95_ms", "latency_p99_ms"):
                if key in constraint_caps:
                    cap = constraint_caps[key]
                    if latency_constraint_ms is None or cap < latency_constraint_ms:
                        latency_constraint_ms = cap

        # 1. Load model
        model = load_model(model_name)

        # 2. Apply quantization
        model = _apply_quantization(model, quant, actual_device)

        # 3. Prepare backend
        model_or_session, is_onnx = _prepare_backend(model, config, actual_device, input_size)

        # 4. Benchmark latency (with early-stop support)
        latency_results, early_stopped = _benchmark_latency(
            model_or_session, is_onnx, batch_size, input_size, actual_device, quant,
            latency_constraint_ms=latency_constraint_ms,
            early_stop_multiplier=early_stop_multiplier,
            early_stop_warmup_iters=early_stop_warmup_iters,
        )

        if early_stopped:
            # This config will never meet constraints — skip accuracy measurement
            result.objective_value = 0.0
            result.constraints = {
                "latency_p95_ms": latency_results["latency_p95_ms"],
                "latency_p99_ms": latency_results["latency_p99_ms"],
                "memory_peak_mb": _measure_peak_memory(actual_device),
                "throughput_qps": latency_results["throughput_qps"],
            }
            result.feasible = False
            result.crashed = False
            result.early_stopped = True

            del model_or_session
            if not is_onnx:
                del model
            return result

        # 5. Measure accuracy
        accuracy = _measure_accuracy(
            model_or_session, is_onnx, data_dir, batch_size, input_size, actual_device, quant,
        )

        # 6. Peak memory
        memory_peak_mb = _measure_peak_memory(actual_device)

        # Build result
        result.objective_value = accuracy  # default: maximize accuracy
        result.constraints = {
            "latency_p95_ms": latency_results["latency_p95_ms"],
            "latency_p99_ms": latency_results["latency_p99_ms"],
            "memory_peak_mb": memory_peak_mb,
            "throughput_qps": latency_results["throughput_qps"],
        }

        # Check feasibility
        if constraint_caps:
            result.feasible = all(
                result.constraints.get(k, 0.0) <= v
                for k, v in constraint_caps.items()
            )
        else:
            result.feasible = True

        result.crashed = False

        # Clean up model reference
        del model_or_session
        if not is_onnx:
            del model

    except torch.cuda.OutOfMemoryError:
        result.crashed = True
        result.error_msg = "CUDA OOM"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        result.crashed = True
        result.error_msg = f"{type(e).__name__}: {str(e)[:300]}"

    finally:
        result.eval_time_s = time.time() - start
        _cleanup(device)

    return result
