#!/usr/bin/env python3
"""Stochastic feasibility boundary probe runner.

Runs each configuration many times under controlled conditions (cold/warm
interleaving) to measure latency stability near the 20ms p95 threshold.

Uses torch.cuda.Event for accurate GPU timing and records full latency
vectors plus GPU state for every repeat.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets

# Import from existing codebase
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tba.benchmarks.model_zoo import load_model, get_input_size
from tba.benchmarks.profiler import _apply_quantization, _prepare_backend


# ---------------------------------------------------------------------------
# GPU state monitoring
# ---------------------------------------------------------------------------

def get_gpu_state(gpu_id: int = 0) -> dict[str, Any]:
    """Query GPU state via nvidia-smi. Returns dict with temp, clock, memory, power."""
    state = {
        "temperature_c": None,
        "clock_mhz": None,
        "memory_used_mb": None,
        "power_draw_w": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=temperature.gpu,clocks.current.sm,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                state["temperature_c"] = int(parts[0])
                state["clock_mhz"] = int(parts[1])
                state["memory_used_mb"] = float(parts[2])
                state["power_draw_w"] = float(parts[3])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return state


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fixed_batch(
    data_dir: str, batch_size: int, input_size: int = 224
) -> torch.Tensor:
    """Load a fixed batch of images from ImageNette validation set.

    Returns a single batch tensor on CPU. The same batch is used for every
    repeat of every config to eliminate data variability.
    """
    val_dir = Path(data_dir) / "val"
    if not val_dir.exists():
        val_dir = Path(data_dir) / "imagenette2-320" / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation dir not found: {val_dir}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(val_dir), transform=transform)

    # Take the first batch_size images deterministically
    images = []
    for i in range(min(batch_size, len(dataset))):
        img, _ = dataset[i]
        images.append(img)

    batch = torch.stack(images)
    return batch


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def time_inference_cuda(
    model_or_session: Any,
    is_onnx: bool,
    batch: torch.Tensor,
    quant: str,
    device: str,
    num_passes: int,
    timeout_s: float = 60.0,
) -> tuple[list[float], bool, str | None]:
    """Run timed inference passes using CUDA events for GPU timing.

    Returns (latencies_ms, timed_out, error_msg).
    """
    latencies = []

    if is_onnx:
        # ONNX Runtime uses its own timing — no CUDA events available
        import onnxruntime  # noqa: F401
        input_name = model_or_session.get_inputs()[0].name
        batch_np = batch.cpu().numpy()
        if quant == "fp16":
            batch_np = batch_np.astype(np.float16)

        for i in range(num_passes):
            t0 = time.perf_counter()
            try:
                model_or_session.run(None, {input_name: batch_np})
            except Exception as e:
                return latencies, False, f"ORT inference error pass {i}: {e}"
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            if elapsed_ms > timeout_s * 1000.0:
                return latencies, True, f"Timeout at pass {i}: {elapsed_ms:.1f}ms"
    else:
        # PyTorch: use CUDA events for accurate GPU timing
        use_cuda = device == "cuda" and torch.cuda.is_available()

        for i in range(num_passes):
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                gpu_batch = batch.to(device)
                if quant == "fp16":
                    gpu_batch = gpu_batch.half()

                torch.cuda.synchronize()
                start_event.record()
                try:
                    with torch.no_grad():
                        model_or_session(gpu_batch)
                except Exception as e:
                    return latencies, False, f"Inference error pass {i}: {e}"
                end_event.record()
                torch.cuda.synchronize()

                elapsed_ms = start_event.elapsed_time(end_event)
                del gpu_batch
            else:
                # CPU path (int8_dynamic)
                cpu_batch = batch.clone()
                if quant == "fp16":
                    cpu_batch = cpu_batch.half()

                t0 = time.perf_counter()
                try:
                    with torch.no_grad():
                        model_or_session(cpu_batch)
                except Exception as e:
                    return latencies, False, f"Inference error pass {i}: {e}"
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            latencies.append(elapsed_ms)
            if elapsed_ms > timeout_s * 1000.0:
                return latencies, True, f"Timeout at pass {i}: {elapsed_ms:.1f}ms"

    return latencies, False, None


def run_warmup(
    model_or_session: Any,
    is_onnx: bool,
    batch: torch.Tensor,
    quant: str,
    device: str,
    num_passes: int,
) -> None:
    """Run warmup passes (results discarded)."""
    if is_onnx:
        input_name = model_or_session.get_inputs()[0].name
        batch_np = batch.cpu().numpy()
        if quant == "fp16":
            batch_np = batch_np.astype(np.float16)
        for _ in range(num_passes):
            model_or_session.run(None, {input_name: batch_np})
    else:
        use_cuda = device == "cuda" and torch.cuda.is_available()
        gpu_batch = batch.to(device)
        if quant == "fp16":
            gpu_batch = gpu_batch.half()
        with torch.no_grad():
            for _ in range(num_passes):
                model_or_session(gpu_batch)
                if use_cuda:
                    torch.cuda.synchronize()
        del gpu_batch


def compute_summary(latencies: list[float]) -> dict[str, float]:
    """Compute summary statistics from a latency vector."""
    arr = np.array(latencies)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


# ---------------------------------------------------------------------------
# Main probe loop
# ---------------------------------------------------------------------------

def probe_config(
    config: dict[str, Any],
    batch_cpu: torch.Tensor,
    device: str,
    repeats: int,
    warmup_passes: int,
    measurement_passes: int,
    cold_sleep: float,
    gpu_id: int,
) -> list[dict]:
    """Run all repeats for a single configuration.

    Alternates cold/warm: repeat 0 = cold, 1 = warm, 2 = cold, ...
    Model is loaded once and reused across all repeats.
    """
    model_name = config["model"]
    backend = config["backend"]
    quant = config["quantization"]
    batch_size = config["batch_size"]

    # Build config dict in the format expected by existing codebase
    eval_config = {
        "model_name": model_name,
        "backend": backend,
        "quantization": quant,
        "batch_size": batch_size,
    }

    input_size = get_input_size(model_name)
    actual_device = device
    if quant == "int8_dynamic":
        actual_device = "cpu"

    # Prepare batch: take first batch_size images from the fixed batch
    if batch_size <= batch_cpu.shape[0]:
        batch = batch_cpu[:batch_size]
    else:
        # Repeat to fill batch size
        repeats_needed = (batch_size + batch_cpu.shape[0] - 1) // batch_cpu.shape[0]
        batch = batch_cpu.repeat(repeats_needed, 1, 1, 1)[:batch_size]

    # Load model once for all repeats
    results = []
    model = None
    model_or_session = None
    is_onnx = False
    load_error = None

    try:
        model = load_model(model_name)
        model = _apply_quantization(model, quant, actual_device)
        model_or_session, is_onnx = _prepare_backend(model, eval_config, actual_device, input_size)
    except Exception as e:
        load_error = f"{type(e).__name__}: {str(e)[:300]}"

    for repeat_idx in range(repeats):
        condition = "cold" if repeat_idx % 2 == 0 else "warm"

        # Cold: clear cache, sleep, re-warmup
        if condition == "cold" and torch.cuda.is_available() and actual_device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(cold_sleep)

        # Warm: just synchronize
        if condition == "warm" and torch.cuda.is_available() and actual_device == "cuda":
            torch.cuda.synchronize()

        gpu_state = get_gpu_state(gpu_id)

        # Handle load failure
        if load_error is not None:
            record = {
                "config": config,
                "repeat_index": repeat_idx,
                "condition": condition,
                "gpu_state": gpu_state,
                "warmup_passes": warmup_passes,
                "measurement_passes": measurement_passes,
                "latencies_ms": [],
                "summary": {},
                "memory_peak_mb": 0.0,
                "feasible": False,
                "crash": True,
                "crash_reason": load_error,
            }
            results.append(record)
            continue

        # Reset peak memory tracking
        if actual_device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        crash = False
        crash_reason = None
        latencies = []
        timed_out = False

        try:
            # Warmup
            run_warmup(model_or_session, is_onnx, batch, quant, actual_device, warmup_passes)

            # Measurement
            latencies, timed_out, err_msg = time_inference_cuda(
                model_or_session, is_onnx, batch, quant, actual_device, measurement_passes,
            )
            if err_msg and not timed_out:
                crash = True
                crash_reason = err_msg
            elif timed_out:
                crash = False
                crash_reason = err_msg  # record but not a crash

        except torch.cuda.OutOfMemoryError:
            crash = True
            crash_reason = "CUDA OOM"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            crash = True
            crash_reason = f"{type(e).__name__}: {str(e)[:300]}"

        # Compute summary and feasibility
        summary = compute_summary(latencies) if latencies else {}
        p95 = summary.get("p95", float("inf"))
        feasible = p95 <= 20.0 and not crash

        memory_peak_mb = 0.0
        if actual_device == "cuda" and torch.cuda.is_available():
            memory_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

        record = {
            "config": config,
            "repeat_index": repeat_idx,
            "condition": condition,
            "gpu_state": gpu_state,
            "warmup_passes": warmup_passes,
            "measurement_passes": measurement_passes,
            "latencies_ms": [round(l, 4) for l in latencies],
            "summary": summary,
            "memory_peak_mb": memory_peak_mb,
            "feasible": feasible,
            "crash": crash,
            "crash_reason": crash_reason,
        }
        results.append(record)

    # Cleanup
    del model_or_session
    if model is not None and not is_onnx:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return results


def run_single_config(args, config, config_index: int, output_file: Path):
    """Run probe for a single config (used in subprocess mode)."""
    import tempfile

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    batch_cpu = load_fixed_batch(args.dataset_path, config["batch_size"])

    records = probe_config(
        config=config,
        batch_cpu=batch_cpu,
        device="cuda" if torch.cuda.is_available() else "cpu",
        repeats=args.repeats,
        warmup_passes=args.warmup_passes,
        measurement_passes=args.measurement_passes,
        cold_sleep=args.cold_sleep,
        gpu_id=args.gpu_id,
    )

    with open(output_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Stochastic feasibility boundary probe runner"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="JSON file with boundary configs (from select_boundary_configs.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/boundary_probe"),
        help="Output directory for probe results (default: results/boundary_probe/)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Total repeats per config (default: 50, alternating cold/warm)",
    )
    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=20,
        help="Warmup inference passes per repeat (default: 20)",
    )
    parser.add_argument(
        "--measurement-passes",
        type=int,
        default=200,
        help="Timed inference passes per repeat (default: 200)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data",
        help="Path to ImageNette dataset directory (default: data)",
    )
    parser.add_argument(
        "--cold-sleep",
        type=float,
        default=5.0,
        help="Seconds to sleep between cold repeats (default: 5.0)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Limit number of configs to probe (for testing)",
    )
    parser.add_argument(
        "--skip-configs",
        type=int,
        default=0,
        help="Skip first N configs (for resuming interrupted runs)",
    )
    parser.add_argument(
        "--append-to",
        type=Path,
        default=None,
        help="Append results to existing JSONL file instead of creating a new one",
    )
    parser.add_argument(
        "--run-single-index",
        type=int,
        default=None,
        help="Run only config at this index, write to --single-output (subprocess mode)",
    )
    parser.add_argument(
        "--single-output",
        type=Path,
        default=None,
        help="Output file for single-config mode (used with --run-single-index)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.config_file.exists():
        print(f"ERROR: Config file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.config_file) as f:
        config_data = json.load(f)

    configs = config_data.get("configs", [])
    if args.max_configs:
        configs = configs[:args.max_configs]

    if not configs:
        print("ERROR: No configs found in config file", file=sys.stderr)
        sys.exit(1)

    # Single-config subprocess mode
    if args.run_single_index is not None:
        idx = args.run_single_index
        if idx >= len(configs):
            print(f"ERROR: Config index {idx} out of range (have {len(configs)})", file=sys.stderr)
            sys.exit(1)
        out = args.single_output or Path(f"_probe_single_{idx}.jsonl")
        run_single_config(args, configs[idx], idx, out)
        sys.exit(0)

    # --- Main orchestrator mode: dispatch each config as a subprocess ---
    skip = args.skip_configs
    configs = configs[skip:]
    print(f"Loaded {len(configs) + skip} configs from {args.config_file}")
    if skip:
        print(f"Skipping first {skip} configs (resuming)")
    print(f"Running {len(configs)} configs")
    print(f"Repeats per config: {args.repeats} ({args.repeats // 2} cold, {args.repeats - args.repeats // 2} warm)")
    print(f"Warmup passes: {args.warmup_passes}, Measurement passes: {args.measurement_passes}")
    print("Mode: subprocess isolation (each config runs in its own process)")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        gpu_name = torch.cuda.get_device_name(args.gpu_id)
        print(f"GPU: {gpu_name} (device {args.gpu_id})")
    else:
        print("WARNING: CUDA not available, running on CPU only")

    # Prepare output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.append_to and args.append_to.exists():
        output_file = args.append_to
        file_mode = "a"
        print(f"Appending to: {output_file}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output_dir / f"probe_{timestamp}.jsonl"
        file_mode = "w"
        print(f"Output: {output_file}")

    print()
    total_repeats = len(configs) * args.repeats
    print(f"Total repeats: {total_repeats}")
    print("=" * 70)

    global_start = time.time()
    total_configs_done = 0
    script_path = Path(__file__).resolve()

    with open(output_file, file_mode) as out_f:
        for cfg_idx in range(len(configs)):
            config = configs[cfg_idx]
            abs_idx = cfg_idx + skip  # absolute index into the full config list
            cfg_start = time.time()
            label = (
                f"{config['model']}/{config['backend']}/"
                f"{config['quantization']}/bs{config['batch_size']}"
            )
            print(f"\n[{cfg_idx + 1}/{len(configs)}] {label}")

            # Temp file for subprocess output
            tmp_out = args.output_dir / f"_tmp_config_{abs_idx}.jsonl"

            # Build subprocess command — child sees the full config list,
            # so pass the absolute index
            cmd = [
                sys.executable, "-u", str(script_path),
                "--config-file", str(args.config_file),
                "--output-dir", str(args.output_dir),
                "--repeats", str(args.repeats),
                "--warmup-passes", str(args.warmup_passes),
                "--measurement-passes", str(args.measurement_passes),
                "--dataset-path", str(args.dataset_path),
                "--cold-sleep", str(args.cold_sleep),
                "--gpu-id", str(args.gpu_id),
                "--run-single-index", str(abs_idx),
                "--single-output", str(tmp_out),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            # Read results from subprocess
            records = []
            if tmp_out.exists():
                with open(tmp_out) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                            out_f.write(line + "\n")
                out_f.flush()
                tmp_out.unlink()

            total_configs_done += 1
            cfg_elapsed = time.time() - cfg_start
            total_elapsed = time.time() - global_start

            if result.returncode != 0 and not records:
                # Fatal crash — record it
                gpu_state = get_gpu_state(args.gpu_id)
                crash_reason = f"Subprocess crashed (exit code {result.returncode})"
                stderr_tail = (result.stderr or "")[-500:]
                if stderr_tail:
                    crash_reason += f": {stderr_tail}"
                for repeat_idx in range(args.repeats):
                    rec = {
                        "config": config,
                        "repeat_index": repeat_idx,
                        "condition": "cold" if repeat_idx % 2 == 0 else "warm",
                        "gpu_state": gpu_state,
                        "warmup_passes": args.warmup_passes,
                        "measurement_passes": args.measurement_passes,
                        "latencies_ms": [],
                        "summary": {},
                        "memory_peak_mb": 0.0,
                        "feasible": False,
                        "crash": True,
                        "crash_reason": crash_reason,
                    }
                    records.append(rec)
                    out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                print(f"  FATAL CRASH (exit {result.returncode}) | {cfg_elapsed:.1f}s")
            elif records:
                feasible_count = sum(1 for r in records if r["feasible"])
                crash_count = sum(1 for r in records if r["crash"])
                latency_summaries = [r["summary"].get("p95", 0) for r in records if r["summary"]]
                if latency_summaries:
                    avg_p95 = np.mean(latency_summaries)
                    std_p95 = np.std(latency_summaries)
                    flip_rate = 1.0 - max(feasible_count, args.repeats - feasible_count) / args.repeats
                    print(
                        f"  p95: {avg_p95:.2f} +/- {std_p95:.2f}ms | "
                        f"feasible: {feasible_count}/{args.repeats} | "
                        f"flip_rate: {flip_rate:.1%} | "
                        f"crashes: {crash_count} | "
                        f"{cfg_elapsed:.1f}s"
                    )
                else:
                    print(f"  All crashed ({crash_count} crashes) | {cfg_elapsed:.1f}s")

            # ETA
            avg_per_config = total_elapsed / total_configs_done
            remaining = (len(configs) - total_configs_done) * avg_per_config
            print(f"  ETA: {remaining / 60:.1f} min remaining")

    total_time = time.time() - global_start
    print(f"\n{'=' * 70}")
    print(f"Done. {total_configs_done} configs probed in {total_time / 60:.1f} min")
    print(f"Results: {output_file}")


if __name__ == "__main__":
    main()
