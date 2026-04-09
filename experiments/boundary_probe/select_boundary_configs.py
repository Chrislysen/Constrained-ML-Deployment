#!/usr/bin/env python3
"""Select configurations near the feasibility boundary for repeat-run probing.

Scans existing DeployBench JSONL result files and classifies configs by their
distance to the 20ms p95 latency threshold. Outputs a ranked candidate list
for the boundary probe experiment.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


# Zone thresholds (ms) relative to the 20ms p95 constraint
ZONES = {
    "DEEP_FEASIBLE":    (0.0,  15.0),
    "NEAR_FEASIBLE":    (15.0, 19.0),
    "GRAY_ZONE":        (19.0, 22.0),
    "NEAR_INFEASIBLE":  (22.0, 26.0),
    "DEEP_INFEASIBLE":  (26.0, float("inf")),
}

LATENCY_THRESHOLD_MS = 20.0
MEMORY_CAP_MB = 512.0

# Target counts per zone
ZONE_TARGETS = {
    "GRAY_ZONE": 10,
    "NEAR_FEASIBLE": 5,
    "NEAR_INFEASIBLE": 5,
    "DEEP_FEASIBLE": 3,
    "DEEP_INFEASIBLE": 3,
}


def classify_zone(p95_latency_ms: float) -> str:
    """Classify a single latency observation into a zone."""
    for zone_name, (lo, hi) in ZONES.items():
        if lo <= p95_latency_ms < hi:
            return zone_name
    return "DEEP_INFEASIBLE"


def parse_results(result_dirs: list[Path]) -> dict[str, list[dict]]:
    """Parse all JSONL result files and group observations by config key.

    Returns a dict mapping config_key -> list of observation dicts.
    Each observation has: p95_latency_ms, memory_peak_mb, feasible, crashed, source_file.
    """
    configs: dict[str, list[dict]] = defaultdict(list)

    for result_dir in result_dirs:
        for jsonl_path in sorted(result_dir.rglob("*.jsonl")):
            for line_num, line in enumerate(jsonl_path.open(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                result = rec.get("result", {})
                config = rec.get("config", {})

                # Extract the 4 core config fields
                model = config.get("model_name", "")
                backend = config.get("backend", "")
                quant = config.get("quantization", "")
                batch_size = config.get("batch_size", 0)

                if not all([model, backend, quant, batch_size]):
                    continue

                key = f"{model}|{backend}|{quant}|{batch_size}"

                crashed = result.get("crashed", False)
                early_stopped = result.get("early_stopped", False)
                constraints = result.get("constraints", {})

                obs = {
                    "p95_latency_ms": constraints.get("latency_p95_ms"),
                    "memory_peak_mb": constraints.get("memory_peak_mb", 0.0),
                    "feasible": result.get("feasible", False),
                    "crashed": crashed,
                    "early_stopped": early_stopped,
                    "source_file": str(jsonl_path),
                }

                configs[key].append(obs)

    return configs


def build_candidate_list(
    configs: dict[str, list[dict]],
) -> list[dict]:
    """Build ranked candidate list from parsed config observations."""
    candidates = []

    for key, observations in configs.items():
        model, backend, quant, batch_size = key.split("|")
        batch_size = int(batch_size)

        # Filter to non-crashed, non-early-stopped observations with valid latency
        valid_obs = [
            o for o in observations
            if not o["crashed"]
            and not o["early_stopped"]
            and o["p95_latency_ms"] is not None
        ]

        if not valid_obs:
            continue

        latencies = [o["p95_latency_ms"] for o in valid_obs]
        memories = [o["memory_peak_mb"] for o in valid_obs]

        # Skip configs that consistently violate memory constraint
        if all(m > MEMORY_CAP_MB for m in memories):
            continue

        median_latency = statistics.median(latencies)
        zone = classify_zone(median_latency)

        # Check if config was observed as both feasible and infeasible
        feasibility_values = set()
        for o in valid_obs:
            is_feasible = o["p95_latency_ms"] <= LATENCY_THRESHOLD_MS
            feasibility_values.add(is_feasible)
        observed_flip = len(feasibility_values) > 1

        candidate = {
            "model": model,
            "backend": backend,
            "quantization": quant,
            "batch_size": batch_size,
            "zone": zone,
            "median_p95_latency_ms": round(median_latency, 3),
            "min_p95_latency_ms": round(min(latencies), 3),
            "max_p95_latency_ms": round(max(latencies), 3),
            "latency_std_ms": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0.0,
            "median_memory_mb": round(statistics.median(memories), 1),
            "num_observations": len(valid_obs),
            "observed_flip": observed_flip,
            "distance_to_threshold_ms": round(median_latency - LATENCY_THRESHOLD_MS, 3),
        }
        candidates.append(candidate)

    return candidates


def select_configs(candidates: list[dict]) -> list[dict]:
    """Select a balanced set of ~20-30 configs across zones."""
    by_zone: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_zone[c["zone"]].append(c)

    # Sort within each zone: prioritize flips, then closeness to threshold
    for zone_name in by_zone:
        by_zone[zone_name].sort(
            key=lambda c: (not c["observed_flip"], abs(c["distance_to_threshold_ms"]))
        )

    selected = []
    for zone_name, target in ZONE_TARGETS.items():
        available = by_zone.get(zone_name, [])
        selected.extend(available[:target])

    # If we have fewer configs than desired, be more permissive
    if len(selected) < 15:
        # Grab more from any zone that has extras
        already_selected = {
            (c["model"], c["backend"], c["quantization"], c["batch_size"])
            for c in selected
        }
        for zone_name in ["GRAY_ZONE", "NEAR_FEASIBLE", "NEAR_INFEASIBLE"]:
            for c in by_zone.get(zone_name, []):
                key = (c["model"], c["backend"], c["quantization"], c["batch_size"])
                if key not in already_selected:
                    selected.append(c)
                    already_selected.add(key)
                    if len(selected) >= 26:
                        break

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Select boundary configs for stochastic feasibility probe"
    )
    parser.add_argument(
        "--result-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing JSONL result files from prior DeployBench runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("boundary_configs.json"),
        help="Output JSON file (default: boundary_configs.json)",
    )
    args = parser.parse_args()

    # Validate input dirs
    for d in args.result_dirs:
        if not d.exists():
            print(f"ERROR: Result directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    print(f"Scanning {len(args.result_dirs)} result directories...")
    configs = parse_results(args.result_dirs)
    print(f"Found {len(configs)} unique configurations across all results")

    candidates = build_candidate_list(configs)
    print(f"Valid candidates (non-crashed, within memory): {len(candidates)}")

    # Print zone distribution
    zone_counts = defaultdict(int)
    for c in candidates:
        zone_counts[c["zone"]] += 1
    print("\nZone distribution:")
    for zone_name in ZONES:
        count = zone_counts.get(zone_name, 0)
        print(f"  {zone_name:20s}: {count}")

    # Count observed flips
    flips = [c for c in candidates if c["observed_flip"]]
    print(f"\nConfigs with observed feasibility flips: {len(flips)}")
    for c in flips:
        print(
            f"  {c['model']:20s} | {c['backend']:15s} | {c['quantization']:15s} | "
            f"bs={c['batch_size']:2d} | {c['min_p95_latency_ms']:.1f}-{c['max_p95_latency_ms']:.1f}ms"
        )

    # Warn if data is sparse
    single_obs = sum(1 for c in candidates if c["num_observations"] == 1)
    if single_obs > len(candidates) * 0.7:
        print(
            f"\nWARNING: {single_obs}/{len(candidates)} configs have only 1 observation. "
            "Zone classification is approximate."
        )

    selected = select_configs(candidates)
    print(f"\nSelected {len(selected)} configs for boundary probe:")

    zone_selected = defaultdict(int)
    for c in selected:
        zone_selected[c["zone"]] += 1
    for zone_name in ZONES:
        count = zone_selected.get(zone_name, 0)
        target = ZONE_TARGETS.get(zone_name, "?")
        print(f"  {zone_name:20s}: {count} (target: {target})")

    # Write output
    output = {
        "description": "Boundary probe candidate configs",
        "latency_threshold_ms": LATENCY_THRESHOLD_MS,
        "memory_cap_mb": MEMORY_CAP_MB,
        "total_candidates_scanned": len(candidates),
        "total_selected": len(selected),
        "configs": selected,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
