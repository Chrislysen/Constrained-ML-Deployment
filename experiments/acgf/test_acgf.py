"""Unit tests for the ACGF evaluator.

Tests the algorithmic core without requiring GPU/models — mocks
evaluate_config to return controlled EvalResult values.

Run:  python -m pytest experiments/acgf/test_acgf.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tba.types import EvalResult
from experiments.acgf.acgf_evaluator import (
    ACGFParams,
    ACGFResult,
    compute_adaptive_k,
    evaluate_with_acgf,
)


# ── compute_adaptive_k tests ─────────────────────────────────────────


class TestComputeAdaptiveK:
    def test_on_boundary_gets_max(self):
        """Distance=0 should give max_reeval."""
        assert compute_adaptive_k(0.0, 3.0, 5) == 5

    def test_at_edge_gets_one(self):
        """Distance just inside band edge should give 1."""
        assert compute_adaptive_k(2.9, 3.0, 5) == 1

    def test_outside_band_gets_zero(self):
        """Distance >= band_width should give 0."""
        assert compute_adaptive_k(3.0, 3.0, 5) == 0
        assert compute_adaptive_k(5.0, 3.0, 5) == 0

    def test_negative_distance(self):
        """Negative distance (infeasible side) works symmetrically."""
        assert compute_adaptive_k(-0.0, 3.0, 5) == 5
        assert compute_adaptive_k(-2.9, 3.0, 5) == 1
        assert compute_adaptive_k(-3.0, 3.0, 5) == 0

    def test_halfway_point(self):
        """Distance at half the band should give ~half of max_reeval."""
        k = compute_adaptive_k(1.5, 3.0, 5)
        assert 2 <= k <= 3  # ceil(5 * 0.5) = 3

    def test_zero_band_gives_zero(self):
        assert compute_adaptive_k(0.0, 0.0, 5) == 0

    def test_zero_max_reeval(self):
        assert compute_adaptive_k(0.0, 3.0, 0) == 0

    def test_monotonicity(self):
        """Closer to boundary → more re-evaluations."""
        ks = [compute_adaptive_k(d, 3.0, 5) for d in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]]
        for i in range(len(ks) - 1):
            assert ks[i] >= ks[i + 1], f"K should decrease: {ks}"


# ── evaluate_with_acgf tests (mocked) ────────────────────────────────


def _make_result(p95: float, feasible: bool, crashed: bool = False) -> EvalResult:
    return EvalResult(
        objective_value=0.85 if not crashed else float("-inf"),
        constraints={
            "latency_p95_ms": p95,
            "latency_p99_ms": p95 * 1.1,
            "memory_peak_mb": 200.0,
            "throughput_qps": 100.0,
        },
        feasible=feasible,
        crashed=crashed,
        eval_time_s=5.0,
    )


CONSTRAINT_CAPS = {"latency_p95_ms": 20.0, "memory_peak_mb": 512.0}
DUMMY_CONFIG = {"model_name": "resnet18", "backend": "pytorch_eager",
                "quantization": "fp32", "batch_size": 1}


class TestEvaluateWithACGF:
    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=45.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_clearly_feasible_single_eval(self, mock_eval, mock_temp):
        """Config well below threshold: single eval, not borderline."""
        mock_eval.return_value = _make_result(15.0, True)

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
            params=ACGFParams(band_width_ms=3.0),
        )
        assert result.evaluations_used == 1
        assert result.is_borderline is False
        assert mock_eval.call_count == 1

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=45.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_clearly_infeasible_single_eval(self, mock_eval, mock_temp):
        """Config well above threshold: single eval."""
        mock_eval.return_value = _make_result(28.0, False)

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
            params=ACGFParams(band_width_ms=3.0),
        )
        assert result.evaluations_used == 1
        assert result.is_borderline is False

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=50.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_borderline_triggers_reeval(self, mock_eval, mock_temp):
        """Config at 19.5ms (distance=0.5, within 3ms band): triggers re-eval."""
        mock_eval.return_value = _make_result(19.5, True)

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
            params=ACGFParams(band_width_ms=3.0, max_reeval=5),
        )
        assert result.is_borderline is True
        assert result.evaluations_used > 1
        # K = ceil(5 * (1 - 0.5/3.0)) = ceil(5 * 0.833) = ceil(4.17) = 5
        assert result.evaluations_used == 6  # 1 initial + 5 re-evals

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=50.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_crash_returns_immediately(self, mock_eval, mock_temp):
        """Crashed config: single eval, no re-evaluation."""
        mock_eval.return_value = _make_result(0.0, False, crashed=True)

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
        )
        assert result.evaluations_used == 1
        assert result.is_borderline is False
        assert result.feasibility_probability == 0.0

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=50.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_budget_cap_limits_reeval(self, mock_eval, mock_temp):
        """Re-evaluations capped by remaining budget."""
        mock_eval.return_value = _make_result(19.5, True)

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
            params=ACGFParams(band_width_ms=3.0, max_reeval=5),
            budget_remaining=3,  # only 2 re-evals possible
        )
        assert result.evaluations_used == 3  # 1 + 2

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=50.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_rescue_detection(self, mock_eval, mock_temp):
        """Initially infeasible but re-evals show feasible → rescued."""
        # First call: infeasible (p95=20.5)
        # Re-evals: feasible (p95=19.0)
        results = [
            _make_result(20.5, False),  # initial: infeasible
            _make_result(19.0, True),   # re-eval 1: feasible
            _make_result(19.2, True),   # re-eval 2: feasible
            _make_result(18.8, True),   # re-eval 3: feasible
            _make_result(19.1, True),   # re-eval 4: feasible
            _make_result(19.0, True),   # re-eval 5: feasible
        ]
        mock_eval.side_effect = results

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
            params=ACGFParams(band_width_ms=3.0, max_reeval=5, confidence_threshold=0.7),
        )
        assert result.was_rescued is True
        assert result.final_result.feasible is True
        # 5/6 feasible = 0.833 >= 0.7
        assert result.feasibility_probability is not None
        assert result.feasibility_probability > 0.7

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=50.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_catch_detection(self, mock_eval, mock_temp):
        """Initially feasible but re-evals show mostly infeasible → caught."""
        results = [
            _make_result(19.8, True),   # initial: feasible (barely)
            _make_result(20.5, False),  # re-eval 1: infeasible
            _make_result(21.0, False),  # re-eval 2: infeasible
            _make_result(20.3, False),  # re-eval 3: infeasible
            _make_result(20.8, False),  # re-eval 4: infeasible
            _make_result(19.9, True),   # re-eval 5: feasible
        ]
        mock_eval.side_effect = results

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
            params=ACGFParams(band_width_ms=3.0, max_reeval=5, confidence_threshold=0.7),
        )
        assert result.was_caught is True
        assert result.final_result.feasible is False
        # 2/6 feasible = 0.333 < 0.7

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=50.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_fixed_k_overrides_adaptive(self, mock_eval, mock_temp):
        """fixed_k parameter should override adaptive K computation."""
        mock_eval.return_value = _make_result(19.5, True)

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
            params=ACGFParams(band_width_ms=3.0, max_reeval=5, fixed_k=2),
        )
        assert result.evaluations_used == 3  # 1 + fixed_k=2

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=50.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_no_latency_constraint_falls_through(self, mock_eval, mock_temp):
        """No latency constraint → single eval, no ACGF logic."""
        mock_eval.return_value = _make_result(19.5, True)

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data",
            constraint_caps={"memory_peak_mb": 512.0},  # no latency
        )
        assert result.evaluations_used == 1
        assert result.is_borderline is False

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=50.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_median_latency_in_final_result(self, mock_eval, mock_temp):
        """Final result should use median p95 across all evaluations."""
        results = [
            _make_result(19.0, True),
            _make_result(19.5, True),
            _make_result(20.5, False),
        ]
        mock_eval.side_effect = results

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
            params=ACGFParams(band_width_ms=3.0, max_reeval=5, fixed_k=2),
        )
        # Median of [19.0, 19.5, 20.5] = 19.5
        assert abs(result.final_result.constraints["latency_p95_ms"] - 19.5) < 0.01

    @patch("experiments.acgf.acgf_evaluator.get_gpu_temperature", return_value=50.0)
    @patch("experiments.acgf.acgf_evaluator.evaluate_config")
    def test_total_eval_time_accumulated(self, mock_eval, mock_temp):
        """Final result eval_time_s should be sum of all evaluations."""
        r1 = _make_result(19.5, True)
        r1.eval_time_s = 5.0
        r2 = _make_result(19.0, True)
        r2.eval_time_s = 4.5
        mock_eval.side_effect = [r1, r2]

        result = evaluate_with_acgf(
            DUMMY_CONFIG, "data", CONSTRAINT_CAPS,
            params=ACGFParams(band_width_ms=3.0, fixed_k=1),
        )
        assert abs(result.final_result.eval_time_s - 9.5) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
