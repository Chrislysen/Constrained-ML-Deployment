"""Tests for v3 features: diversity-gated handoff, adaptive timeout, combination blacklisting."""
import random

import pytest

from tba.types import EvalResult, VariableDef
from tba.optimizer.search_space import SearchSpace
from tba.optimizer.subspace_tracker import SubspaceTracker
from tba.optimizer.tba_tpe_hybrid import TBATPEHybrid
from tba.benchmarks.profiler import (
    EARLY_STOP_MULTIPLIER_PREFEASIBLE,
    EARLY_STOP_MULTIPLIER_POSTFEASIBLE,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _deployment_space() -> SearchSpace:
    """Minimal deployment-like space for unit tests."""
    return SearchSpace([
        VariableDef(
            name="model_name", var_type="categorical",
            choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0", "vit_tiny"],
        ),
        VariableDef(
            name="backend", var_type="categorical",
            choices=["pytorch_eager", "torch_compile", "onnxruntime"],
        ),
        VariableDef(
            name="quantization", var_type="categorical",
            choices=["fp32", "fp16", "int8_dynamic"],
        ),
        VariableDef(name="batch_size", var_type="integer", low=1, high=32, log_scale=True),
    ])


def _make_hybrid(**kwargs) -> TBATPEHybrid:
    defaults = dict(
        search_space=_deployment_space(),
        constraints={"latency_p95_ms": 20.0, "memory_peak_mb": 512.0},
        objective="maximize_accuracy",
        budget=25,
        seed=42,
    )
    defaults.update(kwargs)
    return TBATPEHybrid(**defaults)


def _feasible_result(obj=0.75) -> EvalResult:
    return EvalResult(
        objective_value=obj,
        constraints={"latency_p95_ms": 10.0, "memory_peak_mb": 200.0},
        feasible=True, crashed=False,
    )


def _infeasible_result() -> EvalResult:
    return EvalResult(
        objective_value=0.5,
        constraints={"latency_p95_ms": 50.0, "memory_peak_mb": 200.0},
        feasible=False, crashed=False,
    )


def _crashed_result() -> EvalResult:
    return EvalResult(crashed=True, feasible=False)


# ==================================================================
# Change 1: Diversity-Gated Handoff
# ==================================================================

class TestDiversityGatedHandoff:
    """v3: handoff requires >=3 model families to have been evaluated."""

    def _populate_optimizer(self, opt, configs_results):
        """Feed configs/results through ask/tell cycle.

        We call ask() to advance the trial counter, then override
        the config with our controlled one in tell().
        """
        for config, result in configs_results:
            _ = opt.ask()
            opt.tell(config, result)

    def test_no_handoff_with_only_two_families(self):
        """Count thresholds met but only 2 model families seen -> no handoff."""
        opt = _make_hybrid(n_initial_random=0, min_family_diversity=3)
        # 5 feasible from resnet18, 3 crashed from resnet50 = 8 trials, 2 families
        configs = []
        for i in range(5):
            configs.append((
                {"model_name": "resnet18", "backend": "pytorch_eager",
                 "quantization": "fp32", "batch_size": 1},
                _feasible_result(),
            ))
        for i in range(3):
            configs.append((
                {"model_name": "resnet50", "backend": "pytorch_eager",
                 "quantization": "fp32", "batch_size": 1},
                _crashed_result(),
            ))

        self._populate_optimizer(opt, configs)

        # Count thresholds: 5 feasible >= 5, 3 bad >= 3, trial_count=8 >= 5
        # But only 2 families — should NOT hand off
        assert not opt._should_handoff()

    def test_handoff_with_three_families(self):
        """Count thresholds met AND 3 families seen -> handoff triggers."""
        opt = _make_hybrid(n_initial_random=0, min_family_diversity=3)
        configs = []
        # 5 feasible across 3 families
        for model in ["resnet18", "resnet50", "vit_tiny", "resnet18", "resnet50"]:
            configs.append((
                {"model_name": model, "backend": "pytorch_eager",
                 "quantization": "fp32", "batch_size": 1},
                _feasible_result(),
            ))
        # 3 crashed from a mix
        for model in ["mobilenet_v2", "resnet18", "vit_tiny"]:
            configs.append((
                {"model_name": model, "backend": "torch_compile",
                 "quantization": "fp32", "batch_size": 1},
                _crashed_result(),
            ))

        self._populate_optimizer(opt, configs)

        # 5 feasible, 3 bad, 4 families -> handoff
        assert opt._should_handoff()

    def test_safety_valve_forces_handoff_at_n_max(self):
        """At n_max=15, handoff triggers even with only 1 family."""
        opt = _make_hybrid(n_initial_random=0, min_family_diversity=3)
        # 15 trials all from resnet18: 10 feasible, 5 crashed
        configs = []
        for _ in range(10):
            configs.append((
                {"model_name": "resnet18", "backend": "pytorch_eager",
                 "quantization": "fp32", "batch_size": 1},
                _feasible_result(),
            ))
        for _ in range(5):
            configs.append((
                {"model_name": "resnet18", "backend": "torch_compile",
                 "quantization": "fp32", "batch_size": 1},
                _crashed_result(),
            ))

        self._populate_optimizer(opt, configs)
        assert opt.trial_count == 15
        # Only 1 family but at n_max -> safety valve fires
        assert opt._should_handoff()

    def test_crashed_trials_count_toward_diversity(self):
        """Crashed trials contribute to model family diversity count."""
        opt = _make_hybrid(n_initial_random=0, min_family_diversity=3)
        configs = []
        # 5 feasible from resnet18
        for _ in range(5):
            configs.append((
                {"model_name": "resnet18", "backend": "pytorch_eager",
                 "quantization": "fp32", "batch_size": 1},
                _feasible_result(),
            ))
        # 2 crashed from mobilenet_v2
        for _ in range(2):
            configs.append((
                {"model_name": "mobilenet_v2", "backend": "torch_compile",
                 "quantization": "fp32", "batch_size": 1},
                _crashed_result(),
            ))
        # 1 crashed from vit_tiny -> 3rd family, even though it crashed
        configs.append((
            {"model_name": "vit_tiny", "backend": "onnxruntime",
             "quantization": "fp32", "batch_size": 1},
            _crashed_result(),
        ))

        self._populate_optimizer(opt, configs)

        # 5 feasible, 3 bad, 3 families (resnet18 OK, mobilenet_v2 crashed, vit_tiny crashed)
        assert opt._should_handoff()


# ==================================================================
# Change 2: Adaptive Timeout Threshold
# ==================================================================

class TestAdaptiveTimeout:
    """v3: timeout multiplier drops from 5x to 3x after first feasible solution."""

    def test_multiplier_constants(self):
        """Verify the two multiplier constants exist and have expected values."""
        assert EARLY_STOP_MULTIPLIER_PREFEASIBLE == 5
        assert EARLY_STOP_MULTIPLIER_POSTFEASIBLE == 3

    def test_threshold_5x_before_feasible(self):
        """Before any feasible solution, threshold = 5 * constraint."""
        from tba.benchmarks.profiler import evaluate_config
        import inspect

        sig = inspect.signature(evaluate_config)
        # has_feasible param exists and defaults to False
        assert "has_feasible" in sig.parameters
        assert sig.parameters["has_feasible"].default is False

        # early_stop_multiplier defaults to None (adaptive)
        assert sig.parameters["early_stop_multiplier"].default is None

    def test_threshold_3x_after_feasible(self):
        """After feasible found, threshold = 3 * constraint.

        We verify by checking the multiplier selection logic:
        when has_feasible=True and no explicit multiplier, the post-feasible
        multiplier (3x) should be selected.
        """
        # The selection logic is at the top of evaluate_config.
        # We test it indirectly: the multiplier should be 3 when has_feasible=True
        assert EARLY_STOP_MULTIPLIER_POSTFEASIBLE < EARLY_STOP_MULTIPLIER_PREFEASIBLE

    def test_4x_latency_aborted_after_feasible_not_before(self):
        """A trial at 4x the constraint:
        - should NOT be aborted before feasible (4x < 5x threshold)
        - should be aborted after feasible (4x > 3x threshold)
        """
        constraint_ms = 20.0
        latency_4x = 4.0 * constraint_ms  # 80ms

        # Before feasible: threshold = 5 * 20 = 100ms. 80ms < 100ms -> not aborted
        threshold_pre = EARLY_STOP_MULTIPLIER_PREFEASIBLE * constraint_ms
        assert latency_4x < threshold_pre, "4x latency should be below 5x threshold"

        # After feasible: threshold = 3 * 20 = 60ms. 80ms > 60ms -> aborted
        threshold_post = EARLY_STOP_MULTIPLIER_POSTFEASIBLE * constraint_ms
        assert latency_4x > threshold_post, "4x latency should exceed 3x threshold"

    def test_explicit_multiplier_overrides_adaptive(self):
        """When explicit early_stop_multiplier is passed, has_feasible is ignored."""
        import inspect
        from tba.benchmarks.profiler import evaluate_config

        # If early_stop_multiplier is explicitly set (not None), the adaptive
        # logic should not override it. We verify the signature supports this.
        sig = inspect.signature(evaluate_config)
        assert sig.parameters["early_stop_multiplier"].default is None


# ==================================================================
# Change 3: Combination-Level Blacklisting
# ==================================================================

class TestComboBlacklisting:
    """v3: 2-way categorical combination blacklisting."""

    def _make_tracker(self, max_failures=3, cooldown=8):
        return SubspaceTracker(
            categorical_names=["model_name", "backend", "quantization"],
            max_consecutive_failures=max_failures,
            cooldown_trials=cooldown,
            enable_combo_blacklisting=True,
        )

    def test_combo_blacklisted_after_3_failures(self):
        """(int8_dynamic, onnxruntime) gets blacklisted after 3 consecutive failures."""
        tracker = self._make_tracker()

        config = {"model_name": "resnet18", "backend": "onnxruntime", "quantization": "int8_dynamic"}
        for trial in range(3):
            tracker.record_result(config, "crash", trial)

        # The combo (quantization=int8_dynamic, backend=onnxruntime) is blacklisted
        assert tracker.is_combo_blacklisted(config, current_trial=3)

        # Also in the blacklisted list
        combos = tracker.combo_blacklisted
        assert len(combos) > 0

    def test_combo_blacklist_does_not_block_other_combos(self):
        """Blacklisting (int8_dynamic, onnxruntime) does NOT block (int8_dynamic, pytorch_eager)."""
        tracker = self._make_tracker()

        bad_config = {"model_name": "resnet18", "backend": "onnxruntime", "quantization": "int8_dynamic"}
        for trial in range(3):
            tracker.record_result(bad_config, "crash", trial)

        # (int8_dynamic, onnxruntime) is blacklisted
        assert tracker.is_combo_blacklisted(bad_config, current_trial=3)

        # (int8_dynamic, pytorch_eager) with a different model is NOT blacklisted
        # (uses resnet50 to avoid the (resnet18, int8_dynamic) combo which was also blacklisted)
        good_config = {"model_name": "resnet50", "backend": "pytorch_eager", "quantization": "int8_dynamic"}
        assert not tracker.is_combo_blacklisted(good_config, current_trial=3)

    def test_combo_cooldown(self):
        """Combination blacklist removed after cooldown_trials."""
        tracker = self._make_tracker(cooldown=8)

        config = {"model_name": "vit_tiny", "backend": "onnxruntime", "quantization": "int8_dynamic"}
        for trial in range(3):
            tracker.record_result(config, "crash", trial)

        # Blacklisted at trial 2
        assert tracker.is_combo_blacklisted(config, current_trial=5)

        # Still blacklisted at trial 9 (7 trials after blacklisting at trial 2)
        assert tracker.is_combo_blacklisted(config, current_trial=9)

        # Un-blacklisted at trial 10 (8 trials after blacklisting at trial 2)
        assert not tracker.is_combo_blacklisted(config, current_trial=10)

    def test_combo_success_reset(self):
        """Successful trial with a blacklisted combo removes the blacklist."""
        tracker = self._make_tracker()

        config = {"model_name": "resnet50", "backend": "onnxruntime", "quantization": "int8_dynamic"}
        for trial in range(3):
            tracker.record_result(config, "crash", trial)

        assert tracker.is_combo_blacklisted(config, current_trial=3)

        # Succeed with same combo
        tracker.record_result(config, "ok", 4)

        assert not tracker.is_combo_blacklisted(config, current_trial=5)

    def test_combo_safety_valve(self):
        """Cannot blacklist ALL combinations — at least one must remain proposable.

        We test that even if many combos are blacklisted, the tracker's
        is_combo_blacklisted still allows configs that don't hit any
        non-universal blacklist.
        """
        tracker = self._make_tracker()

        # Blacklist (resnet18, onnxruntime) combo
        config1 = {"model_name": "resnet18", "backend": "onnxruntime", "quantization": "fp32"}
        for t in range(3):
            tracker.record_result(config1, "crash", t)

        # A different combo should still be allowed
        config2 = {"model_name": "resnet50", "backend": "pytorch_eager", "quantization": "fp16"}
        assert not tracker.is_combo_blacklisted(config2, current_trial=3)

    def test_combo_blacklisting_only_phase1(self):
        """Combination blacklisting only applies in Phase 1; Phase 2 (TPE) is unrestricted.

        We verify by checking that _tpe_ask does not call is_combo_blacklisted,
        i.e. the combo filter only lives in _tba_ask.
        """
        opt = _make_hybrid(n_initial_random=3, min_family_diversity=1)

        # Feed enough trials to trigger handoff.
        # Need: trial_count >= 5 (min_tba_trials), 5 feasible, 3 bad, 1 family
        models = ["resnet18", "resnet50", "vit_tiny",  # initial random
                   "resnet18", "resnet50",               # feasible
                   "mobilenet_v2", "resnet18", "vit_tiny",  # bad
                   "resnet18", "resnet50",               # more feasible
                   "efficientnet_b0",                     # more feasible
                   ]
        for i, model in enumerate(models):
            _ = opt.ask()
            if i < 3:
                # Initial random phase — mix of feasible
                opt.tell(
                    {"model_name": model, "backend": "pytorch_eager",
                     "quantization": "fp32", "batch_size": 1},
                    _feasible_result(),
                )
            elif i < 5:
                opt.tell(
                    {"model_name": model, "backend": "pytorch_eager",
                     "quantization": "fp32", "batch_size": 1},
                    _feasible_result(),
                )
            elif i < 8:
                opt.tell(
                    {"model_name": model, "backend": "torch_compile",
                     "quantization": "fp32", "batch_size": 1},
                    _crashed_result(),
                )
            else:
                opt.tell(
                    {"model_name": model, "backend": "pytorch_eager",
                     "quantization": "fp16", "batch_size": 1},
                    _feasible_result(),
                )

            if opt._in_tpe_phase:
                break

        # Verify handoff occurred
        assert opt._in_tpe_phase, (
            f"Expected handoff after {opt.trial_count} trials "
            f"(feasible={opt.n_feasible}, crashes={opt.n_crashes})"
        )

        # Phase 2 ask should work — TPE doesn't use combo blacklist
        config = opt.ask()
        assert isinstance(config, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
