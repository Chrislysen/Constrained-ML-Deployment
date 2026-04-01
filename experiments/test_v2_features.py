"""Tests for v2 features: SubspaceTracker blacklisting and early stop flag."""
import pytest

from tba.optimizer.subspace_tracker import SubspaceTracker


# ------------------------------------------------------------------
# SubspaceTracker tests
# ------------------------------------------------------------------

class TestSubspaceTracker:
    """Unit tests for subspace blacklisting logic."""

    def _make_tracker(self, max_failures=3, cooldown=8):
        return SubspaceTracker(
            categorical_names=["model_name", "backend", "quantization"],
            max_consecutive_failures=max_failures,
            cooldown_trials=cooldown,
        )

    def test_blacklists_after_consecutive_failures(self):
        """int8_dynamic should be blacklisted after 3 consecutive infeasible results."""
        tracker = self._make_tracker()
        all_quants = ["fp32", "fp16", "int8_dynamic"]

        config = {"model_name": "resnet18", "backend": "pytorch_eager", "quantization": "int8_dynamic"}

        # 3 consecutive failures
        for trial in range(3):
            tracker.record_result(config, "infeasible", trial)

        allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=3)
        assert "int8_dynamic" not in allowed
        assert "fp32" in allowed
        assert "fp16" in allowed

    def test_not_blacklisted_before_threshold(self):
        """2 failures should NOT trigger blacklisting (threshold is 3)."""
        tracker = self._make_tracker()
        all_quants = ["fp32", "fp16", "int8_dynamic"]

        config = {"model_name": "resnet18", "backend": "pytorch_eager", "quantization": "int8_dynamic"}

        tracker.record_result(config, "infeasible", 0)
        tracker.record_result(config, "infeasible", 1)

        allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=2)
        assert "int8_dynamic" in allowed

    def test_success_resets_counter(self):
        """A success should reset the consecutive failure counter."""
        tracker = self._make_tracker()
        all_quants = ["fp32", "fp16", "int8_dynamic"]

        config = {"model_name": "resnet18", "backend": "pytorch_eager", "quantization": "int8_dynamic"}

        # 2 failures, then 1 success, then 2 more failures
        tracker.record_result(config, "infeasible", 0)
        tracker.record_result(config, "infeasible", 1)
        tracker.record_result(config, "ok", 2)
        tracker.record_result(config, "infeasible", 3)
        tracker.record_result(config, "infeasible", 4)

        allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=5)
        assert "int8_dynamic" in allowed  # only 2 consecutive, not 3

    def test_cooldown_unblacklists(self):
        """After cooldown_trials, the value should be un-blacklisted."""
        tracker = self._make_tracker(cooldown=8)
        all_quants = ["fp32", "fp16", "int8_dynamic"]

        config = {"model_name": "resnet18", "backend": "pytorch_eager", "quantization": "int8_dynamic"}

        # Blacklist at trial 2 (after 3 failures at trials 0,1,2)
        for trial in range(3):
            tracker.record_result(config, "crash", trial)

        # Still blacklisted at trial 9 (only 7 trials after blacklisting at trial 2)
        allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=9)
        assert "int8_dynamic" not in allowed

        # Un-blacklisted at trial 10 (8 trials after blacklisting at trial 2)
        allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=10)
        assert "int8_dynamic" in allowed

    def test_safety_valve_never_blacklist_all(self):
        """If ALL values would be blacklisted, allow all of them instead."""
        tracker = self._make_tracker()
        all_quants = ["fp32", "fp16", "int8_dynamic"]

        # Blacklist every quantization value
        for quant in all_quants:
            config = {"model_name": "resnet18", "backend": "pytorch_eager", "quantization": quant}
            for trial in range(3):
                tracker.record_result(config, "crash", trial)

        allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=5)
        assert set(allowed) == set(all_quants), "Safety valve should allow all values"

    def test_crash_and_infeasible_both_count(self):
        """Both crash and infeasible statuses should increment the failure counter."""
        tracker = self._make_tracker()
        all_quants = ["fp32", "fp16", "int8_dynamic"]

        config = {"model_name": "resnet18", "backend": "pytorch_eager", "quantization": "int8_dynamic"}

        tracker.record_result(config, "crash", 0)
        tracker.record_result(config, "infeasible", 1)
        tracker.record_result(config, "crash", 2)

        allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=3)
        assert "int8_dynamic" not in allowed

    def test_tracks_multiple_variables(self):
        """Blacklisting should work independently per variable."""
        tracker = self._make_tracker()

        # Blacklist a backend value
        config = {"model_name": "resnet18", "backend": "torch_compile", "quantization": "fp32"}
        for trial in range(3):
            tracker.record_result(config, "crash", trial)

        # backend=torch_compile should be blacklisted
        all_backends = ["pytorch_eager", "torch_compile", "onnxruntime"]
        allowed_backends = tracker.get_allowed_values("backend", all_backends, current_trial=3)
        assert "torch_compile" not in allowed_backends

        # quantization=fp32 should NOT be blacklisted (it succeeded elsewhere conceptually,
        # but here all 3 trials had it — so it actually gets 3 consecutive failures too)
        # However, model_name=resnet18 also got 3 failures
        # This is expected: the tracker is per-(var, value) pair independently
        all_quants = ["fp32", "fp16", "int8_dynamic"]
        allowed_quants = tracker.get_allowed_values("quantization", all_quants, current_trial=3)
        # fp32 was in all 3 crash configs, so it IS blacklisted
        assert "fp32" not in allowed_quants

    def test_success_unblacklists_immediately(self):
        """A successful trial should immediately un-blacklist the value."""
        tracker = self._make_tracker()
        all_quants = ["fp32", "fp16", "int8_dynamic"]

        config = {"model_name": "resnet18", "backend": "pytorch_eager", "quantization": "int8_dynamic"}

        # Blacklist it
        for trial in range(3):
            tracker.record_result(config, "infeasible", trial)

        allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=3)
        assert "int8_dynamic" not in allowed

        # Now succeed
        tracker.record_result(config, "ok", 4)

        allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=5)
        assert "int8_dynamic" in allowed

    def test_blacklisted_property(self):
        """The blacklisted property should return currently blacklisted pairs."""
        tracker = self._make_tracker()

        config = {"model_name": "resnet18", "backend": "pytorch_eager", "quantization": "int8_dynamic"}
        for trial in range(3):
            tracker.record_result(config, "crash", trial)

        bl = tracker.blacklisted
        assert ("quantization", "int8_dynamic") in bl

    def test_simulated_t4_scenario(self):
        """Simulate the T4 failure mode: int8_dynamic always infeasible.

        Without blacklisting, SA would keep hitting int8_dynamic.
        With blacklisting, it should be removed from the pool after 3 attempts.
        """
        tracker = self._make_tracker()
        all_quants = ["fp32", "fp16", "int8_dynamic"]

        # Simulate 10 trials where int8_dynamic always fails
        for trial in range(10):
            config = {
                "model_name": "resnet18",
                "backend": "pytorch_eager",
                "quantization": "int8_dynamic",
            }
            tracker.record_result(config, "infeasible", trial)

            allowed = tracker.get_allowed_values("quantization", all_quants, current_trial=trial + 1)

            if trial < 2:
                # First 3 failures (0,1,2) — not yet blacklisted
                assert "int8_dynamic" in allowed
            elif trial >= 2:
                # After 3rd failure (trial 2), should be blacklisted
                assert "int8_dynamic" not in allowed, (
                    f"int8_dynamic should be blacklisted after trial {trial}"
                )


# ------------------------------------------------------------------
# EvalResult early_stopped field test
# ------------------------------------------------------------------

class TestEvalResultEarlyStopped:
    def test_default_false(self):
        from tba.types import EvalResult
        r = EvalResult()
        assert r.early_stopped is False

    def test_can_set_true(self):
        from tba.types import EvalResult
        r = EvalResult(early_stopped=True)
        assert r.early_stopped is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
