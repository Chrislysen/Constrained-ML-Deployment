"""Tests for the hierarchical mixed-variable search space."""
import random

import pytest

from tba.types import VariableDef
from tba.optimizer.search_space import SearchSpace


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_simple_space() -> SearchSpace:
    """Simple space: 1 categorical, 1 integer, 1 continuous."""
    return SearchSpace([
        VariableDef(name="color", var_type="categorical", choices=["red", "green", "blue"]),
        VariableDef(name="count", var_type="integer", low=1, high=10),
        VariableDef(name="weight", var_type="continuous", low=0.0, high=1.0),
    ])


def make_hierarchical_space() -> SearchSpace:
    """Space with conditional variables mimicking the deployment space."""
    return SearchSpace([
        VariableDef(
            name="backend",
            var_type="categorical",
            choices=["pytorch_eager", "torch_compile", "onnxruntime"],
        ),
        VariableDef(name="batch_size", var_type="integer", low=1, high=64, log_scale=True),
        VariableDef(
            name="num_threads",
            var_type="integer",
            low=1,
            high=16,
            condition="backend in ['pytorch_eager', 'onnxruntime']",
        ),
        VariableDef(
            name="compile_mode",
            var_type="categorical",
            choices=["default", "reduce-overhead", "max-autotune"],
            condition="backend == 'torch_compile'",
        ),
        VariableDef(
            name="quantization",
            var_type="categorical",
            choices=["fp32", "fp16", "int8_dynamic", "int8_static"],
        ),
        VariableDef(
            name="calibration_samples",
            var_type="integer",
            low=50,
            high=500,
            condition="quantization == 'int8_static'",
        ),
    ])


# ------------------------------------------------------------------
# Tests: Sampling
# ------------------------------------------------------------------

class TestSampling:
    def test_simple_sample_all_keys_present(self):
        space = make_simple_space()
        rng = random.Random(42)
        config = space.sample_random(rng)
        assert set(config.keys()) == {"color", "count", "weight"}

    def test_simple_sample_values_in_range(self):
        space = make_simple_space()
        rng = random.Random(42)
        for _ in range(100):
            config = space.sample_random(rng)
            assert config["color"] in ["red", "green", "blue"]
            assert 1 <= config["count"] <= 10
            assert isinstance(config["count"], int)
            assert 0.0 <= config["weight"] <= 1.0

    def test_hierarchical_sample_respects_conditions(self):
        space = make_hierarchical_space()
        rng = random.Random(42)
        for _ in range(200):
            config = space.sample_random(rng)
            backend = config["backend"]
            quant = config["quantization"]

            # compile_mode only when torch_compile
            if backend == "torch_compile":
                assert "compile_mode" in config
                assert "num_threads" not in config
            else:
                assert "compile_mode" not in config
                assert "num_threads" in config

            # calibration_samples only when int8_static
            if quant == "int8_static":
                assert "calibration_samples" in config
                assert 50 <= config["calibration_samples"] <= 500
            else:
                assert "calibration_samples" not in config

    def test_log_scale_integer_stays_in_bounds(self):
        space = make_hierarchical_space()
        rng = random.Random(42)
        for _ in range(200):
            config = space.sample_random(rng)
            assert 1 <= config["batch_size"] <= 64

    def test_all_samples_are_valid(self):
        space = make_hierarchical_space()
        rng = random.Random(42)
        for _ in range(200):
            config = space.sample_random(rng)
            assert space.is_valid(config), f"Invalid config: {config}"


# ------------------------------------------------------------------
# Tests: Neighbor Proposals
# ------------------------------------------------------------------

class TestNeighborProposals:
    def test_neighbor_is_valid(self):
        space = make_hierarchical_space()
        rng = random.Random(42)
        config = space.sample_random(rng)
        for _ in range(100):
            neighbor = space.propose_neighbor(config, temperature=0.5, rng=rng)
            assert space.is_valid(neighbor), f"Invalid neighbor: {neighbor}"

    def test_neighbor_differs_from_original(self):
        space = make_simple_space()
        rng = random.Random(42)
        config = space.sample_random(rng)
        changed = False
        for _ in range(50):
            neighbor = space.propose_neighbor(config, temperature=0.5, rng=rng)
            if neighbor != config:
                changed = True
                break
        assert changed, "Neighbor never differs from original"

    def test_high_temperature_larger_steps(self):
        space = make_simple_space()
        rng_lo = random.Random(42)
        rng_hi = random.Random(42)
        config = {"color": "red", "count": 5, "weight": 0.5}

        dists_lo, dists_hi = [], []
        for _ in range(200):
            n_lo = space.propose_neighbor(config, temperature=0.01, rng=rng_lo)
            n_hi = space.propose_neighbor(config, temperature=1.0, rng=rng_hi)
            dists_lo.append(space.config_distance(config, n_lo))
            dists_hi.append(space.config_distance(config, n_hi))

        # High-temp neighbors should have larger average distance
        assert sum(dists_hi) / len(dists_hi) > sum(dists_lo) / len(dists_lo)

    def test_structural_change_resolves_hierarchy(self):
        space = make_hierarchical_space()
        rng = random.Random(42)
        # Start with pytorch_eager — has num_threads, no compile_mode
        config = space.sample_random(rng)
        config["backend"] = "pytorch_eager"
        config.pop("compile_mode", None)
        if "num_threads" not in config:
            config["num_threads"] = 4
        space._resolve_hierarchy(config, rng)

        # Force a structural move to torch_compile
        neighbor = deepcopy_config(config)
        neighbor["backend"] = "torch_compile"
        space._resolve_hierarchy(neighbor, rng)

        assert "compile_mode" in neighbor
        assert "num_threads" not in neighbor


# ------------------------------------------------------------------
# Tests: Distance
# ------------------------------------------------------------------

class TestDistance:
    def test_identical_configs_zero_distance(self):
        space = make_simple_space()
        config = {"color": "red", "count": 5, "weight": 0.5}
        assert space.config_distance(config, config) == 0.0

    def test_max_distance_opposite_corners(self):
        space = make_simple_space()
        a = {"color": "red", "count": 1, "weight": 0.0}
        b = {"color": "blue", "count": 10, "weight": 1.0}
        d = space.config_distance(a, b)
        assert d == pytest.approx(1.0)

    def test_distance_symmetry(self):
        space = make_hierarchical_space()
        rng = random.Random(42)
        for _ in range(50):
            a = space.sample_random(rng)
            b = space.sample_random(rng)
            assert space.config_distance(a, b) == pytest.approx(
                space.config_distance(b, a)
            )

    def test_distance_handles_missing_conditional_vars(self):
        space = make_hierarchical_space()
        a = {"backend": "torch_compile", "batch_size": 8, "compile_mode": "default",
             "quantization": "fp32"}
        b = {"backend": "pytorch_eager", "batch_size": 8, "num_threads": 4,
             "quantization": "fp32"}
        d = space.config_distance(a, b)
        assert 0.0 < d <= 1.0


# ------------------------------------------------------------------
# Tests: Validation
# ------------------------------------------------------------------

class TestValidation:
    def test_valid_simple_config(self):
        space = make_simple_space()
        assert space.is_valid({"color": "red", "count": 5, "weight": 0.5})

    def test_invalid_categorical_choice(self):
        space = make_simple_space()
        assert not space.is_valid({"color": "purple", "count": 5, "weight": 0.5})

    def test_invalid_integer_out_of_range(self):
        space = make_simple_space()
        assert not space.is_valid({"color": "red", "count": 99, "weight": 0.5})

    def test_invalid_missing_active_variable(self):
        space = make_simple_space()
        assert not space.is_valid({"color": "red", "count": 5})


# ------------------------------------------------------------------
# Tests: VariableDef validation
# ------------------------------------------------------------------

class TestVariableDef:
    def test_categorical_needs_choices(self):
        with pytest.raises(ValueError, match="needs choices"):
            VariableDef(name="x", var_type="categorical")

    def test_numeric_needs_bounds(self):
        with pytest.raises(ValueError, match="needs low and high"):
            VariableDef(name="x", var_type="integer", low=1)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def deepcopy_config(config):
    from copy import deepcopy
    return deepcopy(config)
