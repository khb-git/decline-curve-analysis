"""
Tests for the fitting module.

The most reliable validation: generate synthetic data from known
parameters, fit it, and verify the fitter recovers the inputs.
"""

import numpy as np
import pytest

from src.models.arps import DeclineParameters, rate as arps_rate
from src.fitting.nonlinear_regression import fit_arps, fit_all_arps


@pytest.fixture
def exponential_data():
    """Clean exponential decline data."""
    true_params = DeclineParameters(qi=1000, Di=0.6, b=0)
    t = np.arange(0, 365.25 * 5, 30.4)
    q = arps_rate(t, true_params)
    return t, q, true_params


@pytest.fixture
def hyperbolic_data():
    """Clean hyperbolic decline data."""
    true_params = DeclineParameters(qi=800, Di=0.8, b=0.7)
    t = np.arange(0, 365.25 * 5, 30.4)
    q = arps_rate(t, true_params)
    return t, q, true_params


@pytest.fixture
def noisy_hyperbolic_data():
    """Hyperbolic decline with 10% multiplicative noise."""
    rng = np.random.default_rng(42)
    true_params = DeclineParameters(qi=800, Di=0.8, b=0.7)
    t = np.arange(0, 365.25 * 5, 30.4)
    q_clean = arps_rate(t, true_params)
    q = q_clean * rng.normal(1.0, 0.10, size=len(t))
    return t, q, true_params


class TestExponentialFitting:
    def test_recovers_clean_exponential(self, exponential_data):
        t, q, true = exponential_data
        result = fit_arps(t, q, model_type="exponential")
        assert np.isclose(result.params.qi, true.qi, rtol=0.01)
        assert np.isclose(result.params.Di, true.Di, rtol=0.01)
        assert result.r_squared > 0.999


class TestHyperbolicFitting:
    def test_recovers_clean_hyperbolic(self, hyperbolic_data):
        t, q, true = hyperbolic_data
        result = fit_arps(t, q, model_type="hyperbolic")
        assert np.isclose(result.params.qi, true.qi, rtol=0.02)
        assert np.isclose(result.params.Di, true.Di, rtol=0.05)
        assert np.isclose(result.params.b, true.b, rtol=0.05)
        assert result.r_squared > 0.999

    def test_recovers_noisy_hyperbolic(self, noisy_hyperbolic_data):
        t, q, true = noisy_hyperbolic_data
        result = fit_arps(t, q, model_type="hyperbolic")
        # Looser tolerances for noisy data
        assert np.isclose(result.params.qi, true.qi, rtol=0.10)
        assert np.isclose(result.params.Di, true.Di, rtol=0.20)
        assert np.isclose(result.params.b, true.b, rtol=0.30)
        assert result.r_squared > 0.85


class TestFitDiagnostics:
    def test_covariance_matrix_returned(self, hyperbolic_data):
        t, q, _ = hyperbolic_data
        result = fit_arps(t, q, model_type="hyperbolic")
        assert result.covariance.shape == (3, 3)
        # Diagonal must be non-negative (variances)
        assert np.all(np.diag(result.covariance) >= 0)

    def test_n_observations_correct(self, hyperbolic_data):
        t, q, _ = hyperbolic_data
        result = fit_arps(t, q, model_type="hyperbolic")
        assert result.n_observations == len(t)


class TestFitAllArps:
    def test_returns_all_three_models(self, hyperbolic_data):
        t, q, _ = hyperbolic_data
        results = fit_all_arps(t, q)
        assert set(results.keys()) == {"exponential", "hyperbolic", "harmonic"}

    def test_hyperbolic_fits_hyperbolic_data_best(self, hyperbolic_data):
        t, q, _ = hyperbolic_data
        results = fit_all_arps(t, q)
        # Hyperbolic should have lowest RMSE on hyperbolic data
        rmse_by_model = {k: v.rmse for k, v in results.items() if hasattr(v, 'rmse')}
        assert rmse_by_model["hyperbolic"] <= rmse_by_model["exponential"]


class TestInputValidation:
    def test_mismatched_lengths_raises(self):
        t = np.array([0, 30, 60])
        q = np.array([1000, 900])
        with pytest.raises(ValueError):
            fit_arps(t, q)

    def test_too_few_points_raises(self):
        t = np.array([0, 30])
        q = np.array([1000, 900])
        with pytest.raises(ValueError):
            fit_arps(t, q)
